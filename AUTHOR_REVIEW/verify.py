"""Automated verification for the author review.

Run:  python AUTHOR_REVIEW/verify.py

Every check below is an INDEPENDENT proof you regenerate yourself — nothing here relies on
trusting the assistant. Each prints PASS / FAIL with what it demonstrated.

  1. numba fast pipeline  ==  frozen baseline   (bit-identical, same seed)  -> 🟢 same algorithm
  2. rewritten graph builder  ==  original set/list builder  (identical graph) -> 🟢 same graph
  3. spectre tiling: deterministic + no isolated nodes + hat-like coordination -> 🟣 sane tiling
  4. periodic chevron & comet: exact triangle coverage (no gap, no overlap)     -> 🟣 valid tiling
"""
import sys, os, math, random, itertools
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import numpy as np

results = []
def check(name, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n        {detail}")


# ---------- helpers ----------
def grid(S):
    N = S*S; neigh = []
    for i in range(S):
        for j in range(S):
            a = []
            if i > 0: a.append((i-1)*S+j)
            if i < S-1: a.append((i+1)*S+j)
            if j > 0: a.append(i*S+(j-1))
            if j < S-1: a.append(i*S+(j+1))
            neigh.append(np.array(a, dtype=np.int32))
    nodes = np.arange(N)
    top = np.array(list(range(S)), np.int32); bot = np.array([(S-1)*S+j for j in range(S)], np.int32)
    left = np.array([i*S for i in range(S)], np.int32); right = np.array([i*S+(S-1) for i in range(S)], np.int32)
    return nodes, neigh, top, bot, left, right


# ---------- 1. numba fast == baseline (bit-identical) ----------
def check_numba():
    import percolation_baseline as base
    import percolation_fast as fast
    nodes, neigh, top, bot, left, right = grid(30)
    random.seed(12345); rb = base.percolationStatsI(nodes, neigh, top, bot, left, right, 8)
    random.seed(12345); rf = fast.percolationStatsI(nodes, neigh, top, bot, left, right, 8)  # first call compiles
    same = np.array_equal(np.array(rb.trialResults), np.array(rf.trialResults))
    # bond too
    def gedges(S):
        e = []
        for i in range(S):
            for j in range(S):
                n = i*S+j
                if j < S-1: e.append((n, n+1))
                if i < S-1: e.append((n, n+S))
        return np.array(e, dtype=np.int64)
    edges = gedges(30)
    random.seed(7); bb = base.percolationStatsBondI(nodes, edges, top, bot, left, right, 8)
    random.seed(7); bf = fast.percolationStatsBondI(nodes, edges, top, bot, left, right, 8)
    same_b = np.array_equal(np.array(bb.trialResults), np.array(bf.trialResults))
    check("1. numba serial == baseline (site & bond, same seed)", same and same_b,
          f"site bit-identical={same}, bond bit-identical={same_b}  -> fast path is the SAME algorithm, just compiled")


# ---------- 2. rewritten builder == original set/list builder ----------
def check_builder():
    from scipy.spatial import KDTree
    from hat_generator import (H_init, T_init, P_init, F_init, constructPatch,
                               constructMetatiles, mul, transPt)
    from hat_graph_builder import build_neighbor_graph_fast
    R = 4
    cur = [H_init(), T_init(), P_init(), F_init()]
    for _ in range(R):
        p = constructPatch(*cur); cur = constructMetatiles(p)

    def ref_build(patch, level, tol=1e-5):     # ORIGINAL set/list logic, inline
        coords = []
        def _c(pt, S, lv):
            if lv > 0 and hasattr(pt, 'children'):
                for g in pt.children: _c(g['geom'], mul(S, g['T']), lv-1)
            else:
                for pp in pt.shape:
                    q = transPt(S, pp); coords.append((q['x'], q['y']))
        _c(patch, [1, 0, 0, 0, 1, 0], level)
        nodes = np.array(coords)
        tree = KDTree(nodes); pairs = tree.query_pairs(r=tol)
        parent = list(range(len(nodes)))
        def find(x):
            while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for i, j in pairs:
            ri, rj = find(i), find(j)
            if ri != rj: parent[rj] = ri
        mapping = np.array([find(i) for i in range(len(nodes))])
        uids = np.unique(mapping); id2 = {u: k for k, u in enumerate(uids)}
        n2u = np.array([id2[mapping[i]] for i in range(len(nodes))]); un = nodes[uids]
        edges = set(); idx = 0
        def _e(pt, S, lv):
            nonlocal idx
            if lv > 0 and hasattr(pt, 'children'):
                for g in pt.children: _e(g['geom'], mul(S, g['T']), lv-1)
            else:
                m = len(pt.shape); base = idx
                for i in range(m):
                    aa = int(n2u[base+i]); bb = int(n2u[base+(i+1) % m])
                    if aa != bb: edges.add((min(aa, bb), max(aa, bb)))
                idx += m
        _e(patch, [1, 0, 0, 0, 1, 0], level)
        return un, edges

    def keyc(xy): return (round(float(xy[0]), 3), round(float(xy[1]), 3))
    n_new, nb_new = build_neighbor_graph_fast(p, level=R+1)
    n_ref, e_ref = ref_build(p, R+1)
    nodes_ok = {keyc(x) for x in n_new} == {keyc(x) for x in n_ref}
    e_new = {frozenset((keyc(n_new[i]), keyc(n_new[int(j)]))) for i, arr in enumerate(nb_new) for j in arr if i < j}
    e_refc = {frozenset((keyc(n_ref[a]), keyc(n_ref[b]))) for a, b in e_ref}
    edges_ok = e_new == e_refc
    check("2. rewritten builder == original builder (r=4, geometric graph)", nodes_ok and edges_ok,
          f"node-set identical={nodes_ok}, edge-set identical={edges_ok}  ({len(n_new):,} nodes) -> memory rewrite preserves the graph exactly")


# ---------- 3. spectre tiling sanity ----------
def check_spectre():
    from spectre_generator import build_spectre_patch
    from hat_graph_builder import build_neighbor_graph_fast
    def count(node):
        ch = getattr(node, "children", None)
        return sum(count(c['geom']) for c in ch) if ch else 1
    patch = build_spectre_patch(3)
    ntiles = count(patch)
    nodes, neigh = build_neighbor_graph_fast(patch, level=None)
    deg = [len(n) for n in neigh]
    ok = (ntiles == 559 and sum(1 for d in deg if d == 0) == 0
          and min(deg) == 2 and max(deg) == 4 and 2.2 < sum(deg)/len(deg) < 2.4)
    check("3. spectre level-3 tiling (deterministic count, no isolated nodes, hat-like coordination)", ok,
          f"tiles={ntiles} (=559), nodes={len(nodes):,}, avg deg={sum(deg)/len(deg):.2f} (hat=2.31), "
          f"min={min(deg)} max={max(deg)}, isolated={sum(1 for d in deg if d==0)}")


# ---------- 4. periodic chevron & comet: exact coverage ----------
def check_periodic():
    import hat_generator as hg
    SQ3 = math.sqrt(3); E1 = np.array([1.0, 0.0]); E2 = np.array([0.5, SQ3/2])
    def tile_ab(outline, a, b):
        pts = [(p['x'], p['y']) for p in outline]; n = len(pts); out = [(0.0, 0.0)]
        for i in range(n-1):
            x0, y0 = pts[i]; x1, y1 = pts[i+1]; vx, vy = x1-x0, y1-y0; L = math.hypot(vx, vy)
            s = a if abs(L-round(L)) < 1e-6 else (b/SQ3 if abs(L/SQ3-round(L/SQ3)) < 1e-6 else 1.0)
            out.append((out[-1][0]+s*vx, out[-1][1]+s*vy))
        return np.array(out)
    def align(poly):
        v = poly[1]-poly[0]; ang = math.degrees(math.atan2(v[1], v[0])) % 60.0
        if abs(ang-30.0) < 1e-3:
            c, s = math.cos(math.radians(-30)), math.sin(math.radians(-30))
            return poly @ np.array([[c, -s], [s, c]]).T
        return poly
    def tri_cent(i, j, s):
        vs = [(i, j), (i+1, j), (i, j+1)] if s == 0 else [(i+1, j), (i+1, j+1), (i, j+1)]
        c = sum(v[0]*E1+v[1]*E2 for v in vs)/3.0; return c[0], c[1]
    def pip(x, y, P):
        n = len(P); ins = False; px, py = P[-1]
        for qx, qy in P:
            if ((qy > y) != (py > y)) and (x < (px-qx)*(y-qy)/(py-qy+1e-30)+qx): ins = not ins
            px, py = qx, qy
        return ins
    def tri_set(poly):
        P = [tuple(p) for p in poly]; xs = poly[:, 0]; ys = poly[:, 1]; out = set()
        for i in range(int(xs.min())-2, int(xs.max())+2):
            for j in range(int(ys.min()/(SQ3/2))-2, int(ys.max()/(SQ3/2))+2):
                for s in (0, 1):
                    cx, cy = tri_cent(i, j, s)
                    if pip(cx, cy, P): out.add((i, j, s))
        return out
    def verify(ts, u1, u2):
        cover = {}
        for m in range(-3, 4):
            for n in range(-3, 4):
                si = m*u1[0]+n*u2[0]; sj = m*u1[1]+n*u2[1]
                for (i, j, s) in ts: cover[(i+si, j+sj, s)] = cover.get((i+si, j+sj, s), 0)+1
        return all(cover.get((i, j, s), 0) == 1 for i in range(-2, 3) for j in range(-2, 3) for s in (0, 1))
    def find_lat(ts):
        need = len(ts)//2
        for u1 in itertools.product(range(-4, 5), repeat=2):
            for u2 in itertools.product(range(-4, 5), repeat=2):
                det = u1[0]*u2[1]-u1[1]*u2[0]
                if abs(det) != need: continue
                if verify(ts, u1, u2): return u1, u2
        return None
    oks = {}
    for nm, a, b, ntri in [("chevron", 0.0, 1.0, 4), ("comet", 1.0, 0.0, 8)]:
        ts = tri_set(align(tile_ab(hg.hat_outline, a, b)))
        lat = find_lat(ts) if len(ts) == ntri else None
        oks[nm] = (len(ts) == ntri and lat is not None, len(ts), lat)
    ok = oks["chevron"][0] and oks["comet"][0]
    check("4. periodic chevron (tetriamond) & comet (octiamond): exact gap-free/overlap-free coverage", ok,
          f"chevron: {oks['chevron'][1]} tris lattice={oks['chevron'][2]} | comet: {oks['comet'][1]} tris lattice={oks['comet'][2]}")


if __name__ == "__main__":
    print("=" * 78 + "\nAUTHOR REVIEW — automated proofs\n" + "=" * 78)
    for fn in (check_numba, check_builder, check_spectre, check_periodic):
        try:
            fn()
        except Exception as e:
            check(fn.__name__, False, f"ERROR: {e}")
    print("=" * 78)
    print(f"RESULT: {sum(results)}/{len(results)} checks passed"
          + ("  — all mechanical claims confirmed." if all(results) else "  — SOMETHING FAILED, investigate."))
    sys.exit(0 if all(results) else 1)
