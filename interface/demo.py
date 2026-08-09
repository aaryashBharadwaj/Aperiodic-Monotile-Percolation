"""The "How it works" walk-through backend: a deterministic percolation demo on a small SQUARE
lattice OR a small HAT block, opening sites/bonds one at a time in a fixed order until a cluster
spans. Split out of gui_backend; the demo is DRAWN client-side (HTML/SVG+JS) in gui_app, so this only
supplies the graph, the fixed open orders, and the precomputed first-crossing steps.

    build_demo(lattice, n, seed)  -> the demo graph bundle the JS component draws

Shared primitives (_hat_patch) are imported from interface.gui_backend; the dependency is one-way.
"""
import numpy as np

from builders.direct_graph_builder import build_neighbor_graph_fast, largest_square_center
from interface.gui_backend import _hat_patch


# ----------------------------------------------------------------------------- percolation demo
# A deterministic percolation walk-through for the "How it works" tab, on a small SQUARE lattice OR a
# small HAT block (the reader picks). Open SITES (nodes) or BONDS (edges) one at a time in a fixed
# order and watch clusters grow -- EVERY cluster its own colour -- until one spans top<->bottom or
# left<->right. Site vs bond is the same choice the real study offers; the two are drawn differently
# (site colours the dots, bond colours the connections) so they don't look alike.
def _square_demo(n):
    idx = lambda i, j: j * n + i
    coords = [[float(i), float(j)] for j in range(n) for i in range(n)]
    edges = []
    for j in range(n):
        for i in range(n):
            if i + 1 < n: edges.append([idx(i, j), idx(i + 1, j)])
            if j + 1 < n: edges.append([idx(i, j), idx(i, j + 1)])
    return (coords, edges, n * n,
            set(idx(i, n - 1) for i in range(n)), set(idx(i, 0) for i in range(n)),
            set(idx(0, j) for j in range(n)), set(idx(n - 1, j) for j in range(n)),
            [-0.6, n - 0.4, -0.6, n - 0.4])


def _hat_demo(level=3, window_frac=0.45):
    """Square window of the hat VERTEX graph (corners = nodes, tile edges = bonds) -- consistent with
    the square demo, so site/bond mean the same thing, but on the actual hat tiling."""
    patch_obj, _lvl = _hat_patch(level)
    nodes, neigh = build_neighbor_graph_fast(patch_obj, level=level + 1)
    cx, cy, side = largest_square_center(nodes)
    half = 0.5 * window_frac * side
    inw = lambda p: abs(p[0] - cx) <= half and abs(p[1] - cy) <= half
    inside = [i for i in range(len(nodes)) if inw(nodes[i])]
    idx = {o: m for m, o in enumerate(inside)}
    coords = np.array([nodes[o] for o in inside])
    edges = sorted({(min(idx[o], idx[j]), max(idx[o], idx[j]))
                    for o in inside for j in neigh[o] if j in idx})
    xs, ys = coords[:, 0], coords[:, 1]; band = 0.14 * (2 * half)
    B = lambda mask: set(int(m) for m in np.where(mask)[0])
    return (coords.tolist(), [list(e) for e in edges], len(inside),
            B(ys >= ys.max() - band), B(ys <= ys.min() + band), B(xs <= xs.min() + band), B(xs >= xs.max() - band),
            [float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())])


def build_demo(lattice="square", n=12, seed=0):
    """Small SQUARE or HAT lattice for the walk-through: coords, edges, adjacency, which nodes touch
    each side, fixed random site/bond open orders, a bbox, and the precomputed first-crossing step per
    direction (so the UI can say 'top-bottom at ...' / 'left-right at ...' without a per-frame scan)."""
    if lattice == "hat":
        coords, edges, N, top, bottom, left, right, bbox = _hat_demo()
    else:
        coords, edges, N, top, bottom, left, right, bbox = _square_demo(n)
    M = len(edges)
    neigh = [[] for _ in range(N)]
    for a, b in edges:
        neigh[a].append(b); neigh[b].append(a)
    rng = np.random.default_rng(seed)
    demo = {"lattice": lattice, "coords": coords, "edges": edges, "neigh": neigh, "N": N, "M": M,
            "top": top, "bottom": bottom, "left": left, "right": right, "bbox": bbox,
            "site_order": [int(x) for x in rng.permutation(N)],
            "bond_order": [int(x) for x in rng.permutation(M)]}
    demo["cross_site"] = _crossings(demo, "site")
    demo["cross_bond"] = _crossings(demo, "bond")
    return demo


def _crossings(demo, mode):
    """First step at which each direction first spans (or None). Incremental union-find with a per-
    cluster boundary BITMASK (T|B|L|R), opening one site/bond per step -- O(steps * alpha), so it's
    fast even for a 40x40 block. The JS component re-derives clusters itself for the live drawing;
    this only supplies the crossing steps (jump target + the bracket caption)."""
    N, edges, neigh = demo["N"], demo["edges"], demo["neigh"]
    order = demo["site_order"] if mode == "site" else demo["bond_order"]
    T, B, L, R = 1, 2, 4, 8
    fm = [0] * N
    for v in demo["top"]:    fm[v] |= T
    for v in demo["bottom"]: fm[v] |= B
    for v in demo["left"]:   fm[v] |= L
    for v in demo["right"]:  fm[v] |= R
    parent = list(range(N))
    def find(x):
        r = x
        while parent[r] != r: r = parent[r]
        while parent[x] != r: parent[x], x = r, parent[x]
        return r
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb; fm[rb] |= fm[ra]
    cross = {"top-bottom": None, "left-right": None}
    def check(r, step):
        m = fm[r]
        if cross["top-bottom"] is None and (m & (T | B)) == (T | B): cross["top-bottom"] = step
        if cross["left-right"] is None and (m & (L | R)) == (L | R): cross["left-right"] = step
    if mode == "site":
        isopen = bytearray(N)
        for step, v in enumerate(order, 1):
            isopen[v] = 1
            for nb in neigh[v]:
                if isopen[nb]:
                    union(v, nb)
            check(find(v), step)
            if cross["top-bottom"] and cross["left-right"]: break
    else:
        for step, ei in enumerate(order, 1):
            a, b = edges[ei]
            union(a, b)
            check(find(a), step)
            if cross["top-bottom"] and cross["left-right"]: break
    return cross


# The demo is DRAWN client-side in gui_app.demo_component (HTML/SVG+JS) so its slider updates smoothly
# on drag; Python only supplies the graph + fixed open orders (build_demo) and the crossings.


def _smax_by_k(N, neigh, order):
    """Largest-cluster size after opening the first k sites of `order`, for every k in 0..N. Incremental
    union-find, one pass -- returns a list of length N+1 (index k). Used to read the share s_max/N at any
    occupation p (k = round(p N)) without re-running the sweep."""
    parent = list(range(N)); size = [1] * N; isopen = bytearray(N)
    def find(x):
        r = x
        while parent[r] != r: r = parent[r]
        while parent[x] != r: parent[x], x = r, parent[x]
        return r
    curve = [0]; s_max = 0
    for v in order:
        isopen[v] = 1; rv = find(v)
        for nb in neigh[v]:
            if isopen[nb]:
                ra, rb = rv, find(nb)
                if ra != rb:
                    if size[ra] < size[rb]: ra, rb = rb, ra
                    parent[rb] = ra; size[ra] += size[rb]; rv = ra
        if size[rv] > s_max: s_max = size[rv]
        curve.append(s_max)
    return curve


def build_scaling_demo(sizes=(8, 10, 12, 15, 18, 21, 25, 29, 33, 38), fillings=10, seed=1,
                       p_lo=0.30, p_hi=0.72, p_steps=43):
    """Data for the interactive SCALING toy: a LADDER of SQUARE site-grids of increasing size. For EACH
    size we run `fillings` independent random fillings and, over a grid of occupations p, record the
    share s_max/N the largest cluster fills. The toy plots EVERY filling as a point (an honest scatter --
    at criticality s_max/N has ~25% run-to-run spread that never shrinks with size, so a single filling
    can't be trusted) and the per-size MEAN as the bold point. Across the ladder the means fall on a
    straight log-log line, share ~ L^{d_f-2}: a solid 2D region would give slope 0, a 1D line slope -1;
    the incipient cluster sits between -- d_f ~ 91/48. Nothing is averaged in secret: the mean line is
    built from the very points shown. One filling per size is also handed back for the live thumbnail so
    the drawn grid is itself one of the plotted points. Returns a JSON-friendly dict."""
    rng = np.random.default_rng(seed)
    pgrid = [round(p_lo + (p_hi - p_lo) * i / (p_steps - 1), 4) for i in range(p_steps)]
    grids = []
    for n in sizes:
        coords, edges, N, top, bottom, left, right, bbox = _square_demo(n)
        neigh = [[] for _ in range(N)]
        for a, b in edges:
            neigh[a].append(b); neigh[b].append(a)
        ks = [min(int(round(p * N)), N) for p in pgrid]
        shares = []
        thumb_order = None
        for fi in range(fillings):
            order = [int(x) for x in rng.permutation(N)]
            if fi == 0: thumb_order = order
            curve = _smax_by_k(N, neigh, order)
            shares.append([round(100.0 * curve[k] / N, 2) for k in ks])
        grids.append({"n": n, "N": N, "coords": coords, "edges": edges,
                      "top": list(top), "bottom": list(bottom), "left": list(left), "right": list(right),
                      "bbox": bbox, "order": thumb_order, "shares": shares})
    return {"sizes": list(sizes), "pgrid": pgrid, "pc": 0.5927, "fillings": fillings, "grids": grids}


def _span_onset_k(N, neigh, top, bottom, left, right, order):
    """Number of open sites at which the grid FIRST spans (top<->bottom or left<->right), for one random
    filling. Incremental union-find with a per-cluster boundary bitmask; returns the step (so k/N is the
    occupation p at which this realization percolates). This crossing point varies run to run, and the
    spread of it over many fillings is the finite-size transition width -- which shrinks as L^{-1/nu}."""
    T, B, L, R = 1, 2, 4, 8
    fm = [0] * N
    for v in top:    fm[v] |= T
    for v in bottom: fm[v] |= B
    for v in left:   fm[v] |= L
    for v in right:  fm[v] |= R
    parent = list(range(N)); size = [1] * N; isopen = bytearray(N)
    def find(x):
        r = x
        while parent[r] != r: r = parent[r]
        while parent[x] != r: parent[x], x = r, parent[x]
        return r
    for step, v in enumerate(order, 1):
        isopen[v] = 1; rv = find(v)
        for nb in neigh[v]:
            if isopen[nb]:
                ra, rb = rv, find(nb)
                if ra != rb:
                    if size[ra] < size[rb]: ra, rb = rb, ra
                    parent[rb] = ra; size[ra] += size[rb]; fm[ra] |= fm[rb]; rv = ra
        m = fm[rv]
        if (m & (T | B)) == (T | B) or (m & (L | R)) == (L | R):
            return step
    return N


def build_nu_demo(sizes=(12, 18, 28, 42, 64), fillings=120, seed=2,
                  p_lo=0.45, p_hi=0.75, p_steps=61):
    """Data for the interactive CORRELATION-LENGTH (nu) toy: a ladder of SQUARE grids. For each size we
    run `fillings` fillings and record the occupation p = k/N at which each first spans. The spread of
    those crossing points is the finite-size transition width: small grids percolate over a FUZZY range
    of p, large grids SNAP. The toy plots the spanning-probability curve R(p,L) = fraction of fillings
    spanning by p (a sigmoid that steepens with L) and reads off the width sigma_L; across the ladder
    sigma_L ~ L^{-1/nu}, so the slope of log(width) vs log(L) gives the correlation-length exponent nu
    (2D percolation: 4/3). Returns a JSON-friendly dict (sizes, pgrid, per-size R + mean + width)."""
    rng = np.random.default_rng(seed)
    pgrid = [round(p_lo + (p_hi - p_lo) * i / (p_steps - 1), 4) for i in range(p_steps)]
    grids = []
    for n in sizes:
        coords, edges, N, top, bottom, left, right, bbox = _square_demo(n)
        neigh = [[] for _ in range(N)]
        for a, b in edges:
            neigh[a].append(b); neigh[b].append(a)
        top, bottom, left, right = list(top), list(bottom), list(left), list(right)
        crossings = sorted(_span_onset_k(N, neigh, top, bottom, left, right,
                                         [int(x) for x in rng.permutation(N)]) / N
                           for _ in range(fillings))
        R = [round(sum(1 for c in crossings if c <= p) / fillings, 4) for p in pgrid]
        mean = float(np.mean(crossings)); std = float(np.std(crossings, ddof=1))
        grids.append({"n": n, "N": N, "R": R, "mean": round(mean, 4), "std": round(std, 5)})
    return {"sizes": list(sizes), "pgrid": pgrid, "pc": 0.5927, "fillings": fillings, "grids": grids}
