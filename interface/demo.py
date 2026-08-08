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
