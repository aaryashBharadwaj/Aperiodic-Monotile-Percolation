"""Geometry for the periodic Tile(a,b) family endpoints — Comet (Tile(1,0)) and Chevron (Tile(0,1)).

LIBRARY ONLY (no __main__): builds the Tile(a,b) polygon and a periodic block of it, shared by the
GUI backend (percolation on comet/chevron via gui_backend.build_graph) and the tiling renderer.
The percolation RUN itself goes through the consolidated runner (runner/percolate.py) or the GUI
Run button; see REPRODUCE.md for the exact commands.
"""
import math
import itertools

import numpy as np
import generators.hat_generator as hg
from builders.direct_graph_builder import graph_from_polygons

SQ3 = math.sqrt(3)
E1 = np.array([1.0, 0.0])
E2 = np.array([0.5, SQ3 / 2])


def tile_ab(a, b):
    pts = [(p['x'], p['y']) for p in hg.hat_outline]; n = len(pts); out = [(0.0, 0.0)]
    for i in range(n - 1):
        x0, y0 = pts[i]; x1, y1 = pts[i + 1]; vx, vy = x1 - x0, y1 - y0; L = math.hypot(vx, vy)
        s = a if abs(L - round(L)) < 1e-6 else (b / SQ3 if abs(L / SQ3 - round(L / SQ3)) < 1e-6 else 1.0)
        out.append((out[-1][0] + s * vx, out[-1][1] + s * vy))
    poly = np.array(out)
    v = poly[1] - poly[0]; ang = math.degrees(math.atan2(v[1], v[0])) % 60.0
    if abs(ang - 30.0) < 1e-3:
        c, s = math.cos(math.radians(-30)), math.sin(math.radians(-30))
        poly = poly @ np.array([[c, -s], [s, c]]).T
    return poly


def _subdivide(poly, unit=1.0):
    """Insert the intermediate lattice vertices on any boundary edge longer than one unit.
    The comet (Tile(1,0)) has length-2 boundary edges — two collinear unit edges whose shared
    lattice vertex sits in the MIDDLE of the edge (degree-2). Without that midpoint, two tiles
    meeting along such an edge never share a graph node, so the vertex graph splits into
    disconnected pieces and nothing ever spans (p_c = 1). Re-inserting every unit-spaced vertex
    makes the coincident-vertex dedup across tiles reconnect them. Chevron edges are already
    unit length, so this is a no-op there."""
    out = []
    n = len(poly)
    for i in range(n):
        p = poly[i]; q = poly[(i + 1) % n]
        out.append(p)
        k = int(round(math.hypot(q[0] - p[0], q[1] - p[1]) / unit))
        for t in range(1, k):                 # k-1 interior points (only when k >= 2)
            out.append(p + (q - p) * (t / k))
    return np.array(out)


def tri_set(poly):
    P = [tuple(p) for p in poly]; xs = poly[:, 0]; ys = poly[:, 1]; out = set()
    def cent(i, j, s):
        vs = [(i, j), (i + 1, j), (i, j + 1)] if s == 0 else [(i + 1, j), (i + 1, j + 1), (i, j + 1)]
        c = sum(v[0] * E1 + v[1] * E2 for v in vs) / 3.0; return c
    def pip(x, y):
        n = len(P); ins = False; px, py = P[-1]
        for qx, qy in P:
            if ((qy > y) != (py > y)) and (x < (px - qx) * (y - qy) / (py - qy + 1e-30) + qx): ins = not ins
            px, py = qx, qy
        return ins
    for i in range(int(xs.min()) - 2, int(xs.max()) + 2):
        for j in range(int(ys.min() / (SQ3 / 2)) - 2, int(ys.max() / (SQ3 / 2)) + 2):
            for s in (0, 1):
                c = cent(i, j, s)
                if pip(c[0], c[1]): out.add((i, j, s))
    return out


def find_lattice(ts):
    need = len(ts) // 2
    def verify(u1, u2):
        cover = {}
        for m in range(-3, 4):
            for n in range(-3, 4):
                si, sj = m * u1[0] + n * u2[0], m * u1[1] + n * u2[1]
                for (i, j, s) in ts: cover[(i + si, j + sj, s)] = cover.get((i + si, j + sj, s), 0) + 1
        return all(cover.get((i, j, s), 0) == 1 for i in range(-2, 3) for j in range(-2, 3) for s in (0, 1))
    for u1 in itertools.product(range(-4, 5), repeat=2):
        for u2 in itertools.product(range(-4, 5), repeat=2):
            if abs(u1[0] * u2[1] - u1[1] * u2[0]) == need and verify(u1, u2):
                return u1, u2
    return None


def periodic_block(tile, ncells):
    """Find the lattice vectors that tile the plane with `tile` (via its triangle cover) and stamp an
    ncells x ncells block. Returns (polys, block_span). Shared by periodic_polys (the percolation
    graph) and the renderer (visualiser.run_tiling_render.tiling_polygons). (tri_set depends only on
    the tile's REGION, so it is unaffected by whether `tile` has been _subdivide()d.)"""
    u1, u2 = find_lattice(tri_set(tile))
    U1 = u1[0] * E1 + u1[1] * E2; U2 = u2[0] * E1 + u2[1] * E2
    polys = [tile + m * U1 + n * U2 for m in range(ncells) for n in range(ncells)]
    return polys, float(np.hypot(*U1)) * ncells


def periodic_polys(name, ncells=150):
    """The periodic block of Tile(a,b) leaf polygons for 'comet'/'chevron' — the geometry BOTH the
    direct (graph_from_polygons) and dual (build_dual_from_polygons) graphs are built from. The tile
    is _subdivide()d first so the comet's collinear mid-edge lattice vertices exist (else its vertex
    graph is disconnected). Returns (polys, block_span)."""
    a, b = (1.0, 0.0) if name == "comet" else (0.0, 1.0)
    return periodic_block(_subdivide(tile_ab(a, b)), ncells)


def periodic_graph(name, ncells=150):
    # Same direct (vertex) graph as the hat/spectre -- dedup coincident vertices, perimeter edges,
    # adjacency -- through the shared builder. See direct_graph_builder.graph_from_polygons.
    polys, span = periodic_polys(name, ncells)
    nodes, neigh = graph_from_polygons(polys)
    return nodes, neigh, span
