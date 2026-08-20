import math
from collections import deque

import numpy as np

from builders.dual_graph_builder import collect_leaf_polygons, build_dual_from_polygons
from builders.direct_graph_builder import graph_from_polygons

# The APERIODIC Comet (Tile(1,0)) and Chevron (Tile(0,1)) tilings: the hat's aperiodic arrangement
# deformed to a family endpoint by FOLDING. We scale every unit-edge by sa and every sqrt3-edge by sb
# and re-integrate all vertex positions consistently -- each hat tile's a-edges sum to zero and its
# b-edges sum to zero SEPARATELY, so every cycle still closes and the folded tiling is gap-free.
#   Comet   = (sa, sb) = (1, 0)  -> the sqrt3-edges collapse to points
#   Chevron = (sa, sb) = (0, 1)  -> the unit-edges collapse to points
# The shared graph builders then de-dup the now-coincident collapsed-edge endpoints, which performs
# the edge contraction for free: graph_from_polygons -> the direct (vertex) graph, and
# build_dual_from_polygons -> the Delone dual (tile adjacency). See the memory
# [[collapsed-hat-comet-chevron]]. One known wrinkle: a single hat-generator artifact (2 edges at
# ~0.36 of the patch radius) folds inconsistently -- negligible for percolation (2 edges in millions).

SQ3 = math.sqrt(3)


def _hat_patch(reach):
    # Same metatile-substitution patch the hat direct/dual builders use (kept local to avoid importing
    # the interface layer). reach = number of inflations; leaves are collected all the way down.
    from generators.hat_generator import (H_init, T_init, P_init, F_init,
                                          constructPatch, constructMetatiles)
    cur = [H_init(), T_init(), P_init(), F_init()]
    p = None
    for _ in range(reach):
        p = constructPatch(*cur); cur = constructMetatiles(p)
    return p, reach + 1


def _cell(p):
    # Every hat vertex sits on the triangular lattice (0.5*hexPt(int,int)); snap to its integer cell so
    # shared corners de-dup EXACTLY (a float tolerance mis-merges near the folded collapse points).
    yi = int(round(4.0 * p[1] / SQ3))
    return (int(round(2.0 * p[0] - 0.5 * yi)), yi)


def folded_polys(reach, sa, sb):
    """The folded comet/chevron tile polygons (list of (14,2) arrays) for the aperiodic hat patch."""
    patch, lvl = _hat_patch(reach)
    tiles = collect_leaf_polygons(patch, lvl)

    vindex = {}; verts = []; loops = []
    for poly in tiles:
        loop = []
        for pt in poly:
            c = _cell(pt)
            if c not in vindex:
                vindex[c] = len(verts); verts.append(pt)
            loop.append(vindex[c])
        loops.append(loop)
    coords = np.asarray(verts, float); n = len(coords)

    adj = [set() for _ in range(n)]
    for loop in loops:
        m = len(loop)
        for k in range(m):
            a, b = loop[k], loop[(k + 1) % m]
            if a != b:
                adj[a].add(b); adj[b].add(a)
    neigh = [list(s) for s in adj]

    # Re-integrate positions from the class-scaled edge vectors, BFS from the central vertex (rooting at
    # the centre keeps the folded window we percolate through consistent; the lone artifact sits at r~0.36).
    ctr = coords.mean(axis=0)
    root = int(np.argmin(np.hypot(coords[:, 0] - ctr[0], coords[:, 1] - ctr[1])))
    P = np.full((n, 2), np.nan); P[root] = coords[root]
    dq = deque([root])
    while dq:
        i = dq.popleft()
        for j in neigh[i]:
            if math.isnan(P[j, 0]):
                v = coords[j] - coords[i]; length = math.hypot(v[0], v[1])
                P[j] = P[i] + (sa if length < 0.7 else sb) * v
                dq.append(j)
    unreached = np.isnan(P[:, 0])
    if unreached.any():
        P[unreached] = coords[unreached]

    return [_dedup_ring(P[loop]) for loop in loops]


def _dedup_ring(poly, tol=1e-6):
    """Drop consecutive coincident vertices (a collapsed edge folds to a zero-length side, i.e. two
    coincident corners). Without this the dual would read the collapsed edge's TWO coincident vertices
    as '2 shared vertices' = a shared edge, wrongly linking tiles that meet only at a point; and the
    vertex-mean centroid would double-count the collapse point. After it, a collapsed edge is a single
    corner (1 shared vertex -> not adjacent, per build_dual's >=2 rule)."""
    out = [poly[0]]
    for p in poly[1:]:
        if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > tol:
            out.append(p)
    if len(out) > 2 and math.hypot(out[-1][0] - out[0][0], out[-1][1] - out[0][1]) <= tol:
        out.pop()
    return np.asarray(out)


# (sa, sb) for each endpoint.
_PARAMS = {"comet": (1.0, 0.0), "chevron": (0.0, 1.0)}


def collapse_polys(which, reach):
    sa, sb = _PARAMS[which]
    return folded_polys(reach, sa, sb)


# The TURTLE Tile(sqrt3,1) is the hat Tile(1,sqrt3) with its two edge CLASSES swapped (the a-edges and
# b-edges trade lengths). It is the same folding maker but a NON-degenerate re-embedding -- both scales
# are > 0, so nothing collapses: it's a genuinely different tiling GEOMETRY carrying the SAME adjacency.
# That is the concrete "distinct build" behind the family-invariance claim: the isomorphism checker
# confirms turtle direct/dual == hat direct/dual (identical V/E/degree/triangle/WL), so identical p_c
# by construction -- no separate Monte-Carlo run. Native hat = folded_polys(reach, 1, 1).
_TURTLE_SCALE = (SQ3, 1.0 / SQ3)     # a-edges *sqrt3 (short->long), b-edges *1/sqrt3 (long->short)


def turtle_polys(reach):
    """Turtle Tile(sqrt3,1) tile polygons: the hat leaves re-embedded with the two edge lengths swapped."""
    return folded_polys(reach, *_TURTLE_SCALE)


def turtle_graph(kind, reach):
    """(coords, neighbors, polys) for the turtle; kind in {'direct','dual'}. Isomorphic to the hat."""
    polys = turtle_polys(reach)
    if kind == "dual":
        coords, neighbors, _ = build_dual_from_polygons(polys)
    else:
        coords, neighbors = graph_from_polygons(polys)
    return coords, neighbors, polys


def collapse_graph(which, kind, reach):
    """(coords, neighbors, polys) for the aperiodic comet/chevron. which in {'comet','chevron'};
    kind in {'direct','dual'}. The collapsed-edge de-dup happens inside the shared builders."""
    polys = collapse_polys(which, reach)
    if kind == "dual":
        coords, neighbors, _ = build_dual_from_polygons(polys)
    else:
        coords, neighbors = graph_from_polygons(polys)
    return coords, neighbors, polys
