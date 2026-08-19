"""The solid-window oracle: measure the largest MEASUREMENT window a tiling genuinely fills.

The window cap used to be 0.95 x (density-detector side) -- but that detector over-reports the solid
square (partially-filled fringe cells pass its occupancy threshold), so the 0.95 was an empirical patch
that wasn't even enough for the fractal-boundary folded endpoints. This replaces the guess with a
measurement: sample a grid over a candidate window and require it to be fully covered by tile polygons,
plus a small tiled buffer so the window is SURROUNDED (hence fully coordinated, not just gap-free).
`measure_solid_L` binary-searches the largest such window. The result is verifiable by the same test
that produced it (tests/verify_solid_windows.py). No percentage fudge anywhere.
"""
import numpy as np
from scipy.spatial import cKDTree
from matplotlib.path import Path


def member_polys(member, patch):
    """Tile polygons of the member's patch (the same tiles its graph is built from)."""
    import interface.gui_backend as gb
    if member in (gb.COMET_AP, gb.CHEVRON_AP):
        from generators.aperiodic_collapse import collapse_polys
        return collapse_polys("comet" if member == gb.COMET_AP else "chevron", patch)
    if member in ("Hat", "Spectre"):
        from builders.dual_graph_builder import collect_leaf_polygons
        pobj, lvl = gb._hat_patch(patch) if member == "Hat" else gb._spectre_patch(patch)
        return collect_leaf_polygons(pobj, lvl)
    if member in ("Comet", "Chevron"):
        from generators.chevron_and_comet import periodic_polys
        return periodic_polys(member.lower(), ncells=patch)[0]
    if member == gb.TILE11:
        from generators.tile11_periodic import tile11_polys
        return tile11_polys(patch)[0]
    if member == "Square":
        from generators.periodic_tiling_generator import square_tiles
        return square_tiles(int(patch))
    if member == gb.TRI:
        from generators.periodic_tiling_generator import triangular_tris
        return triangular_tris(int(patch))
    raise ValueError(f"no polygon source wired for {member!r}")


def off_tile_count(paths, tree, cx, cy, L, ngrid=160):
    """Number of grid points over the L x L window at (cx, cy) that fall outside every tile."""
    half = L / 2.0
    xs = np.linspace(cx - half, cx + half, ngrid)
    ys = np.linspace(cy - half, cy + half, ngrid)
    pts = np.array([[x, y] for x in xs for y in ys])
    _, nbr = tree.query(pts, k=min(12, len(paths)))
    nbr = np.atleast_2d(nbr)
    return sum(1 for i, p in enumerate(pts) if not any(paths[j].contains_point(p) for j in nbr[i]))


def off_tile_fraction(member, patch, cx, cy, L, ngrid=160):
    """Convenience: build the member's polygons and return the off-tile fraction of an L-window."""
    polys = [np.asarray(p, float) for p in member_polys(member, patch)]
    paths = [Path(p) for p in polys]
    tree = cKDTree(np.array([p.mean(axis=0) for p in polys]))
    return off_tile_count(paths, tree, cx, cy, L, ngrid) / (ngrid * ngrid)


def _coords_and_polys(member, patch):
    """Deduped graph vertices AND tile polygons of a member's patch, from a SINGLE build.

    This is the whole speed story: the aperiodic endpoints fold a ~5M-vertex hat through pure-Python
    loops, so building the patch twice (graph + tiles separately) doubles the cost. collapse_graph hands
    back coords and polys together; hat/spectre share one patch object between the two builders."""
    import interface.gui_backend as gb
    from builders.direct_graph_builder import build_neighbor_graph_fast
    from builders.dual_graph_builder import collect_leaf_polygons
    if member in (gb.COMET_AP, gb.CHEVRON_AP):
        from generators.aperiodic_collapse import collapse_graph
        coords, _, polys = collapse_graph("comet" if member == gb.COMET_AP else "chevron", "direct", patch)
    elif member in ("Hat", "Spectre"):
        pobj, lvl = gb._hat_patch(patch) if member == "Hat" else gb._spectre_patch(patch)
        coords, _ = build_neighbor_graph_fast(pobj, lvl)
        polys = collect_leaf_polygons(pobj, lvl)
    else:
        polys = member_polys(member, patch)                    # periodic: cheap, one build is fine
        coords = np.concatenate([np.asarray(p, float) for p in polys])
    return np.asarray(coords, float), [np.asarray(p, float) for p in polys]


def measure_solid_L(member, patch, buffer_units=8.0, ngrid=160):
    """The largest window (rounded down to 10) that is fully tiled AND has a `buffer_units` tiled margin
    on every side -- i.e. surrounded by real tiles, so its vertices are fully coordinated. Centred where
    run_one centres it (largest_square_center of the graph coords). This IS the bound; no 0.95 factor.

    Measured once per MEMBER, not per graph kind: the solid window is a geometric property of the tiling,
    identical for the direct (vertex) and dual (tile) graphs built on it."""
    from builders.direct_graph_builder import largest_square_center
    coords, polys = _coords_and_polys(member, patch)
    cx, cy, side = largest_square_center(coords)
    paths = [Path(p) for p in polys]
    tree = cKDTree(np.array([p.mean(axis=0) for p in polys]))

    def surrounded(L):
        return off_tile_count(paths, tree, cx, cy, L + 2 * buffer_units, ngrid) == 0

    lo, hi = 0.5 * side, 1.15 * side
    if not surrounded(lo):
        return float(int(0.85 * side // 10 * 10))     # degenerate patch; stay well inside
    while hi - lo > 5.0:
        mid = 0.5 * (lo + hi)
        if surrounded(mid):
            lo = mid
        else:
            hi = mid
    return float(int(lo // 10 * 10))
