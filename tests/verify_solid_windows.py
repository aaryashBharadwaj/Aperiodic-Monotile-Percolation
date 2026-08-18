"""Independent geometric verification of the measurement-window cap.

The runtime finds the measurement frame with a *density heuristic* (largest_square_center: bin vertices,
mark dense cells, take the largest solid square) and caps windows at 0.95 x its side. That heuristic is
fast but could, in principle, over-reach into the ragged substitution fringe and silently bias spanning
statistics -- most plausibly on degenerate geometry (the folded comet/chevron, whose collapsed edges pile
coincident vertices into cells and distort the occupancy median the threshold is built on).

This test checks that concern directly and reproducibly, for ANY member: it takes the largest capped
window run_one would actually use, samples a dense grid across it, and confirms every sample point lies
inside some tile polygon. off-tile fraction ~ 0  =>  the window is genuinely gap-free and the cap is safe.
A nonzero fraction means the window pokes a gap/fringe and the cap over-reaches on that member.

Run:  python tests/verify_solid_windows.py            # default member set
      python tests/verify_solid_windows.py Hat 6      # one member at a given patch
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from matplotlib.path import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import interface.gui_backend as gb


def _member_polys(member, patch):
    """The tile polygons of the member's patch (same tiles the graph is built from)."""
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
    raise ValueError(f"no polygon source wired for {member!r}")


def off_tile_fraction(member, patch, cap=0.95, ngrid=140):
    """Fraction of a grid over the largest capped window that falls outside every tile."""
    bundle = gb.build_graph(member, "Direct (vertex)", patch)
    cx, cy = bundle["center"]; side = bundle["side"]
    polys = [np.asarray(p, float) for p in _member_polys(member, patch)]
    paths = [Path(p) for p in polys]
    centroids = np.array([p.mean(axis=0) for p in polys])
    tree = cKDTree(centroids)

    half = 0.5 * cap * side
    xs = np.linspace(cx - half, cx + half, ngrid)
    ys = np.linspace(cy - half, cy + half, ngrid)
    pts = np.array([[x, y] for x in xs for y in ys])
    # each point can only be inside a tile whose centroid is near it: test the 12 nearest tiles.
    _, nbr = tree.query(pts, k=min(12, len(polys)))
    nbr = np.atleast_2d(nbr)
    off = 0
    for i, pt in enumerate(pts):
        if not any(paths[j].contains_point(pt) for j in np.atleast_1d(nbr[i])):
            off += 1
    return off / len(pts), side, len(polys)


def main():
    if len(sys.argv) >= 2:
        member = sys.argv[1]
        patch = int(sys.argv[2]) if len(sys.argv) >= 3 else 5
        cases = [(member, patch)]
    else:
        # Production patches where cheap; the inflation members use r=5 as a faithful (much lighter)
        # proxy for the r=6 production patch -- the fringe geometry is scale-consistent, so a clean r=5
        # window certifies the same cap at r=6. (A small periodic block reads MARGINAL because the
        # inscribed square reaches the block's own edge; the production block sizes below do not.)
        cases = [("Hat", 5), ("Spectre", 5), (gb.COMET_AP, 5), (gb.CHEVRON_AP, 5),
                 ("Comet", 420), ("Chevron", 420), (gb.TILE11, 215)]
    print("member                 patch  side   tiles    off-tile %%   window cap")
    worst = 0.0
    for member, patch in cases:
        frac, side, ntiles = off_tile_fraction(member, patch)
        worst = max(worst, frac)
        flag = "OK" if frac < 1e-4 else ("MARGINAL" if frac < 0.005 else "FAIL -- cap over-reaches")
        print("%-22s %4d  %5.0f  %6d    %8.4f    0.95 x side   %s"
              % (member, patch, side, ntiles, 100 * frac, flag))
    print("\nworst off-tile fraction: %.4f%%  ->  %s"
          % (100 * worst, "cap is safe on all tested members" if worst < 1e-4 else "review flagged members"))
    return 0 if worst < 1e-4 else 1


if __name__ == "__main__":
    sys.exit(main())
