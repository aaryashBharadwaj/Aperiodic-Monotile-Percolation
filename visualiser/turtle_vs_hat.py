"""Figure: the turtle Tile(sqrt3,1) is the hat Tile(1,sqrt3) re-embedded with its two edge lengths
swapped. Same combinatorial tiling (tile i <-> tile i), so the same percolation graph and the same p_c
-- shown here, and certified by engine.isomorphism (identical direct+dual V/E/degree/triangle/WL).

    python -m visualiser.turtle_vs_hat [--reach 3] [--out figures_generated/turtle_vs_hat.png]

Native hat = folded_polys(reach, 1, 1); turtle = folded_polys(reach, sqrt3, 1/sqrt3). Both come from the
SAME leaf collection in the SAME order, so a handful of tiles are filled in matching colours across the
two panels to make the correspondence visible.
"""
import argparse
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from generators.aperiodic_collapse import folded_polys, turtle_polys


def _window(polys, frac=0.13):
    """Indices of tiles whose centroid lies in the central `frac` box -- a legible sub-patch, not the
    whole cloud. Uses the HAT centroids so the SAME tile indices are cropped from both panels."""
    ctr = np.array([p.mean(0) for p in polys])
    mid = ctr.mean(0); span = (ctr.max(0) - ctr.min(0)) * frac / 2
    keep = np.where((np.abs(ctr[:, 0] - mid[0]) < span[0]) & (np.abs(ctr[:, 1] - mid[1]) < span[1]))[0]
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reach", type=int, default=3)
    ap.add_argument("--out", default="figures_generated/turtle_vs_hat.png")
    ap.add_argument("--nhighlight", type=int, default=5)
    ap.add_argument("--frac", type=float, default=0.13, help="central fraction of the patch to crop (smaller = more zoomed)")
    args = ap.parse_args()

    hat = folded_polys(args.reach, 1.0, 1.0)          # native hat, same tile order as the turtle
    tur = turtle_polys(args.reach)                    # Tile(sqrt3,1)

    keep = _window(hat, args.frac)
    # pick a few well-separated tiles in the window to colour-match across panels
    ctr = np.array([hat[i].mean(0) for i in keep])
    order = np.argsort(ctr[:, 0] + 0.7 * ctr[:, 1])
    picks = [keep[order[int(k)]] for k in np.linspace(0, len(order) - 1, args.nhighlight)]
    cmap = plt.get_cmap("tab10")
    hl = {idx: cmap(k % 10) for k, idx in enumerate(picks)}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6))
    for ax, polys, title in [(axes[0], hat, r"Hat  $\mathrm{Tile}(1,\sqrt{3})$"),
                             (axes[1], tur, r"Turtle  $\mathrm{Tile}(\sqrt{3},1)$")]:
        base = [polys[i] for i in keep if i not in hl]
        ax.add_collection(PolyCollection(base, facecolors="#f2f2f2", edgecolors="#333333", linewidths=0.6))
        for idx, col in hl.items():
            ax.add_collection(PolyCollection([polys[idx]], facecolors=[col], edgecolors="#111111",
                                             linewidths=1.1, alpha=0.9))
        allpts = np.concatenate([polys[i] for i in keep])
        ax.set_xlim(allpts[:, 0].min(), allpts[:, 0].max())
        ax.set_ylim(allpts[:, 1].min(), allpts[:, 1].max())
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=15)

    fig.suptitle("Same tiling, edge lengths swapped  →  identical graph, identical $p_c$",
                 fontsize=13, y=0.98)
    fig.text(0.5, 0.02, "matched colours = the same tile in both tilings (tile $i \\leftrightarrow$ tile $i$)",
             ha="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=170)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
