import argparse
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless default; a GUI backend overrides this at import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import PolyCollection

# Pure drawing (no percolation). The functions return matplotlib Figures so they are reused verbatim
# It also doubles as a GENERATOR VALIDATION: a correct substitution tiling renders GAP-FREE, so a wrong transform shows up immediately as visible gaps or overlaps.

_BLUE = (0.72, 0.85, 0.96)
_GOLD = (0.95, 0.74, 0.33)
_GREY = (0.82, 0.82, 0.84)
_PURPLE = (0.66, 0.52, 0.80)
_ORANGE = (0.96, 0.64, 0.38)
_GREEN = (0.55, 0.78, 0.55)
S3 = math.sqrt(3)
# The named family members in (a,b) space.
_NAMED = {(1.0, S3): "hat", (S3, 1.0): "turtle", (1.0, 1.0): "spectre",
          (1.0, 0.0): "comet", (0.0, 1.0): "chevron"}

# Percolation-THRESHOLD class of Tile(a,b). The threshold depends only on the graph's adjacency,
# which is invariant under stretching (a,b) as long as the combinatorics don't change  so every
# generic aperiodic member (a != b, both > 0) shares the HAT'S threshold. The classes split only
# where the combinatorics change: a==b (spectre), and the degenerate edges a=0 / b=0 (periodic).
def _threshold_class(a, b):
    if a == 0 and b == 0:
        return "degenerate", _GREY
    if b == 0:
        return "comet (periodic)", _ORANGE
    if a == 0:
        return "chevron (periodic)", _GREEN
    if abs(a - b) < 1e-9:
        return "spectre (a=b)", _PURPLE
    return "aperiodic hat family (a≠b) — one shared threshold", _BLUE

# Shoelace formula: sum of x_i*y_{i+1} - x_{i+1}*y_i over the outline. The SIGN tells you which way
# round the vertices go, and reflecting a tile reverses that so a sign test separates hats from
# mirrored hats without any geometry. np.roll shifts by one to pair each vertex with the next.
def _signed_areas(polys):
    out = []
    for p in polys:
        p = np.asarray(p); x, y = p[:, 0], p[:, 1]
        out.append(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return np.array(out)

# colour by chirality, hue by orientation
def _tile_colors(polys, mode):
    if mode == "chirality":
        a = _signed_areas(polys)
        maj = np.sign(np.median(a)) or 1.0
        return [_BLUE if np.sign(v) == maj else _GOLD for v in a]
    if mode == "orientation":
        polys = [np.asarray(p) for p in polys]
        ang = np.array([math.atan2(p[1, 1] - p[0, 1], p[1, 0] - p[0, 0]) % (2 * math.pi)
                        for p in polys])
        return [cm.twilight(t / (2 * math.pi)) for t in ang]
    return None

# Draw tile polygons (each an (n,2) array of outline vertices) as filled cells. Color_by in {None, 'chirality', 'orientation'}. Returns the matplotlib Figure.
def render_tiling(polys, ax=None, facecolor=_BLUE, edgecolor="black", lw=0.4, title=None,
                  color_by=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure
    fc = _tile_colors(polys, color_by) or facecolor
    ax.add_collection(PolyCollection([np.asarray(p) for p in polys],
                                     facecolors=fc, edgecolors=edgecolor, linewidths=lw))
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12)
    return fig

# Leaf-tile polygons for a named tiling: 'hat', 'spectre', 'comet', 'chevron'
def tiling_polygons(name, level=3, ncells=7):
    name = name.lower()
    if name == "hat":
        from generators.hat_generator import (H_init, T_init, P_init, F_init,
                                   constructPatch, constructMetatiles)
        from builders.dual_graph_builder import collect_leaf_polygons
        cur = [H_init(), T_init(), P_init(), F_init()]
        p = None
        for _ in range(level):
            p = constructPatch(*cur); cur = constructMetatiles(p)
        return collect_leaf_polygons(p, level + 1)
    if name == "spectre":
        from generators.spectre_generator import build_spectre_patch
        from builders.dual_graph_builder import collect_leaf_polygons
        p = build_spectre_patch(level)
        return collect_leaf_polygons(p, level + 1)
    if name in ("comet", "chevron"):
        # Un-subdivided tile (the drawn outline; subdivision would change the orientation colour);
        # the block-stamping is shared with the percolation path via periodic_block.
        from generators.chevron_and_comet import tile_ab, periodic_block
        a, b = (1.0, 0.0) if name == "comet" else (0.0, 1.0)
        return periodic_block(tile_ab(a, b), ncells)[0]
    raise ValueError(f"unknown tiling '{name}' (expected hat/spectre/comet/chevron)")


def _named(a, b, tol=0.04):
    for (na, nb), nm in _NAMED.items():
        if abs(a - na) < tol and abs(b - nb) < tol:
            return nm
    return None


def tile_ab_grid(a_values, b_values):
    from generators.chevron_and_comet import tile_ab
    import matplotlib.patches as mpatches
    na, nb = len(a_values), len(b_values)
    fig, axes = plt.subplots(na, nb, figsize=(1.9 * nb, 1.9 * na))
    axes = np.atleast_2d(axes)
    seen = {}
    for i, a in enumerate(a_values):
        for j, b in enumerate(b_values):
            ax = axes[i, j]; ax.set_aspect("equal"); ax.axis("off")
            cls, col = _threshold_class(a, b)
            seen.setdefault(cls, col)
            if not (a == 0 and b == 0):
                try:
                    render_tiling([tile_ab(a, b)], ax=ax, facecolor=col, lw=0.7)
                except Exception:
                    pass
            nm = _named(a, b)
            if nm:
                ax.set_title(f"{nm}\n({a:.2g},{b:.2g})", fontsize=8, color="darkred", fontweight="bold")
            else:
                ax.set_title(f"({a:.2g},{b:.2g})", fontsize=7, color="0.35")
    # legend: colour -> threshold class
    handles = [mpatches.Patch(facecolor=c, edgecolor="black", label=k) for k, c in seen.items()]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle("Tile(a,b): coloured by percolation-threshold class  —  same colour = same threshold",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="all",
                    choices=["hat", "spectre", "comet", "chevron", "grid", "all"])
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--ncells", type=int, default=7)
    ap.add_argument("--out_dir", default=os.path.join("paper_results", "figures"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # insightful colouring per tiling: hat -> reflected tiles; spectre/family -> rotational classes
    color_mode = {"hat": "chirality", "spectre": "orientation",
                  "comet": "orientation", "chevron": "orientation"}
    todo = (["hat", "spectre", "comet", "chevron", "grid"]
            if args.what == "all" else [args.what])
    for w in todo:
        if w == "grid":
            fig = tile_ab_grid([0.0, 0.35, 0.7, 1.0, 1.35, S3], [0.0, 0.35, 0.7, 1.0, 1.35, S3])
        else:
            polys = tiling_polygons(w, level=args.level, ncells=args.ncells)
            fig = render_tiling(polys, title=f"{w}  ({len(polys)} tiles)", color_by=color_mode[w])
        out = os.path.join(args.out_dir, f"tiling_{w}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}", flush=True)
