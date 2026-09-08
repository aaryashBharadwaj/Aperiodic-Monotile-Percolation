"""Figure builders for the percolation GUI: the tiling-render showcase (visualise) and the two
results plots (finite-size-scaling extrapolation + per-size convergence). Split out of gui_backend so
that module stays focused on the numerics; these render Matplotlib Figures the UI displays.

    visualise(tiling, size, ...)      -> a rendered tiling Figure (+ optional graph overlay)
    fss_figure(result)                -> the I/U/A extrapolation plot (site & bond)
    convergence_figure(result)        -> per-size crossing estimate vs L (I & U)

Shared primitives (resolve_member, the Penrose helpers, family_member, constants) are imported from
interface.gui_backend; the dependency is one-way (gui_backend never imports this module).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from builders.direct_graph_builder import graph_from_polygons
from builders.dual_graph_builder import build_dual_from_polygons
from visualiser.run_tiling_render import tiling_polygons, render_tiling
from generators.chevron_and_comet import tile_ab
from generators.periodic_tiling_generator import square_tiles, triangular_tris
from interface.gui_backend import (resolve_member, family_member, _penrose_tiling, _penrose_polys,
                                    S3, NU, TRI, FAMILY, TILE11, COMET_AP, CHEVRON_AP)


# ----------------------------------------------------------------------------- visualise engine
def _render_polys(tiling, size, a=1.0, b=S3):
    """(polys, title, color_by, penrose_obj) for the visualiser. penrose_obj is returned so a graph
    overlay can use the correct (rhombus-rule) Penrose builder."""
    name = resolve_member(tiling, a, b)
    if name in ("Hat", "Spectre", "Comet", "Chevron"):
        polys = tiling_polygons(name.lower(), level=size, ncells=size)
        color_by = "chirality" if name == "Hat" else "orientation"
        return polys, f"{name}  ({len(polys)} tiles)", color_by, None
    if name in (COMET_AP, CHEVRON_AP):
        from generators.aperiodic_collapse import collapse_polys
        which = "comet" if name == COMET_AP else "chevron"
        polys = collapse_polys(which, size)    # size = hat inflation depth (RENDER_CTL keeps it small)
        return polys, f"{name}  ({len(polys)} tiles)", "chirality", None
    if name == TILE11:
        from generators.tile11_periodic import tile11_polys
        polys, _ = tile11_polys(size)          # size = grow reach (RENDER_CTL keeps it small/legible)
        # chirality (not orientation): the tiling is A vs its mirror B, so the signed-area sign splits
        # them into two clean colours -- the twilight-by-angle 'orientation' map just muddied them.
        return polys, f"Tile(1,1) periodic  ({len(polys)} tiles)", "chirality", None
    if name == "Square":
        polys = square_tiles(size)
        return polys, f"Square  ({len(polys)} tiles)", None, None
    if name == "Penrose":
        t = _penrose_tiling(size, scale=200)
        polys = _penrose_polys(t)
        return polys, f"Penrose  ({len(polys)} triangles)", "orientation", t
    if name == TRI:
        polys = triangular_tris(size)
        return polys, f"Triangular  ({len(polys)} triangles)", "orientation", None
    raise ValueError(f"cannot render {name!r}")


def _overlay_graph(fig, tiling, polys, graph_type, penrose_obj):
    """Draw the actual percolation graph (nodes + edges) on top of the tiles, using the correct
    builder for the selected graph type -- so it shows exactly what gets percolated."""
    name = resolve_member(tiling)
    if graph_type.startswith("Dual"):
        nodes, neigh, _ = build_dual_from_polygons(polys)
    elif name == "Penrose" and penrose_obj is not None:
        from builders.penrose_graph_builder import build_penrose_neighbor_graph
        nodes, neigh, _ = build_penrose_neighbor_graph(penrose_obj)
    else:
        nodes, neigh = graph_from_polygons(polys)
    segs = [[(nodes[i, 0], nodes[i, 1]), (nodes[j, 0], nodes[j, 1])]
            for i, nb in enumerate(neigh) for j in nb if i < j]
    # Thin the overlay as the graph grows so big renders stay legible (a dense mesh of thick lines
    # is a smear); small graphs keep chunky, readable nodes/edges.
    m = len(nodes)
    lw = max(0.12, min(0.6, 55.0 / (m ** 0.5)))
    ms = max(0.8, min(3.2, 150.0 / (m ** 0.5)))
    ax = fig.axes[0]
    ax.add_collection(LineCollection(segs, colors="#111111", linewidths=lw, alpha=0.55, zorder=3))
    ax.plot(nodes[:, 0], nodes[:, 1], ".", color="crimson", ms=ms, zorder=4)
    return len(nodes), len(segs)


# Edge-class colours for the Tile(a,b) family view: a = the unit edges (blue), b = the sqrt(3) edges
# (red), o = any other edge (grey). Same legend the sidebar sliders use, so colouring the outline by
# class shows at a glance which edges each slider stretches.
_FAMILY_EDGE_COL = {"a": "#2e6f95", "b": "#d1495b", "o": "#b0b0b0"}


def _family_tile_edge_classes():
    """Class ('a'/'b'/'o') of each edge of the Tile(a,b) polygon, in vertex order. Recomputed by the
    SAME test tile_ab uses (an edge is 'a' if the hat outline's edge is unit length, 'b' if sqrt(3)),
    so edge i of tile_ab(a,b) -- poly[i] -> poly[(i+1) % n] -- carries class classes[i]. Independent of
    a,b (the classification is on the base hat), so it's computed once and reused for every morph."""
    import math
    import generators.hat_generator as hg
    pts = [(p["x"], p["y"]) for p in hg.hat_outline]
    n = len(pts)
    classes = []
    for i in range(n):
        x0, y0 = pts[i]; x1, y1 = pts[(i + 1) % n]
        L = math.hypot(x1 - x0, y1 - y0)
        if abs(L - round(L)) < 1e-6:
            classes.append("a")
        elif abs(L / S3 - round(L / S3)) < 1e-6:
            classes.append("b")
        else:
            classes.append("o")
    return classes


def visualise(tiling, size, a=1.0, b=S3, graph_type="Direct (vertex)", show_graph=False):
    """Render the chosen tiling at the chosen size. Returns (fig, n_tiles, graph_counts or None).
    This is the generator showcase -- proof the substitution/lattice code produces the real shape.
    No in-figure title (name/count/a-b/class are already in the UI), and a fixed modest figure size
    so it doesn't stretch to the full column width."""
    if tiling == FAMILY:
        member, cls = family_member(a, b)
        if member is None:
            fig, ax = plt.subplots(figsize=(5, 5)); ax.axis("off")
            ax.text(0.5, 0.5, "Tile(0,0)\ndegenerate", ha="center", va="center", fontsize=14)
            return fig, 0, None
        # Neutral fill + edges coloured by class (blue unit / red sqrt(3)) -- the same look as the old
        # slider explainer, so you can see which edges a and b each control as you morph the tile.
        poly = np.asarray(tile_ab(a, b))
        classes = _family_tile_edge_classes()
        n = len(poly)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.fill(poly[:, 0], poly[:, 1], facecolor="#f2f2f4", edgecolor="none", zorder=1)
        segs = [[poly[i], poly[(i + 1) % n]] for i in range(n)]
        cols = [_FAMILY_EDGE_COL[c] for c in classes]
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=3.0,
                                         capstyle="round", joinstyle="round", zorder=2))
        ax.autoscale_view(); ax.set_aspect("equal"); ax.axis("off")
        return fig, 1, None

    # Every tiling (incl. Triangular): draw the TILE image, then let the optional overlay show what
    # counts as nodes/edges -- direct = vertices + tile edges, dual = centroids + adjacency. For the
    # triangular tiling this is the nice one: connecting the triangle centroids (the dual) naturally
    # traces out the honeycomb, on top of the same triangles.
    polys, _title, color_by, pobj = _render_polys(tiling, size, a, b)
    fig, ax = plt.subplots(figsize=(5, 5))
    render_tiling(polys, ax=ax, color_by=color_by, lw=0.4)
    gcounts = None
    if show_graph:
        gcounts = _overlay_graph(fig, tiling, polys, graph_type, pobj)
    return fig, len(polys), gcounts


# ----------------------------------------------------------------------------- results figure
def _means_IUA(raw_I, raw_U):
    """Per-L Intersection / Union / Average estimator means and standard errors, matching
    extrapolate_pc_raw. Returns (mI,sI, mU,sU, mA,sA) as arrays."""
    mI, sI, mU, sU, mA, sA = [], [], [], [], [], []
    for rI, rU in zip(raw_I, raw_U):
        rI, rU = np.asarray(rI), np.asarray(rU); T = len(rI)
        a, b = rI.mean(), rU.mean()
        mI.append(a); mU.append(b); mA.append(0.5 * (a + b))
        si, su = rI.std(ddof=1), rU.std(ddof=1); cov = np.cov(rI, rU)[0, 1]
        sI.append(si / np.sqrt(T)); sU.append(su / np.sqrt(T))
        sA.append(0.5 * np.sqrt(si**2 / T + su**2 / T + 2 * cov / T))
    return tuple(np.array(v) for v in (mI, sI, mU, sU, mA, sA))


def _line_fit(x, y, s):
    """Weighted least-squares (intercept, slope) of y vs x -- used to draw each estimator's FSS line
    over the displayed scaling region. Same weighting as the reported extrapolation."""
    x = np.asarray(x); y = np.asarray(y); w = 1.0 / np.asarray(s) ** 2
    S = w.sum(); Sx = (w * x).sum(); Sxx = (w * x * x).sum(); Sy = (w * y).sum(); Sxy = (w * x * y).sum()
    D = S * Sxx - Sx * Sx
    return (Sxx * Sy - Sx * Sxy) / D, (S * Sxy - Sx * Sy) / D


def fss_figure(result, title=None):
    """Finite-size-scaling extrapolation on the SCALING REGION (L >= the average's cutoff) -- the points
    that determine p_c. The classic three-estimator view: Intersection (high), Union (low) and their
    Average each get a WLS line and L->inf intercept, and the three squeeze onto a common p_c (the
    convergence check). Only post-cutoff points are shown; the discarded small-L points (which carry the
    corrections to scaling) are shown separately by `convergence_figure(mark_cutoff=True)`, which plots
    the full range with the discarded region marked. Site and bond get their own panel. `title`, if given,
    is shown at the top identifying which tiling/graph the figure is for."""
    L = np.asarray(result["L"], dtype=float)
    x = L ** (-1.0 / NU)
    xs = np.linspace(0, x.max() * 1.02, 100)
    channels = [("Site", result["raw_SI"], result["raw_SU"], result["site"])]
    if result.get("bond") is not None:
        channels.append(("Bond", result["raw_BI"], result["raw_BU"], result["bond"]))

    fig, axes = plt.subplots(1, len(channels), figsize=(6.2 * len(channels), 5), squeeze=False)
    for ax, (name, rI, rU, res) in zip(axes[0], channels):
        mI, sI, mU, sU, mA, sA = _means_IUA(rI, rU)
        series = {"I": (np.asarray(mI), np.asarray(sI)),
                  "U": (np.asarray(mU), np.asarray(sU)),
                  "A": (np.asarray(mA), np.asarray(sA))}
        # Show only the SCALING REGION (L >= the average's cutoff). All three estimators get a WLS line
        # fit over the displayed points and drawn to the L->inf intercept: Intersection (high), Union
        # (low) and Average squeeze onto a common p_c. The discarded small-L points are the subject of
        # convergence_figure(mark_cutoff=True), not shown here.
        cutoff = res["A"].get("cutoff_L", float(L.min()))
        kept = L >= cutoff
        xk = x[kept]
        xline = np.linspace(0, xk.max() * 1.02, 100)
        for label, key, color, marker in [("Intersection", "I", "#1f77b4", "o"),
                                          ("Union",        "U", "#d62728", "s"),
                                          ("Average",      "A", "#2ca02c", "^")]:
            m, s = series[key]
            mk, sk = m[kept], s[kept]
            pc, slope = _line_fit(xk, mk, sk)
            ax.errorbar(xk, mk, yerr=sk, fmt=marker, color=color, ms=5.5, capsize=3, alpha=0.9,
                        zorder=4, label=label)
            ax.plot(xline, slope * xline + pc, "-", color=color, lw=1.4, zorder=3)
        # One reported intercept: the three estimators agree to the quoted precision, so a per-estimator
        # number would be three identical values -- show the single reported p_c (the average) instead.
        ax.plot(0, res["A"]["pc"], "X", color="#2ca02c", ms=13, mec="black", mew=0.8, zorder=6,
                label=f"$p_c$ = {res['A']['pc']:.4f}")
        ax.set_title(f"{name} percolation")
        ax.set_xlabel(r"$L^{-1/\nu}$   ($\nu=4/3$,  $L\to\infty$ at 0)")
        ax.set_ylabel(r"crossing $p_c$")
        ax.set_xlim(-xk.max() * 0.05, xk.max() * 1.05)         # zoom to the scaling region shown
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(loc="best", fontsize=9, title=f"cutoff at L ≥ {int(round(cutoff))}")
    if title:
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.suptitle(title, fontsize=14, y=0.965)
    else:
        fig.tight_layout()
    return fig


def convergence_figure(result, title=None):
    """Per-size crossing estimate p_c(L) vs L (linear axis), FULL range, Site and Bond in their own panel.
    The Intersection (comes in high) and Union (comes in low) estimators bracket p_c: the shaded band
    between them narrows as L grows -- the 'you can see it converge' view -- and its edges carry faint
    per-size dots. The reported p_c is a dashed horizontal line; a dashed vertical line marks the fit
    cutoff (points left of it, L < cutoff, are discarded from the extrapolation). `title`, if given, labels
    the tiling/graph at the top (matches fss_figure)."""
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    L = np.asarray(result["L"], dtype=float)
    BAND = "#a9cbe8"; EDGE = "#2c6e9c"; DOT = "#26507d"; PC = "#c0392b"; CUT = "#2e8b57"
    channels = [("Site percolation", result["raw_SI"], result["raw_SU"], result.get("site"))]
    if result.get("bond") is not None:
        channels.append(("Bond percolation", result["raw_BI"], result["raw_BU"], result.get("bond")))

    fig, axes = plt.subplots(1, len(channels), figsize=(6.5 * len(channels), 5.3), squeeze=False)
    for ax, (name, rI, rU, chres) in zip(axes[0], channels):
        mI, sI, mU, sU, mA, sA = [np.asarray(v) for v in _means_IUA(rI, rU)]
        ax.fill_between(L, mI, mU, color=BAND, alpha=0.6, lw=0, zorder=1)               # I-U bracket
        ax.plot(L, mI, color=EDGE, lw=1.2, zorder=3); ax.plot(L, mU, color=EDGE, lw=1.2, zorder=3)
        ax.plot(L, mI, "o", color=DOT, ms=1.7, alpha=0.5, zorder=4)
        ax.plot(L, mU, "o", color=DOT, ms=1.7, alpha=0.5, zorder=4)
        handles = [mpatches.Patch(color=BAND, alpha=0.7, label="Intersection–Union bracket")]
        if chres is not None:
            pc = chres["A"]["pc"]; cut = chres["A"].get("cutoff_L", float(L.min()))
            ax.axhline(pc, color=PC, ls=(0, (6, 4)), lw=1.5, zorder=2)
            ax.axvline(cut, color=CUT, ls=(0, (4, 3)), lw=1.5, zorder=2)                # fit cutoff
            handles += [Line2D([0], [0], color=PC, lw=1.5, ls="--", label=f"$p_c$ = {pc:.4f}"),
                        Line2D([0], [0], color=CUT, lw=1.5, ls="--", label=f"cutoff ($L = {int(round(cut))}$)")]
        ax.set_title(name, fontsize=12, pad=6)
        ax.set_xlabel("system size  $L$"); ax.set_ylabel("crossing  $p_c(L)$")
        ax.grid(axis="y", ls=":", alpha=0.35); ax.set_xlim(L.min() - 15, L.max() + 15)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.legend(handles=handles, loc="upper right", fontsize=9.5, framealpha=0.9)
    if title:
        fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.suptitle(title, fontsize=14, y=0.965)
    else:
        fig.tight_layout()
    return fig


def nu_omega_figure(result, controls=None, omega_lo=0.5, omega_hi=1.5, n_omega=13, B=100, seed=3, L_min=50.0):
    """nu (width-line) vs the ASSUMED correction exponent omega. omega is not measurable at accessible
    sizes, so nu is shown across the whole plausible band [omega_lo, omega_hi] rather than at one value:
    the message is that nu stays consistent with 4/3 for EVERY omega, so the result doesn't hinge on it.
    `controls` is an optional list of (label, result) for exact-nu=4/3 lattices (square/triangular) run
    identically -- the aperiodic run tracking them at every omega is the actual evidence. Shaded = 68%
    bootstrap CIs. Returns a Figure, or None if the run lacks the onset arrays."""
    from engine.analysis import _width_nu
    if result.get("raw_SI") is None:
        return None
    if (np.asarray(result["L"], float) >= L_min).sum() < 4:      # too few large sizes to fit a slope
        return None

    def channels(r):
        L_all = np.asarray(r["L"], float); keep = L_all >= L_min
        keys = [k for k in ("raw_SI", "raw_SU", "raw_BI", "raw_BU") if r.get(k) is not None]
        return L_all[keep], [[np.asarray(a, float) for a, kp in zip(r[k], keep) if kp] for k in keys]

    omegas = np.linspace(omega_lo, omega_hi, n_omega)

    def curve(r, sd):
        L, chs = channels(r)
        Ts = [len(ch[0]) for ch in chs]
        w0 = [np.array([a.std(ddof=1) for a in ch]) for ch in chs]
        sig = [w / np.sqrt(2.0 * (T - 1)) for w, T in zip(w0, Ts)]
        line = np.array([_width_nu(L, w0, sig, om) for om in omegas])
        rng = np.random.default_rng(sd)
        band = np.array([[_width_nu(L, [np.array([a[rng.integers(0, len(a), len(a))].std(ddof=1)
                                                  for a in ch]) for ch in chs], sig, om)
                          for om in omegas] for _ in range(B)])
        return line, np.percentile(band, 16, axis=0), np.percentile(band, 84, axis=0)

    NU = 4.0 / 3.0
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.axhline(NU, color="#111", lw=1.5, zorder=1)
    ax.text(omega_hi + 0.005, NU, r"  $\nu=4/3$", va="center", fontsize=11, fontweight="bold")
    for (lab, cr), col in zip(controls or [], ["#5a8f69", "#c08a3e", "#9467bd"]):
        Lc = np.asarray(cr.get("L", []), float)
        if cr.get("raw_SI") is None or (Lc >= L_min).sum() < 4:
            continue                       # control too thin at these sizes -> skip, don't crash curve()
        ln, lo, hi = curve(cr, seed + 1)
        ax.plot(omegas, ln, color=col, lw=1.6, ls="--", label=lab, zorder=2)
        ax.fill_between(omegas, lo, hi, color=col, alpha=0.12, zorder=1)
    ln, lo, hi = curve(result, seed)
    ax.plot(omegas, ln, color="#2e6f95", lw=2.6, label="this run", zorder=4)
    ax.fill_between(omegas, lo, hi, color="#2e6f95", alpha=0.22, zorder=3)
    ax.axvline(72.0 / 91.0, color="#999", lw=1, ls=":", zorder=1)
    ax.text(72.0 / 91.0 + 0.01, ax.get_ylim()[0], r" $\omega=\frac{72}{91}$ (2D)", fontsize=8,
            color="#777", va="bottom")
    ax.set_xlabel(r"assumed correction exponent  $\omega$")
    ax.set_ylabel(r"extracted correlation-length exponent  $\nu$")
    ax.set_xlim(omega_lo, omega_hi)
    ax.set_title(r"$\nu$ vs assumed $\omega$ — consistent with 4/3 across the whole band")
    ax.grid(alpha=0.15)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def cardy_figure(result):
    """Left-right crossing probability vs rectangle aspect ratio, measured (one series per window size,
    anchored at the square's 1/2-crossing) against Cardy's exact 2D-percolation curve. As the window
    grows the points converge onto the curve -- the conformal-invariance fingerprint of the class.
    `result` keys: aspects, windows, Rh (n_windows x n_aspects). Returns a Figure, or None if empty."""
    from engine.cardy import cardy_pi_h
    aspects = np.asarray(result["aspects"], float)
    Rh = np.atleast_2d(np.asarray(result["Rh"], float))
    windows = np.asarray(result["windows"], float)
    if aspects.size == 0 or Rh.size == 0:
        return None
    ac = np.linspace(aspects.min() * 0.85, aspects.max() * 1.1, 220)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(ac, [cardy_pi_h(a) for a in ac], "-", color="#222", lw=2,
            label="Cardy (exact 2D percolation)")
    n = max(1, len(windows) - 1)
    for i, L in enumerate(windows):
        ax.plot(aspects, Rh[i], "o", ms=6, color=plt.cm.viridis(i / n), label=f"L = {L:.0f}")
    ax.axhline(0.5, color="#bbb", ls=":", lw=1)
    ax.axvline(1.0, color="#bbb", ls=":", lw=1)
    ax.set_xlabel(r"aspect ratio  $a = W/H$")
    ax.set_ylabel(r"crossing probability  $R_h$")
    ax.set_title(r"Crossing probability vs Cardy's formula")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


