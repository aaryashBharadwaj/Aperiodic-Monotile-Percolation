"""Figure builders for the percolation GUI: the tiling-render showcase (visualise) and the two
results plots (finite-size-scaling extrapolation + per-size convergence). Split out of gui_backend so
that module stays focused on the numerics; these render Matplotlib Figures the UI displays.

    visualise(tiling, size, ...)      -> a rendered tiling Figure (+ optional graph overlay)
    fss_figure(result)                -> the I/U/A extrapolation plot (site & bond)
    convergence_figure(result)        -> per-size crossing estimate vs L (I & U)
    df_figure(result)                 -> log-log <S_max> vs L (fractal dimension d_f), if recorded

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
from visualiser.run_tiling_render import tiling_polygons, render_tiling, _threshold_class
from generators.family_geometry import tile_ab
from generators.periodic_tiling_generator import square_tiles, triangular_tris
from interface.gui_backend import (resolve_member, family_member, _penrose_tiling, _penrose_polys,
                                    S3, NU, TRI, FAMILY)


# ----------------------------------------------------------------------------- visualise engine
def _render_polys(tiling, size, a=1.0, b=S3):
    """(polys, title, color_by, penrose_obj) for the visualiser. penrose_obj is returned so a graph
    overlay can use the correct (rhombus-rule) Penrose builder."""
    name = resolve_member(tiling, a, b)
    if name in ("Hat", "Spectre", "Comet", "Chevron"):
        polys = tiling_polygons(name.lower(), level=size, ncells=size)
        color_by = "chirality" if name == "Hat" else "orientation"
        return polys, f"{name}  ({len(polys)} tiles)", color_by, None
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
        col = _threshold_class(a, b)[1]
        fig, ax = plt.subplots(figsize=(5, 5))
        render_tiling([np.asarray(tile_ab(a, b))], ax=ax, facecolor=col, lw=1.6)
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


def fss_figure(result):
    """Finite-size-scaling plot showing ALL THREE estimators -- Intersection, Union, Average -- vs
    L^(-1/nu), each with its data, its WLS fit line and its L->inf intercept. This is the CLI
    plot_extrapolation_IU view: watching I and U squeeze toward the SAME intercept as L->0^+ is the
    sanity check -- if the two don't converge, the crossing/extrapolation is untrustworthy. Site and
    bond get their own panel. Returns a Figure."""
    L = np.asarray(result["L"], dtype=float)
    x = L ** (-1.0 / NU)
    xs = np.linspace(0, x.max() * 1.02, 100)
    STYLES = [("Intersection", "I", "#1f77b4", "o"),
              ("Union",        "U", "#d62728", "s"),
              ("Average",      "A", "#2ca02c", "^")]
    channels = [("Site", result["raw_SI"], result["raw_SU"], result["site"])]
    if result.get("bond") is not None:
        channels.append(("Bond", result["raw_BI"], result["raw_BU"], result["bond"]))

    fig, axes = plt.subplots(1, len(channels), figsize=(6.2 * len(channels), 5), squeeze=False)
    for ax, (name, rI, rU, res) in zip(axes[0], channels):
        mI, sI, mU, sU, mA, sA = _means_IUA(rI, rU)
        series = {"I": (mI, sI), "U": (mU, sU), "A": (mA, sA)}
        for label, key, color, marker in STYLES:
            m, s = series[key]
            ax.errorbar(x, m, yerr=s, fmt=marker, color=color, ms=6, capsize=3, alpha=0.9)
            pc, slope = res[key]["pc"], res[key]["A"]
            ax.plot(xs, slope * xs + pc, "--", color=color, lw=1.4, label=f"{label}: {pc:.4f}")
            ax.plot(0, pc, "X", color=color, ms=12, mec="black", mew=0.7, zorder=5)
        ax.set_title(f"{name} percolation")
        ax.set_xlabel(r"$L^{-1/\nu}$   ($\nu=4/3$,  $L\to\infty$ at 0)")
        ax.set_ylabel(r"crossing $p_c$")
        ax.set_xlim(left=-0.004)
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(loc="best", fontsize=9, title="I & U should meet at 0")
    fig.tight_layout()
    return fig


def convergence_figure(result):
    """Per-size crossing estimate vs L (linear axis), Intersection and Union criteria for site and
    bond -- the CLI plot_percolation_stats_IU view. As L grows I (which comes in high) and U (which
    comes in low) squeeze toward each other onto the same p_c; that visible closing of the bracket is
    the 'you can see it converge' sanity check. The AVERAGE is deliberately NOT drawn here -- it's the
    midpoint of the two, so it looks flat and hides the convergence (that's the extrapolation plot's
    job). Error bars are the crossing-distribution sample std (the width that narrows as ~L^-3/4)."""
    L = np.asarray(result["L"], dtype=float)
    series = [("Site I", result["raw_SI"], "blue", "o"),
              ("Site U", result["raw_SU"], "deepskyblue", "s")]
    if result.get("bond") is not None:
        series += [("Bond I", result["raw_BI"], "red", "^"),
                   ("Bond U", result["raw_BU"], "orange", "x")]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, raw, color, marker in series:
        m = np.array([np.mean(r) for r in raw])
        w = np.array([np.std(r, ddof=1) for r in raw])   # distribution width, not standard error
        ax.errorbar(L, m, yerr=w, fmt=marker + "-", color=color, ms=5, lw=1.0, capsize=3, label=label)
    ax.set_title("Per-size estimate vs L  (I & U criteria)")
    ax.set_xlabel(r"linear system size  $L$")
    ax.set_ylabel(r"crossing $p_c(L)$")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


def df_figure(result):
    """Log-log <S_max> vs L: the incipient spanning cluster's mass grows as L^{d_f}, so the slope on
    log axes IS the fractal dimension. Shows the data, the fitted d_f line, and the exact 2D-percolation
    reference (91/48). Returns a Figure, or None if the run has no exponent (s_max) data."""
    e = result.get("exponents")
    if e is None or result.get("raw_smax") is None:
        return None
    L = np.asarray(result["L"], float)
    smax = [np.asarray(s, float) for s in result["raw_smax"]]
    mean = np.array([s.mean() for s in smax])
    se = np.array([s.std(ddof=1) / np.sqrt(len(s)) for s in smax])
    d_f, (lo, hi) = e["d_f"], e["d_f_ci"]
    logL = np.log(L); xl = np.array([L.min(), L.max()])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.errorbar(L, mean, yerr=se, fmt="o", color="#6a3d9a", ms=6, capsize=3, label=r"data  $\langle S_{\max}\rangle$")
    a = np.mean(np.log(mean) - d_f * logL)                           # anchor the fitted slope to the data
    ax.plot(xl, np.exp(a) * xl ** d_f, "-", color="#6a3d9a", lw=1.7,
            label=f"fit:  $d_f$ = {d_f:.3f}  [{lo:.3f}, {hi:.3f}]")
    DF2D = 91.0 / 48.0                                               # exact 2D percolation value
    a2 = np.log(mean[-1]) - DF2D * logL[-1]
    ax.plot(xl, np.exp(a2) * xl ** DF2D, "--", color="#888888", lw=1.4,
            label=f"2D percolation:  91/48 = {DF2D:.4f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"linear system size  $L$")
    ax.set_ylabel(r"incipient cluster size  $\langle S_{\max}\rangle$")
    ax.set_title(r"Fractal dimension:  $\langle S_{\max}\rangle \sim L^{d_f}$")
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


def df_convergence_figure(result, hmax=0.02):
    """Effective d_f vs L_min -- the assumption-free convergence test. Refit the slope of
    <S_max> ~ L^{d_f} over sizes L >= L_min for a sweep of cutoffs, WITHOUT imposing any corrections-to-
    scaling form. If the data belong to the 2D percolation class, the raw effective exponent drifts on
    its own toward 91/48 as the small-L points (which carry the finite-size corrections) are dropped;
    a fixed reference at 91/48 makes that visible. The cutoff is VALUE-BLIND: a point stays solid only
    while its 95% bootstrap CI half-width is under `hmax` -- fewer sizes remain as L_min rises, so the
    error grows and the volatile tail is excluded on the error budget, not on where the value sits.
    The least-biased usable point (largest L_min still under the ceiling) is annotated as the estimate.
    Returns a Figure, or None if the run has no s_max data / too few sizes."""
    smax = result.get("raw_smax")
    if smax is None or len(smax) < 6:
        return None
    from engine.analysis import fit_exponents
    L = np.asarray(result["L"], float)
    smax = [np.asarray(s, float) for s in smax]
    o = np.argsort(L); L = L[o]; smax = [smax[i] for i in o]
    Lmins = [float(m) for m in L if (L >= m).sum() >= 5]
    xs, ys, half = [], [], []
    for m in Lmins:
        keep = L >= m
        e = fit_exponents(L[keep], [s for s, k in zip(smax, keep) if k], B=300)
        xs.append(m); ys.append(e["d_f"]); half.append((e["d_f_ci"][1] - e["d_f_ci"][0]) / 2.0)
    xs, ys, half = np.array(xs), np.array(ys), np.array(half)
    usable = half <= hmax
    kmax = int(np.where(usable)[0].max()) if usable.any() else -1
    EX = 91.0 / 48.0
    C = "#1f5fa8"
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axhline(EX, ls="--", color="#222", lw=1.5, zorder=5, label=f"91/48 = {EX:.4f}  (2D percolation)")
    ax.plot(xs, ys, "-", color=C, lw=1, alpha=0.3)                          # full curve, faint
    if kmax >= 0:
        ax.fill_between(xs[:kmax + 1], ys[:kmax + 1] - half[:kmax + 1], ys[:kmax + 1] + half[:kmax + 1],
                        color=C, alpha=0.15)
        ax.plot(xs[:kmax + 1], ys[:kmax + 1], "-o", color=C, ms=4, lw=1.6,
                label=f"usable  (95% CI half-width ≤ {hmax:g})")
        ax.plot(xs[kmax + 1:], ys[kmax + 1:], "o", color=C, ms=3, mfc="white", alpha=0.5,
                label="excluded  (too few sizes → CI too wide)")
        ax.axvline(xs[kmax], color=C, ls=":", lw=1.1, alpha=0.7)
        ax.text(0.03, 0.05, f"estimate:  $d_f$ = {ys[kmax]:.3f} ± {half[kmax]:.3f}\n"
                            f"(largest usable $L_{{\\min}}$ = {xs[kmax]:g})",
                transform=ax.transAxes, va="bottom", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.9))
    else:
        ax.plot(xs, ys, "-o", color=C, ms=4)
    ax.set_ylim(min(1.70, float(ys.min()) - 0.02), 1.96)
    ax.set_xlabel(r"$L_{\min}$   (fit uses sizes $L \geq L_{\min}$ up to $L_{\max}$)")
    ax.set_ylabel(r"effective $d_f$  (slope of $\ln\langle S_{\max}\rangle$ vs $\ln L$)")
    ax.set_title(r"$d_f$ convergence — no corrections-to-scaling imposed")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
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


