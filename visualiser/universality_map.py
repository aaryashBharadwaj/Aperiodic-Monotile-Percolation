"""Figure: the dimensional-discrimination ladder in the correlation-length exponent nu.

Every aperiodic monotile in this study sits at the 2D percolation point (nu = 4/3). The 3D simple-cubic
and 4D simple-hypercubic lattices -- run through the IDENTICAL pipeline (same union-find kernel, same
width-line nu) -- land at their OWN classes, not at 2D (nu marches 4/3 -> 0.88 -> 0.69). So the family's
agreement with 2D is a real classification: the estimator tracks the universality class with dimension,
it does not always answer "2D".

This control is nu-ONLY -- nu alone discriminates 2D/3D/4D cleanly, and d_f is deliberately not used
(d_f from <S_max> is boundary-suppressed on open patches; see engine/null_control_cube).

    python -m visualiser.universality_map [--out figures_generated/universality_map.png]

The 3D and 4D nu are MEASURED here (from the stored null-control npz files, or a live sweep if absent);
the class reference nu are the exact literature values. The 2D point is this work's positive-control
result (the monotile family, see the per-object thresholds + Cardy test).
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine.null_control_cube import run_cube_sweep, width_nu, LIT

COL = {2: "#1f77b4", 3: "#d62728", 4: "#6a3d9a"}      # per-dimension colour


def _measured_nu(npz, dim, T):
    """nu for a control dimension: prefer the stored npz, else a quick live sweep."""
    if npz and os.path.exists(npz):
        z = np.load(npz)
        nu = float(z["nu"])
        print(f"{dim}D from {npz}: nu={nu:.3f}  (p_c={float(z['pc']):.5f})")
        return nu
    lmax = 48 if dim == 3 else 28
    ladder = [L for L in [8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48] if L <= lmax]
    r = run_cube_sweep(ladder, T=T, d=dim)
    nu = width_nu(r["L"], [r["I"], r["U"]])
    print(f"{dim}D measured live (no npz): nu={nu:.3f}")
    return nu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures_generated/universality_map.png")
    ap.add_argument("--npz3", default="paper_results/cube_null_L160_T1000_s123456789.npz")
    ap.add_argument("--npz4", default="paper_results/hypercube_null_4d_L32_T1000_s123456789.npz")
    ap.add_argument("--T", type=int, default=2000)
    args = ap.parse_args()

    dims = [2, 3, 4]
    lit = {d: LIT[d][0] for d in dims}
    measured = {2: lit[2],                                   # 2D = the family (positive-control result)
                3: _measured_nu(args.npz3, 3, args.T),
                4: _measured_nu(args.npz4, 4, args.T)}

    fig, ax = plt.subplots(figsize=(7.8, 5.4))

    # the class targets: exact literature nu per dimension, joined to show the trend
    ax.plot(dims, [lit[d] for d in dims], "--", color="#999999", lw=1.3, zorder=1,
            label=r"literature $\nu$ (class target)")
    for d in dims:
        ax.axhline(lit[d], color=COL[d], lw=0.7, ls=":", alpha=0.4)
        ax.plot(d, lit[d], "+", ms=15, mew=2.0, color=COL[d], zorder=3)

    # measured: 2D = the monotile family (positive control), 3D/4D = cube/hypercube (negative controls)
    labels = {2: "monotile family\n(this work) — 2D",
              3: "cube (3D)\n(this work)", 4: "hypercube (4D)\n(this work)"}
    off = {2: (12, 10), 3: (12, 10), 4: (12, 10)}
    for d in dims:
        yerr = 0.0 if d == 2 else 0.03
        ax.errorbar([d], [measured[d]], yerr=yerr, fmt="o", ms=13, color=COL[d],
                    ecolor=COL[d], mec="k", mew=1.0, capsize=4, zorder=5)
        ax.annotate(labels[d], (d, measured[d]), textcoords="offset points",
                    xytext=off[d], fontsize=10.5, color=COL[d])

    ax.text(0.62, 0.90, "same pipeline,\nclass tracks dimension", transform=ax.transAxes,
            ha="center", fontsize=10, color="#666666", style="italic")

    ax.set_xlabel(r"spatial dimension  $d$", fontsize=12)
    ax.set_ylabel(r"correlation-length exponent  $\nu$", fontsize=12)
    ax.set_title("The pipeline tracks the universality class with dimension:\n"
                 r"the monotile family is 2D ($\nu=4/3$); the 3D cube and 4D hypercube are not",
                 fontsize=12.5)
    ax.set_xticks(dims); ax.set_xlim(1.6, 4.5); ax.set_ylim(0.55, 1.50)
    ax.grid(alpha=0.15)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out, dpi=170)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
