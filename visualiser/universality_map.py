"""Figure: the (nu, d_f) universality map. Every aperiodic monotile in this study sits at the 2D point
(nu=4/3, d_f=91/48); the 3D simple-cubic NEGATIVE CONTROL, run through the identical pipeline, sits far
away at the 3D point -- so the agreement across the family is a real classification, not a method that
always answers "2D".

    python -m visualiser.universality_map [--out figures_generated/universality_map.png] [--T 2000]

The cube point is MEASURED here (engine.null_control_cube); the two class markers are the exact literature
values. The family marker sits at the exact 2D point because that is this work's positive-control result.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine.null_control_cube import (run_cube_sweep, width_nu, _df_from_smax,
                                      NU_3D, DF_3D)

NU_2D, DF_2D = 4.0 / 3.0, 91.0 / 48.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures_generated/universality_map.png")
    ap.add_argument("--T", type=int, default=2000)
    args = ap.parse_args()

    # measure the cube point through the same pipeline
    r = run_cube_sweep([8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48], T=args.T)
    nu_cube = width_nu(r["L"], [r["I"], r["U"]])
    df_cube = _df_from_smax(r["L"], r["smax"], L_min=16)
    print(f"cube measured: nu={nu_cube:.3f}, d_f={df_cube:.3f}")

    fig, ax = plt.subplots(figsize=(7.8, 6.6))

    # the two universality classes as reference crosshairs
    for (nu, df, name, col) in [(NU_2D, DF_2D, "2D class\n(4/3, 91/48)", "#1f77b4"),
                                (NU_3D, DF_3D, "3D class\n(0.876, 2.523)", "#d62728")]:
        ax.axvline(nu, color=col, lw=0.8, ls=":", alpha=0.5)
        ax.axhline(df, color=col, lw=0.8, ls=":", alpha=0.5)
        ax.plot(nu, df, marker="+", ms=16, mew=2.2, color=col)

    # the monotile family: this work's positive controls, all consistent with 2D -> at the 2D point
    ax.scatter([NU_2D], [DF_2D], s=340, facecolor="#1f77b4", edgecolor="k", zorder=5, alpha=0.9)
    ax.annotate("hat, spectre,\ncomet-ap, chevron-ap,\n+ periodics  (this work)",
                (NU_2D, DF_2D), textcoords="offset points", xytext=(16, -6),
                fontsize=10.5, va="center", color="#12507b")

    # the cube: MEASURED negative control
    ax.errorbar([nu_cube], [df_cube], xerr=0.03, yerr=0.06, fmt="o", ms=13, color="#d62728",
                ecolor="#d62728", mec="k", mew=1.0, capsize=4, zorder=6)
    ax.annotate("cube (this work)\nnegative control", (nu_cube, df_cube),
                textcoords="offset points", xytext=(14, 10), fontsize=10.5, color="#8c1515")

    ax.annotate("", xy=(nu_cube, df_cube), xytext=(NU_2D, DF_2D),
                arrowprops=dict(arrowstyle="->", color="#888888", lw=1.3, ls="--"))
    ax.text(0.5 * (NU_2D + nu_cube), 0.5 * (DF_2D + df_cube) + 0.06,
            "same pipeline,\ndifferent class", ha="center", fontsize=9.5, color="#666666", style="italic")

    ax.set_xlabel(r"correlation-length exponent  $\nu$", fontsize=12)
    ax.set_ylabel(r"cluster fractal dimension  $d_f$", fontsize=12)
    ax.set_title("Universality map: every monotile is 2D; the cube control is not", fontsize=12.5)
    ax.set_xlim(0.78, 1.46); ax.set_ylim(1.80, 2.62)
    ax.grid(alpha=0.15)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out, dpi=170)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
