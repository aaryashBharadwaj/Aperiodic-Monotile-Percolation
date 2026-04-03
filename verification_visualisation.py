
from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from results import PercolationResults
from percolation import _wls_fit

# Colour palette — one per tiling type
COLOURS = {
    "penrose":    "mediumpurple",
    "square":     "steelblue",
    "triangular": "tomato",
}
KNOWN_COLOUR = "darkorange"

DISPLAY_NAMES = {
    "penrose":    "Penrose",
    "square":     "Square",
    "triangular": "Triangular",
}

EXPONENT = -3 / 4
CONFIDENCE = 0.95


def _extrapolate(raw_I, raw_U, L_values):
    L = np.asarray(L_values, dtype=float)
    x = L ** EXPONENT
    n = len(L)

    mean_I, mean_U, mean_A = [], [], []
    sigma_I, sigma_U, sigma_A = [], [], []

    for rI, rU in zip(raw_I, raw_U):
        rI, rU = np.asarray(rI), np.asarray(rU)
        T = len(rI)
        mI, mU = rI.mean(), rU.mean()
        sI, sU = rI.std(ddof=1), rU.std(ddof=1)
        cov_IU = np.cov(rI, rU)[0, 1]

        mean_I.append(mI); mean_U.append(mU); mean_A.append(0.5 * (mI + mU))
        sigma_I.append(sI / np.sqrt(T))
        sigma_U.append(sU / np.sqrt(T))
        sigma_A.append(0.5 * np.sqrt(sI**2/T + sU**2/T + 2*cov_IU/T))

    pc_I, *_ = _wls_fit(x, np.array(mean_I), np.array(sigma_I), n, CONFIDENCE)
    pc_U, *_ = _wls_fit(x, np.array(mean_U), np.array(sigma_U), n, CONFIDENCE)
    pc_A, *_ = _wls_fit(x, np.array(mean_A), np.array(sigma_A), n, CONFIDENCE)
    return pc_I, pc_A, pc_U


def _make_plot(results_list, known_pc_dict, ptype, raw_I_attr, raw_U_attr):
    # Filter to results that have data for this percolation type
    eligible = []
    for res in results_list:
        raw_I = getattr(res, raw_I_attr)
        raw_U = getattr(res, raw_U_attr)
        if raw_I is not None and raw_U is not None and len(raw_I) > 0:
            eligible.append(res)

    if not eligible:
        print(f"[verification_visualisation] No data for {ptype} percolation, skipping.")
        return

    n_rows = len(eligible)
    fig, ax = plt.subplots(figsize=(10, 2.5 + n_rows * 1.1))

    y_positions = np.arange(n_rows)
    y_labels = []
    legend_handles = []
    known_lines_drawn = set()

    for y_pos, res in zip(y_positions, eligible):
        ttype = res.tiling_type
        color = COLOURS.get(ttype, "grey")
        label = DISPLAY_NAMES.get(ttype, ttype)
        y_labels.append(label)

        raw_I = getattr(res, raw_I_attr)
        raw_U = getattr(res, raw_U_attr)
        pc_I, pc_A, pc_U = _extrapolate(raw_I, raw_U, res.L_values)

        err_lo = abs(pc_A - pc_I)
        err_hi = abs(pc_U - pc_A)

        eb = ax.errorbar(
            pc_A, y_pos,
            xerr=[[err_lo], [err_hi]],
            fmt="o",
            color=color,
            markersize=10,
            capsize=7,
            capthick=2.5,
            elinewidth=2.5,
            label=f"{label}: $p_c = {pc_A:.6f}$  [{pc_I:.6f}, {pc_U:.6f}]",
            zorder=3,
        )
        legend_handles.append(eb)

        # Known analytical value — vertical dashed line
        known = known_pc_dict.get(ttype, {})
        known_val = known.get(ptype.lower())
        if known_val is not None and ttype not in known_lines_drawn:
            # Annotate the line at this row's height so it's clear which is which
            ax.text(
                known_val, y_pos + 0.35,
                f"{known_val:.6f}",
                color=KNOWN_COLOUR,
                fontsize=8,
                ha="center",
                va="bottom",
            )
            known_lines_drawn.add(ttype)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=12)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.set_xlabel("Extrapolated $p_c(\\infty)$", fontsize=12)
    ax.set_title(
        f"{ptype} Percolation — Verification of Extrapolated $p_c(\\infty)$\n"
        f"I/U bounds  |  dashed line = analytical value",
        fontsize=13,
    )

    # Build a clean legend: estimated values + one entry for known line
    known_patch = mpatches.Patch(
        color=KNOWN_COLOUR, label="Known analytical $p_c$", alpha=0.85,
        linestyle="--", fill=False, linewidth=2,
    )
    ax.legend(
        handles=[eb for eb in legend_handles] + [known_patch],
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(0, -0.18),
        borderaxespad=0,
        ncol=1,
    )
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_verification(results_list: list[PercolationResults],
                      known_pc_dict: dict) -> None:
    _make_plot(results_list, known_pc_dict, "Site", "raw_SI", "raw_SU")
    _make_plot(results_list, known_pc_dict, "Bond", "raw_BI", "raw_BU")


# Standalone re-plot

KNOWN_PC_STANDALONE = {
    "penrose":    {"site": None,     "bond": None},
    "square":     {"site": 0.592746, "bond": 0.500000},
    "triangular": {"site": 0.500000, "bond": 0.347296},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-plot verification results from saved .npz files."
    )
    parser.add_argument(
        "--files", nargs="+", required=True,
        help="One or more .npz results files to include in the verification plot."
    )
    args = parser.parse_args()

    results = []
    for path in args.files:
        print(f"Loading {path} …")
        res = PercolationResults.load(path)
        print(f"  {res.tiling_type}  seed={res.seed}  L={res.L_values}")
        results.append(res)

    plot_verification(results, KNOWN_PC_STANDALONE)