
from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
from results import PercolationResults
from percolation import _wls_fit
from matplotlib.ticker import ScalarFormatter
import os


def _savefig(res, name, outdir="figures_generated"):
    """Save the current matplotlib figure. Lets headless/overnight (Agg) runs
    produce the plots as files even though plt.show() is a no-op there."""
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f"{res.tiling_type}_{name}.png"),
                dpi=150, bbox_inches="tight")


# plots the convergence of values with error bars
def plot_percolation_stats_IU(res: PercolationResults) -> None:
    # Error Bar Plot
    L = res.L_values
    mSI, sSI, mSU, sSU, mBI, sBI, mBU, sBU = res.means_stds()

    plt.figure(figsize=(12, 7))
    plt.errorbar(L, mSI, yerr=sSI, fmt="o-",  color="blue",   label="Site I")
    plt.errorbar(L, mSU, yerr=sSU, fmt="s-",  color="cyan",   label="Site U")
    if res.has_bond:
        plt.errorbar(L, mBI, yerr=sBI, fmt="^-", color="red",    label="Bond I")
        plt.errorbar(L, mBU, yerr=sBU, fmt="x-", color="orange", label="Bond U")

    plt.xlabel("Linear System Size ($L$)")
    plt.ylabel("Critical Probability ($p_c$)")
    plt.title(
        f"Site {'& Bond ' if res.has_bond else ''}Percolation (I & U Criteria)\n"
        f"{res.tiling_type}  |  seed={res.seed}  |  trials={res.trials}"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(res, "mean_ci")
    plt.show()


# Plots the finite-scaling of the percolation
def plot_extrapolation_IU(res: PercolationResults, exponent: float = -3 / 4, confidence: float = 0.95) -> None:
    from percolation import _wls_fit

    L = np.asarray(res.L_values, dtype=float)
    x = L ** exponent
    n = len(L)

    def compute_means_sigmas(raw_I, raw_U):
        means_I, means_U, means_A = [], [], []
        sigmas_I, sigmas_U, sigmas_A = [], [], []
        for rI, rU in zip(raw_I, raw_U):
            rI, rU = np.asarray(rI), np.asarray(rU)
            T = len(rI)
            mI, mU = rI.mean(), rU.mean()
            sI, sU = rI.std(ddof=1), rU.std(ddof=1)
            cov_IU = np.cov(rI, rU)[0, 1]
            means_I.append(mI); means_U.append(mU); means_A.append(0.5*(mI+mU))
            sigmas_I.append(sI / np.sqrt(T))
            sigmas_U.append(sU / np.sqrt(T))
            sigmas_A.append(0.5 * np.sqrt(sI**2/T + sU**2/T + 2*cov_IU/T))
        return (np.array(means_I), np.array(sigmas_I),
                np.array(means_U), np.array(sigmas_U),
                np.array(means_A), np.array(sigmas_A))

    configs = []
    mI, sI, mU, sU, mA, sA = compute_means_sigmas(res.raw_SI, res.raw_SU)
    configs.append({
        "title": f"Site Percolation Finite-Size Scaling — {res.tiling_type}",
        "data": [
            ("Intersection", mI, sI, "blue",  "o"),
            ("Union",        mU, sU, "red",   "s"),
            ("Average",      mA, sA, "green", "^"),
        ]
    })
    if res.has_bond:
        mI, sI, mU, sU, mA, sA = compute_means_sigmas(res.raw_BI, res.raw_BU)
        configs.append({
            "title": f"Bond Percolation Finite-Size Scaling — {res.tiling_type}",
            "data": [
                ("Intersection", mI, sI, "blue",  "o"),
                ("Union",        mU, sU, "red",   "s"),
                ("Average",      mA, sA, "green", "^"),
            ]
        })

    for cfg in configs:
        plt.figure(figsize=(12, 7), dpi=100)
        for label, y, sigma, color, marker in cfg["data"]:
            pc_hat, A_hat, _, _, _ = _wls_fit(x, y, sigma, n, confidence)

            plt.plot(x, y, marker, color=color, markersize=8, label=f"{label} Data")
            x_line = np.linspace(0, x.max(), 100)
            y_line = A_hat * x_line + pc_hat
            plt.plot(x_line, y_line, "--", color=color,
                     label=f"{label} Fit: $p_c(\\infty)$={pc_hat:.6f}")
            plt.plot(0, pc_hat, "x", color=color, markersize=10)

        plt.xlabel(f"$L^{{{exponent:.3g}}}$", fontsize=12)
        plt.ylabel("Mean Critical Probability ($\\bar{{p}}_c$)", fontsize=12)
        plt.title(cfg["title"], fontsize=14)
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xlim(left=-0.005)
        plt.tight_layout()
        _savefig(res, "fss_site" if "Site" in cfg["title"] else "fss_bond")
        plt.show()


# Hat-tiling frame visualiser
# Helps see what was actually generated!
def plot_frames(L_values, patch, iter, center_x, center_y) -> None:
    print("\nGenerating Frames visualisation...")
    fig = plt.figure(figsize=(16, 12), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    to_screen = [1, 0, 0, 0, 1, 0]

    try:
        patch.draw(to_screen, level=iter + 1, ax=ax)
    except Exception as e:
        print(f"An error occurred during drawing: {e}")

    for l_value in L_values:
        rect = plt.Rectangle(
            (center_x - l_value / 2, center_y - l_value / 2),
            l_value, l_value,
            fill=False, edgecolor="red", linewidth=1.5, linestyle="--",
        )
        ax.add_patch(rect)

    ax.set_aspect("equal", "box")
    ax.axis("off")
    plt.title(
        f"Hat Tiling (Level {iter}) with Frames Centered at ({center_x}, {center_y})",
        fontsize=16, pad=20,
    )
    plt.show()

def plot_hat_pc_dual_axis_final(res: PercolationResults, exponent: float = -3/4, confidence: float = 0.95):
    """
    Plots extrapolated pc(∞) for Site and Bond on dual independent y-axes.
    Forces full visibility of error bars and removes scientific notation offsets.
    """
    L = np.asarray(res.L_values, dtype=float)
    x = L ** exponent
    n = len(L)

    fig, ax_site = plt.subplots(figsize=(9, 7))
    ax_bond = ax_site.twinx()

    # --- 1. Site Extrapolation Calculation ---
    mean_A_s, sigma_A_s = [], []
    for rI, rU in zip(res.raw_SI, res.raw_SU):
        rI, rU = np.asarray(rI), np.asarray(rU)
        T = len(rI)
        # Combined Mean A = 0.5 * (I + U)
        mean_A_s.append(0.5 * (rI.mean() + rU.mean()))
        # Combined Sigma A accounting for covariance
        sI, sU = rI.std(ddof=1), rU.std(ddof=1)
        cov_IU = np.cov(rI, rU)[0, 1]
        sigma_A_s.append(0.5 * np.sqrt((sI**2 + sU**2 + 2 * cov_IU) / T))

    # WLS Fit to find the infinite-limit threshold pc_A_s
    pc_A_s, _, sigma_pc_s, *_ = _wls_fit(x, np.array(mean_A_s), np.array(sigma_A_s), n, confidence)
    err_s = 1.96 * sigma_pc_s  # 95% Confidence Interval

    # --- 2. Bond Extrapolation Calculation ---
    if res.has_bond:
        mean_A_b, sigma_A_b = [], []
        for rI, rU in zip(res.raw_BI, res.raw_BU):
            rI, rU = np.asarray(rI), np.asarray(rU)
            T = len(rI)
            mean_A_b.append(0.5 * (rI.mean() + rU.mean()))
            sI, sU = rI.std(ddof=1), rU.std(ddof=1)
            cov_IU = np.cov(rI, rU)[0, 1]
            sigma_A_b.append(0.5 * np.sqrt((sI**2 + sU**2 + 2 * cov_IU) / T))

        pc_A_b, _, sigma_pc_b, *_ = _wls_fit(x, np.array(mean_A_b), np.array(sigma_A_b), n, confidence)
        err_b = 1.96 * sigma_pc_b

    # --- 3. Plotting with Explicit Range Control ---
    ax_site.errorbar(0, pc_A_s, yerr=err_s, fmt="o", color="steelblue",
                     markersize=12, capsize=12, capthick=3, elinewidth=3, zorder=5)

    if res.has_bond:
        ax_bond.errorbar(1, pc_A_b, yerr=err_b, fmt="o", color="tomato",
                         markersize=12, capsize=12, capthick=3, elinewidth=3, zorder=5)

    # --- 4. Fixing Axis Display ---
    axes_data = [(ax_site, pc_A_s, err_s)]
    if res.has_bond:
        axes_data.append((ax_bond, pc_A_b, err_b))

    for ax, val, err in axes_data:
        ax.set_ylim(val - 2.0 * err, val + 2.0 * err)
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)

    # Styling and Labels
    ax_site.set_ylabel("Site $p_c(\\infty)$", color="steelblue", fontsize=14, fontweight='bold')
    ax_bond.set_ylabel("Bond $p_c(\\infty)$", color="tomato", fontsize=14, fontweight='bold')
    ax_site.tick_params(axis='y', labelcolor="steelblue", labelsize=12)
    ax_bond.tick_params(axis='y', labelcolor="tomato", labelsize=12)
    
    ax_site.set_xticks([0, 1])
    ax_site.set_xticklabels(["Site", "Bond"], fontsize=15)
    ax_site.set_xlim(-0.8, 1.8)
    ax_site.grid(True, axis='y', linestyle='--', alpha=0.4)
    
    plt.title(f"Extrapolated Thresholds: {res.tiling_type}", fontsize=14, pad=20)
    plt.tight_layout()
    _savefig(res, "threshold_dualaxis")
    plt.show()

def plot_all(res: PercolationResults) -> None:
    """Produce all standard plots for a completed run."""
    plot_percolation_stats_IU(res)
    plot_extrapolation_IU(res)
    plot_hat_pc_dual_axis_final(res)

 # Standalone entry-point for re-plotting from a saved .npz file
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-plot percolation results from a saved .npz file."
    )
    parser.add_argument(
        "--file", required=True,
        help="Path to a .npz results file produced by any runner."
    )
    parser.add_argument(
        "--exponent", type=float, default=-3 / 4,
        help="Finite-size scaling exponent (default: -0.75)."
    )
    args = parser.parse_args()

    print(f"Loading results from {args.file} …")
    res = PercolationResults.load(args.file)
    print(f"  tiling_type : {res.tiling_type}")
    print(f"  seed        : {res.seed}")
    print(f"  timestamp   : {res.timestamp}")
    print(f"  trials      : {res.trials}")
    print(f"  L values    : {res.L_values}")
    print(f"  has_bond    : {res.has_bond}")

    plot_all(res)