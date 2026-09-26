"""Regenerate the two approved Cardy figures from the saved crossing data.

    python -m visualiser.cardy_figures            # writes both into manuscript_figures/approved/
    python -m visualiser.cardy_figures --out-dir some/dir

Plotting only -- reads paper_results/cardy_runs/*.npz and draws; no results logic, no simulation.
Both figures compare each object's measured crossing probability to Cardy's EXACT 2D curve:

  cardy_residuals.png  -- residual (measured - Cardy) per aspect ratio, with binomial error bars.
                          Square (green, 2D-lattice reference) + aperiodic monotiles (blue) sit on zero;
                          the cube/hypercube negatives (red) depart by many sigma.
  cardy_grouping.png   -- one bar per object: the RMS deviation from Cardy at the largest window, on a
                          LINEAR axis, with each run's noise floor sqrt(1/4T) as a black tick. The 2D
                          objects cluster; the non-2D controls sit ~7x further out.

Every run used here is T=24000, so all objects share the same noise floor (0.0032) and the comparison
is apples-to-apples. The noise floor is SHOWN (a tick), never divided out.
"""
import os
import sys
import time
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import matplotlib.lines as ml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from visualiser.paper_style import apply_paper_style

CR = os.path.join(REPO, "paper_results", "cardy_runs")
APPROVED = os.path.join(REPO, "paper_results", "manuscript_figures", "approved")

GREEN, BLUE, RED = "#006d2c", "#2c7fb8", "#b2182b"

# object -> npz file (all T=24000 so noise floors match). The four aperiodic-direct files are the
# original patch6 T=24000 runs; Square and the cube/hypercube negatives were re-run at T=24000.
SQUARE = "cardy_Square_direct_patch350_T24000.npz"
APERIODIC = [("Hat", "Hat_direct_patch6.npz", "o"),
             ("Chevron-ap", "Chevron_aperiodic_direct_patch6.npz", "^"),
             ("Spectre", "Spectre_direct_patch6.npz", "s"),
             ("Comet-ap", "Comet_aperiodic_direct_patch6.npz", "D")]
CUBE3D = "Cube3D_L48_T24000.npz"
CUBE4D = "Cube4D_L20_T24000.npz"


def _load(fn, win_last=True):
    """Return (aspects, residual = Rh - Cardy, binomial 1-sigma error) at the chosen window."""
    z = np.load(os.path.join(CR, fn), allow_pickle=True)
    Rh = np.asarray(z["Rh"][-1] if win_last else z["Rh"][0], float)
    cardy = np.asarray(z["cardy"], float)
    T = int(z["trials"])
    err = np.sqrt(Rh * (1.0 - Rh) / T)
    return np.asarray(z["aspects"], float), Rh - cardy, err


def _rms_and_floor(fn):
    """Absolute RMS deviation from Cardy at the largest window, and that run's noise floor sqrt(1/4T)."""
    z = np.load(os.path.join(CR, fn), allow_pickle=True)
    rms = float(np.sqrt(np.nanmean((z["Rh"][-1] - z["cardy"]) ** 2)))
    return rms, float(z["noise_floor"])


def _save(fig, out_path):
    """savefig to a temp file then atomically replace, retrying past the Windows/OneDrive file lock
    that intermittently throws OSError on the watched approved/ folder."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp%d.png" % os.getpid()   # keep a .png suffix so savefig infers the format
    fig.savefig(tmp, dpi=200, bbox_inches="tight")
    plt.close(fig)
    for _ in range(40):
        try:
            os.replace(tmp, out_path)
            print("saved ->", out_path)
            return
        except OSError:
            time.sleep(0.25)
    raise OSError(f"could not write {out_path} (locked)")


def residuals_figure(out_path):
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.axhline(0, color="0.15", lw=1.4, zorder=2)
    handles, labels = [], []

    def hdr(t):
        handles.append(ml.Line2D([], [], ls="none", marker="")); labels.append(t)

    hdr(r"$\bf{Reference\ 2D\ lattice:}$")
    a, res, err = _load(SQUARE, True)
    h = ax.errorbar(a, res, yerr=err, fmt="*", color=GREEN, ms=8.5, mec="k", mew=0.5,
                    elinewidth=0.9, capsize=1.8, ls="none", zorder=6)
    handles.append(h); labels.append("   Square")

    hdr(r"$\bf{Aperiodic\ monotiles:}$")
    for name, fn, mk in APERIODIC:
        if not os.path.exists(os.path.join(CR, fn)):
            continue
        a, res, err = _load(fn, True)
        h = ax.errorbar(a, res, yerr=err, fmt=mk, color=BLUE, ms=4.4, mec=BLUE, alpha=0.85,
                        elinewidth=0.8, capsize=1.6, ls="none", zorder=4)
        handles.append(h); labels.append(f"   {name}")

    hdr(r"$\bf{Negative\ controls:}$")
    for fn, name, col, mk in [(CUBE3D, "cube (3D)", "#d62728", "s"),
                              (CUBE4D, "hypercube (4D)", "#8b1a1a", "^")]:
        a, res, err = _load(fn, False)
        h = ax.errorbar(a, res, yerr=err, fmt=mk, color=col, ms=6.5, mec="k", mew=0.6,
                        elinewidth=0.9, capsize=2, ls="none", zorder=5)
        handles.append(h); labels.append(f"   {name}")

    ax.axvline(1, color="0.8", ls=":", lw=0.8)
    ax.set_xlabel(r"aspect ratio  $a=W/H$")
    ax.set_ylabel(r"residual  $\Pi_h^{\rm meas}-\Pi_h^{\rm Cardy}$")
    ax.set_title("Residuals of the aperiodic tilings and controls against Cardy's formula")
    ax.set_xlim(0.5, 2.0)
    ax.legend(handles, labels, fontsize=8.5, loc="upper right", ncol=1, framealpha=0.95)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    _save(fig, out_path)


def grouping_figure(out_path):
    rows = [("Square", SQUARE, GREEN),
            ("Hat", "Hat_direct_patch6.npz", BLUE),
            ("Chevron-ap", "Chevron_aperiodic_direct_patch6.npz", BLUE),
            ("Spectre", "Spectre_direct_patch6.npz", BLUE),
            ("Comet-ap", "Comet_aperiodic_direct_patch6.npz", BLUE),
            ("Cube 3D", CUBE3D, RED),
            ("Cube 4D", CUBE4D, RED)]
    vals, floors = [], []
    for _, fn, _c in rows:
        v, fl = _rms_and_floor(fn)
        vals.append(v); floors.append(fl)
    names = [n for n, _, _ in rows]
    cols = [c for _, _, c in rows]
    y = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    # The black tick on each bar is THAT object's noise floor (sqrt(1/4T)); a bar at its tick is
    # indistinguishable from Cardy, a bar well past it has a real deviation. No shaded band -- with a
    # single band the per-object floors (all 0.0032 at T=24000) would be misrepresented.
    for yi, v, fl, c in zip(y, vals, floors, cols):
        ax.barh(yi, v, color=c, alpha=0.92, height=0.62, zorder=3)
        ax.plot([fl, fl], [yi - 0.31, yi + 0.31], color="0.1", lw=1.4, zorder=4)
        ax.text(v + 0.0015, yi, f"{v:.4f}", va="center", ha="left", fontsize=8.4)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10)
    ax.set_xlim(0, 0.11)
    ax.set_xlabel("RMS deviation from Cardy  (crossing-probability units)")
    ax.set_title("Average distance from the Cardy crossing curve", fontsize=11.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [mp.Patch(color=GREEN, label="positive control (2D lattice)"),
               mp.Patch(color=BLUE, label="aperiodic monotiles"),
               mp.Patch(color=RED, label="negative control (non-2D)"),
               ml.Line2D([0], [0], color="0.1", lw=1.4, label="noise floor (this run)")]
    ax.legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.96)
    _save(fig, out_path)


def main():
    ap = argparse.ArgumentParser(description="Regenerate the two approved Cardy figures.")
    ap.add_argument("--out-dir", default=APPROVED, help="where to write (default: manuscript_figures/approved)")
    args = ap.parse_args()
    apply_paper_style()
    residuals_figure(os.path.join(args.out_dir, "cardy_residuals.png"))
    grouping_figure(os.path.join(args.out_dir, "cardy_grouping.png"))


if __name__ == "__main__":
    main()
