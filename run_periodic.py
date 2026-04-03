import argparse
import random
import datetime
import numpy as np
from periodic_tiling_generator import (square_lattice, triangular_lattice,
                                        boundary_sets)
from percolation import (percolationStatsI, percolationStatsU,
                         percolationStatsBondI, percolationStatsBondU,
                         extrapolate_pc_raw)
from results import PercolationResults
from visualisation import plot_all

KNOWN_PC = {
    'square':     {'site': 0.592746, 'bond': 0.500000},
    'triangular': {'site': 0.500000, 'bond': 0.347296},
}

LATTICE_FN = {
    'square':     square_lattice,
    'triangular': triangular_lattice,
}

def _verify(label, pc_hat, pc_std, pc_ci, known_pc):
    lo, hi = pc_ci
    inside = lo <= known_pc <= hi
    status = "PASS ✓" if inside else "FAIL ✗"
    print(f"  [{status}] Known p_c = {known_pc:.6f} | "
          f"Estimated = {pc_hat:.6f} ± {pc_std:.6f} | "
          f"95% CI = [{lo:.6f}, {hi:.6f}]")

def run_lattice(lattice_type, L_values, trials, seed, out_dir):
    print(f"\n{'='*60}\n  {lattice_type.upper()} LATTICE\n{'='*60}")
    raw_SI, raw_SU = [], []
    raw_BI, raw_BU = [], []
    valid_L = []

    for L in L_values:
        print(f"\n--- L={L} ---")
        nodes, neighbors, edges = LATTICE_FN[lattice_type](L)
        top, bottom, left, right = boundary_sets(L)

        statsSI = percolationStatsI(nodes, neighbors, top, bottom, left, right, trials)
        statsSU = percolationStatsU(nodes, neighbors, top, bottom, left, right, trials)
        statsBI = percolationStatsBondI(nodes, edges, top, bottom, left, right, trials)
        statsBU = percolationStatsBondU(nodes, edges, top, bottom, left, right, trials)

        raw_SI.append(statsSI.trialResults); raw_SU.append(statsSU.trialResults)
        raw_BI.append(statsBI.trialResults); raw_BU.append(statsBU.trialResults)
        valid_L.append(float(L))

        avg_site = 0.5 * (statsSI.trials_mean() + statsSU.trials_mean())
        avg_bond = 0.5 * (statsBI.trials_mean() + statsBU.trials_mean())
        print(f"  Site pc: I={statsSI.trials_mean():.4f}, U={statsSU.trials_mean():.4f}, Avg={avg_site:.4f}")
        print(f"  Bond pc: I={statsBI.trials_mean():.4f}, U={statsBU.trials_mean():.4f}, Avg={avg_bond:.4f}")

    known = KNOWN_PC[lattice_type]
    print(f"\n{'='*60}\nEXTRAPOLATION RESULTS — {lattice_type.capitalize()} Lattice\n{'='*60}")
    print("\n--- Site Percolation ---")
    site_results = extrapolate_pc_raw(valid_L, raw_SI, raw_SU)
    print("\n  Verification against analytical p_c:")
    for label in ('I', 'U', 'A'):
        r = site_results[label]
        print(f"  Estimator [{label}]:", end=" ")
        _verify(label, r['pc'], r['pc_std'], r['pc_ci'], known['site'])

    print("\n--- Bond Percolation ---")
    bond_results = extrapolate_pc_raw(valid_L, raw_BI, raw_BU)
    print("\n  Verification against analytical p_c:")
    for label in ('I', 'U', 'A'):
        r = bond_results[label]
        print(f"  Estimator [{label}]:", end=" ")
        _verify(label, r['pc'], r['pc_std'], r['pc_ci'], known['bond'])

    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    res = PercolationResults(
        tiling_type=lattice_type, seed=seed, trials=trials, L_values=valid_L,
        raw_SI=raw_SI, raw_SU=raw_SU, raw_BI=raw_BI, raw_BU=raw_BU,
        extra_meta={"lattice_type": lattice_type},
    )
    res.save(f"{out_dir}/results_{lattice_type}_{ts}.npz")
    plot_all(res)
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lattice', type=str,   default='square',
                        choices=['square', 'triangular'])
    # minimum patch size
    parser.add_argument('--Lmin',    type=int,   default=100)
    # maximum patch size
    parser.add_argument('--Lmax',    type=int,   default=500)
    # step size in between patches
    parser.add_argument('--Lstep',   type=int,   default=100)
    # number of trials per sample
    parser.add_argument('--t',       type=int,   default=20)
    # Use this to run both types
    parser.add_argument('--all',     action='store_true')
    # seed for reproducibility
    parser.add_argument('--seed',    type=int,   default=None)
    # directory to save results
    parser.add_argument('--out_dir', type=str,   default="results_output")
    args = parser.parse_args()

    import os; os.makedirs(args.out_dir, exist_ok=True)

    seed = args.seed if args.seed is not None else random.randrange(2**32)
    random.seed(seed); np.random.seed(seed)
    print(f"[periodic_main] RNG seed: {seed}")

    L_values = list(range(args.Lmin, args.Lmax + 1, args.Lstep))

    if args.all:
        for lattice_type in ('square', 'triangular'):
            run_lattice(lattice_type, L_values, args.t, seed, args.out_dir)
    else:
        run_lattice(args.lattice, L_values, args.t, seed, args.out_dir)