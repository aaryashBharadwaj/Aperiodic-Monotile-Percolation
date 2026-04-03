# this program simply runs periodic and penrose

import argparse
import datetime
import random
import numpy as np
import os

from penrose_tiling_generator import PenroseTiling
from penrose_graph_builder import build_penrose_neighbor_graph, analyze_square_frame as penrose_analyze_square_frame
from periodic_tiling_generator import (square_lattice, triangular_lattice,
                                        boundary_sets)
from percolation import (percolationStatsI, percolationStatsU,
                         percolationStatsBondI, percolationStatsBondU)
from results import PercolationResults
from visualisation import plot_all
from verification_visualisation import plot_verification

KNOWN_PC = {
    "penrose":    {"site": None,     "bond": None},
    "square":     {"site": 0.592746, "bond": 0.500000},
    "triangular": {"site": 0.500000, "bond": 0.347296},
}

LATTICE_FN = {
    "square":     square_lattice,
    "triangular": triangular_lattice,
}

# runs Penrose
def _run_penrose(args, seed, out_dir):
    print("\n" + "="*60 + "\n  PENROSE\n" + "="*60)
    scale = int(args.penrose_Lmax * 2)
    tiling = PenroseTiling(divisions=args.penrose_s, base=5, scale=scale)
    tiling.make_tiling()
    print(f"  {len(tiling.triangles)} triangles")

    master_nodes, master_neighbors, master_edges = build_penrose_neighbor_graph(tiling)

    l_values = np.arange(args.penrose_Lmin, args.penrose_Lmax + 1e-9, args.penrose_Lstep)
    raw_SI, raw_SU, raw_BI, raw_BU, valid_L = [], [], [], [], []

    for l in l_values:
        print(f"\n  L={l}")
        fd = penrose_analyze_square_frame(master_nodes, master_neighbors, master_edges, l, args.bt)
        if fd["node_count"] == 0:
            continue

        statsSI = percolationStatsI(fd["sub_graph_nodes"], fd["sub_graph_neighbors"],
                                    fd["top_boundary_nodes"], fd["bottom_boundary_nodes"],
                                    fd["left_boundary_nodes"], fd["right_boundary_nodes"], args.t)
        statsSU = percolationStatsU(fd["sub_graph_nodes"], fd["sub_graph_neighbors"],
                                    fd["top_boundary_nodes"], fd["bottom_boundary_nodes"],
                                    fd["left_boundary_nodes"], fd["right_boundary_nodes"], args.t)
        statsBI = percolationStatsBondI(fd["sub_graph_nodes"], fd["sub_graph_edges"],
                                        fd["top_boundary_nodes"], fd["bottom_boundary_nodes"],
                                        fd["left_boundary_nodes"], fd["right_boundary_nodes"], args.t)
        statsBU = percolationStatsBondU(fd["sub_graph_nodes"], fd["sub_graph_edges"],
                                        fd["top_boundary_nodes"], fd["bottom_boundary_nodes"],
                                        fd["left_boundary_nodes"], fd["right_boundary_nodes"], args.t)
        raw_SI.append(statsSI.trialResults); raw_SU.append(statsSU.trialResults)
        raw_BI.append(statsBI.trialResults); raw_BU.append(statsBU.trialResults)
        valid_L.append(l)
        print(f"  Site pc: {statsSI.trials_mean():.6f} | Bond pc: {statsBI.trials_mean():.6f}")

    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    res = PercolationResults(
        tiling_type="penrose", seed=seed, trials=args.t, L_values=valid_L,
        raw_SI=raw_SI, raw_SU=raw_SU, raw_BI=raw_BI, raw_BU=raw_BU,
        extra_meta={"s": args.penrose_s, "bt": args.bt},
    )
    res.save(f"{out_dir}/results_penrose_{ts}.npz")
    plot_all(res)
    return res

# runs Periodic 
def _run_periodic(lattice_type, args, seed, out_dir):
    print(f"\n{'='*60}\n  {lattice_type.upper()}\n{'='*60}")
    L_values = list(range(args.periodic_Lmin, args.periodic_Lmax + 1, args.periodic_Lstep))
    raw_SI, raw_SU, raw_BI, raw_BU, valid_L = [], [], [], [], []

    for L in L_values:
        print(f"\n  L={L}")
        nodes, neighbors, edges = LATTICE_FN[lattice_type](L)
        top, bottom, left, right = boundary_sets(L)

        statsSI = percolationStatsI(nodes, neighbors, top, bottom, left, right, args.t)
        statsSU = percolationStatsU(nodes, neighbors, top, bottom, left, right, args.t)
        statsBI = percolationStatsBondI(nodes, edges, top, bottom, left, right, args.t)
        statsBU = percolationStatsBondU(nodes, edges, top, bottom, left, right, args.t)
        raw_SI.append(statsSI.trialResults); raw_SU.append(statsSU.trialResults)
        raw_BI.append(statsBI.trialResults); raw_BU.append(statsBU.trialResults)
        valid_L.append(float(L))
        print(f"  Site pc: {statsSI.trials_mean():.4f} | Bond pc: {statsBI.trials_mean():.4f}")

    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    res = PercolationResults(
        tiling_type=lattice_type, seed=seed, trials=args.t, L_values=valid_L,
        raw_SI=raw_SI, raw_SU=raw_SU, raw_BI=raw_BI, raw_BU=raw_BU,
        extra_meta={"lattice_type": lattice_type},
    )
    res.save(f"{out_dir}/results_{lattice_type}_{ts}.npz")
    plot_all(res)
    return res

# all the inputs from Penrose and periodic 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t",              type=int,   default=200)
    parser.add_argument("--seed",           type=int,   default=None)
    parser.add_argument("--bt",             type=float, default=10.0)
    parser.add_argument("--out_dir",        type=str,   default="results_output")
    parser.add_argument("--penrose_s",      type=int,   default=8)
    parser.add_argument("--penrose_Lmin",   type=float, default=200.0)
    parser.add_argument("--penrose_Lmax",   type=float, default=500.0)
    parser.add_argument("--penrose_Lstep",  type=float, default=50.0)
    parser.add_argument("--periodic_Lmin",  type=int,   default=100)
    parser.add_argument("--periodic_Lmax",  type=int,   default=500)
    parser.add_argument("--periodic_Lstep", type=int,   default=100)
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randrange(2**32)
    random.seed(seed); np.random.seed(seed)
    print(f"[verification_main] RNG seed: {seed}")

    os.makedirs(args.out_dir, exist_ok=True)

    all_results = []
    all_results.append(_run_penrose(args, seed, args.out_dir))
    for lt in ("square", "triangular"):
        all_results.append(_run_periodic(lt, args, seed, args.out_dir))

    plot_verification(all_results, KNOWN_PC)