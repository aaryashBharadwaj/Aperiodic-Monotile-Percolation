import argparse
import random
import datetime
import numpy as np
from penrose_tiling_generator import PenroseTiling
from penrose_graph_builder import build_penrose_neighbor_graph, analyze_square_frame
from percolation import (percolationStatsI, percolationStatsU,
                         percolationStatsBondI, percolationStatsBondU,
                         extrapolate_pc_raw)
from results import PercolationResults
from visualisation import plot_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Patch size
    parser.add_argument('--s',       type=int,   default=8)
    # Trials per sample
    parser.add_argument('--t',       type=int,   default=600)
    # Minimum patch size
    parser.add_argument('--Lmin',    type=float, default=200.0)
    # Maximum Patch size
    parser.add_argument('--Lmax',    type=float, default=500.0)
    # PStep between patch sizes
    parser.add_argument('--Lstep',   type=float, default=10.0)
    # Boundary
    parser.add_argument('--bt',      type=float, default=10.0)
    # Seed for reproducibility
    parser.add_argument('--seed',    type=int,   default=None)
    # Directory where results are stored
    parser.add_argument('--out_dir', type=str,   default="results_output")
    args = parser.parse_args()

    import os; os.makedirs(args.out_dir, exist_ok=True)

    seed = args.seed if args.seed is not None else random.randrange(2**32)
    random.seed(seed); np.random.seed(seed)
    print(f"[penrose_main] RNG seed: {seed}")

    scale = int(args.Lmax * 2)
    print("Building Patch...")
    tiling = PenroseTiling(divisions=args.s, base=5, scale=scale)
    tiling.make_tiling()
    print(f"  Generated {len(tiling.triangles)} triangles")

    print("\nBuilding neighbor graph...")
    master_nodes, master_neighbors, master_edges = build_penrose_neighbor_graph(tiling)

    l_values = np.arange(args.Lmin, args.Lmax + 1e-9, args.Lstep)
    raw_SI, raw_SU = [], []
    raw_BI, raw_BU = [], []
    valid_L = []
    last_fd = None

    # main loop
    for l in l_values:
        print(f"\nAnalyzing L={l}")
        fd = analyze_square_frame(master_nodes, master_neighbors, master_edges, l, args.bt)

        if fd['node_count'] == 0:
            print("  No nodes in frame, skipping.")
            continue

        last_fd = fd
        print(f"  Nodes: {fd['node_count']}  |  Edges: {fd['edge_count']}")
        print(f"  Boundaries — top: {len(fd['top_boundary_nodes'])}, "
              f"bottom: {len(fd['bottom_boundary_nodes'])}, "
              f"left: {len(fd['left_boundary_nodes'])}, "
              f"right: {len(fd['right_boundary_nodes'])}")

        tb = set(fd['top_boundary_nodes']) & set(fd['bottom_boundary_nodes'])
        lr = set(fd['left_boundary_nodes']) & set(fd['right_boundary_nodes'])
        if tb or lr:
            print(f"  WARNING: Boundary overlap! TB={len(tb)}, LR={len(lr)}")

        statsSI = percolationStatsI(fd['sub_graph_nodes'], fd['sub_graph_neighbors'],
                                    fd['top_boundary_nodes'], fd['bottom_boundary_nodes'],
                                    fd['left_boundary_nodes'], fd['right_boundary_nodes'], args.t)
        statsSU = percolationStatsU(fd['sub_graph_nodes'], fd['sub_graph_neighbors'],
                                    fd['top_boundary_nodes'], fd['bottom_boundary_nodes'],
                                    fd['left_boundary_nodes'], fd['right_boundary_nodes'], args.t)
        statsBI = percolationStatsBondI(fd['sub_graph_nodes'], fd['sub_graph_edges'],
                                        fd['top_boundary_nodes'], fd['bottom_boundary_nodes'],
                                        fd['left_boundary_nodes'], fd['right_boundary_nodes'], args.t)
        statsBU = percolationStatsBondU(fd['sub_graph_nodes'], fd['sub_graph_edges'],
                                        fd['top_boundary_nodes'], fd['bottom_boundary_nodes'],
                                        fd['left_boundary_nodes'], fd['right_boundary_nodes'], args.t)

        raw_SI.append(statsSI.trialResults); raw_SU.append(statsSU.trialResults)
        raw_BI.append(statsBI.trialResults); raw_BU.append(statsBU.trialResults)
        valid_L.append(l)
        print(f"  Site pc Mean: {statsSI.trials_mean():.10f} | Bond pc Mean: {statsBI.trials_mean():.10f}")

    print(f"\n{'='*60}\nEXTRAPOLATION RESULTS — Penrose Tiling\n{'='*60}")
    print("\n--- Site Percolation ---"); extrapolate_pc_raw(valid_L, raw_SI, raw_SU)
    print("\n--- Bond Percolation ---"); extrapolate_pc_raw(valid_L, raw_BI, raw_BU)

    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    res = PercolationResults(
        tiling_type="penrose", seed=seed, trials=args.t, L_values=valid_L,
        raw_SI=raw_SI, raw_SU=raw_SU, raw_BI=raw_BI, raw_BU=raw_BU,
        extra_meta={"s": args.s, "bt": args.bt},
    )
    # save and plot
    res.save(f"{args.out_dir}/results_penrose_{ts}.npz")
    plot_all(res)