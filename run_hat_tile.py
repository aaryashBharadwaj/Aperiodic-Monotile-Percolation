import argparse
import random
import datetime
import numpy as np
from hat_generator import H_init, T_init, P_init, F_init, constructPatch, constructMetatiles
from hat_graph_builder import build_neighbor_graph_fast, analyze_square_frame
from percolation import percolationStatsI, percolationStatsU, percolationStatsBondI, percolationStatsBondU, extrapolate_pc_raw
from results import PercolationResults
from visualisation import plot_all, plot_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Patch size. 5 is default
    # Changes may require recalibrating analyze_square_frame
    parser.add_argument('--r',     type=int,   default=5)
    # number of trials per sample
    parser.add_argument('--t',     type=int,   default=1000)
    # minimum patch size
    parser.add_argument('--Lmin',  type=float, default=10.0)
    # maximum patch size
    parser.add_argument('--Lmax',  type=float, default=400.0)
    # step between samples 
    parser.add_argument('--Lstep', type=float, default=10.0)
    # boundary handling
    parser.add_argument('--bt',    type=float, default=1.0)
    # Seed for reproducability. 
    parser.add_argument('--seed',  type=int,   default=123456789)
    # Directory that the results are placed into
    parser.add_argument('--out_dir', type=str, default="results_output")
    args = parser.parse_args()

    import os; os.makedirs(args.out_dir, exist_ok=True)

    # if no seed is provided, it chooses a random number
    seed = args.seed if args.seed is not None else random.randrange(2**32)
    random.seed(seed); np.random.seed(seed)
    print(f"[run_hat_tile] RNG seed: {seed}")

    print("Building Patch...")
    base = [H_init(), T_init(), P_init(), F_init()]
    cur = base
    for _ in range(args.r):
        p = constructPatch(*cur)
        cur = constructMetatiles(p)
    master_nodes, master_neighbors = build_neighbor_graph_fast(p, level=args.r + 1)

    l_values = np.arange(args.Lmin, args.Lmax + 1e-9, args.Lstep)
    raw_SI, raw_SU = [], []
    raw_BI, raw_BU = [], []
    valid_L = []
    last_fd = None

    # main loop
    for l in l_values:
        print(f"\nAnalyzing L={l}")
        fd = analyze_square_frame(master_nodes, master_neighbors, L=l, boundary_thickness=args.bt)

        if fd['node_count'] == 0:
            print("  No nodes, skipping.")
            continue

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
        last_fd = fd
        print(f"  Site pc Mean: {statsSI.trials_mean():.10f} | Bond pc Mean: {statsBI.trials_mean():.10f}")

    print(f"\n{'='*60}\nEXTRAPOLATION RESULTS\n{'='*60}")
    print("\n--- Site Percolation ---"); extrapolate_pc_raw(valid_L, raw_SI, raw_SU)
    print("\n--- Bond Percolation ---"); extrapolate_pc_raw(valid_L, raw_BI, raw_BU)

    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    res = PercolationResults(
        tiling_type="hat_vertex", seed=seed, trials=args.t, L_values=valid_L,
        raw_SI=raw_SI, raw_SU=raw_SU, raw_BI=raw_BI, raw_BU=raw_BU,
        extra_meta={"r": args.r, "bt": args.bt},
    )
    # saves the results
    res.save(f"{args.out_dir}/results_hat_vertex_{ts}.npz")

    # plots at the end
    plot_all(res)
    if last_fd is not None:
        plot_frames(valid_L, p, args.r, last_fd['center_x'], last_fd['center_y'])