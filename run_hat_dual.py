import argparse
import random
import datetime
import numpy as np
from hat_generator import H_init, T_init, P_init, F_init, constructPatch, constructMetatiles
from hat_dual_graph_builder import build_tile_graph, analyze_tile_square_frame
from percolation import percolationStatsI, percolationStatsU, percolationStatsBondI, percolationStatsBondU, extrapolate_pc_raw
from results import PercolationResults
from visualisation import plot_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Patch size. 5 is default
    # Changes may require recalibrating analyze_square_frame
    parser.add_argument('--r',       type=int,   default=5)
    # number of trials per sample
    parser.add_argument('--t',       type=int,   default=1000)
    # minimum patch size
    parser.add_argument('--Lmin',    type=float, default=10.0)
    # maximum patch size
    parser.add_argument('--Lmax',    type=float, default=400.0)
    # step between samples 
    parser.add_argument('--Lstep',   type=float, default=10.0)
    # boundary handling
    parser.add_argument('--bt',      type=float, default=1.0)
    # Seed for reproducibility
    parser.add_argument('--seed',    type=int,   default=123456789)
    # Directory the results are placed into 
    parser.add_argument('--out_dir', type=str,   default="results_output")
    args = parser.parse_args()

    import os; os.makedirs(args.out_dir, exist_ok=True)

    seed = args.seed if args.seed is not None else random.randrange(2**32)
    random.seed(seed); np.random.seed(seed)
    print(f"[run_hat_dual] RNG seed: {seed}")

    print("Building Patch...")
    base = [H_init(), T_init(), P_init(), F_init()]
    cur = base
    for _ in range(args.r):
        p = constructPatch(*cur)
        cur = constructMetatiles(p)

    print("Building tile graph...")
    tile_centroids, tile_neighbors, _ = build_tile_graph(p, level=args.r + 1)

    l_values = np.arange(args.Lmin, args.Lmax + 1e-9, args.Lstep)
    raw_SI, raw_SU = [], []
    raw_BI, raw_BU = [], []
    valid_L = []

    # main loop
    for l in l_values:
        print(f"\nAnalyzing L={l}")
        fd = analyze_tile_square_frame(tile_centroids, tile_neighbors, L=l, boundary_thickness=args.bt)

        if fd['node_count'] == 0:
            print(f"  No tiles found for L={l}, skipping.")
            continue

        print(f"  Tiles in frame: {fd['node_count']}")
        print(f"  Top: {len(fd['top_boundary_nodes'])}, Bottom: {len(fd['bottom_boundary_nodes'])}, "
              f"Left: {len(fd['left_boundary_nodes'])}, Right: {len(fd['right_boundary_nodes'])}")

        statsSI = percolationStatsI(
            fd['sub_graph_nodes'], fd['sub_graph_neighbors'],
            fd['top_boundary_nodes'], fd['bottom_boundary_nodes'],
            fd['left_boundary_nodes'], fd['right_boundary_nodes'],
            args.t
        )
        statsSU = percolationStatsU(
            fd['sub_graph_nodes'], fd['sub_graph_neighbors'],
            fd['top_boundary_nodes'], fd['bottom_boundary_nodes'],
            fd['left_boundary_nodes'], fd['right_boundary_nodes'],
            args.t
        )
        statsBI = percolationStatsBondI(
            fd['sub_graph_nodes'], fd['sub_graph_edges'],
            fd['top_boundary_nodes'], fd['bottom_boundary_nodes'],
            fd['left_boundary_nodes'], fd['right_boundary_nodes'],
            args.t
        )
        statsBU = percolationStatsBondU(
            fd['sub_graph_nodes'], fd['sub_graph_edges'],
            fd['top_boundary_nodes'], fd['bottom_boundary_nodes'],
            fd['left_boundary_nodes'], fd['right_boundary_nodes'],
            args.t
        )

        raw_SI.append(statsSI.trialResults)
        raw_SU.append(statsSU.trialResults)
        raw_BI.append(statsBI.trialResults)
        raw_BU.append(statsBU.trialResults)
        valid_L.append(l)

        print(f"  Tile site pc (I): {statsSI.trials_mean():.6f} ± {statsSI.trials_std():.6f}")
        print(f"  Tile site pc (U): {statsSU.trials_mean():.6f} ± {statsSU.trials_std():.6f}")
        print(f"  Tile bond pc (I): {statsBI.trials_mean():.6f} ± {statsBI.trials_std():.6f}")
        print(f"  Tile bond pc (U): {statsBU.trials_mean():.6f} ± {statsBU.trials_std():.6f}")

    print(f"\n{'='*60}\nEXTRAPOLATION RESULTS — Hat Tiling (Tile Graph)\n{'='*60}")
    print("\n--- Tile Site Percolation ---")
    extrapolate_pc_raw(valid_L, raw_SI, raw_SU)
    print("\n--- Tile Bond Percolation ---")
    extrapolate_pc_raw(valid_L, raw_BI, raw_BU)

    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    res = PercolationResults(
        tiling_type="hat_tile", seed=seed, trials=args.t, L_values=valid_L,
        raw_SI=raw_SI, raw_SU=raw_SU, raw_BI=raw_BI, raw_BU=raw_BU,
        extra_meta={"r": args.r, "bt": args.bt},
    )
    res.save(f"{args.out_dir}/results_hat_tile_{ts}.npz")
    # plotting 
    plot_all(res)