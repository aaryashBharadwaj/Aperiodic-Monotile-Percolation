import argparse
import numpy as np
import tqdm as tqdm
from hat_tiling import H_init, T_init, P_init, F_init, constructPatch, constructMetatiles
from graph_builder import build_neighbor_graph_fast, analyze_square_frame
from percolation import percolationStatsI, percolationStatsU, percolationStatsBondI, percolationStatsBondU
from visualisation import plot_frames, plot_percolation_stats_IU, plot_extrapolation_IU

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # patch size
    parser.add_argument('--r', type=int, default=5)
    # number of trials per lattice size
    parser.add_argument('--t', type=int, default=1000)
    # minimum lattice size
    parser.add_argument('--Lmin', type=float, default=10.0)
    # maximum lattice size
    parser.add_argument('--Lmax', type=float, default=400.0)
    # step size from lattice sizes
    parser.add_argument('--Lstep', type=float, default=10.0)
    # boundary thickness
    parser.add_argument('--bt', type=float, default=1.0)
    args = parser.parse_args()

    print("Building Patch...")
    base = [H_init(), T_init(), P_init(), F_init()]
    cur = base
    for _ in range(args.r):
        p = constructPatch(*cur)
        cur = constructMetatiles(p)
    master_nodes, master_neighbors = build_neighbor_graph_fast(p, level=args.r+1)

    l_values = np.arange(args.Lmin, args.Lmax + 1e-9, args.Lstep)
    mSI, sSI, mSU, sSU = [], [], [], []
    mBI, sBI, mBU, sBU = [], [], [], []

    last_fd = None 

    # primary execution loop
    for l in l_values:
        print(f"\nAnalyzing L={l}")
        fd = analyze_square_frame(master_nodes, master_neighbors, L=l, boundary_thickness = 1)        
        if fd['node_count'] > 0:
            top_set = set(fd['top_boundary_nodes'])
            bot_set = set(fd['bottom_boundary_nodes'])
            left_set = set(fd['left_boundary_nodes'])
            right_set = set(fd['right_boundary_nodes'])
            
            tb_overlap = top_set & bot_set
            lr_overlap = left_set & right_set
            
            if tb_overlap or lr_overlap:
                tqdm.write(f"WARNING L={l}: Boundary overlap! TB={len(tb_overlap)}, LR={len(lr_overlap)}")

        if fd['node_count'] == 0: continue

        # Site Percolation
        statsSI = percolationStatsI(fd['sub_graph_nodes'], fd['sub_graph_neighbors'], fd['top_boundary_nodes'], fd['bottom_boundary_nodes'], fd['left_boundary_nodes'], fd['right_boundary_nodes'], args.t)
        statsSU = percolationStatsU(fd['sub_graph_nodes'], fd['sub_graph_neighbors'], fd['top_boundary_nodes'], fd['bottom_boundary_nodes'], fd['left_boundary_nodes'], fd['right_boundary_nodes'], args.t)
        mSI.append(statsSI.trials_mean()); sSI.append(statsSI.trials_std())
        mSU.append(statsSU.trials_mean()); sSU.append(statsSU.trials_std())

        # Bond Percolation
        statsBI = percolationStatsBondI(fd['sub_graph_nodes'], fd['sub_graph_edges'], fd['top_boundary_nodes'], fd['bottom_boundary_nodes'], fd['left_boundary_nodes'], fd['right_boundary_nodes'], args.t)
        statsBU = percolationStatsBondU(fd['sub_graph_nodes'], fd['sub_graph_edges'], fd['top_boundary_nodes'], fd['bottom_boundary_nodes'], fd['left_boundary_nodes'], fd['right_boundary_nodes'], args.t)
        mBI.append(statsBI.trials_mean()); sBI.append(statsBI.trials_std())
        mBU.append(statsBU.trials_mean()); sBU.append(statsBU.trials_std())
        
        # results in terminal
        print(f"Site pc Mean: {statsSI.trials_mean():.10f} | Bond pc Mean: {statsBI.trials_mean():.10f}")

    # final plot
    plot_percolation_stats_IU(l_values, mSI, sSI, mSU, sSU, mBI, sBI, mBU, sBU)
    plot_extrapolation_IU(l_values, np.array(mSI), np.array(mSU), np.array(mBI), np.array(mBU))
    plot_frames(l_values, p, args.r, fd['center_x'], fd['center_y'])