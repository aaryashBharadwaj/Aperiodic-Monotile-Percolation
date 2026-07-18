"""r=6 / L=1000 DIRECT (hat vertex graph) percolation run.
Uses the numba thread-parallel stats (percolation_fast); seeded/reproducible,
not bit-identical to the baseline. Center tuned for r=6: (515, -273).
Checkpoints after every L so an interruption is recoverable."""
import argparse, random, datetime, time, os
import numpy as np
import matplotlib; matplotlib.use("Agg")     # headless: plots are saved, not shown
from hat_generator import H_init, T_init, P_init, F_init, constructPatch, constructMetatiles
from hat_graph_builder import build_neighbor_graph_fast, analyze_square_frame
from percolation_fast import (percolationStatsI_par, percolationStatsU_par,
                              percolationStatsBondI_par, percolationStatsBondU_par)
from percolation import extrapolate_pc_raw, nu_width_line, isotropy_test
from results import PercolationResults
from visualisation import plot_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--r',        type=int,   default=6)
    ap.add_argument('--t',        type=int,   default=1000)
    ap.add_argument('--Lmin',     type=float, default=10.0)
    ap.add_argument('--Lmax',     type=float, default=1000.0)
    ap.add_argument('--Lstep',    type=float, default=10.0)
    ap.add_argument('--bt',       type=float, default=1.0)
    ap.add_argument('--seed',     type=int,   default=123456789)
    ap.add_argument('--center_x', type=float, default=515.0)
    ap.add_argument('--center_y', type=float, default=-273.0)
    ap.add_argument('--out_dir',  type=str,   default="results_output")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    seed = args.seed if args.seed is not None else random.randrange(2**32)
    random.seed(seed); np.random.seed(seed)
    print(f"[r6_direct] seed={seed} center=({args.center_x},{args.center_y}) "
          f"L={args.Lmin}..{args.Lmax} step {args.Lstep} t={args.t}", flush=True)

    print("Building patch...", flush=True)
    t0 = time.perf_counter()
    cur = [H_init(), T_init(), P_init(), F_init()]
    for _ in range(args.r):
        p = constructPatch(*cur); cur = constructMetatiles(p)
    print(f"  patch built {time.perf_counter()-t0:.1f}s", flush=True)

    # warm-compile the numba kernels single-threaded before the parallel loop
    _wn = [np.array([1], np.int32), np.array([0, 2], np.int32), np.array([1], np.int32)]
    _we = np.array([[0, 1], [1, 2]], np.int32)
    percolationStatsI_par(np.arange(3), _wn, [0], [2], [0], [2], 2, master_seed=1)
    percolationStatsBondI_par(np.arange(3), _we, [0], [2], [0], [2], 2, master_seed=1)
    print("  numba kernels compiled", flush=True)

    print("Building vertex graph (this is the heavy build)...", flush=True)
    t0 = time.perf_counter()
    master_nodes, master_neighbors = build_neighbor_graph_fast(p, level=args.r + 1)
    print(f"  graph: {len(master_nodes):,} nodes  {time.perf_counter()-t0:.1f}s", flush=True)

    l_values = np.arange(args.Lmin, args.Lmax + 1e-9, args.Lstep)
    raw_SI, raw_SU, raw_BI, raw_BU, raw_pR, raw_pD, valid_L = [], [], [], [], [], [], []

    # --- resume: reload completed L values from a matching checkpoint so any death
    #     (kill, OOM, Windows update) is recoverable by simply re-running this script ---
    done_L = set()
    ckpt = f"{args.out_dir}/CHECKPOINT_r6_direct.npz"
    if os.path.exists(ckpt):
        try:
            prev = PercolationResults.load(ckpt)
            if int(prev.seed) == int(seed) and int(prev.trials) == int(args.t):
                for k in range(len(prev.L_values)):
                    valid_L.append(float(prev.L_values[k]))
                    raw_SI.append(np.asarray(prev.raw_SI[k])); raw_SU.append(np.asarray(prev.raw_SU[k]))
                    raw_BI.append(np.asarray(prev.raw_BI[k])); raw_BU.append(np.asarray(prev.raw_BU[k]))
                    done_L.add(round(float(prev.L_values[k]), 6))
                isof = f"{args.out_dir}/CHECKPOINT_r6_direct_isotropy.npz"
                if os.path.exists(isof):
                    d = np.load(isof)
                    for row in d['pR']: raw_pR.append(np.asarray(row))
                    for row in d['pD']: raw_pD.append(np.asarray(row))
                print(f"[resume] reloaded {len(valid_L)} L values "
                      f"(up to L={max(valid_L) if valid_L else 0:.0f}); continuing", flush=True)
            else:
                print("[resume] checkpoint seed/trials mismatch; starting fresh", flush=True)
        except Exception as e:
            print(f"[resume] checkpoint unreadable ({e}); starting fresh", flush=True)

    for i, l in enumerate(l_values):
        if round(float(l), 6) in done_L:
            continue
        tL = time.perf_counter()
        fd = analyze_square_frame(master_nodes, master_neighbors, L=l,
                                  boundary_thickness=args.bt,
                                  center_x=args.center_x, center_y=args.center_y)
        if fd['node_count'] == 0:
            print(f"L={l}: no nodes, skip", flush=True); continue
        tb = set(fd['top_boundary_nodes']) & set(fd['bottom_boundary_nodes'])
        lr = set(fd['left_boundary_nodes']) & set(fd['right_boundary_nodes'])
        if tb or lr:
            print(f"  WARNING boundary overlap TB={len(tb)} LR={len(lr)}", flush=True)

        base = seed + i * 4
        sSI = percolationStatsI_par(fd['sub_graph_nodes'], fd['sub_graph_neighbors'],
                                    fd['top_boundary_nodes'], fd['bottom_boundary_nodes'],
                                    fd['left_boundary_nodes'], fd['right_boundary_nodes'],
                                    args.t, master_seed=base + 0)
        sSU = percolationStatsU_par(fd['sub_graph_nodes'], fd['sub_graph_neighbors'],
                                    fd['top_boundary_nodes'], fd['bottom_boundary_nodes'],
                                    fd['left_boundary_nodes'], fd['right_boundary_nodes'],
                                    args.t, master_seed=base + 1)
        sBI = percolationStatsBondI_par(fd['sub_graph_nodes'], fd['sub_graph_edges'],
                                        fd['top_boundary_nodes'], fd['bottom_boundary_nodes'],
                                        fd['left_boundary_nodes'], fd['right_boundary_nodes'],
                                        args.t, master_seed=base + 2)
        sBU = percolationStatsBondU_par(fd['sub_graph_nodes'], fd['sub_graph_edges'],
                                        fd['top_boundary_nodes'], fd['bottom_boundary_nodes'],
                                        fd['left_boundary_nodes'], fd['right_boundary_nodes'],
                                        args.t, master_seed=base + 3)

        raw_SI.append(sSI.trialResults); raw_SU.append(sSU.trialResults)
        raw_BI.append(sBI.trialResults); raw_BU.append(sBU.trialResults)
        raw_pR.append(sSI.pR); raw_pD.append(sSI.pD)
        valid_L.append(l)
        print(f"L={l:.0f} N={fd['node_count']:,} site={sSI.trials_mean():.6f} "
              f"bond={sBI.trials_mean():.6f}  ({time.perf_counter()-tL:.1f}s)", flush=True)

        try:
            PercolationResults(tiling_type="hat_vertex", seed=seed, trials=args.t, L_values=valid_L,
                               raw_SI=raw_SI, raw_SU=raw_SU, raw_BI=raw_BI, raw_BU=raw_BU,
                               extra_meta={"r": args.r, "bt": args.bt,
                                           "center": (args.center_x, args.center_y)}
                               ).save(f"{args.out_dir}/CHECKPOINT_r6_direct.npz")
            np.savez(f"{args.out_dir}/CHECKPOINT_r6_direct_isotropy.npz",
                     L=np.array(valid_L, float), pR=np.array(raw_pR, float), pD=np.array(raw_pD, float))
        except Exception as e:
            print(f"  [checkpoint skipped: {e}]", flush=True)

    print(f"\n{'='*60}\nDIRECT EXTRAPOLATION (r={args.r})\n{'='*60}", flush=True)
    print("\n--- Site ---");  extrapolate_pc_raw(valid_L, raw_SI, raw_SU)
    print("\n--- Bond ---");  extrapolate_pc_raw(valid_L, raw_BI, raw_BU)
    print("\n--- Site nu (L>=50) ---");  nu_width_line(valid_L, raw_SI, raw_SU, L_min=50)
    print("\n--- Bond nu (L>=50) ---");  nu_width_line(valid_L, raw_BI, raw_BU, L_min=50)
    print("\n--- Isotropy (L>=50) ---"); isotropy_test(valid_L, raw_pR, raw_pD, L_min=50)

    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    res = PercolationResults(tiling_type="hat_vertex", seed=seed, trials=args.t, L_values=valid_L,
                             raw_SI=raw_SI, raw_SU=raw_SU, raw_BI=raw_BI, raw_BU=raw_BU,
                             extra_meta={"r": args.r, "bt": args.bt,
                                         "center": (args.center_x, args.center_y)})
    res.save(f"{args.out_dir}/results_r6_direct_{ts}.npz")
    np.savez(f"{args.out_dir}/isotropy_r6_direct_{ts}.npz",
             L=np.array(valid_L, float), pR=np.array(raw_pR, float), pD=np.array(raw_pD, float))
    plot_all(res)
    print(f"\n[r6_direct] DONE -> results_r6_direct_{ts}.npz", flush=True)
