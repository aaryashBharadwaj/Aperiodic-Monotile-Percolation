import argparse
import os
import sys
import time

import numpy as np

# One runner for every tiling
# Same kernels the GUI uses, so results match either way. 
# Checkpoints after every L -- re-run the same command to resume.
# --patch means inflation level (hat/spectre), subdivisions (penrose), or cell count (periodic).

# Sample Inputs: 

#   python runner/runner.py --tiling Hat --graph direct --patch 6 --lmin 10 --lmax 1000 --gap 10 --trials 1000 --seed 123456789
#   python runner/runner.py --tiling Spectre --graph dual --patch 6 --lmin 20 --lmax 560 --gap 20 --trials 500 --seed 123456789
#   python runner/runner.py --tiling Penrose --graph direct --patch 9 --lmin 20 --lmax 200 --gap 20 --trials 1000 --seed 123456789

_GRAPH = {"direct": "Direct (vertex)", "vertex": "Direct (vertex)",
          "dual": "Dual (tile)", "tile": "Dual (tile)"}


def main():
    ap = argparse.ArgumentParser(description="Run percolation on any tiling (GUI engine + console runner).")
    ap.add_argument("--tiling", required=True,
                    help="Hat | Spectre | Comet | Chevron | Square | Penrose | "
                         "'Triangular -> Honeycomb' | 'Tile(a,b) family'")
    ap.add_argument("--graph", required=True, help="'direct' or 'dual' (or the full graph name)")
    ap.add_argument("--patch", type=float, required=True,
                    help="inflation level (hat/spectre), subdivisions (penrose), or cell count (periodic)")
    ap.add_argument("--lmin", type=float, required=True)
    ap.add_argument("--lmax", type=float, required=True)
    ap.add_argument("--gap", type=float, required=True)
    ap.add_argument("--trials", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--a", type=float, default=1.0, help="Tile(a,b) short-edge length (family only)")
    ap.add_argument("--b", type=float, default=1.7320508075688772, help="Tile(a,b) long-edge length (family only)")
    ap.add_argument("--bt", type=float, default=None,
                    help="boundary-band thickness (default per-tiling: 2 for Penrose, else 1; the "
                         "paper Penrose run used 10)")
    ap.add_argument("--member", default=None, help="override the auto-derived member name")
    ap.add_argument("--kind", default=None, help="override the auto-derived direct/dual")
    ap.add_argument("--job-id", default=None, help="override the auto id (the GUI sets this to monitor)")
    ap.add_argument("--out-dir", default=None, help="where the result .npz is saved (default paper_results/npz/)")
    ap.add_argument("--jobs-dir", default=None, help="(accepted for GUI compatibility; unused)")
    ap.add_argument("--name", default="", help="output file name (default auto from parameters)")
    ap.add_argument("--exponents", action="store_true",
                    help="also run the largest-cluster pass (records s_max/chi/histogram -> d_f, "
                         "gamma/nu, tau). Adds an extra sweep + O(N) snapshot per size (~+30%% time).")
    ap.add_argument("--threads", type=int, default=0, metavar="N",
                    help="number of trials to run concurrently (the parallel dimension is over "
                         "independent trials, on nogil kernels -> real multi-core). 0 = auto "
                         "(cpu_count-1). On a big box raise it; at very large L lower it, since peak "
                         "RAM ~ threads x O(nodes).")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # runner.py is in runner/
    sys.path.insert(0, repo_root)                                             # so the packages import
    import interface.gui_backend as gb
    import runner.jobs as jobs

    # Derive the pieces the GUI passes explicitly but a console user shouldn't have to.
    graph = _GRAPH.get(args.graph.strip().lower(), args.graph)
    kind = args.kind or ("dual" if graph.startswith("Dual") else "direct")
    a, b = round(args.a, 3), round(args.b, 3)
    patch = int(round(args.patch))
    member = args.member or gb.resolve_member(args.tiling, a, b)
    out_dir = args.out_dir or gb.RESULTS_DIR
    bt = args.bt if args.bt is not None else gb.default_bt(member)
    jid = args.job_id or jobs.make_job_id(member, kind, patch, a, b, args.lmin, args.lmax,
                                        args.gap, int(args.trials), int(args.seed))

    p = jobs.job_paths(jid)
    ckpt, stop_flag = p["ckpt"], p["stop"]
    ckpt_tmp = ckpt[:-4] + ".tmp.npz"      # must end in .npz or np.savez appends it and rename fails

    # L list: the SHARED engine definition, identical to the GUI preview/ETA.
    from engine.percolation import l_sweep
    Ls = l_sweep(args.lmin, args.lmax, args.gap)
    total = len(Ls)
    started = time.time()

    from engine.percolation import _NW
    n_threads = args.threads if args.threads and args.threads > 0 else _NW
    print(f"[{member} / {kind}] patch={patch} L={args.lmin:g}..{args.lmax:g} step {args.gap:g} "
          f"({total} sizes) T={args.trials} seed={args.seed}  threads={n_threads}  (job {jid})",
          flush=True)

    def status(state, i, last_line="", result_file=None, error=None):
        jobs.write_status(jid, {
            "job_id": jid, "status": state, "i": i, "total": total, "n_valid": len(valid),
            "last_line": last_line, "member": member, "kind": kind, "patch": patch,
            "seed": args.seed, "trials": args.trials, "lmin": args.lmin, "lmax": args.lmax,
            "gap": args.gap, "started": started, "updated": time.time(), "pid": os.getpid(),
            "result_file": result_file, "error": error, "name": args.name})

    # Accumulators (+ resume from a matching checkpoint if one exists).
    # rPR/rPD site direction-bias crossings; rBPR/rBPD bond; rSM the largest-cluster sizes (d_f).
    valid, rSI, rSU, rBI, rBU, rPR, rPD, rBPR, rBPD, rSM, rSMi, rBSM, rBSMi, skipped = \
        ([] for _ in range(14))
    start_i = 0
    if os.path.exists(ckpt):
        try:
            z = np.load(ckpt, allow_pickle=False)
            start_i = int(z["next_i"])
            valid = [float(x) for x in z["Lvals"]]
            rSI = [row for row in z["SI"]]; rSU = [row for row in z["SU"]]
            rBI = [row for row in z["BI"]]; rBU = [row for row in z["BU"]]
            rPR = [row for row in z["PR"]]; rPD = [row for row in z["PD"]]
            rBPR = [row for row in z["BPR"]]; rBPD = [row for row in z["BPD"]]
            if "SM" in z and z["SM"].size:
                rSM = [row for row in z["SM"]]
            if "SMI" in z and z["SMI"].size:
                rSMi = [row for row in z["SMI"]]
            if "BSM" in z and z["BSM"].size:
                rBSM = [row for row in z["BSM"]]
            if "BSMI" in z and z["BSMI"].size:
                rBSMi = [row for row in z["BSMI"]]
            skipped = [float(x) for x in z["skipped"]]
            print(f"  resuming from checkpoint at size {start_i}/{total}", flush=True)
        except Exception:
            start_i = 0  # unreadable/old-format checkpoint -> start clean
            valid, rSI, rSU, rBI, rBU, rPR, rPD, rBPR, rBPD, rSM, rSMi, rBSM, rBSMi, skipped = \
                ([] for _ in range(14))

    # The job id doesn't distinguish an --exponents run from a plain one, so the same params can resume
    # a checkpoint written WITHOUT the largest-cluster pass. That checkpoint has no per-size s_max, so the
    # d_f accumulators would be short by start_i entries (and fit_exponents would then silently drop d_f).
    # If we want exponents but the resumed accumulators don't line up with the sizes, restart clean.
    if args.exponents and len(rSM) != len(valid):
        start_i = 0
        valid, rSI, rSU, rBI, rBU, rPR, rPD, rBPR, rBPD, rSM, rSMi, rBSM, rBSMi, skipped = \
            ([] for _ in range(14))

    def save_ckpt(next_i):
        np.savez(ckpt_tmp,
                 Lvals=np.array(valid, float), SI=np.array(rSI, float), SU=np.array(rSU, float),
                 BI=np.array(rBI, float), BU=np.array(rBU, float), PR=np.array(rPR, float),
                 PD=np.array(rPD, float), BPR=np.array(rBPR, float), BPD=np.array(rBPD, float),
                 SM=np.array(rSM, float), SMI=np.array(rSMi, float),
                 BSM=np.array(rBSM, float), BSMI=np.array(rBSMi, float),
                 skipped=np.array(skipped, float), next_i=next_i)
        jobs._safe_replace(ckpt_tmp, ckpt)

    try:
        if total < 1:
            status("error", start_i, error="empty L sweep")
            print("empty L sweep -- check lmin/lmax/gap", flush=True)
            return

        status("building", start_i, last_line="building graph...")
        print("  building graph...", flush=True)
        # Penrose: --patch is subdivisions (density), so the physical extent comes from `scale`, which
        # must exceed L_max. Reproduce the original run_penrose rule scale = 2*L_max so the patch is big
        # enough for the whole sweep (else large-L frames fall outside the patch and skip).
        pscale = int(2 * args.lmax) if member == "Penrose" else None
        bundle = gb.build_graph(args.tiling, graph, patch, a, b, scale=pscale)
        # run_one caps windows at 0.95*side (side = inscribed square from largest_square_center, minus a
        # 5% margin so the largest window sits just inside the tiling). Warn a console user whose --lmax
        # overshoots that cap, so a capped sweep isn't a silent surprise (the paper presets sit under it).
        _side = bundle.get("side")
        if _side is not None and args.lmax > 0.95 * _side + 1e-9:
            print(f"  note: --lmax {args.lmax:g} exceeds the safe window cap {0.95 * _side:.0f} "
                  f"(0.95 x inscribed square) for this patch; larger sizes clip the fringe and will be "
                  f"skipped.", flush=True)

        last_i = start_i
        stopped = False
        for i in range(start_i, total):
            if os.path.exists(stop_flag):
                stopped = True
                break
            L = Ls[i]
            try:
                # stride 8 per size: offsets 0-3 = thresholds, 4 = exponents (seed_base+4), 5-7 spare.
                step = gb.run_one(bundle, L, args.trials, args.seed + i * 8, bt,
                                  exponents=args.exponents, nworkers=args.threads)
            except Exception as ex:
                step = {"usable": False, "L": float(L), "err": str(ex)}
            if step.get("usable"):
                valid.append(step["L"]); rSI.append(step["SI"]); rSU.append(step["SU"])
                rBI.append(step["BI"]); rBU.append(step["BU"])
                rPR.append(step["pR"]); rPD.append(step["pD"])
                rBPR.append(step["bond_pR"]); rBPD.append(step["bond_pD"])
                if args.exponents:
                    rSM.append(step["s_max"]); rSMi.append(step.get("s_max_i"))
                    rBSM.append(step.get("bond_s_max")); rBSMi.append(step.get("bond_s_max_i"))
                ms = sum(step["SI"]) / len(step["SI"]); mb = sum(step["BI"]) / len(step["BI"])
                line = f"size {i+1}/{total}  L={L:.0f}  N={step['N']:,}  site={ms:.3f}  bond={mb:.3f}"
            else:
                skipped.append(float(L))
                line = f"size {i+1}/{total}  L={L:.0f}  skipped (outside patch)"
            last_i = i + 1
            save_ckpt(last_i)
            status("running" if last_i < total else "finalising", last_i, last_line=line)
            print("  " + line, flush=True)

        # Finalise: extrapolate (prints the p_c + direction-bias summary to stdout) + SAVE.
        print("\n" + "=" * 60 + f"\n{member} / {kind} extrapolation\n" + "=" * 60, flush=True)
        res = gb.extrapolate_result(valid, rSI, rSU, rBI, rBU, skipped, rPR, rPD, rBPR, rBPD,
                                    rSM if args.exponents else None,
                                    rSMi if args.exponents else None,
                                    rBSM if args.exponents else None,
                                    rBSMi if args.exponents else None)
        result_file = None
        if valid:
            default_name = args.name or (
                f"gui_{member}_{kind}_patch{patch}"
                f"_L{int(min(valid))}-{int(max(valid))}_T{args.trials}_s{args.seed}")
            path = gb.save_result(res, member, kind, args.seed, args.trials,
                                  name=default_name, out_dir=out_dir)
            result_file = os.path.basename(path)
            print(f"\nsaved -> {path}", flush=True)
        status("stopped" if stopped else "done", last_i,
               last_line=("stopped -- partial result saved" if stopped else "complete"),
               result_file=result_file)
        print("stopped early (partial saved)." if stopped else "done.", flush=True)
    except Exception as ex:
        import traceback
        status("error", start_i, error=f"{ex}\n{traceback.format_exc()}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
