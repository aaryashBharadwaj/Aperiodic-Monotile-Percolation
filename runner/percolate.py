"""Consolidated percolation runner — one command runs ANY tiling.

This is both (a) the engine the GUI's Run button launches as a detached background process, and
(b) a standalone console runner you can copy-paste. It reuses the EXACT kernels the GUI uses
(gui_backend.build_graph + run_one), so results are identical whichever way you launch it. After
every L it writes a checkpoint (.npz) and a status JSON; a re-launch with the same parameters
RESUMES from the checkpoint. On completion it extrapolates and saves the result to results_output/.

Console examples (run from the repo root; member/kind auto-derived; graph takes 'direct'/'dual'):
  python runner/percolate.py --tiling Hat     --graph direct --patch 6  --lmin 10 --lmax 1000 --gap 10 --trials 1000 --seed 123456789
  python runner/percolate.py --tiling Spectre --graph dual   --patch 6  --lmin 20 --lmax 560  --gap 20 --trials 500  --seed 123456789
  python runner/percolate.py --tiling Square  --graph direct --patch 300 --lmin 50 --lmax 400 --gap 25 --trials 40000 --seed 123456789
  python runner/percolate.py --tiling "Tile(a,b) family" --graph direct --a 1 --b 1.732 --patch 6 --lmin 10 --lmax 200 --gap 20 --trials 500 --seed 1

--tiling is one of: Hat, Spectre, Comet, Chevron, Square, Penrose, "Triangular -> Honeycomb",
"Tile(a,b) family".  --patch is the inflation level (hat/spectre), subdivisions (penrose), or cell
count (periodic).  Interrupt and re-run the same command to resume.
"""
import argparse
import os
import sys
import time

import numpy as np

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
    ap.add_argument("--out-dir", default=None, help="where the result .npz is saved (default results_output/)")
    ap.add_argument("--jobs-dir", default=None, help="(accepted for GUI compatibility; unused)")
    ap.add_argument("--name", default="", help="output file name (default auto from parameters)")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # percolate is in runner/
    sys.path.insert(0, repo_root)                                             # so the packages import
    import interface.gui_backend as gb

    # Derive the pieces the GUI passes explicitly but a console user shouldn't have to.
    graph = _GRAPH.get(args.graph.strip().lower(), args.graph)
    kind = args.kind or ("dual" if graph.startswith("Dual") else "direct")
    a, b = round(args.a, 3), round(args.b, 3)
    patch = int(round(args.patch))
    member = args.member or gb.resolve_member(args.tiling, a, b)
    out_dir = args.out_dir or os.path.join(repo_root, "results_output")
    bt = args.bt if args.bt is not None else gb.default_bt(member)
    jid = args.job_id or gb.make_job_id(member, kind, patch, a, b, args.lmin, args.lmax,
                                        args.gap, int(args.trials), int(args.seed))

    p = gb.job_paths(jid)
    ckpt, stop_flag = p["ckpt"], p["stop"]
    ckpt_tmp = ckpt[:-4] + ".tmp.npz"      # must end in .npz or np.savez appends it and rename fails

    # L list: the SHARED engine definition, identical to the GUI preview/ETA.
    from engine.percolation import l_sweep
    Ls = l_sweep(args.lmin, args.lmax, args.gap)
    total = len(Ls)
    started = time.time()

    print(f"[{member} / {kind}] patch={patch} L={args.lmin:g}..{args.lmax:g} step {args.gap:g} "
          f"({total} sizes) T={args.trials} seed={args.seed}  (job {jid})", flush=True)

    def status(state, i, last_line="", result_file=None, error=None):
        gb.write_status(jid, {
            "job_id": jid, "status": state, "i": i, "total": total, "n_valid": len(valid),
            "last_line": last_line, "member": member, "kind": kind, "patch": patch,
            "seed": args.seed, "trials": args.trials, "lmin": args.lmin, "lmax": args.lmax,
            "gap": args.gap, "started": started, "updated": time.time(), "pid": os.getpid(),
            "result_file": result_file, "error": error, "name": args.name})

    # Accumulators (+ resume from a matching checkpoint if one exists).
    # rPR/rPD are the site direction-bias crossings; rBPR/rBPD the bond ones.
    valid, rSI, rSU, rBI, rBU, rPR, rPD, rBPR, rBPD, skipped = ([] for _ in range(10))
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
            skipped = [float(x) for x in z["skipped"]]
            print(f"  resuming from checkpoint at size {start_i}/{total}", flush=True)
        except Exception:
            start_i = 0  # unreadable/old-format checkpoint -> start clean
            valid, rSI, rSU, rBI, rBU, rPR, rPD, rBPR, rBPD, skipped = ([] for _ in range(10))

    def save_ckpt(next_i):
        np.savez(ckpt_tmp,
                 Lvals=np.array(valid, float), SI=np.array(rSI, float), SU=np.array(rSU, float),
                 BI=np.array(rBI, float), BU=np.array(rBU, float), PR=np.array(rPR, float),
                 PD=np.array(rPD, float), BPR=np.array(rBPR, float), BPD=np.array(rBPD, float),
                 skipped=np.array(skipped, float), next_i=next_i)
        gb._safe_replace(ckpt_tmp, ckpt)

    try:
        if total < 1:
            status("error", start_i, error="empty L sweep")
            print("empty L sweep -- check lmin/lmax/gap", flush=True)
            return

        status("building", start_i, last_line="building graph...")
        print("  building graph...", flush=True)
        bundle = gb.build_graph(args.tiling, graph, patch, a, b)

        last_i = start_i
        stopped = False
        for i in range(start_i, total):
            if os.path.exists(stop_flag):
                stopped = True
                break
            L = Ls[i]
            try:
                step = gb.run_one(bundle, L, args.trials, args.seed + i * 4, bt)
            except Exception as ex:
                step = {"usable": False, "L": float(L), "err": str(ex)}
            if step.get("usable"):
                valid.append(step["L"]); rSI.append(step["SI"]); rSU.append(step["SU"])
                rBI.append(step["BI"]); rBU.append(step["BU"])
                rPR.append(step["pR"]); rPD.append(step["pD"])
                rBPR.append(step["bond_pR"]); rBPD.append(step["bond_pD"])
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
        res = gb.extrapolate_result(valid, rSI, rSU, rBI, rBU, skipped, rPR, rPD, rBPR, rBPD)
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
