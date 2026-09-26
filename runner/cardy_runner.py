"""Cardy crossing-probability test on a tiling.

Sweeps rectangle aspect ratios a = W/H at one or more window sizes L (rectangles hold area = L^2), and
for each window measures the left-right crossing probability, ANCHORED at the p where the square (a=1)
crosses at 1/2 -- not an external p_c, which is fragile. The other aspect ratios read at that same p are
free predictions that Cardy's formula (engine/cardy.py) must reproduce. Running several window sizes
shows the finite-size deviation dying (crossing probability at criticality is scale-invariant, so it
converges fast). Saves an npz the figure and analysis read. See the plan / engine/cardy.py for theory.

Example:
    python runner/cardy_runner.py --tiling Hat --graph direct --patch 6 \
        --windows 150 300 450 --n-aspects 20 --trials 8000 --threads 0
"""
import os
import sys
import time
import argparse

import numpy as np

_GRAPH = {"direct": "Direct (vertex)", "vertex": "Direct (vertex)",
          "dual": "Dual (tile)", "tile": "Dual (tile)"}


def main():
    ap = argparse.ArgumentParser(description="Cardy crossing-probability test (aspect-ratio sweep).")
    ap.add_argument("--tiling", required=True)
    ap.add_argument("--graph", default="direct", help="direct (vertex) or dual (tile)")
    ap.add_argument("--patch", type=int, required=True, help="patch size big enough to hold the windows")
    ap.add_argument("--windows", type=float, nargs="+", required=True,
                    help="window size(s) L; each aspect uses W=L*sqrt(a), H=L/sqrt(a) so area=L^2")
    ap.add_argument("--n-aspects", type=int, default=20)
    ap.add_argument("--aspect-min", type=float, default=0.5)
    ap.add_argument("--aspect-max", type=float, default=2.0)
    ap.add_argument("--trials", type=int, default=8000, help="per window; noise floor ~ sqrt(1/4T)")
    ap.add_argument("--seed", type=int, default=123456789)
    ap.add_argument("--threads", type=int, default=0, help="trial concurrency (0 = auto)")
    ap.add_argument("--bt", type=float, default=None, help="boundary thickness (default per-tiling)")
    ap.add_argument("--a", type=float, default=1.0, help="Tile(a,b) short edge (family only)")
    ap.add_argument("--b", type=float, default=1.7320508075688772, help="Tile(a,b) long edge (family only)")
    ap.add_argument("--scale", type=float, default=None, help="Penrose physical extent (default 2*max window)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo)
    import engine.pipeline as gb
    from engine.cardy import cardy_pi_h, crossing_probability

    graph = _GRAPH.get(args.graph.strip().lower(), args.graph)
    a, b = round(args.a, 3), round(args.b, 3)
    member = gb.resolve_member(args.tiling, a, b)
    bt = args.bt if args.bt is not None else gb.default_bt(member)
    pscale = args.scale if args.scale else (2 * max(args.windows) if member == "Penrose" else None)

    print("  building graph...", flush=True)
    bundle = gb.build_graph(args.tiling, graph, args.patch, a, b, scale=pscale)
    started = time.time()
    print(f"[Cardy] {member} / {graph}  patch={args.patch}  side={bundle['side']:.0f}  "
          f"windows={[float(w) for w in args.windows]}  aspects={args.n_aspects}  T={args.trials}",
          flush=True)

    aspects = np.round(np.geomspace(args.aspect_min, args.aspect_max, args.n_aspects), 4)
    cardy = np.array([cardy_pi_h(a_) for a_ in aspects])
    floor = float(np.sqrt(0.25 / args.trials))

    windows_done, Rh_rows, pstar_list, band_rows = [], [], [], []
    B_ANCHOR = 400          # bootstrap resamples of the square's pR -> the anchor-uncertainty band
    for wi, L in enumerate(args.windows):
        sq = gb.run_crossing(bundle, L, L, args.trials, args.seed + wi * 1000, bt, nworkers=args.threads)
        if not sq.get("usable"):
            print(f"  L={L:.0f}: square window unusable (too large for this patch) -- skipped", flush=True)
            continue
        sq_pR = np.asarray(sq["pR"], float)
        pstar = float(np.median(sq_pR))
        Rh = np.full(len(aspects), np.nan)
        aspect_pR = [None] * len(aspects)               # keep each aspect's raw crossings for the band
        for ai, asp in enumerate(aspects):
            W = L * np.sqrt(asp); H = L / np.sqrt(asp)
            r = gb.run_crossing(bundle, W, H, args.trials, args.seed + wi * 1000 + ai + 1, bt,
                                nworkers=args.threads)
            if r.get("usable"):
                aspect_pR[ai] = np.asarray(r["pR"], float)
                Rh[ai] = crossing_probability(aspect_pR[ai], pstar)
        # ANCHOR-BOOTSTRAP BAND: p* is a median of T square crossings, so it carries a statistical
        # error that the steep crossing curve amplifies into a CORRELATED shift of every aspect's R_h
        # (this is what makes a small p* wobble show up as a same-sign offset across the whole curve).
        # Resample the square's pR, recompute p* and every crossing, and report the per-aspect std as
        # the (correlated) anchor uncertainty. No extra runs -- pure post-processing of the square's pR.
        rng = np.random.default_rng(args.seed + wi * 1000 + 99991)
        n_sq = len(sq_pR)
        boot_pstars = np.array([np.median(sq_pR[rng.integers(0, n_sq, n_sq)]) for _ in range(B_ANCHOR)])
        band = np.full(len(aspects), np.nan)
        for ai in range(len(aspects)):
            if aspect_pR[ai] is not None:
                band[ai] = float(np.std([crossing_probability(aspect_pR[ai], ps) for ps in boot_pstars]))
        res = Rh - cardy
        rms = float(np.sqrt(np.nanmean(res ** 2)))
        windows_done.append(float(L)); Rh_rows.append(Rh); pstar_list.append(pstar); band_rows.append(band)
        print(f"  L={L:.0f}  N~{sq['N']:,}  p*={pstar:.4f}  RMS(R_h - Cardy)={rms:.4f}  "
              f"(noise floor {floor:.4f}, anchor band ~{np.nanmedian(band):.4f})", flush=True)

    if not windows_done:
        print("no usable windows -- nothing saved.", flush=True)
        return

    out = args.out or os.path.join(gb.RESULTS_DIR,
                                   f"cardy_{member}_{args.graph}_patch{args.patch}.npz")
    np.savez(out, member=str(member), graph=str(graph), patch=args.patch,
             aspects=aspects, cardy=cardy, windows=np.array(windows_done, float),
             Rh=np.array(Rh_rows, float), Rh_band=np.array(band_rows, float),
             pstar=np.array(pstar_list, float),
             trials=args.trials, noise_floor=floor, seed=args.seed)
    print(f"\ndone in {time.time() - started:.0f}s -- saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
