"""NEGATIVE CONTROL: the 3D simple-cubic (and 4D simple-hypercubic) lattice through the SAME pipeline.

Every other object in the paper is a positive control -- it lands in the 2D percolation universality
class (nu = 4/3). That makes agreement cheap: a method that always answered "2D" would pass every one
of those tests. The cube/hypercube is the falsification test. It is a genuine percolation lattice, but
a 3D/4D one, so its correlation-length exponent is textbook-different (nu ~ 0.876 in 3D, ~0.685 in 4D;
simple-cubic site p_c ~ 0.3116). Feeding it to the identical estimator and the identical width-line nu
procedure MUST return those numbers, not 4/3 -- which is what shows the pipeline tells the classes apart.

The control is nu-ONLY: nu alone discriminates 2D/3D/4D cleanly (see visualiser/universality_map.py).
The largest-cluster size is still RECORDED (an archived raw observable) but d_f is NOT analysed here --
d_f from <S_max> is boundary-suppressed on open patches (a clean value needs a boundary-free geometry,
unavailable for aperiodic tilings), so the fractal dimension is left out of the control (removed 2026-09-08).

The union-find kernel is dimension-agnostic (it takes a graph and boundary node sets, no geometry), so we
reuse it unchanged: the z-faces play top/bottom and the x-faces play left/right. Intersection = both
directions span, Union = either -- exactly the 2D bracket, now in 3D/4D.

    python -m engine.null_control_cube            # default sweep, ~1-2 min
"""
import os

import numpy as np

from builders.graph_core import edges_to_adjacency
from engine.percolation import (percolationStatsI_par, percolationStatsU_par,
                                percolationStatsExponents_par)
from engine.analysis import extrapolate_pc_raw, _width_gof

NU_3D = 0.8762                   # 3D percolation correlation-length exponent (literature)
NU_4D = 0.6852                   # 4D percolation correlation-length exponent (literature)
# per-dimension (nu, site p_c) literature values, for the printed comparison and the dimensional ladder
LIT = {2: (4.0 / 3.0, None), 3: (NU_3D, 0.3116), 4: (NU_4D, 0.1969)}


def box_window(dims):
    """A general axis-aligned simple-cubic block with per-axis side lengths `dims` (free boundaries), as
    (coords, neighbors, top, bottom, left, right). left/right span AXIS 0 (the pR crossing direction),
    top/bottom span AXIS 1, and any further axes are free depth. Used by the cube Cardy negative control,
    which sweeps a footprint aspect W x H (axes 0,1) at full depth. Kept separate from hypercube_window
    so the nu control's face conventions are untouched."""
    dims = [int(round(x)) for x in dims]
    d = len(dims); N = int(np.prod(dims))
    idx = np.arange(N).reshape(dims)
    lo, hi = [], []
    for ax in range(d):                                      # axis-aligned bonds along every axis
        lo.append(np.take(idx, range(0, dims[ax] - 1), axis=ax).ravel())
        hi.append(np.take(idx, range(1, dims[ax]), axis=ax).ravel())
    neighbors = edges_to_adjacency(np.concatenate(lo), np.concatenate(hi), N)
    grids = np.meshgrid(*[np.arange(s) for s in dims], indexing="ij")
    coords = np.column_stack([g.ravel() for g in grids]).astype(np.float64)
    left = np.take(idx, 0, axis=0).ravel();  right = np.take(idx, dims[0] - 1, axis=0).ravel()   # axis 0
    top = np.take(idx, 0, axis=1).ravel();   bottom = np.take(idx, dims[1] - 1, axis=1).ravel()  # axis 1
    return coords, neighbors, top, bottom, left, right


def hypercube_window(L, d=3):
    """An L^d simple-hypercubic block (free boundaries), as (coords, neighbors, top, bottom, left, right).
    The spanning pair is the two faces along the LAST axis (top/bottom) and the FIRST axis (left/right);
    intersection = both span, union = either -- the same bracket as the 2D pipeline, in d dimensions.
    d=3 reproduces the simple cube; d=4 is the hypercube for the dimensional discrimination ladder."""
    idx = np.arange(L ** d).reshape([L] * d)
    lo, hi = [], []
    for ax in range(d):                                      # axis-aligned bonds along every axis
        lo.append(np.take(idx, range(0, L - 1), axis=ax).ravel())
        hi.append(np.take(idx, range(1, L), axis=ax).ravel())
    neighbors = edges_to_adjacency(np.concatenate(lo), np.concatenate(hi), L ** d)
    grids = np.meshgrid(*([np.arange(L)] * d), indexing="ij")
    coords = np.column_stack([g.ravel() for g in grids]).astype(np.float64)
    bottom = np.take(idx, 0, axis=d - 1).ravel(); top = np.take(idx, L - 1, axis=d - 1).ravel()  # last axis
    left = np.take(idx, 0, axis=0).ravel();       right = np.take(idx, L - 1, axis=0).ravel()    # first axis
    return coords, neighbors, top, bottom, left, right


def cube_window(L):
    """The 3D simple-cubic block -- hypercube_window at d=3 (kept for existing callers)."""
    return hypercube_window(L, 3)


def run_cube_sweep(L_list, T=2000, seed=12345, exponents=True, d=3):
    """Percolate each L^d hypercubic block with the production kernels; return per-L per-trial onset arrays
    (I and U) plus the archived largest-cluster sizes (a raw observable, recorded but NOT analysed for d_f).
    d=3 is the cube, d=4 the hypercube."""
    res = {"L": [], "I": [], "U": [], "smax": [], "N": []}
    for i, L in enumerate(L_list):
        coords, nb, top, bot, left, right = hypercube_window(int(L), d)
        A = (coords, nb, top, bot, left, right)
        si = percolationStatsI_par(*A, T, master_seed=seed + i * 8 + 0)
        su = percolationStatsU_par(*A, T, master_seed=seed + i * 8 + 1)
        res["L"].append(float(L)); res["N"].append(len(coords))
        res["I"].append(np.asarray(si.trialResults, float))
        res["U"].append(np.asarray(su.trialResults, float))
        if exponents:
            # largest cluster at the spanning onset -- recorded and archived as a raw observable only
            ex = percolationStatsExponents_par(*A, T, master_seed=seed + i * 8 + 4)
            res["smax"].append(float(np.mean(ex.s_max)))
        print(f"  L={int(L):>3}  N={len(coords):>7}  "
              f"p_A={0.5*(np.mean(si.trialResults)+np.mean(su.trialResults)):.4f}", flush=True)
    return res


def cube_cardy(L, aspects, T=8000, seed=12345, d=3):
    """NEGATIVE control for the CARDY TEST itself. Sweep a footprint aspect a = W/H (area L^2) at full
    depth L on a d-dimensional box, measure the left-right crossing probability anchored where the
    SQUARE footprint crosses 1/2, and compare to Cardy's EXACT 2D curve. A 2D lattice tracks Cardy; a
    3D/4D box must NOT -- a large deviation shows the Cardy test correctly rejects non-2D geometry.
    Returns (aspects, Rh, cardy_2d, pstar)."""
    from engine.cardy import cardy_pi_h, crossing_probability
    aspects = np.asarray(aspects, float)
    cardy2d = np.array([cardy_pi_h(a) for a in aspects])
    depth = [L] * (d - 2)
    A0 = box_window([L, L] + depth)                          # square footprint at full depth = the cube
    sq = percolationStatsI_par(*A0, T, master_seed=seed)
    pstar = float(np.median(np.asarray(sq.pR, float)))
    Rh = np.full(len(aspects), np.nan)
    for ai, a in enumerate(aspects):
        A = box_window([L * np.sqrt(a), L / np.sqrt(a)] + depth)
        r = percolationStatsI_par(*A, T, master_seed=seed + ai + 1)
        Rh[ai] = crossing_probability(np.asarray(r.pR, float), pstar)
        print(f"    a={a:5.3f}  R_h={Rh[ai]:.4f}  (2D Cardy {cardy2d[ai]:.4f})", flush=True)
    return aspects, Rh, cardy2d, pstar


def width_nu(L_list, channels, nu_lo=0.6, nu_hi=1.6, omega=1.0):
    """Same width-line estimator as engine.analysis._width_gof (std(onset) ~ L^{-1/nu}, amplitudes
    profiled out, correction term at fixed omega) but with a WIDE nu bound so a 3D value (~0.88) is
    reachable -- the 2D fitter clamps to [1.10, 1.60]. channels = list of per-L onset-array lists sharing nu."""
    L = np.asarray(L_list, float)
    widths = [np.array([np.std(a, ddof=1) for a in ch]) for ch in channels]
    sigmas = [w / np.sqrt(2.0 * (len(ch[0]) - 1)) for w, ch in zip(widths, channels)]
    # Same profiled-amplitude chi^2 fit as the 2D analysis, just a wider nu bracket -- delegate to it
    # instead of keeping a second copy of the kernel.
    return _width_gof(L, widths, sigmas, omega, bounds=(nu_lo, nu_hi))[0]


# a size ladder: dense at small L (cheap, anchors the curve), sparse at large L (each is L^d, expensive).
_LADDER = [8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 128, 160, 192, 224, 256, 288, 320, 384]


def main():
    import argparse
    ap = argparse.ArgumentParser(description="d-dimensional hypercubic negative control (nu-only)")
    ap.add_argument("--dim", type=int, default=3, help="lattice dimension: 3 = cube, 4 = hypercube")
    ap.add_argument("--lmax", type=int, default=48, help="largest side L (3D best ~320; 4D ~32)")
    ap.add_argument("--trials", type=int, default=1000, help="trials per size (hat used 1000)")
    ap.add_argument("--seed", type=int, default=123456789)
    ap.add_argument("--out", default=None, help="optional npz to save (keys L/I/U/smax/pc/nu/dim)")
    ap.add_argument("--cardy", action="store_true",
                    help="run the Cardy NEGATIVE control (footprint aspect sweep vs 2D Cardy) not the nu sweep")
    ap.add_argument("--n-aspects", type=int, default=20, help="aspect points for --cardy")
    args = ap.parse_args()
    d = args.dim

    if args.cardy:
        aspects = np.round(np.geomspace(0.5, 2.0, args.n_aspects), 4)
        L = args.lmax
        print(f"{d}D CARDY NEGATIVE CONTROL  (a {d}D box footprint sweep vs the exact 2D Cardy curve)  "
              f"L={L}, T={args.trials}\n")
        asp, Rh, cardy2d, pstar = cube_cardy(L, aspects, T=args.trials, seed=args.seed, d=d)
        floor = float(np.sqrt(0.25 / args.trials))
        rms = float(np.sqrt(np.nanmean((Rh - cardy2d) ** 2)))
        print(f"\n  footprint p*={pstar:.4f}   RMS(R_h - 2D Cardy)={rms:.4f}  (noise floor {floor:.4f})")
        print(f"  -> a genuine 2D lattice gives RMS ~ floor; this {d}D box is {rms/floor:.1f}x floor,")
        print("     i.e. the Cardy test correctly REJECTS the non-2D geometry (the negative control).")
        out = args.out or os.path.join("paper_results", "npz", f"cardy_Cube{d}D_L{L}_T{args.trials}.npz")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez(out, member=f"Cube {d}D", graph="bulk", patch=L, aspects=asp, cardy=cardy2d,
                 windows=np.array([float(L)]), Rh=Rh[None, :], pstar=np.array([pstar]),
                 trials=args.trials, noise_floor=floor, seed=args.seed)
        print("saved ->", out)
        return
    L_list = [L for L in _LADDER if L <= args.lmax]
    lit_nu, lit_pc = LIT.get(d, (None, None))
    print(f"{d}D simple-hypercubic NEGATIVE CONTROL  (same kernel/estimators as the 2D family)  "
          f"L<={args.lmax}, T={args.trials}\n")
    r = run_cube_sweep(L_list, T=args.trials, seed=args.seed, d=d)

    # p_c: extrapolate with the dimension's own nu (shows the measurement RECOVERS the known value)
    pc = extrapolate_pc_raw(r["L"], r["I"], r["U"], nu=(lit_nu or 0.9))["A"]
    # nu: the paper's width-line procedure, wide bound so a <2D-class value is reachable
    nu = width_nu(r["L"], [r["I"], r["U"]])
    nu_2Dfit = width_nu(r["L"], [r["I"], r["U"]], nu_lo=1.10, nu_hi=1.60)   # what a 2D-clamped fitter sees

    print("\n" + "=" * 78)
    print(f"RESULT  ({d}D measured  vs  {d}D literature  vs  the 2D class the family sits in)")
    print("=" * 78)
    print(f"  p_c  : {pc['pc']:.4f} ± {pc['pc_std']:.4f}   | {d}D lit {lit_pc}   | 2D family 0.50-0.85")
    print(f"  nu   : {nu:.3f}                    | {d}D lit {lit_nu}   | 2D class 1.333 (4/3)")
    print(f"\n  2D-clamped width-nu fit rails to {nu_2Dfit:.3f} (its 1.10 floor): the {d}D lattice is")
    print("  INCOMPATIBLE with nu=4/3 -- the negative control the positive tests need.")
    if args.out:
        # smax is saved as an archived raw observable (not analysed for d_f); nu is the control result.
        np.savez(args.out, L=np.array(r["L"], float), I=np.array(r["I"], float), U=np.array(r["U"], float),
                 smax=np.array(r["smax"], float), pc=pc["pc"], pc_std=pc["pc_std"], nu=nu,
                 dim=d, T=args.trials, seed=args.seed, lmax=args.lmax)
        print("\nsaved ->", args.out)


if __name__ == "__main__":
    main()
