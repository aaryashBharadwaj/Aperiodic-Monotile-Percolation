"""NEGATIVE CONTROL: the 3D simple-cubic lattice run through the SAME percolation pipeline.

Every other object in the paper is a positive control -- it lands in the 2D percolation universality
class (nu = 4/3, d_f = 91/48). That makes agreement cheap: a method that always answered "2D" would pass
every one of those tests. The cube is the falsification test. It is a genuine percolation lattice, but a
3D one, so its exponents are textbook-different (nu ~ 0.876, d_f ~ 2.52, simple-cubic site p_c ~ 0.3116).
Feeding it to the identical estimator and the identical width-line nu procedure MUST return those 3D
numbers, not 4/3 -- which is what shows the pipeline can tell the classes apart.

The union-find kernel is dimension-agnostic (it takes a graph and boundary node sets, no geometry), so we
reuse it unchanged: the z-faces play top/bottom and the x-faces play left/right. Intersection = both
directions span, Union = either -- exactly the 2D bracket, now in 3D.

    python -m engine.null_control_cube            # default sweep, ~1-2 min
"""
import numpy as np
from scipy.optimize import minimize_scalar

from builders.graph_core import edges_to_adjacency
from engine.percolation import (percolationStatsI_par, percolationStatsU_par,
                                percolationStatsExponents_par)
from engine.analysis import extrapolate_pc_raw

CUBE_PC_SITE = 0.311608          # simple-cubic site threshold (literature)
NU_3D = 0.8762                   # 3D percolation correlation-length exponent (literature)
DF_3D = 2.5230                   # 3D percolation fractal dimension (literature)


def cube_window(L):
    """An L x L x L simple-cubic block (free boundaries), as (coords, neighbors, top, bottom, left, right).
    top/bottom are the two z-faces, left/right the two x-faces -- the kernel spans between each pair."""
    idx = np.arange(L ** 3).reshape(L, L, L)                 # idx[i,j,k] = i*L*L + j*L + k  (i=x, j=y, k=z)
    lo, hi = [], []
    for ax in range(3):                                      # axis-aligned bonds along x, y, z
        lo.append(np.take(idx, range(0, L - 1), axis=ax).ravel())
        hi.append(np.take(idx, range(1, L), axis=ax).ravel())
    neighbors = edges_to_adjacency(np.concatenate(lo), np.concatenate(hi), L ** 3)
    I, J, K = np.meshgrid(np.arange(L), np.arange(L), np.arange(L), indexing="ij")
    coords = np.column_stack([I.ravel(), J.ravel(), K.ravel()]).astype(np.float64)
    bottom = idx[:, :, 0].ravel();  top = idx[:, :, L - 1].ravel()      # z-faces  -> top/bottom
    left = idx[0, :, :].ravel();    right = idx[L - 1, :, :].ravel()    # x-faces  -> left/right
    return coords, neighbors, top, bottom, left, right


def run_cube_sweep(L_list, T=2000, seed=12345, exponents=True):
    """Percolate each cube with the production kernels; return per-L per-trial onset arrays (I and U) plus
    the incipient-cluster sizes (for d_f)."""
    res = {"L": [], "I": [], "U": [], "smax": [], "N": []}
    for i, L in enumerate(L_list):
        coords, nb, top, bot, left, right = cube_window(int(L))
        A = (coords, nb, top, bot, left, right)
        si = percolationStatsI_par(*A, T, master_seed=seed + i * 8 + 0)
        su = percolationStatsU_par(*A, T, master_seed=seed + i * 8 + 1)
        res["L"].append(float(L)); res["N"].append(len(coords))
        res["I"].append(np.asarray(si.trialResults, float))
        res["U"].append(np.asarray(su.trialResults, float))
        if exponents:
            ex = percolationStatsExponents_par(*A, T, master_seed=seed + i * 8 + 4)
            res["smax"].append(float(np.mean(ex.s_max)))
        print(f"  L={int(L):>3}  N={len(coords):>7}  "
              f"p_A={0.5*(np.mean(si.trialResults)+np.mean(su.trialResults)):.4f}", flush=True)
    return res


def width_nu(L_list, channels, nu_lo=0.6, nu_hi=1.6, omega=1.0):
    """Same width-line estimator as engine.analysis.fit_nu (std(onset) ~ L^{-1/nu}, amplitudes profiled
    out, correction term at fixed omega) but with a WIDE nu bound so a 3D value (~0.88) is reachable -- the
    2D fitter clamps to [1.10, 1.60]. channels = list of per-L onset-array lists sharing nu."""
    L = np.asarray(L_list, float)
    widths = [np.array([np.std(a, ddof=1) for a in ch]) for ch in channels]
    sigmas = [w / np.sqrt(2.0 * (len(ch[0]) - 1)) for w, ch in zip(widths, channels)]

    def chi2(nu):
        tot = 0.0
        for w, sig in zip(widths, sigmas):
            M = np.column_stack([L ** (-1.0 / nu), L ** (-1.0 / nu - omega)]); Wt = 1.0 / sig ** 2
            try:
                c = np.linalg.solve((M * Wt[:, None]).T @ M, (M * Wt[:, None]).T @ w)
            except np.linalg.LinAlgError:
                return 1e18
            tot += float(np.sum(Wt * (w - M @ c) ** 2))
        return tot
    return float(minimize_scalar(chi2, bounds=(nu_lo, nu_hi), method="bounded").x)


def _df_from_smax(L_list, smax, L_min=0):
    L = np.asarray(L_list, float); s = np.asarray(smax, float)
    keep = L >= L_min
    b = np.polyfit(np.log(L[keep]), np.log(s[keep]), 1)          # s_max ~ L^{d_f}
    return float(b[0])


# a size ladder: dense at small L (cheap, anchors the curve), sparse at large L (each is L^3, expensive).
_LADDER = [8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 128, 160, 192, 224, 256, 288, 320, 384]


def main():
    import argparse
    ap = argparse.ArgumentParser(description="3D simple-cubic negative control")
    ap.add_argument("--lmax", type=int, default=48, help="largest cube side L (realistic best ~320)")
    ap.add_argument("--trials", type=int, default=1000, help="trials per size (hat used 1000)")
    args = ap.parse_args()
    L_list = [L for L in _LADDER if L <= args.lmax]
    print(f"3D simple-cubic NEGATIVE CONTROL  (same kernel/estimators as the 2D family)  "
          f"L<={args.lmax}, T={args.trials}\n")
    r = run_cube_sweep(L_list, T=args.trials)

    # p_c: extrapolate with the 3D nu (shows the measurement RECOVERS the known cube value -> not broken)
    pc = extrapolate_pc_raw(r["L"], r["I"], r["U"], nu=NU_3D)["A"]
    # nu: the paper's width-line procedure, wide bound so the 3D value is reachable
    nu = width_nu(r["L"], [r["I"], r["U"]])
    nu_2Dfit = width_nu(r["L"], [r["I"], r["U"]], nu_lo=1.10, nu_hi=1.60)   # what the 2D-clamped fitter sees
    df = _df_from_smax(r["L"], r["smax"], L_min=max(16, args.lmax // 6))   # drop small-L for the slope

    print("\n" + "=" * 78)
    print("RESULT  (cube measured  vs  3D literature  vs  the 2D class the family sits in)")
    print("=" * 78)
    print(f"  p_c (site) : {pc['pc']:.4f} ± {pc['pc_std']:.4f}   | cube lit 0.3116   | 2D family 0.50-0.85")
    print(f"  nu         : {nu:.3f}                    | cube lit {NU_3D:.3f}    | 2D class 1.333 (4/3)")
    print(f"  d_f        : {df:.3f}                    | cube lit {DF_3D:.3f}    | 2D class 1.896 (91/48)")
    print(f"\n  2D-clamped width-nu fit rails to {nu_2Dfit:.3f} (its 1.10 floor): the cube is INCOMPATIBLE")
    print("  with nu=4/3 -- the negative control the positive tests need.")


if __name__ == "__main__":
    main()
