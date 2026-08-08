"""Results analysis for the percolation engine: finite-size-scaling fits that turn the raw per-L
trial arrays (produced by engine.percolation's *_par classes) into physical numbers.

Kept separate from engine.percolation so that module is PURE SIMULATION (Newman-Ziff kernels +
trial classes) and this one is PURE ANALYSIS (weighted least squares, p_c extrapolation, the
direction-bias check). A runner/backend imports the simulation from percolation and the fits here.

  _wls_fit(...)            -> weighted straight-line fit (the shared FSS primitive)
  extrapolate_pc_raw(...)  -> p_c (I/U/A estimators) from the crossing trials, nu fixed at 4/3
  isotropy_test(...)       -> direction-bias check that validates the averaged p_A estimator
"""
import numpy as np
from scipy import stats


# wls is 'weighted least squares'
def _wls_fit(x, y, sigma, n, confidence):
    # the weight takes the standard error and weighs each result inversely proportional to that
    # a point with half the error bar counts four times as much
    # thus the larger the system, the more confident the result
    w = 1.0 / sigma**2
    W = np.diag(w)
    # Design matrix: column of 1s (the intercept, pc) and column of x (the slope, A).
    X = np.column_stack([np.ones_like(x), x])


    # Normal equations for weighted least squares: solve (X^T W X) beta = X^T W y.
    # For Gaussian errors of known variance this is also the maximum-likelihood line.
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    coeffs = np.linalg.solve(XtWX, XtWy)

    # We do NOT rescale it by reduced chi-squared: with ~1000 trials per point the sigmas are well determined, so treating them as known is the honest choice.
    cov = np.linalg.inv(XtWX)
    pc_hat = coeffs[0]
    A_hat  = coeffs[1]
    pc_std = np.sqrt(cov[0, 0])
    A_std  = np.sqrt(cov[1, 1])

    # Use a t-distribution
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 2)
    pc_ci  = (pc_hat - t_crit * pc_std, pc_hat + t_crit * pc_std)

    return pc_hat, A_hat, pc_std, A_std, pc_ci


def extrapolate_pc_raw(L_list, trials_results_I, trials_results_U, nu=4/3, confidence=0.95):
    L  = np.array(L_list, dtype=float)
    n  = len(L)
    # The finite estimates approach the true p_c using a critical exponent nu = 4/3, so we use this to estimate p_c
    x  = L ** (-1.0 / nu)

    mean_I, mean_U, mean_A = [], [], []
    sigma_I, sigma_U, sigma_A = [], [], []

    for rI, rU in zip(trials_results_I, trials_results_U):
        rI, rU = np.array(rI), np.array(rU)
        T = len(rI)

        mI, mU = rI.mean(), rU.mean()
        sI, sU = rI.std(ddof=1), rU.std(ddof=1)
        # I and U come from the SAME sweeps, so they are correlated.
        cov_IU = np.cov(rI, rU)[0, 1]

        # stores the union, intersection and average estimators
        mean_I.append(mI)
        mean_U.append(mU)
        mean_A.append(0.5 * (mI + mU))

        sigma_I.append(sI / np.sqrt(T))
        sigma_U.append(sU / np.sqrt(T))
        # Var(1/2 (I+U)) = 1/4 (Var I + Var U + 2 Cov(I,U)). The covariance term is not optional:
        # I and U are measured in the same sweep and are strongly positively correlated, so dropping it would understate sigma_A substantially
        # For independent directions, cov ~ 0 and this reduces to the usual formula
        sigma_A.append(0.5 * np.sqrt(sI**2/T + sU**2/T + 2*cov_IU/T))

    results = {}
    for label, y, sigma in [
        ('I', np.array(mean_I),  np.array(sigma_I)),
        ('U', np.array(mean_U),  np.array(sigma_U)),
        ('A', np.array(mean_A),  np.array(sigma_A)),
    ]:
        pc_hat, A_hat, pc_std, A_std, pc_ci = _wls_fit(x, y, sigma, n, confidence)
        results[label] = {
            # A is the fitted slope (scaling amplitude); reported for completeness.
            'pc'    : pc_hat,
            'A'     : A_hat,
            'pc_std': pc_std,
            'pc_ci' : pc_ci,
            'A_std' : A_std,
        }
        print(f"[{label}] Extrapolated p_c     : {pc_hat:.6f} ± {pc_std:.6f}")
        print(f"[{label}] Scaling amplitude A  : {A_hat:.6f} ± {A_std:.6f}")
        print(f"[{label}] {int(confidence*100)}% CI             : [{pc_ci[0]:.6f}, {pc_ci[1]:.6f}]")
        print()

    return results

# Direction-bias check that validates the estimator, NOT a universality claim.
# We measure p_c by crossing a SQUARE window, and "left-right" vs "top-bottom" (and the square
# aspect ratio) are arbitrary conventions. If P(LR) != P(TB) the reported threshold would depend
# on that arbitrary choice, and the direction-averaged estimator p_A would be blending two
# different quantities. Testing p_R - p_D -> 0 confirms the choice does not bias p_c. (This is a
# statement about the measurement we ran, not about the geometry/universality of the tiling.)
def isotropy_test(L_list, pR, pD, nu=4.0/3.0, confidence=0.95, L_min=50):
    # p_R (horizontal) and p_D (vertical) crossing fractions are recorded in the same sweep
    # (paired). We extrapolate their difference to L -> infinity; consistency with zero means
    # the reported p_c is independent of the arbitrary spanning-direction and square choice,
    # which justifies the averaged estimator p_A.
    L = np.asarray(L_list, dtype=float)
    pR = [np.asarray(r, dtype=float) for r in pR]
    pD = [np.asarray(r, dtype=float) for r in pD]
    if L_min is not None:
        keep = L >= L_min
        L = L[keep]
        pR = [r for r, k in zip(pR, keep) if k]
        pD = [r for r, k in zip(pD, keep) if k]
    n = len(L)
    x = L ** (-1.0 / nu)
    d_mean = np.array([(rR - rD).mean() for rR, rD in zip(pR, pD)])
    d_se   = np.array([(rR - rD).std(ddof=1) / np.sqrt(len(rR)) for rR, rD in zip(pR, pD)])
    d_inf, _, _, _, d_ci = _wls_fit(x, d_mean, d_se, n, confidence)
    unbiased = d_ci[0] <= 0.0 <= d_ci[1]
    print(f"[Direction bias] p_R - p_D (L->inf) = {d_inf:+.6f}   {int(confidence*100)}% CI "
          f"[{d_ci[0]:+.6f}, {d_ci[1]:+.6f}]  ->  {'no directional bias' if unbiased else 'DIRECTION-BIASED'}")
    return d_inf, d_ci, unbiased


# ---- Block B: the fractal dimension d_f from the incipient-infinite-cluster size ----
def fit_exponents(L_list, smax, B=200, seed=17):
    """d_f from <s_max> ~ L^{d_f} (largest cluster at first-spanning), with a bootstrap-over-trials CI.
    The other static exponents are NOT measured -- they follow from d_f by hyperscaling (tau = 1 + d/d_f,
    gamma/nu = 2 d_f - d, beta/nu = d - d_f) and are returned as those consequences. (Direct cluster-
    moment estimators of gamma/nu and tau are open-boundary biased, so we deliberately don't record
    or fit them; d_f + nu are the two independent exponents that fix the class.)"""
    rng = np.random.default_rng(seed)
    L = np.asarray(L_list, float); logL = np.log(L)
    smax = [np.asarray(s, float) for s in smax]
    slope = lambda means: float(np.polyfit(logL, np.log(means), 1)[0])
    d_f = slope([s.mean() for s in smax])
    boot = [slope([s[rng.integers(0, len(s), len(s))].mean() for s in smax]) for _ in range(B)]
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    d = 2.0
    return {"d_f": d_f, "d_f_ci": ci,
            "hyperscaling": {"tau": 1 + d / d_f, "gamma_nu": 2 * d_f - d, "beta_nu": d - d_f}}
