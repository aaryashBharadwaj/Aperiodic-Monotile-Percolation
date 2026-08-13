import numpy as np
from scipy import stats

# results analysis for the percolation
# kept seperate from the actual engine


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
# We measure p_c by crossing a SQUARE window, and "left-right" vs "top-bottom" (and the square aspect ratio) are arbitrary conventions. 
# If P(LR) != P(TB) the reported threshold would depend on that arbitrary choice, and the direction-averaged estimator p_A would be blending two different quantities 
# Testing p_R - p_D -> 0 confirms the choice does not bias p_c
def isotropy_test(L_list, pR, pD, nu=4.0/3.0, confidence=0.95, L_min=50):
    # p_R (horizontal) and p_D (vertical) crossing fractions are recorded in the same sweep
    # (paired). We extrapolate their difference to L -> infinity; consistency with zero means
    # the reported p_c is independent of the arbitrary spanning-direction and square choice,
    # which justifies the averaged estimator p_A.
    L = np.asarray(L_list, dtype=float)
    pR = [np.asarray(r, dtype=float) for r in pR]
    pD = [np.asarray(r, dtype=float) for r in pD]
    if L_min is not None:
        # keep filters small L's to avoid finite size effects
        keep = L >= L_min
        L = L[keep]
        pR = [r for r, k in zip(pR, keep) if k]
        pD = [r for r, k in zip(pD, keep) if k]
    n = len(L)
    x = L ** (-1.0 / nu)
    # subtract the differences between rightwards and downwards
    # doesn't subtract means because each trial shares it's random generation, so this is less noisy
    d_mean = np.array([(rR - rD).mean() for rR, rD in zip(pR, pD)])
    d_se   = np.array([(rR - rD).std(ddof=1) / np.sqrt(len(rR)) for rR, rD in zip(pR, pD)])
    # Identical machinery to extrapolate_pc_raw
    # The discarded return values are the slope and the standard errors, which aren't needed
    d_inf, _, _, _, d_ci = _wls_fit(x, d_mean, d_se, n, confidence)
    # Does the confidence interval straddle zero? If yes, the extrapolated difference is consistent with zero
    # If the interval sits entirely above or below zero, the two directions genuinely differ and p_A would be blending two different things
    unbiased = d_ci[0] <= 0.0 <= d_ci[1]
    print(f"[Direction bias] p_R - p_D (L->inf) = {d_inf:+.6f}   {int(confidence*100)}% CI "
          f"[{d_ci[0]:+.6f}, {d_ci[1]:+.6f}]  ->  {'no directional bias' if unbiased else 'DIRECTION-BIASED'}")
    return d_inf, d_ci, unbiased


# ---- Largest-cluster pass: the fractal dimension d_f from the incipient-infinite-cluster size ----
def fit_exponents(L_list, smax, B=200, seed=17):
    rng = np.random.default_rng(seed)
    L = np.asarray(L_list, float); logL = np.log(L)
    # smax is a list of 'largest clusters' for each trial of a given L
    # we average this to get an average largest cluster per L
    smax = [np.asarray(s, float) for s in smax]
    slope = lambda means: float(np.polyfit(logL, np.log(means), 1)[0])
    d_f = slope([s.mean() for s in smax])
    # run a bootstrap for every L because the data is skewed
    # So we randomly select a number of clusters equal to the number of trials, with repetition
    # We use this to approximate the distribution and pick the 95% interval to get a distribution
    boot = [slope([s[rng.integers(0, len(s), len(s))].mean() for s in smax]) for _ in range(B)]
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    # d is the spatial dimension, this is 2D
    d = 2.0
    # return the other universality constants derived by the ones we have
    # direct estimators for γ/ν and τ are biased by open boundaries, so measuring them would give worse numbers than deriving them
    return {"d_f": d_f, "d_f_ci": ci,
            "hyperscaling": {"tau": 1 + d / d_f, "gamma_nu": 2 * d_f - d, "beta_nu": d - d_f}}


# ---- correlation-length exponent nu from the transition width (omega is an INPUT, not a fit) ----
def _width_nu(L, widths, sigmas, omega):
    """The nu minimising the summed weighted residual of  std(onset) = A L^{-1/nu} (1 + B L^{-omega})
    across one or more channels, with omega held fixed and the amplitudes (A, A*B) profiled out linearly
    per channel. Several channels (site/bond, I/U) share nu and omega but keep independent amplitudes,
    which is what tightens nu."""
    from scipy.optimize import minimize_scalar
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
    return float(minimize_scalar(chi2, bounds=(1.10, 1.60), method="bounded").x)


def fit_nu(L_list, channels, omega_lo=0.5, omega_hi=1.5, n_omega=13, B=200, seed=17, L_min=50.0):
    """nu from the finite-size transition width, std(onset) ~ L^{-1/nu}. The correction-to-scaling
    exponent omega is NOT measurable at accessible sizes -- the joint (nu, omega) fit is degenerate and
    the direct omega observables are swamped by noise (needs Ziff-scale statistics) -- so we do NOT fit
    it. Instead nu is extracted for every omega across the band [omega_lo, omega_hi] (the range of
    correction exponents reported across 2D percolation systems) and the resulting BAND is the result:
    the point is that nu stays consistent with 4/3 for the whole band, so the conclusion does not depend
    on omega. `channels` is a list of per-L onset-array lists (e.g. site-I/U and bond-I/U) that share nu
    and omega but have independent amplitudes -- a joint fit. Returns the nu band + bootstrap CIs at the
    band ends. Validate by running it on exact-nu=4/3 lattices (square/triangular): the same procedure
    must return ~4/3 there. L_min drops the smallest sizes (biggest corrections); default 50 matches the
    p_c-extrapolation cutoff."""
    L_all = np.asarray(L_list, float)
    keep = L_all >= L_min
    L = L_all[keep]
    ch_arrs = [[np.asarray(a, float) for a, k in zip(ch, keep) if k] for ch in channels]
    Ts = [len(ch[0]) for ch in ch_arrs]
    widths = [np.array([a.std(ddof=1) for a in ch]) for ch in ch_arrs]
    sigmas = [w / np.sqrt(2.0 * (T - 1)) for w, T in zip(widths, Ts)]
    omegas = np.linspace(omega_lo, omega_hi, n_omega)
    nu_by_omega = {round(float(om), 4): _width_nu(L, widths, sigmas, om) for om in omegas}
    vals = np.array(list(nu_by_omega.values()))
    rng = np.random.default_rng(seed)
    def boot_ci(om):
        out = [_width_nu(L, [np.array([a[rng.integers(0, len(a), len(a))].std(ddof=1) for a in ch])
                             for ch in ch_arrs], sigmas, om) for _ in range(B)]
        return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))
    return {"nu_band": (float(vals.min()), float(vals.max())),
            "omega_band": (omega_lo, omega_hi),
            "nu_by_omega": nu_by_omega,
            "nu_at_omega_lo": nu_by_omega[round(float(omegas[0]), 4)], "ci_at_omega_lo": boot_ci(omega_lo),
            "nu_at_omega_hi": nu_by_omega[round(float(omegas[-1]), 4)], "ci_at_omega_hi": boot_ci(omega_hi)}
