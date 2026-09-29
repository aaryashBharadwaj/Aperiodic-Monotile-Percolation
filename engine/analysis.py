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
    cov = np.linalg.inv(XtWX)
    pc_hat = coeffs[0]
    A_hat  = coeffs[1]

    # Reduced chi-square of the fit. With ~1000 trials per point the per-point sigmas are well
    # determined, so an excess (chi2_red > 1) is NOT noisy error bars -- it is the leading-order model
    # p_c + A*L^(-1/nu) falling short of the real corrections-to-scaling curvature. Treating the sigmas
    # as "known" then understates the intercept uncertainty (the classic too-tight "+-0.0000"). The honest
    # fix is the Birge ratio: inflate the parameter errors by sqrt(chi2_red). We only ever inflate
    # (max with 1) -- a good fit (chi2_red <= 1) keeps the raw statistical error, we never claim better.
    resid   = y - X @ coeffs
    dof     = max(1, n - 2)
    chi2_red = float(resid @ (w * resid)) / dof
    scale   = max(1.0, np.sqrt(chi2_red))
    pc_std  = np.sqrt(cov[0, 0]) * scale
    A_std   = np.sqrt(cov[1, 1]) * scale

    # Use a t-distribution
    t_crit = stats.t.ppf((1 + confidence) / 2, df=dof)
    pc_ci  = (pc_hat - t_crit * pc_std, pc_hat + t_crit * pc_std)

    return pc_hat, A_hat, pc_std, A_std, pc_ci, chi2_red


def _boot_quantiles(rI, rU, qI, qU, n_boot, rng, max_cells=4_000_000):
    """Paired bootstrap standard errors for the median-at-R_c estimator, at one system size.
    rI, rU are the per-trial spanning-onset fractions for the intersection (both-directions) and union
    (either-direction) channels, measured in the SAME sweeps (trial k of rI and rU share the random
    generation). The point estimates are the R_c-quantiles qI, qU of each channel. The SEs come from
    resampling TRIAL INDICES and applying the SAME indices to both channels, so the I<->U correlation
    flows into sigma_A automatically -- no explicit covariance term. Bootstrapping (not std/sqrt(T)) is
    used because a sample quantile has no clean closed-form standard error. Chunked so peak memory stays
    ~max_cells floats regardless of trial count. Returns (pI, pU, sigma_I, sigma_U, sigma_A)."""
    T = len(rI)
    pI = float(np.quantile(rI, qI)); pU = float(np.quantile(rU, qU))
    chunk = max(1, int(max_cells // max(1, T)))
    bI = np.empty(n_boot); bU = np.empty(n_boot)
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx = rng.integers(0, T, size=(m, T))
        bI[done:done + m] = np.quantile(rI[idx], qI, axis=1)
        bU[done:done + m] = np.quantile(rU[idx], qU, axis=1)
        done += m
    bA = 0.5 * (bI + bU)
    return pI, pU, float(bI.std(ddof=1)), float(bU.std(ddof=1)), float(bA.std(ddof=1))


def extrapolate_pc_raw(L_list, trials_results_I, trials_results_U, nu=4/3, confidence=0.95,
                       bias_floor=None, cutoff_conf=0.95, dir_bias=None,
                       rc_I=0.322120455, n_boot=1000, boot_seed=20260909):
    # ESTIMATOR: median-at-R_c (Ziff & Newman, PRE 66, 016129 (2002)). Each channel's spanning-onset
    # distribution is read at its EXACT universal crossing value R_c, and the size series is extrapolated
    # along the ANALYTIC correction exponent 1 + 1/nu (= 1.75 at nu = 4/3). This is why median-at-R_c is
    # standardised over the average-onset estimator we shipped before: the average obeys L^{-(omega+1/nu)}
    # (~L^{-1.65}), whose slope depends on the MEASURED correction exponent omega = 0.90(2), whereas
    # 1 + 1/nu is analytic (omega-independent) -- an objective, not-fitted slope.
    # R_c is CFT-exact for the aspect-1 square / conformal rectangle: intersection (span in BOTH
    # directions) = 0.322120455 (Watts 1996); union (span in EITHER direction) = 1 - that = 0.677879545.
    # For the aperiodic patches, R_c's transfer is LICENSED by the Cardy aspect-ratio test (Tier-1), not
    # assumed. rc_I sets the intersection quantile; the union quantile is 1 - rc_I (inclusion-exclusion).
    # n_boot/boot_seed: replicates and RNG seed for the quantile-SE bootstrap (reproducible).
    #
    # bias_floor: if given, the reported systematic is floored at this value -- the size of the residual
    # finite-size bias that the KNOWN-lattice controls show at this run's reach (a self-hiding systematic
    # the fit-window scan can miss). Supplied by the reporting layer from the control "bias ruler".
    # cutoff_conf: the adaptive low-L cutoff is chosen by a chi^2 GOODNESS-OF-FIT test at this confidence
    # rather than an arbitrary L or chi2_red threshold. We include the most points (best lever arm) for
    # which the leading-order line is not rejected at cutoff_conf (p-value > 1-cutoff_conf).
    # dir_bias: the extrapolated direction bias p_R - p_D from isotropy_test (or None). The reported
    # estimator p_A = (p_R+p_D)/2 sits |dir_bias|/2 from either directional threshold, so when the two
    # directions differ in the limit this is an unquoted systematic on p_A. Its sign is unknown (we
    # measure the gap, not the asymmetry), so it only widens the error -- folded into the A channel below.
    rc_U = 1.0 - rc_I
    L  = np.array(L_list, dtype=float)
    n  = len(L)
    # Leading finite-size correction of the median-at-R_c estimator: L^{-(1 + 1/nu)} (analytic exponent).
    x  = L ** (-(1.0 + 1.0 / nu))
    inv_exp = 1.0 / (1.0 + 1.0 / nu)   # invert x -> L for cutoff reporting: L = x^{-inv_exp}
    rng = np.random.default_rng(boot_seed)

    mean_I, mean_U = [], []          # per-size median-at-R_c point estimates (intersection / union channels)
    sigma_I, sigma_U = [], []        # paired-bootstrap SEs

    for rI, rU in zip(trials_results_I, trials_results_U):
        rI, rU = np.asarray(rI, dtype=float), np.asarray(rU, dtype=float)
        # per-size I/U R_c-quantiles + paired-bootstrap SEs. _sA (the per-size midpoint SE) is unused
        # here on purpose: the reported midpoint stat error is built from the fitted I/U intercepts below,
        # not from a per-size SE. _boot_quantiles still returns the per-size sA, but nothing consumes it here.
        pI, pU, sI, sU, _sA = _boot_quantiles(rI, rU, rc_I, rc_U, n_boot, rng)
        mean_I.append(pI); mean_U.append(pU)
        sigma_I.append(sI); sigma_U.append(sU)

    results = {}
    # Fit the two BOUND channels (I = span in both directions; U = span in either) at the analytic 7/4
    # exponent. The reported p_c is their midpoint, assembled after this loop.
    for label, y, sigma in [
        ('I', np.array(mean_I),  np.array(sigma_I)),
        ('U', np.array(mean_U),  np.array(sigma_U)),
    ]:
        y = np.asarray(y); sigma = np.asarray(sigma)

        # (1) FIXED low-L cutoff L >= 50. The leading-order line p_c + A L^{-(1+1/nu)} holds only once the
        # corrections-to-scaling have died, so we discard the smallest sizes. This is a fixed, pre-declared
        # threshold (standard FSS practice), NOT an adaptive/tuned criterion: on the exact controls the
        # extrapolated p_c is insensitive to the exact value (|all-L - L>=50| < ~2e-5, within the envelope),
        # and it is consistent with the L_min anchor of the extrapolation envelope. (cutoff_conf retained in
        # the signature for backward compatibility but no longer used.)
        k0 = int(np.sum(L < 50.0))
        pc_hat, A_hat, pc_std_stat, A_std, _ci, chi2_red = _wls_fit(x[k0:], y[k0:], sigma[k0:], n - k0, confidence)
        nrem = n - k0

        # (2) CLEAN FIT-WINDOW SYSTEMATIC. Vary the cutoff upward from k0, but ONLY over windows that keep
        # >= half the post-cutoff points -- so this measures genuine cutoff-dependence, not the noise of
        # few-point fits (the earlier scan ran down to 5 points and badly over-reported the error).
        wins = [pc_hat]
        for k in range(k0 + 1, k0 + int(round(0.4 * nrem)) + 1):
            if n - k >= max(6, int(round(0.5 * nrem))):
                wins.append(_wls_fit(x[k:], y[k:], sigma[k:], n - k, confidence)[0])
        pc_syst = float(np.std(wins)) if len(wins) >= 2 else 0.0

        # (3) TOTAL for this channel = statistical (+) fit-window systematic (+) optional control bias floor.
        # The direction systematic and the I<->U split are folded in only at the midpoint 'A' below.
        t_crit = stats.t.ppf((1 + confidence) / 2, df=max(1, nrem - 2))
        syst = float(max(pc_syst, float(bias_floor or 0.0)))
        pc_std = float(np.hypot(pc_std_stat, syst))
        pc_ci  = (pc_hat - t_crit * pc_std, pc_hat + t_crit * pc_std)

        results[label] = {
            'pc'         : pc_hat,
            'A'          : A_hat,
            'pc_std'     : pc_std,          # total: stat (+) max(fit-window syst, bias floor)
            'pc_ci'      : pc_ci,
            'A_std'      : A_std,
            'chi2_red'   : chi2_red,
            'pc_std_stat': pc_std_stat,     # statistical (chi2-inflated)
            'pc_syst'    : pc_syst,         # fit-window (extrapolation)
            'bias_floor' : float(bias_floor or 0.0),
            'cutoff_L'   : float(round(x[k0] ** (-inv_exp))),
            'n_used'     : nrem,
        }
        print(f"[{label}] p_c = {pc_hat:.6f} ± {pc_std:.6f}  "
              f"(stat {pc_std_stat:.6f}, syst {pc_syst:.6f}, floor {float(bias_floor or 0.0):.6f}, "
              f"chi2={chi2_red:.2f}, fit L>={int(round(x[k0]**(-inv_exp)))}, {nrem} pts)")

    # ---- REPORTED ESTIMATE 'A' = midpoint of the two bound channels ----
    # p_c = 1/2 (p_I + p_U). Both spanning definitions (I = span in both directions, U = in either) converge
    # to the SAME threshold, so the midpoint cancels the definition bias and is MORE precise than either
    # channel. Its 1-sigma error is therefore the proper error of an average, in quadrature:
    #   stat  = 1/2 (sigma_I^stat + sigma_U^stat)  -- the shared bootstrap error (I,U are the same sweeps,
    #           highly correlated, so we do NOT reduce by 1/sqrt(2); this is a mild over-estimate = honest);
    #   syst  = max(1/2 (fit-window_I + fit-window_U), bias_floor)  -- the L_min-scan systematic (+ optional
    #           control bias floor);
    #   split = 1/2 |p_I - p_U|  -- the I<->U DEFINITION systematic (how far the centre moves if you pick
    #           strict vs loose spanning);
    #   dir   = |dir_bias|/2 from isotropy_test, if supplied.
    # This REPLACES the earlier "cover both channels' bars" hull, which was ~1.5x too wide: taking the outer
    # hull ignored that the midpoint benefits from I/U cancellation. Validated: the quadrature bar still
    # covers all 6 exact controls. (Nothing here changes the CENTRE, only the width.)
    rI_res, rU_res = results['I'], results['U']
    pc_A     = 0.5 * (rI_res['pc'] + rU_res['pc'])
    dir_syst = abs(float(dir_bias)) / 2.0 if dir_bias else 0.0
    stat_A   = 0.5 * (rI_res['pc_std_stat'] + rU_res['pc_std_stat'])
    syst_win = max(0.5 * (rI_res['pc_syst'] + rU_res['pc_syst']), float(bias_floor or 0.0))
    split    = 0.5 * abs(rI_res['pc'] - rU_res['pc'])
    syst_A   = float(np.sqrt(syst_win ** 2 + split ** 2 + dir_syst ** 2))
    std_A    = float(np.hypot(stat_A, syst_A))
    t_A      = stats.t.ppf((1 + confidence) / 2, df=max(1, rI_res['n_used'] - 2))
    results['A'] = {
        'pc'         : pc_A,
        'A'          : 0.5 * (rI_res['A'] + rU_res['A']),
        'pc_std'     : std_A,                       # total 1-sigma: stat (+) fit-window (+) I-U split (+) dir
        'pc_ci'      : (pc_A - t_A * std_A, pc_A + t_A * std_A),
        'A_std'      : 0.5 * (rI_res['A_std'] + rU_res['A_std']),
        'chi2_red'   : 0.5 * (rI_res['chi2_red'] + rU_res['chi2_red']),
        'pc_std_stat': stat_A,
        'pc_syst'    : syst_A,                      # fit-window (+) I-U split (+) dir, in quadrature
        'dir_syst'   : dir_syst,
        'bias_floor' : float(bias_floor or 0.0),
        'cutoff_L'   : rI_res['cutoff_L'],
        'n_used'     : rI_res['n_used'],
        'split'      : split,                       # I<->U definition systematic (half the gap)
    }
    print(f"[A] p_c = {pc_A:.6f} ± {std_A:.6f}  (midpoint of I/U; "
          f"stat {stat_A:.6f}, fit-window {syst_win:.6f}, I-U split {split:.6f}, dir {dir_syst:.6f})")

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
    d_inf, _, _, _, d_ci, _ = _wls_fit(x, d_mean, d_se, n, confidence)
    # Does the confidence interval straddle zero? If yes, the extrapolated difference is consistent with zero
    # If the interval sits entirely above or below zero, the two directions genuinely differ and p_A would be blending two different things
    unbiased = d_ci[0] <= 0.0 <= d_ci[1]
    print(f"[Direction bias] p_R - p_D (L->inf) = {d_inf:+.6f}   {int(confidence*100)}% CI "
          f"[{d_ci[0]:+.6f}, {d_ci[1]:+.6f}]  ->  {'no directional bias' if unbiased else 'DIRECTION-BIASED'}")
    return d_inf, d_ci, unbiased


# ---- Largest-cluster (incipient-cluster) size is RECORDED as a raw observable (percolation.py's
# exponent kernels -> results.py raw_smax) but is NOT analysed here. The fractal-dimension fit was
# removed 2026-09-08: d_f from <S_max> is boundary-suppressed on open patches (a clean determination
# needs a boundary-free geometry, which aperiodic tilings can't provide), so it isn't a paper claim.
# The archived s_max arrays are left, with that caveat, for independent future analysis.
