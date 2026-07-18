# 🟡 Statistics — the functions that produce your numbers  (`percolation.py`)

**This is the only section where your judgement is the sole safeguard.** These four functions
are novel and cannot be checked against any prior baseline — a wrong formula here is a wrong
published value. For each: read the INTENT, decide if the *maths* is right, then open the cited
lines and confirm the *code* matches. There is intentionally no code pasted here — compare
against the live file.

---

## 1. `_wls_fit(x, y, sigma, n, confidence)`  — `percolation.py:180`

**What it is:** a weighted straight-line fit `y = pc + A·x`, weights = 1/σ², returning the
intercept `pc`, slope `A`, their standard errors, and a Student-t confidence interval on `pc`.
Every p_c, ν and isotropy number flows through this.

**INTENT (pseudocode):**
```
w        = 1 / sigma^2                      # each point weighted by inverse variance
X        = columns [ 1 , x ]                # design matrix for intercept + slope
beta     = solve( Xᵀ W X , Xᵀ W y )         # weighted normal equations  (β = [pc, A])
cov      = inverse( Xᵀ W X )                # covariance of the fitted params
pc, A    = beta[0], beta[1]
pc_std   = sqrt(cov[0,0]);  A_std = sqrt(cov[1,1])
t_crit   = Student_t_quantile( (1+confidence)/2 , dof = n-2 )
pc_CI    = pc ± t_crit * pc_std
return pc, A, pc_std, A_std, pc_CI
```

**CHECK — is the maths right?**
- WLS with weights 1/σ² is the maximum-likelihood line fit for independent Gaussian errors of
  known variance σ². ✔ standard.
- `cov = (XᵀWX)⁻¹` is the exact parameter covariance when weights are the true inverse
  variances. ✔
- t-quantile with `dof = n−2` (2 params fitted). ✔ correct dof.
- **The one judgement call:** this treats σ_i as *known* (not estimated), so it does NOT inflate
  the CI by the reduced-χ². That is the correct choice *because* your σ_i come from ~1000 trials
  (well-determined). Confirm you are comfortable stating that.

**CHECK — does the code match?** Confirm at `percolation.py:180`: `w = 1/sigma**2`,
`X = column_stack([ones, x])`, `solve(X.T@W@X, X.T@W@y)`, `cov = inv(X.T@W@X)`, intercept =
`coeffs[0]`, `t.ppf((1+confidence)/2, df=n-2)`.

---

## 2. `extrapolate_pc_raw(L_list, raw_I, raw_U, nu=4/3, confidence)` — `percolation.py:206`

**What it is:** finite-size-scaling extrapolation of p_c to L→∞ for the three estimators
Intersection (I), Union (U), Average (A = ½(I+U)), using x = L^(−1/ν).

**INTENT (pseudocode):**
```
x = L ^ (-1/nu)                             # FSS abscissa; nu=4/3 -> L^(-3/4)
for each system size L_k:
    rI, rU = per-trial crossing fractions (I and U)   ; T = #trials
    mI, mU = mean(rI), mean(rU)
    sI, sU = std(rI, ddof=1), std(rU, ddof=1)         # sample SDs
    cov    = Cov(rI, rU)                               # I and U are PAIRED per trial
    mean_A = 0.5*(mI + mU)
    sigma_I = sI/sqrt(T)                               # standard error of the mean
    sigma_U = sU/sqrt(T)
    sigma_A = 0.5 * sqrt( sI^2/T + sU^2/T + 2*cov/T )  # SE of the average, WITH covariance
for estimator in {I, U, A}:
    pc, A, pc_std, pc_CI = _wls_fit(x, mean_estimator, sigma_estimator, n)
    report pc ± pc_std and the CI
```

**CHECK — is the maths right?**
- x = L^(−1/ν) is the standard FSS variable; the intercept at x=0 is p_c(∞). ✔
- `sigma = s/√T` is the standard error of the mean → correct per-point weight for the fit. ✔
- **The subtle one — the Average's error bar:** Var(½(I+U)) = ¼(Var I + Var U + 2·Cov(I,U)).
  The code uses `0.5*sqrt(sI²/T + sU²/T + 2·cov/T)`. Note `cov` here is `np.cov(rI,rU)[0,1]`,
  the sample covariance (per-trial), and dividing by T turns each term into a variance-of-mean.
  ✔ this is the correct propagated SE for the mean of A **provided I and U are paired trials**.
  → **Judgement call for you:** confirm rI and rU at a given L are the SAME trials in order
  (paired). If they were independent, cov≈0 and this reduces to the independent formula — still
  valid, just no covariance benefit. (In the parallel pipeline I and U use *different* seed
  streams, so cov≈0 in practice; the formula stays correct either way.)

**CHECK — does the code match?** `percolation.py:206–250`. Confirm `x = L**(-1.0/nu)`, the
per-L mean/std/cov block, `sigma_A = 0.5*np.sqrt(sI**2/T + sU**2/T + 2*cov_IU/T)`, and three
`_wls_fit` calls.

---

## 3. `nu_width_line(L_list, raw_I, raw_U, confidence, L_min=50)` — `percolation.py:253`

**What it is:** measures ν from the decay of the crossing-fraction *width* (spread), using
w(L) ∝ L^(−1/ν) ⇒ slope of log w vs log L = −1/ν.

**INTENT (pseudocode):**
```
keep only sizes with L >= L_min             # drop the most finite-size-biased small systems
raw_A = elementwise 0.5*(rI + rU)           # Average channel
x = log(L)
for channel in {I, U, A}:
    w  = std(crossing fractions, ddof=1) at each L        # the width
    T  = #trials at each L
    se = 1 / sqrt(2*(T-1))                                # SE of log(sample SD), large-T
    slope, slope_std = weighted line fit of ( log w  vs  x )   [ via _wls_fit ]
    nu     = -1 / slope
    nu_err = t_crit * slope_std / slope^2                 # delta method: d(-1/s)/ds = 1/s^2
    report nu ± nu_err
```

**CHECK — is the maths right?**
- w(L) ∝ L^(−1/ν): the width of the pseudo-critical distribution is a standard, p_c-independent
  handle on ν. ✔ (This is *why* you fit width and not the mean drift.)
- **`se = 1/√(2(T−1))`**: this is the large-sample SE of **log(sample standard deviation)** for
  Gaussian data (Var(log s) ≈ 1/(2(T−1))). ✔ This is the right weight because you fit log w, not
  w. Worth confirming against a stats reference — it is the one non-obvious formula.
- **Delta method for ν error:** ν = −1/slope ⇒ dν/d(slope) = 1/slope² ⇒ σ_ν = σ_slope/slope².
  The code multiplies by `t_crit` to make it a CI half-width. ✔
- `L_min=50` is a documented modelling choice (drops small-L corrections). Confirm you are
  reporting it as such (this is exactly the convergence subtlety you already flagged).

**CHECK — does the code match?** `percolation.py:253–278`. Confirm `keep = L>=L_min`,
`w = std(ddof=1)`, `se = 1/sqrt(2*(T-1))`, fit of `log(w)` vs `log(L)`, `nu = -1/slope`,
`nu_err = t_crit*slope_std/slope**2`.

---

## 4. `isotropy_test(L_list, pR, pD, nu=4/3, confidence, L_min=50)` — `percolation.py:281`

**What it is:** tests directional isotropy by extrapolating the mean difference of the
horizontal (p_R) and vertical (p_D) crossing fractions to L→∞; consistency with 0 justifies the
averaged estimator.

**INTENT (pseudocode):**
```
keep only L >= L_min
x = L ^ (-1/nu)
for each L:
    d_mean = mean( pR - pD )                     # PAIRED difference, same trials
    d_se   = std( pR - pD , ddof=1) / sqrt(T)    # SE of the mean difference
d_inf, d_CI = _wls_fit(x, d_mean, d_se, n)       # extrapolate the difference to L->inf
isotropic = ( d_CI contains 0 )
report d_inf and d_CI and the verdict
```

**CHECK — is the maths right?**
- Using the **paired** per-trial difference pR−pD (not the difference of independent means) is
  correct and is what makes d_se small/valid — p_R and p_D come from the SAME sweep. ✔ Confirm
  in the runner that pR and pD are recorded from one sweep (they are: `statsSI.pR/.pD`).
- Extrapolating the difference to x=0 and checking the CI brackets 0 is a clean frequentist test
  of "no directional bias at L→∞". ✔
- Same L^(−1/ν) FSS variable and same `_wls_fit` machinery as p_c. ✔ consistent.

**CHECK — does the code match?** `percolation.py:281–301`. Confirm `d_mean = (rR-rD).mean()`,
`d_se = (rR-rD).std(ddof=1)/sqrt(len)`, `_wls_fit`, `isotropic = d_ci[0] <= 0 <= d_ci[1]`.

---

## Sign-off checklist for this section
- [ ] `_wls_fit`: WLS + (XᵀWX)⁻¹ covariance + t(n−2) CI — maths agreed, code matches.
- [ ] `extrapolate_pc_raw`: L^(−1/ν) FSS, SE=s/√T, Average SE with covariance — agreed, matches.
- [ ] `nu_width_line`: w∝L^(−1/ν), se=1/√(2(T−1)), ν=−1/slope, delta-method error — agreed, matches.
- [ ] `isotropy_test`: paired pR−pD extrapolated to 0 — agreed, matches.
