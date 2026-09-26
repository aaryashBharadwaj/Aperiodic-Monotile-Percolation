# Methods paragraph — the adaptive low-L cutoff (pasteable)

**Placement:** Methods / finite-size-scaling subsection, where the leading-order extrapolation
`p_c(L) = p_c + A L^{-1/nu}` is introduced. Goes right after the fit form, before the error model.

**Companion figure (recommended):** the hat-dual estimator vs L with error bars, the fitted line,
and the discarded small-L points shaded — makes the "measured against the error bars" idea visual.

---

## Draft prose

Finite-size corrections to scaling bend the per-size estimators $p_c(L)$ away from the leading-order
form $p_c(L) = p_c + A\,L^{-1/\nu}$ at small $L$; retaining these points in the fit biases the
extrapolated intercept. Rather than impose an arbitrary minimum size, we fix the fit window with a
$\chi^2$ goodness-of-fit test, so the cutoff is set by the data and their uncertainties rather than by
hand.

For a candidate window we fit the leading-order line by weighted least squares and form the reduced
chi-square $\chi^2_\nu = \mathrm{dof}^{-1}\sum_i [p_c(L_i) - \mathrm{fit}(L_i)]^2/\sigma_i^2$, where
$\sigma_i$ is the statistical uncertainty of the $i$-th estimator. Each residual is thereby measured in
units of that point's own error bar: a straight line that genuinely describes the data yields residuals
of order unity and $\chi^2_\nu \approx 1$, whereas points still carrying corrections-to-scaling lie many
error bars from any straight line and inflate $\chi^2_\nu$. We reject the leading-order fit when its
$\chi^2$ exceeds the 95\% quantile for the relevant number of degrees of freedom. Beginning from the
full range, we discard the smallest available size and re-test, and retain the largest window—i.e. the
lowest cutoff $L_{\min}$, for maximal lever arm—for which the leading-order line is not rejected at this
level. The only tunable input is the confidence level; there is no adjustable correction exponent and no
hand-chosen minimum size.

The criterion is self-adjusting. For the exactly-known lattices and the direct-graph tilings, whose
estimators track the leading-order line to within their error bars across essentially the whole range,
no points are discarded and the cutoff is inert. For the sparser dual graphs, where corrections persist
to larger $L$, the smallest sizes sit tens of error bars from the line and are removed until the
remainder is consistent with linearity—for the hat dual, for example, the retained window is
$L \ge 140$, at which the smallest included size lies within $\sim\!1\sigma$ of the fitted line while the
discarded $L=20$ point lay $\sim\!19\sigma$ away.

---

## One-line version (if space is tight, for a caption or footnote)

> The low-$L$ cutoff is not chosen by hand: we keep the largest fit window for which the leading-order
> line is not rejected by a $\chi^2$ goodness-of-fit test at 95\% confidence, so points are dropped only
> where their departure from linearity exceeds what their error bars allow.

---

# Methods paragraph — the averaged estimator (pasteable)

**Placement:** where the Intersection/Union crossing criteria are defined, before the extrapolation.
**Why it's needed:** explains why we report $p_A$, why the FSS figure draws the line only for the
average, and why Intersection can be strongly curved on the duals without harming the result.

## Draft prose

For each size $L$ we record two spanning criteria per realisation: the Intersection onset $p_I$, the
density at which the patch first spans in *both* the horizontal and vertical directions, and the Union
onset $p_U$, at which it spans in *either*. With $p_R$ and $p_D$ the two directional onsets,
$p_I = \max(p_R,p_D)$ and $p_U = \min(p_R,p_D)$, so $p_I$ approaches $p_c$ from above and $p_U$ from
below. We report the direction-averaged estimator
$p_A = \tfrac12(p_I + p_U) = \tfrac12(p_R + p_D)$.

The average is preferred because it cancels the leading finite-size correction: the Intersection and
Union onsets carry corrections of opposite sign, which largely cancel in the mean, so $p_A$ is far
straighter in $L^{-1/\nu}$ than either criterion alone and admits the earliest cutoff. We do *not*
extrapolate $p_I$ and $p_U$ separately and average the intercepts; we form $p_A$ at each size and
extrapolate it directly. The individual criteria are retained only as a convergence diagnostic — $p_I$
and $p_U$ must bracket $p_A$ and close onto the same intercept as $L\to\infty$. On the sparser dual
graphs the Intersection onset is strongly curved and its own leading-order extrapolation is poorly
constrained; this is expected and harmless, since its information enters the reported value through
$p_A$, whose extrapolation is well determined. (Fitting the curved Intersection onset with an explicit
correction-to-scaling term instead recovers all of its sizes and reproduces $p_A$, confirming the two
criteria are consistent.)

## Figure-caption note (for the FSS figure)

> Filled points enter the reported fit; open points lie below the cutoff and are discarded. The
> extrapolation line and $L\to\infty$ intercept are drawn for the reported average estimator $p_A$; the
> Intersection ($\bullet$) and Union ($\blacksquare$) onsets are shown as raw data to confirm they
> bracket $p_A$ and converge to the same intercept. A single cutoff (that of $p_A$) marks the discarded
> region for all three, since $p_A$ is the extrapolated quantity.

---

# Methods paragraph — direction bias in the error budget (pasteable)

**Placement:** with the error model, right after the isotropy test is introduced.
**Why it's needed:** the isotropy test must have a *consequence*; this states it.

## Draft prose

Because $p_A = \tfrac12(p_R + p_D)$ averages the two spanning directions, a difference between them in
the $L\to\infty$ limit is a systematic on the reported value: $p_A$ sits $|p_R - p_D|/2$ from either
directional threshold. We therefore fold this distance into the estimator's uncertainty whenever the
isotropy test rejects $p_R = p_D$ at 95\% confidence, adding it in quadrature to the statistical and
fit-window terms. Because only the magnitude of the difference is measured and not the sign of the
residual bias on the average, this can only widen the interval, never shift the central value. The
correction is null for every direct-graph tiling (their directions agree within error) and is
appreciable only for the sparser dual graphs; for the hat and spectre duals it is the dominant
systematic, and the quoted errors reflect it.
