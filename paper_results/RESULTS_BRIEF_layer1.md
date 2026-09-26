# Results-substitution brief — Layer 1 (numbers & plots)

**For a second chat that holds the PRE/REVTeX manuscript.** Your job in this pass is *only* to
substitute the production results and figures below into the correct manuscript locations. Do **not**
rewrite methods or prose — later layers handle that.

## Provenance of every number here
- Source: the **v2 production run**, seed `123456789`, `paper_results/aws_run_v2/` (with a few dual
  graphs from `paper_results/aws_run/`). Trials T = 1000 per size.
- Estimator: report **p_A** (the direction-averaged intersection/union estimator) as the central value.
- Uncertainty: the pipeline's total error = Birge-inflated statistical error ⊕ fit-window systematic
  (this is the "±" quoted). Basis is **leading-order** finite-size scaling `p_c + A·L^(−3/4)` with an
  adaptive χ² cutoff. (See "Basis caveat" at the end — a decision the author still owns.)

---

## 0. HEADLINE: the current hat numbers are STALE — replace everywhere

The manuscript quotes hat site `0.822725` and bond `0.798161`. These are superseded by the production
run and are **wrong by ~1.6%** — replace every occurrence (abstract, summary table, discussion,
Table of L-by-L values):

| Hat, direct (vertex) graph | OLD (in manuscript) | **NEW (production)** |
|---|---|---|
| site p_c | 0.822725 | **0.83837 ± 0.00002** |
| bond p_c | 0.798161 | **0.82710 ± 0.00002** |
| dual site p_c | 0.544247 | **0.52320 ± 0.00010** |
| dual bond p_c | (was blank/duality) | **0.37445 ± 0.00003** |

(Dual site ± widened from 0.00006 to 0.00010: the direction-bias systematic is now folded into the
averaged estimator's error — see §8. Central value unchanged.)

Sanity check to keep: the hat (0.83837) and the spectre (0.83900) direct thresholds are nearly equal,
as two structurally-similar aperiodic monotiles must be — this is why the old 0.8227 (which would make
them differ by 0.016) was clearly the stale value.

**Consequence:** the 40-row L-by-L table (`tab:percolation_results`, values converging to 0.8227) is
from the old run. Either regenerate it from the production hat run or delete it (Layer-1 decision D1
already proposes moving it to an appendix — but only with corrected numbers).

---

## 1. Master family table  →  `\label{tab:family}`

Columns: Direct site / Direct bond / Dual site / Dual bond / d_f (direct graph).
All p_A ± total. `[cite]` marks a dual obtained by graph-isomorphism to a known lattice (see §5) —
fill with the exact/known value, not a run.

| Tiling | Direct site | Direct bond | Dual site | Dual bond | d_f (direct) |
|---|---|---|---|---|---|
| Hat, Tile(1,√3) [≡ turtle, generic a≠b] | 0.83837 ± 0.00002 | 0.82710 ± 0.00002 | 0.52320 ± 0.00010 | 0.37445 ± 0.00003 | 1.895 [1.893, 1.897] |
| Spectre (chiral Tile(1,1)) | 0.83900 ± 0.00002 | 0.82788 ± 0.00002 | 0.51929 ± 0.00016 | 0.36995 ± 0.00007 | 1.891 [1.890, 1.893] |
| Periodic Tile(1,1) | 0.83986 ± 0.00002 | 0.82900 ± 0.00002 | 1/2 `[cite: triangular]` | 0.347296 `[cite: triangular]` | 1.885 [1.882, 1.887] |
| Chevron, Tile(0,1), periodic | 0.67237 ± 0.00004 | 0.61814 ± 0.00003 | 0.592746 `[cite: square]` | 1/2 `[cite: square]` | 1.888 [1.886, 1.890] |
| Chevron, Tile(0,1), aperiodic | 0.67218 ± 0.00002 | 0.61752 ± 0.00002 | 0.61374 ± 0.00003 | 0.51546 ± 0.00002 | 1.891 [1.889, 1.893] |
| Comet, Tile(1,0), periodic | 0.74994 ± 0.00004 | 0.72132 ± 0.00003 | 1/2 `[cite: triangular]` | 0.347296 `[cite: triangular]` | 1.883 [1.881, 1.886] |
| Comet, Tile(1,0), aperiodic | 0.74575 ± 0.00002 | 0.71555 ± 0.00001 | 0.52320 ± 0.00010 `[cite: = hat dual]` | 0.37445 ± 0.00003 `[cite: = hat dual]` | 1.897 [1.895, 1.899] |

Note worth a sentence in the discussion: **comet periodic direct site = 0.74994 ± 0.00004 ≈ 3/4**
(the Bharadwaj 3/4 conjecture; 1.5σ from exactly 0.75).

---

## 2. d_f table  →  `\label{tab:df}`  (exact 2D value 91/48 ≈ 1.8958)

| Tiling | d_f (direct) | d_f (dual) |
|---|---|---|
| Hat [≡ generic a≠b] | 1.895 [1.893, 1.897] | 1.854 [1.852, 1.856] |
| Spectre | 1.891 [1.890, 1.893] | 1.807 [1.804, 1.808] |
| Periodic Tile(1,1) | 1.885 [1.882, 1.887] | — |
| Chevron (periodic / aperiodic) | 1.888 / 1.891 | 1.885 [1.883, 1.886] (aperiodic) |
| Comet (periodic / aperiodic) | 1.883 / 1.897 | — |

All direct-graph d_f are consistent with 91/48. The **dual** d_f run low for the hat (1.854) and
especially the spectre (1.807): flag these — they are finite-size-suppressed on the sparser dual
graphs, not a class change. State this honestly rather than claiming clean 91/48 for the duals.

---

## 3. Summary table  →  `\label{tab:summary}`  (hat only)

| | Site p_c | Bond p_c |
|---|---|---|
| Direct (edge) | 0.83837 ± 0.00002 | 0.82710 ± 0.00002 |
| Dual (tile) | 0.52320 ± 0.00010 | 0.37445 ± 0.00003 |

(The old `\revpending` "dual bond" cell is now filled: 0.37445 ± 0.00003, measured directly.)

---

## 4. Validation numbers  →  `\section{Verification}`

| Lattice | Measured site | Known site | Measured bond | Known bond |
|---|---|---|---|---|
| Square (direct) | 0.59286 ± 0.00003 | 0.5927460 | 0.50010 ± 0.00004 | 1/2 (exact) |
| Triangular (direct) | 0.50006 ± 0.00004 | 1/2 (exact) | 0.34734 ± 0.00003 | 2sin(π/18) ≈ 0.347296 |
| **Honeycomb** (= triangular dual) | 0.69716 ± 0.00004 | 0.6970402 | 0.65283 ± 0.00002 | 1−2sin(π/18) ≈ 0.652704 |
| Penrose P3 (direct) | 0.58391 ± 0.00008 | 0.583910 (Ziff–Babalievski) | 0.47690 ± 0.00006 | ≈ 0.477 |

The honeycomb numbers fill the `\revpending` honeycomb-verification cell in the paper. Penrose lands
exactly on the literature value.

---

## 5. Duals settled by isomorphism — the identity IS the result, and it CLOSES them

For several tiles, the tile-adjacency (dual) graph is **provably identical (up to relabelling) to a
known periodic lattice**. That identity, verified by the isomorphism checker, is itself the result:
because the isomorphic lattice's threshold is already known exactly/precisely, the dual is *closed*
with no Monte-Carlo estimate of its own. This is exactly why these duals were run only once at low
precision (in v1) and never re-run at production scale — a precise run would add nothing.

**Present these as "dual ≅ [lattice], so p_c is the known value," not as a measured number.**

| Tile, dual graph | ≅ (proven isomorphism) | p_c site / bond (the known value that closes it) | v1 run (confirmation only) |
|---|---|---|---|
| Comet periodic dual | **triangular lattice** | 1/2 (exact) / 2sin(π/18) ≈ 0.347296 (exact) | 0.49931 / 0.34703 ✓ |
| Chevron periodic dual | **square lattice** | 0.592746 (high-prec.) / 1/2 (exact) | 0.59241 / 0.49984 ✓ |
| Tile(1,1) periodic dual | **triangular lattice** | 1/2 (exact) / 0.347296 (exact) | 0.49594 / 0.34442 ✓ (short, noisy) |
| Comet aperiodic dual | **hat dual** (same novel graph) | 0.52320 / 0.37445 (= the hat-dual row) | 0.52341 / 0.37466 ✓ |

So the `[cite]` cells in the §1 family table should carry the **known lattice value with a footnote
naming the isomorphism** (e.g. "= triangular lattice, Ref. [triangular]"), and the confirmatory v1
numbers above should NOT appear as results — mention them, if at all, only as a one-line check that the
identity holds numerically.

**Only three duals are genuinely novel** (no isomorphism to a known lattice) and are therefore the ones
reported as measured production values: **hat dual, spectre dual, chevron-aperiodic dual**.

---

## 6. Figures / plots — which goes where

Result figures are produced by the pipeline (`visualiser/figures.py`: `fss_figure`,
`convergence_figure`, `cardy_figure`) from the npz above; the author supplies the PNGs. Map:

| Manuscript slot | Content | Data source |
|---|---|---|
| `fig:mean_and_std` (Fig. 1) | Hat direct: p_c mean + 95% CI vs L, site & bond | hat direct npz |
| `fig:site` (Fig. 2) | Hat direct **site** finite-size scaling (I/U/A → p_c) | hat direct npz |
| `fig:bond` (Fig. 3) | Hat direct **bond** finite-size scaling | hat direct npz |
| `fig:mean_and_std_tile` (Fig. 8) | Hat **dual**: p_c mean + CI vs L | hat dual npz |
| `fig:site_tile` (Fig. 9) | Hat **dual** site finite-size scaling | hat dual npz |
| `fig:cardy` (pending) | Crossing prob vs aspect ratio, **hat + spectre**, Cardy curve overlaid + residual subpanel | `cardy_Hat_direct_patch6.npz`, `cardy_Spectre_direct_patch6.npz` |

Cardy data check (already computed, hat): anchor p* ≈ 0.8384 (matches the site threshold), RMS(measured
− Cardy) ≈ 0.004–0.007 across windows, ≈ 1.3–2.3× the √(1/4T) noise floor — report as "consistent with
the statistical floor within ~1.5×, RMS < 1%", **not** "at the floor".

Geometry figures (`hatunit`, `metatile`, `hattile`, `Percolation_Flowchart`) are unchanged — no results.

---

## 7. Basis caveat (author decision — flag, don't silently pick)

These are **leading-order** extrapolations. On the exact-known controls the leading-order fit sits a
touch high: square site +0.00011, triangular site +0.00006, honeycomb site +0.00012 (Penrose is spot
on). This is the corrections-to-scaling systematic. Two honest options:
- **Report leading-order** (as above) and let the control offsets stand as the measured
  "bias ruler" already folded into the error model; or
- **Switch to the corrections fit** (`p_c + A·L^(−3/4) + B·L^(−3/4−ω)`), which lands the controls on
  their exact values but is more cutoff-sensitive (Penrose is the hard case).

The **direct-graph monotile** thresholds are 5-digit-robust either way; the duals (~4-digit) and Penrose
are where the choice moves the 4th–5th digit. Decide once and apply consistently.

## 8. Direction bias — now FOLDED INTO THE ERROR (not just flagged)

The reported estimator is the direction-average p_A = ½(p_R + p_D), so a limit difference p_R ≠ p_D is a
real systematic on p_A: it sits |p_R − p_D|/2 from either directional threshold. Previously the isotropy
test only *printed* this; it is now **folded into the averaged estimator's error** (`dir_syst = |d_inf|/2`,
combined in quadrature) whenever the test statistically flags the object. Its sign is unmeasured (we get
the gap, not the asymmetry), so it widens the error and never moves the central value.

The paired check is clean (consistent with 0) for all **direct-graph monotiles** — their errors are
unchanged. The flagged objects, with their now-widened errors:

| Object | central p_c (unchanged) | error: was → now | folded dir_syst |
|---|---|---|---|
| Hat dual site | 0.52320 | 0.00006 → **0.00010** | 0.00007 |
| Spectre dual site | 0.51929 | 0.00011 → **0.00016** | 0.00013 |
| Comet-periodic direct bond | 0.72132 | 0.00003 → **0.00005** | 0.00004 |

(Comet-aperiodic dual = hat dual by isomorphism, so it inherits the 0.00010. The wildly-flagged periodic
duals — comet/chevron/Tile11 — are the isomorphism-closed cells reported as the known lattice value, so
their measured direction bias is irrelevant.) This makes the isotropy test *consequential* rather than
cosmetic, and the widened bars are what a referee should see for the two novel duals.
