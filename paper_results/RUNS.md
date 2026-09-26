# Run manifest — Aperiodic Percolation

What every run contains and where it lives. Organized by **what the run measures**, not by which batch
it came from. All runs use **seed 123456789**. File name = `Object_graph_patch<P>_L<min>-<max>_T<trials>.npz`
(the seed and the old `gui_`/`cardy_` prefixes are dropped — they were the same for everything).

```
paper_results/
  pc_runs/       percolation-threshold (finite-size-scaling) data + p_c null controls
  cardy_runs/    Cardy crossing-probability data
  manuscript_figures/approved/   the vetted paper figures (+ MANIFEST.md)
  RUNS.md, RESULTS_BRIEF_layer1.md, methods_cutoff_paragraph.md
```

## `pc_runs/` — threshold data (T=1000; the source for every threshold figure)

| object | graph | patch | L range | purpose |
|---|---|---|---|---|
| Hat | direct, dual | 6 | 10–1090 | headline monotile + its novel dual |
| Spectre | direct, dual | 7 | 20–1200 | second monotile + dual |
| Comet (aperiodic) | direct | 7 | 20–1130 | monotile (Tile(1,0) fold) |
| Chevron (aperiodic) | direct, dual | 7 | 20–1200 | monotile + dual |
| Comet (periodic) | direct | 1300 | 20–1100 | periodic partner |
| Chevron (periodic) | direct | 1300 | 20–920 | periodic partner |
| Tile(1,1) periodic | direct | 860 | 20–1080 | Spectre's periodic partner |
| Square | direct, dual | 1300 | 20–1080 | exact validation control |
| Triangular | direct, dual (=honeycomb) | 1300 | 20–780 | exact validation |
| Penrose | direct | 13 | 20–800 | aperiodic validation |
| cube (3D) | — | L160 | — | **null control**: recovers 3D p_c≈0.3117, decisively non-2D |
| hypercube (4D) | — | L32 | — | **null control**: p_c≈0.1969 |

`v2_run.log` is the production run log (provenance). Deliberately **0 runs** (closed by the isomorphism
proofs in `engine/check_identities.py`): comet-aperiodic dual (= hat dual), comet/chevron periodic duals
(= triangular/square), Tile(1,1) dual (= triangular).

## `cardy_runs/` — Cardy crossing data

| object | graph | patch | T | purpose |
|---|---|---|---|---|
| Hat | direct | 6 | 24000 | Cardy conformal-invariance test |
| Spectre | direct | 6 | 24000 | Cardy test |
| Comet (aperiodic) | direct | 6 | 24000 | Cardy test |
| Chevron (aperiodic) | direct | 6 | 24000 | Cardy test |
| Square | direct | 350 | 4000 | **positive control** (2D lattice reads Cardy at the noise floor) |
| Square | direct | 350 | 24000 | **positive control, matched T** (`_T24000.npz`; error bars match the aperiodic runs; L=250 RMS 0.0035 ≈ floor 0.0032). Used for the residuals figure. |
| Cube3D | — | L48 | 4000 | negative control (superseded by the T=24000 run below for figures) |
| Cube4D | — | L20 | 4000 | negative control (superseded by T=24000 below) |
| Cube3D | — | L48 | 24000 | **negative control, matched T** (`_T24000.npz`; floor 0.0032 = same as all 2D runs) |
| Cube4D | — | L20 | 24000 | **negative control, matched T** (`_T24000.npz`) |

**Discrimination (the two approved figures `cardy_residuals.png` + `cardy_grouping.png`).** All figures now use the T=24000 runs, so every object shares the noise floor 0.0032 — the comparison is apples-to-apples. Metric = ABSOLUTE RMS deviation from Cardy (crossing-prob units) at the largest window; the per-run noise floor is shown as a tick, NOT divided out (dividing by the floor was rejected — it let trial count masquerade as physics). Values: Square 0.0035 (≈floor, the 2D positive anchor); aperiodic 0.0043–0.0069 (hat 0.0043, chevron-ap 0.0049, spectre 0.0058, comet-ap 0.0069); then a ~7× empty gap; Cube3D 0.0497, Cube4D 0.0981. So the 2D lattice + the aperiodic tilings cluster within ~2×, and the non-2D controls sit ~7–20× out — a genuine void between the groups, not a chosen cutoff. (Ratio-to-own-floor, for reference only: Square ~1.1×, aperiodic 1.3–2.1×, Cube3D 15×, Cube4D 30× at T=24000 — the ratio inflates with T, which is why the figures use the absolute deviation instead.)

## Gaps — planned in the run scripts but never executed (relevant to the paper)
- **v3 Cardy anchor-band campaign** (`scripts/aws_runs_v3.sh`, `scripts/local_cardy_batch.sh`): the
  **Square positive control** is now DONE (2026-09-25, row above; it carries `Rh_band`). Still missing:
  aperiodic **dual** Cardy, and the **anchor band** (`Rh_band`) on the four aperiodic-direct files (those
  predate the band and lack it).
- **Penrose dual** p_c sweep — never run (direct only).
- **Cube ν at L=320** — only L=160 exists.

## Archive
The full pre-cleanup snapshot (v1/v2 batches, byte-identical duplicates, ~48 old exploratory sweeps,
resume checkpoints, `figures_generated/`, `paper_results/figures/`, non-approved figures, and the run
scripts `scripts/`) was moved out of the repo on 2026-09-25 to, in its original batch layout:
`C:\Users\aarya\Desktop\Percolation Project\_archive_paper_results_2026-09-25\`
Nothing was permanently destroyed — everything trimmed here is recoverable from that snapshot.

> Note: new runs from `runner/runner.py` / `cardy_runner.py` still write to `paper_results/npz/` by default
> (`engine/pipeline.py` `RESULTS_DIR`); move/rename them into `pc_runs/` or `cardy_runs/` to keep this layout.
