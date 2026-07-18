# 🔵 Runners & plumbing  (`run_r6_*.py`, `run_hat_*.py`, `results.py`, `visualisation.py`)

**Why this is lower stakes:** these orchestrate and save; they do not compute a threshold. A bug
here shows up as a **crash or an obviously-wrong run**, not a silently-wrong published number.
Read them to confirm they wire the (already-checked) pieces together correctly.

---

## `run_r6_direct.py` / `run_r6_dual.py` — the r=6 runners

**INTENT (pseudocode):**
```
build patch r ; build graph (direct: vertex graph ; dual: tile graph)
resume: if a matching checkpoint exists (same seed & #trials), reload the L values already done
for each L in Lmin..Lmax:
    if L already done (resume): skip
    cut the L×L frame at centre (515, -273)
    run the 4 estimators in parallel (site I/U, bond I/U) with per-(L,criterion) seeds
    (direct only) also record pR/pD from the site-I sweep for the isotropy test
    append raw per-trial results ; write a checkpoint (so any interruption is recoverable)
after the sweep: extrapolate p_c ; nu width-line (L>=50) ; (direct) isotropy ; save npz + plots
```

**CHECK:**
- **Seeding:** `master_seed = seed + i*4 + {0,1,2,3}` per L-index i and criterion → distinct,
  deterministic streams. Confirm this gives independent, reproducible seeds (it does) and that the
  same `seed` reproduces the run.
- **Resume:** confirm it only reloads when `seed` AND `trials` match the checkpoint, and that a
  reloaded L is skipped (so resume cannot double-count or mix configs). This is the safety net
  that makes killing/restarting harmless.
- **No new statistics live here** — the runners *call* the `01_statistics.md` functions; they do
  not reimplement any maths.

## `run_hat_tile.py` / `run_hat_dual.py` — existing runners, session edits
- `run_hat_tile.py`: records `pR/pD` for isotropy; per-L checkpoint. `run_hat_dual.py`: now also
  runs bond estimators + the measured dual bond. Same call-the-stats-functions pattern; confirm no
  maths inline.

## `results.py` — the load fix (**this one you should actually eyeball**)
- One change in `PercolationResults.load`: metadata that is an **array** (e.g. the `center`
  tuple) is kept as-is instead of `.item()` (which throws on non-scalars). Confirm it is wrapped
  in `try/except (ValueError, AttributeError)` and only affects reading `meta_*` fields — the raw
  per-trial data path is untouched. This bug had made every r6 `.npz` un-loadable.

## `visualisation.py` — `_savefig`
- Adds saving figures to `figures_generated/` so headless/overnight runs produce PNGs (plt.show
  is a no-op there). Cosmetic; confirm it only writes files and does not alter any computed data.

## Sign-off
- [ ] Runners only orchestrate + seed + checkpoint; they call the stats functions, not reimplement them.
- [ ] Resume matches on (seed, trials) and skips done L — cannot mix configs or double-count.
- [ ] `results.py` load fix is read-only metadata handling; raw data path untouched.
