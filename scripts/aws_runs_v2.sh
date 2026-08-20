#!/usr/bin/env bash
# ============================================================================================
# REDO batch (v2) -- "everything at hat-level precision (~±0.0002)".  Design rationale:
#   * Error is REACH-limited, not trial-limited -> bigger patches, NOT more trials (T stays 1000).
#   * Cap the sweep at L_max~1200 (just past the hat's 1090): uniform hat precision, no L^3 waste.
#   * Sparse low-L is cut anyway -> start the sweep at L=200.
#   * KNOWN duals get 0 runs: comet-per=triangular, chevron-per=square, Tile(1,1)-dual=triangular,
#     comet-ap-dual=hat-dual -> proven EXACT by the isomorphism checker (run scripts/iso_check first).
#   * Only genuinely NOVEL, under-resolved objects are re-run here. Hat direct is already hat-level
#     (kept from batch 1); hat DUAL is re-run only to add its d_f (--exponents).
#   * Validation reaches the same L~1100 so the bias-ruler calibration transfers to the objects.
# BEFORE running: recalibrate solid windows at the new patches (estimate.calibrate_solid_windows)
# so the runner's window cap is measured, not the 0.95*side fallback.
# Cost: ~64 h PC wall / ~11 h on a 32-core box / ~$16-25.
# ============================================================================================
set -uo pipefail                        # no -e: one failure must not abort the batch
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
SEED=123456789; R="${PY:-python} runner/runner.py"

echo "===== novel aperiodic monotiles @ r=7, sweep capped L=200..1200, +d_f ====="
$R --tiling "Spectre"           --graph direct --patch 7 --lmin 200 --lmax 1200 --gap 10 --trials 1000 --seed $SEED --exponents --threads 0
$R --tiling "Spectre"           --graph dual   --patch 7 --lmin 200 --lmax 1200 --gap 10 --trials 1000 --seed $SEED --exponents --threads 0
$R --tiling "Comet aperiodic"   --graph direct --patch 7 --lmin 200 --lmax 1130 --gap 10 --trials 1000 --seed $SEED --exponents --threads 0
$R --tiling "Chevron aperiodic" --graph direct --patch 7 --lmin 200 --lmax 1200 --gap 10 --trials 1000 --seed $SEED --exponents --threads 0
$R --tiling "Chevron aperiodic" --graph dual   --patch 7 --lmin 200 --lmax 1200 --gap 10 --trials 1000 --seed $SEED --exponents --threads 0
# (Comet-ap dual = hat dual by identity -> not run.)

echo "===== hat dual re-run @ r=6 only to add d_f (already hat-level reach) ====="
$R --tiling "Hat"               --graph dual   --patch 6 --lmin 200 --lmax 1090 --gap 10 --trials 1000 --seed $SEED --exponents --threads 0

echo "===== periodic monotile DIRECT graphs @ bigger cells for hat-level reach (+d_f) ====="
# L_max = each patch's MEASURED solid window (interface/solid_windows.json): Comet|1300=1100 (clean, fits),
# Chevron|1300=920 (large fringe -> caps below 1100; more cells would only buy reach at n^2 tile cost),
# Tile(1,1)|860=1190 (1080 sits inside).
$R --tiling "Comet"             --graph direct --patch 1300 --lmin 200 --lmax 1100 --gap 10 --trials 1000 --seed $SEED --exponents --threads 0
$R --tiling "Chevron"           --graph direct --patch 1300 --lmin 200 --lmax 920  --gap 10 --trials 1000 --seed $SEED --exponents --threads 0
$R --tiling "Tile(1,1) periodic" --graph direct --patch 860 --lmin 200 --lmax 1080 --gap 10 --trials 1000 --seed $SEED --exponents --threads 0
# (All three periodic DUALS are known lattices -> isomorphism proof, not run.)

echo "===== validation reaching the SAME L~1100 (so the bias ruler transfers) ====="
$R --tiling "Square"     --graph direct --patch 1300 --lmin 200 --lmax 1080 --gap 10 --trials 2000 --seed $SEED --exponents --threads 0
$R --tiling "Square"     --graph dual   --patch 1300 --lmin 200 --lmax 1080 --gap 10 --trials 2000 --seed $SEED --exponents --threads 0
$R --tiling "Triangular" --graph direct --patch 1300 --lmin 200 --lmax 1080 --gap 10 --trials 2000 --seed $SEED --exponents --threads 0
$R --tiling "Triangular" --graph dual   --patch 1300 --lmin 200 --lmax 1080 --gap 10 --trials 2000 --seed $SEED --exponents --threads 0
$R --tiling "Penrose"    --graph direct --patch 12   --lmin 100 --lmax 600  --gap 10 --trials 2000 --seed $SEED --exponents --threads 0   # fixed extent: can't reach 1100

echo "===== v2 complete ====="
