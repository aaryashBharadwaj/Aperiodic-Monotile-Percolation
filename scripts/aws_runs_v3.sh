#!/usr/bin/env bash
# ============================================================================================
# v3 batch = v2 p_c campaign  +  the full Cardy batch (with the anchor band)  +  cube controls.
# One manifest to launch the whole run on one box (target: c7i.8xlarge, 32 vCPU, ~a day serial).
#
# Decisions baked in (2026-09-08):
#   * --exponents on the p_c sweeps ARCHIVES the largest-cluster (s_max) data as an open observable;
#     d_f is NOT analysed (removed -- boundary-limited on open patches).
#   * Cardy runs now report the anchor-bootstrap band natively (Rh_band in each npz).
#   * KNOWN duals get ZERO runs (proven by engine/check_identities): comet-ap dual = hat dual;
#     comet-per/tile11 dual = triangular; chevron-per dual = square.
#   * The extrapolation-exponent fix is RE-ANALYSIS (no reruns) -- applied to the saved arrays after.
#
# Checkpointed per-L: re-run any single line to resume it (safe on spot interruption).
# ============================================================================================
set -uo pipefail                         # no -e: one failure must not abort the batch
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
SEED=123456789
PY="${PY:-python}"
R="$PY runner/runner.py"
C="$PY runner/cardy_runner.py"
T=1000
CC="--windows 150 250 400 550 --n-aspects 20 --aspect-min 0.5 --aspect-max 2.0 --trials 24000 --seed $SEED --threads 0"

echo "===== [1/4] p_c sweeps -- aperiodic @ r=7, hat @ r=6 (--exponents archives s_max) ====="
$R --tiling "Spectre"            --graph direct --patch 7    --lmin 20 --lmax 1200 --gap 10 --trials $T --seed $SEED --exponents --threads 0
$R --tiling "Spectre"            --graph dual   --patch 7    --lmin 20 --lmax 1200 --gap 10 --trials $T --seed $SEED --exponents --threads 0
$R --tiling "Comet aperiodic"    --graph direct --patch 7    --lmin 20 --lmax 1130 --gap 10 --trials $T --seed $SEED --exponents --threads 0
$R --tiling "Chevron aperiodic"  --graph direct --patch 7    --lmin 20 --lmax 1200 --gap 10 --trials $T --seed $SEED --exponents --threads 0
$R --tiling "Chevron aperiodic"  --graph dual   --patch 7    --lmin 20 --lmax 1200 --gap 10 --trials $T --seed $SEED --exponents --threads 0
$R --tiling "Hat"                --graph direct --patch 6    --lmin 10 --lmax 1090 --gap 10 --trials $T --seed $SEED --exponents --threads 0
$R --tiling "Hat"                --graph dual   --patch 6    --lmin 10 --lmax 1090 --gap 10 --trials $T --seed $SEED --exponents --threads 0
# (comet-ap dual = hat dual by isomorphism -> not run.)

echo "===== [2/4] p_c sweeps -- periodic directs (duals are proven lattices -> 0 runs) ====="
$R --tiling "Comet"              --graph direct --patch 1300 --lmin 20 --lmax 1100 --gap 10 --trials $T --seed $SEED --exponents --threads 0
$R --tiling "Chevron"            --graph direct --patch 1300 --lmin 20 --lmax 920  --gap 10 --trials $T --seed $SEED --exponents --threads 0
$R --tiling "Tile(1,1) periodic" --graph direct --patch 860  --lmin 20 --lmax 1080 --gap 10 --trials $T --seed $SEED --exponents --threads 0

echo "===== [3/4] p_c validation controls @ L~1080 (recover known p_c, drop the bias floor) ====="
$R --tiling "Square"             --graph direct --patch 1300 --lmin 20 --lmax 1080 --gap 10 --trials $T --seed $SEED --exponents --threads 0
$R --tiling "Square"             --graph dual   --patch 1300 --lmin 20 --lmax 1080 --gap 10 --trials $T --seed $SEED --exponents --threads 0
$R --tiling "Triangular"         --graph direct --patch 1300 --lmin 20 --lmax 1080 --gap 10 --trials $T --seed $SEED --exponents --threads 0
$R --tiling "Triangular"         --graph dual   --patch 1300 --lmin 20 --lmax 1080 --gap 10 --trials $T --seed $SEED --exponents --threads 0
$R --tiling "Penrose"            --graph direct --patch 13   --lmin 20 --lmax 800  --gap 10 --trials $T --seed $SEED --exponents --threads 0

echo "===== [4/4] Cardy batch (T=24000, 20 aspects, windows->550, anchor band automatic) ====="
# Tier 1 -- aperiodic DUALS (novel; comet-ap dual = hat dual, not run)
$C --tiling "Hat"                --graph dual   --patch 6    $CC
$C --tiling "Spectre"            --graph dual   --patch 7    $CC
$C --tiling "Chevron aperiodic"  --graph dual   --patch 7    $CC
# Core -- weak direct + directs re-run for the band + pipeline validation
$C --tiling "Comet aperiodic"    --graph direct --patch 7    $CC
$C --tiling "Hat"                --graph direct --patch 6    $CC
$C --tiling "Chevron aperiodic"  --graph direct --patch 7    $CC
$C --tiling "Square"             --graph direct --patch 1300 $CC
# Tier 3 -- spectre parity + periodic directs
$C --tiling "Spectre"            --graph direct --patch 7    $CC
$C --tiling "Comet"              --graph direct --patch 1300 $CC
$C --tiling "Chevron"            --graph direct --patch 1300 $CC
$C --tiling "Tile(1,1) periodic" --graph direct --patch 860  $CC

echo "===== controls: cube nu-ladder + cube Cardy negative control ====="
$PY -m engine.null_control_cube --lmax 320 --trials $T                                 # 3D nu
$PY -m engine.null_control_cube --dim 4 --lmax 32 --trials $T                          # 4D nu
$PY -m engine.null_control_cube --dim 3 --cardy --lmax 64 --trials 8000 --n-aspects 20 # cube fails 2D Cardy

echo "===== v3 complete -> paper_results/npz/ ====="
