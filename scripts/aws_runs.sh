#!/usr/bin/env bash
# ============================================================================================
# Production run manifest -- "Percolation on the Family of Aperiodic Monotiles" (Version_4)
#   * runner/runner.py       -> p_c sweeps (site + bond + nu + isotropy in ONE sweep; --exponents adds d_f)
#   * runner/cardy_runner.py -> crossing-probability (Cardy) test
# Results land in paper_results/npz/. Checkpoints are written after every L: re-run any single
# line to RESUME it from where it stopped (safe on spot-instance interruption).
#
# All params come from interface.gui_backend.paper_preset (measured solid-window caps, uniform
# step 10, seed 123456789). Precision: hat/spectre/chevron-ap 4 dp, comet-ap & periodics 3-4 dp.
#
# Cost (measured-anchored): ~60 h wall on a 12-thread desktop; ~4-6 h on a 64-vCPU box.
# BEFORE the full run, benchmark ONE representative sweep on the target instance to confirm the
# wall-time (the cost model was historically unreliable; measure, don't extrapolate).
# ============================================================================================
set -euo pipefail
cd "$(dirname "$0")/.."                 # repo root
export PYTHONIOENCODING=utf-8
SEED=123456789
PY="${PY:-python}"
R="$PY runner/runner.py"
C="$PY runner/cardy_runner.py"
# --threads 0 = auto (cpu_count-1). On a 64-vCPU box that's 63; if the hat run hits a RAM ceiling
# (peak ~ threads x O(nodes)), drop to e.g. --threads 32.

echo "===== [1/3] 14 p_c sweeps ====="
# --- aperiodic headline objects (d_f via --exponents on the DIRECT graph) ---
$R --tiling "Hat"               --graph direct --patch 6   --lmin 10 --lmax 1090 --gap 10 --trials 1000 --seed $SEED --exponents --threads 0
$R --tiling "Hat"               --graph dual   --patch 6   --lmin 10 --lmax 1090 --gap 10 --trials 1000 --seed $SEED --threads 0
$R --tiling "Spectre"           --graph direct --patch 6   --lmin 20 --lmax 580  --gap 10 --trials 1000 --seed $SEED --exponents --threads 0
$R --tiling "Spectre"           --graph dual   --patch 6   --lmin 20 --lmax 580  --gap 10 --trials 1000 --seed $SEED --threads 0
$R --tiling "Comet aperiodic"   --graph direct --patch 6   --lmin 20 --lmax 490  --gap 10 --trials 1000 --seed $SEED --exponents --threads 0
$R --tiling "Comet aperiodic"   --graph dual   --patch 6   --lmin 20 --lmax 490  --gap 10 --trials 1000 --seed $SEED --threads 0
$R --tiling "Chevron aperiodic" --graph direct --patch 6   --lmin 20 --lmax 690  --gap 10 --trials 1000 --seed $SEED --exponents --threads 0
$R --tiling "Chevron aperiodic" --graph dual   --patch 6   --lmin 20 --lmax 690  --gap 10 --trials 1000 --seed $SEED --threads 0
# --- periodic siblings: 840 cells for the 4th p_c digit; no d_f (trivially in-class) ---
$R --tiling "Comet"             --graph direct --patch 840 --lmin 20 --lmax 700  --gap 10 --trials 1000 --seed $SEED --threads 0
$R --tiling "Comet"             --graph dual   --patch 840 --lmin 20 --lmax 700  --gap 10 --trials 1000 --seed $SEED --threads 0
$R --tiling "Chevron"           --graph direct --patch 840 --lmin 20 --lmax 590  --gap 10 --trials 1000 --seed $SEED --threads 0
$R --tiling "Chevron"           --graph dual   --patch 840 --lmin 20 --lmax 590  --gap 10 --trials 1000 --seed $SEED --threads 0
$R --tiling "Tile(1,1) periodic" --graph direct --patch 215 --lmin 20 --lmax 270 --gap 10 --trials 1000 --seed $SEED --threads 0
$R --tiling "Tile(1,1) periodic" --graph dual   --patch 215 --lmin 20 --lmax 270 --gap 10 --trials 1000 --seed $SEED --threads 0

echo "===== [2/3] 4 Cardy crossing tests (strong: T=24000, 20 aspects, direct) ====="
$C --tiling "Hat"               --graph direct --patch 6 --windows 150 250 400 550 --n-aspects 20 --trials 24000 --seed $SEED --threads 0
$C --tiling "Spectre"           --graph direct --patch 6 --windows 150 250 400     --n-aspects 20 --trials 24000 --seed $SEED --threads 0
$C --tiling "Comet aperiodic"   --graph direct --patch 6 --windows 150 250 400     --n-aspects 20 --trials 24000 --seed $SEED --threads 0
$C --tiling "Chevron aperiodic" --graph direct --patch 6 --windows 150 250 400 550 --n-aspects 20 --trials 24000 --seed $SEED --threads 0

echo "===== [3/3] validation controls (T=10000 -> 4 dp; --exponents -> recover 2D d_f=91/48) ====="
$R --tiling "Square"     --graph direct --patch 500 --lmin 20 --lmax 400 --gap 20 --trials 10000 --seed $SEED --exponents --threads 0   # square lattice
$R --tiling "Square"     --graph dual   --patch 500 --lmin 20 --lmax 400 --gap 20 --trials 10000 --seed $SEED --exponents --threads 0   # square (self-dual)
$R --tiling "Triangular" --graph direct --patch 600 --lmin 20 --lmax 400 --gap 20 --trials 10000 --seed $SEED --exponents --threads 0   # triangular lattice
$R --tiling "Triangular" --graph dual   --patch 600 --lmin 20 --lmax 400 --gap 20 --trials 10000 --seed $SEED --exponents --threads 0   # HONEYCOMB (triangular dual)
$R --tiling "Penrose"    --graph direct --patch 11  --lmin 20 --lmax 400 --gap 20 --trials 10000 --seed $SEED --exponents --threads 0   # Penrose (sub=11 -> lmax 400; dual pending, see task)

echo "===== all runs complete -> paper_results/npz/ ====="
