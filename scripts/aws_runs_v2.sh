#!/usr/bin/env bash
# ============================================================================================
# v2 batch -- "EVERYTHING to hat precision".  Directive: match the hat's ~5-digit p_c on every
# object.  The hat's only advantage is REACH (L~1090 at T=1000), so the recipe is simply:
#   * run every object at the hat's reach (L up to ~1100), capped at each patch's MEASURED solid
#     window (interface/solid_windows.json) so no size clips the fringe;
#   * T=1000 and lmin=20 -- EXACTLY the hat's recipe, so the only thing that varies across the family
#     is reach. (The hat hit its ~5-digit precision at T=1000; running it lower-T than everything else
#     would make the reference object look under-sampled. Folded/noisier objects land a hair looser at
#     equal T -- an honest consequence of the tiling, not a shortcut -- still ~4-5 digits.)
#   * --exponents on all, to fill in d_f where v1 lacked it (d_f itself stays ~+-0.02 regardless
#     of reach -- that exponent is just hard; p_c is what reaches hat precision here).
#
#   KNOWN duals get ZERO runs -- proven identical to classic lattices by engine/check_identities:
#     comet-ap dual = hat dual;  comet-per dual = triangular;  chevron-per dual = square;
#     Tile(1,1) dual = triangular.  Cite the exact values, don't measure.
#
#   The CONTROLS (Square/Triangular/Penrose) are run big here for TWO reasons: they validate the
#   pipeline (recover known p_c), AND at L~1100 their leading-order fit bias dies, which drops the
#   bias-ruler FLOOR -- the floor is what currently caps every object at ~3 digits, so lowering it
#   is what lets the hat-precision numbers actually SHOW.
#
# Cost: ~18 USD / ~11 h on a 32-core box (T=1000, itemized in the cost model). One-time.
# BEFORE running: solid_windows.json already holds the measured caps for these patches (committed).
#   The two r=7 comet/chevron-ap windows aren't measured (fold OOMs the Path oracle) -> the runner
#   falls back to 0.95*side, which is safe there (their L_max sits well inside the solid region).
# ============================================================================================
set -uo pipefail                        # no -e: one failure must not abort the batch
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
SEED=123456789; R="${PY:-python} runner/runner.py"
T=1000

echo "===== aperiodic monotiles @ r=7 (L->~1200), hat precision + d_f ====="
$R --tiling "Spectre"           --graph direct --patch 7 --lmin 20 --lmax 1200 --gap 10 --trials $T --seed $SEED --exponents --threads 0
$R --tiling "Spectre"           --graph dual   --patch 7 --lmin 20 --lmax 1200 --gap 10 --trials $T --seed $SEED --exponents --threads 0
$R --tiling "Comet aperiodic"   --graph direct --patch 7 --lmin 20 --lmax 1130 --gap 10 --trials $T --seed $SEED --exponents --threads 0
$R --tiling "Chevron aperiodic" --graph direct --patch 7 --lmin 20 --lmax 1200 --gap 10 --trials $T --seed $SEED --exponents --threads 0
$R --tiling "Chevron aperiodic" --graph dual   --patch 7 --lmin 20 --lmax 1200 --gap 10 --trials $T --seed $SEED --exponents --threads 0
# (comet-ap dual = hat dual by isomorphism -> not run.)

echo "===== hat dual (reach already L~1090; re-run only to add its d_f) ====="
$R --tiling "Hat"               --graph dual   --patch 6 --lmin 20 --lmax 1090 --gap 10 --trials $T --seed $SEED --exponents --threads 0

echo "===== periodic partner DIRECT graphs @ big cells (L capped at measured solid window) ====="
# Comet|1300=1100, Chevron|1300=920 (chevron's fractal fringe caps it below hat reach -> its best,
# ~hat-level precision anyway), Tile(1,1)|860=1190. Duals of all three are 0 runs (proven lattices).
$R --tiling "Comet"             --graph direct --patch 1300 --lmin 20 --lmax 1100 --gap 10 --trials $T --seed $SEED --exponents --threads 0
$R --tiling "Chevron"           --graph direct --patch 1300 --lmin 20 --lmax 920  --gap 10 --trials $T --seed $SEED --exponents --threads 0
$R --tiling "Tile(1,1) periodic" --graph direct --patch 860 --lmin 20 --lmax 1080 --gap 10 --trials $T --seed $SEED --exponents --threads 0

echo "===== validation / controls @ L~1080 (recover known p_c AND drop the bias-ruler floor) ====="
$R --tiling "Square"     --graph direct --patch 1300 --lmin 20 --lmax 1080 --gap 10 --trials $T --seed $SEED --exponents --threads 0
$R --tiling "Square"     --graph dual   --patch 1300 --lmin 20 --lmax 1080 --gap 10 --trials $T --seed $SEED --exponents --threads 0
$R --tiling "Triangular" --graph direct --patch 1300 --lmin 20 --lmax 1080 --gap 10 --trials $T --seed $SEED --exponents --threads 0
$R --tiling "Triangular" --graph dual   --patch 1300 --lmin 20 --lmax 1080 --gap 10 --trials $T --seed $SEED --exponents --threads 0
# Penrose: FIXED geometry -> reach is extent-limited (scale = 2*lmax). patch = subdivision DENSITY;
# 13 fills scale~1600 (L~800) -- the most reach Penrose can give, near-hat precision (its own limit).
$R --tiling "Penrose"    --graph direct --patch 13   --lmin 20 --lmax 800  --gap 10 --trials $T --seed $SEED --exponents --threads 0

echo "===== 3D negative control: cube to L=320 (recover 3D exponents; prove pipeline is not 2D-only) ====="
# Not runner/runner.py -- the cube has its own entry point. L=320 (~33M nodes at the top, sparse ladder)
# is the knee of the L^3 cost curve: exponents within ~1% of the 3D textbook for ~5 USD. T matches the hat.
${PY:-python} -m engine.null_control_cube --lmax 320 --trials $T

echo "===== v2 complete -- 14 family runs + cube; 4 duals skipped by isomorphism proof ====="
