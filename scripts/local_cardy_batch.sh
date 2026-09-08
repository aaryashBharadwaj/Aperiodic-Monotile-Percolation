#!/usr/bin/env bash
# Local Cardy batch -- the runs decided in the planning table (recommended core + all of Tier 3 +
# the cube negative control). Serial; ~15-18 h wall-clock on a 12-core box. Ordered by priority so a
# stop after any line still leaves the most important results done. cardy_runner.py now reports the
# anchor-bootstrap band natively (Rh_band in the npz), so no post-hoc anchor step is needed.
#
# NOT here, and why:  comet-ap dual (== hat dual, WL-hash dedup -> inherits hat dual); all periodic
#   DUALS (proven isomorphic to triangular/square). NOTE: hat direct & chevron-ap direct ARE re-run
#   below -- their values were already fine, but the anchor band CANNOT be retrofitted to the old npz
#   (it never saved the raw pR the bootstrap needs), so a re-run is the only way to put the band on
#   every object for a consistent figure. See the planning table.
#
# If a run reports windows "skipped (outside patch)", bump that line's --patch.
set -e
cd "$(dirname "$0")/.."
COMMON="--windows 150 250 400 550 --n-aspects 20 --aspect-min 0.5 --aspect-max 2.0 --trials 24000 --threads 0"

echo "===== TIER 1: aperiodic DUALS (novel graphs -- show the duals are 2D too) ====="
python runner/cardy_runner.py --tiling Hat                --graph dual   --patch 6   $COMMON
python runner/cardy_runner.py --tiling Spectre            --graph dual   --patch 6   $COMMON
python runner/cardy_runner.py --tiling "Chevron aperiodic" --graph dual  --patch 6   $COMMON

echo "===== CORE: fix the weak direct + the pipeline validation ====="
# comet-ap direct: the weakest existing run (patch6, stalled at L=400, lost extreme aspects) -> patch7
python runner/cardy_runner.py --tiling "Comet aperiodic"  --graph direct --patch 7   $COMMON
# square: validates the Cardy PIPELINE reproduces the exact curve on a known lattice (positive control)
python runner/cardy_runner.py --tiling Square             --graph direct --patch 800 $COMMON
# hat & chevron-ap direct: values already fine, but re-run to attach the anchor band (old npz lack raw pR),
# so the band appears on EVERY object in the figure; chevron-ap also needs it to show its offset is anchor noise
python runner/cardy_runner.py --tiling Hat                --graph direct --patch 6   $COMMON
python runner/cardy_runner.py --tiling "Chevron aperiodic" --graph direct --patch 6 $COMMON

echo "===== TIER 3: parity + periodic belt-and-suspenders ====="
# spectre direct: extend to L=550 for parity with hat (was 3 windows / L=400)
python runner/cardy_runner.py --tiling Spectre            --graph direct --patch 6   $COMMON
# periodic directs: 2D by the periodic-universality default; not required, run to round out the table
python runner/cardy_runner.py --tiling Comet              --graph direct --patch 840 $COMMON
python runner/cardy_runner.py --tiling Chevron            --graph direct --patch 840 $COMMON
python runner/cardy_runner.py --tiling "Tile(1,1) periodic" --graph direct --patch 800 $COMMON

echo "===== CONSIDER: cube negative control for the Cardy TEST (3D must FAIL 2D Cardy) ====="
# 3D box footprint sweep vs the exact 2D Cardy curve; L kept modest since N ~ L^3.
python -m engine.null_control_cube --dim 3 --cardy --lmax 48 --trials 8000 --n-aspects 20

echo "===== DONE -- results in paper_results/npz/cardy_*.npz ====="
