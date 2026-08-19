#!/usr/bin/env bash
# One-shot environment setup for a fresh Ubuntu EC2 instance, then how to launch the run.
# Run this ONCE after you SSH in and have the repo on the box:
#     bash scripts/aws_setup.sh
set -euo pipefail

echo ">>> installing system packages (python, venv, tmux) ..."
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip tmux

echo ">>> creating a virtualenv and installing the 4 Python deps ..."
python3 -m venv ~/perc-venv
source ~/perc-venv/bin/activate
pip install --upgrade pip
pip install numpy scipy numba matplotlib

echo ">>> warming the numba cache (first compile is slow; do it once now) ..."
python - <<'PY'
import numba, numpy as np
@numba.njit(cache=True)
def _w(a):
    s=0.0
    for x in a: s+=x
    return s
print("numba OK, warm:", _w(np.arange(1000.0)))
PY

cat <<'MSG'

============================================================
 SETUP DONE.  To launch the production run:

   source ~/perc-venv/bin/activate      # activate the env
   tmux new -s run                       # a session that survives disconnects
   bash scripts/aws_runs.sh              # <-- the run (~5-7 h)
                                         # detach with Ctrl-b then d; reattach: tmux attach -t run

 When it finishes, from your LOCAL machine download the results:
   scp -i YOUR_KEY.pem -r ubuntu@INSTANCE_IP:~/Aperiodic\ Percolation/paper_results/npz ./

 THEN TERMINATE THE INSTANCE in the EC2 console, or it keeps billing.
============================================================
MSG
