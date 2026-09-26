"""Checkpoint/status plumbing for percolation runs.

runner/runner.py uses this to give each run a deterministic id (so an interrupted run resumes from
its OWN checkpoint instead of starting over) and to write its status/checkpoint files atomically.

    make_job_id / job_paths   -> deterministic id + on-disk paths (status / ckpt / stop)
    write_status / _safe_replace -> atomic status and checkpoint writes

This module formerly also launched runner.py as a detached process and polled it for the GUI's
progress bar. That path was removed -- the GUI now just emits the exact `runner.py` command to run in
a terminal -- so only the pieces the worker itself needs remain here (launch/liveness/list/stop were
GUI-only and are gone). RESULTS_DIR comes from engine.pipeline; the dependency is one-way
(gui_backend never imports this module).
"""
import os
import time
import json
import hashlib

from engine.pipeline import RESULTS_DIR


# A run checkpoints after every L and writes a small JSON status file next to its checkpoint, so a
# crash / machine power-off resumes from the last completed size. These paths and the atomic writers
# are shared with the worker (runner/runner.py).
JOBS_DIR = os.path.join(RESULTS_DIR, "jobs")


def _jobs_dir():
    os.makedirs(JOBS_DIR, exist_ok=True)
    return JOBS_DIR


def make_job_id(member, kind, patch, a, b, Lmin, Lmax, gap, T, seed):
    """Deterministic id from the run parameters: the SAME run maps to the SAME checkpoint, so
    relaunching an interrupted run resumes it instead of starting over."""
    key = f"{member}|{kind}|{patch}|{a}|{b}|{Lmin}|{Lmax}|{gap}|{T}|{seed}"
    h = hashlib.sha1(key.encode()).hexdigest()[:10]
    safe = "".join(c if (c.isalnum() or c in "-") else "_" for c in f"{member}_{kind}")
    return f"{safe}_{h}"


def job_paths(jid):
    jd = _jobs_dir()
    return {"status": os.path.join(jd, jid + ".json"), "ckpt": os.path.join(jd, jid + ".npz"),
            "stop": os.path.join(jd, jid + ".stop")}


def _safe_replace(src, dst, tries=25, delay=0.05):
    """os.replace with retry: on Windows a just-written temp file is often briefly locked by the
    AV/indexer, making the rename fail with PermissionError. Retry, then fall back to a direct
    (non-atomic) copy so a status/checkpoint update is never simply lost."""
    for _ in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            time.sleep(delay)
    try:
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            fdst.write(fsrc.read())
    finally:
        try:
            os.remove(src)
        except OSError:
            pass


def write_status(jid, d):
    """Atomic status write (tmp + replace) so a reader never sees a half-written file. The tmp name is
    pid-unique so concurrent writers can't clobber each other's temp file."""
    p = job_paths(jid)["status"]
    tmp = p + f".{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(d, f)
    _safe_replace(tmp, p)
