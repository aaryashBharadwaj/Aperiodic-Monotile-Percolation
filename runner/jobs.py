"""Background-job plumbing for the percolation GUI. A GUI run launches percolate.py as a DETACHED OS
process that reuses the exact kernels in gui_backend (build_graph + run_one), checkpoints after every
L, and writes a small JSON status file. Split out of gui_backend so the job/status/launch machinery
lives on its own; both the GUI and the worker (runner/percolate.py) use it.

    launch_job(...)          -> spawn percolate.py detached, return its job id
    make_job_id / job_paths  -> deterministic id + on-disk paths (status/ckpt/log/stop)
    write_status / read_job_status / list_jobs / stop_job / clear_job

Shared paths (REPO_ROOT, RESULTS_DIR) are imported from interface.gui_backend; the dependency is
one-way (gui_backend never imports this module).
"""
import os
import sys
import time
import json
import glob
import hashlib
import subprocess

from interface.gui_backend import REPO_ROOT, RESULTS_DIR


# ============================================================ BACKGROUND JOBS
# A GUI run launches percolate.py as a DETACHED OS process that reuses the exact kernels here
# (build_graph + run_one), checkpoints after every L, and writes a small JSON status file. Because
# it's a real process (not the browser rerun-loop), it survives the tab closing, the machine
# sleeping (OS suspend/resume), and even the Streamlit server dying; a full restart/crash resumes
# from the checkpoint. The GUI just polls the status JSON to draw a progress bar. This makes the CLI
# runners the same "engine" the GUI button drives, rather than a separate path.
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
            "log": os.path.join(jd, jid + ".log"), "stop": os.path.join(jd, jid + ".stop")}


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
    """Atomic status write (tmp + replace) so the GUI never reads a half-written file. The tmp name
    is pid-unique because both the launcher process and the worker process write the same status."""
    p = job_paths(jid)["status"]
    tmp = p + f".{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(d, f)
    _safe_replace(tmp, p)


def read_job_status(jid):
    try:
        with open(job_paths(jid)["status"]) as f:
            return json.load(f)
    except Exception:
        return None


def list_jobs():
    """All known jobs (running + finished), newest activity first."""
    out = []
    for f in glob.glob(os.path.join(_jobs_dir(), "*.json")):
        try:
            with open(f) as fh:
                out.append(json.load(fh))
        except Exception:
            pass
    return sorted(out, key=lambda s: s.get("updated", 0), reverse=True)


def stop_job(jid):
    """Cooperative stop: drop a flag file the worker checks between sizes (so it still saves what it
    has). Latency is one L, same as the old in-tab Stop."""
    open(job_paths(jid)["stop"], "w").close()


def clear_job(jid):
    """Forget a finished job (remove its status/checkpoint/log/flag). The final result npz in
    results_output/ is NOT touched."""
    for k, p in job_paths(jid).items():
        for cand in (p, p + ".tmp"):
            try:
                os.remove(cand)
            except OSError:
                pass


def launch_job(tiling, member, graph_type, kind, patch, a, b, Lmin, Lmax, gap, T, seed, name=None):
    """Spawn percolate.py detached and return its job id. Reuses an existing checkpoint (same
    params) automatically. The process outlives this Streamlit server."""
    jd = _jobs_dir()
    jid = make_job_id(member, kind, patch, a, b, Lmin, Lmax, gap, T, seed)
    # Clear any stale stop flag from a previous run of this id.
    try:
        os.remove(job_paths(jid)["stop"])
    except OSError:
        pass
    worker = os.path.join(REPO_ROOT, "runner", "percolate.py")
    argv = [sys.executable, worker, "--job-id", jid, "--tiling", tiling, "--member", member,
            "--graph", graph_type, "--kind", kind, "--patch", str(patch), "--a", str(a),
            "--b", str(b), "--lmin", str(Lmin), "--lmax", str(Lmax), "--gap", str(gap),
            "--trials", str(int(T)), "--seed", str(int(seed)), "--jobs-dir", jd,
            "--out-dir", RESULTS_DIR,
            "--name", name or ""]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    with open(job_paths(jid)["log"], "w") as logf:   # child inherits its own handle; close ours
        if os.name == "nt":
            DETACHED_PROCESS = 0x00000008
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            subprocess.Popen(argv, stdout=logf, stderr=subprocess.STDOUT, env=env, close_fds=True,
                             cwd=REPO_ROOT,
                             creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)
        else:
            subprocess.Popen(argv, stdout=logf, stderr=subprocess.STDOUT, env=env, close_fds=True,
                             cwd=REPO_ROOT, start_new_session=True)
    # Immediate placeholder so the UI shows the job before the worker's first write.
    write_status(jid, {"job_id": jid, "status": "launching", "i": 0, "total": 0, "n_valid": 0,
                       "member": member, "kind": kind, "patch": patch, "seed": int(seed),
                       "trials": int(T), "lmin": Lmin, "lmax": Lmax, "gap": gap, "last_line": "",
                       "started": time.time(), "updated": time.time(), "result_file": None,
                       "error": None, "name": name or ""})
    return jid
