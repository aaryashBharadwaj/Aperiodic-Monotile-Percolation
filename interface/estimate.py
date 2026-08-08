"""Pre-run estimation for the percolation GUI: the machine-dependent per-trial cost calibration
(calibrate -> gui_calib.json), the machine-independent tiling geometry (calibrate_geometry ->
gui_geometry.json), and the no-build node-count + ETA + accuracy readouts the UI shows before a run.
Split out of gui_backend so that module stays focused on building/running graphs.

    plan_run(member, kind, patch, L_values, T)   -> estimated node count + usable-L range + ETA
    geometry(member, kind, patch)                -> (nodes, side) WITHOUT building the patch
    accuracy_estimate(...)                       -> a broad digits-of-p_c band
    calibrate / calibrate_geometry               -> (re)fit the two on-disk models

Shared kernels/constants (build_graph, run_one, warm_up, _frame_data, default_bt, TRI, REPO_ROOT,
LAUNCH_OVERHEAD_S) are imported from interface.gui_backend; the dependency is one-way.
"""
import os
import math
import json
import time

from interface.gui_backend import (REPO_ROOT, build_graph, run_one, warm_up, _frame_data,
                                    default_bt, TRI, LAUNCH_OVERHEAD_S)

CALIB_PATH = os.path.join(REPO_ROOT, "data", "gui_calib.json")     # machine-dependent: per-trial cost (gitignored)
GEOM_PATH = os.path.join(REPO_ROOT, "data", "gui_geometry.json")   # machine-independent: precomputed tiling geometry


def _default_calib():
    # Per-node trial cost GROWS with frame size (cache effects): model it as c_a + c_b*ln(N) seconds
    # per (frame node x 4 estimators). Defaults fitted from a reference machine; refined by calibrate().
    return {"c_a": -4.8e-7, "c_b": 6.2e-8}


def load_calib():
    try:
        with open(CALIB_PATH) as f:
            c = json.load(f)
        return c if "c_b" in c else _default_calib()   # ignore stale flat-cost calib files
    except Exception:
        return _default_calib()


def _c_per_node(calib, N):
    """Per-node trial cost at frame size N, s per (node x 4 estimators). Grows ~ln(N) (cache)."""
    return max(calib.get("c_a", 0.0) + calib.get("c_b", 0.0) * math.log(max(N, 10)), 3.0e-8)


def _time_frame(bundle, L, T, bt=1.0):
    """Time the FULL per-size cost of one frame exactly as the sweep pays it -- frame extraction
    (window cut + sub-graph build + boundary detection) PLUS the 4 estimators, i.e. one run_one.
    Timing only the kernels (as before) left out the frame cost and made the ETA under-predict.
    Returns (frame nodes N, seconds per node per trial) or None if the frame is unusable."""
    t0 = time.perf_counter()
    step = run_one(bundle, L, T, 1, bt)
    dt = time.perf_counter() - t0
    if not step.get("usable"):
        return None
    N = step["N"]
    return N, dt / (T * N)


def calibrate(save=True):
    """Fit the size-dependent per-node cost c(N) = c_a + c_b*ln(N) from a small and a large frame
    (hat r=5). Two points because per-node cost grows with frame size (cache); a single flat cost
    badly underestimates the big production sweeps. Machine-dependent -> gui_calib.json."""
    warm_up()
    t_b = time.perf_counter()
    bundle = build_graph("Hat", "Direct (vertex)", 5)
    build_ref_cpn = (time.perf_counter() - t_b) / max(bundle["nodes"], 1)   # this machine's build s/node
    side = bundle["side"]
    small = _time_frame(bundle, 0.30 * side, 200)     # ~tens of thousands of nodes
    large = _time_frame(bundle, 0.72 * side, 100)     # ~hundreds of thousands of nodes
    if small and large and large[0] > small[0]:
        (N1, c1), (N2, c2) = small, large
        c_b = (c2 - c1) / (math.log(N2) - math.log(N1))
        c_a = c1 - c_b * math.log(N1)
        calib = {"c_a": float(c_a), "c_b": float(c_b), "fit_points": [[N1, c1], [N2, c2]]}
    else:
        calib = _default_calib()
    calib["build_ref_cpn"] = float(build_ref_cpn)
    if save:
        os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
        with open(CALIB_PATH, "w") as f:
            json.dump(calib, f, indent=2)
    return calib


# ----------------------------------------------------------------------------- geometry (no-build ETA)
# The ETA splits into a machine-dependent per-node cost (c_a, c_b from calibrate) and a machine-
# INDEPENDENT part: the tiling's in-frame node DENSITY and the patch's inscribed-square SIDE. Density
# is scale-invariant for the inflation tilings (inflating the hat tiles a bigger area at the same
# density), so we predict a big patch's size WITHOUT building it -- build only the cheap patches once,
# extrapolate the rest. The UI shows size + ETA for r=6 instantly; the heavy graph is built only on Run.
def _measure_geom(member, kind, patch):
    """Build a patch and return (total_nodes, inscribed side, TRUE in-frame node density).
    The frame density -- node_count in an interior L×L window / L^2 -- is what the ETA needs; the
    whole-patch nodes/side^2 overcounts badly because the patch sprawls past its inscribed square.
    (A nodes+edges "work" proxy was tried to capture degree, but it over-corrected the dual and hurt
    the hat -- the main case -- so a plain node count, which nails the hat, is what we use.)"""
    gt = "Dual (tile)" if kind == "dual" else "Direct (vertex)"
    t_b = time.perf_counter()
    b = build_graph(member, gt, patch)
    build_cpn = (time.perf_counter() - t_b) / max(b["nodes"], 1)   # build seconds per patch node
    L = max(10.0, 0.5 * b["side"])                       # a comfortably interior window
    fd = _frame_data(b, L, default_bt(member))
    nc = fd.get("node_count", 0)
    return int(b["nodes"]), float(b["side"]), (nc / (L * L) if nc else 0.0), build_cpn


def calibrate_geometry(save=True):
    """Populate gui_geometry.json by building only CHEAP patches; inflation tilings extrapolate up.
    Machine-independent (pure geometry) -- run once and ship it. Stores whole-patch density (for the
    node-count readout) and the true in-frame node density (for the ETA)."""
    G = {}
    # Hat / spectre: inflation model. Build r=2..4 (cheap); densities constant, side grows by lambda.
    for member in ("Hat", "Spectre"):
        for kind in ("direct", "dual"):
            tbl = {}; frame_dens = 0.0; build_cpn = 0.0
            for r in (2, 3, 4):
                n, s, fd, bcpn = _measure_geom(member, kind, r)
                tbl[str(r)] = [int(n), float(s)]
                if r == 4:
                    frame_dens = fd; build_cpn = bcpn
            s3, (n4, s4) = tbl["3"][1], (tbl["4"][0], tbl["4"][1])
            G[f"{member}|{kind}"] = {"model": "inflate", "table": tbl, "r_built": 4,
                                     "lambda": s4 / s3, "density": n4 / (s4 * s4),
                                     "frame_density": frame_dens, "build_cpn": build_cpn}
    # Penrose: fixed extent, density grows with subdivisions -> table each (all cheap), incl frame dens.
    tbl = {}; build_cpn = 0.0
    for s in range(4, 10):
        n, side, fd, bcpn = _measure_geom("Penrose", "direct", s)
        tbl[str(s)] = [int(n), float(side), float(fd)]
        build_cpn = bcpn                                 # largest subdivision (loop ends at 9) is most representative
    G["Penrose|direct"] = {"model": "table", "table": tbl, "build_cpn": build_cpn}
    # Periodic / triangular: block grows linearly with the cell count; densities constant.
    for member, kind in [("Comet", "direct"), ("Comet", "dual"), ("Chevron", "direct"),
                         ("Chevron", "dual"), (TRI, "direct"), (TRI, "dual"),
                         ("Square", "direct"), ("Square", "dual")]:
        n1, s1, f1, _b1 = _measure_geom(member, kind, 40)
        n2, s2, f2, b2 = _measure_geom(member, kind, 70)
        G[f"{member}|{kind}"] = {"model": "linear", "k": 0.5 * (s1 / 40 + s2 / 70),
                                 "density": 0.5 * (n1 / (s1 * s1) + n2 / (s2 * s2)),
                                 "frame_density": 0.5 * (f1 + f2), "build_cpn": b2}
    if save:
        os.makedirs(os.path.dirname(GEOM_PATH), exist_ok=True)
        with open(GEOM_PATH, "w") as f:
            json.dump(G, f, indent=2)
    return G


def load_geom():
    try:
        with open(GEOM_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def geometry(member, kind, patch):
    """Estimate (nodes, inscribed-square side) for a patch WITHOUT building it, from gui_geometry.json.
    Returns None if the geometry file is missing (caller can fall back to an actual build)."""
    g = load_geom().get(f"{member}|{kind}")
    if not g:
        return None
    if g["model"] == "linear":
        side = g["k"] * patch
        return int(g["density"] * side * side), side
    t = g["table"]
    key = str(int(round(patch)))
    if key in t:                                   # measured exactly
        return int(t[key][0]), float(t[key][1])
    if g["model"] == "table":                      # penrose: nearest tabulated subdivision
        ks = sorted(int(x) for x in t)
        nk = min(ks, key=lambda z: abs(z - patch))
        return int(t[str(nk)][0]), float(t[str(nk)][1])
    # inflate: extrapolate from the largest built patch (density constant, side geometric in lambda)
    side = t[str(g["r_built"])][1] * (g["lambda"] ** (patch - g["r_built"]))
    return int(g["density"] * side * side), float(side)


def _frame_density(g, patch):
    """The true in-frame node density (nodes per area) for the ETA (see calibrate_geometry)."""
    if g["model"] in ("inflate", "linear"):
        return g.get("frame_density", 0.0)
    t = g["table"]                                  # penrose: per-subdivision
    key = str(int(round(patch)))
    if key in t:
        return t[key][2]
    ks = sorted(int(x) for x in t)
    return t[str(min(ks, key=lambda z: abs(z - patch)))][2]


def plan_run(member, kind, patch, L_values, T, calib=None):
    """Pre-run plan from geometry alone (NO graph build): estimated node count, usable-L range, ETA.
    ETA uses the TRUE frame density, not the whole-patch density. Returns None if geometry missing."""
    g = load_geom().get(f"{member}|{kind}")
    ge = geometry(member, kind, patch)
    if not g or ge is None:
        return None
    nodes_est, side = ge
    usable_max = 0.9 * side
    usable = [L for L in L_values if L <= usable_max]
    calib = calib or load_calib()
    density = _frame_density(g, patch)
    # Sum the cost per L using the SIZE-DEPENDENT per-node cost: N(L)=density*L^2, and larger frames
    # cost more per node (cache), so the big production sweeps aren't wildly underestimated.
    sweep_eta = T * sum(density * L * L * _c_per_node(calib, density * L * L) for L in usable)
    # Graph build is a one-time cost the elapsed clock DOES pay, so the ETA must include it or it
    # under-reports (build alone can exceed the whole sweep). Build is linear in patch nodes; the
    # per-node cost is builder-specific (dual >> direct), stored per member|kind in the geometry file
    # and rescaled to THIS machine via the hat|direct build speed measured in calibrate().
    build_cpn = g.get("build_cpn", 0.0)
    geom_ref = load_geom().get("Hat|direct", {}).get("build_cpn", 0.0)
    mach = (calib.get("build_ref_cpn", geom_ref) / geom_ref) if geom_ref else 1.0
    build_eta = build_cpn * nodes_est * mach
    # Each run now launches a FRESH worker process, which pays a flat startup cost the sweep/build
    # models don't see: importing the stack (numba/scipy/matplotlib ~3s) + numba JIT warm-up. Small
    # (~6s typical; the very first run after a cold numba cache can be longer), but without it a tiny
    # sweep reads as "a few seconds" when the wall time is really ~10s+. Add it as a flat term.
    eta = LAUNCH_OVERHEAD_S + build_eta + sweep_eta
    return {"nodes_est": nodes_est, "side": side, "usable_max": usable_max, "n_usable": len(usable),
            "eta": eta, "build_eta": build_eta, "sweep_eta": sweep_eta, "launch_eta": LAUNCH_OVERHEAD_S}


def accuracy_estimate(member, kind, patch, L_values, T):
    """A BROAD band for how many digits of p_c the config buys -- the accuracy analogue of the coarse
    ETA. The extrapolated-p_c scatter shrinks with the largest L (crossing width ~ L^-3/4) and with
    trials (~1/sqrt(T)); the constant is tuned so the paper hat run reads ~3 digits and a quick r=4
    look reads ~1-2. NOT a rigorous CI, and small patches are additionally finite-size BIASED (the
    central value is off), which is why tiny-L configs are flagged 'rough' regardless of scatter."""
    ge = geometry(member, kind, patch)
    if not ge:
        return "—"
    side = ge[1]
    usable = [L for L in L_values if L <= 0.9 * side]
    if len(usable) < 3:
        return "—"
    Lmax = max(usable)
    err = 2.5 * (Lmax ** -0.75) / math.sqrt(max(T, 1))
    if Lmax < 150:
        return "rough (~1 digit)"
    if err < 7e-4:
        return "~3 digits"
    if err < 3e-3:
        return "~2 digits"
    return "~1–2 digits"
