"""Backend logic for the percolation GUI (gui_app.py) -- deliberately Streamlit-free so it can be
tested from the CLI. It does NOT re-implement any numerics: it wires the existing generators, the
consolidated graph builders, the _par percolation kernels, and extrapolate_pc_raw into the calls the
UI needs, for EVERY tiling in the project (the main aperiodic monotiles, the periodic family limits,
and the Penrose / triangular validations).

    build_graph(...)                      -> a graph bundle (coords, neighbors, centre, side, kind)
    run_one(bundle, L, T, seed, bt)       -> percolate one L (site & bond crossing points, + pR/pD)
    extrapolate_result(...)               -> p_c (I/U/A) + direction-bias check from raw trials
    plan_run(member, kind, patch, ...)    -> estimated node count + ETA from geometry, WITHOUT building
    visualise(tiling, size, ...)          -> a rendered tiling Figure (+ optional graph overlay)
    save_result / list_saved / load_saved -> the 'Analyse saved' round-trip (results_output/)

The percolation RUN (any tiling, incl. long/overnight, with checkpoint/resume) goes through the one
consolidated runner percolate.py — launched detached by the GUI's Run button OR from a console
(see REPRODUCE.md). This module holds the shared kernels + analysis both use.
"""
import os
import sys
import math
import json
import time
import glob
import hashlib
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# gui_backend lives in interface/; anchor data files, results_output/ and the worker script to the
# REPO ROOT (one level up), not to interface/. (Package imports resolve via the repo root that the
# entry points -- interface/gui_app.py and runner/percolate.py -- put on sys.path.)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from builders.direct_graph_builder import (build_neighbor_graph_fast, analyze_square_frame,
                                  largest_square_center, graph_from_polygons)
from builders.dual_graph_builder import collect_leaf_polygons, build_dual_from_polygons, analyze_tile_square_frame
from engine.percolation import (percolationStatsI_par, percolationStatsU_par,
                         percolationStatsBondI_par, percolationStatsBondU_par)
from engine.analysis import extrapolate_pc_raw, isotropy_test
from visualiser.run_tiling_render import tiling_polygons, render_tiling, _threshold_class
from generators.family_geometry import tile_ab, periodic_graph, periodic_polys
from generators.periodic_tiling_generator import square_tiles, triangular_tris   # validated periodic geometry

S3 = math.sqrt(3)
NU = 4.0 / 3.0
CALIB_PATH = os.path.join(REPO_ROOT, "data", "gui_calib.json")     # machine-dependent: per-trial cost (gitignored)
GEOM_PATH = os.path.join(REPO_ROOT, "data", "gui_geometry.json")   # machine-independent: precomputed tiling geometry
RESULTS_DIR = os.path.join(REPO_ROOT, "results_output")           # the user's own runs + checkpoints (cwd-independent)
PAPER_DIR = os.path.join(REPO_ROOT, "paper_results")              # curated canonical paper results (shipped with the GUI)

TRI = "Triangular → Honeycomb"          # the dual-of-triangular validation (exact honeycomb)
FAMILY = "Tile(a,b) family"

# Every tiling the portal offers, grouped for the UI.
TILINGS_MAIN = ["Hat", "Spectre", "Comet", "Chevron", FAMILY]
TILINGS_VALID = ["Square", "Penrose", TRI]
TILINGS = TILINGS_MAIN + TILINGS_VALID
GRAPHS = ["Direct (vertex)", "Dual (tile)"]

CATEGORY = {"Hat": "Aperiodic monotile", "Spectre": "Aperiodic monotile",
            "Comet": "Periodic (family limit)", "Chevron": "Periodic (family limit)",
            FAMILY: "One-parameter family", "Square": "Validation (exact)",
            "Penrose": "Validation (aperiodic)", TRI: "Validation (triangular + honeycomb)"}

# Which graph(s) each tiling admits. Penrose is a vertex graph; the honeycomb IS the dual of the
# triangular tiling, so that one is dual-only. The square lattice is self-dual (both give a square
# lattice), so it offers both.
GRAPHS_FOR = {"Hat": GRAPHS, "Spectre": GRAPHS, "Comet": GRAPHS, "Chevron": GRAPHS,
              "Square": GRAPHS, "Penrose": ["Direct (vertex)"], TRI: GRAPHS}

# Percolation patch control per tiling: (label, min, max, default, help). Hat/spectre go to the
# production r=6 -- that's the headline result, it must be reachable here.
PATCH_CTL = {
    "Hat":     ("Patch recursion  r", 2, 6, 4, "Hat metatile inflation depth. r=6 is the production patch (heavy: ~1M+ nodes)."),
    "Spectre": ("Patch level",        2, 6, 4, "Spectre substitution depth."),
    "Comet":   ("Block size  (cells)", 20, 160, 60, "Periodic block: cells per side (paper uses 150)."),
    "Chevron": ("Block size  (cells)", 20, 160, 60, "Periodic block: cells per side (paper uses 150)."),
    "Square":  ("Grid size  n",        20, 160, 60, "n x n square lattice (the textbook check: site 0.5927, bond 0.5)."),
    "Penrose": ("Subdivisions",        4, 9, 7, "Penrose inflation steps (more = finer graph). Deep, converged Penrose runs use the CLI (s=13)."),
    TRI:       ("Grid size  n",        20, 100, 60, "n x n triangular block: direct graph = the triangular lattice, dual = the honeycomb."),
}

# Visualise (render) size control -- deliberately SMALL so the figure is actually drawable/legible.
RENDER_CTL = {
    "Hat":     ("Inflation level", 1, 4, 3),
    "Spectre": ("Level",           1, 4, 3),
    "Comet":   ("Cells",           3, 14, 8),
    "Chevron": ("Cells",           3, 14, 8),
    "Square":  ("Grid n",          3, 20, 10),
    "Penrose": ("Subdivisions",    3, 6, 5),
    TRI:       ("Grid n",          4, 24, 12),
}

# Known/target values to show alongside a run, where we have them.
REFERENCE = {
    ("Square", "direct"): "square lattice (exact): site 0.5927, bond 0.5",
    ("Square", "dual"):   "square lattice (self-dual): site 0.5927, bond 0.5",
    (TRI, "direct"): "triangular lattice (exact): site 0.5, bond 0.3473",
    (TRI, "dual"): "honeycomb (exact): site 0.6970, bond 0.6527",
    ("Comet", "direct"): "our earlier run: site ~0.76",
}


# ----------------------------------------------------------------------------- family mapping
def family_member(a, b):
    """Map a Tile(a,b) choice to the runnable tiling that has its percolation threshold, plus the
    human-readable class. The whole point of the family: every generic aperiodic (a!=b, both>0)
    tiling is graph-isomorphic to the hat, so it PERCOLATES IDENTICALLY -- we run the hat for it."""
    cls, _ = _threshold_class(a, b)
    if a == 0 and b == 0:
        return None, cls
    if b == 0:
        return "Comet", cls
    if a == 0:
        return "Chevron", cls
    if abs(a - b) < 1e-9:
        return "Spectre", cls
    return "Hat", cls


def resolve_member(tiling, a=1.0, b=S3):
    """The concrete runnable tiling behind a UI choice (Tile(a,b) resolves via its geometry)."""
    if tiling == FAMILY:
        return family_member(a, b)[0]
    return tiling


# ----------------------------------------------------------------------------- geometry helpers
def _hat_patch(r):
    from generators.hat_generator import (H_init, T_init, P_init, F_init,
                               constructPatch, constructMetatiles)
    cur = [H_init(), T_init(), P_init(), F_init()]
    p = None
    for _ in range(r):
        p = constructPatch(*cur); cur = constructMetatiles(p)
    return p, r + 1   # build_neighbor_graph_fast level = r+1 for the hat


def _spectre_patch(level):
    from generators.spectre_generator import build_spectre_patch
    return build_spectre_patch(level), None   # level=None -> recurse to leaves


def _penrose_tiling(subdivisions, scale=200):
    from generators.penrose_tiling_generator import PenroseTiling
    t = PenroseTiling(divisions=subdivisions, base=5, scale=scale)
    t.make_tiling()
    return t


def _penrose_polys(tiling):
    """Robinson triangles as (3,2) xy polygons (their vertices are complex numbers)."""
    return [np.array([[t.v1.real, t.v1.imag], [t.v2.real, t.v2.imag], [t.v3.real, t.v3.imag]])
            for t in tiling.triangles]


# ----------------------------------------------------------------------------- graph building
def build_graph(tiling, graph_type, patch, a=1.0, b=S3):
    """Build the requested graph and package everything the sweep needs.
    Returns a picklable bundle: {coords, neighbors, center, side, name, kind, nodes}."""
    name = resolve_member(tiling, a, b)
    if name is None:
        raise ValueError("Tile(0,0) is degenerate -- nothing to build.")
    is_dual = graph_type.startswith("Dual")
    center_mode = "square"

    if name in ("Hat", "Spectre"):
        patch_obj, lvl = _hat_patch(patch) if name == "Hat" else _spectre_patch(patch)
        if is_dual:
            polys = collect_leaf_polygons(patch_obj, (patch + 1) if name == "Hat" else 10)
            coords, neighbors, _ = build_dual_from_polygons(polys)
        else:
            coords, neighbors = build_neighbor_graph_fast(patch_obj, level=lvl)
    elif name in ("Comet", "Chevron"):
        if is_dual:
            polys, _span = periodic_polys(name.lower(), ncells=patch)   # same block as the direct path
            coords, neighbors, _ = build_dual_from_polygons(polys)
        else:
            coords, neighbors, _span = periodic_graph(name.lower(), ncells=patch)
    elif name == "Penrose":
        from builders.penrose_graph_builder import build_penrose_neighbor_graph
        coords, neighbors, _edges = build_penrose_neighbor_graph(_penrose_tiling(patch))
        is_dual = False          # Penrose is a vertex graph
        center_mode = "bbox"     # roughly pentagonal patch -> centre the bbox
    elif name == TRI:
        tris = triangular_tris(int(patch))
        if is_dual:
            coords, neighbors, _ = build_dual_from_polygons(tris)   # tile adjacency -> the HONEYCOMB
        else:
            coords, neighbors = graph_from_polygons(tris)           # vertex graph -> the TRIANGULAR lattice
    elif name == "Square":
        polys = square_tiles(int(patch))
        if is_dual:
            coords, neighbors, _ = build_dual_from_polygons(polys)   # tile adjacency -> square lattice again
        else:
            coords, neighbors = graph_from_polygons(polys)           # perimeter -> square lattice
    else:
        raise ValueError(f"unknown tiling {name!r}")

    if center_mode == "bbox":
        x, y = coords[:, 0], coords[:, 1]
        cx, cy = (x.min() + x.max()) / 2.0, (y.min() + y.max()) / 2.0
        side = min(x.max() - x.min(), y.max() - y.min())
    else:
        cx, cy, side = largest_square_center(coords)

    return {"coords": coords, "neighbors": neighbors,
            "center": (float(cx), float(cy)), "side": float(side),
            "name": name, "kind": "dual" if is_dual else "direct", "nodes": len(coords)}


# ----------------------------------------------------------------------------- the sweep
_WARM = False
def warm_up():
    """Compile the numba kernels once (single-threaded) so the first real L isn't dominated by JIT."""
    global _WARM
    if _WARM:
        return
    wn = [np.array([1], np.int32), np.array([0, 2], np.int32), np.array([1], np.int32)]
    we = np.array([[0, 1], [1, 2]], np.int32)
    percolationStatsI_par(np.arange(3), wn, [0], [2], [0], [2], 2, master_seed=1)
    percolationStatsU_par(np.arange(3), wn, [0], [2], [0], [2], 2, master_seed=1)
    percolationStatsBondI_par(np.arange(3), we, [0], [2], [0], [2], 2, master_seed=1)
    percolationStatsBondU_par(np.arange(3), we, [0], [2], [0], [2], 2, master_seed=1)
    _WARM = True


def _frame_data(bundle, L, bt=1.0):
    cx, cy = bundle["center"]
    # Pick the frame analyzer by graph kind (kept out of the bundle so it stays picklable/cacheable).
    frame = analyze_tile_square_frame if bundle["kind"] == "dual" else analyze_square_frame
    return frame(bundle["coords"], bundle["neighbors"], L,
                 boundary_thickness=bt, center_x=cx, center_y=cy)


def _frame_usable(fd):
    return (fd.get("node_count", 0) >= 4 and min(
        len(fd["top_boundary_nodes"]), len(fd["bottom_boundary_nodes"]),
        len(fd["left_boundary_nodes"]), len(fd["right_boundary_nodes"])) > 0)


def run_one(bundle, L, T, seed_base, bt=1.0):
    """Percolate ONE L-window (the 4 estimators). Returns a dict; usable=False if the frame is empty
    or pokes outside the tiling. Factored out so the GUI can drive the sweep one size per rerun --
    that's what makes it interruptible (a Stop button is processed between steps)."""
    warm_up()
    fd = _frame_data(bundle, L, bt)
    if not _frame_usable(fd):
        return {"usable": False, "L": float(L)}
    A = (fd["sub_graph_nodes"], fd["sub_graph_neighbors"], fd["top_boundary_nodes"],
         fd["bottom_boundary_nodes"], fd["left_boundary_nodes"], fd["right_boundary_nodes"])
    E = (fd["sub_graph_nodes"], fd["sub_graph_edges"], fd["top_boundary_nodes"],
         fd["bottom_boundary_nodes"], fd["left_boundary_nodes"], fd["right_boundary_nodes"])
    # keep the intersection objects: pR/pD (site) and bond_pR/bond_pD validate the p_A estimators
    si = percolationStatsI_par(*A, T, master_seed=seed_base + 0)
    bi = percolationStatsBondI_par(*E, T, master_seed=seed_base + 2)
    return {"usable": True, "L": float(L), "N": fd["node_count"],
            "SI": si.trialResults, "pR": si.pR, "pD": si.pD,
            "SU": percolationStatsU_par(*A, T, master_seed=seed_base + 1).trialResults,
            "BI": bi.trialResults, "bond_pR": bi.pR, "bond_pD": bi.pD,
            "BU": percolationStatsBondU_par(*E, T, master_seed=seed_base + 3).trialResults}


def _pick_lmin(valid, floor=50):
    """Use L_min=floor (the paper's cutoff) for the direction-bias fit if it leaves >=3 sizes; else 0
    (use all), since exploratory GUI sweeps often don't reach L=50 with enough points."""
    return floor if sum(1 for L in valid if L >= floor) >= 3 else 0


def extrapolate_result(valid, rSI, rSU, rBI, rBU, skipped, raw_pR=None, raw_pD=None,
                       raw_bond_pR=None, raw_bond_pD=None):
    """Assemble the result dict + the FULL analysis: p_c extrapolation (I/U/A) and the direction-bias
    check (if pR/pD present) that validates the averaged estimator -- separately for the site
    (raw_pR/raw_pD -> 'isotropy') and bond (raw_bond_pR/raw_bond_pD -> 'isotropy_bond') thresholds.
    Needs >=3 sizes; fields stay None otherwise. Works on a PARTIAL sweep too, so a stopped run still
    yields whatever it collected."""
    out = {"L": valid, "raw_SI": rSI, "raw_SU": rSU, "raw_BI": rBI, "raw_BU": rBU,
           "skipped": skipped, "raw_pR": raw_pR, "raw_pD": raw_pD,
           "raw_bond_pR": raw_bond_pR, "raw_bond_pD": raw_bond_pD,
           "site": None, "bond": None, "isotropy": None, "isotropy_bond": None, "lmin": None}
    if len(valid) >= 3:
        out["site"] = extrapolate_pc_raw(valid, rSI, rSU)
        if rBI and rBU and rBI[0] is not None:
            out["bond"] = extrapolate_pc_raw(valid, rBI, rBU)
        lmin = _pick_lmin(valid); out["lmin"] = lmin
        for key, pR, pD in [("isotropy", raw_pR, raw_pD),
                            ("isotropy_bond", raw_bond_pR, raw_bond_pD)]:
            if pR is not None and pD is not None:
                try:
                    d_inf, d_ci, ok = isotropy_test(valid, pR, pD, L_min=lmin)
                    out[key] = {"d_inf": d_inf, "d_ci": d_ci, "isotropic": bool(ok)}
                except Exception:
                    pass
    return out


def save_result(result, member, kind, seed, T, name=None, out_dir=RESULTS_DIR):
    """Write a completed/partial result to results_output/ in the PercolationResults .npz format (so
    it re-loads in this portal's 'Analyse saved' tab). pR/pD go in a companion <stem>_iso.npz
    (PercolationResults doesn't carry them) so the direction-bias check survives a reload. `name` is
    the chosen file name (sanitised); if omitted a parameter+timestamp default is used. Returns the
    main path."""
    import re, datetime
    from engine.results import PercolationResults
    os.makedirs(out_dir, exist_ok=True)
    if name and name.strip():
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
        if stem.endswith(".npz"):
            stem = stem[:-4]
    else:
        ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        stem = f"gui_{member.lower()}_{kind}_{ts}"
    path = os.path.join(out_dir, stem + ".npz")
    PercolationResults(tiling_type=f"{member}_{kind}", seed=int(seed), trials=int(T),
                       L_values=result["L"], raw_SI=result["raw_SI"], raw_SU=result["raw_SU"],
                       raw_BI=result["raw_BI"], raw_BU=result["raw_BU"],
                       extra_meta={"source": "gui"}).save(path)
    if result.get("raw_pR") is not None and result.get("raw_pD") is not None:
        iso = {"L_values": np.asarray(result["L"], float)}
        for i, (r, d) in enumerate(zip(result["raw_pR"], result["raw_pD"])):
            iso[f"pR_{i}"] = np.asarray(r, float); iso[f"pD_{i}"] = np.asarray(d, float)
        # bond direction-bias arrays (optional -- older runs are site-only)
        if result.get("raw_bond_pR") is not None and result.get("raw_bond_pD") is not None:
            for i, (r, d) in enumerate(zip(result["raw_bond_pR"], result["raw_bond_pD"])):
                iso[f"bpR_{i}"] = np.asarray(r, float); iso[f"bpD_{i}"] = np.asarray(d, float)
        np.savez(os.path.join(out_dir, stem + "_iso.npz"), **iso)
    return path


def list_saved(out_dir=None):
    """Result files for 'Analyse saved': the shipped paper_results/ (canonical) listed FIRST, then
    the user's own results_output/ runs. Excludes _iso companions and checkpoints; newest first
    within each source; de-duplicated by name (paper_results wins). Pass out_dir to scan just one."""
    dirs = [out_dir] if out_dir else [PAPER_DIR, RESULTS_DIR]
    seen = set(); out = []
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        files = [f for f in os.listdir(d)
                 if f.endswith(".npz") and not f.endswith("_iso.npz") and "CHECKPOINT" not in f]
        for f in sorted(files, key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True):
            if f not in seen:
                seen.add(f); out.append(f)
    return out


def load_saved(fname, out_dir=None):
    """Load a saved .npz (PercolationResults + optional _iso companion) into the same result dict the
    live runs produce, so every plot/metric works identically. Returns (result, meta). Searches
    paper_results/ then results_output/ (or just out_dir if given)."""
    from engine.results import PercolationResults
    dirs = [out_dir] if out_dir else [PAPER_DIR, RESULTS_DIR]
    base = next((d for d in dirs if d and os.path.exists(os.path.join(d, fname))), dirs[-1])
    r = PercolationResults.load(os.path.join(base, fname))
    raw_pR = raw_pD = raw_bond_pR = raw_bond_pD = None
    isop = os.path.join(base, fname[:-4] + "_iso.npz")
    if os.path.exists(isop):
        d = np.load(isop)
        nL = len(d["L_values"])
        raw_pR = [d[f"pR_{i}"] for i in range(nL)]
        raw_pD = [d[f"pD_{i}"] for i in range(nL)]
        if "bpR_0" in d:                       # bond arrays present only in newer runs
            raw_bond_pR = [d[f"bpR_{i}"] for i in range(nL)]
            raw_bond_pD = [d[f"bpD_{i}"] for i in range(nL)]
    res = extrapolate_result(list(r.L_values), r.raw_SI, r.raw_SU, r.raw_BI, r.raw_BU, [],
                             raw_pR, raw_pD, raw_bond_pR, raw_bond_pD)
    return res, {"member": r.tiling_type, "kind": "", "seed": r.seed, "T": r.trials,
                 "timestamp": r.timestamp}


# ----------------------------------------------------------------------------- time estimate
# Flat per-run cost of spawning a fresh worker process: import the stack (numba/scipy/matplotlib)
# + numba JIT warm-up. Measured ~3s import + fast (cached) JIT; padded for machine/cold-cache slop.
LAUNCH_OVERHEAD_S = 6.0


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
def default_bt(member):
    """Boundary-band thickness for the frame. Penrose's vertex graph is sparser (larger spacing),
    so it needs a thicker band to catch boundary nodes; everything else uses 1.0."""
    return 2.0 if member == "Penrose" else 1.0


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


def paper_preset(member, kind):
    """The production parameters used in the paper for this object, or None if it isn't a headline
    object (Penrose/triangular are validations, not headline objects). These match the paper runs
    documented in REPRODUCE.md, so the GUI loads exactly 'the real run' in one click:
      Hat     -- r=6, L=10..1000 step 10, T=1000
      Spectre -- level 6, T=500, L=20..int(0.92*side), step max(10, round(Lmax/40/10)*10)
      Comet/Chevron -- 150 cells, T=500, L=20..~0.95*side, step 20
    Spectre/periodic L ranges are computed from the patch extent (via geometry); all use seed 123456789."""
    seed = 123456789
    if member == "Hat":
        return {"patch": 6, "L_min": 10.0, "L_max": 1000.0, "gap": 10.0, "T": 1000, "seed": seed}
    if member == "Spectre":
        ge = geometry("Spectre", kind, 6)
        side = ge[1] if ge else 400.0
        Lmax = int((side * 0.92) // 10 * 10)
        gap = max(10.0, round(Lmax / 40 / 10) * 10)
        return {"patch": 6, "L_min": 20.0, "L_max": float(Lmax), "gap": float(gap), "T": 500, "seed": seed}
    if member in ("Comet", "Chevron"):
        ge = geometry(member, kind, 150)
        side = ge[1] if ge else 300.0
        Lmax = int(side * 0.95 // 10 * 10)
        return {"patch": 150, "L_min": 20.0, "L_max": float(Lmax), "gap": 20.0, "T": 500, "seed": seed}
    return None


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


# ----------------------------------------------------------------------------- visualise engine
def _render_polys(tiling, size, a=1.0, b=S3):
    """(polys, title, color_by, penrose_obj) for the visualiser. penrose_obj is returned so a graph
    overlay can use the correct (rhombus-rule) Penrose builder."""
    name = resolve_member(tiling, a, b)
    if name in ("Hat", "Spectre", "Comet", "Chevron"):
        polys = tiling_polygons(name.lower(), level=size, ncells=size)
        color_by = "chirality" if name == "Hat" else "orientation"
        return polys, f"{name}  ({len(polys)} tiles)", color_by, None
    if name == "Square":
        polys = square_tiles(size)
        return polys, f"Square  ({len(polys)} tiles)", None, None
    if name == "Penrose":
        t = _penrose_tiling(size, scale=200)
        polys = _penrose_polys(t)
        return polys, f"Penrose  ({len(polys)} triangles)", "orientation", t
    if name == TRI:
        polys = triangular_tris(size)
        return polys, f"Triangular  ({len(polys)} triangles)", "orientation", None
    raise ValueError(f"cannot render {name!r}")


def _overlay_graph(fig, tiling, polys, graph_type, penrose_obj):
    """Draw the actual percolation graph (nodes + edges) on top of the tiles, using the correct
    builder for the selected graph type -- so it shows exactly what gets percolated."""
    name = resolve_member(tiling)
    if graph_type.startswith("Dual"):
        nodes, neigh, _ = build_dual_from_polygons(polys)
    elif name == "Penrose" and penrose_obj is not None:
        from builders.penrose_graph_builder import build_penrose_neighbor_graph
        nodes, neigh, _ = build_penrose_neighbor_graph(penrose_obj)
    else:
        nodes, neigh = graph_from_polygons(polys)
    segs = [[(nodes[i, 0], nodes[i, 1]), (nodes[j, 0], nodes[j, 1])]
            for i, nb in enumerate(neigh) for j in nb if i < j]
    # Thin the overlay as the graph grows so big renders stay legible (a dense mesh of thick lines
    # is a smear); small graphs keep chunky, readable nodes/edges.
    m = len(nodes)
    lw = max(0.12, min(0.6, 55.0 / (m ** 0.5)))
    ms = max(0.8, min(3.2, 150.0 / (m ** 0.5)))
    ax = fig.axes[0]
    ax.add_collection(LineCollection(segs, colors="#111111", linewidths=lw, alpha=0.55, zorder=3))
    ax.plot(nodes[:, 0], nodes[:, 1], ".", color="crimson", ms=ms, zorder=4)
    return len(nodes), len(segs)


def visualise(tiling, size, a=1.0, b=S3, graph_type="Direct (vertex)", show_graph=False):
    """Render the chosen tiling at the chosen size. Returns (fig, n_tiles, graph_counts or None).
    This is the generator showcase -- proof the substitution/lattice code produces the real shape.
    No in-figure title (name/count/a-b/class are already in the UI), and a fixed modest figure size
    so it doesn't stretch to the full column width."""
    if tiling == FAMILY:
        member, cls = family_member(a, b)
        if member is None:
            fig, ax = plt.subplots(figsize=(5, 5)); ax.axis("off")
            ax.text(0.5, 0.5, "Tile(0,0)\ndegenerate", ha="center", va="center", fontsize=14)
            return fig, 0, None
        col = _threshold_class(a, b)[1]
        fig, ax = plt.subplots(figsize=(5, 5))
        render_tiling([np.asarray(tile_ab(a, b))], ax=ax, facecolor=col, lw=1.6)
        return fig, 1, None

    # Every tiling (incl. Triangular): draw the TILE image, then let the optional overlay show what
    # counts as nodes/edges -- direct = vertices + tile edges, dual = centroids + adjacency. For the
    # triangular tiling this is the nice one: connecting the triangle centroids (the dual) naturally
    # traces out the honeycomb, on top of the same triangles.
    polys, _title, color_by, pobj = _render_polys(tiling, size, a, b)
    fig, ax = plt.subplots(figsize=(5, 5))
    render_tiling(polys, ax=ax, color_by=color_by, lw=0.4)
    gcounts = None
    if show_graph:
        gcounts = _overlay_graph(fig, tiling, polys, graph_type, pobj)
    return fig, len(polys), gcounts


# ----------------------------------------------------------------------------- percolation demo
# A deterministic percolation walk-through for the "How it works" tab, on a small SQUARE lattice OR a
# small HAT block (the reader picks). Open SITES (nodes) or BONDS (edges) one at a time in a fixed
# order and watch clusters grow -- EVERY cluster its own colour -- until one spans top<->bottom or
# left<->right. Site vs bond is the same choice the real study offers; the two are drawn differently
# (site colours the dots, bond colours the connections) so they don't look alike.
def _square_demo(n):
    idx = lambda i, j: j * n + i
    coords = [[float(i), float(j)] for j in range(n) for i in range(n)]
    edges = []
    for j in range(n):
        for i in range(n):
            if i + 1 < n: edges.append([idx(i, j), idx(i + 1, j)])
            if j + 1 < n: edges.append([idx(i, j), idx(i, j + 1)])
    return (coords, edges, n * n,
            set(idx(i, n - 1) for i in range(n)), set(idx(i, 0) for i in range(n)),
            set(idx(0, j) for j in range(n)), set(idx(n - 1, j) for j in range(n)),
            [-0.6, n - 0.4, -0.6, n - 0.4])


def _hat_demo(level=3, window_frac=0.45):
    """Square window of the hat VERTEX graph (corners = nodes, tile edges = bonds) -- consistent with
    the square demo, so site/bond mean the same thing, but on the actual hat tiling."""
    patch_obj, _lvl = _hat_patch(level)
    nodes, neigh = build_neighbor_graph_fast(patch_obj, level=level + 1)
    cx, cy, side = largest_square_center(nodes)
    half = 0.5 * window_frac * side
    inw = lambda p: abs(p[0] - cx) <= half and abs(p[1] - cy) <= half
    inside = [i for i in range(len(nodes)) if inw(nodes[i])]
    idx = {o: m for m, o in enumerate(inside)}
    coords = np.array([nodes[o] for o in inside])
    edges = sorted({(min(idx[o], idx[j]), max(idx[o], idx[j]))
                    for o in inside for j in neigh[o] if j in idx})
    xs, ys = coords[:, 0], coords[:, 1]; band = 0.14 * (2 * half)
    B = lambda mask: set(int(m) for m in np.where(mask)[0])
    return (coords.tolist(), [list(e) for e in edges], len(inside),
            B(ys >= ys.max() - band), B(ys <= ys.min() + band), B(xs <= xs.min() + band), B(xs >= xs.max() - band),
            [float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())])


def build_demo(lattice="square", n=12, seed=0):
    """Small SQUARE or HAT lattice for the walk-through: coords, edges, adjacency, which nodes touch
    each side, fixed random site/bond open orders, a bbox, and the precomputed first-crossing step per
    direction (so the UI can say 'top-bottom at ...' / 'left-right at ...' without a per-frame scan)."""
    if lattice == "hat":
        coords, edges, N, top, bottom, left, right, bbox = _hat_demo()
    else:
        coords, edges, N, top, bottom, left, right, bbox = _square_demo(n)
    M = len(edges)
    neigh = [[] for _ in range(N)]
    for a, b in edges:
        neigh[a].append(b); neigh[b].append(a)
    rng = np.random.default_rng(seed)
    demo = {"lattice": lattice, "coords": coords, "edges": edges, "neigh": neigh, "N": N, "M": M,
            "top": top, "bottom": bottom, "left": left, "right": right, "bbox": bbox,
            "site_order": [int(x) for x in rng.permutation(N)],
            "bond_order": [int(x) for x in rng.permutation(M)]}
    demo["cross_site"] = _crossings(demo, "site")
    demo["cross_bond"] = _crossings(demo, "bond")
    return demo


def _crossings(demo, mode):
    """First step at which each direction first spans (or None). Incremental union-find with a per-
    cluster boundary BITMASK (T|B|L|R), opening one site/bond per step -- O(steps * alpha), so it's
    fast even for a 40x40 block. The JS component re-derives clusters itself for the live drawing;
    this only supplies the crossing steps (jump target + the bracket caption)."""
    N, edges, neigh = demo["N"], demo["edges"], demo["neigh"]
    order = demo["site_order"] if mode == "site" else demo["bond_order"]
    T, B, L, R = 1, 2, 4, 8
    fm = [0] * N
    for v in demo["top"]:    fm[v] |= T
    for v in demo["bottom"]: fm[v] |= B
    for v in demo["left"]:   fm[v] |= L
    for v in demo["right"]:  fm[v] |= R
    parent = list(range(N))
    def find(x):
        r = x
        while parent[r] != r: r = parent[r]
        while parent[x] != r: parent[x], x = r, parent[x]
        return r
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb; fm[rb] |= fm[ra]
    cross = {"top-bottom": None, "left-right": None}
    def check(r, step):
        m = fm[r]
        if cross["top-bottom"] is None and (m & (T | B)) == (T | B): cross["top-bottom"] = step
        if cross["left-right"] is None and (m & (L | R)) == (L | R): cross["left-right"] = step
    if mode == "site":
        isopen = bytearray(N)
        for step, v in enumerate(order, 1):
            isopen[v] = 1
            for nb in neigh[v]:
                if isopen[nb]:
                    union(v, nb)
            check(find(v), step)
            if cross["top-bottom"] and cross["left-right"]: break
    else:
        for step, ei in enumerate(order, 1):
            a, b = edges[ei]
            union(a, b)
            check(find(a), step)
            if cross["top-bottom"] and cross["left-right"]: break
    return cross


# The demo is DRAWN client-side in gui_app.demo_component (HTML/SVG+JS) so its slider updates smoothly
# on drag; Python only supplies the graph + fixed open orders (build_demo) and the crossings.


# ----------------------------------------------------------------------------- results figure
def _means_IUA(raw_I, raw_U):
    """Per-L Intersection / Union / Average estimator means and standard errors, matching
    extrapolate_pc_raw. Returns (mI,sI, mU,sU, mA,sA) as arrays."""
    mI, sI, mU, sU, mA, sA = [], [], [], [], [], []
    for rI, rU in zip(raw_I, raw_U):
        rI, rU = np.asarray(rI), np.asarray(rU); T = len(rI)
        a, b = rI.mean(), rU.mean()
        mI.append(a); mU.append(b); mA.append(0.5 * (a + b))
        si, su = rI.std(ddof=1), rU.std(ddof=1); cov = np.cov(rI, rU)[0, 1]
        sI.append(si / np.sqrt(T)); sU.append(su / np.sqrt(T))
        sA.append(0.5 * np.sqrt(si**2 / T + su**2 / T + 2 * cov / T))
    return tuple(np.array(v) for v in (mI, sI, mU, sU, mA, sA))


def fss_figure(result):
    """Finite-size-scaling plot showing ALL THREE estimators -- Intersection, Union, Average -- vs
    L^(-1/nu), each with its data, its WLS fit line and its L->inf intercept. This is the CLI
    plot_extrapolation_IU view: watching I and U squeeze toward the SAME intercept as L->0^+ is the
    sanity check -- if the two don't converge, the crossing/extrapolation is untrustworthy. Site and
    bond get their own panel. Returns a Figure."""
    L = np.asarray(result["L"], dtype=float)
    x = L ** (-1.0 / NU)
    xs = np.linspace(0, x.max() * 1.02, 100)
    STYLES = [("Intersection", "I", "#1f77b4", "o"),
              ("Union",        "U", "#d62728", "s"),
              ("Average",      "A", "#2ca02c", "^")]
    channels = [("Site", result["raw_SI"], result["raw_SU"], result["site"])]
    if result.get("bond") is not None:
        channels.append(("Bond", result["raw_BI"], result["raw_BU"], result["bond"]))

    fig, axes = plt.subplots(1, len(channels), figsize=(6.2 * len(channels), 5), squeeze=False)
    for ax, (name, rI, rU, res) in zip(axes[0], channels):
        mI, sI, mU, sU, mA, sA = _means_IUA(rI, rU)
        series = {"I": (mI, sI), "U": (mU, sU), "A": (mA, sA)}
        for label, key, color, marker in STYLES:
            m, s = series[key]
            ax.errorbar(x, m, yerr=s, fmt=marker, color=color, ms=6, capsize=3, alpha=0.9)
            pc, slope = res[key]["pc"], res[key]["A"]
            ax.plot(xs, slope * xs + pc, "--", color=color, lw=1.4, label=f"{label}: {pc:.4f}")
            ax.plot(0, pc, "X", color=color, ms=12, mec="black", mew=0.7, zorder=5)
        ax.set_title(f"{name} percolation")
        ax.set_xlabel(r"$L^{-1/\nu}$   ($\nu=4/3$,  $L\to\infty$ at 0)")
        ax.set_ylabel(r"crossing $p_c$")
        ax.set_xlim(left=-0.004)
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(loc="best", fontsize=9, title="I & U should meet at 0")
    fig.tight_layout()
    return fig


def convergence_figure(result):
    """Per-size crossing estimate vs L (linear axis), Intersection and Union criteria for site and
    bond -- the CLI plot_percolation_stats_IU view. As L grows I (which comes in high) and U (which
    comes in low) squeeze toward each other onto the same p_c; that visible closing of the bracket is
    the 'you can see it converge' sanity check. The AVERAGE is deliberately NOT drawn here -- it's the
    midpoint of the two, so it looks flat and hides the convergence (that's the extrapolation plot's
    job). Error bars are the crossing-distribution sample std (the width that narrows as ~L^-3/4)."""
    L = np.asarray(result["L"], dtype=float)
    series = [("Site I", result["raw_SI"], "blue", "o"),
              ("Site U", result["raw_SU"], "deepskyblue", "s")]
    if result.get("bond") is not None:
        series += [("Bond I", result["raw_BI"], "red", "^"),
                   ("Bond U", result["raw_BU"], "orange", "x")]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, raw, color, marker in series:
        m = np.array([np.mean(r) for r in raw])
        w = np.array([np.std(r, ddof=1) for r in raw])   # distribution width, not standard error
        ax.errorbar(L, m, yerr=w, fmt=marker + "-", color=color, ms=5, lw=1.0, capsize=3, label=label)
    ax.set_title("Per-size estimate vs L  (I & U criteria)")
    ax.set_xlabel(r"linear system size  $L$")
    ax.set_ylabel(r"crossing $p_c(L)$")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


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
