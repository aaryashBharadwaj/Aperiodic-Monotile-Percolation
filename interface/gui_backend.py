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
    save_result / list_saved / load_saved -> the 'Analyse saved' round-trip (paper_results/npz/)

The percolation RUN (any tiling, incl. long/overnight, with checkpoint/resume) goes through the one
consolidated runner runner.py — launched detached by the GUI's Run button OR from a console
(see REPRODUCE.md). This module holds the shared kernels + analysis both use.
"""
import os
import math
import numpy as np

# gui_backend lives in interface/; anchor data files, paper_results/npz/ and the worker script to the
# REPO ROOT (one level up), not to interface/. (Package imports resolve via the repo root that the
# entry points -- interface/gui_app.py and runner/runner.py -- put on sys.path.)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from builders.direct_graph_builder import build_neighbor_graph_fast, largest_square_center, graph_from_polygons
from builders.dual_graph_builder import collect_leaf_polygons, build_dual_from_polygons
from builders.graph_core import analyze_square_frame, analyze_rect_frame   # frame-cutters (direct AND dual)
from engine.percolation import (percolationStatsI_par, percolationStatsU_par,
                         percolationStatsBondI_par, percolationStatsBondU_par,
                         percolationStatsExponents_par, percolationStatsBondExponents_par)
from engine.analysis import extrapolate_pc_raw, isotropy_test, fit_exponents
from visualiser.run_tiling_render import _threshold_class
from generators.chevron_and_comet import periodic_graph, periodic_polys
from generators.periodic_tiling_generator import square_tiles, triangular_tris   # validated periodic geometry

S3 = math.sqrt(3)
NU = 4.0 / 3.0                                                    # imported by visualiser.figures (fss_figure)
# All results live under one folder: paper_results/npz (the .npz + job checkpoints) and
# paper_results/figures (rendered figures). Contents are regenerated/contingent and gitignored;
# only the folders are tracked (.gitkeep). RESULTS_DIR/PAPER_DIR are the same dir now.
RESULTS_DIR = os.path.join(REPO_ROOT, "paper_results", "npz")     # all result .npz + checkpoints (cwd-independent)
PAPER_DIR = RESULTS_DIR                                           # kept as an alias (single results folder)
FIG_DIR = os.path.join(REPO_ROOT, "paper_results", "figures")    # rendered figures (run_tiling_render output)

TRI = "Triangular → Honeycomb"          # the dual-of-triangular validation (exact honeycomb)
FAMILY = "Tile(a,b) family"
TILE11 = "Tile(1,1) periodic"           # periodic partner of the aperiodic Spectre (a=b, weakly chiral)
COMET_AP = "Comet aperiodic"            # hat's aperiodic arrangement folded to Tile(1,0)
CHEVRON_AP = "Chevron aperiodic"        # hat's aperiodic arrangement folded to Tile(0,1)

# Every tiling the portal offers, grouped for the UI.
TILINGS_MAIN = ["Hat", "Spectre", "Comet", "Chevron", COMET_AP, CHEVRON_AP, TILE11, FAMILY]
TILINGS_VALID = ["Square", "Penrose", TRI]
TILINGS = TILINGS_MAIN + TILINGS_VALID
GRAPHS = ["Direct (vertex)", "Dual (tile)"]

CATEGORY = {"Hat": "Aperiodic monotile", "Spectre": "Aperiodic monotile",
            "Comet": "Periodic (family limit)", "Chevron": "Periodic (family limit)",
            COMET_AP: "Aperiodic (family endpoint)", CHEVRON_AP: "Aperiodic (family endpoint)",
            TILE11: "Periodic (a=b, weakly chiral)",
            FAMILY: "One-parameter family", "Square": "Validation (exact)",
            "Penrose": "Validation (aperiodic)", TRI: "Validation (triangular + honeycomb)"}

# Which graph(s) each tiling admits. Penrose is a vertex graph; the honeycomb IS the dual of the
# triangular tiling, so that one is dual-only. The square lattice is self-dual (both give a square
# lattice), so it offers both.
GRAPHS_FOR = {"Hat": GRAPHS, "Spectre": GRAPHS, "Comet": GRAPHS, "Chevron": GRAPHS,
              COMET_AP: GRAPHS, CHEVRON_AP: GRAPHS,
              TILE11: GRAPHS, "Square": GRAPHS, "Penrose": ["Direct (vertex)"], TRI: GRAPHS}

# Percolation patch control per tiling: (label, min, max, default, help). Hat/spectre go to the
# production r=6 -- that's the headline result, it must be reachable here.
PATCH_CTL = {
    "Hat":     ("Patch recursion  r", 2, 6, 4, "Hat metatile inflation depth. r=6 is the production patch (heavy: ~1M+ nodes)."),
    "Spectre": ("Patch level",        2, 6, 4, "Spectre substitution depth."),
    "Comet":   ("Block size  (cells)", 20, 160, 60, "Periodic block: cells per side (paper uses 150)."),
    "Chevron": ("Block size  (cells)", 20, 160, 60, "Periodic block: cells per side (paper uses 150)."),
    COMET_AP:  ("Patch recursion  r", 2, 6, 4, "Hat metatile depth, folded to the comet endpoint (Tile(1,0)). r=6 is heavy (~1M+ nodes)."),
    CHEVRON_AP:("Patch recursion  r", 2, 6, 4, "Hat metatile depth, folded to the chevron endpoint (Tile(0,1)). r=6 is heavy (~1M+ nodes)."),
    TILE11:    ("Patch reach",         40, 1200, 200, "Periodic Tile(1,1): physical grow radius; the solid square window is ~0.7x this."),
    "Square":  ("Grid size  n",        20, 160, 60, "n x n square lattice (the textbook check: site 0.5927, bond 0.5)."),
    "Penrose": ("Subdivisions",        4, 9, 7, "Penrose inflation steps (more = finer graph). Deep, converged Penrose runs use the CLI (s=13)."),
    TRI:       ("Grid size  n",        20, 100, 60, "n x n triangular block: direct graph = the triangular lattice, dual = the honeycomb."),
}

# Visualise (render) size control -- deliberately SMALL so the figure is actually drawable/legible.
RENDER_CTL = {
    "Hat":     ("Inflation level", 1, 4, 2),
    "Spectre": ("Level",           1, 4, 3),
    "Comet":   ("Cells",           3, 14, 8),
    "Chevron": ("Cells",           3, 14, 8),
    COMET_AP:  ("Inflation level", 1, 4, 2),
    CHEVRON_AP:("Inflation level", 1, 4, 2),
    TILE11:    ("Reach",           8, 40, 16),
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
def build_graph(tiling, graph_type, patch, a=1.0, b=S3, scale=None):
    """Build the requested graph and package everything the sweep needs.
    Returns a picklable bundle: {coords, neighbors, center, side, name, kind, nodes}."""
    name = resolve_member(tiling, a, b)
    if name is None:
        raise ValueError("Tile(0,0) is degenerate -- nothing to build.")
    is_dual = graph_type.startswith("Dual")
    center_src = None   # if set, resolve the frame centre from these tile-vertex polygons instead of
                        # the graph nodes. For an APERIODIC dual the nodes are tile CENTROIDS, whose
                        # non-uniform density fools largest_square_center into a wrong, off-centre
                        # window (a different patch of the tiling -> wrong p_c AND a spurious direction
                        # bias). Centring on the vertices puts the dual on the SAME physical square as
                        # the direct graph (the paper's shared centre). Periodic duals are centre-
                        # invariant, so they keep the node-based centre unchanged.

    if name in ("Hat", "Spectre"):
        patch_obj, lvl = _hat_patch(patch) if name == "Hat" else _spectre_patch(patch)
        if is_dual:
            polys = collect_leaf_polygons(patch_obj, lvl)   # lvl = patch+1 (hat) or None (spectre), same as direct
            coords, neighbors, _ = build_dual_from_polygons(polys)
            center_src = polys
        else:
            coords, neighbors = build_neighbor_graph_fast(patch_obj, level=lvl)
    elif name in ("Comet", "Chevron"):
        if is_dual:
            polys, _span = periodic_polys(name.lower(), ncells=patch)   # same block as the direct path
            coords, neighbors, _ = build_dual_from_polygons(polys)
        else:
            coords, neighbors, _span = periodic_graph(name.lower(), ncells=patch)
    elif name in (COMET_AP, CHEVRON_AP):
        # Aperiodic comet/chevron: the hat's aperiodic arrangement FOLDED to the family endpoint
        # (generators/aperiodic_collapse). patch = hat metatile depth. Direct/dual come out of the
        # shared builders once the collapsed-edge endpoints de-dup. The dual nodes are tile centroids,
        # so (like the hat/spectre dual) centre the window on the tile VERTICES, not the centroids.
        from generators.aperiodic_collapse import collapse_graph
        which = "comet" if name == COMET_AP else "chevron"
        coords, neighbors, polys = collapse_graph(which, "dual" if is_dual else "direct", patch)
        if is_dual:
            center_src = polys
    elif name == TILE11:
        # Periodic Tile(1,1): patch = grow radius (reach). Same direct/dual builders as everything else;
        # the tiling itself is built by edge-matching in generators/tile11_periodic (see that module).
        from generators.tile11_periodic import tile11_polys, tile11_graph
        if is_dual:
            polys, _span = tile11_polys(patch)
            coords, neighbors, _ = build_dual_from_polygons(polys)
        else:
            coords, neighbors, _span = tile11_graph(patch)
    elif name == "Penrose":
        from builders.penrose_graph_builder import build_penrose_neighbor_graph
        # patch = subdivisions (DENSITY); `scale` sets the physical EXTENT and MUST exceed the sweep's
        # L_max or the frame pokes outside the patch (sizes skip). The runner passes scale = 2*L_max
        # (the original run_penrose rule); scale=None keeps the small exploratory default.
        coords, neighbors, _edges = build_penrose_neighbor_graph(_penrose_tiling(patch, scale=scale or 200))
        is_dual = False          # Penrose is a vertex graph
        # TODO(author): NEW -- Penrose now uses the shared largest_square_center like every other tiling
        # (was center_mode="bbox"). Its sparse-cloud guard coarsens the grid for Penrose's ~1-node/cell
        # density, returning a real inscribed square instead of the bbox min-extent that overshot the
        # pentagon. NOTE: this shifts Penrose's window (centre + side) -> its p_c must be re-run.
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

    center_pts = np.concatenate(center_src, axis=0) if center_src is not None else coords
    cx, cy, side = largest_square_center(center_pts)

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
    # One generic frame-cutter for both direct and dual (a node is a vertex or a centroid; the
    # windowing is identical). bundle["coords"]/["neighbors"] already hold the right cloud per kind.
    return analyze_square_frame(bundle["coords"], bundle["neighbors"], L,
                                boundary_thickness=bt, center_x=cx, center_y=cy)


def _frame_usable(fd):
    return (fd.get("node_count", 0) >= 4 and min(
        len(fd["top_boundary_nodes"]), len(fd["bottom_boundary_nodes"]),
        len(fd["left_boundary_nodes"]), len(fd["right_boundary_nodes"])) > 0)


def run_one(bundle, L, T, seed_base, bt=1.0, exponents=False, nworkers=0):
    """Percolate ONE L-window (the 4 estimators). Returns a dict; usable=False if the frame is empty
    or pokes outside the tiling. Factored out so the GUI can drive the sweep one size per rerun --
    that's what makes it interruptible (a Stop button is processed between steps).

    nworkers sets how many trials run concurrently (the parallel dimension is over independent trials,
    on nogil numba kernels -> real multi-core). 0 = the engine default (cpu_count-1 / PERCOLATE_THREADS);
    an explicit value is threaded through so the runner's --threads flag can dial it per run."""
    warm_up()
    # 0 -> let each stats class fall back to its _NW default; a positive value overrides it.
    nwkw = {"nworkers": int(nworkers)} if nworkers and int(nworkers) > 0 else {}
    # TODO(author): NEW -- inscribed-square cap with a 5% safety margin. bundle["side"] is the compact
    # filled core from largest_square_center; a window past it pokes into the ragged fringe (measured
    # off-tile AT L=side is ~0.1% hat / ~0.5% spectre via an independent point-in-polygon check, and it
    # grows from there). Capping at 0.95*side keeps the largest window just inside the core -> ~0%
    # clipped, uniformly across every tiling. (A node-COUNT fullness test was tried and rejected: fringe
    # spikes keep the count high, so it accepted windows 50% past the core.) A larger window would still
    # pass the crude per-side node guard below (fringe spikes touch each edge), so this is the real
    # backstop for an over-large --lmax / L_max; the paper presets all sit under 0.95*side.
    side = bundle.get("side")
    if side is not None and L > 0.95 * side + 1e-9:
        return {"usable": False, "L": float(L)}
    fd = _frame_data(bundle, L, bt)
    if not _frame_usable(fd):
        return {"usable": False, "L": float(L)}
    A = (fd["sub_graph_nodes"], fd["sub_graph_neighbors"], fd["top_boundary_nodes"],
         fd["bottom_boundary_nodes"], fd["left_boundary_nodes"], fd["right_boundary_nodes"])
    E = (fd["sub_graph_nodes"], fd["sub_graph_edges"], fd["top_boundary_nodes"],
         fd["bottom_boundary_nodes"], fd["left_boundary_nodes"], fd["right_boundary_nodes"])
    # keep the intersection objects: pR/pD (site) and bond_pR/bond_pD validate the p_A estimators
    si = percolationStatsI_par(*A, T, master_seed=seed_base + 0, **nwkw)
    bi = percolationStatsBondI_par(*E, T, master_seed=seed_base + 2, **nwkw)
    out = {"usable": True, "L": float(L), "N": fd["node_count"],
           "SI": si.trialResults, "pR": si.pR, "pD": si.pD,
           "SU": percolationStatsU_par(*A, T, master_seed=seed_base + 1, **nwkw).trialResults,
           "BI": bi.trialResults, "bond_pR": bi.pR, "bond_pD": bi.pD,
           "BU": percolationStatsBondU_par(*E, T, master_seed=seed_base + 3, **nwkw).trialResults}
    if exponents:
        # OPT-IN largest-cluster pass (extra sweep, s_max tracked incrementally): records the per-trial
        # largest cluster at first-spanning (the incipient infinite cluster) -> d_f. seed_base+4
        # keeps it independent of the four threshold seeds.
        ex = percolationStatsExponents_par(*A, T, master_seed=seed_base + 4, **nwkw)
        out["s_max"] = ex.s_max        # union-onset s_max (d_f; original behaviour)
        out["s_max_i"] = ex.s_inter    # intersection-onset s_max (the d_f onset-definition bracket)
        # bond d_f (mass in sites): universal, so it cross-checks the site value. seed_base+5 (spare slot).
        exb = percolationStatsBondExponents_par(*E, T, master_seed=seed_base + 5, **nwkw)
        out["bond_s_max"] = exb.s_max
        out["bond_s_max_i"] = exb.s_inter
    return out


def run_crossing(bundle, W, H, T, seed_base, bt=1.0, nworkers=0):
    """Percolate ONE rectangular W x H window and return the per-trial left-right spanning onset
    densities. This is what the Cardy crossing test needs: it mirrors run_one but cuts a RECTANGLE
    (aspect a = W/H) and returns the raw left-right crossings (pR) rather than the p_c estimators.
    usable=False if the frame is empty or the rectangle pokes outside the tiling's solid core."""
    warm_up()
    side = bundle.get("side")
    # the rectangle must sit inside the solid inscribed square; cap on its larger half-extent
    if side is not None and max(W, H) > 0.95 * side + 1e-9:
        return {"usable": False}
    cx, cy = bundle["center"]
    fd = analyze_rect_frame(bundle["coords"], bundle["neighbors"], W, H, boundary_thickness=bt,
                            center_x=cx, center_y=cy)
    if not _frame_usable(fd):
        return {"usable": False}
    A = (fd["sub_graph_nodes"], fd["sub_graph_neighbors"], fd["top_boundary_nodes"],
         fd["bottom_boundary_nodes"], fd["left_boundary_nodes"], fd["right_boundary_nodes"])
    nwkw = {"nworkers": int(nworkers)} if nworkers and int(nworkers) > 0 else {}
    si = percolationStatsI_par(*A, T, master_seed=seed_base, **nwkw)   # pR = left-right onset density
    return {"usable": True, "N": fd["node_count"], "W": float(W), "H": float(H), "pR": si.pR}


def run_cardy(bundle, windows, n_aspects=20, aspect_min=0.5, aspect_max=2.0,
              T=4000, seed=123456789, bt=1.0, nworkers=0):
    """Cardy crossing-probability test on a pre-built bundle (same engine the CLI cardy_runner uses).
    For each window size L: anchor at the p where the SQUARE crosses at 1/2 (p* = median onset), then
    read the left-right crossing probability at each aspect ratio a = W/H (W=L*sqrt(a), H=L/sqrt(a),
    area = L^2) at that same p*. Those are free predictions Cardy's exact curve must reproduce.
    Returns the dict cardy_figure/analysis consume: {aspects, cardy, windows, Rh, pstar, trials,
    noise_floor}. Windows too big for the patch are skipped."""
    from engine.cardy import cardy_pi_h, crossing_probability
    aspects = np.round(np.geomspace(aspect_min, aspect_max, int(n_aspects)), 4)
    cardy = np.array([cardy_pi_h(a) for a in aspects])
    floor = float(np.sqrt(0.25 / max(1, T)))
    windows_done, Rh_rows, pstar_list, rms_list = [], [], [], []
    for wi, L in enumerate(windows):
        sq = run_crossing(bundle, L, L, T, seed + wi * 1000, bt, nworkers=nworkers)
        if not sq.get("usable"):
            continue
        pstar = float(np.median(np.asarray(sq["pR"], float)))
        Rh = np.full(len(aspects), np.nan)
        for ai, asp in enumerate(aspects):
            r = run_crossing(bundle, L * np.sqrt(asp), L / np.sqrt(asp), T,
                             seed + wi * 1000 + ai + 1, bt, nworkers=nworkers)
            if r.get("usable"):
                Rh[ai] = crossing_probability(r["pR"], pstar)
        windows_done.append(float(L)); Rh_rows.append(Rh); pstar_list.append(pstar)
        rms_list.append(float(np.sqrt(np.nanmean((Rh - cardy) ** 2))))
    return {"aspects": aspects, "cardy": cardy, "windows": np.array(windows_done, float),
            "Rh": np.array(Rh_rows, float), "pstar": np.array(pstar_list, float),
            "rms": np.array(rms_list, float), "trials": int(T), "noise_floor": floor}


def _pick_lmin(valid, floor=50):
    """Use L_min=floor (the paper's cutoff) for the direction-bias fit if it leaves >=3 sizes; else 0
    (use all), since exploratory GUI sweeps often don't reach L=50 with enough points."""
    return floor if sum(1 for L in valid if L >= floor) >= 3 else 0


def extrapolate_result(valid, rSI, rSU, rBI, rBU, skipped, raw_pR=None, raw_pD=None,
                       raw_bond_pR=None, raw_bond_pD=None, raw_smax=None, raw_smax_inter=None,
                       raw_bond_smax=None, raw_bond_smax_inter=None):
    """Assemble the result dict + the FULL analysis: p_c extrapolation (I/U/A); the direction-bias
    check (if pR/pD present) -- site (raw_pR/raw_pD -> 'isotropy') and bond (raw_bond_* -> 'isotropy_bond');
    and d_f (if raw_smax present -> 'exponents'). Needs >=3 sizes; fields stay None otherwise. Works on
    a PARTIAL sweep too, so a stopped run still yields whatever it collected."""
    out = {"L": valid, "raw_SI": rSI, "raw_SU": rSU, "raw_BI": rBI, "raw_BU": rBU,
           "skipped": skipped, "raw_pR": raw_pR, "raw_pD": raw_pD,
           "raw_bond_pR": raw_bond_pR, "raw_bond_pD": raw_bond_pD, "raw_smax": raw_smax,
           "raw_smax_inter": raw_smax_inter,
           "raw_bond_smax": raw_bond_smax, "raw_bond_smax_inter": raw_bond_smax_inter,
           "site": None, "bond": None, "isotropy": None, "isotropy_bond": None,
           "exponents": None, "exponents_inter": None,
           "exponents_bond": None, "exponents_bond_inter": None, "lmin": None}
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
        if raw_smax is not None:
            try:
                out["exponents"] = fit_exponents(valid, raw_smax)
            except Exception:
                pass
        # d_f from the intersection-onset s_max: the other end of the onset-definition bracket
        if raw_smax_inter is not None and all(s is not None for s in raw_smax_inter):
            try:
                out["exponents_inter"] = fit_exponents(valid, raw_smax_inter)
            except Exception:
                pass
        # bond d_f (union onset, + its intersection bracket): the site/bond universality cross-check
        if raw_bond_smax is not None and all(s is not None for s in raw_bond_smax):
            try:
                out["exponents_bond"] = fit_exponents(valid, raw_bond_smax)
            except Exception:
                pass
        if raw_bond_smax_inter is not None and all(s is not None for s in raw_bond_smax_inter):
            try:
                out["exponents_bond_inter"] = fit_exponents(valid, raw_bond_smax_inter)
            except Exception:
                pass
    return out


def save_result(result, member, kind, seed, T, name=None, out_dir=RESULTS_DIR):
    """Write a completed/partial result to paper_results/npz/ in the PercolationResults .npz format (so
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
    # One file per run: the direction-bias (pR/pD) and d_f (s_union/s_inter) arrays are now folded into
    # PercolationResults instead of separate _iso.npz / _exp.npz companions.
    PercolationResults(tiling_type=f"{member}_{kind}", seed=int(seed), trials=int(T),
                       L_values=result["L"], raw_SI=result["raw_SI"], raw_SU=result["raw_SU"],
                       raw_BI=result["raw_BI"], raw_BU=result["raw_BU"],
                       raw_pR=result.get("raw_pR"), raw_pD=result.get("raw_pD"),
                       raw_bond_pR=result.get("raw_bond_pR"), raw_bond_pD=result.get("raw_bond_pD"),
                       raw_smax=result.get("raw_smax"), raw_smax_inter=result.get("raw_smax_inter"),
                       raw_bond_smax=result.get("raw_bond_smax"),
                       raw_bond_smax_inter=result.get("raw_bond_smax_inter"),
                       extra_meta={"source": "gui"}).save(path)
    return path


def delete_result(fname, out_dir=None):
    """Remove a saved result and its _iso/_exp sidecars from disk -- the GUI 'discard' path for an
    exploratory run the user didn't ask to keep. Silently ignores anything already gone."""
    base = out_dir or RESULTS_DIR
    targets = [fname, fname[:-4] + "_iso.npz", fname[:-4] + "_exp.npz"]
    for t in targets:
        p = os.path.join(base, t)
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


def list_saved(out_dir=None):
    """Result files for 'Analyse saved', newest first. Excludes _iso/_exp companions and checkpoints.
    All results live in paper_results/npz/; pass out_dir to scan a different folder."""
    dirs = [out_dir] if out_dir else [RESULTS_DIR]
    seen = set(); out = []
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        files = [f for f in os.listdir(d)
                 if f.endswith(".npz") and not f.endswith("_iso.npz")
                 and not f.endswith("_exp.npz") and "CHECKPOINT" not in f]
        for f in sorted(files, key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True):
            if f not in seen:
                seen.add(f); out.append(f)
    return out


def load_saved(fname, out_dir=None):
    """Load a saved .npz into the same result dict the live runs produce, so every plot/metric works
    identically. The direction-bias and d_f arrays now live inside the file; for OLDER runs that kept
    them in _iso.npz / _exp.npz sidecars we still read those. Returns (result, meta). Searches
    paper_results/npz/ (or just out_dir if given)."""
    from engine.results import PercolationResults
    dirs = [out_dir] if out_dir else [RESULTS_DIR]
    base = next((d for d in dirs if d and os.path.exists(os.path.join(d, fname))), dirs[-1])
    r = PercolationResults.load(os.path.join(base, fname))
    raw_pR, raw_pD = r.raw_pR, r.raw_pD
    raw_bond_pR, raw_bond_pD = r.raw_bond_pR, r.raw_bond_pD
    raw_smax, raw_smax_inter = r.raw_smax, r.raw_smax_inter
    raw_bond_smax, raw_bond_smax_inter = r.raw_bond_smax, r.raw_bond_smax_inter

    # backward-compat: older runs stored these in sidecar files, not in the main .npz
    if raw_pR is None:
        isop = os.path.join(base, fname[:-4] + "_iso.npz")
        if os.path.exists(isop):
            d = np.load(isop); nL = len(d["L_values"])
            raw_pR = [d[f"pR_{i}"] for i in range(nL)]; raw_pD = [d[f"pD_{i}"] for i in range(nL)]
            if "bpR_0" in d:
                raw_bond_pR = [d[f"bpR_{i}"] for i in range(nL)]
                raw_bond_pD = [d[f"bpD_{i}"] for i in range(nL)]
    if raw_smax is None:
        expp = os.path.join(base, fname[:-4] + "_exp.npz")
        if os.path.exists(expp):
            e = np.load(expp); nL = len(e["L_values"])
            raw_smax = [e[f"smax_{i}"] for i in range(nL)]
            if "smaxI_0" in e: raw_smax_inter = [e[f"smaxI_{i}"] for i in range(nL)]
            if "bsmax_0" in e: raw_bond_smax = [e[f"bsmax_{i}"] for i in range(nL)]
            if "bsmaxI_0" in e: raw_bond_smax_inter = [e[f"bsmaxI_{i}"] for i in range(nL)]

    res = extrapolate_result(list(r.L_values), r.raw_SI, r.raw_SU, r.raw_BI, r.raw_BU, [],
                             raw_pR, raw_pD, raw_bond_pR, raw_bond_pD,
                             raw_smax, raw_smax_inter, raw_bond_smax, raw_bond_smax_inter)
    return res, {"member": r.tiling_type, "kind": "", "seed": r.seed, "T": r.trials,
                 "timestamp": r.timestamp}


# ----------------------------------------------------------------------------- time estimate
# Flat per-run cost of spawning a fresh worker process: import the stack (numba/scipy/matplotlib)
# + numba JIT warm-up. Measured ~3s import + fast (cached) JIT; padded for machine/cold-cache slop.
LAUNCH_OVERHEAD_S = 6.0                                          # imported by interface.estimate (plan_run)


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


def paper_preset(member, kind):
    """The production parameters used in the paper for this object, or None if it isn't a headline
    object (Penrose/triangular are validations, not headline objects). These match the paper runs
    documented in REPRODUCE.md, so the GUI loads exactly 'the real run' in one click:
      Hat     -- r=6, L=10..1000 step 10, T=1000
      Spectre -- level 6, T=500, L=20..int(0.92*side), step max(10, round(Lmax/40/10)*10)
      Comet/Chevron -- 150 cells, T=500, L=20..~0.95*side, step 20
    Spectre/periodic L ranges are computed from the patch extent (via geometry); all use seed 123456789."""
    from interface.estimate import geometry   # lazy: estimate imports gui_backend, so keep this off module load
    seed = 123456789
    if member == "Hat":
        return {"patch": 6, "L_min": 10.0, "L_max": 1000.0, "gap": 10.0, "T": 1000, "seed": seed}
    if member == "Spectre":
        # Direct and dual MUST span the same physical window -- comparing p_c across the two graphs only
        # means anything on a shared extent. The two node clouds give different solid squares, so cap L_max
        # at the SMALLER (the tighter ceiling binds both); this makes the preset kind-independent.
        sides = [geometry("Spectre", k, 6) for k in ("direct", "dual")]
        side = min([s[1] for s in sides if s], default=400.0)
        Lmax = int((side * 0.92) // 10 * 10)
        gap = max(10.0, round(Lmax / 40 / 10) * 10)
        return {"patch": 6, "L_min": 20.0, "L_max": float(Lmax), "gap": float(gap), "T": 500, "seed": seed}
    if member in ("Comet", "Chevron"):
        ge = geometry(member, kind, 150)
        side = ge[1] if ge else 300.0
        Lmax = int(side * 0.95 // 10 * 10)
        return {"patch": 150, "L_min": 20.0, "L_max": float(Lmax), "gap": 20.0, "T": 500, "seed": seed}
    if member in (COMET_AP, CHEVRON_AP):
        # Aperiodic endpoints (folded hat): inflation model, like Spectre. Direct and dual have
        # different solid squares, so cap L_max at the SMALLER so both graphs span the same window.
        sides = [geometry(member, k, 6) for k in ("direct", "dual")]
        side = min([s[1] for s in sides if s], default=300.0)
        Lmax = int((side * 0.92) // 10 * 10)
        gap = max(10.0, round(Lmax / 40 / 10) * 10)
        return {"patch": 6, "L_min": 20.0, "L_max": float(Lmax), "gap": float(gap), "T": 1000, "seed": seed}
    if member == TILE11:
        ge = geometry(member, "direct", 200)
        side = ge[1] if ge else 140.0
        Lmax = int(side * 0.92 // 10 * 10)
        gap = max(10.0, round(Lmax / 30 / 10) * 10)
        return {"patch": 200, "L_min": 20.0, "L_max": float(Lmax), "gap": float(gap), "T": 500, "seed": seed}
    return None
