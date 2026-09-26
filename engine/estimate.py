"""Machine-independent geometry + solid-window models for the pipeline, from geometry alone.

    geometry(member, kind, patch)  -> (nodes, inscribed side) WITHOUT building the patch
    solid_window(member,...)       -> the measured largest fully-tiled window (build_graph / run_one cap)
    calibrate_geometry / calibrate_solid_windows -> (re)fit the two machine-INDEPENDENT on-disk models
                                                    (gui_geometry.json, solid_windows.json)

These feed build_graph's window cap and paper_preset's L_max, so a patch's size and usable window are
known WITHOUT building the (possibly huge) graph. Shared kernels/constants (build_graph, TRI, ...) come
from engine.pipeline; the dependency is one-way (a lazy import in build_graph breaks the cycle).
"""
import os
import json
import time

from engine.pipeline import build_graph, TRI, TILE11, COMET_AP, CHEVRON_AP

_HERE = os.path.dirname(os.path.abspath(__file__))                 # this interface/ folder (the GUI's config lives beside it)
GEOM_PATH = os.path.join(_HERE, "gui_geometry.json")               # machine-independent: precomputed tiling geometry
SOLID_PATH = os.path.join(_HERE, "solid_windows.json")             # measured max fully-tiled window per (member,kind,patch)


# ----------------------------------------------------------------------------- geometry (no-build size)
# The tiling's in-frame node DENSITY and the patch's inscribed-square SIDE are machine-INDEPENDENT.
# Density is scale-invariant for the inflation tilings (inflating the hat tiles a bigger area at the
# same density), so we predict a big patch's size WITHOUT building it -- build only the cheap patches
# once, extrapolate the rest. The UI shows the node count + usable window for r=6 instantly; the heavy
# graph is built only when the run actually happens (in a terminal).
def _measure_geom(member, kind, patch):
    """Build a patch and return (total_nodes, inscribed side). Machine-independent geometry only --
    the pre-run readout needs the node count and the inscribed-square side; nothing times the build."""
    gt = "Dual (tile)" if kind == "dual" else "Direct (vertex)"
    b = build_graph(member, gt, patch)
    return int(b["nodes"]), float(b["side"])


def calibrate_geometry(save=True):
    """Populate gui_geometry.json by building only CHEAP patches; inflation tilings extrapolate up.
    Machine-independent (pure geometry) -- run once and ship it. Stores the whole-patch node density
    and the inscribed side, from which geometry() predicts any patch's node count without building it."""
    G = {}
    # Hat / spectre / aperiodic comet+chevron: inflation model. Build r=2..5; densities constant, side
    # grows by lambda. lambda is fit from the TWO LARGEST built patches (r=4->5), NOT small ones -- the
    # small-patch ratio undershoots the asymptotic inflation factor, and extrapolating it two levels to
    # the production r=6 compounds the error badly (e.g. spectre was -57%). One step from r=5 is accurate.
    for member in ("Hat", "Spectre", COMET_AP, CHEVRON_AP):
        for kind in ("direct", "dual"):
            tbl = {}
            for r in (2, 3, 4, 5):
                n, s = _measure_geom(member, kind, r)
                tbl[str(r)] = [int(n), float(s)]
            s4, (n5, s5) = tbl["4"][1], (tbl["5"][0], tbl["5"][1])
            G[f"{member}|{kind}"] = {"model": "inflate", "table": tbl, "r_built": 5,
                                     "lambda": s5 / s4, "density": n5 / (s5 * s5)}
    # Penrose: fixed extent, density grows with subdivisions -> table each (all cheap).
    tbl = {}
    for s in range(4, 10):
        n, side = _measure_geom("Penrose", "direct", s)
        tbl[str(s)] = [int(n), float(side)]
    G["Penrose|direct"] = {"model": "table", "table": tbl}
    # Periodic / triangular: block grows linearly with the cell count; densities constant.
    for member, kind in [("Comet", "direct"), ("Comet", "dual"), ("Chevron", "direct"),
                         ("Chevron", "dual"), (TRI, "direct"), (TRI, "dual"),
                         ("Square", "direct"), ("Square", "dual"),
                         (TILE11, "direct"), (TILE11, "dual")]:
        n1, s1 = _measure_geom(member, kind, 40)
        n2, s2 = _measure_geom(member, kind, 70)
        G[f"{member}|{kind}"] = {"model": "linear", "k": 0.5 * (s1 / 40 + s2 / 70),
                                 "density": 0.5 * (n1 / (s1 * s1) + n2 / (s2 * s2))}
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


def calibrate_solid_windows(save=True, prod=None, merge=True):
    """Measure the largest fully-tiled+surrounded window for each PRODUCTION (member, kind, patch), via
    the off-tile oracle (builders.solid_window.measure_solid_L). This replaces the 0.95-of-detected-side
    fudge with a measured bound -- the number the presets/GUI/run_one all use. Heavy (one patch build per
    entry); run once and ship solid_windows.json.

    `prod` overrides the default (member, patch) list -- pass the batch's patches when they differ from a
    previous run. With merge=True the results are folded into the existing json (so earlier entries the new
    batch doesn't touch survive). Each entry is timed, printed, and saved incrementally, so a long build
    that is interrupted keeps the windows already measured.

    Only patches whose sweep L_max can actually EXCEED 0.95*side need measuring -- for the clean periodic
    lattices (Square/Triangular) and the fixed-extent Penrose the fallback is already inside the solid
    region, so they are deliberately not listed. The default list tracks scripts/aws_runs_v2.sh.

    The r=7 comet/chevron aperiodic FOLDS are also omitted on purpose: their sweep L_max (1130/1200) sits
    well inside the solid window (measured Spectre|7=1690 scales to ~1370/~1930 for them), so 0.95*side is
    already safe there -- and measuring them means ~8M matplotlib Paths in the off-tile oracle, which needs
    far more RAM than the fold itself (the fold is only ~3GB at r=7). Not worth it for a non-binding cap."""
    from builders.solid_window import measure_solid_L
    from engine.pipeline import TILE11 as _T11
    if prod is None:
        # the cap-BINDING v2 patches only: the two periodic folds whose L_max reaches their window, plus
        # Tile(1,1) and the r=6 hat (both comfortably inside, refreshed for completeness). Spectre|7 was
        # measured once (=1690, non-binding) and survives via merge=True; it isn't rebuilt here.
        prod = [("Comet", 1300), ("Chevron", 1300), (_T11, 860), ("Hat", 6)]
    S = {}
    if merge:
        try:
            with open(SOLID_PATH) as f:
                S = json.load(f)
        except Exception:
            S = {}
    for member, patch in prod:                                 # once per member: the window is geometric,
        t0 = time.perf_counter()
        S[f"{member}|{patch}"] = measure_solid_L(member, patch)  # identical for the direct/dual graphs on it
        print(f"  solid_window {member}|{patch} = {S[f'{member}|{patch}']}  ({time.perf_counter()-t0:.1f}s)", flush=True)
        if save:                                                # incremental: survive an interrupt
            with open(SOLID_PATH, "w") as f:
                json.dump(S, f, indent=2)
    return S


_SOLID = None
def solid_window(member, kind, patch):
    """Measured solid-window L for `member` at `patch`, or None if that member wasn't calibrated
    (callers then fall back to the conservative 0.95 x detected side). `kind` is accepted for a uniform
    call signature but ignored: the solid window is geometric, identical for the direct and dual graphs."""
    global _SOLID
    if _SOLID is None:
        try:
            with open(SOLID_PATH) as f:
                _SOLID = json.load(f)
        except Exception:
            _SOLID = {}
    return _SOLID.get(f"{member}|{int(round(patch))}")


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


