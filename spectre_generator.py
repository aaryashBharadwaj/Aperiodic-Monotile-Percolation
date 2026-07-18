"""Spectre monotile generator — Tile(1,1) with reflections forbidden (chiral).

Deliberately mirrors hat_generator.py: it reuses the SAME affine primitives
(pt, mul, trot, ttrans, transPt) and the SAME metatile+substitution method.
The only differences are the base tile (the Spectre polygon) and the substitution
rules, which are the two-rule chiral system of Smith-Myers-Kaplan-Goodman-Strauss:
    Spectre -> 1 Mystic + 7 Spectres,   Mystic -> 1 Mystic + 6 Spectres.
Substitution data ported from the reference implementation github.com/shrx/spectre
(itself a port of Kaplan's construction), so the tiling can be verified against theirs.
"""
import math
from hat_generator import pt, mul, trot, ttrans, transPt

SQ3 = math.sqrt(3)
IDENTITY = [1, 0, 0, 0, 1, 0]

# The Spectre base outline (14 edges, all equal length, angles 90/120 deg) = Tile(1,1).
SPECTRE_POINTS = [
    pt(0, 0),                      pt(1.0, 0.0),
    pt(1.5, -SQ3/2),               pt(1.5 + SQ3/2, 0.5 - SQ3/2),
    pt(1.5 + SQ3/2, 1.5 - SQ3/2),  pt(2.5 + SQ3/2, 1.5 - SQ3/2),
    pt(3 + SQ3/2, 1.5),            pt(3.0, 2.0),
    pt(3 - SQ3/2, 1.5),            pt(2.5 - SQ3/2, 1.5 + SQ3/2),
    pt(1.5 - SQ3/2, 1.5 + SQ3/2),  pt(0.5 - SQ3/2, 1.5 + SQ3/2),
    pt(-SQ3/2, 1.5),               pt(0.0, 1.0),
]

# Nine metatile types (Kaplan's Gamma..Psi). Gamma is the "Mystic" (two spectres).
TILE_NAMES = ["Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi"]


def transTo(p, q):
    return ttrans(q['x'] - p['x'], q['y'] - p['y'])


# The 4 reference points ("quad") that the substitution uses to align tiles.
QUAD_IDX = [3, 5, 7, 11]


# --- Base tile (leaf) — mirrors HatTile ---
class SpectreTile:
    def __init__(self, shape, label):
        self.shape = shape
        self.label = label
        self.children = None
        self.quad = [shape[i] for i in QUAD_IDX]
    def draw(self, S, ax):
        poly = [transPt(S, p) for p in self.shape]
        ax.fill([p['x'] for p in poly], [p['y'] for p in poly],
                facecolor=_col(self.label), edgecolor=(0, 0, 0), linewidth=0.25)


# --- Supertile — mirrors hat_generator.MetaTile (children of {'T','geom'}) ---
class MetaTile:
    def __init__(self, children, quad):
        self.children = children     # list of {'T': transform, 'geom': tile}
        self.quad = quad             # 4 reference points used by the substitution
        self.shape = quad
    def draw(self, S, ax):
        for ch in self.children:
            ch['geom'].draw(mul(S, ch['T']), ax)


def buildSpectreBase():
    base = {lbl: SpectreTile(SPECTRE_POINTS, lbl) for lbl in TILE_NAMES if lbl != "Gamma"}
    mystic = MetaTile(
        [
            {'geom': SpectreTile(SPECTRE_POINTS, "Gamma1"), 'T': IDENTITY},
            {'geom': SpectreTile(SPECTRE_POINTS, "Gamma2"),
             'T': mul(ttrans(SPECTRE_POINTS[8]['x'], SPECTRE_POINTS[8]['y']), trot(math.pi / 6))},
        ],
        [SPECTRE_POINTS[3], SPECTRE_POINTS[5], SPECTRE_POINTS[7], SPECTRE_POINTS[11]],
    )
    base["Gamma"] = mystic
    return base


def buildSupertiles(sys):
    quad = sys["Delta"].quad
    R = [-1, 0, 0, 0, 1, 0]     # in-substitution reflection of the reference frame (tiling stays chiral)
    rules = [[60, 3, 1], [0, 2, 0], [60, 3, 1], [60, 3, 1], [0, 2, 0], [60, 3, 1], [-120, 3, 3]]

    transformations = [IDENTITY]
    total_angle = 0
    rotation = IDENTITY
    tq = list(quad)
    for ang, frm, to in rules:
        if ang != 0:
            total_angle += ang
            rotation = trot(math.radians(total_angle))
            tq = [transPt(rotation, q) for q in quad]
        ttt = transTo(tq[to], transPt(transformations[-1], quad[frm]))
        transformations.append(mul(ttt, rotation))
    transformations = [mul(R, t) for t in transformations]

    super_rules = {
        "Gamma":  ["Pi",  "Delta", None,  "Theta", "Sigma", "Xi",  "Phi",    "Gamma"],
        "Delta":  ["Xi",  "Delta", "Xi",  "Phi",   "Sigma", "Pi",  "Phi",    "Gamma"],
        "Theta":  ["Psi", "Delta", "Pi",  "Phi",   "Sigma", "Pi",  "Phi",    "Gamma"],
        "Lambda": ["Psi", "Delta", "Xi",  "Phi",   "Sigma", "Pi",  "Phi",    "Gamma"],
        "Xi":     ["Psi", "Delta", "Pi",  "Phi",   "Sigma", "Psi", "Phi",    "Gamma"],
        "Pi":     ["Psi", "Delta", "Xi",  "Phi",   "Sigma", "Psi", "Phi",    "Gamma"],
        "Sigma":  ["Xi",  "Delta", "Xi",  "Phi",   "Sigma", "Pi",  "Lambda", "Gamma"],
        "Phi":    ["Psi", "Delta", "Psi", "Phi",   "Sigma", "Pi",  "Phi",    "Gamma"],
        "Psi":    ["Psi", "Delta", "Psi", "Phi",   "Sigma", "Psi", "Phi",    "Gamma"],
    }
    super_quad = [
        transPt(transformations[6], quad[2]),
        transPt(transformations[5], quad[1]),
        transPt(transformations[3], quad[2]),
        transPt(transformations[0], quad[1]),
    ]
    out = {}
    for label, subs in super_rules.items():
        children = [{'geom': sys[s], 'T': t} for s, t in zip(subs, transformations) if s]
        out[label] = MetaTile(children, super_quad)
    return out


_COLS = {
    "Gamma1": [0.60, 0.40, 0.85], "Gamma2": [0.72, 0.55, 0.92],
    "Delta": [0.90, 0.86, 0.96], "Theta": [1.0, 0.85, 0.6], "Lambda": [0.7, 0.9, 0.7],
    "Xi": [0.6, 0.85, 0.95], "Pi": [0.95, 0.75, 0.75], "Sigma": [0.85, 0.85, 0.6],
    "Phi": [0.75, 0.85, 0.95], "Psi": [0.9, 0.8, 0.9],
}
def _col(label):
    return _COLS.get(label, [0.85, 0.85, 0.9])


def build_spectre_patch(levels):
    """Iterate the substitution `levels` times; return one supertile to draw/analyse."""
    tiles = buildSpectreBase()
    for _ in range(levels):
        tiles = buildSupertiles(tiles)
    return tiles["Delta"]
