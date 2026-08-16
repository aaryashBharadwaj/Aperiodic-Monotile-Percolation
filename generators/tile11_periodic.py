"""Periodic Tile(1,1) -- the weakly-chiral periodic tiling of the equilateral family member (a=b=1),
the periodic partner of the aperiodic Spectre. Kept in its OWN file (not chevron_and_comet) because it
is built a completely different way: the comet/chevron endpoints tile by pure translation on the
unit-triangle grid, which tri_set/find_lattice discovers automatically. Tile(1,1) can't use that -- its
area is irrational in triangle units (it lives on the kisrhombille grid) and it only tiles once
REFLECTIONS are allowed, neither of which the translation-only lattice search can represent.

Instead we place tiles by explicit edge-matching with hat_generator.matchTwo (continuous coordinates, so
no grid; a mirrored neighbour is just a reflection in the transform, so weak chirality is free). The
whole tiling follows from ONE local rule: tile A's six neighbours. Two are non-flipped translates (B, C
-- the vertical column) and four are mirrored (D, E, F, G). Each is specified by the (A-vertex,
neighbour-vertex) pairs that must coincide; two pairs pin the rigid placement, the rest are consistency.
G is NOT given by its own A-pairs (those are ill-posed here); it is placed by the tiling's D<->E, G<->F
point symmetry: G sits on D exactly as F sits on E, i.e. N_G = N_D . N_E^-1 . N_F. Composing these six
transforms on every tile grows the plane; verified gap-free and overlap-free.

Public API mirrors chevron_and_comet: tile11_polys(reach) -> (polygons, span); tile11_graph(reach) ->
(nodes, neighbours, span) via the shared graph_from_polygons (which dedups the shared vertices)."""
import numpy as np

from generators.chevron_and_comet import tile_ab
from generators.hat_generator import matchTwo, mul, inv, pt
from builders.direct_graph_builder import graph_from_polygons

# The base tile A = Tile(1,1), centred at the origin (cached; it never changes).
_A = None
def _base():
    global _A
    if _A is None:
        a = tile_ab(1.0, 1.0)
        _A = a - a.mean(0)
    return _A

# Mirror across the vertical axis -- the ONLY reflection used (the "flip"); no rotations anywhere.
_REFLECT = [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def _P(v):
    return pt(float(v[0]), float(v[1]))


def _ntrans(pairs, flipped):
    """Transform placing a neighbour (flipped => a mirror copy) so the given (A-vertex, neighbour-vertex)
    pairs coincide. Two pairs fully pin a rigid unit-edge placement; matchTwo lays neighbour edge
    (n0->n1) onto A edge (a0->a1). For a flipped neighbour we match against the mirror M and fold the
    reflection into the transform, so the result acts on the base tile A uniformly."""
    A = _base()
    M = A * np.array([-1.0, 1.0])
    (a0, n0), (a1, n1) = pairs[0], pairs[1]
    base = M if flipped else A
    mt = matchTwo(_P(base[n0]), _P(base[n1]), _P(A[a0]), _P(A[a1]))
    return mul(mt, _REFLECT) if flipped else mt


def _neighbour_transforms():
    """A's six neighbour transforms, in A's frame. B,C non-flipped (the vertical column); D,E,F flipped
    from their pairs; G flipped via the D<->E, G<->F symmetry (N_G = N_D . N_E^-1 . N_F)."""
    NB = _ntrans([(3, 13), (4, 12)], False)
    NC = _ntrans([(13, 3), (12, 4)], False)
    ND = _ntrans([(2, 6), (3, 7)], True)
    NE = _ntrans([(6, 2), (7, 3)], True)
    NF = _ntrans([(10, 2), (9, 1)], True)
    NG = mul(ND, mul(inv(NE), NF))          # G is to D as F is to E
    return [NB, NC, ND, NE, NF, NG]


def _apply(T, poly):
    return np.array([[T[0] * x + T[1] * y + T[2], T[3] * x + T[4] * y + T[5]] for x, y in poly])


def tile11_polys(reach):
    """Grow the periodic Tile(1,1) patch out to physical radius `reach` (tile-centroid distance) by
    composing the six neighbour transforms outward from one tile. Returns (list of tile polygons, span).
    Tiles are deduped by centroid so each is placed once regardless of how many neighbours reach it."""
    A = _base()
    NS = _neighbour_transforms()
    ident = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    def key(p):
        c = p.mean(0)
        return (round(c[0], 2), round(c[1], 2))

    placed = {key(A): ident}
    stack = [ident]
    r2 = float(reach) * float(reach)
    while stack:
        T = stack.pop()
        for N in NS:
            T2 = mul(T, N)
            p = _apply(T2, A)
            c = p.mean(0)
            if c[0] * c[0] + c[1] * c[1] > r2:      # outside the target disk -> don't expand through it
                continue
            k = key(p)
            if k in placed:
                continue
            placed[k] = T2
            stack.append(T2)
    polys = [_apply(T, A) for T in placed.values()]
    return polys, float(2.0 * reach)


def tile11_graph(reach):
    """Direct (vertex) graph of the periodic Tile(1,1) patch -- same builder as every other tiling, so
    coincident tile vertices are deduped into shared nodes and tile edges become bonds."""
    polys, span = tile11_polys(reach)
    nodes, neigh = graph_from_polygons(polys)
    return nodes, neigh, span
