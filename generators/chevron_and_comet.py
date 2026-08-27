import math
import itertools

import numpy as np
from matplotlib.path import Path
import generators.hat_generator as hg
from builders.direct_graph_builder import graph_from_polygons

SQ3 = math.sqrt(3)
E1 = np.array([1.0, 0.0])
E2 = np.array([0.5, SQ3 / 2])

# Geometry for the periodic Tile(a,b) family endpoints: Comet (Tile(1,0)) and Chevron (Tile(0,1))

def tile_ab(a, b):
    pts = [(p['x'], p['y']) for p in hg.hat_outline]; n = len(pts); out = [(0.0, 0.0)]
    # goes through each edge of the hat and adjusts it's length according to a and b
    for i in range(n - 1):
        # measure the length of the edge, L is length by Pythagoras 
        x0, y0 = pts[i]; x1, y1 = pts[i + 1]; vx, vy = x1 - x0, y1 - y0; L = math.hypot(vx, vy)
        # since b-edges have length sqrt(3) and a-edges have length 1, we check which it is
        s = a if abs(L - round(L)) < 1e-6 else (b / SQ3 if abs(L / SQ3 - round(L / SQ3)) < 1e-6 else 1.0)
        # build the new vertex by adding a scaling vector, thus preserving direction
        out.append((out[-1][0] + s * vx, out[-1][1] + s * vy))
    # turns it into a (n,2) array
    poly = np.array(out)
    v = poly[1] - poly[0]; ang = math.degrees(math.atan2(v[1], v[0])) % 60.0
    if abs(ang - 30.0) < 1e-3:
        # the code confusingly uses the variable s twice! Sorry! 
        c, s = math.cos(math.radians(-30)), math.sin(math.radians(-30))
        poly = poly @ np.array([[c, -s], [s, c]]).T
    return poly

# It works out which triangular-grid cells the tile covers, and returns them as a set of cell IDs
# This is because cells makes it easier to see if your tiling is covering without overlap or gaps
# This requires tiles built on triangles
def tri_set(poly):
    # Return the set of triangular-grid cells (i, j, s) whose CENTROID lies inside the tile polygon.
    # Containment is matplotlib.path.Path.contains_points -- the same point-in-polygon the rest of the
    # repo uses -- vectorised over every candidate centroid. This is exact for this problem: each
    # centroid is the 1/3-point of a grid triangle and tile edges run along grid lines, so a centroid
    # never sits on a tile edge and the classification matches the old hand-rolled ray cast cell-for-cell.
    xs = poly[:, 0]; ys = poly[:, 1]
    path = Path(np.asarray(poly, float))
    def cent(i, j, s):
        # each cell v[0] * E1 + v[1] * E2 forms a parallelogram, split into two triangles: s=0 the
        # bottom-left, s=1 the top-right. The centroid is the mean of the triangle's three grid vertices.
        vs = [(i, j), (i + 1, j), (i, j + 1)] if s == 0 else [(i + 1, j), (i + 1, j + 1), (i, j + 1)]
        return sum(v[0] * E1 + v[1] * E2 for v in vs) / 3.0
    # Candidate cells over the tile's bounding box, padded by 2 (a tile reaching x=-3.7 needs column -4,
    # which a tight range would miss). One containment call over all their centroids.
    cells, cents = [], []
    for i in range(int(xs.min()) - 2, int(xs.max()) + 2):
        for j in range(int(ys.min() / (SQ3 / 2)) - 2, int(ys.max() / (SQ3 / 2)) + 2):
            for s in (0, 1):
                cells.append((i, j, s)); cents.append(cent(i, j, s))
    if not cents:
        return set()
    inside = path.contains_points(np.asarray(cents, float))
    return {cells[k] for k in range(len(cells)) if inside[k]}

# You have one comet tile. Where do you put copies of it so they cover the plane perfectly?
# Try every possible pair of vectors. There are 6561 pairs (81 choices for u1, 81 for u2). For each pair, ask "does this work?" Return the first that does.
# Note ONLY PERIODIC! And that too only translation, no rotation at all
# An optimised version can simply allign 2 edges that have the same angle and length but it's not worth it
def find_lattice(ts):
    # u1 and u2 are how much to the left and up should the next tile be
    # imagine the first tile starting at (0,0)
    # the determinant tells you the area that has to be spanned by the first tile without the second
    # if this area is less than or greater than the area of the tile, we can instantly reject
    # if not we use verify
    need = len(ts) // 2
    # if a cell is coverered twice it has an overlap if none it has a gap
    # we check a tally for how many times a cell is covered and ensure it's 1
    def verify(u1, u2):
        cover = {}
        # but how do we tell if a cell is empty because it's a gap or just not covered (stamped)?
        # We only check interior cells which prevents those not stamped from looking like gaps 
        # This is safe because any gap on the far right will also be in the middle where cell 0 connects to cell 1
        # A tile sprawling across many lattice cells could produce a false gap at the check boundary so keep in mind
        for m in range(-3, 4):
            for n in range(-3, 4):
                si, sj = m * u1[0] + n * u2[0], m * u1[1] + n * u2[1]
                for (i, j, s) in ts: cover[(i + si, j + sj, s)] = cover.get((i + si, j + sj, s), 0) + 1
        return all(cover.get((i, j, s), 0) == 1 for i in range(-2, 3) for j in range(-2, 3) for s in (0, 1))
    for u1 in itertools.product(range(-4, 5), repeat=2):
        for u2 in itertools.product(range(-4, 5), repeat=2):
            if abs(u1[0] * u2[1] - u1[1] * u2[0]) == need and verify(u1, u2):
                return u1, u2
    return None

def periodic_block(tile, ncells):
    # runs all the parts we looked at and returns integer pairs to find how to tile these
    u1, u2 = find_lattice(tri_set(tile))
    # convert these to world coordinates from lattice coordinates
    U1 = u1[0] * E1 + u1[1] * E2; U2 = u2[0] * E1 + u2[1] * E2
    # creates the full polygons for tiling by just repeating the offset
    polys = [tile + m * U1 + n * U2 for m in range(ncells) for n in range(ncells)]
    # returns the polygons and a rough size for the percolation window
    return polys, float(np.hypot(*U1)) * ncells

# Just a name-to-parameters lookup. Comet is Tile(1,0), chevron is Tile(0,1). Build that tile, hand it to periodic_block.
def periodic_polys(name, ncells=150):
    a, b = (1.0, 0.0) if name == "comet" else (0.0, 1.0)
    return periodic_block(tile_ab(a, b), ncells)


def periodic_graph(name, ncells=150):
    # Same direct (vertex) graph as the hat/spectre: dedup coincident vertices, perimeter edges, adjacency through the shared builder. See direct_graph_builder.graph_from_polygons
    polys, span = periodic_polys(name, ncells)
    nodes, neigh = graph_from_polygons(polys)
    return nodes, neigh, span
