import math
import itertools

import numpy as np
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
    # creates an empty set out for the covered cells (set so duplicates don't build up)
    # xs and ys are the x and y component for every polygon
    P = [tuple(p) for p in poly]; xs = poly[:, 0]; ys = poly[:, 1]; out = set()
    def cent(i, j, s):
        # each cell v[0] * E1 + v[1] * E2 forming a parallelogram
        # this splits each parallelogram into two triangles, s=0 which is the bottom left and s=1 the top right
        # thus each cell is described by i, j, s
        vs = [(i, j), (i + 1, j), (i, j + 1)] if s == 0 else [(i + 1, j), (i + 1, j + 1), (i, j + 1)]
        # averaging the 3 centres gives you the centroid 
        c = sum(v[0] * E1 + v[1] * E2 for v in vs) / 3.0; return c
    # Standard ray casting: fire a ray to the right from the point and count how many edges it crosses. Odd means inside, even means outside
    # Test one ray per triangle centroid 
    # (px, py) and (qx, qy) are vertices that together form the current edge
    def pip(x, y):
        n = len(P); ins = False; px, py = P[-1]
        for qx, qy in P:
            # (qy > y) != (py > y) checks if an edge is above or below the ray, if not it goes through
            if ((qy > y) != (py > y)) and (x < (px - qx) * (y - qy) / (py - qy + 1e-30) + qx): ins = not ins
            px, py = qx, qy
        # if ins is false, it's outside otherwise it's inside
        return ins
    # this finds for a tile, the x values to test (error bounds +- 2)
    # A tile reaching to x = -3.7 needs column -4 checked, and the tight range would start at -3 and miss it.
    for i in range(int(xs.min()) - 2, int(xs.max()) + 2):
        # this checks the y values to test ( error bounds +- 2)
        for j in range(int(ys.min() / (SQ3 / 2)) - 2, int(ys.max() / (SQ3 / 2)) + 2):
            # checks if the chosen cell is inside the tile or not
            # This gives us the tile as a list of grid cells
            for s in (0, 1):
                c = cent(i, j, s)
                if pip(c[0], c[1]): out.add((i, j, s))
    return out

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
