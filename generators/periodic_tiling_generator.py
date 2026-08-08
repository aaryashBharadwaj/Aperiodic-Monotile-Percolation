import math

import numpy as np

S3 = math.sqrt(3)


# Polygon generators: an n x n block of unit tiles, returned as polygons so the caller can feed them
# through the SHARED graph builders (direct_graph_builder.graph_from_polygons /
# dual_graph_builder.build_dual_from_polygons) -- exactly the path the aperiodic tilings take. The
# GUI/percolate use these so square/triangular percolation exercises the same builders as the
# hat/spectre, rather than any bespoke lattice code.
def square_tiles(n):
    """n x n block of unit squares. Perimeter graph = the square lattice (site pc 0.5927); the
    tile-adjacency dual is again a square lattice (self-dual)."""
    return [np.array([[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]], dtype=float)
            for i in range(n) for j in range(n)]


def triangular_tris(n):
    """n x n block of unit triangles (up- and down-pointing). Its tile-adjacency dual is the
    honeycomb -- the exact-value validation for the dual builder."""
    E1 = np.array([1.0, 0.0]); E2 = np.array([0.5, S3 / 2])
    tris = []
    for i in range(n):
        for j in range(n):
            a = i * E1 + j * E2; b = (i + 1) * E1 + j * E2
            c = i * E1 + (j + 1) * E2; d = (i + 1) * E1 + (j + 1) * E2
            tris.append(np.array([a, b, c]))
            tris.append(np.array([b, d, c]))
    return tris