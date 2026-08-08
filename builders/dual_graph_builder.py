import numpy as np
from scipy.spatial import KDTree
from generators.hat_generator import mul, transPt
from builders.graph_core import edges_to_adjacency


# Collect every leaf tile of a metatile PATCH as an array of its (world-space) vertex coordinates.
# Works for any patch with the .shape / .children structure (the hat AND the spectre).
def collect_leaf_polygons(patch, level=0):
    tile_polygons = []
    def _collect_tiles(patch, S, level):
        # If it's not a leaf descend into children, composing the accumulated transform S with
        # the child's relative transform g['T']. getattr(...) handles both the hat (leaves have no
        # children attribute) and the spectre (leaves set children=None) uniformly.
        ch = getattr(patch, "children", None)
        if ch and level > 0:
            for g in ch:
                _collect_tiles(g['geom'], mul(S, g['T']), level - 1)
        # If it is a leaf apply the net transform to every vertex and keep the polygon.
        else:
            poly = np.array([[transPt(S, p)['x'], transPt(S, p)['y']] for p in patch.shape])
            tile_polygons.append(poly)
    _collect_tiles(patch, [1, 0, 0, 0, 1, 0], level)
    return tile_polygons


# Builds the DUAL (the Delone / tile-adjacency graph, NOT the planar dual) from a list of tile
# polygons. Tiling-agnostic: hat, spectre, or the periodic family all just supply their tile
# polygons. It is a lot simpler than the vertex graph since our nodes ARE our tiles.
def build_dual_from_polygons(tile_polygons, tol=1e-5):
    N = len(tile_polygons)

    # One node per tile, positioned at the tile's centroid.
    tile_centroids = np.array([poly.mean(axis=0) for poly in tile_polygons])

    # Flatten every tile's vertices into one array, remembering which tile each vertex belongs to.
    # tile_ids[k] is the owner of vertex k (so tile_ids[20]=1 means flat-vertex 20 belongs to tile 1).
    sizes = np.array([len(p) for p in tile_polygons])
    all_verts = np.concatenate(tile_polygons, axis=0)
    tile_ids = np.repeat(np.arange(N), sizes)

    # Find every pair of vertices sitting at the same point (coincident within tol).
    # output_type='ndarray' returns a (P, 2) array rather than a Python set of tuples, so the
    # rest can be done with array operations.
    tree = KDTree(all_verts)
    pairs = tree.query_pairs(r=tol, output_type='ndarray')
    del tree

    if len(pairs) == 0:
        return tile_centroids, [np.array([], dtype=np.int32) for _ in range(N)], tile_polygons

    # If an edge exists between nodes between two tiles, an edge exists between those tiles
    # So we turn the vertex pairs into the pairs for the tiles it links
    ti = tile_ids[pairs[:, 0]]
    tj = tile_ids[pairs[:, 1]]
    m = ti != tj
    lo = np.minimum(ti[m], tj[m])
    hi = np.maximum(ti[m], tj[m])

    # Count how many coincident vertices each tile pair shares
    # A delone graph is defined by sharing an edge
    # Since sharing only one vertex doesn't share an edge, a tile pair is a neighbour pair exactly when its count is >= 2.
    # Packing (lo, hi) into one integer lets a single np.unique both group and count.
    key = lo.astype(np.int64) * N + hi
    ukey, counts = np.unique(key, return_counts=True)
    edge = ukey[counts >= 2]
    elo = edge // N
    ehi = edge % N

    # Assemble neighbour lists with the shared helper (both directions, sort + bincount).
    tile_neighbors = edges_to_adjacency(elo, ehi, N)
    return tile_centroids, tile_neighbors, tile_polygons


# Dual analogue of analyze_square_frame: cut an L x L window at (center_x, center_y) over the tile CENTROIDS
def analyze_tile_square_frame(tile_centroids, tile_neighbors, L, boundary_thickness=1.0,
                              center_x=515.0, center_y=-273.0):
    # Same centre convention as direct_graph_builder: the centre of the largest square
    # inscribable in the r=6 patch.
    x_min, x_max = center_x - L / 2.0, center_x + L / 2.0
    y_min, y_max = center_y - L / 2.0, center_y + L / 2.0

    coords = tile_centroids
    inside_mask = (
        (coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) &
        (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max)
    )
    inside_indices = np.where(inside_mask)[0]

    if len(inside_indices) < 2:
        return {'node_count': 0}

    inside_coords = coords[inside_indices]
    original_to_new = {orig: new for new, orig in enumerate(inside_indices)}

    N_sub = len(inside_indices)
    sub_neighbors = [[] for _ in range(N_sub)]

    for new_i, orig_i in enumerate(inside_indices):
        for orig_j in tile_neighbors[orig_i]:
            new_j = original_to_new.get(orig_j)
            if new_j is not None:
                sub_neighbors[new_i].append(new_j)

    sub_neighbors = [np.array(n, dtype=np.int32) for n in sub_neighbors]

    # Edge list from adjacency (each unordered pair once) for bond percolation
    sub_edges = []
    for i in range(N_sub):
        for j in sub_neighbors[i]:
            if i < j:
                sub_edges.append((i, j))
    sub_edges = np.array(sub_edges, dtype=np.int32)

    # Boundary detection by centroid proximity to frame edges
    top_mask    = inside_coords[:, 1] >= y_max - boundary_thickness
    bottom_mask = inside_coords[:, 1] <= y_min + boundary_thickness
    left_mask   = inside_coords[:, 0] <= x_min + boundary_thickness
    right_mask  = inside_coords[:, 0] >= x_max - boundary_thickness

    new_top    = np.array([original_to_new[inside_indices[i]] for i in np.where(top_mask)[0]],    dtype=np.int32)
    new_bottom = np.array([original_to_new[inside_indices[i]] for i in np.where(bottom_mask)[0]], dtype=np.int32)
    new_left   = np.array([original_to_new[inside_indices[i]] for i in np.where(left_mask)[0]],   dtype=np.int32)
    new_right  = np.array([original_to_new[inside_indices[i]] for i in np.where(right_mask)[0]],  dtype=np.int32)

    return {
        'L_value': L,
        'sub_graph_nodes': inside_coords,
        'sub_graph_neighbors': sub_neighbors,
        'sub_graph_edges': sub_edges,
        'node_count': N_sub,
        'top_boundary_nodes':    new_top,
        'bottom_boundary_nodes': new_bottom,
        'left_boundary_nodes':   new_left,
        'right_boundary_nodes':  new_right,
        'center_x': center_x,
        'center_y': center_y
    }