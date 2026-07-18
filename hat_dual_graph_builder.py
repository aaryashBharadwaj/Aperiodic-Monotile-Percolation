import numpy as np
from scipy.spatial import KDTree
from hat_generator import mul, transPt
from collections import defaultdict


# this builds the dual for the hat-tile graph
def build_tile_graph(patch, level=0, tol=1e-5):
    tile_polygons = []

    # Collect all tile polygons as arrays of vertex coordinates
    def _collect_tiles(patch, S, level):
        # if this isn't the bottom layer, collect the children
        if level > 0 and hasattr(patch, "children"):
            for g in patch.children:
                _collect_tiles(g['geom'], mul(S, g['T']), level - 1)
        # if this is the bottom layer, add each of the points
        else:
            poly = np.array([[transPt(S, p)['x'], transPt(S, p)['y']] for p in patch.shape])
            tile_polygons.append(poly)

    _collect_tiles(patch, [1, 0, 0, 0, 1, 0], level)

    N = len(tile_polygons)
    print(f"Total tiles collected: {N}")

    # The dual makes each shape a node, thus we make the centre of each one a node
    tile_centroids = np.array([poly.mean(axis=0) for poly in tile_polygons])

    # Build a flat array of all vertices with tile index labels
    # so we can quickly find which tiles share vertices
    all_verts = []
    tile_ids = []
    for i, poly in enumerate(tile_polygons):
        for v in poly:
            all_verts.append(v)
            tile_ids.append(i)

    all_verts = np.array(all_verts)
    tile_ids = np.array(tile_ids)

    # Use KDTree to find all pairs of vertices within tolerance
    tree = KDTree(all_verts)
    pairs = tree.query_pairs(r=tol)

    # For each pair of vertices that are coincident, record which tiles they belong to
    # Count shared vertices between each pair of tiles
    shared_vertex_count = defaultdict(int)

    for i, j in pairs:
        ti, tj = tile_ids[i], tile_ids[j]
        if ti != tj:
            key = (min(ti, tj), max(ti, tj))
            shared_vertex_count[key] += 1

    # Two tiles are neighbours if they share at least 2 vertices (a full edge)
    tile_neighbors = [[] for _ in range(N)]
    for (ti, tj), count in shared_vertex_count.items():
        if count >= 2:
            tile_neighbors[ti].append(tj)
            tile_neighbors[tj].append(ti)

    tile_neighbors = [np.array(n, dtype=np.int32) for n in tile_neighbors]

    return tile_centroids, tile_neighbors, tile_polygons

##################################### SQUARE FRAME HAT TILING BUILD ##################################

# Create the square-frame we analyse the percolation on
def analyze_tile_square_frame(tile_centroids, tile_neighbors, L, boundary_thickness=1.0,
                              center_x=200.0, center_y=-100.0):
    # Same center convention as graph_builder for consistency.
    # For patch r=6 use center (515.0, -273.0). Defaults preserve original r=5 behaviour.

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