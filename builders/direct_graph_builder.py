import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from generators.hat_generator import mul, transPt
from builders.graph_core import edges_to_adjacency


def dedup_vertices(raw, tol=1e-5):
    """Merge vertices within `tol` of each other into unique nodes.

    raw : (V, 2) float array of vertex coordinates (with duplicates where tiles meet).
    Returns (unique_nodes (n_unique, 2), labels (V,) int64) where labels[k] is the unique-node
    id of raw row k. Coincident raw rows share a label; the last-written coordinate wins (they
    agree to within tol, so it doesn't matter which).

    Shared by the DIRECT (vertex) builders -- the hat/spectre patch build below, the periodic family
    blocks, and the penrose builder -- since "nodes are merged coincident corners" is the defining
    idea of the vertex graph. The dual graph does NOT use this: it keeps tiles as nodes and only
    COUNTS coincident corners per tile-pair.
    """
    V = len(raw)
    # query_pairs returns every pair of raw vertices within tol of each other -- the coincidences
    # where tiles meet. output_type='ndarray' keeps it vectorised (not a Python set of tuples).
    tree = KDTree(raw)
    pairs = tree.query_pairs(r=tol, output_type='ndarray')
    del tree
    if len(pairs):
        # Treat the coincident pairs as a sparse adjacency matrix (scipy's representation); its
        # connected components ARE the merged nodes. directed=False because coincidence is
        # symmetric (the edge is bidirectional).
        g = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
                       shape=(V, V))
        n_unique, labels = connected_components(g, directed=False)
        del g
    else:
        # No two vertices coincide -> every raw vertex is already its own node.
        n_unique, labels = V, np.arange(V)
    labels = labels.astype(np.int64)
    unique_nodes = np.empty((n_unique, 2), dtype=np.float64)
    # unique_nodes[id] = that node's coordinates. The last raw row written to a slot wins; all rows
    # sharing a label are coincident within tol, so which one wins doesn't matter.
    unique_nodes[labels] = raw
    return unique_nodes, labels


def graph_from_raw(raw, poly_sizes, tol=1e-5):
    """Direct (vertex) graph for an edge-to-edge polygon tiling, from concatenated outline vertices.

    raw : (V, 2) float array — every tile's outline vertices, concatenated in polygon order.
    poly_sizes : (P,) int array — poly_sizes[k] is the vertex count of polygon k, so the polygons
                 partition raw into contiguous blocks.
    Builds the graph shared by every edge-to-edge polygon tiling (hat, spectre, comet, chevron, ...):
    merge coincident vertices into nodes, connect each tile's consecutive outline vertices
    (perimeter edges, wrapping last->first), and assemble the adjacency. Tilings differ only in how
    they PRODUCE (raw, poly_sizes); this assembly is common. Returns
    (unique_nodes (n, 2), neighbors (list of n int32 arrays)).

    NOT for tilings whose graph edges aren't the tile perimeter — penrose (the rhombus rule) and the
    dual (tile adjacency) build their own (lo, hi) and call graph_core.edges_to_adjacency directly.
    """
    raw = np.asarray(raw)
    poly_sizes = np.asarray(poly_sizes, dtype=np.int64)
    V = len(raw)

    # Merge coincident vertices (the shared points where tiles meet) into unique node ids.
    # labels[k] = node id of raw vertex k. See dedup_vertices for the mechanics.
    unique_nodes, labels = dedup_vertices(raw, tol)
    n_unique = len(unique_nodes)

    # Perimeter edges, from the polygon sizes alone (no coordinates). offsets[k] is where polygon k
    # starts in raw; the chain src=k -> dst=k+1 gives 1-2, 2-3, ..., then each polygon's LAST vertex
    # is redirected back to its FIRST, closing the outline.
    offsets = np.empty(len(poly_sizes) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(poly_sizes, out=offsets[1:])
    src = np.arange(V, dtype=np.int64)
    dst = src + 1
    dst[offsets[1:] - 1] = offsets[:-1]
    del raw, poly_sizes

    # Map the raw-vertex chain onto node ids, drop self-loops (endpoints that merged into one node),
    # and deduplicate: every shared edge is emitted by both adjoining tiles. Pack each undirected
    # edge into one integer (lo*n_unique + hi) so np.unique collapses the duplicates.
    u = labels[src]; v = labels[dst]
    del src, dst, labels
    m = u != v
    lo = np.minimum(u[m], v[m]); hi = np.maximum(u[m], v[m])
    del u, v, m
    key = np.unique(lo * np.int64(n_unique) + hi)
    lo = key // n_unique; hi = key % n_unique
    del key

    neighbors = edges_to_adjacency(lo, hi, n_unique)
    return unique_nodes, neighbors


def graph_from_polygons(polys, tol=1e-5):
    """Direct (vertex) graph from a list of tile-outline polygons.

    polys : iterable of (n_i, 2) arrays, each one tile's outline vertices in order.
    Convenience wrapper over graph_from_raw for callers that already hold the polygons as a list
    (periodic blocks; future one-off tilings): concatenates them into (raw, poly_sizes) and builds
    the graph. Returns (unique_nodes, neighbors).
    """
    polys = [np.asarray(p) for p in polys]
    poly_sizes = np.fromiter((len(p) for p in polys), dtype=np.int64, count=len(polys))
    raw = np.concatenate(polys)
    return graph_from_raw(raw, poly_sizes, tol)

# DIRECT (vertex) graph builder for the patch-based aperiodic tilings -- the hat AND the spectre:
# nodes are tile corners, edges are tile perimeters. This function only COLLECTS every leaf tile's
# raw polygon vertices (in collection order) plus each tile's vertex count; the assembly --
# merge coincident vertices into unique node ids, then build the adjacency list -- is graph_from_raw
# above (the same routine the periodic family blocks reach via graph_from_polygons).
# tol is a choice for which nodes are close enough to be the same node.
def build_neighbor_graph_fast(patch, level=0, tol=1e-5):
    # Exact-size pre-allocation: count the leaf-tile vertices in one light pass (tree walk only, no
    # transforms), then allocate exactly that and fill. This replaces a fixed 20M cap that SILENTLY
    # dropped vertices past the limit and then crashed on the raw/poly_sizes mismatch -- large aperiodic
    # patches (spectre level 7 has ~30M raw vertices) now build correctly, and small patches no longer
    # over-allocate a 320MB buffer.
    def _count(patch, level):
        ch = getattr(patch, "children", None)
        if ch and (level is None or level > 0):
            nxt = None if level is None else level - 1
            return sum(_count(g['geom'], nxt) for g in ch)
        return len(patch.shape)

    raw = np.empty((_count(patch, level), 2), dtype=np.float64)
    poly_sizes = []               # vertices per leaf polygon, in collection order
    # cnt is the cursor to write into raw
    # it is a list for python-specific reason but functions as a counter
    cnt = [0]

    # Takes a MetaTile 'patch' as an input
    def _collect(patch, S, level):
        ch = getattr(patch, "children", None)
        if ch and (level is None or level > 0):
            nxt = None if level is None else level - 1
            # Each childs transform is defined relative to it's parent
            # So the global transform of the child is the product of the parent's accumulated transform with the childs relative transform
            # The result is stored in S and g['T'] is the relative position. nxt is the depth of the child
            for g in ch:
                _collect(g['geom'], mul(S, g['T']), nxt)
        else:
            # Once you reach a leaf, you apply the net transformation and append the coordinates
            shp = patch.shape
            for p in shp:
                q = transPt(S, p)
                raw[cnt[0], 0] = q['x']; raw[cnt[0], 1] = q['y']; cnt[0] += 1
            poly_sizes.append(len(shp))

    _collect(patch, [1, 0, 0, 0, 1, 0], level)
    # Since V is the counter at the end, it's the number of coordinates
    V = cnt[0]
    # Trim the size of raw for efficiency after this is done
    raw = raw[:V]
    # poly_sizes is the index of  how many vertices each polygon has
    # [4, 3, 5] means poly0 has 4 vertices, poly1 has 3, poly2 has 5
    poly_sizes = np.asarray(poly_sizes, dtype=np.int64)

    # This builder only COLLECTS the leaf-tile vertices (above); the graph itself -- merge coincident
    # vertices, connect each tile's perimeter, assemble adjacency -- is graph_from_raw (above), the
    # assembly shared with every edge-to-edge polygon tiling (spectre, comet, chevron, ...).
    return graph_from_raw(raw, poly_sizes, tol)

#Convert adjacency list to Compressed Sparse Row (CSR) format for storage.
def neighbors_to_csr(neighbors):
    neighbor_starts = np.zeros(len(neighbors) + 1, dtype=np.int32)
    total = 0
    for i, nbrs in enumerate(neighbors):
        neighbor_starts[i] = total
        total += len(nbrs)
    neighbor_starts[-1] = total
    
    neighbors_arr = np.zeros(total, dtype=np.int32)
    idx = 0
    for nbrs in neighbors:
        neighbors_arr[idx:idx+len(nbrs)] = nbrs
        idx += len(nbrs)
    
    return neighbors_arr, neighbor_starts


# Give it a list of which nodes to keep and it keeps those 
# Essentially finds edges where both input and output are in the provided list
def create_subgraph(master_nodes, master_neighbors, inside_original_indices):
    # master nodes are nodes of the original, sub_nodes are of the smaller region
    sub_nodes = master_nodes[inside_original_indices]
    num_sub_nodes = len(sub_nodes)
    original_to_new_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(inside_original_indices)}
    sub_neighbors = [[] for _ in range(num_sub_nodes)]
    sub_edges = set()

    # The inner loop iterates edges per node
    # The outer loop iterates over nodes
    for new_idx, orig_idx in enumerate(inside_original_indices):
        for nbr_orig_idx in master_neighbors[orig_idx]:
            # The elements in the dictonary are renumbered so that you still have 0,1,2... (N-1)
            new_nbr_idx = original_to_new_map.get(nbr_orig_idx)
            # add to the new graph only if the nodes are contained within the dictionary
            if new_nbr_idx is not None:
                sub_neighbors[new_idx].append(new_nbr_idx)
                sub_edges.add((min(new_idx, new_nbr_idx), max(new_idx, new_nbr_idx)))
                
    sub_neighbors = [np.array(n, dtype=np.int32) for n in sub_neighbors]
    sub_edges_list = np.array(list(sub_edges), dtype=np.int32)
    
    return sub_nodes, sub_neighbors, original_to_new_map, sub_edges_list

# Find the centre of the largest (axis-aligned) square that fits inside the node cloud, so a square
# frame can be placed on ANY patch without a hardcoded, size-specific centre. Bins nodes into an
# H-sized grid, keeps cells whose occupancy is >= half the median (the "filled" interior, excluding
# the sparse fringe), and runs the classic maximal-all-ones-square DP on that mask. Returns
# (cx, cy, side); H is the grid resolution in the tiling's coordinate units.
def largest_square_center(nodes, H=4.0):
    xs, ys = nodes[:, 0], nodes[:, 1]
    xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()
    nx = int(np.ceil((xmax - xmin) / H)) + 1
    ny = int(np.ceil((ymax - ymin) / H)) + 1
    ix = ((xs - xmin) / H).astype(np.int64)
    iy = ((ys - ymin) / H).astype(np.int64)
    cnt = np.zeros((ny, nx), np.int64); np.add.at(cnt, (iy, ix), 1)
    thr = max(1.0, 0.5 * np.median(cnt[cnt > 0]))    # "filled" = at least half the median occupancy
    F = cnt >= thr
    dp = np.zeros_like(cnt, np.int32); best = bi = bj = 0
    for i in range(ny):
        for j in range(nx):
            if F[i, j]:
                dp[i, j] = 1 if (i == 0 or j == 0) else 1 + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
                if dp[i, j] > best:
                    best, bi, bj = dp[i, j], i, j
    side = best * H
    cx = xmin + (bj - best / 2 + 0.5) * H
    cy = ymin + (bi - best / 2 + 0.5) * H
    return cx, cy, side


# Creates the square frame that create_subgraph uses to create our region. center_x/center_y default
# to None -> auto-placed at the centre of the largest inscribable square (so the SAME code frames any
# patch/size); pass explicit values to pin a centre (the hat r=6 run passes (515,-273), the
# precomputed largest-square centre for that patch).
def analyze_square_frame(master_nodes, master_neighbors, L, boundary_thickness=1.0,
                         center_x=None, center_y=None):
    if center_x is None or center_y is None:
        center_x, center_y, _ = largest_square_center(master_nodes)

    x_min, x_max = center_x - L / 2.0, center_x + L / 2.0
    y_min, y_max = center_y - L / 2.0, center_y + L / 2.0
    
    nodes = master_nodes
    # Find all nodes inside the square region
    inside_mask = (nodes[:, 0] >= x_min) & (nodes[:, 0] <= x_max) & \
                  (nodes[:, 1] >= y_min) & (nodes[:, 1] <= y_max)
    inside_original_indices = np.where(inside_mask)[0]
    inside_nodes_coords = nodes[inside_original_indices]
    # Early return if region is too small
    if len(inside_original_indices) < 2:
        return {'node_count': 0}

    sub_nodes, sub_neighbors, original_to_new_map, sub_edges = create_subgraph(
        master_nodes, master_neighbors, inside_original_indices
    )
    # Identify boundary nodes (nodes within boundary_thickness of edges)
    top_mask = (inside_nodes_coords[:, 1] >= y_max - boundary_thickness)
    bottom_mask = (inside_nodes_coords[:, 1] <= y_min + boundary_thickness)
    left_mask = (inside_nodes_coords[:, 0] <= x_min + boundary_thickness)
    right_mask = (inside_nodes_coords[:, 0] >= x_max - boundary_thickness)
    
    new_top = [original_to_new_map[idx] for idx in inside_original_indices[top_mask]]
    new_bottom = [original_to_new_map[idx] for idx in inside_original_indices[bottom_mask]]
    new_left = [original_to_new_map[idx] for idx in inside_original_indices[left_mask]]
    new_right = [original_to_new_map[idx] for idx in inside_original_indices[right_mask]]
    # Return results for analysis
    return {
        'L_value': L,
        'sub_graph_nodes': sub_nodes,
        'sub_graph_neighbors': sub_neighbors,
        'sub_graph_edges': sub_edges,
        'node_count': len(sub_nodes),
        'edge_count': len(sub_edges),
        'top_boundary_nodes': np.array(new_top, dtype=np.int32),
        'bottom_boundary_nodes': np.array(new_bottom, dtype=np.int32),
        'left_boundary_nodes': np.array(new_left, dtype=np.int32),
        'right_boundary_nodes': np.array(new_right, dtype=np.int32),
        'center_x': center_x,
        'center_y': center_y
    }