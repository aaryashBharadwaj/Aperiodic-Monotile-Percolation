import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from hat_generator import mul, transPt

# Builds a graph representation of the hat tiling by extracting nodes and edges.
# Array-based build: identical graph to the original set/list version, but uses numpy
# + scipy instead of Python sets so patch r>=6 (level>=7, ~15M raw vertices, ~6.7M unique
# nodes) builds in ~2-3 GB instead of ~7 GB. No algorithmic change; output format is the
# same (unique node coords, and neighbours as a list of int32 arrays).
def build_neighbor_graph_fast(patch, level=0, tol=1e-5):
    # Pre-allocate raw-vertex array (grows exponentially with level).
    # Cap 20M so level>=7 (~15M raw vertices) builds without truncation.
    # level=None -> recurse until leaf tiles (for variable-depth patches like the
    # spectre, whose Mystics sit one level deeper); an int keeps the fixed-depth
    # behaviour the hat runners rely on.
    estimated_nodes = 20000000 if level is None else min(1000 * (4 ** level), 20000000)
    raw = np.empty((estimated_nodes, 2), dtype=np.float64)
    poly_sizes = []               # vertices per leaf polygon, in collection order
    cnt = [0]

    def _collect(patch, S, level):
        ch = getattr(patch, "children", None)
        if ch and (level is None or level > 0):
            nxt = None if level is None else level - 1
            for g in ch:
                _collect(g['geom'], mul(S, g['T']), nxt)
        else:
            shp = patch.shape
            for p in shp:
                q = transPt(S, p)
                if cnt[0] < len(raw):
                    raw[cnt[0], 0] = q['x']; raw[cnt[0], 1] = q['y']; cnt[0] += 1
            poly_sizes.append(len(shp))

    _collect(patch, [1, 0, 0, 0, 1, 0], level)
    V = cnt[0]
    raw = raw[:V]
    poly_sizes = np.asarray(poly_sizes, dtype=np.int64)

    # --- merge coincident vertices into unique nodes (connected components of the
    #     "within tol" graph), replacing the Python union-find + dicts ---
    tree = KDTree(raw)
    pairs = tree.query_pairs(r=tol, output_type='ndarray')   # (P,2) array, not a set
    del tree
    if len(pairs):
        g = coo_matrix((np.ones(len(pairs), dtype=np.int8),
                        (pairs[:, 0], pairs[:, 1])), shape=(V, V))
        n_unique, labels = connected_components(g, directed=False)
        del g
    else:
        n_unique, labels = V, np.arange(V)
    labels = labels.astype(np.int64)          # raw vertex -> unique node id
    del pairs

    unique_nodes = np.empty((n_unique, 2), dtype=np.float64)
    unique_nodes[labels] = raw                # coincident points share coords; any wins

    # --- edges: each leaf polygon contributes its outline edges (consecutive vertices,
    #     with wraparound). Vertices were collected polygon-by-polygon, so consecutive
    #     raw indices within a polygon block are adjacent. ---
    offsets = np.empty(len(poly_sizes) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(poly_sizes, out=offsets[1:])
    src = np.arange(V, dtype=np.int64)
    dst = src + 1
    dst[offsets[1:] - 1] = offsets[:-1]       # last vertex of each polygon wraps to its first
    del raw, poly_sizes

    u = labels[src]; v = labels[dst]
    del src, dst, labels
    m = u != v                                # drop self-loops from merged vertices
    lo = np.minimum(u[m], v[m]); hi = np.maximum(u[m], v[m])
    del u, v, m
    key = np.unique(lo * np.int64(n_unique) + hi)   # dedup undirected edges
    lo = key // n_unique
    hi = key % n_unique
    del key

    # --- adjacency as a list of int32 arrays (same format the callers expect) ---
    srcs = np.concatenate([lo, hi])
    dsts = np.concatenate([hi, lo]).astype(np.int32)
    del lo, hi
    order = np.argsort(srcs, kind='stable')
    srcs = srcs[order]; dsts = dsts[order]
    del order
    counts = np.bincount(srcs, minlength=n_unique)
    bounds = np.empty(n_unique + 1, dtype=np.int64)
    bounds[0] = 0
    np.cumsum(counts, out=bounds[1:])
    neighbors = [dsts[bounds[i]:bounds[i + 1]] for i in range(n_unique)]

    return unique_nodes, neighbors

#Convert adjacency list to Compressed Sparse Row (CSR) format for efficient storage.
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

##################################### SQUARE FRAME HAT TILING BUILD ##################################

# Extract a subgraph containing only specified nodes from the master graph.
def create_subgraph(master_nodes, master_neighbors, inside_original_indices):
    sub_nodes = master_nodes[inside_original_indices]
    num_sub_nodes = len(sub_nodes)
    original_to_new_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(inside_original_indices)}
    sub_neighbors = [[] for _ in range(num_sub_nodes)]
    sub_edges = set()
    
    for new_idx, orig_idx in enumerate(inside_original_indices):
        for nbr_orig_idx in master_neighbors[orig_idx]:
            new_nbr_idx = original_to_new_map.get(nbr_orig_idx)
            if new_nbr_idx is not None:
                sub_neighbors[new_idx].append(new_nbr_idx)
                sub_edges.add((min(new_idx, new_nbr_idx), max(new_idx, new_nbr_idx)))
                
    sub_neighbors = [np.array(n, dtype=np.int32) for n in sub_neighbors]
    sub_edges_list = np.array(list(sub_edges), dtype=np.int32)
    
    return sub_nodes, sub_neighbors, original_to_new_map, sub_edges_list

# Extract and analyze a square region of the hat tiling for percolation analysis.
def analyze_square_frame(master_nodes, master_neighbors, L, boundary_thickness=1.0,
                         center_x=200.0, center_y=-100.0):

    # This centre was fine-tuned to ensure the square-frame lies fully within the patch for L = 400 at patch 5.
    # For patch r=6 use center (515.0, -273.0), which admits a max inscribed square of ~1116 units.
    # Defaults preserve the original r=5 behaviour.
    
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