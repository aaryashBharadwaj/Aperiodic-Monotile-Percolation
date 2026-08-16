import numpy as np

# Generic graph operations shared by the direct and dual builders, nothing here knows about tilings.

def edges_to_adjacency(lo, hi, n):
    # edge[i] joins srcs[i] to dsts[i] 
    # So if srcs[4] = 1 and dsts[4] = 2: 1-2 is an edge
    srcs = np.concatenate([lo, hi])
    # it intentionally switches hi and lo, so that [1,2] and [2,1] both exist
    dsts = np.concatenate([hi, lo]).astype(np.int32)
    # sort these by their source
    order = np.argsort(srcs, kind='stable')
    srcs = srcs[order]; dsts = dsts[order]
    bounds = np.empty(n + 1, dtype=np.int64)
    bounds[0] = 0
    # np.bincount(srcs, minlength=n) gives the degree for a given node
    # the cumulative creates an index for each node and gives it to bounds
    np.cumsum(np.bincount(srcs, minlength=n), out=bounds[1:])
    # slicing this creates an adjacency graph
    return [dsts[bounds[i]:bounds[i + 1]] for i in range(n)]


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


# Cuts an L x L window out of a graph and returns it as a standalone subgraph
# It also identifies which existing nodes lie within boundary_thickness of each wall, and returns four sets of them
def analyze_rect_frame(nodes, neighbors, W, H, boundary_thickness=1.0, center_x=None, center_y=None):
    """Cut a W x H rectangular window centred at (center_x, center_y) and return the sub-graph plus its
    four boundary node sets. The square case (W == H = L) is the frame used for the p_c sweeps; W != H
    is used by the Cardy crossing test (aspect ratio a = W/H). Boundaries are the nodes within
    boundary_thickness of each of the four edges."""
    if center_x is None or center_y is None:
        raise ValueError("analyze_rect_frame needs an explicit centre (build_graph supplies it).")
    # min/max of the rectangle: W sets the horizontal half-extent, H the vertical
    x_min, x_max = center_x - W / 2.0, center_x + W / 2.0
    y_min, y_max = center_y - H / 2.0, center_y + H / 2.0

    # Find all nodes inside the rectangular region
    inside_mask = (nodes[:, 0] >= x_min) & (nodes[:, 0] <= x_max) & \
                  (nodes[:, 1] >= y_min) & (nodes[:, 1] <= y_max)
    inside_original_indices = np.where(inside_mask)[0]
    inside_nodes_coords = nodes[inside_original_indices]
    # Early return if region is too small
    if len(inside_original_indices) < 2:
        return {'node_count': 0}

    sub_nodes, sub_neighbors, original_to_new_map, sub_edges = create_subgraph(
        nodes, neighbors, inside_original_indices
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
        'W_value': W,
        'H_value': H,
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


def analyze_square_frame(nodes, neighbors, L, boundary_thickness=1.0, center_x=None, center_y=None):
    """The L x L square window -- the W == H special case of analyze_rect_frame. Kept as a named
    wrapper because the p_c sweeps (run_one, etc.) call it and use the 'L_value' key."""
    fd = analyze_rect_frame(nodes, neighbors, L, L, boundary_thickness, center_x, center_y)
    fd['L_value'] = L
    return fd
