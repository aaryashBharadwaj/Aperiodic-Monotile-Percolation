import numpy as np
from scipy.spatial import KDTree
from hat_tiling import mul, transPt

def build_neighbor_graph_fast(patch, level=0):
    estimated_nodes = min(1000 * (4 ** level), 10000000)
    nodes = np.empty((estimated_nodes, 2), dtype=np.float64)
    node_count = [0]
    
    def _collect_nodes(patch, S, level):
        if level > 0 and hasattr(patch, "children"):
            for g in patch.children:
                _collect_nodes(g['geom'], mul(S, g['T']), level-1)
        else:
            for p in patch.shape:
                pt_screen = transPt(S, p)
                if node_count[0] < len(nodes):
                    nodes[node_count[0]] = [pt_screen['x'], pt_screen['y']]
                    node_count[0] += 1
    
    _collect_nodes(patch, [1,0,0,0,1,0], level)
    nodes = nodes[:node_count[0]]
    
    tree = KDTree(nodes)
    tol = 1e-5
    pairs = tree.query_pairs(r=tol)
    
    parent = np.arange(len(nodes))
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    
    for i, j in pairs:
        union(i, j)
    
    mapping = np.array([find(i) for i in range(len(nodes))])
    unique_ids = np.unique(mapping)
    id_to_idx = {uid: i for i, uid in enumerate(unique_ids)}
    node_to_unique = np.array([id_to_idx[mapping[i]] for i in range(len(nodes))])
    unique_nodes = nodes[unique_ids]
    
    edges_set = set()
    node_idx = 0
    
    def _collect_edges(patch, S, level):
        nonlocal node_idx
        if level > 0 and hasattr(patch, "children"):
            for g in patch.children:
                _collect_edges(g['geom'], mul(S, g['T']), level-1)
        else:
            n = len(patch.shape)
            base_idx = node_idx
            for i in range(n):
                idx1 = node_to_unique[base_idx + i]
                idx2 = node_to_unique[base_idx + (i+1)%n]
                if idx1 != idx2:
                    edges_set.add((min(idx1, idx2), max(idx1, idx2)))
            node_idx += n
    
    _collect_edges(patch, [1,0,0,0,1,0], level)
    
    neighbors = [[] for _ in range(len(unique_nodes))]
    for i, j in edges_set:
        neighbors[i].append(j)
        neighbors[j].append(i)
    neighbors = [np.array(n, dtype=np.int32) for n in neighbors]
    
    return unique_nodes, neighbors

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

##################################### SQUARE FRAME HATTILING BUILD ##################################

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


def analyze_square_frame(master_nodes, master_neighbors, L, boundary_thickness=1.0):
    # Calculate the actual center of the tiling
    center_x = (master_nodes[:, 0].min() + master_nodes[:, 0].max()) / 2.0
    center_y = (master_nodes[:, 1].min() + master_nodes[:, 1].max()) / 2.0
    
    # Create square frame centered on the tiling
    x_min, x_max = center_x - L / 2.0, center_x + L / 2.0
    y_min, y_max = center_y - L / 2.0, center_y + L / 2.0
    
    nodes = master_nodes
    inside_mask = (nodes[:, 0] >= x_min) & (nodes[:, 0] <= x_max) & \
                  (nodes[:, 1] >= y_min) & (nodes[:, 1] <= y_max)
    
    inside_original_indices = np.where(inside_mask)[0]
    inside_nodes_coords = nodes[inside_original_indices]
    
    if len(inside_original_indices) < 2:
        return {'node_count': 0}

    sub_nodes, sub_neighbors, original_to_new_map, sub_edges = create_subgraph(
        master_nodes, master_neighbors, inside_original_indices
    )

    top_mask = (inside_nodes_coords[:, 1] >= y_max - boundary_thickness)
    bottom_mask = (inside_nodes_coords[:, 1] <= y_min + boundary_thickness)
    left_mask = (inside_nodes_coords[:, 0] <= x_min + boundary_thickness)
    right_mask = (inside_nodes_coords[:, 0] >= x_max - boundary_thickness)
    
    new_top = [original_to_new_map[idx] for idx in inside_original_indices[top_mask]]
    new_bottom = [original_to_new_map[idx] for idx in inside_original_indices[bottom_mask]]
    new_left = [original_to_new_map[idx] for idx in inside_original_indices[left_mask]]
    new_right = [original_to_new_map[idx] for idx in inside_original_indices[right_mask]]
    
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