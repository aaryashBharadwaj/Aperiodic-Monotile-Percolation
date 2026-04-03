import numpy as np
from scipy.spatial import KDTree

TOL = 1e-5


def build_penrose_neighbor_graph(tiling):
    # Step 1: Collect ALL vertices from all triangles
    all_vertices = []

    for triangle in tiling.triangles:
        verts = triangle.get_all_vertices()
        tri_verts = []
        for v in verts:
            all_vertices.append([v.real, v.imag])
            tri_verts.append(len(all_vertices) - 1)

    nodes = np.array(all_vertices, dtype=np.float64)

    # Step 2: Deduplicate vertices using KDTree
    tree = KDTree(nodes)
    pairs = tree.query_pairs(r=TOL)

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

    print(f"  Deduplicated to {len(unique_nodes)} unique vertices")

    # Step 3: Collect edges for percolation from each triangle
    print(f"  Extracting percolation edges from triangles...")
    vertex_tree = KDTree(nodes)
    rhombus_edge_set = set()

    for i, triangle in enumerate(tiling.triangles):
        try:
            edges = triangle.get_edges_for_percolation()
            for v_start, v_end in edges:
                v_start_arr = np.array([[v_start.real, v_start.imag]])
                v_end_arr   = np.array([[v_end.real,   v_end.imag]])

                _, start_orig_idx = vertex_tree.query(v_start_arr, k=1)
                _, end_orig_idx   = vertex_tree.query(v_end_arr,   k=1)

                start_idx = node_to_unique[start_orig_idx[0]]
                end_idx   = node_to_unique[end_orig_idx[0]]

                if start_idx != end_idx:
                    edge = (min(start_idx, end_idx), max(start_idx, end_idx))
                    rhombus_edge_set.add(edge)
        except Exception as e:
            print(f"  Warning: Error processing triangle {i}: {e}")
            continue

    print(f"  Collected {len(rhombus_edge_set)} percolation edges")

    # Step 4: Build neighbor lists
    neighbors = [[] for _ in range(len(unique_nodes))]
    for i, j in rhombus_edge_set:
        neighbors[i].append(j)
        neighbors[j].append(i)
    neighbors = [np.array(n, dtype=np.int32) for n in neighbors]

    return unique_nodes, neighbors, np.array(list(rhombus_edge_set), dtype=np.int32)


def analyze_square_frame(master_nodes, master_neighbors, master_edges, L, boundary_thickness):
    # Centre on bounding box centroid
    cx = (master_nodes[:, 0].min() + master_nodes[:, 0].max()) / 2.0
    cy = (master_nodes[:, 1].min() + master_nodes[:, 1].max()) / 2.0

    x_min, x_max = cx - L/2, cx + L/2
    y_min, y_max = cy - L/2, cy + L/2

    inside_mask = (
        (master_nodes[:, 0] >= x_min) & (master_nodes[:, 0] <= x_max) &
        (master_nodes[:, 1] >= y_min) & (master_nodes[:, 1] <= y_max)
    )
    inside_idx    = np.where(inside_mask)[0]
    inside_coords = master_nodes[inside_idx]

    if len(inside_idx) < 2:
        return {'node_count': 0}

    orig_to_new = {orig: new for new, orig in enumerate(inside_idx)}

    sub_neighbors = [[] for _ in range(len(inside_idx))]
    edges_set     = set()

    for new_i, orig_i in enumerate(inside_idx):
        for orig_j in master_neighbors[orig_i]:
            new_j = orig_to_new.get(orig_j)
            if new_j is not None:
                sub_neighbors[new_i].append(new_j)
                edges_set.add((min(new_i, new_j), max(new_i, new_j)))

    sub_neighbors  = [np.array(n, dtype=np.int32) for n in sub_neighbors]
    sub_edges      = np.array(list(edges_set), dtype=np.int32)

    top_mask    = inside_coords[:, 1] >= y_max - boundary_thickness
    bottom_mask = inside_coords[:, 1] <= y_min + boundary_thickness
    left_mask   = inside_coords[:, 0] <= x_min + boundary_thickness
    right_mask  = inside_coords[:, 0] >= x_max - boundary_thickness

    def boundary(mask):
        return np.array([orig_to_new[inside_idx[i]] for i in np.where(mask)[0]], dtype=np.int32)

    return {
        'node_count':            len(inside_idx),
        'edge_count':            len(edges_set),
        'sub_graph_nodes':       inside_coords,
        'sub_graph_neighbors':   sub_neighbors,
        'sub_graph_edges':       sub_edges,
        'top_boundary_nodes':    boundary(top_mask),
        'bottom_boundary_nodes': boundary(bottom_mask),
        'left_boundary_nodes':   boundary(left_mask),
        'right_boundary_nodes':  boundary(right_mask),
        'center_x': cx,
        'center_y': cy,
    }