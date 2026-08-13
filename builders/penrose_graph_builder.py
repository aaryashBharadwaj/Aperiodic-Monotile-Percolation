import numpy as np
from builders.graph_core import edges_to_adjacency
from builders.direct_graph_builder import dedup_vertices

TOL = 1e-5

# The penrose needs a special builder because  of it's diagonal edge requiring special care

def build_penrose_neighbor_graph(tiling):
    tris = tiling.triangles
    N = len(tris)

    # Flatten every triangle's 3 vertices into one array, in (v1, v2, v3) order per triangle,
    # so triangle i owns raw rows 3i, 3i+1, 3i+2.
    raw = np.empty((3 * N, 2), dtype=np.float64)
    for i, t in enumerate(tris):
        raw[3 * i]     = (t.v1.real, t.v1.imag)
        raw[3 * i + 1] = (t.v2.real, t.v2.imag)
        raw[3 * i + 2] = (t.v3.real, t.v3.imag)
    V = len(raw)

    unique_nodes, labels = dedup_vertices(raw, TOL)
    n_unique = len(unique_nodes)
    print(f"  Deduplicated to {n_unique} unique vertices")

    # Since a rhombus doesn't have an edge for it's diagonal we keep (v1, v2) and (v1, v3) but not (v2, v3)
    # Map each endpoint to its unique node, drop self pairs, and dedup.
    base = np.arange(N) * 3
    src = np.concatenate([base, base])
    dst = np.concatenate([base + 2, base + 1])
    u = labels[src]; v = labels[dst]
    m = u != v
    lo = np.minimum(u[m], v[m]); hi = np.maximum(u[m], v[m])
    key = np.unique(lo * np.int64(n_unique) + hi)   # pack (lo,hi) so one np.unique dedups edges
    lo = key // n_unique; hi = key % n_unique
    print(f"  Collected {len(key)} percolation edges")

    neighbors = edges_to_adjacency(lo, hi, n_unique)

    edges = np.column_stack([lo, hi]).astype(np.int32)
    return unique_nodes, neighbors, edges
