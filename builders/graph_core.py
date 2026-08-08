"""The one graph-assembly step shared by BOTH graph builders (direct AND dual).

edges_to_adjacency is the single primitive common to every graph in the project: given a deduped
undirected edge list, build the neighbour adjacency. It lives here, on its own, so that step is
written and verified ONCE -- the direct builder (direct_graph_builder), the penrose builder, and the
dual builder (dual_graph_builder) all end here.

Everything else is builder-specific and lives with its builder, NOT in this shared module: vertex
dedup + the perimeter direct graph are in direct_graph_builder (used by the hat/spectre patch build,
the periodic family blocks, and penrose's vertex dedup); the shared-vertex-count dual assembly is in
dual_graph_builder. Only this one step is genuinely common to direct AND dual, so only it lives here.
"""
import numpy as np


def edges_to_adjacency(lo, hi, n):
    """Build neighbour lists from a deduped undirected edge list.

    lo, hi : equal-length int arrays; edge i connects node lo[i] to node hi[i] (already unique,
             one direction, no duplicates). n : number of nodes.
    Returns a list of n int32 arrays (each node's neighbours). Both directions are emitted, then
    grouped by source with a stable sort + bincount (the same assembly every builder ends with).
    """
    # Emit both directions, then a stable sort by source groups each node's edges contiguously, so
    # bincount + cumsum give the [start, end) slice bounds of every node's neighbour list.
    srcs = np.concatenate([lo, hi])
    dsts = np.concatenate([hi, lo]).astype(np.int32)
    order = np.argsort(srcs, kind='stable')
    srcs = srcs[order]; dsts = dsts[order]
    bounds = np.empty(n + 1, dtype=np.int64)
    bounds[0] = 0
    np.cumsum(np.bincount(srcs, minlength=n), out=bounds[1:])
    return [dsts[bounds[i]:bounds[i + 1]] for i in range(n)]
