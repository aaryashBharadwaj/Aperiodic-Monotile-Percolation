"""Isomorphism / lattice-identity checker.

Two jobs, both of which let us REPLACE a Monte-Carlo p_c run with a proof:

  (A) GRAPH ISOMORPHISM -- two tilings that produce the same graph (up to relabelling) percolate
      identically, so one run covers both. We delegate this to NetworkX: `nx.is_isomorphic` (the VF2
      algorithm) is an EXACT certificate; for very large graphs it falls back to the Weisfeiler-Leman
      hash (fast, strong evidence). This is how we show comet-aperiodic dual == hat dual and turtle == hat.

  (B) LOCAL LATTICE CLASSIFICATION -- a finite patch of a periodic dual can't be *globally* isomorphic to
      an infinite lattice (its boundary differs), so instead we read the boundary-FREE interior signature:
      for every node far enough from the patch edge that its whole local neighbourhood is present, record
      (degree, triangles-through-node). A vertex-transitive lattice gives ONE such pair for the whole
      interior -- triangular=(6,6), square=(4,0), honeycomb=(3,0). A single interior pair that matches a
      reference lattice IS that lattice locally; a spread of pairs is an aperiodic (novel) graph.

The heavy graph theory (isomorphism, WL, triangle counting) is NetworkX; the code here is only the thin
adjacency->graph conversion and the geometric interior classifier. Angles/edge lengths never enter --
percolation ignores them.
"""
import numpy as np
import networkx as nx


# --------------------------------------------------------------------------- graph isomorphism (NetworkX)
def to_networkx(neighbors):
    """Adjacency list (list of int arrays) -> undirected nx.Graph on nodes 0..n-1. Each undirected edge
    is added once (the `i < j` guard)."""
    G = nx.Graph()
    G.add_nodes_from(range(len(neighbors)))
    G.add_edges_from((i, int(j)) for i, row in enumerate(neighbors) for j in row if i < int(j))
    return G


def graph_hash(neighbors, iterations=4):
    """NetworkX Weisfeiler-Leman graph hash: the standard label-independent fingerprint, fast at any size."""
    return nx.weisfeiler_lehman_graph_hash(to_networkx(neighbors), iterations=iterations)


def isomorphic(neighbors_a, neighbors_b, exact=None):
    """Are the two graphs isomorphic (up to relabelling)? Returns (bool, detail).

    Uses NetworkX: VF2 (`nx.is_isomorphic`) is an EXACT certificate but backtracks badly on structured
    lattice graphs (~40 s already at ~3600 nodes), so we only auto-select it for SMALL graphs (< 500 nodes,
    e.g. self-dual validation checks). Larger graphs compare Weisfeiler-Leman hashes (fast at any size --
    ~1 s at 20k -- strong evidence, not a certificate). Set `exact=True`/`False` to force. A vertex/edge-
    count mismatch short-circuits to False."""
    Ga, Gb = to_networkx(neighbors_a), to_networkx(neighbors_b)
    sa = (Ga.number_of_nodes(), Ga.number_of_edges())
    sb = (Gb.number_of_nodes(), Gb.number_of_edges())
    if sa != sb:
        return False, {"method": "size-mismatch", "a": sa, "b": sb}
    if exact is None:
        exact = sa[0] < 500
    if exact:
        ok, method = nx.is_isomorphic(Ga, Gb), "VF2 (exact)"
    else:
        ok, method = graph_hash(neighbors_a) == graph_hash(neighbors_b), "WL-hash (large graph)"
    return ok, {"method": method, "V": sa[0], "E": sa[1]}


# --------------------------------------------------------------------------- interior lattice signature
def _median_edge_length(coords, neighbors, sample=2000):
    coords = np.asarray(coords, float)
    lens = []
    step = max(1, len(neighbors) // sample)
    for i in range(0, len(neighbors), step):
        for j in neighbors[i]:
            lens.append(np.hypot(*(coords[i] - coords[j])))
    return float(np.median(lens)) if lens else 1.0


def interior_indices(coords, neighbors, margin_edges=2.5):
    """Nodes at least margin_edges * (median edge length) inside the bounding box. For these the full
    local neighbourhood (and every triangle through them) is present, so their degree/triangle counts
    are the TRUE lattice values, not boundary-truncated ones."""
    coords = np.asarray(coords, float)
    m = margin_edges * _median_edge_length(coords, neighbors)
    xmn, ymn = coords.min(0); xmx, ymx = coords.max(0)
    inside = (coords[:, 0] >= xmn + m) & (coords[:, 0] <= xmx - m) & \
             (coords[:, 1] >= ymn + m) & (coords[:, 1] <= ymx - m)
    return np.where(inside)[0]


def interior_signature(coords, neighbors, margin_edges=2.5):
    """The boundary-free fingerprint: the multiset of (degree, triangles) over interior nodes, plus the
    single dominant pair and how uniform the interior is. Triangle counts come from NetworkX. A periodic
    lattice -> one dominant pair covering ~all interior nodes; an aperiodic graph -> a spread."""
    idx = interior_indices(coords, neighbors, margin_edges)
    if len(idx) == 0:
        return {"n_interior": 0, "dominant": None, "uniformity": 0.0, "pairs": {}}
    tri = nx.triangles(to_networkx(neighbors))                 # {node: #triangles through it}
    pairs = {}
    for i in idx:
        key = (len(neighbors[i]), int(tri[int(i)]))
        pairs[key] = pairs.get(key, 0) + 1
    dom = max(pairs, key=pairs.get)
    return {"n_interior": int(len(idx)), "dominant": dom,
            "uniformity": pairs[dom] / len(idx), "pairs": pairs}


# reference interior signatures of the classic lattices (degree, triangles-through-node)
REFERENCE_LATTICES = {
    "triangular": (6, 6),
    "square":     (4, 0),
    "honeycomb":  (3, 0),
}


def classify_lattice(coords, neighbors, margin_edges=2.5, uniformity_min=0.90):
    """Name the lattice a (dual) graph's interior locally is, or 'aperiodic/novel'. Returns
    (name, signature_dict). A match requires the interior to be ~uniform (uniformity >= uniformity_min)
    AND its dominant (degree, triangle) pair to equal a reference lattice's."""
    sig = interior_signature(coords, neighbors, margin_edges)
    if sig["dominant"] is None or sig["uniformity"] < uniformity_min:
        return "aperiodic/novel", sig
    for name, ref in REFERENCE_LATTICES.items():
        if sig["dominant"] == ref:
            return name, sig
    return "periodic (unlisted)", sig
