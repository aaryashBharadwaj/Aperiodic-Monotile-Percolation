"""Isomorphism / lattice-identity checker.

Two jobs, both of which let us REPLACE a Monte-Carlo p_c run with a proof:

  (A) EXACT IDENTITY -- two graphs built in the SAME physical coordinate frame are the same graph iff,
      after merging coincident nodes, their edge sets are bit-identical. This is how we show
      comet-aperiodic dual == hat dual (one folds into the other, so the tile-adjacency is preserved),
      and it is the primitive a turtle<->hat re-embedding check calls. No percolation needed: identical
      graph => identical threshold by construction.

  (B) LOCAL LATTICE CLASSIFICATION -- a finite patch of a periodic dual can't be *globally* isomorphic to
      an infinite lattice (its boundary differs), so instead we read the boundary-FREE interior signature:
      for every node far enough from the patch edge that its whole local neighbourhood is present, record
      (degree, triangles-through-node). A vertex-transitive lattice gives ONE such pair for the whole
      interior -- triangular=(6,6), square=(4,0), honeycomb=(3,0). A single interior pair that matches a
      reference lattice IS that lattice locally; a spread of pairs is an aperiodic (novel) graph.

Everything here is graph structure only -- angles and edge lengths never enter (percolation ignores them).
"""
import numpy as np


# --------------------------------------------------------------------------- basic invariants
def degree_histogram(neighbors):
    """{degree: count} over all nodes."""
    degs = np.fromiter((len(nb) for nb in neighbors), dtype=np.int64, count=len(neighbors))
    vals, cnts = np.unique(degs, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnts)}


def edge_count(neighbors):
    """Undirected edge count (adjacency stores each edge twice)."""
    return int(sum(len(nb) for nb in neighbors) // 2)


def triangles_per_node(neighbors):
    """triangles[i] = number of triangles node i sits in. For each node we count neighbour PAIRS that
    are themselves adjacent; that counts each of the node's triangles exactly once."""
    nbr_sets = [set(int(x) for x in nb) for nb in neighbors]
    tri = np.zeros(len(neighbors), dtype=np.int64)
    for i, s in enumerate(nbr_sets):
        nb = list(s)
        c = 0
        for a_idx in range(len(nb)):
            sa = nbr_sets[nb[a_idx]]
            for b_idx in range(a_idx + 1, len(nb)):
                if nb[b_idx] in sa:
                    c += 1
        tri[i] = c
    return tri


# --------------------------------------------------------------------------- Weisfeiler-Lehman
def wl_colors(neighbors, rounds=6):
    """Weisfeiler-Lehman colour refinement. Each round a node's new colour = hash of (its colour, the
    sorted multiset of neighbour colours), then colours are re-packed to 0..k-1. Returns the final
    per-node colour array. Identical WL colour histograms are strong (not complete) evidence of
    isomorphism -- the standard graph fingerprint, and what the turtle==hat identity was checked with."""
    n = len(neighbors)
    colors = np.zeros(n, dtype=np.int64)                       # start: every node the same colour
    degs = np.fromiter((len(nb) for nb in neighbors), dtype=np.int64, count=n)
    colors = degs.copy()                                       # seed with degree (a cheap first split)
    for _ in range(rounds):
        new = np.empty(n, dtype=np.int64)
        code = {}                                              # signature tuple -> dense colour id
        for i in range(n):
            nc = np.sort(colors[neighbors[i]])
            sig = (int(colors[i]),) + tuple(int(x) for x in nc)
            c = code.get(sig)
            if c is None:
                c = len(code); code[sig] = c
            new[i] = c
        if len(code) == len(np.unique(colors)):               # stable: refinement added nothing new
            colors = new
            break
        colors = new
    return colors


def wl_histogram(colors):
    """Canonical, label-independent fingerprint: the sorted tuple of colour-class sizes."""
    _, cnts = np.unique(colors, return_counts=True)
    return tuple(sorted(int(c) for c in cnts))


def graph_fingerprint(neighbors, rounds=6):
    """Bundle the label-independent invariants used to compare two graphs up to isomorphism."""
    return {
        "V": len(neighbors),
        "E": edge_count(neighbors),
        "degree_hist": degree_histogram(neighbors),
        "triangle_hist": _hist(triangles_per_node(neighbors)),
        "wl_hist": wl_histogram(wl_colors(neighbors, rounds)),
    }


def _hist(arr):
    vals, cnts = np.unique(arr, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnts)}


def isomorphic(neighbors_a, neighbors_b, rounds=6):
    """Isomorphism up to relabelling, via matching invariants (V, E, degree/triangle histograms, WL
    histogram). WL is not a complete invariant, so this returns 'strong evidence', not a mathematical
    certificate -- but for these lattice-vs-lattice comparisons it is decisive. Use identical_in_frame()
    when the two graphs share a coordinate frame; that IS a certificate."""
    fa, fb = graph_fingerprint(neighbors_a, rounds), graph_fingerprint(neighbors_b, rounds)
    return fa == fb, {"a": fa, "b": fb}


# --------------------------------------------------------------------------- exact identity in a frame
def identical_in_frame(coords_a, neighbors_a, coords_b, neighbors_b, tol=1e-5):
    """CERTIFICATE-grade identity: when both graphs live in the same physical coordinate frame (e.g. two
    tilings that fold into each other), merge coincident nodes across BOTH, relabel every edge to those
    shared node ids, and check the two undirected edge sets are exactly equal. Bit-identical edge sets
    on shared nodes => the same graph, so the same p_c by construction (no Monte Carlo)."""
    from builders.direct_graph_builder import dedup_vertices
    ca, cb = np.asarray(coords_a, float), np.asarray(coords_b, float)
    na, nb = len(ca), len(cb)
    _, lab = dedup_vertices(np.concatenate([ca, cb], axis=0), tol)     # global label per physical point
    la, lb = lab[:na], lab[na:]

    def edgeset(labels, neighbors):
        es = set()
        for i, row in enumerate(neighbors):
            u = int(labels[i])
            for j in row:
                v = int(labels[j])
                if u != v:
                    es.add((u, v) if u < v else (v, u))
        return es

    ea, eb = edgeset(la, neighbors_a), edgeset(lb, neighbors_b)
    same_nodes = (len(set(int(x) for x in la)) == len(set(int(x) for x in lb))
                  == len(set(int(x) for x in np.concatenate([la, lb]))))
    ok = same_nodes and (ea == eb)
    detail = {"V_a": len(set(int(x) for x in la)), "V_b": len(set(int(x) for x in lb)),
              "E_a": len(ea), "E_b": len(eb), "E_shared": len(ea & eb),
              "E_only_a": len(ea - eb), "E_only_b": len(eb - ea), "nodes_coincide": same_nodes}
    return ok, detail


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
    single dominant pair and how uniform the interior is. A periodic lattice -> one dominant pair
    covering ~all interior nodes; an aperiodic graph -> a spread."""
    idx = interior_indices(coords, neighbors, margin_edges)
    if len(idx) == 0:
        return {"n_interior": 0, "dominant": None, "uniformity": 0.0, "pairs": {}}
    tri = triangles_per_node(neighbors)
    pairs = {}
    for i in idx:
        key = (len(neighbors[i]), int(tri[i]))
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
