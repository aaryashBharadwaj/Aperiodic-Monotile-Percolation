"""Isomorphism / lattice-identity checker -- EXPLICIT constructions only. No VF2, no NetworkX, no
Weisfeiler-Leman hash: the certificate is always an explicit node-to-node mapping phi that we write down
and verify edge-by-edge. Each identity lets us replace a Monte-Carlo p_c run with a proof a referee reads.

  (A) GRAPH IDENTITY (bit-identical, phi = identity).  Two tilings that emit the SAME graph with the SAME
      node ordering percolate identically. Canonicalise each node's neighbour list (sort) and compare the
      arrays. Whole-graph certificate. Used for  comet-aperiodic dual == hat dual  and  turtle == hat.

  (B) DEVELOP ONTO A KNOWN LATTICE.  A periodic dual whose interior is a regular lattice is certified by
      CONSTRUCTING an explicit bijection  phi: node -> integer lattice coordinate  and checking phi maps
      every interior edge onto a lattice neighbour offset. phi is built by developing the planar rotation
      system (the "grow from any node" construction): order each node's neighbours CCW by the drawn angle,
      seed one node at (0,0) with its link aligned to the lattice's offsets, then walk edge-by-edge -- each
      step places a node and fixes its frame from where its already-placed neighbour sits in its own CCW
      order. Used for  comet-periodic dual -> triangular,  tile11 dual -> triangular,  chevron-periodic
      dual -> square.  (This is the composition of the hand-built tile-adjacency <-> herringbone map with
      herringbone <-> triangular, realised in one pass; the herringbone's two vertex orbits are handled
      automatically because the walk assigns coordinates combinatorially.)

Finite-patch BOUNDARY nodes have truncated neighbourhoods (degree < lattice degree); they are placed but
not required and not propagated, since the infinite tiling has no boundary. Only full-degree interior nodes
are verified. Angles/edge lengths enter ONLY to orient the seed and read the rotation system -- percolation
ignores geometry, and the certificate itself is the integer map phi plus the exact edge check.
"""
from collections import deque
import numpy as np

# Target-lattice neighbour offsets, listed CCW. Each offset's negation sits d/2 further round the list --
# the develop step relies on this (offsets[(j + d//2) % d] == -offsets[j]).
TRIANGULAR = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]   # degree 6, hexagonal vertex link
SQUARE     = [(1, 0), (0, 1), (-1, 0), (0, -1)]                     # degree 4
REFERENCE = {"triangular": TRIANGULAR, "square": SQUARE}


# --------------------------------------------------------------------------- (A) bit-identical identity
def bit_identical(neighbors_a, neighbors_b):
    """Are the two graphs identical up to neighbour-list order (phi = the identity map)? Sort each node's
    neighbour list and compare. Returns (bool, detail). This is the whole-graph certificate for the
    aperiodic identities (comet-aperiodic dual == hat dual, turtle == hat)."""
    if len(neighbors_a) != len(neighbors_b):
        return False, {"reason": "vertex-count", "a": len(neighbors_a), "b": len(neighbors_b)}
    for i, (ra, rb) in enumerate(zip(neighbors_a, neighbors_b)):
        a = np.sort(np.asarray(ra, dtype=np.int64))
        b = np.sort(np.asarray(rb, dtype=np.int64))
        if a.shape != b.shape or not np.array_equal(a, b):
            return False, {"reason": "adjacency", "node": i}
    return True, {"phi": "identity", "V": len(neighbors_a)}


# --------------------------------------------------------------------------- (B) develop onto a lattice
def develop_to_lattice(coords, neighbors, offsets):
    """CONSTRUCT phi: node -> integer lattice coordinate by developing the planar rotation system, then
    verify phi preserves every interior edge. `offsets` = the target lattice's CCW neighbour offsets
    (TRIANGULAR or SQUARE). Returns (is_that_lattice, phi, stats).

    is_that_lattice is True iff phi is injective on the interior AND every interior (full-degree) edge maps
    to a lattice offset. phi is the explicit certificate; stats reports edges_preserved / edges_total.
    Pure numpy -- no VF2, no NetworkX, no hashing."""
    d = len(offsets); offset_set = set(offsets)
    adj = [list(map(int, r)) for r in neighbors]
    deg = np.array([len(r) for r in adj])
    X = np.asarray(coords, float)
    # rotation system: each node's neighbours ordered CCW by the drawn angle
    cyc = [[adj[n][k] for k in np.argsort(np.arctan2(X[adj[n]][:, 1] - X[n, 1],
            X[adj[n]][:, 0] - X[n, 0]))] if adj[n] else [] for n in range(len(adj))]
    seeds = [int(n) for n in np.argsort(((X - X.mean(0)) ** 2).sum(1)) if deg[n] == d]
    if not seeds:
        return False, {}, {"reason": "no full-degree node", "V": len(adj)}
    c0 = seeds[0]
    phi = {c0: (0, 0)}; rot = {c0: 0}; Q = deque([c0])
    while Q:
        p = Q.popleft()
        if len(cyc[p]) != d:
            continue
        for k, q in enumerate(cyc[p]):
            off = offsets[(k + rot[p]) % d]
            cq = (phi[p][0] + off[0], phi[p][1] + off[1])
            if q in phi:
                continue
            if deg[q] != d:                         # boundary node: place, do not propagate/require
                phi[q] = cq
                continue
            rot[q] = (k + rot[p] + d // 2 - cyc[q].index(p)) % d
            phi[q] = cq
            Q.append(q)
    # verify injectivity on interior nodes + edge preservation
    seen = {}; collisions = 0
    for n, cc in phi.items():
        if deg[n] == d and cc in seen and deg[seen[cc]] == d:
            collisions += 1
        seen[cc] = n
    good = total = 0
    for n, cc in phi.items():
        if deg[n] != d:
            continue
        for m in adj[n]:
            if m in phi and deg[m] == d:
                total += 1
                good += ((phi[m][0] - cc[0], phi[m][1] - cc[1]) in offset_set)
    ok = (collisions == 0 and total > 0 and good == total)
    return ok, phi, {"V": len(adj), "assigned": len(phi), "collisions": collisions,
                     "edges_preserved": good, "edges_total": total}


def is_lattice(coords, neighbors, name):
    """Convenience: develop the dual's interior onto REFERENCE[name] and return (bool, stats)."""
    ok, _phi, stats = develop_to_lattice(coords, neighbors, REFERENCE[name])
    stats["lattice"] = name
    return ok, stats
