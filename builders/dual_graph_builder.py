import numpy as np
from generators.hat_generator import mul, transPt
from builders.graph_core import edges_to_adjacency
from builders.direct_graph_builder import dedup_vertices

# This calculates the Delone dual not the planar dual, this is because the planar dual isn't a graph

def collect_leaf_polygons(patch, level=0):
    tile_polygons = []
    def _collect_tiles(patch, S, level):
        ch = getattr(patch, "children", None)
        # recurses to the depth given by level; pass level=None to recurse all the way to leaf
        # A fixed level smaller than the deepest branch silently truncates: you get metatile outlines instead of tiles
        if ch and (level is None or level > 0):
            nxt = None if level is None else level - 1
            for g in ch:
                _collect_tiles(g['geom'], mul(S, g['T']), nxt)
        else:
            poly = np.array([[transPt(S, p)['x'], transPt(S, p)['y']] for p in patch.shape])
            tile_polygons.append(poly)
    _collect_tiles(patch, [1, 0, 0, 0, 1, 0], level)
    return tile_polygons


def build_dual_from_polygons(tile_polygons, tol=1e-5):
    """Delone (tile-adjacency) dual: two tiles are neighbours iff they share a boundary EDGE (a segment
    between two consecutive vertices), NOT merely a vertex.

    This is built from shared edges, not the older ">=2 shared vertices" proxy. That proxy is correct on
    clean convex tiles but miscounts on FOLDED tiles: a collapsed edge (a vertex the tile lists twice) or a
    two-corner point contact makes two tiles share two vertices without sharing an edge, faking an
    adjacency. Comparing the actual boundary segments is exact for all of these -- and it is bit-identical
    to the old rule on every non-degenerate tiling (hat/spectre/chevron/square/triangular/Tile(1,1))."""
    N = len(tile_polygons)
    tile_centroids = np.array([poly.mean(axis=0) for poly in tile_polygons])
    sizes = np.array([len(p) for p in tile_polygons])
    all_verts = np.concatenate(tile_polygons, axis=0)
    tile_ids = np.repeat(np.arange(N), sizes)
    V = len(all_verts)

    # Merge coincident vertices across the whole patch -> a single global label per physical point.
    unique_nodes, labels = dedup_vertices(all_verts, tol)
    n_unique = len(unique_nodes)

    # Each tile's boundary edges = consecutive (wrapping) vertex pairs. `nxt` is the next flat index within
    # the SAME tile (the last vertex of a tile wraps back to its first).
    offs = np.concatenate([[0], np.cumsum(sizes)])
    nxt = np.arange(V) + 1
    nxt[offs[1:] - 1] = offs[:-1]
    la = labels; lb = labels[nxt]
    good = la != lb                                       # drop zero-length (collapsed) edges
    lo = np.minimum(la[good], lb[good]).astype(np.int64)
    hi = np.maximum(la[good], lb[good]).astype(np.int64)
    ek = lo * n_unique + hi                               # one key per undirected boundary edge
    tk = tile_ids[good]

    # Keep one (edge, tile) record each, then group by edge. An edge owned by >=2 distinct tiles joins them.
    order = np.lexsort((tk, ek)); ek = ek[order]; tk = tk[order]
    keep = np.concatenate([[True], (ek[1:] != ek[:-1]) | (tk[1:] != tk[:-1])])
    ek = ek[keep]; tk = tk[keep]
    bnd = np.concatenate([[0], np.nonzero(ek[1:] != ek[:-1])[0] + 1, [len(ek)]])
    gs = bnd[:-1]; gl = np.diff(bnd)

    two = gl == 2                                         # the normal case: an interior edge borders 2 tiles
    A = [tk[gs[two]]]; B = [tk[gs[two] + 1]]
    for g in np.nonzero(gl > 2)[0]:                      # rare: an edge meeting >2 tiles (a degenerate point)
        ts = tk[bnd[g]:bnd[g + 1]]
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                A.append(ts[i:i + 1]); B.append(ts[j:j + 1])
    ai = np.concatenate(A); bi = np.concatenate(B)
    if len(ai) == 0:
        return tile_centroids, [np.array([], dtype=np.int32) for _ in range(N)], tile_polygons
    elo = np.minimum(ai, bi); ehi = np.maximum(ai, bi)
    ukey = np.unique(elo.astype(np.int64) * N + ehi)     # dedup (two tiles may share more than one edge)
    tile_neighbors = edges_to_adjacency(ukey // N, ukey % N, N)
    return tile_centroids, tile_neighbors, tile_polygons