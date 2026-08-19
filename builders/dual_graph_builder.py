import numpy as np
from scipy.spatial import KDTree
from generators.hat_generator import mul, transPt
from builders.graph_core import edges_to_adjacency

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
    N = len(tile_polygons)

    # One node per tile, positioned at the tile's centroid.
    tile_centroids = np.array([poly.mean(axis=0) for poly in tile_polygons])

    # Dedup each tile's OWN vertices before the shared-vertex count. A folded tile can list the same
    # physical point twice at NON-adjacent ring indices (which _dedup_ring, consecutive-only, misses);
    # left in, it produces two cross-tile coincidences for a single POINT contact -> count>=2 -> a fake
    # shared EDGE. Deduping per tile makes count>=2 mean ">=2 distinct shared points". Guaranteed no-op for
    # any tiling whose tiles have no repeated vertex (hat/spectre/chevron/square/triangular/Tile(1,1)).
    def _uniq_ring(p):
        k = np.round(p / tol).astype(np.int64)
        _, idx = np.unique(k, axis=0, return_index=True)
        return p[np.sort(idx)] if len(idx) < len(p) else p
    dedup_polys = [_uniq_ring(p) for p in tile_polygons]

    # Flatten every tile's vertices into one array, remembering which tile each vertex belongs to.
    # tile_ids[k] is the owner of vertex k (so tile_ids[20]=1 means flat-vertex 20 belongs to tile 1).
    sizes = np.array([len(p) for p in dedup_polys])
    all_verts = np.concatenate(dedup_polys, axis=0)
    tile_ids = np.repeat(np.arange(N), sizes)

    tree = KDTree(all_verts)
    pairs = tree.query_pairs(r=tol, output_type='ndarray')
    del tree

    if len(pairs) == 0:
        return tile_centroids, [np.array([], dtype=np.int32) for _ in range(N)], tile_polygons

    # If an edge exists between nodes between two tiles, an edge exists between those tiles
    # So we turn the vertex pairs into the pairs for the tiles it links
    ti = tile_ids[pairs[:, 0]]
    tj = tile_ids[pairs[:, 1]]
    m = ti != tj
    lo = np.minimum(ti[m], tj[m])
    hi = np.maximum(ti[m], tj[m])

    # Count how many coincident vertices each tile pair shares
    # A delone graph is defined by sharing an edge
    # Since sharing only one vertex doesn't share an edge, a tile pair is a neighbour pair exactly when its count is >= 2.
    key = lo.astype(np.int64) * N + hi
    ukey, counts = np.unique(key, return_counts=True)
    edge = ukey[counts >= 2]
    elo = edge // N
    ehi = edge % N

    tile_neighbors = edges_to_adjacency(elo, ehi, N)
    return tile_centroids, tile_neighbors, tile_polygons