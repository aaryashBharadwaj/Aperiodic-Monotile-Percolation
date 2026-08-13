import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from generators.hat_generator import mul, transPt
from builders.graph_core import edges_to_adjacency

# The same vertex can sometimes be generated multiple times via different tiles
# This takes vertices that are a chosen 'tol' apart and makes sure there's only one
# This isn't just bug fixing, it's what enables the merging by taking independent tiles and removing their seperation
def dedup_vertices(raw, tol=1e-5):
    V = len(raw)
    tree = KDTree(raw)
    # Pairs is an array of vertex pairs that are close enough to be floating point equivalents
    # The threshold of closeness is tol
    # This creates a table 'unique_nodes' and labels a lookup
    pairs = tree.query_pairs(r=tol, output_type='ndarray')
    del tree
    if len(pairs):
        # g is pairs as a sparse adjcacency matrix since that is scipy's representation
        g = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
                       shape=(V, V))
        # Directed is false since an edge is bidirectional
        n_unique, labels = connected_components(g, directed=False)
        del g
    # No two vertices are the same and you don't have to worry
    else:
        n_unique, labels = V, np.arange(V)
    labels = labels.astype(np.int64)
    unique_nodes = np.empty((n_unique, 2), dtype=np.float64)
    # The last vertex written to a slot becomes the coordinates for it
    # This preserves the coordinates for each node
    # unique_nodes[id] gives you the coordinates for a given node ID
    unique_nodes[labels] = raw
    return unique_nodes, labels

# Takes a raw array of coordinates and generates the graph out of them
def graph_from_raw(raw, poly_sizes, tol=1e-5):
    # converts raw into an np array if it isn't one
    raw = np.asarray(raw)
    # build_neighbor_graph_fast is a python array, this changes it
    poly_sizes = np.asarray(poly_sizes, dtype=np.int64)
    # the length of raw
    V = len(raw)

    # applies the dedup
    unique_nodes, labels = dedup_vertices(raw, tol)
    n_unique = len(unique_nodes)

    # make an array whose size is the number of polygons + 1 and initialise to 0
    offsets = np.empty(len(poly_sizes) + 1, dtype=np.int64)
    offsets[0] = 0
    # Create a cumulative sum of the sizes of each polygon and put it in offsets
    # What this gives you is the index of each new polygon in raw
    np.cumsum(poly_sizes, out=offsets[1:])
    # what we have is a source and destination arrays (src[i], dst[i]) represents an edge src[i]-dst[i]
    # we create the source list, which is just V numbers in order
    src = np.arange(V, dtype=np.int64)
    # for now the destination list is just the same numbers increased by one
    # what this gives is a line since you have 1-2, 2-3, 3-4
    dst = src + 1
    # however at each end of a polygon you loop it back to the first vertex
    dst[offsets[1:] - 1] = offsets[:-1]
    del raw, poly_sizes
    u = labels[src]; v = labels[dst]
    del src, dst, labels
    # Each edge is counted twice from the two polygons that share it so we deduplicate
    m = u != v
    # always put the smaller ID first
    lo = np.minimum(u[m], v[m]); hi = np.maximum(u[m], v[m])
    del u, v, m
    # .unique only works on flat arrays so you flatten each pair into one
    key = np.unique(lo * np.int64(n_unique) + hi)
    lo = key // n_unique; hi = key % n_unique
    del key

    # turns the edge list into a neighbours adjacency graph
    neighbors = edges_to_adjacency(lo, hi, n_unique)
    return unique_nodes, neighbors

# Turns an array of tiles into an array of raw coordinates as the format for raw
# Keeps per-tile vertex counts so tile boundaries aren't lost
def graph_from_polygons(polys, tol=1e-5):
    polys = [np.asarray(p) for p in polys]
    poly_sizes = np.fromiter((len(p) for p in polys), dtype=np.int64, count=len(polys))
    raw = np.concatenate(polys)
    return graph_from_raw(raw, poly_sizes, tol)


# Builds a graph representation of the hat tiling by extracting nodes and edges using numpy and scipy for optimisation
# Collects raw polygon vertices of every leaf tile (in order) + each tile's vertex count
# Merge near duplicates that belong to multiple tiles
# Use deduplication algorithm to give each node a unique Id and build adjacency list (list of int32 arrays)
def build_neighbor_graph_fast(patch, level=0, tol=1e-5):
    # this replaces the predefined array size by doing the same walk as collect but without the transforms
    # Cheap enough to run first so the buffer can be sized exactly
    def _count(patch, level):
        ch = getattr(patch, "children", None)
        if ch and (level is None or level > 0):
            nxt = None if level is None else level - 1
            return sum(_count(g['geom'], nxt) for g in ch)
        return len(patch.shape)

    raw = np.empty((_count(patch, level), 2), dtype=np.float64)
    poly_sizes = []               # vertices per leaf polygon, in collection order
    # cnt is the cursor to write into raw
    # it is a list for python-specific reason but functions as a counter
    cnt = [0]

    # Takes a MetaTile 'patch' as an input
    def _collect(patch, S, level):
        ch = getattr(patch, "children", None)
        if ch and (level is None or level > 0):
            nxt = None if level is None else level - 1
            # Each childs transform is defined relative to it's parent
            # So the global transform of the child is the product of the parent's accumulated transform with the childs relative transform
            # The result is stored in S and g['T'] is the relative position. nxt is the depth of the child
            for g in ch:
                _collect(g['geom'], mul(S, g['T']), nxt)
        else:
            # Once you reach a leaf, you apply the net transformation and append the coordinates
            shp = patch.shape
            for p in shp:
                q = transPt(S, p)
                raw[cnt[0], 0] = q['x']; raw[cnt[0], 1] = q['y']; cnt[0] += 1
            poly_sizes.append(len(shp))

    _collect(patch, [1, 0, 0, 0, 1, 0], level)
    # Since V is the counter at the end, it's the number of coordinates
    V = cnt[0]
    # Trim the size of raw for efficiency after this is done
    raw = raw[:V]
    # poly_sizes is the index of  how many vertices each polygon has
    # [4, 3, 5] means poly0 has 4 vertices, poly1 has 3, poly2 has 5
    poly_sizes = np.asarray(poly_sizes, dtype=np.int64)

    # sends to graph_from_raw to build an adjecency graph
    return graph_from_raw(raw, poly_sizes, tol)


# finds the centre of the largest square that can be generated on an input patch
def largest_square_center(nodes, H=4.0):

    # split the coordinates into their x and y components
    xs, ys = nodes[:, 0], nodes[:, 1]
    # we find the minimum and maximum coordinates, which creates a bounding box
    xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()
    # chop the rectangle into squares with length 4 units
    # x_max - x_min makes the leftmost edge zero, /H converts to cell units
    # Thus anything with x between 12 and 16 gives column 3
    nx = int(np.ceil((xmax - xmin) / H)) + 1
    ny = int(np.ceil((ymax - ymin) / H)) + 1
    ix = ((xs - xmin) / H).astype(np.int64)
    iy = ((ys - ymin) / H).astype(np.int64)
    cnt = np.zeros((ny, nx), np.int64); np.add.at(cnt, (iy, ix), 1)
    # Check if the patch is sparse, if it is we need to make rebuild our cells larger
    # Otherwise it would calculate things on the interior as on the outside 
    if np.median(cnt[cnt > 0]) < 3.0:
        H = H * float(np.sqrt(12.0 / np.median(cnt[cnt > 0])))
        nx = int(np.ceil((xmax - xmin) / H)) + 1
        ny = int(np.ceil((ymax - ymin) / H)) + 1
        ix = ((xs - xmin) / H).astype(np.int64)
        iy = ((ys - ymin) / H).astype(np.int64)
        cnt = np.zeros((ny, nx), np.int64); np.add.at(cnt, (iy, ix), 1)
    thr = max(1.0, 0.5 * np.median(cnt[cnt > 0]))    # "filled" = at least half the median occupancy
    F = cnt >= thr
    dp = np.zeros_like(cnt, np.int32); best = bi = bj = 0
    # Use dynamic programming to find the largest square with every cell valid recursively
    for i in range(ny):
        for j in range(nx):
            if F[i, j]:
                dp[i, j] = 1 if (i == 0 or j == 0) else 1 + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
                if dp[i, j] > best:
                    best, bi, bj = dp[i, j], i, j
    # convert back into non-cell coordinates
    side = best * H
    cx = xmin + (bj - best / 2 + 0.5) * H
    cy = ymin + (bi - best / 2 + 0.5) * H
    return cx, cy, side