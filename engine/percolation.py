import numpy as np
from numba import njit
from builders.graph_core import neighbors_to_csr

# this file holds the main global percolation engine for all inputs
# It uses a Monte-Carlo Newman-Ziff algorithm
# Extrapolation and reading is handled by analysis.py

# runs the window sizes
# It recomputes each number so a floating point doesn't accumulate
def l_sweep(lmin, lmax, gap):
    if lmax <= lmin or gap <= 0:
        return []
    n = int((lmax - lmin) / gap + 1e-9) + 1
    return [float(lmin + k * gap) for k in range(n)]


# read only, checks which group it's in
@njit(cache=True)
def _find(parent, p):
    while p != parent[p]:
        parent[p] = parent[parent[p]]
        p = parent[p]
    return p

# merges a's group with b's group
@njit(cache=True)
def _union(parent, size, a, b):
    ra = _find(parent, a)
    rb = _find(parent, b)
    if ra == rb:
        return
    if size[ra] < size[rb]:
        parent[ra] = rb
        size[rb] += size[ra]
    else:
        parent[rb] = ra
        size[ra] += size[rb]

# In order to find if it reaches a corner we use two virtual nodes
# The virtual node is connected to every node on a particular border
# If the two virtual nodes are in the same component it means the cluster goes from border to border
# This is why we use (N+2)
@njit(cache=True, nogil=True)
def _site_trial(nbrs, starts, is_top, is_bot, is_left, is_right, order, N, intersection):
    # start each node as it's own parent
    # Mixing left-right clusters with top-bottom causes errors so we do two seperate union-finds
    parentTB = np.arange(N + 2)
    sizeTB = np.ones(N + 2, dtype=np.int64)
    parentLR = np.arange(N + 2)
    sizeLR = np.ones(N + 2, dtype=np.int64)
    # Opened is the list of open nodes in the percolation, none in the beginning
    opened = np.zeros(N, dtype=np.bool_)
    vTop = N; vBot = N + 1; vL = N; vR = N + 1
    tb_onset = -1; lr_onset = -1
    # order is the random permutation
    for step in range(N):
        idx = order[step]
        opened[idx] = True
        # if it's at the top, bottom left or right merge it for the respective virtual node
        if is_top[idx]:   _union(parentTB, sizeTB, vTop, idx)
        if is_bot[idx]:   _union(parentTB, sizeTB, vBot, idx)
        if is_left[idx]:  _union(parentLR, sizeLR, vL, idx)
        if is_right[idx]: _union(parentLR, sizeLR, vR, idx)
        # This is the loop for the neighbours of the node idx
        # instead of an adjacency graph we have a flat array
        # starts is the index of where the next node starts so starts[node] is where node's list starts
        # thus for a given node we range from starts[idx] to start[idx + 1]
        for j in range(starts[idx], starts[idx + 1]):
            nb = nbrs[j]
            # only merge with neighbours that are on
            # When its own turn comes later, it'll look back at idx, see it open, and do the merge from that side
            # So every edge between two open sites gets merged exactly once
            if opened[nb]:
                _union(parentTB, sizeTB, idx, nb)
                _union(parentLR, sizeLR, idx, nb)
        # this is to check if it has spanned from top to bottom or left to right
        tb = _find(parentTB, vTop) == _find(parentTB, vBot)
        lr = _find(parentLR, vL) == _find(parentLR, vR)
        # records the point of percolation
        # the inequality makes sure that once you record a step it percolates at, it doesn't override
        if tb and tb_onset < 0: tb_onset = step + 1
        if lr and lr_onset < 0: lr_onset = step + 1
        # for intersection it stops when both left and right occur
        if intersection:
            if tb and lr:
                return step + 1, tb_onset, lr_onset
        # for union, it stops when one or the other occurs
        else:
            if tb or lr:
                return step + 1, tb_onset, lr_onset
    # no percolation! This shouldn't happen
    return N, tb_onset, lr_onset


# This is the bond percolation version
@njit(cache=True, nogil=True)
def _bond_trial(eu, ev, top, bot, left, right, order, num_nodes, num_edges, intersection):
    parentTB = np.arange(num_nodes + 2)
    sizeTB = np.ones(num_nodes + 2, dtype=np.int64)
    parentLR = np.arange(num_nodes + 2)
    sizeLR = np.ones(num_nodes + 2, dtype=np.int64)
    vTop = num_nodes; vBot = num_nodes + 1; vL = num_nodes; vR = num_nodes + 1
    # Here nodes are always there, edges turn off and on
    # Thus the virtual nodes can be merged without requiring us to be in the main loop
    for k in range(top.shape[0]):    _union(parentTB, sizeTB, vTop, top[k])
    for k in range(bot.shape[0]):    _union(parentTB, sizeTB, vBot, bot[k])
    for k in range(left.shape[0]):   _union(parentLR, sizeLR, vL, left[k])
    for k in range(right.shape[0]):  _union(parentLR, sizeLR, vR, right[k])
    tb_onset = -1; lr_onset = -1
    for step in range(num_edges):
        # order is a permutation of edges
        e = order[step]
        u = eu[e]; v = ev[e]
        # each step picks the two endpoints of an edge and connects them
        _union(parentTB, sizeTB, u, v)
        _union(parentLR, sizeLR, u, v)
        tb = _find(parentTB, vTop) == _find(parentTB, vBot)
        lr = _find(parentLR, vL) == _find(parentLR, vR)
        # same as we had in the site percolation version to find if there's directional bias
        if tb and tb_onset < 0: tb_onset = step + 1
        if lr and lr_onset < 0: lr_onset = step + 1
        # same branching as site percolation for intersection and union
        if intersection:
            if tb and lr:
                return step + 1, tb_onset, lr_onset
        else:
            if tb or lr:
                return step + 1, tb_onset, lr_onset
    return num_edges, tb_onset, lr_onset


# Plumbing
# Since python reads a list element by element, it'd be very inefficient
# In older versions it used to be a set, but we need numba to read it
# thus we turn it into an array
def _masks(N, top, bot, left, right):
    it = np.zeros(N, np.bool_); it[list(top)] = True
    ib = np.zeros(N, np.bool_); ib[list(bot)] = True
    il = np.zeros(N, np.bool_); il[list(left)] = True
    ir = np.zeros(N, np.bool_); ir[list(right)] = True
    return it, ib, il, ir

# ---- parallel (nogil numba kernel + threads; per-trial reproducible seeding) ----
from concurrent.futures import ThreadPoolExecutor
import os
# sets the default core usage to cpu - 1, leaving 1 for other tasks 
# otherwise use "PERCOLATION THREADS" amount
_NW = int(os.environ.get("PERCOLATE_THREADS", "0")) or max(1, (os.cpu_count() or 4) - 1)

#########################

# # Runs once before the trial turning the graph into an array
class percolationStatsI_par:
    def __init__(self, nodes, neighbours, top, bot, left, right, trials, master_seed=0, nworkers=_NW):
        # flattens into a csr graph
        N = len(nodes)
        nbrs, starts = neighbors_to_csr(neighbours)
        nbrs = nbrs.astype(np.int64); starts = starts.astype(np.int64)
        it, ib, il, ir = _masks(N, top, bot, left, right)
        seeds = np.random.SeedSequence(master_seed).spawn(trials)

        def one(k):
            # for each trial generate a new permutation of N indices
            rng = np.random.default_rng(seeds[k])
            order = rng.permutation(N).astype(np.int64)
            # this calls the actual percolation function above
            # site_trial's final parameter is 'intersection' so true means intersection
            return _site_trial(nbrs, starts, it, ib, il, ir, order, N, True)

        with ThreadPoolExecutor(max_workers=nworkers) as ex:
            res = list(ex.map(one, range(trials)))
        # calculates the results to return
        self.trialResults = [o / N for o, _, _ in res]
        self.pD = [tb / N for _, tb, _ in res]
        self.pR = [lr / N for _, _, lr in res]

class percolationStatsU_par:
    def __init__(self, nodes, neighbours, top, bot, left, right, trials, master_seed=0, nworkers=_NW):
        N = len(nodes)
        nbrs, starts = neighbors_to_csr(neighbours)
        nbrs = nbrs.astype(np.int64); starts = starts.astype(np.int64)
        it, ib, il, ir = _masks(N, top, bot, left, right)
        seeds = np.random.SeedSequence(master_seed).spawn(trials)

        def one(k):
            rng = np.random.default_rng(seeds[k])
            order = rng.permutation(N).astype(np.int64)
            onset, _, _ = _site_trial(nbrs, starts, it, ib, il, ir, order, N, False)
            # Since union stops as soon as EITHER direction spans, the other onset is still unset thus the union doesn't store left-right/top-bottom
            # this information stil exists due to the intersection for the same run though
            return onset

        with ThreadPoolExecutor(max_workers=nworkers) as ex:
            res = list(ex.map(one, range(trials)))
        self.trialResults = [o / N for o in res]


class percolationStatsBondI_par:
    def __init__(self, nodes, edges, top, bot, left, right, trials,
                 master_seed=0, nworkers=_NW, intersection=True):
        num_nodes = len(nodes)
        edges = np.asarray(edges)
        # This doesn't need the CSR format, so it splits into x and y components, which is what the trial needs
        eu = edges[:, 0].astype(np.int64); ev = edges[:, 1].astype(np.int64)
        # permutation over edges
        M = len(edges)
        # this can connect all the boundary nodes before running since the nodes aren't what turn on or off
        top = np.asarray(list(top), dtype=np.int64);   bot = np.asarray(list(bot), dtype=np.int64)
        left = np.asarray(list(left), dtype=np.int64); right = np.asarray(list(right), dtype=np.int64)
        seeds = np.random.SeedSequence(master_seed).spawn(trials)
        inter = intersection

        def one(k):
            rng = np.random.default_rng(seeds[k])
            order = rng.permutation(M).astype(np.int64)
            return _bond_trial(eu, ev, top, bot, left, right, order, num_nodes, M, inter)

        with ThreadPoolExecutor(max_workers=nworkers) as ex:
            res = list(ex.map(one, range(trials)))
        self.trialResults = [o / M for o, _, _ in res]
        self.pD = [tb / M for _, tb, _ in res]
        self.pR = [lr / M for _, _, lr in res]

# Beauty of object oriented programming
# Everything is already defined so it is just the above with intersection=False
class percolationStatsBondU_par(percolationStatsBondI_par):
    def __init__(self, nodes, edges, top, bot, left, right, trials, master_seed=0, nworkers=_NW):
        super().__init__(nodes, edges, top, bot, left, right, trials,
                         master_seed=master_seed, nworkers=nworkers, intersection=False)


# Finds the largest cluster at percolation using a third union-find
# This is done to find the universality class of the percolation
@njit(cache=True, nogil=True)
def _site_cluster_trial(nbrs, starts, is_top, is_bot, is_left, is_right, order, N):
    parentTB = np.arange(N + 2); sizeTB = np.ones(N + 2, dtype=np.int64)
    parentLR = np.arange(N + 2); sizeLR = np.ones(N + 2, dtype=np.int64)
    parentC  = np.arange(N);     sizeC  = np.ones(N,     dtype=np.int64)
    opened = np.zeros(N, dtype=np.bool_)
    vTop = N; vBot = N + 1; vL = N; vR = N + 1
    onset_u = -1; onset_i = -1
    # s_union is the largest cluster at the union percolation and s_intersection at intersection
    # Since they share an exponent, these should be close together
    s_max = 0; s_union = 0; s_inter = 0
    for step in range(N):
        idx = order[step]
        # it checks the cluster size for the node that just opened
        opened[idx] = True
        if is_top[idx]:   _union(parentTB, sizeTB, vTop, idx)
        if is_bot[idx]:   _union(parentTB, sizeTB, vBot, idx)
        if is_left[idx]:  _union(parentLR, sizeLR, vL, idx)
        if is_right[idx]: _union(parentLR, sizeLR, vR, idx)
        for j in range(starts[idx], starts[idx + 1]):
            nb = nbrs[j]
            if opened[nb]:
                _union(parentTB, sizeTB, idx, nb)
                _union(parentLR, sizeLR, idx, nb)
                _union(parentC,  sizeC,  idx, nb)
        # _find(parentC, idx) gives the root of the cluster that node idx now belongs to
        # sizeC[root] is that cluster's size, maintained for free by _union, which already adds sizes when it merges
        # Every other cluster is untouched this step
        s = sizeC[_find(parentC, idx)]
        # if this was bigger than the previous best, replace the previous best
        if s > s_max:
            s_max = s
        tb = _find(parentTB, vTop) == _find(parentTB, vBot)
        lr = _find(parentLR, vL) == _find(parentLR, vR)
        # union onset: first either-direction span, record it but keep growing to the intersection onset
        if onset_u < 0 and (tb or lr):
            onset_u = step + 1
            s_union = s_max
        # intersection onset: first both-direction span so we stop
        if tb and lr:
            onset_i = step + 1
            s_inter = s_max
            break
    # degenerate patch that never spanned both ways (does not occur for a valid window) 
    if onset_u < 0: s_union = s_max
    if onset_i < 0: s_inter = s_max
    return onset_u, onset_i, s_union, s_inter

# Since this costs a full extra pass and is optional, we put this in a seperate wrapper to the standard stats call
# It is very similar though
class percolationStatsExponents_par:
    def __init__(self, nodes, neighbours, top, bot, left, right, trials, master_seed=0, nworkers=_NW):
        N = len(nodes)
        nbrs, starts = neighbors_to_csr(neighbours)
        nbrs = nbrs.astype(np.int64); starts = starts.astype(np.int64)
        it, ib, il, ir = _masks(N, top, bot, left, right)
        seeds = np.random.SeedSequence(master_seed).spawn(trials)
        def one(k):
            rng = np.random.default_rng(seeds[k])
            order = rng.permutation(N).astype(np.int64)
            return _site_cluster_trial(nbrs, starts, it, ib, il, ir, order, N)

        with ThreadPoolExecutor(max_workers=nworkers) as ex:
            res = list(ex.map(one, range(trials)))
        self.N = N
        # returns the functions at percolation relevent to df
        self.s_union = np.array([r[2] for r in res], dtype=float)
        self.s_inter = np.array([r[3] for r in res], dtype=float)
        self.s_max = self.s_union


# Bond version of the largest-cluster pass: same largest-cluster-at-onset readout, but edges open one at a time
# Cluster mass is counted in SITES (node count), like the site pass, so the bond d_f is directly comparable to the site d_f
# d_f is universal, so this should reproduce the  same value as the site
@njit(cache=True, nogil=True)
def _bond_cluster_trial(eu, ev, top, bot, left, right, order, num_nodes, num_edges):
    parentTB = np.arange(num_nodes + 2); sizeTB = np.ones(num_nodes + 2, dtype=np.int64)
    parentLR = np.arange(num_nodes + 2); sizeLR = np.ones(num_nodes + 2, dtype=np.int64)
    parentC  = np.arange(num_nodes);     sizeC  = np.ones(num_nodes,     dtype=np.int64)
    vTop = num_nodes; vBot = num_nodes + 1; vL = num_nodes; vR = num_nodes + 1
    # nodes are permanent in bond percolation, so wire the boundaries to the virtual nodes up front
    for k in range(top.shape[0]):    _union(parentTB, sizeTB, vTop, top[k])
    for k in range(bot.shape[0]):    _union(parentTB, sizeTB, vBot, bot[k])
    for k in range(left.shape[0]):   _union(parentLR, sizeLR, vL, left[k])
    for k in range(right.shape[0]):  _union(parentLR, sizeLR, vR, right[k])
    onset_u = -1; onset_i = -1
    s_max = 0; s_union = 0; s_inter = 0
    for step in range(num_edges):
        e = order[step]
        u = eu[e]; v = ev[e]
        _union(parentTB, sizeTB, u, v)
        _union(parentLR, sizeLR, u, v)
        _union(parentC,  sizeC,  u, v)          # cluster mass over real nodes only
        s = sizeC[_find(parentC, u)]            # the just-merged cluster contains u
        if s > s_max:
            s_max = s
        tb = _find(parentTB, vTop) == _find(parentTB, vBot)
        lr = _find(parentLR, vL) == _find(parentLR, vR)
        if onset_u < 0 and (tb or lr):
            onset_u = step + 1
            s_union = s_max
        if tb and lr:
            onset_i = step + 1
            s_inter = s_max
            break
    if onset_u < 0: s_union = s_max
    if onset_i < 0: s_inter = s_max
    return onset_u, onset_i, s_union, s_inter

# Same logic
class percolationStatsBondExponents_par:
    def __init__(self, nodes, edges, top, bot, left, right, trials, master_seed=0, nworkers=_NW):
        num_nodes = len(nodes)
        edges = np.asarray(edges)
        eu = edges[:, 0].astype(np.int64); ev = edges[:, 1].astype(np.int64)
        M = len(edges)
        top = np.asarray(list(top), dtype=np.int64);   bot = np.asarray(list(bot), dtype=np.int64)
        left = np.asarray(list(left), dtype=np.int64); right = np.asarray(list(right), dtype=np.int64)
        seeds = np.random.SeedSequence(master_seed).spawn(trials)

        def one(k):
            rng = np.random.default_rng(seeds[k])
            order = rng.permutation(M).astype(np.int64)
            return _bond_cluster_trial(eu, ev, top, bot, left, right, order, num_nodes, M)

        with ThreadPoolExecutor(max_workers=nworkers) as ex:
            res = list(ex.map(one, range(trials)))
        self.N = num_nodes
        self.s_union = np.array([r[2] for r in res], dtype=float)
        self.s_inter = np.array([r[3] for r in res], dtype=float)
        self.s_max = self.s_union


