"""The percolation engine (Newman-Ziff incremental union-find, weighted quick-union with
path halving). The per-trial sweep is numba-JIT-compiled and the graph invariants (CSR
neighbours, boundary masks) are built once per size, not per trial.

Trials run in parallel: the percolationStats*_par classes run the nogil kernel across a thread
pool and draw an independent SeedSequence stream per trial, so a run is reproducible from its
master_seed (though NOT bit-identical to a single serial random.shuffle stream). Every runner —
the production hat/spectre ones and the periodic/comparison ones — uses these classes.

The results-analysis functions (WLS fit, p_c extrapolation, direction-bias check) live in the
sibling module engine/analysis.py; this module is kept pure simulation.

(A pure-Python reference implementation of the engine, and a serial numba twin of these classes,
were kept during development for bit-identity checks; they now live on the `author-review` git
branch.)"""
import numpy as np
from numba import njit
from builders.direct_graph_builder import neighbors_to_csr


def l_sweep(lmin, lmax, gap):
    """The window sizes L for a sweep: count-based (int number of steps, then L_min + k*gap) so it
    never drifts by a float epsilon. The SINGLE definition of the sweep -- the GUI preview/ETA and
    the runner both call this, so the estimate and the actual run use IDENTICAL L (which change the
    result). Returns [] for an empty/invalid range."""
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


# ┌This is the bond percolation version
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
        # record the first step at which each direction spans (mirrors _site_trial), for the
        # direction-bias check. This bookkeeping does NOT alter the returned onset o below.
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

# ---- the trial classes: nogil numba kernel run across a thread pool; per-trial reproducible seeding ----
from concurrent.futures import ThreadPoolExecutor
import os
# Threads for the trial pool. Default = cpu-1. Set PERCOLATE_THREADS to leave cores free -- e.g. so
# the GUI stays responsive while a long background run saturates the machine.
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
        # calculates the results to return. pD (top-bottom) and pR (left-right) are the
        # per-direction crossing fractions, kept for the direction-bias check that validates
        # the direction-averaged p_A estimator (see isotropy_test).
        self.trialResults = [o / N for o, _, _ in res]
        self.pD = [tb / N for _, tb, _ in res]
        self.pR = [lr / N for _, _, lr in res]

# the rest are the same as the above

class percolationStatsU_par:
    """Site UNION criterion, thread-parallel. Seeded via SeedSequence(master_seed)
    so results are reproducible (not bit-identical to the serial random.shuffle path)."""
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
            return onset

        with ThreadPoolExecutor(max_workers=nworkers) as ex:
            res = list(ex.map(one, range(trials)))
        self.trialResults = [o / N for o in res]


class percolationStatsBondI_par:
    """Bond INTERSECTION criterion, thread-parallel. Set intersection=False for UNION."""
    def __init__(self, nodes, edges, top, bot, left, right, trials,
                 master_seed=0, nworkers=_NW, intersection=True):
        num_nodes = len(nodes)
        edges = np.asarray(edges)
        eu = edges[:, 0].astype(np.int64); ev = edges[:, 1].astype(np.int64)
        M = len(edges)
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
        # o = spanning onset (unchanged); pD/pR = per-direction first-span fractions, used by the
        # direction-bias check that validates the bond p_A estimator (only the intersection run's
        # pD/pR are consumed; for the union run one of them may be -1 and is ignored).
        self.trialResults = [o / M for o, _, _ in res]
        self.pD = [tb / M for _, tb, _ in res]
        self.pR = [lr / M for _, _, lr in res]


class percolationStatsBondU_par(percolationStatsBondI_par):
    def __init__(self, nodes, edges, top, bot, left, right, trials, master_seed=0, nworkers=_NW):
        super().__init__(nodes, edges, top, bot, left, right, trials,
                         master_seed=master_seed, nworkers=nworkers, intersection=False)


# ---- Block B: cluster-structure readout for the critical exponents (d_f, gamma/nu, tau) ----
# The two spanning union-finds above carry VIRTUAL boundary nodes, which glue every boundary site
# into one artificial mega-cluster -- so their sizes are fiction. Here we maintain a THIRD, PLAIN
# union-find over real sites only (no virtual nodes) and track the largest real cluster INCREMENTALLY
# as sites merge; at the first-spanning onset (the self-consistent pseudo-critical point -- NOT a
# fixed external p_c) that largest cluster is the incipient infinite cluster, whose size s_max ~
# L^{d_f} gives the fractal dimension. We deliberately record ONLY s_max (d_f): the other static
# exponents (gamma/nu, tau, beta/nu) follow from d_f by hyperscaling, and their DIRECT cluster-moment
# estimators are open-boundary biased, so we don't measure them. Everything is combinatorial (site
# counts), no coordinates -- consistent with the graph-only model.
@njit(cache=True, nogil=True)
def _site_cluster_trial(nbrs, starts, is_top, is_bot, is_left, is_right, order, N):
    parentTB = np.arange(N + 2); sizeTB = np.ones(N + 2, dtype=np.int64)   # spanning detection (TB)
    parentLR = np.arange(N + 2); sizeLR = np.ones(N + 2, dtype=np.int64)   # spanning detection (LR)
    parentC  = np.arange(N);     sizeC  = np.ones(N,     dtype=np.int64)   # PLAIN: real sites only
    opened = np.zeros(N, dtype=np.bool_)
    vTop = N; vBot = N + 1; vL = N; vR = N + 1
    onset = -1; s_max = 0
    for step in range(N):
        idx = order[step]
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
                _union(parentC,  sizeC,  idx, nb)     # real-real merges only
        s = sizeC[_find(parentC, idx)]                 # size of the cluster idx now sits in
        if s > s_max:                                  # running max -> largest cluster so far (O(1))
            s_max = s
        tb = _find(parentTB, vTop) == _find(parentTB, vBot)
        lr = _find(parentLR, vL) == _find(parentLR, vR)
        if tb or lr:                                   # UNION onset = first spanning (incipient cluster)
            onset = step + 1
            break
    return onset, s_max


class percolationStatsExponents_par:
    """OPT-IN Block-B pass: at the first-spanning onset, read the largest real-sites cluster (the
    incipient infinite cluster) -> s_max per trial -> d_f (<s_max> ~ L^{d_f}). NOT run by the
    threshold sweep -- a deliberate extra pass, requested only when d_f is wanted. Records s_max ONLY
    (see the note above the kernel: gamma/nu and tau follow from d_f by hyperscaling and their direct
    estimators are open-boundary biased, so we do not measure them)."""
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
        self.s_max = np.array([r[1] for r in res], dtype=float)


# Analysis of results (WLS p_c extrapolation + direction-bias check) now lives in engine/analysis.py
# so this module stays pure simulation (Newman-Ziff kernels + trial classes).