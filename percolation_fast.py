"""numba-accelerated percolation trials. Same algorithm as percolation.py
(weighted quick-union with path halving), but the per-site sweep is JIT-compiled
and the graph invariants (CSR, boundary masks) are built once per size, not per trial.
Permutations are generated with random.shuffle exactly as the baseline, so results
are bit-identical to percolation_baseline given the same seed."""
import numpy as np
import random
from numba import njit
from hat_graph_builder import neighbors_to_csr


@njit(cache=True)
def _find(parent, p):
    while p != parent[p]:
        parent[p] = parent[parent[p]]
        p = parent[p]
    return p


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


@njit(cache=True, nogil=True)
def _site_trial(nbrs, starts, is_top, is_bot, is_left, is_right, order, N, intersection):
    parentTB = np.arange(N + 2)
    sizeTB = np.ones(N + 2, dtype=np.int64)
    parentLR = np.arange(N + 2)
    sizeLR = np.ones(N + 2, dtype=np.int64)
    opened = np.zeros(N, dtype=np.bool_)
    vTop = N; vBot = N + 1; vL = N; vR = N + 1
    tb_onset = -1; lr_onset = -1
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
        tb = _find(parentTB, vTop) == _find(parentTB, vBot)
        lr = _find(parentLR, vL) == _find(parentLR, vR)
        if tb and tb_onset < 0: tb_onset = step + 1
        if lr and lr_onset < 0: lr_onset = step + 1
        if intersection:
            if tb and lr:
                return step + 1, tb_onset, lr_onset
        else:
            if tb or lr:
                return step + 1, tb_onset, lr_onset
    return N, tb_onset, lr_onset


@njit(cache=True, nogil=True)
def _bond_trial(eu, ev, top, bot, left, right, order, num_nodes, num_edges, intersection):
    parentTB = np.arange(num_nodes + 2)
    sizeTB = np.ones(num_nodes + 2, dtype=np.int64)
    parentLR = np.arange(num_nodes + 2)
    sizeLR = np.ones(num_nodes + 2, dtype=np.int64)
    vTop = num_nodes; vBot = num_nodes + 1; vL = num_nodes; vR = num_nodes + 1
    for k in range(top.shape[0]):    _union(parentTB, sizeTB, vTop, top[k])
    for k in range(bot.shape[0]):    _union(parentTB, sizeTB, vBot, bot[k])
    for k in range(left.shape[0]):   _union(parentLR, sizeLR, vL, left[k])
    for k in range(right.shape[0]):  _union(parentLR, sizeLR, vR, right[k])
    for step in range(num_edges):
        e = order[step]
        u = eu[e]; v = ev[e]
        _union(parentTB, sizeTB, u, v)
        _union(parentLR, sizeLR, u, v)
        tb = _find(parentTB, vTop) == _find(parentTB, vBot)
        lr = _find(parentLR, vL) == _find(parentLR, vR)
        if intersection:
            if tb and lr:
                return step + 1
        else:
            if tb or lr:
                return step + 1
    return num_edges


def _masks(N, top, bot, left, right):
    it = np.zeros(N, np.bool_); it[list(top)] = True
    ib = np.zeros(N, np.bool_); ib[list(bot)] = True
    il = np.zeros(N, np.bool_); il[list(left)] = True
    ir = np.zeros(N, np.bool_); ir[list(right)] = True
    return it, ib, il, ir


class percolationStatsI:
    def __init__(self, nodes, neighbours, top, bot, left, right, trials):
        N = len(nodes)
        nbrs, starts = neighbors_to_csr(neighbours)
        nbrs = nbrs.astype(np.int64); starts = starts.astype(np.int64)
        it, ib, il, ir = _masks(N, top, bot, left, right)
        self.trialResults = []; self.pR = []; self.pD = []
        for _ in range(trials):
            o = list(range(N)); random.shuffle(o)
            order = np.array(o, dtype=np.int64)
            onset, tb, lr = _site_trial(nbrs, starts, it, ib, il, ir, order, N, True)
            self.trialResults.append(onset / N)
            self.pD.append(tb / N)
            self.pR.append(lr / N)

    def trials_mean(self): return np.mean(self.trialResults)
    def trials_std(self): return np.std(self.trialResults)


class percolationStatsU(percolationStatsI):
    def __init__(self, nodes, neighbours, top, bot, left, right, trials):
        N = len(nodes)
        nbrs, starts = neighbors_to_csr(neighbours)
        nbrs = nbrs.astype(np.int64); starts = starts.astype(np.int64)
        it, ib, il, ir = _masks(N, top, bot, left, right)
        self.trialResults = []
        for _ in range(trials):
            o = list(range(N)); random.shuffle(o)
            order = np.array(o, dtype=np.int64)
            onset, _, _ = _site_trial(nbrs, starts, it, ib, il, ir, order, N, False)
            self.trialResults.append(onset / N)


class percolationStatsBondI:
    def __init__(self, nodes, edges, top, bot, left, right, trials):
        num_nodes = len(nodes)
        edges = np.asarray(edges)
        eu = edges[:, 0].astype(np.int64); ev = edges[:, 1].astype(np.int64)
        M = len(edges)
        top = np.asarray(list(top), dtype=np.int64); bot = np.asarray(list(bot), dtype=np.int64)
        left = np.asarray(list(left), dtype=np.int64); right = np.asarray(list(right), dtype=np.int64)
        self.trialResults = []
        for _ in range(trials):
            o = list(range(M)); random.shuffle(o)
            order = np.array(o, dtype=np.int64)
            onset = _bond_trial(eu, ev, top, bot, left, right, order, num_nodes, M, True)
            self.trialResults.append(onset / M)

    def trials_mean(self): return np.mean(self.trialResults)
    def trials_std(self): return np.std(self.trialResults)


class percolationStatsBondU(percolationStatsBondI):
    def __init__(self, nodes, edges, top, bot, left, right, trials):
        num_nodes = len(nodes)
        edges = np.asarray(edges)
        eu = edges[:, 0].astype(np.int64); ev = edges[:, 1].astype(np.int64)
        M = len(edges)
        top = np.asarray(list(top), dtype=np.int64); bot = np.asarray(list(bot), dtype=np.int64)
        left = np.asarray(list(left), dtype=np.int64); right = np.asarray(list(right), dtype=np.int64)
        self.trialResults = []
        for _ in range(trials):
            o = list(range(M)); random.shuffle(o)
            order = np.array(o, dtype=np.int64)
            onset = _bond_trial(eu, ev, top, bot, left, right, order, num_nodes, M, False)
            self.trialResults.append(onset / M)


# ---- parallel (nogil numba kernel + threads; per-trial reproducible seeding) ----
from concurrent.futures import ThreadPoolExecutor
import os
_NW = max(1, (os.cpu_count() or 4) - 1)


class percolationStatsI_par:
    def __init__(self, nodes, neighbours, top, bot, left, right, trials, master_seed=0, nworkers=_NW):
        N = len(nodes)
        nbrs, starts = neighbors_to_csr(neighbours)
        nbrs = nbrs.astype(np.int64); starts = starts.astype(np.int64)
        it, ib, il, ir = _masks(N, top, bot, left, right)
        seeds = np.random.SeedSequence(master_seed).spawn(trials)

        def one(k):
            rng = np.random.default_rng(seeds[k])
            order = rng.permutation(N).astype(np.int64)
            return _site_trial(nbrs, starts, it, ib, il, ir, order, N, True)

        with ThreadPoolExecutor(max_workers=nworkers) as ex:
            res = list(ex.map(one, range(trials)))
        self.trialResults = [o / N for o, _, _ in res]
        self.pD = [tb / N for _, tb, _ in res]
        self.pR = [lr / N for _, _, lr in res]

    def trials_mean(self): return np.mean(self.trialResults)
    def trials_std(self): return np.std(self.trialResults)


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

    def trials_mean(self): return np.mean(self.trialResults)
    def trials_std(self): return np.std(self.trialResults)


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
        self.trialResults = [o / M for o in res]

    def trials_mean(self): return np.mean(self.trialResults)
    def trials_std(self): return np.std(self.trialResults)


class percolationStatsBondU_par(percolationStatsBondI_par):
    def __init__(self, nodes, edges, top, bot, left, right, trials, master_seed=0, nworkers=_NW):
        super().__init__(nodes, edges, top, bot, left, right, trials,
                         master_seed=master_seed, nworkers=nworkers, intersection=False)
