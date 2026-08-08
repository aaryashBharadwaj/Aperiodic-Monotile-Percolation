"""The percolation engine (Newman-Ziff incremental union-find, weighted quick-union with
path halving). The per-trial sweep is numba-JIT-compiled and the graph invariants (CSR
neighbours, boundary masks) are built once per size, not per trial.

Trials run in parallel: the percolationStats*_par classes run the nogil kernel across a thread
pool and draw an independent SeedSequence stream per trial, so a run is reproducible from its
master_seed (though NOT bit-identical to a single serial random.shuffle stream). Every runner —
the production hat/spectre ones and the periodic/comparison ones — uses these classes.

The results-analysis functions (_wls_fit, extrapolate_pc_raw, isotropy_test)
also live here, so a runner needs only `from percolation import ...`.

(A pure-Python reference implementation of the engine, and a serial numba twin of these classes,
were kept during development for bit-identity checks; they now live on the `author-review` git
branch.)"""
import numpy as np
from numba import njit
from scipy import stats
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


##################################### Analysis of Results ##################################

# run-time extrapolation so results are saved to .npz, not just plotted post-hoc

# wls is 'weighted least squares'
def _wls_fit(x, y, sigma, n, confidence):
    # the weight takes the standard error and weighs each result inversely proportional to that
    # a point with half the error bar counts four times as much
    # thus the larger the system, the more confident the result
    w = 1.0 / sigma**2
    W = np.diag(w)
    # Design matrix: column of 1s (the intercept, pc) and column of x (the slope, A).
    X = np.column_stack([np.ones_like(x), x])


    # Normal equations for weighted least squares: solve (X^T W X) beta = X^T W y.
    # For Gaussian errors of known variance this is also the maximum-likelihood line.
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    coeffs = np.linalg.solve(XtWX, XtWy)

    # We do NOT rescale it by reduced chi-squared: with ~1000 trials per point the sigmas are well determined, so treating them as known is the honest choice.
    cov = np.linalg.inv(XtWX)
    pc_hat = coeffs[0]
    A_hat  = coeffs[1]
    pc_std = np.sqrt(cov[0, 0])
    A_std  = np.sqrt(cov[1, 1])

    # Use a t-distribution
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 2)
    pc_ci  = (pc_hat - t_crit * pc_std, pc_hat + t_crit * pc_std)

    return pc_hat, A_hat, pc_std, A_std, pc_ci


def extrapolate_pc_raw(L_list, trials_results_I, trials_results_U, nu=4/3, confidence=0.95):
    L  = np.array(L_list, dtype=float)
    n  = len(L)
    # The finite estimates approach the true p_c using a critical exponent nu = 4/3, so we use this to estimate p_c
    x  = L ** (-1.0 / nu)

    mean_I, mean_U, mean_A = [], [], []
    sigma_I, sigma_U, sigma_A = [], [], []

    for rI, rU in zip(trials_results_I, trials_results_U):
        rI, rU = np.array(rI), np.array(rU)
        T = len(rI)

        mI, mU = rI.mean(), rU.mean()
        sI, sU = rI.std(ddof=1), rU.std(ddof=1)
        # I and U come from the SAME sweeps, so they are correlated.
        cov_IU = np.cov(rI, rU)[0, 1]

        # stores the union, intersection and average estimators
        mean_I.append(mI)
        mean_U.append(mU)
        mean_A.append(0.5 * (mI + mU))

        sigma_I.append(sI / np.sqrt(T))
        sigma_U.append(sU / np.sqrt(T))
        # Var(1/2 (I+U)) = 1/4 (Var I + Var U + 2 Cov(I,U)). The covariance term is not optional: 
        # I and U are measured in the same sweep and are strongly positively correlated, so dropping it would understate sigma_A substantially
        # For independent directions, cov ~ 0 and this reduces to the usual formula
        sigma_A.append(0.5 * np.sqrt(sI**2/T + sU**2/T + 2*cov_IU/T))

    results = {}
    for label, y, sigma in [
        ('I', np.array(mean_I),  np.array(sigma_I)),
        ('U', np.array(mean_U),  np.array(sigma_U)),
        ('A', np.array(mean_A),  np.array(sigma_A)),
    ]:
        pc_hat, A_hat, pc_std, A_std, pc_ci = _wls_fit(x, y, sigma, n, confidence)
        results[label] = {
            # A is the fitted slope (scaling amplitude); reported for completeness.
            'pc'    : pc_hat,
            'A'     : A_hat,
            'pc_std': pc_std,
            'pc_ci' : pc_ci,
            'A_std' : A_std,
        }
        print(f"[{label}] Extrapolated p_c     : {pc_hat:.6f} ± {pc_std:.6f}")
        print(f"[{label}] Scaling amplitude A  : {A_hat:.6f} ± {A_std:.6f}")
        print(f"[{label}] {int(confidence*100)}% CI             : [{pc_ci[0]:.6f}, {pc_ci[1]:.6f}]")
        print()

    return results

# Direction-bias check that validates the estimator, NOT a universality claim.
# We measure p_c by crossing a SQUARE window, and "left-right" vs "top-bottom" (and the square
# aspect ratio) are arbitrary conventions. If P(LR) != P(TB) the reported threshold would depend
# on that arbitrary choice, and the direction-averaged estimator p_A would be blending two
# different quantities. Testing p_R - p_D -> 0 confirms the choice does not bias p_c. (This is a
# statement about the measurement we ran, not about the geometry/universality of the tiling.)
def isotropy_test(L_list, pR, pD, nu=4.0/3.0, confidence=0.95, L_min=50):
    # p_R (horizontal) and p_D (vertical) crossing fractions are recorded in the same sweep
    # (paired). We extrapolate their difference to L -> infinity; consistency with zero means
    # the reported p_c is independent of the arbitrary spanning-direction and square choice,
    # which justifies the averaged estimator p_A.
    L = np.asarray(L_list, dtype=float)
    pR = [np.asarray(r, dtype=float) for r in pR]
    pD = [np.asarray(r, dtype=float) for r in pD]
    if L_min is not None:
        keep = L >= L_min
        L = L[keep]
        pR = [r for r, k in zip(pR, keep) if k]
        pD = [r for r, k in zip(pD, keep) if k]
    n = len(L)
    x = L ** (-1.0 / nu)
    d_mean = np.array([(rR - rD).mean() for rR, rD in zip(pR, pD)])
    d_se   = np.array([(rR - rD).std(ddof=1) / np.sqrt(len(rR)) for rR, rD in zip(pR, pD)])
    d_inf, _, _, _, d_ci = _wls_fit(x, d_mean, d_se, n, confidence)
    unbiased = d_ci[0] <= 0.0 <= d_ci[1]
    print(f"[Direction bias] p_R - p_D (L->inf) = {d_inf:+.6f}   {int(confidence*100)}% CI "
          f"[{d_ci[0]:+.6f}, {d_ci[1]:+.6f}]  ->  {'no directional bias' if unbiased else 'DIRECTION-BIASED'}")
    return d_inf, d_ci, unbiased