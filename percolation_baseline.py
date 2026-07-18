import numpy as np
import random
from tqdm import tqdm
from hat_graph_builder import neighbors_to_csr
from scipy import stats


# Union-Find to find if two elements are in the same spanning cluster
class WeightedQuickUnionUF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
    def find(self, p):
        while p != self.parent[p]:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        return p
    def connected(self, p, q):
        return self.find(p) == self.find(q)
    def union(self, p, q):
        rootP, rootQ = self.find(p), self.find(q)
        if rootP == rootQ: return
        if self.size[rootP] < self.size[rootQ]:
            self.parent[rootP] = rootQ
            self.size[rootQ] += self.size[rootP]
        else:
            self.parent[rootQ] = rootP
            self.size[rootP] += self.size[rootQ]

# Site Percolation
class HatPercolationI:
    def __init__(self, nodes, neighbours, top_set, bottom_set, left_set, right_set):
        self.N = len(nodes)
        self.neighbors_arr, self.neighbor_starts = neighbors_to_csr(neighbours)
        self.top_set, self.bottom_set = set(top_set), set(bottom_set)
        self.left_set, self.right_set = set(left_set), set(right_set)
        self.sites = np.zeros(self.N, dtype=bool) 
        self.wqfTB = WeightedQuickUnionUF(self.N + 2)
        self.wqfLR = WeightedQuickUnionUF(self.N + 2)
        self.vTop, self.vBot = self.N, self.N + 1
        self.vL, self.vR = self.N, self.N + 1
        self.openSite = 0
        
    def open_site(self, idx):
        # No need to check if already open since we're using pre-shuffled order
        self.sites[idx] = True
        self.openSite += 1
        if idx in self.top_set: self.wqfTB.union(self.vTop, idx)
        if idx in self.bottom_set: self.wqfTB.union(self.vBot, idx)
        if idx in self.left_set: self.wqfLR.union(self.vL, idx)
        if idx in self.right_set: self.wqfLR.union(self.vR, idx)
        start, end = self.neighbor_starts[idx], self.neighbor_starts[idx+1]
        for j_idx in range(start, end):
            neigh = self.neighbors_arr[j_idx]
            if self.sites[neigh]:
                self.wqfTB.union(idx, neigh)
                self.wqfLR.union(idx, neigh)
                
    def percolates(self):
        return self.wqfTB.connected(self.vTop, self.vBot) and self.wqfLR.connected(self.vL, self.vR)

class HatPercolationU(HatPercolationI):
    def percolates(self):
        return self.wqfTB.connected(self.vTop, self.vBot) or self.wqfLR.connected(self.vL, self.vR)

class percolationStatsI:
    def __init__(self, nodes, neighbours, top, bot, left, right, trials):
        self.trialResults = []
        self.pR, self.pD = [], []   # isotropy: each direction's first-onset fraction
        for _ in tqdm(range(trials), desc="Site I", leave=False):
            sim = HatPercolationI(nodes, neighbours, top, bot, left, right)
            # Pre-shuffle all sites once
            sites_order = list(range(sim.N))
            random.shuffle(sites_order)

            # Open sites until both directions span; note when each first spans
            tb_onset = lr_onset = None
            for site in sites_order:
                sim.open_site(site)
                tb = sim.wqfTB.connected(sim.vTop, sim.vBot)   # vertical (top-bottom)
                lr = sim.wqfLR.connected(sim.vL, sim.vR)       # horizontal (left-right)
                if tb and tb_onset is None: tb_onset = sim.openSite
                if lr and lr_onset is None: lr_onset = sim.openSite
                if tb and lr:
                    break

            self.trialResults.append(sim.openSite / sim.N)
            self.pD.append(tb_onset / sim.N)   # downward
            self.pR.append(lr_onset / sim.N)   # rightward
            
    def trials_mean(self): return np.mean(self.trialResults)
    def trials_std(self): return np.std(self.trialResults)

class percolationStatsU(percolationStatsI):
    def __init__(self, nodes, neighbours, top, bot, left, right, trials):
        self.trialResults = []
        
        for _ in tqdm(range(trials), desc="Site U", leave=False):
            sim = HatPercolationU(nodes, neighbours, top, bot, left, right)
            sites_order = list(range(sim.N))
            random.shuffle(sites_order)
            
            for site in sites_order:
                sim.open_site(site)
                if sim.percolates():
                    break
                    
            self.trialResults.append(sim.openSite / sim.N)

# Bond Percolation
class HatPercolationBondI:
    def __init__(self, nodes, edges, top_set, bottom_set, left_set, right_set):
        self.num_nodes = len(nodes)
        self.edges = edges
        self.num_edges = len(edges)
        self.wqfTB = WeightedQuickUnionUF(self.num_nodes + 2)
        self.wqfLR = WeightedQuickUnionUF(self.num_nodes + 2)
        self.vTop, self.vBot = self.num_nodes, self.num_nodes + 1
        self.vL, self.vR = self.num_nodes, self.num_nodes + 1
        self.openBonds = 0
        for n in top_set: self.wqfTB.union(self.vTop, n)
        for n in bottom_set: self.wqfTB.union(self.vBot, n)
        for n in left_set: self.wqfLR.union(self.vL, n)
        for n in right_set: self.wqfLR.union(self.vR, n)
        
    def open_bond(self, idx):
        # No check needed with pre-shuffled order
        self.openBonds += 1
        u, v = self.edges[idx]
        self.wqfTB.union(u, v)
        self.wqfLR.union(u, v)
        
    def percolates(self):
        return self.wqfTB.connected(self.vTop, self.vBot) and self.wqfLR.connected(self.vL, self.vR)

class HatPercolationBondU(HatPercolationBondI):
    def percolates(self):
        return self.wqfTB.connected(self.vTop, self.vBot) or self.wqfLR.connected(self.vL, self.vR)

class percolationStatsBondI:
    def __init__(self, nodes, edges, top, bot, left, right, trials):
        self.trialResults = []
        
        for _ in tqdm(range(trials), desc="Bond I", leave=False):
            sim = HatPercolationBondI(nodes, edges, top, bot, left, right)
            edges_order = list(range(sim.num_edges))
            random.shuffle(edges_order)
            
            for edge in edges_order:
                sim.open_bond(edge)
                if sim.percolates():
                    break
                    
            self.trialResults.append(sim.openBonds / sim.num_edges)
            
    def trials_mean(self): return np.mean(self.trialResults)
    def trials_std(self): return np.std(self.trialResults)

class percolationStatsBondU(percolationStatsBondI):
    def __init__(self, nodes, edges, top, bot, left, right, trials):
        self.trialResults = []
        
        for _ in tqdm(range(trials), desc="Bond U", leave=False):
            sim = HatPercolationBondU(nodes, edges, top, bot, left, right)
            edges_order = list(range(sim.num_edges))
            random.shuffle(edges_order)
            
            for edge in edges_order:
                sim.open_bond(edge)
                if sim.percolates():
                    break
                    
            self.trialResults.append(sim.openBonds / sim.num_edges)

##################################### Analysis of Results ##################################

# run-time extrapolation so results are saved to .npz, not just plotted post-hoc

# wls is 'weighted least squares'
def _wls_fit(x, y, sigma, n, confidence):
    # the weight takes the standard error and weighs each result proportional to that
    # the larger the system, the more confident the result
    w = 1.0 / sigma**2
    W = np.diag(w)
    X = np.column_stack([np.ones_like(x), x])

    # This is a design matrix  when you multiply it by a sample result, you get the scaling law
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    coeffs = np.linalg.solve(XtWX, XtWy)

    cov = np.linalg.inv(XtWX)
    pc_hat = coeffs[0]
    A_hat  = coeffs[1]
    pc_std = np.sqrt(cov[0, 0])
    A_std  = np.sqrt(cov[1, 1])

    # Use a t-distribution
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 2)
    pc_ci  = (pc_hat - t_crit * pc_std, pc_hat + t_crit * pc_std)

    return pc_hat, A_hat, pc_std, A_std, pc_ci

# creates the estimated percolation threshold for the finite-square using scaling law
# it does this for union, intersection and average
def extrapolate_pc_raw(L_list, trials_results_I, trials_results_U, nu=4/3, confidence=0.95):
    L  = np.array(L_list, dtype=float)
    n  = len(L)
    # Percolation approaches L ^(-3/4) near the threshold so we use this to estimate p_c
    x  = L ** (-1.0 / nu)

    mean_I, mean_U, mean_A = [], [], []
    sigma_I, sigma_U, sigma_A = [], [], []

    for rI, rU in zip(trials_results_I, trials_results_U):
        rI, rU = np.array(rI), np.array(rU)
        T = len(rI)

        mI, mU = rI.mean(), rU.mean()
        sI, sU = rI.std(ddof=1), rU.std(ddof=1)
        cov_IU = np.cov(rI, rU)[0, 1]

        mean_I.append(mI)
        mean_U.append(mU)
        mean_A.append(0.5 * (mI + mU))

        sigma_I.append(sI / np.sqrt(T))
        sigma_U.append(sU / np.sqrt(T))
        sigma_A.append(0.5 * np.sqrt(sI**2/T + sU**2/T + 2*cov_IU/T))

    results = {}
    for label, y, sigma in [
        ('I', np.array(mean_I),  np.array(sigma_I)),
        ('U', np.array(mean_U),  np.array(sigma_U)),
        ('A', np.array(mean_A),  np.array(sigma_A)),
    ]:
        pc_hat, A_hat, pc_std, A_std, pc_ci = _wls_fit(x, y, sigma, n, confidence)
        results[label] = {
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

# ---- Testing the nu = 4/3 assumption (width line) ----
def nu_width_line(L_list, raw_I, raw_U, confidence=0.95, L_min=50):
    # The width (sample std) of the crossing-fraction distribution scales as w(L) ~ L^{-1/nu},
    # so log w vs log L is a straight line of slope -1/nu. We read nu = -1/slope from a
    # weighted least-squares fit (the same _wls_fit used for p_c). L_min drops the smallest
    # sizes, which carry the strongest finite-size corrections.
    L = np.asarray(L_list, dtype=float)
    raw_I = [np.asarray(r, dtype=float) for r in raw_I]
    raw_U = [np.asarray(r, dtype=float) for r in raw_U]
    if L_min is not None:
        keep = L >= L_min
        L = L[keep]
        raw_I = [r for r, k in zip(raw_I, keep) if k]
        raw_U = [r for r, k in zip(raw_U, keep) if k]
    raw_A = [0.5 * (a + b) for a, b in zip(raw_I, raw_U)]
    n, x = len(L), np.log(L)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 2)
    out = {}
    for label, raw in [('I', raw_I), ('U', raw_U), ('A', raw_A)]:
        w  = np.array([r.std(ddof=1) for r in raw])
        T  = np.array([len(r) for r in raw], dtype=float)
        se = 1.0 / np.sqrt(2.0 * (T - 1.0))          # standard error of log(sample std)
        _, slope, _, slope_std, _ = _wls_fit(x, np.log(w), se, n, confidence)
        nu, nu_err = -1.0 / slope, t_crit * slope_std / slope**2
        out[label] = (nu, nu_err)
        print(f"[{label}] nu = {nu:.4f} +/- {nu_err:.4f}   (2D value 4/3 = {4.0/3.0:.4f})")
    return out

# ---- Testing the isotropy assumption behind the p_A estimator ----
def isotropy_test(L_list, pR, pD, nu=4.0/3.0, confidence=0.95, L_min=50):
    # p_R (horizontal) and p_D (vertical) crossing fractions are recorded in the same sweep
    # (paired). We extrapolate their difference to L -> infinity; consistency with zero means
    # no directional bias, which justifies the averaged estimator p_A.
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
    isotropic = d_ci[0] <= 0.0 <= d_ci[1]
    print(f"[Isotropy] p_R - p_D (L->inf) = {d_inf:+.6f}   {int(confidence*100)}% CI "
          f"[{d_ci[0]:+.6f}, {d_ci[1]:+.6f}]  ->  {'ISOTROPIC' if isotropic else 'anisotropic'}")
    return d_inf, d_ci, isotropic