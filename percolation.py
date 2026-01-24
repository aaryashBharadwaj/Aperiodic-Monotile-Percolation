import numpy as np
import random
from tqdm import tqdm
from graph_builder import neighbors_to_csr

# Weighted Quick Union with path compression for efficient disjoint set operations
class WeightedQuickUnionUF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.count = n
    def find(self, p):
        while p != self.parent[p]:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        return p
    def connected(self, p, q):
        return self.find(p) == self.find(q)
    # Merge two components, attaching smaller tree to larger
    def union(self, p, q):
        rootP, rootQ = self.find(p), self.find(q)
        if rootP == rootQ: return
        if self.size[rootP] < self.size[rootQ]:
            self.parent[rootP] = rootQ
            self.size[rootQ] += self.size[rootP]
        else:
            self.parent[rootQ] = rootP
            self.size[rootP] += self.size[rootQ]
        self.count -= 1

# Intersection site percolation: requires BOTH top-bottom AND left-right spanning:
class HatPercolationI:
    def __init__(self, nodes, neighbours, top_set, bottom_set, left_set, right_set):
        self.N = len(nodes)
        # Convert adjacency list to CSR format for faster neighbor lookups
        self.neighbors_arr, self.neighbor_starts = neighbors_to_csr(neighbours)
        self.top_set, self.bottom_set = set(top_set), set(bottom_set)
        self.left_set, self.right_set = set(left_set), set(right_set)
        self.sites = np.zeros(self.N, dtype=bool)  # Use boolean array
        self.wqfTB = WeightedQuickUnionUF(self.N + 2)
        self.wqfLR = WeightedQuickUnionUF(self.N + 2)
        self.vTop, self.vBot = self.N, self.N + 1
        self.vL, self.vR = self.N, self.N + 1
        self.openSite = 0
        
    def open_site(self, idx):
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
    def __init__(self, nodes, neighbours, top, bot, left, right, trials, mode='Site'):
        self.trialCount = trials
        self.trialResults = []
        
        for _ in tqdm(range(trials), desc="Site I", leave=False):
            sim = HatPercolationI(nodes, neighbours, top, bot, left, right)
            # Pre-shuffle all sites once 
            sites_order = list(range(sim.N))
            random.shuffle(sites_order)
            # Open sites in shuffled order until percolation
            for site in sites_order:
                sim.open_site(site)
                if sim.percolates():
                    break
            
            self.trialResults.append(sim.openSite / sim.N)
            
    def trials_mean(self): return np.mean(self.trialResults)
    def trials_std(self): return np.std(self.trialResults)
    def report(self): print(f"Mean pc: {self.trials_mean():.6f} | Std: {self.trials_std():.6f}")

# Union site percolation: requires EITHER top-bottom OR left-right spanning:
class percolationStatsU(percolationStatsI):
    def __init__(self, nodes, neighbours, top, bot, left, right, trials):
        self.trialCount = trials
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

# Intersection bond percolation: requires BOTH top-bottom AND left-right spanning:
class HatPercolationBondI:
    def __init__(self, nodes, edges, top_set, bottom_set, left_set, right_set):
        self.num_nodes = len(nodes)
        self.edges = edges
        self.num_edges = len(edges)
        self.edge_status = np.zeros(self.num_edges, dtype=bool)  # Boolean array
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
        self.edge_status[idx] = True
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
    def report(self): print(f"Mean Bond pc: {self.trials_mean():.6f} | Std: {self.trials_std():.6f}")

# Union bond percolation: requires EITHER top-bottom OR left-right spanning:
class percolationStatsBondU(percolationStatsBondI):
    def __init__(self, nodes, edges, top, bot, left, right, trials):
        self.trialResults = []
        
        for _ in tqdm(range(trials), desc="Bond U", leave=False):
            sim = HatPercolationBondU(nodes, edges, top, bot, left, right)
            edges_order = list(range(sim.num_edges))
            random.shuffle(edges_order)

            # Open edges until percolation occurs
            for edge in edges_order:
                sim.open_bond(edge)
                if sim.percolates():
                    break
                    
            self.trialResults.append(sim.openBonds / sim.num_edges)