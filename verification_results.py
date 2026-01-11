import numpy as np
from percolation import percolationStatsI, percolationStatsU, percolationStatsBondI, percolationStatsBondU
from visualisation import plot_percolation_stats_IU, plot_extrapolation_IU

# -----------------------------
# Lattice definitions
# -----------------------------
def square_lattice(L):
    N = L * L
    neighbors = [[] for _ in range(N)]
    edges = set()
    for i in range(L):
        for j in range(L):
            idx = i*L + j
            # Right neighbor
            if j < L-1:
                neighbor = i*L + (j+1)
                neighbors[idx].append(neighbor)
                neighbors[neighbor].append(idx)
                edges.add((min(idx, neighbor), max(idx, neighbor)))
            # Bottom neighbor
            if i < L-1:
                neighbor = (i+1)*L + j
                neighbors[idx].append(neighbor)
                neighbors[neighbor].append(idx)
                edges.add((min(idx, neighbor), max(idx, neighbor)))
    return np.arange(N), neighbors, np.array(list(edges), dtype=np.int32)

def triangular_lattice(L):
    N = L * L
    neighbors = [[] for _ in range(N)]
    edges = set()
    for i in range(L):
        for j in range(L):
            idx = i*L + j
            # Right neighbor
            if j < L-1:
                neighbor = i*L + (j+1)
                neighbors[idx].append(neighbor)
                neighbors[neighbor].append(idx)
                edges.add((min(idx, neighbor), max(idx, neighbor)))
            # Bottom neighbor
            if i < L-1:
                neighbor = (i+1)*L + j
                neighbors[idx].append(neighbor)
                neighbors[neighbor].append(idx)
                edges.add((min(idx, neighbor), max(idx, neighbor)))
            # Diagonal (bottom-right)
            if i < L-1 and j < L-1:
                neighbor = (i+1)*L + (j+1)
                neighbors[idx].append(neighbor)
                neighbors[neighbor].append(idx)
                edges.add((min(idx, neighbor), max(idx, neighbor)))
    return np.arange(N), neighbors, np.array(list(edges), dtype=np.int32)

# -----------------------------
# Boundary sets
# -----------------------------
def boundary_sets(L):
    N = L*L
    top = np.arange(L)
    bottom = np.arange(N-L, N)
    left = np.arange(0, N, L)
    right = np.arange(L-1, N, L)
    return top, bottom, left, right

# -----------------------------
# Run percolation test
# -----------------------------
def run_percolation_test(L_values, trials=50, lattice_type='square'):
    mSI_list, sSI_list, mSU_list, sSU_list = [], [], [], []
    mBI_list, sBI_list, mBU_list, sBU_list = [], [], [], []

    for L in L_values:
        print(f"\n=== {lattice_type.capitalize()} Lattice L={L} ===")
        nodes, neighbors, edges = (square_lattice(L) if lattice_type=='square' else triangular_lattice(L))
        top, bottom, left, right = boundary_sets(L)

        # --- Site percolation ---
        statsSI = percolationStatsI(nodes, neighbors, top, bottom, left, right, trials)
        statsSU = percolationStatsU(nodes, neighbors, top, bottom, left, right, trials)
        mSI_list.append(statsSI.trials_mean())
        sSI_list.append(statsSI.trials_std())
        mSU_list.append(statsSU.trials_mean())
        sSU_list.append(statsSU.trials_std())

        # --- Bond percolation ---
        statsBI = percolationStatsBondI(nodes, edges, top, bottom, left, right, trials)
        statsBU = percolationStatsBondU(nodes, edges, top, bottom, left, right, trials)
        mBI_list.append(statsBI.trials_mean())
        sBI_list.append(statsBI.trials_std())
        mBU_list.append(statsBU.trials_mean())
        sBU_list.append(statsBU.trials_std())

        # Print averages
        avg_site = 0.5*(statsSI.trials_mean()+statsSU.trials_mean())
        avg_bond = 0.5*(statsBI.trials_mean()+statsBU.trials_mean())
        print(f"Site pc: I={statsSI.trials_mean():.4f}, U={statsSU.trials_mean():.4f}, Avg={avg_site:.4f}")
        print(f"Bond pc: I={statsBI.trials_mean():.4f}, U={statsBU.trials_mean():.4f}, Avg={avg_bond:.4f}")

    # -----------------------------
    # Plot results
    # -----------------------------
    plot_percolation_stats_IU(L_values, mSI_list, sSI_list, mSU_list, sSU_list,
                              mBI_list, sBI_list, mBU_list, sBU_list)

    # Extrapolation using averages
    mSI_avg = np.array([0.5*(i+j) for i,j in zip(mSI_list, mSU_list)])
    mBI_avg = np.array([0.5*(i+j) for i,j in zip(mBI_list, mBU_list)])
    plot_extrapolation_IU(L_values, mSI_avg, mSI_avg, mBI_avg, mBI_avg)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    system_sizes = [100, 200, 300, 400]  # Increase for more convergence
    trials = 100  # Number of trials per Lattice/Percolation

    print("\n### Testing Square Lattice ###")
    run_percolation_test(system_sizes, trials, lattice_type='square')

    print("\n### Testing Triangular Lattice ###")
    run_percolation_test(system_sizes, trials, lattice_type='triangular')
