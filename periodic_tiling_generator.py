import numpy as np

# generates a square lattice
def square_lattice(L):
    N = L * L
    neighbors = [[] for _ in range(N)]
    edges = set()
    for i in range(L):
        for j in range(L):
            idx = i*L + j
            if j < L-1:
                neighbor = i*L + (j+1)
                neighbors[idx].append(neighbor)
                neighbors[neighbor].append(idx)
                edges.add((min(idx, neighbor), max(idx, neighbor)))
            if i < L-1:
                neighbor = (i+1)*L + j
                neighbors[idx].append(neighbor)
                neighbors[neighbor].append(idx)
                edges.add((min(idx, neighbor), max(idx, neighbor)))
    return np.arange(N), neighbors, np.array(list(edges), dtype=np.int32)

# generates a triangular lattice
def triangular_lattice(L):
    N = L * L
    neighbors = [[] for _ in range(N)]
    edges = set()
    for i in range(L):
        for j in range(L):
            idx = i*L + j
            if j < L-1:
                neighbor = i*L + (j+1)
                neighbors[idx].append(neighbor)
                neighbors[neighbor].append(idx)
                edges.add((min(idx, neighbor), max(idx, neighbor)))
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

# Boundary sets

def boundary_sets(L):
    N = L * L
    top    = np.arange(L)
    bottom = np.arange(N - L, N)
    left   = np.arange(0, N, L)
    right  = np.arange(L - 1, N, L)
    return top, bottom, left, right