# 🟢 Performance pipeline  (`percolation_fast.py`, `percolation_baseline.py`)

**Why this is low residual risk:** `verify.py` check 1 proves the fast serial path is
**bit-identical** to the frozen baseline on the same seed. So "did the optimisation change the
answer?" is answered mechanically: no. Your job here is a spot-read that the *algorithm* is the
one you already trust, plus one conceptual point about the parallel RNG.

`percolation_baseline.py` is a frozen copy of the pre-optimisation `percolation.py` kept solely
as the comparison target for verify.py. Nothing calls it in production.

---

## The JIT kernels — `_find`, `_union`, `_site_trial`, `_bond_trial`

**INTENT (pseudocode) — this is textbook Newman–Ziff incremental union-find:**
```
_find(p):          # path-halving find
    while p != parent[p]:
        parent[p] = parent[parent[p]]      # halve the path
        p = parent[p]
    return p

_union(a,b):       # weighted union by size
    ra, rb = _find(a), _find(b)
    if ra == rb: return
    attach smaller tree under larger; add sizes

_site_trial(order):                        # add sites one at a time in random `order`
    two independent union-finds: TB (top<->bottom) and LR (left<->right)
    for each site idx in order:
        open idx
        if idx on a boundary: union it to that boundary's virtual node
        for each already-open neighbour: union(idx, neighbour) in BOTH TB and LR
        record onset the first step TB spans, and the first step LR spans
        INTERSECTION: stop when BOTH TB and LR span ;  UNION: stop when EITHER spans
    return onset fraction, tb_onset, lr_onset
```

**CHECK — is this the algorithm you already use?** It is the same weighted-quick-union +
path-halving + virtual top/bottom/left/right nodes as the original `percolation.py`. The ONLY
change is `@njit` compilation and building the CSR neighbour arrays + boundary masks **once per
size** instead of per trial. No estimator, criterion, or ordering changed.
- Confirm `percolation_fast.py:_site_trial` / `_bond_trial` match this.
- Confirm the stats classes still use `random.shuffle(list(range(N)))` — that is what makes them
  bit-identical to the baseline (proved by verify.py).

---

## The parallel classes — `percolationStatsI_par`, `..._par` (site U, bond I, bond U)

**INTENT (pseudocode):**
```
build CSR neighbours + boundary masks ONCE
seeds = SeedSequence(master_seed).spawn(#trials)     # independent reproducible seed per trial
run trials across a thread pool (the @njit kernel releases the GIL, so threads run in parallel):
    trial k: order = default_rng(seeds[k]).permutation(N)   # instead of random.shuffle
    onset  = _site_trial(order)
collect onsets / T
```

**CHECK — the one conceptual point (you flagged this yourself):**
- The parallel path is **seeded-reproducible but NOT bit-identical** to the serial baseline,
  because it draws each trial's permutation from an independent `SeedSequence` stream instead of
  one shared `random.shuffle` stream. This is a deliberate, correct choice: independent streams
  per trial are the standard way to parallelise Monte Carlo, and the trials are i.i.d. samples of
  the same distribution either way. It does **not** bias any estimate. Confirm you are comfortable
  stating "results are reproducible from a fixed seed" (true) rather than "bit-identical to the
  serial code" (deliberately not).
- Threads are safe because each trial allocates its **own** union-find arrays inside the njit
  kernel (no shared mutable state); only the read-only CSR/masks are shared.

**Proof to run:** `verify.py` check 1 (serial ≡ baseline). The parallel path shares the exact
same `_site_trial`/`_bond_trial` kernel — only the permutation source differs.

## Sign-off
- [ ] Kernels are the Newman–Ziff union-find you already trust; only compilation + hoisted setup changed.
- [ ] verify.py check 1 PASS (serial bit-identical to baseline).
- [ ] You accept the parallel RNG statement ("reproducible from seed", not "bit-identical").
