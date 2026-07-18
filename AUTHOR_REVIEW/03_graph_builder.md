# 🟢 Graph builder rewrite  (`hat_graph_builder.py`, `hat_dual_graph_builder.py`)

**Why this is low residual risk:** `verify.py` check 2 proves the rewritten builder produces the
**geometrically identical graph** (same node coordinates, same edge set) as the original
set/list version, at r=4. The rewrite was purely for memory (≈7 GB → ≈2 GB) and speed, so the
question "did it change the graph?" is answered mechanically: no.

---

## `build_neighbor_graph_fast(patch, level, tol)` — the array rewrite

**INTENT (pseudocode) — same graph, numpy/scipy instead of Python sets/dicts:**
```
collect raw polygon vertices of every leaf tile (in order), and each tile's vertex count
merge coincident vertices into unique nodes:
    pairs      = KDTree(raw).query_pairs(tol)         # which raw vertices coincide
    labels     = connected_components(graph of pairs) # each cluster -> one unique node id
    unique_nodes[label] = coordinate                  # (coincident points share a coord)
build edges from tile outlines:
    each tile's consecutive vertices (with wraparound) form its edges
    map raw endpoints -> unique node ids ; drop self-loops ; dedup undirected pairs
build adjacency list (list of int32 arrays) from the deduped edges
return unique_nodes, neighbours
```

**CHECK — is the intent right?**
- Merging vertices within `tol` = 1e-5 into one node is exactly what the original did (it also
  KDTree-matched coincident vertices). Connected-components on the coincidence graph is just a
  vectorised replacement for the original's union-find over the same pairs. ✔
- Edges = consecutive vertices of each tile outline (wraparound) — identical rule to the original
  `_collect_edges`. ✔
- Everything is a re-expression in arrays; **no geometric or topological rule changed.** The
  proof is verify.py check 2 (identical node-set AND edge-set vs the original logic).

**CHECK — the two small behavioural additions (read these lines):**
1. **Preallocation cap 10M → 20M** (`hat_graph_builder.py` near top of the function). The old cap
   silently truncated at r≥6 (level-7 has ~15M raw vertices) → an index-out-of-range crash. New
   cap fits it. Confirm the number and that it only affects preallocation size.
2. **`level=None` = recurse-until-leaf.** For fixed-depth patches (the hat) pass an int → behaves
   exactly as before. `None` recurses until a tile has no children — needed for the spectre
   (variable depth). Confirm the branch: `if ch and (level is None or level > 0)`.

---

## `analyze_square_frame(..., center_x=200, center_y=-100)` — new center parameters

**INTENT:** identical square-frame extraction as before, but the frame centre is now a parameter
(defaults reproduce the old hard-coded r=5 centre (200, −100); r=6 uses (515, −273)).

**CHECK:** confirm the defaults match the old constants, and that `center_x/center_y` only shift
where the L×L window is cut — no change to which nodes/edges/boundaries are selected *given* a
centre. Same one-line change in `hat_dual_graph_builder.analyze_tile_square_frame`.

## Sign-off
- [ ] verify.py check 2 PASS (rewritten builder ≡ original graph).
- [ ] Cap 10M→20M is preallocation-only; `level=None` recurse-to-leaf branch reads correctly.
- [ ] `center_x/center_y` default to the old (200,−100) and only translate the frame.
