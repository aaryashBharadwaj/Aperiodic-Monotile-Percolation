# 🟣 Family tiles — spectre + periodic endpoints  (only matters if they enter the paper)

These are new tilings. None of them changes any existing hat result. Verify only if you intend
to publish thresholds for them. Correctness rests on (a) matching a published reference and
(b) exact geometric checks (`verify.py` checks 3 & 4).

---

## `spectre_generator.py` — the Spectre (chiral, Tile(1,1))

**Provenance (state this in the paper):** the base tile coordinates and the substitution rules
are ported **verbatim** from the reference implementation `github.com/shrx/spectre` (a Python
port of Kaplan's construction accompanying Smith–Myers–Kaplan–Goodman-Strauss). The file reuses
your hat generator's own affine primitives (`pt, mul, trot, ttrans, transPt`) and the same
metatile pattern — no new method to describe.

**INTENT (pseudocode) — the two-rule chiral substitution:**
```
base tiles: 8 single-spectre types + 1 "Mystic" (Gamma) = 2 spectres glued
buildSupertiles(system):                     # one inflation step
    compute 8 child transforms from a 7-rule chain of 30-degree rotations + edge alignments
    for each of the 9 metatile types:
        replace it by its list of child tiles (from super_rules) under those transforms
build_spectre_patch(levels): iterate buildSupertiles `levels` times, return one supertile
```

**CHECK:**
- **Do not re-derive the transforms** — verify them by *provenance + output*. Confirm the
  `SPECTRE_POINTS`, `super_rules`, and `transformation_rules` match `shrx/spectre` (diff against
  that repo). Then trust the render: a wrong transform produces visible gaps/overlaps.
- `verify.py` check 3: level-3 is deterministically 559 tiles, the graph has no isolated nodes,
  and coordination averages 2.30 (matching the hat's 2.31 — same monotile vertex structure). A
  broken substitution would fail these.
- Visual: `figures_generated/spectre_patch3.png` — confirm it is the interlocking spectre pattern
  (compare to any published spectre tiling).

---

## Periodic endpoints — chevron (tetriamond) & comet (octiamond)

**Provenance:** the *shapes* are the family endpoints Tile(0,1) and Tile(1,0), obtained by the
documented edge rescaling (8 one-sides→a, 6 r-sides→b) of the hat polykite; the literature names
them a tetriamond and an octiamond. The periodic *tilings* are elementary lattice tilings I
construct and **prove** by exact triangle coverage (no unit-cell is published because it is
considered trivial for a polyiamond).

**INTENT (pseudocode):**
```
rescale hat outline to the endpoint shape ; align it onto the unit triangular lattice
decompose the tile into its unit triangles (test triangle centroids inside the polygon)
search integer lattice bases (u1,u2) with |det| = (#triangles)/2 :
    a basis is VALID iff translating the tile's triangle-set by the whole lattice covers
    every triangle in a window EXACTLY ONCE (no gap, no overlap)
tile by translating the shape along the found (u1,u2)
```

**CHECK:**
- The correctness criterion *is* the definition of a tiling: exact once-coverage. `verify.py`
  check 4 runs it — chevron = 4 triangles, lattice ((−2,0),(−1,−1)); comet = 8 triangles,
  lattice ((−4,0),(−2,−1)) — both pass.
- Caveat to keep honest: a polyiamond can admit **more than one** periodic tiling; these are *a*
  valid periodic tiling of each (the natural pure-translation one), not "the unique" tiling. If
  percolation thresholds depend on the choice, state which tiling you used.
- **Still scratchpad:** the periodic tiler currently lives in a working script, not a project
  module. Before publishing endpoint numbers, promote it to `periodic_generator.py` (like the
  spectre one) and hook it to the graph builder — same pattern, small task.

## Sign-off (only if published)
- [ ] Spectre substitution data diffed against `shrx/spectre`; verify.py check 3 PASS; render is the spectre pattern.
- [ ] Periodic tilings: verify.py check 4 PASS (exact coverage); you accept these are *a* valid periodic tiling.
- [ ] If used for percolation: periodic tiler promoted to a module + builder hookup.
