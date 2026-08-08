# Reproducing the paper's percolation results

Every result comes from **one** consolidated runner, `percolate.py`. It builds the tiling, runs
the Newman–Ziff sweep (the same numba/thread kernels the GUI uses), extrapolates `p_c`, runs the
direction-bias check (site & bond), and saves an `.npz` to `results_output/`. With `--exponents`
it also runs the Block-B cluster pass, recording the fractal dimension `d_f`. It **checkpoints
after every L and resumes** — re-run the same command after an interruption and it picks up where
it stopped.

The GUI's **Run** button launches exactly these commands in the background; the Percolate tab's
"Run from a console instead" panel prints the command for whatever you've configured.

## Running

From the project folder:

```bash
python runner/percolate.py --tiling <name> --graph <direct|dual> --patch <n> \
    --lmin <L> --lmax <L> --gap <ΔL> --trials <T> --seed 123456789
```

- `--tiling`: `Hat`, `Spectre`, `Comet`, `Chevron`, `Square`, `Penrose`,
  `"Triangular → Honeycomb"`, or `"Tile(a,b) family"` (with `--a` / `--b`).
- `--graph`: `direct` (vertex graph) or `dual` (tile-adjacency graph).
- `--patch`: inflation level (Hat/Spectre), subdivisions (Penrose), or block cell-count (periodic/Square).
- member and direct/dual are auto-derived; `--seed 123456789` is the paper seed throughout.
- `--exponents`: also record the fractal dimension `d_f` (an extra cluster sweep, ~+20% time). The
  paper's universality runs use it; leave it off for a quick threshold-only run.
- Windows: prefix with `PYTHONIOENCODING=utf-8` if the console chokes on the arrows.

Progress prints to the console; the `p_c` / direction-bias summary prints at the end; the result `.npz`
lands in `results_output/` (and shows up under the GUI's "Analyse saved" tab).

## The paper results

### Main aperiodic monotiles — Table II (r = 6, T = 1000)

```bash
# Hat — direct (vertex) graph
python runner/percolate.py --tiling Hat --graph direct --patch 6 --lmin 10 --lmax 1000 --gap 10 --trials 1000 --seed 123456789
# Hat — dual (Delone / tile-adjacency) graph
python runner/percolate.py --tiling Hat --graph dual   --patch 6 --lmin 10 --lmax 1000 --gap 10 --trials 1000 --seed 123456789
```

These are the ~10 h runs; they checkpoint, so they survive interruption. (Frame centre auto-resolves
to (515, −272.9), matching the paper's explicit (515, −273).)

### Spectre — the second aperiodic monotile (level 6, T = 500)

```bash
python runner/percolate.py --tiling Spectre --graph direct --patch 6 --lmin 20 --lmax 560 --gap 10 --trials 500 --seed 123456789
python runner/percolate.py --tiling Spectre --graph dual   --patch 6 --lmin 20 --lmax 560 --gap 10 --trials 500 --seed 123456789
```

`--lmax 560` matches the ~55-size level-6 sweep; any L that pokes outside the patch is skipped
automatically, so a slightly generous `--lmax` is safe.

### Periodic family endpoints — Comet, Chevron (block = 150 cells, T = 500)

```bash
python runner/percolate.py --tiling Comet   --graph direct --patch 150 --lmin 20 --lmax 120 --gap 20 --trials 500 --seed 123456789
python runner/percolate.py --tiling Chevron --graph direct --patch 150 --lmin 20 --lmax 120 --gap 20 --trials 500 --seed 123456789
# duals: swap --graph dual
```

The Comet block is a thin strip; frames that exceed it (empty boundary band) are skipped
automatically, so a generous `--lmax` costs nothing.

### Tile(a,b) family (any member by geometry)

```bash
# e.g. the Hat as Tile(1, √3):
python runner/percolate.py --tiling "Tile(a,b) family" --graph direct --a 1 --b 1.732 --patch 6 --lmin 10 --lmax 200 --gap 20 --trials 500 --seed 123456789
```

## Engine validations (recover known thresholds)

### Square lattice — direct-engine validation (site 0.592746, bond 0.5)

```bash
python runner/percolate.py --tiling Square --graph direct --patch 700 --lmin 100 --lmax 600 --gap 50 --trials 40000 --seed 123456789
```

(`patch 700` gives a block large enough for L = 600; Square is self-dual, so `--graph dual` gives the
same lattice.)

### Triangular → Honeycomb — ONE tiling, BOTH engines

The same triangular block validates both graph builders: its **direct** (vertex) graph is the
triangular lattice; its **dual** (tile-adjacency) graph is the honeycomb.

```bash
# direct -> triangular lattice (site 0.5, bond 0.347296)
python runner/percolate.py --tiling "Triangular → Honeycomb" --graph direct --patch 700 --lmin 100 --lmax 600 --gap 50 --trials 40000 --seed 123456789
# dual -> honeycomb (site 0.697040, bond 0.652704)
python runner/percolate.py --tiling "Triangular → Honeycomb" --graph dual   --patch 700 --lmin 100 --lmax 600 --gap 50 --trials 40000 --seed 123456789
```

Together with Square (self-dual) this shows the same generator+builder pipeline reliably recovers
exact known thresholds through both the direct and dual graph representations.

### Penrose (rhombus) — direct, needs the thick boundary band `--bt 10`

```bash
python runner/percolate.py --tiling Penrose --graph direct --patch 13 --lmin 200 --lmax 1800 --gap 100 --trials 25000 --bt 10 --seed 123456789
```

Recovers the literature rhombus-lattice thresholds (site 0.58391, bond ≈ 0.477) to ~4–5 digits
once a corrections-to-scaling fit is applied to the saved raw data. (The default boundary band is 2
for Penrose; the paper run used 10 — hence `--bt 10`.)

