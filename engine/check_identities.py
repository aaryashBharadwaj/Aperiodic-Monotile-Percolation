"""Prove the graph identities that let certain duals be quoted from a known value instead of Monte-Carlo
run. Every proof is an EXPLICIT node-to-node mapping -- no VF2, no NetworkX, no Weisfeiler-Leman hash.

    python -m engine.check_identities            # default small patches (fast)
    python -m engine.check_identities --patch 4  # bigger interior, slower

The five identities:
  * comet-periodic dual   -> triangular  (developed bijection phi, every interior edge verified)
  * chevron-periodic dual -> square      "
  * Tile(1,1) dual        -> triangular  "   -> cite the exact lattice value, no run
  * comet-aperiodic dual  == hat dual    (bit-identical: same graph, phi = identity -> same p_c)
  * turtle == hat          (bit-identical)
The hat / spectre / chevron-aperiodic duals have NO such identity -- they are the paper's measured
aperiodic objects, so this checker makes no claim about them.
"""
import argparse
import interface.gui_backend as gb
from engine import isomorphism as iso


def _graph(tiling, kind, patch):
    b = gb.build_graph(tiling, kind, patch)
    return b["coords"], b["neighbors"]


def periodic(name, tiling, lattice, patch):
    coords, nbrs = _graph(tiling, "Dual", patch)
    ok, st = iso.is_lattice(coords, nbrs, lattice)
    print(f"  {name:26s} dual -> {lattice:10s}: V={st['V']:>6}  interior edges {st['edges_preserved']}/{st['edges_total']}"
          f"  injective={st['collisions'] == 0}  ->  {'PROVEN' if ok else 'NOT ' + lattice}")
    return ok


def bitid(name, na, nb):
    ok, d = iso.bit_identical(na, nb)
    print(f"  {name:34s} V={d.get('V', '?')}  ->  {'IDENTICAL (same p_c by construction)' if ok else 'DIFFER ' + str(d)}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, default=3, help="hat/spectre metatile depth")
    ap.add_argument("--cells", type=int, default=120, help="periodic tiling cell count")
    args = ap.parse_args()

    print("=" * 100)
    print("PERIODIC DUALS -> KNOWN LATTICE   (explicit developed bijection phi, every interior edge verified)")
    print("=" * 100)
    periodic("Comet periodic",    "Comet",              "triangular", args.cells)
    periodic("Chevron periodic",  "Chevron",            "square",     args.cells)
    periodic("Tile(1,1) periodic", "Tile(1,1) periodic", "triangular", max(200, args.cells))

    print("\n" + "=" * 100)
    print("APERIODIC GRAPH IDENTITY   (bit-identical: same graph, phi = identity)")
    print("=" * 100)
    _, na = _graph("Comet aperiodic", "Dual", args.patch)
    _, hn = _graph("Hat", "Dual", args.patch)
    bitid("comet-aperiodic dual == hat dual", hn, na)
    from generators.aperiodic_collapse import turtle_graph
    for kind in ("direct", "dual"):
        _, hn = _graph("Hat", kind.capitalize(), args.patch)
        _, tn, _ = turtle_graph(kind, args.patch)
        bitid(f"turtle {kind} == hat {kind}", hn, tn)


if __name__ == "__main__":
    main()
