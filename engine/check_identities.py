"""Run the isomorphism/identity checker over the monotile family and print which duals are KNOWN
lattices (so they need no Monte-Carlo run) and which are genuinely novel aperiodic graphs.

    python -m engine.check_identities            # default small patches (fast, ~1-2 min)
    python -m engine.check_identities --patch 4  # bigger interior, slower

Verdicts:
  * comet-periodic dual  -> triangular
  * chevron-periodic dual-> square
  * Tile(1,1) dual       -> triangular      } proven lattices: drop from the run, cite the exact value
  * comet-aperiodic dual == hat dual         (isomorphic: comet-ap IS the hat folded, so all graph
                                              invariants match -- V, E, degree/triangle/WL histograms)
  * hat / spectre / chevron-aperiodic duals -> aperiodic/novel (must be measured; kept in the batch)
"""
import argparse
import numpy as np
import interface.gui_backend as gb
from engine import isomorphism as iso


def _graph(tiling, kind, patch, a=None, b=None):
    kw = {} if a is None else {"a": a, "b": b}
    bundle = gb.build_graph(tiling, kind, patch, **kw)
    return bundle["coords"], bundle["neighbors"]


def classify(name, tiling, patch):
    coords, nbrs = _graph(tiling, "Dual", patch)
    lat, sig = iso.classify_lattice(coords, nbrs)
    dom = sig["dominant"]; unif = sig["uniformity"]
    print(f"  {name:26s} dual: V={len(coords):>6}  interior={sig['n_interior']:>6}  "
          f"dominant(deg,tri)={str(dom):>8}  uniformity={unif:5.1%}  ->  {lat.upper()}")
    return lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, default=3, help="hat/spectre metatile depth (bigger = larger interior)")
    ap.add_argument("--cells", type=int, default=120, help="periodic tiling cell count")
    ap.add_argument("--rounds", type=int, default=6, help="Weisfeiler-Lehman rounds for the identity test")
    args = ap.parse_args()

    print("=" * 96)
    print("KNOWN-LATTICE DUALS  (interior degree,triangle signature == a classic lattice -> no MC run needed)")
    print("=" * 96)
    classify("Comet periodic",   "Comet",            args.cells)
    classify("Chevron periodic", "Chevron",          args.cells)
    classify("Tile(1,1) periodic", "Tile(1,1) periodic", max(200, args.cells))

    print("\n" + "=" * 96)
    print("IDENTITY  (comet-aperiodic is the hat FOLDED: same combinatorics, tiles reshaped -> compare up")
    print("           to relabelling, since folding moves the physical coordinates)")
    print("=" * 96)
    ca, na = _graph("Comet aperiodic", "Dual", args.patch)
    hc, hn = _graph("Hat",             "Dual", args.patch)
    ok, d = iso.isomorphic(hn, na, rounds=args.rounds)
    a, b = d["a"], d["b"]
    print(f"  comet-aperiodic dual  vs  hat dual :  V {a['V']}/{b['V']}  E {a['E']}/{b['E']}  "
          f"deg-hist {'=' if a['degree_hist']==b['degree_hist'] else 'X'}  "
          f"tri-hist {'=' if a['triangle_hist']==b['triangle_hist'] else 'X'}  "
          f"WL-hist {'=' if a['wl_hist']==b['wl_hist'] else 'X'}")
    print(f"     -> {'ISOMORPHIC (all invariants match -> same p_c by construction)' if ok else 'NOT isomorphic -- investigate'}")

    print("\n" + "=" * 96)
    print("TURTLE == HAT  (the general maker used on the one member that needs it: Tile(sqrt3,1) is the")
    print("               hat re-embedded with its two edge lengths swapped -> same graph, same p_c)")
    print("=" * 96)
    from generators.aperiodic_collapse import turtle_graph
    for kind in ("direct", "dual"):
        hc, hn = _graph("Hat", kind.capitalize(), args.patch)
        tc, tn, _ = turtle_graph(kind, args.patch)
        ok, d = iso.isomorphic(hn, tn, rounds=args.rounds)
        a, b = d["a"], d["b"]
        print(f"  turtle {kind:6s} vs hat {kind:6s}:  V {a['V']}/{b['V']}  E {a['E']}/{b['E']}  "
              f"deg {'=' if a['degree_hist']==b['degree_hist'] else 'X'}  "
              f"tri {'=' if a['triangle_hist']==b['triangle_hist'] else 'X'}  "
              f"WL {'=' if a['wl_hist']==b['wl_hist'] else 'X'}  ->  {'ISOMORPHIC' if ok else 'MISMATCH -- investigate'}")

    print("\n" + "=" * 96)
    print("NOVEL APERIODIC DUALS  (interior is NOT a single-lattice signature -> genuinely new, must measure)")
    print("=" * 96)
    for name, tiling in [("Hat", "Hat"), ("Spectre", "Spectre"), ("Chevron aperiodic", "Chevron aperiodic")]:
        classify(name, tiling, args.patch)

    print("\nnote: 'uniformity' = fraction of interior nodes on the dominant (deg,tri) pair; a periodic")
    print("      dual is ~100% uniform, an aperiodic dual spreads across several pairs.")


if __name__ == "__main__":
    main()
