# Author review pack — DO NOT PUSH

This folder exists on the local branch `author-review` only. It is **not** part of the
codebase and must never be pushed or merged. Delete it (or the branch) once you have
completed your review.

## Purpose

Your paper states that all AI-assisted code was hand-checked by the authors. This pack is
built so that statement is **true and efficient to make** — it does not do the checking for
you, it makes your checking fast and evidence-backed.

For each changed piece of code you get two things:
1. **INTENT** — pseudocode of what the code is *supposed* to do, written independently of the
   implementation. You judge whether the intent (the algorithm / the maths) is *correct*.
2. A pointer to the **actual code**, so you confirm the implementation *does the intent*.

Your check therefore splits into two independent, easy questions:
- Is the intent right?  → you decide, against your own knowledge / a textbook.
- Does the code match the intent? → you diff the code against the pseudocode.

Nothing here asks you to trust me. Where a claim can be *proved* (e.g. "the fast code is the
same algorithm as the old code"), `verify.py` regenerates that proof on your machine.

## How to use this pack

1. Run the automated proofs:  `python AUTHOR_REVIEW/verify.py`
   Everything it prints PASS for is mechanically confirmed and needs only a spot-read from you.
2. Work through the section files below **in risk order** (🟡 first — that is the real work).
3. For each function: read the INTENT pseudocode, decide if it is correct, then open the cited
   code file/line and confirm it matches. Tick it off.

## Risk map — where your attention actually needs to go

| Risk | What | Why this risk level | Files |
|------|------|--------------------|-------|
| 🟡 **HIGH — verify the maths yourself** | The 4 statistics functions that PRODUCE YOUR NUMBERS | Novel; not equivalence-checkable against anything; a wrong formula = wrong published value | `01_statistics.md` → `percolation.py` |
| 🟢 **Equivalence-proven** | numba fast pipeline; array rewrite of the graph builder | Provably identical/equivalent to code that already existed. `verify.py` regenerates the proof. Low residual risk. | `02_performance.md`, `03_graph_builder.md` |
| 🔵 **Plumbing** | runners, checkpoint/resume, results.py load fix, plotting | A bug here CRASHES — it does not silently corrupt a number. Lower stakes. | `04_runners_plumbing.md` |
| 🟣 **Family tiles (only if they enter the paper)** | spectre generator, periodic endpoint tilings | Verified against published references + exact geometric checks | `05_family_tiles.md` |

## The honest bottom line

The genuine review surface — the part where only *your* judgement counts — is **four functions
in `percolation.py`** (`01_statistics.md`). Everything else is either provably equivalent to
prior code (run `verify.py`) or fails loudly rather than silently. Spend your time on 🟡.
