---
name: theory-tree
description: Run one branch of the gear-machine proof search the way the project does it - theory, test, observe, interpret, branch - with pre-registration, mechanism-first reporting, the fixed vocabulary, and a verdict logged in research/proof/theory_tree.md. Use for any new line of enquiry, prover brief, or round write-up in this repository.
---

# Theory tree

The project's research method, in the human's words: construct a theory, test it, observe the
results, interpret the patterns, see if the patterns can be described, make a theory about the
patterns, repeat. Each repeat is a branch. Depth is preferred when a theory seems strong, breadth
when the idea is exploratory. The twin prime conjecture is accepted as true; the work is finding
the proof, and the proof target is a known object we can point at and say: this will always be in
the window, because the machine works this way, and nothing the machine does can prevent it.

## Before opening a branch

1. Read docs/novel/README.md (the index of prior results) and research/proof/theory_tree.md
   (branches, verdicts, dead ends). Never re-enter a dead end; never re-derive a result the index
   already has. Two branches in September 2026 were rediscoveries because this step was skipped.
2. Say what the branch could find that is NOT in the corpus. If the honest answer is "a machine-
   language version of a known result", do not open it. The exercise is to use the machine to find
   new rules and relationships, not to translate known mathematics into gear analogy. A known
   result met on the way is noted and mapped in one line; a fuller translation is written only
   when it seeds a machine-driven investigation, and is labelled as seeding context.
3. Fix the vocabulary (docs/proof-search/alignment-rules.md section 0 and the README glossary):
   column k = (6k-1, 6k+1); gear g strikes k iff k = +-6^-1 (mod g); opening = column no gear
   strikes; machine {5..y}; anchor = 2, 3, 5 as one object (cycle 30); window = the certified
   range (y, y^2], never a sliding run; section = the window's new part (p^2, q^2); stretch = a
   sliding run; record F(M) = longest opening-free stretch; the budget inequality
   F(M+q') <= F(M) + q' is a target, never a law. Think in openings, not kills.

## Running the branch

4. Pre-register before computing: the theory, the testable predictions with numbers, what would
   refute each, and a scorecard. Write them into the branch document first.
5. Test on the machine: exact computation (full periods, phase reduction, SAT or LP certificates,
   the Lean kernel), section view for pattern checks, never averages alone. Respect the machine:
   at most 4 cores and 3 GB per lane, 16 GB total; the 385-import Lean root crashed Windows once.
6. Observe and report the mechanism first: which gears, which residues, which columns, what
   forces it. Name a standard theorem it resembles only after the mechanism is described, and only
   in the prior-art line. "Explained by CRT" is a description of a mechanism, not a proof that the
   object persists in the window.
7. Stop early: the first sign that a sub-question is re-deriving a known result is the signal to
   stop it and say so in one line, not to finish it for completeness.
8. Interpret: does the finding move toward naming the object that is always in the window? What
   would remove it from the window, and why can the machine not do that? If it cannot answer
   either, it is a position or identity fact, not a lever; say which.

## The tree is a tree, not a list

The file research/proof/theory_tree.md has two parts: the tree (nested, carries the verdicts) and
the log (chronological, append-only). A branch is a node in the tree; it is not "another item
under the root".

- **Nesting is by descent.** A child branch is the theory made from a pattern observed while
  testing its parent. It sits nested under the parent (one more level of indentation) and its
  first sentence says what it was spawned by: "Spawned by the observation that ...". A branch
  with no parent observation is a formulation of the root (there are three: per step, whole
  window, structure of the record) and hangs directly off the root. Nothing else does.
- **The verdict lives on the node**, not in the log: STRONG, OPEN, WEAK, DEAD, FACT (exact,
  kept, not a route), with the pointer to the evidence. When a branch dies, say what survived it
  and where that went ("what survived: the tiling observation, which became branch 5"). When a
  branch is found to be a rediscovery, say which docs/novel entry and close it.
- **Candidates for the target object** are marked CANDIDATE OBJECT on the node, with what would
  have to remove the object from the window and why the machine cannot, or "not yet shown".
- **Facts that are not routes** (an exact identity, a position rule) are kept as FACT nodes under
  the parent whose question they answer, not dropped and not promoted.
- **Numbering follows the nesting** (1, 1a, 1e.i; 5, 5b, 5d). A branch opened under a different
  parent than it was first filed under is moved, with a one-line note in the log.
- **Depth over breadth when a node is STRONG**: open its children before opening a new sibling.
  Breadth (new siblings) is for building the base when nothing is strong.

## Closing the branch

9. Branch document (research/proof/<branch>.md): Pre-registered with scorecard, Setup with exact
   ranges, Results as tables, Mechanism in machine terms, What is new (no located prior art, and
   its use to the route), Verdict, Dead ends with the refuting instance.
10. Update the node in the tree (status, verdict, what survived, children opened) and append one
    log entry: date, lane, one paragraph, new facts first, then refuted predictions, then the stop
    line and the verdict, with kernel names where they exist. The log never replaces the node
    update.
11. Housekeeping: scripts under research/<line>/r<round>/, results under .../results/; large
    generated data (npz, npy, big csv) stays untracked (.gitignore); no local paths or personal
    details in committed files; no attribution trailers on commits.
12. Round summary for the human: ELI5 first, then what is new, then what died and why, then the
    candidate objects for "always in the window" with what would have to remove them.
