---
name: theory-tree
description: Run an open research question as a theory tree - construct a theory, test it, observe, interpret the patterns, branch - with pre-registration, mechanism-first reporting, verdicts kept on a nested tree, and a rule against re-deriving known results. Use for any line of enquiry, agent brief, or round write-up in a research project that keeps a theory tree file.
---

# Theory tree

A method for working an open question the way a researcher does: construct a theory, test it,
observe the results, interpret the patterns, see whether the patterns can be described, make a
theory about the patterns, repeat. Each repeat is a branch. Depth is preferred when a theory
seems strong, breadth when the idea is exploratory. The question itself is the root; the aim is
to arrive at something you can point at and say "this is always so, because the system works
this way, and nothing it does can prevent it."

## The project profile

Everything specific to one project lives in the header of that project's tree file, not in this
skill. Read it first. It should state:

- the tree file's location and the folder for branch documents and scripts;
- the root question, in the project's own words, and the form an answer must take;
- the fixed vocabulary (terms that have caused confusion before, with their one meaning);
- the evidence standards (what counts as exact, measured, proved; which tools certify);
- compute and memory limits per lane and in total;
- standing directions from the project owner;
- the index of prior results to check before opening any branch.

If the project has no tree file yet, create one with that header and an empty tree and log.

## Before opening a branch

1. Read the project's index of prior results and the tree (branches, verdicts, dead ends).
   Never re-enter a dead end; never re-derive a result the index already has.
2. Say what the branch could find that is NOT already known. If the honest answer is "a
   restatement of a known result in the project's language", do not open it. The exercise is to
   find new rules and relationships, not to translate known ones. A known result met on the way
   is noted and mapped in one line; a fuller translation is written only when it seeds a genuine
   investigation, and is labelled as seeding context.
3. Name the parent. A branch is a child of the node whose observation spawned it, or one of the
   root's formulations. Say which observation, in one sentence.

## Running the branch

4. Pre-register before computing: the theory, the testable predictions with numbers, what would
   refute each, and a scorecard. Write them into the branch document first.
5. Test at the project's evidence standard. Prefer exact computation and certificates to
   sampling; report at the extremes and at the mechanism, never averages alone.
6. Observe and report the mechanism first: which parts of the system, which states, what forces
   the outcome. Name a standard theorem or known result it resembles only after the mechanism is
   described, and only in a prior-art line. "Explained by X" is a description of a mechanism, not a
   proof that the property persists.
7. Stop early: the first sign that a sub-question is re-deriving a known result is the signal to
   stop it and say so in one line, not to finish it for completeness.
8. Interpret against the root: does the finding move toward the answer's required form? What
   would break it, and why can the system not do that? If neither can be answered, it is a fact
   (an identity, a position rule), not a route; say which.

## The tree is a tree, not a list

The tree file has two parts: the tree (nested, carries the verdicts) and the log (chronological,
append-only). A branch is a node; it is not "another item under the root".

- **Nesting is by descent.** A child is the theory made from a pattern observed while testing its
  parent. It sits one level deeper and its first sentence says what spawned it. Only the root's
  formulations hang directly off the root.
- **The verdict lives on the node**, not in the log: STRONG (tested, holds, mechanism visible),
  OPEN (untested or partly tested), WEAK (holds, no mechanism), DEAD (refuted or proved unable),
  FACT (exact, kept, not a route), each with a pointer to the evidence. When a branch dies, say
  what survived it and where that went. When a branch turns out to be a rediscovery, say of what
  and close it.
- **Candidates for the answer** are marked CANDIDATE on the node, with what would have to break
  them and why the system cannot, or "not yet shown".
- **Facts that are not routes** are kept as FACT nodes under the parent whose question they
  answer, not dropped and not promoted.
- **Numbering follows the nesting.** A branch re-filed under a different parent is moved, with a
  one-line note in the log.
- **Depth over breadth when a node is STRONG**: open its children before a new sibling. Breadth is
  for building the base when nothing is strong.

## Closing the branch

9. Branch document: Pre-registered with scorecard, Setup with exact ranges, Results as tables,
   Mechanism, What is new (no located prior art, and its use toward the root), Verdict, Dead ends
   with the refuting instance.
10. Update the node (status, verdict, what survived, children opened) and append one log entry:
    date, lane, one paragraph, new facts first, then refuted predictions, then the stop line and
    the verdict. The log never replaces the node update.
11. Housekeeping: scripts and results where the profile says; large generated data stays
    untracked; no local paths or personal details in committed files; commit conventions as the
    project owner set them.
12. Summary for the owner: plain-language first, then what is new, then what died and why, then
    the candidates with what would have to break them.
