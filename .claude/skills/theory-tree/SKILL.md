---
name: theory-tree
description: Run an open research question as a theory tree - construct a theory, test it, observe, interpret the patterns, branch - with pre-registration, mechanism-first reporting, verdicts kept on a nested tree, parts built in their own coordinates before their interactions, a parts ledger as the gate to interaction work, formalisation in the same round, and a rule against re-deriving known results. Use for any line of enquiry, agent brief, or round write-up in a research project that keeps a theory tree file.
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
- the fixed vocabulary (terms that have caused confusion before, with their one meaning), and
  the rule that new objects get names that cannot be confused with the main objects;
- the evidence standards (what counts as exact, measured, proved; which tools certify);
- compute and memory limits per lane and in total, and how many lanes run at once;
- standing directions from the project owner;
- the index of prior results to check before opening any branch;
- the law register: one project-wide numbering of laws, so two documents never both issue
  "L30" (cite by number, never by document position).

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
4. Name the deliverable in the owner's words when the owner has named one (a closed form for X,
   a proof of Y). Every transform, table and side-measurement in the branch serves it; the final
   report puts it first, with its exact hypothesis, or says exactly where it stopped.

## Running the branch

5. Pre-register before computing: the theory, the testable predictions with numbers, what would
   refute each, and a scorecard. Write them into the branch document first. Record the owner's
   own predictions on the scorecard too, so a refutation in the owner's favour is visible.
6. Test at the project's evidence standard. Prefer exact computation and certificates to
   sampling; report at the extremes and at the mechanism, never averages alone.
7. Observe and report the mechanism first: which parts of the system, which states, what forces
   the outcome. Name a standard theorem or known result it resembles only after the mechanism is
   described, and only in a prior-art line. "Explained by X" is a description of a mechanism, not a
   proof that the property persists.
8. Stop early: the first sign that a sub-question is re-deriving a known result is the signal to
   stop it and say so in one line, not to finish it for completeness.
9. State every law with its exact hypothesis, in the part's intrinsic parameters (its smallest
   component, its component count, its range), never in terms of where the part sits. A law
   stated that way holds automatically for every copy of the part elsewhere in the system; that
   is how self-similarity is obtained for free, and how a law's true hypothesis (often weaker
   than the one first written) is found.
10. Check the regime. A law with a hypothesis is worth exactly the fraction of actual use in which
    the hypothesis holds. Measure that fraction. A closed form valid where every component is
    large, used where the components are small, is a different object, and the branch says which
    regime it is in ("free" versus "loaded", or the project's own words).
11. Interpret against the root: does the finding move toward the answer's required form? What
    would break it, and why can the system not do that? If neither can be answered, it is a fact
    (an identity, a position rule), not a route; say which. If the finding's remaining gap is the
    root question restated (a bound whose proof would be the conjecture), say so and mark it ROOT.

## Parts, then interactions

When a branch has found an object that behaves (a path, a pattern, a record), work it the way
the owner set out: How does the system build it? Which parts of the system contribute to its
shape? Is each part measured individually, understood, and proven? If the parts are proven, the
work is to prove how their interactions produce the shape.

- Inventory the parts, each with its status: proven (cite the proof), measured, or new.
- List the interactions already proven (pairwise laws), count which of them the object actually
  uses, and name the lowest-order interaction that is not yet proven on the way from the parts
  to the shape. That interaction is the next child branch.
- Rank observed features of the shape by the order of interaction needed to explain them.

### Build each part in its own coordinate

A part is constructed on its own terms: its own coordinate, its own smallest components, its own
symmetries, with the other parts as inspiration only. Describing a part relative to another
part during construction poisons the analysis: the other part's structure gets folded into the
coordinate and hides the part's own laws (a fold that turned a two-tooth domino machine into
something that looked like arcs and phases). Only after the part's laws are written does one
look for the exact map between coordinates (a conjugacy), which then says which laws are common
property and which are the part's own.

### Find the smallest simple version

For each part, ask: what is the smallest version that keeps the simplicity? Slide the part's
defining parameter (the split, the smallest component, the count) continuously and record, law
by law, where each one breaks and what mechanism breaks it. The breaking points are the true
hypotheses; the survivors are what transfers to the neighbouring part. This is the cheapest way
to learn what one part's laws say about another.

### Cap the search space

Look for the exact region in which parts beyond the ones under study provably do nothing new
(their actions are duplicates of lower parts' actions, or act only on themselves). Prove the cap
and formalise it, even when it is elementary in the literature: it fixes the search space, names
the one missing instrument, and stops "what about the parts we have not built" from reopening
every round.

## Parts, interfaces, shadows

When the question splits into interacting parts, do not open the interaction until every part is
understood on its own. Keep a parts ledger (per part: definition, proved, measured without
proof, open; a gate verdict) and open the interaction only when no part has an open structural
item. Hidden complexity in a part surfaces inside the interaction as an unexplained blocker.

Build the interaction piecewise: each interface between two parts is an object in its own right,
opened as its own node, with a definition, a proof of its law and a closed form where one exists.
An interface that explains only some behaviour is kept as PARTIAL, never filed dead for
incompleteness; partial interfaces compose, and several may be needed for the whole picture.

When an interaction node dies, the unstick protocol gains one step: what object is the blocker the
shadow of? Define candidates for that object creatively, and allow the answer to reopen
construction of any part, not only the interaction.

## Follow the clue

If a report says "the closest thing to the target", "the only candidate", or "the strongest
fact", that object is the next branch, opened at once with breadth of analysis (decompose it,
transform it, compare it across levels and starting points) before any request for direction.
Do not price a lead by how hard the literature says it is; observe it. Understating the best
lead and closing on it is the failure mode this rule exists for. "Not a route" is a bold claim:
an exact structure is never filed as not a route because a bound is not yet in hand; write what
is missing and open the child that looks for it.

## Formalise in the same round

A law found by a prover goes to the formaliser while the branch is warm, with its written proof
and its numeric check. The kernel is an instrument, not a finish: the hypotheses a proof actually
needs are findings (a lower bound that needs no size hypothesis says where a defect comes from;
a primality hypothesis used in exactly one lemma says where the structure is not generic). Read
them back into the tree as laws. Keep formaliser lanes to one at a time when they share a build
file. The manager re-runs the build and the axiom audit independently before recording a
kernel verdict, and never builds more than the touched targets.

## The tree is a tree, not a list

The tree file has two parts: the tree (nested, carries the verdicts) and the log (chronological,
append-only). A branch is a node; it is not "another item under the root".

- **Nesting is by descent.** A child is the theory made from a pattern observed while testing its
  parent. It sits one level deeper and its first sentence says what spawned it. Only the root's
  formulations hang directly off the root.
- **The verdict lives on the node**, not in the log: STRONG (tested, holds, mechanism visible),
  OPEN (untested or partly tested), WEAK (holds, no mechanism), DEAD (refuted or proved unable),
  FACT (exact, kept, not a route), ROOT (exact, but its remaining gap is the root question
  restated), PARTIAL (an interface that explains some behaviour), each with a pointer to the
  evidence. When a branch dies, say what survived it and where that went. When a branch turns
  out to be a rediscovery, say of what and close it.
- **Candidates for the answer** are marked CANDIDATE on the node, with what would have to break
  them and why the system cannot, or "not yet shown".
- **Facts that are not routes** are kept as FACT nodes under the parent whose question they
  answer, not dropped and not promoted.
- **Numbering follows the nesting.** A branch re-filed under a different parent is moved, with a
  one-line note in the log.
- **Depth over breadth when a node is STRONG**: open its children before a new sibling. Breadth is
  for building the base when nothing is strong.
- **Running lanes are on the tree** as OPEN nodes from the moment they launch, so the tree is
  always the current state, and an owner reading it sees what is in flight.

## When stuck: reopen every dead branch (the unstick protocol)

A dead branch is a brick, not an ending: it measured one face of the wall. Whenever the tree
has no STRONG node left, or a round ends with "we do not know why but we are stuck", run this
protocol over EVERY branch marked DEAD, closed, or FACT-not-a-route, and write the result to a
file beside the tree (the project profile names it). For each such branch:

1. **The object we were attacking.** One sentence, the object itself, not the branch name.
2. **The attack vectors.** What was actually tried, each in a line, with what it measured.
3. **The reason for failure.** The precise thing the measurement showed cannot work, and the
   thing it left untouched.
4. **The shadow.** What object is the blocker the outline of? Name candidates.
5. **Two or more creative ideas to get through it.** Not variations of the failed vector: a
   different object, a different coordinate, an inverse or complement of the object, a
   different quantifier (all phases instead of real ones, integers instead of primes), a
   different scale, a composition with another brick, a return to the construction of a part.
6. **For each idea, two or more ways to realise it.** Concrete: the computation, the
   construction, the lemma to attempt, the literature technique to import, the counterfactual
   to build; each with the machine it runs on and what result would count.

A toolbox of coordinate changes to reach for when generating the ideas in step 5 (the owner's
list, to be extended as tools are tried): set theory (the objects as sets and their algebra:
unions, complements, products, fibres, quotients); bitwise operations, on the numbers themselves
and on the sieve and gap structures (blocked patterns as bit strings, strikes as masks, AND/OR/
XOR of gears' patterns, shifts as re-phasings, popcounts as coverage); linear algebra (incidence
and transfer matrices, ranks, kernels, eigenstructure of the gears' actions); complex numbers
and characters (roots of unity, Fourier and pole-phase views); the walk (the path from any point
to the next event, its layered form, its closed form as a least-excluded value); and any other
method with a different native object (order theory, topology of the torus, formal languages
for gap words, generating functions). A tool already refuted on one object is still untried on
another, and a tool that fails on a part with its neighbour folded in may succeed on the part
alone.

Then read the file as a whole: ideas that recur across branches are the weak points of the
wall; open the one that most branches point at. The dead branches are also new bricks: what
each showed is now a constraint every new idea must respect, so list it beside the idea.

## Closing the branch

12. Branch document: Pre-registered with scorecard, Setup with exact ranges, Results as tables,
    Mechanism, Laws numbered from the project register with proof or exception count, What is
    new (no located prior art, and its use toward the root), Verdict, Dead ends with the
    refuting instance, and a list of the part's remaining open items sorted into: closed here,
    measurement with no structural content, root question in disguise, genuinely open on the
    part alone (with the exact statement and what an attack would look like).
13. Update the node (status, verdict, what survived, children opened) and append one log entry:
    date, lane, one paragraph, new facts first, then refuted predictions, then the stop line and
    the verdict. The log never replaces the node update.
14. Housekeeping: scripts and results where the profile says; large generated data stays
    untracked; no local paths or personal details in committed files; commit after every branch
    with the owner's conventions.
15. Summary for the owner: the findings themselves first, in the chat, with their numbers and
    exception counts, not a pointer to a file; then what is new, then what died and why, then
    the candidates with what would have to break them; then, in one line each, what is running
    and what is next. Price nothing by the literature's difficulty; report where the difficulty
    moved, not only what was gained.
