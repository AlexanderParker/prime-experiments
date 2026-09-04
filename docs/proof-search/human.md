# human.md - the state of the hunt, in plain language

(Manager-rewritten after round 30, 2026-09-04. Current-state snapshot; history in git and
docs/proof-search/archive/.)

## The five-minute version

We model twin primes as a machine: one gear per prime, each blocking positions on a fixed
schedule; twins are the positions every gear misses. The target: prove that adding a gear
never stretches the record blocked-run past the budget (old record plus the new prime), for
ALL machines at once. That inequality is the target, measured true at every computable step;
it is not a law.

ROUND 30 IN FOUR RESULTS:

1. THE ELEVENTH STEP FELL, TWICE, BY ROUTES THAT NEVER LOOK AT THE QUANTITY THAT BROKE THE
   CHEAP TOOL. Step ten's shortcut fails at step eleven because a depth number grew. This
   round two other tools closed step eleven anyway: an exact linear-programming certificate
   (8,077 pieces, every one checked in exact arithmetic, covering every case of the machine
   exactly once) and the finer word-based bound (145 against a budget of 171, computed on a
   machine ten trillion times smaller and needing nothing about the new machine). So the
   finite rungs of the ladder do not depend on that depth number at all. Only the uniform
   proof does.

2. THE CRUX IS NOW PINNED TO ITS ARITHMETIC. The open question is whether the longest "legal
   word" L stays bounded across all machines. The control group settled what kind of fact that
   is: on machines with the same structure but moved teeth, L reaches 5 where the real machine
   has 2. So no proof from structure alone can bound L; the real teeth must enter. Where they
   enter is located too: through machine 29 the small value of L is decided by gears 5 and 7
   alone, through a finite check; beyond that, gears 5 and 7 provably cannot do it, and every
   word that dies dies because no slot of the old machine blocks its interior, never because
   of a large gear. One more thing fell out: the length of L is what a random model with the
   machine's own gap histogram predicts, to within one unit; only its last unit is arithmetic.
   The manager's earlier "random model predicts 18" was wrong (the exact figure is 12, and
   with the real histogram it is within a factor of two of the truth).

3. THE HARDEST KERNEL BUILD IS DONE, AND THE CRASH IS EXPLAINED. The 385-case proof of step
   31 to 37 that took the machine down last round is now checked by the proof kernel, built
   in 35 tiers at 4 GB. The 54 GB was not the 385 imports (1.4 GB) but eleven "bridge" proof
   steps written in a style that costs 0.4 GB each; rewritten, a tier costs 2.8 GB. The
   round-29 theorem "kill index = longest legal word + 1" is also in the kernel now, over an
   abstract machine.

4. THE RECORDS' FAMILY TREE. Tracing every exact record back through the machines below it:
   the record at one machine is built from a runner-up of the machine below, not from its
   record, two to five generations deep. And the one place the increment inequality strains
   (three rows at machine 31) is now a single event: the old machine's depth-3 maximiser has a
   padded middle letter, and nowhere else in the corpus does that happen.

Also: the exact word census that Constructor asked for three rounds ago was delivered (its
top value matches a count Mechanic found in round 26 by another route); the mirror is now a
theorem with a companion translation lemma that explains a round-29 puzzle to the number; the
"anchor 2, 3, 5" laws have their own novel-findings document with the prior-art verdict "none
found" for the two-class versions; and the one published paper closest to this work (Holt and
Rudd 2014) turns out to stop exactly where the project's depth quantity starts to matter.

## Honest ledger

The round ran through an Anthropic API incident. Every lane was cut off repeatedly, and the
manager's habit of resuming them on each failure burned the session budget, so the round
finished the next morning. No data was lost: every lane writes results from the worker, and all
six filed with gates the manager re-ran green from clean processes afterwards. Three lanes
refuted their own pre-registered predictions in part (Constructor's eps mechanism, Mechanic's
"length is cover" reading, Formalist's memory price for the old bridge), and the manager's own
null estimate for L was wrong by a factor of 1.5 and is recorded as such. The interactive
visualisation built during the outage was judged not valuable and set aside. Next round the
lanes run on Opus 5, and during any outage they wait rather than retry.

## The map

Route: twins infinite <=> no machine ever covers a window (kernel-checked iff).
(A), (B), (C): closed. (D): true at every computable step and beyond the corpus; eleven
certified rungs, the last two by tools that never see the depth number; the 31 to 37 case-split
proof in the kernel. The uniform obligation: a per-letter residual bounded on literal letters
(measured, at most 4), one padded exception located and explained, the depth-2 slack (measured
9 to 49, unproved), and L bounded, which is the crux, is arithmetic not structural, and now has
a named route: bound the legal alphabet's class densities, then close the last unit by the
cover half.
