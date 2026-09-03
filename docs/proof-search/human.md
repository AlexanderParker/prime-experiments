# human.md - the state of the hunt, in plain language

(Manager-rewritten after round 29, 2026-09-03. Current-state snapshot; history in git and
docs/proof-search/archive/.)

## The five-minute version

We model twin primes as a machine: one gear per prime, each blocking positions on a fixed
schedule; twins are the positions every gear misses. The target: prove that adding a gear
never stretches the record blocked-run past its budget, for ALL machines at once.

ROUND 29 IN FOUR RESULTS:

1. THE TENTH LADDER STEP FELL, AND THE ELEVENTH SHOWED THE TOOL'S CEILING. The three-line
   criterion from last round closed step ten with room to spare (132 against a budget of
   150). At step eleven it fails: the old machine's deepest spectrum value is 177 against
   a budget of 171, on an actual exhibited position. The reason is clean. Each extra unit of
   a quantity called A_kill (the kill index) buys a whole new spectrum level (7 to 16 units) while
   the budget only grows by 4 to 6. Every step with A_kill at most 3 certifies; both
   failures are the steps where it reaches 4 or 5. The finer word-based tool still has 26
   units of room at step eleven, so the ladder is not stuck, but the cheap criterion is done.

2. THE HARD PART OF THE PROOF SPLIT INTO TWO NAMED PIECES. Last round's "one constant"
   (the excess the derivation must control) now obeys an exact recursion: each depth's
   excess is the previous depth's excess minus a per-letter residual. So the uniform bound
   we need is equivalent to two things: the residual is bounded per letter (measured, never
   above 4), and the longest legal word L is bounded across machines. That second piece is
   the crux, because A_kill is exactly L + 1 (a theorem this round): L grew to 4 at machine
   47, a new maximum, and that growth is exactly what broke the cheap criterion. Nothing we
   have says L is bounded; nothing says it is not. That is now the sharpest open question.
   One caution: these two pieces settle the depth part of the obligation. The depth-2 part,
   that the budget exceeds the depth-2 spectrum value by a margin that keeps growing, is a
   separate uniform statement that is measured (margin 9 at machine 11, 37 at machine 47)
   and not proved.

3. THE FORMULA IS STRUCTURAL; ONLY ITS SIZE DEPENDS ON THE TEETH. The identity that computes
   the new record from the old machine's spectrum holds at every one of 27,570 counterfactual
   machines with the teeth moved, even though the budget law fails at 13 to 22 percent of
   them. So last round's warning ("no proof from structure alone") narrows to one place: the
   arithmetic of the teeth enters only through how large the spectrum values get. A Fourier
   route to the same quantity was tested and closed: the walk has no spectral content of its
   own, and the natural character-sum mass is identical across all counterfactuals while the
   record varies by a factor of 2.5, so no such bound can see the record.

4. ARITHMETIC KEPT PAYING, AND ONE LANE PAID FOR A BAD PRICE. An LP lane closed a seventh
   increment step at a machine no scan reaches, with 493 exact certificates, and proved its
   31-to-37 certificate uses the smallest possible case split. A lane that had priced its
   own programme at "1 to 100 core-hours" measured the truth at up to 15,000 times more,
   found its previous round's headline value was only a lower bound because its parallel
   search protocol was invalid, re-proved the value two independent ways (one by a SAT
   solver in 14 minutes on one core), and took its programme off the top of its list. The
   model question that programme was built to decide came out exactly as predicted, to the
   unit: 1398.

Also: the round's opening re-verification passed (the record extension F(61) >= 173 is now
manager-checked by an independent path); the small-machine laws found in the "anchor 2,3,5"
line are now kernel theorems for every gear at once, including the attainment of F(17) = 18 with
its witness (the value was known by scan; the kernel had only the upper bound); and two invariants found five rounds apart (phase saturation and the literal
cap) were proved identical at all 48 residue classes in the kernel.

## Honest ledger

The machine crashed once. A kernel build that assembles 385 case proofs into one root file
reached 54 GB of virtual memory on a 16 GB box, exhausted the pagefile, and took Windows
down; the manager had relaunched that build after its first attempt killed the editor, so
the crash is the manager's error. All 385 case proofs are checked; the root that joins them
is not, and stays out of the default build until it is split into tiers. The Formalist's
report was written by the manager from the lane's own document. Every other lane filed with
gates that the manager re-ran green from clean processes after the reboot. Three lanes
retracted or corrected their own earlier numbers (a 15,000x price, an invalid search
protocol, a 3.2x optimistic kernel price), and one lane's pre-registered scorecard was its
worst yet (6 of 17 refuted) and is recorded as such.

## The map

Route: twins infinite <=> no machine ever covers a window (kernel-checked iff).
(A), (B), (C): closed. (D): true at every computable step and beyond the corpus; ten
certified rungs; increment law kernel-grade at six steps and certified by exact LP
certificates (not yet in the kernel) at a seventh; the depth part of the uniform obligation
now equals two statements - a per-letter residual is bounded (measured) and the longest
legal word is bounded (open, and the crux) - and the depth-2 part (the old record plus the
new prime must exceed the depth-2 spectrum value by a margin that stays large) remains a
separate, unproved uniform statement. Next: the tiered kernel root, rung eleven by the finer
tool, the padded-gap census that would decide the three open rows, and any handle on L.
