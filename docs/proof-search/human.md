# human.md - the state of the hunt, in plain language

(Manager-rewritten after round 26, 2026-08-31. Current-state snapshot; history in git and
docs/proof-search/archive/.)

## The five-minute version

We model twin primes as a machine: one gear per prime, each blocking positions on a fixed
schedule; twins are the positions every gear misses. Strategy: prove no set of gears can ever
block an entire window.

ROUND 26 FOUND THE MACHINE'S EXACT RECORD LAW. For months the project has bounded how big the
machine's record gap can get. This round, three lanes converged from three directions on
something stronger: a closed-form expression that does not bound the record - IT IS THE
RECORD, exactly. One lane supplied the missing half of the proof (the law is attained, not
just an upper bound); the census lane then tested it blind at the two deepest machines
anyone can compute - the theorem predicted 118 and 145, and the measurements came back 118
and 145, with witnesses arriving in mirror pairs exactly as the symmetry theory says they
must. The first step of the law is proved inside the proof kernel with the bigger machine's
period PROVABLY absent from the derivation - the strongest "no hidden scan" guarantee
possible. And an independent optimisation vehicle certified every single step of the ladder
with no hypotheses at all, from lists of small exact fractions.

What this means for the conjecture: the target statement (D) now reads "the record law's
value never exceeds the budget" - and because the law is EXACT, there is no slack to hide
in. One uniform inequality about one closed-form object. Everything else is proven.

## Also this round

- A prediction was refuted and replaced by a theorem: the census lane's rule of thumb for
  chain shapes failed its pre-registered test, and the replacement (phase saturation -
  a pattern is impossible if some gear runs out of usable positions) refutes whole families
  with zero solver calls and gives closed-form ceilings, one attained exactly.
- The mirror symmetry lever was perfected: the machine's symmetry group is proved to be
  exactly one flip - so the lever is worth exactly one factor of two, and the worst-case
  configuration provably never occurs an odd number of times. The maximal gap NEVER occurs
  exactly once, at any machine, unconditionally.
- The side paper's lower-bound claim became a real proof with explicit constants - after
  the lane caught its own constant being wrong by a factor of two in the flattering
  direction, by fetching the primary source for a number it was sure of.
- The manager ran his own pre-registered derivation probe: one prediction refuted (recorded),
  and a candidate "increment law" surfaced that holds at 8 of 9 known steps and fails
  exactly at the one padded step - the derivation target for round 27.
- 46% of one lane's historical solver time turned out to be provably redundant (mirror-image
  words have identical counts) - future runs halved.

## Honest ledger

Every lane self-corrected again: one lane fixed its own published table (three claims, one
previously scored CONFIRMED), one nearly filed a first-computation that a sibling had
already pinned, one owned a badly-sized launch, one killed its own 15-worker run rather
than report around it, and the manager's own probe refuted one of his own predictions. The
gates caught everything; nothing unverified reached the record.

## The map

Route: twins infinite <=> no machine ever covers a window (kernel-checked iff).
(A), (B), (C): closed. (D): restated as one uniform inequality on the now-exact record law;
eight rungs kernel-checked; every rung independently LP-certified; the derivation assets
(increment law, phase saturation, parity lever, uniform-order question) all named and all
but one already theorems. Round 27 is briefed: uniformity or bust.
