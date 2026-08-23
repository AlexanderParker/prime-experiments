# human.md - the state of the hunt, in plain language

(Manager-rewritten after round 20, 2026-08-24. This file is the current-state snapshot;
history lives in git and docs/proof-search/archive/.)

## The five-minute version

We treat the twin prime question as a machine: one gear per prime, each blocking positions
on a predictable schedule; twin primes are the positions every gear misses. The conjecture
fails only if some collection of gears can one day block an entire window - so the proof
strategy is mechanism exhaustion: understand every way gears can conspire, and prove none
of them can do it.

The proof needs four pieces. After round 20, THREE ARE COMPLETELY PROVEN - checked by a
computer down to the last logical step, true for every gear forever. The fourth (called D:
"a new gear can never stretch the record gap by more than its own size") is the wall, and
this round stripped it to its barest form yet: one exact inequality between three countable
quantities. It holds at all seven steps we can measure - and at six of the seven it holds
with EQUALITY, which is the strongest possible hint that it is the true mechanism and not
a coincidence.

This was also the round the human's two new ideas paid off. Expressing the machine in
MATRIX form unified four separately-discovered laws into one identity, and turned the
record gap into an algebraic quantity (a "nilpotency index" - the power at which a matrix
dies to zero). Expressing it in COMPLEX/frequency form gave the machine an exact spectrum -
with the golden ratio, the number from art and seashells, appearing exactly as an
eigenvalue. Nobody put it there.

And a nice external validation: our computed values of a difficulty function matched a
published paper digit for digit - reached by a completely different method, about 1000x
fewer operations. Two of our values settle questions the paper's authors left open.

## What is proven forever (the eternal rules)

- Each gear's schedule: two strikes per revolution at closed-form positions. Each gear
  touches nothing new below its own square.
- Chains of exact repeats cap at 6, as a function of one finite fingerprint (48 cases,
  all checked, kernel-proven both directions this round).
- 12 of 24 corridor classes simply cannot occur; the universal Polignac cap holds in all
  8 classes with an EMPTY axiom list.
- Machine 19 is now fully certified in the proof assistant, including the first end-to-end
  instance of (D) proven about the machine's real gap word.
- No three consecutive twin-candidate slots exist, ever (gear 5 alone forbids it) - found
  live this round by pointing a matrix the "wrong" way.

## The wall, precisely

(D) says: new record <= old record + the new gear's size. Round 20 reduced it to:
max(F2, deepest qualifying window) <= F + q'. Every quantity is exactly countable per
machine; what is missing is the reason it holds for ALL machines. The obstruction is now
an algebra fact: the machine's step operator is a difference of two tensor products
(rank 2), and it provably does not factor gear-by-gear. The difficulty provably starts at
THREE-gear interactions - pairwise knowledge cannot see it. The encouraging physics: when
three gears could conspire, they conspire LESS than pair-level prediction says - by
factors up to 1400. The missing rule is whatever forces that suppression.

## The plan (round 21, briefed)

One algebra, not infinite rules - the human's directive. Instead of hunting a rule per
interaction depth, hunt identities in the single algebra that generates every depth:
- nilpotency growth under adding a gear (the (D) proto-law; measured growth ~ q'/2.5,
  needs <= q');
- the transfer matrix rebuilt to carry corridor phase (the round's new measured law says
  the current state variable is wrong);
- positive-definiteness constraints from the exact spectrum (a size constraint derived
  from position laws - the bridge the proof needs);
- the eigenphase statistics test (the Riemann hunch): do our machines' spectra drift
  toward the same statistics as the Riemann zeros? Exact and falsifiable.
- Near-term decision: one computation (34 checks, ~15 min each) decides (D) at the
  37->41 step.

## Honest ledger

- A log line claiming "(D) PROVED" at machine 41 was INVALID (tiny sample, wrong record) -
  caught and struck. A round-19 claim (k_win=1 at 37->41) was corrected by the full
  period: the padded run carries the record there.
- The paired difficulty values we thought were first computations were published in 2017 -
  we missed a companion paper. Ours are now exact independent replications, and their
  table settles our open y=19 question in our favour.
- No uniform budget holds across the whole family of even gaps (harvester killed that
  generalisation); the twin case's own budget is unaffected.
- The margin at the hardest measured step is 19% - and the follow-up test says the
  shrinkage does NOT continue (next real test at gear 53).

## The map

Route: twins infinite <=> no machine ever covers a window (kernel-checked iff).
Pieces: (A) word list - KERNEL-CLOSED. (B), (C) - closed earlier. (D) - open, in the form
above. Everything else - censuses, spectra, SAT certificates, the novel-findings register
(17 entries, all prior-art checked) - exists to serve that one remaining inequality.
