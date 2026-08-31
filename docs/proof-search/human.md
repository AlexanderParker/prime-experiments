# human.md - the state of the hunt, in plain language

(Manager-rewritten after round 28, 2026-08-31. Current-state snapshot; history in git and
docs/proof-search/archive/.)

## The five-minute version

We model twin primes as a machine: one gear per prime, each blocking positions on a fixed
schedule; twins are the positions every gear misses. The target: prove that adding a gear
never stretches the record blocked-run past its budget, for ALL machines at once.

ROUND 28 IN FOUR RESULTS:

1. THE NINTH LADDER STEP FELL TO THREE LINES. A new criterion - "the new record is at most
   the old machine's deepest spectrum value" - closed the step that had resisted two rounds
   of heavy search, with margin to spare, and it is honest about its limits: it provably
   fails at one earlier step, so it complements rather than replaces the finer tools.

2. THE REMAINING DIFFICULTY IS ONE CONSTANT, UNIFORMLY. The excess the derivation must
   control was measured across every machine and every depth: it always lies between -3 and
   +4, while the naive estimate grows without bound. And the entire open part of the
   increment law compressed to THREE ROWS in a table - all involving the special "padded"
   letter, all at one machine.

3. THE PROOF MUST USE THE ACTUAL TEETH. Among counterfactual machines (same structure, teeth
   moved), exactly one VIOLATES the budget at each step tested - so no proof using only our
   structural theorems can be valid; the specific arithmetic of where the teeth sit (the
   round(q/6) rule) must enter. Conveniently, the same experiment showed WHERE that
   arithmetic acts: pinning just the incoming gear's tooth eliminates nearly all violations.
   The violating configuration is a palindrome - and palindromic windows in the real machine
   are pinned to one known location and provably harmless. That is the derivation's route
   for half the cases; the other half needs a mechanism not yet found.

4. THE BASE OF THE INDUCTION IS NOW KERNEL-GRADE. The increment law - the induction's
   foundation - is verified inside the proof kernel at all six applicable steps, both
   directions, with sharpness (the constants cannot be improved). One census number that
   originally took a 214-million-slot scan was re-proven in the kernel from a single slot
   in 35 seconds.

Also: a lane retracted its own round-27 discovery (an artifact of its measuring instrument -
found by building a better instrument), the record ladder extended one more rung past the
known corpus (F(61) >= 173, computed on a machine ten trillion times smaller), and a
scheduling fix made the kernel work 9x faster (the bottleneck was CPU starvation, not
mathematics).

## Honest ledger

The round was cut short by the weekly API budget; one lane filed through its pre-staged
crash-proof pipeline, whose workers finished their computation unattended - the design
lesson of round 24 paying off in production. Every lane self-corrected again, including one
catching itself republishing its own earlier result as new, and one squarely retracting its
own previous round's headline. Nothing unverified reached the record; one witness
re-verification is explicitly carried as next round's opening item.

## The map

Route: twins infinite <=> no machine ever covers a window (kernel-checked iff).
(A), (B), (C): closed. (D): true at every computable step and beyond the corpus; nine
certified rungs; increment law kernel-grade at its base; the uniform obligation reduced to
one constant (Delta_J, bounded [-3,+4] empirically) with the odd-depth route mapped and the
even-depth mechanism the sharpest open question. Opus budget resets Sep 4; round 29 waits.
