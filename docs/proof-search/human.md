# human.md - the state of the hunt, in plain language

(Manager-rewritten after round 22, 2026-08-25. Current-state snapshot; history lives in git
and docs/proof-search/archive/.)

## The five-minute version

We treat the twin prime question as a machine: one gear per prime, each blocking positions on a
fixed schedule, and twin primes are the positions every gear misses. The conjecture fails only
if some collection of gears could one day block an entire window - so the proof strategy is
mechanism exhaustion: understand every way gears can conspire and show none of them can do it.

Round 22 delivered the thing this project has been circling for months: A GENERATOR. Instead of
needing a separate rule for how 3 gears interact, then 4, then 5, forever, there is now ONE
algebraic object whose expansion produces every layer at once - verified exact at every step we
can check, by two independent implementations. And it pays off immediately: the target statement
(D) now has a form with NO "for every depth" in it at all. It holds if and only if a certain
potential function exists satisfying three one-step inequalities. Formalist then found and
machine-checked an actual example of such a function.

The proof ladder also grew. (D) is now proven by computer, with ZERO assumptions, at FOUR
consecutive steps of the machine (11->13, 13->17, 17->19, 19->23), and a FIFTH step was proved
independently by a completely different method - a "dual certificate" of 37 fractions that
verifies in about 1,500 arithmetic operations where the direct check would need a 1.6-million-slot
scan. Two unrelated routes agreeing on the same rung is the strongest kind of evidence we can
generate short of the theorem itself.

## What the round asked, and what came back

The question was: does the complexity keep needing bigger and bigger rules (arity), or does it
settle? FOUR independent methods answered, and all four agree: NO FIXED-ARITY RULE EXISTS. But
the reason is better than "it diverges" - the required arity is an ARITHMETIC FUNCTION of the
gear being added. Its literal part is capped at 6 forever (already proved); only its padded part
is uncapped. One lane even measured the growth rate: about 4 x log log y, which is so slow that a
degree-10 certificate would still cover machines up to a million. The tool has a horizon rather
than a cliff.

## What is proven forever

- Each gear's schedule, its onset (nothing new below its square), and the corridor structure.
- Chains of exact repeats cap at 6, as a function of one finite fingerprint.
- The two-teeth kill spacing laws, kernel-checked, giving the fuel cap in closed form.
- (D) itself at four consecutive machine steps, hypothesis-free, plus a fifth by dual certificate.
- The generator identity, exact at every checkable step (kernel formalisation in progress).
- First proved upper bounds on the paired Jacobsthal function j_2 - a ladder that had ZERO
  published attempts before this project. Round 22 added a middle rung that beats round 21's by
  more than 300x at the top of the checked range.

## The wall, precisely

Two independent methods now fail at exactly the same place: the 23->29 step, at depth J=5. One
gives 85 against a budget of 74; the other 91-99 against 74. Same step, same depth, different
mathematics. That is no longer "a wall" - it is a specific object we can go and characterise,
and it is round 23's target.

A second honest limit: the generator is arity-free but NOT YET MACHINE-FREE. It produces every
layer, but we do not yet have a certificate that discharges (D) at EVERY machine rather than
one at a time.

## Honest ledger - five self-refutations this round

The quality signal of the round is that every lane corrected itself:
- Constructor's own round-21 headline ("the arity grows") was measured on the wrong object and
  was corrected by Constructor.
- Lateral refuted its own round-21 claim that the Riemann-style statistics could live in the
  non-tensor sector. That bridge is now closed with a reason: where the machine's spectrum is
  rich it factorises; where it does not factorise the spectrum is degenerate or empty.
- Harvester refuted its OWN pre-registered prediction and a round-21 claim, discovering that the
  effect it expected is exactly zero - and in doing so STRENGTHENED the underlying law from an
  upper bound into a prediction of which lifts are admissible.
- Mechanic retracted an alarming result of its own (a near-violation at gear 53 turned out to be
  a prefix artifact - the alarm evaporates against exact values) and recorded that it had spent
  hours computing two values the project already possessed.
- Harvester downgraded two of the register's novelty claims after finding a February 2025 paper
  that postdates our last prior-art sweep. Both results remain true and were derived
  independently; only the labels were wrong. Standing lesson recorded: PRIOR-ART CHECKS EXPIRE.

## Publication

One unit is close: the paired-Jacobsthal upper bounds (first of any strength in the literature),
needing one explicit constant. A second is a short note. The Lean development is a third, separate.
One previously-planned unit was struck entirely on prior-art grounds - honestly, by the lane that
would have authored it.

## The map

Route: twins infinite <=> no machine ever covers a window (kernel-checked iff).
Pieces (A), (B), (C): closed. (D): open, now in a depth-quantifier-free form, proved at four
consecutive steps plus one by an independent vehicle, with its failure point isolated to a single
identified object (J=5 at 23->29). Everything else - censuses, spectra, certificates, the
novel-findings register - exists to serve that.
