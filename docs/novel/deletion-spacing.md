# deletion-spacing - merge deletions are at least q-1 apart, and that is tight

## 1. WHAT IT IS

Plain language. When a new gear q is added to the machine, its deletions of old openings happen
lap by lap (see `merge-law.md`): within one lap the deleted openings all lie in two fixed residue
classes mod q. The lemma says two consecutive deletions inside one lap can never be closer than
q - 1, and that this bound is attained. So a stretch of length G contains at most 1 + G/(q-1)
deletions, long merges are rare, and the growth of the maximum gap under one added gear is
governed by q, not by the old maximum - the mechanical reason an increment law in q is possible
at all.

Precise form (adjacent frame: every odd prime q, gear 3 included, blocks the adjacent pair
{o, o+1} mod q; all machine positions and gaps live in this frame, and are exactly 3x the k-frame
quantities).

> **Lemma (deletion spacing).** Let M be a machine with exposed set E, and add gear q coprime to
> the period. Within one lap, any two consecutive deleted points of E are at least q - 1 apart.
> The bound is tight: it is attained at q = 13 and q = 19.

In the k-frame (slot k = (6k-1, 6k+1), teeth at +-6^{-1} mod q) the same argument gives the
corresponding (q +- 1)/3-scale spacing (the project's law list records it as "deletion spacing
(q+-1)/3"); the adjacent-frame statement above is the proved and measured primary form.

## 2. WHY IT MIGHT BE NOVEL

The statement is elementary, so the honest question is whether it is already in the literature.
Its classical shadow is real: in the ordinary one-residue-class sieve (Holt & Rudd's cycle of
gaps), the elements removed at stage p are exactly the p * (old generators), so two removals are
trivially >= 2p apart - a spacing lemma exists there and is stated (their Lemma 3.1, "the minimum
distance between closures is 2*p_{k+1}"). What is different here: with TWO residue classes
deleted per lap ({phi, phi+1} mod q in the adjacent frame), consecutive deletions are no longer
automatically multiples of q apart - differences of 0 or +-1 mod q are all possible, runs of
consecutive deletions (chains, up to 6 literal members) actually occur, and the bound drops from
2q to q - 1 and needs the small mod-q case analysis rather than being definitional. The tightness
(q - 1 attained) has no counterpart found.

## 3. PROOF

Status: **PROVED (elementary, three-line case analysis); tightness SCRIPT-VERIFIED.**

Proof (from `docs/gear-recursion.md` section 4). Within one lap the deleted points lie in two
residue classes {phi, phi+1} mod q, so any two deleted points differ by 0 or +-1 mod q. Old gaps
are at least 3, so two distinct exposed points differ by at least 3. A difference delta >= 3 with
delta = 0 mod q gives delta >= q; with delta = 1 mod q gives delta >= q + 1 (delta = 1 itself is
excluded); with delta = -1 mod q gives delta >= q - 1. So delta >= q - 1. QED

Pointers:

- Statement, proof, and tightness data: `docs/gear-recursion.md` section 4. Measured minimum
  spacings against the bound q - 1: 12 vs 12 (q = 13), 18 vs 16 (q = 17), 18 vs 18 (q = 19),
  24 vs 22 (q = 23) - attained at q = 13 and q = 19.
- Script: `research/gear_recursion.py` (builds the transform, measures the per-lap deletion
  spacings; the docstring carries the same proof).
- The k = 2 case of the chain condition reproduces the lemma: a single interior gap must be 0 or
  +-1 mod q and at least 3, hence at least q - 1 (`docs/gear-recursion.md` section 4a).
- Not formalised in Lean.

## 4. IMPLICATIONS

Inside the project: a stretch of length G contains at most 1 + G/(q-1) deletions, so a chain of
k deletions needs span >= (k-1)(q-1) - the source of the span law (span >= floor((k-1)/2)*q in
the k-frame law list), of "long merges are rare", and one half of the failed-but-instructive
mechanical bound on chain depth k recorded in `docs/gear-recursion.md` section 4b. It is also the
k = 2 hinge of the saturation theorem (separate entry): when q - 1 > F(M) no interior gap can
qualify, so only k = 1 chains exist.

Outside: in any two-residue-classes-per-prime sieve (twin pairs, paired progressions, Polignac
classes), the same lemma bounds how densely one added prime can strike the surviving set - the
elementary reason the paired Jacobsthal function grows by O(q)-scale merges rather than
proportionally to its own size.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

- The increment law F(M+q') - F(M) <= alpha*q' (open link of the project's route to the twin
  prime conjecture): this lemma is why the increment is q-scale at all.
- Ziller-Morack Conjecture 6 / paired Jacobsthal growth: bounds the per-prime growth mechanism of
  h_2-type functions.
- No named classical problem is solved by it; it is infrastructure.

## 6. PRIOR-ART CHECK

Checked 2026-08-23. Searches run (Claude Code WebSearch; full-text reads via ar5iv) - the sweep
was shared with `merge-law.md`, all 12 recorded queries there; the ones decisive for this entry:

- "Holt Rudd 'cycle of gaps' Eratosthenes sieve primorial recursion copies closures", then
  full-text read of arXiv:1408.6002: their Lemma 3.1 proves "the minimum distance between
  closures is 2*p_{k+1}" in the ordinary sieve - the one-class analogue, trivial there because
  removed elements are p_{k+1} * (old generators).
- "'Jacobsthal function' recursion 'adding a prime' incremental computation g(P*q) from g(P)" and
  "'reduced residues' primorial 'maximal gap' structure 'adjacent gaps' merge OR merging" - no
  two-class spacing statement found.
- Full-text read of arXiv:1611.03310 (Ziller-Morack algorithms) and arXiv:2007.01808 (Ziller,
  gaps at primorials): no spacing lemma of either kind.
- "twin prime sieve '6k-1' '6k+1' pairs wheel maximal gap 'Jacobsthal' paired two residue classes
  per prime": nothing relevant beyond from-scratch sieve implementations.

Nearest published result: **F. B. Holt & H. Rudd**, "Eratosthenes sieve and the gaps between
primes", arXiv:1408.6002 (2014), Lemma 3.1 (closure spacing 2p in the one-residue-class cycle of
gaps).

**Verdict: PARTIAL OVERLAP** (Holt & Rudd Lemma 3.1). The delta: their spacing bound is 2p, holds
in the one-class sieve, and is definitional (removed points are multiples of p); the present
lemma is for the paired two-teeth frame, where consecutive deletions may differ by 0 or +-1 mod q
and genuine runs of deletions occur, the constant drops to q - 1, the proof is a mod-q case
analysis on a set with minimum gap 3, and tightness (attained at q = 13, 19) is established. The
two-class q - 1 bound itself was not found anywhere: NOVEL AS FAR AS SEARCHED within the paired
frame, but as a lemma type it must be called an adaptation of a known one-class fact.
