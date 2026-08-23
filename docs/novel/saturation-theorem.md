# saturation-theorem - if q - 1 > F(M) then F(M+q) = F2(M) exactly

## 1. WHAT IT IS

Plain language. F(M) is the machine's largest gap between openings (the paired Jacobsthal object
in the 6k+-1 frame), and F2(M) is the largest sum of two ADJACENT gaps. When the added gear q is
large relative to the machine - precisely, when q - 1 exceeds the largest gap - the new record
gap is not merely bounded: it is determined exactly. Every sufficiently large added gear produces
the SAME new record, F2(M), read off the old machine. So above this threshold the increment
F(M+q) - F(M) = F2(M) - F(M) does not depend on q at all; "increment ~ q" is a statement about
the small-gear regime only.

Precise form (either frame; adjacent-frame quantities are 3x k-frame).

> **Theorem (saturation).** Let M be a machine with maximum gap F(M), and let q be a prime
> coprime to M's period with q - 1 > F(M). Then
>
>     F(M+q) = F2(M),
>
> where F2(M) is the maximum over the old gap word of the sum of two adjacent gaps.

Sharpness of the regime: along the consecutive chain (q' = next prime after y) one always has
q' < F(M), so the theorem covers far-gear additions only - the compliant and the needed regimes
are disjoint (`docs/proof-search/constructor.md` R15).

## 2. WHY IT MIGHT BE NOVEL

Two classical shadows must be dismissed honestly. First, F(M+q) >= F2(M) always (a k = 1 chain
needs no interior structure) - that direction is elementary and implicit in any merging picture,
including Holt & Rudd's one-class cycle of gaps. Second, "a large modulus factor changes little"
is a familiar heuristic. The content here is the exact EQUALITY with an explicit, best-possible
threshold: above q - 1 > F(M) no chain of two or more deletions can exist (its interior gap would
have to be 0 or +-1 mod q and at least 3, hence at least q - 1 > F(M), impossible), while every
adjacent pair of old gaps IS merged somewhere (each opening is deleted in some lap), so the
maximum saturates at F2(M) exactly. Searched literature contains no statement, in either the
ordinary or the paired frame, computing the Jacobsthal value of P*q exactly from P-level data
under any threshold on q - Holt & Rudd state no maximal-gap results at all, and the computational
literature (Hagedorn, Costello-Watts, Ziller-Morack) treats each modulus from scratch. The
corollary is also not a restatement of anything found: for the gears up to 7, adding 11, 13, 17,
19, 23, 29, 37, 41 or 53 all give F = 21 - the increment is q-independent above the threshold.

## 3. PROOF

Status: **PROVED (elementary, from the chain condition and the deletion-spacing lemma);
SCRIPT-VERIFIED over 48 pairs with zero violations. Not kernel-checked.**

Proof (from `docs/gear-recursion.md` section 4b). A chain with k >= 2 deletions needs an interior
gap that is 0 or +-1 mod q and at least 3, hence at least q - 1 by the deletion-spacing lemma
(`deletion-spacing.md`). No gap of M reaches q - 1, so only k = 1 chains exist. Every k = 1 chain
(a single deleted opening, merging its two adjacent gaps) is realised - each opening of M is
deleted in some lap of the merge transform - so the new maximum is the maximum over single
deletions, which is F2(M). QED

Pointers:

- Statement and proof: `docs/gear-recursion.md` section 4b; restated with the regime analysis as
  R15 in `docs/proof-search/constructor.md`.
- Finite verification: checked over 48 (M, q) pairs with zero violations
  (`docs/gear-recursion.md` section 4b; script lineage `research/gear_recursion.py`).
- The q-independence corollary, checked directly: machine {5, 7} plus any of q = 11, 13, 17, 19,
  23, 29, 37, 41, 53 gives F = 21, increment 6 every time (`docs/gear-recursion.md` section 4b).
- F2 is independently computable by search (`rust2/src/bin/holegap.rs`), validated against the
  pattern in all seven cases y = 7..29, so both sides of the equality have independent
  computations.
- Lean: not formalised. The adjacent kernel-checked material is `proofs/Spectrum.lean`
  (`merged_eq`, `merged_le_spectrum`: merged length is a window sum, F(M+q) <= F_{k_max+1}(M));
  the saturation equality itself and the deletion-spacing lemma it rests on are paper proofs
  only.

## 4. IMPLICATIONS

Inside the project: it corrected a wrong reading of the measured increment law - "increment ~ q"
is not a law about q but the small-gear regime (`docs/gear-recursion.md` section 4b); it makes
F2(M) the right second object of the recursion (F(M+q) >= F2(M) always, with equality in the
saturated regime), which drove the F2/holegap computations and the excess census; and it marks
the exact boundary of what mechanical (residue-free) arguments deliver: the same two inequalities
that prove saturation above the threshold are vacuous below it (constructor.md X10 - the
saturation regime is disjoint from the consecutive chain the route needs).

Outside: it is an exact evaluation of a Jacobsthal-type function on an infinite family of moduli
from finite data: for every prime q > F(M) + 1 coprime to the period, the paired Jacobsthal value
of M+q equals F2(M). The one-tooth analogue (with the same proof shape) would evaluate ordinary
g(P*q) for all large q from the gap word of P; as far as searched no such statement is published.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

- The project's increment-law route to the twin prime conjecture: negatively but usefully - the
  theorem proves alpha = 1 automatically in a regime PROVABLY disjoint from the one the route
  needs, sharpening where the difficulty lives (the q' < F(M) regime).
- Ziller-Morack Conjecture 6 and the paired Jacobsthal family: gives exact h_2-type values on
  sparse gear sets (primorial-over-q moduli) for free, a family the computational literature
  never touches.
- Jacobsthal computation (ordinary frame, via the one-tooth analogue): exact values g(P*q) for
  all q above the threshold, from one computation at P.

## 6. PRIOR-ART CHECK

Checked 2026-08-23. Searches run (Claude Code WebSearch; full-text reads via ar5iv) - sweep
shared with `merge-law.md` (all 12 queries recorded there); decisive for this entry:

- "'Jacobsthal' OR 'maximal gap' primorial 'two adjacent gaps' OR 'adjacent gap sum' threshold
  new prime larger than maximal gap" - nothing: results were the general bounds literature
  (Ford-Green-Konyagin-Maynard-Tao large prime gaps, Jacobsthal bound surveys); none states a
  threshold equality.
- "'Jacobsthal function' recursion 'adding a prime' incremental computation g(P*q) from g(P)" -
  no published relation between g(P*q) and P-level data of any kind.
- Full-text read of arXiv:1408.6002 (Holt & Rudd): recursion and closure-uniqueness proved, but
  explicitly **no formula for the maximal gap and no threshold condition** under which the new
  maximum equals the maximum adjacent-pair sum; their analysis is population dynamics of
  constellations.
- Full-text read of arXiv:2007.01808 (Ziller, gaps between coprimes to primorials): tabulates
  which gap values occur; nearest statement is Conjecture 4.1 (h(k-1) <= N_min(k)), not a
  maximal-gap evaluation.
- Full-text read of arXiv:1611.03310 and abstract of arXiv:1706.03668 (Ziller-Morack, ordinary
  and paired computations): from-scratch algorithms only.
- "mathoverflow OR mathstackexchange Jacobsthal function adding a prime modulus maximal gap
  coprime residues recursion" - nothing relevant.

Nearest published results: **F. B. Holt & H. Rudd**, arXiv:1408.6002 (2014) (the merging
framework in the one-class frame, without any maximal-gap statement); **M. Ziller**,
arXiv:2007.01808 (2020) (gap spectra of primorials, conjectural monotonicity of small gaps);
**M. Ziller & J. F. Morack**, arXiv:1706.03668 (2017) (paired Jacobsthal values computed to
prime 73, from scratch). Background bounds only: Jacobsthal (1960), Erdos (1962), Iwaniec
(1978), Hagedorn (2009), Costello-Watts (2015).

**Verdict: NOVEL AS FAR AS SEARCHED.** No published result was found that evaluates a
Jacobsthal-type function of P*q exactly from P-level data under a threshold on q, in either the
ordinary or the paired frame. Honest caveats: (i) the inequality direction F(M+q) >= F2(M) is
elementary and implicit in any gap-merging picture, so the novelty claim is confined to the exact
equality with its tight threshold; (ii) in the ordinary one-class frame the analogous statement
looks derivable from Holt & Rudd's machinery with modest effort - it is the statement, not the
depth, that appears to be new.
