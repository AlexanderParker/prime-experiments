# The spectrum bound on L: the longest legal word is O(F/q'), not O(1)

Established: Lateral, round 31 (2026-09-04).  Status: **PROVED (paper), SCRIPT-VERIFIED**
at 12 corpus machines and 165,584 counterfactual machines.  Prior art: **not yet checked.**

## 1. WHAT IT IS

Plain language.  The gear machine merges a gap when the incoming gear deletes a run of
consecutive openings.  The run is described by a *word*: the gaps between the deleted
openings, each of which must be 0 or +-d' modulo the new gear q', with the two nonzero
classes strictly alternating.  L(M) is the length of the longest such word that is
actually realised in M.  The project's open hypothesis (B) asked whether L(M) is bounded
by an absolute constant.  This note proves it is bounded by the machine's own record gap
divided by the new gear - and that this is the right shape, because L is *attained* at
the bound on the smallest machines and on the counterfactual family.

Precise form.  Let M = {5, 7, ..., y} be a machine, q' = nextprime(y), u the small
representative of 6^{-1} mod q', d' = 2u, a = d', b = q' - a (so a + b = q' and
3a = q' -+ 1), a_min = min(a, b) = a, and G = F(M + q') the record gap of the next
machine.  Write a realised legal word as m letters, of which p are *padded* (0 mod q')
and n = m - p are *nonzero* (+-d' mod q').  Then, with T = floor((G - 2)/q'),

    (SIMPLE)   L(M) <= 2T + 1,          and, letter-aware,   L(M) <= 2T + 1 - p
    (PARITY)   L(M) <= max( 2T,  2*floor((G - 2 - a_min)/q') + 1 )

In particular  **L(M) <= 2 F(M+q')/q' + 1**.

## 2. WHY IT MIGHT BE NOVEL

The object L is itself project-internal: it is the arity of the deletion event that
carries a maximal gap from one primorial sieve to the next, and it equals A_kill - 1 and
J_max - 2 (kernel-checked, `proofs/WordLegal.lean`, R89).  What the bound does is convert
a *combinatorial* quantity (how many consecutive openings one gear can delete in a row)
into a *metric* one (the record gap divided by the gear), using only the fact that the
two nonzero residue classes alternate and that their smallest representatives sum to
exactly q'.  The "sum to exactly q'" step is special to the twin-tooth pair
{+6^{-1}, -6^{-1}}: the pair of legal gap classes is +-2*6^{-1}, and those two smallest
values are complementary mod q' by construction, not by accident.

It is NOT a restatement of the classical Jacobsthal-gap machinery: the classical bounds
concern the size of gaps, not the arity of the merge that produces them.  It is also not
a restatement of the project's own exposure bound (Constructor R100's EXPCAP, "exposure at
word length m is decided by the gears <= 2m + 2"): EXPCAP is a phase-saturation statement
about small gears and it exceeds the true L by 16 and 18 at m37 and m53, where the
spectrum bound gives 5 at both.

## 3. PROOF

Three ingredients.

(i) CLASS MINIMA.  The smallest positive integer congruent to +d' mod q' is a, to -d' is
b, and to 0 is q'.  (Gated at all twelve corpus machines; immediate from 0 < a, b < q'.)

(ii) ALTERNATION.  T3 (the tooth-transition condition; kernel-checked as part of R89's
word reduction) says the nonzero-class letters strictly alternate, padded letters being
transparent.  Hence in any legal word, consecutive nonzero letters lie in opposite
classes, so any two consecutive nonzero letters sum to at least a + b = q'.  Pairing the
n nonzero letters in order,

    sum of the nonzero letters  >=  floor(n/2) * q'  +  [n odd] * a_min,

and each of the p padded letters is >= q'.  Therefore

    span(word)  >=  p*q'  +  floor(n/2)*q'  +  [n odd]*a_min.                (*)

(iii) THE ATTAINMENT THEOREM (R68, proved; Constructor R68 = R46's Kleene-star identity,
verified exactly at eight steps).  *If consecutive openings x_0 < ... < x_J of M have a
legal middle-gap word then x_J - x_0 <= F(M + q').*  A realised legal word of m letters
occupies m+1 consecutive openings; take one further opening on each side to get the
window x_0 < ... < x_{m+2}, whose m middle gaps are exactly the word.  Hence

    before + span(word) + after  <=  G,     before, after >= 1,
    span(word)  <=  G - 2.                                                   (**)

Combining (*) and (**): p + floor(n/2) <= (G - 2)/q', and since the left side is an
integer, p + floor(n/2) <= T.  Then

    L = n + p  <=  2*floor(n/2) + 1 + p  <=  2(p + floor(n/2)) + 1  <=  2T + 1,

and keeping p explicit, L <= 2T + 1 - p.  For (PARITY): if n is even, (*) gives
p + n/2 <= T and L = n + p <= 2T; if n is odd, (*) gives
p + (n-1)/2 <= (G - 2 - a_min)/q' and L <= 2*floor((G - 2 - a_min)/q') + 1. []

STATUS.  Unconditional given R68 (proved) and T3 (kernel-checked).  No use is made of the
cover half of the realisability CSP, of phase saturation, or of any property of the gears
of M beyond "openings are distinct integers".

SCRIPT VERIFICATION.  `research/lateral_r31.py corpus` (173 assertion gates) checks the
class minima, 3a = q' -+ 1, the bound against the measured L at all twelve corpus
machines, the span accounting (*) and (**) directly on every realised word on record, and
the comparisons with EXPCAP and with the coarser G/a_min form.  `research/lateral_r31.py
family` (22 gates) checks (SIMPLE), (PARITY) and the padded-aware form at every one of
165,584 rows of the tooth-counterfactual family - zero violations, including the family's
L = 5 member, where (PARITY) equals 5 exactly.

## 4. IMPLICATIONS

Inside the project.  Hypothesis (B) - "L(M) <= c_B for an absolute constant c_B" - was
the one genuinely open uniform statement in Constructor's implication chain R99, and
Lateral's round 30 showed it is not structural (the counterfactual family reaches L = 5
where the real machine has 2).  The bound replaces (B) with a *theorem* and, because it
is linear in G = F(M + q'), it can be substituted back into R99 without circularity:

    G <= F_2(M) + c_A L   and   L <= 2(G-2)/q' + 1
        ==>  G <= ( q'(F_2 + c_A) - 4 c_A ) / ( q' - 2 c_A )      for q' > 2 c_A,

which for c_A = 4 delivers the increment inequality (D), F(M+q') <= F(M) + q', whenever

    8 F(M)  <=  q'^2 - (F_2(M) - F(M) + 12) q' + 16.

Measured at the corpus, this holds at 8 of the 13 steps m11..m59 - failing only at the
five small ones, where q' is too close to 2 c_A - and the margin grows: F divided by the
right-hand side falls from 0.87 at m41 to 0.57 at m59.  So the uniform obligation moves
from a combinatorial constant to a *Jacobsthal-square condition* F(M) <~ q'^2/8, with the
corpus sitting at F/q'^2 = 0.038 .. 0.052 throughout.

The caveat that must travel with this.  c_A = 4 is measured on LITERAL letters only; the
padded letter at m31 has eps = -17 (Constructor R101/C6, the F_3-wall event), and with
c_A = 17 the closure needs q' > 34 and is vacuous where it applies.  The closure is
therefore conditional on (A) holding with a small constant over the whole chain, which is
exactly Constructor's open (A-pad).

Outside it.  The statement "the number of consecutive openings a single new prime can
delete from a primorial sieve is at most 2 F/q' + 1" is a clean, elementary constraint on
how maximal gaps in the reduced residue system can grow one prime at a time.  Its shape
(arity <= 2 * record / modulus) is the exact opposite of the intuition that a merge could
recruit arbitrarily many deletions.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

- Twin primes via the gear machine: (D) at every step is the project's target, and this
  bound removes one of its four hypotheses (B) outright.
- Jacobsthal's function: the residual condition 8 F(M) <= q'^2 - O(q') is of the same
  order as the best known upper bounds for the Jacobsthal function of a primorial (which
  are quadratic in y), but needs an explicit constant below 1/8; measured, the corpus is
  a factor 2.4-3.3 inside it.  Any explicit quadratic Jacobsthal bound with a good
  constant would close the gap.
- Polignac / gap-arity: the bound is an upper bound on the arity of a one-prime merge and
  is TIGHT at m11, m13 and m29, and at 13-87% of counterfactual machines - so it is not
  merely an upper estimate, it is attained.

## 6. PRIOR-ART CHECK

**Not yet checked** (agent has no web access).  Terms a check should try: "Jacobsthal
function primorial consecutive deletions", "maximal gap reduced residue system one prime
at a time", "arity of gap merge sieve", "runs of consecutive totatives deleted by a
prime", Holt & Rudd 2014 (already read in-project: their Lemma 3.1 is the one-class
analogue of the deletion spacing, and their results stop where multi-class arity begins),
Ziller 2020, Ford-Green-Konyagin-Maynard-Tao (Jacobsthal lower bounds), Iwaniec 1978
(g(n) << log^2 n).  Expected verdict shape: the *ingredients* (alternating residue
classes summing to the modulus; a maximal-gap upper bound on a spanned window) are
elementary, so PARTIAL OVERLAP is plausible; the *statement about arity* has no obvious
classical shadow.
