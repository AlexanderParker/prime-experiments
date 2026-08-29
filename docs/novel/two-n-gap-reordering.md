# The 2n-gap reordering (human's sort-step idea) - PROVED, AND PRICED

Status changed 2026-08-29 (Lateral, round 27): the law is now PROVED and
generalised, the cyclic closure is proved free, and the framing's promise is
REFUTED by a gated invariance test. Prior art checked and recorded.

## 1. WHAT IT IS

Sort the machine's openings not by position but by PHASE VECTOR (CRT-lex: key =
(k mod q_1, ..., k mod q_n), compared lexicographically). In that order the
adjacent differences (mod P) take EXACTLY 2n DISTINCT VALUES at n gears - 6, 8,
10, 12, 14 at machines [5,7,11] through [5..23]. Natural (positional) order gives
4, 7, 10, 17, 23 at the same machines: irregular, and it is the gap spectrum
whose maximum is F.

Origin: the human's suggestion (2026-08-31) that the sliding windows, viewed in a
suitably SORTED order (a "sort step" per turn), should have an obvious gap
pattern. Manager probe confirmed it on contact; routed to Lateral for round 27.

PRECISE FORM. Let q_1, ..., q_n be the gears (in the order used to build the
key), A_i = Z_{q_i} \ {+-6^{-1} mod q_i} the exposed set (|A_i| = q_i - 2 =: m_i)
written as an increasing list a_i^{(0)} < ... < a_i^{(m_i - 1)} of integers in
[0, q_i), P = prod q_i, N = prod m_i. Put

    w_i = a_i^{(0)} - a_i^{(m_i-1)} mod q_i = -max(A_i) mod q_i     (the wrap)
    D(i, d) = the CRT element that is 0 mod q_{i'} for i' < i,
                                     d mod q_i,
                                     w_{i'} mod q_{i'} for i' > i.

THEOREM (2n law, with values and multiplicities). The cyclic sequence of adjacent
differences of the CRT-lex-ordered opening set takes exactly the 2n values
D(i, 1) and D(i, 2), i = 1..n, and

    mult(D(i, d)) = s_i(d) * prod_{i' < i} m_{i'},
    s_i(1) = q_i - 3 - s_i(2),   s_i(2) = 1 if q_i in {5,7}, else 2.

## 2. THE PROOF

Under CRT the opening set is the product set A_1 x ... x A_n, and lex order on
phase vectors is exactly the MIXED-RADIX ODOMETER on the digit vector
(j_1, ..., j_n), j_i indexing A_i, with j_n least significant. The lex successor
increments the last non-maximal digit i and resets every digit below it. So the
phase-vector difference is

    0 in coordinates i' < i;   delta = a_i^{(j_i+1)} - a_i^{(j_i)} in coordinate i;
    w_{i'} in coordinates i' > i,

i.e. exactly D(i, delta). Since the coordinates below i are 0 and delta != 0, the
carry position i is RECOVERABLE from the difference, so distinct (i, delta) give
distinct elements of Z_P and

    #distinct differences = sum_i d_i,
    d_i = #distinct consecutive differences of the sorted set A_i.        (*)

STEP-TYPE LAW for d_i (proved, and gated on 400 random removals). If a gear
removes a set T from Z_q, the sorted survivors' consecutive differences are
exactly {L+1 : L an INTERIOR maximal run of T} together with 1 whenever two
survivors are adjacent, where "interior" means the run contains neither 0 nor
q-1 (a run at either end of the interval has no survivor on one side and
generates no difference).

For the machine, T = {u_q, -u_q} with u_q = 6^{-1} mod q. The two teeth are never
adjacent - adjacency needs 2u = +-1, i.e. (since 6u = 1) 3 = +-1 mod q, i.e.
q | 2 or q | 4, impossible for q >= 5 - and 0 is never a tooth. So every
interior run has length 1, giving the value 2, and 1-steps survive because
q - 3 - s_i(2) >= 1. Hence d_i = 2 for every gear and the count is 2n. []

THE CYCLIC CLOSURE IS FREE (this is a fact about the machine, not about product
sets in general). The last-to-first difference is CRT(w_1, ..., w_n). Its
coordinate 1 is w_1 = -max(A_1) mod q_1, which is 1 when q_1 - 1 is exposed and 2
when q_1 - 1 is a tooth (q_1 in {5,7}); either way w_1 in {1,2}, so the wrap
difference IS D(1, w_1) and is already counted. So the linear and cyclic distinct
counts are both 2n. (For a general two-point sieve the wrap can be a 2n+1-st
value; for the machine's own teeth it never is.)

ORDER-INDEPENDENCE. d_i depends only on A_i, so by (*) the COUNT is the same for
every one of the n! orderings of the gears used to build the key - although the
2n VALUES are different for different orderings.

THE ODOMETER / DIGITAL-SEQUENCE FORM. Writing E_i for the CRT idempotent of gear
i (E_i = 1 mod q_i, 0 mod the others), the lex enumeration has the closed form

    Phi(t) = sum_i a_i^{(j_i(t))} E_i  (mod P),

with (j_i(t)) the mixed-radix digits of t in radices (m_1, ..., m_n). Phi is an
explicit bijection [0, N) -> O: a generalised van der Corput / Halton-type
DIGITAL SEQUENCE, in which the machine's record gap is exactly P times the
sequence's DISPERSION (largest empty interval).

STATUS: PROVED (elementary), and SCRIPT-VERIFIED exactly by
research/lex_odometer.py (parts A-H, 145 assertion gates, exit 0, log
research/data/lex_odometer.log): the law, the closed-form value set, the
multiplicity table, the free wrap, all n! orderings at n <= 4 plus samples at
n = 5, 6, the step-type law on 400 random removals, and the Phi bijection.

## 3. WHY IT MIGHT BE NOVEL - AND WHY IT IS PROBABLY NOT

The phenomenon "lex order on a product/lattice makes the successor difference set
tiny" is CLASSICAL. Langevin's theorem (recovering the three-distance and
three-gap theorems) says that for a lattice L in R^2 with the lexicographic
order there is a basis u, v such that the lex successor of any lattice point in
an interval is one of w+u, w+v, w+u+v - three values, by exactly the carry
argument used above. Fried and Sos generalise it to ordered abelian groups. Our
statement is the finite, n-factor, CRT version of the same mechanism.

What is NOT in that literature, as far as searched, is the specialisation to
SIEVE OPENING SETS with its exact multiplicity table, the free cyclic closure,
and - the part that actually matters - section 5's deflation.

## 4. IMPLICATIONS

POSITIVE. (i) An exact, sieve-free, O(n)-memory enumeration of the opening set at
ANY machine, in phase order, with O(1) work per opening and 2n precomputed
strides - the natural streaming enumerator for phase-ordered censuses.
(ii) The whole opening set is specified by 3n integers: the n radices m_i and the
2n strides D(i, d). (iii) The machine is exhibited as a digital sequence, so
Jacobsthal's function is literally the dispersion of an explicit generalised van
der Corput sequence.

NEGATIVE, AND IT IS THE HONEST HEADLINE (section 5).

## 5. THE DEFLATION - THE 2n COUNT IS BLIND TO F (gated)

By (*) the count depends on each gear ONLY through "how many distinct interior
run lengths does the removed set have". For a two-point removal that number is 1
in every case except the degenerate terminal pair {q-2, q-1} (a fact about where
one cuts the cycle, not about the sieve). So:

- Re-choosing the teeth at every gear, over 60 admissible tooth vectors at
  mods [5,7,11,13], leaves the distinct-difference count at exactly 2n = 8 EVERY
  TIME, while F ranges over [10, 18] - a factor of 1.8.
- The law does not even need primes: coprime non-prime moduli [8,9,25] with
  two-point removals give 2n = 6 as well.

THEREFORE the 2n law is a coordinate/labelling fact that is invariant under
everything that moves F. The framing "the machine = a trivial product order x an
arithmetic shuffle, and every hard question is a property of the shuffle alone"
is TRUE but empty in the direction that matters: the trivial side is trivial for
reasons independent of the arithmetic, and F is not a statistic of the ORDER
permutation at all (it is a statistic of the VALUES Phi(t), i.e. of the metric,
which the permutation discards). The dual measurement makes the same point: the
number of distinct LEX-INDEX displacements between NATURAL-order neighbours is
5, 25, 95, 368, 1362 at n = 2..6 - it grows, and the complexity has simply been
moved, not reduced.

## 6. UNSOLVED QUESTIONS IT TOUCHES

Three-distance / three-gap theorems for products (Langevin, Fried-Sos,
Chevallier's cyclic-group version, the adelic three-gap theorem); dispersion of
digital / Halton-type sequences in quasi-Monte Carlo. It gives Jacobsthal's
function a QMC-dispersion phrasing, but the QMC dispersion machinery in dimension
one reduces to gap counting, so nothing is imported by it (checked, recorded as a
non-gain rather than a route).

## 7. PRIOR-ART CHECK

Checked 2026-08-29 (Lateral, round 27; web search). Terms run: "three-distance
theorem generalization product set lexicographic order gaps Chinese remainder";
"mixed-radix odometer successor differences number of distinct gaps digital
sequence"; "Langevin lexicographic order lattice three points successor Fried Sos
ordered abelian groups"; "gaps reduced residue system primorial ordered by
residue vector Jacobsthal function"; "dispersion of Halton sequence largest gap
one-dimensional digital net lower bound".

Nearest published results:
- M. Langevin's lexicographic-successor theorem for planar lattices (three
  values), and its recovery of the three-distance and three-gap theorems; E.
  Fried and V. T. Sos's generalisation to ordered abelian groups - reported in
  N. Chevallier, "Cyclic groups and the three distance theorem"
  (http://www.math.uha.fr/chevallier/publication/canadianJM.pdf).
- V. Berthe and C. Reutenauer, "On the three distance theorem"
  (https://www.irif.fr/~berthe/Articles/Intelligencer.pdf); the three-gap theorem
  (https://en.wikipedia.org/wiki/Three-gap_theorem); "A three gap theorem for the
  adeles" (https://arxiv.org/pdf/2107.05147).
- Jacobsthal-function literature on gaps in reduced residue systems of primorials
  (Hagedorn; https://arxiv.org/pdf/1611.03310) - which orders the coprimes
  POSITIONALLY, never by residue vector.

VERDICT: KNOWN IN MECHANISM / PARTIAL OVERLAP. The carry-argument phenomenon is
classical (Langevin; Fried-Sos), and the finite CRT version is folklore-grade -
"an afternoon theorem", exactly as the first probe guessed. The DELTA that is
novel as far as searched is small and technical: the exact multiplicity table,
the free cyclic closure for the machine's own teeth, and the order-independence.
The DELTA that was hoped for - a new handle on the record gap - does not exist,
and section 5 proves it does not. Recorded as a closed line, not a route.
