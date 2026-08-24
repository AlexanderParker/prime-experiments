# The depth-sum identity: the window-sum family has a closed-form sum rule

## 1. WHAT IT IS

Plain language: in the twin-slot sieve machine (gears = primes 5..y, each
blocking two residues, the teeth, mod q), count how often a sum of j
consecutive gaps between openings equals g. Nobody has a formula for any
single depth j - these are the objects whose maxima are the Jacobsthal-type
spectrum F_j. But the SUM over all depths has an exact closed form: the
two-point correlation of the exposed set, which factorises over gears by CRT.

Precise form. Machine M with gears Q = {primes 5..y}, period P = prod Q,
openings = residues avoiding both teeth {u, -u} of every gear (u = 6^{-1} mod
q). Let W_j(g) = number of cyclic windows of j consecutive gaps whose sum is
g, and c_q(g) = #{r mod q : r and r+g both exposed to gear q} (closed form,
round 18: q-2 if q | g; q-3 if 3g = +-1 mod q; q-4 otherwise). Then for every
g >= 1:

    sum_{j >= 1} W_j(g)  =  prod_{q in Q} c_q(g).

Corollary (depth-uniform bound): W_j(g) <= prod_q c_q(g) for every j - an
upper bound on every window-sum count from CRT arithmetic alone, no period
scan, uniform in depth.

## 2. WHY IT MIGHT BE NOVEL

The identity's skeleton is the classical point-process fact "pair correlation
= sum over k-th nearest-neighbour distributions". The content here is (a) the
RHS is EXACTLY computable for the sieve set - a product of per-gear
three-case constants - so the whole window-sum family {W_j}, whose maxima
F_j = max{g : W_j(g) > 0} are the spectrum objects of Jacobsthal-type
problems, is constrained by one finite closed form; (b) the depth-uniform
bound W_j(g) <= prod c_q(g) costs nothing and applies at machines far beyond
any scan (the same arithmetic reaches y = 53+). We are not aware of the sum
rule being stated for Jacobsthal-type sieve structures or used to bound
window-sum counts.

## 3. PROOF

Status: PROVED (one line) + SCRIPT-VERIFIED (finite, exact) + BOTH HALVES
KERNEL-CHECKED at machine 13 (round 22, proofs/DepthSum.lean); the GLUE
between the halves is not.

Proof: every ordered pair of openings (r, r+g) is the endpoint pair of
exactly one window - the one spanning the j gaps between them (j <= g since
gaps are >= 1); conversely every j-window with sum g is such a pair. The RHS
counts the pairs by CRT. QED.

Verification: research/depth_identity.py - machines 11, 13, 17, 19, 23, 29
(periods 385 to 1,078,282,205; 214.7M openings at machine 29), all g = 1..64,
integer-exact, zero mismatches; the closed-form c_q is asserted against brute
force at every use. KERNEL-CHECKED (round 22, proofs/DepthSum.lean), in two halves:
* the "sum over j" half, ABSTRACT and machine-free -
  `DepthSum.window_depth_unique` (a window from a given start summing to g has
  a UNIQUE length, because a strictly increasing position sequence makes
  j |-> pos(a+j) - pos(a) injective) and `DepthSum.depth_partition`
  (sum_j W_j(g) = the count of starts reaching g at SOME depth; the per-depth
  sets are pairwise disjoint).  This is exactly the one-line proof's bijection.
* the `prod_q c_q(g)` half at machine 13 - `DepthSum.depth_sum_at_13`: over
  the whole 5005-slot period, the lag-g opening-pair count EQUALS
  c_5 c_7 c_11 c_13 for every g < 40, and `DepthSum.depth_sum_hl_form` puts
  it in Harvester's Hardy-Littlewood form prod_q (q - nu_q({0,2,6g,6g+2})).
  Both depend on NO AXIOMS AT ALL (pure kernel computation).
NOT done: the glue.  Turning `depth_partition` into a statement about the
period pair-count needs "count over one period of the enumeration = count
over residues" - a periodicity / re-indexing bridge for `Machine13.opSeq`
that round 22 did not build.

## 4. IMPLICATIONS

Inside the project: (i) gives every prefix row the missing UPPER-bound side
in closed form (complements the coverability spectrum COV(M) the Mechanic is
building); (ii) it is the exact ground on which the round-20 renewal
decomposition stands: W_1(g) = [closed form] x [renewal factor], isolating
the interior disjunction as the single unexplained object; (iii) it is an
exact validation sum rule for every census (a full-period gap histogram plus
window sums must hit the closed form exactly - a bug detector).

Outside: a finite, elementary, exact constraint family on Jacobsthal-type
gap structures; possibly useful wherever window sums of sieve gaps are
studied (maximal prime gaps in sieved sets, Ziller-Morack-style computations).

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

Jacobsthal's function (F = F_1's maximiser is the 2-dimensional analogue);
the project's requirement (D) (window sums at qualifying sizes); the
suppression law (round 19) whose p_j objects are ratios of W-type counts.

## 6. PRIOR-ART CHECK

VERDICT: KNOWN (published). Checked 2026-08-24 by Harvester; recorded by the
manager, since the finding belongs to another lane's sweep.

This identity is **Holt, arXiv:2502.20470 (February 2025), Corollary 1**,
specialised to one constellation. Holt's statement: for an admissible
constellation s of length J,

    sum_{j >= J} n_{s,j}(p#) = prod_{q <= p} (q - nu_q(s)).

A twin-slot survivor is a gap of 2 in Holt's cycle, so a twin-slot pair at lag
g is his constellation (2, 6g-2, 2), with boundary points {0, 2, 6g, 6g+2}.
Under that correspondence sum_j W_j(g) = prod_q c_q(g) IS his Corollary 1.
Verified by assertion, not argued: research/holt_correspondence.py checks the
correspondence exactly and confirms N2(g) equals Holt's RHS to the unit.

What this does and does not change:
- The identity is TRUE and the derivation here is sound; only the novelty
  label was wrong. It was derived independently, which is worth recording.
- The kernel check of it (Formalist, DepthSum.lean) is unaffected as
  VERIFICATION - a machine-checked proof of a known theorem is still a
  machine-checked proof, and it is the form the project's other lanes consume.
- The objects still separate in general: at machine 17, g = 5, this project's
  n_g = 4,230 against Holt's n_{s,J} = 0, so the correspondence is a
  specialisation, not an identification of the whole framework.

STANDING LESSON, recorded here because this is where it was learned:
PRIOR-ART CHECKS EXPIRE. Holt 2502.20470 postdates the round-20 and round-21
sweeps, so those sweeps were not careless - they were correct when run. Both
novelty downgrades of round 22 came from material that either postdated the
last sweep (this one) or existed but was unread (Ziller-Morack's ancillary
files, 2017). Any entry in this register whose verdict predates a new arXiv
listing in its area should be re-checked before publication, not before
citation.
