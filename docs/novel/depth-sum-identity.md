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

Status: PROVED (one line) + SCRIPT-VERIFIED (finite, exact).

Proof: every ordered pair of openings (r, r+g) is the endpoint pair of
exactly one window - the one spanning the j gaps between them (j <= g since
gaps are >= 1); conversely every j-window with sum g is such a pair. The RHS
counts the pairs by CRT. QED.

Verification: research/depth_identity.py - machines 11, 13, 17, 19, 23, 29
(periods 385 to 1,078,282,205; 214.7M openings at machine 29), all g = 1..64,
integer-exact, zero mismatches; the closed-form c_q is asserted against brute
force at every use. Kernel-checkable: for a fixed machine this is a finite
statement (a candidate for the Lean ledger; the bridge merged_eq in
proofs/Spectrum.lean already treats window sums).

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

Not yet checked (agent without web access). Nearest expected shadows:
renewal-theory neighbour-order identities; Montgomery-Soundararajan pair
correlations of sieved sets. The delta to check: the closed-form RHS and the
depth-uniform application to Jacobsthal-type spectra.
