# moment-degree-ceiling - every fixed-degree covering certificate for F(M) dies at a computable machine, and the degree a certificate needs grows like 4 log log y

Status: SCRIPT-VERIFIED with EXACT rational arithmetic, both directions
(`research/lp_dual_certs.py`, sections C and D; exact LP core
`research/exact_lp.py`).  The vacuity verdicts are proved by an exact
rational feasible point; the "still bites" verdicts by an exact Farkas
vector verified pointwise over all 2^n - 1 atoms.  Established round 22
(LP-duality dedicated explorer).  Prior-art check: PARTIAL OVERLAP, done
2026-08-24, section 6.

This is a NEGATIVE result with a proof: it closes a route with a reason.
It is the obstruction half of `covering-lp-certificates.md`, whose
constructive half (scan-free Farkas certificates for F(M) <= W) it bounds
from above.

ROUND-23 AMENDMENT - THE CEILING IS NOT THE OPERATIVE LIMIT, AND IT DOES NOT
BIND THE COSTELLO-WATTS FAMILY.  Two corrections to how this entry was read.

(1) THE CEILING IS THE WRONG THRESHOLD FOR THE (D) APPLICATION.  The ceiling
    asks when a degree-l certificate can prove ANYTHING.  A (D) rung landing
    at machine y needs a certificate of width exactly B(y) = F(prev) + y, i.e.
    an integrality gap no worse than B(y)/F(y).  Measured exactly at the steps
    7->11 ... 37->41: 2.29, 1.82, 1.56, 1.48, 1.41, 1.47, 1.28, 1.08, 1.42.
    After the first step it never again exceeds 1.48 and it dips to 1.08 at
    31->37 - it is NOT monotone, but it is asymptotically 1, since
    B(y)/F(y) = 1 + (y - (F(y) - F(prev)))/F(y) and y/F(y) -> 0.  So the
    certificate must be near-tight at every step, and the rung-proving range
    ends FAR below the vacuity ceiling.  Degree 2 is vacuous only from machine
    29, but its rung-proving range ends much earlier.
(2) THE CEILING SAYS NOTHING ABOUT RECURSIVE BOUNDS.  It is a theorem about
    relaxations that keep the joint distribution of at most l gears at a time.
    Costello-Watts (arXiv:1208.5342, read in full round 23) is not of that
    form: its pair term is the EXACT survivor count of a smaller machine,
    obtained by recursion, so its effective degree is unbounded.  The ceiling
    therefore does not apply to it, and the two results are compatible - the
    ceiling closes the fixed-degree LP route and Costello-Watts is the escape
    hatch from it.  Transferred to the two-teeth machine and measured
    (`research/cw_transfer.py`), that escape hatch is nevertheless FAR weaker
    than the dual certificate on this problem: F(13) <= 35, F(17) <= 65,
    F(19) <= 110, F(23) <= 230, F(29) <= 322, i.e. 3.2x to 7.5x above the true
    F, where the dual certificate is at 1.14x-1.82x and proves the rungs.
(3) THE CEILING IS UNCHANGED BY MARGINAL CONSISTENCY (round 23,
    `consistency-over-degree.md`).  The uniform product measure is a global
    distribution, hence a feasible point of the CONSISTENT degree-l relaxation
    too, so every ceiling machine below is also the ceiling machine for the
    Sherali-Adams-style consistent hierarchy at that degree.  Consistency buys
    WIDTH, not MACHINES.

## 1. WHAT IT IS

Plain language.  F(M) is decided by a covering question - is there a window
of W consecutive slots in which every slot is blocked by some gear? - and
covering questions have LP relaxations whose dual solutions are checkable
certificates.  Every such relaxation that only knows about gears l at a
time (a "degree-l" relaxation) is throwing away information, and this
entry measures exactly how much: for each degree l there is a COMPUTABLE
machine beyond which the degree-l relaxation is satisfied by a trivial
solution at EVERY width, so it can prove nothing at all.  The killer is not
a clever configuration - it is the uniform product measure (every gear's
phase independent and uniform), and the reason is Mertens: the expected
number of gears blocking a slot, S1(y) = sum_{5<=q<=y} 2/q, diverges, so
low-order inclusion-exclusion overshoots 1 and every degree-l inequality is
satisfied with room to spare.

Precise form.  Machine M = gears (primes) 5 <= q <= y; gear q blocks slot k
iff k = +-6^{-1} (mod q); A_q(i) = "gear q blocks slot i".  By CRT, choosing
the window position is exactly choosing one phase per gear independently, so

    max coverable width  =  F(M) - 1        (EXACT, not a model).

A degree-l cut is any polynomial g of degree <= l in the indicators with
1{union A_q} <= g pointwise; the degree-l relaxation at width W keeps, per
l-subset of gears, a joint phase distribution, and imposes E[g] >= 1 at
every position for every degree-l cut g.  Any covered window is a feasible
integral point, so INFEASIBILITY certifies F(M) <= W.

CEILING TEST (the object).  The relaxation is vacuous at machine M - feasible
at every width, integrality gap infinite - as soon as the uniform product
measure satisfies every degree-l cut.  Because a cut is a pointwise
inequality, that happens exactly when the product measure's degree-<=l
coverage moments m_S = prod_{q in S} 2/q extend to SOME distribution nu on
{0,1}^gears with nu(empty atom) = 0.  That is a finite LP over the 2^n
atoms with sum_{j<=l} C(n,j) equality rows; its Farkas dual is precisely the
optimal degree-l Bonferroni cut, and the whole test is exact rational
arithmetic.

MEASURED CEILINGS (exact; `V` = vacuous from that machine on, integrality
gap infinite for the whole degree-l family):

    degree l   sharp ceiling machine (multivariate, exact)
      1        13        (S1 = 5112/5005 >= 1: the density bound dies)
      2        29        (bites at 23, vacuous at 29, 31, 37)
      3        >= 151    (bites at 37 directly; >= 151 because the
      4        >= 151     aggregated relaxation below still bites at 127)

    degree l   AGGREGATED ceiling machine (binomial moments only; a LOWER
               bound on the sharp one - the aggregated relaxation is weaker)
      1        13
      2        19
      3, 4     151        (bites at 127, vacuous at 151)
      5, 6     between 3000 and 5000
      7, 8     > 12000

THE DEGREE LAW.  At the aggregated ceiling the mean coverage S1 takes the
values 1.02 (l=1), 1.24 (l=2), 2.12 (l=3,4), 3.14 (l=5,6) - i.e. the degree a
certificate needs at machine y is

    l(y)  ~  2 * S1(y)  ~  4 log log y            (measured, exact points)

which is UNBOUNDED.  No fixed-degree covering certificate family works for
all machines; but the growth is doubly logarithmic, so a degree-10
certificate would still be reaching machines around y = 10^6.

THE CHAIN FAMILY IS EXPONENTIALLY WEAKER (exact closed form).  The chain /
Kounias cut family used in `covering-lp-certificates.md` has a telescoping
uniform-product slope

    s(chain)  =  S1 * prod_{k in chain} (1 - 2/q_k)  +  beta(chain),

where beta depends only on the chain, not on the machine - so the slope is
AFFINE in S1 and every fixed chain dies once S1 is large enough.  Because
beta >= 0 and (1 - 2/q) increases in q, s >= S1 * prod_{t smallest gears},
giving a rigorous death machine for EVERY depth-t chain:

    depth t:      0     1     2      3      4       5        6
    dead from y:  13    53    277    1553   13997   156131   > 10^6

Comparing with the sharp ceilings above: the chain family needs depth 5 to
survive where a general degree-6 cut survives to y ~ 5000, and it needs
prod(1-2/q) < 1/S1, i.e. the chain must reach z ~ exp(sqrt(2A log log y)) -
an EXPONENTIALLY higher degree than the sharp requirement 4 log log y.
The seed's "the moment degree needed grows with the machine" is right, but
the growth rate it exhibits is an artefact of the chain family, not of the
problem.

## 2. WHY IT MIGHT BE NOVEL

- The classical shadow is explicit and should be stated first: this IS the
  covering-dual form of the reason Brun's pure sieve must let its truncation
  level grow with the sieve dimension.  S1 ~ 2 log log y diverges, so a fixed
  Bonferroni truncation overshoots; both here and there the fix is a
  truncation level growing like log log.  Nothing in the MECHANISM is new.
- What does not appear in the literature searched is the QUANTIFIED,
  per-machine, exact version for the Jacobsthal-type covering problem: an
  exact ceiling machine per degree, computed as the feasibility of a finite
  rational LP, with the Farkas dual doubling as the optimal cut; and the
  separation between the sharp ceiling and a named cut family's ceiling.
- The reading it gives the project is the part with no analogue anywhere: it
  converts "how much correlation depth does (D) need?" into a number that can
  be computed at any machine before any search is run.

## 3. PROOF

SCRIPT-VERIFIED, exact: `research/lp_dual_certs.py` sections C and D
(`uv run python research/lp_dual_certs.py C D`), exact LP core
`research/exact_lp.py` (two-phase rational simplex, Bland's rule; its Farkas
extraction is itself checked on 400 random systems with exact verification of
every certificate).

Validity chain.
(a) IP exactness: CRT realises every phase tuple, so max coverable width =
    F(M) - 1.  Asserted against period sieves at machines 7..19.
(b) Relaxation validity: a covered window gives an integral feasible point,
    so LP infeasibility at W implies F(M) <= W.
(c) Vacuity: if the uniform product measure's degree-<=l moments admit a
    completion nu with nu(empty) = 0, then for every degree-l cut g,
    E_product[g] = E_nu[g] >= E_nu[1{union}] = 1, so every constraint holds
    at every position and every width.  The completion is exhibited as an
    exact rational vector and all its moments are asserted equal.
(d) "Still bites": the Farkas vector lam satisfies lam.m > 0 and
    sum_{S subset x} lam_S <= 0 for every nonempty atom x - both asserted
    exactly, the second by an exact subset-sum (zeta) transform over all
    2^n atoms.  Float solutions are used for DISCOVERY only and are repaired
    before verification (lowering lam_empty by the worst violation is always
    legal because every atom contains the empty set and m_empty = 1).
(e) The telescoping slope identity s = S1 * prod + beta is asserted equal to
    the term-by-term chain slope for chains of depth 1..3 at machines 19..37.
(f) S1 comparisons at large y use rigorous integer brackets
    (floor(2*10^40/q) summed), never floats.

Honest limits.  The sharp (multivariate) ceiling test costs 2^n, so it is
computed only to y = 37 (n = 10).  The aggregated test costs n columns and
l+1 rows and runs to y = 12000, but it is a RELAXATION: aggregated-BITES
implies sharp-bites (asserted at every machine where both are computed), but
aggregated-VACUOUS does not prove sharp-vacuous.  So the aggregated ceiling
machines are lower bounds on the sharp ones, and the degree law l ~ 2 S1 is a
MEASURED law on those lower bounds, not a theorem.  There is no proof here
that every fixed degree l eventually goes sharp-vacuous; that is established
degree by degree (done for l = 1 and l = 2).

## 4. IMPLICATIONS

Inside the project.
- It closes, with a reason, the hope that LP duality over the covering IP is
  a general vehicle for (D).  Any certificate of fixed arity has a computable
  last machine.  This is the LP-side answer to the round-22 spine question
  "does the truncation arity stabilise?": IT DOES NOT - and the growth is
  ~4 log log y, i.e. the arity Constructor measured going 3 (m19/23) -> 4
  (m29) is the beginning of a doubly-logarithmic climb, not a plateau.
- It therefore supports the arity-free-generator thesis from an independent
  direction: a mechanism whose required arity grows cannot be a fixed rule.
- It gives a cheap pre-flight check for any future certificate scheme of this
  type: compute S1(y), read off the degree it must carry.

Outside.  For any covering-type extremal question over residues (Jacobsthal,
covering systems, admissible tuples), the same two-line test says in advance
whether a Bonferroni/LP certificate of degree l can possibly bite at that
size, without building the certificate.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

- (D) / the tolerance route: quantifies the minimum correlation depth any
  covering-dual proof of (D) must carry at each step.
- The Jacobsthal upper-bound ladder: the same ceiling bounds what a
  Bonferroni-style counting bound (Costello-Watts family) can achieve at
  fixed truncation order.
- Open: is the sharp ceiling machine finite for EVERY degree l?  (Verified
  for l = 1, 2; the aggregated relaxation goes vacuous for every l tested,
  which is necessary but not sufficient.)
- Open: the exact asymptotic constant in l(y) ~ c * S1(y); measured c ~ 2.

## 6. PRIOR-ART CHECK

Searches run 2026-08-24 (all via web search; terms verbatim):
1. "Jacobsthal function upper bound linear programming duality certificate"
2. "fractional covering LP relaxation covering congruences integrality gap"
3. "Kounias Hunter bound Bonferroni inequalities sieve method prime gaps
    covering"
4. "admissible tuples fractional relaxation linear programming bound prime
    k-tuples covering residue classes"
5. "Boole problem Boros Prekopa sharp Bonferroni bounds linear programming
    dual binomial moments"
6. "covering system of congruences integer programming linear programming
    relaxation dual bound"
7. "Jacobsthal function computation certificate residues covering intervals
    Hagedorn Ziller exhaustive search upper bound proof"
8. "Brun pure sieve truncation level must grow log log x Bonferroni terms
    vacuous number of primes"
9. "Erdos covering systems linear programming relaxation fractional covering
    density bound Filaseta Hough"
10. full text of Costello-Watts, "A computational upper bound on Jacobsthal's
    function" (arXiv:1208.5342), fetched and read.

Nearest published results.
- Prekopa; Boros-Prekopa, "Closed form two-sided bounds..."; "Boole-Bonferroni
  Inequalities and Linear Programming" (Oper. Res. 36, 1988).  THE MACHINERY
  OF SECTION C IS THEIRS: sharp bounds on P(union) from binomial moments as an
  LP, with dual feasible bases.  The aggregated ceiling test is exactly their
  binomial-moment problem.  Delta: they bound a probability; here the same LP
  is used as a VACUITY TEST for a certificate scheme in a number-theoretic
  covering problem, and the ceiling machine per degree is the output.
- Kounias (1968), Hunter (1976), Worsley: the degree-2 cut family.  Standard.
  (Noted in passing: for weights 4/(q_i q_j) the maximum spanning tree is the
  star at the smallest gear, so Hunter-Worsley collapses to Kounias with
  k = 5 here - the seed's choice was already family-optimal at degree 2.)
- Brun's pure sieve: the truncation level must grow with the dimension for
  exactly the reason above.  This is the honest classical statement of the
  phenomenon.
- Costello-Watts (arXiv:1208.5342): upper bounds on h(k) by a recursive
  counting bound with a pairwise correction term
  (sum sum phi_min(floor(r/(p_i p_j)), i-1)) plus an E-term for residue
  co-occurrence.  This is the same species as the closed-form corollary in
  `covering-lp-certificates.md` and is STRONGER (it recurses).  It contains
  no LP, no dual, and no ceiling analysis.
- Hough; Balister-Bollobas-Morris-Sahasrabudhe-Tiba "distortion method" for
  covering systems: density-of-uncovered-set arguments over a product
  measure.  In the language here, those are degree-1 arguments; the ceiling
  test says why degree 1 stops at machine 13 in this problem.

VERDICT: PARTIAL OVERLAP.  The machinery (Boole-Bonferroni LP, Kounias /
Hunter cuts, Brun truncation growth) is all classical and must be cited; the
delta claimed is (i) the per-degree exact ceiling machine for the
Jacobsthal-type covering IP, (ii) the separation between the sharp ceiling and
the chain family's ceiling with the telescoping closed form, and (iii) the
measured degree law l(y) ~ 2 S1(y) ~ 4 log log y for this problem.  No source
found states any of the three.  NOT independently confirmed.
