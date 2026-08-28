# product-measure-frontier - the recursive covering row's margin against the uniform product measure has the closed form W*Pi(y) - Delta(y,W), with asymptotic slope EXACTLY the machine's own survival density; so the row's frontier is a WIDTH, not a machine

Status: SCRIPT-VERIFIED with EXACT rational arithmetic on every claim
(`research/row_decay.py`; no float appears anywhere in that file), plus a
PROVED identity (section 3.1) that the script re-asserts at all 60 machines up
to 300.  Established round 25 (LP-duality dedicated explorer).
Prior-art check: NOT YET CHECKED (agent has no web access this round).

Companion entries: `recursion-consistency-composition.md` (the row and the
vehicle it lives in), `moment-degree-ceiling.md` (the degree-2 vacuity ceiling
this row was built to escape), `covering-lp-certificates.md`,
`consistency-over-degree.md`.

## 1. WHAT IT IS

Plain language.  Round 24 built one extra LP row out of Costello-Watts'
recursion and found something the project had never had: a certificate-species
object that still *sees* the uniform product measure past machine 29, where
every fixed-degree moment certificate has gone blind.  It reported six measured
numbers - the row's margin against that measure at machines 23, 29, 31, 37, 41,
43 - and read them as "the row's own vacuity frontier is machine 41".

That reading is wrong, and this entry says exactly why.  The margin is not a
sequence of six numbers; it is a closed form in two pieces.  One piece is
exactly linear in the window width W with slope the machine's OWN SURVIVAL
DENSITY prod(1 - 2/q).  The other piece is a pure extreme-value quantity - the
total amount by which a maximum exceeds its mean inside the recursion - and it
grows STRICTLY SLOWER than linearly.  So for every machine there is a finite
width past which the row cuts the uniform measure, and the frontier the project
actually cares about is whether the (D) ladder's BUDGET width clears it.  At
machine 41 the budget is 129 and the threshold is 135: the row misses by six.

The precise statements.  Fix a machine M = M(y) with gears q_0 = 5 < q_1 < ...
< q_{n-1} = y (each gear has exactly two teeth), a window width W, and write

  * S_q(r)      = #{positions of [0,W) that gear q blocks at phase r};
  * P_ij(u,v)   = the positions blocked by BOTH q_i (phase u) and q_j (phase v);
  * n_ij(u,v)   = |P_ij(u,v)| - max over the phases of the gears BELOW q_i of
                  the number of positions of P_ij those gears block
                  (the Costello-Watts lowest-blocking-prime pair minimum);
  * f(r)        = W - sum_q S_q(r_q) + sum_{i<j} n_ij(r_i, r_j)   <=   open(r);
  * THE ROW     = "sum_q E[S_q] - sum_{i<j} E[n_ij] >= W", valid at every fully
                  blocked window, i.e. "E[f] <= 0";
  * E_u         = expectation under the UNIFORM product measure on phases.

So the row cuts the uniform product measure exactly when E_u[f] > 0.  Write
s1 = sum_{5<=q<=y} 1/q, pi_i = prod_{k<i}(1 - 2/q_k), and
Pi(y) = prod_{5<=q<=y}(1 - 2/q).

  RESULT 1 (the exact decomposition).
      E_u[f](y, W)  =  (6W/5)(7/10 - s1)  +  N_+(y, W),
      N_+ = sum_{1 <= i < j} E_u[n_ij]  >=  0.
  The leading term is EXACT and in closed form: it is everything the i = 0 pair
  terms and the single-gear terms contribute, because
      sum_r S_q(r) = 2W  and  sum_{u,v} |P_ij(u,v)| = 4W   EXACTLY,
  and because gear 5 has no gear below it, so n_0j = |P_0j| identically.
  The constant 7/10 is the whole of the row's free power, and

      s1(23) = 0.665622680  <  7/10  <  0.700105439 = s1(29):

  the crossing sits between machines 23 and 29, and machine 29 clears it by
  1.06e-4.  Below the crossing the row cuts uniform for free at every width;
  from machine 29 on the leading term is negative and LINEAR IN W, and the row
  survives only on N_+.

  RESULT 2 (the closed form, and the identity).
      E_u[f](y, W)  =  W * Pi(y)  -  Delta(y, W),
      Delta(y, W) = sum_{i<j} (1/(q_i q_j)) sum_{u,v}
                        [ max_r |cov(P_ij(u,v); r)| - (1 - pi_i)|P_ij(u,v)| ]
                  >= 0.
  Delta is the SUMMED EXCESS OF THE PHASE MAXIMUM OVER THE PHASE MEAN in the
  recursion's pair minima - exactly what the vehicle pays for letting each pair
  term privately optimise the lower gears' phases.  Equivalently
      A(y) := 1 - 2 s1 + 4 sum_{i<j} pi_i/(q_i q_j)  =  Pi(y)     (IDENTITY),
  and A(y) is BOTH an exact upper bound on E_u[f]/W at every width AND the
  exact limit of E_u[f]/W as W -> infinity.

  RESULT 3 (the frontier is a width).  Since Pi(y) > 0 at every finite machine,
  and Delta grows sublinearly (measured doubling factor 1.45-1.68 against a
  gain that doubles exactly), EVERY machine has a finite threshold
      W_u(y) = min{ W : E_u[f](y, W) > 0 },
  exact values (this round):

      y        29     31     37     41     43     47     53
      W_u      10     48     83    135    211    362    558
      budget   63     74     95    129    134    150    156
      ratio  6.300  1.542  1.145  0.956  0.635  0.414  0.280

  The ratio falls monotonically through 1 between machines 37 and 41.  Round
  24's "the row loses the product measure at machine 41" is therefore not a
  property of machine 41: it is the statement budget(41) = 129 < 135 = W_u(41).

  RESULT 4 (the exact range of the vehicle).  Combining the row's sign with the
  degree-2 side gives a per-rung verdict with no LP run at all.  Under uniform
  phases EVERY position of the window carries the SAME degree-<=2 moment vector
  (namely the product moments of p_q = 2/q), so one exact rational completion
  decides all W positions at once.  Where the uniform point satisfies both the
  degree-2 cuts and the row, it is an exhibited exact feasible point of the
  FULL composition and the rung is REFUTED for that vehicle - not undecided.

## 2. WHY IT MIGHT BE NOVEL

- The identity A(y) = Pi(y) says the composed row's asymptotic power against
  the product measure is EXACTLY the machine's own survival density - the
  singular-series factor of the twin-prime constant, truncated at y.  The row
  was built from a recursion (Costello-Watts' lowest-blocking-prime partition)
  with no density in sight; that its product-measure slope collapses to
  prod(1 - 2/q) is not visible from the construction.  The proof (section 3.1)
  is a two-line counting identity that appears to be new in this direction:
  the SECOND-ORDER Bonferroni-with-lowest-blocker expansion of the blocked
  density is EXACT, not merely an approximation, because the correction term is
  precisely the count of blockers above the lowest one.
- The frontier is a NECESSARY condition and this round measured how far it is
  from sufficient, which is the honest calibration of what "the first object
  past the vacuity ceiling" is worth.  At 23 -> 29 the row's uniform margin is
  +3.27 - comfortably cutting - and yet the full composition at that width is
  REFUTED by an exhibited exact feasible point, with the LP optimum sitting at
  t = +0.1363, nowhere near zero.  Cutting the product measure and certifying a
  rung are separated by a wide gap, and this entry makes the first quantity
  computable in closed form while the second still needs the LP.
- The reframing "the frontier is a width, not a machine" changes what the
  vacuity ceiling means for this family.  `moment-degree-ceiling.md` establishes
  that fixed-degree certificates die at a computable MACHINE.  The composed row
  does not die at a machine at all - only at a width-to-machine RATIO - which is
  a qualitatively different kind of obstruction and, as far as this project has
  looked, a new one.
- Delta as an object.  The vehicle's entire loss against the product measure is
  a sum of "max minus mean" terms.  That is an extreme-value statement about
  phase optimisation, and it is measurable exactly.  It converts "the vehicle
  is too weak here" into "the extreme-value excess of the pair minima exceeds
  W Pi(y) here", which is a quantity one can attack.

## 3. PROOF

Status: SCRIPT-VERIFIED (finite, exact rational) for every numeric claim, with
the identity of Result 2 PROVED below and re-asserted by the script.
Gate: `research/row_decay.py` sections L, W, X, A, T, D, V - all assertion-
gated, aborting on any disagreement.  Nothing here uses a float.

### 3.0 The two counting lemmas (section L, asserted)

L1.  sum_{r in Z_q} S_q(r) = 2W for every gear q >= 5 and every W.  Count pairs
(r, i): position i is blocked by q at phase r iff r = t - i mod q for one of
the two teeth t, and the two teeth u', -u' are distinct mod q for q > 2.
Hence E_u[S_q] = 2W/q exactly.

L2.  sum_{u,v} |P_ij(u,v)| = 4W (the same count with two teeth on each of two
gears).  Gear 5 is the lowest gear, so for i = 0 the "no lower gear blocks it"
condition is vacuous and n_0j = |P_0j| identically, giving
E_u[n_0j] = 4W/(5 q_j) exactly.

Result 1 follows by substitution: E_u[f] = W - 2W s1 + (4W/5)(s1 - 1/5) + N_+
= (6W/5)(7/10 - s1) + N_+.  Asserted equal to the direct cell-by-cell
computation at machines 11, 13, 17, 19, 23, 29, 31, 37, 41, 43 (exact rational
equality at every one), reproducing round 24's six measured numbers.

### 3.1 The identity A(y) = Pi(y) (PROVED)

Under the uniform product measure, position i is blocked by gear q with
probability exactly 2/q (two teeth out of q phases), independently across
gears.  Let B = the number of gears blocking i.  Every blocker other than the
LOWEST one is a blocker strictly above the lowest, so pointwise

    B  =  1{B >= 1}  +  #{blockers strictly above the lowest blocker}.

Taking expectations: E[B] = 2 s1; P(B >= 1) = 1 - Pi(y); and, partitioning on
the identity of the lowest blocker, the expected number of blockers above it is
sum_i (2/q_i) pi_i * sum_{j>i} (2/q_j) = 4 sum_{i<j} pi_i/(q_i q_j).  Hence

    2 s1  =  (1 - Pi(y))  +  4 sum_{i<j} pi_i/(q_i q_j),

which is exactly A(y) := 1 - 2 s1 + 4 sum_{i<j} pi_i/(q_i q_j) = Pi(y).  QED.
(The script asserts A(y) == Pi(y) as exact rationals at all 60 machines up to
300.)

There is a second, shorter route that also explains WHY the slope is the
survival density: f <= open pointwise, and E_u[open] = W Pi(y) EXACTLY (each
position survives all gears with probability prod(1 - 2/q), independently by
CRT).  So E_u[f] <= W Pi(y) at every width - the bound can never exceed the
expected number of open slots - and Result 2's Delta is precisely the gap.

That A(y) is also the LIMIT needs one more observation: for any FIXED phase
choice r of the gears below q_i, the fraction of P_ij(u,v) that those gears
block tends to 1 - pi_i as W -> infinity, because P_ij is a union of at most
four arithmetic progressions modulo q_i q_j and the lower gears' moduli are
coprime to it, so CRT equidistributes.  The maximum over the FIXED FINITE set
of phase tuples therefore has the same limit, giving
E_u[n_ij]/W -> 4 pi_i/(q_i q_j) and E_u[f]/W -> A(y) = Pi(y).
Measured convergence (machine 31, A = 0.186275):
E_u[f]/W = 0.027196, 0.055096, 0.078045, 0.107508, 0.132572 at
W = 74, 148, 296, 592, 1184.

### 3.2 The deficit and the thresholds (sections D, T)

Delta(y, W) = W Pi(y) - E_u[f], asserted equal to the direct computation at
machines 23, 29, 31, 37, 41 and three widths each.  Exact values at the budget
width and its two doublings:

    y   W        W Pi(y)     Delta     E_u[f]   Delta/W   Delta(2W)/Delta(W)
   23   48       10.2658    6.8035    +3.4623   0.14174        -
   23   96       20.5316   10.7203    +9.8113   0.11167     1.5757
   23  192       41.0632   15.6000   +25.4632   0.08125     1.4552
   29   63       12.5446    9.2790    +3.2656   0.14729        -
   29  126       25.0893   15.3790    +9.7102   0.12206     1.6574
   29  252       50.1785   24.3637   +25.8148   0.09668     1.5842
   31   74       13.7843   11.7718    +2.0125   0.15908        -
   31  148       27.5686   19.4145    +8.1542   0.13118     1.6492
   31  296       55.1373   32.0360   +23.1013   0.10823     1.6501
   37   95       16.7395   16.3337    +0.4059   0.17193        -
   37  190       33.4791   26.0456    +7.4335   0.13708     1.5946
   37  380       66.9582   40.8705   +26.0876   0.10755     1.5692
   41  129       21.6217   21.9864    -0.3646   0.17044        -
   41  258       43.2435   36.9709    +6.2726   0.14330     1.6815
   41  516       86.4869   58.4500   +28.0369   0.11328     1.5810

The gain column doubles exactly (it is W Pi(y)); Delta's doubling factor is
measured in [1.455, 1.682] at every one of the ten doublings taken, i.e.
strictly below 2.  (The implied exponent log2 of that factor is 0.54-0.75; that
is a DESCRIPTION of the measured factors, not a fitted law, and no asymptotic
form for Delta is claimed.)

W_u(y) is computed by bisection and then SHARPENED exactly: E_u[f] is asserted
<= 0 at W_u - 1 and > 0 at each of W_u, ..., W_u + 6 (monotonicity is not
assumed).

### 3.3 Exactness of n_ij at large width (a round-24 limitation removed)

n_ij is a valid lower bound on the true pair count only if the coverage term is
an UPPER bound on the true maximum coverage.  Round 24's routine swept a
2^|P| bytearray and gave up at |P| > 18, falling back to "claim full coverage"
(valid, but it throws information away and makes the measured E_u[f] a lower
bound).  This round adds two exact routes:

  * a FULL-COVER BACKTRACKER - branch on which lower gear covers the first
    uncovered position and at which of its two phases; each gear's phase is
    chosen once, so the depth is at most the number of lower gears and the
    search is complete.  Whenever it succeeds, maxcover = |P| EXACTLY, at any
    |P|;
  * a REACHABLE-MASK SWEEP - the set of achievable covered-masks has size at
    most prod_{k<i} q_k regardless of |P|, so enumerating it is exact and
    cheap where the 2^|P| sweep is hopeless.

The three routes are cross-checked against each other on 15,609 cells at
machines 19/23/29 and two widths each; all agree.  Every number reported above
is exact (the fallback fired nowhere in the tables above); any cell that ever
needs the fallback is counted and the affected value is labelled a lower bound.

## 4. IMPLICATIONS

Inside the project.

- Round 24's frontier statement is CORRECTED: the composed row is never
  uniformly vacuous at any machine (A(y) = Pi(y) > 0 always); it is only ever
  TOO NARROW at the width the (D) ladder needs.  The vehicle's reach for the
  ladder is exactly the steps whose budget clears W_u, i.e. everything through
  31 -> 37, and it fails from 37 -> 41 on - where the failure is now an exact
  REFUTATION (the uniform product measure is an exhibited feasible point of the
  full composition at width 129), not an undecided cell.
- It says what a stronger row would have to do, AND THE ANSWER WAS MEASURED.
  Delta is the excess of a maximum over a mean, so the way to shrink it is to
  stop the pair terms choosing the lower gears' phases privately - exactly the
  STAR-3 construct named in `recursion-consistency-composition.md` section 5.
  Holding the k smallest gears' phases explicit gives n^K >= n pointwise, so
  E_u[f] can only rise.  Measured exactly at the ladder's own budget widths
  (`row_decay.py` section S):

      y    W     W Pi(y)   level 2    STAR-3   STAR-{5,7}
     37   95     16.7395   +0.4059   +8.0012     +11.9812
     41  129     21.6217   -0.3646   +8.8853     +14.2963
     43  134     21.4151   -2.9469   +6.6797     +12.7830
     47  150     22.9521   -5.9284   +5.0991     +12.2560
     53  156     22.9694   -9.4094   +3.1065     +10.8054

  So the FAMILY is not out of range at machine 41 at all - only the level-2
  member is.  Holding ONE gear (the 5) turns -0.36 into +8.89 and keeps the row
  cutting uniform at every budget width through machine 53, and holding two
  gears roughly doubles that again.  Note STAR-k does not change the
  asymptotic slope - it is still bounded by W Pi(y), because f <= open is what
  bounds it - so STAR-k buys FRONTIER, not slope: it eats into Delta, and the
  measured bite is 42% of Delta at machine 41 (21.99 -> 12.74) and 71% for
  STAR-{5,7} (-> 7.33).  What it costs is LP size: the level-2 blocks become
  triples, so the columns grow by a factor of 5 (then 35).
- It gives the vehicle a cheap pre-test.  Deciding "can the composition
  possibly certify machine y at width W?" costs one closed-form evaluation plus
  one exact completion, instead of a multi-hour cut loop.

Outside it.  The identity of section 3.1 is a statement about truncated
Bonferroni expansions: the second-order lowest-blocker expansion of a union's
density is exact, with the correction being the expected number of events above
the first.  If that has an independent life, it is as a remark about when the
Costello-Watts partition makes a second-order expansion tight.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

- Conjecture (D) / the twin-prime route: this maps exactly which rungs the
  composed vehicle can reach and proves it cannot reach 37 -> 41.
- The arity/degree question (`moment-degree-ceiling.md`): the composed row is a
  certificate object whose obstruction is NOT a degree ceiling.  It shows the
  arity law does not govern every certificate family - only the moment ones.
- Open, and now sharply posed: is Delta(y, W) = o(W) with an effective rate?
  A proved sublinear bound on Delta would turn "W_u(y) is finite" from a
  measured fact into a theorem with an explicit W_u.
- ANSWERED THIS ROUND (was the entry's own open question): STAR-3 reduces Delta
  at machine 41 width 129 by 42%, far more than the 1.7% the frontier needed;
  the necessary condition then holds at every budget width through machine 53.
  What is NOT answered is whether the STAR-3 composition, run as an LP, actually
  certifies anything new - the uniform margin is a necessary condition only.

## 6. PRIOR-ART CHECK

NOT YET CHECKED (2026-08-29; this agent has no web access this round).
Terms to search: "lowest blocking prime" second-order expansion exactness;
Costello-Watts arXiv:1208.5342 pair term density; Bonferroni truncation
exactness with lowest-index conditioning; extreme-value excess in covering LP
relaxations.  The two ingredients separately are classical (Bonferroni
truncation; the singular series prod(1-2/p)); the claim needing a check is the
identity A(y) = Pi(y) in this form and the width-frontier reading.
