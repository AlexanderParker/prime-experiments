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
- ANSWERED IN ROUND 25 (was the entry's own open question): STAR-3 reduces
  Delta at machine 41 width 129 by 42%, far more than the 1.7% the frontier
  needed; the necessary condition then holds at every budget width through
  machine 53.  What was NOT answered is whether the STAR-3 composition, run as
  an LP, actually certifies anything new - the uniform margin is a necessary
  condition only.

- ANSWERED IN ROUND 26, BOTH WAYS (see `restricted-covering-certificates.md`).
  The right object is not the STAR-3 LP but the CASE SPLIT: holding gear 5's
  phase at w makes case w the SAME composed vehicle on the position set
  [0,W) minus what gear 5 blocks, and a certificate in every case is a
  certificate of the rung.  It is strictly stronger than the STAR-3 LP (a
  family of case points always mixes into a STAR-3 point; a STAR-3 point does
  not condition into a family of case points).
  * IT DOES CERTIFY SOMETHING NEW, and immediately: at 19 -> 23, budget width
    48 - the cell round 25 REFUTED for the level-2 vehicle with an exhibited
    exact witness - ALL FIVE CASES CERTIFY at iteration zero, with no cut
    generation at all, 38,677 exact certificate operations in total.  So round
    25's refutations bound the level-2 MEMBER, not the family, and this
    entry's "the vehicle's reach is everything through 31 -> 37" understates
    the family by exactly the rungs it had given up.
  * ROUND 25'S REFUTATION OF 37 -> 41 DOES NOT TRANSFER.  That refutation
    rests on the uniform product measure's degree-2 moments being COMPLETABLE
    at machine 41 (n = 11).  Conditioned on gear 5 they are NOT: at n = 10
    (drop 5) and n = 9 (drop 5 and 7) the conditional product moments carry an
    exactly-verified violated degree-2 cut.  Holding one gear revives BOTH
    ingredients, so 37 -> 41 is an open cell again for the stronger species.
  * BUT THE CASE SPLIT STILL DOES NOT REACH MACHINE 41 AT BUDGET WIDTH.
    Measured (not proved): the LP maximum of the recursion row runs
    87.0713, 87.0632, 87.0230, 86.9653, 86.9331, 86.8935, 86.8664, 86.8351,
    86.7949, 86.7556 over ten cut passes against the 78 it must fall below -
    a residual gap of 8.8 falling about 0.03 per pass.  NECESSITY ONLY at m41.
  * The pre-test itself became case-by-case rather than on-average: NOT ONE of
    the 5 / 35 / 385 cases at machines 41..53 (k = 1, 2, 3) has E_u[f_w] <= 0,
    and the case MEANS reproduce this table's STAR-k column exactly (gated).

- ROUND 27: THERE IS A SECOND FRONTIER, AND IT IS NOT THIS ONE.  The width
  frontier of this entry is a statement about the uniform product measure - a
  NECESSARY condition for the vehicle.  Round 27 found a case where necessity
  holds comfortably and the vehicle still does not deliver: machine 43 at width
  117 (the increment width F_2(41) + s_min(43), against the budget width 134).
  The pre-test is passed at every sampled case - E_u[f_w] = +5.62, +10.80,
  +14.01 (min over samples at k = 1, 2, 3) - so nothing here refutes the
  species; but the cut loop's LP maximum falls only 44.2578 -> 43.4856 over
  FIFTEEN passes (654 cut rows, 377 s) against the 43 it must beat - about 0.05
  per pass and decelerating - and one case did not decide in 35 minutes against
  10-40 s per case at width 134.
  So the vehicle's cost is NOT a smooth function of the width: it explodes as W
  approaches the value being proved, while the product-measure margin stays
  healthy.  The width frontier bounds where the vehicle CAN work; a second,
  independent quantity - call it the convergence frontier - bounds where it
  does so affordably, and only the first of the two has a closed form.
  New open question: is the convergence rate of the cut loop predictable from
  E_u[f_w] and W - F(machine) at all, or is it a genuinely separate object?

- ROUND 28 ANSWERS THE ROUND-27 QUESTION, AND THE ANSWER IS "NEITHER - IT IS
  AN INSTRUMENT PROBLEM".  See section 7.  The convergence frontier is not a
  separate species of obstruction at all: the cut loop's limit is the optimum
  of ONE LP, computable directly, and the deceleration is the loop bending
  towards that limit.  At machine 43 width 117 the limit polytope is EMPTY -
  round 27's decelerating loop was converging to a certificate, not to an
  asymptote, and the cell now certifies at ITERATION ZERO.

## 7. ROUND 28 - THE CUT LOOP'S LIMIT IS ONE LP, AND THE SECOND FRONTIER IS A
## WIDTH TOO (AT EACH k)

Status: THEOREM (7.1, two lines) + SCRIPT-VERIFIED exact objects.  Gate:
`research/gate_r28.py GATE`.  Files: `research/cutlimit_r28.py`,
`frontier_r28.py`, `wc_r28.py`, `decel_r28.py`.

### 7.1 The theorem: the loop's limit is the lifted optimum

Fix a case cell: machine y, width W, held gears at phases ws, free gears
q_0 < ... < q_{n-1}, position set pos, and the level-2 relaxation `RelaxStar`.
The cut loop's rows are drawn from the family of EXACTLY VALID degree-2 cuts

    lam_0 + sum_{S subset x, S nonempty} lam_S  >=  1    for every nonempty x,

and a point z satisfies EVERY member of that family at position i exactly when
its degree-<=2 moment vector at i extends to a probability distribution on the
NONEMPTY subsets of the free gears.  Define the LIFTED PROGRAM

    V*  =  max  sum_j frow_j z_j
    over z >= 0 and p_i >= 0 on the 2^n - 1 nonempty subsets, subject to
      (B) every block of z sums to 1;      (L) every consistency link;
      (N) sum_x p_{i,x} = 1 for every i in pos;
      (M) sum_{j : i in O_j, mask(S_j) = m} z_j = sum_{x superset m} p_{i,x}
          for every atom mask m of a subset of size 1 or 2.

THEOREM.  The cut loop's LP maximum is >= V* at every pass, and equals V* at
termination.  PROOF.  (>=) the loop's rows are a subset of the valid cuts, so
its feasible region contains the lifted projection.  (<=) at termination the
EXACT separation oracle has found no violated cut at any position, so every
position's moment vector is completable, so the loop's optimal z lifts to a
feasible (z, p) and its value is at most V*.  QED.

CONSEQUENCE - the exact dichotomy, replacing "does the loop converge":

    V* < |pos|  (or the lifted polytope EMPTY)  the cell IS certifiable;
    V* >= |pos|                                 the cell is NOT, ever.

And the round-27 decomposition of the observed deceleration is forced:

    lp_max_t - |pos|  =  (lp_max_t - V*)  +  (V* - |pos|).
                          the convergence      A CONSTANT OFFSET

A loop whose "gap to the target" decelerates is a loop converging normally to
a limit that is somewhere else.  The two things round 27 could not separate
are separated by computing V* once.

### 7.2 The instrument, validated against the stalling loop

Machine 37, width 88, k = 2, case (0,0), |pos| = 38.  The ordinary cut loop
runs 24 passes in 259 s and stalls at LP maximum 40.4834.  The lifted LP
returns V* = 40.48344218 in 35 s.  The loop was measuring V*, slowly.

Excess e_t = lp_max_t - V* over those 24 passes:
0.372 0.363 0.320 0.275 0.196 0.132 0.098 0.083 0.068 0.054 0.033 0.021
0.014 0.009 0.005 0.004 0.003 0.003 0.001 0.000 ... - GEOMETRIC, ratio about
0.75 per pass.  So the loop's own convergence was never slow; what did not
move was the offset V* - |pos| = +2.483.

### 7.3 A PROOF OF ASYMPTOTE (the first one this family has had)

For a cell with V* >= |pos| an exact witness turns "the loop stalled" into
"the loop cannot succeed".  Rationalising the lifted optimum fails - the
optimum sits ON the completability boundary at 19-24 of the 38 positions, and
no denominator up to 10^10 repairs it.  The fix is to ask for an INTERIOR
point instead: maximise t subject to the recursion row already clearing |pos|
and p_{i,x} >= t at every atom of size <= 2 and at the full atom (the columns
of the incidence matrix that span the degree-<=2 moment space).  At the cell
above t = 6.3139e-4 > 0, and the rationalised primal verifies EXACTLY:

    every block sums to 1, every consistency link holds, ALL 38 positions
    exactly completable, recursion row 38.5021 >= 38 = |pos|.

So machine 37 at width 88 with TWO held gears can never be certified by this
species, however long the cut loop runs.  (It is certified at THREE held gears
in every one of the 385 cases - see `restricted-covering-certificates.md`.)

### 7.4 The round-27 cell: SLOW CONVERGENCE, and my round-27 reading was wrong

Machine 43, width 117 (the increment width F_2(41) + s_min(43)), k = 3, case
(0,0,0), |pos| = 43 - the cell whose LP maximum fell 44.2578 -> 43.4856 over
fifteen passes in 377 s and which round 27 could not decide.

    THE LIFTED POLYTOPE IS EMPTY.

V* = -infinity: level-2 consistency alone, with no recursion row at all,
already excludes a fully blocked window of width 117 in that case.  Round 27's
"about 0.05 per pass and decelerating - at that rate the crossing is ~10 more
passes away" was reading a converging loop as a possible asymptote.  THE
CONVERGENCE FRONTIER, AS ROUND 27 POSED IT, DOES NOT EXIST AT THAT CELL.

That reading is a float LP's infeasibility, hence a measurement; the EXACT form
is a split.  Case (0,0,0) at k = 3 decomposes into its 13 sub-cases at k = 4
(the phases of gear 13, exhaustive by construction), and every one carries its
own exact rational dual certificate: 13/13 CERTIFIED, all at iteration zero,
571,466 exact certificate operations, each re-verified from disk.  So the cell
is excluded by exact certificates, and the round-27 loop was converging to one.

### 7.5 The cost finding: with the lifted duals, there is no loop

When the lifted polytope is nonempty the duals of (M) and (N) give, at each
position, mu_i / nu_i - a valid cut with lam_0 = 0 (dual feasibility at the
p-columns is literally the validity condition), repaired to exact validity by
raising lam_0 to the exact deficit, which only weakens the row.  When the
polytope is EMPTY there are no duals, and the companion program supplies them:
relax (N) to sum_x p_{i,x} = s_i in [0,1], impose the recursion row as a hard
constraint, and maximise sum_i s_i; it is always feasible, its optimum is
|pos| exactly when the lifted polytope is nonempty, and its duals carry the
same cuts.

Seeded with those rows, EVERY certifiable cell measured this round certifies
at ITERATION ZERO.  Two cells that the ordinary loop left STUCK at a 300 s
budget certify in one pass once seeded (m37 W=88 k=3 case (0,5,8): mass optimum
28.98697 < 29 = |pos|, 29 seeded rows, 29,586 exact ops, iteration 0).
The cut loop was never the vehicle; it was a way of discovering the vehicle's
rows one separation at a time.

### 7.6 The frontier is a width at each k

G(y, k, W) = V*(y, k, W, case 0) - |pos| falls with W and crosses once.
Measured (exact rational |pos|, float V*; the crossing itself is then
re-decided exactly by the certificate or the witness):

    y  k    W    |pos|      V*        G        G/W
   23  1   30      18   20.5000   +2.5000   +0.0833
   23  1   32      19   21.0000   +2.0000   +0.0625
   23  1   34      21   22.5455   +1.5455   +0.0455
   23  1   38      23   23.4428   +0.4428   +0.0117
   23  1   40      24   24.2548   +0.2548   +0.0064
   23  1   41      25    EMPTY       -          -
   29  1   36      22   25.3333   +3.3333   +0.0926
   29  1   44      27   30.2967   +3.2967   +0.0749
   29  1   52      31   32.7106   +1.7106   +0.0329
   29  1   60      36   37.0888   +1.0888   +0.0182
   29  1   64      39   39.1508   +0.1508   +0.0024
   31  1   44      27   33.2376   +6.2376   +0.1418
   31  1   48      29   34.6667   +5.6667   +0.1181
   31  1   52      31   36.2273   +5.2273   +0.1005
   37  1   80      48   57.0461   +9.0461   +0.1131
   37  2   88      38   40.4834   +2.4834   +0.0282
   43  3  117      43    EMPTY       -          -
   43  4  117      35    EMPTY       -          -

W_c(y, k) = min{W : G < 0} is located by bisection on the lifted value and the
sign pattern is then ASSERTED width by width over a nine-wide band, not
assumed monotone.  W_c(23, 1) = 41 exactly (F(23) = 34, budget 48), single
crossing confirmed.  The frontier moves DOWN with k, which is the knob the
case split has and the level-2 vehicle did not.

### 7.7 What this does NOT settle

- The lifted program has 2^n columns per position, so it is affordable only
  while the FREE-gear count is small - n <= 8 costs seconds, n = 9 costs
  8-14 minutes at machine 43.  It decides cells; it does not scale to the
  ladder's largest cells any more than the loop does.
- W_c has no closed form here.  What replaced round 27's open question is a
  DECISION PROCEDURE with a two-line correctness proof, plus the observation
  that the frontier is a width at each k - the same shape as Result 3, one
  level up.
- The offset V* - |pos| is an integrality gap of the level-2 case-split
  relaxation.  Bounding it in closed form is the same unsolved problem as
  bounding Delta, one relaxation stronger.

## 7.8 ROUND 29 - THE k = 3 FRONTIER LADDER, AND WHAT THE OFFSET ACTUALLY
## TRACKS

TWO MEASUREMENTS, both by the section-7.1 lifted LP, both with their sign
pattern asserted width by width rather than assumed.

(A) THE FRONTIER LADDER AT THREE HELD GEARS.  W_c(y, 3) = min{W : G < 0} at
the all-zero case, bisected and then asserted single-crossing over the band
around the crossing (`research/lp_cells_r29.py WCALL`, logs
`research/data/r29/wc_m*_k3.json`):

    y            23     29     31     37     41
    W_c(y, 3)    13     31     46     66     81
    F(y)         34     43     58     88     91
    W_c / F(y)  0.382  0.721  0.793  0.750  0.890

  * W_c(y, 3) IS STRICTLY MONOTONE IN y over the five machines the lifted LP
    reaches at k = 3.  Round 28's pre-registered E9 guessed it would NOT be,
    and that half of E9 is REFUTED by this table.
  * THE RATIO IS RISING TOWARDS 1 (0.38 -> 0.89), so the per-case reach is
    closing on the machine's own record gap from BELOW.  Round 28 recorded
    "at machine 41 with k = 3 the case-0 polytope is EMPTY at every width down
    to 92 = F(41) + 1"; the bisection extends that by eleven units - the
    case-0 cell is certifiable down to 81, i.e. TEN BELOW F(41) = 91.  This is
    a per-case statement and stays one: the FULL split must fail below F(y)
    somewhere, because a fully blocked window of width F(y) - 1 exists and its
    held phases put it in SOME case.
  * The ratio's own shape is not monotone (0.382, 0.721, 0.793, 0.750, 0.890),
    so no law is claimed for it - only the direction over five machines.

(B) THE OFFSET AT THE INCREMENT WIDTH IS NOT A FUNCTION OF THE MACHINE.
Round 28 measured V* - |pos| = +9.05 at machine 37, one held gear, at the
31 -> 37 increment width 80, and pre-registered (E12) that this offset "grows
with the machine".  IT DOES NOT, and the reason is arithmetic rather than
asymptotic.  Write the increment width against the new machine's own record:

    step      W_inc = F_2(M) + s_min(q')   F(q')   W_inc - F(q')
    11 -> 13     15 = 11 + 4                 11        + 4
    13 -> 17     22 = 16 + 6                 18        + 4
    17 -> 19     31 = 25 + 6                 25        + 6
    19 -> 23     39 = 31 + 8                 34        + 5
    23 -> 29     49 = 39 + 10                43        + 6
    29 -> 31     65 = 55 + 10                58        + 7
    31 -> 37     80 = 68 + 12                88        - 8     <- the padded step
    37 -> 41    104 = 90 + 14                91        +13
    41 -> 43    117 = 103 + 14              103        +14
    43 -> 47    132 = 116 + 16              118        +14

At 31 -> 37 the increment width is EIGHT BELOW the truth, so a fully blocked
window of that width exists and NO SOUND METHOD can certify it at any k - the
offset there is the machine's own padding excess.  Everywhere else W_inc
exceeds F(q') and the obligation is TRUE, so a positive offset there is an
integrality gap of the relaxation and nothing else.  Measured, all at the
all-zero case:

    step        W_inc   W_inc - F(q')   k=1        k=2        k=3
    31 -> 37      80        - 8         +9.0461    +3.7901    EMPTY (-inf)
    37 -> 41     104        +13         n = 10,    +5.1667    EMPTY (-inf)
                                        out of reach

TWO READINGS, and they point opposite ways, so both are stated:
  * AT FIXED k THE OFFSET DOES GROW: +3.7901 -> +5.1667 at k = 2, the only
    matched pair the lifted LP can reach (k = 1 at machine 41 is n = 10 free
    gears, past the program's scaling wall).  Two points, one k.
  * BUT THE OFFSET IS NOT A PROPERTY OF THE STEP - IT IS A PROPERTY OF
    (step, k), AND THE LADDER PARAMETER ABSORBS IT.  At 37 -> 41 the full
    k = 3 split leaves only 9 of 385 cases with a positive offset (+0.54 to
    +1.83), and all 9 close when each is split once more on gear 13: 376 + 117
    = 493 exact certificates over a partition, so the increment width at
    37 -> 41 IS certified.  At 31 -> 37 no such k exists, because the
    obligation is false.
So "the offset grows with the machine" is true at fixed small k and is the
wrong quantity to watch: what decides certifiability is W_inc - F(q'), which
is negative at exactly one step of the corpus.

## 8. PRIOR-ART CHECK

NOT YET CHECKED (2026-08-29; this agent has no web access this round).
ROUND-28 ADDITION, and it needs the check most: THE LIFTED PROGRAM OF SECTION
7.1 IS NOT CLAIMED AS NEW MATHEMATICS.  Writing a cutting-plane loop over a
moment cone as one extended formulation with a distribution variable per
constraint is the standard lift-and-project / Sherali-Adams move (Lovasz-
Schrijver, Sherali-Adams, Balas), and the "loop's limit = lifted optimum"
theorem is the textbook separation-equals-optimisation observation for that
pair.  What is offered as possibly new is (i) the READING - that a
cutting-plane loop's observed deceleration decomposes into a normal geometric
convergence plus a constant integrality-gap offset, so "the loop is slow" and
"the loop cannot succeed" are separated by one LP solve; (ii) the INTERIOR
witness construction of 7.3 (floor the atom distribution on the low-order
atoms only - flooring all 2^n - 1 atoms is infeasible because the pair moments
are O(1/q_a q_b)); and (iii) the frontier reading W_c(y, k) for this family.
Terms to search: separation-vs-optimisation for the Sherali-Adams level-2
covering relaxation; cutting-plane convergence rate versus integrality gap;
interior points of moment cones by low-order-atom flooring.
Terms to search: "lowest blocking prime" second-order expansion exactness;
Costello-Watts arXiv:1208.5342 pair term density; Bonferroni truncation
exactness with lowest-index conditioning; extreme-value excess in covering LP
relaxations.  The two ingredients separately are classical (Bonferroni
truncation; the singular series prod(1-2/p)); the claim needing a check is the
identity A(y) = Pi(y) in this form and the width-frontier reading.
