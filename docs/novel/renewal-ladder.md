# renewal-ladder - a closed-form CRT ladder of upper bounds on joint qualifying-gap counts mod a primorial, converging to exact and clearing the route's anti-correlation requirement

Status: PROVED (validity of every bound in the ladder - elementary inclusion-exclusion
plus CRT) + SCRIPT-VERIFIED (all values; every bound asserted >= the exact full-period
census where a census exists). The REQUIREMENT it is checked against ((D) via the
suppression law) still contains the fitted constant lambda - that caveat is inherited,
not created here. Established round 20 (constructor). Prior-art check: not yet checked.

## 1. WHAT IT IS

Plain language. Round 19 measured that several consecutive large ("qualifying") gaps in
the primorial pattern are drastically rarer than independence predicts (x26-x1400,
docs/novel/suppression-law.md), and R32 showed a rigorous bound on that joint rate is
the only anti-correlation input the route's open part (D) needs. The obvious rigorous
bound - keep only the requirement that the run's endpoints be exposed, drop the
requirement that nothing between them is exposed - factorises perfectly over the gears
by CRT, but fell short of the requirement by x2-x29 (X20: the missing factor is exactly
the dropped "renewal" condition). This tool restores the dropped condition ONE INTERIOR
POINT AT A TIME: for any finite set of interior points you choose to re-block, the count
of pattern positions is STILL exact closed-form CRT arithmetic, and every choice gives a
valid upper bound on the true run count. Nesting the chosen points gives a monotone
ladder from the exposure bound (no points) down to the exact count (all points), with
cost 2^(number of points) - the practitioner picks the rung the budget affords. Three
rungs (s = 3 points per gap) already clear the route's requirement at every constrained
case, including both cases the exposure bound lost.

Precise form. Machine M = gears (primes) 5 <= q <= y, period P = prod q, slot k exposed
iff k != +-6^{-1} (mod q) for all gears; q' = next prime, c = 6^{-1} mod q'; the
qualifying value set V(q') = {v in [1, F(M)] : v mod q' in {0, 2c, -2c}} (the necessary
residue condition for a merge-chain interior gap). For a gap tuple (v_1..v_m) in V^m
put X = {0, v_1, v_1+v_2, ..., v_1+..+v_m} (the m+1 opening offsets). A run of m
consecutive gaps realising the tuple at slot k requires every offset in X exposed at k
AND every strictly-interior offset blocked. For ANY subset Y of the interior offsets,

    #{k mod P : X exposed, Y blocked}  =  sum over T subseteq Y of
                                          (-1)^|T|  prod_q  c_q(X u T),

where c_q(O) = #{r mod q : r + o avoids both teeth of q for all o in O} - each term a
product of per-gear counts (CRT; this is Lateral's multi-lag c_q). Therefore

    run_m(M)  <=  sum over tuples in V^m  of  #W'(X(tuple), Y),

for every choice of Y, monotone improving as Y grows (nested Y), equal to the exact
count when Y = all interior offsets, and computable WITHOUT constructing the period.
s = 0 (Y empty) is R32's exposure bound. Positions are chosen in balanced-bisection
order so the rungs are nested. A total below 1 is a ZERO CERTIFICATE: no qualifying run
of that depth exists anywhere in the period.

Key measured values (research/tm_renewal_bound.py; exact censuses from
research/tm_resid_runs.py; p_j = run_{j-2}/ngaps):

    machine 23, j=6 (q'=29): exposure s=0 gives p_6 <= 4.3e-3 (short x29 of the
      requirement 1.5e-4); s=3 gives 4.5e-6, s=5 gives 5.0e-7 - CLEARS x300.
      Exact: 0. One tuple survives the ladder: (10,29,10,10).
    machine 29, j=5 (q'=31): s=0 3.6e-2 (short x2.0 of requirement 1.8e-2);
      s=3 2.1e-3, s=5 2.0e-4 - CLEARS x91.  Exact: 8 runs (all permutations of
      {10,10,21} - the machine's entire k=4 fuel).
    machine 29, j=6: s=5 1.4e-6 vs requirement 2.8e-3 - CLEARS x2000. Exact: 0.
    machine 19, j=5, j=6: clears (requirement was never binding). Exact: 0, 0.
    machine 31 (q'=37): p_5 <= 1.4e-2 (exact 8.2e-8: 508 runs), p_6 <= 2.8e-3
      (exact 0) - valid but loose.
    machine 37 (q'=41, period 1.24e12, NO SCAN EXISTS): p_5 <= 3.4e-2,
      p_6 <= 9.8e-3 - the first joint-gap bounds at a machine beyond scan reach.

Honest limits, measured: tightness above the exact count degrades with machine size
(x40 at machine 29 m=2, x5,249 at m=3, x176,649 at machine 31 m=3) because a FIXED
number of blocked points per gap covers a shrinking fraction of growing gaps; and the
2^|Y| inclusion-exclusion cost bars Y = all interiors, so zero certificates at depth
(totals < 1) were NOT reached - the smallest surviving total is 4 (machine 23, m=4,
s=5, true count 0). The ladder proves rate bounds, not absences.

## 2. WHY IT MIGHT BE NOVEL

* The object bounded - the joint distribution of consecutive gap SIZES in the reduced
  pattern mod a primorial, at the near-record threshold - has, as far as round 19's
  eight searches found, no published law at all (see suppression-law.md section 6);
  this entry adds the first RIGOROUS quantitative bound family on that object.
* The mechanism is elementary (Bonferroni-style inclusion-exclusion over chosen
  interior points, factorised by CRT), but its use as a TUNABLE LADDER between the
  trivial exposure bound and the exact count for gap-pattern events mod primorials,
  with nested point sets for monotonicity, appears in none of the Jacobsthal
  computation or reduced-residue literature the project has read (Hagedorn,
  Costello-Watts, Ziller-Morack, Holt-Rudd, Montgomery-Vaughan).
* It is NOT the Selberg/Brun upper-bound sieve: those bound counts of integers
  avoiding residue classes in an interval; this bounds counts of PATTERNS (several
  exposed points with prescribed blocked points between them) across the full period,
  with the blocked conditions handled exactly rather than through sieve weights.
* Classical shadow, stated honestly: "drop conditions, then restore them by
  inclusion-exclusion" is the oldest idea in the subject; the delta is only the
  application (joint gap events mod primorials), the nested-rung construction, and
  the measured fact that three rungs suffice for the route's requirement.

## 3. PROOF

Validity of each rung: a run realising the tuple has all X exposed and ALL interiors
blocked, hence all of Y blocked for any Y a subset of the interiors - so the run event
is contained in W'(X, Y), and #W' counts it. The IE formula for #W' is exact:
expanding prod_{o in Y}(1 - 1_exposed(o)) and using the fact that "exposed at every
point of a set O" is a CONJUNCTION over gears, hence CRT-factorises as prod_q c_q(O).
Monotonicity: Y subset Y' implies W'(X,Y') subset W'(X,Y). Convergence: Y = all
interiors makes W' the run event exactly. Every #W' >= 0 (it counts residues) -
asserted; every rung >= the exact census count at machines 19, 23, 29, 31 - asserted
against research/data/tm_resid_runs.csv (full-period, cyclic-seam-exact censuses).

Scripts: research/tm_renewal_bound.py (the ladder; assertions for validity,
monotonicity, and exactness comparisons), research/tm_resid_runs.py (the exact
censuses), research/anticorr_law.py (round-19 R32 baseline it extends).

## 4. IMPLICATIONS

Inside the project: (D)'s anti-correlation input now has a RIGOROUS supplier at every
measured constrained case - the two cases R32 left open (machine 23 j=6, short x28.8;
machine 29 j=5, short x2.0) are closed with factors x300 and x91 to spare. What
remains heuristic in (D) is no longer any correlation statement but only the
order-statistics step (lambda) connecting p_j to the qualifying maximum - and the
round-20 exact criterion (constructor R39: max(F2, qualmax_j) <= F + q') bypasses
even that at measured steps. The ladder is also the first tool that bounds joint gap
behaviour at machines with unscannable periods (37+), where every prior joint
quantity was scan-only.

Outside: a computable, converging family of upper bounds for pattern counts in
reduced residues mod primorials - e.g. for the Jacobsthal-adjacent question "how
often are two/three consecutive gaps all >= a?", which has no published bounds at
any tightness, this gives certified numbers at any modulus where the per-gear counts
can be enumerated.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

* Part (D) of the tolerance route (the sole open input; with R14+R21+R20+R23 it
  implies twin-prime infinitude): supplies the rigorous rate side.
* The Jacobsthal fine-structure questions of suppression-law.md: gives the bound
  side of the measured deficits.
* Zero-certification at depth (proving qualmax_j = 0 / Q_j = 0 without a scan) is
  the named next construct: it needs an exact counter that beats 2^|Y| - the natural
  candidates are a per-gear transfer DP or Mechanic's COV(M) CRT machinery.

## 6. PRIOR-ART CHECK

Not yet checked (agent without web access this round). Suggested searches: Bonferroni
inequalities for patterns in reduced residues; sieve upper bounds for configurations
of consecutive coprime gaps; Jacobsthal function joint gap bounds; "totient gaps"
consecutive pattern counts primorial.
