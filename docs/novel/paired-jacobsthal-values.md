# paired-jacobsthal-values - exact values of the paired Jacobsthal function h_2 and the per-difference family F_d(y)

Status: COMPUTED / SCRIPT-VERIFIED (finite computations, exhaustive at each y).
Prior-art verdict: PARTIAL OVERLAP - the headline h_2 values are KNOWN (Ziller-Morack
2017 computed them, and further, in a companion note the project had not found);
the per-difference family, the fixed-twin ladder, and the maximiser structure are
novel as far as searched. See section 6, checked 2026-08-23.

## 1. What it is

Plain language. Take the primes up to y and ask: how long can a run of consecutive
positions be in which no position carries a pair of numbers, two apart times some
fixed even difference, both coprime to every prime in the set? The classical
Jacobsthal function asks this for a single number; here it is asked for a PAIR at a
fixed even difference - the two-residue analogue. The project computed, for every
even difference separately and exhaustively, the maximal such run, and hence the
first values of the maximum over all differences - the paired Jacobsthal function
h_2 named by Ziller and Morack - together with which differences attain the maximum.

Precise form. Following Ziller-Morack (arXiv:1706.00317, Definitions 2.1-2.2): for
n in N, j_2(n) is the smallest m such that every paired progression <a,b>_m =
{(a+i, b+i) : i = 1..m} with 2 | (b-a) contains a pair (x,y) with gcd(x,n) =
gcd(y,n) = 1; and h_2(n) = j_2(p_n#), the primorial case. The project works in
halved coordinates (slots 6k+-1 for the twin difference; the general even
difference 2d has reduced difference e): for each e, F_e(y) is the maximal window
length avoiding a slot whose both members are coprime to the primorial of primes
<= y, and h_2 = 2 * max_e F_e (equivalently, in Ziller-Morack's condensed form,
h_2(n) = 6*omega_2(n) + 6; so max_e F_e = 3*omega_2 + 3 - the two conventions
agree at every computed point).

The computed table (exhaustive over every even difference at each y;
research/jacobsthal_family.py, jacobsthal_h2_17.py):

    y    P (odd)   #diffs    h_2   p^2-p   Conj.6   margin
    3        3         1       0      6    holds      -
    5       15         7      18     20    HOLDS    10.0%
    7      105        52      30     42    HOLDS    28.6%
    11    1155       577      66    110    HOLDS    40.0%
    13   15015      7507     150    156    HOLDS     3.8%   <- the dip
    17  255255    127627     192    272    HOLDS    29.4%

The Ziller-Morack Conjecture 6 bound (exact wording, arXiv:1706.00317, Conjecture 6:
"Let n in N >= 3. Then h_2(n) < p_n^2 - p_n") holds at all five points, but the
margin is non-monotone with a one-off dip to 3.8% at y = 13 (vs 10.0, 28.6, 40.0,
29.4). Round 17 resolved the dip: it belongs to the STEP 11->13, not to any
difference class - it needs both a twin prime step (bound grows only x1.42) and a
clean extension of the extremal delta-profile (h_2 gains fully, x2.27); at 17 the
profile must compromise (x1.28) while the bound grows x1.74.

Attached structure (same computations):

- Maximisers: y = 13: e = 344, 734, 839, 916, 2164 (all coprime to P, none small,
  none structured); y = 17: F = 96 at e = 2791, 3176, 5584, 5794, 6361, 6571.
- Delta-profile law (delta_q(e) = min(e mod q, q - e mod q)): maximisers are
  exactly the carriers of specific profiles - (1,1,1,3) at gears <= 11,
  (1,1,1,3,6) at <= 13 (16 of 7507, recall and precision 100%), (1,1,2,4,6,8) and
  (1,1,2,3,4,3) at <= 17 (precision 100%, recall 50/50). Every winning profile
  begins delta_3 = delta_5 = 1.
- Fixed twin difference (d = 2) ladder, halved coordinates: F(2,37) = 264,
  F(2,41) = 273, F(2,43) = 309, F(2,53) >= 426 (pruned search still running;
  needs <= 486 for the tolerance constant; quadratic-law prediction ~441).
- y = 19: exhaustive scan out of reach in-project (2,424,922 differences);
  lifting the gears <= 17 elite gave h_2(19) >= 222 vs bound 342. (Settled by the
  literature - see sections 4 and 6: h_2(19) = 258.)
- A failed prediction, recorded: from the first four points the project predicted
  extrapolation (~330) would BREACH the bound at y = 17; refuted the same round
  by exact computation (192, holds).

## 2. Why it might be novel

h_2 is a named function with a literature: Ziller-Morack define it, conjecture the
bound p_n^2 - p_n, and prove (their Theorem 4.1) that the bound implies Goldbach
AND the infinitude of prime pairs at every fixed even difference. The project
believed, from arXiv:1706.00317 alone, that the literature computes no exact
values ("ZM compute none"). That belief was WRONG - see section 6 - but two layers
remain that the literature does not have:

- the PER-DIFFERENCE resolution: F_e(y) computed and recorded for every even
  difference separately (7,507 differences at y = 13; 127,627 at y = 17), where
  the literature computes only the maximum over all differences;
- the fixed-twin-difference ladder F(2,y) up to y = 43 (and the y = 53 bound),
  which no published table or OEIS sequence contains;
- the maximiser identification and the delta-profile law (which differences are
  extremal and why), which the literature's condensed formulation (free residue
  pairs, difference forgotten) cannot even express without unwinding the CRT.

This is not a restatement of a classical sieve bound: the exact values are finite
combinatorial facts about two-residue coverings, and the classical shadow
(Hardy-Littlewood singular series) ranks densities, not covering maxima - the
project measured that density does not determine the extreme (F_max/lambda ranges
2.88-7.52 over the 31 gcd classes at y = 13).

## 3. Proof / verification

Status: COMPUTED / SCRIPT-VERIFIED (finite). No kernel-checked theorem claims the
table itself; each value is an exhaustive finite computation.

- research/jacobsthal_family.py, research/jacobsthal_h2_17.py - the h_2 table,
  exhaustive over every even difference, and the percentile data.
- research/why13.py, research/maximiser_shape.py, research/h2_19_lift.py - the
  dip analysis, maximiser shapes, and the y = 19 lift.
- rust2/src/bin/maxgap_pruned.rs (+ log research/data/maxgap53_pruned.log) - the
  pruned fixed-twin search; verified identical to the unpruned original on
  F(2,y) for y = 11..37 before being trusted further.
- Kernel-checked adjacent structure (proofs/Polignac.lean, standard axioms or
  fewer): endpoint_run_mod_three (F(2,y) = 0 mod 3, justifying the pruned
  search's mod-3 skip; all known exact values comply) and the mod-3 dichotomy
  for the family (3 | F_d(y) for every gear set iff d != 0 mod 6).
- External cross-check found during the prior-art sweep (section 6): all five
  h_2 values agree exactly with Ziller-Morack's independently computed table
  (different algorithms: their sequential/ILP searches vs the project's
  exhaustive per-difference scan). The project's computation is therefore an
  independent replication at the overlap points.

## 4. Implications

Inside the project:
- Conjecture 6 is the localised form of the route's target family; the margin
  table prices how much room the bound has at each scale, and the dip at 13 is
  the sharp "why is 13 extremal?" question that round 17 answered.
- The tolerance-constant computation still needs F(2,53) <= 486 - a
  per-difference value the literature does NOT supply (it only bounds
  max_e F_e(53) = 711 from h_2(16) = 1422).
- The prior-art sweep itself settles a listed open question: Ziller-Morack's
  table has h_2(19) = 258 < 342, so Conjecture 6 HOLDS at y = 19 (margin 24.6%);
  the project's lift bound (>= 222) was consistent, and its round-17 "compromise"
  scenario (~250) was the right prediction, not the "clean extension" (~288).
- Computing margins from their full table (p_n = 19..73: 24.6, 27.7, 44.6, 38.7,
  46.8, 45.5, 42.2, 40.6, 48.4, 51.6, 48.0, 50.5, 50.5, 50.1 percent), the 3.8%
  dip at 13 remains the unique extreme through p_n = 73, and the margin drifts
  toward ~50% - i.e. h_2 ~ (p_n^2 - p_n)/2 empirically. The dip observation
  survives contact with the full published table; the "one-off" qualifier is now
  verified 19 points deep instead of 5.

Outside the project: the per-difference family is the finer object - Ziller-
Morack's reduction (their Propositions 3.2/3.5) uses one uniform bound for every
even difference, while the family shows the quantity being bounded varies by more
than 2x across differences at fixed y (see twin-percentile.md). A per-difference
Conjecture 6 would be a strictly sharper, previously unformulated statement.

## 5. Unsolved questions or conjectures it touches

- Ziller-Morack Conjecture 6 (open; now known verified to p_n = 73 by their
  computation). By their Theorem 4.1 it implies Goldbach and prime pairs at every
  fixed even difference - so any exact value is data on a live conjecture.
- Goldbach's conjecture and the prime pairs (fixed-difference Polignac)
  conjecture, via that reduction.
- The project's route: lemma (D) and the tolerance constant (needs F(2,53) <= 486).
- OEIS: A288815 exists (h_2 at primorials); the per-difference family and the
  fixed-twin ladder F(2,y) are candidate new sequences.

## 6. Prior-art check (2026-08-23)

Searches run:

- WebSearch: `Ziller Morack "Jacobsthal function" computation arXiv` -> found
  arXiv:1706.03668 (the decisive hit, see below), arXiv:1611.03310, OEIS
  A288815/A072753, Hagedorn's paper.
- Fetched and read in full: arXiv:1706.00317 (Ziller-Morack, "Divisibility in
  paired progressions, Goldbach's conjecture, and the infinitude of prime
  pairs", 2017) - definitions of j_2/h_2, Conjecture 6 exact wording, Theorem
  4.1; NO numerical h_2 values in this paper.
- Fetched and read in full: arXiv:1706.03668 (Ziller-Morack, "A short note on
  the computation of the generalised Jacobsthal function for paired
  progressions", June 2017), including its 32-page ancillary full_details.pdf.
- OEIS text-interface lookups: id:A288815, id:A072753, id:A072752, id:A048670;
  sequence searches 18,30,66,150,192,258,366 (hit: A288815 only);
  21,33,54,75,102 (no results); 33,54,75,102,129 (no results); 16,28,39,57,65
  (no results); 42,66,108,150,204 (no results); 264,273,309 (no results);
  keyword search "paired Jacobsthal" (A288815 only).
- WebSearch: `"paired Jacobsthal" OR "generalised Jacobsthal function for paired
  progressions" -Ziller` -> no third-party follow-ups computing further values.
- WebSearch: `Hagedorn "Jacobsthal" function computation "prime pairs" OR twin`
  -> Hagedorn, "Computation of Jacobsthal's function h(n) for n < 50", Math.
  Comp. 78 (2009): ordinary (single-residue) function only, no paired analogue.
- Fetched: arXiv:2007.01808 (Ziller 2020, "On differences between consecutive
  numbers coprime to primorials"): the unpaired coprime-set gap spectrum -
  Ziller's own later work stayed on the single-residue side.

Nearest prior art, and the correction:

1. KNOWN - the h_2 values. Ziller-Morack's companion note arXiv:1706.03668
   (Table 1) computes h_2(n) exactly for ALL n <= 21 (p_n <= 73): 2, 6, 18, 30,
   66, 150, 192, 258, 366, 450, 570, 708, 894, 1044, 1284, 1422, 1656, 1902,
   2190, 2460, 2622 - agreeing exactly with the project's five values at
   p_n = 5..17, and extending twelve points beyond them. OEIS A288815 (Ziller,
   June 2017) carries the same values; the condensed equivalent A072753
   ("maximum gap in two-stage prime-sieves", h_2 = 6*A072753 + 6) is Ziller's
   own sequence dating to 2002, with values accreted by Ziller, Morack, and
   Giovanni Resta through 2017. The harvester claim "first exact values in the
   literature (ZM compute none)" is FALSE - the project had read arXiv:1706.00317
   (which indeed contains no values) and missed the companion computation note
   posted eleven days later. Their computation was also exhaustive over extremal
   sequences (their ancillary files list every maximal sequence in three
   representations).
2. NOT in the literature - everything per-difference. The Ziller-Morack condensed
   formulation (omega_2: two free residues per prime, the difference eliminated)
   computes only max over differences; neither paper nor ancillary files nor any
   OEIS sequence records per-difference values F_d(y), the fixed-twin ladder
   F(2,y) (21, 33, 54, 75, 102, 129, 264, 273, 309, >= 426 - all six OEIS
   sequence probes empty), which differences attain h_2, or the delta-profile
   law. No follow-up literature (2017-2026) computes further h_2 values or any
   per-difference refinement.
3. Context: ordinary-Jacobsthal computations (Hagedorn 2009 to n = 49;
   Ziller-Morack 1611.03310 to p = 251; A048670 to n = 64) are the one-residue
   analogue; Costello-Watts arXiv:1208.5342 bounds, Ford-Green-Konyagin-
   Maynard-Tao asymptotics - none paired.

VERDICT: PARTIAL OVERLAP. The five headline h_2 values and the Conjecture 6
verification are KNOWN (Ziller-Morack 2017, arXiv:1706.03668 + OEIS A288815;
their table extends to p_n = 73 and settles the project's open y = 19 case:
h_2(19) = 258 < 342). The delta that stands as novel as far as searched: the
exhaustive per-difference family F_d(y), the fixed-twin ladder F(2,y) including
F(2,37) = 264, F(2,41) = 273, F(2,43) = 309, F(2,53) >= 426, the maximiser
identification and delta-profile law, and the margin-dip analysis (the dip is
visible in their published table but nowhere commented on or explained; the
project's round-17 step-11->13 explanation has no counterpart in print). The
project's values are an independent replication of theirs at the five overlap
points, by a different method - which is corroboration, not priority.
