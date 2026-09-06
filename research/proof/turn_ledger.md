# The turn ledger (R4.c.i, prover, 2026-09-07)

Branch of R4.c (the valves), spawned by the onset law: a family (s, s') has no member before
turn max(s, s'), so in turn m = (mQ, (m + 1)Q] only the families with max(s, s') <= m and the
ember families are present. The question of the branch: per turn, the total charges T_m, the
burnt charges B_m and the pure charge P_m = T_m - B_m are counts over the same primes above Q;
does any exact relation between T_m and B_m force P_m > 0, and where does the counting face
of the wall reappear in the ledger. Sources read: valves_reconcile.md, valves_scratch.md,
valves_review.md (I7, I9, 4.1), manifold_census_large.md, the_wall.md (face A, 5l), the tree
node R4.c. Scripts in research/valves/r1/, outputs in research/valves/r1/results/ (gitignored;
every number used is in this file). Laws numbered V1, V2, ... (the valves' series).

Vocabulary as fixed by the owner: ENGINE = primes <= q; MANIFOLD = primes in (q, Q]; VALVES =
the engine acting inside the manifold's open set; EXHAUST = primes above Q; quiet zone (Q, Q^2];
CHARGE s x P (air s q-smooth, fuel P = 1 or a prime above Q); FAMILY (s, s') = the charge pairs
(sP, s'P') at distance 2; PURE CHARGE = family (1, 1) = the twins; EMBER = a q-smooth number
above Q; TURN m = (mQ, (m + 1)Q]; a pair (n, n + 2) belongs to the turn of its lower member n.

## Pre-registered (written before any ledger was computed)

The only formula-level inputs used before computing were the INVENTORY law (which (s, s')
exist) and the YIELD law (each family's density), both from valves_scratch.md; from them the
script valve_count.py produced the admissible-family count per turn and the density weights
below. No ledger data was seen when this section was written.

**Definitions used.** For a turn m and a q-smooth s <= m let pi_{m,s} = the primes in
(mQ/s, (m + 1)Q/s] (all above Q since s <= m). E_m = the embers in turn m (q-smooth numbers in
(mQ, (m + 1)Q]). A charge in turn m is sP with P in pi_{m,s}, or an ember. T_m = charge pairs
(n, n + 2) with n in turn m; B_m = those with air on either member; P_m = those with none.
B_m^fuel = burnt pairs with both fuels > 1; B_m^ember = burnt pairs with an ember member.
A_m = the primes P in turn m with P + 2 manifold-open (the prime-led charge pairs); A'_m the same
with P - 2.

**Theory.** The ledger is exact and structural in its bookkeeping (which families are present,
which residues they occupy) and counting in its values (each family's count is a count of primes
in an interval in the imprint's residue classes). The onset law makes the family sum finite per
turn; the ember law (turns 1, 2) bounds B_m^ember by 2 E_m. Nothing structural relates T_m to B_m
because both are sums of the same kind of prime count with P_m one of the summands; every
inequality B_m <= c T_m with c < 1 is a lower bound on the twin count in the turn in terms of
other prime counts, i.e. the conjecture in the turn's coordinates. What the ledger CAN state
exactly is the shape of the burnt side (its families, its densities' Euler product) and where the
pure share goes as the valves open.

**Predictions with numbers** (scored EXACT / MEASURED / REFUTED below):

- V-P1 (the ledger identities). P_m = T_m - B_m and B_m = B_m^fuel + B_m^ember with
  B_m^fuel = sum over admissible fuelled (s, s') with max(s, s') <= m of N_m(s, s'), exactly
  (a burnt pair with both fuels > 1 has s <= m and s' <= m; the edge case s' = m + 1 needs
  m = 1 and P' = Q + 1 with n = 2Q an ember, so it is not fuelled). Expect 0 exceptions in every
  (q, Q, m). The owner's brief writes B_m as the family sum alone; I predict the ember term is
  needed and is nonzero in most turns (embers exist in every turn m <= 60 at Q <= 10^5 since
  the powers of 2 alone put one in (mQ, (m + 1)Q] whenever Q >= ... not always: expect
  E_m >= 1 for most m and 0 for some at Q = 10^5, m near 60).
- V-P2 (ember bound). B_m^ember <= 2 E_m for every m (each ember carries at most two pairs);
  and for m = 1, 2: B_m = B_m^ember (proved in valves_scratch.md). 0 exceptions.
- V-P3 (valve count). The fuelled families present in turn m are exactly the admissible pairs
  with max(s, s') <= m; count A(m) per the table below (q = 5: 1, 1, 3, 5, 9, 11, 11, 15, 19,
  23, 23, 27, ..., A(60) = 125; q = 7: ..., A(60) = 233; q = 11: ..., A(60) = 327). Expect
  equality for every m <= 60 at Q >= 10^4 and every q; expect a few misses (delay > 0) at
  Q = 10^3 for m >= 9 (the scratch lane saw (9, 5) miss its turn at Q = 10^3).
- V-P4 (no structural inequality). No inequality B_m <= c(m) T_m with c(m) < 1 holds for a
  structural reason. Candidates tested on the tables (each scored hold / fail, with the breaking
  (q, Q, m)):
  - C1: B_m < T_m, i.e. P_m > 0. Expect 0 exceptions at Q >= 10^3, m <= 60 (twin-free runs of
    length Q do not exist there). ROOT: its proof is the conjecture.
  - C2: B_m^ember <= 2 E_m. Structural (proved), 0 exceptions.
  - C3: B_1 <= P_1 and B_2 <= P_2 (embers' charges do not outnumber the twins). Expect 0
    exceptions at Q in [10^3, 10^5] and failures at small Q (Q = 30: P_1 = 2, B_1 = 4, from the
    scratch lane's counts). Counting, not structural.
  - C4: B_m / T_m non-decreasing in m. Expect FAILURES (it is a density with square-root
    fluctuations; the drop at a turn where no new family opens is possible). Owner's brief:
    "it should rise with m as valves open". I predict: rises at the onset turns, fluctuates
    between them, and the monotone version fails at least once per (q, Q).
  - C5: B_m / T_m <= 1 - 1/Sigma_inf(q) with Sigma_inf = (3/2) prod_{3 <= p <= q} p/(p - 2)
    (values 7.5, 10.5, 12.8333; bounds 0.8667, 0.9048, 0.9221). Expect it to hold at every
    m <= 60 with Q >= 10^4 and to fail nowhere or only at Q = 10^3 where the counts are small.
    Counting.
  - C6: P_m / A_m >= prod_{3 <= p <= q} (p - 2)/(p - 1) (0.375, 0.3125, 0.28125), the placement
    law as a per-turn lower bound. Expect it to hold at every m <= 60 (the families (1, s') with
    s' > 60 are not open yet) with possible fluctuation failures at Q = 10^3. Counting.
  - C7: N_m(1, 1) >= N_m(s, s') for every burnt family in every turn (the pure charge is the
    largest family in every turn). Expect 0 exceptions at Q >= 10^4 (each burnt family's
    density weight w(s, s') = (1/(s s')) prod_{odd p | s s'} (p - 1)/(p - 2) is <= 2/3 of the
    pure charge's), possible exceptions at Q = 10^3 in turn 3 from (1, 3)/(3, 1) whose fuel
    sits at the smaller scale. Counting.
- V-P5 (the pure share does NOT vanish with the turn). Owner's brief asks whether B_m / T_m
  tends to 1. I predict no: as m grows every admissible family opens, and the sum of their
  density weights converges to the Euler product Sigma_inf(q) = (3/2) prod_{3 <= p <= q}
  p/(p - 2) (derivation: odd coprime pairs give prod (1 + 2/(p - 2)) = prod p/(p - 2); even
  pairs, where exactly one member is 2 mod 4, add half of that). So T_m / P_m tends to
  Sigma_inf (up to the log corrections of the families' scales) and B_m / T_m to
  1 - 1/Sigma_inf = 1 - (2/3) prod (p - 2)/p: 0.8667 (q = 5), 0.9048 (q = 7), 0.9221 (q = 11).
  Numbers to test at m = 60: predicted Sigma(60) = 6.51, 8.37, 9.45 (partial sums over
  max <= 60), so B/T at m = 60 predicted 0.846, 0.880, 0.894 before log corrections; I expect
  the measured T_m / P_m at Q = 10^4, m = 60 to lie between Sigma(60) and 1.3 Sigma(60) (the
  families at large air have their fuel at the smaller scale mQ/s where primes are denser). The
  share vanishes with q, not with m: 1/Sigma_inf ~ (log q)^-2.
- V-P6 (the prime-led share and the placement law). P_m / A_m = 1 at m = 1, 2 up to the
  embers (the (1, s') valves are closed), then steps down at m = 3, 5, 9, 15, ... as (1, 3),
  (1, 5), (1, 9), (1, 15) open, tracking 1 / Sigma_1(m) with Sigma_1(m) = sum over odd smooth
  s' <= m of (1/s') prod_{p | s'} (p - 1)/(p - 2): predicted 0.600 (m = 3, 4), 0.517 (5..8),
  0.464 (9..14), 0.429 (15..24), 0.397 (m = 60) at q = 5, limit 0.375 = prod (p - 2)/(p - 1);
  the same object as the review's coupling (I9) and placement law (I7), now per turn. Expect
  the measured ratio within 10% of the prediction at Q >= 10^4 for m >= 3.
- V-P7 (first two turns, the scan). Over every integer Q in [10^3, 10^5] (q = 5, 7): P_1 > 0
  and P_2 > 0 always; min P_1 near Q = 10^3 of order 25 (twins in (1000, 2000] number 26 by
  hand), min P_2 similar; B_1 <= P_1 and B_2 <= P_2 always in that range. Over all Q >= 1:
  P_1 = 0 exactly at Q = 1 and Q = 5 (turn (5, 10] has no twin), nowhere else; the last Q with
  B_1 > P_1 is below 100 (Q = 30 has B_1 = 4 > P_1 = 2).

**Owner's predictions on the scorecard** (from the brief): (O1) B_m equals the sum over
families with max(s, s') <= m of their turn count (I predict: plus the ember term); (O2) an
inequality B_m <= c(m) T_m with c(m) < 1 for small m may follow from the structure alone (I
predict: none does; every such inequality is counting); (O3) B_m / T_m rises with m as valves
open and may tend to 1 (I predict: rises to 1 - 1/Sigma_inf < 1); (O4) P_1 >= ember charges in
turn 1 (I predict: yes throughout [10^3, 10^5], no at small Q).

**What would refute each.** V-P1: a burnt pair with both fuels > 1 and max(s, s') > m.
V-P2: an ember carrying three pairs (impossible: at most (e - 2, e) and (e, e + 2)). V-P3: a
family present before its turn (impossible) or absent at its turn at Q >= 10^4 for m <= 60.
V-P5: the measured B_m / T_m at m = 60 above 1 - 1/Sigma_inf, or T_m / P_m outside
[Sigma(60), 1.3 Sigma(60)] at Q = 10^4. V-P6: P_m / A_m off by more than 10%. V-P7: a Q in
[10^3, 10^5] with P_1 = 0 or B_1 > P_1.

## Scorecard (filled after the runs)

| prediction | verdict | evidence |
|---|---|---|
| V-P1 ledger identities, ember term needed | EXACT. P_m = T_m - B_m and B_m = sum over I(m) \ (1, 1) of N_m(s, s') + B_m^ember in all 600 turns (10 ledgers x 60); 0 fuelled burnt pairs with max(s, s') > m; the ember term is nonzero in 50 of 60 turns at (5, 10^4), 47 at (5, 3 x 10^4), 51 at (5, 10^5), all 60 at every q >= 7 run; E_m = 0 in 2 turns of (5, 10^3) only | Results 1, appendix |
| V-P2 B^ember <= 2 E_m; B_m = B^ember for m <= 2 | EXACT, 0 exceptions in 600 turns; the m <= 2 half is the scratch lane's proof | Results 1 |
| V-P3 families present = admissible with max <= m | HALF EXACT, HALF REFUTED. No family is ever present before its turn (0 extra families in 600 turns). Presence at every turn from the onset on fails: equality holds for m <= 8 (Q = 10^3), m <= 19 / 19 / 15 (Q = 10^4, q = 5 / 7 / 11), m <= 26 / 25 / 22 (3 x 10^4), m <= 49 (5, 10^5); the first misses are sparse valves ((16, 18) at m = 20, (25, 27) at m = 27, (32, 50) at m = 50) whose expected count in the turn is below 1. My "every m <= 60 at Q >= 10^4" was wrong: presence per turn is a density, the onset bound is the law | Results 1, valve count |
| V-P4 no structural inequality; C1..C7 | PROVED (V2, the counterfactual fuel): no inequality B_m <= c T_m with c < 1 follows from the imprint, onset, port, inventory and ember laws. C1 holds with 0 exceptions (ROOT); C2 0 exceptions (structural); C3 fails at (7, 10^3) and (11, 10^3) in turn 2 and at every q for small Q; C4 fails everywhere (20-29 decreases per ledger); C5 fails in 9 of 10 ledgers; C6 fails in all 10; C7 holds at Q >= 10^4 (420 turns) and fails 13 times at each Q = 10^3 | Inequalities |
| V-P5 pure share does not vanish; Sigma_inf = (3/2) prod p/(p - 2); T/P in [Sigma(60), 1.3 Sigma(60)] | FORM CORRECT, VALUE REFUTED, THEN CORRECTED. The measured T/P was 1.5-1.9 times the pre-registered Sigma(60) and 1.25-1.33 times its scale-corrected version, uniformly across q and Q. The error was mine: the even families weigh 2/(s s') times the odd factor, not 1/(s s') (the partner's parity is forced by the class of the fuel, doubling its prime chance; the scratch lane's yield formula already had this). With the corrected weight and the scale factor L_m, the aggregate T/P over turns 30..60 is within 2.2% of the prediction at all 7 runs with Q >= 10^4 (ratios 0.999, 1.002, 1.006, 0.978, 0.978, 0.980, 0.996) and within 4% at Q = 10^3. The corrected limit is Sigma_inf = 2 prod_{3 <= p <= q} p/(p - 2) = 10, 14, 17.11, so B/T -> 1 - (1/2) prod (1 - 2/p) = 0.900, 0.929, 0.942: the pure charge's share of the charges tends to the engine's own pair-opening density, not to 0 | Valve count law |
| V-P6 P/A tracks 1/Sigma_1(m) within 10% | REFUTED as stated (deviations to 25% at Q = 10^4, 13% at 10^5, all one-signed), CORRECT with the scale factor: aggregate P/A over turns 30..60 within 1.3% of 1/Sigma_1^log at all 10 runs (ratios 0.981 .. 1.013) | Valve count law |
| V-P7 first two turns | EXACT for P_1, P_2 > 0 on [10^3, 10^5] (all 99,001 Q, q = 5, 7, 11, 13); P_1 = 0 exactly at Q = 1, 5 and P_2 = 0 exactly at Q = 3, 9 (EXACT, as predicted for P_1); min P_1 = 25 at Q = 1031 (predicted "about 25 near 10^3"); B_1 <= P_1 on [10^3, 10^5] for q = 5, 7, 11 but NOT q = 13 (279 failures, last Q = 1355); B_2 <= P_2 REFUTED for q >= 7: 145 failures at q = 7 (last Q = 1150), 1,331 at q = 11 (last 2,963), 5,398 at q = 13 (last 6,562); "last B_1 > P_1 below 100" true at q = 5 (Q = 74), false at q = 7 (Q = 400) | First two turns |
| Owner O1 (B_m = the family sum) | needs the ember term: B_m^ember > 0 in 50-60 of 60 turns per ledger, at most 2 E_m, and it is ALL of B_m in turns 1, 2 | Results 1 |
| Owner O2 (a structural B_m <= c(m) T_m) | REFUTED by proof (V2): the fuel set F = {n > Q coprime to q#, n = 1 mod 3} satisfies every proved valve law and has P_m = 0, B_m = T_m > 0 in every turn | Where the wall reappears |
| Owner O3 (B/T rises with m, tends to 1?) | rises at the onset turns (0.021, 0.088, 0.622, 0.762, 0.819, 0.850, 0.885, 0.903 at m = 1, 2, 3, 5, 9, 15, 30, 60 for (5, 10^4)) and does NOT tend to 1 in m: limit 1 - (1/2) prod_{3 <= p <= q} (1 - 2/p); it tends to 1 in q like 1 - c/(log q)^2 | Valve count law |
| Owner O4 (P_1 >= ember charges) | yes on [10^3, 10^5] for q <= 11; no for q = 13 until Q = 1355; no at small Q (last failure Q = 74, 400, 726, 1355 for q = 5, 7, 11, 13) | First two turns |

## Setup

Scripts (research/valves/r1/): ledger.py q Q [M] (the ledger, M = 60 turns; sieves [1, 61 Q + 2],
marks the manifold-open numbers by the primes in (q, Q], strips the q-smooth air of every member
of every open pair, checks every fuel is 1 or a prime above Q: 0 failures in 10 runs); first_turns.py
Qmax (turns 1 and 2 for every integer Q <= Qmax by prefix sums, cross-checked by brute force at
Q = 30, 210, 1000, 2310, 10000, 12345 for each q); valve_count.py (the admissible-family count per
turn and the density weights from the inventory and yield laws); yield_check.py (the scale-corrected
yield prediction per turn); inequalities.py (the candidate tests and the tables); counterfactual.py
q Q [M] (the ledger with the exhaust replaced by a fuel set that satisfies every structural law).
Outputs in research/valves/r1/results/ (gitignored). Runs: (q, Q) = (5, 7, 11) x (10^3, 10^4,
3 x 10^4) and (5, 10^5), turns 1..60; the first-turns scan for every Q from 1 to 10^5 at q = 5, 7,
11, 13. Wall time under a minute per ledger on one core; memory under 1 GB.

Notation. I(m) = the admissible fuelled families with max(s, s') <= m (INVENTORY: q-smooth,
gcd | 2, same parity, 4 dividing exactly one of an even pair); A(m) = |I(m)|. pi_{m,s} = the primes
in (mQ/s, (m + 1)Q/s]. N_m(s, s') = the family's count in turn m. E_m = embers in turn m. A_m =
primes P in turn m with P + 2 manifold-open; pi_m = primes in turn m. w(s, s') = the family's
density weight relative to the pure charge; L_m(s, s') = its scale factor in turn m (both defined in
the valve count law section).

## Results

### 1. The ledger, exact (identities)

Totals over turns 1..60:

| q | Q | sum T | sum B | sum P | sum B^ember | min E_m | turns with B^ember = 0 | min P_m (turn) | max P_m (turn) |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 10^3 | 6,573 | 5,787 | 786 | 122 | 0 (2 turns) | 15 | 6 (24) | 26 (1) |
| 7 | 10^3 | 8,323 | 7,537 | 786 | 338 | 2 | 0 | 6 (24) | 26 (1) |
| 11 | 10^3 | 9,446 | 8,660 | 786 | 652 | 5 | 0 | 6 (24) | 26 (1) |
| 5 | 10^4 | 41,306 | 36,101 | 5,205 | 150 | 1 | 10 | 65 (43) | 137 (1) |
| 7 | 10^4 | 51,135 | 45,930 | 5,205 | 476 | 3 | 0 | 65 (43) | 137 (1) |
| 11 | 10^4 | 56,588 | 51,383 | 5,205 | 999 | 7 | 0 | 65 (43) | 137 (1) |
| 5 | 3 x 10^4 | 103,029 | 89,711 | 13,318 | 161 | 1 | 13 | 175 (47) | 344 (1) |
| 7 | 3 x 10^4 | 127,003 | 113,685 | 13,318 | 519 | 4 | 0 | 175 (47) | 344 (1) |
| 11 | 3 x 10^4 | 139,962 | 126,644 | 13,318 | 1,187 | 12 | 0 | 175 (47) | 344 (1) |
| 5 | 10^5 | 284,975 | 247,738 | 37,237 | 172 | 1 | 9 | 513 (50) | 936 (1) |

The pure column is the same for every q at fixed Q (786; 5,205; 13,318): the count identity turn
by turn. The identities checked in every one of the 600 turns: P_m = T_m - B_m; B_m = B_m^fuel +
B_m^ember; B_m^fuel = sum over (s, s') in I(m) \ (1, 1) of N_m(s, s') (0 fuelled burnt pairs
with max(s, s') > m, so the sum over the open valves is exactly the fuelled burnt count);
B_m^ember <= 2 E_m; in turns 1 and 2, B_m^fuel = 0. The first three turns at Q = 10^4:

| q | m | T | B | B^fuel | B^ember | P | E | pi | A | fuelled families present |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 1 | 140 | 3 | 0 | 3 | 137 | 37 | 1,033 | 139 | (1, 1) |
| 5 | 2 | 137 | 12 | 0 | 12 | 125 | 21 | 983 | 126 | (1, 1) |
| 5 | 3 | 328 | 204 | 195 | 9 | 124 | 19 | 958 | 219 | (1, 1), (1, 3) 97, (3, 1) 98 |
| 7 | 1 | 148 | 11 | 0 | 11 | 137 | 89 | 1,033 | 142 | (1, 1) |
| 7 | 2 | 155 | 30 | 0 | 30 | 125 | 56 | 983 | 127 | (1, 1) |
| 7 | 3 | 344 | 220 | 195 | 25 | 124 | 47 | 958 | 222 | (1, 1), (1, 3) 97, (3, 1) 98 |
| 11 | 1 | 164 | 27 | 0 | 27 | 137 | 160 | 1,033 | 149 | (1, 1) |
| 11 | 2 | 182 | 57 | 0 | 57 | 125 | 106 | 983 | 129 | (1, 1) |
| 11 | 3 | 361 | 237 | 195 | 42 | 124 | 87 | 958 | 225 | (1, 1), (1, 3) 97, (3, 1) 98 |

(The turn-3 fuelled burnt count 195 is the same for all three q: (1, 3) and (3, 1) are the same
families whatever the engine; only the ember term grows with q.) At Q = 10^5, q = 5: turn 1 has
T = 938, B = 2, P = 936, E = 52; turn 2 has T = 852, B = 18, P = 834; turn 3 has T = 2,007,
B = 1,197 (1,184 fuelled), P = 810. At Q = 3 x 10^4, q = 5: 349 / 5 / 344, 317 / 12 / 305,
730 / 422 / 308.

The families contributing at each turn (fuelled, burnt) are listed with their counts for
m <= 12 under each ledger table in the appendix; the list at every turn is a subset of I(m)
(0 exceptions), and equals I(m) up to the turn given in the scorecard (V-P3). The families
opening in turns 3..12 are: m = 3: (1, 3), (3, 1); m = 4: (2, 4), (4, 2); m = 5: (1, 5), (3, 5),
(5, 1), (5, 3); m = 6: (4, 6), (6, 4); m = 7 (q >= 7): (1, 7), (3, 7), (5, 7), (7, 1), (7, 3),
(7, 5); m = 8: (2, 8), (6, 8), (8, 2), (8, 6); m = 9: (1, 9), (5, 9), (9, 1), (9, 5) and (7, 9),
(9, 7) for q >= 7; m = 10: (4, 10), (8, 10), (10, 4), (10, 8); m = 11 (q = 11): the ten pairs
(1, 11), (3, 11), (5, 11), (7, 11), (9, 11) and mirrors; m = 12: (2, 12), (10, 12), (12, 2),
(12, 10). Turn 7 at q = 5 and turn 11 at q <= 7 open nothing: 7 and 11 are not their air.

The full per-turn ledgers (T, B, B^fuel, B^ember, P, E, pi, A, families present / admissible,
B/T, P/A for m = 1..60) are in the appendix, one table per (q, Q).

### 2. The two counts over the same primes (exact expressions)

A charge in turn m is s P with s q-smooth, s <= m, and P in pi_{m,s} = pi cap (mQ/s, (m + 1)Q/s]
(every such P exceeds Q because s <= m), or an ember e in E_m (s = e, P = 1). The decomposition
n = s P (smooth part times rough part) is unique, so the charge set of turn m is the disjoint union
C_m = union over smooth s <= m of s pi_{m,s}, union E_m. Write C for the charges of the whole zone.

- T_m = sum over smooth s <= m, sum over P in pi_{m,s} of [sP + 2 in C] + sum over e in E_m of
  [e + 2 in C].
- For a fuelled family, N_m(s, s') = #{P in pi_{m,s} : (sP + 2)/s' is an integer in pi}; the
  integrality is the congruence P = -2/s (mod s') (or (aP + 1)/b for even pairs), one class
  modulo s' (modulo s'/2), and the imprint law says which classes of P modulo q# it occupies.
- B_m = sum over (s, s') in I(m), (s, s') != (1, 1), of N_m(s, s') + B_m^ember, where B_m^ember
  = #{pairs in turn m with an ember member} <= 2 E_m, and for m <= 2 the sum is empty.
- P_m = N_m(1, 1) = #{P in pi_{m,1} : P + 2 in pi}.
- The prime-led split: A_m = #{P in pi_{m,1} : P + 2 in C} = P_m + sum over s' >= 3 with
  (1, s') in I(m) of N_m(1, s') + #{P in pi_{m,1} : P + 2 in E}. Every prime P of the turn with
  an open partner is in exactly one family (1, s'), s' = the air of P + 2; the engine gear p
  puts P into a burnt family iff P = -2 (mod p): one class per gear, the placement law (review
  I7). Likewise A'_m with P - 2 and the families (s, 1).

The alternating expression. Legendre over the engine on the charge pairs:

  P_m = sum over d1 | q#, d2 | q# of mu(d1) mu(d2) M_m(d1, d2),
  M_m(d1, d2) = #{charge pairs (n, n + 2) in turn m : d1 | n, d2 | n + 2}
             = sum over (s, s') with d1 | s, d2 | s' of N_m(s, s') (ember families included).

Terms with an odd prime dividing both d1 and d2 vanish (p cannot divide n and n + 2); 2 may
divide both. Substituting the family sums, the double Mobius sum telescopes to N_m(1, 1)
exactly (the sum over d | s of mu(d) is [s = 1]): the alternating expression is an identity with
no content of its own. Its content in a sieve would be estimates of M_m(d1, d2) by (expected
density) x (charges in the turn) with an error, i.e. a level of distribution for the charges in
progressions modulo d1 d2, with each odd engine prime removing two classes of n: a sieve of
dimension 2 on the charge set. That is face A verbatim (the_wall.md A1; review 4.1), reached
here in three lines from the ledger. Nothing in the turn structure changes the dimension: the
families with p | s and the families with p | s' are the two removed classes of p.

The scale factor. The families' fuel sits at different scales: (s, s') in turn m has P in
(mQ/s, (m + 1)Q/s] and partner P' near mQ/s', where primes are denser than at mQ by
log(mQ)/log(mQ/s). This is the one non-arithmetic ingredient of the ledger and it is what the
naive weight sum misses (scorecard V-P5). Explicitly, relative to the pure charge, family
(s, s') in turn m carries L_m(s, s') = int_{mQ}^{(m + 1)Q} dn / (log(n/s) log((n + 2)/s')) divided
by int dn / log^2 n. The largest L over the families open at m = 60: 2.39, 2.47, 2.47 (Q = 10^3,
q = 5, 7, 11), 2.00, 2.05, 2.05 (10^4), 1.88, 1.92, 1.92 (3 x 10^4), 1.77 (5, 10^5); the bound is
(log(61 Q)/log Q)^2 = 2.54, 2.09, 1.96, 1.84.

### 3. Inequalities (each with status and breaking instance)

The pre-registered candidates on the ten ledgers (600 turns). Exception counts per ledger are
listed as q = 5 / 7 / 11 at each Q.

| candidate | status | Q = 10^3 | Q = 10^4 | Q = 3 x 10^4 | (5, 10^5) | first breaking instance (q, Q, m, value) | kind |
|---|---|---|---|---|---|---|---|
| C1 B_m < T_m (P_m > 0) | holds, 0 exceptions in 600 turns | 0/0/0 | 0/0/0 | 0/0/0 | 0 | none (min P_m = 6 at (q, 10^3, 24)) | ROOT: it is the conjecture per turn |
| C2 B_m^ember <= 2 E_m | holds, 0 exceptions | 0/0/0 | 0/0/0 | 0/0/0 | 0 | none | structural (proved: an ember carries at most (e - 2, e) and (e, e + 2)) |
| C3 B_1 <= P_1 and B_2 <= P_2 | fails at small Q | turn 2 fails at q = 7 (22 > 21) and q = 11 (36 > 21) | holds | holds | holds | (7, 10^3, 2, B = 22 > P = 21); scan: last failures Q = 74 / 400 / 726 / 1355 (turn 1), 555 / 1150 / 2963 / 6562 (turn 2) for q = 5 / 7 / 11 / 13 | counting (the embers' count against the twins' count) |
| C4 B_m / T_m non-decreasing | fails everywhere | 28/26/27 decreases | 26/29/26 | 22/21/20 | 22 | (5, 10^3, 6: 0.7927 -> 0.7738); (5, 10^4, 7: 0.8054 -> 0.7875); (5, 10^5, 7: 0.7828 -> 0.7804) | not a law: a density with square-root scatter; it rises at the onset turns only |
| C5 B_m / T_m <= 1 - 1/Sigma_inf (0.900, 0.9286, 0.9416) | fails in 9 of 10 | 25/14/13 | 14/7/1 | 4/1/0 | 5 | (5, 10^3, 12, 0.9009); (5, 10^4, 29, 0.9001); (7, 10^4, 43, 0.9333); (11, 10^4, 46, 0.9421); (5, 3 x 10^4, 40, 0.905); (7, 3 x 10^4, 47, 0.9295); (5, 10^5, 50, 0.9056) | counting; the limit is approached from above inside the zone (scale factor) |
| C6 P_m / A_m >= prod (p - 2)/(p - 1) (0.375, 0.3125, 0.28125) | fails in 10 of 10 | 29/25/20 | 26/14/8 | 17/7/2 | 19 | (5, 10^3, 12, 0.3438); (5, 10^4, 16, 0.360); (7, 10^4, 21, 0.2957); (11, 10^4, 36, 0.2717); (5, 3 x 10^4, 18, 0.3733); (7, 3 x 10^4, 40, 0.3026); (11, 3 x 10^4, 40, 0.2805); (5, 10^5, 28, 0.3689) | counting; same mechanism as C5 |
| C7 N_m(1, 1) >= every burnt family's N_m | holds at Q >= 10^4 (420 turns), fails at 10^3 | 13/13/13 | 0/0/0 | 0/0/0 | 0 | (q, 10^3, 12): P = 11 against (3, 1) = 14, all three q | counting (density weights: every burnt family has w <= 2/3, with the scale factor up to 1.6) |

What holds without exception: C1 (ROOT), C2 (structural), C7 at Q >= 10^4 (counting; its
proof would be the Hardy-Littlewood density of each family). What fails: every candidate that
puts a number between B_m and T_m other than the trivial B_m <= T_m.

### 4. Where the wall reappears

Sorting the ledger's exact statements by what their proof needs:

Structural (proof uses only the imprints and the onsets, i.e. only that a fuel is an odd integer
above Q coprime to q#):
- the identity P_m = T_m - sum over I(m) \ (1, 1) of N_m(s, s') - B_m^ember (V1);
- the finiteness and the exact list of the valves per turn, |I(m)| = A(m) (V3);
- B_m^ember <= 2 E_m, and B_m = B_m^ember for m <= 2 (the scratch lane's ember law);
- the placement of every family on its imprint, the port, the inventory (the reconcile's laws);
- the prime-led identity A_m = P_m + sum N_m(1, s') + (prime, ember) pairs, and that the gear p
  sends P to a burnt family iff P = -2 (mod p) (one class per gear).

Counting (proof needs how many primes lie in an interval, or in an interval and a residue class):
- every value in the ledger: T_m, B_m, P_m, N_m(s, s'), A_m, E_m (E_m is a smooth-number count,
  elementary but a count);
- C3, C5, C6, C7, and the yield law V4 with its limit.

ROOT (the proof would be the conjecture): C1, B_m < T_m, in every turn; and any B_m <= c T_m with
c < 1.

The deliverable of section 3 of the brief, stated once. THE STRUCTURAL LAWS OF THE VALVES CANNOT
SEPARATE P_m FROM B_m, AND THIS IS A THEOREM, NOT A FAILURE TO FIND THE INEQUALITY (V2). Every
proved valve law (imprint, port, inventory, onset, air cap, ember law) is proved from three
properties of the fuel: it is an integer above Q, it is odd, it is coprime to q#. Primality is
never used. So every such law holds verbatim when the exhaust is replaced by any set F of odd
integers above Q coprime to q# (a charge is then s f with f in F or f = 1, and the smooth/rough
decomposition is still unique). Take F = {n > Q : gcd(n, q#) = 1, n = 1 (mod 3)}. A pure charge
would need f and f + 2 both in F, impossible since they differ mod 3; so P_m = 0 in every turn.
The family (1, 3) survives (f = 3 f' - 2 is 1 mod 3 whenever f' is), so do (5, 1), (2, 4), (3, 5),
(1, 9), ...; B_m = T_m > 0. Measured (counterfactual.py): at (5, 10^3), m = 1..12, T_m = B_m =
3, 11, 43, 64, 98, 112, 106, 106, 129, 129, 124, 142 with P_m = 0 throughout, 0 violations of
the imprint, onset, port, inventory and ember laws; at (7, 10^4), m = 1..6: 13, 36, 265, 441,
694, 755, again 0 violations, with (1, 3) = 238, (5, 1) = 190, (2, 4) = 179, (3, 5) = 64 in
turn 6. Consequently no inequality of the form B_m <= c(m) T_m with c(m) < 1, and no lower
bound on P_m, is a consequence of the structural laws: any such inequality is false in a model
that satisfies all of them. The one property of the exhaust that separates the real ledger from
the counterfactual is that it is the set of PRIMES above Q, and what the primes have that F does
not is their count in every interval and every reduced residue class (the primes are equidistributed
mod 3, F is not). That is the counting face, and the ledger says precisely that no other face is
involved in the turn statement: the wall at the turn ledger is face A and nothing else.

The sharpest structural statement that is NOT the conjecture (the deliverable): in turn m the
burnt side of the ledger is the finite explicit list I(m) of A(m) - 1 fuelled valves, each pinned
to its imprint, none present before its air, plus at most 2 E_m ember charges, and every burnt
charge with a prime member is a prime P of the turn with P = -2 or P = 0 modulo an engine gear;
in turns 1 and 2 the burnt side is the ember charges alone, at most 2 E_m of them, E_m the
q-smooth numbers in (mQ, (m + 1)Q]. Everything the pure charge has to beat is named, placed and
bounded by the smooth numbers; whether it beats it is a count of primes.

### 5. The first two turns

Every integer Q from 1 to 10^5, q = 5, 7, 11, 13 (first_turns.py; exact, cross-checked by brute
force at six Q per q). In turns 1 and 2 the only fuelled family is (1, 1) (proved), so
T_m = P_m + B_m with B_m the ember charges: pairs (e, e + 2) or (e - 2, e) with e a q-smooth
number in the turn and the partner a prime above Q, 2 x (a prime above Q) (turn 2 only), or
another ember.

- P_1 = 0 exactly at Q = 1 and Q = 5 (turn (5, 10] has no twin), for every q; P_1 > 0 for every
  Q from 6 to 10^5. P_2 = 0 exactly at Q = 3 and Q = 9 ((6, 9] and (18, 27] have no twin);
  P_2 > 0 for every Q from 10 to 10^5. Min of P_1 + P_2 on [10^3, 10^5]: 45 at Q = 1,031.
- Minimum of P_1 over Q in [10^3, 10^5]: 25, at Q = 1,031 (turn (1031, 2062]; the largest twin
  gap touching the turn is 120, from 1,487 to 1,607). Minimum of P_2: 17, at Q = 1,071 (turn
  (2142, 3213]; largest twin gap touching it 168, from 2,381 to 2,549). Both minima sit at the
  bottom of the range, as the density 2 C_2 Q / log^2 Q says they must; the twin gaps there are
  ordinary. P_1 <= P_2 for 3.9% of Q.
- Is P_1 >= B_1 always? On [10^3, 10^5]: yes for q = 5, 7, 11 (0 failures in 99,001 Q each);
  no for q = 13 (279 failures, the last at Q = 1,355). Over all Q >= 1 the last failure is at
  Q = 74 (q = 5), 400 (q = 7), 726 (q = 11), 1,355 (q = 13), and the number of failing Q is 62,
  248, 725, 1,277. Turn 2 is weaker: B_2 > P_2 last at Q = 555 (q = 5; 386 failures), 1,150
  (q = 7; 859), 2,963 (q = 11; 2,328), 6,562 (q = 13; 6,395). The crossovers grow with q because
  the embers grow like (log Q)^{pi(q) - 1} while the twins grow like Q / log^2 Q: the embers'
  charges are the manifold's record-shorteners (W103) and they outnumber the twins in the first
  two turns until Q is in the hundreds or thousands.
- The extremes on [10^3, 10^5]: max B_1 = 6 (q = 5, at Q = 1,013), 14 (q = 7, Q = 2,592), 31
  (q = 11, Q = 50,176), 60 (q = 13, Q = 98,560); max B_2 = 21 (q = 5, Q = 54,675), 51 (q = 7,
  Q = 93,333), 105 (q = 11, Q = 95,040), 182 (q = 13, Q = 95,040). Max B_1 / P_1 = 0.240, 0.407,
  0.731, 1.320; max B_2 / P_2 = 0.706, 1.412, 2.294, 3.000. Min embers E_1 = 22, 46, 73, 104 and
  E_2 = 14, 31, 52, 75. B_1 = 0 never happens on [10^3, 10^5] for any q (some ember always has an
  open neighbour).
- The 50-step table (Q = round(10^(3 + 2k/49))), q = 5:

| Q | P_1 | B_1 | E_1 | P_2 | B_2 | E_2 |
|---|---|---|---|---|---|---|
| 1000 | 26 | 5 | 22 | 21 | 10 | 15 |
| 1099 | 27 | 6 | 24 | 19 | 12 | 15 |
| 1207 | 31 | 5 | 24 | 25 | 11 | 15 |
| 1326 | 28 | 3 | 24 | 29 | 9 | 15 |
| 1456 | 32 | 3 | 24 | 35 | 10 | 17 |
| 1600 | 34 | 4 | 25 | 40 | 9 | 16 |
| 1758 | 38 | 4 | 26 | 37 | 8 | 17 |
| 1931 | 42 | 3 | 27 | 39 | 9 | 16 |
| 2121 | 47 | 2 | 27 | 37 | 12 | 17 |
| 2330 | 51 | 2 | 27 | 41 | 12 | 18 |
| 2560 | 56 | 2 | 27 | 43 | 11 | 18 |
| 2812 | 56 | 3 | 27 | 46 | 11 | 20 |
| 3089 | 63 | 4 | 29 | 48 | 7 | 18 |
| 3393 | 66 | 4 | 30 | 53 | 11 | 18 |
| 3728 | 69 | 3 | 30 | 59 | 13 | 19 |
| 4095 | 70 | 3 | 30 | 64 | 9 | 19 |
| 4498 | 72 | 3 | 30 | 62 | 10 | 21 |
| 4942 | 79 | 4 | 31 | 67 | 7 | 20 |
| 5429 | 86 | 4 | 32 | 71 | 7 | 21 |
| 5964 | 90 | 4 | 33 | 78 | 9 | 21 |
| 6551 | 97 | 3 | 33 | 88 | 13 | 21 |
| 7197 | 103 | 2 | 33 | 101 | 14 | 23 |
| 7906 | 108 | 2 | 34 | 117 | 15 | 22 |
| 8685 | 116 | 2 | 35 | 119 | 16 | 23 |
| 9541 | 129 | 2 | 36 | 128 | 15 | 22 |
| 10481 | 141 | 3 | 37 | 129 | 14 | 22 |
| 11514 | 161 | 2 | 37 | 144 | 14 | 24 |
| 12649 | 166 | 2 | 37 | 160 | 14 | 24 |
| 13895 | 185 | 2 | 37 | 164 | 13 | 26 |
| 15264 | 201 | 3 | 39 | 177 | 15 | 24 |
| 16768 | 222 | 4 | 40 | 193 | 15 | 24 |
| 18421 | 234 | 3 | 40 | 200 | 18 | 25 |
| 20236 | 250 | 2 | 40 | 222 | 17 | 25 |
| 22230 | 265 | 2 | 40 | 230 | 14 | 28 |
| 24421 | 279 | 3 | 41 | 259 | 12 | 27 |
| 26827 | 318 | 4 | 42 | 267 | 11 | 28 |
| 29471 | 337 | 4 | 43 | 301 | 13 | 28 |
| 32375 | 352 | 5 | 44 | 338 | 12 | 27 |
| 35565 | 376 | 4 | 44 | 367 | 13 | 29 |
| 39069 | 405 | 4 | 45 | 402 | 14 | 28 |
| 42919 | 452 | 5 | 46 | 428 | 14 | 29 |
| 47149 | 497 | 5 | 47 | 457 | 17 | 29 |
| 51795 | 532 | 4 | 48 | 501 | 19 | 28 |
| 56899 | 582 | 3 | 48 | 526 | 20 | 31 |
| 62506 | 642 | 2 | 48 | 584 | 17 | 31 |
| 68665 | 700 | 2 | 48 | 622 | 15 | 33 |
| 75431 | 745 | 3 | 49 | 679 | 13 | 32 |
| 82864 | 809 | 4 | 51 | 725 | 10 | 30 |
| 91030 | 875 | 3 | 52 | 781 | 10 | 32 |
| 100000 | 936 | 2 | 52 | 834 | 18 | 32 |

q = 7, same Q (P_1 / B_1 / E_1 / P_2 / B_2 / E_2): 26/9/46/21/22/32; 27/9/48/19/24/33;
31/11/50/25/20/34; 28/7/51/29/21/34; 32/9/52/35/20/35; 34/11/54/40/22/35; 38/10/55/37/24/38;
42/11/57/39/26/38; 47/11/58/37/29/40; 51/10/59/41/27/41; 56/12/61/43/26/41; 56/14/62/46/25/43;
63/12/65/48/23/43; 66/12/67/53/27/44; 69/9/68/59/26/45; 70/10/69/64/24/47; 72/11/71/62/26/48;
79/10/73/67/28/49; 86/8/74/71/32/51; 90/10/77/78/33/52; 97/10/78/88/31/54; 103/12/80/101/31/54;
108/11/83/117/29/54; 116/10/85/119/31/57; 129/9/87/128/32/57; 141/11/89/129/32/59;
161/10/91/144/33/61; 166/10/92/160/34/62; 185/9/95/164/34/63; 201/8/97/177/37/65;
222/9/100/193/37/66; 234/10/102/200/42/66; 250/9/103/222/42/68; 265/8/105/230/38/71;
279/8/108/259/42/72; 318/10/110/267/39/74; 337/11/112/301/37/77; 352/12/115/338/37/76;
376/11/118/367/38/77; 405/10/121/402/37/78; 452/10/124/428/33/81; 497/10/126/457/36/83;
532/9/128/501/33/84; 582/7/131/526/35/86; 642/7/133/584/37/88; 700/7/135/622/43/91;
745/8/138/679/41/92; 809/8/143/725/41/91; 875/10/146/781/44/93; 936/10/148/834/51/94.
q = 11 at every seventh Q of the table (P_1/B_1, P_2/B_2): 26/17, 21/36; 42/21, 39/38; 69/15,
59/45; 103/26, 101/57; 185/21, 164/62; 318/24, 267/69; 532/31, 501/63; 936/27, 834/101.
q = 13: 26/32, 21/48; 42/31, 39/58; 69/26, 59/70; 103/43, 101/93; 185/40, 164/102; 318/48,
267/114; 532/55, 501/115; 936/56, 834/178.

Small Q, q = 5, as Q: P_1/B_1, P_2/B_2 for Q = 1..30: 1: 0/1, 1/0; 2: 1/1, 1/1; 3: 1/2, 0/3;
4: 1/3, 1/3; 5: 0/5, 1/4; 6: 1/5, 1/5; 7: 1/4, 1/4; 8: 1/5, 1/5; 9: 2/5, 0/6; 10: 2/5, 1/6;
11: 1/4, 1/6; 12: 1/6, 1/6; 13: 1/5, 1/6; 14: 1/6, 2/5; 15: 2/6, 1/6; 16: 2/6, 1/7; 17: 1/5,
1/6; 18: 1/5, 1/5; 19: 1/4, 1/4; 20: 1/4, 2/6; 21: 2/4, 1/7; 22: 2/5, 1/6; 23: 2/5, 1/4;
24: 2/6, 2/4; 25: 2/5, 2/5; 26: 2/5, 2/5; 27: 2/4, 2/8; 28: 2/4, 2/8; 29: 1/4, 2/7; 30: 2/4, 1/6
(Q = 30 reproduces the scratch lane's turn-1 count 4 and turn-2 count 6).

What this measures: the root at one octave, directly. In (Q, 2Q] the charges are the twins and
the ember charges and nothing else; the ember charges are at most 2 E_1 = O((log Q)^{pi(q) - 1});
the twins are 2 C_2 Q / log^2 Q in count and never absent from Q = 6 on. The scan is a record,
not an argument: P_1 = 0 at Q = 5 is the one place (with Q = 1) where the pure charge is absent
from a first turn, and no structural law forbids it anywhere.

### 6. The valve count law

The valves open at turn m are I(m), the admissible pairs with max(s, s') <= m; A(m) = |I(m)|
exactly (INVENTORY, proved), and the ledger confirms that no family is ever present before its
turn (0 extra families in 600 turns). A(m) for m = 1..60:

- q = 5: 1, 1, 3, 5, 9, 11, 11, 15, 19, 23, 23, 27, 27, 27, 29, 35, 35, 41, 41, 47, 47, 47, 47,
  51, 57, 57, 63, 63, 63, 69, 69, 79, 79, 79, 79, 83, 83, 83, 83, 89, 89, 89, 89, 89, 91, 91, 91,
  95, 95, 111, 111, 111, 111, 123, 123, 123, 123, 123, 123, 125.
- q = 7: 1, 1, 3, 5, 9, 11, 17, 21, 27, 31, 31, 35, 35, 41, 45, 53, 53, 59, 59, 67, 71, 71, 71, 77,
  87, 87, 95, 103, 103, 111, 111, 123, 123, 123, 131, 137, 137, 137, 137, 145, 145, 157, 157, 157,
  161, 161, 161, 167, 183, 201, 201, 201, 201, 215, 215, 229, 229, 229, 229, 233.
- q = 11: 1, 1, 3, 5, 9, 11, 17, 21, 27, 31, 41, 45, 45, 51, 57, 65, 65, 71, 71, 79, 85, 95, 95,
  103, 115, 115, 125, 135, 135, 143, 143, 157, 165, 165, 177, 185, 185, 185, 185, 195, 195, 207,
  207, 221, 227, 227, 227, 235, 255, 275, 275, 275, 275, 291, 305, 321, 321, 321, 321, 327.

Closed form as a recursion (exact): A(m) = A(m - 1) unless m is q-smooth, in which case
A(m) = A(m - 1) + 2 #{s' q-smooth, s' < m : (m, s') admissible} + [m = 1]. (Only (1, 1) has
s = s': gcd(s, s') | 2 with s = s' forces s <= 2, and (2, 2) is not admissible.) The steps: 2 at
m = 3 (the pair (1, 3) and its mirror), 2 at 4, 4 at 5, 2 at 6, 6 at 7 (q >= 7), 4 at 8, 4 or 6 at
9, 4 at 10, 10 at 11 (q = 11), 4 at 12, 0 at every non-smooth m. The growth is that of pairs of
smooth numbers: A(m) is of order (number of q-smooth s <= m)^2 times the admissible fraction,
i.e. (log m)^{2 pi(q)} up to constants; at m = 10^4 the counts are 1,715 / 6,779 / 18,575 for
q = 5 / 7 / 11 (the scratch lane's inventory totals, reproduced by valve_count.py).

The fraction they burn. B_m / T_m per turn, (5, 10^4): 0.021, 0.088, 0.622, 0.762, 0.819,
0.850, 0.885, 0.897, 0.903 at m = 1, 2, 3, 5, 9, 15, 30, 45, 60; (7, 10^4): 0.074, 0.194, 0.640,
0.770, 0.844, 0.873, 0.907, 0.922, 0.924; (11, 10^4): 0.165, 0.313, 0.657, 0.779, 0.847, 0.885,
0.915, 0.930, 0.933; (5, 3 x 10^4): 0.014, 0.038, 0.578, 0.760, 0.834, 0.867, 0.887, 0.892,
0.896; (5, 10^5): 0.002, 0.021, 0.596, 0.752, 0.828, 0.853, 0.883, 0.898, 0.900 (every turn in
the appendix). The mechanism, in two parts:

(a) THE VALVES OPEN IN AIR ORDER AND EACH ADDS ITS DENSITY. Relative to the pure charge, family
(s, s') has the density weight w(s, s') = (2^{[s even]} / (s s')) prod over odd p | s s' of
(p - 1)/(p - 2) (from the yield law: the pair of linear forms P, (sP + 2)/s' restricted to the
class of P that makes the second an integer; for an even pair the class also fixes the partner's
parity, which doubles its prime chance: the factor 2^{[s even]} that my pre-registration missed).
So T_m / P_m = Sigma^log(m) := sum over (s, s') in I(m) of w(s, s') L_m(s, s'), with L_m the
scale factor of section 2 (the family's fuel sits at mQ/s where primes are denser). Measured
against this, aggregating turns 30..60: T/P = 9.914 vs 9.923 (5, 10^4); 12.684 vs 12.660
(7, 10^4); 14.245 vs 14.165 (11, 10^4); 9.515 vs 9.731 (5, 3 x 10^4); 12.121 vs 12.388
(7, 3 x 10^4); 13.573 vs 13.848 (11, 3 x 10^4); 9.519 vs 9.555 (5, 10^5): ratios 0.999, 1.002,
1.006, 0.978, 0.978, 0.980, 0.996; at Q = 10^3: 1.007, 1.020, 1.039. Per turn the ratio scatters
at square-root size (min/max 0.80/1.21 at Q = 10^4, 0.92/1.07 at 10^5, m >= 3). The same for the
prime-led share: P_m / A_m = 1 / Sigma_1^log(m), Sigma_1^log = the sum over the families (1, s')
alone; aggregated over turns 30..60 the measured / predicted ratio is 0.999, 1.003, 1.004, 1.001,
1.013, 1.013, 1.012, 1.003, 0.989, 0.981 over the ten ledgers. At m = 1, 2 the ratio is 1 up to
the embers (0.986, 0.992 at (5, 10^4); 0.999, 1.000 at (5, 10^5); 0.919, 0.969 at (11, 10^4)),
then 0.566, 0.473, 0.460, 0.413, 0.380, 0.369 at m = 3, 5, 9, 15, 30, 60 for (5, 10^4) as (1, 3),
(1, 5), (1, 9), (1, 15), ... open, against the placement law's limit 0.375.

(b) THE SUM CONVERGES, SO THE PURE SHARE DOES NOT VANISH WITH THE TURN. Without the scale
factor, the sum over ALL admissible pairs is an Euler product: the odd coprime pairs give
prod over odd p <= q of (1 + 2 sum_k (p - 1)/((p - 2) p^k)) = prod p/(p - 2); the even pairs
(exactly one member 2 mod 4, the other 0 mod 4, odd parts coprime) give the same again (the
2-adic weights 2 x 2^{-(j + 1)} summed over j >= 2 and over the two orders give 1). So

  Sigma_inf(q) = 2 prod_{3 <= p <= q} p/(p - 2) = 10 (q = 5), 14 (q = 7), 154/9 = 17.11 (q = 11),

and the partial sums Sigma(m) reach 8.36, 10.61, 11.87 at m = 60 and 9.97, 13.90, 16.91 at
m = 10^4 (the tail is the sum of 1/(s s') over smooth pairs with max > m, of order
(log m)^{pi(q) - 1} / m). The limit of the pure charge's share is therefore

  P_m / T_m -> 1 / Sigma_inf = (1/2) prod_{3 <= p <= q} (1 - 2/p),

WHICH IS EXACTLY THE ENGINE'S OWN DENSITY OF OPEN PAIRS (n and n + 2 both coprime to q#): once
every valve is open, the fraction of the manifold's open pairs that the engine leaves open is the
fraction of all pairs it would leave open. In the same way P_m / A_m -> prod (p - 2)/(p - 1), the
placement law's density (the fraction of primes P with P + 2 coprime to q#). B_m / T_m rises
with m as the valves open (at the smooth m only; between them it fluctuates), and tends to
1 - (1/2) prod_{3 <= p <= q} (1 - 2/p) = 0.900, 0.929, 0.942 for q = 5, 7, 11, never to 1. It tends
to 1 in q, not in m: 1/Sigma_inf ~ c/(log q)^2 by Mertens. Inside the quiet zone (m <= Q) the
scale factor L keeps T_m / P_m above the partial sum, so the measured B/T at m = 60 (0.903,
0.924, 0.933 at Q = 10^4; 0.896, 0.920, 0.930 at 3 x 10^4; 0.900 at (5, 10^5)) sits at or just
above the m -> infinity limit while only 125 / 233 / 327 of the valves are open; the limit is
approached from above as the log factors fade, i.e. only for m far beyond Q, outside the zone.
Prior art for (b): the singular series of Hardy-Littlewood factorises over primes, so the
limit share equals the engine's local density; the difference from the Mertens-product density of
the manifold's open pairs (a factor e^{-2 gamma}) is Buchstab's, the usual gap between a sieve
density and the true count. Not new; (a) as a per-turn law of the valves, with the scale factor
identified as the whole of the discrepancy, is the measurement this branch adds.

## Laws

- V1 (the turn ledger identity; PROVED, 0 exceptions in 600 turns). For every q, Q and every
  turn m: P_m = T_m - sum over (s, s') in I(m) \ (1, 1) of N_m(s, s') - B_m^ember, with
  0 <= B_m^ember <= 2 E_m; for m <= 2 the sum is empty. Hypothesis: the fuel is above Q and
  coprime to q# (nothing else). Proof: a fuelled burnt pair (sP, s'P') in turn m has sP > sQ so
  s <= m, and s'P' <= (m + 1)Q + 2 with P' > Q gives s' <= m for m >= 2 and, for m = 1, s' = 2 only
  with n = 2Q an ember; the ember bound is that an ember carries at most two pairs.
- V2 (the structural laws are blind to the fuel's primality; PROVED by construction). Every
  proved valve law holds for an arbitrary set F of odd integers above Q coprime to q# in place of
  the exhaust; the set F = {n = 1 (mod 3)} satisfies all of them with P_m = 0 and B_m = T_m > 0
  in every turn (measured: (5, 10^3) m = 1..12 and (7, 10^4) m = 1..6, 0 violations). Hence no
  inequality B_m <= c T_m with c < 1 and no lower bound on P_m follows from the imprint, port,
  inventory, onset and ember laws; the separation of P_m from B_m needs the count of primes in
  intervals and reduced residue classes. This is the ledger's statement of face A. ROOT-marker.
- V3 (the valve count; PROVED from the inventory, presence measured). The fuelled families that
  can be present in turn m are exactly I(m), |I(m)| = A(m) by the recursion above (table to
  m = 60 for q = 5, 7, 11); none is present earlier (0 exceptions in 600 turns); all are present
  from their onset on up to m = 8 (Q = 10^3), 19 / 19 / 15 (Q = 10^4, q = 5 / 7 / 11),
  26 / 25 / 22 (3 x 10^4), 49 (5, 10^5), after which sparse valves miss turns (a density, not a
  law: the first misses (16, 18), (25, 27), (32, 50) have expected count below 1 per turn).
- V4 (the yield per turn; MEASURED). T_m / P_m = sum over I(m) of w(s, s') L_m(s, s') and
  P_m / A_m = 1 / (sum over (1, s') in I(m) of w L), with w = (2^{[s even]}/(s s')) prod over odd
  p | s s' of (p - 1)/(p - 2) and L_m the scale factor; aggregated over turns 30..60, within 2.2%
  (T/P) and 1.3% (P/A) at every run with Q >= 10^4, within 4% at Q = 10^3; per turn, square-root
  scatter. Limits: sum of w over all admissible pairs = 2 prod_{3 <= p <= q} p/(p - 2), so the
  pure share of the charges tends to the engine's pair-opening density (1/2) prod (1 - 2/p) and
  the pure share of the prime-led charges to the placement density prod (p - 2)/(p - 1); B/T
  never tends to 1 in m. Prior art: Hardy-Littlewood / Bateman-Horn for each family; the
  factorised singular series for the limit.
- V5 (the first two turns; EXACT scan, every Q <= 10^5, q = 5, 7, 11, 13). P_1 = 0 iff Q in
  {1, 5}; P_2 = 0 iff Q in {3, 9}. Min P_1 on [10^3, 10^5] = 25 at Q = 1,031 (largest twin gap
  touching the turn 120, from 1,487); min P_2 = 17 at Q = 1,071 (gap 168, from 2,381). B_1 > P_1
  last at Q = 74, 400, 726, 1,355 and B_2 > P_2 last at Q = 555, 1,150, 2,963, 6,562 for
  q = 5, 7, 11, 13. Max B_1 on [10^3, 10^5] = 6, 14, 31, 60; max B_2 = 21, 51, 105, 182.
  Measurement of the root at one octave; no structural content beyond the ember law.

## What is new

- V2, the counterfactual fuel: a one-line model in which every proved valve law holds and the
  pure charge is empty. It turns "we found no structural inequality" into "there is none", and
  it names the single property of the exhaust that any proof must use (its equidistribution in
  reduced classes and its count in intervals). No prior art located for this framing inside the
  project; outside it, it is the standard observation that sieve axioms are satisfied by sets
  with no twins (the parity phenomenon's usual witness is the set of integers with an odd number
  of prime factors; here the witness is the class 1 mod 3, simpler and exact per turn).
- The turn ledger tables themselves (600 turns, every number), the exact per-turn valve list, and
  the exact scan of the first two turns to 10^5 with the crossovers where the embers stop
  outnumbering the twins.
- The corrected weight of the even families (2/(s s')) and the per-turn yield law with the scale
  factor as the whole discrepancy (V4): a measurement of the valves' densities turn by turn at
  the 1-2% level, and the closed-form limit share = the engine's own density (elementary once
  seen; the pre-registration got it wrong by the 2-adic factor, and the ledger caught it).

## Verdict

The turn ledger is exact in its bookkeeping and has no inequality of its own. Per turn, the
burnt charges are the sum over the A(m) - 1 open valves of their counts plus at most 2 E_m ember
charges, and the pure charge is the remainder (V1, 0 exceptions in 600 turns). Every candidate
relation between T_m and B_m other than B_m <= T_m fails on the tables or is the conjecture
(section 3), and V2 proves that none can follow from the structural laws: they are satisfied by
a fuel set with no pure charge at all. The wall reappears at the turn ledger as face A alone:
the only thing that separates the primes above Q from the counterfactual fuel is their count and
distribution in residue classes, i.e. the dimension-2 sieve on the charges, and the alternating
expression for P_m telescopes to an identity the moment one tries to use it. The sharpest
structural statement that is not the conjecture is the named, placed and ember-bounded burnt
side of section 4; the pure charge's share of the turn is a density (V4) that starts at 1 in the
first two turns, falls as the valves open in air order, and tends to the engine's own pair-opening
density, never to zero. The first two turns, scanned exactly to 10^5, are never empty of twins
from Q = 6 (turn 1) and Q = 10 (turn 2) on, with minima 25 and 17 at Q = 1,031 and 1,071.

Status for the tree: V1, V3 FACT (exact bookkeeping, not routes); V2 ROOT-marker (PROVED: the
structural laws cannot reach the root; the gap is the count); V4 a measurement with the
mechanism visible (prior art Hardy-Littlewood); V5 FACT (measurement). Node R4.c.i: the question
"does an exact relation between T_m and B_m force P_m > 0" is answered NO by proof, and the
answer locates the wall exactly.

## Dead ends

- B_m <= c T_m for any c < 1 as a structural law: refuted by V2 (the fuel F = 1 mod 3), and on
  the tables C5 fails at (5, 10^4, m = 29) with B/T = 0.9001 against the limit 0.900.
- B_m / T_m monotone in m: 20-29 decreases per ledger; first at (5, 10^3, m = 6).
- The placement density as a per-turn lower bound on P_m / A_m: fails at (5, 10^4, m = 16) with
  0.360 against 0.375, by the scale factor plus fluctuation.
- "Every admissible valve is present at every turn from its onset": fails from m = 20 at
  Q = 10^4 ((16, 18)), a density.
- The pre-registered Euler product (3/2) prod p/(p - 2): wrong by the even families' 2-adic
  factor; the data refused it by a uniform 4/3 and the derivation found the error. Kept as the
  one place the pre-registration was wrong.
- The alternating expression for P_m as a route: it is Legendre on the charges, telescopes to an
  identity, and any use of it is the dimension-2 sieve (face A).

## Open items of the part, sorted

- Closed here: the per-turn identity (V1); the valve list per turn (V3); whether a structural
  inequality exists (V2: no); the first-two-turns record to 10^5 (V5); the limit of the pure
  share (V4, closed form).
- Measurement with no structural content: the yield law per turn and its scale factor (V4);
  the crossovers Q where embers stop outnumbering twins; the extremes of B_1, B_2.
- Root question in disguise: P_m > 0 per turn (C1); any B_m <= c T_m with c < 1; the presence
  of family (1, 1) in a turn as opposed to the burnt families' presence.
- Genuinely open on the part alone: none of structural kind. The one statement that would move
  the ledger is a level of distribution for the charges (M_m(d1, d2) against its expected value)
  strong enough at dimension 2, which is face A restated; the ledger adds nothing to it and
  says nothing else is needed. An attack, if any, would be on the counterfactual: characterise
  the fuel sets F (odd, above Q, coprime to q#) for which P_m > 0 is forced in every turn, i.e.
  which property beyond equidistribution mod small moduli separates the primes from F. The
  primes are equidistributed in reduced classes to every modulus up to Q^{1/2 - eps} (the level
  of distribution); F = 1 mod 3 is not; a set equidistributed to level 1/2 with no twins would be
  the parity counterexample, which is why the sieve stops there.

## Appendix: the ledgers, every turn

Columns: m, T, B, B_f (fuelled-both), B_e (ember-carrying), P, E (embers), pi (primes), A (primes
with P + 2 open), fam/adm (fuelled families present including (1, 1) / admissible with
max <= m), B/T, P/A. Under each table, the burnt fuelled families with their counts for m <= 12.

#### Ledger q = 5, Q = 1000 (turn m = (mQ, (m + 1)Q]; T total charges, B burnt = B_f fuelled-both + B_e ember-carrying, P pure = twins, E embers, pi primes, A primes with P + 2 open, fam/adm = fuelled families present / admissible with max <= m)

| m | T | B | B_f | B_e | P | E | pi | A | fam/adm | B/T | P/A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 31 | 5 | 0 | 5 | 26 | 22 | 135 | 29 | 1/1 | 0.161 | 0.897 |
| 2 | 31 | 10 | 0 | 10 | 21 | 15 | 127 | 21 | 1/1 | 0.323 | 1.000 |
| 3 | 62 | 41 | 34 | 7 | 21 | 12 | 120 | 41 | 3/3 | 0.661 | 0.512 |
| 4 | 68 | 45 | 40 | 5 | 23 | 9 | 119 | 39 | 5/5 | 0.662 | 0.590 |
| 5 | 82 | 65 | 57 | 8 | 17 | 7 | 114 | 33 | 9/9 | 0.793 | 0.515 |
| 6 | 84 | 65 | 62 | 3 | 19 | 8 | 117 | 39 | 11/11 | 0.774 | 0.487 |
| 7 | 69 | 56 | 53 | 3 | 13 | 6 | 107 | 30 | 11/11 | 0.812 | 0.433 |
| 8 | 93 | 78 | 75 | 3 | 15 | 5 | 110 | 31 | 15/15 | 0.839 | 0.484 |
| 9 | 99 | 84 | 78 | 6 | 15 | 5 | 112 | 37 | 17/19 | 0.848 | 0.405 |
| 10 | 93 | 77 | 74 | 3 | 16 | 5 | 106 | 36 | 21/23 | 0.828 | 0.444 |
| 11 | 95 | 81 | 81 | 0 | 14 | 4 | 103 | 37 | 21/23 | 0.853 | 0.378 |
| 12 | 111 | 100 | 95 | 5 | 11 | 5 | 109 | 32 | 23/27 | 0.901 | 0.344 |
| 13 | 107 | 92 | 91 | 1 | 15 | 3 | 105 | 38 | 25/27 | 0.860 | 0.395 |
| 14 | 92 | 81 | 80 | 1 | 11 | 3 | 102 | 35 | 24/27 | 0.880 | 0.314 |
| 15 | 110 | 98 | 96 | 2 | 12 | 4 | 108 | 37 | 27/29 | 0.891 | 0.324 |
| 16 | 106 | 93 | 90 | 3 | 13 | 3 | 98 | 32 | 31/35 | 0.877 | 0.406 |
| 17 | 110 | 92 | 88 | 4 | 18 | 3 | 104 | 42 | 29/35 | 0.836 | 0.429 |
| 18 | 113 | 101 | 100 | 1 | 12 | 3 | 94 | 33 | 33/41 | 0.894 | 0.364 |
| 19 | 123 | 108 | 104 | 4 | 15 | 4 | 104 | 38 | 34/41 | 0.878 | 0.395 |
| 20 | 111 | 96 | 93 | 3 | 15 | 3 | 98 | 37 | 34/47 | 0.865 | 0.405 |
| 21 | 117 | 102 | 100 | 2 | 15 | 2 | 104 | 40 | 39/47 | 0.872 | 0.375 |
| 22 | 112 | 96 | 95 | 1 | 16 | 1 | 100 | 35 | 36/47 | 0.857 | 0.457 |
| 23 | 114 | 100 | 98 | 2 | 14 | 3 | 104 | 38 | 41/47 | 0.877 | 0.368 |
| 24 | 117 | 111 | 109 | 2 | 6 | 3 | 94 | 27 | 37/51 | 0.949 | 0.222 |
| 25 | 132 | 120 | 118 | 2 | 12 | 2 | 98 | 37 | 42/57 | 0.909 | 0.324 |
| 26 | 120 | 109 | 107 | 2 | 11 | 2 | 101 | 33 | 43/57 | 0.908 | 0.333 |
| 27 | 107 | 92 | 92 | 0 | 15 | 1 | 94 | 29 | 40/63 | 0.860 | 0.517 |
| 28 | 111 | 99 | 97 | 2 | 12 | 2 | 98 | 34 | 41/63 | 0.892 | 0.353 |
| 29 | 112 | 103 | 103 | 0 | 9 | 2 | 92 | 28 | 45/63 | 0.920 | 0.321 |
| 30 | 124 | 113 | 111 | 2 | 11 | 2 | 95 | 31 | 47/69 | 0.911 | 0.355 |
| 31 | 122 | 109 | 106 | 3 | 13 | 3 | 92 | 34 | 48/69 | 0.893 | 0.382 |
| 32 | 125 | 106 | 105 | 1 | 19 | 3 | 106 | 38 | 50/79 | 0.848 | 0.500 |
| 33 | 124 | 111 | 110 | 1 | 13 | 1 | 100 | 32 | 49/79 | 0.895 | 0.406 |
| 34 | 124 | 108 | 106 | 2 | 16 | 2 | 94 | 35 | 47/79 | 0.871 | 0.457 |
| 35 | 112 | 101 | 101 | 0 | 11 | 1 | 92 | 33 | 47/79 | 0.902 | 0.333 |
| 36 | 124 | 115 | 114 | 1 | 9 | 2 | 99 | 28 | 50/83 | 0.927 | 0.321 |
| 37 | 128 | 116 | 115 | 1 | 12 | 1 | 94 | 37 | 55/83 | 0.906 | 0.324 |
| 38 | 116 | 105 | 104 | 1 | 11 | 2 | 90 | 30 | 51/83 | 0.905 | 0.367 |
| 39 | 113 | 104 | 104 | 0 | 9 | 2 | 96 | 27 | 47/83 | 0.920 | 0.333 |
| 40 | 127 | 119 | 116 | 3 | 8 | 2 | 88 | 35 | 50/89 | 0.937 | 0.229 |
| 41 | 120 | 108 | 108 | 0 | 12 | 1 | 101 | 37 | 49/89 | 0.900 | 0.324 |
| 42 | 114 | 102 | 102 | 0 | 12 | 0 | 102 | 33 | 50/89 | 0.895 | 0.364 |
| 43 | 126 | 115 | 112 | 3 | 11 | 2 | 85 | 35 | 54/89 | 0.913 | 0.314 |
| 44 | 114 | 103 | 102 | 1 | 11 | 1 | 96 | 26 | 44/89 | 0.904 | 0.423 |
| 45 | 124 | 117 | 117 | 0 | 7 | 0 | 86 | 27 | 58/91 | 0.944 | 0.259 |
| 46 | 117 | 105 | 101 | 4 | 12 | 3 | 90 | 33 | 50/91 | 0.897 | 0.364 |
| 47 | 118 | 107 | 106 | 1 | 11 | 1 | 95 | 34 | 55/91 | 0.907 | 0.324 |
| 48 | 122 | 109 | 109 | 0 | 13 | 1 | 89 | 32 | 48/95 | 0.893 | 0.406 |
| 49 | 140 | 123 | 123 | 0 | 17 | 2 | 98 | 41 | 57/95 | 0.879 | 0.415 |
| 50 | 118 | 109 | 108 | 1 | 9 | 1 | 89 | 33 | 51/111 | 0.924 | 0.273 |
| 51 | 115 | 101 | 99 | 2 | 14 | 2 | 97 | 37 | 55/111 | 0.878 | 0.378 |
| 52 | 128 | 120 | 120 | 0 | 8 | 1 | 89 | 29 | 60/111 | 0.938 | 0.276 |
| 53 | 112 | 101 | 101 | 0 | 11 | 1 | 92 | 28 | 49/111 | 0.902 | 0.393 |
| 54 | 106 | 98 | 97 | 1 | 8 | 1 | 90 | 31 | 50/123 | 0.925 | 0.258 |
| 55 | 114 | 103 | 102 | 1 | 11 | 1 | 93 | 28 | 56/123 | 0.904 | 0.393 |
| 56 | 120 | 106 | 106 | 0 | 14 | 1 | 99 | 32 | 56/123 | 0.883 | 0.438 |
| 57 | 130 | 121 | 121 | 0 | 9 | 1 | 91 | 28 | 65/123 | 0.931 | 0.321 |
| 58 | 119 | 108 | 108 | 0 | 11 | 1 | 90 | 33 | 56/123 | 0.908 | 0.333 |
| 59 | 124 | 113 | 112 | 1 | 11 | 2 | 94 | 31 | 56/123 | 0.911 | 0.355 |
| 60 | 121 | 111 | 109 | 2 | 10 | 1 | 88 | 25 | 57/125 | 0.917 | 0.400 |

Burnt fuelled families per turn, m <= 12 (family: count): m=1: {}; m=2: {}; m=3: {(1,3) 18, (3,1) 16}; m=4: {(1,3) 16, (2,4) 4, (3,1) 13, (4,2) 7}; m=5: {(1,3) 10, (1,5) 5, (2,4) 5, (3,1) 17, (3,5) 5, (4,2) 6, (5,1) 7, (5,3) 2}; m=6: {(1,3) 14, (1,5) 5, (2,4) 4, (3,1) 11, (3,5) 3, (4,2) 5, (4,6) 4, (5,1) 8, (5,3) 6, (6,4) 2}; m=7: {(1,3) 11, (1,5) 6, (2,4) 2, (3,1) 10, (3,5) 4, (4,2) 5, (4,6) 5, (5,1) 4, (5,3) 3, (6,4) 3}; m=8: {(1,3) 12, (1,5) 4, (2,4) 7, (2,8) 3, (3,1) 14, (3,5) 2, (4,2) 6, (4,6) 5, (5,1) 6, (5,3) 3, (6,4) 3, (6,8) 2, (8,2) 6, (8,6) 2}; m=9: {(1,3) 11, (1,5) 2, (1,9) 9, (2,4) 3, (2,8) 3, (3,1) 9, (3,5) 5, (4,2) 6, (5,1) 5, (5,3) 6, (5,9) 2, (6,4) 7, (6,8) 1, (8,2) 1, (8,6) 1, (9,1) 7}; m=10: {(1,3) 12, (1,5) 5, (1,9) 3, (2,4) 6, (2,8) 3, (3,1) 11, (3,5) 3, (4,2) 5, (4,6) 3, (4,10) 1, (5,1) 4, (5,3) 3, (6,4) 2, (6,8) 1, (8,2) 2, (8,6) 3, (8,10) 1, (9,1) 3, (9,5) 2, (10,8) 1}; m=11: {(1,3) 11, (1,5) 4, (1,9) 8, (2,4) 2, (2,8) 1, (3,1) 10, (3,5) 5, (4,2) 6, (4,6) 3, (4,10) 1, (5,1) 5, (5,3) 6, (5,9) 1, (6,4) 7, (8,2) 3, (8,6) 2, (9,1) 3, (9,5) 1, (10,4) 1, (10,8) 1}; m=12: {(1,3) 11, (1,5) 7, (1,9) 3, (2,4) 8, (2,8) 5, (2,12) 3, (3,1) 14, (3,5) 4, (4,2) 1, (4,6) 4, (5,1) 4, (5,3) 5, (5,9) 1, (6,4) 2, (6,8) 3, (8,2) 5, (8,6) 2, (8,10) 1, (9,1) 4, (10,12) 1, (12,2) 4, (12,10) 3};

#### Ledger q = 7, Q = 1000 (turn m = (mQ, (m + 1)Q]; T total charges, B burnt = B_f fuelled-both + B_e ember-carrying, P pure = twins, E embers, pi primes, A primes with P + 2 open, fam/adm = fuelled families present / admissible with max <= m)

| m | T | B | B_f | B_e | P | E | pi | A | fam/adm | B/T | P/A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 35 | 9 | 0 | 9 | 26 | 46 | 135 | 32 | 1/1 | 0.257 | 0.812 |
| 2 | 43 | 22 | 0 | 22 | 21 | 32 | 127 | 24 | 1/1 | 0.512 | 0.875 |
| 3 | 69 | 48 | 34 | 14 | 21 | 26 | 120 | 43 | 3/3 | 0.696 | 0.488 |
| 4 | 80 | 57 | 40 | 17 | 23 | 20 | 119 | 41 | 5/5 | 0.713 | 0.561 |
| 5 | 91 | 74 | 57 | 17 | 17 | 17 | 114 | 34 | 9/9 | 0.813 | 0.500 |
| 6 | 90 | 71 | 62 | 9 | 19 | 18 | 117 | 39 | 11/11 | 0.789 | 0.487 |
| 7 | 97 | 84 | 73 | 11 | 13 | 14 | 107 | 37 | 17/17 | 0.866 | 0.351 |
| 8 | 114 | 99 | 90 | 9 | 15 | 13 | 110 | 35 | 21/21 | 0.868 | 0.429 |
| 9 | 117 | 102 | 92 | 10 | 15 | 11 | 112 | 38 | 25/27 | 0.872 | 0.395 |
| 10 | 111 | 95 | 90 | 5 | 16 | 12 | 106 | 39 | 27/31 | 0.856 | 0.410 |
| 11 | 118 | 104 | 97 | 7 | 14 | 9 | 103 | 41 | 29/31 | 0.881 | 0.341 |
| 12 | 133 | 122 | 108 | 14 | 11 | 11 | 109 | 36 | 28/35 | 0.917 | 0.306 |
| 13 | 130 | 115 | 106 | 9 | 15 | 9 | 105 | 41 | 32/35 | 0.885 | 0.366 |
| 14 | 120 | 109 | 101 | 8 | 11 | 8 | 102 | 39 | 35/41 | 0.908 | 0.282 |
| 15 | 134 | 122 | 114 | 8 | 12 | 10 | 108 | 40 | 38/45 | 0.910 | 0.300 |
| 16 | 132 | 119 | 111 | 8 | 13 | 7 | 98 | 34 | 42/53 | 0.902 | 0.382 |
| 17 | 128 | 110 | 100 | 10 | 18 | 8 | 104 | 43 | 39/53 | 0.859 | 0.419 |
| 18 | 132 | 120 | 119 | 1 | 12 | 8 | 94 | 35 | 44/59 | 0.909 | 0.343 |
| 19 | 146 | 131 | 126 | 5 | 15 | 7 | 104 | 42 | 45/59 | 0.897 | 0.357 |
| 20 | 141 | 126 | 118 | 8 | 15 | 7 | 98 | 37 | 49/67 | 0.894 | 0.405 |
| 21 | 147 | 132 | 126 | 6 | 15 | 7 | 104 | 46 | 55/71 | 0.898 | 0.326 |
| 22 | 138 | 122 | 119 | 3 | 16 | 4 | 100 | 41 | 51/71 | 0.884 | 0.390 |
| 23 | 145 | 131 | 126 | 5 | 14 | 6 | 104 | 43 | 57/71 | 0.903 | 0.326 |
| 24 | 147 | 141 | 134 | 7 | 6 | 7 | 94 | 34 | 49/77 | 0.959 | 0.176 |
| 25 | 153 | 141 | 138 | 3 | 12 | 6 | 98 | 42 | 55/87 | 0.922 | 0.286 |
| 26 | 144 | 133 | 128 | 5 | 11 | 5 | 101 | 38 | 57/87 | 0.924 | 0.289 |
| 27 | 140 | 125 | 122 | 3 | 15 | 5 | 94 | 36 | 58/95 | 0.893 | 0.417 |
| 28 | 141 | 129 | 126 | 3 | 12 | 6 | 98 | 38 | 58/103 | 0.915 | 0.316 |
| 29 | 143 | 134 | 131 | 3 | 9 | 3 | 92 | 31 | 66/103 | 0.937 | 0.290 |
| 30 | 159 | 148 | 142 | 6 | 11 | 6 | 95 | 36 | 67/111 | 0.931 | 0.306 |
| 31 | 155 | 142 | 135 | 7 | 13 | 6 | 92 | 40 | 67/111 | 0.916 | 0.325 |
| 32 | 159 | 140 | 138 | 2 | 19 | 5 | 106 | 40 | 72/123 | 0.881 | 0.475 |
| 33 | 153 | 140 | 137 | 3 | 13 | 4 | 100 | 39 | 64/123 | 0.915 | 0.333 |
| 34 | 152 | 136 | 132 | 4 | 16 | 5 | 94 | 40 | 64/123 | 0.895 | 0.400 |
| 35 | 145 | 134 | 133 | 1 | 11 | 4 | 92 | 38 | 70/131 | 0.924 | 0.289 |
| 36 | 165 | 156 | 149 | 7 | 9 | 5 | 99 | 35 | 78/137 | 0.945 | 0.257 |
| 37 | 159 | 147 | 143 | 4 | 12 | 4 | 94 | 41 | 76/137 | 0.925 | 0.293 |
| 38 | 149 | 138 | 135 | 3 | 11 | 3 | 90 | 36 | 75/137 | 0.926 | 0.306 |
| 39 | 156 | 147 | 144 | 3 | 9 | 5 | 96 | 34 | 72/137 | 0.942 | 0.265 |
| 40 | 167 | 159 | 154 | 5 | 8 | 4 | 88 | 40 | 77/145 | 0.952 | 0.200 |
| 41 | 155 | 143 | 142 | 1 | 12 | 3 | 101 | 42 | 75/145 | 0.923 | 0.286 |
| 42 | 153 | 141 | 140 | 1 | 12 | 3 | 102 | 39 | 77/157 | 0.922 | 0.308 |
| 43 | 167 | 156 | 149 | 7 | 11 | 6 | 85 | 40 | 85/157 | 0.934 | 0.275 |
| 44 | 153 | 142 | 140 | 2 | 11 | 3 | 96 | 32 | 68/157 | 0.928 | 0.344 |
| 45 | 158 | 151 | 150 | 1 | 7 | 2 | 86 | 32 | 82/161 | 0.956 | 0.219 |
| 46 | 152 | 140 | 135 | 5 | 12 | 4 | 90 | 39 | 76/161 | 0.921 | 0.308 |
| 47 | 150 | 139 | 135 | 4 | 11 | 4 | 95 | 39 | 78/161 | 0.927 | 0.282 |
| 48 | 162 | 149 | 147 | 2 | 13 | 4 | 89 | 36 | 75/167 | 0.920 | 0.361 |
| 49 | 179 | 162 | 160 | 2 | 17 | 3 | 98 | 47 | 83/183 | 0.905 | 0.362 |
| 50 | 153 | 144 | 140 | 4 | 9 | 4 | 89 | 38 | 76/201 | 0.941 | 0.237 |
| 51 | 148 | 134 | 130 | 4 | 14 | 4 | 97 | 41 | 82/201 | 0.905 | 0.341 |
| 52 | 165 | 157 | 154 | 3 | 8 | 3 | 89 | 34 | 85/201 | 0.952 | 0.235 |
| 53 | 148 | 137 | 135 | 2 | 11 | 2 | 92 | 34 | 72/201 | 0.926 | 0.324 |
| 54 | 147 | 139 | 136 | 3 | 8 | 3 | 90 | 39 | 79/215 | 0.946 | 0.205 |
| 55 | 153 | 142 | 139 | 3 | 11 | 4 | 93 | 35 | 83/215 | 0.928 | 0.314 |
| 56 | 163 | 149 | 146 | 3 | 14 | 3 | 99 | 37 | 85/229 | 0.914 | 0.378 |
| 57 | 165 | 156 | 155 | 1 | 9 | 3 | 91 | 35 | 89/229 | 0.945 | 0.257 |
| 58 | 153 | 142 | 140 | 2 | 11 | 2 | 90 | 36 | 82/229 | 0.928 | 0.306 |
| 59 | 166 | 155 | 154 | 1 | 11 | 3 | 94 | 34 | 93/229 | 0.934 | 0.324 |
| 60 | 155 | 145 | 142 | 3 | 10 | 3 | 88 | 32 | 83/233 | 0.935 | 0.312 |

Burnt fuelled families per turn, m <= 12 (family: count): m=1: {}; m=2: {}; m=3: {(1,3) 18, (3,1) 16}; m=4: {(1,3) 16, (2,4) 4, (3,1) 13, (4,2) 7}; m=5: {(1,3) 10, (1,5) 5, (2,4) 5, (3,1) 17, (3,5) 5, (4,2) 6, (5,1) 7, (5,3) 2}; m=6: {(1,3) 14, (1,5) 5, (2,4) 4, (3,1) 11, (3,5) 3, (4,2) 5, (4,6) 4, (5,1) 8, (5,3) 6, (6,4) 2}; m=7: {(1,3) 11, (1,5) 6, (1,7) 6, (2,4) 2, (3,1) 10, (3,5) 4, (3,7) 2, (4,2) 5, (4,6) 5, (5,1) 4, (5,3) 3, (5,7) 2, (6,4) 3, (7,1) 4, (7,3) 4, (7,5) 2}; m=8: {(1,3) 12, (1,5) 4, (1,7) 3, (2,4) 7, (2,8) 3, (3,1) 14, (3,5) 2, (3,7) 1, (4,2) 6, (4,6) 5, (5,1) 6, (5,3) 3, (5,7) 1, (6,4) 3, (6,8) 2, (7,1) 7, (7,3) 2, (7,5) 1, (8,2) 6, (8,6) 2}; m=9: {(1,3) 11, (1,5) 2, (1,7) 1, (1,9) 9, (2,4) 3, (2,8) 3, (3,1) 9, (3,5) 5, (3,7) 3, (4,2) 6, (5,1) 5, (5,3) 6, (5,7) 1, (5,9) 2, (6,4) 7, (6,8) 1, (7,1) 3, (7,3) 2, (7,5) 2, (7,9) 1, (8,2) 1, (8,6) 1, (9,1) 7, (9,7) 1}; m=10: {(1,3) 12, (1,5) 5, (1,7) 3, (1,9) 3, (2,4) 6, (2,8) 3, (3,1) 11, (3,5) 3, (3,7) 4, (4,2) 5, (4,6) 3, (4,10) 1, (5,1) 4, (5,3) 3, (5,7) 1, (6,4) 2, (6,8) 1, (7,1) 4, (7,3) 3, (7,5) 1, (8,2) 2, (8,6) 3, (8,10) 1, (9,1) 3, (9,5) 2, (10,8) 1}; m=11: {(1,3) 11, (1,5) 4, (1,7) 4, (1,9) 8, (2,4) 2, (2,8) 1, (3,1) 10, (3,5) 5, (3,7) 1, (4,2) 6, (4,6) 3, (4,10) 1, (5,1) 5, (5,3) 6, (5,7) 1, (5,9) 1, (6,4) 7, (7,1) 2, (7,3) 3, (7,5) 2, (7,9) 1, (8,2) 3, (8,6) 2, (9,1) 3, (9,5) 1, (9,7) 2, (10,4) 1, (10,8) 1}; m=12: {(1,3) 11, (1,5) 7, (1,7) 4, (1,9) 3, (2,4) 8, (2,8) 5, (2,12) 3, (3,1) 14, (3,5) 4, (3,7) 1, (4,2) 1, (4,6) 4, (5,1) 4, (5,3) 5, (5,7) 2, (5,9) 1, (6,4) 2, (6,8) 3, (7,1) 3, (7,3) 3, (8,2) 5, (8,6) 2, (8,10) 1, (9,1) 4, (10,12) 1, (12,2) 4, (12,10) 3};

#### Ledger q = 11, Q = 1000 (turn m = (mQ, (m + 1)Q]; T total charges, B burnt = B_f fuelled-both + B_e ember-carrying, P pure = twins, E embers, pi primes, A primes with P + 2 open, fam/adm = fuelled families present / admissible with max <= m)

| m | T | B | B_f | B_e | P | E | pi | A | fam/adm | B/T | P/A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 43 | 17 | 0 | 17 | 26 | 73 | 135 | 36 | 1/1 | 0.395 | 0.722 |
| 2 | 57 | 36 | 0 | 36 | 21 | 52 | 127 | 28 | 1/1 | 0.632 | 0.750 |
| 3 | 79 | 58 | 34 | 24 | 21 | 43 | 120 | 45 | 3/3 | 0.734 | 0.467 |
| 4 | 90 | 67 | 40 | 27 | 23 | 34 | 119 | 41 | 5/5 | 0.744 | 0.561 |
| 5 | 98 | 81 | 57 | 24 | 17 | 30 | 114 | 36 | 9/9 | 0.827 | 0.472 |
| 6 | 100 | 81 | 62 | 19 | 19 | 29 | 117 | 40 | 11/11 | 0.810 | 0.475 |
| 7 | 105 | 92 | 73 | 19 | 13 | 25 | 107 | 38 | 17/17 | 0.876 | 0.342 |
| 8 | 121 | 106 | 90 | 16 | 15 | 23 | 110 | 36 | 21/21 | 0.876 | 0.417 |
| 9 | 126 | 111 | 92 | 19 | 15 | 21 | 112 | 39 | 25/27 | 0.881 | 0.385 |
| 10 | 119 | 103 | 90 | 13 | 16 | 20 | 106 | 39 | 27/31 | 0.866 | 0.410 |
| 11 | 135 | 121 | 106 | 15 | 14 | 17 | 103 | 42 | 35/41 | 0.896 | 0.333 |
| 12 | 152 | 141 | 118 | 23 | 11 | 18 | 109 | 41 | 34/45 | 0.928 | 0.268 |
| 13 | 145 | 130 | 115 | 15 | 15 | 17 | 105 | 43 | 38/45 | 0.897 | 0.349 |
| 14 | 135 | 124 | 108 | 16 | 11 | 15 | 102 | 43 | 40/51 | 0.919 | 0.256 |
| 15 | 152 | 140 | 126 | 14 | 12 | 17 | 108 | 44 | 44/57 | 0.921 | 0.273 |
| 16 | 153 | 140 | 128 | 12 | 13 | 14 | 98 | 38 | 52/65 | 0.915 | 0.342 |
| 17 | 143 | 125 | 110 | 15 | 18 | 14 | 104 | 45 | 45/65 | 0.874 | 0.400 |
| 18 | 149 | 137 | 130 | 7 | 12 | 13 | 94 | 36 | 52/71 | 0.919 | 0.333 |
| 19 | 164 | 149 | 137 | 12 | 15 | 15 | 104 | 44 | 53/71 | 0.909 | 0.341 |
| 20 | 153 | 138 | 127 | 11 | 15 | 10 | 98 | 38 | 56/79 | 0.902 | 0.395 |
| 21 | 162 | 147 | 135 | 12 | 15 | 14 | 104 | 48 | 62/85 | 0.907 | 0.312 |
| 22 | 159 | 143 | 133 | 10 | 16 | 9 | 100 | 44 | 61/95 | 0.899 | 0.364 |
| 23 | 167 | 153 | 142 | 11 | 14 | 11 | 104 | 46 | 68/95 | 0.916 | 0.304 |
| 24 | 160 | 154 | 144 | 10 | 6 | 13 | 94 | 35 | 57/103 | 0.963 | 0.171 |
| 25 | 166 | 154 | 149 | 5 | 12 | 9 | 98 | 42 | 64/115 | 0.928 | 0.286 |
| 26 | 165 | 154 | 143 | 11 | 11 | 11 | 101 | 42 | 66/115 | 0.933 | 0.262 |
| 27 | 157 | 142 | 131 | 11 | 15 | 10 | 94 | 36 | 65/125 | 0.904 | 0.417 |
| 28 | 161 | 149 | 142 | 7 | 12 | 9 | 98 | 41 | 71/135 | 0.925 | 0.293 |
| 29 | 167 | 158 | 148 | 10 | 9 | 10 | 92 | 33 | 79/135 | 0.946 | 0.273 |
| 30 | 173 | 162 | 151 | 11 | 11 | 11 | 95 | 38 | 73/143 | 0.936 | 0.289 |
| 31 | 170 | 157 | 147 | 10 | 13 | 9 | 92 | 44 | 76/143 | 0.924 | 0.295 |
| 32 | 175 | 156 | 151 | 5 | 19 | 9 | 106 | 42 | 83/157 | 0.891 | 0.452 |
| 33 | 175 | 162 | 154 | 8 | 13 | 9 | 100 | 42 | 74/165 | 0.926 | 0.310 |
| 34 | 174 | 158 | 149 | 9 | 16 | 9 | 94 | 40 | 76/165 | 0.908 | 0.400 |
| 35 | 172 | 161 | 155 | 6 | 11 | 8 | 92 | 41 | 85/177 | 0.936 | 0.268 |
| 36 | 187 | 178 | 168 | 10 | 9 | 7 | 99 | 38 | 94/185 | 0.952 | 0.237 |
| 37 | 176 | 164 | 154 | 10 | 12 | 8 | 94 | 44 | 87/185 | 0.932 | 0.273 |
| 38 | 175 | 164 | 156 | 8 | 11 | 8 | 90 | 42 | 90/185 | 0.937 | 0.262 |
| 39 | 180 | 171 | 165 | 6 | 9 | 9 | 96 | 37 | 88/185 | 0.950 | 0.243 |
| 40 | 191 | 183 | 174 | 9 | 8 | 7 | 88 | 45 | 91/195 | 0.958 | 0.178 |
| 41 | 178 | 166 | 160 | 6 | 12 | 6 | 101 | 42 | 90/195 | 0.933 | 0.286 |
| 42 | 174 | 162 | 156 | 6 | 12 | 7 | 102 | 44 | 89/207 | 0.931 | 0.273 |
| 43 | 189 | 178 | 167 | 11 | 11 | 11 | 85 | 44 | 98/207 | 0.942 | 0.250 |
| 44 | 173 | 162 | 159 | 3 | 11 | 5 | 96 | 35 | 83/221 | 0.936 | 0.314 |
| 45 | 187 | 180 | 173 | 7 | 7 | 6 | 86 | 33 | 100/227 | 0.963 | 0.212 |
| 46 | 173 | 161 | 154 | 7 | 12 | 7 | 90 | 41 | 94/227 | 0.931 | 0.293 |
| 47 | 172 | 161 | 155 | 6 | 11 | 7 | 95 | 44 | 92/227 | 0.936 | 0.250 |
| 48 | 182 | 169 | 164 | 5 | 13 | 8 | 89 | 40 | 89/235 | 0.929 | 0.325 |
| 49 | 199 | 182 | 177 | 5 | 17 | 7 | 98 | 49 | 99/255 | 0.915 | 0.347 |
| 50 | 174 | 165 | 158 | 7 | 9 | 6 | 89 | 39 | 92/275 | 0.948 | 0.231 |
| 51 | 176 | 162 | 155 | 7 | 14 | 6 | 97 | 46 | 98/275 | 0.920 | 0.304 |
| 52 | 194 | 186 | 181 | 5 | 8 | 6 | 89 | 37 | 107/275 | 0.959 | 0.216 |
| 53 | 170 | 159 | 152 | 7 | 11 | 6 | 92 | 36 | 86/275 | 0.935 | 0.306 |
| 54 | 166 | 158 | 153 | 5 | 8 | 6 | 90 | 42 | 93/291 | 0.952 | 0.190 |
| 55 | 173 | 162 | 158 | 4 | 11 | 6 | 93 | 38 | 99/305 | 0.936 | 0.289 |
| 56 | 191 | 177 | 172 | 5 | 14 | 6 | 99 | 42 | 107/321 | 0.927 | 0.333 |
| 57 | 194 | 185 | 180 | 5 | 9 | 5 | 91 | 40 | 109/321 | 0.954 | 0.225 |
| 58 | 174 | 163 | 159 | 4 | 11 | 6 | 90 | 38 | 99/321 | 0.937 | 0.289 |
| 59 | 195 | 184 | 179 | 5 | 11 | 7 | 94 | 35 | 113/321 | 0.944 | 0.314 |
| 60 | 181 | 171 | 162 | 9 | 10 | 6 | 88 | 33 | 100/327 | 0.945 | 0.303 |

Burnt fuelled families per turn, m <= 12 (family: count): m=1: {}; m=2: {}; m=3: {(1,3) 18, (3,1) 16}; m=4: {(1,3) 16, (2,4) 4, (3,1) 13, (4,2) 7}; m=5: {(1,3) 10, (1,5) 5, (2,4) 5, (3,1) 17, (3,5) 5, (4,2) 6, (5,1) 7, (5,3) 2}; m=6: {(1,3) 14, (1,5) 5, (2,4) 4, (3,1) 11, (3,5) 3, (4,2) 5, (4,6) 4, (5,1) 8, (5,3) 6, (6,4) 2}; m=7: {(1,3) 11, (1,5) 6, (1,7) 6, (2,4) 2, (3,1) 10, (3,5) 4, (3,7) 2, (4,2) 5, (4,6) 5, (5,1) 4, (5,3) 3, (5,7) 2, (6,4) 3, (7,1) 4, (7,3) 4, (7,5) 2}; m=8: {(1,3) 12, (1,5) 4, (1,7) 3, (2,4) 7, (2,8) 3, (3,1) 14, (3,5) 2, (3,7) 1, (4,2) 6, (4,6) 5, (5,1) 6, (5,3) 3, (5,7) 1, (6,4) 3, (6,8) 2, (7,1) 7, (7,3) 2, (7,5) 1, (8,2) 6, (8,6) 2}; m=9: {(1,3) 11, (1,5) 2, (1,7) 1, (1,9) 9, (2,4) 3, (2,8) 3, (3,1) 9, (3,5) 5, (3,7) 3, (4,2) 6, (5,1) 5, (5,3) 6, (5,7) 1, (5,9) 2, (6,4) 7, (6,8) 1, (7,1) 3, (7,3) 2, (7,5) 2, (7,9) 1, (8,2) 1, (8,6) 1, (9,1) 7, (9,7) 1}; m=10: {(1,3) 12, (1,5) 5, (1,7) 3, (1,9) 3, (2,4) 6, (2,8) 3, (3,1) 11, (3,5) 3, (3,7) 4, (4,2) 5, (4,6) 3, (4,10) 1, (5,1) 4, (5,3) 3, (5,7) 1, (6,4) 2, (6,8) 1, (7,1) 4, (7,3) 3, (7,5) 1, (8,2) 2, (8,6) 3, (8,10) 1, (9,1) 3, (9,5) 2, (10,8) 1}; m=11: {(1,3) 11, (1,5) 4, (1,7) 4, (1,9) 8, (2,4) 2, (2,8) 1, (3,1) 10, (3,5) 5, (3,7) 1, (3,11) 2, (4,2) 6, (4,6) 3, (4,10) 1, (5,1) 5, (5,3) 6, (5,7) 1, (5,9) 1, (6,4) 7, (7,1) 2, (7,3) 3, (7,5) 2, (7,9) 1, (7,11) 2, (8,2) 3, (8,6) 2, (9,1) 3, (9,5) 1, (9,7) 2, (10,4) 1, (10,8) 1, (11,1) 1, (11,3) 1, (11,5) 2, (11,9) 1}; m=12: {(1,3) 11, (1,5) 7, (1,7) 4, (1,9) 3, (1,11) 3, (2,4) 8, (2,8) 5, (2,12) 3, (3,1) 14, (3,5) 4, (3,7) 1, (3,11) 2, (4,2) 1, (4,6) 4, (5,1) 4, (5,3) 5, (5,7) 2, (5,9) 1, (6,4) 2, (6,8) 3, (7,1) 3, (7,3) 3, (7,11) 1, (8,2) 5, (8,6) 2, (8,10) 1, (9,1) 4, (9,11) 1, (10,12) 1, (11,1) 2, (11,3) 1, (12,2) 4, (12,10) 3};

#### Ledger q = 5, Q = 10000 (turn m = (mQ, (m + 1)Q]; T total charges, B burnt = B_f fuelled-both + B_e ember-carrying, P pure = twins, E embers, pi primes, A primes with P + 2 open, fam/adm = fuelled families present / admissible with max <= m)

| m | T | B | B_f | B_e | P | E | pi | A | fam/adm | B/T | P/A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 140 | 3 | 0 | 3 | 137 | 37 | 1033 | 139 | 1/1 | 0.021 | 0.986 |
| 2 | 137 | 12 | 0 | 12 | 125 | 21 | 983 | 126 | 1/1 | 0.088 | 0.992 |
| 3 | 328 | 204 | 195 | 9 | 124 | 19 | 958 | 219 | 3/3 | 0.622 | 0.566 |
| 4 | 369 | 255 | 244 | 11 | 114 | 13 | 930 | 212 | 5/5 | 0.691 | 0.538 |
| 5 | 445 | 339 | 333 | 6 | 106 | 12 | 924 | 224 | 9/9 | 0.762 | 0.473 |
| 6 | 483 | 389 | 383 | 6 | 94 | 11 | 878 | 213 | 11/11 | 0.805 | 0.441 |
| 7 | 480 | 378 | 373 | 5 | 102 | 9 | 902 | 210 | 11/11 | 0.787 | 0.486 |
| 8 | 521 | 412 | 408 | 4 | 109 | 7 | 876 | 202 | 15/15 | 0.791 | 0.540 |
| 9 | 597 | 489 | 486 | 3 | 108 | 9 | 879 | 235 | 19/19 | 0.819 | 0.460 |
| 10 | 624 | 528 | 522 | 6 | 96 | 6 | 861 | 214 | 23/23 | 0.846 | 0.449 |
| 11 | 604 | 500 | 495 | 5 | 104 | 6 | 848 | 225 | 23/23 | 0.828 | 0.462 |
| 12 | 673 | 580 | 577 | 3 | 93 | 6 | 858 | 226 | 27/27 | 0.862 | 0.412 |
| 13 | 613 | 522 | 517 | 5 | 91 | 5 | 851 | 217 | 27/27 | 0.852 | 0.419 |
| 14 | 642 | 549 | 545 | 4 | 93 | 5 | 838 | 215 | 27/27 | 0.855 | 0.433 |
| 15 | 667 | 567 | 563 | 4 | 100 | 6 | 835 | 242 | 29/29 | 0.850 | 0.413 |
| 16 | 697 | 616 | 612 | 4 | 81 | 5 | 814 | 225 | 35/35 | 0.884 | 0.360 |
| 17 | 672 | 577 | 574 | 3 | 95 | 4 | 845 | 223 | 35/35 | 0.859 | 0.426 |
| 18 | 707 | 609 | 609 | 0 | 98 | 4 | 828 | 238 | 41/41 | 0.861 | 0.412 |
| 19 | 700 | 615 | 612 | 3 | 85 | 5 | 814 | 226 | 41/41 | 0.879 | 0.376 |
| 20 | 719 | 626 | 625 | 1 | 93 | 4 | 823 | 230 | 46/47 | 0.871 | 0.404 |
| 21 | 721 | 645 | 644 | 1 | 76 | 2 | 811 | 214 | 46/47 | 0.895 | 0.355 |
| 22 | 700 | 607 | 606 | 1 | 93 | 2 | 819 | 225 | 46/47 | 0.867 | 0.413 |
| 23 | 698 | 617 | 615 | 2 | 81 | 5 | 784 | 211 | 47/47 | 0.884 | 0.384 |
| 24 | 734 | 649 | 648 | 1 | 85 | 4 | 823 | 218 | 51/51 | 0.884 | 0.390 |
| 25 | 735 | 660 | 659 | 1 | 75 | 3 | 793 | 213 | 57/57 | 0.898 | 0.352 |
| 26 | 732 | 642 | 638 | 4 | 90 | 3 | 805 | 226 | 57/57 | 0.877 | 0.398 |
| 27 | 733 | 651 | 647 | 4 | 82 | 3 | 790 | 227 | 61/63 | 0.888 | 0.361 |
| 28 | 752 | 666 | 664 | 2 | 86 | 2 | 792 | 238 | 60/63 | 0.886 | 0.361 |
| 29 | 731 | 658 | 654 | 4 | 73 | 4 | 773 | 203 | 61/63 | 0.900 | 0.360 |
| 30 | 772 | 683 | 682 | 1 | 89 | 2 | 803 | 234 | 66/69 | 0.885 | 0.380 |
| 31 | 787 | 700 | 698 | 2 | 87 | 4 | 808 | 230 | 67/69 | 0.889 | 0.378 |
| 32 | 775 | 672 | 669 | 3 | 103 | 3 | 796 | 244 | 75/79 | 0.867 | 0.422 |
| 33 | 749 | 677 | 676 | 1 | 72 | 2 | 778 | 206 | 73/79 | 0.904 | 0.350 |
| 34 | 758 | 681 | 679 | 2 | 77 | 2 | 795 | 209 | 77/79 | 0.898 | 0.368 |
| 35 | 766 | 686 | 684 | 2 | 80 | 2 | 780 | 214 | 74/79 | 0.896 | 0.374 |
| 36 | 769 | 697 | 696 | 1 | 72 | 2 | 765 | 212 | 75/83 | 0.906 | 0.340 |
| 37 | 742 | 673 | 673 | 0 | 69 | 2 | 778 | 203 | 77/83 | 0.907 | 0.340 |
| 38 | 766 | 688 | 687 | 1 | 78 | 2 | 767 | 206 | 78/83 | 0.898 | 0.379 |
| 39 | 795 | 712 | 709 | 3 | 83 | 4 | 793 | 212 | 76/83 | 0.896 | 0.392 |
| 40 | 766 | 689 | 689 | 0 | 77 | 2 | 754 | 203 | 83/89 | 0.899 | 0.379 |
| 41 | 783 | 703 | 702 | 1 | 80 | 2 | 776 | 205 | 85/89 | 0.898 | 0.390 |
| 42 | 775 | 682 | 682 | 0 | 93 | 1 | 772 | 221 | 86/89 | 0.880 | 0.421 |
| 43 | 748 | 683 | 683 | 0 | 65 | 2 | 779 | 207 | 82/89 | 0.913 | 0.314 |
| 44 | 752 | 677 | 677 | 0 | 75 | 2 | 765 | 202 | 84/89 | 0.900 | 0.371 |
| 45 | 747 | 670 | 669 | 1 | 77 | 1 | 752 | 214 | 85/91 | 0.897 | 0.360 |
| 46 | 783 | 718 | 716 | 2 | 65 | 3 | 765 | 218 | 89/91 | 0.917 | 0.298 |
| 47 | 788 | 707 | 706 | 1 | 81 | 2 | 782 | 202 | 86/91 | 0.897 | 0.401 |
| 48 | 753 | 683 | 682 | 1 | 70 | 1 | 761 | 201 | 88/95 | 0.907 | 0.348 |
| 49 | 766 | 688 | 686 | 2 | 78 | 4 | 772 | 212 | 87/95 | 0.898 | 0.368 |
| 50 | 763 | 696 | 695 | 1 | 67 | 1 | 753 | 194 | 94/111 | 0.912 | 0.345 |
| 51 | 797 | 717 | 715 | 2 | 80 | 2 | 770 | 214 | 98/111 | 0.900 | 0.374 |
| 52 | 761 | 670 | 670 | 0 | 91 | 2 | 764 | 223 | 93/111 | 0.880 | 0.408 |
| 53 | 786 | 713 | 712 | 1 | 73 | 2 | 747 | 210 | 95/111 | 0.907 | 0.348 |
| 54 | 793 | 722 | 722 | 0 | 71 | 1 | 750 | 209 | 99/123 | 0.910 | 0.340 |
| 55 | 809 | 738 | 737 | 1 | 71 | 2 | 750 | 217 | 108/123 | 0.912 | 0.327 |
| 56 | 784 | 711 | 709 | 2 | 73 | 1 | 747 | 220 | 99/123 | 0.907 | 0.332 |
| 57 | 774 | 687 | 687 | 0 | 87 | 1 | 769 | 223 | 103/123 | 0.888 | 0.390 |
| 58 | 779 | 701 | 700 | 1 | 78 | 2 | 763 | 210 | 104/123 | 0.900 | 0.371 |
| 59 | 749 | 674 | 674 | 0 | 75 | 2 | 747 | 202 | 100/123 | 0.900 | 0.371 |
| 60 | 817 | 738 | 737 | 1 | 79 | 1 | 763 | 214 | 103/125 | 0.903 | 0.369 |

Burnt fuelled families per turn, m <= 12 (family: count): m=1: {}; m=2: {}; m=3: {(1,3) 94, (3,1) 101}; m=4: {(1,3) 98, (2,4) 29, (3,1) 79, (4,2) 38}; m=5: {(1,3) 82, (1,5) 35, (2,4) 30, (3,1) 70, (3,5) 21, (4,2) 37, (5,1) 35, (5,3) 23}; m=6: {(1,3) 78, (1,5) 41, (2,4) 37, (3,1) 72, (3,5) 25, (4,2) 29, (4,6) 26, (5,1) 25, (5,3) 26, (6,4) 24}; m=7: {(1,3) 76, (1,5) 32, (2,4) 27, (3,1) 80, (3,5) 23, (4,2) 32, (4,6) 27, (5,1) 30, (5,3) 23, (6,4) 23}; m=8: {(1,3) 63, (1,5) 30, (2,4) 30, (2,8) 15, (3,1) 77, (3,5) 19, (4,2) 39, (4,6) 22, (5,1) 35, (5,3) 15, (6,4) 23, (6,8) 10, (8,2) 17, (8,6) 13}; m=9: {(1,3) 69, (1,5) 33, (1,9) 25, (2,4) 35, (2,8) 15, (3,1) 72, (3,5) 21, (4,2) 26, (4,6) 27, (5,1) 32, (5,3) 21, (5,9) 6, (6,4) 26, (6,8) 16, (8,2) 16, (8,6) 12, (9,1) 25, (9,5) 9}; m=10: {(1,3) 51, (1,5) 34, (1,9) 33, (2,4) 35, (2,8) 12, (3,1) 86, (3,5) 23, (4,2) 24, (4,6) 23, (4,10) 11, (5,1) 37, (5,3) 24, (5,9) 5, (6,4) 23, (6,8) 16, (8,2) 19, (8,6) 14, (8,10) 5, (9,1) 26, (9,5) 9, (10,4) 7, (10,8) 5}; m=11: {(1,3) 66, (1,5) 30, (1,9) 25, (2,4) 37, (2,8) 16, (3,1) 68, (3,5) 28, (4,2) 31, (4,6) 22, (4,10) 12, (5,1) 22, (5,3) 19, (5,9) 9, (6,4) 20, (6,8) 8, (8,2) 15, (8,6) 12, (8,10) 1, (9,1) 33, (9,5) 8, (10,4) 9, (10,8) 4}; m=12: {(1,3) 73, (1,5) 31, (1,9) 29, (2,4) 32, (2,8) 17, (2,12) 20, (3,1) 91, (3,5) 21, (4,2) 24, (4,6) 21, (4,10) 7, (5,1) 32, (5,3) 23, (5,9) 4, (6,4) 19, (6,8) 14, (8,2) 11, (8,6) 15, (8,10) 8, (9,1) 19, (9,5) 12, (10,4) 10, (10,8) 7, (10,12) 10, (12,2) 20, (12,10) 7};

#### Ledger q = 7, Q = 10000 (turn m = (mQ, (m + 1)Q]; T total charges, B burnt = B_f fuelled-both + B_e ember-carrying, P pure = twins, E embers, pi primes, A primes with P + 2 open, fam/adm = fuelled families present / admissible with max <= m)

| m | T | B | B_f | B_e | P | E | pi | A | fam/adm | B/T | P/A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 148 | 11 | 0 | 11 | 137 | 89 | 1033 | 142 | 1/1 | 0.074 | 0.965 |
| 2 | 155 | 30 | 0 | 30 | 125 | 56 | 983 | 127 | 1/1 | 0.194 | 0.984 |
| 3 | 344 | 220 | 195 | 25 | 124 | 47 | 958 | 222 | 3/3 | 0.640 | 0.559 |
| 4 | 382 | 268 | 244 | 24 | 114 | 36 | 930 | 212 | 5/5 | 0.702 | 0.538 |
| 5 | 460 | 354 | 333 | 21 | 106 | 31 | 924 | 224 | 9/9 | 0.770 | 0.473 |
| 6 | 495 | 401 | 383 | 18 | 94 | 29 | 878 | 213 | 11/11 | 0.810 | 0.441 |
| 7 | 574 | 472 | 456 | 16 | 102 | 25 | 902 | 231 | 17/17 | 0.822 | 0.442 |
| 8 | 614 | 505 | 495 | 10 | 109 | 22 | 876 | 222 | 21/21 | 0.822 | 0.491 |
| 9 | 694 | 586 | 571 | 15 | 108 | 21 | 879 | 253 | 27/27 | 0.844 | 0.427 |
| 10 | 727 | 631 | 617 | 14 | 96 | 19 | 861 | 238 | 31/31 | 0.868 | 0.403 |
| 11 | 693 | 589 | 579 | 10 | 104 | 17 | 848 | 244 | 31/31 | 0.850 | 0.426 |
| 12 | 766 | 673 | 667 | 6 | 93 | 18 | 858 | 247 | 35/35 | 0.879 | 0.377 |
| 13 | 704 | 613 | 602 | 11 | 91 | 15 | 851 | 240 | 35/35 | 0.871 | 0.379 |
| 14 | 755 | 662 | 652 | 10 | 93 | 13 | 838 | 229 | 41/41 | 0.877 | 0.406 |
| 15 | 790 | 690 | 680 | 10 | 100 | 16 | 835 | 260 | 45/45 | 0.873 | 0.385 |
| 16 | 831 | 750 | 742 | 8 | 81 | 13 | 814 | 243 | 53/53 | 0.903 | 0.333 |
| 17 | 815 | 720 | 705 | 15 | 95 | 13 | 845 | 249 | 53/53 | 0.883 | 0.382 |
| 18 | 830 | 732 | 728 | 4 | 98 | 11 | 828 | 256 | 59/59 | 0.882 | 0.383 |
| 19 | 829 | 744 | 733 | 11 | 85 | 13 | 814 | 241 | 59/59 | 0.897 | 0.353 |
| 20 | 841 | 748 | 742 | 6 | 93 | 10 | 823 | 247 | 66/67 | 0.889 | 0.377 |
| 21 | 896 | 820 | 812 | 8 | 76 | 11 | 811 | 257 | 70/71 | 0.915 | 0.296 |
| 22 | 850 | 757 | 752 | 5 | 93 | 9 | 819 | 253 | 69/71 | 0.891 | 0.368 |
| 23 | 852 | 771 | 764 | 7 | 81 | 11 | 784 | 239 | 70/71 | 0.905 | 0.339 |
| 24 | 889 | 804 | 798 | 6 | 85 | 9 | 823 | 244 | 77/77 | 0.904 | 0.348 |
| 25 | 905 | 830 | 822 | 8 | 75 | 12 | 793 | 245 | 84/87 | 0.917 | 0.306 |
| 26 | 897 | 807 | 794 | 13 | 90 | 8 | 805 | 263 | 86/87 | 0.900 | 0.342 |
| 27 | 908 | 826 | 816 | 10 | 82 | 9 | 790 | 269 | 90/95 | 0.910 | 0.305 |
| 28 | 938 | 852 | 848 | 4 | 86 | 7 | 792 | 272 | 97/103 | 0.908 | 0.316 |
| 29 | 909 | 836 | 832 | 4 | 73 | 8 | 773 | 222 | 99/103 | 0.920 | 0.329 |
| 30 | 958 | 869 | 862 | 7 | 89 | 10 | 803 | 263 | 102/111 | 0.907 | 0.338 |
| 31 | 981 | 894 | 888 | 6 | 87 | 7 | 808 | 255 | 107/111 | 0.911 | 0.341 |
| 32 | 976 | 873 | 867 | 6 | 103 | 9 | 796 | 275 | 115/123 | 0.894 | 0.375 |
| 33 | 940 | 868 | 864 | 4 | 72 | 7 | 778 | 234 | 114/123 | 0.923 | 0.308 |
| 34 | 931 | 854 | 848 | 6 | 77 | 7 | 795 | 241 | 116/123 | 0.917 | 0.320 |
| 35 | 970 | 890 | 881 | 9 | 80 | 8 | 780 | 249 | 120/131 | 0.918 | 0.321 |
| 36 | 979 | 907 | 903 | 4 | 72 | 6 | 765 | 240 | 122/137 | 0.926 | 0.300 |
| 37 | 935 | 866 | 863 | 3 | 69 | 5 | 778 | 235 | 122/137 | 0.926 | 0.294 |
| 38 | 978 | 900 | 895 | 5 | 78 | 8 | 767 | 234 | 127/137 | 0.920 | 0.333 |
| 39 | 976 | 893 | 887 | 6 | 83 | 8 | 793 | 233 | 120/137 | 0.915 | 0.356 |
| 40 | 979 | 902 | 900 | 2 | 77 | 6 | 754 | 238 | 130/145 | 0.921 | 0.324 |
| 41 | 996 | 916 | 915 | 1 | 80 | 6 | 776 | 241 | 132/145 | 0.920 | 0.332 |
| 42 | 976 | 883 | 881 | 2 | 93 | 6 | 772 | 255 | 145/157 | 0.905 | 0.365 |
| 43 | 975 | 910 | 906 | 4 | 65 | 7 | 779 | 239 | 137/157 | 0.933 | 0.272 |
| 44 | 977 | 902 | 898 | 4 | 75 | 5 | 765 | 240 | 140/157 | 0.923 | 0.312 |
| 45 | 982 | 905 | 900 | 5 | 77 | 7 | 752 | 245 | 143/161 | 0.922 | 0.314 |
| 46 | 994 | 929 | 923 | 6 | 65 | 5 | 765 | 244 | 146/161 | 0.935 | 0.266 |
| 47 | 992 | 911 | 908 | 3 | 81 | 6 | 782 | 237 | 142/161 | 0.918 | 0.342 |
| 48 | 982 | 912 | 905 | 7 | 70 | 5 | 761 | 230 | 145/167 | 0.929 | 0.304 |
| 49 | 1004 | 926 | 922 | 4 | 78 | 6 | 772 | 246 | 156/183 | 0.922 | 0.317 |
| 50 | 982 | 915 | 912 | 3 | 67 | 6 | 753 | 231 | 160/201 | 0.932 | 0.290 |
| 51 | 1017 | 937 | 932 | 5 | 80 | 6 | 770 | 245 | 167/201 | 0.921 | 0.327 |
| 52 | 1003 | 912 | 908 | 4 | 91 | 5 | 764 | 260 | 161/201 | 0.909 | 0.350 |
| 53 | 998 | 925 | 922 | 3 | 73 | 5 | 747 | 240 | 165/201 | 0.927 | 0.304 |
| 54 | 1033 | 962 | 961 | 1 | 71 | 5 | 750 | 244 | 168/215 | 0.931 | 0.291 |
| 55 | 1032 | 961 | 958 | 3 | 71 | 6 | 750 | 249 | 179/215 | 0.931 | 0.285 |
| 56 | 1031 | 958 | 953 | 5 | 73 | 3 | 747 | 263 | 175/229 | 0.929 | 0.278 |
| 57 | 1011 | 924 | 921 | 3 | 87 | 4 | 769 | 259 | 175/229 | 0.914 | 0.336 |
| 58 | 1036 | 958 | 952 | 6 | 78 | 6 | 763 | 246 | 180/229 | 0.925 | 0.317 |
| 59 | 986 | 911 | 907 | 4 | 75 | 5 | 747 | 233 | 176/229 | 0.924 | 0.322 |
| 60 | 1034 | 955 | 950 | 5 | 79 | 5 | 763 | 241 | 173/233 | 0.924 | 0.328 |

Burnt fuelled families per turn, m <= 12 (family: count): m=1: {}; m=2: {}; m=3: {(1,3) 94, (3,1) 101}; m=4: {(1,3) 98, (2,4) 29, (3,1) 79, (4,2) 38}; m=5: {(1,3) 82, (1,5) 35, (2,4) 30, (3,1) 70, (3,5) 21, (4,2) 37, (5,1) 35, (5,3) 23}; m=6: {(1,3) 78, (1,5) 41, (2,4) 37, (3,1) 72, (3,5) 25, (4,2) 29, (4,6) 26, (5,1) 25, (5,3) 26, (6,4) 24}; m=7: {(1,3) 76, (1,5) 32, (1,7) 20, (2,4) 27, (3,1) 80, (3,5) 23, (3,7) 17, (4,2) 32, (4,6) 27, (5,1) 30, (5,3) 23, (5,7) 3, (6,4) 23, (7,1) 22, (7,3) 14, (7,5) 7}; m=8: {(1,3) 63, (1,5) 30, (1,7) 20, (2,4) 30, (2,8) 15, (3,1) 77, (3,5) 19, (3,7) 15, (4,2) 39, (4,6) 22, (5,1) 35, (5,3) 15, (5,7) 7, (6,4) 23, (6,8) 10, (7,1) 22, (7,3) 19, (7,5) 4, (8,2) 17, (8,6) 13}; m=9: {(1,3) 69, (1,5) 33, (1,7) 16, (1,9) 25, (2,4) 35, (2,8) 15, (3,1) 72, (3,5) 21, (3,7) 19, (4,2) 26, (4,6) 27, (5,1) 32, (5,3) 21, (5,7) 4, (5,9) 6, (6,4) 26, (6,8) 16, (7,1) 14, (7,3) 15, (7,5) 9, (7,9) 4, (8,2) 16, (8,6) 12, (9,1) 25, (9,5) 9, (9,7) 4}; m=10: {(1,3) 51, (1,5) 34, (1,7) 24, (1,9) 33, (2,4) 35, (2,8) 12, (3,1) 86, (3,5) 23, (3,7) 11, (4,2) 24, (4,6) 23, (4,10) 11, (5,1) 37, (5,3) 24, (5,7) 6, (5,9) 5, (6,4) 23, (6,8) 16, (7,1) 20, (7,3) 14, (7,5) 8, (7,9) 6, (8,2) 19, (8,6) 14, (8,10) 5, (9,1) 26, (9,5) 9, (9,7) 6, (10,4) 7, (10,8) 5}; m=11: {(1,3) 66, (1,5) 30, (1,7) 19, (1,9) 25, (2,4) 37, (2,8) 16, (3,1) 68, (3,5) 28, (3,7) 14, (4,2) 31, (4,6) 22, (4,10) 12, (5,1) 22, (5,3) 19, (5,7) 5, (5,9) 9, (6,4) 20, (6,8) 8, (7,1) 17, (7,3) 15, (7,5) 4, (7,9) 4, (8,2) 15, (8,6) 12, (8,10) 1, (9,1) 33, (9,5) 8, (9,7) 6, (10,4) 9, (10,8) 4}; m=12: {(1,3) 73, (1,5) 31, (1,7) 21, (1,9) 29, (2,4) 32, (2,8) 17, (2,12) 20, (3,1) 91, (3,5) 21, (3,7) 12, (4,2) 24, (4,6) 21, (4,10) 7, (5,1) 32, (5,3) 23, (5,7) 6, (5,9) 4, (6,4) 19, (6,8) 14, (7,1) 15, (7,3) 17, (7,5) 8, (7,9) 5, (8,2) 11, (8,6) 15, (8,10) 8, (9,1) 19, (9,5) 12, (9,7) 6, (10,4) 10, (10,8) 7, (10,12) 10, (12,2) 20, (12,10) 7};

#### Ledger q = 11, Q = 10000 (turn m = (mQ, (m + 1)Q]; T total charges, B burnt = B_f fuelled-both + B_e ember-carrying, P pure = twins, E embers, pi primes, A primes with P + 2 open, fam/adm = fuelled families present / admissible with max <= m)

| m | T | B | B_f | B_e | P | E | pi | A | fam/adm | B/T | P/A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 164 | 27 | 0 | 27 | 137 | 160 | 1033 | 149 | 1/1 | 0.165 | 0.919 |
| 2 | 182 | 57 | 0 | 57 | 125 | 106 | 983 | 129 | 1/1 | 0.313 | 0.969 |
| 3 | 361 | 237 | 195 | 42 | 124 | 87 | 958 | 225 | 3/3 | 0.657 | 0.551 |
| 4 | 400 | 286 | 244 | 42 | 114 | 71 | 930 | 215 | 5/5 | 0.715 | 0.530 |
| 5 | 479 | 373 | 333 | 40 | 106 | 60 | 924 | 227 | 9/9 | 0.779 | 0.467 |
| 6 | 512 | 418 | 383 | 35 | 94 | 56 | 878 | 213 | 11/11 | 0.816 | 0.441 |
| 7 | 587 | 485 | 456 | 29 | 102 | 49 | 902 | 234 | 17/17 | 0.826 | 0.436 |
| 8 | 628 | 519 | 495 | 24 | 109 | 44 | 876 | 222 | 21/21 | 0.826 | 0.491 |
| 9 | 708 | 600 | 571 | 29 | 108 | 42 | 879 | 255 | 27/27 | 0.847 | 0.424 |
| 10 | 738 | 642 | 617 | 25 | 96 | 38 | 861 | 240 | 31/31 | 0.870 | 0.400 |
| 11 | 763 | 659 | 633 | 26 | 104 | 34 | 848 | 261 | 41/41 | 0.864 | 0.398 |
| 12 | 830 | 737 | 721 | 16 | 93 | 33 | 858 | 259 | 45/45 | 0.888 | 0.359 |
| 13 | 775 | 684 | 663 | 21 | 91 | 35 | 851 | 251 | 45/45 | 0.883 | 0.363 |
| 14 | 828 | 735 | 716 | 19 | 93 | 28 | 838 | 243 | 51/51 | 0.888 | 0.383 |
| 15 | 867 | 767 | 747 | 20 | 100 | 29 | 835 | 271 | 57/57 | 0.885 | 0.369 |
| 16 | 896 | 815 | 797 | 18 | 81 | 28 | 814 | 252 | 64/65 | 0.910 | 0.321 |
| 17 | 895 | 800 | 772 | 28 | 95 | 27 | 845 | 258 | 65/65 | 0.894 | 0.368 |
| 18 | 896 | 798 | 786 | 12 | 98 | 23 | 828 | 266 | 71/71 | 0.891 | 0.368 |
| 19 | 905 | 820 | 796 | 24 | 85 | 26 | 814 | 251 | 70/71 | 0.906 | 0.339 |
| 20 | 900 | 807 | 792 | 15 | 93 | 22 | 823 | 256 | 78/79 | 0.897 | 0.363 |
| 21 | 965 | 889 | 871 | 18 | 76 | 25 | 811 | 268 | 84/85 | 0.921 | 0.284 |
| 22 | 928 | 835 | 821 | 14 | 93 | 18 | 819 | 262 | 90/95 | 0.900 | 0.355 |
| 23 | 955 | 874 | 854 | 20 | 81 | 22 | 784 | 251 | 93/95 | 0.915 | 0.323 |
| 24 | 985 | 900 | 881 | 19 | 85 | 21 | 823 | 255 | 98/103 | 0.914 | 0.333 |
| 25 | 997 | 922 | 909 | 13 | 75 | 19 | 793 | 257 | 109/115 | 0.925 | 0.292 |
| 26 | 994 | 904 | 885 | 19 | 90 | 20 | 805 | 272 | 112/115 | 0.909 | 0.331 |
| 27 | 993 | 911 | 891 | 20 | 82 | 19 | 790 | 275 | 112/125 | 0.917 | 0.298 |
| 28 | 1037 | 951 | 939 | 12 | 86 | 15 | 792 | 286 | 125/135 | 0.917 | 0.301 |
| 29 | 1004 | 931 | 920 | 11 | 73 | 19 | 773 | 233 | 127/135 | 0.927 | 0.313 |
| 30 | 1041 | 952 | 937 | 15 | 89 | 19 | 803 | 274 | 129/143 | 0.915 | 0.325 |
| 31 | 1065 | 978 | 967 | 11 | 87 | 15 | 808 | 264 | 130/143 | 0.918 | 0.330 |
| 32 | 1067 | 964 | 953 | 11 | 103 | 18 | 796 | 285 | 141/157 | 0.903 | 0.361 |
| 33 | 1064 | 992 | 979 | 13 | 72 | 17 | 778 | 248 | 149/165 | 0.932 | 0.290 |
| 34 | 1041 | 964 | 950 | 14 | 77 | 15 | 795 | 253 | 147/165 | 0.926 | 0.304 |
| 35 | 1085 | 1005 | 991 | 14 | 80 | 15 | 780 | 269 | 155/177 | 0.926 | 0.297 |
| 36 | 1091 | 1019 | 1007 | 12 | 72 | 15 | 765 | 265 | 156/185 | 0.934 | 0.272 |
| 37 | 1064 | 995 | 986 | 9 | 69 | 13 | 778 | 253 | 157/185 | 0.935 | 0.273 |
| 38 | 1088 | 1010 | 1001 | 9 | 78 | 15 | 767 | 255 | 163/185 | 0.928 | 0.306 |
| 39 | 1095 | 1012 | 1000 | 12 | 83 | 17 | 793 | 254 | 156/185 | 0.924 | 0.327 |
| 40 | 1101 | 1024 | 1015 | 9 | 77 | 13 | 754 | 255 | 170/195 | 0.930 | 0.302 |
| 41 | 1089 | 1009 | 1003 | 6 | 80 | 13 | 776 | 255 | 165/195 | 0.927 | 0.314 |
| 42 | 1083 | 990 | 983 | 7 | 93 | 12 | 772 | 271 | 182/207 | 0.914 | 0.343 |
| 43 | 1100 | 1035 | 1025 | 10 | 65 | 16 | 779 | 263 | 177/207 | 0.941 | 0.247 |
| 44 | 1091 | 1016 | 1010 | 6 | 75 | 11 | 765 | 258 | 182/221 | 0.931 | 0.291 |
| 45 | 1102 | 1025 | 1014 | 11 | 77 | 13 | 752 | 260 | 187/227 | 0.930 | 0.296 |
| 46 | 1122 | 1057 | 1045 | 12 | 65 | 11 | 765 | 258 | 195/227 | 0.942 | 0.252 |
| 47 | 1120 | 1039 | 1032 | 7 | 81 | 13 | 782 | 249 | 185/227 | 0.928 | 0.325 |
| 48 | 1108 | 1038 | 1025 | 13 | 70 | 13 | 761 | 242 | 192/235 | 0.937 | 0.289 |
| 49 | 1136 | 1058 | 1049 | 9 | 78 | 13 | 772 | 268 | 200/255 | 0.931 | 0.291 |
| 50 | 1117 | 1050 | 1046 | 4 | 67 | 11 | 753 | 250 | 212/275 | 0.940 | 0.268 |
| 51 | 1142 | 1062 | 1050 | 12 | 80 | 12 | 770 | 261 | 218/275 | 0.930 | 0.307 |
| 52 | 1137 | 1046 | 1035 | 11 | 91 | 13 | 764 | 281 | 212/275 | 0.920 | 0.324 |
| 53 | 1145 | 1072 | 1062 | 10 | 73 | 11 | 747 | 256 | 216/275 | 0.936 | 0.285 |
| 54 | 1150 | 1079 | 1074 | 5 | 71 | 11 | 750 | 254 | 215/291 | 0.938 | 0.280 |
| 55 | 1168 | 1097 | 1091 | 6 | 71 | 12 | 750 | 267 | 229/305 | 0.939 | 0.266 |
| 56 | 1167 | 1094 | 1085 | 9 | 73 | 7 | 747 | 281 | 228/321 | 0.937 | 0.260 |
| 57 | 1153 | 1066 | 1058 | 8 | 87 | 9 | 769 | 282 | 231/321 | 0.925 | 0.309 |
| 58 | 1176 | 1098 | 1087 | 11 | 78 | 12 | 763 | 265 | 236/321 | 0.934 | 0.294 |
| 59 | 1137 | 1062 | 1053 | 9 | 75 | 11 | 747 | 263 | 239/321 | 0.934 | 0.285 |
| 60 | 1171 | 1092 | 1083 | 9 | 79 | 11 | 763 | 266 | 223/327 | 0.933 | 0.297 |

Burnt fuelled families per turn, m <= 12 (family: count): m=1: {}; m=2: {}; m=3: {(1,3) 94, (3,1) 101}; m=4: {(1,3) 98, (2,4) 29, (3,1) 79, (4,2) 38}; m=5: {(1,3) 82, (1,5) 35, (2,4) 30, (3,1) 70, (3,5) 21, (4,2) 37, (5,1) 35, (5,3) 23}; m=6: {(1,3) 78, (1,5) 41, (2,4) 37, (3,1) 72, (3,5) 25, (4,2) 29, (4,6) 26, (5,1) 25, (5,3) 26, (6,4) 24}; m=7: {(1,3) 76, (1,5) 32, (1,7) 20, (2,4) 27, (3,1) 80, (3,5) 23, (3,7) 17, (4,2) 32, (4,6) 27, (5,1) 30, (5,3) 23, (5,7) 3, (6,4) 23, (7,1) 22, (7,3) 14, (7,5) 7}; m=8: {(1,3) 63, (1,5) 30, (1,7) 20, (2,4) 30, (2,8) 15, (3,1) 77, (3,5) 19, (3,7) 15, (4,2) 39, (4,6) 22, (5,1) 35, (5,3) 15, (5,7) 7, (6,4) 23, (6,8) 10, (7,1) 22, (7,3) 19, (7,5) 4, (8,2) 17, (8,6) 13}; m=9: {(1,3) 69, (1,5) 33, (1,7) 16, (1,9) 25, (2,4) 35, (2,8) 15, (3,1) 72, (3,5) 21, (3,7) 19, (4,2) 26, (4,6) 27, (5,1) 32, (5,3) 21, (5,7) 4, (5,9) 6, (6,4) 26, (6,8) 16, (7,1) 14, (7,3) 15, (7,5) 9, (7,9) 4, (8,2) 16, (8,6) 12, (9,1) 25, (9,5) 9, (9,7) 4}; m=10: {(1,3) 51, (1,5) 34, (1,7) 24, (1,9) 33, (2,4) 35, (2,8) 12, (3,1) 86, (3,5) 23, (3,7) 11, (4,2) 24, (4,6) 23, (4,10) 11, (5,1) 37, (5,3) 24, (5,7) 6, (5,9) 5, (6,4) 23, (6,8) 16, (7,1) 20, (7,3) 14, (7,5) 8, (7,9) 6, (8,2) 19, (8,6) 14, (8,10) 5, (9,1) 26, (9,5) 9, (9,7) 6, (10,4) 7, (10,8) 5}; m=11: {(1,3) 66, (1,5) 30, (1,7) 19, (1,9) 25, (1,11) 15, (2,4) 37, (2,8) 16, (3,1) 68, (3,5) 28, (3,7) 14, (3,11) 6, (4,2) 31, (4,6) 22, (4,10) 12, (5,1) 22, (5,3) 19, (5,7) 5, (5,9) 9, (5,11) 4, (6,4) 20, (6,8) 8, (7,1) 17, (7,3) 15, (7,5) 4, (7,9) 4, (7,11) 3, (8,2) 15, (8,6) 12, (8,10) 1, (9,1) 33, (9,5) 8, (9,7) 6, (9,11) 2, (10,4) 9, (10,8) 4, (11,1) 10, (11,3) 5, (11,5) 1, (11,7) 4, (11,9) 4}; m=12: {(1,3) 73, (1,5) 31, (1,7) 21, (1,9) 29, (1,11) 11, (2,4) 32, (2,8) 17, (2,12) 20, (3,1) 91, (3,5) 21, (3,7) 12, (3,11) 10, (4,2) 24, (4,6) 21, (4,10) 7, (5,1) 32, (5,3) 23, (5,7) 6, (5,9) 4, (5,11) 3, (6,4) 19, (6,8) 14, (7,1) 15, (7,3) 17, (7,5) 8, (7,9) 5, (7,11) 2, (8,2) 11, (8,6) 15, (8,10) 8, (9,1) 19, (9,5) 12, (9,7) 6, (9,11) 2, (10,4) 10, (10,8) 7, (10,12) 10, (11,1) 8, (11,3) 12, (11,5) 2, (11,7) 2, (11,9) 2, (12,2) 20, (12,10) 7};

#### Ledger q = 5, Q = 30000 (turn m = (mQ, (m + 1)Q]; T total charges, B burnt = B_f fuelled-both + B_e ember-carrying, P pure = twins, E embers, pi primes, A primes with P + 2 open, fam/adm = fuelled families present / admissible with max <= m)

| m | T | B | B_f | B_e | P | E | pi | A | fam/adm | B/T | P/A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 349 | 5 | 0 | 5 | 344 | 44 | 2812 | 346 | 1/1 | 0.014 | 0.994 |
| 2 | 317 | 12 | 0 | 12 | 305 | 27 | 2656 | 305 | 1/1 | 0.038 | 1.000 |
| 3 | 730 | 422 | 412 | 10 | 308 | 21 | 2588 | 494 | 3/3 | 0.578 | 0.623 |
| 4 | 904 | 627 | 616 | 11 | 277 | 16 | 2547 | 488 | 5/5 | 0.694 | 0.568 |
| 5 | 1148 | 872 | 862 | 10 | 276 | 15 | 2494 | 553 | 9/9 | 0.760 | 0.499 |
| 6 | 1226 | 950 | 947 | 3 | 276 | 13 | 2465 | 541 | 11/11 | 0.775 | 0.510 |
| 7 | 1215 | 965 | 961 | 4 | 250 | 9 | 2414 | 522 | 11/11 | 0.794 | 0.479 |
| 8 | 1280 | 1030 | 1024 | 6 | 250 | 10 | 2421 | 505 | 15/15 | 0.805 | 0.495 |
| 9 | 1453 | 1212 | 1202 | 10 | 241 | 9 | 2355 | 580 | 19/19 | 0.834 | 0.416 |
| 10 | 1524 | 1245 | 1239 | 6 | 279 | 9 | 2407 | 615 | 23/23 | 0.817 | 0.454 |
| 11 | 1463 | 1234 | 1229 | 5 | 229 | 6 | 2353 | 536 | 23/23 | 0.843 | 0.427 |
| 12 | 1576 | 1357 | 1355 | 2 | 219 | 6 | 2310 | 537 | 27/27 | 0.861 | 0.408 |
| 13 | 1568 | 1328 | 1324 | 4 | 240 | 8 | 2323 | 519 | 27/27 | 0.847 | 0.462 |
| 14 | 1528 | 1295 | 1295 | 0 | 233 | 5 | 2316 | 538 | 27/27 | 0.848 | 0.433 |
| 15 | 1671 | 1448 | 1445 | 3 | 223 | 6 | 2299 | 568 | 29/29 | 0.867 | 0.393 |
| 16 | 1647 | 1432 | 1428 | 4 | 215 | 6 | 2286 | 548 | 35/35 | 0.869 | 0.392 |
| 17 | 1712 | 1468 | 1465 | 3 | 244 | 6 | 2281 | 592 | 35/35 | 0.857 | 0.412 |
| 18 | 1773 | 1558 | 1555 | 3 | 215 | 4 | 2247 | 576 | 41/41 | 0.879 | 0.373 |
| 19 | 1699 | 1459 | 1458 | 1 | 240 | 5 | 2279 | 570 | 41/41 | 0.859 | 0.421 |
| 20 | 1805 | 1581 | 1577 | 4 | 224 | 5 | 2243 | 578 | 47/47 | 0.876 | 0.388 |
| 21 | 1801 | 1579 | 1578 | 1 | 222 | 4 | 2223 | 566 | 47/47 | 0.877 | 0.392 |
| 22 | 1749 | 1533 | 1532 | 1 | 216 | 2 | 2251 | 560 | 47/47 | 0.877 | 0.386 |
| 23 | 1742 | 1534 | 1529 | 5 | 208 | 5 | 2214 | 545 | 47/47 | 0.881 | 0.382 |
| 24 | 1805 | 1592 | 1591 | 1 | 213 | 4 | 2209 | 557 | 51/51 | 0.882 | 0.382 |
| 25 | 1835 | 1631 | 1629 | 2 | 204 | 3 | 2230 | 542 | 57/57 | 0.889 | 0.376 |
| 26 | 1804 | 1594 | 1592 | 2 | 210 | 5 | 2215 | 547 | 57/57 | 0.884 | 0.384 |
| 27 | 1902 | 1680 | 1679 | 1 | 222 | 4 | 2207 | 576 | 61/63 | 0.883 | 0.385 |
| 28 | 1868 | 1666 | 1665 | 1 | 202 | 2 | 2205 | 561 | 63/63 | 0.892 | 0.360 |
| 29 | 1866 | 1646 | 1641 | 5 | 220 | 4 | 2179 | 578 | 63/63 | 0.882 | 0.381 |
| 30 | 1937 | 1718 | 1718 | 0 | 219 | 2 | 2200 | 576 | 69/69 | 0.887 | 0.380 |
| 31 | 1886 | 1687 | 1684 | 3 | 199 | 4 | 2144 | 553 | 69/69 | 0.894 | 0.360 |
| 32 | 1959 | 1747 | 1747 | 0 | 212 | 3 | 2159 | 561 | 78/79 | 0.892 | 0.378 |
| 33 | 1905 | 1685 | 1682 | 3 | 220 | 3 | 2193 | 555 | 79/79 | 0.885 | 0.396 |
| 34 | 1893 | 1672 | 1670 | 2 | 221 | 4 | 2164 | 551 | 79/79 | 0.883 | 0.401 |
| 35 | 1897 | 1695 | 1695 | 0 | 202 | 2 | 2136 | 536 | 78/79 | 0.894 | 0.377 |
| 36 | 1955 | 1736 | 1736 | 0 | 219 | 2 | 2180 | 568 | 81/83 | 0.888 | 0.386 |
| 37 | 1941 | 1728 | 1727 | 1 | 213 | 2 | 2152 | 560 | 82/83 | 0.890 | 0.380 |
| 38 | 1872 | 1667 | 1666 | 1 | 205 | 2 | 2162 | 550 | 82/83 | 0.890 | 0.373 |
| 39 | 1891 | 1674 | 1672 | 2 | 217 | 4 | 2174 | 560 | 82/83 | 0.885 | 0.388 |
| 40 | 1937 | 1753 | 1753 | 0 | 184 | 2 | 2113 | 520 | 87/89 | 0.905 | 0.354 |
| 41 | 1939 | 1733 | 1731 | 2 | 206 | 3 | 2131 | 534 | 88/89 | 0.894 | 0.386 |
| 42 | 1950 | 1748 | 1747 | 1 | 202 | 2 | 2150 | 542 | 87/89 | 0.896 | 0.373 |
| 43 | 1968 | 1768 | 1767 | 1 | 200 | 3 | 2101 | 566 | 88/89 | 0.898 | 0.353 |
| 44 | 1916 | 1706 | 1706 | 0 | 210 | 2 | 2111 | 533 | 88/89 | 0.890 | 0.394 |
| 45 | 2005 | 1788 | 1787 | 1 | 217 | 1 | 2146 | 583 | 88/91 | 0.892 | 0.372 |
| 46 | 1959 | 1753 | 1750 | 3 | 206 | 3 | 2115 | 548 | 91/91 | 0.895 | 0.376 |
| 47 | 1938 | 1763 | 1763 | 0 | 175 | 2 | 2123 | 533 | 90/91 | 0.910 | 0.328 |
| 48 | 1916 | 1716 | 1716 | 0 | 200 | 1 | 2119 | 558 | 94/95 | 0.896 | 0.358 |
| 49 | 1916 | 1719 | 1715 | 4 | 197 | 4 | 2108 | 550 | 93/95 | 0.897 | 0.358 |
| 50 | 1988 | 1774 | 1774 | 0 | 214 | 1 | 2124 | 562 | 103/111 | 0.892 | 0.381 |
| 51 | 1921 | 1728 | 1728 | 0 | 193 | 2 | 2097 | 530 | 103/111 | 0.900 | 0.364 |
| 52 | 1983 | 1780 | 1777 | 3 | 203 | 3 | 2075 | 540 | 107/111 | 0.898 | 0.376 |
| 53 | 1882 | 1690 | 1688 | 2 | 192 | 3 | 2089 | 504 | 102/111 | 0.898 | 0.381 |
| 54 | 1975 | 1773 | 1770 | 3 | 202 | 2 | 2094 | 553 | 118/123 | 0.898 | 0.365 |
| 55 | 1992 | 1791 | 1790 | 1 | 201 | 2 | 2119 | 546 | 109/123 | 0.899 | 0.368 |
| 56 | 1947 | 1747 | 1747 | 0 | 200 | 1 | 2084 | 520 | 113/123 | 0.897 | 0.385 |
| 57 | 1976 | 1777 | 1777 | 0 | 199 | 1 | 2065 | 536 | 117/123 | 0.899 | 0.371 |
| 58 | 1927 | 1739 | 1738 | 1 | 188 | 2 | 2069 | 524 | 113/123 | 0.902 | 0.359 |
| 59 | 1963 | 1768 | 1767 | 1 | 195 | 2 | 2101 | 530 | 118/123 | 0.901 | 0.368 |
| 60 | 1935 | 1733 | 1732 | 1 | 202 | 1 | 2094 | 534 | 120/125 | 0.896 | 0.378 |

Burnt fuelled families per turn, m <= 12 (family: count): m=1: {}; m=2: {}; m=3: {(1,3) 186, (3,1) 226}; m=4: {(1,3) 211, (2,4) 91, (3,1) 230, (4,2) 84}; m=5: {(1,3) 201, (1,5) 75, (2,4) 80, (3,1) 208, (3,5) 69, (4,2) 88, (5,1) 81, (5,3) 60}; m=6: {(1,3) 186, (1,5) 79, (2,4) 84, (3,1) 199, (3,5) 64, (4,2) 76, (4,6) 47, (5,1) 90, (5,3) 54, (6,4) 68}; m=7: {(1,3) 203, (1,5) 69, (2,4) 81, (3,1) 206, (3,5) 56, (4,2) 83, (4,6) 54, (5,1) 87, (5,3) 59, (6,4) 63}; m=8: {(1,3) 184, (1,5) 71, (2,4) 70, (2,8) 36, (3,1) 179, (3,5) 50, (4,2) 78, (4,6) 55, (5,1) 76, (5,3) 66, (6,4) 53, (6,8) 33, (8,2) 41, (8,6) 32}; m=9: {(1,3) 193, (1,5) 86, (1,9) 60, (2,4) 79, (2,8) 41, (3,1) 175, (3,5) 51, (4,2) 66, (4,6) 56, (5,1) 80, (5,3) 59, (5,9) 15, (6,4) 63, (6,8) 33, (8,2) 35, (8,6) 26, (9,1) 62, (9,5) 22}; m=10: {(1,3) 195, (1,5) 78, (1,9) 63, (2,4) 63, (2,8) 28, (3,1) 182, (3,5) 56, (4,2) 80, (4,6) 47, (4,10) 26, (5,1) 74, (5,3) 44, (5,9) 19, (6,4) 52, (6,8) 32, (8,2) 40, (8,6) 25, (8,10) 9, (9,1) 71, (9,5) 21, (10,4) 21, (10,8) 13}; m=11: {(1,3) 168, (1,5) 79, (1,9) 60, (2,4) 69, (2,8) 41, (3,1) 181, (3,5) 52, (4,2) 70, (4,6) 53, (4,10) 27, (5,1) 70, (5,3) 56, (5,9) 18, (6,4) 55, (6,8) 30, (8,2) 35, (8,6) 35, (8,10) 13, (9,1) 68, (9,5) 12, (10,4) 21, (10,8) 16}; m=12: {(1,3) 173, (1,5) 81, (1,9) 64, (2,4) 71, (2,8) 36, (2,12) 60, (3,1) 180, (3,5) 48, (4,2) 70, (4,6) 53, (4,10) 22, (5,1) 71, (5,3) 59, (5,9) 24, (6,4) 46, (6,8) 23, (8,2) 29, (8,6) 21, (8,10) 9, (9,1) 68, (9,5) 25, (10,4) 20, (10,8) 15, (10,12) 15, (12,2) 57, (12,10) 15};

#### Ledger q = 7, Q = 30000 (turn m = (mQ, (m + 1)Q]; T total charges, B burnt = B_f fuelled-both + B_e ember-carrying, P pure = twins, E embers, pi primes, A primes with P + 2 open, fam/adm = fuelled families present / admissible with max <= m)

| m | T | B | B_f | B_e | P | E | pi | A | fam/adm | B/T | P/A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 356 | 12 | 0 | 12 | 344 | 114 | 2812 | 349 | 1/1 | 0.034 | 0.986 |
| 2 | 341 | 36 | 0 | 36 | 305 | 76 | 2656 | 306 | 1/1 | 0.106 | 0.997 |
| 3 | 749 | 441 | 412 | 29 | 308 | 57 | 2588 | 496 | 3/3 | 0.589 | 0.621 |
| 4 | 914 | 637 | 616 | 21 | 277 | 46 | 2547 | 488 | 5/5 | 0.697 | 0.568 |
| 5 | 1162 | 886 | 862 | 24 | 276 | 42 | 2494 | 554 | 9/9 | 0.762 | 0.498 |
| 6 | 1240 | 964 | 947 | 17 | 276 | 34 | 2465 | 544 | 11/11 | 0.777 | 0.507 |
| 7 | 1442 | 1192 | 1178 | 14 | 250 | 31 | 2414 | 584 | 17/17 | 0.827 | 0.428 |
| 8 | 1516 | 1266 | 1242 | 24 | 250 | 29 | 2421 | 564 | 21/21 | 0.835 | 0.443 |
| 9 | 1710 | 1469 | 1451 | 18 | 241 | 24 | 2355 | 641 | 27/27 | 0.859 | 0.376 |
| 10 | 1764 | 1485 | 1468 | 17 | 279 | 26 | 2407 | 656 | 31/31 | 0.842 | 0.425 |
| 11 | 1706 | 1477 | 1459 | 18 | 229 | 22 | 2353 | 591 | 31/31 | 0.866 | 0.387 |
| 12 | 1806 | 1587 | 1575 | 12 | 219 | 19 | 2310 | 579 | 35/35 | 0.879 | 0.378 |
| 13 | 1791 | 1551 | 1543 | 8 | 240 | 20 | 2323 | 563 | 35/35 | 0.866 | 0.426 |
| 14 | 1839 | 1606 | 1598 | 8 | 233 | 18 | 2316 | 588 | 41/41 | 0.873 | 0.396 |
| 15 | 1990 | 1767 | 1755 | 12 | 223 | 18 | 2299 | 617 | 45/45 | 0.888 | 0.361 |
| 16 | 1971 | 1756 | 1742 | 14 | 215 | 17 | 2286 | 599 | 53/53 | 0.891 | 0.359 |
| 17 | 2016 | 1772 | 1761 | 11 | 244 | 16 | 2281 | 637 | 53/53 | 0.879 | 0.383 |
| 18 | 2068 | 1853 | 1845 | 8 | 215 | 14 | 2247 | 625 | 59/59 | 0.896 | 0.344 |
| 19 | 2002 | 1762 | 1750 | 12 | 240 | 15 | 2279 | 609 | 59/59 | 0.880 | 0.394 |
| 20 | 2104 | 1880 | 1872 | 8 | 224 | 15 | 2243 | 617 | 67/67 | 0.894 | 0.363 |
| 21 | 2171 | 1949 | 1943 | 6 | 222 | 13 | 2223 | 639 | 71/71 | 0.898 | 0.347 |
| 22 | 2149 | 1933 | 1925 | 8 | 216 | 11 | 2251 | 631 | 71/71 | 0.899 | 0.342 |
| 23 | 2141 | 1933 | 1925 | 8 | 208 | 14 | 2214 | 632 | 71/71 | 0.903 | 0.329 |
| 24 | 2191 | 1978 | 1973 | 5 | 213 | 9 | 2209 | 630 | 77/77 | 0.903 | 0.338 |
| 25 | 2215 | 2011 | 2001 | 10 | 204 | 14 | 2230 | 619 | 87/87 | 0.908 | 0.330 |
| 26 | 2171 | 1961 | 1952 | 9 | 210 | 12 | 2215 | 625 | 86/87 | 0.903 | 0.336 |
| 27 | 2289 | 2067 | 2062 | 5 | 222 | 11 | 2207 | 641 | 93/95 | 0.903 | 0.346 |
| 28 | 2329 | 2127 | 2123 | 4 | 202 | 9 | 2205 | 636 | 103/103 | 0.913 | 0.318 |
| 29 | 2343 | 2123 | 2111 | 12 | 220 | 11 | 2179 | 653 | 103/103 | 0.906 | 0.337 |
| 30 | 2383 | 2164 | 2160 | 4 | 219 | 11 | 2200 | 653 | 111/111 | 0.908 | 0.335 |
| 31 | 2334 | 2135 | 2129 | 6 | 199 | 8 | 2144 | 620 | 111/111 | 0.915 | 0.321 |
| 32 | 2390 | 2178 | 2172 | 6 | 212 | 11 | 2159 | 635 | 121/123 | 0.911 | 0.334 |
| 33 | 2328 | 2108 | 2100 | 8 | 220 | 9 | 2193 | 617 | 122/123 | 0.905 | 0.357 |
| 34 | 2364 | 2143 | 2136 | 7 | 221 | 9 | 2164 | 631 | 123/123 | 0.907 | 0.350 |
| 35 | 2413 | 2211 | 2206 | 5 | 202 | 10 | 2136 | 624 | 129/131 | 0.916 | 0.324 |
| 36 | 2468 | 2249 | 2248 | 1 | 219 | 8 | 2180 | 656 | 135/137 | 0.911 | 0.334 |
| 37 | 2436 | 2223 | 2220 | 3 | 213 | 6 | 2152 | 634 | 135/137 | 0.913 | 0.336 |
| 38 | 2380 | 2175 | 2169 | 6 | 205 | 9 | 2162 | 630 | 134/137 | 0.914 | 0.325 |
| 39 | 2383 | 2166 | 2160 | 6 | 217 | 9 | 2174 | 631 | 135/137 | 0.909 | 0.344 |
| 40 | 2430 | 2246 | 2242 | 4 | 184 | 9 | 2113 | 608 | 140/145 | 0.924 | 0.303 |
| 41 | 2441 | 2235 | 2230 | 5 | 206 | 8 | 2131 | 620 | 144/145 | 0.916 | 0.332 |
| 42 | 2475 | 2273 | 2268 | 5 | 202 | 7 | 2150 | 618 | 150/157 | 0.918 | 0.327 |
| 43 | 2504 | 2304 | 2302 | 2 | 200 | 8 | 2101 | 650 | 156/157 | 0.920 | 0.308 |
| 44 | 2438 | 2228 | 2226 | 2 | 210 | 6 | 2111 | 615 | 151/157 | 0.914 | 0.341 |
| 45 | 2533 | 2316 | 2311 | 5 | 217 | 8 | 2146 | 674 | 155/161 | 0.914 | 0.322 |
| 46 | 2482 | 2276 | 2268 | 8 | 206 | 7 | 2115 | 626 | 158/161 | 0.917 | 0.329 |
| 47 | 2484 | 2309 | 2307 | 2 | 175 | 7 | 2123 | 614 | 157/161 | 0.930 | 0.285 |
| 48 | 2473 | 2273 | 2270 | 3 | 200 | 5 | 2119 | 656 | 163/167 | 0.919 | 0.305 |
| 49 | 2501 | 2304 | 2300 | 4 | 197 | 6 | 2108 | 634 | 173/183 | 0.921 | 0.311 |
| 50 | 2555 | 2341 | 2337 | 4 | 214 | 7 | 2124 | 642 | 187/201 | 0.916 | 0.333 |
| 51 | 2480 | 2287 | 2283 | 4 | 193 | 8 | 2097 | 622 | 187/201 | 0.922 | 0.310 |
| 52 | 2567 | 2364 | 2360 | 4 | 203 | 7 | 2075 | 625 | 193/201 | 0.921 | 0.325 |
| 53 | 2454 | 2262 | 2259 | 3 | 192 | 7 | 2089 | 587 | 184/201 | 0.922 | 0.327 |
| 54 | 2569 | 2367 | 2361 | 6 | 202 | 7 | 2094 | 639 | 204/215 | 0.921 | 0.316 |
| 55 | 2538 | 2337 | 2334 | 3 | 201 | 6 | 2119 | 631 | 195/215 | 0.921 | 0.319 |
| 56 | 2566 | 2366 | 2365 | 1 | 200 | 4 | 2084 | 606 | 207/229 | 0.922 | 0.330 |
| 57 | 2556 | 2357 | 2355 | 2 | 199 | 5 | 2065 | 614 | 208/229 | 0.922 | 0.324 |
| 58 | 2511 | 2323 | 2319 | 4 | 188 | 8 | 2069 | 606 | 205/229 | 0.925 | 0.310 |
| 59 | 2554 | 2359 | 2355 | 4 | 195 | 6 | 2101 | 614 | 212/229 | 0.924 | 0.318 |
| 60 | 2527 | 2325 | 2323 | 2 | 202 | 5 | 2094 | 618 | 220/233 | 0.920 | 0.327 |

Burnt fuelled families per turn, m <= 12 (family: count): m=1: {}; m=2: {}; m=3: {(1,3) 186, (3,1) 226}; m=4: {(1,3) 211, (2,4) 91, (3,1) 230, (4,2) 84}; m=5: {(1,3) 201, (1,5) 75, (2,4) 80, (3,1) 208, (3,5) 69, (4,2) 88, (5,1) 81, (5,3) 60}; m=6: {(1,3) 186, (1,5) 79, (2,4) 84, (3,1) 199, (3,5) 64, (4,2) 76, (4,6) 47, (5,1) 90, (5,3) 54, (6,4) 68}; m=7: {(1,3) 203, (1,5) 69, (1,7) 61, (2,4) 81, (3,1) 206, (3,5) 56, (3,7) 41, (4,2) 83, (4,6) 54, (5,1) 87, (5,3) 59, (5,7) 16, (6,4) 63, (7,1) 49, (7,3) 37, (7,5) 13}; m=8: {(1,3) 184, (1,5) 71, (1,7) 59, (2,4) 70, (2,8) 36, (3,1) 179, (3,5) 50, (3,7) 38, (4,2) 78, (4,6) 55, (5,1) 76, (5,3) 66, (5,7) 17, (6,4) 53, (6,8) 33, (7,1) 47, (7,3) 39, (7,5) 18, (8,2) 41, (8,6) 32}; m=9: {(1,3) 193, (1,5) 86, (1,7) 60, (1,9) 60, (2,4) 79, (2,8) 41, (3,1) 175, (3,5) 51, (3,7) 40, (4,2) 66, (4,6) 56, (5,1) 80, (5,3) 59, (5,7) 10, (5,9) 15, (6,4) 63, (6,8) 33, (7,1) 47, (7,3) 46, (7,5) 19, (7,9) 15, (8,2) 35, (8,6) 26, (9,1) 62, (9,5) 22, (9,7) 12}; m=10: {(1,3) 195, (1,5) 78, (1,7) 41, (1,9) 63, (2,4) 63, (2,8) 28, (3,1) 182, (3,5) 56, (3,7) 38, (4,2) 80, (4,6) 47, (4,10) 26, (5,1) 74, (5,3) 44, (5,7) 14, (5,9) 19, (6,4) 52, (6,8) 32, (7,1) 48, (7,3) 44, (7,5) 18, (7,9) 15, (8,2) 40, (8,6) 25, (8,10) 9, (9,1) 71, (9,5) 21, (9,7) 11, (10,4) 21, (10,8) 13}; m=11: {(1,3) 168, (1,5) 79, (1,7) 54, (1,9) 60, (2,4) 69, (2,8) 41, (3,1) 181, (3,5) 52, (3,7) 35, (4,2) 70, (4,6) 53, (4,10) 27, (5,1) 70, (5,3) 56, (5,7) 16, (5,9) 18, (6,4) 55, (6,8) 30, (7,1) 56, (7,3) 34, (7,5) 15, (7,9) 10, (8,2) 35, (8,6) 35, (8,10) 13, (9,1) 68, (9,5) 12, (9,7) 10, (10,4) 21, (10,8) 16}; m=12: {(1,3) 173, (1,5) 81, (1,7) 42, (1,9) 64, (2,4) 71, (2,8) 36, (2,12) 60, (3,1) 180, (3,5) 48, (3,7) 26, (4,2) 70, (4,6) 53, (4,10) 22, (5,1) 71, (5,3) 59, (5,7) 14, (5,9) 24, (6,4) 46, (6,8) 23, (7,1) 57, (7,3) 34, (7,5) 14, (7,9) 16, (8,2) 29, (8,6) 21, (8,10) 9, (9,1) 68, (9,5) 25, (9,7) 17, (10,4) 20, (10,8) 15, (10,12) 15, (12,2) 57, (12,10) 15};

#### Ledger q = 11, Q = 30000 (turn m = (mQ, (m + 1)Q]; T total charges, B burnt = B_f fuelled-both + B_e ember-carrying, P pure = twins, E embers, pi primes, A primes with P + 2 open, fam/adm = fuelled families present / admissible with max <= m)

| m | T | B | B_f | B_e | P | E | pi | A | fam/adm | B/T | P/A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 370 | 26 | 0 | 26 | 344 | 218 | 2812 | 358 | 1/1 | 0.070 | 0.961 |
| 2 | 374 | 69 | 0 | 69 | 305 | 149 | 2656 | 309 | 1/1 | 0.184 | 0.987 |
| 3 | 778 | 470 | 412 | 58 | 308 | 114 | 2588 | 502 | 3/3 | 0.604 | 0.614 |
| 4 | 934 | 657 | 616 | 41 | 277 | 96 | 2547 | 490 | 5/5 | 0.703 | 0.565 |
| 5 | 1186 | 910 | 862 | 48 | 276 | 84 | 2494 | 556 | 9/9 | 0.767 | 0.496 |
| 6 | 1265 | 989 | 947 | 42 | 276 | 71 | 2465 | 545 | 11/11 | 0.782 | 0.506 |
| 7 | 1464 | 1214 | 1178 | 36 | 250 | 65 | 2414 | 585 | 17/17 | 0.829 | 0.427 |
| 8 | 1535 | 1285 | 1242 | 43 | 250 | 60 | 2421 | 566 | 21/21 | 0.837 | 0.442 |
| 9 | 1729 | 1488 | 1451 | 37 | 241 | 53 | 2355 | 643 | 27/27 | 0.861 | 0.375 |
| 10 | 1778 | 1499 | 1468 | 31 | 279 | 52 | 2407 | 657 | 31/31 | 0.843 | 0.425 |
| 11 | 1879 | 1650 | 1614 | 36 | 229 | 47 | 2353 | 619 | 41/41 | 0.878 | 0.370 |
| 12 | 1988 | 1769 | 1742 | 27 | 219 | 43 | 2310 | 622 | 45/45 | 0.890 | 0.352 |
| 13 | 1941 | 1701 | 1677 | 24 | 240 | 43 | 2323 | 594 | 45/45 | 0.876 | 0.404 |
| 14 | 2000 | 1767 | 1747 | 20 | 233 | 39 | 2316 | 622 | 51/51 | 0.883 | 0.375 |
| 15 | 2165 | 1942 | 1916 | 26 | 223 | 37 | 2299 | 643 | 57/57 | 0.897 | 0.347 |
| 16 | 2144 | 1929 | 1905 | 24 | 215 | 37 | 2286 | 631 | 65/65 | 0.900 | 0.341 |
| 17 | 2190 | 1946 | 1916 | 30 | 244 | 36 | 2281 | 665 | 65/65 | 0.889 | 0.367 |
| 18 | 2227 | 2012 | 1993 | 19 | 215 | 30 | 2247 | 649 | 71/71 | 0.903 | 0.331 |
| 19 | 2168 | 1928 | 1902 | 26 | 240 | 32 | 2279 | 643 | 71/71 | 0.889 | 0.373 |
| 20 | 2270 | 2046 | 2029 | 17 | 224 | 31 | 2243 | 649 | 79/79 | 0.901 | 0.345 |
| 21 | 2342 | 2120 | 2098 | 22 | 222 | 30 | 2223 | 662 | 85/85 | 0.905 | 0.335 |
| 22 | 2351 | 2135 | 2114 | 21 | 216 | 28 | 2251 | 661 | 95/95 | 0.908 | 0.327 |
| 23 | 2341 | 2133 | 2117 | 16 | 208 | 27 | 2214 | 660 | 94/95 | 0.911 | 0.315 |
| 24 | 2409 | 2196 | 2179 | 17 | 213 | 26 | 2209 | 658 | 102/103 | 0.912 | 0.324 |
| 25 | 2409 | 2205 | 2188 | 17 | 204 | 26 | 2230 | 643 | 115/115 | 0.915 | 0.317 |
| 26 | 2417 | 2207 | 2187 | 20 | 210 | 26 | 2215 | 659 | 113/115 | 0.913 | 0.319 |
| 27 | 2517 | 2295 | 2278 | 17 | 222 | 24 | 2207 | 664 | 123/125 | 0.912 | 0.334 |
| 28 | 2539 | 2337 | 2322 | 15 | 202 | 23 | 2205 | 661 | 135/135 | 0.920 | 0.306 |
| 29 | 2564 | 2344 | 2323 | 21 | 220 | 23 | 2179 | 685 | 134/135 | 0.914 | 0.321 |
| 30 | 2590 | 2371 | 2357 | 14 | 219 | 23 | 2200 | 675 | 142/143 | 0.915 | 0.324 |
| 31 | 2566 | 2367 | 2353 | 14 | 199 | 20 | 2144 | 649 | 143/143 | 0.922 | 0.307 |
| 32 | 2618 | 2406 | 2384 | 22 | 212 | 24 | 2159 | 661 | 155/157 | 0.919 | 0.321 |
| 33 | 2610 | 2390 | 2374 | 16 | 220 | 20 | 2193 | 671 | 162/165 | 0.916 | 0.328 |
| 34 | 2647 | 2426 | 2410 | 16 | 221 | 20 | 2164 | 675 | 163/165 | 0.917 | 0.327 |
| 35 | 2675 | 2473 | 2463 | 10 | 202 | 21 | 2136 | 663 | 173/177 | 0.924 | 0.305 |
| 36 | 2736 | 2517 | 2506 | 11 | 219 | 20 | 2180 | 697 | 178/185 | 0.920 | 0.314 |
| 37 | 2716 | 2503 | 2495 | 8 | 213 | 16 | 2152 | 691 | 179/185 | 0.922 | 0.308 |
| 38 | 2679 | 2474 | 2463 | 11 | 205 | 18 | 2162 | 670 | 176/185 | 0.923 | 0.306 |
| 39 | 2626 | 2409 | 2397 | 12 | 217 | 21 | 2174 | 670 | 180/185 | 0.917 | 0.324 |
| 40 | 2713 | 2529 | 2517 | 12 | 184 | 19 | 2113 | 656 | 185/195 | 0.932 | 0.280 |
| 41 | 2715 | 2509 | 2501 | 8 | 206 | 17 | 2131 | 669 | 191/195 | 0.924 | 0.308 |
| 42 | 2742 | 2540 | 2526 | 14 | 202 | 15 | 2150 | 658 | 193/207 | 0.926 | 0.307 |
| 43 | 2756 | 2556 | 2545 | 11 | 200 | 19 | 2101 | 687 | 202/207 | 0.927 | 0.291 |
| 44 | 2731 | 2521 | 2511 | 10 | 210 | 16 | 2111 | 651 | 209/221 | 0.923 | 0.323 |
| 45 | 2835 | 2618 | 2605 | 13 | 217 | 18 | 2146 | 718 | 214/227 | 0.923 | 0.302 |
| 46 | 2782 | 2576 | 2562 | 14 | 206 | 14 | 2115 | 668 | 214/227 | 0.926 | 0.308 |
| 47 | 2773 | 2598 | 2590 | 8 | 175 | 15 | 2123 | 655 | 213/227 | 0.937 | 0.267 |
| 48 | 2790 | 2590 | 2582 | 8 | 200 | 16 | 2119 | 699 | 229/235 | 0.928 | 0.286 |
| 49 | 2815 | 2618 | 2606 | 12 | 197 | 16 | 2108 | 674 | 236/255 | 0.930 | 0.292 |
| 50 | 2864 | 2650 | 2641 | 9 | 214 | 15 | 2124 | 684 | 250/275 | 0.925 | 0.313 |
| 51 | 2797 | 2604 | 2596 | 8 | 193 | 16 | 2097 | 661 | 254/275 | 0.931 | 0.292 |
| 52 | 2880 | 2677 | 2667 | 10 | 203 | 16 | 2075 | 660 | 259/275 | 0.930 | 0.308 |
| 53 | 2760 | 2568 | 2561 | 7 | 192 | 14 | 2089 | 630 | 249/275 | 0.930 | 0.305 |
| 54 | 2883 | 2681 | 2671 | 10 | 202 | 15 | 2094 | 689 | 266/291 | 0.930 | 0.293 |
| 55 | 2897 | 2696 | 2686 | 10 | 201 | 14 | 2119 | 685 | 269/305 | 0.931 | 0.293 |
| 56 | 2955 | 2755 | 2747 | 8 | 200 | 12 | 2084 | 663 | 287/321 | 0.932 | 0.302 |
| 57 | 2892 | 2693 | 2683 | 10 | 199 | 13 | 2065 | 664 | 284/321 | 0.931 | 0.300 |
| 58 | 2864 | 2676 | 2670 | 6 | 188 | 15 | 2069 | 649 | 280/321 | 0.934 | 0.290 |
| 59 | 2904 | 2709 | 2697 | 12 | 195 | 13 | 2101 | 669 | 291/321 | 0.933 | 0.291 |
| 60 | 2877 | 2675 | 2668 | 7 | 202 | 13 | 2094 | 677 | 300/327 | 0.930 | 0.298 |

Burnt fuelled families per turn, m <= 12 (family: count): m=1: {}; m=2: {}; m=3: {(1,3) 186, (3,1) 226}; m=4: {(1,3) 211, (2,4) 91, (3,1) 230, (4,2) 84}; m=5: {(1,3) 201, (1,5) 75, (2,4) 80, (3,1) 208, (3,5) 69, (4,2) 88, (5,1) 81, (5,3) 60}; m=6: {(1,3) 186, (1,5) 79, (2,4) 84, (3,1) 199, (3,5) 64, (4,2) 76, (4,6) 47, (5,1) 90, (5,3) 54, (6,4) 68}; m=7: {(1,3) 203, (1,5) 69, (1,7) 61, (2,4) 81, (3,1) 206, (3,5) 56, (3,7) 41, (4,2) 83, (4,6) 54, (5,1) 87, (5,3) 59, (5,7) 16, (6,4) 63, (7,1) 49, (7,3) 37, (7,5) 13}; m=8: {(1,3) 184, (1,5) 71, (1,7) 59, (2,4) 70, (2,8) 36, (3,1) 179, (3,5) 50, (3,7) 38, (4,2) 78, (4,6) 55, (5,1) 76, (5,3) 66, (5,7) 17, (6,4) 53, (6,8) 33, (7,1) 47, (7,3) 39, (7,5) 18, (8,2) 41, (8,6) 32}; m=9: {(1,3) 193, (1,5) 86, (1,7) 60, (1,9) 60, (2,4) 79, (2,8) 41, (3,1) 175, (3,5) 51, (3,7) 40, (4,2) 66, (4,6) 56, (5,1) 80, (5,3) 59, (5,7) 10, (5,9) 15, (6,4) 63, (6,8) 33, (7,1) 47, (7,3) 46, (7,5) 19, (7,9) 15, (8,2) 35, (8,6) 26, (9,1) 62, (9,5) 22, (9,7) 12}; m=10: {(1,3) 195, (1,5) 78, (1,7) 41, (1,9) 63, (2,4) 63, (2,8) 28, (3,1) 182, (3,5) 56, (3,7) 38, (4,2) 80, (4,6) 47, (4,10) 26, (5,1) 74, (5,3) 44, (5,7) 14, (5,9) 19, (6,4) 52, (6,8) 32, (7,1) 48, (7,3) 44, (7,5) 18, (7,9) 15, (8,2) 40, (8,6) 25, (8,10) 9, (9,1) 71, (9,5) 21, (9,7) 11, (10,4) 21, (10,8) 13}; m=11: {(1,3) 168, (1,5) 79, (1,7) 54, (1,9) 60, (1,11) 28, (2,4) 69, (2,8) 41, (3,1) 181, (3,5) 52, (3,7) 35, (3,11) 20, (4,2) 70, (4,6) 53, (4,10) 27, (5,1) 70, (5,3) 56, (5,7) 16, (5,9) 18, (5,11) 13, (6,4) 55, (6,8) 30, (7,1) 56, (7,3) 34, (7,5) 15, (7,9) 10, (7,11) 6, (8,2) 35, (8,6) 35, (8,10) 13, (9,1) 68, (9,5) 12, (9,7) 10, (9,11) 10, (10,4) 21, (10,8) 16, (11,1) 32, (11,3) 27, (11,5) 10, (11,7) 5, (11,9) 4}; m=12: {(1,3) 173, (1,5) 81, (1,7) 42, (1,9) 64, (1,11) 42, (2,4) 71, (2,8) 36, (2,12) 60, (3,1) 180, (3,5) 48, (3,7) 26, (3,11) 26, (4,2) 70, (4,6) 53, (4,10) 22, (5,1) 71, (5,3) 59, (5,7) 14, (5,9) 24, (5,11) 10, (6,4) 46, (6,8) 23, (7,1) 57, (7,3) 34, (7,5) 14, (7,9) 16, (7,11) 3, (8,2) 29, (8,6) 21, (8,10) 9, (9,1) 68, (9,5) 25, (9,7) 17, (9,11) 6, (10,4) 20, (10,8) 15, (10,12) 15, (11,1) 36, (11,3) 24, (11,5) 7, (11,7) 4, (11,9) 9, (12,2) 57, (12,10) 15};

#### Ledger q = 5, Q = 100000 (turn m = (mQ, (m + 1)Q]; T total charges, B burnt = B_f fuelled-both + B_e ember-carrying, P pure = twins, E embers, pi primes, A primes with P + 2 open, fam/adm = fuelled families present / admissible with max <= m)

| m | T | B | B_f | B_e | P | E | pi | A | fam/adm | B/T | P/A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 938 | 2 | 0 | 2 | 936 | 52 | 8392 | 937 | 1/1 | 0.002 | 0.999 |
| 2 | 852 | 18 | 0 | 18 | 834 | 32 | 8013 | 834 | 1/1 | 0.021 | 1.000 |
| 3 | 2007 | 1197 | 1184 | 13 | 810 | 25 | 7863 | 1395 | 3/3 | 0.596 | 0.581 |
| 4 | 2298 | 1537 | 1530 | 7 | 761 | 20 | 7678 | 1275 | 5/5 | 0.669 | 0.597 |
| 5 | 3094 | 2328 | 2323 | 5 | 766 | 16 | 7560 | 1523 | 9/9 | 0.752 | 0.503 |
| 6 | 3361 | 2631 | 2622 | 9 | 730 | 13 | 7445 | 1529 | 11/11 | 0.783 | 0.477 |
| 7 | 3210 | 2505 | 2500 | 5 | 705 | 14 | 7408 | 1445 | 11/11 | 0.780 | 0.488 |
| 8 | 3559 | 2853 | 2845 | 8 | 706 | 11 | 7323 | 1427 | 15/15 | 0.802 | 0.495 |
| 9 | 4047 | 3350 | 3345 | 5 | 697 | 11 | 7224 | 1581 | 19/19 | 0.828 | 0.441 |
| 10 | 4110 | 3385 | 3382 | 3 | 725 | 8 | 7216 | 1586 | 23/23 | 0.824 | 0.457 |
| 11 | 3969 | 3264 | 3260 | 4 | 705 | 9 | 7224 | 1588 | 23/23 | 0.822 | 0.444 |
| 12 | 4384 | 3732 | 3729 | 3 | 652 | 8 | 7083 | 1533 | 27/27 | 0.851 | 0.425 |
| 13 | 4438 | 3737 | 3734 | 3 | 701 | 7 | 7105 | 1557 | 27/27 | 0.842 | 0.450 |
| 14 | 4208 | 3564 | 3558 | 6 | 644 | 8 | 7029 | 1502 | 27/27 | 0.847 | 0.429 |
| 15 | 4507 | 3843 | 3839 | 4 | 664 | 8 | 6972 | 1639 | 29/29 | 0.853 | 0.405 |
| 16 | 4683 | 4012 | 4008 | 4 | 671 | 6 | 7014 | 1633 | 35/35 | 0.857 | 0.411 |
| 17 | 4675 | 4023 | 4021 | 2 | 652 | 5 | 6931 | 1596 | 35/35 | 0.861 | 0.409 |
| 18 | 4754 | 4097 | 4093 | 4 | 657 | 5 | 6957 | 1605 | 41/41 | 0.862 | 0.409 |
| 19 | 4765 | 4134 | 4130 | 4 | 631 | 7 | 6904 | 1571 | 41/41 | 0.868 | 0.402 |
| 20 | 4961 | 4317 | 4315 | 2 | 644 | 5 | 6872 | 1604 | 47/47 | 0.870 | 0.401 |
| 21 | 4962 | 4360 | 4358 | 2 | 602 | 4 | 6857 | 1570 | 47/47 | 0.879 | 0.383 |
| 22 | 4863 | 4225 | 4223 | 2 | 638 | 4 | 6849 | 1564 | 47/47 | 0.869 | 0.408 |
| 23 | 4835 | 4230 | 4224 | 6 | 605 | 6 | 6791 | 1498 | 47/47 | 0.875 | 0.404 |
| 24 | 4914 | 4337 | 4334 | 3 | 577 | 5 | 6770 | 1484 | 51/51 | 0.883 | 0.389 |
| 25 | 5194 | 4586 | 4585 | 1 | 608 | 4 | 6808 | 1596 | 57/57 | 0.883 | 0.381 |
| 26 | 5031 | 4422 | 4419 | 3 | 609 | 5 | 6765 | 1554 | 57/57 | 0.879 | 0.392 |
| 27 | 5207 | 4604 | 4602 | 2 | 603 | 3 | 6717 | 1566 | 63/63 | 0.884 | 0.385 |
| 28 | 5158 | 4581 | 4581 | 0 | 577 | 3 | 6747 | 1564 | 63/63 | 0.888 | 0.369 |
| 29 | 5129 | 4531 | 4528 | 3 | 598 | 5 | 6707 | 1530 | 63/63 | 0.883 | 0.391 |
| 30 | 5245 | 4633 | 4632 | 1 | 612 | 2 | 6676 | 1564 | 69/69 | 0.883 | 0.391 |
| 31 | 5227 | 4634 | 4631 | 3 | 593 | 6 | 6717 | 1561 | 69/69 | 0.887 | 0.380 |
| 32 | 5379 | 4779 | 4778 | 1 | 600 | 3 | 6691 | 1592 | 79/79 | 0.888 | 0.377 |
| 33 | 5321 | 4731 | 4731 | 0 | 590 | 3 | 6639 | 1551 | 79/79 | 0.889 | 0.380 |
| 34 | 5265 | 4681 | 4681 | 0 | 584 | 2 | 6611 | 1532 | 79/79 | 0.889 | 0.381 |
| 35 | 5314 | 4728 | 4725 | 3 | 586 | 4 | 6576 | 1551 | 79/79 | 0.890 | 0.378 |
| 36 | 5355 | 4730 | 4728 | 2 | 625 | 2 | 6671 | 1534 | 83/83 | 0.883 | 0.407 |
| 37 | 5357 | 4781 | 4779 | 2 | 576 | 4 | 6590 | 1512 | 83/83 | 0.892 | 0.381 |
| 38 | 5428 | 4835 | 4834 | 1 | 593 | 2 | 6624 | 1596 | 83/83 | 0.891 | 0.372 |
| 39 | 5305 | 4736 | 4732 | 4 | 569 | 5 | 6535 | 1492 | 83/83 | 0.893 | 0.381 |
| 40 | 5429 | 4833 | 4833 | 0 | 596 | 2 | 6628 | 1528 | 89/89 | 0.890 | 0.390 |
| 41 | 5347 | 4772 | 4769 | 3 | 575 | 4 | 6540 | 1521 | 89/89 | 0.892 | 0.378 |
| 42 | 5340 | 4780 | 4780 | 0 | 560 | 2 | 6510 | 1509 | 89/89 | 0.895 | 0.371 |
| 43 | 5338 | 4797 | 4796 | 1 | 541 | 2 | 6511 | 1519 | 89/89 | 0.899 | 0.356 |
| 44 | 5311 | 4738 | 4738 | 0 | 573 | 4 | 6613 | 1533 | 89/89 | 0.892 | 0.374 |
| 45 | 5400 | 4848 | 4848 | 0 | 552 | 1 | 6493 | 1517 | 91/91 | 0.898 | 0.364 |
| 46 | 5405 | 4845 | 4844 | 1 | 560 | 3 | 6523 | 1526 | 91/91 | 0.896 | 0.367 |
| 47 | 5335 | 4796 | 4793 | 3 | 539 | 4 | 6475 | 1515 | 91/91 | 0.899 | 0.356 |
| 48 | 5379 | 4831 | 4831 | 0 | 548 | 1 | 6553 | 1524 | 95/95 | 0.898 | 0.360 |
| 49 | 5338 | 4779 | 4778 | 1 | 559 | 4 | 6521 | 1502 | 95/95 | 0.895 | 0.372 |
| 50 | 5437 | 4924 | 4923 | 1 | 513 | 2 | 6458 | 1480 | 110/111 | 0.906 | 0.347 |
| 51 | 5511 | 4962 | 4962 | 0 | 549 | 2 | 6436 | 1493 | 111/111 | 0.900 | 0.368 |
| 52 | 5417 | 4882 | 4881 | 1 | 535 | 2 | 6493 | 1483 | 111/111 | 0.901 | 0.361 |
| 53 | 5385 | 4833 | 4830 | 3 | 552 | 3 | 6462 | 1503 | 108/111 | 0.897 | 0.367 |
| 54 | 5511 | 4945 | 4944 | 1 | 566 | 1 | 6438 | 1512 | 123/123 | 0.897 | 0.374 |
| 55 | 5478 | 4913 | 4912 | 1 | 565 | 2 | 6402 | 1481 | 122/123 | 0.897 | 0.381 |
| 56 | 5519 | 4999 | 4998 | 1 | 520 | 2 | 6404 | 1495 | 120/123 | 0.906 | 0.348 |
| 57 | 5421 | 4859 | 4858 | 1 | 562 | 1 | 6387 | 1481 | 122/123 | 0.896 | 0.379 |
| 58 | 5452 | 4910 | 4908 | 2 | 542 | 3 | 6436 | 1522 | 123/123 | 0.901 | 0.356 |
| 59 | 5476 | 4927 | 4926 | 1 | 549 | 3 | 6420 | 1480 | 122/123 | 0.900 | 0.371 |
| 60 | 5437 | 4892 | 4891 | 1 | 545 | 1 | 6397 | 1461 | 123/125 | 0.900 | 0.373 |

Burnt fuelled families per turn, m <= 12 (family: count): m=1: {}; m=2: {}; m=3: {(1,3) 585, (3,1) 599}; m=4: {(1,3) 514, (2,4) 217, (3,1) 559, (4,2) 240}; m=5: {(1,3) 525, (1,5) 232, (2,4) 231, (3,1) 546, (3,5) 160, (4,2) 226, (5,1) 233, (5,3) 170}; m=6: {(1,3) 566, (1,5) 233, (2,4) 214, (3,1) 531, (3,5) 168, (4,2) 214, (4,6) 145, (5,1) 215, (5,3) 173, (6,4) 163}; m=7: {(1,3) 516, (1,5) 223, (2,4) 198, (3,1) 507, (3,5) 169, (4,2) 207, (4,6) 162, (5,1) 217, (5,3) 155, (6,4) 146}; m=8: {(1,3) 510, (1,5) 210, (2,4) 203, (2,8) 116, (3,1) 508, (3,5) 142, (4,2) 205, (4,6) 152, (5,1) 223, (5,3) 155, (6,4) 145, (6,8) 76, (8,2) 119, (8,6) 81}; m=9: {(1,3) 521, (1,5) 194, (1,9) 169, (2,4) 221, (2,8) 102, (3,1) 530, (3,5) 166, (4,2) 206, (4,6) 157, (5,1) 229, (5,3) 145, (5,9) 55, (6,4) 142, (6,8) 79, (8,2) 107, (8,6) 81, (9,1) 179, (9,5) 62}; m=10: {(1,3) 472, (1,5) 213, (1,9) 176, (2,4) 201, (2,8) 95, (3,1) 507, (3,5) 131, (4,2) 195, (4,6) 158, (4,10) 54, (5,1) 187, (5,3) 158, (5,9) 65, (6,4) 132, (6,8) 84, (8,2) 103, (8,6) 83, (8,10) 30, (9,1) 183, (9,5) 61, (10,4) 62, (10,8) 32}; m=11: {(1,3) 496, (1,5) 198, (1,9) 189, (2,4) 185, (2,8) 107, (3,1) 474, (3,5) 139, (4,2) 184, (4,6) 138, (4,10) 62, (5,1) 187, (5,3) 146, (5,9) 50, (6,4) 141, (6,8) 63, (8,2) 107, (8,6) 70, (8,10) 26, (9,1) 160, (9,5) 46, (10,4) 60, (10,8) 32}; m=12: {(1,3) 473, (1,5) 223, (1,9) 184, (2,4) 188, (2,8) 98, (2,12) 142, (3,1) 481, (3,5) 147, (4,2) 205, (4,6) 135, (4,10) 50, (5,1) 209, (5,3) 156, (5,9) 52, (6,4) 145, (6,8) 74, (8,2) 102, (8,6) 71, (8,10) 29, (9,1) 177, (9,5) 62, (10,4) 62, (10,8) 34, (10,12) 44, (12,2) 141, (12,10) 45};
