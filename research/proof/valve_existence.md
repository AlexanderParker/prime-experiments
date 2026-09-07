# Existence in the valve (R4.c.iii, prover, 2026-09-07)

Branch of R4.c (the valves), spawned by the owner's reformulation of the target (the_wall.md 5m,
tree log 2026-09-07): "we don't have to see position, just knowing there exists any position would
be proof", and "the position doesn't have to be inside the window, which is the engine; it can be in
the valve." Exact form: for every Q some turn m holds a pure charge, i.e. a twin in (Q, Q^2]; by the
descent (dead_branches_reopened_3.md, idea (a)) this is the engine's window statement at SOME scale
in [sqrt(2Q), Q], not at every scale.

Sources read: turn_ledger.md (V1, V2), valves_reconcile.md, valves_scratch.md,
dead_branches_reopened_3.md (the descent, the adversaries), manifold_census_large.md (W103),
top_machine_lean.md rounds 32-37, island_witness.md, position_frontier.md (theorem (E), the frontier
law, the certified constant), anchor_window.md, the_wall.md (faces A, B, 5l, 5m), the tree node R4.c.
Scripts in research/valves/r2/, outputs in research/valves/r2/results/ (gitignored; every number
used is in this file). Laws numbered V6 onward.

Vocabulary (the owner's, canonical): ENGINE = primes <= q, period q#, its open residues mod q# =
the columns with both members coprime to q#; VALVES = the engine acting inside the manifold's open
set; CHARGE s x P (air s q-smooth, fuel P = 1 or a prime above Q); FAMILY (s, s') = charges at
distance 2; PURE CHARGE = family (1, 1) = the twins; IMPRINT = the residues mod q# a family occupies;
ONSET: a family fires at turn max(s, s'), turn m = (mQ, (m + 1)Q]; EMBER = a q-smooth number above
Q; PORT = the class mod 6 fixed by the air; MANIFOLD = primes in (q, Q], on [1, Q^2] a number is
open iff no prime factor in (q, Q]; EXHAUST = primes above Q. Column k = (6k - 1, 6k + 1); machine
{5..y}; W(y) = the window of {5..y} in columns, (y/6, (y'^2 - 1)/6) with y' the next prime; F(y) =
the record of {5..y} (max-gap convention: the longest gap between consecutive openings over the full
period, so the longest blocked run has F - 1 columns); S(y) = F(y)/W(y) the record's share.

## Pre-registered (written before any script of this branch was run)

### The theory

The zone (Q, Q^2] is the union of the turns m = 1..Q - 1, and each turn is the top slice of the
window of a smaller machine (theorem (E) plus the exhaust cap). So "some turn holds a pure charge"
is a disjunction of Q - 1 statements, one per rung y_m = sqrt((m + 1)Q) in [sqrt(2Q), Q], each of
them a STRENGTHENED window statement at that rung (an opening in the top 1/(m + 1) of the window,
not anywhere in it), and the disjunction as a whole is exactly the window statement at rung Q
(the zone is the window of {5..Q} up to the sliver (Q^2, Q'^2)). I therefore expect:

- the freedom of the turn is real for a PROOF (one may choose the rung at which to exhibit the
  twin, and the smaller rungs have certified records) and empty for the STATEMENT (no turn's
  statement is weaker than the root at its own rung; the disjunction is the root at rung Q);
- the record-based sufficient conditions are ordered: S(y_m) < 1/(m + 1) (the share) implies "no
  blocked run of the window of y_m covers its top 1/(m + 1)" (the position), and the position
  condition is EQUIVALENT to the turn statement, not merely sufficient: turn m of Q is empty iff
  the blocked run ending at the top of the window of y_m is at least Q/6 columns long. So the
  weakest engine statement is the top-run statement, and it is the turn statement renamed;
- the frontier (x >= cL for a blocked run of length L starting at column x, the effective
  machine's law) certifies exactly the turns m < c: c = 1.25 proved gives turn 1 only, the
  measured c = 3.25 gives turns 1, 2, 3; the proved version reaches only the Q for which the
  effective machine at the slice is certified (2Q + 2 <= 59^2, i.e. Q <= 1739), a range where
  the twins are known by hand, so the proved certification is structural, not new information;
- no invariant of the charge set forces a member on the pure imprint: the mirror fixes the pure
  imprint setwise and fixes the empty set too; the pigeonhole over runs of consecutive charges
  fails because the manifold's run ceiling (q' - 1 consecutive open odd numbers) is below the
  engine's twin-free run on the odd line (which is about 3 F(q) odd numbers) at every q >= 7 and
  equal to it at q = 5; the walk over valves has no cap; and every invariant the real fuel has
  that the counterfactual F lacks is either a count (balance) or is also possessed by the
  one-tooth free-phase adversary (mirror closure, one tooth per gear), leaving phase zero as the
  only separating property, whose consequence for existence is the root.

### Predictions, with numbers, and what would refute each

- E-P1 (tiling). The turns m = 1..Q - 1 partition (Q, Q^2]; a twin in the zone lies in exactly
  one turn. Trivial; 0 exceptions. Refuted by nothing (it is arithmetic).
- E-P2 (the descent, exact). For every Q and m <= Q - 1: the twins in (mQ, (m + 1)Q] are exactly
  the openings of the machine {5..y} in that interval, for every real y with y <= Q and
  nextprime(y)^2 > (m + 1)Q + 2; and with y_m = sqrt((m + 1)Q) the interval is the top slice
  (y_m^2 m/(m + 1), y_m^2] of the window at y_m. Checked as sets at (q, Q) = (5, 10^4) for m <= 60
  and at every Q <= 10^4 for m = 1, 2 by direct sieve: 0 exceptions expected. Refuted by one
  column open under the effective machine that is not a twin.
- E-P3 (the top-run share). Define tau(y) = (blocked run of the window of {5..y} ending at the
  window's top, in columns) / W(y). Over every prime rung y in [23, 19,997]: max tau below 0.05,
  attained at a rung below 100; tau(y) < 10^-3 at every rung above 2,000; tau(y) < 1/(m + 1) for
  every m <= 100 at every rung above 300. Refuted by a rung above 2,000 with tau > 10^-3.
- E-P4 (the frontier's certified turns). From the frontier law R_min^>=(L) >= cL on the prefix:
  turn m of Q is nonempty whenever m < c and the slice is inside the prefix of the machine at
  y_m. With c = 1.25 (proved, stretches with top member below 3481): turn 1 for every Q <= 1739.
  With c = 3 (the prefix law, 0 exceptions in 8,375 cells to rung 19,997): turns 1 and 2 for every
  Q with 3Q + 2 < 19,997'^2, i.e. Q <= 1.3 x 10^8. With the period constant 3.25: turns 1, 2, 3.
  With the prefix's realised floor 4.625: turns 1-4. No turn m >= 5 from the frontier at any
  rung. Refuted by an exact top-slice check disagreeing with the certification (a certified
  turn found empty).
- E-P5 (the mirror). The mirror n -> -n - 2 on residues mod q# fixes the pure imprint R(1, 1)
  setwise and has exactly one fixed class in it, n = -1 (mod q#), the spoke class (proof: |R(1, 1)|
  = prod (p - 2) is odd, and 2r = -2 mod q# has the two solutions -1 and q#/2 - 1, the second
  even). The number of charges in a pure-imprint class is the number of twins in that class. At
  (5, 10^4) over the whole zone the three classes 11, 17, 29 mod 30 hold 440,107 twins split within
  1% of equal, mirror pair (11, 17) within 2 sqrt(N) = 760; the same at (7, 10^4) over 15 classes.
  For the counterfactual F and the free-phase adversary every pure-imprint class holds 0 charges:
  the symmetry forces nothing on counts. Refuted by a class count off by more than 3 sqrt(N).
- E-P6 (the pigeonhole). The engine's twin-free run on the odd line, F_odd(q) = the most
  consecutive odd numbers with no pair (n, n + 2) both coprime to q#: predicted 6, 15, 21, 33, 54,
  75 at q = 5, 7, 11, 13, 17, 19 (about 3 F(q)). The manifold's run ceiling: q' - 1 consecutive
  open odd numbers (q' consecutive odd numbers contain a multiple of q'): 6, 10, 12, 16, 18, 22.
  The pigeonhole "a run of consecutive charges longer than the burnt residues available forces a
  pure one" therefore needs a run of F_odd + 1 open odd numbers and the manifold allows at most
  q' - 1: it fails by exactly 1 at q = 5 (7 needed, 6 allowed) and by a growing margin after.
  Measured longest run of consecutive open odd numbers in the zone: 6 at (5, 10^3) and (5, 10^4)
  (the ceiling is attained), below the ceiling at q >= 7. The residue-window version: the most
  charges in any window of length q# in the zone against the burnt residues q# - prod (p - 2) =
  27, 195, 2,175: predicted maxima 9, 30, 200 at Q = 10^4, so it fails by a factor 3 to 10.
  Refuted by a run of open odd numbers longer than q' - 1 (impossible) or by a q#-window with
  more charges than burnt residues.
- E-P7 (the walk over valves). The charge-unit walk from a twin to the next twin (1 + burnt charges
  between) has no cap: its per-turn maximum at (5, 10^4) rises from at most 3 in turn 1 to above 30
  by turn 60, and the ratio max/mean per turn is 4 to 8 (a geometric tail). Refuted by a per-turn
  maximum bounded by a function of the open valve count A(m) that the data do not exceed.
- E-P8 (the counterfactual test). F = {n > Q : gcd(n, q#) = 1, n = 1 (mod 3)} violates: balance on
  gear 3 (all fuel in class 1 mod 3), mirror closure of the valve set (0 of its families mirrored),
  the one-tooth form (its complement in the coprime integers is not one class per manifold prime),
  phase zero. The free-phase adversary (greedy one-class cover of the pure-imprint pairs in turns
  1..60 at (5, 10^4)) violates phase zero only, with about 780 gears used and P_m = 0 for m <= 60.
  Every candidate of E-P5..E-P7 evaluated on both: the mirror count (0 in every class, so nothing
  forced), the pigeonhole premise (never met by any set, so nothing distinguished), the walk
  (infinite for both). Refuted by an invariant that the real charge set has, both adversaries
  lack, and that is not the count of primes.

Owner's predictions on the scorecard (from the brief): (O1) knowing that some turn holds a pure
charge is proof (I agree: the disjunction over turns is the window statement at rung Q); (O2) the
position can be in the valve, so the engine's statement is needed at SOME scale in [sqrt(2Q), Q]
rather than every scale (I predict: at some scale, but in its top-slice form, which is stronger
than the window statement at that scale; the freedom is a choice of rung, not a weakening); (O3)
the frontier sees length at the top of the window and certifies turns from position facts (I
predict turns 1-3 measured, turn 1 proved, and only to Q <= 1739 for the proved part); (O4) an
invariant of the charge set may force a member on the pure imprint (I predict none survives the
free-phase adversary except phase zero, whose consequence is the count).

### Scorecard (filled after the runs)

| prediction | verdict | where |
|---|---|---|
| E-P1 tiling | HELD (arithmetic; 0 exceptions) | Setup |
| E-P2 the descent as sets | HELD: 0 mismatches in 60 turns at (5, 10^4) and in 19,998 cells (m = 1, 2; Q = 3..10^4); the one cell Q = 2, m = 1 differs by the twin (3, 5), which is not a column | Freedom, V6 |
| E-P3 the top-run share | HELD on two clauses, REFUTED on one: max tau = 0.0833 at y = 29 (predicted < 0.05; it is the twin gap 883 -> 1019 under 31^2 = 961), tau < 10^-3 at every rung above 1,123 (predicted above 2,000), tau < 1/(m + 1) for all m <= 362 above rung 300 (predicted 100) | Frontier, table 2 |
| E-P4 the certified turns | HELD, with the proved range corrected upward: turn 1 for every Q in [30, 1859] (F(59) = 161 is on the certified ladder, so y_m <= 59 reaches Q <= 1859, not 1739); turn 2 on the upper part of most rungs' ranges; turn 3 at six rungs on 24 values of Q; turn 4 at one Q; c = 3 gives turns 1, 2 to Q <= 1.3 x 10^8, 3.25 turns 1-3, 4.625 turns 1-4; no turn m >= 5 anywhere; 0 certified turns found empty | Frontier, V8 |
| E-P5 the mirror | HELD: one fixed class (-1 mod q#); zone counts 146,774 / 146,889 / 146,444 (spread 0.3 %), mirror pair off by 115 against 663 = sqrt(N); 0 in every class for F and for the adversary | Invariant (a) |
| E-P6 the pigeonhole | HELD: F_odd = 6, 15, 21, 33, 54, 75 exactly as predicted; ceilings 6, 10, 12, 16, 18, 22; margin 1 at q = 5 and the failure is realised (runs of 6 open odd numbers with no twin: 2 of 14 at Q = 10^3, 4 of 32 at 10^4); the q#-window maxima 11, 38, 298 against 27, 195, 2,175 burnt residues (predicted 9, 30, 200: right order, low) | Invariant (b), V10 |
| E-P7 the walk | HELD: no cap; per-turn maximum 2 -> 53 from turn 1 to 60 at (5, 10^4); max/mean 1.8-5.1 (predicted 4-8: the tail is geometric but lighter than I wrote); measured max / geometric scale 0.74-1.86 at every turn and every q | Invariant (c), V11 |
| E-P8 the counterfactual test | HELD: F violates balance (gear 3: 0 / 80,000), mirror closure (0 of 54 families), one-tooth (115,803 fuel members with a manifold factor), phase zero; the adversary violates phase zero only (776 of 1,226 gears used, 774 at nonzero phase, P_m = 0 for m <= 60); nothing but phase zero separates the real set from both | Invariants, V12 |
| Owner O1 | AGREED and made exact: the disjunction over turns is the window statement at rung Q less the sliver (Q^2, Q'^2) | Freedom |
| Owner O2 | HELD in the form predicted: the freedom is a choice of rung, and the statement needed there is the top-slice form; measured worth in the share form: three integers (Q = 5, 26, 29 rescued by turn 2), and Q = 24, 25 certified by no turn at all while twins sit in (Q, 2Q] | Freedom, V7 |
| Owner O3 | HELD and better than I predicted: turns 1-4 by the prefix floor 4.625 (I wrote 1-3 from the period constant); turn 1 proved to Q <= 1859; the run that sets the constant IS the last empty turn 5 (Q = 132..134) | Frontier, V9 |
| Owner O4 | REFUTED as predicted: phase zero is the only property the real charge set has that both adversaries lack, and its existence consequence is the root | Invariants, V12 |

## Setup

Scripts (research/valves/r2/): certify.py (the certified turns: the share certificate per rung and per Q,
the frontier form for c = 1.25, 3, 3.25, 4.625 against the sieve, the tight instance, the descent as sets
for m = 1, 2 at every Q <= 10^4; results/certify_10000.txt), top_run.py (the top-run share tau(y) at every
prime rung 23..19,997 by the reduction (R), and the descent as sets for 60 turns at (5, 10^4);
results/top_run_19997.txt), turn_scan.py (P_m(Q) exactly for Q <= 10^5, m <= 100;
results/turn_scan_100000_100.json), share_free.py (the share form with the freedom of the turn, Q <= 1859;
results/share_free.txt), invariants.py (the charge set in turns 1..60 under the real fuel, the counterfactual
F and the one-tooth free-phase adversary at (q, Q) = (5, 10^3), (5, 10^4), (7, 10^3), (7, 10^4),
(11, 10^4); results/invariants_all.txt, inv_adv_q*_Q1000.txt), classes_zone.py (the pure charges per pure-
imprint class over the whole zone (10^4, 10^8], q = 5, 7, 11; results/classes_zone_Q10000_q*.json).
Exact ranges: the turn scan sieves to 1.01 x 10^7; top_run sieves the odd numbers to 20011^2 = 4.0 x 10^8
(21,358,532 odd primes, 1,509,195 twins); every other number is an exact sieve of the range named.

Coordinates. Turn m of Q is the integers (mQ, (m + 1)Q]; its columns are K_m(Q) = [klo, khi] with
klo = ceil((mQ + 2)/6), khi = floor(((m + 1)Q + 1)/6), c_m(Q) = khi - klo + 1 columns (c_m = Q/6 up to one);
y_m(Q) = the largest prime <= sqrt((m + 1)Q + 2), the effective machine of the slice by theorem (E);
W(y) = (y'^2 - 1)/6 the window's top column; F(y) the record (max-gap convention: the longest blocked run
has F - 1 columns, so any F consecutive columns hold an opening); the certified ladder F(5..59) = 2, 5, 7,
11, 18, 25, 34, 43, 58, 88, 91, 103, 118, 145, 161 (ladder_closure.md).

## The freedom of the turn, exact

**V6 (the descent for every turn; PROVED, 0 exceptions).** For Q >= 3 and 1 <= m <= Q - 1, the twins with
lower member in (mQ, (m + 1)Q] are exactly the openings of the machine {5..y_m(Q)} in the columns
K_m(Q), and K_m(Q) lies inside the window of y_m: klo > y_m/6 and khi <= W(y_m). Proof: a column of the
slice has both members prime iff no prime <= sqrt(6k + 1) divides either; theorem (E) (position_frontier.md,
(E)) says blocked under all of {5..Q} iff blocked under {5..floor(sqrt(6k + 1))} for 6k - 1 > Q, and
sqrt(6k + 1) <= sqrt((m + 1)Q + 2) < y_m' for every column of the slice; 3 divides no member of a column, 2
neither. The containment: 6klo - 1 > mQ >= y_m and 6khi + 1 <= (m + 1)Q + 2 < y_m'^2. Checked as sets:
60 turns at (5, 10^4) (openings of {5..sqrt((m + 1)Q + 2)} in the slice against the twins there, 0
mismatches, top_run.py) and every (m, Q) with m = 1, 2 and Q = 3..10^4 (19,998 cells, 0 mismatches,
certify.py; the cell Q = 2, m = 1 differs by the twin (3, 5), which is no column). The union of the
turns is one line: the intervals (mQ, (m + 1)Q], m = 1..Q - 1, tile (Q, Q^2] (E-P1).

So existence in the valve at Q, EV(Q) = "some turn m holds a pure charge" = "a twin in (Q, Q^2]", is
exactly the disjunction over m = 1..Q - 1 of TS(y_m; Q, m) = "the machine {5..y_m} has an opening in
K_m(Q)", the top-slice statement at rung y_m in [sqrt(2Q), sqrt(Q^2 + 2)]. Three relations follow
in one line each:

- EV(Q) implies the window statement WS(Q) of the engine {5..Q} (a twin in (Q, Q^2] is an opening in
  the columns (Q/6, (Q^2 + 1)/6] of Q's window); WS(Q) does not imply EV(Q) at a single Q, since the
  window's only opening may sit in the sliver (Q^2, Q'^2). The disjunction over all turns is WS(Q) less
  that sliver.
- No window statement WS(y) at a single rung y implies EV(Q): WS(y) gives a twin in (y, y'^2 - 2],
  which lies in (Q, Q^2] only if y >= Q and y' <= Q. The freedom of the turn is a choice among the
  rungs y_m, and at the chosen rung the statement needed is TS, a strengthening of WS(y_m) (an opening
  in the top 1/(m + 1) of the window rather than anywhere in it). O2 held in this form: the freedom is
  real for a proof (one chooses the rung, and the small rungs have certified records) and is not a
  weakening of the statement.
- The whole family: for all Q, EV(Q) iff for all y, WS(y) iff twins are infinite (kernel-equivalent
  root). So "existence in the valve for every Q" is the root, and the branch's deliverable is which
  finite pieces of it the engine's certified facts reach.

**The weakest engine statement, two forms (V7; PROVED as identities).** Fix Q and m.

- Share form SF_m(Q): F(y_m(Q)) <= c_m(Q). Then any c_m consecutive columns hold an opening of
  {5..y_m}, so TS holds. In the record's share S(y) = F(y)/W(y): S(y_m) <= c_m(Q)/W(y_m), and
  c_m/W(y_m) <= 1/(m + 1) with equality only when (m + 1)Q + 2 = y_m'^2 - 1; so the clean reading
  "S(y_m) < 1/(m + 1)" is NOT sufficient when the sliver (the columns of y_m's window above the slice) is
  large, and the exact form is F(y_m) <= c_m(Q). Since F(y_m)/(y_m^2/6) is 0.28-0.61 on the ladder and
  c_m(Q) = Q/6 ~ y_m^2/(6(m + 1)), SF_m can hold only for m + 1 < 1/0.28 = 3.6: the share form is the
  record law at one of the three rungs sqrt(2Q), sqrt(3Q), sqrt(4Q), never further up.
- Position form PF_m(Q): no blocked run of {5..y_m(Q)} covers K_m(Q); equivalently the maximal blocked
  run containing khi (if khi is blocked) starts above klo; equivalently the blocked run of {5..y_m}
  ending at column khi has fewer than c_m(Q) columns. PF_m(Q) iff TS(y_m; Q, m) iff turn m nonempty:
  it is the turn statement renamed through V6, not a sufficient condition for it.

Which is weaker: PF. SF_m implies PF_m (the record bounds every run). The gap is measured exactly on
the certified range (share_free.py, all m with y_m(Q) <= 59): over Q = 3..1859 the least share-certified
turn is m = 1 at 1,852 values of Q and m = 2 at three (Q = 5, 26, 29; at 26 and 29 F(7) = 5 exceeds c_1 = 4
but not c_2 = 5), and at Q = 24 and Q = 25 NO turn is share-certified (m = 1, 2, 3 have y_m = 7, F = 5, c_m = 4;
m = 4, 5 have y_m = 11, F = 7, c_m = 4 or 5) although (29, 31) and (41, 43) lie in (Q, 2Q]. So the
share form with the freedom of the turn is strictly stronger than existence (false at two Q where
existence is true), the freedom is worth exactly three integers in the certified range and no rung
beyond the third asymptotically, and the position form is the only statement of the two that is
equivalent to existence. The weakest statement about the engine's records implying existence in the
valve for all Q is therefore: for every Q there is a turn m at which the blocked run of {5..y_m(Q)}
ending at column khi_m(Q) is shorter than c_m(Q) columns. That is the root in the descent's
coordinate: ROOT. The share form is the record law S(y) < 1/2 at y = sqrt(2Q) (or 1/3, 1/4 one or two
rungs up), a length law whose proof is face A / face E: ROOT. Between the two sits a third kind of
statement, a position law of the engine that implies PF for a range of turns without being a record
bound: the frontier, next.

## The frontier's certified turns

**The bound on runs at the top of the window.** The frontier law at rung y with constant c: every
blocked run of {5..y} of length L >= d_0 (the initial run excluded) starting at a column x with
6x - 1 > y has x >= cL (position_frontier.md; c = 1.25 proved from the ladder for every run whose top
member is below 61^2, i.e. y <= 59, the L-by-L form; c = 3.25 measured on the full periods m7..m29
with 0 exceptions, c = 3 in the prefix [1, W] at 211 rungs to 19,997 with 8,375 cells and realised
floor 4.625). Read at the top: a run ending at the top column W has x = W - L + 1 >= cL, so
L <= (W + 1)/(c + 1): the run ending at the window's top is at most 1/(c + 1) of the window. That is
"the frontier sees length at the top": a law about where long runs may START becomes a bound on the
LENGTH of the one run that touches the top. Measured against it (top_run.py, table 2): tau(y) = L_top/W
has maximum 0.0833 at y = 29 (13 columns of 156: the twin gap (881, 883) -> (1019, 1021) under 31^2),
against 1/(c + 1) = 0.178 at c = 4.625, slack 2.1.

Table 2, the top-run share by band of rungs (2,254 rungs):

| rungs | max tau (at y) | median tau | max L_top (cols) | rungs with L_top = 0 | top slice 1/(m + 1) open for all m <= |
|---|---|---|---|---|---|
| 23-100 | 0.0833 (29) | 2.2e-3 | 13 | 7 of 17 | 11 |
| 100-300 | 0.0123 (109) | 1.0e-3 | 51 | 13 of 37 | 80 |
| 300-1000 | 0.00275 (337) | 1.0e-4 | 62 | 26 of 106 | 362 |
| 1000-3000 | 0.000273 (1123) | 1.7e-5 | 156 | 58 of 262 | 3,658 |
| 3000-10000 | 0.000077 (3041) | 2.9e-6 | 195 | 149 of 799 | 12,906 |
| 10000-19997 | 0.000009 (10691) | 6.8e-7 | 386 | 195 of 1033 | 112,422 |

**V8 (the frontier's certified turns; PROVED with its hypothesis).** Let y be a prime rung at which the
frontier law holds with constant c for runs of length >= c_m(Q) starting above column (y + 1)/6. Then
for every Q and m with y_m(Q) = y and c x c_m(Q) > klo_m(Q), turn m of Q is nonempty. Proof: if the
turn were empty, K_m(Q) would by V6 be blocked under {5..y}, so a maximal blocked run of length L >=
c_m(Q) starts at some x <= klo; the hypothesis 6x - 1 > y holds since 6klo - 1 > mQ >= y; the frontier
gives x >= cL >= c c_m(Q) > klo, a contradiction. Since klo = m c_m(Q) up to one column, the
condition is m < c up to one column: the frontier with constant c certifies exactly the turns
m < c, for every Q with y_m(Q) in the range where the frontier holds.

Certified by the proved constant. With c = 1.25 the condition 1.25 c_1(Q) > klo_1(Q) holds for every
Q >= 54 (9,977 of the 9,999 Q <= 10^4; the 22 misses are the one-column wobble at Q <= 53), and for m = 2 at one Q only; so the
pure-position constant certifies turn 1 and nothing else. The proved range is y_1(Q) <= 59, i.e.
2Q + 2 < 61^2, Q <= 1859 (the pre-registration's 1739 used 59^2 as the cap; F(59) = 161 is certified,
ladder_closure.md, so the correct cap is 61^2). The L-by-L form of the proved frontier (R_min(L) >=
ceil((y_L^2 - 1)/6) - L + 1, y_L the least rung with F(y_L) >= L + 1) certifies exactly what the share
certificate F(y_m) <= c_m(Q) does, since it is derived from the same ladder through (E); the exact
per-rung result (certify.py, per-rung table) is:

| rung y | F | F/(y^2/6) | turn 1: Q-range, certified from | turn 2 | turn 3 | turn 4 |
|---|---|---|---|---|---|---|
| 5 | 2 | 0.480 | [12, 23] all | [8, 15] from 10 | [6, 11] from 11 | [5, 9] at 9 |
| 7 | 5 | 0.612 | [24, 59] from 30 (32 of 36) | [16, 39] from 28 | [12, 29] from 29 | none |
| 11 | 7 | 0.347 | [60, 83] all | [40, 55] all | [30, 41] at 41 | none |
| 13 | 11 | 0.391 | [84, 143] all | [56, 95] from 64 | [42, 71] from 65 | none |
| 17 | 18 | 0.374 | [144, 179] all | [96, 119] from 106 | none | none |
| 19 | 25 | 0.416 | [180, 263] all | [120, 175] from 148 | none | none |
| 23 | 34 | 0.386 | [264, 419] all | [176, 279] from 202 | [132, 209] from 203 | none |
| 29 | 43 | 0.307 | [420, 479] all | [280, 319] all | none | none |
| 31 | 58 | 0.362 | [480, 683] all | [320, 455] from 346 | none | none |
| 37 | 88 | 0.386 | [684, 839] all | [456, 559] from 526 | none | none |
| 41 | 91 | 0.325 | [840, 923] all | [560, 615] all | none | none |
| 43 | 103 | 0.334 | [924, 1103] all | [616, 735] all | none | none |
| 47 | 118 | 0.321 | [1104, 1403] all | [736, 935] all | none | none |
| 53 | 145 | 0.310 | [1404, 1739] all | [936, 1159] all | [702, 869] at 869 | none |
| 59 | 161 | 0.278 | [1740, 1859] all | [1160, 1239] all | none | none |

So by the certified ladder alone: turn 1 of every Q in [30, 1859] is nonempty (and of Q = 12..23, 28);
turn 2 on the upper part of every rung's range from rung 5 on and on the whole range at rungs 11, 29,
41, 43, 47, 53, 59 (a rung certifies its whole turn-2 range iff F(y) <= y^2/18 - 1, i.e. F/(y^2/6) <=
0.33); turn 3 only at rungs 5, 7, 11, 13, 23, 53 on the top few Q of the range (24 values of Q in all);
turn 4 at Q = 9 only. No certified turn is empty (0 of 2,944 certificates against the sieve: 2,744 in certify.py's first pass with the cap 59^2, the 200 added at rung 59 by the turn scan, whose Q_1 = 6 and Q_2 = 10 cover them). This is
existence in the valve from position facts, with the theorem's hypothesis the frontier at the rung
sqrt(2Q), which the ladder proves to Q <= 1859: a range where the twins are known by hand, so the
proved certification is structural (it says WHY turn 1 is never empty there: no run of Q/6 columns
can start at column Q/6), not new information about the primes.

How far the measured constant carries it. The frontier form at c certifies m < c, and the truth from
the turn scan (Q <= 10^5, m <= 100) is that turn m is never empty from Q_m on with Q_1..Q_5 = 6, 10,
26, 28, 135, and Q_m = 116, 114, 93, 160, 142, 197, 149, 269, 409, 624 at m = 6, 8, 10, 15, 20, 30,
40, 50, 60, 100 (the largest Q with an empty turn among m <= 100 is 623). Against the constants:
c = 3 (the prefix law, 0 exceptions in 8,375 cells to rung 19,997) certifies turns 1, 2 for every Q
with 3Q + 2 < 20011^2, Q <= 1.3 x 10^8; c = 3.25 (the period constant) turns 1-3 to Q <= 10^8; c = 4.625
(the prefix's realised floor) turns 1-4 to Q <= 8.0 x 10^7; and the truth agrees at every Q in the
scan (certify.py: 0 wrong certificates at c <= 4.625 for Q >= 28; the WRONG entries at c = 4.625,
Q = 5..25, are all below rung 23 where the frontier was not measured). No constant on record
reaches turn 5, and the truth says none can:

**V9 (the tight instance is the last empty turn; FACT, exact).** The frontier's minimum ratio 4.625 =
111/24 is the run of m23 at columns 111..134 (position_frontier.md), i.e. the twin gap (659, 661) ->
(809, 811): 6 x 111 - 1 = 665 > 661 and 6 x 134 + 1 = 805 < 809. Turn 5 of Q = 132, 133, 134 is
(660, 792], (665, 798], (670, 804] with columns 111..132, 112..133, 112..134, all inside that run:
P_5 = 0 there and P_5 = 1 at Q = 131 and 135. So the run that sets the measured frontier constant is
exactly the last empty turn 5 (Q_5 = 135), and the constant 4.625 certifies turns 1-4 because turn 5
is the first turn a real twin gap empties above Q = 28. The period constant 3.25 = 13/4 is likewise
the twin gap (71, 73) -> (101, 103) (columns 13..16 of m7, the run that does not move as gears are
added). The frontier's constants are the first twin gaps of the primes, read as position/length, and
they stay the constants of every larger machine because new gears lengthen runs already present
rather than create earlier ones (position_frontier.md, section 1).

What this does and does not give. It gives an existence proof by position for turns 1-4 wherever the
frontier is measured (to Q = 8 x 10^7) and by proof for turn 1 to Q = 1859; the existence in the valve
for those Q then follows from the position of runs, not from a count (the owner's O3, held). It does
not give existence for all Q: the frontier at rung y with any constant c >= 1 implies, through V8 and
V6, a twin in turn 1 of every Q with y_1(Q) = y, i.e. a twin in (x, 2x] for x in a range, and on the
prefix the frontier IS that twin-Bertrand statement (by reduction (R) the prefix's runs are the twin
gaps), so "the frontier holds at every rung with c >= 1" is the root (face E, every local formulation
over-asks): ROOT. The proved constant comes from the ladder at rungs below sqrt(2Q), so the proved
certification for all Q would need the record law F(y) <= y^2/13.5 at every rung (the L-by-L form
needs less: F(y) <= y^2/12 - 1 at y = y_1(Q) exactly), which is the ladder formulation of the root.

## Invariants of the charge set

Each candidate is stated, measured on the real charge set, then tested on the counterfactual fuel F =
{n > Q : gcd(n, q#) = 1, n = 1 (mod 3)} (turn_ledger.md V2) and on the one-tooth free-phase manifold
adversary (dead_branches_reopened_3.md N11: each manifold prime p removes one class c_p, chosen greedily
to cover the pure-imprint pairs in turns 1..60). Numbers at (q, Q) = (5, 10^4), M = 60, unless stated.

### (a) The mirror on charges

What it is. The map n -> -n - 2 on residues mod q# sends the pair (n, n + 2) to (-n - 2, -n), the same
pair reversed, and so sends the imprint of the family (s, s') to the imprint of (s', s) (n = 0 mod p for
p | s goes to -2 mod p). It fixes the pure imprint R(1, 1) setwise; |R(1, 1)| = prod (p - 2) is odd, so
some class is fixed, and 2r = -2 (mod q#) has the solutions -1 and q#/2 - 1 of which only -1 is odd:
exactly one fixed class, the spoke class -1 (mod q#), verified at q = 5, 7, 11 (classes 29, 209, 2309).

What it forces on the real set: (i) the family inventory is mirror-closed: 122 of 123 fuelled
families present with their mirror at (5, 10^4), 108 of 115 at (5, 10^3), 226 of 228 at (7, 10^4), 316
of 318 at (11, 10^4), the exceptions being sparse families at their onset edge; (ii) the counts per
pure-imprint class are mirror-symmetric to square-root accuracy: over the whole zone (10^4, 10^8] the
three classes mod 30 hold 146,774 / 146,889 / 146,444 twins (mirror pair 11, 17 off by 115 against
sqrt(440,107) = 663; the fixed class 29 holds the fewest, by 0.3 %), the 15 classes mod 210 hold
29,065..29,504 (7 mirror pairs, largest difference 224), and in turns 1..60 the 135 classes mod 2310
hold 26..47 with mirror pairs off by at most 11 against sqrt(5205) = 72. What it does not force: any
class to be nonempty. Under F every pure class holds 0 and the family set is not mirror-closed (0 of 54
families present with their mirror: the mirror sends f = 1 (mod 3) to -f - 2 = 0 (mod 3), so the mirror
of F's imprint is disjoint from F). Under the adversary every pure class holds 0 and the family set IS
mirror-closed (114 of 118 at q = 5, 206 of 217 at q = 7): the adversary's strikes are one class per
gear, and the mirror of a one-class strike is a one-class strike, so closure of the inventory does not
need the strikes to be at phase zero. Verdict: the mirror is a symmetry of the IMPRINT (residues), and
its consequences for the charge SET are a closure of the inventory (a set property, shared with the
adversary) and a symmetry of the counts (a count, face A). Nothing on existence: FACT.

### (b) The pigeonhole from the manifold's metric laws

The laws (top_machine_lean.md, the manifold as the wheel of the gears in (q, Q] with teeth {0, -2}, all
kernel-proved): no gap of 4 between consecutive open pairs (L4); the longest run of consecutive open
pairs at step 1 is q' - 3 (L10, attained); the longest step-2 chain of open pairs is q' - 2 (L10,
attained), i.e. at most q' - 1 consecutive open odd numbers (q' consecutive odd numbers contain a multiple
of q'); the chain law (L12) on one gear's consecutive strikes. The engine's twin-free run on the odd
line: F_odd(q) = the most consecutive odd numbers no two adjacent of which are both coprime to q#. A run
of F_odd + 1 consecutive open odd numbers (odd charges) would contain an adjacent pair both coprime to
q#, both manifold-open and above Q, hence a twin: that is the pigeonhole.

**V10 (the pigeonhole fails at every q, by exactly one at q = 5, and the failure is realised; EXACT).**

| q | F_odd(q) (needed run = F_odd + 1) | manifold ceiling q' - 1 | margin | longest run measured at Q = 10^3 / 10^4 | runs of that length holding no twin |
|---|---|---|---|---|---|
| 5 | 6 (7) | 6 | 1 | 6 / 6 (14 / 32 runs) | 2 / 4 |
| 7 | 15 (16) | 10 | 6 | 9 / 9 (2 / 2 runs) | 0 / 0 |
| 11 | 21 (22) | 12 | 10 | - / 11 (1 run) | 0 |
| 13 | 33 (34) | 16 | 18 | | |
| 17 | 54 (55) | 18 | 37 | | |
| 19 | 75 (76) | 22 | 54 | | |

At q = 5 the manifold's ceiling is attained in the zone (runs of 6 open odd numbers exist: 14 at
Q = 10^3, 32 at 10^4) and the pigeonhole needs 7; the runs of 6 that hold no twin (2 and 4 of them) are
the realised failure. The residue-window form is weaker still: the most charge pairs (n, n + 2 both
charges) in any window of q# consecutive integers is 11, 38, 298 at q = 5, 7, 11 (Q = 10^4) against the
burnt odd residues q#/2 - prod (p - 2) = 12, 90, 1,020 that a pigeonhole would have to exceed (against
all burnt residues q# - prod (p - 2) = 27, 195, 2,175 if even charges are counted; the most charges of
any kind in a q#-window is 11, 38, 298 as well). At q = 5 the maximum 11 is exactly what the ceiling
allows (two odd multiples of 7 in 16 consecutive odd numbers kill 4 of the 15 pairs) and the pigeonhole
needs 13: short by 2 in the window form, by 1 in the run form. Mechanism: the engine's twin-free odd
run grows like 3 F(q) (6, 15, 21, 33, 54, 75 against F = 2, 5, 7, 11, 18, 25) while the manifold's
ceiling is set by its smallest gear q' alone and grows like q; the two cross at q = 5 and only there,
by one. Counterfactual test: the premise (a run of F_odd + 1 open odd numbers) is met by no set: F's
longest run is 5 at both q, the adversary's 6 (q = 5, one run, no twin) and 7 (q = 7); the ceiling
forbids the premise for every fuel, real or not, so the pigeonhole distinguishes nothing. FACT (a
metric fact about the two parts' scales), not a route.

### (c) The walk from a charge to the next pure charge

Over the valves in onset order the walk from a twin to the next twin in charge units is 1 + (burnt
charges between). In turns 1 and 2 the only burnt charges are ember charges, so the walk is at most
1 + 2 E_m (V1 / the ember law): measured maxima 2, 2 at (5, 10^4) and 3, 4 at (7, 10^4), 3, 4 at
(11, 10^4). From turn 3 the (1, 3) / (3, 1) valves open and the walk has no cap:

**V11 (the walk is a geometric tail; MEASURED, no exception to the tail law).** Per-turn maximum of the
walk at (5, 10^4): 2, 2, 10, 18, 21, 21, 30, 35, 53 at m = 1, 2, 3, 5, 9, 15, 30, 45, 60 (means 1.02,
1.10, 2.66, 4.23, 5.64, 6.58, 8.57, 9.74, 10.33; max/mean 1.8-5.1); overall maximum 59 (turn 43), 75
at q = 7 (turn 22), 81 at q = 11 (turn 22), rising with q because the burnt charges per turn rise
with q (T_60 = 817, 1,034, 1,171 at q = 5, 7, 11 against P_60 = 79). Against the geometric scale
log(P_m)/(-log(1 - P_m/T_m)) (the expected largest of P_m geometric gaps with the turn's pure share
per charge) the measured maximum is 0.74-1.86 of the scale at every turn listed and every q
(ratios at (5, 10^4): 1.56, 1.01, 0.99, 1.05, 0.90, 0.74, 0.82, 0.88, 1.23). The manifold's record
laws bound runs of consecutive OPEN numbers (charges), not runs of burnt charges: they cap how many
charges can sit together (q' - 2 at step 2), which is the wrong side of the walk. Counterfactual:
under F and under the adversary there is no second pure charge in 60 turns, so the walk is
infinite for both; under the adversary at Q = 10^3 (where its gears run out, below) the surviving
twins are 172, 77, 595 charges apart. FACT: the walk over the valves inherits no cap from any part.

### The adversary's reach (side fact from the counterfactual runs)

The greedy one-tooth adversary at Q = 10^4 empties every turn m <= 60 with 776 of 1,226 gears (q = 5;
775 of 1,225 at q = 7), 774 of them at nonzero phase; at Q = 10^3 it runs out of gears: with all 165
gears used, 19 of the 6,000 pure-imprint pairs in 60 turns survive and the first survivor 1199 is in
turn 1, so at (5, 10^3) the greedy adversary cannot empty even turn 1; at (7, 10^3) it empties turns
1..8 and the first survivor is 9809 in turn 9. Greedy gives a lower bound on the reach only; an exact
cover certificate for turn 1 at (5, 10^3) (100 pure-imprint pairs against 165 one-class gears) would
be a small ILP and was not run. The reach R(q, Q) is dead_branches_reopened_3.md's (c)(ii) and is the
root at rung Q in the exhaust's coordinate; recorded here because it is the first measurement of it.

### (d) The counterfactual test, all candidates at once

**V12 (what F violates, what the adversary violates, and what survives; EXACT at (5, 10^4), (7, 10^4)).**

| property of the fuel / charge set | real | F | adversary | count or set? |
|---|---|---|---|---|
| balance on every engine tooth (fuel per reduced class) | 24,284-24,348 (gear 3), 12,133-12,189 (gear 5) | 0 / 80,000 on gear 3: VIOLATED | 17,304-17,334 (3), 8,505-8,830 (5): has it | count |
| mirror closure of the family inventory | 122 of 123 | 0 of 54: VIOLATED | 114 of 118: has it | set |
| mirror symmetry of the pure classes | pairs within sqrt(N) | 0 = 0 | 0 = 0 | count |
| one tooth per manifold prime (no fuel member has a manifold factor) | 0 members with a manifold factor | 115,803: VIOLATED | 0: has it | set |
| phase zero at every manifold prime (the tooth is the class 0) | yes | no: VIOLATED | 774 gears at nonzero phase: VIOLATED | set |
| the pigeonhole premise (a run of F_odd + 1 open odd numbers) | never met (ceiling) | never met | never met | - |
| the walk to the next pure charge bounded | no (geometric) | infinite | infinite | - |
| a pure charge in every turn m <= 60 | yes, min P_m = 65 | no, P_m = 0 | no, P_m = 0 | count |

F violates four properties (balance, mirror closure, one-tooth, phase zero) and every one of the
three set properties among them is also possessed by the adversary except phase zero. So the only
invariant that the real charge set has and both adversaries lack is phase zero: the manifold's strike
at every prime p in (q, Q] is the class 0, i.e. the fuel's complement is exactly the numbers with a
prime factor in (q, Q], i.e. the fuel is the primes above Q. It is a property of the SET (which
residue each gear removes), not a count of primes, and the pre-registration's prediction that its
consequence for existence is the root holds: the pure charge is monotone in the fuel, dropping gears
enlarges the fuel, so the real fuel is the minimal zero-phase fuel and "the minimal zero-phase fuel
has a member on the pure imprint in every zone" is the conjecture. One more separating fact, for the
record, and it is a count: the phase-zero sieve wastes strikes (two gears' teeth coincide on the
multiples of their product), so the real fuel with all 1,226 gears has 48,632 members in 60 turns
while the adversary with 776 gears at chosen phases has 34,638; the adversary is thinner than the
primes and still empties every turn. O4 refuted as predicted.

## Verdict

1. The freedom of the turn is exact and it is a freedom of RUNG, not of statement (V6, V7): existence
   in the valve at Q is the disjunction over m of the top-slice statements of the machines {5..y_m},
   y_m in [sqrt(2Q), Q]; the disjunction equals the engine's window statement at rung Q less the
   sliver; no plain window statement at any single rung implies it. The weakest engine statement
   implying it for all Q is the position form (the run of {5..y_m} ending at the slice's top is shorter
   than Q/6 columns for some m), which is the turn statement renamed: ROOT. The share form (F(y_m) <=
   c_m(Q), the record law S < 1/2, 1/3, 1/4 at the rungs sqrt(2Q), sqrt(3Q), sqrt(4Q)) is strictly
   stronger: with the freedom of the turn it certifies every Q in [3, 1859] except Q = 24, 25, where
   twins exist, and the freedom is worth three integers there: ROOT (a length law, face A / E).
2. The frontier certifies turns from position facts (V8, proved with hypothesis; V9, exact): turn m
   for m < c. Proved (c = 1.25 from the ladder): turn 1 for every Q in [30, 1859], turn 2 on the upper
   parts of the rungs' ranges, turn 3 at 24 values of Q, turn 4 at Q = 9; 0 certified turns empty.
   Measured (c = 4.625, the prefix floor to rung 19,997): turns 1-4 to Q <= 8 x 10^7, agreeing with the
   truth at every Q <= 10^5 (Q_1..Q_4 = 6, 10, 26, 28), and the run that sets the constant is the last
   empty turn 5 (Q = 132..134, the twin gap 661 -> 809). This is the only existence-by-position result
   on record and it is sharp in the turn coordinate; its universal form (the frontier at every rung with
   c >= 1) is twin-Bertrand on the prefix: ROOT beyond the measured range.
3. No invariant of the charge set forces a member on the pure imprint (V10, V11, V12): the mirror is a
   symmetry of the imprint whose set consequence (inventory closure) the adversary shares; the
   pigeonhole fails at every q, by one at q = 5 with the failure realised; the walk is a geometric
   tail with no cap from any part; phase zero is the one surviving set property and its existence
   consequence is the conjecture. FACT x 3, O4 refuted.
4. What existence in the valve buys over existence in the engine's window: the rung. The root at rung
   Q becomes the record law at rung sqrt(2Q) with the constant 1/2 (or a position law there, the
   frontier with c > 1), a statement about a machine with the square root of the gears, whose ladder
   is certified to 59 and gives existence in the valve for every Q <= 1859 as a corollary; and it
   makes V5 (P_1, P_2 > 0 to 10^5) a corollary of the record law at the square roots. What it does not
   buy: any statement weaker than the localised window statement at some rung, any turn beyond the
   fourth from position facts, any invariant of the charge set, or a change of face: the wall at the
   valves remains face A (count) for the share form and face E (the local statement over-asks) for the
   position form, exactly where the_wall.md 5l/5m placed it.

Status for the tree: V6 PROVED (FACT: the descent for every turn, with the tiling); V7 PROVED
(identity; ROOT-marker for both forms, the position form being existence renamed); V8 PROVED with its
hypothesis (the certified turns; FACT to Q = 1859 proved, to 8 x 10^7 measured); V9 FACT (the tight
instance); V10 EXACT (the pigeonhole's margin); V11 MEASURED (the walk's tail); V12 EXACT (the
classification of the adversaries). Node R4.c.iii: FACT with ROOT marks; no CANDIDATE. The one object
this branch leaves that is neither a count nor a period-scale construction is the frontier's constant:
a first twin gap of the primes (73 -> 101, 661 -> 809) that stays the position-length floor of every
larger machine because new gears never create a run earlier than the runs already there. That is a
statement about how the engine BUILDS its low runs, not about the primes, and it is what the measured
c = 4.625 rests on; whether "new gears lengthen, never precede" can be proved for the prefix's runs of
length >= d_0 is the next child (it would prove turns 1-4 for all Q; it is also, by (R), a statement
about twin gaps, so it is marked ROOT-adjacent until the mechanism is separated from the count).

## Dead ends (with the refuting instance)

- The share form as a universal route, even with the freedom of the turn: false at Q = 24 and 25
  (every turn's F(y_m) exceeds c_m(Q)) while twins (29, 31), (41, 43) lie in (Q, 2Q]; asymptotically
  confined to the rungs sqrt(2Q)..sqrt(4Q) because F/(y^2/6) >= 0.278 on the ladder.
- The clean share reading S(y_m) < 1/(m + 1) as a sufficient condition: not sufficient when the
  sliver is large (rung 53, turn 3: S = 0.250 = 1/4 but F = 145 > c_3(Q) = 117..144 for Q = 702..868).
- Turn 5 from any frontier constant: refuted by Q = 132, 133, 134 (the frontier's own tight instance).
- The pigeonhole in run form: needs 7 open odd numbers at q = 5, the manifold allows 6, and 2 of the 14
  runs of 6 at (5, 10^3) hold no twin; in window form: needs 13 charge pairs per 30, the maximum is 11.
- The mirror as an existence tool: the fixed class -1 (mod q#) holds 0 charges under F and under the
  adversary; on the real set it is the poorest class at q = 5 and 7 (146,444 of a mean 146,702; 330 of a
  mean 347), within one sigma.
- A cap on the walk from A(m) or from the manifold's run laws: the per-turn maximum tracks the
  geometric scale (ratio 0.74-1.86 at every turn), and the manifold's laws cap runs of charges, not of
  burnt charges.
- E-P3's "max tau below 0.05": refuted by rung 29 (0.0833, the twin gap 883 -> 1019 under 961).

Remaining open items of the part, sorted: closed here (the descent for every turn; the two forms and
their order; the certified turns and their exact range; the tight instance; the pigeonhole margin; the
adversary classification); measurement with no structural content (the top-run share table; the
walk's tail; the adversary's reach at Q = 10^3, 10^4); root question in disguise (the position form for
all Q; the frontier at every rung with c >= 1; the minimal zero-phase fuel's pure charge); genuinely
open on the part alone (whether the prefix's frontier floor is set by the low runs alone for every
rung, i.e. "a new gear never creates a blocked run of length >= d_0 starting before the runs already
there": an engine statement about its own construction, testable rung by rung by listing the runs a
new gear creates and where; a proof would give turns 1-4 for every Q by V8).
