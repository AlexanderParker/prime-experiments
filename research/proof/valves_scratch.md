# The valves, built from scratch (scratch lane, 2026-09-07)

Clean-context lane. Sources read: README.md (Glossary and the "One word per object" entry),
research/proof/objects_ledger.md (sections 1 and 2 of ENGINE, MANIFOLD, EXHAUST only), and
research/proof/manifold_census_large.md. Nothing else in the project was read; no literature.
Scripts in research/valves/scratch/, outputs in research/valves/scratch/results/ (gitignored;
every number used is in this file).

## Pre-registered (written before any computation)

Definitions used. Quiet zone (Q, Q^2]. A number there is manifold-open iff it has no prime
factor in (q, Q], iff it is s x P with s q-smooth and P = 1 or a prime above Q (the charge:
air s, fuel P). A charge pair is a manifold-open pair (n, n+2), labelled by its family (s, s').
The engine strikes a number iff a prime <= q divides it, iff its air is > 1. The engine-and-
manifold-open pairs of the quiet zone are the twin prime pairs (exhaust cap, proved).

Predictions, each to be scored exact / measured / refuted:

- P1 (count identity). manifold-open pairs of (Q, Q^2] = sum over families N(s, s'); the
  engine leaves exactly family (1, 1), N(1, 1) = number of twin pairs with lower member in
  (Q, Q^2 - 2]. Expect 0 exceptions; at Q = 10^4 expect N(1, 1) = 440,107 and 1,510 families
  at q = 5 (the two numbers the permitted sources give), the same N(1, 1) at q = 7 and 11.
- P2 (family shape). gcd(s, s') divides 2 and s = s' (mod 2). Every family lies in exactly one
  of four PORTS by n mod 6: n = 5 (mod 6) iff gcd(s s', 6) = 1 (the column port); n = 3 iff
  3 | s; n = 1 iff 3 | s'; n even iff 2 | s (then 2 | s'). Expect 0 exceptions (one-line proof).
- P3 (imprint). The family (s, s') occupies, modulo the engine's period q#, exactly the residues
  n with n = 0 (mod p) for p | s, n = -2 (mod p) for p | s', and n not in {0, -2} (mod p) for
  every other prime p <= q; the number of such residues is prod over odd p <= q not dividing
  s s' of (p - 2). Expect containment with 0 exceptions; attainment of every residue by every
  family with enough members; the count inside one residue class NOT a clean function of
  (s, s', class) - expect fluctuations of square-root size around N(s, s') / |imprint|. In the
  column coordinate: a burnt column charge with g | s sits at the tooth k = 6^-1 (mod g), with
  g | s' at k = -6^-1 (mod g); expect 0 exceptions.
- P4 (stratum cap). In stratum m, i.e. n in (mQ, (m+1)Q], every member with fuel > 1 has air
  <= m; in the bottom stratum every burnt charge has a q-smooth member. Expect 0 exceptions.
- P5 (neighbours). The number of burnt charges between consecutive pure charges is NOT a
  function of the gap; in the bottom stratum it is 0 for all but a handful of gaps (those
  containing a q-smooth number with an open neighbour). Expect no exact law above the bottom
  stratum; expect the mean burnt count to grow linearly with the gap and with the stratum.
- P6 (first pure charge). The first twin prime above Q is not determined or bounded by p_1
  and the smooth numbers near Q; expect it to coincide with p_1 sometimes (Q = 10^4:
  10007, 10009 are both prime) and to be far from it at other Q. Measurement only.
- P7 (what the engine does per family). Every family with s s' > 1 is burnt entirely - not
  "some members": every member of the family has air, so the whole family goes. Expect 0
  survivors outside (1, 1).

Scorecard filled in below after the runs.

## Scorecard (filled after the runs)

| prediction | verdict | evidence |
|---|---|---|
| P1 count identity, (1, 1) = twins, 440,107 at Q = 10^4, 1,510 families at q = 5 | EXACT. 9 of 9 runs: pure charges = twin pairs as sets; N(1, 1) = 440,107 at Q = 10^4 for q = 5, 7, 11 and 8,134 at Q = 10^3; fuelled families 1,510 at q = 5, Q = 10^4 (the permitted source's number counts fuelled families; with the 449 ember families the total is 1,959) | section 1 |
| P2 family shape and the four ports | EXACT, 0 exceptions in 23,969,812 charge pairs (9 runs) | section 2 |
| P3 imprint containment; tooth pin in columns; per-class count not clean | containment EXACT (0 of 45,358 families outside their imprint); tooth pin EXACT (0 exceptions in 3,486,234 column charges, 9 runs); per-class counts fluctuate at square-root size (max deviation 0.25 to 2.1 root-means): the "clean function" half REFUTED as predicted | section 2 |
| P4 stratum cap; bottom stratum burnt charges carry an ember | EXACT, 0 exceptions (9 runs); the ember law extends to the SECOND turn by the same proof, 0 exceptions | sections 2, 5 |
| P5 burnt count between pure charges not a function of the gap | REFUTED-as-function as predicted: at gap 30, Q = 10^4, q = 5 the count takes 10 distinct values over 17,533 gaps; the mean grows linearly with the gap and with the turn | section 3 |
| P6 first pure charge not bounded by p_1 | MEASURED only: t_1 = p_1 in 13.8%, 12.6%, 11.4%, 9.3% of Q at scales 10^4 .. 10^7, tracking 2 C_2 / log Q; correlation of t_1 - Q with p_1 - Q is 0.04 to 0.08 | section 4 |
| P7 every family with s s' > 1 burnt entirely | EXACT (definitional): 0 survivors outside (1, 1) in 9 runs | section 1 |

Two things the pre-registration did not foresee and the data forced: the ONSET law (a family is silent
before the engine's turn max(s, s') and, for small air, fires exactly there) and the INVENTORY law
(which (s, s') exist at all: a local-solvability condition with a mod-4 clause I had wrong on the
first pass - the data refused (2, 6), and the proof followed).

## Setup

Scripts: research/valves/scratch/valves_scan.py q Q (the nine runs), first_pure.py,
ratio_law.py q Q, spokes.py. Outputs in research/valves/scratch/results/ (gitignored).

For each (q, Q) the script sieves [1, Q^2] by the primes in (q, Q] alone (the manifold), lists every
pair (n, n + 2) with Q < n <= Q^2 - 2 and both members manifold-open (the CHARGES), strips the
q-smooth part of each member (the AIR s, s') leaving the FUEL (checked to be 1 or a prime above Q:
0 failures in 23,969,812 pairs), and groups the pairs by (s, s') (the FAMILIES). The engine's
action is read off: a member burns iff its air exceeds 1. Twin primes are sieved independently.

Runs: (q, Q) = (5, 30), (7, 210), (11, 2310), where Q = q# so the engine's period is Q and the
strata of the quiet zone are the engine's turns, and (5, 7, 11) x (10^3, 10^4).

Vocabulary fixed here (one word per object; none reuses a reserved word):
- TURN m: the engine's m-th period above Q, the numbers (mQ, (m + 1)Q] (identical to the
  stratum when Q = q#; for Q = 10^k the same intervals are used and called turns loosely).
- EMBER: a q-smooth number above Q - air with no fuel, a byproduct that reached the quiet zone.
  A family with max(s, s') > Q is an ember family (its ember is its own label).
- PORT: the class of a family by n mod 6 (0 = even, 1, 3, 5 = column).
- IMPRINT: the residue set a family occupies modulo q# (raw line) or modulo q#/6 (columns).
- INVENTORY: the list of labels (s, s') that can exist at all.
- ONSET: the first member of a family, and the turn it falls in.
- YIELD: the count N(s, s') of a family in (Q, Q^2].
- SPOKE: the column of mQ, the pair (mQ - 1, mQ + 1), m = 1 .. Q - 1.

## Results

### 1. The valve set and the count identity

| q | Q | charges | families (fuelled + ember) | pure = twins | burnt | burnt column-port charges |
|---|---|---|---|---|---|---|
| 5 | 30 | 289 | 47 + 68 = 115 | 30 | 259 | 22 |
| 7 | 210 | 9,700 | 488 + 433 = 921 | 621 | 9,079 | 946 |
| 11 | 2,310 | 739,360 | 5,569 + 2,646 = 8,215 | 34,196 | 705,164 | 76,256 |
| 5 | 10^3 | 100,058 | 567 + 264 = 831 | 8,134 | 91,924 | 6,364 |
| 7 | 10^3 | 141,276 | 1,558 + 840 = 2,398 | 8,134 | 133,142 | 13,003 |
| 11 | 10^3 | 171,943 | 2,991 + 1,846 = 4,837 | 8,134 | 163,809 | 18,327 |
| 5 | 10^4 | 5,376,501 | 1,510 + 449 = 1,959 | 440,107 | 4,936,394 | 331,606 |
| 7 | 10^4 | 7,765,397 | 5,586 + 1,750 = 7,336 | 440,107 | 7,325,290 | 685,078 |
| 11 | 10^4 | 9,665,288 | 14,187 + 4,559 = 18,746 | 440,107 | 9,225,181 | 975,062 |

Identity: charges = sum over families of N(s, s'); engine-and-manifold-open = N(1, 1) = the twin
pairs with lower member in (Q, Q^2 - 2], checked as sets (not just counts) in all nine runs. The
engine burns every family except (1, 1) entirely: a family is one air label, every member of it has
that air, so "some member has air" and "every member has air" coincide. The census overlap:
N(1, 1) = 440,107 at Q = 10^4 (the permitted census file's number), the quiet-zone record 420 after
26,261 at q = 5 (the census's), and the 1,510 fuelled families at q = 5 (the ledger's L60 number).
The record moves with q at Q = 10^4: 420 after 26,261 (q = 5), 372 after 18,539 (q = 7), 270
after 14,867 (q = 11) - more families, more charges inside the twin gaps. No gap of 4 in any run.

Ports (n mod 6), q = 5, Q = 10^4: even 2,899,788; port 1 (3 | s') 852,467; port 3 (3 | s)
852,533; column port 771,713 of which 440,107 pure. The two 3-ports are mirror images and agree to
66 in 852,000.

The other view (the manifold acting on the engine's openings): engine-open columns in the quiet
zone 9,999,000 (q = 5), 7,142,143 (q = 7), 5,843,571 (q = 11) at Q = 10^4, of which 440,107 are
manifold-open in every case. The manifold strikes 95.6%, 93.8%, 92.5% of the engine's openings;
the struck ones are exactly the columns with a q-rough composite member.

### 2. Timing: ports, imprints, teeth, and the engine's turns

Ports. gcd(s, s') divides 2 and s = s' (mod 2) in every family (0 exceptions, 45,358 families).
Port law: n = 5 (mod 6) iff gcd(s s', 6) = 1; n = 3 (mod 6) iff 3 | s; n = 1 (mod 6) iff 3 | s';
n even iff 2 | s. 0 exceptions in 23,969,812 pairs. Proof: 3 | n iff 3 | s (the fuel is coprime to
6); 3 | n + 2 iff 3 | s'; otherwise n and n + 2 are both nonzero mod 3, which forces n = 2 (mod 3),
and with n odd, n = 5 (mod 6).

Imprints on the raw line. For a family (s, s') let R(s, s') be the residues n mod q# with
n = 0 (mod p) for p | s, n = -2 (mod p) for p | s', and n not in {0, -2} (mod p) for the other
primes p <= q. Every member of the family lies in R(s, s'): 0 exceptions, 45,358 families.
|R(s, s')| = product over odd p <= q not dividing s s' of (p - 2) (the prime 2 contributes one
class in every case). The family (1, 1) has |R| = 3, 15, 135 at q = 5, 7, 11 - the engine's own
opening count per period. Attainment (every residue of R occupied): 1,877 of 1,959 families at
q = 5, Q = 10^4; 6,057 of 7,336 at q = 7; 10,734 of 18,746 at q = 11 - the shortfall is small
families and ember families (one member each).

Imprints in the column coordinate (the tooth pin). For a column-port charge in column
k = (n + 1)/6 and a gear g in 5..q: if g | s then k = 6^-1 (mod g), the tooth the engine strikes on
6k - 1; if g | s' then k = -6^-1 (mod g); if g divides neither, k is on neither tooth. 0 exceptions
in 771,713 + 1,125,185 + 1,415,169 column charges at Q = 10^4 and in the six smaller runs. So a
burnt column charge sits exactly on the teeth of the gears that burn it, one tooth per gear, the side
decided by which member carries the gear; the gears that do not burn it see it as an ordinary
opening. In the engine's period q#/6 a burnt column family therefore occupies exactly
prod_{g in 5..q, g not dividing s s'} (g - 2) columns, and the family (1, 1) the prod (g - 2)
engine openings.

Counts inside one residue class are not clean. For the large families the per-class counts
scatter at square-root size: family (1, 1) at q = 11, Q = 10^4 over its 135 classes has mean
3,260.1, min 3,157, max 3,356 (max deviation 1.80 root-means); (1, 3): 2,318.2, 2,227 .. 2,402
(1.89); (1, 5) over 45 classes: 2,879.8, 2,775 .. 2,944 (1.95); (1, 7) over 27: 3,149.3,
3,032 .. 3,267 (2.10); at q = 7 the deviations are 0.29 to 1.73, at q = 5 0.25 to 0.78 (and 0.00
for the one-class families (1, 5), (1, 15), (1, 25)). Refuted as a clean function, as pre-registered.

The engine's turns as the clock (ONSET). A fuelled member n = sP has P > Q, so n > sQ: the
family (s, s') has no member below max(sQ, s'Q - 2), i.e. it is silent before turn max(s, s').
Exact: 0 negative delays in 9 runs (delay = onset turn - max(s, s')). Where the family is dense it
fires exactly at that turn: all families with max(s, s') <= 25 have delay 0 at q = 5, Q = 10^4
(<= 24 at q = 7, <= 21 at q = 11; <= 10 at Q = 2310, <= 8 at Q = 10^3, <= 6 at Q = 210, <= 3 at
Q = 30). Onsets at q = 5, Q = 10^4: (1, 1) at 10,007; (3, 1) at 30,027; (1, 3) at 30,109; (2, 4)
at 40,706; (4, 2) at 40,244; (1, 5) at 50,033; (5, 1) at 50,045; (5, 3) at 50,185; (3, 5) at
50,493; (4, 6) at 60,052; (6, 4) at 60,402; (2, 8) at 81,062; (8, 2) at 80,072; (8, 6) at 81,544;
(1, 9) at 90,547; (9, 1) at 90,351; (9, 5) at 90,063; (5, 9) at 92,005; (4, 10) at 100,388;
(8, 10) at 100,088; (10, 4) at 100,610; (10, 8) at 101,510; (2, 12) at 120,082; (10, 12) at 120,730;
(12, 2) at 120,444; (12, 10) at 120,108. Delay histogram over the 1,510 fuelled families: 210 at 0,
78 at 1, 58 at 2, 42 at 3, 40 at 4, 25 at 5, then a long tail (the family's window in turn
max(s, s') has length Q/s and the family's density there is about 1/(phi(s') log^2 Q), so large
air misses its first turn).

Consequently the families present in turn m are exactly the INVENTORY's pairs with max(s, s') <= m,
plus embers: at Q = 10^4 this holds with equality for every m = 1 .. 12 and every q (fuelled
family counts 1, 1, 3, 5, 9, 11, 11/17, 15/21, 19/27, 23/31, 23/31/41, 27/35/45 for q = 5/7/11);
at Q = 10^3 it holds to m = 8 and fails at m = 9 by the missing (9, 5) (onset 10,053, delay 1). In
turns 1 and 2 the only fuelled family is (1, 1).

### 3. The pure charge's neighbours

Offsets. A twin t = 6j - 1 has the pair at t - 2 equal to (3(2j - 1), t), family (3a, 1), and at
t + 2 equal to (t + 2, 3(2j + 1)), family (1, 3a'); offsets +-1, +-3, +-5 are even pairs; offsets
+-6 are the neighbouring columns. Exact by the port law. Measured at q = 5, Q = 10^4 (440,107
twins): a burnt charge at t - 2 for 95,390 twins (family (3, 1) 39,353, (15, 1) 17,532, (9, 1)
14,084), at t + 2 for 95,983 ((1, 3) 39,463, (1, 15) 17,690, (1, 9) 14,148), at t - 1 for 19,319,
t + 1 for 19,245, t - 3 for 17,177 ((4, 2) 2,705, (2, 4) 2,590), t + 3 for 17,444.

Nearest burnt charge (either side): distance 1: 35,472 twins; 2: 161,552; 3: 17,294; 4: 0;
5: 21,999; 6: 11,225; 7: 24,813; 8: 12,737; 9: 15,300; 10: 9,499; 11: 14,127; 12: 3,323. Distance 4
never occurs: if t and t + 4 are open pairs then t + 2 is one (the manifold's no-gap-4 rule, which
the engine cannot undo since it only removes). 37% of twins have a burnt charge at distance 2.

Between consecutive pure charges. The burnt count is not a function of the gap: at q = 5,
Q = 10^4, gap 6 (4,756 cases) has mean 0.44 and 4 distinct values; gap 12 (12,534) mean 0.74, 7
values; gap 18 (9,135) 1.15, 7; gap 24 (5,718) 1.36, 8; gap 30 (17,533) 1.76, 10; gap 36 (5,086)
2.01, 10; gap 42 (15,962) 2.26, 11; gap 48 (8,589) 2.51, 12. At q = 7 the means are 0.70, 1.09,
1.52, 2.04, 2.43, 2.91, 3.31, 3.79; at q = 11 0.83, 1.32, 1.89, 2.43, 3.04, 3.55, 4.13, 4.59. The
mean is linear in the gap; what multiplies it is the burnt density, which rises with the turn:
burnt charges per unit length between consecutive twins, q = 5, Q = 10^4, turns 1 .. 12: 0.0003,
0.0012, 0.0205, 0.0254, 0.0340, 0.0388, 0.0377, 0.0412, 0.0494, 0.0522, 0.0501, 0.0580; the
fraction of twin gaps with no burnt charge inside: 0.978, 0.904, 0.290, 0.219, 0.198, 0.128, 0.167,
0.101, 0.148, 0.094, 0.163, 0.097. The jump between turns 2 and 3 is the onset of (3, 1) and
(1, 3), the first fuelled burnt families.

Turns 1 and 2 exactly. Every burnt charge there has an ember member (proof: air <= 2 by the cap;
air pairs (1, 2), (2, 1) break parity, (2, 2) needs P' = P + 1; so a burnt pair has a member with
fuel 1, which is q-smooth and above Q). Count: burnt charges in turn 1 are at most 2 x embers in
(Q, 2Q]. Measured (embers / burnt): turn 1 at Q = 10^4: 37 / 3 (q = 5), 89 / 11 (q = 7), 160 / 27
(q = 11); turn 2: 21 / 12, 56 / 30, 106 / 57; at Q = 2310, q = 11: 99 / 22 and 70 / 43; at
Q = 210, q = 7: 28 / 9 and 19 / 15; at Q = 30, q = 5: 8 / 4 and 6 / 6. The bottom-turn lists at
Q = 10^4, q = 5: (10935, 10937) with 10935 = 3^7 x 5, (18223, 18225) with 18225 = 3^6 x 5^2,
(19681, 19683) with 19683 = 3^9; at q = 7 add 11025 = 3^2 5^2 7^2, 11907 = 3^5 7^2, 12005 = 5 7^4,
13125 = 3 5^4 7, 14175 = 3^4 5^2 7 (both sides prime: 14173, 14175, 14177), 15309 = 3^7 7,
19845 = 3^4 5 7^2. Consequently the manifold's record in turns 1 and 2 is a twin gap unless an
ember with a prime neighbour lands inside it (the census's W103, rederived from the definitions).

### 4. The first pure charge

t_1(Q) = first twin above Q, p_1 = nextprime(Q). Exact: t_1 >= p_1 (a twin's lower member is
prime). At Q = q#, (q, p_1 - Q, t_1 - Q) = (5, 1, 11), (7, 1, 17), (11, 1, 29), (13, 17, 59),
(17, 19, 41), (19, 23, 41), (23, 37, 641), (29, 61, 101), (31, 1, 419), (37, 61, 101), (41, 71,
269), (43, 47, 179), (47, 107, 107). Only at 47# is the first prime the first twin. The primorial
column (Q - 1, Q + 1) is twin at 5# and 11# only. At Q = 10^k, k = 3 .. 12: p_1 - Q = 9, 7, 3, 3,
19, 7, 7, 19, 3, 39 and t_1 - Q = 19, 7, 151, 37, 139, 37, 7, 277, 817, 61 (t_1 = p_1 at 10^4 and
10^9). In every one of these 33 cases the first charge above Q is the pure one, except Q = 210
where the charges (223, 225) and (225, 227) precede the twin (227, 229): the ember 225 = 15^2 sits
between two primes. Over 100,000 consecutive Q at each scale: P(t_1 = p_1) = 0.138, 0.126, 0.114,
0.093 at 10^4, 10^5, 10^6, 10^7 (2 C_2 / log Q = 0.143, 0.115, 0.096, 0.082); mean (t_1 - Q) /
log^2 Q = 0.93, 0.73, 0.63, 0.68; variance / mean^2 = 0.93, 1.07, 0.85, 0.96 (memoryless to first
order); correlation of t_1 - Q with p_1 - Q is 0.056, 0.062, 0.084, 0.041. The manifold's structure
at Q (p_1, the embers near Q) does not determine or bound t_1 beyond t_1 >= p_1; measurement only.

### 5. Inventory and yield

INVENTORY. Which labels (s, s') exist at all? The equation s'P' - sP = 2 must be solvable with
P, P' coprime to every prime p <= q. Necessary and sufficient: gcd(s, s') | 2, s = s' (mod 2), and
if both are even then s/2 + s'/2 is odd (4 divides exactly one of them). Proof: for odd p <= q,
if p | s then P' = 2/s' (mod p) is a unit, if p | s' then P = -2/s is a unit, if p divides neither
then two choices of P give two residues of P' of which at most one is 0; if p | both, no solution.
For p = 2: same parity is forced; with s = 2a, s' = 2b the equation is bP' - aP = 1 with P, P' odd,
so a + b is odd, and then P' = (aP + 1)/b lifts to every power of 2. Measured: realised fuelled
families are all admissible (0 exceptions in 9 runs), and every admissible pair with
max(s, s') <= 1,500 is realised at q = 5, Q = 10^4 (<= 1,350 at q = 7, <= 1,200 at q = 11, <= 243
at Q = 2310, <= 250 / 216 / 121 at Q = 10^3, <= 50 at Q = 210, <= 9 at Q = 30); the first missing
pairs at Q = 10^4 are (250, 2048), (486, 5000), (500, 4374), (729, 3125) - large s' leaves the
fuel P too short an interval. Admissible pairs with both members <= Q: 1,715 / 6,779 / 18,575 at
q = 5 / 7 / 11, of which 1,510 / 5,586 / 14,187 are realised. The family (2, 2) is impossible
(P' = P + 1) and so is every pair with both members 2 mod 4, e.g. (2, 6), (6, 10), (2, 18).

YIELD. The count of a family follows the local density its imprint prescribes: N_pred(s, s') =
(1 / phi(b)) C(s, s') integral from P_lo to Q^2/s of dP / (log P log((sP + 2)/s')) with b = s'
for odd s' and s'/2 for even, P_lo = max(Q, (s'Q - 2)/s), and C = 2 C_2 prod_{odd p | s}
(p - 1)/(p - 2) prod_{odd p | s'} (p - 1)^2 / (p (p - 2)), times 1/2 when s = 2 (mod 4). At
q = 11, Q = 10^4 over the 310 fuelled families with N >= 5,000: ratio measured / predicted has mean
0.9986, min 0.980, max 1.020. The largest: (1, 1) 440,107 vs 440,154; (3, 1) 313,608 vs 313,271;
(1, 3) 312,952; (1, 5) 129,590 vs 129,344; (5, 1) 128,924; (2, 4) 125,011 vs 124,605; (4, 2)
124,660; (1, 9) 112,132 vs 111,897; (15, 1) 92,508 vs 92,542; (5, 3) 92,180 vs 92,117; (2, 12)
89,352 vs 89,092; (1, 7) 85,032 vs 84,941; (11, 1) 51,548 vs 51,527. The mirror pairs (s, s') and
(s', s) have the same prediction and differ by square-root amounts (313,608 vs 312,952).

SPOKES. When q# | Q the numbers mQ +- 1 are coprime to q#, so the column of mQ is engine-open for
every m: an always-present engine object in the quiet zone (Q copies of the engine's column 0).
Twin spokes among m = 1 .. Q - 1: 11 at 5# (heuristic 4 prod_{3<=p<=q} p^2/(p-1)^2 prod_{p>q}
p(p-2)/(p-1)^2 / log^2(mQ) summed over m: 12.1), 37 at 7# (42.7), 248 at 11# (252.8), 2,097 at
13# (2,102.1), 10,303 at 17# for m <= 200,000 (10,266.6). The manifold decides each spoke; nothing
forces one.

## Interface objects

1. PORT. Definition: the class of a family by n mod 6. Coordinate: raw line (the column port is
   the column coordinate's domain). Law: port 5 iff gcd(s s', 6) = 1, port 3 iff 3 | s, port 1
   iff 3 | s', port 0 iff 2 | s (and then 2 | s'). Count: 0 exceptions in 23,969,812 pairs.
   Proof: above. Corollary: the column coordinate sees exactly the families with air coprime to 6
   (11 / 74 / 302 fuelled families at q = 5 / 7 / 11, Q = 10^4).
2. IMPRINT. Definition: the residue set of a family modulo the engine's period. Coordinate: both
   (raw line mod q#; columns mod q#/6). Law: containment in R(s, s'), |R| = prod over odd p <= q
   not dividing s s' of (p - 2); in columns, each burning gear pins the family to the one tooth on
   the side it divides. Count: 0 exceptions in 45,358 families and 3,486,234 column charges.
   Proof: CRT, one line per prime. The mirror n -> -n - 2 carries R(s, s') to R(s', s).
3. TURN and ONSET. Definition: turn m = (mQ, (m + 1)Q]; onset = a family's first member. Coordinate:
   raw line; the turn is the engine's own period when Q = q#. Law: no member before turn
   max(s, s'); the air of any fuelled member in turn m is at most m; in turns 1 and 2 the only
   fuelled family is (1, 1). Count: 0 exceptions in 9 runs. Proof: n = sP > sQ. Measured: delay 0
   for all families with max(s, s') <= 25 at Q = 10^4; families present in turn m = admissible
   pairs with max <= m for m <= 12 exactly.
4. EMBER. Definition: a q-smooth number above Q. Coordinate: raw line. Law: every burnt charge in
   turns 1 and 2 has an ember member; burnt charges in turn 1 number at most twice the embers in
   (Q, 2Q]. Count: 0 exceptions, 9 runs (3 of 37, 11 of 89, 27 of 160 embers carry a charge at
   Q = 10^4). Proof: above. The number of embers in (Q, 2Q] is at least 1 (the power of 2 there)
   and grows like (log Q)^{pi(q) - 1}.
5. INVENTORY. Definition: the labels (s, s') that can exist. Coordinate: raw line. Law: q-smooth
   s, s' with gcd | 2, same parity, and 4 dividing exactly one of an even pair. Count: 0 realised
   families outside it (9 runs); all admissible pairs realised to max(s, s') = 1,500 at Q = 10^4.
   Proof: local solvability, above. Exact in both directions only up to a height that grows with
   Q; the "all realised" half is a measurement.
6. YIELD. Definition: N(s, s'). Coordinate: raw line. Law: the imprint's local density integrated,
   to within 2% over 310 families (mean ratio 0.9986). Measured, no proof; the closed form is the
   product C(s, s') / phi(b) above.
7. SPOKE. Definition: the column of mQ. Coordinate: columns. Law: engine-open for every m when
   q# | Q (proof: mQ +- 1 is coprime to q#). Count: twin at 11 / 37 / 248 / 2,097 / 10,303 spokes at
   5# .. 17#, matching the local heuristic to 3% from 11# on.

## What would have to break each

- PORT, IMPRINT, tooth pin: a fuel prime sharing a factor with the engine - impossible by the
  definition of fuel (a prime above Q > q). Nothing short of that.
- TURN/ONSET lower bound and the air cap: a fuel prime at or below Q. Impossible by definition.
  The equality half (delay 0) breaks as soon as Q / (s phi(s') log^2 Q) is of order 1: at Q = 10^3
  the family (27, 25) misses its turn by 10; it is a density, not a law.
- EMBER law in turns 1, 2: a fuelled charge with air 2 on both sides, i.e. consecutive odd primes
  P, P + 1 - impossible. A third-turn version fails: (3, 1) fires in turn 3.
- INVENTORY: a realised family with 4 dividing both halves or neither of an even pair, or a common
  odd factor - each contradicts a congruence. The "all realised" half breaks at max(s, s') = 1,500
  at Q = 10^4, where the fuel's interval is too short, and is a measurement.
- YIELD: a 2% law over 310 families; a single family off by 10% would break it, none is.
- SPOKE openness: Q not a multiple of q#. Its twin count is a heuristic.
- Count identity and pure = twins: nothing; it is the exhaust cap read family by family.

## Verdict

Built from the definitions alone, the valves are: a partition of the manifold's open pairs into
families (s, s') with an exact inventory (a congruence condition), an exact placement (each family
on its imprint, burnt families pinned to the burning gears' teeth), an exact clock (a family is
silent before the engine's turn max(s, s'), and turns 1 and 2 carry nothing fuelled but the pure
charge), and a measured yield (each family's count is its imprint's density integrated, to 2%).
The engine's action is the simplest possible: it removes every family but (1, 1), and the families
it removes are all of these things. Everything exact here is a congruence or an inequality of
size; nothing exact says the family (1, 1) has a member in a given turn. The objects "always in
the window because the machine works this way" that this lane can point at are the SPOKE
(engine-open, Q copies of column 0 in the quiet zone) and the EMBER (a byproduct in every
(Q, 2Q]); whether a spoke or the neighbour of an ember is prime is the manifold's decision, and
the data says it is made at the local density and nowhere else. The one exact interface statement
with content beyond congruences is negative: in the engine's first two turns above Q, the only
fuelled family is the pure one, so the manifold's record there is a twin gap, shortened only by
embers.

## Dead ends

- Per-residue-class counts as a clean function of (s, s', class): square-root scatter (1.8 to 2.1
  root-means at q = 11). The imprint is exact as a set, not as a measure.
- The burnt count between consecutive pure charges as a function of the gap: 10 distinct values at
  gap 30. Only its mean is linear.
- The mirror (s, s') <-> (s', s) as an exact symmetry of counts on the range: 313,608 vs 312,952.
  It is exact only on imprints.
- p_1 or the embers near Q bounding the first twin: correlation 0.04 to 0.08; the only exact
  relation is t_1 >= p_1.
- A two-sided ember law past turn 2: (3, 1) and (1, 3) enter at turn 3.
- The first pass at the inventory (without the mod-4 clause) predicted (2, 6), (6, 2), (2, 10) ...
  and the data refused them in every run; the proof then produced the clause. Recorded as the one
  place the pre-registration was wrong.

## Prior art (one line per object, named only now)

- PORT: the residue classes of n mod 6 - elementary.
- IMPRINT: the admissible residue classes of a linear pair (sP, s'P') - the local factors of the
  Hardy-Littlewood / Bateman-Horn singular series.
- INVENTORY: local solvability of s'y - sx = 2 in units - the admissibility condition for the
  linear forms (P, (sP + 2)/s') in Dickson's conjecture / Bateman-Horn; the family (4, 2) is the
  Sophie Germain primes, (2, 4) the safe primes' halves.
- YIELD: the Hardy-Littlewood conjecture for the pair (P, (sP + 2)/s') with the 1/phi(b)
  arithmetic-progression factor (the Bateman-Horn heuristic); the twin case is the pair (1, 1).
- EMBER: a q-smooth number above Q; the pairs (ember, prime at distance 2) are Stormer-type
  smooth neighbours of primes.
- TURN/ONSET: elementary (sP > sQ); no standard name known to me.
- SPOKE: the pairs m q# +- 1 - primorial-multiple twins; no standard name known to me.
- First pure charge: the first twin above Q - the twin analogue of the next-prime function;
  P(t_1 = p_1) tracks 2 C_2 / log Q (Hardy-Littlewood).
