# The core's real-phase leftover (branch R4.d.i, continued)

Prover, round 5 of the stack line, 2026-09-09. Parent: R4.d.i (the base case and the step), spawned
by the observation of research/proof/step_evidence.md section 6: over every stretch of L slots (L
the section's record) the core (gears <= 6L + 1) leaves K unstruck slots, typically 20 (max 56) and
12 at the record stretch on the base-3 section [16129, 260,467,321); the tail (gears in (6L + 1,
cut)) strikes at most one slot each per stretch, typically 199 and 181 at the record; supply
exceeds demand in 100% of stretches and exactly one stretch is twin-free. The manager's reading:
supply against demand cannot close the step; what would have to be bounded is the core's minimum
leftover over positions, the real-phase record of the core, which the loaded record rule
(research/proof/top_machine_7.md L69 = W88, kernel TopMachine.loaded_record_rule) bounds only with
free phases. Scripts research/stack/r5/core_leftover.py; outputs research/stack/r5/results/
(untracked; every number the text uses is in the text). Nothing here is committed by the prover.

Vocabulary: the raw line; a SLOT is (n, n + 2) with n = 5 mod 6, indexed by j with n = n0 + 6j from
the section's first slot n0; a section is [c, c') with c' the square of the first prime at or above
c; a STRETCH of L slots is L consecutive slots, and a RUN is a maximal stretch of struck slots; the
CORE for a length L is the set of gears (primes) in [5, 6L + 1] and the TAIL the primes in (6L + 1,
c) (a tail gear has at most one multiple among the 6L numbers a stretch spans, so strikes at most
one of its slots); a gear strikes the slots whose members it divides (n = 0 or n = -2 mod g). The
core's LEFTOVER on the stretch starting at slot x is K_L(x) = the number of its L slots struck by no
core gear. K_L is periodic in x with period the product of the core gears.

Laws are numbered S10 onward: the brief said S8 onward, but stacked_squares.md section 6 already
issues S8 (the cycle at q#) and S9 (the machine count), and base_and_step.md reserved S10 without
issuing any; the register must not carry two S8s.

Prior results checked before opening (docs/novel/README.md wheels-core-tail and
manifold-loaded-record-rule entries; top_machine_7.md L67-L70; top_machine_8.md W101; the tree
R4.b.ix, R4.b.xi, R4.d, R4.d.i; step_evidence.md sections 5-6): the loaded record rule is on record
as an iff over FREE core phasings (kernel); the core minimisation is a scan of the core period and
the two parity classes cannot be decoupled (W101); the composite record per section is on record
(step_evidence.md section 5); supply against demand is on record (section 6). Nothing on record
computes the real-phase leftover K_L(x) as a function on a section, its minimum, or its law in L.
What this branch can find that is not known: the value and position of the real-phase minimum
leftover on a section against the free-phase minimum; the law of that minimum in L and the length
L0 at which the core alone stops covering a stretch (the core's own real-phase record on the
section); whether the minimum has any content beyond a count (an extreme value of a sum of L
indicators over the section's positions); and whether the recursion (the core's gears are the
survivors of the sections below, i.e. the primes) leaves a trace in the minimum that a counterfactual
gear set or phasing lacks.

## 1. Pre-registered (written before any script ran; verdicts filled in afterwards)

### Theory

T1 (one phasing = one position). A free phase vector of the core is, by CRT, one residue of x modulo
the core period P_core = prod(core gears). The real phases (every gear strikes its multiples) are
the phase vector of x = 0, and the section [c, c') is the stretch of positions [c, c') of the core's
period, which begins at the origin. So "the real phases are one phasing" is exact: the real-phase
leftover on the section is the free-phase leftover function evaluated along one stretch of length
c' - c of a period of length P_core, the stretch that begins c numbers after the origin. A
random-phase counterfactual is the same function evaluated along a stretch beginning at a random
point of the period. The only thing that distinguishes the real section from a random stretch is
its distance from the origin: below (6L + 1)^2 every core-unstruck slot is a twin prime (the exhaust
cap X6, proof_skeleton section 5), above it the core-unstruck slots are the (6L + 1)-rough pairs.

T2 (the free minimum is 0 far beyond the record). The free-phase minimum of the leftover over ALL
phasings is 0 for every L up to the core's FREE record (the longest stretch some phasing of the
core covers alone), and that record is far above the section's record L*: a greedy phasing of the
core (each gear in turn takes the residue that strikes the most still-open slots) covers a stretch
several times L*. Predicted: greedy with core(L*) = primes in [5, 6L* + 1] covers a stretch of at
least 3 L* slots on all three sections (base 3: >= 1,737; base 7: >= 762; base 23: >= 459). Refuted
if greedy fails to cover even L* with core(L*).

T3 (the real minimum is an extreme value of a count). K_L(x) is a sum of L indicators (slot x + i is
core-unstruck) whose mean is fixed by the core's density p(L) = prod over the core of (1 - 2/g)
(Mertens: about 0.88 (3 e^-gamma / ln(6L + 1))^2, i.e. 0.035 at L = 579) and whose variance is BELOW
binomial because the small gears' strikes are periodic (gear 5 strikes exactly 2 of every 5
consecutive slots, so contributes no variance at all to a count over a stretch of length a multiple
of 5). The minimum over the section's N = c' - c positions is then the extreme value of that count:
the smallest k at which N times P(K_L <= k) reaches 1. Predicted at L* = 579 on the base-3 section:
mean K = 20.4 (on record), variance ratio (measured / binomial) between 0.5 and 0.85, minimum
between 1 and 4, attained at fewer than 10 positions, and NOT at the record stretch (K = 12 there,
which sits above the 5th percentile). Refuted if the minimum is 0 (impossible: a K = 0 stretch of
579 is twin-free and would be the record, which has K = 12), if the minimum is 8 or more, or if
the variance ratio is above 1.

T4 (the law of the minimum in L). With the core adjusted to gears <= 6L + 1, min over x of K_L(x)
is 0 exactly while L <= L0, where L0 is the largest L such that the core's own real-phase record
R(6L + 1) (the longest run of slots struck by the primes <= 6L + 1 inside the section) is at least
L; and for L > L0 the minimum grows, on the base-3 section, roughly like L p(L) - z sqrt(v(L)) with
v(L) the measured variance and z about 5.5 (N about 4 x 10^7). Predicted L0 between 300 and 450 on
the base-3 section; min K_2000 between 12 and 20. Refuted if L0 < 200 or > 579 (579 is impossible
by the argument in T3), or if min K_2000 is below 8 or above 30.

T5 (the record against the minimum). The record stretch L* is not where K is smallest; it is a
stretch where K is low (below average) AND every one of its K leftovers is struck by a tail gear.
Predicted: on the base-3 section the number of positions with K_579 <= 12 exceeds 10^5, so the
record stretch is one of more than 10^5 stretches at or below its leftover; the tail's finish at
the record is exactly K = 12 of 12 (twin-free); at every position where K_579 attains its minimum
the tail leaves at least one leftover unstruck (else it would be twin-free). Refuted if fewer than
10^4 positions have K <= 12.

T6 (the recursion's content in the minimum: none). (a) Random-phase counterfactuals of the same
core (each gear's two teeth shifted by a random residue) give min K_L within +-3 of the real one at
every L on the grid, and the real section's minimum lies inside the spread of 3 random seeds;
refuted if the real minimum is above every seed's by 4 or more at three or more L values, or below
by the same. (b) A pairwise-coprime replacement of the core by integers of the same sizes does not
exist: any pairwise coprime set of 485 integers in [5, 3475] coprime to 6 uses 485 disjoint
nonempty sets of primes <= 3475, of which there are exactly 485, so every member is a prime power
and the set is the primes up to prime-power substitutions (a fact, stated as a law below). The
counterfactual that exists is a NON-coprime random integer set of the same sizes striking its
multiples; predicted: its leftover mean is higher (shared factors waste strikes: about 10-30%
higher at L = 579) and its minimum is higher by the same proportion, i.e. the difference is in the
mean, not in a property of the minimum. Refuted if the non-coprime set's minimum is lower than the
real one at L = 579.

T7 (the brief's prediction, put on the scorecard as the owner's/manager's). The real minimum over a
section is far above the free-phase minimum, and it grows with L while the section samples many
periods of the small gears but a vanishing fraction of the core's period.

### Scorecard (filled in section 6)

| # | prediction | refuted by | verdict |
|---|---|---|---|
| P1 (T2) | greedy free phasing of core(L*) covers >= 3 L* on all three sections; free minimum 0 at L* | greedy fails to cover L* | |
| P2 (T3) | base 3, L = 579: min K in [1, 4], attained at < 10 positions, not at the record | min = 0 or >= 8, or > 10 positions | |
| P3 (T3) | variance ratio measured / binomial in [0.5, 0.85] at L = 579 | ratio > 1 or < 0.4 | |
| P4 (T3) | the extreme-value form (smallest k with N P(K <= k) >= 1, from the measured histogram's own tail model) predicts the minimum within +-2 | off by 3 or more | |
| P5 (T4) | L0 in [300, 450] on the base-3 section | L0 < 200 or > 579 | |
| P6 (T4) | min K_2000 in [12, 20] | < 8 or > 30 | |
| P7 (T5) | positions with K_579 <= 12 exceed 10^5 | fewer than 10^4 | |
| P8 (T6a) | random-phase seeds bracket the real minimum within +-3 at every grid L | real above or below all seeds by >= 4 at >= 3 L values | |
| P9 (T6b) | the non-coprime integer set has higher mean and higher minimum at L = 579 | its minimum is lower than the real one | |
| P10 (T7, the brief's) | real minimum far above the free minimum and growing with L | real minimum 0 at L*, or not increasing over the grid | |
| P11 (T1) | the real section's minimum positions lie above (6L + 1)^2 (the generic part), not in the quiet part where leftovers are twins | all minimum positions below (6L + 1)^2 | |

## 2. Setup

Sections (the chain from a base: c_1 = base, c_{k+1} = p_k^2 with p_k the first prime at or
above c_k). Slots are the n = 5 mod 6 with n in the section and n + 2 below the next cut (the
slot (c' - 2, c') holds the next cut itself, a square by S1, and is left out; supply_demand.py kept
it, so its slot counts are one higher).

| section | slots S | record L* | core gears (<= 6L* + 1) | tail gears | sieve limit |
|---|---|---|---|---|---|
| base 3, [16129, 260,467,321) | 43,408,531 | 579 | 485 (<= 3475) | 1,390 | 260,467,361 |
| base 7, [2809, 7,946,761) | 1,323,991 | 254 | 239 (<= 1525) | 168 | 7,946,801 |
| base 23, [529, 292,681) | 48,691 | 153 | 97 (<= 919) | 0 | 292,721 |

The object as computed. For every slot j the smallest gear striking it, m[j] = min{p prime >= 5,
p < cut : p | n or p | n + 2}, sentinel if none (then both members are prime: the slot is a twin,
checked slot by slot against the sieve on all three sections, 0 mismatches once the square slot is
excluded). The core for a length L is the primes <= 6L + 1, so slot j is core-unstruck iff
m[j] > 6L + 1, and K_L(x) is one sliding sum of that indicator; the core's own real-phase RECORD at
threshold t is R(t) = the longest run of consecutive slots with m <= t (the longest stretch the
primes <= t strike entirely). Every quantity is exact (no sampling): the full histogram of K_L over
all S - L + 1 starts, its minimum, every start attaining it, the record stretch's K.

Counterfactuals, same section length and same slot coordinate. (a) Random phases: each gear g
keeps its two teeth (n = a and n = a - 2 mod g) and takes a uniformly random a; seeds 1000-1002.
(b) Random integers: each prime gear p is replaced by a uniformly random integer coprime to 6 in
[p - max(2, p/20), p + max(2, p/20)], distinct, NOT required to be pairwise coprime (section 5
shows why it cannot be), striking its multiples; seeds 2000-2001. The free-phase side: a greedy
phasing (each gear in turn, ascending or descending, takes the residue that strikes the most
still-open slots of [0, L)) gives an upper bound on the free minimum and a lower bound on the free
record.

Models for "a count in disguise". The binomial model: K_L is a sum of L independent indicators of
probability p = (measured mean)/L; its predicted count at leftover k is N times the binomial mass,
and its predicted minimum is the smallest k at which the predicted count of starts with K <= k
reaches 1. The normal model: the same with a normal of the measured variance. z-score of the
minimum: (min - mean)/sd.

Scripts: research/stack/r5/core_leftover.py (free, sections, lscan) and
research/stack/r5/followup.py; results research/stack/r5/results/{free,sections,lscan,followup}.json.
Run times: free 15 s; sections 20 s for base 3 plus its five counterfactual builds (about 2 min);
lscan (31 lengths, five machines, 43.4 M slots each) about 6 min; memory under 1.2 GB per pass.

## 3. Results

### 3.1 K_L(x) as an object on the three sections at L = L* (research/stack/r5/results/sections.json)

| section | L* | starts N | min K | starts at min (clusters) | first minimum at n | mean K | var / binomial var | max K | record stretch: K, its percentile | starts with K <= record's K | core's own run R(6L* + 1) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base 3 | 579 | 43,407,953 | 3 | 64 (1) | 70,722,785 | 20.356 | 16.32 / 19.64 = 0.831 | 56 | 12, 2.07% | 897,507 | 404 |
| base 7 | 254 | 1,323,738 | 1 | 179 (7) | 687,527 | 10.393 | 8.76 / 9.97 = 0.878 | 38 | 1, 0.0135% (one of the 179) | 179 | 241 |
| base 23 | 153 | 48,539 | 0 | 1 (1) | 187,913 | 9.153 | 9.95 / 8.61 = 1.156 | 26 | 0, the minimum itself | 1 | 153 |

The full histogram of K_579 on the base-3 section (leftover : number of starts): 3:64, 4:129,
5:580, 6:2,307, 7:5,759, 8:19,465, 9:51,540, 10:116,601, 11:245,013, 12:456,049, 13:806,486,
14:1,271,075, 15:1,865,010, 16:2,526,847, 17:3,185,629, 18:3,753,514, 19:4,147,788, 20:4,289,876,
21:4,174,282, 22:3,877,786, 23:3,340,291, 24:2,725,422, 25:2,106,592, 26:1,552,116, 27:1,084,345,
28:718,424, 29:451,801, 30:273,419, 31:155,781, 32:87,741, 33:49,548, 34:27,237, 35:15,300,
36:7,703, 37:4,894, 38:3,139, 39:2,299, 40:1,389, 41:794, 42:603, 43:575, 44:511, 45:362, 46:262,
47:204, 48:177, 49:255, 50:147, 51:168, 52:212, 53:189, 54:144, 55:108, 56:1. Median 20.
Base 7 (K_254): 1:179, 2:965, 3:3,779, 4:11,891, 5:29,260, 6:59,713, 7:99,911, 8:144,174,
9:176,603, 10:185,988, 11:174,723, 12:144,643, 13:109,900, 14:74,586, 15:46,390, 16:28,128,
17:14,346, 18:7,374, 19:4,192, 20:2,231, 21:1,218, 22:1,013, 23:733, 24:536, 25:467, 26:270,
27:176, 28:74, 29:41, 30:34, 31:30, 32:41, 33:13, 34:15, 35:24, 36:24, 37:32, 38:21. Median 10.
Base 23 (K_153): 0:1, 1:62, 2:152, 3:382, 4:1,211, 5:2,916, 6:4,431, 7:6,062, 8:6,852, 9:6,864,
10:5,498, 11:4,671, 12:3,152, 13:2,376, 14:1,370, 15:1,041, 16:488, 17:269, 18:101, 19:101,
20:129, 21:121, 22:125, 23:93, 24:50, 25:19, 26:2. Median 9.

The minimum on the base-3 section: the 64 starts x = 11,784,442 .. 11,784,505 (n from 70,722,785)
are one cluster, i.e. one stretch of 642 slots holding exactly 3 core-unstruck slots; of those 3,
two are twin primes and one is struck by the tail (the twin gap there is under 642 slots). It lies
in the generic part (above 3475^2 = 12,075,625; the quiet part below it, where every leftover is a
twin, has minimum 5). The record stretch (start x = 42,655,637, n = 255,949,955) has K = 12, and
every sub-stretch of it of length 540 or more has exactly 12: all twelve leftovers lie in its
middle 501 slots, its first and last 39 slots are core-covered runs. For L = 580 .. 595 the stretch
from the record's start has K = 13 (the slot after the record is the twin that ends it).

The free-phase side (research/stack/r5/results/free.json). The greedy phasing covers [0, L*) with
core(L*) at once (leftover 0 ascending and descending on all three sections), and the largest
stretch it covers with that fixed core is 25,267 slots on base 3 (43.6 L*), 10,347 on base 7
(40.7 L*), 3,282 on base 23 (21.5 L*); with the core growing as gears <= 6L + 1 the same three
numbers. So the free-phase minimum of the leftover is 0 at L*, and stays 0 to at least 21-44 times
L*; the loaded record rule's bound (min over phasings of the cost <= tail count) is slack by the
whole tail at L*. The real-phase minimum is 3, 1, 0.

### 3.2 The law of the minimum in L on the base-3 section (research/stack/r5/results/lscan.json)

Core = primes <= 6L + 1 at every L. Columns: real phases; three random-phase seeds; the
non-coprime random integer set (seed 2000; 1,875 integers, 158,927 non-coprime pairs). R = the
core's own run R(6L + 1). vr = variance / binomial variance.

| L | core gears | real: min / mean / vr / R | phase seeds: min (mean, R) | integers: min / mean / vr / R |
|---|---|---|---|---|
| 50 | 60 | 0 / 3.77 / 0.74 / 146 | 0, 0, 0 (3.76; 144, 164, 154) | 0 / 4.69 / 0.69 / 135 |
| 100 | 108 | 0 / 6.1 / 0.74 / 202 | 0, 0, 0 (6.0; 186, 182, 187) | 0 / 8.5 / 0.65 / 135 |
| 150 | 152 | 0 / 8.1 / 0.75 / 242 | 0, 0, 0 (8.0; 211, 237, 206) | 1 / 12.0 / 0.64 / 135 |
| 200 | 195 | 0 / 9.9 / 0.75 / 259 | 0, 0, 0 (9.8; 241, 244, 256) | 2 / 15.6 / 0.62 / 159 |
| 250 | 237 | 0 / 11.5 / 0.76 / 259 | 0, 0, 0 (11.6; 254, 270, 272) | 3 / 19.0 / 0.61 / 162 |
| 275 | 257 | 0 / 12.3 / 0.76 / 278 | 1, 0, 1 (12.4; 254, 296, 272) | 4 / 20.6 / 0.61 / 162 |
| 300 | 277 | 1 / 13.1 / 0.77 / 278 | 1, 1, 1 (13.2; 254, 296, 289) | 5 / 22.3 / 0.61 / 176 |
| 350 | 315 | 1 / 14.5 / 0.78 / 309 | 1, 2, 1 (14.8; 276, 303, 302) | 5 / 25.6 / 0.60 / 194 |
| 400 | 355 | 1 / 15.9 / 0.79 / 314 | 1, 2, 2 (16.4; 285, 303, 302) | 8 / 28.9 / 0.60 / 194 |
| 450 | 391 | 2 / 17.2 / 0.80 / 344 | 3, 3, 2 (17.9; 285, 321, 302) | 10 / 32.2 / 0.60 / 194 |
| 500 | 429 | 3 / 18.4 / 0.81 / 375 | 3, 4, 3 (19.4; 318, 321, 307) | 13 / 35.4 / 0.60 / 194 |
| 525 | 444 | 4 / 19.1 / 0.82 / 375 | 3, 4, 3 (20.1; 369, 321, 307) | 15 / 36.9 / 0.60 / 204 |
| 550 | 462 | 3 / 19.7 / 0.82 / 404 | 4, 4, 3 (20.9; 369, 322, 307) | 17 / 38.6 / 0.60 / 211 |
| 579 | 485 | 3 / 20.4 / 0.83 / 404 | 5, 3, 3 (21.7; 369, 322, 307) | 19 / 40.4 / 0.60 / 211 |
| 600 | 501 | 3 / 20.8 / 0.83 / 404 | 5, 4, 4 (22.3; 369, 322, 379) | 20 / 41.7 / 0.59 / 211 |
| 650 | 537 | 4 / 22.0 / 0.84 / 404 | 5, 5, 4 (23.7; 369, 326, 379) | 22 / 44.8 / 0.59 / 211 |
| 700 | 573 | 5 / 23.1 / 0.85 / 404 | 5, 7, 5 (25.0; 369, 326, 379) | 26 / 47.9 / 0.59 / 211 |
| 800 | 645 | 7 / 25.2 / 0.88 / 404 | 5, 9, 7 (27.7; 384, 326, 379) | 30 / 54.0 / 0.59 / 211 |
| 900 | 710 | 7 / 27.3 / 0.90 / 404 | 6, 11, 8 (30.4; 384, 354, 379) | 35 / 60.1 / 0.58 / 211 |
| 1000 | 781 | 9 / 29.3 / 0.93 / 471 | 8, 12, 11 (32.9; 403, 354, 379) | 41 / 66.1 / 0.58 / 211 |
| 1200 | 917 | 9 / 33.2 / 0.99 / 471 | 12, 15, 14 (37.9; 403, 354, 379) | 49 / 78.1 / 0.57 / 211 |
| 1500 | 1116 | 16 / 38.9 / 1.14 / 504 | 19, 19, 20 (45.1; 452, 387, 384) | 63 / 95.6 / 0.57 / 211 |
| 2000 | 1436 | 20 / 48.7 / 1.49 / 529 | 29, 29, 25 (56.5; 452, 436, 509) | 89 / 125.2 / 0.56 / 211 |

(The grid also holds L = 75, 125, 175, 225, 325, 375, 425, 475: real minima 0, 0, 0, 0, 1, 1, 2, 2.)

The crossing. The largest L with min K_L = 0 is L0 = 278 for the real phases, with R(6 x 278 + 1 =
1669) = 278 exactly and R(6L + 1) = 278 for every L in 275 .. 281; the seeds give L0 = 254, 296,
272 (R = 254, 296, 272 at their crossings), the integer set 135. Above L0 the real minimum is 1 at
L = 300 .. 400, 2 at 425 .. 475, 3 at 500 .. 600 except 4 at 525, then 4, 5, 7, 7, 9, 9, 16, 20 at
650, 700, 800, 900, 1000, 1200, 1500, 2000. Near the record, for every L from 540 to 595 the
section minimum is 3 (the same cluster; 98 starts at L = 545 shrinking to 48 at L = 595).

Where the minimum sits. For L <= 225 the first minimising start lies in the quiet part (n below
(6L + 1)^2, where every leftover is a twin) and at the line's known large twin gaps: n = 187,913 at
L = 125 and 150 (the base-23 record gap 187,907 .. 188,831), 850,355 at L = 175 .. 225 (the base-31
record). From L = 250 the minimum alternates between the generic part (23,831,993; 80,424,653;
70,722,785 at L = 550 .. 600) and the quiet part (4,557,827 at 425 .. 525; 13,622,507 at 650 ..
800; 34,736,075 at 1500), and at L = 2000 it is at 149,907,017, just above the quiet boundary
144,024,001. The random-phase minima sit at unrelated positions (e.g. 113,354,363; 161,652,227;
149,822,615 at L = 579).

### 3.3 The count models (research/stack/r5/results/followup.json)

Low tail of K_579 on the base-3 section, measured against the binomial model (N times the mass of
Bin(579, p), p = mean/579): real phases 3: 64 vs 67.6; 4: 129 vs 355; 5: 580 vs 1,487; 6: 2,307 vs
5,184; 7: 5,759 vs 15,463; 8: 19,465 vs 40,288; 10: 116,601 vs 193,446; 12: 456,049 vs 628,876;
15: 1,865,010 vs 2,020,826. Random phases (seed 0): 3: 0 vs 20.9; 4: 0 vs 116.9; 5: 149 vs 523;
6: 604 vs 1,945; 8: 5,471 vs 17,224; 10: 37,361 vs 94,245; 12: 190,878 vs 349,133. The measured
low tail is lighter than binomial by a factor 2.5-3 in the range 4-8 (the match at k = 3 is one
cluster of 64 consecutive starts counted 64 times). The binomial extreme-value threshold (the
smallest k with N P(K <= k) >= 1) is 2 for the real section and every seed against measured minima
3, 5, 3, 3; the normal model with the measured variance gives 0. Across the L grid the binomial
threshold runs 1 (L = 500-525), 2 (550-600), 3 (650-800), 4, 5, 7, 10, 16 (900, 1000, 1200, 1500,
2000) against real minima 3-4, 3, 4-7, 7, 9, 9, 16, 20: it under-reads the minimum by 1-3 up to
L = 1000 and reads it exactly at 1500 (16 = 16); for the seeds it under-reads by 1-8 throughout.

z-scores of the minimum, (min - mean)/sd, every machine: base 3 (N = 43.4 M): real -4.30, phase
seeds -4.19, -4.71, -4.67, integer sets -4.52, -4.70 (sqrt(2 ln N) = 5.93, sqrt(2 ln(N/L)) = 4.74);
base 7: real -3.17, seeds -3.69, -3.70, -3.68, integers -3.88, -4.77 (5.31, 4.14); base 23: real
-2.90, seeds -3.38, -3.19, -3.07, integers -3.33, -3.32 (4.65, 3.39).

The variance by threshold (the sliding count over 579 slots of the slots unstruck by the primes
<= t, base 3): t = 13: mean 171.8, var 4.45, ratio to binomial 0.037; t = 31: 107.9, 14.5, 0.166;
61: 79.7, 19.7, 0.286; 101: 65.2, 21.6, 0.373; 199: 49.5, 22.1, 0.487; 301: 43.7, 21.7, 0.538;
601: 35.3, 20.6, 0.620; 1009: 30.3, 19.5, 0.679; 2003: 24.4, 17.9, 0.765; 3475: 20.4, 16.3, 0.831.

The density by position (base 3, t = 3475, twenty equal bins of the section): the CRT product over
the 485 core gears prod(1 - 2/g) = 0.037439; the random-phase machine's section density 0.037441
(bins 0.03726 .. 0.03753, no trend); the real section 0.035158, by bins 0.03403 (n < 13.0 M, which
holds the whole quiet part), 0.03134, 0.03278, 0.03353, 0.03413, 0.03462, 0.03490, 0.03518, 0.03539,
0.03558, 0.03576, 0.03583, 0.03596, 0.03591, 0.03624, 0.03628, 0.03624, 0.03637, 0.03661, 0.03648
(u = ln n / ln 3475 from 2.01 to 2.38 at the bin tops). At L = 2000 (t = 12,001) the real mean is
48.7 against the seeds' 56.5, i.e. density 0.0243 against 0.0283, with 55% of the section in the
quiet part.

Non-coprime integer sets at L*: base 3 seeds: 158,927 and 153,881 non-coprime pairs among 1,875,
core sum of 2/g 3.049 and 3.044 against the real 3.057; mean 40.4 and 44.9 against 20.4; minima 19
and 22 against 3; runs R 211 and 164 against 404. Base 7: 7,331 and 6,732 pairs; means 18.8, 21.1
against 10.4; minima 6, 5 against 1. Base 23: 282 and 270 pairs; means 12.5, 14.2 against 9.2;
minima 3, 5 against 0.

## 4. Mechanism

The real phases are one point of the core's period. A phase vector of the core is, by CRT, one
residue x mod P_core = prod(core gears); the real machine (every gear striking its multiples) is
x = 0, and the section [c, c') is the stretch of positions [c, c') of that period. Everything the
real-phase leftover does that a random phasing does not is therefore a property of the stretch
next to the origin, and there is exactly one such property visible in the data: the section sits
at u = ln n / ln(6L + 1) between 1 and 2.4 for L = L*, and between 1 and 2.1 for L = 2000, so
its core-unstruck slots are pairs of numbers whose prime factors all exceed 6L + 1, i.e. below
(6L + 1)^2 pairs of primes (the twins, the quiet part) and below (6L + 1)^3 pairs of primes or
semiprimes. The CRT product counts every factorisation pattern; next to the origin most patterns do
not exist yet, and the density runs below the product by 6% (L = 579) to 14% (L = 2000), rising bin
by bin toward it as u grows (section 3.3). This is the Buchstab deficit (prior art: Buchstab's
function for rough numbers, its two-dimensional analogue for rough pairs; noted, not pursued). A
random phasing reads the same function at a random point of the period, where every pattern is
present and the density is the product to five digits.

The minimum follows the mean. At every L the real minimum and the seeds' minima are the same
extreme-value statistic of their own distributions: the z-score of the minimum is -4.2 to -4.7 on
the base-3 section for all five machines (real, three seeds, two integer sets), -3.2 to -3.9 on
base 7, -2.9 to -3.4 on base 23, in each section between sqrt(2 ln(N/L)) and sqrt(2 ln N). Where
the real minimum drops below the seeds' (from L = 1200: 9 against 12-15, 16 against 19-20, 20
against 25-29) it does so because its mean is lower by 4.7, 6.2, 7.8, i.e. by the Buchstab deficit
of the quiet part, which is 20%, 31%, 55% of the section at those L; its z-score there is -3.4
against the seeds' -4.2 because the density gradient across the section inflates its variance
(ratio 1.49 at L = 2000). The variance itself is carried by the large gears: the primes <= 13 leave
a count whose variance is 3.7% of binomial (they are periodic in every stretch), the ratio rises
through 0.54 at t = 301 (about L/2) and 0.62 at 601 (about L) to 0.83 at 3475 (6L): a gear that
strikes a stretch a few times or once behaves like a Bernoulli indicator, a gear that strikes it
many times behaves like a constant. The low tail is therefore lighter than binomial, and the
minimum lies 1-3 above the binomial threshold.

The crossing is the core's own record. min K_L = 0 iff the section holds a run of L consecutive
slots each struck by a prime <= 6L + 1, i.e. iff R(6L + 1) >= L; the real crossing L0 = 278 is at
R(1669) = 278 exactly, and the seeds' crossings 254, 296, 272 are their own runs. So "min K_L >= 1
for all L > L0" is the statement that the composite machine of the primes <= t has record below
t/6 on the section, for every t: in the quiet part that is the longest twin gap (the run at
n = 187,913 IS the base-23 record gap), in the generic part the longest run of slots with a
(6L + 1)-smooth member on each side.

The record stretch is not the minimum on base 3. It sits at K = 12, the 2.07th percentile, with
897,507 starts at or below it; it is twin-free because its 12 leftovers are each struck by one of
the 181 tail gears that strike it, and the minimum stretch (K = 3, 64 starts) is not twin-free
because the tail strikes 1 of its 3. On base 7 the record IS a minimum stretch (K = 1, one of 179
starts, the only one whose single leftover the tail strikes); on base 23 the tail is empty and the
record is the unique K = 0 stretch, the core's own record.

The core is rigid. A pairwise coprime set of integers >= 5, coprime to 6 and at most X, has at most
pi(X) - 2 members, and with that many members every member is a prime power (S11). So the
"different gear set of the same size and count" the brief asked for does not exist inside the
loaded record rule's hypothesis (pairwise coprimality, needed for CRT); the counterfactual that
exists drops coprimality, and then shared factors waste strikes: the union coverage falls, the
mean leftover doubles (40.4 against 20.4 at L* on base 3 with the same sum of 2/g to 0.3%), and
the minimum with it (19 against 3); its own z-score is -4.5, the same extreme value. The recursion
(gears = the survivors of the sections below = the primes) is, within the wheel framework, forced
by count and range, and it leaves no property in the minimum that is not the mean.

## 5. Laws

**S10 (ONE PHASING IS ONE POSITION; proved).** Let G be a pairwise coprime gear set with teeth
n = 0, -2 mod g, P = prod G. The map x -> (x mod g)_g is a bijection from Z_P to phase vectors, and
the leftover of the machine at phases (a_g) on the stretch [y, y + L) equals the leftover of the
real machine (all phases 0) on [y - x, y - x + L) with x the CRT lift of (a_g). Hence the real-phase
leftover on a section [c, c') is the free-phase leftover function along the positions [c, c') of
the period, the stretch beginning at the origin, and a random phase vector is the same function
along a stretch beginning at a uniformly random point. *Proof.* CRT. *Measured with it.* The
random-phase density equals the CRT product to five digits (0.037441 against 0.037439, base 3,
t = 3475), the real section's is 0.035158.

**S11 (RIGIDITY OF THE CORE; proved).** Let G be pairwise coprime integers, each >= 5 and coprime
to 6, each <= X. Then |G| <= pi(X) - 2, and if |G| = pi(X) - 2 every member of G is a power of a
distinct prime in [5, X]. *Proof.* Each g has a least prime factor lpf(g) in [5, X]; pairwise
coprimality makes lpf injective on G, so |G| is at most the number of primes in [5, X], which is
pi(X) - 2. With equality lpf is a bijection onto those primes; if some g had a second prime factor
q != lpf(g), the member h with lpf(h) = q shares q with g, contradicting coprimality; so g is a
power of lpf(g). QED. *Use.* The core at L* on base 3 (485 primes in [5, 3475]) admits no
counterfactual of the same count and range inside the loaded record rule's hypothesis other than
prime-power substitutions (5 -> 25, 125, 625, 3125; 7 -> 49, 343, 2401; ... only for p <= 58).
Kernel-ready; a Formalist's item.

**S12 (THE CROSSING; proved, and exact on the data).** min over x of K_L(x) = 0 iff R(6L + 1) >= L,
where R(t) is the longest run of consecutive slots of the section each of which has a member
divisible by a prime in [5, t]. *Proof.* A stretch of L with no core-unstruck slot is a run of
length >= L, and conversely. QED. *Measured.* On the base-3 section L0 = 278 with R(1669) = 278;
random-phase seeds 254, 296, 272; the non-coprime integer set 135. In the quiet part (n < t^2) a
run of R(t) is a twin gap: the minimising runs at L = 125 .. 225 are the base-23 and base-31
record gaps (n = 187,913; 850,355).

**S13 (THE MINIMUM IS AN EXTREME VALUE OF A COUNT; measured, 0 exceptions on 5 machines x 3
sections and 31 lengths).** With mean m_L and standard deviation s_L of K_L over the section's N
starts, the minimum satisfies m_L - z s_L with z between sqrt(2 ln(N/L)) and sqrt(2 ln N): z in
[4.19, 4.71] at L = 579 on base 3 (bounds 4.74, 5.93) for real phases, three random phasings and
two integer sets alike; [3.17, 4.77] on base 7 (4.14, 5.31); [2.90, 3.38] on base 23 (3.39, 4.65).
The distribution is narrower than binomial (variance ratio 0.83 at L = 579, 0.037 for the primes
<= 13 alone) and its low tail lighter, so the binomial threshold under-reads the minimum by 1-3
at L <= 1000. No machine departs from the form; the real phases differ from the random ones only
in m_L (S14).

**S14 (THE DEFICIT NEXT TO THE ORIGIN; measured, mechanism known).** On the section, the density of
core-unstruck slots is below the CRT product by 6.1% at t = 3475 (0.035158 against 0.037439) and
by 14% at t = 12,001 (0.0243 against 0.0283), rising monotonically with position from 0.0313 at
u = 2.05 to 0.0365 at u = 2.38; a random phasing shows no deficit and no trend. Mechanism: below
t^3 a t-rough number is a prime or a product of two primes above t, so the pairs the CRT product
counts mostly do not exist yet (Buchstab; prior art, one line). Consequence for the minimum: from
L = 1200 the real minimum lies below every random-phase seed (9 against 12-15; 16 against 19-20;
20 against 25-29) by the shift of the mean (4.7, 6.2, 7.8), with the z-score less extreme (-3.4
against -4.2 at L = 2000) because the trend inflates the variance (ratio 1.49).

## 6. Scorecard, filled

| # | prediction | verdict |
|---|---|---|
| P1 | greedy free phasing covers >= 3 L* on all three sections; free minimum 0 at L* | **held**: 43.6, 40.7, 21.5 times L*; leftover 0 at L* both orders |
| P2 | base 3, L = 579: min K in [1, 4], < 10 starts, not at the record | **held on value and record** (3; record K = 12); **missed on count**: 64 starts, but one cluster (one stretch) |
| P3 | variance ratio in [0.5, 0.85] at L = 579 | **held**: 0.831 |
| P4 | an extreme-value form predicts the minimum within +-2 | **held for the binomial threshold on the real section** (2 against 3) and two seeds; **missed** on seed 0 (2 against 5) and by the normal model (0 against 3): the true tail is lighter than both |
| P5 | L0 in [300, 450] | **refuted**: L0 = 278 (the core's run at t = 1669 is 278, below my estimate) |
| P6 | min K_2000 in [12, 20] | **held** at the edge: 20 |
| P7 | starts with K_579 <= 12 exceed 10^5 | **held**: 897,507 |
| P8 | random-phase seeds bracket the real minimum within +-3 at every grid L | **held to L = 1000**; from L = 1200 the real minimum is below every seed (by 3, 3, 5 against the nearest), a systematic shift explained by S14, not a bracket failure by the letter (>= 4 at one L only) |
| P9 | the non-coprime integer set has higher mean and minimum at L* | **held**: 40.4 / 19 against 20.4 / 3 (base 3); on all three sections |
| P10 (the brief's) | real minimum far above the free minimum and growing with L | **held**: 3 against 0 at L* (base 3), 1 against 0 (base 7), 0 = 0 (base 23, no tail); growth 0 -> 20 over the grid, non-decreasing except 4 -> 3 at 525 -> 550 |
| P11 | the minimum starts lie in the generic part at L* | **held** at L* = 579 (all 64 above 3475^2; quiet-part minimum 5); at other L the minimum alternates between the parts (section 3.2) |

## 7. Verdict

The real-phase minimum leftover of the core on a section is not a bound the free-phase rule can
reach and not a bound the construction supplies either: it is the extreme value, over the
section's N starts, of a count whose mean is the sieve density of (6L + 1)-rough pairs (the CRT
product less the Buchstab deficit of the stretch next to the origin) and whose spread is narrower
than binomial by the periodicity of the small gears. Against the free-phase bound: 3 against 0 at
L* on base 3 (the free phasing covers 25,267 slots, 43.6 L*), 1 against 0 on base 7, 0 against 0 on
base 23 where the tail is empty and the record is the core's own. In L: zero to L0 = 278 = the
core's own run R(1669) (S12), then 1, 2, 3, 5, 7, 9, 16, 20 at L = 300, 425, 500, 700, 800, 1000,
1500, 2000, each within z = 4.2-4.7 standard deviations of its mean (S13). Against the
counterfactuals: random phasings of the same core have the same crossing to within +-24 (254, 296,
272 against 278), the same minima to within +-2 to L = 1000, and higher minima from L = 1200 by
exactly the shift of the mean (S14); a coprime gear set of another kind does not exist (S11), and
the non-coprime one is worse by its wasted strikes (mean doubled, minimum 19 against 3), the same
extreme value of a larger mean. The recursion's only trace in the minimum is the mean's deficit
next to the origin, which lowers the minimum, never raises it, and is a known density correction.

ROOT marks. (i) "min K_L(x) > 0 for every L above L0" is S12's R(6L + 1) < L, i.e. the composite
record of the primes <= t is below t/6 on the section for every t: in the quiet part that is the
longest twin gap, the root question; in the generic part the same statement for rough pairs, which
is the root question's sieve shadow. (ii) "The tail cannot finish the leftover on any stretch as
long as the section" needs the minimum of K_L at L = the section's length, 43 million on base 3,
where the mean is 1.5 million and the minimum is an extreme value of that: a count, with a margin
that is the count's own. (iii) The record L* is where K is low AND the tail lands on every
leftover; the tail's finish is a coincidence of K placements (step_evidence.md section 6), and
this branch adds that K at the record is not even the minimum (12 against 3, the 2nd percentile),
so no bound on the minimum alone decides the record.

For the tree: R4.d.i's child "the core's real-phase leftover" FACT (S10, S11, S12 exact; S13, S14
measured with the mechanism visible) with a ROOT mark on the minimum; no CANDIDATE. What survived:
the rigidity law S11 (the core is the primes up to prime powers, forced by count and range) as a
Formalist item; the crossing law S12 as the exact form of "the core alone covers a stretch";
the deficit S14 as the one measurable trace of the real phases.

## 8. Dead ends (with the refuting instance)

- "The real phases cannot cover, the free phases can" as a lever: the real phases DO cover every
  stretch to L0 = 278 (0 leftover, the core's own record), and above it the shortfall is 1-20
  against tails of 1,390 gears; nothing in the shortfall is phase-specific (three seeds agree).
- The minimum as the record's location: on base 3 the record's K = 12 is the 2.07th percentile,
  897,507 starts lie at or below it and the minimum stretch (K = 3) is not twin-free.
- A same-size pairwise-coprime counterfactual: does not exist (S11); every attempted draw shares
  factors (158,927 pairs of 1,875).
- The binomial model as the exact form of the minimum: under-reads by 1-3 (the low tail is 2.5-3
  times lighter at k = 4 .. 8); the normal model with the measured variance under-reads by 3-5.

## 9. The part's remaining open items

- Closed here: the value, position and multiplicity of the real minimum on three sections; the
  free bound at L*; the crossing L0 and its identity with the core's run (S12); the rigidity of
  the core (S11); the counterfactual comparison.
- Measurement with no structural content: the z-score band [4.2, 4.7] at L = 579 (an
  extreme-value constant of N and the tail shape); the variance ratio by threshold (which gears
  carry the variance); the exact bin profile of the deficit.
- Root question in disguise: R(t) < t/6 for every t on the section (the quiet-part half is the
  longest twin gap); the minimum of K at L = the section length.
- Genuinely open on the part alone: none that is not a count. The two-dimensional Buchstab profile
  of the deficit (density of t-rough pairs at 1 < u < 3 as a function of u) has a closed form in
  the literature's terms and was not computed here; it would give m_L on the section without
  measurement, and with the variance-by-threshold law the extreme value, i.e. the whole minimum
  as a formula in L, N and the section's position: a formula for a count, not a route.
