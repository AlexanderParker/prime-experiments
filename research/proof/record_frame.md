# Branch 5d.i - the record as a frame of three gears

Parent: theory_tree.md node 5d ("every gear is needed for the record, and record phases are fixed
at the anchor, gear 7 and the top gear with the middle gears free", 7d / research/proof/
anchor_runs_zero.md Q2b, script research/anchor235/r34/q2b_record_classes.py).

Scripts: research/anchor235/r35/, results in research/anchor235/r35/results/.

## Pre-registered (written before any computation of this branch)

### The theory

A record stretch of machine M = {5..q} is made in two parts. A **frame**: the residues of the
stretch's start column under gear 5, gear 7 and the top gear q. A **filling**: the residues of
the middle gears, which cover whatever offsets the frame leaves open. The claim is that the frame
decides *where* a record can sit and the filling decides *whether* it does. If that is exact,
then the root question "does an opening land in the window" becomes: the window contains such and
such frames, so what stops the middle gears from completing a record-length filling there.

Vocabulary as the project fixes it: column k = (6k-1, 6k+1); gear g strikes k iff
k = +-u_g (mod g), u_g = 6^{-1} mod g; window = the certified range (q, q'^2], never a sliding
run; stretch = a sliding run; record F(M) = the longest opening-free stretch. A record stretch
after the opening x is the set of blocked columns x+1 .. x+F-1; its **start** is s = x+1 and an
**offset** j means column s+j, j = 0 .. F-2.

### Predictions, with numbers, and what refutes each

- **P1 (top-gear corridor).** In every record stretch of one machine the top gear's strike
  columns occupy the same residues mod 35 (the corridor), the same set for every record of that
  machine up to the mirror. REFUTED by one record whose top-gear strike residues mod 35 are not
  the mirror image or the equal of another record's of the same machine.
- **P2 (one legal word).** The offsets of the top gear's consecutive strikes inside a record
  differ by 2u' or q'-2u' (the two letters), and the resulting word (letters in order, plus the
  first strike's offset) is the SAME word for every record of the machine, up to reversal.
  NOTE (prior art, stated before testing): that the differences lie in {2u', q'-2u'} is already
  kernel-checked (docs/novel/two-teeth-kill-spacing.md, TwoTeeth.kill_spacing); only the "same
  word in every record" half is being tested here, and the letter-value half will be recorded in
  one line and not re-derived. REFUTED by two records of one machine with different words.
- **P3 (the frame set is one, up to mirror).** The number of distinct (5, 7, top) start-residue
  triples over all record stretches of a machine is 2. REFUTED by 3 or more. Pre-registered
  doubt: the parent observation already reports 20 distinct top-gear phases at m19 (the top gear
  strikes the m19 record once, so its phase has many places to be), so P3 is expected to FAIL at
  m19 and the honest prediction is "2 at the machines where the top gear strikes the record two
  or more times, many where it strikes once".
- **P4 (the filling is not independent gear by gear).** For a frame, let R be the offsets of the
  record length that the three frame gears do not strike, let a **completion** be a choice of
  middle-gear residues covering all of R (equivalently: a column of the period with the frame's
  residues whose whole record-length stretch is blocked), and let n_g be the number of residues
  of middle gear g that strike at least one offset of R. Prediction: completions <= 0.01 * prod
  n_g at m23, and the ratio falls with the machine. REFUTED if completions > 0.1 * prod n_g.
- **P5 (the window test).** At each rung the window (q/6, (q'^2-1)/6] is scanned for columns
  whose (5, 7, top) residues match a record frame; from each, the forward run of blocked columns
  is walked and the first open column (the break) recorded. Prediction: the maximum break offset
  over all frame columns of a window is strictly below F(M) at every rung; i.e. the window's best
  attempt at a record on a frame falls short. REFUTED by one window frame column whose forward
  blocked run reaches F(M)-1.
- **P5b (vacuity check, pre-registered as a possible outcome).** The window has about q^2/6
  columns and the frame period is 35q, so a window contains a full (5, 7, top) frame column only
  from q of order 210 upward. If the count is 0 or 1 at the rungs where the frame is known, the
  test is run again with the (5, 7) part of the frame only, at every rung up to q = 5000, and
  that is reported as a weaker instrument, labelled as such.
- **P6 (systematic break).** The break offset is decided by something nameable: either one
  middle gear carries more than half the breaks, or the break offset's residue mod 35 is
  concentrated (one residue class more than 3x its share), or the break sits within 2 columns of
  a gear square (g^2-1)/6. REFUTED if none of the three holds - in which case the honest verdict
  is "the middle gears' phases in the window are what they are" and the branch stops there.

### Scorecard

| # | prediction | verdict | evidence |
|---|---|---|---|
| P1 | top-gear strikes on one corridor mod 35 in every record | REFUTED | 4, 3, 2, 1, 1, 2 distinct corridors up to mirror at m13..m31 (f3 (d)) |
| P2 | one legal word per machine, up to reversal | REFUTED | 1, 2, 2, 1, 1, 2 distinct words up to reversal at m13..m31 (f3 (d)) |
| P3 | frame set = 2 (one mirror pair) | REFUTED, and in the opposite direction | 8, 8, 4, 2, 2, 4 frames at m13..m31 (f1); from m23 the record set is 4, 2, 4 members and every gear is pinned except the top one or two (f3 (a)) |
| P4 | completions <= 0.01 prod n_g at m23 | CONFIRMED, and sharpened | C/prod n_g = 1.0e-4 at m23, 4.8e-8 at m31; against the proper independence baseline C/E = 0.48, 0.24, 0.12, 0.060, 0.009 at m17..m31 (f3 (c)) |
| P5 | max window break offset < F(M) at every rung | CONFIRMED but vacuous as stated | the only full-frame window columns are 2 at m7, 1 at m29 (break 4), 1 at m31 (break 9); the real number is L* below |
| P5b | fewer than 2 full-frame columns per window below q = 210 | CONFIRMED | expected count = window/(35q) = 0.07..0.21 at every rung m7..m31; actual 0, 0, 0, 0, 0, 1, 1 (f2 Part A) |
| P6 | a nameable decider of the break offset | REFUTED on all three counts | mod-35 concentration 1.65x not 3x; share of breaks within 2 of a gear square 0.0076 against a base rate of the same order; the "gear within one column" test is degenerate (gear 5 scores 100% by construction) (f2 Part C) |

### What this branch could find that is not already known

Known and not to be re-derived: the corridor law (12 of 24 gcd classes forbidden), corridor
resonance (extreme gaps phase-locked mod 35, pinned residues), the two-teeth kill-spacing law,
the two-n-gap reordering, L4 (every gear is a sole striker in an above-record stretch). What is
not on the record is a COUNT: how many columns of a period carry a record's frame, how many of
those complete, and what the completions look like in the window. The parent gives the frame's
existence; this branch asks whether the frame is a positional obstruction with a size.

## Setup (exact ranges)

Full periods, residue sieve, no per-number work, no sampling anywhere except the one control
noted:

| machine | gears | period P | F | scan |
|---|---|---|---|---|
| {5..7} | 5, 7 | 35 | 5 | one array |
| {5..11} | .. 11 | 385 | 7 | one array |
| {5..13} | .. 13 | 5,005 | 11 | one array |
| {5..17} | .. 17 | 85,085 | 18 | one array |
| {5..19} | .. 19 | 1,616,615 | 25 | one array |
| {5..23} | .. 23 | 37,182,145 | 34 | one array |
| {5..29} | .. 29 | 1,078,282,205 | 43 | chunked, 3.2 s |
| {5..31} | .. 31 | 33,426,748,355 | 58 | chunked, 95 s |

m29 and m31 are new full-period scans, not corpus reads; both reproduce the corpus F (43, 58) and
their record starts are re-verified independently in f3 (F-1 blocked columns, both flanks open).
The window test runs at every prime rung q = 11 .. 1999 over the columns
(q/6, (q'^2-1)/6]; the machine's openings there are computed exactly, and an open window column is
a twin pair. Scripts research/anchor235/r35/f1_record_frames.py, f2_window_frames.py,
f3_ledger.py; results in the sibling results/ directory (f1_record_frames.txt, f1_frames.tsv,
f2_window_frames.txt, f2_window.tsv, f3_ledger.txt). The only non-exact number in the document is
the 20,000-vector control in f3 (b), labelled there.

## Results

### 1. The record set collapses; the "frame plus free filling" picture is an artefact of the small machines

Distinct start residues per gear over all record stretches of a machine, after quotienting by the
mirror (1 = the gear is PINNED by the record; f3 (a)):

| machine | F | classes up to mirror | 5 | 7 | 11 | 13 | 17 | 19 | 23 | 29 | 31 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| m13 | 11 | 6 | 1 | 3 | 6 | 2 | | | | | |
| m17 | 18 | 10 | 2 | 3 | 4 | 5 | 3 | | | | |
| m19 | 25 | 10 | 1 | 1 | 4 | 4 | 2 | 4 | | | |
| m23 | 34 | 2 | 1 | 1 | 2 | 2 | 2 | 2 | 1 | | |
| m29 | 43 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | |
| m31 | 58 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 2 |

The parent's picture -- anchor, gear 7 and the top gear fixed, middle gears free -- is exactly what
m19 and m23 show. It does not survive one more rung. At **m29 the record is unique up to mirror**:
two stretches, mirror images, so every gear is pinned and there is no filling to speak of. At
**m31 the free gears are the TOP TWO (29 and 31), and gears 5, 7, 11, 13, 17, 19, 23 are all
pinned** -- the reverse of the parent's split. The number of record stretches over the ladder is
2, 4, 12, 20, 20, 4, 2, 4 at m7..m31: it peaks at m17-m19 and collapses. So the "frame of three
gears" is not a structure of the machine; it is what a record set of twenty members looks like
when only three gears have enough teeth to constrain twenty starts.

Scored as pre-registered: the number of distinct (5, 7, top) frames is 8, 8, 4, 2, 2, 4 at
m13..m31 (f1), against the predicted 2.

### 2. The top gear's corridor and word (P1, P2)

The top gear's strike columns reduced mod 35, and the offsets between its consecutive strikes,
both reduced by the mirror (f3 (d)):

| machine | records | corridors up to mirror | words up to reversal | letters (a, b) |
|---|---|---|---|---|
| m13 | 12 | 4: (8,12) (12,16) (13,17) (14,18) | 1: (4) | 4, 9 |
| m17 | 20 | 3: (7,18) (12,18) (12,23) | 2: (6), (11) | 6, 11 |
| m19 | 20 | 2: (6,12,25) (12,18) | 2: (6), (6,13) | 6, 13 |
| m23 | 4 | 1: (2,10,25) | 1: (8,15) | 8, 15 |
| m29 | 2 | 1: (0,6,25) | 1: (10,19) | 10, 19 |
| m31 | 4 | 2: (2,12,16) (5,9,26,30) | 2: (10,21), (21,10,21) | 10, 21 |

P1 and P2 both hold at exactly the two machines where the record set IS a single mirror pair, and
fail everywhere else, including at m31 where the top gear strikes the record three times in two
records and four times in the other two. So neither is a rule; both are the mirror.

The letter values are the known kernel-checked two-teeth spacing law (TwoTeeth.kill_spacing,
docs/novel/two-teeth-kill-spacing.md) and are not re-derived here: every observed word is over the
alphabet {a, b} with a = 2u', b = q' - 2u', with no padding letter at any machine because the
record is shorter than 2q at all eight.

### 3. The filling: how far below independence (P4)

For each frame, R is the set of offsets of the record length that the three frame gears leave
open; a completion is a choice of middle-gear residues covering all of R, equivalently a column of
the period carrying the frame whose whole record-length stretch is blocked. Note the completions
are the record stretches themselves: a blocked run of F-1 columns cannot be extended, so every
completion is flanked by openings.

| machine | frame columns N = P/(35q) | size of R | completions | independence expectation | actual/expected |
|---|---|---|---|---|---|
| m13 | 11 | 2 | 1 | 0.364 | 2.75 |
| m17 | 143 | 3 | 2 | 4.17 | 0.48 |
| m19 | 2,431 | 6 | 2 | 8.44 | 0.24 |
| m23 | 46,189 | 10 | 2 | 16.97 | 0.118 |
| m29 | 1,062,347 | 16 | 1 | 16.72 | 0.060 |
| m31 | 30,808,063 | 20 | 1 | 115.34 | 0.0087 |

The expectation is N (1 - pi)^|R| with pi = prod (1 - 2/g) over the middle gears. The filling is
NOT independent gear by gear, and the deficit doubles every rung: by m31 a record-length filling
is **115 times rarer than independent middle gears would make it**. Against the pre-registered
weaker baseline prod n_g (n_g = the number of residues of g that strike at least one offset of R),
the ratio is 1.0e-4 at m23 and 4.8e-8 at m31, far inside the pre-registered 0.01.

The break histograms (f1) give the shape: at m31, of the 30,808,063 columns carrying frame
(1,6,2), 14,313,915 break at the first open offset, 8,478,540 at the second, and the survivor
counts past offsets 31, 32, 36, 39, 41, 46, 47, 54 are 2430, 768, 340, 79, 24, 14, 2, 1. The
filling is a covering problem that thins by roughly a half at each of the frame's open offsets and
then falls off a cliff in the last third of the stretch.

### 4. The window test (P5, P5b)

**The exact test is vacuous, for a reason with a number.** The window of {5..q} is
(q/6, (q'^2-1)/6], about q^2/6 columns; a (5, 7, top) frame repeats every 35q columns; so the
expected number of frame columns in a window is q/210, and a window holds one only from q of order
210 upward. Measured (f2 Part A): expected 0.078, 0.070, 0.101, 0.097, 0.128, 0.170, 0.154, 0.206
at m7..m31, actual 2, 0, 0, 0, 0, 0, 1, 1. The two at m7 break at offset 4 (against F-1 = 4, i.e.
they are records -- the m7 window is a third of the m7 period); the one at m29 breaks at offset 4
and the one at m31 at offset 9.

**The real number is L\***, the longest blocked run of {5..q} starting inside the window -- the
window's best attempt at a record, with no frame condition at all (an open window column is a twin
pair, so L* is the largest twin gap below q'^2 in column units):

| q | window cols | F-1 | L* | start of L* | (5,7) of the start | L*/(F-1) |
|---|---|---|---|---|---|---|
| 11 | 27 | 6 | 4 | 13 | (3,6) | 0.67 |
| 13 | 46 | 10 | 4 | 13 | (3,6) | 0.40 |
| 17 | 58 | 17 | 5 | 53 | (3,4) | 0.29 |
| 19 | 85 | 24 | 11 | 59 | (4,3) | 0.46 |
| 23 | 137 | 33 | 24 | 111 | (1,6) | **0.73** |
| 29 | 156 | 42 | 24 | 111 | (1,6) | 0.57 |
| 31 | 223 | 57 | 24 | 111 | (1,6) | 0.42 |
| 37 | 274 | 87 | 24 | 111 | (1,6) | 0.28 |
| 41 | 302 | 90 | 24 | 111 | (1,6) | 0.27 |
| 43 | 361 | 102 | 24 | 111 | (1,6) | 0.24 |
| 47 | 461 | 117 | 27 | 398 | (3,6) | 0.23 |
| 53 | 572 | 144 | 27 | 398 | (3,6) | 0.19 |

Maximum of L*/(F-1) over the ladder = 0.727, at m23, and monotone downward after it. Past the
known-F range L* keeps its shape: 104 at q = 401 (start 10,384), 153 at 601 (31,319), 241 at 1009
(141,726), 251 at 1801 (478,161) -- it moves in long plateaus, changing only when a new maximal
twin gap is passed, while F grows at every rung.

Restricted to the record's own (5, 7) frame class, the window's best run is 4, 4, 4, 4, 4, 9, 24 at
m11..m31 -- i.e. at m23 the record's (5, 7) class does not even contain the window's best attempt
(24 sits on (1,6), while the m23 record frames are (4,4) and (4,6)).

### 5. What decides the break (P6)

At every rung the answer is: nothing nameable. Over the window columns of the record's (5, 7)
class, the break column's residue mod 35 occupies all 15 open classes with the top class at
1.65x its fair share (q = 1009: 1613 of 14,646 against 976); the share of breaks within two
columns of a gear square (g^2-1)/6 is 0.0076 at q = 1009 with median distance 168 (the base rate);
and the "gears within one column of striking the break" test is degenerate -- gear 5 scores 100%
by construction, because an open column's residue mod 5 is always at distance 1 from a tooth. Mean
break offsets rise as 1.88, 7.21, 6.62, 10.92, 15.64, 20.02 at q = 23, 31, 101, 211, 503, 1009,
i.e. like the local twin density and nothing else. P6 is refuted on all three of its counts.

### 6. The one positive mechanism the branch found: the small gears run at coverage maximum

For a stretch of L columns starting at s, gear g strikes c_g(s) of them and m_g(L) is the maximum
of c_g over the g phases. In a record (L = F-1), gear 5 is at its maximum in **every record of
every machine** m13..m31, and gear 7 from m19 on; the top one or two gears are never at their
maximum (f3 (b), (b2)):

| machine | L | 5 | 7 | 11 | 13 | 17 | 19 | 23 | 29 | 31 |
|---|---|---|---|---|---|---|---|---|---|---|
| m13 | 10 | 1.00/1.00 | 0.67/0.14 | 1.00/0.82 | 1.00/0.54 | | | | | |
| m17 | 17 | 1.00/0.80 | 0.40/0.14 | 0.40/0.18 | 0.80/0.62 | 1.00/1.00 | | | | |
| m19 | 24 | 1.00/0.60 | 1.00/0.14 | 1.00/0.36 | 1.00/0.69 | 0.00/0.06 | 0.80/0.53 | | | |
| m23 | 33 | 1.00/0.20 | 1.00/0.43 | 1.00/1.00 | 0.50/0.23 | 1.00/0.88 | 1.00/0.47 | 0.00/0.09 | | |
| m29 | 42 | 1.00/0.80 | 1.00/1.00 | 1.00/0.64 | 1.00/0.46 | 0.00/0.12 | 1.00/0.42 | 1.00/0.65 | 0.00/0.10 | |
| m31 | 57 | 1.00/0.80 | 1.00/0.29 | 1.00/0.36 | 0.00/0.08 | 1.00/0.71 | 1.00/1.00 | 0.00/0.13 | 1.00/0.93 | 0.50/0.68 |

Each cell is (share of the machine's records with that gear at its coverage maximum) / (share of
phases attaining the maximum, i.e. the chance for a column drawn at random). The sharpest entry is
m23 gear 5: one phase in five attains the maximum and every record uses it. The total ledger is
sum c_g = 11, 22, 33, 46, 61, 85 against sum m_g = 12, 22, 34, 48, 63, 88, with 3-7 of the gears
at maximum out of 4-9; the control (20,000 uniform phase vectors) puts P(that many at maximum) at
0.063, 0.014, 0.019, 0.015, 0.121, 0.313 -- so the COUNT is unremarkable and the IDENTITY of the
maximal gears is the content: always the bottom of the machine, never the top.

## Mechanism

Which gears, which residues, which columns.

A record stretch of {5..q} is not a frame with a free filling. From m23 on it is a single residue
class modulo the whole period, up to the mirror: at m29 the two record starts are
200,906,186 and 877,375,978 = P - 200,906,186 - 41, mirror images, so every one of the eight gears
is pinned; at m31 gears 5, 7, 11, 13, 17, 19, 23 take one residue each (3, 1, 1, 4, 10, 7, 2 on one
mirror side) and only gears 29 and 31 have a choice, two apiece. The record's rarity is therefore a
CRT statement of full modulus: the density of record starts is 12/5005, 20/85085, 20/1.6e6,
4/3.7e7, 2/1.1e9, 4/3.3e10 at m13..m31, so the expected number of record starts in a window of
q^2/6 columns falls 1.1e-1, 1.4e-2, 1.1e-3, 1.5e-5, 2.9e-7, 2.7e-8. Nothing in the window
"prevents" a record: the record is one column in ten billion and the window is two hundred columns
long.

Inside the stretch, the labour is split by size and it is visible gear by gear. The bottom gears
run at their coverage maximum -- gear 5 always, gear 7 from m19, gear 11 from m19 -- so they
supply the bulk: at m31 gear 5 alone covers 23 of the record's 57 offsets, gears 5 and 7 together
35, gears 5, 7 and 11 together 39, and the bottom four gears 43 of 57. The top one or two gears are always BELOW their maximum, because what
the record needs from them is not bulk but the two or three offsets nobody else covers; putting
them at maximum coverage would move their teeth onto columns gear 5 already has. That is the
size-graded form of L4's sole-striker requirement, and it is why "every gear is needed" (the
parent's F(M minus g) < F(M)) and "the top gear wastes 99% of its hits" (7d's Q1) are the same
statement seen from two ends.

The filling is a covering problem, not a product. At m31, of the 30.8 million columns of the
period carrying a record's frame, the number surviving each successive frame-open offset is
30.8M, 16.5M, 8.0M, 3.4M, 1.5M, ... , 2430, 768, 340, 79, 24, 14, 2, 1: a near-halving at each
step, ending at exactly one. An independent middle-gear model predicts 115 survivors, not 1, so
the true covering is 115 times harder than independent -- the gears' strikes are two teeth
2u' apart with period g, and inside a stretch of length under 2g a gear can place at most two or
three strikes, at a spacing it does not choose.

And the window: the window's best attempt at a record is L*, the longest blocked run starting in
it, which is the largest twin gap below q'^2 in column units. It sits at 24 columns from q = 23
through q = 43 (the run from column 111, i.e. no twin pair between 665 and 805) and at 27 from
q = 47, while F-1 climbs 33, 42, 57, 87, 90, 102, 117, 144. The ratio peaks at 0.727 at m23 and
falls monotonically thereafter. The break -- the first open column, which in the window IS a twin
pair -- is not decided by any gear, any residue class mod 35, or any gear square. The honest
statement is the pre-registered one: the middle gears' phases in the window are what they are.

## What is new

1. **The record set collapses and the record becomes unique up to mirror.** m29 has exactly two
   record stretches and they are mirror images, so every gear is pinned; m31 has four, with gears
   5 through 23 all pinned and only 29 and 31 free. The parent's "anchor + 7 + top fixed, middle
   free" is a property of m19 and m23 only. Not on the record anywhere; the m29 and m31 record
   positions are new full-period computations.
2. **The frame is not a positional filter with any room in it.** The exact count: N = P/(35q)
   columns of the period carry a record's frame (46,189 / 1,062,347 / 30,808,063 at m23/m29/m31)
   and exactly 2 / 1 / 1 of them complete.
3. **The filling's deficit against independence, with a number that grows:** 0.48, 0.24, 0.118,
   0.060, 0.0087 at m17..m31, i.e. a factor of two per rung, 115x by m31.
4. **The coverage-maximality split:** gear 5 sits at its coverage maximum in every record of every
   machine tested and gear 7 from m19 on, while the top one or two gears never do. This is a
   mechanism for the known position fact 5e (which slot a record gap can start on is dictated by
   F mod 5) rather than a restatement of it, and it extends the same statement to gear 7 and
   gear 11.
5. **The window's best attempt, measured against the record:** L*/(F-1) has its maximum 0.727 at
   m23 and falls monotonically to 0.19 at m53; the exact-frame window test is vacuous below
   q = 210 for the arithmetic reason window/frame period = q/210.

Toward the root: (2) and (5) are the branch's contribution and they are a measurement of the
margin, not a route to it. Nothing here bounds the largest twin gap below q'^2, which is what the
root needs; the branch establishes that the record's own structure gives no obstruction to look
for in the window, because the record is one column mod P and the window is q^2/6 columns.

## Verdict

DEAD as a route; two FACTs kept.

- The theory as stated ("a record is a frame of three gears plus a filling") is REFUTED at m29 and
  m31, and in the direction that closes the branch: there is no frame/filling split at the top
  machines because the record is pinned modulo every gear but the top one or two.
- The window test the branch was opened to run is vacuous by arithmetic (window/frame period =
  q/210 < 1 at every rung whose F is known), and the non-vacuous version of it -- the longest
  blocked run actually starting in the window -- is the largest twin gap below q'^2, a classical
  object the machine does not illuminate here.
- FACT (kept under 5d): the record set collapses to a mirror pair at m29 and to two mirror pairs
  at m31, with every gear below the top two pinned.
- FACT (kept under 5d): gear 5 is at its coverage-maximal phase in every record of every machine
  m13..m31, gear 7 from m19 on, and the top one or two gears never are.
- P6 refuted on all three counts; the honest answer, as pre-registered, is that the middle gears'
  phases in the window are what they are, and the branch stops there.

## Dead ends, with the refuting instance

- "One corridor per machine, up to mirror" (P1): m31 has two, (2,12,16) and (5,9,26,30), and they
  are not mirror images of each other -- the second is self-mirror and comes from records with
  four top-gear strikes, the first from records with three.
- "One legal word per machine, up to reversal" (P2): m17 has (6) and (11); m19 has (6) and (6,13);
  m31 has (10,21) and (21,10,21).
- "The frame set is one up to mirror" (P3): 8 frames at m13 and at m17, 4 at m19 and at m31.
- "A nameable decider of the break offset" (P6): mod-35 concentration 1.65x against the 3x
  pre-registered; 0.0076 of breaks within two columns of a gear square; the per-gear near-miss
  test degenerate.
- "The window contains every (5, 7, top) frame" (the branch's own premise): a window contains
  q/210 of them, so none at any rung below q = 210 -- and the frames are only known to m31.

## Prior art

Nearest located: the corridor law and corridor resonance (docs/novel, pinned residues mod 35 for
extreme gaps) are the mod-35 half of section 2; the two-teeth kill-spacing law
(docs/novel/two-teeth-kill-spacing.md, kernel-checked) is the letter-value half of P2 and was not
re-derived; L4 (research/proof/pair_statement.md) is the sole-striker statement that section 6's
top-gear half restates in coverage units; node 5e is the gear-5 half of section 6 as a position
fact; and L* is the largest twin-prime gap below y^2, a classical object (Polignac / Ziller-Morack
Conjecture 6 territory), which is where the window half of this branch runs out of new ground.
