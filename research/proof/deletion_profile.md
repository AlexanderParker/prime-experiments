# Branch 5d.ii: what each gear holds up, in the period and in the window

Prover, round 35 (2026-09-05). Parent: node 5d (research/proof/anchor_runs_zero.md, question 2)
-- "every gear is needed for the record", F(M minus g) < F(M) for every gear g, exact to m23.
Scripts in research/anchor235/r35/ (self-contained, numpy only, `uv run python <script>` from the
repository root); results in research/anchor235/r35/results/. Nothing committed.

Vocabulary as in docs/proof-search/alignment-rules.md section 0. Column k = (6k-1, 6k+1); gear g
strikes k iff k = +-u_g (mod g), u_g = 6^{-1} mod g; machine M = {5..q}; period P = prod g; an
opening is a column no gear strikes; F(M) = the longest gap between consecutive openings (max-gap
convention, wrap included). The window at rung q is the columns (q/6, W], W = (q'^2-1)/6, q' the
next prime. Window = the certified range, never a sliding run; a stretch is a sliding run.

Two objects, one per frame:

- **F(M)**, the period record, and its **deletion profile** drop(g) = F(M) - F(M minus g), and
  more generally the **deletion lattice** F(M minus S) over every subset S of gears.
- **F_W(q)**, the longest opening-free stretch that lies inside the window at rung q, measured in
  the same max-gap convention between consecutive openings both of which lie in the window, and
  its deletion profile drop_W(g) = F_W(M) - F_W(M minus g) taken on the same column range.

A gear with drop 0 is one the record does not need: the same stretch is blocked without it.

## Pre-registered (written before any script of this round was run) and the scorecard

**P1 (period profile, question 1).** F(M minus g) for g >= 7 at m7..m23 is already on record
(anchor_runs_zero.md Q2), so the drops at those gears are arithmetic on a known table, not a new
measurement; what is new here is gear 5, the sole-strike counts, and the subset lattice.
Predictions: (a) drop(5) is the largest drop at every level, and drop(5) >= 20 at m23.
(b) The caller's prediction -- "largest for the top two or three gears and for gear 5, smallest
for the middle" -- is WRONG in its top half: the profile falls with g, so the largest drop above
gear 5 is at gear 7 at every level (from the known table: 3, 3, 5, 8, 12, 17 at m7..m23), and the
top gear is never the largest. (c) The profile is not monotone in g: at m17 and m23 gear 17 drops
more than gear 13 (7 vs 5 at m17; 11 vs 9 at m23). (d) Every gear is a sole striker in every
record stretch (L4 applies), sole counts small, 1-4 per gear. (e) Lattice: drops are SUBADDITIVE,
F(M) - F(M minus S) <= sum over g in S of drop(g), at every subset at m11..m23, with strict
inequality for most pairs.

**P2 (the window's own record, question 2).** (a) F_W(q) < F(M) at every rung from 11 to 997
(the window is a short piece of the period and the period record is the maximum over all of it,
so this is expected but not forced -- the window is not a uniformly random stretch, and node 7d
found it opening-POOR, so F_W could exceed F). Refuted by one rung with F_W >= F.
(b) F_W/F in [0.5, 1.0] at the rungs where F is on record (q <= 59).
(c) Position: the longest window stretch sits in the upper half of the window at most rungs, and
its distance from W is less than half the window at two thirds of the rungs or more.

**P3 (the window's deletion profile, question 2).** (a) FORCED by the square gate (node 7d's
mechanism: gear g's first exclusive kill in the window is at its own square, column (g^2-1)/6,
because below g^2 every cofactor is smaller than g and already strikes): drop_W(g) = 0 for every
gear g with g^2 > 6*top + 1, top the last column of the window's record stretch. So the zero set
contains every gear above sqrt(6*top+1) and the profile's support is inside the effective machine
{5..sqrt(6*top+1)}. Prediction: the support is a STRICT subset of that at most rungs.
(b) The count of gears with non-zero drop is small and slowly growing: <= 8 at every rung to 997.
(c) The caller's "the gears with zero drop are the large ones, above some fraction of q'": the
predicted fraction is not a fraction of q' at all but sqrt(6*top+1)/q' -- near 1 only if the
record stretch sits at the very top of the window; at a typical position 0.3-0.7 of q'.

**P4 (always and never, question 4).** (a) Gear 5 has non-zero drop_W at >= 90 % of the rungs
11..997; gear 7 at >= 70 %. (b) The ALWAYS set (non-zero drop at every rung) is {5} or empty, not
larger. (c) The NEVER set is empty as a set of fixed gears -- a gear is small at a high rung and
large at a low one -- so the correct statement is per-rung: the zero set at rung q has size
pi(q) - O(pi(sqrt(6W))) and its share of the machine tends to 1. (d) Removing the WHOLE zero set
at once leaves the record stretch blocked at >= 90 % of the rungs (the zero gears are jointly
redundant, not only singly).

**P5 (the two profiles compared, question 3).** (a) The window record's (5,7) phase is NOT pinned:
over the rungs 11..997 the pair ((x+1) mod 5, (x+1) mod 7) takes at least 8 distinct values,
against the period record at m19 where all 20 stretches share one phase mod 35. (b) The set of
gears with non-zero drop in the window WANDERS; consecutive rungs share less than half their
non-zero sets on average. Refuted if the non-zero set is the same gear list at 10 consecutive rungs.

**P6 (the object, question 4/5).** Honest prior: the window profile is a statement about twin
primes (node 7d proved that any window statement provable from tooth positions is one), so no
object found here can be a route on its own; the value of the branch, if any, is the CONTRAST --
the period record needs every gear (L4, a re-phasing argument that requires a full period), the
window record needs only the gears below the square root of its position (the square gate), and
the difference is exactly that re-phasing moves the window. Prediction: the contrast holds and is
exact; no new route.

## Setup (exact ranges)

Four scripts, all exact, each finishing in under a second on one core.

**p1_period_lattice.py** -- full periods of {5..7} .. {5..23} (P = 35 .. 37,182,145), openings by
residue slicing, gaps with the wrap. For every gear g (gear 5 included, which node 5d's table
omits): F(M minus g) on the reduced period P/g. Every record stretch's kill map and sole-strike
map. The full deletion LATTICE F(M minus S) over all 2^k - 1 non-empty subsets S of the gears
(127 subsets at m23), tested for subadditivity against the singleton drops.

**w1_window_profile.py** -- every prime rung q, 7 <= q <= 997 (165 rungs). Window columns
lo = q//6 + 1 .. hi = (q'^2-1)/6. Per column the striker count and the striker sum (so a column
with count 1 names its sole gear). F_W = the largest gap between consecutive openings both inside
the window; head and tail runs reported separately. drop_W(g) is exact: it can be non-zero only
for a gear that is the sole striker of an interior column of EVERY record stretch (removing g must
split all of them), and for those gears F_W(M minus g) is recomputed over the whole window. Also
the zero set Z, the joint deletion F_W(M minus Z), the (5,7) phase, the position. Gate: the
openings in the window are the twin pairs in (q, q'^2); the count matches node 7d's at the shared
rungs.

**c1_compare.py** -- the 13 DISTINCT window record stretches over those 165 rungs, each with its
holder set, sole-column density, (5,7) phase, and its exact MINIMUM BLOCKING SET (branch and bound
set cover, columns as bitmasks); the same quantities for the period record at m7..m23; the
square-gate joint deletion; and drop against drop_W gear by gear at q = 11..23.

**q4_object.py** -- per rung the largest drop and the gear attaining it; the rungs where gear 5 or
7 is not needed; the union of holders over the distinct stretches; the per-holder sole counts in
both frames; and F_W against F({5..z}), z = sqrt(6*top+1).

## Results

### R1. The period profile: the drop falls with the gear, and gear 5 is the top of it

drop(g) = F(M) - F(M minus g), full periods (results/p1_period_lattice.txt). The g >= 7 columns
are arithmetic on node 5d's F(M minus g) table and are not a new measurement; the gear-5 column
and the shape are.

    M          F     5     7    11    13    17    19    23
    {5..7}     5     3     3
    {5..11}    7     3     3     2
    {5..13}   11     5     5     4     4
    {5..17}   18     9     8     5     2     7
    {5..19}   25    13    12     7     4     7     7
    {5..23}   34    17    17    11     9    11     9     9

The profile FALLS with g. Gear 5 is the argmax at every level (tied with gear 7 at m7, m11, m13,
m23); the TOP gear is at or near the minimum from m13 on. The caller's pre-registered shape --
largest at the top two or three gears -- is refuted in that half: at m23 the top gear's drop is 9,
the smallest value in the row, against 17 for gear 5. The profile is not monotone: gear 17 beats
gear 13 at m17 (7 against 2) and at m23 (11 against 9).

Why it has that shape: removing g re-opens exactly g's SOLE columns of the record stretch, so
F(M minus g) is the longest surviving piece, and the small gears own more sole columns (6, 4, 3, 2,
3, 2, 3 for g = 5..23 in the first m23 record stretch) and cut the stretch into more pieces. Every
gear owns at least TWO sole columns except one gear per level -- sole-count multisets [2,2],
[2,2,1], [4,2,2,1], [5,4,2,2,1], [6,5,2,2,2,1], [6,4,3,3,3,2,2] at m7..m23. That every gear owns at
least one is not new: it is node 5d's F(M minus g) < F(M) restated, and L4's sole-striker corollary
in the same place.

Sole-column density of the record stretch (sole columns / interior columns): 1.00, 0.83, 0.90,
0.82, 0.75, 0.70 at m7..m23 -- the period record is a near-perfect tiling, which is branch 5's
observation with a number on it.

**The deletion lattice.** F(M minus S) over all 127 subsets at m23 (and all subsets at the smaller
levels): NO subadditivity violation anywhere -- the joint drop never exceeds the sum of the
singleton drops -- and the slack is large. At m23: (5,7) removes 24 against 34 singly, (7,11) 22
against 28, (13,19) and (19,23) 16 against 18 (the tightest pairs, slack 2). Removing every gear
but 5 leaves F = 2; removing only gear 5 leaves 17.

### R2. The window's own record, and the profile on it

F_W(q) at the 165 rungs (results/w1_window_profile.txt, results/w1_rungs.tsv). Focus rungs:

    q     q'    W        F_W  x       stretch (numbers)     frac   below W  gears  |N|  |Z|
    59    61    620      28   397     2387..2545            0.636      195     15    6    9
    173   179   5340     83   4070    24425..24913          0.761     1187     38   16   22
    499   503   42168    154  31318   187913..188827        0.742    10696     93   22   71
    997   1009  169680   242  141725  850355..851797        0.835    27713    166   23  143

F_W is INHERITED: the window at rung q is (q, q'^2], so the longest blocked stretch below q'^2
stays the record until a longer one appears. Over the 165 rungs there are only THIRTEEN distinct
stretches (x = 12, 52, 58, 110, 397, 980, 2233, 3090, 4070, 10383, 31318, 114742, 141725 with
F_W = 5, 6, 12, 25, 28, 35, 47, 62, 83, 105, 154, 168, 242). F_W is exactly the largest gap between
consecutive twin prime pairs in (q, q'^2), which is what node 7d's kernel identity says it must be;
the object here is not the value but what holds it up.

F_W < F(M) at 13 of the 14 rungs where F is on record, the exception being equality at q = 7 (both
5). The ratio F_W/F falls: 1.00, 0.71, 0.45, 0.33, 0.48, 0.74, 0.58, 0.43, 0.28, 0.27, 0.24, 0.24,
0.19, 0.17 at q = 7..59. Position: the upper half of the window at 101 of 165 rungs.

**The profile at rung 997** (extract; the full per-gear table is in the results file):

    gear     5    7   11   13   17   19   23   29   37   41   47   53  101  103  137  151  157  179  251  269  409  563  709
    kills   97   69   44   37   29   25   21   17   13   12   10    9    4    5    4    4    4    3    2    1    1    1    1
    sole    13    7    3    2    2    1    3    1    1    2    1    3    1    1    1    1    1    1    1    1    1    1    1
    drop   138  126  104   64  105   88   82   22   74   74   19   82   74   87   12   52   72   13   48   74   74   57   49

50 of the 241 interior columns are sole-struck (density 0.207); the other 143 gears of the machine
have drop 0. |N|, the count of gears with a non-zero drop, is 0 to 27 over the rungs, mean 17.8, by
band 3.57 (q < 30), 6.09 (q < 100), 11.73 (q < 300), 17.82 (q < 1000). The caller's "the window's
longest stretch is held up by FEWER gears than the period record, and the gears with zero drop are
the large ones" is half right: the zero set is overwhelming (143 of 166 gears at rung 997) but the
count of holders is not small, and my own pre-registered bound of 8 is badly refuted.

The window profile is NOT ordered by gear size. Gear 61 drops 49 at rung 499 while gear 23 drops 4;
gear 269 drops 74 at rung 997 while gear 29 drops 22. The reason is in the sole counts: in the
window most holders own EXACTLY ONE sole column (15 of 22, 21 of 27, 18 of 26 at the three largest
stretches, each counted at the first rung holding it; 15 of 23 for the 242-stretch at rung 997),
so the drop is decided by WHERE that one column sits, not by how many columns the gear
owns. In the period record at most one gear per level owns a single sole column. That is the
sharpest single contrast this branch found.

**The square gate bounds the support, exactly.** Node 7d's mechanism -- gear g's first exclusive
kill is at its own square, because below g^2 every cofactor is smaller than g and strikes the
column already -- gives a theorem here: a gear with g^2 > 6*top + 1 can be the sole striker of no
column of the stretch, so drop_W(g) = 0. Measured: max(N) <= sqrt(6*top+1) at every one of the 165
rungs, no exception. At rung 997 the gate is 922 and accounts for 11 of the 143 zero drops; the
other 132 are zero by arithmetic accident (no sole column), not by the gate. The gate is exact and
it is not what makes the profile thin.

**Sole-column density and the minimum blocking set** (results/c1_compare.txt), exact set cover:

    stretch x   F_W   gears in M   holders   sole cols   sole density   MIN COVER
    12          5     2            2         4           1.00           2
    110         25    7            6         18          0.75           6
    397         28    13           10        16          0.59           10
    2233        47    28           12        19          0.41           14
    4070        83    35           17        30          0.37           20
    10383       105   51           19        33          0.32           22
    31318       154   82           22        38          0.25           27
    114742      168   143          27        38          0.23           34
    141725      242   155          26        53          0.22           32

Against the period record, where the same computation gives min cover = the WHOLE machine at every
level (2, 3, 4, 5, 6, 7 gears at m7..m23) and sole density 0.70-1.00. That is the rule the branch
was looking for: **the period record needs every gear; the window record needs a fifth of them.**

### R3. The two profiles compared, gear by gear

At the rungs where both frames exist (c1_compare.txt part d):

    M = {5..19}   gear:      5    7   11   13   17   19
                  drop:     13   12    7    4    7    7
                  drop_W:    7    4    2    5    5    2
    M = {5..23}   gear:      5    7   11   13   17   19   23
                  drop:     17   17   11    9   11    9    9
                  drop_W:   18   17   13   12   13    7    0

The two profiles are not proportional and not even co-ordered: at m19 gear 13 is the period's
weakest gear and one of the window's strongest. At m23 gear 23 -- the top gear, essential in the
period -- has drop_W = 0: it holds up nothing in the window's longest stretch.

The (5,7) corridor phase. The window record takes 7 distinct ((x+1) mod 5, (x+1) mod 7) values over
the 13 stretches -- (1,1), (1,4), (1,6), (3,4), (3,6), (4,1), (4,3) -- of the 15 an
opening-started stretch can take. The period record takes 8 values over the six levels, and at m19
all 20 record stretches share the single phase (1,6) (node 5d). Four phases occur in both frames,
three only in the window ((4,1), (4,3), (1,1)) and four only in the period ((4,5), (3,1), (4,4),
(4,6)). So the window's stretch does not sit in the period record's corridor: the corridor pinning
is a property of extremal phase selection and the window has no phase selection.

Stability of N across rungs: mean Jaccard overlap 0.940 between consecutive rungs -- but that is
inheritance, not stability of a rule (the same stretch is the record at many rungs). Within one
stretch the holder set is NESTED-DECREASING in the machine at every one of the 165 rungs: adding a
gear can give a sole column a second striker but can never take one away, so holders only shrink.
Measured at stretch 397: 10 holders at rung 47, then 7, 6, 5, 5, 5 at rungs 53..71. The window
record becomes MORE redundantly covered as the machine grows. Its limit is the set of gears owning
a column (g^a, prime) -- alignment-rules 4.1's pseudo-twin condition, noted and not re-derived.

### R4. Always and never

- **Gear 5 is the always-gear.** It is a holder of all 13 distinct window record stretches; it has
  a non-zero drop at 164 of the 165 rungs (the exception is rung 13, where two stretches tie at
  F_W = 5 so no single gear can split both); and it is the largest drop at 151 of the 165 rungs.
  Gear 7 is next at 157 of 165. So the answer to "a gear whose sole strikes are always present" is
  gear 5, measured, not forced -- and it already fails once.
- **The never set is per-rung, not per-gear.** No fixed gear is never needed: 55 distinct gears
  (max 877) hold up one of the 13 stretches. At a given rung the never set is huge (143 of 166 at
  997) and its share grows, but membership turns over: gear 31 is needed at rungs up to 827 and
  never after; gear 43 up to 431.
- **Individual redundancy is not joint redundancy.** Removing the whole zero set Z at once destroys
  the record stretch at 157 of the 165 rungs: F_W(M minus Z) = 16, 45, 82, 53 at rungs 59, 173,
  499, 997 against F_W = 28, 83, 154, 242. Each zero gear is droppable alone because another gear
  covers its columns; dropping them together removes both coverers of the doubly-struck columns.
  The one set that IS jointly droppable is the square-gate set {g : g^2 > 6*top + 1} -- verified at
  all four focus rungs, and a theorem, since every blocked window column has a prime factor at most
  its own square root.
- **Two reductions, very different sizes.** The smallest INITIAL SEGMENT {5..y} that blocks the
  window record is y = 43, 151, 311, 877 at rungs 59, 173, 499, 997 -- close to the square gate
  (50, 157, 434, 922) and hardly a reduction, because the stretch always contains one column whose
  two numbers both have a large smallest prime factor (at rung 997 it is column 141928 =
  (851567, 851569) with 851567 = 877 x 971). The smallest ARBITRARY subset is 32 of 166 gears. The
  window record is cheap to hold if you may choose the gears and expensive if you must take a
  prefix.

### R5. The effective-machine bound, and why it is the root restated

A blocked column k of the window has 6k-1 or 6k+1 composite and below q'^2, hence with a prime
factor at most sqrt(6k+1). So a window stretch ending at column x is a blocked stretch of the
machine {5..z}, z = sqrt(6x+1), and F_W <= F({5..z}) exactly. Where F({5..z}) is on record:

    stretch      top    z    effective machine   F({5..z})   F_W   F_W/F
    12..17       17     10   {5..7}                5           5   1.000
    52..58       58     18   {5..17}              18           6   0.333
    58..70       70     20   {5..19}              25          12   0.480
    110..135     135    28   {5..23}              34          25   0.735
    397..425     425    50   {5..47}             118          28   0.237

The bound is exactly attained at the first stretch and loosens fast (with the measured F ~ y^2/24
it is off by a factor of about 150 by rung 919). Applied at the TOP of the window it says: the
window is entirely blocked only if F({5..q'}) >= W - q/6, i.e. only if F(q') is within (q+1)/6 of
q'^2/6. That is R2 at the next level, so the 1/6 in the root is this argument's fixed point and it
cannot be iterated. Stopped there.

## Mechanism

Which gears, which columns, what forces the outcome.

The period record is chosen: it is the maximum over all P phases, so it is where the gears' teeth
pack with the least waste. Every gear owns two to six columns of it alone, 70-100 % of its columns
have exactly one striker, and no proper subset of gears blocks it -- the minimum cover is the whole
machine at every level tested. The deletion profile of such an object is essentially the sole-count
profile, which falls with g because gear g owns about 2F/g columns; that is why the drop falls with
g and why gear 5, striking two columns in every five, is the argmax. The lattice is strongly
subadditive for the same reason: two gears' sole columns are different columns, so removing both
cuts the stretch at the union of two cut sets, which is worth less than the two cuts separately.

The window record is not chosen -- it is one fixed stretch of the period, at the phase the primes
give it -- and the profile shows it. Its columns are covered two and three deep: sole density falls
from 1.00 to 0.22, so at rung 997 four columns in five have at least two strikers. Most gears that
hold a column hold exactly one, so their drop is set by that column's position: a sole column in
the middle halves the stretch (gear 269 at rung 997, one column at relative position 83, drop 74),
a sole column near an end does almost nothing (gear 137, relative position 230 of 242, drop 12).
Two thirds to nine tenths of the machine holds nothing at all, and a chosen subset of a fifth of
the gears blocks the whole stretch.

The square gate is the one part of this that is forced. Gear g's strikes in the window are the
columns of g*m; while m < g every prime factor of m is below g and strikes the column already, so g
can be nobody's sole striker below g^2. Hence the support of the window profile lies in
{5..sqrt(6*top+1)} at every rung, no exception in 165. It is exact, and it is not the reason the
profile is thin: at rung 997 it explains 11 of the 143 zero drops.

What would have to change for gear 5 to stop being the always-gear: every column of the window's
longest stretch of the form (5m, p) or (p, 5m) would have to have its 5-multiple carry a second
prime factor below q, or its partner be composite. That is not impossible -- it happens at rung 13
-- it is only unlikely at length 242, where gear 5 strikes 97 columns and six come out sole. The
machine can do it; nothing forbids it. So gear 5's presence is a strong measured regularity, not a
forced object, and the branch does not deliver the thing the round was looking for.

## What is new

1. The period deletion profile's SHAPE: falling with g, argmax gear 5 at every level (drops 3, 3,
   5, 9, 13, 17 at m7..m23, the gear-5 column being new -- node 5d's table starts at gear 7), the
   top gear at or near the minimum. The caller's "largest at the top gears" is refuted.
2. The deletion LATTICE: subadditive with no violation over all subsets at m7..m23, tightest pair
   slack 2 (at m23, (13,19) and (19,23)).
3. F_W as a measured object at 165 rungs with its holder structure: 13 distinct stretches,
   inheritance, F_W < F at every rung but q = 7, and the per-gear drop tables.
4. The redundancy contrast, with numbers: sole density 0.70-1.00 (period) against 1.00 falling to
   0.22 (window); minimum blocking set = the whole machine (period) against 32 of 155 gears
   (window); holders owning exactly one sole column: at most one per level (period) against 15-21
   of 22-27 (window).
5. Individual versus joint redundancy: the zero-drop set is jointly essential at 157 of 165 rungs,
   while the square-gate set is jointly droppable by proof.
6. The holder set of a fixed stretch is nested-decreasing in the machine (one-line proof, measured
   at every rung): the window's record gets more redundant as gears are added.
7. The two reductions: smallest initial segment {5..877} of 166 gears against smallest arbitrary
   subset 32 of 166, at rung 997.
8. The (5,7) phase of the window record is not the period record's corridor phase (three phases
   occur only in the window, four only in the period).

Nearest prior art inside the project, each noted in one line and not re-derived: node 5d's
F(M minus g) < F(M) table (the g >= 7 half of R1); L4's sole-striker corollary (why every gear is a
holder in the period); node 7d's first-exclusive-kill-at-the-square (the square gate, used here to
bound the profile's support) and its kernel identity (F_W is a twin-prime gap); alignment-rules 4.1
("a gear is needed iff it owns a pseudo-twin in the window"), which is the limit of the
nested-decreasing holder sets; the capacity and overlap counting dead end, which is why no count of
kills in the stretch was pursued as a bound.

## Verdict

WEAK, and closed as a route. The two profiles differ systematically and the difference is exactly
describable -- the period record is an extremal, near-perfectly tiled object that needs every gear;
the window record is an ordinary stretch that a chosen fifth of the gears can hold -- but every
window-side quantity is contingent on the primes and none of it is forced. The one forced statement,
the square gate, bounds the profile's support and explains 8 % of the zero drops; the one bound it
gives on F_W is F_W <= F(effective machine), which at the window's top is R2 restated. No object
whose presence in the window is forced was found: the best candidate, gear 5, is a holder of all 13
distinct window records and the largest drop at 151 of 165 rungs, and it already fails once (rung
13), so it is a regularity, not a law.

What survives for the tree: the redundancy contrast as a FACT under node 5d (the period record's
minimum blocking set is the whole machine at every level to m23; the window record's is 32 of 155
at rung 997), the nested-decreasing holder law, and the negative that the zero-drop gears are
jointly essential.

## Scorecard

- P1(a) SPLIT: gear 5 is the argmax at every level (held), but drop(5) = 17 at m23, not >= 20
  (refuted), and it TIES with gear 7 at m7, m11, m13, m23.
- P1(b) HELD: the profile falls with g; gear 7 is the largest above gear 5 at every level; the top
  gear is never the largest. The caller's opposite prediction is refuted.
- P1(c) HELD: gear 17 beats gear 13 at m17 (7 vs 2) and m23 (11 vs 9).
- P1(d) HELD, sharper: every gear owns a sole column in every record stretch; counts 2-6, not the
  predicted 1-4 (no gear owns exactly one at m23).
- P1(e) HELD: no subadditivity violation in any subset at any level.
- P2(a) HELD with one equality: F_W < F at 13 of 14 rungs, equal at q = 7.
- P2(b) REFUTED: F_W/F falls to 0.17 at q = 59, below the predicted floor of 0.5.
- P2(c) HELD: upper half at 101 of 165 rungs.
- P3(a) HELD, and it is a theorem: max(N) <= sqrt(6*top+1) at all 165 rungs; the support is a
  strict subset of the effective machine at every rung (23 of 155 eligible gears at 997).
- P3(b) REFUTED badly: |N| reaches 27, mean 17.8, against the predicted <= 8.
- P3(c) HELD in form: the threshold is sqrt(6*top+1)/q', not a fraction of q'; measured 0.82, 0.88,
  0.86, 0.91 at the four focus rungs (the stretch sits high in the window, so the gate is weak).
- P4(a) HELD: gear 5 at 164/165, gear 7 at 157/165.
- P4(b) HELD: the always set is {5}, and not literally always (rung 13).
- P4(c) HELD: no fixed gear is never needed; the never set is per-rung and its share grows to
  143/166.
- P4(d) REFUTED: Z is jointly droppable at only 8 of 165 rungs, not >= 90 %.
- P5(a) HELD: 7 distinct phases over the 13 distinct stretches, and they are not the period
  record's.
- P5(b) SPLIT: the raw consecutive-rung Jaccard is 0.940 (looks stable), but that is inheritance;
  within a stretch the set only shrinks, and across stretches it turns over completely.
- P6 HELD: the contrast is exact and there is no new route.
- Also refuted in passing (mine, stated in q4_object.py): "F_W minus the window's runner-up gap is
  a ceiling on every drop" is FALSE -- removing a gear opens columns everywhere and shortens the
  runner-up too (rung 433: largest drop 89 against the false ceiling 49).

## Dead ends (with the refuting instance)

- DEAD: "the window's longest stretch is held up by few gears, and the zero-drop gears are jointly
  droppable" -- rung 997, removing the 143 zero-drop gears at once takes F_W from 242 to 53; 157 of
  165 rungs behave the same way.
- DEAD: "the drop profile in the window is ordered by gear size, as the period's is" -- rung 997,
  gear 269 drops 74 and gear 29 drops 22; most holders own one sole column and the order is by
  position, not size.
- DEAD: "the window's record stretch reduces the machine to a small initial segment" -- rung 997,
  the smallest initial segment that blocks it is {5..877}, 149 of 166 gears, forced by the single
  column (851567, 851569).
- DEAD: "the window's runner-up stretch is a ceiling on the drops" -- rung 433, largest drop 89
  against ceiling 49.
- DEAD (one line, not pursued): the effective-machine bound applied at the window's top is R2 at
  the next level, so it cannot be iterated into a proof; the 1/6 in the root is its fixed point.
