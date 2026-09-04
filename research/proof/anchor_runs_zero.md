# Branch 7d: runs as the unit, and the zero mirror as a lever

Prover, round 34 (2026-09-05). Scripts in research/anchor235/r34/ (q1_runs_window.py,
q2_record_position.py, q2b_record_classes.py, q3_d0.py, q4_past_zero.py; each self-contained,
numpy only, run with `uv run python <script>` from the repository root; every run finishes in
under ten seconds on one core). Results in research/anchor235/r34/results/ (one text file per
script). Nothing is committed.

Vocabulary as in docs/proof-search/alignment-rules.md section 0. Column k = (6k-1, 6k+1); gear
g strikes k iff k = +-u_g (mod g), 6u_g = g -+ 1; machine M = {5..q}, period P = prod q; an
opening is a column no gear strikes; F(M) is the longest gap between consecutive openings
(max-gap convention, wrap included), and the record STRETCH is the F-1 blocked columns between
them. The window at level Q is the numbers (Q, Q'^2], Q' the next prime, i.e. the columns
(Q/6, W] with W = (Q'^2-1)/6. A run of gear g over the anchor 2,3,5 is 30g numbers = 5g columns
= g cycles, with six anchor-open hits per run (anchor-235.md section 2); gear g's clean end zone
is the set of columns within h_g of a multiple of 5g, where h_g is the column distance from the
multiple to g's first anchor-open strike (h = 8, 2, 2, 3, 3, 27 for g = 7, 11, 13, 17, 19, 23;
the wide zone at 23 is the class +-7 mod 30, whose first anchor-open hit is 7g). d_0(M) is the
first opening past column 0. Budget inequality = the target (D), never a law.

## Pre-registered (written before any script was run) and the scorecard

P1 (question 1). Gears with zero exclusive kills in the window exist at every one of Q = 59,
173, 499, 997; every such gear lies above 0.8 Q; the count is one to six; no gear below Q/2 has
zero; so the set of gears with an exclusive in-window kill is a proper subset of the machine,
namely the machine minus a few top gears (the square gate of alignment-rules 4.1).
SCORE: three of four parts held, one REFUTED. Zero-exclusive gears are 53 at Q = 59 (0.898 Q),
167 at 173 (0.965 Q), 487 and 499 at 499 (0.976 Q, 1.0 Q) -- all above 0.89 Q, counts 1, 1, 2,
none below Q/2 (the caller's prediction held). But at Q = 997 EVERY one of the 166 gears has
an exclusive kill: the set is the whole machine there. "Exists at every Q" is refuted.

P2 (question 2). (a) Theorem: inside gear g's clean end zone the openings of M and of M minus g
coincide, so a record stretch of M inside g's zone is a blocked stretch of M minus g, impossible
when F(M minus g) < F(M); prediction: no record stretch lies inside any gear's zone at m7..m23,
including the gears with F(M minus g) = F(M). (b) Intersecting a zone is unavoidable for the
small gears, so "never inside" is testable only as containment. (c) The record-stretch set is
closed under the mirror x -> P - x - F and carries no other coincidence: no two starts differ
by a multiple of any 5g. (d) Positions at fractions 0.3-0.7 or at the period's ends.
SCORE: (a) HELD, and more sharply than predicted -- F(M minus g) < F(M) for EVERY gear g at
EVERY level m7..m23, so containment is excluded by the theorem in every cell of the F table;
the two cells where the theorem does not bite (F-2 at m11 with gear 11, F-2 at m17 with gear
13) also show zero containment. (b) HELD (at m19 all 20 record stretches intersect the zones
of both 7 and 11). (c) REFUTED as phrased: start differences divisible by 5g occur far above
chance (m17: 42 pairs agree mod 35 against 5.4 expected; 10 pairs agree at (5, 7, 11, 17)
against 0.03 expected). They are not a symmetry -- the symmetry group of the opening set is
Z/2, proved on record (alignment-rules 3.12) -- they are shared phases: see Results Q2b.
(d) HELD (m23: 0.34, 0.49, 0.51, 0.66; m17 and m19 also have stretches at 0.0014 / 0.9984 and
0.0001 / 0.9999, the period's ends).

P3 (question 3). d_0 = 2, 3, 3, 5, 5, 5, 7, 7, 7, 10 at m7..m41 continues below q' to 59 and
far beyond; d_0 is the column of the first twin prime pair above q at every level; 2 d_0 / F
falls to about 0.1 by m59; the mirror forces F_2(M) >= 2 d_0 and the shield d_0 > (q-1)/6 and
nothing about the window's upper end.  SCORE: HELD in every part (2 d_0 / F = 0.149 at m59).

P4 (question 4). (0, W] has FEWER openings than the period mean W prod(1-2/g), the ratio in
[0.8, 1.05] at every level to 997 drifting toward 1/(2e^{-gamma})^2 = 0.793; two mechanisms of
opposite sign (silent gears above sqrt(6k+1) versus the Mertens bias); the zero stretch sits at
a low-to-middle percentile among typical stretches of length W.
SCORE: direction and limit HELD (ratio 0.939 at Q = 997, falling monotonically by band from
Q = 120 on; 139 of 165 levels below the mean); the interval was WRONG at small Q (1.146 at
Q = 17, 1.088 in the band 60-120; 26 levels above 1); the percentile was UNDERSTATED: from
Q = 251 the zero stretch has fewer openings than 96-100 % of random stretches, and from Q = 401
fewer than every one of 1,000 (8 standard deviations below the mean at Q = 997).

P5 (question 5). No statement of the required kind exists; the region past zero degenerates
to "the primes below Q'^2 are what they are".  SCORE: HELD, with the degeneration made exact
(Mechanism, last paragraph).

## Setup

Q1 (q1_runs_window.py). For Q = 59, 173, 499, 997: W = (Q'^2-1)/6, window columns
Q//6+1 .. W. For every column the number of gears of {5..Q} striking it (two residue classes
per gear, marked by slicing). For every gear: runs covered = window length / 5g; strikes; hits
= strikes on anchor-open columns (k mod 5 in {0, 2, 3}); exclusive kills = hits on columns
struck by no other gear; the column and numbers of the first exclusive kill. Gate: openings
in the window = twin pairs (6k-1, 6k+1) with Q < 6k-1 and 6k+1 < Q'^2, plus the square gate
(column W holds (Q'^2-2, Q'^2), open iff Q'^2-2 is prime). Gate OK at all four Q.

Q2 (q2_record_position.py, q2b_record_classes.py). Full periods of {5..7} .. {5..23}
(P = 35 .. 37,182,145), openings by residue slicing, gaps with the wrap. For every gap equal to
F, F-1, F-2: the start x, the fraction (x+1)/P, the mirror partner P - x - L, the distance from
the stretch to the nearest multiple of 5g for every gear, and the zone relation (inside /
intersects / disjoint) against h_g; F(M minus g) for every g by a separate full period. q2b
prints every record stretch's kill map (which relative columns each gear strikes, exclusive
kills starred) and, for every pair of record stretches, the gears at which the two starts agree
mod g, against the chance rate.

Q3 (q3_d0.py). d_0 by smallest-prime-factor sieve for every prime q <= 33,317 (3,564 primes);
compared with q', W, the corpus F and F_2 ladders, and the first twin pair above q.

Q4 (q4_past_zero.py). For every prime Q from 7 to 997: openings of {5..Q} in (0, W] by residue
slicing (gated against the twin count with both boundary riders, OK at all 165 levels), the
period mean W prod(1-2/g), the shield (Q-1)//6, and the effective-machine prediction
sum over k of prod_{5 <= g <= sqrt(6k+1)} (1-2/g). Typical stretches of length W: the exact
sliding count over the whole period for Q <= 23; for 22 larger levels, 4,000 (Q < 200) or
1,000 (Q > 200) random stretches, a random start in [0, P) being an independent uniform residue
per gear. Then at Q = 59, 173, 499, 997: per gear the column of its first exclusive kill against
its square column (g^2-1)/6, and the band-by-band local opening count against the full-machine
mean and the effective-machine mean, with the number of gears still silent (g^2 > 6k+1) at the
band's top.

## Results

### Q1. The window in run units, per gear

Runs covered (window columns / 5g): at Q = 997 gear 7 sees 4,843 of its runs and gear 997 sees
34; at Q = 59 gear 7 sees 17.5 and gear 59 sees 2.1. Hits and exclusive kills, Q = 997 (full
tables in results/q1_runs_window.txt):

    g      runs   strikes   hits   exclusive   first exclusive kill
    7    4843.3    48433   29059     3921      column 400 = (2399, 2401): 2401 = 7^4 beside the prime 2399
    11   3082.1    30821   18493     2256      column 1993 = (11957, 11959 = 11 x 1087)
    43    788.4     7884    4730      512      column 308 = (1847, 1849 = 43^2)
    883    38.4      383     229        3      column 149963 = (899777, 899779 = 883 x 1019)
    991    34.2      342     204        1      column 166653 = (999917, 999919 = 991 x 1009)
    997    34.0      340     204        2      columns 167662 (997 x 1009), 168327 (997 x 1013)

Zero-exclusive gears: {53} at Q = 59, {167} at 173, {487, 499} at 499, none at 997. Every
exclusive kill of a top gear g is g x (a prime in (Q, Q'^2/g]) beside a prime, or g^2 beside
a prime (the square gate): at Q = 59, gear 43's five exclusive kills are 43 x 43, 43 x 61,
43 x 67, 43 x 73, 43 x 83, each beside a prime; gear 53 has none because 53^2 - 2 = 2807
= 7 x 401 and 53 x 61 +- 2 = 3231 = 3^2 x 359, 3235 = 5 x 647 (in column terms: no column of
the window is struck by 53 alone). The share of hits that are exclusive falls with g/Q: at
Q = 997 it is 0.110 for g < 0.1 Q, 0.054 for g in (0.5, 0.75) Q, 0.010 for g > 0.9 Q -- the top
gears waste 99 % of their hits on columns another gear already strikes. Exclusive kills as a
share of all blocked window columns: 0.370, 0.285, 0.224, 0.195 at Q = 59, 173, 499, 997.

What decides "which gears determine the window's openings": a gear with no exclusive kill can be
dropped without changing the window's openings, and the gears that cannot be dropped are those
owning a column g x m beside a prime with m prime in (Q, Q'^2/g] or m = g -- alignment-rules
4.1's "a gear is needed iff it owns a pseudo-twin in the window". The count of such gears is
the whole machine at Q = 997 and the whole machine minus one or two top gears at the three
smaller Q; whether a given top gear is droppable is decided by the primality of one or two
specific numbers (g^2 - 2, g Q' +- 2), which is why droppability is transient. No proper subset
of gears determines the window's openings at Q = 997, and no proper subset does so stably at
any Q. The exclusive-kill census gives no lever on existence: the columns it counts are
blocked columns, and their number (0.2-0.4 of the blocked columns) says nothing about where
the openings are.

### Q2. Where the record stretches sit

Record stretches (gap F) per level, all mirror-closed (partner P - x - F present in every case;
no self-mirror stretch at any level, as the mirror theorem requires for F > d_0):

    M        P          F   # stretches   fractions (x+1)/P                      F(M minus g) for g = 7, 11, ...
    {5..7}   35         5   2             0.371, 0.543                            2
    {5..11}  385        7   4             0.392, 0.413, 0.574, 0.595              4, 5
    {5..13}  5005       11  12            0.025 .. 0.974 (4 of 12 in 0.3-0.7)     6, 7, 7
    {5..17}  85085      18  20            0.0014 .. 0.9984 (10 of 20 in 0.3-0.7)  10, 13, 16, 11
    {5..19}  1616615    25  20            0.0001 .. 0.9999 (2 of 20 in 0.3-0.7)   13, 18, 21, 18, 18
    {5..23}  37182145   34  4             0.341, 0.489, 0.511, 0.659              17, 23, 25, 23, 25, 25

Runner-ups: gap F-1 is realised at m11 (4), m13 (12), m23 (2: fractions 0.151, 0.849) and NOT
at m7, m17, m19 (spectrum holes at F-1); gap F-2 at m7 (2), m11 (22), m17 (22), m19 (86),
m23 (8) and not at m13.

Clean end zones. Every gear is needed for the record at every level (F(M minus g) < F(M) for
all g, table above), so by the one-line theorem no record stretch, and no F-1 or F-2 stretch,
can lie inside any gear's clean end zone; the scan confirms zero containments in all 6 levels x
3 gap values x every gear. This is the exact form of the pre-registered rule and it carries no
positional content: it says the record stretch needs every gear's teeth, which is the L4
sole-striker statement of research/proof/pair_statement.md in zone language.

Intersection with a zone is common and irregular. Gear 7's zone (half-width 8, 43 % of all
columns): 0 of 2, 0 of 4, 0 of 12, 8 of 20, 20 of 20, 4 of 4 record stretches intersect it at
m7..m23. The top gear's zone: 2 of 4 (m11), 6 of 12 (m13), 12 of 20 (m17), 4 of 20 (m19), 0 of
4 (m23). At m23 the top gear's zone is wide (h = 27, 46 % of the columns) and all four record
stretches and both F-1 stretches are DISJOINT from it, sitting 38-39 columns from the nearest
multiple of 115 -- because the m23 record needs three kills by gear 23 at relative columns
4, 12, 27 or 7, 22, 30 (q2b), i.e. the stretch straddles the teeth +-4 mod 23 twice, which puts
it far from the phase where 23's teeth fall on 5-struck columns. That is the mechanism of
"away from the zone" at m23: the top gear's kill count in the record (1, 2, 1, 3 at
m13..m23, anchor-235.md section 8) sets how many of its teeth the stretch must cover, and the
zone is where those teeth are wasted. With one top-gear kill (m13, m19) the stretch can and does
sit next to a zone.

Distances to the nearest multiple of 5g (results/q2_record_position.txt): at m23 all four record
stretches contain a multiple of 35, 55 and 65 and sit 19-24 from a multiple of 85, 0-9 from one
of 95, 39 from one of 115 -- consistent with the base rates and with nothing else.

### Q2b. The structure of the record-stretch SET (the refutation of P2c, explained)

The kill maps show what the coincidences are. At m19 all 20 record stretches have the SAME
phase mod 35: (x+1) = 1 mod 5 and 6 mod 7 in every case, so gear 5 kills relative columns
{1, 4, 6, 9, 11, 14, 16, 19, 21, 24} and gear 7 kills {1, 3, 8, 10, 15, 17, 22, 24} in every
record stretch -- one mod-35 corridor phase carries every record (the corridor law,
alignment-rules 1.7, and corridor-resonance's pinned residues, in record-set form). Above the
anchor the 20 stretches split into 4 classes by the phase of 11, 16 by the phase of 13, and 20
by the phase of 19: the middle and top gears complete the same anchor-and-7 skeleton in several
ways. At m17: 4 classes mod 35 (sizes 4, 4, 6, 6), and 10 non-mirror pairs agree at
(5, 7, 11, 17) -- the anchor, gear 7, gear 11 AND the top gear 17 in the same phase, only gear
13 different (e.g. x = 117 and x = 32842: gear 13 kills relative {2, 11, 15} in one and {7, 11}
in the other, its exclusive kill at 11 in both). At m23: the two non-mirror pairs agree at
(5, 7, 23) -- anchor, gear 7, and the TOP gear in the same phase, kills by 23 at {4, 12, 27} in
both -- and differ at every middle gear 11, 13, 17, 19. At m13: 6 pairs agree at (5, 7, 11) and
4 at (5, 7, 13). So the record set has, beyond the mirror, a degeneracy in the MIDDLE gears: the
anchor-plus-7 phase and (from m17 on) the top gear's phase are fixed up to mirror, and the
middle gears have two or more completions. This is not a symmetry of the opening set (there is
none beyond the mirror) and it is not new as a mechanism -- it is the "made at the top" picture
of theory_tree.md branch 5 with the anchor's corridor underneath -- but the record-set
statement "same anchor phase, same top-gear phase, free middle" is the exact form of it at the
record, and it holds at every level from m17 on where it can be tested (m17, m23; at m19 the
top gear 19 has 20 distinct phases because its single kill has 20 places to be).

### Q3. d_0 and what the mirror at zero forces

    q     q'   d_0   2d_0   F     F_2    W      6d_0 -+ 1      first twin > q    2d_0/F   2d_0/W
    7     11   2     4      5     -      20     (11, 13)       (11, 13)          0.800    0.200
    11    13   3     6      7     11     28     (17, 19)       (17, 19)          0.857    0.214
    13    17   3     6      11    16     48     (17, 19)       same              0.545    0.125
    17    19   5     10     18    25     60     (29, 31)       same              0.556    0.167
    19    23   5     10     25    31     88     (29, 31)       same              0.400    0.114
    23    29   5     10     34    39     140    (29, 31)       same              0.294    0.071
    29    31   7     14     43    55     160    (41, 43)       same              0.326    0.088
    31    37   7     14     58    68     228    (41, 43)       same              0.241    0.061
    37    41   7     14     88    90     280    (41, 43)       same              0.159    0.050
    41    43   10    20     91    103    308    (59, 61)       same              0.220    0.065
    43    47   10    20     103   116    368    (59, 61)       same              0.194    0.054
    47    53   10    20     118   -      468    (59, 61)       same              0.169    0.043
    53    59   10    20     145   159    580    (59, 61)       same              0.138    0.035
    59    61   12    24     161   -      620    (71, 73)       same              0.149    0.039

d_0 is the column of the first twin prime pair above q at every level, and d_0 <= q' (hence
d_0 in the window by a factor W/d_0 = 10 .. 58) for all 3,564 primes to 33,317, maximum
d_0/q' = 0.2857 at q = 5 (the same gate research/proof/pair_statement.md ran to 10^6).

Exactly when d_0 is in the window: column d_0 has 6 d_0 -+ 1 both free of prime factors <= q;
such a number below q'^2 is prime unless it is q'^2 itself; so d_0 lies in (q/6, W] iff there is
a twin prime pair in (q, q'^2), and then d_0 is the column of the first one. The lower edge
d_0 > (q-1)/6 is forced by the shield (every column k <= (q-1)/6 has 6k+1 <= q, so its numbers
are gears or gear multiples). The upper edge d_0 <= W is twin-Bertrand at scale q'^2, open.

What the mirror forces: (i) the two gaps at column 0 are (d_0, d_0), so F_2(M) >= 2 d_0, and by
the deletion ladder F(M+q') >= F_2(M) >= 2 d_0 (theory_tree.md log 2026-09-04) -- a FLOOR on
the next record from the first twin above q; (ii) F(M) >= d_0 trivially; (iii) the parity lever
on record (alignment-rules 3.12). What it does not force: any upper bound on d_0, on F, or on
the position of the next opening after any column but 0. The floor is loose and getting looser:
F_2 / 2 d_0 = 1.8, 2.7, 2.5, 3.1, 3.9, 3.9, 4.9, 6.4, 5.2, 5.8, 8.0 at m11..m53.

### Q4. The stretch just past zero against the period

    Q     Q'    W        open(0,W]   mean W prod(1-2/g)   ratio   typical stretches of length W (mean, [min, max], percentile of the zero stretch)
    7     11    20       8           8.6                  0.933   8.6 [6, 11], 0.31-0.43 (exact, full period)
    13    17    48       16          14.2                 1.123   14.2 [10, 19], 0.78-0.93 (exact)
    17    19    60       18          15.7                 1.146   15.7 [9, 22], 0.86-0.96 (exact)
    23    29    140      30          29.9                 1.002   29.9 [18, 41], 0.42-0.60 (exact)
    29    31    160      30          31.9                 0.942   31.9 [23, 41], 0.17-0.30 (4000 random)
    59    61    620      92          88.2                 1.043   88.3 [72, 104], 0.76-0.82
    101   103   1768     209         199.1                1.050   199.2 [173, 224], 0.90-0.92
    173   179   5340     480         487.0                0.986   487.1 [451, 525], 0.25-0.28
    251   257   11008    852         878.7                0.970   878.2 [825, 932], 0.035-0.040 (1000 random)
    401   409   27880    1843        1904.6               0.968   1903.9 [1826, 1976], 0.003-0.004
    499   503   42168    2585        2683.1               0.963   2683.1 [2588, 2766], 0.000
    997   1009  169680   8278        8812.8               0.939   8816.9 [8630, 9019], 0.000

By band of Q the ratio open/mean is 1.026 (Q < 30), 1.014 (30-60), 1.047 (60-120), 0.991
(120-250), 0.968 (250-500), 0.948 (500-1000), heading for 1/(2e^{-gamma})^2 = 0.793. The
zero stretch has fewer openings than the period mean at 139 of 165 levels, and from Q = 401 on
fewer than every one of 1,000 random stretches of its length.

Against the effective machine {5..sqrt(6k+1)} at each column, the ratio is 0.79 +- 0.01 at
every level from Q = 100 to 997 (0.785 at 173, 0.795 at 499, 0.795 at 997) and 0.72-0.82 below.

Band by band at Q = 997 (W = 169,680): columns 1..3393 hold 311 openings against a
full-machine mean of 176 (ratio 1.77) with 134 of the 166 gears still silent (g^2 > 6k+1);
columns 16969..33936: 951 against 881 (1.08), 81 gears silent; the top band 144229..169680:
1092 against 1322 (0.83), no gear silent. Against the effective machine every band reads
0.73-0.82. The same shape at Q = 59, 173, 499 (results/q4_past_zero.txt).

## Mechanism

The region past zero is the prime sieve, and the counts say so exactly. Gear g's strikes in
(0, W] land at the columns of g x m; while g x m < g^2 the cofactor m < g has all its prime
factors below g, so the column is already struck by those factors -- g's first EXCLUSIVE kill is
at its own square, column (g^2-1)/6, when g^2 - 2 is prime (45 of 166 gears at Q = 997; the
only gear with an earlier exclusive kill is the top gear's self-pair (Q, Q+2) at the bottom
edge, seen at Q = 59). So at column k only the gears with g^2 <= 6k+1 have done anything a
smaller gear had not already done, and the openings of M in (0, k] are the openings of
{5..sqrt(6k+1)} -- which is the statement "6k -+ 1 are prime iff they have no factor up to their
square root". Two effects follow, of opposite sign, and both are visible in the band table:

(i) fewer effective gears, more openings: the local density prod_{g <= sqrt(6k+1)}(1-2/g)
exceeds the full-machine density by a factor that is 1.77 in the lowest band at Q = 997 and 1 in
the top band;

(ii) the Mertens bias: the true number of twin columns below x is (2e^{-gamma})^{-2} = 0.793
times what the product over primes up to sqrt(x) predicts, so every band sits at 0.73-0.82 of
the effective machine, and the whole stretch (0, W] at 0.79 of it for every Q >= 100.

The net ratio to the full-machine mean is (i) x (ii): above 1 while (i) dominates (Q < 120,
where the bottom bands are a large share of W), below 1 from Q = 120 on, and 0.79 in the limit.
The pre-registered "MORE openings past zero" is false from Q = 120 on and the excess below that
is (i), a consequence of the first-exclusive-kill-at-the-square fact and not of the clean
end zones: the zones (where g's teeth fall on 5-struck columns) are a property of the
(5, g) phase, and the same zones recur every 5g columns throughout the period; what is special
about zero is that ALL cofactors are small there, which is the square, not the zone.

The percentile result is the same fact seen from the period: a random stretch of length W has
its openings at the product density; the zero stretch has them at the twin density, which is
0.79 of the product asymptotically; the difference is 8 standard deviations at Q = 997. The
window is a low-opening outlier among the stretches of its length -- the opposite of a
mirror-induced enrichment.

Why the zero mirror adds nothing here: the mirror maps (0, W] to [-W, 0) and the two halves
carry the same openings (k open iff -k open), so the stretch [-W, W] centred on zero has exactly
2 x open(0, W] + 1 openings; the record stretch is never self-mirror (Q2, every level), so the
mirror pairs record stretches without constraining either; and the only column the mirror
fixes is 0, whose neighbourhood is the shield (blocked to (q-1)/6) and then d_0.

The degeneration, made exact (question 5). A statement about columns (0, W] that is provable
from "gear g's first tooth is at u_g" together with the CRT is a statement about the residue
pattern of {5..Q} on (0, W]; by the kernel identity (alignment-rules 4.1, both boundary riders
gated here at the 165 levels of q4 and the 4 of q1) that pattern IS the set of twin prime pairs in (Q, Q'^2) plus the
square gate. Hence (i) any such statement exact on the record is a statement about twin primes
below Q'^2, and (ii) one strong enough to force an opening in (q/6, W] is "there is a twin
prime pair in (Q, Q'^2)", twin-Bertrand at scale Q'^2. Nothing in the run frame or the zero
mirror bypasses this: the run frame supplies the effective-machine identity above, the mirror
supplies F_2 >= 2 d_0 (a floor) and the parity unit, and the exact-on-the-record statements we
found (the record needs every gear; the record set shares the anchor and top-gear phase) are
period-scale facts with no position in the window.

## Verdict

1. The window in run units is the whole machine, not a subset: at Q = 997 every gear makes an
   exclusive kill in the window; at Q = 59, 173, 499 exactly one or two top gears (53; 167;
   487, 499) make none, decided by the primality of g^2 - 2 and g Q' +- 2 -- the square gate.
   The caller's "none below Q/2" held with room (smallest zero-exclusive gear 0.898 Q).
2. The record stretch is never inside any gear's clean end zone, and the reason is a theorem
   with no window content: every gear is needed for the record (F(M minus g) < F(M) for every g,
   m7..m23), so the stretch cannot live where a gear kills nothing. The record set's symmetry is
   the mirror alone; its extra coincidences are the anchor-plus-7 phase and the top gear's phase
   held fixed while the middle gears complete the stretch in two or more ways.
3. d_0 is the column of the first twin prime pair above q at every level (to 33,317), lies in
   the window by a factor 10-58, and the mirror forces F_2 >= 2 d_0 -- a floor whose slack grows
   (F_2 / 2 d_0 = 8 at m53). No upper bound comes from the mirror.
4. The stretch past zero has FEWER openings than a typical stretch of the period, not more:
   ratio 0.94 at Q = 997 falling to 0.79, below every one of 1,000 random stretches from Q = 401.
   Mechanism: exclusive kills start at the square, so the effective machine at column k is
   {5..sqrt(6k+1)}, and the count is 0.79 x the effective product to within 1 % at every
   Q >= 100 -- Mertens and Hardy-Littlewood in run language.
5. Question 5: it degenerates. Every statement about (0, W] provable from the tooth positions
   is a statement about the twin primes below Q'^2, and the one that would give an opening in
   (q/6, W] is twin-Bertrand at scale Q'^2. Branch 7d yields no lever on existence; what it
   yields is the exact form of two period-scale facts (every gear needed; anchor and top-gear
   phases fixed in the record set) that belong to branch 5 of the tree.

## Dead ends (logged with the refuting instance)

- DEAD: "some proper subset of gears determines the window's openings" as a stable reduction --
  Q = 997, all 166 gears have an exclusive kill (results/q1_runs_window.txt); the droppable
  gears at the smaller Q are decided by one or two primality facts each.
- DEAD: "the zero window is enriched in openings by the alignment of all zones at 0" --
  Q = 997: 8,278 openings in (0, W] against a random-stretch range [8630, 9019] over 1,000
  samples; 139 of 165 levels below the period mean.
- DEAD: "the record stretch avoids the clean end zones" as a positional lever -- it is
  containment only, and containment is excluded by F(M minus g) < F(M), a statement with no
  position in it; intersection is common (m19: 20 of 20 record stretches intersect the zones of
  7 and 11).
- DEAD: "the mirror pairs are the only coincidence among record stretches" -- m17: 10 pairs of
  record stretches agree at (5, 7, 11, 17) against 0.03 expected by chance; m19: all 20 share one
  phase mod 35. Not a symmetry (the group is Z/2, on record); a degeneracy of the middle gears.
- DEAD: "a statement about (0, W] provable from first-tooth positions that forces an opening in
  (q/6, W]" -- it is twin-Bertrand at scale Q'^2 by the kernel identity, gated at 169 levels.
- Pre-registration misses, for the record: P1 "exists at every Q" (997 refutes); P4's interval
  (1.146 at Q = 17) and its percentile (0.000 from Q = 401, far lower than "low-to-middle").
