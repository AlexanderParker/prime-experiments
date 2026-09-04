# The anchor pattern in the window, measured literally (branch 7b)

Prover, round 34 (2026-09-05). Scripts in research/anchor235/r34/ (anchor_window.py the sweep,
mechanism.py the mechanism pass, tables.py the tables and the Buchstab check), results in
research/anchor235/r34/results/ (sweep_<anchor>.tsv one row per level, gears_<anchor>.npz every
gear of every level, gears_<anchor>_Q<Q>.tsv the named levels, mechanism.txt, tables.txt).
Nothing here is committed. Vocabulary as in docs/proof-search/alignment-rules.md section 0; the
anchor is a machine A = {5..a} whose period P_A = prod(5..a) is compared with the window length,
exactly as in anchor-235.md section 7.

The line of enquiry (the human's): the anchor's pattern repeats by CRT and one repeat covers the
window; the kernel fact says an opening of the whole machine {5..Q} inside the window is a twin
prime pair. The open part is what the gears ABOVE the anchor do to the anchor's openings inside
the window. This document measures that literally, gear by gear, and says what the numbers are.

## Setup (exact ranges)

Column k is the pair (6k-1, 6k+1). Gear g >= 5 strikes k iff k = +-u_g (mod g), 6 u_g = g -+ 1.
Level Q is a prime, Q' the next prime, machine {5..Q}.

  window columns   k_lo = floor((Q+1)/6) + 1  ..  k_hi = (Q'^2 - 1)/6 - 1
                   (6k-1 > Q and 6k+1 < Q'^2; the column whose upper member is Q'^2 is excluded
                   since it is never a twin and no gear of {5..Q} strikes it)
  section columns  k_s  = (Q^2 - 1)/6 + 1  ..  k_hi        (6k-1 > Q^2)
  W = k_hi - k_lo + 1 (window length), S = k_hi - k_s + 1 (section length).

Kernel fact on this range: a column open under {5..Q} in the window is a twin prime pair (a
composite member below Q'^2 has a prime factor below Q', hence <= Q, and it is coprime to 6).
The correctness test of every run is survivors = twins counted by a primality sieve.

Anchors: A_min = the smallest {5..a} with P_A >= W (so one repeat of A_min's pattern covers the
window); A = {5..13} (P = 5005, covers the window through Q' = 173, run beyond that anyway, where
the pattern repeats W/5005 times); A = {5..19} (P = 1,616,615, covers the window through
Q' = 3113). For A = {5..19} the sweep starts at Q = 23 (the anchor must lie inside the machine or
its openings are not a superset of the twins, and there must be a gear above it).

Per level and anchor: N_A = anchor openings in the window; for each gear g in (a, Q] ascending,
raw_g = anchor openings struck by g, fresh_g = currently surviving openings struck by g (the
survivors are then removed), N_cur(g) = survivors before g; D_raw = raw_g - 2 N_A / g,
D_fresh = fresh_g - 2 N_cur / g. Same on the section alone. Added after the smoke test (before
the sweep): N_live(g) = survivors at columns k >= (g^2 - 1)/6 and D_live = fresh_g - 2 N_live / g,
for the reason given in the Mechanism section.

Levels: every prime Q from 17 to 5000 (663 levels; A = {5..19} from 23, 661 levels). Runtime
four minutes per anchor, one core each.

## Pre-registered (written before the sweep ran)

Two facts I take as known before running, both from the record, both used to shape the guesses:

- Over a FULL period of a machine M with m gears, the openings in residue class r mod g number
  N_M / g with error below 3^m (lower-sieve.md section 5, the stride result); measured errors are
  a few units to m = 7. Call delta_M(g) = n_P(u_g) + n_P(-u_g) - 2 N_P / g, the full-period
  deviation of gear g's teeth on M. It is a fixed number for (M, g), independent of Q.
- A window shorter than the period is a sub-interval, and a count over a sub-interval is an
  interval-discrepancy question, not the stride result.

Consequence I expect and will test: for a gear g whose lower machine {5..g-1} has period
P_{g-1} <= W, the window holds f = W / P_{g-1} full copies and D_fresh(g) = f delta_{g-1}(g) +
(boundary term), i.e. LINEAR in W with a fixed slope; for a gear whose lower period exceeds W,
D_fresh is a sub-period interval discrepancy with no repeat to magnify it.

Scorecard (verdicts filled after the run, in the Results section):

(a) max_g |D_fresh| over the window's gears.
    Prediction: NOT bounded by a constant independent of Q for any of the three anchors, and the
    growth law differs by anchor.
    - Fixed anchor {5..13}: WORSE than sqrt - linear in W, attained at the first gear above the
      anchor (g = 17) or the second (19), value ~ (W / 5005) delta_A(17) with |delta_A(17)| in
      [0.5, 6.8]; at Q = 1499 (W ~ 380,000) between 40 and 500; at Q = 173 below 10. In
      relative terms D_fresh / N_cur at g = 17 is a CONSTANT in Q (the anchor's fixed residue
      bias at the teeth of 17), of size below 0.5%.
    - Fixed anchor {5..19}: same law with g = 23 and P = 1,616,615, so linear growth only past
      Q' ~ 3113; below that sqrt-like at g = 23: between 1 and 4 times sqrt(2 N_cur / 23).
    - A_min (anchor grows with Q so one repeat covers): sqrt-like, C sqrt(2 N_cur / g) with C in
      [1, 4] at the first gear above the anchor; at Q = 1499 (A_min = {5..19}, g = 23,
      2 N_A / 23 ~ 7,700) between 90 and 350; at Q = 997 between 60 and 240.
    - At the top gears (g within a factor 2 of Q) |D_fresh| <= 3 sqrt(2 N_cur / g) + 1, i.e.
      below 15 at every Q <= 5000, because fresh_g there is a count of primes m in a short
      interval with g m -+ 2 unstruck, an integer of size 0..12.
(b) sum_g fresh_g < N_A always (the margin is the twin count, never zero on the record, so this
    is free). The content is the size: margin / N_A = twins / N_A. Prediction: the ratio
    R(Q) = twins / (N_A prod_{a<g<=Q} (1 - 2/g)) lies in [0.79, 1.00] at every Q and decreases
    toward (e^gamma / 2)^2 = 0.793 (the two-dimensional Mertens factor for sieving to the
    square root); R ~ 0.90 at Q = 173 and ~ 0.85 at Q = 1499 for {5..13}. So margin / N_A falls
    like 1 / ln^2 Q for a fixed anchor, and for A_min it falls slower (the product starts at a
    growing a).
(c) Sign structure of D_fresh.
    - Mirror about the window's midpoint: NONE for the gears with P_{g-1} > W (left-half and
      right-half D_fresh uncorrelated, |r| < 0.2); POSITIVE agreement for the gears with
      P_{g-1} <= W (both halves carry the same fixed bias), sign agreement >= 80% there.
    - Class of g mod 30: NONE (per-class mean of D_fresh within 2 standard errors of 0 at every
      class; no class carries a sign bias above 60/40).
    - Teeth relative to the anchor's period: for gears with P_{g-1} <= W the sign of D_fresh is
      the sign of delta_{g-1}(g) in >= 90% of (Q, g) cases (mechanism: full-period bias times
      the number of repeats); for the others the sign is 50/50 and I name no structure.
    - Teeth against counterfactual teeth: the real teeth +-u_g are NOT distinguished. Ranking
      |D_fresh(real)| among the (g-1)/2 symmetric tooth pairs +-v mod g on the same survivor set,
      the percentile is uniform (mean 0.50 +- 0.05 over all gears at Q = 997, 1499).
(d) Section only (S ~ Q (Q'-Q) / 3 columns).
    - (a-sec) {5..13}: max_g |D_fresh| BOUNDED, below 25 at every Q <= 5000 (the section holds at
      most a few anchor periods, so the linear term is O(delta) and the rest is the anchor's
      interval discrepancy); A_min: sqrt-like, between 10 and 35 at Q = 1499 (g = 23,
      2 N_cur / 23 ~ 120); {5..19}: same as A_min once Q' > 714.
    - (b-sec) R_sec(Q) in [0.6, 1.2] with mean over Q in [0.8, 0.95]; noisier than the window
      because the section holds ~ 220 twins at Q = 1499 against ~ 2,500 in the window.
    - (c-sec) no mirror, no mod-30 class, no tooth distinction (same tests, same thresholds).
(e) The decisive question (brief item 6). Statement E(Q): fresh_g <= (2/g + e_g) N_cur(g) for
    every gear, with E = sum_g max(e_g, 0). Prediction: E(Q) is between 0.05 and 0.5 at every Q,
    growing like ln ln Q; the amount that would still leave one survivor is ln(N_A prod (1-2/g))
    ~ 7 to 8 at Q = 1499, so the record holds the statement with a factor >= 15 to spare - and
    the e_g at the top gears are single integers over N_cur, i.e. the statement holds there for
    the trivial reason that fresh_g <= 3 + (2/g) N_cur. I predict the mechanism gives NO handle
    from the teeth: D_fresh at the gears with P_{g-1} > W behaves as a random two-residue sample
    of the survivors (test (c) last item), and the honest name for E(Q) is the remainder term of
    Brun's sieve in the machine's coordinates.

Refutation criteria, stated now: (a) is refuted for {5..13} if max |D_fresh| at Q = 1499 is below
40 or its argmax is not in {17, 19, 23}; (a) A_min is refuted if C falls outside [1, 4]; (b) is
refuted by any R(Q) outside [0.75, 1.05] for Q >= 100; (c) tooth-distinction is refuted if the
mean percentile lies outside [0.40, 0.60]; (e) is refuted if E(Q) exceeds 1 anywhere, or if the
tooth percentile shows the real teeth systematically low (which would be a handle).

## Results

Correctness: at all 663 levels and all three anchors the survivors equal the twin count in the
window and in the section, and N_A - sum fresh = survivors (sweep_*.tsv column ok = 1 throughout).
A second assertion, added after the smoke test, also held at every gear of every level: no
fresh strike lands on a column below (g^2 - 1)/6 (see Mechanism, item 1).

One structural fact first, because it makes most of the tables anchor-independent: the survivor
set before gear g is the opening set of {5..g-1} in the window, whatever anchor was used, so
fresh_g, N_cur, D_fresh and D_live are IDENTICAL across the three anchors for every g above the
largest anchor gear (here g >= 29). The anchor choice changes only N_A, raw_g, D_raw and the
rows of the first few gears. The tables below are for A = {5..13}; the A_min and {5..19} rows for
their own first gears are given where they differ.

### Level table, A = {5..13} (window = [k_lo, k_hi], section = [k_s, k_hi])

  Q     Q'    W        S     N_A      twins   tw_sec  sum fresh  R      R_sec  max|D_fresh| @g (N_cur, fresh)   max|D_live| @g   max|D_raw| @g  max|D_sec| @g  E_win  E_live E_sec  room
  17    19    56       11    17       17      2       0          1.133  1.133  -2.0 @17 (17, 0)                  -0.2 @17         -2.0 @17       -0.2 @17       0.000  0.000  0.000  2.71
  59    61    609      39    177      91      5       86         1.072  1.159  -3.4 @53 (91, 0)                  +4.6 @43         +2.5 @47       +0.7 @43       0.052  0.232  0.196  4.44
  173   179   5310     351   1575     480     25      1095       0.992  0.775  +8.0 @67 (705, 29)                +11.1 @67        -5.6 @29       +2.7 @47       0.090  0.385  0.438  6.18
  499   503   42084    667   12486    2585    28      9901       0.965  0.656  +20.1 @73 (5472, 170)             +23.0 @73        -5.4 @211      +4.0 @43       0.108  0.360  0.757  7.89
  997   1009  169513   4011  50294    8278    191     42016      0.940  0.921  +42.6 @227 (13898, 165)           +48.6 @227       -5.6 @29       -6.3 @29       0.123  0.349  0.322  9.08
  1499  1511  380269   6019  112826   16595   224     96231      0.943  0.804  +58.7 @211 (32522, 367)           +64.3 @211       +6.0 @1297     +7.0 @53       0.124  0.326  0.426  9.78
  2999  3001  1500499  1999  445199   53804   54      391395     0.925  0.699  +104.8 @587 (88992, 408)          +116.0 @587      -6.5 @1583     -5.5 @17       0.137  0.320  0.918  10.97
  4999  5003  4170834  6667  1237501  130543  200     1106958    0.912  0.874  -186.3 @197 (366054, 3530)        +187.0 @577      +10.6 @257     +6.2 @229      0.147  0.335  0.611  11.87

  room = ln(N_A prod_{a<g<=Q}(1-2/g)), the amount sum_g e_g could reach before the product
  predicts fewer than one survivor. E_win = sum_g max(D_fresh/N_cur, 0), E_live the same on the
  live share, E_sec on the section. Window k-ranges: [4,59], [11,619], [30,5339], [84,42167],
  [167,169679], [251,380519], [501,1500999], [834,4171667].

A_min: a = 11 at Q = 17; 13 at 47..167; 17 at 173..701; 19 at 709..3089; 23 at 3109..4999.
N_A(A_min) = 19, 177, 1391, 11015, 39708, 89074, 351481, 892032 at the eight levels; every other
column of the table is the same to the printed precision except max|D_raw| (+3.9 @163, -6.4 @211,
+11.4 @109, -12.5 @1021, -14.0 @61, +19.8 @677 from Q = 173 on). A = {5..19}: N_A = 137 (Q = 59),
1241, 9850, 39708, 89074, 351481, 976982; max|D_raw| -3.2, -6.1, -7.4, +11.4, -12.5, -14.0, +13.3.

Extremes over all 663 levels (same for all anchors):
- window |D_fresh|: Q = 4951, g = 197, D = -187.4 (N_cur = 359372, fresh = 3461 against
  3648.4). The real pair +-u_197 is the LEAST struck of the 98 symmetric pairs (rank 0.005).
- live share |D_live|: Q = 4987, g = 577, D = +189.7 (N_live = 250796, fresh = 1059 against
  869.3). The real pair is the MOST struck of the 288 pairs (rank 0.998).
- section |D_fresh_sec|: Q = 4919, g = 59, D = +18.6 (N_sec = 2904, fresh = 117 against 98.4,
  3.1 standard deviations of the pair-count distribution, std 6.04); the window count of the
  same gear is 20308 against 20220.4, ratio 1.004 - this extreme is a fluctuation of an
  ordinary gear, the largest of ~400,000 section rows.
- |D_raw| over all rows: {5..13} +11.1 (Q = 2777, g = 1297); {5..19} -20.7 (Q = 3659, g = 43);
  A_min -26.3 (Q = 3671, g = 43, a = 23). Section: -11.8, +15.0, +14.9.

### Gear table at Q = 1499, A = {5..13} (window [251, 380519], W = 380269, S = 6019)

  g     t=ln g/ln Q'  N_cur    N_live   raw    fresh  D_raw   D_fresh  D_live   fresh/(2N_live/g)  rank   N_sec  fresh_sec  D_sec
  17    0.387         112826   112826   13272  13272  -1.6    -1.6     -1.6     1.000              0.375  1785   210        +0.0
  19    0.402         99554    99554    11876  10480  -0.4    +0.6     +0.6     1.000              0.056  1575   160        -5.8
  23    0.428         89074    89074    9805   7745   -6.0    -0.6     -0.6     1.000              0.091  1415   125        +2.0
  53    0.542         58148    58118    4262   2189   +4.4    -5.3     -4.1     0.998              0.192  927    42         +7.0   (section argmax)
  131   0.666         39387    39139    1721   575    -1.5    -26.3    -22.5    0.962              0.969  613    14         +4.6   (~Q^(2/3))
  211   0.731         32522    31931    1070   367    +0.6    +58.7    +64.3    1.213              0.995  504    6          +1.2   (window and live argmax)
  751   0.904         19348    14351    299    53     -1.5    +1.5     +14.8    1.387              0.996  312    2          +1.2   (~Q/2)
  1499  0.999         16596    225      149    1      -1.5    -21.1    +0.7     3.331              0.850  224    0          -0.3   (top)

  rank = percentile of |D_live(real teeth)| among the (g-1)/2 symmetric pairs +-v mod g on the
  same survivor set. The same eight-row excerpt at Q = 17, 59, 173, 499, 997, 2999, 4999 is in
  results/tables.txt (section T1), for both A = {5..13} and A_min.

### Scorecard verdicts

(a) max_g |D_fresh|: NOT bounded - CONFIRMED in the headline, REFUTED in every detail.
    - {5..13}: the argmax is NOT at g = 17, 19, 23 (it is 67, 73, 227, 211, 509, 587, 197 at
      Q = 173 .. 4999); the first gears over the anchor have |D_fresh| <= 6.4 at every level to
      5000, D_raw(17) fits a slope of -0.0004e-3 per column (zero; predicted -0.34e-3). REFUTED.
      Reason in Mechanism item 2: the deviation delta_A(g) is the count in ONE copy at ONE phase,
      and consecutive copies rotate through all g phases, so it cancels over g copies instead of
      accumulating - that is the stride result itself, which I mis-applied.
    - A_min: at the first gear above the anchor the ratio C = D_fresh / sqrt(2 N_cur / g) has
      |C| <= 0.59, 0.48, 0.38, 0.24, 0.11 for a = 11, 13, 17, 19, 23 (D between -18.1 and +17.0
      at a = 23, W ~ 4e6). REFUTED - the anchor is far MORE rigid than sqrt: bounded error.
    - The argmax gear's C over all levels: median 2.87, 90th percentile 5.4, max 7.10. Beyond
      [1, 4]. It is not a fluctuation: it is a bias with a law (Mechanism item 3), of size
      (beta(t) - 1) 2 N_cur / g, i.e. LINEAR in N_cur / g, at the band t = ln g / ln Q' in
      [0.6, 0.8]. Sign negative at t ~ 0.62 (Q = 4999: -186 at g = 197), positive at t ~ 0.75-0.8
      (Q = 2999: +105 at g = 587).
    - Top gears g > Q/2: max |D_fresh| 19.9, 24.5, 53.6 at Q = 997, 1499, 4999. REFUTED as
      stated (I wrote "below 15", thinking of g ~ Q; at g ~ Q/2 the fair share is still ~50).
      At g = Q itself: fresh = 1, 2, 0, 1, 0, 0 with N_live = 26, 28, 193, 225, 54, 200.
(b) sum fresh < N_A: trivially; margin / N_A = twins / N_A = 1.0, 0.514, 0.305, 0.207, 0.165,
    0.147, 0.121, 0.105 at the eight levels. R(Q): 1.133, 1.072, 0.992, 0.965, 0.940, 0.943,
    0.925, 0.912 - decreasing, direction CONFIRMED; values REFUTED (I said 0.90 at 173 and 0.85
    at 1499; two levels near Q = 100 have R = 1.058 > 1.05, the refutation threshold). The
    finite-size factor is the one I left out: pi_2(x) ~ 2 C_2 x / ln^2 x (1 + 2 / ln x + ...),
    which is 1.165 at x = 1511^2; 0.793 x 1.165 = 0.92, near the measured 0.943.
(c) Signs.
    - Halves: gears with P_{g-1} <= W (17, 19, 23): corr(D_left, D_right) = -0.64, sign
      agreement 0.31 - ANTI-correlated, REFUTED; the reason is the bounded total (a surplus in
      one half is paid back in the other). Gears with P_{g-1} > W and g <= Q/2: corr +0.32,
      agreement 0.77 (a bias common to both halves); g > Q/2: agreement 0.28 (the left half is
      dead for g > Q'/sqrt 2, Mechanism item 1). "|r| < 0.2, no structure" REFUTED both ways.
    - g mod 30 (every fifth level, gears with P_{g-1} > W, n ~ 5,500 per class): class means of
      D_fresh 16.9 .. 18.7 with SE 0.65, of D_live 37.5 .. 38.8 with SE 0.63; largest
      between-class difference 1.9 SE. NO class structure - CONFIRMED. (The common positive
      mean is the band law, not a class effect.)
    - sign(D_fresh) = sign(delta_{g-1}(g)) where P_{g-1} <= W: 0.655 of n = 1400. REFUTED.
    - Real teeth against counterfactual teeth: mean percentile 0.95 on the live share (0.998 for
      Q/4 < g <= Q/2, where the real pair is the most struck of all pairs in 99.9% of rows),
      0.89 on the whole window, 0.54 on the section. REFUTED for the window: the real teeth are
      distinguished, and in the direction of MORE strikes on the survivors they can reach.
      On the section the rank is undistinguished (0.54) because the counts are 0..10 and tie.
(d) Section: max |D_fresh_sec| = 18.6 over all levels (below 25 - CONFIRMED, though for the
    wrong reason: it is sqrt-scale noise on fair shares up to 230, not an anchor constant);
    A_min at Q = 1499: 7.0 (below the predicted [10, 35]). R_sec range 0.457 .. 1.339 (Q < 100),
    0.55 .. 1.02 for Q >= 500, band means 0.86, 0.80, 0.80, 0.80, 0.80 - range REFUTED below
    Q = 500 (5 to 30 twins per section), mean CONFIRMED. The section obeys the same band law
    as the window (Mechanism item 3, the non-integrated form).
(e) E_win = 0.09, 0.108, 0.123, 0.124, 0.137, 0.147 at Q = 173 .. 4999 (max 0.148 over all
    levels), room 6.2 .. 11.9: the statement holds with a factor >= 65. E_live = 0.32 .. 0.39,
    flat. E_sec reaches 1.12 (Q = 461) and 1.073 at Q = 269 against room_sec = 2.58: the section
    holds it with a factor 2.4 at worst (Q >= 100). "E > 1 anywhere" is HIT on the section -
    by integrality (fresh_sec in {0, 1, 2} against fair shares 0.1 .. 1 at hundreds of top
    gears, each positive part ~ 1 / N_sec). The signed sum is the exact quantity:
    sum_g D_fresh / N_cur = -ln R = +0.008, +0.035, +0.062, +0.059, +0.078, +0.092.
    The tooth percentile is HIGH, not low: no handle in that direction.

## Mechanism, in the machine's terms

Fix a level Q, the window [k_lo, k_hi] with x = Q'^2, and a gear g above the anchor. Everything
below is what the sweep shows column by column; the named theorems come at the end of the
section, after the machine has spoken.

1. Which columns gear g strikes fresh, and why the residues +-u_g are not positions. The tooth
   condition k = +-u_g (mod g) is, member by member, "g divides 6k -+ 1". So a strike is a column
   holding a multiple g m of the gear, m coprime to 6, and the strike is FRESH iff no lower gear
   already struck that column: no gear q < g divides m, and no gear q < g divides the partner
   g m -+ 2. Write m = 6K -+ 1 for its own column K = the "multiplier column". Then gear g's
   fresh strikes in the window are in bijection with the multiplier columns K in
   (g/6, x/(6g)) whose relevant member is open under {5..g-1} one-sidedly, with the partner
   g m -+ 2 open under {5..g-1} too. Three consequences the sweep measured:
   - Nothing below (g^2 - 1)/6. A multiplier m < g has a prime factor below g (m = 1 is the
     column of g itself, below k_lo). Checked at every gear of every level: zero fresh strikes
     below (g^2 - 1)/6, 400,000 gear-rows. The survivors below that column are twin pairs by
     the kernel fact one level down, and g cannot strike a twin pair above g. This is the
     DEAD ZONE, size N_dead(g); the identity D_fresh = -2 N_dead / g + D_live is exact.
     At Q = 1499, 227 of 233 gears have a non-empty dead zone; the dead zones remove in total
     sum_g 2 N_dead / g = 0.025 N_A (0.015 N_A at Q = 4999) from the gears' reach.
   - The multiplier columns sit g times LOWER in the window than the struck columns. The fair
     share 2 N_live / g is computed from survivors at columns up to x / 6, the strikes are
     computed from one-sided openings at columns up to x / (6 g). Whether these agree is a
     question about how the lower machine's opening density varies with POSITION in the window,
     and it does vary: below column g^2 / 6 the openings of {5..g-1} are exactly the twin pairs
     (both members prime), between g^2 / 6 and g^3 / 6 they are twin pairs plus pairs with one
     member g'-times-prime for a gear g' >= g, and only far above g^3 / 6 does the density reach
     what the period would give. So the one-sided openings the multipliers must be are DENSER
     than the period density when the multipliers are near g (they are then primes, and the
     primes near g are 1.78 times denser than the lower machine's period density), THINNER
     near g^2 (the one-sided openings there are the thinnest: primes have run out and the
     two-gear products have not yet arrived), and at the period density far above g^3.
   - The scale variable is therefore t = ln g / ln Q': the multipliers sit at size x / g =
     g^{2/t - 1}. t -> 1 puts them next to g (Q = 4999, g = 4999: m in (4999, 5007), at most two
     primes); t = 0.62 puts them at g^{2.2} (Q = 4999, g = 197: m up to 127,000 = 3.3 g^2, the
     thinnest zone); t < 0.55 puts them above g^{2.6} (Q = 4999, g = 29: m up to 863,000 =
     35,000 g^2, period density).

2. What forces the sign and the size of D_live (and so of D_fresh). The ratio
   fresh_g / (2 N_live / g) is a function of t alone, the same at every level from 173 to 4999:
   1.000 for t < 0.55; a dip to 0.957 at t = 0.62 (the multipliers in the thin zone near g^2);
   back to 1.000 at t = 0.68; then 1.12, 1.25, 1.40, 1.55, 1.72, 1.87 for t = 0.72, 0.77, 0.82,
   0.87, 0.92, 0.97 (tables.py T2, 230,000 gear-rows pooled over Q >= 500; per-level top-band
   ratio 1.52, 1.47, 1.46, 1.44, 1.44, 1.44 at Q = 173, 499, 997, 1499, 2999, 4999). The section
   obeys the same curve with slightly lower values in the rising part (1.12 .. 1.82) because it
   is a thin slice at size x with the multipliers at size x / g exactly, while the window
   integrates the multipliers from g up. So:
   - the SIGN of D_live is set by t: negative for t in (0.55, 0.67), positive above 0.7, zero
     below 0.55. At Q = 4999 the window extreme is negative (g = 197, t = 0.62, -186) and the
     live extreme positive (g = 577, t = 0.75, +187); at Q = 2999 both are g = 587 (t = 0.80,
     +105 / +116); at Q = 1499 both are g = 211 (t = 0.73, +59 / +64).
   - the SIZE is (beta(t) - 1) 2 N_live / g, LINEAR in the fair share, which is why the argmax
     sits at t in [0.6, 0.8]: large enough t for a bias, small enough g for a large share. The
     ratio C = D / sqrt(2 N_cur / g) at the argmax is 1.7, 1.6, 3.9, 3.4, 5.4, 6.0, -3.1 at the
     eight named levels - not a fluctuation scale.
   - the residual after the t-law is white: z = (fresh - beta(t) 2 N_live / g) / sqrt(2 N_live / g)
     has mean +0.02, std 1.004, max |z| 4.02 over 105,919 gear-rows at Q in [3000, 5000]
     (tables.py T3). At the named levels the largest residuals are +38 (g = 61, Q = 1499,
     z = 0.9), -66 (g = 281, Q = 2999, z = -2.3), -134 (g = 263, Q = 4999, z = -2.7).

3. The real teeth against the other (g-1)/2 tooth pairs on the same survivors. A counterfactual
   pair +-v strikes the survivors in residue classes v and g - v, which are just two classes of
   an even histogram: the survivors mod g at the argmax gear have class std 8.2 against
   sqrt(mean) 12.4 at Q = 1499, g = 211, and 22 against 43 at Q = 4951, g = 197 - the lower
   machine's classes mod g are more even than random. The real pair is different in kind: its
   classes hold exactly the columns of multiples of g, and by item 1 those are empty below
   (g^2 - 1)/6 and over-filled by the factor beta(t) above. Result: the real pair is the MOST
   struck of all pairs on the live share in 99.9% of the gears with Q/4 < g <= Q/2 (mean rank
   0.998), 0.944 for g > Q/2, 0.906 for g <= Q/4; and the LEAST struck at the thin-zone gears
   (rank 0.005 at Q = 4951, g = 197). On the section the ranks are undistinguished (0.54) because
   the counts there are 0..10 and tie. The real teeth are distinguished in the window, and in the
   direction of removing MORE of what they can reach.

4. What does nothing. The class of g mod 30: means of D_fresh per class 16.9 .. 18.7 (SE 0.65) and
   of D_live 37.5 .. 38.8 (SE 0.63) on every fifth level, differences below 2 SE. The anchor's
   residue classes: at every extreme gear the struck columns are spread over k mod 5, 7, 11, 13
   in proportion to the survivors (Q = 4951, g = 197: mod 5 [1165, 0, 1135, 1161] against
   [1153, 0, 1154, 1154]; Q = 4987, g = 577: mod 7 [223, 0, 212, 193, 215, 216] against 212 each).
   The position of the window inside the anchor's period: the anchor's openings, sorted mod any
   gear, deviate from fair by at most 9.68 (Q = 1499, all 233 gears, all residues) and 11.40
   (Q = 4999, 663 gears), against fair shares of 13,274 and 145,588 at g = 17; the first gear
   over any anchor has |D| <= 18.1 at W ~ 4e6 (A_min, a = 23), |C| <= 0.11 .. 0.59. Two halves of
   the window: anti-correlated for the gears whose lower period fits inside the window (17, 19,
   23: corr -0.64, agreement 0.31 - the total is bounded, a surplus in one half is paid back in
   the other), positively correlated for g <= Q/2 above that (agreement 0.77 - the t-bias is
   common to both halves), anti for g > Q/2 (the left half is dead for g > Q'/sqrt 2).

5. Why the anchor's own count is rigid (item 4, last part) - the one proved bound. The anchor's
   openings in residue r mod g are the openings of a RE-TOOTHED anchor (gear q's teeth at spacing
   d_q g^{-1} mod q) over an interval of length W / g in the multiplier coordinate t of
   k = r + g t. So the in-window deviation is bounded by the interval discrepancy of the
   re-toothed anchor. Computed exactly for {5..13} over every interval of every length
   (mechanism.py M5): 7.54 for the real teeth (at length 868; percentile 0.128 among the 180
   symmetric re-toothings), 14.09 at worst over the family. Hence |D_raw(g)| < 30 for every gear
   g and every window, a theorem for the fixed anchor {5..13}; measured maximum 11.1. The
   deviation does not grow with the number of repeats of the anchor because copy j of the anchor
   lands on residue phase -jP mod g, so the g consecutive copies cover the g phases once each
   and the per-copy deviation cancels; that is the stride result of lower-sieve.md section 5,
   which my pre-registration mis-applied as accumulating.

6. The top gear g = Q. Its live share is the section; its multipliers are the primes in
   (Q, Q'^2 / Q), an interval of length ~ 2 (Q' - Q); fresh_Q = 1, 2, 0, 1, 0, 0 at the named
   levels against fair live shares 0.04 .. 0.4. The gears with g > Q/2 still have fair shares up
   to ~ 50 and fresh counts up to 74 (Q = 1499); "at most three in-window hits" (lower-sieve.md
   section 5) is the g = Q statement only.

Names, last. Item 1's one-sided opening density at size g^u under gears below g is Buchstab's
function omega(u) / ln g (omega = 1/u on [1, 2], (u omega)' = omega(u - 1)), and item 2's curve is
omega(2/t - 1) / omega(2/t) (section) and its integral over the multiplier range (window): the
computed curves match the measured bands to 0.5% in every band (tables.py T2: 0.957 / 0.962,
1.117 / 1.120, 1.387 / 1.396, 1.717 / 1.721, 1.902 / 1.872). The 1.78 at t -> 1 is Mertens'
e^gamma. Item 5 is an interval-discrepancy bound on a periodic set. Item 2's white residual is
what the sieve literature calls the remainder term.

## What is new, and what it is worth to the route

Measured here and not located in the project record or, as far as I can tell, in print in this
form (no web access this round; harvester to check):

1. The in-window rigidity constant of the anchor (item 5): |D_raw(g)| < 30 for {5..13}, every
   gear, every window, from the exact re-toothed interval-discrepancy family (worst 14.09, real
   7.54, the real teeth at the 12.8th percentile of the family - the same favourable direction as
   the tooth-counterfactual-percentile entry, on a new statistic). Measured 11.1 to Q = 5000;
   26.3 for the growing anchor A_min. Worth to the route: it bounds the FIRST gear over the
   anchor exactly, and nothing after it - from the second gear on the survivors are the lower
   machine's pattern, not the anchor's.
2. The dead zone as an exact in-window identity (item 1), checked at 400,000 gear-rows: the
   residues +-u_g strike nothing below (g^2 - 1)/6, D_fresh = -2 N_dead / g + D_live, and the
   dead zones total 1.5 - 2.5% of N_A. Worth: it says where in the window a gear can act
   (columns above g^2 / 6 only) and it is the kernel fact one level down; it does not touch what
   the gear does inside its live zone.
3. The per-gear removal law on the real teeth (item 2), measured to 0.5% and stable in Q from
   173 to 4999, with its sign map in t and the window / section distinction (integrated versus
   pointwise). Worth: it locates and sizes every extreme of the sweep; it is a main term. It is
   provable as an asymptotic by the standard route and that proof would carry an error term of
   exactly the size the route cannot absorb.
4. The counterfactual-tooth ranking in the window (item 3): the real pair is the most-struck pair
   in 99.9% of the gears in (Q/4, Q/2] and the least-struck in the thin zone. Worth: it settles a
   pre-registered question in the unfavourable direction - the real teeth remove more, not less,
   of what they can reach - so there is no per-gear removal bound to be had from the teeth
   being special.

Prior art located for the rest: Buchstab (1937) for the density of one-sided openings by scale;
Mertens for the 1.78; the finite-size factor 1 + 2 / ln x in the twin count for R(Q); the stride
result (lower-sieve.md section 5) for the cancellation over copies.

## Verdict

Branch 7b stops here: from the second gear above the anchor onward, what each gear removes
inside the window is Buchstab's one-prime identity read gear by gear (the main term) plus a
white residual of size sqrt(2 N_live / g) (std 1.004 measured), and a bound on that residual
summed over the window's gears to relative accuracy 1 / ln^2 Q is the open sieve problem, not a
machine statement. The one object that is the machine's own - the anchor's in-window rigidity,
proved with constant 30 for {5..13} - is exhausted at the first gear.

The human's three parts, in this language: the anchor repeats (theorem); its openings enter the
window equidistributed mod every higher gear to a bounded error (theorem for {5..13}, measured
for all anchors); an opening of the whole machine in the window is a twin pair (theorem). The
gap between the second and third parts is that the higher gears act not on the anchor's pattern
but on each other's survivors, and the amount they remove from those survivors is set by where
the multiplier columns sit (item 1) plus noise; the record holds the statement "each gear
removes at most (2/g + e_g) N_cur with sum e_g^+ <= 0.15 in the window and <= 1.12 on the
section" at every level to 5000 (rooms 6 - 12 and 2.6 - 5), and the signed sum is exactly -ln R
in [-0.13, +0.09]. Nothing measured here bounds the noise part from the teeth. No docs/novel
entry written: items 1 - 4 above are for the harvester to price; my own reading is that 1 is
elementary, 2 is the kernel fact, and 3 - 4 are Buchstab on the real teeth.

## Dead ends (with the refuting instance)

- DEAD: "D_raw at the first gear over a fixed anchor grows linearly in W with slope
  delta_A(g) / P_A". Gear 17 over {5..13}: D_raw = -2.3, -0.7, -2.8, +0.4, -4.4 at Q = 4969 ..
  4999 against a predicted -1,400; fitted slope -0.0004e-3 per column. The copies rotate
  through the residues; delta cancels over g copies (the stride result, mis-applied by me).
- DEAD: "sign of D_fresh from delta_{g-1}(g) where the lower period fits": agreement 0.655 of
  1,400 rows (gears 17, 19, 23).
- DEAD: "positive half-window agreement for the small gears": corr -0.64, agreement 0.31 - the
  bounded total forces anti-correlation.
- DEAD: "the real teeth are undistinguished from counterfactual teeth in-window": mean rank
  0.95 on the live share, 0.998 for Q/4 < g <= Q/2 (Q = 1499, g = 211: 367 struck against pair
  mean 308, std 11, the largest of 105 pairs). Distinguished, in the direction of more strikes.
- DEAD: "an anchor-class structure in the struck columns": proportional at every extreme gear
  (item 5); the mechanism is multiplicative, not positional.
- DEAD as an invariant: the positive-part sum E on the section - 1.12 at Q = 461 (E > 1) from
  integrality at the top gears; the signed sum -ln R_sec is the quantity with content.
- DEAD as a handle: the tooth-distinction. It is the Buchstab excess, provable as an asymptotic
  and useless as a bound (a proof needs the white part, item 4).
- Not dead but not a route: the dead zone (item 1). It is the kernel fact one level down and
  removes 1.5-2.5% of N_A from the gears' reach in total; it does not touch the remainder.
