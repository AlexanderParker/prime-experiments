# mechanic workstream log

## Round 1 - the fragile census at scale (2026-08-18)

Script: `research/fragile_census.py` (segmented numpy sieve over slot space,
kappa_profile.py style). Run: `uv run python research/fragile_census.py 503
1009 2003 3001 5003 10007 20011 50021` - full prime sweep y = 13..503 plus
sparse large y; y = 50021 (window 4.17e8 slots, members to 2.5e9) takes 52 s.

### Definitions used (calibrated to reproduce the 13-window census exactly)

Window of y: slots k with a member in [y, y^2], i.e. k in
[ceil((y-1)/6), floor((y^2+1)/6)]. Degree of a member = number of distinct
gear divisors (primes 5 <= q <= y). Degree-0 member = prime > y.

- twin: both members degree-0 ((11,13) at y=13 is degree-2, not counted -
  matches the class-tree "9 twins").
- frag_loose: one member degree-0 prime, other composite with exactly one
  distinct gear divisor q (the owning gear); any shape q*p, q^2, q^3, q^2*p...
- frag_semi: frag_loose with the composite a semiprime (not divisible by q^2,
  or equal to q^2).
- Boundary exclusion: a "composite" side that is literally the gear y itself
  (e.g. (29,31) at y=29) is prime, so neither twin nor fragile here.

Sanity anchor y=13: 9 twins, 10 loose fragile, 9 semi (the loose extra is
125 = 5^3). Matches the documented overlap-map census.

### Census (excerpt; full sweep is every prime 13..503 + extras)

    y       W(slots)   twins     fragS      fragL    S/tw   L/tw   pi_win
    13            27       9         9         10   1.000  1.111        33
    53           460      74       138        146   1.865  1.973       393
    101         1684     201       444        477   2.209  2.373      1226
    251        10459     818      2355       2553   2.879  3.121      6266
    503        42085    2585      8097       8833   3.132  3.417     22186
    1009      169513    8278     28822      31367   3.482  3.789     79661
    2003      668335   26870     99458     107839   3.701  4.013    283641
    5003     4170835  130543    530459     572769   4.063  4.388   1567037
    10007   16688341  440665   1894741    2038980   4.300  4.627   5767853
    20011   66736686 1508853   6792966    7288504   4.502  4.830  21356270
    50021  417008404 7816169  37028977   39576257   4.737  5.063 121535157

pi_win = degree-0 prime members in (y, y^2]. S1 (lone-composite members,
regardless of partner): 287,805,085 loose / 271,522,325 semi at y=50021 -
semiprime share of loose fragile is 93.6% there (10/9 loose/semi at y=13,
monotonically declining excess).

### How the ratio evolves: fragile/twins grows without bound, like lnln

The ratio is NOT settling: 1.11 (y=13) -> 2.37 (101) -> 3.42 (503) -> 4.63
(10007) -> 5.06 (50021). Candidate laws, measured:

- fragile ∝ twins: FAILS (ratio grows).
- fragile ∝ W/ln^3(y^2): FAILS (normalised column grows 50 -> 962).
- fragile ∝ pi(y^2)/ln(y^2): FAILS (column grows 1.3 -> 7.0).
- fragile/twins = a*lnln(y^2) + b: FIT, a=3.01, b=-4.48 (semi) /
  a=3.22, b=-4.74 (loose), max abs residual 0.05 / 0.07 over y=101..20011.
  This is a two-parameter fit over ~2.3 decades of lnln - label: fit, not
  law. The lnln form is however the heuristic expectation (below).

### The sharp law (measured to ~1%): fragile = 2 * twins * W1 / pi_win

Model: for a window member m, P(partner prime) ~ P0 * prod_{r|m} (r-1)/(r-2)
(conditioning on r | m frees the partner from gear r: (1-2/r)/(1-1/r) ->
1/(1-1/r)). Summing over prime members: 2*twins = pi_win * P0 (each twin
seen from both sides). Summing over lone composites (weight (q-1)/(q-2) for
the one owning gear; the p-side factor (p-1)/(p-2) is negligible since
p > y): fragile = W1 * P0, where W1 = sum over lone-composite members of
(q-1)/(q-2). Eliminating P0 gives predicted constant

    c = fragile * pi_win / (twins * W1) = 2.

Measured (semi variant / loose variant):

    y      13     101    503    1009   2003   5003   10007  20011  50021
    cS   2.200  1.907  1.956  1.973  1.949  1.974  1.985  1.989  1.9914
    cL   2.245  1.907  1.964  1.978  1.950  1.974  1.985  1.989  1.9917

From y=1009 up the drift is monotone upward toward 2; at y=50021 the error
is 0.43%. Honest label: measured law, HL-consistent, constant derived not
fitted (zero free parameters). It says the fragile census carries no
information beyond (a) the window's partner-prime probability (same P0 that
makes twins from primes) and (b) the lone-composite population with its
(q-1)/(q-2) weight. The lnln growth of fragile/twins is then just
W1/pi_win ~ sum_q 1/q ~ lnln - Mertens divergence, nothing twin-specific.

### Owning-gear decile (share of loose fragile, gears ranked, 10 bins)

    y        d0    d1    d2    d3    d4    d5+
    101     58.3  13.2  13.2   4.4   2.9   8.0
    503     69.8  12.8   7.2   3.7   2.7   3.8
    2003    78.1   9.6   4.8   2.9   1.8   2.8
    10007   84.0   7.0   3.5   2.1   1.3   2.1
    50021   87.9   5.3   2.6   1.6   1.0   1.6

The bottom decile of gears owns a growing near-all of the fragile slots
(gear 5 alone dominates d0). Consistent with ownership frequency ~ 1/q per
gear: the low gears' coprimes are the deciding population, quantified - the
"densest source of fragile slots" reading of the coprime census, now with
its growth law.

### Caveats / discipline notes

- cS at small y (<= 251) oscillates 1.9-2.25: small counts + boundary slots.
  The claim "-> 2" rests on the monotone tail y >= 1009.
- pi_win excludes primes <= y; W1 is restricted to members in (y, y^2] for
  consistency. Fragile classification itself does not clip the top slot's
  member y^2+2; effect is O(1) per window.
- The lnln fit coefficients (3.01, -4.48) are fits; the constant 2 is not.

## Round 2 - per-gear closed form + prefix censuses (2026-08-18)

### Part 1: per-gear fragile counts vs the closed form

Script: `research/fragile_pergear.py` (same sieve, per-gear accumulation via
bincount; y=50021 in 84s). Two predictions tested per gear q:

    pred1(q) = 2*tw * ((q-1)/(q-2)) * S1(q)/pi_win          (raw counts)
    pred2(q) = 2*tw * ((q-1)/(q-2)) * S1w(q)/piw            (size-corrected)

where S1w(q) = sum of 1/ln(m) over lone-q members and piw = sum of 1/ln(m)
over degree-0 prime members - i.e. the partner-prime probability is taken
~ c/ln(m) per member instead of one window-wide average. Semi variant used.

Results (obs/pred by gear-rank band; z = (obs-pred)/sqrt(pred)):

    y=10007 (1228 gears)          obs/pred1    z1     obs/pred2    z2
      rank 0-50%                   0.9937    -8.64     1.0002     0.26
      rank 50-90%                  0.9523    -9.91     1.0018     0.36
      rank 90-99%                  0.9551    -1.56     1.0159     0.54
      rank 99-100%                 1.4477     1.86     1.5427     2.18
    y=50021 (5132 gears)
      rank 0-50%                   0.9963   -22.13     1.0002     1.32
      rank 50-90%                  0.9604   -31.72     1.0015     1.19
      rank 90-99%                  0.9459    -7.26     0.9955    -0.59
      rank 99-100%                 0.9539    -0.64     1.0055     0.07

Findings:
- The raw law has a real systematic: mid/large gears run 4-5% BELOW pred1
  (z ~ -30 at 50021). Cause identified and confirmed: member-size geometry.
  Gear q's lone composites live only in (q*y, y^2), where 1/ln(m) is below
  the average the twins imply; small gears' members spread like the primes.
  The 1/ln(m)-weighted pred2 removes the whole deficit: every band lands at
  1.000-1.016 at 10007 and 0.996-1.006 at 50021, |z| <= 1.4.
- Rare-event tail: the top-1% band excess seen at 10007 (1.54, z=2.2, 25
  events) does NOT persist at 50021 (1.0055, z=0.07, 186 events) -
  fluctuation. Individual top-10 gears: all |z2| <= 2.5 with counts 0-3
  matching pred2 ~ 0.1-1.7. The constant-2 regime holds where necessity
  events are rare; no twin-specific or necessity-specific structure appears
  even at the top-gear scale. Gear 50021 itself: S1=1 (its square), obs=1 -
  50021^2-2 is prime, the square-gate pseudo-twin, within its Bernoulli law.
- Upgraded law statement (measured, zero free parameters):
  frag(q) = 2*tw*((q-1)/(q-2))*S1w(q)/piw, exact to 2e-4 in aggregate and
  to band-level Poisson noise everywhere, y=10007 and 50021.

### Part 2: prefix censuses at the window bottom (for the Constructor)

Script: `research/prefix_census.py`; data: `research/data/prefix_census.csv`
(2400 rows: y ladder 101..100000007, t = 1..200, columns
y,t,k,member_lo,P,n0,n1,n2,margin with margin = t - P(t) = n2 - n0).
Primality by deterministic Miller-Rabin, so any 64-bit y is affordable
(only the first 200 slots are touched). Convention: P counts the member
equal to y itself as prime - open-interval users adjust slot 1 by -1.

Ladder summary (T=200):

    y          1st_dbl  1st_twin>y  minMargin  lastNeg  margin(200)
    101           4         2          -5        99         14
    503           3         4           0         0         29
    1009          4         3          -1        11         40
    10007         2         6          -1         1         73
    100003        2        26           0         0        100
    1000003       2         7           0         0        107
    10000019      2        21           0         0        129
    100000007     2         6           0         0        133

Statistics over 25 windows per decade (150 windows, T=200):

    decade   dbl_mean  dbl_max  tw>y_mean  tw>y_max  minM_min  lastNeg_max
    1e3        3.68       9        6.60       14        -1         11
    1e4        3.04       7       10.84       23        -1          4
    1e5        2.48       4       19.92       36        -1          1
    1e6        2.40       5       13.36       30        -1          2
    1e7        2.36       4       29.72       53        -1          2
    1e8        2.64       6       37.16       73        -1          2

Deep prefix t in [5, 200]: margin never negative for y >= 1e4 (125/125
windows; 9/25 negative at 1e3), touches 0 in ~11% of windows, and the
minimum is always achieved by t <= 11 - after which the margin climbs
roughly linearly (margin(200) ~ 14 at y=101 up to 133 at 1e8).

Readings for the bottom-band attack:
- Onset asymmetry is total: first double arrives at slot ~2-4 (max 9 in 150
  windows, no growth with y), first twin above y at ~ln^2 scale (mean 6.6 ->
  37 across five decades). Doubles outpace twins from the very bottom in
  reality.
- Identity worth stating: margin(t) < 0 forces n0(t) > 0 (margin = n2 - n0).
  So a C2/prefix-pigeonhole refutation of X is ALWAYS a nonconstructive
  twin-existence proof, and the data localises its reach: negativity only at
  t <= 4 for y >= 1e4 (boundary twin at slot 1-2), never in t >= 5. Raw
  prime counting cannot bite the bottom band beyond the first handful of
  slots - the Constructor's onset law needs a sharper invariant, e.g. the
  forced identity P(t) = t for all t below the first double under X
  (zero-slack: any early slot with 0 or 2 primes breaks X immediately if a
  double provably cannot appear before slot L > that t).

### Caveats

- 150 + 12 windows sampled for prefixes; "never negative for t >= 5" is a
  measured regularity on those, not a law. Poisson fluctuations of P over
  200 slots (~sd 5-7) make occasional future dips plausible near 1e3-1e4.
- Per-gear bands use gear RANK (percentile of the gear list), matching
  round 1's decile convention.

## Round 3 - full-window cumulative margin trajectories (2026-08-18)

Script: `research/margin_trajectory.py` (primality-only segmented sieve -
much lighter than the degree sieve: y=50021 full window in 10s, y=200003
(W = 6.67e9 slots, members to 4e10) in 186s). Data for the Constructor:
`research/data/margin_summary.csv`, `margin_checkpoints.csv`,
`margin_bands.csv` (append mode; delete to regenerate).

M(t) = t - P(t) over the whole window, P = prime members among first t
slots (boundary member y counts prime, as rounds 1-2).

### Structural fact stated first, then measured

M, n0, n1, n2 are functions of member primality ONLY - the margin
trajectory is gear-blind. Layer bands (fresh-gear activation at p^2) touch
attribution (which gear kills), never the census. So a band-boundary dip is
impossible unless prime density itself kinks at p^2 - and the measurement
agrees: slope of M over matched windows before/after every band boundary
(h adapted to boundary spacing, mid-band controls):

    y=20011:  band dslope +0.0001 +- 0.0004, control +0.0000 +- 0.0004
    y=50021:  band dslope -0.0005 +- 0.0003, control -0.0002 +- 0.0003
    y=200003: band dslope -0.0001 +- 0.0001, control -0.0001 +- 0.0001

Smooth through every boundary at 1e-4 slope precision. Consequence: the
cumulative statement cannot see layer bands through the margin; band
structure must enter through per-gear/attribution objects.

### The full-window summary (ladder; windows complete, no sampling)

    y        W(slots)    minM  t_min  frac_min   last<0  last<100  M(W)
    101          1684     -5     31    1.8e-2      99       554        457
    149          3676     -4     23    6.3e-3      86       545       1221
    211          7386     -2     13    1.8e-3      44       531       2805
    307         15658     -1      2    1.3e-4      28       515       6631
    401         26734     -1     11    4.1e-4      12       499      12060
    419         29191     -2      8    2.7e-4      18       497      13318
    503         42085      0      1    2.4e-5       0       482      19898
    1009       169513     -1      3    1.8e-5      11       441      89851
    2003       668335      0      1    1.5e-6       0       386     384693
    5003      4170835     -1      2    4.8e-7       4       320    2603797
    10007    16688341     -1      1    6.0e-8       1       266   10920487
    20011    66736686      0      1    1.5e-8       0       254   45380415
    50021   417008404     -1      1    2.4e-9       1       215  295473246
    100003 1666750002      0      1    6.0e-10      0       202 1211681063
    200003 6666833335      0      1    1.5e-10      0       182 4954846523

### Answers to the round-3 questions

1. Min-margin scaling: for y >= 503, minM is 0 or -1, at t_min <= 3, and
   the -1 events are the boundary twin at slot 1-2. There is no later dip
   ANYWHERE - verified over complete windows to 6.67e9 slots. For y < 403
   (member density > 1/slot at the bottom) the dip is real but shallow:
   min -5 at y=101, monotone shallowing to -1/-2 by y ~ 300-420.
2. Danger-zone end: NOT a fraction c of the window and NOT c*y - it is
   member-anchored and O(1) in slots. last<0 <= 11 for all y >= 503; as a
   window fraction it collapses 1.8e-2 -> 1.5e-10. The physics: drift
   dM/dt = 1 - 6/ln(member) turns positive at member e^6 ~ 403 and every
   y >= 503 window starts beyond it, so M is climbing from slot ~1 and
   never returns. The clean law is "M(t) > 0 for all t > t0 with t0 <= 11
   absolute slots" (measured on all 15 complete windows), not t > c*y.
3. Growth shape: M(t) = t - [li(6t+m0) - li(m0)], m0 = 6*k_lo - 1, to 0.1%
   for t > ~1e3 (checkpoints CSV has M vs model at 8 points/decade). It is
   asymptotically linear with slope 1 - 6/ln(member) rising slowly toward
   1; both pure-linear and t/ln t fits fail globally (M/t drifts 0.11 ->
   0.58 across a window; M/(t/ln t) drifts 0.4 -> 7.7).
4. Threshold escape: last t with M(t) < T matches the li-model inversion
   Mhat^-1(T) within a few % (T=100: 182-482 measured vs 195-508
   predicted; T=1e4: within 0.3%). Escape times DECREASE with y at fixed
   T (bigger members = faster early growth).
5. Prime-race envelope (empirical): max |M - Mhat| over all checkpoints is
   0.06-0.18 * sqrt(member), coefficient shrinking with y (0.058 at
   y=200003, deviation 10607 at member 4e10). A cumulative statement of
   the form M(t) >= Mhat(t) - 0.2*sqrt(6t+y) held at every checkpoint of
   every window tested.

### Caveats

- "No later dip" is exhaustive within each computed window (every slot
  checked, not sampled), for the 15 ladder y's. Between-ladder y's not
  computed this round; round-2's 150-window prefix sample supports the
  same at the bottom.
- The envelope 0.2*sqrt(m) is an observed bound at log-spaced checkpoints,
  not a sup over all t; the argmax could fall between checkpoints. minM /
  last_below columns ARE exact (every slot).
- li computed by trapezoid on 4000-point geomspace; error << 1 at these
  scales.

## Round 4 - per-gear supply trajectories R_q(t) (2026-08-18)

Script: `research/supply_trajectory.py` (lpf-attribution sieve: ascending
gears claim unclaimed members, so first claim = smallest prime factor;
y=50021 full window incl. the 13.2M-pair schedule in 85s). Data (append
mode): `research/data/supply_load.csv` (per-checkpoint load metrics),
`supply_pergear.csv` (R_q(t) for every gear at y<=2003, 24 log-spaced
representatives above).

Definitions. R_q(t) = composite members among the first t slots with
lpf = q. Supply identity sum_q R_q(t) = C(t) = 2t - P(t): asserted per
checkpoint, exact everywhere. Boundary: the member equal to y (prime) is
not attributed (consistent with rounds 1-3).

### (1) Band signature - verified definitionally, staircase graded

The margin was gear-blind; R_q(t) is where bands live, and the sieve was
verified against an independent spf-table count
  R_q(t) == #{c in [max(q, ceil(m0/q)), floor(m(t)/q)] : spf(c) >= q}
at every checkpoint for every gear with y^2/q <= 8e6 (all 302 gears at
y=2003): 0 mismatches in 3384/13892/23313/28764 checks at y=503/2003/
10007/50021. (One real bug caught by this: cofactors c < q belong to gear
lpf(c), not q - build-and-test discipline note.)

- Gears q <= sqrt(y): active from slot 1 - these are C4's servers of every
  prefix. No activation delay; composite-cofactor term dominates their
  staircase (T_q share of R_q at window end: 69% for gear 5 at y=503, 76%
  at 2003; declining in q).
- Fresh gears q in (sqrt(y), y): R_q = 0 until exactly
  t_act = (q^2-1)/6 - k_lo + 1 (activation = own square), then the
  layer-law staircase R_q(t) = 1 + pi(m(t)/q) - pi(q) + T_q(t) with
  T_q == 0 while m(t) < q^3 - EXACT (max measured T_q share for
  q > y^(2/3): 0.0000 at all four y). Worked examples y=2003: q=997:
  R(W) = 389 = 1 + pi(4024) - pi(997) exactly; q=1999: R(W) = 2 (square
  3996001 + one semiprime step 1999*2003).

### (2) Load under X and the pair-coincidence schedule

Per checkpoint: active set A(t) = pi(sqrt(m(t))) - 2; mean load
C(t)/A(t); gear 5's share of all supply; rho(t) = 2(t-P)/(2t-P) =
fraction of kills X forces into pair-coincidences; S_pair(t) = exact count
of nontrivial (cross) root-class hits k <= t over ALL gear pairs (the
roots-of-unity supply schedule, 2 classes mod qq' per pair, trivial
same-member roots excluded); tau(t) = (t-P)/S_pair(t) = X-demand share of
the schedule.

    y=50021 trajectory (excerpt):
    t          member       A     g5%    rho    tau   S_pair/n2
    133        50815        46    27.2  0.636  0.167    5.39
    13335      130027       70    27.2  0.642  0.160    5.53
    1333521    8051143      410   25.1  0.747  0.187    5.03
    133352143  800162875    3078  23.6  0.818  0.217    4.47
    417008404  2502100441   5132  23.4  0.829  0.222    4.38

    peak tau (always at t = W):  0.314 (y=503)  0.282 (2003)
                                 0.249 (10007)  0.222 (50021)

### (3) Answer to the key question

NO depth range exists where X's demand exceeds the freedom-free pair
schedule at the counting level - and none can appear at larger y:
- tau(t) rises monotonically through the window (no interior peak) and
  its maximum, at the window END, DECLINES with y: 0.31 -> 0.22 across
  the ladder. Slack is 3.2-4.5x and loosening (S_pair(W)/W grows like
  (sum 1/q)^2 ~ lnln^2 while demand/W -> 1).
- In reality t - P <= n2 <= S_pair holds identically (every cross-class
  hit lands on a double slot, every double slot is cross-hit), so no real
  window can ever exhibit a deficit; the computation quantifies how far
  X sits from the cap.
- Where the equation actually lives: compression. S_pair class hits must
  compress into n2 distinct slots; measured mean multiplicity
  S_pair/n2 = 4.38 at the 50021 window end vs the X-required
  S_pair/(t-P) = 4.50. The entire distance between reality and X is the
  n0 term (7.8M twin slots out of 3.03e8 doubles, 2.6%) - X demands the
  same class hits compress 2.6% harder, not more capacity. The
  contradiction's home, if anywhere in this frame, is the multiplicity/
  union structure of the cross classes (how hard root classes CAN
  overlap), not their count.

### Caveats

- S_pair excludes trivial roots by construction; it counts class hits
  with multiplicity (a slot cross-hit by 3 pairs contributes 3). The
  union equals n2 exactly (both directions verified by the identity
  n2 = t - P + n0 holding at every checkpoint).
- tau at t <= ~10 is noisy (counts of 0-2); the monotone claim is for
  t beyond ~100.
- Multiplicity growth S_pair/n2 ~ 4.4-5.5 measured, rising with y at
  fixed member scale but declining through each window; no closed form
  fitted this round (candidate: second moment of active-pair density).

## Round 5 - cross-root multiplicity distribution vs independence nulls (2026-08-18)

Script: `research/multiplicity_census.py`. Data (append):
`research/data/multiplicity_hist.csv` (full distributions: y, mu, real
count, expected counts under both nulls), `multiplicity_summary.csv`
(moments). y=50021 in 63s.

Key identity making this cheap: slot-cap => a slot's cross-pair
multiplicity is mu(k) = omega_G(mL) * omega_G(mR) exactly (omega_G =
distinct gear divisors; every unordered pair counted once). Cross-checks:
sum mu = S_pair and #{mu>=1} = n2 reproduce round 4's class-count values
exactly at all four y.

Models:
- NULL 1 (coordinator's): pairs' CRT classes independent across pairs;
  exact Poisson-binomial pmf via 128-point DFT of the PGF over all pairs
  (13.2M pairs at 50021), exact per-pair window class counts. Mean == real
  by construction.
- NULL 2 (decomposition): keep the product structure, break the
  arithmetic: mu' = omega'L * omega'R, sides independent, each omega' a
  Poisson-binomial over per-gear class counts.

Results (real / null1 / null2):

    y      mean    P0                cond=mean/(1-P0)      var           tail mu>=9
    503    1.508   .466/.220/.465    2.82/1.93/2.99        4.5/1.5/5.1   .025/.000/.020
    2003   2.039   .384/.129/.393    3.31/2.34/3.51        6.9/2.0/7.7   .044/.000/.038
    10007  2.631   .319/.072/.332    3.86/2.83/4.07       10.0/2.6/10.9  .065/.002/.064
    50021  3.185   .273/.041/.287    4.38/3.32/4.59       13.1/3.2/14.2  .088/.005/.091

Findings:
1. INDEPENDENT-PAIRS NULL MISSES P0 BY A FACTOR ~6.6 (0.041 vs 0.273 at
   50021). Its compression 3.32 vs real 4.38: the real machine compresses
   32-46% harder than independent pairs; excess ratio declines with y
   (1.46 -> 1.32) but the absolute gap grows (0.89 -> 1.06).
2. THE CARRIER IS THE PRODUCT STRUCTURE, i.e. variance + tail: real var is
   4.1x null1, tail mu>=9 is 16x null1 - and null2 (product kept,
   arithmetic broken) reproduces BOTH to a few % (var 14.2 vs 13.1, tail
   .091 vs .088) AND P0 to 1.4 points. Nothing beyond the omega-product
   structure is needed to explain the real multiplicity shape at this
   precision.
3. EXACT SLOT-CAP COVARIANCE: null2's mean exceeds real by
   sum_q p^L_q p^R_q -> 0.0911 = P_primezeta(2) - 1/4 - 1/9 (matched to 4
   decimals at y >= 2003; a gear cannot hit both members, independence
   pretends it can).
4. THE MOMENT STATEMENT THE CONSTRUCTOR ASKED FOR: the 4.38-vs-4.50 gap is
   a ZEROTH-moment statement and nothing else. cond = mean/(1-P0) with
   mean pinned by arithmetic, so
       cond_X - cond_real  <=>  P0_X = P0_real - n0/W.
   Variance and tail carry the real-vs-null1 excess but carry NONE of the
   X-gap: X leaves mu>=1 masses free apart from their sum. In null2
   language: X demands the zero-multiplicity mass sit BELOW the
   product-model prediction by ~n0/W (6.5% relative at 50021), while the
   real window sits 1.4 points BELOW null2 already - split as (a) null2's
   twin-mass overestimate: both-sides-zero mass 0.0242 predicted vs
   0.0187 real (ratio 0.85 -> 0.77 down the ladder, the HL-type
   correction the independent sieve misses), (b) a ~3% singles-mass
   deficit (0.254 real vs 0.263 predicted).
5. So the compression frontier reduces to: HOW LOW CAN P(mu=0) GO below
   the product-model baseline? The real machine undershoots null2 by
   1.3-1.4 points at scale; X needs ~n0/W more. The quantity to bound is
   the joint zero-mass P(omega_L=0 & omega_R=0) (the twin mass itself) -
   the singles mass tracks the model to 3%.

### Caveats
- Null1 pmf is exact Poisson-binomial (DFT), not Poisson approximation;
  per-pair p_i use exact window class counts (boundary slot included -
  O(1/W) convention mismatch vs the real hist's prime-y fix).
- Null2 sides treat the two members as independent; the slot-cap
  covariance shows up as the exact mean gap (item 3), not corrected away.
- "Tail >= 9" etc. are shares of ALL slots (not conditional on mu>=1).

## Round 6 - the zone's fate: sup R(y) to y = 10^7 (2026-08-18)

Scripts: `research/inversion_zone.py` (S1, M2, P and R(t) = (S1^2/M2)/(t-P)
exactly, every slot; full-window scans y <= 100003, prefix T = 8y above),
`research/twinmass_deciles.py` (part 3). Data (append):
`research/data/zone_summary.csv`, `zone_curves.csv` (S1/M2/P/R at dense
checkpoints - the Constructor's requested curves), `zone_anatomy.csv`,
`twinmass_deciles.csv`.

Calibration: sup R = 6.545 (503), 2.899 (2003) match the Constructor's 6.5,
2.9; zone extent at 10007 (3..17206) matches their [~5, 17204]. At tiny t
the raw sup is boundary-convention-sensitive AND circular (an actual twin
at slot 1-2 shrinks t-P and spikes R), so a convention-robust bulk sup
(t >= 64) is tracked alongside. Int64 overflow in M2*(t-P) at W ~ 1.7e9
caught and fixed (float cast) - one garbage row regenerated.

### (1) The sup R(y) curve - it crosses 1: generic forcing ends at y ~ 2-5e6

    y        scan      supR   t*     supB64   zone(lo,hi)    #zone
    503      full     6.545    27    2.652    (5, 2787)       2770
    2003     full     2.899    24    1.752    (6, 6547)       6523
    10007    full     1.929     6    1.305    (3, 17206)     17150
    20011    full     1.923     5    1.176    (5, 24887)     24815
    50021    full     3.000     7    1.103    (4, 40543)     40498
    100003   full     1.032   417    1.032    (26, 39859)    39375
    200003   T=8y     1.010  1637    1.010    (727, 6217)     2540
    500009   T=8y     1.056    39    0.983    (29, 50)          21
    1000003  T=8y     1.020   154    1.020    (104, 194)        79
    2000003  T=8y     1.031    14    1.003    (14, 72)          16
    5000011  T=8y     1.000     2    0.946    EMPTY              0
    10000019 T=8y     1.000     2    0.944    EMPTY              0

- The zone is nonempty through y = 2000003 (16 slots, sup 1.031) and EMPTY
  at 5000011 and 10000019 (sup R = 1.000 attained only as equality at the
  t=2 boundary). Near the threshold it flickers (500009's bulk sup dips to
  0.983 while a 21-slot zone survives at t in [29,50]) - bottom-band prime
  fluctuations, not a clean edge.
- Zone extent collapses toward the bottom before dying: zone_hi/y = 5.5
  (503) -> 1.7 (10007) -> 0.81 (50021) -> 0.40 (100003) -> 0.031 (200003)
  -> 0.0002 (2000003). Full scans to W = 1.67e9 slots confirm nothing ever
  reappears past the bottom zone; the T = 8y prefix has 20-40000x margin.
- Bulk sup declines smoothly: supB64 - 1 ~ y^(-0.6) (fit over 503..100003)
  until fluctuations dominate near 1. Density reading (labeled reading):
  R_bottom ~ eff * n2/(t-P) with eff ~ 0.92-0.97 stable (see anatomy), so
  the sup tracks bottom-band prime density 6/ln y fattening t-P; crossing
  at ln y ~ 15 matches the measured death at 2-5 x 10^6.

### (2) Anatomy at the argmax (zone_anatomy.csv)

m-histograms of the argmax prefixes are m = 0 slots plus a CONCENTRATED
block at m in {4, 6, 9, 12} = products of omega in {2,3,4}; m = 1, 2, 3
slots are essentially absent at the bottom (both members of a bottom-band
double generically carry 2-4 gear divisors; lone-gear members sit beside
primes instead - the fragile census). Concentration is the zone's engine:
CS efficiency (S1^2/M2)/n2 = 0.919-0.966 at every argmax. Worked (y=2003,
t*=24): hist {0:15, 4:7, 6:2}, S1=40, M2=184, CS=8.70 > t-P=3, forcing
n0 >= 6 - and indeed 6 twins sit in those 24 slots. Top M2 contributors
are always the 2-3 slots with m = 12/16 (omega 3x4, 4x4 members like
208943 = 7.11.13.209... etc.); S1 is carried by the m=4/6 bulk.

### (3) Depth-resolved twin mass (round-5 proposal folded in)

Real twin share per depth decile vs the depth-UNIFORM product baseline
(both-sides-zero mass), y=50021: ratio 0.982 (decile 0) declining smoothly
to 0.701 (decile 9); y=10007: 1.058 -> 0.706. Not band-structured: an
HL-shaped allocation ~ mean(1/ln^2 member) per decile reproduces the real
decile counts to 1.000 +- 0.003 (50021) - the global 0.77 is purely the
1/ln^2 density falloff averaged against a flat baseline. At 0.3% precision
there is NO depth structure beyond smooth density in the twin mass.

### Caveats

- supR at t < ~10 is convention-sensitive (boundary member y) and
  partially circular; use supB64 for trend claims. Both are in the CSV.
- T = 8y prefixes above 100003: justified by the collapsing zone_hi/y from
  full scans and by R(t) monotone-declining past the zone in zone_curves;
  not a proof of absence beyond 8y for y >= 2e5.
- The y^(-0.6) fit is a fit. The death location 2-5e6 is exact within the
  scan policy.

## Round 7 - saturated-run census to members 7.2e10 (2026-08-18)

Script: `research/saturated_runs.py`. One absolute segmented primality scan
of k = 1..1.2e10 (members to 7.2e10, 231s) - saturated (load-1) runs are
pure primality objects (no gears), so every window census is a truncation
of the absolute run list. Data (append): `research/data/satruns_ge10.csv`
(every run L >= 10 individually), `satruns_records.csv` (record
progression with side words), `satruns_renewal.csv` (per-decade counts,
L = 8..13+), `satruns_windows.csv` (per-window decile censuses, ladder
2003/10007/50021/200003). 21382 runs with L >= 8 held in memory; window
truncation at k_lo handled and flagged (0 truncation events occurred).

### (1) Max length: L = 13 STANDS across seven more decades - but as a
### record, not a wall

Record progression over 1.2e10 slots: L=10 at k=59 (member 353), then
L=13 at k=2452 (member 14711) - and NOTHING LONGER through member 7.2e10.
L=13 recurs: six instances total,

    k=2452        member 14711        word RLLRRLLLLRLRL
    k=61501443    member 369008657    word LLLRRLLLRRRLL
    k=874166593   member 5244999557   word RLLRRLLRRLRLL
    k=1909351447  member 11456108681  word LLLRRLLLRRRRL
    k=8472005085  member 50832030509  word RRRLLLRRRLLRL
    k=9599932213  member 57599593277  word LLRRLRLLRRRLL

L=12: 21 instances; L >= 10: 757. Bounded-forever claim NOT supported:
the measured L -> L+1 rate ratio at depth is ~0.3 (L=12: 10, L=13: 3 in
decade 10), so the first L=14 is heuristically expected within members
~1e11-1e12 - the scan stopped just short of its expected first arrival.
Label: measured record + heuristic extrapolation, not a law.

### (2) Absolute landmarks - with a correction and a refinement

Structurally exact: runs are primality-only objects, so every window sees
the same integers (censuses below confirm; y=2003 and 10007 both max at
k=2452). REFINEMENT: a window whose bottom excludes a landmark inherits
the next instance - y=50021 (bottom member 50021 > 14711) and y=200003
both have window-max L=13 at k=61501443. "Next record beyond 2452-2464"
= no L=14 anywhere scanned; the next L=13 is at member 3.69e8.

CORRECTION for Lateral: NONE of the six L=13 words is strictly
L/R-alternating (the original landmark reads RLLRRLLLLRLRL, containing an
LLLL block). "Perfect alternation" holds only in the load sense (exactly
one prime per slot = X's forced n1 pattern), not side-wise. Side words are
balanced but blocky - relevant input for the alternation-word study.

### (3) Renewal rate: GROWING per decade, rate ~ (6/ln m)^8 per slot

    decade(member)  slots        L8     L9    L10  L11  L12  L13+
    5               1.5e5        13      6      0    0    0    0
    6               1.5e6        48     15      1    1    1    0
    7               1.5e7       186     43     13    2    0    0
    8               1.5e8       769    146     45    9    2    1
    9               1.5e9      3435    703    122   28    8    1
    10 (1.03 dec)   1.03e10   12655   2445    433   73   10    3

L>=8 counts per decade: 19, 66, 244, 972, 4297, ~22600 (normalized) -
growth factor 3.5 -> 5 per decade, tending to 10 * (d/(d+1))^8. Per-slot
rate declines as (6/ln m)^8 (decade ratios match (d/(d+1))^8 to a few %),
so the per-decade count ~ 10^d * (ln)^-8 grows without bound: the object
X must kill is increasingly abundant with depth while its max length
crawls. Depth-decile structure inside a window (y=50021): smooth decline
(L8: 424 -> 113 from decile 0 to 9), consistent with pure density
falloff, no band anomalies.

### Caveats
- "L=13 unbeaten" is exhaustive to member 7.2e10 (every slot scanned),
  but the extrapolated L=14 arrival sits just beyond - do not lean on 13
  as an absolute constant.
- Renewal-rate model (6/ln)^8 is a fit to decade ratios; counts are exact.
- Side words recomputed independently by Miller-Rabin (assert pl != pr
  passed on all record runs - scan/MR cross-validation).

## Attempts ledger (round 8, per user direction)

Reframing of this workstream's results as attempt -> yield -> limiting
EVENT (not trend). Trend statements are demoted to observations; each
entry names the specific mechanism where the attempt stopped and what
that mechanism offers.

1. Fragile census (r1-2). YIELD: exact zero-parameter law frag(q) =
   2*tw*((q-1)/(q-2))*S1w(q)/piw, 2e-4 accuracy incl. rare tail.
   LIMITING EVENT: the law contains the factor "tw" - the fragile census
   is twin-count-calibrated, so it restates rather than constrains n0.
   MECHANISM TO EXPLOIT: the law is exact enough that any window
   violating it would carry information; none found - the machine is
   HL-exact at the fragile level.
2. Margin trajectories (r3). YIELD: M(t) = t - li-model to 0.1%; danger
   zone is member-anchored (drift sign flips at member e^6 ~ 403 - a
   pinpointable event, the ONLY absolute constant found so far).
   LIMITING EVENT: M is gear-blind (an identity, not a trend); bands
   cannot enter through the census. OFFER: attribution-level objects.
3. Supply/multiplicity (r4-5). YIELD: capacity never binds (exact);
   compression gap = zeroth moment = twin mass, exactly. LIMITING EVENT:
   mean is pinned by arithmetic, so all forcing must come from P(mu=0) -
   an identity that localizes the whole problem into one number per
   prefix. OFFER: that number's product-baseline is tracked to 1.4 points;
   the deviation IS structured (HL twin-mass ratio 0.85 -> 0.77).
4. Inversion zone (r6). OBSERVATION (not law): sup R decays ~ y^-0.6.
   EVENT: zone nonempty iff bottom-band prime surplus exceeds the CS
   inefficiency; revival at any y = a twin in that window's first slots.
   So "zone revives i.o." is an exact, floor-arithmetic-checkable
   restatement of the conjecture LOCALIZED to ~200 slots per window -
   an address, not a dead end. What kills generic revival is fattening
   t - P; what would revive it is exactly what we seek.
5. Saturated runs (r7-8). OBSERVATION: L* = 13 to member 7.2e10.
   EVENT PENDING: the constellation model predicts ~2.6 L=14 instances in
   (7.2e10, 1e12) - the running scan either assigns the first L=14 an
   address or produces a quantified deficit vs the model; both outcomes
   are events, and a deficit would be the more interesting one (it would
   mark the first measured departure of the machine from HL statistics).

### Ledger classification (round 8, continuing the reframe)

Every "closed" route in agents-shared, binned by the event-vs-trend
criterion:
- EVENTS (exact, mechanism visible): reduction iff; horizon; slot-cap;
  supply identity; Bridge identity; mirror moment-vacuity (k -> -k);
  roots-of-unity law; R > 1 => n0 >= 1; defect identity (defect = twin
  count, per slot); drift-sign flip at member e^6 ~ 403.
- TREND OBSERVATIONS in verdict clothing (real measurements, not walls):
  zone death y ~ 3-5e6; sup R ~ y^-0.6; C_CS/M_X growth 1.26 -> 1.58;
  L* = 13; renewal rates; twin-mass ratio 0.77-0.85 drifting.
- IMPORTED HUMAN LIMIT (not a machine event): T1 closed at "exponent
  0.525 vs needed 0.5" - a fact about the published corpus, not about
  the integers. The unexamined machine event underneath: thinnest layer
  bands sit exactly at twin endpoints (the self-reference at the binding
  case) - flagged as candidate reopening; nobody has interrogated it as
  a mechanism rather than an obstacle.

## Round 9 - THE FIRST L=14: address found, model validated (2026-08-18)

Scope note: this round touches only this file, my agents-shared round
append, and research/ files, per the standing scope rule.

### Data provenance (the round-8 kill, resolved)

The round-8 full-range scan was killed during its FINAL PRINTS - after its
CSV writes. research/data/satruns_deep_ge10.csv turned out to hold the
complete range k in [1.2e10, 1.67e11] (3055 runs L >= 10, max k 1.669e11,
zero duplicates). The detached chunk-flushed rescan (launched per the
coordinator's Start-Process pattern) was therefore redundant and was
stopped after precise identification of its PID tree (satruns processes
only; the manager's detached jobs untouched); CSVs verified clean of its
partial appends (0 mismatched rows).

### (1) The verdict: L = 14 EXISTS. New landmark.

    k_start = 46,133,660,494   members 276,801,962,963 .. 276,801,963,043
    L = 14   word LRRLRLRRRRLLRL (blocky, not alternating - consistent
    with the strict-alternation cap of 6)

Independently verified by Miller-Rabin: all 14 slots exactly one prime
member, both boundary slots both-composite (maximality). Appended to
satruns_records.csv. The record progression is now 10 (k=59), 13 (k=2452),
14 (k=4.6e10) - L*=13 stood from member 1.5e4 to 2.8e11, and fell exactly
where the constellation model said it should (below).

### (2) Record-growth law vs the CRT cap [13, 32]

Full-data refit (research/satruns_model.py; deep range folded in):
A_L stable per decade, global A_8..A_13 = 0.252, 0.220, 0.174, 0.135,
0.084, 0.119; log-linear A_L = exp(-0.197L + 0.197) [fit]. Validation:
predicted first L=14 near member 1.6e11 (expected count at the actual
address: 1.2) - found at 2.8e11. Poisson-consistent; NO deficit vs HL
statistics; L*=13 was a record on the curve, never a wall.

Predicted first-arrival ladder to the cap [fit, not law]:
    L=15: ~5e12    L=16: ~2e14    L=17: ~7e15    L=18: ~3e17
    L=20: ~6e20    L=24: ~5e27    L=28: ~9e34    L=32: ~3e42 (CRT cap)
Reading: the cap [13,32] splits into reachable records (15-16, hours-to-
days of compute), astronomical ones (17-31), and the absolute CRT ceiling
at 32 - record growth is ~ +1 length per factor ~40 in member. The L=15
hunt (members to ~1e13, ~6x the last scan) is priced and optional.

### (3) Renewal-rate law refit

Per-slot rate of L >= 8 runs ~ C/(ln m)^beta with beta = 6.81, C = e^8.33
(8 decades, max ln-residual 0.24) [fit]. Naive per-slot independence gives
~8 for the L >= 8 mixture; the flattening to 6.8 is the mixture + HL
corrections. Counts per member-decade keep growing (~19 -> 91k across
decades 5..11); the L >= 13 population now stands at 19 instances
(6 round-7 + 12 new L=13 + the L=14), all MR-verified.

### Caveats
- The ladder beyond L=16 rests on the log-linear A_L extrapolation over
  [8,13] - honest error bars are a factor of several in M(L); the
  qualitative split (reachable / astronomical / capped) is robust.
- beta fit excludes decades with < 10 runs.

## Round 10 - the T1 reopening: thinnest-bands-at-twin-endpoints (2026-08-18)

Scope: this file, my agents-shared append, research/ files only.
Tool: research/band_census.py; data: research/data/band_census_100003.csv
(9,591 bands, heights to 1e10, every slot counted exactly, 124s) and
band_census_2003.csv (calibration).

### (1) Precise event definition

Bands B_i = (p_i^2, p_{i+1}^2), consecutive primes >= 5; in slot space
kb_i < k <= kb_{i+1} with kb = (p^2-1)/6 (exact integer). Thickness
T = (p'^2 - p^2)/6 = g(2p+g)/6, g = the gap. The event as flagged
("thinnest bands sit at twin endpoints") is EXACT BUT TRIVIAL: T is
monotone in g at a height, and g = 2 <=> twin endpoints. The non-trivial
machine event underneath, derived this round (algebra, then verified per
band): for a twin (p, p+2) = (6m-1, 6m+1), T = 4m exactly and the twin's
own PRODUCT SLOT k = 6m^2 sits at offset 2m = T/2 - the exact center of
the band - with L member 36m^2-1 = p(p+2) composite BY the defining twin.
Verified 1223/1223 g=2 bands (plus 60/60 at calibration scale). Every
twin pre-blocks the center of the thinnest band above it: the descent's
self-reference, as one deterministic dead slot per thin band.

### (2) Census at scale (exact counts)

- Per gap class (all heights pooled): twin density per slot is FLAT in g:
  0.0166 (g=2) vs 0.0160-0.0185 (all other g; spread = height mix).
- Decade-matched, the decisive table: g=2 twin density / all-band density
  = 0.984, 1.018, 1.006, 1.002 at height decades 6-9 (center-slot-
  excluded: 0.985, 1.019, 1.006, 1.002). No gap-2 deficit at 0.2-2%
  precision; the exact center-slot deficit is 1/T = 1/(4m), invisible at
  scale.
- Twin-EMPTY bands: ZERO, any gap class, through height 1e10 (min twins
  per band = 2, attained only by the first band (25,49) with 4 slots).
  At heights [1e9, 1e10): worst band = 342 twins in 21,352 slots
  (p=32027, g=2) - exactly its Poisson expectation lambda ~ 342, and the
  worst band is g=2 only because g=2 bands are SHORTEST.
- T1 side: min prime members per band = 6 (the (25,49) band); no band
  approaches prime-emptiness anywhere in range.
- Fragile centers (36m^2+1 prime beside the dead product): 93/1223 = 7.6%
  at P=1e5 (15.0% at P=2003) - declining ~1/ln, density-consistent.

### (3) Verdict

Split verdict, each part exact:
- EXACT LAW (trivial): thinnest <=> twin endpoints, via T = g(2p+g)/6.
- EXACT LAW (new, mechanical): the center-slot pre-block - product slot
  k = 6m^2 at offset T/2, dead by construction, in every twin band. This
  is the entire deterministic content of the self-reference.
- DENSITY ARTIFACT (the rest): thin bands are NOT twin-poor. Per-slot
  twin density in twin-endpoint bands equals the generic density to
  measurement precision at every matched height, and the one dead slot
  is the only deterministic obstruction. What kills the "hostile thin
  bands" reading: the deterministic pre-block does not propagate - the
  remaining 4m-1 slots are statistically generic (Poisson-consistent
  minima).
- Consequence for the descent: the binding case is binding by LENGTH
  ALONE (T ~ (2/3)sqrt(x) at twin endpoints vs g/2-times longer
  elsewhere), i.e. exactly the imported Legendre-class localisation
  problem; the machine adds no obstruction of its own beyond one
  quantified dead slot per thin band. The reopening closes with a
  clean event ledger: self-reference = 1 slot, everything else generic.

### L=15 hunt status (parallel, detached)
Launched via Start-Process (wrapper PID 18504), log
research/data/satruns_L15.log, chunk-flushed + resumable (state file).
Target members ~1.2e13 (K = 2e12 slots), predicted first L=15 at ~5e12.
Chunk 1 flushed at report time (8.2%, ~232s/chunk, ~15h total). Round-8
renewal CSV preserved as satruns_deep_renewal_r8.csv; model loader
handles both schemas.

### Caveats
- "No empty bands" is exhaustive to height 1e10; beyond that it is the
  usual constellation expectation, not data.
- The 0.2-2% density-ratio precision is set by per-decade twin counts;
  no correction for the (tiny) fragile-center correlation was attempted.

## Round 11 - fuel census at scale: k_max = 4 exists, arithmetic-selected (2026-08-18)

Scope: this file, my agents-shared append, research/ files; plus one
user-direct instruction executed and flagged (human.md rewritten in place
as a maintained status snapshot - Alex's direction mid-round).

Tool: research/fuel_census.py (streamed numpy fuel census; counts
co-deletable k-TUPLES N_k - convention-free, equal to maximal-run counts
when k_max <= k; window condition via offset letters {+s, -s, 0 mod q}
with prefix-sum range <= 1; segment-boundary words counted by last-element
newness). Data: research/data/fuel_census.csv. VALIDATION: N3 = 62 at
19->23 with anatomy (8,15)/(15,8) - the corpus fuel census exactly.

### (1) Census across machine steps (full periods through 3.34e10)

    step      period    N2          N3      N4    k_max  N3/N2    N4/N3
    13->17    5.0e3     72          0       0     2      -        -
    17->19    8.5e4     1088        0       0     2      -        -
    19->23    1.6e6     11784       62      0     3      5.3e-3   -
    23->29    3.7e7     243816      0       0     2      0        -
    29->31    1.1e9     8022924     13000   4     4      1.6e-3   3.1e-4
    31->37    3.3e10    114848070   70964   216   4      6.2e-4   3.0e-3

    off-step probes: (19,29) 0 k=3; (19,31) 4; (19,37) 0; (23,31) 276;
    (23,37/41) 0; (29,37) 374; (29,41/43) 0; (31,41) 2; (31,43/47) 0.

k=4 instances: 29->31 has exactly 4 per period, one word class (10,21,10)
= (q-s, s, q-s), two mirror pairs, flanks {4,7}; 31->37 has 216, BOTH
orientations (12,25,12) and (25,12,25), flanks in {1,2,3,5,6,10,11,13}.
All addresses in the census output (research/data + task logs). N5 = 0
everywhere scanned.

### (2) Scaling law - the events, not a trend

- k_max on consecutive steps: 2, 2, 3, 2, 4, 4. It GROWS, glacially, and
  non-monotonically. Mechanism (exact, visible in the off-step table):
  N3 > 0 iff BOTH s = 3^-1 mod q and q-s land on abundant gap values of
  the machine's spectrum; k=4 needs the alternating word (s,q-s,s) or its
  mirror realized by consecutive gaps. Fuel is ARITHMETIC-SELECTED, not
  smooth in y: (23,29) has zero k=3 while (23,31) has 276.
- The k=3/k=2 ratio does NOT trend monotonically (5.3e-3, 0, 1.6e-3,
  6.2e-4 on consecutive steps); the k=4/k=3 ratio THICKENED 3.1e-4 ->
  3.0e-3 across the one step-pair where both exist. Per-opening k=4 rate:
  1.9e-8 (29->31) -> 3.5e-8 (31->37).
- Relative to the k_max = o(ln y) requirement: k_max = 4 at ln y = 3.4;
  two +1 events in six steps; no cap evidence either way yet. The first
  k=5 needs word (s,q-s,s,q-s) (span 2q', interior gaps all <= F_k -
  admissible in principle from 31->37 on); the 37->41 partial scan
  (running) is the live k=5 hunt (s=14, q-s=27).

### (3) Chain condition verified at two new scales + spectra

pred(F_k(M+q')) from the census's flanked merged spans = ACTUAL:
    29->31: pred 58 = F(2,31)/3 = 174/3  (period 1.1e9)
    31->37: pred 88 = F(2,37)/3 = 264/3  (period 3.3e10)
extending the corpus anchors 11/18/25/34/43. Note the k=4 chains do NOT
carry the record (spans 55 and <= 87): the new maximum comes from k=2/3
merges with fatter flanks - fuel length and record growth are separate
channels at these scales.

F_j spectra (max sum of j consecutive gaps, j=1..6; Constructor's ask):
    machine 23: 34 39 50 58 65 77
    machine 29: 43 55 65 70 85 90
    machine 31: (spectrum-only pass running detached, spectrum31.log)
Increments stay q/3-scale (4-15), consistent with flatness.

### Corrections to shared state
- "k_max <= 3 everywhere" (SUMMARY/Constructor r10) covered steps through
  23->29 only; k_max = 4 at 29->31 and 31->37 (corpus round-1 note "k=4
  first at y=29" was right). Not a falsification of k_max = o(ln y).
- Constructor's k-hist convention (maximal runs) vs my N_k (tuples):
  identical where k_max <= 3; N2 differs from their 2-run count by the
  pairs inside longer runs (38 at 19->23). Flagged to avoid bookkeeping
  confusion.

### Running jobs
- fuel37.log: machine 37 partial (1.2e11 of 1.24e12, 4 probes incl. the
  k=5 hunt) - detached, PID noted in log dir.
- spectrum31.log: machine-31 F_j pass.
- satruns_L15.log: 30.3% at report time, max L = 13 so far; new deep
  L=13 instance at member 3,685,669,022,369 (word LRLRLRRLLLLRR) for the
  records list.

### Caveats
- N_k for partial periods are exact on the scanned prefix only (labeled).
- The o(ln y) comparison uses two data points of k_max growth; no fit
  offered, per methodology - the next event (k=5 or its absence at
  37->41) is the informative object.

## Round 12 - the 37->41 falsification test: k=5 absent, and why that is weak evidence

Jobs survived the credit outage except the spectrum pass (relaunched).
Machine 37 partial census landed: 1.200e11 of the 1.237e12 period (9.7%),
2.11e10 openings, 6197s, 4 probes.

    step      openings   N2          N3     N4   k_max
    37->41    2.11e10    163848288   300    0    3
    37->43    "          158745169   230    0    3
    37->47    "          138732684   41     0    3
    37->53    "          183250785   4091   0    3

VERDICT on the live test: NO k=5 anywhere, and no k=4 either - the
eligible k=5 word (14,27,14,27) at 37->41 does not occur on the scanned
prefix. Constructor's cap SURVIVES this test.

HONEST WEIGHT OF THE EVIDENCE (the part that matters): this is much
weaker than it looks. Per-opening rates at 31->37 were N3 1.14e-5,
N4 3.47e-8. If those rates persisted, 37->41 would have shown ~734 k=4
instances - so naively the absence is decisive. But N3 itself is
SUPPRESSED by a factor 830 at this step (1.42e-8 per opening), and
conditioning on that suppression the expected N4 is 0.91. Observing 0 is
therefore consistent with no cap at all. The test did not probe the cap;
it re-measured arithmetic selection.
    => The k=4/k=5 question is decided by WHICH GAP VALUES are abundant
    at a step, not by a length law. A real cap test needs a step whose
    (s, q'-s) pair is abundance-favoured (like 29->31 and 31->37 were),
    scanned to full period - i.e. the informative steps are chosen by
    arithmetic, not by size.

Chain condition again exact at this scale: pred = 90 = F_k for 41/43/47
probes (F(2,41) = 273/3 = 91 adjacent-frame; k-frame max gap 88, F2 90),
q=53 pred 92. Consistent with the corpus chain.

Spectrum-31 F_j pass relaunched detached (spectrum31.log) after the
outage killed it; machine-23/29 spectra already delivered (r11).
L=15 hunt survived the outage (PID 77120): 54.5%, members to ~6.6e12,
max L = 12 in the latest chunks (record L=14 unbeaten; predicted first
L=15 near 5e12, now inside the scanned range - absence so far is a
sub-1-sigma observation, not yet an event).

## Round 13 - the tier table: fuel is load-bearing at exactly one step

Tools: research/spectrum_pass.py (new, F_j only - no probe loops),
research/fuel_census.py (+ --start offset for period slicing).
Data: research/data/spectra.csv, fuel_census.csv, spectrum31.log,
fuel37_k5hunt.log (running).

### (2) F_j spectra - all machines through 31 at FULL period

    machine   F1  F2  F3  F4  F5  F6      increments
    13        11  16  23  26  28  31      5 7 3 2 3
    17        18  25  28  33  35  40      7 3 5 2 5
    19        25  31  35  38  47  50      6 4 3 9 3
    23        34  39  50  58  65  77      5 11 8 7 12
    29        43  55  65  70  85  90      12 10 5 15 5
    31        58  68  85  90  92  97      10 17 5 2 5
    37        (2e11 prefix pass running, spectrum37.log)

Increments stay q/3-scale (2-17) at every depth - flatness-consistent,
no F-scale jump anywhere.

### (3) THE TIER TABLE (new; the excess question answered mechanically)

Deleting k consecutive openings merges k+1 gaps, so a step's record
F(M+q') is realizable only from chains with F_{k+1} >= F(M+q').
Minimum chain length required, per step:

    step      F(M+q')   F2   F3   F4   min k   fuel present (N2,N3,N4)
    13->17    18        16   23   26   2       (72, 0, 0)
    17->19    25        25   28   33   1       (1088, 0, 0)
    19->23    34        31   35   38   2       (11784, 62, 0)
    23->29    43        39   50   58   2       (243816, 0, 0)
    29->31    58        55   65   70   2       (8022924, 13000, 4)
    31->37    88        68   85   90   3       (1.15e8, 70964, 216)

RESULT: at every step but one the record needs only k <= 2 (single or
double deletion). At 31->37 the record 88 EXCEEDS F3(31) = 85, so no
k <= 2 chain can reach it: it requires k >= 3, and since the k=4 chains
there were measured to reach only <= 87, it is carried by a k=3 chain
exactly. LEMMA 2 IS LOAD-BEARING, at exactly one measured step - and
that step is also the tightest against the tolerance budget (below).
Confirms Lateral's "lemma 2 is not vacuous" by an independent route
(spectrum tiers, not excess shares) and localises it to a single step.

### (3b) Excess-share census (Lateral ask) - with a negative

    step      incr  lem1  exc   exc/incr  adj incr/q'  margin vs 2.5
    13->17    7     5     2     0.29      1.235        50.6%
    17->19    7     7     0     0.00      1.105        55.8%
    19->23    9     6     3     0.33      1.174        53.0%
    23->29    9     5     4     0.44      0.931        62.8%
    29->31    15    12    3     0.20      1.452        41.9%
    31->37    30    10    20    0.67      2.432         2.7%  <-- binding
    37->41    3     2     1     0.33      0.220        91.2%

(F(M+q') from the known chain F(2,y)/3; lem1 = F2 - F; exc = F+ - F2.)
Cross-check: adj incr/q' reproduces the graded constants
1.235/1.105/1.174/0.931/1.452/2.432 -> /3 = 0.412, 0.368, 0.391, 0.310,
0.484, 0.811 EXACTLY, from an independent census; the 0.541 vs 0.270
crossover is my 31->37 row.

NEGATIVE RESULT: excess share is NOT a function of fuel population.
Correlation(exc share, N3 per opening) = -0.03 over seven steps. Zero
long-chain fuel still yields substantial excess (23->29: N3 = 0, share
0.44 - pure k=2 merges), and huge fuel yields small excess (29->31:
N3 = 13000, N4 = 4, share 0.20). Mechanism: N2 is ubiquitous (2-5% of
openings at every step), so k=2 merges are always available and excess
MAGNITUDE is set by flank quality; chain length enters as a THRESHOLD
(the tier table), never as a density.

### BUDGET CAUTION (for Constructor/manager)

The binding step 31->37 sits at adjacent-frame incr/q' = 2.432 against
alpha = 2.5: margin 2.7%. Six of seven steps sit at 42-91% margin; the
one that does not is the same step requiring k >= 3 fuel. If the
SUMMARY 3.1x headroom line refers to the FS_max margins that is
consistent; as a statement about the alpha constant itself the measured
worst case is 2.7%. Any step exceeding 2.5 forces the constant up
(alpha = 3 needs F(2,53) <= 513).

### (1) k=5 test at 37->41 - extended hunt running

Prefix scan (9.7% of period, r12): no k=4, no k=5, shown to be weak
evidence (conditioned on the 830x N3 suppression, expected N4 = 0.91).
Extended slice launched detached (slots 1.2e11..6.0e11, single probe
q=41, fuel37_k5hunt.log) - 4x more coverage on the eligible word
(14,27,14,27). Even absence at that coverage is not decisive; the r12
recommendation stands - cap tests must run at ARITHMETIC-FAVOURED steps
at full period, and I can run any step the Constructor nominates.

### Jobs
- fuel37_k5hunt.log (PID 98168): extended 37->41 k=5 hunt.
- spectrum37.log: machine-37 F_j prefix pass.
- satruns_L15.log (PID 77120): 57.7%, members to ~7e12, max L = 12 in
  recent chunks; L=14 record unbeaten; first L=15 predicted near 5e12
  (inside the scanned range now, absence still sub-1-sigma).

## Round 14 - the padding census: padding IS the gear-37 anomaly

Tool: research/padding_census.py (new). Data: research/data/
padding_census.csv, padding31.log, padding37.log (running).
Definition used (Lateral's, verified equivalent to my window condition):
link letters +1 = spacing s = 2u mod q', -1 = spacing q'-s, 0 = spacing
0 mod q' (both kills at the SAME tooth, one lap apart - a PADDED link,
which requires a gap of M of exactly q'). Legality = non-zero letters
alternate, zeros free == prefix-sum range <= 1, so padded links were
always inside my N_k; this tool breaks them out by z = #zeros.

### (1) Padding SUPPLY per step, full period

    step      F(M)  gaps of M      gaps == q'   share      gaps == 2q'
    13->17    11    1484           0            0          0
    17->19    18    22274          0            0          0
    19->23    25    378674         86           2.27e-4    0
    23->29    34    7952174        6            7.54e-7    0
    29->31    43    214708724      2090         9.73e-6    0
    31->37    58    6226553024     26366        4.23e-6    0

ONSET RULE (exact, structural): supply > 0 requires F(M) >= q' - a gap
of exactly q' must fit at all. This is ZERO by structure at 13->17
(F=11 < 17) and 17->19 (F=18 < 19), not merely rare; padding becomes
possible from 19->23 on. 2q' never fits in range (needs F >= 2q').

SCALING: the share is NOT the smooth e^-(q'/lambda) of the SUMMARY's
model - measured 2.27e-4, 7.54e-7, 9.73e-6, 4.23e-6 is erratic and
non-monotone, off the exponential by 20-1000x. Cause identified in the
gap histograms: the tail of the gap distribution is ARITHMETICALLY
SELECTED, not smooth. Machine 23 has gap 28: 322, gap 29: 6, gap 30:
112 - the value 29 is suppressed ~50x against both neighbours; gap 24
is entirely ABSENT from machines 19 and 23 alike. Padding supply is
therefore the same kind of object as round 11's fuel: selected by which
gap values the machine happens to realize, with no smooth law. (Order
of magnitude, 1e-6..1e-4, is in the exponential's ballpark; the
step-to-step pattern is not.)

### (3) Padding vs the tier table - INDEPENDENT AXES, and the answer

    step      F(M+q')  F2   F3   F4   min k   pad supply   winner
    13->17    18       16   23   26   2       0            literal
    17->19    25       25   28   33   1       0            literal
    19->23    34       31   35   38   2       86           literal
    23->29    43       39   50   58   2       6            literal
    29->31    58       55   65   70   2       2090         literal
    31->37    88       68   85   90   3       26366        PADDED
    37->41    91       90   95   103  2*      running      ?
    (* machine-37 F_j from a 16.2% prefix: 88 90 95 103 112 115, so
     these are LOWER bounds; if the full period lifts F2 to >= 91 the
     min k there drops to 1.)

Padding does NOT change the tier bound. A run of k killed openings
merges k+1 gaps whatever its letters are, so the ceiling F_{k+1} >=
F(M+q') is padding-blind. What padding changes is FEASIBILITY: it makes
runs legal that literal letters would break. The two are independent
axes - tier = how many gaps merge; padding = whether the links connect -
and the 31->37 record needs BOTH: k = 3 (tier) AND one padded link.

### THE RESULT: padding is not decorative, it is the whole anomaly

At 31->37 the census splits the runs by z:

    class            count          max flanked span
    z = 0 (literal)  114,750,740    71
    z = 1 (padded)   26,366         88     <- the true F(M+37)
      of which k=2   26,030         85
      of which k=3   336            88     <- the record run
    z >= 2           0              -

LITERAL-ONLY WOULD GIVE 71, NOT 88. The record is unreachable without a
padded link, and the k=3 z=1 class that achieves 88 has just 336 members
in a 3.34e10-slot period. Consequences:
 * independent confirmation of Lateral's winner anatomy
   [kill]-37-[kill]-12-[kill]: k=3 openings, one padded link, from a
   census that never looked for it;
 * the GEAR-37 ANOMALY (on record since round 8) IS the padding onset.
   Without padding the step's increment would be 71-58 = 13, i.e.
   adjacent-frame 1.054 vs budget 2.5 - a 58% margin, unremarkable.
   With padding it is 30, i.e. 2.432 - the 2.7% margin. The entire
   binding-step problem is one padded link;
 * so the route's tightest constraint is not a length effect but an
   AVAILABILITY effect: whether M carries a gap of exactly q'.

### (2) Double-padded runs: ZERO so far, first appearance predicted

z >= 2 count is 0 at every step censused, including 31->37. This is
expected, not surprising: the number of ordered padded pairs sharing a
run scales like supply^2 / gaps, which is 0.02 (19->23), 0.00 (23->29),
0.02 (29->31), 0.11 (31->37). Nothing should have been seen yet.
PREDICTION (stated before the run lands): at 37->41 the gap count is
~2.2e11 and, at a share in the measured 4e-6..1e-5 band, supply is
~1e6, giving supply^2/gaps ~ 5. THE FIRST DOUBLE-PADDED RUN IS EXPECTED
AT 37->41. Hunt launched: full-period machine-37 padding census
(padding37.log, ~10h). Absence there would be an event in its own right -
it would mean padded links repel, which nothing currently predicts.

### Jobs
- padding37.log: the double-padded hunt at 37->41 (full period).
- fuel37_k5hunt.log: extended k=5 slice, still running.
- satruns_L15.log: 60.9%, members to ~7.3e12, max L = 12 recent;
  L = 14 record unbeaten.
- spectra.csv now holds machines 13..37 (37 at 16.2%, lower bounds).

## Round 15 - the frame question settled; onset rule refined to necessary-not-sufficient

Tools: research/padded_link_anatomy.py (new, the units artifact),
research/padding_census.py (multi-probe). Data: padding_census.csv.

### (2) THE FRAME QUESTION - one worked example, units settled

There is NO contradiction between "twin gaps are divisible by 3" and my
26,366 padded links of cost q'. Three frames are in use and they differ
by fixed factors:

  SLOT frame (k)      slot k IS the pair (6k-1, 6k+1). My censuses count
                      gaps as differences of consecutive OPENINGS in k.
  ADJACENT frame      the corpus chain F(2,y) = 6,15,21,33,54,... lives
                      here; unit = 2 integers. Slot distance d -> 3d.
  INTEGER frame       the members themselves. Slot distance d -> 6d.

So one padded link = q' slots = 3q' adjacent = 6q' integers. The
harvester's "cost 3q'" and my "cost q'" are THE SAME LINK in different
units, and their "all gaps divisible by 3" is automatic in the adjacent
frame because every adjacent-frame gap is 3 x (a slot gap).
Independent cross-check at every machine: F_adjacent = 3 x F_slot -
33 = 3x11 (y=13), 174 = 3x58 (y=31), 264 = 3x88 (y=37), and at y=5,
F(2,5) = 6 = 3x2 with F_slot(5) = 2 verified directly.

WORKED EXAMPLE (real, from machine 31 with q' = 37):

    flank opening before : k = 634153
    killed opening 0     : k = 634158   members (3804947, 3804949)
    killed opening 1     : k = 634195   members (3805169, 3805171)
    flank opening after  : k = 634197
    interior slot-gap    : 37 slots = 111 adjacent = 222 integers
    residues mod 37      : [15, 15]  -> SAME residue: one tooth, one lap
    member check         : 3805169 - 3804947 = 222 = 6 x 37 exactly

Note on "same tooth": the shared residue is 15, NOT +-u' (u' = 31 here).
That is correct and worth stating, because it is easy to misread. A link
is padded iff its two openings share ANY residue mod q'; which residue is
irrelevant, because over the new period q'*P_M every phase offset occurs,
so each site fires exactly once (lateral's firing law). The census counts
CO-DELETABLE sites; the phase decides where they fire, not whether.

### (3) The onset rule, re-tested and REFINED - necessary, NOT sufficient

Round 14 offered F(M) >= q' as the onset rule. Re-testing it against 15
new (M, q') pairs splits it into two halves of different status:

  NECESSITY - a THEOREM, not a measurement: a gap of exactly q' cannot
  exist when F(M) < q'. Confirmed at every pair with F(M) < q' (machine
  19 vs 29/31/37/41; machine 23 vs 37/41/43; machine 29 vs 47): supply 0
  in all, by impossibility.

  SUFFICIENCY - FALSE, and here is the counterexample: machine 29 has
  F = 43 >= 41, yet supply(29, 41) = 0 EXACTLY. The value 41 is simply
  not realized as a gap of machine 29, while 43 is (twice).

Supply table (full periods; supply = #gaps of M equal to exactly q'):

    machine  F     q'=29  q'=31  q'=37  q'=41  q'=43  q'=47
    19       25    0      0      0      0      0      0
    23       34    6      20     0      0      0      0
    29       43    -      2090   84     0      2      0
    31       58    -      -      26366  ?      ?      ?

BOUNDARY CASE, sharp: at q' = F(M) exactly (machine 29, q' = 43) the
supply is 2 - precisely the number of maximal gaps in the period. The
necessity bound is attained, and attained minimally.

MECHANISM (documented directly): the gap spectrum has HOLES near its top.
    machine 29 (F=43): ... 36:38  37:84  38:22  39:12  40:8  41:0  42:0  43:2
      missing values below F: 41, 42
    machine 31 (F=58): ... 51:36 52:10 53:34 54:0 55:34 56:0 57:0 58:4
      missing values below F: 54, 56, 57
Padding availability is therefore governed by the gap-value SPECTRUM
(which values are realized at all), not by F alone - the same arithmetic
selection found for fuel in r11 and for supply scaling in r14, now
localised to a single lookup.

SIMPLIFICATION worth having: supply(M, q') = hist_M[q'] exactly - one gap
histogram per machine answers the onset question for every probe at once,
with no run classification. Only the z >= 2 hunt needs run structure.

### (1) The 37->41 verdict - still running

padding37.log (full period 1.237e12) had not landed at filing time; it is
the decisive test of the r14 pre-registered prediction (supply^2/gaps ~ 5
=> first double-padded run expected at 37->41). Reported next round with
anatomy either way. Note the refined onset rule adds a prior caveat to my
own prediction: it assumed supply(37,41) ~ 1e6 from the share band, but
supply is a histogram lookup, and machine 29 just showed a prime value
can be missing outright. If hist_37[41] = 0 the double-padded prediction
is void for this step rather than refuted - the two failure modes must
not be conflated when the log lands.

### Jobs
- padding37.log: full-period machine-37 padding census (the verdict).
- fuel37_k5hunt.log: extended k=5 slice.
- satruns_L15.log: 62.9%, members to ~7.6e12, L=14 record unbeaten.

## Round 16 - the histogram sweep; my r14 double-padding prediction RETRACTED

Tool: research/hist_probe.py (new - implements the r15 simplification
supply(M,q') = hist_M[q'], so one gap histogram answers every probe with
no run classification; 4x faster than padding_census - machine 31 in 233s
vs 993s). Data: research/data/gap_histograms.csv.
Validation: reproduces the r14/r15 full-period padding censuses exactly
(machine 29: 2090, 84, 0, 2 at q' = 31, 37, 41, 43; machine 31: 26366 at
q' = 37).

### (2) THE HISTOGRAM PROBE SWEEP

    machine  coverage  F      q'=..  hist[q'] = padding supply
    29       100%      43     31     2090      CAN pad
                              37     84        CAN pad
                              41     0         CANNOT (value absent)
                              43     2         CAN pad (= #maximal gaps)
    31       100%      58     37     26366     CAN pad
                              41     134       CAN pad
                              43     860       CAN pad
                              47     226       CAN pad
    37       4.85%     70+    41     2948      CAN pad (definitive)
                              43     7074      CAN pad (definitive)
                              47     2295      CAN pad (definitive)
                              53     515       CAN pad (definitive)

HOLES (values below F absent from the FULL spectrum): machine 29 misses
41 and 42; machine 31 misses 54, 56, 57. Machine 37's prefix has not yet
seen 69, but at 4.85% coverage that is INCONCLUSIVE, not a hole - a
prefix bounds hist from below, so a positive entry is definitive and a
zero is not. All four machine-37 entries above are positive, hence
definitive: PADDING IS AVAILABLE at 37->41, 41->43(via 37), 43, 53.

### (1) The 37->41 verdict, and a retraction

The three-way branch is RESOLVED on the supply side without waiting for
padding37.log: hist_37[41] = 2948 already at 4.85% coverage, so the
"prediction VOID (hist = 0)" case flagged in r15 is ELIMINATED. Padding
exists at this step.

But the same measurement RETRACTS my r14 prediction, before the hunt
landed. Scaling to the full period: supply(37,41) ~ 6.08e4, against
gaps ~ 2.18e11, i.e. share 2.8e-7 - roughly 14x BELOW the 4e-6..1e-5
share band I extrapolated from in r14. The r14 estimate assumed supply
~1e6; the truth is ~6e4. Corrected expectation:

    expected double-padded runs at 37->41 = supply^2/gaps = 0.017
    (r14 predicted ~5)

CONSEQUENCE, stated plainly: the 37->41 hunt is NOT an informative test
of double-padding. Absence there confirms nothing (0.017 expected) and
would not support a corridor law either; my r14 "first double-padded run
expected at 37->41" is withdrawn as an artifact of extrapolating a share
band across a step, exactly the arithmetic-selection error this
workstream has now hit three times (r11 fuel, r14 supply, here). The
lesson is consistent: NEVER extrapolate a per-step share; look it up.

WHERE THE EVENT ACTUALLY LIVES (with its reachability priced): the
threshold is supply >= sqrt(gaps).
    machine 41: gaps 8.9e12, needs share >= 3.4e-7
    machine 43: gaps 3.8e14, needs share >= 5.1e-8
    machine 47: gaps 1.8e16, needs share >= 7.5e-9
Measured shares run ~1e-7..1e-6, so machines 41-43 straddle the
threshold. Their periods (5.1e13, 2.2e15) are far beyond full-scan
reach, so the first double-padded run is plausibly COMPUTATIONALLY OUT
OF RANGE rather than merely unobserved - an honest limit, and a case
where only a structural argument (lateral's corridor law) can decide.

### Jobs left running at the pause
- padding37.log: full-period machine-37 padding census (z>=2 hunt +
  exact supply). Now known to be low-value for the z>=2 question; still
  the exact supply and run classification for the step.
- hist37.log: full-period machine-37 histogram (definitive holes list).
- hist41.log: machine-41 prefix histogram (2e11).
- fuel37_k5hunt.log: extended k=5 slice at 37->41.
- satruns_L15.log: 64.9%, members to ~7.8e12, L=14 record unbeaten.
All are detached, chunk-flushed or single-shot, and safe to leave; note
that hist_probe/padding_census print only at exit (Windows buffering),
so an empty log means running, not failed.

## Round 17 - THE FLANK-ENVELOPE CENSUS: the residual of (D) localised to four addresses

Tools: research/flank_envelope.py (new - per-word joint flank census +
unconditional envelope + F_1..F_8 in one stream, full period where
reachable), research/envelope_analysis.py (new - the verdicts below off
the CSVs). Data: research/data/flank_envelope_words.csv, _joint.csv,
_uncond.csv, _spectra.csv, _gaphist.csv.
VALIDATION before anything else: the tool reproduces Constructor r16
exactly at 29->31 - FS_max = 48 at (gL,gR) = (18,30), F = 43, largest
single flank 35 = 0.81F at span 10 falling to 7 = 0.16F at span 41 - and
reproduces my own r11 fuel census (the length-3 word (10,21,10) has
EXACTLY 4 occurrences, flanks in {4,7}) and the r13 spectra (machines
13..31, F_1..F_6, every value).

### (0) THE IDENTITY THAT DOES THE WORK - flanks are not a free variable

An occurrence of a length-ell word is ell+2 CONSECUTIVE GAPS: the left
flank, the ell letters, the right flank. Therefore

    span(w) + FS(occurrence) <= F_{ell+2}(M)        IDENTICALLY

for every word, compatible or not, at every occurrence. Hence part (D)
- FS_max(w) <= F + q' - span(w) - IS IMPLIED, for all words of length
ell, by the pure spectrum inequality

    F_{ell+2}(M) <= F(M) + q'.

This is the Constructor's r10 "excess <= F_{k_max+1} - F2" read as a
SUFFICIENT condition and resolved per word length. It converts (D) from a
statement about flanks into SPECTRUM FLATNESS AT BOUNDED DEPTH: the depth
needed is ell_max + 2 <= litcap(q') + 1 <= 7, and litcap is machine-free
(a function of q' mod 35: 2, 3, 4 or 6).

### (1) THE PER-STEP LEDGER (exact, full period, F_j recomputed here)

A priori (ell_max = litcap(q') - 1, no fuel input):

    step      litcap  ell_max  F_{ell_max+2}   F+q'    verdict
    11->13      2       1        F_3 = 16       20     IMPLIES (D)
    13->17      2       1        F_3 = 23       28     IMPLIES (D)
    17->19      2       1        F_3 = 28       37     IMPLIES (D)
    19->23      4       3        F_5 = 47       48     IMPLIES (D)  (by 1)
    23->29      3       2        F_4 = 58       63     IMPLIES (D)
    29->31      4       3        F_5 = 85       74     short by 11
    31->37      6       5        F_7 = 97+      95     short
    37->41      2       1        F_3 >= 95     129     see caveat

Resolved PER LENGTH the failures shrink further: at 29->31 the criterion
holds for ell = 1 (F_3 = 65 <= 74) and ell = 2 (F_4 = 70 <= 74) and fails
only for ell = 3; at 31->37 it holds for ell = 1,2,3 (F_3 = 85, F_4 = 90,
F_5 = 92, all <= 95) and fails only for ell = 4,5.

With the MEASURED fuel cap folded in (r11 census, full period: k_max = 4
at both 29->31 and 31->37, N_5 = 0), no word of length >= 4 occurs at
all, so 31->37's residual is EMPTY and the ledger becomes

    step      k_max  F_{k_max+1}   F+q'    verdict
    13->17      2      F_3 = 23      28    IMPLIES (D)
    17->19      2      F_3 = 28      37    IMPLIES (D)
    19->23      3      F_4 = 38      48    IMPLIES (D)
    23->29      2      F_3 = 50      63    IMPLIES (D)
    29->31      4      F_5 = 85      74    OPEN (short by 11)
    31->37      4      F_5 = 92      95    IMPLIES (D)  (by 3)
    37->41      3      F_4 >= 103   129    caveat below

THE RESIDUAL, EXHIBITED. Over every consecutive step this search has
measured, the only (step, length) pair where the spectrum ceiling does not
close (D) is (29->31, ell = 3). There are exactly two compatible words of
that length, and one of them never occurs:

    w = (21,10,21)  span 52   0 occurrences in the full 1.078e9 period
    w = (10,21,10)  span 41   4 occurrences, ALL of them:
        k =   220,171,102   flanks (7,7)   FS = 14
        k =   406,081,827   flanks (4,7)   FS = 11
        k =   672,200,337   flanks (7,4)   FS = 11
        k =   858,111,062   flanks (7,7)   FS = 14
    requirement at alpha = 3: FS <= F + q' - span = 43 + 31 - 41 = 33.
    measured maximum 14. Margin +19 = 0.61 q'.

So the open part of (D), across all measured steps, is FOUR ADDRESSES,
each with 19 to spare. This does NOT prove (D) - future steps can produce
new residuals, and the criterion needs an UPPER bound on F_j at each
machine, which is exactly Wall V. What it does is correct the SHAPE of the
residue: it is not "every step" but a computable, currently four-element
set, and everything else is closed by an identity plus full-period
spectra.

CAVEATS, stated with the claim:
 * F_j(37) and F_j(41) come from PREFIXES, so they are LOWER bounds. The
   criterion needs an upper bound, so the 37->41 and 41->43 rows are NOT
   decided by this data - they are only "not falsified" (F_4(37) would
   have to exceed 129, i.e. 26 above the measured 103, to break).
 * the k_max rows import the fuel bound (lemma 2) as an input; it is a
   full-period MEASUREMENT at 29->31 and 31->37, not a theorem.
 * this covers LITERAL compatible words. The padded tier is bounded by the
   same identity with its own length, and at 31->37 the padded record 88
   sits 7 below F + q' = 95 (the 0.19q' the Constructor reports).

### (2) THE CEILING IS ATTAINED - no better length-only bound exists

The identity bound is not slack. At machine 19, word (10,) (compatible at
q' = 29 and q' = 31), over 9,452 occurrences:

    address k = 137,328   flanks (21, 4)   span + FS = 21 + 10 + 4 = 35
                          = F_3(19) EXACTLY.

So span + FS_max reaches F_{ell+2} on the nose. Any attempt to sharpen
"span + FS <= F_{ell+2}" into something smaller must use the word's
letters, not just its length - the length-only ceiling is tight.

### (3) THE MONOTONE ENVELOPE: TRUE PER STEP, FALSE AS A MACHINE LAW

The Constructor's most promising unproven shape is "the largest single
flank falls monotonically with span". Three verdicts, all exact:

 (a) WITHIN A STEP'S COMPATIBLE WORD LIST: monotone in 19 of 19 measured
     word-steps, ZERO violations (machines 11..29, q' = 13..43).
     Confirmed, and the fall is steep: 0.81F at span 10 -> 0.16F at span
     41 (29->31), 0.80F -> 0.28F (19->23), 0.74F -> 0.21F (23->31).

 (b) AS A PROPERTY OF THE MACHINE (pool a machine's compatible words over
     all probed q'): FALSE. Six violations found; the four that matter,
     with addresses:

     machine 29:  span 21 -> max flank 27   (w = (21,),  q' = 31, 205,068 occ)
                  span 25 -> max flank 30   (w = (25,),  q' = 37,  88,548 occ,
                                             address k = 133,490,560)
        - THE CLEAN COUNTEREXAMPLE: a +3 RISE at a larger span with six-
        figure occurrence counts on both sides. Nothing rare about it.
     machine 29:  span 29 -> max flank 15   (w = (29,),   q' = 43,  2,054 occ)
                  span 31 -> max flank 22   (w = (10,21), q' = 31,  6,500 occ,
                                             address k = 661,321,007)
        - and here the LARGER span has MORE occurrences (6,500 vs 2,054),
        which is exactly why its maximum is bigger. The envelope follows
        the occurrence count, not the span.
     machine 19:  span  8 -> max flank 20   (w = (8,),  q' = 23, 10,462 occ)
                  span 10 -> max flank 21   (w = (10,), q' = 29, 9,452 occ,
                                             address k = 137,328)
     machine 23:  span 27 -> max flank  7   (w = (27,), q' = 41, 170 occ)
                  span 29 -> max flank  8   (w = (29,), q' = 43, 6 occ,
                                             address k = 15,554,598)
     (machine 17 supplies a sixth: span 6 -> 12 vs span 8 -> 14.)

 (c) UNCONDITIONALLY (any letters, every span, from the same stream):
     MASSIVELY false. Violating span pairs per (machine, ell = 1..6):
     machine 13: 7/18/21/40/28/20; machine 17: 10/17/25/45/17/49;
     machine 19: 19/21/23/65/69/119; machine 23: 17/44/109/152/179/257.
     Worst single rises: E(11) = 19 -> E(21) = 34 (machine 23, ell = 4);
     E(6) = 15 -> E(7) = 25 (machine 19, ell = 2).

READING (the honest one): the within-step monotonicity is real but it is
an ORDERING OF RARITY, not a law of position. Consecutive compatible spans
differ by q'-scale amounts and their occurrence counts fall by two to five
orders of magnitude (29->31: 7,815,766 / 205,068 / 6,500 / 4 across spans
10/21/31/41). Monotonicity holds on that sparse set and fails as soon as
the span axis is filled in. Deriving (D) from "monotone envelope" would
therefore have to use the rarity, not the monotonicity.

### (4) IS IT PURE RARITY? NO - a measured suppression sits on top

Rarity null (exact, zero free parameters): draw 2*occ flanks
independently from the machine's OWN gap histogram and take the maximum;
report the median of that maximum and the one-sided p-value P(max < obs).
Effective null = min(rarity null, spectrum ceiling F_{ell+2} - span),
since the null is inadmissible where it exceeds the ceiling.

    step      word       occ      span  FS_max  null  ceil  eff  obs-eff   p
    19->23     (8)      10,462      8     25     33    27    27    -2   0.0000
    19->29    (10)       9,452     10     25     33    25    25    +0   0.0000
    19->23  (8,15)          31     23     11     19    15    15    -4   0.0007
    23->29    (10)     243,370     10     33     45    40    40    -7   0.0000
    23->29    (19)         440     19     18     29    31    29   -11   0.0000
    23->31 (10,21)         138     31     11     26    27    26   -15   0.0000
    23->41    (27)         170     27     12     27    23    23   -11   0.0000
    29->31    (10)   7,815,766     10     48     57    55    55    -7   0.0000
    29->31    (21)     205,068     21     30     49    44    44   -14   0.0000
    29->31 (10,21)       6,500     31     24     40    39    39   -15   0.0000
    29->31 (10,21,10)        4     41     14     15    44    15    -1   0.4732
    29->37    (12)   3,197,558     12     46     55    53    53    -7   0.0000
    29->37 (12,25)         187     37     23     29    33    29    -6   0.0287
    29->43    (29)       2,054     29     24     37    36    36   -12   0.0000

Every well-sampled compatible word sits BELOW the independent null at
p = 0.0000, and below the effective null too, by a deficit that GROWS with
the machine: -1..-5 at machines 11-19, -7..-15 at machines 23 and 29.

THE EXCEPTION IS THE ONE THAT MATTERS. The residual word (10,21,10) at
29->31 - the entire open part of (D) - sits at obs = 14 against a rarity
null of 15, p = 0.4732. Its four occurrences behave EXACTLY like four
independent draws from machine 29's gap distribution. There is no
structural suppression there at all; the margin of +19 is a pure
sample-size effect.

So the observed envelope decomposes into three measured effects: (i) the
spectrum ceiling F_{ell+2} - span, an identity, binding for the common
short words (attained at machine 19); (ii) the rarity order statistic,
which sits below the ceiling and is EXACTLY what the residual word
realises; (iii) a structural suppression of 7-15 gap units on the
well-sampled words, growing with the machine. Only (i) is a theorem.
CONSEQUENCE FOR THE CONSTRUCTOR: a derivation of (D) for the long words
cannot come from the monotone envelope (false as a machine law, (3b)) and
cannot come from the ceiling (too weak there: 44 vs the needed 33). It has
to come from RARITY - an upper bound on the number of occurrences of a
long compatible word, times a tail bound on the gap distribution. That is
the shape the data supports, and it is a counting statement about word
occurrences, not a statement about flank sizes.

### (5) THE MARGIN TRAJECTORY - stable, not closing

Minimum over each step's compatible words of F + q' - span - FS_max:

    step      F    q'   min margin   /q'    binding word (span, FS_max)
    11->13     7   13      +12      0.923      (4)      (4, 4)
    13->17    11   17      +10      0.588      (6)      (6, 12)
    17->19    18   19      +12      0.632      (13)     (13, 12)
    19->23    25   23      +14      0.609      (8,15)   (23, 11)
    23->29    34   29      +20      0.690      (10)     (10, 33)
    29->31    43   31      +16      0.516      (10)     (10, 48)

The absolute margin grows (+10 -> +20); the relative margin sits in a flat
band [0.52, 0.92] q' with no downward trend over six steps. The closest
approach is 29->31 at 0.516 q' - the same step that carries the whole
spectrum residual. (The Constructor's +7 = 0.19 q' minimum is the PADDED
tier at 31->37, a different object from these literal words; both are
recorded, neither is shrinking.)

### (6) BONUS EVENT FROM A LANDED JOB: machine 41's padding supply

hist41.log finished: machine 41, prefix 2.000e11 of period 5.0708e13
(0.394%), F >= 90 on the range,

    hist_41[43] = 66,235     hist_41[47] = 25,032
    hist_41[53] =  5,748     hist_41[59] =     33

all definitive (a prefix bounds hist from below). Machine 41 has
8.499e12 openings in its period, so scaling the MEASURED prefix count
along the period gives supply(41,43) ~ 1.68e7 and

    supply^2 / gaps ~ 33     (r14's double-padding statistic)

against the calibrated zeroes elsewhere: 0.020 at 29->31 (observed 0),
0.112 at 31->37 (observed 0), 0.017 at 37->41. LATERAL'S ROUND-16
PREDICTION - first double-padded run at 41->43, not 37->41 - IS
QUANTITATIVELY SUPPORTED by a measured supply, at the first step where the
statistic exceeds 1 by a wide margin.
DISCIPLINE NOTE ON THE EXTRAPOLATION: this scales a count ALONG one
machine's period (CRT-homogeneous, and the prefix is a genuine measurement
at this step), which is a different and far safer operation than the
share-band extrapolation ACROSS steps that I retracted in r16. It is still
an extrapolation: the direct check needs the full 5.07e13-slot period, out
of reach. Reported as a priced prediction, not an observation.

### (7) PRE-REGISTERED, jobs still running at filing

The three big envelope passes did not land inside the round. Predictions
recorded BEFORE they do, so the outcome is a test and not a fit:

 * 31->37 (full period 3.343e10, ~3h): the compatible words of length
   >= 4 - (12,25,12,25) and (25,12,25,12), span 74, and the two length-5
   words at spans 86/99 - should have ZERO occurrences, because the r11
   full-period fuel census gives k_max = 4 there (N_5 = 0). If any of
   them occurs, k_max = 4 is WRONG and my r11 census has a bug; if none
   does, the 31->37 row of the k_max ledger stands and the spectrum
   closes (D) at that step with margin 3.
 * 37->41 (prefix 3e10, 2.4%): litcap(41) = 2, so only the two
   single-letter words (14,) and (27,) exist. Expect FS_max well under
   F + q' = 129 (the machine-37 gap spectrum has F_3 >= 95, so the
   ceiling alone leaves >= 34); a prefix can only FALSIFY, never confirm.
 * 41->43 (prefix 1.5e10, 0.03%): first envelope data for machine 41;
   words (14,) and (29,). Expect F(41) = 90 confirmed on the range and
   both margins comfortably positive.

### Jobs running at filing
- envelope31.log: 31->37 flank envelope, FULL period 3.343e10 (9% at
  961s, ~3h total).
- envelope37.log: 37->41 flank envelope, prefix 3e10 (7.7%, ~2.8h).
- envelope41.log: 41->43 flank envelope, prefix 1.5e10 (17.5%, ~1.2h).
- padding37.log, hist37.log: machine-37 full-period padding/histogram.
- fuel37_k5hunt.log: extended k=5 slice at 37->41.
- satruns_L15.log: 68.2%, k <= 1.367e12, L=14 record unbeaten.
LANDED this round: hist41.log (see (6)); envelope29/29b/29c (machine 29
at q' = 31, 37, 43, all full period).
Note: flank_envelope prints per-segment progress with flush, so unlike
hist_probe/padding_census its logs show live coverage.

### (8) MID-ROUND: the Constructor's question answered by exhibition

Routed via the coordinator: Constructor derived the same identity
independently (span + FS <= F_{k+1} is free) and found the resulting
SPECTRUM-FLATNESS statement FALSE - at 29->31 the unrestricted 5-window
maximum sits 42 above F while only 31 is allowed, and the true increment
is 15. Their question: HOW does the qualifying restriction suppress the
unrestricted maximum? My census answers it by exhibition, not by argument.

Tool: research/unrestricted_max.py (new). It finds every window of j
consecutive gaps attaining F_j, prints its address, and classifies the j-2
interior gaps that would have to be the word.

    machine 23, F_3 = 50: flanks (23,23) interior (4,)      k = 2,082,580
    machine 23, F_4 = 58: flanks (28,23) interior (4,3)     k = 29,098,935
    machine 23, F_5 = 65: flanks (28,10) interior (5,2,20)  k = 36,845,450
    machine 29, F_3 = 65: flanks (39,23) interior (3,)      k = 407,599,253
    machine 29, F_4 = 70: flanks (31,12) interior (4,23)    k = 717,564,717
    machine 29, F_5 = 85: flanks (30,18) interior (4,3,30)  k = 772,741,833
                          flanks (27,18) interior (3,7,30)  k = 725,859,998

OF THE 132 MAXIMISERS CENSUSED AT MACHINES 19, 23 AND 29, ZERO ARE LITERAL
AND ZERO ARE QUALIFYING. The shape is always the same and it is the exact
opposite of a word: the unrestricted maximiser puts TWO NEAR-MAXIMAL GAPS
ON THE FLANKS AND THE MACHINE'S SMALLEST GAPS IN THE INTERIOR (2, 3, 4, 5,
7). A qualifying interior gap is 0 or +-2c mod q' and positive, hence
>= 2u' = a (Constructor's own fuel_bound Theorem 1 - a THEOREM, not a
measurement). The maximisers violate that floor at every step.

So the suppression mechanism is not subtle and it is not luck: THE
UNRESTRICTED MAXIMUM IS ATTAINED BY A SHAPE THE INTERIOR-GAP FLOOR
FORBIDS OUTRIGHT.

### (9) THE QUALIFYING SPECTRUM Q_j - a word-free criterion that closes
### every measured step, and delivers the fuel cap in the same object

Tool: research/qualifying_spectrum.py (new). Define, exactly as the floor
suggests,

    Q_j(M; a) = max sum of j consecutive gaps whose j-2 MIDDLE gaps are
                all >= a,      a = 2u' = 2*round(q'/6).

Every qualifying word's merged window is such a sum, so span + FS <=
Q_{ell+2} for every qualifying word, and (D) is implied by the purely
spectral, word-free inequality  Q_{ell+2}(M; a) <= F(M) + q'.

Measured at FULL PERIOD:

    step     a    F   F+q'  F_3 F_4 F_5 F_6 F_7   Q_3 Q_4 Q_5 Q_6 Q_7   crit
    11->13   4    7   20     16  18  23  26  28    16  17   0   0   0   +4
    13->17   6   11   28     23  26  28  31  34    18  18   0   0   0   +10
    17->19   6   18   37     28  33  35  40  43    28  28  25   0   0   +9
    19->23   8   25   48     35  38  47  50  58    35  37  38   0   0   +10
    23->29  10   34   63     50  58  65  77  83    50  50  49   0   0   +13
    29->31  10   43   74     65  70  85  90  92    65  68  71  71  71   +3
    29->37  12   43   80     65  70  85  90  92    65  68  68  71   0   +9

(crit = F + q' - max_j<=ell_max+2 Q_j; Q_j = 0 means NO qualifying window
of that depth exists, so (D) is vacuous there - see the correction below.)

THE CRUX, at the step where Constructor showed flatness fails:

    29->31:  F_5 = 85 = F + 42   unrestricted    - FAILS (42 > 31)
             Q_5 = 71 = F + 28   qualifying only - PASSES (28 <= 31)

The interior-gap floor alone - one inequality, no compatibility, no
residues, no corridor - brings 42 down to 28 and clears the alpha = 3
budget with margin 3. DIRECT ANSWER TO THE "ARITHMETIC LUCK" CAVEAT: at
the depth that actually binds, the suppression is carried by the SIZE
THRESHOLD, which is a theorem, and it is already sufficient. Residue
coincidence is not needed at this step. (Margin 3 = 0.10 q' is thin, and
that is the honest caveat in the other direction - see (11).)

BONUS, free: Q_j = 0 exactly when no qualifying word of length j-2 exists,
so the SAME object delivers the fuel cap. Measured: Q_j = 0 for j > 5 at
machine 19 (k_max <= 4 openings), j > 6 at machines 17, 23, 29+q'=37,
j > 7 at machine 29 + q' = 31. The route's part (D) and its fuel bound are
one measurement, not two.

### (10) THE SPAN-RESOLVED ENVELOPE - and my own r17 residual, superseded

The unconditional envelope of section (3c) yields a second, sharper
word-free criterion at no extra cost. With H_ell(s) = max flank sum over
ALL runs of ell gaps with interior span exactly s (any letters),

    span(w) + H_ell(span(w)) <= F + q'   implies (D) for w,

using only the word's LENGTH and SPAN. Evaluated on all 44 measured
(step, compatible word) pairs: IMPLIED AT EVERY ONE, including the r17
residual -

    29->31, w = (10,21,10), span 41:  41 + H_3(41) = 41 + 24 = 65 <= 74.

CORRECTION TO MY OWN SECTION (1), inside the same round: the claim "the
open part of (D) over all measured steps is four addresses" is SUPERSEDED.
It was correct for the criterion I used there (the unrestricted F_{ell+2}),
but that criterion is the wrong one: both the qualifying spectrum (9) and
the span-resolved envelope (10) close that word too, without looking at
its occurrences at all. There is now NO residual at any measured step
under either refined criterion. The four addresses stand as data; the
"residual" label on them does not.

CORRECTION TO MY OWN TOOL, recorded: qualifying_spectrum.py first reported
"not implied" whenever Q_j = 0, treating vacuity as failure. Caught on the
29->37 run (Q_7 = 0 with litcap 6) and fixed; the criterion is now the max
over depths, with Q_j = 0 read as "no such word exists". No published
number was affected.

### (11) k_win vs k_max, AND THE PAR-TRADING TEST (coordinator's ask)

Tool: research/kwin_census.py (new). Per depth k it reports the maximum
FLANKED MERGED SPAN ops[i+k] - ops[i-1] over all window-valid k-tuples
(fuel_census's letter frame: prefix-sum range <= 1), with address and
interior word. Validation: reproduces the known records exactly -
F(19->23) = 34, F(23->29) = 43, F(29->31) = 58 - and r11's tuple counts
(11,784 / 62 at 19->23; 13,000 / 4 at 29->31).

    step      k=1   k=2   k=3   k=4    k_max  k_win  spread
    19->23     31    33    34    -        3      3    8.8%
    23->29     39    43     -    -        2      2    9.3%
    29->31     55    58    55    55       4      2    5.2%

PAR TRADING CONFIRMED at these three steps, with the spreads Constructor
predicted (5-9%): the merged maximum is nearly depth-independent, and at
29->31 the deepest chains (k = 3 and k = 4) tie at 55 while the WINNER is
k = 2 at 58. So k_win = 2, 2, 3 - all <= 3, and a deep chain has never
won. The k = 4 chain's four occurrences merge to 55, three short of the
record: fuel exists and loses.
Addresses of the winners: k = 137,307 (19->23, word (15,8)); k =
14,995,460 (23->29, word (10,)); k = 278,620,515 (29->31, word (10,)).
The 29->31 winner is exactly the envelope census's word (10,) with
span 10 + FS_max 48 = 58 - the two independent censuses agree on the
record and on its address.

Machines 31, 37 and 41 are running (see jobs); the falsifying event to
hunt is a single k_win >= 4.

### (12) SHALLOW FLATNESS F_4 - F vs q' (coordinator's ask 2)

Measured at full period, from my own F_j pass (independent of the
Constructor's):

    machine    11   13   17   19   23   29   31    37(prefix)
    F          7    11   18   25   34   43   58    88
    F_4        18   26   33   38   58   70   90    103
    F_4 - F    11   15   15   13   24   27   32    >= 15
    q'         13   17   19   23   29   31   37    41
    ratio    0.85 0.88 0.79 0.57 0.83 0.87 0.86    -

Shallow flatness holds at all seven machines with F_4 known, ratios
0.57-0.88 - confirming Constructor's six and adding machine 31 (32 <= 37,
ratio 0.86). NOTE THE DIRECTION OF THE CAVEAT: machine 37's F_4 = 103 is a
PREFIX LOWER bound, so its row is not a test; it would have to reach 129
to break, i.e. 26 above the measured value. Machine 41's row is running.
Also worth stating plainly: the ratio is FLAT at 0.79-0.88 for six of
seven machines with no downward trend, so shallow flatness is not gaining
room as the machines grow - it is holding station at ~0.85 q'.

### (13) THE WORD-FREE CRITERION AT THE TWO LARGEST REACHABLE STEPS

qspec31 landed (full period 3.343e10, 725s) and qspec41 (prefix 4e10):

    machine 31 (F = 58, F+q' = 95, a = 12):
      j    3    4    5    6    7    8
      F_j  85   90   92   97  104  110
      Q_j  85   90   91   90   88    0     drops 0, 0, 1, 7, 16
    machine 41 (F = 90 on range, F+q' = 133, a = 14, coverage 0.08%):
      j    3    4    5    6    7    8
      F_j 110  112  118  123  130  138
      Q_j 110  112  110  117  122  121

TWO THINGS WORTH STATING. First, Q_7(31) = 88 = F(31->37) EXACTLY - the
qualifying spectrum at the winning depth equals the true record, so the
bound is ATTAINED at the binding step, not slack. Second, the criterion
margin over all steps:

    step     max_j Q_j   F+q'   margin   /q'
    11->13      16        20      +4     0.31
    13->17      18        28     +10     0.59
    17->19      28        37      +9     0.47
    19->23      38        48     +10     0.43
    23->29      50        63     +13     0.45
    29->31      71        74      +3     0.10
    31->37      91        95      +4     0.11
    41->43     110       133     +23     0.17  (prefix, lower bounds)

HONEST WARNING, and it is the opposite of the story so far: the word-free
margin COLLAPSES at the two largest full-period machines, from ~0.45 q' to
0.10-0.11 q'. The word-restricted margin does not (it sits at 0.52 q' at
29->31). So the qualifying spectrum is a clean sufficient criterion that is
running out of room exactly where the machines get big; whether it survives
37->41 is the live test (qspec37 running on a 2e11 prefix - and a prefix can
only FALSIFY this criterion, since Q_j from a prefix is a lower bound).
Note also the criterion is stated with the max over depths: Q_j = 0 means no
qualifying word of that depth exists, i.e. vacuous, not violated. My tool
first read 0 as failure; caught on the 29->37 run and fixed, no published
number affected. The qspec31 log's own CRITERION line predates the fix
(it prints Q_7 = 88; the correct max over j <= 7 is 91).

Machine 41 also supplies the coordinator's shallow-flatness row:
F_4 - F = 112 - 90 = 22 against q' = 43, ratio 0.51 - holds, though as
prefix lower bounds it is "not falsified" rather than verified.

### (14) NEW MEASUREMENT (human directive): THE HOLE STRUCTURE OF THE GAP
### SPECTRUM

The directive lands on a region I found in r14/r15 and closed with "no
smooth law, look it up". That conclusion stands; this is what is on the
other side of it. Tool: research/hole_structure.py (new), off the
full-period gap histograms my envelope census now writes.

(a) THE HOLE LIST - exact, full period, first time enumerated:

    machine 11  F =  7   holes: none
    machine 13  F = 11   holes: {9}
    machine 17  F = 18   holes: {17}
    machine 19  F = 25   holes: {19, 24}
    machine 23  F = 34   holes: {24}
    machine 29  F = 43   holes: {41, 42}

Holes are RARE (0-2 per machine, against 7-41 realised values) and sit at
the TOP of the spectrum: 0.82F, 0.94F, 0.76F, 0.96F, 0.95F, 0.98F - with
ONE exception, v = 24 at machine 23, which sits at 0.71F.

(b) ABSENCE IS TRANSIENT - the inheritance question, answered:

    13 -> 17:  9 HEALED          17 -> 19: 17 HEALED
    19 -> 23: 19 HEALED, 24 INHERITED     23 -> 29: 24 HEALED

Five of six holes are filled by the very next gear; exactly one (v = 24)
survives a step, and it survives the step where the two machines' F differ
most. NO hole is ever CREATED below the previous machine's F. So the
spectrum fills in monotonically from below as gears are added, and the
holes are a boundary effect that the next gear repairs.

(c) THE RESIDUE LAW - a real new object. hist_M[v] is strongly non-flat in
v mod p, and the SHAPE IS STABLE ACROSS MACHINES AND CONVERGING (entries
are class share x p, so 1.00 = flat):

    machine   mod 2        mod 3              mod 5
    11      0.97 1.03   0.58 0.67 1.75   0.83 0.90 2.22 0.83 0.21
    17      0.91 1.09   0.61 0.83 1.56   1.04 0.85 1.87 0.88 0.36
    23      0.88 1.12   0.64 0.91 1.45   1.13 0.81 1.74 0.92 0.40
    29      0.88 1.12   0.65 0.93 1.42   1.16 0.80 1.70 0.93 0.41
    machine 29, mod 7: 0.78 0.90 1.64 1.15 0.67 1.40 0.45

Every entry moves monotonically with the machine and is settling. The two
richest classes mod 7 are v = 2 and v = 5, which are exactly +-s for gear 7
(s = 2*6^{-1} = 5 mod 7); mod 5 the richest is v = 2 = s. So the letter
values of the SMALL gears are visible in the gap histogram of the WHOLE
machine - a relationship between the corridor gears and the gap census that
nobody had looked at. It is not the naive endpoint-survival count, which
predicts v = 0 mod p richest and +s / -s equal; measured, v = 2 mod 5
(1.70) beats v = 0 (1.16) and v = 3 (0.93). Unexplained.

(d) BUT THE RESIDUE LAW DOES NOT PREDICT THE HOLES. Scoring every value in
the top half of each spectrum by R(v) = prod_p share_p(v mod p),
p = 2,3,5,7:

    machine 13: hole 9 ranks 2 of 7 lowest       - hit
    machine 19: hole 24 ranks 1 of 14            - hit
    machine 23: hole 24 ranks 2 of 18            - hit
    machine 29: holes 41, 42 rank 7 and 10 of 23 - miss
    machine 17: hole 17 ranks 10 of 10 (HIGHEST score) - flat miss

The single INHERITED hole (v = 24) is the one the residue score predicts at
both machines that carry it; the rest are not a residue-marginal
phenomenon at p <= 7.

(e) THE CONSTRUCT THAT WOULD HAVE TO BE BUILT, named as the directive
requires. A gap of exactly v at machine M means v-1 CONSECUTIVE SLOTS ALL
KILLED with both endpoints unkilled. That is not a residue-marginal
question about v, it is a COVERING-FEASIBILITY question about the gear set:
can the gears' 2-tooth classes cover an interval of length v-1 while
sparing its two ends? So the hole set is the complement of the COVERABILITY
SPECTRUM of M, and the object to build is

    COV(M) = { L : an interval of L consecutive slots is coverable by the
               gears 5..M, with both flanking slots spared }

Three reasons this is the right construct and not just another census:
 1. it is computable WITHOUT SCANNING THE PERIOD - it is CRT arithmetic on
    the gear set, so it reaches machines 37, 41, 43, 53 whose periods
    (1.2e12, 5.1e13, 2.2e15) are beyond any scan;
 2. it therefore yields UPPER bounds on F(M) and on the F_j, which is the
    single missing input for my own qualifying-spectrum criterion at those
    steps - every prefix row in this round's tables is "not falsified"
    rather than verified precisely because a scan gives lower bounds only;
 3. the machinery already half-exists in another workstream: harvester's
    pruned F(2,53) search answers "is a run of length L coverable" for one
    L at a time (its log reads "run of 423 is coverable"). What has never
    been built is the SPECTRUM version - all L at once, per machine - which
    is exactly the hole structure, and which would join my gap census to
    their record search and to lateral's corridor in one object.

That is my proposal for the next round, and it is the answer to "what would
have to be built": not a bigger scan, a coverability spectrum.

### (15) JOBS - a sweep killed the detached set; all relaunched

A process sweep during the round killed every long-running job (mine and
the inherited ones). Findings were not lost - the tools are chunk-flushed
or single-shot into CSVs - but coverage was. Relaunched and running:
satruns_L15 (resumed from its state file at k = 1.391e12, 69.5%),
padding37, hist37, fuel37_k5hunt, and my envelope31 / envelope37 /
envelope41. Still running from this round: kwin_census at machines 31, 37,
41 (the k_win >= 4 hunt) and qspec37.
LANDED: qspec29, qspec31, qspec41, unrestricted_max at 19/23/29,
kwin_census at 19/23/29, hist41, envelope29 x3.
