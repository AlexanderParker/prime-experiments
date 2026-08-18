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
