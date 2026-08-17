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
1. INDEPENDENT-PAIRS NULL FAILS BY A FACTOR ~6.6 ON P0 (0.041 vs 0.273 at
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

### (1) The sup R(y) curve - IT CROSSES 1. The zone dies at y ~ 2-5 x 10^6

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
