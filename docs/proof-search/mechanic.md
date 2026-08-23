# mechanic workstream log (compacted)

Compacted 2026-08-23; full verbatim rounds 1-19 log at
archive/mechanic-full-r1-19.md. (This file's entries ran rounds 1-17;
round 8 is the attempts ledger.) Below: cumulative state - final tables,
refuted claims kept as refuted, standing rules, open watches.

MANDATE: the workstream's measurement arm. Exact censuses of the slot
machine at scale (windows, runs, fuel, padding, spectra). Product is
exact numbers and named events, never trends; every tool validated
against a known anchor before its numbers are used.

## Definitions (stated once)

- Slot k = pair (6k-1, 6k+1); the two integers are its MEMBERS.
- Window of y: k in [ceil((y-1)/6), floor((y^2+1)/6)] (members in
  [y, y^2]). Gear = prime 5 <= q <= y. Degree = # distinct gear
  divisors; degree-0 = prime > y. Boundary: the member equal to y counts
  prime; a "composite" side equal to gear y is prime (neither twin nor
  fragile).
- twin: both members degree-0. frag_loose: one member degree-0, other
  composite with exactly one distinct gear divisor q (owning gear).
  frag_semi: frag_loose with composite a semiprime or q^2. Anchor y=13:
  9 twins, 10 loose, 9 semi (extra = 125 = 5^3).
- P(t) = prime members in first t window slots; M(t) = t - P(t);
  n0/n1/n2 = slots with 0/1/2 prime members; margin = n2 - n0 = M.
- R_q(t) = composite members in first t slots with lpf = q. Supply
  identity: sum_q R_q(t) = 2t - P(t), exact.
- mu(k) = omega_G(mL) * omega_G(mR); S_pair(t) = sum mu = nontrivial
  cross root-class hits; #{mu>=1} = n2; tau = (t-P)/S_pair.
- Zone functional R(t) = (S1^2/M2)/(t-P), S1/M2 = moments of per-slot
  m = omega_G(mL)*omega_G(mR).
- Saturated run: maximal run of load-1 slots (exactly one prime member
  each); L = length; word = L/R side of the prime per slot.
- Bands B_i = (p_i^2, p_{i+1}^2); thickness T = g(2p+g)/6 slots.
- Machine M = gears 5..M on slot space; step M->q' adds the next gear.
  Openings = surviving slots; gap = difference of consecutive openings
  (SLOT frame). Frames: slot x1, adjacent x3, integer x6 (one padded
  link = q' slots = 3q' adjacent = 6q' integers; F_adjacent = 3 F_slot,
  checked at every machine).
- Fuel N_k = co-deletable k-tuples of openings at M->q' (letters
  {+s, -s, 0 mod q'}, legality = prefix-sum range <= 1; s = 2u' mod q').
  k_max = largest k with N_k > 0. N_k counts TUPLES; Constructor's
  k-hist counts maximal runs (identical only where k_max <= 3).
- Padded link: two openings sharing a residue mod q' (letter 0) - needs
  a gap of M equal to exactly q'; z = # zeros in a run's word.
  supply(M, q') = hist_M[q'], one gap-histogram lookup, exact.
- F(M) = max gap; F_j(M) = max sum of j consecutive gaps. Flanks = the
  two gaps bounding an occurrence; FS = flank sum; span(w) = sum of
  letters. litcap(q') = max literal word length, machine-free function
  of q' mod 35 (2, 3, 4 or 6).
- Qualifying spectrum Q_j(M; a) = max sum of j consecutive gaps whose
  j-2 MIDDLE gaps are all >= a, a = 2u' = 2*round(q'/6). Q_j = 0 means
  no qualifying window of that depth (vacuous, NOT violated).
- k_win = depth of the chain achieving the step's record merged span.

## Census results

### C1. Fragile census (r1) - fragile = 2 * twins * W1 / pi_win
Repro: research/fragile_census.py (503 1009 2003 3001 5003 10007 20011
50021; full prime sweep y = 13..503 plus sparse large y).

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

S1 lone-composite members at 50021: 287,805,085 loose / 271,522,325 semi
(semiprime share 93.6%). Zero-free-parameter law: c = fragile * pi_win /
(twins * W1) = 2, W1 = sum (q-1)/(q-2) over lone-composite members:

    y      13     101    503    1009   2003   5003   10007  20011  50021
    cS   2.200  1.907  1.956  1.973  1.949  1.974  1.985  1.989  1.9914
    cL   2.245  1.907  1.964  1.978  1.950  1.974  1.985  1.989  1.9917

Monotone toward 2 from y=1009; 0.43% error at 50021. fragile/twins grows
like lnln (fit a=3.01/b=-4.48 semi, a=3.22/b=-4.74 loose - FIT, not law;
= Mertens divergence of W1/pi_win). Owning-gear decile shares of loose
fragile (gears ranked, d0 lowest; gear 5 dominates d0):

    y        d0    d1    d2    d3    d4    d5+
    101     58.3  13.2  13.2   4.4   2.9   8.0
    503     69.8  12.8   7.2   3.7   2.7   3.8
    2003    78.1   9.6   4.8   2.9   1.8   2.8
    10007   84.0   7.0   3.5   2.1   1.3   2.1
    50021   87.9   5.3   2.6   1.6   1.0   1.6

### C2. Per-gear closed form (r2)
Repro: research/fragile_pergear.py. Size-corrected law frag(q) =
2*tw*((q-1)/(q-2))*S1w(q)/piw (S1w, piw = 1/ln(m)-weighted sums): exact
to 2e-4 aggregate, Poisson at band level. obs/pred2 (z2) by gear-rank
band 0-50/50-90/90-99/99-100%: y=10007: 1.0002(0.26), 1.0018(0.36),
1.0159(0.54), 1.5427(2.18); y=50021: 1.0002(1.32), 1.0015(1.19),
0.9955(-0.59), 1.0055(0.07). Unweighted, a real 4-5% deficit at
mid/large gears (member-size geometry) - the 1/ln(m) weight removes all
of it; the 10007 top-1% excess was fluctuation (gone at 50021, 186
events). No twin- or necessity-specific structure anywhere, incl. gear
50021 itself (50021^2-2 prime, within its Bernoulli law).

### C3. Prefix censuses at the window bottom (r2)
Repro: research/prefix_census.py; data research/data/prefix_census.csv
(2400 rows, y = 101..1e8, t = 1..200; P counts member y as prime).

    y          1st_dbl  1st_twin>y  minMargin  lastNeg  margin(200)
    101           4         2          -5        99         14
    503           3         4           0         0         29
    1009          4         3          -1        11         40
    10007         2         6          -1         1         73
    100003        2        26           0         0        100
    1000003       2         7           0         0        107
    10000019      2        21           0         0        129
    100000007     2         6           0         0        133

25 windows per decade (150 windows, T=200):

    decade   dbl_mean  dbl_max  tw>y_mean  tw>y_max  minM_min  lastNeg_max
    1e3        3.68       9        6.60       14        -1         11
    1e4        3.04       7       10.84       23        -1          4
    1e5        2.48       4       19.92       36        -1          1
    1e6        2.40       5       13.36       30        -1          2
    1e7        2.36       4       29.72       53        -1          2
    1e8        2.64       6       37.16       73        -1          2

Margin never negative for t in [5,200] at y >= 1e4 (125/125 windows);
min always by t <= 11. First double at slot ~2-4 (no growth with y);
first twin above y at ~ln^2 scale. Identity: margin(t) < 0 forces
n0(t) > 0 - a prefix-pigeonhole refutation of X is a nonconstructive
twin proof, reach localised to t <= 4.

### C4. Full-window margin trajectories (r3)
Repro: research/margin_trajectory.py; data research/data/
margin_summary.csv, margin_checkpoints.csv, margin_bands.csv.

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

(a) M is gear-blind (primality only); band boundaries smooth at 1e-4
slope precision (y = 20011/50021/200003, matched controls). (b) For
y >= 503: minM in {0,-1} at t_min <= 3 (boundary twin), NO later dip,
exhaustive to 6.67e9 slots. (c) Danger zone member-anchored O(1): drift
1 - 6/ln(member) flips positive at member e^6 ~ 403; last<0 <= 11 for
all y >= 503. (d) M(t) = t - [li(6t+m0) - li(m0)] to 0.1% for t > ~1e3;
linear and t/ln t fits fail globally. (e) Envelope: max |M - Mhat| =
0.06-0.18 sqrt(member) at checkpoints, coefficient shrinking (0.058 at
y=200003); M(t) >= Mhat(t) - 0.2 sqrt(6t+y) held everywhere tested.

### C5. Supply trajectories and pair schedule (r4)
Repro: research/supply_trajectory.py; data research/data/supply_load.csv,
supply_pergear.csv. Sieve verified vs independent spf-table count: 0
mismatches in 3384/13892/23313/28764 checks at y=503/2003/10007/50021.
Fresh gears q in (sqrt(y), y): R_q = 0 until t_act = (q^2-1)/6 - k_lo + 1
(own square), then R_q(t) = 1 + pi(m(t)/q) - pi(q) + T_q(t), T_q == 0
while m(t) < q^3 - EXACT (T_q share 0.0000 for q > y^(2/3); gear 5's T_q
share of R_q: 69% at y=503, 76% at 2003). y=50021 excerpt:

    t          member       A     g5%    rho    tau   S_pair/n2
    133        50815        46    27.2  0.636  0.167    5.39
    1333521    8051143      410   25.1  0.747  0.187    5.03
    417008404  2502100441   5132  23.4  0.829  0.222    4.38

Peak tau (always at t = W): 0.314 (503), 0.282 (2003), 0.249 (10007),
0.222 (50021). NO depth where X's demand exceeds the pair schedule; tau
rises monotonically within a window, its max declines with y;
t - P <= n2 <= S_pair identically. The whole reality-vs-X distance is
compression: S_pair/n2 = 4.38 vs X-required 4.50 at 50021 - the n0 term
(2.6% of doubles). The problem lives in P(mu=0).

### C6. Multiplicity distribution vs nulls (r5)
Repro: research/multiplicity_census.py; data research/data/
multiplicity_hist.csv, multiplicity_summary.csv. Null1 = pairs' CRT
classes independent (exact Poisson-binomial via DFT); null2 = product
structure kept, arithmetic broken. Real / null1 / null2:

    y      mean    P0                cond=mean/(1-P0)      var           tail mu>=9
    503    1.508   .466/.220/.465    2.82/1.93/2.99        4.5/1.5/5.1   .025/.000/.020
    2003   2.039   .384/.129/.393    3.31/2.34/3.51        6.9/2.0/7.7   .044/.000/.038
    10007  2.631   .319/.072/.332    3.86/2.83/4.07       10.0/2.6/10.9  .065/.002/.064
    50021  3.185   .273/.041/.287    4.38/3.32/4.59       13.1/3.2/14.2  .088/.005/.091

Null1 misses P0 by ~6.6x; null2 reproduces var/tail/P0 to a few % - the
omega-product structure is the whole carrier. Exact slot-cap covariance:
null2 mean excess = P_primezeta(2) - 1/4 - 1/9 = 0.0911 (4 decimals at
y >= 2003). The X-gap is a ZEROTH-moment statement only: cond_X <=>
P0_X = P0_real - n0/W. Real twin mass sits below null2 by ratio 0.85 ->
0.77 down the ladder (the HL correction); singles mass tracks the model
to 3%. Depth-resolved (r6, research/twinmass_deciles.py, data
twinmass_deciles.csv): HL 1/ln^2 allocation reproduces real decile twin
counts to 1.000 +- 0.003 at 50021 - no depth structure beyond density.

### C7. Inversion zone sup R(y) (r6) - generic forcing dies at y ~ 2-5e6
Repro: research/inversion_zone.py; data research/data/zone_summary.csv,
zone_curves.csv, zone_anatomy.csv. supB64 = bulk sup over t >= 64
(convention-robust; supR at t < ~10 is boundary-sensitive + circular).

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

Zone empty at 5000011 and 10000019; supB64 - 1 ~ y^(-0.6) [fit]. Anatomy:
argmax prefixes = m=0 slots plus a concentrated block m in {4,6,9,12};
CS efficiency 0.919-0.966. Worked (y=2003, t*=24): hist {0:15, 4:7, 6:2},
S1=40, M2=184, CS=8.70 > t-P=3 forcing n0 >= 6 (6 twins present). Zone
revival at any y = a twin in that window's first ~200 slots - an exact
floor-checkable restatement of the conjecture, localised.

### C8. Saturated runs to member ~1e13 (r7-r9; hunt ongoing)
Repro: research/saturated_runs.py, research/satruns_model.py; data
research/data/satruns_ge10.csv, satruns_records.csv, satruns_renewal.csv,
satruns_windows.csv, satruns_deep_ge10.csv (k in [1.2e10, 1.67e11]),
satruns_deep_renewal.csv (+_r8), satruns_L15.log + state file.

Records: L=10 at k=59 (member 353); L=13 at k=2452 (member 14711); L=14
at k = 46,133,660,494 (members 276,801,962,963..276,801,963,043), word
LRRLRLRRRRLLRL, MR-verified maximal. L*=13 stood member 1.5e4 -> 2.8e11
and fell on the constellation curve (predicted first L=14 near 1.6e11,
expected count 1.2 at the actual address - Poisson-consistent, no HL
deficit). The six original L=13 instances (all MR-verified, none
side-alternating):

    k=2452        member 14711        word RLLRRLLLLRLRL
    k=61501443    member 369008657    word LLLRRLLLRRRLL
    k=874166593   member 5244999557   word RLLRRLLRRLRLL
    k=1909351447  member 11456108681  word LLLRRLLLRRRRL
    k=8472005085  member 50832030509  word RRRLLLRRRLLRL
    k=9599932213  member 57599593277  word LLRRLRLLRRRLL

(L>=13 population later 19, incl. deep L=13 at member 3,685,669,022,369,
word LRLRLRRLLLLRR.) Windows inherit landmarks: y=2003/10007 max at
k=2452; y=50021/200003 at k=61501443. Renewal per member-decade:

    decade  slots      L8     L9    L10  L11  L12  L13+
    5       1.5e5      13      6      0    0    0    0
    6       1.5e6      48     15      1    1    1    0
    7       1.5e7     186     43     13    2    0    0
    8       1.5e8     769    146     45    9    2    1
    9       1.5e9    3435    703    122   28    8    1
    10      1.03e10 12655   2445    433   73   10    3

Per-slot L>=8 rate ~ C/(ln m)^beta, beta = 6.81, C = e^8.33 [fit, 8
decades]; per-decade counts keep growing (~19 -> 91k, decades 5..11).
A_8..A_13 = 0.252, 0.220, 0.174, 0.135, 0.084, 0.119; A_L =
exp(-0.197L + 0.197) [fit]. First-arrival ladder [fit, NOT law], CRT cap
[13,32]: L=15 ~5e12, L=16 ~2e14, L=17 ~7e15, L=18 ~3e17, L=20 ~6e20,
L=24 ~5e27, L=28 ~9e34, L=32 ~3e42 (+1 length per factor ~40 in member).
Strict L/R-alternation cap is 6; runs alternate only in the load sense.

### C9. Band census (r10) - thin bands are NOT twin-poor
Repro: research/band_census.py; data research/data/band_census_100003.csv
(9,591 bands, heights to 1e10, every slot exact), band_census_2003.csv.
Exact new law: for a twin (6m-1, 6m+1) the band above has T = 4m and the
twin's product slot k = 6m^2 sits at offset 2m = T/2 - dead center,
composite by the defining twin. Verified 1223/1223 g=2 bands (+60/60 at
calibration). The rest is density: decade-matched g=2 twin density /
all-band density = 0.984, 1.018, 1.006, 1.002 at height decades 6-9
(center-excluded 0.985, 1.019, 1.006, 1.002); center-slot deficit = 1/T.
Twin-EMPTY bands: ZERO through height 1e10 (min 2 twins, only band
(25,49)); worst band in [1e9,1e10) = 342 twins = its Poisson lambda
(g=2 only because g=2 bands are shortest). Min prime members per band =
6. Fragile centers (36m^2+1 prime): 93/1223 = 7.6% at P=1e5 (15.0% at
2003), ~1/ln decline. Verdict: self-reference = exactly 1 deterministic
dead slot per thin band; the binding case binds by LENGTH alone.

### C10. Fuel census (r11-r12) - k_max = 4 exists, arithmetic-selected
Repro: research/fuel_census.py (+ --start); data research/data/
fuel_census.csv. Validation: N3 = 62 at 19->23 with anatomy (8,15)/(15,8)
= corpus census exactly.

    step      period    N2          N3      N4    k_max
    13->17    5.0e3     72          0       0     2
    17->19    8.5e4     1088        0       0     2
    19->23    1.6e6     11784       62      0     3
    23->29    3.7e7     243816      0       0     2
    29->31    1.1e9     8022924     13000   4     4
    31->37    3.3e10    114848070   70964   216   4

Off-step probes (N3): (19,29) 0; (19,31) 4; (19,37) 0; (23,31) 276;
(23,37/41) 0; (29,37) 374; (29,41/43) 0; (31,41) 2; (31,43/47) 0.
k=4 anatomy: 29->31 exactly 4 per period, one word class (10,21,10),
flanks {4,7}; 31->37 has 216, both orientations (12,25,12)/(25,12,25),
flanks in {1,2,3,5,6,10,11,13}. N5 = 0 everywhere scanned. Fuel is
ARITHMETIC-SELECTED (N3 > 0 iff s and q'-s land on abundant gap values),
not smooth in y. 37->41 partial (9.7% of period, 2.11e10 openings):

    step      N2          N3     N4   k_max
    37->41    163848288   300    0    3
    37->43    158745169   230    0    3
    37->47    138732684   41     0    3
    37->53    183250785   4091   0    3

No k=5 or k=4 - but WEAK evidence: N3 is suppressed 830x at this step;
conditioned on that, expected N4 = 0.91, so 0 is consistent with no cap.
Chain condition exact at each new scale: pred F(M+q') = F(2,q')/3 = 58
(29->31), 88 (31->37), 90/91 (37->41 probes), 92 (q=53).

### C11. F_j spectra, tier, excess (r13)
Repro: research/spectrum_pass.py; data research/data/spectra.csv.

    machine   F1  F2  F3  F4  F5  F6
    13        11  16  23  26  28  31
    17        18  25  28  33  35  40
    19        25  31  35  38  47  50
    23        34  39  50  58  65  77
    29        43  55  65  70  85  90
    31        58  68  85  90  92  97

(Machine 37 prefix 16.2%: 88 90 95 103 112 115 - LOWER bounds.)
Increments q/3-scale (2-17) at every depth. Tier rule: the step record
F(M+q') is realizable only from chains with F_{k+1} >= F(M+q'); min k
per step 13->17..31->37 = 2, 1, 2, 2, 2, 3. Lemma 2 (k >= 3) is
load-bearing at exactly one step: 31->37 (record 88 > F_3 = 85; k=4
chains reach <= 87, so a k=3 chain carries it). Excess census
(lem1 = F2 - F, exc = F+ - F2):

    step      incr  lem1  exc   exc/incr  adj incr/q'  margin vs 2.5
    13->17    7     5     2     0.29      1.235        50.6%
    17->19    7     7     0     0.00      1.105        55.8%
    19->23    9     6     3     0.33      1.174        53.0%
    23->29    9     5     4     0.44      0.931        62.8%
    29->31    15    12    3     0.20      1.452        41.9%
    31->37    30    10    20    0.67      2.432         2.7%  <- binding
    37->41    3     2     1     0.33      0.220        91.2%

NEGATIVE: corr(exc share, N3 per opening) = -0.03 - excess magnitude is
set by flank quality (N2 ubiquitous, 2-5% of openings everywhere); chain
length enters only as a threshold. BUDGET: binding step 31->37 sits 2.7%
under alpha = 2.5; alpha = 3 needs F(2,53) <= 513.

### C12. Padding census (r14-r16) - padding IS the gear-37 anomaly
Repro: research/padding_census.py, research/padded_link_anatomy.py,
research/hist_probe.py; data research/data/padding_census.csv,
gap_histograms.csv. hist_probe validated: reproduces full-period padding
censuses exactly (m29: 2090/84/0/2 at q' = 31/37/41/43; m31: 26366).

Total gaps of M per step (full period): 1484, 22274, 378674, 7952174,
214708724, 6226553024 (13->17 .. 31->37). Gap-tail selection example:
machine 23 has gap 28: 322, 29: 6, 30: 112 - value 29 suppressed ~50x
against both neighbours. Supply (# gaps of M equal to exactly q'):

    machine  F     q'=29  q'=31  q'=37  q'=41  q'=43  q'=47  q'=53
    19       25    0      0      0      0      0      0      -
    23       34    6      20     0      0      0      0      -
    29       43    -      2090   84     0      2      0      -
    31       58    -      -      26366  134    860    226    -
    37(4.85%)70+   -      -      -      2948   7074   2295   515

Machine-37 prefix entries are lower bounds but POSITIVE = definitive:
padding available at 37->41 on. Machine 41 (prefix 0.394% of period
5.0708e13, F >= 90 on range): hist_41[43] = 66,235, [47] = 25,032,
[53] = 5,748, [59] = 33. Onset rule: F(M) >= q' NECESSARY (theorem;
confirmed at every F(M) < q' pair) but NOT SUFFICIENT (supply(29,41) = 0
despite F = 43 >= 41 - spectrum hole). Boundary sharp: at q' = F(M)
(m29, 43) supply = 2 = # maximal gaps. 2q' never fits.

31->37 z-split (the anomaly resolved):

    class            count          max flanked span
    z = 0 (literal)  114,750,740    71
    z = 1 (padded)   26,366         88     <- the true F(M+37)
      of which k=2   26,030         85
      of which k=3   336            88     <- the record run
    z >= 2           0

Literal-only would give 71, not 88: the record needs k=3 AND one padded
link (tier and padding are independent axes; the tier bound F_{k+1} is
padding-blind). Without padding the increment would be 13 (adjacent
1.054, 58% margin); with it, 30 (2.432, the 2.7% margin). The whole
binding-step problem is one padded link. Worked link (m31, q'=37):
openings k = 634158 / 634195, residues [15,15] mod 37, member gap 222 =
6x37 - padded iff the openings share ANY residue mod q' (the phase
decides where a site fires, not whether).

Double-padding: z >= 2 count 0 everywhere censused; statistic
supply^2/gaps = 0.020 (29->31), 0.112 (31->37), 0.017 (37->41), ~33 at
41->43 (measured m41 prefix scaled ALONG its period - CRT-homogeneous,
the safe direction). Threshold supply >= sqrt(gaps): needed shares
3.4e-7 (m41), 5.1e-8 (m43), 7.5e-9 (m47) vs measured ~1e-7..1e-6. First
double-padded run expected at 41->43 (Lateral's prediction,
quantitatively supported), but the 5.07e13 period is beyond scan reach -
plausibly decidable only structurally (corridor law).

### C13. Flank envelope + the (D) criterion (r17)
Repro: research/flank_envelope.py, research/envelope_analysis.py,
research/unrestricted_max.py, research/qualifying_spectrum.py,
research/kwin_census.py; data research/data/flank_envelope_words.csv,
_joint.csv, _uncond.csv, _spectra.csv, _gaphist.csv, qspec_table.csv.
Validation: reproduces Constructor r16 at 29->31 exactly (FS_max 48 at
(18,30), F 43), r11 fuel anatomy, r13 spectra.

IDENTITY: span(w) + FS(occurrence) <= F_{ell+2}(M) identically, so (D)
[FS_max(w) <= F + q' - span(w)] is implied by F_{ell+2}(M) <= F(M) + q',
depth needed <= litcap(q') + 1 <= 7. Ceiling TIGHT: machine 19, word
(10,), k = 137,328, flanks (21,4): span + FS = 35 = F_3(19) exactly.

Ledger with measured fuel cap folded in (k_max from C10):

    step      k_max  F_{k_max+1}   F+q'    verdict
    13->17      2      F_3 = 23      28    IMPLIES (D)
    17->19      2      F_3 = 28      37    IMPLIES (D)
    19->23      3      F_4 = 38      48    IMPLIES (D)
    23->29      2      F_3 = 50      63    IMPLIES (D)
    29->31      4      F_5 = 85      74    OPEN (short by 11)
    31->37      4      F_5 = 92      95    IMPLIES (D)
    37->41      3      F_4 >= 103   129    prefix - not decided

The 29->31 residual = the length-3 words: (21,10,21) span 52, 0
occurrences in the full 1.078e9 period; (10,21,10) span 41, exactly 4:

    k = 220,171,102  flanks (7,7)  FS 14      k = 406,081,827  (4,7)  11
    k = 672,200,337  flanks (7,4)  FS 11      k = 858,111,062  (7,7)  14

Requirement at alpha=3: FS <= 33; measured max 14, margin +19 = 0.61q'.
(The "four addresses = the residual" LABEL was superseded in-round: both
refined criteria below close this word too; addresses stand as data.)

Monotone envelope: (a) within a step's compatible word list: monotone
19/19, fall 0.81F -> 0.16F; (b) as a machine law: FALSE - six
violations, cleanest machine 29 span 21 -> flank 27 vs span 25 -> flank
30 (w=(25,), q'=37, 88,548 occ, k = 133,490,560); the envelope follows
OCCURRENCE COUNT, not span; (c) unconditionally: massively false (7-257
violating span pairs per (machine, ell)). Occurrence counts fall 2-5
orders across a step's compatible spans (29->31: 7,815,766 / 205,068 /
6,500 / 4 at spans 10/21/31/41).

Rarity null (draw 2*occ flanks from the machine's own gap histogram,
take max; eff = min(null median, ceiling F_{ell+2} - span)):

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

Every well-sampled compatible word sits below the null at p = 0.0000,
deficit growing with the machine (-1..-5 at m11-19, -7..-15 at m23/29):
a real structural suppression on top of rarity. THE EXCEPTION:
(10,21,10) at 29->31, obs 14 vs null 15, p = 0.4732 - pure rarity, no
suppression. Envelope = (i) spectrum ceiling (theorem, tight) + (ii)
rarity order statistic + (iii) 7-15 gap-unit suppression; a derivation
of (D) for long words must come from occurrence-count x gap-tail bounds,
not from monotonicity (false) or the ceiling (too weak: 44 vs needed 33).

Literal-word margin trajectory (min over compatible words):

    step      F    q'   min margin   /q'    binding word (span, FS_max)
    11->13     7   13      +12      0.923      (4)      (4, 4)
    13->17    11   17      +10      0.588      (6)      (6, 12)
    17->19    18   19      +12      0.632      (13)     (13, 12)
    19->23    25   23      +14      0.609      (8,15)   (23, 11)
    23->29    34   29      +20      0.690      (10)     (10, 33)
    29->31    43   31      +16      0.516      (10)     (10, 48)

Flat band [0.52, 0.92] q', no downward trend. (Constructor's +7 = 0.19q'
is the PADDED tier at 31->37, a different object; neither shrinking.)

Unrestricted maximisers (exhibition): of 132 maximisers of F_j at
machines 19/23/29, ZERO literal and ZERO qualifying - the shape is
always near-maximal FLANKS with the machine's smallest gaps interior,
forbidden outright by the interior-gap floor >= 2u' (Constructor's
Theorem 1, a theorem):

    machine 23, F_3 = 50: flanks (23,23) interior (4,)      k = 2,082,580
    machine 23, F_4 = 58: flanks (28,23) interior (4,3)     k = 29,098,935
    machine 23, F_5 = 65: flanks (28,10) interior (5,2,20)  k = 36,845,450
    machine 29, F_3 = 65: flanks (39,23) interior (3,)      k = 407,599,253
    machine 29, F_4 = 70: flanks (31,12) interior (4,23)    k = 717,564,717
    machine 29, F_5 = 85: flanks (30,18) interior (4,3,30)  k = 772,741,833
                          flanks (27,18) interior (3,7,30)  k = 725,859,998

Qualifying spectrum Q_j (word-free: (D) implied by Q_{ell+2} <= F + q'),
full period:

    step     a    F   F+q'  F_3 F_4 F_5 F_6 F_7   Q_3 Q_4 Q_5 Q_6 Q_7   crit
    11->13   4    7   20     16  18  23  26  28    16  17   0   0   0   +4
    13->17   6   11   28     23  26  28  31  34    18  18   0   0   0   +10
    17->19   6   18   37     28  33  35  40  43    28  28  25   0   0   +9
    19->23   8   25   48     35  38  47  50  58    35  37  38   0   0   +10
    23->29  10   34   63     50  58  65  77  83    50  50  49   0   0   +13
    29->31  10   43   74     65  70  85  90  92    65  68  71  71  71   +3
    29->37  12   43   80     65  70  85  90  92    65  68  68  71   0   +9

Crux at 29->31: F_5 = F + 42 fails (42 > 31) but Q_5 = F + 28 passes -
the size threshold (a theorem) alone converts "arithmetic luck" into
sufficiency at the binding depth; residue coincidence not needed. Q_j = 0
also delivers the fuel cap in the same object (Q_j = 0 for j > 5 at m19,
j > 6 at m17/m23/m29+37, j > 7 at m29+31). Largest reachable steps
(qspec31 full period; qspec41 prefix 0.08%):

    machine 31 (F=58, F+q'=95, a=12):  F_j (j=3..8) 85 90 92 97 104 110
                                       Q_j          85 90 91 90  88   0
    machine 41 (F=90 range, F+q'=133, a=14): F_j 110 112 118 123 130 138
                                             Q_j 110 112 110 117 122 121

Q_7(31) = 88 = F(31->37) exactly - the bound is ATTAINED at the binding
step. Criterion margin per step: +4, +10, +9, +10, +13 (0.31-0.59 q'),
then +3 (29->31, 0.10q'), +4 (31->37, 0.11q'), +23 (41->43, prefix).
WARNING: the word-free margin COLLAPSES to 0.10-0.11 q' at the two
largest full-period machines; the word-restricted margin does not.

Span-resolved envelope (second word-free criterion): span(w) +
H_ell(span(w)) <= F + q', H_ell(s) = max flank sum over ALL runs of ell
gaps with interior span exactly s. Implied at ALL 44 measured (step,
compatible word) pairs, incl. (10,21,10): 41 + H_3(41) = 41 + 24 = 65
<= 74. Under either refined criterion there is NO residual at any
measured step.

k_win census (research/kwin_census.py; validated vs known records and
r11 tuple counts) - max flanked merged span per depth:

    step      k=1   k=2   k=3   k=4    k_max  k_win  spread
    19->23     31    33    34    -        3      3    8.8%
    23->29     39    43     -    -        2      2    9.3%
    29->31     55    58    55    55       4      2    5.2%

Par trading confirmed (5-9% spreads); a deep chain has never won - at
29->31 the k=4 fuel exists and loses (55 vs record 58). Winners:
k = 137,307 (19->23, word (15,8)); k = 14,995,460 (23->29, (10,));
k = 278,620,515 (29->31, (10,) - matches the envelope census's record
address, span 10 + FS 48 = 58).

Shallow flatness F_4 - F vs q': 11, 15, 15, 13, 24, 27, 32 at machines
11..31 = ratios 0.85, 0.88, 0.79, 0.57, 0.83, 0.87, 0.86 - holds at all
seven, flat at ~0.85 q', NOT gaining room. Machine 37 (>= 15) and
machine 41 (22/43 = 0.51) rows are prefix lower bounds ("not falsified",
not verified).

### C14. Hole structure of the gap spectrum (r17)
Repro: research/hole_structure.py off full-period gap histograms.
Hole lists (exact, full period; values below F absent from the spectrum):

    machine 11  F =  7   holes: none
    machine 13  F = 11   holes: {9}
    machine 17  F = 18   holes: {17}
    machine 19  F = 25   holes: {19, 24}
    machine 23  F = 34   holes: {24}
    machine 29  F = 43   holes: {41, 42}
    machine 31  F = 58   holes: {54, 56, 57}

Holes rare (0-3 vs 7-41 realised values), near the top (0.71F-0.98F;
the 0.71F is v=24 at m23). Healing: 9, 17, 19, 24 all healed by the
next gear; only v = 24 survives one step (19 -> 23); no hole is ever
created below the previous machine's F - the spectrum fills monotonically
from below. Residue law (class share x p; 1.00 = flat), stable and
converging across machines:

    machine   mod 2        mod 3              mod 5
    11      0.97 1.03   0.58 0.67 1.75   0.83 0.90 2.22 0.83 0.21
    17      0.91 1.09   0.61 0.83 1.56   1.04 0.85 1.87 0.88 0.36
    23      0.88 1.12   0.64 0.91 1.45   1.13 0.81 1.74 0.92 0.40
    29      0.88 1.12   0.65 0.93 1.42   1.16 0.80 1.70 0.93 0.41
    machine 29, mod 7: 0.78 0.90 1.64 1.15 0.67 1.40 0.45

Richest classes are +-s of the small gears (v = 2, 5 mod 7 = +-s for
gear 7; v = 2 = s mod 5) - the small gears' letters are visible in the
whole machine's gap histogram; NOT the naive endpoint-survival
prediction (v = 0 richest, +s/-s equal; measured v=2 mod 5 at 1.70
beats v=0 at 1.16). UNEXPLAINED. The residue score R(v) = prod_p
share_p(v mod p), p <= 7, does NOT predict the holes: hits at m13
(rank 2/7), m19 (1/14), m23 (2/18); misses at m29 (ranks 7, 10 of 23)
and m17 (rank 10/10, HIGHEST score). Proposal on file: build COV(M) =
{ L : an interval of L consecutive slots coverable by gears 5..M with
both flanks spared } - CRT arithmetic, no period scan, reaches machines
37..53, and yields the UPPER bounds on F(M)/F_j that every prefix row
lacks; joins the gap census, harvester's record search, and lateral's
corridor in one object.

## Refuted / retracted (kept as refuted)

1. r1 candidate laws: fragile prop. to twins, to W/ln^3(y^2), to
   pi(y^2)/ln(y^2) - all FAIL (ratio/normalised columns grow). Killed by
   the census table.
2. r5: independent-pairs null as a compression model - misses P0 by
   ~6.6x, var by 4.1x, tail by 16x. The product structure is the carrier.
3. r7 "L = 13 bounded forever": not supported, and the record fell at
   member 2.8e11 exactly on the constellation curve (r9). Records are
   on curves, never walls.
4. r7 "the L=13 landmark word is strictly alternating": FALSE - all six
   words are blocky; alternation holds only in the load sense (strict
   cap 6).
5. r10 "thinnest bands are twin-hostile" (T1 reopening premise): refuted
   - twin density in twin-endpoint bands equals generic at 0.2-2%
   precision; the only deterministic obstruction is the one center slot.
6. Shared-state "k_max <= 3 everywhere" (SUMMARY/Constructor r10):
   corrected r11 - k_max = 4 at 29->31 and 31->37.
7. r12: the 37->41 k=5 absence as a cap test - downgraded to near-zero
   information: N3 suppressed 830x there, conditional expected N4 =
   0.91. Cap tests must run at arithmetic-favoured steps, full period.
8. r14 exponential padding-share model e^-(q'/lambda): off 20-1000x,
   non-monotone - the gap tail is arithmetically selected.
9. r14 onset rule "supply > 0 iff F(M) >= q'": sufficiency FALSE
   (supply(29,41) = 0 despite F = 43 >= 41). Necessity is a theorem.
10. r14 prediction "first double-padded run at 37->41" (statistic ~5):
    RETRACTED r16 before the hunt landed. Killed by the histogram
    lookup: supply(37,41) ~ 6.08e4, not ~1e6; corrected expectation
    0.017. Third instance of the share-extrapolation error.
11. r17 "monotone flank envelope" as a machine law: FALSE, six
    violations with addresses (cleanest: m29 span 25 -> flank 30 beats
    span 21 -> 27, six-figure occurrence counts both sides).
12. r17 "the open part of (D) is four addresses": superseded within the
    round - the qualifying spectrum and span-resolved envelope close
    (10,21,10) too; no residual remains. The addresses stand as data.
13. Unrestricted spectrum flatness F_{ell+2} <= F + q': FALSE at 29->31
    (F_5 sits 42 above F, only 31 allowed). Replaced by Q_j, which
    passes.
14. Tool-bug ledger (caught by validation): cofactors c < q belong to
    gear lpf(c) (r4); int64 overflow in M2*(t-P) at W ~ 1.7e9, one
    garbage row regenerated (r6); qualifying_spectrum.py read Q_j = 0 as
    failure, fixed (r17) - qspec31 log's CRITERION line predates the fix
    (prints Q_7 = 88; correct max over j <= 7 is 91).

## Open questions / watch-items

- k_win >= 4 WATCH: a single k_win >= 4 anywhere falsifies "deep chains
  never win". kwin_census running at machines 31, 37, 41.
- Q_j MARGIN COLLAPSE: word-free margin fell 0.45q' -> 0.10-0.11 q' at
  the two largest full-period machines. Live test = qspec37 (2e11
  prefix; can only falsify, Q_j from a prefix is a lower bound).
- 31->37 BUDGET: adjacent incr/q' = 2.432 vs alpha = 2.5, margin 2.7%;
  any step over 2.5 forces alpha = 3, which needs F(2,53) <= 513.
- (D) at 37->41 / 41->43: undecided - F_j(37), F_j(41) are prefix LOWER
  bounds; F_4(37) must exceed 129 (26 above measured 103) to break. The
  missing upper bounds are exactly what COV(M) would supply.
- FIRST DOUBLE-PADDED RUN: expected at 41->43 (statistic ~33); period
  5.07e13 beyond scan reach - likely structural-only (corridor law).
- L=15 HUNT: predicted first arrival ~5e12; scan resumed at k =
  1.391e12 (69.5%), L=14 unbeaten; absence so far sub-1-sigma.
- k=5 FUEL: eligible word (14,27,14,27) at 37->41; extended slice
  running (fuel37_k5hunt); even absence at that coverage not decisive.
- PRE-REGISTERED (r17 envelope jobs): 31->37 full period must show ZERO
  occurrences of length >= 4 compatible words (spans 74/86/99), else
  r11's k_max = 4 is buggy; 37->41 / 41->43 prefix envelopes expect
  comfortable positive margins (falsify-only).
- MACHINE-37 FULL PERIOD: padding37 (exact supply + z>=2), hist37
  (definitive hole list; is 69 a hole? inconclusive at 4.85%).
- RESIDUE LAW of the gap histogram (C14): converging, small-gear letters
  visible, direction contradicts naive survival - UNEXPLAINED.
- COV(M) coverability spectrum: proposed build (CRT arithmetic, no
  scan; upper bounds on F/F_j at machines 37..53).
- ZONE REVIVAL: "sup R > 1 revives i.o." is an exact restatement of the
  conjecture localised to ~200 slots per window - an address to attack.
- r2 "margin never negative for t >= 5" is a measured regularity on
  150+12 windows, not a law; dips near y ~ 1e3-1e4 remain plausible.

## Standing rules (earned)

1. NEVER extrapolate a per-step share - look it up. Hit three times
   (r11 fuel, r14 supply, r16 retraction). supply(M,q') = hist_M[q'] is
   one lookup. Scaling ALONG one machine's period (CRT-homogeneous) is
   safer than scaling ACROSS steps - still label it extrapolation.
2. Events, not trends (r8 ledger criterion). Label every number: exact /
   measured law (zero free parameters) / fit / record / extrapolation;
   fits get residuals; records sit on curves, not walls.
3. Prefix scans give LOWER bounds on hist/F_j/Q_j: positive entries
   definitive, zeros inconclusive; a prefix can only FALSIFY a criterion
   needing upper bounds. Never let "not falsified" read as verified.
4. Q_j = 0 and any vacuous case mean "no such object exists", not
   "criterion violated".
5. Frames: slot x1, adjacent x3, integer x6 - state the frame with every
   gap number. Boundary: member y counts prime. N_k counts tuples vs
   Constructor's maximal runs (identical only for k_max <= 3).
6. Validate every new tool against a known census before using its
   numbers (caught the r4 cofactor bug, r6 overflow, r17 vacuity bug).
7. Cap/falsification tests run at ARITHMETIC-FAVOURED steps at full
   period; informative steps are chosen by arithmetic, not size.
8. Distinguish "prediction void" (supply absent) from "prediction
   refuted" (supply present, event absent); state the branch, and
   pre-register predictions, before jobs land.
9. Scope per round: this file, own agents-shared append, research/
   files only.
10. Long jobs: detached, chunk-flushed, resumable (state files).
    hist_probe/padding_census print only at exit on Windows - an empty
    log means running, not failed.

## Round 20 (2026-08-23/24)

### R20.A Landed round-19 tail jobs (finished after r19 filed; all folded here)

1. F(2,53) = 435 EXACT (maxgap53_pruned.log). alpha=3 needed <= 513:
   PASSES with 15% room. Slot frame: F(53) <= 145 = 435/3 (frame identity
   F_adjacent = 3 F_slot, checked at every machine).
2. MACHINE 37 FULL PERIOD, two independent scans (hist37 11,829 s;
   padding37 19,694 s; period 1.2368e12):
   - F(37) = 88 EXACT. Definitive hole list: 13 holes below F =
     {73,74,75,76,78,79,80,81,82,83,84,86,87}; 69 is NOT a hole.
   - supply(37,41) = 61,460 gaps == 41 (2.820e-7); gaps == 82: 0;
     z >= 2 runs: 0. Runs: z=0: 1,688,650,276 (max flanked span 90);
     z=1: 61,460 - k=2: 58,416 (max 83), k=3: 3,044 (max 91).
   - STEP RECORD 37->41 = 91 at k=3 z=1: PADDING CARRIES THE RECORD
     AGAIN (as at 31->37); k_win(37->41) = 3 on the full period (the
     8.09% kwin37 prefix said k_win=1, record 90 - the prefix missed
     the record class; prefix k_win is a lower-quality object).
3. kwin31 FULL PERIOD (31->37, 3,264 s): k_win = 3, record 88 = F(37) -
   merge law exact end-to-end against hist37's direct scan. k=4 tuples:
   216 (max merged 68: fuel exists, loses by 20). KW=8 search: ZERO
   k >= 5 tuples -> r17 PRE-REGISTERED TEST CONFIRMED (no length >= 4
   compatible word occurs at 31->37; r11's k_max = 4 stands). Par spread
   22.7% (largest yet). kwin41 (0.08%): record 97 at k=1 (prefix).
4. fuel37_k5hunt (48.5% of period): N1..N4 = 83,267,937,292 /
   655,392,949 / 1,173 / 0. Superseded by R20.C (SAT decides the caps).
5. qspec37 (16.2%): Q_3 = 95 vs F+q' = 129; litcap(41) = 2 so only Q_3
   binds. NOT falsified; margin 0.83q' vs the 0.10-0.11q' collapse at
   29/31. THE COLLAPSE IS A litcap-6 PHENOMENON: qtab31 shows margin
   0.108 at q'=37 (litcap 6) vs 0.341..0.738 at litcap-2 primes; 41 and
   43 are litcap-2. Watch narrowed, not closed: next litcap-6 prime
   after 37 is 53. qspec41 (0.08%): Q_j <= 122 vs 133 everywhere.
6. L=15 HUNT COMPLETE to k = 2e12 (member 1.2e13): NO L=15. Deep-range
   census: L=13: 48, L=14: 5, L=15: 0. All five L=14 addresses:
   k = 46,133,660,494 / 410,898,686,641 / 706,483,435,891 /
   1,663,183,851,213 / 1,984,490,922,377 (word LRLLRRRLRRLLLL).
   Ratio-based expectation for L=15 in range: 5 x (5/48) = 0.52 ->
   P(absence) = 0.59: sub-1-sigma, record on curve.
7. envelope31/37 DIED mid-scan (10-13% coverage, no error - the same
   external process sweep as r19); envelope41 "PROVED" line is INVALID
   (its tiny prefix saw F = 73 vs true F(41) = 91) - word/flank rows
   stand as prefix data only. The one pre-registered claim they carried
   (31->37 length >= 4 words) was settled by kwin31 (item 3).

### R20.B Gap-pair census / p_j (deliverable a; Constructor's object)

Full period at 13/17/19/23 (this round) + 29/31 (r19 tail, full) +
machine 37 at 12.9% (1.6e11 slots, 28.19e9 gaps). Lags 1-5, run-min
m = 2..6, all floors; CSVs gap_pair_{hist,joint}.csv (deduped - r19 had
already written 13..23; tonight's identical duplicates removed),
summary research/data/pj_deficits.csv.

p_m/p_1^m at each machine's own floor a = 2u'(next prime):

    machine  a    p1       m=2     m=3     m=4     m=5     m=6
      13     4  0.3733   0.890   0.752   0.556   0.373   0
      17     6  0.1852   1.123   0.721   0.305   0       0
      19     6  0.2410   1.090   0.806   0.297   0.020   0
      23     8  0.1371   0.939   0.319   0.029   0.016   0
      29    10  0.1188   0.801   0.162   0.049   0.014   0
      31    12  0.0766   0.581   0.149   0.053   0.005   0
      37    14  0.0530   0.469   0.155   0.056   0.014   0.0009

Deficit x6.5-x6.7 at m=3, x18-x20 at m=4, x70-x190 at m=5 - stable
across 23..37. Lag-resolved: deficit at lags 1-3, EXCESS at lags 4-7
(see R20.D - one phenomenon, the corridor).

### R20.C COV(M) BY CRT+SAT (deliverable b) - exact spectra, no scan

research/cov_sat.py (+ cov_slot.py, cov_spectrum.py, cov_gap.py earlier
forms). Gear q blocks {a_q, a_q + s_q} mod q, s_q = -2u_q, phase free;
CRT realises every phase vector; questions become ~300-var CNF. Every
SAT witness is CRT'd to an explicit k and machine-verified by assert.
Engineering: pysat's C CardEnc and Minisat22 segfault over many
instantiations - pure-Python sequential counter + Cadical153 is stable.

VALIDATION (all EXACT matches):
- gap spectra: all 8 full-period machines 11..37 - F and complete hole
  lists (m37's 13 holes: 11,829 s scan -> 123 s SAT).
- F_j: m23 j=2..6 = 39/50/58/65/77; m29 j=2..6 = 55/65/70/85/90;
  m31 j=2..5 = 68/85/90/92. Witnesses reproduce r17 census addresses
  (k = 2,082,580; 29,098,935; 407,599,253; 725,859,998; 4,665,550,942).
- fuel: 31->37 k=4 SAT on exactly the two r11 words (12,25,12) /
  (25,12,25); k=5 all-UNSAT (= kwin31).
- pair: (34,34) refuted at m23, (34,5) realized at the known F_2
  address - Constructor's adjacency NO reproduced.

NEW EXACT RESULTS (beyond any scan):
- MACHINE 41 (period 5.07e13) COMPLETE: F(41) = 91; holes below F =
  {84, 87, 89}; tail 92..100 all refuted. Two independent methods agree
  (COV vs merge-law record from padding37). Hole ladder 11..41:
  0,1,1,2,1,2,3,13,3. Healing law holds (89 >= F(37); 84, 87 survive
  one step).
- F_2(37) = 90 EXACT (witness gaps [2,88]): lemma-1 margin at 37 is
  F_2 - F = 2 <= 41, i.e. 0.95q' of room - the largest measured.
- ADJACENCY of two maximal gaps refuted at m31 (58,58), m37 (88,88),
  m41 (91,91) - previous reach was y <= 23.
- FUEL CAPS AT FULL PERIOD BY SAT (fuel_sat.py): N_4(37->41) = 0
  (53 legal words all refuted) -> k_max(37->41) = 3 EXACT.
  N_4(41->43) = 0 (120 words) -> k_max(41->43) = 3.
- THE FIRST DOUBLE-PADDED RUN EXISTS: word (43,43) at 41->43, witness
  k = 116,431,845,582 (openings k, k+43, k+86 sharing one residue mod
  43, z = 2). The r16 prediction (statistic ~33, "plausibly decidable
  only structurally") is CONFIRMED, structurally, with an address.
  Also gap 86 = 2q' occurs at m41 ((86,) realized): first 2q' padding
  anywhere. Realizable 41->43 padded words: (14,43),(29,43),(43,14),
  (43,29),(43,43).
- STANDING BOUNDS (honest, resumable): F_3(37) in [97, 163] (S=164..178
  all refuted, ~10-20 min per refutation; the (D)-decision at 37->41
  needs <= 129 - 34 refutations away, next-round job, tool ready).
  Q_3(37) refutations same order (605 s probe) - no shortcut found.
  F(43) >= 103 (witnessed), 104 REFUTED, tail [105,118] undecided
  (refutations 1.5-2 h each at 12 gears); holes below 103 possible but
  none observed. F(47) >= 118 (witnessed). F(53): witness hunt toward
  145 stopped deliberately at close: F(53) >= 136 (witnessed; no
  refuted v observed anywhere below), <= 145 (pinned by F(2,53) = 435);
  v in [137,145] undecided - v=137's refutation ran 80+ min before the
  stop (suggestive of a hole; NOT claimed).

### R20.D THE CORRIDOR RESONANCE (new law, measured; docs/novel/)

bool_lag_census.py (8/16-lag boolean pattern census, full period
17..31), corridor_resonance.py, transfer_spectrum.py, dft_events.py.

1. THE WAVE: the qualifying-gap indicator's autocorrelation is a barely
   damped oscillation. m29 floor 10, lags 1..15:
   0.801 0.684 0.510 0.800 1.112 1.257 1.204 0.995 0.781 0.717 0.848
   1.082 1.254 1.250 1.094 (trough 3, peak 6, trough 10, peak 13-14;
   second cycle undamped). m31 floor 12: trough 2, peak 5-6, trough
   9-10, peak 12-13. Period ~ 35/mean_gap (8.2/7.5/7.0/6.5 predicted at
   19/23/29/31; measured peaks 8/7-8/6-7/~7).
2. THE MECHANISM: slot-separation autocorrelation of big-gap left
   endpoints peaks EXACTLY at 35/70/105: m23: 3.22/3.45/2.63;
   m29: 3.64/4.37/2.94; m31: 3.41/4.20/2.97 (neighbours 0.17-1.3;
   sep 70 > sep 35 everywhere). Endpoints are PINNED mod 35: invariant
   core {10,12,18} enriched >= 1.2x at all five machines (17..31);
   companions drift (17 small machines, 7 at 23/29, 5 at 31); exact
   four-way tie 10/12/17/18 at m17 (2.42) and m19 (2.13), tie-pairs
   (10,18),(12,17) at m23. Poorest {28,30,32,33} at 0.12-0.46.
3. NOT MARKOV: the process is not k-step Markov for k <= 4 (TV of exact
   factorisations, m29 floor 10 W16: 0.151/0.134/0.092/0.080; m31
   floor 12: 0.088/0.060/0.043/0.041). The value-level one-step chain
   predicts NO deficit at lags 2-5 (0.99-1.06) where the census says
   0.51-0.68: THE ANTI-CORRELATION LIVES AT RANGE 2-3 AND IS INVISIBLE
   TO LAST-GAP STATE. Constructor's transfer matrix needs corridor
   phase (mod 35 at least) in its state. Pattern counts are exactly
   mirror-symmetric (machine reversal) - checked at every machine.
4. lam2 -> phi/3 (matrix frame meets complex frame): subleading
   eigenvalue of the measured lag-1 transfer matrix, real negative:
   |lam2| = 0.6273/0.5959/0.5722/0.5583/0.5515/0.5462/0.5425 at
   m13..m37. Distance to phi/3 = 0.53934 (Lateral's golden gap):
   +0.0880/+0.0566/+0.0329/+0.0190/+0.0122/+0.0069/+0.0032 -
   geometric, factor ~0.6/machine. kappa(2) = 0.5448 is PASSED and
   DEAD as a limit. Convergence to phi/3 CONJECTURED (7 exact points,
   no fit claimed).
5. DFT identity events: c_q(g) = inverse transform of the exposed-set
   power spectrum |hat1_A(t)|^2 = 4cos^2(2 pi u t/q) - ZERO mismatches,
   all gears 5..53, all lags (376 checks); corridor mod 35: census =
   c5*c7 = product-spectrum inverse DFT, 35/35. Gap-histogram ripple:
   |H_5(1)|/H0 falls 0.31 -> 0.18 (m13..m37) while arg H_5(1) = +126
   deg +- 2 at ALL SEVEN machines (m7: +121 -> +133 slow drift) - the
   C14 residue law has a machine-independent PHASE that is NOT the
   naive +-s ripple (which would give 0/180). UNEXPLAINED; handed to
   Lateral.

### R20.E Watch updates

- k_win >= 4 WATCH: intact. k_win = 3 (31->37 full), 3 (37->41 full,
  padded record), prefix-only at 41->43. No k_win >= 4 anywhere.
- Q_j MARGIN: collapse does NOT continue at 37->41/41->43 (litcap-2);
  the binding regime is litcap-6 primes - next test q' = 53.
- FIRST DOUBLE-PADDED RUN: FOUND (R20.C) - watch closed.
- MACHINE-37 FULL PERIOD: landed - hole list definitive, 69 settled
  (not a hole) - watch closed.
- L=15: absence sub-1-sigma at 1.2e13; hunt idle (no scan running).
- NEW WATCH: F_3(37) <= 129 decision (34 refutations, resumable at
  S=163); machine-43 tail [105,118]; lam2 at m41+ (needs joint census
  beyond 37 - only COV-type pair tools can reach).

### R20.F Tooling and incidents

- New tools: cov_sat.py (gap/window/qualifying/pair/fuel SAT, witness
  round-trip), fuel_sat.py, gap_pair_census.py (--start slices),
  bool_lag_census.py (+analyze_bool_lag.py, W-parametric),
  corridor_resonance.py, transfer_spectrum.py, dft_events.py,
  pj_deficits.csv builder (inline).
- PROCESS SWEEP (r19's phenomenon, recurring): 15+ background jobs
  killed in waves every ~30-120 min through the night. Defense that
  worked: per-instance logging + vmin/smax resume args; after adding
  them nothing was lost. Two "killed" numpy jobs (gappair37 160e9,
  bool16 m31 full period) had actually completed their final writes -
  check the CSV before rerunning a "killed" job.
- CSV hygiene: gap_pair CSVs deduped (r19+r20 identical blocks);
  machine-37 slice block removed as overlapping the complete 12.9%
  block; bool_lag16_31.csv kept full-period block only.
