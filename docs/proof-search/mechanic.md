# mechanic workstream log (compacted, rounds 1-24)

Compacted 2026-08-29 into ONE cumulative summary over rounds 1-24. Full
verbatim: `archive/mechanic-full-r1-19.md`, `archive/mechanic-full-r20-24.md`
(the r1-19 compact core + the verbose round 20-24 appends this file
replaces). Round narrative and cross-lane traffic: the round appends of
`agents-shared.md`.
Cumulative STATE only - final tables, constructs with validation status,
refuted claims kept as refuted, open watches, standing rules. Chronology
dropped. Where a later round corrected an earlier one the corrected value
stands in the census sections and the correction is recorded once under
"Retracted / corrected". Round 22 has no section: its central object (the
marked spectrum Q^[J]) was a tool bug, retracted in r23 - see R15.

MANDATE: the workstream's measurement arm. Exact censuses of the slot
machine at scale (windows, runs, fuel, padding, spectra, dictionaries).
Product is exact numbers and named events, never trends; every tool
validated against a known anchor before its numbers are used.

## Definitions (stated once)

- Slot k = pair (6k-1, 6k+1); the two integers are its MEMBERS.
- Window of y: k in [ceil((y-1)/6), floor((y^2+1)/6)] (members in
  [y, y^2]). Gear = prime 5 <= q <= y. Degree = # distinct gear divisors;
  degree-0 = prime > y. Boundary: the member equal to y counts prime.
- twin: both members degree-0. frag_loose: one member degree-0, other
  composite with exactly one distinct gear divisor q (owning gear).
  frag_semi: frag_loose with composite a semiprime or q^2. Anchor y=13:
  9 twins, 10 loose, 9 semi (extra = 125 = 5^3).
- P(t) = prime members in first t window slots; M(t) = t - P(t);
  n0/n1/n2 = slots with 0/1/2 prime members; margin = n2 - n0 = M.
- R_q(t) = composite members in first t slots with lpf = q. Supply
  identity sum_q R_q(t) = 2t - P(t), exact.
- mu(k) = omega_G(mL)*omega_G(mR); S_pair = sum mu; #{mu>=1} = n2;
  tau = (t-P)/S_pair. Zone functional R(t) = (S1^2/M2)/(t-P), S1/M2 =
  moments of per-slot m = omega_G(mL)*omega_G(mR).
- Saturated run: maximal run of load-1 slots (exactly one prime member
  each); L = length; word = L/R side of the prime per slot.
- Bands B_i = (p_i^2, p_{i+1}^2); thickness T = g(2p+g)/6 slots.
- Machine M = gears 5..M on slot space; step M->q' adds the next gear.
  Openings = surviving slots; gap = difference of consecutive openings
  (SLOT frame). Frames: slot x1, adjacent x3, integer x6 (one padded link
  = q' slots = 3q' adjacent = 6q' integers; F_adjacent = 3 F_slot, checked
  at every machine). Period P(M) = prod_{5<=q<=M} q; openings per period =
  prod_{5<=q<=M}(q-2).
- Fuel N_k = co-deletable k-tuples of openings at M->q' (letters
  {+s,-s,0 mod q'}, legality = prefix-sum range <= 1; s = 2u' mod q').
  k_max = largest k with N_k > 0 = A_kill(M->q'). N_k counts TUPLES;
  Constructor's k-hist counts maximal runs (identical iff k_max <= 3).
- Kill-chain word = tuple of gaps spanned by a k-chain; legality by three
  theorems (residue legality v mod q' in {0,+-2u'}; T3 window validity =
  prefix-sum range <= 1; span caps from the deletion ladder, corpus ladder
  and hole lists).
- Padded link: two openings sharing a residue mod q' (letter 0) - needs a
  gap of M equal to exactly q'; z = # zeros in a run's word.
  supply(M,q') = hist_M[q'], one lookup, exact.
- F(M) = max gap; F_j(M) = max sum of j consecutive gaps. Flanks = the two
  gaps bounding an occurrence; FS = flank sum; span(w) = sum of letters.
  litcap(q') = max literal word length, machine-free function of q' mod 35
  (2, 3, 4 or 6).
- Qualifying spectrum Q_j(M; a) = max sum of j consecutive gaps whose j-2
  MIDDLE gaps are all >= a, a = 2u' = 2*round(q'/6). Q_j = 0 means no
  qualifying window of that depth (vacuous, NOT violated). Floor override
  a = 1 gives Q_j(M;1) = F_j(M).
- (D) at M->q': F(M+q') <= F(M) + q'.
- WORD-FREE CRITERION at M->q': max_J Q_J(M; 2u'(q')) <= F(M) + q'.
  Sufficient for (D); the all-depths form is what a kernel rung consumes
  when depth is closed by no_big_run rather than by a fuel cap. The
  depth-capped form maxes over J <= k_max + 1 only.
- k_win = depth of the chain achieving the step's record merged span.
- Marked spectrum Q^[J](old): the r22 object, computed on the OLD
  machine's period with J-1 marks. NAMING (an easy off-by-one): a scan
  "old -> new" computes Q_J(new; a) with a = 2u'' set by the gear q''
  AFTER new, so it decides the step new -> q'', budget F(new) + q''.
  Index a transfer object by the STEP IT DECIDES. Sandwich lemma
  (Constructor): Q_J(new) <= Q^[J](old) <= max_{j<=J} Q_j(new), hence
  max_J Q^[J](old) = max_J Q_J(new) always.
- Lap-phase survival regimes: R0 = no survival requirement, R1 = endpoints
  survive phase c, R2 = endpoints and marks all survive.
- Order-m closure of a dictionary: walks whose every contiguous m-window
  is realised.

## Census results

### C1. Fragile census (r1) - fragile = 2 * twins * W1 / pi_win
Repro: research/fragile_census.py (full prime sweep y = 13..503 + sparse
large y).

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

Lone-composite members at 50021: 287,805,085 loose / 271,522,325 semi
(semiprime share 93.6%). Zero-free-parameter law c = fragile*pi_win /
(twins*W1) = 2, W1 = sum (q-1)/(q-2) over lone-composite members:

    y      13     101    503    1009   2003   5003   10007  20011  50021
    cS   2.200  1.907  1.956  1.973  1.949  1.974  1.985  1.989  1.9914
    cL   2.245  1.907  1.964  1.978  1.950  1.974  1.985  1.989  1.9917

Monotone toward 2 from y=1009; 0.43% error at 50021. fragile/twins grows
like lnln (fit a=3.01/b=-4.48 semi, 3.22/-4.74 loose - FIT, = Mertens
divergence of W1/pi_win). Owning-gear decile shares of loose fragile
(gears ranked, d0 lowest; gear 5 dominates d0):

    y        d0    d1    d2    d3    d4    d5+
    101     58.3  13.2  13.2   4.4   2.9   8.0
    503     69.8  12.8   7.2   3.7   2.7   3.8
    2003    78.1   9.6   4.8   2.9   1.8   2.8
    10007   84.0   7.0   3.5   2.1   1.3   2.1
    50021   87.9   5.3   2.6   1.6   1.0   1.6

### C2. Per-gear closed form (r2)
Repro: research/fragile_pergear.py. frag(q) = 2*tw*((q-1)/(q-2))*S1w(q)/piw
(S1w, piw = 1/ln(m)-weighted): exact to 2e-4 aggregate, Poisson at band
level. obs/pred2 (z2) by gear-rank band 0-50/50-90/90-99/99-100%:
y=10007: 1.0002(0.26), 1.0018(0.36), 1.0159(0.54), 1.5427(2.18);
y=50021: 1.0002(1.32), 1.0015(1.19), 0.9955(-0.59), 1.0055(0.07).
Unweighted there is a real 4-5% deficit at mid/large gears (member-size
geometry) which the 1/ln(m) weight removes entirely; the 10007 top-1%
excess was fluctuation (gone at 50021, 186 events). No twin- or
necessity-specific structure anywhere, incl. gear 50021 itself.

### C3. Prefix censuses at the window bottom (r2)
Repro: research/prefix_census.py; research/data/prefix_census.csv (2400
rows, y = 101..1e8, t = 1..200).

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

Margin never negative for t in [5,200] at y >= 1e4 (125/125 windows); min
always by t <= 11. First double at slot ~2-4 (no growth with y); first
twin above y at ~ln^2 scale. Identity: margin(t) < 0 forces n0(t) > 0 - a
prefix-pigeonhole refutation of X is a nonconstructive twin proof, reach
localised to t <= 4.

### C4. Full-window margin trajectories (r3)
Repro: research/margin_trajectory.py; research/data/margin_summary.csv,
margin_checkpoints.csv, margin_bands.csv.

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

(a) M is gear-blind (primality only). (b) For y >= 503: minM in {0,-1} at
t_min <= 3 (boundary twin), NO later dip, exhaustive to 6.67e9 slots.
(c) Danger zone member-anchored O(1): drift 1 - 6/ln(member) flips
positive at member e^6 ~ 403; last<0 <= 11 for all y >= 503.
(d) M(t) = t - [li(6t+m0) - li(m0)] to 0.1% for t > ~1e3; linear and
t/ln t fits fail globally. (e) Envelope max |M - Mhat| = 0.06-0.18
sqrt(member), coefficient shrinking (0.058 at y=200003); M(t) >= Mhat(t)
- 0.2 sqrt(6t+y) held everywhere tested.

### C5. Supply trajectories and pair schedule (r4)
Repro: research/supply_trajectory.py; research/data/supply_load.csv,
supply_pergear.csv. Sieve verified vs independent spf-table count: 0
mismatches in 3384/13892/23313/28764 checks at y=503/2003/10007/50021.
Fresh gears q in (sqrt(y), y): R_q = 0 until t_act = (q^2-1)/6 - k_lo + 1,
then R_q(t) = 1 + pi(m(t)/q) - pi(q) + T_q(t), T_q == 0 while m(t) < q^3 -
EXACT (T_q share 0.0000 for q > y^(2/3); gear 5's T_q share of R_q 69% at
y=503, 76% at 2003). y=50021 excerpt:

    t          member       A     g5%    rho    tau   S_pair/n2
    133        50815        46    27.2  0.636  0.167    5.39
    1333521    8051143      410   25.1  0.747  0.187    5.03
    417008404  2502100441   5132  23.4  0.829  0.222    4.38

Peak tau (always at t = W): 0.314/0.282/0.249/0.222 at y =
503/2003/10007/50021. NO depth where X's demand exceeds the pair schedule;
t - P <= n2 <= S_pair identically. The whole reality-vs-X distance is
compression: S_pair/n2 = 4.38 vs X-required 4.50 at 50021 - the n0 term
(2.6% of doubles). The problem lives in P(mu=0).

### C6. Multiplicity distribution vs nulls (r5-r6)
Repro: research/multiplicity_census.py; research/data/multiplicity_hist.csv,
multiplicity_summary.csv. Null1 = pairs' CRT classes independent (exact
Poisson-binomial via DFT); null2 = product structure kept, arithmetic
broken. Real / null1 / null2:

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
0.77 down the ladder (the HL correction); singles mass tracks to 3%.
Depth-resolved (twinmass_deciles.py, twinmass_deciles.csv): HL 1/ln^2
allocation reproduces real decile twin counts to 1.000 +- 0.003 at 50021 -
no depth structure beyond density.

### C7. Inversion zone sup R(y) (r6) - generic forcing dies at y ~ 2-5e6
Repro: research/inversion_zone.py; research/data/zone_summary.csv,
zone_curves.csv, zone_anatomy.csv. supB64 = bulk sup over t >= 64 (supR
at t < ~10 is boundary-sensitive and circular).

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
argmax prefixes = m=0 slots plus a block m in {4,6,9,12}; CS efficiency
0.919-0.966. Worked (y=2003, t*=24): hist {0:15, 4:7, 6:2}, S1=40, M2=184,
CS=8.70 > t-P=3 forcing n0 >= 6 (6 twins present). Zone revival at any y =
a twin in that window's first ~200 slots - an exact floor-checkable
restatement of the conjecture.

### C8. Saturated runs to member ~1.2e13 (r7-r9, hunt completed r20)
Repro: research/saturated_runs.py, satruns_model.py; research/data/
satruns_ge10.csv, satruns_records.csv, satruns_renewal.csv,
satruns_windows.csv, satruns_deep_ge10.csv, satruns_deep_renewal.csv
(+_r8), satruns_L15.log + state file.
Records: L=10 at k=59 (member 353); L=13 at k=2452 (member 14711); L=14 at
k = 46,133,660,494 (members 276,801,962,963..963,043), word LRRLRLRRRRLLRL,
MR-verified maximal. L*=13 stood member 1.5e4 -> 2.8e11 and fell on the
constellation curve (predicted first L=14 near 1.6e11, expected count 1.2
at the actual address - Poisson-consistent, no HL deficit). The six
original L=13 instances (MR-verified, none side-alternating):

    k=2452        member 14711        word RLLRRLLLLRLRL
    k=61501443    member 369008657    word LLLRRLLLRRRLL
    k=874166593   member 5244999557   word RLLRRLLRRLRLL
    k=1909351447  member 11456108681  word LLLRRLLLRRRRL
    k=8472005085  member 50832030509  word RRRLLLRRRLLRL
    k=9599932213  member 57599593277  word LLRRLRLLRRRLL

Windows inherit landmarks: y=2003/10007 max at k=2452; y=50021/200003 at
k=61501443. Renewal per member-decade:

    decade  slots      L8     L9    L10  L11  L12  L13+
    5       1.5e5      13      6      0    0    0    0
    6       1.5e6      48     15      1    1    1    0
    7       1.5e7     186     43     13    2    0    0
    8       1.5e8     769    146     45    9    2    1
    9       1.5e9    3435    703    122   28    8    1
    10      1.03e10 12655   2445    433   73   10    3

Per-slot L>=8 rate ~ C/(ln m)^beta, beta = 6.81, C = e^8.33 [fit, 8
decades]. A_8..A_13 = 0.252, 0.220, 0.174, 0.135, 0.084, 0.119; A_L =
exp(-0.197L + 0.197) [fit]. First-arrival ladder [fit, NOT law], CRT cap
[13,32]: L=15 ~5e12, 16 ~2e14, 17 ~7e15, 18 ~3e17, 20 ~6e20, 24 ~5e27,
28 ~9e34, 32 ~3e42 (+1 length per factor ~40 in member). Strict
L/R-alternation cap is 6; runs alternate only in the load sense.
L=15 HUNT COMPLETE to k = 2e12 (member 1.2e13): NO L=15. Deep-range census
L=13: 48, L=14: 5, L=15: 0. All five L=14 addresses: k = 46,133,660,494 /
410,898,686,641 / 706,483,435,891 / 1,663,183,851,213 / 1,984,490,922,377
(word LRLLRRRLRRLLLL). Ratio expectation for L=15 in range 0.52,
P(absence) = 0.59 - sub-1-sigma, record on a curve.

### C9. Band census (r10) - thin bands are NOT twin-poor
Repro: research/band_census.py; research/data/band_census_100003.csv (9,591
bands, heights to 1e10, every slot exact), band_census_2003.csv.
Exact law: for a twin (6m-1, 6m+1) the band above has T = 4m and the twin's
product slot k = 6m^2 sits at offset 2m = T/2 - dead center, composite by
the defining twin. Verified 1223/1223 g=2 bands (+60/60 at calibration).
The rest is density: decade-matched g=2 twin density / all-band density =
0.984, 1.018, 1.006, 1.002 at height decades 6-9 (center-excluded 0.985,
1.019, 1.006, 1.002); center-slot deficit = 1/T. Twin-EMPTY bands: ZERO
through height 1e10 (min 2 twins, only band (25,49)); worst band in
[1e9,1e10) = 342 twins = its Poisson lambda. Min prime members per band =
6. Fragile centers (36m^2+1 prime): 93/1223 = 7.6% at P=1e5 (15.0% at
2003), ~1/ln decline. Verdict: self-reference = exactly 1 deterministic
dead slot per thin band; the binding case binds by LENGTH alone.

### C10. Fuel census - k_max = 4 exists, arithmetic-selected (r11-12, corrected r24)
Repro: research/fuel_census.py (+ --start); research/data/fuel_census.csv
(now carries a `start` column). Validation: N3 = 62 at 19->23 with anatomy
(8,15)/(15,8) = corpus census exactly.

    step      period    N2          N3      N4    k_max
    13->17    5.0e3     72          0       0     2
    17->19    8.5e4     1088        0       0     2
    19->23    1.6e6     11784       62      0     3
    23->29    3.7e7     243816      0       0     2
    29->31    1.1e9     8022924     13000   4     4
    31->37    3.3e10    114848070   70964   216   4

MACHINE 37 FULL PERIOD (three chained ranges, period sums - the labelling
correction is R19): N_1 = 214,551,930,429, N_2 = 1,688,714,780,
N_3 = 3,052, N_4 = 0, so k_max(37->41) = 3 by exhaustive scan. N_3 = 3,052
re-derived with NO scan by research/a_kill.py (CRT+SAT, 14 s): realised
words (14,41):1525, (41,14):1525, (27,41):1, (41,27):1 - sum exact, every
witness assert-verified, so no 3-tuple straddled a resume junction.
N_4 = 0 carries NO junction caveat (all 53 legal 4-words SAT-refuted over
the whole period).
Off-step probes (N3): (19,29) 0; (19,31) 4; (19,37) 0; (23,31) 276;
(23,37/41) 0; (29,37) 374; (29,41/43) 0; (31,41) 2; (31,43/47) 0. k=4
anatomy: 29->31 exactly 4 per period, one word class (10,21,10), flanks
{4,7}; 31->37 has 216, both orientations (12,25,12)/(25,12,25), flanks in
{1,2,3,5,6,10,11,13}. N5 = 0 everywhere scanned. Fuel is
ARITHMETIC-SELECTED (N3 > 0 iff s and q'-s land on abundant gap values),
not smooth in y. 37->4x partial-period probes (37->41 superseded above):

    step      N2          N3     N4   k_max
    37->43    158745169   230    0    3
    37->47    138732684   41     0    3
    37->53    183250785   4091   0    3

SAT-decided caps at full period (fuel_sat.py, cov_sat.py):
k_max(37->41) = 3 EXACT (53 legal 4-words refuted; also by scan);
k_max(41->43) = 3 EXACT (120 words); A_kill(43->47) = 3 EXACT (C22);
A_kill(47->53) >= 3, k=4 open (C22). Chain condition exact at each new
scale: pred F(M+q') = F(2,q')/3 = 58 (29->31), 88 (31->37), 91 (37->41),
145 (q'=53).

### C11. F_j spectra, tier, excess (r13; m37 exact r21/r24; m41-47 r20-24)
Repro: research/spectrum_pass.py, f3_one.py, j5_multi.py (floor-1
transfer), cov_sat.py; research/data/spectra.csv, research/data/f3s/ (one
log per S).

    machine   F1  F2  F3  F4  F5  F6
    13        11  16  23  26  28  31
    17        18  25  28  33  35  40
    19        25  31  35  38  47  50
    23        34  39  50  58  65  77
    29        43  55  65  70  85  90
    31        58  68  85  90  92  97
    37        88  90  97 105 113 120     EXACT, full period

Machine 23 to depth 8: 34 39 50 58 65 77 83 88 (research/m23_ladder.py,
full 37,182,145-slot cyclic period; reproduced independently by Formalist).
Machine 31 to depth 8: 58 68 85 90 92 97 104 110.
MACHINE 37 EXACT (supersedes the r13 prefix row 88 90 95 103 112 115, a
LOWER bound). Sources: full-period scan (fuel37_k5hunt_part2.log,
34,143 s) with F_1/F_2/F_3 independently anchored by CRT+SAT (F_2 = 90,
witness gaps [2,88]; F_3 = 97, witness k = 990,209,189,833, gaps
[37,23,37], 55 UNSAT refutations S = 98..152, cap F_2 + F_1 = 178 by
theorem); F_2/F_3 also reproduced from machine 23 by the floor-1 transfer.
F_4/F_5/F_6 were single-source from the chained scan and are now
junction-checked (m37_junction_check.py: every window of up to 6 gaps
touching a resume junction or the cyclic wrap; worst straddling 6-window
61 vs F_6 = 120).

BEYOND 37 (exact where stated):
    F(41) = 91   F_2(41) = 103   F_3(41) = 110   F_4(41) <= 145
    F(43) = 103  F_2(43) <= 118  F_3(43) = 125
    F(47) = 118  F_2(47) = 134 (r25)   F_3(47) >= 145 (<= 263)
    F(53) = 145  F_2(53) = 159 (r26, C30; >= is unconditional, <= conditional
                 on the span cap 200 - the deletion-ladder cap F_2(53) <= F(59)
                 is unavailable, the corpus F ladder stopping at 53)
    F(59) >= 159 (r26: deletion ladder applied to F_2(53); see C15/C30)
F_2(41) = 103 EXACT with NO descent: cap F_2(41) <= F(43) = 103 (deletion
ladder), floor by SAT witness k = 21,157,523,372,970, gaps [28,75].
F_3(41) = 110 EXACT (r24/f3_41_decide.log): floor-1 transfer from m23 +
{29,31,37,41}, seed 110, span cap 125 (above the deletion-ladder cap 118,
so the cap conditions nothing) - 87,914,175 windows walked, 317
phase-expanded, 814 s, NO 3-gap window of span > 110; floor from SAT
witness k = 30,382,499,692,410, gaps [77,11,22], re-asserted by direct
+-u arithmetic outside the scan and outside SAT.
F_3(43) = 125 EXACT, a first computation (f3_43_prune, 567 s CPU,
117,075,902 windows, cap 150 > the deletion-ladder cap 145); witness CRT'd
and re-verified independently at k = 585,018,519,787,775, gaps [30,28,67].
Same scan gives F_2(43) <= 124.
F_3(47) >= 145 (f3_47_prune COMPLETED: no 3-window of span in (145,200] at
machine 47; 156,146,894 windows, 850 s CPU); witness CRT'd and re-verified
at k = 36,068,193,854,725,102, gaps [28,33,84]. Exact value open only above
the 200 span cap (deletion-ladder ceiling 263).

Tier rule: the step record F(M+q') is realizable only from chains with
F_{k+1} >= F(M+q'); min k per step 13->17..31->37 = 2, 1, 2, 2, 2, 3.
Lemma 2 (k >= 3) is load-bearing at exactly one step: 31->37 (record 88 >
F_3 = 85; k=4 chains reach <= 87). Excess census (lem1 = F2 - F, exc =
F+ - F2):

    step      incr  lem1  exc   exc/incr  adj incr/q'  margin vs 2.5
    13->17    7     5     2     0.29      1.235        50.6%
    17->19    7     7     0     0.00      1.105        55.8%
    19->23    9     6     3     0.33      1.174        53.0%
    23->29    9     5     4     0.44      0.931        62.8%
    29->31    15    12    3     0.20      1.452        41.9%
    31->37    30    10    20    0.67      2.432         2.7%  <- binding
    37->41    3     2     1     0.33      0.220        91.2%

Lemma-1 margin at m37 (F_2 - F = 2, 0.95q', largest measured) is
STRUCTURAL: both m37 maximal gaps carry gaps of 2 on both sides, so
F_2(37) = 90 IS the minimum gap 2 abutting the maximal gap 88 (which also
explains why m37's second maximal-gap address sits two slots off the F_2
witness). NEGATIVE: corr(exc share, N3 per opening) = -0.03 - excess is
set by flank quality (N2 ubiquitous, 2-5% of openings everywhere); chain
length enters only as a threshold. BUDGET: the binding step 31->37 sits
2.7% under alpha = 2.5; alpha = 3 needs F(2,53) <= 513, and F(2,53) = 435
EXACT (maxgap53_pruned.log) - PASSES with 15% room.
Shallow flatness F_4 - F vs q': 11, 15, 15, 13, 24, 27, 32 at machines
11..31 = ratios 0.85, 0.88, 0.79, 0.57, 0.83, 0.87, 0.86 - flat at
~0.85 q', NOT gaining room. Machine 37 exact: F_4 - F = 17 (0.41).

### C12. Padding census - padding IS the gear-37 anomaly (r14-16; m37/m41 r20)
Repro: research/padding_census.py, padded_link_anatomy.py, hist_probe.py;
research/data/padding_census.csv, gap_histograms.csv. hist_probe
validated: reproduces full-period padding censuses exactly (m29:
2090/84/0/2 at q' = 31/37/41/43; m31: 26366).
Total gaps of M per period: 1485, 22275, 378675, 7952175, 214708725,
6226553025 at machines 13..31 - i.e. exactly prod(q-2), as a CIRCLE must give.
(CORRECTED r25: these were printed one short in every earlier round, the
linear-close defect - C26. The old row read 1484, 22274, 378674, 7952174,
214708724, 6226553024.) The separate figure 1,688,711,736 at machine 37 is
NOT a gap total (prod(q-2) = 217,929,355,875 there) - it is the 37->41
z-split total of C12's own table, and is labelled as such below.
Gap-tail selection: machine 23 has gap 28: 322, 29: 6, 30: 112 - value 29
suppressed ~50x against both neighbours. Supply (# gaps of M = exactly q'):

    machine  F     q'=29  q'=31  q'=37  q'=41  q'=43  q'=47  q'=53
    19       25    0      0      0      0      0      0      -
    23       34    6      20     0      0      0      0      -
    29       43    -      2090   84     0      2      0      -
    31       58    -      -      26366  134    860    226    -
    37       88    -      -      -      61460  -      -      -

supply(37,41) = 61,460 (2.820e-7 of openings) at FULL PERIOD; gaps == 82:
0; z >= 2 runs: 0 (supersedes the 4.85%-prefix row 2948/7074/2295/515 and
the r14 estimate). Machine 41 (prefix 0.394% of period 5.0708e13):
hist_41[43] = 66,235, [47] = 25,032, [53] = 5,748, [59] = 33. Onset rule:
F(M) >= q' NECESSARY (theorem) but NOT SUFFICIENT (supply(29,41) = 0
despite F = 43 >= 41 - spectrum hole). Boundary sharp: at q' = F(M) (m29,
43) supply = 2 = # maximal gaps. 2q' never fits at machines <= 37.

    31->37 z-split       count          max flanked span
    z = 0 (literal)      114,750,740    71
    z = 1 (padded)       26,366         88   <- the true F(M+37)
      k=2                26,030         85
      k=3                336            88   <- the record run
    z >= 2               0

    37->41 z-split (padding37 full period, 19,694 s)
    z = 0                1,688,650,276  90
    z = 1                       61,460  91   <- STEP RECORD, k=3
      k=2                        58,416  83
      k=3                         3,044  91
    z >= 2                            0

PADDING CARRIES THE RECORD AT 37->41 AS AT 31->37; k_win(37->41) = 3 on
the full period. Literal-only would give 71, not 88, at 31->37: the record
needs k=3 AND one padded link (tier and padding are independent axes; the
tier bound F_{k+1} is padding-blind). Without padding the 31->37 increment
would be 13 (adjacent 1.054, 58% margin); with it, 30 (2.432, the 2.7%
margin) - the whole binding-step problem is one padded link. Worked link
(m31, q'=37): openings k = 634158 / 634195, residues [15,15] mod 37,
member gap 222 = 6x37 - padded iff the openings share ANY residue mod q'.
Double-padding statistic supply^2/gaps = 0.020 (29->31), 0.112 (31->37),
0.017 (37->41), ~33 at 41->43 (m41 prefix scaled ALONG its period -
CRT-homogeneous, the safe direction). FIRST DOUBLE-PADDED RUN EXISTS,
found STRUCTURALLY not by scan: word (43,43) at 41->43, witness
k = 116,431,845,582 (openings k, k+43, k+86 sharing one residue mod 43,
z = 2). Gap 86 = 2q' also occurs at m41 - first 2q' padding anywhere.
Realizable 41->43 padded words: (14,43),(29,43),(43,14),(43,29),(43,43).
COUNT of (43,43) at machine 41 = 4 EXACT per period (32 s by cov_count
model enumeration, research/data/m41/count_4343.log): 116,431,845,582 /
21,381,235,210,387 / 29,327,142,044,062 / 50,591,945,408,867, each
re-verified by assert (openings at +0/+43/+86, all 84 other interior slots
blocked, both links padded) and CROSS-CHECKED BY THE MIRROR LAW - exactly
two mirror pairs summing to P - 86 = 50,708,377,254,449
(m41_4343_verify.py). The z=2 shape recurs at 43->47 and 47->53 (C22).

### C13. Flank envelope + the (D) criterion (r17; Q_j table corrected r21)
Repro: research/flank_envelope.py, envelope_analysis.py,
unrestricted_max.py, qualifying_spectrum.py, kwin_census.py,
qspec_audit.py; research/data/flank_envelope_words.csv, _joint.csv,
_uncond.csv, _spectra.csv, _gaphist.csv, qspec_table.csv. Validation:
reproduces Constructor r16 at 29->31 exactly (FS_max 48 at (18,30), F 43),
r11 fuel anatomy, r13 spectra.
IDENTITY: span(w) + FS(occurrence) <= F_{ell+2}(M) identically, so (D)
[FS_max(w) <= F + q' - span(w)] is implied by F_{ell+2}(M) <= F(M) + q',
depth needed <= litcap(q') + 1 <= 7. Ceiling TIGHT: machine 19, word
(10,), k = 137,328, flanks (21,4): span + FS = 35 = F_3(19) exactly.
Ledger with the measured fuel cap folded in (k_max from C10):

    step      k_max  F_{k_max+1}   F+q'    verdict
    13->17      2      F_3 = 23      28    IMPLIES (D)
    17->19      2      F_3 = 28      37    IMPLIES (D)
    19->23      3      F_4 = 38      48    IMPLIES (D)
    23->29      2      F_3 = 50      63    IMPLIES (D)
    29->31      4      F_5 = 85      74    OPEN (short by 11)
    31->37      4      F_5 = 92      95    IMPLIES (D)
    37->41      3      F_4 = 105    129    IMPLIES (D)

The 29->31 residual = the length-3 words: (21,10,21) span 52, 0
occurrences in the full 1.078e9 period; (10,21,10) span 41, exactly 4:

    k = 220,171,102  flanks (7,7)  FS 14      k = 406,081,827  (4,7)  11
    k = 672,200,337  flanks (7,4)  FS 11      k = 858,111,062  (7,7)  14

alpha=3 needs FS <= 33; measured max 14, margin +19 = 0.61q'. Both refined
criteria below close this word too; addresses stand as data. The envelope
follows OCCURRENCE COUNT, not span (monotonicity as a machine law is
FALSE - R11): counts fall 2-5 orders across a step's compatible spans
(29->31: 7,815,766 / 205,068 / 6,500 / 4 at spans 10/21/31/41).
Rarity null (draw 2*occ flanks from the machine's own gap histogram, take
max; eff = min(null median, ceiling F_{ell+2} - span)):

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
real structural suppression on top of rarity. THE EXCEPTION: (10,21,10) at
29->31, obs 14 vs null 15, p = 0.4732 - pure rarity. Envelope = (i)
spectrum ceiling (theorem, tight) + (ii) rarity order statistic + (iii)
7-15 gap-unit suppression; a derivation of (D) for long words must come
from occurrence-count x gap-tail bounds, not from monotonicity (false) or
the ceiling (too weak: 44 vs needed 33).

    Literal-word margin (min over compatible words)
    step      F    q'   min margin   /q'    binding word (span, FS_max)
    11->13     7   13      +12      0.923      (4)      (4, 4)
    13->17    11   17      +10      0.588      (6)      (6, 12)
    17->19    18   19      +12      0.632      (13)     (13, 12)
    19->23    25   23      +14      0.609      (8,15)   (23, 11)
    23->29    34   29      +20      0.690      (10)     (10, 33)
    29->31    43   31      +16      0.516      (10)     (10, 48)

Flat band [0.52, 0.92] q', no downward trend; the word-restricted margin
NEVER collapsed. (Constructor's +7 = 0.19q' is the PADDED tier at 31->37,
a different object; neither shrinking.)
Unrestricted maximisers: of 132 maximisers of F_j at machines 19/23/29,
ZERO literal and ZERO qualifying - the shape is always near-maximal FLANKS
with the machine's smallest gaps interior, forbidden by the interior-gap
floor >= 2u' (Constructor's Theorem 1):

    machine 23, F_3 = 50: flanks (23,23) interior (4,)      k = 2,082,580
    machine 23, F_4 = 58: flanks (28,23) interior (4,3)     k = 29,098,935
    machine 23, F_5 = 65: flanks (28,10) interior (5,2,20)  k = 36,845,450
    machine 29, F_3 = 65: flanks (39,23) interior (3,)      k = 407,599,253
    machine 29, F_4 = 70: flanks (31,12) interior (4,23)    k = 717,564,717
    machine 29, F_5 = 85: flanks (30,18) interior (4,3,30)  k = 772,741,833
                          flanks (27,18) interior (3,7,30)  k = 725,859,998

QUALIFYING SPECTRUM Q_j, full period - THE r21-CORRECTED TABLE (four of
seven rows as first printed in r17 were WRONG; R14):

    step     a    F   F+q'  F_3 F_4 F_5 F_6 F_7   Q_3 Q_4 Q_5 Q_6 Q_7   crit
    11->13   4    7   20     16  18  23  26  28    16  18  20   0   0   +4
    13->17   6   11   28     23  26  28  31  34    18  23   0   0   0   +10
    17->19   6   18   37     28  33  35  40  43    28  31  32  34   0   +9
    19->23   8   25   48     35  38  47  50  58    35  37  38   0   0   +10
    23->29  10   34   63     50  58  65  77  83    43  50  55  60   0   +13
    29->31  10   43   74     65  70  85  90  92    65  68  71  71  71   +3
    29->37  12   43   80     65  70  85  90  92    65  68  68  71   0   +9

Re-derived with qualifying_spectrum.py AND verified by direct enumeration
at each disputed address (qspec_audit.py, 9 entries asserted; e.g. 17->19
j=6 at k = 9,173 has gaps [2,7,6,7,8,4] = 34, all middles >= 6). The
ALL-DEPTHS maxima differ from the capped "crit" column: at 23->29 the
all-depths max is 60, margin +3, not +13. Machine 23 to depth 8:
Q_j(23;10) j=3..8 = 43 50 55 60 0 0; longest run of gaps >= 10 is 4.
Crux at 29->31: F_5 = F + 42 fails (42 > 31) but Q_5 = F + 28 passes - the
size threshold (a theorem) alone converts "arithmetic luck" into
sufficiency at the binding depth. Q_j = 0 also delivers the fuel cap in
the same object (Q_j = 0 for j > 5 at m19, j > 6 at m17/m23/m29+37, j > 7
at m29+31). Largest reachable steps (qspec31 full period; qspec41 prefix
0.08%):

    machine 31 (F=58, F+q'=95, a=12):  F_j (j=3..8) 85 90 92 97 104 110
                                       Q_j          85 90 91 90  88   0
    machine 41 (F=91, F+q'=133, a=14, PREFIX): F_j 110 112 118 123 130 138
                                               Q_j 110 112 110 117 122 121

Q_7(31) = 88 = F(31->37) exactly - the bound is ATTAINED at the binding
step. Span-resolved envelope (second word-free criterion): span(w) +
H_ell(span(w)) <= F + q', H_ell(s) = max flank sum over ALL runs of ell
gaps with interior span exactly s. Implied at ALL 44 measured (step,
compatible word) pairs, incl. (10,21,10): 41 + 24 = 65 <= 74. Under either
refined criterion NO residual remains at any measured step through 41->43.

    k_win census (kwin_census.py; validated vs known records, r11 tuples)
    step      k=1   k=2   k=3   k=4    k_max  k_win  spread
    19->23     31    33    34    -        3      3    8.8%
    23->29     39    43     -    -        2      2    9.3%
    29->31     55    58    55    55       4      2    5.2%
    31->37      -     -    88    68       4      3   22.7%
    37->41      -     -    91     -       3      3     -

Par trading confirmed (5-22.7% spreads). At 29->31 the k=4 fuel exists and
LOSES (55 vs record 58); at 31->37 and 37->41 the record IS carried at
k=3. Winners: k = 137,307 (19->23, (15,8)); k = 14,995,460 (23->29, (10,));
k = 278,620,515 (29->31, (10,) - matches the envelope census's record
address, span 10 + FS 48 = 58). kwin31 FULL PERIOD (3,264 s): k_win = 3,
record 88 = F(37) - merge law exact end-to-end against hist37's direct
scan; k=4 tuples 216 (max merged 68, loses by 20); the KW=8 search found
ZERO k >= 5 tuples, confirming the r17 pre-registered test (no length >= 4
compatible word occurs at 31->37; r11's k_max = 4 stands). NO k_win >= 4
ANYWHERE.

### C14. Hole structure of the gap spectrum (r17; m37/m41 exact r20-21)
Repro: research/hole_structure.py off full-period gap histograms;
cov_sat.py for m37/m41.

    machine 11  F =  7   holes: none
    machine 13  F = 11   holes: {9}
    machine 17  F = 18   holes: {17}
    machine 19  F = 25   holes: {19, 24}
    machine 23  F = 34   holes: {24}
    machine 29  F = 43   holes: {41, 42}
    machine 31  F = 58   holes: {54, 56, 57}
    machine 37  F = 88   holes: {73,74,75,76,78,79,80,81,82,83,84,86,87}
    machine 41  F = 91   holes: {84, 87, 89}
    machine 43  F = 103  hole at 102 confirmed; no others observed

Hole-count ladder 11..41: 0, 1, 1, 2, 1, 2, 3, 13, 3. m37's list is
DEFINITIVE (full-period scan 11,829 s, reproduced by SAT in 123 s); 69 is
NOT a hole. Machine 41 COMPLETE by COV-SAT (period 5.07e13, never
scanned): F(41) = 91, holes {84,87,89}, tail 92..100 all refuted - two
independent methods agree (COV vs the merge-law record from padding37).
Healing law holds at every step: 9, 17, 19, 24 healed by the next gear;
only v = 24 survives one step (19 -> 23), as do 84 and 87 at 41; 89 >=
F(37); no hole is ever created below the previous machine's F - the
spectrum fills monotonically from below. v = 102 is a hole below F(43).
Residue law (class share x p; 1.00 = flat), stable and converging:

    machine   mod 2        mod 3              mod 5
    11      0.97 1.03   0.58 0.67 1.75   0.83 0.90 2.22 0.83 0.21
    17      0.91 1.09   0.61 0.83 1.56   1.04 0.85 1.87 0.88 0.36
    23      0.88 1.12   0.64 0.91 1.45   1.13 0.81 1.74 0.92 0.40
    29      0.88 1.12   0.65 0.93 1.42   1.16 0.80 1.70 0.93 0.41
    machine 29, mod 7: 0.78 0.90 1.64 1.15 0.67 1.40 0.45

Richest classes are +-s of the small gears (v = 2, 5 mod 7 = +-s for gear
7; v = 2 = s mod 5) - the small gears' letters are visible in the whole
machine's gap histogram; NOT the naive endpoint-survival prediction (v = 0
richest, +s/-s equal; measured v=2 mod 5 at 1.70 beats v=0 at 1.16).
UNEXPLAINED. The residue score R(v) = prod_p share_p(v mod p), p <= 7,
does NOT predict the holes: hits at m13 (rank 2/7), m19 (1/14), m23
(2/18); misses at m29 (ranks 7, 10 of 23) and m17 (rank 10/10, HIGHEST).

### C15. The corpus F ladder, complete to 53 (r21, r23)
F(2,y) plus the frame identity F_adjacent = 3 F_slot determines F(y):

    y         19   23   29   31    37    41    43    47    53      59
    F(2,y)    75  102  129  174   264   273   309   354   435   >=477
    F(y)      25   34   43   58    88    91   103   118   145   >=159

Machines 19..41 match our independent measurements 6/6 where both exist.
THE y = 59 ENTRY IS NEW (r26) AND IS A LOWER BOUND, NOT A LADDER VALUE: the
corpus ladder has no 53->59 rung, and F(59) >= F_2(53) = 159 comes from the
deletion-ladder bound K3 applied to this lane's own F_2(53) computation (C30).
It is UNCONDITIONAL - the >= side of F_2(53) rests on an exhibited witness at
machine 53 - and it leaves (D) at 53->59 (which needs F(59) <= 204) at most 45
of room.
F(2,47) = 354 IS A FIRST COMPUTATION (r23): the corpus had no 43->47 rung
("NOT RUN"). Method: rust2/src/bin/maxgap_pruned.rs, the endpoint-law-
pruned covering search, validated first on two known values (y = 41 from
L = 270: F(2,41) = 273 in 15 s; y = 43 from L = 300: F(2,43) = 309 in
199 s), then at y = 47: RUN OF L = 354 IS NOT COVERABLE -> F(2,47) <= 354
(research/data/maxgap47_pruned.log), while F(47) >= 118 (COV-SAT witness)
-> F(2,47) >= 354. Hence F(2,47) = 354 and F(47) = 118 EXACT. Independent
consistency: single-L probes at 390 and 417 also refute.
READING TRAP (this lane burned twice): maxgap_pruned prints "F(2,y) = L"
whenever L refutes, whatever L it started at - that means "L is not
coverable", i.e. F <= L, and is the exact value only when everything below
is known coverable.
Increment F(43) -> F(47) = 15; adjacent incr/q' = 45/47 = 0.957, far under
alpha = 2.5. The MERGE-LAW cross-check at 43 and 47 REMAINS OPEN (both
values came from the covering search); 21 independent SAT refutations at
machine 43 (v = 102, 104-109, 111-116, 118, 120-126) agree exactly with
the pattern F(43) = 103 demands.

### C16. Gap-pair census / p_j (r20)
Full period at 13/17/19/23/29/31 + machine 37 at 12.9% (1.6e11 slots,
28.19e9 gaps). Lags 1-5, run-min m = 2..6, all floors; CSVs
research/data/gap_pair_{hist,joint}.csv, summary pj_deficits.csv.
p_m/p_1^m at each machine's own floor a = 2u'(next prime):

    machine  a    p1       m=2     m=3     m=4     m=5     m=6
      13     4  0.3733   0.890   0.752   0.556   0.373   0
      17     6  0.1852   1.123   0.721   0.305   0       0
      19     6  0.2410   1.090   0.806   0.297   0.020   0
      23     8  0.1371   0.939   0.319   0.029   0.016   0
      29    10  0.1188   0.801   0.162   0.049   0.014   0
      31    12  0.0766   0.581   0.149   0.053   0.005   0
      37    14  0.0530   0.469   0.155   0.056   0.014   0.0009

Deficit x6.5-x6.7 at m=3, x18-x20 at m=4, x70-x190 at m=5 - stable across
23..37. Lag-resolved: deficit at lags 1-3, EXCESS at lags 4-7 - one
phenomenon, the corridor (C17).

### C17. The corridor resonance (r20; docs/novel/)
Repro: research/bool_lag_census.py (+analyze_bool_lag.py; full period
17..31), corridor_resonance.py, transfer_spectrum.py, dft_events.py;
research/data/bool_lag16_31.csv.
1. THE WAVE: the qualifying-gap indicator's autocorrelation is a barely
   damped oscillation. m29 floor 10, lags 1..15: 0.801 0.684 0.510 0.800
   1.112 1.257 1.204 0.995 0.781 0.717 0.848 1.082 1.254 1.250 1.094
   (trough 3, peak 6, trough 10, peak 13-14; second cycle undamped). m31
   floor 12: trough 2, peak 5-6, trough 9-10, peak 12-13. Period ~
   35/mean_gap (8.2/7.5/7.0/6.5 predicted at 19/23/29/31; measured peaks
   8/7-8/6-7/~7).
2. MECHANISM: slot-separation autocorrelation of big-gap left endpoints
   peaks EXACTLY at 35/70/105: m23 3.22/3.45/2.63; m29 3.64/4.37/2.94;
   m31 3.41/4.20/2.97 (neighbours 0.17-1.3; sep 70 > sep 35 everywhere).
   Endpoints PINNED mod 35: invariant core {10,12,18} enriched >= 1.2x at
   all five machines 17..31; companions drift (17 small machines, 7 at
   23/29, 5 at 31); exact four-way tie 10/12/17/18 at m17 (2.42) and m19
   (2.13), tie-pairs (10,18),(12,17) at m23. Poorest {28,30,32,33} at
   0.12-0.46.
3. NOT MARKOV for k <= 4 steps (TV of exact factorisations, m29 floor 10
   W16: 0.151/0.134/0.092/0.080; m31 floor 12: 0.088/0.060/0.043/0.041).
   The value-level one-step chain predicts NO deficit at lags 2-5
   (0.99-1.06) where the census says 0.51-0.68: THE ANTI-CORRELATION LIVES
   AT RANGE 2-3 AND IS INVISIBLE TO LAST-GAP STATE. A transfer matrix
   needs corridor phase (mod 35 at least) in its state. Pattern counts are
   exactly mirror-symmetric (machine reversal) at every machine.
4. lam2 -> phi/3: subleading eigenvalue of the measured lag-1 transfer
   matrix, real negative: |lam2| = 0.6273/0.5959/0.5722/0.5583/0.5515/
   0.5462/0.5425 at m13..m37; distance to phi/3 = 0.53934: +0.0880/
   +0.0566/+0.0329/+0.0190/+0.0122/+0.0069/+0.0032 - geometric, factor
   ~0.6/machine. kappa(2) = 0.5448 is PASSED and DEAD as a limit.
   Convergence to phi/3 CONJECTURED (7 exact points, no fit claimed).
5. DFT identity events: c_q(g) = inverse transform of the exposed-set power
   spectrum |hat1_A(t)|^2 = 4cos^2(2 pi u t/q) - ZERO mismatches, all gears
   5..53, all lags (376 checks); corridor mod 35: census = c5*c7 =
   product-spectrum inverse DFT, 35/35. Gap-histogram ripple: |H_5(1)|/H0
   falls 0.31 -> 0.18 (m13..m37) while arg H_5(1) = +126 deg +- 2 at ALL
   SEVEN machines (m7: +121 -> +133 slow drift) - the C14 residue law has
   a machine-independent PHASE that is NOT the naive +-s ripple (which
   would give 0/180). UNEXPLAINED.
6. MACHINE-31 CORRIDOR-PHASE CENSUS, FULL PERIOD, both moduli
   (6,226,553,025 gaps; cross-check vs tm_resid_runs.csv EXACT). Depth-3
   V-runs, exact count 508, predicted by:

    independent  39,072.91 (x76.9)   VALUE          2,241.51 (x4.41)
    PHASE mod 35  2,337.51 (x4.60)   HYBRID mod 35    803.50 (x1.58)
    PHASE mod 385 1,561.20 (x3.07)   HYBRID mod 385   683.07 (x1.35)

   The phase chain's subleading eigenvalue is COMPLEX: |lam_2| = 0.836951
   (0.517060 + 0.658131i) at mod 35, 0.998581 at mod 385 - the mod-385
   chain has almost NO spectral decay, so its good fit is carried by the
   state space, not by a gap.

### C18. Record multiplicity and the mirror law (r21)
Repro: research/record_multiplicity.py (direct full-period scan),
mirror_law.py.

    machine    13   17   19   23   29   31   [37]  [41]
    mult       12   20   20    4    2    4    [2]   [4]

m23/m29/m31 reproduce the single-source SAT ladder exactly by an
independent method; 13/17/19 are new; m37/m41 measured once.
MIRROR LAW: EVERY ENTRY IS EVEN, necessarily - each gear blocks the
symmetric pair {u_q, -u_q}, so the opening set is closed under k -> -k mod
P and maximal gaps come in MIRROR PAIRS summing to P - F. Verified at
13-29 with zero self-mirror gaps; m31's four and m37's two are exact
mirror pairs. An application of the machine-reversal symmetry (C17.3), not
a new law; it also cross-checks the four (43,43) addresses at m41 (C12).
ADJACENCY of two maximal gaps REFUTED at m31 (58,58), m37 (88,88), m41
(91,91) - previous reach was y <= 23.

### C19. Depth-3 V-run counts (r21, Constructor's asks)
run_3(31; V(37)) = 508 CERTIFIED COMPLETE. Six nonzero words:
(12,12,25):139, (12,25,12):188, (25,12,12):139, (12,25,25):7,
(25,12,25):28, (25,25,12):7; all 58 others zero (44 by spectrum prune, 14
by UNSAT).
run_3(37; V(41)) = 8 EXACT. Sole word the padded palindrome (14,41,14),
witness k = 1,120,456,097,388, re-verified independently. Shape echo:
29->31's only k=4 word was (10,21,10).

### C20. (D) and the word-free criterion through 47->53 (r23)
Repro: research/deletion_ladder.py (all asserted), j5_multi.py,
multi_witness_verify.py. (D) at alpha=3 IS DECIDED TRUE AT EVERY STEP
THROUGH 47->53 - with the C15 ladder complete this is arithmetic:

    19->23  34 <= 48  +14     31->37  88 <= 95   +7      43->47 118 <= 150  +32
    23->29  43 <= 63  +20     37->41  91 <= 129  +38     47->53 145 <= 171  +26
    29->31  58 <= 74  +16     41->43 103 <= 134  +31

CRITERION MARGINS, EXACT AND ALL-DEPTHS (the quantity a hypothesis-free
theorem consumes), replacing every prefix row. Row "M -> q'" =
max_J Q_J(M; 2u'(q')) against F(M) + q':

    step     max_J Q_J(M; a)   budget F(M)+q'   margin   /q'    litcap(q')
    13->17         23                28           +5     0.29        2
    17->19         34                37           +3     0.16        2
    19->23         38                48          +10     0.43        4
    23->29         60                63           +3     0.10        3
    29->31         71                74           +3     0.10        4
    31->37         91                95           +4     0.11        6
    37->41        114               129          +15     0.37        2
    41->43        132               134           +2     0.047       4
    43->47        152               150           -2      -          -     FAILS
    47->53        177               171           -6      -          6     FAILS

The 23->29, 29->31 and 31->37 rows reproduce C13's all-depths values
exactly (60, 71, 91) by a completely different method - the strongest
control on the transfer machinery. Q_J(37;14), J = 2..7 = 90, 97, 103,
110, 112, 114 is a NEW EXACT OBJECT (previously only a prefix lower bound
at J = 3); max 114 <= 129 certifies 37->41 hypothesis-free with the
all-depths quantity, margin +15 = 0.37 q'.
THE TWO FAILURE WITNESSES (multi_witness_verify.py: openings where
claimed, every other interior slot blocked, every middle gap >= floor):

  Q_7(43; 16) >= 152 at k = 110,350,776,715,218 (m43, P = 2.180e15)
      gaps [35, 20, 20, 17, 20, 17, 23], middles [20,20,17,20,17] >= 16
      145 interior slots blocked, asserted.  Budget F(43) + 47 = 150.
  Q_7(47; 18) >= 177 at k = 41,120,916,229,562,503 (m47, P = 1.025e17)
      gaps [14, 20, 36, 19, 20, 45, 23], middles [20,36,19,20,45] >= 18
      170 interior slots blocked, asserted.  Budget F(47) + 53 = 171.

WHAT THIS DOES AND DOES NOT KILL. NOT (D) - both steps hold with room: the
theorem is true, the vehicle stops proving it. It KILLS the all-depths
(hypothesis-free) form of the word-free criterion from 43->47 on. It does
NOT kill the DEPTH-CAPPED form: the merge law only ever needs depths
j <= k_max + 1, and both failures live at J = 6 and 7 ALONE - every depth
<= 5 sits under budget at both steps (below the seed, hence <= 149 and
<= 170). RESTORATION THRESHOLDS: k_max <= 5 restores 43->47 (fails only at
J=7); k_max <= 4 restores 47->53 (fails at J=6,7). Both attacked in r24
(C22); 43->47 IS RESTORED.
MECHANISM: the criterion maximises over windows whose j-2 MIDDLE gaps all
clear a = 2u'. The floor grows with the added gear (16 at m43, 18 at m47)
but the mean gap only like 1/prod(1-2/q) - 6.26 and 6.54 slots - so a
qualifying window is a run of consecutive gaps each ~3x the mean. Up to
machine 41 the deep ones are simply ABSENT, so Q_J collapses to 0 or
plateaus; at 43/47 depth-7 runs exist for the first time and the criterion
immediately clears the budget. The failure is ARITHMETIC (when do six
consecutive gaps >= 2u' first occur), not asymptotic.
MARGIN LADDER, all ten steps: +5, +3, +10, +3, +3, +4, +15, +2, -2, -6 -
never comfortable, inside 0.13 q' at four of ten. litcap NOTE: r20/r21's
"margin tracks litcap" came from a row of DIFFERENT q' at ONE machine
(common-mode F); the ladder above is ACROSS machines, where the margin
does NOT order by litcap (litcap-4 steps run 0.047-0.43, litcap-2 steps
0.16-0.37). The failing step IS litcap-6, consistent with the hedge, but
litcap-4 41->43 sits at +2 and is nearly as tight. Honest reading: the
margin is small, non-monotone and arithmetically selected.

### C21. Gap 4-tuple dictionaries (r23-r24)
Repro: research/gap_tuples.py, gap_tuples_par.py (range-partitioned +
merge), gap_tuples_lean.py (2^28-bool "seen" array replaced by a set, runs
in a few hundred MB). Opening count asserted against prod(q-2) and the
maximal gap against F(y) in every run.

    machine 23:  15,696 realised 4-tuples   research/data/gap_tuples_23_4.csv  (1 s)
    machine 29:  45,854 realised 4-tuples   research/data/gap_tuples_29_4.csv  (32 s)
    machine 31: 115,193 realised 4-tuples   research/data/gap_tuples_31_4.csv  (564 s)
    machine 37: 291,675 realised 4-tuples   research/data/gap_tuples_37_4.csv
                DELIVERED r25 - six range workers over the full period
                P(37) = 1,236,789,689,135, merged with both assertions
                passing: openings 217,929,355,875 = prod(q-2) EXACTLY and
                max gap = 88 = F(37). Induced levels 75 / 2,053 / 30,325.
    machine 41: 4,239,676 4-tuple certified SUPERSET (K4 transfer from the
                exact m37 dictionary, 77 s) - the FIRST dictionary at a
                machine beyond every scan (P(41) = 5.07e13); see K4.

TWO FREE CONSISTENCY CHECKS THE TOOLS DID NOT KNOW ABOUT, both passed:
- m37 exact: 75 distinct gap values = 88 - 13, and the 13 missing values are
  EXACTLY C14's independently-derived m37 hole list.
- The SIX WORKERS' RANGE STATISTICS ARE MIRROR-PAIRED: w0 and w5 agree to the
  unit (36,321,559,350 openings, max gap 88, 249,494 distinct tuples), as do
  w1/w4 (36,321,559,303 / 85 / 249,493) and w2/w3 (36,321,559,284-5 / 77 /
  249,474). Six equal ranges tiling a period closed under k -> -k must pair
  like this (C18); they were computed by six independent processes.

Machine 31 induced levels: 3-tuples 15,019; 2-tuples 1,253; distinct gap
values 55 = 58 - 3, the three missing values being exactly C14's hole list
{54, 56, 57} - a free consistency check the tool did not know about.
VALIDATION: m31 delivered two ways (single-process vs four independent
range workers; CSVs BYTE-IDENTICAL); m23 re-derived BYTE-IDENTICAL on a
clean process; m29 re-derived by a SECOND implementation (base-(F+1)
packing, range-partitioned, boundary-overlap reads) BYTE-IDENTICAL after
sort. GROWTH ~7.7x per level (55, 1,253, 15,019, 115,193, ...), so
enumerating by tuple to the depth a machine-37 window needs (j ~ 10; a
span-105 window can hide ~6 killed interiors) is ~1e9 entries. The right
version enumerates by KILL PATTERN: the pattern is a choice of which
interiors die, and the phase condition on it is two residues mod q'.

### C22. A_kill at 43->47 and 47->53 - the criterion handoff (r24)
Repro: research/a_kill.py + a_kill_word.py + a_kill_par.py; logs
research/data/r24/akillp_43_47.log, akillp_47_53.log; every REALISED
witness CRT'd and assert-verified at the machine.
ANCHOR (re-run clean post-outage): at 37->41 the tool returns N_3 = 3052
EXACT in 14 s - equal to the corrected full-period scan sum - with the
complete realised-word inventory (14,41):1525, (41,14):1525, (27,41):1,
(41,27):1, and N_4 = 0, independently re-proving k_max(37->41) = 3 with no
junction caveat. Fifteen hours of scan against 14 s of SAT, agreeing to
the unit.

43->47: A_kill = 3 EXACT. THE CRITERION IS RESTORED AT 43->47.
  k=3 level, all 15 words decided:
    REALISED (5): (16,47) k=1,536,721,187,856,312
                  (31,47) k=1,685,419,613,249,542
                  (47,16) k=2,146,450,460,877,525
                  (47,31) k=535,717,811,356,625
                  (47,47) k=149,017,826,597,238
    ZERO (10): (16,31),(31,16),(16,78),(31,63),(63,31),(78,16),
               (16,94) 667 s UNSAT, (94,16) 374 s, (63,47) 1937 s,
               (47,63) 2156 s
  k=4 level: ALL NINE candidate 4-words ZERO (spans 94/110/125/141; three
    refuted structurally at 0 SAT calls, six by UNSAT in 7-73 s each;
    76 s total). N_4(43->47) = 0.
  => k_max = 3 <= 5 = the restoration threshold: the merge law needs
  depths j <= 4 only, and every J <= 6 sits under the budget 150.
  DOUBLE-SOURCED at the top spans: the span-125/141 zeros agree with the
  F_3(43) = 125 prune theorem by an independent method. NEW EVENT: (47,47)
  is a DOUBLE-PADDED 3-chain (both gaps = 0 mod 47) - the z=2 shape first
  found at 41->43 recurs.

47->53: A_kill >= 3, EXACT VALUE UNDECIDED AT CLOSE.
  k=3 level, 15 of 19 words decided:
    REALISED (11): (18,35),(18,53),(18,88),(35,18),(35,53),(35,71),
                   (53,18),(53,35),(53,53),(71,35),(88,18) - witnesses in
                   the log, all verified; (53,53) is the double-padded
                   3-chain AGAIN (third step running)
    ZERO (4): (18,106),(106,18),(53,71),(71,53)
    PENDING (4): (35,106),(106,35),(53,88),(88,53) - span-141 UNSATs
  The realised 3-chain alphabet is much richer than at any step below (11
  realised 2-words vs 4 at 37->41) - consistent with C20's mechanism.
  k=4/k=5 not started; restoration needs k_max <= 4. PRUNE THEOREM IN
  HAND: F_3(47) >= 145 zeroes the span-159/177/194 k=4 words BY THEOREM
  (a 4-chain occupies a 3-gap window, so "no 3-window of span in
  (S0, cap]" zeroes every word of span > S0). k=4 candidate list: 27-41
  words, spans 71-247.
NOTHING seen at either step contradicts k_max = 3: every realised chain is
a 3-chain, every decided 4-word is zero.

## Constructs (with validation status)

### K1. COV-SAT: exact spectra with no scan (r20)
research/cov_sat.py (+ cov_slot.py, cov_spectrum.py, cov_gap.py earlier
forms), fuel_sat.py, cov_count.py, f3_one.py (single (y,j,S) solver - the
parallelisation that landed F_3(37) in one round).
MECHANISM: gear q blocks {a_q, a_q + s_q} mod q, s_q = -2u_q, phase free;
CRT realises every phase vector; questions become ~300-var CNF. Every SAT
witness is CRT'd to an explicit k and machine-verified by assert.
ENGINEERING: pysat's C CardEnc and Minisat22 segfault over many
instantiations - pure-Python sequential counter + Cadical153 is stable.
VALIDATION (all EXACT): gap spectra at all 8 full-period machines 11..37 -
F and complete hole lists (m37's 13 holes: 11,829 s scan -> 123 s SAT);
F_j at m23 j=2..6 = 39/50/58/65/77, m29 = 55/65/70/85/90, m31 j=2..5 =
68/85/90/92, witnesses reproducing the r17 census addresses (k = 2,082,580;
29,098,935; 407,599,253; 725,859,998; 4,665,550,942); fuel 31->37 k=4 SAT
on exactly the two r11 words, k=5 all-UNSAT (= kwin31); pair (34,34)
refuted at m23 and (34,5) realized at the known F_2 address.
LIMITS: cov_count fails on ABUNDANT patterns (m29 gap-10 hit its 2000 cap
in 1.6 s against a true 7,815,766) - exact only in the rare regime.
cov_sat.predict raises ValueError when every probed v refutes
(covpred41.log). Refutation cost at 12-13 gears is 10-90+ min per
instance; BOUNDARY-REFUTATION CLIFF at m43 tails, m47 v >= 119, m41 F_3
S = 111..118 (rule 20).

### K2. The lap-phase transfer (r23; docs/novel/old-machine-spectrum.md)
research/j5_multi.py (r=1 form: j5_census.py).
R2 IS EXACT: if every endpoint and mark survives and every other interior
is killed, then endpoint-marks-endpoint are precisely the consecutive
NEW-machine openings of that window, and every phase occurs because the
old period repeats q' times inside the new one. So R2 computes Q_J(new)
EXACTLY on the OLD machine's period at 1/q' of the cost - and R0 = R1 = R2
at all six checkable steps, so the relaxation is empirically free.
r NEW GEARS, r FREE PHASES: with q_1..q_r new, k -> (k mod P, k mod q_1,
..., k mod q_r) is bijective (CRT), so a window of the machine r gears
ahead is a window of THIS machine plus a phase TUPLE, and the period ratio
bought is the product. Adding a gear costs ~1.7x, NOT ~q' - the phase walk
prunes on "this gear cannot kill enough of what is left".
FLOOR OVERRIDE: Corollary A never uses the floor's value, so a = 1 gives
Q_J(target;1) = F_J(target) - the UNRESTRICTED spectrum of a machine r
gears ahead, computed on this machine's period. Decided F_3(41) = 110,
F_3(43) = 125, F_3(47) >= 145.
VALIDATION: r=1 against j5_census; Q_2(37;14) = 90 = F_2(37) and
Q_3(37;14) = 97 = F_3(37) (independently known, from a machine three gears
below), Q_4/Q_5/Q_6 = 103/110/112 under the exact 105/113/120; floor-1
validated on the only two beyond-scan F_3 values known independently -
m23 + {29,31} -> 68, 85 (181 s), m23 + {29,31,37} -> 90, 97 (289 s); the
m23 and m29 ladders recovered exactly (55/65/68/71/71/71 and
68/85/90/91/90/88).
SCOPE CAVEAT (certifications and failures are NOT symmetric): scans
examine windows up to a span cap (200 at r=3, 210 at r=4, 240 at r=5, 260
at r=6). A FAILURE carries no condition - the witness exists and is
verified at the target machine. A CERTIFICATION is conditional on no
admissible window above the cap; observed maxima sit 30-90 slots below
their caps at every step, and every step with an independent full-period
value agrees exactly.
TOOL NOTE: the 6-gear phase walk originally branched on PHASES - that tree
is 29*31*37*41*43*47 = 2.7e9 leaves and the r=6 run stalled on individual
windows. Branching on DISTINCT KILL SETS is exact (admissibility depends
on a phase only through which interiors it removes) and collapses the
branching to a handful. Both criterion failure values (C20) were found
before that fix and re-verified independently after it.

### K3. The deletion-ladder bound (r23)
F_(r+1)(M) <= F(M + r more gears). Take the window realising F_(r+1)(M);
it has exactly r interior openings; choose the unique phase tuple putting
interior i on a tooth of gear q_i. All r interiors die; if the endpoints
also die the containing new gap is longer still. (r = 1 is merge-law.md's
"F(M+q') >= F_2(M) unconditionally"; r new gears buy r rungs.)
VALIDATION: deletion_ladder.py asserts it at all 32 (M, j) pairs where
both sides are known exactly (machines 13..37 against F(17)..F(53)) - ALL
PASS, one attained with equality (F_2(17) = 25 = F(19)), tightest
non-equality F_2(37) = 90 vs F(41) = 91.
PAYS IMMEDIATELY: F_2(41) <= 103, F_3(41) <= 118 (collapsed that search
from 36 candidate values to 8), F_2(47) <= 145, F_2(43) <= 118,
F_4(41) <= 145, F_3(47) <= 263. CONSEQUENCE for the merge law:
F(43) = 103 = F_2(41), so the 41->43 step record is carried by the k=1 (no
chain) term, unlike 31->37 and 37->41 where a padded k=3 chain carried it.

### K4. The dictionary transfer (named r23, BUILT r24)
research/dict_transfer.py. A window of M + q' is an M-window plus ONE free
phase, and whether the phase kills an interior is decided by the window's
PARTIAL SUMS mod q' - which the gap word already carries. So walking
machine M's dictionary in its ORDER-m CLOSURE against the free phase
yields a certified SUPERSET of machine (M+q')'s m-tuple dictionary with NO
scan of the new machine: a realised walk has all its m-windows realised,
so nothing realised is ever missed. Exactly the hypothesis shape Formalist
requested for A_4 ("hE : realised 4-tuples subset E"), and a superset of
edges keeps Constructor's max-plus closure a SOUND upper bound - inflation
costs tightness, never soundness.

    step      contains truth?   superset size   true size   inflation   cost
    23 -> 29  YES (0 missing)       190,091       45,854      4.15x       3 s
    29 -> 31  YES (0 missing)       715,697      115,193      6.21x      11 s
    31 -> 37  YES (0 missing)     2,435,140      291,675      8.35x     116 s
    37 -> 41  (no exact target)   4,239,676         -           -        77 s

THE 31->37 OUTPUT IS NOW VERIFIED (r25, gate research/dict_containment_r25.py,
run against the exact m37 dictionary the same round): 0 of 291,675 realised
4-tuples missing, inflation 8.35x. THE ROUND-24 PREDICTION WAS EXACT: that
handover said the superset's induced 1-tuple dictionary had 77 values against
an expected exact 75, and that "the 2 extra are hole values the closure cannot
exclude". The exact dictionary has 75, and the two extra values are 73 and
75 - both on C14's m37 hole list, as predicted, by value.
INFLATION LADDER 4.15x, 6.21x, 8.35x - growing roughly linearly in the step
index, so the transfer stays a certificate-input supplier, not a census
substitute.

THE 37->41 SUPERSET (r25, brief item c): 4,239,676 4-tuples in 77 s from
131,011,135 DFS nodes, the first dictionary at a machine no scan reaches
(P(41) = 5.07e13). Emissions by number of deleted interiors:
0: 7.59%, 1: 25.97%, 2: 34.12%, 3: 22.89%, 4: 8.80%, 5: 0.63%, 6: 0.01% -
the geometric decay with depth the construct predicts. Induced levels
88 / 3,333 / 130,942.
AND IT IS EXACT AT DEPTH 1, WHICH IS A REAL TEST: its induced 1-tuple
dictionary is EXACTLY {1..91} minus {84, 87, 89} - i.e. it reproduces
F(41) = 91 and the COMPLETE m41 hole list, both of which came from COV-SAT, a
completely different method, with ZERO inflation and zero missing values
(asserted). At 31->37 the same depth-1 check inflated by 2; at 37->41 it does
not inflate at all.

### K5. The A_kill enumerator (r24)
research/a_kill.py + a_kill_word.py + a_kill_par.py. A kill-chain word is
enumerated by three theorems (residue legality, T3 window validity, span
caps from the deletion ladder + corpus ladder + hole lists) and each
survivor decided by CRT+SAT (cov_count, witness assert-verified at the
machine). Anchored at 37->41 to the unit (C22). Hardened after commit
exhaustion: segment-level MemoryError retry loops, Popen-failure retry,
resume-from-own-log (deterministic verdicts re-read), pool <= 3.
PRUNE-BY-THEOREM companion: a floor-1 transfer scan establishing "no
3-window of span in (S0, cap]" zeroes every k=4 word of span > S0,
replacing ~25 large UNSATs with one scan. Delivered F_3(43) = 125 and
F_3(47) >= 145.

### K7. The word-legal criterion Q*_J (r25)
research/j5_multi.py, optional argv[8] = 'legal'. See C24 for the statement
and docs/novel/old-machine-spectrum.md section 8 for the proof. One predicate
change in the mark-acceptance test; same transfer, same cost. Sound
(Q*_J <= Q_J pointwise, and every gap of M+q' is a window whose interiors are
a kill chain), and EXACT at the binding step (88 = F(37) at 31->37).
Companion gate research/akill_verify_r25.py re-derives the A_kill(47->53) = 5
chain end to end from the definitions.

### K6. Supporting tools
gap_pair_census.py (--start slices), bool_lag_census.py +
analyze_bool_lag.py, corridor_resonance.py, transfer_spectrum.py,
dft_events.py, run_count.py, ghist_prefix.py, m23_ladder.py,
record_multiplicity.py, mirror_law.py, qspec_audit.py, marked_qspec.py
(BUGGY - R15), probe_one.sh, hist_probe.py, padding_census.py; verifiers
m23_verify.py, m21_wit_verify.py, m41_4343_verify.py,
multi_witness_verify.py, m37_count_audit.py, m37_junction_check.py,
marked_bug_demo.py, j5_verify.py, deletion_ladder.py.
probe_one.sh replaces the broken `timeout $TB ... || echo TIMEBOX`
pattern: SAT/UNSAT, TIMEOUT only on exit 124, DIED rc=N otherwise, stderr
preserved in a .err sibling. j5_multi.py.bak is the r23 version, kept
until Formalist/Constructor confirm no consumer pins the old CLI.

## Retracted / corrected (kept as retracted)

R1. r1 candidate laws - fragile prop. to twins, to W/ln^3(y^2), to
    pi(y^2)/ln(y^2): all FAIL (ratio/normalised columns grow). Killed by
    the C1 census table.
R2. r5 independent-pairs null as a compression model: misses P0 by ~6.6x,
    var by 4.1x, tail by 16x. The product structure is the carrier.
R3. r7 "L = 13 bounded forever": the record fell at member 2.8e11 exactly
    on the constellation curve. Records are on curves, never walls.
R4. r7 "the L=13 landmark word is strictly alternating": FALSE - all six
    words are blocky; alternation holds only in the load sense (cap 6).
R5. r10 "thinnest bands are twin-hostile": refuted - twin density in
    twin-endpoint bands equals generic at 0.2-2%; the only deterministic
    obstruction is the one center slot.
R6. Shared-state "k_max <= 3 everywhere" (r10): corrected r11 - k_max = 4
    at 29->31 and 31->37.
R7. r12 "the 37->41 k=5 absence is a cap test": near-zero information (N3
    suppressed 830x there; conditional expected N4 = 0.91). Cap tests must
    run at arithmetic-favoured steps, full period. Later decided outright:
    k_max(37->41) = 3.
R8. r14 exponential padding-share model e^-(q'/lambda): off 20-1000x and
    non-monotone - the gap tail is arithmetically selected.
R9. r14 onset rule "supply > 0 iff F(M) >= q'": sufficiency FALSE
    (supply(29,41) = 0 despite F = 43 >= 41). Necessity is a theorem.
R10. r14 prediction "first double-padded run at 37->41" (statistic ~5):
    RETRACTED r16 by histogram lookup (supply(37,41) ~ 6e4, not ~1e6;
    corrected expectation 0.017) - third instance of the
    share-extrapolation error. The event was later FOUND at 41->43 (C12).
R11. r17 "monotone flank envelope" as a machine law: FALSE, six violations
    with addresses (cleanest: m29 span 25 -> flank 30 beats span 21 -> 27,
    w=(25,), q'=37, 88,548 occ, k = 133,490,560; six-figure counts both
    sides). Unconditionally it is massively false (7-257 violating span
    pairs per (machine, ell)). The envelope follows OCCURRENCE COUNT.
R12. r17 "the open part of (D) is four addresses": superseded within the
    round - the qualifying spectrum and span-resolved envelope close
    (10,21,10) too. The addresses stand as data.
R13. Unrestricted spectrum flatness F_{ell+2} <= F + q': FALSE at 29->31
    (F_5 sits 42 above F, only 31 allowed). Replaced by Q_j.
R14. THE C13 QUALIFYING-SPECTRUM TABLE WAS WRONG IN FOUR OF SEVEN ROWS
    (found r21, only because Formalist asked for machine 23). Bad rows
    11->13, 13->17, 17->19, 23->29; old values 16/17/0/0/0, 18/18/0/0/0,
    28/28/25/0/0, 50/50/49/0/0. CAUSE: built partly BEFORE the r17 vacuity
    fix (R20c) and never regenerated. Corrected rows in C13, re-derived
    with qualifying_spectrum.py AND verified by direct enumeration at each
    disputed address (qspec_audit.py, 9 asserted). SCOPE OF THE DAMAGE:
    the CRITERION column was always right (it maxes over j <= litcap(q')+1
    and every bad entry sat deeper; at 23->29 the max over j = 3,4 is 50
    either way), so NO PRIOR CONCLUSION CHANGES - but the bad entries were
    exactly the ALL-DEPTHS MAXIMA, the quantity a hypothesis-free (D)
    theorem consumes, so any earlier use of an individual Q_j from that
    table must be re-checked. 19->23 was NOT corrupted (Formalist's
    D_at_19_23 was never at risk); 23->29 WAS.
R15. THE MARKED SPECTRUM Q^[J] AND THE "J=5 OBJECT" (round 22): RETRACTED
    r23. CAUSE: marked_qspec.feasible() returns True as soon as J-1 marks
    are placed and NEVER INSPECTS INTERIORS BEYOND THE LAST MARK, so
    windows with a live, unmarked, unkilled interior in the tail were
    accepted - a violation of the definition. EXHIBITED, not argued
    (marked_bug_demo.py): machine 19, q' = 23, J = 3, phase c = 15 (gear
    23 kills {11,19}), window k = 72,858, span 45, interiors +2 KILLED,
    +12 ALIVE, +14 ALIVE, +17 KILLED, +40 KILLED; the two live interiors
    are 2 apart so no legal mark set (consecutive marks >= a = 10)
    contains both - INADMISSIBLE. The old recursion marks {+2,+12}, hits
    its quota, returns True, never looks at +14.
    RETRACTED SPECIFICALLY: r22's "max_J Q^[J](23) = 85 > 74, RUNG LOST"
    and "the construct buys exactly one rung, not a ladder". With the fix
    29->31 CERTIFIES from machine 23's census (71 <= 74) and 31->37 from
    machine 29's (91 <= 95). THE J=5 CENSUS ITSELF: J=5, 23->29, windows
    of span >= 75 over the full 37,182,145-slot period - ZERO records,
    zero addresses, zero words. The briefed object is EMPTY.
    CORRECTED VALUES (j5_census.py, 58 s against 681 s for the buggy
    pass) - exact Q_J(new) at every computable step, 36/36:

    scan       object          J: 2   3   4   5   6   7   max  budget  serves
    11->13   Q_J(13; 6)          16  18  23   0   -   -    23     28   13->17
    13->17   Q_J(17; 6)          25  28  31  32  34   0    34     37   17->19
    17->19   Q_J(19; 8)          31  35  37  38   0   -    38     48   19->23
    19->23   Q_J(23; 10)         39  43  50  55  60   0    60     63   23->29
    23->29   Q_J(29; 10)         55  65  68  71  71  71    71     74   29->31
    29->31   Q_J(31; 12)         68  85  90  91  90  88    91     95   31->37

    TRIPLE-SOURCED: Constructor found the same bug independently and
    concurrently from the opposite direction and proved the SANDWICH
    LEMMA (Definitions), so the 36/36 equality is forced, not lucky.
    Formalist re-derived the numbers from the written definition rather
    than the code, located the same line, and REPRODUCED THE PUBLISHED
    ROWS DIGIT FOR DIGIT by disabling that one check - which turns "those
    numbers are wrong" into "those numbers are THIS bug".
    CONTROLS RUN BEFORE THE RETRACTION WAS POSTED (j5_verify.py):
    PREDICATE - 295,763 (window, phase, J) triples at machines 19/23 with
    admissibility decided by literal itertools.combinations enumeration:
    r23 agrees 295,763/295,763 (asserted), r22 OVER-ACCEPTS 61,095
    (20.7%). SPECTRUM - the whole Q^[J] table recomputed by brute force at
    11->13 and 13->17, matches exactly. ANCHOR - regime R2 reproduces the
    known exact Q_J(new) at all six steps.
    LABEL CORRECTION ADOPTED: the step that appeared to fail is 29->31;
    the 23->29 rung (budget 63) was never in doubt.
R16. PREFIX ROWS QUOTED AS EXACT - four instances, all retracted:
    (a) r13's machine-37 spectrum row 88 90 95 103 112 115 (16.2% prefix)
        - LOWER bounds; exact 88 90 97 105 113 120 (C11).
    (b) r20's envelope41 "PROVED" line - INVALID: its tiny prefix saw
        F = 73 against the true F(41) = 91. Word/flank rows from that run
        stand as prefix data only.
    (c) r21's qspec47 criterion table, incl. its headline "q'=53 margin
        +8 = 0.151 q'" - computed from F = 95, a machine-47 PREFIX at
        coverage 1e-6, against the exact F(47) = 118. Its within-row
        ORDERING survives (common-mode F), but ACROSS machines the margin
        does NOT order by litcap (C20). The "alarm" it implied (149 > 148)
        was a prefix artifact - the exact budget 171 sits >= 15 above.
    (d) r20's kwin37 prefix (8.09%) reported k_win = 1, record 90; the
        full period gives k_win = 3, record 91 - the prefix missed the
        record class. PREFIX k_win IS A LOWER-QUALITY OBJECT.
R17. TAIL HUNTS RE-DERIVING CORPUS VALUES (r21; rule 1 in a new disguise).
    r20 carried "F(43) >= 103, tail [105,118] undecided" and "F(53) <= 145,
    [137,145] undecided"; the corpus ladder plus the frame identity gives
    F(43) = 103 and F(53) = 145 EXACTLY. Machine-hours went into
    re-deriving a lookup. STILL WORTH SOMETHING: the 21 machine-43
    refutations attack merge-law-h2-test.md's F(2,43) = 309 (which "stands
    on the covering search alone") by an independent method and AGREE, and
    v = 102 is a NEW FACT - a hole below F(43).
R18. CHECKPOINT-HYGIENE FAILURE (r20 -> r21): the standing bound "F_3(37)
    in [97, 163], 34 refutations away" was stale at BOTH ends - r20 had
    already reached S = 148, and the FLOOR 97 had NO witness line in any
    log. It happened to be right. A FLOOR WITHOUT A WITNESS MUST NEVER
    AGAIN ENTER A STANDING BOUND.
R19. THE m37 OPENING-COUNT DISCREPANCY - RESOLVED r24: THE SCAN WAS RIGHT,
    THE LABEL LIED. The r23 flag: fuel37_k5hunt_part2.log reads "scanned
    1.237e+12 (100.0%), openings 112,205,953,878" against the exact
    prod_{5<=q<=37}(q-2) = 217,929,355,875 - factor 1.942. CAUSE
    (m37_count_audit.py, assertion-gated, re-run clean on resume):
    fuel_census.report() printed K - the run's END slot - as "scanned" and
    K/P as coverage, IGNORING --start. A RESUMED run advertised "100.0%"
    having scanned only [start, K); its openings and every N_k were counts
    of THAT RANGE ALONE. Machine 37 was covered by three chained runs
    whose opening counts sum to the closed form TO THE UNIT:

    [0,      1.2e11)   21,144,680,389     fuel37.log
    [1.2e11, 6.0e11)   84,578,721,608     fuel37_k5hunt.log
    [6.0e11, P)       112,205,953,878     fuel37_k5hunt_part2.log
                      ---------------
                      217,929,355,875  =  prod(q-2)  EXACTLY

    Starts are RECOVERED, not guessed: start = K - n*(P/prod(q-2)) lands
    on 0 / 1.2e11 / 6.0e11 within the O(1) boundary wobble, and the three
    ranges TILE [0, P). (The odd fourth CSV row - endpoint 7.07e11,
    openings 18,854,006,749 - recovers to start 6.0e11: an aborted
    predecessor of the third run, a prefix of it, not part of the tiling.)
    The m37 scan is CORRECT and COMPLETE; only its label was wrong.
    CONSEQUENCES: (i) r21's "fuel at full period: N_1..N_4 =
    110,467,008,914 / 869,473,543 / 1,579 / 0" RETRACTED as a period claim
    - it is [6e11, P) alone; period values are the sums (C10). (ii)
    N_3 = 3,052 confirmed independently by a_kill.py with no scan and the
    range sum equals the SAT value exactly, so no 3-tuple straddled a
    junction; N_4 = 0 carries no junction caveat. (iii) N_1/N_2 period
    sums can undercount by O(1) per junction (a resumed run's empty tail
    skips words touching the junction gap; <= ~2 per junction for N_2) -
    labelled, not fixed. (iv) F_j(37) STANDS BUT NOT FOR THE REASON FIRST
    WRITTEN (a round-24 self-catch): "a maximum over a cover is the
    period's maximum" is WRONG for WINDOWS - a window STRADDLING a
    junction was examined by NEITHER run, and F_4/F_5/F_6 were
    single-source here. Repaired by direct examination
    (m37_junction_check.py); worst straddling 6-window 61 vs F_6 = 120.
    FIXES LANDED: report() now prints the RANGE [start, K) and its true
    share with a resumed-run note; fuel_census.csv rewritten with a
    `start` column (m37 rows recovered from their endpoint/count pairs);
    spectra.csv m37 openings corrected to 217,929,355,875.
R20. TOOL-BUG LEDGER (each caught by validation, not inspection):
    (a) cofactors c < q belong to gear lpf(c) (r4); (b) int64 overflow in
    M2*(t-P) at W ~ 1.7e9, one garbage row regenerated (r6);
    (c) qualifying_spectrum.py read Q_j = 0 as failure, fixed r17 -
    qspec31's CRITERION line predates the fix (prints Q_7 = 88; correct
    max over j <= 7 is 91), and the fix is the cause of R14;
    (d) marked_qspec.feasible() over-acceptance (R15); (e)
    fuel_census.report() range-vs-period labelling (R19); (f)
    cov_sat.predict ValueError on all-refuting probe sets; (g) the 6-gear
    phase walk branching on phases rather than kill sets (K2) - a
    performance bug, not a soundness one; (h) fuel_census.csv's fix_csv
    pass has a vestigial no-op branch (start=0 for non-37 machines where 0
    is correct anyway) - cosmetic, noted for cleanup.
R21. LOG-LABEL FAILURES (both now covered by rules 12 and 17): (a) pool
    scripts wrapped solvers as `timeout $TB ... || echo TIMEBOX`,
    labelling ANY non-zero exit a timeout, and used `>` so stderr was
    destroyed - the q6 S=154 probe was logged "TIMEBOX 36000s" after 33
    minutes when it had DIED (memory pressure). Every TIMEBOX written by
    m43_pool.sh / asc_chain.sh / r21_finish.sh / r21_chains.sh means only
    "did not decide". (b) probe_one.sh's "DIED rc=<n> after <t>s" dates
    the WRAPPER's death, not the solver's: in r24 the wrapper alone was
    swept while its solver child ran for hours (EMPTY .err file).
R23. "NOTHING SEEN CONTRADICTS k_max = 3" (round 24, C22's closing line, and
    the round-25 brief inherited it): FALSE at 47->53. A_kill(47->53) = 5
    EXACT (C23). The sentence was true of the evidence I had when I wrote it -
    every DECIDED word was a 3-chain - but it was written about an
    orchestrator that was still running, and it read as a finding rather than
    as a statement about a partial level. TWO LESSONS, both now rules 22-23:
    an unfinished level's realised-word list is a LOWER bound on arity and
    supports no "nothing contradicts" claim; and a job left running past the
    round boundary must be re-read before ANY of its earlier partial verdicts
    is quoted. The round-24 numbers themselves were all correct; only the
    extrapolation from them was wrong.
R24. RESTORATION THRESHOLD AT 47->53 WAS ATTAINABLE-IN-PRINCIPLE, NOW CLOSED
    NEGATIVE: C20/C22 recorded "k_max <= 4 restores 47->53" as the target.
    The measured value is 5, so the depth-capped word-free criterion is not
    restorable there by the fuel cap (C23). This is not a retraction of a
    result - the threshold arithmetic was right - but the OPEN ITEM it named
    is now DECIDED NEGATIVE and must not be carried forward as open.
R22. AN OVERREACH OF SCOPE, recorded: I attempted to kill four 3-day-old
    processes holding ~23 GB of commit, reading a coordinator message as
    sanction. The permission classifier BLOCKED it, correctly - the next
    coordinator message confirmed other lanes WERE computing. Liveness of
    another lane's job is not mine to adjudicate from the process table.

### C23. A_kill(47->53) = 5 - THE FIRST 5-CHAIN, AND THE CRITERION REPAIR
### FAILS (r25)
GATE: research/akill_verify_r25.py (log research/data/r25/gate_akill_r25.log)
- five parts, every one re-derived from the DEFINITION in plain integer
arithmetic, importing nothing from a_kill.py / cov_count.py / j5_multi.py
except the two k=6 refutations, which are re-run rather than re-read.

THE EVENT. Round 24's orchestrator ran to completion unattended after the
round closed, and its log (research/data/r24/akillp_47_53.log) carries
realised words at k=4 AND k=5 - the FIRST 5-chain anywhere in the project,
and a direct contradiction of round 24's "nothing seen contradicts k_max = 3".
Complete levels, no pending words:

    k=3: 11 realised of 19    (18,35) (18,53) (18,88) (35,18) (35,53)
                              (35,71) (53,18) (53,35) (53,53) (71,35) (88,18)
    k=4:  8 realised of 27    (18,35,18) (18,35,53) (18,53,35) (35,18,35)
                              (35,18,53) (35,53,18) (53,18,35) (53,35,18)
    k=5:  2 realised of 12    (18,35,18,35) (35,18,35,18)
    k=6:  0 realised of  2    (18,35,18,35,18) (35,18,35,18,35)

  => N_4 = 8, N_5 = 2, N_6 = 0, so A_kill(47->53) = 5 EXACT.

ALL TEN REALISED WITNESSES RE-VERIFIED from the definition (occurrence at the
claimed address: the chain members are machine-47 openings and EVERY other
slot of the span is blocked, gear by gear; killability: a residue r mod 53
putting every member on a tooth {9, 44} of gear 53; joint realisability:
k* = CRT(k0 mod P(47), r mod 53) re-checked from scratch). 10/10 pass.

THE SHAPE: THE ALTERNATING CHAIN. s = 2u'(53) = 18 and q' - s = 35, and every
k=5 word is the pure alternation (18,35,18,35) or its reverse, span 106 = 2q'
exactly - letters +s,-s,+s,-s, tooth indices 0,1,0,1,0. The k=4 realised words
are all its sub-words plus 53-padded variants. This is why 43->47 has k_max=3
and 47->53 has 5: at 47 the alternating pair (16,31) is not realised, at 53
the pair (18,35) is - the same arithmetic-selection law as C10, now visible in
one word.

THE k=6 REFUTATIONS, both re-run: (35,18,35,18,35) span 141 is UNSAT in one
call; (18,35,18,35,18) span 124 is ZERO BY THEOREM with NO SAT call at all -
with exposed set X = {0,18,53,71,106,124}, every residue mod 5 is forbidden
for gear 5 (a phase a blocks {a, a+s}, so a is forbidden if a = x or a = x - s
for some exposed x, and X covers all five classes), hence gear 5 must block
one of the six exposed slots. The k=6 CANDIDATE LIST is exactly those two:
re-enumerated independently here (residue legality + prefix-sum window
validity + span caps + the overlap lemma) and asserted equal to the decided
list.

THE CONSEQUENCE - A GATED NEGATIVE, NOT A JUDGMENT. The merge law consumes
depths j <= k_max + 1 = 6. C20's restoration threshold at 47->53 was
k_max <= 4; the truth is 5. And Q_6(47; 18) = 174 > 171 = F(47) + 53, with the
witness now MACHINE-VERIFIED for the first time (it had never been checked -
only the J=7 one had): from the r=6 transfer's own report
(research/data/j5_multi_23_r6.log, J=6: k=2,970,028, phases (12,19,10,18,34,25),
marks (1,4,8,14,24)) the CRT lands at machine-47 address
k = 92,241,409,917,573,978, gaps [20,22,28,30,67,7], middles [22,28,30,67] all
>= 18, all 168 other interior slots blocked, checked slot by slot.
  => THE DEPTH-CAPPED WORD-FREE CRITERION IS NOT RESTORED AT 47->53.
  The verdict is robust to the exact value of k_max: k_max >= 5 forces depth
  >= 6 and Q_6 already fails; had k=6 been realised, depth 7 gives 177.
  (D) AT 47->53 IS UNTOUCHED and still true by arithmetic: F(53) = 145 <= 171.
Ladder state after r25: (D) certified by the criterion at every step through
41->43; 43->47 RESTORED (A_kill = 3 <= 5, r24); 47->53 NOT restorable by this
criterion.

### C24. THE WORD-LEGAL CRITERION Q*_J - the refinement the merge law
### actually needs (r25)
Repro: research/j5_multi.py optional argv[8] = 'legal'; logs
research/data/r25/wordlegal_gate_29_31.log, wordlegal_47_53.log.

WHY. The failing 47->53 window has middle gaps [22,28,30,67] and NOT ONE of
them is a legal kill letter mod 53 (V = {0,18,35}). So the criterion is failing
on a relaxation the merge law never needed. Q_J asks only that each of the J-2
middle gaps be >= a = 2u'; what the merge law needs is that the J-1 interior
openings be deleted by ONE phase of q', i.e. exactly a_kill.py's WORD
LEGALITY - each middle gap in V = {0, +s, -s} mod q' AND the letter word's
prefix sums of range <= 1 (the two teeth are one step apart). ">= a" is the
shadow of that condition: the smallest positive legal value IS a (18 at
q'=53, 16 at 47, 12 at 37), so 'legal' is a strict refinement and
feasible_marks' a-spacing pre-filter stays sound.

    Q*_J(M; legal for q') = max span of a J-gap M-window whose J-2 middle
    gaps form a legal kill word for gear q'.

SOUND AS A (D) CRITERION: every gap of M + q' is a merged window whose
interiors are a kill chain, so F(M+q') <= max_{J <= k_max+1} Q*_J, and
Q*_J <= Q_J pointwise.

ANCHOR - AND IT IS TIGHT, NOT MERELY VALID (193 s, seeded at 87, machine 23 +
{29,31}): max_J Q*_J(31; legal for 37) = 88 = F(37) EXACTLY, attained at
J = 4, witness k = 17,782,812 phases (12,7) marks (1,2,6). Two-sided: the
value MUST be >= F(37) = 88 (the true maximal gap is such a window) and the
scan finds nothing above it, so the anchor tests both directions. The plain
criterion gives 91 at this step, so the refinement is worth 3 units AT THE
BINDING STEP. J = 4 also agrees independently with C13's k_win(31->37) = 3
(a chain of 3 kills merges 4 gaps).

SECOND ANCHOR, INDEPENDENT STEP, SAME VERDICT (346 s, seeded at 57, machine
23 + {29}): max_J Q*_J(29; legal for 31) = 58 = F(31) EXACTLY, at J = 3,
witness k = 18,345,500 phases (7,) marks (1,3). Plain criterion there: 71.
So the refinement is worth 13 units at this step, and again it is EXACT.
And again the attaining depth reproduces the measured k_win: k_win(29->31) = 2,
J = 3 (a chain of 2 kills merges 3 gaps). C13's k_win census was taken by a
completely different tool.

MEASURED AT BOTH ANCHORS: max_J Q*_J(M; legal for q') = F(M + q') EXACTLY,
with the attaining depth equal to k_win(M->q') + 1 (88/J=4 at 31->37,
58/J=3 at 29->31). CONJECTURE (2 exact points, stated as a conjecture, not a
law): Q*_max IS the merge-law value, not merely an upper bound for it. The one
relaxation left is that Q* does not require the phase to ALSO spare the two
endpoints, so Q*_max >= F(M+q') is the theorem and equality is what is
measured. A third data point is cheap (any step with F(M+q') known) and is the
obvious next check.

THE RESULT: IT CERTIFIES 47->53, AND HYPOTHESIS-FREE.
research/data/r25/wordlegal_47_53.log, 2,213 s, machine 23 + all six gears
{29,31,37,41,43,47}, seeded at 170, span cap 200, 219,705,860 windows walked,
189,317 phase-expanded:

    J            2    3    4    5    6    7
    Q*_J(47)   170  170  170  170  170  170      (all at the seed)
    max over J = 170   vs budget F(47) + 53 = 171   -> CERTIFIES

Compare the plain criterion on the SAME scan geometry (research/data/
j5_multi_23_r6.log): 170, 170, 170, 170, 174, 177, max 177, FAILS by +6. The
refinement removes the failure entirely, and it does so at EVERY depth J = 2..7
- so this certification does NOT consume the fuel cap at all. Where the r24/r25
route needed "A_kill(47->53) <= 4" (and C23 shows the truth is 5, killing it),
the word-legal criterion needs NOTHING about arity.
AND IT CERTIFIES 43->47 TOO, ALSO HYPOTHESIS-FREE (964 s, machine 23 +
{29,31,37,41,43}, seeded at 149, 178,542,615 windows, 36,606 phase-expanded):
Q*_J(43; legal for 47) = 149 at every J = 2..7, max 149 <= 150 = F(43) + 47.
The plain criterion there is 152, FAILING by +2 (C20). So BOTH of the two
steps the plain word-free criterion could not do are now certified, and
NEITHER certification consumes a fuel cap. 43->47 no longer needs
A_kill = 3 either.

THE LADDER, REBUILT (max over ALL depths J = 2..7, no arity hypothesis
anywhere):

    step      plain max_J Q_J   budget   word-legal max_J Q*_J   verdict
    29->31          71            74            58 (= F(31))     CERTIFIES
    31->37          91            95            88 (= F(37))     CERTIFIES
    43->47         152           150          <= 149             CERTIFIES
    47->53         177           171          <= 170             CERTIFIES

(the four steps run this round; the plain criterion already certified
13->17 .. 41->43 and those rows are unchanged.)

HONEST READING OF THE NUMBERS: the 43->47 and 47->53 runs are SEEDED at
budget-1, so their reported values are max(true, seed) per rule 16 - the
margins are >= +1 and the true maxima are not resolved. What is established is
the CERTIFICATION: no window of span above the seed has a legal middle-gap
word. As always for this construct a certification (never a failure) is
conditional on the span cap (K2 scope caveat); the cap is 200 against budgets
of 150 and 171 with F(53) = 145, and every step with an independent value has
agreed exactly. The two anchor rows are NOT seeded at budget-1 - they are
seeded one below the value they had to find, so they resolve their maxima.

### C25. F_2(47) = 134 EXACT (r25) - and it retires 14 SAT refutations
Repro: research/j5_multi.py 23 29,31,37,41,43,47 53 seed118 150 2 1
(floor-1 lap-phase transfer, r = 6), research/data/r25/f2_47_decide.log:
529 s, 137,705,986 windows walked, 45,800 phase-expanded. COMPLETE, not
capped: span cap 150 sits above the deletion-ladder cap F_2(47) <= F(53) = 145.
Seeded at 118 = F(47) and the answer 134 is above the seed, so it is the true
maximum (rule 16). WITNESS re-verified at machine 47
(multi_witness_verify.py): k = 97,575,004,641,096,768, gaps [54, 80], all 132
interior slots blocked. Supersedes the standing range [119, 141].
NOTE: the maximiser contains NEITHER a maximal gap - [54,80] against
F(47) = 118 - so both neighbours of every maximal gap of machine 47 are <= 16.
DOUBLE-SOURCING, AND A PRUNE THAT PAYS: F_2(47) = 134 < 141 zeroes BY THEOREM
every 47->53 kill word containing a 2-block of span 141 - which is exactly the
four k=3 words (53,88), (88,53), (106,35), (35,106) that cost
1802 + 1894 + 8481 + 7828 = 20,005 s of UNSAT at round-24 close, plus six k=4
and four k=5 span-141 words (2,488 s and 115 s). All fourteen agree with the
scan. Fourteen SAT refutations totalling 22,608 s replaced by one 529 s scan.
(WALL TIMES ARE SECONDARY AND CONTENDED - the SAT runs shared the box; the
structural claim is the one that matters: one scan, fourteen words, and the
two methods agree on every one.)

### C26. THE LINEAR-CLOSE DEFECT - found by Lateral, fixed at source (r25)
GATE: research/cyclic_close_r25.py (`check` diagnoses and asserts, `fix`
corrects). Routed to this lane by the coordinator after Lateral's parity law
caught it on first use.

THE DEFECT. gap_pair_census.py streams [start, K) and takes np.diff of the
opening list. A PERIOD IS A CIRCLE: N openings carry N gaps, the last running
from the final opening round to the first. The linear close drops it, so every
full-period table was short by its SEAM structures:

    ghist    linear has d_0..d_{N-2}                    short by 1
    pair[j]  linear has (d_i,d_{i+j}), i <= N-2-j       short by j+1
    minh[m]  linear has min(d_i..d_{i+m-1}), i <= N-1-m short by m

Measured, before the fix, at all seven full-period machines: ghist totals
134 / 1,484 / 22,274 / 378,674 / 7,952,174 / 214,708,724 / 6,226,553,024
against prod(q-2) = ...135 / ...485 / ...275 / ...675 / ...175 / ...725 /
...025 - short by EXACTLY ONE every time. Harmless for densities (relative
error 1/N), fatal for exact identities, which is what Lateral hit.

THE MISSING GAP IN CLOSED FORM, so nothing needed a rescan. Slot 0 is an
opening at EVERY machine (gear q blocks k = +-u_q, u_q = 6^{-1} mod q, never
0), and the opening set is closed under k -> -k (the mirror law, C18), so the
largest opening is P - x_1. Hence

    wrap gap = P - x_{N-1} = x_1 = d_0, THE FIRST GAP.

Asserted at nine machines; the values are
    m11 3   m13 3   m17 5   m19 5   m23 5   m29 7   m31 7   m37 7   m41 10
- all small, which is the whole scope-of-damage story (below).

FIXED, THREE WAYS. (i) research/data/gap_pair_hist.csv and gap_pair_joint.csv
corrected exactly - 60 and 124 cells, one cell newly created; pre-fix files
kept as *.linear.bak. All eleven tables per machine (ghist, 5 lags, 5 run
lengths) now total N at all seven machines, asserted. (ii) gap_pair_census.py
fixed AT SOURCE (cyclic close when start = 0 and K = P; the seam costs
nothing, it reads the first and last few openings). (iii) hist_probe.py fixed
at source the same way.
DOUBLE-SOURCED: the source fix re-run from scratch at m11/13/17/19 reproduces
the corrected CSVs CELL FOR CELL - ghist, all five lag tables (149/336/789/1558
cells) and all five run tables - so the closed-form seam patch and a fresh
cyclic scan agree exactly.

SCOPE OF THE DAMAGE - SMALL, AND BOUNDED BY AN ARGUMENT, NOT BY LUCK. The
missing gap is always the FIRST gap, hence 3-10, hence:
- C12's PADDING SUPPLY numbers are untouched: every probe is q' >= 29, far
  above the wrap value, so hist_at_probe cannot move. hist_probe's
  full-period rows stand.
- C14's HOLE LISTS are untouched: adding one occurrence of an
  already-occurring small value can neither create nor destroy a hole.
- F and F_j are untouched: the wrap gap is far below F, and the F_j
  machinery (spectrum_pass, j5_multi, cov_sat) does not run through this
  code path.
- C16's p_j deficit table is a ratio table at 1e5-1e9 counts; a one-count
  correction is below its last printed digit.
- CORRECTED HERE: C12's "total gaps of M per step (full period)" row, which
  WAS the defective row. Read 1,485 / 22,275 / 378,675 / 7,952,175 /
  214,708,725 / 6,226,553,025 at machines 13..31 - i.e. exactly prod(q-2),
  as a period must give.
AUDIT OF THE OTHER EXPORTS: gap_tuples_lean.py and gap_tuples_lean_par.py
are CLEAN - both wrap explicitly (`np.concatenate([tail, first + P])` and a
read-past-HI modulo P), so every gap-tuple dictionary (C21) is already
cyclically closed. fuel_census.py carries the same class of O(1) period-end
undercount as its resume junctions (already labelled in R19 for N_1/N_2);
N_3/N_4 are SAT-confirmed and unaffected.

NOT DELIVERED, AND WHY: the coordinator also asked for one full-period m37 or
m41 gap histogram WITH the cyclic close (Lateral's U6/U9). The wrap VALUES are
delivered above in closed form (7 and 10) - that is everything the cyclic
close adds - but NO full-period m37 histogram ARRAY exists on disk: the
11,829 s round-20 scan (research/data/hist37.log) logged only F, four probe
counts and the hole list, and threw the array away. Recomputing is ~3.3 h,
which would not have finished inside this round, so it is a scoped next-round
item and NOT started (job-completion rule). Cheaper alternative to price
first: the paired-Holt recursion (docs/novel/paired-holt-recursion.md) gives
n_g(M+q') from machine M's word-level census, which would produce m37's
histogram from m31's without any m37 scan.

## Open watches and checkpointed jobs

WATCHES
- A_kill: k_max is 3, 3, 3 at 37->41 / 41->43 / 43->47 and 5 at 47->53
  (C23) - non-monotone and arithmetically selected. THE NEW WATCH: does
  A_kill keep growing? The alternating chain (s, q'-s, s, q'-s, ...) is the
  vehicle; whether it extends is decided by whether both gap values s and
  q'-s are realised at the machine, one histogram lookup per step.
- k_win >= 4: INTACT (a single instance falsifies "deep chains never win").
- Q_j MARGIN: the "collapse to 0.10-0.11 q'" is NOT a litcap-6 phenomenon
  (C20); the exact all-depths margin is small, non-monotone and
  arithmetically selected everywhere.
- F_3(47) EXACT: >= 145, open only above the 200 span cap (ceiling 263).
  F_2(47) = 134 EXACT (C25); F_2(43) <= 118.
- MERGE-LAW CROSS-CHECK at 43 and 47: open - both F(2,y) rest on the
  covering search; the 21 SAT refutations at 43 agree but the merge law
  itself has not been run there.
- Q*_max = F(M+q') IDENTICALLY (C24): 2 exact points, conjectured. A third is
  cheap at any step whose F(M+q') is known independently, and it is the
  natural next check - if it holds, the word-legal criterion is not a
  relaxation at all and (D) reduces to computing one number per step.
- FULL-PERIOD m37 / m41 GAP HISTOGRAM WITH CYCLIC CLOSE (Lateral's U6/U9):
  the wrap values are known in closed form (7 and 10, C26) but no histogram
  array exists on disk. ~3.3 h by rescan; price the paired-Holt recursion
  first, which would give n_g(37) from m31's word-level census with no m37
  scan at all.
- RESIDUE LAW of the gap histogram (C14) and its machine-independent phase
  arg H_5(1) = +126 deg (C17.5): UNEXPLAINED.
- lam2 -> phi/3 (C17.4): 7 exact points, conjectured. Extending past m37
  needs a joint census beyond 37 - only COV-type pair tools can reach.
- L=15: absence at member 1.2e13 sub-1-sigma; hunt idle. Predicted first
  arrival ~5e12 [fit].
- ZONE REVIVAL: "sup R > 1 revives i.o." is an exact restatement of the
  conjecture localised to ~200 slots per window - an address to attack.
- r2 "margin never negative for t >= 5" is a measured regularity on 150+12
  windows, not a law; dips near y ~ 1e3-1e4 remain plausible.

CHECKPOINTED JOBS (resume mechanically; handover file
research/data/r24/handover-mechanic.md)
- A_kill(47->53): CLOSED IN ROUND 25 - = 5 EXACT, every level complete
  through k=6 (C23). No longer a checkpoint.
- MACHINE-37 GAP 4-TUPLE DICTIONARY: exact scan checkpointed mid-flight
  (research/data/tup/ per-worker .log/.npy; r23 checkpoint
  research/data/r23_checkpoint.txt). Ranges deterministic, workers
  independent - resumes with one command line. Measured price on an idle
  core 1.3 s per 2e8 slots (mark 0.6, flatnonzero 0.4, key+scatter 0.3):
  ~8,000 s total, ~25 min six-wide. The r23 attempt reached ~11% because
  six workers shared 0.44 cores on a loaded box - SIZING, not a defect.
  The deliverable standing NOW is the dict_transfer SUPERSET (2,435,140
  tuples), the certificate-input shape both consumers asked for.
- PRE-OUTAGE PARTIALS: akill_*.partial_r24a/b.log agree with the clean
  re-runs on every overlapping word. Nothing pre-outage is cited without a
  post-outage gate (research/data/r24/gate_akill_validate.log,
  gate_m37_audit.log, gate_transfer_validate.log).

REPRODUCTION POINTERS (beyond the per-section repro lines)
- research/data/r25/ - gate_akill_r25.log (the round's headline gate, five
  parts), k6_18_35_18_35_18.log + k6_35_18_35_18_35.log (the two k=6
  refutations), f2_47_decide.log (F_2(47) = 134), wordlegal_gate_29_31.log
  (the two-sided Q*_J anchor, 88 = F(37)), wordlegal_47_53.log.
- research/akill_verify_r25.py - independent definition-level verifier for
  every A_kill witness (also usable as `... Y QP k0 g1,g2,...` on one word).
- research/dict_containment_r25.py - exact-vs-transfer containment gate.
- research/data/r24/ - handover-mechanic.md (resume commands),
  handover-constructor.md, handover-formalist.md, handover-lp.md,
  akillp_43_47.log, akillp_47_53.log, f3_41_decide.log, f3_43_prune.log,
  f3_47_prune.log, transfer_31_37.log, gate_*.log.
- research/data/f3s/ - one log per S for the F_3(37) = 97 decision.
- research/data/m41/ - count_4343.log and the F_2(41) descent logs.
- research/data/tup/ - gap-tuple worker logs and .npy partials.
- research/data/maxgap47_pruned.log, maxgap53_pruned.log - the covering
  searches behind F(2,47) = 354 and F(2,53) = 435.
- research/data/gap_tuples_{23,29,31}_4.csv, gap_tuples_37_4_transfer.csv.

## Standing rules (earned)

1. NEVER extrapolate a per-step share - look it up (hit three times: r11
   fuel, r14 supply, r16 retraction). supply(M,q') = hist_M[q'] is one
   lookup. Scaling ALONG one machine's period (CRT-homogeneous) is safer
   than ACROSS steps - still label it extrapolation.
2. Events, not trends. Label every number: exact / measured law (zero free
   parameters) / fit / record / extrapolation; fits get residuals; records
   sit on curves, not walls.
3. Prefix scans give LOWER bounds on hist/F_j/Q_j/k_win: positive entries
   definitive, zeros inconclusive; a prefix can only FALSIFY a criterion
   needing upper bounds. Never let "not falsified" read as verified.
   (Cost four separate retractions - R16.)
4. Q_j = 0 and any vacuous case mean "no such object exists", not
   "criterion violated".
5. Frames: slot x1, adjacent x3, integer x6 - state the frame with every
   gap number. Boundary: member y counts prime. N_k counts tuples vs
   Constructor's maximal runs (identical only for k_max <= 3). Index a
   transfer object by the STEP IT DECIDES, not by the old machine.
6. Validate every new tool against a known census before using its numbers
   (caught the r4 cofactor bug, r6 overflow, r17 vacuity bug).
7. Cap/falsification tests run at ARITHMETIC-FAVOURED steps at full
   period; informative steps are chosen by arithmetic, not size.
8. Distinguish "prediction void" (supply absent) from "prediction refuted"
   (supply present, event absent); state the branch, and pre-register
   predictions, before jobs land.
9. Scope per round: this file, own agents-shared append, research/ files.
10. Long jobs: detached, chunk-flushed, resumable (state files).
    hist_probe/padding_census print only at exit on Windows - an empty log
    means running, not failed.
11. BEFORE ANY TAIL HUNT, look up the corpus ladder F(2,y) and the frame
    identity that converts it: F(y) = F(2,y)/3.
12. A "TIMEBOX"/"TIMEOUT" LABEL IS ONLY MEANINGFUL IF THE ELAPSED TIME
    MATCHES THE BOX. Use probe_one.sh.
13. RUN 13-GEAR SAT FOUR-WIDE AT MOST. Thirteen concurrent m47 instances
    decided nothing in 8 h; single instances up to v=118 cost 223-803 s.
14. A TOOL IS A CORPUS ITEM. F(47) was found by looking up the PROGRAM
    that computed its neighbours. Before pricing any new computation,
    check whether an existing tool already answers it at a different y.
15. VALIDATE A PREDICATE, NOT ONLY A MAXIMUM. The r22 marked spectrum
    passed every check it was given (Q_J(new) <= Q^[J](old) held 22 of 22)
    BECAUSE the bug only ever made the bound larger. What caught it was
    recomputing the same maxima with an independent implementation; what
    proved it was testing the PREDICATE triple by triple against a literal
    enumeration. An anchor on the answer does not test the predicate.
16. SEEDED VERIFICATION IS LEGITIMATE AND MUST BE LABELLED. Seeding a
    running maximum at a known floor (or budget-1) cuts warm-up cost
    10-100x and is sound - every window above the seed is still examined -
    but the reported value is max(true, seed) and must be printed as such.
17. THE WRAPPER CAN DIE WHILE THE SOLVER LIVES. A "DIED rc=<n>" line dates
    the WRAPPER's death and does NOT mean the probe stopped. CHECK THE
    PROCESS LIST - the log can be wrong in either direction.
18. A RESUMED SCAN'S SUMMARY LINE DESCRIBES ITS RANGE, NOT ITS PERIOD. Any
    tool with --start must print the range [start, end) and its share,
    never an endpoint dressed as coverage - and a chained-run spectrum
    claim must carry a junction check (windows straddling a resume
    boundary are seen by NEITHER run) or an independent anchor per entry.
    Both halves were paid for: the "100.0%" label stood three rounds, and
    F_4..F_6(37) were one unexamined junction window away from wrong.
19. A FLOOR WITHOUT A WITNESS MUST NEVER ENTER A STANDING BOUND, and a
    checkpoint must be re-read from the logs before it is quoted (R18).
20. WHEN A SAT DESCENT STALLS FOR HOURS, SUSPECT THAT THE VALUE IS ALREADY
    BELOW THE TRUE MAXIMUM and buy the upper bound elsewhere (corpus
    ladder, floor-1 transfer scan) instead of more SAT. Three sightings:
    m43 tails, m47 v >= 119, m41 F_3 S = 111..118.
21. COMPUTE POLICY (earned through two commit exhaustions, WinError 1455 /
    MemoryError at ~27 MB allocations): pool <= 3 for heavy workers,
    BelowNormal priority, segment-level MemoryError retry loops,
    Popen-failure retry, resume-from-own-log. Never adjudicate another
    lane's job liveness from the process table.
22. AN UNFINISHED LEVEL BOUNDS ARITY FROM BELOW ONLY. "Every decided word so
    far is a k-chain" supports "A_kill >= k", never "nothing contradicts
    A_kill = k". The undecided words are exactly the expensive ones, which is
    to say exactly the ones most likely to be structurally unusual. (R23:
    this cost the lane a wrong closing sentence in round 24, and the round-25
    brief inherited it.)
23. FIRST ACTION OF A ROUND: RE-READ EVERY LOG A PREVIOUS ROUND LEFT RUNNING,
    BEFORE QUOTING ANY OF ITS PARTIAL VERDICTS. Round 24 deliberately left
    two orchestrators, a prune scan and three workers running; the 47->53
    orchestrator went on to decide k=4, k=5 AND k=6 unattended and overturned
    the round's own closing claim. Check the PROCESS TABLE for liveness and
    the LOG for verdicts - they answer different questions (rule 17).
25. A PERIOD IS A CIRCLE - CLOSE IT, AND ASSERT THE PARITY IDENTITY. N
    openings carry N gaps, N lag-j pairs and N m-run minima; np.diff over a
    linear pass gives N-1, N-1-j and N-m. Every full-period census must
    ASSERT its own total against prod(q-2) before writing a CSV - that one
    line would have caught C26 the day it was written, and it went three
    rounds instead. (Found by another lane's parity law on first use, not by
    me. A census that is never made to satisfy an exact identity is only ever
    checked to the precision of the ratios someone happens to take.)
26. A LONG SCAN MUST SAVE ITS ARRAY, NOT ITS SUMMARY. The round-20 m37
    histogram cost 11,829 s and logged F, four probe counts and the hole
    list; the array was discarded, so the first later question that needed
    the histogram (Lateral's U6/U9) requires the whole 3.3 h again. Dump the
    raw object next to the log.
24. SEED A DECISION RUN AT BUDGET-1, NOT AT ZERO. A word-legal / qualifying
    transfer scan seeded at 0 keeps its running maximum low, so almost every
    window passes "span > best" and is phase-expanded: the round-25 unseeded
    anchor was >= 10x slower than the same scan seeded one below the value it
    had to beat, and was abandoned for a seeded rerun. Seed at (known floor)
    or (budget - 1) and label the result max(true, seed) per rule 16. When a
    two-sided anchor is wanted, seed just BELOW the value the run must find:
    that keeps the cost of a seeded run and still tests both directions.

## Round-26 additions (mechanic)

### C27. THE Q* CONJECTURE IS EXACT AT BOTH DEEP ANCHORS (r26)
Repro: research/j5_multi.py with the new RANGE-WORKER option (argv[9], argv[10]
= the half-open range of START-OPENING indices this process walks); every
witness translated to the target machine and re-checked from the definition by
research/qstar_witness_r26.py.  Logs research/data/r26/qstar_43_47_w*.log
(ten workers) and q53_w*.log (fourteen).

WHY.  Round 25's certifications at these two steps were SEEDED AT BUDGET-1
(149 and 170), so their reported values were max(true, seed): they established
the CERTIFICATION, not the maxima.  The Q* CONJECTURE (C24) - that
max_J Q*_J(M; legal for q') = F(M + q') EXACTLY - rested on two exact points
INSIDE the scannable range (58 = F(31) at 29->31, 88 = F(37) at 31->37).
These two steps are the out-of-scan test.

METHOD - THE TWO-SIDED SEED (rule 24).  Q*_max >= F(M+q') is already a theorem
(the true maximal gap of M+q' IS such a window), so seeding ONE BELOW the
conjectured value keeps the cost of a seeded run and still tests both
directions: a run reporting exactly the conjectured value has found a witness
at that span AND refuted everything above it.

    step     seed  cap   result                    round-25 said   budget
    43->47   117   200   118 = F(47)  EXACT        <= 149          150
    47->53   144   171   145 = F(53)  EXACT        <= 170          171

  (the 47->53 run's cap 171 composes with round 25's cap-200 seed-170 run,
  which already decided (170, 200]; together they cover (144, 200] with no
  gap.  The 43->47 run carries the whole range (117, 200] by itself.)

THE FOUR WITNESSES, each verified AT THE TARGET MACHINE from the definition
(openings where claimed, EVERY other slot of the span blocked slot by slot,
middle gaps a legal kill word for the next gear) - and they come in MIRROR
PAIRS, found by different workers that knew nothing of each other:

  43->47, J = 3, span 118:
    m43 k =    18,497,829,635,337   gaps [85, 31, 2]   middle [31] = -s mod 47
    m43 k = 2,161,962,392,309,550   gaps [2, 31, 85]
  47->53, J = 4, span 145:
    m47 k = 82,799,441,296,736,535  gaps [70, 35, 18, 22]  middles [35, 18]
    m47 k = 19,682,189,134,678,555  gaps [22, 18, 35, 70]  middles [18, 35]

  THE 47->53 MAXIMISER'S MIDDLE WORD IS THE ALTERNATION (35, 18) = (q'-s, s) -
  the same object C29 shows controls fuel arity.  The two constructs meet on
  one window.

CONTROL: the ten 43->47 workers' window counts sum to 178,542,615 - DIGIT FOR
DIGIT the round-25 serial total at a completely different seed.  Cost of the
extra resolution: expansions rose 36,606 -> 5,419,312 (148x) at 43->47 and
189,317 -> 4,611,029 (24x) at 47->53; 51,500 and 74,736 core-seconds.

THE CONJECTURE'S SCOREBOARD - FOUR EXACT POINTS, TWO OF THEM BEYOND EVERY SCAN:

    step      max_J Q*_J   F(M+q')   attaining J   k_win(M->q')
    29->31        58          58          3          2  (measured, C13)
    31->37        88          88          4          3  (measured, C13)
    43->47       118         118          3          2  <- PREDICTION
    47->53       145         145          4          3  <- PREDICTION

  The attaining depth equalled k_win + 1 at both steps where k_win had been
  measured independently (a chain of k kills merges k+1 gaps).  It therefore
  PRE-REGISTERS k_win = 2 at 43->47 and k_win = 3 at 47->53 - claims C13's
  k_win census (which stops at 37->41) has never tested, and each one
  kwin_census run away.  Both are consistent with A_kill: 3 at 43->47 and 5 at
  47->53, i.e. longer chains EXIST at both steps and neither carries the
  record - the same "par trading" C13 measured lower down.

WHAT IT MEANS.  Q* is not a relaxation at all at any step where it has been
computed: (D) at a step reduces to computing ONE NUMBER on a small machine's
period.  And the margins the ladder actually has are not round 25's +1: they
are 150 - 118 = +32 at 43->47 and 171 - 145 = +26 at 47->53.
SCOPE, unchanged and stated every time: a certification (never a failure) is
conditional on the span cap - 200 and 171 here, against budgets 150 and 171 -
and every step of this construct with an independent value has agreed exactly.
### C28. FULL-PERIOD GAP HISTOGRAMS BY LAP-PHASE TRANSFER, CYCLICALLY CLOSED (r26)
Repro: research/ghist_transfer.py (worker / merge / optional --delta); gate
research/ghist_gate_r26.py (log research/data/r26/gate_ghist_r26.log); CSVs
research/data/r26/ghist_{13,17,19,23,29,31,37}.csv; handover for the asking
lane research/data/r26/handover-lateral-U6-U9.md.

THE ASK (Lateral's U6/U9, carried over from r25's "NOT DELIVERED") and A
CORRECTION TO ITS PREMISE.  The brief said "your tiling runs cover the period,
only the close was missing".  They do not: the m37 tiling workers computed the
DISTINCT-4-TUPLE SET, not counts, and there were never any m41 tiling runs at
all (the m41 dictionary is a dict_transfer superset).  The round-20 m37 scan
that could have supplied the array threw it away (rule 26).  So the histograms
had to be built.

THE CONSTRUCTION - K2's BIJECTION USED FOR COUNTING, NOT MAXIMISING.  Machine
OLD has period P and openings O; add gears q_1..q_r, T = prod q_i, new period
T*P.  Slot x + jP survives gear q_i iff x avoids the two teeth of PHASE
c_i = -jP mod q_i, and P is invertible mod each q_i, so j -> (c_i(j)) is a
BIJECTION from the T laps onto all T phase tuples.  Hence

   new machine's gaps = (internal gaps of S_c over ALL T phase tuples)
                      + (the T LAP-BOUNDARY gaps, taken IN LAP ORDER),

S_c = {x in O : x in no tooth of any c_i}.  The boundary term is exactly what a
linear close drops (C26, rule 25) - so this construction is cyclically closed BY
BUILD, not by patch.  gap(j -> j+1) = P - last(S_{c(j)}) + first(S_{c(j+1)}),
and the last of them IS the period's wrap gap.  Asserted at merge: total =
prod(q-2) and sum(g*count) = period.

VALIDATION - SIX MACHINES CELL FOR CELL, AND TWO INDEPENDENT METHODS BEYOND:
  m13/17/19/23/29/31 identical, cell for cell, to the round-25 CYCLICALLY
  CORRECTED census gap_pair_hist.csv (10/17/23/33/41/55 cells; totals 1,485 ...
  6,226,553,025 all = prod(q-2); every wrap gap = the first gap, C26's closed
  form).  m31 DOUBLE-SOURCED WITHIN THE CONSTRUCT: from machine 23 + {29,31}
  (899 laps) AND from machine 19 + {23,29,31} (20,677 laps) - different base
  machines, different lap counts, the same 55 cells.
  m37 EXACT, FULL PERIOD, THE NEW OBJECT: 217,929,355,875 gaps over
  1,236,789,689,135 slots, in 4,764 core-seconds (six workers, 794 s each)
  against the round-20 direct sieve's 11,829 s - and it reproduces that scan on
  everything the scan recorded: F = 88, the complete 13-value hole list, and all
  four padding supplies hist[41] = 61,460, hist[43] = 144,162, hist[47] = 48,722,
  hist[53] = 10,390.  Those four are far above the wrap gap (7), so the
  linear-close defect cannot touch them: an exact cross-method check.
  NEW NUMBERS the round-20 scan never produced: hist[59] = 28 and hist[61] = 108
  - the padding supplies for 37->59 and 37->61, and note they are NOT monotone
  in q' (28 at 59 against 108 at 61), the arithmetic selection of C10/R8 again.

THE GEAR-5 TRANSFORM, NOW EXACT AND CYCLIC THROUGH m37 (research/
gear5_transform_r26.py) - the object Lateral's U6/U9 waits on:

    machine   arg H_5(1)    |H_5(1)|/H0   mean gap
     m13      +129.7765      0.307453      3.3704
     m17      +127.8077      0.265725      3.8198
     m19      +126.3336      0.237506      4.2691
     m23      +126.3521      0.218002      4.6757
     m29      +126.0588      0.202323      5.0221
     m31      +125.7680      0.188132      5.3684
     m37      +125.6592      0.178687      5.6752   <- NEW

C17.5 read this as "+126 deg +- 2 at ALL SEVEN machines, machine-independent".
On exact cyclic data the ladder is monotone DOWNWARD from m19 (one +0.02 uptick
at m23), has crossed BELOW 126 and stayed there, with increments -1.97, -1.47,
+0.02, -0.29, -0.29, -0.11: decaying, but not to 126.
AND IT CAUGHT ONE MORE INSTANCE OF THE C26 DEFECT, IN ANOTHER LANE'S NUMBERS:
round 21's m31 mod-5 class counts were [1475661970, 976216219, 2069637131,
1175760034, 529277670]; the exact cyclic counts differ in EXACTLY ONE CELL by
EXACTLY ONE - class 2, because the m31 wrap gap is 7 and 7 = 2 (mod 5).  The
size and the location of the discrepancy are both PREDICTED by "wrap gap =
first gap", so the two constructions agree to the unit and disagree by exactly
the one gap a linear close drops.  Nothing concluded moves (relative error
1.6e-10); any future exact-integer identity on those counts must use the
cyclic row.

THE DELTA FAST PATH (--delta), AND AN HONEST NEGATIVE.  The q children of one
phase-tuple parent differ only by the removal of two residue classes (~2/q of
the elements), so a child's histogram is the parent's with a local correction
at each removed element: a maximal run of removed indices [S..E] deletes the
parent gaps D[S-1..E] and creates one merged gap arr[E+1] - arr[S-1], and
consecutive runs are separated by a kept element so their D-index ranges are
disjoint.  That is O(2n/q) instead of O(n) and should have been ~20x.
MEASURED ALONE AND SEQUENTIALLY on the same m37 slice (benchmark protocol):
simple 71 s, delta 65 s - 1.09x.  The saving is eaten by numpy per-call
overhead: 25 calls on 400-750k arrays cost what 4 calls on 7M cost.  The path
is KEPT because its output is BIT-IDENTICAL to the simple path (hist, first and
last arrays all equal, asserted), which makes it a permanent equivalence gate -
but it is not the speedup it was built to be.

m41: NOT DELIVERED, AND PRICED.  T = 1,363,783 laps at the measured 0.062 s/lap
(alone) = ~85,000 core-seconds, each worker holding ~330 MB.  Launched at 15
workers, it drove free RAM from 6.0 GB to 1.1 GB with the CPU counter at 38% -
paging, not computing - produced no completed worker in 80 minutes, and was
killed; free RAM went straight back to 6.0 GB.  Not restarted: at a
memory-safe 6-8 workers it is a 3-5 hour job and would not have finished
in-round (job-completion rule).  One command, tool gated at six machines.
### C29. A_kill(53->59), THE PREDICTOR REFUTED, AND PHASE SATURATION (r26)
GATE: research/akill_verify_r26.py -> ALL ASSERTIONS PASSED (log
research/data/r26/gate_akill_r26.log). Pre-registration written BEFORE any SAT
call: research/data/r26/prereg_akill_53_59.md. Screen:
research/alt_obstruct_r26.py. Novel doc: docs/novel/phase-saturation-arity.md.

THE PRE-REGISTERED PREDICTOR (round 25's C23 shape, restated as a test):
A_kill(M->q') >= 5 <=> the alternating pair (s, q'-s) is realised at machine M.
The "=>" half is a theorem (overlap lemma); the "<=" half was the content.
At 53->59: u' = 10, s = 20, q'-s = 39; legal gap values 20,39,59,79,98,118,138.
Registered predictions: P1 the pair (20,39) is realised; P2 hence A_kill >= 5
(the 5-chain shape RECURS); P3 A_kill = 5 exactly; P4 F_2(53) in [155,175].

THE MEASUREMENT, AND IT SPLITS THE PREDICTION IN HALF.
  P1 CONFIRMED.  (20,39) REALISED at machine 53, witness
     k0 = 5,408,553,654,414,421,963; (39,20) REALISED, witness
     k0 = 1,522,353,991,400,668,678.  Both re-derived in the gate from the
     DEFINITION (3 consecutive m53 openings with exactly that gap word, every
     other slot of the span blocked gear by gear; killable at r = 49 resp. 10
     mod 59, teeth {10,49}; CRT k* re-verified from scratch).
  P2 REFUTED.  (20,39,20) [k=4], (39,20,39) [k=4], (20,39,20,39) [k=5],
     (39,20,39,20) [k=5] and (20,39,20,39,20) [k=6] are ALL ZERO - and every
     one of them with ZERO SAT CALLS.
  => THE 5-CHAIN SHAPE DOES NOT RECUR AT 53->59.  Pair realisability is
  NECESSARY and NOT SUFFICIENT, so the round-25 predictor is a one-sided test
  only.  A_kill(53->59) >= 3 (rule 22: the full level was not run).

WHAT REPLACES IT - THE PHASE-SATURATION OBSTRUCTION, A THEOREM WITH NO SOLVER.
In the CRT/COV encoding (K1) gear q blocks {a, a + s_q} mod q for a free phase
a, s_q = -2*6^{-1} mod q.  A word with exposed offsets X occurs somewhere only
if every gear has a phase avoiding X:

    FREE_q(X) = Z_q \ ( (X mod q) u ((X - s_q) mod q) )  must be NON-EMPTY,

and |FREE_q(X)| >= q - 2|X|, so only gears q < 2|X| can ever fire - the whole
content is at gears 5, 7, 11.  This is C23's gear-5 argument, which round 25
ran once by hand, turned into a screen.

THE ALTERNATION CEILING, CLOSED FORM (gate part D):

    step        s   q'-s   ceiling   dead gear   measured A_kill
    31->37     25    12       6        5              4
    37->41     14    27       2        5              3
    41->43     29    14       2        5              3
    43->47     16    31       2        5              3
    47->53     18    35       5        5              5   <- ATTAINED
    53->59     20    39       3        7              -
    59->61     41    20       3        5              -
    61->67     45    22       4        7              -

So A_kill(47->53) = 5 sat EXACTLY at its ceiling, and at 53->59 the ceiling
falls back to 3: 53 WAS SPECIAL, and now for an arithmetic reason rather than
an observation.  Note 53->59 is the first step where the binding gear is 7,
not 5.

SOUNDNESS AND REPRODUCTION (gate parts B and C, both asserted):
- 37 words known REALISED at five steps (37->41, 41->43, 43->47, 47->53,
  53->59), each with an independent machine-verified SAT witness: the
  obstruction calls NONE of them zero.
- It REPRODUCES the three structural zeros already on record -
  (18,35,18,35,18) at 47->53 (C23 found this by hand) and (16,31), (31,16) at
  43->47 (on C22's zero list, which paid SAT for them).
- It agrees with cov_count.build_pattern, which returns None in exactly this
  case: a_kill_word.py prints "ZERO (0 calls, 0.0s)" for every obstructed
  word.  CONTROLS RE-RUN at 47->53 this round and they reproduce C23 exactly:
  (18,35) REALISED, (18,35,18,35) REALISED, (18,35,18,35,18) ZERO with 0 SAT
  calls, (35,18,35,18,35) ZERO with 1 SAT call.

THE SCREEN AS A LEVEL PRUNE (alt_obstruct_r26.py, pure arithmetic, instant):
zeroes 3/11 of the k=3 words at 37->41, 6/15 at 41->43, 6/15 at 43->47,
4/19 at 47->53, 7/36 at 53->59; and at deeper levels 27/41 (47->53 k=4),
16/22 (k=5), 4/5 (k=6) - at 41->43 it closes the k=6 level outright
(1 legal word, obstructed: N_6 = 0 BY THEOREM), and at 43->47 the k=7 level
(2 words, both obstructed).

HONEST LIMITS.  The ceiling bounds the ALTERNATION family only; A_kill is a
maximum over all legal words and the PADDED letters (multiples of q') give
words the obstruction does not kill - which is exactly why A_kill is 3 while
the alternation ceiling is 2 at the three steps 37->41, 41->43, 43->47.  The
full A_kill(53->59) level campaign was NOT run: without F_2(53)/F_3(53) span
caps the screen still leaves 29 words at k=3 and 170 at k=5 needing SAT at
14 gears, which would not have finished in-round.  A_kill(53->59) >= 3 is what
is established.
### C30. F_2(53) = 159 - a first computation, and it prices the next rung (r26)
Repro: research/j5_multi.py 23 29,31,37,41,43,47,53 59 seed145 200 2 1 plain
LO HI (floor-1 lap-phase transfer, r = 7 - the deepest transfer run to date);
logs research/data/r26/f2_53_{head,mid,w1}.log, three range workers TILING
[0, 7,952,175) exactly.  Witness verified at machine 53 by
research/qstar_witness_r26.py --nolegal.

    range                        max 2-window span
    [0,         1,590,435)             152
    [1,590,435, 3,180,870)             151
    [3,180,870, 7,952,175)             159   <- the maximum

    F_2(53) = 159, seeded at 145 = F(53) and the answer sits above the seed
    (rule 16), so it is the true maximum up to the span cap.

WITNESS, re-verified at machine 53 from the definition: k =
327,666,424,664,536,738, gaps [77, 82], all 157 other interior slots of the
span blocked, checked slot by slot.  Ratio F_2/F_1 = 159/145 = 1.097, in the
measured band (1.17, 1.02, 1.13, 1.14 at m31, m37, m41, m47).
SCOPE: the >= 159 direction is UNCONDITIONAL (exhibited witness).  The <= 159
direction is conditional on the span cap 200, and here that condition is real -
the deletion-ladder cap F_2(53) <= F(59) is unavailable because the corpus F
ladder stops at 53.  The trivial cap is 2F(53) = 290.

TWO CONSEQUENCES, both immediate:
1. A NEW LOWER BOUND ON THE NEXT CORPUS RUNG, unconditional.  The deletion
   ladder (K3) gives F_2(M) <= F(M + one more gear), so

        F(59) >= F_2(53) = 159,   equivalently  F(2,59) >= 477

   where the corpus ladder previously stopped at F(2,53) = 435 with nothing
   at 59.  (D) at 53->59 needs F(59) <= F(53) + 59 = 204, so the remaining
   room at that step is at most 45.
2. IT PRICES THE A_kill(53->59) CAMPAIGN.  Feeding 159 in as the 2-block span
   cap, and then applying the phase-saturation screen (K9) and the mirror law
   (rule 27), the levels collapse:

        level   legal words   after F_2 cap   after screen   SAT calls
        k=3          36            19             12             6
        k=4         170            59             19            11
        k=5         776           169             20            10

   i.e. 982 words -> 27 solver calls, a 36x cut, with every step of the cut a
   theorem.  That is the scoped next-round item; it was NOT started this round
   because 27 refutations at FOURTEEN gears would not have finished in-round
   (job-completion rule).
### C31. THE PHASE-SATURATION SCREEN APPLIED TO A GAP-TUPLE DICTIONARY (r26)
Repro: research/screen_tuples_r26.py (log research/data/r26/screen_41_r26.log);
output research/data/r26/gap_tuples_41_4_screened.csv.

The obstruction of C29/K9 is not special to kill words - it applies to ANY
prescribed pattern of openings, so it applies to Constructor's gap-tuple
dictionaries.  A gap m-tuple is realised at machine M only if every gear has a
phase avoiding all m+1 exposed offsets; and since |FREE_q(X)| >= q - 2(m+1),
for a 4-tuple ONLY GEARS 5 AND 7 can ever fire.  Two lookups per tuple, seconds
over millions of rows.

SOUNDNESS GATE FIRST - it must remove NOTHING from a set of realised tuples:

    machine 23  15,696 exact 4-tuples  ->  15,696 survive  (0 removed)
    machine 29  45,854                 ->  45,854          (0)
    machine 31 115,193                 -> 115,193          (0)
    machine 37 291,675                 -> 291,675          (0)

468,418 tuples known realised by full-period scan, zero false kills.

APPLIED TO CONSTRUCTOR'S m41 ARITY-4 SUPERSET (their stated blocker for rung
nine: the dict_transfer superset is inflated enough that 12/12 sampled
superset-YES tuples were CRT-refuted):

    input     4,239,676 tuples (research/data/gap_tuples_41_4_transfer.csv)
    gear 5 has no admissible phase for   780,486  -> ZERO BY THEOREM
    gear 7 has no admissible phase for   644,616  -> ZERO BY THEOREM
    gears >= 10 can never fire
    SURVIVORS 2,814,574 (66.39%);  1,425,102 removed by arithmetic alone

and the induced 3-tuple dictionary falls 130,942 -> 111,899 with it (the
induced 1- and 2-tuple dictionaries, 88 and 3,333, are unchanged - they were
already exact against COV-SAT).  A screened superset is still a superset, so
it drops straight into the same certificate slot with no soundness argument to
redo; it is simply tighter.
### K8. The lap-phase GAP HISTOGRAM transfer (r26)
research/ghist_transfer.py (worker / merge / --delta), gate
research/ghist_gate_r26.py.  K2's bijection used for COUNTING: laps of the new
machine are phase-filtered copies of the old machine's opening set, so the new
machine's whole gap histogram is a sum over T phase tuples plus T lap-boundary
gaps taken IN LAP ORDER - the cyclic close, exactly (rule 25).  Both period
identities asserted at merge (total = prod(q-2), sum(g*count) = period).
Gated cell for cell against the round-25 corrected census at m13/17/19/23/29/31,
m31 additionally from TWO different base machines, m37 against the round-20
direct 11,829 s sieve (F, the 13 holes, four padding supplies), m41 against
COV-SAT (F = 91 and the hole list {84,87,89}).  See C28.

### K9. The phase-saturation screen (r26)
research/alt_obstruct_r26.py, gate research/akill_verify_r26.py.  A word whose
exposed set X leaves some gear q with NO admissible phase - i.e.
(X mod q) u ((X - s_q) mod q) = Z_q, s_q = -2*6^{-1} mod q - is ZERO with no
solver call.  |FREE_q| >= q - 2|X|, so only gears q < 2|X| can fire: the whole
content is at gears 5, 7, 11.  Sound (never zeroes any of the 37 words known
realised), reproduces every structural zero on record, reverse-invariant, and
gives a CLOSED-FORM ceiling on the alternating chain per step.  See C29 and
docs/novel/phase-saturation-arity.md.

## Retracted / corrected (round-26 additions)

R25. THE ALTERNATION-PAIR PREDICTOR - REFUTED BY ITS OWN PRE-REGISTERED TEST.
    Round 25's C23 closed with the observation "at q' = 47 the alternating pair
    (16,31) is not realised, at q' = 53 the pair (18,35) is", and round 26's
    brief promoted it to a checkable predictor: A_kill(M->q') >= 5 iff the pair
    (s, q'-s) is realised at M.  Pre-registered
    (research/data/r26/prereg_akill_53_59.md) and tested at 53->59: the pair
    (20,39) IS realised - two definition-level verified witnesses - and the
    4-letter and 5-letter alternations are nevertheless ZERO BY THEOREM.
    NOTHING MEASURED IN ROUND 25 IS WRONG; the INFERENCE from it was.  Pair
    realisability is NECESSARY (overlap lemma) and NOT SUFFICIENT.  Replaced by
    the phase-saturation ceiling (C29), which is a theorem, costs no solver
    call, and retrodicts every step including the two the predictor got right.

## Standing rules (round-26 additions)

27. DECIDE ONE WORD PER REVERSE CLASS.  #occ(w) = #occ(reverse(w)) EXACTLY
    (Lateral's mirror law: the opening set is closed under k -> -k).  Their
    audit of THIS LANE's round-24 logs found 82 decisions in which every
    reverse pair agreed and 12,877 s of 27,946 s - 46% - was spent on the
    redundant half, including two of the four span-141 words that cost
    20,005 s.  a_kill_par.py now collapses each level to its reverse classes
    and copies the verdict; the legal-word lists are reverse-closed and the
    phase-saturation screen is provably reverse-invariant (both asserted in
    research/akill_verify_r26.py).
28. SET A WORKER'S PROGRESS STRIDE FROM ITS OWN SHARE, NOT THE WHOLE JOB.  This
    round's histogram workers printed every 20,000 laps and had 5,735 each;
    the Q* range workers print every 200,000 start indices and have 795,217
    each.  Both ran for forty minutes with NO output at all, so "slow",
    "stalled" and "crashed" were indistinguishable and a healthy job was
    nearly killed twice.  Stride = worker share / 20, and print at start-up.
29. A REFUTATION MAY BE FREE - RUN THE ARITHMETIC SCREEN BEFORE THE SOLVER.
    Before paying for an UNSAT, test whether some SMALL gear has no admissible
    phase against the pattern's exposed set (K9).  It costs microseconds, it
    fired on 3-27 words per level at every step measured, and one of the words
    it kills at 47->53 had already been refuted by hand in round 25 - i.e. the
    project has been paying SAT for facts that are two lines of modular
    arithmetic.  General form: a pattern question has a PIGEONHOLE LAYER
    (does every gear have somewhere to stand?) below its search layer.
30. PROCESS COUNT IS NOT LOAD - MEASURE % PROCESSOR TIME.  With 43 python
    processes box-wide the CPU sat at 42%: the box was PAGING, not computing,
    and every lane's jobs were running at a fraction of speed while looking
    busy.  Check \Processor(_Total)\% Processor Time and free physical memory
    together; if utilisation is far below the process count, the fix is FEWER,
    FASTER processes, not more of them.
