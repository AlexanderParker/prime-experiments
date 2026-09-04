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

## Round-27 additions (mechanic)

### C32. F_4(41) = 118 EXACT - a first computation, and it prices a census (r27)
Repro: research/j5_multi.py 23 29,31,37,41 43 seed109 150 4 1 plain LO HI
(floor-1 lap-phase transfer, r = 4), two range workers TILING [0, 7,952,175);
logs research/data/r27/f4_41_w{0,1}.log, 301 s each = 602 core-seconds.
Witnesses re-verified AT MACHINE 41 by research/qstar_witness_r26.py --nolegal.

    J        Q_J(41; 1)      status
    2            109         seeded, NOT resolved (F_2(41) = 103 < seed)
    3            110         = F_3(41), the KNOWN exact value - two-sided anchor
    4            118         = F_4(41), NEW AND EXACT

NOT SPAN-CAPPED, which is the point: the run's cap 150 sits ABOVE the
deletion-ladder cap F_4(41) <= F(41 + 3 gears) = F(53) = 145, so no window was
excluded by the cap and the value is unconditional.  The standing entry was
"F_4(41) <= 145" (C11), i.e. nothing.

THREE CONTROLS, none of which the tool was told about:
 (a) J = 3 returns 110 = F_3(41), seeded one below at 109, so the run had to
     FIND it as well as refute everything above - and the maximiser it reports
     is k = 30,382,499,692,410, gaps [77, 11, 22], which is C11's round-24 SAT
     witness AT THE SAME ADDRESS, found by a different vehicle.
 (b) The two workers' J = 4 maximisers are an exact MIRROR PAIR in machine 23's
     coordinates: 4,834,947 + 32,347,080 = 37,182,027 = P(23) - 118.  Two
     processes sharing no state, reproducing k -> -k (C18).  Verified at
     machine 41:
         k = 33,044,111,735,752  gaps [51, 2, 50, 15]  span 118
         k = 17,664,265,518,665  gaps [15, 50, 2, 51]  span 118
     each with all 114 other slots of the span blocked, checked slot by slot.
 (c) The maximising 4-tuple (51,2,50,15) and its reverse are both PRESENT in
     the round-26 screened superset - so the cap below is attained, i.e. tight.

### C33. THE m41 EXACT 4-TUPLE CENSUS: PRICED, PART-DELIVERED (r27)
Repro: research/price_m41_census_r27.py (pricing), research/m41_census_r27.py
(worker / merge / gate), research/m41_spancap_r27.py (the F_4 cap).
Brief item (a).  The brief asked for the exact census over the round-26
screened superset (2,814,574 candidates).  IT WAS PRICED FIRST, and the price
is the finding.

THE PRICE, MEASURED NOT GUESSED (span-stratified sample, 24 reverse classes per
stratum, machine 41 = 11 gears, Constructor's exact cover CSP as decider):

    span band   reverse classes   realised/sampled   mean s/decision   core-s
      1 -  60          59,245          24/24              0.032         1,871
     61 -  80         141,672          24/24              0.189        26,809
     81 - 100         294,791           4/24              3.495     1,030,389
    101 - 110         206,984           0/24              4.252       880,087
    111 - 120         220,687           0/24              4.184       923,314
    121 - 130         210,175           0/24              2.984       627,059
    131 - 140         197,032           0/24              2.775       546,746
    141 - 145          76,957         (not sampled)           -             -
    TOTAL           1,407,543 reverse classes        >= 4,036,276 core-seconds

i.e. >= 1,121 CORE-HOURS, or 187 wall hours at six workers.  The competing
vehicle - the period route, ghist_transfer extended to emit tuples - is bounded
BELOW by ~2e5 core-seconds (round 26 measured 0.062 s/lap for the HISTOGRAM
alone over T = 1,363,783 laps = 85,000 core-s, and tuple emission needs a
6.2M-element scatter per lap on top).  That route enumerates all
prod(q-2) = 8.499e12 openings of machine 41's period, which is irreducible for
any period vehicle.
=> BOTH VEHICLES EXCEED THE ROUND BY ONE TO THREE ORDERS OF MAGNITUDE.  Round
26's "~85,000 core-s" figure was for the histogram, a strictly easier object,
and must not be quoted for the census.

WHAT WAS DELIVERED INSTEAD, in three parts.

(1) THE F_4(41) SPAN CAP - a whole band of the census closed with NO SOLVER.
A realised 4-tuple is four CONSECUTIVE gaps, so its span is at most F_4(41) BY
DEFINITION.  C32 pins that at 118 where the standing bound was 145 - which is
exactly the superset's own maximum span, i.e. pruned nothing.

    dict_transfer superset (K4)       4,239,676 -> 2,736,800   (35.45% zeroed)
    phase-saturation screened (C31)   2,814,574 -> 1,747,819   (37.90% zeroed)
    induced 3-tuple dictionary          111,899 ->    95,331

    COMBINED, BY THEOREM ALONE AND IN SECONDS:
        4,239,676  ->  2,814,574  (C31 phase saturation)  ->  1,747,819
        58.8% of Constructor's arity-4 superset removed with no solver call
    research/data/r27/gap_tuples_41_4_screened_spancap.csv

(2) THE EXACT SHARD, BY ASCENDING SPAN.  research/m41_census_r27.py decides
candidates in ascending span order, so at any stopping point the deliverable is
"the exact machine-41 4-tuple dictionary is COMPLETE at every span <= S", with
S read off the logs.  Five workers, mirror-halved (rule 27 - the superset's
reverse-closure is ASSERTED, not assumed), each resuming from its own log.

(3) THE DECIDER GATE, TWO-SIDED, at the three machines whose exact 4-tuple
dictionary this lane scanned in full (C21).  research/m41_census_r27.py gate,
90 s, ALL ASSERTIONS PASSED:

    m23  positive 15,696/15,696 YES   negative 2,000/2,000 NO   (7 gears)
    m29  positive  3,000/ 3,000 YES   negative 3,000/3,000 NO   (8 gears)
    m31  positive  2,000/ 2,000 YES   negative 2,000/2,000 NO   (9 gears)

The NEGATIVE controls are 4-tuples built from the machine's own realised gap
VALUES but absent from its exact dictionary - Formalist's round-26 lesson (an
audit without positive controls is not an audit) applied in both directions.

    THE SHARD AS DELIVERED (two waves, five workers each, both resuming from
    the same per-worker logs):
      research/data/r27/gap_tuples_41_4_exact_le77.csv
      COMPLETE AT EVERY SPAN <= 77: 169,981 reverse classes decided, ZERO
      undecided, 338,855 tuples realised.
      Inflation of the screened superset over that region: 1.0028x
      (339,793 candidates -> 338,855 realised).
      178,886 reverse classes decided in total, 868 refuted.

    THE INFLATION ONSET IS SHARP AND IT IS AT SPAN 68.  Every one of the
    137,000 reverse classes of span <= 67 is REALISED; the first refutation
    anywhere is at span 68, and the refuted count then climbs

        span    68  69  70  71  72  73   74   75   76   77
        refuted  2   0   6  14  26  17   68  117  117  105

    against the timed sample's 20/24 refuted at span 81-100 and 24/24 at
    101-140.  So the dictionary transfer (K4) is EXACT below 68 and collapses
    over the next ~30 units of span.  That is a fact about the CLOSURE: a
    machine-37 walk short enough to be pinned by its 4-windows is realised;
    past ~68 the order-4 closure stops determining it.

### C34. THE m37 QUALIFYING SPECTRUM, RE-DERIVED AND EXTENDED (r27)
Repro: research/j5_multi.py 23 29,31,37 41 seed87 200 8 14 plain LO HI, two
range workers TILING the period; logs research/data/r27/qspec37_w{0,1}.log,
2,613 and 2,602 s.  Both workers report the same row.

    J             2    3    4    5    6    7    8
    Q_J(37; 14)  90   97  103  110  112  114  112      budget F(37)+41 = 129

C20's round-23 row for J = 2..7 is 90, 97, 103, 110, 112, 114 - REPRODUCED
EXACTLY four rounds later by an independent run, this time with witnesses, and
extended to J = 8.  Two witnesses re-verified AT MACHINE 37 from the definition
(research/qstar_witness_r26.py):

    J=7  k = 1,006,677,586,778  gaps [4,23,22,15,15,28,7]  span 114
         middles [23,22,15,15,28] all >= 14, 107 other slots blocked
    J=8  k =   965,213,765,810  gaps [2,15,20,26,17,15,15,2]  span 112
         middles [15,20,26,17,15,15] all >= 14, 104 other slots blocked

THE SPECTRUM TURNS OVER AT J = 8: 114 -> 112.  Formalist found the same shape
at machine 31 (Q_j(31;12) rises 68,85,90,91 then falls 90,88) and called it a
new structural fact; it recurs at machine 37 one depth deeper, so the binding
depth is again interior, not terminal.

FOR CONSTRUCTOR, and please check the index convention before quoting (rule 5).
Your round-26 row is Q_2..Q_6 = 88, 90, 97, 103, 110 at floor 14 with Q_7 lost
to the memory event and bounded by the layer argument at <= 174.  Your values
match mine shifted by one - your Q_3..Q_6 = 90, 97, 103, 110 = my Q_2..Q_5 -
which reads as your J counting OPENINGS where mine counts GAPS (J openings =
J-1 gaps), and is consistent with your Q_2 = 88 = F(37) = "my Q_1".  On that
reading YOUR MISSING Q_7 IS 112 and your Q_8 would be 114, both far under the
layer bound 174 and under budget.  I am not adjudicating your indexing - the
row above is stated in MY convention with witnesses, so it can be re-indexed
without ambiguity.

### C35. (D) AT 53 -> 59 IS DECIDED TRUE - the first step past the corpus ladder (r27)
Repro: research/j5_multi.py 23 29,31,37,41,43,47,53 59 seed203 260 7 20 legal
LO HI (word-legal lap-phase transfer, r = 7 - the deepest word-legal run to
date); two range workers TILING [0, 7,952,175); logs
research/data/r27/f59_A_w{0,1}.log, 1,806 and 1,807 s.

THE VEHICLE IS THE RECORD LAW, and this is its first use as a COMPUTATIONAL
instrument beyond every value the project or the corpus holds.  Constructor's
round-26 attainment theorem makes

    F(M + q')  =  max_J Q*_J(M; legal for q')     (equality, both directions)

so machine 59's maximal gap is computable on MACHINE 23's period - period
37,182,145 against machine 59's 1.96e19, a ratio of 5.3e11.  Machine 59 is
never built.

    J                2    3    4    5    6    7
    Q*_J(53; 59)   203  203  203  203  203  203     (all at the seed)
    max over J = 203   vs budget F(53) + 59 = 204   ->  CERTIFIES

    ==>  F(59) <= 203 < 204,   so (D) HOLDS AT 53 -> 59.

WHY THIS STEP WAS OPEN.  The corpus F(2,y) ladder stops at y = 53, so 53 -> 59
was the first step of the ladder with NO upper bound on the new machine's F
anywhere - round 26 could only say F(59) >= F_2(53) = 159 (deletion ladder, an
exhibited witness) and "at most 45 of room remains".  Every earlier step of C20
was decided by arithmetic from two known F values; this one had to be computed.

SCOPE, stated as always: the certification is conditional on the span cap 260
(no word-legal window of machine 53 of span above 260 at depth <= 7).  Every
step of this construct with an independent value has agreed exactly, and the
observed maxima sit far below their caps.  A FAILURE would have carried no
condition; this is a certification, so it does.

A SECOND DIVIDEND, and it is what closed brief item (b): because a realised
k-chain kill word IS a word-legal window of J = k-1 gaps, this one run refutes
EVERY kill word of span in (203, 260] at every depth J <= 7, with no solver.
See C36.

### C36. A_kill(53 -> 59): THE LEVEL CLOSED WITHOUT A SINGLE UNSAT (r27)
Repro: research/a_kill_par.py 53 59 3 6 --pool 3 (log
research/data/r27/akill_53_59.log), with the scan-derived verdicts recorded by
research/akill_scan_verdicts_r27.py and the level re-derived by the gate
research/akill_verify_r27.py.  a_kill's DEFAULT_CAPS gains a row for machine 53:
[145, 290, 435] - the UNCONDITIONAL caps F_2 <= 2F, F_3 <= 3F, deliberately NOT
this lane's own F_2(53) = 159, whose upper direction is span-cap-conditional.

C30 priced this at 27 solver calls.  The actual cost of the k=3 level was ZERO
solver calls for every refutation, because three completed SCANS own the
verdicts and a scan is cheaper than a 14-gear UNSAT (rules 20 and 29):

    source                                   refutes                    words
    phase-saturation screen (K9)             gear has no free phase        7
    r26 F_2(53) scan  (seed 145, cap 200)    2-window span in (159,200]    9
    r27 top-band scan (seed 145, cap 158)    2-window span in (152,158]    4
    r27 F(59) stage A (seed 203, cap 260)    any J<=7 span in (203,260]    5+18
    SAT                                      -                             0

AND THE COST OF NOT DOING THIS WAS MEASURED, not imagined: the first launch of
the campaign put three of those words on pysat at 14 gears and they were still
running after TWO HOURS each.  Every one of them is refuted by a scan that was
already on disk or that cost 1,300-1,800 s of walking.

THE SCOPE POINT THAT MAKES IT SOUND, and it is worth stating because it looks
circular and is not: "F_2(53) <= 159" IS conditional on the round-26 span cap
200, but "NO 2-window of machine 53 has span in (159, 200]" is NOT - a span cap
conditions claims about spans ABOVE it only.  The refutations use the second
statement.  Likewise the top band: the round-27 run seeded at 145 with cap 158
returns 152 from both range workers, so no 2-window has span in (152, 158];
with F_2(53) = 159 realised, machine 53's adjacent-pair span spectrum has a
SIX-WIDE HOLE at 153..158 immediately below its maximum.  That is a new datum
and it is what killed the two words SAT could not.

    A_kill(53 -> 59) = 4 EXACT.  Every level complete:
        N_3 = 8 realised of 36 legal words
             (20,39) (20,59) (20,98) (20,118) and their four reverses
        N_4 = 1 realised of 18:  (20, 98, 20)  - a palindrome
             witness k = 5,179,823,167,446,585,215, re-verified from the
             definition (4 consecutive m53 openings, killable at r = 49 mod 59)
        N_5 = 0, and the k=5 LEVEL IS EMPTY BEFORE ANY DECISION: a 4-letter
             word needs both of its 3-letter sub-words realised, and the only
             realised 3-letter word is the palindrome (20,98,20), whose two
             overlaps cannot both be it.  The overlap lemma closes the arity.
    THE WHOLE CAMPAIGN COST ONE UNSAT: (39,20,98) span 157, 2,666 s.  Every
    other refutation came from the screen (7 words), the band table (63 words)
    or the mirror law.

    AND THE SHAPE IS THE PADDED ALTERNATION.  C29's phase-saturation ceiling
    for the PURE alternation at this step is 3, and the pure alternation
    (20,39,20) IS zero, exactly as the theorem says.  What carries arity 4 is
    (20, 98, 20) = (s, q' + (q'-s), s) - the alternation with ONE LAP OF
    PADDING inserted.  Against the ceiling table:

        step      31->37 37->41 41->43 43->47 47->53 53->59
        ceiling      6      2      2      2      5      3
        A_kill       4      3      3      3      5      4
        A - ceil    -2     +1     +1     +1      0     +1

    so at every step whose pure-alternation ceiling is 2 or 3, A_kill sits
    exactly ONE above it, and the lifting word is always padded.  THE NAMED
    NEXT CONSTRUCT (measurement directive): the phase-saturation ceiling of the
    PADDED alternation family (s, q'+(q'-s), s, ...) - a closed form in the
    small gears, exactly like C29's, and the object that would turn "+1 at four
    steps" into a theorem instead of a pattern.

EVERY REALISED 3-CHAIN CARRIES THE LETTER s = 20.  The four partners are
39 = q'-s, 59 = q', 98 = q' + (q'-s), 118 = 2q' - i.e. s paired with a lap of
padding, in every combination that fits under F(53) = 145.  Nothing pairs two
non-s letters: (39,59), (39,79), (59,59), (59,79) are all zero, and (59,59) -
the DOUBLE-PADDED 3-chain that had occurred at 41->43, 43->47 and 47->53, three
consecutive steps - is ZERO here by the phase-saturation screen, with no solver
call.  The run of double-padded 3-chains ends at 53 -> 59.

### C37. THE DESCENDING-BAND SWEEP - a technique note, with its cost law (r27)
Repro: research/f59_sweep_r27.py.  The lap-phase transfer's cost splits into
WALKING windows (set by the span cap) and PHASE-EXPANDING the windows whose
span exceeds the running best (set by the seed).  Rule 24 already said to seed
high; this measures what seeding buys, at fixed cap and fixed period:

    band            cap  seed  expansions/worker  s/worker  verdict
    (203, 260]      260   203        130,627        1,806    EMPTY -> F(59)<=203
    (193, 204]      204   193        ~115,000       1,038    EMPTY -> F(59)<=193
    (183, 194]      194   183        ~285,000       ~4,200   (see below)

so LOWERING THE SEED BY TEN COSTS ABOUT FOUR TIMES THE ROUND, while lowering
the CAP makes the walk cheaper.  The consequence is a technique: run bands
(lo_i, hi_i] with hi_{i+1} = lo_i in DESCENDING order.  Each band is priced
separately, each yields a monotonically improving upper bound the moment it
finishes, and the FIRST NON-EMPTY band's maximum is the answer, every larger
span having been refuted already.  A single run seeded at the floor pays for
every band at once and reports nothing until it is done.
This is the second time this lane has had to learn the same lesson in a new
form (rule 24 was the first); it is now rule 31.

### C33b. THE SHARD, GATED (r27)
research/data/r27/gap_tuples_41_4_exact_le77.csv, checked after emission:
  338,855 tuples, REVERSE-CLOSED (asserted - the mirror halving is only sound
  if the emitted set is, and a copied verdict that lost its partner would show
  up here); max span exactly 77; every tuple present in the screened superset
  (a decision can only ever REMOVE from a superset); 944 tuples = 472 reverse
  classes refuted over the same region.
  Induced dictionaries of the shard are restricted to span <= 77, so they are
  LOWER bounds on machine 41's true induced dictionaries, not the dictionaries
  themselves.

### C38. F(59) BRACKETED, AND THE BAND TABLE (r27)
Repro: research/f59_sweep_r27.py (the descending ladder), plus two targeted
bands; gate research/akill_bands_r27.py re-reads all six scans and re-asserts
that each is a set of range workers TILING machine 23's period exactly.

    band          jmax  seed  cap   reported max   verdict
    (145, 158]      2    145  158        152       no 2-window span in (152,158]
    (145, 200]      2    145  200        159       no 2-window span in (159,200]   [r26]
    (152, 184]      3    152  184        161       no J<=3 window in (161,184]
    (178, 184]      7    178  184        178       EMPTY  (4,930 s x 7 workers)
    (183, 194]      7    183  194        183       EMPTY  (3,256 s x 7 workers)
    (193, 204]      7    193  204        193       EMPTY  (1,038 s x 7 workers)
    (203, 260]      7    203  260        203       EMPTY  (1,806 s x 2 workers)

    ==>  161 <= F(59) <= 178      (budget 204, so (D) HOLDS with >= 26 to spare)

AND THAT DECIDES THE MANAGER'S INCREMENT LAW AT A STEP IT HAD NEVER SEEN.  The
candidate law is F(M + q') - F_2(M) <= s_min(q') = min(2u', q' - 2u'); at
53 -> 59 that is s_min = 20 and F_2(53) = 159, so the law predicts
F(59) <= 179.  MEASURED: F(59) <= 178.  THE LAW HOLDS, with the increment
pinned to F(59) - F_2(53) in [2, 19] against a cap of 20.  This was
PRE-REGISTERED as B2 before the scan (research/data/r27/prereg_mechanic_r27.md)
and is the round's one prediction that was a genuine bet rather than a
corollary: the law's single known failure is the PADDED step 31 -> 37, and
53 -> 59 has padding available (the F(59) lower-bound witness itself has a
2q' = 118 padded interior), so it could have gone the other way.

THE LOWER BOUND IS AN EXHIBITED WINDOW, verified at machine 53 from the
definition (research/qstar_witness_r26.py):
    k = 2,505,673,933,219,103,747   openings at k + [0, 10, 128, 161]
    gaps [10, 118, 33], span 161, all 158 other slots of the span blocked
    middle gap 118 = 2q' - a legal kill letter, and a DOUBLE-PADDED interior
so by the attainment theorem F(59) >= 161, up from round 26's 159.
FREE CONTROL: the same run returns Q*_2 = 159 = F_2(53) AT ROUND 26's OWN
WITNESS ADDRESS k = 15,468,233 (in machine-23 coordinates) - a different run,
a different seed, a different worker count, the same address.

WHAT WOULD CLOSE IT, priced: the only spans left are (161, 178] and, by the
depth-3 band, only at depths J >= 4 above 161.  A band (161, 184] at JMAX = 7 was
LAUNCHED AND KILLED - no worker reached its first progress stride in 35
minutes, a projection past four hours, because a 23-unit-wide band at that span
level expands an enormous number of windows.  The affordable shape is narrow
bands: the width and the ABSOLUTE SPAN LEVEL both drive the expansion count,
and an 11-unit band cost 1,038 s at level ~198 and 3,256 s at level ~188.


## Standing rules (round-27 additions)

31. RUN A MAXIMUM-FINDING SCAN AS A DESCENDING LADDER OF BANDS, NOT AS ONE LOW-
    SEEDED RUN.  Rule 24 says seed high; this says what to do when you do not
    know the value.  A run seeded at `lo` with cap `hi` decides exactly
    "the maximum in (lo, hi], or lo if empty", so bands with hi_{i+1} = lo_i
    compose with no gap, each finishes, each tightens the bound the moment it
    lands, and the FIRST NON-EMPTY band from the top gives the answer.
    MEASURED at 53->59: lowering the seed ten units cost ~4x the run, so a
    single run seeded at the floor would have paid for every band at once and
    reported nothing until it was done.  A band also has a use its target does
    not: an EMPTY band refutes every OTHER object whose span lands in it (here
    it closed a whole A_kill level, at every depth, with no solver call).
32. A DEPTH CAP IS A COST CONTROL, NOT JUST A SCOPE CHOICE.  The same band
    (152, 184] at JMAX = 7 was projected at NINE HOURS per worker and was
    killed; at JMAX = 3 - which is all the question needed, because the words
    being refuted were 3-gap windows - it runs in 45 minutes.  The walk's break
    condition is `lbmax > JMAX - 1`, so JMAX sets how deep every window is
    pursued.  Before launching, ask what DEPTH the question actually needs.
33. SET THE PROGRESS STRIDE INSIDE THE TOOL FROM THE WORKER'S SHARE, NOT FROM A
    GLOBAL CONSTANT.  Rule 28 said this about orchestrators; j5_multi.py still
    prints every 198,804 GLOBAL start indices, so a worker with a small or
    late range can run for an hour before its first line and is
    indistinguishable from a hang.  It cost me a wrong liveness read twice this
    round, once on a job that genuinely was mis-sized and once on one that was
    not.

## Retracted / corrected (round-27 additions)

R26. MY OWN ROUND-27 PRE-REGISTRATION, THREE OF EIGHT SCORED PREDICTIONS
    REFUTED BY MY OWN RUNS (research/data/r27/prereg_mechanic_r27.md, written
    before any solver call):
    A1  "A_kill(53->59) = 3 exactly."  REFUTED: (20, 98, 20) is REALISED at
        k = 4 (witness 5,179,823,167,446,585,215).  The reasoning was that the
        alternation ceiling is 3 and that at every step whose ceiling was below
        5 the padded words lifted A_kill to exactly 3.  The padded words lift
        it FURTHER here.  The alternation ceiling itself is not touched - the
        pure alternation (20,39,20) IS zero, as the theorem says - so what is
        refuted is my extrapolation from three steps, not C29.
        (This is standing rule 1 in a new costume: I extrapolated a per-step
        pattern instead of computing it.)
    A2  "(59,59) is realised - the double-padded 3-chain, which has occurred at
        41->43, 43->47 and 47->53."  REFUTED: it is ZERO, and by the phase-
        saturation screen with no solver call.  The run of double-padded
        3-chains ends at 53->59.
    C1  "The screened superset is exact at every span <= 80."  REFUTED: the
        first refutation is at span 68 (2 of 7,008 classes).  The onset is
        sharp and it is twelve units lower than I predicted.
    CONFIRMED: A3 (all nine k=3 words above the F_2(53) cap are zero - an
    independent SAT-free cross-check of C30's upper direction), B1 ((D) holds
    at 53->59), C3 (F_4(41) in [113,125]; it is 118).

## Open watches and checkpointed jobs (round-27 additions)

WATCHES
- THE m41 EXACT 4-TUPLE CENSUS: complete at every span <= 77, frontier
  checkpointed.  research/m41_census_r27.py `work 5 w 145 HOURS` resumes each
  worker from its own log with one command line; `merge 5 145` recomputes the
  frontier and re-emits the dictionary (this round used it twice, and the
  second wave moved the frontier 75 -> 77).  The remaining population is
  ~1,229,000 reverse classes and the measured price of the rest is ~4.0e6
  core-seconds, so it is a MULTI-ROUND object, not a next-round one, unless a
  cheaper decision vehicle appears.  What would change that: an exact m41
  ADJACENT-PAIR dictionary (3,333 candidates) or an exact 3-tuple dictionary
  (95,331 after the span cap) would prune the 4-tuples by the overlap lemma for
  free - but pairs are the EXPENSIVE end of the CSP (fewer open points, larger
  gear domains), so that has to be priced before it is launched.
- THE INFLATION ONSET AT SPAN 68 is a fact about the ORDER-4 CLOSURE of the
  dictionary transfer, measured at one step (37 -> 41).  Does the onset span
  scale with the machine, with F, or with the mean gap?  One more step would
  say; 31 -> 37 is affordable (the exact m37 dictionary exists) and is the
  cheap test.
- F_2(59) NOT COMPUTED.  It needs r = 8 and a seed, and the seed wants F(59)
  first.  With the band technique it is a scoped next-round item and it would
  give F(61) >= F_2(59) by the deletion ladder, extending the corpus lower
  bound one rung further.
- Q_8(37; 14) = 112 TURNS OVER from Q_7 = 114 - the second machine at which the
  qualifying spectrum is non-monotone in depth (Formalist found it at m31).
  Whether the turnover point is arithmetic or structural is untested.

## Round-28 additions (mechanic)

### C39. THE INFLATION-ONSET LADDER, AND THE ONSET LAW (r28)
Repro: research/onset_r28.py (four steps from the dictionaries already on
disk), research/onset_ladder_r28.py (three more steps, small machines
recomputed from the period), research/onset_law_r28.py (mechanism + law),
research/onset_oos_r28.py (the out-of-sample test); logs
research/data/r28/{onset_anatomy,onset_law,onset_oos,y5}.log.  Brief item (c).
Novel doc: docs/novel/dictionary-monotonicity-onset.md.

THE QUESTION.  Round 27 found the 37 -> 41 arity-4 dictionary transfer EXACT
below span 68 and refuting sharply above it.  Is 68 predictable from the
machine's constants?

THREE CLOSED FORMS WERE PRE-REGISTERED BEFORE THE LADDER WAS MEASURED
(research/data/r28/prereg_mechanic_r28.md D1-D3): F_2 one machine back
(F_2(31) = 68), 2F two machines back (2 F(23) = 68), and a constant ratio to
F(M) (0.773).  ALL THREE FAIL AT EVERY OUT-OF-SAMPLE STEP; the third only ever
matched its own calibration point.  D4 - my own registered expectation that all
three would fail - is the one that stood.

THE LADDER, EXACT AT EIGHT STEPS (both dictionaries exact at all eight; the
small machines' were recomputed in-round from the CYCLICALLY CLOSED period with
F and F_4 asserted against their known values):

    step       11->13 13->17 17->19 19->23 23->29 29->31 31->37 37->41
    onset         13     15     17     25     31     41     53     68
    onset/F(M)  1.857  1.364  0.944  1.000  0.912  0.953  0.914  0.773
    onset/F_2(M)1.182  0.938  0.680  0.806  0.795  0.745  0.779  0.756

- no ratio is constant, and the ladder 13,15,17,25,31,41,53,68 is SMOOTHER
  than F itself (successive ratios 1.15, 1.13, 1.47, 1.24, 1.32, 1.29, 1.28
  against F's 1.57, 1.64, 1.39, 1.36, 1.26, 1.35, 1.52) - the onset is NOT
  arithmetic-selected the way F is.

THE LAW, AND IT IS A RECURSION, NOT A FORMULA IN THE LETTERS.  With q'' the
next prime after q':

        onset(M -> q')  =  min span of [ (D_4(q'') \ D_4(q'))  INTERSECT
                                         the transfer's own emissions ]

"the transfer M -> q' first over-generates exactly where the NEXT machine's new
repertoire begins - it emits, one gear ahead of schedule, the tuples that only
become realisable when the following gear is added."
    HIT AT 6 OF 6 in-sample steps 13->17 ... 31->37, exactly - AND REFUTED AT
    THE BOTTOM RUNG 11->13 (research/onset_m11_r28.py, run because the ladder
    was cheap there and nothing had been fitted to it): onset = 13 while
    min span D_4(17)\D_4(13) = 10, witness (2,2,1,5).
    THE REFINEMENT THAT FIXES IT IS THE LAW'S OWN MECHANISM, not a patch: the
    right-hand side must be intersected with WHAT THE TRANSFER CAN EMIT, and
    machine 11's dictionary (73 4-tuples) has no walk emitting (2,2,1,5) at
    all - asserted, not inferred.  Intersected, the minimum is 13 = the onset.
        REFINED FORM 8/8, SIMPLE FORM 7/8 at arity 4.
    AND THE ARITY-3 TEST SETTLES WHICH FORM IS THE LAW.  D_3(M) is the induced
    3-tuple dictionary of D_4(M) EXACTLY (every realised 3-tuple sits inside a
    realised 4-tuple), so a second arity costs no scan
    (research/onset_arity3_r28.py ARITY, logs research/data/r28/onset_arity{2,3}.log):

        step            11->13 13->17 17->19 19->23 23->29 29->31 31->37
        onset (arity 2)     -      -      -     27     41     50     66
        onset (arity 3)    17     14     20     25     36     44     57
        onset (arity 4)    13     15     17     25     31     41     53

        output arity   refined law   simple law   steps with NO onset
             2           3/3 tested     1/3                 3
             3           6/6            2/6                 0
             4           8/8            7/8                 0
             5           3/3            2/3                 0    <- new scans
             6           3/3            2/3                 0    <- new scans
             7           2/2            2/2                 0    (1 step skipped:
                                     the depth-0 lemma genuinely fails at m=7 there)
            TOTAL       25/25          16/25

    ARITY 5 (research/onset_arity5_r28.py, log onset_arity5.log) needed exact
    5-TUPLE dictionaries, which m11..m23 supply in seconds; it keeps the SOURCE
    at the exact 4-tuple dictionary, so the CLOSURE is still order 4 and only
    the OUTPUT size moves - the variation that asks whether the law is about
    the transfer or about the arity the chain happens to consume.  It is about
    the transfer.  Onsets 13, 17, 18 at 11->13, 13->17, 17->19.
    The depth-0 lemma is asserted at arities 2, 3, 4 AND 5.

    so the SIMPLE form's arity-4 record was the luck of rich dictionaries, and
    the intersection with the emissions IS the law.  THE LAW IS
    ARITY-INDEPENDENT.  (31->37 is not testable at arity 3: it needs D_3(41),
    and the m41 shard's induced 3-tuples are span-restricted, i.e. a LOWER
    bound, not the dictionary.  The arity-3 onsets are mostly HIGHER than the
    arity-4 ones: a shorter pattern is pinned by the same order-4 closure for
    more span, which is the direction it must go.)
    AND THE LAW TRACKS THE SCREEN.  The walk screen (C40) moves ONE onset
    (13->17: 15 -> 17).  The law's right-hand side is intersected with the
    transfer's EMISSIONS, and the walk screen changes what an emission is - so
    this is the law's own variable moving, not a re-run.  Under the walk screen
    the refined law is 6 OF 6, with the 13->17 right-hand side moving to 17 in
    step with the onset (research/onset_law_ws_r28.py, log onset_law_ws.log).
    Running total for the refined form: 31 OF 31 across six output arities
    (2,3,4,5,6,7) and two screens.
    THE CAUSAL VERSION IS 8/8 (and implies the refined form): every tuple
    refuted AT the onset span is realised at machine q''.  Witnesses (1,2,3,7),
    (1,5,4,5), (3,2,3,9), (1,5,2,17), (8,2,6,15), (5,5,25,6), (10,2,28,13).
    AND THE SEVENTH STEP OUT OF SAMPLE: nu(41 -> 43), computed from the
    round-27 m41 SHARD alone (span cap 75, inside the shard's exact region;
    no m43 dictionary, no scan, no solver), is 68 - reproducing round 27's
    MEASURED onset(37 -> 41) by a route that never saw it.  Witness
    (5, 36, 2, 25).

THE PARTIAL MECHANISM, and it says what a proof must supply.  Emissions split
by DEPTH (interiors deleted by q'); depth 0 is realised by C40's lemma, so
every refutation needs a walk of >= 5 M-gaps that the order-4 closure admits.
    X_5(M) := min span of a 5-walk whose two 4-windows are realised at M but
              which is not itself realised at M
            = 9 AT EVERY MACHINE 13, 17, 19, 23, WITH THE SAME WITNESS
              (1,2,3,2,1) - and that witness is PHASE-SATURATED at gear 5
              (X = {0,1,3,6,8,9}, X u (X-3) = Z_5), hence zero at every machine
              by K9.
    This explains the UNSCREENED onset EXACTLY: it is 9 at all seven steps.
    Y_5(M) (unrealised AND not phase-saturated) is a lower bound on the
    screened onset, and IT WAS EXTENDED TO m29 THIS ROUND
    (research/y5_m29_r28.py, log research/data/r28/y5_m29.log) by a STREAMED
    full-period machine-29 pass - the round's named open construct, built:

        machine   m13  m17  m19  m23  m29
        X_5         9    9    9    9    9      (always the same witness)
        Y_5        10   17   18   22   30
        onset      15   17   25   31   41
        onset/Y_5 1.50 1.00 1.39 1.41 1.37

    so the multiplicity residue is NOT growing without bound - at the three
    largest machines where both are known the ratio sits in [1.37, 1.41], a
    band of width 0.04.  Y_5(29) = 30 with witness (1, 19, 1, 7, 2).
    THE m29 PASS IS ITSELF GATED TWO WAYS: the cyclic close is asserted (N gaps,
    sum = P, wrap gap = first gap, max = 43 = F(29)), and the exact 5-tuple
    dictionary's INDUCED 4-tuple dictionary is EXACTLY the round-25 full-period
    census (45,854 tuples) - two independent full-period scans agreeing cell for
    cell.  New object on disk: machine 29's exact 5-tuple dictionary,
    research/data/r28/gap_tuples_29_5.csv, 208,668 tuples, ASSERTED
    reverse-closed and ASSERTED to have max span 85 - which INDEPENDENTLY
    CONFIRMS Constructor's round-28 F_5(29) = 85, by a full-period scan against
    their scan-free route.
    AND THE ARITY-5 ONSET AT A BIG STEP.  With D_5(29) on disk the onset test
    runs at OUTPUT ARITY 5 for the step 23 -> 29 (the arity-5 tests had only
    ever reached 17 -> 19, on 37,000-slot periods):
    ONSET(23 -> 29, arity 5) = 30, against 31 at arity 4 and 36 at arity 3, and
    the depth-0 lemma at arity 5 holds there (q' = 29 > 2(m+1) = 12).
    research/onset_arity5_big_r28.py.
    A FULLY STREAMED VERSION (research/y5_stream_r28.py) reaches machine 31 -
    period 3.34e10, 6.23e9 openings, never materialising the gap array - and it
    was VALIDATED AT m29 FIRST, reproducing the in-memory tool exactly (208,668
    tuples, X_5 = 9, Y_5 = 30, induced 4-tuple dictionary EQUAL to the census)
    - AT FOUR DIFFERENT BLOCK SIZES (2^22, 2^23, 2^24, 2^26), which is the
    control that matters for a streamed tool, since the block boundary is
    exactly where a carry bug would hide.
    Two bugs it caught in itself, both by the cyclic-close assertions rather
    than by eyeball: the wrap gap was being compared to the first OPENING
    instead of the first GAP, and the four 5-windows straddling the seam were
    missing (they are now the windows of carry ++ [wrap] ++ head4).
    AND IT REACHED MACHINE 31.  The stream ran over all 33,426,748,355 slots
    (997 blocks, 1,262 s) with the cyclic close asserted (wrap = first = 7,
    max = 58 = F(31)), giving machine 31's EXACT 5-TUPLE DICTIONARY - 636,575
    distinct 5-tuples, whose INDUCED 4-tuple dictionary is EXACTLY the round-25
    full-period census (115,193): a THIRD independent two-scan agreement.  Then

        X_5(31) = 9 (the universal witness again), Y_5(31) = 38 with witness
        (2, 3, 2, 1, 30), against onset(31 -> 37) = 53:  onset / Y_5 = 1.395.

    THE FULL LADDER, six machines:

        machine    m13   m17   m19   m23   m29   m31
        X_5          9     9     9     9     9     9
        Y_5         10    17    18    22    30    38
        onset       15    17    25    31    41    53
        onset/Y_5 1.50  1.00  1.39  1.41  1.37  1.40

    - at the FOUR largest machines the ratio is 1.389, 1.409, 1.367, 1.395, a
    band of width 0.042.  The multiplicity residue is a near-constant FACTOR,
    not a growing gap.  (The 5-tuple dictionary was NOT written to disk at m31:
    the running process predated the emission edit by fifteen minutes.  Cheap to
    redo - the pass is 21 minutes - and it is what an arity-5 onset-law test at
    23 -> 29 would need.)
    Y_5 at m37 remains out of reach by scan (period 1.2e12); the construct for
    it is a lap-phase transfer emitting 5-TUPLES instead of extremal values -
    the same K2 bijection with a different payload - and it is not priced.

### C40. THE DEPTH-0 LEMMA AND THE WALK SCREEN (r28)
Repro: research/onset_anatomy_r28.py (the lemma + the trim),
research/onset_walkscreen_r28.py (the walk screen, ladder + m41).

THE DEPTH-0 LEMMA (proved, three lines, no scan):

        D_m(M)  SUBSET  D_m(M + q')   for every prime q' > 2(m+1),

in particular the realised 4-tuple dictionary only ever GROWS along the ladder.
Proof: a realised m-tuple at opening y_0 has m+1 exposed offsets, so at most
2(m+1) < q' residues are forbidden for the new gear's phase A = (u' - y_0) mod
q'; P(M) is invertible mod q', so A runs over ALL residues across the q' laps -
pick a lap with an admissible A.  Then every point survives and the m+1
openings are still CONSECUTIVE (a new opening between them would be an old one).
    CHECKED at arities 2, 3, 4 at all six exact pairs 13->17 ... 31->37 and at
    D_4(37)|span<=77 subset the round-27 exact m41 shard; and at arities 5, 6,
    7 at the small steps where exact m-tuple dictionaries exist (arity 7 is
    where it FAILS at 11->13, which is the sharpness table below firing as an
    assertion inside research/onset_arity5_r28.py).

AND THE HYPOTHESIS IS SHARP (research/depth0_sharp_r28.py).  Sweeping m upward:

    step     proof covers   first m at which D_m(M) is NOT inside D_m(M+q')
    7->11       m <= 4                 6     witness (2,1,2,2,1,2)
    11->13      m <= 5                 7     witness (3,2,2,1,2,2,3)
    13->17      m <= 7                 8     witness (5,2,2,1,2,2,1,4)
    17->19      m <= 8                 9     witness (2,5,5,2,1,2,5,2,5)

At q' = 17 and 19 the first failure is at EXACTLY the first m the proof does
not cover - the hypothesis is tight, not an artefact - and at q' = 11, 13 it
has slack 1.  Every witness is a dense small-gap pattern, which is what
saturates the new gear's phase set.  (This was pre-registered inside the script
as "the first failure should sit ABOVE 2(m+1), not at it" - CONFIRMED, by 1-2
in m.)

AND IT PAYS AT ONCE.  Of the 874,087 reverse classes of the machine-41 arity-4
screened superset, 145,907 (16.7%, = 291,675 tuples) ARE ALREADY IN D_4(37) and
are therefore YES BY THEOREM - at EVERY span, including the bands round 27
priced at 3.5 s a decision:

    band       classes   free (in D_4(37))   PAID
      1- 60     59,245        59,137          108
     61- 77    110,736        77,181       33,555
     78- 80     30,936         4,207       26,729
     81- 90    122,031         4,954      117,077
     91-100    172,760           424      172,336
    101-110    206,984             4      206,980
    111-118    171,395             0      171,395

so the exact census's remaining PAID population is 728,180 classes, not 874,087,
and the free share collapses with span exactly as F_4(37) = 105 predicts.
On the WALK-SCREENED superset the same table reads 857,186 classes, 145,907
free (17.0%), 711,279 paid - the walk screen's removals land almost entirely
above span 100 (206,984 -> 201,712 and 171,395 -> 161,217), i.e. exactly in the
bands that cost 3-4 seconds a decision.

THE WALK SCREEN - the round-26 screen applied to the right object.  C31 screens
the EMITTED tuple; but every point of the underlying WALK, the deleted
interiors included, is an M-opening, so the whole walk must have an admissible
phase at every gear q <= M.  Screening the walk is SOUND (a realised walk has an
actual phase), STRICTLY STRONGER (it sees obstructions the emission has
forgotten - the universal (1,2,3,2,1) among them) and a PREFIX PRUNE (the
bad-phase set only grows).

    step      truth      raw   emission-screened   WALK-screened   walk+emis
    13->17    1,281    2,283          1,967            1,901        1,901
    17->19    4,489    9,118          7,849            7,601        7,601
    19->23   15,696   66,238         47,623           42,045       42,045
    23->29   45,854  190,091        130,069          121,671      121,671
    29->31  115,193  715,697        471,135          419,990      419,990
    31->37  291,675 2,435,140      1,182,475        1,153,814    1,153,814

    walk-screened == walk+emission at ALL SIX STEPS: THE WALK SCREEN SUBSUMES
    THE EMISSION SCREEN.  Inflation falls 4.054x -> 3.956x at 31->37 and
    3.034x -> 2.679x at 19->23; and at 13->17 it RAISES THE ONSET (15 -> 17).
    Soundness asserted at every step: no realised tuple removed.

DELIVERED TO CONSTRUCTOR: research/data/r28/gap_tuples_41_4_walkscreened.csv,
1,714,020 4-tuples against round 27's 1,747,819 - ASSERTED to be a SUBSET of it
and ASSERTED to contain all 338,855 tuples of the exact m41 shard.  The DFS
pruned 15,186,064 of 102,740,755 nodes.  The gear list is capped at 26 by a
COMPUTED argument, not a guess: s = 2*6^{-1} mod 41 = 14, so kills are >= 14
apart, span <= F_4(41) = 118 allows at most 9 kills, so a walk has at most 13
exposed points and no gear above 26 can saturate.  The 37 -> 41 onset is
UNCHANGED at 68 under the walk screen (a control: the walk screen removes only
things that were refuted anyway in that region).

### C41. THE m41 EXACT SHARD, EXTENDED - AND THE COST LAW RE-PRICED (r28)
Repro: research/m41_shard_r28.py (price / work / merge).  Brief item (b).
Round 27 left the exact m41 arity-4 dictionary COMPLETE AT EVERY SPAN <= 77
with ~1.23M reverse classes and ~4.0e6 core-seconds still to pay.

TWO THINGS CHANGED THE PRICE, BOTH BY THEOREM AND BOTH IN SECONDS.
1. THE DEPTH-0 LEMMA (C40) decides 145,907 of the 874,087 reverse classes YES
   with no solver - 16.7%, at every span.
2. The band table above shows where that free half lives, and it is NOT
   uniform: 99.8% free at span <= 60, 70% at 61-77, 14% at 78-80, 4% at 81-90,
   0.2% at 91-100, 0% above 110 - the collapse tracks F_4(37) = 105 exactly.

So the remaining PAID population is 728,180 reverse classes, and the honest
next-band prices, at round 27's measured per-decision costs, are

    band 78- 80    26,729 paid classes    ~5.1e3 core-s   (this round)
    band 81- 90   117,077 paid classes    ~4.1e5 core-s   (a 23-h job at 5 wk)
    band 91-100   172,336 paid classes    ~6.0e5 core-s

DELIVERED: THE FRONTIER MOVES 77 -> 80.  Four workers, 19,292 paid decisions,
ZERO undecided (Y = 17,303, N = 1,989), 8,610-8,818 s each - inside the 3.5 h
deadline, so no class was left hanging.  Merged with the free half:

    research/data/r28/gap_tuples_41_4_exact_le80.csv  (gate:
    research/shard_gate_r28.py - reverse-closed, max span exactly 80, agrees
    with the round-27 shard CELL FOR CELL below 77, inside the walk-screened
    superset, and contains all 280,911 m37 4-tuples of span <= 80)
    THE EXACT MACHINE-41 4-TUPLE DICTIONARY IS COMPLETE AT EVERY SPAN <= 80:
    395,941 tuples (338,855 at span <= 77 in round 27), 370,263 reverse classes
    carrying a verdict of which 140,525 are FREE by the depth-0 lemma.
    Inflation of the screened superset over that region: 1.0149x.

Above 80 the price is the wall: span 81-90 alone is 117,077 paid decisions at
~3.5 s, i.e. 23 hours at five workers.  The census remains a multi-round object
and is honestly labelled as one.

THE REFUTED-BY-SPAN TABLE, EXTENDED THREE ROWS (research/onset_41_extend_r28.py,
log onset_41_extend.log).  Round 27 could only publish it to span 77 - the old
frontier - and it now runs to 80, with the walk-screened superset alongside:

    span    68  69  70  71  72  73  74  75  76  77   78   79    80
    refuted  4   0  12  28  52  34 136 234 228 210  900 1284  2585
    (identical under the emission screen and the walk screen: the walk screen's
     removals all lie above span 100, so it changes nothing in this region -
     which is itself the control that says the two screens agree where both
     have been checked against exact truth)

    ONSET still 68 under both screens; inflation over span <= 80 is 1.0144x.

### C42. THE PEAK DEPTH OF THE QUALIFYING SPECTRUM (r28, brief item d)
Repro: research/peak_depth_r28.py (log research/data/r28/peak_depth.log).
Round 27 left "Q_8(37;14) = 112 turns over from Q_7 = 114; whether the turnover
point is arithmetic or structural is untested".

THE CHEAP EXACT VEHICLE.  Over a machine's CYCLIC gap array, with R[i] the run
of consecutive gaps >= a starting at i and S the prefix sums,

    Q_j(M; a) = max { S[i+j] - S[i] : R[i+1] >= j - 2 },

so EVERY depth of every machine up to 23 is exact in seconds - no transfer, no
solver, no seed, and the WHOLE profile rather than the first few depths.

    machine  a       2    3    4    5    6    7   ...
       m11   4      11   16   18   20    0    0
       m13   6      16   18   23    0    0    0
       m17   6      25   28   31   32   34    0
       m19   8      31   35   37   38    0    0
       m23  10      39   43   50   55   60    0

    GATED CONTROL (an assertion in the script, not an eyeball): every
    Q_3..Q_7 entry at m11, m13, m17, m19, m23 reproduces C13's published row
    EXACTLY, by a vehicle sharing no code with qualifying_spectrum.py, which
    produced them.

THE ANSWER TO (d), AND IT LOCATES THE TRANSITION.  At every machine <= 23 the
qualifying spectrum is MONOTONE UP TO VACUUM - the peak is the LAST non-empty
depth (5, 4, 6, 5, 6 at m11..m23) and there is NO turnover.  At m31 the peak is
INTERIOR (5 of 7) and at m37 it is interior (7 of 8).  So

    "the peak is terminal"  ->  "the peak is interior"
    happens between machine 23 and machine 31,

and the turnover is not a property of large depth per se: it is the point at
which the qualifying-run structure stops being the binding constraint and the
FLANK structure takes over.  m29 is the one machine in the gap (C13 gives
Q_2..Q_7 = 55 65 68 71 71 71, a PLATEAU, with Q_8 unmeasured); deciding whether
m29 turns over needs one full-period m29 pass (1.078e9 slots, ~1.7 GB for the
prefix-sum array), which is the NAMED CONSTRUCT for this item and was not run
this round - the box was at 48-59 GB of a 63.6 GB commit limit all round.

### C43. F(59) = 161 EXACT - the corpus ladder gains a rung it never had (r28)
Repro: research/f59_pin_r28.py run 7 (band (161, 178] at JMAX = 5, seven range
workers TILING machine 23's period exactly); logs
research/data/r28/f59_pin_161_178_J5_w{0..6}.log.  Lower half re-verified from
the definition by research/f59_lower_r28.py; UPPER half re-asserted end to end
by research/f59_upper_r28.py, which re-reads every round-27 band, re-asserts
each band's tiling, and asserts the bands COVER (178, 260] WITH NO HOLE - the
(178,184] band is the one that closes the gap between 178 and 183 and is NOT in
akill_bands_r27.py's list, so this is a new check, not a re-run.  Brief item
(a).

    ALL SEVEN WORKERS REPORT max over J = 161, i.e. THE BAND IS EMPTY.
    Round 27 had already refuted every span above 178 (four bands, JMAX = 7)
    and exhibited a J = 3 window of span 161.  Therefore

        F(59) = 161   EXACT      (equivalently F(2,59) = 483)

    and every depth J = 2, 3, 4, 5 reports 161 - the record is carried at
    J = 3, i.e. k_win(53 -> 59) = 2.

WHY IT WAS AFFORDABLE THIS ROUND AND NOT LAST.  Round 27 launched this exact
band at JMAX = 7 and KILLED it (no worker reached its first progress stride in
35 minutes).  The depth cap is now a THEOREM (standing rule 34).  Stated with the
index convention checked (rule 5): a word-legal window of J gaps has J-1
INTERIOR OPENINGS, all deleted by one phase of q', i.e. it carries a realised
kill chain of ARITY J-1 (whose WORD has J-2 letters - A_kill counts openings,
not letters).  A_kill(53 -> 59) = 4 EXACT with N_5 = 0 (C36) therefore forces
J - 1 <= 4, so Q*_6 = Q*_7 = 0 and JMAX = 5 is EXHAUSTIVE.  Measured on one identical 20,000-index probe run alone:
JMAX = 5 completes in 57 s, JMAX = 7 does not complete in 600 s.  The full band
then cost 7 workers x ~4.8 h under a loaded box.

THE LOWER HALF IS AN EXHIBITED OBJECT, re-derived this round from the
definition at machine 53 (14 gears, slot by slot):
    k = 2,505,673,933,219,103,747, openings at k + [0, 10, 128, 161],
    gaps [10, 118, 33], all 158 other slots of the span blocked,
    middle 118 = 2q' = the letter 0 (TWO laps of padding).

CONSEQUENCES, all immediate:
1. THE CORPUS LADDER.  F(2,y) had no 53 -> 59 rung; round 26 could only say
   F(59) >= 159 and round 27 bracketed [161, 178].  It is now a value.
2. (D) AT 53 -> 59 with a margin of 43, not 26: 161 <= 204 = F(53) + 59.
3. THE INCREMENT LAW: F(59) - F_2(53) = 161 - 159 = +2, against
   s_min(59) = 20.  The law holds with 18 to spare.
4. CONSTRUCTOR'S Delta BAND SURVIVES AN OUT-OF-SAMPLE STEP.  Their round-27
   finding is Delta_J = Q*_J - F_2 in [-3, +4] uniformly in M and J, measured
   at m11..m41.  At machine 53 - outside their sample - Delta_J = +2 at EVERY
   J = 2..5.  Their uniform-constant claim, not the s_min form, is what the
   new step confirms.
5. THE DELETION LADDER IS NEARLY TIGHT HERE: F_2(53) = 159 <= F(59) = 161,
   slack 2.
6. IT UNBLOCKS THE TENTH RUNG: F_4(43) <= F(43 + three gears) = F(59) = 161,
   so a span cap of 180 makes F_4(43) UNCONDITIONAL (C44).
7. AND IT RETRO-UPGRADES F_2(53) FOR FREE.  C30 recorded F_2(53) = 159 with the
   upper direction CONDITIONAL on the round-26 span cap 200, explicitly because
   "the deletion-ladder cap F_2(53) <= F(59) is unavailable, the corpus F ladder
   stopping at 53".  It is available now: F_2(53) <= F(59) = 161 < 200, so the
   cap excluded nothing and

        F_2(53) = 159 IS NOW UNCONDITIONAL.

   The same argument frees every earlier run whose cap exceeded its own
   deletion-ladder bound once F(59) is known - and it is the reason C44's and
   C45's new values carry no span condition either.

MY PRE-REGISTRATION WAS WRONG, AND IT WAS THE ROUND'S ONLY REAL BET
(research/data/r28/prereg_mechanic_r28.md, written before the band):
    A1 "the band is non-empty, F(59) >= 165"          REFUTED - it is empty.
    A2 "the attaining depth is J = 4"                 REFUTED - it is J = 3.
    A4 "Delta_4, Delta_5 exceed Constructor's [-3,4]" REFUTED - all are +2.
    A3 "Delta_J <= s_min = 20 at every J"             CONFIRMED (all +2).
The reasoning behind A1 was that the k_win census (C13) shows the step record
carried at k_win = 3 at two of four measured steps, and that the realised
2-letter word (20,118) has letters summing to 138 with room for flanks.  What
actually happens is that the deeper words, though REALISED, never occur with
large enough flanks - the same "occurrence count, not span" law this lane
established in C13 and then argued past.  Standing rule 1 again, in its third
costume.

### C43b. THE M1 AUDIT (r28, routed in by the coordinator)
Repro: research/m1_audit_r28.py (log research/data/r28/m1_audit.log), all
assertions passing.  Constructor's M1 - "the legal kill alphabet is
{a, b, q'}" - is REFUTED.  QUESTION: does any mechanic claim lean on it?
ANSWER: NO, and this lane's own data is corroboration of the refutation.
This lane's legality test has always been RESIDUE-based (j5_multi.legal_word
and a_kill both accept any v with v mod q' in {0, +s, -s}, plus the prefix-sum
range condition), and that set is infinite.  At 53 -> 59 the letters actually
enumerated were {20, 39, 59, 79, 98, 118, 138} - FOUR of the seven outside M1's
alphabet - and the letters in the REALISED words (C36, complete levels) are
{20, 39, 59, 98, 118}, of which 98 = q'+(q'-s) and 118 = 2q' are outside it.
The arity-4 carrier (20, 98, 20) has one of the omitted values as its MIDDLE
letter and is exactly what lifts A_kill from 3 to 4.  So round 27's C36 was
already evidence against M1, filed before M1 was refuted.

### C15-UPDATE. THE CORPUS F LADDER (r28)
    y         19   23   29   31    37    41    43    47    53    59
    F(2,y)    75  102  129  174   264   273   309   354   435   483
    F(y)      25   34   43   58    88    91   103   118   145   161
The y = 59 column is NEW AND EXACT (C43), replacing round 26's lower bound
(>= 159) and round 27's bracket [161, 178].  It is the first ladder value the
project computed rather than looked up, and it was computed on machine 23's
period - a period ratio of 5.3e11 to machine 59's.
SCOPE, stated as always: the upper direction rests on the round-27 bands above
178, which were run with span caps 184/194/204/260; the lower direction is an
exhibited machine-53 window re-verified from the definition this round.

### C44. F_2(43) = 116, F_3(43) = 125, F_4(43) = 132 - THE TENTH RUNG'S
###      SHOPPING LIST, ALL EXACT AND ALL UNCONDITIONAL (r28)
Repro: research/j5_multi.py 23 29,31,37,41,43 47 seed102 180 4 1 plain LO HI
(floor-1 lap-phase transfer, r = 5), three range workers TILING
[0, 7,952,175); logs research/data/r28/fj43_w{0,1,2}.log, 660-720 s each.
Routed in by the coordinator from Constructor's round-28 filing: their
spectrum-depth certificate needs F_2(43), F_3(43), F_4(43) and one emptiness
certificate at J_max(43)+1.

    J        Q_J(43; 1)     status
    2           116         = F_2(43), NEW AND EXACT (standing entry: <= 118)
    3           125         = F_3(43), the KNOWN exact value - TWO-SIDED ANCHOR
    4           132         = F_4(43), NEW AND EXACT (standing entry: nothing)

    max over J = 132  vs budget F(43) + 47 = 150  ->  CERTIFIES

NOTHING HERE IS SPAN-CONDITIONAL, and that is a direct dividend of C43.  The
run's cap is 180 and the deletion-ladder caps are F_2(43) <= F(47) = 118,
F_3(43) <= F(53) = 145, F_4(43) <= F(59) = 161 - the last of which only became
a number this round.  180 sits above all three, so no window was excluded by
the cap.  The seed 102 = F(43) - 1, and F_2(43) >= F(43) = 103, so the seed
hides nothing either.

THREE CONTROLS, none of which the tool was told about:
 (a) J = 3 returns 125 = the known F_3(43), seeded 23 below it, so the run had
     to FIND it as well as refute everything above.
 (b) The J = 3 witness verified at machine 43 is k = 1,595,441,702,157,105 with
     gaps [67, 28, 30] - the exact REVERSE of C11's round-24 SAT witness
     [30, 28, 67] at a different address: the mirror law, unprompted.
 (c) Two of the three range workers independently report the same 116/125/132.
All three maxima re-verified AT MACHINE 43 from the definition
(research/qstar_witness_r26.py --nolegal, 12 gears, slot by slot):
    F_2: k = 2,161,962,392,309,552  gaps [31, 85]         114 others blocked
    F_3: k = 1,595,441,702,157,105  gaps [67, 28, 30]     122 others blocked
    F_4: k =   280,183,736,276,020  gaps [18, 24, 8, 82]  128 others blocked

THE EMPTINESS CERTIFICATE IS FREE - NO RUN AT ALL.  A word-legal 5-window
carries a realised kill chain of ARITY 4 at the step 43 -> 47, and
A_kill(43 -> 47) = 3 EXACT by full-period decision (C10/C22) means N_4 = 0, so

    Q*_5(43; legal for 47) is EMPTY BY THEOREM,  i.e. J_max(43) = 4.

This is the same argument that made JMAX = 5 exhaustive for the F(59) pin
(standing rule 34): the completed arity level IS the depth cap.

CONSEQUENCE FOR CONSTRUCTOR: their criterion F(M+q') <= max_{2<=J<=J_max} F_J(M)
now reads F(47) <= max(116, 125, 132) = 132 at the tenth rung, against the
budget F(43) + 47 = 150 - so (D) AT 43 -> 47 IS CERTIFIED with margin 18, from
machine 43's spectrum alone, with no word list, no flank envelope and no
realisability oracle.  (The corollary F(47) <= 132 is weaker than the known
exact 118, as it must be - the criterion is a certificate, not a computation.)
CORRECTION TO C11: the entry "F_2(43) <= 118" is superseded by F_2(43) = 116,
and "F_4(43)" had no entry at all.

### C45. F_5(41) = 128 EXACT - Constructor's other "small job" (r28)
Repro: research/j5_multi.py 23 29,31,37,41 43 seed118 165 5 1 plain LO HI
(r = 4), three range workers TILING [0, 7,952,175); logs
research/data/r28/f5_41_w{0,1,2}.log, ~380 s each.  Routed in by the
coordinator; Constructor called F_5(37) and F_5(41) "the same small job" and
they were right, but only because F(59) landed first.

    F_5(41) = 128 EXACT, UNCONDITIONAL.

The cap 165 sits above the deletion-ladder cap F_5(41) = F_{4+1}(41) <=
F(41 + four gears) = F(59) = 161 - a bound that DID NOT EXIST before C43, since
the corpus ladder stopped at 53.  Seed 118 = F_4(41), and F_5 >= F_4 always, so
the seed hides nothing.  (F_5(37) = 113 was already exact in C11; the standing
entry for F_5(41) was nothing at all.)

THE TWO MAXIMISERS ARE AN EXACT MIRROR PAIR, from two processes sharing no
state: 4,834,937 + 32,347,080 = 37,182,017 = P(23) - 128 (C18's k -> -k).
Verified AT MACHINE 41 from the definition, slot by slot:
    k = 33,044,111,735,742  gaps [10, 51, 2, 50, 15]  123 others blocked
    k = 17,664,265,518,665  gaps [15, 50, 2, 51, 10]  123 others blocked
AND THE SHAPE IS INHERITED: round 27's F_4(41) = 118 maximisers were
[51,2,50,15] and [15,50,2,51] at k = 33,044,111,735,752 and
17,664,265,518,665 - the SAME addresses (the second identical, the first ten
slots along).  So the depth-5 record is the depth-4 record with ONE MORE GAP
OF 10 prepended, not a different configuration: F_5(41) = F_4(41) + 10.

FREE BONUS FOR THE NINTH RUNG: the run's own budget line is F(41) + 43 = 134,
and 128 <= 134, so Constructor's spectrum-depth criterion certifies (D) at
41 -> 43 using the UNRESTRICTED depth-5 value - i.e. that rung does not need
the Q*_5(41) emptiness certificate at all, it is robust to the J_max choice.

### C46. F_3(47) = 145 EXACT AND UNCONDITIONAL (r28)
Repro: research/j5_multi.py 23 29,31,37,41,43,47 53 seed144 165 3 1 plain LO HI
(r = 6), three range workers TILING [0, 7,952,175); logs
research/data/r28/f3_47_w{0,1,2}.log, ~600 s each.

    F_3(47) = 145.  The standing entry (C11) was "F_3(47) >= 145 (<= 263)".

The cap 165 sits above the deletion-ladder cap F_3(47) <= F(47 + two gears)
= F(59) = 161 - again a bound that did not exist before C43 - so nothing here
is span-conditional.  Seeded at 144, so the run had to FIND 145 as well as
refute everything above it, and all three workers report the same row
(Q_2 = 144 at the seed, Q_3 = 145).

THE CONTROL IS AS GOOD AS THIS LANE HAS EVER GOT: the witness translates to
machine-47 address k = 36,068,193,854,725,102 with gaps [28, 33, 84] - which is
C11's round-24 witness, at the SAME ADDRESS, found by a completely different
vehicle (that one was the endpoint-law-pruned covering search f3_47_prune; this
one is the lap-phase transfer from machine 23).  Re-verified slot by slot at
machine 47: 4 openings at k + [0, 28, 61, 145], all 142 other slots blocked.

### C11-UPDATE. THE F_j SPECTRA AFTER ROUND 28
    machine   F_1  F_2  F_3  F_4  F_5  F_6
    13         11   16   23   26   28   31
    17         18   25   28   33   35   40
    19         25   31   35   38   47   50
    23         34   39   50   58   65   77
    29         43   55   65   70   85   90
    31         58   68   85   90   92   97
    37         88   90   97  105  113  120
    41         91  103  110  118  128    -     <- F_5 NEW this round (C45)
    43        103  116  125  132    -    -     <- F_2, F_4 NEW this round (C44)
    47        118  134  145    -     -    -     <- F_3 pinned this round (C46)
    53        145  159    -    -     -    -
    59        161   ?     -    -     -    -    <- F_1 NEW this round (C43)
F_2(11) = 11 and the m11..m29 F_2 column were independently re-derived this
round from the cyclically closed period (research/onset_ladder_r28.py's
machinery) and agree with the corpus row cell for cell.

### C47. THE WITNESS GATE (r28)
research/witness_gate_r28.py (log research/data/r28/witness_gate.log).  Every
exhibited window this round produced, re-checked AT ITS OWN MACHINE from the
definition - the J+1 offsets are openings, every other slot of the span is
blocked, gear by gear - importing nothing from the tools that found them:

    F(59) >= 161   m53  gaps [10,118,33]        158 others blocked
    F_2(43) = 116  m43  gaps [31,85]            114
    F_3(43) = 125  m43  gaps [67,28,30]         122
    F_4(43) = 132  m43  gaps [18,24,8,82]       128
    F_5(41) = 128  m41  gaps [10,51,2,50,15]    123
    F_5(41) mirror m41  gaps [15,50,2,51,10]    123
    F_3(47) = 145  m47  gaps [28,33,84]         142

plus the mirror identity 4,834,937 + 32,347,080 = P(23) - 128 asserted.  ALL
ASSERTIONS PASSED.

## Standing rules (round-28 additions)

34. A DEPTH CAP CAN BE A THEOREM, NOT A BUDGET CHOICE.  Rule 32 said "ask what
    depth the question needs"; this says the answer is sometimes COMPUTABLE.
    A word-legal window of J gaps has J-1 INTERIOR OPENINGS deleted by one
    phase, i.e. a realised kill chain of ARITY J-1 (its word has J-2 letters -
    A_kill counts OPENINGS, and getting that index wrong is rule 5's trap), so
    Q*_J = 0 whenever J - 1 > A_kill; with A_kill(53->59) = 4 closed in r27,
    JMAX = 5 is EXHAUSTIVE for the F(59) pin, not a scope choice.  Measured on
    one identical 20,000-index probe, run alone: JMAX = 5 completes in 57 s,
    JMAX = 7 does not complete in 600 s.  Round 27 launched the same band at
    JMAX = 7 and had to kill it.  Before choosing a depth, look for a COMPLETED
    ARITY LEVEL that fixes it.
35. SCREEN THE OBJECT THE SEARCH ACTUALLY WALKS, NOT THE OBJECT IT EMITS.  The
    phase-saturation screen had been applied to the emitted tuple for two
    rounds; the transfer walks a longer object whose deleted interiors are also
    openings, so the obstruction belongs there.  Sound, strictly stronger,
    SUBSUMES the emission screen at all six steps, and it is a prefix prune
    rather than a post-filter.  General form: when a construct emits from a
    richer intermediate object, put the arithmetic obstruction on the
    intermediate.
36. BEFORE PAYING A SOLVER FOR "IS THIS OLD OBJECT STILL THERE?", ASK WHETHER
    IT MUST BE.  The depth-0 lemma is three lines and it decides 16.7% of a
    1.4M-decision census - a population this lane had been pricing at 3.5 s a
    head.  Rule 29's pigeonhole layer has a twin: a MONOTONICITY layer.
37. COUNT MY OWN PROCESSES AGAINST THE COMPUTE POLICY, AND WATCH COMMIT, NOT
    JUST FREE RAM.  Mid-round I had 12 compute processes up while other lanes
    had 9 plus two Lean builds holding 2.4 GB; committed memory reached 59.3 of
    the box's 63.6 GB limit, total CPU sat at 42%, and my own headline job was
    running at a quarter of its measured solo speed.  Rule 30 said measure
    utilisation; this adds: measure \Memory\Committed Bytes against the commit
    limit, and when the box is loaded the fix is to kill MY OWN lowest-value
    job, not to wait.

## Retracted / corrected (round-28 additions)

R27. MY ROUND-28 PRE-REGISTRATION - THE HEADLINE BET LOST, AND THREE OF THE
    FOUR ONSET FORMULAS WERE MINE TO LOSE
    (research/data/r28/prereg_mechanic_r28.md, written before any band).
    A1 "the band (161,178] is non-empty; F(59) >= 165"  REFUTED - it is EMPTY
       and F(59) = 161.  My reasoning was the k_win census (the step record is
       carried at k_win = 3 at two of four measured steps) plus the realised
       2-letter word (20,118) leaving room for flanks.  What the machine does
       instead is what THIS LANE established in C13 and then argued past: the
       envelope follows OCCURRENCE COUNT, not span - the deeper words are
       realised but never occur with big enough flanks.  Standing rule 1 in its
       third costume.
    A2 "the attaining depth is J = 4"                   REFUTED - it is J = 3.
    A4 "Delta_4, Delta_5 exceed Constructor's [-3,+4]"  REFUTED - every
       Delta_J is +2, so their uniform band survives an out-of-sample machine.
    A3 "Delta_J <= s_min(59) = 20 at every J = 2..5"    CONFIRMED.
    D1 "onset = F_2 of the machine one below" (F_2(31) = 68)   REFUTED
    D2 "onset = 2F two machines below"        (2F(23) = 68)    REFUTED
    D3 "onset/F(M) is the constant 0.773"                      REFUTED
       - all three fitted the single round-27 data point and none survives a
       second step.  Fitting a closed form to one measurement is the same
       error as extrapolating a per-step share (rule 1); I made it three ways
       at once and registered all three so they could be shot.
    D4 "none of the three works; the onset sits closer to F(M) than to F_2(M)"
       CONFIRMED on both clauses - the three fail, and |onset - F(M)| <
       |onset - F_2(M)| at 6 of the 8 steps (it fails at the two smallest
       machines, m11 and m13).
R29. I RECORDED A PROCESS KILL THAT NEVER HAPPENED.  Seeing no machine-31
    python process and no emitted CSV, I wrote into this log that the m31
    streamed pass "was killed before it wrote the CSV or computed Y_5(31)".
    It was not: it exited NORMALLY, having already printed Y_5(31) = 38; my
    liveness check landed after it finished, and the CSV is missing only
    because the running process predated the emission edit by fifteen minutes.
    Corrected in place.  Standing rule 23 in a new costume - re-read the log
    before quoting a verdict about a process, and "it died" is a verdict.
    (The OTHER kill in this round's negatives - the seven F_2(59) workers - IS
    real: they left no completion line, their launcher died with them, and the
    partial logs stop mid-range.)

R28. AN INDEX SLIP IN MY OWN PROSE, CAUGHT ON REVIEW AND CORRECTED IN PLACE.
    The depth-cap argument (standing rule 34) was written as "a word-legal
    window of J gaps carries a realised (J-1)-LETTER kill word".  It does not:
    it carries J-1 INTERIOR OPENINGS deleted by one phase - a kill chain of
    ARITY J-1 - and that chain's WORD has J-2 letters.  A_kill counts OPENINGS
    (C10's own definition: "N_k counts TUPLES"), so the inequality that matters
    is J - 1 <= A_kill, which is what every script actually computed and what
    the gate actually asserts (its R27_WORDS table is keyed by ARITY and the
    empty level is arity 5).  NOTHING COMPUTED IS AFFECTED - the JMAX = 5 cap,
    the F(59) pin and C44's emptiness certificate are all unchanged - but the
    sentence was wrong in four documents and is now right in all of them.
    This is standing rule 5 (check the index convention before quoting) applied
    to my own text rather than to another lane's.

    E1 "the peak depth of Q_J is non-decreasing in M"  REFUTED by my own
       exact table: 5, 4, 6, 5, 6 at m11..m23 (then 5 at m31, 7 at m37).  It is
       not monotone and it was never going to be - the peak is terminal below
       m31 and interior from m31 on, which is a different statement about a
       different regime.

## Open watches and checkpointed jobs (round-28 additions)

WATCHES
- THE ONSET LAW IS MEASURED, NOT PROVED (26 of 26 across four output arities
  and two screens, plus one out-of-sample prediction).  What would settle it:
  bound, for the smallest over-generated span, the deletion budget of an
  order-4-closure walk against the deletion budget of ONE extra gear.  The
  cheapest new evidence is the 41 -> 43 rung, which needs D_4(43) - i.e. the
  m41 shard extended, or an m43 dictionary.
- Y_5(M) IS NOW COMPUTED AT m29 AND m31 (30 and 38, by streamed full-period
  passes over 1.08e9 and 3.34e10 slots, C39), and the residual ratio onset/Y_5
  is 1.389, 1.409, 1.367, 1.395 at m19, m23, m29, m31.  m37 is NOT reachable by
  scan (period 1.2e12); the construct that would reach it is a lap-phase
  transfer emitting 5-TUPLES rather than extremal values - the same K2
  bijection, a different payload - and it has not been priced.
  ALSO OUTSTANDING AND CHEAP: machine 31's 5-tuple dictionary was computed but
  NOT written to disk (the running process predated the emission edit), so a
  21-minute re-run would bank it - and it is exactly what an arity-5 onset-law
  test at 23 -> 29 needs, the arity-5 onset there being already measured at 30.
- THE m41 EXACT 4-TUPLE CENSUS: COMPLETE AT EVERY SPAN <= 80 (was 77).  The
  remaining paid population is 711,279 reverse classes on the walk-screened
  superset; span 81-90 alone is 117,077 decisions at ~3.5 s = 23 h at five
  workers.  Still a multi-round object.  research/m41_shard_r28.py work/merge
  resumes from the per-worker logs and skips the depth-0-free half.
- DOES m29's QUALIFYING SPECTRUM TURN OVER?  The "peak terminal -> peak
  interior" transition happens between m23 and m31 and m29 is the machine in
  the gap (C13 gives a PLATEAU 71, 71, 71 with Q_8 unmeasured).  One
  full-period m29 pass decides it; the prefix-sum array is ~1.7 GB, which is
  the reason it was not run on this box this round.
- THE WALK SCREEN should replace the emission screen everywhere the transfer is
  used.  research/data/r28/gap_tuples_41_4_walkscreened.csv is the drop-in
  tighter superset (1,714,020 vs 1,747,819, asserted subset, asserted to
  contain the exact shard).

## Round 29

GATE (one command, clean process, imports nothing from the tools that produced
the numbers):
    uv run python research/gate_mechanic_r29.py      -> ALL ASSERTIONS PASSED
    uv run python research/crt_slots_r29.py          -> ALL ASSERTIONS PASSED
    uv run python research/chain_depth_r29.py gate   -> ALL ASSERTIONS PASSED
    uv run python research/witness47_r29.py          -> ALL ASSERTIONS PASSED
Pre-registration: research/data/r29/prereg_mechanic_r29.md, written before any
round-29 script existed.  Persistent results: research/r29_results.txt.
Gate log: research/data/r29/gate.log.  New files this round:
research/crt_slots_r29.py, research/fj47_r29.py, research/witness47_r29.py,
research/criterion_margin_r29.py, research/chain_depth_r29.py,
research/gate_mechanic_r29.py, research/r29_results.txt.
JOB COMPLETION: finished and recorded - the seed-174 band run (64/64 shards),
the machine-31 and machine-37 chain-depth passes, every gate.  Stopped and
recorded as stopped - the seed-145 F_J(47) run (12/64 shards, resumable), the
word-legal Q*_J(47) run (0 shards, killed on the control argument), the six
machine-41 workers (423/1147 chunks, resumable).  Nothing left running.

### C48. THE CRT SLOTS FOR FORMALIST (r29, brief item a)

Repro: research/crt_slots_r29.py (log research/data/r29/crt_slots.log); gate
section A.  Formalist's verdict 36 - "realisability is cheap in the kernel once
a witness is a SLOT and not a phase vector" - applied to every F_2 record this
project owns.  Each line below is an adjacent OPENING TRIPLE of its own
machine, re-derived here from the definition (slot k blocked by gear q iff
k = +-6^{-1} mod q), with the two neighbours outside the window located so the
triple is pinned as maximal rather than merely exhibited:

  F_2(41) = 103  m41  y = 21,157,523,372,970
                 openings y, y+28, y+103   flanks (7, 4)    101 interior blocked
  F_2(53) = 159  m53  y = 327,666,424,664,536,738
                 openings y, y+77, y+159   flanks (6, 3)    157 interior blocked
  F_2(59) = 173  m59  y = 307,199,471,342,884,027,665
                 openings y, y+100, y+173  flanks (13, 4)   171 interior blocked
  F_2(59) = 173  m59  y = 13,260,587,016,151,412,007
                 openings y, y+73, y+173   flanks (4, 13)   171 interior blocked

Each y is asserted to lie in [0, P(machine)); P(41) = 50,708,377,254,535,
P(53) = 5,431,526,412,865,007,455, P(59) = 320,460,058,359,035,439,845.
FIVE CONSECUTIVE OPENINGS, which is the form a kernel `AdjPair` chain wants:
  m41  21157523372963 / ...70 / ...98 / 21157523373073 / ...77   word [7,28,75,4]
  m53  327666424664536732 / ...738 / ...815 / ...897 / ...900    word [6,77,82,3]
  m59  307199471342884027652 / ...665 / ...765 / ...838 / ...842 word [13,100,73,4]
  m59  13260587016151412003 / ...007 / ...080 / ...180 / ...193  word [4,73,100,13]
BLOCKER CERTIFICATE, emitted per witness: for every other slot of the span, the
SMALLEST gear that blocks it, so "all 171 interior slots are blocked" becomes
171 single modular equalities on numerals rather than one existential.  The
residue vector y mod q for every gear is emitted with it.

THE TWO m59 SLOTS ARE AN EXACT MIRROR PAIR, and the flanks mirror too:
    y_A + y_B + 173 = P(59) = 320,460,058,359,035,439,845
    gap words [100,73] / [73,100] and flank pairs (13,4) / (4,13).
Pre-registered (A2, A3) and confirmed.  Third instance of the mirror law
producing a free control on a pair of maximisers (C45's F_5(41) pair, C44's
F_3(43) reversal, now this).

WHAT THE SCAN'S COMPLETENESS CLAIM RESTS ON - the honest separation the brief
asked for.  None of these three numbers comes from a period scan; the machine is
never built.  They come from the lap-phase transfer (j5_multi.py), whose
completeness is a CRT bijection: an address k in [0, P(y)) is determined by
(k mod P(23), k mod q_1, ..., k mod q_r), the machine-y openings inside a window
of machine-23 openings are exactly the machine-23 openings that no new gear's
phase deletes, so every J-window of machine y is exactly one pair (machine-23
window, phase tuple) and conversely.  The scan walks EVERY start opening of
machine 23's period (7,952,175, tiled across workers, cyclically closed) and at
each one every distinct KILL SET of phase tuples - one representative phase per
distinct kill set is exact, because admissibility depends on the phase only
through which interiors it removes.  Two cuts make it finite, and only one of
them bites at J = 2:
  (i)  THE SPAN CAP.  Only windows of span <= C are examined, so what is proved
       is "no 2-window of machine y has span in (v, C]".  That is the exact
       maximum only when F_2(y) <= C is known independently.  At y = 41 it is
       (deletion ladder F_2(41) <= F(43) = 103); at y = 53 it is (F_2(53) <=
       F(59) = 161 < 200 - C43's dividend, which is what retired C30's span
       condition); AT y = 59 IT IS NOT, because the corresponding bound is
       F_2(59) <= F(61) and F(61) is not a number this project owns.  The
       round-28 run used C = 220 (research/f2_59_r28.py).
  (ii) THE DEPTH CAP JMAX.  Irrelevant at J = 2: a 2-window has one interior
       opening and no depth question at all.  (It is what capped C43's F(59)
       band, via A_kill.)
So, stated as Formalist should transcribe them: F_2(41) = 103 and F_2(53) = 159
are EXACT AND UNCONDITIONAL; F_2(59) SPLITS - ">= 173" is unconditional and is
what the slot above carries into the kernel, while "<= 173" is conditional on
"no 2-window of machine 59 has span in (173, 220]", which is what the round-28
scan proved and which is NOT a theorem about machine 59.  The kernel can carry
the lower half today; the upper half must travel with its span condition until
F(61) exists.

### C49. RUNG ELEVEN (47 -> 53): THE SPECTRUM-PLUS-DEPTH CERTIFICATE FAILS,
### AND THE FAILURE IS A_kill's, NOT MACHINE 47's (r29, brief item b)

Repro: research/fj47_r29.py (sharded, crash-proof, resumable) driving
j5_multi.py 23 29,31,37,41,43,47 53 seed<S> 290 6 1 plain LO HI; logs
research/data/r29/fj47/ and research/data/r29/fj47_s174/; gate section D.
Inputs already on record: F(47) = 118, F_2(47) = 134 (C25), F_3(47) = 145
(C46), A_kill(47 -> 53) = 5 EXACT with N_6 = 0 (C23), so J_max(47) = 6 and
Q*_7(47) is EMPTY BY THEOREM (standing rule 34) - the emptiness certificate the
criterion needs is FREE, exactly as at 43 (C44).

    F_4(47) in [154, 174],  F_5(47) in [167, 174],  **F_6(47) = 177 EXACT**
    max over J = 2..6  =  177   vs budget F(47) + 53 = 171   ->  FAILS by 6

F_6(47) = 177 IS EXACT AND UNCONDITIONAL: the seed-174 band run covers 100% of
machine 23's period (64 of 64 shards, 7,952,175 start indices, exact tiling
asserted in gate section D) with span cap 290, and 177 is the largest span it
finds at any depth.  F_4 and F_5 are bracketed rather than pinned because the
seed-174 run reports them only as "<= 174" and the seed-145 run (which would
pin them) reached 18.7% before being stopped - and they are NOT NEEDED, since
F_J is non-decreasing in J and the criterion consumes F_{J_max} alone.

THE MAXIMISER IS A SLOT, NOT A PHASE VECTOR (research/witness47_r29.py, gate
section E).  The transfer reports k = 26,216,680 on machine 23 with phases
(3,21,29,26,26,27) for gears (29,31,37,41,43,47) and marks (5,10,16,17,19); CRT
on t (t = -c_q * P(23)^{-1} mod q at each new gear) lifts it to

    MACHINE 47, slot x = 46,615,676,895,423,125  (P(47) = 102,481,630,431,415,235)
    seven consecutive openings at offsets [0, 42, 70, 103, 107, 115, 177]
    gap word [42, 28, 33, 4, 8, 62],  span 177,  all 171 other slots blocked

re-checked at machine 47 from the definition, importing nothing from the tool
that found it.  177 > 171, therefore
**max_{2<=J<=J_max(47)} F_J(47) > F(47) + 53 and Constructor's round-28
spectrum-plus-depth certificate does NOT close 47 -> 53.**

SPAN CAP: 290 = 2 F_3(47), at or above the SUBADDITIVITY ceiling of every depth
in range (F_{a+b} <= F_a + F_b gives F_4 <= 2F_2 = 268, F_5 <= F_2 + F_3 = 279,
F_6 <= 2F_3 = 290).  Nothing here is span-conditional.

AND THE FAILURE IS NOT AN ACCIDENT OF MACHINE 47.  Since F_J(M) is
non-decreasing in J, the criterion's margin at a step is exactly
    margin(M -> q')  =  F(M) + q'  -  F_{A_kill+1}(M),
so it is a statement about the DEPTH the fuel census allows.  Tabulated exactly
(research/criterion_margin_r29.py):

    step      A_kill  J_max  F_Jmax(M)  budget  margin  verdict
    13 -> 17     2      3        23        28     +5    CERTIFIES
    17 -> 19     2      3        28        37     +9    CERTIFIES
    19 -> 23     3      4        38        48    +10    CERTIFIES
    23 -> 29     2      3        50        63    +13    CERTIFIES
    29 -> 31     4      5        85        74    -11    FAILS
    31 -> 37     4      5        92        95     +3    CERTIFIES
    37 -> 41     3      4       105       129    +24    CERTIFIES
    41 -> 43     3      4       118       134    +16    CERTIFIES
    43 -> 47     3      4       132       150    +18    CERTIFIES
    47 -> 53     5      6       177        171     -6    FAILS

    by A_kill:  2 -> {+5,+9,+13}   3 -> {+10,+16,+18,+24}   4 -> {-11,+3}
                5 -> {-6}
    F-ladder cost per level (F_{J+1} - F_J, exact): m37 [2,7,8,8,7],
    m41 [12,7,8,10], m43 [13,9,7], m47 [16,11,...]

EVERY step with A_kill <= 3 certifies; both failures and the one +3 squeaker are
the steps with A_kill >= 4.  MECHANISM, not trend: each extra unit of A_kill
admits one more level of the F ladder, which costs 7-16 units, while the budget
only gains q' - q'_prev (4 to 6 at this end of the ladder).  The criterion is
therefore a FINITE-DEPTH criterion in the literal sense - it works exactly while
the fuel census keeps J_max small, and the fuel census is arithmetic-selected
(C10), not monotone.  THIS IS THE HONEST SCOPE OF CONSTRUCTOR'S R81: it is not
"8 of 9 and one exception", it is "every A_kill <= 3 step, and it will fail
again at the next A_kill >= 4 step".

WHAT CLOSES THE RUNG INSTEAD, and it needs no run: the ATTAINMENT THEOREM (R68,
proved two-sided in round 26) says F(M+q') = max_J Q*_J(M) with Q*_J the
WORD-LEGAL spectrum, and Q*_J <= F_J is exactly where the criterion loses its
30 units at 29 -> 31.  At 47 -> 53 the same theorem plus the corpus value
F(53) = 145 gives max_J Q*_J(47) = 145 <= 171, margin 26.  That is a
certificate that uses machine 53's own record; the INDEPENDENT computation of
max_J Q*_J(47) from machine 23 was priced and NOT RUN (below).

COSTS, AS OP COUNTS (benchmark protocol), on one identical 20,000-start-index
probe - windows walked is 517,183 in every row, so the row is the price of the
phase expansion alone:

    configuration                     phase-expanded   wall (secondary)
    seed 171, floor 1  (band only)          111             1 s   [alone]
    seed 145, floor 1                     3,800            10 s   [alone]
    seed 133, floor 1                    14,949           225 s   [alone]
    seed 145, word-legal for 53           3,800            32 s   [5 of my
    seed 144, word-legal for 53           4,596            46 s    procs up]

Two readings.  (a) THE SEED IS THE PRICE: 12 units of seed (145 -> 133) is 3.9x
the expansions and 22x the wall - the per-expansion cost rises too, because the
surviving windows are deeper.  (b) WORD-LEGALITY IS NOT FREE: at IDENTICAL
expansion count (3,800) the legal check costs 3.2x, which is `feasible_marks`
searching at a = 18 instead of a = 1 plus `legal_word` itself.  Extrapolated
prices for the whole period (7,952,175 start indices): 400 s at seed 171,
4,000 s at seed 145, 89,500 s at seed 133, 12,700 s word-legal at seed 145.

PER-ITEM STATUS, as the brief asked:
    F_2(47) = 134   EXACT-UNCONDITIONAL, already on record (C25); the deletion
                    ladder F_2(47) <= F(53) = 145 confirms the cap it was
                    computed under.
    F_3(47) = 145   EXACT-UNCONDITIONAL (C46).
    F_6(47) = 177   EXACT-UNCONDITIONAL, NEW (seed-174 band run, 64/64 shards,
                    100% of machine 23's period, span cap 290)
    F_4(47) in      EXHIBITED >= 154 and >= 167 by the seed-145 run (18.7%
      [154,174]     coverage, stopped to fund item (c)); <= 174 by the band
    F_5(47) in      run at 100%.  Exact values NOT COMPLETED; price of the
      [167,174]     remaining 81.3% of the seed-145 run ~3,250 s of single-core
                    walk.  Not needed for the rung.
    max_J F_J(47)   = 177 EXACT, > 171 -> the criterion fails by 6.
    Q*_J(47)        NOT ATTEMPTED.  Priced at 12,700 s single-core (3.5
                    core-hours).  Killed deliberately mid-round after the
                    attainment theorem made it a CONTROL rather than a
                    decision: max_J Q*_J(47) = F(53) = 145 follows from R68 and
                    a corpus value already on record.
    Q*_7(47) = 0    EXACT-UNCONDITIONAL BY THEOREM, no run (A_kill = 5).

### C50. THE ANCHOR-235 CHAIN DEPTH AND THE RECORD LAW AT 31 / 37 / 41 (r29,
### brief item c) - AND D_g = A_kill(M -> g) IS AN IDENTITY, NOT A COINCIDENCE

Repro: research/chain_depth_r29.py (build | gate | run g | merge); logs
research/data/r29/chain{31,37,41*}.log; gate sections B and C.
anchor-235 section 9f defines, for the layer g on M = {5 .. prev(g)},
    D_g   = longest run of consecutive M-openings whose slot residues mod g all
            lie in ONE two-class set {r, r+d},  d = 2*6^{-1} mod g
    F_g   = max over such runs of (gap before) + (run span) + (gap after)
and computes both on ONE LOWER PERIOD, because the g copies of the lower period
realise every deletion phase exactly once.  chain_depth.py stops at g = 29
(it materialises the lower period as an array).  This round carries it to 31,
37 and 41 with NO full-period array beyond machine 29.

THE VEHICLE.  (1) Machine 29's opening list is built once as 214,708,725 uint32
entries plus three uint8 residue arrays (1.5 GB on disk, memory-mapped, never
in a process's own heap).  (2) The {5..31} lower sequence is streamed as 31
chunks of it and the {5..37} lower sequence as 31 x 37 = 1147 chunks, each chunk
being the machine-29 list under one or two residue filters - so the 1.24e12-slot
period with 2.18e11 openings is walked without ever holding it.  (3) THE PHASE
IS NOT LOOPED OVER: mapping residues by d^{-1} turns "{r, r+d} for some r" into
"two adjacent values", so one rolling max/min over a length-L window decides all
g phases at once and the winning phase is read back as r = s*d mod g.  Cost per
level is one pass, not g passes.  (4) Cyclic closure is EXACT - the lower
sequence's head is appended at position + P_lower with the residues of the
SHIFTED slot, so the copy-to-copy phase change is carried, not ignored (this is
what chain_depth.py's X2 = [X, X+P] already does correctly, and getting it wrong
would silently lose every record that straddles a copy boundary).

VALIDATION FIRST: the vehicle reproduces chain_depth.py's published row exactly
at all seven rungs it can reach - D = 2,1,2,2,2,3,2 and F = 5,7,11,18,25,34,43
at g = 7..29 (chain_depth.py prints F-1).  Then:

    g   lower      lower period   openings      D_g  record   corpus F(g)
    31  {5..29}     1.078e9 slots  214,708,725   4     58        58   MATCH
    37  {5..31}     3.343e10       6,226,553,025 4     88        88   MATCH
    41  {5..37}     1.237e12     217,929,355,875 3     91        91   MATCH

    machine 31 (30 s, EXACT full lower period, one chunk)
      L   merged  before span after   phase r  copy j  slot of survivor before
      1     55      30    0    25       3       30     32,481,956,680
      2     58      18   10    30      14       20     21,844,264,615   <- record
      3     55       2   31    22       7        1      1,495,243,370
      4     55       7   41     7      29       12     13,159,557,555

    machine 37 (898 s, EXACT full lower period, 31 streamed chunks)
      L   merged  before span after   phase r  copy j  slot of survivor before
      1     68      33    0    35      16        8    273,663,783,992
      2     85      18   37    30      33       29    974,041,253,237
      3     88      28   49    11      30       34  1,145,973,108,145   <- record
      4     68       3   62     3      23       21    702,105,074,232

FIVE THINGS FALL OUT, all pre-registered (C1-C4) and all confirmed:

1. **D_g = A_kill(M -> g) EXACTLY.**  D_31 = 4 = A_kill(29->31), D_37 = 4 =
   A_kill(31->37), D_41 = 3 = A_kill(37->41), and in the small-g gate
   D_17 = D_19 = 2, D_23 = 3, D_29 = 2 reproduce A_kill(13->17) = 2,
   A_kill(17->19) = 2, A_kill(19->23) = 3, A_kill(23->29) = 2.  This is an
   IDENTITY, not a measurement: both count maximal runs of consecutive
   M-openings that one phase of g deletes (C10's "co-deletable k-tuples", whose
   legality condition "prefix-sum range <= 1" is exactly "all in one two-class
   set").  Two vehicles built four rounds apart in two different languages
   compute the same integer, 7 for 7.  ALSO: the sample-vs-census halves MEET -
   a streamed pass gives D_g >= v and C10's exact full-period fuel census gives
   D_g <= A_kill, so a partial pass that reaches A_kill has proved the value.
2. **THE RECORD LAW HOLDS AT 31, 37 AND 41**: max(before + span + after) = 58 =
   F(31), 88 = F(37), 91 = F(41).  All three survivors were then re-derived at
   the TARGET machine slot by slot (gate section B): at machine 31, slot
   21,844,264,615 and slot+58 are openings, all 57 slots between are blocked,
   and exactly TWO machine-29 openings sit inside at +18 and +28; at machine 37,
   slot 1,145,973,108,145 and slot+88 are openings, all 87 between blocked,
   exactly THREE machine-31 openings inside at +28, +65, +77; at machine 41,
   slot 7,244,836,295,007 and slot+91 are openings, all 90 between blocked,
   exactly THREE machine-37 openings inside at +15, +56, +70.
3. **THE ATTAINING RUN LENGTH IS k_win.**  L = 2 at g = 31, L = 3 at g = 37 and
   L = 3 at g = 41, which is C13's k_win census (k_win(29->31) = 2,
   k_win(31->37) = 3, k_win(37->41) = 3) recovered by a completely different
   vehicle, 3 for 3.
4. **THE L = 1 ROW IS F_2 OF THE LOWER MACHINE, EVERY TIME** - 55 = F_2(29),
   68 = F_2(31), 90 = F_2(37), which is a free control on each pass (a run of
   one deleted opening merges exactly two lower gaps).  It is also why the
   record cannot be carried at L = 1 anywhere at this end of the ladder:
   F_2(lower) < F(g) at all three.  More generally the L-row maximum IS
   Q*_{L+1}(M; word-legal for g) WITH PADDING INCLUDED, and the m37 pass reads
   off Q*_2(31) = 68, Q*_3(31) = 85, Q*_4(31) = 88, Q*_5(31) = 68, whose maximum
   88 = F(37) is R68's attainment theorem verified end to end on a 6.2e9-opening
   sequence.  NOTE FOR CONSTRUCTOR: these are NOT the numbers in your round-28
   Delta table, which is the LITERAL-middles restriction (Delta_3(31) = +2, i.e.
   Q*_3 = 70); the padded letter 37 is what takes 70 to 85 and it is the letter
   your three failing rows carry.
5. **A FREE CROSS-VEHICLE HIT, UNPROMPTED**: the machine-31 record's survivor
   sits at lower-period position 278,620,515 - which is C13's kwin_census
   winner for 29 -> 31, "k = 278,620,515, word (10,)", with the same span 10 and
   the same flank sum 18 + 30 = 48.  Round 17's envelope census and round 29's
   anchor-235 stream produce the same address.

MACHINE 41 IS A DELIBERATE PARTIAL SWEEP, AND SAYING SO IS THE POINT.  The full
pass is 1147 chunks; it ran on six resumable workers over disjoint j37 ranges,
each dumping its own JSON after EVERY chunk with the list of chunks done, so the
stop leaves an exactly specified sample rather than a guess.  THE WORKERS WERE
STOPPED DELIBERATELY at coverage 423 of 1147 chunks (36.9%), with twelve laps
j37 = 0, 1, 7, 8, 13, 14, 19, 20, 25, 26, 31, 32 COMPLETE and the rest partial;
`chain_depth_r29.py merge 41 w0..w5` prints that line and
research/data/r29/chain_41.json carries the chunk list.
BOTH ANSWERS ARE NEVERTHELESS EXACT, because in each case the sample supplies
one half and an existing exact census the other:
  D_41 = 3   : the sample gives D_41 >= 3 (reached in the FIRST chunk of all six
               workers) and C10's A_kill(37->41) = 3 EXACT gives D_41 <= 3.
  record 91  : the sample EXHIBITS a merged gap of 91 (machine-41 slot
               7,244,836,295,007, verified there slot by slot) and C14's
               F(41) = 91 EXACT (COV-SAT, full period) caps it.
What the remaining coverage would buy is only the exact per-L rows Q*_J(37),
not either headline.  That is standing rule 39.

### C11-UPDATE. THE F_j SPECTRA AFTER ROUND 29
    machine   F_1  F_2  F_3      F_4        F_5     F_6
    47        118  134  145  [154,174]  [167,174]   177   <- new this round
(the rest of the table is unchanged from the round-28 C11-UPDATE.)

## Standing rules (round-29 additions)

38. A DELIBERATE PARTIAL SWEEP MUST DUMP AFTER EVERY UNIT OF COVERAGE.  Round
    28 lost seven workers and round 29 lost six more to the same silent commit
    death, and in both cases the loss was total because the result was written
    only at the end.  Every long streamed pass now writes its own JSON after
    each chunk WITH THE LIST OF CHUNKS DONE, and resumes from it.  The
    consequence is not just crash-safety: it converts "I had to stop" from an
    apology into a SAMPLE WITH A STATED SUPPORT.
39. A SAMPLE AND A CENSUS CAN MEET IN THE MIDDLE.  A streamed partial pass gives
    a LOWER bound on a maximum; an already-decided arity level gives the UPPER
    bound.  D_41 >= 3 (one chunk of a sample) plus A_kill(37->41) <= 3 (C10,
    exact) is an EXACT value from 0.1% coverage.  Before pricing a full sweep,
    ask which half of the answer is already on record.
40. WHEN A RUN'S ANSWER IS IMPLIED BY A THEOREM PLUS A NUMBER YOU ALREADY HAVE,
    IT IS A CONTROL, NOT A DECISION - price it as one.  I launched the
    word-legal Q*_J(47) sweep before noticing that R68 plus the corpus
    F(53) = 145 already gives its answer.  Killed at 3.5 core-hours; the
    remaining value (an INDEPENDENT computation of F(53) from machine 47) is
    real but is a control, and controls do not get the round's last cores.
41. RAISE THE SEED TO THE QUESTION, NOT TO THE ANSWER.  The exact F_4/F_5/F_6
    ladder costs 4,000 s; the question actually asked - "does the maximum clear
    171?" - costs 400 s at seed 171 and is answered UNCONDITIONALLY by one
    exhibited window.  Rule 16 says the answer must sit above the seed; this
    adds: choose the seed from the DECISION the number has to make.

## Retracted / corrected (round-29 additions)

R30. I LOST SIX WORKERS AND A STREAMED PASS TO THE EXACT FAILURE MODE I WROTE
    STANDING RULE 37 ABOUT, IN THE ROUND AFTER I WROTE IT.  Six j5_multi
    workers (item b) and the machine-37 pass died SILENTLY and simultaneously -
    no traceback, no completion line, launcher exited 0.  Committed memory was
    at 37-41 of 63.6 GB and five other lanes had ~12 python processes up; my own
    contribution was seven jobs PLUS two foreground profiling scripts that each
    loaded the memmapped machine-29 arrays.  Same cause as round 28's kill and
    the same trigger: running a diagnostic in the foreground on a box already
    carrying my own workers.  RECOVERED, not reported around: the item-(b)
    partial logs are valid maxima over the index ranges they walked, and they
    already contained the round's headline (a 174-span 6-window); the machine-37
    pass was relaunched under standing rule 38 and completed.  COST: ~40 minutes
    of six-core work and ~15 minutes of one-core work.
R31. MY PRE-REGISTERED PRICE FOR MACHINE 41 WAS WRONG BY 5x.  C5 predicted the
    exact g = 41 pass in "under 90 minutes at 6 workers"; the measured cost is
    26 s a chunk x 1147 chunks = 8.2 core-hours, i.e. ~2.7 h at 3 workers and
    ~1.4 h at 6.  I priced it from the machine-31 pass (30 s for ONE chunk)
    without multiplying by the 37 laps of the outer loop - an arithmetic slip in
    my own pre-registration, caught by the run.
R32. SCORING THE ROUND-29 PRE-REGISTRATION (research/data/r29/prereg_mechanic_r29.md)
    A1 four witnesses re-verify                       CONFIRMED (4/4)
    A2 the m59 pair is a mirror pair                  CONFIRMED (exact)
    A3 the flanks mirror too                          CONFIRMED ((13,4)/(4,13))
    B1 "the certificate FAILS at 47 -> 53"            CONFIRMED - and this was
       the deliberate two-sided call, made from the F_J increment ladder before
       any window above 171 had been seen.
    B2 F_4 = 154 +-4 / F_5 = 164 +-6 / F_6 = 174 +-8  CONSISTENT on all three:
       F_4 in [154,174], F_5 in [167,174], F_6 = 177 EXACT (3 over the centre,
       inside the band).  F_4/F_5 not banked as exact.
    B3 the maximum is at J = 6                        CONFIRMED (177 at J = 6)
    B4 band run under 2 h at 6 workers                CONFIRMED - the full band
       run finished at 4 workers, and its measured price is 400 s single-core
    C1 D_31 = D_37 = 4, D_41 = 3                      CONFIRMED (3/3)
    C2 record law = 58 / 88 / 91                      CONFIRMED (3/3)
    C3 attaining L = 2, 3, 3                          CONFIRMED (3/3)
    C4 the L = 1 row is F_2 of the lower machine      CONFIRMED (55, 68, 90)
    C5 g = 41 exact under 90 min at 6 workers         REFUTED (R31)

## Open watches and checkpointed jobs (round-29 additions)

- F_4(47), F_5(47), F_6(47) EXACT: the seed-145 sharded run is resumable
  (research/fj47_r29.py run <workers> 64) and 12 of 64 shards are complete on
  disk.  Remaining price ~3,250 s of single-core walk.
- max_J Q*_J(47) BY INDEPENDENT COMPUTATION (not via F(53)): priced at 3.5
  core-hours, resumable (research/fj47_r29.py run <workers> 64 legal).  It is a
  control, and its value is that it would re-derive F(53) = 145 from machine 23
  with machine 53 never built.
- THE MACHINE-41 CHAIN-DEPTH PASS is resumable per worker; the full sweep is
  8.2 core-hours and the completed chunk list travels in the JSON.
- THE CRITERION-MARGIN LADDER (C49) has one missing cell: A_kill(53 -> 59) = 4
  gives J_max(53) = 5, and F_4(53), F_5(53) are not on record, so the 53 -> 59
  row cannot be filled.  It is the cheapest test of the "A_kill >= 4 fails"
  reading, and its budget is F(53) + 59 = 204 against F_5(53).

## Round 30

GATE (one command, clean process, imports nothing from the tools that produced
the numbers):
    uv run python research/gate_mechanic_r30.py     -> ALL ASSERTIONS PASSED
       A the 64-shard tiling of the seed-144 word-legal run, its per-J maxima,
         the J = 4 witness lifted to a machine-47 slot and then to a machine-53
         slot;  B every killer-profile extension re-checked (legal extension,
         SAT set recomputed, refuted; cover-only verdicts re-derived by a direct
         period scan at m19/m23);  C V2 = A_kill - 1 at every scanned machine
         and both attaining runs re-checked as consecutive openings;  D the
         four lifted record slots.
    uv run python research/resrun_r30.py gate       -> ALL ASSERTIONS PASSED
         (V2 = D_g - 1 against anchor235/chain_depth.py at g = 7..29)
Pre-registration: research/data/r30/prereg_mechanic_r30.md, written before
any round-30 script existed (A1-A6, B1-B5, C1-C6, D1-D3; scored in R33).
Persistent results: research/r30_results.txt.  Logs research/data/r30/*.log.
New files: research/qstar47_r30.py, resrun_r30.py, wordkill_r30.py,
genealogy_r30.py, gate_mechanic_r30.py.
THE BRIEF THIS ROUND WAS MECHANISM PROBES, NOT BIGGER MACHINES: three probes
of hidden structure and one fetch.  Every probe below is an exact event on a
named object with the object exhibited; the only rates are the MODEL columns,
and they are labelled model.

### C51. THE INDEPENDENT max_J Q*_J(47; legal for 53) = 145 - RUNG ELEVEN
### CLOSED WITH MACHINE 53 NEVER CONSULTED (r30, brief item d)

Repro: research/qstar47_r30.py run 6 64 (logs research/data/r30/q47_s144/,
driver log q47_driver.log); gate section A.  Round 29 closed 47 -> 53 only by
R68 plus the corpus F(53) = 145 - a control.  This is the decision: the
word-legal spectrum of machine 47 for gear 53, computed on MACHINE 23's period
by the six-gear lap-phase transfer (j5_multi.py, mode legal), seeded at 144 -
ONE BELOW the value the attainment theorem predicts, so the run is two-sided
(rule 41: raise the seed to the question) - with span cap 290 (at or above the
subadditivity ceiling of every depth <= 6, so nothing is span-conditional)
and depth cap 6 = J_max(47) = L(47) + 2 (R89, L(47) = 4 exact).

    Q*_2(47) <= 144   Q*_3(47) <= 144   Q*_4(47) = 145   Q*_5(47) <= 144   Q*_6(47) <= 144
    max_J Q*_J(47; legal for 53) = 145  <=  171 = F(47) + 53     margin 26

64 of 64 shards tile the 7,952,175 start openings exactly (asserted).  The
values reported "<= 144" are AT THE SEED and are brackets, not values.  THE
WITNESS, from shard 14: machine-23 start 8,413,890, phases (27,4,16,24,4,24)
for gears (29,31,37,41,43,47), marks (4,7,12); CRT lifts it to

    MACHINE 47, slot 82,799,441,296,736,535: openings [0, 70, 105, 123, 145],
    gap word [70, 35, 18, 22], middles (35, 18) = classes (-, +) mod 53 - legal
    and alternating - 141 other slots blocked;

which is EXACTLY the round-26 anchor slot (C27) found by a different seed and
a different worker set.  Lifted once more with the phase of 53 that deletes
the three interiors: MACHINE 53, slot 4,182,064,658,553,345,935 is a gap of
exactly 145.  So F(53) >= 145 is exhibited and F(53) <= 145 follows from the
run plus the attainment theorem - F(53) = 145 re-derived from machine 23's
period (ratio 1.5e11) with machine 53 never built.  STATUS: EXACT-
UNCONDITIONAL.  COST: 7,709 shard-seconds = 2.14 core-hours at High
priority against the round-29 price of 3.5 (pre-registered D3 <= 5 core-hours).

### C52. L AS A RESIDUE-RUN STATISTIC: THE LENGTH IS A DENSITY EFFECT, THE
### LAST UNIT IS ARITHMETIC (r30, probe a)

Repro: research/resrun_r30.py scan M | report | models; wordkill_r30.py words
M --crt; data research/data/r30/resrun_m*.json, words_m*_crt.json, log
models_31_37.log.  Definitions (fixed in the pre-registration): for a prime
g > M, d = 2*6^{-1} mod g; V1 RAW = longest run of consecutive gaps of M with
residues in {0, +d, -d}; V2 T3 = V1 with strict alternation of the nonzero
classes = L_g(M) = D_g - 1; occ_L = the number of length-L windows of
consecutive gaps that are legal alternating words; MODEL-U = ln N / ln(g/3)
(the brief's uniform-residue count); MODEL-D = independent letters with the
REAL class densities p0, p+, p- of M's exact cyclic gap histogram - longest
run ln N / ln(1/lam) with lam = p0 + p+ + p- (raw) or p0 + sqrt(p+ p-) (T3,
the transfer-matrix rate), and E[occ_L] = N times the total weight of legal
class words of length L (3-state DP).  V3 = the word vehicle's ceiling
(alphabet + spectrum caps + phase saturation, no cover decision); V4 = V3 +
CRT realisability.  Scanned: m11..m23 full periods, m29 the memory-mapped
opening list, m31 the full lower period streamed in 31 chunks (2.8 h at one
core under a 100%-loaded box), m37 a DELIBERATE PARTIAL sweep of 12 of 1147
chunks (support stated in the JSON).

    M  q' |Lam|      N       modelU  D-raw  D-T3  V1  V2=L   occ_1..occ_L (measured / model) ; occ_{L+1}
    11 13   1        135      3.3    1.6   0.0    1    1    6/6
    13 17   2      1,485      4.2    2.4   1.8    1    1    72/72
    17 19   2     22,275      5.4    3.3   2.2    1    1    1088/1088
    19 23   3    378,675      6.3    3.7   2.8    2    2    11784/11784  62/73.6 ; 0/1.1
    23 29   3  7,952,175      7.0    4.6   2.4    2    1    243816/243816 ; 0/27.3
    29 31   4  214,708,725    8.2    5.8   3.7    3    3    8.02e6/8.02e6  13000/15100  4/279 ; 0/0.53
    31 37   4  6.23e9         9.0    5.6   4.0    3    3    1.148e8/1.15e8  70964/175000  216/1610 ; 0/2.47
    37 41   6  2.18e11       10.0    5.4   4.0    2*   2    [1.05% sweep] 1.77e7/1.77e7  27/10500 ; 0/40.8   (full period: occ_3 = 0 vs 3.9e3, L = 2 exact, C10)
    (* partial-sweep lower bound over 2,279,993,244 gaps = 12 of 1147 chunks; L(37) = 2 is exact from A_kill(37->41) = 3)

FOUR THINGS THE TABLE DECIDES:
1. THE VARIANT THAT SUPPRESSES THE LENGTH IS THE ALPHABET, NOT THE COVER.
   Along the ladder MODEL-U -> MODEL-D(raw) -> MODEL-D(T3) -> measured V2 the
   drops at the next prime are 2.4/2.1/0.7 (m29), 3.4/1.6/1.0 (m31),
   4.6/1.4/2.0 (m37): the largest single drop is ALWAYS the first one - the
   legal letters are a 3-6 value alphabet whose frequencies in the gap
   histogram are far below 3/g.  Pre-registered A5 (largest drop last) is
   REFUTED.  With the real densities the independent-letter model predicts
   the longest run to within one unit at every scanned machine (3.7 vs 3, 4.0
   vs 3, 4.0 vs 2, 2.8 vs 2, 2.4 vs 1) - the run LENGTH is a density
   statistic.
2. THE OCCURRENCE COUNT AT THE TOP LENGTH IS WHERE THE ARITHMETIC SHOWS.  At
   m29 the model predicts 279 legal 3-windows for gear 31; the period carries
   4 - two mirror pairs of (10,21,10), the m29 Q*-maximiser's middle word -
   while the 2-windows are 13,000 against 15,100 (0.86) and the 1-windows
   exact by construction.  At m31 (full lower period, 6.2e9 gaps) the ratios
   for gear 37 are 1.00, 0.41, 0.13, 0 (216 realised 3-windows against 1,610;
   0 against 2.5 at length 4).  At m23 the model predicts 27 legal 2-windows
   for gear 29; there are 0.  At m37 it predicts 3,900 legal 3-windows for
   gear 41; there are 0 (L(37) = 2, exact) - and already at length 2 the 1%
   sweep finds 27 against a model 10,500 (0.0026), because every realised
   2-word of m37 for gear 41 carries the padded letter 41 and the pure
   alternation (14,27) is unrealised.  So the deficit is not a smooth
   correction: the count tracks independence to within 15% at length 1-2 at
   the small machines and collapses by 8x, 70x, 400x, infinity at the top
   lengths.  This is Constructor's eps/Phi object and this lane's C13
   "occurrence count governs" seen at the letter level: the last unit or two
   of L are decided by the cover half, everything before by the histogram.
3. THE SUPPRESSION AT THE TOP IS NOT ALTERNATION: V1 - V2 is 0 at every
   next-prime cell except m23 (2 vs 1, the raw run (10,10) is two + letters).
   A2 confirmed.
4. THE NEXT PRIME IS USUALLY BUT NOT ALWAYS THE MAXIMISING GEAR.  L_g(M) by
   CRT for every prime g <= 130 (words_m*_crt.json, V4 = V2 wherever both
   exist, 8 of 8 cells): m29 has L_31 = 3 > L_37 = 2 > 1; m31 has L_37 = 3 >
   L_41 = L_53 = 2; but m23 has L_31 = 2 > L_29 = 1 and m37 has L_53 = 3 >
   L_41 = 2 (letters {18,35,53,71,88}).  A4 as worded ("not the maximum at
   >= 5 of 8") is REFUTED - the next prime is the maximum at 3 of 5 scanned
   machines - but it is not special either: what sets L_g is the alphabet
   size |Lambda_g(M)|, which the next prime usually maximises.
THE WORD VEHICLE'S CEILING IS ONE ABOVE THE TRUTH: V3 = V4 + 1 at 7 of the 8
next-prime cells (m19 3/2, m23 2/1, m31 5/3, m37 3/2, m41 3/2, m43 3/2, m47
5/4; m29 3/3) - the arithmetic screens (alphabet, spectrum caps, phase
saturation at every gear) leave exactly ONE length that only the cover
decision removes.  A6 confirmed.  At m47 the one survivor is the pure
alternation (35,18,35,18,35), Constructor's refuted length-5 word.
ATTAINING RUNS (section view; slot = the opening before the run):
    m19 q'=23  RAW slot 1,297 [8,8] residues (8,8) classes (+,+);  T3 slot 9,382 [8,15] (+,-)
    m23 q'=29  RAW slot 16,363 [10,10] (+,+);  T3 slot 77 [10]
    m29 q'=31  RAW = T3 slot 220,171,102 [10,21,10] residues (10,21,10) classes (-,+,-)
    m31 q'=37  RAW slot 115,954,443 [25,12,12] (+,-,-);  T3 slot 143,358,780 [25,12,25] (+,-,+)
    m37 q'=41  RAW = T3 (1% sweep) slot 109,580,398 [14,41] residues (14,0) classes (+,0)
    Other gears at m31 (full period): g=41 L=2 with occ_2 = 2 against a model
    14,000; g=53 L=2 with 224 against 1,020; g=59, 61 RAW 3 but T3 1 (the raw
    runs are (d,d,d) repeats that alternation forbids).

### C53. THE KILLER PROFILE OF WORD EXTENSIONS: THE EXTENSIONS DIE OF THE
### COVER HALF, MOSTLY WITH NO OPEN CONSTRAINT AT ALL (r30, probe b)

Repro: research/wordkill_r30.py kill M; data research/data/r30/killer_m*.json;
gate section B.  For every realised legal word of length L(M) (grown level by
level by the overlap lemma with CRT decisions; at m47 taken from
Constructor's round-29 exhaustive decision) and every T3-legal one-letter
extension at either end (letters = all class values <= F(M), holes included),
two exact attributions: SAT = the gears whose single-gear free set is empty
(the r26 screen; a theorem confines it to g < 2|X|), and y* = the OPEN-
CONSTRAINT KILLER PREFIX: R(S) is the realisability CSP with the open
constraint imposed only on the gears of S and every other gear's phase FREE
(it still helps cover); R only gets harder as S grows, so y* = min{y' :
R({g <= y'}) infeasible} is found by bisection.  y* = 0 means R(empty) is
already infeasible: NO SLOT OF M BLOCKS THE PUNCTURED INTERIOR - the word
dies of the cover half alone, with no tooth position of any open point
needed.  y* = M means the top gear's open constraint is needed.

    M -> q'  L  realised length-L words        ext. classes  y* = 0   y* = 5   y* = 7   bracket   SAT non-empty
    19 -> 23  2  (8,15) (15,8)                        4         4        -        -        -          0
    23 -> 29  1  (10) (19) (29)                       4         3        -        1        -          2 ({5})
    29 -> 31  3  (10,21,10)                           2         2        -        -        -          2 ({5},{5,7})
    31 -> 37  3  (12,25,12) (25,12,25)                4         3        -        1        -          2 ({5})
    37 -> 41  2  (14,41) (27,41) (41,14) (41,27)     15         8        5        -        2          7 (all {5})
    41 -> 43  2  (14,43) (29,43) (43,14) (43,29) (43,43)  19    -        9        -       10          9 (all {5})
    (every extension REFUTED at the full machine, 0 undecided, 0 realised;
     y* = 5 means gear 5's open constraint alone excludes the L+3 open points;
     "bracket" = R(empty) or R({5,7}) undecided at the relaxed budget)

The corridor kills are exactly the words with two literal letters in a row of
the same class through a padded letter or the pure alternation: (10,19) at
m23 and (12,25,12,25) at m31 die at y* = 7 with SAT empty - gears 5 and 7
JOINTLY, their blocked pattern occurring in M (R(empty) feasible, re-derived
by the gate at m23) - the corridor mod 35 exactly; at m37 and m41 the
literal-letter extensions (14,27,41), (27,14,41), (14,29,43), (14,43,29),
(29,14,43) and every (.,41,41)/(.,43,43)-type word are saturated by GEAR 5
ALONE (five open points leave gear 5 no phase).  The kills by a LARGE letter
(55, 68, 82 at m37) are cover-only: no slot of machine 37 blocks the punctured
interior.  At m41 the relaxed instances R(empty) did not decide at 10M nodes
for 10 of 19 classes, so the m41 profile is "9 gear-5 saturations + 10
refuted-unattributed"; NO extension anywhere was attributed to the open
constraint of a gear above 7.  B1 confirmed (SAT only at 5 and 7, and only at
5 from m37 on); B2 confirmed (28 of 48 classes pooled are not single-gear
saturated); B3 REFUTED in the opposite direction (the joint kills need NO
open constraint, y* = 0, or the corridor's, y* = 7; none at y* >= 13); B4
UNRESOLVED at m41 (9 of 19 corridor kills decided, 10 unattributed) and not
attempted at m43/m47; B5 confirmed (0 realised, 0 undecided full decisions).
COST: m37 15 classes at 0-894 s each on 4 workers (High); m41 19 classes at
0-2,173 s each - the relaxed R(empty) instance at m41 is the expensive
object, not the full decision (which is 0 s when gear 5 saturates).

### C54. RECORD GENEALOGY: RECORDS DO NOT RECRUIT RECORDS - THEY RECRUIT
### RUNNER-UPS WHOSE LARGEST GAP WAS ITSELF MERGED (r30, probe c)

Repro: research/genealogy_r30.py [--rank]; logs research/data/r30/genealogy.log,
genealogy_rank.log; gate section D.  For a record window of y = M + q' at
slot k, the M-openings inside (k, k+F) are the deleted chain and the ancestor
is the (L+1)-gap window of M they cut; each gap of that window is in turn a
merged window of the machine below.  The whole tree is computed by residue
arithmetic on the slot (no scan); the F(43), F(47), F(53), F(59) record slots
are obtained by LIFTING the recorded word-legal windows with the phase that
deletes their interiors and are verified at the target (426824541409250,
34905861380755417, 4182064658553345935, 73115517300464200662).

    step    record  ancestor (window of M, J)  deleted teeth  vs F_J(M)          largest gap   generations
    23->29    43    [10,10,23] J=3   phase 14  '-+'          runner-up by 7      23 INHERITED     1
    29->31    58    [18,10,30] J=3   phase  8  '+-'          runner-up by 7      30 merged        5
    31->37    88    [28,37,12,11] J=4 phase 3  '++-'         runner-up by 2      37 merged        4
    37->41    91    [15,41,14,21] J=4 phase 19 '--+'         runner-up by 14     41 merged        2
    41->43   103    [28,75] J=2      phase 22  '-'           RECORD = F_2(41)    75 merged        3
    43->47   118    [85,31,2] J=3    phase 17  '+-'          runner-up by 7      85 merged        4
    47->53   145    [70,35,18,22] J=4 phase 45 '+-+'         F_4(47) not on record  70 merged     2
    53->59   161    [10,118,33] J=3  phase 39  '--'          F_3(53) not on record 118 merged     3

(phase = slot mod q'; teeth = which tooth each deleted opening sits on, in
order; generations = consecutive levels down the largest gap at which that gap
is merged, i.e. has a deleted opening of the machine below inside it.)
Scores: C1 RR-SPECTRUM (ancestor is the F_J(M) maximiser) 1 of 8 - CONFIRMED
FALSE as a law; C2 RR-DEPTH (largest gap merged below) 7 of 8 - CONFIRMED;
C3 generations >= 2 at all 7 steps from 29->31 on and >= 3 at 5 - CONFIRMED;
C4 (a top-3 gap value of M inside the ancestor) 0 of 8 - CONFIRMED; C5 (F_J
records' largest gap merged below) 12 of 12 (F_2/4/5(41), F_2/3/4(43),
F_2/3/6(47), F_2(53), F_2(59) x2) - CONFIRMED; C6 ancestor RANK among M's own
J-windows by span (#strictly above): m13 12, m17 60 (J=3) / 0 (J=2), m19
218, m23 8, m29 18 - above 10 at 4 of 6 measured, CONFIRMED.
THE M31 RECORD'S FULL TREE, as the example: 58 <- m29 [18,10,30] <- 30 = m23
[7,23] <- 23 = m19 [5,15,3] <- 15 = m17 [2,6,7] <- 6 = m13 [5,1], 7 = m13
[5,2]: FIVE generations, every one a runner-up (deficits 7, 9, 12, 13, 10, 9
against the F_J of its machine).  A record is an ordinary window whose gaps
were each made one level down by an ordinary merge, sitting at the top on the
teeth of the last gear - anchor-235 9d's "ordinary lower stretch, record made
at the top three or four layers", now with the addresses.
WHAT F(M+q') COULD BE COMPUTED FROM: not the top-k J-windows of M by span
(the ancestor's rank is 9-219), and not M's spectrum records.  By the
attainment theorem it is max over the realised legal words w (1-4 per
machine) of max over the OCCURRENCES of w of (gap before + span + gap after).
For the long words the occurrences are few (4 for (10,21,10) at m29, 2 mirror
pairs) and each is a CRT solution that crt_dict.count_solutions can
enumerate scan-free; for the short words (the L = 1 and L = 2 rows, 8e6 and
1.3e4 occurrences at m29) the flank order statistic Phi(w) is exactly what a
scan supplies and no enumeration reaches.  So the record is scan-free
precisely when it is carried at depth L >= 3, and that is the regime the
ladder is entering (k_win = 3 at 31->37 and 37->41; 47->53 and 53->59 are
carried at J = 4 and J = 3).  Named, not built: the count of occurrences of
each realised legal word by CRT enumeration at m41..m47, with the flank sum
read off each solution - the counted census Constructor asked for (occ(q';M)),
delivered as a list of slots instead of a number.

### C11-UPDATE. THE F_j SPECTRA AFTER ROUND 30
    machine   Q*_2  Q*_3  Q*_4  Q*_5  Q*_6   (word-legal for 53, machine 47)
    47        134  <=144   145  <=144 <=144   max 145 = F(53), two-sided
(F_2 = Q*_2 by definition; the F_J row of C11 is unchanged.)

## Standing rules (round-30 additions)

42. SEED ONE BELOW THE PREDICTED ANSWER WHEN A THEOREM PREDICTS IT.  A run
    seeded at 144 for a quantity R68 said was 145 is two-sided (it must find
    the value AND find nothing above) and costs 1.2x the seed-145 run; it
    turned a control into a decision with a witness.
43. A KILL HAS TWO HALVES AND THEY MUST BE ATTRIBUTED SEPARATELY.  The
    realisability CSP has open constraints (teeth of the open points) and
    cover constraints (the blocked interior); "which gear kills it" is
    meaningless for the cover half (fewer gears cover less) and is answered
    for the open half by relaxing all other gears' open constraints.  Test
    R(empty) FIRST - most refutations this round needed no open constraint,
    and a bisection that assumes R(empty) feasible reports a wrong gear (it
    did, once, before the gate caught it at m19).
44. THE MODEL COLUMN IS THE INSTRUMENT, NOT THE RESULT.  The independent-
    letter model with the real class densities is what turned "L is small"
    into "the run length is a histogram statistic and the count at the top
    length is not" - two different mechanisms that the length alone cannot
    separate.  Report occ_L / E[occ_L] at every L, not the maximum.

## Retracted / corrected (round-30 additions)

R33. SCORING THE ROUND-30 PRE-REGISTRATION (research/data/r30/prereg_mechanic_r30.md)
    A1 V1 <= 0.7 modelU at m19..m37 (q')     CONFIRMED (0.32, 0.29, 0.37, 0.33, <= 0.7)
    A2 V1 - V2 <= 2 at q'                   CONFIRMED (0,0,0,0,1,0)
    A3 modelD-T3 / V2 >= 1.5 at m19..m37     REFUTED - 1.4, 2.4, 1.23, 1.33, 2.0: the
       length model is right to within a unit; the deficit is in occ_L (70x, oo)
    A4 next prime not the max at >= 5 of 8   REFUTED as worded (not the max at 2 of 5
       scanned: m23 g=31, m37 g=53)
    A5 largest drop is model -> measured     REFUTED - it is modelU -> modelD (the
       alphabet's density) at every scanned machine
    A6 V3 >= V4 + 1 at >= 5 of 8             CONFIRMED (7 of 8)
    B1 SAT killers only at g <= 13           CONFIRMED (5 and 7 only)
    B2 >= 50% not single-gear saturated      CONFIRMED (28 of 48 pooled classes)
    B3 joint kills have y* >= 13 half the time, y* = M once per machine
                                             REFUTED the other way: every decided joint
       kill is y* = 0 (no open constraint needed) or y* = 7 (the corridor); none >= 13
    B4 corridor {5,7,11} kills < 50% at m41..m47   UNRESOLVED (m41: 9 of 19 decided as
       gear-5 kills, 10 refuted but unattributed; m43/m47 not attempted)
    B5 zero realised, <= 2 undecided         CONFIRMED (0 realised, 0 undecided at the
       full machine, every machine m19..m41)
    C1-C6                                    CONFIRMED (1/8, 7/8, 7/7 and 5/8, 0/8,
                                             12/12, 4/6)
    D1 max = 145 at J = 4, nothing above     CONFIRMED, witness = the r26 anchor slot
    D2 Q*_3, Q*_5, Q*_6 <= 144               CONFIRMED
    D3 <= 5 core-hours                       CONFIRMED (2.14)
R34. THE RESUME MESSAGE THIS SESSION RECEIVED ("nothing of yours is on disk")
    WAS STALE: the session was never cut, and every file above was already on
    disk when it arrived.  Recorded so the manager's log and the lane's agree.

## Open watches and checkpointed jobs (round-30 additions)

- The m37 residue-run scan is a 12-of-1147-chunk partial (research/data/r30/
  resrun_m37_c12.json); the full sweep at ~20 primes is ~90 core-hours on
  this loaded box.  The m31 scan is complete (resrun_m31.json).
- Killer profiles at m43 and m47 NOT attempted (the m43 realised 2-words need
  ~8 arity-2 decisions at m43; the m47 extensions of (18,35,18,35) are 9
  reverse classes of 6-point patterns).  The 10 unattributed m41 classes and
  the 2 m37 brackets need R(empty)/R({5,7}) at a larger relaxed budget.  The
  vehicle is research/wordkill_r30.py kill M --workers 4 --nodes N.
- The counted occurrence census of realised legal words by CRT enumeration
  (C54, named) is not built.
