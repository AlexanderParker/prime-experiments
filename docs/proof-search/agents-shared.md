# agents-shared.md - findings exchange for the proof-search team

## SUMMARY (manager-rewritten each round - read this first; details below and in workstream docs)

State: round 11 in progress (restarted after a session-limit kill; mechanic ran through).

ROUND-11 NEWS (read before your continuation): (1) THE K=4 EVENT EXISTS - mechanic found exactly
4 instances per period at step 29->31, all one word class (10,21,10), two mirror pairs,
addresses listed in mechanic's data. "k_max <= 3 everywhere" below is CORRECTED to: k_max by
step = 2,2,3,2,4 at 13->17 .. 29->31; fuel is arithmetic-selected (N3>0 iff s and q-s both hit
abundant gap values), not smoothly y-driven. Machine-31 full census + machine-37 partial still
running (will fold when they land). F_j spectra for machines 23/29 delivered:
(34,39,50,58,65,77) / (43,55,65,70,85,90). Chain condition verified at 1e9 scale (pred 58 =
actual F_k(31) = 58). (2) F(2,53): the log advanced - "run of 420 is coverable", the search is
past 420. (3) human.md is now a CURRENT-STATE snapshot revised in place (user direction) - no
round logs there; history lives here and in workstream docs. (4) L=15 hunt at 31.5%, max L=13
so far (one new deep L=13 at member 3,685,669,022,369).

State after round 10 - the tolerance route reduced to two named statements, the adjacency
question answered NO, and the T1 reopening closed with an exact self-reference law.

THE UNIFICATION (constructor): both tolerance lemmas are now ONE structure. With F_j = max sum
of j consecutive gaps (the gap spectrum), rigorously excess <= F_{k_max+1} - F2, and lemma 1 is
the first spectrum increment. The whole tolerance hypothesis = SPECTRUM FLATNESS (increments are
q/3-scale, not F-scale) + FUEL BOUND (k_max = o(ln y) suffices; measured k_max <= 3 everywhere,
62 k=3 chains matching the corpus fuel census exactly). Fuel is LOCAL - genuinely
corridor-approachable; flatness inherits the escape-distance obstruction (Wall V).

ADJACENCY: NO (constructor, answering lateral's target): at y = 13/17/19/23 two maximal gaps
can never be adjacent - certified by class arithmetic + one period scan. Per-machine alpha1
closes with a three-tier check (A3 machine-free / mod-385 strata disjointness / direct), written
out at y=13. Honest limit: the tier-C residual grows (4 at y=13 -> 96 at y=23), so scale needs
mod-5005; uniformity in y still open.

THE PINNING LAW (lateral): the neighbourhood word pins the mod-385 address to <= 4 offsets,
UNIFORMLY in y (206/206 words, five machines; gear 5 unique always). #top-stratum classes <= 4
x #words; observed 6-14 classes, flat, while gap counts swing 20-106. Drift recursion REFUTED
(reachability 18/20 -> 0/4): the address is local - address = pin(word), not inherited.
Machine-independent alpha1 now needs exactly one open piece: UNIFORMITY OF THE NEAR-TOP WORD
GRAMMAR (is the word-shape family finite a priori from flank alphabet {1..5} + chain skeleton +
pinning?).

T1 REOPENING CLOSED (mechanic): the exact content is two laws - (trivial) thickness T is
monotone in the gap so g=2 bands are thinnest, T = 4m at a twin; (real) every twin dead-centers
the thinnest band above it: its product slot k = 6m^2 sits at offset 2m = T/2 exactly, one dead
slot per band, 1223/1223 verified. Everything else is density artifact (9,591 bands to 10^10:
decade-matched g=2/all ratios 0.984-1.018; zero twin-empty bands; min primes/band = 6). The
descent's binding case binds by length alone - the imported Legendre-class problem, no added
machine hostility.

FORMAL LEDGER (green, 992 jobs, 10 targets): Corridor.lean extended - endpoint_law,
endpoint_law_34 (G = 34 mod 35 forces a mod 35 in {3,18,33}), adjacency_law +
forbidden_pairs_count (= 294, full 35x35 table by decide +kernel, no native_decide),
no_chain_of_forbidden, n2_packing (W/33 <= n2, choice flagged removable). Harvester: the
assembly line CLOSED - card_triple_inter_eq (8 CRT side classes) + three_gear_master (26
filter-card terms, subtraction-free, any distinct odd primes, any prefix) - the formal master
formula for 3 gears end to end. Polignac.lean = 44 theorems. Proof note for mod-105/385
attempts: one-shot omega dies at 5 dvd atoms; use per-gear iffs + interval_cases (formalist).

SEARCHES: F(2,53) log still header-only (>= 420 standing; needs <= 486). Harvester's
assessment: a pruned restart (endpoint-law filter, 2-5x, resume support) beats continuing
unpruned - implementation authorized round 11. L=15 hunt running detached (satruns_L15.log,
~15h to 1.2e13, chunk-flushed).

CARRY-OVER FACTS (round 9, still load-bearing): escape distance = 1 - bounded-modulus corridors
constrain position, never magnitude (Wall V for global flatness). First L=14 at k =
46,133,660,494 (member 2.768e11), HL model validated; ladder L=15 ~ 5e12, L=32 ~ 3e42.
prime_adjacent_run_le (32-cap) on [propext, Quot.sound]. Alpha1 empirics 0.52-1.16, no trend.

SCOPE RULE (all agents, standing): write ONLY your own workstream doc, your round append here,
and files you created in research/ or proofs/. The SUMMARY, human.md, other workstreams' logs,
and all corpus docs (docs/*.md outside proof-search/) are off-limits without an explicit
manager instruction in your brief. (Rounds 9-10 compliance: all five agents clean.)

ROUND-11: Constructor -> the FUEL BOUND (the corridor-approachable half): what mechanically caps
chain length k_max (measured <= 3)? Chain condition + side-alternation + spacings {2u', q-2u'} +
exposure counting - derive an absolute or o(ln y) cap, or name the exact obstruction. Lateral ->
the extreme-value grammar: is the near-top word-shape family finite A PRIORI (flank alphabet
{1..5} + chain skeleton + pinning)? That is the single open piece of machine-independent alpha1.
Mechanic -> chain-length census at scale (k_max across many machines and heights - the empirical
side of the fuel bound); keep L=15 hunt running. Formalist -> the y=13 alpha1 certificate: tier
A (A3/no_chain_of_forbidden - exists), tier B (mod-385 strata disjointness at y=13), tier C (4
direct checks) as one kernel-checked theorem "alpha1 = 1 holds at machine 13" - the first
machine-checked instance of lemma 1. Harvester -> IMPLEMENT the pruned F(2,53) restart
(authorized: new binary rust2/src/bin/ or flag on maxgap, endpoint-law filter, verify identity
against known F2 values y <= 23 first, resume from state, launch detached, report PID + log
path; do NOT kill the running processes - manager handles that after verification).
## Toolbelt inventory (all verified this session)
- research/umbrella_tools.py: closed-form umbrella membership/edges for any gear set (min-rooms)
- research/slip_path.py: state_walk (per-slot gear states + kill attribution), mex_jump,
  chain_prediction (stride growth from gap word, correct k-frame window {phi, phi+s})
- research/slip_bezout.py: slip-chain->Bezout alignment, product sign law, nudge constructor
- research/chain_census.py: chain-length census + fuel words; research/band_attribution.py
- research/minimal_subset.py, sufficient_subset.py, event_horizon.py, layer_ledger.py,
  coprime_census.py, kappa_exact.py, kappa_profile.py, deficit_scan.py
- proofs/BlockedSlots.lean: kernel-checked reduction (iff), builds clean; lake at ~/.elan/bin/lake.exe

## Established laws (session-proven unless marked measured)
- Horizon theorem: gears < y decide the open interior (y, y^2) exactly; top gear's unique acts
  are boundary only. Layer law: one layer's novelty = {y^2} + {y*c : c prime in (y, y'^2/y)}.
- Composite root law: every squarefree product of set gears acts unshadowed exactly once per
  window (its own value), if it fits. Root ordering: shadow < q^2, square, then coprimes.
- Necessity law: gear q needed iff it owns a pseudo-twin (root kill beside a prime) in window.
  Square gate: q^2-2 primality (prime at 5,7,13,19,29,37,43,47).
- h(L) >= d proven for L = 1,2,3 at every y (k-frame); kappa(2) limit = 2 - (11/3)C = 0.5448.
- Deletion spacing (q+-1)/3; span law >= floor((k-1)/2)*q; chain condition verified (predictions
  18/25/34 exact); fuel words rare (k=4 first at y=29, word (10,21,10)); measured: fuel abundance
  may explain gear-37 increment anomaly.
- Measured at scale: in-window max stride ~ 0.47*log^3(member)/6 slots; stride/window collapses
  2.1e-2 (y=101) -> 6.0e-7 (y=100003, members to 1e10). 27.4M twins generated+verified.
- Every gear/composite: 2 teeth summing to modulus, shield centred in short umbrella, umbrellas
  1/3+2/3, self-blocks own pair (u' = round(q/6)); u'-column doubling = the twin sequence itself.
- The one open quantity (all equivalent forms): the all-umbrella slot recurs inside every window
  (Reduction A, kernel-checked equivalent to the conjecture).

## Round findings
(appended by agents below)

## Constructor round 1 (2026-08-18)
Condition X formalised: some window W(y) has zero twins. Full derivation + script:
docs/proof-search/constructor.md, research/constructor_ledger.py. Key exact results:
- Slot cap lemma: no gear blocks both members of a slot (q|2 impossible), so root kills
  per slot are 0/1/2 from distinct gears; lpf attribution makes the supply ledger
  overlap-free: sum_q R(q) = C = 2N - P exactly (composite root law absorbs multi-gear
  acts as product acts, never extra kills).
- Zero-slack theorem: X forces the census in every prefix: n1(t) = P(t),
  n2(t) = N(t) - P(t) - every root kill load-bearing, doubles count pinned.
- Sharpest exact necessary condition (C2, prefix pigeonhole): X => every run of
  consecutive slots holds at most as many primes as slots; margin N(t) - P(t)
  = n2(t) - n0(t) >= 0 at every prefix ("doubles stay ahead of twins from the bottom").
- Measured violation: y=13 and y=23 fail even globally (P - N = +7 both: too few
  composites to block every slot - X impossible outright below ~e^6). y=47 passes
  globally (+46, capacity abundant as corpus 5.1 warns) but fails C2 by +7: members
  (53,283) hold 46 primes in 39 slots. All violations live in the bottom band
  (< member 283); first twin at slot #1-2 of every window, first double never before
  k=20 ((119,121), first double-composite slot in N).
- Two closures recorded: run condition is an equivalence (no shortcut; but the gears
  drop out and the 46-prime excess pattern is provably admissible - no modulus blocks
  recurrence); the pair-coincidence doubles bound closes exactly (s(z) < 1 needs
  z <= 137, band < 139^2, but then Brun-Titchmarsh allows 1.22 primes/slot > 1 - empty
  intersection, the corpus-5.2 squeeze on the doubles side).
- Proof target: bottom band. Round 2 = double-onset law (exact fragile-run lower bound
  from deletion spacing at window start) + descent consequence (X at y forces a
  twin-free run of ~sqrt(y)*log y slots at the TOP of the sqrt(y)-machine's window,
  against measured log^3 strides - possible induction via the layer law).

## Lateral round 1 (2026-08-18) - tooth-sharing tested end to end
Tool: research/tooth_sharing.py. Full log: docs/proof-search/lateral.md.
- NEW closed form (verified 60/60 twin pairs to 2000): a twin pair's 4 within-pair
  double-kill CRT classes mod P = p(p+2) are exactly {+-u', +-u'(p+1)}. The +-u'
  slots are split kills (own pair + mirror); the mixed class IS the twin-product
  slot: 6*u'(p+1) - 1 = (p+1)^2 - 1 = p(p+2). So each level-sqrt(N) twin pair marks
  the level-N window at exactly 2 deterministic slots: its own slot and its product
  slot. Generalisation: for ANY two real gears the cross classes are pinned at the
  semiprime qq' slots - the machine's overcount = the semiprime census.
- Redistribution law (tested, 400 draws/config, matches per pair incl. sign flips):
  sharing a pair's tooth phase changes expected in-window wasted kills by
  1 - 2R/P, R = K mod P. Over full periods sharing changes nothing (prod(q-2)
  conservation) - the mechanism is purely positional, never cardinal.
- SETTLED NEGATIVES (exact, each with its mechanism named - reopen only against the mechanism):
  (a) tooth-sharing cannot close the recursion by counting: net survivor effect is
      O(T(y)) vs needed ~K/log^2, and in the real machine both guaranteed wasted
      kills land on already-decided slots (self-block slot, already the -T(y)
      ledger term; product slot, composite by construction). Zero new open slots.
  (b) max stride is INSENSITIVE to tooth-sharing: shared-vs-independent phases
      differ by +0.02 +- 0.05 slots; real machine sits at z = -0.6 in the random-
      phase population. Stride is not the observable through which sharing acts.
  (c) "umbrella nesting" is not a separate mechanism: any two gears' short
      umbrellas are concentric at joint shields; only the coinciding edges are
      twin-specific, and those are the +-u' pinned classes. One mechanism total.
  (d) rich-vs-poor real-prime gear sets: all measured differences decompose into
      kill-density mismatch. Use phase-randomised controls, not prime-matched.
- Non-generic-phase anomaly (measured, unexplained beyond semiprime pinning): real
  machine's in-window overcount is z = +6.1 above random phases and lone-killer
  (fragile) count z = -5.9 below, while max stride is dead average. The machine's
  phase vector (+-round(q/6)) is a special point of phase space by every waste
  metric yet generic by stride. Proposed next probe: is it extremal for anything?
- Curiosity for the necessity law: when p(p+2)+2 is prime (p = 5, 149, 179, 239,
  269, 419, 569, 1289, 1319, ...) the twin pair JOINTLY owns a pseudo-twin at its
  product slot - joint necessity of a gear pair, a notion the per-gear law misses.

## Mechanic round 1 (2026-08-18) - fragile census at scale
Tool: research/fragile_census.py (segmented numpy; y=50021, 4.17e8 slots, 52s).
- Fragile/twins grows without bound: 1.11 (y=13) -> 3.42 (503) -> 4.63 (10007)
  -> 5.06 (50021). Fits a*lnln(y^2)+b with a~3.0-3.2 (fit, not law). So fragile
  slots are NOT proportional to twins, window/log^3, or pi(y^2)/log - they carry
  an extra Mertens lnln factor.
- Sharp measured law, zero free parameters: fragile * pi_win / (twins * W1) = 2,
  where W1 = sum over lone-composite members (one gear divisor q) of (q-1)/(q-2),
  pi_win = primes in (y,y^2]. Measured 1.95-1.99 from y=503, monotone toward 2,
  error 0.43% at y=50021. Meaning: fragile count = lone-composite population x
  the SAME partner-prime probability that makes twins from primes, boosted by the
  HL weight of the owning gear. No twin-specific structure in the fragile census.
- Ownership concentrates downward: bottom decile of gears owns 58% (y=101) ->
  88% (y=50021) of fragile slots (~1/q frequency per gear). Any necessity/
  minimal-subset argument should treat the small gears' fragile load as the
  generic case and large-gear fragile slots as rare events (top-half deciles
  own <2% at scale).
- Semiprime vs any-composite variants differ by <7% and converge (93.6% semi at
  y=50021); the loose extras are q^a and q^2*p shapes.

## Mechanic round 2 (2026-08-18) - per-gear closed form + prefix censuses
Tools: research/fragile_pergear.py, research/prefix_census.py.
Data for Constructor: research/data/prefix_census.csv (y ladder 101..1e8+7,
t=1..200, columns y,t,k,member_lo,P,n0,n1,n2,margin; margin = t-P = n2-n0;
convention: member equal to y counts as prime - adjust slot 1 for open interval).
- Per-gear law CONFIRMED incl. rare-event tail, one refinement required: the raw
  form 2*tw*((q-1)/(q-2))*S1(q)/pi_win runs 4-5% low for mid/large gears (z~-30
  at y=50021) - pure member-size geometry (gear q's lone composites live only in
  (q*y, y^2)). Size-corrected form with 1/ln(m) weights,
  frag(q) = 2*tw*((q-1)/(q-2))*S1w(q)/piw, S1w = sum 1/ln m over lone-q members,
  piw = sum 1/ln m over window primes, is exact to 2e-4 aggregate and Poisson-
  clean in EVERY gear band at y=10007 and 50021 (all |z| <= 1.4; top-1% band
  1.0055, z=0.07). The necessity-scale tail (S1(q)=O(1), top gears) obeys the
  same law - no twin- or necessity-specific structure anywhere in the fragile
  census. Gear 50021 owns exactly its square pseudo-twin (50021^2-2 prime).
- Prefix censuses (150 windows, 25 per decade 1e3..1e8, first 200 slots, exact):
  first DOUBLE at slot 2.4-3.7 mean, max 9, y-independent; first twin above y at
  ~ln^2 scale (mean 6.6 -> 37 over five decades). Margin N-P never below -1
  anywhere; negative only at t <= 4 for y >= 1e4 (boundary twin at slot 1-2);
  for t in [5,200] margin >= 0 in 125/125 windows y >= 1e4, min reached by
  t <= 11, then climbs ~linearly to 70-133 at t=200.
- Strategic identity for C2: margin < 0 forces n0 > 0, so every prefix-pigeonhole
  refutation of X is a nonconstructive twin-existence proof; measured reach ends
  by slot ~4 (y >= 1e4). Raw prime counting cannot bite the bottom band beyond
  the first few slots. Sharper hook consistent with our data: under X, zero-slack
  forces P(t) = t exactly for every t below the first double slot - so ANY
  proved lower bound L on double onset pins the first L-1 slots to exactly one
  prime each; the real windows violate "exactly one prime per slot" almost
  immediately (twin at slot <= 2 in most windows, else early 0-prime slots).

## Mechanic round 3 (2026-08-18) - full-window margin trajectories
Tool: research/margin_trajectory.py (primality-only sieve: y=200003, W=6.67e9
slots, members to 4e10, 186s). Data: research/data/margin_summary.csv,
margin_checkpoints.csv, margin_bands.csv. COMPLETE windows, every slot checked.
- STRUCTURAL CLOSURE + measurement: M(t) = t - P(t) (and n0,n1,n2) depend only
  on member primality - the margin is GEAR-BLIND. Layer bands touch attribution
  only. Measured anyway: slope of M across every band boundary p^2 vs matched
  mid-band controls: difference 0 at 1e-4 precision (y=200003: -0.0001+-0.0001,
  controls identical). No dip, smooth through every boundary. The cumulative
  statement cannot see layer bands through the census; band structure must enter
  via per-gear objects.
- Min-margin scaling: y >= 503 => minM in {0,-1} at t_min <= 3 (the -1 = the
  boundary twin at slot 1-2); NO later dip anywhere in any complete window up to
  6.67e9 slots. last<0 <= 11 absolute. Sub-e^6 regime (y <= 419): dips to -5
  (y=101), shallowing monotonically - drift dM/dt = 1 - 6/ln(member) crosses 0
  at member e^6 ~ 403, and every y >= 503 window starts past it.
- Danger-zone shape: NOT "M > 0 for t > c*y" and not a window fraction (frac
  collapses 1.8e-2 -> 1.5e-10). It is member-anchored and O(1)-absolute:
  "M(t) > 0 for all t > 11" held in all 15 complete ladder windows y >= 503.
- Growth law: M(t) = t - [li(6t+m0) - li(m0)] to 0.1% past t ~ 1e3; asymptot.
  linear, slope 1 - 6/ln(member); t/ln t fails. Threshold escape times = li-model
  inversion within a few % (0.3% at T=1e4); escape time DECREASES with y at
  fixed T (~T/(1-6/ln y) slots).
- Empirical prime-race envelope for the cumulative statement: max |M - Mhat|
  over all checkpoints = 0.06-0.18*sqrt(member), coefficient shrinking with y
  (0.058 at 2e5). "M(t) >= Mhat(t) - 0.2*sqrt(6t+y)" held at every checkpoint
  of every window tested (checkpoints log-spaced, 8/decade; envelope is
  checkpoint-level, min/last-below columns are exact every-slot).

## Mechanic round 4 (2026-08-18) - per-gear supply R_q(t) + pair schedule
Tool: research/supply_trajectory.py (lpf-attribution sieve; y=50021 full window
incl. 13.2M-pair schedule, 85s). Data: research/data/supply_load.csv (t, member,
P, n0, n2, C, A_active, mean_load, g5_share, rho, S_pair, tau per checkpoint),
supply_pergear.csv (R_q(t): every gear at y<=2003, 24 reps above).
- R_q(t) verified DEFINITIONALLY against an independent spf-table count at every
  checkpoint, 0 mismatches (28764 checks at 50021). Supply identity
  sum_q R_q = 2t - P asserted exact at every checkpoint.
- BAND SIGNATURE (what the gear-blind margin could not see): gears q <= sqrt(y)
  active from slot 1 (C4's servers); fresh gears activate at exactly
  t_act = (q^2-1)/6 - k_lo + 1 and follow R_q(t) = 1 + pi(m(t)/q) - pi(q) +
  T_q(t) with T_q == 0 for m(t) < q^3 EXACTLY (measured 0.0000 for all
  q > y^(2/3); worked: q=997 at y=2003, R(W) = 389 = 1 + pi(4024) - pi(997)).
  Composite-cofactor term dominates small gears (gear 5: 76% of R at y=2003).
- LOAD/SCHEDULE: S_pair(t) = exact nontrivial cross-root class hits over all
  pairs (roots-of-unity schedule); tau(t) = (t-P)/S_pair = X-demand share.
  tau rises monotonically in t (no interior peak), maxes at window END, and the
  max DECLINES with y: 0.314/0.282/0.249/0.222 at y=503/2003/10007/50021.
  KEY ANSWER: no depth range exists where X's demand exceeds the freedom-free
  schedule, and the slack (3.2-4.5x) grows with y (S_pair/W ~ lnln^2 vs
  demand/W -> 1). Capacity can never be the contradiction.
- WHERE THE EQUATION LIVES: compression. In reality t-P <= n2 <= S_pair
  identically (cross-hit <=> double, both directions checked). Measured mean
  multiplicity S_pair/n2 = 4.38 at 50021 window end; X requires
  S_pair/(t-P) = 4.50. The full distance reality-to-X is the n0 term (2.6%):
  X demands the same class hits compress 2.6% harder into distinct slots.
  The X-consistency equation should be phrased as a MULTIPLICITY/union bound
  on cross-root overlaps (how hard root classes can pile up), not a count.
  Multiplicity profile measured: 5.4 near window bottom -> 4.4 at end (y=50021),
  rising with y; no closed form fitted yet (candidate: second moment of active-
  pair density - offered to Constructor/Lateral for the flagship equation).

## Mechanic round 5 (2026-08-18) - multiplicity distribution vs independence
Tool: research/multiplicity_census.py (y=50021 in 63s). Data:
research/data/multiplicity_hist.csv (full distributions + both null models),
multiplicity_summary.csv. Identity: slot-cap => mu(k) = omega_G(mL)*omega_G(mR)
exactly; sum mu = S_pair and #{mu>=1} = n2 reproduce round-4 values exactly.
- INDEPENDENT-PAIRS NULL (CRT classes, independent across pairs; exact
  Poisson-binomial via DFT over all 13.2M pairs) misses P(mu=0) by 6.6x
  (0.041 vs 0.273 at y=50021) - the deviation IS the located structure. Real
  compression 4.38 vs null 3.32: the machine
  compresses 32-46% harder than independence; ratio declines with y (1.46 ->
  1.32), absolute gap grows (0.89 -> 1.06).
- THE CARRIER IS THE PRODUCT STRUCTURE: real var = 4.1x null, tail mu>=9 = 16x
  null. A second null keeping the product (mu' = omega'L*omega'R, independent
  Poisson-binomial sides) reproduces the real var and tail to a few % AND P0 to
  1.4 points. Exact bonus identity: null2 mean - real mean = sum_q p^L_q p^R_q
  -> 0.0911 = primezeta(2) - 1/4 - 1/9 (slot-cap covariance, 4 decimals).
- MOMENT ANSWER FOR THE FLAGSHIP: the 4.38-vs-4.50 gap is ZEROTH-moment only.
  cond = mean/(1-P0), mean pinned by arithmetic => cond_X - cond_real is
  equivalent to P0_X = P0_real - n0/W. Variance/tail carry the real-vs-
  independence excess but NONE of the X-gap. In product-model language: X
  demands the zero-mass sit ~n0/W (6.5% rel.) below the model baseline; the
  real window already sits 1.3-1.4 points below it, split as (a) the model's
  twin-mass overestimate (both-zero mass 0.0242 vs real 0.0187; ratio 0.85 ->
  0.77 down the ladder - the HL correction independence misses) and (b) a ~3%
  singles-mass deficit. The compression frontier is exactly: how low can
  P(omega_L=0 & omega_R=0) (the twin mass) go below the product baseline -
  everything else in the distribution is product-structure bookkeeping.

## Mechanic round 6 (2026-08-18) - the zone's generic edge located at y ~ 2-5e6
Tools: research/inversion_zone.py, twinmass_deciles.py. Data:
research/data/zone_summary.csv, zone_curves.csv (S1/M2/P/R dense checkpoints),
zone_anatomy.csv, twinmass_deciles.csv. Calibration matches Constructor (6.5,
2.9 at 503/2003; extent 3..17206 at 10007). Full-window scans to y=100003
(1.67e9 slots); prefix T=8y above (justified: zone_hi/y collapses 5.5 -> 0.031
across full scans, R declines past the zone).
- sup R(y) CROSSES 1: zone nonempty through y=2000003 (16 slots, t in [14,72],
  sup 1.031), EMPTY at 5000011 and 10000019. Near-threshold flicker (500009
  bulk-dips to 0.983 with a 21-slot zone at t in [29,50]). Bulk sup (t>=64,
  convention-robust): 2.652 (503) -> 1.103 (50021) -> 1.010 (2e5) -> 0.944
  (1e7); (supB-1) ~ y^-0.6 (fit). Density reading: CS efficiency stays
  0.92-0.97 while bottom-band prime density 6/ln y fattens t-P; crossing at
  ln y ~ 15 as measured. WARNING for zone-based hopes: raw sup at t<10 is
  partially CIRCULAR (an actual boundary twin spikes R by shrinking t-P) -
  at 5e4-2e6 the zone is mostly an early-twin detector, as Constructor
  predicted; no moment-only forcing survives past y ~ 5e6.
- ANATOMY at argmax: double mass concentrated at m in {4,6,9,12} (products of
  omega 2-4); m=1,2,3 absent from the bottom band (lone-gear members sit beside
  primes - the fragile census). Concentration = the zone's engine. Worked
  y=2003 t*=24: {0:15,4:7,6:2}, CS=8.70 > t-P=3 forces n0>=6; 6 twins real.
- TWIN MASS BY DEPTH (round-5 question closed): ratio to flat product baseline
  declines 0.98 -> 0.70 across deciles (y=50021), but an HL 1/ln^2(member)
  allocation reproduces every decile to 1.000 +- 0.003: NO band structure in
  the twin mass at 0.3% precision - the 0.77 global is pure density falloff.
  Depth cannot be a lever in the compression frontier.

## Mechanic round 7 (2026-08-18) - saturated-run census to member 7.2e10
Tool: research/saturated_runs.py (one absolute primality scan, k to 1.2e10,
231s - runs are gear-free objects, window census = truncation). Data:
research/data/satruns_ge10.csv (all 757 runs L>=10), satruns_records.csv,
satruns_renewal.csv, satruns_windows.csv (decile censuses, 4-y ladder).
- (1) L* = 13 STANDS to member 7.2e10 - no L=14 anywhere (every slot scanned).
  But it RECURS: six L=13 instances (members 14711, 3.69e8, 5.24e9, 1.15e10,
  5.08e10, 5.76e10); L=12: 21; L>=10: 757. Measured L->L+1 rate ratio ~0.3 at
  depth => first L=14 heuristically due within members ~1e11-1e12: 13 is a
  RECORD ON A SLOWLY GROWING CURVE, not a wall. Do not build bounds on "13".
- (2) Absolute-landmark law refined: every window sees the same integers;
  y=2003/10007 max at k=2452, and windows whose bottom EXCLUDES a landmark
  inherit the next instance (y=50021 and 200003 both max L=13 at k=61501443).
  CORRECTION for Lateral: none of the six L=13 side words is strictly
  L/R-alternating (the landmark reads RLLRRLLLLRLRL, with an LLLL block).
  "Perfect alternation" is load-only (one prime/slot); side words are blocky.
- (3) Renewal GROWS: L>=8 runs per member-decade 19, 66, 244, 972, 4297,
  ~22600 (decades 5..10), factor 3.5->5/decade; per-slot rate matches
  (6/ln m)^8 to a few % (decade ratios (d/(d+1))^8). Count/decade ~
  10^d/ln^8 -> unbounded: X's must-kill object grows increasingly abundant
  while its max length crawls. In-window depth structure is smooth density
  falloff only (y=50021 deciles 424 -> 113 for L=8), no band anomalies.

## Formalist round 1 (2026-08-18)
The HORIZON THEOREM is now kernel-checked: proofs/Horizon.lean (namespace `Horizon`,
mathlib-only imports, builds in the lake project alongside BlockedSlots, zero sorry,
axioms = [propext, Classical.choice, Quot.sound] via AxiomCheck.lean). Statements:
- `Horizon.exists_prime_factor_lt`: y < m, m < y*y, m not prime → ∃ p prime, p < y ∧ p ∣ m.
- `Horizon.prime_of_no_prime_factor_lt`: contrapositive — in the open window (y, y*y),
  no prime factor strictly below y means prime.
- `Horizon.twin_of_no_prime_factor_lt`: y < m, m+2 < y*y, no prime p < y divides m or
  m+2 → m and m+2 are both prime.
Note the STRICT bound p < y: this is sharper than BlockedSlots.survivor_iff_twin (q ≤ y),
formalising "the top gear's unique acts are boundary only" for the open interior.
Composable by anyone: import Horizon. Next target: layer law's arithmetic core
(novelty = {y^2} ∪ {y*c}), a minFac-characterisation argument on the same machinery.

## Formalist round 2 (2026-08-18)
The LAYER LAW's arithmetic core and the SLOT-CAP LEMMA are now kernel-checked:
proofs/Layer.lean (namespace `Layer`, mathlib-only, registered in lakefile, zero sorry,
standard axioms only via AxiomCheck.lean; slot_cap needs only [propext, Quot.sound]).
- `Layer.slot_cap`: for q ≥ 3, ¬(q ∣ m ∧ q ∣ m+2) — no gear blocks both members of a
  slot (it would divide 2). The constructor ledger's overlap-free floor, now formal.
- `Layer.minFac_lt_or_eq`: no prime strictly in (y, y'), m composite, m < y'² →
  lpf(m) < y ∨ lpf(m) = y.
- `Layer.eq_mul_prime_of_minFac_eq`: lpf(m) = y, y² < m < y³ → m = y·c with c PRIME,
  y < c (strongest form — c prime, not just lpf(c) ≥ y; no Bertrand used).
- `Layer.layer_novelty` (the composite law): no prime in (y, y'), thin layer y'² ≤ y³
  (holds for consecutive primes from y = 3, caller discharges it), m composite,
  y² < m < y'² → (∃ p prime < y, p ∣ m) ∨ (m = y·c, c prime, c > y). I.e. one layer's
  composite novelty is exactly {y²} ∪ {y·c : c prime} — boundary point excluded by the
  open bounds, so this + Horizon covers the whole ladder of windows.
The gap hypothesis is phrased identically to BlockedSlots.survivor_step's (∀ q, q.Prime
→ y < q → q < y' → False): the three files compose with no adapters. Next target:
zero-slack/supply identity sum_q R(q) = 2N − P as a Finset partition (slot_cap + lpf
attribution are its two legs, both now kernel-checked); alternative: h(2) ≥ d.

## Constructor round 2 (2026-08-18)
Tools: research/double_onset.py; full derivation in docs/proof-search/constructor.md
sections 7-10. Executed the manager's steering: onset law priority, descent capped.
- ROOTS-OF-UNITY LAW (Lateral's pinning made an iff, verified both directions on the
  full y=47 window): slot k is hit by gear pair {q,q'} iff 36k^2 = 1 mod qq'; trivial
  roots +-1 = same-member (semiprime-multiple) slots, nontrivial roots +-r
  (r = CRT(+1 mod q, -1 mod q')) = cross-member. So a slot is DOUBLE iff 6k lands on a
  nontrivial root of unity mod some active semiprime. Doubles are one fixed subset of N
  (zero freedom); prefix double censuses need semiprime arithmetic only, no primality
  tests (offered to Mechanic). Twin-pair gears recover r = p+1, Lateral's classes.
- DOUBLE-ONSET LAW: L0(y) = first-double lag from window start. Unconditionally n2 = 0
  on the first L0 slots; under X that prefix must be PERFECTLY fragile (exactly one
  prime per slot, primes at average gap exactly 6). Unconditional cap via
  Montgomery-Vaughan Brun-Titchmarsh (named, exact): L0(y) <= L* = 27129 for EVERY y
  (ln(6L+2) <= 12 + 4/L; 6L*+2 = 162776 vs e^12 = 162755). No window anywhere opens
  with more than 27129 prime-containing slots.
- Measured (442 windows, y <= 3163): max L0 = 17 (y=13), collapsing; L0 = 0 in 153/442.
- LIMITING EVENT (decisive, recorded): the onset route alone cannot refute X. The fact that
  would contradict the forced alternation is pi(y+H) - pi(y) >= H/6 + 1, H = 6L0+2
  (superdense short-interval bound, Hensley-Richards strength - named, NOT assumed),
  and as a universal statement at onset scale it is FALSE: 310/442 real windows have a
  twin-free onset prefix, i.e. X's forced alternation is actually realised there. The
  contradiction must be CUMULATIVE - C2 margins over prefixes spanning several onset
  events (round 1's violations end at member ~283, past first double k=20). Redirect.
- DESCENT (one page, per caution): exact step - X at y => every layer band
  (y'^2, y''^2) above y is twin-free. Unproven input in one sentence: "every window
  has a twin in its top c-fraction" = Reduction A with a constant (named, not assumed).
  LAYER-LAW WEAKENING identified: the bands tile (y, y^2), so the induction needs only
  "SOME single layer band above y contains a twin" - an interval of length ~2y'g(y')
  at its own machine's horizon, layer scale not window scale, where the layer law
  leaves twinhood to gears <= y' plus a <=3-element exception list. Still bounded-gap
  strength. Measured slack: band/stride ratio 2.2 (y'=97) -> 231 (y'=9973). Stopped.
- For Formalist: slot-cap lemma (q|6k-1 & q|6k+1 => q|2) and the roots-of-unity iff
  are two-line kernel candidates on the BlockedSlots machinery.

## Lateral round 2 (2026-08-18) - anomaly resolved into identities; extremality settled by enumeration
Tool: research/overcount_census.py. Full derivation: docs/proof-search/lateral.md round 2.
- The z=+6.1/z=-5.9 anomaly is now a THEOREM (difference of two exact formulas):
  (i) Real overcount/lone are a pure divisor census, verified EXACTLY equal to the
      window array (marks 6127, overcount 335, survivors 54208, lone 5465):
      overcount = SAME + B, SAME = sum over members of (omega_G - 1) = semiprime
      census (= 190, one per gear pair: every qq' <= 6K+1 lands exactly once as a
      member; higher multiples never == +-1 mod 6 in range; triples out of range),
      B = 145 slots with both members gearful (10 = twin own-slot pins; 135 at the
      Bezout representatives of q'b - qa = +-2). Multiplicity census {1:5465,
      2:319, 3:8}.
  (ii) Random-phase side needs NO simulation: P(q kills k) = 2/(q-1) exactly,
      independent across gears, giving closed forms E[marks] = sum (K-floor(K/q))
      *2/(q-1) = 6126.22, E[overcount] = 287.13, E[lone] = 5560.57; Monte Carlo
      agrees at |z| < 1 on all metrics.
  Anomaly = 335 - 287.13 = +47.87 and 5465 - 5560.57 = -95.57; the lone deficit
  closes by the same accounting (Delta_lone = Delta_distinct - Delta_multi =
  -47.09 - 48.48): the ~48 deterministic coincidences are counted once as lost
  distinct slots, once as gained multi slots. One cause, two faces. Formalist
  note: the census identity is "kills = divisibility" bookkeeping, mechanisable.
- EXTREMALITY REFUTED (exact full enumeration, no sampling): the real phase
  vector +-u' is merely HIGH on waste metrics (top 10-25%), never extremal:
  argmax/argmin only in the degenerate {5,7} mirror space (6 configs); in the
  full 2-teeth space of {5,7,11} (11550 configs) it ranks 1716th on overcount,
  2536th on lone; {5,7,11,13} mirror space rank 18/180. Window sweep: argmax at
  2 of 7 window lengths only. No variational handle exists; "special point of
  phase space" = "the census is deterministic", nothing stronger.
- Only non-formula piece left in overcount: B's positions (Bezout reps of
  q'b - qa = +-2; gap-2 pairs pin at u', larger gaps scatter). Candidate next:
  gap-graded closed form for split positions -> complete overcount formula at any
  scale, linking sqrt-scale prime gaps to the higher window's ledger; or hand the
  exact addresses of the 145 split + 8 triple slots to the constructor's
  bottom-band push.

## Lateral round 3 (2026-08-18) - gap-graded split law; overcount is now a formula
Tool: research/split_gap_law.py. Derivation + tables: docs/proof-search/lateral.md round 3.
- THE LAW (verified vs brute CRT, all 2850 prime pairs q < q' <= 400, zero fails):
  the split class of pair (q, q'=q+g) - q kills left, q' kills right - has least
  representative x = (q'(b0 + i*q) - 1)/6 with m0 = (-2 q^{-1}) mod g,
  b0 = (2 + m0*q)/g, i = (q'-b0)*q^{-1} mod 6; the other class is P - x, P = qq'.
  This is the SUMMARY's "nontrivial root of 36k^2 = 1 mod qq'" in closed form.
  Depth x ~ P(m0/g + i)/6; m0 = 0 iff g = 2, so g=2 is the UNIQUE gap with b0 = 1
  identically: its split pins at x = u' <= K in every window at every scale.
  Other gaps: floor depth ~P/(6g), reached only when the mod-6 alignment i=0 lands.
- COMPLETE OVERCOUNT FORMULA, exact at three real scales (machine window,
  each piece checked independently, total vs window array):
  overcount = SAME + PAIRSPLIT - CORR;
  y=53: 250+296-147 = 399; y=101: 1157+1490-815 = 1832; y=211: 6367+8651-5185 = 9833.
  SAME = inclusion-exclusion over squarefree gear products (pure floor counting);
  PAIRSPLIT = law classes only (no CRT, no sieve); CORR = multi-gear-side overlap
  (census-exact; NOT small at scale; mechanically expandable to floor arithmetic
  via higher (s_L,s_R) product-pair terms if needed - deferred).
- PAYOFF FOR CONSTRUCTOR: the window's split-double supply is an explicit
  functional of the prime-pair gaps below y: PAIRSPLIT = sum F(g, q mod 6g; K).
  Gap dependence is real and sharp at the largest pairs (P > 3K, where alignment
  bites): twin pairs hit 100% at every tested scale (y=101/211/503) vs non-twin
  ~51%. TWINS BELOW y ARE THE UNIQUE GAP CLASS WITH UNCONDITIONALLY GUARANTEED
  CONTRIBUTION TO THE LEVEL-y^2 DOUBLES LEDGER (the law forces x = u' <= K);
  everything else is residue-alignment-conditional. The self-reference of 17d,
  quantified: doubles supply = (guaranteed, from T(y) twins) + (alignment-rated,
  from all other pairs), both computable per pair by floor arithmetic.
- Note for the bottom-band push: the guaranteed g=2 pins sit at u' <= y/6 - the
  guaranteed doubles live exactly in the bottom band. Offering next: bottom-band
  double-onset supply (finite list of pairs able to place a split below slot t),
  or CORR formula-ization; coordinator's pick.

## Constructor round 3 (2026-08-18)
Tools: research/cumulative_margin.py; full text constructor.md sections 11-12.
Consumed Mechanic prefix CSV (reconciles: their minMargin includes member y itself).
- CUM STATED AND SETTLED. CUM: every window (y,y^2) has a run I with P(I) > N(I);
  CUM_band: the run within (y, y+Delta), all measured violators within 700 of y.
  VERDICT (proved, two lines each way): CUM is EXACTLY EQUIVALENT to Reduction A -
  lossless reparametrisation, gears drop out; there is NO ingredient weaker than the
  conclusion. Placement: form (b) h(L)>=d strictly oversufficient (review 7.2/s4);
  review's tail bound N(L) <= P exp(-cL/y) sufficient and weaker, sieve-side, still
  the genuine open middle; CUM = the prime-side dual at exactly conjecture strength.
  Sieve side and prime side are pigeonhole-duals of one ledger; transfer is zero-cost
  and zero-gain (parity floor beta_2 ~ 4.3-4.9 vs 2 on one side, density 1/6 vs 1/ln
  on the other).
- Full-window margin data (y = 47..5003): E(y) = 7, 4, 3, 3, 3, 3, 3 - collapses to
  a FLAT 3; realising runs shrink to 3-5-slot clusters just above y (283, 1277-1303,
  2657-2713, 5639-5659); min M(t) >= -1 from y=199 up, negativity only in the first
  slots. As y grows the pigeonhole surplus vanishes: CUM's measured margin over bare
  twin-existence is ~2 slots and shrinking. Diagnostic + literature bridge, no
  logical leverage. Recommend the team retire count-side attacks on X.
- LAYER-BAND ROUTE SCOPED vs literature (one page, sec 12). Need: twin in every band
  (y'^2, y''^2) - length x^(1/2+o(1)) at height x = y'^2, thinnest 4*sqrt(x)+4
  exactly when (y',y'') is itself a twin (self-reference = binding case). Proven:
  Maynard-Tao density is AMPLE (polylog average spacing between bounded-gap pairs -
  surplus x^(1/2)/polylog); localisation stops at exponent 0.525 (Alweiss-Luo 2018,
  arXiv:1707.05437, anchored at the Baker-Harman-Pintz single-prime floor - any
  improvement below 0.525 for pairs improves single primes first). Curiosity:
  x^0.525 < 4 sqrt(x) until x ~ 1.2e24, but AL is asymptotic/ineffective - no
  theorem at accessible heights. LIMIT TOWER, in order: T1 a prime in every band
  (OPEN - Legendre-class; not implied by RH; Cramer suffices) -> T2 bounded-gap pair
  in every band (exponent deficit 0.025) -> T3 gap exactly 2 (parity, 246 -> 2, no
  partial result). The descent input halts at T1 - an IMPORTED CORPUS LIMIT (0.525
  is a fact about published methods, not the machine); the machine event beneath it
  (thinnest bands sit exactly at twin endpoints) remains uninterrogated.
- Net strategic read for the manager: both round-3 avenues reach named external
  limit-events (parity/superdensity; Legendre localisation - the latter imported).
  The constructor's exact-ledger toolkit has met its limiting events on
  count-shaped statements; remaining in-corpus
  hope is the review's multiplicative tail route (F(M+q)/F(M) aggregate) and the
  Lateral/Mechanic structural lines (fuel words, Bezout split supply), which the
  ledger constrains but does not decide.

## Formalist round 3 (2026-08-18)
The SUPPLY IDENTITY is now kernel-checked as a Finset partition: proofs/Supply.lean
(namespace `Supply`; the first composing file - imports Horizon AND Layer; zero sorry,
standard axioms only, registered in lakefile, AxiomCheck extended).
- `Supply.minFac_mem_gears`: in the window (y, y*y), a composite's root lpf(m) is a
  prime < y (Horizon discharges it; root = minFac).
- `Supply.card_composites_eq_sum_roots` (THE IDENTITY, partition form): for ANY Finset
  S with every member in (y, y*y): #composites(S) = sum over primes p < y of
  R(p) = #{m in S : m composite, minFac m = p}. Root attribution is a function, so
  the ledger is overlap-free by construction - sum_q R(q) = C exactly.
- `Supply.card_eq_primes_add_sum_roots` (ledger form): #S = P + sum_p R(p).
  C = 2N - P is call-site arithmetic once S comes in N pairs.
- `Supply.roots_ne` (slot-level corollary via slot_cap): odd m => lpf(m) != lpf(m+2) -
  a double slot's two kills always come from distinct gears.
Composability note: the window hypothesis is per-member (∀ m ∈ S, y < m ∧ m < y*y), so
S is any Finset - intervals, the ±1 mod 6 members, or prefixes all instantiate it; the
constructor's prefix statements can be built on these fibers directly. Next target:
zero-slack census pinning under Condition X (n1 = P, n2 = N - P as prefix Finset
statements - the substrate C2 sits on); alternative: h(2) >= d product inequality.

## Constructor round 4 (2026-08-18) - flagship: the X-consistency equation
Tool: research/x_consistency.py (consumes Lateral's split_gap_law closed forms;
all identities asserted at every prefix, y = 101, 211, 503). Full text:
constructor.md section 13.
- ARITHMETIC CENSUS THEOREM (unconditional substrate): slot type by gear marks =
  census exactly: no mark <=> twin (horizon), one mark <=> fragile, both <=> double.
  Identity at every prefix: P(t) = t - D(t) + n0(t).
- THE EQUATION: X(y) <=> P(t) = t - D(t) for every t <=> p_k + d_k = 1 at every
  slot. Demand = prime census of (y,y^2); supply D(t) = union of split classes
  +-x_{qq'} mod qq' in Lateral's closed form - explicit functional of primes/gaps
  below y. Prefix-graded: g=2 pairs (twins below y) are the UNIQUE unconditionally
  pinned supply (m0=0, pin u' <= (y+1)/6, bottom-anchored); all other gaps enter at
  depth ~P/(6g) only on mod-6 alignment (~51% of large pairs).
- OVERDETERMINATION TEST, answered: degrees of freedom ZERO on both sides; formally
  y^2/6 equations over pi(y) gap inputs, but the census theorem makes demand side
  ITSELF gear arithmetic (horizon = "primality above y IS non-divisibility below"),
  so the system collapses to n0(t) = 0 - no residual structure. What X needs P(t)
  to do: sit at its unconditional POINTWISE FLOOR t - D(t) at all N prefixes.
  Below-conflict impossible (floor = the identity's minimum). Above-conflict vs
  Montgomery-Vaughan: headroom rho(t) = (t-D)lnH/2H provably <= 1, measured max
  0.4687 / 0.4785 / 0.4828 (y = 101/211/503), drifting to 1/2: THE FORCED FLOOR
  SITS AT HALF THE MV CEILING - the parity factor 2 photographed live. Any theorem
  separating P(t) from its floor at a single t IS a twin-existence theorem
  (separation = n0(t)).
- VERDICT: the equation is SATISFIABLE, for an exact reason - forced value =
  unconditional minimum, all unconditional ceilings a parity factor above. Genuine
  new content = the supply decomposition: X's doubles budget has exactly one
  guaranteed line item, the g=2 pins from twins below y, measured 5-9% of split
  incidences at every scale and depth (8.9/7.0/5.4% full window; 8.9/6.4/6.0%
  bottom band); the other 91-95% is alignment-conditional and empirically ample.
  Self-reference quantified, not closed. Priced completeness note: squeezing the
  MV constant 2 -> 1 at H ~ y^2 would bite rho -> 1/2, and that constant's
  rigidity is itself parity-class (Motohashi/Siegel-zero linkage for the
  progression form). The ledger and the analytic parity limit are one object, two faces.
- Coordination: Mechanic's per-gear R_q(t) composes as the attribution-graded
  demand side when posted; Lateral's CORR formula-ization would upgrade the
  incidence counts (CORR overlap measured large: +806/+5162/+40960) to distinct-
  double counts in closed form - that is the remaining formula gap in D(t).

## Harvester round 1 (2026-08-18) - adjacent-statement survey + Polignac/Goldbach transfer
Full survey: docs/proof-search/harvester.md. Result-first: nothing in the corpus touches
Legendre-class band statements or any fixed-gap Polignac CONJECTURE (parity/localisation
limit-events, as priced in rounds 2-3); the harvestable layer is the REDUCTION FRAME and the
exact finite laws. Ranking (reachability x value):

| rank | candidate | reach | value | note |
|---|---|---|---|---|
| 1 | Per-gap Polignac reduction iff + slot-cap transfer, in Lean | done | moderate | first bite, below |
| 2 | Goldbach window reduction + exact converse, in Lean | done | modest | same file |
| 3 | g=2 pinning theorem (only twins pin their split class at u' <= (y+1)/6) | high | small-mod | next bite |
| 4 | Overcount census identity (SAME + PAIRSPLIT - CORR) as formal theorem | high | low-mod | = Constructor's CORR ask |
| 5 | F(2,y) table -> OEIS/data note (F(2,53) unfinished at >= 420) | compute | small | new data per review |
| 6 | Fragile constant-2 law | HL-class | - | unconditionally unreachable |
| 7 | Band statements (T1/Oppermann/Brocard), fixed-gap Polignac | ~0 | - | retired |

FIRST BITE EXECUTED - proofs/Polignac.lean, BUILDS CLEAN, standard axioms only
([propext, Classical.choice, Quot.sound] via #print axioms on all nine theorems), the
kernel ledger's 6th file. Contents:
- slot_cap_gap: odd prime blocks both members of a gap-2d slot => q | d (Layer.slot_cap
  content = the d=1 case, recovered as slot_cap_twin). This is THE transfer condition:
  every corpus law whose proof rests on slot-cap holds verbatim for gap 2d at gears
  coprime to d; gears q | d collapse to one residue = the HL factor, mechanically.
- survivorGap_iff_pair: windowed survivor <=> prime pair at gap 2d (any d).
- gapPairs_infinite_iff_survivor_in_window (d): PER-GAP IFF - Polignac for 2d is
  EQUIVALENT to the windowed survivor statement for the d-pattern. d=1 recovers
  BlockedSlots.twins_infinite_iff_survivor_in_window (survivorGap_one_iff: definitional).
  Sharpens Ziller-Morack Thm 4.1 (sufficiency, all differences at once) to machine-
  checked per-difference equivalences.
- goldbach_of_survivor / survivor_of_goldbach_rep: N is a sum of two primes if the paired
  sieve (n, N-n) has a survivor with both parts above sqrt N; converse exact on central
  representations. The ZM frame trio (twins / Polignac-d / Goldbach) is now formal.
All three statement families verified computationally first
(research/polignac_transfer_check.py: zero fails; d in {0,1,2,3,5,6}, all even N < 2000).
Strategic note: Horizon and Supply are already gap-blind; after this file the only
genuinely twin-specific objects in the corpus are the phase vector +-u' and its g=2
pinning - which is why rank 3 is the next bite (it is the first structural fact that
DISTINGUISHES twins from other gaps inside the general frame, and it feeds Constructor's
supply decomposition directly). Publishable unit if wanted: formalisation note =
Polignac.lean + BlockedSlots.lean + the F(2,y) data.

## Formalist round 4 (2026-08-18)
The ZERO-SLACK CENSUS is now kernel-checked: proofs/Census.lean (namespace `Census`,
sixth lakefile target, zero sorry, standard axioms via AxiomCheck). Slots k carry
(6k-1, 6k+1); per-slot counters use real Nat.Prime (composable with Horizon/Supply);
all statements over ARBITRARY Finset T of slots (prefixes = range t, but windows and
layer bands specialise identically).
- `Census.census_partition`: n0 + n1 + n2 = N.  `Census.comps_eq`: C = n1 + 2*n2.
- `Census.primes_add_comps`: P + C = 2N.  `Census.primes_eq`: P = n1 + 2*n0.
- `Census.n0_eq_zero_iff`: n0 = 0 <-> no slot in T is a twin pair (= Condition X on T).
- `Census.census_pinned` (THE PINNING): n0 = 0 -> n1 = P and n2 = N - P; also
  `census_pinned_add` (n2 + P = N, subtraction-free) and `census_pinned_prefix`
  (T = range t: hX : no twin slot below t -> n1(t) = P(t), n2(t) = t - P(t)).
This is the demand side of the round-4 flagship, formal: under X the doubles count is
an exact functional of the prime census, per prefix, with zero freedom. Constructor
can cite census_pinned_prefix directly for the X-consistency equation's demand leg.
Next target (proposed): the bridge identity - compsIn(range t) = Supply's
root-partitioned sum over the member Finset, giving sum_q R_q(t) = n1 + 2*n2
kernel-checked end to end (LHS skeleton of the X-consistency equation).

## Lateral round 4 (2026-08-18) - master supply formula, exact at every prefix
Tool: research/supply_formula.py. Derivation: docs/proof-search/lateral.md round 4.
- MASTER FORMULA (CORR formula-ized, one signed sum, all floor arithmetic):
  overcount(t) = sum over coprime squarefree gear-product pairs (s_L | 6k-1,
  s_R | 6k+1), total >= 2 gears, of (-1)^{#gears} N(s_L,s_R;t); each N is one
  CRT class mod s_L*s_R. One-sided terms = SAME, single-single = PAIRSPLIT
  (gap law), both-sided >= 3 gears = -CORR. Both-sided restriction = B(t) =
  # slots with both members gearful.
- CONSTRUCTOR'S n2 EXACTLY (the flagship supply side): n2(t) = B(t) - U(t),
  U(t) = #{u'(q) <= t : partner member gearful} - finite, explicit, confined to
  the bottom y/6 slots (31 slots at y=211). Multiplicity bridge: overcount =
  SAME + U + n2 (n2 counts distinct both-composite slots; all stacking is in
  SAME, all prime-member exceptions in U). Spectrum at y=211: {1:1846, 2:2037,
  3:1465, 4:1038, 5:294, 6:108, 7:6} - hubs to depth 7, absorbed exactly.
- VERIFIED AT EVERY PREFIX t (max abs diff over all t in [1,K], not spot
  checks): y=101 (2940 terms) and y=211 (17022 terms): overcount 0, B 0, n2 0.
  t=K (y=211): SAME 6367 + U 31 + n2 3435 = overcount 9833. OK.
- AVAILABILITY SCHEDULE (bottom band, y=211): u' pins arrive first and alone
  (t = 1,2,3,5,7,10,...) - ALL early double supply is prime-membered (U-type,
  invisible to n2). First SAME at t=6 (35). FIRST n2 SLOT AT t=20 = (119,121),
  matching the Constructor's measured onset exactly; its anatomy: split(7,11) +
  split(17,11) - hub(7*17|11) = 1. Under X the demand n2(t) = N(t)-P(t) has
  ZERO supply before t=20 for y >= 211. n2 growth after onset ~0.463/slot.
- X-CONSISTENCY EQUATION now fully two-sided: under X, N(t) - P(t) = B(t) - U(t)
  for every t; left = prime census of (y,y^2), right = floor arithmetic over
  primes/gaps below y. Caveat: term enumeration O(#products^2) - fine to y~500,
  needs pruning beyond (formula itself scale-free). U test must be "partner
  gearful", not "partner prime" (boundary: partner prime just above y).
- Offering next: overdetermination scan (where do the two sides' derivatives
  bind in real windows) or enumeration pruning to make the equation testable at
  y ~ 1e4; coordinator's pick.

## Lateral round 5 (2026-08-18) - machinery at y=10^4; the binding defect IS the twin count
Tool: research/derivative_scan.py. Full log: docs/proof-search/lateral.md round 5.
- SCALE REACHED via pruning-by-role: PAIRSPLIT evaluated by the gap law alone
  (no product enumeration) and verified EXACTLY against sieve incidence
  sum omega_l*omega_r: 13,861 pairs / 301,026 incidences (y=1009) and 753,378
  pairs / 43,908,326 incidences (y=10007). U by pure arithmetic == sieve (133 /
  1023 slots). Sieve spot-verified vs trial division (2000 slots, 0 miss, both).
- REALITY FORM OF THE FLAGSHIP IDENTITY, verified per-slot (max residual 0 over
  all 16.69M slots at y=10007): P(t) = t + T_win(t) - B(t) + U(t). Under X,
  T_win = 0: THE BINDING DEFECT OF THE X-EQUATION IS EXACTLY THE TWIN COUNT,
  one unit per twin slot. Kernel-checkable bookkeeping (all terms census
  objects) - natural Lean target coupling Census.lean to the supply side.
- DERIVATIVE SCAN (reality is exactly X-like on twin-free runs, dP = 1 - dn2
  there). Findings at y=1009 / 10007 (440,870 twins, max stride 478 ~ the
  0.47 log^3/6 law):
  (a) prime load in top-1% strides (length-weighted, depth-binned baseline):
      87% / 90% of ambient - reality's most X-like stretches carry ~90% of
      full prime load while pairing none. X needs 100% over the whole window.
      Frontier in one line: X-like at length ~478 costs 10% prime load;
      X needs length 1.7e7 at zero discount. Caveat: deficit is partly
      conditioning; its content is its SMALLNESS and its upward trend with
      scale (0.869 -> 0.901).
  (b) hub ground generic (hub-rate/ambient 0.999 / 1.006): near-binding loci
      are NOT hub-enriched - the mechanic's capacity verdict, seen locally.
  (c) bottom band is stride-hostile: max stride in first 1% of window is half
      the global max (242 vs 478); top strides live at median depth ~0.6. Any
      bound that binds only in the bottom band fights reality at its strongest.
- Numbers at y=10007 (K=16,690,008): B 11,362,820  U 1,023  n2 11,361,797
  T 440,870  P 5,769,081  overcount 38,821,888 (bridge exact); g=2 share of
  PAIRSPLIT 3.1% (4.7% at y=1009).
- Offering next: (1) the load-length frontier (max prime load vs twin-free run
  length at fixed depth - the exact curve a compression bound must dominate);
  (2) hand the per-slot identity to Formalist/Harvester for kernel-checking.

## Constructor round 5 (2026-08-18) - compression bound + tool inventory + INVERSION ZONE
Tools: research/compression_bound.py, compression_zone.py (no primality tests -
prime <=> unmarked by horizon; identities asserted). Full text: constructor.md
sections 14-16. Mechanic's moment CSV not yet posted; moments computed directly,
match their round-4 S_pair/tau at overlaps.
- COMPRESSION STATEMENT EXACT: with m_k = omega_L*omega_R, S1 = sum m_k,
  M2 = sum m_k^2 (both freedom-free floor arithmetic; M2 = 4-tuple CRT co-hits):
  X <=> the fixed schedule compresses at M_X(t) = S1/(t-P) exactly, every prefix.
  M_X/M_real = 1 + n0/(t-P): X needs 5-22% harder pile-up than reality in range.
  Contradiction needs an unconditional ceiling C(t) < M_X(t) somewhere.
- INVENTORY, computed on our exact system: union bound = floor only. Bonferroni-2
  VACUOUS everywhere (mean m > 3). Cauchy-Schwarz/Turan ceiling C_CS = M2/S1:
  MANAGER'S 2x EXPECTATION SETTLED NEGATIVE - measured: C_CS/M_X = 1.26 -> 1.58
  (y=211->5003), growing (tracks lnln-divergent dispersion) while the needed window
  narrows (1.22 -> 1.05). Observation, not a wall: the gap locates the missing
  structure in the dispersion (product tail), not the mean. Large sieve/MV = same 2nd-moment
  content (rho -> 1/2 photograph); Selberg Lambda^2 bounds n0 from ABOVE (factor
  ~4 over HL) - wrong direction vs n0 = 0.
- INVERSION ZONE (new, sharpest): R(t) = (S1^2/M2)/(t-P) > 1 forces n0 > 0 by
  moment arithmetic alone. Zone NONEMPTY at every y tested: sup R 19.6 (y=101),
  6.5 (503), 2.9 (2003), 1.44 (10007); extent [~5, 17204] at 10007. Worked: y=503
  t=4: S1=3, M2=5, CS=1.8 > t-P=1 - twin (521,523) forced, not searched. Turning
  the zone into a theorem needs P(t) > t - S1^2/M2: density 0.42-0.80/slot =
  0.07-0.13/integer, superdense class - parity re-entry point (i). Ceilings can't
  dip below parity-twisted configurations on same moments - re-entry point (ii),
  measured 26-58% overshoot vs 5-22% need. Asymptotically the zone degenerates to
  the early-twin detector (CS efficiency -> 1 at the immediate bottom).
- THE EDGE (one paragraph, sec 16): three non-generic features unused by sieve
  axioms - (a) freedom-free placement (bottom-band n2 starvation: no double before
  absolute slot 20, U-pins subtract guaranteed supply from n2 exactly at demand
  onset - this CREATES the inversion zone; a generic system with our moments would
  not have it); (b) mirror symmetry (palindromic prefix identities, invisible to
  translation-averaged large sieve); (c) all-order exactness via the master formula
  (beyond sieve axioms, but full use = tautology). The unexhausted direction:
  positional (mirror-aware) third-moment bounds on the starved bottom band, seeking
  R > 1 with sub-superdense prime input. Everything else reaches the parity limit
  (imported corpus boundary) by measurement.

## Formalist round 5 (2026-08-18)
The BRIDGE IDENTITY is kernel-checked: proofs/Bridge.lean (namespace `Bridge`, imports
Supply + Census, zero sorry, standard axioms via AxiomCheck). The supply and demand
sides of the X-consistency equation now meet in one formal statement.
- `Bridge.members T` = T.image lo ∪ T.image hi (the slot set's members); lo/hi
  injective, images disjoint (5 vs 1 mod 6) => `card_members`: #members = 2N.
- `Bridge.card_comps_members` / `card_primes_members`: member-side composite/prime
  counts = Census.compsIn / Census.primesIn (the two groupings count the same Finset).
- `Bridge.sum_roots_eq_census` (THE BRIDGE): for any slot Finset T with slot-level
  window bounds (∀ k ∈ T, y < lo k ∧ hi k < y*y): sum over primes p < y of
  R_p(T) = #{m ∈ members T : composite, minFac m = p} equals n1(T) + 2*n2(T).
- `Bridge.sum_roots_pinned`: under X (n0 = 0), Σ_p R_p = P + 2*(N − P) — the pinned
  demand, supply-side expressed. `Bridge.slot_roots_ne`: (6k−1).minFac ≠ (6k+1).minFac
  for k ≥ 1 (per-gear ledgers overlap-free even at double slots).
Constructor: the X-consistency equation's LHS skeleton is now citable end to end
(Horizon → Supply → Census → Bridge, all composable hypotheses, prefixes = range t).
NOTE for manager: Polignac.lean (added to the lakefile this round by another
workstream) currently fails to compile (5 errors, its own file); the six formalist
libs build green independently — `lake build BlockedSlots Horizon Layer Supply Census
Bridge` succeeds; plain `lake build` will fail until Polignac is fixed by its owner.
Next target (proposed): per-gear fiber of the bridge - R_q alone + the per-gear cap
(multiples-of-q bound), first step toward the freedom-free semiprime supply arithmetic.

## Harvester round 2 (2026-08-18) - the g=2 pinning theorem, kernel-checked
Coordinator-approved bite executed. Full detail: docs/proof-search/harvester.md sec 6.
proofs/Polignac.lean extended with section "The g = 2 pinning" (7 new theorems);
computational verification FIRST (research/twin_pin_check.py: 81 twin pairs to 3000 -
pin, class-iff exhaustive over two periods to p=150, mirror, product slot; uniqueness
scan over all prime pairs q < q' <= 400: 20 own-slot pins found, ALL g=2; zero fails).
- `twin_mod_six`: p, p+2 prime, p > 3 => p = 5 mod 6 (slot coordinate exact).
- `twin_pin`: the pair IS slot u = (p+1)/6 (6u-1 = p, 6u+1 = p+2, p | left,
  p+2 | right) - the split representative in closed form; existence trivialised by
  self-blocking (the pin is the pair).
- `twin_pin_le`: u <= (y+1)/6 for EVERY y >= p - the guaranteed double sits in the
  bottom band of every window at every scale, unconditionally. Formal location half of
  "twins below y are the unique guaranteed line item of the level-y^2 doubles ledger".
- `twin_split_class_iff`: slot k split-killed by {p,p+2} (p left, p+2 right) IFF
  k = u mod p(p+2) - the full CRT class as an iff (g=2 case of the roots-of-unity law).
- `twin_mirror_slot` (second class at P-u), `twin_product_slot` (same-member double at
  u(p+1), member = p(p+2) exactly - machine re-ingesting its own output, formal).
- `own_slot_pin_gap_two` (UNIQUENESS): an odd prime pair (q, q+g) split-killing the
  slot holding q itself forces g = 2. Only twins pin at their own slot. The
  quantitative other half (depth ~P/(6g) for g > 2, mod-6 alignment rate) stays
  paper-side (research/split_gap_law.py) - priced, not formalised.
STATUS: whole ledger GREEN - `lake build` (all 7 targets incl. Bridge + Polignac)
"Build completed successfully" (986 jobs); the mid-round failure the Formalist flagged
was my round-2 edit in flight, now fixed. Axiom audit, all 16 Polignac theorems:
standard axioms; the twin-pin seven need only [propext, Quot.sound] except
twin_split_class_iff (+Classical.choice).
Lean notes for the team: omega does NOT combine congruences across moduli - decompose
to one modulus with explicit witnesses; [MOD n] notation needs
`import Mathlib.Data.Nat.ModEq` (not transitively available via BlockedSlots);
`Nat.dvd_sub` here is the old `Nat.dvd_sub'`; `le_or_lt` absent - use `Nat.lt_or_ge`.
For Constructor/Lateral: the U-list kernel of the master formula (n2 = B - U) is now
formally characterised (existence + location + class-exactness + twin-uniqueness);
U-membership ("partner gearful") and the B-side remain the open formal gap.
Next bites offered: overcount census identity in Lean (= Formalist's bridge line from
the supply side), or F(2,53) completion for the data note; coordinator's pick.

## Formalist round 6 (2026-08-18)
The PER-GEAR FIBER + CAPS are kernel-checked: proofs/Gear.lean (namespace `Gear`,
eighth target, zero sorry, standard axioms; first file to REUSE BlockedSlots -
card_blocked_by_le is the interval bound).
- `Gear.R q S` := #{m in S : composite, minFac m = q} - one gear's ledger line, named.
- `Gear.supply_eq_sum_R` / `Gear.sum_R_eq_census`: Supply identity and Bridge restated
  in R form (definitional; cite these when talking per-gear).
- `Gear.R_le_card_multiples`: R q S <= #multiples of q in S (root => divisible).
- `Gear.R_prefix_le`: R q (members (range t)) <= 6t/q + 2 (members live below 6t;
  BlockedSlots.card_blocked_by_le on [0,6t) does the rest). Not fought sharper.
- `Gear.sq_le_of_minFac_eq` + `Gear.R_eq_zero_of_below_sq` (SHADOW LAW): a gear
  supplies nothing below q^2 - its ledger line opens at q^2. Guard discovered during
  proof: minFac 0 = 2, so the law needs 1 < m (window hypotheses give it for free);
  anyone doing per-gear counting over raw Finsets should carry the same guard.
Gear ledger state: line (R), total (= n1 + 2*n2), cap (6t/q + 2), onset (q^2). Next
target (proposed): semiprime refinement - for q < y <= q^2, gear q's class in the
window is exactly {q*c : c prime} member-wise (Layer.eq_mul_prime_of_minFac_eq),
giving R q = #(partner primes) - first exact formula of the freedom-free supply side.

## Lateral round 6 (2026-08-18) - the load-length frontier is ABSOLUTE; target L ~ 14-32
Tool: research/load_frontier.py. Full log + tables: docs/proof-search/lateral.md round 6.
- FRONTIER: maxload(L) = max prime load on twin-free L-runs (open interior; X-ceiling
  = 1/slot). Curve: 1.0000 up to L* = 13, then 13/14, 0.85 (L=20), 0.80 (25),
  0.7188 (32), 0.52 (100), 0.32-0.43 at L = maxstride. IDENTICAL at y = 1009/
  3163/10007 because the record-holders are ABSOLUTE integer landmarks: L* = 13
  at slots 2452-2464 (primes 14713..14783, perfect L/R alternation, no twins) at
  every scale; the L=100 record at absolute slot ~31,350. The frontier is a
  property of the integers; the window truncates it from below at s0 ~ y/6.
- RENEWABLE: restricting to depth >= 0.5 still gives saturated (load-1) runs of
  length 9-10 at every scale (members to 5e7). gap(L) = 0 for L <= ~10 at ALL
  depths. Caveat: persistence of saturated runs at all depths is itself an
  HL-admissible constellation statement (expected true, unprovable).
- TARGET SCALE for any compression bound: L ~ 14-32 (gap 0.07-0.28, renewable).
  For L >= 63 the gap is > 0.44 - reality never gets close; bounds aimed at
  long runs have no leverage - reality never enters that regime.
- BOTTOM BAND double face: round 5 said stride-hostile; round 6 adds LOAD-
  OPTIMAL - at y=10007 the band [s0, s0+y] contains the global record runs up
  to L ~ 100 (ceiling touched inside the inversion zone's own band). Starved of
  length, not of load. Record-run interiors show P-rate + n2-rate = 1 exactly
  (0.80+0.20 at L=25; 0.52+0.48 at L=100) - the constructor's forced perfect
  alternation is REALIZED by reality up to length 13 (saturated runs are pure
  n1: every slot one prime + one lone composite = fragile-dense).
- lpf anatomy of record runs: 57-70% of interior composites killed by gears
  <= 13 - small gears do the composite work at the frontier.
- PART 3 VERDICT: frontier runs and chain/fuel maximal strides are DIFFERENT
  extremal families. Load-extremal: short, absolute, prime-dense, constellation-
  governed. Length-extremal: deep, load ~0.3, gap-word-governed. They merge only
  at L = maxstride. Chain analysis cannot see the binding region L ~ 14-32.
- Offering next: (1) exact census of saturated runs by (length, depth) - the
  count curve of the object X must kill, vs the inversion zone's R(t);
  (2) alternation-word structure of saturated runs vs the machine's mirror laws
  (feeds constructor's mirror-aware third moments).

## Constructor round 6 (2026-08-18) - zone fate settled; moment ladder's limiting event identified
Tool: research/zone_fate.py (ladder to y = 10^7; LP moment ceilings). Full text:
constructor.md sections 17-18. Mechanic round-5 CSVs consumed; their "X-gap is
zeroth-moment only" corroborated independently at orders 2-3.
- ZONE FATE: R = eff * boost (eff = CS efficiency, boost = 1 + n0/(t-P)). No
  single crossing: the zone's generic forcing ends between y ~ 3e6 and 5e6 (sup R:
  1.44 at 1e4 -> 1.01-1.03 at 1e6-3e6; first EMPTY windows y = 5000011 and
  10000019, confirmed at T = 200000) and revives sporadically. KILLER = the
  boost side: twin surplus n0/(t-P) collapses ~1/ln^2 y (2.00 -> 1.08-1.13 at
  argmax) while eff erodes slowly (0.96 -> 0.86-0.94, lnln dispersion).
- REVIVAL LAW + adversarial verdict (max skepticism as ordered): windows opening
  with a twin in <= 4 slots revive the zone at ANY y - verified at y = 5000087,
  5000101, 5000539 (sup R = 1.923). But every twin (p,p+2) sits in the first
  slots of the window of any prime y just below p, so "the zone revives i.o."
  IS the twin prime conjecture. The inversion zone is a bottom-twin DETECTOR
  (certifies from moments + P without exhibiting the pair), never a generator.
  Unconditional content exhausted.
- MIRROR THEOREM (2 lines): k -> -k swaps omega_L/omega_R and fixes m; all
  mirror-augmented moments double, every ratio invariant. Mirror-awareness is
  VACUOUS at moment level, any order. (Answers both round-6 mirror questions.)
- THIRD-MOMENT CEILINGS (sharp LP moment-problem bounds): integer order-2 LP
  beats continuous CS by 0.3-0.5% (still refutes at the y=10007 zone edge
  t=17204 where CS breaks even: 7744 > 7702). Order-3 conservative: adds ZERO
  (cubic never in basis). Order-3 with the legitimate cap m <= (log_5 y^2)^2:
  +0.6-2.8% (basis (5,6,cap)); y=50021 band: 25,093 vs demand 25,157 - short by
  64. Window scale: ceiling 5.24 vs need 3.54 - the ~48% chasm untouched.
- NET: the moment ladder converges too slowly; the X-gap is zeroth-moment (twin
  mass), invisible to all power moments; the positional strip beyond moments =
  bottom-twin detection = the conjecture. LIMITING EVENT for the whole ladder:
  zeroth-moment invisibility (the X-gap lives in a mass no power moment weighs;
  the twin mass itself is a scoreboard, not a mechanism). Redirect to structural
  fronts - placement, pins, and word constraints are the objects moments cannot see.

## Harvester round 3 (2026-08-18) - SAME-side census kernel-checked; self-block composed with Census
Coordinator-approved bites executed. Full detail: docs/proof-search/harvester.md sec 7.
proofs/Polignac.lean extended (now imports Census - first file composing with the
formalist's census); computational verification FIRST (research/same_census_check.py:
105 prime pairs 5 <= q < r < 60, class-iff exhaustive over two periods left+right,
floor count at 11 t-values per pair, window "exactly once", own-value reps: zero fails).
New theorems (12), the first layer of the master supply formula formal:
- `six_mul_class`: slot-map inversion - for m coprime to 6, {k : 6k = c mod m} is ONE
  class mod m, any target c. `left_dvd_iff`/`right_dvd_iff`: member divisibility =
  residue condition (6k = 1 left, 6k = m-1 right). `class_rep_unique`, `not_dvd_six`.
- `card_class_Ico` (THE FLOOR-COUNT PRIMITIVE): #{k in [1,t] : k = a mod m}
  = (t + m - a)/m for 1 <= a <= m. Every floor term of SAME/PAIRSPLIT reduces to it.
- `same_left_census` / `same_right_census`: distinct primes q, r >= 5 - slots whose
  left (resp. right) member both divide are ONE CRT class mod qr, count
  (t + qr - a)/qr over the first t slots. The SAME-side pair term, exact.
- `same_census_once` (COMPOSITE ROOT LAW, windowed): a <= t < a + P => exactly one
  coincidence - "exactly once if it fits" with the fit hypotheses explicit.
- `same_left_own_value`: qr = 5 mod 6 => the class rep IS slot (qr+1)/6, member qr
  itself - "acts at its own value" explicit.
- `twin_pin_self_block` (second bite): the pin slot u of twin (p,p+2) has
  Census.slotComps u = 0 (a REAL twin slot, both members prime) yet is never a
  BlockedSlots.Survivor of any machine with bound >= p - the machine is blind to its
  own pair; the formal reason the U-pin list is invisible to n2.
STATUS: whole ledger GREEN - `lake build`, all 8 targets, 988 jobs, zero sorry. Axiom
audit: all 28 Polignac theorems standard axioms only. Polignac.lean is now 28 theorems:
the ZM-frame reductions (r1), the g=2 pinning (r2), the SAME census + self-block (r3).
Lean gotchas added to harvester.md: card_insert_of_notMem rename;
Ico_succ_right_eq_insert_Ico lives in namespace Nat; rwa-at rewrites the ModEq modulus
too (orient the equation and rewrite in the goal); induction + succ_div_of_dvd/not_dvd
sidesteps omega's no-division-by-variables limit.
Remaining formula gap after this layer: multi-gear products (squarefree s_L, s_R with
>= 3 gears - the signed CORR terms) and PAIRSPLIT's closed-form rep (Lateral's
m0/b0/i law) - the latter is the natural next Lean bite if wanted; both reduce to
six_mul_class + card_class_Ico instances.

## Constructor round 7 (2026-08-18) - THE ATTEMPTS MAP delivered
Artifact: docs/proof-search/attempts-map.md (renamed from impossibility-map.md per the
framing directive; the programme's key prose record, written to handover standard:
exact statement / limiting event / yield per route, provenance-tagged to corpus,
review, and workstream rounds).
- ORGANISATION: three limit-event classes + an equivalence ring. Class I ABUNDANCE
  (capacity 5.1, two-scale 5.2, doubles squeeze, tooth-sharing counting, tau-slack at
  scale - all first moments, each stopped by measured surplus, exact at stated
  scales). Class II SUPERDENSITY/LOCALISATION (global C1's reach ends at the e^6
  drift event; onset route limited twice over - superdense need AND the forced
  pattern realised in 310/442 windows; zone-as-theorem needs 0.07-0.13
  primes/integer, with the 0.525 BHP floor an IMPORTED CORPUS LIMIT; layer-band
  descent halts at T1 = Legendre-class, imported, before its twin content engages).
  Class III PARITY/SECOND-MOMENT (X-equation satisfiable - floor at half the MV
  ceiling, rho -> 1/2; moment ceilings vs need: divergence observed, 2x expectation
  settled negative; Selberg bounds n0 the wrong way; kappa form (b) oversufficient +
  regime gap; the X-gap is ZEROTH-moment only - the parity component is partly an
  imported corpus limit). Class IV EQUIVALENCE-TO-TARGET (CUM, run family, zone
  revival, floor separation, one-band descent - each proved at full conjecture
  strength; the ring doubles as an instant classifier for future proposals).
- NULL-LEVER LEDGER (sec 5): mirror (theorem), depth (0.3% precision), extremality
  (full enumeration), tooth-sharing cardinality, census gear-blindness, hub
  enrichment, chain/fuel-vs-binding-region - each with its closing evidence.
- SURVIVING TOOLBOX (sec 6): finite-reach refuters (C1 + inversion zone with its
  kernel-checkable n0-forcing certificates), absolute caps (L* = 27129, slot 20,
  (q+-1)/3), the exact laws (census theorem, roots-of-unity, gap law, master
  formula, g=2 pinning + uniqueness), the four photographs, the 9-file Lean ledger.
- HONEST RESIDUE (sec 7, the open list): structural fronts (L ~ 14-32 binding
  region; saturated-run census; alternation words - with the HL-constellation
  caveat carried); the review's MULTIPLICATIVE TAIL route (untried, the one
  sufficient-but-not-oversufficient statement never attacked); the Lean supply
  program's remaining gaps (PAIRSPLIT rep, CORR terms, U/B side); the zone's
  finite certificate format (publishable with the Polignac frame); F(2,53);
  and the explicit non-claim: the map records limiting events for moment/capacity/
  localisation/reformulation arguments; ideas that address the events themselves -
  or arrive outside the three classes - are open by construction.

## Harvester round 4 (2026-08-18) - PAIRSPLIT kernel-checked; master formula's formal core complete
Coordinator-approved bite executed. Full detail: docs/proof-search/harvester.md sec 8.
Computational verification FIRST (research/pairsplit_check.py: 210 ordered prime pairs
5 <= q, r < 60, split-class iff exhaustive over two periods both orientations, floor
counts, mirror role-swap, g=2 rep == pin on all twin pairs in range: zero fails;
cross-consistent with Lateral's split_gap_law closed form, g=2: m0=0, b0=1, x=u').
New theorems in proofs/Polignac.lean (built clean on first compile):
- `split_class`: distinct primes q, r >= 5 - the slots where q strikes the LEFT member
  and r the RIGHT (q | 6k-1, r | 6k+1) are ONE CRT class mod qr, floor count
  (t + qr - a)/qr over the first t slots. Mirror class = role swap. Machinery:
  Nat.chineseRemainder target c = CRT(1 mod q, r-1 mod r), funneled through
  six_mul_class at modulus qr; and-to-product via Nat.modEq_and_modEq_iff_modEq_mul.
- `split_rep_twin_eq_pin` (g=2 LOOP-CLOSER): for a twin pair, any below-modulus
  representative of the split class IS the pin u = (p+1)/6 - the pinning section and
  the PAIRSPLIT section now meet in one theorem; with twin_pin_le this is the formal
  "twins below y are the unique unconditionally guaranteed doubles line item".
- `twin_split_count`: twin split count = (t + p(p+2) - u)/(p(p+2)), = 1 exactly on
  u <= t < u + p(p+2) - the guaranteed bottom-band double, counted in closed form.
STATUS: ledger GREEN - `lake build` all 8 targets, 988 jobs, zero sorry; axiom audit
standard ([propext, Classical.choice, Quot.sound]; the loop-closer needs only
[propext, Quot.sound]). Polignac.lean = 31 theorems, four sections.
MILESTONE for the team: both structural layers of the master supply formula
(overcount = SAME + PAIRSPLIT - CORR) now have their class-and-count core
kernel-checked; the remaining formal gap is the signed multi-gear combination (CORR,
>= 3-gear products) - pure bookkeeping over the same two primitives (six_mul_class +
card_class_Ico), available as a future bite. Also still on the shelf: F(2,53)
completion for the data note (rank-5 candidate).

## Formalist round 7 (2026-08-18)
The SEMIPRIME REFINEMENT is kernel-checked (extends Gear.lean, still 8 targets, zero
sorry, standard axioms): one gear's supply line is now an EXACT PRIME COUNT.
- `Gear.semiprime_of_fiber`: composite m rooted at q with m < q^3 => m = q*c, c prime,
  q <= c (equality exactly at the square q^2 - the shadow-law onset, no special case).
- `Gear.partners q S` := fiber.image (m -> m/q); `Gear.R_eq_card_partners`:
  R q S = #(partners q S) UNCONDITIONALLY (bijection needs nothing - fiber members are
  multiples of their root).
- `Gear.mem_partners` (THE FORMULA): for q prime and members in (1, q^3):
  c in partners q S <-> c prime, q <= c, q*c in S. So R_q = #{partner primes}, exact.
- Helpers now available to all: `Gear.not_prime_mul` (product of two primes is
  composite), `Gear.minFac_mul` ((q*c).minFac = q for primes q <= c),
  `Gear.window_bounds` (window + y^2 <= q^3 adapter).
CORRECTION to the round brief (verified by counterexample before proving): the regime
q < y <= q^2 is NOT enough - m = 175 = 5*35 is rooted at 5 in window (25, 625) with
composite cofactor. The honest large-gear regime is member < q^3 (window form
y^2 <= q^3, gears q >= y^(2/3)); below that, cofactors can themselves be q-rooted
composites and the line needs the general root recursion, not the semiprime formula.
Next target (proposed): slot placement of the line - q*c = 6k+-1 determines the
semiprime slot k, connecting R_q's members to the lateral workstream's pinned classes
(placement side of the X-equation). Alternative: h(2) >= d.

## Lateral round 7 (2026-08-18) - word laws (one proved cap); HL caveat scoped
Tool: research/alternation_words.py. Full log: docs/proof-search/lateral.md round 7.
- FRAME: saturated-run words are MACHINE words (letter = the unhit side), so the
  positional mirror law k -> -k (reverse + swap L/R) applies exactly. Tested on
  90/333 maximal runs len >= 8 at y = 3163/10007:
  (a) parity theorem (proved): odd-length runs are never self-mirror (middle
      letter would equal its own complement). Data: 0 odd palindromes; even
      self-mirror runs common (16/250 at L=8).
  (b) mirror statistics: TV(dist(w) vs dist(revcomp w)) = 0.33 << TV(reverse)
      0.56 and TV(complement) 0.60 at L=8 - the symmetry is specifically
      reverse-complement, as the mirror law predicts. Letter marginal 0.4996.
  (c) duplicate words = CRT alignment: identical-word position differences
      divisible by 35 in 55% of pairs (baseline ~3%); forced-letter fraction
      0.729 by gears <= 13. Landmark 13-word unique in 1.67e7 slots.
- STRICT-ALTERNATION CAP, PROVED: strict LRLR... saturated runs are primes at
  gaps 8,4,8,4...; offsets cover Z/5 at length 7 (L-first) / 6 (R-first), so
  GEAR 5 ALONE caps strict alternation at 6 (L-first) / 5 (R-first). Data: max
  = 6 exactly, at absolute landmark slot 19125, letters LRLRLR (L-first) -
  phase matches the theorem. FOR CONSTRUCTOR: X's forced local patterns can
  never be strictly alternating beyond 6 slots; repeats (LLLL etc.) are
  mandatory - the local constraint is CRT, not alternation.
- HL CAVEAT SCOPED (full page in lateral.md): persistence(L) ("every interior
  has an L-saturated run") is equivalent to a Bertrand-type postulate for the
  L-run sequence (next run before the square of the last, tower bands tile).
  Ladder: persistence(1) = THEOREM (Brun); persistence(2) = disjunctive
  Polignac {4,6,8} - the exact provability frontier, weaker than twins, beyond
  bounded-gap 246; persistence(L>=3) = disjunctive HL at tuple size L (L=13 is
  13-tuple class). Per-y it is a FINITE computation (load_frontier.py is the
  decision procedure); only the "for all y" is conjectural. The caveat cannot
  hurt the programme: the frontier is descriptive, never a premise - if
  persistence fails anywhere, reality drifts AWAY from X and bounds get easier.
  Illegitimate use only: citing renewability to declare bounds at L <= 13
  impossible.
- Offering next: the complete word grammar - the exact set of infinitely-
  extendable letter patterns compatible with the small-gear teeth (generalizes
  the cap theorem; positive-description complement to the attempts map).

## Mechanic round 10 (2026-08-18) - T1 reopening interrogated: verdict split, event ledger clean
Tool: research/band_census.py; data: research/data/band_census_100003.csv (9,591
bands = ALL bands (p^2, p'^2) to height 1e10, every slot exact). Full working:
mechanic.md round 10.
- EVENT DEFINITION: thickness T = g(2p+g)/6 exactly => "thinnest <=> twin
  endpoints" is exact but TRIVIAL (monotone in g). The real machine event found
  beneath: for twin (6m-1, 6m+1), T = 4m and the twin's PRODUCT SLOT k = 6m^2
  sits at offset exactly T/2 - the band's center - dead by construction
  (member 36m^2-1 = p(p+2)). Verified 1223/1223. Every twin pre-blocks the
  center of the thinnest band above it - the self-reference is real, mechanical,
  and exactly ONE SLOT deep.
- CENSUS VERDICT: thin bands are NOT twin-poor. Decade-matched g=2 vs all-band
  twin density: ratio 0.984/1.018/1.006/1.002 (decades 6-9), center-excluded the
  same; the deterministic deficit is 1/(4m), invisible at scale. ZERO twin-empty
  bands of any gap class through height 1e10 (min = 2, the (25,49) band; at
  heights 1e9-1e10 the worst band holds 342 twins in 21,352 slots = exactly its
  Poisson lambda; it is g=2 only because g=2 bands are SHORTEST). Min primes per
  band = 6 - T1's object is nowhere near failing in range.
- CONSEQUENCE FOR THE DESCENT: the binding case is binding by LENGTH ALONE -
  the machine contributes one quantified dead center slot and is otherwise
  statistically generic inside thin bands. The T1 difficulty is exactly the
  imported Legendre-class localisation problem, with no additional machine
  hostility. Reopening closed with a clean ledger: self-reference = 1 slot;
  fragile centers (36m^2+1 prime) at 7.6%, density-consistent.
- L=15 HUNT running detached across rounds (wrapper PID 18504, log
  research/data/satruns_L15.log, chunk-flushed/resumable; members to ~1.2e13,
  ~15h; predicted first arrival ~5e12; chunk 1 flushed, max L=12 so far).

## Mechanic round 9 (2026-08-18) - THE FIRST L=14 FOUND; constellation model validated
Data: research/data/satruns_deep_ge10.csv (complete range, members 7.2e10..1.002e12
- the round-8 "killed" run had flushed its CSVs before dying; verified complete,
deduplicated, boundaries clean). Full working: mechanic.md round 9.
- NEW LANDMARK: L=14 at k = 46,133,660,494 (members 276,801,962,963..3,043), word
  LRRLRLRRRRLLRL, MR-verified per slot + maximality both ends. Record progression:
  10 (k=59), 13 (k=2452), 14 (k=4.6e10). L*=13 stood from member 1.5e4 to 2.8e11
  and fell where the HL-constellation model predicted (first arrival ~1.6e11,
  expected count 1.2 at the actual address - Poisson-consistent, NO deficit).
  The machine tracks constellation statistics through the record scale.
- RECORD LADDER to Lateral's CRT cap [13,32] (fit): L=15 ~5e12, L=16 ~2e14,
  L=20 ~6e20, L=32 ~3e42. The cap splits: reachable (15-16), astronomical
  (17-31), absolute ceiling 32. Record ~ +1 per factor ~40 in member. The L=15
  hunt (~6x last scan) is priced, optional.
- RENEWAL REFIT: per-slot rate of L>=8 runs ~ C/(ln m)^6.81 (8 decades, max
  ln-resid 0.24; naive independence ~8; flattening = mixture + HL corrections).
  L>=13 population now 19 instances, all MR-verified.
- For Lateral's word grammar: the L=14 word has an RRRR block and single-letter
  runs - another blocky specimen for the corridor analysis; landmark words remain
  never strictly alternating (cap 6 respected with margin).

## Mechanic round 8 (2026-08-18) - L=14 hunt status + framing directive executed
- L=14 HUNT (approved chunk): first full-range run (k to 1.67e11, members to
  1.002e12) was killed at ~70% and, because the scanner wrote CSVs only at
  completion, its in-memory findings were lost - lesson recorded. Scanner rebuilt
  RESUMABLE (research/satruns_deep.py: 8e9-slot chunks, results flushed per chunk,
  state file; one command resumes). The exhaustive bound therefore still stands at
  member 7.2e10 (no L=14; round 7). Relaunch decision left to coordinator/user.
- THE EXPERIMENT IS SHARP EITHER WAY (research/satruns_model.py): constellation-
  calibrated rates A_L are stable across decades (0.25 -> 0.12, log-slope -0.13/L)
  and predict ~2.6 L=14 instances in (7.2e10, 1e12). Finding one assigns the next
  landmark's address; finding none is a >90% quantified deficit vs the HL model -
  the first measured departure of the machine from constellation statistics, which
  would be the MORE interesting event. Round 7's zero-to-7.2e10 is only a 31%
  Poisson tail - not yet evidence of suppression. Context: Lateral's round-8 CRT
  cap bounds the whole game in [13, 32] - record growth is now a finite question
  per length, exactly the framing directive's standard.
- Note: the k=2452 landmark is itself an early-arrival outlier vs the model
  (A_13 = 8.4 locally vs 0.15 trend) - the famous landmark is a bottom-band
  constellation fluctuation, consistent with the bottom band's measured richness.
- FRAMING DIRECTIVE EXECUTED across all artifacts this round (mechanic + three
  delegated passes): impossibility-map.md renamed/reframed to attempts-map.md
  (attempt -> yield -> limiting event; imported corpus limits tagged as candidate
  reopenings; trends demoted to observations); constructor.md, lateral.md,
  human.md, harvester.md, class-tree.md tone-passed meaning-preserving;
  factual fix landed in lateral.md (landmark side words are blocky, NOT
  alternating - round-7 correction now consistent everywhere); mechanic.md
  carries the attempts ledger + event-vs-trend classification of the whole
  shared ledger. Candidate reopening flagged for the manager: the T1/0.525
  closure is an imported corpus limit; the machine event beneath it (thinnest
  layer bands sit exactly at twin endpoints) is uninterrogated as a mechanism.

## Formalist round 8 (2026-08-18)
PLACEMENT of the supply line is kernel-checked: proofs/Placement.lean (namespace
`Placement`, ninth target, zero sorry, standard axioms - sign_law needs only propext).
The ledger now knows WHERE every large-gear supply member sits, not just how many.
- `Placement.prime_mod_six`: primes >= 5 are +-1 mod 6. `Placement.sign_law`:
  on unit classes, (a*b) % 6 = 1 <-> signs agree (the sign law, formal).
- `Placement.slotOf m` := (m+1)/6 - ONE formula recovers the slot from EITHER member
  (no case split; simplification over the brief). `lo_slotOf`/`hi_slotOf`: a ≡5 (resp
  ≡1) mod 6 number IS the lower (resp upper) member of its slot.
- `Placement.mem_members_iff_slot`: for unit-class m, m ∈ members T <-> slotOf m ∈ T.
- `Placement.slot_injOn_partners` (THE INJECTION): c -> slotOf (q*c) is injective on
  Gear.partners q S (regime: q prime >= 5, members < q^3). Mixed-sign collisions die
  on Layer.slot_cap - two multiples of q at distance 2.
- `Placement.card_slots_of_line`: the line occupies exactly R q S distinct slots.
- `Placement.R_slots_eq` (PLACED COUNT): over slots [1,t) with 6t <= q^3:
  R_q = #{c prime, q <= c, slotOf(q*c) ∈ [1,t)} (carrier range (6t)).
Convention note: placement statements use slot interval Ico 1 t - slot 0 is degenerate
(members 0,1). Census identities are Finset-generic and unaffected.
Next target (proposed): (a) twin-product pin - slotOf (p*(p+2)) arithmetic connecting
Placement to Polignac's pinned classes; or (b) h(2) >= d product inequality.

## Constructor round 8 (2026-08-18) - multiplicative route OPEN: the tolerance theorem
Tool: research/multiplicative_route.py. Full text: constructor.md sec 19; the
attempts map has a round-8 amendment (limit-event class V added, residue item 2
resolved into a target). Inputs: exact F(2,y) chain 6..354 (y=5..47), F(2,53) >= 420.
- RATIO DATA: per-step r = F'/F vs window budget (q'/q)^2: OVER at 6 of 12 steps
  (5->7, 11->13, 17->19, 29->31, 31->37, 41->43); cumulative under (sum ln r
  4.078 vs 4.481); F/requirement flat at 0.32-0.44. Sharp per-step ratio bounds
  are stopped by lumpiness (same event as additive 1.8q); uniform ratio caps > 1
  are excluded a priori (pi(y) steps vs y^2 budget). Viable shape: incr <= alpha*q.
- TOLERANCE THEOREM (the result): corpus 6a's closure ("per-step bounds cannot
  deliver") holds ONLY for the odd-sum elementary chain and its 1.8 threshold.
  With sharp prime sums the tolerance GROWS: alpha*(y) = [(y^2-y)/2-354]/
  [S(y)-328] = 5.64 (y=101), 8.71 (1e4), 13.3 (1e6) ~ ln y. Hence:
  incr <= 2.5*q AT EVERY CONSECUTIVE STEP BEYOND 47 => F(2,y) < (y^2-y)/2 for
  all y => survivor in every window => TWINS INFINITE. Verified exactly at every
  prime y in [53, 1e6] (zero failures, worst ratio 0.6557 at y=113, alpha=3);
  Rosser-Schoenfeld closes y > 1e6. Observed max incr/q = 2.432 (gear 37) sits
  BELOW 2.5: the gear-37 anomaly refutes only the sharp 1.8, not the route.
- MECHANISM: saturation regime (q > F(M)) is a theorem with alpha = 1
  (F(M+q) = F2 <= 2F, incr <= F < q) but the consecutive chain NEVER enters it
  (q < F(M) throughout). In-range: no a-priori cap; the chain condition gives
  the exact split incr = (F2-F) + excess, fuel-gates the excess, and leaves two
  missing lemmas: (a) F2 - F <= alpha1*q (top-gap anti-clustering; measured
  <= 1.24q), (b) excess <= alpha2*q (fuel-merge control; measured <= 1.62q,
  fuel abundance explains the spikes, does not bound them). Corpus 5.5: gap
  structure alone cannot do it; the input is word arithmetic (forbidden
  configurations).
- LIMIT-EVENT CLASSIFICATION: clears classes I, II, IV; not class III by the
  dimension-1 test (the same increment bound would sharpen Iwaniec where parity
  does not obstruct, and is unproven even there). Names a further class (map
  amendment, class V): extreme-value control of sieve patterns (= review's
  regime gap, corpus 5.5) - an event to interrogate, not yet priced.
- DECISION POINT: F(2,53) prices the constant (alpha=2.5 <=> F(2,53) <= 486;
  alpha=3 <=> <= 513; current partial 420, incr so far 1.245q). The review's
  discriminating measurement now carries this route too - recommend finishing it.
- NET: the only route on the books whose missing lemma is about the machine's
  own gap word rather than about primes. Best-shaped open statement the
  programme has; hand the two lemmas to the structural fronts (fuel/forbidden-
  configuration machinery is native there).

## Harvester round 5 (2026-08-18) - CORR triple kernel-checked; per-term core of the master formula COMPLETE
Choice reported: CORR triple over F(2,53) - the triple reduces entirely to the proven
primitives (six_mul_class + card_class_Ico + chineseRemainder), a bounded Lean bite;
F(2,53) is open-ended compute (final uncoverable proof is the expensive step), stays
shelved. Full detail: docs/proof-search/harvester.md sec 9.
Computational verification FIRST (research/corr_triple_check.py: 20 triples from
{5..19}, all 3 role splits = 60 two-sided cases, membership exhaustive over one
period, floor counts, signed identity with overlap == triple: zero fails).
New theorems in proofs/Polignac.lean (built clean on first compile):
- `twoSided_class` (GENERAL BOTH-SIDED TERM): coprime moduli mL, mR > 1 coprime to 6
  => slots with mL | left, mR | right are ONE CRT class mod mL*mR, count (t+M-a)/M.
  Subsumes split_class and yields EVERY both-sided master-formula term in one
  statement (all squarefree gear products qualify).
- `corr_triple_class`: first genuinely new CORR case - (qr | left, s | right) is one
  class mod qrs, closed count; ten lines, pure instantiation. Other role splits are
  further instantiations.
- `corr_triple_signed` (THE SIGN, subtraction-free): |A or B| + |triple| = |A| + |B|
  for the two split classes sharing right gear s; only hypothesis Coprime q r. The
  inclusion-exclusion step formal - the triple class is exactly what the signed sum
  removes converting incidences to distinct slots.
STATUS: ledger GREEN - `lake build` all targets, 990 jobs, zero sorry; axiom audit
standard on all four. Polignac.lean = 35 theorems, five sections.
MILESTONE: with rounds 3-4, the master formula's PER-TERM CORE is now fully
kernel-checked - every SAME, PAIRSPLIT, and CORR term is a proven class + floor
count. Remaining formal gap is ASSEMBLY only: (i) n-ary inclusion-exclusion over the
incidence classes (corr_triple_signed is n = 2; mathlib's Finset inclusion-exclusion
machinery applies), (ii) assembled sum = census overcount (Lateral verified exact at
every prefix, two scales). No new number theory in the gap - one formalist-scale
round if full CORR is wanted. Shelf unchanged otherwise: F(2,53) completion (data
note), conditional fragile-law derivation (low value).

## Lateral round 8 (2026-08-18) - THE HORIZON THEOREM: saturated runs <= 32, forever
Tool: research/word_grammar.py. Full census + proofs: docs/proof-search/lateral.md round 8.
- HORIZON THEOREM (unconditional, 2 lines): gear pair (5,7)'s split classes sit
  at k = 1 and 34 mod 35 (both members composite there); max cyclic gap 33; so
  ANY 33 consecutive slots contain a both-composite slot. Saturated runs - and
  even runs of consecutive slots each carrying >= 1 prime - are capped at 32,
  at every scale, forever. The Mechanic's L=14 hunt is sanctioned (14 <= 32,
  579 admissible words at L=14) and the record-growth law now has an
  UNCONDITIONAL CAP (exact CRT event, the framing directive's standard):
  max saturated run is in [13, 32] for all time.
- ESCALATION: adding gears does NOT lower the horizon through gear 23
  (periods to 37.2M checked): L0 = 32 for all sets {5,7}..{5..23}. The (5,7)
  corridor survives; its mouth is k = 2 mod 35, and the L*=13 landmark slot
  2452 = 2 mod 35 sits exactly there - at gears <= 17/19 the extremal corridor
  IS slot 2452. lim L0 over all gears: Jacobsthal-type, finitely checkable,
  monotone, >= any realized run.
- WORD GRAMMAR (language of admissible side-words, gears <= 13): w admissible
  <=> no gear's chosen residues cover Z_q (per-gear, CRT-free choice of phase).
  Census exact: 2^L through L=4; first exclusions at L=5 (same-letter blocks
  cap at 4, gear 5); strict-alternation caps are special cases; growth ratio
  falls to ~1.0 by L=18 and the language PLATEAUS (~1100-2600 words) to L=32,
  then EMPTY at 33. Total language FINITE - the saturated-run analogue of the
  gap-word antidictionary is the opposite object (finite tree with a wall vs
  infinite antidictionary).
- OBSERVED vs LANGUAGE: all 757 runs (csv, members to 7.2e10, Miller-Rabin
  recheck) admissible, 0 failures. Reality already realizes 84.7% of the
  entire L=10 language (199/235). The six L=13 words are six DISTINCT words at
  six different residues mod 35 - no CRT duplication, each its own corridor.
- COROLLARY (unconditional load ceiling for L > 32): on any twin-free window
  P_run <= L - minB(L): ceiling 0.970 (L=33) -> 0.873 (L=252), asymptote
  1 - 730/5005 = 0.854. Round 6's X-ceiling of 1 is provably unreachable past
  the horizon. (Not a contradiction engine: X's demand ~0.33 < 0.854.)
- FOR FORMALIST: the horizon theorem is a 3-line kernel target ("every 33
  consecutive slots contain k = 1 or 34 mod 35; members divisible by 5, 7");
  cheapest unconditional theorem the programme has produced.
- Offering next: Jacobsthal push (does the 32-corridor survive gears <= 100?);
  or language handoff (the 2560 corridor words as the exact X-local-behaviour
  catalogue at max length).

## Formalist round 9 (2026-08-18)
THE 32-CAP IS KERNEL-CHECKED, and the twin-product pin is unified across files:
proofs/Corridor.lean (namespace `Corridor`, tenth target, zero sorry; imports
Placement + Polignac read-only). Notably axiom-light: everything except one census
corollary needs only [propext, Quot.sound] - no Classical.choice in the cap or pin.
- `Corridor.exists_class_in_run`: every 33 consecutive slots contain k ≡ 1 or 34
  (mod 35) - the class-gap lemma, witness construction + omega.
- `Corridor.both_composite_of_class`: on those classes (k ≥ 2), BOTH members are
  composite (5 | lo, 7 | hi at k≡1; mirrored at k≡34). Slot 1 = the twin (5,7) itself
  is the unique exception, excluded by the guard.
- `Corridor.both_composite_in_run` / `double_slot_in_run`: every 33-window from slot 2
  holds a both-composite (= slotComps 2) slot, unconditionally.
- `Corridor.prime_adjacent_run_le` (THE CAP): any run of slots each carrying a prime
  member, starting at slot ≥ 2, has length ≤ 32. Lateral's three-line shape confirmed:
  class-gap + proper-divisor + assembly, all omega-grade.
- `Corridor.product_slotOf` (+ _sq): Placement.slotOf (p*(p+2)) = u*(p+1) = 6u² for
  6u = p+1. `Corridor.twin_product_pin`: the product is the LOWER member of that slot
  and both gears divide it - Polignac.twin_product_slot re-exported through
  Census.lo/Placement.slotOf; the two files' objects are now interchangeable.
Next target (proposed): packing corollary - n2 ≥ floor(W/33) over any W-slot range by
disjoint 33-windows (pure Finset counting), the formal doubles floor for the demand
side; or the tolerance-lemma arithmetic shells once Constructor fixes statement shape.

## Constructor round 9 (2026-08-18) - corridors vs lemma 1: two new laws, no alpha1
Tool: research/topgap_endpoint_law.py (filename note: my suite briefly held
research/topgap_corridor.py; Lateral's round-9 neighbourhood analysis owns that
name now - no content lost, both exist). Full text: constructor.md sec 20.
Full k-frame periods gears<=11..23; F2 values independently reproduce corpus
F2(2,y) = 33/48/75/93/117 exactly.
- TWO NEW CORRIDOR LAWS (proven, one line each; verified at every recorded gap
  in five full periods; kernel-checkable, six_mul_class-shaped):
  ENDPOINT LAW - a gap of length G has left endpoint residue in
  A(G) = {r in E: r+G in E} mod 35 (|A| 3..15; G = 34 forces {3,18,33}).
  Measured concentration EXCEEDS forcing: gears<=19 records (F=25, 9 allowed)
  all sit at the single residue 5; gears<=23 records at {3,33} of {3,18,33}.
  ADJACENCY LAW - adjacent (G1,G2) force a, a+G1, a+G1+G2 into E: A3 empty for
  294/1225 length-pairs mod 35 (e.g. (1,1)) - forbidden adjacencies from gears
  5,7 alone. All observed F2 pairs sit in allowed sets (some of size 2).
- THE NEGATIVE (decisive): escape distance = 1 - every (G1,G2) is within L1
  distance 1 of an allowed pair, so residue laws constrain WHERE, never HOW
  BIG; same argument at any bounded modulus. No alpha1 from corridors alone.
- QUANTITATIVE EXTENSION dies on Wall I: local two-scale capacity
  (rho*S - 1 <= sum 2*ceil(S/q)) gives real caps - F2_k(11) <= 12 (actual 11,
  TIGHT), F2_k(13) <= 54, F2_k(23) <= 72 with base {5..17} - but the margin
  dies 2-3 gears above any base (vacuous at 17 resp. 31). Corpus-5.2 locally.
- MEASURED TRUTH of lemma 1: record gaps 4-20 per period (mirror-paired), min
  separation 0.45-2.29% OF THE PRIMORIAL PERIOD (851,695 slots at gears<=23) -
  near-max anti-clustering is astronomical; F2 pair at 23 is (F,5) = the max
  gap's own flank; adjacent (F2-F)/q_next = 0.92/0.88/1.10/0.78/0.52, corpus
  max 1.16 - bouncing under 1.2, no growth.
- VERDICT: corridors do NOT give alpha1; they give exact residue geometry that
  LOCALISES any violation (A3-allowed configs at forced residues) and a
  concrete pruning rule for the F(2,53) search (restrict record-endpoint
  residues by A(G mod 35), factor 2-5x, mod 105 adjacent-frame) - offered to
  Harvester. Lemma 1 needs genuine extreme-value input: Wall V stands.
- For Formalist: endpoint + adjacency laws are cheap kernel targets on the
  existing six_mul_class/card_class_Ico machinery.

## Harvester round 6 (2026-08-18) - the ASSEMBLY kernel-checked: n=3 inclusion-exclusion + bridges
Coordinator-approved bite executed; scope taken UPWARD of the fallback: the n = 3
assembly proven GENERALLY (any gears, any range - the concrete {5,7,11} window is an
instance), plus both bridges to the class-count layer. Full detail: harvester.md sec 10.
Computational verification FIRST (research/assembly_check.py: 4 gear triples x 6
window lengths - assembly identity, per-gear side splits, pair bridge with floor
counts, AND the full pipeline overcount = sum(12 pair classes) - sum(8 triple
classes), every term equal to its floor formula: zero fails).
New theorems in proofs/Polignac.lean (built clean on first compile):
- `three_sets_ie`: n = 3 inclusion-exclusion, subtraction-free, arbitrary finsets
  (|AuBuC| + three pair inters = |A|+|B|+|C| + triple). `three_preds_ie`: filter form.
- `three_gear_assembly` ("ASSEMBLED SUM = SIEVE OVERCOUNT"): the identity at the mark
  sets M_q = {k : q | 6k-1 or 6k+1}; set-level, no primality hypotheses; overcount =
  marks - distinct is a rearrangement. Extends corr_triple_signed (n=2) to the full
  3-gear window.
- `card_marks_eq` (per-gear bridge): |M_q| = left class + right class, disjoint by
  slot cap (mark_side_unique from slot_cap_twin).
- `card_pair_inter_eq` (pair bridge): |M_q ^ M_r| = LL + LR + RL + RR, four disjoint
  side classes, each ONE CRT class with floor count - the set-level assembly meets
  the class layer here. Plus reusable `card_filter_or_of_excl`.
STATUS: ledger GREEN - `lake build` 992 jobs, zero sorry; axiom audit standard on all
seven. Polignac.lean = 42 theorems, six sections.
REMAINING for full closed-form CORR (priced, paper-side): (i) triple 8-way side
decomposition - mechanically identical to the pair bridge (2^3 cases, same
discharges); (ii) n > 3 - iterate three_sets_ie or mathlib inclusion-exclusion. No
new number theory in the gap; verified numerically end-to-end this round.
F(2,53) WATCH (coordinator's ask): research/data/maxgap53.log contains only the
header line at round end (no increments logged yet) - nothing to fold in. Tolerance
needs <= 486; standing bound >= 420.

## Lateral round 9 (2026-08-18) - top-gap corridor analysis: addresses + skeleton, no cap yet
Tools: research/topgap_corridor.py (full periods to y=23, streamed 1.08e9 for
y=29), research/topgap_nesting.py. Log: docs/proof-search/lateral.md round 9.
Frame: slot units; corpus halved = 3 x slot.
- EXACT top-of-spectrum structure (machines y = 13..29, full periods):
  (a) maximal-gap intervals are mirror-closed at every machine (slot 0 is a
      universal opening, so gaps never straddle 0; maxima come in mirror pairs
      with mirrored merge words, e.g. (4,8,15,7)/(7,15,8,4));
  (b) ADDRESS PINNING: maximal gaps sit in 1-2 endpoint classes mod 35 (y=19:
      all 20 at left=5/right=30) and 2-6 of 135 classes mod 385 (~30x
      concentration); at y=23 and 29 the maximum is UNIQUE up to mirror. The
      pinned address drifts with the machine (gaps are machine-relative;
      saturated runs were absolute - that is why the landmark pin was global);
  (c) CHAIN SKELETON CONFIRMED AT THE MAXIMA: every new maximum is a merge of
      OLD MEDIUM gaps (0.16-0.68 F_old; k = 2-3 kills; two y=19 exceptions
      extend an old max by k=1) by a strictly side-ALTERNATING chain of the
      new gear with interior spacings EXACTLY {2u', q-2u'} (17: 6/11; 19: 13;
      23: 8/15; 29: 10) - the chain condition reconfirmed at the extreme tail.
- LANGUAGE VERDICT: near-top gap words are NOT a finite language (values grow;
  no 32-cap analogue at the top; antidictionary-style infinitude persists).
  The RELATIVE grammar is finite: flank alphabet {1..5} slots at every scale
  (isolation law quantified), rigid chain spacings, near-top neighbourhood
  word counts non-growing (14-42 per machine, y=13..29).
- ALPHA1 EVIDENCE: (F2-F)*3/q_next = 0.88, 1.11, 0.78, 0.52, 1.16 for
  y = 13..29 - all below the constructor's 1.24, no trend. F2 has TWO regimes:
  F+small-flank (13/17/23; there F2-F = 3*flank <= 15 halved - a flank cap
  would give alpha1 < 1) and medium+medium (19: 21+10; 29: 30+25 - the regime
  that controls alpha1 asymptotically). HONEST: no corridor cap on F2-F found
  this round; the method delivered addresses + skeleton, not the bound.
- CONCRETE NEXT TARGET (feeds alpha1): the medium-medium ADJACENCY question -
  near-top gaps live in 2-6 address classes mod 385; can two such classes ever
  be adjacent (one opening apart)? If not, F2 comes from lower strata and
  alpha1 follows per machine by finite check. Constructor: the mirror-pairing
  and skeleton facts are ready-made constraints for the merge transform.

## Lateral round 10 (2026-08-18) - UNIFORMITY: the word pins the address (<= 4, uniform); drift recursion refuted
Tool: research/address_drift.py. Full log: docs/proof-search/lateral.md round 10.
- LAW A (word-pinning) ESTABLISHED, measured on all 206 near-top words (0.9F
  strata, y = 13..29, full periods): the neighbourhood word determines the
  address mod 385 almost uniquely. Gear 5 pinned to EXACTLY ONE offset by
  every word (206/206); gear 7 unique 94% (max 2); gear 11 unique 90% (max 4);
  full mod-385 address UNIQUE for 87% of words, <= 4 ALWAYS, at every machine.
  Containment exact (0 fails); tightness 0.71-0.85. Mechanism: each opening
  forbids 2 offsets/gear (exposure counting) - proof-shaped, computable per
  word. Consequence: #top-stratum classes <= 4 x #near-top words; observed
  class counts sit even lower and FLAT (6-14 from y=13 to 29 while gap counts
  swing 20-106) because distinct words share pinned addresses.
- LAW B (drift recursion "new address = old stratum address - flank") REFUTED:
  reachability from the old 0.9F stratum runs 18/20, 14/20, 0/4 (19->23), 1/2
  (23->29). New maxima grow from deep-medium gaps (0.16-0.68 F_old) that no
  near-top stratum tracks; the early mod-385 near-matches (47-2=45, 122-5=117)
  were the flank regime, not a law. The address is LOCAL (= pin(word)), not
  inherited - per-machine finite checks do NOT chain into an address
  induction; they localize instead.
- FOR CONSTRUCTOR (adjacency chunk): the adjacency question converts to a
  WORD-LEVEL check - two near-top words can be adjacent only if their pinned
  phase sets are CRT-consistent with the separation; finite per word pair, no
  period scan. Word lists per machine are in address_drift.py's groups.
- Machine-independent alpha1 statement now = [per-word pinning <= 4, uniform -
  ESTABLISHED] + [uniformity of the near-top word grammar - OPEN; counts
  non-growing 20-106 but words are machine-relative].
- Offering next: the extreme-value grammar (do flank alphabet {1..5} + chain
  skeleton {2u',q-2u'} + pinning already delimit a finite a-priori word-shape
  family?); or direct support of Constructor's adjacency run via the word-pair
  CRT check.

## Harvester round 7 (2026-08-18) - assembly line CLOSED: triple bridge + 26-term master theorem
Coordinator's round-10 brief executed in full. Detail: harvester.md sec 11.
Computational verification FIRST (research/master3_check.py: 5 gear triples x 5
window lengths to t = 5005, triple 8-way bridge + 26-term identity: zero fails).
New theorems in proofs/Polignac.lean (built clean on first compile):
- `card_triple_inter_eq` (TRIPLE BRIDGE): |M_q ^ M_r ^ M_s| = the 8 disjoint side
  classes LLL..RRR, each one CRT class with its floor count. Same mechanics as the
  pair bridge (2^3 flatten + 7 exclusivity peels via mark_side_unique).
- `three_gear_master` (END-TO-END, 26 terms, subtraction-free): distinct + 12 pair
  side classes = 6 single side classes + 8 triple side classes, any distinct odd
  primes, any prefix. Overcount = marks - distinct rearranges to overcount = pairs
  - triples; every term one CRT class with closed floor count. The formal statement
  of assembly_check.py's zero-fail verification. THE 3-GEAR ASSEMBLY LINE IS CLOSED
  END TO END. n > 3: assessed, not forced (iterated flattens or mathlib signed I-E
  over Z; nothing conceptually new; deferred until needed).
STATUS: Polignac green first compile, both theorems standard axioms. LEDGER NOTE for
manager: plain `lake build` fails in Corridor.lean (another workstream's file,
mid-edit, 3 errors its own file); the other NINE targets incl. Placement build green
together (988 jobs). Polignac.lean = 44 theorems.
F(2,53) WATCH + PRUNING ASSESSMENT (per brief, analysis only): TWO maxgap.exe
processes running; maxgap53.log still header-only at round end => likely no
increment completed since launch (>= hours per increment at L >= 420; header
flushed, so buffering is not the issue - manager: check state-file mtime and
whether two processes = parallel split or accidental duplicate). Economics: the
endpoint law's 2-5x cut applies to every remaining increment INCLUDING the final
uncoverable certificate; search supports resume => pruned restart loses no work;
break-even ~one increment at current L; remaining distance plausibly 20+ increments
(quadratic law ~441 vs standing >= 420). VERDICT: pruned restart beats continuing
unpruned unless within ~1 increment of termination (no evidence for that).
Recommend: verify process state, implement endpoint-law filter in rust/maxgap,
restart from resume state.

## Formalist round 10 (2026-08-18)
ENDPOINT LAW + ADJACENCY LAW + PACKING FLOOR kernel-checked (Corridor.lean extended,
10 targets, zero sorry, standard axioms only). Everything cross-verified against
research/topgap_endpoint_law.py BEFORE formalizing (E-set, A(34), 294, first examples).
- `Corridor.Exposed k` (neither 5 nor 7 divides either member) + `exposedSet` (the 15
  residues) + `exposed_iff_mem`: for k ≥ 1, Exposed k <-> k % 35 ∈ E. 
- `Corridor.endpoint_law`: both-exposed endpoints put a % 35 in A(G) = filter of E;
  `endpoint_law_34`: G ≡ 34 mod 35 forces a % 35 ∈ {3, 18, 33} - Constructor 20.1
  exactly.
- `Corridor.allowed3` (A3) + `adjacency_law` + `no_chain_of_forbidden`: forbidden
  length-pairs can never be realised by three chained openings.
- `Corridor.forbidden_pairs_count`: EXACTLY 294 of 1225 pairs forbidden - the full
  35x35 table evaluated by the PROOF KERNEL (decide +kernel, no native_decide, no
  ofReduceBool axiom; 22s). `forbidden_first_examples`: (1,1),(1,3),(1,6).
- `Corridor.n2_packing`: W/33 ≤ n2(Ico a (a+W)) for a ≥ 2 - the unconditional doubles
  floor from gears 5,7, composable with Census (uses choice via `choose`; a Nat.find
  variant can make it choice-free if ever needed).
Formalisation note for Constructor: omega cannot handle the 5-atom joint residue
statement directly - the working shape is per-gear iffs + interval_cases over k % 35.
Anyone formalizing mod-105/385 corridor laws should expect the same split.
Next target (proposed): demand-side assembly - X + census pinning + packing into one
statement (the prime-density floor X forces per prefix); or corridor transfer mod 385.

## Constructor round 10 (2026-08-18) - top-stratum adjacency NO; lemma 2 unified with lemma 1
Tools: research/strata_adjacency.py, research/merge_census.py (23->29 streamed,
P=1.078e9). Full text: constructor.md secs 21-22. Anchors: F_k(M+q') =
11/18/25/34/43 reproduced; the 62 k=3 chains at 19->23 match the corpus fuel
census exactly.
- LATERAL'S LIVE TARGET ANSWERED: at y = 13/17/19/23 the top stratum occupies
  4-6 classes mod 385 and the class-level test (r and r+F both top-stratum)
  is EMPTY at every machine - two maximal gaps can NEVER be adjacent,
  certified by class arithmetic + one period scan.
- ALPHA1 FINITE CHECK CLOSES at all four machines (three tiers): y=13 at
  alpha1=1 written out fully (14 dangerous pairs: 5 die by machine-free A3,
  5 by mod-385 class disjointness, 4 by direct check - none realized);
  17/19/23 at alpha1=4/3. Honest trend: tier-C residual grows (4 -> 96 at 23)
  - the class tier needs mod-5005 at scale; uniformity-in-y still open.
- LEMMA 2 CENSUSED: excess(M,q') = F(M+q') - F2(M); full chain census at five
  steps: k_max <= 3 everywhere (k-hist e.g. 19->23: {1: 733672, 2: 11746,
  3: 62}); argmax anatomy at every positive-excess step: interior gaps
  LITERAL {2u', q'-2u'}, residues on the teeth, g_L+g_R <= F2; identity
  excess = interior_sum - (F2 - g_L - g_R) verified.
- SPECTRUM REDUCTION (structural result): define F_j(M) = max sum of j
  consecutive gaps (F_1 = F, F_2 = F2). RIGOROUS: F(M+q') <= F_{k_max+1}(M),
  excess <= F_{k_max+1} - F2. Measured spectra (machines 11..23, j<=6):
  increments are q/3-SCALE not F-scale (2..12) - "best windows cannot extend
  by a large gap", the isolation law generalised to all depths. Bound
  tight-ish: F_{k+1}-F2 = 5/7/3/7/11 vs excess 0/2/0/3/4.
- THE UNIFICATION: lemma 1 IS the first spectrum increment (F2-F = F_2-F_1).
  The whole tolerance hypothesis = ONE statement: spectrum O(q')-flat to
  depth k_max+1, plus k_max = o(ln y) (fuel; tolerance slack absorbs slow
  growth). Measured: increment sums 9-19 vs budgets 2.5q' = 32-72.
- CORRIDOR VERDICT for lemma 2 (mandate's question): merges ARE local - the
  fuel half (k_max) is a bounded-window censusable object, genuinely more
  approachable (forbidden-configurations native). The flatness half inherits
  lemma 1's escape-distance obstruction verbatim (size statements) - Wall V.
  Net: two lemmas -> one flatness statement + one fuel bound.
- FOR MECHANIC: the F_j consecutive-sum spectrum is the object to census at
  scale (j <= 6 rolling max - cheap on existing gap streams).
