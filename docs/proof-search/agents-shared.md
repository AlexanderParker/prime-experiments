# agents-shared.md - findings exchange for the proof-search team

## SUMMARY (manager-rewritten each round - read this first; details below and in workstream docs)

State after round 6: THE MOMENT PROGRAM IS CLOSED, by convergent independent verdicts. The
inversion zone is a bottom-twin DETECTOR, never a generator (it revives exactly when a twin sits
in a window's first slots - "zone revives infinitely often" IS the conjecture); it dies
generically at y ~ 3-5e6 (empty at 5,000,011 and 10,000,019; (sup-1) ~ y^-0.6; killed by the
twin-surplus/prime-density side, not M2). Mirror-awareness is VACUOUS at moment level, any order
(two-line theorem: k -> -k swaps omega_L/omega_R, fixes m_k). Depth is not a lever (twin-mass
decline is pure density falloff to 0.3%). LP/order-3 ceilings move the needle <3% against a 48%
chasm. The Constructor's count/moment toolkit is spent - by their own account.

WHAT THE ZONE LEAVES BEHIND (real, finite, kernel-checkable): where R(t) = (S1^2/M2)/(t-P) > 1,
moments force n0 >= 1 unconditionally - at y=2003, t*=24 the histogram forces n0 >= 6 (six real
twins from floor arithmetic). Valid for every y < ~3e6 and sporadically beyond.

STRUCTURAL RESIDUE (the live fronts): the load-length frontier is ABSOLUTE - record twin-free
runs are the same integer landmarks at every scale (perfect X-alternation realized to length 13
at slots 2452-2464); binding scale for any bound is L ~ 14-32; long-run bounds fight a phantom;
chain/fuel objects differ from binding-region objects below L ~ 160. Saturated-run persistence is
itself HL-constellation-class (flagged caveat).

LEAN LEDGER (green, 988 jobs, manager-verified): 9 files. New this round: Gear.lean (per-gear
ledger lines R_q, caps, prefix bound 6t/q+2, shadow law R_q = 0 below q^2) and Polignac.lean's
SAME-side census (slot-map inversion, floor-count primitive, the pair term of the master formula
exact, the windowed composite root law "once if it fits", own-value law) + twin_pin_self_block
(the machine is formally blind to its own pair) - first composition with Census.

ROUND-7 (the strategic turn): Constructor -> THE IMPOSSIBILITY MAP (definitive write-up of every
closed route with its exact reason - the programme's key prose artifact). Mechanic -> saturated-
run census by (length, depth) vs R(t). Lateral -> alternation-word structure vs the mirror laws +
scope the HL-constellation caveat honestly. Formalist -> semiprime refinement (R_q = #partner
primes in the thin layer - first exact supply formula formal). Harvester -> PAIRSPLIT closed-form
rep in Lean (completes the master formula's formal core).
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
- REFUTATIONS (be aware before spending effort here):
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
  Poisson-binomial via DFT over all 13.2M pairs) FAILS on P(mu=0) by 6.6x
  (0.041 vs 0.273 at y=50021). Real compression 4.38 vs null 3.32: the machine
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

## Mechanic round 6 (2026-08-18) - THE ZONE DIES AT y ~ 2-5 x 10^6
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
- CLOSURE (decisive, recorded): the onset route alone cannot kill X. The fact that
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

## Lateral round 2 (2026-08-18) - anomaly closed into identities; extremality refuted
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
  theorem at accessible heights. FAILURE TOWER, in order: T1 a prime in every band
  (OPEN - Legendre-class; not implied by RH; Cramer suffices) -> T2 bounded-gap pair
  in every band (exponent deficit 0.025) -> T3 gap exactly 2 (parity, 246 -> 2, no
  partial result). The descent input dies at T1 before its twin content engages.
- Net strategic read for the manager: both round-3 avenues terminate at named
  external walls (parity/superdensity; Legendre localisation). The constructor's
  exact-ledger toolkit is exhausted on count-shaped statements; remaining in-corpus
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
  progression form). The ledger and the analytic wall are one wall, two faces.
- Coordination: Mechanic's per-gear R_q(t) composes as the attribution-graded
  demand side when posted; Lateral's CORR formula-ization would upgrade the
  incidence counts (CORR overlap measured large: +806/+5162/+40960) to distinct-
  double counts in closed form - that is the remaining formula gap in D(t).

## Harvester round 1 (2026-08-18) - adjacent-statement survey + Polignac/Goldbach transfer
Full survey: docs/proof-search/harvester.md. Verdict-first: nothing in the corpus touches
Legendre-class band statements or any fixed-gap Polignac CONJECTURE (parity/localisation
walls, as priced in rounds 2-3); the harvestable layer is the REDUCTION FRAME and the
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
  MANAGER'S 2x EXPECTATION REFUTED - worse: C_CS/M_X = 1.26 -> 1.58 (y=211->5003)
  and GROWING (tracks lnln-divergent dispersion) while the needed window narrows
  (1.22 -> 1.05). Diverges, does not land at 2x. Large sieve/MV = same 2nd-moment
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
  R > 1 with sub-superdense prime input. Everything else terminates at the parity
  wall by measurement.

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
  long runs fight a phantom.
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

## Constructor round 6 (2026-08-18) - zone fate settled; third-moment front closed
Tool: research/zone_fate.py (ladder to y = 10^7; LP moment ceilings). Full text:
constructor.md sections 17-18. Mechanic round-5 CSVs consumed; their "X-gap is
zeroth-moment only" corroborated independently at orders 2-3.
- ZONE FATE: R = eff * boost (eff = CS efficiency, boost = 1 + n0/(t-P)). No
  single crossing: the zone dies GENERICALLY between y ~ 3e6 and 5e6 (sup R:
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
  bottom-twin detection = the conjecture. Constructor's count/moment toolkit is
  fully spent - recommend reassignment to structural fronts (or wind-down).

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
