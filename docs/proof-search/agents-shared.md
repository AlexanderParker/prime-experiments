# agents-shared.md - findings exchange for the proof-search team

## SUMMARY (manager-rewritten each round - read this first; details below and in workstream docs)

State after round 3. The landscape is fully mapped, with honest prices on every route: (i) local
attack dead (round 2); (ii) the cumulative statement is EXACTLY equivalent to Reduction A -
lossless reparametrization, gears drop out, diagnostic value only; (iii) the layer-band descent
dies at T1, "a prime in every band" - Legendre-class, technology (Alweiss-Luo) stops at exponent
0.525 vs the needed 0.5, before bounded gaps (T2) or parity (T3) even engage. Thinnest bands occur
exactly at twin endpoints - the self-reference sits at the binding case.

The genuinely live structure is the QUANTIFIED SELF-REFERENCE: the window's double supply is a
freedom-free functional of prime gaps below y (gap-graded split law, closed form, verified on all
2850 pairs to 400; overcount formula exact at three scales), and GAP 2 IS THE UNIQUE GAP CLASS
whose contribution is unconditionally guaranteed at every scale (pins at u' <= y/6, bottom band).
Meanwhile the margin census is gear-blind and rigid: M(t) follows t - li(6t+m0) + li(m0) to 0.1%,
min margin is 0/-1 at t <= 3 with NO later dip through 6.67e9-slot windows (y to 200003), and
layer bands are invisible to it at 1e-4 - band structure can only enter via per-gear attribution.

ROUND-4 FLAGSHIP (constructor + lateral converge): the X-CONSISTENCY EQUATION. Under X the doubles
demand is pinned exactly (n2(t) = N(t) - P(t), zero slack); the doubles supply is freedom-free
semiprime/split arithmetic - an explicit functional of primes and gaps below y. Equate them: X
imposes an exact equation between the prime census of (y, y^2) and the gap structure below y.
Write it, and determine whether ANY admissible configuration can satisfy it - overdetermination is
the contradiction candidate. Mechanic feeds the attribution side (per-gear R_q(t) trajectories);
formalist pins the zero-slack census (substrate of the demand side).

Kernel-checked so far (5 files, all standard axioms): reduction (iff), horizon (strict p < y),
slot-cap, layer novelty (strongest form), supply identity + distinct-roots corollary (Supply.lean,
first composing file).
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
