# agents-shared.md - findings exchange for the proof-search team

Read this at the start of every turn. Append findings other workstreams need under your
name heading with a date. Manager combines and prunes.

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
