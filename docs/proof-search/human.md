# human.md - proof-search snapshot

## ELI5 SUMMARY (rewritten each round)

Round 5 produced the first genuinely new WEAPON of the search: the inversion zone. Using nothing
but counting arithmetic (no prime facts at all), there is a criterion - a ratio computed from the
machine's fixed schedule - which, whenever it exceeds 1, FORCES a twin to exist in that stretch.
We used it to prove the twin (521,523) exists from four slots of pure arithmetic, no searching.
The catch, stated honestly: the zone where the criterion bites shrinks as numbers grow (its
strength fell from 19.6 to 1.44 across our test range), and proving it stays above 1 forever runs
into the same deep wall as everything else. But it is the first unconditional twin-forcing tool
the machinery has produced, and next round we track exactly where its power goes.

Everything else sharpened the picture: the difference between reality and a twin-free world is
now known to live in a single "zero" statistic - literally the twin count itself, nowhere else
(the fancier statistics all match between the two worlds). The books identity is verified
slot-by-slot out to 16.7 million slots. And the verified-proof ledger grew to eight files,
including this round the "bridge" (the equation's skeleton, fully machine-checked) and a small
gem: the UNIQUENESS theorem that only twin pairs stamp their own address - the mark of gap-2 that
no other gap has, now at the highest standard of certainty.

One warning from the terrain scan: the region where a proof must bite (the bottom of each window)
is exactly where reality is strongest - so the eventual argument must use the machine's special
structure (its mirror symmetry, its pinned slots), not generic tools. That is where round 6 digs.

## Round 1 (2026-08-18)

**Formalist** - proofs/Horizon.lean kernel-checked, zero sorries, standard axioms: the horizon
theorem (strict p < y), its contrapositive, and the twin-pair corollary. Lake lib registered;
BlockedSlots untouched.

**Mechanic** - fragile census exact to y = 50021 (members 2.5e9). Fragile/twins grows ~ lnln
(Mertens): 1.11 -> 5.06. Bottom-decile gears own 88% of fragile load at scale. Candidate
zero-parameter identity: fragile*pi_win/(twins*W1) -> 2 (measured 1.95-1.99, 0.43% at top end) -
the fragile census is the lone-composite population times the same partner-prime structure that
makes twins. Tool: research/fragile_census.py.

**Lateral** - new closed form: a twin gear-pair (p, p+2) marks the higher window at exactly two
deterministic slots (own slot + product slot 6u'(p+1)-1 = p(p+2)); coincidence classes mod p(p+2)
are {+-u', +-u'(p+1)}. Refuted: tooth-sharing cannot close the recursion by counting (O(T(y)) vs
K/log^2; stride dead flat under sharing). Anomaly found: the machine's real phase vector is wildly
non-generic in overcount (z = +6.1) and fragile count (z = -5.9) yet stride-generic. Tool:
research/tooth_sharing.py.

**Constructor** - Condition X (a twin-free window) formalized with an exact ledger: kills per slot
are 0/1/2 from distinct gears (slot-cap lemma); lpf attribution makes supply exactly 2N - P;
zero-slack theorem: X forces the full prefix census (fragile = primes count, doubles = the rest,
no waste). Sharpest condition C2: under X the doubles must outpace in-window primes from the first
slot. Real windows: X is impossible OUTRIGHT below y ~ 403 (too few composites exist - P - N > 0);
at y = 47 the global ledger passes but C2 fails in the bottom band by +7. The violation zone is
always the window's bottom band. Tool: research/constructor_ledger.py.

**Manager synthesis.** The constructor's descent idea (X at y forces a half-window stride event at
scale sqrt(2y)) re-derives Reduction A at constant 1/2 rather than reducing its difficulty - noted
honestly. The genuinely new leverage: (a) X-impossible-below-403 is a finite counting theorem
candidate (formalizable); (b) C2 localizes any hypothetical X-failure to the bottom band, where
the layer law and horizon machinery are strongest; (c) the mechanic's constant-2 identity and the
lateral's phase-anomaly both say the window censuses carry structure beyond density - possible
variational handle.

**Round 2 tasks queued**: Mechanic - per-gear fragile closed form incl. the top-gear rare-event
tail. Constructor - make X-impossible-below-403 rigorous + the bottom-band double-onset law from
deletion spacing. Formalist - the layer law's arithmetic core in Lean. Lateral - test whether the
real phase vector is extremal for any window observable.

## Round 2 (2026-08-18)

**Constructor** - doubles proven freedom-free (36k^2 = 1 mod qq' iff); unconditional onset cap
L0 <= 27129; DECISIVE: onset-prefix refutation closed (310/442 real windows realize X's forced
alternation); descent bottomed out exactly - needs "one layer band always holds a twin"
(bounded-gap strength), band/stride slack 2.2 -> 231. Tool: research/double_onset.py.

**Mechanic** - per-gear fragile law exact after 1/ln(m) weighting (2e-4, Poisson-clean everywhere
incl. top-1% tail); prefix censuses across 150 windows: first double at slot 2.4-3.7 (y-free),
margin >= 0 from t=5 in 125/125; identity: prefix-pigeonhole refutations are nonconstructive twin
proofs whose reach ends by slot ~4. Tools: research/fragile_pergear.py, prefix_census.py + CSV.

**Formalist** - Layer.lean kernel-checked (970 jobs): slot_cap; layer novelty in strongest form
(fresh composite = y*c, c prime, no Bertrand, composable with survivor_step). Standard axioms.

**Lateral** - overcount anomaly closed as a theorem (real = exact divisor census: 190 semiprime +
145 Bezout split; random side closed-form; lone deficit same accounting); extremality REFUTED by
full enumeration (rank 1716/11550). Tool: research/overcount_census.py.

**Manager synthesis** - onset route closed by convergence of constructor + mechanic; frontier is
now (a) the cumulative margin statement and (b) the layer-band descent, precisely one notch above
known bounded-gap theorems. Round 3: mechanic measures full-window margin trajectories; constructor
formulates the cumulative statement + scopes layer-band vs Maynard-Tao exactly; formalist does the
supply identity; lateral derives the gap-graded Bezout split law (sqrt-scale gaps -> ledger).

## Round 3 (2026-08-18)

**Lateral** - gap-graded split law in closed form, verified on all 2850 prime pairs to 400; the
complete overcount formula exact at three scales; gap 2 = the unique unconditionally guaranteed
doubles supplier at every scale (pins in the bottom band). Self-reference quantified. Tool:
research/split_gap_law.py.

**Constructor** - CUM proved exactly equivalent to Reduction A (lossless; diagnostic only).
Layer-band failure tower priced: T1 prime-in-every-band (Legendre-class; localisation technology
stops at exponent 0.525 vs needed 0.5) -> T2 bounded-gap localisation -> T3 parity. Thinnest bands
occur at twin endpoints. Full-window excess E(y) flat at 3, realized by clusters just above y.
Tool: research/cumulative_margin.py.

**Formalist** - Supply.lean kernel-checked (974 jobs): the supply identity as a Finset partition,
ledger form, and the distinct-roots slot corollary; first composing file (imports Horizon+Layer).
Five bricks total, all standard axioms.

**Mechanic** - full windows to y=200003 (6.67e9 slots, 186s): min margin is 0/-1 at t<=3 with no
later dip anywhere; M(t) = t - li(6t+m0) + li(m0) to 0.1%; danger zone is member-anchored O(1)
(crossover at e^6 ~ 403); layer bands invisible to the census at 1e-4 - attribution objects
required. Tool: research/margin_trajectory.py + CSVs.

**Manager synthesis** - all cheap routes now priced: local dead, cumulative = the conjecture,
layer-band = Legendre-class. Live: the quantified self-reference. Round 4 flagship = the
X-consistency equation (demand side pinned by zero-slack, supply side pinned by the freedom-free
gap functional); mechanic builds per-gear R_q(t); formalist pins zero-slack census in Lean.

## Round 4 (2026-08-18)

**Constructor** - X-consistency equation written and tested: X <=> P(t) = t - D(t) (the
unconditional floor) at every prefix; SATISFIABLE - no overdetermination; the floor sits at
exactly half the Montgomery-Vaughan ceiling (parity factor live); g=2 pins the unique guaranteed
budget line (5-9%). Tool: research/x_consistency.py.

**Mechanic** - R_q(t) trajectories (verified vs independent spf count, one bug self-caught):
capacity NEVER the contradiction (slack 3.2-4.5x, loosening); the reality-to-X distance is
compression: multiplicity 4.38 vs demanded 4.50 (2.6%). Tool: research/supply_trajectory.py.

**Lateral** - master formula: overcount(t) as one signed sum, exact at every prefix (y=101, 211);
n2(t) = B(t) - U(t) exactly (U = finite u'-pin list, bottom band); availability schedule matches
the measured onset to the slot. Flagship now two-sided: under X, N-P = B-U at every t. Tool:
research/supply_formula.py.

**Formalist** - Census.lean kernel-checked: census identities, n0_eq_zero_iff (= Condition X),
census_pinned + prefix form - the demand leg formal. Six libs then, seven with Polignac.

**Harvester (round 1)** - survey: band statements unreachable (0.525 wall), constant-2 HL-class;
bite: Polignac.lean kernel-checked - per-gap transfer (slot_cap_gap: q | both members iff q | d),
per-d equivalence Polignac_d <=> windowed survivor, Goldbach window reduction; sharpens
Ziller-Morack to machine-checked per-difference equivalences.

**Manager synthesis** - the equation is satisfiable; the contradiction, if reachable, is a
second-moment/compression statement (4.38 vs 4.50) - the corpus's kappa frontier reached
independently with exact arithmetic. Parity visible as the exact 1/2. Round 5 = the compression
frontier from all five sides.

## Round 5 (2026-08-18)

**Constructor** - compression bound stated exactly; tool inventory computed on the real system:
union/Bonferroni vacuous, Cauchy-Schwarz ceiling DIVERGES from the need (2x hope refuted), Selberg
wrong direction. NEW: the inversion zone - R(t) = (S1^2/M2)/(t-P) > 1 forces a twin by moments
alone; nonempty at every y tested; proves (521,523) from floor arithmetic. Edge identified:
mirror-aware third moments on the starved bottom band. Tools: compression_bound.py, compression_zone.py.

**Mechanic** - multiplicity distribution vs two exact nulls: the X-gap is zeroth-moment ONLY; the
joint zero-mass IS the twin mass (0.77-0.85 of product baseline); exact slot-cap covariance
constant; variance/tail = product structure, not X. Tool: multiplicity_census.py.

**Lateral** - master formula to y=10007 (three orders further); reality identity per-slot exact at
16.7M slots: P = t + T_win - B + U (the X-defect is exactly the twin count); derivative scan:
bottom band stride-hostile, top-1% strides carry 87-90% of prime load. Tool: derivative_scan.py.

**Formalist** - Bridge.lean: sum R_p = n1 + 2*n2 kernel-checked (the equation's formal skeleton
complete); caught the mid-round Polignac breakage and isolated it correctly.

**Harvester** - the g=2 pinning theorem kernel-checked (7 theorems): twins sit at 5 mod 6, ARE
their own pin slot, class-iff, mirror, product slot, and UNIQUENESS (only g=2 pins its own slot);
ledger repaired and green (986 jobs, verified independently by manager).

**Manager synthesis** - first unconditional twin-forcing criterion in hand; its domain shrinks
toward the parity wall; round 6 tracks sup R(y) exactly, opens the mirror-aware third-moment
front, maps the load-length frontier, and extends the Lean ledger per-gear.
