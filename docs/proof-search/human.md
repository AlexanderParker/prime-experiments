# human.md - proof-search snapshot

## ELI5 SUMMARY (rewritten each round)

We assume twins run out somewhere (condition X) and hunt the contradiction. Three rounds in, every
easy door has been checked and priced honestly: the local door is closed (real numbers behave
exactly as the forgery would, near the start); the "audit the whole book" door turns out to BE the
original problem in disguise (proved equivalent, nothing gained); and the elegant "every band has
a twin" shortcut turns out to need something as hard as a famous 100-year-old open problem
(Legendre's: a prime between consecutive squares) before twin-ness even enters.

What is genuinely alive is a loop the machine built itself: the "double-blocked" slots that X
desperately needs are supplied, with zero freedom, by the primes and prime-GAPS at the smaller
scale - and pairs with gap exactly 2 (twins!) are the only gap type whose supply is guaranteed at
every scale. So a twin-free stretch at scale N would have its books balanced by twin structure at
scale sqrt(N). Round 4's flagship: write that balance as one exact equation (demand pinned by X,
supply pinned by arithmetic) and see whether the equation can be satisfied at all. If it is
overdetermined - if no arrangement of primes can pay both sides - that is the contradiction.

Also: the margin bookkeeping is now measured to forty billion numbers per window with a formula
accurate to 0.1%, and five machinery bricks are computer-verified at the highest standard of
certainty (the fifth, the supply ledger, landed this round).

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
