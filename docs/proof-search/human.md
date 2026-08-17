# human.md - proof-search snapshot

## ELI5 SUMMARY (rewritten each round)

We assume a twin-free stretch (condition X) and hunt the contradiction. Round 1 showed X forces
the machine into a zero-slack pattern and pointed at the window's bottom band. Round 2 delivered
the verdict on that idea, honestly: the LOCAL attack is dead. Real windows actually behave, near
their start, exactly the way X would force them to - so no local argument can ever refute X. And
"double-composite" slots (the thing X needs plenty of) arrive almost immediately in every real
window, so there is no forced starvation window to exploit. What survives is bigger-picture: the
contradiction must come from bookkeeping over long stretches (the running margin between
composites and primes), or from a beautiful reduction found this round - X at height y would
force EVERY "layer band" above y to be twin-free, so it would suffice to prove that some single
layer band always contains a twin. That statement is close in spirit to the famous bounded-gaps
theorems (Zhang/Maynard), which prove twins-like pairs recur SOMEWHERE forever but not in every
band - the gap between "somewhere" and "in every band" is now our precise frontier.

Meanwhile the machinery got sharper: the "double" slots turn out to be completely determined by
simple arithmetic (no freedom at all - they sit where 36k^2 = 1 modulo a product of two gears);
the strange specialness of the machine's phases was fully explained and turned into a formula
(and the hope that the machine is "extremal" was cleanly disproven by brute-force enumeration);
the fake-twin census now has an exact per-gear law. Four bricks are now machine-verified in Lean:
the reduction, the horizon theorem, the slot-cap lemma, and the layer-novelty theorem.

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
