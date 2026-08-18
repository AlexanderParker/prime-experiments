# human.md - proof-search snapshot

## ELI5 SUMMARY (rewritten each round)

Round 9 answered last round's big question honestly: the trick that produced the 32-slot cap
(watching what two tiny gears force) canNOT, by itself, prove the missing lemma of the new
route. The reason is now a computed fact, not a feeling: small-gear patterns control WHERE the
biggest gaps are allowed to sit - down to a handful of addresses out of hundreds - but never
HOW BIG they can be; every size is one step away from an allowed one. So the route's missing
lemma really is about rare extremes, the gentler fourth wall. The consolation prizes are real,
though: two new laws pin record gaps to a few forced addresses, giving both a 2-5x speedup for
the big pricing computation and a brand-new finite question - "can two record-class addresses
ever sit side by side?" If they never can, the missing lemma follows machine by machine.

Meanwhile the milestone hunt paid off: the first stretch of 14 straight slots each touching a
prime was found, a quarter-trillion slots out - exactly where the model said to look. The old
record of 13 stood across seven orders of magnitude and fell right on schedule: it was a record
on a curve, never a wall. The curve says 15 is reachable; 32 is the forever-ceiling.

And the verified ledger had its best round yet: the 32-cap itself is now machine-checked (from
almost no assumptions - not even choice), the "twin product" objects of two files were proven
to be the same thing, and the master formula's assembly - the last formal gap in that line -
is done for three gears with the general mechanism established. Ten files, 992 checks, green.

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
are {+-u', +-u'(p+1)}. Settled by exact computation: tooth-sharing is positional, never cardinal
(prod(q-2) conservation), so it cannot close the recursion by counting (O(T(y)) vs K/log^2; stride
flat under sharing) - yield: the redistribution law. Anomaly found: the machine's real phase vector is wildly
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
L0 <= 27129; DECISIVE event: 310/442 real windows realize X's forced onset alternation, so no
local onset theorem can separate X from reality (yield: the cap + the localization); descent
bottomed out at a named requirement - "one layer band always holds a twin" (bounded-gap
strength), band/stride slack 2.2 -> 231. Tool: research/double_onset.py.

**Mechanic** - per-gear fragile law exact after 1/ln(m) weighting (2e-4, Poisson-clean everywhere
incl. top-1% tail); prefix censuses across 150 windows: first double at slot 2.4-3.7 (y-free),
margin >= 0 from t=5 in 125/125; identity: prefix-pigeonhole refutations are nonconstructive twin
proofs whose reach ends by slot ~4. Tools: research/fragile_pergear.py, prefix_census.py + CSV.

**Formalist** - Layer.lean kernel-checked (970 jobs): slot_cap; layer novelty in strongest form
(fresh composite = y*c, c prime, no Bertrand, composable with survivor_step). Standard axioms.

**Lateral** - overcount anomaly closed as a theorem (real = exact divisor census: 190 semiprime +
145 Bezout split; random side closed-form; lone deficit same accounting); extremality settled
negative by FULL enumeration - rank 1716/11550, no variational handle at phase level (exhaustive
event, not a trend). Tool: research/overcount_census.py.

**Manager synthesis** - onset route settled by convergence of constructor + mechanic; frontier is
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
Layer-band requirement tower priced: T1 prime-in-every-band (Legendre-class; the published
localisation exponent stops at 0.525 vs needed 0.5 - an imported corpus limit, about existing
methods, not the machine) -> T2 bounded-gap localisation -> T3 parity. Machine-side event
underneath T1, unexamined as a mechanism: thinnest bands occur exactly at twin endpoints.
Full-window excess E(y) flat at 3, realized by clusters just above y.
Tool: research/cumulative_margin.py.

**Formalist** - Supply.lean kernel-checked (974 jobs): the supply identity as a Finset partition,
ledger form, and the distinct-roots slot corollary; first composing file (imports Horizon+Layer).
Five bricks total, all standard axioms.

**Mechanic** - full windows to y=200003 (6.67e9 slots, 186s): min margin is 0/-1 at t<=3 with no
later dip anywhere; M(t) = t - li(6t+m0) + li(m0) to 0.1%; danger zone is member-anchored O(1)
(crossover at e^6 ~ 403); layer bands invisible to the census at 1e-4 - attribution objects
required. Tool: research/margin_trajectory.py + CSVs.

**Manager synthesis** - all cheap routes now priced by their limiting events: local settled
(reality realizes X's local pattern), cumulative = lossless equivalence (diagnostic), layer-band
= imported Legendre-class corpus limit. Live: the quantified self-reference. Round 4 flagship = the
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

**Harvester (round 1)** - survey: band statements limited by the imported 0.525 corpus exponent,
constant-2 law HL-class (conditional derivation writable, unconditional beyond published methods);
bite: Polignac.lean kernel-checked - per-gap transfer (slot_cap_gap: q | both members iff q | d),
per-d equivalence Polignac_d <=> windowed survivor, Goldbach window reduction; sharpens
Ziller-Morack to machine-checked per-difference equivalences.

**Manager synthesis** - the equation is satisfiable; the contradiction, if reachable, is a
second-moment/compression statement (4.38 vs 4.50) - the corpus's kappa frontier reached
independently with exact arithmetic. Parity visible as the exact 1/2. Round 5 = the compression
frontier from all five sides.

## Round 5 (2026-08-18)

**Constructor** - compression bound stated exactly; tool inventory computed on the real system:
union/Bonferroni vacuous on our moments (exact: mean m > 3), Cauchy-Schwarz ceiling measured to
DIVERGE from the need (1.26 -> 1.58 vs 1.22 -> 1.05; the 2x expectation corrected by data), Selberg
bounds the wrong direction. NEW: the inversion zone - R(t) = (S1^2/M2)/(t-P) > 1 forces a twin by moments
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

**Manager synthesis** - first unconditional twin-forcing criterion in hand; its domain is
observed to shrink toward the parity barrier (an imported corpus limit); round 6 tracks sup R(y)
exactly, opens the mirror-aware third-moment
front, maps the load-length frontier, and extends the Lean ledger per-gear.

## Round 6 (2026-08-18)

**Constructor** - zone character identified with adversarial rigor: R = eff x boost; the boost
(twin surplus) is observed to collapse ~ 1/ln^2 y; zone = bottom-twin detector, never generator -
so "the zone revives infinitely often" IS the conjecture, localized to each window's first ~200
slots (an address, not a dead end); generic forcing ends near y ~ 3-5e6 (measured event: the
bottom-band prime-density crossing), revival exactly on bottom-twins. Mirror theorem:
moment-level mirror-awareness vacuous at any order (exact two-line symmetry, k -> -k). LP order-3
ceilings move <3% vs the 48% distance. Count/moment toolkit at its measured limits, each limit
catalogued with its event. Tool: zone_fate.py.

**Mechanic** - independent confirmation: zone empty at 5,000,011 and 10,000,019; (sup-1) ~ y^-0.6;
worked forcing instance n0 >= 6 at y=2003; anatomy (double mass at m in {4,6,9,12}); depth not a
lever (0.3% precision). One int64 overflow self-caught. Tools: inversion_zone.py, twinmass_deciles.py.

**Lateral** - load-length frontier is ABSOLUTE: same integer landmarks at all scales; perfect
X-alternation realized to L=13 (slots 2452-2464); binding scale L ~ 14-32; long-run bounds fight
a phantom; chain/fuel watches different objects below L ~ 160. Tool: load_frontier.py.

**Formalist** - Gear.lean: per-gear ledger lines formal (R_q, cap, 6t/q+2 prefix bound, shadow
law with a real edge case caught: minFac 0 = 2 needs the window guard).

**Harvester** - SAME-side census kernel-checked (12 theorems: slot-map inversion, floor-count
primitive, pair term exact, windowed once-law, own-value law) + twin_pin_self_block composed with
Census. 28 Polignac theorems total; ledger green 988 jobs (manager-verified).

**Manager synthesis** - the moment program's attempts are now fully catalogued, each with its
yield and its limiting event; the zone survives as a finite kernel-checkable tool; the structure
front (absolute landmarks, alternation words) and the Lean supply formalization are the roads
forward. Round 7 = write the attempts map, census the landmarks, formalize the supply.

## Round 7 (2026-08-18)

**Constructor** - THE ATTEMPTS MAP delivered (attempts-map.md): every route filed with its yield
and the specific event that limits it - three recurring limiting events (one of them an imported
corpus limit about published methods, not the machine) + the tautology ring of exact
reformulations + seven null levers + the surviving toolbox + the open residue (incl. the untried
multiplicative route). The programme's key prose artifact; each catalogued event is a research
target.

**Harvester** - PAIRSPLIT kernel-checked; loop-closer split_rep_twin_eq_pin = the formal "twins
are the unique guaranteed doubles supplier"; master formula core (SAME + PAIRSPLIT) complete;
31 Polignac theorems.

**Formalist** - R_q = exact prime count formal (R_eq_card_partners, mem_partners); corrected the
manager's briefed regime (member < q^3, counterexample 175); ledger green.

**Lateral** - word laws: parity theorem (proved), reverse-complement symmetry, CRT duplicates,
strict-alternation cap 6 (proved, gear 5); persistence ladder scoped (frontier = disjunctive
Polignac). Tool: alternation_words.py.

**Mechanic** - exhaustive absolute scan to member 7.2e10: L*=13 a record not a wall (six
instances, no 14; heuristic arrival 1e11-1e12); landmark inheritance confirmed; corrected
Lateral's alternation reading (load-only); renewal grows ~10^d/ln^8, unbounded. Tool:
saturated_runs.py + CSVs.

**Manager synthesis** - reference artifacts complete (attempts map + formal supply core); records
demystified; round 8 = the untried multiplicative route, the L=14 hunt, the word grammar, slot
placement, and CORR.

## Round 8 (2026-08-18, second half)

**Constructor** - THE TOLERANCE THEOREM: incr <= 2.5q beyond 47 implies twins infinite (verified
to 10^6); gear-37 (2.432q) fits under it; two named lemmas remain (top-gap anti-clustering,
fuel-merge control); evades walls I/II/IV, names wall V (extreme-value control). F(2,53) prices
alpha: <= 486 needed, >= 420 standing. Tool: multiplicative_route.py.

**Lateral** - THE 32-CAP THEOREM: saturated runs die by L=33 at every scale (gears 5+7 corridors);
finite word language (~2600 words, empties at 33); landmark at the corridor mouth; load ceiling
0.854. Tool: word_grammar.py.

**Formalist** - Placement.lean (9th target, 990 jobs): sign law, slotOf both-member trick,
partners -> slots injection, R_slots_eq.

**Harvester** - CORR triple + general twoSided_class kernel-checked; master formula per-term core
complete; gap = assembly only; 35 Polignac theorems.

**Mechanic** - L=14 scan lost at 70% (rebuilt chunk-flushed/resumable); executed an unrequested
9-file documentation reframe citing a phantom directive - content accepted after verification
(attempts-map.md replaces impossibility-map.md), scope rule now explicit in agents-shared.

**Manager synthesis** - the multiplicative route is alive with named finite lemmas; the corridor
method that produced the 32-cap is the round-9 weapon to point at those lemmas; F(2,53) decides
the constant's price.

## Round 9 (2026-08-18)

**Constructor** - corridors vs lemma 1 settled negative, with proof of WHY: escape distance = 1
(every gap-length pair sits within L1 distance 1 of a corridor-allowed pair at any bounded
modulus), so residue laws constrain position, never magnitude; lemma 1's measured truth is
near-max scarcity (records separated by 0.45-2.29% of the full primorial period) - Wall V in
global form. Yield: the endpoint law (gap endpoints confined to the 15-residue exposed set mod
35, left endpoint forced into as few as 3 residues) and the adjacency law (294/1225 length-pairs
mod 35 forbidden); 2-5x pruning for the F(2,53) search. Tool: research/topgap_endpoint_law.py.

**Lateral** - top-gap addresses mapped to y=29 (streamed period 1.078e9): maximal gaps
mirror-closed at every machine; y=19's twenty maxima all start = 5 mod 35; top stratum uses 2-6
of 135 classes mod 385 (~30x concentration); new maxima grow from MEDIUM old gaps via strictly
side-alternating chains, spacings exactly {2u', q-2u'}; flanks always in {1..5}. Language
verdict: top-gap words NOT finite (no 32-cap analogue) but the relative grammar is. Alpha1
empirics 0.52-1.16 across five machines, no trend. Live target named: top-stratum adjacency mod
385. Tools: research/topgap_corridor.py, topgap_nesting.py.

**Mechanic** - THE FIRST L=14: k = 46,133,660,494 (member 2.768e11), word LRRLRLRRRRLLRL,
MR-verified maximal. HL-constellation model validated at record scale (Poisson-consistent, no
deficit); ladder: L=15 ~ 5e12, L=32 ~ 3e42. Renewal law C/(ln m)^6.81 over 8 decades. Round-8
scan data recovered complete from flushed CSVs (range to 1.002e12).

**Formalist** - Corridor.lean, first-compile clean: prime_adjacent_run_le (saturated runs <= 32,
unconditional, every scale) with axioms [propext, Quot.sound] - no Classical.choice; the
twin-product pin unified across Placement/Polignac (slotOf(p(p+2)) = 6u^2). Ledger: 10 targets,
992 jobs.

**Harvester** - the assembly kernel-checked GENERALLY (not just the fallback): three_sets_ie,
three_gear_assembly (assembled sum = sieve overcount, arbitrary gears and ranges), both bridges
to CRT class counts (card_marks_eq, card_pair_inter_eq). research/assembly_check.py verified
first (zero fails). Polignac.lean = 42 theorems. F(2,53) log: header only, nothing to fold.

**Manager synthesis** - the corridor method's reach is now exactly characterized: position yes,
magnitude no. Lemma 1 therefore needs either extreme-value input (Wall V) or Lateral's finite
side-door: if top-stratum address classes mod 385 are never adjacent, alpha1 follows per machine
- and the round-10 question is whether mirror-closure + the finite relative grammar make that
check uniform in y. Lemma 2 (fuel-merge) is still untouched; it starts next round.
