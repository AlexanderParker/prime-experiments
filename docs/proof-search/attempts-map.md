# The attempts map

Round-7 deliverable of the proof-search team's constructor workstream, 2026-08-18;
reframed 2026-08-18 per the user's framing directive. The directive, which this
document now follows and which future entries should follow: record every route as an
ATTEMPT -> YIELD -> LIMITING EVENT. A limit is only final when it is a pinpointable
event with an understood mechanism; a limit based on a trend to infinity is an
OBSERVATION that locates missing structure, not a wall; a limit imported from the
published literature (a best-known exponent, an unproven standard conjecture) is a
fact about human methods, not about the integers, and is marked as a CANDIDATE
REOPENING. Understanding a limiting event - why it occurs, what mechanism produces
it - is itself a research avenue, and each entry keeps what the attempt yielded.

This document is the record of every route to a twin-prime contradiction that this
team and the corpus before it have attempted, each with (a) the exact statement of
the route, (b) the limiting event, and (c) the yield. It is organised by the
underlying limit event each route meets, because the limit events are few and the
routes are many, and routes that LOOK different usually meet the same event wearing
different clothes.

Provenance shorthand: [corpus] = docs/handover.md; [review] = docs/review-2026-08-17.md;
[C-n], [L-n], [M-n], [H-n], [F-n] = constructor / lateral / mechanic / harvester /
formalist round n of this session (details in the workstream docs and agents-shared.md).
Status labels are used in the corpus's sense: *theorem* (proved, sometimes
kernel-checked), *computation* (exact, exhaustive at stated scale), *measured*
(reproducible data, no proof), *equivalence* (reformulation at full conjecture
strength).

## 0. The target and the frame the routes are stated in

Condition X: some window (y, y^2) contains zero twin slots - every slot k (pair
6k-1, 6k+1, both members strictly interior) has a composite member. Refuting X for
every y is the twin prime conjecture (kernel-checked iff, proofs/BlockedSlots.lean).
Equivalent forms accumulated by the programme, all exact, all interchangeable:

* Reduction A [corpus]: the all-umbrella slot recurs inside every window.
* Census form [C-1, kernel-checked as Census.census_pinned]: X <=> n1(t) = P(t) and
  n2(t) = t - P(t) at every prefix - the zero-slack census.
* Floor form [C-4]: X <=> P(t) = t - D(t) at every prefix, D = the double count -
  the window's prime census sits at its unconditional pointwise floor.
* Compression form [C-5, M-4]: X <=> the freedom-free cross-root hit schedule S1(t)
  compresses into distinct slots at mean multiplicity exactly S1/(t - P), every prefix.
* Supply form [L-4]: X <=> N(t) - P(t) = B(t) - U(t), right side pure floor
  arithmetic over the primes and prime gaps below y (master formula).

These equivalences are the frame, not routes: each is lossless, which is precisely why
none of them alone is progress (section 4). A contradiction must at some point replace
an exact quantity with a BOUND, and every bound tried so far has met one of the limit
events below.

---

## Limit event I. Abundance: first-moment / capacity arguments

The mechanism: the machine's covering capacity exceeds its covering obligation by a
divergent factor - sum 2/q over gears diverges, so capacity is abundant, not scarce.
Any argument of the form "the gears cannot cover that much" meets this event at a
computable surplus. Every route below is a differently-dressed first moment, and each
meets the same event: abundance, with the surplus measured exactly. (The divergence
itself is exact arithmetic; the event is pinpointable per route as the y where the
surplus crosses the route's threshold.)

**I.1 One-scale capacity counting.** [corpus 5.1]
(a) A run of L slots receives at most ~2L/q positions from gear q; if sum 2/q < 1 a
run cannot be fully covered. (b) Limiting event: sum 2/q >= 1 from y = 13 on; by
y = 47 the gears carry 132% surplus, and the surplus grows without bound. Works only
for y <= 11, never again - the crossover at y = 13 is the event. (c) Yield: the
per-gear cap 2(floor(L/q)+1) itself (exact, now kernel-checked as Gear.R_prefix_le
in the R-form) - the supply side of every later ledger.

**I.2 Two-scale repair.** [corpus 5.2]
(a) Split gears at z; small gears cover a full period F(z), large gears are counted by
capacity; bound follows if F(z) * 2 * sum_{z<q<=y} 1/q < 1. (b) Limiting event: the
optimum is always z = 3, where the condition needs sum 1/q < 1/6 against an actual
0.51 at y = 13; shortfall 3-5x (growing with y - the growth is an observation, the
3x shortfall at the optimum is the event). (c) Yield: the F(z)-periodicity trick
(used correctly later in graded/banded arguments, C4 grading).

**I.3 The doubles pair-coincidence squeeze.** [C-1 sec 5b; re-examined at the moment
level in C-5]
(a) Under X the doubles demand n2 = N - P must be met by cross-member coincidences of
distinct gear pairs (slot-cap lemma); bound the supply by sum over pairs 2L/(qq') and
win if the pair density s(z) = (sum 1/q)^2 - sum 1/q^2 < 1. (b) Limiting event: an
exact empty intersection - s(z) < 1 requires active gears <= 137, hence band top
< 139^2 = 19321, hence ln(6L) < 9.87 - but a band that short admits up to
12/ln(6L) >= 1.22 primes per slot (Brun-Titchmarsh), above the 1 - s < 1 per slot
that X is forced to supply. Non-vacuous coincidence bound and prime-thin band never
coexist; the intersection is empty at specific, named thresholds. (c) Yield: the
computation pattern (it is the corpus-5.2 squeeze, now on the doubles side - a
useful sanity template for any proposed short-band bound).

**I.4 Tooth-sharing counting.** [L-1]
(a) Twin-pair gears (p, p+2) share tooth structure; hope: sharing wastes kills at a
rate that starves the covering. (b) Limiting event: over full periods sharing changes
nothing (prod(q-2) conservation - the mechanism is positional, never cardinal);
in-window the net survivor effect is O(T(y)) against a needed ~K/log^2, and both
guaranteed wasted kills land on already-decided slots (self-block slot, product slot
- composite by construction). Zero new open slots. Max stride is insensitive to
sharing (+0.02 +- 0.05 slots). The conservation law is the event: sharing is
positional by theorem. (c) Yield: the pinned-class closed form {+-u', +-u'(p+1)}
- which became the roots-of-unity law, the gap law, and the g=2 guaranteed supply
line (the single most productive yield of any attempt in this session).

**I.5 Capacity at scale, final form.** [M-4]
(a) Last version: perhaps X's demand exceeds the freedom-free cross-root schedule
S_pair(t) in some depth range. (b) Limiting event: none exists in any window
computed - tau(t) = (t-P)/S_pair rises monotonically to its window-end max, and that
max declines with y (0.314 -> 0.222, y = 503 -> 50021); slack 3.2-4.5x (the decline
and the lnln^2 growth of slack are observations, not a wall; the per-window absence
of a binding range is exact at 4.17e8 slots, matching what [corpus 5.1] proved at
y=13). (c) Yield: S_pair(t) itself, the exact schedule, and tau as the compression
dial - the objects the moment program (limit event III) was then fought on.

---

## Limit event II. Localisation: prime lower bounds beyond current methods

The mechanism: several routes reduce X's refutation to "primes must appear at density
> 1/6 per integer in a short range" (superdense, Hensley-Richards class) or "a prime
must appear in an interval of exponent 1/2" (Legendre class). Published unconditional
technology stops at density ~1/ln x and exponent 0.525 (Baker-Harman-Pintz)
(imported corpus limit - a fact about published methods, not the machine; candidate
reopening). The 0.525 floor is anchored within the literature (improving pair
localisation below it improves single-prime localisation first), which prices the
reopening honestly: it is a literature-wide event, not a machine event. Where a route
below meets a MACHINE event as well (a real window realising X's pattern), that is
stated separately - those are the decisive ones.

**II.1 Global pigeonhole (C1).** [C-1]
(a) X requires P <= N: at least as many composite members as slots. (b) Limiting
event: holds only while prime density among +-1 mod 6 members exceeds 1/2, i.e. y
below ~e^6 = 403 (measured: P - N = +7 at y = 13 and y = 23; passes with +46 slack
at y = 47). The density crossover at member e^6 is the event - the same absolute
constant that anchors the margin drift [M-3]. (c) Yield: unconditional refutation of
X for all small y by pure counting - the base case of everything.

**II.2 Prefix/run pigeonhole, local form (the onset route).** [C-2, M-2]
(a) No double can exist before the first double slot (onset lag L0(y)); under X the
onset prefix is perfectly fragile - exactly one prime per slot. Hope: prove a twin
inside it. (b) Limiting event, twofold. The input that would close it -
pi(y+H) - pi(y) >= H/6 + 1 at H = 6*L0 + 2 - is superdense-class (imported corpus
limit); but the decisive event is a machine fact: it is FALSE as a universal
statement, because 310 of 442 real windows realise the forced alternation exactly
(twin-free onset prefix). No theorem can refute X in a region where X's forced
pattern actually occurs - an exact counterexample census, the strongest kind of
closure. Onset asymmetry at scale [M-2]: first double at slot 2.4-3.7 mean (max 9,
y-independent), margin never below -1, negative only at t <= 4 for y >= 1e4 - the
local weapon's reach ends by slot ~4. (c) Yield: the unconditional onset cap
L0(y) <= 27129 for every y (Montgomery-Vaughan, exact crossover at e^12) - a clean
absolute theorem - and the forced-alternation lemma (under X, P(t) = t below the
first double), the sharpest local statement of zero-slack.

**II.3 The inversion zone, as a route to a theorem.** [C-5, C-6, M-6]
(a) Where R(t) = (S1^2/M2)/(t - P) > 1, second-moment arithmetic alone forces
n0 >= 1 (at y = 2003, t* = 24 the histogram forces n0 >= 6 - six real twins from
floor arithmetic). Route: prove R(t*) > 1 for all y. (b) Limiting event: turning the
zone into a theorem needs P(t) > t - S1^2/M2, a short-prefix prime lower bound at
0.42-0.80 per slot (0.07-0.13 per integer) - superdense class (imported corpus
limit). Machine event alongside: the zone empties generically - empty at the
computed windows y = 5,000,011 and 10,000,019, with the mechanism identified
(the twin-surplus boost 1 + n0/(t-P) collapsing ~1/ln^2 y, not M2's dispersion);
the decay fit (sup-1) ~ y^-0.6 is an observation, not a wall. Its revivals are
windows opening with a twin - see IV.3 for the exact equivalence, and section 6 for
what the zone still IS. (c) Yield: a finite-domain refuter valid for every y below
~3e6 and sporadically beyond, certifying twins from moments + P without exhibiting
the pair - kernel-checkable, the strongest finite artifact of the session.

**II.4 The layer-band descent.** [C-2, C-3]
(a) X at y makes every layer band (y'^2, y''^2) above y twin-free; the induction
closes if every band above some point contains a twin. Bands are x^(1/2+o(1)) long at
height x; the thinnest (4*sqrt(x)+4) occur exactly when (y', y'') is itself a twin -
the self-reference sits at the binding case. (b) Limiting event, a tower, in order:
T1 - a PRIME in every band - is OPEN (Legendre class: implied by Legendre's
conjecture, NOT implied by RH, Cramer suffices); T2 - a bounded-gap pair in every
band - published localisation stops at exponent 0.525 (Alweiss-Luo 2018,
arXiv:1707.05437; density is ample, placement is not); T3 - gap exactly 2 - the
parity step (246 -> 2), no partial result in the literature. T1 and T2 are imported
corpus limits (facts about published methods, not the machine; candidate
reopenings). The route stops at T1 before its twin content engages. UNEXAMINED
MACHINE EVENT flagged here [M-8 classification]: the thinnest bands sitting exactly
at twin endpoints is a pinpointable structural fact of the machine that no one has
interrogated as a mechanism rather than an obstacle - the self-reference at the
binding case is itself a research avenue. The one-band form is equivalent to
Reduction A by the tiling (see IV.5). (c) Yield: the tower itself (a precise map of
what any descent-type argument must cross, with the literature anchored:
Maynard-Tao density surplus x^(1/2)/polylog vs the 0.025 exponent deficit), and the
layer law's localisation of induction inputs from window scale to layer scale.

---

## Limit event III. Parity / second-moment: bounds that cannot see the twin mass

The mechanism: the classical parity barrier, met here in exact-arithmetic form -
every inequality that discards positional information down to moments (any order
computed) lands a factor above X's need, and the factor is measured, not conjectured.
As a literature boundary this is an imported corpus limit; but the session gave it a
machine-side statement that is sharper and exact [M-5, corroborated C-6]: the entire
reality-to-X distance lives in the ZEROTH moment of the multiplicity distribution -
the twin mass P(omega_L = 0 & omega_R = 0) - which no power moment sees, at any
order (0^j = 0 is the mechanism, stated in one line). Note also what the machine has
that sieve axioms do not model: freedom-free placement, mirror structure, and
all-order exactness [C-5 sec 16] - the parity boundary is a fact about
moment-shaped arguments, not about every argument.

**III.1 The X-consistency equation, count level.** [C-4, with L-4, F-4/5]
(a) Under X the demand is pinned with zero slack (n2(t) = t - P(t), kernel-checked)
and the supply is freedom-free floor arithmetic (master formula). Equate; hope the
system is overdetermined. (b) Limiting event, exact: degrees of freedom are ZERO
on both sides - the census theorem (horizon) makes the demand side ITSELF gear
arithmetic, so the y^2/6 equations collapse to n0(t) = 0 with no residual structure.
What X needs P(t) to do is sit at its unconditional pointwise FLOOR t - D(t); a
conflict from below is impossible (the floor is the identity's minimum), and from
above the Montgomery-Vaughan ceiling sits at exactly twice the floor's asymptote -
headroom rho(t) measured 0.4687/0.4785/0.4828 (y = 101/211/503), drifting to 1/2.
The parity factor 2, photographed live. Any theorem separating P(t) from its floor at
one t IS a twin-existence theorem. Note the MV constant's own rigidity is
parity-class (Motohashi's Siegel-zero linkage for the progression form): the ledger
and the analytic boundary are one mechanism, two faces. (c) Yield: the equation as
the programme's central identity (all five forms of section 0), the rho photograph,
and the supply decomposition - the g=2 pins from twins below y as the unique
unconditionally guaranteed line item of X's doubles budget (5-9% at every scale and
depth; the other 91-95% alignment-conditional and ample).

**III.2 Moment ceilings on compression: union, Bonferroni, CS, LP, third order.**
[C-5, C-6]
(a) X <=> the hit schedule compresses at M_X = S1/(t-P); reality compresses at
M_real = S1/n2; contradiction needs an unconditional ceiling C(t) < M_X(t) somewhere.
(b) Limiting event, each ceiling priced exactly: the union bound gives a floor,
never a ceiling; Bonferroni-2 is vacuous at every scale tested (mean m > 3); the
Cauchy-Schwarz/Turan ceiling C_CS = M2/S1 overshoots the need by 1.26x -> 1.58x
(y = 211 -> 5003) and growing (it tracks the lnln-divergent dispersion) while the
window a winning ceiling must hit narrows (M_X/M_real = 1.22 -> 1.05) - the two move
apart on both ends (observation: a measured divergence, quantifying how far
moment-shaped ceilings sit from the need); the expectation that the ceiling would
land at the parity factor 2x was tested and settled: it is worse than 2x
asymptotically. Integer LP refinement of CS: +0.3-0.5%. Third moment: exactly zero
conservatively (the cubic never enters the optimal basis); +0.6-2.8% with the
legitimate arithmetic cap. Against a 48% chasm at window scale. (c) Yield: the
sharp LP moment-problem machinery (it extends the inversion zone's edge slightly and
is the right formal container for any future finite-order claim), and the measured
divergence itself as a pricing template for "just use a better inequality"
proposals.

**III.3 The Selberg / upper-bound direction.** [C-5; corpus context in review sec 6]
(a) Selberg Lambda^2 and the large sieve bound survivor counts. (b) Limiting event,
structural: they bound n0 from ABOVE (factor ~4 over the HL prediction), and X
asserts n0 = 0 - no upper bound contradicts zero; the direction mismatch is the
event. The large sieve's content on this class system is exactly the
translation-averaged second moment of III.2. In standard vocabulary [review sec 6]
this is the dimension-2 sieving exponent: what lower-bound sieves certify is a
survivor in windows of length ~y^beta with beta_2 ~ 4.3-4.9, against the needed
exponent 2; Reduction A is precisely "beta_2 can be pushed to 2 for the twin
pattern". (c) Yield: the vocabulary bridge - our exact ledger and the parity
literature name the same obstruction, which prices any future "new inequality"
claim immediately.

**III.4 The kappa/hazard program.** [corpus 5.4-5.6, item 22; review secs 2-5]
(a) Form (b): hazard h(L) >= d for all L would give N(L) <= P(1-d)^L and refute X
geometrically. (b) Limiting event, two-part, priced by the review: it is
OVERSUFFICIENT (implies near-optimal pair-Jacobsthal growth F ~ y log^2 y, in direct
tension with the corpus's own quadratic growth reading - at most one is true), and
its provable region (fixed small L: h(1), h(2), h(3) are theorems, kappa(2) limit
= 2 - (11/3)C = 0.5448) is separated from the needed region (L ~ F, L*d ~ 8-18 and
growing) by a regime gap no fixed-order correlation expansion crosses - the regime
gap is the event, and it now has its own entry (limit event V), where the machine's
native instruments point at the exact open statement. The per-step gear recursion
cannot be summed in the elementary form (gear-37 increment 2.432q > 1.8q [corpus
5.4] - since corrected: see the round-8 amendment); chain length k is not boundable
from gap structure alone (compatible for every k once F >= q-1 [corpus 5.5]; k = 4
fuel exists at gears<=29 [chain-conditions]). (c) Yield: the three proved hazard
inequalities, the exact psi/deficit machinery (research/kappa_exact.py,
deficit_scan.py), the fuel-word census - and the review's multiplicative route,
since opened as the round-8 amendment's live target.

**III.5 Where the gap actually lives: the zeroth moment.** [M-5; corroborated C-6]
(a) Hope: some higher moment or distributional feature of the multiplicity spectrum
distinguishes X from reality. (b) Limiting event, the sharpest exact statement of
the session's second half: the mean is pinned by arithmetic, variance and tail carry
the real-vs-independence excess but NONE of the X-gap; the entire distance is P0,
the both-unmarked mass - i.e. the twin density itself, invisible to every power
moment (0^j = 0). The product null (independent omega_L, omega_R sides) reproduces
variance, tail, AND P0 to ~1.4 points; the real machine sits below the product
baseline by exactly the HL correction (ratio 0.85 -> 0.77 down the ladder,
observation). (c) Yield: the product-model bookkeeping, and a scoreboard: the twin
mass is a REPORTING METRIC, not a mechanism (user direction, 2026-08-18) - it tells
any proposed argument exactly what it must force and verifies models to fractions
of a percent, but improving its measurement is not a path; the forcing must come
from mechanism-level structure (placement, pins, alternation constraints).

---

## Limit event IV. Equivalence-to-target: the tautology ring

The mechanism: the ledger is exact in both directions, so every reformulation built
from its identities alone is at full conjecture strength. These are events of the
cleanest kind - each was proved equivalent, so no one need walk that road again
expecting new leverage; what each contributes is a new coordinate system on the same
object. The pattern: the gears drop out, a prime-counting statement remains, and its
violation measure IS the twin count.

**IV.1 The cumulative statement (CUM).** [C-3]
(a) For every y, some run I in (y, y^2) has P(I) > N(I). (b) Limiting event: exactly
equivalent to Reduction A (two lines each way via the slot-cap pigeonhole; a twin
slot is an excess-1 run). There is no ingredient weaker than the conclusion.
(c) Yield: the diagnostic margin M(t) and excess E(y) (measured collapsing to a
flat 3, carried by 3-5-slot clusters within ~700 of y), and the literature bridge to
prime clusters.

**IV.2 The run/prefix condition as a family.** [C-1, C-3]
(a) X => every run holds at most one prime per slot; sharpest at run length 1.
(b) Limiting event: at length 1 it IS X; the family adds measurability, not
strength; its unconditional reach is limit event II's (ends at ~e^6 globally,
slot ~4 locally). (c) Yield: the margin trajectory as the standard violation meter
(M-3's li-model fits it to 0.1%).

**IV.3 The inversion zone's revival.** [C-6]
(a) Hope: the zone revives infinitely often, refuting X unboundedly. (b) Limiting
event: every twin (p, p+2) sits in the first slots of the window of any prime just
below p, and a revival's fuel IS a bottom twin: "the zone revives infinitely often"
is equivalent to the twin prime conjecture. The zone is a bottom-twin DETECTOR,
never a generator - and, stated as an address rather than a closure: the conjecture
is now localised to a floor-arithmetic-checkable event in the first ~200 slots of
each window. (c) Yield: the detector itself (II.3c), plus the adversarial-checking
discipline that caught the circularity before it was claimed.

**IV.4 Separating the floor.** [C-4]
(a) Hope: some unconditional theorem forces P(t) > t - D(t) at one prefix. (b)
Limiting event: the separation equals n0(t) identically - any such theorem is a
twin-existence theorem by definition. (c) Yield: an instant classifier for proposed
attacks: if a proposal's conclusion implies P(t) exceeds its floor anywhere, it is
priced at full conjecture strength before any work is done.

**IV.5 The one-band descent.** [C-2 caution, C-3]
(a) X at y is refuted if some layer band in (y, y^2) holds a twin. (b) Limiting
event: the bands tile the window, so one-band = twin-in-window = Reduction A; only
the every-band form buys a height-uniform theorem, and that form is II.4. (c) Yield:
the tiling observation, which localises what any induction genuinely needs.

---

## 5. Levers tested and settled (events, each with its mechanism)

Structural features that looked like handles, each settled by theorem, exhaustive
enumeration, or 0.3%-precision measurement. These closures satisfy the event
criterion - the mechanism is visible in every line - and each says what any future
use of the feature must add:

* **Mirror symmetry, at moment level** [C-6, theorem, two lines]: k -> -k swaps
  omega_L/omega_R and fixes m_k; every mirror-augmented moment doubles and every
  ratio in the programme is invariant. Settled at any moment order. What it rules
  IN: any mirror edge must use positions jointly with signs - the surviving strip.
* **Depth** [M-6, measured to 0.3%]: the twin mass by depth decile is reproduced
  exactly by a flat HL 1/ln^2(member) allocation - no band structure; the global
  0.77-vs-baseline is pure density falloff. Depth alone carries no signal beyond
  density; any depth-based claim must produce structure this measurement did not see.
* **Phase extremality** [L-2, exact full enumeration]: the real phase vector +-u' is
  merely high on waste metrics (rank 1716/11550 on overcount in the {5,7,11} space),
  never extremal. No variational handle; "special point of phase space" means
  exactly "the census is deterministic".
* **Tooth-sharing as cardinality** [L-1]: positional only; prod(q-2) conservation
  (see I.4 - whose yield became the g=2 line, the counter-example to "settled
  means sterile").
* **Layer bands through the census** [M-3, 1e-4 precision]: the margin M(t) is
  GEAR-BLIND - slope across every band boundary p^2 equals matched controls. Band
  structure can enter only via per-gear attribution objects, never via the census.
* **Hub enrichment at binding loci** [L-5]: hub-rate/ambient = 0.999-1.006 at the
  most X-like stretches - near-binding regions are not hub-enriched.
* **Chain/fuel structure at the binding scale** [L-6]: load-extremal runs (short,
  absolute, prime-dense) and length-extremal strides (deep, gap-word-governed) are
  different extremal families, merging only at L = maxstride. Chain analysis does
  not see the binding region L ~ 14-32; binding-region work needs its own
  instruments (the saturated-run census, M-7/8).

## 6. Yields: the toolbox the attempts built

Nearly every attempt left an exact object behind; this is the possibility side of
the map, and it is long.

* **Finite-reach refuters, unconditional**: C1 pigeonhole (all y with a prime-rich
  prefix; everything below ~e^6); the inversion zone (moments + P force n0 >= 1,
  valid every y < ~3e6 and sporadically beyond; forces n0 >= 6 at y = 2003 from
  floor arithmetic; kernel-checkable, finite).
* **Absolute constants and caps**: onset cap L0(y) <= 27129 for every y (MV at
  e^12); first double-composite slot k = 20 in all of N; deletion spacing (q+-1)/3;
  the drift-sign event at member e^6 ~ 403; the tower exponents as literature
  anchors (0.525 vs 1/2; band lengths 4*sqrt(x)+4 minimum).
* **Exact laws** (all verified, several kernel-checked): the arithmetic census
  theorem; the roots-of-unity law (doubles = nontrivial square roots of 1 mod active
  semiprimes); the gap-graded split law (closed-form class representatives); the
  master supply formula (n2 = B - U, exact at every prefix); the g=2 pinning theorem
  with uniqueness (only twins pin at their own slot - kernel-checked, the first
  formal fact distinguishing twins from other gaps); the supply/census/bridge
  identities.
* **The photographs** (measured invariants that price future proposals instantly;
  observations, held as calibration, not walls): rho -> 1/2 (the parity factor as
  MV-headroom); M_X/M_real = 1 + n0/(t-P) (the whole X-gap as twin share);
  C_CS/M_X growth while the need narrows; tau max declining (capacity slack
  growing); g=2 supply share 5-9%.
* **The Lean ledger** (9 files, green, standard axioms): BlockedSlots, Horizon,
  Layer, Supply, Census, Bridge, Gear, Polignac (28 theorems incl. the ZM-frame
  per-gap equivalences, the g=2 pinning, the SAME census), AxiomCheck. The reduction,
  the frame, and the demand side of the X-equation are machine-checked end to end.

## 7. Open avenues (the live map)

This map records attempts, not fate. Standing open, with owners and priced caveats:

1. **The structural fronts** [L-6, M-7/8]: the load-length frontier is ABSOLUTE
   (record twin-free runs are fixed integer landmarks; the load-1 pattern realised
   to length 13 at slots 2452-2464, recurring - six L=13 instances to member
   7.2e10, first L=14 not yet located; the constellation model expects it within
   members ~1e11-1e12, and either its address or a quantified model deficit is an
   event worth having) and the binding scale for any compression bound is
   L ~ 14-32 - short-run structure, not asymptotics. Open: the alternation-word
   structure of record runs vs the machine's laws (note [M-7]: record-run side
   words are NOT strictly alternating - RLLRRLLLLRLRL at the landmark). Priced
   caveat, flagged: saturated-run persistence at all depths is itself
   HL-constellation-class; this front maps the binding region exactly and its
   scoping is Lateral's task.
2. **The multiplicative tail route** [review sec 7, point 3; opened in C-8]: see
   the round-8 amendment - now the map's most concretely-targeted avenue, with an
   explicit sufficient constant (2.5q) verified to 10^6 and two named missing
   lemmas.
3. **The Lean supply program** [F-6, H-3]: remaining formal gaps are PAIRSPLIT's
   closed-form representative (the m0/b0/i law), the signed multi-gear CORR terms,
   and U-membership/B-side - all reducible to six_mul_class + card_class_Ico
   instances already in the ledger. Completing them makes the master formula, and
   hence the entire X-consistency equation, kernel-checked at every prefix.
4. **The zone's finite domain as certificates** [SUMMARY r6]: every y < ~3e6 admits
   a finite, kernel-checkable moment certificate of n0 >= 1 (sometimes n0 >= 6). Not
   a route to the conjecture; a genuinely new certificate FORMAT (twin existence
   from floor arithmetic without exhibiting the pair), publishable alongside the
   Polignac frame.
5. **The discriminating measurement** [review 7a]: F(2,53), in progress at >= 416,
   decides the growth-law question (quadratic vs geometric) on which the strategic
   read of III.4 rests; F(2,59) would confirm.
6. **The self-reference at the binding case** [flagged M-8]: the thinnest layer
   bands occur exactly at twin endpoints (II.4). A pinpointable structural fact,
   so far treated only as an obstacle; uninterrogated as a mechanism.
7. **What this map cannot rule out**: a genuinely new idea beyond the recorded
   events - in particular, anything that uses positional information jointly with
   signs at full level (beyond moments, short of tautology - the strip identified
   in [C-5 sec 16] and narrowed but not emptied by [C-6]). The map's claim is not
   that X is unrefutable; it is that every refutation-shaped argument expressible
   in first and second moments, capacity, localisation-below-published-exponents,
   or lossless reformulation now has its limiting event on record, with the
   mechanism named.

## Closing note

The single lesson of the map, earned twelve routes over: **this machine's exactness
is double-edged**. Every identity is lossless, so every reformulation is the
conjecture again; every bound is lossy in a recorded way, and the loss is always
either abundance (event I), a prime lower bound beyond current published methods
(event II - the one event partly imported from the literature rather than the
machine), or the parity factor (event III) - and these are not metaphors but
numbers with addresses: 132% surplus at y = 47, 0.525 vs 0.5 in the literature,
rho -> 1/2, a zeroth-moment gap no power moment sees. The trend-shaped entries above
are observations that locate missing structure; the event-shaped entries are places
to stand. The programme's live directions are, accordingly, not a better inequality
but different information: the structural fronts and multiplicative aggregate of
section 7, the strip between moments and tautology, and the two flagged unexamined
machine events (the binding-case self-reference; the L=14 experiment). Everything
else on this map has its limiting event named, which is what makes it a map.

---

## Amendment, round 8: the fifth limit event, and open avenue 2 resolved into a target

Open avenue 2 (the multiplicative route) was opened in [C-8]
(constructor.md sec 19; research/multiplicative_route.py). Outcome: the route is
LIVE - it evades limit events I, II, and IV outright, and the dimension-1 test
shows it is not event III either (its increment statement is unproven even for the
one-residue Jacobsthal function, where Iwaniec's theorem exists and parity does
not obstruct). The map therefore gains a fifth limit event, and this one arrives
with partial theorems attached:

**Limit event V. Extreme-value control of sieve patterns.** The mechanism: bounding
the GROWTH of the top of a sieved set's gap distribution under one gear-addition -
the regime the review called the regime gap [review sec 5] and the corpus met as
k-unboundability [corpus 5.5]. No fixed-order moment expansion reaches it (that
is event III's territory ending); no capacity count touches it (event I); no prime
input appears in it (event II). It is a statement about the machine's own gap word.

What now sits ON this event, exactly stated [C-8]:

* **Corrected closure.** Corpus 6a ("per-step increment bounds cannot deliver
  the constant") holds only for the odd-sum elementary chain and its 1.8
  threshold. With the sharp prime sum the tolerance is alpha*(y) ~ ln y
  (5.6 at y=101, 13.3 at 10^6), so ANY fixed per-step constant delivers:
  **incr <= 2.5q at every consecutive step beyond 47 implies twins infinite**
  (verified exactly to y = 10^6, worst ratio 0.656; Rosser-Schoenfeld beyond;
  observed maximum 2.432q at gear 37 - the earlier "anomaly" is harmless here).
* **The two missing lemmas**, both gap-word extreme-value statements: top-gap
  anti-clustering (F2 - F = O(q)) and fuel-merge control (excess = O(q));
  measured <= 1.24q and <= 1.62q at their separate maxima. The saturation
  regime (q > F(M)) is already a theorem with alpha = 1, but the consecutive
  chain never enters it.
* **Decision point:** F(2,53) (running, >= 420) prices the constant - alpha=2.5
  demands <= 486, alpha=3 demands <= 513.

Classification note for future proposals: an attack belongs to event V iff its
missing input is a statement about the machine's own gap word (adjacency and
alignment at the top of the gap distribution) with no prime-counting content.
Event V is where the corpus's structural instruments - merge transform, chain
condition, fuel census, isolation law, forbidden configurations - are the native
tools, and it is the only entry on this map with a nonempty stock of partial
theorems pointing at the exact open statement. It is, by the framing directive's
own standard, the map's best specimen: a limit that arrived as a pinpointable
event, whose mechanism is understood, and which is already an avenue.
