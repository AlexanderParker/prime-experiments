# Constructor round 1: Condition X, its exact ledger, and where real windows refute it

Workstream: proof by construction/contradiction from the proven mechanical laws.
Script: `research/constructor_ledger.py` (all numbers below reproduced by it; assertions
in the script enforce every identity claimed here).

## 1. Definitions (exact)

* Slot `k` = the pair `(6k-1, 6k+1)`. Gear `q` (prime >= 5) **blocks** `k` iff
  `k = +-c_q (mod q)`, `c_q = 6^{-1} mod q` - equivalently iff `q` divides a member.
* **Window** `W(y)` = `{k : y < 6k-1 and 6k+1 < y^2}` (both members strictly inside
  `(y, y^2)`). `N = |W(y)|`. `P` = prime members among the `2N` members,
  `C = 2N - P` composite members.
* **Root kill**: a composite member `m` of the window is killed at its root by the unique
  gear `lpf(m) <= sqrt(m) < y` (horizon theorem: the top gear contributes nothing
  interior; verified `R(top) = 0` below).
* Census: `n0 / n1 / n2` = slots with `0 / 1 / 2` composite members (**twin / fragile /
  double**).

**Condition X** (the contradiction target): *there exists a window `W(y)` containing no
twin - every slot of `W(y)` has at least one composite member, i.e. receives at least one
root kill.*

## 2. Two lemmas that make the ledger overlap-free

**L1 (slot cap).** No gear blocks both members of a slot: `q | 6k-1` and `q | 6k+1`
imply `q | 2`, impossible for `q >= 5`. Hence (i) a gear root-kills at most one member
per slot; (ii) the two root kills of a double slot come from *distinct* gears; (iii) a
slot carries at most 2 primes and at most 2 root kills - the multiplicity per slot is
0, 1, or 2 and nothing else.

**L2 (partition of supply).** Root (lpf) attribution partitions the composites: each of
the `C` composite members has exactly one root gear, so with `R(q)` = number of window
members whose least prime factor is `q`,

    sum_q R(q) = C = 2N - P        (asserted in the script, all y)

with **zero double-counting**. Where does the overlap accounting go? Into acts, not
kills: by the composite root law, a member with several gear factors is the site of a
squarefree-product act (each pair overlaps same-member exactly once per window, at its
own value, if it fits); those are extra *acts* on a member already killed, never extra
killed members. The ledger below is therefore exact, not an inclusion-exclusion estimate.

Per gear, `R(q)` decomposes by the root ordering as

    R(q) = [y < q^2 < y^2] + #{r prime > q : y < qr < y^2} + H(q),

square + coprimes + higher-order `H(q)` (cofactor composite with lpf >= q, e.g.
`125 = 5^3`, `175 = 5^2*7`). The corpus formula `pi(y^2/q) - pi(q) + 1` is the first two
terms; the tables below show `H(q)` explicitly (it is large only for gear 5).

## 3. The census under X: the zero-slack theorem

Identities (hold always, asserted): `n0 + n1 + n2 = N`, `n1 + 2 n2 = C`,
`P = n1 + 2 n0`, and therefore

    N - P = n2 - n0.

**Zero-slack theorem.** Under X (`n0 = 0`) the census is *forced*, globally and in every
prefix `(y, t]`:

    n1(t) = P(t),      n2(t) = N(t) - P(t) = C(t) - N(t)   for every t in (y, y^2].

Every prime member sits in a fragile slot (a pseudo-twin), every one of the `C` root
kills is load-bearing, and the number of doubles is pinned exactly - X leaves the
machine no slack whatsoever. First consequences, each exact:

* **C1 (global supply).** `P <= N`: the window must contain at least as many composite
  members as slots (`C >= N`).
* **C2 (prefix/run pigeonhole).** For every run `I` of consecutive slots inside the
  window: `P(I) <= N(I)`. Proof: under X each slot holds at most one prime; conversely
  more than `N(I)` primes in `N(I)` slots force a 2-prime slot, a twin, by L1.
  Equivalently: the prefix margin `N(t) - P(t) = n2(t) - n0(t)` never goes negative -
  *the doubles must stay ahead of the twins from the very bottom of the window*.
* **C3 (pseudo-twin ledger).** With `PT(q)` = number of `q`'s root kills whose slot
  partner is prime: `sum_q PT(q) = n1 = P - 2 n0`, so X demands `sum_q PT(q) = P` with
  `PT(q) <= R(q)` and `PT(top) = R(top) = 0` (horizon) - the gears strictly below `y`
  must deliver one prime-adjacent root kill for every prime of the window.
* **C4 (grading).** Everything localizes: a composite in `(y, t)` has root gear
  `<= sqrt(t)`, so the prefix demands of C2/C3 must be met by the `pi(sqrt(t)) - 2`
  gears active at depth `t` - the supply side of every prefix is a *small* machine.

**Sharpest single necessary condition (the round-1 statement):**

> **(C2, prefix form.) If some window `(y, y^2)` has no twin, then for every
> `t <= y^2` the members in `(y, t)` contain at least as many composites as primes:
> `P(t) <= N(t)`, i.e. the running margin `n2(t) - n0(t)` is `>= 0` at every prefix.**

Exact, non-asymptotic, and checkable slot by slot; its violation margin (the run
excess `max_I [P(I) - N(I)]`) measures how far a real window is from X.

## 4. Computation at y = 13, 23, 47

`uv run python research/constructor_ledger.py` output, condensed. Census and margins:

    y   N    P    C    n0   n1   n2   N-P   min prefix margin      max run excess
    13  25   32   18    9   14    2    -7   -7 at member ~109      +7 on members 17..109
    23  83   90   76   21   48   14    -7   -9 at member ~283      +9 on members 29..283
    47 359  313  405   61  191  107   +46   -7 at member ~283      +7 on members 53..283

Per-gear supply (exact `R(q)` vs square+coprime formula; `PT(q)` = prime-adjacent kills):

    y=13: 5: 10 (form 9, H=1) PT 9 | 7: 6 (6, 0) PT 5 | 11: 2 (2, 0) PT 0 | 13: 0 - horizon
    y=23: 5: 33 (25, H=8) PT 23 | 7: 19 (18,1) PT 11 | 11: 11 PT 5 | 13: 7 PT 5
          17: 4 (formula 5: the -1 is the boundary slot (527,529), 529 = 23^2, excluded
          by the strict window - not an error in the law) | 19: 2 PT 2 | 23: 0 - horizon
    y=47: 5: 144 (81, H=63) PT 78 | 7: 82 (62, H=20) PT 38 | 11: 46 PT 23 | 13: 35 PT 14
          17: 25 PT 10 | 19: 23 PT 10 | 23: 16 PT 6 | 29: 12 PT 5 | 31: 10 PT 4
          37: 6 PT 2 | 41: 4 PT 0 | 43: 2 PT 1 | 47: 0 - horizon

Pseudo-twin ledger: demand `P`, supply `sum PT = n1`, deficit `= 2 n0` exactly
(18 at y=13, 42 at y=23, 122 at y=47) - the deficit *is* twice the twin count, so C3 is
a restatement, useful as attribution (which gears carry the fragile load: overwhelmingly
5 and 7) rather than as an independent test.

**Which requirement fails hardest.**

* At `y = 13` and `y = 23`, **C1 already fails**: `P - N = +7` in both windows. There
  are not enough composite members to give every slot one. X is impossible outright at
  these scales by counting alone - no placement argument needed. (This is *not* the
  vacuous capacity sum of corpus 5.1: it compares composites to slots, not gear
  capacity to positions, and it genuinely bites - but only while prime density among
  `+-1 mod 6` members exceeds 1/2, i.e. roughly below `e^6 ~ 403`.)
* At `y = 47`, C1 passes with room to spare (`C - N = +46`; capacity abundant, exactly
  as the corpus warns) - **but C2 still fails by 7**: the run of 39 slots with members
  in `(53, 283)` carries 46 primes. Verified directly: primes in `[53,283]` = 46,
  slots fully inside = 39, excess = +7.
* In all three windows the violating run lives entirely **below member 283** - the
  bottom band. The first twin arrives at slot #1 or #2 of the window every time, while
  the first double cannot arrive before `k = 20` (`(119,121)` is the first
  double-composite slot in the whole integer line - every slot `k <= 19` has a prime
  member). The race C2 monitors (doubles vs twins from the bottom) is lost immediately
  in every real window: **the bottom band is the proof target.**

## 5. Attempts recorded with their limiting events (corpus discipline: what each yielded, and the exact event that limits it)

**5a. The run condition is an equivalence, not a shortcut.** A twin slot is itself a
run with excess +1, so: some run has `P(I) > N(I)` iff the window has a twin. C2's
value is not logical strength but *shape*: the gears have dropped out entirely - the
negation of X is a pure prime-clustering statement (some interval in `(y, y^2)` holds
more primes than slots), with a measurable margin. And no congruence obstruction can
block it: the 46-prime pattern of `[53, 283]` is admissible as a constellation (for
`p <= 43` its elements are primes `> p`, so residue `0 mod p` is unoccupied; for
`p >= 47` there are more residues than the 46 elements) - nothing mechanical forbids an
excess run from recurring above any height.

**5b. The pair-coincidence doubles bound - limited by an exact empty-intersection event.** Try to refute X
at large `y` by bounding the doubles supply: by L1 each double slot in a prefix is a
cross-member coincidence of a *distinct* gear pair `(q, r)`, occupying 2 CRT classes
mod `qr`, so in `L` consecutive slots with active gears `<= z`
(`g` of them):

    n2 <= L * s(z) + g(g-1),    s(z) = (sum 1/q)^2 - sum 1/q^2   over gears 5..z.

Under X this forces `P(band) >= L(1 - s(z)) - g(g-1)`. For the bound to be non-vacuous
we need `s(z) < 1`: computed, `s(127) = 0.959`, `s(139) = 1.005` - so `z <= 137`, hence
band top `t < 139^2 = 19321` and `ln(6L) < 9.87`. But a band that short can hold up to
`~ 2 * 6L / ln(6L)` primes (Brun-Titchmarsh), i.e. up to `12/ln(6L) >= 1.22` per slot -
already above the `1 - s(z) < 1` per slot that X is forced to supply. The two
requirements - non-vacuous coincidence bound (short band) and prime-thin band (long
band) - have empty intersection: the limiting event, exact. Same shape as corpus 5.2,
now recorded for the doubles side of the ledger too.

**5c. What survives.** The count layer of the ledger yields exactly one non-vacuous
weapon (C2), it kills X unconditionally in every window we can reach computationally,
and its guaranteed failure region is the bottom band. Past the density crossover
(`y > ~403`) the *average* no longer forces excess runs, and by 5a proving they always
exist is Reduction-A-strength. So the next layer must use placement in the bottom band,
where the machine serving the demand is small (C4: only `pi(sqrt(t)) - 2` gears), the
doubles onset is structurally delayed (no double before `k = 20`; deletion spacing
`(q+-1)/3` spreads each gear's kills), and X's forced census (`n1(t) = P(t)` from the
very first slot) is at its most brittle.

## 6. Proposed round 2 (two concrete chunks) - as steered by the manager, executed in sections 7-9

1. **Bottom-band double-onset law.** Under X the prefix census is forced from slot #1.
   Compute, across a ladder of `y`, the exact onset lag of the first double slot above
   `y` versus the first twin above `y`, and derive from deletion spacing + the CRT
   classes an exact lower bound on the fragile run that must open every window under X.
   Target statement: a window prefix of `L0(y)` slots in which X forces
   `P(prefix) = N(prefix)` (all fragile) while the active gears' kill spacing makes
   `n2 = 0` unavoidable there - then compare `L0` against unconditional prime-gap facts.
2. **The descent consequence (self-similarity).** X at `y` makes the band
   `(y, min(4y, y^2))` twin-free; that band is the *top* of the window of
   `y' ~ 2*sqrt(y)`. So X at `y` forces the sqrt-scale machine to sustain a twin-free
   run of `(y'^2 - y)/6` slots at its window top - of order `sqrt(y) * log y` slots,
   against measured max strides of order `log^3` (corpus stride table). Quantify the
   forced-vs-measured gap exactly down the ladder, and check whether the layer law
   (each layer's novelty is a short explicit list) can turn the descent into an
   induction: X at `y` implies a stride violation at scale `sqrt(y)` unless X-like
   scarcity already holds there.

---

# Constructor round 2: the double-onset law, its exact cap, and its closure

Script: `research/double_onset.py` (all assertions and tables below). The Mechanic's
round-2 prefix censuses were not yet posted at computation time; the onset census here is
self-contained and cheap (442 windows, y <= 3163).

## 7. The roots-of-unity law (Lateral's pinning, generalised to an iff)

**Law.** For any two distinct gears q, q', slot k is hit by both iff `36 k^2 = 1 (mod qq')`,
i.e. iff `6k` is a square root of unity mod qq'. The trivial roots `+-1` are the
same-member hits (`qq'` divides one member: the semiprime-multiple slots, the composite
root law's sites). The nontrivial roots `+-r`, `r = CRT(+1 mod q, -1 mod q')`, are the
cross-member hits. Hence:

> **A slot is double (both members composite) iff `6k` lands on a nontrivial root of
> unity mod qq' for some active pair `{q, q'}` (q | one member, q' | the other, both
> `<= sqrt(member)`).**

*Proof.* If q | 6k-1 and q' | 6k+1 then (6k)^2 = (6k-1)(6k+1)+1 = 1 mod qq', and
6k = +1 mod q, -1 mod q' is neither +1 nor -1 mod qq'; q = q' is impossible (slot cap,
L1). Conversely a nontrivial-root landing gives q, q' dividing opposite members, each
`< member`, so both members composite. Verified both directions on the full window of
y = 47 (359 slots). For twin-pair gears (p, p+2) the nontrivial root is r = p+1 and one
recovers Lateral's classes {+-u', +-u'(p+1)} exactly.

Consequences: the doubles' locations carry **no freedom at all** - they are one fixed
subset `D = {k : both 6k+-1 composite}` of the integers, the union of the pinned
`+-r_{qq'}` classes; a window merely slides along it. Prefix double censuses are
therefore computable by semiprime arithmetic alone, with no primality testing (handed
to the Mechanic).

## 8. The double-onset law and its unconditional cap

Define `k_start(y)` = first window slot, `k1(y)` = first double slot at or above
`k_start`, and the **onset lag** `L0(y) = k1 - k_start`.

**Onset law.** Unconditionally, `n2 = 0` on the first `L0(y)` slots of the window (no
gear pair can place a double there - the nearest pinned landing is at `k1`). Hence
**under X the onset prefix is perfectly fragile: every one of its `L0` slots holds
exactly one prime and one composite** - `P(prefix) = N(prefix) = L0`, primes at average
gap exactly 6 along `(y, y + 6 L0)`.

**Unconditional cap (Brun-Titchmarsh).** A run of `L` slots each containing a prime
holds `>= L` primes in an interval of length `6L + 2`; Montgomery-Vaughan's
`pi(x+H) - pi(x) < 2H / ln H` (all `x`, `H >= 2`) forces `ln(6L+2) <= 12 + 4/L`.
Computed exactly: **`L0(y) <= L* = 27129` for every `y`** (`6 L* + 2 = 162776`,
`e^12 = 162755`). This needs no hypothesis - X or not, no window anywhere begins with
more than 27129 prime-containing slots.

**Measured onset census** (all 442 windows, 13 <= y <= 3163):

    y      k_start  k1   L0   twins in onset prefix
    13        3     20   17    7
    23        5     20   15    6
    47        9     20   11    4        (k1 = 20 = slot (119,121), the first
    ...                                  double-composite slot in N)
    max L0 = 17 (at y = 13); L0 = 0 in 153/442 windows (window opens on a double);
    >= 1 twin strictly before the first double in 132/442 windows.

## 9. The onset route's limiting event, and the named missing fact

**What would contradict the forced alternation.** To kill X at onset scale one must
*prove* a twin inside the onset prefix, i.e. the unconditional fact

> `pi(y + H) - pi(y) >= H/6 + 1` with `H = 6 L0(y) + 2`  (equivalently, by the C2
> pigeonhole with `n2 = 0`: a twin in `(y, y + 6 L0(y) + 2)` for every `y`)

- a **superdense short-interval prime lower bound at density 1/6**, Hensley-Richards
strength. It is named here and *not assumed*. No unconditional theorem approaches it:
short-interval lower bounds (Heath-Brown, Baker-Harman-Pintz `x^0.525`) give density
`~ 1/ln y`, an order below `1/6` past `y ~ e^6`.

**And it is not merely unproven - as a universal statement at onset scale it is
FALSE.** In 310 of 442 real windows the onset prefix contains *no* twin: the perfect
prime/composite alternation X demands there is actually realised. The onset prefix is
therefore **consistent** with X, and no theorem can refute X inside it. The limiting
event, exact:

> The double-onset route cannot produce the contradiction by itself. `L0(y)` is capped
> unconditionally at 27129 and measured collapsing to 0 (max 17, at y = 13); the
> forced all-fragile prefix is realised in 70% of real windows. The contradiction must
> be **cumulative**: C2's margin over prefixes long enough to contain several onset
> events - which is exactly where round 1 found the real violations (runs ending at
> member ~283, well past the first double at k = 20).

What survives for the team: the roots-of-unity law (doubles = pinned semiprime
arithmetic, no primality tests needed - Mechanic), the absolute cap L* = 27129 (a
clean unconditional theorem of the programme), and the redirection of the contradiction
hunt from "first double" to "doubles deficit over cumulative prefixes".

## 10. The descent, stated exactly and stopped (one page, per manager's caution)

**Descent step (exact).** If W(y) is twin-free, then for every prime y' with
`sqrt(y) < y' < y`: every twin of W(y') lies in `(y', y]` - the window W(y') ends in a
terminal twin-free run of `(y'^2 - y)/6` slots, a fraction `1 - (y - y')/(y'^2 - y')`
of that window. Taking y' near `sqrt(2y)` this is the manager's ~1/2-window stride
event; taking y' = y^(2/3) it is all but a `y^(-1/3)` sliver. Iterating down the gear
ladder: X at y forces every layer band `(y'^2, y''^2)` (y'' = nextprime(y')) lying
above y to be twin-free.

**The unproven input, in one sentence:** *"every window W(y') contains a twin in its
top c-fraction for some fixed c < 1"* - which is Reduction A with a constant, i.e.
precisely the stride bound the programme lacks; as first sketched the descent re-derives
Reduction A at constant ~1/2 and proves nothing new. (Named, not assumed.)

**Where the layer law genuinely weakens the input.** The layer bands tile `(y, y^2)`,
so the induction does not need a twin in the top c-fraction of any *window* - it needs
only: **some single layer band `(y'^2, y''^2)` above y contains a twin.** A layer band
has length `y''^2 - y'^2 ~ 2 y' g(y')` (g = prime gap) - *layer scale, not window
scale* - and inside it, by the layer law + horizon theorem, twinhood is decided by the
gears `<= y'` except at an explicit list of at most a few semiprime slots. So the
input weakens from "twin in the top half of a quadratic window" to "twin in one
interval of length `~ 2 y' g(y')` sitting at its own machine's horizon" - bottom-band
scale, exactly where the team's proof target already lives. It remains bounded-gap
strength and unproven. Measured slack (band slots vs measured max stride
`0.47 ln^3/6`):

    y' = 97:    band 132 slots      stride ~60     ratio 2.2
    y' = 997:   band 4012 slots     stride ~206    ratio 19.4
    y' = 9973:  band 113220 slots   stride ~489    ratio 231.4

The forced twin-free run exceeds every measured stride by a growing factor - the input
is measured-true with widening slack and proven nowhere. Paused here: the limiting
event is the absence of any bounded-gap localisation theorem at exponent 1/2 (an
imported corpus limit, priced exactly in section 12), not a fact about the machine.

---

# Constructor round 3: the cumulative statement settled, and the layer-band route scoped

Scripts: `research/cumulative_margin.py` (full-window margin trajectories, this round);
Mechanic's `research/data/prefix_census.csv` consumed (their round-3 full-window CSV was
not yet posted; overlap at y = 101, 1009 reconciles exactly once their convention
"member = y counts as prime" is adjusted - their minMargin -5 at y=101 is this margin's
-4 plus the boundary member 101).

## 11. The cumulative statement, exactly - and the equivalence verdict

**Statement CUM.** *For every prime y >= 5, the window (y, y^2) contains a run I of
consecutive slots with P(I) > N(I) (strictly more prime members than slots).*

**Statement CUM_band** (sharpened by everything measured): *the run can be demanded
inside the bottom band (y, y + Delta) - every violating run found in rounds 1-3 lies
within 700 of y.*

CUM's truth refutes X at every y. But the verdict, proved both ways in two lines each:

* CUM(y) implies a slot of I holds 2 primes (pigeonhole; slot-cap lemma caps a slot at
  2), hence a twin in the window, hence not-X(y).
* A twin slot IS a run with excess +1. So E(y) := max over runs of [P(I) - N(I)] is
  >= 1 iff the window holds a twin.

> **Verdict: CUM is exactly equivalent to Reduction A. Not a strengthening, not a
> weakening, not a new statement - a lossless reparametrisation in which the gears
> drop out.** The "weakest known-unproven ingredient it needs" is itself: because the
> pigeonhole equivalence is lossless in both directions, there is NO ingredient
> strictly weaker than the conclusion. Anything implying CUM implies twin-in-every-
> window directly.

Placement against the corpus's other forms (docs/review-2026-08-17.md):

    form (b), h(L) >= d pointwise   STRICTLY STRONGER than needed (oversufficient;
                                    possibly false past y ~ 400 - review section 4)
    review's tail bound             sufficient, far weaker than (b), unproven;
    N(L) <= P exp(-cL/y), c > 6     lives on the sieve side (multiplicative route)
    CUM                             EXACTLY equivalent; lives on the prime side
    Reduction A                     the same point, sieve-side vocabulary

The two attack surfaces are pigeonhole-duals of one ledger: sieve side = bound the
blocked runs (dimension-2 Jacobsthal; parity floor beta_2 ~ 4.3-4.9 against the needed
exponent 2 - review section 6); prime side = superdense clustering (density 1/6 per
integer needed, 1/ln x available). The ledger transfers the problem between the two
surfaces at zero cost and zero gain.

**What the full-window margin data says** (new this round, y = 47..5003, full windows):

    y      N        P        twins   E(y)  realising run (members)   window fraction
    47     359      313      61      7     53..283                   0.003..0.109
    101    1682     1225     201     4     107..283                  0.001..0.018
    199    6566     4118     574     3     221..283                  0.0006..0.002
    503    42083    22186    2585    3     1277..1303                0.003
    1009   169511   79661    8278    3     1277..1303                0.0003
    2003   668333   283641   26870   3     2657..2713                0.0002
    5003   4170833  1567037  130543  3     5639..5659                0.00005

    min M(t): -7 (47), -4 (101), -1 (199), 0/-1 from 503 up, always within the first
    few slots (Mechanic: no negativity at t >= 5 for y >= 1e4, 125/125 windows).

E(y) collapses to a flat 3 and the realising runs shrink to 3-5-slot dense clusters a
bounded-looking distance above y (283, 1277-1303, 2657-2713, 5639-5659 - always within
~700 of y in these samples; measured, no law claimed). Honest reading: **as y grows the
pigeonhole surplus vanishes - CUM's measured margin over bare twin-existence is two
slots of excess and shrinking.** The cumulative form's content at scale is carried
entirely by small prime clusters just above y; it degenerates toward the twin statement
itself (trend observation, not a wall - it locates where CUM's content concentrates).
Its residual value is diagnostic (M(t), E(y) are computable violation meters)
and bibliographic (it lands the problem in the prime-cluster literature, where the
partial results are quantified) - not logical leverage. The genuinely open middle
ground remains the review's multiplicative tail bound, which CUM neither implies nor
needs.

## 12. Layer bands vs known bounded-gap theorems, scoped exactly (one page)

**The need.** The descent induction (section 10) wants the every-band form: there is
Y0 such that for every prime y' >= Y0 the layer band (y'^2, y''^2),
y'' = nextprime(y'), contains a twin. (The one-band-per-window form is equivalent to
Reduction A by the tiling - no gain; every-band is what buys a single height-uniform
theorem, and is a short-interval twin statement.)

**Band lengths** at height x = y'^2: typical 2*sqrt(x)*ln(sqrt(x)) = x^(1/2 + o(1));
thinnest - when (y', y'') is itself a twin pair, the recursion's self-reference -
exactly 4*sqrt(x) + 4. Even the THICKEST band available inside a given window is only
~ sqrt(x) * (largest prime gap below y), and every unconditional large-gap theorem
(Ford-Green-Konyagin-Maynard-Tao) is polylog - all bands have exponent 1/2 + o(1).

**What is proven, exactly.**

* Maynard-Tao: infinitely many consecutive-prime pairs differing by <= 246, with
  >> x/(log x)^K of them below x. Average spacing between bounded-gap pairs: polylog.
  **Density is ample** - surplus factor x^(1/2)/polylog against a band. Density is
  not what fails.
* Localisation: Alweiss-Luo (arXiv:1707.05437; Res. Number Theory 4:24, 2018): for
  every delta in [0.525, 1] there exist k, d such that for x sufficiently large,
  [x - x^delta, x] contains >> x^delta/(log x)^k pairs of consecutive primes
  differing by <= d. The floor delta = 0.525 is inherited from Baker-Harman-Pintz
  (2001): even ONE prime in [x - x^theta, x] is known for no theta < 0.525. Any
  improvement of the pair localisation below 0.525 would first improve single-prime
  localisation - Legendre-strength progress. (Imported corpus limit: a fact about
  published methods, not about the machine.)
* Computed curiosity: x^0.525 < 4*sqrt(x) until x = 4^40 ~ 1.2e24, so below 1e24 the
  Alweiss-Luo interval literally fits inside every band - but the theorem is
  asymptotic with an ineffective onset, so this yields nothing at accessible heights.
  The honest comparison is exponents: need 1/2 (+ o(1)), have 0.525.

**The tower of inputs, each with its limiting event:**

    T1  a PRIME in every layer band          OPEN. Implied by Legendre's conjecture
        (band contains the Legendre interval (y'^2, (y'+1)^2)); NOT implied by RH
        (RH gap bound O(sqrt(x) log x) exceeds the thin band 4*sqrt(x)); implied by
        Cramer. Limits the route before twins are even mentioned.
    T2  a pair with gap <= d in every band   above T1; proven localisation stops at
                                             exponent 0.525 (Alweiss-Luo); need 1/2.
    T3  a pair with gap exactly 2            the parity step on top: no bounded-gap
                                             theorem controls WHICH even difference
                                             occurs; 246 -> 2 is Zhang -> twin.

> **Scoping verdict: the layer-band route's unproven input is NOT "bounded-gap pairs
> recur" alone. It decomposes as (density: already proven, with room to spare) +
> (localisation: a Legendre-class open problem, exponent deficit 0.025 anchored at
> the BHP floor) + (parity: the full 246 -> 2 step, no partial result).** The thin
> bands - produced exactly when the sqrt-scale machine has a twin - are the binding
> case: the descent input meets its first limiting event at T1, "a prime between
> consecutive prime squares", before its twin content is even engaged. The 0.525
> floor is an imported corpus limit; the adjacent MACHINE event - thinnest bands
> occur exactly at twin endpoints, the self-reference sitting at the binding case -
> remains uninterrogated as a mechanism and is a candidate reopening.

Stop.

---

# Constructor round 4 (flagship, with Lateral): the X-consistency equation

Script: `research/x_consistency.py` (consumes Lateral's `research/split_gap_law.py`
closed forms; every identity asserted slot-by-slot at y = 101, 211, 503). Mechanic's
per-gear R_q(t) CSV was not yet posted at computation time; the attribution grading
below is by gear pair (Lateral's axis), the per-gear refinement composes with theirs
when it lands.

## 13. The equation, the overdetermination test, and the verdict

**The arithmetic census theorem** (unconditional; the equation's substrate). Classify
window slot k by gear marks (gear q marks a member it divides):

    type 0 (no mark)    <=> both members prime   (twin)      - horizon theorem
    type 1 (one mark)   <=> exactly one prime    (fragile)   - marked => composite
    type 2 (both marked)<=> both composite       (double)

So with d_k = [k is type 2], D(t) = sum d_k, and p_k = # prime members:

    P(t) = t - D(t) + n0(t)    for every prefix t          (identity, no hypothesis)

**THE X-CONSISTENCY EQUATION.** X(y) (zero twins in the window) holds iff

    P(t) = t - D(t)   for every t in [1, N]      iff      p_k + d_k = 1 at every slot.

Demand side: P(t), the prime census of (y, y^2). Supply side: D(t), freedom-free
arithmetic below y - by the roots-of-unity law the double set is the union of the
split classes +-x_{qq'} mod qq', and Lateral's gap law gives every representative in
closed form: x = (q'(b0 + iq) - 1)/6, b0 = (2 + m0 q)/g, m0 = (-2 q^{-1}) mod g,
i = the mod-6 alignment. **Prefix grading of the supply:** a pair contributes nothing
before its first landing; g = 2 pairs (twins below y) are the unique gap class with
m0 = 0, pinning at u' <= (y+1)/6 - anchored at the window's bottom edge,
unconditionally; every other gap class enters at depth ~ P/(6g) and only when the
alignment i = 0 lands (~51% of large pairs, measured by Lateral). So D(t) =
(guaranteed part, from the twins below y) + (alignment-rated part, from all other
prime pairs below y) - an explicit functional of the primes and prime gaps below y.

**Computed, real windows** (all assertions pass at every one of the N prefixes):

    y     N       P       D       n0     floor t-D   max rho    g=2 share of
                                         at t=N      (MV test)  split incidences
    101   1682    1225    658     201    1024        0.4687     8.9% (bottom band 8.9%)
    211   7384    4579    3431    626    3953        0.4785     7.0% (bottom band 6.4%)
    503   42083   22186   22482   2585   19601       0.4828     5.4% (bottom band 6.0%)

    violation profile n0(t) = P(t) - (t - D(t)): grows ~linearly from the first
    slots (y=503: 30 at t=200, 392 at t=4208, 2585 at t=N) - the equation fails
    everywhere, by exactly the twin count.

**The overdetermination test, answered exactly.**

* Degrees of freedom: ZERO. Both sides are determined by the same integers; X
  selects nothing. Formally the system is N ~ y^2/6 equations over the ~ pi(y) gap
  variables below y (the supply side's only inputs) - overdetermination ratio
  ~ y ln y / 6 - but the constraints are not independent tests against anything
  free: the census theorem makes the demand side ITSELF gear arithmetic (the horizon
  theorem is precisely the statement that primality above y = non-divisibility below
  y). Each slot equation p_k + d_k = 1 fails exactly on twin slots. The equation
  system collapses back to "n0(t) = 0 for all t" with no residual structure.
* What X needs P(t) to do: sit at its unconditional POINTWISE FLOOR t - D(t) at
  every one of N consecutive prefixes. Conflict from below: impossible - the floor
  is the unconditional minimum (P >= t - D is the identity with n0 >= 0 dropped).
  Conflict from above (Montgomery-Vaughan pi(y+H) - pi(y) <= 2H/lnH, H = 6t+2, fair
  game): the headroom ratio rho(t) = (t - D(t)) / (2H/lnH) is provably <= 1
  (rho <= P lnH/2H <= 1 is MV itself applied to the floor's majorant) and measured
  at max 0.47-0.48, drifting monotonically toward 1/2. **The forced floor sits at
  half the MV ceiling - the parity factor 2, photographed live in the ledger.** Any
  unconditional theorem separating P(t) from its floor (P(t) > t - D(t) for a single
  t) IS a twin-existence theorem, since the separation equals n0(t).

> **Verdict: the equation is satisfiable, for an exact reason.** The forced value is
> the unconditional pointwise minimum of the demand side, and every unconditional
> ceiling sits a parity factor above it (measured 0.48 -> 1/2). The
> overdetermination hope fails not for lack of constraints (there are y^2/6 of them
> per window) but because the census theorem makes the two sides one arithmetic:
> the prime census above y is not an external quantity X must luckily match - it IS
> the gear structure below y, evaluated upward. The residual, genuinely new content
> of the equation is the SUPPLY DECOMPOSITION: the doubles ledger X leans on has
> exactly one unconditionally guaranteed supply line - the g = 2 classes pinned at
> u' by the twins below y - measured at 5-9% of split incidences at every scale and
> depth tested, the rest alignment-conditional. The self-reference of the programme
> is now a measured line item in X's own budget: a twin-free window above requires,
> among its doubles, the pins manufactured by the twins below. It quantifies the
> recursion; it does not close it, because the alignment-rated 91-95% is empirically
> ample and nothing unconditional forbids it from covering the loss.

Where a count-level attack could still bite, stated for completeness and priced: an
improvement of the MV/Brun-Titchmarsh constant 2 toward 1 at H ~ y^2 would squeeze
the measured rho -> 1/2 against the ceiling; that constant's rigidity is itself
parity-class (improving Brun-Titchmarsh uniformly is known to force Siegel-zero
consequences - Motohashi's classical observation for the progression form). The
ledger and the analytic wall are the same wall, seen from opposite sides.

---

# Constructor round 5: the compression bound, the tool inventory, and the inversion zone

Scripts: `research/compression_bound.py`, `research/compression_zone.py` (no primality
tests anywhere - prime <=> unmarked, by the horizon theorem; every census identity
asserted). Mechanic's round-5 moment CSV was not yet posted; the moments here are
computed directly and match their round-4 S_pair/tau checkpoints where they overlap.

## 14. The compression statement, exactly

Per interior slot k let m_k = omega_L(k) * omega_R(k) (distinct-gear divisor counts of
the two members; m_k >= 1 iff the slot is double). Freedom-free totals (pure
arithmetic below y; M2 is the 4-tuple CRT co-hit count, floor arithmetic like S1):

    S1(t) = sum m_k   (the cross-root hit schedule; Mechanic's S_pair)
    M2(t) = sum m_k^2 (second moment)        n2(t) = #{m_k >= 1}

**The compression form of X.** X(y) <=> n2(t) = t - P(t) at every t <=> the fixed
hit schedule compresses into distinct slots at mean multiplicity

    M(t) = M_X(t) := S1(t) / (t - P(t))     exactly, at every prefix.

Reality delivers M_real(t) = S1(t)/n2(t); the identity n2 = (t-P) + n0 gives
M_X / M_real = 1 + n0/(t-P): X demands compression harder than reality by exactly
the twin share. **A contradiction requires an unconditional ceiling C(t) on
achievable mean multiplicity with C(t) < M_X(t) somewhere.**

## 15. The inventory: what each unconditional tool actually delivers (computed)

Measured at window ends (interior windows, all prefixes computed):

    y      M_real   M_X     need M_X/M_real   C_CS = M2/S1   C_CS/M_X
    211    2.505    3.063   1.223             3.859          1.260
    503    2.822    3.189   1.130             4.503          1.412
    2003   3.311    3.543   1.070             5.431          1.533
    5003   3.631    3.813   1.050             6.007          1.576

* **Union bound (Bonferroni-1):** n2 <= S1 - a floor M >= 1, never a ceiling; no
  forcing content.
* **Bonferroni-2:** n2 >= S1 - sum C(m_k,2) would give a ceiling - VACUOUS at every
  scale and every checkpoint tested (sum C(m_k,2) > S1 as soon as mean m > 3).
* **Cauchy-Schwarz / Turan second moment:** n2 >= S1^2/M2, ceiling C_CS = M2/S1,
  legitimately unconditional (M2 is freedom-free arithmetic). **The manager's
  expectation "the ceiling lands at exactly 2x the need" is settled negatively by
  computation - the measured behaviour is the opposite.** C_CS/M_X is 1.26 -> 1.58
  and GROWING (it tracks the m-distribution's dispersion <m^2>/<m>^2 ~
  lnln-divergent), while the window a winning ceiling must hit, (M_real, M_X),
  NARROWS as 1 + n0/(t-P) -> 1 (1.22 -> 1.05 in range). The two move apart on both
  ends; the second moment does not land at 2x - the ratio grows through the whole
  measured range (trend observation; it locates the missing structure in the
  m-dispersion, not in the mean).
* **Large sieve / Montgomery-Vaughan:** on this class system the large sieve is the
  translation-averaged second moment - same content as C_CS; its scalar photograph
  is round 4's rho -> 1/2. Selberg's Lambda^2 gives upper bounds on n0 (factor ~4
  above HL truth) - the WRONG direction against X, which asserts n0 = 0; no upper
  bound on n0 contradicts 0.

**The inversion zone (new, and the round's sharpest finding).** In the bottom band
the CS bound EXCEEDS X's demand: define R(t) = (S1^2/M2)/(t - P). Wherever
R(t) > 1, moment arithmetic alone forces n2 > t - P, i.e. n0 > 0 - X refuted with
no twin exhibited. Measured: the zone {t : R(t) > 1} is NONEMPTY at every y tested:

    y       sup R    at t   zone extent      closing P-bound needed at argmax
    101     19.64      99   [75, 1319]       0.802/slot  (0.134/integer)
    211     15.08      75   [22, 1862]       0.799/slot  (0.133/integer)
    503      6.55      26   [4, 2786]        0.748/slot  (0.125/integer)
    1009     5.83      21   [12, 4367]       0.722/slot  (0.120/integer)
    2003     2.90      23   [5, 6546]        0.622/slot  (0.104/integer)
    5003     2.91       5   [5, 11291]       0.418/slot  (0.070/integer)
    10007    1.44      22   [5, 17204]       0.476/slot  (0.079/integer)

Worked instance (y=503, t=4): prefix moments S1=3, M2=5 give n2 >= 9/5 > t-P = 1 -
the twin (521,523) forced by arithmetic plus the prime count, not found by search.
**Where parity re-enters, precisely:** (i) turning the zone into a theorem needs
P(t) > t - S1^2/M2 - a short-prefix prime LOWER bound at density 0.42-0.80/slot
(0.07-0.13/integer), superdense class; nothing unconditional is published below the
0.525 exponent (imported corpus limit), and no short-interval lower bound approaches
0.07/integer. (ii) the
ceilings themselves: any bound using only sieve-axiom moments cannot dip below what
parity-twisted configurations achieve on the same moments; measured overshoot
26-58% and widening against a needed margin of 5-22% and narrowing. sup R declines
toward 1 as the bottom band's prime density thins - the zone is real, persistent
through y = 10007, and asymptotically it degenerates into the early-twin detector
(at the immediate bottom m is concentrated near 1, CS efficiency -> 1, and R > 1
becomes equivalent to n0 >= 1).

## 16. The possible edge (one paragraph)

Three features of this class system are invisible to generic large-sieve/moment
axioms. (a) **Freedom-free placement**: the class representatives are Bezout-pinned
(gap law), and the bottom band is structurally n2-starved - no double before
absolute slot 20, and the guaranteed g=2 supply lands prime-membered (U-pins),
subtracted from n2 exactly where X's demand starts. This starvation is what CREATES
the inversion zone; a generic system with the same moments would not have it.
(b) **Mirror symmetry**: classes come in +- pairs, so prefix counts satisfy
palindromic complement identities within each period - a positional constraint the
translation-averaged large sieve cannot see. (c) **All-order exactness**: the master
formula computes every moment, odd orders and signs included, beyond sieve axioms -
but used in full it reproduces n2 exactly and returns to the tautology. The honest
edge is therefore the strip between order-2 moments and full exactness, applied
inside the zone where the system is provably non-generic: positional (mirror-aware)
third-moment bounds on the starved bottom band, seeking R > 1 forced by arithmetic
with a sub-superdense prime input. That is the one unexhausted direction this
inventory leaves standing; everything else terminates at the parity wall by
measurement, not metaphor.

---

# Constructor round 6: the zone's fate, and the third-moment front opened, computed, and located at its limiting events

Scripts: `research/zone_fate.py` (bottom-band ladder scan to y = 10^7 + LP moment
ceilings). Mechanic's round-5 moment data consumed (multiplicity_summary.csv /
hist.csv - their "the X-gap is zeroth-moment only" verdict is corroborated
independently below by direct LP computation at orders 2 and 3).

## 17. The fate of the inversion zone

R(t) = (S1^2/M2)/(t-P) decomposes exactly as R = eff * boost: eff = (S1^2/M2)/n2
(Cauchy-Schwarz efficiency, eroded by the m-dispersion) and boost = n2/(t-P) =
1 + n0/(t-P) (the twin surplus, the zone's only fuel). Ladder (T = 50000 band,
empties re-confirmed at T = 200000):

    y          sup R    at t     eff     boost    zone
    10007      1.442      22    0.962    1.500    [5, 17204]
    20011      1.923       4    0.962    2.000    [4, 24886]
    50021      2.000       5    1.000    2.000    [5, 40496]
    100003     1.032     416    0.919    1.123    [25, 39858]
    200003     1.010    1636    0.910    1.111    [726, 6216]
    500009     1.056      38    0.864    1.222    [28, 49]
    1000003    1.020     153    0.902    1.131    [103, 193]
    2000003    1.031      13    0.928    1.111    [13, 71]
    3000017    1.021      60    0.943    1.083    [19, 67]
    5000011    1.000       -       -        -     EMPTY (T = 200000)
    7000003    1.019      21    0.934    1.091    [21, 24]
    10000019   1.000       -       -        -     EMPTY (T = 200000)

**Answers to the mandate.** (i) sup R does NOT cross 1 at a single y - the zone's
generic forcing fades by trend (shrinking extents, sup -> 1+; observation), with
the first empty windows at the specific events y = 5000011 and 10000019 - and it
revives sporadically. (ii) The killer is the
P-side/boost: the twin surplus n0/(t-P) collapses like the bottom-band twin share
(~1/ln^2 y: boost at argmax 2.00 -> 1.08-1.13 down the ladder), while eff erodes
slowly (0.96 -> 0.86-0.94, the lnln m-dispersion). The zone needs boost > 1/eff
and loses when the first bottom twin arrives after ~10 doubles have accumulated.
(iii) Mirror-restricted prefix pairs change nothing - proof in section 18.

**The revival law, and the adversarial conclusion the manager ordered.** Windows
that open with a twin in their first few slots revive the zone at ANY y: found and
verified at y = 5000087, 5000101, 5000539 (twin within <= 4 slots; sup R = 1.923,
eff 0.962, boost 2.0 - the (n2, demand) = (2, 1) pattern). But this is a
self-reference, stated exactly: every twin (p, p+2) lies in the first slots of the
window of any prime y just below p, so **"the zone revives for infinitely many y"
is equivalent to the twin prime conjecture**, and a revival's fuel IS a bottom
twin. Maximum-skepticism verdict, as demanded: no certificate "R(t*) > 1 for all
y" can exist short of the conjecture itself - the inversion zone is a DETECTOR of
bottom twins (it certifies them from moments + P without exhibiting the pair),
never a generator. That equivalence is the route's limiting event, and also its
yield: the conjecture now has an ADDRESS - an exact, floor-arithmetic-checkable
restatement localized to each window's first slots.

## 18. The third-moment front: opened, computed, closed

**The mirror theorem (two lines, answering both chunks' mirror questions).** The
involution k -> -k (any period) maps members 6k+-1 to -(6k-+1), so it swaps
omega_L and omega_R and fixes m(k) = omega_L*omega_R. Every mirror-augmented
prefix-pair moment is therefore exactly twice the original, P doubles too, and
every ratio in this programme (R, eff, boost, M_X, every ceiling) is invariant.
**Mirror-awareness is vacuous at the moment level** - any edge from the mirror
must use positions jointly with signs, not moments of any order.

**LP moment-problem ceilings** (the sharp lower bound on n2 given S1, M2, M3 is a
small LP; solved by active-set enumeration, feasibility on the full range; the
arithmetic cap m <= (log_5 y^2)^2 is a theorem and is used where stated):

    scale                       C_CS    C2_int   C3(capped)  need M_X   true M_real
    y=2003  full window         5.431   5.387    5.241       3.543      3.311
    y=5003  full window         6.007   6.006    -           3.813      3.631
    y=10007 zone prefix 17204   5.793   5.761    5.726       5.793      5.043
    y=50021 band 50000          6.303   6.271    6.247       6.231      5.624

* The integer order-2 LP beats continuous CS by 0.3-0.5%: at the y=10007 zone edge
  (t = 17204, where CS exactly breaks even, R = 1) it still refutes: n2 >= 7744 >
  7702 = demand. A real but cosmetic extension of the zone.
* Order 3 with conservative tail handling adds NOTHING (LP2 = LP3 everywhere; the
  cubic never enters the optimal basis). With the legitimate arithmetic cap the
  cubic enters (basis (5, 6, cap)) and tightens a further 0.6-2.8% - at y=50021's
  band the bound reaches 25,093 vs demand 25,157: short by 64. At window scale the
  ceiling is 5.24 vs need 3.54 - the ~48% chasm is untouched.

**Verdict, honest both ways.** Third moments and mirror-awareness do NOT tighten
the ceiling toward M_X in any material way: the moment ladder converges to
exactness far too slowly, and the mirror contributes nothing by symmetry. This
corroborates Mechanic's round-5 finding by an independent route: the X-gap lives
entirely in the zeroth moment (the twin mass P(omega_L = omega_R = 0)), which no
power moment of m sees. The twin mass is a reporting metric - the exactness
scoreboard - not a mechanism; forcing must come from placement/pin/alternation-
level structure. The compression frontier at fixed moment order has reached its
limiting events (the mirror theorem; the measured LP convergence, <3% movement
against a 48% gap); what the zone analysis adds is that the POSITIONAL
bottom-band content - the one strip beyond moments this inventory left standing -
is exactly bottom-twin detection, i.e. the conjecture again, now with an address.
The count/moment yields are banked; the open mechanisms belong to the structural
workstreams.

---

# Constructor round 8: the multiplicative route - ratio data, tolerance theorem, wall verdict

Script: `research/multiplicative_route.py`. Inputs: the exact consecutive-machine
chain (adjacent frame) F(2,y) = 6, 15, 21, 33, 54, 75, 102, 129, 174, 264, 273,
309, 354 at y = 5..47 [corpus section 1; covering-bound-route sec 16], F(2,53) >=
420 partial; the merge recursion, chain condition, fuel censuses, saturation
theorem [gear-recursion.md, chain-conditions.md]; requirement F(2,y) < (y^2-y)/2
[gear-recursion sec 6].

## 19. Multiplicative accounting of stride growth

**19.1 The ratio data.** r = F'/F per consecutive step, against the window budget
(q'/q)^2 (the factor the requirement grows by):

    step     r=F'/F  budget   verdict   incr  incr/q   F'/requirement
    5->7     2.500   1.960    OVER        9   1.286    0.714
    7->11    1.400   2.469    under       6   0.545    0.382
    11->13   1.571   1.397    OVER       12   0.923    0.423
    13->17   1.636   1.710    under      21   1.235    0.397
    17->19   1.389   1.249    OVER       21   1.105    0.439
    19->23   1.360   1.465    under      27   1.174    0.403
    23->29   1.265   1.590    under      27   0.931    0.318
    29->31   1.349   1.143    OVER       45   1.452    0.374
    31->37   1.517   1.425    OVER       90   2.432    0.396
    37->41   1.034   1.228    under       9   0.220    0.333
    41->43   1.132   1.100    OVER       36   0.837    0.342
    43->47   1.146   1.195    under      45   0.957    0.328
    cumulative: sum ln r = 4.078 vs budget 4.481; margin ratio flat at 0.32-0.44.

Per-step multiplicative bounds of the exact budget shape r <= (q'/q)^2 are FALSE
at 6 of 12 steps - the same lumpiness that limited the additive 1.8q [corpus 6a],
seen multiplicatively. Uniform ratio caps r <= c > 1 cannot close, by exact
counting: pi(y) steps against a y^2 budget force the per-step geometric mean to 1. The only
viable shape is r <= 1 + alpha*q/F(M), i.e. the ADDITIVE per-step law
incr <= alpha*q - the multiplicative view's contribution is the bookkeeping that
shows which shapes can possibly close, not a new bound.

**19.2 The tolerance theorem (the round's result).** Corpus 6a closed per-step
increment bounds against the threshold 1.8 - but that threshold comes from the
odd-sum elementary step (sum q <= (y^2+2y-3)/4, all odd numbers). Gear-recursion
line 366 already noted the sharp prime sum gives "C/log y" and never drew the
per-step consequence. Drawn here, verified exactly:

> **THEOREM (conditional closure, any crude constant).** If the increment law
> F(M+q) - F(M) <= alpha*q holds at every consecutive-gear step with q > 47, for
> ANY fixed alpha <= alpha*(y)-scale - in particular alpha = 2.5 or 3 - then
> F(2,y) <= 354 + alpha*(S(y) - 328) < (y^2 - y)/2 for every prime y >= 53
> (S = prime sum). Checked exactly at every prime y in [53, 10^6]: zero failures,
> worst ratio 0.6557 at y = 113 (alpha = 3); beyond 10^6 by Rosser-Schoenfeld
> (S(y) < 1.25506 y^2/ln y, sufficient once ln y > 7.54). With y <= 47 known
> directly, this gives a survivor in every window - twins infinite.

The critical constant GROWS: alpha*(y) = [(y^2-y)/2 - 354]/[S(y) - 328] = 5.64 at
y = 101, 8.71 at 10^4, 13.3 at 10^6 - asymptotically ln y. So the corpus's
gear-37 refutation (2.432q) touches nothing here: 2.432 < 2.5, and the route
needs no sharp constant at all. **Per-step increment bounds DO deliver - corpus
6a's closure holds only for the elementary odd-sum chain.** The open link is now
a single mechanical statement: no consecutive step ever exceeds 2.5q (equivalently
3q, with more margin and a later finite base).

Decision point: F(2,53). alpha = 2.5 demands F(2,53) <= 486, alpha = 3 demands
<= 513, the window budget itself demands <= 450 eventually-per-step; the running
search sits at >= 420 (incr >= 66 = 1.245q so far, unfinished). The measurement
the review already ranked as decisive [review 7a] now also prices this route's
constant.

**19.3 The mechanism question: does the chain condition give an a-priori cap?**
Two regimes, sharply split:

* **Saturation regime (q - 1 > F(M)): YES, a theorem.** No two consecutive
  openings can both be deleted (deletion spacing >= q-1 adjacent), so
  F(M+q) = F2(M) [gear-recursion, proved], and incr = F2 - F <= F < q - the
  per-step law holds with alpha = 1, automatically. But along the CONSECUTIVE
  chain q is always the next prime and q < F(M) throughout (47 < 354): the
  compliant regime and the needed regime are disjoint. The saturation theorem
  covers far-gear additions, never the chain the conjecture walks.
* **In-range regime (q <= F(M)): NO a-priori cap.** The chain condition's raw
  cap is r <= k_max + 2 - a constant, exponentially over budget. What it does
  give is the exact decomposition incr = (F2 - F) + excess [corpus 5.4] with the
  two pieces separately measured O(q) ((F2-F)/y <= 1.24; excess/q <= 1.62, both
  at their separate maxima), and the FUEL GATE: excess > 0 requires a chain word
  ((s, q-s) alternation among consecutive openings) present in the current gap
  word - censused rare (11808/62/0 at gears<=19; k=4 first at gears<=29) but
  exploding at fuel-rich machines (70,964 pairs at gears<=31/q=37 - the gear-37
  spike, mechanism identified, unbounded). The needed lemmas are exactly:
  (a) top-gap anti-clustering: F2 - F <= alpha1*q (the best adjacent pair
  exceeds the max by O(q); the isolation law - minimal flanks at maximal gaps -
  is the empirical reason, but corpus's own correction notes F2 lives at medium
  gap pairs, so isolation alone does not prove it);
  (b) fuel-merge control: excess <= alpha2*q (chain-extending words near the
  top merge only O(q) of new span). Both are statements about the extreme upper
  tail of the gap word joint with residue alignment - corpus 5.5 proved gap
  structure alone CANNOT bound k in this regime; the missing input is the word
  arithmetic (forbidden-configurations machinery).

**19.4 Wall verdict, per the map's discipline.** The multiplicative route
genuinely EVADES all three mapped walls: not Wall I (no capacity comparison
anywhere - it bounds one extreme statistic's growth); not Wall II (no prime
lower bound appears in hypothesis or conclusion); not Wall IV (incr <= 2.5q is
strictly stronger than the conjecture - honestly lossy, as a route must be).
And it is NOT Wall III either, by the dimension-1 test: the same increment
statement for the ordinary one-residue Jacobsthal function would give
h(P(y)) << y^2/ln y, sharper than Iwaniec's theorem - unproven even where
parity does not obstruct (Iwaniec's dimension-1 bound exists; its increment
refinement does not). The obstruction is a FOURTH wall, distinct from the
map's three: **extreme-value control of sieve patterns** - the review's
"regime gap" (sec 5) and corpus 5.5's k-unboundability, now with an exact
target on it. Amendment filed to the attempts map.

**Standing verdict:** the multiplicative route is the programme's best-shaped
open statement: one mechanical hypothesis (no step exceeds 2.5q), any crude
constant suffices with tolerance growing like ln y, the finite verification is
done to 10^6 and closed beyond, all needed instruments (merge transform, chain
condition, fuel census, isolation, forbidden configurations) already exist, and
partial theorems already cover the saturation regime. It does not evade every
wall - it names a new one - but it is the only route on the books whose missing
lemma is a statement about the machine's own gap word rather than about primes.

---

# Constructor round 9: corridors vs lemma 1 (top-gap anti-clustering)

Script: `research/topgap_endpoint_law.py` (full k-frame periods, gears <= 11..23;
every law asserted at every recorded gap; F2 values independently reproduce the
corpus's F2(2,y) = 33, 48, 75, 93, 117 exactly). Filename note: my suite was
briefly at research/topgap_corridor.py; Lateral's complementary round-9
neighbourhood/streaming analysis now owns that name - mine moved. Frames:
k-frame below; adjacent/halved = 3 x k-frame.

## 20. What corridors do and do not give for F2 - F <= alpha1*q

**20.1 Two new corridor laws (proven, one line each; verified at every gap in
five full periods).** Openings are exposed slots, and exposed slots lie in the
15-residue set E mod 35 (gears 5, 7). Hence:

* **Endpoint law.** A machine gap of length G runs between openings a and a+G,
  so its left endpoint satisfies a mod 35 in A(G) = {r in E : r+G mod 35 in E}.
  |A| ranges 3..15 with shift; G = 34 mod 35 forces a in {3, 18, 33}. Measured
  concentration goes BEYOND the forcing: at gears<=23 (F = 34) the four record
  gaps sit at {3, 33} - two of the three allowed; at gears<=19 (F = 25, nine
  residues allowed) all twenty records sit at the SINGLE residue 5.
* **Adjacency law.** Adjacent gaps (G1, G2) force the opening chain a, a+G1,
  a+G1+G2 into E: allowed set A3(G1, G2). **294 of the 1225 length-pairs mod 35
  have A3 empty** - forbidden adjacencies from gears 5 and 7 alone (first
  examples (1,1), (1,3), (1,6), ...). Every observed F2-realising pair sits in
  its allowed set, at machines where the allowed set has as few as 2 residues.

Both laws are six_mul_class/card_class_Ico-shaped and kernel-checkable -
offered to Formalist/Harvester. Practical payoff: the F(2,53) covering search
can prune candidate record-gap endpoints by the endpoint law (factor 15/|A| =
2-5x depending on G mod 35; transfers to the adjacent frame mod 105).

**20.2 The decisive negative: residue laws cannot cap sizes.** Computed over
all 1225 pairs: **every (G1, G2) is within L1 distance 1 of a corridor-ALLOWED
pair** (escape distance = 1). A near-maximal gap has ~35 candidate lengths in
its range, so any residue exclusion is evaded by a +-1 slide in one component.
Corridor arithmetic constrains WHERE top-gap configurations sit, never HOW BIG
they are - at modulus 35 and, by the same argument, at any bounded modulus
(the exposed set's own max gap stays O(1), so escape distance stays O(1)).
No alpha1 follows from bounded-modulus arithmetic, however many corridor
levels are stacked.

**20.3 The corridor's only quantitative extension is local capacity - and it
dies on Wall I.** Refining the corridor to a density statement (base gears B
with exposed density rho; killers q in (B, y] supply 2*ceil(S/q) deletions per
span S) gives an exact cap: a fully covered span obeys
rho*S - 1 <= sum 2*ceil(S/q), so F2_k(y) <= (2#K + 1)/(rho - 2 sum 1/q) when
the margin is positive. Computed:

    base {5,7}   -> F2_k(11) <= 12   (actual 11 - TIGHT); F2_k(13) <= 54
                    (actual 16); y = 17: VACUOUS (2 sum 1/q = 0.453 > 3/7)
    base {5..17} -> F2_k(23) <= 72   (actual 39); F2_k(29) <= 11441 (near-
                    vacuous); y = 31: VACUOUS

A real little theorem family (the y = 11 cap is one off the truth), but the
margin rho - 2 sum 1/q dies two to three gears above ANY base - the two-scale
squeeze of corpus 5.2 in local form. Wall I, as the map predicts.

**20.4 The measured truth lemma 1 rests on (the extreme-value picture).**
Full-period record censuses: 4-20 maximal gaps per period (mirror-paired), and
the minimum separation between record gaps is **0.45-2.29% of the entire
primorial period** (851,695 slots at gears<=23) - near-maximal gaps are
astronomically anti-clustered in reality, five-plus orders beyond what the
lemma needs. F2 itself: at gears<=23 the realising pair is (F, 5) - the max
gap's own flank; alpha1-evidence along the chain (adjacent (F2-F)/q_next):
0.92, 0.88, 1.10, 0.78, 0.52 (gears<=11..23), corpus continuation 1.16 max at
add-31, 0.15 min at add-41 - bouncing, never above 1.2, no growth.

**20.5 Verdict (honest, per the mandate).** The corridor method does NOT give
alpha1. What it gives is exact residue geometry: any hypothetical lemma-1
violation must place two adjacent near-max gaps in an A3-allowed configuration
at endpoint-law-forced residues - a shrinking of the certificate and search
space (and a concrete pruning rule for F(2,53)), not a prohibition. The truth
of lemma 1 in the data is carried by near-max SCARCITY - record separations at
percent-of-primorial scale - which is exactly the extreme-value structure of
Wall V, beyond bounded-modulus arithmetic by the escape-distance argument and
beyond local capacity by 20.3. Lemma 1 needs genuine extreme-value input; the
corridor's role is to localise where that input must bite.

---

# Constructor round 10: top-stratum adjacency answered; lemma 2 stated, censused, and unified with lemma 1

Scripts: `research/strata_adjacency.py` (chunk 1), `research/merge_census.py`
(chunk 2; step 23->29 chunk-streamed over P = 1.078e9). All census anchors
verified: F_k(M+q') = 11/18/25/34/43 reproduced exactly; the 62 k=3 chains at
step 19->23 match the corpus fuel census number exactly.

## 21. Chunk 1: can two top-stratum classes mod 385 be adjacent? NO - and the
per-machine alpha1 certificate closes

**Lateral's live target, answered.** At every machine y = 13, 17, 19, 23 the
top stratum (maximal gaps) occupies 4-6 classes mod 385, and the class-level
adjacency test - is any r and r+F both a top-stratum left endpoint mod 385 -
returns EMPTY at all four machines. Two maximal gaps can never be adjacent,
certified by class arithmetic alone (given the address census, one period
scan each).

**The alpha1 finite check, three-tier structure** (dangerous pair (s1,s2) =
adjacent gap sizes with s1+s2 > F_k + alpha1*q'/3):

    machine  alpha1  dangerous  tier A      tier B         tier C   realized
                     pairs      (A3 empty)  (385-disjoint) residual
    13       1       14         5           5              4        NONE - closes
    17       4/3     38         10          26             2        NONE - closes
    19       4/3     78         11          63             4        NONE - closes
    23       4/3     230        56          78             96       NONE - closes

Proof-of-concept written out at y = 13, alpha1 = 1 (budget F2_k <= 16.67,
actual 16): the 14 dangerous pairs die as 5 by the machine-free A3 law, 5 by
mod-385 strata-class disjointness, 4 by direct verification ((7,11), (8,10)
and mirrors - class-compatible but unrealized). Honest trend: tier C grows
with the machine (4 -> 96 at y=23) because medium strata spread over more
classes mod 385 - the class tier needs the next corridor level (mod 5005) to
stay sharp at scale, and the uniformity-in-y question (the drift of pinned
addresses) remains exactly as Lateral posed it.

## 22. Chunk 2: lemma 2 - precise statement, census, and the spectrum reduction

**Precise statement (as the tolerance ledger uses it).** At consecutive step
M -> M+q': excess(M, q') := F(M+q') - F2(M); lemma 2 demands
excess <= alpha2*q'. Excess is positive iff a k >= 2 chain (k consecutive
M-openings all deleted by q') beats every k = 1 merge.

**Full merge census** (every chain in the full joint period, five steps):

    step      chains        k-hist                    excess_k  3*excess/q'
    11->13    264           {1: 258, 2: 6}            0         0.000
    13->17    2,897         {1: 2825, 2: 72}          2         0.353
    17->19    43,462        {1: 42374, 2: 1088}       0         0.000
    19->23    745,480       {1: 733672, 2: 11746,     3         0.391
                             3: 62}
    23->29    15,660,527    {1: 15416705, 2: 243822}  4         0.414

Argmax anatomies: interior gaps are LITERAL {2u', q'-2u'} at every step with
excess > 0 (13->17: 11 = 17-6; 19->23: (8,15) = (2u', q'-2u'); 23->29:
10 = 2u'); interior residues sit exactly on the teeth; and g_L + g_R <= F2 at
every argmax. The identity excess = interior_sum - (F2 - g_L - g_R) verified
at all five steps.

**The spectrum reduction (the round's structural result).** A k-chain's
merged gap is a sum of k+1 CONSECUTIVE old gaps. Define the consecutive-sum
spectrum F_j(M) = max sum of j consecutive gaps (F_1 = F, F_2 = F2). Then,
rigorously:

    F(M+q') <= F_{k_max+1}(M)     and     excess <= F_{k_max+1}(M) - F2(M),

k_max = the longest realized chain. Measured spectrum (j = 1..6):

    machine <=11:  7 11 16 18 23 26     increments 4 5 2 5 3
    machine <=13: 11 16 23 26 28 31     increments 5 7 3 2 3
    machine <=17: 18 25 28 33 35 40     increments 7 3 5 2 5
    machine <=19: 25 31 35 38 47 50     increments 6 4 3 9 3
    machine <=23: 34 39 50 58 65 77     increments 5 11 8 7 12

The bound is tight-ish (F_{k+1} - F2 = 5/7/3/7/11 vs actual excess
0/2/0/3/4) and the increments are q/3-SCALE, not F-scale: the best j-window
cannot extend by a large gap - the isolation law generalised to all depths,
measured.

**The unification.** Lemma 1 IS the first spectrum increment
(F2 - F = F_2 - F_1). The whole tolerance hypothesis now reads as ONE
statement about ONE object:

> increment(M -> q') <= F_{k_max+1}(M) - F(M) = sum of the first k_max
> spectrum increments. The multiplicative route closes if (i) the
> consecutive-sum spectrum is O(q')-flat to depth k_max+1, and (ii) k_max
> grows slower than ln y (fuel). Measured: k_max <= 3 at all five steps,
> increment sums 9-19 against budgets 2.5q' = 32-72.

**Verdict on corridor-reachability (the mandate's question).** Merges are
indeed local (span ~ k*q'), and the fuel half (k_max) is a bounded-window,
censusable object where forbidden-configuration machinery is native - genuinely
more approachable than record sizes. But the size half (spectrum flatness) is
the same extreme-value family as lemma 1 - the escape-distance obstruction
applies verbatim to spectrum increments (they are size statements). So lemma 2
splits: fuel = local, Wall-V-adjacent but with real partial tools (corpus 5.5
bounds it from gap structure NO, word arithmetic OPEN); flatness = Wall V
proper, unified with lemma 1. Net gain of the round: two lemmas have become
one spectrum-flatness statement plus one fuel bound, both censusable at scale
- the F_j spectrum is offered to the Mechanic as the object to track.

---

# Constructor round 11: the fuel bound - a literal cap theorem, a tail-run theorem, and the honest residue

Script: `research/fuel_bound.py` (+ the mod-210 verification run, logged in the
shared append). Consumed mid-round: Mechanic's k=4 event (step 29->31, unique
word class (10,21,10), 4 per period, mirror-paired; k_max by step = 2,2,3,2,4).

## 23. What limits chain length

**23.1 Theorem (tail-run cap, residue-free, one line).** Every qualifying
interior gap of a chain is = 0 or +-2c mod q' and positive, hence >= 2u' (the
smallest positive representative of +-3^{-1} mod q'). A k-chain's k-1 interior
gaps are therefore CONSECUTIVE gaps all >= 2u', so

    k_max(M, q') <= T(M, 2u') + 1,

T = the longest run of consecutive gaps >= 2u'. Measured T = 3, 2, 4, 3, 4, 5
across steps 11->13 .. 29->31; the cap is loose (realized k_max 2,2,3,2,4)
because chains also need residue alignment - which is exactly what 23.2 counts.

**23.2 THE LITERAL CAP THEOREM (exposure counting at modulus 35).** A literal
chain (interior spacings exactly the alternating {2u', q'-2u'}) has member
positions r, r+2u', r+q', r+q'+2u', ... - an interleaved walk of period 70
mod 35 that must stay inside the 15-residue exposed set E. The maximal run is
a function of q' mod 210 ONLY (q' mod 35 and u' = (q'+-1)/6 mod 35). Computed
exactly over all 48 invertible residue classes (verified as a class function
against every prime to 5000, zero mismatches):

    cap 2: 24 classes    cap 3: 4 classes    cap 4: 14 classes
    cap 6: 6 classes (q' = 37, 53, 83, 127, 157, 173 mod 210)

> **Literal chains have at most 6 members, for every gear, forever** - a
> 48-class finite check, kernel-checkable on the corridor machinery.

The cap EXPLAINS the realized selection: k_max by step = 2, 2, 3, 2, 4 at
gears with caps 2, 2, 4, 3, 4 - saturated at q' = 17, 19, 31; the k=4 event
sits exactly at a cap-4 gear, and the mandate's test case is answered:
**k=5 at q' = 31 is FORBIDDEN mod 35** (the word (10,21,10,21) needs 5 walk
members; cap(31) = 4). Prediction, falsifiable this round: the first literal
k = 5 or 6 can only occur at a cap-6 gear - and the running 31->37 census IS
at one (37 is a cap-6 class). Any extension beyond the cap requires a PADDED
link: a qualifying spacing outside the literal pair, hence >= q' - each
padded link consumes a gap >= q' ~ y, a doubly-tail object.

**23.3 The honest negative: fuel does not fold into flatness for free.** The
residue-free ceiling Q_{k+1} (max sum of k+1 consecutive gaps with middles
>= 2u'; increment <= max_k Q_{k+1} - F rigorously) EXCEEDS the 2.5q' budget
at 4 of 6 steps - always at the deepest windows (j ~ T+1):

    step      T  Q-F by depth (k-frame)          budget  realized incr
    11->13    3  +4 +9 +11 +13                   10.8    4   EXCEEDS
    13->17    2  +5 +7 +12                       14.2    7   within
    17->19    4  +7 +10 +13 +14 +16              15.8    7   EXCEEDS
    19->23    3  +6 +10 +12 +13                  19.2    9   within
    23->29    4  +5 +9 +16 +21 +26               24.2    9   EXCEEDS
    29->31    5  +12 +22 +25 +28 +28 +28         25.8    15  EXCEEDS

The deep windows exist in the gap word but are not realized (e.g. 29->31:
Q_5 = 71 = F+28 vs realized 58 = F+15) - the residue selection carries real
weight, factor ~2 at the top. So the certified ceiling must be WORD-INDEXED:
by 23.2 literal words are <= 6 long (at most a handful per step), so per step

    increment <= max over the <= 6 literal words of (word span + flank sum
                 at the word's occurrences) + the padded tier (gaps >= q').

**23.4 The exact obstruction, named.** What remains unproven is the control
of FLANK SUMS AT LITERAL-WORD OCCURRENCES: each word's occurrences are one
CRT class pair (pinned addresses, Lateral's law), and the question "how large
can the two gaps flanking an occurrence be" is gap-size adjacency at pinned
positions - Wall V, but now of BOUNDED COMPLEXITY (<= 6 words, 2 flanks,
pinned addresses) instead of an unbounded extreme-value statement. Growth
statement: k_max <= min(T+1, litcap + #padded), T drifts (renewal estimate
~ln^2 y, harmless: deep windows carry near-minimal sums - Q plateaus at
Q_5 = Q_6 = Q_7 at 29->31), padded links are priced at one >= q' gap each,
and the increment danger stays at depth <= litcap <= 6.

**23.5 Falsification criteria for the Mechanic's k_max census.**
(a) Any literal chain with more members than litcap(q' mod 210) - falsifies
23.2's table. (b) Any chain of any kind with k > T(M, 2u') + 1 - falsifies
23.1 (use as a census assert). (c) A literal k = 5 or 6 at a NON-cap-6 gear -
falsifies the class function. (d) Any realized chain containing a padded link
(interior spacing >= q') - would open the padded tier for the first time;
flag interior gaps >= q' explicitly. (e) The 31->37 census: literal k = 5 or
6 there is CONSISTENT (cap-6 gear); k = 7+ anywhere falsifies the absolute
cap.

---

# Constructor round 12: the word-indexed tolerance theorem - an identity, and it closes the budget

Scripts: `research/word_ceiling.py`, `research/flank_bound.py`. Data consumed:
Mechanic's `research/data/fuel_census.csv` (F2(29) = 55, F2(31) = 68, N_k by
step), spectra for machines 23/29; Lateral's pinning law and k=4 dissection.
Correction recorded: my first pass this round mis-indexed word flanks and
mis-modelled firing, inflating every tier; the numbers below are the corrected
run, and they are anchored - the formula reproduces all six known F(M+q').

## 24. The word-indexed formula

**24.1 Statement (an identity, not a ceiling).** At a consecutive step
M -> M+q' with u' = round(q'/6), a = 2u', b = q'-a, L = litcap(q' mod 210)
(round 11, <= 6), let W(q') be the alternating words in {a,b} of length
<= L-1 together with the padded words (some letter a multiple of q' or
= +-2c mod q' and >= q'). Call w COMPATIBLE if some tooth residue r in
{c, q'-c} has all partial sums r + (prefix of w) again in {c, q'-c}
(1-2 valid starts, computable from q' alone). Then

    F(M+q') = max( F2(M),  max over COMPATIBLE w in W(q') of
                            [ span(w) + FS_max(w; M) ] )

with FS_max(w; M) = max over occurrences of w in M's gap word of
(gap before + gap after).

*Why an identity.* Upper bound: any merge is a run of consecutive gaps whose
interiors are qualifying values - i.e. an occurrence of some w in W - plus its
two flanks. Lower bound: gcd(P_M, q') = 1, so the q' CRT copies of M's period
realize every residue shift; hence EVERY occurrence of a compatible word fires
in exactly |valid starts| of the copies, and its merge is realized somewhere.
Incompatible words never fire at all. Word list and compatibility come from
q' mod 210 alone; only occurrences and flanks come from M.

**24.2 Verification and the budget (the round's result).** All six measured
steps, k-frame:

    step      F    F2   C_lit (binding w)   C_pad   max = F(M+q')  incr  budget
    11->13    7    11   8   (4)             -       11  = 11       4     10.8
    13->17    11   16   18  (6)             -       18  = 18       7     14.2
    17->19    18   25   25  (13)            -       25  = 25       7     15.8
    19->23    25   31   34  (8,15)          31      34  = 34       9     19.2
    23->29    34   39   43  (10)            40      43  = 43       9     24.2
    29->31    43   55   58  (10)            49      58  = 58       15    25.8

The identity holds exactly at every step, and **every step is WITHIN the
2.5q' budget**. Round 11's residue-free Q-ceiling exceeded budget at 4 of 6
steps; word-indexing closes all four. The mechanism of the closure is visible:
the deep Q-windows (e.g. Q_5 = F+28 at 29->31) are sums over gaps whose
interiors merely EXCEED 2u'; the qualifying words require the interiors to
equal a or b exactly, and those occurrences sit among small flanks.

**24.3 The one missing bound, named and localised.** Tolerance needs, for
each compatible w,

    FS_max(w; M)  <=  F(M) + 2.5q'/3 - span(w).

Measured margins (allowance minus actual) are +7.2 to +21.2 at every word of
every step - a factor ~3 of room. The structural reason, computed: **no
top-stratum gap ever flanks a compatible-word occurrence** (counts L 0, R 0
at all 17 word-step pairs), and the largest single flank runs 0.43-0.81 of F.
So FS_max - F measures -6 to +5, i.e. FS_max <= F + 0.16q' in range against
the ~0.5q' allowance.

This is exactly round 10's adjacency question with a word in between: an
(ell+2)-point correlation where A3 was a 3-point one. The tier machinery
transfers verbatim (A-tier from gears 5,7; B-tier from mod-385 pinned
addresses; C-tier direct), and the trivial bound FS <= 2F is useless (gives
incr = O(F), not O(q')) - the content is precisely that the two flanks
cannot both be near-maximal at a pinned separation.

**24.4 The firing ratio does not strengthen the ceiling (honest answer).**
Firing is binary, not fractional: a word either has a valid tooth start (then
every one of its occurrences fires in |valid starts| of the q' copies) or it
has none (then it never fires anywhere). There is no surviving "fraction" to
multiply the ceiling by - which is precisely why 24.1 is an identity rather
than an inequality. What firing DOES explain is the decoupling Lateral
measured: at 29->31 the k=4 word (10,21,10) is compatible and does fire, but
its tier is 55 < 58 - the record is carried by the single-letter word (10)
with a big flank, not by the long chain. Fuel length and record growth are
separate channels, now with the reason: long words have small flanks.

**24.5 Route status.** Tolerance route = literal cap (proven, round 11) +
word-indexed identity (this round, verified at six steps) + flank-sum bound
(the sole open input, bounded complexity: <= 6 words per step, pinned
addresses, two flanks, ~3x measured margin). Falsification for the running
censuses: any step where max over compatible words of span + FS_max fails to
equal F(M+q') would break 24.1; any FS_max exceeding F + 2.5q'/3 - span would
break the budget at that step (31->37 is the next test: words (12,25,12) and
mirrors at a cap-6 gear, F = 58, budget incr 30.8).
