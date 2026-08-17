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

## 5. Closures (recorded in the corpus discipline: what fails, exactly why)

**5a. The run condition is an equivalence, not a shortcut.** A twin slot is itself a
run with excess +1, so: some run has `P(I) > N(I)` iff the window has a twin. C2's
value is not logical strength but *shape*: the gears have dropped out entirely - the
negation of X is a pure prime-clustering statement (some interval in `(y, y^2)` holds
more primes than slots), with a measurable margin. And no congruence obstruction can
block it: the 46-prime pattern of `[53, 283]` is admissible as a constellation (for
`p <= 43` its elements are primes `> p`, so residue `0 mod p` is unoccupied; for
`p >= 47` there are more residues than the 46 elements) - nothing mechanical forbids an
excess run from recurring above any height.

**5b. The pair-coincidence doubles bound closes - the squeeze, exact.** Try to refute X
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
band) - have empty intersection. Same shape as corpus 5.2, now closed for the doubles
side of the ledger too.

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

## 9. Closure of the onset route, and the named missing fact

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
therefore **consistent** with X, and no theorem can refute X inside it. Exact closure:

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
is measured-true with widening slack and proven nowhere. Stop.
