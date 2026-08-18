# lateral workstream log

## Round 1 (2026-08-18): does tooth-sharing at scale sqrt(N) do anything at scale N?

Premise taken seriously: twins at scale sqrt(N) are exactly the tooth-sharing gear
pairs of the machine deciding scale N (u'-column doubling, 17c/17d). Question: is
there a mechanism by which that sharing forces or obstructs twins at scale N?
Tool built: `research/tooth_sharing.py` (all claims below are its output).

### Mechanisms formulated

**M1 (kill-waste, positional).** In centred coordinates every gear's teeth are
+-u', so a twin pair (p, p+2) shares BOTH tooth values. Consequence (verified, 60/60
twin pairs to 2000): the pair's 4 within-pair double-kill CRT classes mod P = p(p+2)
are ALL pinned in closed form:

    { +u', -u', +u'(p+1), -u'(p+1) }  mod P

- +-u' are SPLIT kills (each gear kills a different member). k = u' is the pair's
  own slot (self-block); k = P - u' is its mirror, e.g. (269,271): k = 72854,
  6k-1 = 271*1613, 6k+1 = 269*1625.
- the mixed class is the **twin-product slot**: 6*u'(p+1) - 1 = (p+1)^2 - 1 = p(p+2)
  exactly. Both gears strike the SAME member, the semiprime p(p+2) itself.
- so at level y every twin gear pair donates >= 2 deterministic in-window wasted
  kills (own slot u' <= y/6 and product slot ((p+1)^2)/6 - ish <= (y^2)/6), where a
  generic pair of comparable gears expects only 4K/(pp') ~ 0.
- corollary (curiosity): when p(p+2)+2 is prime (p = 5, 149, 179, 239, 269, 419,
  569, 1289, 1319, ...), the twin pair *jointly* owns a pseudo-twin at its product
  slot - joint necessity in the sense of the necessity law.

**M2 (umbrella nesting).** Twin gears share u', so their short umbrellas nest
exactly around joint shields. Refuted-by-analysis as a separate mechanism: ANY two
gears' short umbrellas are concentric at joint shields (k = 0 mod qq'), with joint
width 2*min(u') - 1 - essentially the same for a twin pair and a matched generic
pair. The only twin-specific fact is that the *edges* coincide (4 teeth on 2 edge
slots), and those edge coincidences ARE the +-u' pinned classes of M1. M2 collapses
into M1; there is one mechanism, not two.

**M3 (counting identity).** u'-doubling as a count: #distinct first-strike slots =
#gears - T(y). Recorded as an identity; no force found in it this round (see the
conservation point below, which caps what any counting version can do).

### The quantitative law, and its test

Conservation first: over a full period the survivor count is prod(q-2) regardless
of phases - sharing can never change HOW MANY slots survive a period, only WHERE.
So any effect lives in sub-period windows, and the predicted law is

    E[waste_shared - waste_indep] = 1 - 2R/P,   R = K mod P

per pair in a window of length K (sign flips at R = P/2: pairs with period > 2K
waste MORE in-window when sharing, pairs with many periods inside average to 0).

**Test (part 2)**: 15 twin pairs (101..601), K = 60000, 400 random-phase draws per
configuration, comparing shared-phase (B) vs independent-phase (C) synthetic gears
(same moduli, same tooth counts - zero density confound). Result: measured B-C
matches 1 - 2R/P pair by pair, including every predicted sign flip (e.g. (239,241):
measured +0.907, predicted +0.917; (179,181): measured -0.692, predicted -0.704).
Totals: measured +2.03 vs predicted +1.91. **Law confirmed.**

**Test (part 3)**: full 20-gear set (10 twin pairs 269..601), real teeth (A) vs 300
draws each of B and C, global metrics over [1, 60000]:

| metric     | A (real) | E[B] shared | E[C] indep | B-C (se)      | z of A in C |
|------------|----------|-------------|------------|---------------|-------------|
| overcount  | 335      | 288.25      | 287.02     | +1.23 (0.64)  | +6.1        |
| survivors  | 54208    | 54162       | 54161      | +1.30 (0.65)  | +5.9        |
| maxstride  | 5        | 5.36        | 5.34       | +0.02 (0.05)  | -0.6        |
| lone       | 5465     | 5558        | 5560       | -2.37 (1.31)  | -5.9        |

- sharing per se (B-C) moves survivors by ~ +1.3 per 10 twin pairs - consistent
  with sum(1 - 2R/P) = +1.59 for these pairs, i.e. exactly the law, nothing more.
- **max stride is untouched** (B-C = +0.02 +- 0.05; A sits at z = -0.6). The
  coverage quantity Reduction A needs is insensitive to tooth-sharing.
- A's overcount z = +6.1 above random phases is NOT twin-specific: for real gears
  every cross-pair coincidence class is pinned at slots of the semiprimes q*q'
  (the twin-product pinning generalises to all gear pairs), and all products
  <= 6K+1 land in-window deterministically. The real machine is a maximally
  "semiprime-aligned" phase configuration; randomising phases removes that.
- lone-killer (fragile proxy) is correspondingly depressed in the real machine
  (z = -5.9), mostly by the cross-pair semiprime pinning, only -2.4 of it by
  twin-sharing.

**Test (part 4, the literal experiment)**: RICH = the 20 real twin gears vs POOR =
20 matched non-twin primes, low window + 100 random windows. Every difference
(survivors -104, lone +99, overcount +10) decomposes into the kill-density
mismatch (sum 2/q: 0.1021 vs 0.1001) - the matched-real-primes design is
confound-dominated and shows nothing the synthetic design doesn't show cleanly.
Max stride: mean diff +0.08 +- 0.07 - null within error.

### Verdict (exact, with yields)

1. **The counting route through tooth-sharing is limited by an exact counting
   event.** The effect is real, exactly quantified, and orders too small: net survivor gain from sharing is
   O(T(y)) per window against a needed ~K/log^2. Worse, in the REAL machine the
   two guaranteed wasted kills per twin pair land on slots that are already
   decided: the pair's own slot (already excised by self-blocking, the -T(y) term
   of the window identity) and the twin-product slot (composite by construction).
   The waste buys no new open slot at all. M1 cannot close the recursion by count.
2. **Tooth-sharing does not move max stride.** Measured dead flat. Any recursion
   closure via sharing must find a different observable; stride is not it.
3. What survives is **positional and exact**: the feedback loop "twins at sqrt(N)
   are structure at N" now has a closed form - each level-sqrt(N) twin pair marks
   the level-N window at exactly two deterministic slots (its own slot and its
   product slot), and contributes its 4 pinned classes {+-u', +-u'(p+1)} mod P
   forever after. The self-reference is real but (so far) inert as a lower bound.

### Proposed next chunk

The z = +6 anomaly is the interesting residue: the real machine's phase vector
(all teeth at +-round(q/6)) is a highly non-generic point of the phase space -
overcount far above random, fragile count far below, yet max stride dead average.
Lateral question worth one experiment: is the real phase vector *extremal* for any
window observable (min/max stride, fragile count, survivor clustering) over the
space of phase choices, or merely non-generic? If the machine's own phases
minimise some coverage functional, that is a variational handle the straight-ahead
workstreams don't have. Secondary: the joint-necessity slots (p(p+2)+2 prime)
connect twin gears to the necessity law - census whether twin pairs are jointly
necessary more or less often than generic pairs.

## Round 2 (2026-08-18): the anomaly closed into identities; extremality refuted

Steering taken: close the z = +6.1 / z = -5.9 anomaly before any grand probe.
Tool: `research/overcount_census.py` (imports round-1 tooth_sharing.py). Setting as
round 1: 20 gears (10 twin pairs 269..601), window [1, K], K = 60000, V = 6K+1.

### A. The anomaly is a theorem (two exact formulas)

**Real side - overcount and lone are a pure divisor census.** With
cnt(k) = omega_G(6k-1) + omega_G(6k+1) (omega_G = # gears dividing a member):

    overcount = SAME + B
    SAME = sum over members v <= V of (omega_G(v) - 1)   [semiprime census]
    B    = # slots with BOTH members gearful             [split census]

Checked against the window array: marks 6127, overcount 335, survivors 54208,
lone 5465 - **all four EXACT matches** (identity, not estimate). Decomposition:

- SAME = 190 = exactly the number of gear pairs {q,q'}: every product qq' <= 6K+1
  = 360001 (max 599*601 = 359999 - the window is precisely big enough), each lands
  exactly once as a member (multiples 2..4 of qq' are never == +-1 mod 6; 5qq' > V).
  Triple products >> V, so no higher terms. Fully deterministic.
- B = 145 split slots, of which 10 are the twin own-slots u'(p) (the round-1 pins);
  the other 135 sit at the Bezout representatives of q'b - qa = +-2, position
  governed by the pair gap (gap 2 -> pinned at u'; larger gaps -> scattered).
- kill-multiplicity distribution over killed slots: {1: 5465, 2: 319, 3: 8}.

**Random side - closed form, no simulation.** For phases v uniform on
[1,(q-1)/2] independent: exactly one v hits any k with q not dividing k, so
P(q kills k) = 2/(q-1), independent across gears. Hence exact expectations
(k <= K < any qq', so k has at most one gear divisor - shield handling exact):

    E[marks]     = sum_q (K - floor(K/q)) * 2/(q-1)              = 6126.22
    E[distinct]  = sum_k 1 - prod_{q not|k} (1 - 2/(q-1))         (formula)
    E[overcount] = E[marks] - E[distinct]                        = 287.13
    E[lone]      = sum_k sum_q p_q prod_{q'!=q} (1-p_q')         = 5560.57

Monte Carlo (200 draws): all four metrics agree with the formulas at |z| < 1.

**The anomaly, closed:** real - E[random] = 335 - 287.13 = **+47.87** overcount
and 5465 - 5560.57 = **-95.57** lone. Both are now differences of formulas. The
lone deficit closes by the same accounting, as predicted: Delta_lone =
Delta_distinct - Delta_multi = -47.09 - 48.48 = -95.57 - the ~48 deterministic
coincidences (190 semiprimes + 145 Bezout splits vs ~287 expected) are counted
once as lost distinct slots and once as gained multi slots. One cause, two faces,
zero mystery. (E[marks] = real marks to 0.9: the 2/(q-1) formula collapses to
2K/q, so the phase randomisation preserves supply exactly - the anomaly was
always pure position.)

### B. Extremality: refuted by exact enumeration

Full phase-space enumeration (every configuration, no sampling), real vector
ranked on overcount (want argmax), lone (argmin), survivors, max stride:

| machine | space | configs | overcount | lone | survivors | maxstride |
|---|---|---|---|---|---|---|
| {5,7}, K=8 | mirror +-v | 6 | **ARGMAX** | **ARGMIN** | **ARGMAX** | rank 5/6 |
| {5,7}, K=8 | full 2-teeth | 210 | rank 3 | rank 14 | rank 20 | rank 153 |
| {5,7,11}, K=20 | mirror | 30 | rank 3 | rank 5 | rank 5 | rank 9 |
| {5,7,11}, K=20 | full | 11550 | rank 1716 | rank 2536 | rank 3886 | rank 2748 |
| {5,7,11,13}, K=28 | mirror | 180 | rank 18 | rank 20 | rank 12 | rank 119 |

Window sweep ({5,7,11}, mirror, K = 10..40): real is argmax overcount at K = 10
and K = 35 only; at the machine's own window K = 20 it is 5 vs max 6.

**Verdict: the real phase vector is merely high (top 10-25% on waste metrics),
never extremal beyond the degenerate 2-gear mirror space.** There is no
variational characterisation of the machine's phases to exploit; the "special
point of phase space" language from round 1 should be read as "the census is
deterministic", nothing stronger. Grand extremality probe cancelled - it would
have found nothing the census identities don't already say.

### Proposed next chunk

The only non-formula ingredient left in the overcount is B's positions: split
slots sit at Bezout representatives of q'b - qa = +-2, and gap 2 pins them at u'
(round 1). Candidate law: a gap-graded closed form for split-slot positions
(gap 4, gap 6, ...), which would make in-window overcount a complete formula at
any scale - and connects the GEAR GAP DISTRIBUTION to coverage, i.e. the prime
gaps at scale sqrt(N) feeding the ledger at scale N through the machine's own
laws. Alternatively, redirect to support the constructor's bottom-band target:
the 8 triple-kill slots and 145 split slots all have exact addresses now - check
where they sit relative to the bottom band.

## Round 3 (2026-08-18): the gap-graded split law, and the complete overcount formula

Steering taken: derive where a gap-g gear pair's split slots sit, in closed form;
assemble and test the complete overcount formula; state the payoff for the
Constructor. Tool: `research/split_gap_law.py`. All checks exact.

### The law

For gears q < q' = q + g (g even), the split class "q kills left, q' kills right"
(q | 6k-1, q' | 6k+1) solves q'b - qa = 2, member 6k+1 = q'b. With q' = q + g and
t = a - b this is gb - qt = 2, so b = 2 g^{-1} (mod q), and the least
representative is pure arithmetic:

    m0 = (-2 * q^{-1}) mod g          (depends only on q mod g)
    b0 = (2 + m0 q) / g               (exactly integral)
    i  = (q' - b0) * q^{-1} mod 6     (mod-6 alignment, i in 0..5)
    x  = (q' (b0 + i q) - 1) / 6      (least k; the other class at P - x, P = qq')

Verified against brute CRT for ALL 2850 prime pairs 5 <= q < q' <= 400, zero
failures, mirror class = P - x always. In the SUMMARY's language: x is the
nontrivial square root of 1 mod qq' (36x^2 = 1, 6x = +1 mod q, -1 mod q'),
now in closed form.

**Depth gradation.** x ~ P(m0/g + i)/6. Since m0 = 0 iff g | 2, **g = 2 is the
unique gap with b0 = 1 identically**: its split rep is x = u' = round(q/6), depth
~P/(6q) - the twin pin, inside every window at every scale, unconditionally.
Every other gap has b0 >= (2+q)/g, so its lowest possible split depth is ~P/(6g),
and even that is reached only when the alignment lands (i = 0). Examples:
(101,103) x/P = 0.0016; (97,101) g=4: 0.7508; (101,107) g=6: 0.8894;
(101,113) g=12: 0.0280 (aligned) vs (89,101) g=12: 0.6947 (not).

### The complete overcount formula

    overcount = SAME + PAIRSPLIT - CORR
    SAME      = sum_{j>=2} (-1)^j sum_{squarefree products of j gears <= V}
                mult(product)                       [pure floor counting]
    PAIRSPLIT = sum over gear pairs of in-window hits of the two law classes
                                                    [pure law + floor, no CRT]
    CORR      = sum over both-members-gearful slots of (omega_l*omega_r - 1)
                                                    [multi-gear-side overlap]

Tested at three REAL scales (all gears 5..y, the machine's own window
K = (y^2-1)/6), each piece independently against the divisor census and the
total against the window array:

    y =  53: overcount = 250 + 296 - 147 = 399   == array 399   (exact)
    y = 101: overcount = 1157 + 1490 - 815 = 1832 == array 1832 (exact)
    y = 211: overcount = 6367 + 8651 - 5185 = 9833 == array 9833 (exact)

SAME formula == census and PAIRSPLIT law == census at every scale. Honest note:
CORR is NOT small (it grows with scale because small gears stack omega on
members); it is census-exact here, and mechanically expandable in the same
inclusion-exclusion framework ((s_L, s_R) product pairs, |s_L|+|s_R| >= 3) if a
100% floor-arithmetic formula is ever needed - deferred, not blocked.

### The payoff: the doubles ledger is a functional of the prime gaps below y

PAIRSPLIT = sum over pairs of F(g, q mod 6g; K) - the split-double supply of the
window is an explicit functional of the prime-pair difference structure at gear
scale. Gap dependence, measured (y=211): mean in-window splits per pair by gap:
g=2: 43.8, g=4: 24.8, g=6: 26.7, g>12: 6.8; hit rates 100% / 92% / 85% / 93%.
The clean claim is at the LARGEST pairs, where alignment bites (P > 3K):

    y=101: twins 1/1 = 100%   non-twin 21/33 = 63.6%
    y=211: twins 4/4 = 100%   non-twin 58/114 = 50.9%
    y=503: twins 3/3 = 100%   non-twin 274/539 = 50.8%

**Twin pairs at gear scale are the unique gap class whose contribution to the
window's double population is unconditionally guaranteed at every scale** (the
law forces x = u' <= K always); every other gap contributes at a residue-
alignment rate that decays toward ~1/2 for the largest pairs. This is the
round-1 self-reference back with numbers: T(y) guaranteed split doubles (plus
mirrors when they fit) flow into the level-y^2 ledger from the twins below y,
while the rest of the supply is conditional on alignment. For the Constructor:
the doubles' supply term in the cumulative statement decomposes as
(guaranteed, from twins below y) + (alignment-rated, from all other pairs),
both sides now computable by floor arithmetic per pair.

### Proposed next chunk

Two candidates, Constructor-serving: (1) formula-ize CORR (the higher product-
pair terms) so overcount is complete floor arithmetic at any scale; (2) aim the
law at the bottom band: the g=2 pins sit at u' <= y/6, i.e. the guaranteed
doubles live exactly in the bottom band the team has made the proof target -
derive the bottom-band double-onset supply (which pairs can place a split below
a given slot t) as an explicit finite list per window.

## Round 4 (2026-08-18): the master supply formula - exact at every prefix depth

Steering taken: formula-ize CORR, prefix-grade everything, answer the multiplicity
question. Tool: `research/supply_formula.py`. All checks are max-abs-diff over
EVERY prefix t in [1, K], not spot checks.

### The master formula (CORR formula-ized)

Using (c-1)[c>=1] = c - 1 + [c=0] and inclusion-exclusion on both members
simultaneously, the whole round-3 decomposition collapses into ONE signed sum
over coprime pairs of squarefree gear products (s_L | 6k-1, s_R | 6k+1), each
pair one CRT class mod s_L s_R, each count pure floor arithmetic:

    overcount(t) = sum_{|s_L|+|s_R| >= 2} (-1)^{#gears} N(s_L, s_R; t)

Taxonomy: one-sided terms = SAME; (q,q') single-single terms = PAIRSPLIT (the
gap-law classes); both-sided terms with >= 3 gears = -CORR. So CORR is now the
same floor arithmetic as everything else - round 3's census crutch is gone. The
both-sided restriction of the sum is

    B(t) = # slots <= t with both members gearful
         = sum_{s_L, s_R both nonempty} (-1)^{#gears} N(s_L, s_R; t).

### The Constructor's n2, exactly

In the window every composite member has a gear factor (horizon), and the only
PRIME gearful members are the gears themselves, sitting at their self-block
slots u'(q). Hence, with U(t) = #{u'(q) <= t : partner member gearful}:

    n2(t) = B(t) - U(t)          (distinct both-composite slots)
    overcount(t) = SAME(t) + U(t) + n2(t)     (the multiplicity bridge)

U is finite, explicit, and confined to the bottom y/6 slots (max u' = u'(y)).
This answers the multiplicity question exactly: n2 counts distinct slots, ALL
same-member stacking lives in SAME, all prime-member exceptions in U - no
approximation anywhere. Deep hubs are real at scale: kill-multiplicity spectrum
at y=211 is {1:1846, 2:2037, 3:1465, 4:1038, 5:294, 6:108, 7:6} (cnt up to 7),
and B/n2 absorb them correctly because the signed sum telescopes per slot.

### Verification, prefix granularity

    y=101 (K=1700):  2940 terms; max over ALL t of |formula - census|:
                     overcount 0, B 0, n2 0
    y=211 (K=7420): 17022 terms; max over ALL t: overcount 0, B 0, n2 0
    t=K components (y=211): SAME 6367, PAIRSPLIT 8651, CORR 5185, B 3466,
                     U 31, n2 3435; bridge 6367+31+3435 = 9833 = overcount. OK.

### The availability schedule (supply-arrival curve for the Constructor)

Bottom band of y=211, exact event list (excerpt): u' pins arrive first and
alone - t = 1 (5,7), 2 (11,13), 3 (17,19), 5 (29,31), 7 (41,43), 10, 12, 17,
18, 23, 25, 30, 32, 33 - all prime-member B slots (they feed U, not n2). First
SAME at t=6 (35 = 5*7). **First n2 slot at t=20 = (119,121)** - matching the
Constructor's measured "first double never before k=20" exactly, now with its
anatomy: 119 = 7*17, 121 = 11^2, B-contribution = split(7,11) + split(17,11)
- hub(119|11) = 2 - 1 = 1. Between t=1 and t=19 the ledger's entire double
supply is prime-membered (U-type): under X, demand n2(t) = N(t) - P(t) has NO
supply to draw on before t=20 at any y >= 211 - consistent with (and now
explaining) the Constructor's onset findings. Decile curves printed in the
tool; n2 grows near-linearly (~0.463 per slot at y=211) after onset.

### Honest caveats

- Term enumeration is O(#products^2) (4M candidate combos at y=211, 17022
  surviving). Fine to y ~ 500; beyond that needs pruning (e.g. meet-in-the-
  middle on product size). The FORMULA is scale-free; only enumeration costs.
- U's definition has boundary sensitivity (partner prime just above y is not
  gearful), handled exactly by the arithmetic; anyone reusing U should keep the
  "partner gearful", not "partner prime", test.

### Proposed next chunk

The flagship's supply side is now delivered: under X, N(t) - P(t) = B(t) - U(t)
at every t, left side pure prime census of (y, y^2), right side pure floor
arithmetic over primes/gaps below y. Offer: (1) overdetermination scan - measure
in real windows where the two sides' DERIVATIVES disagree (real windows have
slack; X has none; locate the slots where the rigidity binds); (2) close the
enumeration cost so the equation is testable at y ~ 10^4.

## Round 5 (2026-08-18): the machinery at y = 10^4, and the derivative scan

Steering taken: pruning to reach scale, then the per-slot binding map.
Tool: `research/derivative_scan.py` (numpy sieve + gap law + arithmetic U).

### Pruning that survived contact

The O(#products^2) term enumeration was only ever needed for the CORR-class
terms; the labour splits along what each piece is good at:
- PAIRSPLIT by the gap law directly: O(pi(y)^2) closed-form reps, no product
  enumeration. At-scale verification: law total == sieve incidence
  sum_k omega_l*omega_r EXACTLY - 13,861 pairs (301,026 incidences) at y=1009,
  753,378 pairs (43,908,326 incidences) at y=10007. The gap law is now
  verified at three orders of magnitude beyond its round-3 exhaustive range.
- U by pure arithmetic (u'(q), partner-gearful): == sieve at both scales
  (133 slots at y=1009, 1023 at y=10007).
- The telescoped remainder (SAME, B, hubs) by vectorized sieve, spot-verified
  against independent trial division (2000 random slots each scale, 0 miss).

### The reality form of the flagship identity, verified per-slot at 1.67e7 slots

With n_j = # slots with j composite members: P = 2n0 + n1, n2 = B - U,
n0 = T_win, hence the X-consistency equation's reality form is

    P(t) = t + T_win(t) - B(t) + U(t)    at every t
    (per-slot: dP = 1 + dT - dB + dU, max residual 0 at both scales)

**The binding defect of the flagship identity IS the twin count.** X <=> the
identity binds (T = 0); reality's deviation is exactly one unit per twin slot.
Totals (y=10007, K=16,690,008): B 11,362,820, U 1,023, n2 11,361,797,
T 440,870, P 5,769,081, overcount 38,821,888 (bridge OK); g=2 share of
PAIRSPLIT 3.1% (4.7% at y=1009).

### The derivative scan: geometry of the near-binding loci

Reality is exactly X-like on twin-free runs (dP = 1 - dn2 slot by slot there).
Scanning all runs (440,870 twins at y=10007; max stride 478, consistent with
the measured 0.47 log^3/6 law):

- **Prime load in the most X-like stretches: 87-90% of ambient.** Top-1% of
  strides, length-weighted, depth-binned baseline: P-rate/ambient = 0.869
  (y=1009), 0.901 (y=10007). Selection pushes this below 1 (fewer primes =
  fewer pairing chances = longer runs) - the lateral point is how SMALL the
  deficit is: reality's longest X-like stretches still carry ~90% of full
  prime load while pairing none of it. X must carry 100% (PNT-pinned) over the
  entire window. The compression frontier in one line: reality can do X-like
  behaviour at length ~478 while shedding 10% of its prime load; X needs it at
  length 1.7e7 while shedding none.
- **Hub ground is generic: hub-rate/ambient = 0.999 / 1.006.** The near-binding
  loci are NOT hub-enriched - X-likeness is not achieved by extra pile-up,
  consistent with the mechanic's capacity-never-binds verdict, now seen locally.
- **Geometry: the bottom band is stride-hostile.** Top-1% strides live at
  depths 0.06-0.99 (median ~0.6); max stride inside the first 1% of the window
  is half the global max (242 vs 478 at y=10007; 35 vs 242 at y=1009). The
  bottom band - where U lives and supply arrives late (round 4) - is exactly
  where reality never comes close to X-behaviour. A compression bound that
  binds only in the bottom band fights reality where reality is strongest.

### Honest caveats

- The P-rate deficit in long runs is partly conditioning (runs are found where
  primes are thin), not a mechanism; its informative content is its smallness
  and its trend UP with scale (0.869 -> 0.901): the X-likeness discount is
  shrinking, i.e. long runs look more like ambient ground as y grows.
- Depth-binned ambient (100 bins) is a crude baseline; a log-local baseline
  moved numbers by < 0.01 in spot checks.

### Proposed next chunk

The scan says the binding defect is T itself, so the compression bound must be
a statement about how much prime load a twin-free stretch can carry. Offer:
(1) the load-length frontier: empirical + exact-arithmetic curve of max prime
load vs run length at fixed depth (the quantitative object the bound must
dominate); (2) feed the Harvester: the per-slot identity P = t + T - B + U is
kernel-checkable bookkeeping (all four terms are census objects; no analysis),
a natural next Lean target coupling Census.lean to the supply side.

## Round 6 (2026-08-18): the load-length frontier

Steering taken: map max prime load vs twin-free run length against the X-ceiling
(load 1 per slot - the C2 pigeonhole ceiling: a twin-free slot carries at most
one prime member). Tool: `research/load_frontier.py`, open interior only
(slots with both members > y), scales y = 1009, 3163, 10007.

### The frontier curve, and its first surprise: it is ABSOLUTE

maxload(L) = max over twin-free L-windows of prime-members/L:

    L        1..13   14     16     20    25    32    50    100   200   478
    maxload  1.0000  .9286  .875   .85   .80   .7188 .60   .52   .43   .32

The frontier touches the X-ceiling exactly up to **L* = 13**, then the gap
opens by exactly one missing prime (13/14), and decays like a staircase of
fixed rationals (20/25, 23/32, 52/100...). The surprise: these values are
IDENTICAL across all three scales because the record-holders are the SAME
absolute integer landmarks - L* = 13 is achieved at slots 2452-2464 (members
14713..14783: primes 14713R 14717L 14723L 14731R 14737R 14741L 14747L 14753L
14759L 14767R 14771L 14779R 14783L - side word RLLRRLLLLRLRL, blocky, NOT
strictly alternating (see round 7: strict alternation caps at 6) - no twins)
at every y.
The L=100 record sits at absolute slot ~31,350 at every y. The frontier is a
property of the integers; the window only truncates it from below (s0 ~ y/6).

**Renewability check** (does the ceiling-touching survive when the landmarks
exit the window?): restricting run starts to depth >= 0.1 and >= 0.5 of the
window still gives saturated (load-1) runs of length 9-12 at every scale
(e.g. y=10007: L*=11 at members ~1.9e7, L*=10 at members ~5.1e7). So gap(L)=0
for L <= ~10 is renewable at ALL depths tested, and L <= 13 near the bottom.
Caveat, labelled: nothing known FORCES saturated runs to persist at all
depths forever - that persistence is itself a prime-constellation statement
(HL-admissible, so expected true; no published method reaches it - an imported
corpus limit, about methods, not about the machine).

### Where the gap is narrowest - the target scale

gap(L) = 0 for L <= 13; opens at L = 14 with 1/14 and stays < 0.29 through
L = 32. **The compression-bound target is L ~ 14-32**: reality hugs the
X-ceiling there, renewably, at every depth. For L >= 63 the gap exceeds 0.44 -
no leverage there: reality never approaches the ceiling, so a bound binding
only at long runs constrains nothing reality does.

### Bottom-band branch (for the inversion-zone push)

At y = 10007 the bottom band [s0, s0+y] = [1669, 11676] CONTAINS the global
record runs up to L ~ 100: the band frontier equals the global frontier at
small L (load 1 through L = 13 inside the band). Round 5 said the bottom band
is stride-HOSTILE (short runs); this round adds: it is load-OPTIMAL (the
ceiling-touching runs live there, where prime density ~6/ln y is highest).
The constructor's starved band is starved of length, not of load - the
mirror-aware third-moment push must handle short saturated runs in its own
band. Inside every record run the twin-free identity is visible exactly:
P-rate + n2-rate = 1 per slot (L=25 record: 0.80 + 0.20; L=100: 0.52 + 0.48).

### Anatomy of the record-holders

- L <= 13 records: n2-inside = 0 - pure n1 (every slot one prime, one lone
  composite). The X-local pattern at maximal load is the perfect alternation
  the constructor found forced on onset prefixes - reality DOES realize it,
  at length up to 13.
- lpf of the interior composites: 57-70% are killed by gears <= 13 at every
  record examined - the small gears do the composite work in ceiling runs
  (consistent with mechanic's bottom-decile ownership at scale).
- Composite members inside record runs are lone-killed (n1), so record runs
  are fragile-dense: every slot of a saturated run is a pseudo-twin slot.

### Part 3: frontier runs vs chain/fuel maximal strides - DIFFERENT objects

For L <= ~126 the record-load windows sit in ORDINARY parent strides (lengths
32-154, mostly below the top-1% cut); only for L >= ~160 does the record
necessarily live inside a top-1% stride, converging at L = maxstride to the
max stride itself with load 0.32-0.43 (round 5's long-run loads). Two
distinct extremal families:
  - load-extremal runs: short, shallow/absolute, prime-dense, n1-saturated -
    governed by prime constellations, NOT by gear chains;
  - length-extremal runs (the chain/fuel objects): deep, load-depressed
    (~0.3), governed by gap-word arithmetic.
They merge only at the top of the length range. The chain-condition analysis
cannot see the frontier's binding region (L ~ 14-32), and the frontier
analysis adds nothing to max-stride growth - complementary tools, not rivals.

### Proposed next chunk

The frontier says the bound must kill "saturated runs of length >= L0" for
some L0 it can reach. Offer: (1) exact census of saturated runs by length and
depth (how many L-saturated runs exist per window - the object whose
NON-emptiness reality keeps demonstrating; its count curve vs the inversion
zone's R(t) may localize the fight); (2) the alternation structure: saturated
runs force strict L/R alternation patterns (visible in the exhibit) - check
whether the mirror/parity structure of alternation words is constrained by
the machine's laws (connects to constructor's mirror-aware third moments).

## Round 7 (2026-08-18): alternation words obey the mirror; the caveat scoped

Tools: `research/alternation_words.py`. Frame first: a saturated slot's letter
is DETERMINED by the machine - in the open interior, prime = unhit, so
letter(k) = the side no gear hits. Saturated runs are the one-sided stretches
of the machine's hit pattern; their words are machine words, and the
positional mirror law (k -> -k reverses order and swaps L/R) makes exact,
testable predictions. All tests at y = 3163 and 10007 (90 and 333 maximal
runs of length >= 8).

### Word laws found (and proved where marked)

1. **Parity theorem (proved, 2 lines).** An odd-length word cannot equal its
   own reverse-complement (its middle letter would equal its own complement),
   so odd-length saturated runs are NEVER self-mirror. Data: 0 odd palindromes
   (forced); even-length self-mirror runs exist and are common (16 of 250 at
   L=8, y=10007).
2. **Mirror statistics confirmed, and specifically.** Word distributions are
   far closer to symmetric under reverse-complement than under reverse alone
   or complement alone - TV distances at L=8 (N=250): 0.328 vs 0.564 vs
   0.600; same ordering at L=9 and both scales. Exactly what the k -> -k law
   predicts (revcomp is the machine symmetry; nothing produces the other two).
   Letter marginal 0.4996 (mirror predicts 0.5).
3. **Duplicate words are CRT-alignment, not chance.** Identical L=8 words
   recur at position differences divisible by 5 in 86% of duplicate pairs
   (baseline 20%), by 7 in 63% (baseline 17%), by 35 in 55% (baseline ~3%).
   Word recurrence = small-gear skeleton recurrence: the forced-letter
   fraction is 0.729 measured (gears <= 13; crude CRT prediction 0.703).
4. **The landmark word is unique in range.** RLLRRLLLLRLRL occurs exactly
   once (slot 2452) in 1.67e7 slots - no recurrence of the full 13-word yet.
5. **Strict-alternation cap = 6, PROVED.** Strict LRLR... saturated runs
   correspond to primes at alternating gaps 8,4,8,4,...; the offset residues
   mod 5 cover all of Z/5 at length 7 (L-first phase) and length 6 (R-first),
   so gear 5 alone caps strict alternation at 6 slots (L-first) / 5 (R-first).
   Data: max strict alternation = 6, at slot 19125 at BOTH scales (another
   absolute landmark), letters LRLRLR - the L-first phase, exactly as the
   theorem requires. Corollary for the Constructor: "perfect alternation" in
   the strict L/R sense is impossible beyond 6 slots anywhere, at any scale -
   X's forced local patterns must be non-strict (repeats like the landmark's
   LLLL are the norm; the constraint is CRT, not alternation).

### Scoping the HL-constellation caveat (the honest one-pager)

**The statement.** persistence(L): every level-y open interior (y, y^2)
contains a saturated run of length L. Because tower bands tile - interior(y)
is slots (y/6, y^2/6), interior(y^2) starts where it ends - persistence(L) is
EQUIVALENT to: the increasing sequence r_1 < r_2 < ... of L-saturated-run
positions satisfies 6 r_{n+1} - 1 < (6 r_n + 1)^2, a Bertrand-type postulate
("the next run arrives before the square of the last").

**The strength ladder.**
- persistence(1): THEOREM, unconditional. If all large primes had prime
  partners, twin density would equal prime density, contradicting Brun. So
  gap(1) = 0 forever, provably.
- persistence(2): equivalent (up to Brun-generic side conditions) to
  infinitely many prime pairs at distance in {4,6,8} with the mod-6 side
  structure - disjunctive Polignac. OPEN; strictly weaker than the twin
  conjecture (a disjunction over three gaps), beyond published technology
  (best unconditional bounded gap: 246 - an imported corpus limit). This is
  the exact provability frontier: it sits between L=1 (theorem) and L=2
  (bounded-gap-8 class).
- persistence(L), L >= 3: L primes, one per slot, in 6L+O(1) consecutive
  integers with prescribed sides - disjunctive Hardy-Littlewood at tuple size
  L. Each observed word is its own admissibility witness (it happened), so HL
  predicts recurrence with density x/log^L x. L = 13 is 13-tuple class -
  far beyond twin strength in tuple size, softened only by the disjunction
  over all valid words. The exactly-one-prime side conditions are NOT the
  obstruction (composites are generic, 73% CRT-forced); the L-primes-in-a-
  short-interval part is.
- Decidable-in-principle structure: for each FIXED y the statement is a
  finite computation (the window is finite; load_frontier.py is the decision
  procedure), and the tower bands are disjoint, so verified bands stay
  verified - the conjectural content is ONLY the "for every y" quantifier.
  Empirical status: renewable to depth 0.5 for L <= 10 (members to 5e7);
  L = 13 witnessed only at the absolute landmark; its next Bertrand band is
  unexplored.

**Why the caveat cannot hurt the programme.** The frontier is a DESCRIPTIVE
upper envelope of reality's X-likeness, never a premise. If persistence(L)
fails at some depth, reality's most X-like stretches shorten there, the gap
to X widens, and any bound gets easier. The only illegitimate use of the
frontier would be citing renewability as proof that bounds at L <= 13 are
impossible - that direction, and only that direction, is conjecture-strength.

### Proposed next chunk

The strict-alternation cap generalizes: every periodic word pattern has a
CRT covering obstruction at some length (gear 5 caps strict alternation at
6; which patterns does {5,7} cap, at what lengths?). Candidate: the complete
"word grammar" - the exact set of infinitely-extendable letter patterns
(eventually-periodic words compatible with all small-gear teeth), which
would characterize what X's local behaviour CAN look like, unconditionally -
a positive-description complement to the attempts map.

## Round 8 (2026-08-18): the complete word grammar - and its finite horizon at 32

Steering taken: the full language of saturated-run side-words under small-gear
CRT, its growth, and the emptiness question. Tool: `research/word_grammar.py`.
(Frame note: the briefed period 30030 is n-space; slot space is 5005.)

### Admissibility, exactly

Word w (letters = prime side) is admissible iff some phase makes every prime
side avoid every small-gear tooth. Letter L at position i forbids phase
u_q - i (mod q); letter R forbids -u_q - i. Each position forbids exactly one
residue per gear, and per-gear allowed-phase sets combine freely by CRT, so:

    w admissible  <=>  for every q: the chosen residues do not cover Z_q.

Phase view: a slot where the small machine hits BOTH sides admits no letter at
all - and these B-slots are exactly the split/Bezout classes of gear pairs
from round 3. The language is nonempty at length L iff the CRT period has an
L-window free of B-slots.

### THE HORIZON THEOREM: saturated runs never exceed 32. Ever.

Gear pair (5,7) alone has B-classes {1, 34} mod 35 (its two split classes:
5 | 6k-1 and 7 | 6k+1 at k = 1 mod 35, mirrored at 34). Max cyclic gap = 33,
so ANY 33 consecutive slots contain a slot with both members composite.
Hence every saturated run - at every scale, forever - has length <= 32.
Two lines, unconditional. And the cap extends beyond saturation: any run of
consecutive slots each carrying >= 1 prime is also <= 32 (a twin slot only
adds avoid-constraints).

Escalation check: does adding gears lower the horizon? NO, through gear 23:
L0 = 32 for {5,7}, {5,7,11}, {5,7,11,13} (period 5005, 730 B-slots, 4
surviving corridor phases), {..17} (85085), {..19} (1.6M), {..23} (37.2M,
8.6M B-slots) - the (5,7) corridor survives every extension tested. Details
with a face: the corridor starts at k = 2 mod 35, and the L* = 13 landmark
sits at slot 2452 = 2 mod 35 - AT THE CORRIDOR MOUTH; at gears <= 17 and
<= 19 the extremal corridor's absolute start IS slot 2452. The landmark lives
where it does because that is the widest small-gear corridor.
Whether lim L0 = 32 over ALL gears is a Jacobsthal-type question: finitely
checkable per gear set, monotone non-increasing, and >= any realized run
length (so >= 13, and >= 14 if the Mechanic's hunt lands). The Mechanic's
L = 14 hunt is sanctioned: 14 <= 32, and the language at 14 has 579 words.

### The language census (gears <= 13, exact)

    L      1   4   5   10   13   14   17   18..26      31   32   33
    |lang| 2  16  30  235  474  579  1176  ~1140-1570  2560 2560  0

- All 2^L words admissible through L = 4; first exclusions at L = 5 (LLLLL,
  RRRRR): same-letter blocks cap at 4 - gear 5's law. Strict alternation caps
  (6 L-first / 5 R-first, round 7) confirmed in-language as special cases.
- Growth is NOT exponential: the ratio falls from 2.0 to ~1.0 by L = 18 and
  the language PLATEAUS (~1100-2600 words, L = 18..32) while 2^L passes 10^9.
  Past L ~ 17 the language is a fixed finite family of corridor words.
- The language is FINITE in total and is empty from L = 33 on (= L0 + 1, matching the
  horizon computed independently from B-gaps). Sharp contrast with the
  corpus's gap-word antidictionary (docs/forbidden-configurations.md), which
  is infinite: the saturated-run language is the OPPOSITE kind of object - a
  finite tree with a wall.

### Observed 757 runs vs the language

All 757 words (recomputed from research/data/satruns_ge10.csv via
Miller-Rabin, members to 7.2e10) are admissible - 0 failures. Coverage of the
language: L = 10: 199 distinct observed words / 235 in language = 84.7%
already realized; L = 11: 29.0%; L = 12: 5.3%; L = 13: 6/474 = 1.3%. The six
L = 13 runs have six DISTINCT words at six different residues mod 35
(2,3,13,17,5,18) - each uses a different corridor phase; no CRT-duplicates
(consistent with round 7: duplication requires congruence).

### Corollary: an unconditional load ceiling past the horizon

On any twin-free window, B-slots carry zero primes, so P_run <= L - minB(L):

    L        33     50     100    200    252     asymptote
    ceiling  0.970  0.920  0.910  0.880  0.873   1 - 730/5005 = 0.854

First unconditional maxload < 1 beyond the horizon (round 6's X-ceiling line
of 1 is now provably unreachable for L > 32). Honest note: reality sits far
below (0.52 at L = 100), and X's global demand (~0.33) is below the 0.854
asymptote too - the ceiling closes the L > 32 frontier, it does not create a
contradiction.

### Proposed next chunk

(1) Jacobsthal push: does the 32-corridor survive gears <= 100? (per-set
finite check; a drop would LOWER the unconditional cap; survival to large Q
suggests lim = 32). (2) Hand the horizon theorem + language to the Formalist:
"any 33 consecutive slots contain k = 1 or 34 mod 35, whose members are
divisible by 5 and 7" is a 3-line kernel-checkable fact with the run-cap as
corollary - the cheapest unconditional theorem the programme has produced.

## Round 9 (2026-08-18): the corridor method pointed at the top of the gap spectrum

Steering taken: (1) addresses of maximal gaps and their flanks mod 35/385/5005;
(2) is the near-top gap language finite like the saturated-run language, or
infinite like the antidictionary; feed alpha1 (F2 - F <= alpha1*q).
Tools: `research/topgap_corridor.py` (full periods to y=23, streamed period
1,078,282,205 for y=29), `research/topgap_nesting.py` (cross-machine nesting).
Frame: slot space; corpus halved units = 3 x slot units.

### Exact structure found at the top

1. **Mirror pairing (exact, all machines).** The set of maximal-gap intervals
   is closed under k -> -k at every machine tested (slot 0 is a universal
   opening - the all-gears shield - so no gap straddles 0, and maximal gaps
   come in proper mirror pairs; merged gap words appear in mirrored pairs,
   e.g. (4,8,15,7)/(7,15,8,4) at y=23, (10,10,23)/(23,10,10) at y=29).
2. **Address pinning (the landmark analogue - per machine, YES).** Maximal
   gaps concentrate into 1-2 endpoint classes mod 35 (y=19: all twenty at
   left = 5, right = 30 = -5; y=23: {3,33}; y=29: {2,25}) and 2-6 classes
   mod 385 out of 135 available (~30x over baseline; top-200 gaps' endpoint
   classes concentrate up to 5.7x). At y=23 and 29 the maximal gap is UNIQUE
   up to mirror. Unlike the L*=13 landmark the pinned address DRIFTS with the
   machine - gaps are machine-relative objects, saturated runs are absolute.
3. **Chain skeleton at the maxima (theorem-matching).** Every new maximum
   M_y -> M_y' is a merge of old gaps by an alternating chain of the new gear:
   kill sides strictly alternate (R,L / L,R,L / R,L,R in every case), and the
   interior kill spacings are EXACTLY {2u', q-2u'} of the new gear (17: 6/11;
   19: 13; 23: 8/15; 29: 10) - the chain condition's {phi, phi+s} law,
   reconfirmed independently at the extreme tail itself.
4. **Growth stratum: new maxima grow from MEDIUM old gaps.** Old-gap sizes
   under new maxima: 0.16-0.68 F_old (chains k = 2-3), except two y=19 cases
   where an old MAXIMAL gap extends by k=1 (18+7). The corpus correction
   ("F2 lives at medium gap pairs") is the generic regime; max-extends-max is
   the exception, not the rule.

### The language verdict (question 2)

The near-top gap language is **NOT finite in absolute terms** - gap values
grow with y, there is no 32-cap analogue for the top of the gap spectrum (the
antidictionary-like infinitude persists). What IS finite/stable is the
RELATIVE grammar of top-gap neighbourhoods, three alphabets:
  - flanks of near-top gaps: {1,2,3,4,5} slots at every machine tested
    (isolation law, quantified; top flank pairs (2,2),(2,3),(1,3),(2,5));
  - chain interior spacings: exactly {2u'_q, q-2u'_q} (rigid, theorem-backed);
  - near-top neighbourhood word counts stay small and non-growing (14-42
    distinct 5-gap words per machine, no trend from y=13 to 29).
So: top-gap neighbourhood = [small flank] [medium gaps] [rigid chain
skeleton] - a finite grammar in structure, infinite in values.

### alpha1 evidence (for the Constructor)

    y                13     17     19     23     29
    F (slot)         11     18     25     34     43
    F2               16     25     31     39     55
    F2-F (halved)    15     21     18     15     36
    (F2-F)*3/q_next  0.88   1.11   0.78   0.52   1.16

All below the constructor's measured 1.24; no growth trend (non-monotone).
F2 anatomy splits into two regimes: F + small flank (y = 13, 17, 23 - there
F2 - F = 3*flank <= 15 halved, and a flank cap would give alpha1 < 1) and
medium+medium (y = 19: 21+10; y = 29: 30+25 - the regime that must control
alpha1 asymptotically). Honest verdict: no corridor CAP on F2 - F was found;
the corridor method delivers the addresses and the rigid skeleton, not the
bound. If anti-clustering has a proof in this frame, it lives in the
medium-medium regime's address classes (2-6 classes mod 385 for the top
stratum - a finite check per machine of whether two such classes can sit
adjacent), which is a concrete next target but NOT closed this round.

### Proposed next chunk

(1) The medium-medium adjacency question, exactly: for machine M, which pairs
of top-stratum address classes (mod 385/5005) can be ADJACENT in the opening
sequence (separated by one opening)? If the pinned classes of near-top gaps
can never be adjacent, F2 comes from strictly lower strata and alpha1 follows
per machine by finite check. (2) Alternatively hand the mirror-pairing +
skeleton facts to the Constructor as constraints on their merge transform.

## Round 10 (2026-08-18): uniformity - the word pins the address; the drift recursion dies

Steering taken: (1) is the pinned address a computable function of the machine
(drift recursion)? (2) is the top-stratum class count uniformly bounded, with
a proof-shaped reason? Tool: `research/address_drift.py` (near-top strata at
0.9F for y = 13..29, full periods, y=29 streamed).

### LAW A - word-pinning: ESTABLISHED (the uniformity engine)

The neighbourhood word of a near-top gap (openings in a window of 20 slots
each side) determines its address mod 385 almost uniquely: each opening must
avoid both teeth of each small gear, forbidding 2 offsets per gear, and the
~10 openings around a top gap leave almost nothing:

  - gear 5: pinned to EXACTLY ONE offset by every near-top word - 206/206
    across all five machines;
  - gear 7: unique for 94% of words, never more than 2 offsets;
  - gear 11: unique for 90%, never more than 4;
  - gear 13: 1-5 offsets (looser - fewer teeth per window, as expected);
  - full mod-385 address: UNIQUE for 87% of words ((1,1,1) in 180/206 cases),
    <= 4 always, at every machine (max 4,4,4,4,3 for y = 13..29).

Containment is exact (0 fails in 206 words: every observed address is
word-compatible) and tightness is high (71-85% of predicted phases are
realized). Consequence: #top-stratum classes <= sum over near-top words of
#phases(word) <= 4 x #words - and the observed class counts sit far below
even that (6-14 classes, FLAT from y=13 to 29, while near-top gap counts
swing 20-106), because distinct words share pinned addresses. The uniformity
the coordinator asked about lives here: per-word pinning <= 4 is the
proof-shaped half (exposure-criterion counting, computable per word); the
flat class count is measured, mechanism = word-overlap on shared skeletons.

### LAW B - drift recursion: REFUTED as stated

Candidate law "new max address = old top-stratum address - left flank"
(suggested by the striking mod-385 near-matches 47-2=45, 122-5=117, 115-5=110,
252-2=250, 322-2=320 at the first two steps) fails systematically at later
steps: reachability of new maximal addresses from the old 0.9F stratum
(self-or-plus-first-flank) runs 18/20, 14/20, then 0/4 (step 19->23) and 1/2
(23->29). Reason: new maxima grow from DEEP-medium old gaps (0.16-0.68 F_old,
round 9) that no near-top stratum tracks. A recursion through stratum
addresses would have to carry the whole medium spectrum - not an induction
anyone can close. The early-step matches were real but coincidental to the
flank regime; the honest law is LOCAL, not inherited:

    address = pin(word),  not  address = f(previous address).

### What a machine-independent alpha1 statement now needs

Two halves, one established: [per-word pinning <= 4 mod 385, uniform in y -
ESTABLISHED, mechanism exposure-counting] + [uniformity of the near-top word
grammar itself - OPEN: word counts are non-growing (20-106, no trend) but
words are machine-relative objects]. For the Constructor's adjacency chunk:
the adjacency question can now be run at the WORD level - two near-top words
can sit adjacent only if their pinned phase sets are CRT-consistent with the
separation, a finite check per word pair, no period scan. That converts
"can two top-stratum classes be adjacent" from a per-machine scan into a
grammar-level computation on the observed word lists (available in
address_drift.py's groups).

### Proposed next chunk

The open half: characterize which words CAN be near-top (the extreme-value
grammar) - specifically whether the flank alphabet {1..5} + chain skeleton
{2u', q-2u'} + pinning constraints already delimit a finite word-shape family
whose pinned classes can be enumerated a priori (then alpha1's adjacency
check becomes machine-independent arithmetic). Alternatively, support the
Constructor's adjacency computation directly with the word-pair CRT check.

## Round 11 (2026-08-18): two grammars, one clean reduction, and the k=4 event dissected

Steering taken: is the near-top word-shape family finite a priori; does grammar
finiteness reduce to the fuel bound; does the pinning law hold for the k=4
event? Tools: `research/word_shapes.py`, `research/k4_pinning.py`.

### The formalization: there are TWO grammars, and they answer differently

**INTERIOR grammar (one gear step, u'-free).** A merge word with k interior
kills is side-alternating with spacing word alternating sigma = 2u'_q and
sigma-bar = q - sigma, so the spacing pattern is determined by its initial
type: EXACTLY 2 candidates per k ((s,q-s,s,...) or its swap). Abstracting
parts to c classes: |shapes(k)| <= 2 c^(k+1) - finite for each k, machine-
independently. **The clean reduction holds at this level: the interior
grammar is finite iff k_max is bounded** - and with k_max growing (2,2,3,2,4
by step), the graded form below is the honest statement.

**BOUNDARY grammar (the pinning window, W = 20).** Finite a priori but
trivially so (compositions, 2^20 - 1 per half); CRT-admissibility cuts it to
a machine-independent superset of 3798 half-shapes (enumerated exactly;
pruning valid by monotonicity - extensions of inadmissible words stay
inadmissible). The NEGATIVE that matters: the observed family does NOT
stabilize inside it. Cross-machine full-shape recurrence is ZERO at every
machine (0/24, 0/20, 0/102, 0/30, 0/22); max flank part grows 7 -> 13 with y
(the {1..5} alphabet was a first-flank fact only - deeper parts track typical
gap sizes); observed halves = 123 = 3.2% of admissible, essentially disjoint
per machine. Mirror closure exact everywhere. So: finite a-priori SUPERSET
yes, a-priori list of OCCURRING shapes no - extreme-value selection roams
inside the fixed admissible family without repeating.

### The k=4 event under the grammar (and the fuel-site anatomy)

Phase-free site census over machine 29's full period (1.078e9): exactly 4
sites with spacing word (10,21,10) - positions 220171102, 406081827,
672200337, 858111062, two mirror pairs under the M29 mirror (confirming the
Mechanic's N4 = 4) - and ZERO sites for the grammar's other permitted k=4
word (21,10,21). The grammar allowed two words; arithmetic selection realizes
one. Two sharpenings the census could not see:

- **Only ONE site is phase-aligned with the real gear 31** (672200337, where
  p = u_31 = 26 mod 31; sides LRLR). The M29 mirror does NOT commute with
  gear 31's teeth (P29 is a unit mod 31), so fuel sites mirror-pair but
  REALIZED chains need not: 1 of 4 fires. (Curiosity: site 858111062 sits on
  gear 31's shield, p = 0 mod 31 - permanently sterile.)
- **The realized k=4 merge does NOT set the record**: its machine-31 gap is
  [672200330, 672200382], G = 52, word (7,10,21,10,4), while F(31) = 58 comes
  from a k=3 site with better parts. Fuel k_max and the record are decoupled:
  more fuel does not mean a bigger gap - parts matter as much as links.
- **PINNING HOLDS for the k=4 object**: neighbourhood word pins the address
  to 3 phases mod 385 (<= 4), observed address in the set. The pinning law
  survives its first k=4 test.

### The k-graded statement (what "finite grammar per k, k growing" buys)

With the Mechanic's spectra F_j(23) = (34,39,50,58,65,77) and F_j(29) =
(43,55,65,70,85,90), the graded increments (F_{k+1} - F)/q_next are:

    k          2      3      4      5
    at 23:   0.55   0.83   1.07   1.48
    at 29:   0.71   0.87   1.35   1.52

So a k-graded tolerance lemma - increment <= F_{k_max+1} - F <= alpha(k) q -
holds with alpha(4) ~ 1.4 and alpha(5) ~ 1.5 at these machines, comfortably
under the 2.5 budget even at k_max = 5. What the grading buys: the constant
is priced PER FUEL LEVEL, shapes(k) is finite per level, and k_max grows
glacially and arithmetic-selected (not smoothly y-driven). What it does NOT
buy: a bound on F_{k+1} - F is spectrum flatness - the grading prices Wall V,
it does not evade it. Machine-independent alpha1 remains open exactly there.

### Proposed next chunk

The fuel-site phase-alignment ratio (1 of 4 at the k=4 event) suggests a
selection law worth quantifying: what fraction of fuel sites fire across
steps and k (N_k sites vs realized chains)? If alignment is ~2/q per site,
realized high-k events are doubly rare - fuel abundance x phase alignment -
which would make the effective k_max of REALIZED chains grow slower than the
census k_max, tightening the graded constant. One pass over existing census
machinery per step.

## Round 12 (2026-08-18): the firing law is exact - and it refutes my own round-11 claim

Steering taken: alignment fraction at the N4 populations, the law behind it,
and what double rarity does to the graded constant. Tools:
`research/firing_ratio.py`, `research/firing_law_check.py`,
`research/firing3137.py`, `research/graded_constant.py`.

### The firing law (derived, then verified with zero violations)

Inside a chain of gear q', consecutive kills sit at the two teeth {u, -u}
alternately, so a kill at u is followed by a step of -2u = q'-s and a kill at
-u by a step of +2u = s (s = 2u mod q'). **The spacing word's FIRST entry
therefore fixes the orientation and hence a SINGLE firing residue:**

    word starts with s      ->  site fires iff p = -u (mod q')
    word starts with q'-s   ->  site fires iff p = +u (mod q')

One residue, not two: per-window firing density 1/q', HALF the naive 2/q'
(k=1 kills are the exception - they fire at both teeth, 2/q').

Verified by recomputing every site's actual kill-set from gear q' directly
(the checker asserts both directions - predicted-fired must fire, predicted-
not must not): **zero violations** over 13,062 sites at 19->23 and 29->31.
Measured per-window fractions: 428/13000 = 0.0329 vs 1/31 = 0.0323 (k=3);
2/62 at 19->23 (small sample, 1/23 = 0.043).

### SELF-CORRECTION: round 11's "1 of 4 fired" was a one-window artifact

I reported last round that only 1 of the 4 k=4 fuel sites is phase-aligned,
and that site 858111062 (on gear 31's shield) is "sterile forever". **Both
claims are wrong.** The new machine's period is q'*P_old, and P_old is
invertible mod q', so each site recurs at q' distinct residues across the
q' phase windows: **every fuel site fires exactly once per new-machine
period**, at the computable address

    j = (fire - p) * P_old^{-1}  (mod q'),   firing position p + j*P_old.

Verified for all four k=4 sites: j = 12, 30, 0, 18, giving positions
13,159,557,562 / 32,754,547,977 / 672,200,337 / 20,267,190,752, each with
chain residues [26,5,26,5] - all teeth, all four fire. The "1/4" was measured
inside one machine-29 period only.

Same artifact corrupted my round-11 record claim: "realized k=4 gives G=52
while F(31)=58 comes from a k=3 site". The 52 was the best merge in ONE
window; F(31)=58 lives in a different phase window. Anything I said last
round about fuel and records being decoupled is withdrawn.

### Consequence for the graded constant: NO multiplier (honest negative)

    realized k-chains per NEW period = N_k     (exactly - no suppression)
    realized density                 = N_k / P_new = (1/q') x site density

Alignment is a DENSITY factor, never a count factor. The Constructor's
word-indexed ceiling gets no free multiplier from it. The hoped double
rarity (fuel x alignment) does not exist: it is one rarity, counted twice.

### The graded table (what actually binds)

    step      q   F_old F_new  incr/q  lemma1  excess  exc/q      N3   N4
    13->17   17     11    18   0.412   0.294       2  0.118       0    0
    17->19   19     18    25   0.368   0.368       0  0.000       0    0
    19->23   23     25    34   0.391   0.261       3  0.130      62    0
    23->29   29     34    43   0.310   0.172       4  0.138       0    0
    29->31   31     43    58   0.484   0.387       3  0.097   13000    4
    31->37   37     58    88   0.811   0.270      20  0.541   70964  216

Max increment/q' = 0.811 (31->37) against the 2.5 budget - headroom 3.1x, no
step binds. But the shape is the warning: **excess overtakes lemma 1 exactly
at the largest fuel population** (0.541 vs 0.270 at 31->37, where N3 = 70,964
and N4 = 216), which is precisely what "realized = N_k per period" predicts.
Fuel abundance drives excess and alignment does not damp it, so the excess
share should keep growing with the fuel census - the 2.5 budget is safe at
these sizes on measured numbers only, and lemma 2 is not vacuous.

(Still running at write-up time: the 31->37 site-residue histogram over the
full 3.34e10 period - a uniformity check on how the 216 sites spread across
the 37 phase windows. Not load-bearing now that firing is once-per-period by
the law; it can only refine the density statement.)

### Proposed next chunk

The excess/lemma-1 crossover at 31->37 is the real signal. Offer: price the
excess share as a function of the fuel census - is excess/q' ~ c * log(N3)/q'
or ~ (F_{k+1}-F2)/q' with the spectrum doing the work? Two more steps of
spectrum data (machines 37, 41) would settle whether the excess share
saturates or keeps climbing; that is the quantity the tolerance route's
constant actually depends on, and my round-11 graded framing priced the wrong
half of it.

## Round 13 (2026-08-18): the excess law - a mechanism, a crossover, and predictions

Steering taken: excess/lemma-1 split vs fuel population; one-off or trend;
advance predictions for 37->41 and 41->43; restate the graded tolerance under
the corrected firing model. Tools: `research/merge_decompose.py`,
`excess_law.py`, `excess_predict.py`, `merge3137.py`.

### The corrected firing law buys an exact, cheap algorithm

Because every site fires exactly once per new period (round 12), residues drop
out of the record question entirely:

    F(M+q') = max over k >= 1, over all k-sites, of ( o[i+k] - o[i-1] )

where a k-site is k consecutive OLD openings whose spacing word is one of q''s
two alternating literal words (k=1: any opening), and o[i-1], o[i+k] bracket
it. This is the Constructor's word identity made computational - no new-period
scan, no residue bookkeeping. **Verified exactly at five steps**: F_new =
18, 25, 34, 43, 58 for 13->17, 17->19, 19->23, 23->29, 29->31, every value
matching the known F. (k=1 reproduces F2 identically, as it must.)

### The excess law

    excess = F_new - F2 = max over nonempty words w of [ span(w) - deficit(w) ]
    deficit(w) := F2 - FS_max(w;M)        (extreme-value deficit: a word with
                                           fewer occurrences samples worse flanks)

Spans are fixed by q' alone: k=2 -> {s, q'-s}, k=3 -> q', k=4 -> {q'+s, 2q'-s},
k=5 -> 2q', k=6 -> 2q'+s. Occurrences and flanks come from M.

### The crossover is a TREND with a mechanism, not a one-off

    step      q   F2  F_new  excess  short-span  long-span  winner
    13->17   17   16     18       2           6         11  short-compatible
    17->19   19   25     25       0           6         13  short-compatible
    19->23   23   31     34       3           8         15  short-compatible
    23->29   29   39     43       4          10         19  short-compatible
    29->31   31   55     58       3          10         21  short-compatible
    31->37   37   68     88      20          12         25  span >= 20 > 12 -
                                                            NOT the short word

Five steps in a row the winner is (consistent with) the SHORT k=2 word; at
31->37 the excess exceeds the short span outright, so the winner has migrated
to a longer word. Mechanism, measured: fitting all 13 (word, occurrence)
observations,

    deficit ~ 2.52 * ln(openings / occurrences) - 1.17     (residual sd 3.4)

and ln(openings/occurrences) ~ span / lambda with lambda = mean gap. So

    span - deficit ~ span * (1 - 2.52/lambda)

and lambda grows (3.37, 3.82, 4.27, 4.68, 5.02, ~5.37 for machines 13..31 -
Mertens-slow but monotone). The bracket goes 0.25 -> 0.34 -> 0.41 -> 0.46 ->
0.50 -> 0.53: **longer words become profitable as the machine's mean gap
grows.** That is the crossover, and it predicts continued climbing. Note the
fit UNDER-predicts the 31->37 excess (it gives ~8, actual 20), i.e. deficits
at scale are shrinking faster than the log fit - climbing, if anything,
harder than modelled.

### Predictions, stated in advance (falsifiable by the machine-37/41 census)

    step 37->41 (s=14, k=2 spans {14,27}, k=3 span 41; F(37)=88, F2(37)=90)
      H-SAT  : excess ~ 14 - (6..8) = 6..8    -> F(41) ~ 96..98
      H-CLIMB: excess ~ 27 - (8..12) = 15..19 -> F(41) ~ 105..109
      DISCRIMINATOR: F(41) <= 100 favours SAT; F(41) >= 103 favours CLIMB.
    step 41->43 (s=29, k=2 spans {14,29}, k=3 span 43)
      H-SAT  : excess ~ 6..8;  H-CLIMB: excess ~ 17..21 (needs F2(41)).
      Note the reversal: at q'=43 the LONG k=2 span is 29 = s, so a climbing
      winner shows up as a much larger jump than at 41.

My own expectation, on the mechanism above: CLIMB at both steps.

### The graded tolerance, restated under the corrected model

    increment = F(M+q') - F(M) = [F2(M) - F(M)] + excess
              = lemma1 * q'  +  max_w [span(w) - deficit(w)]

and, PROVIDED deficits are non-negative (measured: all 13 observations
positive, but NOT proved - FS is a sum of two NON-adjacent gaps while F2 is
the max sum of two ADJACENT gaps, so FS_max <= F2 is an empirical fact here,
not an identity), the cap-6 theorem gives an unconditional ceiling

    excess <= span_max = 2q' + s <= 2.67 q'      =>   increment/q' <= lemma1 + 2.67.

Two honest consequences:
* the ceiling 2.67 EXCEEDS the 2.5 budget, so the cap alone does not deliver
  the tolerance hypothesis - the deficit term is load-bearing, exactly as the
  Constructor's missing FS_max bound says;
* but the tolerance constant alpha*(y) grows like ln y (5.64 at y=101, 8.71 at
  1e4, 13.3 at 1e6), while this ceiling is a CONSTANT multiple of q'. So even
  unlimited climbing is asymptotically safe: **lemma 2 cannot break the route
  asymptotically; only the finite range and lemma 1 can.** That is the useful
  half of this round.

Measured increments (unchanged, now with the corrected model behind them):
0.412, 0.368, 0.391, 0.310, 0.484, 0.811 - max 3.1x under budget.

### Proposed next chunk

Prove or refute deficit >= 0, i.e. FS_max(w) <= F2 for every literal word w.
It is the one gap between the cap-6 theorem and an unconditional lemma 2, it
is a pure statement about the old machine's gap sequence (no primes), and it
is the kind of statement the corridor machinery has closed before: two
non-adjacent gaps bracketing a word occurrence cannot jointly exceed the best
adjacent pair. If it holds, lemma 2 is DONE unconditionally and the tolerance
route reduces to lemma 1 alone.

### CORRECTION to round 13, same day - the algorithm was incomplete twice over

The 31->37 run came back at F_new = 71 against a known lower bound of 88
(mechanic's 9.7% scan of machine 37 already exhibits a gap of 88). Diagnosis
and fix, with both failure modes recorded:

* `merge_decompose.py` matched only the LITERAL spacing VALUES {s, q'-s} and
  their alternating words. It therefore missed **padded links**: two killed
  openings may also sit at the SAME tooth (spacing = 0 mod q', costing a gap
  >= q'), or at opposite teeth a full period further apart (spacing = +-2u mod
  q' but larger). Undershoots: 71 vs >= 88.
* `merge_general.py` then allowed every spacing = {0, +-2u} mod q'. Too
  permissive: the +-2u letters must ALTERNATE (a +2u step goes -u -> +u, so it
  is only legal FROM tooth -u; two +2u steps in a row would land on +3u, not a
  tooth). Overshoots: 45 vs 43 at 23->29, on the illegal word (10,10).
* `merge_correct.py` is the right condition - spacings = 0 or +-2u mod q', with
  the non-zero letters alternating and 0's insertable freely. **Re-verified
  exactly at all five steps: 18, 25, 34, 43, 58.**

What this does to the round-13 claims:

* SURVIVES - the exact-algorithm form (F(M+q') = max over maximal legal killed
  runs of o[i+k] - o[i-1], from the OLD machine alone) and the excess law's
  shape (excess = max over runs of [span - deficit]).
* SURVIVES - the crossover direction at 31->37: F(37) >= 88 and F2(31) = 68
  (exact), so excess >= 20 > the short k=2 span 12. The winner is definitely
  not the short word.
* CHANGES - the winner at 31->37 is NOT a longer LITERAL word: the best literal
  configuration reaches only 71. It must involve a padded link. So the round-13
  mechanism story ("longer literal words become profitable as lambda grows")
  is at best half the story; padding is the other half, and the deficit fit was
  calibrated on literal words only.
* **WITHDRAWN - the asymptotic safety argument for lemma 2.** It rested on
  excess <= span_max = 2q' + s <= 2.67 q', which used the cap-6 theorem - and
  that theorem is stated for LITERAL chains. Padded runs are not capped by it
  (each padded link buys span >= q' at the cost of needing a gap >= q' in M,
  which exists whenever F(M) >= q', true at every step from 23->29 on). Until
  padded runs are bounded, there is NO constant ceiling on excess/q' from this
  argument, and my claim that "lemma 2 cannot break the route asymptotically"
  is unsupported. Constructor should not build on it.

The predictions for 37->41 and 41->43 stated above were also derived from
literal spans only, so they are lower-biased; treat the H-CLIMB branch as a
floor rather than an estimate. The discriminator (F(41) <= 100 vs >= 103)
still separates the hypotheses, but a padded winner could exceed both ranges.

Corrected next chunk: bound the padded runs. The concrete question is now
"how many padded links can a killed run carry?", i.e. how often can consecutive
openings of M at spacing = 0 mod q' (gap >= q', so a top-stratum gap of M)
chain together - which is exactly the top-gap adjacency machinery from rounds
9-10 pointed at a new target, and the same object as the Constructor's
"beyond-cap extension needs a padded link" remark.

### Round 13, final: the 31->37 winner is a PADDED run - crossover = padding onset

`merge_correct.py` on the full machine-31 period (3.34e10):

    STEP 31->37 (u=31, letters A=25 B=12 mod 37): F_old 58, F2 68, F_new 88
      winner: 3 kills at 9,463,664,103, spacings (37, 12), span 49,
              flanks 28+11, padded links: [37]
      excess = 20 (+0.541 q')

**F_new = 88 exactly** - matching the mechanic's independently exhibited gap of
88 from a 9.7% scan of machine 37. The corrected algorithm is now verified at
SIX steps: 18, 25, 34, 43, 58, 88.

The anatomy settles the mechanism question. The winning run is
[kill] --37--> [kill] --12--> [kill], i.e. one PADDED link of exactly q' = 37
(two kills at the SAME tooth, a gap of exactly 37 in machine 31) followed by
one literal B-link of 12. Span 49 = q' + B, beating the longest available
literal span (k=3, 37) - which is precisely why the literal-only algorithm
stalled at 71.

So the corrected story:

* the first five steps have LITERAL winners (spans 11, 13, 23, 10, 10);
* 31->37 is the **first padded winner** - the crossover is a PADDING ONSET,
  not the "migration to longer literal words" I proposed earlier today;
* the shape of the earlier reasoning survives in a modified form: a padded link
  buys span q' at the price of needing a gap of exactly q' in M (share
  ~ e^{-q'/lambda}), so as lambda grows padding becomes affordable - the same
  span-versus-scarcity race, with padding as the vehicle rather than long
  literal words;
* and the ceiling really is gone: padding has no cap-6 analogue, so nothing in
  hand bounds excess/q'. The withdrawal stands.

Self-consistency check on whether a cheap bound exists: with k-1 links each
>= min(s,q'-s) ~ q'/3 and flanks <= 2F(M), one gets G <= 2F(M) + (k-1)F(M) and
k <= 3G/q' + 1, which rearranges to G(1 - 3F(M)/q') <= 2F(M) - vacuous whenever
F(M) > q'/3, i.e. always in this regime. No easy ceiling; the padded-run bound
has to come from the arithmetic of how often gaps of exactly q' can chain, not
from counting.

## Round 14 (2026-08-18): the padding lemma - a ceiling, and exactly where it dies

Chunk: bound the padded runs. Tools: `research/padding_bound.py`,
`padding_horizon.py`, `padding31.py`.

### The lemma (exact, from the F_j spectrum)

A legal killed run of k kills occupies k+1 CONSECUTIVE gaps of M (its k-1
links plus two flanks), so its merged gap obeys G <= F_{k+1}(M). Suppose a run
carried TWO padded links with j literal links between them. Those j+2 links are
j+2 consecutive gaps of M summing to at least 2q' + j*L, where L = min(s,q'-s)
is the cheapest literal link. Hence

    two padded links require   F_{j+2}(M) >= 2q' + j*L   for some j >= 0,

and contrapositively:

> **PADDING LEMMA.** If F_{j+2}(M) < 2q' + j*L for every j >= 0, then every
> legal killed run carries AT MOST ONE padded link. The j=0 case,
> F_2(M) < 2q', is the headline: two padded links can never be adjacent.

A companion threshold: if 2q' > F(M) then no gap of M is 2q', so every padded
link has size EXACTLY q'.

### Verified at every step computed, and confirmed empirically

    step      F(M)  F2(M)   q'   2q'   pad size = q'?   p <= 1?   span ceiling
    13->17      11     16   17    34   yes (vacuous)    yes       5.71 q'
    17->19      18     25   19    38   yes (vacuous)    yes       6.37 q'
    19->23      25     31   23    46   yes              yes       5.70 q'
    23->29      34     39   29    58   yes              yes       5.69 q'
    29->31      43     55   31    62   yes              yes       6.35 q'
    31->37      58     68   37    74   yes              yes       6.35 q'
    37->41    >=88   >=90   41    82   NO               NO        NONE

(13->17 and 17->19 are vacuous: F(M) < q', so padding is impossible at all.)

Empirical census over full periods - gaps = 0 mod q', and padded links per
maximal legal run:

    19->23: gaps of 23: 86    adjacent padded pairs 0   max padded/run = 1
    23->29: gaps of 29:  6    adjacent padded pairs 0   max padded/run = 1
    29->31: gaps of 31: 2090  adjacent padded pairs 0   max padded/run = 1
    13->17, 17->19: no gaps = 0 mod q' at all           max padded/run = 0

Every padded gap found has size exactly q' - never 2q' - as the second
threshold predicts. Zero adjacent padded pairs anywhere, as the lemma
requires. (31->37 census running; the lemma already proves p <= 1 there, and
round 13's winner is a single padded link of exactly 37.)

### What it buys: the ceiling is restored, at a bigger constant

With p <= 1 and padded size exactly q', a run is
`[literal chain] --q'-- [literal chain]`, and the cap-6 theorem applies to each
literal segment separately (it is a property of that segment's own opening
sequence), so

    k <= 12    and    span <= 2*(2q' + s) + q' = 5q' + 2s <= 6.35 q'.

That restores the span ceiling I withdrew in round 13 - at 6.35q' rather than
the 2.67q' literal-only reasoning had given. Honest scope: this bounds the
SPAN, not the increment; the flanks are still two gaps of M, each up to F(M),
so a ceiling on excess/q' still needs the Constructor's FS_max bound. What is
new is that the span half is no longer open.

### And exactly where it dies - at the very next step

Both enabling conditions are ratios against 2q', and both climb monotonically:

    step     13->17 17->19 19->23 23->29 29->31 31->37 | 37->41
    F(M)/2q'   0.32   0.47   0.54   0.59   0.69   0.78 |  1.07
    F2(M)/2q'  0.47   0.66   0.67   0.67   0.89   0.92 |  1.10

F and F2 grow superlinearly against the next prime, so once past 1 they stay
past. **The padding ceiling is a small-machine phenomenon that ends exactly at
37->41**, where simultaneously (i) gaps of 2q' = 82 become possible
(F(37) >= 88), so a single padded link can be worth 2q'; and (ii) two adjacent
padded links stop being excluded (F_2(37) >= 90 > 82). The answer to "does a
bound fall out" is therefore: **yes for machines up to 31, and no
asymptotically** - stated plainly because the honest version is the useful one.

Falsifiable consequence for the census: at 37->41 I expect the first run with
two padded links, and/or the first padded link of size 82. If the Mechanic's
machine-37 scan finds neither, the lemma's threshold is not tight and there is
a further mechanism suppressing padding - which would be worth much more than
the lemma itself.

### Proposed next chunk

Two options, coordinator's pick. (1) Chase the suppression: run the padded-link
census at 37->41 directly (machine 37's period is 1.24e12 - expensive but the
padded links only need gaps >= 41, so a gap-filtered scan is far cheaper than a
full merge scan) and see whether p <= 1 survives past its proof. (2) Attack the
flank half instead: with span now bounded at 6.35q' for the computed range, the
whole excess question reduces to FS_max, and my pinning/address machinery from
rounds 9-10 applies directly to the two flank gaps of a winning run.

## Round 15 (2026-08-18): frame stated; and the 37->41 census predicted from the corridor

Tools: `research/padding_37_41.py`, `padding_corridor_law.py`. Also folded: my
own full-period 31->37 padding census (26,367 gaps of exactly 37, max 1 padded
link per run, 0 adjacent padded pairs) - agreeing with the mechanic's 26,366 to
within one, presumably a period-wrap convention difference worth one line from
whoever cares.

### 1. Frame check (my side, stated unambiguously)

**All lateral gap numbers are in SLOT units.** Slot k is the pair (6k-1, 6k+1);
openings are surviving slots; a gap is a difference of slot indices.
Conversions: member-space gap = 6 x slot gap; corpus halved-coordinate gap
= 3 x slot gap. Therefore

    lateral "a padded link costs exactly q'"  ==  harvester "3q'" (halved).

Same fact, one frame factor. Independent check of the factor against the
corpus: F(2,43) = 309 halved, and 309 = 3 x 103 with 103 sitting in the
mechanic's machine-37 F_j spectrum. My measured padded values, all steps,
are exactly q' in slots (23, 29, 31, 37) = 69, 87, 93, 111 halved = 3q'.
No disagreement to settle on my side; harvester owns the write-up.

### 2. The 37->41 question, decided in advance where it can be

**BRANCH A - if a double-padded run IS found.** The ceiling does not collapse.
Constructor's count cap gives p <= (F + 5q'/6)/q' = 2.98, so p <= 2 (p = 3 is
arithmetically impossible). With F_2(37) < 123 both links must be exactly q',
so the run is forced to be `[literal chain] --q'-- [kill] --q'-- [literal
chain]` and the span ceiling moves

    p = 1:  span <= 5q' + 2s = 233 = 5.68 q'
    p = 2:  span <= 6q' + 2s = 274 = 6.68 q'

i.e. **exactly one q' worse, not a collapse.** The general form is
span <= (4+p)q' + 2s.

**BRANCH B - if none is found, the mechanism is the corridor, and I can prove
the adjacent case outright.** Every opening lies in the 15-residue exposed set
E mod 35 (avoiding the teeth of gears 5 and 7). Two adjacent padded links of
sizes a q', b q' put three consecutive openings at r, r+a g, r+(a+b) g mod 35
with g = q' mod 35. For q' = 41, g = 6, and

    r, r+6, r+12 all in E  has ZERO solutions over all 15 r in E.

So **two adjacent equal padded links are impossible at 37->41 by the (5,7)
corridor alone** - no spectrum input, hence unaffected by the fact that the
machine-37 F_j values are only prefix lower bounds. That is the repulsion
mechanism the coordinator asked for, and it is exact.

### 3. The general law behind it (and why this is not a trend)

Feasibility depends only on q' mod 35, and the classes split cleanly:

    adjacent EQUAL padded links (1,1) possible:  q' = 23, 37, 43, 47, 53, 67,
                                                 73, 83, 97 ...
    impossible:                                  q' = 29, 31, 41, 59, 61, 71,
                                                 79, 89 ...

Exactly 12 of the 24 invertible classes mod 35 forbid it - a 50/50 property of
q' mod 35, **not a trend in scale**. And there is a perfect dichotomy in the
table: whenever the (1,1) shape is feasible the unequal shapes (1,2) and (2,1)
are infeasible, and whenever (1,1) is infeasible the unequal shapes have
exactly 2 phases each. So padding structure switches on and off with the
residue of q', which is why the smooth supply^2/gaps model cannot predict it -
the same lesson as round 11's fuel: arithmetic selection beats the smooth law.

### 4. What is actually still open at 37->41, and how sharp it is

Two shapes survive the corridor at q' = 41 and must be settled by the spectrum:

* **adjacent UNEQUAL** (one link q', one 2q' = 82; corridor-feasible at r =
  0, 5, 12, 17): needs F_2(37) >= 123. Measured prefix gives >= 90; the ratio
  F_2/F across machines 13..31 runs 1.45, 1.39, 1.24, 1.15, 1.28, 1.17, so
  the plausible true F_2(37) is ~105-115. 123 is above that band - unlikely
  but not excluded.
* **NON-adjacent** (two padded links with j >= 1 literal links between): needs
  F_{j+2}(37) >= 2q' + jL = 82 + 14j. For j = 1 that is **F_3(37) >= 96, and
  the measured prefix stands at 95**.

**The whole census outcome turns on one unit of F_3(37).** If the full period
lifts F_3(37) from 95 to 96 or beyond, non-adjacent double padding becomes
feasible; if F_3(37) stops at 95, then combined with the corridor result every
shape is excluded and the answer is a clean no.

**My pre-registered prediction: NO double-padded run at 37->41** - the adjacent
case by proof, the rest by the spectrum margins above. This contradicts the
supply^2/gaps ~ 5 estimate, and it should: that model counts pairs without
asking whether the corridor admits the shape.

### Proposed next chunk

If the census agrees, the corridor law generalises the padding lemma from a
spectrum threshold (which expires) to a residue criterion (which does not):
"adjacent equal padded links are impossible for half of all q'". Worth doing
next: extend the corridor feasibility test to the full padded-run shape
(p links, arbitrary sizes, with literal chains attached) so the ceiling
(4+p)q' + 2s can be evaluated per q' mod 35 rather than per machine - that
would be a scale-free version of round 14's dated lemma.

## Round 16 (2026-08-18): the mod-5 AP lemma - a padding shape law that never expires

Tools: `research/corridor_shapes.py`, `corridor_ap_lemma.py`.

### The lemma (gear 5 alone, scale-free)

Gear 5 exposes only 3 of its 5 residues: every opening has k mod 5 in {0,2,3}
(teeth at 1 and 4). Four terms of an arithmetic progression with common
difference coprime to 5 occupy four DISTINCT residues mod 5. Three residues
cannot hold four. Hence

> **AP LEMMA.** No run of openings ever contains FOUR openings in arithmetic
> progression with common difference q' - for every prime q' > 5.

Verified exhaustively over all (r, g) mod 5 with g invertible: zero exceptions.

### What it forbids

Alternating literal links come in pairs summing to q' (minimally s and q'-s),
so a p=2 run with j=2 literal links between its padded links has offsets

    0, q', q'+v, 2q', 3q'   -  which CONTAINS the 4-term AP {0, q', 2q', 3q'}.

So **j = 2 is impossible for every q'**, unconditionally. The same AP appears
in three mutually adjacent padded links, so **p = 3 all-adjacent is impossible**
too. Exhaustive residue check over all 840 invertible (g, v) pairs mod 35:

    j = 0 : feasible for 50% of pairs      (round 15's coin-flip, confirmed)
    j = 1 : feasible for 32%
    j = 2 : feasible for 0%   - ALWAYS IMPOSSIBLE
    j = 3 : feasible for  4% of abstract pairs, but 0 of 546 actual primes
            11..4000 (v = s or q'-s is tied to q', not free)
    j = 4 : feasible for 0%   - ALWAYS IMPOSSIBLE

and feasibility is a function of q' mod 210 (42 distinct residues, zero
clashes), matching the Constructor's word-list modulus.

> **SHAPE LAW.** Two padded links in one run can only be separated by j = 0 or
> j = 1 literal links. Verified for every prime to 4000; j = 2 and j = 4 proven
> outright by the AP lemma.

This is the answer to "does the ceiling hold past 37->41 by structure": **yes**.
Round 14's threshold F_2(M) < 2q' expired at 37->41 because it was a spectrum
condition; the shape law is a gear-5/7 residue fact and never expires. With the
count cap p <= F/q' + alpha/3 and j in {0,1}, the padded-run shape family is
finite and scale-free, and span <= (4+p)q' + 2s stands on structure.

### (2) The knife-edge: NO, the corridor cannot settle it

Honest negative. The j=1 shape at 37->41 has two variants:

    literal 14: offsets 0, 41, 55, 96  -> mod 35 [0,6,20,26]  phases 12, 32 OK
    literal 27: offsets 0, 41, 68, 109 -> mod 35 [0,6,33,4]   IMPOSSIBLE

so the cheap variant survives the corridor, and the census question still turns
on F_3(37) >= 96 against a prefix of 95. What the corridor DID do is kill the
expensive variant, which is why the surviving threshold is exactly 96 and not
109 - the knife-edge is sharp *because* the corridor removed the alternative.

### (3) Predictions banked

    step     j=0 (adjacent)          j=1                       j>=2
    37->41   corridor IMPOSSIBLE     needs F_3(37) >= 96       impossible
    41->43   corridor OK, needs      needs F_3(41) >= 100      impossible
             F_2(41) >= 86
    43->47   corridor OK, needs      needs F_3(43) >= 110      impossible
             F_2(43) >= 94

F(37) = 88 already, so F(41) > 88 and F_2(41) >= F(41) > 86: the adjacent
shape at 41->43 is comfortably above threshold. F(43) = 103 (corpus
F(2,43) = 309 = 3 x 103), so F_2(43) >= 103 > 94, likewise clear.

> **BANKED PREDICTION: the first double-padded run appears at 41->43, not at
> 37->41.** At 37->41 the adjacent shape is corridor-forbidden and the only
> survivor is a one-unit spectrum question; at 41->43 the adjacent shape is
> corridor-allowed and the spectrum is not close.

### Proposed next chunk

The AP lemma is a two-line kernel target of exactly the shape the Formalist has
been taking (gear-5 residue arithmetic, no analysis): "openings have k mod 5 in
{0,2,3}; four terms of a q'-AP are four distinct residues mod 5; therefore no
four openings in q'-AP". Its corollary - j = 2 impossible, p = 3 all-adjacent
impossible - is the first padding bound that is scale-free. Alternatively:
extend the AP lemma to gear 7 (which exposes 5 of 7) to see whether SIX
openings in q'-AP are forbidden, which would cap padded structure further.

## Round 17 (2026-08-18): the corridor is complete at mod 35 - and p <= 2 is NOT provable

Tools: `research/corridor_complete.py`, `padding_onset.py`.

### (1) COMPLETENESS LEMMA - and the j=1 shape is genuinely feasible

> **Lemma.** A shape with n openings can be blocked by gear q only if q <= 2n.
> Gear q has two teeth, so it forbids at most 2n phases out of q; if 2n < q,
> some phase always survives. Constraints from distinct gears are independent
> by CRT, so the joint feasible set is the product of the per-gear sets - a
> shape is corridor-feasible iff it is feasible gear by gear.

Consequences: n = 4 or 5 -> only gears 5 and 7 can block, so **the mod-35 test
IS the entire corridor** and no larger modulus can ever help. Gear 11 first
enters at n = 6, gear 13 at n = 7.

The 37->41 j=1 shape has n = 4 openings (0, 41, 55, 96), and every gear leaves
phases: 5 -> 1/5, 7 -> 2/7, 11 -> 7/11, 13 -> 5/13, 17 -> 10/17, ... So

> **The j=1 shape is GENUINELY FEASIBLE. No corridor at any modulus kills it.**

Retroactive bonus: every shape analysed in rounds 15-16 had n <= 5, so those
mod-35 verdicts were already complete - the coordinator's mod-385/1155 question
has a clean structural answer rather than needing computation.

### (2) The first unobstructed step is 41->43, and it is FORCED

A shape is unobstructed iff corridor-feasible AND spectrum-affordable (its cost
<= F_j(M), necessary because the run's gaps are consecutive gaps of M):

    step      shape  cost  need   have   corridor  verdict
    19->23    j=0      46  F_2      31   OK        short by 15
    19->23    j=1      54  F_3      35   OK        short by 19
    23->29    j=0/1  58/68  -        -   NO        corridor EXCLUDES
    29->31    j=0/1  62/72  -        -   NO        corridor EXCLUDES
    31->37    j=0      74  F_2      68   OK        short by 6
    31->37    j=1      86  F_3      85   OK        short by ONE
    37->41    j=0      82  F_2       -   NO        corridor EXCLUDES
    37->41    j=1      96  F_3    >=95   OK        short by ONE
    41->43    j=0      86  F_2       ?   OK        see below
    43->47    j=0      94  F_2       ?   OK        -

F is **monotone in the machine** - adding a gear only deletes openings, so gaps
only grow - hence F(41) >= F(37) = 88; and F_2 >= F always. Therefore
F_2(41) >= 88 > 86 = 2 x 43. Combined with corridor feasibility at q' = 43:

> **41->43 is the first step with no obstruction of any kind, and the spectrum
> side is GUARANTEED rather than merely likely.** (Feasible is not the same as
> occurring: this removes every obstruction, it does not construct the run.)

**Near-miss worth recording:** the j=1 shape misses by EXACTLY ONE at two
consecutive steps - 31->37 needs 86 against F_3(31) = 85, and 37->41 needs 96
against F_3(37) >= 95. Two one-unit misses in a row. I have no mechanism for
that and flag it as an observation, not a law.

### (3) p <= 2 is NOT provable - honest negative

First, the AP lemma generalises usefully:

> **GENERALISED AP LEMMA.** Four openings at pure q'-multiples i*q' whose four
> values of i are DISTINCT mod 5 are impossible. (Round 16's lemma is the case
> i = 0,1,2,3.)

Applied to three padded links with j-patterns (j1, j2), j1, j2 in {0,1}:

    (0,0): pure multiples i = {0,1,2,3}  - 4 distinct mod 5  -> IMPOSSIBLE
    (1,1): pure multiples i = {0,1,3,4}  - 4 distinct mod 5  -> IMPOSSIBLE
    (0,1): pure multiples i = {0,1,2} only - lemma silent
    (1,0): pure multiples i = {0,1} only   - lemma silent

and the two survivors are corridor-feasible for 4 of 27 primes tested, **first
at q' = 43** (also 47, 103). So p = 3 is structurally permitted from 41->43 on,
and **p <= 2 does not follow from the AP lemma.**

Consequence for the ceiling, stated honestly: the shape family is genuinely
constrained (j in {0,1} between consecutive padded links; the (0,0) and (1,1)
triples dead at every scale), but p itself is capped only by the arithmetic
count bound p <= F/q' + alpha/3, which grows with F/q'. So

    span <= (4+p)q' + 2s   with p <= F/q' + alpha/3   =>   span <= F + O(q'),

not O(q'). My round-16 phrasing "the ceiling stands on structure" was too
strong: the SHAPE law is permanent, the COUNT is not - and the count is what
the ceiling constant depends on. Corrected here rather than left standing.

### Proposed next chunk

The count is now the whole question, and it is a spectrum question again:
p padded links cost >= p*q' and occupy p + (literals) consecutive gaps, so
p <= F_{p+literals}(M)/q'. The AP lemma kills the cheap (all-adjacent and
(1,1)) arrangements, which forces the surviving p=3 shapes to spend literals -
so the interesting quantity is the CHEAPEST surviving p-shape as a function of
p, which is a finite computation per q' mod 210. If that cost grows faster than
F_j(M), p is capped structurally after all.

## Round 18 (2026-08-18): the exposed-set autocorrelation - a construct for the erraticity

Back in the lateral lane, and folding in the human's directive: build a NEW
object out of RELATIONSHIPS between the machine's parts, and treat
"arithmetically selected, no smooth law" as a target rather than a verdict.
Tools: `research/exposed_autocorr.py`, `residual_demand.py`, `autocorr_fit.py`,
`openings_ap.py`, `openings_ap2.py`. (Tooling note: switched to
`.venv/Scripts/python.exe` per the user.)

### The new object: GEAR x LAG

Everything the search has measured lives on an object we already had a name
for. The unmeasured relationship here is **gear against lag**. For gear q let
A_q = Z_q minus its two teeth be the exposed set (|A_q| = q-2), and define its
**autocorrelation at lag g**:

    c_q(g) = |{ r in A_q : r + g in A_q }|
           = the number of phases keeping BOTH ends of a lag-g pair exposed.

**Closed form** (derived from the tooth geometry, then brute-force verified
over gears 5..31 at all lags, 0 mismatches):

    c_q(g) = q - 2   if q | g                (both ends on the same tooth pattern)
           = q - 3   if g = +-2u_q  (mod q)  (opposite teeth - the LITERAL-LINK lag)
           = q - 4   otherwise               (generic)

The content is that **the three cases of the autocorrelation are exactly the
three tooth-relationships** - and the middle case is precisely the literal-link
condition that the padding work spent rounds 12-17 on, arrived at here from a
completely different direction. Gear 5 (u=1, 2u=2) gives c = 3 / 2 / 1: lags
= +-1 mod 5 are suppressed by a factor of THREE by gear 5 alone.

### What it explains

For a lag-g pair the number of admissible endpoint phases mod 35 is **exactly
c_5(g) * c_7(g)**, ranging over {3, ..., 15} - a five-fold swing driven entirely
by the two smallest gears. Measured against full-period gap histograms:

    machine 23:  g      24    25    26    27    28    29    30    31
                 count   0  1404   310   170   322     6   112    20
                 c5*c7   3     9     4     6    10     3    12     3

The notorious cases fall out. Gap 24 is absent at machines 19 AND 23 and has
the **minimum possible** c_5*c_7 = 3. Gap 29, sitting at count 6 between
neighbours at 322 and 112, also has c_5*c_7 = 3. Of the four gap values absent
below F across the three machines, three carry the minimum value 3.

Quantified honestly, by regressing log(count) on g (the smooth decay) with and
without log(c_5 c_7):

    machine 19:  R^2 0.449 -> 0.463   ( 3% of residual variance,  9 points)
    machine 23:  R^2 0.856 -> 0.896   (28% of residual variance, 19 points)
    machine 29:  R^2 0.913 -> 0.934   (24% of residual variance, 27 points)

> **"No smooth law, only the histogram" is not right.** There is a law; it is
> multiplicative and arithmetic rather than smooth, it has a three-line closed
> form, and it accounts for about a quarter of what was being called noise.

### The second construct, and what it rules out

The autocorrelation is only the ENDPOINT half. The other half is the interior:
a gap of exactly g needs all g-1 interior slots killed. Residual demand

    D(g) = min over admissible phases of #{interior slots exposed to 5 and 7}

is what must be bought from gears >= 11. Measured against the supply
sum 2*ceil((g-1)/q): **slack is positive at every g at both machines (8 to 16
spare kills)**. So the absences are NOT capacity-structural - gap 24's absence
is arithmetic selection plus rarity, not impossibility. That is a clean
negative and it tells the next investigator not to look for a covering
obstruction.

### Side result from the same round: the openings AP theorem

Round 16's AP lemma was stated for difference q'. It generalises to arbitrary
difference: gear q leaves q-2 residues, and an L-term AP with gcd(d,q)=1
occupies min(L,q) distinct residues, so L > q-2 forces a tooth. Hence

> **an AP of L openings has common difference divisible by every gear
> q < L + 2** - so 3 consecutive equal gaps require 5 | g, 5 equal gaps require
> 35 | g, 9 require 385 | g, and an AP of L >= y+2 openings needs difference at
> least the full primorial P(y).

Verified on full periods of machines 13, 17, 19, 23, 29: zero violations, and
the longest run of equal consecutive gaps is 3-4 at every machine **with
g = 5 exactly in every case** - the theorem's minimal witness, realised.

### Honest limits, and the construct that would be needed next

The autocorrelation explains ~1/4 of the residual erraticity. The rest lives in
the interior condition, and the reason I did not build that construct this
round is specific, not fatigue: **the interior condition does not factorise over
gears.** Endpoint exposure is a conjunction ("open at 0 AND at g"), so it splits
into a product over q by CRT - which is exactly why c_q(g) exists in closed
form. The interior condition is a DISJUNCTION per slot ("killed by SOME gear"),
and disjunctions do not factorise. The object that would be needed is the
**gap correlation function of the machine** - the full (g+1)-point correlation
with 2 exposed ends and g-1 covered interiors - and it requires
inclusion-exclusion over which gear kills which interior slot rather than a
per-gear product. That is a genuinely bigger construct and it is where the
complexity actually lives; it is the right next build, not a wall.

Two further relationships in the same family, unbuilt and cheap to start:
higher-order autocorrelations c_q(g_1, g_2) (gear x lag-pair - which would give
the same treatment to the flank sums that part (D) needs, since FS is a
two-lag object), and the autocorrelation of the exposed set against the
PADDED lag q' itself, which is where the two threads would meet.
