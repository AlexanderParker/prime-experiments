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
Max stride: mean diff +0.08 +- 0.07. Nothing.

### Verdict (brutal)

1. **The counting route through tooth-sharing is closed.** The effect is real,
   exactly quantified, and orders too small: net survivor gain from sharing is
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
14759L 14767R 14771L 14779R 14783L, alternating sides, no twins) at every y.
The L=100 record sits at absolute slot ~31,350 at every y. The frontier is a
property of the integers; the window only truncates it from below (s0 ~ y/6).

**Renewability check** (does the ceiling-touching survive when the landmarks
exit the window?): restricting run starts to depth >= 0.1 and >= 0.5 of the
window still gives saturated (load-1) runs of length 9-12 at every scale
(e.g. y=10007: L*=11 at members ~1.9e7, L*=10 at members ~5.1e7). So gap(L)=0
for L <= ~10 is renewable at ALL depths tested, and L <= 13 near the bottom.
Caveat, labelled: nothing known FORCES saturated runs to persist at all
depths forever - that persistence is itself a prime-constellation statement
(HL-admissible, so expected true, and unprovable by current technology).

### Where the gap is narrowest - the target scale

gap(L) = 0 for L <= 13; opens at L = 14 with 1/14 and stays < 0.29 through
L = 32. **The compression-bound target is L ~ 14-32**: reality hugs the
X-ceiling there, renewably, at every depth. For L >= 63 the gap exceeds 0.44
and the bound would be fighting a phantom - reality never gets close.

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
