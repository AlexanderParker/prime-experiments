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
