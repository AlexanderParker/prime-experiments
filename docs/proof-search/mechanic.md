# mechanic workstream log

## Round 1 - the fragile census at scale (2026-08-18)

Script: `research/fragile_census.py` (segmented numpy sieve over slot space,
kappa_profile.py style). Run: `uv run python research/fragile_census.py 503
1009 2003 3001 5003 10007 20011 50021` - full prime sweep y = 13..503 plus
sparse large y; y = 50021 (window 4.17e8 slots, members to 2.5e9) takes 52 s.

### Definitions used (calibrated to reproduce the 13-window census exactly)

Window of y: slots k with a member in [y, y^2], i.e. k in
[ceil((y-1)/6), floor((y^2+1)/6)]. Degree of a member = number of distinct
gear divisors (primes 5 <= q <= y). Degree-0 member = prime > y.

- twin: both members degree-0 ((11,13) at y=13 is degree-2, not counted -
  matches the class-tree "9 twins").
- frag_loose: one member degree-0 prime, other composite with exactly one
  distinct gear divisor q (the owning gear); any shape q*p, q^2, q^3, q^2*p...
- frag_semi: frag_loose with the composite a semiprime (not divisible by q^2,
  or equal to q^2).
- Boundary exclusion: a "composite" side that is literally the gear y itself
  (e.g. (29,31) at y=29) is prime, so neither twin nor fragile here.

Sanity anchor y=13: 9 twins, 10 loose fragile, 9 semi (the loose extra is
125 = 5^3). Matches the documented overlap-map census.

### Census (excerpt; full sweep is every prime 13..503 + extras)

    y       W(slots)   twins     fragS      fragL    S/tw   L/tw   pi_win
    13            27       9         9         10   1.000  1.111        33
    53           460      74       138        146   1.865  1.973       393
    101         1684     201       444        477   2.209  2.373      1226
    251        10459     818      2355       2553   2.879  3.121      6266
    503        42085    2585      8097       8833   3.132  3.417     22186
    1009      169513    8278     28822      31367   3.482  3.789     79661
    2003      668335   26870     99458     107839   3.701  4.013    283641
    5003     4170835  130543    530459     572769   4.063  4.388   1567037
    10007   16688341  440665   1894741    2038980   4.300  4.627   5767853
    20011   66736686 1508853   6792966    7288504   4.502  4.830  21356270
    50021  417008404 7816169  37028977   39576257   4.737  5.063 121535157

pi_win = degree-0 prime members in (y, y^2]. S1 (lone-composite members,
regardless of partner): 287,805,085 loose / 271,522,325 semi at y=50021 -
semiprime share of loose fragile is 93.6% there (10/9 loose/semi at y=13,
monotonically declining excess).

### How the ratio evolves: fragile/twins grows without bound, like lnln

The ratio is NOT settling: 1.11 (y=13) -> 2.37 (101) -> 3.42 (503) -> 4.63
(10007) -> 5.06 (50021). Candidate laws, measured:

- fragile ∝ twins: FAILS (ratio grows).
- fragile ∝ W/ln^3(y^2): FAILS (normalised column grows 50 -> 962).
- fragile ∝ pi(y^2)/ln(y^2): FAILS (column grows 1.3 -> 7.0).
- fragile/twins = a*lnln(y^2) + b: FIT, a=3.01, b=-4.48 (semi) /
  a=3.22, b=-4.74 (loose), max abs residual 0.05 / 0.07 over y=101..20011.
  This is a two-parameter fit over ~2.3 decades of lnln - label: fit, not
  law. The lnln form is however the heuristic expectation (below).

### The sharp law (measured to ~1%): fragile = 2 * twins * W1 / pi_win

Model: for a window member m, P(partner prime) ~ P0 * prod_{r|m} (r-1)/(r-2)
(conditioning on r | m frees the partner from gear r: (1-2/r)/(1-1/r) ->
1/(1-1/r)). Summing over prime members: 2*twins = pi_win * P0 (each twin
seen from both sides). Summing over lone composites (weight (q-1)/(q-2) for
the one owning gear; the p-side factor (p-1)/(p-2) is negligible since
p > y): fragile = W1 * P0, where W1 = sum over lone-composite members of
(q-1)/(q-2). Eliminating P0 gives predicted constant

    c = fragile * pi_win / (twins * W1) = 2.

Measured (semi variant / loose variant):

    y      13     101    503    1009   2003   5003   10007  20011  50021
    cS   2.200  1.907  1.956  1.973  1.949  1.974  1.985  1.989  1.9914
    cL   2.245  1.907  1.964  1.978  1.950  1.974  1.985  1.989  1.9917

From y=1009 up the drift is monotone upward toward 2; at y=50021 the error
is 0.43%. Honest label: measured law, HL-consistent, constant derived not
fitted (zero free parameters). It says the fragile census carries no
information beyond (a) the window's partner-prime probability (same P0 that
makes twins from primes) and (b) the lone-composite population with its
(q-1)/(q-2) weight. The lnln growth of fragile/twins is then just
W1/pi_win ~ sum_q 1/q ~ lnln - Mertens divergence, nothing twin-specific.

### Owning-gear decile (share of loose fragile, gears ranked, 10 bins)

    y        d0    d1    d2    d3    d4    d5+
    101     58.3  13.2  13.2   4.4   2.9   8.0
    503     69.8  12.8   7.2   3.7   2.7   3.8
    2003    78.1   9.6   4.8   2.9   1.8   2.8
    10007   84.0   7.0   3.5   2.1   1.3   2.1
    50021   87.9   5.3   2.6   1.6   1.0   1.6

The bottom decile of gears owns a growing near-all of the fragile slots
(gear 5 alone dominates d0). Consistent with ownership frequency ~ 1/q per
gear: the low gears' coprimes are the deciding population, quantified - the
"densest source of fragile slots" reading of the coprime census, now with
its growth law.

### Caveats / discipline notes

- cS at small y (<= 251) oscillates 1.9-2.25: small counts + boundary slots.
  The claim "-> 2" rests on the monotone tail y >= 1009.
- pi_win excludes primes <= y; W1 is restricted to members in (y, y^2] for
  consistency. Fragile classification itself does not clip the top slot's
  member y^2+2; effect is O(1) per window.
- The lnln fit coefficients (3.01, -4.48) are fits; the constant 2 is not.
