# Branch R4.a - THE TOP MACHINE AT THE PERIOD SCALE

Parent: R4 (the owner's reframing, 2026-09-06). Spawned by the observation that the machine
{5..q} opens exactly prod(g-2) twin slots per period P and that a twin is exactly one of those
openings that also survives the primes above q. The window is dropped; the whole period is the
arena.

Scripts research/anchor235/r59/, results research/anchor235/r59/results/ (gitignored; every
number this document relies on is written out here).

---

## 0. Vocabulary (the owner's, 2026-09-06)

There are TWO MACHINES on the same track, built the same way: a gear per prime, teeth at
k = +- 6^{-1} (mod g).

- **BOTTOM machine** = the gears {5..q}. Period P = prod_{5<=g<=q} g columns (6P consecutive
  numbers). Its openings over one period are the **twin candidates**: exactly
  N = prod (g-2) of them (CRT, proved).
- **TOP machine** = the gears above q. Only the gears up to Z = isqrt(6P+1) act on columns
  below P, so the top machine on this track is the primes in (q, Z].
- **Home column** of a top gear g: the column that holds the prime g itself,
  h(g) = round(g/6). Since g = 6 h(g) -+ 1, the home column is the least nonnegative
  representative of g's own striking class: h(g) = u_g mod g if g = 5 (mod 6), else
  h(g) = -u_g mod g. A top gear is **placed** on the track at its home column.
- **Proper teeth.** A top gear's strike on its home column is NOT a kill: the member it
  divides there is the prime g itself. The top machine's teeth are the PROPER multiples: gear
  g kills column k iff k = +- u_g (mod g) and k != h(g).
- **Doubly occupied placement**: a column carrying two top gears, i.e. a twin prime pair
  inside the top machine's own range.

Everything below uses bottom/top. "Strike" = a top gear meeting a bottom-open column;
"kill" = a strike that is not the gear's home column.

---

## 1. Pre-registered (written before any computation)

### 1.1 The theory

At the period scale the bottom machine is exact and the top machine's action on it is exactly
describable.

- **T1 (exact first moments).** Every top gear g strikes the bottom machine's openings at
  exactly the rate 2/g over a full period, with absolute error below the
  inclusion-exclusion term count 3^m (m = number of bottom gears) - the stride result
  (docs/proof-search/lower-sieve.md section 5) applied gear by gear.
- **T2 (the twisted copies).** A top gear g strikes a bottom-open column k iff, writing the
  struck member as g m, the cofactor m is q-rough and the partner g m -+ 2 is q-rough. In the
  coordinate m this is exactly the opening set of a TWISTED BOTTOM MACHINE: teeth at
  m = 0 (mod h) and m = -+ 2 g^{-1} (mod h) for every bottom gear h. So the top machine's
  action on the bottom machine's openings decomposes into 2(pi(Z) - pi(q)) coherent twisted
  copies of the bottom machine, one per (top gear, side), each with prod(h-2) openings per
  period.
- **T3 (level of distribution 1).** Joint strike counts of any set of top gears are exact
  (error below the same 3^m scale) while the product of the gears stays below P.
- **T4 (the exact twin identity).** For every column k with q/6 < k < P:
  k is a twin pair **iff** k is bottom-open and top-open (proper teeth). Nothing else is
  needed; no window, no localisation.
- **T5 (placement).** The top gears are placed on the track at their home columns, all inside
  the first Z/6 columns of the period. A home column is bottom-open iff the partner g -+ 2 is
  q-rough, which forbids exactly ONE residue class per bottom gear: the placement set is the
  opening set of a ONE-TOOTH machine (dimension 1), while double occupancy is the two-tooth
  object (dimension 2).

### 1.2 Predictions, with numbers, and what refutes each

| # | prediction | refuted by |
|---|---|---|
| **P1** | For every top gear g and every q in {11,13,17,19,23}, the strike count on the bottom machine's openings over [0,P) satisfies \|X_g - 2N/g\| < 2 * 3^m (m = 3,4,5,6,7 -> bounds 54, 162, 486, 1458, 4374). | any g with deviation at or above the bound |
| **P2** | The twisted-copy identity holds with **0 mismatches**: for every top gear g and each side, the set of cofactors m of g's strikes is exactly the opening set of the twisted machine with teeth {0, -+2 g^{-1}} restricted to the m-range and to the forced class mod 6; that twisted machine has exactly N openings per m-period 6P. | one mismatching m at any g |
| **P3** | Survivor curve. S(Z)/(N_range * prod_{q<g<=Z}(1-2/g)) = 1/(4 e^{-2 gamma}) = 0.7925 to within 0.02 at every q (s = 2); ratio 1.00 to within 0.03 at s >= 4.27. With PROPER teeth, S(Z) = the exact twin count in the range, 0 mismatches; with improper teeth it is short by exactly the number of twins having a member <= Z. | ratio outside those bands; any mismatch in the twin identity |
| **P4** | Joint pair moments: \|X_{g,h} - 4N/(gh)\| < 2 * 3^m for every pair with gh < P; beyond that the absolute deviation stays O(1) while the prediction 4N/(gh) falls below 1, so the relative error diverges. The prediction crosses 1 already at gh = 4N < P. Triples the same with 8N/(ghk). | an absolute deviation above the bound at gh < P |
| **P5** | Bilinear / switching. The fraction of strikes whose cofactor m is prime equals the sieve prediction pi(X)/Phi(X,q) to within 5%, because the twisted condition costs primes and q-rough numbers the same factor prod(1 - 1/(h-1)). The switching identity (ordered prime-cofactor strikes with m <= Z) = 2 * (unordered both-prime pairs) + (number of top gears whose square is a member of a bottom-open column) holds with 0 mismatches. | fraction outside 5%; any mismatch |
| **P6** | Brun truncations. Order 1 is a lower bound that goes negative once sum 2/g exceeds the slack; order 2 an upper bound, order 3 a lower bound; and because the pair terms are EXACT (P4), the order-2 error equals the order-3 term to within 10% while all products stay below P. | order-2 error more than 10% from the order-3 term in that regime |
| **P7** | Nothing measured contradicts "face A alone": no measured quantity at the period scale distinguishes the real bottom machine's survivor curve from generic dimension-2 sieve behaviour except the classical s-dependent constant. | a departure of the measured ratio from a function of s alone |
| **P8** | Placement. Every top gear's home column lies in [1, Z/6]; the fraction of top gears whose home column is bottom-open equals prod_{5<=h<=q}(1 - 1/(h-1)) to within 5% (0.750, 0.600, 0.540, 0.506, 0.482 at q = 11..23). The placements avoid exactly one class per bottom gear (one-tooth machine), verified with 0 exceptions. | a placed gear outside the prefix; a home column struck on the gear's own side |
| **P9** | Double occupancy. The number of doubly occupied placements equals the number of twin pairs in (q, Z], and its Hardy-Littlewood prediction 2 C_2 Z/(ln Z)^2 is met to within the usual accuracy. No q in the range has zero doubly occupied placements. | a q with no doubly occupied placement |
| **P10** | The owner's question, stated exactly: "the top machine has a doubly occupied placement for every q" is EQUIVALENT to the infinitude of twin primes (both directions), and it is weaker than the window statement, since Z = isqrt(6P) is exponentially larger than q. | a proof either way (not expected); an error in the equivalence |

### 1.3 Scorecard

| item | verdict |
|---|---|
| P1 first moments exact to 3^m | **CONFIRMED**, 0 exceptions in 2,338 (q, gear) cells; maxima 1.69 / 5.41 / 7.83 / 14.29 / 25.20 against bounds 54 / 162 / 486 / 1,458 / 4,374, and the true growth is 2^m, not 3^m (3.7) |
| P2 twisted-copy identity, 0 mismatches | **CONFIRMED**, 4,676 copies, 17,035,903 cofactors, 0 mismatches (3.6) |
| P3 survivor curve, the s = 2 handicap | **PARTLY REFUTED**: the twin identity is exact (0 mismatches at five q) but the ratio at s = 2 is 1.0774, 0.9998, 0.9462, 0.9229, 0.8926, not 0.7925 +- 0.02; it approaches the classical constant as 0.79305 (1 + c/ln Z) with c = 1.39, 1.34, 1.27, 1.32, 1.21 (3.8) |
| P4 joint moments, level of distribution | **CONFIRMED**, 0 exceptions in 19,956 pairs with gh < P and in the 13 sampled triples with ghr < P; transition measured (3.7) |
| P5 switching / bilinear | **CONFIRMED**: E = 2D + Q exact at all five q; prime-cofactor fraction within 0.01% of the sieve prediction at q = 23 (3.10) |
| P6 Brun truncations | **PARTLY CONFIRMED**: the order-2 error equals the order-3 term to 0%, 5.9%, 10.6% at z = 40, 60, 100 and to 17.8%, 22.3% at z = 200, 400 (3.9) |
| P7 face A alone | **CONFIRMED** for every sieve statistic; the clutch adds one thing the sieve view does not have (3.3, 3.11) |
| P8 placement geometry | **CONFIRMED with one tolerance miss, and strengthened**: 0 own-side violations in 15,549 checks (the one-tooth condition), and the bottom-open fraction is within 6.7%, 2.7%, 3.4%, 3.3%, 0.6% of prod(1 - 1/(h-1)) - outside the pre-registered 5% only at q = 11. The working expectation that home columns are equidistributed over the permitted classes is REFUTED and replaced by an exact 2 : 1 law (3.5) |
| P9 double occupancy | **CONFIRMED**: 3, 9, 26, 78, 268 doubly occupied placements = the twin pairs in (q, Z] at every q; all bottom-open; no triples |
| P10 equivalence | **PROVED** in 4.4; and the placement statement is weaker than the window statement |

Stop rules pre-registered: any sub-question that reduces to re-deriving the fundamental lemma,
Brun's theorem, Chen's theorem or the Selberg parity example is stopped at the first sign and
recorded in one line under Dead ends.

---

## 2. Setup (exact definitions)

Column k is the pair (6k-1, 6k+1). Gear g (a prime >= 5) strikes column k iff
k = +- 6^{-1} (mod g); write u_g = 6^{-1} mod g, so g | 6k-1 iff k = u_g and g | 6k+1 iff
k = -u_g (mod g).

- **Bottom machine** {5..q}: m = pi(q) - 2 gears, period P, N = prod(g-2) openings per period.
- **Top machine**: the primes in (q, Z], Z = isqrt(6P+1), built the same way, every gear
  starting at column 0, **no exemptions**. Its own period is the product of its gears, which is
  astronomically beyond the range, so over [0, P) it is not periodic at all.
- **Home column** of a top gear g: h(g) = round(g/6), the column holding the prime g itself. A
  top gear is **placed** there. A **home strike** is a top gear meeting its own home column; the
  member it divides there is the prime itself.
- **Clutch**: every column of [0, P) classified by the pair (bottom state, top state): both
  open, bottom-open/top-closed, bottom-closed/top-open, both closed.
- **Twisted machine.** For a top gear g and a side eps in {+1, -1}, M^(g,eps) is the set of
  integers m with m != 0 (mod h) and m != -eps * 2 g^{-1} (mod h) for every bottom gear h, in
  the class m = -eps * g^{-1} (mod 6). Same shape as the bottom machine but with **separation
  2 g^{-1} instead of the machine's own 2 * 6^{-1}**: the coherent family c/r of W3
  (separation_drives_K.md N-S2) at c = 2, r = g.

  Derivation, one line each. If g | 6k-1 write 6k-1 = g m; then h | g m iff m = 0 (mod h) and
  h | g m + 2 iff m = -2 g^{-1} (mod h). If g | 6k+1 write 6k+1 = g m; then h | g m iff m = 0
  and h | g m - 2 iff m = +2 g^{-1}.

| q | bottom gears m | P (columns) | N = prod(g-2) | 6P (numbers) | Z = isqrt(6P+1) | top gears | k_min |
|---|---|---|---|---|---|---|---|
| 11 | 3 | 385 | 135 | 2,310 | 48 | 10 | 3 |
| 13 | 4 | 5,005 | 1,485 | 30,030 | 173 | 34 | 3 |
| 17 | 5 | 85,085 | 22,275 | 510,510 | 714 | 120 | 4 |
| 19 | 6 | 1,616,615 | 378,675 | 9,699,690 | 3,114 | 435 | 4 |
| 23 | 7 | 37,182,145 | 7,952,175 | 223,092,870 | 14,936 | 1,739 | 5 |

k_min = floor((q+1)/6) + 1 is the first column both of whose members exceed q. Everything is
computed over a FULL period by direct enumeration, never by sampling. Scripts:
`period_core.py` (moments, twisted copies, survivor curve, switching, placement),
`period_clutch.py` (the top machine's own pattern and the four cells),
`clutch_patterns.py` (the placement law, twin gaps, the first kill, the origin).

---

## 3. Results

### 3.1 The top machine's own pattern over [0, P)

Built independently of the bottom machine, over the same columns.

| q | top-open columns | density | prod_{q<g<=Z}(1 - 2/g) | ratio | longest open run | longest closed run | where | longest closed run above the placement prefix |
|---|---|---|---|---|---|---|---|---|
| 11 | 179 | 0.464935 | 0.436373 | 1.0655 | 7 | 7 | col 0 | 7 |
| 13 | 1,629 | 0.325475 | 0.307356 | 1.0590 | 5 | 27 | prefix | 24 |
| 17 | 20,322 | 0.238844 | 0.218553 | 1.0928 | 6 | 114 | col 21 | 30 |
| 19 | 289,813 | 0.179272 | 0.164156 | 1.0921 | 7 | 378 | col 142 | 58 |
| 23 | 5,046,102 | 0.135713 | 0.126197 | 1.0754 | 8 | 1,376 | col 1,147 | 104 |

Four facts, all exact:

1. **It is not periodic over the range, and it is not uniform.** Its density falls
   monotonically across the period: at q = 23 the twenty block densities are 0.1727, 0.1593,
   0.1515, 0.1466, 0.1431, 0.1401, 0.1376, 0.1357, 0.1338, 0.1323, 0.1306, 0.1296, 0.1284,
   0.1272, 0.1265, 0.1254, 0.1248, 0.1237, 0.1229, 0.1224 - a 41% fall from the first block to
   the last. The bottom machine's block densities are constant to six figures (0.2139 twenty
   times), because P is its period. Tested directly: the top pattern agrees with its own
   translate by a bottom subperiod (5, 35, 385, 5005, 85085, 1616615) at 0.7709, 0.7722,
   0.7706, 0.7715, 0.7726, 0.7752 against the independent expectation 0.7719 - exactly chance,
   no periodicity at any bottom subperiod.
2. **Its density exceeds the independent-gear product** by 6-9% at every q (1.0655, 1.0590,
   1.0928, 1.0921, 1.0754). A column is top-open iff neither member has a prime factor in
   (q, Z], and since the range is Z^2 each member is then q-smooth times at most one prime
   above Z: a one-dimensional-looking object with a Buchstab excess, not a two-tooth sieve.
3. **Every top gear strikes exactly its fair share of the columns**: |kills - 2P/g| < 1 at
   every gear and every q (maxima 0.62, 0.66, 0.67, 0.67, 0.68 over 2,338 gears), so the kill
   count is round(2P/g), from 2,564,286 (g = 29) down to 4,981 (g = 14,929) at q = 23. There
   are no "few-strike" gears at the period scale - that is a window phenomenon; over the period
   the smallest and largest top gears differ in workload by a factor of 515, not by orders.
4. **Its open runs are short and its closed runs are long.** The open-run spectrum at q = 23 is
   1: 3,804,784; 2: 504,459; 3: 64,997; 4: 7,972; 5: 952; 6: 102; 7: 19; 8: 2 - geometric decay
   by about 7.5 per step, longest run 8. The longest closed run, 1,376, sits INSIDE the
   placement prefix and is an artefact of the home strikes (3.4); above the prefix the longest
   closed run is only 104.

### 3.2 The clutch: four cells

| q | both open | bottom-open / top-closed | bottom-closed / top-open | both closed | independence prediction for both-open | **coupling** |
|---|---|---|---|---|---|---|
| 11 | 64 | 71 | 115 | 135 | 62.77 | 1.0197 |
| 13 | 457 | 1,028 | 1,172 | 2,348 | 483.33 | 0.9455 |
| 17 | 4,607 | 17,668 | 15,715 | 47,095 | 5,320.24 | 0.8659 |
| 19 | 57,372 | 321,303 | 232,441 | 1,005,499 | 67,885.64 | 0.8451 |
| 23 | 895,791 | 7,056,384 | 4,150,311 | 25,079,659 | 1,079,213.86 | **0.8300** |

(The independence prediction is bottom density x top density x P; cells sum to P at every q.)
The couplings of the other three cells are all within 5% of 1: bottom-open/top-closed 0.983,
1.026, 1.042, 1.034, 1.027; bottom-closed/top-open 0.989, 1.023, 1.048, 1.047, 1.046; both
closed 1.009, 0.989, 0.985, 0.990, 0.993. **All the coupling of the clutch is in the both-open
cell**, and it is negative and strengthening: the two machines leave 17% fewer columns jointly
open than two independent machines of the same densities would. That deficit is the classical
s = 2 handicap seen as a correlation between two machines rather than as a sieve constant (3.8).

**The sub-split of the bottom-open / top-closed cell.** A top strike there is either a home
strike (the member is the top prime itself, so the column is a twin) or a proper strike (a real
kill):

| q | home strikes only | at least one proper strike |
|---|---|---|
| 11 | 3 | 68 |
| 13 | 9 | 1,019 |
| 17 | 26 | 17,642 |
| 19 | 78 | 321,225 |
| 23 | 268 | 7,056,116 |

The home-only sub-cell is exactly the set of doubly occupied placements (3, 9, 26, 78, 268 at
every q, 0 mismatches, 3.5): a column whose only top strikes are home strikes has BOTH members
prime and both below Z. So

    twins of the period = (both-open cell) + (home-only sub-cell)

= 64+3, 457+9, 4,607+26, 57,372+78, 895,791+268 = 67, 466, 4,633, 57,450, 896,059, verified
against an independent sieve of the members (0 column mismatches over 38,889,216 columns). The
home strike is not a nuisance to exempt: it is the clutch's marker for "this candidate is a
twin whose member the top machine happens to own".

**Runs inside the cells** (longest consecutive run / longest gap between members of the cell):

| q | both open | b-open/t-closed | b-closed/t-open | both closed |
|---|---|---|---|---|
| 11 | 2 / 24 | 2 / 22 | 5 / 14 | 5 / 10 |
| 13 | 2 / 82 | 2 / 34 | 5 / 43 | 10 / 11 |
| 17 | 2 / 153 | 2 / 36 | 5 / 120 | 17 / 10 |
| 19 | 2 / 519 | 2 / 36 | 6 / 388 | 24 / 12 |
| 23 | 2 / 2,522 | 2 / 48 | 8 / 1,416 | 29 / 12 |

Both cells that need the bottom machine open have longest run exactly 2 at every q - gear 5
alone forces it. The both-closed cell reaches the bottom machine's own record run (10, 17, 24 at
q = 13, 17, 19) and falls short at q = 23 (29 against 33): from q = 23 the top machine leaves an
opening inside the bottom machine's record stretch.

**Both-closed by which member each machine strikes** (1 = lower, 2 = upper, 3 = both), q = 23:
b1t1 2,081,938; b1t2 2,757,317; b1t3 4,073,517; b2t1 2,755,854; b2t2 2,079,757; b2t3 4,075,147;
b3t1 2,084,814; b3t2 2,085,318; b3t3 3,085,997. The table is symmetric under swapping the two
members to within 0.1% (the mirror, 3.4), and the same-member cells (b1t1, b2t2) are 25% SMALLER
than the different-member cells (b1t2, b2t1): a member already divisible by a bottom gear is
less likely to carry a second factor in (q, Z]. That is arithmetic, not a coupling between the
machines.

### 3.3 The window inside the clutch: the two machines do not touch there

A proper strike of a top gear g on a bottom-OPEN column has member g m with m q-rough and m > 1,
hence m >= the least prime above q, hence the member exceeds q * q' > q^2. So the top machine
cannot kill anything in the window. Measured:

| q | first proper kill on a bottom-open column | by gear | window top column (q^2-1)/6 | above the window |
|---|---|---|---|---|
| 11 | 28 | 13 | 20 | yes (1.40x) |
| 13 | 60 | 19 | 28 | yes (2.14x) |
| 17 | 60 | 19 | 48 | yes (1.25x) |
| 19 | 140 | 29 | 60 | yes (2.33x) |
| 23 | 140 | 29 | 88 | yes (1.59x) |

**The window is exactly the part of the period on which the clutch has no interaction**: there
the bottom-open/top-closed cell contains only home strikes, so bottom-open = twin. That is the
window lemma (docs/proofs/01-the-route.md Theorem 1) read as a clutch fact, and it says what the
lemma does not: the window is the only region of the period where one machine decides the answer
alone, and it is 2.4e-6 of the period at q = 23.

Above the window the top machine takes over and eats the bottom machine's candidates: over the
whole period it kills 0.5075, 0.6867, 0.7920, 0.8483, **0.8873** of them.

### 3.4 The shared origin

The two machines share exactly one thing: the origin. Column 0 is open in both at every q (its
members are -1 and 1; every gear's teeth are at +-u_g, never at 0), and both are built from the
same origin, so both are mirror-symmetric about it: the state of column -k is the state of
column k with the two members swapped. Verified directly, **0 mismatches** for both machines over
2,853,543 columns each side in total (2,000,000 of them at q = 23) (the bottom machine against its translate k -> P-k, the
top machine by rebuilding it on the negative columns). The clutch classification is therefore an
even function of k.

What the shared origin forces is severe and local:

| q | ceil(Z/6) (the placement prefix) | first both-open column after 0 | longest both-open gap of the period | at column | top-open density in the prefix | over the period |
|---|---|---|---|---|---|---|
| 11 | 8 | 10 | 25 | 110 | 0.1250 | 0.4649 |
| 13 | 29 | 30 | 83 | 4,070 | 0.0690 | 0.3255 |
| 17 | 119 | 135 | 154 | 31,318 | 0.0252 | 0.2388 |
| 19 | 519 | 520 | 520 | **0** | 0.0135 | 0.1793 |
| 23 | 2,490 | 2,523 | 2,523 | **0** | 0.0040 | 0.1357 |

Inside the placement prefix every top gear sits on its own home column and strikes it, so a
column of the prefix is top-open only if BOTH members are q-smooth. The top machine is therefore
almost completely shut at the origin: 0.4% density there against 13.6% over the period at
q = 23, i.e. 3% of its own average. Consequence, exact and exceptionless from q = 19: **the
longest both-open stretch of the entire period is the one that starts at the origin**, and it
ends at the first twin pair both of whose members exceed Z (columns 520 and 2,523 against
ceil(Z/6) = 519 and 2,490). So the shared origin is not a place of agreement between the
machines; it is the one place where the top machine's home strikes wipe the both-open cell out
entirely, and what they wipe out is exactly the twins with a small member.

The bottom machine returns to its origin phase at every multiple of P; the top machine never
does within any computable range. The clutch's own period is the product of every prime up to Z,
so within [0, P) the clutch never repeats and its only exact self-similarity is the mirror.

### 3.5 The placement view, and an exact law

The 2,338 top gears are placed at their home columns h(g) = round(g/6), all inside the prefix
[1, ceil(Z/6)] - 2,490 columns of 37,182,145 at q = 23.

| q | placed | distinct columns | 1 gear | 2 gears (doubly occupied) | 3+ | home column bottom-open | fraction | prod(1 - 1/(h-1)) | bottom-open prefix columns |
|---|---|---|---|---|---|---|---|---|---|
| 11 | 10 | 7 | 4 | 3 | 0 | 6 | 0.600 | 0.5625 | 3 |
| 13 | 34 | 25 | 16 | 9 | 0 | 18 | 0.529 | 0.5156 | 9 |
| 17 | 120 | 94 | 68 | 26 | 0 | 56 | 0.467 | 0.4834 | 30 |
| 19 | 435 | 357 | 279 | 78 | 0 | 192 | 0.441 | 0.4565 | 116 |
| 23 | 1,739 | 1,471 | 1,203 | 268 | 0 | 762 | 0.438 | 0.4358 | 537 |

- **A home column is never struck on the gear's own side** (the member there is the prime g and
  no bottom gear divides it): 0 violations in 15,549 (gear, bottom gear) checks. So the placement
  condition is a ONE-TOOTH condition - one forbidden class per bottom gear, not two - and the
  placement set is a dimension-1 object of density prod(1 - 1/(h-1)), measured to within 6.7%,
  2.7%, 3.4%, 3.3%, 0.6%.
- **THE PLACEMENT RESIDUE LAW (exact, new).** Home columns are NOT equidistributed over the
  permitted classes; the distribution is exactly 2 : 1. As r runs over the residues mod 6h that
  are coprime to 6 and nonzero mod h (there are 2(h-1) of them), the home column
  k = (r -+ 1)/6 mod h takes each of the h-2 non-tooth classes of gear h exactly twice and each
  of gear h's own two tooth classes exactly once. One-line proof: for r = 5 (mod 6),
  k = (r+1) u_h omits only u_h; for r = 1 (mod 6), k = (r-1) u_h omits only -u_h. Verified
  exhaustively over all residues for all 25 (machine, bottom gear) pairs, 0 exceptions; on the
  real top primes at q = 23 the per-class ratio other/tooth is 1.968, 1.999, 1.973, 1.926, 2.054,
  2.036, 2.001 at gears 5..23 against the law's 2.000. The law is exactly why the placement
  density is prod(1 - 1/(h-1)): a placement lands on a tooth class of gear h with probability
  2/(2(h-1)) = 1/(h-1).
- **A doubly occupied placement is a twin pair of top gears and is always bottom-open**: 3, 9,
  26, 78, 268 of them, equal at every q to the number of twin pairs in (q, Z], all bottom-open
  (0 exceptions in 384). No column ever carries three gears (a prime triple g, g+2, g+4 above 3
  is impossible).
- Of the 537 bottom-open prefix columns at q = 23, 268 carry two top gears, 226 carry one, and 43
  carry none - the last are the prefix columns both of whose members are products of two primes
  above q, the first at column 140 (3.3).

### 3.6 The twisted copies

For every top gear and each side the cofactor set of its strikes on the bottom machine's
openings was generated twice - once from the strikes, once from the twisted machine M^(g,eps) -
and compared element by element.

| q | (gear, side) copies | cofactors compared | mismatches |
|---|---|---|---|
| 11 | 20 | 112 | 0 |
| 13 | 68 | 1,698 | 0 |
| 17 | 240 | 33,230 | 0 |
| 19 | 870 | 676,053 | 0 |
| 23 | 3,478 | 16,324,810 | 0 |

Total 4,676 copies, **17,035,903 cofactors, 0 mismatches**. The per-period count was checked
directly at six copies per machine (30 checks, 0 exceptions): each twisted machine has exactly
N = prod(h-2) openings per m-period of length 6P, the same count as the bottom machine itself.
So the top machine's action on the bottom machine's openings is exactly a union of
2(pi(Z) - pi(q)) coherent twisted copies of the bottom machine, one per (gear, side), all at the
rational separation 2/g.

### 3.7 Moments: nothing is inexact at the period scale

**First moments.** X_g = bottom-open columns of [0,P) struck by top gear g, against 2N/g:

| q | cells | max \|X_g - 2N/g\| | at gear | pre-registered bound 2*3^m | max relative deviation | at gear |
|---|---|---|---|---|---|---|
| 11 | 10 | 1.69 | 29 | 54 | 0.2185 | 47 |
| 13 | 34 | 5.41 | 29 | 162 | 0.1525 | 163 |
| 17 | 120 | 7.83 | 193 | 486 | 0.0694 | 691 |
| 19 | 435 | 14.29 | 409 | 1,458 | 0.0402 | 3,067 |
| 23 | 1,739 | 25.20 | 2,393 | 4,374 | 0.0122 | 14,159 |

0 exceptions in 2,338 cells. **New, and sharper than the stride bound:** the measured maximum
grows like 2^m, not 3^m - max / 2^m is 0.21, 0.34, 0.24, 0.22, 0.20 across the five machines, so
the inclusion-exclusion term-count bound 3^m is loose by (3/2)^m. The relative deviation FALLS
with q (0.22 to 0.012) because the fair share grows faster than the error.

**Joint moments.** Pair counts against 4N/(gh):

| q | pairs measured | with gh < P | max \|X - 4N/(gh)\| there | exceptions above 2*3^m | with gh >= P | max dev there |
|---|---|---|---|---|---|---|
| 11 | 45 (all) | 5 | 1.19 | 0 | 40 | 1.62 |
| 13 | 561 (all) | 236 | 3.36 | 0 | 325 | 1.74 |
| 17 | 7,140 (all) | 3,670 | 6.88 | 0 | 3,470 | 2.46 |
| 19 | 15,000 | 8,084 | 12.13 | 0 | 6,916 | 2.50 |
| 23 | 15,000 | 7,961 | 16.15 | 0 | 7,039 | 2.52 |

Triples against 8N/(ghr), 4,000 sampled per machine: max deviation 0, 0, 2.01, 0.98, 0.30 for
ghr < P and 0.96, 1.70, 3.45, 2.66, 1.00 for ghr >= P; 0 exceptions.

**Where exactness stops - it does not.** The absolute deviation is bounded by a constant of the
bottom machine on BOTH sides of gh = P; what changes is the size of the prediction. Mean
absolute deviation and mean prediction by bucket of gh/P at q = 23:

| log10(gh/P) | -5 | -4 | -3 | -2 | -1 | 0 |
|---|---|---|---|---|---|---|
| pairs | 3 | 9 | 241 | 1,642 | 6,066 | 7,039 |
| mean \|dev\| | 12.14 | 5.21 | 3.77 | 2.15 | 1.00 | 0.59 |
| mean prediction | 11,640 | 1,277 | 219.9 | 25.56 | 2.70 | 0.44 |

The deviation FALLS as gh grows while the prediction falls faster. The relative error reaches 1
where the prediction crosses 1, i.e. at gh = 4N = 1.403 P, 1.187 P, 1.047 P, 0.937 P, 0.855 P -
so the crossing moves through the period between q = 17 and q = 19; it is not at gh = P.

**Reading.** Over the period there is no level-of-distribution obstruction of any kind: every
moment of every order is exact to a constant of the bottom machine alone (about 2^m against
N ~ e^q), for every modulus, every class, up to and beyond the period. Level of distribution 1,
confirmed exactly.

### 3.8 The survivor curve

S(z) = bottom-open columns in [k_min, P) that no top gear up to z strikes; s = ln(6P)/ln z; the
independent-gear prediction is N_range * prod_{q<g<=z} (1 - 2/g).

**The twin identity, exact.** At z = Z the survivors are exactly the twin pairs of the range with
both members above Z, and the twins with a member below Z are exactly the home-only cell:

| q | S(Z) | twins in [k_min, P) | difference | twins with a member <= Z | survivors that are not twins |
|---|---|---|---|---|---|
| 11 | 63 | 66 | 3 | 3 | 0 |
| 13 | 456 | 465 | 9 | 9 | 0 |
| 17 | 4,606 | 4,632 | 26 | 26 | 0 |
| 19 | 57,371 | 57,449 | 78 | 78 | 0 |
| 23 | 895,790 | 896,058 | 268 | 268 | 0 |

**The curve.** Measured ratio S(z) / (N_range * prod):

| q | s = 4.27 | s = 3.0 | s = 2.5 | s = 2.2 | s = 2.0 |
|---|---|---|---|---|---|
| 11 | 1.0000 (z = 6) | 0.9966 (z = 13) | 0.9788 (z = 23) | 0.9950 (z = 37) | 1.0774 |
| 13 | 1.0012 (z = 17) | 1.0014 (z = 31) | 0.9693 (z = 61) | 0.9304 (z = 109) | 0.9997 |
| 17 | 0.9998 (z = 23) | 1.0046 (z = 79) | 0.9566 (z = 191) | 0.9003 (z = 397) | 0.9462 |
| 19 | 0.9999 (z = 43) | 1.0116 (z = 211) | 0.9667 (z = 619) | 0.8922 (z = 1,499) | 0.9229 |
| 23 | 1.0000 (z = 89) | 1.0132 (z = 607) | 0.9708 (z = 2,179) | 0.8819 (z = 6,229) | 0.8926 |

The shape at q = 23, finely sampled: 1.0000 at s >= 4.27, 0.9995 at s = 4, 1.0004 at 3.5, 1.0132
at 3.0, 1.0105 at 2.8, 0.9889 at 2.6, 0.9708 at 2.5, 0.9453 at 2.4, 0.9143 at 2.3, 0.8819 at
2.2, 0.8685 at 2.15, **0.8603 at s = 2.09 (the minimum, z = 9,839)**, 0.8650 at 2.05, 0.8926 at
2.00. The curve is flat at 1 down to s = 3.5, bulges 1.3% above 1 near s = 3, falls to a minimum
at s ~ 2.1 and turns UP over the last few gears. The same minimum sits at s = 2.11 at q = 19
(0.8818 at z = 2,029, rising to 0.9229 at s = 2): the position of the minimum is a function of s
alone across two machines whose periods differ by a factor of 23. That oscillation is the
machine's own version of the sieve's f(s), F(s) pair. The dimension-2 lower-bound function
f_2(s) is identically ZERO for s <= beta_2 = 4.2664 (Diamond-Halberstam-Richert;
iwaniec_two_class.md) - precisely where this measured ratio is 1.0000 to four decimals.

**The 4 e^{-2 gamma} question, answered with a correction.** The pre-registered value at s = 2
was 1/(4 e^{-2 gamma}) = 0.79305. Measured: 1.0774, 0.9998, 0.9462, 0.9229, 0.8926. The
prediction is refuted at these sizes, and the reason is convergence, not the constant: fitting
ratio = 0.79305 (1 + c / ln Z) gives c = 1.39, 1.34, 1.27, 1.32, 1.21 - flat - so the ratio does
approach the classical constant at the classical rate 1/ln Z, and at q = 23 (ln Z = 9.61) it is
still 13% above it. The square-vector branch's over-count (R2.a.i.a.1.b: model over real 1.2628
against 4 e^{-2 gamma} = 1.2619) was measured at q = 50,000, far beyond the reach of an exact
period computation; the period scale reproduces the direction and the rate but cannot reach the
constant.

### 3.9 Brun on the clutch: the truncations

Inclusion-exclusion truncations of S(z) at orders 1, 2, 3 (Bonferroni: odd orders lower bounds,
even orders upper bounds), q = 23, all moduli below P:

| z | top gears | true S(z) | order 1 | order 2 | order 3 | order-2 error | order-3 term |
|---|---|---|---|---|---|---|---|
| 40 | 3 | 6,551,741 | 6,460,888 | 6,553,647 | 6,551,741 | +1,906 | 1,906 |
| 60 | 8 | 5,289,316 | 4,795,075 | 5,339,306 | 5,286,204 | +49,990 | 53,102 |
| 100 | 16 | 4,271,469 | 3,119,488 | 4,483,075 | 4,246,403 | +211,606 | 236,672 |
| 200 | 37 | 3,180,270 | 794,011 | 3,894,217 | 3,026,007 | +713,947 | 868,210 |
| 400 | 69 | 2,568,665 | -951,329 | 3,902,664 | 2,186,707 | +1,333,999 | 1,715,957 |

At z = 40 the machine has only three top gears, so order 3 is the complete inclusion-exclusion
and is exact - the order-2 error equals the order-3 term to the digit. As gears are added the
order-2 error tracks the order-3 term at 100%, 94.1%, 89.4%, 82.2%, 77.7%: the truncation error
IS the next term. The order-1 truncation is useless from the first row and negative by z = 400;
at z = 1,000 it is -3,166,759 against a truth of 1,949,456. Neither bound survives to s = 2.

**The point of this measurement.** The period scale removes every remainder term a sieve normally
carries (3.7: all moments exact). The truncated inclusion-exclusion is not improved by one
column: its error is a sum of EXACT terms of the next order, and those terms grow before they
shrink, because sum_{q<g<=Z} 2/g = 2(ln ln Z - ln ln q) = 2.24 at q = 23, so the first-order term
alone already exceeds the whole count. Brun's device (let the truncation order grow with z) is
what saves it, and the resulting lower bound is positive only above the dimension-2 sifting limit
4.2664. Exactness buys nothing.

### 3.10 The bilinear g-m switching

Every strike is a pair (g, m) with the struck member equal to g m; m = 1 is the gear's own home
column. Counting the same set two ways (columns k >= k_min):

| q | strike incidences | m = 1 | m prime | m composite | fraction prime | sieve prediction | ratio |
|---|---|---|---|---|---|---|---|
| 11 | 112 | 6 | 106 | 0 | 0.9464 | 0.8237 | 1.1491 |
| 13 | 1,698 | 18 | 1,517 | 163 | 0.8934 | 0.8755 | 1.0205 |
| 17 | 33,230 | 56 | 24,475 | 8,699 | 0.7365 | 0.7320 | 1.0062 |
| 19 | 676,053 | 192 | 411,963 | 263,898 | 0.6094 | 0.6090 | 1.0007 |
| 23 | 16,324,810 | 762 | 8,370,993 | 7,953,055 | 0.5128 | 0.5127 | 1.0001 |

The prediction is (pi(X_g) - pi(q))/2 * prod(1 - 1/(h-1)) summed over (gear, side): the twisted
tooth m != -eps 2 g^{-1} (mod h) removes one class out of the h-1 nonzero ones, so it costs
primes and q-rough numbers the SAME factor and cancels in the ratio. It is met to 0.01% at
q = 23. The m = 1 count (6, 18, 56, 192, 762) is exactly the number of bottom-open placements of
3.5, at every q.

**The switching identity, exact at all five q.** Let E = ordered strikes (g, m) with m prime and
m <= Z, D = unordered pairs of distinct top gears whose product is a member of a bottom-open
column, Q = top gears whose square is such a member. Then E = 2D + Q: (60; 26; 8),
(595; 286; 23), (6,958; 3,444; 70), (86,438; 43,091; 256), (1,317,795; 658,435; 925). Every one
an exact equality.

So the prime-cofactor part of the top machine's action is a symmetric bilinear object: the
incidence (gear, cofactor) is invariant under swapping the roles whenever both are top gears, and
the twisted machine in the m coordinate for gear g is the twisted machine in the g coordinate for
gear m - the coherent families 2/g and 2/m are one object read from two sides.

**What it does not give, measured.** Both sides count the same number, so the switching gives an
identity, not an inequality. There is no dyadic range in which one side is easier: the cofactor
runs over [1, 6P/29] and the gear over (q, Z], and every dyadic split of that bilinear sum is
exact term by term (3.7). Chen's switching needs an upper bound on one side strictly better than
the lower bound needed on the other; here the two coincide. Stopped.

### 3.11 The twin gap and the two machines' records

The object the root cares about is the longest twin-free stretch. At the period scale:

| q | twins in [k_min, P) | longest gap between consecutive twins | at column | fraction of the period | max gap inside the window | window length | bottom record F | top machine's longest closed run above the prefix |
|---|---|---|---|---|---|---|---|---|
| 11 | 66 | 25 | 110 | 0.29 | 5 | 17 | 7 | 7 |
| 13 | 465 | 83 | 4,070 | 0.81 | 5 | 25 | 11 | 24 |
| 17 | 4,632 | 154 | 31,318 | 0.37 | 5 | 44 | 18 | 30 |
| 19 | 57,449 | 255 | 811,652 | 0.50 | 6 | 56 | 25 | 58 |
| 23 | 896,058 | 502 | 22,713,905 | 0.61 | 12 | 83 | 34 | 104 |

Two things follow, and they are the branch's answer to where the solution space lives.

1. **The twin-free record is a joint object; neither machine can make it alone.** As runs of
   columns: the bottom machine's longest closed run is 6, 10, 17, 24, 33; the top machine's
   longest closed run above the placement prefix is 7, 24, 30, 58, 104; the longest twin-free run
   is 24, 82, 153, 254, 501 - **1.85, 2.41, 3.26, 3.10, 3.66 times the SUM of the two**. The
   record is made in the interaction, not in either machine.
2. **And the interaction is not selective.** Inside the longest twin-free stretch the bottom
   machine is closed on 0.760, 0.711, 0.740, 0.780, 0.783 of the columns; pooled over the 100
   longest stretches, 0.787, 0.754, 0.754, 0.773, 0.790, against the period-average
   bottom-closed density 0.649, 0.703, 0.738, 0.766, 0.786. The ratio is 1.212, 1.072, 1.021,
   1.010, **1.005** - monotone to 1. In the record stretches the bottom machine works at exactly
   its ordinary rate; every one of the ~21% of columns it leaves open is killed by the top
   machine. **The twin-free record is made entirely by the top machine covering the bottom
   machine's ordinary leftovers**, with no help from the bottom machine being unusually closed
   there.

Finally, the twin gap over the period (502 columns at q = 23) is six times the window (83
columns), and the max twin gap per block is flat across the period (287, 365, 472, 385, 401, 360,
447, 439, 478, 431, 443, 448, 502, 439, 468, 420, 433, 470, 499, 483). So nothing at the period
scale can give the window statement: twin-free stretches far longer than the window exist inside
the period, and what saves the window is only that it sits in the first 2.4e-6 of the period,
where the numbers are small.

---

## 4. Mechanism

### 4.1 Why everything is exact, and what that costs

The bottom machine's opening set is periodic mod P; a top gear's two teeth are two classes mod g
with g coprime to P; so a strike count is a count of openings in classes of a modulus coprime to
the period, and inclusion-exclusion over the m bottom gears has at most 3^m terms each with error
below 1 (the stride result, lower-sieve.md section 5). The same argument covers joint moments of
any order while the product of the gears stays below P, and the measurements show even the 3^m
bound is loose by (3/2)^m. The cost is that the arena is 6P ~ e^q numbers wide, so the sifting
parameter s = ln(6P)/ln Z is pinned at exactly 2 by construction: **exactness and s = 2 are the
same fact**, because Z is the square root of the range only because the range is the period.

### 4.2 The twisted copies, and why they are coherent

Dividing the struck member by g turns the two bottom teeth into {0, -+2 g^{-1}} at every bottom
gear: a machine of the same shape with separation 2 g^{-1} at EVERY gear. That is the coherent
family c/r with c = 2, r = g of W3, and by N-S2 (coherence closed under CRT) the pairwise diagonal
satisfies g(S_h + S_h') = 2 (mod h h'). The pairwise overlap of two such copies in the column
coordinate is governed by the mean-overlap identity N-S1 - four CRT classes forming a translate of
{0, S_g, S_h, S_g + S_h}, mean overlap exactly 4m/(gh) for ANY separations - which is the
4N/(gh) of 3.7, and is why the separation cannot be a lever here either (separation_drives_K.md,
W3 already answered).

### 4.3 The clutch, in one paragraph

The bottom machine is exactly periodic, exactly uniform, and its work is done: it opens N
candidates per period in a pattern that does not vary from one end of the period to the other.
The top machine is not periodic over the range at all, is 41% denser at the start of the period
than at the end, and is nearly shut inside the placement prefix where its own gears sit. The two
machines do not interact at all below column q_next^2/6 - that is the window - and above it the
top machine progressively eats 89% of the bottom machine's candidates. Their only shared structure
is the origin and the mirror it forces. All of the correlation between them lives in one cell
(both-open, coupling 0.83 and falling) and that coupling is the classical s = 2 handicap. The
twin-free record is 3.7 times the sum of the two machines' own records, and it is built out of the
bottom machine working at exactly its average rate while the top machine covers everything it
leaves.

### 4.4 The placement question, decided

**Claim.** "For every q the top machine has a doubly occupied placement" is EQUIVALENT to the
infinitude of twin primes.

Proof. A doubly occupied placement is a column carrying two top gears, i.e. a pair of primes g,
g+2 both in (q, Z_q], Z_q = isqrt(6 P_q + 1): a twin pair above q. (=>) Given any bound B take
q > B; the doubly occupied placement is a twin pair above B, so twins are unbounded. (<=) If
twins are infinite then for every q the interval (q, Z_q] with Z_q ~ e^{q/2} eventually contains
one; the five computed machines have 3, 9, 26, 78, 268. QED.

It is strictly weaker than the window statement (which asks for a twin in (q, q^2], while this
asks for one in (q, e^{q/2}]), and strictly stronger than R4's own survivor statement (a twin
below 6P). The three order as

    window (twin in (q, q^2])  =>  placement (twin in (q, sqrt(6 P_q)])
                               =>  survivor (twin in (q, 6 P_q]),

each implication strict in range, all three equivalent to the conjecture when quantified over all
q. So the owner's question - can the top machine ever reach a state with no doubly occupied
placement - is the conjecture itself, in its weakest useful form.

**What the placement view adds.** It puts the parity barrier at one named step. A placement is
bottom-open iff ONE class per bottom gear is avoided (3.5, the placement residue law) - a
dimension-1 condition, where the sieve works and the fundamental lemma applies with no parity
loss; measured at 0.438 of top gears at q = 23 against the exact prediction 0.4358. A placement is
doubly occupied iff TWO classes per bottom gear are avoided - dimension 2, where f_2(s) = 0 up to
s = 4.27. The step from "the partner g+2 is q-rough" (provable) to "the partner is prime" is
exactly the E_1 versus E_2 question of Chen's theorem, and the machine gives nothing extra on it:
3.10 measures that split at the sieve's own prediction to 0.01%.

---

## 5. What is new

1. **The clutch decomposition** (3.2), with the four cell counts, the couplings, and the sub-split
   of the bottom-open/top-closed cell into home strikes and proper kills. The home-only sub-cell
   equals the set of doubly occupied placements at every q (0 mismatches), so a home strike is a
   marker for a twin with a small member, not a nuisance to exempt.
2. **The window as the clutch's zero-interaction region** (3.3): the first proper kill of a
   bottom-open column is at 28, 60, 60, 140, 140 against window tops 20, 28, 48, 60, 88. This is
   the window lemma restated, but the clutch form says what the lemma does not: the window is the
   ONLY part of the period where one machine decides alone, and it is 2.4e-6 of the period at
   q = 23.
3. **THE PLACEMENT RESIDUE LAW** (3.5), exact and not in the project's register: the home columns
   of the top gears meet each non-tooth class of a bottom gear exactly twice and each of that
   gear's own two tooth classes exactly once. It is the reason the placement density is exactly
   prod(1 - 1/(h-1)), i.e. the reason placement is dimension 1.
4. **The origin law** (3.4): the longest both-open stretch of the whole period starts at column 0
   and ends at the first twin above Z, and it is the period's longest from q = 19 on; the top
   machine sits at 3% of its average density inside its own placement prefix. That is the shared
   origin's only consequence beyond the mirror, and it is a strong one.
5. **The record is a joint object** (3.11): the longest twin-free run is 1.85, 2.41, 3.26, 3.10,
   3.66 times the sum of the two machines' own longest closed runs, and inside it the bottom
   machine is closed at 1.005 times its average rate at q = 23 (ratio falling to 1). The twin-free
   record is made by the top machine covering the bottom machine's ordinary leftovers.
6. **The twisted-copy decomposition verified exactly** (3.6, 17.0 million cofactors, 0 mismatches),
   identifying the coherent family c/r of W3 as the top machine's natural coordinate rather than as
   a counterfactual family.
7. **A sharper stride constant** (3.7): the deviation of a top gear's strike count from its fair
   share grows like 2^m, not the 3^m the inclusion-exclusion term count gives.
8. **The correction the branch was for.** R4's reading is confirmed in every measured particular -
   level of distribution 1, all moments exact to all orders, faces B, D and E absent - and it buys
   nothing, because the truncation error of a sieve on this machine is the size of the NEXT EXACT
   TERM, not the size of a remainder (3.9). Face A is not "our error terms are too big"; it is "the
   main terms alternate and do not converge at s = 2", and the period scale is the ideal case in
   which that is visible with no other obstruction present.

## 6. Verdict

**FACT, exact; the reframing is confirmed and it does not open a route, and the clutch is the part
worth keeping.** At the period scale the level of distribution is 1, every joint moment is exact,
and the top machine decomposes into coherent twisted copies of the bottom machine. Face A stands
alone and unchanged. The bilinear g-m switching is an exact symmetry that gives an identity, not
an inequality. What the clutch adds is a vocabulary in which three things are sharp: the window is
the zero-interaction region of the two machines; the placement of a top gear is a dimension-1
event and its double occupancy is the dimension-2 one, which is where the parity barrier sits; and
the twin-free record is a joint object 3.7 times either machine's own, built from the bottom
machine at its average rate. Nothing measured contradicts the reading "face A alone".

## 7. Dead ends

- **Exact moments as a lever.** Stopped at 3.9. Refuting instance: at q = 23, z = 400 the order-2
  error is +1,333,999 against an order-3 term of 1,715,957, and the generic Bonferroni bound with
  the same terms is the identical number. Exactness of the terms does not improve a truncated
  inclusion-exclusion.
- **The g-m switching as a Type II lever.** Stopped at 3.10 after E = 2D + Q was verified at five
  machines. Both sides count the same set and every dyadic split is exact, so there is no
  asymmetry for a Chen-type argument.
- **The survivor curve as a new function.** Stopped at 3.8: it is the classical dimension-2 sieve
  function; its s = 2 value converges to 4 e^{-2 gamma} at rate 1/ln Z and the position of its
  minimum (s = 2.09, 2.11) is a function of s alone. Already on the tree at R2.a.i.a.1.b.
- **Equidistribution of placements.** The working expectation (home columns spread evenly over
  the permitted classes) is refuted at 3.5 and replaced by the exact 2 : 1 law; the placement
  density prediction of P8 survives, missing its 5% tolerance only at q = 11 (6.7%).
- **The top machine's longest closed run as an object.** Stopped at 3.1: the period's longest
  (1,376 at q = 23) is inside the placement prefix and is made of home strikes, i.e. of twins. The
  honest quantity is the run above the prefix (104).
- **The period-scale statement as a route to the window.** Stopped at 3.11: twin-free stretches of
  502 columns exist inside the period against a window of 83, and the maximum is flat across the
  period. Nothing at the period scale implies the window statement.

## 8. Anything that holds without exception, with counts

| statement | count | exceptions |
|---|---|---|
| twisted-copy identity: the cofactor set of a top gear's strikes = the opening set of M^(g,eps) | 4,676 copies, 17,035,903 cofactors | 0 |
| each twisted machine has exactly prod(h-2) openings per m-period 6P | 30 copies | 0 |
| \|X_g - 2N/g\| < 2 * 3^m on the bottom machine's openings | 2,338 (q, gear) cells | 0 |
| \|X_{g,h} - 4N/(gh)\| < 2 * 3^m for gh < P | 19,956 pairs | 0 |
| \|X_{g,h,r} - 8N/(ghr)\| < 2 * 3^m for ghr < P | 13 such triples out of 20,000 sampled | 0 |
| every top gear's kill count over [0,P) is within 1 of 2P/g | 2,338 gears | 0 |
| (both-open) + (home-only) = the twin set of the period, column by column | 38,889,216 columns | 0 |
| the mirror: the clutch state is an even function of k | 2,853,543 columns each side, both machines | 0 |
| column 0 is open in both machines | 5 machines | 0 |
| a top gear's home column is never struck by a bottom gear on the gear's own side | 15,549 (gear, bottom gear) checks | 0 |
| the placement residue law (2 on non-tooth classes, 1 on each tooth class) | 25 (machine, bottom gear) pairs, all residues | 0 |
| home-only sub-cell = doubly occupied placements = twin pairs in (q, Z] | 5 machines, 384 placements | 0 |
| every doubly occupied placement is bottom-open | 384 placements | 0 |
| no placement carries three top gears | 2,338 placements | 0 |
| the first proper kill of a bottom-open column lies above the window | 5 machines | 0 |
| the switching identity E = 2D + Q | 5 machines | 0 |
| the m = 1 strike count equals the number of bottom-open placements | 5 machines | 0 |
| the longest both-open run gap of the period starts at column 0 | 2 machines (q = 19, 23) | 0 above q = 17; false at q = 11, 13, 17 |
