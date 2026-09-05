# The neighbour-sum profile law (branch 2g.i, prover, 2026-09-05)

Parent: node **2g** (three-gap repulsion) under R1.2 (the chain statement), which itself hangs off
node 5b (adjacency repulsion / the suppression law).  What spawned this branch: the manager's
observation in `research/anchor235/r45/results/spectrum_profile.txt` that the maximum neighbour
sum of a gap of size `v` is capped by `F(M) + 1` once `v` is above a small threshold, uniformly
over gap sizes -- a statement about `M` alone that, at the letters of the incoming gear, is the
one-letter chain statement.

What this branch can find that is not already known: `Phi(v)` at the letters and `Delta_J` at
legal cells are on record (alignment-rules 736-790); the UNIFORM profile over all gap sizes, its
threshold `v_0`, its behaviour at the two deepest rungs the period allows, its status on the
tooth-counterfactual family, and its MECHANISM (which gears close which junction, and whether the
two flanks of a wide gap can be glued into one blocked run by CRT) are not.

## 0. Pre-registered (written before any new computation)

### Definitions fixed here

- Machine `M = {5..y}`, period `P = prod_{g in M} g`, column `k` is the pair `(6k-1, 6k+1)`.
  Gear `g` strikes `k` iff `k = +-u_g (mod g)` with `u_g = 6^{-1} mod g`.
- **Gap convention (max-gap):** a gap is the distance between consecutive openings, taken
  cyclically over the period.  `F(M) = max gap`.  A gap of size `s` has `s - 1` blocked columns
  strictly inside it, so the longest blocked run is `F - 1`.
- **The profile.**  For a realised gap size `v`, `N(v) := max over gaps of size v of
  (left neighbour gap + right neighbour gap)`.  The 3-run span at that maximum is `N(v) + v`.
- **The threshold.** `v_0(M) :=` the least `v` such that `N(w) <= F(M) + 1` for every realised
  `w >= v`.
- **Letters.** `a = 2 round(q'/6)`, `b = q' - a`, for the incoming gear `q'` (file 05).
- **Attainment terms.** `Q*_J(M; q')` = the largest span of a word-legal `J`-run (file 16);
  `Q*_1 = F(M)`, `Q*_2 = F_2(M)`, and `F(M + q') = max_J Q*_J` (the attainment identity).

### Theory

T. **The neighbour-sum profile law.** For every machine `M` there is a small `v_0(M)` such that
every gap of size `v >= v_0` has its two neighbour gaps summing to at most `F(M) + 1`.  Below
`v_0` the profile spikes: a tiny middle gap lets two long flanks sit side by side.

Why it would matter: at `v = a` or `v = b` the 3-run is exactly a one-letter merge, so
`Q*_3 = max(N(a) + a, N(b) + b) <= F + 1 + b = F + q' + 1 - a`, i.e. the depth-3 chain statement
with slack `a - 1` -- from a statement about `M` alone.  It needs `v_0 <= a`.

### Predictions, with the number that would refute each

- **P1 (m29).** `F({5..29}) = 43`; predict `N(v) <= 44` for all `v >= v_0` with `v_0 in {6, 8, 10}`,
  and `v_0 <= 10 = a(31)`.  REFUTED if `v_0 > 10`, or if no threshold exists below `F/2`.
- **P2 (m31).** `F({5..31}) = 58`; predict `N(v) <= 59` for `v >= v_0`, `v_0 <= 12 = a(37)`.
  REFUTED if `v_0 > 12`.
- **P3 (the spikes).** The `v` that exceed `F + 1` are all `v <= 7`, and the global maximum of
  `N` sits at `v <= 5`; these are node 2g's tiny-middle 3-run records.  REFUTED by a spike at
  `v >= 8` at m29 or m31.
- **P4 (the letters).** `N(a)` lands in `[F - 2, F + 1]` at every rung m13..m31.  REFUTED by
  `N(a) < F - 2` or `> F + 1`.
- **P5 (attainment).** At each rung the term attaining `F(M + q')` is `Q*_3` or `Q*_4`, never
  `Q*_2` from m19 up.  I expect `Q*_3` to attain at m23 -> m29 (`33 + 10 = 43`) and at
  m29 -> m31, and `Q*_4` at m19 -> m23 (since `Q*_3(m19) = 33 < 34 = F(m23)`).
  REFUTED if `Q*_2` attains at any rung m19..m31, or if some `J >= 5` attains.
- **P6 (the family).** On 200 random symmetric-tooth members at m13, m17, m19 the profile law
  holds with `v_0 <= a` at **60-95%** of members; I expect a minority of violators.  If it holds
  at 100% the law is teeth-free (a property of any two-class machine) and the proof should not
  use the real teeth; if it fails at more than half the members it needs the real teeth.
- **P7 (concatenation).** For a 3-run `(L, v, R)` the glue construction is: place the left
  flank's blocked columns and the right flank's blocked columns adjacent, i.e. find `z` with
  `z = x_0 + 1 (mod g)` for every gear used on the left and `z = x_2 - L + 2 (mod g)` for every
  gear used on the right.  The two residues differ by exactly `v + 1`, so a gear used on BOTH
  sides forces `g | v + 1`.  Prediction: the construction SUCCEEDS at fewer than 20% of the
  attaining 3-runs with `v >= v_0`, because gear 5 must serve both flanks whenever both exceed
  five columns and `5 | v + 1` only for `v = 4, 9, 14, ...`.  REFUTED if it succeeds at more
  than 50%.
- **P8 (L6).** L6 (`left tiling = negated right tiling gear by gear`, equal exactly on gears
  dividing the opening's column) is about ONE opening.  A gap of size `v` has two openings.
  Prediction: L6 gives nothing that depends on `v` beyond `v = 2`, i.e. it is `v`-blind; the
  `v`-dependence in the profile comes from the `g | v + 1` condition of P7, not from L6.
  REFUTED if a `v >= v_0` consequence of L6 is exhibited that fails at `v = 2`.

### Scorecard

| # | Prediction | Result |
|---|---|---|
| P1 | m29 profile capped at `F + 1` above `v_0 <= 10` | REFUTED |
| P2 | m31 profile capped at `F + 1` above `v_0 <= 12` | held at m31, law dead at m29 |
| P3 | spikes only at `v <= 7`, global max at `v <= 5` | first clause REFUTED, second HELD |
| P4 | `N(a)` in `[F - 2, F + 1]` at every rung | REFUTED |
| P5 | attaining term is `Q*_3` or `Q*_4`, never `Q*_2` from m19 | HELD |
| P6 | family: 60-95% of members obey with `v_0 <= a` | REFUTED low (43-61%) |
| P7 | concatenation succeeds at fewer than 20% of attaining 3-runs | REFUTED (52% / 95.5%) |
| P8 | L6 is `v`-blind | REFUTED on the letter, held in spirit |

(Filled table with the numbers in section 8.)

Stop rules: any sub-question that reduces to the peel bound, the triple inequality or the
middle-sum lemma (file 16) is stopped in one line and cited, not re-derived.

## 1. Setup

All numbers below are exact, on FULL periods, in integer arithmetic.  Machines m11..m31 are
`{5..11}` .. `{5..31}`, periods `385, 5005, 85085, 1616615, 37182145, 1078282205, 33426748355`.
The two deep rungs were sieved in 3e7-column chunks by four processes, each owning a contiguous
range plus a 4096-column margin on each side, so that every gap of the period is counted exactly
once (attributed by its LEFT endpoint) and every neighbour and every 7-run is complete;
m29 took 14.6 s and m31 379.6 s at 4 cores.  Scripts:
`research/anchor235/r45/deep_profile.py` (profiles, spectra, `Q*_J`),
`research/anchor235/r45/profile_mechanism.py` (family, mechanism tables, glue test),
`research/anchor235/r45/verify_profile.py` (the exceptionless checks).
Outputs in `research/anchor235/r45/results/`.

Instrument check before any new claim: the script reproduces the manager's m11..m23 profile line
for line; it returns `F = 7, 11, 18, 25, 34, 43, 58` and `F_2 = 11, 16, 25, 31, 39, 55, 68` at
m11..m31, the recorded ladder; its `Q*_5` maximisers are `(7,10,21,10,7)` at m29 and
`(3,25,12,25,3)` at m31, the two words on record in docs/proofs/16; its `Q*_3` maximiser at m31
is `(18,37,30)`, the recorded `F_3` wall; and `max_J Q*_J(m31; 37) = 88 = F({5..37})`, the ladder
value, computed without ever sieving m37's 1.24e12-column period.

## 2. Results

### 2.1 The headline: the `F + 1` law is REFUTED at m29, at the letter itself

`v_0(M)`, the least `v` above which `N(w) <= F + 1` for every realised `w`:

| machine | m11 | m13 | m17 | m19 | m23 | m29 | m31 |
|---|---|---|---|---|---|---|---|
| `F` | 7 | 11 | 18 | 25 | 34 | 43 | 58 |
| `q'` | 13 | 17 | 19 | 23 | 29 | 31 | 37 |
| letter `a` | 4 | 6 | 6 | 8 | 10 | 10 | 12 |
| `v_0` (cap `F+1`) | 7 | 6 | 8 | 8 | 6 | **21** | 8 |
| `v_0 <= a`? | no | yes | no | yes | yes | **no** | yes |

At m29 the threshold is 21 against a letter of 10, and the failure is not marginal and not at an
irrelevant `v`: it is AT the letter.  `N(10) = 48 = F + 5` at m29, and the attaining 3-run is
`(18, 10, 30)`, span 58.  The record corroborates it independently: R3.h has
`F({5..31}) = 58 = 23 + 10 + 25` as gaps of `{5..29}`, another 3-run with middle 10 and flank
sum 48.  Every `v` in `{9, 10, 12, 20}` also breaks `F + 1` at m29 (`N = 54, 48, 46, 45`
against 44).

So the route "the one-letter chain statement `Q*_3 <= F + q'` from a statement about `M` alone
with cap `F + 1`" is DEAD.  The manager's reading of the m23/m29/m31 records as
`N(letter) = F(M) - 1, F(M), F(M) - 1` is right only at the first: measured
`N(a) - F = -3, +1, -1, 0, -1, +5, -2` at m11..m31.

`v_0` is not monotone and not stable (7, 6, 8, 8, 6, 21, 8): m29 is where the mid-range gap sizes
happen to reach `F + 2`, and nothing forbids a recurrence.  A constant-`c` version is dead too:
`max_{v >= 6} (N(v) - F) = 3, 1, 3, 3, 1, 12, 8`, while the chain needs `c <= a - 1 = 3, 5, 5, 7,
9, 9, 11`; m29's 12 is over.

### 2.2 The corrected law, exceptionless: `N(v) <= F_2(M)` for every `v >= 6`

`F_2(M)` is the largest sum of two gaps sharing an opening; write `N(0) := F_2`.  The profile then
says: separating the two gaps by any gap of size at least 6 never lets their sum grow.

| machine | `F` | `F_2` | `max_{v>=6} N(v)` (at `v`) | exceptions `v>=6` | `v < 6` with `N(v) > F_2` |
|---|---|---|---|---|---|
| m11 | 7 | 11 | 10 (v=6) | none | `v=1: +3` |
| m13 | 11 | 16 | 12 (v=6) | none | `v=1: +1`, `v=5: +2` |
| m17 | 18 | 25 | 21 (v=7) | none | none |
| m19 | 25 | 31 | 28 (v=7) | none | `v=2: +2` |
| m23 | 34 | 39 | 35 (v=7) | none | `v=2: +4`, `v=4: +7` |
| m29 | 43 | 55 | **55 (v=7)** | none | `v=2: +3`, `v=3: +7` |
| m31 | 58 | 68 | 66 (v=7) | none | `v=2: +1` |

Zero exceptions over seven machines and 6.4 billion gaps, TIGHT at m29 (`N(7) = 55 = F_2`).
The maximum over `v >= 6` sits at `v = 7` at five of the seven machines and at `v = 6` at the two
smallest.  Below the threshold the spikes are the tiny middles of node 2g: `v = 2` at five of
seven machines, and the global maximum of `N` sits at `v <= 5` at all seven
(at `v = 1, 5, 2, 2, 4, 3, 2` for m11..m31).

### 2.3 The attainment terms, and which one attains

`Q*_J(M; q')` on full periods, with the attaining term marked:

| `M` | `q'` | `Q*_1 = F` | `Q*_2 = F_2` | `Q*_3` | `Q*_4` | `Q*_5` | attains | `F(M+q')` |
|---|---|---|---|---|---|---|---|---|
| m11 | 13 | 7 | **11** | 8 | - | - | `J=2` | 11 |
| m13 | 17 | 11 | 16 | **18** | - | - | `J=3` | 18 |
| m17 | 19 | 18 | **25** | **25** | - | - | `J=2` tie `J=3` | 25 |
| m19 | 23 | 25 | 31 | 33 | **34** | - | `J=4` | 34 |
| m23 | 29 | 34 | 39 | **43** | - | - | `J=3` | 43 |
| m29 | 31 | 43 | 55 | **58** | 55 | 55 | `J=3` | 58 |
| m31 | 37 | 58 | 68 | 85 | **88** | 68 | `J=4` | 88 |

The attaining term's word, rung by rung: `(5,6)` at m11, `(5,11,2)` at m13, `(7,18)` and
`(5,13,7)` (the tie) at m17, `(7,15,8,4)` at m19, `(10,10,23)` at m23, `(18,10,30)` at m29,
`(28,37,12,11)` at m31.  The last row is out of sample: 88 is `F({5..37})` on the recorded ladder,
reproduced here from m31's period alone.

The depth-3 maximiser is a LETTER middle at m13, m17, m19, m23, m29 -- and at m31 it is the
PADDED middle `v = 37 = q'` (`N(37) = 48`, span 85), the `F_3` wall of alignment-rules 3.7.
Letter-only, `Q*_3` at m31 would be `max(56+12, 45+25) = 70`.  So the profile at the letters is
not the whole of `Q*_3` from m31 on.

`N` at the letters, and what `Q*_3` is made of:

| `M` | `a` | `N(a)` | `N(a)+a` | `b` | `N(b)` | `N(b)+b` | `Q*_3` | maximiser |
|---|---|---|---|---|---|---|---|---|
| m13 | 6 | 12 | 18 | 11 | 7 | 18 | 18 | tie |
| m17 | 6 | 17 | 23 | 13 | 12 | 25 | 25 | `b` |
| m19 | 8 | 25 | 33 | 15 | 17 | 32 | 33 | `a` |
| m23 | 10 | 33 | 43 | 19 | 18 | 37 | 43 | `a` |
| m29 | 10 | 48 | 58 | 21 | 30 | 51 | 58 | `a` |
| m31 | 12 | 56 | 68 | 25 | 45 | 70 | 85 | padded `v=37` |

The LITERAL depth-3 maximum (letters only) is `max(N(a)+a, N(b)+b) = 8, 18, 25, 33, 43, 58, 70`
at m11..m31, so `Q*_3^literal - F_2 = -3, 2, 0, 2, 4, 3, 2`, which is the recorded `Delta_3`
(`-3, 2, 0, 2, 4, 3, 2` at m11..m31, alignment-rules 3.7) at all seven machines -- a seventh
independent instrument check, and the reason the profile AT the letters is stopped here as a known
result.  The new content is the profile away from the letters.  Including the padded middle,
`Q*_3(m31) - F_2 = 85 - 68 = 17`, which is the `F_3` wall and not a `Delta_3` cell.

### 2.4 The counterfactual family (200 random symmetric-tooth members, full periods)

Members are the alignment-rules section 5 family: same gears, teeth at `+-v_g` with `v_g` uniform
in `1..(g-1)/2`.  m13 has only 180 members, so all 180 were used.

| | m13 (`a=6`) | m17 (`a=6`) | m19 (`a=8`) |
|---|---|---|---|
| members | 180 (all) | 200 | 200 |
| real member `v_0`(cap `F+1`) / `v_0`(cap `F_2`) | 6 / 6 | 8 / 1 | 8 / 3 |
| members with `v_0`(cap `F+1`) `<= a` | 104/180 (58%) | 85/200 (43%) | 122/200 (61%) |
| members with `v_0`(cap `F_2`) `<= 6` | 177/180 (98%) | 189/200 (95%) | 188/200 (94%) |
| `v_0`(cap `F+1`) min/med/max | 1 / 6 / 13 | 2 / 7 / 16 | 1 / 8 / 20 |
| `v_0`(cap `F_2`) min/med/max | 1 / 2 / 8 | 1 / 1 / 8 | 1 / 2 / 11 |

Read: the `F + 1` form is not a property of two-class machines at all -- 39% to 57% of members
break it, the worst reaching `v_0 = 20` at m19 (member teeth `(2,2,5,3,7,3)`, `F = 20`,
`F_2 = 39`) -- and the real machine breaks it too, at m17 and m29.  It is not that the law needs
the real teeth: the law is false.

The `F_2` form IS nearly structural: 94-98% of members obey it at the same threshold 6, and no
member exceeds threshold 11.  So a proof of `N(v) <= F_2` for `v >= 6` cannot be purely
family-combinatorial either (2-6% of members break it), but it is close -- the same
~90%-structural signature L6 has on this family (pair_statement.md L6: present at ~90%).

### 2.5 Mechanism, exact

**The glue lemma (proved here; new).**  Let `x_0 < x_1 < x_2 < x_3` be four consecutive openings
of `M`, `L = x_1-x_0`, `v = x_2-x_1`, `R = x_3-x_2`.  Pick any map `sigma : M -> {left, right}`
and let `z` be the CRT solution of

    z = x_0 + 1 (mod g)          for every gear with sigma(g) = left,
    z = x_0 + v + 1 (mod g)      for every gear with sigma(g) = right.

Then:

(i) **the column `z + L - 1` is an opening of `M`, for EVERY `sigma`.**  For a left gear
`z + L - 1 = x_0 + L = x_1 (mod g)` and `x_1` is an opening, so `g` misses it; for a right gear
`z + L - 1 = x_0 + v + L = x_2 (mod g)` and `x_2` is an opening, so `g` misses it.  Every gear is
one or the other, so nothing strikes `z + L - 1`.

(ii) if in addition the columns `z .. z+L-2` and `z+L .. z+L+R-2` are all blocked, then the
opening at `z+L-1` has a left gap `>= L` and a right gap `>= R`, so `F_2(M) >= L + R`, i.e.
`N(v) <= F_2(M)` at this run.

A sufficient condition for (ii): the left-assigned gears cover the left flank and the
right-assigned gears cover the right flank -- then every column of the target is blocked by the
same gear that blocked it in the original run.  In practice gears assigned to one side also
strike the other side at their translated phase, and the test below uses the machine itself.

Part (i) is the structural reason the construction can NEVER deliver the stronger `F` bound: the
glue always leaves a hole, so it produces an adjacent PAIR, never a single run.  That is exactly
why the `F + 1` form has no proof of this shape -- and, from m29, is false.

**The concatenation test, exact by CRT.**  Two versions, each enumerating all `2^k` assignments of
the `k` gears and checking the machine column by column at the CRT point.  Version A is the
manager's shift (`v + 1`, no hole; success proves `N(v) <= F + 1`); version B is the shift `v`
with the hole (success proves `N(v) <= F_2`).  Run over EVERY attaining 3-run at every realised
`v >= 6`:

| machine | attaining runs, `v >= 6` | A succeeds | B succeeds | B failures at |
|---|---|---|---|---|
| m13 | 48 | 24 (50%) | **48 (100%)** | - |
| m17 | 90 | 40 (44%) | **88 (98%)** | `v=6` (2) |
| m19 | 124 | 86 (69%) | **118 (95%)** | `v=6` (4), `v=7` (2) |
| m23 | 184 | 80 (43%) | **172 (93%)** | `v=7` (6), `v=8` (4), `v=11` (2) |
| total | 446 | 230 (52%) | **426 (95.5%)** | 20, all at `v` in {6,7,8,11} |

At the LETTERS specifically, B succeeds at 66 of 68 attaining runs (m13 `v=6`: 12/12, `v=11`: 8/8;
m17 `v=6`: 26/28, `v=13`: 12/12; m19 `v=8`: 2/2, `v=15`: 2/2; m23 `v=10`: 2/2, `v=19`: 2/2), while
A succeeds at 28 of 68 and at **0 of 8** at m19 and m23 -- the tight glue stops working at the
letters exactly as the machine grows, one rung before the `F + 1` law itself fails.  At `v = 2`
version B succeeds 0 of 20 times, which it must: at m19, m23, m29 the bound it would prove is
false there.  Every B success had its middle column open (0 of 426 blocked), as part (i) forces.

**What the strikers look like at an attaining run** (m23, `v = 10 = a`, the run `(10, 10, 23)` at
`x_0 = 14995460` whose span 43 is `F({5..29})`): every gear is a sole striker somewhere in the
run -- 5 at eight offsets, 7 at five, 11 at four, 13 at three, 17 at two, 19 at two, 23 at one --
which is L4 on this run, so no gear is idle and no gear can be dropped.  The middle gap's class
per gear (`v mod g` against `{0, +d_g, -d_g}`) is what the chain law reads: `v = 10` is `pad` for
gear 5 and a legal letter for `q' = 29` and for no other gear present.

**A refuted mechanism hypothesis.**  I expected the B failures to be explained by a gear that is a
sole striker on BOTH flanks and does not divide `v` (such a gear cannot serve both sides).  It is
present at 8 of 8 failures at m17/m19 -- but also at 72 of the 118 successes at m19.  It is
necessary-looking and not discriminating; the failures are not explained by it.  What is true of
all 20 failures: they sit at the four smallest `v` in the law's range, where the flanks are
longest.

**Why the profile has the shape it has (which residues).**  `N(v)` tracks the MULTIPLICITY of gap
size `v`, and the multiplicity is set by gear 5.  Openings lie in classes `{0, 2, 3}` mod 5, so
the number of ordered class pairs realising a gap `= s (mod 5)` is `w(s) = 3, 1, 2, 2, 1` for
`s = 0, 1, 2, 3, 4`.  Measured counts at m29: `1: 17.5M (w1), 2: 46.7M (w2), 3: 27.2M (w2),
4: 14.2M (w1), 5: 39.7M (w3), 6: 10.5M (w1), 7: 22.7M (w2), 8: 8.26M (w2), 9: 2.48M (w1),
10: 7.82M (w3)`.  Exceptionless over m13..m31: each of the sizes 4, 6, 9 (the `w1` sizes below 10)
has a STRICTLY smaller count than both its neighbours, 17 cases of 17 (size 9 is unrealised at
m13); and `N(6) < N(7)` at m17, m19, m23, m29, m31 (`17<21, 22<28, 31<35, 47<55, 59<66`).  That is
why the maximum of the profile over `v >= 6` sits at `v = 7`: 6 is a `w1` (rare) size and 7 is the
first `w2` size at or above 6.  Recorded as a mechanism for the SHAPE
only; turning a multiplicity into a maximum is the rate-to-maximum step of the wall's face A4, and
nothing is claimed past it.

### 2.6 L6 across a gap

L6 (pair_statement.md, proved) is about ONE opening `x`: a gear's left offset set is the negation
of its right offset set, equal exactly when `g | x`.  Across a GAP the same two lines of algebra
give a `v`-shifted form.  Write `p_g` for the distance from `x_1` leftward and `q_g` for the
distance from `x_2 = x_1 + v` rightward to `g`'s nearest strike, and `d_g = 2u_g`.  Since
`x_1 - p_g = +-u_g` and `x_2 + q_g = +-u_g`,

    p_g + q_g = -v,  -v + d_g,  or  -v - d_g   (mod g),

the first case exactly when the two ends use the SAME tooth.  At `v = 0` this is L6.  Verified
with 0 violations on full periods: 5,936 (gap, gear) pairs at m13, 111,370 at m17, 2,272,044
sampled at m19.

What it forces, and does not.  The `v`-dependence is a pure translation by `-v`: the three
admissible classes move with `v` and nothing else changes, so there is no `v >= v_0` consequence
that fails at `v = 2` -- L6 constrains the two FIRST STRIKES jointly and says nothing joint about
the two LENGTHS, which is R3.h.i's verdict at a junction, unchanged across a gap.  One thing it
does say that is new and useful: the class `0` (both ends on the same tooth, `p_g + q_g` an exact
multiple of `g`) is available at a gap only when `g | v`, and `g | v` is precisely the condition
under which a single gear could serve both flanks of the glue.  So the glue's shared-gear
condition is L6's equality case `R_g = L_g` transplanted from `x = 0 (mod g)` to `v = 0 (mod g)`.

## 3. What is new

1. **The neighbour-sum profile of a machine, `N(v)` for every gap size, on full periods to m31.**
   Recorded before: `Phi(v)` at the letters and `Delta_J` at legal cells (alignment-rules
   3.7).  New: the whole function, its threshold, and the fact that at the letters it IS
   `Delta_3` (so that part is a rediscovery, closed in one line).
2. **The `F + 1` form is false.**  Refuting instance m29: `N(10) = 48` against `F + 1 = 44`,
   attaining 3-run `(18, 10, 30)`.  Not on record anywhere; the manager's observation held only
   through m23.
3. **The corrected law `N(v) <= F_2(M)` for every `v >= 6`**, 0 exceptions at m11..m31, tight at
   m29.  In words: two gaps separated by a gap of size at least 6 never sum to more than two gaps
   sharing an opening.  Not on record: R3.h.i's "the maximum flank sum at junctions IS `F_2(M)`"
   is the `v = 0` case by definition; the statement for `v >= 6` is a genuine extension and is not
   implied by the triple inequality (which is exactly `max(L, R) + v <= F_2`).
4. **The glue lemma**, proved: for any two-colouring of the gears the CRT re-phasing of a 3-run
   leaves its middle column OPEN, so the construction can only ever produce an adjacent pair and
   hence only ever the `F_2` bound, never an `F` bound.  This is a structural obstruction, not a
   measurement, and it says in advance that no argument of this shape can prove the `F + 1` form.
5. **The concatenation test as a mechanism**, exact by CRT over all `2^k` assignments: 426 of 446
   attaining 3-runs with `v >= 6` glue (m13..m23), 66 of 68 at the letters; the tight version that
   would give `F + 1` manages 230 of 446 and 0 of 8 at m19/m23's letters.
6. **L6 across a gap**: `p_g + q_g = -v, -v +- d_g (mod g)`, verified with 0 violations in
   2.39 million (gap, gear) pairs, with `v = 0` recovering L6, and with `g | v` identified as the
   transplanted equality case that lets one gear serve both flanks.
7. **`F({5..37}) = 88` reproduced from m31's period alone** by `max_J Q*_J(m31; 37) = Q*_4 = 88`,
   with the maximising word `(28, 37, 12, 11)` -- the attainment identity used as an instrument at
   a rung whose own period (1.24e12 columns) was never sieved.
8. **The gear-5 weight of a gap size**, `w(s mod 5) = 3, 1, 2, 2, 1`, as the mechanism for the
   profile's saw-tooth and for the profile's maximum sitting at `v = 7`.

Prior art checked in docs/novel/README.md: the nearest entries are the suppression law
(`F_j - qualmax_j`, round 19, a conditional-mean statement) and the renewal ladder (closed-form
upper bounds on joint qualifying-gap counts).  Both are about counts and means of gaps next to a
long gap; neither states an extremal cap on the neighbour SUM at a fixed middle size, and neither
has the `F_2` form or the glue construction.  `two-teeth-kill-spacing` and `merge-law` supply the
letters and the merge grammar used here and are cited, not re-derived.

## 4. Toward the root

**The question as posed -- "what would have to be proved for `N(v) <= F + c` (any constant `c`)
for all `v >= v_0` with `v_0` below the letters" -- has no answer, because that statement is
false.**  At m29 `max_{v >= 6} (N(v) - F) = 12`, and the one-letter chain needs the constant to be
at most `a - 1 = 9`.  The `F + 1` form is broken at m29 by 4 at the letter itself, and the family
breaks it at 39-57% of members.  That closes the manager's route as posed.

The replacement, and its parts:

- **Proven, and used:** the merge law and chain law with the letters `a = 2 round(q'/6)`,
  `b = q' - a` (file 05, T1-T5); the attainment identity `F(M+q') = max_J Q*_J` (file 08),
  re-verified here at seven rungs; the peel bound and the hypothesis-free triple inequality
  `g_L + w + g_R <= F_2 + min(g_L, g_R)` (file 16, Theorem D), which is exactly
  `max(L, R) + v <= F_2` and does NOT give `L + R <= F_2`; the middle-sum lemma (file 16, Theorem
  A), which bounds the flank sum only relative to `Q*_J`; L6 (pair_statement.md), extended here
  across a gap; and the glue lemma proved here.
- **Measured, not proven:** `N(v) <= F_2(M)` for `v >= 6` (7 machines, 0 exceptions, tight at
  m29); `Delta_3 in [-3, +4]` (on record); the family rates 94-98%.
- **The lowest-order interaction not yet proven**, and the next child branch: *for every 3-run of
  `M` with middle `v >= 6` there is a two-colouring of the gears whose CRT re-phasing blocks the
  glued target.*  This is a finite covering/partition condition on two windows of the machine -- no
  density, no counting, no transfer, at a modulus that grows with the machine (the product of the
  gears), and it quantifies over the machine's own phases only.  It is the shape the wall asks for
  (faces A, B, D).  Measured 426/446; the residue is 20 runs at `v` in `{6, 7, 8, 11}`.

**Is the concatenation construction the proof?  No, on two independent counts.**
(a) Even at 100% it proves only `N(v) <= F_2`, by part (i) of the glue lemma -- the middle column
of the glued object is provably open, so the object is an adjacent pair and the bound is `F_2`.
(b) The `F_2` cap does not close the one-letter chain.  It gives
`Q*_3^literal <= F_2(M) + b`, and the budget `F + q' = F + a + b` then requires
`F_2(M) - F(M) <= a(q')`.  Measured `F_2 - F = 4, 5, 7, 6, 5, 12, 10` against
`a = 4, 6, 6, 8, 10, 10, 12`: it FAILS at m17 (7 > 6) and at m29 (12 > 10).  So the split of the
depth-3 obligation into "an `M`-only profile cap" plus "`F_2 - F` small" is lossy at two of seven
rungs, and the second half is node 5b's quantity, where the only route on record is a
conditional-mean statement that cannot supply an extremal bound.

So the honest placement: the profile is a FACT about `M` with a constructive mechanism at 95.5% of
its extremal runs, not a route to the chain statement.  What it removes from the tree is a route
that looked live: node 2g's three-gap repulsion, read as "`N(v) <= F + 1` above a small
threshold", is refuted, and any successor of it must be stated against `F_2`, not against `F`.

## 5. Exceptionless statements, with counts

- **E1.** `N(v) <= F_2(M)` for every realised gap size `v >= 6`: 0 exceptions at m11, m13, m17,
  m19, m23, m29, m31 -- every gap of every full period, 6,226,553,025 gaps at m31 alone.  Tight
  once (`N(7) = 55 = F_2` at m29).
- **E2.** The glued middle column is an opening, for every two-colouring: proved, and 0 of the 426
  successful glues had it blocked.
- **E3.** `p_g + q_g = -v, -v + d_g` or `-v - d_g (mod g)` at every gap and gear: 0 violations in
  2,389,350 (gap, gear) pairs (m13 full, m17 full, m19 sampled at 200,000 gaps).
- **E4.** `F(M + q') = max_J Q*_J(M; q')` at all seven rungs m11 -> m13 .. m31 -> m37, against the
  recorded `F` ladder `11, 18, 25, 34, 43, 58, 88`.
- **E5.** Each of the gear-5-rare sizes 4, 6, 9 has a strictly smaller gap count than both its
  neighbours at every machine that realises it: 17 of 17.
- **E6.** The global maximum of `N` sits at `v <= 5` at all seven machines
  (`v = 1, 5, 2, 2, 4, 3, 2`), and the maximum over `v >= 6` sits at `v = 7` at m17..m31.

## 6. Verdict

- **The branch's theory T (the `F + 1` profile law) is DEAD.**  Refuting instance: m29,
  `N(10) = 48 > 44 = F + 1`, run `(18, 10, 30)`; and 39-57% of the counterfactual family.
- **What survived is a different, exceptionless law**, `N(v) <= F_2(M)` for `v >= 6`, with a
  proved partial mechanism (the glue lemma) and a named unproven interaction (the two-colouring
  covering condition).  Status: FACT (exact, kept), not a route -- it does not close the chain
  because `F_2 - F <= a` fails at m17 and m29.
- **Child branch opened by this one:** the two-colouring covering condition, 426/446, residue at
  `v` in `{6,7,8,11}`.  It is combinatorial, adversarial-safe and at a growing modulus, which is
  the wall's shape; whether it can be proved is untested.

## 7. Dead ends, each with its refuting instance

- **D1.** `N(v) <= F + 1` above a threshold below the letters.  m29: `N(10) = 48`, `F + 1 = 44`.
- **D2.** Any `N(v) <= F + c` with `c` constant and small enough for the chain.  m29 needs
  `c >= 12`; the chain allows `c <= a - 1 = 9`.
- **D3.** The tight glue (version A) as a proof of any `F`-form bound.  Blocked in principle by
  part (i) of the glue lemma (the middle column is always open), and in practice 0 of 8 at the
  letters of m19 and m23.
- **D4.** "A gear that is a sole striker on both flanks and does not divide `v` explains the glue
  failures."  Present at 8 of 8 failures but also at 72 of 118 successes at m19.
- **D5.** The profile at the letters as a new object.  It is `Delta_3` (`2, 0, 2, 4, 3` at
  m13..m29, matching the record), stopped as a rediscovery.
- **D6.** `v_0` as a stable machine invariant.  It runs `7, 6, 8, 8, 6, 21, 8`.

## 8. Scorecard, filled

| # | Prediction | Result |
|---|---|---|
| P1 | m29 profile capped at `F + 1` above `v_0 <= 10` | **REFUTED** -- `v_0 = 21`, `N(a) = 48 = F + 5` |
| P2 | m31 profile capped at `F + 1` above `v_0 <= 12` | HELD at m31 (`v_0 = 8`), but the law is dead at m29 |
| P3 | spikes only at `v <= 7`; global max at `v <= 5` | first clause **REFUTED** (m29 spikes at 9, 10, 12, 20); second clause HELD 7/7 |
| P4 | `N(a)` in `[F-2, F+1]` at every rung | **REFUTED** at m29 (`+5`) and m11 (`-3`) |
| P5 | attaining term `Q*_3` or `Q*_4`, never `Q*_2` from m19 | HELD from m19 (`J = 4, 3, 3, 4`); `Q*_2` attains at the two small rungs m11 and m17 |
| P6 | family: 60-95% obey the `F+1` form with `v_0 <= a` | **REFUTED low** -- 58%, 43%, 61%; the `F_2` form instead holds at 98%, 95%, 94% |
| P7 | concatenation succeeds at fewer than 20% | **REFUTED** -- version A 52%, version B 95.5% |
| P8 | L6 is `v`-blind | **REFUTED on the letter** -- L6 carries `v` as a translation (`p+q = -v +- d_g`); but the spirit holds: it still forces nothing about the two lengths |
