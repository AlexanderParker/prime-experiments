# The length face: `F(q) < q^2/6` as a construction, what constrains a run, where parity bites (measured), and the verdict

Theorist lane (Fable), 2026-09-11. Parent: `research/proof/proof_skeleton.md` section 16, which names
the length face of step 8 (the record below the square: `F(q) < q^2/6` for every prime `q` gives
step 8 at every link) and says its gap is the parity barrier in covering form. Scripts in
`research/anchor235/r76/` (prefix `lf_`); outputs in `research/anchor235/r76/results/` (untracked).
Every number this document relies on is written into it. Nothing is committed; the tree is not edited.

Vocabulary by construction (0.1): column, run, stretch, section. Not used: the four forbidden words.
Local labels LF1..LF4 (proved) and the readings are NOT register numbers; the manager assigns those.

---

## 0. Pre-registered (written before the scripts ran; the scorecard is filled in section 6)

### 0.1 The objects, by construction

- **The line** is the counting numbers. **Column** `k` is the pair `(6k - 1, 6k + 1)`; column 0 is
  `(-1, 1)`. Every number coprime to 6 is a member of exactly one column.
- **Gear** `g` (a prime `>= 5`) **strikes** column `k` iff `g` divides `6k - 1` or `6k + 1`, i.e. iff
  `k = +-u_g (mod g)` with `u_g = 6^{-1} mod g`. The two residue classes `+-u_g` are the gear's
  **teeth**. In numbers: `6 u_g -+ 1` is a multiple of `g` below `6g`, and the smaller of `u_g`,
  `g - u_g` is `k_g`, the column that holds `g` itself (`g = 6 k_g +- 1`); so the teeth are `+-k_g`,
  **each gear's tooth is its own column**, and the distance between the teeth is
  `2 k_g = 3^{-1} (mod g)`, which is `(g + 1)/3` for `g = 2 (mod 3)` and `(g - 1)/3` for
  `g = 1 (mod 3)`: one third of the gear. (`k_5 = 1, k_7 = 1, k_11 = 2, k_13 = 2, k_17 = 3, k_19 = 3,
  k_23 = 4, k_29 = 5, k_31 = 5, k_37 = 6, k_41 = 7, k_43 = 7, k_47 = 8, k_53 = 9`.)
- **Engine** `{5..q}` = the gears that are the primes from 5 to `q`; period `P_q = prod g`; the
  struck/open pattern is periodic with period `P_q` and symmetric under `k -> -k`.
- A column is **open** under the engine if no gear strikes it; column 0 is open under every engine.
- A **run** of length `L` at `x`: the columns `x, ..., x + L - 1` all struck. **`F(q)`** = the largest
  distance between consecutive open columns = (longest run) + 1. Certified: `F = 7, 11, 18, 25, 34,
  43, 58, 88, 91, 103, 118, 145` at `q = 11..53`.
- **The target (the length face):** `F(q) < q^2/6` for every prime `q`. In numbers: among any
  `ceil(q^2/6)` consecutive columns one is open, i.e. every interval of `q^2` consecutive numbers
  contains `n = 5 (mod 6)` with neither `n` nor `n + 2` divisible by a prime in `[5, q]`. Why it gives
  step 8: section `k + 1` of the construction is `[p_k^2, p_{k+1}^2)` with `p_{k+1} = nextprime(p_k^2)`,
  the engine striking it is `{5..q}` with `q = prevprime(p_{k+1})`, and its length in columns is
  `(p_{k+1}^2 - p_k^2)/6 > (q'^2 - q')/6 > q^2/6` with `q' = p_{k+1}` the next prime after `q`; a run
  shorter than `q^2/6` cannot cover it, the open column is a twin prime pair (skeleton 5), and the
  sections tile the line (skeleton 9).
- **Phase vector** of a stretch `[x, x + L)`: `(x mod g)_{5 <= g <= q}`. Gear `g` strikes offset `o` of
  the stretch iff `x + o = +-k_g (mod g)`. By CRT every phase vector occurs exactly once per period.
- **The fixed-separation family**: the machines with gears `{5..q}`, teeth `t_g +- k_g (mod g)` for
  arbitrary phases `t_g`; its record `F_sep(q)` is the longest run over all phase vectors.
- **Smooth and rough parts.** For a member `n`, `s(n)` = the product of the prime powers of `n` with
  prime `<= q` (the engine's part), `r(n) = n / s(n)` the rough part. `n` is struck iff `s(n) > 1`.
- **Liouville sign** `lambda(n) = (-1)^{Omega(n)}`, `Omega` counting prime factors with multiplicity.
  **Column sign** `sigma(k) = lambda(6k - 1) lambda(6k + 1)`. The sieve-visible parity of a column is
  `(-1)^{Omega(s(6k-1)) + Omega(s(6k+1))}`; the striker parity is `(-1)^{number of gears striking k}`.
- **The opening set and its parity twin.** `O(q)` = the open columns; `O^+ = {k in O : sigma = +1}`
  (with column 0), `O^- = {k in O : sigma = -1}`. `b := (q'^2 - 1)/6` is the column whose upper
  member is `q'^2`.
- **Sieve data of an interval `I`** for `d` a product of distinct gears: `A_d(I) = #{k in I : every
  gear of d strikes k}`, `S_d(I) = sum of sigma(k) over those k`, plain remainder
  `r_d = A_d - 2^{omega(d)} |I| / d`.

### 0.2 Theory

- **T1 (the reduction).** The strike pattern on `[x, x + L)` is a function of the phase vector alone,
  and every phase vector is realised, so `F(q) = F_sep(q)`: the length face is the covering problem
  of the fixed-separation family and nothing else. Multiplicativity (phase zero, V14: struck sets are
  multiples) is invisible inside a stretch at height `x > L/4 + 1`, because no member of the stretch
  is a multiple of another member there; every record run is at such a height. The only real-teeth
  content of the length face is the separation `2 k_g = 3^{-1} (mod g)`.
- **T2 (the parity twin).** `O^-` has no element below `b`, because a `q`-rough number below `q'^2`
  is prime and `lambda(p) lambda(p') = +1`. So the twin `O^-` of the opening set violates the
  target at the origin at every `q`, while its sieve data agrees with `O^+`'s to square-root size.
  This is the parity example (Selberg 1949) in these coordinates; it is noted, not re-derived, and
  its numbers are measured.
- **T3 (inside a run nothing is signed).** In a fully struck run the invisible signs (`lambda` of the
  members, of their rough parts, the column sign `sigma`) are balanced; the visible striker parity is
  odd-biased because a run is a near-tiling. The real teeth break no sign symmetry inside a run.
- **T4 (the attempt).** No fact on record bounds `F_sep(q)`; every route is a count (dead by the
  moment ceiling), the budget (ROOT, R1), or the twin decomposition, whose missing piece at the
  origin is the twin primes below `q'^2`. Verdict ROOT.

### 0.3 Predictions, each with the number that refutes it

- **P1 (gates).** An independent column sieve reproduces `F(11..29) = 7, 11, 18, 25, 34, 43`, the first
  record start of m23 at `12,694,429` and of m29 at `200,906,186`, and re-verifies the record runs of
  m31 (`1,468,940,243`, run 57) and m37 (`90,816,580,903`, run 87) in place. Refuted by any mismatch.
- **P2 (the invisible signs are balanced in the high records).** At the four record runs at height
  `>= 10^7` columns (m23, m29, m31, m37), each of `sum lambda(members)`, `sum lambda(rough members)`,
  `sum sigma` has `|z| <= 2.5` (`z = sum / sqrt(count)`), and lies inside the spread of the
  fully-struck-run control. Refuted by `|z| > 3` at any of the 12 cells.
- **P3 (the visible parity is odd-biased in every fully struck run).** The mean striker parity per
  column over fully struck runs of length `>= 20` is `<= -0.3` at every engine m23..m37, against
  `prod(1 - 4/g) = +0.019` (m23) for random stretches. Refuted by a mean above `-0.2`.
- **P4 (the twin's origin gap).** The first element of `O^-` is `b` exactly when `q'^2 - 2` is prime
  (m11: `167` prime; m17: `359`; m23: `839`; m31: `1367`) and later otherwise (m13: `287 = 7 x 41`;
  m19: `527 = 17 x 31`; m29: `959 = 7 x 137`; m37: `1679 = 23 x 73`); in every case above `q^2/6`.
  Refuted by an `O^-` element below `b`.
- **P5 (the twin's data is within tolerance).** On the prefix `[1, b)` at `q = 23..53`,
  `max_d |S_d| / sqrt(A_d) <= 3` over every `d` with `A_d >= 1`. Refuted by a ratio above 4.
- **P6 (the twins above the origin are fair thinnings).** On `[0, 10^8]` the record gaps of `O^+`
  and of `O^-` (excluding the origin gap) lie inside the min-max range of 20 fair half-thinnings of
  `O` at every `q = 11..37`. Refuted by a record outside the range at two or more engines.
- **P7 (blind: the prefix records).** The records of m17 (column 118) and m19 (111) lie below `q'^2/6`;
  their rough members are prime, so `sigma` is biased there; the same engines' fully struck runs at
  height are balanced. Refuted by a balanced `sigma` at the prefix records.

Not pre-registered, recorded as observations in section 3: the size of the twins' records relative
to `F(q)` and to `q^2/6`, and the form of the first `O^-` column at the larger cuts.

---

## 1. Construction

### 1.1 What a fully struck run of `L` columns is, in the numbers

Take `L` consecutive columns `x .. x + L - 1`: the `2L` numbers `6x - 1, 6x + 1, 6x + 5, 6x + 7, ...,
6(x + L - 1) + 1`, i.e. every number coprime to 6 in the interval `[6x - 1, 6x + 6L - 5]`, paired at
distance 2. The run is fully struck iff **in every pair at least one member has a prime factor in
`[5, q]`**; equivalently iff `36k^2 - 1 = (6k - 1)(6k + 1)` has a prime factor `<= q` for every `k` in
the run. The record `F(q) - 1` is the longest such run anywhere on the line.

Exhibit, the first record run of m23 (33 columns at `12,694,429`, numbers `76,166,573 ..
76,166,767`), per column: the engine part `s` of `6k - 1`, its sign, the number of prime factors of its
rough part; the same for `6k + 1`; the column sign (`lf_verify.py` (4), from `results/census.json`):

```
        k     s(6k-1) lam Om_r | s(6k+1) lam Om_r | sigma
12,694,429      133   +1  2   |    25   -1  1   |  -1
12,694,430       17   -1  2   |     1   -1  3   |  +1
12,694,431       55   -1  1   |     7   +1  1   |  -1
12,694,432        1   -1  1   |    23   +1  1   |  -1
12,694,433       13   -1  2   |     1   -1  3   |  +1
12,694,434        1   -1  1   |     5   -1  2   |  +1
12,694,435        1   +1  2   |    19   +1  1   |  +1
12,694,436       35   +1  2   |     1   -1  1   |  -1
12,694,437        1   -1  1   |    13   +1  1   |  -1
12,694,438        1   -1  1   |   539   +1  1   |  -1
12,694,439        1   -1  3   |     5   +1  1   |  -1
12,694,440       23   +1  1   |     1   +1  2   |  +1
12,694,441        5   -1  2   |    17   -1  2   |  +1
12,694,442       11   +1  1   |     1   +1  2   |  +1
12,694,443        7   -1  2   |     1   +1  2   |  -1
12,694,444        1   -1  1   |     5   +1  3   |  -1
12,694,445        1   +1  2   |     7   +1  3   |  +1
12,694,446      325   -1  2   |     1   +1  2   |  -1
12,694,447       17   -1  2   |     1   +1  2   |  -1
12,694,448       19   -1  2   |     1   -1  1   |  +1
12,694,449        1   -1  3   |    55   -1  1   |  +1
12,694,450        7   +1  3   |    13   -1  2   |  -1
12,694,451        5   +1  1   |     1   -1  1   |  -1
12,694,452        1   -1  1   |     7   +1  1   |  -1
12,694,453      121   +1  2   |     1   -1  1   |  -1
12,694,454        1   -1  1   |   475   -1  2   |  +1
12,694,455        1   +1  2   |    23   -1  2   |  -1
12,694,456        5   -1  2   |     1   -1  1   |  +1
12,694,457        7   +1  3   |     1   +1  2   |  +1
12,694,458        1   +1  2   |    17   -1  2   |  -1
12,694,459       13   -1  2   |    35   -1  1   |  +1
12,694,460        1   -1  1   |    11   +1  1   |  -1
12,694,461        5   -1  2   |     1   -1  1   |  +1
```

38 of the 66 members are struck (5 columns have both members struck), 28 are rough; the rough
members have 1, 2, 3 prime factors in 14, 10, 4 cases (none is prime-free: at height `7.6 x 10^7`
a 23-rough number is a product of 1 to 3 primes above 23). Signs: `sum lambda = -10` over 66,
`sum sigma = -3` over 33, `sum` of the striker parity `= -19` over 33 (26 columns have an odd
number of strikers). The run is a near-tiling: mean multiplicity 1.394 strikers per column.

### 1.2 The real teeth against free phases, and the reduction (PROVED)

The free-phase adversary (Ziller-Morack's `h_2`, OEIS A072753) chooses two arbitrary classes per gear.
The real engine has, for each gear, the classes `+-k_g`. Between them sits the fixed-separation
family of 0.1: two classes at distance `2 k_g`, one free phase per gear.

**LF1 (the length face is the fixed-separation family).** For every `x >= 1` and `L >= 1` the
struck/open pattern of `{5..q}` on `[x, x + L)` is the function of the phase vector `(x mod g)_g`
given by: offset `o` is struck iff `x + o = +-k_g (mod g)` for some `g`. Every phase vector is
realised by some `x` in `[1, P_q]` (CRT). Hence `F(q) = F_sep(q)`, and any proof of `F(q) < q^2/6`
proves that **no choice of one phase per gear, with the teeth at the fixed distance `2k_g`, covers
`q^2/6` consecutive integers.** *Proof.* The definitions. QED

**LF2 (multiplicativity is invisible above height `L/4`).** If `x > L/4 + 1`, no member of the
stretch `[x, x + L)` is a multiple of another member of it. *Proof.* A member is at least `6x - 1`;
its smallest proper multiple coprime to 6 is `5(6x - 1) = 30x - 5`; the stretch ends at
`6(x + L - 1) + 1 = 6x + 6L - 5`; and `30x - 5 > 6x + 6L - 5` iff `x > L/4`. QED
Consequence: phase zero (V14, "if `n` is struck every multiple of `n` is struck") says nothing
inside a record run: the record runs sit at `x / L = 384,679` (m23), `4,783,481` (m29),
`25,770,881` (m31), `1,043,868,746` (m37). It acts inside the construction's sections, which are
LOW stretches: section `k + 1` starts at column `a = (p_k^2 - 1)/6` and is `(p_{k+1}^2 - p_k^2)/6`
long, so `a / L` is about `1 / p_{k+1}`: `20 / 2668 = 0.0075` at `[121, 16129)`,
`2688 / 43,408,532 = 6.2 x 10^-5` at `[16129, 260,467,321)`. (The finer statement of skeleton
sections 13-15, twins between consecutive prime squares, has `a / L = (p^2 - 1)/(p'^2 - p^2) >= p/4`:
a high stretch, where multiplicativity is again invisible.)

So, for the length face, **the real teeth add exactly one thing over free phases: the separation
`2 k_g = 3^{-1} mod g`**, and nothing over the fixed-separation family. What that separation is
worth, measured on the record (free run lengths A072753 against `F - 1`, same gear sets):

| gears | free run (A072753) | real run `F - 1` | free / real | free run / (`q^2/6`) | real run / (`q^2/6`) |
|---|---|---|---|---|---|
| {5..23} | 60 | 33 | 1.82 | 0.68 | 0.37 |
| {5..29} | 74 | 42 | 1.76 | 0.53 | 0.30 |
| {5..31} | 94 | 57 | 1.65 | 0.59 | 0.36 |
| {5..37} | 117 | 87 | 1.34 | 0.51 | 0.38 |
| {5..41} | 148 | 90 | 1.64 | 0.53 | 0.32 |
| {5..43} | 173 | 102 | 1.70 | 0.56 | 0.33 |
| {5..47} | 213 | 117 | 1.82 | 0.58 | 0.32 |
| {5..53} | 236 | 144 | 1.64 | 0.50 | 0.31 |
| {5..73} | 436 | not certified | | 0.49 | |

The separation buys a factor 1.3 to 1.8 in length; among the tooth families (same gears, any teeth
`+-v_g`) the real engine's `F` sits at the 13th-26th percentile (docs/novel `tooth-counterfactual-
percentile`), and the real separation does not drive the cover number `K` (the_wall.md 5a). A
constant, never an exponent: the free adversary's own record is at half the square and falling
slowly (0.68 at 23 to 0.49 at 73), and Conjecture 6 (`h_2 < p_n^2 - p_n`) is open for it.

---

## 2. What constrains a run of length `q^2/6`, and what does not

Every fact is stated with its exact hypothesis and what it says about a run of `L = q^2/6` columns
at a height `x > L/4` (by LF2 the only regime the length face has).

| fact on record | exact hypothesis | what it says about the run | verdict |
|---|---|---|---|
| the exact cover (C1 of first_realisation.md; W16 in the wheels' coordinate) | none | a run of `L` at `x` IS a cover of `[0, L)` by the gears' two classes at the phases `x mod g`; capacity: gear `g` covers at most `2 ceil(L/g)` offsets | DECIDES (the census DP `u45_census.py` computes `c(d)` for `d` up to the record at m37, state `2^d`); BOUNDS only while `2 sum 1/g < 1`: `2(1/5 + 1/7 + 1/11) = 0.8675` gives `L <= 45` at `q = 11` (`F(11) - 1 = 6`); `2(1/5 + 1/7 + 1/11 + 1/13) = 1.0214 > 1` at `q = 13`, vacuous for every `q >= 13` |
| joint counts (pairs, triples, ...) | `gh <= L` | in any `L` consecutive columns `g` and `h` both strike `4L/(gh) +- 4` columns, for EVERY choice of teeth (CRT); the same for triples | TEETH-BLIND: no count of joint strikes distinguishes the real teeth from any family member; the teeth enter only through which residues fall inside `[0, L)` (the `+-4`), i.e. through the exact cover, never through a count; fixed-degree certificates are vacuous from m13 (degree 1), m29 (degree 2), m151 (degree 3) (`moment-degree-ceiling`); the LP with consistency proves `F` to m19 only |
| the parity law W17 (`F_top = 2m - (m mod 2)`) | every gear `> 2m + 1`, `m` the gear count | nothing: the engine contains 5, so the hypothesis fails for every engine with 2 or more gears | DOES NOT APPLY (its regime is the opposite of the engine's: large gears, each at most one domino) |
| the mex form W39 | every gear `> 2m` | nothing, same hypothesis | DOES NOT APPLY |
| the arc floor (ArcFloor.lean; skeleton 11a) | twin gears `(g, g+2)` | the two gears' first common column is at `(g + 4)/3` in the folded coordinate; the common columns are 4 classes mod `g(g+2)` | POSITIONAL (the prefix); at height it is a pairwise count, teeth-blind by the row above; measured weak (0.6-11 % of a pair's strikes, skeleton 12); the first long runs meet twin-gear coincidence columns at the chance rate (44 against 51.6, first_realisation.md 4.2) |
| the merge law W13, the chain law W12, the saturation theorem | `F(M) < (q' -+ 1)/3` for the kill-spacing bound; `q' - 1 > F(M)` for saturation | a gap of `M + q'` is a merge of gaps of `M` whose interior openings `q'` strikes, kills at least `(q' -+ 1)/3` apart | BIND ONLY IN THE FREE REGIME: `F(7) = 5 > 4 = (11 + 1)/3`, `F(11) = 7 > 4`, `F(13) = 11 > 6`; never from `7 -> 11` on |
| the budget `F(M + q') <= F(M) + q'` (R1) | the conjecture at every step | summed from the certified `F(23) = 34`: `F(q) <= 34 + sum_{29 <= g <= q} g`, which is `124 < 140` at 29, `155 < 160` at 31, `192 < 228`, `233 < 280`, `276 < 308`, `323 < 368`, `376 < 468` at 37..53, and `< q^2/6` for every larger `q` (`sum g ~ q^2 / (2 ln q)`) | ROOT: R1 contains a twin-Bertrand postulate (node 1e, `F(M+q') >= F_2(M) >= 2 d_0(M)`) |
| the record law and the fusion lemma (`F(M+q') = max_J Q*_J`; `B_{L+1} = max(F(M+q'), R)`, `R = max_m P(m) + |m| + S(m)`) | none | `F(M + q')` is an exact function of the `(L+2)`-windows of `M`'s gap word; the deepest term is a maximal legal word plus the widest flanks the engine hangs on it | EXACT RECURSION, NO BOUND: the order law's upper half `B_{L+1} <= F + q'` is refuted at `43 -> 47` (`153 > 150`, order_law_beyond_41.md 5.3); the remainder `R <= F + q'` fails on 8.6 % / 18.3 % of budget-holding tooth-family members (fusion_lemma.md 3.4) |
| the origin clump W6 and the home-strike run (E7) | column 0 | the columns `1..` holding the gears themselves are struck (run of `2, 2, 4, 4, 4, 6, 6, 6, 9` at m11..m41); near 0 the pattern is the gears' own columns | POSITIONAL, at `x <= q/6`; says nothing at `x > L/4` |
| the mirror `k -> -k` | none | records come in mirror pairs; `x_min <= P/2` | NOTHING ON LENGTH |
| phase zero / multiplicativity (V14, V17) | none | struck sets are multiples | INVISIBLE at `x > L/4 + 1` (LF2); it is the section's fact, not the run's |
| the census as a covering count (`c(d)`, engine_laws_m37.md 1.4) | none | `c(d)` = the number of phase vectors with offset 0 open and `1..d-1` struck; `F = max{d : c(d) > 0}` | DECIDES, in the same sense as the exact cover; W32's first-hit law from it is a large-`N` law (U5) and a count |

Read together: **nothing on record bounds the fixed-separation family's record at height, except
the finite computations.** Every fact that carries a hypothesis on the gears' sizes has the
opposite regime (large gears), every count is teeth-blind, every positional fact is at the origin,
and the two exact instruments (the cover, the census) decide without bounding. What the real teeth
supply to the length face is the separation, worth a constant (1.2).

---

## 3. Where parity bites, as a construction and measured

### 3.1 The object a sieve cannot distinguish: the run against its parity twin

A sieve's inputs on an interval `I` are the counts `A_d(I)` for `d` up to a level `D < |I|`, each
known to within a remainder, and its output is a lower bound on `#(O cap I)`. Split the opening set
by the column sign: `O = O^+ cup O^-`.

**LF3 (the twin is empty at the origin).** `O^- cap [0, b) = {}` for every `q`, `b = (q'^2 - 1)/6`.
*Proof.* A column `k < b` open under `{5..q}` has both members `q`-rough and below `q'^2`, hence
both prime (skeleton 3), hence `sigma(k) = (-1)(-1) = +1`. QED
So the twin `O^-` has a gap from the origin of length at least `b`, and `b = (q'^2 - 1)/6 > q^2/6`:
**the parity twin of the opening set violates the target at every `q`**, by at least the section's
own length `(q'^2 - q^2)/6`. Meanwhile `O^+ cap [0, b)` is the set of twin prime pairs below `q'^2`:
`29, 30, 41, 48, 50, 61, 74, 87` columns at `q = 23, 29, 31, 37, 41, 43, 47, 53` (measured; and
every one of them is a twin prime pair, 8 of 8 cuts, `lf_census.py` part B).

**LF4 (the twin's sieve data).** The sequence `a_k = (1 - sigma(k))/2` on `I` has
`sum_{k in I, every gear of d strikes k} a_k = (A_d - S_d)/2`: its remainder differs from the plain
one by `S_d / 2`, and its sifted count is `#(O^- cap I)`. If a sieve argument proved
`#(O cap I) > 0` from `(2^{omega(d)}/d, |r_d| <= tolerance)` at every interval of length `q^2/6`, the
same argument applied to `a_k` would prove `#(O^- cap I) > 0` at `I = [1, b)`, which is false by LF3,
PROVIDED `|S_d|/2` is within the tolerance. Measured on the prefix `[1, b)` (`lf_census.py` part B;
every `d` a product of at most 6 distinct gears):

| `q` | `b` | columns | open (all twin primes) | `S_1` | `S_1 / sqrt(columns)` | `max_d |S_d| / sqrt(A_d)` (at `d`, `A_d`, `S_d`) | ratios `> 2` / `> 3` | max plain remainder |
|---|---|---|---|---|---|---|---|---|
| 23 | 140 | 139 | 29 | -17 | -1.44 | 1.96 (13, 21, -9) | 0 / 0 of 55 | 1.17 |
| 29 | 160 | 159 | 30 | -21 | -1.67 | 2.14 (23, 14, -8) | 1 / 0 of 80 | 1.67 |
| 31 | 228 | 227 | 41 | -17 | -1.13 | 2.00 (253, 4, -4) | 0 / 0 of 109 | 1.61 |
| 37 | 280 | 279 | 48 | -21 | -1.26 | 2.00 (253, 4, -4) | 0 / 0 of 128 | 1.50 |
| 41 | 308 | 307 | 50 | -21 | -1.20 | 2.24 (253, 5, -5) | 1 / 0 of 157 | 2.01 |
| 43 | 368 | 367 | 61 | -19 | -0.99 | 2.24 (253, 5, -5) | 1 / 0 of 207 | 1.74 |
| 47 | 468 | 467 | 74 | -33 | -1.53 | 2.24 (805, 5, -5) | 3 / 0 of 263 | 2.55 |
| 53 | 580 | 579 | 87 | -51 | -2.12 | 2.89 (5, 232, -44) | 5 / 0 of 327 | 2.21 |

The twist's remainders are of square-root size (never 3 standard deviations, 11 of 1,326 cells above
2), which any sieve at level `D <= |I|^{1 - eps}` tolerates (`sum_{d <= D} 2^{omega(d)/2} sqrt(|I|/d)`
is `|I|^{1 - eps/2}` against a main term `|I| / ln^2 q`). So at each of these eight cuts, **the
sieve's inputs do not separate the opening set from a set that is empty on an interval of the
target's length**: the parity barrier for the length face, exact at each `q`, not asymptotic.
Prior art, one line: this is Selberg's parity example (the sets of even and odd `Omega`, sifted
to the square root) transposed to pairs; the dimension-2 sifting limit `beta_2 = 4.2665` and
Conjecture 6 are on record (docs/novel `j2-upper-bound`); nothing here is re-derived.

**Where the twin's first element is.** The first `O^-` column is the column `(r^2 - 2, r^2)` of a
prime `r` with `r^2 - 2` prime, at 12 of 12 cuts (`lf_twin.py`, `lf_verify.py` (3)): `r = 13` at
`q = 11` (column 28 = `b`), `19` at 13 and 17 (60), `29` at 19 and 23 (140), `37` at 29 and 31
(228), `43` at 37 and 41 (308), `47` at 43 (368), `61` at 47 and 53 (620). It is `b` itself exactly
when `q'^2 - 2` is prime (P4 confirmed: 167, 359, 839, 1367, 1847, 2207 prime at `q = 11, 17, 23,
31, 41, 43`; `287 = 7 x 41`, `527 = 17 x 31`, `959 = 7 x 137`, `1679 = 23 x 73`, `2207` prime but
`b = 468` at 47 holds `2807 = 7 x 401`, `3479 = 7^2 x 71` at 53). This is the square column of
`OneStepE.new_iff` (the new gear's only new strike below its square, iff `q'^2 - 2` is prime) seen
from the twin: the first place where a rough member is composite and its partner prime. An
observation with a mechanism (the candidates between `q'^2` and the next square are the few
products `q' q''` and their partners are struck), not a law.

### 3.2 The census inside the record runs (the brief's measurement)

`lf_census.py`: for each record run, every member factored (`sympy.factorint`); the same for
**RAND** (300 stretches of the same length at uniformly random columns of `[1, P/2]`; unconditioned,
with 6.9-15.5 open columns per stretch on average) and **RUNS** (fully struck runs of length `>= 20,
28, 36, 45` at m23..m37, found by scanning `[1, 1.8 x 10^7]`, `[1, 2 x 10^8]`, `[10^9, 1.1 x 10^9]`,
`[10^9, 1.1 x 10^9]`: 200, 200, 136, 11 runs). `z = sum / sqrt(count)`; the RAND and RUNS columns
are pooled means per unit with the pooled `z`; "pct" is the record's per-run mean's percentile among
the RUNS runs' per-run means. Second measurement: the record runs' `sum lambda` and `sum sigma` by
the division sieve agree with `factorint` at 4 of 4 engines (`lf_verify.py` (2)).

| engine, run | statistic | REC sum / N (z) | RAND mean (z) | RUNS mean (z) | REC pct in RUNS |
|---|---|---|---|---|---|
| m23, 33 at 12,694,429 | `lambda` all members | -10 / 66 (-1.23) | +0.0045 (+0.64) | -0.0127 (-1.18) | 0.15 |
| | `lambda` struck members | -2 / 38 (-0.32) | +0.0191 (+1.92) | -0.0042 (-0.30) | 0.35 |
| | `lambda` rough members | -8 / 28 (-1.51) | -0.0106 (-1.05) | -0.0242 (-1.47) | 0.14 |
| | `lambda` of rough parts of struck members | 0 / 38 (0.00) | -0.0674 (-6.78) | -0.0533 (-3.76) | 0.62 |
| | `sigma` per column | -3 / 33 (-0.52) | +0.0030 (+0.30) | +0.0051 (+0.33) | 0.33 |
| | sieve-visible parity per column | -21 / 33 (-3.66) | +0.0309 (+3.08) | **-0.3856 (-25.4)** | 0.06 |
| | striker parity per column | -19 / 33 (-3.31) | +0.0065 (+0.64) | **-0.4622 (-30.4)** | 0.24 |
| m29, 42 at 200,906,186 | `lambda` all | -2 / 84 (-0.22) | -0.0032 (-0.50) | +0.0096 (+1.05) | 0.42 |
| | `lambda` rough | +3 / 35 (+0.51) | +0.0007 (+0.07) | -0.0043 (-0.30) | 0.68 |
| | `sigma` | -12 / 42 (-1.85) | -0.0095 (-1.07) | +0.0066 (+0.51) | 0.04 |
| | sieve-visible parity | -20 / 42 (-3.09) | +0.0451 (+5.06) | **-0.3885 (-29.9)** | 0.28 |
| | striker parity | -20 / 42 (-3.09) | +0.0187 (+2.10) | **-0.4675 (-36.0)** | 0.50 |
| m31, 57 at 1,468,940,243 | `lambda` all | -8 / 114 (-0.75) | -0.0030 (-0.55) | -0.0028 (-0.28) | 0.25 |
| | `lambda` rough | -1 / 45 (-0.15) | +0.0123 (+1.54) | -0.0109 (-0.68) | 0.51 |
| | `sigma` | -5 / 57 (-0.66) | -0.0048 (-0.63) | +0.0014 (+0.10) | 0.24 |
| | sieve-visible parity | -21 / 57 (-2.78) | +0.0292 (+3.82) | **-0.3642 (-25.8)** | 0.54 |
| | striker parity | -25 / 57 (-3.31) | +0.0126 (+1.65) | **-0.4319 (-30.6)** | 0.51 |
| m37, 87 at 90,816,580,903 | `lambda` all | +14 / 174 (+1.06) | -0.0017 (-0.39) | +0.0361 (+1.17) | 0.73 |
| | `lambda` rough | -1 / 65 (-0.12) | +0.0015 (+0.22) | +0.0216 (+0.44) | 0.45 |
| | `sigma` | -9 / 87 (-0.96) | -0.0045 (-0.73) | -0.0076 (-0.17) | 0.27 |
| | sieve-visible parity | -35 / 87 (-3.75) | +0.0408 (+6.60) | **-0.3270 (-7.5)** | 0.27 |
| | striker parity | -43 / 87 (-4.61) | +0.0244 (+3.95) | **-0.4373 (-10.0)** | 0.36 |

Tiling numbers: mean strikers per column 1.394, 1.452, 1.491, 1.552 at the four records (RUNS:
1.396, 1.475, 1.535, 1.563); both-struck columns 5 of 33, 7 of 42, 12 of 57, 22 of 87 against 7.7,
10.6, 15.4, 25.0 in random stretches of the same length; odd-striker columns 26 of 33, 31 of 42,
41 of 57, 65 of 87. Rough members' prime-factor counts at the records: `{1: 14, 2: 10, 3: 4}`,
`{1: 9, 2: 19, 3: 7}`, `{1: 14, 2: 21, 3: 9, 4: 1}`, `{1: 14, 2: 30, 3: 19, 4: 2}`, the same shape as
the controls.

**Reading (P2, P3 confirmed).** In the four record runs at height the three invisible signs are
balanced: 12 cells, `|z| <= 1.85` (the largest, `sigma` at m29, `-12/42`, is at the 4th percentile of
the RUNS runs; the other eleven are between the 14th and 73rd). The visible parity is odd-biased in
EVERY fully struck run, record or not: `-0.33` to `-0.46` per column pooled over the RUNS controls,
against `+0.006` to `+0.045` in random stretches (the period mean of the striker parity is
`prod(1 - 4/g) = +0.019` at m23, W51), and the records are ordinary among the runs on it (6th to
54th percentile). Mechanism: a fully struck run is a near-tiling (1.3-1.55 strikers per column;
both-struck columns two thirds of a random stretch's), so most columns have exactly one striker; the
striker count is the sieve-VISIBLE part of the column (which gears divide the members), and the
rough parts' signs, the INVISIBLE part, are balanced. The real teeth break no sign symmetry inside
a run.

**The prefix records (P7 confirmed).** m17's record (17 at column 118) and m19's (24 at 111) lie
below `q'^2/6` (60 and 88): there `sigma = -9/17` and `-14/24`, the rough members are prime in 13
of 15 and 18 of 20 cases, and the rough parts of every struck member are prime (`-18/18`,
`-26/26`). The same engines' fully struck runs at height are balanced (`sigma` RUNS mean `-0.009`,
`+0.002`). The bias is the height's (below `q'^2` rough means prime, and the rough parts of struck
members are below `q'^2/5`), not the run's.

### 3.3 The twins' own records (`lf_twin.py`, `X = 10^8` columns; every gap re-verified by `factorint`)

| `q` | `q^2/6` | `b` | record of `O` on `[0, X]` (`F`) | record of `O^+` | record of `O^-` excluding the origin | origin gap of `O^-` | fair half-thinning, 20 seeds: mean (min-max) | one-sided twins (`lambda(6k-1) = -1`, `lambda(6k+1) = -1`) |
|---|---|---|---|---|---|---|---|---|
| 11 | 20.2 | 28 | 7 (7) | **77** | **68** | **28** | 73.4 (66-84) | 68, 67 |
| 13 | 28.2 | 48 | 11 (11) | **95** | **83** | **60** | 87.2 (77-103) | 77, 90 |
| 17 | 48.2 | 60 | 18 (18) | **102** | **100** | **60** | 98.0 (90-112) | 105, 110 |
| 19 | 60.2 | 88 | 25 (25) | **107** | **114** | **140** | 110.7 (100-122) | 112, 111 |
| 23 | 88.2 | 140 | 34 (34) | **115** | **132** | **140** | 123.8 (105-148) | 133, 123 |
| 29 | 140.2 | 160 | 38 (43; record beyond `X`) | 130 | 132 | **228** | 132.0 (117-157) | 133, 123 |
| 31 | 160.2 | 228 | 45 (58) | 130 | 144 | **228** | 140.9 (125-177) | 144, 182 |
| 37 | 228.2 | 280 | 63 (88) | 153 | 145 | **308** | 149.2 (137-177) | 144, 182 |

Bold: above `q^2/6`. Readings:

1. **P6 confirmed, 8 of 8**: the twins' records above the origin lie inside the fair-thinning
   range at every engine (all sixteen values inside; the closest to an edge is `O^+` at m11, 77
   against a range 66-84). Inside each record gap the twin has 18-28 openings of the other
   sign, i.e. `log2(#openings) = 24` of them at `X = 10^8`: a twin gap is a run of consecutive
   same-sign openings, and the twin is a fair coin on `O` above the origin.
2. **The twin loses the small-gear regularity.** `O`'s record is 2.4 to 11 times SHORTER than its
   twins' on the same range (7 against 77 at m11; 34 against 115-132 at m23; 63 against 145-153 at
   m37), and the twins' records equal a Bernoulli thinning's. The engine's record is short because
   of the exact positions (the gears 5 and 7 force an opening within every 35 columns), and the
   sign twist erases that.
3. **Consequence for the target.** At `q = 11, 13, 17, 19, 23` the twins violate the target at
   generic positions (77 > 20, 95 > 28, 102 > 48, 107 > 60, 115 > 88); at `q = 29, 31, 37` they do not
   on `10^8` columns (130 < 140, 144 < 160, 153 < 228; over the full period a thinning's record is
   about `2 F`, `0.62 q^2/6` to `0.78 q^2/6` at 29..53, still below), while `O^-`'s origin gap
   exceeds `q^2/6` at all twelve cuts 11..53 (28, 60, 60, 140, 140, 228, 228, 308, 308, 368, 620, 620
   against 20, 28, 48, 60, 88, 140, 160, 228, 280, 308, 368, 468). So for `q >= 29` **the parity
   barrier for the length face bites at exactly one place: the origin**, i.e. the phase vector 0,
   i.e. the section.
4. The share of `O` with `sigma = -1` by dyadic band of height (m37): 0.239 on `[280, 560)`, 0.294,
   0.421, 0.471, 0.471, 0.496 on the next five bands, 0.500 from `[8,960, ...)` on: the twin is
   unfair only below about `32 b` (numbers below about `32 q'^2`), where rough members are still
   mostly prime; that is the height mechanism of 3.2 seen on the opening set.

### 3.4 Answer to the brief's question

Do the real teeth's runs have the parity property that blocks the sieve? **No**: inside a run the
invisible signs are balanced (3.2), and above the origin the twin of the opening set is a fair coin
(3.3). Does something specific to `+-6^{-1}` break the symmetry? **Yes, in one place**: below `q'^2`
every open column is a twin prime pair, `sigma = +1` (LF3), and that is the consequence of phase
zero (struck = multiple; rough below the square = prime) at the phase vector 0. The parity barrier
for the length face is therefore realised by a genuine set (`O^-`) and located at the origin. The
run at the origin is the section, the position face; the length face's runs at height carry no
parity structure, visible or invisible, beyond the tiling that any sieve sees.

---

## 4. Proof attempt, or the exact obstruction

### 4.1 What a proof must do, from sections 1-3

Bound `F_sep(q)` (LF1) using no count (section 2: every count is teeth-blind and the fixed-degree
certificates are vacuous), no sieve (3.1: the twin `O^-` has the same data and a gap of `b`), and no
multiplicativity inside the run (LF2). What remains is the exact cover with the separation
`2 k_g = 3^{-1} mod g` and the phase vector free.

### 4.2 Attempts

- **A (counting the cover, with forced overlap).** Capacity `sum 2 ceil(L/g) >= L` is vacuous from
  `q = 13` (`1.0214 > 1`). Forced waste: `g`, `h` double-strike `4L/(gh) +- 4` columns, every triple
  `8L/(ghi) +- 8`, teeth-blind; the signed sums truncated at degree 1, 2, 3 are satisfiable with
  zero open columns from m13, m29, m151 (`moment-degree-ceiling`, cited). The full inclusion-
  exclusion is exact and needs `L > 3^{pi(q) - 2} ln^2 q` (the Legendre rung). Dies where recorded.
- **B (the budget).** `F(q) <= 34 + sum_{29 <= g <= q} g < q^2/6` from `q = 29` on (section 2). ROOT
  (R1 contains twin-Bertrand at every step; node 1e). Any per-step bound `F(M+q') <= F(M) + C' q'`
  is the same: with the induction hypothesis it puts a twin below `(C q^2 + C' q')/2` columns.
- **C (the twin decomposition).** `O = O^+ cup O^-`, and the target is "`O^+ cup O^-` meets every
  interval of `q^2/6` columns". By LF3 `O^-` misses `[0, b)`, so at the origin the target is
  "`O^+` meets `[1, q^2/6]`", i.e. `O^+ cap [1, q^2/6] != {}`, i.e. **there is a twin prime pair in
  `(q, q^2 + 1]`**: the columns `1 .. floor(q^2/6)` hold the numbers `5 .. 6 floor(q^2/6) + 1 <= q^2 + 1
  < q'^2`, an open one has both members `q`-rough hence prime and above `q`. That is twin-Bertrand
  at scale `q^2`, the base of the root. Above the origin `O^-` and `O^+` are (measured) fair
  thinnings and the target holds for both from `q = 29` on `10^8` columns; but the length face
  quantifies over every interval, the origin included, and there the twin fails and the proof must
  produce the twin primes below `q'^2`.
- **D (the exponent map, to say exactly what is root).** Let the lemma be `F(q) < C q^2` for all
  `q`. If `C < 1/6`: the section at link `k` has length `> (q'^2 - q')/6 > (q^2 - q)/6 >= C q^2` for
  every `q >= 1/(1 - 6C)`, so every link with such `q` holds and twin primes are infinite (skeleton
  9); with `C = 0.42/6 = 0.07`, the certified maximum of `6F/q^2`, the threshold is `q >= 1.7`: every
  link. So **every `C < 1/6` is the twin prime conjecture and more** (a twin pair in every
  `[y - 6Cy, y]`, `y = q^2`). If `C >= 1/6`, or the exponent is in `(2, 4.27)`: not known to imply
  any parity-sensitive statement, not reachable by the dimension-2 sieve (`beta_2 = 4.2665`; every
  exponent below 4 is conjecturally beyond every sieve, the proved floor being `2 kappa / e = 1.47`,
  docs/novel `j2-upper-bound` (c)), and NOT what step 8 needs. The paired-Iwaniec question (docs/
  novel `j2-lower-ladder` (P3)) lives in that band; step 8 does not.

### 4.3 The smallest missing lemma, as a construction, with its smallest instance

**Lemma (missing).** For every prime `q` and every phase vector `(t_g)_{5 <= g <= q}`, the union of
the residue classes `t_g + k_g` and `t_g - k_g (mod g)` misses some integer in every `floor(q^2/6)`
consecutive integers.

- It is `F(q) < q^2/6` at every `q` (LF1). Smallest instance not certified: `q = 59` carries the
  M9 caveat (`F(59)` boxed in `161..178` against `580`); `q = 61` is clean and open: `F(61) >= 171`
  (SAT lower bound, node 1d), needed `< 620`; the exact-cover search over fifteen gears at length
  620 has no capacity pruning (`sum 2 ceil(620/g) = 1,142 >= 620`).
- Its restriction to the single phase vector `t = 0` and the single interval `[1, floor(q^2/6)]` is
  "a twin prime pair in `(q, q^2 + 1]`" (4.2 C). Its parity twin (LF3) fails at that phase.
- **It is the parity problem itself**: with `C = 1/6` (or any `C < 1/6`) it implies the twin prime
  conjecture (4.2 D), its sieve data cannot separate it from a false statement (LF4, measured at
  eight cuts), and the counting and per-step routes are the moment ceiling and R1. **ROOT, plainly.**
- What the machine can still decide: finite instances (`q = 59, 61` by exact cover or SAT, at a
  cost beyond the r34 instruments' reach so far); the twins' full-period records at `q <= 31` (a
  measurement: whether a fair thinning of `O` ever reaches `q^2/6` inside a period, expected of
  order one at `q = 53`); nothing that moves the exponent.

---

## 5. What is new

1. **LF1 and LF2**: the length face is the fixed-separation covering family and nothing else; phase
   zero (multiplicativity) is invisible inside any stretch at height `x > L/4 + 1`, which every
   record run is (`x/L` from `3.8 x 10^5` at m23 to `1.0 x 10^9` at m37) and no section of the
   construction is (`a/L` about `1/p_{k+1}`). The real teeth's whole content for the length face
   is `2 k_g = 3^{-1} (mod g)`, worth a factor 1.3-1.8 against free classes (table, 1.2).
2. **LF3 and LF4 with numbers**: the opening set's parity twin `O^-` is empty on `[0, b)`,
   `b = (q'^2 - 1)/6 > q^2/6`, at every `q`, and its sieve data on that interval is within
   square-root size of the plain data (`max |S_d|/sqrt(A_d) = 1.96..2.89`, 0 of 1,326 cells above 3,
   at `q = 23..53`). The parity barrier for the length face exhibited as a real set at each cut,
   with the first `O^-` element the square column `(r^2 - 2, r^2)` of a prime `r` with `r^2 - 2`
   prime at 12 of 12 cuts.
3. **The census inside the record runs**: invisible signs balanced (12 cells, `|z| <= 1.85`, ordinary
   among fully struck runs), visible striker parity odd-biased in every fully struck run (`-0.33` to
   `-0.46` per column; 26/33, 31/42, 41/57, 65/87 odd columns at the records) because a run is a
   near-tiling; the prefix records' sign bias is the height's. The real teeth's runs carry no parity
   property beyond what a sieve sees.
4. **The twins' records**: fair thinnings above about `32 b` (8 of 8 engines inside the 20-seed
   range; 18-28 openings of the other sign inside each record gap), 2.4 to 11 times the engine's
   own record on `10^8` columns, above `q^2/6` at generic positions for `q <= 23` and not for
   `q >= 29`; the origin gap above `q^2/6` at all twelve cuts. The barrier's location for the length
   face: the origin, i.e. the section.
5. **The exponent map** (4.2 D): every `C < 1/6` is the twin prime conjecture; the band `C >= 1/6`
   and exponents in `(2, 4.27)` is open, sieve-unreachable, and not needed by step 8.
6. **Instruments**: the division-sieve `Omega` scan (`lf_twin.py`, `6 x 10^8` numbers in 24 s), the
   record-run census with two controls (`lf_census.py`), and the second-measurement script
   (`lf_verify.py`).

Prior art, one line each: the parity example is Selberg (1949) and Tao's account (2007), cited at
W52; the dimension-2 sifting limit and Conjecture 6 are docs/novel `j2-upper-bound`; the
Ziller-Morack table is `paired-jacobsthal-values`; the moment ceiling and the LP certificates are
docs/novel entries; the square column is `OneStepE.new_iff`. Nothing outside the repository is
claimed; the Chowla-type negative bias of `S_1` on the prefix (`-17..-51`) is noted, not pursued.

---

## 6. Verdict and scorecard

**The length face is ROOT.** `F(q) < q^2/6` for every `q` is the fixed-separation covering
statement (LF1); with any constant below `1/6` it implies the twin prime conjecture (4.2 D); its
parity twin is a real set that violates it at every `q` at the origin with indistinguishable sieve
data (LF3, LF4, measured at eight cuts); inside runs at height the real teeth carry no sign
structure a sieve does not see (3.2, 3.3); and the only structure the real teeth add to the length
face is the one-third separation, measured to be worth a constant. Where the difficulty moved:
nowhere new. It is confirmed to sit at the origin, phase vector 0, the section, where the length
face and the position face are the same statement: a twin prime pair below `q'^2`.

- PROVED (in writing): LF1, LF2, LF3, LF4 (given the measured tolerance). Kernel: none.
- MEASURED: the census (3.2, two methods at 4 of 4 records), the twins (3.3, every gap re-verified),
  the twin's data (3.1, one method on the prefix; the record runs' `S_d` by two methods).
- ROOT: the missing lemma of 4.3; its phase-0 instance is twin-Bertrand at scale `q^2`.

| # | prediction | verdict | evidence |
|---|---|---|---|
| P1 | gates | **CONFIRMED** 6 of 6 `F`; both first record starts exact; m31, m37 records verified in place | `lf_gate.py`, `results/gate.json` |
| P2 | invisible signs balanced at the four high records, `|z| <= 2.5` | **CONFIRMED** 12 of 12 cells, max `|z| = 1.85` | 3.2 |
| P3 | striker parity `<= -0.3` per column in fully struck runs | **CONFIRMED** `-0.46, -0.47, -0.43, -0.44` (RUNS) at m23..m37; random stretches `+0.006..+0.024` | 3.2 |
| P4 | first `O^-` column `= b` iff `q'^2 - 2` prime; always `> q^2/6` | **CONFIRMED** 8 of 8 at 11..37 (and 4 more at 41..53) | 3.1, `lf_verify.py` (3) |
| P5 | twist remainders `max |S_d|/sqrt(A_d) <= 3` on the prefix | **CONFIRMED** 8 of 8 cuts, max 2.89 | 3.1 |
| P6 | twins' records inside the fair-thinning range | **CONFIRMED** 16 of 16 values at 8 engines | 3.3 |
| P7 | prefix records signed by height, the same engines' runs balanced | **CONFIRMED** `sigma = -9/17, -14/24` at the records; RUNS `-0.009, +0.002` | 3.2 |

---

## 7. Dead ends (bricks), each with its refuting instance

| idea | dies at | instance | why it cannot be revived |
|---|---|---|---|
| a bound on the run from phase zero (multiples) | every record run | `x/L = 384,679` at m23: no member of the run is a multiple of another (LF2) | multiplicativity acts only at `x <= L/4`, the section's regime, not the record's |
| the capacity of the cover | `q = 13` | `2(1/5 + 1/7 + 1/11 + 1/13) = 1.0214 > 1` | the cover can be arbitrarily long by count from four gears on |
| any joint count as a real-teeth handle | every pair, triple | `g, h` double-strike `4L/(gh) +- 4` for every choice of teeth | CRT; the teeth enter only through the `+-4` |
| the parity law W17 or the mex form on the engine | every engine with two gears | needs every gear `> 2m + 1`; gear 5 | the opposite regime |
| the merge/chain/saturation laws | `7 -> 11` | `F(7) = 5 > (11 + 1)/3 = 4` | the free regime ends before the engine begins |
| the budget summed as a length bound | every step | `124 < 140` at 29, but R1 contains `2 d_0(M) <= F + q'` | ROOT (node 1e) |
| the order law's upper half as a length bound | `43 -> 47` | `B_3 = 153 > 150` | refuted (order_law_beyond_41.md) |
| a sieve or LP argument for the target | every `q` | `O^-` empty on `[0, b)` with `max |S_d|/sqrt(A_d) <= 2.89` (23..53); LP ceilings m13/m29/m151 | the parity example is a real set at each cut |
| a parity signature of the real runs that a sieve cannot see | m23..m37 | `sigma = -3/33, -12/42, -5/57, -9/87` | balanced; the only signature (odd striker count) is visible |
| the twins' generic gaps as the barrier's witness for `q >= 29` | m29..m37 | 130 < 140, 144 < 160, 153 < 228 on `10^8` columns | above the origin the twin is a fair coin with record about `2F`; the witness is the origin |
| a weaker exponent as a stepping stone for step 8 | any `C >= 1/6` | the section is `q^2/6 + (q'^2 - q^2)/6` long and nothing else bounds the run inside it | step 8 needs exactly the root exponent with the root constant |

## 8. Open items on the part alone, sorted

- **Closed here.** The reduction (LF1, LF2); the twin at the origin (LF3, LF4); the census inside
  the record runs; the twins' records to `10^8`; the exponent map.
- **Measurement with no structural content.** The twins' full-period records at `q <= 31`
  (thinning law expected); the Chowla-type sums on longer prefixes; the first `O^-` column beyond 53.
- **Root question in disguise.** The missing lemma of 4.3; its phase-0 instance; any `C < 1/6`.
- **Genuinely open on the part alone, with the attack.** (i) `F(61) < 620` as a finite instance: an
  exact-cover search over 15 gears at length 620 with the r34 CRT+SAT instrument (no capacity
  pruning; feasibility unknown); success closes one more link's engine, nothing more. (ii) The
  band `C >= 1/6` / exponent in `(2, 4.27)` (the paired-Iwaniec question): not a route to step 8,
  a separate problem; if the owner wants it opened it is a new node, not this one.

## 9. Files

- `research/anchor235/r76/lf_common.py` - the column sieve and run extraction
- `research/anchor235/r76/lf_gate.py` - the gates; `results/gate.json`
- `research/anchor235/r76/lf_census.py` - the record-run census with RAND and RUNS controls, the
  record runs' sieve data, the prefix twist data; `results/census.json`, `results/census.log`
- `research/anchor235/r76/lf_twin.py` - the parity twin on `[0, 10^8]`; `results/twin.json`,
  `results/twin.log`
- `research/anchor235/r76/lf_verify.py` - the second measurements; `results/verify.log`
- `results/` is untracked; nothing is committed; `theory_tree.md` is not edited.
