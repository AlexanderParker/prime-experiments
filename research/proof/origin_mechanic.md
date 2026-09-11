# The mechanic of the origin: the dilation form of the machine, its census on real and killed sections, the hunch as a statement, and the verdict

Theorist lane (Fable), 2026-09-11 (late). Parent: `research/proof/proof_skeleton.md` Part IV.4, which
says what would close step 8: a reason, built from the construction, that the primes below `p_{k+1}`
cannot strike every column between `p_k^2` and `p_{k+1}^2`, using the square-root rule (skeleton 3)
and the hand-up (skeleton 6), and not a count. Scripts in `research/anchor235/r77/` (prefix `og_`);
outputs in `research/anchor235/r77/results/` (untracked). Every number this document relies on is
written into it. Nothing is committed; the tree and the proof document are not edited.

Vocabulary by construction (section 0.1): line, gear, fold, survivor, slot, column, section, machine,
dilate, least striker, quotient, depth. Not used: the four forbidden words.

---

## 0. Pre-registered (written before any script of this branch ran; the scorecard is filled in section 8)

### 0.1 The objects, by construction

- **The line** is the counting numbers `1, 2, 3, ...`. A **gear** of size `g` strikes `g, 2g, 3g, ...`.
- **The fold** is the two gears `2, 3`. Its **survivors** are the numbers they do not strike: exactly
  `6j - 1` and `6j + 1` for `j = 1, 2, ...`. Call the set of survivors `S`. (`S = {5, 7, 11, 13, 17, 19, 23,
  25, 29, 31, 35, ...}`; `1 = 6 x 0 + 1` is counted as a survivor too, it is the empty product.)
- **Column** `j` is the pair `(6j - 1, 6j + 1)`; every survivor above 1 is a member of exactly one
  column; the member `6j - 1` is the column's **left** member, `6j + 1` its **right** member. Column 0
  is `(-1, 1)`. A **slot** is a column both of whose members are counted (the skeleton's word; same
  object).
- **The section at the cut `p`** (`p` a prime `>= 5`, `p'` the next prime): the numbers strictly between
  `p^2` and `p'^2`, i.e. the columns `a + 1 .. b - 1` with `6a + 1 = p^2` and `6b + 1 = p'^2`. Its length
  is `l_p - 1 = b - a - 1` columns. **The engine at the cut `p`** is `{5..p}`, the gears that are the
  primes from 5 to `p`. (This is the section of the finer statement 8e; the sections of the
  construction are the next item.)
- **The sections of the construction** (skeleton 4): `c_1 = 3`; `p_k` = the least prime `>= c_k`;
  `c_{k+1} = p_k^2`. Section `k + 1` is `[c_{k+1}, c_{k+2}) = [p_k^2, p_{k+1}^2)`, columns `a_k + 1 .. b_k - 1`
  with `6a_k + 1 = p_k^2`, `6b_k + 1 = p_{k+1}^2`, and its engine is `{5..q}` with `q` the largest prime
  below `p_{k+1}`. Base 3: `[9, 121)` (engine `{5, 7}`), `[121, 16129)` (engine `{5..113}`); base 5:
  `[25, 841)` (`{5..23}`), `[841, 707281)` (`{5..839}`); base 7: `[49, 2809)` (`{5..47}`),
  `[2809, 7890481)` (`{5..2803}`); base 11: `[121, 16129)`; base 13: `[169, 29929)` (`{5..167}`).
- **Strike, open.** Gear `g` strikes the survivor `n` iff `g` divides `n`; it strikes column `j` iff it
  strikes a member. A survivor (a column) is **open** under a set of gears if none strikes it.
- **The dilate** of `S` by `g`: `g.S = {g m : m in S}`. **The dilation form** (section 1, proved there):
  for a gear `g >= 5`, the survivors `g` strikes are exactly `g.S`.
- **The least striker** of a struck survivor `n` under the engine `{5..q}`: the least gear dividing
  `n`; it is the least prime factor of `n` whenever that is `<= q`. **The quotient** is `n / (least
  striker)`. **The depth** of `n` is the number of steps of the recursion `n -> n / lpf(n) -> ... -> 1`,
  i.e. the number of prime factors of `n` counted with multiplicity, `Omega(n)`.
- **The rough set below a gear.** `R_g = {m in S : no prime below g divides m}` (`1 in R_g`; `R_5 = S`).
- **The tooth family at the cut `p`** (first_realisation.md 3.5, kernel `SquareColumn.FamilyBlocked`):
  the same gears `{5..p}`, each with two teeth `+-v_g (mod g)`, `1 <= v_g <= (g - 1)/2`, striking column
  `j` iff `j = +-v_g (mod g)`. The real engine is the member `v_g = k_g` = the column holding `g`
  (`g = 6 k_g +- 1`), because `g | 6j +- 1` iff `j = -+ 6^{-1} = +-k_g (mod g)`. A member **kills** the
  section if it strikes every column of it. Exhibited killers (first_realisation.md 3.5): at `p = 17`
  the teeth `(v_5, v_7, v_11, v_13, v_17) = (1, 1, 2, 6, 1)` (15 killers of 1,440); at 29
  `(1, 1, 1, 2, 1, 2, 4, 2)` (6,030 of 1,995,840); at 37 `(2, 1, 2, 2, 4, 3, 3, 9, 13, 5)`; at 41
  `(1, 2, 1, 4, 6, 3, 6, 7, 8, 1, 1)`; at 43 `(1, 2, 4, 3, 7, 5, 3, 5, 9, 12, 12, 16)`; at 47
  `(1, 3, 4, 5, 6, 3, 2, 7, 2, 17, 8, 20, 3)`; at 53 `(2, 1, 4, 5, 7, 4, 2, 8, 2, 7, 4, 10, 22, 14)`.
- **V17's machine** (phase_zero.md V17): the gears `{5..q}` together with every twin-prime member
  inside the section, each striking its multiples (so each added gear strikes its own column by a
  home strike). It strikes every column of the section by construction.
- **A phantom strike**: a tooth-family strike on a column whose struck member is not divisible by the
  striking gear. The real engine has none.

### 0.2 Theory (the dilation form, stated before measuring; proved in section 1)

- **T1 (the dilation form).** Since every gear `g >= 5` is itself a survivor, `g m` is a survivor iff `m`
  is; so the survivors struck by `g` are exactly `g.S`, the struck set of `{5..q}` on `S` is the union
  of the dilates `g.S`, `5 <= g <= q`, and the open set is `S` minus that union. By the hand-up the
  gears of the next machine are open survivors of the sections below, so the machine is `S` with its
  own open elements as dilation factors.
- **T2 (the recursion and where it ends).** A struck `n` has a least striker `g_0`, a quotient
  `m = n / g_0 in R_{g_0}` (no prime below `g_0` divides `m`), and `m` is open under `{5..q}` or struck
  by a gear `>= g_0`; iterating ends at 1 after `Omega(n)` steps. Below `p_{k+1}^2` the quotient of
  every struck `n` is below `p_{k+1}^2 / g_0`, and `m` is prime (or 1) whenever `m < g_0^2`, i.e. whenever
  `n < g_0^3`. At height `x` (columns) the members are about `6x` and the quotients about `6x / g_0`,
  far above `g_0^2` for every gear, so nothing in the recursion is decided by size there.
- **T3 (what the origin forces).** Below `p_{k+1}^2` a survivor is struck iff it is composite (skeleton
  3), so a fully struck section is a section every column of which holds a composite; the recursion
  writes each such composite as `g_0 m` with `5 <= g_0 <= sqrt(n) <= m`, `m in R_{g_0}`. Hence the census
  invariant of a real section: **every struck member has an integer quotient `m >= g_0 >= 5`**. A
  tooth-family killer violates it at every phantom strike (the struck member is not a multiple), and
  V17's machine violates it at every added gear (quotient 1, striker above the square root).
- **T4 (the hunch as a statement).** The struck set decomposes by least striker:
  `struck = disjoint union over g of g.R_g`; in columns, the columns whose least striker is `g` are the
  images of the open survivors `m = 6i + s` of the machine below `g` under the affine map
  `(i, s) -> g i + s k_g`, with the side flipped iff `g = -1 (mod 6)`. That is the nesting: scaled,
  shifted copies of the lower machines' open patterns, one per gear. No gear covers a whole column
  (a gear divides at most one member of any column, since it does not divide 2), so no dilate cancels
  the fold; a column is struck twice only by two different gears. Prior art, one line: the count of
  this decomposition is Buchstab's identity / Legendre's formula (V16 in phase_zero.md, FACT/ROOT);
  it is not re-derived here, only the structure is used.
- **T5 (the expected verdict, stated in advance).** The dilation form together with the finite fold and
  the hand-up characterises the real machine (section 5): every counter-machine on record breaks
  exactly one of the three. So a proof from them alone is a proof of the conjecture from the
  definition of the primes, and the instruments that read the definition at the origin are counts.
  The lane expects the obstruction to be the parity problem in dilation form: periodic monoids with
  the same dilation structure whose irreducibles mix primes and products of two primes.

### 0.3 Predictions, each with the number that refutes it

- **P1 (gates).** The column sieve gives the eight twin gears of machine 2 in `[9, 121)` (columns
  2, 3, 5, 7, 10, 12, 17, 18), `F(23) = 34`, the killer `(1, 1, 2, 6, 1)` strikes all eleven columns
  49..59 at `p = 17`, and the real engine `{5..17}` leaves exactly columns 52 `(311, 313)` and 58
  `(347, 349)` open there. Refuted by any mismatch.
- **P2 (the prime-quotient rule).** In every real section (finer, `q = 11..53`; construction, bases
  3, 5, 7, 11, 13 link 1 and bases 3, 5, 7 link 2) every struck member with `n < g_0^3` has a prime
  quotient: 0 exceptions. Above `g_0^3` the quotient is prime in some cases and composite in others
  (both occur in every section with at least 20 columns). Refuted by one exception to the first clause.
- **P3 (depth at the origin).** The maximum depth in the finer section at `q` is `floor(log_5(q'^2))`
  when `5^{floor(log_5 q'^2)}` lies inside the section and one less otherwise: 5 at `q = 53`
  (`3125 = 5^5` in `[2809, 3481)`), 4 at `q = 23..47`, 3 at `q = 11..19`. Refuted by any other maximum.
- **P4 (the census invariant of T3).** In every real section every struck member has an integer
  quotient `>= 5`; in every exhibited killer the number of columns whose only strikes are phantom is
  at least the number of twin-prime columns of the section (2 at `p = 17`: columns 52, 58; the
  section's twin count at every other cut); in V17's machine the columns struck only by the added
  gears are exactly the twin columns and their quotient is 1. Refuted by a real section with a
  non-integer or `< 5` quotient, or a killer with fewer phantom-only columns than twins.
- **P5 (height against origin).** At the four record runs at height (m23 at 12,694,429; m29 at
  200,906,186; m31 at 1,468,940,243; m37 at 90,816,580,903) the share of struck members whose quotient
  by the least striker is prime is below 0.35, while in the finer sections at the same engines
  (`q = 23, 29, 31, 37`) it is above 0.5. Refuted by a height share above 0.5 or an origin share below
  0.35 at any of the four.
- **P6 (the nesting identity).** For every real section and every gear `g` of its engine, the set of
  columns whose least striker is `g` equals `{g i + s k_g : 6i + s in R_g, g(6i + s) in the section}`
  exactly, with the side of the struck member equal to `s` if `g = 1 (mod 6)` and `-s` otherwise.
  0 mismatches. (An identity; its test is a gate on the construction, not evidence for the root.)
- **P7 (distance to the nearest killer).** Let `d(p)` be the least number of gears whose teeth must be
  moved from the real teeth to reach a killer of the finer section at `p`. Prediction: `d(17) = 2`
  (gears 13 and 17), `d(p) >= 2` at every cut with a killer (no single gear's phantom strikes kill),
  and every nearest killer moves at least one gear above the section's length `l_p - 1`. Refuted by
  `d = 1` anywhere or a nearest killer that moves only gears below `l_p - 1`.
- **P8 (the dilation twin).** The monoid `S^+ = {n in S : Omega(n) even}` has the dilation structure
  (closed under multiplication; its irreducibles are the products of two primes `>= 5`; its sections
  are between consecutive irreducible squares, `[625, 395641)` first), and step 8 holds on it (a
  column with both members irreducible in `S^+`) at its first two sections, with at least 100 such
  columns in `[625, 395641)`. The monoid generated by the primes `= 1 (mod 4)` has the dilation
  structure and NO two irreducibles at distance 2 in any section. Refuted by a section of `S^+`
  without such a column.

Owner's predictions on the scorecard (from the hunch, as this lane reads it): the nesting is exact
(P6, expected to hold) and the nesting is what protects the sections (expected by the lane to hold
for the finer statement at `p = 17, 29, 37..53` where free classes kill, and NOT to be needed at any
section of step 8 where the free-class record is known: `h_2 < q^2 / 6` at every `q <= 73`).

---

## 1. The dilation form as a construction

Everything in this section is proved by the construction written next to it; the measured checks are
gates on the scripts, not evidence for the root.

**D1 (the dilate is the strike set).** A gear `g >= 5` is a prime, so it is not struck by the fold and is
itself a survivor: `g = 6 k_g + eps_g` with `eps_g = +-1` and `k_g` the column holding `g`. For any
`m`, `g m = +-1 (mod 6)` iff `m = +-1 (mod 6)`, because `g` is invertible modulo 6. Hence the survivors
`g` strikes (its multiples that the fold left) are exactly `g.S = {g m : m in S}`. The struck set of the
engine `{5..q}` on `S` is the union of the dilates `g.S` over its gears; the open set is `S` minus it.
By the hand-up (skeleton 6) the gears of machine `k + 1` are the open survivors of section `k + 1`, so
the machine is `S` with its own open elements as dilation factors. PROVED.

**D2 (a dilate in columns: the real teeth are the dilation form).** For a survivor `m = 6i + s`
(`s = +-1` its side, `i` its column) and a gear `g = 6 k_g + eps_g`:
`g m = (6 k_g + eps_g)(6 i + s) = 6 (g i + s k_g) + eps_g s`.
So `g m` sits in column `g i + s k_g` on the side `eps_g s`: the dilate `g.S` occupies the columns
`= +-k_g (mod g)`, the left members of `S` going to one tooth and the right members to the other, with
the sides swapped iff `g = -1 (mod 6)`. This is the real-teeth member of the tooth family (length_face.md
0.1: each gear's tooth is its own column; kernel `SquareColumn.real_teeth`, `blockedZ_eq_family`). So at
the level of columns the dilation form adds nothing to the real teeth: it IS the real teeth. PROVED.

**D3 (the nesting: the struck set is the disjoint union of dilates of the lower machines' open sets).**
A struck survivor `n` has a unique least prime factor `g_0 <= q`; write `n = g_0 m`. No prime below `g_0`
divides `m`, so `m in R_{g_0}` (the open survivors of the machine `{5..g_0^-}`, `1` included).
Conversely every `g m` with `m in R_g` is struck with least striker `g`. Hence
`struck({5..q}) = disjoint union over g <= q of g.R_g`,
and in columns, by D2, the `(column, side)` pairs of the members with least striker `g` are exactly
`{(g i + s k_g, eps_g s) : 6 i + s in R_g}`. Each gear contributes a scaled, shifted copy of the open
pattern of the machine below it, with the sides swapped when `g = -1 (mod 6)`. PROVED; measured as
P6: 0 mismatches over 198,798 members in 19 sections (section 2, tables A and B; the first run showed
exactly one mismatch at each cut where `p'^2 - 2` is composite, the member `p'^2 - 2` of column `b`,
which lies strictly between `p^2` and `p'^2` but outside the section's columns `a+1 .. b-1`; the
identity's range was restricted to the section's columns and the mismatches went to 0).
Prior art, one line: the COUNT of this decomposition, `#struck up to x = sum_g #(R_g up to x/g)`, is
Buchstab's identity iterated, i.e. Legendre's formula (phase_zero.md V16, FACT and ROOT as a route).
It is not re-derived here and no count is taken from it.

**D4 (the recursion and where it ends).** `n = g_0 m` with `m in R_{g_0}`. If `m` is composite its least
prime factor is `>= g_0`, so `m >= g_0^2` and `n >= g_0^3`. Hence: **`n < g_0^3` forces the quotient `m`
to be prime or 1.** The recursion `n -> n / lpf(n) -> ...` ends at 1 after `Omega(n)` steps, and
`Omega(n) <= log_5 n`; below `p'^2` that is at most `floor(2 log_5 p')`: 3 at `p' <= 19` (`5^3 = 125 <
p'^2 < 625`), 4 at `p' = 29..53`, 5 at `p' = 59` (`3125 = 5^5 < 3481`). PROVED. Prior art, one line: the
first clause is the two-prime lemma (kernel `CoreLeftover.primeOrSemiprime_of_rough_lt_cube`) with the
least prime factor as the threshold; noted, not re-derived.

**D5 (the origin's census invariant).** Below `p'^2` a survivor is struck iff it is composite (skeleton 3).
So in a real section every struck member is `n = g_0 m` with `5 <= g_0 <= sqrt(n) <= m`, `m` an
integer survivor. In a tooth-family member a phantom strike (a class hit on a column whose struck
member the gear does not divide) has no such `m`; in V17's machine every added gear strikes its own
column with `m = 1` and `g_0 = n > sqrt(n)`. PROVED by the definitions; measured as P4 (table C).

**D6 (what the origin forces that height does not).** At height `x` (columns) the members are about
`6x` and every quotient `m = n / g_0` is about `6x / g_0 >= 6x / q`, far above `g_0^2` for every gear
once `x > q^3 / 6`; so at the record runs (`x >= 1.27 x 10^7` at m23) no struck member satisfies
`n < g_0^3` and D4 decides nothing there; `m` is prime or not by its own arithmetic, at the rate of a
generic rough number (measured 0.16-0.27, table E). At the origin D4 decides the quotient of every
tail strike (`g_0 > n^{1/3}`), and every open column is a twin prime pair. PROVED (the size statements)
and MEASURED (the rates).

---

## 2. Results as tables

All by `og_census.py` (spf table), `og_family.py`, `og_nearest.py`, `og_brute.py`, `og_height.py`
(`sympy.factorint`), `og_monoid.py`; gates by `og_gate.py`: the eight twin gears of machine 2 in
`[9, 121)` at columns `2, 3, 5, 7, 10, 12, 17, 18`, `F(23) = 34`, the killer `(1, 1, 2, 6, 1)` striking all
of 49..59, the real `{5..17}` leaving exactly 52 and 58: all reproduced (P1).

### Table A. The finer sections `[p^2, p'^2)` under `{5..p}`: the dilation census

`pq` = share of struck members whose quotient by the least striker is prime; "below cube" = the
struck members with `n < g_0^3` (their `pq` is 1.000 at every row: P2, 0 exceptions); "above cube" =
the rest with their `pq`; depth = `Omega` of the least-struck member.

| `p` | columns | open (twins) | both struck | `pq` all | below cube (`pq`) | above cube (`pq`) | max depth | depth histogram |
|---|---|---|---|---|---|---|---|---|
| 11 | 7 | 2 | 1 | 0.800 | 2 (1.000) | 3 (0.667) | 3 | {2: 4, 3: 1} |
| 13 | 19 | 7 | 4 | 0.750 | 4 (1.000) | 8 (0.625) | 3 | {2: 9, 3: 3} |
| 17 | 11 | 2 | 3 | 0.778 | 3 (1.000) | 6 (0.667) | 3 | {2: 7, 3: 2} |
| 19 | 27 | 4 | 4 | 0.826 | 8 (1.000) | 15 (0.733) | 3 | {2: 19, 3: 4} |
| 23 | 51 | 8 | 13 | 0.767 | 13 (1.000) | 30 (0.667) | 4 | {2: 33, 3: 9, 4: 1} |
| 29 | 19 | 2 | 5 | 0.706 | 6 (1.000) | 11 (0.545) | 4 | {2: 12, 3: 4, 4: 1} |
| 31 | 67 | 11 | 22 | 0.696 | 17 (1.000) | 39 (0.564) | 4 | {2: 39, 3: 16, 4: 1} |
| 37 | 51 | 7 | 14 | 0.636 | 8 (1.000) | 36 (0.556) | 4 | {2: 28, 3: 14, 4: 2} |
| 41 | 27 | 3 | 11 | 0.667 | 7 (1.000) | 17 (0.529) | 4 | {2: 16, 3: 7, 4: 1} |
| 43 | 59 | 11 | 25 | 0.667 | 10 (1.000) | 38 (0.579) | 4 | {2: 32, 3: 14, 4: 2} |
| 47 | 99 | 13 | 32 | 0.640 | 18 (1.000) | 68 (0.544) | 4 | {2: 55, 3: 27, 4: 4} |
| 53 | 111 | 13 | 46 | 0.663 | 19 (1.000) | 79 (0.582) | 5 | {2: 65, 3: 27, 4: 5, 5: 1} |

P3 confirmed at 12 of 12: the maximum depth is 3 at `p = 11..19`, 4 at `23..47`, 5 at 53 (the member
`3125 = 5^5` at column 521). P6 (the nesting identity): 0 mismatches over 645 members in these twelve
sections.

### Table B. The construction's sections `[p_k^2, p_{k+1}^2)` under `{5..q}`

| base, link | section | `q` | columns | open (twins) | both struck | `pq` all | below cube | above cube (`pq`) | max depth | P6 members / mismatches |
|---|---|---|---|---|---|---|---|---|---|---|
| 3, 1 | [9, 121) | 7 | 18 | 8 | 0 | 1.000 | 10 | 0 | 2 | 10 / 0 |
| 3, 2 (= 11, 1) | [121, 16129) | 113 | 2,667 | 276 | 1,097 | 0.607 | 439 | 1,952 (0.518) | 6 | 3,488 / 0 |
| 5, 1 | [25, 841) | 23 | 135 | 29 | 28 | 0.811 | 44 | 62 (0.677) | 4 | 134 / 0 |
| 5, 2 | [841, 707281) | 839 | 121,127 | 6,224 | 68,889 | 0.447 | 10,757 | 104,146 (0.390) | 8 | 183,792 / 0 |
| 7, 1 | [49, 2809) | 47 | 459 | 74 | 139 | 0.706 | 113 | 272 (0.585) | 4 | 524 / 0 |
| 7, 2 | [2809, 7890481) | 2803 | 1,323,991 | 48,249 | 836,197 | 0.381 | 85,020 | 1,190,722 (0.337) | 9 | not run (size) |
| 13, 1 | [169, 29929) | 167 | 4,959 | 455 | 2,213 | 0.573 | 736 | 3,768 (0.490) | 6 | 6,717 / 0 |

Depth histograms: base 3 link 2 `{2: 1451, 3: 767, 4: 156, 5: 16, 6: 1}`; base 5 link 2
`{2: 51413, 3: 44514, 4: 15467, 5: 3047, 6: 420, 7: 40, 8: 2}`; base 7 link 2 `{2: 485972, 3: 507112,
4: 218069, 5: 53885, 6: 9310, 7: 1249, 8: 134, 9: 11}` (`5^9 = 1,953,125` is in the section); base 13
link 1 `{2: 2582, 3: 1538, 4: 340, 5: 42, 6: 2}`. P2: 0 exceptions in all seven sections. The share of
prime quotients falls with the section's top (1.000, 0.81, 0.71, 0.61, 0.57 at the first links with
tops 121 .. 29,929; 0.45 and 0.38 at the second links with tops `7.1 x 10^5` and `7.9 x 10^6`).

### Table C. The killed sections: the tooth family's exhibited killers and V17 (`og_family.py`)

`L` = the section's column count; "moved" = the gears whose teeth differ from the real teeth;
"phantom" = family strikes on a column whose struck member the gear does not divide; "phantom-only"
= columns all of whose family strikes are phantom; the real engine's invariant D5 was checked on every
struck member of every section: 0 violations at all seven cuts.

| `p` | columns (`L`) | moved gears | phantom / real strikes | phantom-only columns | twins of the section | who kills each twin (family gear) |
|---|---|---|---|---|---|---|
| 17 | 49..59 (11) | 13, 17 (both `> L`) | 4 / 10 | 2: {52, 58} | 2: {52, 58} | 52 by 17; 58 by 13 |
| 29 | 141..159 (19) | 11, 17, 19, 29 | 10 / 19 | 4: {143, 147, 150, 152} | 2: {143, 147} | both by 29 |
| 37 | 229..279 (51) | 5, 17, 23, 29, 31, 37 | 40 / 38 | 19 | 7 | 238, 247, 248, 278 by 5; 242, 268 by 5 and 17; 270 by 29 |
| 41 | 281..307 (27) | 7, 11, 13, 17, 23, 29, 31, 37, 41 | 31 / 15 | 13 | 3 | 283 by 17, 29; 287 by 11, 31; 298 by 11 |
| 43 | 309..367 (59) | 7, 11, 13, 17, 19, 23, 31, 37, 41, 43 | 71 / 28 | 33 | 11 | 312: 11; 313: 7, 17; 322: 13; 325: 23; 333: 17; 338: 7; 347: 17, 19; 348: 7, 11, 13, 23; 352: 7; 355: 7; 357: 41 |
| 47 | 369..467 (99) | 7, 11, ..., 47 (11 gears) | 117 / 50 | 53 | 13 | 373: 47; 378: 11; 385: 13, 17; 390: 37; 397: 17; 425: 11; 432: 31; 443: 41; 448: 17; 452: 7; 455: 11; 465: 7, 17; 467: 31, 47 |
| 53 | 469..579 (111) | 5, 11, ..., 53 (13 gears) | 156 / 31 | 80 | 13 | 495: 47; 500: 17; 520: 17; 528: 5, 13, 19; 542: 5, 47; 543: 5, 11, 29; 550: 23; 555: 19; 560: 31; 562: 5, 37; 565: 11; 577: 5, 13, 23; 578: 5, 41 |

V17 at the same seven cuts: the added gears are the twin members (4, 4, 14, 6, 22, 26, 26 numbers);
the columns struck only by added gears are exactly the twin columns, 7 of 7 cuts, every such strike
with quotient 1. P4 confirmed: phantom-only columns `>=` twins at 7 of 7 (equal only at 17).

### Table D. The distance from the real teeth to the nearest killer (`og_nearest.py`; `og_brute.py` as the second measurement)

`d(p)` = the least number of gears whose teeth must move from `+-k_g` to reach a killer.

| `p` | `L` | twins | `d(p)` | the moved sets at distance `d` | a gear `> L` among the moved? | second measurement |
|---|---|---|---|---|---|---|
| 11, 13, 19, 23, 31 | 7, 19, 27, 51, 67 | 2, 7, 4, 8, 11 | none (no killer; searched to `d = 6`) | | | first_realisation.md 3.5: 0 killers |
| 17 | 11 | 52, 58 | **2** | {11, 13}, {11, 17}, {13, 17} | yes, in all three | brute force over 1,440: 15 killers, minimum distance 2, three at it: `(1,1,2,6,1)`, `(1,1,3,1,3)`, `(1,1,3,2,2)` |
| 29 | 19 | 143, 147 | **1** | {29}, tooth 2 in place of 5 | yes | brute force over 1,995,840: 6,030 killers, minimum distance 1, unique: `(1,1,2,2,3,3,4,2)` |
| 37 | 51 | 7 | 5 | {13, 23, 29, 31, 37} (teeth 5, 6, 13, 9, 16), unique | no gear `> 51` exists | search only |
| 41 | 27 | 3 | 2 | {11, 23}, {11, 29}, {11, 41}, {13, 29} | two of four | search only |
| 43, 47, 53 | 59, 99, 111 | 11, 13, 13 | `>= 6` (killers exist; none to `d = 5`) | | | first_realisation.md 3.5: killers by cover search |

P7 REFUTED at `p = 29`: one gear suffices. The tail clause holds at 17 and 29 and is vacuous from 37
on (every gear is below the section's length there).

### Table E. Height against the origin (`og_height.py`, `sympy.factorint`; 100 random stretches of the record's length at columns uniform in `[x/2, 2x]` as controls)

| engine | record run (`x`, `L`) | record `pq` | record below cube | record depth histogram | controls `pq` | controls mean depth | origin section (finer) `pq` | origin below cube | origin depth histogram |
|---|---|---|---|---|---|---|---|---|---|
| m23 | 12,694,429, 33 | 9/33 = 0.273 | 0 | {2: 9, 3: 15, 4: 7, 5: 2} | 0.230 | 3.26 | 33/43 = 0.767 | 13 | {2: 33, 3: 9, 4: 1} |
| m29 | 200,906,186, 42 | 8/42 = 0.190 | 0 | {2: 8, 3: 20, 4: 11, 5: 3} | 0.206 | 3.38 | 12/17 = 0.706 | 6 | {2: 12, 3: 4, 4: 1} |
| m31 | 1,468,940,243, 57 | 10/57 = 0.175 | 0 | {2: 10, 3: 30, 4: 12, 5: 2, 6: 3} | 0.181 | 3.45 | 39/56 = 0.696 | 17 | {2: 39, 3: 16, 4: 1} |
| m37 | 90,816,580,903, 87 | 16/87 = 0.184 | 0 | {2: 16, 3: 30, 4: 32, 5: 6, 6: 2, 7: 1} | 0.158 | 3.61 | 28/44 = 0.636 | 8 | {2: 28, 3: 14, 4: 2} |

P5 confirmed 4 of 4 (height below 0.35, origin above 0.5). The origin shares agree with table A by the
other method (0.767, 0.706, 0.696, 0.636: two measurements). At height no struck member is below its
striker's cube; the record runs are ordinary among random stretches on this census.

### Table F. Machines with the dilation structure on other monoids (`og_monoid.py`)

| monoid | closed under `x`? periodic? | irreducibles | cuts (`g_1, g_2, g_3`; `c_2, c_3, c_4`) | twin-irreducible columns in the sections | verdict on "8" |
|---|---|---|---|---|---|
| `S` (the fold's survivors) | yes, yes | the primes `>= 5` | the construction | tables A, B | the conjecture |
| `S^+ = {n in S : Omega(n) even}` | yes, no | products of two primes `>= 5` | 25, 629, 395,651; 625, 395,641, `1.57 x 10^11` | 11,969 in `[625, 395641)` (first `(695, 697) = (5 x 139, 17 x 41)`); 95,582 on the prefix `[395641, 4 x 10^6)` | holds (measured) |
| `S_H`, `H = {+-1} mod 12`, column `(12j - 1, 12j + 1)` | yes, yes | primes `= +-1 (12)` and products of two primes `= +-5 (12)` | 11, 131, 17,183; 121, 17,161, 295,255,489 | 558 in `[121, 17161)`: 139 prime-prime, 303 mixed, 116 semiprime-semiprime; 65,639 on `[17161, 4 x 10^6)` | holds (measured); the parity mix inside one periodic dilation machine |
| `M` = generated by 5 and the primes `= 1 (mod 6)` | yes, **no** | 5 and the primes `= 1 (mod 6)` | 5, 31, 967; 25, 961, 935,089 | **0** in `[25, 961)` among 20 columns with both members in `M` (first `(35, 37)`, `(65, 67)`, `(95, 97)`, `(125, 127)`; 74 irreducibles in the section); **0** in `[961, 935089)` among 9,985 such columns (36,812 irreducibles) | **violates 8 at every section**: a left member `= 5 (mod 6)` lies in `M` iff it carries an odd number of factors 5, so it is irreducible only if it is 5 itself |
| generated by the primes `= 1 (mod 4)` | yes, no | those primes | 5, 13; 25, 169 | none: two of them are never at distance 2 | violates trivially (no column has both members in it) |

P8 confirmed (`S^+`), and the violating machine exhibited (`M`).

---

## 3. Mechanism

**What the dilation form adds over free classes, exactly.** At the level of columns, nothing (D2): the
dilates are the real teeth. At the level of members, one invariant (D5): every strike is a
factorisation `n = g_0 m` with `5 <= g_0 <= sqrt(n) <= m`, `m` a survivor, and the recursion on `m` is
the factorisation of `n`. The three counter-machines are separated by that invariant alone: the family
killers carry 4 to 156 phantom strikes per section (table C), V17 carries strikes with `m = 1`, the real
section carries none of either (0 violations at seven cuts). The invariant is also the tautology of the
origin: below `p'^2`, struck = composite, so a fully struck section is a section every column of which
holds a composite, and step 8 is "some column holds two primes". The census makes the tautology
visible, it does not lift it.

**What the recursion forces at the origin.** Every quotient is decided by size when `n < g_0^3` (D4);
those are 16 % to 30 % of the struck members in the finer sections (table A), and their share falls as
the section's top grows (6.4 % at base 7 link 2). Above the cube the quotient is prime at 0.53 to 0.73
in the finer sections, 0.34 to 0.68 in the construction's, and 0.16 to 0.27 at height (table E): the
origin's quotients are prime because they are small (below `p'^2 / g_0`, so prime with the density of
primes there), not because of anything the recursion forces. The one thing the recursion forces
beyond D4 is the depth: at most `floor(2 log_5 p')` at the origin (3, 4, 5 realised exactly as
predicted) against 5 to 7 at the record runs. Neither is a covering constraint.

**How a killer kills, in the dilation coordinate.** A killer must strike every twin column, and a twin
column has no divisor among the gears, so every strike on it is phantom (phantom-only columns `>=`
twins, 7 of 7). At `p = 17` the two twins 52 `(311, 313)` and 58 `(347, 349)` are out of reach of the
gears below the section's length at their real teeth (`52, 58 = 2, 3 (mod 5)`, `3, 2 (mod 7)`,
`8, 3 (mod 11)` against the teeth `+-1, +-1, +-2`) and the two gears above it, 13 and 17, are pinned by
D2 to their multiples in the section: `13 x 23 = 299` (column 50), `13 x 25 = 325` and `17 x 19 = 323`
(both in column 54). The killers move 13 or 17 (or 11) off those pins onto the twins. The real strike
map (`og_maps.py`): `49: 5x59 | 50: 7x43, 13x23 | 51: 5x61 | 52: OPEN | 53: 11x29 | 54: 5x65, 13x25, 17x19 |
55: 7x47 | 56: 5x67 | 57: 7x49, 11x31 | 58: OPEN | 59: 5x71`. At `p = 29` a single gear kills, uniquely
among 1,995,840 members: the two twins 143 `(857, 859)` and 147 `(881, 883)` are symmetric about column
`145 = 5 x 29` (`143, 147 = -+2 (mod 29)`), so the tooth pair `+-2` of gear 29 covers both, while the
real tooth pair `+-5` sits at `29 x 29 = 841` (column 140, the square, outside the section) and
`29 x 31 = 899` (column 150), and losing column 150 costs nothing because `901 = 17 x 53` strikes it
too. That is the smallest instance of the dilation pin as the whole protection of a section: one gear,
one pair of multiples, and the twins at the pin's mirror image.

**The nesting is exact and is not a constraint.** D3 relabels the union of the dilates by least
striker; the union is unchanged. Every column struck by `g` is `g i +- k_g` for SOME `i`; the nesting
says which `i` gives the least strike, and the cover does not care which strike is least. So the
nesting forbids nothing that the real teeth do not already forbid; what it adds is the recursion, whose
content is D4 and D5.

Prior art, one line each, at the stop points: the semiprime count of the tail strikes on core-open
columns is the kernel's `CoreLeftover.leftover_eq_card_twins` and, as a count of `g x prime` in a short
interval, Brun / Chen territory (proof_skeleton.md IV.1): stopped. The share of prime quotients as a
function of the section's top is the density of primes among rough numbers below `p'^2 / g_0`
(Legendre / Buchstab): stopped.

---

## 4. The hunch as a statement and its test

The owner's hunch, in the construction's words:

- **H1 (nested fragments).** "The patterns are nested fragments of the lower pattern, incorporating
  folds of all the gears below." Statement: the struck pattern of `{5..q}` is the disjoint union over
  its gears `g` of the image of the open pattern of `{5..g^-}` under `(i, s) -> g i + s k_g`, sides
  swapped iff `g = -1 (mod 6)`. This is D3. PROVED; 0 mismatches over 198,798 members (P6).
- **H2 (no higher gear can cancel the 2, 3 fold).** "None of which are ever able to cover 2, 3 because
  no higher primes are a multiple of 6, 2, or 3." Statement: every dilate `g.S` lies inside `S` and
  has both a left and a right member in every `g` consecutive columns (D2); a gear divides at most one
  member of any column (it does not divide 2), so no dilate covers a column and a column is struck on
  both sides only by two different gears. The fold's pattern (both members present in every column)
  is preserved in every dilate and in every union of dilates. PROVED by D2 and the definitions.
- **H3 (they just cover some of its runs with theirs; what covering would strike a whole section).** A
  whole section is struck iff every column `j` in `a + 1 .. b - 1` is `g i + s k_g` for some gear `g <= q`
  and some survivor `6 i + s` with `g (6 i + s)` inside the section, i.e. iff every column holds a
  composite. With the phases free (the tooth family) that is an exact cover of `L` columns by two
  classes per gear; with the real teeth it is the same cover with every phase fixed at the square
  column (length_face.md LF1, phase vector 0). The nesting does not shrink the union (section 3), so
  it forbids the cover in no way that the real teeth do not; and whether the real teeth forbid it is
  step 8.

**The smallest instance where the nesting (the pinned dilates) is the reason a section holds a twin,
i.e. where free classes kill and the dilates do not:** the finer section at `p = 17`, columns 49..59
(15 of 1,440 family members kill; the real member leaves 52 and 58), with the mechanism of section 3;
next `p = 29` (6,030 of 1,995,840; one gear's pin is the whole protection) and every cut 37..53. For
step 8's own sections no such instance is known: at the three proved links the free classes cannot
kill either (capacity `2 ceil(18/5) + 2 ceil(18/7) = 14 < 18` at `[9, 121)`; the free-class record
`h_2 = 60 < 135` at `[25, 841)`, `213 < 459` at `[49, 2809)`, length_face.md table 1.2), and at every
`q` where `h_2` is known (`q <= 73`) it is below `q^2 / 6` (ratio 0.68 falling to 0.49). So for step 8
the nesting has never been seen to be the reason; the real teeth's separation and pins are needed for
the finer statement 8e, and for 8 the free-class conjecture (Ziller-Morack Conjecture 6, open) would
already suffice. Owner's prediction on the scorecard: the nesting is exact (held); it is the protection
where the free classes fail (held for 8e at 17, 29, 37..53; no instance for 8).

---

## 5. The dilation structure characterises the machine; the three counter-machines break one axiom each

**C (characterisation).** Let `S` be the survivors of the fold `{2, 3}`, i.e. every `+-1 (mod 6)`. A
dilation machine on `S` is a set of gears `G` inside `S` with struck `= union of g.S`, and the hand-up
holds when every gear is an open element of its own section and every open element of a section is a
gear. Then `G` is exactly the set of elements of `S` that are not products of two smaller elements of
`S`, i.e. the primes `>= 5`: an element `n = g m` with `g, m in S`, `1 < g <= m`, is struck by the gear
below `g` that divides `g` (repeat the step on `g` until it stops, at most `log_5 g` times), so it is not open and not a gear; an element with
no such factorisation is divisible by no gear below it, so it is open and a gear. Since `S` is all of
`+-1 (mod 6)`, every composite survivor factors inside `S`, and the irreducibles are the primes. So
**dilation + finite fold + hand-up = the real machine**, with the square-root rule a theorem (skeleton
3). PROVED. The three counter-machines on record and the one built here:

| counter-machine | dilation (strikes are multiples) | finite fold (`S` = every `+-1 mod 6`) | hand-up / gears below the square root | violates 8 | where |
|---|---|---|---|---|---|
| the tooth family (first_realisation.md 3.5) | **no** (phantom strikes) | yes | yes | yes, for 8e at 17, 29, 37..53 | table C |
| V17 (phase_zero.md) | yes | yes | **no** (the section's own twins as gears) | yes, every section | table C |
| `M` = generated by 5 and the primes `= 1 (mod 6)` | yes | **no** (`M` is not periodic) | yes | yes, from `[25, 961)` on | table F |

Each breaks exactly one axiom and violates 8; the machine with all three is the real one (C). This is
the exact sense in which "dilation + hand-up + square-root rule" cannot be the whole reason: `M` has
all of D1-D6 (its strikes are multiples, its least strikers are least irreducible factors, its
quotients are in `M`, its square-root rule and hand-up are the monoid's, its sections are between
consecutive irreducible squares) and it holds no twin irreducible beyond `(5, 7)`.

---

## 6. Proof attempt, and the exact obstruction

**6.1 The attempt.** Take the section `[p_k^2, p_{k+1}^2)` and suppose every column struck. By D5 each
column `j` holds a composite `n_j = g_j m_j`, `5 <= g_j <= sqrt(n_j) <= m_j`. Sort the columns by
`g_j`. The gears `g <= L` (`L` the section's length in columns) strike their two classes periodically
inside the section; the gears `g > L` strike at most two columns each, at the pinned places `g x m`
with `m` a survivor in `(p_k^2 / g, p_{k+1}^2 / g)`, and for `g > p_{k+1}^{2/3}` those `m` are prime (D4).
So a fully struck section says: every column left open by the gears `<= L` (the core-open columns)
holds a semiprime `g x m'` with `g > L` a gear and `m'` a survivor with no factor below `g`. That is the
core / tail split of proof_skeleton.md III.1 (kernel `loaded_record_rule`,
`leftover_eq_card_twins`) read at the origin. The next step compares the number of core-open columns
with the number of such semiprimes in the section: a count, and every count is matched by a machine
with no open slot (proof_skeleton.md IV.2). Stop line: the semiprime count is Brun / Chen; the
core-open count is the sieve's main term; both are on the register. The attempt ends at a count.

**6.2 The exact obstruction, as a construction with its smallest instance.** Any argument that uses
only D1-D6, the hand-up and the square-root rule applies verbatim to the monoid `M` of table F
(replace "prime" by "irreducible in `M`", "survivor" by "element of `M`"), where its conclusion is false
at the section `[25, 961)`: twenty columns hold two elements of `M` (`(35, 37)`, `(65, 67)`, `(95, 97)`,
`(125, 127)`, ...), seventy-four of `M`'s irreducibles lie in the section, and no column holds two of
them, because every left member in `M` carries an odd number of factors 5. So a proof of 8 must use the
one axiom `M` lacks: **the finite fold, `S` = every `+-1 (mod 6)`.** In the construction the finite
fold is used in exactly one place: skeleton 2, the strike-class law (a gear strikes exactly the two
columns `+-k_g` in every `g` consecutive columns, and every column has both members in `S`). That law
is the sieve's input (the counts `A_d` of length_face.md 0.1 are its consequences) and the tooth
family's premise (the classes with the pins removed). And at the origin the sieve's input does not
separate the open set from a set that is empty on the section: length_face.md LF3 and LF4 (the parity
twin `O^-`, empty below `(q'^2 - 1)/6`, with `max |S_d| / sqrt(A_d) <= 2.89` over 1,326 cells at
`q = 23..53`).

Plainly: **it is the parity problem again**, now in the dilation coordinate, with the obstruction
located one step more precisely than before: the dilation axioms (multiples, quotients, the
square-root rule, the hand-up) are shared with a machine that violates 8 at `[25, 961)`; the axiom that
excludes that machine enters the construction only as the strike-class law; and the strike-class law's
content at the origin is the sieve data that LF3 / LF4 show insufficient. The smallest instance of the
obstruction: `M` at `[25, 961)`; the smallest instance of the sieve's insufficiency at the origin: `O^-`
at `q = 11` (empty below column 28 against a target of 20 columns, length_face.md 3.1).

What is NOT excluded: a use of the strike-class law that is not a count and not a cover of free phases
(a use of the pins' arithmetic, as at `p = 29`, where the protection is one gear's multiples). Nothing
on record is such a use, and this lane found none: the pin at 29 protects because `901 = 17 x 53` covers
the column the pin abandons, which is again a fact about which composites exist in the section.

---

## 7. What is new

1. **The dilation form written out (D1-D6)** and the exact statement of what it adds over free classes:
   the census invariant D5 (every strike is `g_0 x m` with `5 <= g_0 <= sqrt(n) <= m`) and the recursion's
   size law D4; at the level of columns it is the real teeth and nothing more.
2. **The nesting identity D3 in column form** with the affine map `(i, s) -> g i + s k_g` and the side
   swap at `g = -1 (mod 6)`: the owner's "nested fragments" as an exact, proved statement, checked at
   198,798 members with 0 mismatches. Its count is Buchstab / Legendre (noted, not used).
3. **The dilation census of the origin against height** (tables A, B, E): prime quotients at 0.64-0.83
   in the finer sections and 0.57-1.00 at the construction's first links, falling to 0.45 and 0.38 at the
   second links and to 0.16-0.27 at the record runs; below-cube strikes 16-30 % at the origin and 0 at
   height; depth at most 3, 4, 5 at the origin exactly as `floor(2 log_5 p')`, up to 7 at height.
4. **The census difference between a real and a killed section** (table C): phantom strikes 4 to 156
   per killer, phantom-only columns 2 to 80 against 2 to 13 twins; V17's added strikes have quotient 1.
5. **The distance to the nearest killer** (table D): `d = 2, 1, 5, 2` at `p = 17, 29, 37, 41`, `>= 6` at
   43, 47, 53, none at 11, 13, 19, 23, 31; **the one-gear killer at 29**, unique among 1,995,840, with its
   mechanism (twins mirror-symmetric about `5 x 29`, the abandoned pin column re-covered by `17 x 53`).
6. **The characterisation C** and the completed triple of counter-machines, each breaking exactly one
   axiom; **the third counter-machine `M`** (dilation, hand-up and square-root rule intact, finite fold
   broken, 8 violated at `[25, 961)` with 20 two-member columns and 0 twin irreducibles).
7. **The parity mix inside one periodic dilation machine** (`S_H`, `H = {+-1} mod 12`): 558
   twin-irreducible columns in `[121, 17161)`, of which 139 prime-prime, 303 mixed, 116
   semiprime-semiprime: the sets a sieve cannot tell apart, living together in a machine with every
   dilation axiom and the finite fold, where "8" holds abundantly. And `S^+` (irreducibles = the
   semiprimes), the parity twin as a monoid, where "8" also holds (11,969 columns in `[625, 395641)`).

Prior art, one line each: the monoid machines are Beurling generalised number systems embedded in the
integers (Beurling 1937); that sieve methods cannot separate their irreducibles by parity is the
classical parity phenomenon (Selberg 1949, cited at W52 and in length_face.md); no web check was run in
this lane (no web access), so items 5-7 carry the status "prior-art check not yet run". Item 2's count
is Buchstab / Legendre; the two-prime lemma is in the kernel.

---

## 8. Verdict and scorecard

**ROOT.** The dilation form is exact and is the real teeth (D2) plus the multiplicative invariant of the
origin (D5); the owner's nesting is true, proved and exact (D3), and is not a covering constraint
(section 3); the smallest instance where the pinned dilates protect a section that free classes kill is
the finer section at `p = 17`, with `p = 29` the smallest one-gear instance; for step 8's own sections
no instance is known where the dilates are needed. The proof from the dilation structure ends at a
count (6.1), and the exact obstruction is the monoid `M`, which has every dilation axiom and violates 8
at `[25, 961)`; the axiom that excludes `M` is the finite fold, which enters the construction only as
the strike-class law, whose content at the origin is the sieve data LF3 / LF4 already show insufficient.
The parity problem again, in a third coordinate; where the difficulty moved: from "a mechanic of the
origin" to "a use of the strike-class law that is neither a count nor a free-phase cover", with the
one-gear pin at 29 as the only exhibited non-count protection, and it rests on which composites exist.

- PROVED (in writing): D1-D6, H1, H2, C. Kernel: none new (D2 is `real_teeth` / `blockedZ_eq_family`;
  D4's first clause is `primeOrSemiprime_of_rough_lt_cube`).
- MEASURED: tables A-F; the new readings (`d(17) = 2`, `d(29) = 1`, the origin shares) by two methods.
- ROOT: 6.1 and 6.2.

| # | prediction | verdict | evidence |
|---|---|---|---|
| P1 | gates | **CONFIRMED** 5 of 5 | `og_gate.py` |
| P2 | `n < g_0^3` forces a prime quotient, 0 exceptions; both cases above the cube | **CONFIRMED** 19 of 19 sections, 0 exceptions; both cases in every section with `>= 19` columns (at `p = 11`, 7 columns: 2 prime, 1 composite above the cube) | tables A, B |
| P3 | maximum depth 3, 4, 5 by `floor(2 log_5 p')` | **CONFIRMED** 12 of 12 | table A |
| P4 | the census invariant; phantom-only `>=` twins; V17's added-only columns = twins with quotient 1 | **CONFIRMED** 7 of 7, 7 of 7, 7 of 7 | table C |
| P5 | height `pq < 0.35`, origin `pq > 0.5` | **CONFIRMED** 4 of 4 (0.27, 0.19, 0.18, 0.18 against 0.77, 0.71, 0.70, 0.64) | table E |
| P6 | the nesting identity, 0 mismatches | **CONFIRMED** after restricting the identity to the section's columns (the first run: 1 mismatch at each of 9 sections, all the member `p'^2 - 2` of column `b`); 0 of 198,798 | tables A, B |
| P7 | `d(17) = 2`, `d >= 2` everywhere, a tail gear moved in every nearest killer | **`d(17) = 2` CONFIRMED; `d >= 2` REFUTED at 29 (`d = 1`, unique); the tail clause holds at 17, 29 and is vacuous from 37 on** | table D, `og_brute.py` |
| P8 | `S^+` holds 8 with `>= 100` columns; the `1 (mod 4)` monoid violates | **CONFIRMED** (11,969; and `M` added as the sharper violator) | table F |
| owner | the nesting is exact; it is the protection where free classes fail | exact: **held**; protection: **held for 8e** at 17, 29, 37..53; **no instance for 8** | sections 3, 4 |

---

## 9. Dead ends (bricks), each with its refuting instance

| idea | dies at | instance | why it cannot be revived |
|---|---|---|---|
| the nesting as a covering constraint beyond the real teeth | every section | D3 relabels the union `g.S` by least striker; the union is the real-teeth cover | a relabelling changes no cover |
| the recursion's end (the least prime factor) as a lever | the tail strikes | `n < g_0^3` gives `g_0 x prime`; the next step is the semiprime count | the two-prime lemma plus a count (IV.2) |
| the prime-quotient share as a signature of the origin | base 7 link 2 | 0.38 at top `7.9 x 10^6`, falling with the top; 0.16-0.27 at height | it is the prime density below `p'^2 / g_0`, a count |
| "one moved gear can never kill" (the pins as a two-gear protection) | `p = 29` | tooth 2 of gear 29 kills, unique among 1,995,840; twins at `145 -+ 2`, `145 = 5 x 29` | the pin's mirror image can hold both twins |
| the dilation axioms as the whole reason | `[25, 961)` | the monoid `M` (5 and the primes `= 1 (mod 6)`): 20 two-member columns, 0 twin irreducibles | `M` has every dilation axiom; only the finite fold excludes it |
| the finite fold as a non-count lever | the construction | it enters only as skeleton 2, the strike-class law = the sieve's input | LF3 / LF4: the input does not separate `O` from `O^-` at the origin |
| the parity twin as a monoid violating 8 | `S^+`, `S_H` | 11,969 and 558 twin-irreducible columns in their first sections | the sieve-indistinguishable sets satisfy "8"; the barrier is the method, not the set |

## 10. Open items on the part alone, sorted

- **Closed here.** D1-D6, H1-H3, C; the census at the origin and at height; the killers' census; `d(p)`
  at `p <= 41`; the three-counter-machine table.
- **Measurement with no structural content.** `d(p)` at 43, 47, 53 (`>= 6`; a cover search with a
  cost function would decide it); the prime-quotient share along the chains beyond link 2; `S_H` for
  other `H`.
- **Root question in disguise.** 6.1's count; any bound on the core-open columns of the section against
  the section's semiprimes.
- **Genuinely open on the part alone, with the attack.** A use of the strike-class law at the origin
  that is neither a count nor a free-phase cover: the candidate object is the pin structure of the tail
  gears (`g x m` with `m` small, `m` prime for `g > p'^{2/3}`), which at `p = 29` is the whole protection.
  An attack: for each cut, list the core-open columns and the tail pins that cover them, and ask whether
  the pins' positions (`g x m`, `m` a prime in `(p^2/g, p'^2/g)`) are forced to miss a core-open column
  by an arithmetic relation rather than by count; the tooth family with only the tail gears freed is
  the control. This is the position face (IV.3) restricted to the tail, and it is a new node if opened.

## 11. Files

- `research/anchor235/r77/og_common.py` - the constructions (fold, columns, sections, spf table, strikes)
- `research/anchor235/r77/og_gate.py` - P1
- `research/anchor235/r77/og_census.py` - tables A, B and P6; `results/census.json`, `results/census.log`
- `research/anchor235/r77/og_family.py` - table C; `results/family.json`, `results/family.log`
- `research/anchor235/r77/og_nearest.py`, `og_brute.py` - table D and its second measurement;
  `results/nearest.json`, `results/nearest.log`, `results/brute.json`
- `research/anchor235/r77/og_height.py` - table E; `results/height.json`, `results/height.log`
- `research/anchor235/r77/og_monoid.py` - table F; `results/monoid.json`
- `research/anchor235/r77/og_maps.py` - the strike maps of section 3; `results/maps.txt`
- `results/` is untracked; nothing is committed; `theory_tree.md` and `proof_skeleton.md` are not edited.
