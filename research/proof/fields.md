# The fields: the struck set split by the count of prime factors, each field as a construction, their relations, the section against each field, and the verdict

Theorist lane (Fable), 2026-09-11 (late). Parent: `research/proof/proof_skeleton.md` Part IV.4 (the
demand: a mechanic of the origin that is not a count) and `research/proof/the_wall.md` 5o. The owner's
idea (2026-09-11, verbatim in the brief): put the different interactions into separate fields, the gears'
home strikes in one field, the squares in another, the composites with exactly 2, 3, 4, ... factors each
in their own field, the overlay of all fields being the full set of hits; then analyse each field's own
structure as the machine grows, how the fields relate, the section relative to each field, and how each
sits on the 6-cycle, looking for a location rule in a field's own coordinates. Parallel lane, not
duplicated: `research/proof/fold_mechanic.md` (the two classes of the fold; its F1-F7 are cited, not
re-derived). Scripts in `research/anchor235/r79/` (prefix `fd_`); outputs in
`research/anchor235/r79/results/` (untracked). Every number this document relies on is written into it.
Nothing is committed; the tree and the proof document are not edited.

Vocabulary by construction (section 0.1). Not used: the four forbidden words. The owner's word **field**
is the new object and is defined in 0.1 before use.

---

## 0. Pre-registered (written before any script of this branch ran; the scorecard is filled in section 9)

### 0.1 The objects, by construction

- **The line**: the counting numbers. **A gear** of size `g` strikes `g, 2g, 3g, ...`. **The fold** is the
  two gears `2, 3`; its **survivors** `S` are the numbers neither strikes: `1, 5, 7, 11, 13, 17, 19, 23, 25,
  ...`, i.e. every `6j - 1` and `6j + 1`.
- **Column** `k` (`k >= 1`) is the pair `(6k - 1, 6k + 1)`; `6k - 1` is its **left member** and `6k + 1` its
  **right member**. Every survivor above 1 is a member of exactly one column. The **class** of a survivor is
  `+1` if it is `= 1 (mod 6)` (a right member) and `-1` if `= 5 (mod 6)` (a left member); the class of a
  product is the product of the classes (fold_mechanic.md 0.1, F7).
- **A prime gear** is a prime `p >= 5`; it is a survivor, of class `eps_p`, in column `k_p` (`p = 6 k_p + eps_p`).
  `P_-` are the primes of class `-1` (`5, 11, 17, 23, 29, 41, ...`), `P_+` those of class `+1` (`7, 13, 19,
  31, 37, 43, ...`).
- **`Omega(n)`** is the number of prime factors of the survivor `n` counted with multiplicity (`Omega(1) =
  0`); **`Omega_-(n)`** counts the factors in `P_-` with multiplicity, `Omega_+(n)` those in `P_+`.
- **Field `j`** (`j >= 1`), written `F_j`, is the set of survivors with `Omega = j`: `F_1` is the primes `>= 5`
  (each gear's own size, its **home strike**); `F_2` the products of exactly two primes `>= 5` (`25, 35, 49,
  55, 65, 77, 85, 91, 95, 115, 119, 121, ...`); `F_j` the products of exactly `j` primes `>= 5`. **The
  square field** `Q = {p^2 : p >= 5 prime}` is the diagonal of `F_2`. The fields partition `S \ {1}`.
- **A field hits a column** `k` (on the left / on the right) if its left / right member lies in the field.
  `H_j` is the set of columns hit by `F_j`; a column is **blind to `F_j`** if `F_j` does not hit it, `B_j` is
  the set of such columns. **The overlay** of fields `>= 2` is `F_2 u F_3 u ...` = the composite survivors;
  its column set is `H_2 u H_3 u ...`.
- **The dilate** of a set `A` by `g` is `g.A = {g a : a in A}`.
- **The engine at `q`** is `{5..q}`, the prime gears from 5 to `q`; it **strikes** a survivor it divides;
  `q' = nextprime(q)`; its **sight** is `[1, q'^2)`: there a survivor is struck iff it is composite
  (skeleton 3), so on the sight **the struck set is exactly the overlay of fields `>= 2`**.
- **Sections.** The finer section at the prime `p` is the columns strictly between `p^2` and `p'^2`
  (`p' = nextprime(p)`), engine `{5..p}`; the sections of the construction are `[p_k^2, p_{k+1}^2)`
  (skeleton 4), engine `{5..q}`, `q = prevprime(p_{k+1})`, columns `a + 1 .. b - 1` with `6a + 1 = p_k^2`,
  `6b + 1 = p_{k+1}^2`. Bases and links as in origin_mechanic.md 0.1.
- **A twin column**: both members in `F_1`. **A both-`F_j` column**: both members in `F_j` (`j = 1` is the
  twin). **Struck column**: some member in a field `>= 2`. So: **column `k` is a twin iff `k` is blind to every
  field `>= 2`**, i.e. `k in B_2 n B_3 n ...` (definition; the target of task (3) is a rule for that
  intersection that is not a count).
- **The column signature** of a struck column is the multiset of `Omega` values of its composite members
  (`(2)`, `(3)`, `(2, 2)`, `(2, 3)`, ...). A column is **hit by exactly one field** if its signature has one
  distinct value.
- **The mirror about `6gm`** (a multiple of `6g`): the map `n -> 12gm - n`. It sends `S` to `S`, swaps the
  classes, sends column `gm + t` to column `gm - t` with sides swapped, and sends multiples of `g` to
  multiples of `g`. A **symmetric pair of `F_j` about `6gm`** is `{n, 12gm - n}` with both in `F_j`; it is
  **through `g`** if `g | n`.
- **Depth of a field in a section**: `F_j` is empty below `5^j` (the least product of `j` primes `>= 5`),
  so a section below `p'^2` holds fields `2 .. J` with `J = floor(2 log_5 p')` (origin_mechanic.md D4, P3;
  leftover_depth.md S17 gives the sharper threshold `nextprime(t)^j` for `t`-rough members). Cited, not
  re-measured.
- **The cube-core of the section at `p'`**: the gears below `p'^{2/3}`; a column is **cube-core-open** if no
  such gear strikes it.
- **Height**: the four record runs of origin_mechanic.md table E (m23 at column 12,694,429, 33 columns;
  m29 at 200,906,186, 42; m31 at 1,468,940,243, 57; m37 at 90,816,580,903, 87), every column struck.

### 0.2 Theory: the exact facts, stated before measuring, proved in section 1

- **T1 (the nesting per field).** `F_j = union over primes p >= 5 of p.F_{j-1}` for `j >= 2`, and with the
  least prime factor as label the union is disjoint: `F_j = disjoint union over p of p.(F_{j-1} n R_p)`,
  `R_p` the survivors with no prime factor below `p`. (Origin_mechanic.md D3 per depth.)
- **T2 (every field is the primes, dilated by 5).** `F_j n 5S = 5.F_{j-1}` for every `j >= 2` (a survivor
  divisible by 5 with `Omega = j` is `5 m` with `Omega(m) = j - 1`, and conversely). Hence `F_{j-1} =
  (F_j n 5S) / 5`: **each field determines the field below it exactly, and `F_1` is recovered from any `F_j`
  by `j - 1` such steps.** So a location rule (a closed form for the hit set) for any field `F_j` gives one
  for the primes, and none exists for any field unless it exists for `F_1`. The same for the square field:
  `F_1 = sqrt(Q)`.
- **T3 (no field is periodic; the overlay on the sight is).** A subset `A` of `S` is periodic with period `P`
  (`6 | P`) if for `n in S`, `n in A iff n + P in A`. `F_1` is not periodic (`5 in F_1`, `5 + 5P = 5(1 + P)`
  is composite). If `F_j` were periodic with period `P`, then `F_j n 5S` would be periodic with period `5P`
  (`n in 5S iff n + 5P in 5S`), so `F_{j-1}` would be periodic with period `P` by T2; by induction `F_1`
  would be. So **no field is periodic**. The overlay of fields `>= 2` restricted to the engine's sight is the
  struck set of `{5..q}`, periodic in the column with period `prod g` (skeleton 2). **The fields are an
  aperiodic partition of a periodic set.**
- **T4 (the class rule per field).** A survivor's class is `(-1)^{Omega_-}` (fold_mechanic.md F7). So `F_j`
  hits a right member iff the member has an even number of class `-1` factors, a left member iff odd. For
  `j = 2`: right members by same-class pairs (`P_- P_-` and `P_+ P_+`), left members by cross-class pairs
  (`P_- P_+`). **The square field hits only right members** (`p^2 = 1 (mod 6)`), so `Q` is blind to every
  left member.
- **T5 (the mirror law of a dilate).** `g(6m - 1) = 6(gm - k_g) - eps_g` and `g(6m + 1) = 6(gm + k_g) +
  eps_g`: the two members of column `m`, dilated by `g`, land in columns `gm -+ k_g`, symmetric about column
  `gm` (the number `6gm`), the lower one on the side `-eps_g`, the upper on the side `eps_g`. More generally
  the mirror `n -> 12gm - n` maps `g(6m + i)` to `g(6m - i)`. Hence: **the symmetric pairs of `F_j` through
  `g` about `6gm` are exactly the dilates by `g` of the pairs `{6m - i, 6m + i}` (`i = 1, 5, 7, 11, 13, ...`)
  with both members in `F_{j-1}`; the innermost radius `i = 1` is the column `m` with both members in
  `F_{j-1}`, a twin column for `j = 2`.** The pairs `{6m - i, 6m + i}` of primes are the representations
  `12m = r + r'` (prior art: Goldbach; Hardy-Littlewood's conjecture for the count; noted, not derived). So
  every field `j >= 2` has mirror axes at every multiple of `6g` for every prime `g`, and the mirror
  symmetry through `g` at `6gm` is the additive pairing of `F_{j-1}` about `6m`.
- **T6 (the blind sets of the deep fields).** A member `n` of `F_j` in the section below `p'^2` has least
  prime factor `<= n^{1/j} < p'^{2/j}`. So **`F_j` is blind to every column open under the gears `<
  p'^{2/j}`**: `B_j` contains the open set of the machine `{5..p'^{2/j}}`, a periodic set with a closed
  form (two residues per gear). For `j = 3` that is the cube-core; on the cube-core-open columns of the
  section every strike is `F_2`, and a cube-core-open column is a twin iff it is blind to `F_2`. (The
  two-prime lemma, kernel `CoreLeftover.primeOrSemiprime_of_rough_lt_cube`, in the fields' coordinate;
  cited.) **The square field is empty in every finer section** (no prime strictly between `p` and `p'`);
  in the construction section `[p_k^2, p_{k+1}^2)` it is `{r^2 : r prime, p_k < r < p_{k+1}}`, the squares
  of the gears of machine `k` above `p_k`.
- **T7 (the expected verdict, stated in advance).** The twin columns are the complement of the overlay of
  fields `>= 2`; a partition of the overlay does not change its complement, so no rule stated on the
  fields' labels alone can locate a column of the complement except through T2, i.e. through the
  primes themselves, which is step 8 restated. What a sieve can compute about the labels is the overlay's
  divisibility counts, and length_face.md LF3 / LF4 exhibit a set with the same counts that is empty on
  an interval of the target's length: the parity twin. The lane expects (5) to end there, with the
  smallest instance `O^-` at `q = 11` (empty below column 28 against a target of 20 columns, length_face.md
  3.1), and expects the mirror symmetry the owner anticipates to be real, growing, and exactly the Goldbach
  pairing of T5, whose innermost radius is the twin.

### 0.3 Predictions, each with the number that refutes it

- **P1 (gates).** (a) The column sieve gives the eight twin gears of machine 2 in `[9, 121)` at columns
  `2, 3, 5, 7, 10, 12, 17, 18`. (b) On base 3 section 4 `[16129, 260467321)` the members with no prime
  factor `<= 301` number, by `Omega = 1, 2, 3, 4`: `14,218,065 / 10,991,941 / 192,920 / 0`
  (leftover_depth.md, L' = 50); the section's twin count is `1,027,948`. (c) The depth histograms of
  origin_mechanic.md table A at `p = 23` (`{2: 33, 3: 9, 4: 1}`) and `p = 53` (`{2: 65, 3: 27, 4: 5, 5:
  1}`), and the twin counts 276, 29, 6,224, 74, 48,249, 455 at base 3 link 2, base 5 links 1-2, base 7
  links 1-2, base 13 link 1. Refuted by any mismatch.
- **P2 (identities as gates, 0 mismatches).** On every section: `F_j n 5S = 5.F_{j-1}` (T2); the class
  rule (T4); `Q` on right members only; the mirror law T5: for every gear `g` of the engine and every
  `m`, the columns `gm -+ k_g` are hit through `g` by `F_j` iff column `m` is both-`F_{j-1}`. Refuted by
  one mismatch.
- **P3 (the deep fields' confinement, T6).** On every finer section `p = 11..53` and every construction
  section run, every `F_j` member with `j >= 3` lies in a column struck by a gear `< p'^{2/j}`: 0
  exceptions. The square field is empty in every finer section and equals the squares of the primes in
  `(p_k, p_{k+1})` in every construction section. Refuted by one exception.
- **P4 (the class bias per field).** In the three largest sections run (base 3 link 3, base 7 link 2,
  base 5 link 2) the right / left member count of `F_2` exceeds 1, of `F_3` is below 1, of `F_4` exceeds
  1 (the sign alternates with `j`; prior art if it holds: Meng's alternating bias for products of `k`
  primes, noted at the stop). Refuted if the sign fails to alternate in any of the three.
- **P5 (columns hit by exactly one field, origin against height).** In the finer sections at `p = 23..53`
  the share of struck columns hit by exactly one field is at least 0.75 and field 2 is that field in at
  least 0.6 of them; at the four record runs the share is below 0.65 and field 2's share of it below 0.35.
  Refuted by a finer section below 0.75 or a run above 0.65.
- **P6 (the mirror symmetry and its growth).** (a) For each gear `g <= 13` and each section, the number of
  symmetric `F_2` pairs through `g` about `6gm`, summed over the axes `m` inside the section, equals the
  number of prime pairs `{6m - i, 6m + i}` with both members in the cofactor range: an identity, 0
  mismatches. (b) The mean number of symmetric pairs per axis (through `g = 5`) grows along the chain
  from base 3: link 1 < link 2 < link 3; and the innermost-radius share (twins) of all symmetric pairs
  falls below 0.05 at link 3. (c) Off the multiples of `g` the symmetric-hit rate of `F_2` about `6gm`
  equals the density of `F_2` among the survivors of the section to within 3 standard deviations in at
  least 9 of the 12 finer sections: no symmetry beyond the `g`-part. Refuted by (a) one mismatch, (b) a
  non-monotone mean or a twin share above 0.05, (c) more than 3 sections off by 3 sd.
- **P7 (the cube-core against field 2).** On every finer section `p = 23..53`, every strike on a
  cube-core-open column is in `F_2` (0 exceptions, T6), the cube-core-open count is between 1.3 and 3
  times the twin count, and every such `F_2` strike is `g x r` with `g >= p'^{2/3}` and `r` prime with
  `g <= r < p'^2 / g`. Refuted by an exception or a ratio outside `[1.3, 3]`.
- **P8 (the field-2 hit set has no rule but the primes).** Stated as T2; the measurement is P2. The
  section's twin columns computed as `B_2 n ... n B_J` (from the fields) equal the twin columns computed
  directly (from primality): 0 mismatches (a gate on the definitions).

Owner's predictions on the scorecard (from the idea, as this lane reads it): (a) each field has its own
structure with patterns emerging as the machine grows: the lane expects the patterns of every field to be
the primes' patterns dilated (T2), so what emerges is `F_1`'s structure at scale `1/p` in every field;
(b) the fields relate by dilation: expected to hold exactly (T1); (c) the section relative to each field
pinpoints locations: expected to hold for the deep fields (confined to the small gears' teeth, T6) and
for the square field (the cuts are the square field, T6), and not for `F_2`; (d) growing mirror symmetry
in the composite fields: expected to hold as the Goldbach pairing (T5), growing with `m`, the twin being
the innermost radius; (e) rules in each field to locate twin gaps: expected to be refuted (T7).

---

## 1. The fields as a construction: the exact facts, each proved

Everything here is proved by the construction written next to it; the measured checks (`fd_gate.py`,
`fd_fields.py`) are gates on the scripts, not evidence for the root.

**E1 (the fields partition the struck set on the sight).** On the sight `[1, q'^2)` of the engine `{5..q}` a
survivor is struck iff it is composite (skeleton 3), and a composite survivor has `Omega >= 2`; so the
struck set on the sight is `F_2 u F_3 u ... u F_J` with `J = floor(2 log_5 q')` (`5^J <= q'^2 < 5^{J+1}`),
a disjoint union. Every twin column of the sight is a column blind to all of `F_2 .. F_J`, and conversely.
PROVED (definitions). Gate: the twins computed as `B_2 n ... n B_J` equal the twins computed by primality,
0 mismatches at 20 sections and 13 sights (table 2.1, column "tw").

**E2 (the nesting per field, T1).** `F_j = union over primes p >= 5 of p.F_{j-1}`: a product of `j` primes is
`p` times a product of `j - 1`, for any of its prime factors `p`; with `p` = the least prime factor the
representation is unique, so `F_j = disjoint union over p of p.(F_{j-1} n R_p)`. PROVED. (Origin_mechanic.md
D3, restricted to one depth.)

**E3 (every field is the primes dilated by 5, T2).** `F_j n 5S = 5.F_{j-1}` (`j >= 2`): a survivor `5m` has
`Omega(5m) = Omega(m) + 1`, so `5m in F_j` iff `m in F_{j-1}`; and every multiple of 5 in `S` is `5m` with
`m in S`. Hence `F_{j-1} = (F_j n 5S)/5` and, iterating, `F_1 = (F_j n 5S n 5^2 S ... )/5^{j-1}`: **the
primes are read off any field by dividing its multiples of `5^{j-1}` by `5^{j-1}`.** Likewise
`F_1 = sqrt(Q)`. PROVED. Gate: `F_j n 5S = 5.F_{j-1}` with the cofactor's `Omega` computed separately,
0 mismatches at 20 sections and 13 sights (table 2.1, "T2"). Consequence: a closed form for the hit set of
any field (its "location rule") is a closed form for the primes; **no field has a location rule that the
primes do not have.**

**E4 (no field is periodic; the overlay on the sight is, T3).** Call `A subset S` periodic with period `P`
(`6 | P`) if `n in A iff n + P in A` for all `n in S`. `F_1` is not periodic: `5 in F_1` but
`5 + 5P = 5(1 + P)` is composite and in `S` (`P = 0 (mod 6)`). If `F_j` (`j >= 2`) had period `P`, then
`F_j n 5S` would have period `5P` (`n in 5S iff n + 5P in 5S`, and `n in F_j iff n + 5P in F_j` since
`5P` is a multiple of `P`), so `F_{j-1} = (F_j n 5S)/5` would have period `P`; descending, `F_1` would.
So **no `F_j` is periodic, for any `j`.** The overlay `F_2 u ... u F_J` on the sight is the struck set of
`{5..q}`, which in the column is the union of the residues `+-k_g (mod g)` (skeleton 2), periodic with
period `prod g`. PROVED. So the fields are an aperiodic partition of a periodic set: the periodicity lives
in the union and in nothing finer.

**E5 (the class rule per field, T4).** `chi(n) = (-1)^{Omega_-(n)}` (fold_mechanic.md F7), so a member of
`F_j` is a right member iff its count of class `-1` factors is even, a left member iff odd. For `j = 2`:
right members are `P_- P_-` and `P_+ P_+`, left members `P_- P_+`. **The square field is right-only**
(`Omega_- (p^2) = 0` or `2`). PROVED. Gate: 0 class-rule mismatches and 0 squares on the left at 20
sections and 13 sights (table 2.1).

**E6 (the mirror law of a dilate, T5).** With `g = 6 k_g + eps_g`: `g(6m - 1) = 6(gm - k_g) - eps_g` and
`g(6m + 1) = 6(gm + k_g) + eps_g`, so the members of column `m` dilated by `g` sit in columns `gm -+ k_g`,
mirror images about column `gm`, on the sides `-eps_g` and `eps_g`. The mirror `n -> 12gm - n` maps
`g(6m + i)` to `g(6m - i)` for every `i`, so **the symmetric pairs of `F_j` through `g` about `6gm` are the
dilates of the pairs `{6m - i, 6m + i}` with both members in `F_{j-1}`** (`i = 1, 5, 7, 11, 13, ...`, the
radii at which `6m +- i in S`), and the innermost radius `i = 1` is the both-`F_{j-1}` column `m`: for
`j = 2` a twin column. PROVED. Gate: at every gear `g` of the engine and every axis `m` with both
`gm -+ k_g` in the section, the member at `(gm - k_g, side -eps_g)` is `g(6m - 1)` and at `(gm + k_g, side
eps_g)` is `g(6m + 1)`, each with `Omega` one more than its cofactor: 0 mismatches over 171,243 axes in 18
sections (table 2.1, "T5"). Prior art, one line: the pairs `{6m - i, 6m + i}` of primes are the
representations `12m = r + r'` (Goldbach; Hardy-Littlewood's count); noted, not derived, and the stop line
of section 6.

**E7 (the blind sets of the deep fields, T6).** A member `n` of `F_j` below `p'^2` satisfies
`lpf(n)^j <= n < p'^2`, so `lpf(n) < p'^{2/j}`, and the column of `n` is struck by the gear `lpf(n)`.
Hence **`F_j` is blind to every column open under `{5..p'^{2/j}}`**, a periodic set with two residues per
gear. For `j = 3` the machine is the cube-core `{g < p'^{2/3}}`; on its open columns every composite
member is in `F_2` with least factor `g >= p'^{2/3}` and cofactor `r` prime with `g <= r < p'^2/g`. So on
the cube-core-open columns **a column is a twin iff it is blind to `F_2`**, and **`F_2` there consists of
products of two primes from the band `[p'^{2/3}, p'^{4/3})`**. PROVED (the two-prime lemma, kernel
`CoreLeftover.primeOrSemiprime_of_rough_lt_cube`, in the fields' coordinate; S17 gives the sharp thresholds
`nextprime(t)^j`; cited). Gate: 0 exceptions to the confinement at every `j >= 3` in 20 sections, and 0
non-`F_2` strikes on cube-core-open columns (section 5). **The square field is empty in every finer
section** `(p^2, p'^2)` (no prime strictly between `p` and `p'`), and in the construction section
`[p_k^2, p_{k+1}^2)` it is `{r^2 : r prime, p_k < r < p_{k+1}}`: the squares of the gears of machine `k`
above `p_k`, i.e. **the cuts of the construction are the square field** (`c_{k+1} = p_k^2`) and the finer
sections of 8e are its gaps. PROVED (definitions); gate in section 5.

**E8 (the twin as "open here, struck there").** Combine E2 and E7. A column `c` of the section is struck
through the tail gear `g` (`g >= p'^{2/3}`) on side `s` iff `c = +-k_g (mod g)` and the cofactor
`u = (6c + s)/g` is prime, i.e. `u` is open under the engine `{5..sqrt(p'^2/g)}` on the stretch
`(p^2/g, p'^2/g)`. So: **`c` is a twin iff `c` is cube-core-open and every tail cofactor of `c` is composite
(struck at its own scale by a lower engine).** A strike in the section is an opening at scale `1/g` under a
lower engine, and an opening in the section is a column whose every tail cofactor is a strike at its scale.
PROVED (E2 + E7). Gate: twins computed as "cube-core-open and unpinned" equal the twins directly, 0
mismatches at 20 sections (section 5, "twins from pins"). Prior art, one line: the COUNT of this
statement is Buchstab's identity iterated once (the count of `t`-rough pairs less the tail's semiprime
pins); noted, not derived.

---

## 2. Results: each field's own structure (`fd_fields.py`)

### 2.1 The fields in the sections

Per field `j`: members (left / right), columns hit `|H_j|`, both-`F_j` columns (both members in `F_j`).
"One-field" = share of struck columns whose composite members all lie in one field, and the share of those
that are `F_2`. Gates: T2 mismatches, class-rule mismatches, squares on the left, T5 mismatches / axes
checked, twins-from-fields mismatches. `J` is the deepest field present.

| section | columns | `q` | twins | struck | both struck | `J` | `F_2`: members (L/R), cols, both | `F_3` | `F_4` | `F_5 ..` | one-field / `F_2` share | gates T2, class, sqL, T5, tw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| finer 11 | 7 | 11 | 2 | 5 | 1 | 3 | 5 (3/2), 4, 1 | 1 (1/0), 1, 0 | | | 1.000 / 0.800 | 0, 0, 0, 0/1, 0 |
| finer 13 | 19 | 13 | 7 | 12 | 4 | 3 | 13 (5/8), 10, 3 | 3 (2/1), 3, 0 | | | 0.917 / 0.818 | 0, 0, 0, 0/9, 0 |
| finer 17 | 11 | 17 | 2 | 9 | 3 | 3 | 10 (6/4), 9, 1 | 2 (0/2), 2, 0 | | | 0.778 / 1.000 | 0, 0, 0, 0/5, 0 |
| finer 19 | 27 | 19 | 4 | 23 | 4 | 3 | 23 (11/12), 21, 2 | 4 (2/2), 4, 0 | | | 0.913 / 0.905 | 0, 0, 0, 0/15, 0 |
| finer 23 | 51 | 23 | 8 | 43 | 13 | 4 | 44 (23/21), 37, 7 | 11 (6/5), 11, 0 | 1 (0/1), 1, 0 | | 0.860 / 0.838 | 0, 0, 0, 0/32, 0 |
| finer 29 | 19 | 29 | 2 | 17 | 5 | 4 | 16 (7/9), 13, 3 | 5 (2/3), 4, 1 | 1 (1/0), 1, 0 | | 0.941 / 0.750 | 0, 0, 0, 0/11, 0 |
| finer 31 | 67 | 31 | 11 | 56 | 22 | 4 | 61 (29/32), 46, 15 | 16 (10/6), 16, 0 | 1 (0/1), 1, 0 | | 0.875 / 0.796 | 0, 0, 0, 0/46, 0 |
| finer 37 | 51 | 37 | 7 | 44 | 14 | 4 | 41 (21/20), 33, 8 | 15 (7/8), 15, 0 | 2 (1/1), 2, 0 | | 0.864 / 0.711 | 0, 0, 0, 0/35, 0 |
| finer 41 | 27 | 41 | 3 | 24 | 11 | 4 | 27 (17/10), 21, 6 | 7 (2/5), 7, 0 | 1 (1/0), 1, 0 | | 0.792 / 0.842 | 0, 0, 0, 0/18, 0 |
| finer 43 | 59 | 43 | 11 | 48 | 25 | 4 | 54 (27/27), 42, 12 | 17 (8/9), 17, 0 | 2 (1/1), 2, 0 | | 0.729 / 0.829 | 0, 0, 0, 0/44, 0 |
| finer 47 | 99 | 47 | 13 | 86 | 32 | 4 | 82 (38/44), 67, 15 | 32 (19/13), 31, 1 | 4 (1/3), 4, 0 | | 0.814 / 0.729 | 0, 0, 0, 0/77, 0 |
| finer 53 | 111 | 53 | 13 | 98 | 46 | 5 | 108 (54/54), 85, 23 | 30 (14/16), 30, 0 | 5 (2/3), 5, 0 | `F_5`: 1 (1/0) | 0.765 / 0.840 | 0, 0, 0, 0/88, 0 |
| base 3 link 1 `[9, 121)` | 18 | 7 | 8 | 10 | 0 | 2 | 10 (4/6), 10, 0 | | | | 1.000 / 1.000 | 0, 0, 0, 0/5, 0 |
| base 3 link 2 `[121, 16129)` | 2,667 | 113 | 276 | 2,391 | 1,097 | 6 | 2,391 (1,180/1,211), 1,866, 525 | 919 (469/450), 858, 61 | 161 (77/84), 161, 0 | `F_5`: 16 (9/7); `F_6`: 1 (0/1) | 0.786 / 0.731 | 0, 0, 0, 0/2,699, 0 |
| base 5 link 1 `[25, 841)` | 135 | 23 | 29 | 106 | 28 | 4 | 112 (55/57), 95, 17 | 21 (11/10), 21, 0 | 1 (0/1), 1, 0 | | 0.896 / 0.884 | 0, 0, 0, 0/87, 0 |
| base 5 link 2 `[841, 727609)` | 121,127 | 839 | 6,224 | 114,903 | 68,889 | 8 | 102,376 (51,106/51,270), 81,017, 21,359 | 60,187 (30,167/30,020), 53,254, 6,933 | 17,560 (8,731/8,829), 17,225, 335 | `F_5`: 3,201 (1,619/1,582), both 5; `F_6`: 426 (207/219); `F_7`: 40 (21/19); `F_8`: 2 (1/1) | 0.650 / 0.605 | 0, 0, 0, 0/162,361, 0 |
| base 7 link 1 `[49, 2809)` | 459 | 47 | 74 | 385 | 139 | 4 | 399 (195/204), 321, 78 | 113 (59/54), 111, 2 | 12 (5/7), 12, 0 | | 0.847 / 0.804 | 0, 0, 0, 0/376, 0 |
| base 7 link 2 `[2809, 7946761)` | 1,323,991 | 2,803 | 48,249 | 1,275,742 | 836,197 | 9 | 1,052,196 (525,909/526,287), 847,116, 205,080 | 731,044 (365,669/365,375), 632,856, 98,188 | 259,467 (129,643/129,824), 250,528, 8,939 | `F_5`: 58,219 (29,148/29,071), both 219; `F_6`: 9,605 (4,782/4,823), both 2; `F_7`: 1,263 (638/625); `F_8`: 134 (65/69); `F_9`: 11 (6/5) | 0.589 / 0.541 | 0, 0, 0, (not run), 0 |
| base 13 link 1 `[169, 29929)` | 4,959 | 167 | 455 | 4,504 | 2,213 | 6 | 4,449 (2,207/2,242), 3,450, 999 | 1,858 (939/919), 1,716, 142 | 365 (177/188), 362, 3 | `F_5`: 43 (22/21); `F_6`: 2 (1/1) | 0.763 / 0.705 | 0, 0, 0, 0/5,334, 0 |
| base 3 link 3 `[16129, 260467321)` | 43,408,531 | 16,127 | 1,027,948 | 42,380,583 | 30,218,414 | 12 | 31,420,800 (15,709,963/15,710,837), 25,902,245, 5,518,555 | 25,945,536 (12,973,072/12,972,464), 22,091,538, 3,853,998 | 11,313,381 (5,656,486/5,656,895), 10,705,861, 607,520 | `F_5`: 3,151,329 (1,575,735/1,575,594), both 28,043; `F_6`: 644,079 (321,989/322,090), both 519; `F_7`: 106,586, both 3; `F_8`: 15,184; `F_9`: 1,885; `F_10`: 199; `F_11`: 17; `F_12`: 1 | 0.523 / 0.467 | 0, 0, 0, (not run), 0 |

(Gate P1 (b), by the same census: the 301-rough members of base 3 link 3 by `Omega = 1, 2, 3, 4` are
14,218,065 / 10,991,941 / 192,920 / 0, exactly leftover_depth.md's numbers; the twin counts of all nine
sections on record reproduced; the depth histograms of origin_mechanic.md table A at `p = 23, 53`
reproduced once "depth" was read as the `Omega` of the member carrying the least striker, the first run of
the gate having read it as the minimum `Omega` and failed: `{2: 37, 3: 6}` against `{2: 33, 3: 9, 4: 1}`.)

### 2.2 The fields on the machine sights `[1, q'^2)` (columns `1 .. b - 1`)

| `q` | `q'` | columns | twins | struck | `J` | `F_2`: members (L/R), cols, both | `F_3` | `F_4`, `F_5` | one-field / `F_2` share |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 11 | 19 | 9 | 10 | 2 | 10 (4/6), 10, 0 | | | 1.000 / 1.000 |
| 11 | 13 | 27 | 11 | 16 | 3 | 17 (8/9), 15, 2 | 1 (1/0), 1, 0 | | 1.000 / 0.938 |
| 13 | 17 | 47 | 18 | 29 | 3 | 31 (13/18), 26, 5 | 4 (3/1), 4, 0 | | 0.966 / 0.893 |
| 17 | 19 | 59 | 20 | 39 | 3 | 43 (20/23), 36, 7 | 6 (3/3), 6, 0 | | 0.923 / 0.917 |
| 19 | 23 | 87 | 24 | 63 | 3 | 67 (31/36), 58, 9 | 10 (5/5), 10, 0 | | 0.921 / 0.914 |
| 23 | 29 | 139 | 32 | 107 | 4 | 113 (55/58), 96, 17 | 21 (11/10), 21, 0 | 1 (0/1) | 0.897 / 0.885 |
| 29 | 31 | 159 | 34 | 125 | 4 | 130 (62/68), 110, 20 | 26 (13/13), 25, 1 | 2 (1/1) | 0.904 / 0.867 |
| 31 | 37 | 227 | 45 | 182 | 4 | 193 (92/101), 157, 36 | 42 (23/19), 41, 1 | 3 (1/2) | 0.896 / 0.847 |
| 37 | 41 | 279 | 52 | 227 | 4 | 235 (113/122), 191, 44 | 57 (30/27), 56, 1 | 5 (2/3) | 0.890 / 0.822 |
| 41 | 43 | 307 | 55 | 252 | 4 | 264 (131/133), 213, 51 | 64 (32/32), 63, 1 | 6 (3/3) | 0.881 / 0.824 |
| 43 | 47 | 367 | 66 | 301 | 4 | 319 (158/161), 256, 63 | 81 (40/41), 80, 1 | 8 (4/4) | 0.857 / 0.826 |
| 47 | 53 | 467 | 79 | 388 | 4 | 402 (196/206), 324, 78 | 113 (59/54), 111, 2 | 12 (5/7) | 0.848 / 0.805 |
| 53 | 59 | 579 | 92 | 487 | 5 | 512 (251/261), 410, 102 | 143 (73/70), 141, 2 | 17 (7/10); `F_5`: 1 (1/0) | 0.832 / 0.812 |

### 2.3 Readings of tables 2.1 and 2.2 (mechanism, then the count)

1. **Period.** None, for any field, at any size (E4, a proof). The tables show it in the simplest way: the
   hit set of `F_2` in `[9, 121)` is the full two-tooth progression of `{5, 7}` on the 18 columns (10 hit
   columns = the CRT count `18 - 8`), because every cofactor there is below `25` and so prime; from the
   first section holding a composite cofactor (`125 = 5 x 25` at `q = 11`) the progression of gear 5 is
   split between `F_2` (`5 x prime`) and `F_3` (`5 x semiprime`), and the split follows the primes among
   the cofactors, which have no period.
2. **Density per side.** Every field sits on both sides at near-equal weight (E5 says which factor
   patterns go where), and the excess alternates in sign with `j`: in the three largest sections `F_2` has
   more right members (15,710,837 against 15,709,963; 526,287 against 525,909; 51,270 against 51,106),
   `F_3` more left (12,972,464 right against 12,973,072 left; 365,375 against 365,669; 30,020 against
   30,167), `F_4` more right again (5,656,895 against 5,656,486; 129,824 against 129,643; 8,829 against
   8,731): 9 of 9 signs as predicted (P4). Mechanism (E5): the right side of `F_2` holds the same-class
   pairs including the squares, and the primes themselves lean to class `-1` (`F_1` left / right:
   7,109,338 / 7,108,727 at base 3 link 3; 268,131 / 267,912 at base 7 link 2; 29,275 / 29,187 at base 5
   link 2; 932 / 914 at base 3 link 2); the sign then alternates with the count of class `-1` factors. Prior art, one line: the alternating Chebyshev bias for products of `k` primes (X.
   Meng, Algebra & Number Theory 12 (2018), under GRH-type hypotheses); the machine reproduces its sign
   at every `j <= 4`; stopped. The bias is of relative size `3 x 10^{-5}` to `3 x 10^{-3}` here and is not a
   covering constraint.
3. **The share each field carries.** By members, `F_2` carries 1.000, 0.61, 0.46 of the composite members
   along the chain from base 3 (links 1, 2, 3) and 0.56, 0.45 along base 7; `F_3` overtakes `F_2` by
   columns hit nowhere in these sections but reaches 0.85 of `F_2`'s column count at base 3 link 3
   (22,091,538 against 25,902,245). The per-column depth census by least striker is origin_mechanic.md
   tables A, B (cited, not redone).
4. **Both-`F_j` columns.** Twin semiprimes (both members in `F_2`) are 525 against 276 twin primes at base
   3 link 2 and 5,518,555 against 1,027,948 at link 3 (5.4 times); both-`F_3` 3,853,998; both-`F_4`
   607,520; both-`F_5` 28,043; both-`F_6` 519; both-`F_7` 3. These are the innermost-radius seeds of the
   mirror pairs of the field above (E6). No both-`F_j` column exists for `j >= 8` in any section run.
5. **Columns hit by exactly one field.** The share falls with the section's top: 1.000, 0.786, 0.523 along
   base 3; 0.896, 0.650 along base 5; 0.847, 0.589 along base 7; and among them `F_2` is the field in
   0.73, 0.47 (base 3 links 2, 3). Mechanism: more fields are present the higher the top (`J = 2, 6, 12`),
   and a both-struck column draws its two members from different fields at the rate the fields mix. The
   finer sections at `p = 23..53` sit at 0.73 to 0.94 (P5's origin clause is refuted at one of eight,
   `p = 43`: 0.729), with `F_2` the single field in 0.71 to 0.84 of them.

---

## 3. Relations between the fields

**The dilation map.** `F_{j-1} -> F_j`, `m -> p m`, one map per prime `p`, each injective; their images
cover `F_j` (E2), and the images of the map `m -> 5m` alone are exactly `F_j n 5S` (E3). So the family of
fields is one object under dilation: `F_j` is `F_1` dilated by all products of `j - 1` primes, and `F_1` is
recovered from `F_j` by dividing out `5^{j-1}`. In columns the map is origin_mechanic.md D2:
`(i, s) -> (p i + s k_p, eps_p s)`, and in the section it lands `F_{j-1}`'s members from the stretch
`(p^2/p_0, p'^2/p_0)` (for `p_0 = p`) into the section, so the field `F_j` of a section is assembled from
the fields `F_{j-1}` of the lower stretches `(lo/p, hi/p)`, one per prime `p <= q`, and the whole
recursion bottoms out at the primes of those stretches: the gears of the lower machines (the hand-up,
skeleton 6).

**How much of the struck set each field carries.** Table 2.1 by members and by columns; the per-column
census by the least striker's member is origin_mechanic.md tables A, B (cited). By columns hit, `F_2`
carries 0.61 of the struck columns at base 3 link 3 (25,902,245 of 42,380,583), `F_3` 0.52, `F_4` 0.25,
`F_5` 0.074, `F_6` 0.015, `F_7` 0.0025 (the shares overlap, since 30,218,414 columns are both-struck).

**Columns hit by exactly one field.** Section 4.

**Which field a strike belongs to is decided at the cofactor's scale.** By E2, the strike of gear `g` on
column `c` (side `s`) is in `F_j` iff its cofactor `(6c + s)/g` is in `F_{j-1}`. So the field labels on the
teeth of `g` in the section are the field labels of the stretch `(lo/g, hi/g)` at scale `1/g`, read through
the affine map: the fields of a section are the fields of lower stretches, dilated. There is no other
relation: the partition of `g`'s teeth into fields is the partition of the cofactor stretch into fields,
which is (by E3, iterated) the primes of that stretch.

---

## 4. Columns hit by exactly one field: the origin against height (`fd_height.py`)

At height (the four record runs of origin_mechanic.md table E, every column struck) against the finer
sections (table 2.1). Controls: 20 random stretches of the run's length at columns uniform in `[x/2, 2x]`.

| run | columns | one-field share | `F_2` share of one-field | signatures (Omega of the composite members) | `F_2` members L/R | controls: one-field share mean (min-max) | controls `F_2` share |
|---|---|---|---|---|---|---|---|
| m23 at 12,694,429 | 33 | 0.667 | 0.364 | `2`: 5, `2+2`: 3, `2+3`: 6, `2+4`: 2, `2+5`: 1, `3`: 4, `3+3`: 5, `3+4`: 2, `4`: 4, `5`: 1 | 7/13 | 0.522 (0.281-0.697) | 0.418 |
| m29 at 200,906,186 | 42 | 0.381 | 0.250 | `2`: 1, `2+2`: 3, `2+3`: 16, `2+4`: 3, `2+5`: 3, `3`: 5, `3+3`: 3, `3+4`: 4, `4`: 3, `4+4`: 1 | 9/20 | 0.482 (0.366-0.625) | 0.411 |
| m31 at 1,468,940,243 | 57 | 0.404 | 0.217 | `2`: 2, `2+2`: 3, `2+3`: 20, `2+4`: 5, `2+5`: 1, `2+6`: 2, `3`: 8, `3+3`: 5, `3+4`: 4, `3+5`: 2, `4`: 3, `4+4`: 1, `6`: 1 | 14/22 | 0.480 (0.382-0.554) | 0.374 |
| m37 at 90,816,580,903 | 87 | 0.425 | 0.270 | `2`: 1, `2+2`: 9, `2+3`: 22, `2+4`: 8, `2+5`: 1, `2+6`: 1, `2+7`: 1, `3`: 3, `3+3`: 9, `3+4`: 13, `3+5`: 2, `3+6`: 1, `4`: 8, `4+4`: 5, `5`: 2, `5+6`: 1 | 26/26 | 0.418 (0.353-0.494) | 0.328 |
| finer `p = 23 .. 53` (origin) | 19-111 | 0.73-0.94 | 0.71-0.84 | table 2.1 | | | |

Readings. (i) At the origin a struck column is typically a single `F_2` strike (signature `2`: 24 of 43
struck columns at `p = 23`, 40 of 98 at `p = 53`, with `2+2` next at 7 and 23); at height it is
typically two strikes from two fields (`2+3` is the mode at m29, m31, m37: 16 of 42, 20 of 57, 22 of 87)
and the single-`F_2` signature almost disappears (1 of 42, 2 of 57, 1 of 87). Along the chain from base 3
the mode moves from `2` (10 of 10; 849 of 2,391) to `2+3` (9,545,411 of 42,380,583 at link 3, against
4,841,895 for `2` and 5,518,555 for `2+2`). Mechanism: at the origin every column with one composite member is a prime beside a
composite, and the composite is `g x prime` for every `g > n^{1/3}` (E7); at height both members are
composite in most columns (open columns are rare) and the fields mix at their line densities. (ii) The
record runs are ordinary among random stretches of their length on this census: m23's 0.667 sits inside
its controls' range 0.281-0.697, m29 at 0.381 against 0.366-0.625, m31 0.404 against 0.382-0.554, m37
0.425 against 0.353-0.494. (iii) P5 as pre-registered: origin clause held at 7 of 8 (refuted at `p = 43`,
0.729), height clause held at 3 of 4 (refuted at m23, 0.667 and `F_2` share 0.364); the direction held at
every pair (every origin section above every run). A count; it is recorded, not pursued.

---

## 5. The section against each field: the blind sets (`fd_blind.py`)

**Confinement of the deep fields (E7, P3).** For every section run and every `j >= 3`, every member of `F_j`
lies in a column struck by a gear below `p'^{2/j}`: 0 exceptions in 20 sections (the `F_3` members number 1,
3, 2, 4, 11, 5, 16, 15, 7, 17, 32, 30 in the finer sections `p = 11..53`, and 919; 21; 60,187; 113;
731,044; 1,858 in the construction sections; `F_4` to `F_9` likewise, 0 exceptions each). The machines the
deep fields are confined to, at the finer section at `p = 53` (`p' = 59`): `F_3` to the teeth of
`{5, 7, 11, 13}` (bound 15.16), `F_4` to `{5, 7}` (7.68), `F_5` to `{5}` (5.11: the one member is
`3125 = 5^5`). At base 7 link 2 (`p' = 2819`): `F_3` to the 44 gears below 199.56, `F_4` to the 14 below
53.09, `F_5` to the 7 below 23.99, `F_6` to `{5, 7, 11, 13}`, `F_7`, `F_8` to `{5, 7}`, `F_9` to `{5}`.

**The square field (E7).** Empty in every finer section (12 of 12); in the construction sections it is the
squares of the primes strictly between `p_k` and `p_{k+1}`: `[25, 49]` in `[9, 121)`; `[49, 121, 169, 289,
361, 529]` in `[25, 841)`; 25 squares (169, 289, 361, ...) in `[121, 16129)`; 11 in `[49, 2809)`; 136 in
`[841, 727609)`; 393 in `[2809, 7946761)`; 33 in `[169, 29929)`. Every one a right member (E5). So the
square field's location rule is exact and known: its members are the cuts of the construction and the
squares of the lower machine's gears; it is blind to every left member and to every column but
`(r^2 - 1)/6`.

**The cube-core against `F_2` (E7, E8, P7).** Cube-core = the gears below `p'^{2/3}`; "core-open" = columns
of the section no cube-core gear strikes; "pins" = the `F_2` strikes on core-open columns, each `g x r`
with `g >= p'^{2/3}` the least factor and `r` prime; "exceptions" = strikes on core-open columns that are
not of that form (E7 says none).

| section | `p'` | cube-core (bound) | core-open | twins | open / twins | pins | pinned cols | exceptions | twins from pins (= core-open, unpinned) |
|---|---|---|---|---|---|---|---|---|---|
| finer 11 | 13 | {5} (5.53) | 4 | 2 | 2.00 | 2 | 2 | 0 | 2 |
| finer 13 | 17 | {5} (6.61) | 11 | 7 | 1.57 | 4 | 4 | 0 | 7 |
| finer 17 | 19 | {5, 7} (7.12) | 3 | 2 | 1.50 | 1 | 1 | 0 | 2 |
| finer 19 | 23 | {5, 7} (8.09) | 12 | 4 | 3.00 | 8 | 8 | 0 | 4 |
| finer 23 | 29 | {5, 7} (9.44) | 21 | 8 | 2.62 | 14 | 13 | 0 | 8 |
| finer 29 | 31 | {5, 7} (9.87) | 8 | 2 | 4.00 | 8 | 6 | 0 | 2 |
| finer 31 | 37 | {5, 7, 11} (11.10) | 23 | 11 | 2.09 | 15 | 12 | 0 | 11 |
| finer 37 | 41 | {5, 7, 11} (11.89) | 15 | 7 | 2.14 | 9 | 8 | 0 | 7 |
| finer 41 | 43 | {5, 7, 11} (12.27) | 10 | 3 | 3.33 | 8 | 7 | 0 | 3 |
| finer 43 | 47 | {5, 7, 11, 13} (13.02) | 17 | 11 | 1.55 | 7 | 6 | 0 | 11 |
| finer 47 | 53 | {5, 7, 11, 13} (14.11) | 31 | 13 | 2.38 | 20 | 18 | 0 | 13 |
| finer 53 | 59 | {5, 7, 11, 13} (15.16) | 32 | 13 | 2.46 | 23 | 19 | 0 | 13 |
| base 3 link 1 | 11 | {} (4.95) | 18 | 8 | 2.25 | 10 | 10 | 0 | 8 |
| base 3 link 2 | 127 | 7 gears to 23 (25.27) | 571 | 276 | 2.07 | 342 | 295 | 0 | 276 |
| base 5 link 1 | 29 | {5, 7} (9.44) | 57 | 29 | 1.97 | 30 | 28 | 0 | 29 |
| base 5 link 2 | 853 | 22 gears to 89 (89.94) | 14,300 | 6,224 | 2.30 | 9,810 | 8,076 | 0 | 6,224 |
| base 7 link 1 | 53 | {5, 7, 11, 13} (14.11) | 137 | 74 | 1.85 | 71 | 63 | 0 | 74 |
| base 7 link 2 | 2,819 | 44 gears to 199 (199.56) | 114,586 | 48,249 | 2.37 | 81,124 | 66,337 | 0 | 48,249 |
| base 13 link 1 | 173 | 9 gears to 31 (31.05) | 923 | 455 | 2.03 | 548 | 468 | 0 | 455 |

E8 holds as a gate at 20 of 20 (twins = core-open and unpinned); the exceptions column is 0 throughout
(every strike on a core-open column is `g x r` with `g >= p'^{2/3}`, `r` prime). P7's ratio clause
`[1.3, 3]` is refuted at 2 of 8 finer sections (`p = 29`: 4.00; `p = 41`: 3.33). The pins per tail gear
fall with the gear as the cofactor range `(p^2/g, p'^2/g)` shrinks: at base 7 link 2 from 1,093 pins for
`g = 211` to 0 for `g = 2797, 2801` and 1 for `g = 2803`; at base 3 link 2 from 39 for `g = 29, 31` to 1
for `g = 113`.

**The pins written out at the finer section `p = 29`** (columns 141..159, `p' = 31`, cube-core `{5, 7}`,
8 core-open columns, twins 143 and 147): `(142, L, 23 x 37)`, `(145, L, 11 x 79)`, `(145, R, 13 x 67)`,
`(150, L, 29 x 31)`, `(150, R, 17 x 53)`, `(152, R, 11 x 83)`, `(157, R, 23 x 41)`, `(158, R, 13 x 73)`:
8 pins on 6 columns, two columns pinned twice (145 and 150), and the two unpinned core-open columns are the
twins. This is origin_mechanic.md section 3's instance in the fields' coordinate: the pins are the
`F_2` members of the section whose least factor is a tail gear, and their positions are the primes
`37, 79, 67, 31, 53, 83, 41, 73` of the cofactor stretches, dilated. At `p = 17` (columns 49..59, cube-core
`{5, 7}`, 3 core-open, twins 52, 58) the single pin is `(53, R, 11 x 29)`.

**What the blind sets say, exactly.** `B_j` (`j >= 3`) contains the open set of `{5..p'^{2/j}}`, periodic
and closed-form; `B_2` contains no periodic set beyond the twins themselves (every residue class mod every
gear holds `F_2` members: the teeth of `g` hold `g x prime`). So the intersection `B_2 n B_3 n ... n B_J`
is `B_2 n (cube-core-open)` in effect, i.e. "core-open and unpinned" (E8), and the closed-form part of
the intersection is the cube-core's open set, a CRT set of density `prod_{g < p'^{2/3}} (1 - 2/g)` on the
section; what the pins remove from it is decided by the primes in the cofactor stretches. **No
intersection of blind sets forces a twin without a count**: the core-open set is a count (its density
times the section's length, the extreme value of that count over positions being the object of
core_leftover.md S13), and the pins are `F_2 = 5^{-1}`-equivalent to the primes (E3). What the fields'
coordinate adds is the exact form E8, which is the count's mechanism, not a replacement for it.

---

## 6. The 6-cycle and the mirrors (`fd_mirror.py`, `fd_mirror2.py`)

**How each field sits on the two classes.** E5 and table 2.1: every field lies on both sides at near-equal
weight; the side of a member is the parity of its class `-1` factor count; the square field is right-only;
the excess alternates in sign with `j` (2.3 (2); prior art Meng 2018 at the stop). The classes decide which
member of a column a field strikes, and by fold_mechanic.md F3 the column cover does not see them.

**The bifurcation of the fold under dilation** is E6's first sentence: the column `m` (its two members at
distance 2) dilated by `g` becomes two columns `gm -+ k_g`, mirror images about column `gm`, the members
now at distance `2g` and on the sides `-eps_g`, `+eps_g` (swapped when `g = -1 (mod 6)`). So every twin
column of the line reappears, once per prime `g`, as a mirror pair of `F_2` hits at distance `2k_g` about
`6gm`, and every both-`F_{j-1}` column as a mirror pair of `F_j`. The mirror pairs are struck columns:
the twins below feed the strikes above (the hand-up), never openings.

**The identity (P6(a)).** The symmetric `F_2` pairs through `g` about `6gm`, counted on the section's members
divisible by `g`, equal the prime pairs `{6m - i, 6m + i}` in the cofactor range, at every gear `5, 7, 11, 13`
and every one of 15 sections (table below, "through = direct"): 0 mismatches.

**The growth (P6(b)).** Through `g = 5`, per axis `30m` with both cofactors in range, all radii:

| range | axes | `F_2` symmetric pairs | mean per axis | max per axis | axes with a radius-1 pair (twin `m`) | radius-1 share | `F_3` pairs (from both-`F_2` pairs) | mean | radius-1 share |
|---|---|---|---|---|---|---|---|---|---|
| base 3 link 1 `[9, 121)` | 3 | 6 | 2.00 | 3 | 3 | 0.500 | 0 | 0 | - |
| base 3 link 2 `[121, 16129)` | 533 | 24,963 | 46.83 | 122 | 80 | 0.003 | 27,944 | 52.43 | 0.003 |
| base 3 link 3 `[16129, 260467321)`, 40 sampled axes, all radii | 40 | 4,437,768 | 110,944 | 404,553 | 3 | `7 x 10^{-7}` | not run | | |
| sight `q = 7` | 3 | 6 | 2.00 | 3 | 3 | 0.500 | 0 | 0 | - |
| sight `q = 11` | 5 | 11 | 2.20 | 4 | 4 | 0.364 | 0 | 0 | - |
| sight `q = 13` | 9 | 26 | 2.89 | 6 | 5 | 0.192 | 2 | 0.22 | 0 |
| sight `q = 17` | 11 | 40 | 3.64 | 6 | 6 | 0.150 | 3 | 0.27 | 0 |
| sight `q = 19` | 17 | 79 | 4.65 | 9 | 8 | 0.101 | 10 | 0.59 | 0 |
| sight `q = 23` | 27 | 173 | 6.41 | 13 | 11 | 0.064 | 39 | 1.44 | 0.051 |
| sight `q = 29` | 31 | 209 | 6.74 | 14 | 12 | 0.057 | 51 | 1.65 | 0.059 |
| sight `q = 31` | 45 | 390 | 8.67 | 18 | 17 | 0.044 | 119 | 2.64 | 0.042 |
| sight `q = 37` | 55 | 532 | 9.67 | 21 | 19 | 0.036 | 215 | 3.91 | 0.033 |
| sight `q = 41` | 61 | 631 | 10.34 | 23 | 20 | 0.032 | 254 | 4.16 | 0.028 |
| sight `q = 43` | 73 | 861 | 11.79 | 30 | 22 | 0.026 | 378 | 5.18 | 0.021 |
| sight `q = 47` | 93 | 1,248 | 13.42 | 30 | 24 | 0.019 | 698 | 7.51 | 0.017 |
| sight `q = 53` | 115 | 1,888 | 16.42 | 41 | 29 | 0.015 | 1,034 | 8.99 | 0.013 |

The mean grows monotonically over the 13 sights (2.00 to 16.42) and along the chain (2.00, 46.83,
110,944), and the twin's share of the symmetry falls (0.500 to 0.015; `7 x 10^{-7}` at link 3). With the
radius capped at 601 the per-axis mean FALLS along the chain (2.00, 28.25, 6.52 at links 1, 2, 3, the
last on 89,503 sampled axes): the growth is the growth of the reach (the number of radii, about `m`),
not of the pairing's density per radius, which falls with the cofactors' size (the prime density). The
radial profile is flat: at link 3 (89,503 sampled axes) the pairs at radii `1, 5, 7, 11, 13, ...` number
2,594; 3,534; 3,170; 2,865; 2,795; ...; 2,599 at 61, with the bumps at `5, 25, 35, 55` (3,534; 3,428;
4,103; 3,857) and `7, 49` (3,170; 3,135) being the radii `i` divisible by 5 or 7, where `6m +- i` avoid
the classes of 5 or 7 at once. The radius-1 term (the twin) is one term of a flat profile: nothing
distinguishes it from radius 11 or 61 but its position. Prior art, one line: the count of pairs about
`6m` is the Goldbach count of `12m` (Hardy-Littlewood's conjecture gives its growth and the bumps at
`5 | i`, `7 | i` are its singular series); stopped.

**Off the multiples of `g` (P6(c)), and the mechanism found on the second measurement.** The symmetric-hit
rate of `F_2` about `6gm` for members not divisible by `g`, against the null "the partner is in `F_2` at
the section's `F_2` density":

| section | `g` | off-axis hits | symmetric | global null (`z`) | local-density null (`z`) | hits coprime to `m`: symmetric / null (`z`) | hits sharing a prime with `m`: symmetric / null (`z`) |
|---|---|---|---|---|---|---|---|
| finer `p = 11 .. 53`, `g = 5, 7, 11, 13` | | 2 to 939 per cell | | all 44 cells with hits within 3 sd (`z` from -2.36 to +2.53) | | | |
| base 3 link 2 | 5 | 518,445 | 236,128 | 232,396 (+10.42) | 232,401 (+10.43) | 214,610 / 216,685 (-6.01) | 21,518 / 15,715 (+62.45) |
| base 3 link 2 | 7 | 391,639 | 175,966 | 175,555 (+1.32) | 175,561 (+1.30) | 158,252 / 160,863 (-8.78) | 17,714 / 14,698 (+33.55) |
| base 3 link 2 | 11 | 262,328 | 116,748 | 117,590 (-3.31) | 117,591 (-3.32) | 104,960 / 106,851 (-7.80) | 11,788 / 10,740 (+13.64) |
| base 3 link 2 | 13 | 224,867 | 99,630 | 100,798 (-4.95) | 100,791 (-4.93) | 89,598 / 91,488 (-8.43) | 10,032 / 9,302 (+10.21) |
| base 5 link 1 | 5 | 1,027 | 362 | 426 (-4.05) | 423 (-4.09) | 264 / 374 (-7.75) | 98 / 50 (+9.80) |
| base 5 link 1 | 7 | 805 | 288 | 334 (-3.29) | 335 (-3.55) | 214 / 291 (-6.19) | 74 / 44 (+6.36) |
| base 5 link 1 | 11 | 573 | 206 | 238 (-2.69) | 240 (-2.99) | 164 / 211 (-4.41) | 42 / 29 (+3.39) |
| base 5 link 1 | 13 | 496 | 176 | 206 (-2.71) | 204 (-2.62) | 142 / 181 (-3.93) | 34 / 23 (+3.25) |
| base 7 link 1 | 5 | 13,807 | 5,648 | 6,001 (-6.06) | 6,000 (-6.14) | 4,702 / 5,449 (-13.67) | 946 / 551 (+22.94) |
| base 7 link 1 | 7 | 10,672 | 4,312 | 4,639 (-6.38) | 4,631 (-6.33) | 3,556 / 4,121 (-11.89) | 756 / 510 (+14.80) |
| base 7 link 1 | 11 | 7,229 | 2,940 | 3,142 (-4.79) | 3,136 (-4.74) | 2,472 / 2,780 (-7.88) | 468 / 357 (+7.98) |
| base 7 link 1 | 13 | 6,234 | 2,566 | 2,710 (-3.67) | 2,700 (-3.49) | 2,176 / 2,396 (-6.07) | 390 / 304 (+6.70) |

P6(c) as pre-registered held on the finer sections (44 of 44 cells within 3 sd) and its reading ("no
symmetry beyond the `g`-part") was refuted on the three construction sections (`|z|` up to 10.4), where
the first run showed both signs. The second measurement found the mechanism, the same at 12 of 12
cells: **the mirror `n -> 12gm - n` preserves divisibility by every prime of `gm`**, so a member sharing a
prime `l` with `m` has a partner sharing it, and the pair is `l` times a symmetric pair about `6gm/l`,
symmetric in `F_2` iff both cofactors are prime: the Goldbach pairing at scale `1/l`, at a rate above the
`F_2` density (ratio 1.37, 1.21, 1.10, 1.08 at base 3 link 2 for `g = 5, 7, 11, 13`; 1.72 to 1.28 at base 7
link 1); while a member coprime to `m` has a partner coprime to `m` and to `g`, which must avoid the primes
of `6gm` and so is a semiprime less often than the density says (ratio 0.99, 0.98, 0.98, 0.98 at base 3
link 2; 0.86 to 0.91 at base 7 link 1). The finer sections are within noise because their axes `m` are
few and small. The local-density null changes nothing (the density gradient along the section is not the
cause). So the exact statement of the mirror symmetry of a field is: **the symmetric structure of `F_j`
about `6gm` is carried by the primes dividing `6gm` and by nothing else; through each such prime `l` it is
the additive pairing of `F_{j-1}` about `6gm/l`; pairs coprime to the axis are below the null.** The
mirror about `6gm` is the mirror `n -> -n` of the machine `{primes of gm}` (W7's mirror at another
centre; CRT), and what it preserves is that machine's strike pattern; a symmetry of the overlay, not of
any field alone.

---

## 7. Location rules found, field by field

| field | location rule | status | blind to |
|---|---|---|---|
| `F_1` (the gears' home strikes) | none: the primes | the root | every composite column member |
| square field `Q` | `{(r^2 - 1)/6 : r prime}`, right members only; the cuts `c_{k+1} = p_k^2`; in section `k + 1` the squares of machine `k`'s gears above `p_k`; empty in every finer section | PROVED (E5, E7), 20 of 20 sections | every left member; every column but `(r^2 - 1)/6`; every finer section |
| `F_2` | `{p k_r + eps_r k_p}` over primes `p <= r`, side `eps_p eps_r`; equivalently the teeth of each gear `g` thinned by the primality of the cofactor; `F_1 = (F_2 n 5S)/5` | PROVED (E2, E3); no closed form without the primes | the twins and the columns whose composite members are all in deeper fields |
| `F_j`, `j >= 3` | `union_p p.F_{j-1}`; confined to the columns struck by `{5..p'^{2/j}}` in the section below `p'^2`; `F_{j-1} = (F_j n 5S)/5` | PROVED (E2, E3, E7); 0 exceptions to the confinement at 20 sections | the open set of `{5..p'^{2/j}}`, a periodic closed-form set |
| the overlay `F_2 u ... u F_J` on the sight | the struck set: `union_g {+-k_g (mod g)}`, periodic with period `prod g` | PROVED (skeleton 2; E1, E4) | the open set of the engine: the twins below `q'^2` |
| mirror axes of `F_j` | every multiple of `6g` for every prime `g`; the symmetry through each prime `l | gm` is the pairing of `F_{j-1}` about `6gm/l`; radius 1 = the both-`F_{j-1}` column | PROVED (E6) + measured (section 6) | pairs coprime to the axis |

**Can the overlay's gaps be located from the fields' rules?** No, and exactly why: the gaps are the
complement of the overlay; the overlay on the sight has a closed form (periodic) and its complement on
the section is the phase-zero cover question (length_face.md LF1, step 8); the fields partition the
overlay, and a partition of a set carries no information about the set's complement. The only way the
labels can bear on the complement is through E3, which recovers `F_1` from any field, i.e. the primes'
positions: step 8 restated.

---

## 8. Proof attempt, and the exact obstruction

**8.1 The attempt, in the fields' coordinate.** Take the section `[p^2, p'^2)`, engine `{5..q}`, and suppose
every column struck. By E7 the columns open under the cube-core `{g < p'^{2/3}}` are struck only by
`F_2`, each by a pin `g x r` with `g >= p'^{2/3}` a tail gear and `r` a prime in `[g, p'^2/g)` (E8). So the
core-open columns (a periodic set: the CRT-open columns of the cube-core on the section) are covered by
the dilates `g.(F_1 n (p^2/g, p'^2/g))` over the tail gears `g`. The fields' labels have been used
completely: `F_3` and above are confined to the core-struck columns (E7), `F_2`'s part on the core-open
columns is written as the primes of the cofactor stretches dilated (E3, E8), and the class rule (E5) says
only which side each pin lands on. The statement reached is: **the primes in the stretches
`(p^2/g, p'^2/g)`, `g` a tail gear, dilated by `g`, cover the cube-core-open columns of the section.** The
next step compares the number of core-open columns with the number of pins that can land on them (table
5: 571 against 342 pins on 295 columns at base 3 link 2; 14,300 against 9,810 on 8,076 at base 5 link 2;
114,586 against 81,124 on 66,337 at base 7 link 2): a count, and every count is matched by a machine
with no open column (proof_skeleton.md IV.2). Non-count: the positions of the pins are the positions of
the primes of the cofactor stretches, which the induction hypothesis (a twin in every lower section)
does not locate. Stop line: the core-open count is the sieve's main term; the pin count is the
semiprime count in a short interval (Brun / Chen); the pins' positions are the primes' positions.

**8.2 The exact obstruction, as a construction.** (i) Set-theoretic, before any sieve: let `C` be the
composite survivors on the sight and `{F_j}` its partition by `Omega`. Any relabelling `{F'_j}` of `C`
(any partition of the same set) has the same complement, the same twins, the same overlay. So no
property of the labels that is invariant under relabelling (which field hits where, how the fields
relate, their mirror structure, their blind sets as sets of columns of the overlay) can decide the
complement. (ii) A property that is NOT invariant is one that sees `F_1`, and by E3 every field sees
`F_1` exactly: `F_1 = (F_j n 5S n ... n 5^{j-1} S)/5^{j-1}`. So the labels do carry the whole
information, and using it is using the primes' positions: the root. (iii) What a sieve computes about
the labels is their divisibility counts; the twist by the parity of the labels is invisible to it: the
column sign of length_face.md is `sigma(k) = (-1)^{Omega(L_k) + Omega(R_k)}`, the parity of the sum of
the two members' field indices, and a twin column is the column with field indices `(1, 1)`. LF3 / LF4
(length_face.md 3.1) exhibit, at each `q = 11 .. 53`, a set (`O^-`, the open columns with odd index sum)
with the same sieve data as the open set to square-root size and no element below `(q'^2 - 1)/6`: **the
sieve cannot see the field index mod 2, and the twin condition is a condition on the field indices.**
Smallest instance: `q = 11`, `O^-` empty below column 28 against a target of 20 columns (length_face.md
3.1; cited, not rebuilt).

**Plainly: it is the parity problem again**, and the fields' coordinate makes it exact in one sentence:
the fields are the `Omega`-strata of the overlay, the sieve sees the overlay and not the strata, the
twins are the columns whose strata are `(1, 1)`, and Selberg's example is the set of columns with an odd
stratum sum. The owner's proposal, to split the overlay into fields and study each, is the split the
sieve cannot make; what the split shows is that each field is the primes dilated (E3), aperiodic (E4),
and confined for `j >= 3` to the small gears' teeth (E7), and that the overlay alone is periodic.

**What is NOT excluded**, stated as a construction: a use of E8 ("a twin is a core-open column whose every
tail cofactor is composite, i.e. struck at its own scale") that is neither a count nor a free-phase
cover. E8 is a fixed-point statement: openings here are columns whose tail cofactors are strikes there.
The tail cofactors of a column `c` sit at the scales `c/g` for the tail gears `g` whose teeth hold `c`;
those are lower stretches under lower engines, where by the induction the twins exist; but a twin there
is an opening there, which is a strike here (the hand-up), and E8 needs strikes there for an opening
here. So the induction hypothesis feeds the wrong side of E8, and this lane found no way to turn it. The
one instance where the tail pins alone decide a section, `p = 29` (section 5; origin_mechanic.md 3),
rests on which cofactors are prime.

---

## 9. What is new

1. **E3, every field is the primes dilated by 5**: `F_{j-1} = (F_j n 5S)/5`, so no field has a location
   rule the primes lack, and the square field's rule is the primes' (`F_1 = sqrt(Q)`). Elementary; stated
   as the reason the owner's per-field location rules cannot exist.
2. **E4, no field is periodic and the overlay on the sight is**: an aperiodic partition of a periodic set,
   with a two-line proof through E3.
3. **E6 with its census (section 6)**: the mirror law of a dilate as the exact source of every field's
   mirror axes; the symmetry of `F_j` about `6gm` through each prime of `gm` is the additive pairing of
   `F_{j-1}` about the reduced axis, the twin is its innermost radius; growth 2.00, 46.83, 110,944 per
   axis along the chain from base 3 with the twin's share falling to `7 x 10^{-7}`; the flat radial
   profile; and the sign split found on the second measurement (sharing a prime with the axis: above
   the null at 12 of 12; coprime to the axis: below it at 12 of 12), i.e. the symmetry is carried by the
   axis's primes and by nothing else.
4. **E7, the deep fields' blind sets**: `F_j` (`j >= 3`) confined to the teeth of `{5..p'^{2/j}}` in the
   section below `p'^2`, 0 exceptions at 20 sections; the square field empty in every finer section and
   equal to the cuts and the lower machine's squares in the construction's sections.
5. **E8, the twin as "open here, struck there"**: a column is a twin iff it is cube-core-open and every
   tail cofactor is composite; gate 20 of 20; the pins written out at `p = 17, 29`.
6. **The census** (tables 2.1, 2.2, 4, 5): per field, per side, per section and per sight, with the
   both-`F_j` columns (5,518,555 twin semiprimes against 1,027,948 twin primes at base 3 link 3), the
   one-field shares (1.000, 0.786, 0.523 along base 3; 0.38-0.67 at the record runs, ordinary among
   controls), and the alternating side bias at `j = 2, 3, 4` (9 of 9 signs; Meng 2018 at the stop).
7. **The obstruction in the fields' words** (8.2): the sieve cannot see the field index mod 2 and the
   twin is the column with indices `(1, 1)`; the relabelling invariance (i) is a set-theoretic brick in
   front of the parity brick.

Prior art, one line each, at the stop points: the symmetric prime pairs about `6m` are Goldbach's
representations of `12m` (Hardy-Littlewood 1923 for the count and its singular series); the alternating
side bias is Meng, Algebra & Number Theory 12 (2018); the `Omega`-strata counts are Landau's (not used);
the two-prime lemma and the thresholds are the kernel's and S17; the parity example is Selberg (1949)
as cited in length_face.md. No web check was run in this lane (no web access); items 1-5 carry the
status "prior-art check not yet run"; the lane expects 1, 2, 4 to be found elementary and unrecorded
and 3's sign split to be a known consequence of the singular series.

---

## 10. Verdict and scorecard

**ROOT, with the fields closed as a coordinate for a location rule.** The fields are exact objects: the
`Omega`-strata of the overlay, each the primes dilated (E3), none periodic (E4), the overlay on the sight
periodic (E1), each sitting on both sides by the parity of its class `-1` count (E5), each with mirror
axes at every multiple of `6g` whose symmetry is the pairing of the field below about the reduced axis
(E6, measured with the growth the owner expected: 2.00, 46.83, 110,944 pairs per axis along base 3, and
the mechanism of the off-axis sign split), the deep fields confined to the small gears' teeth (E7) and
the square field to the cuts (E7). The location rules found are the square field's and the deep fields'
blind sets; `F_2` has none but the primes'. The overlay's gaps cannot be located from the fields' rules
because a partition does not see its complement (8.2 (i)) and because seeing it means seeing the primes
(E3); and what a sieve computes about the labels cannot see their parity (8.2 (iii)): the parity problem
again, with the smallest instance `O^-` at `q = 11`. Where the difficulty moved: nowhere new; the fields'
coordinate states the wall's sentence exactly (the sieve sees the overlay, the twin is a statement about
the strata), and adds E8 as the fixed-point form of the origin, whose induction runs the wrong way.

- PROVED (in writing): E1-E8. Kernel: none new (E7 is `primeOrSemiprime_of_rough_lt_cube` per depth; E6
  is D2 read twice).
- MEASURED: tables 2.1, 2.2, 4, 5, 6; the new readings (the alternating bias, the off-axis sign split, the
  uncapped growth) by two methods or on two or more sections each.
- ROOT: 8.1, 8.2.

| # | prediction | verdict | evidence |
|---|---|---|---|
| P1 | gates | **CONFIRMED** (a), (b), (c) in full, after correcting the gate's reading of table A's depth | `fd_gate.py`, `results/gate.log` |
| P2 | identities T2, T4, `Q` right-only, T5: 0 mismatches | **CONFIRMED** 0 at 20 sections and 13 sights; T5 0 of 171,243 axes in 18 sections | table 2.1 |
| P3 | confinement of `F_j`, `j >= 3`; the square field | **CONFIRMED** 0 exceptions, 20 of 20; empty in 12 of 12 finer sections; the lower machine's squares in 7 of 7 construction sections | section 5 |
| P4 | side bias alternates with `j` in the three largest sections | **CONFIRMED** 9 of 9 signs (sizes `3 x 10^{-5}` to `3 x 10^{-3}`) | 2.3 (2) |
| P5 | one-field share `>= 0.75` at `p = 23..53` with `F_2 >= 0.6`; `< 0.65` at the runs with `F_2 < 0.35` | **origin 7 of 8** (refuted at `p = 43`: 0.729); **height 3 of 4** (refuted at m23: 0.667, `F_2` 0.364); direction held at every pair | section 4 |
| P6 | (a) identity; (b) growth along base 3 and twin share `< 0.05` at link 3; (c) off-axis rate at the density within 3 sd in `>= 9` of 12 finer sections | **(a) CONFIRMED** 0 mismatches; **(b) CONFIRMED** 2.00 < 46.83 < 110,944, share `7 x 10^{-7}`; **(c) CONFIRMED on the finer sections** 12 of 12, **and its reading refuted** on the construction sections, mechanism found (the axis's primes) | section 6 |
| P7 | cube-core: 0 exceptions; ratio in `[1.3, 3]`; pins `g x r` | **0 exceptions 20 of 20; pins' form 20 of 20; ratio 6 of 8** (refuted at `p = 29`: 4.00, `p = 41`: 3.33) | section 5 |
| P8 | twins from the fields = twins direct | **CONFIRMED** 0 mismatches, 20 sections, 13 sights | table 2.1 |
| owner (a) | each field has its own structure as the machine grows | **held as E3**: the structure is the primes' at scale `1/p`; no field has a period | E3, E4 |
| owner (b) | the fields relate by dilation | **held exactly** (E2) | E2 |
| owner (c) | the section relative to each field pinpoints locations | **held for the deep fields and the square field**, not for `F_2` | E7, section 5 |
| owner (d) | growing mirror symmetry in the composite fields | **held**: 2.00, 46.83, 110,944 pairs per axis; it is the Goldbach pairing through the axis's primes, the twin its innermost radius | E6, section 6 |
| owner (e) | rules in each field to locate twin gaps | **refuted** as sufficient: 8.2 (i)-(iii) | section 8 |

---

## 11. Dead ends (bricks), each with its refuting instance

| idea | dies at | instance | why it cannot be revived |
|---|---|---|---|
| a location rule for `F_2` (or any `F_j`) not through the primes | E3 | `F_1 = (F_2 n 5S)/5`: the primes below 25 are `{25, 35, 55, 65, 85, 95, 115}/5` | a rule for `F_j` is a rule for `F_1` |
| a period of a field | E4 | `5 + 5P = 5(1 + P)` composite for every period `P` | descends from `F_j` to `F_1` by E3 |
| the fields' labels as a lever on the overlay's gaps | every section | any relabelling of the composites leaves the twins fixed (8.2 (i)) | a partition does not see its complement |
| the labels' parity through a sieve | `q = 11` | `O^-` empty below column 28 against 20 (length_face.md LF3 / LF4) | the sieve sees the overlay's divisibility counts, not the strata |
| the blind sets intersected as a non-count rule | every section | `B_2 n ... n B_J` = core-open and unpinned (E8); core-open is a CRT count, pins are the primes' positions | the count's mechanism, not its replacement |
| the mirror symmetry as a lever on the twin | the radial profile | at link 3 radius 1 holds 2,594 pairs among a flat profile of 2,599 to 4,103 per radius | the twin is one radius of the Goldbach pairing |
| "no symmetry beyond the `g`-part" | base 3 link 2, `g = 5` | `z = +10.42`; sharing a prime with `m`: +62.45, coprime: -6.01 | the mirror preserves every prime of `6gm`; corrected the same day |
| E8's induction | every section | the induction gives openings below, E8 needs strikes below | the hand-up turns lower openings into strikes here, not into openings |

## 12. Open items on the part alone, sorted

- **Closed here.** E1-E8; the census on 20 sections and 13 sights; the mirror census with its mechanism;
  the blind sets; the pins at `p = 17, 29`.
- **Measurement with no structural content.** The one-field shares as a function of the section's top;
  the exact size of the alternating bias; the radial profile's singular-series bumps; the pins per tail
  gear.
- **Root question in disguise.** 8.1's last line; any bound on the pins against the core-open columns.
- **Genuinely open on the part alone, with the attack.** None on the fields as a coordinate. What this lane
  hands back is E8 as an object: the twin as the fixed point "open here, every tail cofactor struck
  there". An attack that is not a count would have to show that the tail gears' cofactor stretches
  `(p^2/g, p'^2/g)` cannot all place a prime on the core-open columns' cofactors at once; the control is
  the tooth family with only the tail gears' cofactors freed (origin_mechanic.md 10's control, now with
  the fields' labels on the cofactors: a phantom pin is a pin whose cofactor is not in `F_1`).

## 13. Files

- `research/anchor235/r79/fd_common.py` - the constructions (fold, columns, `Omega` census, sections, sights)
- `research/anchor235/r79/fd_gate.py` - P1; `results/gate.json`, `results/gate.log`, `results/base3_link3_omega.npz`
- `research/anchor235/r79/fd_fields.py` - tables 2.1, 2.2 and the gates P2, P8; `results/fields.json`, `results/fields.log`
- `research/anchor235/r79/fd_blind.py` - section 5, P3, P7; `results/blind.json`, `results/blind.log`
- `research/anchor235/r79/fd_height.py` - section 4, P5; `results/height.json`, `results/height.log`
- `research/anchor235/r79/fd_mirror.py`, `fd_mirror2.py` - section 6, P6 and its second measurement;
  `results/mirror.json`, `results/mirror.log`, `results/mirror2.json`, `results/mirror2.log`
- `results/` is untracked; nothing is committed; `theory_tree.md` and `proof_skeleton.md` are not edited.
