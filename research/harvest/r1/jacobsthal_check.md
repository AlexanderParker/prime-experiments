# Three covering records, ordered: the real teeth, the project's adversary, and `h_2`

Harvester lane, round 1 of the correction pass, **2026-09-06**.  This file corrects
**verdict 1** of `research/proof/law_register.md` ("the object has a published name, and so
does the project's target") and every row of that register that cites Ziller-Morack
arXiv:1706.00317 / arXiv:1706.03668, OEIS A288815 or "Conjecture 6".

Script: `research/harvest/r1/jacobsthal_check.py`.  Output:
`research/harvest/r1/results/jacobsthal_check.out` (gitignored; every number is reproduced
below).  All 24 checks agree on the first run.

---

## 0.  The error, in one paragraph

The register said: `F_top + 1` **is** Ziller-Morack's paired Jacobsthal function `h_2`, and
their Conjecture 6, `h_2(n) < p_n^2 - p_n`, **is** the project's window statement, so the
project's target is already in print.  The identification is false from the third primorial
onwards.  Ziller-Morack's `j_2(n)` is defined (arXiv:1706.00317 Def. 2.1-2.2, as recorded in
`docs/novel/j2-upper-bound.md`) as the least `m` such that **every** paired progression
`<a,b>_m = {(a+i, b+i) : i = 1..m}` with `2 | b - a` carries a pair coprime to `n`.  The
even difference `D = b - a` is quantified **over**, not fixed.  For each prime `p | n`,
position `i` is blocked iff `i = -a` or `i = -a - D (mod p)`; by CRT both `a` and `D` may be
chosen to give **any** pair of residues, independently at every prime.  So `h_2` is the
**free-residue two-class covering record**: two arbitrary classes per gear.  The project's
machine is the single instance `D = 2` - teeth at `0` and `-2`, separation `3^{-1} (mod g)`
in the column coordinate - and that instance is one competitor in the maximum, not the
maximum.

This is not new knowledge inside the project.  `docs/novel/jk-growth-discriminator.md` has
the correct reading and the correct engine (`j_k = D (m+1)`, `m` = the longest run coverable
by `k` non-zero classes per prime; `D = 6` at `k = 2`, giving `A288815 = 6 A072753 + 6`), and
reproduces **nine published A288815 values** with it.  `docs/proofs/20-adversarial-lemma-small-K.md`
also has it right ("the two classes **arbitrary** and the primes an **initial segment**").
The register's verdict 1 contradicts both.

---

## 1.  (a) The real twin-candidate gap at primorials, against `6 F(M)`

Direct sieve of `{k : gcd(k(k+2), p_n#) = 1}` over the full period `p_n#`, maximum cyclic
difference of consecutive survivors.  No CRT shortcut, no reuse of the project's ladder.

| `n` | `p_n` | `p_n#` | survivors | **real max gap** | `6 F(p_n)` | A288815(`n`) |
|---|---|---|---|---|---|---|
| 3 | 5 | 30 | 3 | **12** | 12 | 18 |
| 4 | 7 | 210 | 15 | **30** | 30 | 30 |
| 5 | 11 | 2 310 | 135 | **42** | 42 | 66 |
| 6 | 13 | 30 030 | 1 485 | **66** | 66 | 150 |
| 7 | 17 | 510 510 | 22 275 | **108** | 108 | 192 |
| 8 | 19 | 9 699 690 | 378 675 | **150** | 150 | 258 |
| 9 | 23 | 223 092 870 | 7 952 175 | **204** | 204 | 366 |

Seven of seven agree with `6 F(M)`.  Two of seven agree with A288815, and only at `n = 4`
does the agreement mean anything (`n = 3`: 12 against 18).  The largest instance,
`p_9# = 223 092 870`, is a full 223-million-cell sieve; `6 F(23) = 6 x 34 = 204`, A288815(9)
would be 366.  **`W72`'s rescaling therefore validates the machine against the real
twin-candidate gaps, not against Ziller-Morack's Table 1.**  The register's "free external
check" instruction, taken literally, would have reported a machine-wide failure.

The unit convention throughout: a gap of `w` columns carries `w - 1` blocked columns
(`README.md` glossary), the integer gap is `6 w`, and both A288815 and the table above are in
integer units, so `A288815(n) = 6 (omega_2 + 1)` with `omega_2 = A072753(n)` the covering
length.

## 1b.  (b) `F(M)` recomputed by a column sieve

Independent of the ladder recorded in the documents: full-period sieve of the columns
`k = +- 6^{-1} (mod g)` for `g in {5..M}`, widest gap between survivors.

| gears | `{5}` | `{5,7}` | `{5..11}` | `{5..13}` | `{5..17}` | `{5..19}` | `{5..23}` |
|---|---|---|---|---|---|---|---|
| this sieve | 2 | 5 | 7 | 11 | 18 | 25 | 34 |
| documents | 2 | 5 | 7 | 11 | 18 | 25 | 34 |

Seven of seven.  The project's ladder is confirmed; the mismatch with A288815 is not a
project arithmetic error.

## 2.  (c) The free two-residue covering record, against A072753

Exhaustive DFS: cover the run `[0, L)` of columns, each gear `g in {5..M}` contributing two
**arbitrary** residue classes mod `g`, branching on which unused gear covers the leftmost
uncovered column, pruned by the exact per-gear capacity `2 floor(L/g) + min(2, L mod g)`.
The largest coverable `L` is reported.

| `K` | gears | free record | A072753 |
|---|---|---|---|
| 1 | `{5}` | **2** | 2 |
| 2 | `{5,7}` | **4** | 4 |
| 3 | `{5,7,11}` | **10** | 10 |
| 4 | `{5..13}` | **24** | 24 |
| 5 | `{5..17}` | **31** | 31 |

Five of five.  Resta's OEIS comment - "the maximal `m` such that there exist pairs
`(a_i, b_i)` mod `prime(i)`, `3 <= i <= n`, covering every number in `1..m`" - is exactly this
search, and `A288815 = 6 A072753 + 6` is exactly the rescaling to integers.  **A072753 is the
published table of the free-residue two-class covering record**, to 19 gears.

## 3.  (d) The project's own adversary `A(K)` (`docs/proofs/20`)

Same DFS, but each gear's two classes are forced to separation `3^{-1} (mod g)` (free phase),
and the **primes are free**: the pool is every prime in `[5, 3L+2]` (Lemma 4 of doc 20 makes
larger primes single-column types, granted as an unlimited wildcard).

| `K` | `A(K)` here | `A(K)` doc 20 | longest covered |
|---|---|---|---|
| 1 | **2** | 2 | 1 |
| 2 | **5** | 5 | 4 |
| 3 | **7** | 7 | 6 |
| 4 | **16** | 16 | 15 |
| 5 | **22** | 22 | 21 |

Five of five, by an implementation written for this file and independent of
`research/anchor235/r53/`.

## 4.  The three objects in one table

Longest **coverable** run of columns (the blocked-count convention, `F - 1`):

| `K` | gears | real `F(M) - 1` | project adversary `A(K) - 1` | free two-class (A072753) |
|---|---|---|---|---|
| 1 | `{5}` | 1 | 1 | **2** |
| 2 | `{5,7}` | 4 | 4 | 4 |
| 3 | `{5,7,11}` | 6 | 6 | **10** |
| 4 | `{5..13}` | 10 | **15** | **24** |
| 5 | `{5..17}` | 17 | **21** | **31** |

Continuing from the documents (`docs/proofs/20`, verified there, not re-run here):

| `K` | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|
| real `F(M) - 1` | 24 | 33 | 42 | 57 | 87 |
| `A(K) - 1` | 27 | 36 | 44 | 67 | 87 |
| A072753 | 42 | 60 | 74 | 94 | 117 |

**Where equality holds and where it first breaks.**

- **real = project adversary** at `K = 1, 2, 3` and again at `K = 10`; **first breaks at
  `K = 4`** (10 against 15), where `{5,7,11,17}` covers 15 columns and the initial segment
  `{5,7,11,13}` covers 10.  The cause is doc 20's arc mechanism: `3 a_g = g -+ 1`, so twin
  gears share an arc and the initial segment must buy both members of every twin pair
  (`the_wall.md` 5d).  These are equalities of value, not of definition: fixing the gear set
  to `{5..p_K}` makes the fixed-separation free-phase adversary **identical** to the real
  machine over its period (every phase vector occurs once per period, `the_wall.md` 5a), so
  the two columns can only differ through the choice of primes.
- **real = free two-class** at `K = 2` only (4 = 4).  It is **strictly below at `K = 1`**
  (1 against 2) and **strictly below at every `K >= 3`** (6 against 10 at `{5,7,11}`, and the
  ratio widens: 87 against 117 at `K = 10`).  The domination `F(M) - 1 <= A072753(M)` is a
  theorem, not a measurement: `D = 2` is one admissible even difference among those `h_2`
  maximises over.
- **project adversary vs free two-class**: no implication either way (they free different
  things - doc 20 frees the primes, `h_2` frees the classes), but arithmetically the free
  classes win at every `K` in `1..10` except `K = 2`, where both are 4.  Doc 20 states this
  already and its transcription of A072753 in column units (`2, 4, 10, 24, 31, 42, 60, 74,
  94, 117`) is correct.

---

## 5.  The corrected verdict

1. **`h_2` at primorials is the free-residue two-class covering record**, `A288815 =
   6 A072753 + 6`, the adversary of `the_wall.md` face 5a and of `docs/proofs/20`.  It is not
   the project's `F`.
2. **The project's `F(M)` is the real-teeth record** - the `D = 2` instance - equal to `h_2`
   at `{5, 7}` and **strictly below from `{5, 7, 11}` on** (7 against 11 as gaps, 6 against 10
   as covering lengths).  Its own external validation is the real twin-candidate gap at
   `p_n#`, `6 F(M)` = 12, 30, 42, 66, 108, 150, 204 at `n = 3..9`, computed above.
3. **Ziller-Morack's Conjecture 6 is the adversarial window statement**, strictly stronger
   than the project's: `h_2(p_n#) < p_n^2 - p_n` gives, in column units,
   `F(M) <= A072753 + 1 < (p_n^2 - p_n)/6 < (p_{n+1}^2 - 1)/6` = the project's window, so
   **Conjecture 6 implies the project's window statement**, and by their Theorem 4.1 it
   implies Goldbach and the infinitude of prime pairs at every even difference.  The converse
   fails: the project's statement constrains only one of the even differences `h_2` maximises
   over.  Conjecture 6 is verified to `p_21 = 73` (A288815 carries exactly 21 terms).

**Consequence for `docs/proofs/20`.**  Its Theorem A (no `K` primes at fixed separation cover
`W(K) = (p_{K+1}^2 - 1)/6` columns, `K = 1..10`) is **not** a partial result toward
Conjecture 6, and should not be recorded as one.  The two statements strengthen the project's
window statement along **different** axes: doc 20 frees the primes and keeps `D = 2`;
Conjecture 6 keeps the initial segment and frees `D`.  Neither implies the other, and at
`K = 1..10` Conjecture 6's adversary is the larger at every `K` except `K = 2` (117 against
87 at `K = 10`), while its window is the smaller (`(37^2 - 37)/6 = 222` columns against doc
20's 280).  Every instance of Conjecture 6 that doc 20's range touches is in any case already
verified by Ziller-Morack's own computation.  What doc 20 **should** record as prior art is
A072753 itself: the published table of the covering record its own `A(K)` is compared against,
to 19 gears, which is where the free-class adversary of `the_wall.md` 5a is already tabulated.

**Consequence for the register's "free external validation" action (W72).**  It cannot be run
against A288815.  The rescaling `6 F_col + 5` is exact and the validation that exists is the
one in section 1: `6 F(M)` against the real twin-candidate gaps at `p_n#`, seven of seven.
Separately, A072753 **is** validated by the project - not by `F`, but by
`docs/novel/jk-growth-discriminator.md`, whose `j_k` engine reproduces nine published A288815
values (`z = 2..31`) exactly; the register's phrase "which the project has not computed beyond
small `K`" is wrong on that point too, and section 2 above adds an independent check of the
first five.

---

## 6.  Where the correction was written

- `research/proof/law_register.md` - summary verdict 1; rows W16, W30, W38, W72, W84; the
  prior-art vocabulary note; the naming-corrections list.
- `docs/novel/README.md` - the "wheels and the exhaust" header entry (2026-09-06) and the
  `wheels-anchor-rescaling` entry's ACTION line.
- `docs/proofs/22-top-machine-laws.md`, `docs/proofs/23-stack-and-exhaust.md` - prior-art
  paragraphs.
- `research/proof/the_wall.md` section 5a - one line naming A072753 as the published table of
  the free-residue adversary's record.
- `docs/proofs/20-adversarial-lemma-small-K.md` - "Prior art" gains the statement of what
  Conjecture 6 is and is not relative to Theorem A.
