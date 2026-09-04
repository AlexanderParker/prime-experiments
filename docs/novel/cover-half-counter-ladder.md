# The cover-half counter ladder: exposure cap, Bonferroni depth, and why no counter bounds L

Constructor, round 30.  Status per statement in section 3; nothing is announced as new
until section 6 carries a dated prior-art verdict.

---

## 1. WHAT IT IS

**The problem.**  The project's derivation target splits (R91) into (A) a per-letter
residual and (B) `L(M)` bounded, where `L(M)` is the length of the longest REALISED legal
word - a run of consecutive gaps of `M` whose residues mod `q'` lie in `{0, +d, -d}`
(`d = 2 * 6^{-1} mod q'`) with the nonzero classes alternating.  In the anchor-2,3,5
language (anchor-235.md section 9f) the same object is `D_{q'} - 1`, one less than the
longest run of consecutive lower openings whose residues mod `q'` sit in one two-class set
`{r, r+d}`; Mechanic measured `D_g = A_kill` at seven rungs and R89 proves
`A_kill = L + 1`.  The only machine-free instrument the project has on the COVER half of
realisability ("every interior slot blocked") is R43's pruned inclusion-exclusion counter,
`N(w) = sum_{T subset Y} (-1)^|T| prod_g c_g(X u T)`.  This note asks what that counter, or
any fixed truncation of it, can say about `L`.

**Plain language.**  A legal word is realised when some slot of the machine has the word's
`m+1` prefix-sum points open AND every slot strictly between them blocked.  The first half
("the points are open") is decided by the small gears alone and is what phase saturation
tests; the second half ("everything between is blocked") is what makes the word rare.  A
counter that truncates inclusion-exclusion at a fixed depth sees the first half exactly
and the second half hardly at all: the counts it produces are period-scale numbers, and
they do not reach zero until the machine itself is small enough.  So no fixed-depth counter
caps `L`; and the exposure half's own cap on `L` - which IS computable and finite at every
machine reached - grows with the machine while `L` does not.

### 1.1 THE LADDER

For a word length `m`, with `Lambda` the legal alphabet (residue-legal values `<= F(M)`,
with or without the realised-value filter), define

    A_m       abstract T3-legal words over Lambda (closed form, 1.3);
    S_m       those surviving phase saturation at EVERY gear of M - equivalently, by
              CRT, those whose prefix-sum set X has a slot with all m+1 points open -
              equivalently the depth-0 term E_0(w) = prod_g c_g(X) of R43's counter
              is >= 1  (the EXPOSURE half);
    S_m^(s)   those whose depth-s Bonferroni upper bound
              E_s(w) = sum_{|T| <= s} (-1)^|T| N(X u T)  (s even)  is still >= 1;
    D_m       the REALISED words, N(w) > 0  (the COVER half).  D_m > 0  iff  L(M) >= m.

    A_m >= S_m = S_m^(0) >= S_m^(2) >= S_m^(4) >= ... >= D_m.

    EXPCAP(M) = max{ m : S_m > 0 }     (the exposure cap on L);
    CORRCAP   = the same at gears {5,7} only  (R75's object).

In the residue-run form: `S_m` counts the length-`(m+1)` residue-run PATTERNS (point sets
whose consecutive differences are `0` or `+-d mod q'`, alternating) that have a slot with
all points open; `D_m` counts those realised as runs of CONSECUTIVE openings.

### 1.2 THEOREM (the exposure half is decided by the small gears) - PROVED

> A word of length `m` survives phase saturation at `M` iff it survives at the
> sub-machine `{g in M : g <= 2m+2}`.  Hence `S_m(M)` and `EXPCAP(M)` depend on `M` only
> through the gears below `2m+3` and the alphabet `Lambda(M)`.

Proof: a gear `g` has `g` translates and the `m+1` points of `X` forbid at most `2(m+1)` of
them (two teeth each), so for `g > 2(m+1)` some translate always fits.  (Mechanic's
`|FREE_g| >= g - 2|X|`, round 26, read as a statement about `S_m`.)  Asserted numerically
at every `(M, m)` with `m <= EXPCAP(M)+1` at m11..m53.

### 1.3 THE ABSTRACT COUNT - closed form, asserted

With `p` padded letters and `l_a`, `l_b` letters of the two nonzero classes,

    A_m = sum_{k=0}^{m} C(m,k) p^k T(m-k),   T(0) = 1,
    T(n) = l_a^{ceil(n/2)} l_b^{floor(n/2)} + l_b^{ceil(n/2)} l_a^{floor(n/2)}   (n >= 1),

equal to direct enumeration at `m = 1..6` at every machine m11..m53.

### 1.4 THE MEASURED LADDER - exact, gated

`research/word_count_r30.py` (log `research/data/r30/word_count_r30.log`; the R75 CORRCAP
row is reproduced exactly by an automaton on the 35 x 3 corridor states as the gate).
`D_m` from the counted census (`research/occ_census_r30.py`) at m11..m37 and from
R85's CRT rows at m41.

    M     L   CORRCAP(5,7)   EXPCAP(all gears)   EXPCAP - L    S_m (all gears), m = 1, 2, 3, ...
    m11   1        1               1                 0         1
    m13   1        1               1                 0         2
    m17   1        1               1                 0         2
    m19   2        4               4                 2         3, 7, 9, 2
    m23   1        2               2                 1         3, 4
    m29   3        3               3                 0         4, 4, 1              (D_m = 3, 2, 1)
    m31   3        5               5                 2         4, 9, 12, 6, 1       (D_m = 4, 6, 2)
    m37   2       25              18 (12 realised)  16 (10)   6, 21, 52, 97, 182, 335, 571, 834, 947, 902, 820, ... 2 at m = 18
    m41   2       25              13                11        6, 19, 39, 54, 73, 104, 131, 130, 101, 70, 38, 12, 2   (D_m = 6, 5, 0)
    m43   2       11              10                 8        6, 19, 41, 70, 108, 154, 185, 144, 56, 8
    m47   4        5               5                 1        6, 15, 14, 6, 1
    m53   3      INF              21                18        7, 29, 77, 170, 343, 628, 970, 1282, 1512, 1500, ..., 2 at m = 21

("12 realised" at m37: with the hole `82` removed from the alphabet the cap is 12.)

Three facts.

* **`EXPCAP - L` is not bounded along the ladder** (16, 11, 8, 18 at m37, m41, m43, m53
  against 0..2 below).  The exposure half over-caps `L` by an amount that is
  arithmetic-selected: at m47 it is exact to one (`CORRCAP = 5`, `L = 4`), at m53 the
  corridor is infinite and only gears 11 and above cap it, at 21.
* **Fixed-depth Bonferroni adds nothing.**  At every exposure survivor of every length at
  m19..m31 (and `m <= 3` at m37), the depth-2 and depth-4 upper bounds `E_2(w)`, `E_4(w)`
  are `>= 1`: `S^(2)_m = S^(4)_m = S_m` at all 21 cells.  Meanwhile the exact `N(w)` sits
  far below `E_0(w)`: `min E_0/N` over realised words is 6..16 at `m = 1`, 845..10,742 at
  `m = 2`, 145,158 (m29) and 312,151 (m31) at `m = 3`, 4,344,055 (m37, `m = 2`) - growing
  in both `m` and `M`.
* **The first-moment threshold** (observation): with `f_legal` the fraction of gaps that are
  legal letters (from the counted census), the `m` at which `N f_legal^m < 1` is 4, 5, 6, 6, 6
  at m19..m37 against `L = 2, 1, 3, 3, 2`.

### 1.5 THE VERDICT - a gated negative

**No fixed-depth truncation of the counter, and no exposure-only argument, bounds `L(M)`
uniformly in `M`.**  The term that grows is the depth-0 count `E_0(w) = prod_g c_g(X)`:
it is the number of slots with the pattern's points open, a `P`-scale number (`>= prod_{g >
2m+2} (g - 2m - 2)` for every exposure survivor), and the higher Bonferroni terms are
`P`-scale corrections of bounded ratio to it, so `E_s(w) < 1` cannot happen at a fixed `s`
until `E_0(w)` itself is `O(1)` - i.e. until the exposure half already kills the word.  The
exposure half's cap is computable and finite at every machine reached, but it is
`EXPCAP(M)`, which is 16-18 above `L` at m37 and m53.  A uniform bound on `L` therefore
needs the cover half at FULL depth (`2^{|Y|}` per word, R43's cost curve), applied to a set
of `S_m` words that is itself unbounded in `M`.  This is R75's "the order lives in the cover
half" made quantitative: the cover half is not merely where the bound lives, it is where ALL
of it lives.

---

## 2. WHY IT MIGHT BE NOVEL

* The ladder `A_m >= S_m >= S_m^(s) >= D_m` separates the two halves of realisability by
  instrument, and the exposure half's cap `EXPCAP` is a new finite invariant of a machine
  (values above).  R75 computed its gears-5,7 shadow (`CORRCAP`); here it is computed at all
  gears, with the theorem that only gears below `2m+3` matter.
* The negative is sharper than "the counter is expensive": it says the exposure count
  over-counts the realised words by factors of `1e5`-`1e6` at word length 3 and that fixed
  truncations of the alternating sum are exactly useless (zero kills) on the survivors.
* In Jacobsthal-type language: the number of realisable residue-run patterns of length `m`
  in a two-class sieve is not controlled by any moment of bounded order of the covering
  system.

---

## 3. PROOF / STATUS

| statement | status | pointer |
|---|---|---|
| exposure decided by gears `<= 2m+2` | **PROVED**, asserted at every cell | section 1.2, `word_count_r30.py` |
| closed form for `A_m` | **PROVED**, asserted vs enumeration `m <= 6` | section 1.3 |
| `S_m`, `EXPCAP` at m11..m53 | **SCRIPT-VERIFIED**, exact (DFS over T3 words with per-gear bitmasks; CORRCAP reproduces R75 at all nine cells) | `research/data/r30/word_count_r30.log` |
| `S^(2) = S^(4) = S` on the survivors, m19..m37 | **SCRIPT-VERIFIED**, 21 cells, exact integers | same |
| `D_m` at m11..m37 | **SCRIPT-VERIFIED** from the counted census (exact, cyclic, gated) | `research/occ_census_r30.py` |
| the verdict of 1.5 | **PROVED** for fixed-depth truncations given the measured `EXPCAP` growth; "no counter of any kind" is JUDGMENT, NOT RESULT | section 1.5 |

---

## 4. IMPLICATIONS

* (B) - `L(M)` bounded - cannot be supplied by the exposure half or by any bounded-depth
  inclusion-exclusion; the derivation needs a statement about the cover half at full depth
  or a different object entirely.
* For Mechanic's residue-run form of `L`: the exposure survivors `S_m` are the candidate
  residue-run patterns; a run statistic on the lower period measures `D_m` directly, and the
  two meet in the middle exactly as `D_g` and `A_kill` did in round 29.
* The counted census that supplies `D_m` (and `occ`, `Phi`) is now a streamed vehicle that
  reaches m37's period with no array beyond one chunk (`research/occ_census_r30.py`).

---

## 5. UNSOLVED QUESTIONS IT TOUCHES

Boundedness of `A_kill = L + 1` (open); requirement (D) and the tolerance route (R14/R26);
Jacobsthal-type extremal problems for the two-dimensional sieve; the cost of exact zero
certificates for residue-run patterns (`2^{|Y|}`, R43).

---

## 6. PRIOR-ART CHECK

**Not yet checked** (this lane has no web access).  Search terms for the manager:
"Bonferroni inequalities sieve of Eratosthenes covering runs", "inclusion-exclusion
truncation lower bound covering systems residue classes", "longest run of consecutive
sieved integers in prescribed residue classes", "Jacobsthal function residue run".
Nearest relatives inside the project: `renewal-ladder.md` (R38, the ladder's depth `s`
terms), `uniform-order-bound.md` (R74/R75, the corridor cap), `even-j-mechanism.md` (R89,
the word reduction this note bounds).
