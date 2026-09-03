# The even-J mechanism: the word reduction, the same-tooth lemma, and the par-trading residual

Constructor, round 29.  Status per statement in section 3; nothing is announced as new
until section 6 carries a dated prior-art verdict.

---

## 1. WHAT IT IS

**The problem.**  The project's live obligation (D) at a step `M -> M + q'` decomposes, by
the attainment theorem, into one inequality per depth `J`:

    Delta_J(M)  =  Q*_J(M; q') - F_2(M)  <=  s_min(q'),

`Q*_J` the maximum span of a *word-legal* `J`-window (`per-j-window-analogues.md`).  Round 28
found that the **odd** members of that family have a route - at `J = 5` the maximiser is a
palindrome at both machines where `J = 5` exists, so the mirror/antipode lever bites - and
that the **even** members do not: Theorem B of round 28 proves a literal even-`J` window is
*never* a palindrome, so at even `J` the lever has nothing to act on.  This note is the
replacement.

**Plain language.**  When a new gear is added it deletes a run of openings and the gaps
around them merge.  The deleted openings sit alternately on the gear's two teeth.  If the
run is *even* in the right sense, the first and last deleted opening are on the SAME tooth,
and the whole middle of the merged gap is then a multiple of the new gear.  What this note
shows is that the depth of such a run is not a new quantity at all - it is the length of the
longest word of legal letters the old machine actually realises - and that going one letter
deeper costs the merged gap almost exactly the letter it just bought.  "Almost exactly" is a
number, and it is small.

### 1.1 THEOREM (THE WORD REDUCTION) - PROVED

> Let `Lambda(M)` be the legal letters (gap values `v <= F(M)` with `v = 0` or `+-2c` mod
> `q'`), and let a **legal word** of `M` be a run of consecutive gaps of `M`, all in
> `Lambda(M)`, whose nonzero T3 classes strictly alternate.  Write `L(M)` for the length of
> the longest legal word that `M` actually realises.  Then for every `J >= 3`
>
>     Q*_J(M; q')  >  -inf   <=>   L(M) >= J - 2,
>
> so `J_max(M) = L(M) + 2` and `A_kill(M -> q') = L(M) + 1`.

**Proof.** (=>) The `J-2` middles of a word-legal `J`-window are, by definition, `J-2`
consecutive gaps of `M`, each a legal letter, T3-alternating: a realised legal word of
length `J-2`.  (<=) Conversely, take any occurrence of a realised legal word of length
`J-2`; the gap immediately before and the gap immediately after extend it to `J` consecutive
gaps, and word-legality constrains only the middles, so that window is word-legal.  For
`A_kill`: a chain of `k` consecutive openings killed at one phase of `q'` has `k-1` interior
gaps, which are forced legal by T2/T3, so `A_kill <= L + 1`; and conversely a realised legal
word of length `k-1` is, by the attainment theorem's CRT step, killable in full at some
translate, so `A_kill >= L + 1`. []

**ATTRIBUTION, and it matters.**  The `(=>)` half is Mechanic's round-28 index observation
("a word-legal window of `J` gaps has `J-1` interior openings deleted by one phase, i.e. a
realised kill chain of arity `J-1`; its word has `J-2` letters"), which gave the inequality
`J_max <= A_kill + 1`.  What is added here is the converse, which turns the inequality into
an **identity**, and the `L(M)` formulation, which is the operationally useful one: the
depth cap of the whole per-J family is a question about the **shallowest** dictionary the
project has - "what is the longest run of consecutive gaps that are all legal letters?" -
and it is decidable by a handful of CRT calls with no census.  R81 recorded
`J_max = A_kill + 1` as MEASURED 8/8; it is a theorem.

### 1.2 THEOREM (THE SAME-TOOTH LEMMA) - PROVED

Assign tooth `t_i in {+,-}` to the `i`-th killed opening.  A middle of class `0` (padded)
leaves the tooth fixed; a middle of class `+-1` flips it.  Hence

> the middle span `x_{J-1} - x_1` is `= 0 mod q'` **exactly when the number of NON-PADDED
> middles is even**, and `= +-2c mod q'` otherwise.

For a LITERAL even-`J` window all `J-2` middles are non-padded, so the middle span is
`= 0 mod q'` and therefore `>= ((J-2)/2) q'`: **the first and last deleted opening of a
literal even-`J` chain sit on the same tooth.**  This is round-28's Theorem A in its even
case, now with a reason rather than a count, and it is the structural fact the palindrome
route cannot supply.  Checked on **every** realised legal word at every machine with an
exact source: 38 words, 0 violations, and the two padded even-`J` maximisers (`(12,37)` at
m31, `(41,14)` at m37) have middle sums `49 = 12 mod 37` and `55 = 14 mod 41` - the
hypothesis is non-vacuous.

### 1.3 THE NEW OBJECT: THE PAR-TRADING RESIDUAL

For a realised legal word `v` write `Phi(v)` for its **flank envelope**, the maximum of
`g_L + g_R` over occurrences.  For `v = u.x` (drop the last letter) or `v = x.u` (drop the
first) define

    eps(v)  =  Phi(u) - Phi(v) - x.

`eps` is the amount by which the flank envelope FAILS to pay exactly the letter just added.
It is the exact form of par trading (R30), and it is the derivative of the whole family:
`span(v) + Phi(v) = span(u) + Phi(u) - eps(v)`, so when `u` and `v` are the maximising words
at depths `J-1` and `J`,

    Q*_J  -  Q*_{J-1}  =  - eps(v),      hence
    Delta_J  =  Delta_{J-1}  -  eps,     Delta_2 = 0.

(For a general pair `u`, `v` the identity is the one on spans; the chain form needs both to
be maximisers, and section 1.4(c) measures exactly that case separately from 1.4(b), which
measures `eps` over every cell.)

So **`Delta_J = O(1)` uniformly in `J` is exactly: `eps` is `O(1)` per letter, and `L(M)` is
bounded.**  That is a clean decomposition of the round's derivation target into two named
lemmas, one about a single letter and one about depth.

### 1.4 THE MEASUREMENTS - exact, gated

`research/evenj_r29.py`, log `research/data/r29/evenj.log`.  Sources: full cyclic period
scans at m11..m23; Mechanic's exact full-period 4-tuple censuses at m29/m31/m37; the exact
m29 5-tuple census for length-3 words.  The script reproduces **21 of the 22 recorded
`Q*_J` cells of R68/R81, with 0 mismatches** (the 22nd, `Q*_5(31)`, needs a 5-tuple census
of m31 that does not exist and is printed as NO DATA, not filled in).

**(a) `L(M)` and the depth law.**

    machine    11  13  17  19  23  29  31  37
    L(M)        1   1   1   2   1   3   3   2
    J_max=L+2   3   3   3   4   3   5   5   4     recorded 3 3 3 4 3 5 5 4   8/8
    A_kill=L+1  2   2   2   3   2   4   4   3     recorded 2 2 2 3 2 4 4 3   8/8

Every `L` is CERTIFIED (the next length up has no realised legal word, in a source of
sufficient arity), so every `EMPTY` cell of the per-J table is a certificate.  Beyond m37 the
identity is checked against the recorded `A_kill` values rather than a dictionary:
`A_kill(41) = 3`, `A_kill(43 -> 47) = 3`, `A_kill(47 -> 53) = 5` and `A_kill(53 -> 59) = 4`
give `L = 2, 2, 4, 3` and `J_max = 4, 4, 6, 5`.  **At m43 and m47 the word side was computed
directly this round and agrees, which turns the identity into a two-way tool:**
`research/l43_words_r29.py` refutes all eight phase-saturation survivors among the 31
candidate length-3 legal words at m43 (0 undecided), so `L(43) = 2` and
`A_kill(43 -> 47) <= 3` with no census anywhere; and `research/l47_words_r29.py` decides the
whole of `L(47)` in FOUR CRT calls - `(18,35,18,35)` REALISED in 4 s, `(35,18,35,53)`,
`(35,18,53,35)` and `(35,18,35,18,35)` refuted, 0 undecided - giving `L(47) = 4` and hence
`A_kill(47 -> 53) = 5` and `J_max(47) = 6`.  That last is an INDEPENDENT CONFIRMATION of a
value the project obtained in round 25 by a chain census, at a cost of 19 core-minutes and
with no period anywhere; `(18,35,18,35)` is the literal alternation `abab` and is the first
realised legal 4-word recorded in the project.  The m53 entry is an out-of-sample
confirmation worth naming: **Mechanic's round-28 `F(59) = 161` run took `JMAX = 5` as
EXHAUSTIVE on exactly this argument**, and R89 is the theorem that licenses it.
In particular `J = 6` is empty at every machine m11..m43 for the single reason that
`L <= 3` there - **no J = 6 sweep is needed anywhere below m47.**  And the cap is NOT
monotone: `L(47) = 4` makes `J = 6` non-empty at m47, the first machine where it is, and it
is a new maximum for `L` - so any reading of the earlier rows as "`L <= 3`" is refuted by
this round's own measurement.

**(b) The par-trading residual, over every cell where both `u` and `v` are realised.**

    30 (machine, word, direction) cells
      LITERAL cells  (v contains no padded letter):  14 of 14 have |eps| <= s_min
      PADDED  cells  (v contains q' or a+q' ...)  :  10 of 16
    mean |eps| = 5.60   against  mean s_min/2 = 5.80

All six failures carry the padded letter, and they are large: `eps = -20` twice (m31,
dropping `37` from `(12,37)` and `(37,12)`), `+13` twice (m31, `(25,37)`/`(37,25)`), `+15`
twice (m37, `(27,41)`/`(41,27)`).  **Par trading is a LITERAL law; the padded letter breaks
it, and it breaks it at exactly the machine whose rows fail.**

**(c) Along the maximising chain - the quantity `Delta_J = O(1)` is actually about - the
residual is far smaller than `s_min`:**

    m11 J=3 +3 | m13 J=3 -2 | m17 J=3 +0 | m19 J=3 -2, J=4 -1 | m23 J=3 -4
    m29 J=3 -3, J=4 +3, J=5 +0 | m31 J=3 -2, J=4 -1 | m37 J=3 +0

    max |eps| over the twelve chain cells = 4,   against s_min running 4 .. 14.

The literal `Delta_J` table it produces is

    machine    11  13  17  19  23  29  31  37
    J = 3      -3  +2  +0  +2  +4  +3  +2  +0
    J = 4       E   E   E  +3   E  +0  +3   -      (E certified empty, - no literal word)
    J = 5       E   E   E   E   E  +0  nd   E

reproducing round 28's numbers from an independent vehicle and adding the m31 literal
`J = 4` maximiser `(6,25,12,28)`, span 71, `Phi = 34`, which was not previously exhibited.

**(d) The even-J flank ceiling.**  `Phi_J <= F_2(M) - b` at every non-empty literal even-J
cell: margins `+5` at m19, `+10` at m29, `+9` at m31.  (Predicted 5, 10, 9 before the run.)

**(e) The half-split.**  Writing an even-J maximiser as `h_L = g_L + w_1`, `h_R = w_{J-2} +
g_R`, the 2F wall allows `h_L + h_R <= 2 F_2`; measured,

    m19 (7,15,8,4)     h 22/12   min/F_2 0.387   span/F_2 1.097
    m29 (22,10,21,2)   h 32/23   min/F_2 0.418   span/F_2 1.000
    m31 (11,12,37,28)  h 23/65   min/F_2 0.338   span/F_2 1.294   (padded)
    m31 (6,25,12,28)   h 31/40   min/F_2 0.456   span/F_2 1.044   (literal)
    m37 (21,14,41,15)  h 35/56   min/F_2 0.389   span/F_2 1.011   (padded)

The smaller half sits in `[0.338, 0.456] F_2` at every cell - **R22's "both flanks maximal
is forbidden" in quantitative form at even depth**, and the span sits at `1.00-1.29 F_2`
against the `2 F_2` the wall permits.

**(f) The work word-legality does, `F_J - Q*_J`** - the quantity the
spectrum-plus-depth certificate discards - from the same sources, `F_1` and `F_2` asserted
against the corpus in every row:

    M      F_J, J = 1..              F_J - Q*_J at J = 3, 4, 5
    m11    7,11,16,18,23,26           8    .    .
    m13    11,16,23,26,28,31          5    .    .
    m17    18,25,28,33,35,40          3    .    .
    m19    25,31,35,38,47,50          2    4    .
    m23    34,39,50,58,65,77          7    .    .
    m29    43,55,65,70,85             7   15   30
    m31    58,68,85,90                0    2    .
    m37    88,90,97,105               7   14    .

    J = 3 : 8 cells, 0..8, mean 4.9 | J = 4 : 4 cells, 2..15, mean 8.8 | J = 5 : 30

**Legality's work grows with depth and shows NO parity effect**: the even/odd split is
structural (palindromes, same tooth), not a size effect.  The `J = 5` cell, 30, is exactly
why the spectrum-plus-depth certificate fails at `29 -> 31`.  Free cross-check: the m29 row
reproduces round 28's new `F_5(29) = 85` by a different vehicle from the one that first
produced it, and the m31/m37 rows reproduce the recorded spectra exactly.

---

## 2. WHY IT MIGHT BE NOVEL

* The word reduction turns a depth quantifier into a dictionary question at arity 1, and the
  direction that matters (the converse) is the one that makes the emptiness certificates of
  the per-J family FREE: `J = 6` needs no computation at any machine on record.  The
  classical shadow - "a sum of `j` consecutive gaps is at most `F_j`" - has no depth cap in
  it at all.
* The same-tooth lemma is elementary but it is the first statement in the project that
  separates even from odd depth by an ARITHMETIC fact (which tooth the chain ends on) rather
  than by a counting parity.
* `eps(v)` is a new construct: a per-word derivative of the flank envelope.  The measured
  statement "the flank envelope pays for an added literal letter to within 4" is the sharp
  local form of an anti-clustering statement the literature states only asymptotically, and
  it isolates the padded letter as the sole violator.

---

## 3. PROOF / STATUS

| statement | status | pointer |
|---|---|---|
| the word reduction `Q*_J > -inf <=> L >= J-2` | **PROVED** (two directions, the first Mechanic's r28 index observation) | section 1.1 |
| `J_max = L + 2`, `A_kill = L + 1` | **PROVED**, and SCRIPT-VERIFIED 16/16 against the recorded rows | `research/evenj_r29.py` GATE 2 |
| `L(43) = 2`, `L(47) = 4` computed from the word dictionary | **SCRIPT-VERIFIED**, exact, scan-free, 0 undecided; `L(47) = 4` independently confirms `A_kill(47 -> 53) = 5` | `research/l43_words_r29.py`, `research/l47_words_r29.py` |
| the same-tooth lemma | **PROVED** (T2/T3), SCRIPT-VERIFIED on 38 realised legal words, 0 violations | section 1.2 |
| `Delta_J = Delta_{J-1} - eps` | **PROVED** (definitional, from the word reduction) | section 1.3 |
| `|eps| <= s_min` at literal cells | **MEASURED** 14/14 - NOT proved | section 1.4(b) |
| `|eps| <= s_min` at padded cells | **REFUTED**, 10/16, all six failures carrying `q'` | section 1.4(b) |
| `max |eps| = 4` along maximising chains | **MEASURED**, 12 cells | section 1.4(c) |
| `Phi_J <= F_2 - b` at literal even J | **MEASURED** 3/3, margins 5/10/9 | section 1.4(d) |
| half-split band `[0.338, 0.456] F_2` | **MEASURED**, 5 cells | section 1.4(e) |

GATE: `uv run python research/evenj_r29.py` - reproduces 21 of the 22 recorded `Q*_J` cells
with 0 mismatches, asserts `F` and `F_2` out of every source before using it, and prints
`NO DATA` (never a filled-in value) for the one cell no exact source reaches.

---

## 4. IMPLICATIONS

* **The derivation target has a decomposition.**  `Delta_J = O(1)` uniformly in `J` follows
  from two lemmas: (A) `|eps|` bounded per literal letter, and (B) `L(M)` bounded.  Neither
  is proved; both are measured over the whole corpus; and (B) is the same statement as
  "A_kill is bounded", which the project has been tracking since R45 under a different name.
* **The padded letter is the whole residue.**  Every failure in this note - the six `eps`
  failures, the `span/F_2 = 1.294` outlier, and (round 28) the three failing rows of the
  increment law - is a cell containing `q'` as a realised gap.  The even-J LITERAL half is
  clean at every cell of the corpus.
* **Emptiness is now free at even depth.**  `J = 6` is certified empty everywhere from a
  one-line dictionary fact.  For a new machine the depth cap costs the decision of its
  legal words of length `L+1`, which is a handful of CRT calls (at m43 this round: 31
  candidate words, 23 refuted by phase saturation for free).

---

## 5. UNSOLVED QUESTIONS IT TOUCHES

Requirement (D) and the tolerance route (R14/R26); `Delta_J = O(1)` uniformly in `J` (the
project's current derivation target); the boundedness of `A_kill`, equivalently of `L(M)`,
which is open in general; Jacobsthal-type extremal problems for the two-dimensional sieve.

---

## 6. PRIOR-ART CHECK

**Not yet checked** (this lane has no web access).  Search terms for the manager:
"longest run of consecutive gaps in a prescribed residue class, sieve of Eratosthenes",
"maximal deleted chain when a prime is added to a sieve, alternating residue constraint",
"flank envelope of a pattern of consecutive gaps", "Jacobsthal function depth of merge".
Nearest relatives inside the project: `two-teeth-kill-spacing.md` (T1-T5, on which both
theorems rest), `per-j-window-analogues.md` (the family this note's even half completes),
`kleene-generator.md` (`Q*_J` is layer `J-2` of the star).
