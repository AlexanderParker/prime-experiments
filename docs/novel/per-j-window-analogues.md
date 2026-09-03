# The per-J window analogues: Q*_J, the middle-sum lemma, and the J-parity of palindromes

Constructor, round 28.  Status per statement below; nothing here is announced as new until
section 6 carries a dated prior-art verdict.

---

## 1. WHAT IT IS

The project's live obligation (D) at a step `M -> M + q'` says the new machine's record gap
does not exceed `F(M) + q'`.  The manager's round-27 reduction turned the depth-3 part of
that into ONE inequality about the old machine alone (the *triple inequality*).  This
document states, proves what can be proved, and measures exactly, the **whole family of which
that inequality is the J = 3 member** - and shows the family is FINITE and terminates by
J = 5 at every machine on record below m47.

**Plain language.** When a new gear is added, some runs of consecutive openings get deleted
and the gaps around them merge into one long gap.  The deleted openings sit at spacings the
new gear's two teeth allow, and those spacings must alternate between the two teeth.  A
*window* of J consecutive old gaps is what merges when J-1 openings die.  The question
"how long can the merged gap be?" is one question per J, and the alternation makes the
middle of the window almost completely determined - only the two end gaps are free.  So the
J-th question has only two free coordinates, however deep J is, and the deep questions turn
out to be the EASY ones: past J = 5 the answer is that no such window exists at all.

**Precise form.**  Let `M = {5..y}`, `q' = nextprime(y)`, `u' = round(q'/6)`,
`a = 2u'`, `b = q' - 2u'` (so `a + b = q'`, `a < b`, `s_min = a`).

> **Definition.**  A **word-legal J-window** of `M` is a run of J consecutive gaps
> `(g_L, w_1, ..., w_{J-2}, g_R)` such that every middle `w_i` is `0` or `±2c` mod `q'`
> (T2) and the nonzero classes strictly alternate, padded middles (`≡ 0`) being transparent
> (T3).  Write
>
>     Q*_J(M; q') = max span of a word-legal J-window        (Q*_2 = F_2(M) identically)
>     Delta_J(M)  = Q*_J - F_2(M)
>     Phi_J(M)    = max flank sum g_L + g_R over word-legal J-windows.
>
> The **per-J analogue** of the triple inequality is `Delta_J <= s_min(q')`, one statement
> per J; and by the merge law `max_J Delta_J = F(M+q') - F_2(M)`, so the whole increment law
> is exactly this finite list.

### 1.1 THEOREM A (the middle-sum lemma) - PROVED from T1-T3

In a **literal** word-legal J-window the `J-2` middles alternate between class `a` and class
`b`, so the two class-counts differ by at most one; every class-`a` middle is `>= a` and
every class-`b` middle is `>= b`, with `a + b = q'`.  Hence, with `k = floor((J-2)/2)`,

        middle sum  >=  k*q'                (J even)
        middle sum  >=  k*q' + a            (J odd)

with equality iff every middle takes its smallest positive representative.

**Consequence.**  A literal J-window's span exceeds its flank sum by an amount that grows by
`q'` every two levels of J.  The per-J analogue therefore forces the flank envelope to
collapse at that rate:

        Phi_J  <=  F_2(M) + s_min(q') - m_min(J),

so at J = 5 the flanks may sum to at most `F_2 - q'`, and at J = 6 to at most
`F_2 + a - 2q'`.  That is par trading (R30) in exact form, and it is why the deep layers are
the cheap ones.

### 1.2 THEOREM B (the J-parity of palindromes) - PROVED from T3

For **J even** a literal word-legal J-window is **never a palindrome**: the middle class word
has even length and strictly alternates, so reversing it exchanges the two classes, and
`a != b` at every gear.  For **J odd** the middle class word is forced palindromic
(`c, -c, c, ...`), so the window is a palindrome exactly when its middle *values* are
symmetric and its two flanks are equal.

### 1.3 PROPOSITION C (reversal-uniqueness) - PROVED, given Lateral's mirror theorem

The set of realised word-legal J-windows is closed under reversal (occurrence counts are
reversal-invariant by the mirror involution `k -> -k`; T2 is value-wise and T3's alternation
is reversal-invariant), and span is reversal-invariant.  Hence **the set of maximising WORDS
is reversal-closed; if it is a single word, that word is a palindrome.**

### 1.4 THEOREM D (the peel bound) - PROVED, hypothesis-free

Deleting either flank of a word-legal J-window leaves a word-legal (J-1)-window, so

        Q*_J  <=  Q*_{J-1} + min(g_L, g_R) at the argmax,

the exact J-analogue of R78's free reduction (which is the case J = 3, `Q*_2 = F_2`).

### 1.5 THE MEASURED TABLE - SCRIPT-VERIFIED, exact, gated

`research/perj_window.py` (full-period scans at m11..m23, Mechanic's exact 4-tuple censuses
at m29/31/37) and `research/perj_scanfree.py` (CRT descent, no period anywhere).  Every cell
is exact; `EMPTY` means *certified empty*, not *no data*.

     M   q'  s_min  F_2 | Delta_3 Delta_4 Delta_5 Delta_6 | J_max  A_kill+1  maximiser at J_max
    11   13     4    11 |    -3    EMPTY   EMPTY   EMPTY  |   3       3      -
    13   17     6    16 |    +2    EMPTY   EMPTY   EMPTY  |   3       3      -
    17   19     6    25 |    +0    EMPTY   EMPTY   EMPTY  |   3       3      -
    19   23     8    31 |    +2     +3     EMPTY   EMPTY  |   4       4      (4,8,15,7)
    23   29    10    39 |    +4    EMPTY   EMPTY   EMPTY  |   3       3      -
    29   31    10    55 |    +3     +0      +0     EMPTY  |   5       5      (7,10,21,10,7)  PAL
    31   37    12    68 |    +2     +3      +0     EMPTY  |   5       5      (3,25,12,25,3)  PAL
    37   41    14    90 |    +0    (pad+1) EMPTY   EMPTY  |   4       4      (15,41,14,21)
    41   43    14   103 | <=116  <=100    EMPTY   EMPTY  |   4       4      -

(Delta columns are LITERAL middles; the padded column is separated because it is the whole of
the project's one failing step.  `Delta_3`/`Delta_4` including padded middles are +17/+20 at
m31 and +1 at m37 J=4; everywhere else the padded value is below the literal one.)

Three facts the table establishes.

* **`J_max(M) = A_kill(M) + 1` at all eight censused machines**, by a vehicle independent of
  the one that measured `A_kill`.  The per-J program TERMINATES, and below m47 it terminates
  at J = 5.
* **`Delta_J` is bounded by a small constant uniformly in BOTH M and J**: every literal cell
  lies in `[-3, +4]`, against `s_min` growing linearly in `q'`.  The excess over `F_2` does
  not grow with depth - it SHRINKS (`Delta_5 = 0` exactly, at both machines where J = 5 is
  non-empty).
* At m41 the two live cells are bounds, not exact values: `Q*_3(41) <= 116` (R80, round 27,
  superset sweep, padded half - and independently reproduced this round by the enumerative
  vehicle, whose span-117 level is fully decided: 58 candidates, 0 realised, 0 undecided)
  and `Q*_4(41) <= 100` (this round, enumerative CRT descent from
  the `F_4(41) = 118` ceiling, 678 candidates, 0 undecided).  Both are far under the budget
  `F_2 + s_min = 117`, so the per-J analogue is CERTIFIED at every J at 41 -> 43.
* **`A_kill(41) = 3` exactly** (new).  `Q*_5(41)` is certified EMPTY scan-free, so
  `A_kill(41) <= 3`; R45's padded 2-words give `>= 3`.  This closes the project's open item
  O7 without needing `F_3(41)`.

### 1.6 THE PALINDROME DICHOTOMY - MEASURED, and it splits exactly on Theorem B

At every cell the maximising word is unique up to reversal (exactly one canonical class
attains the maximum).  Then:

    J = 3  (11 cells)   maximiser is a reversal PAIR - never a palindrome
    J = 4  ( 4 cells)   maximiser is a reversal PAIR - never a palindrome (Theorem B forbids
                        it outright for literal windows)
    J = 5  ( 2 cells)   maximiser is UNIQUE and SELF-REVERSE - a PALINDROME, both times

So the manager's round-28 "extremal implies palindromic" step is **true exactly at the deep
odd layer** - which is the layer that decides `A_kill` and the layer at which `Delta_J = 0` -
and **false at J = 3 and J = 4**, where Theorem B explains half of it (even J cannot be
palindromic at all) and measurement the other half.

### 1.7 ROUND 29 - THE DEPTH QUANTIFIER IS CLOSED, AND THE `EMPTY` CELLS ARE FREE

Section 1.5's third bullet recorded `J_max(M) = A_kill(M) + 1` as MEASURED at eight
machines by two independent vehicles.  It is a **theorem**:

> `Q*_J(M; q') > -inf` iff `L(M) >= J - 2`, where `L(M)` is the length of the longest
> *realised* word of legal letters with alternating nonzero classes.  Hence
> `J_max(M) = L(M) + 2` and `A_kill(M -> q') = L(M) + 1`.

Proof and attribution in `docs/novel/even-j-mechanism.md` section 1.1 (the forward half is
Mechanic's round-28 index observation; the converse and the `L` formulation are round 29).
Consequences for this document:

* every `EMPTY` cell of the `Delta_J` table is now a **one-line dictionary fact**, not a CRT
  sweep: `J = 6` is empty at every machine m11..m43 because `L <= 3` everywhere;
* the depth cap of a NEW machine costs the decision of its legal words of length `L+1` -
  at m43 that is 31 candidates of which phase saturation refutes 23 for free;
* `L(M)` at m11..m37 is `1, 1, 1, 2, 1, 3, 3, 2`, every value CERTIFIED (the next length up
  has no realised legal word), reproducing the recorded `J_max` and `A_kill` rows 16/16.

The literal `Delta_J` table is reproduced by a third vehicle (`research/evenj_r29.py`, a
word-indexed flank census over full-period scans at m11..m23 and Mechanic's exact censuses
at m29/m31/m37; 21 of the 22 recorded `Q*_J` cells reproduced, 0 mismatches), which also
exhibits the m31 **literal** `J = 4` maximiser for the first time: `(6, 25, 12, 28)`, span
71, `Phi = 34`, middle sum `37 = q'` exactly.

The even-`J` half of the family - the half the palindrome route of section 1.6 provably
cannot reach - is worked out in the companion note.

---

## 2. WHY IT MIGHT BE NOVEL

* The object `Q*_J` is the J-th layer of the project's own Kleene generator (R46), so the
  family is not a classical one; but Theorem A is a statement about *Jacobsthal-type extremal
  gap words under a two-tooth alternation constraint* and I know of no analogue.  The
  classical shadow is the trivial one - "a sum of j consecutive gaps is at most F_j" - which
  carries no alternation and therefore no `q'`-per-two-levels growth.
* Theorem B is an elementary parity observation, but the *use* is not: it says the mirror
  route (self-mirror windows are antipode-pinned and never qualifying) can only ever bite at
  ODD depth, which is a structural restriction on a proof strategy, not on an object.
* The measured statement "`Delta_J` is a bounded constant uniformly in J" is the sharp form of
  the increment law and is strictly stronger than the form the manager pre-registered
  (`Delta_J <= s_min`, a linearly growing budget).

---

## 3. PROOF / STATUS

| statement | status | pointer |
|---|---|---|
| Theorem A (middle-sum lemma) | **PROVED** (T1-T3, three lines) | this file, section 1.1 |
| Theorem B (J-parity of palindromes) | **PROVED** (T3 + `a != b`) | section 1.2 |
| Proposition C (reversal-uniqueness) | **PROVED** given Lateral's mirror theorem | section 1.3 |
| Theorem D (peel bound) | **PROVED**, hypothesis-free | section 1.4 |
| the Delta_J table | **SCRIPT-VERIFIED, exact** | `research/perj_window.py`, `research/perj_scanfree.py`; logs `research/data/r28/perj_window.log`, `perj_m41.log` |
| `J_max = A_kill + 1`, 8/8 | **SCRIPT-VERIFIED** (two independent vehicles) | as above vs R45 |
| `A_kill(41) = 3` | **SCRIPT-VERIFIED, scan-free** | `perj_scanfree.py --y 41 --J 5 --floor 0` |
| the palindrome dichotomy | **MEASURED** (13 cells, exhaustive per cell) | as above |
| `Delta_J` bounded by a constant uniformly in J | **MEASURED** (13 cells) - NOT proved | - |

GATES, all green: `perj_window.py` reproduces R68's independently computed exact `Q*_J` table
at every cell it covers (11, 8 / 16, 18 / 25, 25 / 31, 33, 34 / 39, 43 / 55, 58, 55, 55 /
68, 85, 88, 68 / 90, 90, 91) and recovers `F(M)` and `F_2(M)` from every data source before
comparing anything.  `perj_scanfree.py` - a completely different vehicle, CRT arithmetic from
the gear list with no period - reproduces `Q*_4(19) = 34`, `Q*_4(29) = 55`, `Q*_4(31) = 88`,
`Q*_4(37) = 91`, `Q*_5(29) = 55`, `Q*_5(31) = 68` and the EMPTINESS of `Q*_4(23)`,
`Q*_5(19)`, `Q*_5(37)`, value and witness.

---

## 4. IMPLICATIONS

* **The finite lemma list is now explicit and short.**  The whole depth->=3 half of the
  increment law at a step is:
  `L3: Delta_3 <= s_min`, `L4: Delta_4 <= s_min`, `L5: Delta_5 <= s_min`,
  and `L_J` for `J > A_kill(M)+1` is VACUOUS by an emptiness certificate that costs no census.
  Below m47 only L3, L4, L5 are non-vacuous, and L5 is `0 <= s_min`.
* **The deep layers are the cheap end of the CRT oracle**, not the expensive one: a 6- or
  7-point pattern has small gear domains, so a J = 5 or J = 6 sweep at m41 costs seconds to
  minutes where an arity-2 refutation at the same machine costs tens of seconds.  Every
  emptiness certificate in the table above was produced this way.
* **The free reduction cannot do J >= 4.**  Theorem D peels one flank, and the peeled flanks
  at the measured argmaxes are 2..15 - so `Delta_J <= Delta_{J-1} + minflank` never lands
  inside `s_min`.  The J >= 4 obligation is therefore genuinely new content, and it is where
  the residual work sits (it is small: `Delta_4 <= 3` at every measured step).
* **For the manager's palindrome route:** it applies at odd J only (Theorem B), and at odd
  J >= 5 it is exactly right (measured 2/2).  Since J = 5 is the deepest non-empty layer at
  every machine below m47, "kill the self-mirror windows" would close the deepest layer of
  the finite list outright.

---

## 5. UNSOLVED QUESTIONS IT TOUCHES

* Requirement (D) / the tolerance route (R14, R26) - this is the depth->=3 half of its sole
  open input, reduced to three inequalities per step.
* Jacobsthal-type extremal problems for the two-dimensional (twin) sieve: `Q*_J` is a
  constrained `F_J`, and `Delta_J = O(1)` is an anti-clustering statement of a shape the
  literature states only asymptotically.
* Open item O7 of the project (exact `A_kill(41)`) - CLOSED here at 3.
* The counterfactual obstruction (manager, round 28): see the companion note in
  `docs/novel/uniform-order-bound.md` and section 6 of this file's round-28 append - the
  violating counterfactual's window is the odd-J palindromic shape with `b = F_old`, and
  `F(M) mod q'` is a legal letter at exactly one of the twelve corpus steps.

---

## 6. PRIOR-ART CHECK

**Not yet checked** (this lane has no web access).  Suggested search terms for the manager:
"Jacobsthal function consecutive gaps alternating residue constraint",
"maximal gap after adding a prime to a sieve, chain of deleted elements",
"palindromic extremal configurations in sieve gap words",
"Holt Rudd cycle recursion deleted run length".
The nearest known relatives inside the project are the merge law (PARTIAL OVERLAP with
Holt-Rudd) and the two-teeth kill spacing law T1-T5 (`docs/novel/two-teeth-kill-spacing.md`),
on which Theorems A and B rest.
