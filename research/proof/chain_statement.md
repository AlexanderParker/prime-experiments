# The chain statement: Q*_J(M) <= F(M) + q' for every M and every J >= 3

Prover B, round 32 (2026-09-04).  Vocabulary as fixed by
`docs/proof-search/alignment-rules.md` section 0.  Every claim below carries its status
(PROVED / REDUCED TO / MEASURED) and, where a computation is involved, the gate command and
the output line.  Nothing here calls the budget inequality a law.

## Pre-registration (written before any computation)

Read the record first; the computations were designed after reading and BEFORE running.
Predictions, to be scored at the end of this file:

P1. The chain statement fails on the tooth-counterfactual family (same gears, mirror-symmetric
    teeth at +-v_q, v_q ranging over 1..(q-1)/2) at every level m11, m13, m17, m19, with the
    incoming tooth FREE (v_{q'} in 1..(q'-1)/2).  Expected rate: below 1% of rows (the
    record's budget-inequality failure rate is 0.00-0.56%; the depth-2 half fails at one
    member only, so nearly every budget failure should be a CHAIN failure).
P2. It also fails with the incoming tooth PINNED to v_{q'} = round(q'/6) (so that
    a = 2u', 3a = q' -+ 1 hold exactly) at m17 and m19 - i.e. the letters' arithmetic
    alone does not carry the statement; the OLD gears' teeth enter.  At m11/m13 pinned I
    expect zero or near-zero failures (the increment-law pinned rates are 0/0/1.1/6.5%).
P3. Among chain violators the pair statement F_2 <= F + q' HOLDS at (nearly) all of them:
    the chain statement is not a consequence of the pair statement plus the family-invariant
    ingredients (T1-T3, peel, middle-sum, same-tooth, mirror, attainment, gaps <= F).
P4. Violators occur in BOTH the literal and the padded case, at J = 3 and J = 4 at least.
    The maximum excess Q*_J - F - q' over the family grows with the machine.
P5. The real machine (teeth at round(q/6) for every gear) satisfies the chain statement at
    m11..m23 with the recorded per-J table reproduced exactly:
    Q*_3(11) = 8; Q*_3(13) = 18; Q*_3(17) = 25; Q*_3(19) = 33, Q*_4(19) = 34;
    Q*_3(23) = 43; and max(F_2, max_J Q*_J) = F(M+q') = 11, 18, 25, 34, 43 by direct sieve.
P6. Real old gears with a FREE incoming tooth (the real M, a wrong incoming letter pair):
    chain violations exist at some (M, v') - the proof must use the incoming tooth too.
P7. Par trading along the maximising chain (Q*_J - Q*_{J-1} = -eps) on the family is NOT
    confined to [-4, +4]: |eps| reaches at least s_min at some member at m17 or m19.
P8. On the real machines m11..m37 (r30 counted census), the smallest per-word slack
    F + q' - span(w) - Phi(w) over realised legal words is attained by a length-1 or
    length-2 word, is non-negative everywhere, and is <= 4 at at least one machine.

What would count as a proof landing: any of the four routes producing
Q*_J <= F + q' (+ c) from ingredients that hold at every machine.  What would count as the
obstruction being exact: a family member satisfying every listed ingredient with
Q*_J > F + q'.

---

## 1. The statement, exactly, and its split

Machine `M = {5..p}`, next prime `q'`, `u' = round(q'/6)` (so `6u' = q' -+ 1`), letters
`a = 2u'`, `b = q' - a` (`a + b = q'`, `3a = q' -+ 1`, `a < b`).  A gap value `v` is a legal
letter iff `v mod q' in {0, a, b}`; `0` is padded, `a`/`b` literal.  A legal word is a run of
consecutive gaps of `M`, all legal letters, whose nonzero classes strictly alternate (T3, padded
letters transparent).  A word-legal `J`-run is `J` consecutive gaps `(g_L, w_1..w_{J-2}, g_R)`
whose `J-2` middles form a legal word; `Q*_J(M)` is its maximal span, `-inf` if none.

> **CHAIN STATEMENT.**  `Q*_J(M) <= F(M) + q'` for every `M` and every `J >= 3`.

With the pair statement `F_2(M) <= F(M) + q'` it is the budget inequality exactly, by the
attainment identity `max(F_2(M), max_{J >= 3} Q*_J(M)) = F(M + q')` (R68, both directions).

**The split.**  A word-legal `J`-run is LITERAL if every middle is `a` or `b` mod `q'`, PADDED if
some middle is `0` mod `q'`.

- Literal: the middles alternate between the classes `a` and `b`; with `k = floor((J-2)/2)` the
  middle sum `S` is `>= k q'` (`J` even) or `>= k q' + a` (`J` odd) (middle-sum lemma, Theorem A).
  The statement needs `g_L + g_R <= F + q' - S`.  For the three smallest literal words:
  `(a)`: `g_L + g_R <= F + b`;  `(a,b)` or `(b,a)`: `g_L + g_R <= F`;  `(a,b,a)`: `g_L + g_R <= F - a`.
- Padded: some middle is a positive multiple of `q'`, so the run contains an old gap `>= q'` and
  `q' <= F(M)` is necessary (onset gate).  For the smallest padded word `(q')`:
  `g_L + g_R <= F`; for `(j q')`: `g_L + g_R <= F - (j-1) q'`.

In both cases the statement is a bound on the FLANKS of an occurrence of a legal word - the
flank envelope `Phi(w) = max over occurrences of (g_L + g_R)`:

> **Lemma 1 (exact restatement).  PROVED.**  The chain statement at `(M, q')` holds iff
> `Phi(w) <= F(M) + q' - span(w)` for every realised legal word `w` of `M`; equivalently iff
> every gap of `M + q'` produced by deleting two or more consecutive openings of `M` has
> length `<= F(M) + q'`.
>
> *Proof.*  `Q*_J = max over realised legal words w of length J-2 of (span(w) + Phi(w))` is the
> word reduction R89 (the middles of a word-legal `J`-run are a realised legal word of length
> `J-2`; any occurrence of such a word with its two flanking gaps is a word-legal `J`-run).  A
> gap of `M + q'` with `>= 2` deletions is a word-legal `J`-run with `J >= 3` by T2 + T3
> (necessity), and every word-legal `J`-run is a stretch inside a gap of `M + q'` at some
> translate (attainment, CRT).  []

## 2. The attack, as numbered lemmas

### Lemma 2 (the floors and the onset).  PROVED (on record: T2-T4, R90, onset gate).

Literal `J >= 4`: `x_3 - x_1 = w_1 + w_2 = 0 mod q'` is a 2-run of `M`, so `q' <= F_2(M)`.
Padded any `J`: `q' <= F(M)`.  Literal `J = 3`: `a <= F(M)`.  In the family below these hold
at every violator, so they are floors, not obstructions.

### Lemma 3 (what the family-invariant ingredients give).  PROVED, and INSUFFICIENT.

Call `I` the ingredient set: every gap `<= F`; every two adjacent gaps sum `<= F_2`; T1-T3;
the class minima `a, b, q'`; the peel bound; the middle-sum and same-tooth lemmas; the mirror;
attainment; the realisability CSP.  From `I` alone:

    Q*_3 <= F_2 + min(g_L, g_R) at the argmax <= F_2 + F,      (peel; also <= 2F_2 - a)
    Q*_J <= (J/2) F_2                     (J even, pairing the J gaps into J/2 adjacent pairs)
    Q*_J <= ((J-1)/2) F_2 + F             (J odd)

and nothing sharper: these are attained by abstract gap sequences satisfying `I`.  Against the
target `F + q'` every one of them is short by at least `F_2 - q' + min flank` at `J = 3` and by
`2F_2 - F - q'` at `J = 4` (the 2F wall).  That no cleverer combination of `I` closes the gap is
not a judgment; it is Lemma 4.

### Lemma 4 (the obstruction is exact).  EXACT, exhaustive on the family.

Every member of `I` holds at every machine of the tooth-counterfactual family (same gears,
teeth at `+-v_q`, `v_q in 1..(q-1)/2`; `I` is proved from CRT and two symmetric teeth).  Yet:

    level  incoming tooth  rows    chain violators   pair violators   pair holds at violators
    m11    free (6)         180      1  (0.56%)        0                1 of 1
    m11    pinned            30      0                 0                -
    m13    free (8)        1440      1  (0.07%)        0                1 of 1
    m13    pinned           180      0                 0                -
    m17    free (9)      12960     36  (0.28%)        0               36 of 36
    m17    pinned          1440      3  (0.21%)        0                3 of 3
    m19    free (11)   142560    193  (0.14%)       11              192 of 193
    m19    pinned        12960     46  (0.35%)        1               46 of 46

Max excess `Q*_J - F - q'`: 1, 1, 6, 11 (free) and 0, 0, 3, 9 (pinned) at m11..m19 - growing.
Violating cells at m19 (free) by depth `J = 3..7`: 15, 118, 82, 21, 1; literal 147 rows, padded
54 rows; the deepest is the literal `(a,b,a,b,a)` at `J = 7`.  `L` reaches 5 on the family
(the record's member) against 2 on the real machine.

**Where the real teeth enter, sharpened.**  Call an old gear DEGENERATE if `v_q = (q-1)/2`
(its two teeth adjacent, `2v = -1 mod q`; the antipode is struck) - a configuration the real
teeth exclude (`AnchorChain.neighbour_of_hit`).  Classifying the violators
(`chain_viol_classify_r32.py`):

    level   violators   with a degenerate old gear   non-degenerate, free tooth   non-degenerate, PINNED
    m11         1            1                           0                          0 of    8 rows
    m13         1            0                           1                          0 of   40 rows
    m17        36           33                           3                          0 of  280 rows
    m19       193          141                          50 (9 more have a = 1)      0 of 2240 rows

So: (a) `I` alone is refuted by non-degenerate members (m13 `(1,2,3,1)`, `v' = 7`); (b) `I` +
`3a = q' -+ 1` is refuted (pinned violators at m17, m19), but every pinned violator carries an
adjacent-teeth gear; (c) `I` + "no gear has adjacent teeth" is refuted (the 50 free
non-degenerate violators at m19); (d) `I` + BOTH - no adjacent teeth AND `3a = q' -+ 1` - has
NO violator found in 2,568 rows to m19 (an m23 sample is reported in section 7).  The pair
statement, by contrast, fails at exactly one pinned non-degenerate member of m19 (the record's
wrap-pair member `(1,1,4,3,5,2)`).  This is a measurement, not a law: the smallest ingredient set
with no known counterexample to the chain statement is
`I + {2u_q != +-1 mod q for every gear} + {3a = q' -+ 1}`; both extra facts are consequences of
`6u = +-1`, and both are already in the kernel.

Witnesses (every ingredient of `I` checkable by hand on the printed run):

- m13, teeth `(v_5, v_7, v_11, v_13) = (1, 2, 3, 1)`, `q' = 17`, `v' = 7`, `a = 3`, `b = 14`:
  `F = 14`, `F_2 = 17`, budget 31; the literal `J = 5` run `(7) + (3, 14, 3) + (5)` has span 32.
  No gear of this member has adjacent teeth, the antipode is open, `a >= 2`; the pair statement
  holds (17 <= 31).  The middle letter `b = 14 = F(M)`: the violating run is the odd-`J`
  palindrome `(a, b, a)` with `b` the old record - the shape the record already named for the
  counterfactual increment violators.
- m17, teeth `(1, 2, 1, 1, 1)`, `q' = 19`, `v' = 7`, `a = 5`: `F = 15`, `F_2 = 24`, budget 34;
  literal `J = 4`: `(6) + (14, 5) + (10)` = 35.  Non-degenerate; pair holds.
- m17 PINNED (`v' = 3`, `a = 6`, `b = 13`, `3a = 18 = q' - 1`): teeth `(2, 3, 3, 2, 4)`, `F = 16`,
  `F_2 = 23`, budget 35; literal `J = 5`: `(9) + (6, 13, 6) + (4)` = 38, excess 3.

So no proof of the chain statement can proceed from `I`, nor from `I` plus the pair statement,
nor from `I` plus the incoming letters' arithmetic `3a = q' -+ 1` alone.  Gate:
`uv run python research/proof/chain_family_r32.py family 11 13 --direct` (G2: attainment by
direct sieve 180/180 and 1440/1440), `... family 17 19 --procs 4`,
`... realfree`, `uv run python research/proof/chain_viol_classify_r32.py 11 13 17 19`.

### Lemma 5 (the real old gears carry the statement, whatever the incoming letters).  EXACT, m11..m23.

With the REAL old teeth and the incoming tooth `v'` ranging over all of `1..(q'-1)/2` (so the
letters are `+-2v'`, generally with `3a != q' -+ 1`), the chain statement holds at every
`(M, v')`, m11..m23: worst margins `F + q' - max_J Q*_J` are 4 (m11, `a = 6`), 5 (m13, `a = 5`),
9 (m17), 9 (m19, `a = 2`, where `L = 3`), 13 (m23).  (`chain_family_r32.py realfree`.)  So at
these levels the statement is a property of the old gears' actual teeth `6u_q = +-1`, not of the
incoming tooth - the opposite of the increment law, where the record found the incoming tooth
carrying most of it.  Measured, five machines; not a law.

### Lemma 6 (the real machine, and where it is tight).  EXACT, m11..m37.

Direct recomputation on one lower period reproduces the recorded `Q*_J` table cell for cell at
m11..m23 (gate G1) and the attainment identity by direct sieve of `M+q'` (gate G2, m11..m19).
Per-word slack `F + q' - span(w) - Phi(w)` from the r30 counted census
(`uv run python research/proof/chain_slack_r32.py`, attainment gate OK at all eight machines):

    M     budget   smallest slack   word          J    Phi   argmax flanks   literal/padded
    m11     20        12            (4)           3     4    (3,1)           literal
    m13     28        10            (6), (11)     3    12,7                  literal
    m17     37        12            (13)          3    12    (7,5)           literal
    m19     48        14            (8,15)        4    11    (4,7)           literal
    m23     63        20            (10)          3    33    (23,10)         literal
    m29     74        16            (10)          3    48    (30,18)         literal
    m31     95         7            (12,37)       4    39    (11,28)         PADDED
    m37    129        38            (14,41)       4    36    (21,15)         PADDED

The minimum slack in the corpus is 7, at the padded 2-word `(12,37)` of m31 (the `F_3`-wall
event); the literal cells never go below 10.  The binding word is always of length 1 or 2.

## 3. The four routes

**(i) Par trading as a theorem, `Phi(u.x) <= Phi(u) - x + c`.  REFUTED as a consequence of `I`.**
Iterating it from `Phi(empty) = F_2` gives `Q*_J <= F_2 + (J-2) c`, which with `L <= 2F(M+q')/q' + 1`
would close the chain given the pair statement and a small `c`.  But `eps = Phi(u) - Phi(v) - x`
is not bounded by anything in `I`: on the family the chain step `Q*_J - Q*_{J-1} = -eps` along
maximising chains ranges over `[-13, +10]` at m17 (free) and `[-11, +7]` at m17 pinned, against
`s_min = 6`; on the real machine it is `-20` once (m31, padded).  The decomposition lemma
`eps = d - g_out` says why: `eps` is the difference of two flank order statistics, each of size
up to `F`, and `I` bounds neither the difference nor the flank `g_out` below `F`.  Par trading
is the statement to be proved, restated per letter.

**(ii) The literal case with the pair statement as a black box.  REDUCED, not closed.**
What the pair statement supplies at a literal `J`-run: `g_L <= F + q' - w_1` and
`g_R <= F + q' - w_{J-2}`, hence `g_L + g_R <= 2F + 2q' - w_1 - w_{J-2}`; at `(a,b)` that is
`2F + q'` against the needed `F`.  Exactly what must be added is
`Phi(w) <= F + q' - span(w)` for the literal words - i.e. the literal chain statement itself;
the pair statement gives no more than "gaps `<= F`" gives once `F > q'` (m19 on).  And on the
family the pair statement holds at EVERY chain violator (Lemma 4), so the implication
"pair => chain" has no proof from `I`.  The `a`-cell is the binding literal cell at m11..m29
(slacks 12, 10, 14, 15, 20, 16 for the letter `a`; `(a,b)` is binding at m19 with 14).

**(iii) The padded case by the record law one level down.  REDUCED, not closed.**
A padded middle `w = j q'` is an old gap `>= q'`.  It is a merge at the previous step only if
`w > F(M^-)`, and `q' > F(M^-)` fails from m29 on (`F(23) = 34 < 37`, `F(29) = 43 > 37`);
the recursion has no base and, where it applies, it lower-bounds `w` by old gaps rather than
upper-bounding the flanks.  The padded `J = 3` statement is `g_L + g_R <= F - (j-1) q'` around a
gap `j q'`: "a gap of size `q'` has small neighbours", the isolation of large gaps, on record as
unexplained.  It is the tight case: slack 10 at `(37)` and 7 at `(12,37)` at m31.  On the
family padded violators exist (m17 free: 4 of 36; shapes `(0b)`, `(a0)` at `J = 4`).

**(iv) Survivor-generator contraction across layers.  REFUTED.**
`Q*_J` is layer `J-2` of the max-plus star, but the layers are not monotone in either direction:
real machine `Q*_3, Q*_4 = 33, 34` (m19), `58, 55, 55` (m29), `85, 88, 68` (m31); family steps
in `[-13, +10]`.  A layer-to-layer map that appends a letter `x` and swaps one flank has
weight `x + g_new - g_old`, unbounded in sign under `I`; there is no contraction constant, and
the observed near-constancy `Delta_J in [-3, +4]` on the real machine is a cancellation
(`d = 27` against `g_out = 28` at m31), not an algebraic property of the operator.

## 4. Verdict: no proof lands; the obstruction and the smallest unproved statement

No proof of the chain statement, of its literal case, or of any bound `Q*_J <= F + q' + c` with
`c` independent of `M`, is obtainable from `I`, from `I` + the pair statement, or from `I` + the
incoming-letter arithmetic - Lemma 4 exhibits machines satisfying all of these where the
statement fails with excess growing with the level (1, 1, 6, 11 at m11..m19 free; 0, 0, 3, 9
pinned).  Any proof must use the OLD gears' teeth `6u_q = +-1 mod q` (Lemma 5 says those
alone carry it at m11..m23, for every incoming letter pair).  The ingredients on record that use
those teeth are: teeth never adjacent (`AnchorChain.neighbour_of_hit`), the antipode open
(`Mirror.antipode_open`), and the corridor `E_35` itself (the specific exposed set, not merely
some exposed set).  Adjacency accounts for EVERY pinned violator (3 of 3 at m17, 46 of 46 at
m19) and for most free ones (33 of 36, 141 of 193), and the sub-family with neither adjacent
teeth nor a wrong incoming letter has no violator in 2,568 rows - but the free non-degenerate
violators (1 at m13, 3 at m17, 50 at m19, e.g. m19 teeth `(1,1,1,2,1,5)`, `v' = 5`, `a = 10`:
`(5) + (13,10,13,10) + (2) = 53 > 50`, a literal `J = 6` alternation) show that the two facts
are needed TOGETHER, and no mechanism on record combines them: the corridor's size-blindness
(escape distance 1) is on record, and nothing on record converts `6u_q = +-1` into a flank
bound.  A proof, if there is one, lives in the interaction between the incoming letters
`a = (q' -+ 1)/3` and the old gears' non-adjacent teeth - i.e. in the arithmetic of the real
tooth `3^{-1}` at every level at once.

**The smallest unproved statement**, in the exact form the chain statement takes at its binding
cells (Lemma 1):

> For every machine `M` with next prime `q'`, and every occurrence of the letter `a = 2u'` as a
> gap of `M`, the two flanking gaps sum to at most `F(M) + b`;  every occurrence of the padded
> letter `q'` has flanks summing to at most `F(M)`;  every occurrence of `(a,b)` has flanks
> summing to at most `F(M)`.

The first is the binding literal cell at m11..m29 (slack 12, 10, 14, 15, 20, 16 - the letter `a`
alone, `Phi(a) = 4, 12, 17, 25, 33, 48`); the second and third are the padded and even-`J`
binding cells (m31: `Phi(37) = 48 <= 58`, slack 10; `Phi(12,37) = 39`, slack 7).  Each is a
statement about the flank order statistic of one gap value, and each fails on a counterfactual
machine satisfying `I`.

What to kernel-check if anything: Lemma 1 (the equivalence) is a two-line consequence of
`WordLegal.chain_iff_word` and the attainment theorem, and is not yet a named declaration.

## 5. Scoring the pre-registration

P1 CONFIRMED (0.56 / 0.07 / 0.28 / 0.14 % at m11..m19 free).  P2 CONFIRMED at m17 (3 of
1440) and m19 (46 of 12960), 0 at m11/m13 as predicted - with the unpredicted rider that every
pinned violator has an adjacent-teeth gear.  P3 CONFIRMED (pair holds at 1 / 1 / 36 / 192 of
1 / 1 / 36 / 193 violators).  P4 CONFIRMED (literal `J = 3..7` and padded `J = 4, 5, 6`; excess
1, 1, 6, 11 growing).  P5 CONFIRMED (G1 5/5, G2 4/4 real, 1620/1620 family).  P6 REFUTED: the
real old gears satisfy the chain statement for EVERY incoming tooth at m11..m23 (worst margin 4).
P7 CONFIRMED (`|eps|` up to 13 at m17 against `s_min = 6`, 21 at m19 against 8).  P8 half: the
smallest slack is a length-1/2 word and non-negative everywhere, but the minimum is 7 (m31),
never `<= 4`.

## 7. The non-degenerate pinned sub-family one level up (m23 sample)

A full sweep of the 22,400 non-degenerate pinned members at m23 costs ~3 core-hours (period
223M each) and was out of budget; a fixed-seed sample was run instead
(`uv run python research/proof/chain_family_r32.py ndpinned 23 600 4`, seed 32, 215 s):

    600 members, q' = 29, v' = 5 (a = 10): chain violators 0; pair violators 0;
    margin F + q' - max_J Q*_J: min 2, median 16, max 27; L = 1 / 2 / 3 at 19 / 506 / 75 members.

So the sub-family "no adjacent teeth and `3a = q' -+ 1`" has no chain violator found in
2,568 exhaustive rows to m19 plus 600 sampled rows at m23 - with a margin of 2 at one m23
member, so the statement is not comfortably true there either; nothing here is a law.  The
real machine's margins at the same levels are 12, 10, 12, 14, 20.

## 6. Files

- `research/proof/chain_family_r32.py` - the vehicle (real gates, family, real-M/free-tooth).
- `research/proof/chain_viol_classify_r32.py` - violator classification.
- `research/proof/chain_slack_r32.py` - per-word slack on the r30 counted census.
- logs `research/proof/chain_family_r32_m11_m13.log`, `chain_family_r32_m17_m19.log`,
  `chain_family_r32_ndpinned_m23.log`; violators
  `research/proof/chain_family_r32_viol_m{11,13,17,19}.json`; the m19/m23 non-degenerate
  pinned rows `chain_family_r32_ndpinned_m{19,23}.json`.
- Shared-log block: `docs/proof-search/agents-shared.md`, "## Prover B (chain statement)".
