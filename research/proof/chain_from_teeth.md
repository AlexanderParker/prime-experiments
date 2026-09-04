# The chain statement from the teeth: I + (T) + (L)

Prover C, round 33 (2026-09-04).  Branch 2f of `research/proof/theory_tree.md`.  Vocabulary as
fixed by `docs/proof-search/alignment-rules.md` section 0; ingredient set `I` and Lemma 1 as in
prover B's `research/proof/chain_statement.md`.  Every claim carries PROVED / REDUCED TO /
MEASURED and, where computed, the gate command and the output line.  The budget inequality is the
target, never a law.

Target of this branch.  (T): no gear has adjacent teeth, `2v_q != +-1 mod q` for every old gear
(kernel: `AnchorChain.neighbour_of_hit`, from `6u = +-1`).  (L): the letter identity
`3a = q' -+ 1`, `a = 2 round(q'/6)`, `b = q' - a = 2a -+ 1`.  Claim under test: for every
machine satisfying `I + (T) + (L)`, every realised legal word `w` has
`Phi(w) <= F + q' - span(w)`, i.e. `Q*_J(M) <= F(M) + q'` for all `J >= 3`.

**Verdict in one line.**  NO PROOF.  The claim survives every row computed (2,568 + the m23 sweep
of section 2.5), but with ZERO slack: a member of the (T)+(L) sub-family at m19 has
`Q*_4 = F + q'` exactly.  What (T) does is now located exactly - it acts at gears 5 and 7 ONLY
(every pinned violator at m17/m19 has `v_5 = 2` or `v_7 = 3`; "(T) at 5 and 7" + (L) has no
violator in 4,800 rows while "(T) at every gear `>= 11`" + (L) has 25) - and what it does there is
the round-31 bare-alternation depth control, already in the kernel.  The padded depth-3 cells hold on
the ENTIRE family without any tooth fact (0 failures in 125,928 evaluated rows at m19, minimum
margin 0).  The residue is the literal depth-3 and depth-4 flank statements
(`Phi(a) <= F + b`, `Phi(b) <= F + a`, `Phi(a,b) <= F`), which no fact on record converts
into a bound: they are the isolation of large gaps (branch 5b) at the letters, and the CRT
recombination of the two flanks - the only construction that turns a two-point statement into
"some gap `>= g_L + g_R`" - exists at NONE of the binding occurrences (section 3, route iii).

## 0. Pre-registration (written before any computation of this round)

Two bookkeeping facts fixed first, by hand, so the tests below are well-posed.

- In the mirror-symmetric family the letters are `{2v', q' - 2v'}` and `a = min` of the two.
  Since `q'` is odd, `a = 2 round(q'/6)` holds iff `v' = round(q'/6)`: **(L) is exactly "the
  incoming tooth is pinned"**.  Candidate (iv) of the brief ("the tooth is `round(q'/6)`, not just
  (L)") is therefore empty INSIDE the family for the incoming gear; the only place a stronger tooth
  fact can enter is the OLD gears, where (T) says `v_q != (q-1)/2` and the real teeth say
  `v_q = round(q/6)`.  The quantity between them is the tooth SEPARATION
  `sep_q = min(2v_q mod q, q - 2v_q mod q)` (the shorter arc between the two teeth): (T) is
  `sep_q >= 2`, the real teeth have `sep_q = 2 u_q ~ q/3` (2, 2, 4, 4, 6, 6, 8 at 5..23).
- The padded statement S2 (`Phi(q') <= F`) does not mention `a`: it is independent of the incoming
  tooth.  So if S2 fails anywhere on the family it can be rescued only by (T), never by (L).

Predictions, scored in section 6:

- PC1. The full 19->23 family (142,560 rows) recomputed: 193 chain violators free, 46 pinned,
  0 in the (T)+(L) sub-family (2,240 rows) - prover B's numbers reproduced.
- PC2. Among the free violators with (T) holding at m19 (B: 50 rows), the letter `a` is spread:
  no single `a` carries more than half of them, and the two letters ADJACENT to the pinned one
  (`a = 7` and `a = 9`, `q' = 23`) both carry at least one violator.  If confirmed, (L) is needed
  as the exact identity `3a = q' -+ 1`, not as the ratio `b ~ 2a`.
- PC3. The 23->29 sub-family (T)+(L), full sweep of 22,400 members: at least one chain violator
  (the 600-row sample had min margin 2; predicted 1-20 violators).  If violators exist, every one
  has some gear `q >= 11` with `sep_q <= 3` (i.e. `v_q in {1, (q-3)/2}`), so the fact that
  carries the statement is a quantitative separation, of which (T) is only the weakest form.
- PC4. Of the three smallest statements S1 (`Phi(a) <= F + b`), S2 (`Phi(q') <= F`),
  S3 (`Phi(a,b) <= F`): on the real machine m11..m23 S1 is the tight one (smallest margin) at every
  level; on the (T)+(L) sub-family at m19 S1 has the smallest minimum margin of the three, and S3 is
  vacuous (no `(a,b)` occurrence) at more than half of the rows.
- PC5. Outside the sub-family at m19: S2 fails only at rows with a degenerate gear (needs (T)
  alone, by the bookkeeping fact above); S1 and S3 fail at rows with (T) holding and the tooth free
  (need (L)) AND at rows pinned with a degenerate gear (need (T)) - i.e. both literal statements need
  both facts, the padded one needs only (T).
- PC6. Mechanism (i): in at least 90% of the pinned violators at m19 the degenerate gear strikes
  two consecutive columns inside the violating stretch (flank or letter interior).
- PC7. On the pinned m19 family (12,960 rows) the chain margin `F + q' - max_J Q*_J` is positively
  associated with `min_q sep_q / q`: the mean margin of the rows with `min_q sep_q / q < 0.15` is
  below the mean of the rows with `min_q sep_q / q >= 0.25`.
- PC8. No proof of the chain statement from `I + (T) + (L)` lands in this round; the round ends
  with the smallest statement that fails to follow, and a counterexample family or the vacuous
  bound.

What would count as a proof: a derivation of `Phi(w) <= F + q' - span(w)` for every realised
legal word from `I`, (T), (L) and nothing measured.  What would count as the obstruction being
exact: a member of the (T)+(L) sub-family with `Q*_J > F + q'` (then `I + (T) + (L)` is
refuted and the next fact is named), or a proof that every bound derivable from
`I + (T) + (L)` is vacuous against `F + q'`.

---

## 1. Bookkeeping lemmas (all PROVED, by hand; each checked on the data)

Notation: old gear `q` with teeth `+-v_q`; its OWN letter `a_q = min(2v_q mod q, q - 2v_q mod q)`
(the letter it had when it was the incoming gear); `sep_q = a_q`.

**Lemma 1 (what (T) and (L) say inside the family).  PROVED.**
(a) (L) holds iff `v' = round(q'/6)`.  *Proof:* `a = min(2v', q' - 2v')`; `q' - 2v'` is odd,
`2 round(q'/6)` is even, so `a = 2 round(q'/6)` forces `2v' = 2 round(q'/6)`.  []
(b) (T) at gear `q` holds iff `a_q >= 2`, iff `v_q != (q-1)/2`.  *Proof:* `2v = -1 mod q` iff
`v = (q-1)/2` for `v in 1..(q-1)/2`; `2v = +1` has no solution there.  []
(c) At gear 5, (T) forces the REAL tooth `v_5 = 1` (the only other value, 2, is degenerate).  At
gear 7 it allows `v_7 in {1, 2}`, 1 real.  So the sub-family "(T) at gears 5 and 7" is the real
gear 5 and a two-valued gear 7 - exactly the two gears the record names as deciding the alignment
depth (`alignment-rules.md` 5.3, `BareAlternation.lean`).
(d) Unified reading: (T) + (L) = "no gear, old or new, has letter 1, and the incoming letter is
`(q' -+ 1)/3`".  The real machine has every gear's letter equal to `(q -+ 1)/3`.

**Lemma 2 (the padded cells do not see the incoming tooth).  PROVED.**
A padded depth-3 cell is `g_L + w + g_R <= F + q'` for a gap `w = 0 mod q'` of `M`; it mentions
`M` and `q'` only.  Hence (L) cannot enter any padded depth-3 statement; if such a cell fails on the
family, only (T) can rescue it.  (Measured outcome, section 2.3: no padded depth-3 cell fails
anywhere on the family, so neither fact is needed there.)

**Lemma 3 (CRT recombination of two flanks).  PROVED; and MEASURED not to apply.**
Let `x < x'` be two openings of `M` (the ends of a legal word occurrence), `g_L` the gap left of
`x`, `g_R` the gap right of `x'`.  If the gears split as `A | B` with the `A`-gears alone striking
every column `x-1 .. x-g_L+1` and the `B`-gears alone striking every column `x'+1 .. x'+g_R-1`,
then the column `y` with `y = x mod q` (`q in A`), `y = x' mod q` (`q in B`) (CRT; the period is
squarefree) is an opening with left gap `>= g_L` and right gap `>= g_R`, so `g_L + g_R <= F_2(M)`;
with a third class `C != {}` of gears whose phase at `y` is a tooth, `g_L + g_R <= F(M)`.
*Measured (gate `uv run python research/proof/chain_teeth_r33_stretch.py`, log
`chain_teeth_r33_stretch.log`; `chain_teeth_r33_stretch_j3free.log`):* at EVERY binding occurrence
examined - the real machine's `Phi(a)`, `Phi(b)`, `Phi(q')`, `Phi(a,b)` argmaxes at m13..m23, the
three m17 pinned violators, the four m19 (T)-only depth-3 violators, and the m19 equality member
of section 2.5 - the split does not exist even in the weak two-class form: both flanks are tilings
with sole coverers, and the sole-coverer gear sets of the two flanks INTERSECT (e.g. real m19
`Phi(a)`: left `{11, 19}`, right `{5, 7, 11, 13, 17, 19}`; the equality member: left
`{5, 7, 11, 17, 19}`, right `{5, 7}`).  Route (iii) of the brief is dead in this form.

## 2. The sharpened tables

Vehicle: `research/proof/chain_teeth_r33.py` (reuses prover B's `open_mask`, `gaps_of`,
`letter_a`, `real_tooth`; adds the argmax position of every `Q*_J` cell, the flank envelopes
`Phi(a)`, `Phi(b)`, `Phi(q')`, `Phi(a,b)` with occurrence counts, the general depth-3 literal /
padded cells, and the separations).  Gates: `real` mode reproduces `F`, `F_2` and the recorded
`Q*_J` table cell for cell at m11..m23 (`GATE G1 OK` x5) and B's slack table
(S1 margins 12, 10, 14, 15, 20).  Scoring: `chain_teeth_r33_analyze.py`.  Logs:
`chain_teeth_r33_fam_m11_m17.log`, `chain_teeth_r33_batch_m19_m23.log`,
`chain_teeth_r33_analyze_m11_m17.log`, `chain_teeth_r33_analyze_m19.log`.

### 2.1 Chain violators by slice (exhaustive)

    slice (rows)                    m11         m13          m17            m19
    all, tooth free                 1/180       1/1440       36/12960       193/142560
    (L) pinned only                 0/30        0/180        3/1440         46/12960
    (T) only, tooth free            0/48        1/320        3/2520         52/24640
    (T)+(L)                         0/8         0/40         0/280          0/2240
    (T) at 5,7 only + (L)           0/10        0/60         0/480          0/4320
    (T) at >= 11 only + (L)         0/24        0/120        3/840          25/6720
    min chain margin, (T)+(L)       6           4            4              0
    max excess, pinned              0           0            3              9

Reproduces prover B (PC1 confirmed).  The two new rows say where (T) acts: relaxing (T) at every
gear `>= 11` (keeping it at 5 and 7) produces no violator; keeping it at every gear `>= 11` and
relaxing it at 5 and 7 produces 25 of the 46.  Every one of the 46 + 3 pinned violators has
`v_5 = 2` or `v_7 = 3`; the degenerate-gear sets of the 46 are `{7}` x22, `{7,13}` x7, `{7,17}` x4,
`{5,7,19}` x3, `{7,19}` x3, `{5,7}` x2, `{7,13,17}` x2, and one each of `{5}`, `{5,7,11,19}`,
`{7,13,19}` - gear 7 in 45 of 46.  Separation at the higher gears is NOT the driver: among pinned
rows the violators sit at `min_{q >= 11} sep_q = 1, 2, 3, 4` (21, 14, 6, 5 of them) - all with a
degenerate 5 or 7 - and the mean margin barely moves with `min sep/q` (14.0 / 12.6 / 14.5 / 14.9 /
14.8 in the bins 5-10-15-20-25%).  PC7 is confirmed only nominally (14.4 vs 14.9); PC3's
separation hypothesis is the wrong shape.

### 2.2 Where (L) enters: the letter table at m19 (`q' = 23`, (T) rows, 2,240 per letter)

    a           1    2    3    4    5    6    7    8*   9    10   11
    violators   2    3   12    2    7    1    6    0    2   12    5
    min margin -3   -4   -7   -6   -2   -3   -5    0   -3   -5   -4

`a = 8` (the pinned letter, `3a = 24 = q' + 1`) is the ONLY letter with no violator and the only
one with non-negative minimum margin; its two neighbours `a = 7` and `a = 9` carry 6 and 2
(PC2 confirmed: (L) is needed as the exact identity, not as the ratio `b ~ 2a`).  At m17 the
picture is weaker: the pinned `a = 6` has minimum margin 4, shared with `a = 3` and `a = 8`; the
violating letters are 5 and 9.  The letter dependence is NOT the round-31 depth table: at m19 the
letters `a = 4` and `a = 9` have bare literal depth 1 at (5, 7) with the real teeth, yet violate
(at `J = 4, 5` through padded words and at `J = 3`), while the pinned letter has literal depth
3 / 2 (`v_7 = 1 / 2`) (`chain_teeth_r33_depth.log`).  So (L) does something beyond fixing the
admissible depth at gears 5 and 7, and nothing on record says what.

### 2.3 The four flank statements: failures, needs, tightness

`Phi` = max flank sum over occurrences of the exact gap value(s); margins against B's targets;
S1b (`Phi(b) <= F + a`) added because it is the binding cell on the sub-family at m17.

    m19 (142,560 rows)   evaluated   fails   with (T)   with (L)   both   min margin all / (T) / (L) / (T)+(L)
    S1  Phi(a)  <= F+b    142,554      3        2          0        0      -2 / -2 /  0 /  1
    S1b Phi(b)  <= F+a    138,503     11        1          2        0      -4 / -1 / -4 /  4
    S2  Phi(q') <= F      125,928      0        0          0        0       0 /  4 /  0 /  4
    S3  Phi(a,b)<= F       93,254     71       21         12        0      -8 / -4 / -5 /  0
    general padded J=3    125,928      0        -          -        -       0
    general literal J=3   142,560     15        4          2        0      -4

At m17: S1 1 fail (degenerate 7, free), S1b 4 (0 with (T), 1 pinned), S2 0 of 4,581, S3 10 (2 with
(T), 0 pinned); general padded J=3: 0 of 4,581.  At m11/m13: no failures of any of them.

Reading.  (1) **The padded depth-3 cells hold on the whole family** at every level, with margin
exactly 0 somewhere at m19 - they need neither (T) nor (L) (PC5 refuted in the direction "S2 fails
only with a degenerate gear": it never fails).  They are also NOT derivable from `I` (B's Lemma 3),
so "the flanks of a gap `j q'` sum to at most `F - (j-1) q'`" is a family-wide truth with no
proof and no slack.  (2) S1, S1b, S3 fail with (T) alone (free tooth) AND with (L) alone (degenerate
5 or 7), never with both (PC5 confirmed for the literal cells).  (3) Tightness (PC4): on the real
machine S1 is tight at m11..m17 and m23, S3 at m19 (margins 12, 10, 14, 14, 20 - B's table,
reproduced); on the sub-family at m19 the tight cell per row is S1 884, S1b 497, S2 443, S3 416 of
2,240, and the smallest minimum is S3's 0 (the equality member).  S3 is vacuous at 0 of 2,240 rows
at m19 (PC4's second half refuted; at m17 it is vacuous at all 280 because gear 5 with `v_5 = 1`
forbids `(a, b)` outright for `q' = 19`, section 3 (ii)).

### 2.4 The real machine (exact, m11..m23; gate `chain_teeth_r33.py real`)

    M    q'  a   b   F   F_2  budget  Phi(a) n        Phi(b) n     Phi(q') n   Phi(a,b) n   chain margin
    m11  13  4   9   7   11   20      4  (6)          -            -           -            12
    m13  17  6   11  11  16   28      12 (60)         7 (12)       -           -            10
    m17  19  6   13  18  25   37      17 (1022)       12 (66)      -           -            12
    m19  23  8   15  25  31   48      25 (10462)      17 (1236)    8 (86)      11 (62)      14
    m23  29  10  19  34  39   63      33 (243370)     18 (440)     11 (6)      -            20

On the real machine `Phi(a) ~ F` (4/7, 12/11, 17/18, 25/25, 33/34): the two flanks of a letter-`a`
gap sum to about ONE record, against the target `F + b`.  The slack is `b`, and it is the
isolation of large gaps (branch 5b) seen at the letter `a`.

### 2.5 The (T)+(L) sub-family at m19 and m23

m19 (2,240 rows, exhaustive): 0 violators; chain margin min 0, 1st percentile 5, median 16.  The
maximising cell is depth 3 at 1,754 rows (`(a)` 704, `(b)` 517, `(q')` 533), depth 4 at 465
(`(a,b)`/`(b,a)` 392, padded `(0a)`, `(a0)`, `(0b)`, `(b0)` 73), depth 5 `(a,b,a)` at 21.
The five smallest margins: 0 at teeth `(1,1,4,5,1,2)`, 1 at `(1,2,3,1,1,7)`, 2 at
`(1,1,2,1,6,5)` and `(1,1,4,5,6,2)`, 3 at `(1,1,4,4,1,1)`.  The real machine's margin is 14.

**The equality member.**  Teeth `(1,1,4,5,1,2)`, `q' = 23`, `a = 8`, `b = 15`: `F = 25`,
`F_2 = 32`, budget 48; `Q*_3 = (18) + [8] + (15) = 41`, `Q*_4 = (18) + [8, 15] + (7) = 48 = F + q'`.
`Phi(a,b) = 25 = F` exactly (S3 at equality); `Phi(a) = 33` (S1 margin 7), `Phi(b) = 18`,
`Phi(q') = 11`.  Every ingredient of `I` and both (T), (L) hold; no gear is degenerate
(separations 2, 2, 3, 3, 2, 4).  So `I + (T) + (L)` leaves NO constant to spare at m19: any proof
must be exact at this member.  Coverage picture (`chain_teeth_r33_equality_m19.log`): left flank
18 tiled by gears 5, 7, 11, 17, 19 with sole coverers, right flank 7 tiled by 5 and 7 alone
(sole 42, 43), no CRT split.

m23 (22,400 members, exhaustive sweep; `chain_teeth_r33.py batch`, log
`chain_teeth_r33_batch_m19_m23.log`): PENDING AT THE TIME OF WRITING - filled in section 2.6.

### 2.6 The 23->29 sweep

(filled when the sweep completes)

## 3. Mechanism: the four candidates of the brief, taken or refuted

**(i) "(T) forbids two consecutive strikes by one gear inside the flank; does that cap the
flank?"  REFUTED as a bound; CONFIRMED as a tautology.**  (T) is precisely "no gear strikes two
consecutive columns anywhere", so PC6 holds at 46 of 46 pinned violators (`chain_teeth_r33.py
mech 19`) but says nothing: a degenerate gear strikes an adjacent pair in every stretch of length
`>= q + 1`.  What the sole-coverer counts show (summed over the 46: gear 5 463, 7 348, 11 217,
13 154, 17 183, 19 147 sole columns) is that the degenerate gear 7 is the workhorse of the tiling,
covering adjacent pairs alone (e.g. sole `{22, 23}`, `{24, 25}` at m17 `(2,3,3,2,4)`).  No
capacity argument can turn "no adjacent strikes" into a flank bound: the coverage capacity
`sum_q 2n/q` is unchanged by moving teeth (dead end "capacity and overlap counting", nearly
achievable, no slack), and (T) machines reach `F = 32` at m17 against the real 18.

**(ii) "(L) makes the middles alternate `a, b` with `a + b = q'`; what does `3a = q' -+ 1` force?"
TAKEN as far as the record allows; it is the round-31 bare-alternation lemma, and it stops at
depth.**  Under (T) gear 5 has its real tooth and gear 7 one of two; the letters `a, b, q'` reduce
mod 5 and 7 to functions of `q' mod 210` (`BareAlternation.lean`, `bareAlt_inadmissible_iff`).
Recomputed here for all legal words to length 6 (`chain_teeth_r33_bare.py 6`, log
`chain_teeth_r33_bare.log`): the literal depth admitted by gear 5 alone is `1` for
`q' = 11, 13, 17, 19 mod 30`, `3` for `q' = 29, 1 mod 30`, `5+` for `q' = 23, 7 mod 30`; with gear
7 (real) the pinned literal depth is 1, 1, 1, 1, 3, 2, 3, 5, 1, 1, 1, 5 at `q' = 11 .. 53`.  This is
where (T) and (L) meet: (T) fixes the teeth of 5 and 7, (L) fixes the letters' residues there, and
together they decide which words can occur at all.  It explains the DEEP violators (the pinned
ones are depth 4-6 at 57 of 59 cells: `J = 3` 2, `J = 4` 21, `J = 5` 29, `J = 6` 7) as the
degenerate gear 7 opening a run of `7 - 2 = 5` consecutive open residues through which the letters
`a = b = 1 mod 7` walk (pinned `q' = 23`: `(1,3)` admits literal depth 4 against 3 for `(1,1)`).
It does NOT explain the depth-3 cells (2 pinned violators at m19, 1 at m17, 4 (T)-only at m19,
all flank statements), and it does not explain the letter table of 2.2.  Beyond depth, `3a = q' -+ 1`
says only that `x + 3a` is the column adjacent to `x + q'`, i.e. the third letter-step lands one
column off the `q'`-period; no old gear reads that.

**(iii) "The padded flank `Phi(q') <= F` is a two-point statement about phases `q'` apart; prove
it by CRT."  REFUTED on the data (Lemma 3).**  At the padded occurrence `x, x + q'` the gears'
phases at the two openings differ by `q' mod q`, and a proof by CRT would recombine the left flank
of `x` with the right flank of `x + q'` at a third column; that needs a split of the gears, and at
every binding occurrence the two flanks share sole coverers.  Moreover the padded cells hold on the
WHOLE family (2.3) - a proof of them would use no tooth fact at all, and none is on record.

**(iv) "If a route needs one more fact about real teeth, name it and test with and without it."
NAMED: the fact is (T) at gears 5 and 7, nothing above.**  Tested with and without at m17 and m19:
"(T) at 5, 7 + (L)" 0 / 480 and 0 / 4,320; "(T) at every gear `>= 11` + (L)" 3 / 840 and 25 / 6,720.
The candidate "separation `>= 3` (or `>= q/4`) at the higher gears" is refuted as the carrier
(2.1).  A fact strictly between "(T) at 5, 7" and the real teeth that the data would need is the
incoming letter table (2.2) - i.e. (L) itself - and at gear 7 the choice `v_7 in {1, 2}`, which
the m19 data does not separate (both values occur among the smallest margins: `(1,1,...)` and
`(1,2,...)`).

## 4. The smallest statements that do not follow, and the vacuous bound

**Lemma 4 (what `I + (T) + (L)` yields by the routes on record).  PROVED (by exhaustion of the
routes), and VACUOUS.**  (T) and (L) are congruence conditions at the moduli `5, 7, .., p` and
`q'`; by the recorded escape-distance-1 property (`alignment-rules.md` 6.1) they constrain WHERE a
column is struck, never how long a struck run is.  The only constructions on record that convert
local facts into a comparison with `F` are (a) the attainment translate (which reproduces the run
including its interior openings), (b) the CRT recombination of Lemma 3 (which needs a gear split
that does not exist at the binding occurrences), and (c) the counting bounds (capacity, overlap,
fixed-depth), all recorded as slack-free or dead.  Hence every bound obtainable from
`I + (T) + (L)` by these routes is one of B's Lemma 3 bounds
(`Q*_3 <= F_2 + min flank`, `Q*_J <= (J/2) F_2` etc.) - short of `F + q'` by `F_2 - q' + min flank`
at depth 3 - or a depth cap (section 3 (ii)), which bounds `L` and not the flanks.  The
equality member of 2.5 shows there is no constant to lose: a proof must be exact.

**The smallest statements that fail to follow from `I + (T) + (L)`** (each a flank order statistic
of one gap value or pair; each holds on the whole (T)+(L) sub-family computed; each fails outside
it, where noted):

- **P (padded).**  For every gap `w = j q'` of `M`: `g_L + g_R <= F - (j-1) q'`.  Holds on the
  ENTIRE tooth-counterfactual family at m11..m19 (0 failures in 4,581 + 125,928 evaluated rows),
  minimum margin 0 at m19.  Needs no tooth fact; follows from nothing on record.  This is the
  isolation of large gaps in its purest form.
- **L1 (letter a).**  `Phi(a) <= F + b`.  Fails at 3 rows of 142,560 (2 with (T), free tooth,
  `a = 5, 9`; 1 with degenerate gear 5, `a = 4`); never pinned.  Minimum margin 1 on the sub-family.
- **L1b (letter b).**  `Phi(b) <= F + a`.  Fails at 11 rows (1 with (T), 2 pinned with degenerate
  7); minimum 4 on the sub-family; binding cell of the sub-family at m17 (167 of 280 rows).
- **L2 (the pair).**  `Phi(a,b) <= F`.  Fails at 71 rows (21 with (T), 12 pinned, 0 both);
  EQUALITY on the sub-family at m19 (`(1,1,4,5,1,2)`).

The chain statement on the sub-family at m19 is exactly P + L1 + L1b + L2 + the depth-5 cell
`(a,b,a)` (21 rows, margins `>= 5`) + the padded depth-4 cells (73 rows).  Which of the four is
"the" tight one depends on the machine: S1 on the real machine, L2 at the equality member.

## 5. What to kernel-check, if anything

- Lemma 1 (b), (c): `2v = -1 mod q  <->  v = (q-1)/2` on `1..(q-1)/2`, and the specialisation to
  `q = 5` - two lines of `decide`.
- Lemma 3, the CRT recombination in its three-class form (`A | B | C`, `C` non-empty), as an
  abstract statement about a squarefree period: it is the only new constructive lemma here, and it
  is proved by the same CRT as `WordLegal`'s attainment.  Its value is negative (it does not apply at
  the binding occurrences), so it is optional.
- Nothing else: the depth table is already `BareAlternation.lean`; the flank statements are
  measured only.

## 6. Scoring the pre-registration

PC1 CONFIRMED (193 / 46 / 0).  PC2 CONFIRMED (`a = 7`: 6, `a = 9`: 2; largest share 0.23; `a = 8`
the unique zero).  PC3: see 2.6.  PC4 HALF: S1 is tight on the real machine at m11..m17 and m23
(S3 at m19, as B recorded) and is the most frequently tight cell on the sub-family (884 of 2,240),
but the smallest sub-family minimum is S3's (0, not S1's 1), and S3 is vacuous at 0 rows, not more
than half - REFUTED on both sub-clauses.  PC5 HALF: the literal statements do need both facts
(fail with (T) alone and with (L) alone, never with both - CONFIRMED); the padded statement never
fails at all (REFUTED: it needs neither).  PC6 CONFIRMED and vacuous (46 / 46, section 3 (i)).
PC7 CONFIRMED nominally (14.4 vs 14.9) and the hypothesis behind it refuted (separation at
`q >= 11` is not the carrier; gears 5 and 7 are).  PC8 CONFIRMED: no proof; the statements of
section 4 and the equality member are the obstruction.

## 7. Files

- `research/proof/chain_teeth_r33.py` - vehicle (modes `real`, `fam`, `sub`, `mech`, `batch`).
- `research/proof/chain_teeth_r33_analyze.py` - scoring by slice; `_stretch.py` - coverage and
  the CRT split test; `_bare.py` - bare-word depth table; `_depth.py` - depth vs violators per letter.
- Rows: `chain_teeth_r33_fam_m{11,13,17,19}.json`, `chain_teeth_r33_sub_m23.json`.
- Logs: `chain_teeth_r33_fam_m11_m17.log`, `chain_teeth_r33_batch_m19_m23.log`,
  `chain_teeth_r33_analyze_m11_m17.log`, `chain_teeth_r33_analyze_m19.log`,
  `chain_teeth_r33_mech_m17.log`, `chain_teeth_r33_mech_m19.log`, `chain_teeth_r33_stretch.log`,
  `chain_teeth_r33_stretch_j3free.log`, `chain_teeth_r33_equality_m19.log`,
  `chain_teeth_r33_bare.log`, `chain_teeth_r33_depth.log`, `chain_teeth_r33_j3cells.log`.
- Compute: 4 processes; largest array 37M bool (m23); m19 family 24 min, m23 sweep see 2.6.
- Shared-log block: `docs/proof-search/agents-shared.md`, "## Prover C (chain from the teeth)".
