# The coverability spectrum `COV(M)` by SAT -- second build

Instrument: **`research/cov_sat_r32.py`**.  Solver: **CaDiCaL 1.9.5**
(python-sat `cadical195`, in `.venv-sat`).  Per-run result files: one JSON per
`(M, L, J, flanks)` in `research/data/proof/`; logs `research/data/proof/*.log`
(gitignored).  Pre-registration for section (c):
`research/data/proof/prereg_c.txt`, written before any `m61+` instance was
built.

---

## 0a. CORRECTION: `COV(M)` was already built, in round 20

This work was commissioned on the premise that `COV(M)` is "the one size
instrument the record names and never built".  **That premise is false, and the
sentence it rests on is stale.**

`alignment-rules.md` section 6.5 says `[named construct; NOT BUILT]`.  But
`research/cov_sat.py` -- committed at `fe4c390`, "proof-search round 20 ...
COV-SAT reaches machine 41 complete" -- **is** `COV(M)`, built by the mechanic
lane, with the same CRT-phase-vector mechanism.  `mechanic.md` K1 records what
it achieved: exact gap spectra with complete hole lists at all eight
full-period machines m11..m37 (m37's 13 holes in 123 s of SAT against an
11,829 s scan), machine 41 complete (`F(41) = 91` and its hole list), `F_j` at
m23 `j = 2..6`, m29, m31 `j = 2..5`, with every witness CRT'd to an explicit
column and machine-verified.  It records the same wall this build hit:
"BOUNDARY-REFUTATION CLIFF at m43 tails, m47 `v >= 119`".

**Process failure, stated plainly:** this build was written to
`research/cov_sat.py` and overwrote round 20's file.  The collision was caught
at the end of the round from `git status` showing the file as *modified* rather
than new.  `research/cov_sat.py` has been **restored byte-for-byte** to its
committed state (`git diff` is empty) and this build now lives at
`research/cov_sat_r32.py`.  Nothing of round 20 is lost.  **Recommend
`alignment-rules.md` 6.5 be corrected: `COV(M)` is BUILT, `research/cov_sat.py`,
round 20, mechanic lane, and 6.5 should point at `mechanic.md` K1.**

### What the second build adds, honestly

1. **An independent re-implementation that agrees.**  Different author,
   different encoding details (pysat `CardEnc` seqcounter vs round 20's
   hand-rolled counter; `cadical195` vs `Cadical153`), same answers at every
   machine: fifteen `(M, J)` values two-sided, all equal to the corpus.
2. **NEW -- `Q*_J`, the word-legal spectrum.**  Round 20 built `Q_j`, the
   *word-free* one, whose middles satisfy the **size shadow** `>= a`
   (`min_middle` in `solve_window`).  `Q*_J` uses the **sharp** predicate
   (middles in `V = {0,+a,-a} mod q'` AND T3-alternating) and is a different,
   strictly smaller object -- the one 3.7 says satisfies
   `max_J Q*_J = F(M+q')` exactly.  See section 4.
3. **NEW -- the left-flank monotone form**, and a caveat about it (below).
4. **NEW -- values past the wall**: `F(m61) >= 171`, `F(m67) >= 175`,
   `F(m71) >= 185`.  The record's COV work stops at m41 complete with m43/m47
   refuted at the boundary.
5. A labelling defect in `alignment-rules.md` 3.7 (section 2 below).

### The caveat on the left-flank form -- it is not simply better

Round 20 scans the **whole** gap spectrum `v = 1..F+3` in the both-flanks form,
which is what makes it sound: `COV(M)` genuinely has **holes** (m37 misses
`v = 73,74,75,76,78..84,86,87` and then realises 88), so "climb until the first
UNSAT" in that form returns 72, not 88.  The left-flank form fixes that with a
single UNSAT because it is downward closed -- but **it is not cheaper**: at m37
this build paid 3,990,129 conflicts / 276 s for the one left-flank UNSAT, while
round 20 got `F` *and the complete hole list* in 123 s.  The left-flank form's
real value is that it licenses **bisection**, which matters when the range is
wide and unknown (past the wall) and not when the target is already bracketed.
That trade is stated here rather than claimed either way.

---

## 0. What the instrument decides, and why there is no period

Vocabulary is `alignment-rules.md` section 0.  Gear `q >= 5` strikes column `k`
iff `k = +-u_q (mod q)`, `u_q = round(q/6)`.  Machine `M = {5..p}`.

Anchor a candidate stretch at an unknown column `c`, write `t = 0..L+1` for the
offsets, and put `s_q = -c mod q`.  Then

    gear q strikes offset t   <=>   s_q = t -+ u_q  (mod q)

-- exactly two phase values of gear `q` strike a given offset.  The moduli are
distinct primes, so by CRT the phase vector `(s_q)` ranges over all of
`prod Z_q` as `c` ranges over the period: **a phase vector IS a column**.  This
is section 2.8's realisability CSP posed as a covering problem.  The encoding
carries no auxiliary variables at all in the `J = 1` case:

    variables   y_{q,s}, s in Z_q                              exactly-one
    flanks      ~y_{q,t -+ u_q}   for the flank offsets t      unit clauses
    interior    OR_q ( y_{q,t-u_q} v y_{q,t+u_q} )  t = 1..L   one clause each

For `J > 1` one selector `o_t` per interior column, an at-most-`(J-1)`
cardinality constraint, `o_t v OR_q(...)` and `~o_t v ~y_{q,...}`.  At `m97`
the whole instance is ~1,100 variables: the difficulty is combinatorial, not
size.

### The soundness of the two directions (this had to be fixed once, in round)

The first version climbed `L` in the **both-flanks** form and stopped at the
first UNSAT.  That is **unsound**: "both flanks spared" is not downward closed
in `L`, so `L` SAT and `L+1` UNSAT does not bound `F_J`.  The monotone
predicate is the **left-flank-only** one,

    C_J(L) = "some run of L columns with at most J-1 of them open has an open
              column immediately to its left"

A sub-run has no more openings than the run, so `C_J(L) => C_J(L-1)`; and
`max { L : C_J(L) } = F_J(M) - 1` exactly (one way is the maximal gap-run
itself; the other slides the window right of the nearest open column at or
below its left end, which can only remove interior openings).  So:

* **LOWER bound** = a **both-flanks SAT** at `L`, witness re-verified: it
  exhibits an actual run of at most `J` gaps summing to `L+1`, so `F_J >= L+1`.
* **UPPER bound** = a **left-flank UNSAT** at `L`: by downward closure it rules
  out every longer stretch too, so `F_J <= L`.

The correction costs 6-20x in solver conflicts (m31 `F`: 33,553 -> 664,600
conflicts; m31 `F_2`: 194,494 -> 2,811,829).  Every UNSAT reported below is in
the **left** form.  `bisect` binary-searches the monotone predicate, which puts
the cheap far-from-tight UNSATs first and only one tight UNSAT at the end.

### What counts as an answer

* Every SAT answer carries the **witness phase vector** `(s_q)`, and every
  witness is re-verified by `verify_witness` -- plain integer residue
  arithmetic on the phase vector, no CNF, no solver -- before its value is
  reported.  `column_of` then gives the witness's actual column `c mod P` by
  CRT, so each witness has an address in the period even though no period was
  built.
* Every UNSAT is **solver-certified, not proved**: CaDiCaL 1.9.5 via
  python-sat, encoding `research/cov_sat_r32.py:build`, left-flank form.  The
  claim is that **no covering has been found**, never that none exists.  No
  DRAT proof was checked; that is the named price.
* The budget inequality `F(M+q') <= F(M) + q'` is the **target**, never a law.

---

## 1. GATE (a): the machines with a full period

`.venv-sat/Scripts/python.exe research/cov_sat_r32.py gate` -- **ALL ASSERTIONS
GREEN**.  Ten two-sided decisions, each `L = v-1` both-flanks SAT (witness
re-verified) and `L = v` left-flank UNSAT, against the scanned corpus.

| statement | value | SAT conflicts | UNSAT conflicts | secs (SAT/UNSAT) | spared interiors |
|---|---|---|---|---|---|
| `F(m11)`   | 7  | 0     | 7      | 0.00 / 0.00 | -    |
| `F(m13)`   | 11 | 0     | 43     | 0.00 / 0.00 | -    |
| `F(m17)`   | 18 | 5     | 208    | 0.00 / 0.00 | -    |
| `F(m19)`   | 25 | 164   | 1,921  | 0.00 / 0.01 | -    |
| `F(m23)`   | 34 | 350   | 9,777  | 0.00 / 0.06 | -    |
| `F_2(m11)` | 11 | 0     | 26     | 0.00 / 0.00 | [6]  |
| `F_2(m13)` | 16 | 13    | 191    | 0.00 / 0.00 | [5]  |
| `F_2(m17)` | 25 | 106   | 908    | 0.00 / 0.01 | [7]  |
| `F_2(m19)` | 31 | 194   | 6,269  | 0.00 / 0.04 | [10] |
| `F_2(m23)` | 39 | 4,368 | 45,234 | 0.04 / 0.47 | [34] |

Every value equals the scanned corpus value.  **The SAT instrument agrees with
the period wherever a period exists.**

Witness addresses (phase vectors `q:s_q`, and the column `c` of the left flank
`mod P`, both from `research/data/proof/cov_F_*.json`):

```
F(m11)  = 7   5:0 7:4 11:0                                        c = 220        (P = 385)
F(m13)  = 11  5:3 7:4 11:10 13:8                                  c = 122        (P = 5005)
F(m17)  = 18  5:0 7:4 11:6 13:5 17:16                             c = 79425      (P = 85085)
F(m19)  = 25  5:0 7:2 11:0 13:9 17:9 19:2                         c = 1217480    (P = 1616615)
F(m23)  = 34  5:2 7:2 11:5 13:4 17:6 19:8 23:8                    c = 18165208   (P = 37182145)
F_2(m11)= 11  5:3 7:4 11:10                        spared [6]     c = 122
F_2(m13)= 16  5:3 7:0 11:1 13:9                    spared [5]     c = 3332
F_2(m17)= 25  5:0 7:2 11:0 13:7 17:9               spared [7]     c = 110
F_2(m19)= 31  5:3 7:5 11:3 13:6 17:6 19:12         spared [10]    c = 1118917
F_2(m23)= 39  5:2 7:2 11:5 13:4 17:6 19:8 23:8     spared [34]    c = 18165208
```

Note `F(m23)` and `F_2(m23)` return the **same phase vector**: the depth-2
record at m23 is the depth-1 record stretch extended by one further gap of 5,
the interior opening sitting at offset 34 = `F(m23)`.

---

## 2. (b) The corpus across the scan wall

Same protocol.  `--` marks a decision that did not finish inside the round's
budget; those rows are reported as one-sided.

| statement | corpus | SAT decides | UNSAT decides | SAT conf | UNSAT conf | secs (SAT/UNSAT) |
|---|---|---|---|---|---|---|
| `F(m29)`   | 43  | `>= 43`  | `<= 43`  | 370       | 57,705    | 0.00 / 0.57 |
| `F(m31)`   | 58  | `>= 58`  | `<= 58`  | 18,297    | 664,600   | 0.16 / 21.7 |
| `F(m37)`   | 88  | `>= 88`  | `<= 88`  | 44,134    | 3,990,129 | 0.44 / 276  |
| `F(m41)`   | 91  | `>= 91`  | (below)  | 2,218,737 | (below)   | 93 / --     |
| `F(m43)`   | 103 | `>= 103` | (below)  | 861,101   | (below)   | 35 / --     |
| `F(m47)`   | 118 | `>= 118` | (below)  | 2,360,672 | (below)   | 124 / --    |
| `F_2(m29)` | 55  | `>= 55`  | `<= 55`  | 14,908    | 283,234   | 0.14 / 5.6  |
| `F_2(m31)` | 68  | `>= 68`  | `<= 68`  | 228,411   | 2,811,829 | 3.6 / 166   |
| `F_2(m37)` | 90  | `>= 90`  | (below)  | 153,033   | (below)   | 2.0 / --    |
| `F_2(m41)` | 103 | `>= 103` | (below)  | 2,211,129 | (below)   | 99 / --     |
| `F_2(m43)` | 116 | (below)  | (below)  | --        | --        | --          |
| `F_2(m47)` | 134 | `>= 134` | (below)  | 2,632,757 | (below)   | 194 / --    |

**Every SAT witness reproduced the corpus lower bound exactly, and every UNSAT
that finished reproduced the corpus upper bound exactly -- with no period.**
`F(m37) = 88` and `F_2(m31) = 68` are the deepest two-sided SAT decisions that
completed: `F(37)`'s period is 1.2e12 columns, so this is the corpus's own
value re-derived without touching it.

Witness addresses beyond m23:

```
F(m29)  = 43   5:3 7:5 11:6 13:3 17:7 19:0 23:9 29:28                 c = 877375977
F(m31)  = 58   5:0 7:2 11:3 13:9 17:16 19:7 23:13 29:23 31:30         c = 31957808055
F(m37)  = 88   5:0 7:2 11:3 13:5 17:16 19:2 23:13 29:1 31:22 37:34    c = 1145973108145
F(m41) >= 91   5:3 7:3 11:6 13:12 17:2 19:4 23:5 29:8 31:2 37:0 41:28 c = 50077677123072
F_2(m29)= 55   5:0 7:4 11:4 13:12 17:11 19:2 23:11 29:27  spared [20] c = 858386140
F_2(m31)= 68   5:3 7:3 11:6 13:12 17:7 19:15 23:1 29:11 31:8  spared [33] c = 6249797152
F_2(m37)>= 90  5:3 7:2 11:8 13:5 17:4 19:10 23:6 29:0 31:4 37:17 spared [88] c = 90816580902
```

### A defect in the record found on the way

`alignment-rules.md` section 3.7 lists the full-period spectra and then
`41 (prefix, lower bounds) 110 112 118 123 130 138`, immediately after rows
(`13`, `17`, `19`, `23`, `29`, `31`) that are `F_1..F_6`.  Read in that column
the row claims `F(41) >= 110`, which contradicts the corpus `F(41) = 91` and
the verified SAT witness at `L = 90`.  The row is **not wrong, it is
mislabelled**: `mechanic.md:620` carries it as
`machine 41 (F=91, F+q'=133, a=14, PREFIX): F_j 110 112 118 123 130 138`, i.e.
**`j = 3..8`**, a different index range from its neighbours in the same
alignment-rules list.  Anyone reading 3.7 alone will read `F_1(41) = 110`.
Recommend 3.7 label the m41 row `(j = 3..8)`.

---

## 3. (c) Beyond the scan wall

Predictions were pre-registered in `research/data/proof/prereg_c.txt` **before
any `m61+` instance was built**.  What was purchasable inside the round's
budget was the **lower-bound** side: a both-flanks SAT witness at `L`,
re-verified by residue arithmetic, giving `F(M) >= L+1`.  These are new
values -- `F(m59) = 161` was the last corpus entry, and `m71`'s period has 26
digits.

| machine | free bound (monotone) | SAT decides | witness cost | column of the witness |
|---|---|---|---|---|
| m61 | `>= 161` (`= F(m59)`) | **`F(m61) >= 171`** | 4,186,202 conf / 190 s  | 8.077e21 |
| m67 | `>= 171`              | **`F(m67) >= 175`** | 14,535,371 conf / 1304 s | 8.914e23 |
| m71 | `>= 175`              | **`F(m71) >= 185`** | 8,646,304 conf / 506 s  | 1.699e25 |

("free bound" is `F` non-decreasing in the machine -- adding a gear only
strikes more columns, so every struck run survives.  Each row beats it.)

Witness addresses:

```
F(m61) >= 171  5:3 7:5 11:3 13:9 17:5 19:7 23:1 29:16 31:8 37:23 41:24 43:9
               47:34 53:39 59:25 61:58
               c = 8076906068473253863882
F(m67) >= 175  5:2 7:2 11:0 13:6 17:5 19:10 23:13 29:15 31:8 37:0 41:19 43:21
               47:45 53:15 59:52 61:3 67:16
               c = 891376154341410777619613
F(m71) >= 185  5:3 7:5 11:3 13:0 17:16 19:7 23:8 29:9 31:11 37:26 41:19 43:14
               47:11 53:13 59:41 61:14 67:51 71:20
               c = 16993228655883031324211777
```

### Scored against the pre-registration

Pre-registered bands were `F(m61) in 172..196`, `F(m67) in 196..223`,
`F(m71) in 223..251`.  **The predictions are NOT scored**, because only lower
bounds were purchased and a lower bound cannot fall inside or outside a band.
`F(m61) >= 171` and `F(m67) >= 175` sit *below* their bands and are consistent
with them; `F(m71) >= 185` likewise.  P1 is **undecided, not hit and not
missed**, and it stays on the record as an open bet for the next run at this.

P5 (the cost prediction) is the one that scored, and it scored **half**: it
said the instrument would reach m53 and stall at or before m67 on the UNSAT
side.  It reached **m37** two-sided and stalled at **m41** -- worse than
predicted by three rungs, because P5 was written against the *both-flanks*
UNSAT cost and the sound left-flank form costs 6-20x more.  It was right that
the report past the stall would be lower bounds with verified witnesses.

### The pair excess `F_2 - F`, and `b`

The brief asks for `F_2 - F` against `b = q' - a`, `a = 2 round(q'/6)`.  Where
SAT decided **both** members, the column is:

| machine | `F` | `F_2` | `F_2 - F` | `q'` | `a` | `b` | `F_2 - F <= b` |
|---|---|---|---|---|---|---|---|
| m11 | 7  | 11 | 4  | 13 | 4  | 9  | yes (margin 5) |
| m13 | 11 | 16 | 5  | 17 | 6  | 11 | yes (margin 6) |
| m17 | 18 | 25 | 7  | 19 | 6  | 13 | yes (margin 6) |
| m19 | 25 | 31 | 6  | 23 | 8  | 15 | yes (margin 9) |
| m23 | 34 | 39 | 5  | 29 | 10 | 19 | yes (margin 14) |
| m29 | 43 | 55 | 12 | 31 | 10 | 21 | yes (margin 9) |
| m31 | 58 | 68 | 10 | 37 | 12 | 25 | yes (margin 15) |
| m37 | 88 | `>= 90` | `>= 2` | 41 | 14 | 27 | not decided from below alone |

**Seven machines decided two-sidedly by SAT, all seven satisfy
`F_2 - F <= b`.**  That is the corpus's own margin re-derived without a period,
not new evidence about the inequality: these are the same seven machines the
period already covers.

**Beyond the wall the pair-excess column is EMPTY and that is the honest
result of this round.**  `F_2` needs both a lower and an upper bound to give an
excess, and no `F_2` value at `m61+` was purchased: the `m61` `F_2` lower-bound
climb was started twice (at `L = 180`, then restarted at `L = 176`) and
produced no witness inside its budget.  The pair statement therefore has
**not** been tested past the wall.  Its price is named in section 6.

### The budget inequality past the wall: not purchasable this way

The plan was to certify `F(M + q') <= F(M) + q'` directly, as a left-flank
UNSAT at `L = F(M) + q'` -- an UNSAT far *above* the true `F`, which is the
cheap end.  At `m61`, `L = 222 = F(m59) + 61`, that decision **did not
finish** in ~50 minutes and was abandoned.

There is no cheaper fallback: **the counting bound is vacuous from m37 up.**
A window of `L` columns can be struck by gear `q` at up to `2 ceil(L/q)`
offsets, and `sum_{q = 5}^{37} 2/q = 1.518 > 1`, so capacity exceeds `L` at
every `L` for every machine from m37 on and the pigeonhole gives nothing.
Every upper bound on `F` past m37 has to be bought from the solver.

---

## 4. (d) The word-legal restriction `Q*_J`

`Q*_J(M; q')` is `F_J` with the `J-1` interior spared columns required to be
**word-legal**: every middle in `V = {0, +a, -a} mod q'` and the letter word
T3-alternating.  The instrument encodes that predicate **at its source** rather
than through T2/T3: a set of points has legal middles and alternates iff the
points all lie on **one phase of gear `q'`** -- the differences of two elements
of `{s+u', s-u'}` are `0` and `+-2u' = +-a`, and the alternation is exactly the
two-tooth bookkeeping.  So `q'` is added as one more phase variable required to
strike every spared interior; the flanks are deliberately left unconstrained
with respect to `q'`, because the record's `Q*_J` puts no condition on the two
outer gaps.  `verify_witness` then checks **both** formulations independently
and asserts they agree, on every witness.

Because `Q*_J` is **not** monotone in `L`, the whole range is scanned; there is
no first-UNSAT stop.

### GATE: `max_J Q*_J(M; q') = F(M + q')`

| machine | `Q*_2` | `Q*_3` | `Q*_4` | `Q*_5` | `max_J` | `F(M+q')` |
|---|---|---|---|---|---|---|
| m23 (`q'=29`) | 39 | 43 | empty | empty | **43** | **43** |
| m29 (`q'=31`) | 55 | 58 | 55 | 55 | **58** | **58** |

Both agree.  Two further independent confirmations fall out:

* `Q*_2 = F_2` at both machines (39 at m23, 55 at m29): the depth-2 record
  stretch's interior opening is word-legal at both.
* The `Q*_5(m29)` witness has spared interiors `[7, 17, 38, 48]` inside
  `L = 54`, i.e. the gap word `(7, 10, 21, 10, 7)` -- **exactly** the
  self-reverse `J = 5` maximiser `alignment-rules.md` 3.7 records at m29, found
  here by a completely different vehicle.
* `Q*_J(23)` is empty for every `J >= 4`, consistent with the record's
  "emptiness is upward closed" and with `A_kill` at m23.

Witness addresses:

```
Q*_2(m23) = 39  5:2 7:2 11:5 13:4 17:6 19:8 23:8      spared [34]        c = 18165208
Q*_3(m23) = 43  5:3 7:5 11:6 13:3 17:7 19:0 23:9      spared [23,33]     c = 22186642
Q*_2(m29) = 55  5:0 7:2 11:3 13:9 17:15 19:12 23:21 29:8   spared [30]   c = 133490530
Q*_3(m29) = 58  5:3 7:0 11:0 13:10 17:8 19:13 23:22 29:28  spared [30,40] c = 799661632
Q*_4(m29) = 55  5:0 7:2 11:6 13:3 17:11 19:10 23:16 29:23  spared [2,23,33] c = 416961165
Q*_5(m29) = 55  5:0 7:3 11:1 13:4 17:2 19:5 23:16 29:18 spared [7,17,38,48] c = 858111055
```

Note `Q*_3(m23) = 43` and its phase vector is a **prefix of** the `F(m29) = 43`
witness (`5:3 7:5 11:6 13:3 17:7 19:0 23:9`, then `29:28`): the m29 record
stretch is literally the m23 word-legal depth-3 run with gear 29 slotted in to
kill both of its interior openings.  That is the merge law happening in front
of the solver, and the instrument recovers it without a period.

---

## 4b. THE LADDER, as this instrument leaves it

`cov_sat_r32.py table` collects every per-run JSON into this.  "exact" means both
directions decided: a re-verified both-flanks witness at `L = v-1` and a
left-flank UNSAT at `L = v`.

| M | `F` by SAT | `F_2` by SAT | corpus `F` / `F_2` |
|---|---|---|---|
| m11 | **7** exact  | **11** exact | 7 / 11 |
| m13 | **11** exact | **16** exact | 11 / 16 |
| m17 | **18** exact | **25** exact | 18 / 25 |
| m19 | **25** exact | **31** exact | 25 / 31 |
| m23 | **34** exact | **39** exact | 34 / 39 |
| m29 | **43** exact | **55** exact | 43 / 55 |
| m31 | **58** exact | **68** exact | 58 / 68 |
| m37 | **88** exact | `>= 90`      | 88 / 90 |
| m41 | `>= 91`      | `>= 103`     | 91 / 103 |
| m43 | `>= 103`     | `>= 116`     | 103 / 116 |
| m47 | `>= 118`     | `>= 134`     | 118 / 134 |
| m53 | not attempted | not attempted | 145 / 159 |
| m59 | not attempted | not attempted | 161 / 173 |
| **m61** | **`>= 171`** | none found | **no corpus entry** |
| **m67** | **`>= 175`** | not attempted | **no corpus entry** |
| **m71** | **`>= 185`** | not attempted | **no corpus entry** |

**Fifteen `(M, J)` values decided exactly, all fifteen equal to the corpus.
Ten more lower bounds, every one equal to or beyond the corpus where a corpus
value exists.  Three new rows past the wall, all one-sided.**

The `F_2(m59) <= 173` entry the brief flagged as "conditional on record" is
**still conditional**: it was not attempted, and nothing here makes it
unconditional.

---

## 5. Reproducing this

```
.venv-sat/Scripts/python.exe research/cov_sat_r32.py gate            # section 1
.venv-sat/Scripts/python.exe research/cov_sat_r32.py check 37 1 88   # two-sided
.venv-sat/Scripts/python.exe research/cov_sat_r32.py bisect 43 1 102 # F by bisection
.venv-sat/Scripts/python.exe research/cov_sat_r32.py ub 61 1 222     # upper bound only
.venv-sat/Scripts/python.exe research/cov_sat_r32.py lb 61 1 161     # lower bound only
.venv-sat/Scripts/python.exe research/cov_sat_r32.py legal 29 3 30 62  # Q*_3
.venv-sat/Scripts/python.exe research/cov_sat_r32.py table           # collect results
```

---

## 6. Prices, and what did not finish

Cost is reported as **solver conflicts** (the op-count proxy), with wall
seconds secondary.  Jobs ran concurrently on up to 6 cores, so **the wall
seconds across different rows are not comparable to each other**; the conflict
counts are.

### The shape of the cost

| decision | UNSAT conflicts (left form) |
|---|---|
| `F(m29) <= 43` | 57,705 |
| `F(m31) <= 58` | 664,600 |
| `F(m37) <= 88` | 3,990,129 |
| `F_2(m29) <= 55` | 283,234 |
| `F_2(m31) <= 68` | 2,811,829 |

Roughly **6-11x per rung** on the `F` ladder.  Extrapolated, `F(m41) <= 91`
is a ~4e7-conflict decision and `F(m43) <= 103` a ~4e8 one.

The SAT (lower-bound) side has the same shape but shifted: it is cheap until
`L` approaches `F`, then climbs steeply -- `m71` at `L = 176` cost 164,825
conflicts (2.9 s) and at `L = 180` cost 13,334,483 (1167 s), a 81x jump for
four columns.  So **the lower-bound side saturates too**, and it saturates well
before the pre-registered `F` estimate.  Whether that means the estimates are
too high or merely that CDCL struggles in the last stretch is **not decided by
anything measured here**.

The measured corner matches round 29's own lesson about the k-axis: the
expensive instance is the **tight** one, not the large one.  `m41` is the
tightest step on the corpus (`dF = 3` against `q' = 41`) and it is by far the
dearest rung: `F(m41) >= 91`'s SAT direction alone cost 2,218,737 conflicts
against `m43`'s 861,101 for a *longer* stretch.

### Not finished, with its price

| target | status | price paid | what it would take |
|---|---|---|---|
| `F(m41) <= 91`  | did not finish | ~85 min, killed | est. ~4e7 conflicts, ~1 core-hour |
| `F_2(m37) <= 90`| did not finish | ~65 min, killed | est. ~4e7 conflicts |
| `F_2(m41) <= 103`| did not finish | ~80 min, killed | est. ~1e8 conflicts |
| `F(m43) <= 103` | did not finish | ~25 min, killed | est. ~4e8 conflicts, ~8 core-hours |
| `F_2(m43) <= 116`| did not finish | ~30 min, killed | est. ~1e9 conflicts |
| `F(m47..m59)`, both directions | **not attempted** | - | out of reach of this encoding |
| `F(m61) <= 222` (the budget inequality at 59->61) | did not finish | ~50 min, abandoned | unknown; no counting fallback exists |
| `F_2(m61+)`, hence every pair excess past the wall | no witness found | two starts, `L = 180` and `L = 176` | unknown |
| `F(m73..m97)` | **not attempted** | - | the round ran out before the cores did |
| `F_3` at m47..m61 (brief item d, first half) | **not attempted** | - | the `J = 2` rungs did not finish, so `J = 3` was never reached |
| `Q*_J` at m59, m61 (brief item d, second half) | **not attempted** | - | gated at m23 and m29 only |
| a DRAT proof for any UNSAT | **not attempted** | - | this is the standing caveat on every UNSAT here |

The two-sided ladder therefore stops at **m37** for `F` and **m31** for `F_2`,
and past that point this instrument delivers bounds, not values.

### The honest summary of what this second build is worth

* The construct is **not new** -- round 20 built it (section 0a).  What is
  confirmed here is that it **reproduces independently**: a different author,
  a different cardinality encoding and a different CaDiCaL agree at every
  machine, fifteen two-sided values all equal to the corpus, no period built.
* `Q*_J` **is** new: round 20 built the word-free `Q_j`, not the sharp one.
  It gates at m23 and m29 and it recovers the record's own m29 palindrome.
* It reaches **past the wall in one direction only**: verified lower bounds at
  m61, m67, m71, the deepest witness at column 1.699e25.  Round 20's COV
  stopped at m41 complete with m43/m47 refuted at the boundary, so these three
  rows are the genuine ladder extension.
* It does **not** reach the upper bounds past the wall, which is the direction
  6.5 wanted `COV(M)` for.  6.5's claim that `COV(M)` "reaches machines 37, 41,
  43, 53 whose periods are beyond any scan" is **half confirmed, and round 20
  had already confirmed the same half**: m37 and m41 yes, m43 and m53 no.  The
  construct is right; a plain CDCL encoding of it is not enough.
* The obvious next moves, none taken: a DRAT-checked UNSAT so the upper bounds
  stop resting on the solver word; symmetry breaking (the reflection
  `s_q -> L+1-s_q` is a global involution, worth ~2x, which is one rung of
  nothing); and -- the one that might actually matter -- a **cube-and-conquer**
  split on the small gears' phases, since gears 5, 7, 11 have tiny domains
  after the flank units and 5x7x11 = 385 cubes would parallelise the UNSAT
  side across cores instead of running one solver per machine.
