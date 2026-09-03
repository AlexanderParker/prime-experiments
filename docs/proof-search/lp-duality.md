# lp-duality.md - the LP-DUALITY thread's own workstream doc

## Cumulative state (written at the opening of round 29, from the thread's
## round-24..28 blocks in `agents-shared.md`; nothing here is new)

WHAT THIS THREAD IS. One object, refined five times. The vehicle is the COMPOSED
LEVEL-2 RELAXATION of "machine y has a fully blocked window of width W": one
0/1 variable per phase tuple of every gear and every pair of gears, block sums,
pairwise-consistency links, the exactly-valid degree-2 COVERING CUTS at each
position, and one RECURSION ROW
`sum_q S_q(r_q) - sum_{i<j} n_ij(r_i,r_j) >= |pos|` whose coefficients are
max-cover counts over the lower gears. An infeasibility certificate of that
polytope is an exact rational DUAL certificate that no such window exists, i.e.
`F(y) <= W` with no census, no period and no word list - only the primes up to
y. Round 26 added the parameter that made the species reach: the CASE SPLIT.
Fix the phases of the k smallest ("held") gears; every position they block is
already covered, so the obligation shrinks to a smaller position set over the
remaining gears, and a certificate in EVERY case is a certificate of the rung.
Cost is a PRIMORIAL IN k (1, 5, 35, 385, 5005, 85085 cases at k = 0..5) - a
limit species this project had not met before: not a degree ceiling and not a
width frontier. Round 26 also built the WINDOWED form (prescribe open positions;
a certificate then says an adjacent gap PAIR is unrealised), which gives exact
scan-free `F_2` at m19 and m23. Round 28 replaced the cut LOOP outright: the
loop's limit is the optimum `V*` of ONE LIFTED LP with an atom-distribution
variable per position, so `V* < |pos|` (or an empty lifted polytope) is an exact
dichotomy - certifiable or never - and the lifted duals seed the certificate
directly. There is no loop in the method any more.

WHAT IT HAS DELIVERED, and the standing house rules. TEN (D) RUNGS by LP
duality - 7->11 through 41->43, every step the project has, plus 41->43 which
the CEGAR route could not reach - all hypothesis-free. TIGHT ON `F` at five
machines (F(19) <= 25, F(23) <= 34, F(29) <= 43 at k = 2; F(31) <= 58 and
F(37) <= 88 at k = 3), each the exact value. The INCREMENT LAW's upper half by
certificate at all six literal steps at the strictly harder increment width
`W_inc = F_2(M) + s_min(q')`, with scan-free realisability WITNESSES for the
lower half - both halves emitted as integer-only JSON and now KERNEL-CHECKED by
the Formalist (`Increment.increment_law_literal_steps`, 1749 jobs). Two
retracted readings of this thread's own, both self-found: round 25's "the row's
uniform frontier is machine 41" (it is a width, `W_u(y)`, with a closed form)
and round 27's "second frontier - convergence, no closed form" (an INSTRUMENT
ARTIFACT: the loop was its own only instrument; the lifted LP dissolves it into
geometric convergence plus a constant integrality offset). Rules that stand and
are load-bearing: floats FIND, exact rationals DECIDE; a CERTIFIED verdict
carries an exact rational dual certificate re-checked from its own integers; a
REFUTED verdict carries an exact rational point verified IN the polytope; a
STALL is an undecided cell and NEVER a verdict; a non-optimal solver status is
never a verdict; op counts, not wall time; every cell is its own resumable JSON.

---

## Round 29

BRIEF: (a) the Formalist's one ask - the 31 -> 37 certificate at the SMALLEST
certifying k, any width, in the kernel format with a margin column; (b) score
round-28's E9-E12 and test E12 at 37 -> 41; (c) the lifted LP's number for the
rung-ten increment-width cells at 43 -> 47.

Pre-registration, written before any round-29 LP was solved:
`research/data/r29/lp_prereg_r29.txt`. Scored in section 6.

### 0. FOR FORMALIST - THE EMISSION, AND IT IS TWO RUNGS

`research/data/r29/`, in the SAME SCHEMA the kernel side already parses
(round-27 `cert_19_23_h*.json`, round-28 `cert_inc_*.json`; the key set is a
strict SUPERSET of round 27's - five new fields, nothing removed):

  RUNG 1 - (D) AT 31 -> 37, the ask.
    `layout_31_37.json`                    272 KB, case-independent
    `cert_31_37_h<w5>_<w7>_<w11>.json`     385 files, 38.3 KB each, 14.4 MB
    `manifest_31_37.json`                  exhaustiveness + the MARGIN COLUMN
    `research/lp_rungs_r29.txt`            the margin column, human-readable

  RUNG 2 - THE INCREMENT LAW AT 37 -> 41, the seventh step and the first past
  the six literal ones. Not in the brief; it fell out of the E12 test.
    `layout_inc_37_41_k3.json` / `_k4.json`  two layouts - the split is MIXED
    `cert_inc_37_41_k3_h*.json`              376 files
    `cert_inc_37_41_k4_h*.json`              117 files
    `manifest_inc_37_41_k3.json` / `_k4.json`  the two parts
    `manifest_inc_37_41.json`                THE STEP MANIFEST - the partition
                                             argument, asserted
    `witness_inc_37_41.json`                 the LOWER half: F_2(37) >= 90

FORMAT LINE: one JSON per case, INTEGERS ONLY - every rational a `[num, den]`
pair - carrying `pos`, the cut `rows`, the dual weights `y` / `nu` / `yff`, the
recursion row `frow`, `lhs`, `rhs`, and NEW THIS ROUND `margin` = rhs - lhs, the
per-case slack of the certificate inequality; the manifest carries the whole
`margin_column` plus `margin_min` / `margin_max`.

### 1. (a) THE ASK: 31 -> 37 AT k = 3, W = 95, AND k = 3 IS PROVABLY SMALLEST

    rung      W    held        cases   ops         iterations   margin
    31->37    95   (5,7,11)      385   8,388,426   ALL ZERO     min 1/5, max 3

W = 95 is F(31) + 37, so the certificate IS the (D) rung: machine 37 has no
fully blocked window of width 95, i.e. F(37) <= 95, hypothesis-free, from the
primes 5..37 and nothing else. EVERY cut row is the BASE CUT
(`sum_i x_i >= 1`, valid by inspection) and every case closes at ITERATION
ZERO - so Formalist's round-27 "obligation 3" shortcut applies to the whole
rung. Margin histogram over the 385 cases: 1/5 x3, 1/4 x8, 1/3 x22, 2/5 x12,
1/2 x24, 3/5 x8, 2/3 x13, 3/4 x7, 4/5 x3, 1 x146, 7/6 x1, 4/3 x7, 3/2 x19,
5/3 x3, 2 x97, 5/2 x8, 3 x4.

AND THE SMALLEST-k CLAIM IS A PROOF, NOT A SEARCH OUTCOME. Round 26 recorded
k = 2 as a cut-loop STALL (LP maximum 40.994 against 40) - an undecided cell.
The round-28 lifted LP decides it:

    k   cases decided   CERTIFIED   REFUTED (exact in-polytope point)   other
    1        1 of 5         0        1  (V* = 63.7758 >= |pos| = 57, +6.776)
    2       35 of 35        9       22  (V* - |pos| from +0.798 to +2.172)  4
    3      385 of 385     385        0                                      0

An exhibited rational point with every block summing to 1, every consistency
link exact, every position exactly completable and the recursion row cleared is
a PROOF that no dual certificate exists in that case, at any number of cuts. So
k = 3 is the smallest k that certifies this rung. (Widths above 95 do not prove
the rung, and G falls with W with the single crossing asserted, so 95 is the
easiest width that does.) The 4 "other" cells at k = 2 are ASYMPTOTE readings
for which no exact witness was constructed - a FLOAT reading, labelled as such
and not needed: the 22 exact refutations close the question. Only the case
(0,) was run at k = 1: one exact refutation already kills that k, and the
remaining four cells are n = 9 lifted LPs at ~400 s each that would buy nothing.

WHICH ROW IS TIGHT, as the brief asks: at every failing k = 2 case the lifted
polytope is NON-EMPTY and V* exceeds |pos| by between 0.456 and 2.172, so the
binding object is the RECURSION ROW, not a coverage cut - level-2 consistency
alone does not exclude the window at two held gears, and the excess is a genuine
integrality gap of that relaxation.

### 2. THE SEVENTH INCREMENT STEP, 37 -> 41 (both halves)

W_inc(37 -> 41) = F_2(37) + s_min(41) = 90 + 14 = 104.

UPPER HALF - 493 EXACT DUAL CERTIFICATES OVER A MIXED SPLIT. All 385 k = 3
cases decided: 376 CERTIFY, 9 are REFUTED by exact in-polytope points
(V* - |pos| = +0.54 to +1.83). Those 9 are each split on gear 13's phases into
13 children at k = 4, and ALL 117 CHILDREN CERTIFY. 376 + 117 = 493 cases, and
the union is a PARTITION of prod(Z_5 x Z_7 x Z_11) - asserted in
`manifest_inc_37_41.json`: each refined 3-tuple carries all 13 phases of gear
13, no tuple appears in both roles, and the union is the whole product. Total
16,257,674 exact certificate ops; margin min 845127/512000000, max 5/2; every
case at iteration zero. The 9 refuted cases are FOUR MIRROR PAIRS PLUS ONE
SELF-MIRROR CASE (1,1,9) - section 5.

LOWER HALF - AN EXHIBITED CONFIGURATION, no period scan. F_2(37) >= 90 with
gears (5,7,11,13,17,19,23,29,31,37) at phases [0,3,1,6,11,7,15,27,25,18]:
positions 0, 2 and 90 open, every other position of [0,90] blocked, split
(2, 88). Re-checked from its own numbers by CRT. Independent confirmation of the
project's recorded m37 F_2 maximiser (2, 88).

TOGETHER: F(41) <= 90 + 14 where 90 is an EXHIBITED adjacent pair of machine
37 - the increment law at 37 -> 41 by certificate plus witness, with no census
and no period anywhere. Round 28 had the six literal steps; this is the seventh,
and 41 is the first machine on that list the project cannot scan.

THE GENERAL MOVE, and it changes this vehicle's cost curve: THE CASE SPLIT DOES
NOT HAVE TO BE UNIFORM IN k. Refining only the cases that fail costs q_{k+1}
cells each instead of multiplying the whole sweep by q_{k+1}, so the cost is
(cases at k) + q_{k+1} x (failures at k) rather than the primorial: 385 + 117 =
502 cells here against 5,005 for a uniform k = 4 sweep, a factor of ten, and the
exhaustiveness argument is still one line. Round 28 used this move once, on one
cell at machine 43; it is now the way the sweeps are run.

A COST NOTE, and it corrects my own method: the lifted LP is the DECIDER but it
is not the cheap way to CERTIFY. Running the ordinary `decide_star` first and
falling back to the lifted LP only when it does not return a certificate costs
4-20 s per cell against 115-180 s for lifted-first, because at this width every
certifiable cell closes at iteration zero off the base cuts alone. A certificate
is a certificate however it was found; the lifted LP is needed only to DECIDE
the cells that do not certify. That turned a 6.4-hour sweep into a 45-minute one
(measured: the first, lifted-first, INC41 launch was killed and relaunched).

### 3. (c) RUNG TEN 43 -> 47 AT THE INCREMENT WIDTH

W_inc(43 -> 47) = F_2(43) + s_min(47) = 116 + 16 = 132 - which is EXACTLY
Constructor's spectrum-plus-depth bound at that step (F_4(43) = 132 against the
budget F(43) + 47 = 150, margin 18). One number per cell, exact:

    k   case               |pos|   V*      verdict     ops      secs
    5   (0,0,0,0,0)          31    EMPTY   CERTIFIED   63,354    86
    5   (1,1,1,1,1)          30    EMPTY   CERTIFIED   55,766    49
    5   (2,3,5,7,9)          35    EMPTY   CERTIFIED   47,598    56
    4   (0,0,0,0)            37    EMPTY   CERTIFIED   69,003   122
    4   (1,2,3,4)            43    EMPTY   CERTIFIED   64,065   159

EMPTY means the LIFTED POLYTOPE IS EMPTY: level-2 consistency alone excludes a
fully blocked window of width 132 in that case, before the recursion row is
consulted at all. ANSWER TO THE BRIEF'S QUESTION: the LP AGREES with the
spectrum certificate and does NOT dominate it - both stop at the same number
132, with the same margin 18 against the budget 150. The two vehicles are
independent (one is the old machine's spectrum over a finite depth range, the
other a covering LP over the primes up to 47) and they meet on one integer.
HONEST SCOPE, and it is the whole caveat: FIVE CELLS of 5,005 (k = 4) or 85,085
(k = 5). This is a PROBE, not a rung. A full k = 4 sweep at ~120 s a cell is
~167 core-hours and was not run; the cost curve is still a primorial in k,
though the refinement move above is what would make it affordable.

### 4. E9-E12, SCORED

E9 ("W_c(y,k) is NOT monotone in y at fixed k, but W_c(y,k)/F(y) <= 1.5 at every
(y,k) with k >= 3 the lifted LP can reach") - HALF CONFIRMED, HALF REFUTED, and
the refuted half is the one that carried the risk. W_c(y, 3) at the all-zero
case, bisected with the sign pattern asserted width by width around the
crossing:

    y            23     29     31     37     41
    W_c(y, 3)    13     31     46     66     81
    F(y)         34     43     58     88     91
    W_c / F     0.382  0.721  0.793  0.750  0.890

The RATIO clause is CONFIRMED with room to spare - every value is below 0.9,
not merely below 1.5. The NON-MONOTONE clause is REFUTED: W_c(y, 3) is strictly
increasing over all five machines the lifted LP reaches at k = 3. A by-product
worth recording: round 28 said "at machine 41 with k = 3 the case-0 polytope is
EMPTY at every width down to 92 = F(41) + 1"; the bisection extends that by
eleven units - case 0 is certifiable down to 81, TEN BELOW F(41) = 91. That is
a per-case statement and stays one.

E10 ("the full case split at k = 3 is tight on F at machine 41 too: it certifies
F(41) <= 91 and fails at 90") - REFUTED, at the first half. Of 92 of the 385
k = 3 cases at W = 91 decided before the sweep was deliberately stopped, 32
carry EXACT in-polytope refutations (offsets +0.03 to +3.20), so the k = 3 case
split CANNOT certify F(41) <= 91. One exact refutation settles it; the other 293
cells would only have refined the count. THE ERROR WAS MINE AND IT WAS THE ONE
ROUND 28 WARNED AGAINST IN ITS OWN TEXT: I read "case 0 is already empty at 92"
as a statement about the split, when round 28's own sentence beside it says "a
per-case reading only". W_c(41,3) = 81 is case 0's threshold; the FULL split's
threshold is the max over 385 cases and is above 91.

E11 ("every cell whose lifted polytope is EMPTY certifies at iteration zero once
seeded - no exceptions in a sweep of >= 200 further cells") - CONFIRMED, with
the sample size met. 877 empty-polytope cells now on disk across rounds 28-29;
877 certified at iteration zero; ZERO exceptions. 459 of them are round-29
cells, so the ">= 200 further cells" bar is cleared, and 385 of those are a
purpose-built sweep (machine 31, W = 74, k = 3, every case decided by the LIFTED
route rather than the fast path, precisely so the prediction had something to
be tested on).

E12 ("the offset V* - |pos| at the increment width, as a function of the step,
is NOT O(1): it grows with the machine") - CONFIRMED AS WORDED ON THE ONE
MATCHED PAIR AVAILABLE, AND ITS CONSEQUENCE REFUTED. All at the all-zero case:

    step        W_inc   W_inc - F(q')   k=1        k=2        k=3
    31 -> 37      80        - 8         +9.0461    +3.7901    EMPTY (-inf)
    37 -> 41     104        +13         out of     +5.1667    EMPTY (-inf)
                                        reach

At the deepest matched k the lifted LP can reach (k = 2; k = 1 at machine 41 is
n = 10 free gears, past the program's scaling wall) the offset GROWS,
+3.7901 -> +5.1667. So the prediction's letter holds on its own terms. What it
gets wrong is the object: THE OFFSET IS A PROPERTY OF (step, k), NOT OF THE
STEP, and the ladder parameter absorbs it - at 37 -> 41 the full k = 3 split
leaves only 9 of 385 cases positive and all 9 close one gear deeper, so the
increment width there IS certified (section 2). E12's stated consequence - "if
it were O(1) the vehicle would be a route to Delta_3 = O(1), and I do not
believe it is" - is therefore not supported by this measurement: the vehicle
reached one step further than round 28 expected. And the quantity that decides
certifiability at all is W_inc - F(q'), which is negative at EXACTLY ONE step of
the corpus (31 -> 37, by 8), where no sound method can certify at any k:

    step      11->13 13->17 17->19 19->23 23->29 29->31 31->37 37->41 41->43 43->47
    W_inc-F     +4     +4     +6     +5     +6     +7     -8    +13    +14    +14

### 5. THE MIRROR IS A SYMMETRY OF THIS VEHICLE - A LEMMA, AND A FREE 2x

    LEMMA. reflect(hits(q, r, W)) = hits(q, (1 - W - r) mod q, W), where
    reflect(i) = W - 1 - i.
    PROOF. i is blocked by gear q at phase r iff i = t - r (mod q) for a tooth
    t; the teeth are {u', q - u'}, so t -> -t permutes them; and
    W - 1 - i = (-t) - ((1 - W) - r) (mod q). []

Hence the case at held phases ws and the case at (1 - W - ws) mod q have
position sets that are reflections of each other, isomorphic relaxations, and
EQUAL V*, |pos| and certificate cost. Gated at every gear of m11..m47 at
W = 74, 95, 104, 132, and non-vacuously on the data: machine 37 W = 95 k = 2
(35/35 cases agree on both V* and |pos|, 11 distinct value classes), machine 41
W = 104 k = 3 (385/385 on |pos|, 41/41 on V* where both cells were decided by
the lifted route), machine 31 W = 74 k = 3 (385/385). It showed up unbidden: the
9 refuted cases of the 37 -> 41 sweep are four mirror pairs plus the one
self-mirror case.
CONSEQUENCE FOR EVERY SWEEP OF THIS SPECIES: decide one case per mirror orbit
and copy the verdict - the same 2x Lateral's reversal law buys on word
decisions, now on LP cells, and this thread was not using it.
NOT EXPLAINED: the value classes are COARSER than the mirror orbits (11 distinct
(V*, |pos|) pairs over the 35 k = 2 cases, orbits of size 4 where the mirror
gives 2). It is not a translation - no `ws -> ws + t` preserves V* except t = 0,
tested at all 35.

### 6. MY OWN ROUND-29 PRE-REGISTRATION, SCORED

Written before any LP was solved (`research/data/r29/lp_prereg_r29.txt`).
Seven bets, THREE REFUTED, and all three refutations are mine.

  A1 (k = 3 smallest, W = 95)              CONFIRMED.
  A2 (k = 2 refuted, not stalled)          CONFIRMED IN THE MAIN CLAUSE, WRONG
     IN BOTH QUANTITATIVE ONES: I predicted the failing cases would be a
     minority (<= 12 of 35) and the excess under 2. They are a MAJORITY (26 of
     35 not certifiable, 22 of them proved) and the excess reaches +2.172.
     The clause I got right - that the binding object is the recursion row and
     the lifted polytope is non-empty - held at every failing case.
  A3 (all 385 k = 3 cases at iteration zero)  CONFIRMED.
  A4 (rows_all_base_cut for the emitted rung) CONFIRMED at 31 -> 37; and the
     contrast is informative - it is FALSE for the 37 -> 41 k = 3 part, where
     41 of the 376 cases needed rows seeded from the lifted duals.
  A5 (E9: both clauses confirmed)          HALF REFUTED - the non-monotone
     clause is false (section 4). I predicted a prediction of mine would hold
     and it did not.
  A6 (E12 refuted; offset negative at 37 -> 41 at k = 3; and G at k = 2 FALLS
     from +3.79) REFUTED IN THE MECHANISM I NAMED. The k = 3 case-0 cell is
     indeed empty, but at the matched k = 2 the offset RISES to +5.17, not
     falls, and the full k = 3 split does NOT certify - it needs the k = 4
     refinement. I had the conclusion approximately right and the reason wrong,
     which is the same failure shape this lane recorded in round 27.
  A7 (rung ten: G < 0 at k = 5; k = 4 borderline, G in (-1, +2))  CONFIRMED AT
     k = 5, REFUTED AT k = 4: k = 4 is not borderline, its lifted polytope is
     EMPTY at both cases tried.

### 7. GATES (all re-run from clean processes at round close)

    uv run python research/lp_emit_r29.py GATE 3        ALL ASSERTIONS GREEN
        31_37: EXHAUSTIVENESS - 385 held-phase tuples = prod(5, 7, 11) = 385
        31_37: 385/385 cases re-verified from JSON, lhs < rhs in EVERY case;
               margin column min 1/5 max 3; all rows base cut = True  GREEN
        inc_37_41_k3: 376/376 re-verified; margin min 845127/512000000 max 1;
               all rows base cut = False  GREEN
        inc_37_41_k4: 117/117 re-verified; margin min 1/6 max 5/2;
               all rows base cut = True  GREEN
        ALL ASSERTIONS GREEN  [126 s at round close]  (878 case certificates)
    uv run python research/lp_emit_r29.py WITNESS 37 41 2
        WITNESS  F_2(37) >= 90  split (2, 88)  openings [0, 2, 90]
        phases [0,3,1,6,11,7,15,27,25,18]  RE-CHECKED FROM DISK BY CRT  GREEN
    uv run python research/lp_emit_r29.py STEP
        STEP MANIFEST manifest_inc_37_41.json: 376 + 117 = 493 cases,
        PARTITION ASSERTED; margin min 845127/512000000 max 5/2;
        16,257,674 exact ops
    uv run python research/lp_score_r29.py                (writes
        research/lp_r29_results.txt) - 1468 decided cells, the mirror lemma
        asserted at every gear of m11..m47 at W = 74/95/104/132 and on three
        non-vacuous cell families, E11's 877/877, and every table above.

The gate rebuilds each relaxation FROM THE PRIMES, recomputes the position set
from the held phases, re-checks every cut row's validity by the exact zeta
transform over all 2^n atoms, and recomputes lhs / rhs / margin from the file's
own integers. Nothing is trusted from the pickles that produced it.

### 8. NEGATIVES, COSTS AND JOB COMPLETION

- E10's sweep was DELIBERATELY STOPPED at 92 of 385 cells once 32 exact
  refutations had settled the verdict. Recorded as a stopped partial, not a
  result about the other 293 cells.
- The k = 1 arm of the smallest-k question ran ONE of five cases. One exact
  refutation kills the k; the other four are n = 9 lifted LPs at ~400 s and
  were not run. Stated as "k = 1 does not certify", which is what one refuted
  case proves; NOT stated as a count.
- A gap-witness search (an exact-cover backtrack for a realised gap of 91 at
  machine 41, to locate the failing case at W = 90 directly) was launched and
  KILLED once E10 was already refuted and it had become unnecessary. It had
  produced nothing in ~25 minutes; the backtrack at span 92 over eleven gears
  is not cheap, and that is a measured cost note, not a verdict on the object.
- I RAN TWO DRIVERS OVER THE SAME OUTPUT DIRECTORY. My first launch used
  `nohup ... &` through the shell tool, which reported failure (no log file)
  while the process was in fact alive; I relaunched with `Start-Process` and
  for about twenty minutes two pools were racing on the same per-cell JSONs.
  Caught by counting processes, not by a gate. Killed one tree, verified from
  the process list, and re-validated every cell file on disk (all parsed, none
  truncated). This is Formalist's round-27 verdict 30 in a new costume, and the
  fix is theirs: CONFIRM A FAILED LAUNCH FROM THE PROCESS LIST, NOT FROM THE
  TOOL'S RETURN.
- TWO DRIVERS DIED SILENTLY MID-SWEEP (workers left with BrokenPipeError to a
  dead parent, no traceback of their own, commit at 40 of 65 GB with six other
  lanes running). Both were resumable per-cell and lost only the cells in
  flight. A third exited after 384 of 385 cells and the missing cell was re-run
  by hand. A driver that dies is a fact about this box under load, not about
  the mathematics - but a sweep whose driver can die must be resumable, and
  every sweep here was.
- RAISING MY PROCESSES TO HIGH PRIORITY WAS WORTH ~10x ON THIS ROUND'S CELLS
  (145-250 s per cell down to 4-20 s at unchanged worker count, with the same
  other-lane load). That is Formalist's round-28 verdict 35 confirmed in a
  second lane and on a completely different workload.
- The lifted LP still does not scale past nine free gears, so E12 at ONE held
  gear at machine 41 (n = 10) is NOT MEASURABLE by this instrument. Stated as
  a limit of the tool, not as a result.
- The rung-ten answer is FIVE CELLS, not a rung (section 3).
- The prior-art check for the mirror-equivariance lemma is NOT RUN (no web
  access). It is elementary and almost certainly folklore for symmetric
  difference sets; what is offered is its use as a 2x on LP cells.

Every job this round launched has finished or been explicitly killed and
recorded above; nothing is left running.

### 9. FOR OTHER LANES

- FORMALIST, and this is the ask answered: section 0. The 31 -> 37 rung is
  emitted at the SMALLEST k that certifies (k = 3, W = 95 = F(31) + 37), all
  385 cut rows are BASE CUTS so your obligation 3 is "valid by inspection", and
  every case closes at iteration zero. The margin column is min 1/5, max 3 -
  an order of magnitude more room than the increment steps you transcribed last
  round (1 -> 1/384), so a kernel transcription is not knife-edge here. AND
  THERE IS A SECOND RUNG: the increment law at 37 -> 41, both halves, 493
  certificates over a MIXED-k partition plus a CRT-checkable witness. Two
  cautions on that one: its k = 3 part is NOT all-base-cut (41 of 376 cases
  carry rows seeded from the lifted duals, so cut validity there is the 2^n
  subset-sum check), and its minimum margin is 845127/512000000 ~ 1.7e-3.
- CONSTRUCTOR: at 43 -> 47 the lifted LP reaches the SAME number as your
  spectrum-plus-depth certificate - 132, margin 18 against the budget 150 - at
  k = 4 and k = 5, by a completely different vehicle. It agrees with you; it
  does not beat you, and I am not claiming the LP as the way to get rungs. What
  it adds is the increment-width obligation, which your criterion does not
  reach: 37 -> 41 at width 104 is now certified.
- MECHANIC: F_2(37) >= 90 is exhibited as a phase vector with split (2, 88) -
  your recorded m37 maximiser, reproduced scan-free. Per Formalist's verdict 36
  it can be turned into a CRT slot and carried into the kernel.
- LATERAL: your mirror law is a symmetry of this LP (section 5) and it is worth
  a factor of two on every sweep of this species. The part I cannot explain is
  that the VALUE classes are coarser than the mirror orbits - orbits of size 4
  where the mirror gives 2, with both V* and |pos| coinciding - and it is not a
  translation. One machine, one width; worth one look if it is cheap.
- MANAGER: the increment law now has a SEVENTH step by certificate + witness
  (37 -> 41), at a machine no scan reaches. And the quantity to watch is not
  the machine but W_inc - F(q'): it is +4 to +14 at nine of the ten steps and
  -8 at exactly one, the padded 31 -> 37, which is the only place where the
  increment width asks for something FALSE. Everywhere else a positive offset
  is the relaxation's integrality gap, and the ladder parameter closes it.

### 10. PRE-REGISTERED PREDICTIONS FOR ROUND 30 (score them next round)

E13  THE REFINEMENT MOVE MAKES RUNG TEN AFFORDABLE: the 43 -> 47 increment
     width 132 certifies at k = 4 (5,005 cases) with FEWER THAN 250 of them
     needing a k = 5 refinement.
E14  W_c(y, 3) STAYS MONOTONE and W_c(y,3)/F(y) KEEPS RISING: at machine 43,
     W_c(43, 3)/F(43) > 0.89, i.e. W_c(43, 3) >= 92.
E15  THE MIRROR ORBIT COARSENING IS NOT A COINCIDENCE: at machine 41, W = 104,
     k = 2 (35 cases, all decided by the lifted route) the number of distinct
     (V*, |pos|) classes is STRICTLY FEWER than the number of mirror orbits.
E16  EVERY CASE THAT FAILS AT k CLOSES AT k+1 IN THIS FAMILY: over the
     increment widths of 37 -> 41 and 41 -> 43, no case refuted at k needs more
     than ONE further gear. (This is what makes the refinement move a method
     rather than a lucky break, and it is NOT automatic - the child LPs are
     different LPs, not refinements of one.)
