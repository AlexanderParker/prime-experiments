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

---

## Round 30

BRIEF: (a) price, then certify, the 47 -> 53 step at the manager's
W_inc = F(47) + 53 = 171 - the (D) rung the spectrum-plus-depth criterion
cannot certify (F_6(47) = 177 > 171); (b) the mirror halving as a theorem and
a script; (c) score E13-E16 and pre-register two new predictions about (a).

TERMINOLOGY, and a note on my own earlier usage: this round's W_inc is the
manager's F(M) + q' = 171.  Rounds 27-29 wrote "W_inc" for the STRICTER
increment-law width F_2(M) + s_min(q') = 134 + 18 = 152 at this step; that
width is not the target here and the two are kept apart below.

Pre-registration, written before any machine-53 LP was solved:
`research/data/r30/lp_prereg_r30.txt` (P1-P6, and the plan for E13-E16).
Scored in section 6.

### 0. THE PRICE TABLE (arithmetic only, before any solve)

    level  held           cases   mirror   self-mirror   |pos|          free  cols    links
                                  orbits   case          min/max/mean   n
    k = 3  (5,7,11)         385     193    (0,6,3)       55/63/59.96    11    56,124  3,530
    k = 4  (5,7,11,13)    5,005   2,503    (0,6,3,6)     45/55/50.74    10    51,691  3,060
    k = 5  (5,7,11,13,17) 85,085  42,543   (0,6,3,6,0)   37/51/44.77     9    46,183  2,584

Every level has exactly ONE self-mirror case (each q is odd, so
2w = 1 - W mod q has one solution), so the mirror halves every level to
(cases + 1)/2 orbits.  Predicted price (P4): 100-300 k exact ops per k = 3
case, 30-120 s a cell; the plan was a mixed-k tree rooted at k = 3.

### 1. THE ROOT MOVED FROM k = 3 TO k = 4, AND THE REASON IS THE FINDING

THE k = 3 PROBE.  Case (0,0,0) at k = 3 (n = 11 free gears, |pos| = 60): the
plain cut loop's LP maximum sits at 64.336 against the 60 it must fall below
and creeps 64.336 -> 64.223 over NINE passes and 1,066 s (STUCK at the budget).
The lifted LP - the instrument that would say whether that is convergence or an
asymptote - does not scale to n = 11.  So the k = 3 level is NOT affordable:
193 orbits at >= 1,000 s each with no decision at the end.

ONE GEAR DEEPER THE PICTURE INVERTS.  At k = 4 the SAME phases (0,0,0,0) give
a base-cut polytope that is INFEASIBLE at iteration zero - HiGHS returns
status 2 on the very first LP, the level-1 coverage rows plus block sums plus
consistency links exclude a fully blocked window of width 171 before the
recursion row or any degree-2 cut is consulted - and the common-slack LP's
duals close an exact certificate in ~20 s (LP 10.8 s, build 1.3 s, certificate
~5 s; 73-83 k exact ops; peak working set 192 MB).  Round 28 saw the same
inversion at machine 37 width 88 (k = 2 asymptote, k = 3 empty); here it is
between k = 3 "+4.3 and crawling" and k = 4 "empty before the recursion row".

So the tree is rooted at k = 4: 2,503 orbit representatives on four workers
at High priority, plain loop capped at 4 passes / 120 s, every refusal split
on gear 17 into 17 children at k = 5 (n = 9, where the lifted LP decides
exactly).

### 2. THE TREE AT 47 -> 53 - THE k = 4 LEVEL, AND THE k = 5 REFINEMENT

THE k = 4 LEVEL (`lp_tree_r30.py LEVEL 53 171 4 4 120`; log
`research/data/r30/lp_level53_k4.log`; 10,324 s wall on four workers after
the priority fix below, 45,713 s of summed cell time = 12.7 core-hours):

    representatives decided      2,503  (one per mirror orbit of 5,005 cases)
    CERTIFIED                    2,407  = 4,813 of 5,005 cases  (96.16%)
      at ITERATION ZERO, every row the base cut     2,398
      after 2 / 3 / 4 cut passes (seeded rows)      5 / 3 / 1
      exact ops per certificate   mean 77,771, min 70,199, max 160,651
      total exact ops             187,193,969 (representatives)
      wall per certified cell     14.9 s mean at High priority
      margin column               min 1/16384, max 3; 27 cases under 1/100
      the self-mirror case (0,6,3,6): CERTIFIED, margin 1
    REFUSED (NOCERT-SPLIT)          96  = 192 cases  (3.84%)

EVERY ONE OF THE 96 REFUSALS IS THE SAME OBJECT: the plain loop STUCK with
the LP maximum of the RECURSION ROW between 0.05 and 2.2 ABOVE |pos| at the
first solve and creeping down by 0.01-0.4 per pass (e.g. case (0,0,2,12):
47.960 -> 47.403 against 47 over twelve passes and 611 s; case (1,6,1,2):
45.836 -> 45.626 against 45 over four passes).  No refusal is a coverage cut
and none is an in-polytope refutation - the lifted LP that would decide
"asymptote or slow convergence" is out of reach at n = 10 - so the tight row
of every refusal is the recursion row, exactly as pre-registered (P3), and
the level-2 cuts are exhausted pass by pass without closing it.  A refusal
cost 103 s on average (max 742 s from the first driver's 12-pass budget).
And the refusals repeat VALUES across cases that are not mirror images: the
first-solve maximum 48.184 at (0,6,2,3), (1,0,3,4), (1,0,5,10), (2,1,4,5);
50.531 at five cases; 49.549, 47.314 and 47.960 at four and three each - the
same value-class coarsening round 29 saw at m37 and section 5 measures at
m41.

THE k = 5 REFINEMENT (children of the refusals on gear 17, canonicalised
under the mirror; none of the 96 refused representatives is self-mirror, so
96 x 17 = 1,632 children).  PRICED FIRST on a four-orbit sample: 68 children,
68/68 CERTIFIED at iteration zero, 7-10 s each, 66-82 k ops, margins 1/5 to
8/3, 143 s on four workers - so the whole refinement prices at ~57 minutes,
inside the manager's one-hour cap, and was run.  RESULT (log
`lp_refine53_k5.log`, 3,261 s wall for the remaining 1,564 cells on four
workers - 57 minutes in all, the price to the minute):

    k = 5 children decided        1,632  (96 refused orbits x 17 phases of gear 17)
    CERTIFIED                     1,632 / 1,632
      at ITERATION ZERO, every row the base cut      1,632
      exact ops per certificate   66,083 - 82,211 (mean ~71,000)
      total exact ops             115,934,896
      margin column               min 1/11, max 3
    REFUSED                           0

EVERY CHILD OF EVERY REFUSAL CERTIFIES, AND AT ITERATION ZERO.  So the
96 recursion-row refusals at k = 4 are all closed by one more held gear, and
not marginally: no child needed a cut pass, none reached the lifted LP, and
the k = 5 margins (min 1/11) are wider than the k = 4 level's (min 1/16384).
The tree is complete: 2,407 certified k = 4 representatives plus 1,632
certified k = 5 children, mirror-expanded, partition prod(Z_5 x Z_7 x Z_11
x Z_13 x Z_17) - asserted by the step manifest of section 2b - and

    F(53) <= 171 = F(47) + 53    (D) AT 47 -> 53, HYPOTHESIS-FREE, BY 8,077
    EXACT RATIONAL DUAL CERTIFICATES OVER THE PRIMES 5..53 (4,039 decided,
    4,038 mirror-transcribed) - THE RUNG THE SPECTRUM-PLUS-DEPTH CRITERION
    CANNOT CERTIFY, CLOSED BY LP DUALITY WITH NO A_kill ANYWHERE IN IT.

Total exact certificate ops over the tree: 303,128,865 on the decided
representatives (the transcribed half costs no ops).  Summed cell time
45,713 + 13,567 = 59,280 s = 16.5 core-hours; wall 2.9 h + 0.95 h on four
workers.

### 2b. THE STEP, EMITTED (`research/lp_emit_r30.py`; files in `research/data/r30/`)

    layout_47_53_k4.json / layout_47_53_k5.json     the two case-independent layouts
    cert_47_53_k4_h<w5>_<w7>_<w11>_<w13>.json       4,813 files (2,407 decided + 2,406 mirrored; (0,6,3,6) is its own mirror)
    cert_47_53_k5_h<w5>_..._<w17>.json              3,264 files (1,632 decided + 1,632 mirrored)
    manifest_47_53_k4.json / manifest_47_53_k5.json the levels, each with its MARGIN COLUMN
    manifest_47_53.json                             THE STEP MANIFEST - the partition, asserted
    research/lp_rungs_r30.txt                       the margin columns, human-readable

FORMAT LINE: one JSON per case, INTEGERS ONLY, schema lp-case-split-
certificate/2 = round 29's schema 1 made SPARSE (`frow_nz` and `nu_nz` list
the nonzero recursion-row coefficients and link weights by index into the
layout; `rows_base_cut_positions` + `base_cut` stand for the rows when every
row is the base cut; `expand_v1` in the emitter recovers schema 1 exactly and
the round-27 reference checker verifies the expansion unchanged), carrying
`pos`, `y`, `yff`, `frhs`, `lhs`, `rhs`, `margin` = rhs - lhs, `ops`,
`iterations`, and `mirror_of` on a transcribed file.  A dense schema-1 file
at n = 10 would be ~1 MB; the sparse one is ~33 KB.
GATE LINES: section 2c.

### 2c. GATES (all from clean processes)

    uv run python research/lp_emit_r30.py GATE 4 0.02
        k=4: PARTIAL SPLIT - 4813 of prod(5, 7, 11, 13) = 5005 tuples (the step manifest
             states the partition); 4813/4813 cases re-verified from JSON, lhs < rhs in
             EVERY case (2406 of them mirror-transcribed); margin column min 1/16384
             max 3; all rows base cut = False; reference checker agreed on 105 files GREEN
        k=5: PARTIAL SPLIT - 3264 of prod(5, 7, 11, 13, 17) = 85085 tuples; 3264/3264
             cases re-verified from JSON, lhs < rhs in EVERY case (1632 of them
             mirror-transcribed); margin column min 1/11 max 3; all rows base cut =
             True; reference checker agreed on 67 files GREEN
        STEP MANIFEST manifest_47_53.json: 4813 + 3264 cases, PARTITION ASSERTED over
             85085 leaves ({4: 81821, 5: 3264}); margin min 1/16384 max 3; 606,183,237
             exact ops (decided + transcribed)
        ALL ASSERTIONS GREEN [2556 s]
        (schema 2 expanded to schema 1; relaxation rebuilt from the primes at each
        file's OWN held phases; every cut row re-checked by the exact zeta transform;
        lhs / rhs / margin recomputed from the file's own integers; the round-27
        reference checker check_case_json re-run unchanged on every self-mirror case
        and a 2% random sample - 172 files)
    uv run python research/lp_mirror_r30.py GATE29 2      ALL ASSERTIONS GREEN [164 s]  (section 3)
    uv run python research/lp_mirror_r30.py GATE29T 1     ALL ASSERTIONS GREEN [142 s]  (section 3b)
    uv run python research/lp_score_r30.py                -> research/lp_r30_results.txt
    uv run python research/lp_emit_r30.py TXT             -> research/lp_rungs_r30.txt

Emission size: 152.5 MB at k = 4 (32.4 KB a file) + 83.2 MB at k = 5 (26.1 KB); the
manager decides what is committed.  Every job this round launched has finished or
been killed and recorded; nothing is left running.

A PROCESS FINDING WORTH EVERY LANE'S ATTENTION.  The first hour of the sweep
ran at ~2 cells a minute against the ~10 a minute the workers' own 20 s
cells allow.  The workers were at High priority; the DRIVER was not, and on a
box at 100% CPU (34 python processes of five lanes on 20 cores) the
Normal-priority parent could not dispatch tasks fast enough - the four
High-priority workers sat idle waiting for it.  Raising the driver (psutil,
from outside) took the rate to 11 cells a minute at once.  Round 28/29's
"raise the WORKERS to High" is only half the lever; the pool's parent must go
with them, and `lp_tree_r30.drive` now does it.  Also: my first driver was
killed and relaunched once (to cut the refusal budget from 12 passes to 4)
after confirming from the process list that every worker had exited; the
85 cells on disk were kept, the relaunch skipped them, and no two drivers
ever shared the directory.

### 3. THE MIRROR, MADE EXACT - THEOREM AND SCRIPT (`research/lp_mirror_r30.py`)

    THEOREM (mirror transcription).  Let the case-split relaxation at held
    phases ws have position set pos, columns (S, r), links, cut rows (i, lam),
    recursion row frow and rhs |pos|.  Define
        m_q(r) = (1 - W - r) mod q,   rho(i) = W - 1 - i,
        MIRROR(ws) = (m_q(w_q))_q,     pi(S, r) = (S, (m_q(r_q))_{q in S}),
    and pi on links by parent column (children permuted by v -> m(v), which
    the link sum does not see).  Then, with the round-29 lemma
    rho(hits(q, r, W)) = hits(q, m_q(r), W) applied gear by gear:
      (1) pos(MIRROR ws) = rho(pos(ws));
      (2) O_{pi(j)} = rho(O_j), so |O| is preserved;
      (3) frow(MIRROR ws)[pi(j)] = frow(ws)[j] - for pairs, max-cover over
          the lower gears is a function of the family of hit-restricted
          subsets of P, which rho maps bijectively;
      (4) cut validity is a condition on lam alone, so (rho(i), lam) is valid;
      (5) TRANSCRIPTION: rows' = [(rho(i), lam)], y' = y, yff' = yff,
          nu'[pi(t)] = nu[t] gives a'_{pi(j)} = a_j for every column, pi
          preserves blocks, so lhs' = lhs, rhs' = rhs, margin' = margin: an
          exact dual certificate of MIRROR(ws) with the same op count.  []

    COROLLARY.  Decide one representative per mirror orbit; transcribe the
    other member.  The self-mirror case is its own representative.

THE GATE, on the round-29 31 -> 37 rung (385 cases, k = 3, W = 95):
    uv run python research/lp_mirror_r30.py GATE29 2
    -> 385/385 transcribed certificates RE-VERIFIED from JSON alone
       (relaxation rebuilt from the primes AT THE MIRRORED CASE by the
       round-27 reference checker `check_case_json`, every cut row
       re-checked, lhs/rhs/margin recomputed); self-mirror case (3, 2, 8);
       ALL ASSERTIONS GREEN [164 s].
AND A FACT THE GATE TURNED UP: against the certificate the round-29 sweep
found INDEPENDENTLY for the mirrored case, the transcription has EQUAL margin
in only 261 of 385 cases, an equal op count in 1 of 385, and the identical
dual in 0 of 385.  The theorem says the mirrored case ADMITS a certificate of
the same margin; the float solver, run on the isomorphic LP, found a
DIFFERENT dual 124 times (the rounding grid in `certificate_star` is
path-dependent).  So the mirror is not "the solver would have found the same
thing" - it is a genuine second certificate, and a cheaper one.

### 3b. AND THE COARSENING ROUND 29 COULD NOT NAME IS A SECOND TRANSCRIPTION -
### THE TRANSLATION LEMMA

Round 29 measured the (V*, |pos|) classes of a sweep COARSER than its mirror
orbits (11 classes over 35 cases at m37 W = 95 k = 2) and wrote "it is not a
translation - no ws -> ws + t preserves V* except t = 0, tested at all 35".
This round's E15 data reproduced the shape (14 classes over 18 orbits at
m41 W = 104 k = 2) and the position sets of the eight-case class turned out
to be EXACT TRANSLATES of one another - as subsets of [0, W), not modulo
anything.  IT IS A TRANSLATION, with a boundary condition a test of "every
case" cannot see:

    THEOREM (translation transcription).  If pos(ws + t) = pos(ws) - t as
    subsets of [0, W) - i.e. the held gears block [0, t) at ws and
    [W - t, W) at ws + t (t > 0; symmetrically for t < 0) - then with
    rho(i) = i - t and m_q(r) = (r + t) mod q the five claims of the mirror
    theorem hold verbatim, and (rows - t, y, nu o pi_t^-1, yff) is an exact
    dual certificate of ws + t with the same lhs, rhs, margin and op count.
    (i in hits(q, r, W) iff i - t in hits(q, r + t, W) for i in pos, both
    endpoints inside the window; the lower gears' hit-restricted subsets of
    a pair overlap are mapped bijectively; cut validity is a condition on
    lam alone.)  []

GATED ON ROUND-29 DATA (`lp_mirror_r30.py GATE29T`): 484 translation
transcriptions from 330 of the 385 certificates of 31 -> 37 onto their
translate cases (shifts -3..3: 11, 44, 187, 187, 44, 11), EVERY ONE
RE-VERIFIED from JSON alone by the round-27 reference checker, with lhs, rhs
and margin equal to the source's.  ALL ASSERTIONS GREEN [142 s].

AND IT ACCOUNTS FOR THE COARSENING EXACTLY.  Classes of the case split under
the group generated by the mirror and the boundary-blocked translations
(arithmetic only, `research/lp_r30_results.txt`):

    sweep              cases   mirror orbits   mirror+translation classes   measured value classes
    m37 W=95   k=2       35        18                 11                      11  (round 29)
    m41 W=104  k=2       35        18                 14                      14  (E15, this round)
    m37 W=95   k=3      385       193                100                       -
    m41 W=104  k=3      385       193                125                       -
    m43 W=134  k=3      385       193                125                       -
    m47 W=132  k=4    5,005     2,503              1,243                       -
    m53 W=171  k=4    5,005     2,503              1,391                       -
    m53 W=171  k=3      385       193                125                       -

At both sweeps where the value classes were measured the count MATCHES the
mirror+translation class count, and at m41 W = 104 k = 2 every one of the 14
exact-translate pairs has equal V* (14/14) while the 19 non-translate
"phase exchanges" of E20 fail.  Round 29's open item is closed: the value
classes ARE the orbits of {mirror, boundary-blocked translation}.  THE
SAVING NOBODY USED THIS ROUND: at m53 W = 171 k = 4 the classes number
1,391 against the 2,503 orbits that were decided - a further 1.8x (3.6x
over the 5,005 cases), free for every future sweep of this species and
gated here on 484 certificates.  It goes into the tree driver next round;
this round's sweep was already running when the lemma was found.

### 4. E14, SCORED EARLY - AND THE PER-CASE FRONTIER HAS CROSSED THE TRUTH

W_c(43, 3) by the round-29 bisection (`research/lp_side_r30.py E14`, lifted
LPs at n = 9, 146-195 s each at High priority; sign pattern asserted width by
width at 103..109):

    y            23     29     31     37     41     43
    W_c(y, 3)    13     31     46     66     81    106
    F(y)         34     43     58     88     91    103
    W_c / F     0.382  0.721  0.793  0.750  0.890  1.029

E14 (W_c(43,3) >= 92) CONFIRMED - and by more than it asked: THE RATIO HAS
CROSSED 1.  At machine 43 the case-0 cell with three held gears is certifiable
only from width 106 on, i.e. NOT at the truth F(43) = 103, nor at 104 or 105
(G = +1.634, +1.341, +1.341 there; EMPTY from 106).  Round 28's "the per-case
reach is right at the truth at three held gears" was a machine-41 statement;
at machine 43 the k = 3 per-case frontier is three above F.  (Rung 41 -> 43
at W = 134 is unaffected - 134 is far above 106.)

### 5. E13, E15, E16 - TWO SCORED, ONE RUN FOR TEN MINUTES AND DROPPED, ONE NOT RUN

E15 (m41, W = 104, k = 2: strictly fewer (V*, |pos|) classes than mirror
orbits) - CONFIRMED.  All 18 orbit representatives decided by the lifted
route (`lp_side_r30.py E15`, 133-230 s each at n = 9; V* is a float reading
of the lifted optimum, the cells' verdicts are round 29's):

    (0,0) 50.166742   (0,1) 49.365180   (0,2) 50.964917   (0,3) 50.799994
    (0,4) 50.273260   (0,5) 50.383425   (0,6) 49.365180   (1,0) 49.365180
    (1,1) 49.123153*  (1,3) 50.034810*  (1,4) 49.841528*  (3,0) 50.799994
    (3,1) 49.937554*  (3,2) 49.365180   (3,3) 49.219114*  (3,4) 48.729009*
    (3,5) 49.510071*  (3,6) 49.083255*            (* = |pos| 44, else 45)

14 distinct (V*, |pos|) classes over 18 orbits (35 cases).  The coarsening
is again a class of FOUR orbits - (0,1), (0,6), (1,0), (3,2), i.e. the eight
cases {(0,1),(2,1),(0,6),(2,3),(1,0),(1,2),(3,2),(4,0)} - plus one pair of
orbits, (0,3) and (3,0).  Note the shape: (0,1) with (1,0), (0,3) with (3,0)
- the two held phases EXCHANGED - which is not a translation and not the
mirror.  For LATERAL, as last round: one machine, one width, a symmetry the
mirror does not generate.

E13 (43 -> 47 at W = 132 certifies at k = 4 with < 250 refinements) - NOT
RUN.  The 2,503-orbit sweep at m47 (n = 8) prices at ~10-14 core-hours
against a round already carrying the 13-core-hour k = 4 level above; not
started (job-completion rule).  Unscored, carried.

E16 (no case refuted at k needs more than one further gear, over the
increment widths of 37 -> 41 and 41 -> 43) - HALF ON RECORD, HALF DROPPED.
37 -> 41: on record (9 refusals at k = 3, all 117 children certified).
41 -> 43 at W = 117: a k = 3 tree driver (`LEVEL 43 117 3`) ran for ten
minutes on one Normal-priority worker, landed NO cell on the saturated box,
and was killed at the manager's commit cap; dropped for the round.  PRICE:
193 orbits at n = 9, ~15-30 s a certified cell by the fast path and ~150-200 s
a lifted decision on a refusal, plus the k = 4 children of the refusals -
2-4 core-hours; the same driver resumes it.  NAMED NEXT TARGET, and it is
also the eighth increment step by certificate if it closes.

### 6. MY OWN ROUND-30 PRE-REGISTRATION, SCORED (`research/data/r30/lp_prereg_r30.txt`)

  P1  (certified fraction at the FIRST k, i.e. k = 3, ABOVE 376/385 = 0.9766)
      REFUTED IN ITS PREMISE, AND BELOW IN ITS ANALOGUE.  The k = 3 level was
      not affordable at all (case 0: 1,066 s, STUCK, no decision) - so "the
      first k" became k = 4, and there the fraction is 4,813/5,005 = 0.9616,
      BELOW 0.9766.  The manager's second question is answered: BELOW.
  P2  (every k = 3 refusal closes at k = 4)  NOT TESTED as posed; the k = 4
      -> k = 5 analogue is CONFIRMED OUTRIGHT: 96 of 96 refusals close one
      gear deeper, 1,632 of 1,632 children, no k = 6 cell needed - E16's
      shape at the eleventh rung.
  P3  (the tight row at a refusal is the RECURSION ROW, and the smallest
      free gear carries it - at most 3 of 17 children refuse)  CONFIRMED on
      both clauses: 96/96 refusals are a stuck recursion row with the
      level-2 cuts exhausted, and 0 of 17 children refuse at every one of
      the 96 (the bound "at most 3" was loose by three).
  P4  (price)  a: 100-300 k ops per k = 3 case - NOT MEASURED (no k = 3
      certificate exists); at k = 4 it is 70-161 k, mean 78 k.  c: "30-120 s
      a k = 3 cell" - REFUTED (> 1,000 s and no decision).  d: "a k = 4 child
      costs 40-90 s; the whole tree < 10 core-hours" - the per-cell price was
      OVER-estimated (14.9 s certified) and the tree UNDER-estimated: 12.7
      core-hours for the k = 4 level alone plus ~4 for the k = 5 refinement,
      ~17 in all, because the root moved from 193 to 2,503 orbits.  REFUTED.
  P5  (>= 90% of certified cases at iteration zero with base cuts)
      CONFIRMED: 2,398 of 2,407 = 99.6%.
  P6  (margin min < 1/10, max >= 2)  CONFIRMED: 1/16384 and 3.
  (b) (every transcribed certificate re-verifies, equal margin and ops)
      CONFIRMED 385/385 at 31 -> 37 and on every emitted case of 47 -> 53.

### 7. PRE-REGISTERED PREDICTIONS FOR ROUND 31 (score them next round)

E17  53 -> 59 AT ITS (D) WIDTH 204 (machine 59, 15 gears): the k = 4 level
     (n = 11 free gears) is NOT affordable by the plain loop - fewer than
     half of a 50-cell probe certify at iteration zero - and the affordable
     root is k = 5 (85,085 cases, 42,543 orbits, n = 10).
E18  THE INCREMENT-LAW WIDTH AT 47 -> 53, F_2(47) + s_min(53) = 152, is a
     genuinely harder object than 171: at k = 4 the certified fraction of a
     100-orbit sample is BELOW 80%.
E19  W_c(47, 3)/F(47) > 1.03 - the k = 3 per-case frontier stays above the
     truth once it has crossed (43: 1.029).
E20  (written, then TESTED THE SAME HOUR on the 18 values already on disk,
     and REFUTED - recorded here rather than carried): "the value-class
     coarsening is the phase exchange V*(w5, w7) = V*(w7 mod 5, w5 mod 7)".
     It holds at 16 of 35 cases and fails at 19 - e.g. (0,4) = 50.273 against
     (4,0) = 49.365 - so the exchanges (0,1)~(1,0) and (0,3)~(3,0) are inside
     the coincidence, not its law.  The eight-case class is
     {(0,1),(0,6),(1,0),(1,2),(2,1),(2,3),(3,2),(4,0)}; no map of the held
     phases I can name generates it.  Replaced by:
E20' THE EIGHT-CASE CLASS IS NOT A COINCIDENCE OF THE FLOAT: rebuilding the
     four representatives' lifted LPs in exact rationals (one exact solve
     each) gives four EQUAL optima, and the |pos| = 45 position sets of the
     eight cases are pairwise DIFFERENT as subsets of [0, 104) (so the
     equality is not an isomorphism of position sets).
