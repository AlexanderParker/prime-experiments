# The killer-word specification

Manager, round 29 (2026-09-02). Status: derived from the kernel iff (no new theorem), then
tested by research/killer_probe_r29.py (pre-registration
research/data/r29/manager_killer_prereg.md, log research/data/r29/killer_probe.log).

## 1. Why this document exists

The human's framing (2026-09-02): instead of proving (D) rung by rung, write down what a gap
word would have to DO, mechanically, to block twin-prime candidates for ever; then ask whether
the machine-building rules can produce such a word. This is the contrapositive of the (D)
route, but it is weaker than (D) - it tolerates occasional (D) failures - and it names the
quantity to watch.

## 2. The specification

From twins_infinite_iff_survivor_in_window: twins are finite iff there is a Q such that for
every machine M with largest gear q >= Q the word has no opening in the slot window
(q/6, W(M)], W(M) = (q_next^2 - 1)/6. Every survivor in that window is a twin prime pair.
Write g_0(M) for the origin gap (slot 0 to the first survivor k_0 >= 1). The killer word must
satisfy, at every rung past Q:

  K1  size          g_0(M) > W(M). Since g_0 <= F(M), the record must reach the window.
  K2  placement     the window-sized gap must be the ORIGIN gap, every rung, for ever.
  K3  mechanism     g_0 grows at rung q' only by fusion at the origin front: k_0 dies iff
                    q' | (6k_0 - 1)(6k_0 + 1); the interior openings killed in one rung
                    number at most A_kill <= 5 (uniform-order theorem) and are spaced by the
                    letters {2u', q'-2u', q'}, each <= q'.
  K4  accumulation  W grows by (q_next'^2 - q_next^2)/6 per rung; the killer must keep pace
                    on average, for ever, with no rung where it falls behind.
  K5  anatomy       the only (D)-violating configuration ever observed (counterfactual tooth
                    family) is a below-median parent whose incoming gear fuses a record window
                    of the shape [a, s_min, F_old, s_min, a'] (round 28, slack +1) or a deeper
                    fusion (this round, slack up to +6 at 17->19).

## 3. What the probe measured (exact, exhaustive where stated)

### 3.1 K1 is a CONSTANT-FACTOR fight, not a growth-rate fight

Real machine, every rung with a corpus record value:

    q      5    7   11   13   17   19   29   31   37   41   43   47   53
    F/W  .250 .250 .250 .229 .300 .284 .269 .254 .314 .295 .280 .252 .250

The record sits at ONE QUARTER of the reduction window, flat from q = 5 to q = 53. It is
not the case that the window runs away from the record; both are quadratic at these sizes
and the gap between them is a constant. (D)-uniform would force F/W <= ~3/ln q -> 0, but
slowly (0.76 at q = 53); the data at q <= 53 cannot distinguish "flat at 1/4" from "decaying
like 3/ln q". THE QUANTITY TO WATCH IS F/W. A rung where it rises toward 1 is the only kind
of rung that could ever hurt.

The counterfactual family says the same thing one notch worse: the worst tooth placement in
V(y) reaches F = 11, 25, 32, 43 at y = 11, 13, 17, 19 against W = 28, 48, 60, 88 - W/maxF =
2.55, 1.92, 1.88, 2.05. Even the extremal symmetric sieve with these gears sits at half the
window; the twin machine at a quarter. Neither is diverging from the window at these sizes.

### 3.2 K2 is essentially never satisfied, even in the family

Real machine, q <= 200: g_0(M) is the slot of the first twin prime above q (gate: 44/44).
g_0/W falls from 0.25 (q = 5) to 0.0053 (q = 191). The origin gap moved at 13 of 44 rungs,
and at EVERY rung that moved it, the incoming gear killed exactly ONE leading opening
(origin-front fusion depth = 1, never 2, against the arity ceiling 5). The origin is the
quietest place in the word.

Family: the number of members whose record gap IS the origin gap is 1, 1, 0, 0, 0 at
y = 7, 11, 13, 17, 19 (out of 6, 30, 180, 1440, 12960). Worst origin gap in the family:
7, 10, 15, 25 against W = 28, 48, 60, 88. Every violator's record (38 of 38 across the four
steps) is away from the origin.

### 3.3 K4: the one known violation mechanism does not chain - but it does grow

Single-rung (D) violators in the full family, step by step (m23 = all 142,560 members,
log research/data/r29/killer_probe_full23.log):

    step        family   violators   max slack   base rate   family max F   W
    7 -> 11        30        0          -5          0            11         28
    11 -> 13      180        1          +1        0.56%          25         48
    13 -> 17     1440        1          +1        0.07%          32         60
    17 -> 19    12960       36          +6        0.28%          43         88
    19 -> 23   142560      203         +11        0.14%          61        140

New this round: round 28's "exactly one violator, slack +1" was a small-step artefact. The
violator count and the worst single-rung slack both grow with the machine (+1, +1, +6, +11)
while the rate stays around 0.1-0.6%.

THE CHAINING TEST (P-K3/P-K6): every violator at step n was tested at step n+1 with every
possible next tooth - 1 + 1 + 36 violators, and at m23 the full family, where all 203
violators were checked against their parents. NOT ONE violator has a violating parent, and
not one has a violating child. Every m23 violator's parent was UNDER budget by 4 to 18.

TWO-RUNG SLACK (P-K4) - the pre-registered prediction "max <= +1" HELD at 7->13 (-4),
11->17 (-5), 13->19 (-3) and was REFUTED at 17->23: one member of 142,560,
(1,3,1,2,5,8,5), has slack -7 at 17->19 and +11 at 19->23, two-rung +4. Its lineage:
F = 5, 8, 13, 15, 27, 61 at m7..m23 - a MEDIAN m19 parent (27 against median 28) whose
child is the m23 family MAXIMUM (61). So the candidate lemma "two-rung (D) family-wide" is
dead: a single large violation can outrun a mildly under-budget rung. What survives is
strictly the chaining statement: consecutive violations do not occur (0 of 241 violators at
five steps), and every violation is preceded by an under-budget rung.

Why, mechanically: violating parents are at or below the family median (F_old 14..19 at
m17, median 19; 27 at the extreme m19 case, median 28), and violating children land at the
top of the next family (35..43 at m19, max 43; 61 at m23 = the maximum). A (D) violation is
a typical-or-better old machine whose next record jumps to the family's extreme - it is
measured against F_old, and an unremarkable F_old is what makes the budget beatable. The
child then carries a large F_old and the next rung is comfortably under budget.

The envelope E(y) = max_V F itself obeys (D) at four of five steps: E steps 11 -> 25 -> 32
-> 43 -> 61 give E(y') - E(y) - q' = +1, -10, -8, -5. The worst placement's record grows
slower than the budget once the machine has three gears.

## 4. Status of each condition

    K1  constant-factor: real 1/4 of window, family worst 1/2, both flat at q <= 53.
        NOT excluded by rate; excluded so far by the constant. Open in general.
    K2  never observed in 14,616 machines beyond m11; the real machine's origin front
        fuses one opening at a time at 13/44 rungs. Open in general (no theorem).
    K3  theorem-grade inputs (uniform-order A_kill <= 5; letter set). Proved.
    K4  chaining refuted at every testable depth (0 of 241 violators at five steps have a
        violating parent or child; 157,140 machines sieved). Two-rung (D) is NOT family-wide
        (one m23 member at +4). Single-rung worst slack grows (+1, +1, +6, +11).
    K5  single-rung anatomy: below-median parent + fusion to family top. Measured.

## 5. What this changes

(i) The proof target can be stated as F(M)/W(M) < 1 for all M, or even just
    g_0(M) <= W(M) - weaker than (D)-uniform, and monitored by a single ratio that is
    flat at 0.25.
(ii) The counterfactual family is a null model for CHAINING, and the answer is clean:
    (D) violations never follow one another (five steps, 241 violators, zero chains). The
    "two-rung (D) family-wide" lemma that this suggested was tested in the same run and is
    DEAD at m23 (one member at +4), so no teeth-insensitive two-step induction is available
    either. The surviving family-wide statement is weaker and must be phrased as a
    conditional: a rung that violates (D) is always preceded by a rung that under-ran it.
    Whether that is a theorem about symmetric two-teeth sieves or another small-machine
    artefact (as the "exactly one violator" claim turned out to be) is undecided - the
    violator count grows with the machine, so this is a rung-30 test, not a law.
(iv) The envelope E(y) = max_V F is a new object worth tracking: it obeys (D) at four of
    five steps with margins 5-10, i.e. even the extremal placement is under budget once the
    machine has three gears. If E obeys (D) uniformly the twin machine does too, trivially -
    and E is a covering-design extremum, the kind of thing sieve theory bounds.
(iii) Rungs stay evidence, not the theorem. This document is the spec the theorem must
    satisfy, not a proof of it.
