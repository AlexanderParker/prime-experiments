# The spectrum-plus-depth certificate for (D), and the ninth rung

Constructor, round 28.

---

## 1. WHAT IT IS

**Plain language.** To bound the new machine's record gap you have, until now, had to know
which runs of old gaps can actually merge - a word list, a flank envelope, and either a
full-period scan or a counterexample-guided loop with an expensive realisability oracle.
This criterion needs neither.  It needs two things about the OLD machine only: how long a
run of *j* consecutive gaps can be (its spectrum `F_j`), and *how deep a merge can go at all*
(one emptiness certificate).  If the spectrum stays under budget over that finite depth
range, (D) holds at the step.

**Precise form.**  Let `M -> M + q'` be a step, `Q*_J(M; q')` the word-legal J-window maximum
(defined in `per-j-window-analogues.md`), `F_j(M)` the j-th spectrum value (max sum of j
consecutive gaps), and

    J_max(M) = the largest J with a word-legal J-window   ( = A_kill(M) + 1 ).

> **THEOREM (spectrum-plus-depth certificate).**
>
>     F(M + q')  <=  max_{2 <= J <= J_max(M)}  F_J(M),
>
> so (D) at alpha = 3 holds at the step whenever that maximum is `<= F(M) + q'`.

**Proof.**  Three ingredients, all already established or elementary.

1. **R68's attainment theorem** (proved both ways, round 22; verified at eight steps):
   `F(M + q') = max_J Q*_J(M; q')`.
2. **`Q*_J <= F_J` by definition**: a word-legal J-window IS a window of J consecutive gaps
   of `M`, so its span is at most the j = J spectrum value.
3. **Emptiness is upward closed**: deleting either flank of a word-legal J-window leaves a
   word-legal (J-1)-window - the surviving middles are a sub-sequence of the old ones, so T2
   holds pointwise and T3's nonzero-class alternation is inherited.  Hence if no word-legal
   `J_0`-window exists, none exists at any `J >= J_0`, and `Q*_J = -inf` there.

Combining: the max over J is a max over `2 <= J <= J_max`, and each term is at most `F_J`. []

## 1.1 THE NINTH RUNG - the first application, and it closes a two-round stall

At `41 -> 43` the project has, or this round supplies, every ingredient:

    F_2(41)  = 103   EXACT      (R72, scan-free)
    F_3(41) <= 117   upper bound (max induced 3-sum of Mechanic's screened superset -
                                  a superset of the realised 3-tuples, so its max is an
                                  upper bound on F_3)
    F_4(41)  = 118   EXACT      (Mechanic, round 27, first computation, 602 core-s)
    Q*_5(41) = -inf  NEW        (every word-legal 5-window candidate refuted, by phase
                                 saturation or by an exact CRT decision, zero undecided)

    =>  F(43) = max_J Q*_J  <=  max(103, 117, 118) = 118  <  134 = F(41) + 43.

**(D) AT 41 -> 43 IS CERTIFIED, margin +16.**  This is the ninth rung of the (D) ladder,
left open by R72 (round 26), R79 (round 27) and by this round's own CEGAR attempt.  The
certificate uses no census of machine 41, no period, no CEGAR loop and no oracle stall.

Corollaries: `F(43) <= 118` (a first upper bound derived from m41 alone; the true value is
103), and **`A_kill(41) = 3` exactly** - `Q*_5(41) = -inf` gives `<= 3`, R45's realised
padded 2-words give `>= 3`.  This closes the project's open item O7 without needing
`F_3(41)` exactly.

## 1.2 THE CRITERION IS GENUINE - IT HAS A FAILING CASE

Run over every step whose spectrum is complete:

     M    q'  J_max | F_2  F_3  F_4  F_5 | bound  budget  verdict
    11    13     3  |  11   16   18   -  |   16     20   CERTIFIES  +4
    13    17     3  |  16   23   26   -  |   23     28   CERTIFIES  +5
    17    19     3  |  25   28   33   -  |   28     37   CERTIFIES  +9
    19    23     4  |  31   35   38   -  |   38     48   CERTIFIES +10
    23    29     3  |  39   50   58   -  |   50     63   CERTIFIES +13
    29    31     5  |  55   65   70   85 |   85     74   FAILS     -11
    31    37     5  |  68   85   90   92 |   92     95   CERTIFIES  +3
    37    41     4  |  90   97  105   -  |  105    129   CERTIFIES +24
    41    43     4  | 103  117  118   -  |  118    134   CERTIFIES +16

8 of 9.  The one failure, `29 -> 31`, is exactly the step where the deep layer is non-empty
AND the record-to-gear ratio is smallest: there `F_5(29) = 85` but `Q*_5(29) = 55`, thirty
under the spectrum, so the word-legality constraint is doing thirty units of real work.  That
is the content of the criterion - it discards legality, and at one step in nine that costs
too much.

**`F_5(29) = 85` is itself new** (first computation): the m29 spectrum is now
`43, 55, 65, 70, 85`.  Method: a realised 5-tuple has both of its 4-sub-tuples realised, so
Mechanic's exact 4-tuple census gives an overlap-filtered candidate set (428 tuples of sum
> 84); descending by sum and deciding each by `crt_dict.decide_cover` gives the maximum with
every larger candidate refuted, 0 undecided.  The four maximisers are two mirror pairs -
`(27,3,7,30,18)`, `(30,4,3,30,18)` and their reverses - another live confirmation of
Lateral's mirror theorem.

---

## 1.3 THE TENTH RUNG - 43 -> 47 (round 29), AND WHAT IT DOES *NOT* PROVE

Mechanic's round-28 shopping-list delivery makes the tenth rung immediate:

    F_2(43) = 116   EXACT, unconditional      (Mechanic r28)
    F_3(43) = 125   EXACT, unconditional      (the known value, reproduced)
    F_4(43) = 132   EXACT, unconditional      (Mechanic r28, new)
    J_max(43) = 4                              (A_kill(43 -> 47) = 3)

    =>  F(47)  <=  max(116, 125, 132) = 132  <  150 = F(43) + 47.

**(D) AT 43 -> 47 IS CERTIFIED, MARGIN +18** (`research/rung10_r29.py`, clean process, log
`research/data/r29/rung10_sweep.log`).  The gate re-asserts each recorded `F_J(43)` against
its own deletion-ladder cap before using it, re-derives `F` and `F_2` at m11..m19 from the
period, and checks the arithmetic.

**THE HYPOTHESIS LEDGER, in full.**  H1 merge law + T2/T3; H2 the attainment theorem; H3
`Q*_J <= F_J`; H4 upward-closed emptiness; plus the three `F_J(43)` values and
`A_kill(43 -> 47) = 3`.  And one more, which round 28 did not write down:

> **THE DELETION-LADDER CAP** (proved, three lines).  If `x_0 < ... < x_j` are consecutive
> openings of `M` and `q'_1..q'_{j-1}` are the next `j-1` primes, then by CRT some translate
> `x + t.P(M)` has `x_i + t.P(M) = c_i` mod `q'_i` for every interior `i` at once, so every
> interior opening is killed and `F_j(M) <= F(M + {the next j-1 primes})`.
> At m43: `F_2 <= F(47) = 118`, `F_3 <= F(53) = 145`, `F_4 <= F(59) = 161`.

Mechanic's three values are exhaustive *because of that cap* (their search ran to span 180).
The cap for `j = 2` is `F(47)`, and `F(47) <= 150` is what the rung asserts.  **So rung ten
is not a logically independent bound on `F(47)`** - and neither is any rung below m59, since
the corpus knows `F` outright there.  What a rung establishes is that the certificate's
obligation at that step is a *bounded, finite, old-machine-only* computation; rung ten
ratifies that and nothing more.  Recording this is the round's correction to the round-28
framing, which did not separate the two claims.

**THE PRICE OF THE INDEPENDENT VERSION, measured.**  Dropping the deletion cap leaves only
the machine-43-internal caps `F_j <= j.F(43)`, and the obligation "no word-legal `J`-window
of span in `[151, j.103]`" for `J = 2, 3, 4` is

    J = 2 :     812 candidates,    614 survive phase saturation
    J = 3 :  18,068 candidates,  7,948 survive
    J = 4 : 130,983 candidates, 29,510 survive
    TOTAL                       38,072 exact CRT decisions

at a measured 30-46 s per instance at a 300,000-node budget, at which budget only about a
quarter are decided at all.  That is the number, and it is why the independent version was
not bought this round.

## 1.4 THE ELEVENTH RUNG FAILS, AND THE FAILURE IS A_kill's (Mechanic, round 29)

`47 -> 53` does NOT certify.  `A_kill(47 -> 53) = 5` EXACT (the project's only 5-chain), so
`J_max(47) = 6`, and

    F_6(47) = 177   EXACT and unconditional   vs   budget F(47) + 53 = 171    FAILS by 6

`F_6(47) = 177` is a first computation: the floor-1 lap-phase transfer from machine 23 with
six new gears, seeded at 174 and capped at 290 = `2 F_3(47)`, which is at or above the
SUBADDITIVITY ceiling of every depth in range (`F_{a+b} <= F_a + F_b`), over 100% of machine
23's period (64 of 64 shards).  The maximiser is exhibited as a SLOT, not a phase vector:
machine 47, `k = 46,615,676,895,423,125`, seven consecutive openings at offsets
`[0, 42, 70, 103, 107, 115, 177]`, gap word `[42, 28, 33, 4, 8, 62]`, all 171 other slots of
the span blocked, re-checked at machine 47 from the definition.

**AND THE PATTERN IS NOT ABOUT THE MACHINE - IT IS ABOUT `A_kill`.**  Since `F_J` is
non-decreasing in `J`, the criterion's margin at a step is exactly

    margin(M -> q')  =  F(M) + q'  -  F_{A_kill(M -> q') + 1}(M),

so section 1.2's table sorts by `A_kill`, not by `M`:

    A_kill = 2 :  margins  +5, +9, +13
    A_kill = 3 :  margins  +10, +16, +18, +24
    A_kill = 4 :  margins  -11 (29 -> 31),  +3 (31 -> 37)
    A_kill = 5 :  margin   -6  (47 -> 53)

EVERY `A_kill <= 3` STEP CERTIFIES; both failures and the single `+3` squeaker are the
`A_kill >= 4` steps.  The mechanism is arithmetic, not statistical: one extra unit of
`A_kill` admits one more level of the `F` ladder, which costs 7-16 units (measured
increments `F_{J+1} - F_J`: m37 `[2,7,8,8,7]`, m41 `[12,7,8,10]`, m43 `[13,9,7]`,
m47 `[16,11]`), while the budget gains only `q' - q'_prev`, which is 4 to 6 at this end of
the ladder.  So the honest scope of the criterion is **"it certifies exactly the steps whose
fuel census is shallow"**, and since `A_kill` is arithmetic-selected and not monotone (C10),
it will fail again at the next `A_kill >= 4` step.  This is a scope statement, not a
refutation: section 1.2's "8 of 9, one exception" reads as an exception because eight of the
nine steps happen to have `A_kill <= 3` or a large `q'`.

What still closes `47 -> 53` is the word-legal half: R68's attainment theorem plus the corpus
value `F(53) = 145` gives `max_J Q*_J(47) = 145 <= 171`, margin 26 - i.e. word-legality is
doing 32 units of work at this step, the same role it plays at `29 -> 31` (30 units).
Cheapest next test of the reading: `53 -> 59`, where `A_kill = 4`, `J_max(53) = 5`, budget
`F(53) + 59 = 204`, and `F_4(53)`, `F_5(53)` are not on record.

Gate: `uv run python research/gate_mechanic_r29.py` (sections D and E);
table: `research/criterion_margin_r29.py`; witness: `research/witness47_r29.py`.
Status: **script-verified, exact integers, exhibited witness** (no float anywhere).

## 2. WHY IT MIGHT BE NOVEL

The criterion replaces the whole word/flank apparatus by "spectrum, over a capped depth
range".  R31 conjectured a version of this ("corrected flatness": `F_j - F <= q' + lambda(j-2)L`)
with a heuristic depth-dependent correction; the correction is not needed - what is needed is
that the depth RANGE terminate, which is a combinatorial fact about the two-tooth alternation
and can be certified with no census.  R52 showed the machine-free wall sits in the two-gap
statement; this criterion sits on the other side of that wall (it consumes exact `F_j`), and
its content is that above `J_max` there is nothing to consume.

---

## 3. STATUS

| statement | status | pointer |
|---|---|---|
| the certificate theorem | **PROVED** from R68 + two elementary lemmas | this file |
| upward-closedness of emptiness | **PROVED** (one line from T2/T3) | this file |
| `Q*_5(41) = -inf` | **SCRIPT-VERIFIED**, exact, scan-free, 0 undecided | `research/perj_scanfree.py --y 41 --J 5 --floor 0`; `research/rung9_perj_cert.py --recheck` re-derives it from scratch |
| `A_kill(41) = 3` | **SCRIPT-VERIFIED** (with R45's lower bound) | as above |
| `F_5(29) = 85` | **SCRIPT-VERIFIED**, exact, scan-free, 0 undecided | `research/rung9_perj_cert.py` header; overlap filter + descending CRT |
| the 9-step table, incl. the failure at 29->31 | **SCRIPT-VERIFIED** | `research/rung9_perj_cert.py`, log `research/data/r28/rung9_perj_cert.log` |
| (D) at 41 -> 43 | **CERTIFIED** | as above |

All `F_j` inputs are asserted against the corpus record inside the script before use, and the
m41 spectrum is recomputed from Mechanic's files rather than quoted.

---

## 4. IMPLICATIONS

* The (D) ladder is now certified through **41 -> 43**, nine rungs, with 43 -> 47 and
  47 -> 53 already true by arithmetic (`F(47) = 118 <= F(43) + 47 = 150`;
  `F(53) = 145 <= 171`) and 53 -> 59 decided by Mechanic in round 27.
* **The next rung's shopping list is explicit and short**: to certify `43 -> 47` by this route
  one needs `F_2(43), F_3(43), F_4(43)` (none on record) and an emptiness certificate at
  `J = J_max(43) + 1`.  The emptiness half is the cheap half - a 5- or 6-point CRT pattern has
  small gear domains - so the cost is the spectrum, which is Mechanic's lane and their existing
  vehicle.
* It explains why the CEGAR loop stalled: the loop was reconstructing the spectrum bound
  edge by edge from a 1.7M-state abstraction, while the actual missing fact was a single
  yes/no question about depth 5.

---

## 5. UNSOLVED QUESTIONS IT TOUCHES

Requirement (D) and the tolerance route (R14/R26); the Jacobsthal-type spectrum `F_j` of the
two-dimensional sieve; project items O7 (closed) and O2.

---

## 6. PRIOR-ART CHECK

**Checked 2026-09-03 (harvester, round 30).  Verdict: PARTIAL OVERLAP in the one-class
shadow - the two ingredients "a new gap is a sum of consecutive old gaps whose interiors are
removed" and "the CRT step that kills any prescribed interior" are Holt-Rudd's recursion and
their Theorem 2.3 - and NOVEL AS FAR AS SEARCHED for the certificate itself:
`F(M+q') <= max_{2 <= J <= J_max} F_J(M)` with a FINITE depth cap supplied by a
combinatorial fact about the added gear, and the `A_kill` scope statement (section 1.4).**

| item | exact statement | source | relation |
|---|---|---|---|
| Holt-Rudd Lemma 2.1 (the cycle-of-gaps recursion) | "The cycle of gaps `G(p_{k+1}#)` is derived recursively from `G(p_k#)`. ... R2. Concatenate `p_{k+1}` copies of `G(p_k#)`. R3. Add adjacent gaps as indicated by the elementwise product `p_{k+1} * G(p_k#)`" | arXiv:1408.6002, p. 5 (READ); same as Lemma 2.1 of Holt arXiv:1510.00743 (READ) | the ONE-class merge law: every gap of the next stage is a sum of consecutive gaps of the current stage.  It is the source of `Q*_J <= F_J` in that setting.  Two classes per prime are not treated. |
| Holt-Rudd Theorem 2.3 | "Each possible closure of adjacent gaps in the cycle `G(p_k#)` occurs exactly once in the recursive construction of `G(p_{k+1}#)`" (proof: CRT - "Exactly one of these [`p_{k+1}` copies] has residue 0 mod `p_{k+1}`") | arXiv:1408.6002, p. 8 (READ) | the CRT step of the DELETION-LADDER CAP (section 1.3) and of the attainment theorem, in one-class form.  Iterating it over the next `j-1` primes is exactly `F_j(M) <= F(M + next j-1 primes)`; Holt-Rudd do not draw that corollary. |
| Holt-Rudd Lemma 3.1 / Cor. 3.2 | for a constellation of length `j` and sum `g < 2p_{k+1}`, the `j+1` closures "occur in distinct copies", so each instance yields `p_{k+1} - j - 1` intact copies, `j-1` interior closures of length `j-1`, and two exterior closures that "increase the sum" | arXiv:1408.6002, pp. 11-12 (READ) | the threshold `g < 2p_{k+1}` is precisely the regime where no two interior points are removed in one copy - the one-class `A_kill = 1` regime.  Above it Holt-Rudd give no depth statement.  The certificate lives above it (spans `F_J(M) > 2q'` routinely) and its depth cap `J_max = L + 2` is what replaces their hypothesis. |
| Holt 2015, section 7.2 remark | "Initially the largest gap in `G(13#)` is `g = 22`; the gap `g = 52` is first created in closures by `p = 73` and this continues to be the largest gap through the rest of this process" | arXiv:1510.00743, p. 44 (READ) | an EMPIRICAL record-gap remark about the survival process of one cycle; no bound. |
| Ziller 2020 Prop. 2.7 | `m in D(k) => m in D(k+1)` | arXiv:2007.01808, p. 7 (READ) | the CONVERSE direction to the deletion cap at arity 1: old gaps persist.  See `dictionary-monotonicity-onset.md` section 6. |
| Hagedorn 2009 | `h(n)` for `n < 50` by backtracking with an a-priori capacity bound | Math. Comp. 78 (2009) - NOT OBTAINED (two HTTP 403s, re-tried 2026-09-03) | one-class computation; SECONDARY. |
| Costello-Watts 2015; Iwaniec 1978 | explicit / asymptotic upper bounds on `j(n)` at dimension 1 | Math. Comp. 84 (2015) 1389-1399; Demonstratio Math. 11 (1978) 225-232 (lane record) | bounds on the ONE-class function, none by a spectrum-over-depth criterion. |

NONE FOUND: any published inequality bounding the maximal gap after adding a prime by the
`j`-consecutive-gap spectrum of the previous stage over a bounded depth range; any statement
that the depth of a merge is capped by a dictionary quantity of the old sieve.  Searches
run: "Jacobsthal function spectrum consecutive gaps upper bound adding a prime"; "Jacobsthal
function upper bound adding a prime recursion g(p_{k+1}#) g(p_k#) Hagedorn Holt maximal gap
sum of consecutive gaps"; "maximal gap growth sieve of Eratosthenes one more prime"; "Holt
Rudd cycle recursion Jacobsthal maximal gap next prime".
