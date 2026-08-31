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

**Not yet checked.**  Search terms for the manager: "Jacobsthal function spectrum consecutive
gaps upper bound adding a prime", "maximal gap growth sieve of Eratosthenes one more prime",
"Holt Rudd cycle recursion maximal gap bound".
