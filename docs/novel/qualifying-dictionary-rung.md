# The qualifying-window dictionary: a (D) rung whose certificate is the size of a dictionary, not of a period

Status: KERNEL-CHECKED given one named census hypothesis, at TWO consecutive
steps (`proofs/Machine31.lean`, `theorem D_29_31`; `proofs/Machine37.lean`,
`theorem D_31_37`; axiom footprint `[propext, Classical.choice, Quot.sound]`,
and the twelve dictionary checks `Machine29.D2_ok .. D7_ok` and
`Machine31.D2_ok .. D7_ok` depend on NO AXIOMS AT ALL). The census inputs are
SCRIPT-VERIFIED exactly over machine 29's full 1,078,282,205-slot period and
machine 31's full 33,426,748,355-slot period by `research/qual_dict.py`,
four-way gated by `research/qual_dict_gate.py` and
`research/qual_dict_gate31.py` (both GREEN). Build green at 1410 jobs, zero
sorries, no `native_decide`. Established round 25 (Formalist).
Prior-art check: NOT YET CHECKED (section 6).

ROUND-26 ADDENDUM (Formalist): **THE CENSUS HYPOTHESIS IS NOW A ONE-PERIOD
CLAIM.** `Census29` and `Census31` were stated as `∀ n` - claims about EVERY
index of an infinite gap word - while the gates verify ONE PERIOD. The step
between them was an unstated assumption; it is now a theorem
(`proofs/Machine29Cen.lean`, `proofs/Machine31Cen.lean`,
`proofs/LadderPeriod.lean`, build green at 1426 jobs):

    theorem Machine29.census29_of_period (h : Census29P) : Census29
    theorem Machine31.census31_of_period (h : Census31P) : Census31
    theorem LadderPeriod.D_29_31_period (h : Census29P) (n) : g31 n <= 43 + 31
    theorem LadderPeriod.D_31_37_period (h : Census31P) (n) : g37 n <= 58 + 37

`Census29P` is `Census29` with every clause restricted to `opSeq29 n <= P` -
the 214,708,725 openings of one period, exactly the finite object
`research/qual_dict.py` scans. The engine is `Periodic.index_reduce`: every
index's forward gap word is the forward gap word of an index whose opening
lies in the first period. It needs only that the opening PREDICATE is periodic
(one `omega` per gear) - NO walk and NO base case - which is why it works at a
machine whose period a kernel will never enumerate.

Verdict 21 still stands: the census is not kernel-checked. What changed is
that the unverified part is now finite as well as explicit.

## 1. WHAT IT IS

Plain language. Requirement (D) at a step `M -> M + q'` says every gap of the
new machine is at most `F(M) + q'`. Every rung of the ladder so far has been
paid for by a full period scan of the OLD machine - 37,182,145 residues at
19->23, and round 24 measured the next one at about 170 hours. This document
replaces the period by a DICTIONARY: a list of 15,860 short integer tuples
that is all the certificate ever reads, and over which a kernel decides the
rung in minutes.

The reduction has three steps, and the third is the one that makes the object
small.

**(a) The merge law needs only two facts about the old machine.**
`MergeLaw.newgap_le_step` (R39, kernel-checked since round 22) turns
`F_2(M) <= B` and `Q_j(M; a) <= B` for all `j >= 3` into "every gap of
`M + q'` is at most `B`", where `a = 2u''` is the tooth floor of the gear
after `q'`, and `Q_j(M; a)` is the QUALIFYING spectrum - the largest sum of
`j` consecutive gaps of `M` whose `j-2` INTERIOR gaps all reach `a`.

**(b) The qualifying spectrum is a statement about short windows only.**
`Q_j(M; a)` quantifies over windows, and a window that violates it is a
`j`-tuple of consecutive gaps. So the whole input is the family

        D_j(M; a) = { (g_n, ..., g_{n+j-1}) : g_{n+1}, ..., g_{n+j-2} >= a }

of REALISED qualifying `j`-windows, one set per depth.

**(c) The family is finite and short, because qualifying runs terminate.**
Let `K` be the longest run of consecutive gaps of `M` that reach `a`. Then
`D_j = empty` for `j > K + 2`, so `Q_j` is vacuous beyond that depth and the
family has `K + 1` members. `K` is small and does not grow with the machine
the way the period does: measured 3, 4, 5 at machines 19, 23, 29.

The result at the live step:

    machine 29, floor a = 10 (gear 31's teeth are {5, 26}), budget
    B = F(29) + 31 = 43 + 31 = 74, period 1,078,282,205 slots / 214,708,725 gaps

        j        2      3      4      5      6      7      8
        |D_j|   730  3,692  6,688  3,915    789     46      0
        Q_j      55     65     68     71     71     71      -

    15,860 tuples in all; max_j Q_j = 71 <= 74, margin 3.

**THE THEOREM.** With `Census29` naming the single unproved input - that
those six lists CONTAIN every realised qualifying window of machine 29, and
that no six consecutive machine-29 gaps reach 10 -

```lean
theorem D_29_31 (h : Machine29.Census29) (n : ℕ) : Machine31.g31 n ≤ 43 + 31
```

Everything else is kernel-checked: the six dictionary bounds by
`decide +kernel` with an EMPTY axiom footprint, machine 29's and machine 31's
own opening enumerations, the teeth, the containment, machine 29's
enumeration completeness, and the merge law.

**THE COST STATEMENT, which is the point.** Round-24 verdict 17 measured the
per-rung period-scan vehicle at this step at about 170 hours of kernel time.
This certificate builds, from a cold `.olean` set, in **under five minutes**
(`Machine29D4`, the largest module at 6,688 tuples, is 55 s; the rung file
itself 13 s). The certificate's size is the size of the dictionary, and the
dictionary is 68,000 times smaller than the period.

## 2. WHY IT MIGHT BE NOVEL

The individual ingredients are not new: bounding a longest run by a finite
set of realised local patterns is the standard "transfer matrix / subshift of
finite type" move, and the merge law and the qualifying spectrum are this
project's own earlier results. What appears to be new is the combination and
what it buys:

* the **termination of qualifying runs** (`K` small, measured 3/4/5) is what
  makes the local-pattern description FINITE at all - without the floor `a`,
  the analogous dictionary would have to describe every window and would be
  the period again. The floor is supplied by the NEXT gear's teeth, so the
  object that makes the certificate finite comes from one step further up the
  ladder than the step being certified;
* consequently the certificate at a step has size `O(sum_j |D_j|)`, which is
  governed by the tail behaviour of the gap distribution above `a` and not by
  the primorial. Empirically 990, 2,911, 15,860 tuples at machines 19, 23, 29
  against periods of 3.8e5, 3.7e7, 1.1e9 slots - a growth rate of roughly
  5x per gear against 30x per gear for the period;
* it separates a Jacobsthal-type upper-bound proof into a CENSUS half (finite,
  independently checkable, no logic) and a LOGICAL half (kernel-checked, no
  arithmetic beyond addition), with an explicit finite interface between them.
  That is the shape round-24 verdict 15 asked for, and it makes the
  unverifiable part of the proof exactly one list.

The honest shadow to check: run-length / cluster expansions for admissible
tuples in sieve theory, and subshift-of-finite-type descriptions of gap
sequences of periodic sieves.

## 3. PROOF / STATUS

KERNEL-CHECKED (Lean 4 + mathlib, zero sorries, no `native_decide`, no
`Lean.ofReduceBool`), build green at 1392 jobs:

    proofs/Machine29D2..D7.lean   Dj_ok : Dj.all (fun t => Nat.ble (sum t) Qj)
                                        = true       -- NO AXIOMS AT ALL
    proofs/Machine29Q.lean        opSeq29_surj       -- enumeration, no scan
    proofs/Machine29Dict.lean     spectrum29_two : SpectrumBound g29 2 55
                                  qual29_all : ∀ j ≥ 3, QualBound g29 5 j 71
                                  criterion_29_31 : max 55 71 ≤ 43 + 31  -- no axioms
    proofs/Machine31.lean         D_29_31 : g31 n ≤ 43 + 31
                                  g31_le_of_census : g31 n ≤ 71

SCRIPT-VERIFIED, exactly, over the full period (`research/qual_dict.py`), and
gated four ways (`research/qual_dict_gate.py`, ALL FOUR GATES GREEN):

1. CHUNK INDEPENDENCE - the 1,078,282,205-slot period is scanned twice with
   unrelated chunk sizes (40,000,000 and 23,456,789); all six dictionaries,
   the whole `F_j` ladder and the run length come out identical. A window
   straddling a chunk junction (mechanic's standing rule 18, which cost that
   lane a round) cannot survive this: the two runs put their junctions in
   entirely different places.
2. CYCLIC SEAM - the gap word is closed into a ring explicitly and the gap
   count is asserted equal to `prod (q - 2) = 214,708,725`.
3. TRANSCRIPTION - the Lean literals are parsed back out of the `.lean` files
   and compared as sets with the scan; all six identical.
4. CORPUS AGREEMENT - the same scanner run at machines 19 and 23 reproduces
   `F_j(19) = 25, 31, 35, 38`, `Q_j(19; 8) = 31, 35, 37, 38`,
   `F_j(23) = 34, 39, 50, 58, 65, 77, 83, 88` and
   `Q_j(23; 10) = 39, 43, 50, 55, 60`, every one of which is a kernel-checked
   theorem in this ledger, and `F(29) = 43`.

NOT PROVED, and named: `Census29` itself. It is a full-period claim; no Lean
kernel has seen machine 29's period and none is asked to.

TWO INDEPENDENT CONFIRMATIONS FALL OUT OF THE SAME SCAN, and they matter
because they were open or corrected elsewhere in the project:

* `F_2(29) = 55` exactly - the literal integer Constructor's `A_5(23)`
  survivor closure produces with no machine-29 scan, and Mechanic's
  full-period pair census. Three independent routes, one integer.
* `Q_J(29; 10) = 55, 65, 68, 71, 71, 71` for `J = 2..7` - EXACTLY the
  CORRECTED marked spectrum of round-24 verdict 12c, entry for entry,
  including the `J = 5` value 71 whose published predecessor (85) was the DP
  artefact that had made the 29->31 rung look lost. The correction is now
  confirmed by a fourth route, and it is the number the rung stands on.

## 4. IMPLICATIONS

Inside the project:

* it climbs (D) to a SIXTH rung, 29->31, the first past the wall round-24
  verdict 17 measured, and it does so in five minutes rather than 170 hours;
* it makes every remaining rung a CENSUS problem rather than a kernel problem.
  What Mechanic (or anyone) must deliver per rung is now one finite list plus
  a run bound, in a format an independent implementation can reproduce;
* it retires the "each rung needs its own period scan" reading of round-22
  verdict 9. That verdict said no function of the old machine's MARGINAL data
  supplies the next rung's input, and it stands - but a dictionary is not
  marginal data, and a dictionary is enough;
* it gives the `hE : realised ⊆ E` shape of verdict 15 its first instance,
  and shows that the right `E` is not the 4-tuple dictionary (45,854 rows,
  and its qualifying subgraph has CYCLES, so no potential over it is finite -
  measured, `research/a4_potential.py`) but the STRATIFIED qualifying family,
  which is three times smaller and acyclic by construction.

Outside: an upper bound on a Jacobsthal-type maximal gap whose machine-checked
part is a few thousand integer additions, with the empirical part isolated in
a list that anyone can regenerate. The general statement - a periodic sieve's
merged maximal gap is certified by the finite set of its qualifying windows,
whose size is governed by the run length above the next prime's tooth floor
rather than by the primorial - is the transportable part.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

Requirement (D) and the twin route (R14/R26); Jacobsthal `g(n)` and the
paired `h_2` of Ziller-Morack (the `Q_j` ladder is a refinement of `h_2` that
is not tabulated anywhere we know); the project's arity law (this certificate
does NOT have a fixed arity - its depth is `K + 2` and `K` grows, 3, 4, 5,
which is the arity law appearing again, now as the dictionary's depth).

Open, and named rather than answered:

* does `K` (the longest qualifying run) stay small? It is 3, 4, 5 at
  machines 19, 23, 29 with floors 8, 10, 10. If `K` grows like `log`, the
  dictionary family stays cheap forever; if it grows like the machine, this
  vehicle dies too, and at a predictable place.
* the certificate reads the dictionary but proves nothing about it. The
  dictionary-transfer of Mechanic (`research/dict_transfer.py`) computes a
  certified SUPERSET of a machine's tuple dictionary from the previous
  machine's, so `Census29` is in principle derivable from `Census23`. Chaining
  that - so that ONE census at a small machine underwrites every rung above
  it - is the construct this document does not build.
* `Q_j` saturates at 71 for `j = 5, 6, 7` at machine 29 (and at 38 for
  `j = 4, 5` at machine 19). Saturation of the qualifying spectrum before it
  goes vacuous is unexplained and looks structural.

## 6. PRIOR-ART CHECK

Not yet checked (agent without web access). Terms for the manager:
"subshift of finite type gap sequence sieve"; "transfer matrix Jacobsthal
function upper bound"; "run length admissible tuples primorial sieve";
"certificate for the longest run in a periodic residue-covered sequence";
"formal verification Jacobsthal function Lean"; "local pattern dictionary
bound longest gap wheel sieve". Expected nearest art: Holt-Rudd's cycle
recursion (one class per prime, no qualifying floor, so no analogue of the
run-termination that makes this finite), Hagedorn's and Ziller-Morack's
computations (values, not certificates), and the general SFT/transfer-matrix
literature. The delta to check is the qualifying floor supplied by the NEXT
gear and the resulting finiteness of the certificate.

## 7. ROUND-25 ADDENDUM: THE SEVENTH RUNG, AND WHAT THE SECOND DATA POINT SAYS

The vehicle was applied a second time, at 31->37, the round it was built. That
is the useful test of a template, and it produced one structural surprise.

    machine   period (slots)   floor a   |D_2..D_{K+2}|    K    max_j Q_j   budget
    19             378,675        8            990        3        38         48
    23          37,182,145       10          2,911        4        60         63
    29       1,078,282,205       10         15,860        5        71         74
    31      33,426,748,355       12         43,185        5        91         95

**THE DICTIONARY GROWS ~3-5x PER GEAR WHILE THE PERIOD GROWS ~30x.** That is
the quantitative form of the claim in section 1, now with four points:
990, 2,911, 15,860, 43,185 against 3.8e5, 3.7e7, 1.1e9, 3.3e10.

**`K` DID NOT GROW FROM 29 TO 31.** The open question of section 5 - does the
longest qualifying run stay small? - has its first informative data point: `K`
is 3, 4, 5, 5. `K` sets the DEPTH of the family (`j` runs to `K + 2`), so if
it stays bounded the vehicle survives arbitrarily far; if it grows like the
machine, the vehicle dies at a predictable place. Two consecutive machines at
`K = 5` is weak evidence, but it is evidence, and the floor `a` rose from 10
to 12 between them, which is the effect that should suppress `K`.

**THE QUALIFYING SPECTRUM IS NOT MONOTONE.** At machine 31,

    Q_j(31; 12)  =  68,  85,  90,  91,  90,  88     for j = 2..7

- it rises to a peak at `j = 5` and then FALLS. At machines 19, 23 and 29 the
qualifying spectrum was non-decreasing and then saturated (31,35,37,38 /
39,43,50,55,60 / 55,65,68,71,71,71). Machine 31 is the first case in this
project where it turns over before going vacuous. The consequence is not
cosmetic: **the constraint that binds the 31->37 rung is a FIVE-gap window
with three qualifying interiors** - not the two-gap statement (68, well under
budget) and not the deepest non-vacuous window (88). Any argument that treats
the two-gap statement as the whole obligation, or that assumes the binding
depth is the last one, is false from machine 31 onwards.

There is a plain mechanism to test: as `j` grows the interiors are forced to
be large (each at least `a`) AND numerous, and past some depth the arithmetic
that permits many large gaps in a row also forces the two END gaps to be
small. `Q_j` turning over is that trade crossing. Whether the turnover point
is a function of `a` and `F` alone is open and cheap to measure - it needs
only the dictionaries this vehicle already produces.
