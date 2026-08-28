# The scan-free certificate: requirement (D) at one step as a finite CRT computation

Status: SCRIPT-VERIFIED (finite, exact integers) at the four steps 19->23,
23->29, 29->31 and 31->37; gated against the round-24 full-period scan
dictionaries at the first three (agreement on every query, 100%). Established
round 25 (Constructor), executing the round-24 named next construct "the
scan-free dictionary and the chain".
Scripts: `research/crt_dict.py` (the oracle + its validation gate),
`research/scanfree_dict.py` (the dictionary, + its gate against the dumps),
`research/chain_cegar.py` (the certificate loop),
`research/chain_a4.py` (the exact-F engine),
`research/twogap_threshold.py` (the three-instrument measurement of section 5).
Prior-art check: NOT YET CHECKED (section 6).

## 1. WHAT IT IS

Plain language. Requirement (D) - the sole open input of the tolerance route -
says that adding one gear to a sieve machine cannot grow its largest gap by
more than the gear itself: `F(M + q') <= F(M) + q'`. Round 23 reduced (D) at
one step to a max-plus closure plus a finite list of yes/no facts of the form
"does machine M really have these four consecutive gaps?", and round 24
measured that list: 181, 90 and 955 questions at the three steps, answered
from a dumped list produced by scanning the machine's whole period. The period
is the obstruction: machine 31 has 3.34e10 slots, machine 37 has 1.24e12, and
the scan route was declared dead at ~170 h per rung.

This document records that those questions never needed the period. Each one
is a finite constraint-satisfaction problem over the gears, and answering them
that way turns (D) at a step into a computation whose only input is the LIST OF
PRIMES up to y.

**THE REALISABILITY CSP.** Slot `k` is blocked by gear `q` exactly when
`k = +-u_q mod q`, `u_q = 6^{-1} mod q`. By CRT a slot is the same thing as a
phase vector `(a_q)_q`, `a_q = k mod q`. So for a tuple of gap values
`(v_1, .., v_m)` with prefix-sum points `X = {0, v_1, v_1+v_2, ...}` and
interior points `Y = (0, span) \ X`:

**THEOREM (realisability = feasibility).** `(v_1,..,v_m)` occurs as `m`
consecutive gaps of `M = {5..y}` if and only if the following system has a
solution:

    (open)   a_q  not in  {+-u_q - x mod q : x in X}       for every gear q
    (cover)  for every t in Y there is a gear q with a_q = +-u_q - t (mod q).

Proof: immediate from CRT plus the definition of blocking - `k + i` is blocked
by `q` iff `a_q = +-u_q - i`. The period never appears; only `pi(y) - 2`
variables with domains of size `<= q` do. QED

The system is a set-cover with one variable per gear. `research/crt_dict.py`
decides it exactly by backtracking with (i) bitmask coverage, (ii) branching on
the uncovered point with fewest live options, and (iii) a CAPACITY BOUND (if
the unassigned gears cannot cover the still-uncovered points even at their
individually best phases, the node is dead). (iii) is what makes REFUTATIONS -
the only answers a certificate may act on - affordable.

**THE DICTIONARY.** Levels are built with the overlap lemma (R45: every
contiguous sub-tuple of a realised tuple is realised):

    D_1 = {v : realised},   D_m = {t + (v) : t in D_{m-1},
                                   t[1:]+(v) in D_{m-1}, realised}

so `F_j(M) = max span in D_j` and `D_3, D_4` are exactly the inputs the history
abstraction `A_4` of R49 needs.

**THE CERTIFICATE.** Put the CSP oracle where round 24 had the dump. The
counterexample-guided loop over the machine-free system `MF_4` (R52) then
proves `F(M+q') <= F(M) + q'` with no period, no dump and no given integer.
Every deletion is licensed by an exhaustive refutation, so the running bound is
an upper bound on `F(M+q')` at every stage, and an UNDECIDED query (node budget
exceeded) deletes nothing.

## 2. WHY IT MIGHT BE NOVEL

The CSP encoding itself is not new to this project - Mechanic's round-21
COV-SAT/COV-COUNT (`research/cov_sat.py`, `cov_count.py`) already decides a
single pattern this way, by SAT. What is new here:

* the whole certificate loop runs on it, so (D) at a step becomes a
  self-contained finite computation from the primes up to `y` - previously it
  needed one full-period scan per rung;
* the DICTIONARY is built as a level-by-level object, not one pattern at a
  time, which is what makes `F_j(M)` for all `j` and the abstraction `A_4`
  available without a period;
* consequently the LOW SPECTRUM of a wheel sieve (`F`, `F_2`, `F_3`, `F_4`)
  becomes computable at machines whose period is far beyond enumeration;
* and requirement (D)'s remaining content - the two-gap statement - acquires a
  purely combinatorial form (section 5) with no max-plus layer in it at all.

Classical shadow to check: Jacobsthal-function computations (Hagedorn;
Ziller-Morack) do search for maximal gaps in reduced residue systems, and
covering-system feasibility is a standard SAT/CSP encoding. The delta to check
is the two-teeth (paired) setting, the level-by-level dictionary, and the use
of the oracle inside an abstraction-refinement certificate.

## 3. PROOF / STATUS

SCRIPT-VERIFIED, exact integers, with the gates below.

**(a) The oracle is exact.** `crt_dict.py validate` checks the decision against
R43's independent pruned inclusion-exclusion COUNTER (`qualrun_zerocert.
pattern_count`) on every gap tuple of arity 1, 2 and 3 at machines 11, 13, 17 -
decision `==` (count > 0) in every case - and against nine published anchors
(the (8,15) word at m19, (10,21) at m23, the k=4 fuel word (10,21,10) at m29
and its zero partner (21,10,21), the m19/m23 holes at 24, the m37 depth-3 word
(14,41,14)).

**(b) It recovers the corpus ladder with no scan.** Same gate:

    machine   11  13  17  19  23  29  31  37
    F         7   11  18  25  34  43  58  88     all == corpus
    F_2      11   16  25  31  39  55  68  90     all == corpus / Mechanic

**(c) The dictionary contains the scanned one.** `scanfree_dict.py gate` at
machines 19, 23, 29: the scan-free `D_4` is a SUPERSET of the round-24
full-period dumps with ZERO missing tuples (361 / 962 / 3430 dumped, T3-filtered);
`F_1, F_2` match the corpus exactly; `F_1..F_4` = [25,31,35,38], [34,39,50,58],
[43,55,65,70].

**(d) The certificate reproduces round 24 exactly, query for query.**
`chain_cegar.py --shadow y` runs the CRT oracle and the round-24 dump side by
side and asserts agreement on every query:

    step        queries (arity 4 + arity 2)   bound   budget   agreement
    19 -> 23        106 + 75  = 181            48       48     181/181
    23 -> 29         28 + 62  =  90            63       63      90/90
    29 -> 31        761 + 194 = 955            74       74     955/955

- the SAME counts R58 measured with the dumped oracle, and no disagreement in
either direction.

**(e) The first rung beyond every dump.** `chain_cegar.py --step 31`:
31 -> 37 CERTIFIED, bound 95 <= budget 95, 3,399 queries, 356 s, oracle time
267 s. No scan and no dumped dictionary exists at this step.

**(f) The exact value, not just the budget.** `chain_a4.py`: `A_4` built over
the scan-free `D_3, D_4` returns `F(M + q')` EXACTLY at 19->23, 23->29, 29->31
(34, 43, 58), reproducing R49's numbers and, at machine 29, R49's exact system
size (3,513 edges).

## 4. IMPLICATIONS

Inside the project:

* the round-24 verdict "the obligation at a step is one finite dictionary of
  realisability facts" is upgraded from a MEASUREMENT to a COMPUTATION: the
  dictionary is generated, not looked up;
* the per-rung scan vehicle, declared dead at ~170 h for 29->31, is replaced by
  a minutes-to-hours CRT computation, and the ladder extends past every scanned
  machine;
* the input each rung consumes is only `(y, q', F(M))`, and `F(M)` is itself an
  output of the previous rung (`A_4` over the previous dictionary), so the
  ladder is self-propelling;
* every deletion the certificate makes is a small, independently checkable
  arithmetic refutation - the shape a kernel proof wants.

Outside: the low spectrum `F, F_2, F_3, ...` of a wheel sieve is computable
without enumerating the wheel, at sizes where enumeration is impossible.

## 5. THE TWO-GAP LAW IN COVERING FORM (and what still supplies it)

R58's slack sweep showed the obligation is EXACTLY `F_2(M) <= F(M) + q'`.
Section 1 turns that into a statement with no algebra in it:

**THE TWO-GAP LAW, COVERING FORM.** For every pair `(g1, g2)` with
`g1 + g2 > F(M) + q'`, the system "the three points `0, g1, g1+g2` open, all
`g1+g2-2` interior points covered by the gears `<= y`" is INFEASIBLE.

Three machine-free instruments act on that form; `research/twogap_threshold.py`
measures all three exactly.

* **CAPACITY** (the counting side): kills 5 of the 15 over-budget pairs at
  m23 and 0 of 3 / 78 / 231 / 1128 / 1176 at m19 / m29 / m31 / m37 / m41, with
  the worst-case ratio climbing 1.33, 0.98, 1.04, 1.11, 1.15, 1.20 - it kills
  only pairs with BOTH gaps near F and never the asymmetric ones. NOT a
  supplier, but not vacuous either (which the local form X12 did not
  distinguish).
* **THE FIRST MOMENT** (independence model, `rho = prod (1 - 2/q)`): the model
  gets the two-gap law RIGHT at every machine - model increment
  `F_2 - F` = 5, 6, 9, 11, 12, 14, 16, 18, 19 against `q'` = 13..43, ratio a
  flat 0.35-0.48; true increment 4, 5, 7, 6, 5, 12, 10, 2, 12, ratio 0.05-0.39.
  Unlike the histogram (X35) and the corridor (X34), which both saturate at 2F,
  independence does not saturate.
* **ASYMPTOTICS**: solving `E_1 = E_2 = 1` in closed form, the model increment
  is `~ log(F)/log(1/(1-rho))`, i.e. `O(log^3 y)` against a budget `q' ~ y`.
  Measured decay of `incr/q'`: 0.385 (y=11), 0.103 (1487), 0.047 (5261),
  0.0189 (20509), 0.0145 (29917). **In the first-moment model the two-gap
  statement gets easier without bound** - R31's "deeper cases are the easier
  ones", now for layer 0 and in closed form.

So the residue is sharply located: the two-gap law is comfortably true for a
FIRST-MOMENT reason with a polylog-vs-linear margin, and the missing step is
not a better inequality but an unconditional transfer of that first moment into
a bound - which no rearrangement-invariant (X35) and no congruence potential
(X32) can perform. That is exactly the extreme-value control of Wall V, now
with its scale named.

## 6. PRIOR-ART CHECK

Not yet checked (agent without web access). Terms for the manager:
"Jacobsthal function computation covering system SAT"; "maximal gap reduced
residue system CRT feasibility"; "counterexample-guided abstraction refinement
max-plus longest path"; "second largest gap sieve of Eratosthenes";
"admissible tuple realisability wheel sieve decision procedure".
Expected nearest art: Hagedorn's and Ziller-Morack's Jacobsthal computations
(one class per prime, and they enumerate rather than decide per pattern);
CEGAR in program verification (the technique, not this object); the project's
own COV-SAT (round 21, Mechanic) which decides ONE pattern this way - the delta
is the dictionary, the certificate loop, and the covering form of section 5.
