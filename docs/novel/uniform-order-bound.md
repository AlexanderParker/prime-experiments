# A uniform bound on the truncation order of the qualifying-run algebra

Round 27 (constructor).  Gate: `research/uniform_order.py` -> "all assertions passed"
(log `research/data/r27_uniform_order.log`).

## 1. WHAT IT IS

Plain language.  The project's certificate chain proves requirement (D) at a step
M -> M + q' by closing a finite max-plus system A_m whose state is the last m-1 gap
VALUES of the old machine.  The system only bounds anything if it is NILPOTENT
(acyclic); if it is cyclic it says "infinity" and certifies nothing.  Which order m
is enough had been a MEASUREMENT at nine machines and nothing more - R67 named it
"(i) THE ORDER: nothing says which m makes A_m nilpotent at machine M", and R25/R26
both closed asking for "a bound, any bound, valid at every machine".

This note supplies one for the candidate cycle the project has always used as the
order's proxy - the pure alternation - and shows, sharply, that no bound of the same
kind exists for the others.

Precise form.  For the step M = {5..y} -> q' = nextprime(y) put u' = round(q'/6),
a = 2u', b = q' - 2u' (the two literal letters, R40/T1).  Define

    A_relax(M) = min{ m : one of the two m-letter ALTERNATIONS
                          (a,b,a,...) / (b,a,b,...) is NOT realised as m
                          consecutive gaps of M }.

THEOREM (uniform alternation order).  For every machine M with y >= 7,

    A_relax(M) <= 5,

and A_relax(M) <= 4 unless q' = 37, 53, 83, 127, 157 or 173 (mod 210).  For the one
smaller machine, M = {5} (q' = 7), A_relax = 1 directly.  The bound is proved by
PHASE SATURATION at gears 5 and 7 alone, so the whole statement is a function of
q' mod 210 with no machine in it.

The six exceptional classes are EXACTLY R20's litcap-6 classes - the classes at which
the literal chain cap is 6 rather than 2, 3 or 4.  That is not a coincidence: litcap
is the longest alternation whose prefix-sum walk stays in the corridor E mod 35, and
by CRT a translate of a point set fits inside the exposed sets of gears 5 and 7
separately iff it fits inside E mod 35.  Phase saturation at {5,7} and the literal cap
are the same arithmetic seen from two sides; the only difference is that litcap
maximises over the two starting letters where the order minimises.

THE COMPANION NEGATIVE, and it is the sharper half.  A_relax tests ONE candidate
cycle.  A_m is nilpotent only when EVERY legal cycle is broken, and padded letters
(gaps = 0 mod q', i.e. q', 2q', ...) are T3-transparent, so cycles mixing padded and
literal letters are legal too.  Define

    CORRCAP(q', F) = the longest T3-legal word with all values <= F whose
                     prefix-sum walk stays inside the corridor E mod 35
                     - the strongest cap gears 5 and 7 can EVER give.

Measured exactly (part 7 of the gate):

    step        F(M)   F/q'   CORRCAP        step        F(M)   F/q'   CORRCAP
    19->23        25    1.1      4           37->41        88    2.1     25
    23->29        34    1.2      2           41->43        91    2.1     25
    29->31        43    1.4      3           43->47       103    2.2     11
    31->37        58    1.6      5           47->53       118    2.2      5
                                             53->59       145    2.5   INFINITE

and INFINITE at every larger ratio tested (F/q' = 3.3, 4.5, 7.0, 13.7).  So:

  PHASE SATURATION CAPS THE ORDER ONLY WHILE F(M)/q' IS SMALL.  The first step at
  which gears 5 and 7 impose no cap at all is 53 -> 59, and since F(M)/q' grows
  without bound along the chain (F is Jacobsthal-scale, q' ~ y), NO fixed set of
  small gears can ever cap the order again.  The alternation is the last cycle the
  corridor can kill.

## 2. WHY IT MIGHT BE NOVEL

- The bounded object is the truncation order of a max-plus (tropical) transfer system
  whose alphabet is the gap word of a primorial covering system.  The bound is not on
  a gap, a chain length or a count but on HOW MUCH HISTORY a sound abstraction of the
  covering dynamics needs - a statement of automata-theoretic shape proved by residue
  arithmetic.
- The pigeonhole underneath (Mechanic's phase saturation) is elementary and its
  novelty is claimed only for the object.  What is new here is (i) that the pigeonhole
  is a CLOSED FORM in q' mod 210 for the order, (ii) that its exceptional set is
  exactly the litcap-6 set - identifying two independently discovered arithmetic
  invariants of the machine as one - and (iii) the sharp threshold F/q' at which the
  method dies, which is a statement about when an exposure argument stops seeing a
  covering argument.
- It is NOT the standard "a prime p covers at most L/p positions" bound: the content is
  an AVOIDANCE condition (does a gear have any admissible phase at all), and it fires
  at k ~ q/2, far below the covering threshold.

## 3. PROOF

STATUS: SCRIPT-VERIFIED (finite and exact) for the computational parts; the theorem
itself is a two-line argument on top of Mechanic's phase-saturation theorem.

(a) PHASE SATURATION (Mechanic, round 26, docs/novel/phase-saturation-arity.md).  Gear
g blocks slots k = +-c_g (mod g), c_g = 6^{-1} mod g, so a word with prefix-sum offsets
X occurs at slot k0 only if k0 + X lies inside E_g = Z_g \ {+-c_g}.  If no translate of
X fits inside E_g for some gear g of M, the word occurs nowhere.  (Note this uses only
the EXPOSED half of the realisability CSP; the COVER half - every interior point
blocked - is not used, so the refutation is a fortiori valid for "m consecutive gaps".)

(b) THE CLOSED FORM.  X for the m-letter alternation is {0, a, q', q'+a, 2q', ...}, so
X mod g is determined by (a mod g, q' mod g); and 3a = q' -+ 1 with the sign fixed by
q' mod 6, so a mod g is determined by q' mod 6g.  With g in {5, 7} everything is
therefore a function of q' mod 210.  The gate enumerates all 48 invertible classes:

    PS-order 2 : 24 classes    PS-order 4 :  2 classes  (23, 187)
    PS-order 3 : 16 classes    PS-order 5 :  6 classes  (37, 53, 83, 127, 157, 173)

so the maximum over all classes is 5, which is the theorem.  Adding gears 11 and 13
(moduli 2310 and 30030) refutes NOTHING further - checked exhaustively; the six classes
are stable.  Independently, the gate sweeps every prime q' < 20000 directly (no classes)
with all gears of M up to 100 and reproduces the same distribution and the same
exceptional residues.

(c) GATES.  The refuter reproduces phase-saturation-arity.md's alternation-ceiling
column exactly at all eight steps 31->37 .. 61->67 (in Mechanic's own convention: the
phase starting with s = 2*6^{-1} mod q'), and refutes NONE of the seven words the
project has on its realised record with machine-verified witnesses.

(d) A_relax EXACTLY, at nine machines, from full-period data (part 1): direct cyclic
scans at m11..m23, Mechanic's exact 4-tuple censuses at m29/31/37, and R45's exact CRT
pattern count at m41.  The result CORRECTS THE PUBLISHED LADDER:

    machine   11 13 17 19 23 29 31 37 41
    A_relax    1  2  2  3  2  3  4  2  2      (this round, all from data)
    R45 table  1  2  2  3  2  3  4  3  2      (m37 entry was an assumption)

R45's `arity_ladder.py` hardcodes "the 1- and 2-letter alternations are realised" at
machines 29, 31 and 37 instead of looking them up.  At m37 that is false: gear 5 refutes
(14, 27) by phase saturation, and the exact m37 dictionary confirms it is absent.
A_relax(M) <= PS-order(q') is asserted pointwise at all nine machines.

(e) CORRCAP is a longest-path/cycle computation on 30 states (corridor residue x tooth)
with the legal values <= F as steps; INFINITE means the graph has a cycle.

## 4. IMPLICATIONS

INSIDE THE PROJECT.
- R67's residue (i) - "nothing says which m makes A_m nilpotent" - is answered for the
  alternation and answered NEGATIVELY for the general case, which is more useful than
  either half alone: the chain's order is not going to be bounded by congruence
  arithmetic, and the search for a uniform order should be aimed at the COVER half of
  the realisability CSP, not the exposure half.
- The identification of the six exceptional classes with the litcap-6 classes means the
  project has one arithmetic invariant here, not two.  q' = 37 mod 210 is the class of
  the machine's only known padded step (31 -> 37) AND the class where the order is
  hardest - the same six residues that R20 found capping literal chains at 6.
- A_relax is NOT the order the chain needs (see the measured N below), so the corrected
  table changes a proxy, not a certificate.  Nothing filed depended on the m37 entry.

OUTSIDE.  If the tropical-abstraction route is written up, this is the statement that
says how much memory the abstraction needs and where that question stops being a
congruence question - the honest boundary of the elementary method.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

- THE REAL ORDER.  Define N(M) = smallest m at which A_m is acyclic.  Then
  max(2, A_relax) <= N <= A_res.  Measured exactly this round: N = 2, 2, 2, 3, 2, 3, 4, 3
  at m11..m37.  The folk identity N = max(2, A_relax), which held 7 of 7 in R49, is
  REFUTED at the eighth machine: A_relax(37) = 2 but N(37) = 3, and the extra order is
  bought by a PADDED cycle (the legal value set at m37 is {14, 27, 41, 55, 68, 82} and
  41, 82 are transparent to T3).  Whether N is bounded at all is OPEN and is now the
  correctly-posed form of R67(i).
- Is A_relax <= 4 uniformly after all?  At the only litcap-6 machine in the corpus,
  m31, A_relax = 4 in fact - the 4-letter alternation (12,25,12,25) is absent from the
  exact census - but the refutation there uses the COVER half, which no bounded gear set
  can supply uniformly (gears 5 and 7 can cover at most ~1.37 q' of the 2q' - 5 interior
  points of a 4-letter alternation, and larger machines make covering EASIER, not
  harder).  So a uniform 4 will not come from this method.
- Jacobsthal/Polignac: the order is the depth parameter of the merge law
  F(M+q') <= max_{j <= order} Q*_j(M), so any uniform bound on it removes a measured
  input from the project's route to the twin-prime reduction.

## 6. PRIOR-ART CHECK

Not yet checked (no web access in this lane).  Terms for the checker: "nilpotency order
tropical transfer matrix covering system", "memory order max-plus automaton sieve gap
word", "admissible phase avoidance residue pigeonhole Jacobsthal", "alternating deletion
chain covering system order", "litcap primorial residue class 210 chain cap".  Expect
PARTIAL OVERLAP on the pigeonhole (standard) and on max-plus longest-path theory
(classical); the claim of novelty is the closed form for the order, the identification
with the litcap classes, and the F/q' threshold at which the method dies.
