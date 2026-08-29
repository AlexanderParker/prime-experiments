# kleene-generator - the merge law's Kleene star: F(M+q') as one tropical identity, and (D) as one arity-free certificate

Status: SCRIPT-VERIFIED (exact, full period) - the identity is asserted against the known
F(M+q') at all six scannable consecutive steps 11->13 .. 29->31 in
`research/kleene_generator.py`; the certificate equivalence and the finite-state
abstraction ladder are computed in the same script with assertions.  Established round 22
(constructor); the HISTORY LADDER of section 4b added round 23
(`research/kleene_history.py`, `research/machinefree_cert.py`), which overturns round 22's
"no bounded state certifies at 29 -> 31".  THE CERTIFICATE DIRECTION IS KERNEL-CHECKED since round 22 (formalist,
proofs/Potential.lean): `Potential.IsPotential` is the three one-step clauses,
`Potential.D_of_potential` proves that a potential bounds every legal chain of every
length (abstract over the state type, so Constructor's (opening, tooth) states are
covered), and `Potential.merged_le_of_potential` states it in the gap-word vocabulary -
its conclusion is (D)'s merged window `g a + windowSum g (a+1) l + g (a+l+1) <= F + q'`
and NO hypothesis of it quantifies over `l`.  A CONCRETE POTENTIAL IS EXHIBITED at
19->23 (proofs/Potential19.lean): `h19` = the qualifying tail unfolded three deep,
`h19_C1/C2/C3` discharged by machine 19's kernel ladder alone - (C2) holds with equality
in every branch, its deepest branch IS `Machine19.no_big_run` (Q_6 = 0), and (C3)'s four
cases are exactly F_2..F_5 <= 31, 35, 38, 47 against the budget 48.  So the certificate
form is not vacuous, and the recipe is generic: at any machine with a Q_J = 0 the tail
function unfolded to depth J-2 is a potential and (C3)'s cases are that machine's own
spectrum ladder.  NOT kernel-checked: the Kleene identity itself (an equality, needing
max-plus matrix machinery and the machine's K) and the CONVERSE direction (a potential
always exists - the place where nilpotency of K is used, h being the least
super-solution).  Prior-art check: not yet checked (agent without web access).

## 1. WHAT IT IS

Plain language.  Round 21 showed that adding a gear is an exact Kronecker recursion
whose new maximal gap is the NILPOTENCY INDEX OF A SUM of two Kronecker products, and
that no function of "marginal" data bounds that index - the bound is a joint
realizability statement whose ARITY (how many consecutive kills you must look at before
the chain is refuted) moves around with the machine.  That looked like an infinite ladder
of per-arity rules.  It is not.  The whole family collapses into ONE equation, because
the index of the sum is a KLEENE STAR - the max-plus analogue of (I - K)^{-1} - of a
single nilpotent tropical operator built from the old machine's gap word.  A Kleene star
is by construction a statement about all orders at once, so the arity never appears in
the statement; it appears only as the number of terms that happen to be nonzero when you
expand the star at a particular machine.

Precise form.  Let M be a machine with cyclically ordered openings o_1 < ... < o_n and
gaps d_i = o_{i+1} - o_i.  Let q' be the next gear, c = 6^{-1} mod q', u' = round(q'/6),
letters a = 2u' and b = q' - 2u' (the two nonzero tooth-difference classes, T1 of
two-teeth-kill-spacing.md).  On the state space

    STATES = { (i, s) : i an opening index, s in {+, -} the current tooth }

define the max-plus matrix K and two flank vectors (max-plus: (+) = max, (x) = +):

    K[(i,s), (i+1,s')] = d_i     if d_i mod q' in {0, a, b} and (s -> s') is the T3
                                 transition of d_i's class:
                                     class 0 (padded): s' = s        (transparent)
                                     class a:  legal only from s = -, lands s' = +
                                     class b:  legal only from s = +, lands s' = -
                       = -inf    otherwise
    L(i)   = d_{i-1}   (left flank)          R(i,s) = d_i   (right flank)

THEOREM (KLEENE IDENTITY).   With K* = (+)_{m >= 0} K^m the max-plus Kleene star,

        F(M + q')  =  L^T (x) K* (x) R      -  an IDENTITY, not a bound.

K is nilpotent (K^A = 0, A = the fuel arity k_max), so K* is a finite sum; but the
identity names no truncation depth.  Its m-th layer L (x) K^m (x) R is exactly the
depth-(m+2) qualifying window maximum qualmax_{m+2}, so one algebra generates every
layer of the ladder R39 wrote out by hand.

COROLLARY (TROPICAL DUAL CERTIFICATE - the arity-free form of (D)).  (D) at alpha = 3,
i.e. F(M+q') <= F(M) + q', holds at a step IF AND ONLY IF there is a potential
h : STATES -> Z with

    (C1)  h(i,s) >= d_i
    (C2)  h(i,s) >= d_i + h(i+1,s')     for every legal qualifying transition
    (C3)  d_{i-1} + h(i,s) <= F(M) + q'

Necessity: h = K* (x) R works.  Sufficiency: any super-solution of (C1)-(C2) dominates
K* (x) R entrywise, by induction on the star.  NO DEPTH INDEX APPEARS ANYWHERE in
(C1)-(C3): each is a one-step, one-opening inequality.  This is max-plus (tropical) LP
duality for the longest-path problem that F(M+q') is.

COROLLARY (nilpotency additivity, arity-free).  In R41's recursion
B_new S_new = (B_M S_M) (x) S' + (E_M S_M) (x) (B' S'), the second summand is nilpotent
of index 2 and the first of index F(M).  The index of the sum is therefore not bounded
by any function of the two indices (round 21's counting boundary), but it IS given
exactly by L^T (x) K* (x) R.  Nilpotency additivity is arity-free: the arity is a
property of how many terms of the star are nonzero, not of the statement.

MEASURED, the same script (exact, full period, machines 11..29):

  step        index(K)   L (x) K* (x) R   F(M)+q'   margin      layer maxima
  11 -> 13        2          11  = F(13)     20      +9  0.69q'   [11, 8]
  13 -> 17        2          18  = F(17)     28     +10  0.59q'   [16, 18]
  17 -> 19        2          25  = F(19)     37     +12  0.63q'   [25, 25]
  19 -> 23        3          34  = F(23)     48     +14  0.61q'   [31, 33, 34]
  23 -> 29        2          43  = F(29)     63     +20  0.69q'   [39, 43]
  29 -> 31        4          58  = F(31)     74     +16  0.52q'   [55, 58, 55, 55]

index(K) equals the fuel chain length k_max at every step, and h is always the LEAST
super-solution (every state tight) - the certificate is exactly saturated.

## 2. WHY IT MIGHT BE NOVEL

- Kleene stars of max-plus matrices are classical (Cuninghame-Green; Baccelli et al.,
  *Synchronization and Linearity*), and "longest path = max-plus closure" is textbook.
  What is not classical is the OBJECT: the theorem says that the maximal gap of a sieve
  after adding one prime is exactly the max-plus closure of a two-state-per-opening
  automaton read off the old sieve's gap word, with the automaton's transition rule
  supplied by a residue law (T3) rather than by the geometry.  The identity is what
  turns "Jacobsthal-type growth" into a shortest/longest-path problem.
- The dual certificate (C1)-(C3) is, as far as the project has looked, the first form of
  the increment law in which every clause is a ONE-STEP inequality.  All earlier forms
  (R21 word identity, R31 suppression-corrected flatness, R39 qualmax criterion) quantify
  over a depth index j and therefore over an unbounded family of statements.
- It converts the project's open input into a *witness-existence* problem (exhibit h),
  which is the standard shape a proof of such a law would take, and it is exactly the
  covering-LP-duality idea the project flagged as untested - in its tropical rather than
  linear form.

Honest shadow: for a FIXED machine this is a finite computation and the identity is
"just" longest path.  The content is (i) that the merge law makes the automaton exist at
all (the T3 alternation is what makes K a partial map rather than a general graph), and
(ii) that the certificate form removes the depth quantifier from (D).

## 3. PROOF

(<=)  By the merge law, every gap of M + q' is either a gap of M or the merge of a
MAXIMAL run of k >= 1 consecutive M-openings o_i, ..., o_{i+k-1} all killed by q'.  Its
length is d_{i-1} + (d_i + ... + d_{i+k-2}) + d_{i+k-1}.  Two consecutive killed
openings sit on teeth {+-c} mod q', so their spacing is 0 or +-2c mod q' (T2), i.e. the
interiors qualify; and the class transitions are forced (T3): +2c only from -c, -2c only
from +c, 0 keeps the tooth.  Hence the run is exactly a K-path of length k-1 starting at
(i, s) where s is o_i's tooth, and its length is L(i) + (K^{k-1} (x) R)(i,s).

(>=)  Conversely let P be any K-path of length k-1 from (i,s).  q' is coprime to the old
period, so among the q' CRT copies of the old period inside the new period there is one
in which o_i ≡ s·c (mod q'); by T3 the whole path is then killed, so the k openings are
all deleted and the new gap containing them is AT LEAST L(i) + (K^{k-1} (x) R)(i,s).
Taking the maximum over paths and using that any strictly longer realised run is itself a
path, the two maxima agree.  (The k = 1 layer already gives F_2(M) >= F(M), so surviving
old gaps never attain the maximum.)                                                  []

Certificate corollary: (C1)-(C2) say h is a super-solution of h >= R (+) K (x) h; the
max-plus Kleene star is the least such (standard, and re-verified numerically here); so
(C3) for some super-solution implies (C3) for the star, which is (D); and conversely the
star itself is a super-solution.                                                     []

SCRIPT: `research/kleene_generator.py` (dense; identity asserted against KNOWN_F at
machines 11, 13, 17, 19, 23 full period - log `research/data/kleene23.log`) and
`research/kleene_stream.py` (segmented, ~300 MB; machines 23 and 29 - log
`research/data/kleene_stream_23_29.log`).  The two implementations agree digit for
digit at machines 19 and 23.  Super-solution and leastness asserted; layer maxima
printed.
Supporting kernel results already exist for the ingredients: T1-T5 in
`proofs/TwoTeeth.lean`, the merge law and residue necessity in `proofs/MergeLaw.lean`
(`interior_gap_mod`).  The identity itself is NOT yet kernel-checked; at a fixed machine
it is a finite integer statement and is a natural Formalist target.

## 4. IMPLICATIONS

Inside the project:
- (D) becomes "exhibit h".  Every earlier form of (D) quantified over a depth j; this one
  does not.  The route's sole open input now has a witness shape.
- The arity question is answered in the only way that matters: nilpotency additivity IS
  arity-free, so an arity-free vehicle exists even though the measured arity is not
  monotone (arity_ladder.py: k_max = 2,2,2,3,2,4,4,3 at machines 11..37 - it goes down
  as well as up, tracking the added gear's arithmetic, not the machine's size).
- The finite-state question is now sharp and measurable.  Replace h by a function of a
  BOUNDED local state and the certificate becomes machine-free.  Measured (same script,
  sound class-level max-plus closure, so every "bound" below is a genuine upper bound on
  F(M+q')):

    step        value only        (phase 35, value)   (phase 385, value)   budget
    11 -> 13    11  certifies     11  certifies       11  certifies         20
    13 -> 17    21  certifies     21  certifies       20  certifies         28
    17 -> 19    30  certifies     28  certifies       28  certifies         37
    19 -> 23    CYCLIC (vacuous)  45  certifies       42  certifies         48
    23 -> 29    60  certifies     60  certifies       45  certifies         63
    29 -> 31    CYCLIC (vacuous)  99  FAILS by +25    99  FAILS by +25       74

  The value-only abstraction is CYCLIC exactly where the infinite alternating word
  survives the 2-point relaxation (machines 19 and 29) - R41's counting boundary, now
  visible as "the abstract operator is not nilpotent", and a non-nilpotent tropical
  operator bounds nothing at all.  Adding the corridor phase mod 35 restores nilpotency
  at both, and at 19 -> 23 it also certifies (D): that is the corridor-resonance carrier
  (R42) doing proof work rather than statistical work.
  ROUND-22 NEGATIVE, NOW OVERTURNED (round 23, see section 4b): at 29 -> 31 no
  CORRIDOR-PHASE state certified - mod 35, 385 and 5005 gave 99, 99, 91 against a budget
  of 74 (exact 58).  The diagnosis in that round ("the crude sound edge weight; a long
  chain paired with a large flank that never co-occur") was right, and both defects are
  removed by putting GAP HISTORY in the state instead of more corridor phase.

## 4b. THE HISTORY LADDER - the bounded state that does certify (round 23)

Define, for m >= 2, the m-POINT HISTORY ABSTRACTION A_m: the state of opening i is the
tuple of the last m-1 gap values (d_{i-m+2}, ..., d_i), optionally together with the
corridor phase of o_i mod 35 / 385, and the tooth.  An EDGE exists exactly when the
m-tuple of consecutive gaps it encodes is REALISED somewhere in the period and the T3
transition of the middle gap's class is legal.  Because the gap value now lives in the
state, three things become exact that were maxima over a class in the round-22
abstraction: the edge weight (= d_i), the base R (= d_i) and - for m >= 3 - the LEFT
FLANK L (= d_{i-1}).  A_2 is the round-22 "value only" state.

Every real chain maps to an abstract walk of the same weight, so the class-level closure
is a SOUND upper bound on F(M+q') at every m, and it is non-increasing in m.  Measured
exactly, full period, `research/kleene_history.py`:

    step        exact  budget   A_2   A_2+35  A_2+385   A_3  A_3+35  A_3+385   A_4  A_4+35
    11 -> 13      11      20     11      11       11     11     11       11     11     11
    13 -> 17      18      28     21      21       20     18     18       18     18     18
    17 -> 19      25      37     30      28       28     25     25       25     25     25
    19 -> 23      34      48   CYCL      45       42     35     35       34     34     34
    23 -> 29      43      63     60      60       45     43     43       43     43     43
    29 -> 31      58      74   CYCL      99       99     85     85       72     58     58
    31 -> 37      88      95   ----      ---      ---   CYCL    ---      115     88     88

(the 31 -> 37 row is A_3 value-only CYCLIC, A_3 + phase 385 = 115, A_4 = 88 = exact,
A_5 = 88; full period 33,426,748,355 slots, 6.23e9 gaps, 4,924 s)

AT THE FAILING STEP, TWO GAPS OF HISTORY BEAT ANY AMOUNT OF CORRIDOR PHASE: at 29 -> 31
the round-22 ladder went 99, 99, 91 as the phase modulus went 35, 385, 5005, while A_3
with NO phase at all gives 85 from 1,460 states, A_3 + phase 385 CERTIFIES (72 <= 74), and
A_4 - three gap values, phase-free, 14,368 states and 3,513 edges at machine 29 - is EXACT
at all SEVEN scannable steps, machine 31 included.  (The two axes are different information, not nested: at 19 -> 23 the round-22
mod-5005 state gives the exact 34 where A_3 alone gives 35.)  So (D) at
29 -> 31 is certifiable from a bounded local state after all; the object that was missing
was the machine's DICTIONARY OF REALISED GAP 4-TUPLES, not a finer congruence.

A_m is nilpotent exactly when m > A_relax(M) (R45's relaxation arity) - measured at all
seven steps: A_relax = 1, 2, 2, 3, 2, 3, 4 and the smallest acyclic order is
2, 2, 2, 3, 2, 3, 4.
So the round-21 counting boundary is precisely "A_m is not nilpotent below the arity",
and the arity is the order of history the certificate needs.

STILL NOT MACHINE-FREE, and now measured rather than guessed.  Replace "realised
m-tuple" by "CORRIDOR-ADMISSIBLE m-tuple with values in 1..F" - a machine-free edge set
depending only on (F, q') - and the closure blows up (`research/machinefree_cert.py`):

    step        budget  exact   MF_3 mod 35   MF_3 mod 385   MF_4 mod 35   layer 0 (lemma 1)
    11 -> 13       20      11      15  OK         15  OK         15  OK          14
    13 -> 17       28      18      31 +3          31 +3          31 +3           21
    17 -> 19       37      25      47 +10         47 +10         47 +10          36
    19 -> 23       48      34     111 +63        111 +63        111 +63          50
    23 -> 29       63      43     105 +42        105 +42        105 +42          67
    29 -> 31       74      58     125 +51        125 +51        125 +51          86
    31 -> 37       95      88     211 +116       211 +116       211 +116        116

HOW MANY MACHINE FACTS THE CERTIFICATE ACTUALLY NEEDS (`research/cegar_cert.py`).  The gap
between MF_4 (125) and A_4 (58) is a set of yes/no facts "is this gap 4-tuple realised?".
Counterexample-guided refinement - close, read off a maximising walk, ask about its tuples,
delete the unrealised ones (sound at every stage) - measures the count.  From the pure
machine-free start the bound falls 125 -> 86 and then STOPS, because 86 = 43 + 43 is layer
0 and layer 0 uses no edge at all.  Given ONE extra integer, F_2(29) = 55 (lemma 1's
left-hand side), the bound falls 125 -> 74 and (D) IS CERTIFIED after 6,395 queries.  So at
29 -> 31 the obligation is lemma 1 plus 6,395 four-tuple facts, against a 1,078,282,205-slot
period scan.  (Honest: the oracle is the dumped realised set, so this SIZES the obligation
rather than discharging it without a scan; and the refinement is greedy, so 6,395 is an
upper bound for that strategy.)

The three MF columns are IDENTICAL at every step: neither a finer corridor modulus nor more
history buys anything once "realised" is weakened to "corridor-admissible".  Layer 0 -
which is lemma 1, F_2 <= F + q', with no chain in it at all - already fails machine-free
from 19 -> 23 on, and its value is 2F or 2F - 2 at every step: the corridor constrains
where a gap sits, never how big it is (X11, X13, in their sharpest form yet).

Outside: for any sieve whose per-prime strike pattern has a finite-state grammar, the
one-prime increment of the maximal gap is a max-plus closure and the increment law has a
dual-certificate form.  The two-residue (paired / twin) case is the first where the
grammar is genuinely two-letter with forced alternation.

## 4c. ROUND 26 - THE IDENTITY IS A TWO-SIDED LAW, AND MECHANIC'S Q* CONJECTURE IS IT

WHAT HAPPENED.  Round 25 (mechanic, `docs/novel/old-machine-spectrum.md` section 8)
built the WORD-LEGAL CRITERION independently, from the census side, and registered as a
CONJECTURE on two exact points:

    Q*_J(M; legal for q') = max span of a J-gap window of M whose J-2 MIDDLE gaps
        (i)  are each = 0 or +-s (mod q'),  s = 2u' mod q', and
        (ii) induce a letter word (0/+1/-1) of prefix-sum range <= 1;
    CONJECTURE:  max_J Q*_J = F(M + q')  -  not merely an upper bound.

Q*_J IS qualmax_J IS LAYER J-2 OF K*.  Condition (i) is exactly K's edge condition
`d_i mod q' in {0, a, b}`; condition (ii) is exactly K's T3 tooth transition (letter 0
keeps the tooth, +-1 are the two tooth swaps, and "prefix-sum range <= 1" says the walk
stays inside the two teeth); the two unconstrained flanks are exactly L and R.  So

    Q*_J  =  max over K-paths of J-2 edges of  L (x) K^{J-2} (x) R  =  layer J-2 of K*,

and the conjecture "max_J Q*_J = F(M+q')" is, verbatim, the identity of section 1.  It
is therefore a THEOREM, proved in section 3 in round 22; the direction mechanic left
open (Q*_J <= F(M+q') for every J) is the (>=) half of that proof - the CRT choice of
the killing copy.

STATED ON ITS OWN, because the theorem deserves a form that needs no max-plus:

  ATTAINMENT THEOREM.  Let x_0 < ... < x_J be consecutive openings of M whose J-2 middle
  gaps satisfy (i) and (ii).  Then x_J - x_0 <= F(M + q').
  PROOF.  (i)+(ii) hold iff there is a tooth assignment t_1..t_{J-1} in {+,-} with
  x_{i+1} - x_i = (t_{i+1} - t_i) c (mod q'), c = 6^{-1} mod q'.  Fix one and set
  r = t_1 c - x_1 (mod q'); then x_i + r = t_i c (mod q') for every interior i.  The
  joint period of M + q' is P(M) q' with gcd(P(M), q') = 1, so some translate
  x + jP(M), jP(M) = r (mod q'), is a window of M with the same gaps in which gear q'
  blocks EVERY interior.  No opening of M + q' then lies strictly between x_0 and x_J,
  so the gap of M + q' containing that interval is at least x_J - x_0 (equality iff q'
  also spares x_0 and x_J - and if it does not, the containing gap is only LONGER).  []

  J = 2 is mechanic's own DELETION LADDER F_2(M) <= F(M + one gear).  The theorem is its
  extension to every depth, and it is the exact converse of the criterion direction, so
  together they give  max_J Q*_J = F(M + q')  identically.

THE COMPUTATION (round 26, `research/qstar.py`, log `research/data/r26_qstar.log`).  A
three-line proof deserves an independent check, and the "for every J" quantifier is
closed computationally rather than by a depth cap: every real legal window maps to an
A_4 walk of the same weight, so the A_4 CLOSURE is an upper bound on max_J Q*_J over ALL
J at once, and A_4's layer vector TERMINATES, which rigorously caps the depth (no
abstract walk of length k means no legal window of J = k+2 gaps).  Per-depth values are
then computed exactly by descending-span search seeded at the A_4 layer bound, against
the realised-tuple oracle (mechanic's exact full-period censuses at arity <= 4 where they
exist, the scan-free CRT decision of `research/crt_dict.py` elsewhere):

    M    q'   F(M)  F(M+q')   Q*_2  Q*_3  Q*_4  Q*_5   max_J Q*_J   J*    verdict
    11   13     7       11      11     8     -     -           11    2     EXACT
    13   17    11       18      16    18     -     -           18    3     EXACT
    17   19    18       25      25    25     -     -           25    2,3   EXACT
    19   23    25       34      31    33    34     -           34    4     EXACT
    23   29    34       43      39    43     -     -           43    3     EXACT
    29   31    43       58      55    58    55    55           58    3     EXACT
    31   37    58       88      68    85    88    68           88    4     EXACT
    37   41    88       91      90    90    91     -           91    4     EXACT

EIGHT steps, all exact - the seven scannable ones and, because mechanic's round-25 m37
census makes it decidable, 37 -> 41, beyond every Q* computation anyone had run.  The
attaining depth J* reproduces k_win + 1 at every step (mechanic's two anchors J = 3 at
29->31 and J = 4 at 31->37 among them), and the attaining WINDOWS are the project's known
extremal objects, re-derived: (4,8,15,7) at 19->23 is R41's k=3 record window,
(7,10,21,10,7) at 29->31 is R50's exact J=5 inventory, (11,12,37,28) at 31->37 is R25's
padded winner, and (2,88) at 37->41 is mechanic's F_2(37) maximiser.  NEW: F(41) = 91 is
attained at the window (21,14,41,15).

WHAT THE EXACTNESS MEANS - and it cuts both ways.
* POSITIVE: the criterion is not a relaxation of (D), it is a CHANGE OF REPRESENTATION.
  Q* is computed on the OLD machine's period (or, scan-free, on its dictionary), so it
  prices F(M+q') without ever building M+q'.  That is the whole value, and it is a real
  one: it is how the ladder runs past every scannable machine.
* NEGATIVE, and it should be said plainly: BECAUSE Q*_max = F(M+q') EXACTLY, "the
  criterion certifies (D)" is not weaker than "(D) holds" - it is the same statement.
  The margins the criterion reports (0.52-0.69 q' at the literal steps) are the TRUE
  margins of (D) itself, not slack in a relaxation, and no proof can be obtained by
  exploiting looseness in the criterion, because there is none.
* CONSEQUENCE FOR MECHANIC'S TWO SEEDED ANCHORS: their runs at 43->47 and 47->53 were
  seeded at budget-1 and reported <= 149 and <= 170.  The theorem gives the exact values
  outright: max_J Q*_J(43; legal for 47) = F(47) = 118 and max_J Q*_J(47; legal for 53)
  = F(53) = 145.  Both certifications hold with margins 32 and 26, not +1.
* CONSEQUENCE FOR THE RECORD: R39 reported the criterion value equal to F(M+q') "at 6 of
  7, slack 2 at 23->29".  The exact computation gives max_J Q*_J(23; legal for 29) = 43
  = F(29), attained at J = 3 by the window (10,10,23).  The recorded slack is an
  artifact; equality holds at 7 of 7 - as the theorem requires it to, at every step,
  for ever.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

- The increment law / part (D): this is now its certificate form, and the remaining task
  is a CLOSED FORM for h (or for a bounded-state super-solution) valid at every machine.
  Round 23 narrows this to ONE object: the machine's set of realised 4-tuples of
  consecutive gaps.  Given that dictionary, a 14k-state max-plus closure settles (D) at
  machine 29 exactly; the open part is producing (or soundly over-approximating) the
  dictionary without a period scan.  Corridor congruences provably do not (section 4b);
  the pruned-IE exact pattern counter (R43, `research/qualrun_zerocert.py`) decides one
  tuple by CRT arithmetic with no scan, and only the tuples whose window sum could exceed
  F + q' need deciding, which is a small set.
- Ziller-Morack Conjecture 6 / h_2 growth: the per-prime increment is a longest path in
  an automaton of size O(n); the growth question becomes a bound on that path.
- Jacobsthal g(n): the one-residue analogue is the same construction with a single tooth
  (no alternation), where K is a partial map with the SAME nilpotency question.
- Tropical/max-plus spectral theory: K nilpotent means the tropical spectral radius is
  -inf; the interesting invariant is the star's entrywise size, not an eigenvalue - which
  is why round 20's search for a spectral gap found none (R35/R36).

## 6. PRIOR-ART CHECK

Not yet checked (agent without web access this round).  Suggested searches: "max-plus
Kleene star longest path sieve"; "Jacobsthal function transfer matrix increment";
"tropical certificate for a combinatorial gap bound"; "Cuninghame-Green nilpotent
max-plus"; "Holt Rudd cycle of gaps automaton"; "wheel sieve gap word automaton".
Nearest known art the project has read: Holt & Rudd's cycle-of-gaps recursion
(one-residue, and it recomputes the cycle rather than certifying its maximum), and the
project's own merge-law entry.  The delta claimed here is the identity + the
depth-quantifier-free certificate, not the max-plus machinery.
