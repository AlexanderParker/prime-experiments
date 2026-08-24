# kleene-generator - the merge law's Kleene star: F(M+q') as one tropical identity, and (D) as one arity-free certificate

Status: SCRIPT-VERIFIED (exact, full period) - the identity is asserted against the known
F(M+q') at all six scannable consecutive steps 11->13 .. 29->31 in
`research/kleene_generator.py`; the certificate equivalence and the finite-state
abstraction ladder are computed in the same script with assertions.  Established round 22
(constructor).  THE CERTIFICATE DIRECTION IS KERNEL-CHECKED since round 22 (formalist,
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
  HONEST NEGATIVE, and it is the important half: at 29 -> 31 NO bounded state tested
  certifies - mod 35, 385 and 5005 give 99, 99, 91 against a budget of 74 (exact 58).
  The generator is arity-free; it is NOT yet machine-free.  The loss is in the crude
  sound edge weight (max over all source openings realising a class edge) and in pairing
  a long chain with a large flank that never co-occur; tightening those is the named
  next construct.

Outside: for any sieve whose per-prime strike pattern has a finite-state grammar, the
one-prime increment of the maximal gap is a max-plus closure and the increment law has a
dual-certificate form.  The two-residue (paired / twin) case is the first where the
grammar is genuinely two-letter with forced alternation.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

- The increment law / part (D): this is now its certificate form, and the remaining task
  is a CLOSED FORM for h (or for a bounded-state super-solution) valid at every machine.
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
