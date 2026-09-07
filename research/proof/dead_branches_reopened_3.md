# Dead branches reopened, third pass: the wall at the ledger (lateral lane, 2026-09-07)

The unstick protocol with the shadow step (SKILL.md, "Parts, interfaces, shadows"), run over
the newest wall face: V2 of the turn ledger (research/proof/turn_ledger.md), and over the three
earlier ROOT nodes it touches: W103 (the manifold's record, manifold_census_large.md), the
engine's skip half (pad_cap.md, node 4.i.b.ii.a) and L65 (the quiet-zone record,
top_machine_6.md). Format as in dead_branches_reopened_2.md; the faces A-E and the bricks N1-N7
of the earlier passes apply to every idea. Vocabulary: ENGINE (primes <= q), VALVES (the engine
acting inside the manifold's open set; the families (s, s')), MANIFOLD (primes in (q, Q]; smooth
zone, quiet zone (Q, Q^2]), EXHAUST (primes above Q); charge = s x P (air s, fuel P), pure charge
= family (1, 1), ember = a q-smooth number above Q, turn m = (mQ, (m + 1)Q].

Thinking and small spot checks only (script in the lane's scratch directory, one core, seconds
each; every number used is in this file). No tree or index edits.

New bricks this round (constraints every idea below must respect):
- N8 (V2). Every proved valve law (imprint, onset, port, inventory, ember) uses only that a fuel
  is odd, above Q and coprime to q#; F = {n > Q : gcd(n, q#) = 1, n = 1 (mod 3)} obeys them all
  with P_m = 0 and B_m = T_m > 0 in every turn. No inequality forcing the pure charge follows from
  those laws; the alternating expression telescopes to an identity.
- N9 (W103, L65). The manifold's quiet-zone record is a twin gap of the bottom stratum, identical
  across engines, shortened only by embers with a prime neighbour; bounding it is the root.
- N10 (pad_cap). Every cap on the engine's word depth L is a function of T = F(M + q')/q'; a
  constant cap on L_skip is equivalent to F(M + q') <= C q' along the ladder (ROOT).
- N11 (this pass, measured). The one-number-tooth manifold adversary: the free-phase sieve in
  which each manifold prime p in (q, Q] removes ONE class c_p of its choice empties the pure
  charge of the first 60 turns at Q = 10^4 with 776 of the 1,226 gears (q = 5; 773 at q = 7),
  while satisfying every proved valve law, keeping the fuel balanced between the classes 1 and
  2 mod 3 (41,145 against 41,068) and keeping the valve set mirror-closed (114 of 118 families
  present with their mirror). So balance, mirror closure and "one tooth per gear on the number
  line" do not force the pure charge per turn either.

Spot checks run (scratch script fuel_adversary.py; q = 5 unless stated, Q = 10^4):

| check | result |
|---|---|
| the family set of F = 1 mod 3 | present iff s' = s + 2 (mod 3): 11 of 11 at (5, 10^3, 12 turns), 5 of 5 at (7, 10^4, 6 turns); NO family is present together with its mirror (s', s); (3, 1), (1, 5), (4, 2) absent, (1, 3), (5, 1), (2, 4) present |
| greedy one-tooth cover of the twin candidates (n, n + 2 both coprime to q#) in turns 1..m by manifold primes with free classes | gears needed K(m) = 57, 87, 110, 148, 234, 369, 776 at m = 1, 2, 3, 5, 10, 20, 60; sum of 2/p over the gears used 1.93, 2.10, 2.18, 2.28, 2.44, 2.58, 2.78; smallest gear always 7 (11 at q = 7), largest 929 .. 9,629 |
| the resulting fuel set under counterfactual.py's checks | 0 violations of onset, inventory, imprint, port, ember law and ember bound in all runs; P_m = 0 in every turn; B_m = T_m; in turns 1, 2 the burnt side is ember charges alone, as V1 says |
| first twin spoke m_1(q) (least m with m q# +- 1 both prime) | 1, 2, 1, 6, 8, 11, 4, 16, 22, 4, 74, 24, 37, 28, 14 at q = 5 .. 59: no structural floor, the heuristic q^2 / log^2 q scale |

---

## V2. The wall at the ledger (the counterfactual fuel; face A)

- **Object.** The pure charge P_m of a turn as a consequence of the valves' structural laws: an
  inequality B_m <= c T_m with c < 1, or any lower bound on P_m, derived from imprint, onset,
  port, inventory and ember alone.
- **Vectors.** The exact ledger (V1, 600 turns); seven candidate inequalities C1-C7 on the
  tables (all fail or are ROOT except the ember bound); the alternating (Legendre) expression
  for P_m (telescopes); the counterfactual F = 1 mod 3 (V2, proved by construction).
- **Failure, precisely.** The laws never use primality; they hold for any set of odd integers
  above Q coprime to q#. The counterfactual F has no internal pairs at distance 2, so P_m = 0
  while every law holds. What the measurement left untouched is WHAT F LACKS, as an object:
  - (i) F is one-sided on gear 3: every element is 1 mod 3, i.e. 1 mod 6, so as a lower member
    it is always the left partner of a number divisible by 3 (port 1) and never enters the
    column port (n = 5 mod 6). In tooth language: F sits entirely on the tooth -2 of gear 3 and
    never on the tooth 0. The primes occupy both teeth of every engine gear at equal rates.
  - (ii) F's valve set is mirror-free. A family (s, s') exists in F iff s' = s + 2 (mod 3)
    (from sP + 2 = s'P' with P, P' = 1 mod 3), so (s, s') and (s', s) are never both present
    (4 = 0 mod 3 is false): checked, 11 of 11 and 5 of 5 above. The real valve set is
    mirror-closed (the reconcile: the mirror n -> -n - 2 carries R(s, s') onto R(s', s); every
    admissible family realised to max(s, s') = 1,500). Mirror closure of the VALVE SET is a
    which-residues law of the fuel that the ledger never wrote down; it says the fuel meets the
    classes +2 and -2 of every engine gear.
  - (iii) F is the open set of a sieve in which the prime 3 removes TWO classes of the number
    line ({0, 2}) instead of one. In pair coordinates a two-class number-gear at distance 2 is
    an anti-domino: it strikes every column mod 3 ({0, 2} on n and on n + 2 covers all three
    classes), so it empties the pure charge on its own. The primes are the open set of a sieve
    with ONE class per gear on the number line (the multiples), and one number-class is what
    makes every gear a domino {x, x + 2} in pair coordinates (W3, the partner law, kernel).
  - (iv) The property in one sentence: THE FUEL'S PAIR STRUCTURE IS THE EXHAUST'S TWIN-GEAR
    SET, and the exhaust is the open set, on the quiet zone, of the one-tooth zero-phase sieve
    of the whole stack below it (S3, the exhaust cap: open iff prime). So the pure charge of tier
    k is the twin gears of tier k + 1, and the valve laws are blind to the fact that the fuel is
    an OPEN SET at all. The counterfactual is not the open set of any one-tooth sieve.
  - What N11 adds: (i), (ii) and (iii) are each necessary and jointly insufficient. The
    one-tooth free-phase manifold adversary is balanced, mirror-closed, one-tooth per gear, and
    still empties 60 turns. What it lacks against the primes is exactly two things: its phases
    are free (the primes' one class is 0, the home strike), and its reach is finite. By CRT a
    free manifold phase vector is a translate of the real one by a multiple of q#, so "the
    free-phase manifold adversary empties the whole zone (Q, Q^2]" is "some translate of the
    window of {5..Q} is fully blocked", i.e. F({5..Q}) >= W(Q): the root at rung Q, in the
    exhaust's coordinate. Measured, its reach at Q = 10^4 is of order 10^2 turns (776 gears for
    60 turns, the growth K(m) ~ m^0.64 extrapolating the full 1,226 gears to about 120 turns)
    against the 10^4 turns of the zone: the Jacobsthal-scale slack Q / log^2 Q, not the factor
    four of the small rungs.
- **The shadow.** The blocker is the outline of THE HOME STRIKE: the one fact about the exhaust
  the valves never used is that its sieve has phase zero at every gear (each prime strikes its
  own multiples), which is the only sieve whose struck set is closed under multiplication, which
  is what the exhaust cap's proof uses (n = pm forces a small factor). In the column coordinate
  the zero-phase vector of the stack on the zone is the window's phase vector at rung Q, which
  face C3 measured as typical for the island witness in the ARC. At the whole-window scale it is
  not typical in count (it leaves the prime number theorem's survivors, 21% below the model,
  C5), and that count is face A. So the shadow object is a which-residues property (phase zero)
  whose only known consequence at the window scale is a count. Candidates to develop:

  **(a) The fuel as the exhaust's gear set, one tier up.** Definition: the exhaust X = primes
  above Q as a machine with teeth {0, -2} on the raw line. On the quiet zone its strikes are
  home strikes only (S3), so the fuel = X's home-strike set and the pure charge = X's twin
  gears, which one tier up (on (Q^2, ...]) collide at (g + 4)/3 with the shared short arc
  (g + 1)/3 (docs/proofs/21, Cor. 2.1, no size hypothesis). The ladder: pure charge of zone k =
  shared-arc gear pairs of tier k + 1. Walking UP is climbing cuts (docs/proofs/23: not
  progress). Walking DOWN is exact and it is Theorem (E) (5k): a column above Q is blocked
  under all primes iff blocked under {5..sqrt(6k + 1)}, so turn m of Q, (mQ, (m + 1)Q], is the
  top slice of the window of the machine y_m = sqrt((m + 1)Q), of relative length 1/(m + 1).
  Hence: P_1(Q) > 0 iff the machine {5..sqrt(2Q)} has an opening in the TOP HALF of its window,
  which follows from F(y) < W(y)/2 (measured F/W = 0.23-0.31 at y = 7..53), and P_m(Q) > 0
  follows from F(y_m) < W(y_m)/(m + 1). The descent spends the slack as 1/(m + 1): the ledger's
  turns 1-3 are within the measured factor four, turn m >= 4 over-asks (face E), and the base
  case is the certified ladder. First test: none needed, this is exact; the finding is that the
  ledger's V5 (P_1, P_2 > 0 for every Q >= 10) is the record law F < W/2, F < W/3 at the square
  roots, and no descent closes on the twin gears alone because tier k + 1's full gear set, not
  its twin pairs, decides zone k + 1.
  What is new in it: the fixed point is the square-root map, and the invariant to look for is
  one that a twin at scale y transmits to scale y^2. The measured candidate is the walk from
  q^2 for q the lower member of a twin (667 twins to 5,000: a twin within 265 columns); the wall
  says a bound on it is twin-Bertrand at scale q/3 (E4). Test that would count: a map twin ->
  twin near the square whose image is non-empty for a STRUCTURAL reason (an always-open
  neighbour of a square, the way column 0 is always open), which no measurement has suggested.

  **(b) The exhaust's dominoes on the engine's wheel.** Definition: prime P > Q places the
  domino {P - 2, P} on the pair line (the two pairs containing it); two dominoes intersect iff
  the primes are twins (W29's "pairwise intersecting dominoes lie in a span of 2"); no three
  pairwise intersect (P, P + 2, P + 4 not all prime). So the pure charge = the intersecting
  exhaust dominoes, and the prime-led charges A_m, A'_m are the dominoes whose other cell is a
  charge position. The engine acts on the prime-led charges with ONE tooth: P goes to a burnt
  family iff P = -2 (mod p) for some engine gear (the placement law). So THE PURE CHARGE IS THE
  OPEN SET OF A ONE-CLASS SIEVE ON THE PRIME SEQUENCE (class -2 at every gear, phase fixed), and
  the walk from a twin to the next twin is the walk of that one-class sieve over the primes.
  Prior art, one line: this is the sieve of the shifted primes {P + 2}, dimension 1, level of
  distribution 1/2 by Bombieri-Vinogradov, which is Chen's setting (P_2, not P_1, by parity);
  the valves' disjunction over air (families (1, s') with s' q-smooth) is a version of the
  almost-prime disjunction restricted to the zone. The structural law the brief asks for (what
  forces two exhaust dominoes to intersect on one engine period) is therefore: which residues
  mod the engine the consecutive primes take, and one-class Jacobsthal on the integers bounds a
  run of struck integers by j(q#) - 1, but on the SUBSEQUENCE of primes the classes -2 mod p of
  consecutive primes carry no constraint (a burnt run of prime-led charges is unbounded at the
  density's geometric rate). First test (cheap, decisive for the candidate): the longest run of
  consecutive prime-led charges all burnt, per turn, against the geometric prediction
  log(A_m)/log(1/(1 - prod (p - 2)/(p - 1))): a run law with a structural cap would be new;
  agreement with the geometric tail kills the candidate as a route and files it as FACT.

  **(c) The three-record ladder and the classification of adversaries.** The adversaries that
  kill the pure charge while obeying the valve laws, with what each lacks:
  - congruence adversaries (F = 1 mod 3; F' = {all prime factors = 1 mod 3}, which is even
    multiplicatively closed): one-sided on an engine tooth, mirror-free valve set, an
    anti-domino at 3. Lack BALANCE (occupancy of both teeth of every engine gear) and MIRROR
    CLOSURE. These are which-residues properties allowed by face A, and the primes have them
    (every reduced class of every modulus).
  - the one-tooth free-phase manifold adversary (N11): balanced, mirror-closed, one tooth per
    gear; lacks PHASE ZERO, and its reach is the record of {5..Q} in turn units.
  - the full-equidistribution adversary: a set with no pairs at distance 2 and equidistributed
    in reduced classes to level 1/2 exists in the extrinsic form F_lambda = {n : lambda(n) = -1,
    lambda(n + 2) = +1} (Liouville; membership of n refers to n + 2). It lacks INTRINSIC-NESS:
    membership of n decided by n alone. An intrinsic set (a condition on n's own factorisation
    and residue) with no pairs at distance 2 must couple n to n + 2 exactly, and the only exact
    coupling between n and n + 2 is congruential, which forces one-sidedness on some gear.
    Prior art, one line: this is the pretentious / non-pretentious dichotomy (the counterfactual
    is defined by the character mod 3; Tao's two-point logarithmic Elliott theorem says a
    non-pretentious bounded multiplicative function cannot avoid the correlation at shift 2,
    while the primes' indicator is not multiplicative and the twin problem in that language is
    the von Mangoldt two-point correlation, out of reach). Nothing new to derive there; the
    machine reading is the useful residue: "intrinsic + balanced + one-tooth + phase zero" is
    the list, and the first three are exactly what the free-phase manifold adversary has, so
    phase zero at the window scale is the whole difference.
  The real fuel's coordinate-free properties, for the record: intrinsic; balanced on every
  engine gear (Dirichlet); mirror-closed valve set; the open set of the one-tooth zero-phase
  sieve of the stack; multiplicatively generated together with the smooth numbers (the charge
  set is (q-smooth) x (fuel u {1}), and on the zone the fuel is the set of irreducibles of that
  monoid, vacuously since products of two fuels exceed Q^2). First test: is "the fuel is the open
  set of a one-tooth sieve with phase zero by SOME subset of the primes in (q, Q]" enough? No,
  trivially and in the wrong direction: dropping gears enlarges the fuel and the pure charge is
  monotone in the fuel, so the real fuel is the MINIMAL zero-phase fuel and the statement is the
  root. Filed as the reason the counterfactual axis is exhausted: every property short of
  "phase zero at every prime up to Q" is satisfied by an adversary with zero pure charge for
  as many turns as its gear count allows.

  **(d) The walk over charges.** Definition: from a point x the next charge is
  min over q-smooth s of s x nextprime(max(Q, x/s)) (L62, W81); the next pure charge is the
  s = 1 term at the first prime P with P + 2 prime, i.e. the one-class walk of (b). Layered over
  the valves in onset order: turn m opens A(m) families, and the walk from a twin to the next
  twin in CHARGE units is 1 + (burnt charges in the gap). The manifold's hop collapse (L38/L58,
  chain <= 2) needs g > F_G + 3, a gear larger than the record of the machine it joins; every
  engine gear is smaller than the manifold's record by two orders (5 against 924 at Q = 10^5),
  so the valves' walk is in the loaded regime and cannot collapse. Where the collapse DOES hold
  is exact and already on record: in turns 1 and 2 the only burnt charges are ember charges,
  at most 2 E_m, so the walk from a twin in charge units is at most 1 + 2 E_m (V1) and the
  charge sequence is the twin sequence with a sprinkle of embers (W103 from the other side);
  and the exhaust's own gears on the zone (silent, S3). First test: the maximum over turns of
  the charge-count walk between consecutive twins, per turn, against A(m) x (mean twin gap) x
  (burnt density): expected unbounded from turn 3 (the (1, 3)/(3, 1) charges sit next to 37%
  of twins and between them at a linear rate). A structural cap would be new; none is expected.
  In the column port the valves' walk is the engine's layered walk (the hit law and chain law
  of the anchor line, exact to {5..23}) evaluated on the manifold-open subsequence; the double
  hop rule (lower gap = 0 or +-d_g mod g) is a statement about consecutive lower openings, and
  on a subsequence the gaps are unconstrained, so the chain depth per layer inherits no cap.

- **Realisations.**
  - (a)(i) State and file the descent as a one-line theorem: P_m(Q) > 0 whenever
    F(sqrt((m + 1)Q)) < W(sqrt((m + 1)Q))/(m + 1) (Theorem (E) plus the exhaust cap). Value:
    it turns V5's scan into a corollary of the record law at rungs <= sqrt(2 x 10^5) = 447,
    where SAT lower bounds exist but no upper bound: a target for the certified ladder, not a
    route. Reopens: nothing; it fixes which rung a turn belongs to.
  - (a)(ii) The twin-to-twin map: for every twin gear (p, p + 2) of the manifold, the walk from
    p^2 and from p(p + 2) to the next twin, against the walk from a lone prime's square; if twin
    gears' walks are systematically shorter (they are not expected to be: R2.a found the walk
    decided by the old gears), the map is a candidate object. Cheap, q <= 5,000 exists in
    walk_path.md's data. Reopens the exhaust.
  - (b)(i) The burnt-run law on the prime-led charges (the test above), from the existing
    ledgers. Reopens the valves as a one-class object on the primes.
  - (b)(ii) The disjunction as the sieve's object: write the zone record bound as "some family
    (1, s'), s' q-smooth, or (1, 1), has a member in every interval of length L above Q" and
    price it against the almost-prime short-interval results (a literature line; the needed L
    is of order Q^0.6 at Q = 10^5, far below any short-interval sieve range). Reopens: nothing
    unless the literature has a P_2 result at that scale, which it does not.
  - (c)(i) Add to the valve laws, as exact laws of the REAL valve set: balance (every family
    (s, s') and its mirror (s', s) present with counts equal to square-root accuracy; measured
    313,608 against 312,952 for (1, 3)/(3, 1)) and mirror closure of the inventory; record that
    the counterfactual violates both and the free-phase adversary satisfies both. Reopens the
    valves' parts ledger with two which-residues items, both insufficient alone.
  - (c)(ii) The fuel-side covering reach R(q, Q) = the most turns the free-phase one-tooth
    manifold can empty: measure it fully (greedy with all 1,226 gears at Q = 10^4; an ILP
    certificate for small m the way K(d) was certified) and its growth with Q; R(q, Q) < Q for
    every Q is the root at rung Q in the exhaust's coordinate, and R/Q is the slack in turn
    units (of order log^2 Q / Q, not 1/4). Reopens the exhaust as an adversary family with a
    published neighbour (A072753 is the free-CLASS record over an initial segment of primes;
    this is the free-PHASE fixed-domino record over the manifold's primes, a third column of
    the jacobsthal_check.md table, real <= this <= free class).
  - (d)(i) The charge-count walk per turn from the ledgers (the test above). (d)(ii) The
    engine's layered walk on the column-port charges, layered by gear in onset order, hops per
    layer and chain depth at q = 5..13, Q = 10^4: the expected result is "no cap"; the finding
    would be the first turn at which a layer hops three times, which the full-column walk never
    does at layer 11 ("layer 11 never hops twice").

## W103. The manifold's record is the bottom stratum's twin gap (ROOT)

- **Object.** The manifold's quiet-zone record as a manifold quantity.
- **Vectors.** The census at Q = 10^5 and 3 x 10^5 for q = 5..13; the strata; the split of a
  record by a smooth number with a prime neighbour (850,500 = 2^2 3^5 5^3 7 at q = 7);
  identical records across engines at 14 of 19 Q.
- **Failure.** The record is a twin gap: the engine enters through the embers alone, and the
  twin primes are the same for every engine. Untouched: the record's LOCATION (1.35-2.63 Q, the
  bottom stratum, sticky across Q), and the reason it is the bottom stratum: the onset law. In
  turn m there are A(m) families; the bottom has one. The record sits where the charge set is
  the pure charge alone, which is the thinnest the zone ever is.
- **The shadow.** The record is a twin gap in (Q, 3Q], and by Theorem (E) the twins there are
  the openings of the engine {5..sqrt(3Q)} in its window; consecutive twins are consecutive
  openings (S3), so every twin gap in (Q, 3Q] is at most F({5..sqrt(3Q)}) columns. The
  manifold's record is bounded by the engine's record at the square root, exactly, and the
  bound is loose by the ratio of the period's record to the window's gaps (L65's 3.2-24 against
  the prime-gap floor is the same looseness seen from below). W103 descends to the engine's
  record; it is not a manifold object. The object the blocker outlines is the engine's record
  in the WINDOW (not the period): the window's longest opening gap, F_W(y) <= F(y), which no
  branch has measured as its own object at the rungs where F is known.
- **Ideas and realisations.**
  - Idea 1: the window record F_W(y) against the period record F(y). Realisations: (i) tabulate
    F_W at y = 7..53 (full periods exist) and at y to 5,000 by sieve (the window is y^2/6
    columns, trivial), against F(y) and against the twin gaps: if F_W/F is bounded away from 1
    by a rule (the window is one specific translate: the zero-phase one), the manifold's record
    has a computable engine bound with the slack visible; (ii) the position of the window's
    longest gap relative to the column of q'^2 (the square gate) and to the spokes: the record
    of the zone at Q = 10^5 sits at 1.879 Q, in the top half of the window of y = 447. Reopens
    the engine (its window record as a part).
  - Idea 2: the record's onset reading. Realisations: (i) per turn, the longest manifold-open
    gap R_m against the family count A(m) and the yield sum Sigma^log(m): the record profile
    over turns (W83's U, corrected by X22) as a function of the valves open, from the existing
    ledgers; (ii) if R_m falls with A(m) by a rule, the record of the zone is the m = 1, 2 value
    by the onset law, and "the record is made where no valve is open" is a mechanism statement
    for W103 rather than a location. Reopens the valves (the onset law as the record's cause).

## The engine's skip half (pad_cap.md; a constant cap on L_skip is F(M + q') <= C q'; ROOT)

- **Object.** L_skip(M), the longest realised legal word with a skip letter (2q', a + q',
  b + q'), and a uniform cap on it.
- **Vectors.** E1 (the small alphabet {a, b, q'} capped at CORRCAP_3 <= 8 by gears 5 and 7,
  uniform); the skeleton (S_0, S_1) on two tooth-APs of step q'; E2 (L + 1 <= 2 Omega(T + 1),
  T = floor((F(M + q') - 2)/q'), corridor part only); E3 (the exact corridor-plus-span cap,
  slack 0 or 1 at nine of nine rungs, tight at m29 and m53); the counting cap (vacuous, ratio
  1.4-2.2); the merge forest's depth, the record's composition, the corridor and the gear-5 lock
  refuted as the shadow of the PAD alphabet.
- **Failure.** Every cap is a function of F/q'; the skip half's growth is the record in gear
  units. Untouched: the pullback with ALL gears (only 5, 7, 11, 13 were used), the cover half of
  the skip words at m41-m47, and whether the arc floor (docs/proofs/21 Theorem 3, a real-teeth
  fact) has a pullback analogue.
- **The shadow.** E2 says a realised word is a pair of opening sets of the PULLBACK MACHINE
  M^{(q')} (same gears, separations 2 u_g q'^{-1} mod g: the real machine read along an AP of
  step q', which is a member of the counterfactual family with the teeth moved by the factor
  q'^{-1}; 5i's twisted copies are M read along a later gear's stride, and the pullback is
  that object at the gear q'), in T + 1 consecutive
  multipliers. So the cap on L is the RICH-INTERVAL FUNCTION of a family member:
  Omega_M(n) = the most openings of M^{(q')} in n consecutive multipliers over all phases. This
  is the dual of the record (F = the longest interval with no opening; Omega = the fullest
  interval of length n), it is universal (a gear >= 2n + 1 can be phased off any window of n
  multipliers, so only the gears <= 2n decide it: a finite object per n, the same for every
  machine containing the primes to 2n), and by the one-orbit reduction it is the free-phase
  maximum. Prior art, one line: the maximal size of a set of columns in an interval of length n
  avoiding two classes per prime is the two-class admissible-tuple function (Hensley-Richards
  rho*(x) in one class); in the pullback the separations are twisted (2 u_g q'^{-1}), a fixed
  but non-real separation per gear, which is the object the record's dictionary never named.
  The shadow of the skip half is therefore not the record's growth alone but the pair
  (poorest interval of M, richest interval of M's twist along q'): the record of M + q' is the
  longest stretch of M all of whose openings lie on q'-teeth, i.e. a poor interval of M whose
  few openings are a rich pullback interval.
- **Ideas and realisations.**
  - Idea 1: Omega with all gears, and its law. Realisations: (i) compute Omega_{M^{(q')}}(n)
    exactly for n <= 20 at m19..m53 (a covering computation of docs/proofs/20-21's kind on a
    two-tooth machine with twisted separations; the gears above 2n do nothing) and the sharpened
    cap L + 1 <= 2 Omega^{full}(T + 1): if it is tight at more rungs than E3's two, the word
    depth IS the pullback's richness, a law; (ii) the arc-floor analogue: c(g, h; L) = 0 for
    L <= max(a_g, a_h) fails for random separations and holds for the real ones; the pullback's
    separations are coherent (all one third, scaled by one unit q'^{-1}), so test the floor on
    the pullback: coherence was measured worthless for F (branch 6) and for K (W3) and never for
    Omega. Reopens the engine (its twists as parts).
  - Idea 2: records are made where the old machine is richest along the new gear's stride.
    Realisations: (i) at every record fusion on file (m13..m53, witnesses in pad_cap 2.6), the
    translate of the pullback along q' at the record's start, and whether the openings of M
    inside the record stretch attain Omega^{full}(T + 1) (the richest possible) or sit below it
    by a fixed deficit; R3.h's "records are made at the ends by the top three gears" says the
    junctions are old openings hit by top-gear teeth, which is the same statement in the other
    coordinate; (ii) if the record is always at a richest translate, the record's start class
    mod q' x (small gears) is computable from Omega's maximisers, a formula for the record's
    position at the new gear (thin place 3 in a new coordinate, allowed by face B because the
    modulus is q' x gears <= 2n and grows). Reopens the engine's record as a two-coordinate
    object.

## L65. No upper bound on the quiet-zone record, and the exact reason (ROOT)

- **Object.** An upper bound on the zone record, which needs a lower bound on the density of
  open pairs in a short interval above Q.
- **Vectors.** The zone known as a rule, an enumeration, a count and a walk (L57-L62); the
  prime-gap floor (L63, W82); the family decomposition W79; W84's correction (an upper bound
  needs an open pair from ANY ONE family, a finite disjunction, weaker than the twin problem).
- **Failure.** Parity-blocked: every family is a binary linear prime problem. Untouched: the
  disjunction as an object of the valves (it is the onset law's family list I(m) per turn,
  plus embers), and the record's stratum (the bottom, where the disjunction has one term).
- **The shadow.** The record is made where the disjunction is shortest. In the bottom stratum
  the disjunction is (1, 1) alone plus ember charges; from turn 3 it has (1, 3), (3, 1); the
  yield sum Sigma^log(m) grows to about 10-17 by turn 60 and the per-turn record falls. So the
  object is the record of the UNION of the first A(m) families, which is a covering-by-families
  question: does the union of the open valves' imprints, each a fixed residue set mod q#,
  with each family's members at its own density, leave a gap of length L? For m = 1, 2 the
  union is the pure charge and the answer is the root; for large m the union's density tends to
  the engine's pair-opening density (V4) and the record of the union is an ordinary gap of a
  set of density 1/Sigma_inf, bounded by nothing structural but small in practice. The shadow
  is the descent of section W103: the record is at the bottom because the descent's slack is
  1/(m + 1) and the bottom is m = 1.
- **Ideas and realisations.**
  - Idea 1: the disjunction's covering form. Realisations: (i) for each turn m, the longest run
    of charge-free positions among the positions the imprints ALLOW (the union of R(s, s') over
    I(m), a residue set mod q# with prod(p - 2) x A(m)-ish classes): a gap of the zone is a run
    of allowed positions with no charge; measure the ratio (zone record in turn m) / (longest
    run of allowed positions with no charge, in the engine's period) to see whether the record
    is an imprint effect (residues) or a fuel effect (primes); (ii) at m = 1 the allowed
    positions are the engine's openings (prod(p - 2) per period) and the run is the engine's
    record F(q) in the raw line; the zone record 924 at Q = 10^5 against 6 F(13) = 66: the
    record is a fuel effect by a factor 14 at the bottom, and the question is how the factor
    behaves with m. Reopens the valves (imprint unions as a part).
  - Idea 2: price the disjunction in the literature once, cheaply, and close it. Realisations:
    (i) the statement "some (1, s') with s' q-smooth, or (1, 1), has a member in (x, x + L]
    for every x in (Q, Q^2]" at L = Q^0.6 is an almost-prime twin statement in a short interval
    far below the sieve's range; record the best known short-interval exponent for
    "p + 2 = P_2" (a harvester line) and file the disjunction as parity-blocked at every
    scale the zone needs; (ii) nothing else.

---

## Reading the file as a whole

1. **Every ROOT node here is the engine's record at another rung, read in another coordinate.**
   The ledger's turn m is the top 1/(m + 1) of the window of sqrt((m + 1)Q) (Theorem (E) plus
   the exhaust cap); the manifold's record is at most the engine's record at sqrt(3Q); the
   quiet-zone record is the same at m = 1; the skip half is the record in gear units. The wall
   at the ledger is face A because the fuel's only unused property, phase zero at every prime
   up to Q, has as its only known window-scale consequence a count.
2. **The counterfactual axis is exhausted, and it left two exact laws and one new adversary.**
   Balance on every engine tooth and mirror closure of the valve set are which-residues laws
   of the real fuel that F = 1 mod 3 violates (checked: the counterfactual's valve set is
   exactly the half-inventory s' = s + 2 mod 3). The one-tooth free-phase manifold adversary
   has both and empties 60 turns with 776 gears (N11); its full reach R(q, Q) is the root at
   rung Q in the exhaust's coordinate and a third column for the three-record table (real
   teeth, this, free class). Neither law is a route; both belong in the valves' parts ledger.
3. **The one object in this file that is not the record restated is its dual, the rich-interval
   function of the pullback along the new gear.** It is the exact cap of the skip half (E2/E3,
   slack 0 or 1 at nine of nine rungs, tight at two), it is universal and finite per n (gears
   above 2n do nothing), it is a max over translates and not a count, its modulus grows with the
   machine, and it says what a record of M + q' is made of in a coordinate the merge grammar
   never used: a poor interval of M whose openings form a rich interval of M's twist along q'.
   The candidate most of the blockers point at is therefore the record's TWO-COORDINATE
   structure (poorest along the column, richest along the new gear's stride), and the
   pullback's coherent separations are the only untested place where the real one-third
   separation could matter (measured worthless for F and K, never measured for Omega).

**Brief to launch (one paragraph).** Open "the rich half of the record": for the machines
m19..m53 and each next gear q', compute the pullback M^{(q')} (separations 2 u_g q'^{-1} mod g)
and its rich-interval function Omega^{full}(n) exactly for n <= 20 with all gears (only gears
<= 2n bite; the one-orbit reduction makes it a free-phase maximum, ILP or exhaustive per n);
sharpen E2 to L + 1 <= 2 Omega^{full}(T + 1) and score its tightness against L at every rung
(E3 is tight at two of nine; a law would be tight at most); at every record fusion on file, test
whether the openings of M inside the record stretch attain Omega^{full} at that translate
(records made at richest translates), and whether the arc floor of docs/proofs/21 has a pullback
form under the coherent twisted separations. Pre-register: (P1) Omega^{full}(n) falls strictly below
the corridor value Omega_{5,7}(n) first at some n_0(q') <= 12 at every rung, decided by the
gears 11..23 whose pullback arcs exceed n_0/2 (a gear can be phased off an n-window only if
both its teeth fit in the complementary g - n residues); (P2) the sharpened cap is tight at at least five
of nine rungs; (P3) the record stretch sits at a richest translate at every rung from m23 on,
which would make the record's start class computable mod q' x (gears <= 2T + 2). Refutation of
P3 at one rung with the deficit named is the finding; confirmation names the record's position
as a formula in a growing modulus. Engine only; the manifold, valves and exhaust untouched; two
cores, the r67 instruments (pc_skip.py, the closure dictionaries) suffice.
