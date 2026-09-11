# The wall: every blocker on the tree read together (manager, 2026-09-06)

The owner's instruction: the blockers are not dead ends, they are the edges of the wall we are
trying to break through; read together they outline the shape of the target and show where it
is thin. This document keeps that map. Every entry states the blocker precisely, where it was
established, what it forbids, and what it leaves open. Vocabulary as in the theory tree's
profile. Numbers are the record's; nothing here is new computation.

## 1. The target, stated once

Root: for every machine {5..y} an opening lands in the window (y, y^2]. Kernel-equivalent to
twin primes infinite. Three formulations on the tree: per step (the ladder), whole window,
structure of the record. Everything measured says the target holds with large slack:

- the record F is a quarter of the window at every computed machine (F/W = 0.25, 0.25, 0.25,
  0.23, 0.30, 0.28, 0.27, 0.25, 0.31, 0.30, 0.28, 0.25, 0.25 at y = 7..53), so the longest
  blocked stretch anywhere in the period is four times shorter than the window;
- the walk from q^2 lands on a twin within 2..79 columns to q = 100,003 and within 265 to
  q = 5,000; the section is thousands of columns;
- the island witness: for every integer coprime to 30 above 2849 an offset 12 mod 35 past
  q^2 within 0.152 of the top gear's arc is open, 0 exceptions to 200,000, minimum count of
  open islands rising 2, 4, 12, 21, 57, 107 by band.

So the wall is not between us and a marginal truth; it is between us and a truth that holds by
a factor of four (in length) or by dozens (in count). That is the first thing the shape says:
the proof does not need to be sharp.

## 2. The blockers, each stated precisely

### Face A: counting cannot see it (parity)

- A1. Class-count-only sieve bounds at the window's scale. Any bound on the window statement
  that uses only how many residue classes each prime removes (two) and not which, is a
  dimension-2 sieve; its lower function vanishes below s = 4.27 while the window sits at
  s = 2. The two-class transfer of Iwaniec gives F(y) <= C y^4.27, not C y^2; explicit finite
  certificates are 1.7x to 35x over budget. A class-count bound with constant below 1/6 IS
  the twin prime conjecture (Ziller-Morack Conjecture 6). (Branch 3a, iwaniec_two_class.md;
  docs/novel/j2-upper-bound.md.)
- A2. Fixed-depth counting, capacity counting, overlap counting on the real record. The
  strike budget sum of 2/g is always enough to cover the record stretch; the overlap the
  real teeth force is nearly achieved by the record, so no slack. (Dead-ends list.)
- A3. Counting through islands. Large gears strike islands at exactly 2/g, so the counting
  margin through the island set equals the unrestricted one and crosses 1 at q = 53. The
  cover-side first moment equals the depth function's product, 10^19 at q = 1487.
  (reachability.md N-R5; cover_number.md.)
- A4. The rate-to-maximum step. Every rate the machine has is proven exact (2/g per gear;
  the doubling law, exactly 2 chi classes; full-period equidistribution with error below
  3^m). Every branch that reaches a rate then needs "the maximum does not exceed what the
  rate suggests", and that step is never available: the renewal ladder, the suppression law,
  the walk-length null (the twin-gap null to 2%), the section spectrum (the gear lines).

What Face A forbids: any proof whose only input is densities, counts, or class numbers.
What it leaves open: proofs that use which residues, or covering arguments that are not
sieve bounds.

### Face B: position cannot see length (escape distance 1)

- B1. Residue arithmetic at any bounded modulus certifies positions, never sizes: the
  corridor mod 35, the 12-of-24 forbidden classes mod 210, the slot rule F mod 5, the
  gear-5 lock, the record's phase pinning, the 15-class law for the walk length mod 35.
  All forced, all exact, all positional; the record escapes any bounded-modulus constraint
  by one column. (Dead-ends list; 5e, 5g, 5d, walk_path.md.)
- B2. The zero mirror and the region past zero. The pattern near zero is thinner than the
  period mean (0.79 of it asymptotically), not richer; any statement about (0, W] provable
  from tooth positions is a statement about the twins below Q'^2. (7d.)
- B3. Records are made at the ends, not the middles: ordinary lower gaps fused at their
  junctions by exactly three top gears (m29: 10 + 10 + 23; m31: 23 + 10 + 25). That says what
  a record is made of, with no length in it; F = flank + letters + flank is the merge grammar
  restated. (R3.h.)
- B4. The hinge. Every window stretch has a column struck by one gear, but no length rule in
  that gear can exist: at fixed stretch the hinge gear falls as the machine grows (877 -> 409
  at length 241). (5g.)

What Face B forbids: any proof whose object is where things sit modulo something fixed.
What it leaves open: objects whose modulus grows with the machine.

### Face C: the real machine is typical (no hidden structure)

- C1. Symmetry: the symmetry group of the opening set is exactly Z/2 (the mirror). (Kernel.)
- C2. Coherent spacings: the real one-third tooth spacing gives an F at the 14th-22nd
  percentile of random symmetric spacings; coherence explains nothing. (Branch 6.)
- C3. The square phase vector: real vectors q^2 mod g, locally-square vectors, and random
  vectors fail the island witness at the same rate (0.9984 +- 0.0033 over 6.3 million each);
  index parity is worth 1%; the QR screen adds no factor. (R2.a.i.a.1.b.)
- C4. The walk from q^2 is a typical tooth start in length (percentile 0.53) and the section
  spectrum is just the gear lines. (walk_transforms.md.)
- C5. The one measured difference between the real numbers and any phase model is the
  sifting level: at s = 2 the real machine has 21% FEWER openings than the model (the
  classical 4 e^-2gamma), which points away from the target, not toward it.

What Face C forbids: a proof that finds the real machine special among its family by a
symmetry, a spacing, or the squareness of its phases.
What it leaves open: the specific teeth as arithmetic (which residues), not as a symmetry.

### Face D: transfer (rare is not never)

- D1. The island witness is generic: covers exist for random vectors with probability
  1.3e-1 at d = 60 falling to 3.3e-7 at d = 1100, and the real vectors are typical (C3).
  Failure probability is positive at every arc; the first moment with the s = 2 correction
  predicts 16.5 failures against 17 observed, all below q = 2849.
- D2. A cover is realised by exactly 2^K residue classes of q modulo a product of K gears
  that exceeds q^2 (proved), so a failure pins q^2 as an integer; but there are about 2.7^m
  covers (10^54 at d = 1120) against a class density of 10^-30. Counting classes over covers
  is vacuous by 10^24. (cover_number.md.)
- D3. The transfer needed is equidistribution of q^2 (or of q) in structured sets modulo
  products far above q^2, beyond Bombieri-Vinogradov range and beyond any known theorem.

What Face D forbids: proving the witness for real q from its rarity among all vectors.
What it leaves open: statements that hold for ALL phase vectors (adversarial), which need no
transfer at all.

### Face E: every local formulation over-asks

- E1. The per-step ladder contains a twin-Bertrand postulate: F(M+q') >= F_2(M) >= 2 d_0(M)
  is a theorem and d_0 is the column of the first twin above the top gear, so any per-step
  bound implies a twin below a bound in q. (Branch 1e, prover A.)
- E2. The chain statement needs the real higher gears' teeth: every ingredient set short of
  the real machine has counterexamples (2f refuted at 23 -> 29: 62 > 61).
- E3. The section statement (a twin in every section) is stronger than twin primes: a dead
  section is a twin gap of order 4 sqrt(x). (word-tree, anchor-235 section 7.)
- E4. The walk-frame statements L < d, and the island witness inside the arc, are twin
  primes within about q/3 numbers of q^2: twin-Bertrand strength at scale q/3.

What Face E forbids: proving the target by proving something local and stronger.
What it leaves open: the whole window, all of it, as the target; it is the only formulation
that does not over-ask, and every measured quantity says it holds by a factor of four.

## 3. The shape the faces make

Put together, the faces say what the proof must be:

1. It must use which residues the primes strike (A), not how many.
2. Its object must live at a modulus that grows with the machine (B), not a fixed one.
3. It must not rely on the real machine being special by symmetry, spacing, or squareness
   (C); the specific teeth can enter only as arithmetic.
4. It must hold for every phase vector, or it must not need transfer (D).
5. It must be about the whole window, using the factor-four slack (E), not about a short
   interval near q^2 or near zero.

Conditions 1, 4 and 5 together name one object, and it is already on the tree: THE
ADVERSARIAL COVERING NUMBER. Let K(d) be the least number of gears (any primes above 7,
each with any reachable phase, the two classes at the gear's own fixed separation
2 x 6^-1 mod g) whose strikes cover every island of an interval of length d. The window
statement follows from

    K(W(q)) > pi(q) - 3   for every prime q   (the gears of {5..q} above 7 are pi(q) - 3),

because then the real machine, which is one adversary with one phase per gear, cannot cover
the islands of its own window, so an island is open, so a twin. This statement needs no
transfer (D, it quantifies over all phase choices), uses the fixed separation (A, it is
"which residues": two classes at a fixed offset, one phase), lives at a modulus that grows
(B, the product of the cover's gears), does not need the real machine to be special (C, the
real machine only has to be ONE adversary), and is about the whole window (E). It is
combinatorial: a covering-system lower bound, not a sieve bound.

## 4. What is measured about that object

- K(d) is exact at 23 arcs to d = 1330 (3 at d = 35 up to 22 at 1330), ILP-certified;
  achieved covers give K <= 26, 32, 40, 46 at d = 1750, 2240, 3360, 4480.
- The counting requirement (sum 2/g >= 1 over the K smallest gears) is bounded: 10 gears
  cover any length by count. K grows past it. The growth is bought by two things and only
  two: one phase per gear (the larger half) and the fixed separation (a factor 1.5); with a
  free separation the optimal cover is a perfect partition equal to counting (4 arcs, 0
  exceptions). So the fixed separation and the one-phase rule are exactly what makes covering
  harder than counting, and they are exactly the real teeth.
- K depends on the island count and the cheapest gear the bar leaves, not on the arc.
- Optimal covers contain all of 11..31 from d = 385; every gear covers at least two islands.
- Needed: K(W(q)) > pi(q) - 3 with W ~ q^2/6, i.e. K(d) > pi(sqrt(6 d)) - 3 ~ 2.4 sqrt(d)/ln d.
  Measured at d = 1330: K = 22 against 21 needed (q = 89). At d = 4480: K <= 46 against 35
  needed (q = 163); the exact K there is boxed in [22, 46]. Written as K(d) ~ pi(sqrt(c d)):
  c ~ 7.1 at d = 1330 and c <= 11.5 at 4480, against c = 6 needed. The all-columns version
  (cover every column, not only islands) has c = 24 from F/W = 0.25: a factor of four in
  length, a factor of two in gear count. The island restriction spends most of that slack;
  the plain-columns version keeps it.

So the tight quantity is the constant c in K(d) ~ pi(sqrt(c d)): 6 is the target, 24 is
measured for whole columns, 7 to 11 for islands. The fit d/(ln d)^3 over d <= 4480 cannot
persist (K_island <= K_columns ~ sqrt d), so the measured growth is the pre-asymptotic part of
a sqrt(d) law with a constant above 6.

## 5. Weak points

W1. The adversarial covering constant. The statement "the primes up to q, each removing two
    residue classes at its own fixed separation with one phase, cannot cover an interval of
    q^2/6 columns" is Ziller-Morack Conjecture 6 in fixed-separation form. Face A says sieve
    methods cannot prove it (parity, exponent 4.27). Nothing on the tree or in the record says
    a COVERING-SYSTEM argument cannot: the covering literature (Erdős covering congruences;
    the minimum-modulus theorem, Balister-Bollobás-Morris-Sahasrabudhe-Tiba) works with
    distinct moduli and small classes, which is our setting, and it is not a sieve. The
    measured mechanism (growth bought by one phase per gear and by the fixed separation) says
    where an argument must bite: a gear used once at a fixed separation wastes a fixed
    fraction of its strikes on columns already struck, and the waste is forced, not
    statistical. That is the overlap lower bound, dead for the real record (A2) because the
    record nearly achieves counting, but measured at a FACTOR OF TWO for the adversary on
    islands (K = 20 against counting 10 at d = 1120). Overlap is dead as a bound on F; it is
    alive as a bound on K. Nobody has tried to prove it.

W2. Whole columns, not islands. The island version spends the slack (c from 24 down to 7-11).
    The plain version, K_columns(d) > pi(sqrt(6d)), has the factor four and is exactly F < W
    for every member of the one-phase fixed-separation family. The family measurements say
    the real machine is NOT special in F (C2), so the statement to prove is about the family,
    and the family is a finite combinatorial object at every d: the minimum number of primes
    whose fixed-separation pairs can cover d consecutive columns. Its first values are the F
    ladder read backwards (K_columns(d) = the number of gears of the smallest machine with
    F >= d): 5, 7, 11, 18, 25, 34, 43, 58, 88, 91, 103, 118, 145, 161 at 2..15 gears.

W3. The place where two faces disagree. Face C says the real machine is typical; Face E says
    every local statement needs the real teeth (2f refuted by a member with real 5 and 7 and a
    pinned tooth). Both are measured. They are consistent only if what the real teeth supply
    is not a symmetry but an arithmetic coincidence the family lacks: the separation
    2 x 6^-1 mod g is the SAME rational one third at every gear, so two gears' separations
    are compatible modulo each other in a fixed way (the corridor is the mod-35 shadow of
    this). Branch 6 tested coherence as a driver of F and found nothing; it did not test it as
    a driver of K, the adversarial cover. That test is cheap: K(d) for the family with random
    separations against K(d) with the real one-third separation, at the same d.

W4. The slack itself. Every measured statement holds by a margin that grows with q (open
    islands 2 -> 107; F/W flat at 0.25 while W grows as q^2). A proof needs only a fixed
    positive margin, and needs it only eventually (the finite part is certified by the
    ladder). The shape says: aim at the weakest true statement, "some island in the whole
    window is open", not at the arc, not at the section, not at the step.

## 5a. Update after W3 (2026-09-06)

W3 answered: the real separation does not drive K (it is the mode of the random distribution at
every arc; coherent separations give the same K). W1 is dead for lack of slack: on islands the
target K(d) > pi(sqrt(6d)) - 3 is met by one gear at d >= 560 and with equality at 140 and 280.
W2 stands, with a correction: for whole columns the adversary with one phase per gear over all
primes up to q is exactly the real machine over its period (every phase combination occurs once
per period), so K_columns(W(q)) > pi(q) - 3 IS F(y) < y^2/6, the root, in covering language. It
is not easier; it is the same wall seen as a covering problem, which is the one framing face A's
sieve "no" does not cover.

The free-residue adversary of face A has a PUBLISHED TABLE (added 2026-09-06, harvester r1).
Face A's strictly stronger adversary - two ARBITRARY classes per gear instead of the real teeth
at separation 3^{-1} (mod g) - is exactly Ziller-Morack's h_2, and its record is OEIS A072753,
19 terms (gear sets {5} to {5..73}): 2, 4, 10, 24, 31, 42, 60, 74, 94, 117, 148, 173, 213, 236,
275, 316, 364, 409, 436, with A288815 = 6 A072753 + 6 the integer form. Against the real
machine's F(M) - 1 = 1, 4, 6, 10, 17, 24, 33, 42, 57, 87 on the same sets: equal at {5,7} alone,
the free adversary winning by a widening margin thereafter. So face A's "the free-residue
adversary is strictly stronger than the root" is not only true, it is tabulated, and its window
statement h_2(n) < p_n^2 - p_n is a named open conjecture (Ziller-Morack Conjecture 6) that
implies the project's window statement and, by their Theorem 4.1, Goldbach. Verification:
research/harvest/r1/jacobsthal_check.md.

W5. The unfitted brick (the owner's reading of R3.h). A record is ordinary lower gaps (bricks)
glued at junctions by the top gears' teeth (mortar). Bricks and mortar are proven objects (the
merge grammar, the chain law, the bare-word cap of six). The unbounded part is the two flanks,
and a flank is a walk: from a column where a top gear's tooth lands on an old opening, walk in
the old machine to the next old opening on each side. Proven about that two-sided walk: the
left tiling is the right tiling negated gear by gear (L6). Measured: the flanks sum to less
than the budget at every rung; the suppression law. Never decomposed the way the path from q^2
was. That is the next brick to pull apart.

## 5b. Update after W5 (2026-09-06)

W5 closed by a theorem: junctions are ordinary openings (the junction condition is a congruence
mod q' and the old machine is periodic mod P coprime to q'), so the flank brick is F_2(M) itself
and cannot be fitted by structure at the junction. What the flank decomposition gave, exact: the
window has at most two junctions, the column of q' and the column of q'^2, and their flanks are
d_0 (the twin-Bertrand quantity) and the walk from q'^2 (the square-gate walk); the length of a
flank is decided in the middle band of gears, which strike at a constant rate 0.796; the flanks
are coupled by the anchor's residue classes, not by the negation lemma; and no bound in terms of
the buckets at the junction holds (the only exceptionless rule uses the gears the walk misses).

The wall after all five weak points: W1 dead (no slack on islands), W2 is the root in covering
language, W3 answered (the real teeth are typical), W4 stands (the slack is a factor of four on
whole columns and every margin grows), W5 closed (the brick is the pair statement). The one
framing the sieve "no" does not cover is the covering problem on whole columns with one phase
per gear at the fixed separation, which is F(y) < y^2/6 itself; the one fact that distinguishes
the real teeth in any adversarial measure is the tail gears' tooth distance (0.69 of the arc,
outside the random range), and it is too small to move a cover number.

## 5c. Update after the unstick round (2026-09-06)

Thin place 2 (the level-3 dictionary, via the glue): the neighbour profile gave N(v) <= F_2(M)
for every gap size v >= 6, exceptionless to m31, with the glue lemma proved as mechanism; but
the glue as a covering statement is false where it matters (the m29 run (18, 10, 30) resists
every construction; the glue's whole content is one column, the shadow), and the F_2 cap cannot
close the chain statement (needs F_2 - F <= a, false at m17 and m29). Kept: the J-run outer law
(g_1 + g_J <= F_2 whenever every middle is >= 6, 3.3 million runs) and the shadow and move
lemmas. Thin place 4 (separation compatibility): dead; fully compatible members violate the
budget, and coherence raises the violation rate. Face C acquires its first exception: the real
teeth are atypical in gluability (99.6th percentile of the family). Face C should now read:
typical in every symmetry, spacing and squareness measure, atypical in how separably its flanks
are struck (left flank by one set of gears, right by another). That is a which-residues
property, allowed by face A, and it has not been followed.

Thin place 6 measured (separability.md): dead; gluability is not separability, the shared gears
are 5 and 7 (the top gears are the free ones), the one-third separation maximises sharing, and the
face-C exception shrinks to a factor 2.4 at matched cells. Two exact facts kept: the letter gears
of a middle gap v are the prime factors of 3v - 1 and 3v + 1; and the run that resists every local
certificate at m29 is the m31 record class itself.

Open thin places: 1 (count gears, not columns: the forced-striker set grows with the span; now
also pointed at by separability: how many gears carry no sharing obligation), 3 (the record class
as a formula), 5 (moments over q for the island witness).

## 5d. Correction after gear_count.md (2026-09-06)

Section 5a said the whole-column adversary with one phase per gear over all primes to q is the
real machine over its period, so the covering statement is the root. That is true only when the
gear SET is fixed to {5..q}. The adversary that also chooses the gears is strictly stronger:
the best 4-gear machine blocks a span of 16 ({5, 7, 11, 17}) where {5, 7, 11, 13} blocks 11,
because the mechanism is the arc, not the size (a gear beyond its umbrella contributes only a
bare domino of length a_g = 2u_g, its size invisible). The real machine is a WORSE coverer than
an adversary with the same number of gears, by a factor that falls with K (1.45, 1.22, 1.12,
...), because 3 a_g = g -+ 1 makes twin gears share an arc, so {5..q} carries only
pi(q) - 2 - pi_2(q) distinct arcs and must buy both members of every twin pair. Two consequences
for the wall: (i) the arc multiset is a which-residues handle where the real machine is
measurably on the good side, the only such handle found; (ii) the proven gear count (forced
strikers) saturates at spans of about 2q/3 and is useless against the window, so thin place 1
is dead in its proven form; what remains of it is the open lemma A(K) < (p_{K+1}^2 - 1)/6 (the
longest span any K-gear machine can block stays below the next prime's window), which is
stronger than the root and has no upper bound on the tree. Open thin places: 3 (the record
class as a formula), 5 (moments over q), and the arc multiset (why sharing an arc costs the
real machine, quantified: the real minimum cover of a window stretch is 1.5-2.1 times the free
minimum for the same span).

## 5e. After the second unstick round (2026-09-06)

All thin places tested. 1 dead (forced gears saturate); 2 dead (the glue buys one column); 3
dead in its sharp reading (the record is not at the alignment points); 4 dead and reversed
(coherence is a liability); 5 dead by proof (a bound below 1 on the failing fraction is the
conjecture); 6 dead (gluability is not separability); the arc multiset dead and reversed (twin
gears are the cheapest small gears; which arcs is worth nothing). The adversarial lemma
A(K) < (p_{K+1}^2 - 1)/6 is exact to K = 12 with margin 2.7-3.8, flat, and its residual is the
capacity statement h_S(L) with a gear count: face A.

What the wall now says, in one sentence: every route that starts from the machine's structure
ends at one of two statements, "the primes up to q cannot cover q^2/6 consecutive columns"
(a Jacobsthal-type covering bound at exponent 2, which no sieve reaches and no covering method
has been tried on) or "rare among all phase vectors transfers to never for real q" (beyond
equidistribution), and the machine's structure (positions, arcs, phases, symmetries,
twin-gear duplication, separability) has been measured not to shorten either. The one
untried technique named in the record is the distortion method of covering systems (W1 of the
first map), a literature-and-construction lane.

## 5f. The covering side, measured (2026-09-06)

The distortion method (the untried technique) applies to the machine unchanged, with a budget
sum 4/g^2 < 0.365 that never saturates; localised to an interval it dies by the collapse lemma
(fibres of one column once the modulus exceeds the interval) and needs a level of distribution
at dimension 2 to survive: the parity barrier reached from the covering side. So face A now has
two faces of its own: sieve (dimension 2, exponent 4.27) and covering (collapse; shortest
addressable interval exp(theta(q^0.73))). The claimed positive (the localised budget proves the
adversarial lemma for K <= 10) was FALSE: the localised inequality fails at {5, 7, 11, 17}
(covers 15 columns at eta = 0.693) and the tabulated budget is the union bound. The lemma for
K <= 10 is now PROVED another way (docs/proofs/20): certified infeasibility of an exact 0/1
program, with the span lemma and the head collision as new reasoning tools. One crack named: a second moment over arithmetic blocks instead
of congruence fibres. The half-column map added three theorems (fibre, fixed point = twin
column, uncoupled distances = twice the twin columns above the machine) and no lever: coupling
constrains strikes, a gap's endpoints are openings.

## 5g. The one-block inequality, and its withdrawal (2026-09-06)

The block prover left the covering half as sum over gears of (4/g^2) rho_g^2 < 1, rho_g the
strike-rate excess of g on the current survivors. The manager's second check: each term is
alpha_g^2, the square of the fraction of the CURRENT survivors that g strikes, so the inequality
is trivially valid (it says no gear strikes all the survivors) and its whole content is in
bounding alpha_g, whose denominator is the actual survivor count. Per-gear upper bounds with any
constant above 1 fail by accumulation: the kills total W(1 - delta_q), essentially the window,
so the induction needs the exact fair share at every gear, which is the root. The earlier
"trivial bound short by a factor 3.7" assumed the fair share in the denominator: withdrawn. The
block form of the distortion engine is the trivial criterion; only the fibre form has content,
and it collapses on an interval. Face A stands with both its sides (sieve and covering) and
nothing in this round has moved it.

## 5h. The order of interaction (2026-09-06)

The collision laws (R2.d.i) give the wall a third precise form beside the sieve dimension and
the transfer: the least order of gear interaction whose joint maximum falls below the window
grows like K - 3 (1, 2, 2, 3, 4, 5, 6, 7, 8, 8 at K = 3..12). Pairwise laws prove the
adversarial lemma at K = 4, four-gear blocks at K = 5 and 6, and no bounded order reaches all
K; a new gear's net contribution to coverage turns negative once the small gears' strike sum
passes one half, which is why the pairwise form is invalid from K = 9. Any proof by
interactions must therefore be an induction whose order grows with the machine, and the
tree has no such object. What is proved on this side: the adversarial lemma to K = 10 by
certificate (docs/proofs/20), the shared-arc law (twin gears collide at (g + 4)/3), the arc
floor, the linear deficit law with slope 4/(gh).

## 5i. The period-scale formulation (the owner, 2026-09-06)

Drop the window. The openings of machine q over one period are exact and every later gear's
first moment on them is exact to 3^m out of e^q; every joint moment is exact while the product
of the gears involved stays below the period. That is level of distribution 1. Faces B, D and E
disappear at this scale; face A stands alone: sifting the openings by the primes up to the
square root of the range is s = 2 in dimension 2, below the limit 4.27, and no structure of the
window was ever what stopped it. The new object: the second machine (gears above q) acts on the
first machine's openings as a union of coherent twisted copies of the first machine, one per
later gear g, with separation 2 g^-1 at every gear. Whether that union carries bilinear (Type
II) structure the sieve cannot see is the only question the window could not ask. MEASURED
(period_scale.md, q <= 23): it does not, at these sizes. With every other obstruction removed, face
A shows its true form: the Brun main terms alternate and do not converge at s = 2 (the order-2
error equals the order-3 term); the exact pair terms give the same number as the generic
Bonferroni bound; switching is an identity. And the placement residue law names the barrier in
the machine's own terms: a top gear's placement on the track is a dimension-1 event, a double
placement (a twin) is dimension 2.

## 5j. The window line at the end of its leads (2026-09-06)

Measured from five sides (the pair statement, the neighbour profile, the gate and its row, the
pinned letter, the record as a 2-run), the budget's tightness inside the window is carried by
3- and 4-runs of ORDINARY old gaps (rank 0.27-0.83 of the spectrum) with a letter in the
middle, in the band [15, 36] at 29 -> 31, and no local certificate exists for that band. What
the line produced: the gate closes at the certified row top (proved scan-free by LP duality at
m19, m23, m29; by CRT search one machine beyond every scan); the record is saturated (every gear
a sole striker inside it); the top of the spectrum is pinned to F_2; the exact ladder now
reaches F(37) = 88 and F(41) = 91 from m23's period alone, with the budget slack 14, 20, 16, 7,
38 along it; the pinned letter is refuted at 37 -> 41. Instruments in hand: the closure step
with the span-threshold prune, the CRT row search, the configuration enumerator. The one
object still open inside the window: the chain statement at depths 3 and 4 on the band. Per
the owner, this is the point to open the manifold (then called the top machine) on its own
terms (R4).

## 5k. Location pinpointed (2026-09-06)

The owner's last window round asked where in the window the twin slot is, with the engine
(then called the lower machine) only. Answer, proved and measured: the window can be emptied
only from the bottom.
Theorem (E): a column above q is blocked under {5..q} iff blocked under {5..sqrt(6k + 1)}, so
the effective machine at every column is exact and no blocked stretch of length L can begin
before 1.25 L (unconditional to 59^2, conditional on the ladder beyond); measured 3.25 L with
no exception. From q = 1427 the longest blocked run of the whole prefix is the initial run
from column 1, of length q/6 (the first twin above q), while the window is q^2/6. So the window
statement is exactly d_0 <= W, the diagonal walk of the engine, the first twin above
q; every structured family of candidates carries twins at the window's own rate by an identity
(a rule written in the engine's residues cancels its own saving); the manifold is
irrelevant inside the window. The location is the bottom; the mechanism there is the primes
themselves (twin gear pairs striking their home columns, then composites of small factors),
which is twin-Bertrand at scale q. The engine alone has said everything it can about
where; per the owner, the manifold (then called the top machine) is next.

## 5l. The wall in the canonical words (harvester, 2026-09-07)

No new claims; the faces A-E and the updates 5a-5k mapped onto the four parts (engine = the
primes up to q; manifold = the primes in (q, q#] on the raw line; valves = the engine acting
inside the manifold's open set; exhaust = every tier above the manifold). Older sections above
say motor, wheels, top machine, bottom machine, lower machine and clutch for these.

- Face A (counting) belongs to the METHOD, not to a part: any argument whose only input is how
  many classes each gear removes is a dimension-2 sieve at s = 2, and 5i shows it standing alone
  once the engine's openings are taken over one full period (level of distribution 1). A4, the
  rate-to-maximum step, is the same face wherever a rate of any part is proved exact.
- Face B (position cannot see length) belongs to the ENGINE: the corridor, the gear-5 lock, the
  slot rule, the record's phase pinning and the hinge (5g) are all engine facts at a bounded
  modulus. 5k is the engine's last word on position: the window statement is d_0 <= W.
- Face C (the real machine is typical) belongs to the ENGINE: symmetry (the mirror), spacing,
  squareness and the cover number are engine measurements; the one exception, the gluability
  knock (5c, 5e), is an engine fact about which residues its teeth strike.
- Face D (transfer) belongs to the VALVES: the island witness, the cover number and the second
  moment (5e) are statements about the engine's openings just past q^2 under gears above q,
  i.e. the engine acting in the manifold's range; 5i says the transfer face disappears at the
  period scale, where the valves' coupling is exact.
- Face E (every local formulation over-asks) belongs to the VALVES: the twin-Bertrand quantity
  d_0 (E1), the section statement (E3) and the walk frame (E4) are all about a twin appearing
  where engine and manifold meet; the chain statement (E2) is the engine's own.
- The covering side (5a, 5d, 5f, 5g, 5h): the adversarial covering number, the free-gear
  adversary A(K), the distortion method and the collision laws are statements about the FAMILY
  the engine belongs to, worked in the engine's coordinate; the order obstruction (5h) is the
  wall's third face, (O), and it is an engine-family statement.
- 5j (the window line at the end of its leads) and 5k (location pinpointed) are the engine
  alone; 5k's closing sentence, "the top machine is next", opened the manifold.
- The MANIFOLD and the EXHAUST have no face of their own on this wall: their records are the
  root in their own coordinates (the quiet-zone record is a twin gap; the exhaust's record is
  family (1, 1)), and nothing in either part bounds a twin-free run. That is a fact about where
  the difficulty sits, not a fifth face: the wall stands at the valves, and it is face A when
  seen from the period scale.

## 6. What the wall says is NOT worth another branch

Anything that (a) reduces to a count, (b) lives at a fixed modulus, (c) looks for the real
machine to be special by symmetry or squareness, (d) needs transfer for real q, or (e) proves
a twin in a short interval. Every branch closed on the tree is one of these, and their
closures are the measurements above.

## 5m. Existence without count or position (owner, 2026-09-07)

The owner: "we don't have to see position, although that would be strong; just knowing there
exists any position regardless of where it's located would be proof." Exactly right, and it
sharpens the wall. Every existence proof found so far has one of two shapes. Existence by count
(the number of open pairs in the window is positive) is face A: a two-dimensional sieve that
stops at 4.27 windows and never reaches one; the turn ledger's counterfactual fuel proves it again
inside the valves. Existence by construction (a position the machine cannot strike) has always
produced its position at the scale of the period W (CRT, the mirror, the symmetry group, the
richest translate), never inside (y, y^2], a vanishing fraction of W; inside a short interval,
"there is one" without a count means "here", and position never sees length. So the wall is:
existence inside a short interval with neither a count that reaches it nor a location that
survives. The third shape, the target in the owner's words, is an invariant the window carries
by construction that forces an opening without naming the column, the way column 0 is open in
every machine because nothing strikes 1. Column 0 is the trivial instance at position 0; the
search is for a second such object at the window's scale. The island witness (offsets 12 mod 35
past q^2, 0 exceptions to 200,000, no mechanism) is the only candidate on record that is neither
a count nor a period-scale construction.

## 5n. Theorem (E)'s exact hypothesis (kernel, round 39)

Theorem (E) holds for columns inside the next prime's square (6k + 1 < q'^2) and fails at the
square column and beyond: the refuting instance q = 5, k = 8 (47, 49) is in the kernel
(`OneStepE.E_needs_prefix`). Every use of (E) in this document (5k, the effective machine at a
column, the location pinpointed to the bottom) was inside the window, where it is a theorem
(`OneStepE.blocked_iff_sqrt`). At one step the new gear's only new strikes below its square are
the home column d_0(M), iff (q', q' + 2) is twin, and the square column, iff q'^2 - 2 is prime
(`OneStepE.new_iff`); every maximal run strictly inside the prefix is inherited (`maxRun_succ`).

## 5o. The wall after the finalisation round (manager, 2026-09-11, late)

The single proof document (proof_skeleton.md, Parts I-IV) has one unproved statement, step 8:
between every cut and its square there is a twin pair, i.e. the primes below p_{k+1} cannot
strike every column between p_k^2 and p_{k+1}^2. Its faces were closed one by one today, each
with a built witness, and they meet at one place:

- Length (research/proof/length_face.md): F(q) < q^2/6 would give 8 and is the twin prime
  conjecture in covering form. The parity twin is built: the open columns whose members have
  Liouville product -1 are empty below q'^2/6 and sieve-indistinguishable from the open set
  (0 of 1,326 cells above 3 sd). The real teeth add only the one-third separation (a factor
  1.3-1.8 in length, never an exponent) and break the sign symmetry only at the origin.
- Count (III.2, III.3): every count is matched by a machine with no open slot.
- Position (research/proof/first_realisation.md): only for the finer statement 8e; the first
  run of a length has no floor (p = 29); the gear set alone cannot give 8e.
- The origin (research/proof/origin_mechanic.md): the dilation form is exact and is the real
  teeth; the owner's nesting is TRUE and exact (the struck set is the disjoint union of the
  dilates g . R_g) and forbids nothing beyond the teeth; the recursion forces prime quotients
  below g^3 (a count); a THIRD counter-machine, the monoid generated by 5 and the primes
  = 1 mod 6, keeps dilation, hand-up and the square-root rule and fails 8 at [25, 961).

The axiom map, exact: the real machine = dilation + the finite fold 2, 3 + the hand-up; the
tooth family breaks dilation, V17 breaks the hand-up, the monoid breaks the fold; each is
necessary. The fold enters the construction only through the strike-class law (skeleton 2),
which is the sieve's input, and the sieve's input is insufficient at the origin. So a proof of 8
must use the fold in a way that is not the strike-class law: something the classes +-1 mod 6
and their dilates do that the monoid's irreducibles do not.

What is NOT worth another branch (added to section 6): any count; any record bound; any
position floor; the nesting as a constraint; the parity of Omega inside runs. Named and not
excluded: a non-count use of the tail pins g x m (m a small prime) for the finer statement;
for 8 itself the section is longer than every gear, so there is no tail and no pin. The wall
for 8 is now one sentence: the primes below p_{k+1} are exactly the irreducibles of the
survivors S = +-1 mod 6 under dilation, and nothing on record distinguishes that structure
from the monoid's except the fold's two classes themselves.

## 5p. After the fold lane (manager, 2026-09-11, late)

The missing axiom is not a property of the gear set. Thin the primes by exactly the twin
lowers: the thinned set keeps both classes at every scale, equidistribution, dilation, the
square-root rule and the side-swap rule, and its monoid kills every section; every two-class
monoid is transparent (8 holds on it iff it keeps a twin lower). What the thinning breaks is
the line: a twin (t, t + 2) with t removed is open and t is not a gear. So the axiom a proof
must use is the strike-class law's first half, the completeness of the line: every survivor of
the fold is a gear or a multiple of a smaller gear. The classes are invisible to the column
cover; the fold's sign n mod 3 is sieve-visible; the invisible part of Liouville is the count
of class +1 factors. The wall for 8 in one sentence, revised: the line is complete and the
sieve's input cannot see completeness at the origin; a proof needs completeness used at phase
zero in a way that is neither a free-phase cover nor a count.
