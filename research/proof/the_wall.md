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

## 6. What the wall says is NOT worth another branch

Anything that (a) reduces to a count, (b) lives at a fixed modulus, (c) looks for the real
machine to be special by symmetry or squareness, (d) needs transfer for real q, or (e) proves
a twin in a short interval. Every branch closed on the tree is one of these, and their
closures are the measurements above.
