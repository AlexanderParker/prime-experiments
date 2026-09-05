# Dead branches reopened, second pass (manager, 2026-09-06)

The unstick protocol applied to the branches that died in the second and third rounds
(2g.i and children, 2f.i, R2.b and child, R2.a.i.a.1.c, R3.i, R2.c and child, and the sharp
reading of thin place 3). Format as in dead_branches_reopened.md. Constraints from the wall
(faces A-E) apply to every idea; the new constraints this round added are listed first.

New constraints (bricks from this round):
- N1. The one-block distortion inequality is trivially valid; any per-gear upper bound with a
  constant above 1 fails by accumulation (kills total the whole window). A covering argument
  must work with second moments over structure that keeps densities fractional (fibres) and
  those collapse on an interval.
- N2. Twin gears are the cheapest small gears: arc duplication helps blocking; which arcs is
  worth nothing; the real machine is an optimal 10-gear blocker.
- N3. Coherence of separations is a liability; fully compatible members violate the budget.
- N4. Junctions are ordinary openings; the flank brick is F_2(M).
- N5. Gluability is not separability; the shared gears are 5 and 7; the top gears are free.
- N6. Any bound below 1 on the failing fraction of the certified island object is the
  conjecture.
- N7. The letter gears of a gap size v are the prime factors of 3v +- 1 (column v/2); the
  fixed points of halving are the twin columns; uncoupled even sizes are twice the twin
  columns above the machine, depleted by 12-128x but present.

---

## 2g.i. The neighbour profile (F + 1 law dead; N(v) <= F_2 survives)

- **Object.** The two neighbours of a gap of moderate size.
- **Vectors.** Full-period profiles to m31; the glue construction; the family.
- **Failure.** N(10) = 48 > F + 1 = 44 at m29; the surviving cap F_2 cannot close the chain
  (needs F_2 - F <= a, false at m17, m29).
- **Idea 1: the profile of the SECOND neighbours.** N(v) caps the immediate neighbours; the
  J-run outer law caps the ends of longer runs by F_2 whenever every middle is >= 6. The unused
  freedom is the middles' sizes. Realisations: (i) tabulate, for J = 4, 5, the maximum of
  g_1 + g_J as a function of the middle SUM (not the middle count) and test whether it falls
  with the sum by a rule (the middle-sum lemma gives the sum >= k q' for legal words; a profile
  "outer sum <= F_2 - c x middle sum" would be a new cap); (ii) if the profile is linear in the
  middle sum with slope near 1, the chain statement's deep cases follow from it directly.
- **Idea 2: the cap as a two-sided walk from the MIDDLE.** N(v) <= F_2 came from gluing; the
  m29 resistant run is the m31 record. Realisations: (i) walk outward from the middle gap's two
  ends in the machine below and record, at each step, the smallest gear whose long arc is
  exceeded (the umbrella); the two walks' umbrella gears must be distinct until the arcs
  exceed the sum; test whether the flank sum is bounded by the point where the two umbrella
  sets meet (a gear-count cap on the pair, respecting face A since it counts gears); (ii) at
  the resistant m29 run, list the umbrella sets step by step and see which gear the two walks
  share first.

## 2g.i.a. The glue as a covering statement (dead; shadow and move lemmas)

- **Object.** A colouring of the gears that blocks the glued target.
- **Vectors.** Colourings, three-way splits, offset and cross glues, the family.
- **Failure.** The glue buys one column (the shadow), most gears cannot move (move lemma), and
  the resistant run is the next record.
- **Idea 1: glue in the OTHER coordinate.** The glue re-phases gears; the shadow is a column.
  Realisations: (i) glue at the level of the gap WORD instead of columns: the left flank's word
  and the right flank's word concatenated is a word of the machine iff its junction letters
  are legal; test at the resistant runs whether the concatenated word is realised somewhere
  in the period (a word search, exact at m11..m23), which bounds L + R by F_2 whenever it is;
  (ii) count how often the concatenated word exists when the column glue fails.
- **Idea 2: accept the shadow, bound it.** The glue's only miss is one column. Realisations:
  (i) prove N(v) <= F_2 + 1 by the glue plus the shadow (the shadow column is either struck or
  it is an extra opening that splits the pair into a triple with a 1-gap; a 1-gap's neighbours
  are bounded by the alignment law's domino structure), and check the constant against the
  measured tightness (N(7) = 55 = F_2 at m29); (ii) test whether the shadow column is ever
  open at the resistant runs; if never, find what strikes it.

## 2f.i. Separation compatibility (dead and reversed)

- **Object.** The real teeth's CRT compatibility as the chain statement's ingredient.
- **Vectors.** Incompatibility counts on all recorded violators; coherent sub-families swept.
- **Failure.** Fully compatible members violate; coherence raises the violation rate.
- **Idea 1: the violators' common property is in the TOP gears' phases, not in separations.**
  Every violator (compatible or not) is a chain: the new gear's teeth land on old openings with
  legal spacing. Realisations: (i) for every violator on record, list the residue of the
  violating stretch's start modulo the top three gears and test whether it is always the
  coverage-maximal phase (the allocation law of 5g holds at 340 of 348 REAL records; on
  violators it may fail, and a violator that is NOT locally optimal would be a new brick);
  (ii) test whether real machines are exactly the members whose records are locally optimal
  at every gear (a characterisation, not a proof).
- **Idea 2: the counting warning as a tool.** "The family cannot decide by frequency; only a
  construction can." Realisations: (i) build a violator at a rung where the real machine holds
  by re-phasing ONE real gear (the smallest change) and record which gear and by how much: the
  distance in tooth-moves from the real machine to the nearest violator; (ii) if that distance
  is always at least two gears, the real machine sits in a "basin" of the family; measure the
  basin's size by rung.

## 2g.i.a.i. Separability (dead; shared gears are 5 and 7)

- **Object.** Flanks blocked by disjoint gear sets.
- **Vectors.** Separation index, shared gears, the one-third separation's sharing, the record.
- **Failure.** Counting forbids disjoint covers below y = 109; the shared gears are the anchor.
- **Idea 1: separability with the anchor factored out.** Since 5 and 7 are always shared, ask
  separability of the gears above 7 only. Realisations: (i) recompute the separation index over
  gears >= 11 at every hard run; (ii) if the flanks are separable above 7 at the real records
  and not at the family's, the fact face C's exception was pointing at is "the anchor is shared,
  the rest is split", a which-residues statement about the anchor's corridor.
- **Idea 2: separability at the record's JUNCTIONS instead of its flanks.** Realisations: (i)
  for each record (as flank + letters + flank), which gears strike the letters' interior and
  which the flanks; test whether the letter gears (the top three) never strike the flanks'
  interiors (the "mortar gears are free" fact read as a separability); (ii) if so, the record
  is a product of two independent systems joined by three gears, and its length is the sum of
  two independent flank lengths plus letters: an independence structure the family may lack.

## R2.b and R2.b.i. Gear count and the arc multiset (dead; twins are cheap)

- **Object.** A length bound from a count of gears.
- **Vectors.** Forced strikers, minimum covers, the adversarial ladder A(K) to K = 12, arcs.
- **Failure.** Forced gears saturate; which arcs is worth nothing; A(K) is a capacity statement.
- **Idea 1: PROVE the ladder for small K.** A(3) = 7 has a one-paragraph proof by the hole-
  distance dictionary; the distortion budget proves A(K) < W_{K+1} for K <= 10. Realisations:
  (i) assemble the theorem "no K primes above 3, each with two classes at its fixed separation,
  cover the next prime's window, for K <= 10" as a written proof with the budget argument made
  explicit and checked, and file it under docs/proofs/: new mathematics, bounded, and the first
  proven statement of the adversarial lemma; (ii) push the constructive proof of A(K) itself
  (exact values) to K = 5 or 6 by the dictionary mechanism, to see whether the mechanism has a
  general form.
- **Idea 2: the ladder's RATIO, not its value.** A(K) / W_{K+1} is flat (0.26-0.37). Realisations:
  (i) test whether the ratio at K is decided by the smallest gear NOT in the optimal set (the
  first excluded prime), which the hole-distance mechanism suggests; (ii) if so, the ratio's
  flatness is a statement about which primes get excluded, a finite combinatorial rule.

## R2.a.i.a.1.c. The second moment over q (dead by proof)

- **Object.** A vanishing fraction of failing q.
- **Vectors.** Exact pair densities, Chebyshev, the divisor law of coupling gears.
- **Failure.** Any bound below 1 on the failing fraction is the conjecture (N6).
- **Idea 1: a WEAKER object that is not the conjecture.** Replace "some island open in the
  arc" by "some island open in the whole window" (the arc spends the slack). Realisations: (i)
  compute the failing fraction for the window object (islands over the whole window, of
  which there are about q^2/210) at X = 1000..64000: it should be exactly zero far earlier;
  (ii) redo the N6 proof for the window object: is a bound below 1 on ITS failing fraction
  also the conjecture? (Yes if "no open island in the window" is "no twin in the window", which
  it is.) Then the weakening must be in the quantifier over q: "for a positive fraction of q"
  is still twins infinite. So: no weaker object exists on this axis; record it and close.
- **Idea 2: use the sub-Poisson count.** The open-island count is sub-Poisson by a fixed
  factor with an exact mechanism (a generic gear costs -4/g^2, repaid while g < m). Realisations:
  (i) the repayment stops at g = m (the island count); test whether the count's variance law
  predicts the record law's increments (the depth of chains is where variance is paid);
  (ii) treat the variance deficit as the machine's "repulsion budget" and compare it with the
  suppression law's deficits (x26..x1400): if they are the same number in two coordinates, the
  suppression law has an exact formula.

## R3.i. The half-column map (dead; fibre, fixed point, uncoupled classification)

- **Object.** The gap spectrum read through v -> v/2.
- **Vectors.** Identities, fibre, fixed points, spectrum holes, records in columns 5 and 6.
- **Failure.** Coupling constrains strikes, endpoints are openings.
- **Idea 1: the depletion as a sum rule.** Uncoupled even sizes (twice the twin columns above
  the machine) are depleted 12-128x. If twins were finite, no even y-rough size below y^2/3
  would be depleted for y > Y. Realisations: (i) find an exact identity on the multiplicity
  function of gap sizes (the parity theorem says each size >= 2 occurs an even number of times;
  the total length and count are fixed; the mirror pairs sizes) and test whether the
  multiplicities of coupled sizes alone can satisfy it, i.e. whether a spectrum with NO
  depleted sizes in a range is consistent with the identities at m11..m23 (a finite check:
  perturb the measured spectrum by filling the depleted sizes to the coupled level and see
  which identity breaks); (ii) if an identity breaks, the machine's spectrum FORCES depleted
  sizes, and each depleted size is twice a twin column: a self-referential route with a
  finite computation at its base.
- **Idea 2: the record in column coordinates.** Records live in columns 5 and 6 at m29/m31
  (the top gears' home columns). Realisations: (i) predict the record's letters at rungs 37..59
  from the home columns of the top three gears and check against the corpus; (ii) the fixed-
  point theorem says the halving descent of a record closes on twin columns; test whether the
  closure set of the record at rung q always contains the twin columns of the gears that MAKE
  the record (the top three), so that the record is built from twin columns below it.

## R2.c and R2.c.i. The distortion method and the block moment (dead; collapse; trivial)

- **Object.** A covering-side bound on the interval the gears can cover.
- **Vectors.** BBMST engine translated; fibre and block budgets; the adversarial gate.
- **Failure.** Fibres collapse; blocks are the trivial criterion; the survivor lower bound is
  the root.
- **Idea 1: fibres of a SUB-machine.** Collapse happens when the product of gears used exceeds
  the interval. Use fibres modulo the product of the small gears only (which fits the window)
  and treat the remaining gears as a perturbation inside each fibre. Realisations: (i) run the
  engine with fibres mod Q_small = product of gears to 17 (fits W at q = 997) and the big gears
  entering with their exact in-fibre strike counts; the budget then has an exact head and a
  tail whose first moments are exact per fibre (each fibre is an arithmetic progression of
  step Q_small, on which a big gear g strikes at rate 2/g exactly over g fibres); test whether
  the second-moment term across fibres stays below the room at q = 59..997; (ii) if the cross-
  fibre second moment is the obstruction, name it exactly (it is the discrepancy of the big
  gears' strikes across residue classes of Q_small, a finite object per q).
- **Idea 2: the small-K theorem as the base of an induction.** The budget proves the adversarial
  lemma to K = 10. Realisations: (i) find the exact reason the budget fails at K = 11 (which
  term, which gear) and whether a re-partition (fibres mod the first 5 gears, blocks beyond)
  restores it at K = 11, 12 against the exact A(K); (ii) if a mixed partition works to K = 12,
  test its growth: does the mixed budget's threshold L* grow polynomially in q.

## Thin place 3, sharp reading (the record is not at the all-teeth columns)

- **Object.** The record class as a formula.
- **Vectors.** Distance to all-teeth columns (random).
- **Failure.** No anchoring at the alignment points.
- **Idea 1: the record's residues from the allocation law.** 5g's allocation law predicts 340
  of 348 gear placements. Realisations: (i) compute the CRT position of the class predicted by
  the law (coverage-maximal phase subject to sole columns, resolved at the 8 exceptions by the
  measured value) and compare with the true record position at m13..m31; (ii) if the law
  predicts the class up to the top two gears, the record's position is computable from the
  gears and its distance to the window is a number with a formula.
- **Idea 2: the record's position as a fraction of the period.** Measured 0.3-0.7 or at the
  ends. Realisations: (i) tabulate R / P for every record and near-record class to m31 and
  test whether the fraction is a simple rational in the gears' phases (the mirror gives
  R and P - R); (ii) test the same for the family: if random members' records sit anywhere,
  the real machine's 0.3-0.7 is a which-residues fact.

---

## Reading the file as a whole

1. **Prove the small cases** (R2.b Idea 1, R2.c Idea 2): the adversarial lemma to K = 10 is
   within reach as a written theorem; A(K)'s exact values to K = 5 or 6 by the dictionary
   mechanism. New mathematics, bounded, and the base a later induction would need.
2. **The spectrum's depletion as a sum rule** (R3.i Idea 1): the only self-referential route on
   the tree with a finite computation at its base; if the machine's spectrum forces depleted
   sizes, each is twice a twin column above the machine.
3. **Fibres of a sub-machine** (R2.c Idea 1): the one way to keep the distortion engine's
   content past the collapse, with an exactly computable obstruction.
4. **Words, not columns** (2g.i.a Idea 1, 2g.i.a.i Idea 2): the glue and the separability
   both asked column questions; the machine's grammar is in gap words.
