# Every dead branch reopened (the unstick protocol; manager, 2026-09-06)

The owner's rule: for every branch that died, list the object we were attacking, the attack
vectors, the reason for failure, two or more creative ideas to get through it, and for each idea
two or more ways to realise it. A dead branch is a brick; the constraint it established is
listed beside each idea so no idea repeats a measured failure. Vocabulary as in the tree's
profile. "Family" = the counterfactual family: same gears, symmetric two-tooth shape, teeth
moved. Numbers are the record's.

Cross-cutting constraints every idea must respect (the faces of the wall): (A) a count or
density alone cannot decide; (B) a fixed modulus sees position only; (C) the real machine is
typical by symmetry, spacing and squareness; (D) rarity among all phase vectors does not
transfer to real q; (E) a local statement (near zero, near q^2, per step) over-asks.

---

## 1a. Phase-shift descent: F(M) <= F_2(M minus its top gear)

- **Object.** The record of M as a one-hole stretch of the machine one gear down.
- **Vectors.** Sole-coverer counting at the record (top gear covers 1-3 columns); the descent
  inequality tested at m17, m23.
- **Failure.** The top gear makes 2-3 kills in the record, so the record of M is a 2-4-hole
  stretch of M minus top, and the descent is the spectrum-plus-depth bound (2e), which fails as
  a uniform tool. Untouched: the descent by the SMALLEST gear, or by any gear other than the top.
- **Idea 1: descend by a middle gear, not the top.** The deletion profile (5d.ii) says gear 5
  holds up the most and the top gear the least; the natural descent removes the gear whose
  removal shortens the record least, not the top. Realisations: (i) compute, for every record
  at m11..m31, the hole count of the record under M minus g for every g, and the g minimising
  it; test whether F(M) <= F_{h+1}(M minus g*) holds with h bounded (h = 1 or 2) for that g*;
  (ii) build the descent chain by always removing the least-needed gear and see whether the
  chain's hole counts stay bounded to m31 (if yes, the record is a bounded-hole stretch of a
  machine with the SAME top gear but fewer middle gears, a different object from 2e).
- **Idea 2: descend by removing a tooth, not a gear.** A gear has two teeth; removing one tooth
  leaves a one-class gear. The record under "M with one tooth of g removed" is an object between
  M and M minus g. Realisations: (i) measure F(M with tooth t of g removed) for every (g, t) at
  m11..m23 and see whether some single tooth removal already destroys the record (a "keystone
  tooth"); (ii) if keystone teeth exist, count them per record and test whether the record is
  always a 1-hole stretch of the machine with the keystone tooth removed (a descent by half a
  gear, which 2e never considered).

## 1b. Descent through the survivor generator (F_2 at M is layer 0 of the algebra one gear down)

- **Object.** The one-hole record as the first layer of the nested next-opening algebra.
- **Vectors.** Recursion through layers; hole costs by depth J (m29: 12, 10, 5, 15, 5).
- **Failure.** No base case, and the costs are not monotone in J. Untouched: the recursion run
  from the TOP down (start at the record, peel), and any base other than "small machine".
- **Idea 1: a base at the record itself.** The record set collapses to one class from m23 (5d);
  a recursion whose base is the unique record configuration, not a small machine, has a base.
  Realisations: (i) express the record of M + q' as the nested formula applied to the record
  CLASS of M (the pinned residue vector) and measure whether the recursion depth is bounded
  (one or two layers) when started there; (ii) test whether the runner-up classes (F - 3 and
  below, given the spectrum isolation by 3 at m29 and m31) are generated from the record class
  by a bounded number of single-gear re-phasings.
- **Idea 2: make the costs monotone by changing what is counted.** Hole cost per layer is not
  monotone; the coverage deficit per layer (5g's allocation law: coverage relative to the
  maximum, subject to sole columns) might be. Realisations: (i) recompute the layer table of
  9d in coverage-deficit units instead of hole counts and test monotonicity at m11..m31; (ii) if
  monotone, take the deficit as the potential of the recursion and bound the record by the sum
  of per-layer deficits (a telescoping argument, which needs monotonicity and nothing else).

## 1c. One-class transfer (one-hole(P_k) = j(P_{k+1}))

- **Object.** The two-class pair statement carried over from the one-class Jacobsthal ladder.
- **Vectors.** Literature sweep; the one-class identity through k = 18; A072753 (the two-class
  maximum over class assignments) violates the increment once (10 -> 24 at 13).
- **Failure.** Unasked in print in either class count; the two-class max over assignments
  violates, so the real teeth are needed. Untouched: the transfer in the OTHER direction (from
  the machine to one class), and the transfer at the level of mechanism rather than value.
- **Idea 1: the machine as two interleaved one-class sieves.** Column k is blocked iff 6k - 1 or
  6k + 1 is struck; each member separately is a one-class sieve on an arithmetic progression.
  The two-class record is the longest run where the two one-class sieves' openings never
  coincide. Realisations: (i) compute the two one-class opening sets separately on full periods
  and measure the longest run in which they avoid each other, against the product of their
  gap structures (is the two-class record the "coincidence gap" of two one-class patterns, and
  is there a one-class theorem about coincidence gaps of two coprime-shifted sieves); (ii)
  import the one-class transfer literally: Hagedorn's one-hole identity (k holes with r - k
  primes iff no holes with r primes) restated for each member's sieve, then ask what it says
  about coincidences.
- **Idea 2: transfer the mechanism, not the inequality.** The one-class increment holds because
  two kills by one new prime need a gap that is a multiple of p, impossible while j < 2p. The
  two-class analogue is the chain law (kills 0 or +-d apart). Realisations: (i) find the exact
  k at which the one-class mechanism stops (j(P_k) >= 2 p_{k+1}, k = 19 in Hagedorn) and the
  analogous rung in the machine (first rung where F >= 2 x letter), and compare what happens to
  the increment just after each (does the one-class increment jump when the mechanism fails, as
  the machine's did at 31 -> 37); (ii) build the one-class analogue of the padded letter and
  test whether the one-class ladder past k = 19 obeys the machine's bare-word cap.

## 1e. The column-0 obstruction (pair statement at column 0 = twin-Bertrand)

- **Object.** The pair statement at the one junction where every gear's phase is zero.
- **Vectors.** Mirror at column 0 (F_2 >= 2 d_0), d_0 measured to 33,317 (d_0 <= q' always,
  10-58x inside the window), L2-L6 proved.
- **Failure.** Any per-step bound implies d_0 <= (F + q')/2, a twin below a bound in q; every
  route is twin-Bertrand or a Rankin-type lower bound on F. Untouched: the pair statement with
  column 0 EXCLUDED (the junction theorem says column 0 is never a junction of M + q', only its
  CRT translate is), and the question whether the translate carries the same difficulty.
- **Idea 1: quotient out the column-0 orbit.** The junction theorem (R3.h.i) puts the flank pair
  (d_0, d_0) at 15,107 junctions of m23, all CRT translates of column 0. Excluding that one
  orbit, is the pair statement provable? Realisations: (i) recompute the pair statement's
  extremal junctions at m11..m31 with the column-0 orbit removed and see whether the maximiser
  changes (if the maximiser never IS the orbit, the obstruction is not where the statement is
  tight); (ii) prove the pair statement on the orbit separately: at a translate of column 0 the
  phase vector is known exactly (all gears at zero), so the flanks are d_0 on both sides and the
  statement reads 2 d_0 <= F + q'; test whether d_0 <= F/2 already (measured slack 8x at m53),
  i.e. whether the orbit is never the binding case, and prove a WEAK bound d_0 <= F/2 from the
  walk laws (the walk from zero passes the primes' own columns, each struck by its own gear, so
  d_0 is bounded by the first column past the last gear's column that no gear reaches: a
  computable object with no twin in it).
- **Idea 2: replace twin-Bertrand by an object the machine owns.** d_0 is the first column whose
  two numbers have no factor in the machine; the machine also owns the first column with no
  factor below some SMALLER bound. Realisations: (i) define d_0(M; B) = first opening past zero
  under the sub-machine {5..B} and measure how d_0(M) relates to d_0(M; sqrt q): if
  d_0(M) = d_0(M; sqrt q) at every level (the big gears never strike the first small-machine
  opening past zero), the twin-Bertrand quantity is a small-machine quantity, provable by the
  layer law; (ii) measure the largest B such that d_0(M) = d_0(M; B) and how it scales.

## 2a. Par trading as a theorem (each added letter paid by the flank envelope)

- **Object.** Delta_J = Q*_J - F_2(M) bounded uniformly, via the residual eps(v) per letter.
- **Vectors.** eps measured in [-21, +15] on the family against s_min 8; the cancellation
  lemma eps = d - g_out.
- **Failure.** eps is a cancellation of two large terms, not a smallness; on the family it is not
  bounded by s_min. Untouched: eps as a function of the letter's POSITION in the word, and the
  quantity that is conserved along a chain rather than per letter.
- **Idea 1: find the conserved quantity.** Delta_5 = 0 exactly at both machines where J = 5
  exists, and the J = 5 maximisers are self-reverse palindromes. Something is conserved at full
  depth. Realisations: (i) tabulate, along every maximising chain, the running sum of eps and
  the running coverage deficit (5g) and look for a linear combination that is exactly zero at
  the end of every chain; (ii) test whether the palindrome structure forces Delta = 0 (a
  palindromic word's flanks are mirror images, so by L6 the flank sum is determined by one
  side), which would turn par trading at full depth into a symmetry statement.
- **Idea 2: par trading per PAIR of letters.** The middle-sum lemma says middles alternate
  classes a, b, so letters come in pairs summing to q'. Realisations: (i) measure eps over
  consecutive letter PAIRS (eps(v.x.y) - eps(v)) and test whether the pair residual is bounded
  by s_min where the single-letter residual is not; (ii) if pairs are bounded, the odd letter at
  the end is the whole difficulty: measure it as its own object (the last-letter residual) and
  relate it to the bare-word cap.

## 2c. Padded case by the record law one level down

- **Object.** A padded middle (an old gap that is a multiple of q') as itself a merge one level
  down.
- **Vectors.** The base q' > F(M minus top) needed for the descent; fails from m29.
- **Failure.** No base. Untouched: the padded gap as a merge TWO levels down, and the padded
  middle's own flanks (the F_3 wall at m31 has the padded letter 37 as the middle of the F_3
  maximiser, Phi(37) + 37 = F_3 exactly, resting on one occurrence).
- **Idea 1: the one-occurrence structure.** Phi(37) at m31 rests on one mirror pair. A padded
  failure that rests on one configuration is a configuration, not a law; classify it.
  Realisations: (i) for every padded letter at every rung to m41, count the occurrences of the
  padded gap with maximal flanks and list their residue vectors; if each is a single CRT class,
  test whether the class is a translate of the record class (if so, padded events are the
  record seen from one gear down and the record law applies to them); (ii) test whether removing
  the single occurrence's class (a measure-zero set) makes par trading hold at all padded cells
  (the rider says yes at m31): then the padded case is "one configuration per rung", and a proof
  needs only to show that configuration is never in the window.
- **Idea 2: pad in the other direction.** A padded middle is a gap of exactly q' (or a
  multiple); by the chain law its two ends are on the SAME tooth. Realisations: (i) measure the
  flanks of same-tooth junction pairs at distance exactly q' as their own profile N_same(q')
  against F, at every rung, and compare with the opposite-tooth flanks N(d), N(q' - d): if the
  same-tooth profile is systematically lower, the padded case is easier than the literal one and
  the m31 event is an outlier to be classified; (ii) check the same-tooth junction pairs against
  L6: both ends at the same tooth means the same residue mod q', so the left flank of one end
  and the right flank of the other are negated tilings of each other only if g divides x, which
  is a residue condition that can be counted exactly.

## 2d. Survivor-algebra contraction across layers

- **Object.** The depth family as one max-plus algebra whose layers contract.
- **Vectors.** Layer monotonicity tested; layers non-monotone.
- **Failure.** Non-monotone in the natural order. Untouched: a different order of the gears
  (by coverage deficit, by sole-column count, by arc), and a norm other than length.
- **Idea 1: reorder the layers.** The nested formula is exact in any order of the gears (the
  closure does not care). Realisations: (i) run the layered walk with gears ordered by
  decreasing coverage deficit at the record (5g's allocation law) and test whether the survivor
  counts become monotone; (ii) order by arc length (the long arc g - a_g, which decides the
  band structure of R3.h.i) and test the same.
- **Idea 2: contract a different quantity.** Length is not monotone, but the number of holes
  weighted by the umbrella bound (a stretch of span S contains a tooth of every gear with long
  arc below S + 2, proved) might be. Realisations: (i) define the layer potential as the number
  of gears NOT yet forced to strike (long arc above the current span) and show it decreases by
  at least one per layer that hops; (ii) if it does, the number of hopping layers is bounded by
  the potential at the start, which is a function of the span alone: a length bound from a
  counting of gears, not of columns (face A forbids counting columns; it says nothing about
  counting gears whose arcs are forced).

## 2e. Spectrum-plus-depth (F_J only, no legality)

- **Object.** F(M + q') <= F_{A_kill + 1}(M).
- **Vectors.** The bound tested at every rung; fails at 29 -> 31 and 47 -> 53 where A_kill >= 4.
- **Failure.** Without legality the spectrum bound is loose exactly when chains are deep.
  Untouched: the spectrum of legal-middle runs only (the legal spectrum), which is what the
  record uses.
- **Idea 1: the legal spectrum as the object.** Define F_J^leg(M) = the longest J-run all of
  whose middles are letters; measure it. Realisations: (i) compute F_J^leg at m11..m31 for
  J = 3, 4, 5 and test F(M + q') <= F_{A_kill+1}^leg(M) at every rung, including the two
  failing ones; (ii) measure the ratio F_J^leg / F_J: if it is bounded away from 1 and
  decreasing in J, legality is exactly the slack 2e lacked, quantified.
- **Idea 2: bound A_kill by the arcs.** A_kill = L + 1 and L <= 2F(M + q')/q' + 1 (proved) is
  circular in F. Realisations: (i) test whether L is bounded by the number of gears whose long
  arc exceeds F(M) (the gears that CAN be missed by a record-length stretch), a quantity with
  no F(M + q') in it; (ii) test whether the deepest chain at each rung uses only middle-band
  gears (R3.h.i's band II), so that its depth is bounded by the band's size.

## 2f. The adjacent-teeth sub-family (chain statement from invariant ingredients + two kernel facts)

- **Object.** The chain statement on every family member without adjacent teeth and with
  3a = q' -+ 1.
- **Vectors.** 2,568 exhaustive rows to m19, a 600-row sample at m23, then the 23 -> 29 sweep.
- **Failure.** A member with gears 5 and 7 real, no adjacent teeth, incoming tooth pinned, gives
  F(M + 29) >= 62 > 61. Untouched: WHAT the refuting member has that the real machine lacks
  (W3 found the real machine's separations are compatible under CRT: r(S_g + S_h) = c mod gh).
- **Idea 1: classify violators by separation compatibility.** Realisations: (i) for every chain
  violator on the family to m29, compute the number of incompatible gear pairs (no rational r
  with r(S_g + S_h) constant mod gh) and test whether every violator has at least one such
  pair while the real machine has none; (ii) define the sub-family "all pairs compatible under
  one rational r" (this includes the real machine at r = 3 and the coherent families of W3)
  and sweep it to m29 for chain violators; if zero, the chain statement's ingredient is
  pairwise compatibility, a residue identity, not a symmetry (face C allows arithmetic).
- **Idea 2: the violator as a machine with a different anchor.** The refuting member has real
  5 and 7 but moved 11..23. Realisations: (i) find the smallest set of gears that must be real
  for the chain statement to hold on the sub-family with those gears real and the rest free
  (the "real core"): sweep with real {5,7,11}, {5,7,11,13}, ... and record the first core with
  zero violators to m29; (ii) if the core is the whole machine, the chain statement is not
  decomposable and the tree should say so as a theorem-shaped fact; if the core is small, the
  statement is about the core's residues and can be attacked mod the core's product.

## 3a. Explicit-constant Iwaniec-type bound for the two-class sieve

- **Object.** F(y) <= C_2 y^2 with C_2 < 1/6 by a sieve bound.
- **Vectors.** Iwaniec's shifted sieve transferred to two classes; finite certificates.
- **Failure.** The engine becomes a dimension-2 sieve, exponent 4.27, not 2; a class-count bound
  with C_2 < 1/6 is the conjecture. Untouched: sieve weights that see the separation (which
  residues), and the covering-system literature's non-sieve methods.
- **Idea 1: a sieve with separation-aware weights.** The Selberg sieve optimises weights over
  class counts; the machine's two classes are at a fixed offset d_g, so the sifting function has
  a known second moment across gears (W3: the four struck residues of two gears are a translate
  of {0, S_g, S_h, S_g + S_h}). Realisations: (i) write the two-class Selberg sieve with the
  pairwise correlation term computed exactly from the fixed separations instead of bounded
  generically, and see whether the dimension drops below 2 in the main term (the parity barrier
  is a statement about the generic sieve; a sieve that knows the offsets is a different sieve);
  (ii) test numerically: the Selberg upper bound for openings in a window with exact pairwise
  terms against the generic one at q = 59..997, and whether the gain is a constant or grows.
- **Idea 2: the distortion method of covering systems.** Balister-Bollobas-Morris-Sahasrabudhe-
  Tiba prove that covering systems need a small minimum modulus by a distortion/weighting
  argument, not a sieve. Realisations: (i) a literature lane: state the machine's problem (primes
  up to q, two classes each at fixed separation, one phase per gear, cover an interval of length
  W) in that paper's terms and determine what their bound gives for the longest coverable
  interval (they bound moduli; we need length); (ii) if the method bounds length at all, compute
  its constant on the machine and compare with 1/6 and with the measured 1/24.

## 7b. The anchor pattern inside the window

- **Object.** The openings of a fixed anchor machine inside a window, and what each higher gear
  removes.
- **Vectors.** Measured at every prime to 5000; the anchor's in-window rigidity proved for
  {5..13}; each later gear's take follows one curve in ln g / ln Q'.
- **Failure.** The rigidity is exhausted at the first gear above the anchor; from the second gear
  on it is a known one-prime identity. Untouched: a MOVING anchor (the anchor as the machine
  below sqrt of the column), and rigidity as a statement about differences rather than counts.
- **Idea 1: the moving anchor.** At column k the effective machine is {5..sqrt(6k+1)} (7d), so
  the natural anchor at k is that machine, whose period is far above k. Realisations: (i) prove
  the in-window rigidity for the effective machine at every k (the interval discrepancy of a
  periodic set over an interval shorter than its period is bounded by the number of runs in the
  pattern, a finite computation per machine); (ii) use the rigidity to bound the gap between
  consecutive effective-machine openings inside the window by the effective machine's record
  F(sqrt(6k)), which is the self-similar statement "the window's runs at column k are runs of a
  machine of size sqrt(6k)": then the root becomes F(y) < y^2/6 for the effective machine at
  the window's top, which is the root again but now stated with the machine SMALLER than the
  window by a square root, a different quantifier.
- **Idea 2: rigidity of differences.** Counts are face A; the difference pattern (gaps) of the
  anchor's openings inside the window is exact and periodic. Realisations: (i) measure the gap
  word of the anchor's openings across a window and the gap word of the survivors after each
  gear, as words, and test whether each gear's action on the word is a fixed substitution (a
  morphism on words), which would make the window's gap word a substitution sequence; (ii) if it
  is, the longest run in a substitution sequence is bounded by the substitution's growth, a
  known kind of statement (not a sieve, not a count).

## 5b. Adjacency repulsion (found to be the suppression law)

- **Object.** Gaps next to a large gap are shorter than independence gives.
- **Vectors.** F_2 against shuffled gaps; the negated-tiling mechanism (L6 proved, size
  consequence not); rediscovered as the round-19 suppression law with the renewal ladder.
- **Failure.** The rigorous side (renewal ladder) stops at the rate-to-maximum step; at column 0
  the correlation is +1. Untouched: the repulsion as an EXACT profile (the manager's N(v) <= F + 1
  for v >= v_0, running now), and repulsion at distance two (the gap after the gap after).
- **Idea 1: the exact profile.** Running (research/proof/neighbour_profile.md). Realisations: as
  briefed there: extend to m31, test on the family, test the concatenation of the two flanks.
- **Idea 2: repulsion as a conservation law.** The flanks are coupled by the anchor's residue
  classes (R3.h.i: L^+ = 1 mod 5 forces L^- in {0, 2, 4}), which is a residue fact and cannot
  bound sizes; but the SUM of a gap and its two neighbours over a full run of the corridor might
  be conserved. Realisations: (i) measure the sum of every window of three consecutive gaps
  against its position mod 35 and test whether the maximum over each residue class is attained
  by the same configuration (a conserved "triple mass" per class); (ii) test whether the
  suppression is a consequence of the gear-5 lock alone: compute the profile N(v) for the
  machine {5, 7} exactly and see whether its shape (spikes at tiny v, flat at F + 1 above) is
  already there; if so the law is a two-gear theorem lifted by CRT.

## 5d.i. The record as a frame of three gears

- **Object.** Record = frame (5, 7, top) plus a filling by the middle gears.
- **Vectors.** Frame counting, completion counts, the window's frame columns.
- **Failure.** The record set collapses to one class from m23, so there is no frame/filling
  split; the window holds q/210 frame columns. Untouched: the record as a frame of ALL gears
  (the pinned class) and what the pinned class's residues are as a function of the gears.
- **Idea 1: the pinned class as a formula.** From m23 the record is one residue class mod P up
  to mirror; its residue at each gear is a number. Realisations: (i) tabulate the record's
  residue at each gear g for m23, m29, m31 and test candidate formulas (the coverage-maximal
  phase subject to sole columns, which 5g says holds at 340 of 348 cells; the phase that puts
  the stretch's start right after g's tooth; the phase that centres the stretch between g's
  teeth) to see which formula predicts the residue at every gear; (ii) if a formula predicts
  all but the top two gears, the record is computable from the gears alone without a period
  scan, and its position in the period is a computable number whose distance from the window
  is a computable number: the "never in the window" fact becomes a computation, not a
  measurement.
- **Idea 2: the record class as a point on the torus, and the window as a box.** The window is
  the set of columns below W; a residue class is a point on the torus; the class is in the
  window iff its CRT representative is below W. Realisations: (i) compute the CRT representative
  R of the record class at m11..m31 and its ratio R/P; test whether R/P is bounded away from
  zero by a rule (mirror pairs give R and P - R; the smaller is at most P/2; measured fractions
  0.3-0.7 or at the period's ends); (ii) test the same for every class attaining F - 1, F - 2
  (the near-record classes) to see whether near-record classes ever come close to the window.

## 5d.ii. What each gear holds up, in the period and in the window

- **Object.** The deletion profile F(M) - F(M minus g) and the window's version.
- **Vectors.** Set cover of the record; the window's holders; the square gate; the nested-
  decreasing holder law.
- **Failure.** Every window-side quantity is contingent on the primes. Untouched: the holder set
  as a function of the stretch's LENGTH only (the holder law is monotone in the machine), and
  the minimum number of holders a stretch of length S can have.
- **Idea 1: the holder count as a function of length.** By the umbrella bound every gear with
  long arc below S + 2 strikes a stretch of span S; those are forced holders. Realisations: (i)
  compute, for every blocked stretch in every window to q = 997, the number of gears that strike
  it (its holder count) against S, and the minimum over stretches of a given S; test whether the
  minimum holder count grows with S by a rule (forced gears plus at least one middle-band gear
  per some length); (ii) invert: the largest S whose minimum holder count is at most K, as a
  function of K, is a length bound in terms of a gear count, which face A allows (it counts
  gears, not columns).
- **Idea 2: the jointly-essential zero-drop set.** Removing all zero-drop gears at once destroys
  the window stretch at 157 of 165 rungs. Realisations: (i) find the smallest subset of the
  zero-drop set whose joint removal destroys the stretch (a minimal joint-essential set) and
  its size distribution; (ii) test whether that set is always a set of gears in one arc band
  (R3.h.i's middle band), which would say the window's stretch is held by the middle band
  jointly and by no gear singly, a structural fact with a gear count in it.

## 5g. The hinge as a length lever (dead) and the gear-5 lock (position)

- **Object.** A window stretch's length bounded by the teeth of the one gear that alone strikes
  some column of it.
- **Vectors.** Hinge existence (always), hinge gear size, hinge position, three length rules.
- **Failure.** By the nested-decreasing holder law the hinge gear falls as q grows at fixed
  length, so no bound in the hinge gear can hold. Untouched: the hinge at the FIRST machine
  where the stretch appears (the birth machine), where the holder set is largest.
- **Idea 1: the birth hinge.** A window stretch is born at one rung (the first machine that
  blocks it fully); at birth its holder set is largest and its hinge gear is largest.
  Realisations: (i) for every window stretch to q = 1999, find its birth rung and its hinge gear
  at birth; test whether the birth hinge gear exceeds a fixed fraction of the birth rung (the
  measured 57% above q/2 was over all rungs, diluted by later rungs); (ii) test the length rule
  L <= f(hinge gear) at birth only.
- **Idea 2: the lock as a coordinate.** The gear-5 lock pins gear 5's phase against any maximal
  stretch. Realisations: (i) use the lock to reduce the machine: every maximal stretch lives in
  one of the five gear-5 phases (the lock says which for a given length mod 5), so the record
  problem on M is the record problem on M minus 5 restricted to the columns of one gear-5
  class, i.e. a machine with a coarser column coordinate (5 columns become one) and the
  remaining gears' teeth re-expressed; compute that reduced machine's teeth and its record and
  check the identity F(M) = 5 F_red + c exactly at m11..m23 (7a's floor((F - 2)/5) says the
  cycle version holds; this is the gear-5-phase version); (ii) iterate the lock: does gear 7
  lock against maximal stretches of the reduced machine? If locks iterate, the record is the
  record of a machine with all small gears folded in, i.e. the anchor grows with the machine.

## 6. Coherent spacings

- **Object.** The real machine's one-third tooth spacing as the cause of its small record.
- **Vectors.** Coherent spacing vectors (c/2d mod g, d <= 30) against random symmetric vectors
  at m13, m17.
- **Failure.** Same F distribution; the real machine at the 14th-22nd percentile. Untouched:
  coherence as a driver of anything other than F (W3 tested K: also nothing), and coherence of
  TRIPLES (three gears' compatibility), which pairwise compatibility does not decide.
- **Idea 1: triple compatibility.** Realisations: (i) for triples of gears, compute the joint
  struck set mod ghk for the real separations and for random ones and count the columns struck
  by all three (triple overlaps); test whether the real machine's triple overlaps are extreme
  in the random distribution where pair overlaps are not (W3: pair mean overlap is 4m/gh for
  every separation, so pairs cannot distinguish; triples can); (ii) if triples distinguish, test
  whether F on the family correlates with the triple-overlap statistic rather than with
  coherence per se.
- **Idea 2: coherence at the window, not in the period.** F is a period quantity; the window is
  where every gear's phase is q^2 mod g. Realisations: (i) on the family, compute the window's
  longest stretch F_W for coherent versus random members with the SAME phase vector (the real
  q^2 residues) and see whether coherence matters there; (ii) measure whether the tail gears'
  extreme tooth distance (0.69 of the arc, W3) shortens F_W specifically, since tail gears are
  the ones that act in the window sparsely.

## 7a. Cycles as the unit

- **Object.** The anchor-30 cycle's dead-cycle record and its increment.
- **Vectors.** F_c computed to m31; the identity F_c = floor((F - 2)/5); class dependence.
- **Failure.** The cycle frame is the column frame divided by five. Untouched: cycles of a
  LARGER anchor for a fixed machine (anchor 210 for machines above 48, where 7a's own doc says
  untouched-cycle structure returns), and cycles as the unit for the SPECTRUM, not the record.
- **Idea 1: the anchor that grows with the machine.** Anchor 210 has structure only for q > 48;
  anchor 2310 for q > 480. Realisations: (i) at machines m53..m97 (records by SAT lower bounds
  and the corpus), compute the dead-cycle record with anchor 210 and test the analogue identity
  F_c210 = floor((F - c)/48); if the identity holds with the same shape, the cycle frame is
  always the column frame rescaled and the idea is closed for good; if it does NOT, the larger
  anchor sees something; (ii) compute the cycle chain law for anchor 210 (a gear's hits on
  cycles at fixed fractions of its run) and its depth per layer.
- **Idea 2: cycles for the spectrum.** Realisations: (i) the spectrum of dead-cycle runs (all
  lengths, with multiplicities) against the column spectrum rescaled; the holes in the column
  spectrum (24 at m19 and m23; 41, 42 at m29; 56, 57 at m31) should appear as holes in the cycle
  spectrum if they are structural; (ii) test whether spectrum holes are cycle-aligned (a missing
  length v with v = 4 mod 5 would be a cycle-boundary effect: 9, 19, 24 are all 4 mod 5).

## 7d. Runs as the unit and the zero mirror

- **Object.** The region just past zero, where every gear's tooth has just landed, as a source
  of openings.
- **Vectors.** Opening counts past zero against the period mean; exclusive kills by gear; the
  effective machine at column k.
- **Failure.** Thinner than the mean (0.79 asymptotically) because the effective machine at k is
  {5..sqrt(6k+1)}; any statement about (0, W] is a statement about the twins below Q'^2.
  Untouched: the region just past the ANTIPODE (P +- 1)/2, the other always-open column, where
  every gear's phase is the mirror of zero's, and the region past the CRT translates of zero.
- **Idea 1: the antipode.** Realisations: (i) measure the opening density and the longest run in
  (antipode, antipode + W] at m11..m23 against the period mean and against (0, W]; the
  antipode has every gear at phase (g +- 1)/2, so its neighbourhood is the region where every
  gear's tooth is FARTHEST, the opposite of zero; (ii) if the antipode's neighbourhood is rich
  (above the mean), the machine has a computable rich region, and the question becomes whether
  a rich region ever coincides with a window (it cannot, the antipode is at P/2), but the
  mechanism of richness is a new brick.
- **Idea 2: the walk from a translate of zero.** The junction theorem puts (d_0, d_0) at 15,107
  m23 junctions, the CRT translates of zero. Realisations: (i) measure the walk from each
  translate of zero forward beyond d_0 (the second, third openings) and test whether the walk
  from a translate is the walk from zero exactly (it should be, by translation of every gear's
  phase), which gives d_0 a period-wide presence: every translate carries the twin-Bertrand
  flank, so the pair statement's binding case recurs 2^m times per period; (ii) count the
  translates of zero inside the window: if none can be (the translates are the solutions of
  6k = +-1 mod every gear, and the window contains none but zero itself for q > some bound),
  the window is free of the binding case by a computable margin.

## R2.a. The machine feeds on itself (FACT, not a route)

- **Object.** The recursion gears -> openings -> gears, and the walk from q^2.
- **Vectors.** W1-W4; the chain of landings; the birth neighbourhood.
- **Failure.** The walk is decided by the old gears; L < d is twin-Bertrand at scale q/3; the
  chain of landings has no rule. Untouched: the recursion run BACKWARD (from a twin gear pair to
  the window it was born in, and to the machine whose record it was inside), and the transfer
  rule W3's admissible set {7, 17, 31} as a structure.
- **Idea 1: the backward recursion.** Realisations: (i) for every twin gear pair (g, g + 2) of a
  machine, find the machine {5..y} in whose window it was an opening and the blocked stretch it
  terminated there (its birth stretch) and its length; test whether the birth stretch lengths of
  the twin gears of {5..q} are bounded by a function of g (they are gaps between consecutive
  twins below g, so this is the twin-gap sequence read as the machine's own gears' birth
  certificates; the question is whether the machine's later behaviour, e.g. its record, depends
  on its gears' birth stretches); (ii) test whether the record of {5..q} is made (R3.h) by gears
  whose birth stretches are long or short.
- **Idea 2: the admissible carry-over set {7, 17, 31}.** The only gears that can strike both
  beside a birth column and inside the pair's own walk at the nearest offset. Realisations: (i)
  derive the set from the transfer rule (it is the set of primes dividing one of four fixed
  small polynomial values) and list the admissible sets for every offset j, i; (ii) test whether
  the admissible sets, being fixed and small, make the walk from g^2 of a twin gear g depend on
  the birth neighbourhood through a bounded number of gears, i.e. whether the walk's first few
  columns are predictable from the birth pattern (the walk's start is the one place where a
  cross-level rule exists; the chain of landings had none, but the chain of STARTS might).

## R2.a.i.a.1.a. The cover number (the transfer obstruction)

- **Object.** The island witness for real q, from its rarity among all phase vectors.
- **Vectors.** K(d) certified; the 2^K classes per cover; the product above q^2; the first moment.
- **Failure.** 2.7^m covers against a 2^K class density, vacuous by 10^24; equidistribution
  beyond any known theorem. Untouched: covers as a structured set (they are not arbitrary
  subsets; a cover is a set of gears with phases whose union contains a fixed set), and the
  covers realised by REAL q as a set (each failing q gives one cover; the map q -> cover).
- **Idea 1: count covers up to the machine's symmetries.** Two covers that differ by a
  re-phasing of a gear that keeps its island set are the same cover for the class count.
  Realisations: (i) count covers modulo the equivalence "same union on the islands" (the union
  is what matters; the 2.7^m count is over gear-island assignments, many giving the same union);
  (ii) count the unions that are actually realisable by a real q (the map from failing q to its
  cover is injective on classes mod the product; count the image at the 17 real failures and
  see whether the realised covers share a structure, e.g. all use the same small gears).
- **Idea 2: replace equidistribution of q by equidistribution of the ISLANDS.** The statement
  "some island in [1, d) is open at q" is symmetric: it is also "q^2 is not a root of the
  cover's polynomial system". Realisations: (i) for fixed q, the islands i in [1, d) with
  q^2 + 6i - 2 and q^2 + 6i both composite form a set S_q; the witness says S_q is not all of
  the islands; the complement is the set of twin pairs in a short interval above q^2 in a fixed
  residue class; count the twin pairs in [q^2, q^2 + 6d) in the class 12 mod 35 by the exact
  machine rate (the doubling law) and its variance over q (the first moment is 16.5 vs 17 with
  the s = 2 correction), then ask for the SECOND moment over q: if the variance is of the order
  of the mean, a Chebyshev bound gives "most q have an open island", which is weaker than "all"
  but is a theorem-shaped statement that no branch has yet written down; (ii) the third-moment
  or Janson version for the number of open islands, to bound the fraction of failing q by an
  explicit function that goes to zero (from "rare" to "vanishing fraction", a real step even if
  not "never").

## R2.a.i.a.1.b. Squares are even

- **Object.** The square phase vector as the reason covers are never realised.
- **Vectors.** Real, locally-square and random vectors on the island witness; index parity; the
  global-square test.
- **Failure.** All fail at the same rate; the square condition is implied by the range
  condition. Untouched: the SIGN structure (q^2 = (+-q)^2, so the vector is the same for q and
  -q, i.e. for q and P - q; the walk from q^2 and from (P - q)^2 are the same), and the vectors
  of q^2 for q in a fixed residue class mod a small modulus.
- **Idea 1: the vector as a function of q mod small moduli.** Realisations: (i) group q by
  q mod 210 and measure the island-witness slack (open islands in the arc) per class; if some
  classes are systematically richer, the witness could be proved for those classes first by a
  residue argument on the small gears (the small gears' strikes are exact functions of q mod
  210), leaving the rest to the large gears; (ii) the same grouping for the walk length L.
- **Idea 2: squares of consecutive primes.** q and the next prime q' have q'^2 - q^2 = (q' - q)
  (q' + q), a small multiple of a number near 2q. Realisations: (i) measure the correlation of
  the island-witness slack between consecutive primes; if strong, the witness for q' follows
  from the witness for q plus a local statement about the offset (q'^2 - q^2)/6, which is a
  statement about one section, the section view the owner asked for from the start; (ii) test
  whether the open island of q, shifted by (q'^2 - q^2)/6, is ever the open island of q' (a
  carried witness), and with what frequency.

## W1 / W3. The real separation and the island cover (dead for lack of slack)

- **Object.** K(d) on islands driven by the fixed one-third separation.
- **Vectors.** K for real, random, coherent and free separations; the pairwise overlap formula.
- **Failure.** K_real is the mode of the random distribution; the island target has zero to one
  gear of slack. Untouched: the WHOLE-COLUMN cover with fixed separation (which is the root),
  and the separation's effect on the tail gears, which is extreme (0.69 of the arc) but too
  small to move K on islands.
- **Idea 1: the tail-gear effect on whole columns.** Realisations: (i) measure the contribution
  of gears in (q/2, q] to the real record stretch (they make 3 + 2 + 2 kills at m29, R3.h) and
  compare with the family: does the real separation force the tail gears to waste more strikes
  (land on already-blocked columns) than random separations do, at record stretches, and is
  the waste a fixed fraction (the record law's mechanism at the top three gears); (ii) if the
  tail waste is systematically larger for the real teeth, the family's records should be longer
  than the real one on average, which is measured (14th-22nd percentile): quantify whether the
  whole percentile gap is the tail effect.
- **Idea 2: a different island set.** Islands for bound 7 have no slack; islands for a bound
  that GROWS with q might. Realisations: (i) define islands relative to bound B = sqrt(q) (the
  offsets no gear up to sqrt(q) can reach; a q-dependent but computable set of density about
  1/ln^2) and measure the cover number of THOSE islands in the arc against the number of
  gears above sqrt(q): the target K_B(d) > pi(q) - pi(sqrt q) has the same form but the
  strikers are now only the large gears, each striking the arc at most twice (the layer
  structure), so the count of available strikes is 2 pi(q) against islands of number
  (4d/35) prod (1 - 2/g) ... compute it; (ii) the same with B = q^alpha for several alpha to
  find the alpha at which the slack is largest.

## R3.h.i. The flank brick (closed by the junction theorem)

- **Object.** Structure at a junction that bounds the two flanks.
- **Vectors.** Bucket vectors both sides; L6 exact; gear bands; anchor coupling; the window's two
  junctions; the inverse-shape bucket bound.
- **Failure.** Junctions are ordinary openings, so the flank brick is F_2(M) itself. Untouched:
  the two flanks as a PAIR of walks with the negation lemma applied to the pair of END
  columns rather than the junction, and the middle band's constant 0.796.
- **Idea 1: the ends, not the junction.** A 3-run (L^-, v, L^+) has four openings: the outer
  ends and the two ends of the middle gap. L6 applies at each. Realisations: (i) apply L6 at the
  two OUTER openings: the tiling beyond each outer end is the negated tiling inside; for the
  3-run to be maximal, the columns just beyond both outer ends are open, which by L6 constrains
  the residues inside at the two ends jointly with the middle gap's length; write the joint
  residue system for (L^-, v, L^+) and count its solutions exactly per (L^-, v, L^+) at m11..m19
  (the level-3 gap dictionary by CRT, the way the LP lane did level 2); (ii) test whether the
  dictionary's admissible triples are bounded in L^- + L^+ for v >= v_0 by the residue system
  alone (a CRT count, which the escape-distance ceiling says cannot bound lengths; but the
  count can be ZERO for specific triples, and a zero is a bound).
- **Idea 2: the constant 0.796.** Realisations: (i) compute the arc-average prediction for the
  middle band's strike rate (a gear with short arc a <= S - 2 and long arc b >= S + 2 strikes a
  stretch of span S at a fraction (S + 1 + a)/g of its phases, averaged over the band's gears)
  and compare with 0.796 at each q; if it matches, the constant is a rate and the band is
  understood; if it does not, the deficit is a mechanism; (ii) split the band by which tooth
  strikes (same or opposite to the junction's tooth) and see whether the two halves have
  different constants.

## The dead-ends list (methods, not branches)

- **Bounded-modulus residue arithmetic (escape distance 1).** Object: positions mod a fixed
  modulus as a bound. Failure: the record escapes by one column. Idea 1: a modulus that grows
  with the length: certify "no stretch of length S" mod the product of the gears with long arc
  below S + 2 (the forced strikers), which is a modulus that grows with S; realise by (i)
  computing the forced-striker product for S = F + 1 at each machine and checking whether the
  residue system mod that product already has no solution (an exact CRT check per machine),
  (ii) if it does have solutions, counting how many gears beyond the forced ones are needed to
  kill them (the "escape count", which may be bounded). Idea 2: escape distance as the object:
  measure the escape distance for every modulus (how far past the modulus's bound the record
  actually goes) and its law; realise by (i) tabulating escape distance against modulus size at
  m11..m23, (ii) testing whether the escape is always exactly one column (a rigid fact that
  would itself be a lemma: the record exceeds every bounded-modulus bound by exactly one).
- **Fixed-depth counting, capacity, overlap.** Object: strikes against columns. Failure: the
  record nearly achieves counting. Idea 1: count gears instead of strikes (the umbrella bound
  and the holder count, see 2d and 5d.ii above); realise as there. Idea 2: overlap on the
  ADVERSARY rather than the record (K on whole columns, W2), realise by (i) certifying
  K_columns(d) by ILP at d = 35..1330 as was done for islands, (ii) comparing it with the F
  ladder read backwards to confirm the identity K_columns(d) = number of gears of the smallest
  machine with F >= d, which would make the F ladder an ILP object with certificates.
- **Pairwise convexity / SDP (stops at m19).** Object: a convex relaxation of the covering
  problem. Failure: the relaxation is loose past m19. Idea 1: add the proven structural cuts
  (gear-5 lock, allocation law, chain law, bare-word cap) as constraints and re-solve; realise by
  (i) the LP lane's certificate machinery with the new cuts at m23, m29, (ii) measuring which cut
  tightens most. Idea 2: relax the WORD, not the columns: an LP over gap words with the merge
  grammar as constraints (transfer matrices were refuted for F, but an LP over words with the
  chain law is a different object); realise by (i) writing the word LP at m11..m19 and comparing
  its bound with F, (ii) adding the neighbour-profile law N(v) <= F + 1 as a cut if it holds.
- **Symmetry beyond the mirror; letter size; congruence potentials.** Closed cleanly (the group
  is Z/2; letter size does not drive L; potentials certify nothing). The one reopening: the
  mirror composed with the CRT translates of zero (7d Idea 2): the symmetry group of the
  opening set is Z/2, but the STABILISER of the record class is trivial and the record class has
  2^m "all-teeth" columns around it; realise by counting the all-teeth columns (solutions of
  6k = +-1 mod each gear) inside the record stretch at m11..m31.

---

## Reading the file as a whole

Ideas that recur across branches, and so mark the wall's thin places:

1. **Count gears, not columns** (2d, 5d.ii, the umbrella bound, the dead-ends list): the forced
   strikers of a stretch of span S are exactly the gears with long arc below S + 2, a proven
   set that grows with S; a length bound in terms of a gear count is not forbidden by face A.
2. **The exact level-3 dictionary** (R3.h.i Idea 1, 2c Idea 2, 5b): the neighbour profile and
   the joint residue system for (L^-, v, L^+), computable by CRT the way the LP lane did level
   2; zeros in the dictionary are bounds.
3. **The record class as a formula** (5d.i, 1b, 7d): the record is one class from m23; if its
   residues are predictable from the gears, its distance from the window is computable.
4. **Compatibility of separations under CRT** (2f, 6, W3): the one arithmetic property that is
   the real machine's own; test it as the ingredient of the chain statement.
5. **Moments over q for the island witness** (R2.a.i.a.1.a Idea 2): from "rare" to "a vanishing
   fraction of q fail", a theorem-shaped step nobody has written.
