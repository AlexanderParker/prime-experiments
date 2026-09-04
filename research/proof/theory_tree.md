# Theory tree toward the proof (started 2026-09-04; nested 2026-09-05; method: .claude/skills/theory-tree)

## Project profile (read by the theory-tree skill; everything project-specific lives here)

- **Tree file:** this file. Branch documents in research/proof/<branch>.md; scripts in
  research/<line>/r<round>/ with results in .../results/ (large generated data untracked).
- **Root question:** for every machine {5..y} an opening lands inside the window, i.e. the longest
  opening-free stretch stays below the window's growth, F(y) < W(y) - y/6. Accepted as true; the
  work is the proof. The answer must be, in the human's words (2026-09-05), a known object we can
  point at and say "this will always be in the window, because the machine works this way, and
  nothing the machine does can prevent it." Candidates are marked CANDIDATE OBJECT below.
- **Vocabulary** (docs/proof-search/alignment-rules.md section 0; README glossary): column k =
  (6k-1, 6k+1); gear g strikes k iff k = +-6^-1 (mod g); opening = column no gear strikes; machine
  {5..y}; anchor = 2, 3, 5 as one object (cycle 30); window = the certified range (y, y^2], never a
  sliding run; section = the window's new part (p^2, q^2); stretch = a sliding run; record F(M) =
  longest opening-free stretch; the budget inequality F(M+q') <= F(M) + q' is a target, never a
  law. Think in openings, not kills.
- **Evidence standards:** kernel (Lean, `cd proofs; lake env lean AxiomCheck.lean` with no
  sorryAx), exact (full periods, phase reduction, SAT or LP certificates), measured, open. Pattern
  checks on the section; mechanism at the extremes, never averages alone.
- **Compute:** at most 4 cores and 3 GB per lane, 16 GB total; the 385-import Lean root crashed
  Windows once (tiered roots only).
- **Prior results index:** docs/novel/README.md (read before opening any branch; two 2026-09-04
  branches were rediscoveries), docs/proof-search/alignment-rules.md, docs/proofs/.
- **Standing directions (the human's):** use the machine to find NEW rules and relationships; note
  known results in a line, never rewrite them into machine analogy unless it seeds a machine-driven
  investigation and is labelled as such; describe a mechanism before naming a theorem it resembles
  ("explained by CRT" is a description, not a proof the object persists); no attribution trailers
  on commits; round summaries plain-language first.

Statuses: STRONG (tested, holds, mechanism visible), OPEN, WEAK (holds, no mechanism), DEAD
(refuted or proved unable), FACT (exact, kept, not a route). The tree below carries the verdicts;
the log at the bottom is chronology only.

## The tree

- **ROOT. An opening always lands in the window.** For every machine {5..y} the longest
  opening-free stretch stays below the window's growth, F(y) < W(y) - y/6. Accepted as true; the
  work is the proof. Three formulations hang off the root: per step, whole window, and the
  structure of the record itself.

  - **R1. Per-step formulation (the ladder).** The budget inequality F(M+q') <= F(M) + q' at every
    step; summed, it keeps F below W. Theorem (attainment identity): budget = PAIR statement and
    CHAIN statement. Eleven rungs certified. STRATEGIC VERDICT (2026-09-04, from 1e below):
    F(M+q') >= F_2(M) >= 2 d_0(M) is a theorem and d_0 is the column of the first twin pair above
    the top gear, so ANY per-step bound implies a twin-Bertrand postulate; the per-step form asks
    for more than the kernel route needs. Status OPEN, and at least as hard as twin-Bertrand.

    - **1. Pair statement F_2(M) <= F(M) + q'.** Free while F(M) < q' (through m17); content from
      m19; slack 5..25 widening. OPEN.
      - 1a. Phase-shift / sole-coverer descent: "the record of M is a one-hole stretch of M minus
        its top gear, so F(M) <= F_2(M^-)". Spawned by the tiling observation (records are near-
        perfect tilings, every gear a sole coverer somewhere). DEAD 2026-09-04: fails at m17
        (18 > 16) and m23 (34 > 31); the top gear makes 2-3 kills in the record, so the descent
        is the spectrum-plus-depth bound already known to fail (2e). What survived: the tiling
        observation itself, which became branch 5.
      - 1b. Descent through the survivor generator (F_2 at M is layer 0 of the algebra one gear
        down). DEAD: recursion with no base, hole costs not monotone in J (m29: 12, 10, 5, 15, 5);
        same verdict as 2d.
      - 1c. One-class transfer: one-hole(P_k) = j(P_{k+1}) through k = 18, so the one-class pair
        statement is the one-class increment statement. Literature (2026-09-04): unasked in print
        in either class count; the published two-class maximum over class assignments violates the
        increment once (A072753, 10 -> 24 at 13), so the real teeth are needed. CLOSED as a
        transfer; kept as the classification "needs the teeth".
      - 1d. Data past the scan wall by SAT (coverability spectrum). INSTRUMENT: lower bounds only
        beyond m41 (F(61) >= 171, F(67) >= 175, F(71) >= 185); no upper bound, so F_2(59) <= 173
        stays conditional and the pair statement is untested past m31.
      - 1e. Mirror at column 0: F_2 >= 2 d_0. Spawned by the always-open column 0 and the mirror.
        Became the OBSTRUCTION (prover A, research/proof/pair_statement.md): the pair statement at
        column 0 reads 2 d_0 <= F + q', the window's first opening within half the budget; every
        route to it is twin-Bertrand (d_0 <= q') or a Rankin-type lower bound on F against a bound
        on the first twin. Lemmas proved there: L2 (F_2 <= F + min flank, free through m31), L3
        (column-0 equivalence), L4 (every gear is a sole striker in any above-record stretch),
        L5, L6 (left tiling = negated right tiling, equal iff g | x).
        - 1e.i. d_0 measured to level 33,317 (7d, 2026-09-05): d_0 is the column of the first twin
          pair above q at every level, d_0 <= q', inside the window by 10-58x; the mirror forces
          F_2 >= 2 d_0 and nothing more, slack growing to 8x at m53. FACT; confirms the floor only.

    - **2. Chain statement Q*_J(M) <= F(M) + q' for J >= 3.** Kernel at six literal steps as the
      increment form; padded case open (m31 event). OPEN.
      - 2a. Par trading as a theorem (each added letter paid by the flank envelope; Delta_J measured
        in [-3, +4]). DEAD (prover B): eps in [-21, +15] on the family against s_min 8.
      - 2b. Literal case from the middle-sum lemma with the pair statement as black box. Reduced to
        the literal flank envelope; OPEN, and the envelope's per-J form assumes a measured
        inequality (docs/proofs/16).
      - 2c. Padded case by the record law one level down. DEAD: no base, q' > F(M^-) fails from m29.
      - 2d. Survivor-algebra contraction across layers. DEAD: layers non-monotone.
      - 2e. Spectrum-plus-depth (F_J only, no legality). DEAD as a uniform tool: fails at 29->31 and
        47->53 (A_kill >= 4). Legality must be used.
      - 2f. Adjacent-teeth sub-family. Spawned by prover B's observation that every pinned chain
        violator on the family has a gear with adjacent teeth (impossible for real gears,
        AnchorChain.neighbour_of_hit) and the sub-family with no adjacent teeth and 3a = q' -+ 1 had
        zero violators in 2,568 rows to m19. STRONG, then REFUTED (prover C, 23->29 sweep): member
        teeth (1,1,4,2,7,1,5), gears 5 and 7 real, no adjacent teeth, incoming tooth pinned, gives
        F(M + 29) >= 62 > budget 61. Verdict: no ingredient set short of the real higher gears'
        teeth has zero counterexamples.
      - 2g. Three-gap repulsion (from 5b below, feeds the chain): every 3-run whose middle gap is
        >= q' stays within F + q' at P_5..P_8 and m11..m19; the 3-run record always has a tiny
        middle between two big flanks; prover C's padded statement P (flanks of a gap j q' sum to
        <= F - (j-1) q', 0 failures in ~130k family rows, margin 0 once) is its exact-multiple case.
        STRONG as a pattern, no mechanism, unproved.

  - **R2. Whole-window formulation.** F(y) < y^2/6 directly, by a bound that uses the teeth. The
    least demanding formulation (it localises the next twin only below y^2). In print as a
    conjecture (Ziller-Morack 2017 Conjecture 6 at the real teeth); no two-class upper bound of any
    kind in print. OPEN.
    - 3a. Explicit-constant Iwaniec-type bound for the two-class sieve. DEAD (prover D,
      research/proof/iwaniec_two_class.md): the engine becomes a dimension-2 sieve whose lower
      function vanishes for s <= 4.27 while the window sits at s = 2; finite certificates 1.7x ->
      35x over budget, growing as z^3.68; a class-count bound with constant below 1/6 is the
      conjecture itself. Rediscovery of docs/novel/j2-upper-bound.md (rounds 22-25). Any count-only
      route is closed; R2 survives only through the specific teeth.
    - 7b. The anchor pattern inside the window, measured literally (2026-09-05,
      research/proof/anchor_window.md). Spawned by the human's proof shape (a pattern that repeats,
      lands in the window at a higher level than needed, whose survivors carry twins). FACT, new:
      the anchor {5..13} is rigid in every window to Q = 5000 (openings sorted modulo any higher
      gear miss their fair share by fewer than 30, proved from the interval discrepancy of the 180
      re-toothed anchors). CANDIDATE OBJECT, but exhausted at the first gear above the anchor:
      after it the survivors are the lower machine's pattern, each later gear's take follows one
      curve in ln g / ln Q' with white residual, and from the second gear on the branch re-derives a
      known one-prime identity. DEAD as a route.

  - **R3. Structure of the record: how a record stretch is made.** If what makes a record is
    understood, the object that survives it may be nameable. Spawned by the tiling observation
    (out of 1a).
    - **4. Genealogy (records recruit runner-ups).** WEAK: exact at 8 steps (ancestor a runner-up
      by 2-14, largest gap merged one level down 7 of 8, 1-5 generations), no rule stated; the
      theory "bounded branching bounds growth" is untested.
    - **5. Made at the top (near-perfect tiling).** STRONG as an observation: overlap in a record
      stretch is tiny, the top three or four gears do the work, the top gear alone covers one or
      two columns. Refinement 2026-09-04: the one-hole record is its own extremal object (at m29 it
      is the pair (30, 25), neither a record gap), so "join cost = record + ordinary neighbour" is
      too narrow.
      - 5b. Adjacency repulsion: gaps next to a large gap are shorter than independence gives.
        Spawned by the data F_2 - F = 1.1-1.8 typical gaps, below the ln ln N of independent gaps.
        TESTED, holds and grows (F_2 actual 11..39 against shuffled 12..55; gap after a gap >= 0.7F
        below the mean at every machine). Mechanism hypothesis: the left tiling at an opening is
        the negated right tiling gear by gear, and a good tiling is generically not self-dual
        (proved as L6; the size consequence is not). Then found to be the round-19 SUPPRESSION LAW
        with the RENEWAL LADDER as its rigorous side (docs/novel); what stays heuristic there is
        the rate-to-maximum step, the same step every branch meets. Structural (95% of family
        members) but at column 0 the correlation is +1, so not the route. STRONG pattern, closed
        as a branch; child 2g above.
      - 5d. Every gear is needed for the record, and record phases are fixed at the anchor, gear
        7 and the top gear with the middle gears free (7d, 2026-09-05, research/proof/
        anchor_runs_zero.md). Exact to m23: F(M minus g) < F(M) for every g; at m19 all 20 record
        stretches share one phase mod 35; at m17 ten non-mirror pairs agree at (5, 7, 11, 17) and
        differ only at 13. FACT, new at record scale. CANDIDATE OBJECT: what the top gear and gear
        7 do to each other at a record is the one place the record is made. OPEN, next to test.
        - 5d.i. The record as a frame of three gears. Spawned by 5d's observation that the
          record set shares the (5, 7, top) phase with the middle gears free. Theory: the frame
          decides where a record can sit, the middle gears' filling decides whether it does; the
          window contains every frame, so what stops the filling there, and is it systematic.
          OPEN, prover running 2026-09-05 (research/proof/record_frame.md).
        - 5d.ii. What each gear holds up, in the period and in the window. Spawned by 5d's
          observation that every gear is needed for the record. Theory: the deletion profile
          F(M) - F(M minus g) over g, taken on the period record and on the window's own longest
          stretch, differ systematically; gears that never hold up a window stretch would be a
          named object. OPEN, prover running 2026-09-05 (research/proof/deletion_profile.md).
      - 5e. Where a record gap can start: the slot F mod 5 dictates (7a, 2026-09-05,
        research/proof/anchor_cycles.md). F = 1 mod 5 starts on 11|13, F = 4 on 17|19, F = 2 or 3
        on a mirror pair of slots, F = 0 on any; exact at all eight full periods to m31. FACT, new,
        position only.
      - 5f. Position facts kept as breadth, not opened (docs/novel): corridor resonance (big gaps
        recur at slot separations 35, 70, 105, left endpoints pinned to residues {10, 12, 18} mod
        35), the golden spectral gap (gear 5's local frequency mode is phi, phi/3 a machine-
        independent spectral gap). Both subject to the escape-distance-1 ceiling; a spectral
        large-sieve route would give count bounds and meet the rate-to-maximum step.
    - **6. Coherent spacings.** Theory: the real teeth's one rational spacing (d_g = 3^-1 mod g)
      makes the real machine an outlier with small F. DEAD 2026-09-04: coherent spacing vectors
      have the same F distribution as random symmetric vectors at m13 and m17; the real machine
      sits at the 14th and 22nd percentile; coherence explains nothing. The outlier's mechanism
      stays open.
    - **7a. Cycles as the unit** (the anchor 2,3,5 line, 2026-09-05, research/proof/
      anchor_cycles.md). Theory: the dead-cycle record has its own, smaller increment. Identity
      proved: F_c(M) = floor((F(M) - 2)/5) exactly (gear 5's teeth, the mirror, 5 | P; checked to
      m31, 6.7e9 cycles), so the cycle frame is the column frame divided by five and no cycle
      increment below q'/5 exists that is not the budget inequality sharpened. Mechanism at the
      record: live cycles each with one open slot 29|31, the new gear taking consecutive entries of
      its open-multiplier list (79/79 literal); dead cycles need three gears except j = 2 mod 7.
      REFUTED: q'/15, class q' mod 30 dependence (the hit set is class-free), the wall bound as a
      certificate. DEAD as a route; 5e kept.
    - **7d. Runs as the unit and the zero mirror** (2026-09-05, anchor_runs_zero.md). Theory: the
      region just past zero, where every gear's tooth has just landed, is rich in openings. DEAD:
      it is thinner than the period mean (0.94 at Q = 997 falling to 0.79, below all 1,000 random
      stretches from Q = 401) because exclusive kills start at g^2 so the effective machine at
      column k is {5..sqrt(6k+1)}; every gear makes an exclusive kill in the window at Q = 997 so
      no proper subset of gears determines it; any statement about (0, W] provable from tooth
      positions is a statement about the twins below Q'^2. 5d kept.

## Dead ends on record (do not re-enter; alignment-rules.md section 6 and 8)
Residue arithmetic at any bounded modulus (escape distance 1); gears 5,7 capping the padded depth
past 53->59 (CORRCAP infinite); fixed-depth counting (kills nothing); pairwise convexity / SDP
(stops at m19); capacity and overlap counting (nearly achievable, no slack); transfer matrices over
the gap word (refuted twice); symmetry levers beyond the mirror (group is Z/2); letter size as the
driver of L (refuted on the family); congruence-class potentials (certify nothing); class-count-only
sieve bounds at the window's scale (dimension-2 limit); coherent spacings; the cycle frame as a
route; the region past zero as a source of openings.

## Standing directions (the human's)
Read docs/novel/README.md before opening any branch (two 2026-09-04 branches were rediscoveries).
Use the machine to find NEW rules and relationships; note known results in a line, never rewrite
them into machine analogy unless it seeds a machine-driven investigation and is labelled as such.
Describe a mechanism before naming a theorem it resembles; "explained by CRT" is a description,
not a proof the object persists. Every branch is judged by whether it moves toward the target
object. Window = certified range; stretch = sliding run; the budget inequality is a target, not a law.

## Log
- 2026-09-04: tree opened. Lanes running: prover A (branch 1), prover B (branch 2), SAT instrument
  (1d), literature (1c, 3). Manager on 1a and 5.
- 2026-09-04, manager, branch 1a descent: DEAD. F(M) <= F_2(M^-) fails at m17 (18 > 16) and m23 (34 > 31);
  the top gear makes 2-3 kills in the record stretch (sole-coverer counts 2, 1, 1, 2, 1, 3 at m7..m23, all
  its strikes there landing on old openings at m23), so the record of M is a 2-4-hole stretch of M^-, and
  the descent is exactly the spectrum-plus-depth bound F(M) <= F_{kills+1}(M^-), already on record and
  already known to fail as a uniform tool. Likewise F_2(M) <= F_3(M^-) fails at m17, m19, m23.
  What survives of 1a: the tiling observation (branch 5). No bound.
- 2026-09-04, manager, branch 5 refinement: "join cost = record + ordinary neighbour" is too narrow -
  at m29 the one-hole record 55 is the pair (30, 25), neither a record gap (F = 43). The one-hole record
  is its own extremal object; the statement to prove is about it directly: the longest one-hole stretch
  exceeds the longest zero-hole stretch by less than the next prime.
- 2026-09-04, manager, branch 5 data: F_2 - F = 1.40, 1.48, 1.83, 1.41, 1.07 typical gaps at m11..m23
  (pairs (5,6), (5,11), (7,18), (21,10), (34,5); at m29 (30,25) on record). An independent-gaps model
  puts the one-hole excess near ln ln N typical gaps (2-3 here); the machine is BELOW that at every
  machine. Theory to test (branch 5b): gaps adjacent to a large gap are shorter than independent
  gaps would be - a negative correlation between W^-(x) and W^+(x) at an opening x, mechanism
  candidate: for each gear the nearest kills on the two sides of x are at distances d^- + d^+ = +-a_g
  (mod g), so one side's kill pattern fixes the other's per gear. OPEN, untested.
- 2026-09-04, manager, branch 5b TESTED, holds and grows: F_2 actual 11, 16, 25, 31, 39 at m11..m23 against
  the same gaps shuffled 12-14, 18-22, 27-36, 36-43, 50-55 (20 shuffles); E[gap after a gap >= 0.7F] =
  2.77, 3.07, 3.44, 3.52, 3.09 vs mean gap 2.85, 3.37, 3.82, 4.27, 4.68; after the record 2.0, 3.0, 3.7,
  2.6, 3.0. Status STRONG as a pattern. Mechanism hypothesis (residue-exact, unproved as a bound): at an
  opening x, gear g's two teeth sit at right offsets {t, t + a_g} and left offsets {g - t - a_g, g - t}
  (mod g); the left tiling is the negated right tiling gear by gear (the mirror W^-(x) = W^+(-x)); a
  good tiling of the right is generically not self-dual, so the left ends sooner. Handed to prover A.
  Worked gear-5 table at an opening: x = 0 mod 5 -> right strikes {1,3}, left {2,4}; x = 2 -> {1,4} both
  sides; x = 3 -> right {3,5}, left {2,5} (neither side's column 1 is struck by 5).
- 2026-09-04, literature lane (research/proof/literature_increment.md): branch 3 is IN PRINT as a
  conjecture - Ziller & Morack 2017 Conjecture 6, h_2(n) < p_n^2 - p_n, with their Theorem 4.1 that it
  implies infinitely many prime pairs for every even difference; ours is that conjecture at the real
  teeth (F(y) < y^2/6). No two-class upper bound of any kind exists in print; a constant below 1/6 is
  unattempted, not known out of reach. Branch 1c: the increment inequality is UNASKED in print in either
  class count (nearest: multiplicative h(k) < 2h(k-1), Hajdu-Saradha 2012 / Ziller 2019); the one-hole
  identity is Hagedorn 2009 Prop 2.5 (Haugland): k holes with r-k primes <=> no holes with r primes.
  The published two-class MAXIMUM over class assignments violates the increment once (A072753: 10 -> 24
  at 13), so the real teeth are needed. Manager's caution (mine, unverified): the window L ~ y^2/6 lies
  below the dimension-2 sieve limit (Selberg/DHR beta_2 ~ 4.27), so generic sieve upper bounds cannot
  reach branch 3; it needs the teeth.
- 2026-09-04, prover B (research/proof/chain_statement.md): NO PROOF. 2a par trading DEAD as a consequence of
  the invariant ingredients (eps in [-21, +15] on the family vs s_min 8); 2b reduced to the literal flank
  envelope; 2c DEAD (no base: q' > F(M^-) fails from m29); 2d DEAD (layers non-monotone). Chain violators on
  the family 1/180, 1/1440, 36/12960, 193/142560 (free tooth), 0, 0, 3, 46 pinned; the pair statement holds at
  every violator but one, so pair => chain has no proof from the shared ingredients. NEW branch 2f, STRONG:
  every pinned violator has a gear with ADJACENT teeth (2u_q = +-1, impossible for real gears:
  AnchorChain.neighbour_of_hit); the sub-family with no adjacent teeth AND 3a = q' -+ 1 has ZERO chain
  violators in 2,568 exhaustive rows to m19 and a 600-row sample at m23 (min margin 2). Theory: the chain
  statement follows from the invariant ingredients plus those two kernel facts. Smallest unproved statement:
  flanks of an occurrence of a sum to <= F + b; of q' to <= F; of (a, b) to <= F. Deepen next.
- 2026-09-04, prover A (research/proof/pair_statement.md): NOT PROVED, obstruction exact. The mirror makes
  column 0's pair (d_0, d_0) with d_0 = the first open column after 0 = the column of the first twin prime
  pair above p (2,3,3,5,5,5,7,7,7,10 at p = 7..41). The pair statement at column 0 is 2 d_0 <= F + q': the
  window's first opening within half the budget. Every route to it is twin-Bertrand (d_0 <= q', i.e. a twin
  pair in (p, 6q'], OPEN) or a Rankin-type lower bound on F against a bound on the first twin (the twin
  conjecture, quantitative). So (D) uniformly contains a quantitative twin-existence statement at every prime;
  the real teeth enter the pair statement at column 0 AS twin existence. Elsewhere the pair statement is a
  one-hole Jacobsthal statement, FREE through m31 (F_2 <= F + min flank, L2), content from m37. Branch 5b's
  adjacency correlation is structural (95% of family members) but cannot be the route (at column 0 it is +1).
  Lemmas proved: L2, L3 (column-0 equivalence), L4 (every gear is a sole striker in any above-record stretch,
  teeth-free, both worlds; single-gear re-phasing certificate), L5, L6 (left tiling = negated right tiling,
  equal iff g | x). Branch 1 status: OPEN, at least as hard as twin-Bertrand; a proof must LOCATE the next
  opening after a point at every scale - the walk (anchor line) is the object.
- 2026-09-04, manager, branch 5c (repulsion in three-gap runs, both worlds, full periods): every 3-run whose
  middle gap is >= q' stays within F + q' (P_5..P_8 and m11..m19, max 3-run with big middle 22, 28, 34, 42
  and 32 against budgets 27, 39, 45, 57, 48); the 3-run RECORD F_3 always has a tiny middle between two big
  flanks (2 between 12,12; 2 between 16,16; 6 between 22,12; 2 between 22,22; 7 between 10,18). Prover C's
  padded statement P (flanks of a gap j q' sum to <= F - (j-1) q', 0 failures in ~130k family rows, margin 0
  once, no teeth) is the exact-multiple case of this. Status STRONG as a pattern, unproved, no mechanism.
- 2026-09-04, manager, STRATEGIC (from prover A's column-0 verdict): F(M+q') >= F_2(M) >= 2 d_0(M) is a theorem
  (deletion ladder + mirror), and d_0 is the column of the first twin pair above p. So ANY per-step increment
  bound F(M+q') <= f(F(M), q') implies d_0 <= f(F, q')/2, a twin pair below a bound in p - a twin-Bertrand
  postulate, open. The ladder (D) therefore asks for MORE than the theorem needs: the kernel route needs only
  an opening in (y, y^2], i.e. F(y) < y^2/6 (branch 3), which localises the next twin only below y^2. The
  per-step formulation over-asks by exactly a twin-Bertrand statement. Consequence for the tree: branch 3
  (direct window bound using the teeth) is the least demanding formulation; branches 1-2 (pair, chain)
  cannot be proved without twin-Bertrand. Caveat (mine): branch 3 at scale y^2 sits below the dimension-2
  sieve limit, so it needs the specific teeth, not a generic sieve; nothing in print attempts it.
- 2026-09-04, manager: branch 3a OPENED - explicit-constant Iwaniec-type bound for the two-class sieve, aimed
  at F(y) <= C_2 y^2 with C_2 < 1/6 (Ziller-Morack Conjecture 6 is the same target; no explicit constant in
  print). Prover D launched: reproduce Iwaniec 1978 with constants, redo for two classes, compare to 1/6,
  name the lossiest step. Running alongside prover C's 23->29 sweep and the SAT instrument.
- 2026-09-04, SAT lane (research/proof/cov_spectrum.md): branch 1d - COV(M) was BUILT in round 20 (mechanic.md
  K1, research/cov_sat.py, m41 complete); the harvest tag was stale. New verified lower bounds F(61) >= 171,
  F(67) >= 175, F(71) >= 185; 15 two-sided decisions all equal to the corpus; Q*_5(29) witness (7,10,21,10,7)
  reproduced. UNSAT cost grows 6-11x per rung; no upper bound past m41, so the pair statement is untested past
  m31 and F_2(59) <= 173 stays conditional. Counting fallback vacuous from m37 (sum 2/q > 1). Status: the
  instrument gives lower bounds only beyond the wall.
- 2026-09-04, prover D (research/proof/iwaniec_two_class.md): branch 3a DEAD. Iwaniec's shifted sieve
  transfers to two classes verbatim, but the engine (Rosser's linear sieve) becomes a dimension-2 sieve whose
  DHR lower function vanishes for s <= beta_2 = 4.27 while the window sits at s = 2: the two-class transfer
  gives F(y) <= C y^{4.27+eps}, not C_2 y^2. Explicit finite certificates: one-class 0.67 -> 0.19 p^2; two-class
  1.7x -> 35x OVER budget, growing as z^3.68. Class-count-only methods bound ZM's h_2, and h_2 <= 6 C_2 y^2
  with C_2 < 1/6 IS the twin prime conjecture. Branch 3 survives only through the specific teeth.
- 2026-09-04, manager, branch 6 OPENED: COHERENT SPACINGS. The real machine's tooth spacings are one rational
  for every gear (d_g = 3^-1 mod g: the teeth split each gear 1:2), so "gear g double-strikes at distance w"
  is the multiplicative event g | 3w(3w-1)(3w+1); a counterfactual member has arbitrary spacings. Inside the
  window a gear q' strikes at most three columns (layer law), so chains are a full-period object, not a
  window object. Theory to test: coherent spacing vectors (any single rational r) give systematically small F
  on the family; if so, coherence is the outlier's mechanism.
- 2026-09-04, manager, branch 6 DEAD: coherent spacing vectors (v_g = (c/2d) mod g for rationals c/d, d <= 30)
  have the same F distribution as random symmetric vectors at m13 (n = 77: min 10, median 13, max 20 vs
  random min 10, median 13, max 20) and m17 (n = 62: 15/19/25 vs 14/19/30); every coherent member below the
  real machine has a degenerate gear (adjacent teeth). The real machine's 1/3 spacing sits at the 14th (m13)
  and 22nd (m17) percentile of the random family. Coherence per se explains nothing; the outlier's mechanism
  stays open (record 9.3 item 22).
- 2026-09-04, prover C (23->29 sweep, in progress): branch 2f REFUTED. Member teeth (1,1,4,2,7,1,5): gears 5 and
  7 real, no adjacent teeth, incoming tooth pinned; F = 32, F_2 = 48, budget 61; literal depth-4 run
  (18) + [10, 19] + (15) = 62, with only the end openings surviving at the phase that puts the middles on the
  teeth of 29, so F(M + 29) >= 62 > 61. Phi(a, b) = 33 > F = 32 (statement L2 fails; the pair statement holds).
  I + (T) + (L) is not sufficient; the chain half needs the higher gears' real teeth too. Status: OPEN, no
  ingredient set with zero counterexamples short of the real machine itself.
- 2026-09-05, manager, REVIEW OF docs/novel (overdue): two of today's branches were rediscoveries.
  (i) Branch 3a's verdict is docs/novel/j2-upper-bound.md rounds 22-25: the two-class exponent sits at the
  dimension-2 sifting limit 4.266, ZM Conjecture 6 asks for exponent 2, the blocker is parity via ZM Thm 4.1;
  three explicit upper rungs on j_2 (down to exponent 8.04) and a lower ladder exist. (ii) Branch 5b is the
  round-19 SUPPRESSION LAW (docs/novel/suppression-law.md: adjacent large gaps anti-correlated, deficits
  x26..x1400 vs independence, lag-1 only, rebound at lag 2), whose rigorous side is the RENEWAL LADDER
  (docs/novel/renewal-ladder.md, round 20): closed-form CRT upper bounds on joint qualifying-gap counts,
  nested, converging to exact, first joint bounds at machines beyond scan; what stays heuristic in (D) there
  is the order-statistics step from a rate to a maximum - the same step every branch here hits.
  NEW LEADS not in the tree: (a) renewal ladder + SAT zero-certification for prover C's flank statements at
  finite machines; (b) self-similarity items: golden-spectral-gap (gear 5's golden mode, phi/3), corridor-
  resonance (extreme gaps phase-locked mod 35), kleene/survivor generators (the depth family as one max-plus
  algebra); (c) breadth: eigenvalue-statistics (GUE), nontensor-sector, farey-chebyshev-spectrum.
  Rule from now on: read docs/novel/README.md's index before opening any branch.
- 2026-09-05, manager, self-similarity items read: golden-spectral-gap (proved + T3 law kernel-checked:
  gear 5's local frequency +-2 mode is phi exactly and phi/3 is a machine-independent spectral gap, the
  Fourier face of "gear 5 controls all corridor phenomenology"; its named open construct is a large-sieve
  inequality on window counts from the exact spectrum) and corridor-resonance (measured, exact counts: big
  gaps recur at slot separations 35, 70, 105 with almost no damping, left endpoints pinned to residues
  {10, 12, 18} mod 35 at every machine; the process is not Markov to order 4, the memory is the corridor
  phase). Verdict for the tree: both are POSITION facts (where extreme gaps sit, how they recur), subject to
  the escape-distance-1 ceiling; the spectral large-sieve route would give COUNT bounds and meets the same
  rate-to-maximum step as the renewal ladder. Real self-similarity, no size lever found in them. Kept as
  breadth items, not opened as branches.
- 2026-09-05, writer lane (docs/proofs/, 19 files + index): seven recorded proofs found weaker than recorded
  when written out - (1) the per-J flank envelope Phi_J <= F_2 + s_min - m_min(J) assumes the MEASURED
  Delta_J <= s_min; proved is only Phi_J <= Q*_J - m_min(J) (docs/proofs/16; alignment-rules 3.7 corrected);
  (2) saturation needs F(M) < 2u_q in column units (docs/proofs/06); (3) the attainment identity has no Lean
  theorem, its sharp direction is a written assembly (08); (4) the record law at 17 is verified at both ends,
  not derived (09); (5) the Polignac cap's reduction to 8 representatives is a written step (13); (6) the
  alignment law now has a full CRT proof (04) where the record had a check on 103 gear sets; (7) the mod-35
  completeness lemma holds for n <= 5 by the q <= 2n bound (14), resolving the harvest disagreement.
- 2026-09-05, prover (branch 7d, research/proof/anchor_runs_zero.md): RUNS AS THE UNIT / ZERO MIRROR - no lever
  on existence. (1) Every gear makes an exclusive kill in the window at Q = 997 (one or two top gears make none at
  Q = 59, 173, 499, decided by g^2-2 and gQ'+-2 primality: the square gate), so no proper subset of gears determines
  the window's openings. (2) The record stretch is never inside any gear's clean end zone because every gear is
  needed for the record (F(M minus g) < F(M) for every g, m7..m23) - a theorem with no position in it; the record SET
  has, beyond the mirror, a middle-gear degeneracy: same anchor+7 phase and same top-gear phase, middle gears complete
  the stretch in several ways (m17: 10 non-mirror pairs agree at (5,7,11,17); m19: all 20 records share one phase mod
  35; m23: the two non-mirror pairs agree at (5,7,23)) - branch 5 / corridor law at the record. (3) d_0 = column of
  the first twin above q at every level to 33,317, d_0 <= q'; the mirror forces F_2 >= 2 d_0 only, slack growing to 8x
  at m53. (4) The stretch (0,W] has FEWER openings than a random stretch of its length (ratio to the period mean 0.94
  at Q = 997 -> 0.79; below all 1,000 random stretches from Q = 401): exclusive kills start at g^2, so the effective
  machine at column k is {5..sqrt(6k+1)} and the count is 0.79 x the effective product at every Q >= 100 (Mertens
  bias). (5) DEAD: any statement about (0,W] provable from tooth positions is a statement about twins below Q'^2, and
  the one forcing an opening in (q/6, W] is twin-Bertrand at scale Q'^2. Scripts research/anchor235/r34/.

- 2026-09-05, prover 7a (cycles as the unit), research/proof/anchor_cycles.md. NEW, exact: (N1) a record gap can start
  only on the twin slot that F mod 5 dictates (F = 1 mod 5: slot 11|13; F = 4: 17|19; F = 2, 3: mirror pairs on
  {29|31, 17|19} and {29|31, 11|13}; F = 0: any), exact at all eight full periods {5..7}..{5..31}; position content only.
  (N2) the dead-cycle record is F_c(M) = floor((F(M) - 2)/5) exactly, proved from gear 5's teeth, the mirror and 5 | P,
  exact to {5..31} (6.7e9 cycles). Consequence: the cycle frame is F/5 in disguise, so no cycle increment bound below
  q'/5 exists that is not a sharpened budget inequality (best on record 0.162 at 31->37). Mechanism at the record
  {5..29}: an 8-run of cycles, live cycles each with one open slot 29|31, gear 29 taking consecutive entries of its
  open-multiplier list (2u' apart), 79/79 glue kills literal at all machines. Dead cycles need three distinct gears
  except j = 2 mod 7 where gear 7 takes two slots (persists at every machine). REFUTED: q'/15 as increment bound
  (8 of 13 rungs), the wall bound H_1 as a certificate, any class q' mod 30 dependence (the six-residue hit set is
  R_g = -30^-1 x {11,13,17,19,29,31} mod g, class-free). Branch 7a DEAD as a route.
- 2026-09-05, prover 7b (the anchor pattern in the window, measured literally at every prime level 17..5000, three
  anchors), research/proof/anchor_window.md. Gate held at all 400,000 gear-rows: survivors = twins in every window and
  section; gear g strikes no survivor below column (g^2-1)/6. NEW, exact: the anchor is rigid inside the window - the
  openings of {5..13} sorted modulo any higher gear deviate from their fair share by less than 30 in every window (proved
  from the interval discrepancy of the 180 re-toothed anchors; real teeth 7.54, worst 14.09; measured <= 11.4 at
  W = 4.2e6 columns). That rigidity is exhausted at the first gear above the anchor: after it the survivors are the lower
  machine's pattern, not the anchor's. What each later gear removes follows a curve in t = ln g / ln Q' alone (1.000 of
  fair share for t < 0.55, 0.957 at t = 0.62, 1.87 as t -> 1), same for every anchor; mechanism: where the multiplier
  columns m = (6k -+ 1)/g sit relative to g^2 (primes near g dense, thin near g^2). The residual after that curve is
  white (z mean 0.02, sd 1.004, max 4.02 over 105,919 gear-rows). The real tooth pair is the most-struck of all (g-1)/2
  pairs for 99.9% of gears in (Q/4, Q/2], the unfavourable direction. No mod-30, anchor-class or mirror structure in
  the discrepancies. STOP LINE: from the second gear on the branch re-derives a known one-prime identity gear by gear;
  stopped there per the human's direction. Branch 7b DEAD as a route.
- 2026-09-05, manager, on branch 7 as a whole. The human's proof shape has three parts: the pattern repeats (exact),
  its survivors in the window are twins (kernel), it lands in the window (open). The anchor frame does not change the
  third part: the cycle record is F/5, the anchor's openings are rigid in the window but the gears above the anchor
  take their share from the lower machine's pattern, not the anchor's, and the region past zero is thinner than the
  period mean, not richer. Candidate objects for "always in the window" produced this round, none yet shown forced:
  the record's phase structure (anchor + gear 7 + top gear fixed, middle gears free; every gear needed), the slot rule
  for where a record can start (F mod 5), the anchor's in-window rigidity (exhausted at the first gear).
- 2026-09-05, manager: opened 5d.i (record frame of three gears) and 5d.ii (deletion profile, period versus window) under 5d, the STRONG node with a CANDIDATE OBJECT; two provers under the theory-tree skill.
