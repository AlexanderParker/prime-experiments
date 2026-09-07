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
- **Unstick file:** research/proof/dead_branches_reopened.md (the skill's protocol: for every dead branch the object, the attack vectors, the reason for failure, two or more ideas through, two or more realisations each). Rerun whenever no STRONG node is left.
- **Standing directions (the human's):** use the machine to find NEW rules and relationships; note
  known results in a line, never rewrite them into machine analogy unless it seeds a machine-driven
  investigation and is labelled as such; describe a mechanism before naming a theorem it resembles
  ("explained by CRT" is a description, not a proof the object persists); no attribution trailers
  on commits; round summaries plain-language first.

Statuses: STRONG (tested, holds, mechanism visible), OPEN, WEAK (holds, no mechanism), DEAD
(refuted or proved unable), FACT (exact, kept, not a route). The tree below carries the verdicts;
the log at the bottom is chronology only.

The view: docs/theory_tree.html is generated from this file by `uv run python
research/tools/tree_view.py` (collapsible tree, verdict chips, search, unexhausted-only filter,
radial map, log). Regenerate and commit it after every branch; this file stays the source.

Canonical words (owner, 2026-09-06): ENGINE (primes <= q) -> VALVES (the engine acting inside
the manifold's open set; the families are the valves) -> MANIFOLD (the primes in (q, q#] as a
machine; regions smooth zone and quiet zone) -> EXHAUST (the tiers above). "Motor", "wheels",
"top machine" and "clutch" in older nodes below read as engine, manifold, manifold and valves.
Back pressure = the manifold's strikes (owner, 2026-09-06).

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

      - 2g.i. The neighbour-sum profile (the manager's scan, research/anchor235/r45/, then
        research/proof/neighbour_profile.md, 2026-09-06). Spawned by the observation that the
        records at m23, m29, m31 are 3-runs of the machine below with a letter as the middle
        and N(letter) = F +- 1. The F + 1 law is DEAD, killed at m29 by 4 at the letter itself
        (N(10) = 48 against F + 1 = 44, run (18, 10, 30)); thresholds v_0 = 7, 6, 8, 8, 6, 21, 8
        at m11..m31 against letters 4, 6, 6, 8, 10, 10, 12; no constant-c version survives
        (max N(v) - F over v >= 6 is 3, 1, 3, 3, 1, 12, 8). WHAT REPLACES IT, exceptionless on
        full periods to m31 (6.4 billion gaps): N(v) <= F_2(M) for every realised v >= 6, tight
        once (N(7) = 55 = F_2 at m29); spikes only at v <= 5. MECHANISM, PROVED (the glue lemma):
        re-phasing the right flank by CRT under any two-colouring of the gears makes the glued
        middle column an opening (it equals x_1 modulo every left gear and x_2 modulo every
        right gear), so the glued object is an adjacent PAIR, bounded by F_2 and never by F;
        this is why no argument of that shape could prove the F + 1 form. The F_2 glue succeeds
        at 426 of 446 attaining 3-runs with v >= 6 at m13..m23 and 66 of 68 at the letters. On
        200 family members the F + 1 form holds at 43-61% (false, not real-teeth); the F_2 form
        at 94-98% (near-structural). L6 across a gap (new, 0 violations in 2.39 million pairs):
        p_g + q_g = -v or -v +- d_g (mod g), a translation only. Why it does not close the chain:
        the F_2 cap gives Q*_3 <= F_2 + b, so the budget needs F_2 - F <= a, and F_2 - F =
        4, 5, 7, 6, 5, 12, 10 against a = 4, 6, 6, 8, 10, 10, 12 fails at m17 and m29. FACT about
        M. Instrument gate: F({5..37}) = 88 produced from m31's period alone (Q*_4, word
        (28, 37, 12, 11)). CHILD NAMED: for every 3-run with v >= 6 there is a two-colouring of
        the gears whose CRT re-phasing blocks the glued target; finite, covering-theoretic, no
        density, no transfer, modulus grows with the machine (the wall's shape); residue 20 runs
        at v in {6, 7, 8, 11}.
        - 2g.i.a. The glue as a covering statement (research/proof/glue_covering.md). DEAD as
          a route: the covering statement is false exactly where it matters. Of 862 attaining
          3-runs with v >= 6 at m13..m31, 756 have v >= min(L, R) where the constant colouring
          is the peel bound; the glue's own rate on the rest is 30 of 106, falling 50%, 62%,
          40%, 0 of 22, 23% at m17..m31, and over all 3-runs with L + R > F it is 4% at m31.
          PROVED, new: the SHADOW LEMMA (the covering instance has exactly two single-sided
          columns, x_1 - v and x_2 + v; the all-left colouring is the run itself and misses only
          the shadow, so the glue's whole content is buying one column; min miss 1 at 178 of 178
          failures) and the MOVE LEMMA (recolouring a gear translates its strikes by v, so a
          strike survives iff v = 0 or +-d_g mod g: padded gears move free and never cover the
          shadow, letter gears keep one tooth, all others lose everything; with L4 every move
          is paid for and the payment cascades). The case that resists every construction is
          the m29 run (18, 10, 30) at x_0 = 278,620,515, the one that killed the F + 1 law.
          NEW EXCEPTIONLESS LAW (the J-run outer law): for J consecutive gaps with every middle
          >= 6, g_1 + g_J <= F_2, 0 exceptions in 3,278,972 runs, J = 3..8, m13..m23, maximum
          falling with J; drop the middle condition and it breaks at once. THE FIRST FACE-C
          EXCEPTION: the real teeth are atypical in gluability, 62.5% against a pooled 9.4%,
          the 99.6th percentile of 223 comparable m19 members, not explained by the count of
          letter gears (exactly the family mean). Toward the root: Q*_3 <= F_2 + b needs
          F_2 - F <= a (fails m17, m29); the level-4 glue gives Q*_4 <= F_2 + q', needing
          F_2 <= F, false at every rung; the deficit F_2 - F is depth-independent because the
          glue forces the hole at every depth. N(v) <= F_2 for v >= 6 survives with no
          constructive route.
          - 2g.i.a.i. Separability of flanks by gear (thin place 6; research/proof/
            separability.md). DEAD as a route; the clue deflates and reverses. Gluability is
            not separability: 0 of 106 hard attaining 3-runs (real, m13..m31) and 0 of 2,832
            family hard runs have disjoint flank covers; the reason is counting (two disjoint
            covers need sum of 2/g >= 2, first reached at y = 109), a face-A obstruction that
            cannot carry face C's exception. The shared gears are the BOTTOM, not the top: one of
            gears 5, 7 is in every minimum shared set (106 of 106); the top gear is forced at
            0 of 16 (m19), 2 of 44 (m31); at 10 of 10 record stretches m17..m31 the layer's own
            top gear is NOT in the minimum shared set (shared {5, 13, 17, 19} at m29,
            {5, 7, 13, 17} at m31): "made at the top" and "separable at the top" are the same
            fact with the sign reversed, the mortar gears are the free ones. The shared gears
            are exactly the ones the move lemma cannot move (0 of 106 all-movable). THE
            ONE-THIRD SEPARATION MAXIMISES SHARING: d_g = (g +- 1)/3 is two thirds of the way to
            the widest admissible separation, so wide teeth straddle both flanks; real P(strike
            both) is at least the mean over separations at 256 of 256 cells; exact arc
            condition verified 2,702 of 2,702. NEW EXACT ARITHMETIC: for the real teeth
            3 d_g = 1 (mod g), so the letter gears of a middle gap v are Leg(v) = {g : g divides
            3v - 1 or 3v + 1}, verified 400 of 400 (v = 6 gives {17, 19}, v = 10 gives {29, 31},
            v = 7 gives {5, 11}). NEW CONNECTION: the m31 record class, read at layer 29 as the
            word (18, 10, 30), IS the resistant m29 run of 2g.i.a at x_0 = 278,620,515: the run
            no local certificate can prove is the one the next gear fuses into the record.
            Bounded-loss glue is false and useless (loss up to 24 at m31; any c > 0 worsens
            F_2 - F <= a). The clue itself, re-measured at matched (v, slack) cells: the real
            machine's hard runs occupy 1, 3, 5 shapes up to mirror; family glues at 25-30% at
            the matched cell, real 5 of 7 mirror classes (P = 0.029); face C's exception
            survives in direction and shrinks from a factor 6.6 to about 2.4. What goes forward:
            the divisor form of Leg and the count of gears carrying no sharing obligation, a
            gear-counting question (thin place 1).
      - 2f.i. Separation compatibility as the chain statement's ingredient (research/proof/
        compatibility_chain.md; thin place 4 of the wall). DEAD: three recorded budget
        violators are FULLY compatible (every gear on one rational): m17 (1,3,4,4,4) with
        rational 8/1, F(M + 19) = 40 > 38; m17 (2,3,3,3,3) with 6/1, 38 > 37; m11 (1,1,5)
        with 29/18, 25 > 24, all re-verified by direct sieve. Compatibility is a LIABILITY:
        coherent members at m17 violate at 8.0% (B = 10) and 2.1% (B = 30) against a family
        rate of 0.28%; neutral at m19. One incompatible gear does not protect (2,627 members
        exhaustive at 23 -> 29: 1 chain violator (2,2,1,2,8,7,5) with Q*_4 = 65 > 63, against 0
        in a matched random control). The 2f refuting member is incompatible as predicted (22
        of 28 pairs, every one containing a moved gear). Mechanism: two gears strike a
        rectangle mod gh with two diagonals; coherence fixes only one (d+ = c r^-1), the other
        is arbitrary, and the real m23 record itself double-strikes at distances 2, 5, 7 at
        the pair (5, 7), the same configuration a violating stretch uses. The tail-gear tooth
        distance is refuted as the alternative (m19 (1,2,2,2,6,3) has every separation above
        the real minimum and F(M + 23) = 52 > 48). New exception-free facts: admissible
        rationals must be coprime to every gear, so they are the 3-smooth ones plus those with
        all prime factors above q (counts 203, 155, 125, 97, 71, 47 at m11..m31; one third is
        available only because 3 is anchor, not gear); the (T) + (L) sub-family at m23 has 4
        budget violators, not 2. Methodological: the family cannot decide these questions by
        frequency (every protective region holds a few hundred members with expected violators
        below one); only a construction can.

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

    - R2.a. The machine feeds on itself (research/proof/self_feeding.md; register entry
      docs/novel/walk-tooth-frame.md, prior art not yet checked). Spawned by the kernel identity
      read across levels (a twin gear pair is an opening of a lower machine in its window).
      FACT, not a route; 13 of 13 pre-registered items resolved, none refuted. Exact, zero
      exceptions unless stated, q = 5..4999 (667 walks): (W1) the walk from q^2 starts ON a tooth
      of the top gear (6 k_0 = q^2 - 1, so k_0 = -6^-1 mod q) and the top gear strikes the whole
      walk exactly once, at its first column; its next strike is d = 2c mod q columns on (2u_q or
      q - 2u_q by q mod 6), and the walk length L stays below d at every q above 53 (one
      exception, q = 53; worst L/d = 0.52 at q = 137, median 0.02). (W2) the deepest layer that
      hops is the top gear iff q^2 - 2 is prime (the square gate): 153 open, all top; 514 shut,
      none. (W3) level-free transfer rule, 832,915 checks: a gear striking column k + j beside a
      birth column strikes column i of that pair's own walk iff it divides (6j)^2 + 6i - 2,
      (6j)^2 + 6i, (6j+2)^2 + 6i - 2 or (6j+2)^2 + 6i; at j = +-1, i = 0 the admissible gears are
      exactly {7, 17, 31} (3,093 carry-overs of 50,906, no other gear). (W4) the next level's walk
      starts at 6k^2 - 2k, exactly 2k below the pair's twin-product column, and both newest gears
      strike it once at distance 2k = (g+1)/3. The chain of landings (97 levels, 46 starts, to
      12 digits) has no rule, as pre-registered. Root reading: the walk is decided by the old
      gears (gear 5 makes 40% of 18,743 hops, gears above sqrt(q) 15%); W1 and W2 rest on L < d,
      a twin-Bertrand-strength statement at scale q/3. Position objects, no size lever.
      - R2.a.i. The path taken apart (the owner's direction, 2026-09-05; research/proof/
        walk_path.md and walk_transforms.md; register entries docs/novel/walk-path-parts.md and
        walk-path-transforms.md, prior art not yet checked). Spawned by W1. STRONG as a
        description, exact, 2,260 walks q = 5..19,997. How the machine builds the path, in the
        owner's frame. PARTS, all proven or one-line: (i) the anchor: q^2 = 1 or 19 mod 30, so
        the walk starts on slot 29|31 (q = +-1, +-11 mod 30) or 17|19 (q = +-7, +-13), never
        11|13, gear 5 never strikes the first column and always strikes offset 1, and takes
        offsets {1, 4} or {1, 3} mod 5 by q's class (so L >= 2, L is never 1 mod 5, and L mod 35
        lies in a 15-element set fixed by q^2 mod 35, 0 exceptions); (ii) each gear g: two
        progressions in the offset i, difference g, separation d_g, phase a function of q^2 mod g
        (tooth rule, docs/proofs/02; 493 million checks); (iii) the QUADRATIC-RESIDUE BAR (new):
        gear g can strike offset i at all only if 2 - 6i or -6i is a square mod g, so which gears
        can reach an offset is q-free (3/4 of the machine generically, all of it at i = -6t^2,
        exactly the gears = +-1 mod 8 at i = 0), and the walk's phase vector is a square in every
        coordinate (density 2^-pi(q) of phase space) while L does not notice it (percentile 0.53
        among tooth starts); (iv) the top gear is INERT on its own walk: it is the smallest striker
        of no path column but offset 0 (0 exceptions, q = 53 included; stronger than W1), and the
        q^2 column is the unique tooth of q in its window where q is the sole striker of its
        member (0 of 337,011 teeth), so the walk starts at the shallowest tooth. INTERACTIONS: the
        proven order-two laws (chain, merge, neighbour-of-hit, tooth sharing, gear-5 lock) are all
        the path uses, thinly: a median of 8 gears strike twice, the walk's stretch is a two-piece
        fusion at 2,234 of 2,259 paths, three at 25, never four; the depth profile is dip -
        plateau - spike (2.42 / 3.24-3.39 / 3.77 against sum 2/g = 3.18 and sum 2/(g-2) = 3.70),
        the spike being neighbour-of-hit; per-offset mean depth is a fixed arithmetic function of
        the offset alone (root counts, correlation 0.97-0.998), and the landing avoids the
        high-depth offsets (0 landings on the 8 highest against 500 of 2,260 on the 8 lowest).
        Two-sided tooth law (2 exceptions, q = 31 backward and q = 53 forward, both in the short
        arc): L < d and L^- < q - d, i.e. the blocked run through the q^2 tooth is shorter than q.
        Re-phasing a gear shortens L only if it is a sole striker (0 of 13,861 counterexamples).
        The square start is a long start (mean L 24.8 against 20.0 over 57,125 tooth starts) and
        the square sub-torus costs 13% of the reachable maximum. Nulls: the section spectrum is
        the gear lines; k_0 is not distinguished by local density; nothing crosses chain levels
        but the frame; L is the twin-gap null to 2% from q = 200 (a rate, stopped). THE FIRST
        UNPROVEN INTERACTION, named by both provers: the length itself - that the 2 pi(q)
        progressions do not cover the d = 2u_q offsets from offset 1 - of unbounded order
        (minimum blocking set median 9, max 43; 88% of paths contain a column blocked only by a
        gear above sqrt q). CANDIDATE OBJECT: the reachability landscape (the q-free set of gears
        that can reach each offset) with the landing preferring its low points; child opened.
        - R2.a.i.a. The reachability landscape (research/proof/reachability.md; register entry
          docs/novel/reachability-landscape.md, prior art not yet checked). Spawned by the
          quadratic-residue bar. STRONG, exact, and it names a CANDIDATE OBJECT. Parts, all
          proven: (i) bar size in closed form, |Bar(g)| = (g + 1 - chi_g(2) - chi_g(-2))/4, so no
          gear reaches every offset (gear 5 reaches offsets 1, 3, 4 mod 5 only; gear 7 reaches
          0, 1, 2, 4, 6 mod 7); (ii) the islands for bound B (offsets no gear <= B can reach) are
          exactly prod |Bar(g)| classes mod P_B by CRT: 4 classes mod 35 for B = 7, namely
          {5, 10, 12, 17} mod 35; 12 mod 385; 48; 192; 960; 5,760 at B = 11..23; (iii) the
          doubling: gear g strikes offset i for exactly 2 chi_g(i) residue classes of q mod g,
          never an odd number, so its mean rate over offsets is exactly 2/g and exactly 0 on a
          quarter of them - the bar concentrates strikes, it does not reduce them (0 of 21,531
          cells); (iv) large gears strike islands at exactly the machine's rate 2/g (0.9956 of
          predicted over 103,899 sightings), so the counting margin through islands is identical
          to the unrestricted problem (strikes per island = sum 2/g, 2.70 at B = 7) and crosses 1
          at q = 53: no counting proof through islands at any B. THE OBJECT (N-R4, 0 exceptions
          in 2,026 primes): for every prime q in (1487, 20000] some offset i = 5, 10, 12 or 17
          (mod 35) with 1 <= i < d = 2u_q is struck by no gear at all; the minimum number of such
          open islands per q grows (0, 0, 0, 4, 12 by band), and 17 primes below 1487 fail (the
          landing is then on a non-island). So "L < d" is witnessed on a FIXED, q-free set of
          offsets of density 4/35 past the square, with growing slack. The landing is an island
          for B = 7 in 32% of walks (not the pre-registered 90%; the four smallest islands 5, 10,
          12, 17 are the four commonest landings, 21% of all), and its island preference is
          exactly order one against a per-gear independent null (0.99, 0.93, 0.88, 0.92 of
          prediction at B = 7..17): 87% of the variance of the depth function is gears 5, 7, 11,
          13, so "the landing avoids deep offsets" IS "the landing prefers islands". Landscape
          mirror: i -> d_g - i preserves the bar iff g = 1 mod 4 and maps it into the reachable
          set iff g = 3 mod 4, so the island set has no reflection symmetry from B = 7 on. THE
          INTERACTION TO PROVE, in the machine's terms: for every prime q there is an offset
          i = 5, 10, 12, 17 (mod 35) with 1 <= i < d such that q is not congruent to +-s modulo
          any gear g in (7, q] for any root s of -6i or 2 - 6i. The sifted variable is q itself
          against a fixed target set; what the landscape does not give is a count. What would
          have to happen for the object to fail: every one of the ~d/9 islands in the top gear's
          arc struck by some gear in (7, q]; not seen above 1487.
          - R2.a.i.a.1. The island witness under pressure (research/proof/island_witness.md;
            register entry docs/novel/island-witness-integers.md, prior art not yet checked).
            Spawned by N-R4 and its 17 failures. STRONG. THE OBJECT SHARPENED: (i) it is about
            integers, not primes: for every integer q coprime to 30 above 2849 (52,574 of them
            to 200,000) some island in [1, d) is open; composites behave exactly like primes;
            every multiple of 5 fails (13,333 of 13,333), by a proved law: a gear dividing q
            relocates its strikes onto the classes i = 0 and i = 2 x 6^-1 mod g, which for
            gear 5 are exactly {0, 2} mod 5, where all four islands lie; powers of 5, 49 and 121
            are the only prime-power failures; (ii) 0 exceptions in 17,748 primes in (1487,
            200000], minimum open islands per band 2, 4, 12, 21, 57, 107, strictly increasing;
            (iii) ONE CLASS SUFFICES: i = 12 mod 35 alone witnesses from q = 5477 (0 of 17,261),
            each of the four classes separately from 13,001; (iv) THE ARC SHRINKS: a free island
            sits inside [1, 0.152 d) for every prime in (20000, 200000] (0 of 15,722), and its
            absolute offset never exceeds 2,392 anywhere to 200,000. So: for every integer q
            coprime to 30 from 2849 on, the column of q^2 + 6i with i = 12 mod 35 and i < 2,392
            is open for some i, i.e. q^2 + 6i - 2 and q^2 + 6i are a twin prime pair. (v) the 17
            prime failures: 16 in the short arc (q = 73 the exception), no residue coincidence,
            exact minimum covers up to 24 gears (0.42 of the islands, most gears taking one
            island each), 20 of 21 failures fragile (deleting one gear frees an island). (vi) THE
            COVER NUMBER K(d), the branch's contribution toward the root: with every gear free
            to choose any reachable phase, used once, the exact minimum number of gears that
            strike every island of [1, d) is K = 3, 4, 6, 9, 14, 20 at d = 35 .. 1120 (ILP,
            certified optimal), against a bounded counting requirement 2, 4, 5, 7, 9, 10:
            counting stalls, covering grows. A failure at q pins q modulo a product of at least
            K(d) gears, 1.1e32 at d = 1120 where q is about 3,000. Inside the real machine the
            minimum blocking set of the struck islands grows linearly (5 to 220 gears from
            q = 127 to 19,699). (vii) B = 11 and 13: witness thresholds 9,281 and 33,623; the
            failure sets nest the other way from the brief's guess (islands nest downward, so
            failures nest upward, 0 exceptions in 17,982). Refuted: covers of at most 6 gears;
            a bounded adversarial cover. THE INTERACTION TO PROVE, sharpened: no integer q
            coprime to 30 can lie in a covering residue class of K(d) or more gears in (7, q]
            with d = 2u_q; the growth of K(d) is the quantity to understand.
            - R2.a.i.a.1.a. The cover number K(d) (research/proof/cover_number.md). Spawned by
              the growth of the adversarial cover. FACT, exact, and it names the obstruction.
              K(d) exact at 23 arcs to d = 1,330, every value ILP-certified: 3, 4, 5, 6, 7, 8,
              9, 10, 11, 12, 13, 14, 14, 15, 16, 17, 18, 19, 19, 20, 21, 22, 22 at d = 35 ..
              1330, against a counting requirement 2..11 that stalls; growth d/(ln d)^3 with
              K (ln d)^3 / d = 6.15 +- 0.20 over sixteen consecutive arcs (not pi(c sqrt d);
              the sqrt fit under-predicts from d = 1,190). NOT the counterfactual family's
              ladder: with a free tooth separation the optimal cover is a perfect partition of
              the islands equal to counting (4 arcs, 0 exceptions), so the family's row is the
              easy one; the machine's fixed separation 2 x 6^-1 mod g costs a factor 1.5 and
              the one-phase-per-gear rule the larger half (at d = 1,120: counting 10, rule
              dropped 12, real 20); the strike budget contributes nothing. K depends on the
              island count and the cheapest gear the bar leaves, not on the arc (K_7, K_11,
              K_13 agree within 1 at equal island count, 11 comparisons). PROVED, no counting:
              a cover with phases is realised by exactly 2^K residue classes of q modulo the
              product of its gears (doubling law once per gear; 324 million residues checked,
              0 exceptions), and that product exceeds q^2 at every d >= 70 (21 of 22 covers),
              so a failure is not a density event: the residue vector determines q^2 as an
              integer and at most one q realises a given (cover, phase) pair. Optimal covers
              contain all of 11, 13, 17, 19, 23, 29, 31 from d = 385 and every gear takes at
              least two islands from d = 70 (0 exceptions); the optimal gear set is far from
              unique. The real machine's minimum cover exceeds K(d) at all 197 recorded
              failures (ratio 1.0 - 2.6, not monotone). First moment with exact rates:
              expected failures above 2,849 = 0.0012 (and it under-predicts the band [1000,
              3000) by 14x, an honest miss); the cover-side moment is the depth function's
              product, the parent's counting wall, stopped. WHY IT DOES NOT CLOSE, exactly: the
              class count per cover is exact and tiny (2^K over a product above q^2), but the
              number of covers is about 2.7^m, 10^54 at d = 1,120, against a class density of
              10^-30: vacuous by 10^24. Dead: pi(c sqrt d); the family identification; counting
              as the cause; the cover-side moment; the compulsory-prefix lever (failures sit at
              the 61st percentile of the small gears' own coverage). NEXT INTERACTION named by
              the prover: bound the number of covers a real machine can produce. MANAGER'S NOTE
              (2026-09-06): over q that count is the number of failing q itself, so as posed it
              is circular; the honest form of the open interaction is "why does the real
              phase vector (q^2 mod g, all squares) never realise one of the 10^54 covers", and
              nothing on the tree yet distinguishes the square vector from a random one in
              length (R2.a.i, percentile 0.53). Background ILP at d = 2,240 (bounds 22..32)
              still running when the branch closed; its result cannot change any statement.

            - R2.a.i.a.1.a.i. Does the real separation drive K(d)? (weak point W3 of the wall;
              research/proof/separation_drives_K.md). FACT; answer: no, and the island target
              has no slack. K_real = 6, 9, 14, 17, 20, 22 at d = 140 .. 1330 is the MODE of the
              random-separation distribution at all six arcs (189 draws, 239 ILP rows all
              certified; percentiles 0.50, 0.46, 0.75, 0.48, 0.50, 0.63); coherent separations
              c/r for r = 3, 5, 7 give the same K as the real one at every arc (all 20 at
              d = 1120). Mechanism, exact: two gears' four struck residues are a translate of
              {0, S_g, S_h, S_g + S_h} mod gh, so the mean pairwise overlap is exactly 4m/(gh)
              for every separation (72 checks, 0 exceptions); coherence is closed under CRT
              (r (S_g + S_h) = c mod gh, 32,490 checks, 0 exceptions) and bites only on the tail
              gears (real tooth distance 0.69 of the arc against 0.50 random, outside the whole
              random range, 180 draws), too few to move K. Toward the root: the island target
              K(d) > pi(sqrt(6d)) - 3 is met by exactly ONE gear at d = 560, 840, 1120, 1330
              and with EQUALITY at d = 140, 280 (measured c = 3.8, 4.9, 6.2, 6.0, 6.2, 7.1
              against 6 required); the cheapest pi(sqrt(6d)) - 3 gears leave 0, 2, 3, 3, 4, 4
              islands open. The pairwise overlap route is vacuous (needs 0.20-0.31 islands of
              overlap per pair against a CRT mean of 0.22-0.53, ratio flat 0.59-0.60, and the
              conversion to K loses a factor 1.5-2). VERDICT: W1 (overlap on islands) is DEAD
              for lack of slack; W3 is answered (typical); the live statement is W2, whole
              columns, which has the factor four - and for whole columns the adversary with one
              phase per gear over all primes to q IS the real machine's period, so W2 is the
              root F(y) < y^2/6 itself in covering language, not an easier statement.
            - R2.a.i.a.1.b. Squares are even (the owner's suggestion, 2026-09-06;
              research/proof/square_vector.md). Spawned by the obstruction at R2.a.i.a.1.a and
              P7. FACT; the reading is decided: OUTCOME C, the square structure is irrelevant.
              Real vectors q^2 mod g and independent locally-square vectors fail the island
              witness at 0.029653 against 0.029700 over 6.3 million vectors of each kind on 30
              machines (ratio 0.9984 +- 0.0033), and every derived statistic agrees to within
              3% (open-island mean and minimum, per-offset opening profile, walk length 14.562
              against 14.561, minimum blocking set within 0.4 gears). Index parity is worth 1.1%
              pooled with an arc-dependent sign; one gear made square moves the rate 5-25% with
              signs that differ by gear and arc; squareness does not accumulate. Reading (b),
              reachability as a residue condition on g modulo 24 i, is exact (moduli 280,
              39480, 4920 at i = 12, 47, 82; exactly a quarter barred; 0 disagreements to
              200,000) and already spent: it defines the island set and does nothing more. The
              one global-integer effect is the sifting level, not the squares: the exact
              phase-vector model reproduces a real integer's opening count to 0.03% at s = 3.2
              but is 26% high at s = 2, the object's own configuration, and there the open
              islands are the twin pairs above q^2 (Hardy-Littlewood over real 1.0021 at
              q = 50,000; model over real 1.2628 against the classical 4 e^-2 gamma = 1.2619).
              That correction repairs the parent's first-moment miss: 9.90 predicted failures
              below q = 6000 against 17 observed becomes 16.51 against 17; the parent's blame on
              island correlation was wrong in sign (correlation makes the model over-predict).
              And it points the wrong way for a proof: the real machine has a fifth fewer
              openings than any phase-vector model. Random vectors also stop failing: failure
              rate 1.3e-1, 5.4e-2, 2.4e-2, 4.0e-3, 7.5e-4, 1.2e-4, 3.7e-5, 1.3e-5, 3.3e-7 at
              d = 60 .. 1100; at d = 954 (one arc past the last real failure) free 1.47e-5,
              locally square 1.27e-5, real 1.06e-5. The owner's sharp test, run to the end: 82
              explicit failing locally-square vectors at d = 954 (covers of 33-49 gears, moduli
              10^66 - 10^100 times q^2), 0 of 82 with a perfect-square CRT lift, the QR screen
              over outside gears decaying as 2^-t; control: 21 of 21 real failures have R = q^2
              exactly. It adds no factor: the QR screen is implied by the square condition,
              which is weaker than the range condition already used. VERDICT: the witness holds
              because covers are rare among ALL phase vectors at these arcs, and the real
              vectors are typical; nothing the phase vector is, as a square, prevents a cover.
              DEAD END recorded: the phase vector being a square. The proof obstruction is now
              purely: transfer from "rare among all vectors" to "never for real q", i.e.
              equidistribution of q^2 modulo products far above q^2, with the count of covers
              10^24 beyond the class density.

    - R2.b. Count gears, not columns (thin place 1; research/proof/gear_count.md). DEAD in its
      proven form, not a restatement, and it corrects the wall. THE INVERSION IS NOT THE F LADDER
      READ BACKWARDS: the best 4-gear sub-machine is {5, 7, 11, 17}, blocking span 16, against
      {5, 7, 11, 13}'s 11; 22 against 18 at K = 5; 28 against 25 at K = 6, exhaustive over every
      K-subset of the primes 5..149 (1.1 million subsets at K = 6); ratios 1.45, 1.22, 1.12,
      falling. So the adversarial covering statement quantifies over gear SETS as well as phases
      and is strictly stronger than F < y^2/6 (the wall's 5a corrected). MECHANISM, THE ARC: with
      {5, 7, 11} fixed, F({5, 7, 11, g}) = 11 for every prime g from 13 to 103 except g = 17, 19
      (short arc 6), where it is 16; the holes 5, 7, 11 leave are 6 apart and only a gear with
      arc 6 has that domino. PROVED, the complement of the umbrella bound: a gear with long arc
      >= S + 2 contributes to a stretch of span S at most two columns at distance exactly a_g, a
      bare domino whose size is invisible; with 3 a_g = g -+ 1, two gears share an arc iff they
      are a twin prime pair, so {5..q} carries only pi(q) - 2 - pi_2(q) distinct arcs and must
      buy both members of every twin pair: the machine's own twin gears make it a worse coverer
      than an adversary with the same number of gears (the exact minimum cover of the real
      window stretch is 1.5-2.1 times the free minimum for the same span; the adversary eats
      about a quarter of the slack: A(4, 5, 6) = 16, 22, 28 against windows 48, 60, 88, margin
      3.0, 2.7, 3.1 in span against the real machine's 4.4, 3.3, 3.5). The proven gear count
      saturates: the forced set is the whole machine for every span above about 2q/3
      (saturation spans 4, 6, ..., 20 at m7..m31 and 664 at q = 997 against windows to
      169,680, short by a factor q/4); forced is not needed (from span 10 at m17 on, the forced
      set exceeds the minimum cover; at m31 span 20 is covered by 5 gears while all 9 are
      forced). h(F) = n at all eight machines; the first sparable span below F is F - 3, -2, -4,
      -2, -4, -9, -8, -13 (m31, the gear spared is 23). The gear-count inequality
      S - 1 <= sum over forced gears of 2 ceil(S/g) + 2(n - f) is genuine and never binding
      (94 against 57 at the m31 record) because its first term is the counting bound. WHAT
      SURVIVES: one clean open lemma, A(K) < (p_{K+1}^2 - 1)/6 where A(K) is the longest span any
      K-gear sub-machine can block, with nothing on the tree bounding A(K) above; and one new
      handle, the ARC MULTISET, a which-residues property face A permits, in which the real
      machine is measurably worse than the adversary. Gates: the F ladder 2, 5, 7, 11, 18, 25,
      34, 43, 58 and 5d.ii's window covers reproduced exactly.

    - R2.b.i. The arc multiset (research/proof/arc_multiset.md). DEAD as a handle, with the
      reading reversed. A(K) exact over ALL primes: 2, 5, 7, 16, 22, 28, 37, 45, 68, 88, 101, 115
      at K = 1..12 (A(9) = 68 is 10 above the recorded bound), A(13) >= 137; the open lemma
      A(K) < (p_{K+1}^2 - 1)/6 holds at every K <= 12 with margin 2.7-3.8 and A/W flat
      (0.26-0.37): the adversarial form has nearly the root's own slack. A/F({5..p_K}) = 1.45,
      1.22, 1.12, 1.09, 1.05, 1.17, 1.00, 1.11, 1.12: at K = 10 the real machine {5..37} is an
      optimal blocker. Every optimum from K = 2 contains the twin pair 5, 7; the K = 7 optimum
      has three duplicated arcs; the optimum takes the smallest gears, never the smallest
      distinct arcs. DE-TWINNING LOWERS THE RECORD at every rung (F_real / F_detwinned = 1.10,
      1.29, 1.47, 1.70, 1.48, 1.66 at m13..m31) and mean F at fixed gear count is monotone
      INCREASING in the number of duplicated arcs (5,383 sets, no exception): a twin pair is the
      cheapest pair of small gears, so arc duplication helps blocking. Which arcs is worth
      nothing: relaxing every big gear to any two columns anywhere raises A(K) by 0 to 5
      columns (1.00-1.11). The domino-matching identity is exact (223 instances, 0
      disagreements); new proved tool, the type lemma (a gear with long arc >= L realises
      exactly {}, {i, i + a}, {i} for i < a or i >= L - a), and a MILP whose infeasibility
      certificate replaces a 40-minute search with half a second. Short-arc multiset fixes F at
      K = 3 (146 of 146) and fails from K = 4. The hole-distance mechanism is real one level
      up ({5..17} at L = 27 leaves two holes only at distances 5, 7 or 18; 18 is realised only
      by 53, and {5, 7, 11, 13, 17, 53} blocks 28 = A(6) while every other sixth gear gives 25
      or 26; A(3) = 7 has a one-paragraph proof of this shape). Record words at every rung
      reproduce the wall's B3; every letter is a_q' or b_q'; twin rungs are not special
      (increments inside the non-twin range). Residual, restated: bound h_S(L), the least holes
      k primes leave in a run of L; it alone fixes A(K) to 11%; a capacity statement with a
      gear count, face A's A2 again.
    - R2.a.i.a.1.c. The second moment over q (research/proof/second_moment.md; thin place 5).
      DEAD by proof: any bound strictly below 1 on the fraction of failing q for the certified
      object implies the twin prime conjecture (N-M6), and the first moment is a twin count in
      (q^2, (q + 1)^2) at s = 2, a lower-bound sieve, so the chain never starts. Measured
      B(X) = 11.4 (ln X)^2 / X, flat to 6% over a 64-fold range (0.51 .. 0.023 at X = 1k .. 64k):
      it vanishes as a measurement. Raw Chebyshev cannot work (Var / E^2 -> 0.145; 85% of the
      variance is the systematic run of the mean across a band). KEPT, exact: the gears
      coupling two islands at separation delta are exactly the prime factors above 7 of delta,
      3 delta - 1 and 3 delta + 1, overlap in {0, 2, 4} never odd (0 exceptions in 359,712,683
      cells; the same divisor form as Leg(v)); the joint density formula matches brute force
      to 1e-16 and the derived variance to 5e-12; the count of open islands is SUB-POISSON,
      variance over mean in [0.76, 0.81] at all 42 exactly computed q, with an exact mechanism
      (a generic gear costs -4/g^2, repaid 89-94% while g < m, exactly 50% at g = m); the
      one-class witness's last failure is q = 11,921 over all integers coprime to 30 (0
      exceptions in 30,955 machines above); the s = 2 handicap 4 e^-2 gamma confirmed to 2%
      on 33,868 machines; twin-prime products are not bad machines.

    - R2.c. The distortion method on the machine's covering problem (literature and
      construction; research/proof/distortion_method.md). OUTCOME (c) with one positive
      result. The method's engine (BBMST Theorem 3.1) carries no minimum-modulus hypothesis
      and applies to the machine unchanged; its core inequality reduces at the optimal
      parameter to a budget eta = sum of E[alpha_i^2] < 1, which over the period is
      sum 4/g^2 < 0.36455 for every set of primes >= 5, a covering budget that never
      saturates (unlike the capacity sum 2/g). But its conclusion is density only, and the
      exact CRT density beats its bound by 10^4 to 10^6 at q = 59..499. Localised to an
      interval it dies by the COLLAPSE LEMMA (one line, phase-free): once the product of the
      gears used exceeds the interval, each fibre holds one column, alpha is 0/1-valued, and
      the second moment collapses to the union bound; on the real window the localised budget
      is 1.07, 1.22, 1.47, 1.71 at q = 59..499, above 1 and diverging like 2 log log q, with
      collapse from the fifth or sixth gear; the shortest interval it can address is
      exp(theta(q^0.73)), worse than the sieve's q^4.27. What it needs to survive is a level-
      of-distribution input at dimension 2: the parity barrier from the covering side. THE
      CLAIMED POSITIVE (withdrawn by R2.d): the localised budget was said to prove A(K) <
      (p_{K+1}^2 - 1)/6 for K <= 10; the localised inequality is false ({5, 7, 11, 17} covers 15
      columns with eta = 0.693) and the tabulated eta_max is the union bound, vacuous from K = 4. Localising with average first
      moments is false, not weak (it would give A(7) <= 9.3 against the certified 37): any
      covering-side attempt must carry phase-adversarial first moments. Prior art: no theorem
      in the covering corpus has an interval in its conclusion; the fixed separation appears
      once (FKMPT Remark 7) as an aside; Stevens' H(r) <= 2 r^{2 + 2e log r} is the only printed
      interval bound of this shape and is 10 to 10^15 times the truth. Children named: gear
      ordering to minimise collapse; a second moment over ARITHMETIC BLOCKS instead of
      congruence fibres (the one crack in the collapse); the budget sum 4/g^2 < 0.36455 as a
      machine-free quantity to test against the tree's other reductions.
      - R2.c.i. The second moment over arithmetic blocks (research/proof/block_moment.md).
        DEAD as a route, with the sharpest single inequality on the tree left behind. The whole
        per-part algebra of the distortion engine is partition-agnostic; the fibre structure
        is used only to evaluate alpha = 2/g by CRT. With the whole window as one block and
        the real teeth the exact budget is eta_B = 0.358, 0.359, 0.365, 0.366, 0.365 at
        q = 59..997 (below 1 at all 107 machines to 599, max 0.367) against the fibre budget
        1.07..1.91: the collapse lemma is an artefact of the partition. But the a priori
        (phase-adversarial) block budget busts at the SIXTH GEAR (cumulative 0.16, 0.39, 0.57,
        0.76, 0.92, 1.08 at g = 5..19, independent of block length, interval and q), and the
        rescuing hypothesis "every block holds its fair share of survivors" is FALSE (it gives
        0.944 < 1 for {5, 7} on 4 columns, which the machine covers). The ceiling error is
        free; the conditioning costs a factor 7-8.5, and that is the whole mechanism. Blocks
        are worse than fibres from g = 13 for 86 consecutive primes; block thresholds are 3-13
        orders worse than fibre thresholds; ordering is an exponent for fibres and a constant
        for blocks. Unconditional reach: three to five gears, 0% of the window. WHAT SURVIVES:
        the budget in one coordinate, eta = sum over gears of (4/g^2) rho_g^2 with rho_g the
        strike-rate excess of g on the current survivors (the same quantity as 7b's curve),
        and the root's covering half follows if the weighted L2 mean of rho stays below 1.663;
        measured max 1.35 (q = 199), 1.52 (q = 997); rho = 1.000 to three decimals through
        g = 23 at q = 997, far past the provable g = 17. MANAGER'S CHECK (research/anchor235/
        r52/bm_trivial_tail.py): the head is exact (rho = 1) only for gears whose lower period
        times g fits the window (to 17 at q = 997), plus one gear with rho <= 2; the tail under
        the TRIVIAL bound rho_g <= 1 / delta_{<g} (every strike lands on a survivor) is 2.53
        against room 0.69 from g = 19 (ratio 3.7), 2.23 against 0.67 from g = 29 (3.3), 1.29
        against 0.64 from g = 97 (2.0): the trivial bound misses by a factor 2 to 3.7, the
        smallest miss of any unconditional bound on the tree, and the head cannot be extended
        because the lower period grows by a factor g per gear. The unproven part is exactly
        the in-window equidistribution of the lower machine's survivors modulo the middle
        gears (7b's curve, rho = 1 for ln g / ln Q' < 0.55, measured white), face A4 and D in a
        single weighted inequality. Child named: the shape of the rho_g profile (head, middle
        band, top gears), the three-band shape of the flank decomposition.

    - R2.d. The small-K theorem (docs/proofs/20-adversarial-lemma-small-K.md; working in
      research/proof/small_K_theorem.md). PROVED, new mathematics, bounded. THEOREM A: for every
      K <= 10, no K primes above 3, each striking two residue classes at its own separation
      3^-1 mod g with any phase, cover W(K) = (p_{K+1}^2 - 1)/6 consecutive columns (W = 8, 20,
      28, 48, 60, 88, 140, 160, 228, 280); certified at L = W(K) directly by infeasibility of an
      exact 0/1 program over the type-reduced item list (HiGHS; 53 to 34,099 binaries; 27 s in
      all), corroborated by a solver-free exhaustive search at K <= 6, a third search at K <= 5,
      agreement with round 50, and reproduction of the certified F ladder. THEOREM B: A(K) = 2,
      5, 7, 16, 22, 28 at K = 1..6 exactly, by reasoning (four lemmas: the arc law, the capacity
      bound, the SPAN LEMMA - a pair at distance t inside a run shorter than the gear forces
      g = 3t -+ 1 for even t and g = (3t -+ 1)/2 for odd t, so a distance names at most two
      primes - and the type lemma; A(1), A(2) pure reasoning; A(3) reasoning plus a hand table
      with the recorded one-paragraph proof's three gaps filled) down to a proved-complete
      finite case list and exhaustive enumeration at K = 4, 5, 6 (5, 18, 53 cases). New tool:
      the HEAD COLLISION (gears 5 and 7 cannot be simultaneously maximal and disjoint; deficit
      1, 1, 2 at L = 16, 22, 28), which kills the counting-tight case at every K >= 4.
      CORRECTIONS TO THE RECORD: the distortion lane's "positive" (its localised budget proves
      the lemma for K <= 10) is FALSE, not merely unproved: {5, 7, 11, 17} covers 15 columns
      with localised eta = 0.693, eight such instances at K = 2..9; the tabulated eta_max is
      not an upper bound on the localised second moment (the step alpha <= 2/m fails once a
      fibre is longer than the gear, and the code patched that regime with a lower bound);
      what eta_max is is the union bound on the collapsed gears, vacuous at every K >= 4. The
      K = 5 optimum is {5, 7, 11, 23, 29} (F = 22), not {5, 7, 11, 13, 17} (F = 18). Not
      proved: any induction step; these are the finite base of the open lemma, stronger than
      the root; the residual (a lower bound on the tiler function h_S(L)) is unchanged.
    - R2.c.ii. Fibres of a sub-machine (research/proof/submachine_fibres.md). DEAD as a route,
      with one unconditional theorem. The exact budget is monotone increasing in the cut at all
      27 cells (0.3655 at Q_s = 1 up to 1.686 at Q_s = 85,085 and 1.848 fully refined, q = 997),
      reproducing the block budget at one end and the fibre budget at the other. The per-fibre
      first moment is exact iff Q_s g <= L, so the admissible cut is tiny (Q_s = 5 from q = 23,
      35 from 199, 385 from 2297) and the freeze buys exactly one gear (proved). NEW THEOREM
      (SF-CAP): head gears kill whole fibres, so the survivor count inside a live fibre starts
      at the fibre length; thresholds 1.3e2, 4.2e4, 1.7e6, 1.7e10, 1e17, 1e25, 1e39 at q = 17 ..
      1999, better than the full refinement by 3 to 801 orders and than the one block (never
      finite), still exp(theta(q^0.60)), vacuous at every real window (closest q = 37 at
      1.0033). Cross-fibre obstruction named: the L2 discrepancy of a gear's strikes across the
      Q_s classes, exactly 0 over a period, 0.0002-0.0078 on the window; a single-phase
      adversary cannot be bad in all fibres at once (worst-phase budget 0.41 falling to 0.367);
      the gap to the input-free bound is 4.7x to 9.0x and growing. 7b's rigidity does not close
      it: it bounds a signed aggregate, the budget needs a mean square per class. Adversarial
      gate passed (validity at all 9 MILP witnesses), reach only K = 1. Correction to R2.c: its
      reach to K = 10 used the uniform-measure first moment, exceeded 1.8-3.6x on the covered
      witnesses; conditional on an equidistribution step. RESIDUAL, EXACTLY: periodic-set-in-a-
      short-interval (7b's kind, proved) for the 3-5 head gears with Q_{<g} g <= W; level of
      distribution for everything above, modulus Q_s g up to about W^{1/2}, every modulus and
      every class, sifted set, dimension 2.
    - R2.d.i. Collision laws for gear pairs (research/proof/collision_laws.md). PROVED in its
      parts, DEAD as an induction. Exact, 0 exceptions: the collision deficit is linear with
      slope 4/(gh), c(g, h; L + gh) = c(g, h; L) + 4 for every pair and every separation
      (248,334 real and 67,400 random instances; one-line proof); the SHARED-ARC LAW: gears with
      the same short arc a collide from L = a + 1 (14,340 configurations), so every twin pair
      collides at L = (g + 4)/3, the earliest possible (onsets 3, 5, 7, 11, 15, 21, 25 at (5, 7)
      .. (71, 73); below g at 7 of 7 twin pairs against 0 of 246 non-twin); the ARC FLOOR for the
      real separation: c = 0 for L up to max(a_g, a_h) (253 of 253 pairs; fails on 134 of 759
      random draws); the head collision is the a = 2 instance and c(5, 7; L) > 0 for every
      L >= 21. Real against random is signed and split: twin pairs at the earliest possible
      onset, non-twin pairs at 3.1 times the random median; the real separation is extreme in
      both directions. Triples: increment 4(g + h + k) - 8 per period, sub-additive by exactly
      8/(ghk) in rate. The record is maximal gear by gear (M = 0, 1, 0, 1, 2 at m11..m23); its
      whole price is overlap. The all-pairs bound is FALSE (refuted at K = 9, 10 by recorded
      covers); the valid form is a partition into blocks, L <= sum over blocks of the block's
      joint maximum. NEW POSITIVE, by reasoning: the block-2 (matching) bound proves the
      adversarial lemma at K = 4 (46 < 48 over all 1,093 four-sets, where counting gives 51),
      block-4 proves K = 5 (55 < 60) and K = 6 (87 < 88); block-4 fails from K = 7. NEW
      OBSTRUCTION, exact: the least block size whose joint maximum falls below the window is
      1, 2, 2, 3, 4, 5, 6, 7, 8, 8 at K = 3..12, about K - 3: no interaction law of bounded
      order reaches the lemma for all K; a new gear's net contribution (2/q')(1 - 2 sum 1/g)
      turns negative from {5, 7, 11, 13} on, which is exactly why the all-pairs form is
      invalid. The induction step being sought is not a pairwise law and cannot be made one.
  - **R4. The period-scale formulation (the owner's reframing, 2026-09-06; TO BE TAKEN UP AFTER
    THE WINDOW'S PROMISING LEADS ARE EXHAUSTED, per the owner).** Ignore the window.
    Machine {5..q} opens exactly prod(g - 2) twin slots per period (proved, CRT). A twin exists
    among them iff some opening survives every prime in (q, sqrt(6 P_q)], the SECOND MACHINE
    built from the gears above q. "Some opening of {5..q} in (q, P_q] survives the second
    machine" is weaker than the window statement and still gives twins infinite. At this
    scale every later gear g strikes the openings at exactly 2/g with error below 3^m
    (negligible against prod(g - 2) ~ e^q), and every product of later gears below P is exact
    too: a level of distribution of essentially 1. So faces B (position), D (transfer) and E
    (over-asking) of the wall vanish; face A (the dimension-2 sieve limit 4.27 against
    s = 2) stands alone, in its purest form (Brun's almost-primes, Chen's theorem are the
    known reach). MACHINE-NATIVE STRUCTURE at this scale: a later gear g strikes an opening
    iff the cofactor m = (6k +- 1)/g is q-rough with g m -+ 2 q-rough, i.e. the strikes of g
    on the openings of {5..q} ARE the openings of the coherent twisted machine with
    separation 2 g^-1 mod h at every gear h <= q (W3's family c/r with r = g), each with
    exactly prod(h - 2) openings per period; the second machine's action decomposes into
    coherent copies of the first machine, one per later gear. OPEN.
    THE ZONES (owner, 2026-09-06, made exact by the manager). On a range of K columns the top
    gears split by how they act: REPEATING gears g <= sqrt(6K) turn past their own square inside
    the range and have exclusive kills (the square gate); NON-REPEATING gears sqrt(6K) < g <=
    6K/q' kill bottom-open columns only at members g m with m q-rough and m < g, so every such
    kill coincides with a smaller top gear's (redundancy lemma, one line: the smallest prime
    factor of m is a top gear below g striking the same column); SILENT gears g > 6K/q' strike
    only their own home column. So the effective top machine on the range is exactly
    {q'..sqrt(6K)}, a gear machine of the same construction with no anchor. Mixed composites
    (bottom times top) never touch a bottom-open column; pure-top composites with cofactor
    below the gear are the redundant zone. THE BOTTOM IS EXACT AGAINST THE TOP from about
    3^m g / delta columns on (the inclusion-exclusion error 3^m is uniform in the interval
    length), i.e. from about e^{1.1 q / ln q}: far above the window (q^2), far below the
    primorial (e^q); products of top gears are exact while below the range over 3^m (level of
    distribution 1 in the zone). Read for a fixed window y^2/6: split the machine at
    q ~ ln y ln ln y and the bottom is rigid inside the window against every larger gear (the
    anchor rigidity of 7b generalised), the top (q, y] has level 1 on the bottom's openings, and
    what remains is the sifting range s = 2 (face A in two-machine language). First things to
    formalise when the line opens: the redundancy lemma and the rigidity generalisation.
    CONSTRUCTION RULE FOR THE TOP MACHINE (owner, 2026-09-06): study it on its own, not in
    relation to the bottom's kills (a clutch interaction) and not in the bottom's coordinate.
    The column (6k - 1, 6k + 1) is the bottom's anchor folded into the ruler, so the top machine
    is written on the RAW LINE: gears = primes in (q, Q], each striking its multiples; its twin
    object is a pair (n, n + 2), so in pair coordinates each gear has teeth at 0 and -2 mod g,
    separation 2 (the bottom's "one third" is this separation seen through the 6-fold: a
    clutch fact). Its own anchor is the wheel of its smallest gears q' q'' q''' with
    (q' - 2)(q'' - 2)(q''' - 2) twin slots per turn; the pair (-1, +1) straddling every wheel
    multiple is open for every gear (its column 0); the reflection n -> -n - 2 is its mirror.
    Its interaction laws for pairs and n-tuples of gears carry the chain and merge laws with
    letters {2, g - 2}. Whether a set of its lowest gears structures left/right slots as 2, 3,
    5 do is to be asked of the wheel in its own residues. Only afterwards the clutch: the
    6-fold map of the top's raw-line pattern into columns, and the bottom's kills of the top's
    openings. Do not poison the construction by comparing with the bottom; take inspiration
    only.
    OPENING TASK when this line starts (owner, 2026-09-06: the coupling will draw off previous
    findings; the framework shapes the research): re-file the record by object. Motor facts
    (records, spectrum, chain and merge laws, the gear-5 lock, the corridor); wheel facts (the
    top machine alone: mirror about the origin, in-range density, closed-run records, the
    placement residue law); clutch facts (the layer law and square gate, the walk from q^2 and
    the top gear's single strike, the near-twins at most three per rung, the island witness,
    the window's longest stretch as the largest twin gap, each gear's in-window take on one
    curve, the both-open cell carrying all the coupling).
    THE ANALOGY (owner, 2026-09-06): the bottom machine is the MOTOR, understood in depth; the
    top machine is the WHEELS, until now inferred only from the motor's odd behaviour (like
    diagnosing a car that will not move by looking only at the engine and theorising a gearbox);
    the CLUTCH engages the two. Isolate each object, learn its rules separately, then study the
    clutch. Order of work once the window's leads are exhausted: the top machine alone, then the
    clutch.
    REFINEMENT (owner, 2026-09-06): the TOP machine is independent of the bottom - the primes
    above q up to the range's edge, teeth +-6^-1, every gear starting at column 0, no exemption,
    its own openings, kills and runs (its period is far beyond the range, so its in-range
    pattern is not periodic: identify and formalise it first). THE CLUTCH is the interaction
    layer: every column classified by (bottom state, top state) - both open = twin; bottom open
    and top closed = a candidate killed by a top prime (own prime on its home column, or a proper
    factor); bottom closed and top open; both closed = mixed composites. The clutch's own
    patterns (joint runs, correlations, what the shared origin at column 0 forces) are where the
    owner expects the solution space to live.
    LINE OPENED 2026-09-06 after the window round R2.e (location pinpointed at the bottom, not
    closed). Running: R4.b the top machine on its own terms (research/proof/top_machine_1.md;
    scripts research/topmachine/r1/) under the construction rule; and the re-filing of the
    record by object (research/proof/refiled_by_object.md).
    - R4.b. The top machine on its own terms (research/proof/top_machine_1.md; scripts
      research/topmachine/r1/). STRONG: 21 laws, the top machine is a DOMINO MACHINE. L3 the
      partner law (proved): a gear's strike never comes alone, its partner is exactly 2 away, so
      each gear's struck set is a disjoint union of dominoes {x, x + 2} (0 exceptions in 5.6
      million struck residues, 8 wheels); everything else follows. L4 the forbidden gap: a gap
      of exactly 4 between open pairs is impossible in any top machine (0 in 12 wheels and 27
      range machines to 10^7). L2/L10: every gear's slots form arcs (g - 3, 1), the short arc
      collapsed to the single shield n = -1; longest run of open pairs = q' - 3; longest step-2
      chain = q' - 2 (12 of 12, attained). L17 THE PARITY LAW (new, proved): if every gear
      exceeds 2m + 1 then F_top = 2m - (m mod 2): the record is decided by the parity of the
      gear count, not by the gears' sizes (170 cases, 0 exceptions; increments alternate +3,
      +1). L16: F_top is exactly the largest L for which [0, L) can be tiled by the gears'
      letters {2, g - 2}, one phase per gear (15 of 15 against full-period scans). L18: record
      multiplicity per wheel universal in m alone (18, 24, 480, 720 for m = 3..6; 14 wheels).
      L11: the run spectrum is the second difference of prod(g - 2 - L), an arithmetic
      progression of common difference exactly 6 for every three-gear wheel. L19 THE
      CONJUGACY: n -> 6^-1 (n + 1) carries the top machine's open-pair set exactly onto the same
      gears in the bottom's column coordinate (0 mismatches, 2.2 million residues), so counting
      and symmetry laws are common property and the METRIC laws (arcs, runs, records) are the
      top machine's own. Transferring unchanged: the mirror n -> -n - 2 (fixed point the
      shield), the symmetry group (Z/2)^m with adjacency group Z/2, even gap counts except
      length 1, the chain, merge and alternation laws (0 exceptions in 118,341 / 34,646 / 816),
      prod(g - 2) and prod(g - 4). New: L6 the origin clump of 2(q' - 3) + 1 forced slots; L15
      prod(g - 3) member-sharing dominoes; L20 no fold at all (flat mod 2, 3, 6). L21 the gear
      zone: on a range, the pair n <= Z is open iff both members are q-smooth, so the record
      always sits just above the origin clump, below Z (18 of 18); in-use density exceeds the
      CRT product by 5-17%; a fixed gear set is flat to 4-5 figures on a range 10^4 times below
      its wheel. REFUTED: "the top gears make the record" (the smallest gears do the work,
      2L/g strikes each); "the first stretch above 0 is the most open" (for the in-use machine
      it is the least open); the budget holds with vast slack (max increment 7 over 69 exact
      ladder steps to gear 97). Open coincidence: the counts of gap 3 and gap 5 are equal in
      every wheel whose gears all exceed 7 (9 wheels), unequal iff 7 is a gear.
    - R4.b.i. The top machine's laws in Lean (proofs/TopMachine.lean, proofs/TopMachineWheel.lean;
      ledger research/proof/top_machine_lean.md). KERNEL: 59 declarations, zero sorries, no
      native_decide, standard axioms or fewer (manager gate: lake build TopMachine
      TopMachineWheel green at 1001 jobs; audit of no_gap_four, parity_upper, conjugacy,
      wheel_count, merge_law, chain_law: propext, Classical.choice, Quot.sound or less).
      PROVED: L1 (teeth and count), L2 (arcs and the shield), L3 (the partner law and the
      domino form, no hypothesis), L4 (the forbidden gap, in the stronger form: n and n + 4
      open implies n + 2 open), L5 (the wheel count by CRT), L6 (shield, antipodes, the origin
      clump), L7 (mirror and its fixed point), L8 (sufficiency, per-gear necessity, adjacency),
      L10 (run bound and attainment; step-2 chain bound and attainment), L12 (chain law, iff,
      no hypothesis), L13 (merge law, no hypothesis), L17 upper bound (L + (m mod 2) <= 2m
      for odd gears above 2m + 1), L19 (the conjugacy to the column coordinate, with existence
      and the Census form). CLOSED IN ROUND 33 (proofs/TopMachineCrt.lean, 29 declarations):
      the Finset-indexed CRT lemma exists_crt / crt_unique (Finset induction and Bezout); L17's
      attainment parity_attained by the explicit tiling (ceil(m/2) gears on the even dominoes,
      floor(m/2) on the odd), hence the equality parity_law as IsGreatest; L8 the exact group:
      affine_group (necessity, gears prime and >= 5, the only use of primality in the library),
      exists_symmetry (every sign vector realised), sign_count (exactly 2^m residues mod W).
      Manager gate: lake build TopMachine TopMachineWheel TopMachineCrt green at 1392 jobs;
      audit of the seven theorems: propext, Classical.choice, Quot.sound; zero sorries. 88
      declarations in all. Before round 33 nothing assumed primality, only size, oddness and
      coprimality; affine_group now does, for composite gears sharing a factor with c. The existing MergeLaw / TwoTeeth infrastructure did not transfer with
      d = 2 (teeth symmetric about 0 there, the offset pair {0, -2} here); L12 and L13 were
      proved directly.
    - R4.b.ii. The wheels, second pass (research/proof/top_machine_2.md; scripts
      research/topmachine/r2/). STRONG: laws L22-L38, both open facts of the first pass closed.
      L22 THE GAP CENSUS LAW: the number of consecutive open pairs at distance d per wheel is
      N_d = sum over S in [1, d-1] of (-1)^|S| prod_g (g - |E_g(S)|), E_g(S) = {0, -2, -d, -d-2}
      u {-j, -j-2 : j in S} mod g; exact by CRT, 0 mismatches over 15 wheels at every d and over
      the whole 6.7 x 10^9 period of {7..31} for d <= 16. L24 (W1 closed): gaps 3 and 5 share
      the polynomial prod(g-4) - 2 prod(g-5) + prod(g-6) because both have four forbidden
      classes per gear and two "some gear here" requirements; the gap-3 classes collapse only
      for g | 15 (never a gear), the gap-5 classes for g | 7, so 7 is the unique separating
      gear (16 of 16; the only coincident pair for d <= 16). L25: N_d = sum_k (-1)^k sigma_{m-k}
      M_k(d), gear-independent exactly when the vanishing-moment count r(d) reaches m, value
      (-1)^m M_m: L18's universal multiplicities (18; 96, 24, 24; 480; 6480, 1440, 720) are a
      corollary. L26: r(d) is the parity covering number for every d <= 16 except d = 4, so
      F_top = max{d : r(d) <= m} - 1 reproduces L17 from the census, and d = 4 is the unique
      place the gap's closed-boundary cover differs from the record's free-boundary cover (L4
      explained). L28/L29 tuples: joint census = CRT product, deviation 0 (14 triples and
      quadruples); 2^m all-struck classes, exactly two (n = 0, -2) where all m dominoes
      coincide: THE ORIGIN IS THE MACHINE'S UNIQUE TOTAL COLLISION and the origin clump is its
      shadow; no three distinct traces pairwise overlap (0 of 1,354), so the record cover is a
      perfect tiling for even m and wastes exactly one unit for odd m (11 of 11). L30
      (exhaustive, all 1,540 triples and 7,315 quadruples of odd primes 7..97): F_top = 5 or 6
      and 8 or 9, decided by one bit, whether 7 is a gear (0 exceptions); mechanism: the long
      letter g - 2 is odd and the only parity-crossing piece, shown only by a gear with
      g <= L + 1. L31: F_top depends only on m and on the gears <= F_top + 1; large gears enter
      by their number only (90 cases, 0 exceptions; pre-registered threshold 2m + 3 REFUTED as
      written, 12 of 70). W2 REPLACED: L32 F_range(N) = max{d : W/c(d) <= N} - 1 within 1 unit
      at 19 of 21 checkpoints; the wheel record IS reached, at 10.9% of the period for {7..31},
      0.037% for {13..41}, 0.005% for {19..47}; the first pass's ceiling was an artefact of
      stopping at 10^7. L33: the 8 record blocks of {7..31} form four mirror pairs summing to
      W - 33 and occupy two residues mod 1001, one mod 11, one mod 17: the record's position is
      pinned by its small gears. L34-L36 THE ANCHOR QUESTION, answered exactly and negatively:
      an anchoring gear needs g - 2 <= 2, i.e. g <= 4; 2 qualifies in the bottom only because
      it is the unique prime dividing the separation, collapsing its teeth; top gears have
      g - 2 >= 5, the corridor has density >= 0.58, is uniformly filled (5 of 5) and palindromic
      from the shield (6 of 6): no fold, no direction. Two independent anchors instead: q'
      fixes the local metric (run q' - 3, chain q' - 2, clump 2(q' - 3) + 1; 14 of 14) and m
      alone fixes the record. L37/L38 THE REMOVAL LAW: raising the split divides W by q', the
      open count by q' - 2, the dominoes by q' - 4, grows the three ceilings and drops F_top by
      3 (m even) or 1 (m odd), nesting ratio exactly 1 - 2/q'; sharp form: F_top(G minus g) is
      the same whichever gear leaves (9 of 9 in the large-gear regime, failing exactly outside
      it). OPEN: L22 in the kernel (exists_crt now available, R4.b.i); the vanishing of M_k(d)
      for k < r(d) (verified to d = 16, unproved); L31 as a formula. See R4.b.iii.
    - R4.b.iii. The walk and the transforms of the top machine (owner: closed forms and proofs
      for locating the next opening; research/proof/top_machine_3.md, laws numbered L30-L45
      there, a numbering clash with R4.b.ii's L30-L38: cite by document). STRONG, CLOSED FORM
      FOUND. THE NEXT OPEN PAIR (L30 there): L(x) = mex{ (-x) mod g, (-x - 2) mod g : g in G },
      the next open pair after x is x + L(x); hypothesis every gear > 2m; proved (two case
      splits); 0 mismatches over 1,448,287 positions in 9 wheels; SHARP: {7, 11, 13, 17} fails
      at 36 positions (7 < 8). O(m) operations, no scan, no period. THE NEXT TWIN CANDIDATE
      (single-number view, runs of three; L34/L35): R(x) = mex{ (-x), (-x - 1), (-x - 2) mod
      g }, hypothesis every gear > 3m; the record of the triple machine is exactly 3m when every
      gear >= 3m + 3 (proved; 7 wheels, m = 3, 4, 5 give 9, 12, 15, 0 exceptions). Why 3m and
      not 2m - (m mod 2): the pair piece {x, x + 2} is a gapped domino confined to one parity
      class and cannot tile; the triple piece {x, x + 1, x + 2} is solid and tiles exactly; the
      parity defect belongs to the separation, not the tooth count. The location bound
      re-proved from the mex (odd gears > 2m + 1 give L(x) <= 2m - (m mod 2), the listed
      numbers fall into <= m same-parity pairs). IN USE (L32/L33): the same form with each
      residue replaced by its arithmetic progression is exact on 24,000 walks (0 mismatches;
      160-443 gears; N to 10^7), typical walk median 2-10, 99th percentile 19-66, against
      records 71-3,006; the proved covering bound L <= 2m/(1 - 2 H_S) is non-vacuous only while
      sum_{q < g <= L} 1/g < 1/2 (true at q = 17, 19 for N = 10^6: bounds 4,289 and 753;
      vacuous at 10^7): pre-registration partly REFUTED here; the in-use bound is the open
      item. THE LAYERED WALK COLLAPSES (L38): if g > F_G + 3 the hop chain is at most 2, and a
      double hop occurs iff the landing is on tooth -2 and the lower gap is exactly 2, a
      one-line non-recursive layer (12 layers, 391,048 positions, 0 exceptions; smallest gear
      first is the right order, adding 7 last gives chains of 3); the nested form exact on
      23,432 walks at fewer than one hop per walk. DISTRIBUTION (L36/L37): with C(j) the
      all-struck window count, #{L = j} = C(j) - C(j + 1), N_d = C(d - 1) - 2 C(d) + C(d + 1),
      F_top = max{j : C(j) > 0}, mean walk = (1/W) sum C(j), the exact dual of L11; C(j) is an
      alternating sum of shifted wheel products with path-convolution coefficients and provably
      NOT a product. SPECTRAL (L40/L41): per-gear factor -(1 + omega^{2a})/g, which in the
      shield coordinate n + 1 is the real -(2/g) cos(2 pi a/g): the top machine is the u = 1
      machine, the bottom's fold replaced by one translation; the transform is nonzero at all
      32,077 frequencies and so are the run indicators (their kernel zeros unreachable since
      L + 2 < q'); the spectrum decides product questions (run ceiling q' - 3) and CANNOT decide
      F_top. BITWISE (L42/L43): striker-parity bias exactly prod(g - 4), the domino-count
      polynomial; {XOR = 1} is a subset of {blocked}, so XOR bounds the record from BELOW, and
      the bound is tight in all 10 wheels at both parities because a record block covered
      exactly once always exists (odd-m waste is an edge singleton, not a double cover):
      pre-registration refuted for odd m into a stronger law. CHARACTERS (L44/L45): the
      pair-correlation B(d) = prod c_g(d), c_g = g - 2, g - 3, g - 4 for d = 0, +-2, else (400
      values, 0 mismatches); never vanishes, so holes exist only in the consecutive census: {4}
      in the pair view, {2, 3} in the triple view (both proved). KERNEL NEXT, cheapest first:
      the mex closed forms L30 and L34 (no CRT needed), the triple-hole proof, B(d) via
      card_filter_crt, the triple record's upper bound (shape of parity_upper), the
      distribution identities, the XOR lower bound.
    - R4.b.iv. The in-use next-opening bound (spawned by R4.b.iii; research/proof/
      top_machine_4.md, laws L46-L56; scripts research/topmachine/r4/). STRONG as a law of the
      in-use machine, and it says the in-use record is a DIFFERENT OBJECT from the wheel
      record. THE ZONE LAW (proved in two lines, 0 exceptions in 130,230 cells over 36
      machines): for gears (q, Q] on [1, N], a pair n <= Q - 2 is open iff n and n + 2 are
      both q-smooth (any larger prime factor is itself a gear). So the range record is the
      largest gap of the finite q-smooth-pair list below Q, value AND position exact (the
      block starts one cell above the gap's lower end; 161 at q = 5 for every N), and for Q
      beyond the largest smooth pair s(q), F_range >= sqrt(N) - s(q) - 2 (s(5) = 160,
      s(7) = 8,748, s(11) = 19,600, s(13) = 246,400, s(17) = 672,280, s(19) = s(23) =
      23,718,420, s(29) = 354,365,440, s(37) = 9,447,152,317; prior art in a line: Stormer
      1897 / Lehmer 1964 for the finiteness). Ratio truth to bound 1.000-2.000, exactly 1.000
      at 17 of 36 machines (q = 5, N = 10^8: bound 9,838, truth 9,846). CONSEQUENCES: no bound
      in (q', m) exists (F_range >= (m log m)/2; F/2m = 1.5, 2.6, 3.4, 4.0 at N = 10^5..10^8,
      still climbing); in use THE TAIL IS EMPTY (certified covers give F_top >= Q, 3 to 43
      times 2m, in 26 of 27 machines), so the parity and covering apparatus is a statement
      about tail gears only; the covering bound 2m/(1 - 2 H_S) is alive iff F < exp exp(1/2 +
      sum_{p <= q} 1/p - M) (35 of 36) and 40-50x loose where alive; union bounds cannot
      reach the truth at all (a covering pattern pins x modulo the product of its gears, so
      counting stops at 2 log N / log q' = 10-19 against records 113-9,846). NO SATURATION
      (expectation refuted): the gear zone is [1, Q], not [1, sqrt N], so at fixed N = 10^7 the
      record is linear in the largest gear (186 ... 999,876 as Q goes 316 to 10^6), and on a
      fixed region strictly above every zone the record still climbs (200 to 1,511). ABOVE THE
      ZONE the machine is a different size: record A(q, N) = 24 to 419 over the whole tested
      range, 23x below the zone record at q = 5, fitting a first-hit model within 0.97-2.38;
      A wins in 10 of 36 machines (all q >= 13, N <= 10^6), each q with one crossover N after
      which the zone wins for good. WHEEL RECORD, exact core/tail rule: F_top = max{L : the
      minimum over core phases of the domino cost D(U) <= t} with D the sum of ceil(run/2)
      over step-2 runs per parity class (0 mismatches on 13 known records, 89 sets decided);
      the additive form F_core + 2t - defect holds at exactly the 64 empty-core sets (the
      parity law) and fails at all 25 with a core. REFUTED pre-registrations: s(7) = 448; the
      tail is nonempty in use; the record sits in the first stretch; A <= 400; the record
      starts at s(q) + 1. OPEN: a bound above the zone (A(q, N)); the first-hit model as a
      law.
    - R4.b.v. Sliding the split down (owner: which top-machine laws survive with gears 2, 3, 5,
      7, 11 present; the smallest top machine that retains the simplicity; by the conjugacy a
      top machine starting at 5 IS the bottom machine {5..q}, so this measures what transfers).
      DONE (research/proof/top_machine_5.md, laws L50-L59; scripts research/topmachine/r5/).
      THE SMALLEST SIMPLE MACHINE IS q' = 5, for everything structural and for the conjugacy
      at once: gear 5 is the first gapped domino and the first gear set where 6 is
      invertible, so "the domino machine begins" and "this is the bottom machine {5..q}" are
      the same event. Thresholds, exact: (a) the mex closed form's true hypothesis is
      F_top(G) < q', not q' > 2m (L50, one-line proof; {6, 11, 13} separates them: q' = 2m = 6,
      F = 5, 0 failures in 858 positions); failure rate as q' crosses: 30-70% at q' = 2,
      23-45% at 3, 2.9-6.3% at 5, 0-0.65% at 7, 0 from 11. (b) the parity law's sharp
      threshold is q' >= 2m + 1 for even m and q' >= 2m + 3 for odd m (7 boundary pairs,
      m = 2..8, 0 exceptions, probed with coprime composites 9, 15, 21, 25); document 1's
      q' > 2m + 1 is not necessary at even m. (c) record = free-domino tiling is exactly
      equivalent to (b) (28 of 28; pre-registered q' > F + 1 REFUTED). (d) the symmetry group
      is absolute from q' >= 3, order 2^(number of odd gears); gear 2 costs one factor of 2
      because +1 = -1 mod 2. (e) L31 (the record depends on m and the gears <= F + 1 only)
      is ABSOLUTE, holding at every q' >= 2 (27 families, 117 gear sets, 0 disagreements, down
      to {2, 3, 5}): the one record law that survives all the way down. TRANSITIONS: gear 3
      gives teeth {0, 1}, a solid domino crossing parity, no two open pairs adjacent, the line
      folded mod 3, the twin-candidate view empty, gap 4 still forbidden; gear 2 gives one
      tooth, two mirror fixed points, the group halved; 2 and 3 together leave exactly n = 5
      mod 6 open: the anchor recovered as a two-gear top machine. TRANSFERS TO {5..q}
      UNCHANGED: wheel count, mirror, group, the census law L22, the degree law, the joint
      census, the correlation product, the C-identities, full spectral support, the covering
      record, the sub-threshold reduction. TRANSFERS MODIFIED (2 becomes 2 u_g): arcs
      (g - 2u_g - 1, 2u_g - 1), letters {2u_g, g - 2u_g}, the run ceiling = gear 5's long arc
      = 2 for every {5..q}, the correlation coefficient at +-2u_g, the hop collapse (chain
      still <= 2; the double-hop rule becomes "lower gap = forward letter of the landing's
      tooth", 10,860 hits, 0 exceptions). FAIL WITH NO ANALOGUE: the partner law, the
      forbidden gap 4 ({5, 7} still forbids it; {5, 7, 11} has no hole at all), the origin
      clump, the parity law and L26 / L29, the gear zone. Two exact statements about the
      bottom: L56 the anchor rescaling F(G u {2, 3}) = 6 F_col(G) + 5, F(G u {3}) = 3 F_3 + 2,
      F(G u {2}) = 2 F_2 + 1 (30 of 30; manager's note: this is the fold as a coordinate
      identity, a run of F columns is 6F + 5 raw pairs between openings at n = 5 mod 6, so it
      is exact and known, kept as FACT); L57 the bottom machine's next-opening formula
      M_B(x) = mex of the union over gears of {(+-u_g - x) mod g + kg <= B}, self-certified
      exact whenever M_B <= B (890,501 certified walks, 0 mismatches, {5..q}, q <= 31, 4-70
      numbers per call; the uncorrected two-per-gear form is wrong at 12% of positions for
      {5..31}; manager's note: exact, but a bounded scan of the residue progressions in
      disguise; its content beyond the sieve is the certificate).
    - R4.b.vii. The zone of tranquillity (owner: past Q - sqrt Q, or the first gear whose
      square exceeds Q, a zone that should be completely knowable, where the top machine's gap
      alignments can be located and sized; manager's reading to test: in (Q, Q^2] a number is
      open iff it is q-smooth times at most one prime above Q). DONE (research/proof/
      top_machine_6.md, laws L57-L66; scripts research/topmachine/r6/). THE QUIET ZONE IS
      (Q, Q^2] AND ITS RULE STARTS AT 1 (L57, two-line proof, 45 machines, every cell of
      [1, Q^2], 0 exceptions): n <= Q^2 is unstruck iff n = s P with s q-smooth and P = 1 or
      one prime above Q; the smooth-zone rule is the case P = 1; no transition band; the rule
      dies at p_1^2 (p_1 the first prime above Q), not Q^2. THE OWNER'S SECOND NUMBER IS RIGHT:
      the lower edge of the zone's own record region is g_0^2, the square of the first gear
      with g_0^2 > Q (2.0%, 1.3%, 0.5%, 0.2%, 0.05% above Q as Q grows to 10^7); Q - sqrt Q is
      not an edge of anything (at q = 5, Q = 10^4 it sits inside one struck block running from
      161 to 10,006). The zone is layered: at height x the smooth cofactor is at most x/p_1,
      attained in every stratum; in the bottom stratum (Q, 2Q] the only cofactor is 1, so
      unstruck = prime or q-smooth (L59). COMPLETE KNOWLEDGE, four ways, 0 mismatches: the
      enumeration from the rule with no sieve (13 machines); the count Psi(X, q) + sum over
      s <= X/p_1 of (pi(X/s) - pi(Q)); the family decomposition (every open pair labelled
      (s, s') solving s' P' - s P = 2; 1,510 families at q = 5, Q = 10^4; each family's count
      is INDEPENDENT of q, raising q adds families and changes none; family (1, 1) carries
      pi_2(10^8) - pi_2(10^4) = 440,107 exactly at q = 5 and q = 11); the walk as the mex in a
      new guise, nextadm(x) = min over q-smooth s of s * nextprime(max(Q, ceil(x/s))), 0
      mismatches in 11,489,920 positions, the next open pair by iterating it (600,000 walks,
      mean 2.3-3.5 iterations). ALIGNMENTS ALWAYS OCCUR, FLOOR PROVED: any prime gap in
      (Q, 2Q] free of q-smooth numbers is a run of struck pairs (48 of 48; truth 3.22-24.00
      times the floor). The record's profile through the zone is a U (371, 419, ..., 116, ...,
      311 at q = 5, Q = 10^4): the bottom is family-starved but short, the top has every
      family but twice the log and is 1,800x longer; the bottom wins 28 of 48, narrowly. HOW
      BIG: 183-419 at Q = 10^4, at 1.35 Q to 2.63 Q, about 2/Q of the zone, growing with Q
      but not monotonically, 23.5x below the smooth zone's linear record at Q = 10^4 (3.0x at
      Q = 316); this object IS R4.b.iv's A(q, N), eight values and positions reproduced
      exactly. NO UPPER BOUND, AND THE REASON IS EXACT (L65): the zone's bottom stratum is
      the family (1, 1), the twin primes above Q, so bounding the zone record above is
      bounding the gaps between twin primes. Free confirmations: no gap of 4 in 49,433,381
      range gaps; W1 survives as an approximate equality. REFUTED pre-registrations: monotone
      density through the zone (unimodal, peak near Q^1.5); the record always at the bottom
      (20 of 48 elsewhere); the record block bounded by twin primes (13 of 42); nothing
      special at g_0^2 (refuted in the owner's favour).
    - R4.b.viii. The stack and the exhaust (owner, 2026-09-06: a third machine above the
      top, gears from Q to Q's primorial). KERNEL, round 35 (proofs/MachineStack.lean, 48
      declarations, docs/proofs/23-stack-and-exhaust.md): cut q 0 = q, cut q 1 = q#,
      cut q (k+2) = the product of the primes in (cut k, cut k+1]; tier, stack, Spans,
      Smooth. stride_containment: every gear of tier k+3 spans tier k+1 and strikes at most
      2 of its positions per stride (card_strikes_window_le_two), and the spanned tier's
      pattern repeats inside one stride (tier_pattern_repeats); not_spans_below: a gear of
      tier k+2 is at most tier k+1's period, equal only for a one-gear tier. THE EXHAUST CAP
      exhaust_home_or_echo: C < n <= C^2 and p > C dividing n gives n = p or a prime <= C
      divides n; kernel finding: primality of the exhaust gear is never used, any divisor
      above the cut behaves the same. open_iff_twin: open under all primes <= C on (C, C^2]
      iff twin prime; stack_eq_primesLE and stack_open_iff_twin under CutMono; unconditional
      first step by Bertrand: wheels_open_iff_twin (motor + wheels leave a pair open on
      (q#, (q#)^2] iff it is a twin prime, hypothesis 2 <= q). Zone laws: smooth_zone (L46),
      wheels_smooth_zone, quiet_zone (n <= Q^2 open iff n = s P, s q-smooth, P = 1 or a prime
      above Q). HONEST GAP: CutMono (cuts nondecreasing) is a prime-density statement, not
      stack arithmetic, and is FALSE at q = 2, 3 (at q = 3 the cuts run 3, 6, 5, 1: tier 3 is
      {5}, tier 4 empty); carried as a hypothesis beyond the first step. Manager gate: build
      of the five targets green at 2244 jobs; audit of stride_containment, not_spans_below,
      exhaust_home_or_echo, open_iff_twin, wheels_open_iff_twin, stack_open_iff_twin,
      smooth_zone, quiet_zone: propext, Classical.choice, Quot.sound; zero sorries. The
      library is 206 declarations. Earlier assessment kept below. FACT, not opened: every top-machine law is stated in (smallest gear, gear
      count) only, so the third machine is the top machine with split (Q, Q') and obeys the
      same laws (zone law below Q', smooth-times-one-prime in (Q', Q'^2], mex and parity only
      where Q > 2 m_3, which fails); the removal law L37/38 is this self-similarity one gear
      at a time. Stride containment: TRUE for the bottom (every third gear exceeds P, so it
      strikes at most 2 positions per bottom period, and the bottom's full twin-slot pattern
      sits inside every stride); FALSE for the middle (its period prod(q, Q] is far above Q,
      so only third gears above that period contain a full middle period, and those are
      silent on ranges below it). Below Q^2 the third machine's gears are all above the
      square root and their only role is striking the numbers smooth times one large prime:
      the zone rule of R4.b.vii from the other side. The owner's hope (bottom twins open to
      alignment) reduces to counting the sparse strikes on the bottom's prod(g - 2) slots per
      period: the counting face of the wall. OWNER'S CAP (2026-09-06), exact form: in (Q, Q^2]
      every third-gear strike is a home strike (n = P prime above Q, exempt) or a duplicate
      of a bottom or middle strike (n = s P with s > 1 having a factor <= Q), so an open pair
      of bottom + middle there is a twin prime, for every q, to infinity: nothing above Q
      touches the window (Q, Q^2] (every third gear non-repeating or silent there; the
      sieve-to-the-square-root fact in machine form). The stronger reading, a bottom period
      with no third-gear strikes at all, exists by CRT only at heights where gears beyond the
      avoided set are active: exposure there is not twin-ness. Consequence: the search is
      capped to bottom + in-use middle + their clutch on the zone (Q, Q^2] = R4.b.vii's zone
      of tranquillity; the difficulty sits in the middle machine's in-use regime (R4.b.iv: no
      bound yet). THE TOWER (owner): with machine k + 1 = the gears from the top of machine k
      to machine k's period, the lower cut of machine k + 2 is machine k's period, so every
      gear of machine k + 2 strides a full period of machine k, for every k, and machine k + 1
      never fits except at its silent top; the redundancy cap repeats at every level. It
      turns the tower into a ladder of windows at the primorial rungs q, q#, (q#)#, ..., each
      question "machines 1..k + 1 leave an open pair in (P_k, P_k^2]", the window statement at
      a sparse set of rungs; theorem (E) already says the effective machine is exact, so no
      new interactions, but the shape is fixed: known machines + one in-use machine (smallest
      gear = previous period) + clutch, on a zone of tranquillity; the missing instrument is
      the same at every rung, an in-use bound, found once and carried up. Reopen if R4.b.vii
      names the object governing alignments in (Q, Q^2]; the third machine is then the split
      to test its self-similarity.
    - R4.b.ix. Closing the wheels' open laws (owner's rule 2026-09-06: no clutch until motor,
      wheels and exhaust are fully understood; the objects ledger research/proof/
      objects_ledger.md is the gate). The loaded record rule (core/tail domino cost) to be
      proved, the free/loaded boundary as its corollary, the moment vanishing of L25, the
      kernel shape of the census law L22, and the complete list of the wheels' open items.
      STRONG, all four items closed (research/proof/top_machine_7.md, laws L67-L75; scripts
      research/topmachine/r7/). THE LOADED RECORD RULE, PROVED both directions with no side
      hypothesis (L69): [0, L) is coverable iff some phasing of the gears <= L + 1 leaves a
      residue set of domino cost <= t(L) = #{g > L + 1}, so F_top(G) = max{L : min over U of
      D_L(U) <= t(L)}: the record's dependence on m and the gears below F + 1 (document 2's
      L31) is now a formula. Rests on the piece law L67 (a gear can join the two ends of the
      window iff g <= L + 1, which is exactly why the tail is g > L + 1) and the matching
      lemma L68 (the domino cost is the exact minimum number of pieces; dominoes never cross
      parity). 0 mismatches against full-period scans on 6,659 pairwise-coprime odd gear sets
      (5,006 loaded), 0 of 13 known records, 0 of 8,855 triples and quadruples; the wrong
      boundary core = {g <= L} fails at 605 sets, always by exactly 1 cell. THE BOUNDARY IS A
      COROLLARY: empty core gives D(L) = 2 floor(L/4) + min(L mod 4, 2) and max{L : D(L) <= m}
      = 2m - (m mod 2), the parity law derived (0 mismatches, m = 1..200); the sharp threshold
      2m + 1 (even m) / 2m + 3 (odd m): at q' = 2m + 1 the one core gear offers one
      ends-joining piece, at even m leaving runs of m and m - 1 (cost m > t), at odd m two
      runs of m - 1, even, tiling exactly (cost m - 1 = t): one parity bit is the whole + 2.
      The record cover is a free tiling iff D(F) <= m iff the parity law holds (settles the
      refuted guess q' > F + 1). L70 a closed-form upper bound F_top <= Lcap, 0 violations,
      exact on all free wheels tested. THE MOMENT VANISHING, PROVED (L73): M_k(d) is
      (-1)^(d-1) times the top multilinear coefficient of f^k, and a term of f^k touches all
      d - 1 variables only if k pieces {p - 2, p} cover [1, d - 1]; below the covering number
      every term misses a variable. L74: the covering number is the record's own cost
      function, r(d) = D(d - 1) (the two excluded pieces are singletons and singletons never
      help); verified exactly to d = 26, 0 failures, the eight published multiplicities (18;
      96, 24, 24; 480; 6480, 1440, 720) reproduced. d = 4 is no longer an exception: the only
      two pieces covering interior position 2 are the two the census excludes, so r(4) is
      infinite and M_k(4) = 0 for every k: the forbidden gap is the degenerate case of the
      same covering statement. L22's kernel shape written as three separable lemmas (local
      characterisation; inclusion-exclusion over interior positions; CRT product per subset),
      each verified alone on 5 wheels (0, 0, 0) and assembled (0). LEDGER ENTRY: 6 closed
      (L31 as a formula, the vanishing moments, r(d) closed form, the d = 4 exception, L18's
      multiplicities, the free-tiling guess); 5 measurements or compute cutoffs; 2 the twin
      conjecture in a range coordinate (not the wheels' business); 4 GENUINELY OPEN on the
      wheels alone: the complexity of min_U D_L(U) (the natural child), non-cancellation of
      M_{r(d)} (L75, 0 exceptions of 24), L22 in the kernel, a parity-refined capacity
      bound. Scorecard: 12 of 12 held in substance (P3's named instances refuted; P4's scan
      cap lowered to 2.4 x 10^7).
    - R4.b.xii. The loaded record rule in Lean (round 36; proofs/TopMachineRecord.lean, 49
      declarations; CutMono addendum in proofs/MachineStack.lean, 14 declarations; ledger
      research/proof/top_machine_lean.md). KERNEL: the piece law (trace_subset_domino,
      trace_card_le_two, trace_one_parity: a tail gear's strikes in the window lie in one
      domino; trace_crosses_parity_iff: a gear's piece crosses parity iff g <= L + 1;
      ends_join_iff: a gear strikes both 0 and L - 1 iff g = L + 1 exactly, a distinction the
      branch did not draw); the matching lemma (Piece, CoveredBy, domCost as an sInf;
      card_le_two_mul_of_coveredBy; domCost_union_parity: the cost splits across parity
      classes; domCost_run a j = (j + 1)/2 by cardinality plus the explicit tiling); THE
      LOADED RECORD RULE both directions: cost_le_tail_of_coverable needs NO hypothesis at all
      (the tail's definition L + 1 < g is the only thing used), coverable_of_cost_le_tail needs
      pairwise coprimality alone (no size hypothesis on tail gears: the residue -(c + 2)
      makes any gear strike both c and c + 2), loaded_record_rule the iff, record_set_eq and
      record_isGreatest_iff (F_top in the rule's form), coverable_mono; the boundary
      corollary boundary_cost = 2 floor(L/4) + min(L mod 4, 2), boundary_greatest, and
      parity_law_of_rule, a second independent proof of the parity law using neither
      parity_upper nor parity_attained (oddness used in one place: at even m to rule out
      g = 2m + 2). CUTMONO UNCONDITIONAL: cut_succ_gt_four_mul (prime q >= 5, k >= 1 gives
      4 cut_k < cut_{k+1}) and cutMono_of_five_le, by one strong induction on b with a
      halving step (no logarithms, no reals; three Bertrand primes when 8a <= b < 16a, two
      when b < 8a and a >= 16; base q >= 7 via q# >= 30q, q = 5 via 7 * 11 * 13 = 1001 > 120,
      cut 5 1 = 30 by interval_cases), so stack_eq_primesLE, exhaust_silent and
      stack_open_iff_twin are unconditional for every prime base q >= 5. Manager gate: build
      of the six targets green at 2246 jobs; audit of loaded_record_rule,
      cost_le_tail_of_coverable, coverable_of_cost_le_tail, parity_law_of_rule,
      ends_join_iff, domCost_union_parity, cut_succ_gt_four_mul, cutMono_of_five_le,
      stack_open_iff_twin: propext, Classical.choice, Quot.sound; zero sorries, no decide.
      The library is 269 declarations. NOT ATTEMPTED: the capacity bound L70, the moment
      vanishing L73/L74 (needs Boolean-cube Mobius inversion), the census law L22 (K2's
      general powerset inclusion-exclusion is the missing machinery).
    - R4.b.xiii. The gap census law in Lean (round 37, Fable; proofs/TopMachineCensus.lean, 41
      declarations). KERNEL: gap_census (register W22): the number of residues n mod W with n
      and n + d consecutive open pairs equals the alternating sum over subsets S of the
      interior positions of prod_g (g - |E_g(S)|), with E_g(S) the image of the offsets
      {0, 2, d, d + 2} u S u (S + 2) under off g; hypotheses ONLY gears positive and pairwise
      coprime (no g >= 5, no primality, no oddness: the size hypotheses of the branch are
      needed only to evaluate |E_g(S)|, not to prove the law). Three general lemmas carry it:
      K1 the residue characterisation (consecOpenN_iff_residues), K2 card_filter_forall_not, a
      fully general inclusion-exclusion over a Finset of predicates (no hypotheses; mathlib's
      version needs a Fintype and was unusable), K3 card_avoid_prod, the count of residues mod
      a product of coprime moduli avoiding per-gear forbidden Finsets factors as a product;
      wheel_count and the pair correlation are its special cases (pair_corr_teeth). Check:
      gap_four_zero, the census at d = 4 is identically 0 from the formula by pairing S with
      S u {2} (equal products, opposite signs): the forbidden gap derived from the algebra.
      Manager gate: build of the seven targets green at 2248 jobs; audit of gap_census,
      card_filter_forall_not, card_avoid_prod, gap_four_zero: propext, Classical.choice,
      Quot.sound; zero sorries, no decide. The library is 310 declarations. NOT ATTEMPTED:
      the moment vanishing L73/L74 (needs the per-subset evaluation of |E_g(S)|, the
      elementary-symmetric expansion and a Boolean-cube Mobius inversion, none in the kernel).
    - R4.b.x. The exhaust, first pass (spawned by the objects ledger: the exhaust is the only
      object with no measurement and no branch document). Self-similarity of tier 3 measured
      against the wheels' laws in its own parameters; CutMono's exact elementary form and
      written proof from Bertrand; the zones and the redundancy lemma on a range; the
      exhaust's own record (expected ROOT); the exhaust's action below and above the window
      (home / echo census, the first strike that is neither). STRONG (research/proof/
      exhaust_1.md, laws X9-X24; scripts research/exhaust/r1/): the exhaust is now a measured
      object. SELF-SIMILARITY MEASURED: four tier wheels (q = 5: {31, 37, 41, 43}; q = 7:
      {211, 223, 227}; q = 11: {2311, 2333}; control tier 2 {7, 11, 13}), full periods,
      18,095,756 residues, 15 wheel-law checks each, 0 exceptions; on ranges the zone laws with
      "smooth" = q#-smooth verified cell by cell over 21,669,556 cells, 0 exceptions; p_1 = the
      first prime above Q is the first admissible non-smooth number every time. Sharpening
      X10: the run-start count prod(g - 2 - L) is FALSE at L = 1 (the count is prod(g - 2)) and
      true for all L >= 2; the chain count prod(g - 1 - L) has no exception (the shield is a
      singleton arc for runs but merges with the other tooth in the step-2 order). X23: the
      gap-3 = gap-5 identity transfers verbatim to tier 3 and fails only on the control wheel
      with gear 7. CUTMONO IS A THEOREM (O-X1 closed): Lemma A (dyadic Bertrand) for a >= 1 and
      2^t a <= b the product of the primes in (a, b] exceeds a^t 2^(t(t-1)/2); Lemma B for
      a >= 5, b >= 4a the product exceeds b (and 4b if a >= 16 or b >= 8a); X12 for every prime
      q >= 5 and every k >= 1, cut_{k+1} > 4 cut_k, so CutMono holds for all k unconditionally;
      X13 the degeneracy's exact cause: the base step is the product of the primes up to q
      being >= 4q, which fails at q = 2, 3 AND 4 (not recorded before) and holds from 5; below
      the threshold one dyadic interval fits, Bertrand supplies one prime, and one prime <= b
      never exceeds b (cut_2 = 5 < 6 at q = 3). Written for transcription (needs only
      Nat.exists_prime_lt_and_le_two_mul). Cut scale: cut_2 = 215,656,441 at q = 5 (80 digits
      at 7, 973 at 11, 12,930 at 13); theta(cut_2) = 215,639,987.078 by segmented sieve, so
      cut_3 at q = 5 has 93,651,247 digits; tier 3 at q >= 7 is worked as the loaded exhaust
      on a range, losing nothing since every law is in intrinsic parameters. THE REDUNDANCY
      LEMMA, RANGE FORM (O-X2 closed), stronger than the kernel's window form: for ANY integer g
      with g^2 > N, every multiple of g in [1, N] is g itself or has a prime factor strictly
      below g (one inequality, no primality, the whole range); silent gears (g > N/2) touch
      exactly the two pair positions g and g - 2; 7,357,725 strikes and 5,019 silent gears, 0
      exceptions. Zone census at N = 10^7: 664,579 gears, only 446 repeat; the rest give one
      home strike each plus 5,973,710 echoes; the echo share of the work above sqrt N climbs
      80.8, 85.2, 88.1, 90.0% across four decades. THE EXHAUST'S RECORD IS ROOT (O-X4 closed
      as a statement): the bottom stratum (Q, 2Q] holds only primes and cut-smooth numbers, so
      a tier's family (1, 1) is the twin primes above Q; the prime-gap floor holds at four
      machines (ratios 7.18, 9.43, 7.67, 13.00); X24 the twin counts are IDENTICAL across
      splits at equal Q (25 at Q = 997, 64 at Q = 3137) while the smooth-member counts differ
      (86 vs 528): the family carrying the obstruction is the one the split cannot touch. The
      owner's O2 REFUTED as stated and replaced by the regime law X20: F_range = max(F_smooth,
      F_quiet) exactly at all 9 machines; at q = 5 tier 3 the record crosses into the smooth
      zone between N = 10^6 and 10^7 (34 to 210 against 66 to 122), at q = 7 tier 3 not by
      10^8 (40 against 55); mechanism: raising the split densifies the q#-smooth-pair list and
      shrinks its gaps; the q = 5 list saturates at 423 pairs (largest 354,365,440, Stormer)
      so F_smooth tends to Q - 2 - s with slope 1, while at q = 7 the list is still growing at
      10^11; X22 the U-profile fails at q = 7 tier 3 (strata 23, 20, 21, 27, 30, 35, ...). WHAT
      THE EXHAUST DOES: on [1, q#] no strike at all; on the windows (30, 900] and (210, 44100]
      all 57,344 incidences are home or echo, 0 neither, and every open pair receives exactly
      2.000 exhaust strikes, both home (X17); above the window the first non-redundant strike
      is exactly p_1^2 (961 = 31^2 at Q^2 + 61; 44,521 = 211^2 at Q^2 + 421; X18, depth
      floor(log_{p_1} x)); the exhaust's share of the open pairs is 0.000% on (Q, Q^2], then
      36.9, 62.8, 75.0, 82.0, 86.6% over the next five decades at q = 5 (X19): the cap is
      exactly zero on one stretch and the exhaust is the majority partner one decade later.
      NEW OPEN ITEM O-X6: the crossover height Q*(C) of X20 is the one place in the exhaust
      where the missing instrument is an UPPER bound on a prime gap (a known theorem), not the
      conjecture. O-X3 stands as the root question named; O-X5 (prior art) stands, X10 and X13
      the two items most likely new. Instrument error recorded: step-2 chains on an odd period
      must be counted on the single doubling cycle, not per parity class (the wrong version
      manufactured q' - 3 false exceptions at every wheel including the control). The law
      register (research/proof/law_register.md) carries X9-X24 since 2026-09-07: the exhaust's
      first pass adds no new mathematics (Bertrand iterated; Eratosthenes to the root in range
      form); its value was measurement.
    - R4.b.xi. The wheels' last open laws (spawned by R4.b.ix's ledger entry: four items
      genuinely open on the wheels alone). Non-cancellation of the top moment M_{r(d)} (W-law
      to prove), the structure and complexity of the core minimisation in the record formula,
      the parity-refined capacity bound. STRONG, closed (research/proof/top_machine_8.md, laws
      W95-W102 (written as W86-W93, renumbered); scripts research/topmachine/r8/). NON-CANCELLATION PROVED (W95): M_{r(d)}(d) =
      (-1)^r r! C_r(d), every minimum cover carries the same sign (-1)^r because its only
      covering subfamily is itself, so cancellation is impossible; exact at every d = 2..26
      including d = 4 (C = 0); all 34 minimum covers of d <= 14 have mu = (-1)^r by direct
      summation. The count is closed-form by d mod 4: C_r(d) = d/4 - 1, 1, (d + 6)/4,
      ((d + 1)/4)^2; the eight published multiplicities are r! C_r (6 * 3, 24 * 4, 24 * 1,
      24 * 1, 120 * 4, 720 * 9, 720 * 2, 720 * 1), 8 of 8; the forbidden gap 4 is the case
      d/4 - 1 = 0. STRONGER THAN ASKED, THE COVER POLYNOMIAL: the whole universal signature is
      sum_e c_e(d) z^e = sum_t C_t(d) (1 - z)^t z^(d + 3 - t) (every coefficient, d <= 26, 0
      mismatches), hence N_d(G) = sum_t C_t(d) (-1)^t Delta^t P_G(d + 3 - t) (27 universal
      cases against L22 and scans, 0 mismatches); the moment vanishing, r(d) = D(d - 1), the
      gap-3 = gap-5 identity and the moment form fall out in a line each; c_e(d) is O(d^2) by
      a transfer matrix (tabulated to d = 60), closing the census-beyond-20 item for the
      universal regime; a direct CRT bijection (gap <-> minimum cover + gear assignment) gives
      N_d = r! C_r with no inclusion-exclusion. THE MINIMISATION'S STRUCTURE: D splits by
      parity, and via 2^-1 mod W_core the two classes are two windows of one adjacent-teeth
      core wheel, [0, ceil(L/2)) and [H, H + floor(L/2)) with H = (W_core + 1)/2 (0 mismatches
      on 19,896 phase vectors); the record is a scan of W_core (median gain 29,939x on the
      family; no gain when the tail is empty, i.e. in use). The classes cannot be decoupled
      (min(D_e + D_o) > min D_e + min D_o on 3,611 of 5,006 loaded sets, on every set with
      >= 4 core gears, already at {5, 7}); the run structure cannot be dropped (1,875 sets);
      a gear's dominoes cannot keep a fixed grid (they alternate, g odd); anchoring a core
      gear at cell 0 fails on exactly 210 sets, all {7} + four gears > 13 at F = 12, where
      phase 3 is the unique optimum. How special the optimum is: median 4 distinct D values at
      L = F, median 5.71% of phasings at the minimum, unique up to reflection on 2,965 sets,
      2 phasings in 3,172,455 at {5, 9, 11, 13, 17, 29}; full enumeration re-decides all 6,659
      records with 0 mismatches; NO polynomial algorithm found or claimed (the item is
      reclassified as a complexity question, every structural ingredient proved). THE BOUND
      (W102): the parity-refined Lcap2 proved, F <= Lcap2 <= Lcap, 0 violations on 6,659; exact
      on all 1,653 free sets and on every set with core density rho < 0.376 (3,349 of 3,350
      below 0.4), never above 0.7, vacuous at rho >= 1: both capacity bounds are density
      bounds with slack ~ 1/(1 - rho); in use rho > 1 once Q > q^1.65. The slack is the
      run-parity kind 1,875 times and the overlap kind 702 times ("loose iff overlap"
      REFUTED); P16 half-refuted (48% of loaded sets exact, not a majority). LEDGER LINE: of
      the four items, non-cancellation (W95) and the bound (W102) are closed, the minimisation is reduced
      to a W_core scan with every structural ingredient proved (a complexity question, not a
      structural one), L22 is in the kernel (round 37): ON PAPER THE WHEELS HAVE NO OPEN
      STRUCTURAL ITEM. Register note: document 7's L67-L75 still lack W-numbers.
    - R4.b.vi. The walk laws in Lean (round 34; proofs/TopMachineWalk.lean, 70 declarations;
      ledger research/proof/top_machine_lean.md). KERNEL: mex_form, the next open pair after x
      is x + mexS as an IsLeast statement, with the hypothesis 2m < g placed exactly where it is
      used (openness of x + mex; struckness below the mex needs only g > 0); mexS_le (<= 2m) and
      mexS_le_parity (<= 2m - (m mod 2) for odd gears above 2m + 1, by reuse of parity_upper: the
      walk below the mex IS a run of struck pairs). triple_mex_form and triple_law: the record
      of the run-of-three machine is exactly 3m as IsGreatest; the upper bound needs only
      3m < g (one notch weaker than the branch's 3m + 3), and triple_attained needs NO size
      hypothesis at all (one multiple per gear kills all three starts of its block), which
      shows in the kernel that the parity defect comes from the domino's gap, not the tooth
      count. no_start_gap: two run-of-three starts at distance 2 or 3 are never consecutive,
      with no hypothesis at all. pair_corr: the pair-correlation count is prod corrCoeff(g, d)
      (g - 2, g - 3, g - 4 by d = 0, +-2, else) for gears >= 5. One primitive unified the
      three laws: off g y = least j with g | y + j. Manager gate: lake build TopMachine
      TopMachineWheel TopMachineCrt TopMachineWalk green at 1394 jobs; audit of mex_form,
      mexS_le_parity, triple_mex_form, triple_law, no_start_gap, pair_corr: propext,
      Classical.choice, Quot.sound; zero sorries. 158 declarations in the library. NOT
      ATTEMPTED: the in-use mex with truncated progressions, the harmonic bound, the C(j)
      distribution and hop laws, spectral and bitwise (no DFT in the files), the gap census
      law L22.
    - R4.c. THE VALVES (formerly "the clutch"; OPENED 2026-09-07: the ledger's gate cleared,
      every part with no open structural item that is not ROOT). First step, the owner's
      hybrid, running: the SCRATCH lane (Fable, clean context: only the glossary, the ledger's
      definitions and proved laws, the census table; forbidden the wall, the tree, the
      refiled facts) builds the interface objects from the definitions
      (research/proof/valves_scratch.md); the REVIEW lane (Fable) sorts the 36 refiled facts,
      the wall, theorem (E), the conjugacy, the family decomposition into interface objects
      with coordinate flags and inherited assumptions, and puts three predictions and one
      red flag on record (research/proof/valves_review.md); RECONCILED 2026-09-07
      (research/proof/valves_reconcile.md). THE VALVES HAVE A MECHANISM. Found by both,
      solid: the IMPRINT (a family (s, s') occupies exactly the residues mod q# with n = 0
      for p | s, n = -2 for p | s', neither for the other engine primes; in columns each
      burning gear pins the family to one tooth; CRT proof, 0 exceptions in 45,358 families);
      the pure charge's count independent of q (the count identity, 9 of 9 runs); the record
      a twin gap (EMBER: every burnt charge in turns 1 and 2 has a q-smooth member above Q,
      proof by the air cap plus parity). Scratch-only, new: PORT (a family's class mod 6 is
      fixed by its air; the fold is the port; one-line proof, 0 exceptions in 23,969,812
      pairs); TURN and ONSET (turn m = (mQ, (m + 1)Q]; a family has no member before turn
      max(s, s'), proof n = sP > sQ; measured exact: every family with max <= 25 fires at its
      turn; turns 1 and 2 carry no fuelled family but (1, 1)): THE VALVES OPEN IN ORDER OF
      THEIR AIR; INVENTORY (exists iff q-smooth, gcd | 2, same parity, 4 divides exactly one
      of an even pair; proof by local solvability; the data refused (2, 6) and the proof
      followed); YIELD (N(s, s') = the imprint's local density integrated, within 2% over 310
      families; measured; prior art the Bateman-Horn local factors, family (4, 2) the Sophie
      Germain primes); SPOKE (columns of mQ engine-open when q# | Q); neighbour facts (nearest
      burnt charge at distance 2 for 37% of twins, always (3a, 1)/(1, 3a'); burnt count between
      twins not a function of the gap); the first pure charge t_1 >= p_1 only. Review-only,
      PARTIAL: the corridor, the island witness and K(d) (does not translate into families),
      the four cells, the zero-interaction region, (E)'s exception set, the twisted copies.
      The red flag held. SHAPE OF THE ROOT: in turns 1 and 2 the only fuelled valve is (1, 1)
      and the other charges are embers; the question is whether the twins in (Q, 3Q] can be
      absent while the embers are present.
      - R4.c.i. The turn ledger (spawned by the onset law). Per turn m the total charges T_m,
        the burnt charges B_m = the sum of the open valves' yields, the pure charge
        P_m = T_m - B_m; whether an exact relation between T_m and B_m forces P_m > 0, and
        where the counting face of the wall reappears. ANSWERED NO, BY PROOF
        (research/proof/turn_ledger.md, laws V1-V5; scripts research/valves/r1/). V1 the
        ledger identities, 0 exceptions in 600 turns (10 ledgers x 60): P_m = T_m - sum over
        the open valves I(m) minus (1, 1) of N_m(s, s') - B_m^ember with 0 <= B_m^ember <= 2 E_m
        (E_m the embers in the turn); the family sum needs the ember term (nonzero in 47-60 of
        60 turns); in turns 1 and 2 the burnt side is the ember charges alone. THE SHARPEST
        STRUCTURAL STATEMENT THAT IS NOT THE CONJECTURE: everything the pure charge must beat
        is named, placed and bounded by smooth numbers (the list I(m) of A(m) - 1 fuelled
        valves, each on its imprint, none present before its air, plus at most 2 E_m ember
        charges); whether it beats it is a prime count. V2, THE WALL AT THE LEDGER, PROVED:
        every proved valve law (imprint, onset, port, inventory, ember) uses only "fuel is odd,
        above Q, coprime to q#"; the counterfactual fuel set F = {n > Q : gcd(n, q#) = 1,
        n = 1 mod 3} satisfies all of them with P_m = 0 and B_m = T_m > 0 in every turn (0
        violations at (5, 10^3) and (7, 10^4)); so no inequality B_m <= c T_m with c < 1 can
        follow from the valve laws, and the wall at the ledger is face A and nothing else; the
        alternating expression P_m = sum mu(d_1) mu(d_2) M_m(d_1, d_2) telescopes to an identity
        and using it is a dimension-2 sieve on the charges (face A in three lines). Candidate
        inequalities: C1 (P_m > 0) 0 exceptions, ROOT; C2 (the ember bound) structural; C3
        (B_1 <= P_1, B_2 <= P_2) fails at (7, 10^3) turn 2 (22 > 21) and for small Q; C4
        (monotone B/T) fails everywhere; C5 fails in 9 of 10; C6 fails 10 of 10; C7 (the pure
        charge is the largest family) holds at Q >= 10^4 (420 turns), fails at 10^3. THE FIRST
        TWO TURNS, every Q <= 10^5, q = 5, 7, 11, 13: P_1 = 0 iff Q in {1, 5}; P_2 = 0 iff Q in
        {3, 9}; min P_1 on [10^3, 10^5] = 25 at Q = 1031 (twin gap 120 from 1487); min P_2 = 17
        at Q = 1071 (gap 168 from 2381); embers outnumber twins in turn 1 last at Q = 74, 400,
        726, 1355 and in turn 2 last at 555, 1150, 2963, 6562 for q = 5, 7, 11, 13. THE VALVE
        COUNT A(m) exact to 60 (q = 5: 1, 1, 3, 5, 9, 11, 11, 15, 19, 23, ... 125; q = 7 to 233;
        q = 11 to 327), recursion A(m) = A(m - 1) + 2 #{s' < m smooth admissible with m} at
        smooth m; no valve ever early (0 extra in 600 turns); all present from onset to
        m = 19 / 19 / 15 at Q = 10^4, 49 at (5, 10^5), then sparse valves miss turns. THE BURNT
        FRACTION B/T = 0.021, 0.088, 0.622, 0.762, 0.819, 0.850, 0.885, 0.903 at m = 1, 2, 3, 5,
        9, 15, 30, 60 for (5, 10^4), rising only at smooth m; law T_m/P_m = sum over I(m) of
        w L_m with w = 2^[s even]/(s s') prod over odd p | s s' of (p - 1)/(p - 2), within 2.2%
        over turns 30-60 at every Q >= 10^4; B/T does NOT tend to 1 in m: the pure share tends
        to (1/2) prod_{3 <= p <= q} (1 - 2/p), the engine's own pair-opening density (0.100,
        0.071, 0.058), and tends to 1 in q like 1 - c/(log q)^2. Refuted own predictions
        recorded (the 2-adic factor of the even families; "every valve present at every turn
        from onset"; B_2 <= P_2 for q >= 7). THE SHADOW: the property of the fuel that the
        proved laws do not use and the counterfactual lacks is the fuel's own distribution
        across the engine-open classes at distance 2 (F sits in one class mod 3 and has no
        pairs at distance 2 inside it); a set hitting every class need not contain such pairs
        (the parity barrier's example), so the object is the primes' pair correlation itself.
        UNSTICK PASS 3 (research/proof/dead_branches_reopened_3.md): (a) the fuel as the
        exhaust's gear set walks DOWN exactly via theorem (E): turn m of Q is the top
        1/(m + 1) of the window of sqrt((m + 1) Q), so P_1(Q) > 0 follows from
        F(y) < W(y)/2 at y = sqrt(2Q); every valve turn is the engine's window at a lower
        rung, the descent spending the slack as 1/(m + 1); (b) the exhaust's dominoes on
        the engine's wheel: the engine acts on prime-led charges with one tooth, so the
        walk to the next twin is a one-class sieve on the prime sequence, dimension 1,
        parity-blocked; (c) adversaries: what F lacks as objects is balance on gear 3's
        teeth, mirror closure of the valve set (checked 11 of 11, 5 of 5) and one
        number-tooth per gear; a one-tooth free-phase manifold adversary with all three
        still empties the pure charge of the first 60 turns at Q = 10^4 with 776 of 1,226
        gears; what remains is phase zero at every prime up to Q, i.e. the real fuel; (d)
        the valves' walk is loaded from turn 3 and cannot collapse. Every ROOT node
        descends to the engine's record (W103, L65 to F at sqrt(3Q); the skip half to
        F/q'). THE ONE OBJECT THAT IS NOT THE RECORD RESTATED: its dual, the rich-interval
        function Omega of the pullback along the new gear (the most openings of the twist
        M^(q') in n consecutive multipliers), E2/E3's exact cap on the skip half with slack
        0 or 1 at nine of nine rungs, universal and finite per n, a max over translates
        rather than a count. Recommended and opened: the rich half of the record (engine).
        - R4.c.iii. Existence in the valve (owner, 2026-09-07: "we don't have to see
          position; just knowing there exists any position would be proof", and "the position
          doesn't have to be inside the window, which is the engine; it can be in the valve").
          Exact form: for every Q some turn m holds a pure charge = a twin in (Q, Q^2]; by the
          descent, the engine's window statement at SOME scale in [sqrt(2Q), Q], not every
          scale. Branch: the freedom of the turn; the frontier's certified turns (position
          facts seeing length at the top of the window); invariants of the charge set forcing
          a member on the pure imprint, each tested against the counterfactual fuel. DONE
          (research/proof/valve_existence.md, laws V6-V12; scripts research/valves/r2/).
          V6 the descent exact (0 mismatches in 60 turns at (5, 10^4) and 19,998 cells at
          m = 1, 2). THE FREEDOM OF THE TURN IS A CHOICE OF RUNG, NOT A WEAKENING: the
          weakest engine statement implying existence in the valve is the position form
          (some turn at which the blocked run of {5..y_m} ending at the slice's top is shorter
          than Q/6 columns), which is the turn statement renamed, ROOT; the share form
          (S < 1/(m + 1) at y = sqrt((m + 1) Q), no rung beyond m = 3 since F/(y^2/6) >= 0.278)
          is strictly stronger and certifies every Q in [3, 1859] except 24 and 25; the
          freedom is worth exactly three integers (Q = 5, 26, 29); the disjunction over turns
          is the window statement at rung Q less a sliver. V8 (PROVED with its hypothesis):
          a frontier constant c at rung y_m certifies exactly the turns m < c, because the
          frontier sees length at the top (the run ending at the window's top is at most
          1/(c + 1) of the window). With c = 1.25 from the certified ladder (F(59) = 161):
          turn 1 for every Q in [30, 1859], turn 2 on the whole range at rungs 11, 29, 41,
          43, 47, 53, 59, turn 3 at 24 values of Q, turn 4 at Q = 9; 0 of 2,944 certificates
          empty. Measured (prefix floor c = 4.625, 0 exceptions to rung 19,997): turns 1-4
          to Q <= 8 x 10^7; no constant reaches turn 5. V9 (EXACT): the frontier's tight
          instance 4.625 = 111/24 IS the twin gap 661 -> 809 and it is the last empty turn 5
          (Q = 132, 133, 134; Q_5 = 135); the period constant 3.25 is the gap 73 -> 101: the
          frontier's constants are the primes' first twin gaps, fixed for all larger machines
          because new gears lengthen runs rather than precede them; the universal frontier is
          twin-Bertrand on the prefix, ROOT beyond the measured range. INVARIANTS: the mirror
          fixes the pure imprint with one fixed class (-1 mod q#), forces inventory closure
          (122 of 123 families) and count symmetry, nothing on existence (FACT); the
          pigeonhole V10 fails at every q (needed F_odd + 1 = 7, 16, 22, 34, 55, 76 open odd
          numbers against the manifold ceiling q' - 1 = 6, 10, 12, 16, 18, 22; margin 1 at
          q = 5 and realised: 2 of 14 runs of six hold no twin; FACT); the walk V11 has no cap
          (per-turn max 2 to 53 at (5, 10^4), 0.74-1.86 of the geometric scale; the manifold's
          laws cap runs of charges, the wrong side; FACT). COUNTERFACTUAL TEST V12: F violates
          balance, mirror closure, one-tooth and phase zero; the one-tooth adversary (776 of
          1,226 gears, P_m = 0 to m = 60) violates phase zero only: PHASE ZERO IS THE SOLE
          SURVIVING INVARIANT, a set property (which class each gear removes) whose existence
          consequence is the conjecture; side fact: the greedy adversary cannot empty even
          turn 1 at (5, 10^3) with all 165 gears. Next child named: "a new gear never creates
          a run of length >= d_0 before the runs already there", an engine construction
          statement; its proof would give turns 1-4 for all Q via V8.
          - R4.c.iii.a. New gears lengthen, never precede (spawned by V9: the frontier's
            constants are fixed for all larger machines because a new gear lengthens runs
            rather than creating a long run earlier). The statement: adding q' never creates
            a run of length >= d_0(M) before the first such run of M; a proof gives turns 1-4
            for all Q via V8. FACT with a ROOT mark and one refutation (research/proof/
            lengthen_never_precede.md, laws E6-E8; scripts research/anchor235/r69/; the
            m37-m43 period table pending its scan). E6 (PROVED, direct-sieve gate at 428
            rungs): theorem (E) at one step, with its exact two-column exception set: below
            W(q) = (q'^2 - 1)/6 the new gear q' newly blocks exactly the two columns whose
            number is q' times 1 or the twin partner, never the column of 5q' (the brief's
            "+-u_{q'}" was wrong); identity h(q') <= d_0(M) with equality iff (q', q'') is a
            twin pair. E7 (PROVED, 0 exceptions in 8,152 cells over 2,260 rungs): the frontier
            of the PREFIX [1, W] is inherited exactly up the ladder, the absorbed lengths and
            the run through the square column being the complete list of changes: new gears
            lengthen and never precede ON THE PREFIX. ON THE FULL PERIOD THE SENTENCE IS FALSE:
            8 of 11 steps m5 -> m43 have a run of length >= L appearing earlier after the gear
            is added, every exception a merge of order 2-4 beyond W(q) or the one straddle at
            17 -> 19 (R_min(6) = 151 at m11 against 89 at m13; 89 against 61 at m17; 61 against
            59 at m19). Form C (the constant monotone) false twice on the prefix (17 -> 19:
            10.6 -> 5.364; 19 -> 23: 5.364 -> 4.625) and never again to 19,997; the floor 4.625
            attained on rungs [23, 131] and never undercut; the staircase of constants is
            eleven twin gaps, each the minimiser on an interval of rungs; c_pre infinite from
            1427 (the initial run is the longest run of the prefix). E8 (PROVED with its
            proviso): H_c(q) implies H_c(q') for c <= 6.25 and q' >= 118 provided the
            STRADDLING run (the run of M + q' through the square column W(q)) satisfies
            x_s >= c L_s whenever L_s >= d_0(q'); interior section runs disposed of by a prime
            gap bound (E8-a, prior art, q' >= 118); inherited runs by E7. THE RESIDUE, EXACT: in
            the twin coordinate the straddling condition reads "a twin gap across a prime square
            that is longer than the prime is at most 21.6% of its start" (t_1 - t_0 >= q' + 6
            implies t_1 - t_0 <= (t_0 + 7)/4.625 + 6); it is vacuous at every rung above 487 in
            the measured range (the straddling run at most 484 columns against an initial run
            of q'/6) and rests on 24 rungs where it bites, all with ratio >= 4.75; it is ROOT in
            face E's sense (a two-sided gap bound at the square, stronger than existence,
            implying infinitely many twins by chaining), and the object is NOT d_0 <= W but the
            twin gap across the prime square. CORRECTION TO V8: its proof uses a hidden
            hypothesis, d_0(y_m(Q)) <= klo_m(Q), a twin in (y_m, mQ] (true in 399,973 of
            399,973 cells to 10^5 and supplied by the induction it certifies); and its frontier
            hypothesis is non-vacuous only at rungs 5, 7, 11, 23, so above Q = 148 the measured
            certification of turns 1-4 is the share form F_pre(y) < c_m(Q), already ROOT: the
            frontier's constant certifies turns 3 and 4 for Q = 106..148 and nothing else.
            Formalist target named: E6 (divisibility of 6k -+ 1 by q' below q'^2, kernel-sized)
            and E7 as its corollary on maximal runs. Genuinely open on the part alone: none.
        - R4.c.iv. Phase zero in the manifold's own terms (owner's correction, 2026-09-07:
          the descent to the engine's certified range and rungs was a regression to the
          approach known not to reach; the proof space is the valves' domain (Q, Q^2] with the
          manifold's and the valves' laws on the raw line; window, rung, ladder and the descent
          FORBIDDEN in this branch). The manifold-native wall: the fuel is not free (it is the
          primes), the only structure-respecting counterfactual is free phase, and the
          free-phase adversary kills the pure charge; the real manifold's distinguishing
          property is PHASE ZERO (a gear strikes exactly its multiples). Branch: which
          manifold and valve laws hold under free phase (no phase-zero content) and which
          fail; which fail for the parity-defined adversary too (content beyond
          multiplicativity); the charge set as a structure (burn as multiplication by air);
          the record laws on the charges. DONE, ROOT honestly (research/proof/phase_zero.md,
          laws V13-V17; scripts research/valves/r3/). X = SATURATION (V14, proved): all phases
          zero iff the struck set is a union of ideals iff the open set is closed under
          divisors and under multiplication by air; the real charge set has 0 violations in
          107,750,211 closure tests at (5, 10^4); no adversary on record has it: free-phase
          manifolds fail 76.5% of closures, the one-tooth adversary and V2's F have composite
          fuel, the parity adversary {even Omega} is a sub-semigroup but not divisor-closed
          (closure fails at exactly the odd-Omega air, never at even-Omega air). X DOES NOT
          FORCE A PURE PAIR (V17): the saturated sieve with the twin members in (Q, Q^2] added
          as gears (880,214 gears above Q) passes every row the real manifold passes and has
          0 pure pairs in all 9,999 turns; what excludes it is the cut, and saturation plus
          the cut is the manifold's definition. LAW TABLE SPLIT: 13 phase-free, 1
          position-only, 10 phase-zero, of which 3 beyond multiplicativity (one property,
          saturation, counted three ways). Phase-free by the translation lemma V13 (a
          free-phase manifold is the real one translated by the CRT solution of t = c_g; 15
          of 15 wheels exact, periods to 215,656,441): domino form, no gap 4 (a tautology for
          any number set), run and chain ceilings, parity law, record rule, census law, mex
          form, wheel count and correlation product, symmetry group; imprint, port,
          inventory by V2's proof. Phase-zero via "fuel above Q" only (so held by the parity
          adversary too): onset, air cap, ember law, the quiet-zone "only if"; nothing on
          record uses the order of the multiples beyond P > Q. The one-tooth adversary is not
          a sieve on the raw line (breaks the run ceiling). Burn is a bijection air x pure ->
          burnt on numbers, giving T - E = sum_s N_pure(Q^2/s) exactly (23,899,705 both sides,
          0 difference at four (q, Q)): Legendre's identity, FACT; the pair system is not
          closed. V15 (new, measured), THE SLICE PROFILE: a random slice is flat at the
          period density in every turn and gap length (56 census cells within 2.5 sqrt N);
          the real slice is poor at the bottom (turn 1: 0.107 = 47% of 0.228; small gaps at
          0.5-6% of the period count in turns 1-2), rich in the middle (turn 60: 127%; pairs
          175%), at the period value at the top; mechanism: onset closes the families at the
          bottom, a single prime's density 1/log x beats the period's rough density in the
          middle (Buchstab against Mertens, prior art); the record (420 after 26,261, turn 2)
          sits in the emptied bottom: W103 in density terms. The record rule is silent about
          the slice (at L = 420 the free core covers the whole window; the period record
          >= 1,880, 4.5x the zone record; the capacity bound vacuous). Side facts: the
          Liouville-signed charge census is -6,635 of 5,376,501 (0.1%) against a pure share
          of 8.2%: the parity barrier in one number. Correction to V12: the one-tooth
          adversary's fuel is not divisor-closed.
        - R4.c.v. Openings aligning with the engine's twin slots (owner's mechanism,
          2026-09-07: the manifold's longest opening is its first gear's arc; if it aligns
          over the engine's twin slots, which occur at a definite frequency, that is the
          proof, within or outside the certified range). Manager's two size facts on the
          scorecard: the longest opening q' - 3 is far shorter than the engine's blocked runs
          (about 6 F(M)), so an opening must land on a slot, not span to one; and the longest
          opening is rare (density about exp(-(q' - 1) sum 1/g)) and may not occur below Q^2.
          Branch: the opening spectrum on the quiet zone; the engine's gap at each opening;
          the engine-plus-first-gear composite (period q# x q') and how each further gear
          thins the aligned openings against CRT and against free phase; forced alignment.
          DONE, FACT with a ROOT mark (research/proof/opening_alignment.md, laws V20-V24;
          scripts research/valves/r4/). Size fact (i) HOLDS: the engine's longest blocked run
          of pairs is 11, 29, 41 (q = 5, 7, 11) against the run ceilings 4, 8, 10, and even
          the mean slot spacing 10, 14, 17.1 exceeds the ceiling: an opening holds at most
          ceil(L/6) slots. Size fact (ii) HALF: the ceiling length's density is exactly
          prod_{g in (q, Q]} (1 - (q' - 1)/g) = 1.5e-4 to 1.4e-8, yet the ceiling length IS
          present below Q^2 in 6 of 7 runs (247 / 3,634 / 78,660 at q = 5 for Q = 10^3, 10^4,
          10^5; 6 / 14 at q = 7; 6 at (11, 10^4)), absent only where CRT expects 0.4; law of
          the longest opening below Q^2: attained whenever the CRT count of the ceiling
          length is >= 1, measured 1.1 to 4.6 times CRT. V22 (EXACT, 0 exceptions in 65,115):
          an ember-free opening lies in turn >= the max of its smooth vector; slot-free
          openings can enter at turn L + 2, slot-holding ones not before turn 12 (L = 3..8),
          16, 18 (L = 9, 10); the first aligned length-3 opening at (5, 10^4) sits at 129,586,
          turn 12, exactly on the threshold; the earliest twins in length-2 openings are
          ember-seeded (2,592 = 2^5 3^4; 21,600 = 2^5 3^3 5^2). ALIGNMENT IS GENERIC, NOT
          STRUCTURED (V21): over a full period the engine-plus-first-gears composite is
          exactly uniform and phase-blind (periods to 9.7 million; every count equals its CRT
          integer; two random phase vectors identical; 0 mismatches at q = 5, 7, 11; the arc
          table of q' uniform per position: 3, 15, 135); on the quiet zone the fraction of
          openings of length L holding a slot is 0.78-1.0 times the uniform |S_L|/q# at every
          length with >= 100 openings (V24: 0.081/0.100, 0.184/0.200, 0.290/0.300, 0.382/0.400
          at (5, 10^4)), the random-phase copy at 1.00, the slot position inside the opening
          uniform. OWNER'S "long openings always hold a slot" REFUTED with counts: slot-free
          openings at the ceiling 2,247 of 3,634 (q = 5, 10^4), 47,569 of 78,660 (10^5), 3 of
          14 (q = 7), 1 of 6 (q = 11). Pre-registration refuted the other way (V23): the
          opening spectrum on (Q, Q^2] EXCEEDS CRT by a factor rising with L (1.09, 1.19, 1.25,
          1.38 at (5, 10^4); to 4.2 at L = 10, q = 11) while the free-phase copy sits at 1.000
          at every L: phase zero makes the fuel the primes, denser than the CRT sieve by
          e^gamma/u at height Q^u and denser still at height y/s for members with air s,
          compounded over the block (Mertens / Buchstab, one line). Forced alignment: the
          first aligned opening of length 4 sits at 20,476 / 185,529 / 1,885,304 for Q = 10^3,
          10^4, 10^5, linear in Q, past q# q'; only its turn is pinned (20, 18, 18) just above
          the V22 threshold. The V12 adversary: aligned fraction uniform, first aligned
          opening of every length <= 4 in turn 61 (the first it did not cover); it is a
          machine on the fuel, not a manifold on the integers (1,809 openings over the run
          ceiling, longest 9), and "aligned" for it is a surviving pure-imprint pair, not a
          twin. ROOT: an opening holding a slot below Q^2 IS a twin (the cap), so "some
          opening holds a slot below Q^2" is the conjecture on the quiet zone restated; what
          the branch adds is localisation: twins sit in openings of length 1 in 92-95% of
          cases (404,635 of 440,107 at (5, 10^4)), in the ceiling length in 0.1-0.3%, with a
          flat per-pair slot rate (0.081, 0.092, 0.097, 0.095), and slot-holding long
          openings cannot exist below turn 12 without an ember.
        - R4.c.ii. The rich half of the record (spawned by unstick pass 3; reopens the
          ENGINE): Omega^full(n) with all gears for the pullbacks at m19..m53; sharpen E2;
          do records sit at the pullback's richest translates; does the arc floor
          (proofs/21) have a pullback form under the twisted separations. DONE, ROOT
          honestly (research/proof/rich_half.md, laws E4, E5; scripts research/anchor235/r68/;
          the 37 -> 41 positions addendum pending its scan). Omega^full(n) exact to n = 20 at
          nine rungs: only gears 5, 7 and 11 (by a unit) ever decide it; every gear above 7
          adds nothing on the corpus. The sharpened cap's slack is IDENTICAL to E3's (1, 1, 0,
          1, 1, 1, 1, 1, 0 at m19..m53; the two are the same count of the corridor's two tooth
          classes). Records sit at the richest translate at 4 of 7 rungs, not at 3 (deficits
          1, 1, 2 at 23 -> 29, 29 -> 31, 31 -> 37): the record trades a pullback opening for a
          long flank, the flank slot struck by gear 5 (four of four known positions to
          29 -> 31) or a single large gear; the start class is not a formula (26,208 classes
          at the maximum for 2 records per period: richness selects 1-2% of translates,
          poorness selects the record). E4 (PROVED, new): the arc floor is a theorem for ANY
          separations with arcs >= 2, which is why it holds on the real teeth, fails on random
          draws, and fails on the pullback's arc-1 gears exactly at L = 2 (0 exceptions in
          2,473,871); the twin collision (g + 4)/3 has no pullback form (the shared arc 3a =
          g -+ 1 is destroyed by q'^-1); E5 the rich direction's own +4 law. ROOT: Omega is a
          maximum over translates and every cap it yields is linear in T = F(M + q')/q'; the
          existence of a richest translate is a free-phase fact that says nothing about the
          poorness around it; the record is the poorest interval whose openings are on the
          teeth, a condition Omega does not see. Open on the part alone: whether gears >= 13
          ever lower Omega at n > 20 (cheap to n = 26); E4 and E5 into the kernel.
      First step when opened, owner's hybrid: a scratch lane with clean context (the three
      objects' definitions, the glossary, the ledger's proved laws; no clutch facts, no wall)
      defining interface objects on the wheels' coordinate and the motor's; a review lane
      sorting the 36 refiled clutch facts, the wall, theorem (E), the conjugacy and the
      family decomposition into interface objects, flagging every fact measured in the
      motor's coordinate; then reconcile (both = solid; scratch-only = new; review-only =
      suspect until reproduced; disagreements = predictions on record and the separating
      test). Each interface object is its own node with a definition, a law with proof, and
      a closed form where one exists; PARTIAL interfaces are kept. OWNER'S MECHANISM FOR THE
      CLUTCH (2026-09-06): the motor is the engine, pistons driving the prime gaps on an
      infinite camshaft (exact: the pattern repeats every q# columns and inside the window the
      gaps are the engine's alone, theorem (E)); the clutch is a VALVE TRAIN that takes the
      engine's combustion byproducts (squares and compounds of the engine's primes = the
      q-smooth numbers) off the engine, exiting through the wheels' manifold into the exhaust.
      Mapped exactly: in the quiet zone every wheels-open number is byproduct x one wheel
      prime, s P; each family (s, s') is a valve; the motor striking the cofactor s vents every
      family with s > 1; the family (1, 1), the only one with no byproduct, is what stays (the
      twin primes); the manifold's outlet is p_1^2, the exhaust's first non-redundant strike.
      Caution kept: the engine makes the byproducts and also does the venting (the motor
      acting inside the wheels' open set), not a separate mechanism. Engine terms adopted
      (glossary): combustion = a strike; fuel = the wheel prime P, air = the smooth cofactor
      s, charge = s P, burning when the motor strikes the air; family (1, 1) = the pure charge
      that never burns; back pressure = non-echo exhaust strikes (zero in the window, first at
      p_1^2); valve timing = the placement residue law; knocking = an unexplained symptom, the
      standing knock being the gluability anomaly (real teeth glue 2.4x the family).
    - R4.d. THE STACKED MACHINES BY SQUARES (owner's reconstruction, 2026-09-07: the manifold
      as built, all gears to q#, is a mess of interactions; rebuild it as a stack: machine 2 =
      the primes from q' to q'^2, run over [1, q#]; machine 3 = the primes from the next prime
      after q'^2 to its square; and so on until q# is reached; analyse each machine's square
      part and its band separately by the anchor's 30-cycles (open / closed / mixed), across q,
      find the rules, relate the machines, then connect to machine 1's cycles). Manager's two
      claims on the scorecard: the band structure (below its first gear's square a machine's
      strikes are home strikes and echoes, so on the band [g_k^2, g_{k+1}^2) exactly machines
      1..k act and a jointly open slot is a twin: the exhaust cap once per band) and the count
      (about log2(q / log q) machines: three at q = 17, 19, 23). BUILT, FACT with a ROOT mark
      (research/proof/stacked_squares.md, laws S1-S7; scripts research/stack/r1/,
      stacked_squares.py runs q = 23 in 11 s). THE BAND STRUCTURE (claim A) EXACT, 0
      exceptions at 23 machine instances and 16 bands (q = 5..23), once the square part is the
      half-open [1, g_k^2): every new strike below g_k^2 is a home strike, and on every band the
      slots open under machines 1..k are the twins slot by slot (296,672 + 296,783 + 296,350
      twins on band 3 at q = 23). S1: every prime's square is 1 or 19 mod 30, so g_k^2 always
      lands on a twin slot and is the machine's first genuine strike (the inclusive square
      part fails claim A at exactly one number per machine). COUNT (claim B): g_3 = 127, 173,
      293, 367, 541, 853 and g_4 = 134,699, 292,693, 727,613; banded machines 2, 2, 2, 3, 3, 3 at
      q = 7..23, = 1 + floor(log2(theta(q)/ln q')) exactly (a fourth at q = 41), plus one
      home-and-echo machine on top. S3 (single-gear law, proved): a gear >= 11 strikes at most
      one number per cycle, gear 7 two at j = 2 mod 7, no gear >= 7 closes a cycle alone; every
      closed cycle of machines 2 and 3 uses >= 3 distinct gears (minimum 3 in every 500-cycle
      sample, peak 5, up to 11); the engine closes with 2 at the gear-7 double. CLASSIFICATION:
      machine 2's square part closed at every cycle but 0-2 (25 of 27 at q = 23); its band
      open/closed 432/9,082 of 24,225 cycles at q = 23 against CRT 358/9,902; machine 3's band
      73,124/3,618,540 of 7,412,175 against CRT 117,752/3,107,139: the CRT expectation REFUTED
      in both directions. S4 (band composition, EXACT): on its band a machine's open numbers
      are exactly the g_k-smooth numbers and the g_k-smooth multiples of one prime >= g_{k+1}
      (asserted at every machine k >= 2), the prime part (30/8)/ln x and the smooth x prime part
      matching the measured decomposition to 0.01-0.025 per bin: prod(1 - 6/g) is the wrong
      expectation on a band. S5 (the half law, Mertens): every square-built machine has
      prod(1 - 1/p) = 0.502-0.524 (machine 2) and 0.5016-0.5026 (machine 3), so all have the
      same CRT classification (1 : 27 : 36)/64: the construction is the EQUAL-WEIGHT
      decomposition of the manifold. S7 (cross-machine, EXACT): the gears of machine k + 1 are
      exactly the primes of band k, and the twins of band k are exactly the double-home slots
      of machine k + 1 (0 mismatches at all 16 bands); band starts chain by nextprime(.)^2;
      closed positions of k + 1 bear no relation to those of k through squares (rates 0.50,
      0.66, 0.52 against densities 0.58, 0.59, 0.49). S6 (joint-open ratios): on band k the
      lower machines leave the g_k-rough pairs and machine k strikes exactly the rough
      composites, so the ratio twins / product of open fractions is P(prime | rough) / P(open
      under k): 1.9-5.0 at the band's start (4.95 in the first bin of band 3 at q = 23),
      crossing 1 between u = 2.75 and 3.25, 0.81-0.87 at u = 4 (e^gamma/2 per number);
      whole-band 1.29 to 0.88 on band 2 (q = 7..23), 4.32, 3.50, 2.08 on band 3. THE CYCLE AT
      q# (claim C) REFUTED for machines k >= 2: open under the engine only for q <= 7 (q# + 11
      divisible by 11 from q = 11; the slot (q# + 29, q# + 31) always open), and under machines
      k >= 2 open at 0 of 16 instances (7 mixed, 9 closed). ROOT: the one route-shaped
      statement, "every band holds a twin", is twin-Bertrand between consecutive squares of
      the chain. Pre-registration misses recorded (P3 "never closed", P5's profile, P6's
      interval).
      - R4.d.i. The base case and the step (owner, 2026-09-07: "definitely worth pursuing the
        base case; the smallest machine that builds a twin pair is 2, 3"). The chain's links
        are the square intervals [g, g^2) along g -> nextprime(g^2); [2, 4) has no pair at
        distance 2, [3, 9) has (3, 5) and (5, 7), so the base machine is {2, 3, 5, 7}; by S7
        the step is "machines 1..k, each with twin gears, leave an open slot on band k".
        Branch: the chain from the base and other bases, every link's twin count; how the
        lower machines' twin gears act on the band (the collision law's role); the
        start-of-band excess (up to 5x independence just above g_k^2) and its mechanism (only
        machine k's gears up to x / g_k have acted at height x); the first twin above g_k^2 as
        the base-case quantity. OPEN, prover running on Fable (research/proof/base_and_step.md).
    - R4.a. The two machines and the clutch, built exactly at q = 11..23 (research/proof/
      period_scale.md). FACT, exact; the reframing is confirmed and, at these sizes, opens no
      route; PARKED here per the owner (after the window). Level of distribution 1 exact: max
      |strikes - 2N/g| = 1.7, 5.4, 7.8, 14.3, 25.2 at q = 11..23 against the 3^m bound (0
      exceptions in 2,338 cells; pairs 0 exceptions in 19,956 with gh < P; the true error grows
      like 2^m). Twisted copies exact (4,676 copies, 17 million cofactors, 0 mismatches): the
      top machine's action is a union of coherent copies of the bottom at separation 2/g. THE
      CLUTCH'S FOUR CELLS at q = 23: both open 895,791; bottom-open/top-closed 7,056,384;
      bottom-closed/top-open 4,150,311; both closed 25,079,659; all the coupling is in the
      both-open cell (0.83-1.02 of independence), the other three within 5% of independence.
      THE WINDOW IS THE CLUTCH'S ZERO-INTERACTION REGION: the first proper kill of a
      bottom-open column lies above the window top at every q (2.4e-6 of the period at
      q = 23), the only region where one machine decides alone. Twins = both-open plus
      home-only exactly (0 mismatches in 38.9 million columns); the home-only cell IS the
      doubly occupied placements. PLACEMENT RESIDUE LAW (new, exact, 25 machine-gear pairs):
      home columns meet each non-tooth class of a bottom gear exactly twice and each of its
      two tooth classes once, so placement density is exactly prod(1 - 1/(h - 1)): placement
      is dimension 1, double occupancy dimension 2, and that step is the parity barrier,
      named. ORIGIN LAW: both machines are mirror-symmetric about column 0 (0 mismatches);
      from q = 19 the longest both-open stretch of the whole period starts at column 0 (520,
      2,523 columns) and ends at the first twin above the range. The twin-free record is a
      joint object (1.9-3.7 times the sum of the two machines' own closed records), made by the
      top machine covering the bottom's ordinary leftovers (bottom closed at 1.005 of its
      average inside those stretches). Survivor curve: 1.000 at s >= 4.27, minimum 0.86 at
      s = 2.1, 0.79305 (1 + c / ln Z) at s = 2. EXACTNESS BUYS NOTHING: the order-2 Brun error
      equals the order-3 term at 100% .. 78%, identical to the generic Bonferroni number; face
      A is not "the error terms are too big" but "the main terms alternate and do not converge
      at s = 2", visible with every other obstruction removed. Switching gives an identity
      (E = 2D + Q exact at all five q), no Chen-type asymmetry. The placement question is
      proved equivalent to the twin prime conjecture and is weaker than the window statement.


    - R2.e. Location inside the window, lower machine only (the owner's round, 2026-09-06: one
      more round to pinpoint location; if it does not close, construct the top machine).
      - R2.e.i. The position-length frontier (research/proof/position_frontier.md). STRONG:
        a proved reduction of the window statement to one run. THEOREM (E), proved in a line:
        for every column with 6k - 1 > q, blocked under {5..q} iff blocked under
        {5..floor(sqrt(6k + 1))} (the cofactor's least prime factor), so the effective machine
        at a column is exact, and R_min(L) >= ceil((y_L^2 - 1)/6) - L + 1: an induction on the
        machine, unconditional for stretches whose top member is below 59^2 = 3481 (the
        certified ladder), conditional on the ladder beyond. Measured: R_min(L) = 1 for every
        L < d_0 and R_min(L) >= 3.25 L for every L >= d_0, 0 exceptions in 113 period cells
        (m7..m29) and 8,375 window cells (q = 23..19,997); the frontier is bimodal with nothing
        between; the ladder delivers c = 1.25 (1.54 for L >= 6) unconditionally, and c = 3 would
        need F(y) <= y^2/24 (refuted at y = 5, 7, 11): the frontier constant and the record
        constant are one number in two coordinates. The induction's failing step is NOT the
        big-gear fusions at the top of the window (zero: at 23 frontier stretches including
        q = 997's 241-column window record there are 0 columns without an effective striker
        and the effective machine leaves each stretch in one piece); the exception set of (E)
        is exactly the TWIN GEAR PAIRS striking their own home columns (7 of 7 count matches),
        all below (q + 1)/6. MAIN FINDING: from q = 1427 on the longest blocked run of the whole
        prefix [1, W] IS THE INITIAL RUN from column 1 (2,038 of 2,038 rungs, 0 exceptions);
        the Pareto staircase of the prefix collapses to the single point (1, d_0 - 1);
        (d_0 - 1)/(q/6) in [0.97, 1.35], median 1.005: the initial run is q/6, linear, while
        the window record grows like log^2 q. TOWARD THE ROOT: the window statement is
        R_min(W) > 1, i.e. "the exception set stops below W", and the exception set is
        {L < d_0}, so it reads d_0 <= W exactly; all of [q/6, W] is provably removed from
        suspicion (x >= 1.25 L), leaving one run: the initial run, the diagonal walk of the
        bottom machine, the first twin above q. Real-teeth, both: c at the 85th-90th
        percentile of the family, and the column-1 anomaly reproduced by 0 of 60 members
        (real/family-median initial run 4.35, 10.6, 14.1 at q = 211, 401, 997, growing). Also
        exact: the mirror R_max = P - R_min - L + 1 (88 of 88); the same low columns 13, 53, 59,
        111 serve every machine (column 111, the twin gap 661 -> 809, the tight point from m19
        on). VERDICT: location pinpointed. The window can be emptied only from the bottom; the
        top machine is irrelevant inside the window (E); what remains is d_0 <= W, the first
        twin above q, decided by the bottom machine's diagonal alone.

      - R2.e.ii. Structured families of slots (research/proof/structured_families.md). DEAD
        as a route, by an identity. Every located family carries twins at the window's own
        rate: on 661 disjoint sections (130,644 twins) the normalised excess is within 1.2
        sigma of 1.000 for every family (islands 1.006 +- 0.005 against the rest of the
        corridor; the tree's candidate object is indistinguishable from ordinary corridor
        columns). The common value is the s = 2 handicap e^{2 gamma}/4 = 0.793, flat across q.
        MECHANISM: the singular series of any non-tooth residue family is 12 C_2 divided by the
        small-gear factor the family "saves", exactly cancelling it: a location rule cancels
        itself; survivors(F)/survivors(window) = density(F)/prod_{5..y}(1 - 2/g) <= 1 with
        equality only for the whole opening set. Emptiness is Poisson at the fair rate (nine
        families, observed/expected empty sections 0.87..1.04; the last empty rung is where the
        expected count crosses 1). Thickness: the window is at s = 1.79 at q = 4999, every proper
        family lower; the dimension-2 sieve needs n >= 6.2e15 against W = 4.2e6. Per-gear takes
        are one curve belonging to the RANGE, not the family (all families agree to 2% in every
        bin; the island family sits on the anchor family's curve; no gear takes less than 2/g on
        any family). NEW EXCEPTIONLESS LAW (661 of 661): gear 5 is barred at the column-0 offset
        for every q, and gear 7 is barred there iff q = +-2 (mod 7), so the always-open column of
        node 3 is a B = 7 island family on exactly one third of rungs (the first link between
        node 3 and R2.a.i.a); k_0 mod 35 takes only six values and the island set meets each,
        never emptily. Corrections: windows at consecutive rungs overlap almost totally, so
        pooled sigmas are false; two of the brief's families hold every twin of the window by
        identity and are not location rules. A family defined by residues modulo the lower
        machine's period can never beat existence, because that period is invertible modulo
        every gear above it; only a rule whose definition involves the gears above could.

  - **R3. Structure of the record: how a record stretch is made.** If what makes a record is
    understood, the object that survives it may be nameable. Spawned by the tiling observation
    (out of 1a).
    - **4. Genealogy (records recruit runner-ups).** WEAK: exact at 8 steps (ancestor a runner-up
      by 2-14, largest gap merged one level down 7 of 8, 1-5 generations), no rule stated; the
      theory "bounded branching bounds growth" is untested.
      - 4.i. The merge forest, exact (research/proof/merge_forest.md; the toolbox pass). FACT,
        exact, with one new identity and one new object; not a route. BRANCHING IDENTITY
        (proved, 8 rungs, 0 exceptions): the order distribution of a rung is the second
        difference of its chain-count sequence, n_J = C_{J-1} - 2 C_J + C_{J+1}, with C_0 = q' N,
        C_1 = 2N, C_2 = 2 A_0 + A_d a closed form in the old spectrum by residue class; at
        rung 31 the order distribution of 6.2 billion gaps (5,805,160,589 / 413,380,422 /
        7,999,018 / 12,992 / 4) comes out of five numbers; max order = 1 + D_q' at 8 of 8. The
        mean order is teeth-free, exactly q'/(q' - 2) at 8 real rungs and 21 of 21 family
        members; only the tail depends on the teeth, and the real machine makes a third as
        many triple fusions as typical (percentile 0.20) because its letters are never common
        gap sizes. NODE 4 REFUTED: branching is bounded and bounds nothing (J_max x max piece
        fraction = 2.4, 2.0, 3.5 at m23, m29, m31). NEW OBJECT, THE FRONTIER: F(M + q') =
        max over old sizes a of (a + Rest(a)) exactly, where Rest(a) is the most a gap of size
        a gains by fusion; the record is made at an interior a / F_old = 0.60, 0.68, 0.58-0.70
        at the top three rungs, and Rest(F_old) = 3, 2, 3, 7, 7, 5, 5, 2 (falling): a gap that
        swallows the old record whole gains at most 2 columns at m31. The strengthening
        Rest(a) <= q' would give the budget inequality in one line; it holds at rungs 7..29
        and FAILS at 29 -> 31 (max rest 34 > 31). Record lineages: orders 3, 2, 2, 3, 2, 4, 3,
        3; the largest piece has depth 0 at 7 of 8 rungs, is a record of its own rung at 7,
        11, 17, 19 and not from 23 on, mass rank above 0.99 from rung 17 (ordinary in size,
        extreme in rarity); the top is not closed (big gaps built from the old top third fall
        0.89 -> 0.33); 1,336 distinct full lineages at m23, the record's one of 59,940.
        Order theory: ancestors at every layer are a contiguous run, so the ladder is the
        laminar family of column intervals (the exact m31 forest in 33 s). CHILD NAMED: why
        Rest(a) collapses as a -> F(M): 5b's repulsion at the top of the spectrum with the
        chain law attached.
        - 4.i.b. The branching identity, proved and pushed (research/proof/
          branching_identity.md; the owner flagged the identity). PROVED in general: n_J =
          C_{J-1} - 2 C_J + C_{J+1} with C_0 = q' N, by a two-line run-length inversion with
          the cyclic boundary handled exactly (proviso: some gap is not 0 or +-d mod q', strict
          at all 8 rungs). NEW AND PROVED: C_r = W_{r-1} + Z_{r-1}, the old machine's legal-word
          count plus its all-pad count, at every depth (the factor 2 in C_1 = 2N and
          C_2 = 2 A_0 + A_d is the all-pad term); C_r = 0 for r > L + 1; max order = L + 2 =
          J_max (joining the forest to docs/proofs/10); tail form sum_{J' >= J} n_{J'} = C_{J-1}
          - C_J, so the merge fraction is an identity with exact deficit C_2 / ((q' - 2) N).
          Verified 8 of 8 rungs by three routes; m31's order distribution of 6.23 billion gaps
          is C_1..C_5 = 429,417,450 / 8,025,014 / 13,000 / 4 / 0. SECOND MOMENT, closed form,
          proved: Var = 2[(q' - 4) + S (q' - 2)] / (q' - 2)^2 with S = sum_{r >= 2} C_r / N;
          exact at 8 rungs and 42 of 42 family member-rungs; the teeth live in the variance and
          nowhere lower (real machine at percentile 0.19 of the family, identical to its n_3
          percentile since the variance is affine in S). SIZE SIDE, proved: m_{M+q'}(v) = sum_J
          sum over J-windows of span v of eps_J, eps_J in {0, 1, 2} (q' - 4, q' - 3, q' - 2 at
          J = 1) a function of the window's letters alone; reproduces every multiplicity at 7
          rungs and every m31 gate; least depth = J_max, sharp (3, 2, 3, 3, 3, 4, 3, 5), with
          truncation error C_K - C_{K+1} in closed form. CLOSURE, proved: the depth-m dictionary
          of M + q' with multiplicity is determined by the depth-K_m window dictionary of M,
          K_m <= m J_max, and K_m tracks J_max, not the rung (K_2 = 4, 4, 4, 4, 6, 5): not an
          open hierarchy, bounded iff L is bounded (the known open rider). NOT A ROUTE, as
          pre-registered: finite depth but not finite state (dictionary sizes 41, 730, 7,184,
          45,854, 208,668, 720,527 at m29 by depth); F(M + q') = max_J Q*_J(M) (at 29 -> 31: 43,
          55, 58, 55, 55 against budget 74) and nothing bounds a dictionary's extremes by its
          predecessor's; the residual is the chain statement, unchanged. Prior art handled in a
          line each (paired-Holt recursion reproduced; renewal ladder cited; dictionary-
          monotonicity-onset distinguished: it uses the set, this uses the multiset of realised
          windows and is exact). VERDICT CORRECTED (owner, 2026-09-06: "not a route is a bold
          claim"): the closure is ROUTE-SHAPED, a deterministic finite-depth recursion on exact
          objects whose extremes are the records; what is missing is an inequality on its
          extremes, and "nothing here bounds it" is a brick, not a verdict. Two children: the
          recursion as an instrument for the ladder past the scan wall (exact records beyond
          F(59), each a new test of the budget), and the search for a monotone or contracting
          functional of the recursion (legal-word density per opening by depth, all-pad
          density, the order variance), the place a bound on the extremes would live.
          - 4.i.b.i. The ladder past the wall by the closure (research/proof/ladder_closure.md).
            INSTRUMENT, exact, reaching two rungs no scan can: F(37) = 88 and F(41) = 91 from
            m23's period alone (the ladder m23 -> m29 -> m31 -> m37 in 195 s from a 3.9-million-
            row dictionary; m41 by a span-threshold prune in 1,992 s and 2.4 GB), with every
            gate exact (F_j rows, F_2(37) = 90, F_3(37) = 97, opening counts and periods,
            multiplicities, m37's thirteen spectral holes, n_J and Q*_J digit for digit, L and
            L_pad). New numbers: n_J(31 -> 37) = 205,591,124,261 / 12,223,428,142 / 114,732,724
            / 70,532 / 216 and Q*_J = 58, 68, 85, 88, 68; n_J(37 -> 41) = 8,065,074,943,615 /
            432,481,162,322 / 1,688,770,136 / 3,052 / 0 and Q*_J = 88, 90, 90, 91: only 3,052 of
            m41's 8.5 trillion gaps are fourfold fusions and the record is one of them
            (confirmed twice). The record's composition at every rung: flank + alternating
            legal word + flank (witnesses (23, 10, 10), (18, 10, 30), (11, 12, 37, 28)).
            K_m tracks J_max, not the rung, over machines from 7.9 million to 8.5 trillion gaps;
            the refined depth bound K_m - m <= 2(J_max - 1) fails at 23 -> 29. NEW TOOL WITH A
            LEMMA: the span-threshold prune (a window spanning less than theta has no sub-run
            spanning theta; theta = F(M) + 1 is free since F(M + q') >= F_2(M)); at m41 the
            pruned dictionary is 186 windows on 2,656 openings and holds the record. THE WALL,
            measured: a rung spends 2 to 6 depth and returns 3 to 5; F(43) needs a depth-18
            dictionary of about 10^8 rows (stopped at 3 GB). BUDGET SLACK along the extended
            ladder: 14, 20, 16, 7, 38 at 23 -> 29 .. 37 -> 41, not monotone; the narrow rung
            37 -> 41 (slack 7) is where the record's depth climbs from 3 to 4. Truncation is a
            lower-bound instrument only (depth-3 floor 90 against 91). Child named: the Q*_J
            peak sits at J <= 4 at all five rungs, including both where J_max = 5.
          - 4.i.b.ii. Monotone functionals of the recursion (research/proof/
            monotone_functional.md, 980 lines; scripts research/anchor235/r66/). STRONG,
            with the engine's gate items 2 and 3 answered and item 1 open. THE ORDER LAW
            (measured, 0 exceptions): the order of interaction the budget needs is exactly
            k* = L(M) + 1 = J_max - 1, where B_k = the widest span of a level-k admissible
            fusing word, a functional of the k-window table D_k(M) alone that bounds F(M + q')
            for every k: B_{L+1} <= F + q' at 9 of 9 computable rungs (margins 3, 6, 9, 7, 7,
            13, 3, 16, 7) and B_L > F + q' at 7 of 7 rungs from 13 -> 17 (over by 5, 12, 8, 34,
            11, 23, 32); mechanism: the binding term is always the deepest fusion J = J_max (7
            of 7); at order L two de Bruijn steps are free and buy the record twice (23 -> 29
            extremal word (34, 29, 34) at phase 5, span 97 against budget 63, a word m23 does
            not contain), at order L + 1 one step is free and the maximum lands inside; not an
            artefact (a second ladder from K_0 = 16 gives identical dictionaries and B_k);
            37 -> 41 undecided (m37 sticks at depth 2, B_2 = 161 > 129 so k* >= 3, the law
            predicts 3). Payoff: the budget at 29 -> 31 is a statement about 45,854 rows
            standing for 214,708,725 gaps. THE MONOTONE ONE: Phi = B_{L+1}, the only candidate
            of eleven that both bounds the next record (9 of 9, overshoot 1, 3, 0, 3, 5, 1, 17,
            0, 0) and is budget-monotone (8 of 8, increments 4, 1, 10, 9, 5, 25, -2, 30, each
            <= q'); its theorem, B_{L+1}(M; q') <= F(M) + q' for all M, implies the budget at
            every rung, is strictly stronger at 4 of 6 decisive rungs, and its state is a
            10^2 to 10^5 row table, not the machine; what remains: a cap on L, and one lemma
            (the level-(J_max - 1) relaxed deepest fusion is within budget; smallest instance
            at 23 -> 29: realised 2-windows (a, b), (b, c) with b a letter give a + b + c <=
            F + q', tightest witness (25, 10, 25) at phase 4, 60 <= 63). EVERY OTHER CANDIDATE
            DIES with its rung and merge: F_J at 19 -> 23, J = 6 (exact identity F_J(m23) =
            F_{J+t}(m19), t = 3, 3, 4, 4, 4, 4, so 77 - 50 = 27 > 23); top-3 sum at 13 -> 17;
            excess over threshold by the identity (a + b - x)_+ >= (a - x)_+ + (b - x)_+;
            J_max non-monotone; the excursion at 13 -> 17 (+33.06 against 17); the
            letter-floor discount = the pair statement; W_1/N rises at five rungs. Kept: the
            second-largest realised value is budget-monotone at 9 of 9 (true, not implied,
            useless as a bound). GATE ITEM 1, L(M) BOUNDED: OPEN. L_bare <= 5 proved (measured
            1, 0, 1, 1, 1, 2, 1, 3, 3, 1); L_pad is the open half (0..2 here, 2, 2, 3, 3 in the
            corpus above) and nothing on this instrument caps it (the PAD alphabet grows with
            the machine); new: the order law gives L a job, a cap L <= c turns the budget into
            a uniform bounded-order statement about D_{c+1}. GATE ITEM 2, THE BAND: PASSES. At
            29 -> 31 the complete enumeration of D_4(m29) = 45,854 rows carrying all
            214,708,725 gaps (loss 0, fusion masses n_2, n_3, n_4 = 413,380,422 / 7,999,018 /
            12,992 digit for digit): max span with largest piece in [15, 36] is 58 at J = 3
            (witnesses (18, 10, 30), (23, 10, 25)) and 55 at J = 4 against F(m29) + 31 = 74,
            margin 16, the pre-registered value; the extremal chain pays exactly the letter
            floor a_L = 10 in the middle (19 of 21) and alternates +-d at J = 4; the largest
            old piece falls with chain depth (43, 35, 22 at J = 2, 3, 4). Scorecard: M1-M6, M8
            confirmed, M7 confirmed with the collapse located.
            - 4.i.b.ii.a. Cap the padded word (spawned by the order law giving L a job: a cap
              L <= c makes the budget a bounded-order statement about D_{c+1}; the engine's
              last structural item). The PAD alphabet at every rung, the junction-gear
              counting cap, the letter-value cap, the shadow if uncapped. STRONG, the
              engine's last structural item resolved (research/proof/pad_cap.md, laws E1-E3;
              scripts research/anchor235/r67/). E1 (PROVED, uniform): every realised legal
              word over the small alphabet {a, b, q'} has length <= CORRCAP_3(q' mod 210) <= 8
              (the bare cap of docs/proofs/12 with the pad q' added; finite at 48 of 48
              classes, values 2, 3, 4, 5, 6, 8, never 7); 0 exceptions on 14 rungs (m5..m37
              computed, m41..m53 corpus), tight at m29 ((10, 21, 10)) and m53. Every padded
              word the record ever measured (L_pad = 0..3 to m47) is in that class, so the
              padded half AS MEASURED is closed; E1 explains why the eight 3-words over {20,
              39, 59} were zero (CORRCAP_3(59) = 2). THE OPEN HALF IS THE SKIP HALF (letters
              2q', a + q', b + q'; first instance (20, 98, 20) at m53) AND IT IS ROOT: E2
              (proved) L + 1 <= 2 Omega(T + 1), T = floor((F(M + q') - 2)/q'), tight per class
              at m29 and m53; E3, the exact corridor + span cap, 9 of 9 rungs with slack 0 or
              1; none uniform: a constant cap on L_skip is equivalent to F(M + q') <= C q',
              stronger than the budget. Regime (proved, gear 5 alone): L <= 5 whenever
              F(M + q') <= 5q' + 1, i.e. the whole corpus (F/q' <= 2.73). MECHANISM: a word's
              openings lie on two tooth progressions of step q'; in units of q' each class is
              a set of openings of the pullback machine (same gears, separations 2 u_g q'^-1),
              holding at most 15/35 of the multipliers under gears 5 and 7; the small alphabet
              needs the two classes to cover every multiplier (2 x 15/35 < 1: capped), a skip
              letter leaves a multiplier to neither class and the contradiction disappears;
              gear 5 alone does not cap, gear 7 does. P3 confirmed at m37: the first rung
              where a skip 2-word is size-feasible, none of the four candidates is realised in
              the complete D_2(m37) though the corridor allows each. Exact tables: the PAD
              alphabet {23}, {29}, {31}, {37, 49}, {41, 55, 68} at m19..m37 (holes 41, 82
              corridor-allowed); every realised legal word to m37 with counts. REFUTED: the
              counting cap (capacity/need 1.4-2.2); k_L = L + 1 fails at m29 (the palindrome
              (10, 21, 10) cannot chain), replaced by k_L in {L, L + 1}; the merge forest's
              depth, the record's composition and the gear-5 lock are not the shadow. THE
              SHADOW: the record in gear units, |A_pad| ~ 3 F(M)/q' - 2 for letters and
              T = F(M + q')/q' for words. Open on the part alone (not structural): the skip
              words at m41, m43, m47 by a copy-law scan; the full-gear Omega for the pullback
              separations; formalising E1 (BareAlt plus one letter) and E2.
        - 4.i.a. The frontier's collapse at the top (research/proof/frontier_collapse.md).
          FACT, exact; a partial route. THE TOP LAW (8 of 8 rungs): Rest(F_old) = N(F_old) if
          F_old = 0 or +-d (mod q'), else n1(F_old), the old record's largest single neighbour;
          so the top's slack s(F_old) = q' - Rest(F_old) is an identity (4, 9, 10, 10, 12, 18,
          24, 29). Mechanism: the letter floor a_L = q'/3 colliding with neighbour shortness: a
          second junction forces the added piece to be an interior gap, hence = 0, +-d mod q'
          and >= a_L, and the old record's neighbours are 2, 2, 3, 5, 7, 5, 5, 2 against
          a_L = 2, 4, 4, 6, 6, 8, 10, 10. THE FUSION-RATE IDENTITY (proved from file 05; 137
          cells): an old gap is fused in exactly 4, 3 or 2 of the q' copies (generic / +-d / 0
          mod q') and is an interior piece in 0, 1, 2; junction availability never collapses,
          only piece size does. THE SEPARATING QUANTITY: occurrences with a letter-sized
          neighbour, 1,858 .. 32 in the interior against 0 at a = F_old at every rung; the
          availability gate "not legal and has(a) = 0 implies no J >= 3 fusion" verified (40
          cells). The collapse is a rarity factor 2.3 times a suppression factor 1.6-2.3.
          Attaining fusions: a >= 0.9 F_old gives J = 2 except exactly when a is a letter
          (15 of 15); the interior band is J >= 3 at 26 of 39 cells, all with letter interiors.
          The slack profile s(a) is a saw, not a valley (5-7 strict local minima; 63 of 63
          family members non-convex): no convexity argument; a proof must case-split on
          availability. THE 29 -> 31 FAILURE of Rest <= q': at a = 21, size 55, order 5, the
          recorded Q*_5 maximiser (7, 10, 21, 10, 7), not the (18, 10, 30) run; missed by 3;
          Rest <= q' + 3 holds 8 of 8 (tight once) and Rest <= q' for a >= 0.605 F_old holds
          8 of 8. CORRECTION: N(v) <= F_2 does not bound Rest(a) (7 cells with Rest > N, five
          at J = 3 with a at an end); what is an identity is max_a (a + Rest_2(a)) = F_2(M).
          PLACEMENT: the frontier splits along the attainment identity, the J = 2 half IS the
          pair statement and the J >= 3 half IS the chain statement; the collapse at the top
          lives in the pair half; at m31 the deep-chain cap covers a <= 14, the top bound
          a >= 26, and the uncovered band [15, 25] contains the record maximiser a = 25; Rest
          is not monotone. Family: all three recorded budget violators reproduced plus two
          more found; 5 of 5 break at an interior-legal a; two also break at a = F_old with
          Rest(F_old) = 21 and 14 against the real machine's worst of 7. CHILD NAMED: the
          availability gate has(a) > 0, for which old sizes a some occurrence has a neighbour
          of size = 0, +-d mod q' and >= a_L: a statement about M alone, monotone at the top
          where Rest is not, and once it is 0 the frontier is the pair statement.
          - 4.i.a.i. The availability gate (research/proof/availability_gate.md). FACT, a
            partial route, empty as a bound. The gate closes at about four fifths of F_old and
            drifts up (a_gate / F_old = 1.00, 0.00, 0.57, 1.00, 0.72, 0.92, 0.85, 0.81 at the
            rungs to 31); it splits into two branches, a legal (a is itself a letter: uncapped,
            hence a_gate = F_old at rungs 7 and 17) and a legal neighbour (capped). THE GATE
            LADDER (proved, 137 cells): J = 3 needs a legal or hasM(a) > 0; J >= 4 needs
            hasM(a) > 0 always. THE GATE IS ONE ROW OF THE LEVEL-2 DICTIONARY: a_hasM = the
            largest a such that (a, a_L) is an adjacent pair of M (7 of 8 rungs; at that a the
            only legal neighbour occurring is the short letter, 6 of 6), with the closed form
            a_hasM <= F_2(M) - a_L proved in one line (68 of 68 including 60 family members;
            deficits 0, 3, 7, 3, 7, 3, 4, 10). THE DECIDING NEGATIVE: that cap helps only when
            F_2 - F < a_L, and F_2 - F = 2, 2, 4, 5, 7, 6, 5, 12 against a_L = 2, 4, 4, 6, 6, 8,
            10, 10 fails at 17 -> 19 and 29 -> 31, exactly where the F_2 cap failed: one
            obstruction, two routes. Three regimes: above the gate only the pair statement
            (slack 5..24); between a_hasM and a_gate only J = 3 at legal middles (a + N(a) <=
            F_old + q', 5 of 5); below, the chain statement. The band between the deep-chain
            cap and the gate is empty to rung 19, then [13, 20], [22, 25], [15, 35]; a* inside
            at 3 of 3; its minimum slack is the global budget slack; it does not shrink. Family:
            the real a_gate at or below the family median; all five recorded budget violators
            break with the gate OPEN (5 of 5). Also exceptionless: hasL = hasR (mirror); a_gate
            = 0 iff the rung's max merge order is 2; has2(21) = 4 at m31 = the number of
            order-5 gaps (the four (7, 10, 21, 10, 7) palindromes). RESIDUAL, exactly: the row
            v = a_L of the level-2 dictionary of M is empty above c F(M) with c < 1: a finite,
            M-only statement about one gap size (a gap of the new gear's short-letter size
            never sits next to a gap above c F); the LP lane already certifies cells of that
            dictionary at m19, m23.
            - 4.i.a.i.a. The short-letter row (research/proof/short_letter_row.md). STRONG.
              The parent's residual is PROVED scan-free at the three rungs that matter, by LP
              duality: the row (a, a_L) of the adjacent-pair dictionary is empty above its
              realised top at m19 (above 20), m23 (above 25), m29 (above 35; 270,070 exact
              operations), and the vehicle refuses at exactly the realised top of each row; at
              29 -> 31 that is a bound the proved pair cap F_2 - a_L = 45 > 43 = F could not
              give. r(a_L)/F = 1.00, 0.00, 0.43, 0.64, 0.67, 0.80, 0.74, 0.81, 0.79 at rungs
              5 -> 7 .. 31 -> 37, flat at four fifths over the top five rungs. NEW MEASURED LAW,
              THE PINNED LETTER: F(M) <= a_L + r(a_L) <= F(M) + 3 at 8 of 8 rungs (excess 2, 0,
              2, 0, 3, 1, 2, 0), confirmed OUT OF SAMPLE at 31 -> 37 on the full 33.4-billion-
              column period after pre-registering 43 <= r(12) <= 49: r(12) = 46 and
              a_L + r(a_L) = 58 = F exactly. Not a property of a general size (v + r(v) exceeds
              F + 3 at 23 of 41 sizes at m29). PROVED MECHANISMS: the pair filter (three
              translates of a two-tooth set cover at most 6 residues, so only gear 5 can forbid
              a neighbour class; exactly 6 of 25 classes (a, v) mod 5 impossible; 0 of 872
              realised pairs violate; it is the LP vehicle's zeroth-order dead clause); the
              closer law ({3 a_L - 1, 3 a_L + 1} = {q', q' -+ 2}, so the only gears that can
              close an a_L-gap of M are q' and, when q' = 2 mod 3 and q' + 2 is prime, q' + 2;
              9 of 9 rungs); the row height separates on c_5(a_L) at 9 of 9 (c_5 = 4 gives
              ratio <= 0.667, c_5 <= 3 gives >= 0.735). Twin rungs decide the closers, not the
              row. THE DECIDING NEGATIVE: the residue obstruction explains 0, 1, 0, 3, 0, 0, 0,
              1 of a deficit running 0, 3, 3, 3, 3, 4, 8, 9 and growing; a per-letter CRT
              enumeration cannot do better (the product of allowed classes stays positive once
              gear 5 is passed): the row's emptiness is a per-machine covering fact, which is
              why the certificates are the answer. The band [15, 35] at 29 -> 31 is now a
              certified object; the pinned letter would give [15, 36] from a formula. Handed
              forward: prove the pinned letter (first test: the family at m17/m19/m23), or find
              a form for the dual weights of the cell (a, a_L) that survives the machine.
              - 4.i.a.i.a.1. The pinned letter (research/proof/pinned_letter.md). UNPROVED,
                and it is a REAL-TEETH law: only 43 of 63 tooth-counterfactual members obey
                0 <= E(a_L) <= 3 (range -6 .. +7), and every step of the glue construction is
                tooth-invariant, so no glue argument can prove the constant 3; a proof must use
                u_g = 6^-1 mod g and 3 a_L = q' -+ 1. The generalisation to uncoupled sizes is
                REFUTED under all three readings (worst: v = 20 at {5..29}, Leg(20) = {59, 61}
                disjoint from M, yet v + r(v) = 55 = F_2 = F + 12); the letter is chain-law-
                coupled at 7 of 8 rungs (by gear 5 in the pad at the tight rungs), so the law
                is not an instance of uncoupledness. PROVED, new: THE SPARE-GEAR LEMMA - if a
                2-run has a gear that is neither obstructed at the middle opening nor a sole
                striker inside the run, then F(M) >= a + v (0 counterexamples in 13,616 runs;
                133 of 133 runs of span above F have no free gear); its contrapositive: the
                excess E(v) > 0 is equivalent to "no free gear", the price of buying the middle
                opening from a gear already carrying a flank column alone. EXCEPTIONLESS: at the
                attaining 2-run of every realised size, every gear of M is a sole striker of
                some interior column (90 of 90 over five machines): L4 extended to the whole
                r(v) profile. The one-gear glue fails where it matters (losses up to 7;
                re-phasing depth 1, 1, 1, 2, 4, unbounded). The gate closes at F + 3 - a_L
                (3, 4, 6, 8, 15, 20, 27, 36), one above the certified 35 at 29 -> 31; band
                [15, 36]. Twin rungs: the law holds at 4 of 4 with the smallest excesses; the
                closing gear is never q' - 2. The record is not a depth-2 function of M's
                dictionary (short by 1..21 at 8 of 8); the 3-run form max_l (l + N(l)) hits it
                at 5 of 8 and gives 85 against 88 at 31 -> 37.
                - 4.i.a.i.a.1.a. The pinned letter's arithmetic (research/proof/
                  pinned_arithmetic.md). THE LAW IS HALF REFUTED out of sample: at 37 -> 41
                  (M = {5..37}, period 1.24e12, never scanned) the exact CRT search gives
                  r(14) = 63, so a_L + r(a_L) = 77 against F = 88; the lower half fails by 11
                  (witness verified at column 470,382,204,623). The upper half
                  a_L + r(a_L) <= F + 3 survives 9 of 9 rungs with slack 3, 1, 3, 0, 2, 1, 3, 14:
                  the constant 3 was an artefact of the scannable machines. THE REAL-TEETH
                  INPUT IS ONE COORDINATE: v = (3v) d_g (mod g) for every distance and gear
                  (1,200 of 1,200 cells) because d_g = 3^-1, i.e. in the coordinate n = 6k every
                  gear's teeth sit at +-1 (the members n -+ 1); the letter's content is only
                  the integer 6 a_L = 2(q' + eps). (This is the owner's raw-line view of the
                  top machine, met from the bottom's side.) FOUR EXCEPTIONLESS LAWS: the
                  forced-gear law (g strikes inside EVERY a_L-gap iff 2g <= q' + eps + eps_g + 3:
                  the bottom half of the machine cannot avoid the letter gap; 45 of 45); the
                  forced-cover count (forced minima sum to exactly a_L - 1 at 6 of 9 rungs);
                  the twin-partner law (the partner strikes at most one interior column, 50 of
                  50); D3 (at most three gears can ever be obstructed at a letter run, 5,400
                  cells). THE CRT SEARCH IS THE BRANCH'S USABLE PRODUCT: 22 of 22 row tops
                  reproduced scan-free, including the LP lane's 20, 25, 35 at about 100 times
                  lower cost and six m31 rows only a 33-billion-column pass had produced; it
                  reaches one machine beyond every scan. But it is a decision procedure, not a
                  bound: F never enters it, and the arithmetic's own share of the kills above
                  the row is zero at exactly the two rungs that carry the budget's tightness.
                  Family: no single gear is to blame (repairing a real tooth helps no more than
                  a wrong one); coherence is global. Other letters: b + r(b) <= F + 7 and
                  q' + r(q') <= F + 9, the shortest letter the tightest. The gate now closes at
                  the certified row top itself (12, 20, 25, 35, 46), reproducing [15, 35] at
                  29 -> 31 and adding 31 -> 37. NEXT NAMED: the record gap itself as a 2-run
                  (both closed routes compared two covering optima that grow by different
                  mechanisms: F jumps 43 -> 58 -> 88 by a J = 4 fusion while a_L + r(a_L) goes
                  45 -> 58 -> 77).
                  - 4.i.a.i.a.1.a.i. The record gap as a 2-run (research/proof/record_2run.md).
                    FACT, no candidate. n1(v), the largest single neighbour of a gap of size v,
                    computed for the first time (6.23 billion gaps at m31; the whole top band of
                    m37 scan-free). The top is pinned to F_2, not F: the least valid c in
                    v + n1(v) <= F + c on v >= 0.8 F is 4, 5, 7, 6, 5, 12, 7, and D_top = F_2 -
                    max_{v >= 0.8F} (v + n1(v)) = 0 at six machines and 3 at m31, where the
                    unique F_2 pair is (35, 33) with larger member 0.60 F: the pair statement is
                    almost a top-of-spectrum statement and stops being one at the deepest
                    machine. RECORD SATURATION (new, exhaustive, 68 of 68 record occurrences of
                    eight machines): every gear of M is the sole striker of a column INSIDE the
                    record gap itself, stronger than the spare-gear lemma; the configuration is
                    frozen, so n1(F) is a max over 2 m(F) determined numbers (m(F) = 4, 12, 20,
                    20, 4, 2, 4, 2). Suppression is real beyond rarity (n1 below the rarity
                    null at 34 of 35 top-band cells, deficit growing to 8.4). Gear 5 buys the
                    record's ends (strikes the first column outside at 124 of 130 ends). Out of
                    sample at m37: m(88) = 2, n1(88) = 2, N(88) = 4, the two fusion words
                    (28, 37, 12, 11) and its mirror, 89 and 90 certified empty, top band 88, 85,
                    77, 72, 71 with 13 certified holes, the record isolated by 3 for the third
                    machine running. THE SWITCH at rung 19 -> 23, sharp: from there the record is
                    built from ordinary sizes fused at a letter (largest-piece rank fraction
                    0.35, 0.30, 0.27, 0.33; no top-3 piece at 4 of 4 rungs; same for F_2). THE
                    RECORD AS A 2-RUN IS NOT WHERE THE DIFFICULTY LIVES: budget slack at its own
                    2-run 10, 10, 12, 18, 24, 29, 32, 39 at rungs 13 -> 41, non-decreasing; the
                    tightness is carried by 3- and 4-runs of ordinary sizes (rank 0.27-0.83)
                    with a letter middle. The holes 41, 42 at m29 have a larger capacity margin
                    than the record and no local certificate: the same wall as the letter's row.
                    Instrument: the configuration enumerator reproduces 24 of 24 sieved cells.
                    Child named: the letter's N(l) as a 3-run and what the padded middle buys.
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
      - 5d. Every gear is needed for the record, and the record set is pinned (7d, then 5d.i and
        5d.ii, 2026-09-05). Exact: F(M minus g) < F(M) for every g at m7..m23, and the minimum
        blocking set of the period record is the whole machine (set cover, m7..m23). The record
        set: 2, 4, 12, 20, 20, 4, 2, 4 stretches at m7..m31; at m29 one mirror pair (every gear
        pinned), at m31 four stretches with every gear but 29 and 31 pinned. CORRECTION: the
        first reading "anchor + 7 + top gear fixed, middle gears free" was an m19/m23 artefact; at
        m31 the free gears are the top two. From m23 the record is one residue class mod the
        period up to mirror. FACT. The candidate object reading is withdrawn: pinning says where
        the record is, not that an opening is forced into the window.
        - 5d.i. The record as a frame of three gears (research/proof/record_frame.md). Spawned by
          5d's first reading. Theory: frame (5, 7, top) decides where, the middle gears' filling
          decides whether. DEAD as a route: the record set collapses (above), so there is no
          frame/filling split; the window holds q/210 frame columns (0 or 1 at every rung), so the
          briefed window test is vacuous; the non-vacuous version (longest blocked run starting in
          the window, L*) is 24 columns from q = 23 to 43 and 27 from 47 while F - 1 climbs 33 to
          144, max L*/(F-1) = 0.727 at m23 falling to 0.19 at m53, and it is the largest twin gap
          below q'^2, which is what the root needs and what nothing here bounds. FACTs kept:
          (i) completions of a record frame are 2 / 1 / 1 at m23 / m29 / m31 against a proper
          independence baseline that makes them 0.0087 as likely at m31 (a factor of two rarer per
          rung); (ii) coverage-maximality split: gear 5 sits at its coverage-maximal phase in every
          record of every machine m13..m31, gears 7 and 11 from m19 on, and the top one or two
          gears never do (the sole-striker requirement L4 in coverage units; a mechanism for 5e).
          Refuted: one top-gear corridor and one word per machine (true only at m23 and m29 where
          the record set is one pair); break offsets concentrated mod 35 (1.65x, not 3x) or near a
          gear square (0.8%).
        - 5d.ii. What each gear holds up, in the period and in the window
          (research/proof/deletion_profile.md). Spawned by 5d's "every gear needed". WEAK, closed
          as a route: the contrast is exact and describable, every window-side quantity is
          contingent on the primes, no forced object. FACTs: (i) the period record needs every
          gear (minimum blocking set = whole machine, m7..m23) while the window's longest stretch
          needs a chosen fifth (32 of 166 gears at rung 997; smallest initial segment {5..877});
          (ii) the period deletion profile falls with g and gear 5 tops it (drops 3, 3, 5, 9, 13,
          17 at m7..m23; the top gear near the minimum), refuting "largest at the top gears";
          (iii) the window profile is ordered by column position, not gear size: most holders own
          one sole column, a central one halves the stretch, an end one does nothing; (iv) zero-
          drop gears are individually redundant but jointly essential (removing all of them
          destroys the window stretch at 157 of 165 rungs); the only provably droppable set is the
          square gate g^2 > 6 top + 1, exact at all 165 rungs but explaining 11 of 143 zero drops;
          (v) nested-decreasing holder law, one-line proof: for a fixed stretch the set of gears
          holding it up can only shrink as the machine grows (a sole column can gain a striker,
          never lose one); (vi) gear 5 holds every window record at 164 of 165 rungs and is the
          largest drop at 151; no gear is never needed. Stop lines: F_W is the largest twin gap in
          (q, q'^2) (7d's identity); F_W <= F(effective machine) at the window's top is R2 one
          level up with fixed point 1/6, not iterable.
      - 5e. Where a record gap can start: the slot F mod 5 dictates (7a, 2026-09-05,
        research/proof/anchor_cycles.md). F = 1 mod 5 starts on 11|13, F = 4 on 17|19, F = 2 or 3
        on a mirror pair of slots, F = 0 on any; exact at all eight full periods to m31. FACT, new,
        position only.
      - 5f. Position facts kept as breadth, not opened (docs/novel): corridor resonance (big gaps
        recur at slot separations 35, 70, 105, left endpoints pinned to residues {10, 12, 18} mod
        35), the golden spectral gap (gear 5's local frequency mode is phi, phi/3 a machine-
        independent spectral gap). Both subject to the escape-distance-1 ceiling; a spectral
        large-sieve route would give count bounds and meet the rate-to-maximum step.
      - 5g. The coverage profile and the hinge (research/proof/gear5_lock.md, 2026-09-05).
        Spawned by 5d.i's coverage-maximality split and 5d.ii's hinge column. PROVED, position
        only: THE GEAR-5 LOCK. Every maximal blocked stretch of every machine, at every length
        (record, runner-up, window stretch, anywhere) has gear 5 at its coverage-maximal phase.
        Proof in five cases from gear 5's teeth {+-1} mod 5 and the two flanking openings being
        non-teeth; exhaustive to L = 2000; gated at all 62 records of m13..m31 and at every
        maximal blocked stretch of every window at 295 rungs (1.7 million stretches). Node 5e
        (the F mod 5 slot rule) is this theorem read at the stretch's start, now uniform in the
        machine and the length. Residual: a maximal stretch of length L leaves gears 7..q exactly
        floor(3L/5) columns in gear 5's two-and-one pattern (iterating is 7a, dead). Forced object,
        nothing the machine does prevents it, but it pins a phase, not a length.
        - Allocation law at records (correction to 5d.i): every gear of a record is at its
          coverage maximum SUBJECT TO keeping the columns only it strikes, 340 of 348 gear-cells
          over all 62 records (exceptions: gear 13 at four m17 records, gear 19 at four m19);
          the gears below maximum are middle ones (17 at m19/m29; 13 and 23 at m23/m31), and the
          top gear is at maximum in every record at m13 and m17, so "top gears never" was a
          reading of m23/m29. FACT.
        - Two laws, not one: period records have 78% of gear-cells at maximum and 2.3% free
          deficits; window longest stretches 30% and 63% (292 of 295 rungs carry a free deficit);
          only the gear-5 lock is shared. FACT.
        - Theory A's counting half stopped as pre-registered: capacity is 54% loose and
          loosening (sum of maxima over L = 1.20 .. 1.54 at m13..m31); the missing quantity is a
          lower bound on overlap (1, 5, 9, 13, 19, 28), the dead overlap count.
        - Theory B (hinge) DEAD: hinges always exist (295/295; they are the pseudo-twins of
          alignment-rules 4.1), but the hinge gear exceeds q/2 at 57% only, is central at 31%
          (below uniform), and every length rule fails (L <= g_h at 8 rungs, e.g. q = 421, L = 104,
          g_h = 97). No rule of that family can exist: by the nested-decreasing holder law the
          hinge gear falls as q grows at fixed L (877 -> 409 for the 241-column stretch over
          q = 919 .. 1669).
        - Small facts: the record is isolated by 3 in the gap spectrum at m29 and m31 (no 41, 42
          below 43; no 56, 57 below 58); the 295 rungs to 1999 carry only 11 distinct window
          stretches (the maximal twin gaps), so per-rung counts are not independent samples.
    - **R3.h. Ends or middles** (research/proof/ends_or_middles.md). The human's question of
      2026-09-04, answered on the exact records. ANSWER: it is the ends. A record is a row of
      ORDINARY lower gaps whose junctions the top three gears strike: m29's 43 = 10 + 10 + 23 as
      gaps of {5..23} (whose own record is 34); m31's 58 = 23 + 10 + 25 as gaps of {5..29}
      (record 43); m23's 34 = 4 + 8 + 15 + 7 as gaps of {5..19} (record 25). Through {5..17} the
      record is an ordinary stretch (largest piece 30% of it at m29, 31% at m31) and its seven or
      eight junctions are closed by exactly three gears (19 + 23 + 29 taking 3 + 2 + 2 at m29;
      23 + 29 + 31 taking 3 + 2 + 2 at m31), each on its own teeth. No lower machine's record sits
      inside a record at the top three layers at m23/m29/m31 (lower records do sit inside at
      gears 5 and 7 always, and at 13 and 17 at m31). A record is never corridor-extremal (0 of
      44) and its mod-35 phase has no consistent direction (escape distance 1 again). F = flank +
      letters of the top gear + flank with only the two flanks free, which restates the budget
      inequality as a flank condition (2-fusion: the pair statement; deeper fusion demands
      strictly shorter flanks; slacks 20, 16, 14, 12 at m29, m31, m23, m19), a reformulation of
      the merge grammar, labelled as such, landing on node 2g. Window contrast over 1.3 million
      stretches at 160 rungs: the top gear never removes a survivor from the window's longest
      stretch (0 of 160); that stretch is a two-piece fusion at every rung; a fusion of four or
      more by one gear occurs nowhere in any window while the m23 record is one; a three-fusion
      occurs everywhere but only on stretches of median 0.34 F_W. The asked-for statement ("a
      four-piece fusion never occurs inside a window") is true as measured and does not bound
      F_W: a two-piece fusion of two long pieces is long, and that is how the window builds them
      (203 + 39 at q = 997). FACT, exact, not a route.
      - R3.h.i. The flank brick: the two-sided walk from a tooth (weak point W5;
        research/proof/flank_walk.md). Spawned by the owner's bricks-and-mortar reading of
        R3.h. FACT, with one theorem that closes a route. THE JUNCTION THEOREM (proved): the
        junction condition is a congruence mod q' and the old machine is periodic mod P with
        gcd(P, q') = 1, so over the period of M + q' the flank pairs at junctions are exactly the
        flank pairs at all old openings, each twice; a junction is an ordinary opening and the
        maximum flank sum at junctions IS F_2(M). The flank brick cannot be fitted by structure
        at the junction. Measured: 0 violations of the pair statement at 583,881 period
        junctions (slack 9, 12, 12, 17, 26 at m11..m23) and 70 window junctions; max over
        window openings within F_W + q' at 152 of 152 rungs. L6 MADE EXACT: b_g^+ + b_g^- is a_g
        or g - a_g at every opening and gear (0 exceptions in 10.3 million pairs), which forces
        exactly two things: a gear acts on both flanks only if a_g <= S - 2, and a gear that
        misses the stretch has g - a_g >= S + 2 (tight at slack 2 everywhere); nothing joint
        about the two lengths. THREE GEAR BANDS: gears with g - a_g < S + 2 strike at 100.00%
        (2.28 million cells); the middle band strikes at 0.796 +- 0.004, constant over
        q = 59..997; the top band falls 0.36 -> 0.18; the length is decided in the middle band.
        THE FLANKS ARE COUPLED BY THE ANCHOR, not by L6 (0 exceptions in 8.8 million openings):
        L^+ = 1 mod 5 forces L^- in {0, 2, 4}, L^+ = 4 forces {0, 1, 3}, L^+ = 2 forbids 4,
        L^+ = 3 forbids 1; with gear 7, 931 of 1,225 pair classes mod 35 are admissible. The
        anti-correlation of the flanks is not a residue artefact (conditioning mod 35 doubles
        it, -0.045 -> -0.089 at m23); in the window the raw correlation is +0.048 from the
        twin-density trend and -0.023 detrended. THE WINDOW HAS AT MOST TWO JUNCTIONS, and we
        know which (0 mismatches, 152 rungs): the column of q' (iff q' is a twin member, 28
        rungs) and the column of q'^2 (iff q'^2 - 2 is prime, W.a's square gate, 42 rungs). At
        the top junction the flanks under {5..q} are the two-sided walk from q'^2 under
        {5..q'} (0 mismatches of 42); at the bottom junction L^- = round(q'/6) = d_0(M) at all
        28: the twin-Bertrand quantity is literally one flank of the window's own bottom
        junction. THE BRICK IS NOT AN ATOM: 20 of 20 flanks at the ten longest m23 junctions
        are themselves 2- or 3-piece fusions closed by 17, 19 or 23; F = flank + letters +
        flank recurses; at all five period maxima the outermost blocked column of each flank
        is held by a gear <= 13. INVERSE SHAPE REFUTED: no fixed number of buckets bounds the
        walk (L^+ <= b_(1) + b_(2) fails at 62% of m23 junctions; at 705 junctions the sum of
        every bucket is below L^+); the only exceptionless rule is the umbrella bound, which
        uses the gears the walk MISSES: a two-sided stretch of span S is struck by every gear
        with long arc below S + 2. Toward the root: the pair statement follows from (H) "the
        smallest gear missing a two-sided stretch at a junction is at most (3/2)(F + q')", the
        two-sided form of W.a's L < d, of unbounded order; column 0 is never a junction (the
        shield) but its CRT translate is, with the same flanks (d_0, d_0), realised at 15,107
        m23 junctions, so the obstruction is not evaded. Corrections: neighbour-of-hit acts on
        M + q', not on the old walks.
    - **R3.i. The half-column map** (the manager's reading of Leg(v) and the coupling law;
      research/proof/half_column.md). DEAD as a route, with five identities and three theorems.
      Identities (0 exceptions): both letters of a gear point at its home column (1,228 gears to
      10,007); for even v, Leg(v) = the prime factors of the members of column v/2; for odd v,
      exactly one of (3v -+ 1)/2 is a member of the quarter column (v -+ 1)/4; island coupling
      at separation delta = Pad union Leg. THE FIBRE THEOREM: exactly three distances have
      half-column c, namely 2c, 4c - 1, 4c + 1, and they are exactly the short letter of both
      members and the long letter of each: the fibre of the map over a column is the alphabet
      of that column's gears (2,000 columns, 0 exceptions); this is why twin gears share an
      arc. THE FIXED-POINT THEOREM (proved): a column is a fixed point of the halving descent
      iff it is a twin column (a proper factor of a member has a strictly smaller home
      column); the m29 and m31 record trees close on 6 and 8 columns with terminals {1, 2, 3, 5}
      = the twin columns of the closure. THE UNCOUPLED CLASSIFICATION: for v < y^2/3, v is
      uncoupled in {5..y} iff v is y-rough and its half-column is a twin column above y
      (5,505 of 5,505 cells, y prime 5..199): the even distances a machine cannot couple are
      exactly twice the twin columns above it, the short letters of the twin gears it does not
      yet own. The spectrum rule is dead both ways (uncoupled sizes are realised: 24 occurs
      1,180 times at m29), but the graded form is exceptionless: an uncoupled size is depleted
      by a factor 12 to 128 against its coupled neighbours (10 of 10), a step at zero coupling
      gears, and the flip is exact (v = 41 absent at m29, realised 134 times when 31 arrives,
      41 = a_31 + 31). Records in column coordinates: m29's 10 + 10 + 23 and m31's 23 + 10 + 25
      live in columns 5 and 6 only (10 = 2 x 5 is the shared short letter of column 5 = (29,
      31), which holds both top gears; 23 and 25 = 4 x 6 -+ 1 are the two long letters of
      column 6); every record letter lands on the new gear's home column, 11 of 11. m31 also
      lacks gap size 54. The window: the stretch's half-column lands in the frame (below q/6)
      at 142 of 160 rungs, not in the window. Why it does not bound F: coupling constrains
      strikes while a gap's endpoints are openings; the map grades the spectrum and cannot cap
      the record. Residual (HC-R): prove the depletion factor, which needs merge-history
      counting, thin place 1 again.
    - R3.i.a. The spectrum's depletion as a sum rule (research/proof/spectrum_sum_rule.md).
      DEAD as a route: the identities force nothing (the LP interval for m(24) at m29 under
      every identity is [0, 15.8 million] against the truth 1,180; zero is feasible at every
      uncoupled size at m19..m31), because coupling enters every identity only through
      c_q(v) in {q - 4, q - 3, q - 2}, minimal for uncoupled v: the identities can certify
      depletion, never force it. KEPT, exact: the recursion m_{M+q'}(v) = c_{q'}(v) m_M(v) +
      Merge(v), survival exact at 137 of 137 cells, reproducing the m31 spectrum in 30 s (prior
      art: the paired-Holt recursion's first coefficient, cited and stopped); the identity
      A(v) = prod c_q(v) >= prod(q - 4) with equality iff v is uncoupled, and A(1) = m(1) so
      rho(1) = rho(2) = 1 exactly; dividing out A(v) deflates the recorded depletion 4-5x, most
      uncoupled cells then inside the ordinary scatter; every spectrum hole is a phase hole,
      never a span hole (7 of 7); no size is ever lost and every size is born a merge (137/137,
      55/55). The self-reference is exact (an uncoupled even size below y^2/3 exists iff a
      twin column with a y-rough index sits in the window, 2,260 of 2,260 rungs) and points
      the wrong way: it is the root's covering problem one class higher.
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
route; the region past zero as a source of openings; the phase vector being a square (real vectors fail the island witness exactly as often as locally-square and random ones).

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
- 2026-09-05, provers 5d.i and 5d.ii (record_frame.md, deletion_profile.md). Both DEAD/WEAK as routes; node
  5d corrected: the record set collapses to one mirror pair at m29 and four stretches at m31 with only the top two
  gears free, so "middle gears free" was an m19/m23 artefact and the frame/filling theory has no object. New exact
  facts: record-frame completions 2/1/1 at m23/m29/m31, 115x rarer than independence at m31; coverage-maximality
  split (gear 5 always at its coverage-maximal phase in a record, top gears never); period record needs every gear,
  window record a chosen fifth; period deletion profile falls with g with gear 5 on top; window holders ordered by
  column position; zero-drop gears jointly essential; square gate exact but weak; nested-decreasing holder law
  (proved). Neither branch bounds the largest twin gap below q'^2, which is the root. Next: the only STRONG parent
  left under R3 is 5 itself; the tree needs a new observation, not a new sibling.
- 2026-09-05, manager: opened 5g (coverage profile, hinge column) under 5; one prover under the skill. Result outputs from now on live in research/<line>/r<round>/results/ untracked.
- 2026-09-05, prover 5g (gear5_lock.md): THE GEAR-5 LOCK PROVED - every maximal blocked stretch of every
  machine at every length has gear 5 at its coverage-maximal phase (five-case proof from the teeth {+-1} mod 5
  and the flanking openings; exhaustive to L = 2000; 1.7 million window stretches, no exception); 5e is the same
  theorem read at the start. Records: every gear at its coverage maximum subject to keeping its sole columns
  (340/348); the gears below maximum are middle ones, and the top gear is at maximum at m13 and m17 (5d.i's
  "top never" corrected). Period and window stretches obey two different allocation laws sharing only the
  lock. Counting half stopped (capacity 54% loose, overlap lower bound is the dead end). Hinge DEAD as a
  length lever, with a reason that closes the family (hinge gear falls as q grows at fixed L). Verdict for
  R3: the structure-of-the-record line yields forced POSITION objects (lock, corridor, slot rule) and no
  length lever; three rounds of depth under node 5 confirm it. Next: change formulation, not sibling.
- 2026-09-05, manager: R3 line verdict recorded (position objects, no length lever); opened R2.a (the machine feeds on itself, observation-first) and R3.h (ends or middles, the human's question on the exact records). Two provers.
- 2026-09-05, provers R2.a (self_feeding.md) and R3.h (ends_or_middles.md). R2.a: four exact walk-frame rules
  W1-W4 (the walk from q^2 starts on the top gear's tooth and is struck by it once; deepest hopping layer is the
  top gear iff q^2 - 2 is prime; a level-free transfer rule for which gears carry over from a birth column into
  the pair's own walk, admissible set {7, 17, 31} at the nearest offset; the next level's walk starts at
  6k^2 - 2k), zero exceptions in 667 walks and 832,915 checks; chain of landings has no rule; register entry
  docs/novel/walk-tooth-frame.md; FACT, not a route (the walk is made by the old gears; L < d is twin-Bertrand at
  scale q/3). R3.h: the human's question answered - the record is made of the ENDS: ordinary lower gaps fused at
  their junctions by exactly three top gears (m29: 10 + 10 + 23 of {5..23}; m31: 23 + 10 + 25 of {5..29}), no
  lower record inside at the top three layers, never corridor-extremal; in the window the longest stretch is a
  two-piece fusion at every rung and the top gear never removes a survivor from it; a four-piece fusion never
  occurs in any window but that does not bound F_W. Both FACT. Standing verdict after this round: every
  formulation tried (per-step, whole-window, record structure, anchor frame, walk frame) yields exact position
  and mechanism facts and stops at the same length statement; the tree needs a formulation in which length is
  the primary object, or a new observation.
- 2026-09-05, manager: opened R2.a.i (the path taken apart) at the owner's direction; two provers, breadth of analysis on the walk from q^2.
- 2026-09-05/06, provers W.a and W.t (walk_path.md, walk_transforms.md): the path from q^2 taken apart in the
  owner's frame. Parts proven: anchor slot and gear-5 offsets pinned by q mod 30 (never slot 11|13; offset 1
  always struck; L never 1 mod 5; 15-class law mod 35), each gear two progressions with square phase, the
  quadratic-residue bar (which gears can reach an offset is q-free; at i = 0 exactly gears = +-1 mod 8), the top
  gear inert beyond offset 0, q^2 the unique sole-striker tooth of q in its window. Interactions: the path uses
  only proven order-two laws, thinly (two-piece fusion at 2,234 of 2,259). Depth profile dip-plateau-spike with
  per-offset mean depth a fixed function of the offset; landing avoids high-depth offsets. First unproven
  interaction: the length (unbounded-order covering). Register entries walk-path-parts.md, walk-path-transforms.md.
  Opened R2.a.i.a (the reachability landscape), one prover.
- 2026-09-06, prover L (reachability.md): the landscape closed in form; CANDIDATE OBJECT named - for every prime
  q from 1489 to 19,997 some offset in the fixed set {5, 10, 12, 17} mod 35 past the square, below the top gear's
  next tooth, is struck by no gear (0 exceptions above 1487; 17 failures below; slack growing). Islands are exact
  CRT classes; large gears strike them at exactly 2/g so counting through islands gives nothing new; the
  interaction to prove is stated with q as the sifted variable. Opened R2.a.i.a.1 (the witness under pressure),
  one prover.
- 2026-09-06, prover I (island_witness.md): the witness is about integers coprime to 30, not primes (0 failures
  in 52,574 above 2849; every multiple of 5 fails by a proved relocation law); 0 exceptions in 17,748 primes to
  200,000 with the minimum open-island count strictly increasing; one class i = 12 mod 35 suffices from 5477;
  the free island sits inside 0.152 d and its absolute offset never exceeds 2,392. Cover number K(d) = 3, 4, 6, 9,
  14, 20 at d = 35..1120 (ILP-certified) grows while the counting requirement stays bounded. Opened
  R2.a.i.a.1.a (the cover number), one prover.
- 2026-09-06, prover K (cover_number.md): K(d) exact at 23 arcs to 1,330, growth d/(ln d)^3, not the counterfactual
  family's ladder (free separation gives a perfect partition equal to counting); growth bought by one phase per
  gear and the fixed separation, not by the strike budget. Proved: a cover is realised by exactly 2^K classes
  modulo a product above q^2 (a failure pins q^2 as an integer). Obstruction named exactly: 2.7^m covers against
  a 2^K class density, vacuous by 10^24 at d = 1,120. The night's line: R2.a.i -> R2.a.i.a -> R2.a.i.a.1 ->
  R2.a.i.a.1.a, all exact, one candidate object with 0 exceptions to 200,000 and its proof obstruction stated in
  the machine's terms. Paused for the owner's direction: the next interaction as posed is circular.
- 2026-09-06, manager: opened R2.a.i.a.1.b (squares are even) at the owner's suggestion; one prover, the three-vector experiment.
- 2026-09-06, prover S (square_vector.md): "squares are even" decided - OUTCOME C. Real, locally-square and random
  phase vectors fail the island witness at the same rate (0.9984 +- 0.0033 over 6.3 million vectors each); index
  parity worth 1%; reachability mod 24 i exact and spent. The only global effect is the sifting level s = 2,
  where the classical 4 e^-2 gamma over-count appears and repairs the first moment (16.51 predicted failures
  against 17 observed). Random vectors stop failing at the same arcs as real ones (3.3e-7 at d = 1100). 82 failing
  locally-square vectors, 0 with a square CRT lift; the square condition is implied by the range condition.
  Dead end: the phase vector being a square. The obstruction is transfer, not structure.
- 2026-09-06, prover W3 (separation_drives_K.md): the real separation does not drive the adversarial cover
  (K_real is the mode of the random distribution at every arc); the island target has zero to one gear of
  slack; W1 dead, W3 answered, W2 is the root in covering language. The wall document updated. Next brick (the
  owner's bricks-and-mortar reading of R3.h): the two flanks of a record are walks from a top-gear tooth in the
  machine below; open as R3.h.i.
- 2026-09-06, prover F (flank_walk.md): the junction theorem (junctions are ordinary openings; the flank brick
  IS F_2(M)); L6 exact at 10.3 million pairs with its two forced consequences; three gear bands, the length decided
  in the middle band at a constant strike rate 0.796; flanks coupled by the anchor mod 5 and 35; the window has at
  most two junctions (the column of q' and the column of q'^2), with the bottom flank equal to d_0 and the top
  flanks equal to the walk from q'^2; the brick recurses; the inverse-shape bucket bound refuted. W5 closed: the
  flank brick is the pair statement itself. Every weak point on the wall is now tested.
- 2026-09-06, manager scan then prover N (neighbour_profile.md): the neighbour-sum profile. F + 1 law dead at m29
  by 4 at the letter; replaced by N(v) <= F_2(M) for v >= 6, exceptionless to m31, with a PROVED mechanism (the glue
  lemma: CRT re-phasing under a two-colouring glues the two flanks into an adjacent pair, so F_2 is the natural cap).
  The F_2 cap cannot close the chain statement (needs F_2 - F <= a, fails at m17, m29). Child: the glue as a finite
  covering statement. Dead-branches file written (dead_branches_reopened.md) with five recurring thin places.
- 2026-09-06, provers G and C2 (glue_covering.md, compatibility_chain.md). Both DEAD as routes: the glue covering
  statement is false where it matters (the glue buys one column, the shadow; the m29 run (18,10,30) resists every
  construction); separation compatibility is a liability, not a protection (fully compatible members violate the
  budget at m11, m17). Kept: the shadow lemma and the move lemma (proved), the J-run outer law g_1 + g_J <= F_2 for
  runs with middles >= 6 (3.3 million runs, 0 exceptions), and the first face-C exception: the real teeth are
  atypical in gluability (99.6th percentile). Thin places 2 and 4 of the wall are now measured closed; 1, 3, 5 open.
- 2026-09-06, prover S6 (separability.md): thin place 6 DEAD; gluability is not separability (counting forbids
  disjoint flank covers below y = 109); shared gears are 5 and 7, the top gears are the free ones; the one-third
  separation maximises sharing; Leg(v) = {g | 3v +- 1} exact; the resistant m29 run is the m31 record class; face
  C's exception shrinks to a factor 2.4 at matched cells. Next: thin place 1 (count gears, not columns).
- 2026-09-06, prover H (gear_count.md): thin place 1 dead in its proven form (forced gears saturate at 2q/3; forced
  is not needed), but the branch corrects the wall: the adversarial covering with free gear SETS is strictly
  stronger than the root (best 4 gears block 16, not 11), the mechanism is the arc (a gear beyond its umbrella is a
  bare domino of size a_g; twin gears share arcs, so the real machine must buy both members of every twin pair),
  and A(K) < (p_{K+1}^2 - 1)/6 is the clean open lemma. Wall 5a corrected; the arc multiset added as a handle.
- 2026-09-06, manager scan (research/anchor235/r49/allteeth_record.py): thin place 3's sharp reading is DEAD - the distance from every record and near-record stretch to the nearest all-teeth column (a column struck by every gear, 2^m per period) is random (median equal to the random expectation at m11..m23), so the record is not anchored at the machine's alignment points. By-product, provable in a line: the blocked run through an all-teeth column is always exactly 1 (both neighbours open, since 6(k +- 1) = +-1 +- 6 mod g vanishes only if g divides 4 or 8). Opened: the arc multiset (R2.b.i) and the second moment over q (R2.a.i.a.1.c).
- 2026-09-06, provers ARC and MOM (arc_multiset.md, second_moment.md). Arc multiset DEAD as a handle and the
  reading reversed (twins are the cheapest small gears; de-twinning LOWERS the record; which arcs is worth
  nothing, the count everything; the real machine is an optimal 10-gear blocker); A(K) exact to K = 12 with the
  open lemma holding at margin 2.7-3.8 flat. Second moment DEAD by proof (a bound below 1 on the failing fraction
  is the conjecture). Every thin place of the wall has now been tested; every one reduces to face A (capacity with a
  gear count) or face D/E. Kept: the type lemma, the MILP certificate tool, the coupling-gear divisor law, the
  sub-Poisson count with its mechanism.
- 2026-09-06, lane D and prover HC (distortion_method.md, half_column.md). Distortion: outcome (c) by the collapse
  lemma; the engine applies to the machine with budget sum 4/g^2 < 0.365; localised it proves A(K) < window for
  K <= 10 and fails from 11; the crack is a second moment over arithmetic blocks (child opened). Half-column map:
  the fibre theorem, the fixed-point theorem (fixed points are twin columns), the uncoupled classification (the
  even distances a machine cannot couple are twice the twin columns above it), records in columns 5 and 6;
  dead as a route (coupling constrains strikes, endpoints are openings).
- 2026-09-06, prover B (block_moment.md) and manager check: the block second moment is partition-agnostic and the
  exact budget on the real window is 0.36, but the a priori budget busts at the sixth gear and the fair-share
  hypothesis is false; what survives is the single inequality sum (4/g^2) rho_g^2 < 1 (rho the in-window strike-
  rate excess on survivors), exact head 0.31-0.36, trivial tail short by a factor 2 to 3.7: the smallest miss of any
  unconditional bound on the tree, and the unproven part is 7b's equidistribution curve. Round closed; every named
  lead and child run; the wall's statement stands in its sharpest form.
- 2026-09-06, manager's second check on R2.c.i: the one-block inequality is trivially valid (each term is alpha_g^2) and its content is the survivor lower bound itself; the 'factor 3.7' assumed the fair share in the denominator and is withdrawn. Wall 5g rewritten.
- 2026-09-06, prover SR (spectrum_sum_rule.md): sum rule DEAD (identities cannot force depletion); the spectrum recursion made exact to m31 with A(v) = prod c_q(v) explaining most of the depletion; every hole is a phase hole. Opened R2.c.ii (fibres of a sub-machine).
- 2026-09-06, lane T (docs/proofs/20, small_K_theorem.md): THEOREM A proved (no K <= 10 primes with fixed-separation pairs cover the next prime's window; certified 0/1 infeasibility, corroborated four ways) and THEOREM B proved (A(K) exact to K = 6 by reasoning and proved-complete case lists); the span lemma and the head collision are new tools; the distortion lane's positive withdrawn (its localised inequality is false); K = 5 optimum corrected to {5,7,11,23,29}.
- 2026-09-06, prover FS (submachine_fibres.md): sub-machine fibres DEAD as a route (the exact budget is monotone in the cut; the admissible cut buys one gear); SF-CAP theorem (thresholds better by 3-801 orders, still exp(q^0.6)); the residual is level of distribution 1/2 at every modulus and class in dimension 2. Manager note (bitwise toolbox): the XOR of the gears' masks is the parity of the striker count, and in the window that is the Liouville function of 36k^2 - 1 (Omega(N_k) = 2 + depth(k) for squarefree columns), so the machine's parity bit is a Chowla-type object; recorded, not opened. Next: pairwise collision laws (the head collision generalised to twin gear pairs).
- 2026-09-06, prover HC2 (collision_laws.md): shared-arc law and arc floor proved; twin pairs collide earliest; the block-matching bound proves the adversarial lemma by reasoning at K = 4, 5, 6; the required interaction order grows like K - 3, so no bounded-order law reaches all K. The wall's sharpest structural statement so far.
- 2026-09-06, manager, toolbox pass (option 2 chosen by the owner): bitwise lands on Liouville of 36k^2 - 1 (Chowla) and the bucket vector; inclusion-exclusion is Brun; the merge recursion as an operator is the renewal ladder; characters give the pole-phase law (real teeth: each gear's Fourier coefficient at a is nearly a function of a mod 6). The one object the tools reach that is unexploited: the merge forest (node 4 WEAK, untested). Opened 4.i, one prover.
- 2026-09-06, prover GF (merge_forest.md): branching identity proved (orders = second difference of chain counts; mean order q'/(q'-2) teeth-free); node 4 refuted; the frontier F(M+q') = max_a (a + Rest(a)) with Rest collapsing at the top (Rest(F_old) <= 7; rest <= q' fails only at 29->31). Opened 4.i.a.
- 2026-09-06, prover FR (frontier_collapse.md): the top law (Rest(F_old) = the old record's largest single neighbour, or its neighbour sum when F_old is a letter) and the fusion-rate identity proved; the frontier splits into the pair half (J = 2) and the chain half; the uncovered band at m31 is [15, 25] and holds the record maximiser; the availability gate named as the child.
- 2026-09-06, owner: the period-scale formulation (R4). Manager's reading: level of distribution 1 at the period scale removes faces B, D, E; face A stands alone; the second machine decomposes into coherent twisted copies of the first. Opened R4.a.
- 2026-09-06, prover AG (availability_gate.md): the gate ladder proved, the gate is one row of the level-2 dictionary with the closed form a_hasM <= F_2 - a_L, useful only when F_2 - F < a_L (fails 17->19, 29->31: the same obstruction as the F_2 cap); the band [15, 35] at m31 holds the record maximiser and does not shrink. Residual: the short-letter row of the adjacent-pair dictionary is empty above c F.
- 2026-09-06, prover BI (branching_identity.md): the branching identity proved in general; C_r = W_{r-1} + Z_{r-1} (legal words plus all-pad) proved; second moment in closed form (the teeth live in the variance); the size side proved with eps in {0,1,2}; closure at depth K_m <= m J_max proved (finite depth, not finite state). The count side of the spectrum's evolution is now closed-form; the size side's extremes are the chain statement.
- 2026-09-06, owner: 'not a route is a bold claim'. Node 4.i.b corrected to route-shaped; opened the ladder instrument (4.i.b.i) and named the functional search (4.i.b.ii).
- 2026-09-06, prover R4 (period_scale.md): the two machines built exactly to q = 23; level 1 exact; the clutch's coupling is entirely in the both-open cell; the window is the clutch's zero-interaction region; placement residue law (dimension 1 vs 2 = the parity barrier named); exactness buys nothing (Brun's main terms alternate at s = 2). Parked under R4 per the owner.
- 2026-09-06, prover SL (short_letter_row.md): the gate's residual PROVED scan-free by LP duality at m19, m23, m29; the pinned letter F <= a_L + r(a_L) <= F + 3 at 8 of 8 rungs and confirmed out of sample at 31 -> 37 (r(12) = 46, a_L + r = 58 = F exactly); the pair filter and the closer law proved; the row's emptiness is a per-machine covering fact. Opened the pinned letter.
- 2026-09-06, prover PL (pinned_letter.md): the pinned letter is a real-teeth law (family violates the constant), unproved; the glue route dead; the spare-gear lemma proved; every gear is a sole striker at the attaining 2-run of every realised size. Opened the arithmetic child.
- 2026-09-06, prover PA (pinned_arithmetic.md): the pinned letter's lower half REFUTED out of sample at 37 -> 41 (77 against 88); the upper half survives with growing slack; the real-teeth input is the single coordinate n = 6k with teeth at +-1; four exceptionless laws; the CRT row search reaches one machine beyond every scan. Named next: the record gap as a 2-run.
- 2026-09-06, prover RG (record_2run.md): record saturation (every gear a sole striker inside the record gap, 68 of 68); the top of the spectrum is pinned to F_2 not F; the switch to ordinary pieces is sharp at 19 -> 23; the record as a 2-run is not where the difficulty lives (slack non-decreasing 10 -> 39); the tightness sits in 3- and 4-runs of ordinary sizes with a letter middle; m37 top band certified scan-free.
- 2026-09-06, prover LD (ladder_closure.md): the closure as an instrument reaches F(37) = 88 and F(41) = 91 exactly from m23's period alone with every gate exact; m41's record is one of 3,052 fourfold fusions among 8.5 trillion gaps; the span-threshold prune (lemma) is the tool; budget slack 14, 20, 16, 7, 38 along the extended ladder; F(43) needs about 10^8 dictionary rows. The window's leads are now run to their verdicts.
- 2026-09-06, owner: one more round on location inside the window with the lower machine only; opened R2.e.i (the position-length frontier) and R2.e.ii (structured families); if neither closes, the top machine is next.
- 2026-09-06, prover SF (structured_families.md): structured families DEAD by identity (a family defined mod the lower machine's period cancels its own saving; every family carries twins at the window's own rate; the islands indistinguishable from ordinary corridor columns, 1.006 +- 0.005); gear 7 barred at the column-0 offset iff q = +-2 mod 7.
- 2026-09-06, prover PF (position_frontier.md): theorem (E) proved (the effective machine at a column is exact); R_min(L) >= 3.25 L for L >= d_0 and = 1 below, 0 exceptions; from q = 1427 the longest run of the prefix is the initial run; the window statement reduces exactly to d_0 <= W, the initial run of the bottom machine's diagonal, with all of [q/6, W] provably safe. Location pinpointed: the bottom. The round did not close it; per the owner, the top machine is next.
- 2026-09-06, prover TM (top_machine_1.md): the top machine on its own terms is a domino machine (partner law), with the forbidden gap 4, the parity law F_top = 2m - (m mod 2), the tiling characterisation, universal record multiplicity, the spectrum as a second difference, and the conjugacy n -> 6^-1(n+1) onto the bottom's coordinate (counting and symmetry laws common, metric laws its own). The wheels have their rules; the clutch is next.
- 2026-09-06, Formalist (top_machine_lean.md): the top machine's laws L1-L8, L10, L12, L13, L17 (upper bound), L19 kernel-checked in two new libraries, 59 declarations, zero sorries, standard axioms; L8's exact group and L17's attainment will not close this round. Manager gate green.
- 2026-09-06, Formalist round 33 (top_machine_lean.md, proofs/TopMachineCrt.lean): the two holes closed. exists_crt (Finset CRT), parity_attained and parity_law (F_top = 2m - (m mod 2) as an equality), affine_group / exists_symmetry / sign_count (the symmetry group is exactly (Z/2)^m). Build green at 1392 jobs, standard axioms, zero sorries. Of the 21 laws, the statements about gear sets are now kernel-checked except L16 (tiling characterisation, written proof), L18 (multiplicity, measured) and the counting laws by CRT.
- 2026-09-06, prover TM2 (top_machine_2.md): the wheels' second pass. The gap census law L22 (exact inclusion-exclusion product, 0 mismatches to the full period), W1 closed (gaps 3 and 5 share a polynomial; 7 is the unique separating gear), L18 derived from the census (L25), the forbidden gap 4 explained as the one place closed and free boundary covers differ (L26), the origin as the unique total collision with the clump its shadow (L29), the record of small wheels decided by whether 7 is a gear (L30, exhaustive), F_top depends on m and the gears below F_top + 1 only (L31), W2 replaced (the wheel record is reached, at a computable fraction of the period, L32), no top anchor exists and why (L34-L36), the removal law (L37/L38). Next: the walk lane (R4.b.iii), then L22 into the kernel.
- 2026-09-06, prover TM3 (top_machine_3.md): the owner's deliverable found. The next open pair after x is x + mex{(-x) mod g, (-x-2) mod g} when every gear exceeds 2m (proved, sharp, 0 mismatches in 1.45 million positions); the next twin candidate (run of three) is x + mex over three residues when every gear exceeds 3m, with record exactly 3m; the layered walk collapses to hop chains of length at most 2; the walk distribution is the dual of the run spectrum; the spectrum cannot decide the record; XOR bounds the record from below, tight. In use the mex form is exact but its proved bound is vacuous past 10^6: the in-use bound is the open item. Next: Formalist on the mex laws.
- 2026-09-06, Formalist round 34 (proofs/TopMachineWalk.lean): the owner's closed form is a theorem. mex_form (the next open pair after x is x + mex of the two residues per gear, gears above 2m), triple_mex_form and triple_law (the run-of-three record is exactly 3m; attainment needs no size hypothesis), no_start_gap, pair_corr. Green at 1394 jobs, standard axioms, zero sorries; 158 declarations in the top-machine library.
- 2026-09-06, prover TM4 (top_machine_4.md): the in-use record is the largest gap of the q-smooth-pair list below Q (zone law, proved, exact in value and position), so it is linear in the largest gear and no bound in (q', m) exists; in use the tail is empty and the parity apparatus is about tail gears only; union bounds provably cannot reach the truth; above the zone the record is a different, small object (24-419) with no proved bound. The wheel record's exact core/tail rule found (domino cost). The next-opening deliverable stands as: proved closed form for gears above 2m; exact zone law in use; open above the zone.
- 2026-09-06, prover TM5 (top_machine_5.md): the split slid down. The smallest simple machine is q' = 5 (domino machine and bottom machine {5..q} begin together); the mex form's true hypothesis is F_top < q'; the parity threshold is 2m + 1 (even m) / 2m + 3 (odd m); the symmetry group is absolute from 3; the record's dependence on m and the gears below F + 1 is absolute down to {2, 3, 5}. Transfers to the bottom listed unchanged / modified / none; the bottom's next-opening formula as a self-certified mex over truncated progressions (exact, 890,501 walks). The anchor rescaling law is the fold as a coordinate identity.
- 2026-09-06, prover TM6 (top_machine_6.md): the quiet zone. The rule n = smooth times at most one prime above Q holds on all of [1, Q^2] (0 exceptions, 45 machines); the owner's g_0^2 is the true lower edge of the zone's record region, Q - sqrt Q is not; complete knowledge four ways including a q-independent family decomposition and the walk as a smooth-times-nextprime minimum; alignments always occur with a proved prime-gap floor; the record (183-419 at Q = 10^4, sitting at 1.35-2.63 Q) is R4.b.iv's A(q, N); no upper bound because the bottom stratum is the twin primes above Q themselves (L65).
- 2026-09-06, owner's standing order: no clutch until motor, wheels and exhaust are fully understood (objects ledger as the gate); the clutch to be built piecewise as interface objects, each with a proof and ideally a closed form; when stuck, find the shadow of the object the blocker traces and reopen construction of any part; partial interfaces kept, never dismissed. Skill and memory updated (clutch-strategy).
- 2026-09-06, Formalist round 35 (proofs/MachineStack.lean, docs/proofs/23): the stack and the exhaust kernel-checked: stride containment two tiers down, non-containment one tier down, the exhaust cap (home strike or echo on (C, C^2], primality of the exhaust gear not needed), open iff twin prime on the window, the smooth and quiet zone laws. Gap stated: CutMono is a prime-density hypothesis, false at q = 2, 3, proved only for the first step (Bertrand).
- 2026-09-06, prover TM7 (top_machine_7.md): the wheels' open laws closed. The loaded record rule proved both ways (F_top = max{L : min_U D_L(U) <= #{g > L + 1}}, 0 mismatches on 6,659 gear sets), the parity law and its sharp threshold derived as corollaries (one parity bit is the whole + 2), the moment vanishing proved with r(d) = D(d - 1) and the forbidden gap 4 absorbed as the degenerate case, the census law in kernel shape. Four items genuinely open on the wheels alone; two are the conjecture in a range coordinate.
- 2026-09-06, prover EX1 (exhaust_1.md): the exhaust measured for the first time. Self-similarity of tier 3 confirmed on full periods (0 exceptions in 18 million residues) with one sharpening (the run-start count at L = 1); CutMono is a theorem from q = 5 (cut_{k+1} > 4 cut_k, dyadic Bertrand; the base step fails at q = 2, 3, 4 exactly); the redundancy lemma in range form needs no primality; the exhaust makes no strike below q#, only home strikes and echoes on the window (57,344 of 57,344), its first non-redundant strike is exactly p_1^2, and it is the majority partner one decade above the window; its own record is ROOT; the owner's saturation prediction refuted into the regime law F_range = max(F_smooth, F_quiet).
- 2026-09-06, Harvester (law_register.md): the wheels' and exhaust's laws registered W1-W85, X1-X8 with prior-art verdicts (14 known, 18 known variant, 38 new, 22 standard tool, 1 refuted: W9 fails once N_1 = 0, W67 is the true form). Strongest novelty cluster: the collision law W29 with W17, W44, W71 (finite two-class interval covers; the covering-systems literature works on infinite covers with distinct moduli). The gap census W22-W24 is the two-class version of Brown 2024 (arXiv:2311.06873), cite. W64's constant 160 is Lehmer's last {2,3,5}-smooth pair. MANAGER CORRECTION IN FLIGHT: the register's verdict that the record is Ziller-Morack's paired Jacobsthal h_2 and that their Conjecture 6 IS the window statement is wrong as stated: OEIS A288815 = 6 A072753 + 6 and A072753 is the FREE-residue two-class covering record (the wall's 5a adversary); the real twin-candidate gap at primorials is 6 F(M) (12, 30, 42, 66, 108, 150 by direct scan), strictly below h_2 from p = 11 (42 against 66; 528 against 708 at 37). So Conjecture 6 is the ADVERSARIAL window statement, stronger than ours and implying it. A Harvester is computing the three ordered records (real, the project's fixed-separation adversary of docs/proofs/20, free) and correcting the register, docs/novel, docs/proofs/20, 22, 23 and the wall's 5a.
- 2026-09-06, Harvester (jacobsthal_check.md): the correction verified and applied. Three ordered records at K = 1..5 (longest coverable run, F - 1 convention): real 1, 4, 6, 10, 17; the project's adversary of docs/proofs/20 (free primes, D = 2) 1, 4, 6, 15, 21; free two-class (A072753) 2, 4, 10, 24, 31; the real twin-candidate gap at p_n# equals 6 F(M) for n = 3..9 (12, 30, 42, 66, 108, 150, 204 by direct sieve to 223,092,870), not A288815. Verdict: h_2 at primorials is the free-residue two-class record (Ziller-Morack's j_2 quantifies over the even difference D, which by CRT is two arbitrary classes per prime); F(M) is the real-teeth instance D = 2, F(M) - 1 <= A072753 with equality at {5, 7} alone; Conjecture 6 is the adversarial window statement and IMPLIES ours (and Goldbach), not the converse. docs/proofs/20 is NOT a partial result toward Conjecture 6 (the two adversaries strengthen along different axes: doc 20 frees the primes and keeps D = 2; ZM keeps the initial segment and frees D); it now cites A072753 as the published table of the adversary it compares against. The project had the right reading already in docs/novel/jk-growth-discriminator.md section 6; the register's error came from OEIS returning 403 to the lane. Follow-up flagged: sweep docs/novel/j2-lower-ladder.md and j2-upper-bound.md for any sentence reading the project's F as h_2.
- 2026-09-06, Formalist round 36 (proofs/TopMachineRecord.lean, MachineStack.lean addendum): the loaded record rule is a theorem both ways, necessity with no hypothesis and sufficiency with coprimality alone; the parity law re-proved independently from the rule; a gear strikes both ends of the window iff g = L + 1 exactly (sharper than the branch's <=); CutMono unconditional for prime q >= 5 by a halving induction on Bertrand, so the stack's cap holds for every base from 5. Green at 2246 jobs, standard axioms, zero sorries; 269 declarations.
- 2026-09-06, Formalist round 37 (proofs/TopMachineCensus.lean): the gap census law W22 is a theorem with gears positive and coprime only; general inclusion-exclusion and multi-forbidden-set CRT lemmas added; the forbidden gap 4 derived from the formula's algebra. Green at 2248 jobs, standard axioms, zero sorries; 310 declarations.
- 2026-09-06, prover TM8 on Fable (top_machine_8.md): the wheels' last open laws. Non-cancellation proved (every minimum cover has sign (-1)^r; M_{r(d)} = (-1)^r r! C_r with C_r closed-form by d mod 4, the eight multiplicities 8 of 8); the cover polynomial gives the whole universal signature and N_d as a t-th difference formula (0 mismatches, 27 cases); the core minimisation reduced to two windows of one core wheel, classes provably not decouplable, no polynomial algorithm claimed; the parity-refined capacity bound proved, exact below core density 0.376. The wheels have no open structural item on paper.
- 2026-09-07, manager (manifold_census_large.md, overnight runs): the manifold census at Q = 10^5 and 3 x 10^5. The quiet-zone record at Q = 10^5 is the same gap at the same place for q = 5, 7, 11, 13 (924 after 187,907, a twin-prime gap in the bottom stratum): the manifold's record is a twin gap, the ROOT verdict at scale; family (1, 1) identical across engines (27,411,455; 203,707,420); at Q = 3 x 10^5 the record differs between q = 5 and 7 by one pure-air pair (850,500 = 2^2 3^5 5^3 7 open for q = 7 only) splitting the twin gap 1,452 into 151 + 1,301; gap 4 absent in billions; gap 3 = gap 5 approximately whenever 7 is not a gear. Back-pressure correction applied. Two lanes relaunched from disk (register on Fable, engine's monotone functional on Opus).
- 2026-09-07, Harvester (law_register.md): rows W86-W102 and X9-X24 registered (126 rows in all; the new rows 6 known, 7 known variant, 14 new, 6 standard tool). Verdicts that matter: the loaded record rule W88 is a KNOWN VARIANT of Ziller-Morack's one-class rejection criterion (arXiv:1611.03310 Cor. 2.3) with the two-class delta as the content (the tail piece is a domino, the price is Gallai's edge-cover identity W87 KNOWN, the boundary g > L + 1 because a point has no parity); the cover polynomial W97 is the edge cover polynomial of a path with two pendant singletons (Akbari-Oboudi 2013) and the difference formula W98 is NEW as an identity (Brown 2024 has the signed subset sum, no generating polynomial); the exhaust's first pass adds no new mathematics (Bertrand iterated, Eratosthenes to the root in range form). Web-search budget exhausted mid-pass: Hagedorn, Iwaniec, Akbari-Oboudi texts and OEIS not read this round; listed in the register.
- 2026-09-07, cleanup of old contexts (owner's request): human.md and agents-shared.md rewritten as current-state snapshots in the canonical words; README 'Where things stand' to 7 September; the objects ledger folded to one verdict per object (ENGINE not yet, three items; MANIFOLD yes on paper; EXHAUST measured once, CutMono in the kernel) with stale numbers updated; the wall gains 5l mapping its faces onto engine / manifold / valves / exhaust; docs/proofs/README maps the retired words. Two facts surfaced: 310 declarations is the manifold-and-exhaust library only (the engine's Lean corpus has no total on record); docs/proofs holds 23 written proofs, not 21.
- 2026-09-07, manager (manifold_census_large.md addendum, owner's question): the identical record across four engines is a mechanism, not a coincidence: in the bottom stratum only primes and q-smooth numbers are open, and the record twin gap contains no smooth number with a prime neighbour (0, 1, 2, 3 smooth numbers inside for q = 5, 7, 11, 13, none adjacent to a prime). It is the usual case: identical at 14 of 19 values of Q from 20,000 to 200,000, sticky over ranges of Q; every exception is a smooth number with a prime neighbour splitting the twin gap. W103 (measured, ROOT).
- 2026-09-07, prover MF1 (monotone_functional.md, finished from disk after the limit): THE ORDER LAW, the interaction order the budget needs is exactly L(M) + 1 (B_{L+1} <= F + q' at 9 of 9, B_L > F + q' at 7 of 7, 0 exceptions); Phi = B_{L+1} is the monotone functional, the only one of eleven that bounds the next record and is budget-monotone; its theorem implies the budget at every rung and needs a cap on L plus one fusion lemma; the band check at 29 -> 31 passes with margin 16 by complete enumeration; L(M) bounded stays OPEN (L_pad uncapped). Engine gate: item 2 closed, item 3 found (measured), item 1 open.
- 2026-09-07, prover LP1 on Fable (pad_cap.md): the padded word capped uniformly over the small alphabet (E1, proved, <= 8, 0 exceptions on 14 rungs; every measured padded word is in that class); the skip half is ROOT (a constant cap is equivalent to F(M + q') <= C q'); the mechanism is the pullback machine's cover of the multipliers, gear 7 the capper. The engine's last structural item is resolved: measured half proved, remaining half the conjecture in disguise. THE GATE OPENS: engine, manifold and exhaust have no open structural item that is not ROOT. Next per the owner's plan: the valves' hybrid first step (scratch lane with clean context on Fable; review lane; reconcile).
- 2026-09-07, valves review lane (valves_review.md, 852 lines): ten interface objects sorted with status (I1 the conjugacy as the fold on n = 5 mod 6; I2 the anchoring valves 2, 3, 5, 7 and the corridor mod 35 = the opening set of {5, 7}; I3 the family decomposition and the pure charge; I4 the echo set and home strikes, outer boundary p_1^2; I5 the twisted copies = back pressure in the cofactor coordinate; I6 theorem (E) and its exception set, the inner boundary; I7 the placement residue law = valve timing, dimension 1 vs 2 the parity barrier; I8 the island witness and K(d), which does not translate into families; I9 the four cells and the level of distribution, translated into zones; I10 the zero-interaction region, a property of every split). Eighteen inherited assumptions found, chiefly: the column coordinate is not a definition but the first valve (gears 2 and 3 fold); the manifold's metric laws (dominoes, chains, gap 4, parity record) never reach the valves' domain; every tier strikes its own twins as home strikes so the valves never see a smooth-zone twin. Five predictions on record for the reconcile (families occupy exactly the classes of P mod 30 solving s'P' - sP = 2, spot-checked; every family's count independent of q; the valves' record is a twin gap; timing = one forbidden class per engine gear per sign; RED FLAG: any manifold metric law surviving into the valves' open set).
- 2026-09-07, valves scratch lane on Fable, clean context (valves_scratch.md), and the reconcile (valves_reconcile.md): the valves have a mechanism. Each family is a valve with an imprint (proved, both lanes), a port (class mod 6 fixed by its air, proved), an onset (it opens at the turn equal to its air, proved as a bound, exact as measured), a yield (a local density, measured; Bateman-Horn). The pure charge is the valve with no air: no onset, in every turn, the same count for every engine. Turns 1 and 2 carry no fuelled valve but (1, 1), the rest are embers. The review's four predictions were rediscovered from the definitions; the red flag held. Next: the turn ledger.
- 2026-09-07, prover VT1 on Fable (turn_ledger.md): the turn ledger is exact (V1, 0 exceptions in 600 turns) and the wall reappears at it BY PROOF (V2): a counterfactual fuel set obeying every proved valve law has zero pure charge in every turn, so imprint, onset, port, inventory and ember cannot force P_m > 0; the wall at the ledger is face A and nothing else. Measured: the first two turns never empty for Q > 9 to 10^5 (min P_1 = 25 at Q = 1031); the valve count A(m) exact with its recursion; the pure share tends to the engine's own pair density in m, not to 0. Shadow: the fuel's pair correlation across the engine-open classes, i.e. the primes' own twin structure.
- 2026-09-07, unstick pass 3 (dead_branches_reopened_3.md): every valve object descends to the engine's record at a lower rung (turn m of Q is the top 1/(m + 1) of the window at sqrt((m + 1) Q), exact via theorem (E)); adversaries with balance, mirror closure and one tooth per gear still empty the pure charge, so what remains is the real fuel's phase zero at every prime; the one object not the record restated is its dual Omega, the rich-interval function of the pullback. Opened: the rich half of the record, back in the engine.
- 2026-09-07, prover RH1 on Fable (rich_half.md): the rich half of the record is the corridor's richness (gears 5, 7, 11 decide Omega; the sharpened cap equals E3 value for value); records sit at the richest translate at 4 of 7 rungs and the start class is not a formula; E4 proved (the arc floor holds for any separations with arcs >= 2); ROOT honestly (Omega is a max over translates, every cap linear in F/q'). The dual coordinate is the record's cap restated, sharper by a constant and not in shape.
- 2026-09-07, owner: the LADDER is the problem as an analogy: it forces building the proof upward, where unseen rungs could kill the conjecture; the exhaust says the ladder never has to reach beyond the window, and the manifold's back pressure on the valves says the same from below. The descent found today (turn m of Q = the top 1/(m + 1) of the window at sqrt((m + 1) Q), exact via theorem (E)) is the mechanical form: the object at scale Q is the object at scale sqrt(Q); not an ascent but a self-similar map. Ladder retired as a proof framing; kept only as the name of the certified record table (rungs = its entries).
- 2026-09-07, owner: existence suffices (wall 5m) and the position can be in the valve, not the engine's window. Exact weakening: a twin anywhere in (Q, Q^2] = the engine's window nonempty at some scale in [sqrt(2Q), Q]. Opened R4.c.iii, existence in the valve.
- 2026-09-07, Formalist round 38 on Fable (proofs/ArcFloor.lean, 66 declarations, namespace ArcFloor): E4 the arc floor in the kernel for any separations with arcs >= 2 (collision_eq_zero_of_arcs, no hypothesis beyond the arcs; the exception at arc 1 is an exact value, collision_two_of_arc_one); the near gear needs only 3 <= g, sharp at g = 2; the document's g, h >= 5 not needed; file 21's claim that any proof must use 3a = g -+ 1 is refuted in the kernel (the separation's value never appears); arc_real_ge_two shows the real teeth satisfy it. File 21's Theorem 1 (collision_add, +4 per common period, any separations with two distinct teeth and coprime gears) and E5 (coincidence_add, the min form) proved from one per-phase identity. Manager gate: lake build TopMachineWheel ArcFloor green at 1231 jobs; audit: standard axioms; zero sorries. Files 20 and 21 had no kernel before this round.
- 2026-09-07, prover VE1 on Fable (valve_existence.md): existence in the valve is the window statement at some rung (the freedom of the turn is worth three integers), ROOT; but the frontier sees length at the top: V8 proved that a frontier constant c certifies the turns m < c, giving turn 1 for every Q in [30, 1859] from the certified ladder and turns 1-4 to 8 x 10^7 from the measured prefix floor; V9 exact: the frontier's constants 4.625 and 3.25 are the twin gaps 661 -> 809 and 73 -> 101, fixed for all larger machines because new gears lengthen runs rather than precede them. Phase zero is the sole invariant surviving the counterfactuals. Next child: new gears lengthen, never precede (engine construction).
- 2026-09-07, prover NP1 on Fable (lengthen_never_precede.md): theorem (E) at one step proved with its exact two-column exception set (E6); the prefix frontier inherited exactly up the ladder (E7, 0 exceptions in 8,152 cells), so new gears lengthen and never precede on the prefix, while on the full period the sentence is false (8 of 11 steps, order-2-4 merges); the frontier for all rungs reduces (E8) to the straddling condition, a bound on the twin gap across each prime square, ROOT in face E's sense; V8 corrected: a hidden hypothesis and, above Q = 148, the certification is the share form. The frontier's constants are eleven twin gaps on a staircase; 4.625 from rung 23, never undercut.
- 2026-09-07, Formalist round 39 on Fable (proofs/OneStepE.lean, 32 declarations): theorem (E) at one step in the kernel (blocked_succ_iff; the exact exception set new_iff: the home column d_0(M) iff (q', q'+2) is twin, the square column iff q'^2 - 2 is prime; home_isLeast_open: the new home column IS d_0(M)); prefix inheritance for maximal runs (maxRun_succ). Finding: theorem (E) as phrased in position_frontier.md is false outside the prefix (refuting instance q = 5, k = 8 in the kernel, E_needs_prefix); the true form needs 6k + 1 < q'^2 (blocked_iff_sqrt), which is where it was used; position_frontier.md and the wall (5n) corrected. Hypotheses: q never assumed prime; the next-prime property enters only as 'no prime strictly between q and q''. Manager gate: lake build OneStepE green at 740 jobs; audit standard axioms; zero sorries.
- 2026-09-07, owner: the record's 'window' and the descent to rungs were a regression; the solution space is horizontal, the manifold (fully understood) and its interaction with the engine, with the exhaust as the known upper limit; derive the manifold's behaviour from scratch instead of translating back. Manager: accepted; the descent stays a fact, not a tool; opened R4.c.iv, phase zero in the manifold's own terms, with window / rung / ladder / descent forbidden.
- 2026-09-07, prover AL1 on Fable (opening_alignment.md): the owner's alignment mechanism measured. The manifold's longest opening is present below Q^2 (6 of 7 runs) but is far shorter than the engine's blocked runs, holds at most ceil(L/6) slots, and is slot-free in 62% of cases at (5, 10^4); alignment of openings with the engine's slots is exactly uniform and phase-blind over full periods and 0.78-1.0 of uniform on the quiet zone; the one phase-zero effect is the opening spectrum exceeding CRT (the fuel is the primes, Mertens); a slot-holding opening below Q^2 is a twin, so the statement is the conjecture restated. Localisation kept: twins sit in openings of length 1 in 92-95% of cases; slot-holding long openings need turn >= 12 or an ember (V22, exact).
- 2026-09-07, prover PZ1 on Fable (phase_zero.md): phase zero is exactly saturation (the open set closed under divisors and multiplication by air; proved, 0 violations in 107,750,211 tests), which no adversary has, and which does not force a pure pair (the saturated sieve with the twins added as gears has none: what excludes it is the cut). Law table: 13 phase-free by the translation lemma, 10 phase-zero, 3 beyond multiplicativity and all three are saturation. The slice profile is new: poor at the bottom (onset), rich in the middle (Buchstab), flat at the top; the manifold's record sits in the emptied bottom. The Liouville-signed charge census is 0.1% against a pure share of 8.2%: the parity barrier in one number. ROOT.
- 2026-09-07, owner: revisit the construction; stack machines by squares (machine 2 = primes q' to q'^2 run to q#, machine 3 from the next prime to its square, ...), classify each machine's anchor cycles as open / closed / mixed on its square part and its band, across q, then relate the machines and connect to machine 1's cycles. Vocabulary: machine 1 (the engine), machine 2 (formerly the manifold's first block), machine k; square part; band. Opened R4.d, builder running.
- 2026-09-07, builder SQ1 on Fable (stacked_squares.md): the stack by squares built exactly. The band structure holds with 0 exceptions (half-open square parts; every prime's square is a twin slot, S1); the count of machines is 1 + floor(log2(theta(q)/ln q')); no gear >= 7 closes a cycle alone and every closed cycle uses >= 3 gears (S3); on a band a machine's open numbers are exactly the smooth numbers and smooth-times-one-prime (S4), so the CRT cycle expectation is wrong in both directions; every square-built machine has Mertens product 1/2 (S5): the construction is the equal-weight decomposition of the manifold; the gears of machine k + 1 are the primes of band k and the twins of band k are its double-home slots (S7, exact); the joint-open ratio is P(prime | rough)/P(open) (S6). The cycle at q# is not open for machines >= 2. ROOT: every band holds a twin is twin-Bertrand between consecutive squares.
- 2026-09-07, owner: pursue the base case; the smallest machine that builds a twin pair is 2, 3. Manager: the base link is [3, 9) with twin gears (3, 5), (5, 7); the step is exact by S7; opened R4.d.i.
