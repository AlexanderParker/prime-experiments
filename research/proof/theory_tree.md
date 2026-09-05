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
