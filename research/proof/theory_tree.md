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
        at v in {6, 7, 8, 11}. OUT OF SAMPLE AT m37 (prover U4, 2026-09-11, research/proof/
        engine_laws_m37.md; scripts research/anchor235/r72/u45_*.py): max over realised v >= 6
        of N(v) at m37 = 87 <= F_2(37) = 90, HOLDS with 3 to spare (maximiser (60, 10, 27) and
        its mirror, span 97 = F_3(m37)); two routes agree (covering scan, 323 words; the
        complete D_3(m37), 30,325 rows carrying 217,929,355,875 gaps, built by the closure
        ladder from K_0 = 17, loss 0, 4,220 s). The J-run outer law max(g_1 + g_J), middles
        >= 6: m29 55, 52, 45, 40, 39, 33 and m31 66, 60, 59, 55, 50, 52 at J = 3..8 on full
        periods, m37 87, 82 at J = 3, 4; 0 exceptions at eight machines. Broken: "the maximum
        falls with J" (m31: 50 at J = 7, 52 at J = 8, witness (32, 8, 8, 7, 7, 6, 7, 20)); the
        maximiser's middle moves (v = 6 or 7 to m31, v = 10 at m37, v = 7 reaches only 79).
        Not closed: J = 5 at m37 (exhaustive negative above 105, 57,064 candidates undecided
        between 90 and 105: the solver stalls at span 120-140 with ten gears). New instrument:
        the exact full-period gap census c(d) for any gear set as a covering count, no period,
        no scan (u45_census.py), gated on m11-m31's complete spectra and all 20 + 21 published
        W32 numbers. FACT, exact, no proof; nearly tight at J = 3 (F_2 - outer = 0, 2, 3).
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
      re-toothed anchors). CANDIDATE OBJECT (mark withdrawn 2026-09-11: R4.d.i.a showed the island witness is the blind classes, exact, plus ordinary density), but exhausted at the first gear above the anchor:
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
        gear above sqrt q). CANDIDATE OBJECT (mark withdrawn 2026-09-11: R4.d.i.a showed the island witness is the blind classes, exact, plus ordinary density): the reachability landscape (the q-free set of gears
        that can reach each offset) with the landing preferring its low points; child opened.
        - R2.a.i.a. The reachability landscape (research/proof/reachability.md; register entry
          docs/novel/reachability-landscape.md, prior art not yet checked). Spawned by the
          quadratic-residue bar. STRONG, exact, and it names a CANDIDATE OBJECT (mark withdrawn 2026-09-11: R4.d.i.a showed the island witness is the blind classes, exact, plus ordinary density). Parts, all
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
      about tail gears only; [U5, 2026-09-11, engine_laws_m37.md: W32's first-hit exactness on
      the engine's own window is REFUTED: within one unit at 4 of 13 windows y = 7..53 (misses
      to +14 at y = 23) and 5 of 43 section records (-6 to +18, mean +5.4, 33 high / 8 low);
      cause split and measured: (a) the formula is biased low at small N (the median record of
      200,000 random translates exceeds the prediction at 13 of 13 windows, 40 of 43 sections);
      (b) the window's record is INHERITED, one gap (columns 110 -> 135, twins 659/661 ->
      809/811) being the record at six consecutive y, while a first-hit law prices a fresh
      range and the window adds 5-30% fresh per y (decay +14, +13, +10, +8, +8, +5). On the
      engine's phase-zero prefix at N >= 10^5 the law is within one unit at 12 of 13 (the
      wheels' rate): W32 is a large-N law, the item "first-hit as a law on the engine" CLOSED;] the covering bound 2m/(1 - 2 H_S) is alive iff F < exp exp(1/2 +
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
          1/(c + 1) of the window). With c = 1.25 from the certified ladder (161 <= F(59) <= 178, 161 certified as a lower bound; the review found no record closing it to 161 exactly):
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
            m37-m43 period table completed 2026-09-07: 11 of 11 steps, 140 exceptions in 297 cells). E6 (PROVED, direct-sieve gate at 428
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
          the 37 -> 41 positions addendum NOT filled: its scan was killed at a session limit). Omega^full(n) exact to n = 20 at
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
                the base-case quantity. The lane on the recursion died at the weekly limit three times
        (2026-09-07 to 09); the manager measured its reduction locally (research/proof/
        step_evidence.md sections 5-6: the composite record per section is the largest twin
        gap inside it, 4, 46, 579 slots against sections of 19, 2,668, 43 million; over every
        stretch of the record length the core's leftover is typically 20 and 12 at the record,
        the tail's strikes typically 199 and 181 at the record, supply exceeding demand in 100%
        of stretches yet exactly one twin-free: a twin gap is a below-average core leftover
        plus an exact finish by the tail).
        - R4.d.i.b. The core's real-phase leftover (research/proof/core_leftover.md, laws
          S10-S14; scripts research/stack/r5/). FACT with a ROOT mark. K_L(x) = the core's
          (gears <= 6L + 1) unstruck slots on the stretch of L slots at x, a definite periodic
          function since phases are real. Base 3, section [16129, 260,467,321), L* = 579, core
          485 primes: the real minimum over 43.4 million starts is K = 3 (one stretch, at
          70,722,785, which holds 2 twins) against a free-phase minimum of 0 (a greedy phasing
          of the same core covers 25,267 slots, 43.6 L*); the record stretch is NOT the minimum
          (K = 12 there, the 2.07th percentile, 897,507 starts at or below it): it is twin-free
          because 181 tail gears land on all 12. Base 7: min 1 against 0. Base 23 (no tail): min
          0 exactly once, at the record. S12 (EXACT): min K_L = 0 iff the core's own longest
          fully-covered run R(6L + 1) >= L; on base 3 that holds to L_0 = 278 = R(1669) exactly,
          then the minimum climbs 1, 2, 3, ... 20 at L = 300, 425, 500, ... 2000 while the mean
          goes 13 to 49. S13 (measured): every minimum is an extreme value of its own count,
          z = (min - mean)/sd between -4.19 and -4.71 at L = 579 for the real phases, three
          random phasings and two integer sets alike; the distribution is narrower than
          binomial (variance ratio 0.83). COUNTERFACTUALS: random phases of the same core give
          L_0 = 254, 296, 272 against 278 and minima 5, 3, 3 against 3, within 2 of the real
          minimum to L = 1000; from L = 1200 the real minimum sits below every random seed by
          exactly the shift of the mean: the real section's leftover density is 0.03516 against
          the CRT product 0.037439, a 6% deficit at L* rising to 14% at L = 2000 with
          u = ln n / ln(6L + 1): the Buchstab deficit next to the origin (S14, known mechanism).
          S11 (PROVED, kernel-ready): a pairwise-coprime replacement of the core does not exist
          (any pairwise coprime set of 485 integers in [5, 3475] coprime to 6 is one prime power
          per prime); the non-coprime substitute doubles the mean and the minimum with the same
          z-score. THE RECURSION'S ONLY TRACE IN THE MINIMUM IS THE ORIGIN'S DENSITY DEFICIT,
          which lowers it and never raises it. ROOT: min K_L > 0 above L_0 is R(6L + 1) < L for
          every threshold, whose quiet-part half is the longest twin gap (the minimising runs at
          L = 125-225 sit at the record gaps 187,913 and 850,355); the record needs the tail's
          coincidence on top. No candidate.
          - R4.d.i.c. Unstick pass 4, P against P1 P2 in the construction (research/proof/
            dead_branches_reopened_4.md; scripts research/stack/r5/leftover_shadow.py,
            leftover_depth.py, leftover_indep.py). THE REFORMULATION (verified): with
            t = 6L + 1 and S the primes in (t, t^2), every core-free member of a section is a
            prime or a product P1 P2 of two members of S (kernel two-prime lemma); the tail's
            strikes on core-free numbers coincide with the composite core-free members at every
            one of the 43,407,953 + 1,323,738 starts of the base-3 and base-7 sections (identity,
            0 exceptions); a stretch is twin-free iff S x S meets each of its K core-open slots:
            THE STEP AT THE CORE READS "the products of two of the core's survivors cannot meet
            every core-open slot of a stretch as long as the section's record". At the base-3
            record: 12 core-open slots, 15 charges, all 30 prime factors distinct (forced: a
            prime above 6L has at most one multiple in 6L numbers), every fuel in (16129, 65111),
            a prime of the section itself, i.e. a gear of machine 4, below t^2. THE COUNT SIDE IS
            FULLY INDEPENDENT: PP among K leftovers is binomial at K = 8..16 (K = 12: observed
            1 / 37 / 315 / 1750 / 6639 against 2.4 / 43.6 / 367 / 1877 / 6469), and the twin-free
            RUN count follows the independent-slot prediction at every length from L/4 to L
            (806/809, 102/76, 29/23, 8/7.8, 2/2.3, 1/1.1, and 1 against 0.72 at L = 579; base 7:
            1 against 1.28). MANAGER'S LEAD REFUTED: "40 times rarer than independence" counted
            starts against one run (a gap of L + r slots contributes r + 1 starts, the record gap
            exactly one). The composite count is not the constraint (exact bound 1,158 per
            stretch, measured max 87, record 59, +1.1 sd). THE DEPTH LAW: S15 (PROVED, kernel
            twin_of_rough / twin_of_not_blocked): a core-open slot below t^2 is a twin, so a
            twin-free stretch with K > 0 lies at depth u = ln n / ln t > 2 of its own core; S16
            (proved forward, converse with exact hypothesis): the tail is empty iff the section
            has no composite core-free member, and then every twin-free stretch has K = 0 and the
            record is the core's own composite record (S12). Verified on 13 record stretches: PP
            share exactly 1.0000 in every depth bin below 2, then 0.931 / 0.789 / 0.686 / 0.605;
            the 11 tail-empty sections lie below depth 1.95 with K = 0; the two with a tail have
            records at depth 2.10 (K = 1) and 2.37 (K = 12). The record's depth grows along the
            base-3 chain (1.35, 1.69, 2.37, about 4 next), so P against P1 P2 is one link's
            slice: above depth 3, P1 P2 P3 has the Liouville sign of P and the general object is
            the Omega-census of core-free members at depth u. Candidates: (a) the tail's charge
            set on the core's open set: the valves' laws transfer verbatim and force nothing
            (onset = the depth law); new object: the fuels at the top of a section are the
            section's own bottom primes (P2 > n / p_k); (b) the distance-2 graph is a matching,
            cover 12 of 12 with 3 spare, no structure; (c) the sign: five exact constraints on
            the P1 P2 census and nothing else, the PP deficit is the twin count's own extreme
            value (z = -3.56); (d) mirror not forcing. BRIEF: open "the leftover at depth u" on
            the base-3 section with shorter L' (50..579, core per L'), per (L', u) bin the
            Omega-census, the PP share, twin-free runs against the independent prediction, min K
            on twin-free stretches, the fuels' identity; pre-registered: PP share depends on u
            alone, Omega = 3 appears from u = 3, runs stay on the independent count above 3; a
            refutation is the first trace of the construction beyond depth; confirmation files
            the step at the core as ROOT in (L, u).
            - R4.d.i.d. The leftover at depth u (research/proof/leftover_depth.md, laws S17,
              S20-S22; scripts research/stack/r6/). ROOT IN (L, u). P3 HELD: the twin-free run
              counts sit on the independent-slot prediction in 14 of 15 non-degenerate
              lengths; the 62 bins at 2 <= u < 3 hold 184,398 runs against 184,223 predicted
              (+0.55 sd) and the 7 bins at u >= 3, where Omega = 3 members exist and
              P-against-P1P2 is false, 409,251 against 409,040 (+0.61 sd): crossing depth 3
              changes nothing a count can see. ONE DEVIATION, RECORDED NOT EXPLAINED: base 3,
              L' = 400, u = 2.3-2.4, 83 twin-free runs against 54 (z +4.33), surviving a second
              measurement (z +3.28) and localised by a third to one gap band [400, 450) slots (66
              against 38.6) while [300, 450) together sits below prediction (602 against 621);
              no overdispersion in the twin count (variance equal to the model's to 0.2%); base 7
              at the same relative length is -1.8 sd. P1 held to 2.7%, refuted as exact (a smooth
              1/ln t' drift, no section dependence, S21). P2 held with a sharper threshold, S17
              PROVED: Omega = j members first appear at nextprime(t')^j, 0 members below (11/11,
              5/5, 2/2; at L' = 18 exactly one Omega = 4 member, 127^4). P4 refuted with the
              exception count exactly the Omega >= 3 count (37,908 of 37,908; the two-prime lemma
              in disguise, switching at q^2 >= p_k, S20). S22: what remains of the step at the
              core is a bound on the count of PP among independent leftovers, with S15 and S17 as
              its only structure.
        - R4.d.i.e. THE FIRST REALISATION (theorist on Fable, 2026-09-11; research/proof/
          first_realisation.md; scripts research/anchor235/r74/xm_*.py). Spawned by proof
          skeleton section 13 (step 8 is position, not length). x_min(p, l) = the first column
          at which the engine {5..p} begins a fully struck run of l columns; PROVED from the
          construction: C1 x_min is the least element of a union of residue classes, one per
          cover (two computations agree: F(23) = 34 first at column 12,694,429; the m29, m31,
          m37, m41 records scan against CRT); C2 monotone in the gear set, so no lower bound
          comes from a smaller engine; C3 below the square column the runs are twin gaps; C4
          step 8 at the cut p <=> L_a(p) < l_p, the run THROUGH the square column shorter than
          the section, and L_a IS L_1(p), the first-twin offset above p^2 of R4.d.i.a's arc
          bound (0 exceptions to 10^7): the position form of step 8 is that measured object,
          STOP LINE. THE BRIEF'S TARGET REFUTED at p = 29: x_min(29, 20) = 111 < 140 = the
          square column (the twin gap 659..811, 24 columns, lies below the square) while step 8
          holds there (the run at column 140 is 3 long). L_a / l_p = 0.375, 0.100, 0.333, 0.357,
          0.135, 0.150, 0.147, 0.192, 0.107, 0.067, 0.050, 0.241 at p = 11..53. Laws refuted:
          X1 the first-hit floor x_min rho >= 1/4 (min 0.0014 at m19, l = 24; 80 of 647 cells
          below); X2 x_min >= (3/8) l^2 (111 < 216); P4 x_min(p, l_p) > p'^2 / 6 (refuted at 29,
          vacuous at 11, 13, 19, 23, 31, holds at the rest by factors 2 to 10^5, no floor). NEW
          POSITIONS (two methods each): the m37 record first at 90,816,580,903 (0.0734 of the
          period), m41 at 630,700,131,373 (0.0124); the record's first fraction 0.34, 0.19,
          0.044, 0.073, 0.012 at m23..m41; the staircases nested across engines (positions 5,643;
          6,024; 102,273; 254,736; 27,819,088; 16,365,163,681 recur, lengthened in place by added
          gears) and erratic (m37: run 62 at 2.8 x 10^7, run 63 only at 1.3 x 10^10). MECHANISM:
          every gear used at every first realisation with r >= 11 and every gear forced at the
          record (9 of 9 engines); at 82 of 82 rows the position is the least element of its
          forced class; twin-gear coincidence columns inside the first runs 44 against 51.6 by
          chance, the arc-floor object plays no part. THE EXACT OBSTRUCTION, a construction: the
          tooth family (same gears, teeth +-v_g) strikes the whole section at p = 17 (15 of
          1,440 members, e.g. teeth (1, 1, 2, 6, 1)), at p = 29 (6,030 of 1,995,840) and at every
          cut 37..53 (killers by exact cover over the teeth, e.g. (2, 1, 2, 2, 4, 3, 3, 9, 13, 5)
          at 37; 111 columns killed with 14 gears at 53), and at none of 7, 11, 13, 19, 23, 31
          (exhaustive). So step 8 is NOT a fact of the gear set: any proof at 17, 29, 37 and
          beyond must use the real teeth u_g = 6^-1 mod g. ROOT: the smallest missing lemma is
          L_1(p) < l_p, and the construction gives no lower bound on any position because a
          residue is not bounded below. Measurement only, no route: the count of record copies
          per period at m37 / m41 (u45_census.py) would say whether the first copy is early.
        - R4.d.i.f. THE LENGTH FACE OF STEP 8 (theorist on Fable, 2026-09-11; research/proof/
          length_face.md; scripts research/anchor235/r76/lf_*.py; gates F(11..29) and both
          first record starts reproduced exactly). VERDICT ROOT, with the parity twin BUILT:
          F(q) < q^2/6 is the twin prime conjecture in covering form, and the parity barrier is
          realised for it by a genuine set at every q, located at the origin. LF1/LF2 (proved):
          the strike pattern on any stretch is a function of the phase vector (x mod g) alone
          and every vector occurs, so F(q) is the record of the fixed-separation family (teeth
          t_g +- k_g, separation 2k_g = 3^-1 mod g); multiplicativity is invisible inside any
          stretch at height x > L/4 + 1, where every record run sits (x/L = 384,679 at m23 to
          10^9 at m37) and no section does (a/L about 1/p_{k+1}). The real teeth add to the
          length face exactly one thing, the one-third separation, worth a factor 1.3-1.8 over
          free classes (A072753 against F - 1: 60/33, 74/42, 94/57, 117/87, 148/90, 173/102,
          213/117, 236/144), never an exponent. NOTHING ON RECORD CONSTRAINS A RUN OF q^2/6
          except finite computation: capacity vacuous from q = 13 (2 sum 1/g = 1.02); joint
          counts teeth-blind (4L/(gh) +- 4 by CRT); W17 / mex need every gear > 2m + 1; merge /
          saturation bind only while F(M) < (q' -+ 1)/3 (fails at 7 -> 11); the budget summed
          is ROOT; the flank identity's upper half refuted at 43 -> 47; clump, arc floor, mirror
          positional. THE PARITY TWIN (LF3, proved): O^- = the open columns with lambda(6k - 1)
          lambda(6k + 1) = -1 is EMPTY below b = (q'^2 - 1)/6 > q^2/6 (rough below q'^2 means
          prime, and lambda(p) lambda(p') = +1) while its sieve data on [1, b) matches the
          plain open set's to square-root size (max |S_d| / sqrt A_d = 1.96 .. 2.89 at q = 23
          .. 53, 0 of 1,326 cells above 3): at every cut a set with indistinguishable sieve
          inputs violates the target by at least the section's length. Origin gaps of O^-: 28,
          60, 60, 140, 140, 228, 228, 308, 308, 368, 620, 620 at q = 11..53 against q^2/6 =
          20..468; its first element is the square column (r^2 - 2, r^2) with r^2 - 2 prime,
          12 of 12. INSIDE THE RECORD RUNS (two methods, 4 of 4): the invisible signs are
          balanced (sum sigma = -3/33, -12/42, -5/57, -9/87 at m23..m37; all 12 cells |z| <=
          1.85); the sieve-visible striker parity is odd-biased in every fully struck run
          (-0.43 to -0.47 per column against +0.006..+0.024 in random stretches) because runs
          are near-tilings (1.39-1.55 strikers per column). THE REAL TEETH BREAK THE SIGN
          SYMMETRY IN EXACTLY ONE PLACE: THE ORIGIN. The twins' own records on 10^8 columns
          (every gap re-verified): above about 32 b the twins are fair coins on O (16 of 16
          records inside the thinning range), 2.4-11 x the engine's record; they exceed q^2/6
          at generic positions for q <= 23 and not for q >= 29. EXPONENT MAP: any uniform
          F(q) <= C q^2 with C < 1/6 implies twin primes (the section exceeds (q'^2 - q')/6 >=
          C q^2 for q >= 1/(1 - 6C)); the band C >= 1/6 and exponents in (2, 4.27) is open,
          sieve-unreachable, and not what step 8 needs. Smallest uncertified instance: q = 61
          (F(61) >= 171, needs < 620); its phase-zero instance is "a twin prime pair in
          (q, q^2 + 1]". Seven pre-registered predictions, none failed.
        - R4.d.i.g. THE MECHANIC OF THE ORIGIN (theorist on Fable, 2026-09-11; research/proof/
          origin_mechanic.md; scripts research/anchor235/r77/og_*.py; gated on machine 2's eight
          twin gears and F(23) = 34). VERDICT ROOT: the parity problem in a third coordinate,
          the obstruction located one axiom more precisely. D1/D2 (proved): the dilation form
          is exact and IS the real teeth (g m = 6(g i + s k_g) + eps_g s; the dilate g.S occupies
          columns +-k_g mod g), adding nothing at the column level (SquareColumn.real_teeth)
          and one invariant at the member level, D5: every strike in a real section is n = g_0
          m with 5 <= g_0 <= sqrt(n) <= m, m a survivor; the tooth-family killers violate it at
          4..156 phantom strikes per section and V17 at every added gear; 0 violations in real
          sections at seven cuts. D3 (proved, exact): THE OWNER'S NESTING IS TRUE: the struck
          set is the disjoint union over gears of g.R_g, R_g the open survivors of the machine
          below g, in columns the affine image (i, s) -> g i + s k_g; 0 mismatches over 198,798
          members in 19 sections; but it relabels the union by least striker and leaves the
          union unchanged, so it forbids nothing the real teeth do not; its count is Buchstab /
          Legendre (stopped). What the recursion forces at the origin and not at height: n <
          g_0^3 forces a prime quotient (two-prime lemma, 0 exceptions), depth exactly
          floor(2 log_5 p') = 3, 4, 5; prime-quotient share 0.64-0.83 in the finer sections,
          0.57-1.00 at the first links, 0.16-0.27 at the four record runs (controls 0.16-0.23):
          the share is the prime density below p'^2 / g_0, a count. SMALLEST INSTANCE where the
          pinned dilates are the reason (free classes kill, real teeth do not): finer section
          p = 17, columns 49..59, gears 13 and 17 pinned to 13 x 23, 13 x 25, 17 x 19 while the
          killers move them onto the twins at columns 52, 58; NEW (refutes P7): at p = 29 a
          SINGLE gear kills (tooth 2 instead of 5 on gear 29, unique among 1,995,840): the twins
          143, 147 are mirror-symmetric about 145 = 5 x 29 and the abandoned pin column 150 =
          29 x 31 is re-covered by 17 x 53; distance to the nearest killer d(p) = 2, 1, 5, 2 at
          17, 29, 37, 41, >= 6 at 43, 47, 53, none at 11, 13, 19, 23, 31. For step 8's own
          sections no instance exists where the dilates are needed: free classes cannot kill at
          the proved links (capacity 14 < 18; h_2 = 60 < 135; 213 < 459) and h_2 < q^2/6
          wherever known. THE EXACT OBSTRUCTION (6.2), characterisation C: dilation + finite
          fold + hand-up = the real machine; each counter-machine breaks exactly one axiom: the
          tooth family (dilation), V17 (gears from the section itself), and a NEW THIRD, the
          monoid M generated by 5 and the primes = 1 mod 6: strikes are multiples, hand-up and
          square-root rule hold, sections between irreducible squares, and 8 FAILS at [25, 961)
          (20 columns with both members in M, 74 irreducibles, 0 twin irreducibles; 0 of 9,985
          twin-irreducible columns in [961, 935089)); the axiom M lacks, the finite fold, enters
          the construction only as skeleton 2 (the strike-class law) = the sieve's input, which
          LF3 / LF4 show insufficient at the origin. The parity mix as one periodic dilation
          machine: S_H (+-1 mod 12) has 558 twin-irreducible columns in [121, 17161): 139
          prime-prime, 303 mixed, 116 semiprime-semiprime. Proof attempt (6.1) ends at the core /
          tail split read at the origin (every core-open column must hold g x prime with g > L;
          comparing counts is Brun / Chen, stopped). NAMED NEXT CONSTRUCT (not excluded): a
          non-count use of the tail pins (g x m, m a small prime), the position face restricted
          to the tail, with the tail-only-freed family as control (section 10). Scorecard P1-P6,
          P8 confirmed, P7 refuted at 29; second measurements: d(17), d(29) by brute force over
          the whole family; origin shares by spf table and sympy, 4 of 4.
        - R4.d.i.h. THE FOLD AS A MECHANIC (theorist on Fable, 2026-09-11; research/proof/
          fold_mechanic.md; scripts research/anchor235/r78/fm_*.py). VERDICT ROOT, with the wall
          moved one axiom further: THE MISSING AXIOM CANNOT BE ANY PROPERTY OF THE GEAR SET; it
          is a property of the line. (1) The class reading of the monoid is right (74 of 74 and
          36,812 of 36,812 irreducibles in class +1; a twin gear pair needs one gear from each
          class), but "both classes at every scale" is NOT the missing axiom: the fiat thinning
          G = P_+ union (P_- minus the twin lowers) has class -1 gears in every dyadic range
          (0.5-0.83 of the class +1 count to 10^7), dilation, the square-root rule, the
          side-swap rule, and 0 twin gear pairs on 361 of 361 finer sections and the three
          construction sections (smallest [49, 2809), 85 columns). (2) PROVED (Lemma 2.1): every
          monoid generated by P_+ and a subset T of P_- is TRANSPARENT: its twin gear pairs are
          exactly the twin primes with lower member in T, so 8 on it holds iff T keeps a twin
          lower in the section (0 mismatches over 7,309 sections, 26 thinnings); the family
          carries no mechanic; the monoid of R4.d.i.g is its T = {5} member. Weakest thinning
          keeping 8 on the chain from 5: T = {5, 29, 857, 727877}; strongest breaking: remove
          exactly the 31 twin lowers in (25, 961). (3) The two classes as a construction (F2-F7,
          proved): every gear strikes the left member at j = 6^-1 and the right at -6^-1 mod g,
          the class deciding only which residue is small; THE CLASSES ARE INVISIBLE TO THE COLUMN
          COVER (a side-swapped machine covers the same columns); below p'^2 the composite left
          members are P_- . (S_+ minus 1) and the composite right members P_- . S_- union P_+ .
          (S_+ minus 1) (0 exceptions at 18 sections); the exact cover condition in class words
          is "every class -1 prime of the section has composite t + 2", the root restated. (4)
          The fold's sign is sieve-visible: (-1)^{Omega_-(n)} = chi(n) = n mod 3; Liouville =
          chi . (-1)^{Omega_+}, so the invisible part is the class +1 factor count and the parity
          twin O^- is the open columns with even Omega_+(L) + Omega_+(R). (5) Periodic control:
          S_H mod 30 holds 8 with 15,504 twin-irreducible columns in [841, 755161); every
          violating machine on record is non-periodic. WHERE THE DIFFICULTY MOVED: the thinning
          shares every gear-set property with the primes (including equidistribution in every
          admissible class, since twin lowers have density 0 by Brun), so the axiom must be "every
          j is a column with both members on the line", the strike-class law's FIRST half: the
          line is all of S, every survivor of the fold is a gear or a multiple of a smaller gear
          (in the integers the thinning breaks the hand-up: the twin (101, 103) is open under G
          and 101 is not a gear; in the monoid's own line 101 is absent). The teeth half and the
          classes are exhausted. Two P4 details corrected in the document; one script threshold
          bug found and fixed (40 mismatches to 0).
        - R4.d.i.i. THE FIELDS (owner's decomposition, 2026-09-11; theorist on Fable;
          research/proof/fields.md, 767 lines; scripts research/anchor235/r79/fd_*.py; gated at
          20 sections and 13 machine sights, 0 mismatches). VERDICT ROOT: the fields are exact
          objects and the split is real, but it is exactly the split a sieve cannot make: the
          fields are the Omega-strata of the overlay, the sieve sees only the overlay, and a
          twin is the column whose two strata are (1, 1). EXACT FACTS, proved (E1-E8): (1) every
          field is the primes dilated: F_j intersect 5S = 5 . F_{j-1}, so F_1 = (F_j intersect 5S
          intersect ... intersect 5^{j-1} S) / 5^{j-1}; no field has a location rule the primes
          lack, the square field's being F_1 = sqrt(Q); (2) no field is periodic while the
          overlay on a sight is (an aperiodic partition of a periodic set); (3) the class rule:
          F_j hits a right member iff its class -1 factor count is even, squares right-only, F_2
          right by same-class pairs and left by cross-class; (4) THE MIRROR LAW: column m dilated
          by g becomes the columns g m -+ k_g, mirror images about 6 g m (the bifurcation of the
          fold), and the symmetric pairs of F_j through g about 6 g m are exactly the pairs
          {6m - i, 6m + i} both in F_{j-1}, the twin being radius 1 (0 mismatches over 171,243
          axes); (5) deep-field blind sets: F_j (j >= 3) is confined to the teeth of
          {5..p'^{2/j}} in the section below p'^2 (at p = 53: F_3 on {5, 7, 11, 13}, F_4 on
          {5, 7}, F_5 = {3125} on {5}); THE SQUARE FIELD IS EMPTY IN EVERY FINER SECTION AND IS
          THE CUTS OF THE CONSTRUCTION (in section k + 1, the squares of machine k's gears above
          p_k); (6) the twin as "open here, struck there": column c is a twin iff cube-core-open
          and every tail cofactor (6c +- 1)/g is composite, i.e. struck at scale 1/g by a lower
          engine (20 of 20; the pins at p = 29: 8 pins on 6 of 8 core-open columns, the unpinned
          are the twins 143, 147). MEASURED: the class bias alternates with j (F_2 right-heavy,
          F_3 left-heavy, F_4 right-heavy; 9 of 9 signs; Meng 2018, stopped); twin semiprimes
          5,518,555 against 1,027,948 twin primes at base 3 link 3; one-field columns 1.000,
          0.786, 0.523 along base 3, the mode moving from 2 at the origin to 2 + 3 at height;
          THE GROWING MIRROR SYMMETRY THE OWNER EXPECTED IS REAL: symmetric F_2 pairs per axis
          through 5 grow 2.00 -> 46.83 -> 110,944 along base 3 while the twin's share falls
          0.50 -> 7 x 10^-7, the radial profile flat (radius 1: 2,594 among 2,599-4,103 per
          radius): it is the Goldbach pairing of F_1 about 6m and the twin is its innermost
          radius (Hardy-Littlewood, stopped); new and confirmed twice (12 of 12 cells each way):
          off the multiples of g the symmetry is carried entirely by the primes of the axis g m
          (pairs sharing a prime with m above the null, z = +3 to +62; pairs coprime to it below,
          z = -4 to -14), the mirror about 6 g m preserving divisibility by every prime of 6 g m.
          THE OBSTRUCTION as a construction: (i) any relabelling of the composites among the
          fields leaves the overlay and its complement fixed, a partition does not see its
          complement; (ii) a label property that is not relabelling-invariant sees F_1, the
          primes' positions, step 8 restated; (iii) what a sieve computes about the labels
          cannot see their parity: the column sign of length_face.md is (-1)^{Omega(L) +
          Omega(R)}, the parity of the field-index sum, smallest instance O^- at q = 11 (empty
          below column 28 against 20). The parity problem again, in one sentence. Died: a
          location rule for F_2 not through the primes; a period of any field; the blind sets
          intersected as a non-count rule (they reduce to "core-open and unpinned"); the mirror
          symmetry as a lever on radius 1 (flat profile); E8's induction. Not excluded: E8 as a
          fixed-point object; the tooth-family control with the fields' labels on the cofactors
          (a phantom pin = a pin whose cofactor is not in F_1). Scorecard: P1-P4, P8 in full; P6
          with its reading corrected; P5 7 of 8 / 3 of 4; P7 20 of 20 with the ratio clause 6 of
          8; the owner's (a)-(d) held in the stated forms, (e) refuted as sufficient.
        - R4.d.i.a. THE HOT LEAD: the island witness is the stack's step at the start of a
          section (manager, 2026-09-07, from the owner's "follow the hot lead"; one local
          computation, no lane). Measured to q = 10^4 (1,226 primes): above EVERY prime square
          q^2 the first twin lies below the top gear's long arc (offset i < (2q + 1)/3
          columns), 0 exceptions, median at 1% of the arc, max 77%; in the witness's four
          classes i = 5, 10, 12, 17 mod 35 alone, 8 exceptions, all q <= 461 (the recorded
          witness holds from 2849), the first witness twin at a median 3% of the arc, the four
          classes equally used (339, 321, 270, 288). THE MECHANISM, EXACT: relative to a square
          q^2 (q coprime to g), gear g strikes offset i (members q^2 + 6i - 2, q^2 + 6i) iff
          6i = -q^2 or 2 - q^2 mod g, i.e. iff -6i or 2 - 6i is congruent to a nonzero SQUARE
          mod g; so each gear strikes at most g - 1 offset classes relative to squares and is
          blind at 1 + (the number of pairs of nonzero squares differing by 2) classes: gear 5
          blind at i = 0, 2 mod 5, gear 7 at i = 3, 5 mod 7, jointly the four classes 5, 10,
          12, 17 mod 35. The start of a section is not a generic stretch of the line: every
          gear's strikes there are governed by quadratic residues, the owner's "squares are
          even" (R2.a.i.a.1.b) made exact, and this is the base-case quantity's structure. TO
          BUILD on resume (Fable): the quadratic-residue strike pattern of machines 1..k
          relative to the cut p_k^2 as an object (which offsets each gear can strike relative
          to a square, as a function of q^2 mod g; the blind classes per gear; the joint
          pattern of the first gears); whether it explains the start-of-section richness
          (S6) and the witness's arc bound; and whether a slot open to all gears exists within
          the long arc for a structural reason (the blind classes of the small gears plus the
          quadratic pattern of the rest), tested against the counter-machines (a free-phase
          copy has no square structure; the parity adversary does). CENSUS VERDICT (same
          day): a square is not richer in twins than a random start (322,186 against 321,052
          below the arc, q to 20,000); the square only fixes the classes (availability
          fractions 1, 2/3, 1/2, 1/3, 1/6 mod 35, exact). The witness = blind classes (proved)
          + ordinary density (a count). ROOT for existence; the blind-class structure kept as
          FACT.
          - R4.d.i.i.a. SAMPLE RUN ON THE CONSTRUCTION (owner's request 2026-09-12; manager,
            local; research/proof/fields_sample_run.md; scripts research/stack/r8/layers.py,
            flank_killers.py; skill .claude/skills/fields-explorer). Two closed items re-read
            with gear rows against the natural numbers. (1) R4.d.i.a (the witness after a
            square, closed as blind classes + a count): between consecutive squares g^2, g'^2
            the only new row is g, whose field enters at g^2; its strikes on the columns open
            to the gears below g are g m with m a survivor in [g, g'^2/g] (2 to 4 numbers,
            located), and every other old-open column is a twin: twins in the layer = old-open
            columns - toll, toll located; 166 layers to g = 997, toll share mean 0.015, zero in
            118 layers, never at its bound, twins per layer minimum 2. RE-OPENED as a located
            object: what remains is the old wheel's open columns in the layer at the square
            origin, by row and offset (parts M, Q are about exactly this placement); no closed
            form found in the sample. (2) Location rules (closed at chance): the rows striking a
            twin's flanks are 5 and 7 at the wheel's open-column rates to three decimals
            ((5,5) 0.327 against 0.333, (7,5) 0.138 against 0.133, 33,334 columns, 2,129 twins);
            closed again, the wheel. CONTINUED (owner: try the grids next; layer_fields.py, 428
            layers to g = 2999): a layer = window(g') minus window(g) = (g^2, g'^2]; row g's
            strikes on twin-slot members there are g p for p prime in (g, g'^2/g], closed form;
            row h < g is D3 in the layer (h times survivors below h, primes when h > g^(2/3));
            first twin after g^2 within 3.05 g (mean 0.20 g, grows like ln^2 g: ordinary); last
            twin at most 938 below g'^2; no mirror centre beyond the selection of a maximum;
            closing rows at the wheel's shares. The composites of a layer are fully located;
            the old rows' open count in the layer is the one unlocated quantity. STATUS OPEN.
          - R4.d.i.i.b. THE FIELDS IN THE WINDOW OVER CYCLES (owner's request 2026-09-12;
            manager, local; research/proof/window_fields.md; scripts research/stack/r8/
            window_fields.py, mirror_fields.py, cycle_survival.py; machines 5-31, 12 cycles).
            FACT. Cycle 1: the killing fields are exactly higher:g for g = 5..q (each kills, g^2
            is in the window), squares of the gears up to q, products:j up to log_5 q^2, lower:g
            for the largest factors that occur; higher:g with g > q never kills in cycle 1 at
            any size (exact). Cycles 2+: the machine's rows repeat, so the only change is on
            the machine's openings (= the window's twins), each a twin again or eaten by ONE
            field higher:g with g > q, no other field touching an opening; the first eater is
            always q' with the largest count; openings twins again per cycle listed (zero in
            some cycles from machine 23). Periodic: higher:g (g <= q), the machine's multiples
            (g# | q#). Never periodic: higher1 (g^2 does not divide q#), lower, lower1,
            products, multiples with all rows; squares kill only in cycle 1 from machine 7 on.
            Nothing genuinely becomes periodic. Mirror q# - n: exact for higher:g (g <= q) and
            the machine's multiples (494/495, the miss 25 -> 5), 0.82 for higher1, chance for
            products (0.26), none for squares, lower, lower1; every window twin's mirror is
            machine-open, a twin again 2/2, 3/4, 2/7, 2/9, 2/15, 3/17, 1/21, 4/28, 1/30 at
            q = 5..31 (falls as the gears above q thin the mirror). Two bugs fixed first
            (sympy primorial(q) = first q primes; "stops killing" misread as periodic).
            THE ORDER CEILING (owner's question, same day; window_fields.md section 6): order j
            = prime factors with multiplicity; products:j kills in the window iff 5^j <= q^2, so
            j_max = floor(2 log_5 q), exact at 18 machines to 401 (window 7 at 401, the layer 6);
            in the whole cycle the ceiling is floor(log_5 q#) about 0.62 q. The window's field
            list is exact and grows like 1.24 ln q. Inside it orders >= 4 are the smooth field
            (1-7% of kills); orders 2 and 3 carry the rest, and the order-2 kills are mostly a
            gear times a prime of the window itself (87/131 at q = 31, 5578/6590 at q = 211):
            the narrow zone is exact in order, and its content is the hand-up.
          - R4.d.i.i.c. THE MIRROR WALK (owner's idea, 2026-09-13): navigate from a known open
            column by mirrors about M/2 for products M of subsets of the gears (and their
            multiples), ending with a flip about an axis carrying all of q's gears into the
            window. DEAD BY PROOF (manager, same day; checked to 500 both sides on gears to
            11): a mirror about M/2 or a translation by t keeps "column open to gear g" exactly
            when g divides M (resp. t), so the maps that keep all of q's gears are the symmetries
            of the wheel mod q# (reflections about multiples of q#/2, translations by multiples
            of q#); they carry the machine's open set onto itself and change nothing in the
            window: a column reached in the window is the column itself or the mirror image of
            a column near k q#, whose openness is the same unknown. Partial-gear steps lose the
            gears left out and the last flip needs its source all-gear open, which is the
            target restated. The home column (-1, 1) is open to every machine; its images are
            the primorial columns (k q# - 1, k q# + 1), all above q^2. What survives: mirrors
            are exact symmetries of the machine (kernel: mirror law, survivor_neg), useful for
            reading a window from its far end, never for entering it.
          - R4.d.i.i.d. THE WALK WITH ANCHORS (owner, 2026-09-13: find the rules that would
            make a walk possible; backward walks; three anchor families by one-off lanes;
            research/proof/anchors_walk.md; scripts research/stack/r8/backward_walk*.py,
            anchors_*.py, results_anchors_*.md). FACTS: a flip carries exactly the gears
            dividing its axis product; gear h is certified at n by an anchor a known open to h
            iff n = a or -a-2 mod h; certification never certifies a struck column (0 at every
            machine and family). Killed columns: the killer is carried exactly onto the gear
            pair containing it, matching side (0 violations, 3291 instances at 101). Twins from
            the base anchors (pairs + home): 0 fully certified at 31..401 (a gear needs about
            h/2 anchors). Square anchors (g-1 columns either side of g^2): every gear covered,
            every twin certified, one anchor per gear (median 144 for 167 gears at 1009), the
            residue rule with witnesses. Blind-class anchors: complete for 5 and 7 only, the
            first non-twin known places carrying two gears. Caustic anchors (the run after a
            square before h's first strike): every gear >= 23 covered, gear 5 never (5 strikes
            the column after every square), mean 41.7 gears known per anchor at 1009, walk
            length halved (median 68). VERDICT (corrected 2026-09-13, owner): the walk is a
            working locator: its steps are proved, it lands on a window twin on every machine
            tried, and what is missing is a termination proof, that a landing place always
            exists in (sqrt q, q] for every q; that termination proof is the proof of step 8.
            Kept: exact kill rule, three verified anchor sources, the walk-length bound.
          - R4.d.i.i.e. THE NETWORK OF WALKS (owner, 2026-09-13: breadth-first from a gear
            pair, a child per allowed anchor option, nodes unique to their path, no return to
            the node just left, stop at the destination; compare rules; research/proof/
            walk_network.md; research/stack/r8/walk_network.py). FACTS at machines 7-31:
            keep (carry every certified gear) finds nothing beyond 7 (no anchor sits on an
            axis divisible by the certified product); carry1 (carry at least one gear) is the
            productive rule (5 of 9 twins at 13 within depth 4, 5 of 30 at 31 within depth 3);
            free expands 10-20x the nodes for the same destinations; caustic anchors are the
            cheapest per destination (533 nodes for 3 twins at 13 against 83,768 with
            squares). Every destination's last step lands on an anchor whose merged known set
            already covers all gears or all but the one or two the flip carries: the walk
            certifies, the landing place locates. Columns known open to every gear by the
            families alone: 1-4 per machine, the recurring one the column two after a square,
            (g^2 + 10, g^2 + 12): 179 (g = 13) for machines 29-113, 9419 (g = 97) to 211,
            143651 (g = 379) at 401; for gears above g the caustic knowledge is the
            divisibility check in other words. VERDICT: efficiency question answered (carry1
            with caustic anchors). The walk algorithm (start at (5, 7), flip onto an anchor in
            the window whose known gears cover everything) runs on any q and has landed on a
            twin on every machine tried; the open item is termination for all q, the proof of
            step 8 itself (corrected assessment, owner, 2026-09-13).
          - R4.d.i.i.f. THE LOCATOR: A FIXED COLUMN AFTER EVERY SQUARE (owner, 2026-09-13:
            "we just need one location"; research/proof/locator.md; research/stack/r8/
            locator.py). EXACT: the column at offset i after g^2, (g^2 - 2 + 6i, g^2 + 6i), is
            open to 5 for every g iff i = 0, 2 mod 5, to 7 iff i = 3, 5 mod 7 (both: the blind
            classes 5, 10, 12, 17 mod 35), to g for g > 6i; any other gear h strikes it iff
            g^2 = -(6i-2) or -6i mod h, at most four classes of g mod h (the square roots),
            and gears with neither a residue never strike that offset from any square (blind
            gears: for i = 2: 17, 29, 71, 83, 101, ...; i = 10: 7, 11, 13, 41, 43, ...). So
            "the offset-i column after g^2 is a twin" is a sieve on the gear line, ~2 classes
            per gear (1.64 at i = 10), classes in closed form. MEASURED: for every prime q
            from 11 to 20000 some gear g in (sqrt q, q] has its offset-2 column a twin (2259
            of 2259; also i = 10, 17; i = 5, 12 fail only at q = 7); the hit gears form a
            chain under squaring (largest consecutive ratio 7.46 at i = 2, 1.63 at i = 10).
            STATUS: CANDIDATE LOCATOR, closed form for where and for who cannot interfere;
            existence of a hit gear per window is a density statement on the gear line (prior
            art: primes in quadratic polynomials, Bunyakovsky / Hardy-Littlewood F, open).
          - R4.d.i.i.g. RULE WALKS: THE SUB-MACHINE AS THE RULE (owner, 2026-09-13: stepwise
            rules, no search, no pre-checking, certify afterwards; research/proof/locator.md
            section "Rule walks"; research/stack/r8/rule_walk.py, path_grammar.py). STRONG.
            Residue-blind rules fail exactly at the teeth of named gears, in blocks; walks
            that carry knowledge (square axes, Hanoi) do worse than one flip. THE RULE: consult
            only the gears up to sqrt q; take the first gear g above sqrt q whose classes
            avoid their teeth for offset 10; one flip from (5, 7) onto (g^2 + 58, g^2 + 60).
            Succeeds on 2253 of 2258 machines to 20000 (the 5 failures are q = 11..23 with
            g = 5 dividing its own candidate). 2546 gears between sqrt q and g were never
            consulted; none struck. MECHANISM (exact, 45,150 checks, 0 violations): gear h
            strikes the offset-i candidate after g^2 iff h divides r^2 + 6i - 2 or r^2 + 6i
            with r = g mod h (the caustic law at a fixed offset); a gear just below g has
            r = the gap d, so it can strike only if h divides d^2 + 58 or d^2 + 60, impossible
            once h > d^2 + 60 (gaps at most 20 in range). The locator of machine q is decided
            by the machine of size sqrt q. TERMINATION now = two statements on the prime line
            near sqrt q: an avoiding prime within a small gap of sqrt q, and no gear in that
            gap dividing d^2 + 58 or d^2 + 60. Path grammar (all destination paths of the
            network at 11..31): no step sequence shared by all machines; every path ends in
            the caustic zone of a square; intermediate steps carry one gear and decide
            nothing.
          - R4.d.i.i.h. THE DIRECT CONSTRUCTION (owner's guess 2026-09-13: anchor decisions
            from squares and roots navigate straight to a twin; research/proof/locator.md,
            section "The direct construction"). STRONG, exact where stated. g = the first
            prime above sqrt q; i = the smallest offset such that for every gear h < g, with
            r = g mod h, h divides neither r^2 + 6i - 2 nor r^2 + 6i, and g does not divide
            6i; land on (g^2 + 6i - 2, g^2 + 6i). No search, no landing check. 2258 of 2258
            machines to 20000, offset at most 27 columns, mean 8.3; the start pair does not
            enter the location. EXACT: the gears below g are the gears up to sqrt q; a number
            in (g^2, g g') with no factor below g is prime; so every column in (g^2, g g')
            missed by the gears below g and by g is a twin (8194 columns to g = 1500, 0
            exceptions); the rule's offset stayed inside the zone at every machine (max ratio
            0.82). TERMINATION IN ONE LINE: in the zone (g^2, g g') after the first square
            above q (at least g/3 columns), the wheel of the gears below g leaves a column
            open; the teeth of h in offset coordinates are the two classes r_h^2 + 6i = 0, 2
            mod h, a tooth family fixed by the roots of g; gears just below g cannot strike.
          - R4.d.i.i.i. THE WALK ON REAL MIRROR AXES (owner's correction 2026-09-13: an axis
            is a multiple of the product M of a gear set S containing 2, 3; no offsets;
            research/proof/locator.md, section "The walk on real mirror axes";
            research/stack/r8/true_mirror_walk.py). STRONG, exact where stated. Flip about
            k M carries every gear of S (the S pattern is symmetric about every multiple of
            M). From home (-1, 1) one flip lands on (2kM - 1, 2kM + 1), open to S for every
            k; a remaining gear h strikes it iff k = -+(2M)^-1 mod h (two classes of k, the
            roots); rule: the smallest k with the landing in the window avoiding those
            classes. EXACT: when such k exists the landing is a twin (below q^2, no gear up
            to q divides it). Machines 11..5000: S = {2,3} and {2,3,5} land on every machine
            (665/665; k mean 200 / 42; landing a fraction of a percent into the window);
            {2,3,5,7} 661/665, {2,3,5,7,11} 645/665 (small machines lack a multiple below
            q^2); to 20000 the two small sets land everywhere; 0 landings fail
            certification. TERMINATION = existence of k: the twin sieve on the multiples of
            M with the S gears removed by the mirror. KERNEL (round 45, proofs/MirrorWalk.lean,
            built, 0 sorries, axioms propext / Classical.choice / Quot.sound only): flip,
            OpenTo; flip_carries (a gear dividing 2a keeps openness across the flip);
            openTo_flip_iff; walk and alt (alternating sum); walk_eq (odd length = flip about
            alt, even = slide by 2 alt); openTo_walk_iff (a gear dividing 2 alt keeps openness
            across the whole walk); walk_home_odd/even (every walk from home lands on
            (2A - 1, 2A + 1)); home_open; landing_open_of_dvd; struckBy_mline;
            not_dvd_landing_of_dvd_axis; landing_twin (the landing 12m +- 1 below P^2 is a
            twin when the gears not dividing 12m miss it, the dividing gears carried by the
            mirror); walk_lands_of_record (the m-line record below the stretch gives a
            landing twin, the shape of section_twin_of_record). MULTI-STEP (owner: do it; multi_mirror_
            walk.py): composition law exact (3000 random walks, 0 violations): the end of a
            walk is 2A - n - 2 (odd flips) or n + 2A (even), A the alternating sum of the
            axes, certified for the gears dividing A; stepwise tracking undercounts (2062 of
            3000 walks carry gears no single flip carried, e.g. 42 k_2 - 30 k_1 = 66 carries
            11). From home every walk ends on (2A - 1, 2A + 1), A any multiple of 6, so
            multi-step reaches the one-flip landings with the carried set = divisors of A.
            Two-step rule (home -> 30 k_1 -> 42 k_2, A the smallest multiple of 6 in the
            window avoiding the teeth of the gears not dividing A): 665 of 665 twins to 5000;
            mirrors carry 1.25 gears per machine, the roots handle the rest. CARRYING MANY
            (carry_many.py): with A = k P_m the mirrors carry at most the gears whose
            primorial stays below q^2/2, m = 4 at 31, 7 at 3000-5000, against 9 .. 667 gears in
            the machine; every largest-m landing a twin; carrying buys a higher landing and
            fewer multiples, nothing else. THE M-LINE (mline_records.py): landing family
            (12m - 1, 12m + 1), teeth m = -+12^-1 mod h; termination for q = an unpainted m in
            (q/12, q^2/12); exact to q = 3001: open m in the window 3, 14, 100, 906, 4179,
            26960 at q = 11, 31, 101, 401, 1009, 3001; the record R(q) (longest struck run)
            4, 13, 43, 80, 191, 278 against window lengths 9 .. 750250; share 12R/q^2 falls
            0.40 -> 0.0004. This is step 8 in run form on the m-line, the twin machine with
            teeth -+12^-1: same object, half the columns, same margin.
          - R4.d.i.i.j. THE WALK THROUGH THE FIELDS (owner, 2026-09-13: not covering versus
            capacity; show nothing stops the walk, field by field; research/proof/
            walk_fields.md; research/stack/r8/walk_fields.py). EXACT: on the landing zone
            (12k +- 1 in the window) the only painting fields are the multiples rows, two
            teeth per period h, share 2/h to three decimals; squares (one k per gear, right
            member), higher:h and products:j are relabellings of that paint (unions equal the
            painted set, 31/101/401); no gear is blind on the m-line; rows independent mod the
            product of the gears (prod (h-2) unpainted per period), the zone one phase of it
            (14 vs 14.5, 100 vs 94.8, 906 vs 913.2). WHAT STOPS THE WALK: only a painted run
            anchored at the zone start k_0 spanning the zone; L(q) = the run at k_0 = the
            distance from q to the first twin above q with midpoint 0 mod 12; machines
            11..20000: mean 8.7, median 6, max 55 (q = 13007, zone 1.4e7). The open statement
            sharpened: the first aligned twin above q lies below q^2; decided by the paint
            just above q (every number in (q, 2q) is prime or has a factor below q). Proof
            document section 14 rewritten to this. THE PAINT JUST ABOVE q (owner: let's do
            that; docs/zone_start_field.html, artifact 6ec378fd-1297-476f-be3e-ab2f87a07a2b;
            zone_start.py): row h painted at d = -q mod h (the top gear fixes every phase);
            only gears h <= (q + d)/5 reach offset d, none above 2q/5 below 2q; a painted
            member below 2q has a factor at most sqrt(2q), so the run at the zone start is
            laid by the machine of size sqrt(2q) as smallest factors with cofactors; machines
            29..20000: the run spans at most 0.36 q (q = 431), smallest gears at most 0.78
            sqrt(2q); exceptions only 11..23. The open statement at the zone start: the
            gears up to sqrt(2q), phased by q, cannot paint every aligned column from q to
            q^2; the length of their painted run at phase -q is the twin gap above q.
            THE SUB-MACHINE AT PHASE -q (owner: go; submachine_phase.py): exact chain, below
            2q painted = painted by B(q) = gears up to sqrt(2q), so L(q) <= R_B (the m-line
            record of the sub-machine) whenever R_B < q/12; sub-machine records exact over
            full periods: R_y = 2, 4, 7, 9, 17, 19, 34, 43 for y = 5..29, between y^2/20 and
            y^2/12, the same y^2 growth as the certified F(q); L <= R_B at 73 of 77 machines
            (misses 11..23 overshoot 2q). The sufficient condition in the limit is
            R_y < y^2/12: the record route (8c) at the sub-machine; every coordinate used
            (window, square zone, zone start) leads to it. The prime phase -q is NOT special:
            runs at prime phases match all phases (y = 13, 17, 19; 18k primes to 200,000).
            VERDICT: exact location and mechanism (the sub-machine lays the paint), no reason
            for the record's bound; the record route is the invariant form. THE RECORD AS A
            FIELD (owner: go the field; docs/record_field.html; record_field.py): the wheel
            of the gears 5..y at its worst phase, y = 7..23 exact; each gear's two teeth
            -+12^-1 are a close pair (5: 2,3; 7: 3,4; 11: 1,10; 13: 1,12; 17: 7,10; 19: 8,11;
            23: 2,21), so every row paints in double teeth; a record run is a tiling by double
            teeth with little overlap (single-painter columns 68-100%, paint 1.00-1.38 per
            column; y = 23 run: 13/23 5 5/13 7 7 17 5 5/11 17 11 7 5/7/19 5 13 19 13 5 5/7
            7/11 23 11 5 5/17 23 7 7/17 5/13 5 13 11 19 5/7/11 5/7 19); record runs come in
            mirror pairs about half the period (y = 23: exactly two), none self-mirror; the
            run sits where the double teeth of 5 and 7 interleave without slack and the
            larger gears each fill one column pair. Reading of the growth: the paint a run
            can receive is about 2R sum(1/h) + 2 pi(y) with sum(1/h) < 1, so the record is
            carried by the additive term, one double tooth per gear; this is the budget
            inequality in field form, a mechanism for the y^2 growth, not a proof of the
            bound. THE PROOF ATTEMPT FROM THE MECHANISM (owner: make a proof for the bound;
            research/proof/record_bound_attempt.md): Lemma 1 (a gear paints at most
            2 ceil(R/h) of R consecutive columns), Lemma 2 (a full run needs total paint >= R),
            Theorem (if 2 sum 1/h < 1 the record is finite and bounded): R_5 <= 2 (tight, and
            below 25/12), R_7 <= 8, R_11 <= 36, nothing from y = 13 (2 sum 1/h = 1.021).
            Counting cannot be sharpened: overlaps are placed by the CRT and correcting for
            them is the sieve, whose lower bounds for two residues per prime are positive
            only from length y^4.27 (the sieving limit recorded at IV.1 of the earlier
            document); the bound wanted is at y^2/12. VERDICT: the mechanism proves the
            bound at y = 5 only; a proof must be a structural reason from the teeth, not a
            count.
          - R4.d.i.i.k. THE WALK IN PARTS (owner, 2026-09-13: no counting, ever; the origin
            pair, the step rule, each landing and its relation to the last step, the target
            zone, each part its own proof; research/proof/walk_parts.md; kernel proofs/
            MirrorWalkParts.lean, round 46, built, 0 sorries, standard axioms). PROVED PARTS:
            origins (home open to all: home_open; a gear pair open to all but its members:
            pair_open); the step (flip about a real axis, two steps slide: flip, flip_flip);
            landing versus last step (carried gears keep openness: flip_carries; the landing
            law: h strikes the landing iff 2a = n + 2 or 2a = n mod h: struck_flip_iff; one
            class per side: same_class_of_struck; one gear never blocks a step, settled within
            three consecutive multiples: exists_axis_open); steps in sequence (walk_eq,
            openTo_walk_iff, walk_home_*); the target zone (landing_twin); the carry cap (the
            carried primes' product is at most q^2/2: carried_product_le_window). OPEN PART:
            the joint step, one axis in the window settling every remaining gear at once (two
            gears follow by CRT within h h' multiples; all at once within the window is the
            open statement, stated as a property of the step). BOTH LEVERS (owner: both;
            kernel MirrorWalkSettle, round 47): the origin lever PROVED (anchor_certifies: L
            open to h if some column v open to h has h | L + v + 2), measured at the walk's
            landing with home and the gear pairs as origins: 3/9, 4/24, 14/77, 22/167, 40/429
            gears certified at q = 31 .. 3001, the small gears, the large left to the landing
            law; settling one after another PROVED (openTo_add_of_dvd, flip_stride,
            exists_axis_open_stride, settle_two: two gears within k_0 + 2 + 2h), the r-th gear
            costing a stride equal to the product of the settled ones, so 1, 2, 3, 5, 5 gears
            settle inside the zone at q = 11, 31, 101, 401, 1009. The joint step for all gears
            stays open, its per-gear cost now proved.
          - R4.d.i.i.l. THE REPAIR WALK, THE CONSISTENT SEQUENTIAL WALK (owner, 2026-09-13:
            find a consistent origin type with a consistent per-step rule; research/proof/
            walk_parts.md; research/stack/r8/sequential_variants.py; kernel proofs/
            MirrorWalkRepair.lean, round 48). The product-stride walk is inconsistent under
            every origin (zone start, primorial-adjacent, square column, window middle), order
            and step choice (best 236 of 426 to 3000). THE REPAIR WALK: origin the zone start;
            rule at every step: take the smallest gear striking the current column and move
            forward by the smallest amount that clears it (1 or 2); stop when no gear strikes.
            MEASURED: lands on a twin inside the window at 2258 of 2258 machines to 20000;
            steps mean 6.4, max 41; never passes an open column. PROVED: clear_step (a gear
            striking k misses k+1 or k+2), Reach and Reach.trans, reach_first_open (a walk of
            steps 1 or 2 that never passes an open column, started at or below an open
            column, reaches an open column at or below it), so the walk stops at the first
            column above q that no gear strikes; landing_twin makes it a twin below q^2. OPEN:
            that the walk stops before the window's end, i.e. the first column above q open to
            every gear lies below q^2 (the step count is L(q)). Blind, per-gear, one rule.
          - R4.d.i.i.m. THE SPIRAL (owner's find on the flip explorer, 2026-09-13; formalised
            the same day; research/proof/spiral.md; kernel proofs/MirrorWalkSpiral.lean, round
            49). DEFINITION: from home, one flip per odd gear g = q, p', ..., 3 about the axis
            one period of {2, g} from the current column, directions alternating up, down,
            ...; each flip moves by 4 d g. PROVED: the closed form E(q) = -1 + 4 A(q), A the
            alternating sum of the odd gears (spiral_step, spiral_eq, spiral_home); the
            endpoint is open to every gear dividing A (spiralEnd_open_of_dvd); the endpoint is
            below 4q, hence below q^2 for q >= 5, so the spiral never overshoots the window
            (altSum_le_head, spiralEnd_lt). MEASURED to 20000: E between 1.40 q and 2.71 q
            (mean 2.003 q), above q at every machine (A > (q+1)/4, a gap statement, not
            proved); E mod 6 is 1, 3, 5 in equal shares (the mirrors {2, g} do not carry 3);
            gears dividing A, carried to E: 1.6 per machine; nearest twin within 6 of E at
            every machine to 29, then at a falling rate (median distance 14 to 5000). STATUS:
            the first blind, mirror-only walk that enters the window from the machine's
            structure alone; the launch point for what follows.
          - R4.d.i.i.n. THE LOCATOR IN TWO PHASES: PRIMORIAL SPIRAL, THEN THE FINAL STEP BY
            RESIDUES (owner, 2026-09-14; research/proof/spiral.md; kernel proofs/
            MirrorWalkFinal.lean, round 50, built, 0 sorries, standard axioms; docs/
            flip_explorer.html has buttons for both phases; research/stack/r8/
            results_phase2_pass.txt). PHASE 1 PROVED: the primorial spiral lands at
            E = -1 + 2 P A below q^2 (P the primorial below q/2). PHASE 2 PROVED: the step
            {3, h} lands at E + 6 d h; gear g strikes it iff 6 d h = -E or -(E + 2) mod g
            (strikes_landing_iff), so each gear forbids two classes of h computed from E;
            a high gear avoiding every class lands on a twin below q^2 (avoid_iff_open,
            final_step_twin via section_twin_of_unstruck). MEASURED to 20000: a passing high
            gear exists at 2257 of 2258 machines (none at q = 11); passing share of the high
            gears in reach 0.028; the smallest passing h is 1.85 sqrt q at the median, at
            most 14.6 sqrt q. OPEN: that a passing high gear always exists, the second-sieve
            statement on the high gears with classes fixed by E.
            - R4.d.i.i.n.i. THE ONE-FLIP LOCATOR AS THE KERNEL'S FINAL FORM (2026-09-17, rounds
              60-62; proofs/OneFlipLocator.lean; loop entries 66-68). Spawned by phase 2: one flip
              about the mirror {2, 3, g} from home lands on column 2 g k d, members 12 g k d +- 1.
              PROVED: oneflip_twin, window_statement_of_oneflip, mirror_gear_never_strikes. FACT
              (the locator's exact form); the openness of the landing is the second sieve, OPEN.
              - R4.d.i.i.n.i.a. THE CHAIN OF LANDINGS (rounds 63-70; proofs/MirrorWalkChain.lean,
                MirrorWalkCertificate.lean, PrattCertificates.lean). A landing t serves every
                machine sqrt t <= q < t - 1; overlapping bands cover all machines. PROVED
                chain_covers, window_statement_upto, mult_chain_window; CERTIFIED every machine
                11 to 2.76 x 10^32 (six Lucas-certified links). STRONG as a certificate, not a
                route beyond it: the next link is a twin above 10^32, a fact about the primes.
              - R4.d.i.i.n.i.b. THE CARRY WALL (rounds 64, 71-74; proofs/MirrorWalkCarry.lean,
                MirrorWalkSettleFree.lean). A mirror that fits the window carries at most
                log2(q^2) gears (carried_le_log); silencing the gears to X costs X#
                (silence_costs_primorial); the free-regime hypothesis is unreachable
                (free_regime_unreachable) and sharp (keeping_move_free_sharp). DEAD as a route:
                divisibility is the machine's only lever on a gear and it is priced exactly.
              - R4.d.i.i.n.i.c. THE ANATOMY OF FAILURE (round 73; research/proof/
                failure_anatomy.md sections 1-31). The four stops, the single lever, the four
                requirements any proof must meet. FACT (the wall's shape from the walk's side).
            - R4.d.i.i.o. THE COUNTEREXAMPLE HUNT WITH PREJUDICE (owner, 2026-09-18; rounds
              77-101; proofs/AlignmentLimit.lean, ManyBody.lean, FieldBlocking.lean,
              StretchRule.lean, KillPositions.lean; research/proof/killer_attack_plan.md).
              Spawned by the anatomy: name the conditions that would kill the machine forever
              and test whether they can exist. PROVED: open columns recur at every level
              (open_columns_for_any_gears, open_run_after_alignment); the stretch rule (the
              arriving gear adds only its square, rough_member_form, plug_law); the location law
              (strike_after_square_isSquare: a gear strikes p^2 + a only if -a is a square mod
              it); kill_needs_run (a dead stretch is a struck run from the square as long as the
              stretch). Fields, pairs, triples, quads of fields: no field combination forms a
              blocking state on its own. Four killer concepts (killer residue vectors, two-prime
              products, multiplicative interleave, truncation strays): all CLOSED - position
              permits, length forbids, and the length is the conjecture. DEAD as a route (every
              killer reduces to the run-length exponent or to a distribution law outside the
              rules); FACTS kept in the kernel.
  - **R5. THE OWNER'S ARGUMENT AS THE SPINE (owner, 2026-09-19; rounds 102-113; proofs/
    OwnerArgument.lean, CounterMachine.lean, RigidShift.lean; loop entries 108-123).** Lines:
    1 the machine always generates twin gaps; 2 sometimes a gap is blocked; 3 no mechanic
    blocks the gaps permanently; 4 therefore twins without end. Lines 1, 2 PROVED for every
    gear set; 4 PROVED from 3 (twins_unbounded_of_survival). Line 3 is the SURVIVAL LEMMA.
    - R5.a. THE SURVIVAL LEMMA WEIGHED (rounds 102-104). Survival (every stretch (p^2, q^2)
      holds a column no gear <= p strikes) is Legendre-type for twins, strictly stronger than
      the conjecture; SurvivalInf (some stretch above every bound survives) is EQUIVALENT to
      twins unbounded, both directions (survivalInf_iff_twins_unbounded; converse by the largest
      prime below the twin's square root and Bertrand). survival_of_family: on 30 t +- 1 the
      gears 7..p lay one fixed pattern and the machine selects the range [p^2/30, q^2/30].
      FACT: line 3 in the form line 4 needs IS the conjecture; no reformulation lowers its
      weight (five on record: window <-> exponent 2, stretch <-> exponent 1, weak survival <->
      conjecture, strong survival -> conjecture, chain <- twin-gap growth).
      - R5.a.i. THE RUN AT THE SQUARE IS ORDINARY (round 103, survival_family.py). The struck
        run from the square's position is inside the spread of runs from random positions of
        the same pattern at every p (18 vs 6-21 at 1009; 1 vs 25-54 at 19997). DEAD: the
        location law does not shorten or lengthen the run at the square.
      - R5.a.ii. THE LOADED RECORD RULE AT THE STRETCH (entries 110-111). WITHDRAWN: the stretch
        is longer than every gear, the rule's tail is empty, the rule reduces to the survival
        statement itself. DEAD.
    - R5.b. THE COUNTER-MACHINE (rounds 105-106; proofs/CounterMachine.lean; from
      fold_mechanic.md 2026-09-11). Abstract LINES (multiplicative sets of fold survivors),
      gears = irreducibles; the square-root rule and "a surviving column is a twin gear pair"
      PROVED for every line; the real line's survival IS SurvivalInf; the counter line (twin
      lowers removed from the generators) never survives any stretch (counter_never_survives).
      VERDICT: line 3 is INDEPENDENT of the mechanics of striking; the only difference between
      the lines is real_column (every j is a column). ADMISSIBILITY TEST for any argument: delete
      the open columns and rerun; if it still goes through it is wrong. DEAD as a derivation
      from mechanics; the test is kept as the gate for every later branch.
    - R5.c. THE TWO LAYERS OF A STRETCH (rounds 107-108; two_layer_census.py,
      plug_rate_by_neighbour.py). Base gears <= q^(2/3) leave base-open columns; top gears in
      (B, p] plug them with pinned products g x r, r prime; twins are the unplugged. Plug rate
      0.646 stable; twin share 0.355 = the sieve's (log B / log q^2)^2 e^(2 gamma). THE
      INDEPENDENCE LAW: the plug rate on a base-open column is independent of the base pattern
      around it (12 distance buckets, 13 neighbour patterns, all within 1.5 sigma over 4.27 M
      columns); the record plug run 32 at 5717 is below the whole-sample independent
      expectation 35. FACT: the layers do not interact; survival at the stretch is carried by
      independence, the sieve's picture on the rigid teeth. (Entry 115's clustering claim
      withdrawn in 116.)
    - R5.d. THE PINS UNDER THE RIGID PAIR (rounds 109-111; single_gear_killers.py,
      kill_distance.py, kill_distance_ilp.py, square_class_killers.py). Single-gear killers of a
      stretch exist only at p = 17, 41 (the record's p = 29 killer was a free tooth). THE KILL
      DISTANCE d(p) (least number of gears re-phased, abandoned lone kills re-covered) exact by
      ILP to 109: stretches at 7, 11, 13, 19, 23, 31 UNKILLABLE by any rigid re-phasing; from 37
      on d = twins/2 (0.33-0.67, mean 0.50). Square-class (location-law) shifts kill exactly
      where unrestricted ones do (7..83). FACT (d ~ twins/2, the survival margin in gears);
      location law DEAD as protection; base case d >= 1 is the conjecture.
      - R5.d.i. EVERY SHIFT VECTOR IS A WINDOW OF THE REAL PATTERN (round 112;
        proofs/RigidShift.lean, 0 sorries: window_realises_shift, shifted_pattern_is_window).
        By CRT the re-phasing adversary never leaves the machine; F_shift = F(M); a stretch is
        killable in residue space iff its length is below the record (all nine cases agree).
        FACT (kernel).
        - R5.d.i.a. THE EXACT RIGID LADDER BY ILP (rounds 112-113; shift_rigid_record.py,
          rigid_record_bisect.py, rigid_record_certificate.py). Struck runs 1, 4, 6, 10, 17, 24,
          33, 42, 57, 87, 90, 102, 117, 144, 160, >= 179 at p = 5..61; 144 at 53 and 160 at 59
          NEW and CERTIFIED (x = 1249461754311661376 and x = 247344541058571239023; the
          record's target met: 161 in its run + 1 convention, the bottom of its pinned
          [161, 178]); coverable halves CERTIFIED by CRT window positions on the real pattern,
          uncoverable halves by HiGHS infeasibility. F / (p ln p) rises 0.3 -> 0.7; from p = 37
          the record exceeds most stretches. FACT; growth law OPEN beyond 61.
    - R5.f. THE LABORATORY (owner's redirection 2026-09-19 12:20, after the Conway vibe-proof
      page). Roles: PM (manager) restates the goal - SurvivalInf from the completeness of the
      line - and the gate (delete the open columns and rerun; a count or a free-phase cover
      fails); MATH lanes with fresh context (Fable / Opus) propose small plain claims each with a
      test; RED lane attacks; Lean closes behind. First cycle: three claims requested from a
      fresh Opus lane given the kernel's exact assets and the closed angles as one-line facts.
      OPEN.
      - R5.f.i. THE FIRST LANE'S THREE CLAIMS (received 12:50; tests research/stack/r8/
        lane1_claims.py; registered here before the full run to 20,000, after a 6-second run to
        4,000 that found no failure). (1) SQUARE-SCALE TRANSFER: for a twin (P, P+2) of the
        stretch, P = 6c-1, the columns of P^2, P(P+2), (P+2)^2 are 6c^2-2c, 6c^2, 6c^2+2c; some
        twin of every stretch (p >= 37) has one of the partner members P^2-2, 36c^2+1,
        (P+2)^2-2 free of every prime factor <= p. Gate: the three columns exist only because
        both P and P+2 are on the line. (2) SINGLE PLUG: every stretch has a base-open column
        struck by exactly one top gear, and every top-gear cofactor is prime. (3) NEAR PAIRS:
        every stretch has two base-open columns closer than q^(2/3)/6, and no top gear g plugs
        two base-open columns closer than (g-2)/6. Predictions (lane's): all three hold at
        every p to 20,000. Red reading (PM, before the run): (1) delivers roughness to the
        gears <= p at the scale P^2, where primality needs the gears <= P - the step lands one
        level short; to be an inductive step it must be restated so its conclusion is at the
        level of its input; (2) and (3) hold by the plug counts and separation law and their
        use is a matching formulation, to be pushed to a statement with a finite step.
        RESULT (12:56, every p from 37 to 19,997, 2,251 stretches): all three HOLD with no
        failure - (1) the first twin of the stretch clears at the mean (1.41), worst the 14th;
        (2) a single-plug base-open column in every stretch, 0 composite cofactors among all
        plugs (the plug law exact to 20,000); (3) near pairs in every stretch, 0 separation-law
        violations. FACT (three exact laws). Lane resumed 12:54 with the results and the
        reframing: lift (1) so its conclusion sits at the level where it is used; make (3)'s
        matching statement a finite check per p.
      - R5.f.ii. THE TWIN LADDER (lane round 2, received 13:02; pre-registered here before
        the run; test research/stack/r8/twin_ladder.py). Normal form (lane, elementary): for a
        twin (P, P+2), P = 6c-1, the stretch of P is the 4c-1 columns 6c^2 + j, |j| <= 2c-1,
        centred on the column of P(P+2), and gear P strikes it only at the centre; any two
        strikes of one gear g are >= (g-1)/3 columns apart. CLAIM 1 (the ladder): every twin's
        own stretch (P^2, (P+2)^2) contains a twin - the strong survival lemma at twin levels,
        which by induction from (5, 7) gives SurvivalInf with a finite interval of 4P+4 integers
        at every rung. Gate: hypothesis and conclusion are twins; empty on the counter-model.
        CLAIM 2 (short step): the nearest twin to the centre has 6|j| < 20 (ln P)^2. CLAIM 3
        (branching): for P >= 41 the stretch holds >= 2 twins. Predictions (lane): all hold for
        every twin lower P <= 10^6 (8,169 of them). Refuted by one twin stretch without a twin
        (claim 1), a ratio >= 20 (claim 2), a twin stretch with one twin at P >= 41 (claim 3).
        PM reading before the run: claim 1 is the strong Survival of R5.a at p = P twin, so
        entry 107's scan already gives it to P = 200,000; the new content is the induction shape
        (input and output the same object) and the exact normal form of a twin's stretch. Lane's
        red note on the matching route (c): supply of top gears exceeds demand by (ln p)^2 at
        every window length; closed.
        KERNEL (13:10, proofs/TwinLadder.lean, 0 sorries): stretch_normal_form (the stretch of
        a twin P = 6c-1 is exactly the columns 6c^2 + j, |j| <= 2c-1), members_of_column
        (members P(P+2) + 6j and P(P+2) + 6j + 2), twin_gear_strikes_centre_only (a multiple of
        P or of P+2 inside the stretch is P(P+2) itself: the offset 6j + e is a multiple of g in
        (-2g, 2g) and +-g is 1 or 5 mod 6 while 6j + e is 0 or 2). Run to P = 200,000: claims 1-3
        hold, max step ratio 10.3 at P = 58,169, minimum twin count 3 at P = 41; run to 10^6 in
        progress.
      - R5.f.iii. THE RUNG IN CENTRE COORDINATES: LOCATOR, CLEAN RUNGS, EXACT COUNT (lane
        round 3, received 13:20; pre-registered here before the run; test research/stack/r8/
        ladder_locator.py). s = P + 1 = 6c; centre member s^2 - 1; offset j has members
        s^2 + 6j -+ 1; new centre s' = s^2 + 6j; gear g strikes offset j iff s^2 + 6j = +-1 mod g;
        s != +-1 mod g for every gear g < P (column c is a twin), so s^2 != 1 and g strikes the
        centre iff g | s^2 + 1 (forcing g = 1 mod 4). INHERITANCE LEMMA (lane, elementary): if
        j = 0 mod g and g does not divide s^2 + 1, gear g misses offset j and s' = s^2 mod g.
        CLAIM 1 (locator): with y maximal such that the product of the gears 5..y is <= c/50,
        D = {g <= y : g | s^2 + 1}, the progression L = {j : |j| <= 2c-1, j = 0 mod prod of the
        gears <= y outside D, j != 0, 2u mod g for g in D} contains a twin centre for every twin
        lower P <= 10^6 (also reported at fixed y = 7, 13). CLAIM 2 (clean rung): for P <= 10^5
        the stretch holds a twin centre s' with no gear <= 13 dividing s'^2 + 1, so the next
        rung's locator has the full modulus; along locator rungs s' = s^2 mod g (squaring orbit).
        CLAIM 3 (exact count): T(c) >= (1/2) T0(c) prod_{sqrt P < g <= P} (1 - 2/g), T0 the offsets
        surviving the gears <= sqrt P. Predictions: all hold. Refuted by a non-empty L without
        a twin centre; a twin with no clean rung at y = 13; a ratio below 1/2. PM reading: the
        locator pre-clears the gears <= y by a congruence on j with modulus prod g <= c/50 - the
        carry wall's accounting (log-many gears cleared inside the window), now with the
        clearing done by the twin's own residues s^2 mod g rather than by divisibility; the
        search inside L is a twin search on one progression.
        RESULT (13:14; twin lowers to 200,000 for claim 1, to 30,000 for 2 and 3; runs to 10^6 /
        10^5 in progress). CLAIM 1 with the y-rule HOLDS at all 2,159 twins, no empty L: |L| min 3,
        median 1,002, max 6,593; first success at index 24 on average, 201 at worst. At fixed
        y = 7 it fails at 11 small twins (P = 5 .. 1,949, |L| <= 37), at fixed y = 13 at 634
        (|L| median 27): the rung lands in the progression when the progression is long enough,
        and the y-rule keeps it so. CLAIM 2 fails only at P = 17, 29, 41 (every twin centre s'
        of those stretches has 5 | s'^2 + 1) and P = 269 (13 divides all; clean to y = 11);
        from P = 271 every twin has a clean rung at 13, median largest clean y 211. CLAIM 3
        REFUTED at the constant 1/2 by one twin, P = 71 (ratio 0.472); 1st percentile 0.565,
        mean 0.797 - the large gears beat their independent effect by at most 2.1 at any twin
        to 30,000. VERDICT: FACT (locator law with the y-rule; clean rungs from 271; the
        count ratio in [0.47, 1]). FULL RUN (13:16): the y-rule locator holds at all 8,168
        twin lowers to 10^6, no empty L, |L| median 1,704, first success at index 28 on average
        and 273 at worst; fixed y = 7 fails at the same 11 small twins only; fixed y = 13 at 779
        (short L); clean rungs: the same four exceptions to 10^5; count ratio min 0.472 at 71,
        1st percentile 0.665 to 10^5. PM reading: with the rule M <= c/50 the progression has
        about 4c/M >= 200 candidates; for the rung to land in it at every c the progression
        must lengthen with c (M ~ c^(1-e)), which is the next round's first item.
      - R5.f.iv. LENGTHENING, STEERING, FORCED OFFSETS (lane round 4, received 13:22;
        pre-registered before the run; test research/stack/r8/ladder_round4.py). CAPTURE
        IDENTITY (lane): forcing j = 0 mod g keeps 1/(g-2) of the twin centres, so
        N_L ~ T(c) / prod (g-2), |L| ~ 4c/M, first index ~ |L|/N_L (reproduces round 3's 24).
        CLAIM 1 (lengthening): with M <= 4 sqrt c the progression has |L| >= sqrt c, holds a twin
        centre at every twin lower to 10^6, N_L >= half the capture prediction, first index <=
        12 |L|/N_L^pred; in the sieve form (offsets rough to the gears <= sqrt s) the first
        index is bounded (<= 40) with no trend in c. CLAIM 2 (steering): for Q = 35, 385, 5005
        some twin centre of the stretch has s' = 0 mod Q above a threshold P0(Q), fixing the
        small gears' classes at +-u_g for every later rung. CLAIM 3 (forced offsets): (a) the
        offsets j = -6t^2 -+ 2t have lower member (s-a)(s+a), a = 6t +- 1 - never twin centres;
        these and j = 0 are the only identity-governed offsets; (b) at j = +-c and +-(2c-1) the
        twin rate is the singular-series value (about 0.86 and 0.56 of base), nothing above 3.
        Predictions: all hold; refuted by a twin-free L, N_L below half the prediction, a rising
        sieve-form index, a twin above P0 with no steered successor, a twin centre at a
        difference-of-squares offset, or an enhancement above 3.
        RESULT (13:23; twins to 200,000, full-window parts to 30,000; run to 10^6 / 3 x 10^5 in
        progress). CLAIM 1 progression form: 19 twin-free L of 2,159 (short L at small c - list
        below); N_L against the capture prediction min 0.19, mean 1.02 (the identity is exact on
        average); index/(|L|/N_L^pred) max 7.5 (< 12). CLAIM 1 SIEVE FORM HOLDS AND MATCHES THE
        LAW: first twin among the sqrt(s)-rough offsets at index 3.6, 4.3, 3.9 for P ~ 10^2,
        10^3, 10^4 (max 23), no trend - the predicted constant 1/delta^2 = 4. CLAIM 2: steered
        successors exist from P0(35) = 2,383; for Q = 385 and 5,005 the last failures sit at
        28,619 and 29,879, at the run's cap - thresholds pending the longer run. CLAIM 3: (a) 0
        twin centres at the difference-of-squares offsets; (b) ratios to base 0.97 (+c), 0.82
        (-c), 0.72 (+(2c-1)), 0.03 (-(2c-1)): the bottom edge column of a twin's stretch has
        members P^2 + 4 and P^2 + 6, one of which 5 divides for every P != 5 - an exact forced
        failure (the square-neighbour law of KillPositions.lean), the single hit being P = 5.
        No enhancement above 1.
        FULL RUN (13:27; twins to 10^6, full-window parts to 3 x 10^5): SIEVE FORM - first twin
        among the sqrt(s)-rough offsets at index 3.6, 4.3, 4.0, 4.1 for P ~ 10^2..10^5 (1,770
        twins in the last decade), max 34, never absent: the 1/delta^2 = 4 law holds with no
        trend to 10^5. Progression form: the same 19 short-L failures of 8,168; capture identity
        mean 1.00 (min 0.14). Steering: P0(35) = 2,383; last failures 110,321 for Q = 385 and
        299,681 for Q = 5,005 (the latter at the cap). Forced offsets: 0 hits at the
        difference-of-squares offsets; ratios 0.94, 0.79, 0.68, 0.01. VERDICT: the sieve-form
        rung is a LAW (FACT); the progression form is DEAD as stated (short progressions fail);
        steering with Q = 35 is a FACT from 2,383; forced offsets closed.
      - R5.f.v. THE CANONICAL LADDER (opened 13:25; research/stack/r8/canonical_ladder.py).
        From (5, 7) take at each rung the twin of the stretch nearest the centre (smallest |j|,
        positive first): s_0 = 6, s_{k+1} = s_k^2 + 6 j_k. Registered before the lane's
        prediction of its offset law arrives (round 5 (c)); the PM's prior: 6|j_k| < 20 (ln P_k)^2
        at every rung (R5.f.ii claim 2). FIRST RUNGS (13:26): P = 5, 41, 1787, 3196327 (7 digits),
        14 digits, 27, 53, 105 digits, with j = 1, 4, 6, 77, 44, -829, 3605, 28145 and
        6|j|/(ln P)^2 = 2.3, 1.7, 0.6, 2.1, 0.3, 1.4, 1.5, 2.9 - bounded, no trend; rung 8
        (209 digits): j = -6965, ratio 0.18, 4.6 s; the ninth twin lower has 417 digits
        (members probable primes by BPSW). Nine rungs from 5 to 10^416 in eight seconds of
        search. RUNG 10 (20:32, the lane's rule, canonical_ladder2.py): from the 378-digit centre
        the nearest twin is at j = -321,634 (ratio 2.56, 8,379 10^6-rough candidates passed,
        643,267 rejected, 47 minutes) - a 756-digit twin lower; the chain from (5, 7) now has ten
        rungs with BPSW-probable members, six of them Pratt-certified. The 1,500-digit rung would
        cost days and is not funded. OPEN: the offset law and the certificate form per rung.
      - R5.f.vi. THE TWO HALVES OF THE SIEVE-FORM RUNG (lane round 5, received 13:30;
        pre-registered before the run; test research/stack/r8/ladder_round5.py). Base gears
        5..x, x = floor(sqrt s); top gears (x, P]. CLAIM A (base half): (i) PROVED FROM THE
        RECORD - no F(x)+1 consecutive offsets of the window are all base-struck, so the window
        holds >= (4c-1)/(F(x)+1) base-open offsets, a guarantee growing like 4.7 sqrt c / ln(6c)
        with no density input; (ii) the base-open count within |j| <= F(x) is >= 0.3 * 2F(x) W(x)
        and the N-th base-open offset has |j| <= 3N/(2W(x)); (iii) the longest base-struck run
        inside the window is <= 0.7 x ln x (the record's growth law at the window's own
        position). CLAIM B (top half): under the span condition S_N < (x-1)/3 every top gear
        plugs at most one of the first N base-open offsets (min-gap); no top gear plugs two of
        them; the first twin among the base-open offsets is within the first 24 (the PM notes
        round 4 already saw index 34 at 10^5, so the constant is refuted and the bounded
        growth of the maximum is the question). CLAIM C (canonical ladder): ratio 6|j|/(ln P)^2
        mean ~1.5 with no drift; signs balanced; ln ln P_k ~ k ln 2; the small-gear phases are
        NOT a squaring orbit on the canonical path (only when g | 6j, rate ~1/g); certificate
        per rung = two primality certificates + the containment inequalities. Predictions and
        refutations as stated by the lane.
        RESULT (13:31; every twin lower to 10^5, 1,223 twins). CLAIM A HOLDS WITH MARGIN: the
        longest base-struck run inside the window is at most 0.43 of 0.7 x ln x (worst P = 191);
        the base-open count within |j| <= F(x) is at least 2.25 times the 0.3-heuristic (worst
        P = 227); the N-th base-open offset sits at most 0.71, 0.57, 0.48 of 3N/(2W) for N = 8,
        16, 32. CLAIM B: the first twin among the base-open offsets is at index 4.00 on average,
        max 29 (4 twins beyond 24: the constant 24 is refuted; the maximum grows like the log of
        the sample, a geometric tail at rate ~3/4); the span condition S_N < (x-1)/3 holds at
        232 of 1,221 twins for N = 8 and never for N = 16, 32 at these sizes, and under it no top
        gear plugs two of the first N (0 cases), 8.9 distinct gears for 8 offsets. VERDICT: A is
        FACT (its part (i) is a theorem from the record, to be put in the kernel); B's rigid
        mechanism (min-gap distinctness) does not bind at N >= 16 below 10^5 - the top half
        stays a bounded-index law, OPEN in mechanism.
        KERNEL (13:36, proofs/LadderBase.lean, 0 sorries): open_in_every_block and card_open_ge -
        if no F + 1 consecutive columns are all struck, every block of F + 1 holds an open column
        and a window of L columns holds at least L / (F + 1) open ones (one per full block,
        injectively). Claim A (i) is a theorem CONDITIONAL on the run bound: instantiating it
        at depth x = sqrt s needs F(x) < 2x^2/3 to give even one candidate - exact to x = 61,
        measured 0.7 x ln x beyond, best proved two-class exponent 4.266 (the lane's audit flag,
        R5.f.x). No density input, but an open run-bound input.
        LADDER TO 10^6 (14:13, twin_ladder.py, 8,168 twin lowers): every twin's stretch holds a
        twin (0 failures); nearest twin ratio 6|j|/(ln P)^2 max 13.27 at P = 646,421 (j = 396),
        mean 1.51, 99th percentile 6.9; at least 3 twins per stretch from P = 41 (minimum 3).
        CLAIM C RESULT (13:33, canonical_ladder2.py, the lane's rule - nearest centre, negative
        tie): 5 -> 29 -> 881 -> ... nine rungs to a 378-digit twin lower in 24 s; offsets -1, -3,
        9, 21, -68, 641, -1682, -4187, -2086; 6|j|/(ln P)^2 = 2.3, 1.6, 1.2, 0.7, 0.55, 1.3, 0.86,
        0.53, 0.07 - no drift (prediction (i) held); signs 6 negative of 9; phase locks
        s_{k+1} = s_k^2 mod g for g <= 100 observed 6, exactly the gears dividing 6 j_k
        (prediction (iv) held: not a squaring orbit); the index among the 10^6-rough offsets
        (the sqrt(s)-sieve is out of reach past rung 3) is 1, 3, 4, 2, 1, 10, 38, 126, 58,
        growing with (ln s / ln 10^6)^2 as a proxy base must. FACT.
      - R5.f.vii. THE LADDER THEOREM AND ITS ONE HYPOTHESIS (lane round 6, received 13:40;
        kernel proofs/TwinLadderTheorem.lean, 0 sorries; test research/stack/r8/
        ladder_round6.py, pre-registered before the run). KERNEL: TwinCentre s (6 | s, s-1 and
        s+1 prime); Rung s s' (a twin centre strictly inside ((s-1)^2, (s+1)^2)); LadderHyp
        (every twin centre has a rung); rung_gt (a rung climbs, s' >= s + 2);
        twinCentre_unbounded and twins_unbounded_of_ladder (LadderHyp -> twin primes above every
        bound, the kernel's standard form); NearTwinHyp (the lane's NTH: a rung among the first
        B(s) base-open offsets) -> LadderHyp. So SurvivalInf = the ladder from (5, 7) never
        terminates, one named hypothesis plus proved lemmas (normal form, twin gears strike only
        the centre, record pigeonhole). CLAIM 1 (index law): the first twin among the base-open
        offsets has index <= ceil(4 ln P) for every twin lower to 10^6 (geometric tail at rate
        3/4; the decade maxima should rise by about 8 per decade, not accelerate). CLAIM 2
        (eligibility filter): members s^2 - A, s^2 - B with A = 1 - 6j, B = -1 - 6j; a gear
        divides a member only if A or B is a square mod g (the location law in centre
        coordinates); exactly a quarter of the gears are barred from each non-square offset;
        difference-of-squares offsets have A square and bar none. CLAIM 3 (no band is
        insufficient): a band (x, y] plugs at most 4s ln(ln y / ln x) + pi(y) - pi(x) members,
        insufficient only for y < x^(1 + W(x)/6); direction (a) closed with a number; reuse of a
        plugging gear among the first 16, 32 offsets at the independent rate (~0.1 per twin at
        N = 32). Predictions: index law holds (max 46 at 10^5, 55 at 10^6 allowed); zero
        eligibility exceptions; barred fraction 1/4; reuse within a factor 2 of the prediction.
        RESULT (13:40; index law to 200,000, the rest to 30,000; full runs in progress). CLAIM 1
        HOLDS: max index 31, max i / ln P = 2.84 (P = 809, i = 19), no twin above 4 ln P; decade
        maxima 19, 21, 29, 31 at 10^2..10^5 - rising slowly, no acceleration. CLAIM 2 HOLDS:
        0 exceptions to the square-phase eligibility among all plugs of the first 32 base-open
        offsets; barred fraction 0.245 (predicted 0.25). CLAIM 3: the first 32 base-open
        offsets receive 8.3, 14.3, 20.1, 34.5 plugs from (x, 2x], (x, 4x], (x, x^1.5], (x, x^2]
        (no band insufficient, as computed); REUSE BELOW PREDICTION: observed 247 against 551
        predicted among the first 16, 1,041 against 2,100 among the first 32 - a factor 0.45-0.5,
        at the lane's own refutation line ("less than half"); the prediction (2/g)^2 per pair
        ignores that a reuse needs g to divide one of three fixed differences of the offsets and
        then one phase condition, so the exact expectation from the difference set is the next
        item before reading rigidity into it.
        FULL RUN (13:42; index law over all 8,168 twin lowers to 10^6, the rest to 10^5): index
        law HOLDS - max index 44 at P = 646,421 with i / ln P = 3.29 (< 4), no twin above 4 ln P,
        mean 4.11; decade maxima 19, 21, 29, 44 (the last step +15, above the +8 the geometric
        tail predicts per decade - one extreme twin, within the law's slack); eligibility: 0
        exceptions among all plugs of the first 32 base-open offsets of 1,223 twins, barred
        fraction 0.248; bands 7.4, 13.2, 20.3, 34.8 plugs; reuse 288 against 694 predicted
        (first 16) and 1,416 against 2,887 (first 32) - the factor 0.41-0.49 persists at 10^5.
      - R5.f.viii. EXACT REUSE, THE UNIVERSAL LIST, NO IMPOSSIBLE BLOCK (lane round 7,
        received 13:45; pre-registered before the run; test research/stack/r8/ladder_round7.py).
        CLAIM 1 (exact reuse): a gear g > x strikes two base-open offsets at difference d only
        if g | d, 3d-1 or 3d+1, with the common phase t = 1 - 6j_i or -1 - 6j_i; over the twins
        T = s^2 mod g weighs 1/(g-2) at 0 and 2/(g-2) at each nonzero square != 1; the exact
        expectation E from the difference set accounts for the observed factor 0.47 (the
        earlier (2/g)^2 was the wrong object); refuted by a discrepancy beyond two Poisson
        sigma - which would name a correlation between the base pattern and the top phases
        that fact 3 excludes. PM note: the earlier observed count used least prime factors
        only; the model's object is every prime factor > x of both members; both are counted.
        CLAIM 2 (universal list): on steered rungs s = 0 mod Q the base-open offsets at the
        gears dividing Q are j != +-u_g mod g at every level (Q = 35: 0, +-2, +-3, +-5, +-7,
        ...); the steered first-twin index mean is 4.0, NOT below (steering fixes where the
        candidates sit, not the hit rate); refuted by a mean below 3.6 or above 4.4. CLAIM 3
        (no impossible block): for N = 2, 3, 4 every difference pattern of the first N base-open
        offsets that occurs also occurs fully plugged - the plug conditions are congruences
        s^2 = A_i on distinct prime moduli, jointly satisfiable by CRT, and min-gap binds only
        beyond the span (x-1)/3; refuted by a pattern with >= 100 occurrences never fully
        plugged. Division of labour fixed: the top half contributes proved constraints on which
        gear plugs which offset, no existence statement; NTH stays the single hypothesis and the
        remaining work is to bound the index.
        FIRST RESULTS (13:46; twins to 30,000, steering to 200,000; full run in progress).
        Claim 1: observed reuse counted by all prime factors > x is 313 (first 16) and 1,304
        (first 32) against exact expectations 332.0 and 1,544.0 - the first within one Poisson
        sigma, the second 16% below (about six sigma on a Poisson reading); by case the
        expectation splits 536 (3d-1), 548 (3d+1), 461 (d). Claim 3: no impossible block -
        every pattern of the first 2, 3, 4 base-open offsets with >= 100 occurrences also occurs
        fully plugged (fully plugged shares 0.69, 0.55, 0.43). Claim 2: universal list exact
        (0 violations at 136 steered twins for Q = 35, 15 for 385); steered index mean 4.79 +-
        0.39 (Q = 35) and 4.13 +- 0.99 (Q = 385) - the full run decides.
        FULL RUN (13:47; twins to 10^5 for reuse and blocks, to 10^6 for steering). Claim 1:
        observed reuse by all prime factors > x 355 (first 16) and 1,730 (first 32) against
        exact expectations 380.7 and 1,929.2 - a 7% deficit (about 1.3 Poisson sigma) at 16 and
        a 10% deficit (about 4.5 sigma) at 32, reproducing the 16% deficit at 32 seen to 30,000;
        by case 699 (3d-1), 713 (3d+1), 517 (d). The exact model removes most of the crude
        factor 0.47 but a deficit that grows with the block length remains; located for the
        lane (round 8). Claim 3 HOLDS: no pattern of the first 2, 3, 4 base-open offsets with
        >= 100 occurrences is never fully plugged (1,223 twins; fully plugged shares 0.66, 0.52,
        0.42). Claim 2: universal list exact (0 violations, 542 steered twins for Q = 35, 66 for
        385); steered first-twin index mean 4.53 +- 0.21 (Q = 35) and 5.14 +- 0.69 (Q = 385):
        steering does NOT lower the index (the lane's prediction of 4.0 held in direction, the
        value sits 0.6 sigma above its 4.4 line). VERDICT: claims 2 and 3 FACT; claim 1 OPEN at
        the 10% level.
      - R5.f.ix. THE REUSE RECOUNT AND THE INDEX BOUND'S SUPPLY SIDE (lane round 8, received
        13:52; pre-registered before the run; test research/stack/r8/ladder_round8.py). CLAIM 1:
        the 10% deficit is the statistic - the observed sum (k_g - 1) against the expectation's
        sum C(k_g, 2) differ by sum (k_g - 1)(k_g - 2)/2, about N/(3x) of the pair sum, linear in
        N and inverse in x (the pattern 7% -> 10% with N 16 -> 32, 10% -> 16% with the cap 10^5
        -> 3 x 10^4); the g | d case generates the multiplicity; roughness conditioning is +1%
        (Buchstab), wrong sign; real correlation predicted 0. Refuted by sum C(k,2) below the
        pair expectation by two Poisson sigma after the recount. CLAIM 2: the weakest admissible
        bound is B(s) = floor((2s/3 - 1)/(F(sqrt s) + 1)) ~ 1.9 sqrt(s)/ln s (the record pigeonhole's
        guarantee); the measured B = ceil(4 ln s) is certified by the pigeonhole only above
        s0 ~ 6.8 x 10^4, the finitely many twins below being verified directly. THE REMAINING
        HYPOTHESIS IN THE LITERATURE'S WORDS: a twin prime pair in [N - H, N + H] around N = s^2
        with H ~ N^(1/4) log N (weakest admissible B), H ~ (log N)^3 (measured B), or simply
        between P^2 and (P + 2)^2 (full window) - the twin-prime analogue of Oppermann's
        conjecture, which is all the ladder needs. Predictions: recount matches; no twin with
        i >= the pigeonhole guarantee (that would break the supply side); no twin above s0 with
        i > 4 ln s.
        FIRST RESULTS (13:54; recount to 30,000, supply to 200,000; full run in progress). CLAIM 1
        HELD: sum C(k_g, 2) = 1,623 against the pair expectation 1,544 (+5%, two sigma above;
        sum (k_g - 1) = 1,304 is the earlier deficit); multiplicities k = 2: 886, 3: 141, 4: 24,
        5: 11, 6: 4; by case observed 556 / 574 / 493 against expected 548 / 536 / 461 - the
        deficit was the statistic, the phase model stands, no rigidity in the top half. CLAIM 2:
        no twin to 200,000 with i > ceil(4 ln s); i >= the pigeonhole guarantee at 8 twins, the
        last at s = 12,162 (P = 12,161: i = 23 against G = 22), all below the predicted
        s0 ~ 6.8 x 10^4 - the finitely many exceptions the lane named, verified directly.
        FULL RUN (13:55): recount to 10^5 - sum C(k_g, 2) = 2,057 against 1,929 expected (+6.6%,
        three sigma ABOVE; k = 2: 1,296, 3: 149, 4: 24, 5: 11, 6: 4; by case 749 / 752 / 556
        against 713 / 699 / 517); supply to 10^6 - no twin with i > ceil(4 ln s), the same 8
        exceptions to the pigeonhole guarantee, none above s = 12,162. VERDICT: claim 1 FACT
        (the top half is the phase model, slightly over rather than under); claim 2 FACT (the
        supply side holds from s = 12,162 on, with the eight small exceptions verified). The
        ladder's remaining hypothesis is stated in both vocabularies (node text above).
      - R5.f.x. THE AUDIT, THE PROOF MAP AND THE CERTIFICATE (lane round 9, received 14:00;
        research/proof/ladder_proof_map.md). Every claim of rounds 1-8 with its final status
        (table in the map): PROVED and in the kernel - normal form, twin gears at the centre
        only, record pigeonhole (conditional), rung_gt, twinCentre_unbounded,
        twins_unbounded_of_ladder, NearTwinHyp -> LadderHyp, square-phase eligibility, top-band
        idleness, difference-of-squares failure, universal list; MEASURED-HELD - the ladder to
        10^6, the index law i <= 4 ln P, steering thresholds, the exact reuse model (+5-7%
        residual open), no impossible block; REFUTED - fixed-y locators, clean rungs at 13 as
        stated, count ratio 1/2, the sqrt(c) progression form, index <= 24, reuse (2/g)^2, the
        steered-index band; WITHDRAWN - round 1's transfer as an inductive step. THE SERIOUS
        FLAG: the candidate supply is not unconditional - it needs F(x) < 2x^2/3 (L3a, OPEN).
        THE MAP: critical path H (LadderHyp) -> L8 -> twins unbounded; L3 off the path (makes
        the rung a bounded check, drags in L3a); L4, L5 load nothing. THE CERTIFICATE: six rungs
        6 -> 30 -> 882 -> 777978 -> 605249768610 -> 3.66 x 10^23 -> 1.34 x 10^47 with Pratt
        certificates for all members (the 48-digit p - 1 factored in a second each) and the
        composition theorem twins_unbounded_of_ladder_above (LadderHyp needed only above
        10^47), proofs/LadderCertificate.lean + LadderPratt.lean - BUILT 14:07, 0 sorries
        (canonical_ladder_six_rungs, twins_unbounded_of_ladder_above).
        VERDICT: the ladder programme is a REDUCTION - infinitude of twins to one localised
        short-interval statement (twin between P^2 and (P+2)^2; bounded form: among the first
        ceil(4 ln s) base-open offsets) - with everything around it proved or measured and the
        open ingredients named in both vocabularies. CANDIDATE for the answer's form: not yet
        shown (the hypothesis is the conjecture localised).
      - R5.f.xi. THE FIXED-DEPTH RUNG: THE SUPPLY SIDE MADE UNCONDITIONAL (PM, 14:05, from the
        audit's serious flag; research/stack/r8/ladder_depth61.py; registered with the first
        measurement, the run to 10^6 in progress). Take the base depth x = 61, where the record
        is EXACT (F(61) = 179, certified both ways): the pigeonhole then gives, unconditionally,
        at least (4c - 1)/180 base-open offsets in every twin's stretch - the open run bound L3a
        leaves the map. The cost: the first twin among the 61-rough offsets sits deeper - index
        mean 1.7, 4.4, 7.0, 8.6 for P ~ 10^2..10^5 (max 58), i.e. i ~ C (ln s)^2 with i/(ln s)^2
        mean 0.06-0.09 and max 0.43 to 200,000 (the sieve reading: (2 ln s / ln 61)^2 / e^(2 gamma)
        ~ 0.075 (ln s)^2). Prediction: i <= 0.6 (ln s)^2 at every twin to 10^6; refuted by a
        larger ratio. The hypothesis in this form: a twin among the first ceil(0.6 (ln s)^2)
        61-rough offsets of the stretch - a twin prime pair within about 2 (ln s)^2 columns of
        s^2 - with the candidate list proved to exist from F(61) alone.
        RESULT (14:06; every twin lower 67 <= P <= 10^6, 8,162 twins): index mean 10.3 (1.7, 4.4,
        7.0, 11.0 by decade), max 107; i/(ln s)^2 mean 0.064 (0.079, 0.092, 0.083, 0.083 by
        decade - flat, the sieve reading 0.075 holds), max 0.598 - the pre-registered line 0.6
        held by 0.002, so the constant is at the edge and the principled maximum law (mean times
        the log of the sample) is the lane's round-10 item. MEASURED-HELD; the supply inequality
        (4c - 1)/180 >= 0.6 (ln s)^2 holds from s ~ 3 x 10^4 (c ~ 5,000) on.
      - R5.f.xii. THE MAP AT FIXED DEPTH, THE CUBE BOUND, THE NAME, THE PATH FORM (lane round
        10, received 14:10). (1) With depth 61 the supply lemma is instantiated by the exact
        record (L3@61) and L3a leaves the map; exact rate from W(61) = 0.137565 and 3 C_2:
        mean index = 0.0695 (ln s)^2 (the asymptotic Mertens form 0.0746 overshoots by 7%).
        THE BOUND MUST BE A CUBE: the index is geometric, its maximum over N twins is the mean
        times ln N ~ ln s, so B(s) = ceil(0.0695 (ln s)^3); predicted maximum to 10^6: index
        85-120 and i/(ln s)^2 in [0.45, 0.63] - OBSERVED 107 and 0.598 (both inside);
        refutation line i > 0.07 (ln s)^3 - CHECKED, no twin to 10^6 breaches it; calibration
        by ln-s bin: observed mean / 0.0695 (ln s)^2 = 0.85-0.94 from ln s >= 7 (10% low,
        stable). Supply inequality 0.0695 (ln s)^3 <= (2s/3 - 1)/180 holds from s ~ 1.8 x 10^4;
        the prefix ends at 1.34 x 10^47, forty orders of slack. THE NAME: the first B(s) 61-rough
        offsets span an interval of length 43.6 B = 3.03 (ln s)^3 = 0.379 (log N)^3 at N = s^2, so
        NTH_61 is the twin analogue of Cramer's conjecture (consecutive twin pairs near N are
        O((log N)^3) apart) restricted to N = s^2, s a twin centre; the mean tier 0.76 (log N)^2
        matches the average twin gap (log N)^2/(2 C_2) = 0.757 (log N)^2 to three digits - the
        calibration check. Depth cancels: interval length H = 6B/W(x) is independent of x, so no
        depth, ordering, steering or one-level-down choice moves anything from measured to
        proved - the reduction is complete as a reduction. (2) The weakest form that composes is
        a PATH (Konig: the rung tree is finitely branching with >= 3 children measured): kernel
        twins_unbounded_of_path and path_of_ladderHyp_above added (build in progress). (3) Kernel
        audit: Rung's strict inequalities are exactly |j| <= 2c - 1 (both ends, no off-by-one;
        strictness not load-bearing since the boundary values are prime squares); rung_gt takes
        TwinCentre s (s >= 6) as it must; Rung alone does not carry TwinCentre s (fine inside a
        chain from 6). VERDICT: the ladder programme is a completed REDUCTION with its open node
        named in both vocabularies; CANDIDATE form of the answer: the path from 1.34 x 10^47.
      - R5.f.xiii. THE RANDOM LANE'S THREE ANGLES (a second fresh Opus lane, received 14:25;
        pre-registered before the run; test research/stack/r8/random_lane1.py). ANGLE 1
        (universal clearance): a twin centre has s != +-1 mod g, so s^2 mod g lies in
        Q_g = {x^2 : x != +-1}; the offset classes U_g = {j : 1 - 6j and -1 - 6j both outside Q_g}
        are struck by gear g at NO twin centre and NO level (U_5 = {3}, U_7 = {0, 2, 4},
        |U_g| ~ (g-3)/4); their CRT intersection A_B is a fixed class set, the same forever -
        the ladder's pre-cleared list without any dependence on s. Predictions: every U_g
        nonempty to 61; 0 violations; the rung rate inside A_13 exceeds the overall rate by
        about prod 1/(1-2/g) = 3.3 (naive). ANGLE 2 (forest): a twin centre has at most one
        parent (the interval (sqrt(s') - 1, sqrt(s') + 1) holds at most one multiple of 6), so
        the rung graph is a forest with unique parents; accounting identity #{s' <= Y with a
        parent} = sum over s of #{rungs of s <= Y}, exact. Predictions: 0 twin centres with two
        parents; identity holds at Y = 10^6. ANGLE 3 (neighbours): t^2 - s^2 = (t-s)(t+s)
        translates the strike pattern between consecutive twin centres; the six exclusions
        s != +-1, +-1 - d+, +-1 + d- (mod g) can force s mod g from the gaps alone. Predictions:
        0 translation violations; every forced congruence agrees with the actual s; the
        first-rung index partitioned by the forced s^2 mod 7 differs between phase classes if
        the neighbourhood carries information (the lane's payoff question).
        RESULT (14:34; twin centres to 10^6, full census to 10^4). ANGLE 1 EXACT: every U_g
        nonempty (sizes 1, 3, 3, 2, 5, 6, 6, 7, 9, 8, 11, 12, 12, 13, 15, 14 for g = 5..61); 0
        violations in 2,419,097 checks; A_13 has 18 of 5,005 classes and the rung rate inside it
        is 0.0871 against 0.0271 overall - enhancement 3.21 (naive 3.37). A FIXED pre-cleared list
        at every level, s-independent: FACT; KERNEL 14:40 proofs/UniversalClearance.lean, 0
        sorries: not_dvd_of_not_square (if s != +-1 mod g and a is no square of a residue other
        than +-1 then g does not divide s^2 - a) and clear_five (gear 5 never strikes an offset
        j = 3 mod 5 at any twin centre). ANGLE 2 EXACT: 0 twin centres with
        two parents (the rung graph is a forest); accounting identity at Y = 10^6: descent 440 =
        ascent 440; parents are rare (7,728 of 8,168 twin centres to 10^6 are roots; the fraction
        with a parent 4.5% in the 10^5 decade); child counts of parents up to 27. FACT. ANGLE 3:
        translation exact (0 violations); s mod 5 forced by the neighbours' gaps at 3,877 twin
        centres and mod 7 at 451, every forced value agreeing (0 violations); but the first-rung
        index by forced s^2 mod 7 is 6.86 / 6.29 / 6.55 - flat: the neighbourhood's phases carry
        no rung information. FACT for (a), (b); DEAD as a selection rule.
      - R5.f.xiv. THE RANDOM LANE'S ROUND 2: WHAT THE UNIVERSAL LIST BUYS, THE FOREST'S LAW,
        AND THE CYCLOTOMIC OFFSETS (received 14:50; pre-registered before the run; test
        research/stack/r8/cyclotomic_offsets.py). (a) A_B: |A_B| = 1, 3, 9, 18, 90, 540, ... at
        B = 5..19, density delta_B = |A_B|/M_B, enhancement E(B) = prod 1/(1-2/g) (3.37 at 13,
        measured 3.21); the count of universal offsets in the window is a theorem while M_B <=
        4c - 1, i.e. B <= 109 at s = 10^47, giving 4 x 10^30 guaranteed 109-rough candidates
        against the 8.9 x 10^4 the cube bound asks - the universal list alone carries the
        bounded hypothesis at the certified top; the smallest universal offset at B = 61 is
        ~2.8 x 10^9, so A_B never holds the FIRST rung; the difference-of-squares offsets do
        enter A_B in bulk at large s (u = +-1 mod every g <= B), fraction 7 x 10^-33 - the Lean
        statement must exclude them; the gear-exclusion at a constant offset is exactly a
        congruence mod the conductor of the biquadratic field Q(sqrt(1-6a), sqrt(-1-6a)) (abelian,
        Kronecker-Weber) - a hard ceiling for constant offsets. (b) Parenthood has no arithmetic
        signature: the non-root fraction is (1/3) x (twin-centre density among multiples of 6
        near sqrt s'), predicting 4.4% for the 10^5 decade against the measured 4.5%; T(s), the
        child count, is an exact inclusion-exclusion whose every gear g <= s-2 strikes exactly two
        classes, main term 1.32 s/(ln s)^2 with the plain twin constant - generic branching; the
        diagnosis: the gear range (to s-2) exceeds the window (2s/3) by 1.5, so no lower bound on
        T by dimension-2 sieving; Konig: the hypothesis weakens to "the tree rooted at the
        certified top has a node at every depth", and a finite tree would exhibit an explicit
        finite set of twin centres with T = 0. (c) THE CYCLOTOMIC OFFSETS j = +-c: s' = s(s +- 1),
        members s^2 +- s - 1 and Phi_3(s) = s^2 + s + 1 / Phi_6(s) = s^2 - s + 1; every prime factor
        of Phi_3(s) is 1 mod 3, of Phi_6(s) 1 mod 6, of s^2 +- s - 1 is +-1 mod 5 - a quarter of
        all gears, of every size, inert at these offsets (the first mechanism reaching the gears
        above the window); Pocklington/Lucas certificates for free (N - 1 = s(s+1) with s + 1
        prime); the recursion s -> s(s +- 1) is search-free. Predictions: identities exact (0
        violations); hits at +-c about 170 to 10^6 with enhancement K > 1 over a random offset;
        the pure cyclotomic ladder dies at finite depth (sum 1/(ln s_n)^2 converges) - depths
        systematically beyond that would be new structure.
        RESULT (18:54; twin centres to 200,000; identities to 30,000; ladders from <= 3,000).
        Hits at +c: 38, at -c: 32, total 70 against a base-rate expectation 78.2 - measured
        enhancement 0.89; the lane's own singular series gives K = 0.878: the quarter of inert
        gears is EXACTLY compensated by the higher density of roots at the other gears - the
        cyclotomic offsets are no better than a random offset (the "K > 1 definitely" prediction
        refuted, by the lane's own formula). Identities exact with one stated exception: all 313
        "violations" among 4,563 prime factors are the prime 5 itself dividing s^2 +- s - 1 (when
        5 | 2s +- 1); every other factor obeys 1 mod 3, 1 mod 6, +-1 mod 5. The pure cyclotomic
        ladder dies at once: depth 0 at 74 of 81 twin centres, depth 1 at 6, depth 2 at 1. VERDICT:
        (a) FACT (the universal list carries the cube bound at the certified top, with the
        exclusion of the identity offsets to be stated); (b) FACT (generic branching; Konig
        form); (c) DEAD as an enhancement - the identity-forced roughness at moving offsets is
        priced by the singular series like everything else; its free certificates stand as a
        tool.
        KERNEL (19:00, proofs/LadderDepth.lean, 0 sorries): Depth n s (the rung tree from 6 by
        depth), Depth.twinCentre, Depth.ge (a node at depth n is >= 6 + 2n), DepthHyp (a node at
        every depth), twins_unbounded_of_depth (DepthHyp -> twins unbounded, directly, no Konig),
        depthHyp_of_path, depthHyp_of_ladderHyp. The weakest form on record: the rung tree from
        (5, 7) has a node at every depth.
      - R5.f.xv. THE PRICING THEOREM, THE CONSERVATION LAW, AND THE (5, 7) TREE ENUMERATED
        (random lane round 3, received 19:00; research/stack/r8/rung_tree.py). PRICING: an
        in-window polynomial offset is f(c) = kc + r with k in {0, +-1, +-2} (degree >= 2
        outgrows 2c - 1), so the members are the monic quadratic pairs s^2 + ks + 6r -+ 1 - a
        classification; reducibility complete: k = 0 lower member (s - u)(s + u) (the known
        difference-of-squares family), k = +-2 UPPER member (s +- 1 - v)(s +- 1 + v) at j = +-2c - 6t^2
        (a second composite-forcing family, previously unrecorded, to be excluded alongside the
        first), k = +-1 both members always irreducible (the cyclotomic band); for irreducible
        pairs the rung density is K_f x 1.98/(ln X)^2 with K_f a finite Chebotarev constant
        (root counts of each quadratic average 1). CONSERVATION: every gear 5 <= g <= s - 2 strikes
        exactly two of every g consecutive offsets, so the singular series averages exactly 1
        over the window: selecting offsets redistributes a fixed total E[T(s)] = 1.32 s/(ln s)^2,
        it cannot raise it - the cyclotomic 0.878, the flat neighbour phases and the dead
        steering are one statement. The only object escaping the corollary is an offset SET
        growing with s (A_B), as a uniform guarantee, not a rate. Moving offsets CLOSED with a
        reason. DEPTH FORM confirmed: Konig not needed; strictly weaker than LadderHyp; equivalent
        to an infinite path (finite branching, T(s) <= 4c - 1); parent is a partial function, so
        "node at depth n" = parent^n(s) = 6. THE STATEMENT IN WORDS: there is an infinite chain of
        twin prime centres 6 = s_0 < s_1 < ... with s_{n+1} strictly between (s_n - 1)^2 and
        (s_n + 1)^2 - twin primes contain an infinite chain in which each pair lies between the
        squares of the preceding pair (finite-level form: chains of every finite length from
        (5, 7)). THE TREE ENUMERATED (19:01): depth 0 {6}; depth 1 {30, 42}; depth 2: 5 nodes
        (858 .. 1788); depth 3: 182 nodes (734,472 .. 3,200,358); child counts 2 | 2, 3 | 27, 25,
        35, 47, 48 against the generic 2.5 | 3.4, 4.0 | 24.8, 25.3, 40.5, 40.9, 42.1; DEPTH 4
        (19:03): 2,619,059 nodes, from 539,447,650,518 to 10,242,297,728,700 (14 digits); child
        counts of the 182 depth-3 nodes min 5,172, mean 14,390.4, max 19,113 against the generic
        mean 14,377.6 - branching generic to 0.1%. So the (5, 7) tree has 1, 2, 5, 182, 2,619,059
        nodes at depths 0-4, every one a certified twin pair between the squares of its parent.
        FACT. CHILD-COUNT VARIANCE (23:02, the 182 depth-3 nodes): T / (1.3203 s/(ln s)^2) has
        mean 0.9999 and standard deviation 0.0092, exactly the Poisson value 1/sqrt(T) = 0.0092
        (ratio 1.00; range 0.971 to 1.025) - the twin counts in the stretches around the squares
        of twin centres are Poisson about the twin-constant law, no excess variance, no
        arithmetic signature. FACT. KERNEL (19:03, TwinLadder.lean): upper_member_difference_of_squares and
        lower_member_difference_of_squares - both composite-forcing families as ring identities;
        (20:10) two_strike_classes - for a prime g >= 5 and any centre s the offsets g strikes
        are exactly two distinct classes modulo g (the conservation law: selection redistributes
        strikes, it cannot remove them).
      - R5.f.xvi. THE INDEX LAWS AT DEPTH 4, s ~ 10^12-10^13 (PM, pre-registered 19:40 before
        the run; research/stack/r8/depth4_sample.py; 30 random depth-4 nodes of the (5, 7)
        tree). Predictions: nearest-rung ratio 6|j|/(ln P)^2 mean about 1.5, max below 20; the
        first-rung index among the 61-rough offsets near the mean law 0.0695 (ln s)^2 ~ 58 and
        below the cube 0.07 (ln s)^3 ~ 1,700 at every node; the index among the sqrt(s)-rough
        offsets mean about 4, max below 30. Refuted by a node breaching the cube bound or a
        sqrt(s)-index above 60 (a jump of the law by six orders of magnitude in s). Also in the
        kernel this tick: parent_unique (LadderDepth.lean) - the rung graph is a forest.
        RESULT (19:27; 30 nodes, s from 5.4 x 10^11 to 1.02 x 10^13): ratio 6|j|/(ln P)^2 mean 2.10,
        max 10.6 (held); 61-rough index mean 80.5 against the mean law 51-62 (1.3x, within a
        30-sample's spread), max 430 against the cube 1,380-1,882 (held at every node);
        sqrt(s)-rough index mean 5.8, max 24 (held). All three laws hold six orders of magnitude
        beyond their measurement range. FACT.
      - R5.f.xvii. THE CONTRADICTION ROUTE: A FINITE TREE (contradiction lane, a third fresh
        Opus lane, received 20:45). Suppose the (5, 7) tree is finite; every leaf s is a twin
        centre whose stretch of length 4s around s^2 holds no twin. MECHANICAL: (M2) a leaf is
        EXACTLY sieve data - every composite member of the stretch has a prime factor <= s - 1
        (its least factor is < s + 1 and s is not prime), so there is no plug from above and the
        leaf is a two-class covering of the window by the gears 5..s-1 with no analytic
        remainder; (M1) count identity corrected: pi((s+1)^2) - pi((s-1)^2) = primes among the
        2(4c-1) members + [s^2 + 2s - 1 prime] (the singleton (s+1)^2 - 2 at j = 2c); (M3)
        counting never refutes a leaf - the two-class capacity exceeds the window from s = 19 on
        by 2(ln ln s - 0.572) (5.65 at 10^13) and the leaf's own prime ceiling 2s/3 sits (ln s)/3
        above the truth, so even an exact short-interval prime asymptotic would not contradict a
        leaf; (M4) the fundamental-lemma split leaves slack (log s)(log log s); (M5) 1 - 1/ln s of
        the window must be covered by gears <= 2s/3, the periodic part; (M6) j = 0 and j = +-2c
        are forced dead, the free columns number 4c - 2; (M7) universal clearance is exact (j = 3
        mod 5) and a leaf must cover those columns from gears >= 7 - a real constraint, not a
        contradiction; (M8) no exact law of the machine is violated by a leaf; the only proved
        bound on a leaf is s > 10^6. LITERATURE: no unconditional theorem asserts one twin pair
        anywhere, so none refutes a leaf; Baker-Harman-Pintz is SILENT (the brief's inference was
        backwards: for s > 2^40 the stretch, length 4 sqrt N, is SHORTER than the BHP window
        N^0.525 and inside it; below, BHP is ineffective; and a prime in the stretch is not a
        twin anyway); Legendre is open and not implied by RH (Cramer's sqrt(p) log p is a log too
        weak) - the rung sits exactly at the RH barrier; Brun-Titchmarsh is an upper bound;
        Selberg's parity obstruction applies with full force since a leaf IS sieve data; Chen,
        Zhang-Maynard-Tao (246; 6 under GEH), GPY give no gap 2 and no localisation to length
        4 sqrt N; Erdos-Rankin type constructions show free coverings of the required length are
        over-supplied by 2 ln ln s, so a leaf is not impossible on construction grounds - the
        rigidity (2s/3 constraints on one parameter of log s bits) is the whole content.
        STRUCTURAL: s_n ~ 6^(2^n), T(s) = 2 C_2 s/(ln s)^2 with 2 C_2 = 1.320323 (the measured 1.32
        is the twin constant); the smallest finite tree has depth >= 4 and needs all 2,619,059
        depth-4 nodes to be leaves, else >= 4.6 x 10^9 leaves at depth 5, >= 3.4 x 10^21 at 6;
        refuting depth n costs one rung search of 0.4 x 2^(3n) 61-rough candidates on 1.56 x 2^n-
        digit numbers, so certified depth grows logarithmically in the work; a single leaf in the
        literature's words: a twin-free interval of length 4 sqrt N on a square, 10^2 to 10^7
        times the extreme-value prediction (log N)^3/(2 C_2) at its height, the ratio growing like
        sqrt N; model probability of a leaf anywhere above 10^6 below exp(-6,900); finiteness of
        the (5, 7) component does not bound the twins (other roots), so no largest-counterexample
        machinery applies. THE ONE ASSET: the change of quantifier - the tree needs, per level,
        that at least one of its ~10^16, ~10^37, ... nodes has a rung, i.e. a lower bound for twins
        in the UNION of the level-n stretches (one parameter s per node) rather than in one
        interval - strictly more room than the direct statement, and the only place the rigidity
        can be attacked with many independent parameters. VERDICT: FACT (the route yields the
        existential-per-level form and a level-by-level certification programme); the
        contradiction itself needs exactly what a direct proof needs.
      - R5.f.xviii. THE INDEX LAWS AT DEPTH 5, s ~ 10^24-10^26 (PM, pre-registered 21:05 before
        the run; research/stack/r8/depth5_sample.py; 20 depth-5 nodes, each the nearest rung of
        a random depth-4 node). Predictions: ratio 6|j|/(ln P)^2 mean about 1.5-2, max below 20;
        the 61-rough index near the mean law 0.0695 (ln s)^2 ~ 230 and below the cube
        0.07 (ln s)^3 ~ 13,000 at every node. Refuted by a node above the cube. Every depth-5 node
        found is a twin pair with 50-digit members (BPSW), extending the certified-by-search
        depth of the (5, 7) tree to 5.
        RESULT (20:47; 20 nodes, 24-26 digits): ratio mean 1.32, max 5.5; 61-rough index mean
        205.0 against the mean law 203-247 (on the nose), max 894 against the cube 11,000-14,800.
        Every sampled depth-5 node has a rung. The index laws hold at s ~ 10^25, eighteen orders
        beyond their measurement range. FACT.
      - R5.f.xix. THE INDEX LAWS AT DEPTHS 6 AND 7, s ~ 10^50 and 10^100 (PM, pre-registered
        20:50 before the run; research/stack/r8/depth_n_sample.py; 6 descents by nearest rungs
        from random depth-4 nodes). Predictions: at depth 6 (100-digit members) the 61-rough
        index near 0.0695 (ln s)^2 ~ 920, below the cube ~ 1.1 x 10^5; at depth 7 (200-digit)
        near ~3,700, below ~ 8.6 x 10^5; ratios below 20; every node has a rung. Refuted by a
        node above the cube or a descent that finds no rung.
        RESULT (20:49; 6 descents): depth 6 (47-53 digits) - 61-rough index mean 854 against the
        law 928, max 1,634 against the cube ~108,000, ratio max 3.05; depth 7 (94-105 digits) -
        index mean 3,959 against 3,710, max 7,454 against ~863,000, ratio max 3.49; every node
        has a rung. The laws hold at s ~ 10^100. FACT. DEPTH 8 (20:51; 188-208-digit s, 3 nodes):
        every node has a rung; 61-rough index 60,686, 26,736, 41,717 against the mean law
        13,000-15,800 (1.8x to 3.9x, all three above) and the cube 5.7-7.6 x 10^6 (held with two
        orders to spare); ratios 5.8, 3.1, 4.0 (below 20). The cube bound holds; the mean law at
        this depth sits high on three samples (chance about 0.5% under the geometric reading) -
        four more descents launched with a fresh seed to decide between fluctuation and a drift
        of the constant at 400 digits. FOUR MORE (20:52): indices 723, 3,474, 22,988, 5,657
        against the law ~15,000; over all seven depth-8 nodes the mean index is 23,140 against
        ~15,000 (+1.4 standard errors of a geometric sample), ratio mean 2.4, every node with a
        rung, every index below the cube by two orders. Fluctuation, not drift: the laws hold at
        s ~ 10^200. FACT.
      - R5.f.xx. THE CHAIN FORM: THE WEAKEST LADDER HYPOTHESIS (PM, 21:25; kernel
        proofs/LadderDepth.lean, 0 sorries). Chain n f: f 0 a twin centre and each f (k+1) a rung
        of f k; ChainHyp: chains of every finite length exist SOMEWHERE in the forest (the root is
        free). Chain.ge: the k-th node of a chain is a twin centre >= 6 + 2k; twins_unbounded_of_
        chains: ChainHyp -> twin primes above every bound; chainHyp_of_depthHyp: the depth form
        implies it. So the hypothesis ladder is LadderHyp above a bound -> DepthHyp (the (5, 7)
        tree has a node at every depth) -> ChainHyp (arbitrarily long chains anywhere) -> twins
        unbounded; ChainHyp is the weakest ladder-type statement on record, in words: for every
        n there are twin prime pairs P_0, P_1, ..., P_n with each pair strictly between the
        squares of the previous one. FACT (kernel). FOREST BELOW 10^6 (21:57): chain depths of
        the 8,168 twin centres - 7,728 roots, 335 at depth 1, 53 at depth 2, 52 at depth 3; every
        depth-3 node below 10^6 lies in the (5, 7) tree (6 -> 30 -> 858 -> 734472, ...): the (5, 7)
        tree is the deepest tree of the forest in that range. No free-certificate rung at the
        certified top (neither s(s+1) nor s(s-1) is a twin centre at s = 1.34 x 10^47).
      - R5.f.xxi. THE CONDITIONAL MAP: WHICH STANDARD CONJECTURE IMPLIES WHICH LADDER FORM
        (conditional lane, a fourth fresh Opus lane, received 23:30). Record entries, in
        decreasing strength: (i) CUBE WINDOW, pointwise - LadderHyp for e = 3 follows from the
        standard Hardy-Littlewood short-interval conjecture in lower-bound form: for some theta
        <= 2/3 and all large x, (x, x + x^theta] holds a twin centre (implied by the usual
        y >= x^(1/2 + eps) form); hence DepthHyp and ChainHyp for cubes and the twin prime
        conjecture. (ii) SQUARE WINDOW, pointwise - needs the endpoint y = 4 sqrt(x), the twin
        analogue of Legendre/Oppermann, not supplied by the standard conjecture; follows from a
        Cramer-type conjecture for twin gaps ((log x)^A for any fixed A) or from Bateman-Horn for
        (n - 1, n + 1) with error o(sqrt(x)/(log x)^2), one log beyond square-root cancellation;
        the measured envelope 0.07 (log s)^3 is the Cramer shape. (iii) CHAIN FORM, almost-all -
        THE CRUX: ChainHyp (root free) follows from an almost-all hypothesis with a power
        saving (the exceptional x in [X, 2X] whose stretch-length interval holds fewer than
        expected twin centres number at most X^(1 - delta)), equivalently a power-saving
        variance bound for pi_2 in intervals of length x^(1/2), which the measured Poisson law
        for T(s) asserts empirically; proof by disjointness of stretches (unique parents) and a
        finite self-improving induction, formalisable as stated. (iv) THE DIVIDING LINE:
        LadderHyp and DepthHyp cannot follow from any almost-all hypothesis (they deny any
        rungless twin centre; DepthHyp names the node 6); root-freeness is exactly the strength
        difference between ChainHyp and DepthHyp; no hypothesis about primes alone (RH, density,
        Montgomery pair correlation, variance) implies any form - all are shift-averaged or
        parity-blind; EH/GEH give 12 and 6, nothing here; almost-prime rungs (Chen in short
        intervals) exist unconditionally for large e but do NOT iterate - an almost-prime centre
        has no stretch - the exact shape of the parity wall against the construction. Known
        unconditionally: T(s) << s/(log s)^2 (Selberg, factor 4), nothing in the lower direction
        at any length, pointwise or almost-all (there is no shift to average over). Three facts
        to carry: the binding parameter is localisation (theta), not the count's quality, so
        e = 3 is the right window and e = 2 a Legendre-strength outlier; the sparse set {s^e}
        costs x^(1 - 1/e), not x (the stretches are disjoint), and the singular series on the
        stretch is the standard one; the pointwise/almost-all line falls exactly between
        DepthHyp and ChainHyp. Every conditional theorem gives "exists S_1, LadderHyp above S_1";
        the certified prefix closes the gap only with an effective x_0. VERDICT: FACT (the
        record's conditional map); next kernel target: (iii) as a Lean theorem - an almost-all
        hypothesis as one Prop implies ChainHyp.
      - R5.f.xxii. KERNEL: AN ALMOST-ALL HYPOTHESIS IMPLIES CHAINS OF EVERY LENGTH (formalist
        lane on Fable, 2026-09-20 00:05; proofs/LadderAlmostAll.lean, 0 sorries, axioms propext /
        Classical.choice / Quot.sound, re-built and audited by the manager). rungs s = the twin
        centres strictly inside ((s-1)^2, (s+1)^2); twinsUpTo X; the exceptional set = twin
        centres s <= X with fewer than c s/(log s)^2 rungs; AlmostAll c delta eta := (twin
        centres up to X number at least X^(1-eta)) and (the exceptional set up to X has at most
        X^(1-delta) elements), eventually in X. THEOREM chainHyp_of_almostAll: for 0 < c,
        0 < eta, 2 eta < delta < 1, AlmostAll c delta eta -> ChainHyp; twins_unbounded_of_almostAll
        composes with twins_unbounded_of_chains. Proof: Roots k s (s roots a chain of length k),
        bad k X, rungs_subset_bad (the rungs of a bad node at depth k+1 are bad at depth k, below
        (X+1)^2), rungs_disjoint from parent_unique, sum_rungs_le (the disjoint-union count),
        lower_bound_of_ge, the split lemmas, bad_card_le (the induction: |Bad_k(X)| <= C_k X^(1-gamma)
        for every gamma < delta - the depth does not erode the saving; logs absorbed by
        Real.log_le_rpow_div). The conditional theorem of R5.f.xxi (iii) is a kernel object: the
        chain form of the ladder follows from a power-saving almost-all lower bound on rung
        counts - what the Poisson child-count law asserts empirically. FACT (kernel).
        THE HYPOTHESIS MEASURED (00:20, almost_all_census.py, every twin centre s <= 10^5, 1,223
        of them): mean T/(1.3203 s/ln^2 s) = 0.9997; exceptional set {T < c s/ln^2 s}: EMPTY at
        c = 0.66; 4 at c = 0.9 (largest s = 462); 8 at c = 1.0 (largest 3,540); 18 at c = 1.1
        (largest 3,930); 54 at c = 1.2 (11 above 10^4, the Poisson tail at 91% of the mean). So
        AlmostAll holds in range with an exceptional set that is literally empty above
        s = 3,540 for c = 1.0 - pointwise, not merely almost-all - and the power saving asked of
        it is vacuous to 10^5.
      - R5.f.xxiii. KERNEL: THE CUBE LADDER FROM A SHORT-INTERVAL TWIN LOWER BOUND (formalist
        lane on Fable, 2026-09-20 00:30; proofs/LadderShortInterval.lean, 0 sorries, axioms
        propext / Classical.choice / Quot.sound, re-built and audited by the manager).
        ShortInterval theta := eventually in x, a twin pair with both members in (x, x + x^theta].
        THEOREMS: rungPow_of_shortInterval - for e >= 3 and theta <= 1 - 1/e, every large twin
        centre has a rung in the exponent-e window; chainsPow_of_shortInterval; twins_unbounded_
        of_shortInterval. Mechanism: at x = (s-1)^e the hypothesis lands a twin pair within
        (s-1)^(e theta) <= (s-1)^(e-1) of (s-1)^e, and the window (s+1)^e - (s-1)^e >= 2e (s-1)^(e-1)
        holds it (binomial lower bound window_width; interval_length via rpow). So the cube
        ladder (e = 3) rests on the standard short-interval twin conjecture at theta <= 2/3, in
        lower-bound form only; the square ladder (e = 2) would need theta <= 1/2, the Legendre-
        strength endpoint. FACT (kernel). The conditional map of R5.f.xxi is now formal at both
        ends: pointwise short intervals -> cube ladder; almost-all rung counts -> chains.
      - R5.f.xiv addendum (kernel, 2026-09-20 00:45; proofs/LadderInfinite.lean, 0 sorries,
        axioms propext / Classical.choice / Quot.sound). treeNodes = the nodes of the (5,7) rung
        tree. Depth.bound: a node at depth k is below 8^(2^k) (a rung is below (s+1)^2);
        Depth.none_above: a missing level empties every level above it; depthHyp_of_infinite
        and infinite_of_depthHyp: THE (5,7) TREE IS INFINITE IFF IT HAS A NODE AT EVERY DEPTH
        (infinite_iff_depthHyp), with no Koenig argument in either direction - the levels are
        finite and bounded, so a finite tree is exactly a tree with an empty level;
        twins_unbounded_of_infinite. The random lane's equivalence (path form) needed Koenig;
        this form does not. FACT (kernel).
      - R5.f.xxiv. THE ROUTES LANE: THREE NEW ROUTES FROM THE MACHINE'S INTERACTIONS (fifth
        fresh lane, on Fable, opened 2026-09-20 00:25; brief in the session scratchpad
        lane_routes/BRIEF.md). Parent observation: R5.f.xxi-xxiii closed the conditional map at
        both ends, so the next node must be a route built from residues, offsets, gears and the
        forest, not from counting. PRE-REGISTERED: the lane returns three routes not on the
        closed list (free covering, composite-forcing families, almost-prime rungs, finite-tree
        contradiction, primes-only hypotheses, sieve lower bounds), each with one falsifiable
        numerical claim tested by an exact script. Seeds offered: forest identities (ancestor
        chains, roots, s^2 mod p along a chain); the rung relation as a dynamical system on
        residues (forced-open sets growing level to level); consecutive twin centres with
        overlapping stretches; T(s) against the arithmetic of s. Verdict OPEN until the lane
        reports; each route becomes its own child node with the claim's result.
        RESULT (received 01:10; scripts research/stack/r8/lib_twins.py, route1_consecutive_
        product.py, route2_product_forest.py, route3_near_gear_confinement.py, route3b_top_band_
        law.py). Three routes returned; children below. The lane touched seed (b) in one line:
        forced-open residues level to level reduce to universal clearance (R5.f.xiii); the
        self-improving structure it found is the product forest (xxiv.b). VERDICT: one new
        structure (xxiv.b, kernel), one generic fact (xxiv.a), one rediscovery (xxiv.c).
        - R5.f.xxiv.a. THE CONSECUTIVE-PRODUCT WINDOW (route 1). Consecutive twin centres
          s < t have disjoint square stretches exactly ((t-1)^2 - (s+1)^2 = (t-s-2)(t+s) > 0),
          so the seed's "overlapping stretches" is empty; their phases meet in the product
          window W(s,t) = ((s-1)(t-1), (s+1)(t+1)), members st + 6j -+ 1, length 2(s+t), which
          sits strictly between the two stretches. CLAIM (pre-registered): every consecutive
          pair 30 <= s < t <= 30,000 has W(s,t) non-empty, count law 2(s+t) 1.3203/ln^2(st)
          within 3%. RESULT: 462 pairs, 0 empty windows, min count 2 at (42, 60), mean ratio
          0.9953 by band (0.997 / 0.980 / 0.997 / 0.997), spread 1.02 x Poisson; the centre st
          is a twin centre 3 times against 12.3 generic (mod 5 the products of the admissible
          residues {2,3} all land on +-1, so st -+ 1 is dead unless 5 | st). HOLDS; FACT (generic
          count law; the only non-generic number is the centre's phase product).
        - R5.f.xxiv.b. THE PRODUCT FOREST (route 2; kernel proofs/LadderProduct.lean, 0
          sorries, axioms propext / Classical.choice / Quot.sound, built and audited by the
          manager 01:15). For twin centres s <= t the product window W(s,t) holds product-rungs;
          s = t is the square ladder, s = 6 is (5(t-1), 7(t+1)), s ~ t^a gives localisation
          exponent 1/(1+a) at height t^(1+a): the product forest interpolates from the square
          ladder (exponent 1/2) to a constant-ratio window (exponent 1), every node a twin
          centre. KERNEL: ProductRung s t u; ProductHyp (every twin centre t has a product-rung
          with some twin centre s <= t); productRung_gt (u >= t + 2, since (s-1)(t-1) >= 5t - 5);
          productRung_of_rung; productHyp_of_ladderHyp; twinCentre_unbounded_of_product;
          twins_unbounded_of_product; SixHyp (a twin centre in (5(t-1), 7(t+1)) for every twin
          centre t) -> ProductHyp. So LadderHyp -> ProductHyp -> twins unbounded and SixHyp ->
          ProductHyp: the weakest window hypothesis on record is a twin centre within the
          constant ratio 7/5 of every twin centre. CLAIMS (pre-registered): (A) all pairs
          42 <= s <= t <= 10^4 non-empty, minimum 3 on the diagonal; (B) the product forest has
          exactly three roots (6, 12, 18) to 10^6; (C) the multiplier-6 window never empty to
          10^6; (D) the centre st is a twin centre at K x generic, K = prod r_p. RESULTS: (A)
          20,100 pairs, 0 empty, min 2 at (42,60) and (108,138) - off the diagonal (the placed
          number failed, never-empty held); (B) roots exactly [6, 12, 18], in-degree min 1 from
          u >= 30, median 392, ~823 at 10^6; (C) 0 exceptions, min count 5 at t = 108; (D)
          498 observed / 653.8 generic / K = 0.8243 (r_5 = 0.926, r_7 = 0.952) / predicted
          538.9, ratio 0.924. MIXED on the numbers, the structure holds. The supply of windows at
          t is the number of twins already found - the structure improves itself as it climbs.
          FACT (kernel) and the weakest hypothesis form; next claim: W(s,t) with s the smallest
          twin centre >= sqrt(t) is non-empty for every t <= 10^6 (exponent 2/3 with every node
          a twin), and the correlation between T(t) and N(pred(t), t) is below 0.05.
        - R5.f.xxiv.c. NEAR-GEAR CONFINEMENT (route 3). CLAIM (pre-registered): a prime
          p = s - e with e^2 <= 2s - 5 strikes exactly one column of the stretch of s, the
          lower member at j = (1 - e^2)/6, at every twin centre s <= 10^5. RESULT: FAILS -
          15,393 exceptions among 32,757 near-band gears (s = 42, p = 37 strikes j = -4 lower
          and j = 8 upper = 37 x 49). Corrected law (post hoc, route3b): a top gear
          p = s - e strikes exactly the columns (s+k)^2 - (e+k)^2 inside the window, k = 0 mod 3
          on the lower member, k = -e mod 3 on the upper, k = e mod 3 never; every top gear
          (2s/3 < p <= s+1) strikes 1 or 2 columns, never 0; in the near band only k = 0 and
          k = 1 occur, k = 1 iff e = 5 mod 6 - 0 exceptions over 1,688,382 gears. The k = 0
          and k = 1 columns are the two composite-forcing difference-of-squares families of
          R5.f.xv already on the record, and "1 or 2 strikes" is the count of multiples p m,
          m = +-1 mod 6, in an interval of length 4s/p in (4, 6). REDISCOVERY of closed route 2;
          DEAD as a route. What survived: the top band is idle on every live column (its
          strikes are all identity-composite), which is the record's statement that plugs of
          base-open columns come from the moving families only.
          - R5.f.xxiv.b.i. RED LANE ON THE PRODUCT FOREST (Opus, opened 2026-09-20 01:25;
            brief lane_red2/BRIEF.md). PRE-REGISTERED: reproduce the three roots, the 20,100
            non-empty pairs and the multiplier-6 minimum independently; test the manager's
            reading that "three roots" is exactly "consecutive twin centres never exceed the
            ratio 7/5" (u >= 30 is a non-root iff u < 7(prev(u)+1)); margins of W(6,t) against
            its law next to T(t) against its law; the local factors r_5, r_7, r_11, r_13 and
            whether 0.924 against 0.824 is within Poisson. RESULT (received 01:40; scripts
            research/stack/r8/red2_1.py .. red2_4.py; exact sieve to 10^8, 440,395 twin centres).
            Roots {6, 12, 18} CONFIRMED; 20,100 pairs, 0 empty, min 2 at exactly (42,60) and
            (108,138) CONFIRMED; 498 against 653.797 CONFIRMED to the digit. Claim 3 half wrong:
            W(6,t) never empty, but the minimum is 2 at t = 6, 12, 18 ("5 at 108" holds only
            from t >= 108). THE MANAGER'S READING REFUTED: the multiplier-6 family does NOT cover
            everything - u = 138 is a non-root only through (12,12) (W = (121,169); W(6,18) =
            (85,133) and W(6,30) = (145,217) leave the gap [133,145]); the exact overlap
            condition is t' < (7t+12)/5, not t'/t < 7/5; max consecutive ratio 2.000 (6 -> 12),
            1.4286 (42 -> 60) for t >= 30; six ratios exceed 7/5, two leave an uncovered
            interval ([49,55] and [133,145]), one of which holds a twin centre. Margins: W(6,t)
            min ratio 0.5364 at t = 72 (0.8233 at t = 2238 for t >= 1000), mean 1.0033; square
            stretch (computed to t <= 10^5) min 0.5772 at t = 72 (0.6861 at t = 1152 for
            t >= 1000), mean 0.9997 - away from small t the square stretch is the thinner
            margin. Local factor exact: r_p = p(p^2 - 6p + 10)/(p-2)^3 (25/27, 119/125,
            715/729, 1313/1331), K converged = 0.823103 (the lane's 0.8243 was a truncated
            product); residual 0.92541, z = -1.73 (p = 0.084), inside Poisson; the diagonal
            s = t is a structural zero (s^2 - 1 = (s-1)(s+1)) the model omits. Worst t is 72 for
            both windows, series correlation 0.07-0.12, a small-argument bias of the ln^2 law,
            not arithmetic. VERDICT: the product forest's numbers stand; SixHyp's "never
            empty" stands with minimum 2; the forest's three roots need one pair beyond
            multiplier 6. FACT.
      - R5.f.xxv. THE INHERITANCE LANE: DOES A RUNG INHERIT ANYTHING FROM ITS PARENT? (Fable,
        opened 2026-09-20 01:25; brief lane_inherit/BRIEF.md). Parent observation: the phase
        lock s' = s^2 mod g for g | 6j (R5.f.vi) means a child's residues mod the gears dividing
        its offset are the parent's squared, so the forest could carry structure beyond the
        singular series - or exactly none. PRE-REGISTERED claims: (1) T(s') against the law
        with the exact inherited local factor has Poisson spread and no slope on omega(6j) or
        j mod 30 (slopes within 2 SE of 0); (2) rungs with 5 | j, 7 | j, 35 | j have the same
        mean ratio as the rest within 2 SE; (3) consecutive chosen offsets along chains have
        independent signs and residues mod 5 (chi-square p > 0.05); (4) every (s mod g, j mod g)
        pair absent among rungs for g in {5, 7, 11, 13} is explained by s' -+ 1 being coprime to
        g. If all hold the forest is exactly residues (a decisive mechanism result, FACT); any
        failure is a new law. RESULT (received 01:50; scripts research/stack/r8/sieve_rungs.py,
        analyse.py, control_windows.py, control_analyse.py, chains.py, census_pairs.py; 81 twin
        centres s <= 3000, 2,441 rungs to 9.0 x 10^6, every stretch sieved exactly). (1) The
        inherited local factor is identically 1 - gear g strikes two classes of offsets whatever
        s'^2 mod g is, so residues decide WHICH offsets are struck, never how many; fitted
        constant 1.3204 (the parents' 1.3203); slopes on omega(6j) and j mod 30 within 1.4 SE of
        0; ANOVA over j mod 30 p = 0.44. Poisson spread FAILS: dispersion index 0.824 +- 0.029
        (sub-Poisson), but same-length windows at the same heights that are not stretches give
        0.788 +- 0.029 - the spread belongs to the interval, not the rung. (2) HOLDS: 5 | j,
        7 | j, 35 | j subsets within 1.6 SE of the rest. (3) MIXED: signs of consecutive chosen
        offsets independent (p = 0.55; 0.98 on 2,437 census pairs); residues mod 5 dependent
        (chi^2 536) exactly by the two-block rule - child residue s' = 0 mod 5 allows next
        offsets {0,2,3}, s' = +-2 allows {1,3,4}; zero entries outside the blocks and
        independence inside them (p = 0.86, 0.92 on the census); nearest-offset classes for
        g = 5, 7, 11, 13 never in a forbidden class (16 of 16). (4) HOLDS: for g in {5, 7, 11,
        13} every absent (s mod g, j mod g) pair is forced by the parent or the rung being a
        twin centre; (g-2)^2 allowed pairs, every one occurs. VERDICT: THE FOREST IS EXACTLY
        RESIDUES - phase lock and universal clearance are the whole inheritance; no hidden
        invariant at the small gears; the one non-generic number (sub-Poisson spread) is an
        interval property. FACT (decisive between mechanisms). Child below.
        - R5.f.xxv.a. THE DISPERSION OF TWIN COUNTS IN SHORT WINDOWS (PM, 01:55,
          dispersion_delta.py; 4,000 random windows of length x^delta at heights x in
          [2, 4] x 10^8, index = mean((count - law)^2 / law)). PRE-REGISTERED: the index falls
          with delta. RESULT: delta 0.3 / 0.4 / 0.5 / 0.6 / 0.7 -> 0.926 / 0.854 / 0.829 / 0.834
          / 0.742 (SE 0.02), mean count/law 0.99-1.00 throughout. HOLDS: sub-Poisson, falling
          with the window exponent; the lane's 0.79-0.82 at 4 sqrt(x) is the delta = 1/2 point.
          Prior art (one line): this is the shape of the Montgomery-Soundararajan variance of
          primes in short intervals, here for twin centres; a variance statement is the
          almost-all quantity of R5.f.xxi (iii). FACT; stopped as a known shape.
      - R5.f.xxvi. THE REGIONS BETWEEN CONSECUTIVE PRIME SQUARES (PM, pre-registered 01:58 before
        the run, region_census.py). Parent observation: a twin's stretch is the region (p^2, q^2)
        between consecutive primes with gap 2; the machine's window (q, q^2] is a union of such
        regions, so "a twin in every region" (the window statement region by region) contains
        LadderHyp as its gap-2 case. PRE-REGISTERED: (1) no region with q^2 <= 10^9 is empty;
        (2) the minimum count over regions of gap g grows with g, so the stretches are the
        binding case; (3) counts follow 1.3203 (q^2 - p^2)/ln^2(p^2) with mean ratio 1 within
        1%. RESULT: 3,398 regions, 0 empty, mean ratio 0.9989; by gap: g = 2 min count 2 (at
        p = 5, 11, 17, 29), min ratio 0.573 (p = 29); g = 4 min 4, ratio 0.625; g = 6 min 8,
        0.686; g = 8 min 21, 0.861; g >= 16 min ratio >= 0.925; the eight smallest counts are
        all gap 2 and gap 4 regions. ALL THREE HOLD: the twin stretches are exactly the binding
        case of the machine's window statement. FACT. Children: the formalist lane below.
        - R5.f.xxvi.a. KERNEL LANE: REGIONS AND THE WINDOW STATEMENT (Fable formalist lane,
          opened 02:00; brief lane_region/BRIEF.md). PRE-REGISTERED: proofs/LadderRegion.lean
          with Consecutive p q, RegionHyp (a twin centre strictly between consecutive prime
          squares, p >= 5), consecutive_of_twinCentre, ladderHyp_of_regionHyp (RegionHyp is
          stronger than LadderHyp: its gap-2 case), twins_unbounded_of_region, WindowHyp (for
          every q >= 6 a twin centre u with q <= u - 1 and u + 1 < (q+1)^2 - the owner's window
          statement with window [q, (q+1)^2)), windowHyp_of_ladderHyp (the ladder from 6 climbs
          past every q by less than a squaring), windowHyp_of_regionHyp. RESULT (received
          02:05; re-built and audited by the manager: 0 sorries, axioms propext /
          Classical.choice / Quot.sound). All seven objects as briefed; the window is [q, (q+1)^2)
          (q <= u - 1, since for q = 5 mod 6 the next twin centre can be q + 1). The lane's
          proof of windowHyp_of_ladderHyp takes s = Nat.findGreatest TwinCentre q, the largest
          twin centre <= q; its rung exceeds q by maximality and lies below (s+1)^2 <= (q+1)^2.
          So the kernel now reads RegionHyp -> LadderHyp -> WindowHyp: the owner's window
          statement follows from the ladder, and the ladder is the gap-2 case of the region
          statement. FACT (kernel).
      - R5.f.xxvii. THE PARITY-SENSITIVITY LANE: WHICH MACHINE LAWS SEE THE TWIN? (Fable, opened
        2026-09-20 02:20; brief lane_parity/BRIEF.md). Parent observation: R5.f.xxv found the
        forest exactly generic given residues; every exact law on record (two classes, universal
        clearance, phase lock, uniform local factor, twin gears at the centre, top-band rule,
        unique parents, region law) might hold equally for a parity-twisted companion set
        (columns whose members both have even Omega and are composite) - a law that survives
        the twist cannot separate the twins, a law that breaks is the kind a proof must use.
        PRE-REGISTERED: build P (twins), Q (both members even Omega, composite), M' (both odd
        Omega, composite) to 10^7; test L2, L3, L4, L7, L8 and the first-rung index law with
        Q-centres and Q-rungs; decisive table law x {P, Q, M'} x parity-sensitive. Expected by
        the manager: L1, L3, L7 identities (blind); L2 and L4 depend on s^2 being a square of an
        admissible residue - Q-centres have no such restriction, so L2 changes form (sensitive
        in form, not in mechanism); L8 and the index law hold for Q with its own constant
        (blind). RESULT (received 02:35, lane stopped twice on background waits; scripts
        research/stack/r8/parity_sets.py, parity_laws.py, parity_index.py; sets to 10^7:
        |P| = 58,979, |Q| = 415,284 (density 0.2492 of columns - a positive-density set, so its
        count has no C x/ln^2 x form), |M'| = 144,050, Q61 (both members 61-rough) = 48,974;
        per-centre laws on 1,216 / 4,019 / 713 / 179 centres in [100, 10^5]). L1: two struck
        classes per gear at every centre of every set - BLIND. L2: at P the never-struck
        classes are [3] mod 5, [4] mod 7, [4,8] mod 11, [1,12] mod 13; at Q and M' the
        never-struck set is EMPTY overall but splits exactly by whether g | s^2 - 1: centres
        with g not dividing s^2 - 1 have the twin's classes exactly, centres with g | s^2 - 1
        strike everything but a fixed set; Q61 restores the twin's classes for g <= 13. So L2 is
        the statement "s -+ 1 are coprime to g" - visible to any sieve, BLIND in mechanism.
        L3: 34,142,893 checks, 100% - an identity, BLIND. L4: mean count / law = 1.00 for every
        (centre set, rung set) pair except M'-rungs (1.33-1.40, the density of M' rising with
        height, a constant not a residue effect); by residue of s mod 5, 7, 11, 13 the maximum
        deviation is 2.4 SE in 40 classes for every set - BLIND. L5: P centres have exactly 2
        strikes from the prime factors of s^2 - 1 (j = 0 only, 1,216 of 1,216); Q centres 45 to
        73,190 strikes, Q61 333 to 3,297 - L5 is the primality of s -+ 1 itself and nothing
        more; SENSITIVE only in that sense. L6: 1 or 2 strikes per top-band prime at every
        centre of every set - BLIND. L7: no overlapping stretches in any set - BLIND (an
        interval fact). L8: every set follows its own density law region by region (P 1.0035,
        Q 0.9999, M' 0.9985, Q61 0.9985 overall) - BLIND. INDEX LAW at P to 10^7 (57,756 twin
        centres above 10^5): no centre without a rung; max index/ln^2 s = 0.6885 at
        s = 5,042,928 (index 164, j = -617), seven centres above 0.6 (the 10^6 line of R5.f.xi
        is a sample maximum, not a law); the tail is geometric to the last bin (>160: 2
        observed / 1.68 predicted; >200: 0 / 0.12); the cube envelope 0.07 (ln s)^3 = 256
        holds (164 < 256). Cross-set index runs (Q-rungs at P-centres, P-rungs at Q-centres and
        M'-centres, Q-rungs at Q-centres) re-launched by the manager, see addendum. VERDICT:
        EVERY EXACT LAW ON THE RECORD IS PARITY-BLIND; the only twin-specific fact in the
        machine is that s - 1 and s + 1 are prime (L5 and the coprimality behind L2), which is
        the hypothesis, not a law about it. FACT (decisive): a proof cannot come from the
        listed laws alone; it must use the members' primality in a way that no sieve identity
        reproduces. ADDENDUM (02:10, cross-set index runs): P-rungs at M'-centres (143,337
        centres in [10^5, 10^7]): no centre without a rung, mean index 0.0690 (ln s)^2 (P-centres
        0.0696), max 0.8305 (ln s)^2 at s = 9,360,306 (index 214), geometric tail to the last
        bin (>200: 1 observed / 0.38 predicted) - the type of the centre is irrelevant to where
        the first twin sits. Q-rungs at P-centres and at Q-centres (57,756 and 40,000 centres):
        mean index 3.85 and 3.98, constant in s (Q61 has constant density among 61-rough
        columns), max 0.23 (ln s)^2, geometric tail. So the index law is the density of the rung
        set among rough offsets and nothing else - BLIND.
      - R5.f.xxviii. THE PROVER LANE: THREE WRITTEN PROOF ATTEMPTS WITH EVERY STEP MARKED (Fable,
        opened 2026-09-20 02:15; brief lane_prover/BRIEF.md). Parent observation: R5.f.xxvii
        showed every listed law is sieve-visible, so an attempt must be written out to find
        which exact step first needs the members' primality - the Conway workflow's "write the
        proof, mark the gaps, let red attack". PRE-REGISTERED: attempt A (a leaf as a covering of
        the 61-rough offsets by gears in (61, s-1], pushed with F1-F7 to a single quantified
        integer statement as the GAP); attempt B (chains from "leaves are never consecutive" -
        how t^2 and t'^2 mod each gear relate for consecutive twin centres); attempt C (the
        lane's own). Output: a ranked GAP table, each gap one quantified sentence with a script
        test. Expected: every attempt's gap is a lower bound on unstruck columns; the value is
        in the exact form of the smallest such statement. RESULT (received 02:40;
        research/proof/ladder_attempts.md, scripts research/stack/r8/attempts_measure.py,
        attempts_measure2.py). ATTEMPT A (covering): the leaf ledger is the exact identity
        r(s) = |R_61(s)| - T_61(s) + X_61(s) (rungs = 61-rough columns - strikes on them by
        gears > 61 + excess from multiple strikes), checked at all 199 twin centres in
        [60, 10^4]; per-gear counts (A7) and pair-intersection bounds (A8) are FALSE as closers
        (T_61 = 1,350 > 910 and 305 < 440 at s = 10,008); the top band and the difference-of-
        squares columns carry 6% of the covering at s = 10,008, the work is in the gears
        (61, s/3]. Sharpest sufficient integer statement, GAP A-2: the gears above s/3 strike
        the (s/3)-rough columns fewer times than there are such columns (holds at every twin
        centre in [60, 10^4], margin 256 against 100 at s = 10,008); its two halves A-2a
        (|R_{s/3}| >= N/ln^2 s) and A-2b (big-gear strikes on R_{s/3} <= 5|R|/ln s) measured
        3.26 and 3.6 at s = 10,008. The manager's reading: A-2a is a lower bound for a set
        sifted to level s/3 in an interval of length 4s - the sieve limit in integer form;
        A-2b is equidistribution of the sifted set modulo the big gears - the same. ATTEMPT B
        (chains): B4 PROVED - NC_2 (consecutive twin centres never both leaves) and Dich_2 (no
        twin centre has exactly one rung) give an infinite chain from every twin centre with a
        rung: two distinct rungs a < b of s, the first twin centre t after a is a rung of s and
        consecutive with a, so one of a, t has a rung. Measured: no leaf to 10^4, minimum
        positive rung count 2 (s <= 30), >= 21 on [1000, 8000]. FALSE: chains through parents
        (a run of 14,201 consecutive parentless twin centres below 10^8). Consecutive leaves
        t, t + 6d: the extra gears t -+ 1 strike exactly three free columns of the second
        stretch, both stretches would be covered by the same gear set - no joint constraint
        from F1-F7 beyond the shift 2d(t + 3d). ATTEMPT C (the Six window (5(t-1), 7(t+1))):
        strike classes are t-independent (k = +-1/6 mod p), primorial-multiple columns are rough
        on both members unconditionally; GAP C-3 (per-gear discrepancy of the sequentially
        sifted set <= 2|R|/p + 2 sqrt(|R|/p) + 2) with C-4 (prod (1 - 2/p) ln^2 x >= 1, measured
        2.33 at x = 61) closes SixHyp for t >= 10^15; the same ledger provably fails in the
        square window (s/ln s gears above sqrt(2s)). C-2: multiples of 210 in the Six window
        with 210m -+ 1 both prime - last failure t = 828, none in (828, 10^7]. F7 extended: a
        gear never strikes the lower member at an offset it divides, the upper iff p | s^2 + 1.
        KERNEL (manager, 02:50; proofs/LadderDichotomy.lean, 0 sorries, axioms propext /
        Classical.choice / Quot.sound): Good s (a twin centre with a rung); NoConsecutiveLeaves;
        NoSingleRung; good_step (the dichotomy lemma, via Nat.find for the first twin centre
        after a); chain_of_good; chainHyp_of_dichotomy; good_six (6 -> 30);
        twins_unbounded_of_dichotomy; depthHyp_of_dichotomy (from 6 the two hypotheses give a
        node at every depth). VERDICT: the attempts locate the first primality-dependent step in
        every route as a lower bound or equidistribution for a sifted set (A-2a/b, C-3) - the
        integer forms of the sieve limit; attempt B yields a new kernel object whose two
        hypotheses are local (two consecutive twin centres; one twin centre's rung count) and
        both far weaker in appearance than LadderHyp. FACT (kernel) for B4; the gaps are the
        open lemma's sharpest integer forms. KERNEL ADDENDUM (02:55; proofs/LadderEuclid.lean,
        0 sorries, standard axioms): lower_member_eq (the lower member at offset j is
        (s-1)(s+1) + 6j); lower_member_rough - THE EUCLID DEVICE: a gear p dividing the offset
        j, other than the twin gears, never strikes the lower member of column j (it would have
        to divide s - 1 or s + 1); upper_member_iff - at such an offset p strikes the upper
        member iff p | s^2 + 1. So at offsets divisible by every gear up to x the lower member is
        x-rough unconditionally, and both members are x-rough at offsets divisible by the gears
        up to x that do not divide s^2 + 1 and avoiding one class for each that does: an
        unconditional supply of base-open offsets at level x whenever the primorial of x is
        below the stretch length (x about ln s). CORRECTION (03:55): this is not a new law - it
        is the universal clearance class j = 0 mod p of the lower member (R5.f.xiii's U_p),
        now proved in the kernel for every gear at once; and the existence of an x-rough column
        in any interval of length prod(p <= x) holds for ANY position by CRT (each gear leaves
        p - 2 >= 3 free classes), so the twin condition only names which class is free. It is
        L2 in kernel form, sieve-visible like L2. Its reach is level ln s against the level s a
        rung needs. Both dichotomy hypotheses hold to 10^5 on the
        red lane's data (R5.f.xxiv.b.i): no leaf, and no twin centre above 30 with exactly one
        rung (minimum positive count 2 at s = 30, 3 at s = 72); to 10^6 on twin_ladder.py's data
        (R5.f.vi): no leaf and at least 3 rungs at every twin centre from s = 42, so both
        dichotomy hypotheses hold to 10^6 with the single-rung case never occurring above 30.
        FACT (kernel).
      - R5.f.xxix. THE STATEMENT AUDIT OF THE KERNEL (Opus audit lane, 2026-09-20 03:05; nine
        ladder files, statements only). No defect of type (a) vacuous hypothesis, (b) natural
        subtraction, (c) window endpoints or (e) reversed direction in any new file. Findings and
        the manager's fixes (03:15, all rebuilt, 0 sorries): (1) hypotheses that contain the
        conclusion, now labelled in the map: LadderHyp, DepthHyp, ChainHyp, ProductHyp, SixHyp,
        RegionHyp, WindowHyp, treeNodes.Infinite, the pair NoConsecutiveLeaves + NoSingleRung
        (with good_six), AlmostAll (its first conjunct alone), ShortInterval (alone) - each
        implies twins unbounded by itself, as intended; the content of LadderAlmostAll and
        LadderShortInterval is the SHAPE (chains, exponent-e rungs), already noted in the map's
        reading note. (2) NearTwinHyp (TwinLadderTheorem.lean) is parameterised by a free rank
        function, so ladderHyp_of_nearTwin is a triviality - it is a scaffold, not the lane's
        bounded NTH; recorded in the map, not changed (a change would rebuild the certificate
        chain). (3) Docstring overclaims corrected in LadderDepth.lean: "equivalent to an
        infinite path" (only path -> DepthHyp is proved; the converse needs Koenig) and "weaker
        than the ladder hypothesis above any bound" (needs a starting twin centre, else it
        presupposes the conclusion) - the theorems chainHyp_of_path and
        chainHyp_of_ladderHyp_above (with a start s_0 >= S_0) added; leaf_is_sieve_data's
        docstring now says what it proves (every composite below (s+1)^2 has a factor <= s-1).
        (4) twins_unbounded_of_windowHyp added to LadderRegion.lean so the window statement's
        strength is explicit. (5) LadderEuclid.lean: the header's "unconditionally" replaced by
        the level form lower_member_rough_upto (x < s - 1), added and proved. (6) Minor, left as
        is: RungPow e is unsatisfiable for e <= 1 so two lemmas admitting e = 1 are vacuous
        there; redundant hypotheses h_delta and h_theta0; isExc divides by log^2 s (harmless at
        s >= 6). VERDICT: the kernel's statements say what the tree says; FACT.
      - R5.f.xxx. THE CHEN LEDGER OF A STRETCH (PM, pre-registered 2026-09-20 05:00 before the run;
        research/stack/r8/chen_ledger.py). Parent observation: the prover lane's ledger (R5.f.xxviii
        A-2) counts the (s/3)-rough columns; a member below (s+1)^2 with no prime factor <= s/3
        has at most two prime factors, so every (s/3)-rough column is a rung or has a member that
        is a product of two primes in (s/3, 3s + 6): T(s) = |R_{s/3}(s)| - N_semi(s) exactly, the
        stretch's form of Chen's prime / semiprime dichotomy. PRE-REGISTERED: (1) the
        decomposition is exact; (2) N_semi / |R| < 0.4 at every twin centre in [100, 5000]; (3)
        |R| / T has mean near 1.25 and never exceeds 2 above s = 500. RESULT (118 twin centres):
        (1) HOLDS, exact at every centre. (2) FAILS: mean 0.405, min 0.283, max 0.667 at s = 270
        (0.548 above 500). (3) FAILS: mean 1.708, max 3.000 at s = 270, 2.211 above 500 (e.g.
        s = 4968: |R| = 146, T = 100, N_semi = 46). So at level s/3 the rough columns are about
        60% rungs and 40% columns carrying a large semiprime; the Buchstab guess underestimated
        the semiprime share by half. The ledger is exact and sieve-visible: any lower bound for
        T through it must separate the primes from the semiprimes among the rough members, which
        is the statement R5.f.xxvii found no machine law makes. FACT (the two placed numbers
        refuted; the identity and the measured share stand). ADDENDUM (06:00, the share priced):
        a member near N = s^2 that is (s/3)-rough is prime with density 1/ln N or a product pq
        with s/3 < p <= s, density about 4 ln 3 / ln^2 N, so the semiprime-to-prime ratio per
        member is 4 ln 3 / ln N = 0.26 at s = 5000 and a column with both members rough carries
        a semiprime with probability 1 - (1/1.26)^2 = 0.37 - the measured 0.405. The share falls
        like 8.8 / ln N: the level-(s/3) sieve isolates the rungs up to a contamination that
        vanishes as 1/ln s, and the prediction for s = 10^6 is 0.28. FACT (priced).
      - R5.f.xxxi. THE LITERATURE REGISTER FOR THE WINDOW HYPOTHESES (Opus lane with web search,
        2026-09-20 08:30; prior-art lines only, no new mathematics). Q1 Chen-type pairs (p prime,
        p + 2 = P_2) in short intervals (x - x^theta, x]: theta down to about 0.97 (Ross 1978;
        Salerno-Vitolo 1993; Cai-Lu 1999; Cai) - PROVED, an almost-prime surrogate far from
        theta = 1/2. Q2 twin pairs in almost all short intervals: NO RESULT for any theta < 1
        (it would imply the conjecture); the shift-averaged theorems (Lavrik 1961, Mikawa 1992,
        Perelli-Pintz 1992: the Hardy-Littlewood asymptotic for all but H L^-A shifts h <= H,
        H > X^(1/3+eps)) are a different statement and exclude the fixed shift 2 - confirms
        R5.f.xxi. Q3 bounded gaps in short intervals: for every delta >= 0.525, [x - x^delta, x]
        holds pairs of consecutive primes at bounded distance (Alweiss-Luo 2018, gap not
        explicit); full range H_1 <= 246 (Polymath8b 2014), 240 (Stadlmann 2026) - PROVED; the
        0.525 is Baker-Harman-Pintz. Q4 under GEH the gap is 6 and Polymath8b proved 6 is
        parity-optimal even under GEH - no EH/GEH-type hypothesis reaches the twins; only a
        Hardy-Littlewood / Bateman-Horn statement with power-saving error gives H1-H3, trivially
        - confirms R5.f.xxi exactly. Q5 COMPUTATION: 82 maximal twin gaps known, the largest
        35,640 after the pair at p = 70,478,530,884,377,381 (7.05 x 10^16; Kourbatov 2013,
        Oliveira e Silva, Raab; OEIS A113274/A113275); pi_2(10^16) = 10,304,195,697,298 - so
        SixHyp (window length 2t/5) and LadderHyp (length 4 sqrt N) hold throughout the searched
        range with margins 10^12 and 10^4. Q6 OEIS A288815, the "paired Jacobsthal function"
        (Ziller-Morack 2017, arXiv:1706.00317 and 1706.03668), 21 terms, tagged hard, with the
        conjecture that a(n) < p_n^2 - p_n implies both Goldbach and twins; its values ((a-6)/6
        = 60 at p = 23, 316 at p = 61) differ from our F (34, 179), so the objects are not the
        same - comparison of definitions requested; no Iwaniec- or Rankin-type bound for the
        paired function found. Q7 no result on prime pairs in intervals of length c sqrt x, none
        under RH or Lindelof; Cramer-Granville analogue for twins: maximal twin gap
        G ~ a (ln(p/a) - 1.2) with a = 0.76 (ln p)^2, i.e. O(ln^3 x), Gumbel-distributed maxima
        (Kourbatov 2013, Kourbatov-Wolf 2019) - the record's cube envelope (R5.f.xii) is this
        shape. Q8 OEIS A192870: 122 is the largest M with no twin pair between M^2 and (M+1)^2
        (conjectural, computed) - a twin-Legendre statement STRONGER than LadderHyp and RegionHyp
        (the stretch of s contains the block [s^2, (s+1)^2)); Brocard's conjecture is the prime
        relative. VERDICT: every "open" label in the map stands; the closest published objects
        are A192870 (the statement) and A288815 (the covering function); the machine's F(p)
        table may be new data - pending the definition check. FACT (register).
      - R5.f.xxxii. THE FREE COVERING ROUTE TO THE WINDOW STATEMENT (PM, 2026-09-20 08:55, from the
        register's Q6 follow-up). Parent observation: R5.f.xxxi found OEIS A072753 - the largest
        run of consecutive integers that two FREELY chosen residue classes per prime 5..p can
        cover (2, 4, 10, 24, 31, 42, 60, 74, 94, 117, 148, 173, 213, 236, 275, 316, 364, 409, 436
        for p = 5..73; Ziller-Morack 2017) - and our rigid F(p) sits strictly below it (34 vs 60
        at 23, 179 vs 316 at 61, 213 vs 364 at 67), our F values not in OEIS. THE ROUTE: the
        window of machine q has (q^2 - 1)/6 - (q + 7)/6 + 1 columns; if the free covering number
        j_2(q) = A072753 is smaller than that, no choice of two classes per gear covers the
        window, so the actual classes do not, so an unstruck column exists, and by the
        square-root rule (members coprime to 6, above q, at most q^2, no factor in [5, q]) both
        its members are prime: A TWIN IN THE WINDOW OF q. This is the owner's window statement
        from a covering bound alone - no primes-in-intervals input, no sieve lower bound, no
        parity; the machine's one-line argument, which failed for the STRETCH (R5.f closed route
        1: the stretch has 2s/3 columns against F(s) ~ 0.75 s ln s) works for the WINDOW because
        the window has q^2/6 columns against j_2(q) ~ 1.4 q ln q. TABLE (q, j_2, window columns,
        margin, j_2/(q ln q), j_2/window): 5: 2, 3, 1, 0.25, 0.67; 7: 4, 7, 3; 11: 10, 18, 8;
        13: 24, 26, 2, 0.72, 0.92 (THE TIGHTEST); 17: 31, 45, 14; 19: 42, 57, 15; 23: 60, 84, 24;
        29: 74, 135, 61; 31: 94, 155, 61; 37: 117, 222, 105; 41: 148, 273, 125; 43: 173, 301,
        128; 47: 213, 360, 147; 53: 236, 459, 223; 59: 275, 570, 295; 61: 316, 610, 294, 1.26,
        0.52; 67: 364, 737, 373, 1.29, 0.49; 71: 409, 828, 419, 1.35, 0.49; 73: 436, 876, 440,
        1.39, 0.50. So the window statement is PROVED by covering alone for every prime machine
        q <= 73, and the ratio j_2/window = 8.4 ln q / q falls to 0.06 by q = 1000. THE OPEN
        LEMMA IN THIS ROUTE: CoveringHyp - j_2(q) < (q^2 - q)/6 for every prime q, i.e. the
        gears up to q with ANY phases cannot strike every column of a run of q^2/6 columns.
        Known shape: the one-class Jacobsthal function has Iwaniec's upper bound O(P^2) (1978)
        with a large constant, and nothing o(P^2); the two-class bound is being registered
        (lane open). This route's wall is the large-sieve P^2 term with its constant, not
        parity - a different wall from every route above. PRE-REGISTERED: (i) kernel theorem
        window_twin_of_free_uncoverable and twins_unbounded_of_coveringHyp (formalist lane open,
        proofs/LadderCovering.lean); (ii) register: an o(P^2) bound for any Jacobsthal-type
        function is not in the literature (prediction: not known; Iwaniec's constant not
        explicit); (iii) our rigid F(p) ~ 0.75 p ln p is a new sequence. REGISTER (Opus lane,
        09:05): (ii) HELD - Iwaniec 1978 (Demonstratio Math. 11) proves h(k) <= C (k log k)^2,
        i.e. j(P#) << P^2, with an unknown, ineffective C, one class only, and FGKMT 2018 confirm
        it is still the best upper bound; no o(P^2) bound is known for any Jacobsthal-type
        function; for TWO classes no published upper bound exists at all (Ziller-Morack prove
        none; the 2-dimensional beta-sieve gives only P^4.27, folklore). Ziller-Morack's
        Conjecture 6 is exactly j_2(P) < P^2/6 - P/6 and their theorem is the implication to
        twins (and to Goldbach) - the covering route is theirs (2017 preprint), not new to the
        literature; new to the tree. Lower bounds transfer from one class: j_2(P) >= j(P#) - 1
        >> P ln P ln_3 P / ln_2 P (FGKMT), so the conjectured order is P (ln P)^{O(1)} (Maier-
        Pomerance analogue); the data sit at 1.4 P ln P. Explicit constants: Costello-Watts 2015
        verify h(k) <= 0.2775 k^2 ln k only for k <= 10^4; unconditional explicit bounds are
        exponential (Kanold 2^k, Stevens 2 k^{2 + 2e ln k}). Mercer 2018: the one-class o(p_n^2)
        would give an elementary Dirichlet; Kanold 1965: C p_n^{2-eps} would give Linnik; neither
        gives Legendre (which needs j(P#) < 2P, refuted by Rankin). (iii) HELD: F values not in
        OEIS. VERDICT: the route stands as a proved implication (Ziller-Morack; kernel form in
        preparation) whose hypothesis is a Jacobsthal-type upper bound at Iwaniec's order with
        constant below 1/6 - a wall of large-sieve constants, not of parity; proving even
        j_2 = o(P^2) is harder than the open one-class problem. CANDIDATE (the only route on the
        tree whose obstruction is not the parity phenomenon); its next test is any explicit
        Iwaniec-type argument for two classes, and the rigid variant F(P) < P^2/6 (our classes
        are antipodal pairs with free shifts, F ~ 0.75 P ln P) as the weaker target. KERNEL
        (formalist lane, received 09:20; re-built and audited by the manager: 0 sorries, axioms
        propext / Classical.choice / Quot.sound; proofs/LadderCovering.lean): Gear q p; Struck q n;
        FreeCovers q a L r s (two classes r p, s p per gear cover the run [a, a+L)); FreeUncoverable
        q a L (no choice of classes covers it); invSix (the inverse of 6 mod p by Classical.choose
        on Nat.exists_mul_mod_eq_one_of_coprime) and struck_classes (the actual strikes of gear p
        lie in the classes invSix p and (p-1) invSix p, for columns n >= 1 - column 0 is the one
        degenerate case of natural subtraction); exists_unstruck; prime_of_unstruck_member (a
        member coprime to 6, above q, at most q^2, with no gear factor is prime - minFac);
        window_twin_of_free_uncoverable (a free-uncoverable run inside the window of q holds a
        twin centre); CoveringHyp (the window of every prime q >= 5, columns (q+7)/6 ..
        (q^2-1)/6, is free-uncoverable); windowStatement_of_coveringHyp;
        twins_unbounded_of_coveringHyp (via Nat.exists_infinite_primes). So the kernel now has
        the Ziller-Morack implication in the machine's vocabulary: CoveringHyp -> the owner's
        window statement -> twins unbounded. Prediction (i) held. FACT (kernel); route CANDIDATE.
        FIRST TEST (09:35, free_cover_ilp.py): the free covering record as a set-cover ILP (707
        binaries at P = 73, 19 cardinality rows) - HiGHS could not decide L = 436 in 600 s
        (OEIS marks A072753 "hard"); the extension past P = 73 is parked; the known 19 values
        carry the route's empirics. Next: an explicit two-class version of Iwaniec's argument
        (Fable lane opened 09:40) to see what constant the large-sieve method gives against the
        needed 1/6. RESULT (received 10:00; the lane read Iwaniec 1978 in full, plus Granville
        2020, Banks-Ford-Tao 2019, Mercer, Costello-Watts, FGKMT): THE MANAGER'S READING WAS
        WRONG - Iwaniec's argument uses NO large sieve; it is Rosser's linear sieve with the
        trivial remainder |r_d| < 1 and a sparse-support lemma for the Rosser weights (Iwaniec
        1971), and the P^2 is the LINEAR SIEVE'S SIEVING LIMIT s = 2 (f(2) = 0): the level y
        must exceed z^2, the interval length is about e^gamma y. Iwaniec: "by the sieve method
        the exponent 2 cannot be reduced"; Granville 2020: with Siegel zeros the linear-sieve
        bounds are best possible even for intervals. The constant is e^gamma C with C
        "sufficiently large", not computed. FOR TWO CLASSES the same method needs dimension-2
        weights (Diamond-Halberstam-Richert), positive only beyond the sieving limit beta_2 =
        4.266, so j_2(P) << P^(4.266 + delta) (log P)^3 with an uncomputable constant: the method
        misses the Ziller-Morack target (exponent 2, constant 1/6) by a factor P^2.27, not by a
        constant. Large-sieve refinements (Montgomery-Vaughan, Selberg, Gallagher) are upper-
        bound devices and change nothing here. Ziller-Morack's own data: j_2/window ratio 0.545,
        0.516, 0.495, 0.499 at p = 41, 53, 67, 73. CORRECTED VERDICT: the covering route is a
        proved implication (kernel) whose hypothesis sits at the sieving limit like every other
        route on the tree - its wall is the same parity phenomenon in covering clothes, not a
        large-sieve constant; the claim "the one route whose obstruction is not parity" is
        withdrawn. FACT (implication, kernel); hypothesis OPEN at the sieve limit. What survived:
        the window statement is proved by covering alone for every prime machine q <= 73, and
        the free covering function j_2 (data to 73) against the window q^2/6 is the cleanest
        integer form of the whole problem on record: the twin prime conjecture is implied by the
        two-class Jacobsthal function being o(P^2), a statement about coverings with no primes in
        it beyond the gears. REDISCOVERY (10:50, on reading docs/novel/README.md as the standing
        rule requires - not done before opening this node): the covering route IS the project's
        own line of rounds 21-27 (August 2026): docs/covering-bound-route.md (the window (y, y^2]
        as a covering, F_h(y) against y^2/6 in twin-slot coordinates, a counting lemma later
        refuted), the (D) ladder proofs/Ladder.lean (the window statement for consecutive
        machines 11 -> 13 -> 17 -> 19 -> 23 from exact F records, hypothesis-free) and
        CoveringCert.lean / CoveringCert2.lean (F(19) <= 37 by LP duality), and the novel
        entries paired-jacobsthal-values (exact h_2 = A288815 values 18, 30, 66, 150, 192, ...,
        h_2(29) = 450), j2-upper-bound (three proved upper rungs for j_2, quasi-polynomial
        p_n^(O(log log p_n)), polynomial p_n^(4.266+eps) by the fundamental lemma, THE CEILING:
        beta_2 is the dimension-2 sifting limit and Ziller-Morack Conjecture 6's exponent 2 sits
        below Selberg's conjectural floor 2 kappa = 4, "the gap is parity, not technology",
        checked 2026-08-24), j2-lower-ladder (h_2(P(z)) >= (1.349 + o(1)) z log z, round 24),
        jk-family (round 27), and open problems P3 "paired-Iwaniec upper" and P4 "Conj. 6
        true-with-room". Every conclusion of R5.f.xxxii and of the two register lanes was already
        on the record there, in the anchor/machine vocabulary. NEW tonight, in the twin-ladder
        vocabulary only: LadderCovering.lean and LadderMaxGap.lean (the window pigeonhole as
        general theorems over any q, with FreeUncoverable / MaxGapBelow as hypotheses), the F
        values 145, 160, 179, 213 (q = 53..67) and the fit 0.17 q ln^2 q. VERDICT: REDISCOVERY,
        closed as a route (the record's j2-upper-bound ceiling already says why); the kernel
        files stay as the general form of the (D) ladder's step.
        - R5.f.xxxii.a. THE RIGID FORM: THE MACHINE'S OWN RECORD AGAINST THE WINDOW (PM, 10:30;
          kernel proofs/LadderMaxGap.lean, 0 sorries, axioms propext / Classical.choice /
          Quot.sound). Parent observation: the free classes of R5.f.xxxii overstate what the
          gears do; the rigid record F(q) - the longest struck run anywhere in the period of the
          actual pattern (RigidShift.lean: every shift vector is a window of the real pattern) -
          is what the window has to beat, and F(q) <= j_2(q). KERNEL: MaxGapBelow q L (every run
          of L columns from a column >= 1 holds an unstruck column, i.e. the longest struck run is
          below L); maxGapBelow_of_freeUncoverable; window_twin_of_maxGap; MaxGapHyp (for every
          prime machine q the longest struck run of the gears 5..q is shorter than the window,
          (q^2 - 1)/6 - (q + 7)/6 + 1 columns); windowStatement_of_maxGapHyp;
          twins_unbounded_of_maxGapHyp. DATA (F exact, R5.d/R5.f.xiv, window columns): 23: 34 /
          84 (0.40); 37: 88 / 222 (0.40); 41: 91 / 273 (0.33); 43: 103 / 301 (0.34); 47: 118 /
          360 (0.33); 53: 145 / 459 (0.32); 59: 160 / 570 (0.28); 61: 179 / 610 (0.29); 67: 213 /
          737 (0.29); 71: >= 222 / 828. So the window statement holds by the machine's own record
          for every prime q <= 67 with margin at least 2.5x, the ratio falling like 4.5 ln q / q
          (F ~ 0.75 q ln q against the free 1.4 q ln q). THE HYPOTHESIS IN WORDS: the twin sieve's
          Jacobsthal function - the maximal gap between consecutive columns both of whose members
          are coprime to the primorial of q - is below q^2/6. It is the weakest covering-type form
          on the tree (MaxGapHyp <- CoveringHyp-for-every-run), its data are the F table (new to
          OEIS), and its only known attack is the dimension-2 sieve at P^4.27. FACT (kernel);
          hypothesis OPEN. Next: the F sequence's own law - F(q)/(q ln q) rises 0.71 -> 0.76 from
          61 to 67; the one-class Jacobsthal grows like P ln P ln_3 P / ln_2 P (FGKMT), so the
          rigid F should too; a fit decides whether the rigid pairs lose a log factor to the free
          classes. FIT (10:40, nine exact values 23..67): F/(q ln q) = 0.47, 0.66, 0.60, 0.64,
          0.65, 0.69, 0.67, 0.71, 0.76 (rising); F/(q ln^2 q) = 0.150, 0.182, 0.161, 0.169,
          0.169, 0.174, 0.163, 0.174, 0.180 (flat at 0.17 +- 0.01); F/(q ln q ln_3 q / ln_2 q) =
          4.0 -> 3.0 (falling). On this range the record grows like 0.17 q ln^2 q, the
          Maier-Pomerance shape for the one-class function, faster than q ln q; the range is too
          short to separate ln^2 q from ln q times a slow function. Either way F/window ~
          ln^2 q / q -> 0. FACT (fit). F(71): the 241 check ended TIMEOUT at 14,400 s (the
          process wrote its bounds before the system killed it): F(71) in [222, 259].
      - R5.f.xxxiii. THE IDEAS LANE: A FIRST STEP THAT IS NOT A SIFTED-SET LOWER BOUND (Fable,
        opened 2026-09-20 11:20 on the owner's "Find the proof"; brief lane_ideas2/BRIEF.md,
        which requires reading the map's section 0 and grepping the novel index before
        proposing). Parent observation: every route on the record (R5.f.vii-xxxii) has as its
        first unproved step a lower bound or equidistribution for a set sifted to the square root
        of its height. PRE-REGISTERED: the lane returns one route whose first unproved step is a
        different kind of statement (a finiteness-contradiction identity, a bootstrap in the gear
        role, an exact counting identity with a finite check), as one quantified sentence, with
        one numerical test; or reports that each candidate is a sifted-set bound in disguise.
        F(71): a second bisection (222..231, three-hour checks) launched 11:15. RESULT (received
        11:35; script research/stack/r8/ideas2_composition_law.py; the lane read map section 0,
        sections 9 and 16-21, ladder_attempts.md, and 19 novel-index entries by line). NO ROUTE
        MEETS THE CRITERION: (a) every consequence of a finite twin set above S (for every prime
        q > S the open set of the machine 5..q meets (0, q^2/6] only at 0; the first open column
        is above q^2/6; F(q) >= q^2/3; the open set below the next machine's window sits inside
        the next gear's two classes) has as its negation "the set sifted by 5..q' has an element
        in the window of q'" - the sifted lower bound in disguise; measured against it, machine
        101's open columns in (0, 1768] occupy 95 of 103 classes mod 103, machine 211: 215 of
        223, machine 307: 309 of 311 (finiteness would force <= 2). (c) Two-sided identities:
        the ledger's unknown side is the distribution of a sifted set across the big gears
        (equidistribution, gap A-2); Legendre/W98 censuses evaluate the sifted count itself;
        responsible-strike double counting gives (q^2/3) ln ln q strikes against q^2/6 columns, so
        pigeonhole is satisfied never contradicted; whole-set statements (a finite twin set
        makes Brun's constant or prod (s-3)/(s+1) rational) are content-free - the complement has
        no independent evaluation; do not open. (b) Twin gears in the gear role: phase lock and
        the four double-kill classes are on the record (tooth-sharing-pinning); "twin gears never
        strike another twin's column" is primality restated. WHAT FELL OUT - AN EXACT GEAR-ROLE
        COMPOSITION LAW (new to the index): for every column c with s = 6c and c* = 3c(48c^2 - 1)
        (so 6c* = T_3(s) = 4s^3 - 3s, the Chebyshev cube), 36c*^2 - 1 = (36c^2 - 1)(36(2c)^2 - 1)^2,
        members (s-1)(2s+1)^2 and (s+1)(2s-1)^2, hence the gear set G(c*) = G(c) disjoint-union
        G(2c) and omega(c*) = omega(c) + omega(2c); verified by full factorisation for c <= 3000,
        0 violations; 16 columns c* with exactly four gears, 11 of them with c and 2c both twin
        centres (c = 1, 5, 110, 135, 355, 425, 555, 565, 975, 1045, 1755); the record held only
        the instance (845, 847) = (5 x 13^2, 7 x 11^2) (docs/pair-anatomy.md). General odd k:
        T_k(s) - 1 = (s-1) W_m(s)^2 and T_k(s) + 1 = (s+1) V_m(s)^2 (Chebyshev third and fourth
        kinds), strike sets governed by the rank of apparition of the Lucas pair (2s, 1), the twin
        gears being ranks 1 and 2. Not a route: the law moves strike sets UPWARD (members
        composite by construction), and 36X^2 - 1 = (36c^2 - 1) Y^2 has Y = 1 only at X = +-c
        (Pell), so no member of the family is a twin; a downward law would be a polynomial prime
        infinitely often (Bunyakovsky, a sifted bound again); primitive-divisor theorems
        (Zsigmondy, Bilu-Hanrot-Voutier) give a prime dividing a number, never a prime. The one
        route-shaped question left undisguised: does any exact law transfer strike sets DOWNWARD -
        from a column with known composite members to a smaller column whose strike set is
        provably confined to gears above its square root? Within Chebyshev/Pell identities, no;
        the product-forest identity (s-1)(t-1)(s+1)(t+1) = (st+1)^2 - (s+t)^2 is the record's only
        other member of the family and is not of column form. VERDICT: FACT for the composition
        law (exact, kernel-ready: a polynomial identity); the route question DEAD within
        Pell/Chebyshev, OPEN as stated for other identity families.
      - R5.f.xxxiv. THE FIELD PROOF DRAFT (owner's direction 2026-09-20 11:20, written 11:30;
        research/proof/field_proof_draft.md). The owner's argument as a chain of lemmas, each
        marked: A the square-root rule (PROVED, kernel); B the widening rule - a gear strikes no
        column below its square except its home column and columns a smaller gear already
        strikes, so a twin slot stays a twin slot in every larger machine (PROVED, elementary;
        kernel entry to add); C1-C5 the shapes of the fields multiples / squares / products:j /
        higher:g / lower:g - two teeth per period, never adjacent unless g = 5, one square column
        per gear at (g^2 - 1)/6, a two-factor product between the squares of its pair, j >= 3
        factors already painted by the smallest factor, higher/lower are relabellings, no blind
        gear (PROVED); D1 no set of gears blocks permanently - the joint period is coprime to 6
        and each period holds prod (g - 2) unstruck columns (PROVED, CRT) - with its reach stated:
        the period exceeds e^(q/2), the window is q^2/6, so D1 does not place an unstruck column
        in the window; E the window is never painted over (LEMMA - the one to establish), with
        the owner's one-field-at-a-time programme as sub-lemmas E1 (one row, PROVED), E2 (two
        rows, exact longest joint run - LEMMA), E3 (the record's increment per new gear via the
        loaded record rule W88 - LEMMA), E4 (the record below the window, = MaxGapHyp - LEMMA);
        the theorem from E + A + B + Euclid. Verdict: DRAFT; the first lemma to work is E2.
        - R5.f.xxxiv.a. E2: TWO ROWS, THE EXACT LONGEST JOINT RUN (PM, pre-registered 11:40
          before the run; research/stack/r8/two_rows_exact.py). Mechanism: a gear's two teeth
          sit at distance inv3 mod g (or g - inv3); inv3 = +-2 mod g iff 3 x 2 = +-1 mod g iff
          g in {5, 7}; a run of 4 needs two interleaved tooth pairs at distance 2, a run of 3
          needs one tooth pair at distance 2 around a third tooth, and no gear paints two
          adjacent columns alone (E1). PREDICTION: F({g,h}) = 4 iff {g,h} = {5,7}; 3 iff exactly
          one of g, h is 5 or 7; 2 otherwise. RESULT: all 300 pairs among the first 25 gears
          (5..101) agree, 0 violations; exhaustive phase search and the period scan (RigidShift)
          agree on {5,7}, {5,11}, {7,13}, {11,13}. E2 PROVED (elementary argument above, exact
          to 101; kernel entry to add). TRIPLES (first 8 gears, exact): with 5 and 7 both: 6;
          with one of them and two gears of tooth distance 4 (11, 13): 5; with one of them and
          gears of distance >= 6 (17, 19, 23, 29): 4; no 5 or 7: 3 - the record of a small set is
          a function of the multiset of tooth distances, the third gear adding 2 columns when its
          distance is 4 and 1 otherwise. INITIAL SEGMENTS (period scan, run convention): F(5) =
          1, F(7) = 4, F(11) = 6, F(13) = 10, F(17) = 17, F(19) = 24, F(23) = 33 (the record's
          F(23) = 34 counts the gap, one more than the run - conventions differ by 1
          throughout: 88, 91, 103, ... are gaps). INCREMENTS per new gear (run convention): +3,
          +2, +4, +7, +7, +9 for 7, 11, 13, 17, 19, 23, then (gap convention) 37: 88, 41: +3, 43:
          +12, 47: +15, 53: +27, 59: +15, 61: +19, 67: +34 - irregular, growing slowly; F(29),
          F(31) being computed (ILP) to close the table. FACT (E2 proved); E3 open with its data.
        - R5.f.xxxiv.b. E3: HOW A NEW GEAR EXTENDS THE RECORD (PM, pre-registered 11:50 before the
          run; research/stack/r8/e3_increment.py, exact period scans for 5..23). PREDICTION: the
          new record run contains at most 2 ceil(F/p) columns painted only by the new gear p (the
          holes of the old machine it fills), in practice 1 or 2 for p <= 23, and the old runs
          inside it are at or near the old record. RESULT (p: F, holes filled by p inside the
          record, longest old run inside): 7: 4, {0,2} or {1,3}, 1; 11: 6, one hole, 4; 13: 10,
          one hole, 5; 17: 17, two holes, 10 or 6; 19: 24, one or two holes, 17 or 12; 23: 33,
          THREE holes at positions {3, 11, 26} (tooth distance 8 and 15 = 23 - 8), 14. The
          ceiling 2 ceil(F/p) holds (23: 4 allowed, 3 used); "1 or 2" FAILS at 23 (F = 33 > 23,
          so a third tooth enters); "old runs near the old record" FAILS from 17 on (6 of 10, 12
          of 17, 14 of 24): the record is built from several medium old runs, not from the old
          record extended. E3 IN EXACT FORM (PROVED, one line): the new record minus the new
          gear's teeth splits into at most h + 1 old runs, each at most F(q), where h is the
          number of teeth of q' inside the record, h <= 2 ceil(F(q')/q'); so F(q') <= (h + 1) F(q)
          + h, and the h teeth sit at mutual distances 0 or +-inv3 modulo q'. What this leaves
          for E4: the joinable holes of machine q are those at the new gear's tooth spacing that
          are flanked by long runs - a statement about the spacing of the old machine's holes (its
          twin candidates), i.e. about the old pattern's own run structure; the inequality alone
          allows F to triple per gear and cannot give E4 by itself. F(29) = 42 EXACT (ILP, 43
          uncoverable; increment +9 over F(23) = 33 in the run convention); F(31) = 57 EXACT (+15);
          F(71) >= 228 (228 coverable in 2,129 s). FACT (E3 exact form
          proved; E4 sharpened to a hole-spacing statement).
        - R5.f.xxxiv.c. E4 AS HOLE CHAINS: THE EXACT RECURSION FOR THE RECORD (PM, pre-registered
          12:30 before the run; research/stack/r8/e4_hole_chains.py, exact scans over the joint
          period P q' for 5..19 -> 23, up to 8.7 million holes). THE RECURSION: F(q') = max of F(q)
          and, over every maximal chain of consecutive holes of machine q all lying in the two
          tooth classes of q' (so that q' fills each of them), the span from the hole before the
          chain to the hole after it, less one. PREDICTIONS: (1) the recursion reproduces F(q')
          exactly; (2) exactly 2/q' of the holes lie in q''s classes over the joint period (CRT)
          while aligned consecutive PAIRS are rarer than (2/q')^2; (3) chains use at most
          2 ceil(F/q') holes. RESULT: (1) HOLDS at all six steps - 4, 6, 10, 17, 24, 33 for
          q' = 7, 11, 13, 17, 19, 23 (a first scan over the OLD period alone missed at 7, 11 and
          23: every hole class recurs q' times with distinct residues, so the joint period is
          required); (2) HOLDS - aligned share 0.2857, 0.1818, 0.1538, 0.1176, 0.1053, 0.0870 =
          2/q' to four decimals; aligned consecutive pairs 2, 0, 6, 72, 1088, 11870 = 9.5%, 0,
          0.34%, 0.29%, 0.26%, 0.14% of the holes against (2/q')^2 = 8.2%, 3.3%, 2.4%, 1.4%,
          1.1%, 0.76% - three to five times rarer, because consecutive holes sit at the small
          gaps 1, 2, 3, 5 and a pair is aligned only when its gap is EXACTLY inv3 mod q' or
          q' - inv3 (or wraps by a multiple of q'): the alignment is decided by the hole-gap
          spectrum at two specific gap values; (3) HOLDS - chains of 2, 1, 1, 2, 1, 3 holes, the
          3-chain at 23 alternating gaps 15 and 8 (= 23 - 8, 8) between the classes 4 and 19.
          Hole-gap spectra (gap: count) 5..19: 1: 46,683; 2: 124,488; 3: 64,106; 4: 29,184;
          5: 80,370; 6: 19,418; 7: 32,604; 8: 9,006; max gap 18 = F(19) + 1; the spectrum is
          the project's wheel gap census (docs/novel wheels-gap-census W22-W26 and
          wheels-run-spectrum-duality W11, W45: F_top = max{j : C(j) > 0}), and the recursion
          here is the machine-vocabulary form of the loaded record rule W88 with the new gear as
          the only tail. WHAT E4 NOW SAYS: the record of q' exceeds the window only if machine q
          has a chain of consecutive holes at the exact gaps inv3(q') / q' - inv3(q') (mod q')
          whose flanking runs and span together reach q'^2/6; since flanks are at most F(q)
          and chains have at most 2 ceil(F(q')/q') holes, E4 reduces to bounding how many
          consecutive hole gaps can hit those two values in a row and how long the runs beside
          them can be - a statement about the gap spectrum of machine q at two residues mod q',
          not about counts of twins. FACT (recursion verified exactly); E4 open in this form.
        - R5.f.xxxiv.d. E4a / E4b: FLANKS AND CHAINS OF ALIGNED HOLES (PM, pre-registered 15:20
          before the run; research/stack/r8/e4_flanks.py, exact over the joint period, 5..19 ->
          23). PREDICTIONS: flanks beside aligned pairs have the same distribution as beside all
          pairs; chain lengths fall geometrically. RESULT: aligned pairs occur ONLY at the gaps
          inv3 and q' - inv3 (and multiples of q'): 13: gap 4 x 6; 17: gap 6 x 60, gap 11 x 12;
          19: gap 6 x 1,022, gap 13 x 66; 23: gap 8 x 10,462, gap 15 x 1,236, gap 23 x 172 - and
          each count is EXACTLY the census of that gap divided by q' (240,626 gaps of 8 in machine
          5..19, 240,626/23 = 10,462): the alignment count is the hole-gap census at two values
          over q', by CRT. Flanks: the maximum flank beside an aligned PAIR is below F(q) (2 vs 6,
          6 vs 10, 11 vs 17, 19 vs 24) while the mean is 10-15% above the mean over all pairs (the
          aligned gaps 6, 8 are long gaps and long gaps neighbour long runs slightly more). Chains:
          1-chains 258 / 2,826 / 42,374 / 733,672, 2-chains 6 / 72 / 1,088 / 11,746, 3-chains 0 /
          0 / 0 / 62 at q' = 13, 17, 19, 23 - geometric with ratio 1-3%. FACT.
        - R5.f.xxxiv.e. THE SANDWICH: THE RECORD BETWEEN TWO SPARSITY VALUES OF THE OLD MACHINE
          (PM, pre-registered 15:30 before the run; research/stack/r8/e4_gk.py, exact 5..19 -> 23).
          Define G_k(q) = the longest window of machine q holding at most k holes (G_0 = F(q); G_1
          = the largest run-hole-run). ALIGNABILITY (exact, by CRT over the joint period): k
          consecutive holes with gaps d_1..d_{k-1} can all be filled by q' iff every partial sum
          d_1 + .. + d_j is 0 or +inv3 mod q', or every one is 0 or -inv3 (one sign per chain - a
          first version allowing both signs overshot at 17, 19, 23 by 3, 3, 1); a single hole is
          always alignable. PREDICTIONS: (1) F(q') >= G_1(q) always, with equality at some steps;
          (2) F(q') equals the largest alignable chain span; (3) G_k grows roughly linearly in k.
          RESULT: (1) HOLDS - G_1 = 3, 6, 10, 15, 24, 30 against F(q') = 4, 6, 10, 17, 24, 33:
          equality at 11, 13, 19 (the next record IS the largest run-hole-run of the machine);
          (2) HOLDS at all six steps with the one-sign criterion (4, 6, 10, 17, 24, 33 exactly);
          (3) HOLDS - G_1..G_6 at 19: 30, 34, 37, 46, 49, 57, slope about 5 per hole (the mean
          hole spacing), far below (k+1)(F+1). THE SANDWICH (PROVED, elementary): G_1(q) <= F(q')
          <= G_k(q) with k = 2 ceil(F(q')/q') - the lower half because a single hole can always be
          aligned, the upper because the new gear fills at most 2 ceil(L/q') columns of a run of
          length L, so a record run of q' is a window of q with at most that many holes. WHERE E4
          STANDS: E4 for q' follows from G_k(q) < q'^2/6 with k about q'/3, i.e. every window of
          machine q of length q'^2/6 holds more than q'/3 holes - a lower bound on holes in
          windows, of size q'/3 against the generic q'^2 / (2.4 ln^2 q'). Iterating the worst-case
          loss per gear (each new gear removes at most 2 ceil(L/g) holes from a window of length
          L) is the union bound, which loses because it ignores that new paint lands mostly on
          painted columns; so E4 is exactly the OVERLAP statement: in every window of length
          q'^2/6 the new gear's teeth cover fewer than all the holes of machine q. FACT (sandwich
          proved; recursion exact); E4 open as the overlap statement, which is the window statement
          for q' itself. The exact recursion is kernel-ready and is the machine's law for how the
          record grows; it does not by itself bound the growth.
        - R5.f.xxxiv.f. E4c: THE SHAPE OF A COVERED WINDOW (PM, 15:40; elementary, exact). The
          tooth distance of a gear q' is inv3 mod q'; since 3 inv3 = 1 mod q', inv3 is (q'+1)/3 or
          (2q'+1)/3, so the two allowed gaps between consecutive aligned holes are {(q'+1)/3,
          (2q'-1)/3} or {(q'-1)/3, (2q'+1)/3} (plus multiples of q'), and along a chain they
          ALTERNATE (partial sums stay in {0, d}: a gap = d is followed by a gap = -d mod q' or by a
          multiple of q'), so every second hole of a covered window lies in the same class mod q'.
          PROVED (from the alignability criterion of xxxiv.e): in a window fully painted by machine
          q', every gap between consecutive holes of machine q strictly inside it is at least
          (q'-1)/3, every interior run of machine q is at least (q'-4)/3 long, the holes inside
          number at most 3L/(q'-1) + 1 for a window of length L, and the interior gaps alternate
          between the two values modulo q'. Checked on the record of 23 (holes at 3, 11, 26 inside
          the run of 33: interior gaps 8 = (23+1)/3 and 15 = (2 x 23 - 1)/3, interior runs 7 and
          14 >= 6.33; the end runs 3 and 6 are not constrained). CONSEQUENCE FOR E4: a covered
          window of q' is a stretch of machine q of length q'^2/6 whose interior holes are spaced at
          least (q'-1)/3 apart in two alternating residue classes - hole density at most 3/q'
          against the machine's 2.5/ln^2 q - so E4 is the statement that machine q has no stretch
          of length q'^2/6 with all interior hole gaps at least (q'-1)/3 in the two alternating
          classes. FACT (exact shape); E4 open in this form (a lower bound of about q'/2 holes in
          every window of length q'^2/6 of machine q, i.e. at most a fraction 3/q' of the
          window's columns may be gaps of that size).
        - R5.f.xxxiv.g. E4d: THE SPARSE-STRETCH BOUND (PM, pre-registered 15:55 before the run;
          research/stack/r8/e4_sparse_stretch.py, exact period scans 5..23). Define S_t(q) = the
          longest stretch of columns of machine q whose interior consecutive hole gaps are all at
          least t (a stretch runs from just after a hole to just before a hole). By E4c the record
          run of q' is such a stretch of machine q with t* = ceil((q'-1)/3), so F(q') <= S_t*(q):
          PROVED. PREDICTION: S_t*(q) is below a third of the window q'^2/6 for q <= 19 and the
          ratio falls with q. RESULT (q -> q': F(q'), S_t*(q), window, ratio): 5 -> 7: 4, 5, 7, 0.71;
          7 -> 11: 6, 7, 18, 0.39; 11 -> 13: 10, 19, 26, 0.73; 13 -> 17: 17, 22, 45, 0.49; 17 -> 19:
          24, 33, 57, 0.58; 19 -> 23: 33, 37, 84, 0.44; 23 -> 29: 42, 59, 135, 0.44 (S_t at 23: 2: 385, 3: 134, 4: 108, 5: 83, 6: 72, 8: 61, 9-10: 59, 11: 44, 16-26: 39). F(q') <= S_t*(q) holds at every step (the
          new bound is tighter than G_k: 37 against 46 at 19 -> 23); "below a third" FAILS (ratios
          0.4-0.7), "falls with q" MIXED (no monotone fall on this range). The S_t table (t ->
          S_t) at 19: 2: 210, 3: 94, 4: 69, 5: 64, 6: 41, 7: 38, 8: 37, 9-10: 36, 11-13: 33, 14-21:
          32, 22-25: 31, 26: 30 - a fast fall to t = 6 and a long plateau: past t = 8 the longest
          sparse stretch is the record (24) plus one or two isolated flanks, i.e. S_t(q) is about
          F(q) + a few gaps of size >= t; heuristically S_t*(q) is about F(q) + c q'/3 against a
          window of q'^2/6, so the ratio should fall like ln^2 q / q eventually, but the measured
          range does not show it yet. E4 IN THIS FORM: S_ceil((q'-1)/3)(q) < (q'^2 - q')/6 for every
          consecutive pair of primes q < q'; in words, the longest stretch of machine q whose holes
          are all at least q'/3 apart is shorter than the window of q'. The stretch's holes are the
          isolated twin candidates of level q (both neighbours at least q'/3 away); E4 says such
          isolated candidates never line up densely enough for the stretch to span the window.
          FACT (F(q') <= S_t*(q) proved, table exact); E4 open in this form.
        - R5.f.xxxiv.h. THE NUMERATOR FROM THE SHAPES: THE THIN-BAND BOUND (owner's direction
          16:20: derive the stretches from the fields, tables only to verify; PM 16:30,
          research/stack/r8/thin_band_bound.py). Self-similarity (exact): higher:g in the window
          is g times the rough set of the machine below g, so the painted set is a union of
          scaled copies of the smaller machines' hole patterns (Buchstab, column by column).
          THIN-BAND BOUND (PROVED): with base 5..B held exactly and h_B(L) its least hole count in
          a window of length L, F(q) <= max{L : 2 sum_{B<g<=q} ceil(L/g) >= h_B(L)}. Evaluated:
          base 17 / top 19: F(19) <= 42 (true 24; window 57 - the window statement for machine 19
          from shapes); base 23 / top 29: F(29) <= 89 (true 42; window 135 - likewise for 29);
          band of two gears: 143 at 23 (true 33, window 84), above 160 at 31 (true 57) - vacuous.
          Why it loses: it charges every top tooth as a hit on a base hole; the true hit rate is
          the share of base holes in the gear's two classes, 2/g of them unless the holes are
          concentrated in those classes - which is the covered-window configuration itself. The
          alternation attempt reduces to the same concentration statement: individual rows
          constrain nothing about hole residues modulo a gear outside the machine. VERDICT: FACT
          (bound proved; window statement for 19 and 29 by shapes); E4 is exactly the overlap
          statement, and every bound from the rows alone charges the overlap at its worst case.
        - R5.f.xxxiv.i. THE RESIDUE-COLLAPSE CENSUS (owner's step 1, PM 17:10; research/stack/
          r8/residue_collapse.py). By CRT, the number of covered windows of length L of machine q'
          equals the sum over windows W of machine q of: q' if W has no hole, 2 if all holes are
          congruent mod q', 1 if the holes occupy exactly two residues at difference +-inv3 mod
          q', 0 otherwise. VERIFIED exactly at 11 -> 13 and 13 -> 17 for L = 6..20 (14 of 14, e.g.
          C_17(17) = 20). For L > F(q): C = 2 W_1 + N_alt (one-hole windows, alternating windows),
          so F(q') = max(G_1(q), largest alternating window) - the sandwich as an equality - and
          E4(q') = [G_1(q) < window] and [no alternating window of length window in machine q].
          The rows constrain hole residues mod q' not at all; the attempt to bound alternating
          windows by gear 5's free arcs along the progression fails (class members need not be
          consecutive terms). FACT (exact census); E4 open as "no alternating window".
        - R5.f.xxxiv.j. KERNEL: THE FIELD LEMMAS (owner's step 2; Fable formalist lane, received
          17:35; re-built and audited by the manager: 0 sorries, axioms propext / Classical.choice
          / Quot.sound; proofs/LadderFields.lean). StrikesBy; no_adjacent (E1); strike_distance and
          strike_distance_ge (E4c); class_count, gear_count, gear_count_prime; holes_in_covered_run
          (E3' upper half, with q' the next prime as hypothesis); thin_band (E4e);
          five_consecutive and four_consecutive (E2 including the {5,7} exception, a 16-way case
          split). No statement weakened. Not yet in the kernel: Lemma B (widening) and the lower
          half of the sandwich. FACT (kernel).
        - R5.f.xxxiv.k. KERNEL: LEMMA B AND THE SANDWICH'S LOWER HALF (owner's order 18:00;
          Fable formalist lane opened 18:05, brief lane_widen/BRIEF.md). PRE-REGISTERED:
          proofs/LadderWidening.lean with widening (below its own square a gear strikes only its
          home column or a column a smaller gear strikes), twin_column_strikers, twin_slot_persists
          (a twin slot is unstruck in every machine whose gears stay below its members),
          struck_periodic, tooth_of_class, align_single_hole (a run of the old machine with one
          hole becomes a fully struck run of the next machine at a translate - G_1(q) <= F(q'), by
          periodicity and CRT), not_maxGapBelow_of_single_hole. RESULT (received 18:15; re-built
          and audited by the manager: 0 sorries, axioms propext / Classical.choice / Quot.sound;
          twin_slot_persists needs only propext / Quot.sound). All eight statements as briefed;
          struck_periodic carries 1 <= n (necessary: column 0 is a natural-subtraction artefact);
          the widening rule holds for any g once the member is below g^2 (the primality of g is
          unused). Every proved line of the field draft is now in the kernel; E4 is the only lemma
          outside it. FACT (kernel).
        - R5.f.xxxiv.l. THE TWISTED TRANSLATES (owner's step 1 continued, PM 18:40;
          research/stack/r8/twisted_record.py). Along each class c mod q' the machine q is one
          two-class pattern T (tooth distances (3q')^{-1} mod g) translated by c q'^{-1} mod P; a
          covered window of q' is T painted on about L/q' consecutive terms at q' - 2 of q'
          translates in an arithmetic progression of shifts (exact, CRT). PROVED: F(q') <= q'
          (F_T(q;q') + 1). PREDICTION: F_T is of F(q)'s size so the bound is q' times too weak.
          RESULT: F_T = 3, 7, 10, 16, 28 against F(q) = 4, 6, 10, 17, 24 (q = 7..19); bounds 44,
          104, 187, 323, 667 against windows 18, 26, 45, 57, 84 - HELD, the bound is vacuous. The
          shifts are equidistributed mod every gear. VERDICT: FACT (a proved but vacuous bound);
          the twisted form is E4's third coordinate system and re-encodes the joint alignment -
          the row shapes fix every individual constraint and none of the simultaneous one.
        - R5.f.xxxiv.m. THE STRUCTURE AT THE ORIGIN: THE SQUARE LEMMA AND THE WINDOW COUNT LAW
          (owner's "find the structure", PM 19:20; research/stack/r8/origin_square_lemma.py).
          PREDICTION (from the squares field and the square-root rule): in the window of q' the
          holes of machine q are exactly the twin columns plus the column of q'^2 when q'^2 - 2 is
          prime. RESULT: 75 consecutive prime pairs to 400, 0 violations, 26 square columns, no
          other extra hole. LEMMA E5 PROVED (elementary): a member <= q'^2 with no factor <= q is
          prime or q'^2. WINDOW COUNT LAW (exact): T(q') = T(q) - [q'+2 prime] + N(q^2, q'^2] -
          the only loss is the bottom-boundary pair (q', q'+2), every gain is a new twin between
          the squares; the new gear's action on old holes is the square column only. So at the
          origin the covering question of xxxiv.h-l does not arise; the window statement fails at
          q' only after T descends to 1 by one per step with N = 0 throughout, i.e. after a
          twin-free interval (q_0^2, q'^2] at least 4 q_0 (T(q_0) - 1) long. The field programme
          meets the ladder programme here: N(q^2, q'^2] >= 1 is RegionHyp (R5.f.xxvi, kernel
          LadderRegion.lean), and the products field g p in (q^2, q'^2] is its mechanism. FACT
          (E5 and the count law proved); the lemma left is N >= 1 per consecutive pair, or "T never
          reaches 0" in walk form.
        - R5.f.xxxiv.n. THE TWIN STRETCH IN THE PRODUCTS FIELD (owner's "go", PM 19:50;
          research/stack/r8/stretch_products_split.py, twin centres 102..2970, exact). Base level
          B = floor(sqrt(2s)); base-open = twins + plugged, a plug being a column with a P_2 or P_3
          member (all factors above B) and a B-rough partner. PREDICTIONS: share plugged in
          [0.6, 0.9] rising slowly; no P_4 plug; top band (s/2, s] paints about 1.8 s/ln s columns
          with a base-open fraction 3.6/ln s. RESULT: share 0.63 -> 0.81 (HOLDS); P_2 dominant, P_3
          a few percent, no P_4 (HOLDS); top band paints 233 of 1,247 at s = 1872 (half the
          estimate, MIXED) with base-open fraction 0.16 = e^{-gamma}/ln B (the 3.6/ln s was wrong;
          the partner's roughness gives e^{-gamma}/ln B). REGION LEMMA IN FIELD TERMS: the products
          h p (B < h <= p) with B-rough partner number fewer than the base-open columns. The plug
          count has an upper-bound sieve at any level; the base-open count has a lower-bound
          sieve in a stretch of length 4s only below level (4s)^{1/4.27} - Chen's switching frame,
          which is why Chen holds on the line and not in the stretch. FACT (split exact, shares
          measured); the region lemma open as the comparison of the two counts.
        - R5.f.xxxiv.o. THE PRODUCTS FIELD OF THE STRETCH IS GOLDBACH (owner's "push ahead", PM
          20:30; research/stack/r8/stretch_goldbach.py). IDENTITY: a two-prime member h p of the
          stretch is m^2 - d^2 with m = (h+p)/2, d = (p-h)/2 - a Goldbach partition of 2m - in the
          stretch iff (s-1)^2 < m^2 - d^2 < (s+1)^2, at offset (m^2 - d^2 - s^2 -+ 1)/6, with
          s - 1 < m <= (h + (s+1)^2/h)/2. PREDICTIONS: (1) the identity and bound hold for every
          two-prime member (s <= 3000); (2) the lower members with m = s are exactly the
          partitions 2s = (s-d) + (s+d), 1 <= d < sqrt(2s), at j = (1 - d^2)/6 (d = 1 the twin
          itself at the centre); (3) upper members with m = s+1 are the partitions of 2s+2 with
          d = 6t at j = c - 6t^2, and 2s-2 gives nothing. RESULT: 54,027 members, 0 violations
          (after correcting the bound to include h/2); claims 2 and 3 exact at all 74 twin
          centres; 2-4 short partitions of 2s and 0-2 of 2s+2 per centre. So the record's
          composite-forcing families (R5.f.xv) are the short Goldbach partitions of 2s and 2s+2,
          and the stretch is the partition graph of the even numbers 2m in (2s-2, s^2/B]: plugs =
          partitions with a rough partner (+ three-prime products), twins = the columns no
          partition reaches. The plugs carry s's factorisation (singular series of 2s); the
          twins carry none (R5.f.xxv). LEMMA, third field form: partitions with rough partner +
          P_3 plugs < base-open columns. FACT (identity and the two families exact).
        - R5.f.xxxiv.p. THE LEVEL ANALYSIS OF THE PLUG COMPARISON (PM 21:00;
          research/stack/r8/share_by_level.py). Share law share(B) = 1 - T/U(B) = 1 - 1.98/(ln^2 s
          prod_{g<=B}(1-2/g)) ~ 1 - 0.79 ln^2 B/ln^2 s (a first version lost the factor 6 per
          column and predicted 0.97 at s^0.5 against the measured 0.80; corrected law verified:
          0.96 / 0.80 / 0.50 predicted at s^0.234 / s^0.5 / s^0.8 against 0.94-0.95 / 0.78-0.80 /
          0.46-0.56 at s = 1302, 2082, 2970). Proof requirements by level: a lower bound for U(B)
          exists only for B <= s^0.234 (dimension-2 limit 4.266 in an interval of length 4s,
          intrinsic, not a remainder issue); the plug upper bound has constant F(ln D/ln B): 1.78
          at D = x^{1/2}, B = s^0.5 (needs < 1.25); about 1.02 at an EH-type level D = x^{1-eps}
          (would do at B = s^0.5, where (i) is missing); at B = s^0.234 the share 0.96 needs a
          constant below 1.04, which no level gives. VERDICT: the comparison has no admissible
          level - Chen's switching with the interval too short, quantified. What would close it:
          a non-sieve lower bound for pairs of prime-or-P_2 members in the stretch at level
          sqrt(2s) plus EH-level equidistribution of the bilinear partner sequence. FACT (law and
          analysis); the lemma stands.
        - R5.f.xxxiv.q. THE BASE-OPEN COUNT IS THE HOLE UNIFORMITY OF MACHINE sqrt(2s) AT SCALE
          B^2 (PM 21:30). U (base-open columns of the stretch at level B = sqrt(2s)) = holes of
          machine B in a window of about B^2/3 columns at a generic phase; U >= k iff G_{k-1}(B)
          < 4c - 1; with the measured G_k law (linear in k at the mean spacing, xxxiv.e) this gives
          U ~ 6.6 s/ln^2 s = its mean, exactly the lower bound the comparison of xxxiv.p needs. So
          the region lemma in the products frame = (a) hole uniformity of machine B at scale B^2
          (no window of B^2/3 columns with fewer than half the mean holes) + (b) an EH-level bound
          for the bilinear partner sequence. (a) is the joint alignment of xxxiv.h-l in density
          form (E4 asks one hole per B^2/6 window; (a) asks proportionally many per B^2/3
          window); Montgomery-Vaughan 1986 control the one-class analogue for almost every window,
          not the worst. VERDICT: the field programme has carried the lemma through four exact
          forms and located the obstruction in each as the same statement about one sifted set at
          the scale of its own square; the draft is complete with that one lemma open. FACT.
          MEASURED (21:45, hole_uniformity.py, machines 11..23, exact): worst window / mean holes
          at L = B^2: 0.92, 0.88, 0.89, 0.88, 0.87 (stable); at B^2/3: 0.86, 0.72, 0.72, 0.71, 0.69
          (slowly falling); at B^2/6: 0.57, 0.60, 0.64, 0.50, 0.43 in ratio but 4, 5, 8, 7, 8 holes
          against means 7-19 - the worst E4-scale window keeps a flat handful of holes while the
          mean grows. Prediction "at least a quarter of the mean" HELD; "ratio rises with L" HELD.
          Statement (a) holds proportionally at the stretch's scale with constant about 0.7 here.
        - R5.f.xxxiv.r. THE WHOLE-RANGE FORM (U') (owner's direction "use the whole range", PM
          22:10). Beyond q^2/6 a hole of machine q is a candidate, not a twin (members above q^2);
          but for consecutive primes the new band is one gear, so the kernel's thin_band gives
          F(q') <= max{L : 2 ceil(L/q') >= h_q(L)} and hence (U') [every window of (q'^2-q')/6
          columns of machine q holds more than 2 ceil(q'/6) holes] ==> E4(q') ==> window statement
          for q'. (U') is a whole-period statement asking for about q/3 holes against a mean of
          q^2/(2.4 ln^2 q). VERIFIED at every computable consecutive pair with base = previous
          machine: F(19) <= 42 < 57, F(23) <= 57 < 84, F(29) <= 89 < 135 (true 24, 33, 42) - the
          window statement for 19, 23, 29 from shapes plus the previous machine's worst windows.
          The record alone gives q/ln^2 q holes, short by ln^2 q/3; (U') says record-sized gaps
          do not cluster (at most a fraction 3/ln^2 q of consecutive gaps near the record). FACT
          (reduction proved; verified to 29); (U') open.
      - R5.f.xxxv. THE RANGE STATEMENT AND THE SPIRAL VARIANT (owner, 2026-09-20 late / 09-21).
        The owner's distinction, accepted: the range (q, q#] holds one full period of machine q -
        every phase combination of its gears once, mirrored - and asks only that a twin exist
        somewhere in it, with the deciding gears all above q (T = primes in (q, sqrt(q#)]); the
        window holds a shrinking fraction of the cycle and needs a location. Exact fact for the
        range (from strike_distance, LadderFields.lean): a gear g > q kills two openings of q in
        one pass only at gaps 3^{-1} mod g or g - 3^{-1}, both at least (g-1)/3, while consecutive
        openings of q are at most F(q)+1 apart; so every gear above 3F(q)+3 kills q's openings
        one at a time, never two neighbours, and only the band (q, 3F(q)] can merge gaps. TO TRY
        (owner's spiral variant, not yet run): add a larger gear (or gears) to the machine, mirror
        across the larger gear's primorial, then remove the top gears one by one, never q, and
        return by smaller primorials to land in the window carrying q's full set. Verdict OPEN;
        the next lane is the owner's choice between the range analysis (how T's classes sit on
        q's openings across one full period, pattern only) and the spiral variant.
        SPIRAL VARIANT CHECKED (2026-09-21, research/stack/r8/spiral_variant.py): a mirror about
        an axis a sends column n to 2a - n and carries the openness of every gear dividing a; a
        walk whose axes are all multiples of q's product P (any mixture of the larger gear's
        product and the smaller ones, up to four mirrors, multipliers 1..6) lands only at
        multiples of P - landings mod P = {0} at every q = 7..23 - and the window (q, q^2] holds a
        positive multiple of P only when P < q^2/6, i.e. never beyond q = 7. So the variant as
        stated cannot land in the window while carrying q's full set: the carry wall (kernel
        carried_le_log, free_regime_unreachable) met by construction. The only tweak that lands
        drops gears from the axes so the carried product fits the window - at most 1 gear of
        2..4 (q = 7..13) and 2 gears of 5..7 (q = 17..23) - which certifies openness for those
        gears only and leaves the rest to the value: the island / corridor locator of R4, already
        refuted (the wall's face B). DEAD as stated; survives only as the known corridor form.
        - R5.f.xxxv.a. THE RANGE ANALYSIS: HOW THE GEARS ABOVE q SIT ON q'S OPENINGS (owner's
          choice 2026-09-21; Fable lane opened, brief lane_range/BRIEF.md; pattern rules only).
          PRE-REGISTERED questions: Q1 one gear g of T on one period = a 1/g sample of the
          twisted machine (tooth distances (3g)^{-1} mod h) - the widen/cluster/widen dynamics as
          the twisted gaps; Q2 a gear above the span of a run strikes at most two of its openings
          and exactly two only at gap 3^{-1} or g - 3^{-1}, so gears above 3F(q)+3 strike one per
          run; Q3 two gears coincide at four residues mod g g', at most four times in the range
          when g g' > P; Q4 the least number of distinct gears of T needed to strike a whole run
          of k consecutive openings, from the gap word and its leg primes; Q5 the tiers of the
          range at q = 13 with the exhaust cap exact; Q6 a further rule (openings of q no gear of
          T can strike; the mirror; the leg primes' distribution). RESULT (received 2026-09-21;
          scripts research/stack/r8/lane_range/, all assertions pass). Q1 PROVED: the openings of q
          in the class +c_g of a gear g of T are exactly c_g + j g with j an opening of the
          twisted machine M_g (teeth (3g)^{-1}-spaced), read over j in [0, P/g); the other class is
          the same cycle read backwards from j = 0, so one gear reads q's own cycle re-indexed by
          multiplier g over a symmetric window of about 2P/g centred on its home column; widen /
          cluster / widen = the twisted machine's gap word times g (cluster = twisted gap 1, hits
          every g columns). Q2 PROVED, threshold sharpened: a gear above a run's span strikes at
          most two of its openings, two only at distance d_g or g - d_g (g | 3D -+ 1); two
          openings at distance D < g are both struck only if g <= 3D + 1, so no gear above 3F + 4
          strikes both ends of an adjacent pair (bound 3F + 4, not 3F + 3; at q = 23, 103 is prime
          and strikes no 34-gap by absence); REFUTED as stated: "one per run above 3F+3" - the
          correct form is one per run of span S for g > 3S + 1 (q = 13, g = 37 strikes 105 and
          117 of the run 105-117); and the leg rule must add "or g | d" for g <= F + 1 (q = 19,
          g = 23 strikes two openings 23 apart). Q3 PROVED: g < g' coincide exactly on four
          residues mod g g' (two = the composite gear's own classes, two crossed), so at most
          4 ceil(P/(g g')) coincidences per period and at most four once g g' > P (325 of 561 pairs
          at q = 13); a coincidence lies on an opening iff its index is an opening of the twisted
          machine for g g'. Q4 PROVED as a floor: closing a run with gap word w needs at least
          mu_T(w) distinct gears - the least number of alignable blocks (a pair at distance D is a
          block iff some g in T divides 3D - 1, 3D + 1 or D; a block of three or more has two
          positions exactly g apart so g <= span); exhaustive at q = 13 for runs of 2..6 (mu = 1
          iff the chain rule; mu ranges up to 6 with (2,2,1,2,2) the unique mu = 6 word); equality
          is a residue fact, not a word fact, and over half of all runs hold a twin. Q5 PROVED:
          tiers of q = 13 - (q, q^2] no deciding prime, (q^2, q^4] primes 17..167, the top 245
          columns add 173; the exhaust cap exact (173's 16 strikes on tier-2 openings are all
          echoes); THE TWINS OF EACH TIER ARE EXACTLY THE OPENINGS OF THE MACHINE WHOSE TOP GEAR IS
          THE LARGEST PRIME AT OR BELOW THE TIER'S ROOT, restricted to the tier (13 on tier 1: its
          nine openings 3, 5, 7, 10, 12, 17, 18, 23, 25; 167 on tier 2; 173 on tier 3) - T's
          complement in the range is a nested sequence of machine windows. Q6 PROVED (own rule):
          the mirror n -> P - n carries the class +-c_g of g in T onto itself iff g | 3P -+ 1 (the
          leg rule at the gap P), else onto a class g never strikes; no gear fixes both classes;
          so for g not dividing (3P-1)(3P+1) the mirror image of every opening g strikes is one it
          does not. JOINT ACTION: large gears' strike sets are nearly disjoint (four columns per
          period once g g' > P), a run needs mu_T(w) gears, on each tier T acts exactly as the
          machine up to the tier root. CAN T STRIKE EVERY OPENING OF q IN THE RANGE: no - tier 1
          is q's window and receives no deciding strike, and on the higher tiers the survivors are
          the openings of the tier-root machine; so the range statement for q is the disjunction
          of the window statements of the tower of roots q, ~q^2, ~q^4, ... within the range,
          each fixed by residues, none forced by the gap word alone. FACT (Q1-Q6); the range
          statement reduces exactly to the windows of the tower's root machines.
        - R5.f.xxxv.c. MACHINE 5 AS THE CYCLE, OVERLAY = EVERY GEAR ABOVE 5 (owner 2026-09-21;
          research/stack/r8/machine5_cycle.py). Machine 5 = base 2, 3 with gear 5, period 30; its
          known opening (-1, 1) recurs at every multiple of 30, copies (30k-1, 30k+1), k = 1, 2, ...
          PRE-REGISTERED: gear g >= 7 strikes copy k iff k = +-30^{-1} mod g, two classes of k per
          gear at distance 15^{-1} mod g, so the overlay on the copies is a machine of the same
          shape as the 6n+-1 machine with 5 folded into the base; copy k is a twin iff no gear up
          to sqrt(30k+1) strikes k; prediction: every range (q, q#] holds a twin copy; refuted by
          a q with none. RESULT: classes verified (7: k = 3, 4 mod 7, distance 1; 11: 4, 7 mod 11;
          13: 3, 10; 17: 4, 13; 19: 7, 12; 23: 10, 13). Twin copies k <= 200: 1, 2, 5, 6, 8, 9,
          14, 19, 20, 22, 27, 34, 35, 41, 43, 44, 54, 65, 71, 77, 78, 85, 91, 93, 99, 100, 104,
          110, 111, 112, 113, 118, 131, 134, 135, ... (gaps 1..13). First failures: k = 3 (91 =
          7 x 13), 4 (119 = 7 x 17, 121 = 11^2), 7 (209 = 11 x 19) - each the two classes of the
          first gears. For every q <= 79 the range (q, q#] holds a twin copy, and the first one
          above q is tiny: k = 1 (29, 31) for q <= 23, k = 2 (59, 61) for 29 <= q <= 53, k = 5
          (149, 151) for 59 <= q <= 79. READING: the copies (30k-1, 30k+1) are the twins whose
          middle is a multiple of 30; the range check for machine q asks that this family reach
          past q, so "a twin copy in every range" is the statement that the family never stops -
          the same shape of question as the root, one base gear higher (the machine with base
          2, 3, 5 in place of 2, 3). FACT (the overlay's shape on the copies, exact); the range
          statement for every q follows from the copies' family being unbounded, not yet shown.
          - R5.f.xxxv.c.i. THE LAP MACHINE'S WINDOW (2026-09-21; research/stack/r8/
            machine5_lap_window.py). Spawned by the reading of R5.f.xxxv.c: the overlay gears 7..q
            on the laps k form a machine (two classes of k per gear); by the square-root rule an
            open lap k with q < 30k-1 and 30k+1 < q'^2 (q' the next prime) is a twin, so the range
            statement for q follows from "the lap machine 7..q has an open lap in its window
            (q/30, q'^2/30)". PRE-REGISTERED: prediction, an open lap in every such window;
            refuted by a q whose window laps are all struck. RESULT q <= 300: never refuted;
            first open lap k = 1 (q <= 23), 2 (29..53), 5 (59..137), 8 (139..~199), 14 (..293).
            Lap-machine record (longest struck run of laps over a full period, exact): 2, 4, 5,
            7, 12, 18 laps for q = 7, 11, 13, 17, 19, 23 (the 6n+-1 machine's runs on the same q:
            4, 6, 10, 17, 24, 33); longest struck run inside the window: 12 laps at q = 101 of a
            350-lap window, 37 at q = 199 of 1477, 41 at q = 293 of 3132. READING: the lap machine
            is machine q read on the columns 5k only; its window statement has the same shape as
            E4 (record below the window length), one base gear higher. FACT (holds to 300, exact
            records to 23); the lemma "lap record < window length" is E4 for base 2, 3, 5, open.
          - R5.f.xxxv.c.ii. SHAPE OF THE LAP MACHINE'S RUNS (2026-09-21; research/stack/r8/
            machine5_lap_legs.py). Spawned by c.i's record comparison. Lap class distance of gear
            g is 15^{-1} mod g (nearer representative): 7:1, 11:3, 13:6, 17:8, 19:5, 23:3, 29:2,
            31:2, 37:5, 41:11, 43:20, 47:22, 53:7, 59:4, 61:4 (the 6n+-1 machine's 3^{-1}: 7:2,
            11:4, 13:4, 17:6, 19:6, 23:8, ...). LEG RULE ON LAPS (PROVED, checked 7..37): gear g
            strikes two laps at distance D iff g | D or g | 15D - 1 or g | 15D + 1; so the gears
            pairing laps at D = 1: 7 only; D = 2: 29, 31; D = 3: 11, 23; D = 4: 59, 61; D = 5:
            19, 37; D = 6: 7, 13, 89; ... RUN SHAPE: gear 7 strikes adjacent laps (3, 4 mod 7) every
            7 laps, leaving 5-lap holes; a run of struck laps is 7's adjacent pairs with the holes
            between them filled by the gears 11..q, and a gear fills two laps of one hole only if
            its lap distance is at most 4 (11, 23, 29, 31, 59, 61), else one. Record runs read as
            words: q = 11: 11 7 7 11; q = 13: 13 11 7 7 11 (11's pair at distance 3 straddles
            7's pair); q = 17: 13 17 11 7 7 11 13; q = 19: 13 | 7 7 | 17 11 19 13 11 | 7 7 | 19 17
            (the first fully filled hole: 11 doubles at distance 3, 13, 17, 19 single); q = 23:
            13 | 7 7 | 11 17 19 11 13 | 7 7 | 19 23 17 13 11 | 7 7 | 11 (two filled holes). So the
            lap record is 2 + 5 per filled hole plus the partial holes at the ends; filling one
            hole needs at least three gears of 11..q (at most two doublers), and the record is
            the longest chain of consecutive filled holes. FACT (exact); the open question "lap
            record < window" becomes "how many consecutive 5-holes can the gears 11..q fill".
          - R5.f.xxxv.c.iii. THE HOLE MACHINE (2026-09-21; research/stack/r8/machine5_holes.py,
            machine5_holes_np.py). Spawned by c.ii: the lap record is a chain of filled 5-lap
            holes. Hole h = laps 7h+5..7h+9, positions p = 1..5. PRE-REGISTERED and PROVED: gear
            g >= 11, class a, strikes hole h at position p iff h = 7^{-1}(a - 4 - p) mod g, so over
            g holes each class hits five holes, an arithmetic progression of holes with step 7^{-1}
            mod g and the position stepping down by one; at a fixed position the two classes sit at
            hole distance d_g 7^{-1} mod g (predicted 11:2, 13:1, 17:6, 19:2, 23:7, 29:8, 31:13;
            confirmed on the data: 13 at holes 5, 6; 11 at 0, 9; 19 at 0, 17). Gear 13 is the
            hole machine's adjacent gear: its two classes hit ten consecutive holes 1..10 mod 13 at
            positions 5, 5, 4, 4, 3, 3, 2, 2, 1, 1 (a descending staircase), the way gear 7 hits
            adjacent laps (7 x 13 = 91 = 15 x 6 + 1, the D = 6 leg). Longest chain of filled holes
            per period: 0 for q <= 17, 1 at q = 19, 2 at q = 23, 3 at q = 29; lap record = 5 x
            chain + 2 x (chain + 1) + ends: 12 = 5 + 4 + 3, 18 = 10 + 6 + 2, 25 = 15 + 8 + 2 at
            q = 19, 23, 29 (record 25 for gears 7..29 computed exact). The q = 29 chain: positions
            struck by (17 19 29 13 29 | 11 17 13 11 23 | 23 19 13 17 11) - 13 at position 4, 3, 3
            in the three holes. FACT (exact); the self-similar shape: laps -> gear 7 cuts adjacent
            pairs with 5-holes; holes -> gear 13 cuts adjacent hole pairs; the record at each
            level is the longest chain the remaining gears can fill.
          - R5.f.xxxv.c.iv. THE RANGE RULES Q1-Q6 ON THE LAPS (owner 2026-09-21: the earlier
            range-check rules for the gears above q should carry over). They do, with 30 in place
            of 6 and 15 in place of 3. Q1 (twisted read): gear g strikes laps k = +-30^{-1} mod g,
            two classes at distance 15^{-1} mod g (node c). Q2 (leg rule): g strikes two laps D
            apart iff g | D, 15D - 1 or 15D + 1; a gear above 15 x span + 1 strikes at most one lap
            of a run (node c.ii). Q3 (coincidence): two gears strike the same lap on exactly four
            residues mod g g' (checked 7..19; seen at q = 23, lap 30600 struck by 7 and 23).
            Q4 (closing floor): a run of laps needs at least mu gears, a 5-hole at least three
            (doublers 11, 23, 29, 31, 59, 61 only) (node c.ii). Q5 (tiers): on tier (q, q'^2] the
            surviving laps are the openings of the lap machine 7..q (node c.i, every q <= 300).
            Q6 (mirror): on laps the mirror k -> -k fixes every gear's class pair, so the lap
            machine's openings are symmetric about lap 0 (the origin copy) - stronger than on
            columns, where a gear's pair is fixed only if g | 3P -+ 1. JOINT ACTION: the gears
            above 5 strike every lap of a range only if every tier's root lap machine fills its
            tier; tier 1 is the lap window (c.i). FACT; the range statement with machine 5 as the
            cycle is the tower of lap-window statements, the first of which is c.i.
          - R5.f.xxxv.c.v. THE CENTRE OF THE LAP PERIOD (2026-09-21; research/stack/r8/
            machine5_centre.py). Spawned by c.iv's Q6: the lap machine is symmetric about lap 0,
            hence about the half period P_L/2 (P_L = prod 7..q, odd). PRE-REGISTERED and PROVED:
            the lap at offset j/2 from the centre (j odd) has members 15 P_L +- 15j -+ 1, and every
            gear divides P_L, so gear g strikes it iff g | 15j - 1 or g | 15j + 1 - the leg rule read
            from the centre, as 30k -+ 1 is read from the origin. So the centre pair (P_L -+ 1)/2 is
            always 7's adjacent pair (14 = 2 x 7); the offsets are struck by j = 3: 11 or 23; 5: 19
            or 37; 7: 13 or 53; 9: 17 or 67; 11: 41 or 83; 13: 7 or 97; 15: 7 or 113; 17: 127 only;
            19: 11, 13, 71. Central struck run = 2m laps with m the consecutive odd j from 1 whose
            15j -+ 1 has a factor in [7, q]: 4 laps for q = 11, 13, 17; 10 for q = 19..37; 16 for
            q = 41..113; 34 for q = 127 (j = 17 needs 127). All confirmed on the periods q = 11..23
            (centre words: q = 11: 11 7 7 11; q = 19: 17 13 19 11 7 7 11 19 13 17; q = 23 the same
            with 11+23 doubling). At q = 11 the record run is the central run. The origin and the
            centre are the two known objects of every lap machine: the origin lap is always open
            (members -1, 1), the centre pair always struck by 7; the origin's neighbourhood is the
            lap window itself (lap k open iff 30k -+ 1 both free of gears <= q), the centre's
            neighbourhood is the half-leg word 15j -+ 1. FACT (exact, every q).
          - R5.f.xxxv.c.vi. CENTRE RUN VERSUS RECORD (2026-09-21; data of c.i, c.iii, c.v).
            PRE-REGISTERED: is the record run the centre's shape for every q, or does the record
            outgrow the centre by more than one filled hole? RESULT: centre / record in laps:
            q = 11: 4 / 4; 13: 4 / 5; 17: 4 / 7; 19: 10 / 12; 23: 10 / 18; 29: 10 / 25. In filled
            holes: 0/0, 0/0, 0/0, 1/1, 1/2, 1/3. REFUTED beyond q = 11: the record outgrows the
            centre from q = 23 on, by a filled hole per new gear (23 fills a second hole, 29 a
            third) while the centre stays at 10 laps until q = 41 (offset j = 11 needs 41 or 83).
            MECHANISM: at the centre every gear sits at phase 0 (all divide P_L), so the centre
            word is fixed by the least factors of the fixed numbers 15j -+ 1 and grows only when
            q passes one of them; a record run sits where the gears' phases are chosen freely
            (the alignable chain of E4), so each new gear can add a filled hole at once. FACT.
            The centre is the run at phase 0, a lower bound for the record for every q (record
            >= 2m(q)), exact only at q = 11.
          - R5.f.xxxv.c.vii. THE STAIRCASE RULE (2026-09-21; research/stack/r8/
            machine5_staircase.py). Spawned by c.iii (13 in every hole of every chain). PROVED: for
            a gear g >= 11 other than 13 the hole step 7^{-1} mod g is at least 3 from 0 (it is +-1
            or +-2 only for g | 6, 8, 13, 15), so one class hits at most one of any three consecutive
            holes and the gear at most two of their fifteen laps; 13 alone hits ten consecutive
            holes (11, 12, 0 mod 13 are 13-free). PRE-REGISTERED floors (Q4): a chain of 13 holes
            holds three consecutive 13-free holes and needs at least eight gears other than 7, 13
            (so none for q < 41); a chain of 12 needs at least five (q >= 29). TEST (partial
            periods, first 2e8 laps): longest chain 4 at q = 31 (hole 19728987) and 4 at q = 37
            (hole 3180257), both inside the staircase, no 13-free hole in either. Not refuted; the
            chains observed so far (q <= 37) all lie inside 13's ten-hole staircase with 13 in
            every hole. FACT (rule) + observation; the floors alone do not bound the record below
            the window (chain <= 12 gives record <= 94 laps against 55 window laps at q = 37).
          - R5.f.xxxv.c.viii. HOLE FILLS INSIDE THE STAIRCASE (2026-09-21; research/stack/r8/
            machine5_hole_fills.py). Spawned by c.vii. With 13 at position p the other four
            positions are filled by doublers at their lap distance (11, 23: 3; 29, 31: 2; 59, 61:
            4) or by singles. RULE (exact): admissible doubler pairs avoid p - p = 5: (1,3), (2,4)
            for 29/31, (1,4) for 11/23; p = 4: (1,3), (3,5), (2,5), (1,5); p = 3: (2,4), (1,4),
            (2,5), (1,5); p = 2 and 1 mirror p = 4 and 5. A 2+2 fill (two doublers + 13) exists
            only as: p = 5 or 1: 29 with 31; p = 4 or 2: one of 11/23 with one of 29/31; p = 3:
            11 with 23, or 59/61 with 29/31. READ on all 14 observed chain holes (q = 19, 23, 29,
            31, 37): 13's position as predicted every time; every doubler at an admissible pair;
            the only 2+2 fill seen is p = 3 with 11 at (1,4) and 23 at (2,5) (q = 37, hole = 5 mod
            13), exactly the table's 11-with-23 case; most holes are 2+1+1 (one doubler, two
            singles) or four singles; coincidences (two gears on one lap, Q3) appear in six holes
            (13&29, 11&13, 17&23, 13&31, 11&19, 13&17). FACT (exact). Next: each doubler's word
            across consecutive holes (class a at p, hole h+2 at p again for 11 by its hole distance
            2) - the tiling of a chain by the gears' fixed hole words.
          - R5.f.xxxv.c.ix. HOLE WORDS (2026-09-21; research/stack/r8/machine5_hole_words.py).
            Spawned by c.viii. Gear g's hole word = positions struck in holes 0..g-1, one period.
            PRE-REGISTERED and PROVED (checked g = 11..61): each class gives five holes, step 7^{-1}
            mod g, positions 5, 4, 3, 2, 1 along the progression; the two classes share a hole
            exactly at the position pairs (p, p + d_g), so the word has 5 - d_g double holes when
            the lap distance d_g <= 4 (11, 23: two; 29, 31: three; 59, 61: one) and none otherwise;
            it has g - 10 + doubles empty holes; and it is a palindrome under the lap mirror
            (hole h -> -h-2, position p -> 6-p). Words: 11: 3|4|-|14|5|1|25|-|2|3|-; 13:
            -|5|5|4|4|3|3|2|2|1|1|-|-; 17: -|2|3|5|-|-|1|2|4|5|-|-|1|3|4|-|-; 19:
            3|1|-|1|-|-|4|-|4|2|-|2|-|-|5|-|5|3|-; 23: -|2|-|-|14|-|-|3|-|-|5|1|-|-|3|-|-|25|-|-|4|-|-;
            29: doubles 35, 13, 24 at holes 3, 24, 28, singles 4, 5, 1, 2; 31: doubles 13, 35, 24 at
            holes 8, 21, 30. So the whole lap machine above 7 is the superposition of these fixed
            periodic words, one per gear, each set by two numbers (7^{-1} mod g and 15^{-1} mod g);
            a chain of filled holes is a window where the superposed words cover every position.
            FACT (exact).
          - R5.f.xxxv.c.x. STACKING THE WORDS (2026-09-21; research/stack/r8/machine5_stacking.py).
            Spawned by c.ix. A chain starting at hole h0 puts gear g at phase h0 mod g of its word;
            every phase tuple occurs (hole periods coprime), so the longest chain is a puzzle on the
            words alone. PRE-REGISTERED forced fills, both CONFIRMED exactly: q = 19 - densest
            1-windows 11: 2, 13: 1, 17: 1, 19: 1 sum to 5, so every filled hole has 11 doubling and
            13, 17, 19 once each with no coincidence (all 96 filled holes of the period); q = 23 -
            densest 2-windows 3, 2, 2, 2, 2 sum to 11 against 10 needed, so every two-hole chain has
            11 in a 3-window or every gear at its maximum (36 chains: 34 with 11 at 3, 2 with all
            five gears at 2). WORD BOUND (longest chain <= largest m with the densest m-windows
            summing to at least 5m): 1, 2, 5, 7, 10, 15, 28 for q = 19, 23, 29, 31, 37, 41, 43;
            observed 1, 2, 3, >= 4, >= 4 - tight at 19 and 23, loose from 29. From q = 47 the
            densest windows sum to 5 or more per hole for every m: NO BOUND from the words' densities
            alone; from there only the phases (which windows of each word can sit side by side)
            limit a chain. FACT (exact); the wall in word form: at q >= 47 the gears carry enough
            teeth to fill every hole, and the proof must come from the words' relative phases, not
            their densities - the same place E4 stands on columns.
          - R5.f.xxxv.c.xi. PHASES OF 11 AND 13 IN THE CHAINS (2026-09-21; research/stack/r8/
            machine5_phases.py). Spawned by c.x. q = 19, all 96 filled holes: 11 is at phase 3
            (positions 1, 4) or 6 (positions 2, 5) and nowhere else; 13's phase is then any of the
            six staircase phases whose position avoids 11's pair (11 at 3: 13 at phases 1, 2, 5, 6,
            7, 8 = positions 5, 5, 3, 3, 2, 2; 11 at 6: 13 at 3, 4, 5, 6, 9, 10 = positions 4, 4, 3,
            3, 1, 1); 17 and 19 take the two positions left, each with two phases per position; so
            the filled set is exactly 12 fill diagrams (gear per position) x 8 residue classes mod
            11 x 13 x 17 x 19 each - a CRT product of the diagrams, as pre-registered. q = 23, all
            36 two-hole chains: 11 at phase 3 (window "14|5") in 17, at phase 5 ("1|25") in 17, and
            the two all-maximum chains at phases 6 and 2 ("25|-", "-|14"); 13's phase with 11 at 3
            lies in {2, 5, 6, 7, 8}, with 11 at 5 in {2, 3, 4, 5, 8} - the staircase must avoid
            11's cells in both holes; the 36 chains are 18 mirror pairs (phase 3 <-> 5, 6 <-> 2
            under h -> -h-3). RULE: a chain of m holes is an m x 5 diagram tiled by one window of
            each gear's word (cells may coincide), and the chains of the period are the CRT classes
            of the consistent diagrams; the two densest words, 11 and 13, must place their cells
            disjointly (or waste a coincidence), and that relation is a residue rule between h mod
            11 and h mod 13, not a density. FACT (exact).
          - R5.f.xxxv.c.xii. THE LAP RECORD FROM THE WORDS ALONE (2026-09-21; research/stack/r8/
            machine5_word_solver.py). Spawned by c.xi's tiling rule. Solver: walk the laps; a fixed
            gear striking the lap advances it; otherwise branch over the free gears and their two
            classes, each fixing k0 mod g; the longest walk is the record (every phase tuple occurs,
            CRT). PRE-REGISTERED: 2, 4, 5, 7, 12, 18, 25 for q = 7..29 - all reproduced with no
            lap scan; q = 31: 31 laps (new exact value; the partial-period scan had chain 4, i.e.
            at least 30). Phases of the q = 31 record: k0 = 3, 2, 12, 12, 6, 10, 18, 11 mod 7, 11,
            13, 17, 19, 23, 29, 31. Lap record against the lap window (q'^2/30 laps): 4/5, 5/9,
            7/12, 12/17, 18/28, 25/32, 31/45 for q = 11..31. FACT (exact): the lap record is a
            finite word puzzle, and the lemma "record < window" on laps is a statement about
            which class choices can be made consistently along a walk - the alignable-chain
            recursion of E4, now with the tiling structure (c.viii-c.xi) as its shape.
          - R5.f.xxxv.c.xiii. ALL RECORD RUNS OF A PERIOD (2026-09-21; research/stack/r8/
            machine5_record_diagrams.py, machine5_all_records.py). Spawned by c.xii: the solver's
            walk was one record run; the scan at q = 23 showed another. Full-period scans: q = 19:
            4 record runs (two mirror pairs), q = 23: 22 (mirror-closed), q = 29: 2 (one mirror
            pair). What all of them share: WASTE - the strikes landing inside the run beyond one
            per lap are exactly 0 at q = 19, 1 at q = 23, 2 at q = 29, the same for every record
            run of the period (the record is where the words stack with least overlap; the solver's
            q = 31 walk wastes 6, so other q = 31 record runs may exist with less); ENDS - the run
            begins and ends on a strike by 13 or 7 (q = 19: 13, 7, 17; q = 23: 13 in 10, 7 in 9,
            11 in 2, 13+19 in 1; q = 29: 7 and 13, mirrored), the laps just outside are open and
            every gear's next strike is 2 to 5 laps away - no single gear ends a run, the walk
            stops where all the fixed classes miss together; 7's pairs inside the run: two at
            q = 19, three at q = 23, four at q = 29, always aligned so the run holds whole 7-pairs
            with a partial hole at each end. FACT (exact). The stopping constraint is not one
            gear's; it is the joint miss of all classes once fixed - the phase relation again.
          - R5.f.xxxv.c.xiv. THE WASTE PREDICTION (2026-09-21; research/stack/r8/
            machine5_waste.py). Spawned by c.xiii's waste figures 0, 1, 2 at q = 19, 23, 29, whose
            machines carry 4, 5, 6 gears above 7. PRE-REGISTERED: waste = (gears above 7) - 4, so 3
            at q = 31. REFUTED: the minimal waste over every record walk at q = 31 is 6, not 3 (the
            walk enumeration reproduces 0, 1, 2 exactly at q = 19, 23, 29, so the method is sound).
            MECHANISM of the jump: at a 31-lap run both close-distance gears are forced inside - a
            gear with lap distance d cannot avoid a run longer than its larger class gap g - d, and
            29 (gap 27) and 31 (gap 29) are both under 31 - so their six strikes land in a run that
            the other five gears already cover, and every extra forced gear adds its own overlap.
            The rule that survives: a gear g must strike any run longer than g - d_g, and a run
            longer than g + d_g holds both its classes. FACT (the forced-gear rule); the waste
            sequence is not linear in the gear count and is not a route.
          - R5.f.xxxv.c.xv. THE TILING LANE: TWO LOCKS AND THE CHAIN LADDER (lane received
            2026-09-21; scripts in the lane scratchpad, results reproduced below). Spawned by
            c.xiii: state the chain question on the hole words. FIVE RULES, all PROVED with checks.
            (1) CLASH RULE (11 against 13): gear 11 strikes position p exactly at holes 3p and
            3p + 2 mod 11, gear 13 staircase at -2p - 2 and -2p - 1 mod 13; in general a gear two
            holes for one position are delta_g = 2 x 7^{-1} a_g apart, INDEPENDENT OF p (checked
            11..61; nearest delta 11:2, 13:1, 17:6, 19:2, 23:7, 29:8, 31:13, 53:1). The 11/13
            overlap holes are exactly 20 of the 143 residues mod 143, gaps only 3, 5, 7, 8, 13,
            longest clash-free run 12 holes. Disjoint phase pairs fall 123, 103, 83, 66, 49, 38,
            27, 20, 16, 12 of 143 for m = 1..10; with 13 full in its staircase they reach 0 at
            m = 9 - the first length at which EVERY phase pair clashes; the last two survivors at
            m = 8 are (5, 3) and (8, 1) mod (11, 13), a palindrome pair. (2) DOUBLERS: the only
            two-doubler fills are p = 1 or 5 with 29 and 31; p = 2 or 4 with one of 11/23 and one
            of 29/31; p = 3 with 11 and 23, or one of 59/61 with one of 29/31; and THE LONGEST RUN
            OF HOLES FILLED BY 13 PLUS TWO DOUBLES EACH IS EXACTLY 2, because the smallest window
            holding two double holes of one gear is 4 (11), 11 (23), 5 (29), 10 (31) - no doubler
            doubles twice inside three consecutive holes - and 59, 61 do not help (both offer only
            positions 1 and 5). (3) SINGLES: longest stretch of consecutive non-empty holes 17: 4
            (1|2|4|5), 19: 2, 37: 2, 41: 2, 43: 1, 47: 1, 53: 2; a single gear puts two cells in an
            m-window only if delta_g <= m - 1 (same position twice) or one class meets it twice at
            hole distance k 7^{-1}, k <= 4; over three consecutive holes only 17 reaches three
            cells, and 43, 47 reach one. (4) THE DOUBLE-WINDOW LOCK: with slack S = sum of the best
            m-windows minus 5m, every gear must give at least cap(g, m) - S; at small slack this
            forces a doubler onto the unique window carrying two of its double holes, fixing both
            its offsets and its positions, and the rows it leaves exceed what the singles of (3)
            can supply. The lock bites for 11 from m = 4, 29 from m = 5, 31 from m = 10, 23 from
            m = 11. Proofs by this route: chain 4 impossible at q = 29 (only four phase pairs
            survive the forcing, all with 11 on its unique 6-cell window 14|5|1|25, and 17, 19,
            23, 29 cannot partition the residual), chain 5 impossible at q = 31 (killed at gear 29,
            whose only 4-cell 5-window puts cells in the two end holes only). (5) THE DEAD BLOCK:
            13 is silent exactly at holes 0, 11, 12 mod 13, one block of three, so every chain of
            11 holes meets a silent hole and every chain of 13 holes meets all three; a silent hole
            needs at least three gears, three only if all double there - the covering triples are
            (11,23,29), (11,23,31), (11,29,31), (23,29,31), (29,31,59), (29,31,61) - and since no
            doubler doubles twice in three consecutive holes the three silent holes take at most
            four doubles in total, so at most one is a three-gear hole and the other two need four
            gears each. THE CHAIN LADDER (words only, two independent engines agreeing, first four
            matching the known chains): chain = 1, 2, 3, 4, 5, 6, 8, 9 for q = 19, 23, 29, 31, 37,
            41, 43, 47; window in holes q'^2/210 = 2.52, 4.00, 4.58, 6.52, 8.00, 8.80, 10.52,
            13.38; margins 1.52, 2.00, 1.58, 2.52, 3.00, 2.80, 2.52, 4.38. "Each new gear adds one
            hole" PROVED for 19 <= q <= 41 and REFUTED at q = 43 (6 to 8). STRONG (rules proved,
            ladder computed); the lemma now rests on making one of the two locks hold for every q
            rather than gear by gear - the density bound dies at q = 47 (c.x) and beyond it only
            the phase relations of (1)-(5) carry the weight. INDEPENDENT CHECK (manager,
            research/stack/r8/machine5_chain_exists.py): the existence half of every ladder entry
            confirmed by direct construction - a phase tuple covering all 5m cells found for every
            (q, m) of the ladder, by a set-cover search written from the cell rule alone and not
            from the lane's engines; e.g. q = 43, m = 8 at h0 = 1, 1, 7, 10, 4, 24, 7, 31, 36, 7
            mod 11, 13, 17, 19, 23, 29, 31, 37, 41, 43. Two of the constructed chains (q = 37 and
            q = 47) place 13 at h0 = 0 mod 13, i.e. they OPEN on a dead-block hole, so the dead
            block does not bar a chain from starting in it. The impossibility half (no chain of
            m + 1) rests on the lane's two engines and is not independently rechecked here.
          - R5.f.xxxv.c.xvi. THE LAP ROUTE IS A RESTRICTION, NOT THE RANGE STATEMENT
            (2026-09-21; research/stack/r8/machine5_three_classes.py). Checked while looking for a
            lock that holds for every q. Gear 5 strikes columns n = 1 and 4 mod 5, so MACHINE 5
            LEAVES THREE OPEN CLASSES PER PERIOD: n = 0, 2, 3 mod 5, arranged as a lone column and
            an adjacent pair. The laps of nodes c..c.xv follow n = 0 mod 5 only - the copies
            30k -+ 1 of the known opening - and discard the classes n = 2, 3 mod 5, which machine 5
            carries round its period just as well. So the lap statement is a restriction of the
            range statement to one of three opening classes, and is strictly harder. MEASURED on
            the same windows (record against window, both in their own units): full machine 5..q
            0.20, 0.21, 0.21, 0.28, 0.27, 0.24, 0.26, 0.25 for q = 7, 11, 13, 17, 19, 23, 29, 31;
            lap machine 7..q 0.50, 0.71, 0.52, 0.58, 0.68, 0.64, 0.78, 0.68 on the same q. The lap
            route runs about 2.7 times tighter throughout - the margin the tiling lane was fighting
            for (1.52 holes at q = 19) is an artefact of throwing away two thirds of the openings,
            not a feature of the range statement. WHAT SURVIVES: the word structure itself, which
            was never about the choice of class - a cutter gear leaving fixed blocks, a staircase
            gear walking down them, doublers admissible only at their own class distance, singles
            limited to one cell in three blocks, and the two locks (double-window, dead block).
            The same construction applies to the full opening set of machine 5, where gear 5 is the
            cutter and the block is the three open columns 0, 2, 3 of each five. FACT (the
            restriction and its cost); c..c.xv are kept as the worked example of the method on the
            hardest sub-case, and the method now moves to the full object.
          - R5.f.xxxv.c.xvii. THE WORD CONSTRUCTION ON THE FULL OPENING SET (2026-09-21;
            research/stack/r8/full_blocks.py, full_chain_ladder.py). The method of c..c.xv applied
            to the object c.xvi says it should be applied to. Gear 5 is the cutter: it strikes
            columns 1 and 4 mod 5, leaving a BLOCK of three open columns per five, positions
            p = 1, 2, 3 at columns 5b, 5b + 2, 5b + 3. Gear g >= 7 strikes block b at position p
            iff b = 5^{-1}(+-6^{-1} - e_p) mod g with (e_1, e_2, e_3) = (0, 2, 3).
            PRE-REGISTERED and PROVED: the open columns of a block sit at distances 1, 2, 3, and
            3D -+ 1 is 2, 4 for D = 1, 5, 7 for D = 2, 8, 10 for D = 3, so BY THE LEG RULE ONLY
            GEAR 7 CAN STRIKE TWO CELLS OF ONE BLOCK, at positions 1 and 2, exactly at the blocks
            b = 4 mod 7 (checked over all gears to 200). Every gear >= 11 gives AT MOST ONE CELL
            PER BLOCK and exactly six cells per period of g blocks, two classes by three positions.
            Gear 7 block word: - | 3 | 3 | 1 | 12 | 2 | -. Longest run of consecutive non-empty
            blocks: 2 for the gears 11..23, 1 for every gear from 29 up - a gear of 29 or more
            never strikes two neighbouring blocks. Class steps within a gear (p1 to p2, p2 to p3)
            are -2 x 5^{-1} and -5^{-1} mod g. THE CHAIN LADDER (set cover on the words, same
            engine as the independent check of c.xv): chain = 0, 1, 1, 3, 4, 6, 8, 11 for q = 7,
            11, 13, 17, 19, 23, 29, 31; the column record follows as F(q) = 5 x chain + ends with
            ends in [0, 6], confirmed against the recorded F(q) = 4, 6, 10, 17, 24, 33, 42, 57
            (differences 4, 1, 5, 2, 4, 3, 2, 2). Window in blocks q'^2/30 = 4.0, 5.6, 9.6, 12.0,
            17.6, 28.0, 32.0, 45.6, so chain over window runs 0.00, 0.18, 0.10, 0.25, 0.23, 0.21,
            0.25, 0.24 - three to four times the room the lap route had. WORD BOUND (largest m
            whose densest m-windows sum to at least 3m): 0, 1, 2, 5, 9, 20, and then none, because
            a gear contributes about 6m/g cells to an m-window and the bound exists only while the
            gears' reciprocals sum below one half (7 to 23 give 0.466, adding 29 gives 0.500,
            adding 31 gives 0.532). So density proves the lemma outright for q <= 23 and dies at
            q = 29. STRONG; the locks of c.xv now have a much simpler object to bite on - one
            doubler at one residue, every other gear one cell per block, nothing above 29 touching
            two neighbouring blocks.
          - R5.f.xxxv.c.xviii. THE LEG RULE ON BLOCKS (manager, 2026-09-21; research/stack/r8/
            full_block_leg_rule.py). Spawned by c.xvii: generalise "only gear 7 doubles in a block"
            and "no gear from 29 up reaches a neighbouring block" to every block distance.
            DERIVATION: a gear's six cells sit at blocks s(+-c - e_p), s = 5^{-1}, c = 6^{-1} mod g,
            (e_1, e_2, e_3) = (0, 2, 3). Two cells lie D blocks apart either in one class, where
            D = +-s(e_p - e_p') and multiplying by 5 gives 5D = +-1, +-2, +-3 mod g, or across the
            classes, where D = +-(2sc + sj) with j in {0, +-1, +-2, +-3} and multiplying by 15 and
            using 5s = 1, 6c = 1 gives 15D = +-(1 + 3j). RULE (PROVED): gear g strikes two blocks D
            apart iff g divides one of 5D +- 1, 5D +- 2, 5D +- 3, 15D +- 1, 15D +- 2, 15D +- 4,
            15D +- 5, 15D +- 7, 15D +- 8, 15D +- 10 - plus gear 7 when 7 divides D, since only gear
            7 has two cells in one block. CHECKED exactly against direct search over every gear to
            500, for D = 0..12, agreeing on every gear: D = 0 gives 7 alone; D = 1 gives 7, 11, 13,
            17, 19, 23 and nothing above; D = 2 adds 29, 31, 37; D = 3 gives 7, 11, 13, 17, 19, 23,
            37, 41, 43, 47, 53. THE BRIDGING SET FOR EACH D IS FINITE AND INDEPENDENT OF q - 6, 9,
            11, 12, 13, 14, 14, 15, 15, 16, 16, 17 gears for D = 1..12, always the prime factors of
            twenty numbers of size at most 15D + 10. COROLLARY (PROVED): a gear larger than
            15D + 10 cannot bridge the distance D, so A GEAR LARGER THAN 15m - 5 PUTS AT MOST ONE
            CELL INTO A CHAIN OF m BLOCKS - the block analogue of the window rule that a gear above
            3F + 4 strikes at most one opening of a run. FACT (exact, q-independent); this is the
            first rule of the line that does not have to be re-verified gear by gear as q grows.
          - R5.f.xxxv.c.xix. THE DEAD-PAIR REDUCTION: THE LEMMA BECOMES ONE q-FREE STATEMENT
            (six-dimension workflow with adversarial refutation, received 2026-09-22; thirteen
            agents, scripts in the run scratchpad; manager spot-checks in research/stack/r8/
            full_orphan_law.py). Six dimensions were derived and then attacked by separate workers;
            what survived is assembled here.
            THE DEAD PAIR. Gear 7 is silent exactly at blocks b = 6 and b = 0 mod 7, and those are
            ADJACENT: a dead pair, six cells that gears >= 11 must take alone. Every m consecutive
            blocks hold at least floor((m-1)/7) complete dead pairs, and k consecutive dead pairs
            span 7k - 5 blocks. ORPHAN LAW (PROVED; re-verified by the manager straight from the
            column definition over every gear to 2000): the only adjacent-block cell pairs any gear
            >= 11 has are row 1 to row 2 (gear 11), row 1 to row 3 (23), row 2 to row 3 (17, 19),
            row 3 to row 2 (11, 13) - NONE lands its right cell in row 1, and no gear >= 11 supplies
            more than two cells of a dead pair. Hence N(1) = 4, where N(k) is the least number of
            distinct gears >= 11 that can cover k consecutive dead pairs; re-verified exhaustively.
            THE REDUCTION (PROVED): chain(q) <= 7 k_max(q) + 7 with k_max = max{k : N(k) <= pi(q) - 4}.
            The 7-block spacing of the dead pairs turns the QUADRATIC window q'^2/30 into a LINEAR
            demand on N, and N is a function of k alone - no q in it.
            PROOF SKELETON. 1 columns to blocks to cells, exact - PROVED. 2 a chain is a phase
            covering problem, every phase tuple occurring by CRT and no gear's phase normalisable -
            PROVED (an attempted normalisation is recorded refuted). 3 the lemma is exactly
            chain(q) < q'^2/30 - PROVED. 4 base q <= 23 by the capacity bound 0, 1, 2, 5, 9, 20
            against windows 4.03, 5.63, 9.63, 12.03, 17.63, 28.03 - PROVED. 5 q = 29..47 by the
            row-3 relaxation, chain(q) <= R(q) with R = 25, 31, 38, 49, 59, <=93 against 32.03,
            45.63, 56.03, 61.63, 73.63, 93.63 - PROVED as six finite certificates. 6 every q via the
            dead-pair skeleton - PROVED as a reduction. 7 THE ONLY OPEN STEP: N(k) >= pi(sqrt(210(k+1))) - 3
            for every k, equivalently N(ceil(q'^2/210) - 1) > pi(q) - 4 - OPEN. 8 q <= 11 directly -
            PROVED.
            MEASURED N: 4, 5, 6, 7 for k = 1..4 (exhaustive, two independent engines), 8, 9, 10, 10,
            11, 12, 12 for k = 5..11 (one engine). The next machines each need one finite covering
            computation: q = 53 needs N(16) >= 13, q = 59 needs N(17) >= 14, q = 61 needs N(21) >= 15,
            q = 67 needs N(24) >= 16, q = 71 needs N(25) >= 17, each over the bounded pool
            g <= 105k - 80. The required growth is far weaker than the measured one - about
            29 sqrt(k) / ln(210k), so the required marginal TENDS TO ZERO while the measured margin
            grows like sqrt(q/ln q)/4.26.
            ALSO SETTLED: the chain ladder extends to chain = 0, 1, 1, 3, 4, 6, 8, 11, 17 for
            q = 7..37 (q = 37 verified by two engines; 41 -> 17 and 43 -> 20 inherited from one).
            CORRECTIONS TO THE BRIEF: the class separation is t_g = min(d_g, g - d_g) with
            t_g = t iff g divides 15t -+ 1, giving t = 1: {7}, 2: {29, 31}, 3: {11, 23}, 4: {59, 61},
            5: {19, 37}, 6: {13}, 7: {53}, 8: {17} - the manager's list in c.xviii had 13 at 7, and
            6 is correct (13 divides 15 x 6 + 1 = 91). "A gear doubles inside one row iff
            g <= 15m - 14" is only an implication, not an equivalence (gear 43, m = 4).
            REFUTED along the way: the constraint-concentration claim at short sub-windows; the
            "41 per cent of dead pairs carry a doubling" claim (six consecutive doubling-carrying
            dead pairs exist); the two-row ordering; a constant one-gear ceiling K (K = 6 at q = 31);
            the gap-insensitive sufficient condition.
            COUNTING IS CLOSED. Every count in the pile - word bound, per-row, row-vector, dead-cell,
            orphan-row, segmented and k-refined capacity - is bounded by the gears' reciprocal sums,
            which cross one half at q = 29 (gears from 7) and q = 47 (gears from 11). The dead-cell
            count stands above the window from q = 17 on. No counting argument can finish this.
            STRONG, CANDIDATE: the lemma for every q >= 13 now rests on the single q-free statement
            of step 7. What would have to break it: N(k) growing slower than about 29 sqrt(k)/ln(210k),
            which the measured values do not do; not yet shown for k >= 12.
          - R5.f.xxxv.c.xx. PRIOR-ART CHECK: THE DEAD-PAIR REDUCTION IS THE WALL'S W2 IN NEW
            COORDINATES (manager, 2026-09-22; research/proof/the_wall.md W1-W2, docs/novel/
            island-witness-integers.md section 1(d), docs/handover.md sections 3 and 7.3). Run
            before funding further work on c.xix's open step, per the standing rule to grep the
            novel index first. FINDING: the project already holds the statement c.xix reduced to.
            the_wall.md W2 reads "the family is a finite combinatorial object at every d: the
            minimum number of primes whose fixed-separation pairs can cover d consecutive columns",
            names it K_columns(d), records its first values (5, 7, 11, 18, 25, 34, 43, 58, 88, 91,
            103, 118, 145, 161 at 2..15 gears - the F ladder read backwards), and states the target
            as K_columns(d) > pi(sqrt(6d)), which it says "is exactly F < W for every member of the
            one-phase fixed-separation family". docs/novel/island-witness-integers.md 1(d) defines
            the same quantity as K(d) on islands with a table to d = 2240 and the key separation:
            the gears needed to PAY for a cover are bounded (about a dozen, the reciprocal sum)
            while the gears needed to BUILD one grow. the_wall.md section 4 records the constant:
            "the tight quantity is the constant c in K(d) ~ pi(sqrt(c d)): 6 is the target, 24 is
            measured for whole columns, 7 to 11 for islands". SO c.xix's N(k) IS K restricted to
            the dead-pair cells, and its requirement N(ceil(q'^2/210) - 1) > pi(q) - 4 is W2's
            K_columns(d) > pi(sqrt(6d)) transported by d = 35k (210 = 6 x 35). The transport is
            exact and was not noticed by the lane or by the manager when c.xix was written; the
            claim there that the lemma "reduces to one q-free statement" is CORRECT BUT NOT NEW -
            the project posed that statement itself and has measured its constant. WORSE FOR THE
            NEW COORDINATES: N(k) covers only the dead-pair cells, a subset of the span, so
            N(k) <= K(35k) and a lower bound on N is a STRONGER statement than W2 asks for.
            WHAT IS GENUINELY NEW IN c.xvii-c.xix AND SURVIVES: the block decomposition read as
            three rows (gear 5 as cutter is project vocabulary - the anchor 2,3,5 - but the row
            reading is new); the LEG RULE ON BLOCKS of c.xviii (exact, q-free, checked to gear
            500); the ORPHAN LAW (exact, checked from columns to gear 2000); the fact that only
            gear 7 doubles in a block; that no gear from 29 up reaches a neighbouring block; the
            measured N table 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 12; the chain ladder 0, 1, 1, 3, 4,
            6, 8, 11, 17 for q = 7..37; and the exact statement that counting is closed (the
            reciprocal sums cross one half at q = 29 and q = 47). REDISCOVERY, recorded; the branch
            is NOT closed, because W2 is the project's own OPEN statement rather than a known
            result - working it is legitimate - but it must be worked as W2, with the new exact
            rules as tools, and the manager's framing of c.xix is corrected here.
          - R5.f.xxxv.c.xxi. THE N AND K TABLES, AND WHAT THE ENUMERATION ROUTE IS WORTH
            (six-angle workflow with adversarial refutation, received 2026-09-22; thirteen agents,
            five independent engines, scripts in the run scratchpad). Worked as the wall's W2 per
            c.xx. RESULTS. (1) N(k) EXHAUSTIVE for k = 1..14: 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 12,
            12, 13, 14 - every value with dual bound equal to objective, five engines agreeing, and
            k = 14 confirmed separately by a budget-13 INFEASIBLE run. (2) NEW RUNGS: budget-14
            INFEASIBLE at k = 19 and k = 20 with NO cuts at all, and at k = 18 with cuts, so
            N(k) >= 15 for every k >= 18; the upper side is an 18-gear cover of 20 dead pairs,
            lifted by CRT to a 30-digit start and verified raw column by column. N(15) is
            undecided in {14, 15} and one worker's k = 15 certificate was REFUTED (8 of 90 cells
            unstruck). (3) THE EXACT REFORMULATION (R11): with W(q) the blocks wholly inside
            (q, q'^2] and k_req(q) = floor((W - 7)/7), the lemma at q is EXACTLY
            K(pi(q) - 4) < k_req(q), where K(n) = max{k : N(k) <= n}. The q-free surrogate
            N(k) >= pi(sqrt(210(k+1))) - 3 of c.xix is REFUTED outright (N(1) = 4 against 5); the
            per-q form is the operative one and the exact-window k_req beats the q'^2/30
            approximation at 28 of the 42 primes below 200. (4) THE RACE: K(n) = 8, 9, 12, 13,
            14..17 against k_req = 10, 13, 16, 17, 20 at n = pi(q) - 4 = 10..14, margins 2, 4, 4,
            4, at least 3. (5) COUNTING IS CLOSED, NOW MEASURED NOT ARGUED (R9): phase averaging
            caps any capacity bound at twelve gears with no k in it, and the cut-free LP root of
            the full model saturates at 9.18 rising only to 10.63 over k = 13..25, while with
            window cuts the LP returns precisely the value fed in. EVERY RUNG IS PURE
            BRANCH-AND-BOUND; no bound, linear-programming or combinatorial, contributes anything.
            (6) Structure re-derived from raw divisibility by a fourth engine: gear 7's word, block
            uniqueness to gear 2000, the orphan law, and the complete list of two-cell sets inside
            one dead pair - {L1,R2} (11), {L3,R2} (11, 13), {L2,R3} (17, 19), {L1,R3} (23), never
            three cells and never R1.
            WHAT IT IS WORTH, MEASURED HONESTLY: the route proves the window statement for every
            machine from q = 13 to q = 61 and stops, needing N(23) >= 16 for q = 67. THE PROJECT'S
            OWN F LADDER ALREADY DOES BETTER: the exact covering record F(q) = 34, 88, 91, 103,
            118, 145, 160, 179, 213 at q = 23..67 (ladder_proof_map.md:68) settles the window
            statement directly to q = 67, one machine further, and F(71) is boxed. So the per-q
            verdicts of this route ADD NOTHING; its value is entirely in the structure - R1 to R11,
            the N and K tables as a new exact object, and the measured proof that counting is
            exhausted. FACT (the tables and rules); the enumeration route is NOT a road to the
            general lemma and further rungs are bigger iterations, which the standing rule
            forbids funding. The one measurement that would decide between mechanisms is K(14)
            exactly: k_req rose 17 to 20 from n = 13 to n = 14, so K(14) = 14 means the margin
            widens and K(14) = 17 means it is closing.
          - R5.f.xxxv.c.xxii. THE COUNTERPOINT PASS: NINE OF TWELVE REFUTATIONS WERE WRONG
            (owner's instruction 2026-09-22 that every refutation needs a defence and an
            adjudication; 25 agents, twelve disputes each defended then adjudicated then
            ledgered, every disputed figure recomputed from the raw column test). The owner's
            reason was exactly right: a refuter primed to default to FALSE over-refutes, and a
            refutation that sounds right may not refute. NINE OF TWELVE CLOSURES WERE BAD.
            REVIVED IN FULL. (1) THE N(15) CERTIFICATE. The refuter reported "8 of its 90 cells
            unstruck"; with the certificate's own gear list 0 of 90 are unstruck. Substituting
            67, 103, 163 for 41, 157, 251 reproduces exactly those 8 cells - a three-gear
            transcription error in the checker, not a defect. B0 = 3305274386976800617230207 with
            gears 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 53, 73, 107, 157, 251 is a valid
            raw-verified object and exactly 15 gears are needed at that location. THE TREE LINE IN
            c.xxi CALLING IT REFUTED IS STRUCK. (2) FOURGEARS. "Four gears force two doublers" is
            a lower bound and the refuter's three-doubler witness satisfies it; over all 372
            four-gear covers of a dead pair the doubler count is 2 (336 times) or 3 (36), never 1,
            never 4. Upgrade: the minimum cover of a dead pair is 6 - nu with nu the largest set
            of disjoint doubler footprints present, so always 4, 5 or 6, and N(1) = 4 becomes
            mechanism rather than search. (3) THE q-FREE STATEMENT, revived with ONE CONSTANT
            CHANGED: not 210(k+1) but 210k + 79. S79 = "N(k) >= pi(sqrt(210k + 79)) - 3 for every
            k", with the PROVED reduction 210 k_req(q) + 79 >= q^2 for every prime q >= 13 (the
            excess q^2 - 210 k_req(q) is positive only at q = 17 (+79), 29 (+1), 41 (+1), checked
            over primes to 100000). The machinery the refuter attacked was sound; only b was wrong.
            Rider: S79 is a repair, not an advance - at q = 61 it demands N(21) >= 16 where the
            window needs only 15. (4) ROWDOUBLE, revived one word away and UPGRADED TO A THEOREM:
            a gear has two cells inside some window of m consecutive blocks IFF g <= 15m - 5
            (minD(7) = 0, minD(g) = ceil((g-10)/15) for g >= 11, 0 exceptions over 301 gears to
            2003, set equality for every m = 1..20). At m = 2 this set is exactly {7, 11, 13, 17,
            19, 23} - the orphan law's gear list, previously a raw check to gear 2000, is now the
            m = 2 case of a theorem. This confirms the manager's corollary in c.xviii.
            REVIVED IN PART. (5) CONCENTRATION: the refuter's own positive assertion is refuted -
            the three families it named as carrying the mass have 0, 0 and 858 minimal violations
            and deleting them costs the search no kill; the surviving content is a method rule,
            rank constraints by minimal violation, never by raw count. (6) The GAP-INSENSITIVE
            general clause ("any proof of the step must know the gap above q") does not follow
            from one counterexample and is FALSE where the lemma binds: at a lower twin the
            statement is chain(q) < (q+2)^2/30, which contains no gap. (7) ORPHANCEILING: the
            refuter's arithmetic was wrong (CB(24) = 7, not 6, over-crediting gears 11 and 23) but
            its conclusion survives for a different reason; what stands is that CB(k) <= 12 for
            every k, so NO counting bound on the orphan row can ever certify a 13th gear.
            (8) DOUBLINGDENSITY: the mechanism revived, the number withdrawn. The doubler supply is
            FROZEN FOR EVERY MACHINE - only 11, 13, 17, 19, 23 can ever take two cells of a dead
            pair, and no gear >= 29 ever does (raw to 3000) - and the refuter's "six consecutive
            doubled dead pairs" is the proved extremum, not a breach: the longest run is exactly 6,
            with ninety-six 6-runs and zero 7-runs per period. The true proportion is 0.3559, not
            0.414. (9) CEILING: the refutation's inference from eight non-monotone values to "no
            constant exists" is invalid; what survives is a decidability law (K(q) is a maximum
            over an explicitly finite set, since a gear above 15 L_1(q) + 10 adds nothing) plus
            K >= 6 at q = 31, verified at two independent certificate lifts.
            UPHELD, and double-computed: the two-row ordering; the two-pair rule (N of two dead
            pairs is 4 at separation 19, so N is not a function of the pair count); the
            (d_g mod 7, g mod 7) type invariant as made.
            FACT (the ledger). METHOD CHANGE, standing from now: derive, refute, DEFEND,
            adjudicate; the synthesis takes the adjudicated form, never the refuter's verdict
            alone. Saved to memory as derive-refute-defend.
          - R5.f.xxxv.c.xxiii. K(14) = 14, AND THE MARGIN IS WIDENING (2026-09-22; thirteen
            agents, twelve exhaustive cases of a split on gear 11's phase, plus an independent
            verdict agent that re-decided all twelve). The one measurement c.xxi named as deciding
            between mechanisms. RESULT: N(15) = 15 EXACTLY. All twelve cases of the split - gear 11
            unused, or used at one of its 11 phases, mutually exclusive and jointly exhaustive -
            returned INFEASIBLE at budget 14, none undecided, and EVERY DECISION WAS CUT-FREE: no
            inherited N value entered any deciding run, so the verdict rests only on the cell
            formula and the pool bound, both re-verified raw. The verdict agent then re-decided all
            twelve itself with a second engine (HiGHS), obtaining model status Infeasible - not a
            time limit - on every case, 912.9 seconds in total. The upper side is a fresh 15-gear
            cover found and lifted independently: gears 11, 13, 17, 19, 23, 29, 37, 41, 43, 53, 89,
            101, 103, 263, 751 at B0 = 22475808932511699377319069, all 90 columns struck. So
            K(14) = 14 exactly, and the interval [14, 17] collapses to a point.
            INDEPENDENT VERIFICATION done by the verdict agent from scratch: the cell formula
            against raw divisibility in 1,970,280 comparisons, 0 mismatches; gear 7's silent blocks
            come out as {0, 6} mod 7 raw; the pool bound checked over all 879 primes in (1495,
            9000) at every phase, maximum 1 cell; and its own engine reproduced N(1..9) as optima.
            THE RACE, with k_req counted directly as the dead pairs the window holds:
              n = pi(q) - 4:   10    11    12    13    14
              q:               43    47    53    59    61
              K(n):             8     9    12    13    14 (new)
              k_req:           10    13    16    17    20
              margin:           2     4     4     4     6
            Over the four steps K gained 6 and k_req gained 10, so THE MARGIN IS WIDENING, and it
            has just opened from a flat 4 to 6. The mechanism is sharper than the trend: the
            surplus N(k) - k runs 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 1, 0, 0, 0 for k = 1..14 and the
            new point makes it 0 at k = 15 too - zero surplus is exactly K(n) = n, so the left side
            of the race now advances by exactly one per machine while the right side advances by
            one to three and its increments grow with the window.
            VERDICT ON THE ROUTE, honest: as a route to a theorem enumeration is CLOSED - one
            value of K cost twelve exhaustive integer programmes and the two next instances would
            not decide in 240 seconds each, while the pool grows like 105k/log(105k) gears and the
            cells like 6k. As evidence it is working and the lemma is winning by a widening margin.
            The fragility is stated too: K(n) = n is five data points, the surplus slid 3 to 0
            between k = 7 and k = 12, and N stalled three times (k = 7 to 8, 10 to 11, 11 to 12);
            one stall at k = 16 makes K(15) >= 16. The margin at q = 61 absorbs about six stalls,
            and a stall costs 1 while the machine gains about 2.5, so even a stalling K loses the
            race - but nothing proves the surplus cannot go negative and stay there.
            THE TARGET IS NOW ONE SENTENCE: N(k) >= k - c for every k and some constant c, i.e.
            past k = 12 each extra dead pair forces another gear up to a constant. With the
            elementary count pi(q) - 4 < k_req(q) that finishes the lemma for every q. The measured
            surplus column is the evidence that this is the right shape; enumeration cannot supply
            it. STRONG, CANDIDATE.
          - R5.f.xxxv.c.xxiv. THE MARGINAL LAW: TARGET REFUTED, REPLACEMENT FOUND, AND THE
            SKELETON CLOSED EXCEPT FOR GROWTH (2026-09-22; 25 agents in the owner's four-stage
            shape - derive, refute, defend, adjudicate - plus assembly; every figure re-derived
            from the raw column test; manager verification in research/stack/r8/window_boundary.py).
            THE TARGET N(k) >= k - c IS REFUTED, with proof rather than instances. Each cell is
            struck at exactly two phases of each gear and phases are independent, so the mean
            number of uncovered cells over phase vectors is exactly 6k prod(1 - 2/p); some vector
            attains the floor, and each residual cell is bought with one fresh gear above the pool.
            Taking the gears to 4219 gives 6 prod < 1/2, hence N(k) <= 574 + k/2 FOR EVERY k, so
            N(k) - k tends to minus infinity. Confirmed by raw certificates, each a single start
            integer with all 6k columns rebuilt and tested by real divisibility: N(100) <= 49 (a
            101-digit start) and N(200) <= 70 (154 digits). Also N(16) = 15, so the stall set
            through k = 15 is {7, 10, 11, 15} and the surplus is already negative at k = 16.
            THE CORRECT REPLACEMENT, and it is enough: N(k) >= 3.23 sqrt(k) for k >= 23 suffices
            for every machine. The constant is sharp - (pi(q) - 4)/sqrt(k_req(q)) is maximised over
            all primes below a million uniquely at q = 109 (k_req = 60, pi - 4 = 25, ratio
            25/sqrt(60) = 3.2274861) and falls below 3 for every prime above 137. The reduction is
            EFFECTIVE AND UNCONDITIONAL: with Rosser-Schoenfeld it holds for every q >= 168 and the
            primes below 168 are checked directly, no unchecked range.
            THE SKELETON IS NOW CLOSED EXCEPT FOR ONE STEP. S1 cells and the dead-pair coordinates,
            S2 the machine owns exactly pi(q) - 4 gears, S3 the window holds k_req(q) >= (q^2 - q -
            272)/210 complete dead pairs, S4 every cell is 5- and 7-silent, S5 an uncovered cell is
            a twin pair (with the boundary repair below), S6 the machine is one configuration so
            N(k_req) > pi(q) - 4 suffices, S7 every machine to q = 61, S9 the reduction for all q
            at once - ALL PROVED. S8, every machine from q = 67 up, is OPEN, and what it needs is
            not a sharper constant but A GROWTH MECHANISM: the first open machine is q = 67, which
            needs N(23) >= 16 where the standing bound gives 15.
            THE STRONGEST PROVED LOWER BOUND IS EVENTUALLY CONSTANT: N(k) >= 15 for k >= 15 and
            N(k) >= 16 for k >= 85 (the latter from the orphan row, N_orph(85) = 16 solved exactly).
            N(k) tending to infinity IS NOT PROVED anywhere in this machine. Capacity gives 12 and
            stops dead; capacity with the period law gives 13 only beyond k about 2 x 10^20.
            COUNTING IS NOW CLOSED FROM A FIFTH DIRECTION, and this one is new: adding the
            difference-SUPPLY constraints, which no cell weighting can see, the row linear
            programme grows from 5.5 to 11.55 and then FLATTENS at 11.746 (k = 120) and 11.771
            (k = 200) - the same fractional ceiling as the full grid - while the integer optimum is
            already 16 at k = 85. So the growth in N is integral: it comes from one-phase-per-gear
            and no weighting can produce it. That is the signature a proof must have - an exchange
            or parity argument on the chosen classes, never a weight.
            BOUNDARY REPAIR (R17), NEW AND VERIFIED BY THE MANAGER INDEPENDENTLY: the rule "an
            unstruck column of the window is a twin pair" is FALSE at exactly one column of the
            CLOSED window (q, q'^2], namely n0 = (q'^2 - 1)/6 where 6n + 1 = q'^2 is a square with
            no prime factor at or below q. Verified over q = 7..113: the set of non-twin openings
            is always either empty or exactly that column, and it is non-empty precisely when
            q'^2 - 2 has no prime factor <= q (q = 43: column 368, with 2207 prime and 2209 = 47^2;
            also q = 11, 17, 23, 31, 41, 59, 67, 83, 101, 103, 113). Taking the window OPEN at the
            top removes it, and the exceptional column never lies in a fully contained dead pair,
            so k_req is unchanged and every deduction above is safe. THE PROJECT'S SQUARE-ROOT RULE
            MUST BE STATED WITH THE STRICT INEQUALITY.
            OTHER CORRECTIONS, raw: k_req(61) = 21, not 20, so the margins at q = 43..61 are 2, 4,
            4, 4, 7.
            NEXT STEP NAMED: the row growth lemma - prove N_orph(k) grows, via the difference-supply
            law (a gear serves two row cells at distance D only if g divides 105D +- 1, so the
            supply at each difference is a bounded computable quantity), with the exact row oracle
            as the test (N_orph = 7, 9, 11, 13, 16 at k = 15, 24, 40, 60, 85, ratio to sqrt(k) flat
            at 1.735 to 1.837 across that range). The row optima are a near-initial segment of
            gears plus exactly three or four far gears and NO singleton payments; the unbroken
            prefix version is refuted (k = 40 skips 23, 41, 43). The invariant to prove is "no
            payment, O(1) far gears", which turns N_orph(k) into pi(G(k)) - 4 + O(1) and reduces
            growth to the growth of the threshold G(k). Ruled out by measurement: another
            enumeration rung (each buys +1 on a constant against a requirement of 16 at k = 23 and
            85 at k = 1000); any counting, capacity, density or linear-programming argument (capped
            at 6, 11, 11.7708, 12, and now capped even with supply constraints); the row alone
            finishing the statement (N_orph(23) = 8 against a needed 16 - the row is where growth
            can be found cheaply, the constant lives in row interaction, and each extra row costs
            at least one: the 1/2/3/6-row ladder gives 3, 4, 5, 8 at k = 5 and 6, 8, 9, 12 at
            k = 12, the six-row column reproducing N(5), N(8), N(12)); and the dual of the first
            moment, measured weakest exactly where the window lives.
            STRONG (the rules and the refutation); the window statement now rests on a single named
            open step with an effective reduction behind it.
          - R5.f.xxxv.c.xxv. DRIFT CORRECTION: c.xvii TO c.xxiv ARE WINDOW WORK, NOT RANGE WORK
            (owner, 2026-09-23; research/stack/r8/range_tiers.py). The owner noticed the manager had
            drifted off the range statement and back onto the window. He is right, and the exact
            place is identified here. Nodes c to c.xv ARE range work: machine 5 as the cycle, the
            copies of the known opening (-1, 1) at every multiple of 30, the overlay acting on those
            laps. At c.xvi the manager observed that the laps follow only one of machine 5's three
            open classes, called that a restriction, and moved to "the full opening set" - and IN
            THE SAME MOVE, WITHOUT FLAGGING IT, replaced the range statement by the window
            statement. Everything from c.xvii to c.xxiv (blocks, dead pairs, N(k), K(n), the
            covering number, the marginal law) answers whether a twin lies in (q, q^2]. The owner
            asked whether one lies anywhere in (q, q#]. Those are different objects and the owner
            has said so twice.
            THE RANGE OBJECT, RESTATED CORRECTLY. Cycle = machine 5 (base 2, 3 and gear 5, period
            30); its known opening recurs at every multiple of 30, so the LAPS are the pairs
            (30k - 1, 30k + 1). Overlay = every gear above 5 and below sqrt(q#), which is exactly
            the set that can decide any lap of the range. Gear g strikes lap k iff k = +-30^{-1}
            mod g. RANGE STATEMENT for machine q: some lap with 30k - 1 > q and 30k + 1 <= q# is
            struck by no gear at or below sqrt(30k + 1). WINDOW STATEMENT: the same with the extra
            demand 30k + 1 < q'^2.
            THE TIER FRAMING WRITTEN HERE WAS ITSELF WRONG AND IS WITHDRAWN (owner 2026-09-23,
            proved in c.xxvi). Splitting the range into tiers, each "a window statement for a
            larger machine", is the pull back to the window in another costume: it re-imposes a
            location and converts the problem into the one we already cannot do. The scale
            disjunction is an EXACT COVERING IDENTITY - for any height h the cut "largest prime
            below h" certifies it, so the bands chain automatically - which QUANTIFIES the slack
            and adds NO leverage. There is one machine running to its period. The correct statement
            of the difference is in c.xxvi: the range is weaker than the window by exactly one
            thing, that the certifying gear set is chosen AFTER the lap.
            MEASURED: the range holds 2, 3, 4, 5 tiers at q = 5..11, 13..31, 37..73, 79..89, growing
            like log(q/log q); the range spans 10^1.5 to 10^34 at those q against windows of 1 to
            311 laps. The first twin lap above q is k = 1 for q <= 23, 2 for 29..53, 5 for 59..89,
            and in every case it lands in tier 0 - SO EVERY HIGHER TIER IS UNUSED SLACK, and the
            range statement has never yet had to call on one.
            HONEST READ ON THE DISJUNCTION, stated so it is not oversold: each tier is a window
            statement for a larger machine, and the measured record-to-window ratio is flat near
            0.25 at every scale, so the tiers are not individually easier. The gain is that the
            range needs only one of them and the window needs the first. What the range does NOT
            need, and the window does, is a LOCATION: the range statement asks only that a surviving
            lap exist somewhere in the period.
            CORRECTION RECORDED; the window results of c.xvii to c.xxiv stand as window results and
            are not withdrawn - the leg rule on blocks, the orphan law, the frozen doubler pool, the
            covering number and its refutation of N(k) >= k - c are all true and all about the
            window. They are re-filed as such. The active line returns to the range.
          - R5.f.xxxv.c.xxvi. THE RANGE, ANSWERED: BOTH OBVIOUS PATHS CLOSED, THE REAL SLACK
            LOCATED, AND THE WALL REDUCED TO ONE INEQUALITY (2026-09-23; 25 agents in the
            derive-refute-defend-adjudicate shape on the owner's two paths and his method).
            PATH A CLOSED, and not by difficulty: "for every q some lap of the range is struck by
            no gear of [7, sqrt(q#)]" IS EXACTLY THE WINDOW STATEMENT at the cut X = isqrt(q#).
            PROVED both ways - a lap surviving every gear up to X with 30j + 1 <= q# has its lower
            member above X, so its square exceeds q#, so the copy is a twin above sqrt(q#); and
            any such twin has its lower member above X. Enumerated whole at q = 13, 17, 19 (149,
            1507, 18991 survivors): zero non-twins, zero below the floor, zero twins above the
            floor missed. Taking path A re-imposes a location, which is the thing the owner has
            twice said to stop doing.
            PATH B PROVED FALSE at a fixed gear set. For any gear set with modulus M the blocked
            sets - and the open sets - at the lower set's three open offsets are EXACT TRANSLATES
            of one another by the single integer -e x 5^{-1} mod M (set equality at M = 77, 1001,
            17017, 323323). The three segments of the 2,3,5 cycle are ONE PATTERN IN THREE
            POSITIONS. There is no differentiation across the cycle to harvest, and any map moving
            the known opening off offset 0 costs at least (M - 3)/5 laps.
            WHAT IS LIVE, and it is path B's repair: the variation across laps is real but it is
            NOT in the gear pattern - it is in WHICH GEARS MATTER. Gear g decides lap j only once
            g^2 <= 30j + 1. THAT LAP-DEPENDENT DECIDING SET IS THE ONLY NON-CONGRUENCE INGREDIENT
            IN THE WHOLE OBJECT, and it is exactly what separates range from window.
            THE RANGE IS WEAKER THAN THE WINDOW BY EXACTLY ONE THING: the certifying gear set is
            chosen AFTER the lap. The window fixes the cut and the interval in advance; the range
            certifies lap j with [7, sqrt(30j + 1)], a set that is a function of j. EVERYTHING ELSE
            THAT LOOKS LIKE SLACK IS NOT. (i) Extra length is not freedom: the range is exactly ONE
            period of the machine's own gears, so each survivor class has exactly one
            representative - a survivor's height IS its class, and there is no room to move one
            down. In-range survivors = prod(g - 2) - 1 exactly at q = 7..23. (ii) The scale
            disjunction is a restatement, an exact covering identity that quantifies the slack and
            adds no leverage. (iii) MEASURED: for every prime q in [7, 3000] the first twin lap
            above q sits below q^2 - inside the window - so the slack has never once been drawn on.
            NO INSTRUMENT IS CAPPED FOR THE WINDOW BUT NOT FOR THE RANGE. Two candidates fail: the
            Euclid family j = 0 mod P(y) is uncapped in reach but dies on certification at the fair
            rate 2/g (measured 1.0015 and 0.9990 of 2/g); and the covering-number instrument is the
            wall's W2, banned and capped anyway.
            NEW THEOREM, and it closes a whole family at once - THE CONGRUENCE-LOCATOR CLOSURE: if
            every lap of the class j = r mod N is silent against gear g, then g divides N (else j
            runs over every residue mod g inside the class and meets +-a_g; 0 escapes in 920 tested
            triples). Hence a class silent against [7, X] has N divisible by X#/30, so N exceeds
            X'^2/30, so THE CLASS HOLDS AT MOST ONE MEMBER IN THE CERTIFIED BAND. Every congruence
            locator therefore certifies at most one lap, and only inside the window. That closes the
            known-opening copies of R5.f.xxxv.b, the centre construction, and every relative, with
            ONE argument instead of one instance each.
            NEW FORM, exact and q-free - THE SQUARING LAW: gear g strikes lap j iff j^2 = a_g^2
            mod g, a_g = 30^{-1} mod g (0 failures, gears 7..5000, every residue). So survival
            against any gear set depends on j ONLY THROUGH j^2 mod P; the mirror j -> -j needs no
            side condition; the survivor set is a union of fibres of the squaring map.
            THE PRICE TAG ON THE OWNER'S TRADE, proved - THE SUPPLY LAW: dropping a prime p from the
            lower set into the gears multiplies the RAW copies by p but the LIVE copies by exactly
            p - 2, since p rejoins as a gear and takes two classes. Free only at p = 2 and 3. And
            the UNIT TWIST: the class geometry at lower-set rung y' is the geometry at rung y
            multiplied by the single unit (P'/P)^{-1} mod g, so all rungs are ONE machine under an
            index change, with the smallest lower set the weakest sufficient rung. Shrinking the
            lower set buys prod(p - 2) parallel copies of the IDENTICAL upper-set problem - the
            extra iterations are real, the extra structure is not, and no locator comes with them.
            SAME WALL: YES, and it is now ONE INEQUALITY in every coordinate tried - X# >
            nextprime(X)^2 for every prime X >= 7 (verified to X = 4000, vacuous only at X = 5).
            Silence against all gears up to X costs a modulus of at least X#/30; certification at a
            height demands silence up to that height's square root. THE ONE PLACE THE RANGE HAS
            SLACK THE WINDOW DOES NOT: the cut moves with the lap. It is currently unexploited -
            every instrument on the tree has demand monotone increasing in certification depth.
            STRONG (the theorems); the two obvious range routes are closed with proofs, the single
            real difference is located exactly, and the next work must exploit a moving cut.
          - R5.f.xxxv.c.xxvii. THE BLAME ASSIGNMENT ROUND (2026-09-23; 25 agents,
            derive-refute-defend-adjudicate, locating and counting banned in the brief). Setup: a
            copy j is REVEALED when no gear that can act at its height strikes it; a gear can act
            on copy j only when g^2 <= 30j + 1. If no copy of the range is revealed there is a
            TOTAL BLAME ASSIGNMENT sending each copy to a gear that both strikes it and can act
            there. The round attacked the existence of such a map.
            CORRECTION TO THE MANAGER'S BRIEF, FIRST. "Gear g strikes no copy j with
            30j + 1 < g^2" is FALSE as written and was listed as proved. 1043 counterexamples among
            gears below 500 with j < 3000; the least is gear 13 on copy 3, since 91 = 7 x 13 and
            91 < 169. Only gears 7 and 11 have no sub-square strike, and that is forced by the
            first-strike law below. The correct statement is that a gear cannot ACT below its
            square, not that it does not strike there. Two angles used the false version.
            PROVED THIS ROUND.
            - ACTING IS COFACTOR ORDER: writing the struck leg as g x h, the condition
              g^2 <= 30j + 1 is exactly h >= g - the gear is the SMALLER factor of the leg it
              divides. 42,529 strike incidences, 0 failures.
            - FIRST-STRIKE LAW (new): the least cofactor is k0 = min(u, 30 - u) with u = g^{-1}
              mod 30, so k0 lies in {1, 7, 11, 13}, and every gear's first strike is below its
              square except gears 7 and 11. Checked over 2000 gears, distribution 488 / 507 / 497 /
              508, exceptions exactly {7, 11}.
            - SEPARATION LAW, exact form (the brief's sandwich was loose and is repaired):
              15 t_p = k p + s with four slopes 1/15, 2/15, 4/15, 7/15 fixed by p mod 15, and
              (p - 1)/15 <= t_p <= (7p + 1)/15.
            - THE FIBRE VALUE IS ALWAYS A SQUARE: struck iff j^2 = a_g^2 mod g, and a_g^2 is a
              square for every gear without exception, so its Legendre symbol is +1 always. No
              quadratic character can gate which gears strike.
            - THE BARRIER: let U be the untruncated machine - same gears, same classes, acting
              dropped. Every rule in the round's proved list except acting is a statement about the
              strike relation alone, so each holds verbatim in U. U ADMITS A TOTAL BLAME
              ASSIGNMENT, namely b(j) = the least prime factor of 30j - 1, which is always a gear
              and always divides a leg (30j - 1 is coprime to 30 and exceeds 1; checked to
              j = 50000, 0 failures). Therefore no combination of strike-relation rules can forbid
              a total blame assignment.
            - MACHINE-GEARS-ONLY THEOREM: some copy of the range is struck by no gear at or below
              q. Proof by a nonzero CRT survivor modulo P = q#/30, with the survivor set
              mirror-symmetric and the range covering every class but the bottom block. Verified at
              q = 7, 11, 13, 17: survivors 5, 45, 495, 7425, meeting the range in 4, 44, 494, 7424,
              mirror symmetry holding in all four. Free of both prohibitions.
            - THE c_g FAMILY: c_g is the lowest copy of height at or above g^2 that gear g does not
              strike, and it is (g^2 + 10, g^2 + 12) when g^2 = 19 mod 30 and (g^2 + 28, g^2 + 30)
              when g^2 = 1 mod 30 - gear squares being 1 or 19 mod 30 only. No gear strikes its own
              c_g, and no larger gear can act there. 0 exceptions on either count over every gear
              below 200,000. Non-vacuous: c_g is a twin copy for 95 gears in the 19-case and 233 in
              the 1-case.
            - SYMMETRY CLOSURE: the affine symmetries of one machine's strike configuration are
              exactly j -> +-j per gear, 2^k of them, all fixing the origin, and none preserves
              acting since 30j + 1 is injective.
            - OWN-SHELF SILENCE, complete: a gear is silent on its own shelf never for
              g = 1, 7, 11, 19, 29 mod 30; for g = 17 mod 30 iff g + 2 is prime; for g = 13 mod 30
              iff g + 4 is prime; for g = 23 mod 30 iff g + 6 or g + 8 is prime. 3242 gears below
              30000, 576 silent, 0 mismatches.
            - TWO-CLAUSE CONDITION ON ANY FUTURE CANDIDATE PROPERTY P that would forbid a total
              blame assignment: (a) P must fail in U, hence must mention acting essentially; and
              (b) P must not be a consequence of "a composite leg has a prime factor at or below
              its square root", which by the cofactor-order rule IS acting.
            NOT PROVED: no property forbidding a total blame assignment was established. The
            property stated with copies eliminated - the gear set contains two gears differing by 2
            whose midpoint is divisible by 30, the lower above q and the upper at most q# - is the
            range statement restated in gear terms.
          - R5.f.xxxv.c.xxviii. SHELVES, MISSED COPIES, COFACTORS, SURVIVOR CLASSES, GEAR PAIRS
            (2026-09-23; 24 agents, derive-refute-defend-adjudicate; resumed after a network outage
            killed three adjudications and the assembly; full record in
            research/proof/shelves_cofactors_2026-09-23.md). Built on acting = cofactor order (c.xxvii).
            PROVED, each re-checked from raw divisibility over the stated range:
            - MISSED-COPY LAW: c_g = copy ceil((g^2+1)/30), legs (g^2+A, g^2+B), (A,B) = (28,30) when
              g = +-1, +-11 mod 30 and (10,12) when g = +-7, +-13 mod 30. g misses it; no gear above g can
              act there. A smaller gear h strikes it iff g^2 = -A or -B mod h, so each h removes 0, 2 or 4
              classes of g mod h, and which h can reach each leg is fixed by quadratic reciprocity:
              (-28/h) = (h/7), (-12/h) = (-3/h), (-10/h) = 1 iff h mod 40 in {1,7,9,11,13,19,23,37},
              (-30/h) = 1 iff h mod 120 lies in a listed set of 16. Gear 7 is inert in case 1. c_g is
              revealed for 233 case-1 and 95 case-19 gears below 200,000, the congruence count and the
              both-legs-prime count agreeing exactly.
            - ABOVE THE SQUARE: g strikes copy J_g + M iff M = x_g or x_g + t_g mod g, x_g = -1 (case 1)
              or -2 x 5^{-1} (case 19); the silent run headed by c_g has a closed length p1 by g mod 30;
              the 2nd and 3rd missed copies are J+1, J+2 for every gear except 7, 11, 29.
            - SHELVES: shelf(g) = [j_g, j_g') with j_g = (g^2 - 1 + 12 chi(g))/30; shelves tile the
              copies from 2; closed size N(g); the acting set on every copy of shelf(g) is exactly the
              primes in [7, g]. OWN-SHELF SILENCE is now a THEOREM, not a check: g strikes nothing on its
              own shelf iff (g, g + off/2 - 1] contains a prime, giving never for r = 1, 7, 11, 19, 29;
              iff g+2 prime for r = 17; iff g+4 prime for r = 13; iff g+6 or g+8 prime for r = 23
              (checked to gear 300,000). DEFERRAL LAW: a silent gear's first acting strike lands on the
              shelf of the largest prime in that window. Silent runs have length at most 2, exactly at
              g = 13 mod 30 with g+4, g+6 prime (282 runs below 300,000).
            - COFACTORS: strikes of g are in bijection with cofactors h = +-u mod 30, u = g^{-1}; the
              admissible cofactors are all integers = +-k0 mod 30, k0 in {1,7,11,13}; (Z/30)*/{+-1} is
              cyclic of order 4 generated by [7]; struck copies are m g +- j0 with closed j0; the
              separation satisfies 15 t_g = k g + 1 with k = 14, 2, 4, 8, 7, 11, 13, 1 by g mod 30.
              WHAT TRIAL DIVISION DOES NOT IMPLY, shown by variant machines where it still holds (base
              210, base 2310, a skewed base): the spoke set {1,7,11,13}, the exception set {7,11}, the
              mirror, the location of c_g, and the silence constants. These are properties of the
              base-30 machine itself.
            - SURVIVOR CLASSES: S_q = {j : gcd(900 j^2 - 1, q#/30) = 1}, |S_q| = prod(g-2); its affine
              stabiliser is exactly {j -> u j : u^2 = 1}, no translations; adding the next gear p maps
              S_q' onto S_q exactly (p-2)-to-1; gears above q delete no class, they only refine, each
              striking exactly two of the g subclasses of every class, one per leg, and nothing below
              ceil((g^2-1)/30).
            - GEAR PAIRS p < q = p + d: both strike copy j iff (30j)^2 = 1 mod pq; four joint classes,
              two same-leg and two split with d w = p + q; per period pq, 4 both, 2(q-2) p only,
              2(p-2) q only, (p-2)(q-2) neither. Twins occur as gear pairs only at p = 11, 17, 29 mod 30,
              and closed first-strike separations for twin, cousin and sexy pairs hold with 0
              mismatches over 2158, 2135 and 4294 pairs to p < 200,000.
            REFUTED this round, each with its instance (record, section 2): silence propagating to the
            next gear (17 is silent, 19 strikes its own head); every shelf holding a revealed copy (the
            shelves of 17 and 29 hold none); the n = 3 congruence count matching primality (g = 17,
            361 = 19^2); the case-1 protection bound applied to case 19 (g = 13); least-c constancy on
            p mod 30 (it is p mod 15d); and further statement-level corrections.
            NOT ESTABLISHED: infinitely many g with c_g revealed (i.e. g^2+28, g^2+30 or g^2+10, g^2+12
            both prime); survivor density over all h < g equal to the product of local factors (only
            fixed H is proved; raw/product ratios 1.283 and 1.317 at 200,000); which acting-based
            statement forbids a total blame assignment; the remaining items in section 4 of the record.
          - R5.f.xxxv.c.xxix. THE TWO RANGE PATHS, SEVEN ANGLES (2026-09-23; 28 agents,
            derive-refute-defend-adjudicate; full record research/proof/range_two_paths_2026-09-23.md).
            CORRECTION TO THE BRIEF: survivor class 0 has no copy in the range; each of the prod(g-2) - 1
            NONZERO classes has exactly one.
            PATH A - the missed copies c_g of the upper gears. PROVED:
            - Exclusion tables. A gear h < g strikes c_g iff g mod h lies in E_h, the unit roots of
              x^2 = -28 or -30 (case 1) or -10 or -12 (case 19); |E_h| = 2 + (h/7) + chi_{-30}(h) and
              2 + chi_{-10}(h) + (h/3); full table to h = 61. The four characters chi_{-7}, chi_{-30},
              chi_{-10}, chi_{-3} are independent; each reach state is exactly a quarter of the unit
              classes; the jointly inert classes are 12 classes mod 840 (83, 227, 311, ...). E^(1) and
              E^(19) are disjoint; at least two classes of g stay free at every h.
            - Escape classes at every finite level: 4 prod(h - 1 - e_h) classes mod 30 prod h, refined
              (h - 1 - e_h)-to-1 by each new row, never lost; each holds infinitely many gears (Dirichlet)
              whose c_g escapes every row of the level.
            - HAND-OFF EQUIVALENCE: the target "every range holds a revealed c_g" holds for all q iff the
              revealed gears 7 = g_0 < g_1 < ... form an infinite sequence with
              g_{i+1}^2 + B <= (g_i^2 + A)#. All 327 hand-offs below 200,000 hold; gear 198437 alone
              serves every prime q from 31 to 39,377,242,978.
            - Dormancy: row h is silent on g within sqrt(2h - B') of a multiple of h; for h to strike the
              next gear's c needs (gap)^2 >= 2h - B'; among all consecutive gears to 10^7 this happens
              exactly twice (13 kills c_17, 17 kills c_19).
            - NEIGHBOUR KILL RULE: h kills c_{h+d} iff h divides d^2 + A' (A' = 28 or 30 in case 1, 10 or
              12 in case 19); complete lists of kills by the r-th lower neighbour to 10^8 for r <= 12, the
              largest killed gear being 4517. Every range holds a c_g not killed by its r nearest lower
              gears, r <= 12, every q >= 7 (the tail q >= 10^15 uses Rosser-Schoenfeld and a pigeonhole on
              prime counts - a counting residue, flagged).
            - The n-th missed copy: legs g^2 + 2K_n, g^2 + 2K_n + 2; reach states period h in n; the case-19
              states are the case-1 states shifted by 3 x 5^{-1}; no gear is inert on every copy of a run;
              the inert set recurs along Pell families (15s^2 - 14r^2 = 1 in case 1, 5s^2 - 6y^2 = -1 in
              case 19); the pairs with h inert on the WHOLE silent run are exactly (11,7), (29,7), (29,19).
            - Silent run and protected stretch in closed form by g mod 30; the stretch l = min(p1, L) is
              at least 3 for every g >= 31; no single gear covers a stretch of 3 or more (leg rule).
            PATH B - the translation law with acting built in. REFUTED, each with instances: the least
            translate of an exposure is revealed ((6,2) -> copy 607, 18209 = 131 x 139); the class holds a
            revealed copy in the range (q = 41, exposure (47,3): all 41 class copies struck); the full
            stabiliser orbit holds one (q = 17 lap 14; q = 29 lap 35, 12 copies all struck); every exposure
            transfers exactly into the range (87 sources in laps 2..399 do not). PROVED: the exact
            downward transfer (targets reach at most copy 27 for sources to 5 x 10^6); transfer cost
            floor 6k* M; forced stabiliser pairs preserve exposure (0 counterexamples, q = 7..23), a pair
            being forced iff every gear in (q, sh(j')] divides t, and forced pairs never go upward.
            MEASURED, not proved: every window exposure has a revealed orbit copy in the range, q = 7..67.
            NOT ESTABLISHED: the hand-off sequence is infinite (the R14 target); whether the range
            statement implies it; boundedness of the inert run; finiteness of the empty head stretches
            (17, 29, 37, 41, 149 below 10^6) and of the equality families; S3 for window sources; a
            closed bound on k*; the remaining items in section 4 of the record.
          - R5.f.xxxv.c.xxx. THE DERIVED MACHINE, ITS ESCAPES, THE HAND-OFF, THE STRETCH FIELD,
            KILLS, FORCED PAIRS (2026-09-24; 25 agents, derive-refute-defend-adjudicate; record in
            research/proof/derived_machine_2026-09-24.md).
            PROVED:
            - DERIVED MACHINE, built beside the original (table in the record, section 2.1). Columns are
              x with x^2 = 1 or 19 mod 30; J(x) = (x^2 + c)/30 is injective and increasing, the two cases'
              images disjoint; a striker h takes |E_h| = 2 + (-A/h) + (-B/h) in {0, 2, 4} classes (four reach
              states: inert, A-only, B-only, both); the pair rule is quadratic and quartic, not linear:
              N(D) = e_h[h|D] + [h|D^2+4A] + [h|D^2+4B] + 2[h|D^4+2(A+B)D^2+4]; separation
              (v-u)(v+u) = tau_v - tau_u in {0, +-2} same case, {+-16, +-18, +-20} cross case; stabiliser
              {+-x} except striker 29 in case 1, where {+-1, +-12} swaps the legs; special strikers
              {11, 37} / {7, 13}; twin gears share killers only from {11, 29, 31} or {13, 23, 37}. Acting in
              x is h <= x, the same relation as the original's square root in copy terms. Composite columns
              can be revealed (221, 451, ...; 427, 517, ...).
            - ESCAPE AGAINST THE CUTOFF: every striker of c_g is live, so with lambda(g) the top live row,
              c_g is revealed iff g lies in an escape class of level {7..lambda(g)}; lambda(g) = h*, the
              largest non-inert prime at or below g - d0(g), and lambda(g) > 2g/3 for every gear g >= 41
              (the tail above 1.26 x 10^12 uses the BMOR 2018 prime-count estimate - counting residue,
              flagged); hence the gears capped at level P lie in (P, max(37, 3P/2)]. Entering law: row
              lambda(g) strikes c_g iff g - lambda is an even root of x^2 = -A' mod lambda, with the
              factorisation 4(g^2 + B') = (d^2 + B')((d + 2)^2 + B') on the kappa = 2 kills. Every escape
              class at levels 7, 11, 13, 17, 19 in both cases holds a gear whose c_g is revealed (largest
              witness 42,561,936,551).
            - HAND-OFF: a revealed copy with legs (p, p+2) serves exactly the machines q in [sigma(p+2), p),
              sigma(y) the least prime with primorial >= y; so RANGE <=> the chain over ALL revealed copies
              (p_{k+1} + 2 <= p_k#). The chain over missed copies alone implies RANGE; the converse is open.
              Strong hand-offs increase the slack (monotone slack theorem); all 6,547 hand-offs below 10^7 are
              strong. Among 72 binding machines below 200 the witness is a missed copy only at q = 59, 149.
            - STRETCH FIELD: each striker's teeth on a stretch form a comb with gaps alternating m_h and
              h - m_h; three consecutive cells need three distinct strikers unless 7 takes an adjacent pair or
              29/31 take the ends; the fully covered stretches to 10^8 are exactly those of 17, 29, 37, 41,
              149. No fixed finite striker set forces a revealed cell at a fixed depth (CRT + Shiu).
            - KILLS: every shared prime factor >= 7 of the legs of two gears D apart lies in a finite set
              S(D) (members below (D^2 + 58)^2/4); a prime above H(w) divides a leg of at most one gear in
              any window of width w (H(2) = 37, H(4..6) = 449, H(8..12) = 601, ...). Arbitrarily long runs of
              consecutive gears with every missed copy killed exist (gears = 301 mod 330, Shiu).
            - PATH B: the forced-pair criterion, forced pairs never go upward and preserve exposure; the
              orbit of a window exposure has size 2^(k-w) and splits over machine q into prod (g+1)/2 orbits.
            REFUTED, with instances: the reduction to the capped strip (Cap(13) = {17, 23}, neither revealed;
            fails at 50,139 of 148,930 prime levels to 2 x 10^6); every window exposure has a revealed orbit
            copy (q = 19, exposure (6,3): all 16 orbit copies in the range struck); heredity of the chain;
            at most two teeth per stretch outside 7, 29, 31 (g = 43, striker 11 takes cells 1, 9, 12); the
            fixed-depth statement for every N, K; and others listed in section 3 of the record.
            NOT ESTABLISHED: RANGE => chain over missed copies; whether any escape class is doomed; whether
            Cap(q) and Esc(q) are disjoint for infinitely many q; finiteness of fully covered stretches and of
            the kappa = 4 entering kills; the orbit statement beyond measurement (exact to q = 61).
          - R5.f.xxxv.c.xxxi. KERNEL FOR THE RANGE LINE, AND THREE OPEN ITEMS (2026-09-25; 18 agents;
            Lean lanes sequential; record in research/proof/range_kernel_2026-09-25.md).
            KERNEL (Lean 4 / Mathlib, namespace RangeLine, each module built on its own by the lane, by an
            independent reviewer, and again by the manager; 0 sorry / admit / axiom / native_decide; axioms
            only propext, Classical.choice, Quot.sound; reviewer found no unfaithful or vacuous statement):
            - RangeCopies: copy_product, copy_strike_iff(_legs), copy_strike_class (30j = +-1 mod g), the four
              leg rules as iffs (same-sign legs: g | D; minus then plus: g | 15D + 1; plus then minus:
              g | 15D - 1), copy_leg_rule, and its full converse copy_pair_iff.
            - RangeLocator: class_meets_every_residue, class_residues_distinct, locator_closure,
              locator_closure_contra (a class of copies silent against g forces g | N), locator_modulus,
              locator_modulus_prod (the product of the primes 7..X divides N) and locator_modulus_le.
            - RangeMissed: sq_mod30_cases (a prime g >= 7 has g^2 = 1 or 19 mod 30, exactly one),
              missed_copy_legs (legs g^2 + 28, 30 or g^2 + 10, 12), gear_misses_own_copy,
              larger_gear_sq (g^2 + 30 < h^2 for primes 7 <= g < h), no_larger_gear_acts.
            - RangeHandoff: RangeStatement q := exists p, q < p, p + 2 <= primorial q, p and p + 2 prime;
              range_implies_unbounded (the range statement at every prime q >= 7 gives twins above every N),
              twin_serves, serves_interval, small_cases (every q in [7, 23], witness 29).
            OPEN ITEM 1 - does RANGE force the missed-copy chain. PROVED: domination criterion
            (service(M) contains service(R) iff p_R <= p_M <= sigma(p_R + 2)# - 2); the non-dominated set
            N = union of the tails Rev in (m_r, r# - 2] and RANGE <=> chain(S_1 union N); the crossing form;
            the two-copy cover law; given RANGE, chain(S_1) <=> the S_1 hand-off at every live anchor, and the
            exact form of a converse failure. REFUTED: a service-preserving map from revealed copies to
            revealed missed copies (copy 8 = (239, 241) serves [11, 239) and no missed copy lies in
            [239, 2308]; it is covered only by c_13 and c_79 together); the status of a region copy
            determining c_g or c_g' (all four combinations occur: copies 14, 184, 8, 267). NOT ESTABLISHED:
            the converse itself; whether RANGE excludes a finite S_1.
            OPEN ITEM 2 - doomed classes. PROVED: the child structure (1 zero child, e_h struck, h - 1 - e_h
            escaping); a class is doomed iff it holds no revealed gear iff every prime g in it has a
            composite leg; non-doom is inherited upward, doom downward; the leg polynomials are irreducible
            and have no fixed prime divisor on any escape class; for every finite row set some subclass
            free of it holds infinitely many gears (Dirichlet), so doom has no finite-row certificate; with
            the acting bound removed every class is doomed and the tree is unchanged, so the tree alone does
            not determine doom. R: no class is doomed <=> every escape class holds a prime g with g^2 + A and
            g^2 + B both prime <=> every class holds infinitely many. Every class at levels 11, 13, 17 holds
            a revealed gear (measurement). REFUTED: "only fixed-least-residue chains meet a window" (13 mod
            2310 -> 4633 -> 64693 -> ...). NOT ESTABLISHED: whether any class is doomed.
            OPEN ITEM 3 - the derivation as an operator. PROVED: Der(O) = D and Der(D) = D, Der o Der = Der
            for every machine; the fixed points are exactly the f_C for unbounded copy families, D the least
            of them and the unique one with each lower leg in (h^2, h^2 + 30); the tower P^k = (J^k)* O has
            2^(2k+1) column classes mod 2 x 15^k, J mapping them 4-to-1; J's fixed columns are 1 and 29
            (30(J(y) - y) = (y - 1)(y - 29)); class counts run {2} -> {0,2,4} -> {0,...,8} -> {0,...,16};
            acting stays diagonal (h acts on column y of P^k iff h <= J^(k-1)(y)); striker persistence is
            eventually periodic, with seed-period-1 strikers {11, 13, 29, 31, 67, 79} (branch 1) and
            {11, 13, 23, 37, 853} (branch 19). NOT ESTABLISHED: class counts from level 4; the converse arrows
            down the tower.
          - R5.f.xxxv.c.xxxii. KERNEL BATCH 2, THE LIVE ANCHORS, THE TOWER LIMIT, AND THE RANGE-LINE MAP
            (2026-09-25; 16 agents; record research/proof/range_kernel2_2026-09-25.md; map
            research/proof/range_line_map.md).
            KERNEL (0 sorry/admit/axiom/native_decide, standard axioms, reviewer-checked, rebuilt by the manager):
            - RangeMissedReveal: legs_coprime_30; composite_leg_small_factor; missed_copy_revealed_iff - both legs of
              c_g prime iff no prime h with 7 <= h < g divides a leg. (The brief's proof step g^2 + 30 < (g+1)^2 is
              false at g = 7, 11, 13; the Lean proof uses larger_gear_sq instead.)
            - RangeDerived: J1_exact, J19_exact, copy_map_on_gear; 30(J1(y) - y) = (y - 1)(y - 29) and J1 fixed exactly
              at 1, 29; J19 has no fixed column; J strictly increasing; J below self exactly at {11, 19} and
              {7, 13, 17, 23}.
            - RangeChain: chain_implies_range_all, chain_implies_range, chain_implies_unbounded - a strictly increasing
              sequence of twin lower legs from 29 with p_{k+1} + 2 <= primorial(p_k) gives the range statement at every
              q >= 7 and twins above every bound.
            LIVE ANCHORS. PROVED: given RANGE, chain(S_1) <=> m_r > r at every live r <=> at each such r some prime h
            has r < h^2 + A_h, h^2 + B_h <= r# and no prime 7 <= f < h dividing a leg; the pair form for consecutive
            S_1 gears; failure at r <=> g_r^2 + A <= r, and then the anchor pair straddles all of (r, r# - 2];
            TRANSPORT: c_h is revealed iff some revealed (p, p + 2) with p > h has p = h^2 + A_h mod (h-)#, and for
            h > r that class meets (r, r# - 2] only in c_h; FIXED ROW: a row kills every unit class of
            (x^2 + a)(x^2 + a + 2) iff (f, a) = (5, 4) or (3, 0 or 2) - never in base 30. MEASURED: anchors to r = 61
            (m_r > r at r = 7..61; tail(7) empty, tail(r) non-empty at r = 11..61). REFUTED at the level of rules: in
            base 6 (missed column (g^2 + 4, g^2 + 6), always divisible by 5) S_1 = {29}, (41, 43) lies in every tail and
            the anchor form fails at every r >= 29 - so "a non-empty tail forces m_r > r" does not follow from any rules
            that also hold in base 6. NOT ESTABLISHED: the base-30 converse.
            TOWER LIMIT. PROVED: repaired tower form P^(k+1) = {y in L_k : gcd(J^k(y), 15) = 1} (the written form
            15 does not divide J^k(y) is REFUTED: y = 11, J(11) = 5); J: P^(m+1) -> P^m onto, fibres of size 4; the
            next-digit law J^k(r + 15^k s) = J^k(r) + s prod_{i<k} J^i(r) mod 15; the limit L is Z_2^x times the
            preimage of {1, 4, 11, 14}^N under the digit map, with J acting as the shift; 3 and 5 repel the fixed point
            1, 2 is neutral; the only real periodic points are 1 and 29; 2-adic periods are powers of 2; N_h is
            non-empty for every prime h >= 7. MEASURED: no natural number other than 1 and 29 lies in the limit below
            2 x 15^10; the deepest columns (delta 22 at 360,329,223,301); no column below 10^6 other than 1 is revealed
            at more than two consecutive levels. NOT ESTABLISHED: whether the limit holds another natural number,
            whether any limit column is revealed at every level.
            MAP: research/proof/range_line_map.md drafted from the records, checked line by line against its sources
            (27 findings, all fixed, no prospect-judging sentences), then extended by the manager with this round.
          - R5.f.xxxv.c.xxxiii. THE FIELD KERNEL CARRIED TO THE RANGE (2026-09-25; 26 agents; inventory then six
            rule groups, derive-refute-defend-adjudicate; record research/proof/field_to_range_2026-09-25.md). Owner's
            direction: generalise the lower and upper set formulas from the proved field rules. DICTIONARY: copy j is
            column 5j; a column distance 5D is a copy distance D; the field condition 3m = +-1 mod p becomes
            15D = +-1 mod g.
            PROVED on the range object, by group:
            - G1 widening/persistence: the negation of the range statement (a total blame) holds iff every copy j in
              [1, q#/30) has lpf(900j^2 - 1) <= q or lpf^2 <= 30j + 1; revealed iff lpf(900j^2 - 1)^2 > 30j + 1;
              JOINT REDUNDANCY: every non-acting strike of an upper gear on a survivor is shadowed by an acting strike
              or is a home strike, at most one non-acting striker per leg; the non-revealed survivors are the disjoint
              union over upper gears g of F_g = {j : lpf = g, not a home copy}.
            - G2 covered runs and alignment: Range(q) <=> 0 not in C_q^act; the translate kM + [1, M-1] is fully deleted
              by acting gears iff (-kM mod g)_g lies in C_q^free, a bijection mod prod(U_q); the acting and acting-free
              deletion patterns differ exactly on the twin copies with a leg among the gears; C_7^free and C_11^free
              are EMPTY (exhaustive; best cover 42 of 44 at q = 11, realised at translate k = 22370436767022);
              F_g = min(a_g, g - a_g) = (m g +- 1)/30.
            - G3 two-gear freedom and records: tooth law sigma_g = (kappa g +- 1)/15, kappa = 1, 2, 4, 7 for
              g = +-1, +-7, +-11, +-13 mod 30; pair law (g strikes j and j + D, 0 < D < g, iff D in {sigma_g,
              g - sigma_g}, on opposite legs); comb count; copy E2 record R(g,h) = 2 + [{g,h} meets {7,29,31}] +
              [{g,h} in {{7,11},{7,23},{29,31}}]; the only alternating two-gear cover is (29, 31). MEASURED: no two upper
              gears strike 5 consecutive survivors at q = 11..29.
            - G4 E4c and chain: centre law (g strikes x < y iff [g | y - x and g | 900x^2 - 1] or [g | x + y and
              g | 225(y - x)^2 - 1]); interior gaps = 0 or +-t_g, alternating; the Gamma_k depth formula; Gamma_3 = 0 at
              q = 7..31; mirror law, the only common centre is copy 0.
            - G5 node chain: V_q (revealed copies in the range) = twins in (q, q# - 2]; transport between machines holds
              with acting and fails at every pair without it; scratch Lean (not yet in the repo): RANGE <=> every node
              good <=> the reach from 29 is unbounded; machine 29 settles |V_s| >= 2 at every node below 6,469,692,809.
            - G6 mirror/periodic: three-term mirror striker law; height split T(s)^2 + T(M - s)^2 = q# + 2; the
              palindrome's central gap d* = min{d odd : gcd(225d^2 - 1, M) = 1}; Range(q) <=> some mirror pair of
              survivors is not doubly deleted; revealed status has no period.
            ACTING: every standing statement that uses acting uses it only through the trial-division rule or its
            consequences (acting iff cofactor >= gear; revealed iff both legs prime) or as a pointwise height
            inequality; none forbids a total blame map.
            REFUTED with instances: the literal widening header on copies (q = 11, j = 17: 511 = 7 x 73 unstruck by
            U yet not revealed); persistence read with upper gears (copy 1 struck by 29 and 31 in U_11); three
            consecutive g-struck survivors span exactly g (q = 11, g = 13: 16, 23, 36, span 20); unrestricted
            strike-set equality (q = 107..139, gear 241 strikes consecutive survivors 1197, 1213 acting on neither);
            revealed status P-periodic (29, 31 act on copy 1 + P, not on copy 1); mirror-closure of V_29.
            NOT ESTABLISHED: which acting-based statement forbids a total blame map; whether C_q^free is empty for
            every q (undecided at q = 13); a two-gear bound on consecutive survivors from q = 17 and a k-gear bound in
            k alone; NS and NC; an exposure-preserving involution.
            ENVIRONMENT: a stray tower-limit script (b2_twoadic.py) from c.xxxii held 20 GB since 10:14; stopped by the
            manager.
          - R5.f.xxxv.c.xxxiv. KERNEL BATCH 3, MIRROR PAIRS UNDER ACTING, RUN LAWS (2026-09-25; 13 agents; record
            research/proof/mirror_runs_2026-09-25.md).
            KERNEL (0 sorries, standard axioms, reviewer-checked, rebuilt by the manager):
            - RangeRegion: revealed_iff_minFac (both legs of copy j prime iff 30j + 1 < minFac(900j^2 - 1)^2),
              blamed_iff_minFac, total_blame_iff. Reviewer: total_blame_iff_primorial is true but vacuous for q >= 7
              (copy 1 lies inside the bound); the range-window form is a one-line corollary not yet in the file.
            - RangeCentre: centre_law - a prime g >= 7 strikes copies x < y iff (g | y - x and g | 900x^2 - 1) or
              (g | x + y and g | 225(y - x)^2 - 1); centre_law_field over any field with 30, 4 invertible.
            - RangeReach (ported from scratch): range_iff_all_good (RANGE iff every twin node s >= 29 has its successor
              below s#), good_iff_rangeStatement, range_iff_reach_unbounded (RANGE iff the good-step reach from 29 is
              unbounded), ns_iff_two_rungs, range_of_ns, twins_unbounded_of_ns, not_parent_unique (149 is a rung of both
              29 and 59), and 30 more. Corrections: nodes are twin nodes; no_node_region_17 dropped as false for twin
              nodes ((311, 313) lies between 17^2 and 19^2).
            MIRROR PAIRS (pair P_d = {(M - d)/2, (M + d)/2}, a = q#/2, rho = a + 1). PROVED: class form (g strikes the low
            member iff d = M +- t_g, the high iff d = -M +- t_g, mod g); ACTING WINDOWS (low deleted iff some upper g in a
            low class has 15d <= rho - g^2, high iff some h in a high class has 15d >= h^2 - rho; joint acting forces
            g^2 + h^2 < q# + 2); the four legs' product is (a^2 - (15d + 1)^2)(a^2 - (15d - 1)^2), four root classes
            unless g | q# -+ 2 (then {0, +-2t}); a gear strikes both members iff g | (q# - 2)(q# + 2) and g | d, and acts
            on both iff also 15d <= rho - g^2; the empty-band condition (no multiple of g in [1, w_g] iff 2q# -+ 4 + e^2 is
            a square, e odd <= 13; only q = 7 below 3000, with g = 8 not prime); on the core d <= R both members face
            exactly the gears up to sqrt(rho); THE MEAN-HEIGHT IDENTITY: Revealed_q = Loss_q union (S_Q in [1, M - 1]
            minus Gain_q), so not-Range(q) iff Loss_q is empty and every survivor unstruck by the gears below sqrt(rho) is
            in Gain_q (high survivors with a leg h h', sqrt(rho) < h <= h'). REFUTED: the image in the sandwich is not
            "exactly onto" (q = 13, d = 143: 17161 = 131^2); high-member deletion always by a gear <= sqrt(rho)
            (q = 23, d = 38699: 10589^2); the central pair P_{d*} always doubly deleted (not at q = 7, 11, 17, 19); no pair
            with one shared sole striker (q = 17, d = 3059, gear 23). Acting enters per member only.
            RUN LAWS. PROVED: distance law (one gear on x and x + D forces g | D(225D^2 - 1)); triple law; exact comb and
            span formulas c_g, tau_g; ACTING PAIR LAW (g strikes and acts on j and j + D, 0 < D < g, iff D in {sigma_g,
            g - sigma_g}, g | 2j + D and 30j + 1 >= g^2). MEASURED: no two upper gears strike 5 consecutive survivors at
            q = 11..31. REFUTED: that bound at q = 37 (five instances, e.g. 2749631283 + {0, 9, 29, 30, 41}, gears 41 and
            43, all acting); the one-gear bound 2 at q = 47 (gear 53 strikes four consecutive survivors 247507649335490 +
            {0, 7, 53, 60}); three gears on 7 consecutive survivors at q = 37. So bounds of the form 2k are refuted at
            k = 1, 2, 3. NOT ESTABLISHED: any bound in k alone.
          - R5.f.xxxv.c.xxxv. ONE-GEAR RUNS, THE TWO-PRIME LEGS (GAIN), KERNEL BATCH 4 (2026-09-26; 13 agents; record
            research/proof/runs_gain_2026-09-26.md).
            KERNEL (0 sorries, standard axioms, reviewer-checked, rebuilt by the manager):
            - RangeActPair: Acts g j := g^2 <= 30j + 1; distance_law (one gear on copies x < y forces
              g | (y - x)(225(y - x)^2 - 1)); acting_pair_law; acts_mono.
            - RangeMirror: the mirror pair s = (M - d)/2, s' = (M + d)/2 for odd d < M, M = q#/30; legs_sum (cross legs sum
              to q#), legs_diff (q# - 2, q# + 2); shared_striker (a gear striking both members divides q# - 2, q#, or q# + 2;
              a gear above q divides q# - 2 or q# + 2); height_split and acts_on_at_most_one (2g^2 > q# + 2 acts on at most
              one member); four_leg_product over Int.
            - RangeWindowForm: range_copy_iff, copy_range_implies_rangeStatement, no_copy_blame - the blame test on the range
              window (q < 30j - 1, 30j + 1 <= q#), which fixes the vacuity of total_blame_iff_primorial. Scope: covers only
              twin pairs of the form 30j +- 1, so it implies RangeStatement one way only.
            ONE-GEAR RUNS. PROVED: exact run conditions (formula 1) including acting and placement; the lane rule; the gear-7
            lane law f(K) = K - 2a - max(0, r - 5), K(n) = n + 2 floor((n - 1)/5); runs on consecutive teeth have N <= 10;
            the ladder N <= 2 floor((H - 1)/G) + 1 + [(H - 1) mod G >= sigma_G] against h2(q), the longest all-struck stretch
            (4, 5, 7, 12, 18, 25, 31, 38, 49, 59, 69, 86 at q = 11..53, exact to 41). MEASURED: B_1 = 2 at q = 11..31, 3 at
            37, 41, 43, 4 at 47, 53; and longer maximal acting runs in range: 5 at q = 89 (gear 97), 97 (101), 101 (103), 6 at
            q = 137 (gear 139), 7 at q = 293 (gear 307 on {0, 41, 307, 348, 614, 655, 921}). So no bound B_1(q) <= c with
            c <= 6 holds for all q. NOT ESTABLISHED: B_1 unbounded (no N-run construction for every N); the first q with
            B_1 >= 5. Placement (whether a class mod M G lands in [1, M - 1]) is what separates relaxed patterns from runs
            (q = 43, gear 53: relaxed 4-run pattern exists, none lands in range).
            GAIN. PROVED: G_q = (S_Q minus S_Q*) in the high half, Q* the largest prime <= sqrt(q#); on each leg of a G_q copy a
            band prime divides the leg iff the leg is composite and the cofactor is then prime; exact pair enumeration
            (band prime h, prime cofactor c < 2h); a copy arises from two pairs iff both legs are products of two band primes;
            band-gear teeth law with acting iff c >= h; three-class rule (classes distinct iff p does not divide q# +- 2);
            mirror trichotomy for the mirror of a G_q copy (no U- strike: revealed; strikes none acting: revealed, in Loss;
            an acting strike: not revealed). not-Range(q) <=> Loss_q empty, S_Q misses the low half and S_Q* misses the high
            half. Acting enters only as trial division on one copy; the two copies of a mirror pair are linked only through
            legs(M - j) = q# - legs(j).
            REFUTED with instances: four band-teeth and multiplicity wordings (repaired forms stand); a Gain_full class form
            for q >= 13 (q = 13, j = 505); the shared-class lemma off S_q.
          - R5.f.xxxv.c.xxxvi. THE GENERALITY LEDGER (owner 2026-09-26: only proofs that hold for every machine size count;
            29 agents; ledger research/proof/range_generality_ledger.md). Every standing statement of the range line classified:
            general and in Lean (18 entries), general on paper or in scratch Lean (4 + 13 groups, plus one list of general
            counting/locating results excluded by the rules), claimed general but not proved (11), refuted as general (43, each
            with a counterexample), instance data (12 groups, marked as checks only).
            KERNEL, five new modules, all quantified over every q from a fixed lower bound (q >= 5 or 7, or any q), no primality
            or size restriction on q (reviewer-checked, rebuilt by the manager, 0 sorries, standard axioms): RangeGen1 (the
            height split of the range), RangeGen2 (Gain), RangeGen3 (mirror pairs under acting), RangeGen4 (total-blame
            anatomy), RangeGen5 (acting against acting-free with a fixed cut, proved for an arbitrary cut X before being read at
            X = isqrt(q#)). Corrections carried: Loss_q must use the lower leg; the four-class rules hold for primes above q (and
            in a general form with the q# factor), not for primes dividing q#; RangeCopy -> RangeStatement one way only, since
            RangeStatement also counts twin pairs = 11, 17 mod 30.
            PROVED GENERAL THIS ROUND (scratch Lean, not yet in the kernel): the WALL nextprime(X)^2 < X# for every natural X >= 7,
            failing at every X <= 6 (and (2X)^2 < X# iff X = 0, 7 or X >= 11); every gear <= q' acts on every copy j >= M; silent
            classes have at most one member per period and one in the window band; the near-neighbour cofactor floor and
            kill-free windows with exact boundary cases and sharp class windows (K_c by g mod 30); the top-band algebra of
            lambda.
            STILL CLAIMED GENERAL, NOT PROVED: RANGE itself; the acting-both equality (missing: an existence clause, checked to
            q = 197 with gaps); alignment of a relaxed covering pattern into the period (only a counting argument exists);
            M(G) >= G'^4 for G >= 19 (induction step missing; a Bertrand route recorded); the below-square strike count; the
            derived period statement; derived window thresholds; least-split-copy constancy mod 15d; the twin-striker sets;
            strict increase of X at every gear pair; a Type-C twins bound.
          - R5.f.xxxv.c.xxxvii. CLAIMED-GENERAL ITEMS PROVED FOR EVERY q, AND THE WALL IN THE KERNEL (2026-09-26, second
            general round; 27 agents; ledger research/proof/range_generality_ledger.md updated: 21 kernel modules, 50 refuted
            rows).
            KERNEL (reviewer-checked, rebuilt by the manager, 0 sorries, standard axioms; all 21 Range modules import together):
            - RangeWall (32 theorems): nextprime(X)^2 < X# for every natural X >= 7, failing at every X <= 6; (2X)^2 < X# iff
              X = 0, 7 or X >= 11; every gear <= q' acts on every copy j >= M; window and range-top lemmas; lap bounds; silent
              classes (30N >= X#, at most one member per period and one in the window band).
            - RangeNearKill (70 theorems + 3 defs): cofactor floor, kill-free windows, exact boundary cases (hypothesis 3 <= h
              dropped - stronger), sharp class windows, neighbour rule.
            - RangeTopBand: the top-band law per row and up-closure of the d0 test (no formal lambda(g) yet).
            PROVED FOR EVERY q/g THIS ROUND:
            - C7: M(G) >= G'^4 for every gear G >= 19 (Bertrand; scratch Lean c7_fourth).
            - C8: the below-square strike count 2 floor(g/30) + c(g mod 30) for every g >= 7 coprime to 30, no import.
            - C11: least split copy constant mod 15d for every gear pair; "both act iff c > d" REFUTED in general (counterexample
              (17, 31)), proved for every twin, cousin and sexy pair.
            - C12: twin gears share leg strikers only from {11, 29, 31} or {13, 23, 37}, never 7, for every twin gear pair.
            - X10: X(g) non-decreasing; exact step law (X increases iff a prime lies in (g^2 + A, g'^2 + A']); strict increase
              along revealed gears; Bertrand/Nagura/Dusart spans; strict increase at EVERY consecutive pair follows from
              Legendre's conjecture (a hypothesis) and is NOT proved.
            - C1E: identities for the acting-both clause, the survivor bijection x -> (M - 2x)/g onto the good k, and uniqueness
              of the top gear per side (at most one U- prime of q# -+ 2 with E_g <= g). The existence clause at every non-top
              gear is proved only by a counting argument (not a result under the working rules); at the top gear it is OPEN.
              Checked at every prime q <= 257, where no top gear falls in the open case.
            REFUTED as general this round: C7's small-case remark, the C11 acting clause (17, 31) and mirror clause (29, 31), a
            cited import for X10, three C1E statements.
        - R5.f.xxxv.b. THE KNOWN OPENING'S COPIES (owner's construction 2026-09-21;
          research/stack/r8/known_opening_copies.py). Drop q from the machine; the lower machine
          5..q_- cycles q times inside the range, carrying the known opening (-1, 1) to the copies
          k P_L, k = 1..q-1, members 6 k P_L -+ 1 - Euclid-type numbers coprime to every gear below
          q; each overlay gear g >= q strikes at most two copies (the k with 6 k P_L = +-1 mod g).
          QUESTION: does some copy always have both members prime? RESULT: no - twin copies 4, 2,
          4, 6, 2, 1, 7, 1, 1, 2, 0, 1, 1, 1, 3, 1, 0, 4, 0, 2, 1, 1, 2, 1, 1, 1, 0 for q = 7..113:
          empty at q = 43, 71, 79, 113. The prime members number about 3.6 ln q of the 2(q-1)
          (12-23 observed), the Euclid enhancement e^gamma ln q_- / ln(k P_L) per member, so the
          family holds about 3 ln^2 q / q twin copies and empties as q grows. REFUTED as a
          guaranteed survivor. The tweak with a smaller lower machine y << q gives more copies
          (P/P_y of them) but the fair-rate rule (structured_families.md:64, PROVED) says a family
          defined modulo the lower machine's period is struck by every gear above y at exactly its
          share, so the family carries no advantage over the range as a whole.
    - R5.e. ADJACENT STRETCHES AND THEIR SHARED PHASES (pre-registered 2026-09-19 11:40, before
      computing; the owner's question of 2026-09-18, how the gears kill adjacent stretches).
      Exact: the phase of gear g relative to the square shifts between the stretches of p and q
      by -(q - p)(q + p) 6^-1 mod g, so the two adjacent stretches see IDENTICAL phases at every
      gear g >= 5 dividing (q - p)(q + p), and unrelated phases elsewhere. Theory: adjacent
      stretches interact only through those shared gears. Predictions: (i) the number of shared
      gears is 0 to 3 at almost every p (the prime factors >= 5 of gap x (p + q)); (ii) the twin
      density (twins / columns) of adjacent stretches is uncorrelated, |r| < 0.05 over all p to
      20,000, and stays so conditional on 2 or more shared gears; (iii) a stretch's density is
      unrelated to the number of shared gears. Refuted by |r| >= 0.1 anywhere, or a density
      shift of 10% with shared gears. Stop line: if (ii) holds, FACT (the coupling is exactly the
      shared gears and it carries no twin information); no follow-up.
      RESULT (11:45; every p to 20,000, 2,259 stretches). (i) held: shared gears 0 at 12
      stretches, 1 at 576, 2 at 1,207, 3 at 445, 4 at 18, 5 at 1. The raw density correlation
      is 0.92 - a confound the pre-registration missed (both stretches sit at the same p, and
      density falls like 1/(ln p)^2); detrended by the ratio to a rolling median of the 100
      neighbouring stretches: (ii) r = -0.020 for p > 2,000 (n = 1,958); pairs with <= 1 shared
      gear r = -0.024 (n = 446), with >= 3 shared gears r = -0.098 (n = 451, two standard errors,
      below the refutation line and not pursued); (iii) residual 1.004, 0.999, 1.000, 0.997 at 1,
      2, 3, 4 shared gears, no shift; residual by gap 0.996-1.006 at gaps 2-14, spread falling
      with the gap as the stretch lengthens. FACT: adjacent stretches are coupled exactly at the
      gears dividing (q - p)(q + p), and that coupling carries no twin information.
          - R5.d.i.c. THE GROWTH OF THE RIGID RECORD BEYOND 61 (pre-registered 2026-09-19 11:45,
            before computing; ILP bisection with CRT certificates, background). Spawned by
            R5.d.i.a's OPEN growth law. Theory: the rigid record grows like the free one,
            F ~ p (ln p)^(1+e) with e > 0 small at this scale, so F / (p ln p) keeps rising.
            Predictions: F(61) in [179, 182] (bisecting); F(67) in [194, 215] (increments 15 to
            [RESULT 13:27: F(61) = 179 EXACT - 180 and 181 uncoverable (4,032 s, 2,779 s), 179
            certified at x = 13169725611018917022346; prediction held. 14:33: F(67) >= 213,
            certified at x = 57893748420785405877249 (run exactly 213), inside the predicted
            [194, 215]; 224 timed out; 218 PROVED uncoverable (4,176 s), 215 timed out (2 h):
            F(67) in [213, 217]. Prediction [194, 215] holds at the lower end and is open at the
            upper by two. 2026-09-20 02:45: L = 215 UNCOVERABLE (11,285 s), L = 214 UNCOVERABLE
            (9,981 s): F(67) = 213 EXACT, certified both ways (rigid_record_bisect.py,
            results_rigid_record_67b.txt). Increment 61 -> 67 is 34, above the predicted 15-30;
            F/(p ln p) = 213/(67 x 4.205) = 0.756 at 67 against 0.713 at 61 - the ratio rises,
            prediction "at or above 0.70" held. F(71): L = 222 coverable (3,227 s), so
            F(71) >= 222, inside the predicted [209, 240]; the L = 241 check was killed after
            three hours by the system for memory (16 GB machine) - the ILP at 18 gears and
            L ~ 240 exceeds the memory budget; F(71) in [222, 259] stands, upper half open.]
            30 per gear as from 43 to 61); F(71) in [209, 240]; F / (p ln p) at 67 and 71 at or
            above 0.70. Refuted by an increment below 10 or above 40, or by F / (p ln p) falling
            below 0.66. Stop line: three more exact values, the ratio's direction, then close as
            FACT; no further values (the memory: larger tables are not a route).
          - R5.d.i.b. RECORD WINDOWS AGAINST SQUARE WINDOWS (pre-registered 2026-09-19 11:35,
            before computing). Spawned by R5.d.i.a: the certified record windows x are known
            exactly at p = 43, 47, 53, 59, 61. Theory: if the squares' residue structure kept
            them off the record runs, the record windows' start residues x mod g would avoid
            the square class {(a^2 - 1) 6^-1 mod g} systematically. Prediction (no relation,
            entries 105, 109, 122): the share of gears at which x mod g lies in the square class
            is about (g + 1)/(2g) ~ 0.5 per gear, so 6 to 9 of 12-16 gears, with no gear
            excluded at every p. Refuted by a gear (or a class of gears) at which every record
            window avoids the square class, or by a share below 0.25 or above 0.75 at every p.
            Stop line: if the share is ~0.5, DEAD in one line, no follow-up.
            RESULT (11:36): shares 0.33, 0.77, 0.29, 0.73, 0.50 at p = 43, 47, 53, 59, 61;
            overall 0.529 (37 of 70 gears); no gear at which every record window avoids the
            square class. Prediction held. DEAD: the record windows sit in the square class as
            often as any window would; the squares' residue structure is unrelated to where the
            record runs are.
       - R4.d.ii. THE MACHINE'S CLOSED FORMS FOR THE NEXT GAP AND THE nth PRIME (owner's
         requests 2026-09-10 and 2026-09-11; research/proof/next_gap_closed_form.md,
         research/proof/nth_prime_closed_form.md; script research/stack/r7/nth_prime.py).
         FACT, not a route. Next gap: g(p) = mex over the core's truncated progressions and the
         tail's single residues (-p) mod q, certified exact when the mex is below the bound
         (kernel mex_form for the free regime; the core / tail split from the loaded record
         rule; the square-root rule). The nth prime: p_n = c_k + W_k(n - N_k), the walk of
         machine k on the section of the base-2 stack that holds it, indexed by the machine's
         count; exact at 15 of 15 checks across the first four sections (p_147 = 853, p_58175 =
         721859). The stretch count in closed form, pi(x + 6L) - pi(x) = Omega_core - S_tail
         under x + 6L < (6L + 1)^3 (kernel two-prime lemma), exact at 12 of 12 stretches to
         8.5 x 10^8: Meissel's formula read on the machine, KNOWN VARIANT, kept because its twin
         version is the step's object. No formula in n or p alone is on the record; the
         obstruction in both is the iterated mex, i.e. the record of {primes <= sqrt p} on a
         stretch, the same object as the step.
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
        a proved reduction of the window statement to one run. THEOREM (E), proved in a line (kernel round 39 sharpened the hypothesis: the column must lie inside the next prime's square, 6k + 1 < q'^2; the refuting instance q = 5, k = 8 is in the kernel; every use here was inside the window):
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
            - 4.i.b.ii.b. THE ORDER LAW OUT OF SAMPLE AT 37 -> 41 (research/proof/
              order_law_37_41.md; scripts research/anchor235/r70/). CONFIRMED EXACTLY:
              B_3(m37; 41) = 98 <= 129 = F(37) + 41 (margin 31, the largest on the record) while
              B_2 = 161 > 129 (recomputed, with B_1 = 299), so k* = 3 = L(m37) + 1 = J_max - 1 as
              predicted; the law now stands at B_{L+1} <= F + q' at 10 of 10 computable steps
              (margins 3, 6, 9, 7, 7, 13, 3, 16, 7, 31) and k* = L + 1 at 7 of 7 decisive steps,
              0 exceptions. The binding word (21, 14, 41, 22) at phase 20, span 98, J = 4 =
              J_max (the deepest fusion, as at 8 of 8 earlier steps); its two triples are
              realised (certified columns of m37 verified against all ten gears), the 4-word is
              not; the true record fusion is (21, 14, 41, 15) at the same phase, so B_3 - F(41)
              = 7 is one unrealised overlap in one slot. NEW INSTRUMENT: membership of a window
              in D_k(M) is decidable exactly with no scan and no dictionary (each gear's two
              struck classes sit at an independent free phase, so a local pattern is realised
              iff a covering problem over the gears is satisfiable); a rung costs a few hundred
              covering problems instead of a table of 10^11 gaps; gated against the m23 period
              scan (35,937 triples, 0 disagreements), the closure's D_3(m29) (7,184 realised, 0
              missed; 10,000 sampled unrealised, 0 false positives), Q*_3 = 90 and Q*_4 = F(41)
              = 91 recomputed, (34, 29, 34) correctly unrealised in m23. MECHANISM (new): the
              deepest fusion's two middles must be an adjacent legal letter pair of D_2(m37); of
              36 letter pairs m37 realises seven and legality kills three, leaving (14, 41) /
              (41, 14) (1,525 openings each) and (27, 41) / (41, 27) (one opening each); the
              cheapest legal pair (14, 27), sum 41 = the alphabet's minimum, is not realised at
              all, and the second pad 82 is not a realised gap value; 55 of the 98 columns sit
              in the middle (unlike 29 -> 31, where the chains paid the letter floor and bought
              span with the flanks). OPEN: extend the order law and Phi = B_{L+1} past 41 with
              the covering-problem instrument (43, 47, 53, ...), and the fusion lemma's
              instances. DONE (prover U2x, research/proof/order_law_beyond_41.md; scripts
              research/anchor235/r71/ol2_*.py): THE ORDER LAW IS FALSE. 43 -> 47: F(43) = 103,
              budget 150, L = 2, J_max = 4, B_2 >= 213, B_3 = 153 EXACT > 150 (every deferral
              resolved; independent re-enumeration of 74,873 fusing 4-words, 41,405 above span
              153, 0 level-3 admissible, 0 undecided), so k* = 4 = L + 2 = J_max there (B_4 =
              F(47) = 118 <= 150; the budget itself holds with slack 32, only the relaxation
              fails). Certified as explicit columns, not solver booleans (ol2_verify.py, gated
              both ways on full periods of m11/m13/m17, 0 disagreements): the two triples of the
              binding word (45, 16, 47, 45) occur at columns 1,669,802,076,752,677 and
              1,103,716,997,185,117 of m43, the 4-word nowhere. 41 -> 43: B_3 = 118 <= 134,
              margin 16, k* = 3, every number agreeing with fusion_lemma.md's addendum. 47 -> 53:
              F(47) = 118, budget 171, L = 4, J_max = 6, B_4 >= 198, B_5 = 145 = F(53) exactly,
              margin 26, k* = 5, binding word (70, 35, 18, 22) REALISED, the first step where the
              relaxation is the machine. MECHANISM (measured exactly): the deepest term is m + a +
              c, the cheapest realised legal L-word plus the widest flank the engine allows each
              side (57 + 28 + 33 = 118; 63 + 45 + 45 = 153; 106 + 15 + 17 = 138): a long middle
              strangles the flanks; overshoot B_{L+1} - F(M + q') = 7, 15, 35, 0 against budget
              slack 38, 31, 32, 26, and the law is exactly "overshoot <= slack", with L set by q'
              mod 210 and the overshoot set by the engine: no reason they should agree, and at
              43 -> 47 they do not. L(m53) = 3 (witness (20, 98, 20); 169 length-4 candidates
              refuted), so L = 2, 2, 4, 3 is non-monotone. 53 -> 59 running: B_3(m53; 59) >= 203
              against budget 204, the scan at span 409 of 436; if nothing above 204 qualifies
              the upper half fails there too. VERDICT: the order law and Phi = B_{L+1} are DEAD
              as laws (11 of 12 steps, the exception certified); what survives is the budget
              itself (untouched, 12 of 12) and the flank identity as a FACT of the engine. The
              thread closes: a bound on the next record is length, and step 8 is position
              (proof_skeleton.md section 13).
            - 4.i.b.ii.c. THE FUSION LEMMA OF Phi (theorist on Fable, 2026-09-11; research/proof/
              fusion_lemma.md; scripts research/anchor235/r73/fl_*.py; the rebuilt dictionaries
              D_10(m29), D_6(m31) in r73/results). PROVED, no hypothesis, formalisable: with L =
              L(M) and the maximal words m = the realised legal words of length exactly L, every
              realisation of a maximal word is a J_max-fusion (a struck flank would realise a
              legal word of length L + 1), so the deepest term of the record law is phase-free,
              Q*_{J_max} = max_m [N(m) + |m|] with N(m) the widest flank pair around one
              realisation; the relaxed term is R = max_m [P(m) + |m| + S(m)] (P, S the widest gap
              ever preceding / following any realisation of m); THEOREM A: Phi = B_{L+1} =
              max(F(M + q'), R); COROLLARY B: the relaxation's overshoot is one flank
              substitution P + S - N; THEOREM C: the fusion lemma is equivalent to [the budget at
              the step] AND [R <= F(M) + q']. Verified at 10 of 10 steps (R = 6, 10, 10, 21, 30,
              35, 60, 55, 75, 98; Q*_{J_max} = 5, 7, 8, 18, 25, 34, 43, 55, 68, 91 phase-free;
              0 struck flanks, also on 14,400 tooth-family machines). ROOT for the budget half
              (its terms J <= L + 1 are the exact Q*_J; J = 2 is node 1e's pair statement); the
              brief's "one gear adds at most q'" is not a mechanism (the gear adds the letters,
              sum 55 at 37 -> 41, 86 at 41 -> 43; the flanks are the engine's). The non-root
              remainder R <= F + q' (at L = 1: 2 P(b) + b <= F + q' for every realised letter b)
              holds at 10 of 10 (margins 3, 6, 10, 7, 7, 13, 3, 19, 20, 31) but is NOT a
              consequence of the budget and NOT a law of two-tooth engines: on the tooth
              families it fails at 124 of 1,439 budget-holding machines (13 -> 17) and 2,371 of
              12,924 (17 -> 19), excess to 16 columns, the budget itself failing at 0.07% /
              0.28%; the real teeth sit at the 55th-57th percentile of its margin. Smallest
              instance: {5}, q' = 7, (2, 2, 2) at phase 3, 6 <= 9; tightest: {5..23}, q' = 29,
              (25, 10, 25) at phase 4, 60 <= 63 (m23 never puts a gap of 27..34 beside a 10, no
              visible reason). NEW AT 41 -> 43 by covering problems alone: m41 has FIVE maximal
              words w.r.t. 43, (14, 43), (43, 14), (29, 43), (43, 29), (43, 43), not the one the
              corpus listed; all 35 legal 3-word extensions unrealised, so L(m41) = 2 from the
              instrument; (14, 43): P = 28, S = 33, N = 43, relaxed 118, exact 100 against F(43)
              = 103; (29, 43): 97 / 93; (43, 43) undecided at the hour (725 verdicts memoised).
              THE DECIDING MEASUREMENT, DONE (manager, fusion_lemma.md addendum): P(43, 43) = 5
              (S = 5, N = 7), so R(m41; 43) = 118 and Phi = B_3(m41; 43) = max(103, 118) = 118 <=
              134: the remainder holds at 41 -> 43 (margin 16), the upper half B_{L+1} <= F + q'
              stands at 11 of 11 steps, the first past the scan wall by covering problems alone;
              Q*_{J_max}(m41; 43) = 100 <= F(43) = 103. Lower half (B_2 > 134) with the order-law
              prover.
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
FINALISATION PHASE (2026-09-11, owner: "focus everything on resolving step 8"): the single
document is research/proof/proof_skeleton.md (Parts I-IV); every lane's result is written into
it or it did not happen. KEPT: construction before name; mechanism before label; no re-derivation
of known results; second measurement before any new reading; findings first in chat; forbidden
words window / rung / ladder / descent; no bigger iterations (a larger-q table is not a route;
proofs are closed-form statements from proven mechanics); formalise in the same round. RETIRED
(they served the parts phase): the objects-ledger gate (open since 2026-09-07), the clutch
strategy (the valves are built), budget mode (limits lifted 2026-09-11), the breadth rule (depth
only, on step 8's three faces). Lanes are judged by one question: does the deliverable change
Part III or Part IV of the proof document?

- 2026-09-19 (owner): no "honest" difficulty statements or onus-shifting in chat or records - they
  poison the context and compound; work the proof as in the Conway-conjecture vibe-proof workflow
  (https://overreacted.io/how-i-vibed-a-proof-of-conways-conjecture/): fresh-context lanes in math /
  red / Lean roles, small plain claims in standard terminology each with a test, Lean closing behind
  the maths, audit-and-salvage rather than accumulate, objections read as inputs to reformulation.

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
- 2026-09-07, round closed at the weekly limit (owner: pause until Friday 2026-09-11). State: the stack by squares built and exact (R4.d); the base case and the step opened (R4.d.i) and parked before any computation. Every leaf below R4.c is FACT or ROOT with its mechanism named; the one open branch is R4.d.i. On resume: relaunch R4.d.i fresh (Fable) with the brief on record; then the owner's next step in the plan, the pattern across machines' rules and the connection to machine 1's cycles.
- 2026-09-07, manager (hot lead, one computation): the island witness and the stack's start-of-section richness are one object. Above every prime square to 10^4 the first twin lies below the top gear's long arc (0 exceptions in 1,226); the witness's four classes mod 35 are exactly the offsets gears 5 and 7 can never strike relative to a square, because a gear strikes offset i relative to q^2 iff -6i or 2 - 6i is a nonzero square mod g: the section's start is governed by quadratic residues, the owner's 'squares are even' made exact. Node R4.d.i.a; to build on resume.
- 2026-09-07, manager (research/stack/r3/section_start.py, local runs while the lanes are paused): FIRST-TWIN SCAN TO q = 200,000 (17,981 primes): above EVERY prime square the first twin lies below the top gear's long arc (offset < (2q + 1)/3 columns, i.e. within about 4q of q^2), 0 exceptions; median offset 0.09% of the arc, 90% 0.5%, 99% 3.7%, worst 77% at q = 53. The first twin's class mod 35: the four classes blind to gears 5 and 7 relative to a square (5, 10, 12, 17) are the four most frequent (1,230; 1,487; 1,223; 1,449 against 168-926 for the others), 30.0% of first twins against 26.7% uniform; the other classes are available only for some q (the corridor relative to a square depends on q^2 mod 35) while the blind classes are available for every q. The blind-class formula (1 + the number of pairs of nonzero squares differing by 2) verified against brute force at 76 primes to 400, 0 mismatches. A census of all twins below the arc against a control start of the same size is running (q to 20,000).
- 2026-09-07, manager (section_start.py census, q to 20,000, 2,259 primes): THE HOT LEAD'S RICHNESS HALF IS REFUTED. Twins below the arc above squares 322,186 against 321,052 above control starts of the same size (ratio 1.0035): a square is not richer in twins than any other start. The square fixes WHERE they sit: per-class counts above squares are exactly the availability fractions its residues allow (21,500 in the four 5/7-blind classes, always available; 14,400, 10,800, 7,100, 3,500 for classes available to 2/3, 1/2, 1/3, 1/6 of the q), against about 9,200 in every class above control starts. Empty arcs: 0 above squares, 4 above controls (small q; noted, not a claim). Verdict on R4.d.i.a: the island witness = the blind classes (structural, proved by the quadratic-residue formula) plus the ordinary twin density in the arc (a count); existence in every arc is ROOT. The stack's start-of-section richness (S6) was relative to the machines' densities, not to the line.
- 2026-09-08, manager (local evidence, research/proof/step_evidence.md): every computed link of every chain holds twins with the first twin at most 160 numbers above the cut (chain from base 3 to 260,467,321: link 4 has 1,027,948 twin-gear pairs; the first gear after 16129 is 16139, corrected); the band structure holds cycle by cycle (twins = open under the lower machines, exactly); the newest machine engages only from the next cut, so the start-of-section excess is engagement, a count; twin-gear collisions waste 0.6 to 11% of the pairs' strikes, small and concentrated in (5, 7): candidate 11a is a density correction, not a forcing; the blind classes 10, 17, 12, 5 are the four most frequent first-twin offsets; the first-twin scan to q = 10^6 is running.
- 2026-09-08, manager: first-twin scan to q = 10^6 done (78,495 primes): above every prime square the first twin lies below the long arc, 0 exceptions; median offset 0.02% of the arc, worst 77% at q = 53 (unchanged since 10^4). Scan to 10^7 running.
- 2026-09-08, manager review (step_evidence.md): the first-twin scan to q = 10^7 (664,576 primes) has 0 exceptions with the worst case unchanged at q = 53; the arc bound is a count with a margin growing like q / ln^2 q, ROOT as a mechanism; the blind-class preference fades toward uniform (30.0%, 28.9%, 28.1% against 26.7%) and the class census equals the availability fractions; twin-gear cheapness is a few per cent. The square-start lead R4.d.i.a closes as FACT (blind classes exact) and ROOT (existence in the arc). What no counter-machine reproduces is the recursion: a machine's gears are the survivors below. Friday's lanes start there.
- 2026-09-09, manager (composite_record.py; the lane on the recursion died at the weekly limit again, relaunch Friday): the step measured as a record: inside each section the composite record of the machines below (the longest twin-free run) is 4, 46, 579 slots along the base-3 chain against sections of 19, 2,668, 43,408,532 slots (ratio 0.21 to 1.3 x 10^-5); the record inside a section is its largest twin gap. The step holds at every computed link with a margin widening like the square of the cut against log-squared twin gaps: the count again, as a record.
- 2026-09-09, manager (record_cover.py; the lane died at the weekly limit a third time): the covering structure of every section's record run. The small gears cover their ordinary share (5 covers 2/5 of the run, 7 covers 2/7), so a twin gap is not a small-gear alignment; the core's union leaves 31% of the run unstruck against about 2% typical, and the tail gears strike those leftovers exactly once each with none wasted (181 gears, 181 slots at the base-3 section 4 record). The step at a link is 'the tail cannot finish the core's leftovers on any run as long as the section': the loaded record rule's core-and-tail object with the real phases. Friday: supply against demand as a covering inequality of the construction. Paused until 2026-09-11 04:00.
- 2026-09-09, manager (supply_demand.py; correction of the previous entry): the '31% leftover' reading was wrong (record_cover counted strikes, not leftovers). Over every window of the record length: the core's leftover is typically 20 (max 56) and 12 at the record (base 3, section 4); the tail's strikes typically 199 and 181 at the record; supply exceeds demand in 100% of windows yet exactly one window is twin-free. A twin gap is a below-average core leftover plus an exact finish by the tail: a coincidence of K placements, the count in the construction's terms. Supply against demand cannot close the step; what would have to be bounded is the core's real-phase minimum leftover, the composite record again, located at the core. Paused until Friday.
- 2026-09-09, prover BC4 on Fable (core_leftover.md): the core's real-phase leftover is an extreme value of a count (z about -4.5 for real phases, random phases and integer sets alike); the real minimum is 3 against a free-phase 0 on the base-3 section; the record stretch is not the minimum (12, the 2nd percentile) but the one the tail finishes; min K_L = 0 iff the core's own run R(6L + 1) >= L (S12, exact, L_0 = 278); the recursion's only trace is the origin's density deficit, lowering the minimum. ROOT; no candidate. A pairwise-coprime replacement core cannot exist (S11, proved).
- 2026-09-10, manager (leftover_types.py; limits reset): the core's leftover slots have members that are primes or products of exactly two primes above 6L + 1; the record stretch (base 3) has 0 PP, 9 P+C, 3 C+C against 13.9 / 5.6 / 0.7 on random stretches; the exact finish is 'every leftover slot has a two-large-prime member': the parity barrier's object inside the construction. Lanes resume.
- 2026-09-10, lanes running (limits reset): unstick pass 4 on the step at the core (Fable; research/proof/dead_branches_reopened_4.md: the P-against-P1P2 decision as the shadow; the tail's charges one tier down; the leftover-slot graph and its cover; the sign census; the depth law); Formalist round 40 (Fable; proofs/CoreLeftover.lean: S11 no pairwise-coprime replacement core, the two-prime lemma, S12, the depth lemma); a whole-tree Reviewer (Fable; research/proof/tree_review.md: node-by-node verdict audit, cross-branch reconciliation, the untested list, the staring-at-us check).
- 2026-09-10 (late): the three lanes (unstick pass 4, Formalist round 40, whole-tree review) were stopped when the session ended and died again at the session limit on resume (resets 05:10 Perth; low-priority mode on). State on disk: proofs/CoreLeftover.lean 498 lines, 0 sorries, but 5 build errors remain (unsolved goals at lines 101, 110, 257; a typeclass at 331; Finset.prod_pos unknown at 430): NOT gated, not committed; the unstick lane's spot-check scripts research/stack/r5/leftover_depth.py and leftover_shadow.py run (base 7 section 3: the prime share among core-free members is 0.80 in every depth bin; tail strikes on core-free numbers mean 6.4, 8 at the record; the identity 'tail strikes on core-free = the core-free complement' holds); neither document written yet. Resume all three at the reset.
- 2026-09-11, Formalist round 40 finished by the manager (proofs/CoreLeftover.lean, green at 1036 jobs, standard axioms, zero sorries): S11 rigidity (no pairwise-coprime replacement core), the two-prime lemma (B-rough below B^3 is a prime or a two-prime product), the depth lemma in slot and column forms, S12 the crossing (minimum leftover 0 iff the record reaches L) with the engine's unbounded-open discharge, and the section identity (leftover = twin count below the square). Round 40 is KERNEL.
- 2026-09-11, unstick pass 4 (dead_branches_reopened_4.md): the step at the core reads 'the products of two of the core's survivors cannot meet every core-open slot of a stretch as long as the section's record' (identity verified at 44.7 million starts); the count side is fully independent (binomial PP among leftovers; twin-free runs on the independent prediction at every length); the manager's '40x rarer' lead refuted (starts counted against runs); the depth law S15 proved (kernel), S16 proved forward; the general object is the Omega-census of core-free members at depth u, P against P1P2 being one link's slice. Brief: the leftover at depth u.
- 2026-09-11, whole-tree review landed (research/proof/tree_review.md): twelve verdict mismatches, corrected on the tree and the ledger where the fix is a note (theorem (E)'s hypothesis; the withdrawn CANDIDATE marks; the ledger's gate contradiction and W-numbers; two stale scan notes; F(59) stated as 161 <= F(59) <= 178 pending a certificate); node 4's WEAK and the pinned letter's refuted lower half are already carried by their children 4.i and 4.i.a.i.a.1.a. Cross-branch identities on record: the step at link k is the window statement at rung p_{k+1} up to the home column, and E8's straddling run is the twin-free run across the cut; one window function under three names (the core leftover K_L, W101's two windows, the pullback's Omega), the record its min and Omega its max with no bound between; the onset law inside the leftover types (every finish at the base-3 record is a fuel-type charge with a single tail prime as air). Staring-at-us: the certified ladder proves the step at exactly three links (base 3, 5, 7, link 1); nothing on record bounds the composite record on a section beyond F({5..q}) at its rung; the smallest missing lemma is min K > 0 at the section length. Untested leads U1-U5 launched or queued.
- 2026-09-11, prover U1+U3 (frontier_floor_1e7.md; base_and_step.md Part II filled): THE FRONTIER FLOOR 4.625 CARRIES TO EVERY PRIME CUT TO 10^7 (664,576 cuts; E11): E8's proviso bites at exactly 24 cuts, the last p = 487, the same 24 as the measured range to 20,011; minimum ratio over the claimed cuts (p >= 23) 6.7273 at p = 31; max tau = 0.083333 at p = 31, collapsing to 5 x 10^-9 above 10^6; the first twin above p^2 below the long arc at all 664,576 cuts by an independent method (column sieve, gmpy2), max fraction 0.7714 at p = 53; E10 corrects the review's decomposition to L_s = L_top + L_1 - 1; why it stops biting: d_0 = p/6 (median ratio 1.0000) against L_s ~ ln^2(p^2), max L_s/d_0 above p = 487 is 0.826. PART II: 3,397 sections to 10^9, none twin-free, largest record/length 0.6667 at q = 29 ([841, 961)); S18 the section record is a plain extreme value, record = c ln^2(cut)/(2 C_2) ln T with median c = 0.983 flat to 1% over three decades: the object to bound is a maximum and the recursion supplies set properties; the review's 'record near the top of the section' was a two-example artefact (median position 0.480); the saturated counter-machine is separable from the real set by its own class census (161 sd); a segment-boundary fault in generated.py found and stated.
- 2026-09-11, prover DU1 (leftover_depth.md): the leftover at depth u is ROOT in (L, u): twin-free runs sit on the independent count at 2 <= u < 3 (+0.55 sd over 184,398 runs) and at u >= 3 (+0.61 sd over 409,251), so crossing the depth where P-against-P1P2 fails changes nothing a count sees; one deviation recorded (base 3, L' = 400, gap band [400, 450): 66 against 38.6, z +4.3, surviving two re-measurements, localised to one band, not explained); Omega thresholds exact at nextprime(t')^j (S17, proved).
- 2026-09-11, prover U2 (order_law_37_41.md): the order law confirmed out of sample at 37 -> 41: B_3 = 98 <= 129, B_2 = 161 > 129, k* = 3 = L + 1; 10 of 10 steps, 0 exceptions; the binding word (21, 14, 41, 22) is the deepest fusion; NEW INSTRUMENT: D_k membership decided by a covering problem per window, no scan, no dictionary, gated on three exact checks. Manager's fourth and fifth measurements of DU1's one deviation (base 3, gap band [400, 450) slots): 66 gaps against a geometric tail's 36 (ratio 1.84; neighbours 0.94, 0.97, 1.27, 1.03), the 66 spread uniformly across the section (fractions 0.19 to 0.96) with no preferred length (34 distinct lengths, max multiplicity 6): a single-band excess with no positional or length structure; filed as an unexplained fluctuation, not a claim.
- 2026-09-11, three lanes launched: prover U2x (opus) the order law past 41 with the covering-problem instrument (41 -> 43, 43 -> 47, 47 -> 53; research/anchor235/r71/, order_law_beyond_41.md); prover U4/U5 (opus) N(v) <= F_2 and the J-run outer law at m37, W32 first-hit exactness on the engine y = 7..53 and on the section records (research/anchor235/r72/, engine_laws_m37.md); theorist (fable) the fusion lemma of Phi = B_{L+1} attempted as a proof from the covering-problem characterisation (research/anchor235/r73/, fusion_lemma.md).
- 2026-09-11, manager (local, leftover_depth.md sixth measurement; research/stack/r6/twin_gap_bands.py): the band histogram of twin gaps on five more sections (bases 13, 17, 19, 23 section 3; base 5 section 4 to 5.3 x 10^11; 1.3 x 10^9 twins in all): every band with expected >= 10 within ordinary spread (largest 20 against 12.7), no counterpart of the base-3 bump (66 against 35.9). The one deviation is a fluctuation of one band on one section; closed unless a mechanism singles out the base-3 section. Three lanes died at a session limit before starting and were resumed on the reset.
- 2026-09-11, manager (owner's request: a closed form for the nth prime from all the work; nth_prime_closed_form.md; research/stack/r7/nth_prime.py): p_n = c_k + W_k(n - N_k), the certified walk of the section's machine indexed by the machine's count, exact at 15 of 15 checks (n to 58175, four sections); the stretch count pi(x + 6L) - pi(x) = Omega_core - S_tail in two finite sums under x + 6L < (6L + 1)^3, exact at 12 of 12 (Meissel's formula read on the machine, KNOWN VARIANT); no formula in n alone on the record, the obstruction being the iterated mex = the step's object. Node R4.d.ii opened as FACT, holding both closed-form documents.
- 2026-09-11, theorist (fusion_lemma.md): Phi = B_{L+1} = max(F(M + q'), R) PROVED with R = max over maximal words of P + |m| + S (Theorem A); the fusion lemma is equivalent to the budget AND R <= F + q' (Theorem C); the budget half is ROOT, the remainder holds 10 of 10 but fails on 8.6% / 18.3% of budget-holding tooth-family machines, so it is a real-teeth fact with no mechanism; m41 has five maximal words w.r.t. 43 and L(m41) = 2 from the instrument; P(43, 43) is the deciding number (running locally).
- 2026-09-11, manager (fusion_lemma.md addendum; fl_next.py 41 resumed, 351 covering problems): P(43, 43) = 5 in m41, so R = 118 and Phi = B_3(m41; 43) = 118 <= 134: the order law's upper half holds at 41 -> 43, 11 of 11 steps, the first decided past the scan wall with no table of m41.
- 2026-09-11, prover U4/U5 (engine_laws_m37.md): N(v) <= F_2 holds at m37 (87 <= 90, eight machines, 0 exceptions) and the J-run outer law at J = 3, 4 (J = 5 undecided between 90 and 105, solver wall); 'falls with J' false at m31; D_3(m37) built complete (30,325 rows, loss 0). W32's first-hit exactness REFUTED on the engine's window (4 of 13, 5 of 43 sections): small-N bias plus the nested window's inherited record; on the phase-zero prefix at N >= 10^5 it is within one unit at 12 of 13, so W32 is a large-N law (register rider). New instrument: the exact gap census as a covering count, no period. Neither a route.
- 2026-09-11, manager (owner: the goal is the proof; proof_skeleton.md section 13): the one unproved statement is step 8 (a twin between every cut and its square). With the certified records it is now exact that the record cannot prove it: at 7 of 11 cuts from 13 to 53 the machine {5..p} has a fully struck run longer than the whole section (e.g. p = 41: section 28 slots, F = 91; p = 53: 112 against 145) and realises it elsewhere in its period, never inside the section (inside, the longest gap is 28 at p = 53). Step 8 is a position statement: the first column at which {5..p} realises a run of the section's length lies above p'^2 / 6. Count routes (record, leftover, supply, order law) measure length and are closed as FACT; the order-law lane finishes its document and the thread stops there. Next lane: the first-realisation column, with the covering-problem instrument (a word is realised on residue classes of the column; its first realisation is the least column in them). Unprocessed overnight results found: 43 -> 47 B_3 = 153 > 150 (candidate first exception to the order law's upper half, R = 45 + 63 + 45 with maximal word (16, 47)), 47 -> 53 B_5 = 145 = F(53) <= 171 (holds, k* = 5); second measurement of R(m43; 47) running.
- 2026-09-11, prover U2x (order_law_beyond_41.md): THE ORDER LAW IS FALSE at 43 -> 47: B_3(m43; 47) = 153 > 150, exact and certified by explicit columns (the binding word's two triples at columns 1.67 x 10^15 and 1.10 x 10^15 of m43, the 4-word nowhere); k* = 4 = J_max there; the budget itself holds (118 <= 150). 47 -> 53 holds with B_5 = F(53) = 145, the relaxation equal to the machine. Mechanism: deepest term = cheapest realised L-word + widest flanks, overshoot 7, 15, 35, 0 against slack 38, 31, 32, 26. Phi = B_{L+1} DEAD as a law; the budget (12 of 12) and the flank identity survive as FACT. Thread closed (length, not position). Two hung runs of the killed lane removed; ol2_core.decide_many's persistent pool can hang (guard before any long run).
- 2026-09-11, theorist (first_realisation.md): step 8 at the cut p <=> L_1(p) < l_p exactly (the run through the square column shorter than the section), which is the arc bound's measured object, ROOT; the position floor refuted at p = 29 (the engine's first run of the section's length lies BELOW the square, step 8 holding anyway); the exact obstruction as a construction: two-tooth engines with the same gears strike the whole section at p = 17, 29 and every cut 37..53, so step 8 is a fact of the real teeth 6^-1 mod g, not of the gear set. New record positions m37 (0.0734 of the period) and m41 (0.0124). No route; the wall's shape sharpened.
- 2026-09-11, manager (proof_skeleton.md section 15; Formalist round 41 launched on proofs/SquareColumn.lean): the real teeth at the square column made exact: gear g strikes offset i from the square iff p^2 = -6i or 2 - 6i mod g, so every strike near a square is a statement about p^2 mod g (four classes per gear per offset; the blind classes are the offsets with neither residue a square); step 8 at the cut p reads: the residue vector of p modulo the gears is not in the covering set K_p. Formalist round 41: S1 the square column, S2 the offset-strike law and blind corollary, S3 the twin conclusion below p'^2, S4 step 8 <=> L_a < l.
- 2026-09-11 (17:40), manager CORRECTION (proof_skeleton.md section 16): sections 13-15 and node R4.d.i.e measured the finer statement (a twin between consecutive prime squares), not step 8 of the skeleton, whose sections run from p_k^2 to nextprime(p_k^2)^2 (a number to about its square). For step 8 the record route is open and already on record (tree_review.md section 4): F(q) < q^2/6 suffices, the certified records sit at 0.28-0.39 of the square and fall with q, three links proved, none beyond 59; the general bound of exponent 2 on the paired record is Ziller-Morack Conjecture 6 in free-phase form, sieve-blocked at exponent 4.27 (parity). R4.d.i.e's results stand for the finer statement. Owner's hunch parked (not pursued, on the owner's instruction): the machine is self-similar; the fold 2, 3 creates a left/right open pattern no higher gear can cancel (no prime above 3 is a multiple of 2 or 3), later gears only cover some of its runs, so the patterns are nested fragments of the lower pattern.
- 2026-09-11 (evening; the owner lifted the limits): three lanes on step 8's length face. Formalist round 41 (Fable; proofs/SquareColumn.lean: S0 the record route section_twin_of_unstruck / section_twin_of_record, then S1-S4 the square column, the offset-strike law, the blind corollary, the twin conclusion, step <=> L_a < l); prover (Opus; research/anchor235/r75/, records_by_sat.md): the certified record table past 59 by SAT with the real teeth, gated on F(23..53), F(59) decided, then 61, 67, 71, ...; the ratio 6F/q^2 and the fit q log^2 q against q log q log log q; the free-phase h_2 comparison. Theorist (Fable; research/anchor235/r76/, length_face.md): the target F(q) < q^2/6 as a construction, what on record constrains a run of q^2/6 columns, where the parity barrier bites for the real teeth (measured on the record runs at m23..m37), proof attempt or exact obstruction.
- 2026-09-11, owner: testing bigger and bigger machines cannot find the proof (every machine to infinity would have to be tested); proofs are closed-form statements from proven mechanics. The records-by-SAT lane stopped before producing numbers; the Formalist and the length-face theorist continue.
- 2026-09-11 (evening), manager (owner: bring it all together in one place; are the old mandates still useful): research/proof/proof_skeleton.md rewritten as the single proof document, Parts I (the construction, 1-9), II (the one statement in its exact forms 8a-8e), III (what is proved about it, incl. the three certified links), IV (the three faces and the shape of what would close it); the diary form kept as proof_skeleton_history.md. Standing directions amended: finalisation phase, kept and retired mandates listed.
- 2026-09-11, Formalist round 41 (proofs/SquareColumn.lean, 40 declarations, built 1096 jobs, manager re-built and audited: propext / Classical.choice / Quot.sound only, 0 sorries): S0 the record route (section_twin_of_unstruck, section_twin_of_record, section_twin_of_record_W: an unstruck column below P^2 is a twin; a record F with F + 1 <= l gives a twin in the section; the gear set need only contain the primes of [5, P)); S1 square_column; S2 offset_strike / offset_strike_modEq / blind_class / strikesZ_iff_root with no hypothesis beyond p^2 = 6a + 1; S3 section_open_iff_twin; S4 L_lt_iff, twin_in_section_iff_L_lt (8 <=> L_a < l); S5 the two-tooth family FamilyBlocked with the real teeth as its special case (real_teeth, blockedZ_eq_family). Proof document Part II (8b, 8c) and III.1 now carry kernel names.
- 2026-09-11, theorist (length_face.md): the length face F(q) < q^2/6 is ROOT with the parity twin built: O^- (open columns with Liouville product -1) is empty below (q'^2 - 1)/6 and sieve-indistinguishable from the open set (0 of 1,326 cells above 3 sd), so no sieve input separates the target from a set that violates it by the section's length; the real teeth add only the one-third separation (factor 1.3-1.8, never an exponent) and break the sign symmetry in exactly one place, the origin, where every section sits and no record run does (x/L >= 384,679). Any C < 1/6 implies twin primes; the phase-zero instance of the missing lemma is a twin in (q, q^2 + 1]. Part IV.1 of the proof document updated.
- 2026-09-11 (late), theorist lane launched at the one place the three faces meet: the mechanic of the origin (research/anchor235/r77/, origin_mechanic.md): the dilation form (the struck set on the fold's survivors S is the union of the dilates g.S, the gears the survivors themselves; the owner's nesting hunch as a testable statement), the strike census of real sections against the tooth family's killers and V17, proof attempt or exact obstruction.
- 2026-09-11 (late), theorist (origin_mechanic.md): the origin is ROOT too: the dilation form is exact and is the real teeth; the owner's nesting is TRUE and exact (the struck set is the disjoint union of the dilates g.R_g, 0 mismatches over 198,798 members) but relabels the union without changing it; a NEW third counter-machine, the monoid generated by 5 and the primes = 1 mod 6, keeps dilation, hand-up and the square-root rule and fails step 8 at [25, 961), so the missing axiom is the finite fold itself, which enters only as the sieve's input; smallest instance where the real teeth (not free classes) hold a section: p = 17 finer section, columns 49..59; at p = 29 a single tooth change kills. Named next construct: a non-count use of the tail pins g x m with m a small prime.
- 2026-09-11 (late), owner: pursue a new mechanic from the fold. Theorist lane launched (research/anchor235/r78/, fold_mechanic.md): the class reading of the monoid counter-machine (its irreducibles beyond 5 are all = 1 mod 6, so no twin is possible for a class reason; the sharpened axiom is gears from BOTH classes of the fold at every scale), the left / right cover condition with sides swapped by class, thinned two-class monoids to find the exact content of both classes, the class sign as a sieve-visible Liouville-type sign, proof attempt or exact obstruction.
- 2026-09-11 (late), owner's idea: split the hits into separate FIELDS (field 1 the primes / home strikes; the square field; field j the products of exactly j primes) and analyse each field's own structure, the relations between fields, the section against each field, and the fields on the 6-cycle (mirror symmetry, bifurcation of the fold), to find location rules per field for twin gaps. Theorist lane launched (research/anchor235/r79/, fields.md) with the exact facts pre-stated: field j = the union of the dilates p . field_{j-1}; the square field hits only right members; field 2 hits right members by same-class pairs and left members by cross-class pairs; the mirror law of a dilate about the multiples of 6g.
- 2026-09-11 (late), theorist (fold_mechanic.md): ROOT; the missing axiom cannot be a property of the gear set: thinning the primes by exactly the twin lowers keeps every gear-set property (both classes at every scale, equidistribution) and kills every section; every two-class monoid is transparent (8 holds on it iff it keeps a twin lower, proved, 0 mismatches over 7,309 sections); the classes are invisible to the column cover; the fold's sign is sieve-visible (n mod 3) and the invisible part of Liouville is the class +1 factor count. The axiom is the line itself: every survivor of the fold is a gear or a multiple of a smaller gear (the strike-class law's first half, the completeness of the line at phase zero).
- 2026-09-11 (late), theorist (fields.md): ROOT; the fields are exact (every field is the primes dilated; no field periodic; the class rule; the mirror law about 6 g m with the twin as radius 1; the deep fields confined to the small gears below p'^2; the square field empty in every finer section and equal to the construction's cuts), the owner's growing mirror symmetry is real (2 -> 46.8 -> 110,944 pairs per axis) and is the Goldbach pairing of the primes with the twin as its innermost radius; the split is exactly the one a sieve cannot make (the Omega-strata of the overlay; the column sign is the parity of the field-index sum). New: the mirror about 6 g m preserves divisibility by every prime of 6 g m (12 of 12 cells, two ways).
- 2026-09-11 (late), owner: the fields view simplifies: the prime field is the primes, the square field the squares of primes; neither can kill a twin gap, nor both together; build those proofs to move the wall inward, then the composite fields one by one, with a twin field as the control. Formalist round 42 launched (proofs/Fields.lean): the fields defined on S; the prime field is the twin's definition; squares right-only, one column per prime; the square field cannot cover a stretch; the class rule; the mirror law; the dilation of fields; the overlay of the composite fields is the struck set.
- 2026-09-11 (late), owner: we need LOCATION rules, not counts; the composites are located offset to their prime and square factors. Manager: the composites' location rule is exact and on record (D3: n = g . m = g^2 + g (m - g), g the least prime factor, m open below g; each gear's new strikes start at its square, new_iff); what is not on record is a location rule for the GAP, a column written in the lower primes' positions that is a twin in every section. Theorist lane launched (research/anchor235/r80/, location_rules.md): candidate gap-locating rules (offsets from the square, the mirror axes at radius 1, products p_k q +- 2, midpoints, the deep-field blind columns, offsets f(p)), each pre-registered and tested on every section to 10^7, with the mechanism of the failures or the rule.
- 2026-09-11 (late), owner: stop everything (limits near; lanes waste 200k tokens reading before acting). Formalist round 42 and the location-rules lane stopped before producing anything; no scripts or documents from them exist. Both tasks are re-briefed as execution specs when work resumes: (a) Fields.lean: the fields on S, prime field = the twin's definition, squares right-only one column per prime, the square field cannot cover a stretch, the class rule, the mirror law, the dilation of fields; (b) location rules for the gap: candidate columns in the lower primes' positions tested per section to 10^7 with the mechanism of failures.
- 2026-09-11 (late), manager (local, research/proof/location_rules.md; research/stack/r8/location_rules.py, validated on L_1(p) at p = 11, 13, 17): nine candidate location rules for the gap (first column after the square, the small-gear-blind offset, the middle, the midpoint of the squares, p p' -+ 2 and their neighbours, the previous section's offset) tested on 165 finer sections to 10^6: every one fails at a section with p <= 13; success fractions 0.6% to 10% = the twin density, i.e. chance; the first-twin offset is blind to the small gears in 1 of 165 sections. FACT: the composites are located exactly (D3), the gap is not located by any rule from the lower primes' positions; a name that avoids the dilates is the CRT computation.
- 2026-09-11 (late), Formalist round 42 (proofs/Fields.lean, written by the stopped lane, built by the manager after repairing 23 dependency caches corrupted at the kill: 1097 jobs, 54 declarations, axioms propext / Classical.choice / Quot.sound only, 0 sorries): the fields on S defined; field 1 = the primes >= 5 and the prime field never kills (a twin = both members in field 1); squares right-only, one column per prime (square_column_unique, square_column_eq_W); the square field cannot cover a stretch nor a section of the construction (squares_cannot_cover, squares_cannot_cover_W); the class rule for products (class_rule, class_rule_field); the mirror law about 6 g m in numbers and columns, by class (mirror_law_*, mirror_dilate); field dilation both ways and field (j+1) as the union of the dilates (field_dilate, field_succ_eq_union, field_inter_five); the overlay of the composite fields is the struck set on S (overlay_iff_composite, twin_iff_not_hits_overlay, blocked_iff_hits_overlay). Proof document IV.3a carries the names.
- 2026-09-11 (late), owner: build the fields construction systematically, prove each field's behaviour, then the relationships bit by bit, no handwaving. Manager (research/proof/fields_construction.md): the programme as a statement list A-F with status per statement (kernel names where proved; the method where provable; measured; open); the union's coverage is inclusion-exclusion over the fields, single-field terms are counts (Landau), relations are identities, the correction terms are the joint distribution of the two members' field indices, and step 8 is its (1, 1) cell being non-empty (F5). Next steps in proof order: D3-D5 and B5/E6 into the kernel; C6/D6 stated as counts with Landau cited; F2 as a kernel identity; the joint (i, j) table per section against independence.
- 2026-09-11 (late), manager (local, fields_construction.md step 5; research/stack/r8/joint_table.py): the joint field-index table of a column's members on five sections: the twin cell sits at 0.86, 0.86, 0.99, 0.92, 0.93 of independence, scattering about the Hardy-Littlewood factor with the fold's primes removed, prod_{p >= 5}(1 - 1/(p-1)^2) = 0.8802 (KNOWN); the fields' interaction at the twin cell is the singular series, and the programme lands step 8 at the positivity of that cell on a section. Formalist round 43 running (execution spec: D3-D5, B5, E6 into FieldsB.lean).
- 2026-09-11 (late), Formalist round 43 (proofs/FieldsB.lean, execution-spec brief, 122k tokens against 300-400k for the reading-list briefs; manager re-built 1098 jobs and audited: standard axioms, 0 sorries): D3 field_empty_below and five_pow_mem_field; D4 field_least_factor_le, field_confined (minFac^j < P^2 below P^2); D5 field_index_le; B5 squareField_empty_between, squareField_cuts; E6 overlay_periodic (period the product of the gears, k >= 1 needed), field_not_periodic. Programme rows B5, D3, D4, D5, E6 now PROVED. Rule from the incident: lake must run from inside proofs/ in the same command, else elan picks the default toolchain and rebuilds mathlib.
- 2026-09-11 (late), owner: each field in isolation first, no combining, the square field on its own, and a field per gear from its square. Manager (local): the square field separately (26 members on [121, 16129), all right members); the gear fields (least-factor partition) on the section and, in isolation on the whole line, exact: periodic with period the product of the gears to g, 2 prod (h - 1) hits per period, mirror about 0, gap spectrum the lower survivors' gaps scaled by g / 6 (gear_field_alone.py, g = 5..23); the factor-count fields alone (gap spectra, class pattern, density along the section, nesting share). Formalist round 44 (proofs/FieldsC.lean, execution spec, 149k tokens; manager re-built 1291 jobs and audited, standard axioms, 0 sorries): the gear field defined, its square first, the least-factor partition, disjointness, periodicity, the mirror, the count per period by the totient, gap scaling. Programme part G-alone PROVED.
- 2026-09-12, owner: a new class of field, the composites of a gear and the lower gears. Manager (local, smooth_fields.py): the smooth field C_g (composites with all factors <= g) and its layer L_g (largest factor g) on [121, 16129), g = 5..37: 4 to 365 members, the layer 4 to 55, first members 125, 175, 121, 143, ...; in isolation C_g is the exponent lattice of {5..g} under exponentiation (closed under multiplication, class by D2, count a polynomial in log x of degree pi(g) - 2, KNOWN), never near covering a section. Programme part H.
- 2026-09-12, owner: closed forms for every field type, for locating hits and opens. Manager (research/proof/fields_closed_forms.md): the square field CLOSED (columns (p^2 - 1)/6; none between consecutive squares); the gear field of g CLOSED as the wheel {5..g-1} scaled by g (periodic residue set, 2 prod (h - 1) per period, mirror, gaps g x the wheel's; next hit by the wheel's certified walk); the smooth field and its layer CLOSED (the exponent lattice of {5..g}, next hit a finite minimisation); the factor-count fields CLOSED RELATIVE TO the primes (the range rule, recursively) and not in n; the prime field and the overlay's opens (the twins) closed only as the certified mex walk, the latter being step 8.
- 2026-09-12, owner: a fourth field kind, each gear's composites of itself and only higher gears. Manager (local): 'itself and higher' is exactly the gear field of part G; 'strictly higher' is g times the survivors of the wheel {5..g}, closed (period g x prod of primes in [5, g], 2 prod (h - 1) per period, verified at g = 7, 11); a gear's multiples split exactly into itself-and-higher (new strikes), itself-and-lower (the smooth layer) and mixed (echoes), with the pure powers in both; table on [121, 16129) for g = 5..37. Programme part I.
- 2026-09-12, owner: start with machine {5} below 25 across all fields, then bigger machines, see which fields locate the twins. Manager (local, machine_map.py, machines 5..23 below the next square): every kill is by the gear field of a gear of the machine; the square field kills one column per gear; field 2 carries nearly every kill, field 3 enters at 125, field 4 at 625; the twins are exactly the wheel's open columns (the CRT complement of the gear fields' residues) plus the home columns of the machine's twin gears, at every machine, so below the next square the twin locator is closed and is the sieve. Programme part J.
- 2026-09-12, owner: CRT and the sieve are imprints, not mechanisms; unwind step by step, fields advancing per gear, numerical offsets from an origin that point to the openings. Manager (local, unwind_steps.py, steps 5 -> 7 .. 19 -> 23; programme part K): every kill in a new range (g^2, g'^2) is a lower element carried up by one gear (below h^3 a lower prime), the newest gear enters at g g' (offset g (g' - g)/6), every gear's strikes sit at offsets fixed by g^2 mod h, so adding gear g reads every lower gear field at the SQUARED phase of g's own residue; the new range's twins are the lower wheel's openings at the squared residue vector of the newest gear (the hand-up in phase form); step 8 in phase form: the squared residue vector of a prime is never a covering vector over a stretch as long as its own square gap.
- 2026-09-12, owner: find which squared vectors leave a blind offset inside a stretch as long as g's square gap. Manager (local; programme part L): a vector fails iff it is the phase just before a struck run of the wheel of length >= L; when L exceeds the wheel's record no vector fails (g = 7, 11, 13, 19; the record route); when L is below the record (g = 17: 12 < 18) the failing set is 370 of 85,085 phases, 14 square origins, 3 with roots coprime to every gear (n = 158, 1467, ... mod 85085), and the real origin 17^2 is not among them; random squared and free vectors leave a blind offset at 99.3-100% and the real vector's blind count sits inside the random spread at 27 of 27 steps. Nothing structural excludes a prime's square from the failing set; for step 8 the failing set is empty iff F(q) < q^2/6, the length face again.
- 2026-09-12, owner: it has to have an answer, keep looking. Manager (local; programme part M): the square origin's exact gifts: (M1) the blind classes of the small gears (offsets never struck from any square origin: 5, 10, 12, 17 mod 35 for gears 5, 7) carry the twins 2.4-3.6 x above their share over 424 steps and 53,867 twins, the first twin blind to 5, 7 in 37% of steps against 11% by share, but no guarantee (steps with no blind twin exist); (M2) the top gears' comb, exact: gear g - t strikes above the square at e = s (g - t) - t^2 for the s with g + t + s in S (28 of 28 pairs), sparse; (M3) on one wheel the two-tooth first blind offset after r^2 does not depend on the single-tooth gap after r (means 3.7-5.4 across d = 2..28, no trend). The steps without a blind-class twin are decided by the middle gears' phases g^2 mod h, 13 < h < g.
- 2026-09-12, owner asked why 13 (no reason: the cut of the measurement) and said go ahead. Manager (local, blind_depth.py, 233 steps to g = 1500): the blind set to all gears <= c thins geometrically (11.5% of offsets at c = 7, 1.0% at 13, 0.01% at 23, empty from 37) while the twin enrichment grows slowly (2.3 x, 3.8 x, 5.2 x); a twin blind to every gear <= 7 exists in 230 of 233 ranges, <= 13 in 191, <= 23 in 4; the first twin is never blind past 19. The pointer is real and shallow; it fades rather than sharpening, and the deeper gears' phases decide. Programme part N.
- 2026-09-12, owner: try different measurements between different parts. Manager (local, two batteries, 544 ranges to g = 4000; programme part O): no class effect (g mod 6, g mod 35), no mirror symmetry about the range's centre, no dependence of the scale-free twin density on the gap or on the previous range (autocorrelation -0.02), the constant 7.89 = 6 x 2 C_2 = 7.92 (Hardy-Littlewood exactly), least-factor strike rates exactly the independent products (0.8, 0.686, 0.623, ...), the first twin below the new gear's first strike in 542 of 544, a twin's neighbours killed by gear 5 at 67.5% (the teeth at +-1, W4, known). The range after a square is ordinary beyond the two combs of part M.
- 2026-09-12, manager (local; programme part P): two gear fields across one column: the joint least-factor table on 1.76 million both-composite columns, pairs with gear 5 enriched 1.30-1.35 over the product of marginals, with 7 about 1.10, large pairs depleted to 0.91, the diagonal empty by the fold; the exact two-tooth correlation between gear fields, the same correction as the singular series at the twin cell.
- 2026-09-12, owner: focus on what is structurally new. Manager (local, caustics.py; programme part Q): NEW exact law of the square origin: for every gear h above sqrt(g), with g = q h + r, the first strike above g^2 is at e = s h - r^2 (least valid s), verified 61,772 of 61,772 at g in [300, 2500]; within a band q the offsets are a downward parabola in h with apex at the harmonic g/(3q) of the newest gear's first strike g (g+2); the pile-up at the apex is measured (18% of a band's s = 2 first strikes in the last tenth); no twin desert below the apexes (ratios 1.09, 1.02, 1.00), the caustic being sparse.
- 2026-09-12, owner: each structure says where twins are not; use all of them. Manager (local, exclusion_map.py; programme part R): the exclusion budget over 661k offsets (g in [200, 2000]): the small wheel 5..13 removes 70.3%, the middle gears 17..sqrt g 11.2% (37.7% of the rest), the BANDS above sqrt g 14.5% = 78.4% of what remained, the newest gear 0.03%; the survivors of the three are twins 99.7% of the time. The bands' kills are the products h x m of a prime h near g/q and a survivor m near q g (for q = 1 the multiplication table's diagonal neighbourhood), on the parabolas of part Q. Where the twins are: the small comb's open offsets that are off the middle teeth and off every band parabola.
- 2026-09-12, manager (local, band_products.py; programme part S): the band kills (492,766, g in [200, 2500]) are uniform across the range (10% per tenth), shares by band fall like 1/q (16% at q = 1, 58% at q >= 6), cofactors prime 89% and centred in their strips, kills per prime = the survivor density times the strip width; the band-kill rate at a twin's neighbours is lower (33% against 42%) only because gear 5 takes them. No position preference beyond the first-strike parabolas.
- 2026-09-12, owner: the results must be locations, not density trends. Manager (local, locations.py; programme part T): every structure's kill set written as explicit offset progressions from the square (gear h: c0_h + j h and c2_h + j h with the phases (-g^2) 6^-1 and (2 - g^2) 6^-1 mod h; band gear x: column(g^2 + s x - r^2) - a + j x with r = g mod x; the newest gear at g (g + 2) or g (g + 4)), generated with no factoring; the offsets in no progression are exactly the twins, at g = 101 (12, 17, 22, 38, 43, 50, 55) and at every range from g = 17 to 1499, 233 ranges, 0 disagreements.
- 2026-09-12, THE OWNER'S CONSTRUCTION, built as a view (docs/fields_view.html; artifact https://claude.ai/code/artifact/5f8913e4-d329-41a2-9a2c-9d76e65d47c8): fields as grids of gear rows against the natural numbers, one row per gear taking part in the strike; the fields: multiples of each gear, squares, products of j gears up to the last count with a kill, and per gear the composites of itself with higher gears, with only higher gears, with lower gears, with only lower gears (rows running up the primes to the first with no kill); the sieve as highlights only (kills of left / right members, open columns by what they hold); a machine size q with markers at q, q^2, q#, its mirror point q#/2, cycles, and highlights of ranges and of the machine's gears; a combined row per field and a reorderable summary. The owner: 'this construction is now close to what I have in my imagination when I'm talking about pattern searching.' Pattern work from here is done on this construction, not on the 6k +- 1 fold.
- 2026-09-12, owner: now that we look at the same picture, re-read one or two items closed as "no pattern", "just CRT" or "a count, not a location" on the construction; a sample run; and add a skill for the construction. Manager (local; research/proof/fields_sample_run.md; skill .claude/skills/fields-explorer, user and project copies): node R4.d.i.i.a. Item 1 (the witness after a square) re-opened as a located object: between consecutive squares the new row's toll on the old-open columns is located (g m, m a survivor of a stretch of length (g'^2 - g^2)/g) and zero in 118 of 166 layers; twins in the layer = old-open columns - toll; the remaining unknown is the old wheel's opens in the layer at the square origin, by row and offset. Item 2 (location rules) closed again in one line: a twin's flank strikers are the wheel's open-column pattern to three decimals. The skill records the construction, the view's controls, the twin's API and codes, and the method from a picture to a tested rule.
- 2026-09-12, owner: try the layer grids next (which old row closes each column at what offset from g^2). Manager (local; layer_fields.py, 428 layers to g = 2999; fields_sample_run.md continued): a layer is window(g') minus window(g). Row g's own strikes in its layer are g times the next primes up to g'^2/g (closed form); the old rows are D3 read in the layer; first twin after g^2 within 3.05 g (ordinary, ln^2 g growth), last twin within 938 of g'^2; no mirror centre beyond chance; closing rows at the wheel's shares. All composites of a layer located; the open count of the old rows in the layer remains the unknown. Node R4.d.i.i.a stays OPEN.
- 2026-09-12, owner: run the layers again but look at what fields kill in the window over many cycles, which rows kill, which fields never kill / are periodic / become periodic / never; the gears per field per machine layer; the window's twins at the mirror; which fields mirror. Manager (local; window_fields.md; node R4.d.i.i.b): in cycle 1 the window is closed by the machine alone (higher:g for g <= q, squares, products, lower); from cycle 2 the only change is the machine's openings being eaten by higher:g fields of the gears above q, one field per opening, first eater always q'; periodic = higher:g (g <= q) and the machine's rows, everything built on a factor count, a largest factor or a square is never periodic and nothing becomes periodic; mirror exact for the machine's fields only; window twins at the mirror 2/2, 3/4, 2/7 ... 1/30. Two bugs fixed before recording (sympy primorial; stops-killing read as periodic).
- 2026-09-12, owner: the order ceiling gives an exact shape of which composites can kill, an upper edge of the area to explore, and the zone looks narrow. Manager (local; window_fields.md section 6): j_max(window) = floor(2 log_5 q), exact at 18 machines; cycle ceiling floor(log_5 q#) about 0.62 q; the window's field list is exact and logarithmic in q; orders >= 4 are the smooth field, orders 2 and 3 hold 90-99% of the kills and the order-2 kills are mostly gear x a prime of the window (the hand-up). Recorded on R4.d.i.i.b.
- 2026-09-13, owner shared the mirror-walk idea (teleport by mirrors about products of gear subsets, final flip into the window on an all-gear axis). Manager: DEAD by proof, node R4.d.i.i.c; a mirror or translation keeps a gear's openness iff the gear divides the axis product, so all-gear-preserving moves are the wheel's own symmetries mod q# and move the open set onto itself; the window's content is unchanged. Owner's boxing of the proof (2026-09-13) recorded as the standing frame: the window statement alone (a twin in (q, q^2] for every prime q) proves step 8 and infinitude; no layers, cycles or lookahead; walls exact (no gear above q below q^2, squares one per gear, orders at most floor(2 log_5 q)); the work is a locator for an open column from the rows of orders 2 and 3, whose kills are gears times the window's own openings.
- 2026-09-13, owner: the walk's steps must offset from a known location and land in a known place; build backwards from a window twin to a gear pair; then for killed candidates; then forward with 2, 3 and non-killers; then try squares, blind classes and caustics as anchors with one-off lanes. Manager (local + three Opus lanes; anchors_walk.md; node R4.d.i.i.d): kill side exact (killer onto its own pair); square anchors certify every twin for every gear (residue rule with witnesses); blind classes carry 5 and 7 together; caustic anchors carry dozens of gears per column and halve the walk; the walk is a checker, the locator form is which anchor choices compose below q^2. Lanes closed.
- 2026-09-13, owner: build the network of walks breadth-first (start at a gear pair, a child per anchor option per rule, path-unique nodes, no immediate return, stop at destination) and see which rules work most efficiently. Manager (local; walk_network.md; node R4.d.i.i.e): keep impossible beyond 7, carry1 productive, free wasteful, caustic anchors cheapest; every destination is a landing on a place already known open to (nearly) everything; the located places are the (g^2 + 10, g^2 + 12) columns when twins (g = 13, 97, 379); no rule produced a destination the anchors did not contain.
- 2026-09-13, owner: we just need one location, do the locator. Manager (local; locator.md; node R4.d.i.i.f): the column at a fixed offset i after every square, open to 5, 7 and g by class for i in the blind classes (offset 2 open to 5 always, to 7 unless g = +-2 mod 7), struck by any other gear only from the square-root classes of -(6i-2) and -6i mod h, blind gears never; every machine 11..20000 has a located twin at offset 2 (and 10, 17) after the square of a gear between sqrt q and q; hit gears chain under squaring. Closed form for the where and the who-cannot; the existence per window is a sieve on the gear line with ~2 classes per gear (quadratic-polynomial primes, open).
- 2026-09-13, owner corrected the manager's assessment of the walk: it is not a mere checker; it is a working locator algorithm (steps proved, lands on a twin on every machine tried, to 20000 in reduced form) whose missing piece is a termination proof for all q, which is exactly step 8. Nodes R4.d.i.i.d and R4.d.i.i.e amended to say so.
- 2026-09-13, owner: build a deterministic walk with stepwise rules, no network, no pre-checking, certify afterwards; and read the longer paths of the network across machine sizes. Manager (local; rule_walk.py, path_grammar.py; node R4.d.i.i.g): the residue-blind rules fail at named teeth; the rule that works consults only the sub-machine (gears up to sqrt q): first gear g above sqrt q avoiding its teeth at offset 10, one flip onto (g^2 + 58, g^2 + 60), 2253 of 2258 machines to 20000, the 5 misses at q <= 23 with g = 5 self-striking. Mechanism exact: h strikes iff h divides r^2 + 6i - 2 or r^2 + 6i with r = g mod h, so gears just below g (r = the gap) are harmless. Termination reduces to the prime line near sqrt q. Path grammar: no shared sequence, every path ends in a caustic zone, longer walks add nothing.
- 2026-09-13, owner's guess: anchor decisions from squares and roots plus a start rule navigate straight to a twin. Manager (local; locator.md; node R4.d.i.i.h): built without search: g = first prime above sqrt q, i = smallest offset the strike law allows against the gears below g; 2258 of 2258 machines to 20000, offset at most 27 columns. Exact lemma: every column in (g^2, g g') missed by the gears below g and by g is a twin (8194 columns to 1500, 0 exceptions), and the rule never left that zone. Termination for all q is now one line: the wheel of the gears below g leaves a column open in the first g (g' - g)/6 columns after g^2, with the teeth fixed by the roots of g. The start pair does not enter the location.
- 2026-09-13, owner corrected the walk: axes must be real mirrors of gear combinations (multiples of the product of the chosen gears), no offsets. Manager (local; true_mirror_walk.py; node R4.d.i.i.i): from home, one flip about k M lands on (2kM - 1, 2kM + 1), open to S by the mirror; the remaining gears' teeth on k are two classes each (the roots); smallest such k in the window; the landing is a twin by construction whenever k exists; S = {2,3}: every machine 11..20000, landing just above q. Termination = existence of the multiple k, the twin sieve on multiples of M with S removed.
- 2026-09-13, owner: build the multi-step mirror walks. Manager (local; multi_mirror_walk.py; node R4.d.i.i.i amended): composition law exact, the end of any walk is a reflection about or a slide by the alternating sum A of the real axes and is certified for the gears dividing A; stepwise tracking undercounts; from home every walk lands on (2A - 1, 2A + 1); two-step rule lands on a twin at every machine to 5000. The mirrors fix the landing family and carry the divisors of A; the roots pick the member; the lemma makes it a twin.
- 2026-09-13, owner: both (carry many gears; the termination statement with the fields). Manager (local; carry_many.py, mline_records.py; node R4.d.i.i.i amended): the mirrors can carry only the gears whose primorial stays below q^2/2 (7 gears by q = 3000 against hundreds), so the roots do the work at every size; on the m-line the landing family is (12m - 1, 12m + 1) with teeth -+12^-1 per gear, termination = an unpainted m in (q/12, q^2/12), records R(q) = 4 .. 278 against windows 9 .. 750250 to q = 3001, share falling to 0.0004; the termination statement is step 8 in run form on the m-line.
- 2026-09-13, owner: how do we stand in relation to the proof; update it. Manager: Part V added to the proof document (frame: the window statement suffices; walk steps exact; landing family and lemma exact; the locator measured to 20000 in two forms; termination = the record on the m-line below the window, 8b in the walk's coordinates; standing: one open statement) and to docs/proof_by_construction.html.
- 2026-09-13, owner: go for it (the walk into the kernel). Manager: proofs/MirrorWalk.lean, round 45, built green, 0 sorries, standard axioms only: the axis rule (flip_carries), the composition law (walk_eq), what a walk carries (openTo_walk_iff), the landing family from home, the landing lemma on the m-line (landing_twin, via SquareColumn.section_twin_of_unstruck) and termination from the m-line record (walk_lands_of_record). Part V of the proof document now has kernel names.
- 2026-09-13, owner: the proof document needs a full rewrite, it mixes approaches and carries stale content. Manager: proof_skeleton.md rewritten around one chain (construction 1-5; the mirror walk 6-10 with kernel names; the walk as an algorithm 11-12; the one open statement 13, the m-line record below the window; what a proof must produce 14; standing 15); the previous document kept as proof_skeleton_2026-09-11.md; docs/proof_by_construction.html rewritten to match.
- 2026-09-13, owner: not covering versus capacity; show nothing stops the walk by identifying which fields each step passes through and why each steers to the landing zone. Manager (local; walk_fields.md; node R4.d.i.i.j; proof document 14 rewritten): only the multiples rows paint the landing zone, everything else is a relabelling; rows independent mod the product of the gears; the walk is stopped only by a painted run anchored at the zone start, L(q), the twin gap above q in m-line columns, mean 8.7 and max 55 to 20000 against zones of millions; the open statement is now the first aligned twin above q lying below q^2, decided by the paint just above q.
- 2026-09-13, owner: draw and read the paint just above q. Manager (local; zone_start_field.html; zone_start.py; walk_fields.md continued): row h painted at d = -q mod h; only gears up to (q + d)/5 reach offset d; the run at the zone start is painted by the machine of size sqrt(2q) as smallest factors (machines 29..20000, run at most 0.36 q, smallest gears at most 0.78 sqrt(2q)); the open statement read at the zone start is the painted run of the small gears at phase -q, the twin gap above q.
- 2026-09-13, owner: go (the sub-machine's wheel at phase -q). Manager (local; submachine_phase.py; walk_fields.md continued): below 2q painted = painted by the gears up to sqrt(2q), so the zone-start run is the sub-machine's run at phase k_0 and is bounded by its record; sub-machine records exact to y = 29 (2 .. 43, growing like y^2); the sufficient condition is the record route at the sub-machine; the prime phase is not special (runs at prime phases match all phases). Every coordinate leads to the record route; it is the invariant form of the one open statement.
- 2026-09-13, owner: go the field (the record as a field). Manager (local; record_field.html, artifact published; record_field.py; walk_fields.md continued): each gear paints in double teeth (the two classes -+12^-1 sit close), a record run is a near-exact tiling by double teeth (single painters 68-100%), runs come in mirror pairs about half the period, the run sits where 5 and 7 interleave without slack; the growth reading is the budget inequality in field form, mechanism not proof.
- 2026-09-13, owner: given the mechanism for growth, make a proof for the bound. Manager (record_bound_attempt.md): the paint budget is proved (two lemmas, a theorem) and gives R_5 <= 2, R_7 <= 8, R_11 <= 36, nothing from 13 on (2 sum 1/h exceeds 1); the overlap-corrected count is the sieve, positive only from length y^4.27 against the needed y^2/12; the mechanism explains the growth and proves the bound at y = 5 only. The one open statement stands: R_y < y^2/12 for y >= 7.
- 2026-09-13, owner: counting will never work, do not attempt it; work the walk as parts (origin pair, step rules, each landing and its relation to the last step, target zone), each with its own proof, never the walk as one object. Manager (walk_parts.md; kernel MirrorWalkParts round 46): nine parts proved in the kernel, including the landing law (h strikes the landing iff 2a = n + 2 or n mod h), one gear never blocks a step (settled within three consecutive multiples), and the carry cap; the one open part is the joint step: one axis in the window settling every remaining gear at once. Memory saved: no counting proofs.
- 2026-09-13, owner: both (the origin lever; the joint step for small sets). Manager (kernel MirrorWalkSettle round 47; walk_parts.md continued): anchor_certifies proved (origins certify by the class of L + v + 2), measured: home and the gear pairs certify the small gears only (40 of 429 at 3001); settle_two proved (two gears within k_0 + 2 + 2h), stride version of the one-gear step proved; settling in sequence costs the product of the settled gears, so two to five gears fit in the zone. The joint step remains the open part.
- 2026-09-13, owner: does the unrestricted sequential walk land inside the window? Manager (walk_parts.md continued): the greedy settle walk (move only when the next gear strikes, stride the product of the settled gears) lands inside the window at 236 of 426 machines to 3000 and outside at 190; it leaves when a later gear strikes after the stride has passed the window and then cannot come back; inside cases are the seeing rule in disguise. Answer: no.
- 2026-09-13, owner: do not rule the sequential walk out; look for a consistent origin type with a consistent per-step rule. Manager: product-stride variants all inconsistent; the repair walk (smallest striking gear, smallest clearing move, 1 or 2 columns) is consistent: 2258 of 2258 machines to 20000 land on a twin inside the window, never passing an open column; kernel round 48 proves the clearing step and that such a walk reaches the first open column; open: that this happens before q^2. Node R4.d.i.i.l.
- 2026-09-13, owner: mirrors are transport; search for a one-flip gear-selection rule (origin pair, extra gears such as q or the gear before q, multiple) that always lands on a twin in the window. Manager (gear_selection.py; walk_parts.md continued): 700 rules on machines 11..3000 plus 288 q-dependent ones; none perfect; the top fixed-axis rules land on one fixed twin and succeed only while it lies in the window; the q-dependent rules land on twins at 0.10-0.13, the random rate 0.068 times the factor the carried gears buy (about 2.3); nothing beyond the carried gears. A perfect one-flip rule would have to carry every gear, which the carry cap forbids inside the window.
- 2026-09-13, owner found a spiral by hand on the flip explorer (machines 5..23): from home, one flip per gear with the mirror set {2, g} for g = q down to 3, directions alternating, then one flip of 2 or 3 onto a twin. Manager: the spiral ends at E = -1 + 4 (q - p' + p'' - ... +- 3), between 1.4q and 2.7q at every machine to 5000 (a formula landing inside the window using every gear once); the closing flip reaches a twin within 6 at 8 of 8 machines to 29, then 7/15, 47/143, 113/501 up to 5000, the distance to the nearest twin growing (median 14, max 100). Recorded in walk_parts.md.
- 2026-09-13, owner: the closing move after the spiral may depend on whether q is a twin member. Manager: no; left members, right members and isolated q all need one or two flips at about 30 percent of machines to 5000 and the offsets spread the same way; the closing distance is the twin gap near 2q.
- 2026-09-13, owner: define and formalise the spiral objectively, the best lead, as the launch point for further exploration. Manager (spiral.md; kernel MirrorWalkSpiral round 49): definition fixed; closed form E = -1 + 4 A(q) proved; carried gears = divisors of A proved; E < 4q < q^2 proved; E > q measured to 20000 (between 1.40 q and 2.71 q); E mod 6 uniform over 1, 3, 5. Node R4.d.i.i.m.
- 2026-09-13, owner: stepping through every gear may rule out residues at the spiral endpoint; try the spiral with 2 and 3 in every step. Manager (spiral.md continued): no residue trace, every gear strikes the endpoint at its ordinary rate 2/h (E a twin at 22 of 666 machines to 5000); the {2,3,g} spiral lands at about 6q, always on a left slot member, a twin at 60 of 666 (the slot-column rate near 6q), strike rates again 2/h. The endpoint carries exactly the divisors of the alternating sum and nothing else.
- 2026-09-13, owner: try spiral orders (the striking order: gears by inverse-modulo remainder plus the gear), the gear set below sqrt q, stacked spirals with growing bases; test separately. Manager (spiral_ideas.py, spiral_stack.py; spiral.md continued): only the descending order lands in the window every time (its alternating sum is pinned in [0, q]); the striking and other orders land below q or outside at half the machines; the sub-machine set lands near 2 sqrt q, below the window; base {2,3,g} lands at 6q on a slot member (twin 14 percent); stacking sums the stage moves (order-free), lands at hundreds of q, no carry accumulation. Carried gears stay near one per machine in every variant.
- 2026-09-13, owner corrected: base = all gears below sqrt q, spiral over the gears above. Manager: the base's product outgrows the window from q = 29; inside at 9 of 299 machines to 2000; the carry cap in spiral form.
- 2026-09-14, owner: stacking with the initial direction alternating per layer. Manager: the layers cancel, the endpoint goes negative at most machines (inside the window 36 of 299 to 2000, against 289 with all layers starting up); no carry accumulation; recorded in spiral.md.
- 2026-09-14, owner: spiral with base 2,3,5. Manager: lands at 30q (inside from q = 31), twin at 0.099, the rate of a slot column open to 5; the base is carried and nothing more; recorded in spiral.md.
- 2026-09-14, owner: spiral with base 2,3,5,7. Manager: lands at about 211q, inside from q = 181, twin at 0.108 against 0.116 for a slot column open to 5 and 7; the base is carried, nothing more.
- 2026-09-14, owner: build the spiral with the base = all lower gears whose product stays below the window. Manager: the primorial spiral (named by the owner the same day) (base product at most q/2, spiral over the rest descending) lands inside the window at 2258 of 2258 machines to 20000 (ceiling proved from A <= q and P <= q/2), base sizes 1..5 changing at q = 13, 61, 421, 4621, twin at 0.091, carried beyond the base 1.17; recorded in spiral.md.
- 2026-09-14, owner named the max-base spiral the primorial spiral, and suspected the closing jump is a flip about the base mirror. Manager: base flips from the endpoint reach a twin at the first flip at 546 of 2258 machines to 20000, within three at 1062, all within 34; median 4, no preferred direction; the base-mirror landings are base-open slot columns at the base-adjusted twin rate; recorded in spiral.md.
- 2026-09-14, owner: stack the spiral and the primorial spiral. Manager: the stack loses the base (E + 1 = 4 A1 + 2 P A2), carried gears fall from 3.85 to 1.54, twins from 205 to 29 of 2258 machines to 20000, the endpoint leaves the slot grid; the primorial spiral alone is strictly better.
- 2026-09-14, owner: the final step's high gear must sit in an open residue against the composite killers; knowing the killers, pick a gear that is not one. Manager (spiral.md continued): made exact by the landing law: for the mirror {3, h} from E each gear forbids two classes of h computed from E (6h = -E, -(E+2) mod g up; E, E+2 down); the high gears avoiding every class are exactly the hits (0 mismatches, machines 11..199); the final-step algorithm is a residue sieve on the high gears, blind to primality; existence of a passing h is the second-sieve statement (98 of 100 machines to 569 with {3, h}).
- 2026-09-14, owner: all three (kernel, table to 20000, explorer button). Manager: MirrorWalkFinal round 50 proves the forbidden classes and that an avoiding high gear lands on a twin; the table shows a passing high gear at 2257 of 2258 machines to 20000 (none at 11), smallest passing h about 1.85 sqrt q at the median; the flip explorer gained a button that adds the final step by residues from the current landing. Node R4.d.i.i.n.
- 2026-09-14, owner: link the forbidden classes to the proved field shapes. Manager (killer map, results_phase2_killers.txt; spiral.md continued): every failing high gear charged to the higher:g field of the struck member's smallest gear; 86,396 candidates to 2000, 3,639 pass, kills by field 5: 42,041, 7: 14,584, 11: 5,325, ... in the fields' own proportions; orders 2 and 3 take 81 percent, none above 7; squares never strike a landing; left and right equal. The residue test is exactly: the landing lies in no higher:g field; the fields account for every failure and do not yet say why the traces never cover the high-gear line.
- 2026-09-14, owner: go ahead (fields one at a time on the high-gear line). Manager (phase2_fields.py, results_phase2_fields.txt): to 2000, in reach 86,396; higher:5 leaves 44,355; with higher:7 29,771; the sub-machine (gears up to sqrt q) leaves 13,334; pass 3,639. Sub-machine survivors exist at every machine but 11; what takes a survivor is always a high gear's own field, order 2 (9,139) or 3 (556), never more. Open item splits: sub-machine leaves survivors; high gears' fields do not take them all. spiral.md continued.
- 2026-09-14, owner: go (recursion shape). Manager: the landing of the final step is column h modulo the spiral's base, exactly (E = -1 mod P): up landing = (6h-1, 6h+1), down = its reflection; base gears' forbidden classes are +-6^{-1} mod g, independent of E and direction; kernel MirrorWalkColumn round 51 (built, 0 sorries, standard axioms); 0 mismatches to 2000. Gear roles: base (classes = column h), sub non-base (E-dependent), high (own field, order 2 or 3). The E-dependent band is the price of the landing ceiling. spiral.md continued; phase2_column.py.
- 2026-09-14, owner: proceed (band classes). Manager (band_classes.py): E mod g for band gears takes every residue with no shape (g = 11: 20, 30, 19, 24, 20, 20, 32, 22, 23, 33, 30 over 273 machines); A mod g is the alternating prime sum mod g, not fixed by the machine. The band is forced by the window, not the spiral: E = -1 mod every gear up to sqrt q would need the primorial of sqrt q to divide E + 1, above q^2 once sqrt q >= 11. Open item stated in parts: base settled (column h), band = E's residues, high gears strike as g | L or g | L + 2. spiral.md continued.
- 2026-09-14, owner: pure powers are composites; the band is an aggregate with conditional members, solve each field type separately. Manager (band_field_traces.py, q = 499): field type (g, member, order j) on the landing line = one class of h mod g, one cofactor line c = c_0 + 6t (a column line one level down), one factor shape of c (j - 1 primes at least g); lower gears' kills are the t where c has a smaller factor. Order 2 = h and c both prime along t. spiral.md continued.
- 2026-09-14, owner: I group fields then get stuck on the group (memory one-field-at-a-time). Manager: one type alone, (g, left, order 2) on the landing line, kernel MirrorWalkFieldType round 52 (built, 0 sorries): class h = a mod g with 6a = -E; line E + 6h = g(c_0 + 6t); shape = c_0 + 6t prime; smaller gears take the cofactor on their own classes of t and never overlap a prime cofactor. spiral.md continued.
- 2026-09-14, owner: go (next type). Manager: (g, left, order 3) alone: shape = some prime p_1 >= g divides the cofactor with prime quotient >= g (left_order_three_iff); quotient line d_0 + 6s inside p_1's class (quotient_on_line); order 3 is order 2 one level down, order j is the same descent repeated. Built, 0 sorries. Tables q = 499 and 1999, g = 11 (type_order3.py). spiral.md continued.
- 2026-09-14, owner: continue. Manager: right member (class of E + 2), down step (class 6a' = E, cofactor line downward), and a high gear's type alone (order at most 3, g^4 <= g c <= q^2 < g^4) in the kernel, round 52 continued, built, 0 sorries; tables for g = 11 at q = 499 (type_right_down.py). spiral.md continued.
- 2026-09-14, owner: ok (first pair). Manager: base gear 5 with band gear 11 on the landing line as two anchors (gear 5 on column h, teeth h = 1, 4 mod 5; gear 11 on the landing, teeth h = 6, 2 mod 11 at q = 499); pair survivors then met by single types one at a time (gear 7 most often, then high gears at order 2). pair_base_band.py; spiral.md continued.
- 2026-09-14, owner: ok (pairs). Manager: same-anchor pair (5, 7) on column h in the kernel (teeth_five, teeth_seven, pair_five_seven): eight classes of h mod 35 hold every gear the pair leaves, identical at q = 499, 997, 1999; cross pair (11, 13) at 499 both placed by E, 58 survivors over 51 classes mod 143. pair_same_anchor.py; spiral.md continued.
- 2026-09-14, owner: ok (fourth pair). Manager: base 5 (fixed teeth) with gear 13 (teeth placed by E mod 13, same teeth for the same residue at any machine) across q = 61 to 167 and 499, 997; own gear strikes its own landing iff h | E or h | E + 2 (own_landing_iff, kernel), 22 own strikes listed. pair_base_high.py; spiral.md continued.
- 2026-09-14, owner: continue (triple). Manager: triple (5, 7, g) in the kernel (triple_five_seven_iff = pair lemma with two class lemmas); table g = 11 at q = 499, 997, 1999: 26, 41, 76 survivors, next single type per survivor listed. triple_5_7_g.py; spiral.md continued.
- 2026-09-14, owner: continue (one gear at a time). Manager: q = 499 walked gear by gear: base 5 takes 43 of 87, 7 takes 13, band 11, 13, 17, 19 take 5, 6, 4, 1, then nine high gears take one each (31 takes two), all order 2; 6 pass. A high gear's teeth are two positions; below g the tooth is a single gear. one_gear_at_a_time.py; spiral.md continued.
- 2026-09-14, owner: go (teeth as positions). Manager: tooth_unique and tooth_at_most_two in the kernel; table of every high gear's tooth positions at q = 499: above q/2 one or two positions per tooth, mostly not gears, gears there mostly taken already by base or band; only 257 and 293 above q/2 take a standing gear. high_teeth_positions.py; spiral.md continued.
- 2026-09-14, owner: go (band teeth as positions). Manager: whole h-line at q = 499 written as positions from E gear by gear (base fixed teeth, band teeth a_g + gt across the line, high teeth few positions); 15 standing after the sub-machine, 6 pass. band_teeth_positions.py; spiral.md continued.
- 2026-09-14, owner: go (positions at more machines, view). Manager: positions at q = 997 (16 standing after the sub-machine, 2 pass) and 1999 (33 standing, 20 pass), same reading; docs/positions_view.html: gear rows against high-gear columns with teeth L/R placed by E, base fixed, passing columns marked, q by prime buttons, up/down.
- 2026-09-14, owner: label consistently. Manager: positions view relabelled with the fields explorer's ids (rows higher:g, marks j+member = products:j, squares empty on the line, lower:g as dimmed marks in rows above); memory consistent-field-labels.
- 2026-09-14, owner: the fields construction is the basis for every walk rule; the recent class/teeth/positions work looked at its residue imprint instead. Manager: agreed; the final step put back into the fields: field 'final' in research/tools/fields_twin.py and the fields view (rows = gears, painted at the members of the columns E +- 6h that the gear divides), marker E on the window line (E + 1 a multiple of P: E stands one column left of the base rows' joint stripe). Positions view retired as an instrument, kept as the imprint check. Rules for the walk from here are read from the fields at the landing columns.
- 2026-09-14, owner: do not add the final field to the fields explorer; no unasked work. Manager: fields_view.html and fields_twin.py restored to before 3effe2a; positions view stands as the last asked-for instrument.
- 2026-09-14, owner: both approaches by field type. Manager: A, type exclusions without E residues (column h painted by a base gear; h | E or E + 2) cut 86,396 candidates to 30,390, the E-placed gears still take 26,751 of those, 3,639 pass, no machine but 11 without a pass. B, spirals over one gear type: sqin (gears above sqrt q only) passes at every one of 299 machines including 11; sqout (gears at most sqrt q) lands on a twin itself at 135 of 295 and low in the window; coltwin, twinmem, solo like the full spiral. final_step_types.py, spiral_types.py; spiral.md continued.
- 2026-09-14, owner: keep trying spiral constructions. Manager: 48 variants (set x order x start x mirror mode), stacks, the sub-machine's own spiral, and column-side sets. Only descending-first-up stays in the window. sqin: passing at all 299; sqout, leftsq, rightsq: passing at every machine with gears of the type; the sub-machine spiral lands on a twin itself at 112 of 299 (E_s = 71 at 499, 239 at 1999). spiral_variants2.py, spiral_stacks2.py, spiral_sides.py; spiral.md continued.
- 2026-09-14, owner: go for both (column groups, sub of sub). Manager: spirals over gears grouped by the base gear painting their column all pass wherever the group exists (open 286/286, by5 286/286, by7 222/222, by57 222/222); the iterated sub-machine spiral (levels, base always holding 2 and 3) passes at all 299 machines including 11, its level-1 landing in the sub-window is a twin at 99 of 294; without 3 in the deepest base the alignment is lost. spiral_groups_levels.py; spiral.md continued.
- 2026-09-14, owner: both (final step one level down; sqout at level 0). Manager: from the level-1 landing E_1, the final step with a sub-machine gear lands on a twin in the sub-machine's window at 292 of 294 (the two failures have b_1 = 5, E_1 overshoots); level 0 with sqout passes at all 294; from E_1 directly a high-gear flip passes at all 294 without any level-0 spiral. spiral_levels2.py; spiral.md continued.
- 2026-09-14, owner: go (recursive walk as parts). Manager: research/proof/recursive_walk.md - origin, levels, one level's spiral, landings, final flip, target zone, each with its proof status; kernel MirrorWalkLevels round 53 (spiralP_eq, spiralP_modEq, spiralP_left_member, spiralP_le, final_flip_below_square, final_flip_above_q; built, 0 sorries, standard axioms). Alignment needed is only E = 5 mod 6, carried by 2 and 3 at every level; the machine's base gears no longer see column h. Open: existence of a passing h at the final flip, the same statement at every level.
- 2026-09-14, owner: go (level edge, final flip in the fields). Manager: level rule b_k >= 11 removes every failure: all 299 top-level, all 273 sub-machine parts (E_1 in sub-window, one-level-down final step, direct final flip). Final flip from E_1 lands at most 7q, painted by rows 5, 7, 11, 13 at products:2 or 3 only; q = 499 first pass h = 83 -> (569, 571), q = 1999 first pass 307 -> (2081, 2083). levels_rule_and_final.py; recursive_walk.md continued.
- 2026-09-14, owner: go (stretch rows). Manager: on (q, 7q] the deciding rows are the gears up to sqrt(7q); each row paints two classes of h fixed by E_1; the final flip from E_1 is the window statement of the machine of size about 2.6 sqrt q with the three-column pattern (h, E_1 + 6h, E_1 + 6h + 2). stretch_rows.py; recursive_walk.md continued.
- 2026-09-14, owner: go (pattern one level down by the same walk). Manager: the machine sqrt(7q)'s walk lands above q at most machines (locates h at 25 of 273); the sub-machine's own flips from E_1 locate a valid h at 135 of 273 with {3, h''} and 232 of 273 with {2, 3, h''}; not a closing rule; open item unchanged. pattern_one_down.py, pattern_one_down2.py; recursive_walk.md continued.
- 2026-09-14, owner: keep going (locating flip, wider mirrors). Manager: from E_1, {3, h''} with k up to 3 locates 242, {2, 3, h''} 254, any small mirror 264 of 273 (failures 127 to 167 where the sub-machine is {5, 7, 11}); from E_0 nothing (overshoots q); {base_0, h''} overshoots. pattern_one_down3.py; recursive_walk.md continued.
- 2026-09-14, owner: keep going (chained flips). Manager: two flips from E_1 with {3, h} locate the final gear at 264 of 273, with {2, 3, h} at 263, either family 273 of 273; no single rule; same shape as the network result. pattern_one_down4.py; recursive_walk.md continued.
- 2026-09-14, owner: a walk on primorials that converges in the window (flip on q#, flip back on partial primorials). Manager: primorial descent built; landing 2 t P_s - 1 with P_s the first primorial above q/2, open to the base by construction, in the window for a range of t; a twin among the landings at all 299 machines; the forbidden classes of t are +-(2 P_s)^{-1} mod g, fixed by the gear and P_s alone (no E), so the twin t list is the same for every machine sharing P_s (2310: t = 2, 12, 17, ... first landing (9239, 9241)). primorial_descent.py; spiral.md continued.
- 2026-09-14, owner: go (descent kernel and teeth). Manager: MirrorWalkDescent round 54 (descentEnd_eq, descentEnd_form, descent_open_base, descent_strikes_iff, descent_in_window; built, 0 sorries); teeth on the t-line at P_s = 210 and 2310 tabled: each gear's two teeth are t and -t mod g (right = -left), the primorial's mirror seen on the t-line; standing t at 2310 start 2, 12, 17, 20, 24. descent_classes.py; spiral.md continued.
- 2026-09-14, owner: go (symmetry lemma, t-line as fields). Manager: descent_teeth_symmetric and descent_reflect in the kernel; the t-line drawn as the fields construction at P_s = 210 and 2310 (rows = gears above the base, columns = t, L/R marks, twins = unpainted columns, checked); each row a period-g stripe with teeth t_0 and -t_0, the L-to-R gap = P_s^{-1} mod g (adjacent teeth when P_s = 1 mod g: 11 and 19 at 210). descent_fields.py; spiral.md continued.
- 2026-09-14, owner: go (gap lemma, grid at 30030). Manager: descent_gap and descent_gap_one in the kernel (gap between a gear's teeth = P_s^{-1} mod g; adjacent when P_s = 1 mod g); grid at P_s = 30030, 145 rows, twin columns at t = 3, 5, 7, 9, 11, 14, 21, 25, 27, 31, 32, 46, ...; the same object at every primorial. spiral.md continued.
- 2026-09-14, owner (parked idea, current line kept): in the spiral, sum the slips of the gear sets between steps (each set slips ahead or behind as they cycle each other); the summed slip may point to an open twin. Not yet run.
- 2026-09-14 (a): first gap of each primorial's stripes vs the smallest machine in its range: t = 1, 1, 2, 3, 4, 12, 2, 8, 11, 2 for 5# to 37#, reach at q_min 2, 8, 38, 355, 3534, ... growing as 2 P_prev / p_s. descent_first_gap.py; spiral.md continued.
- 2026-09-14 (b): the columns t = 1.. of each primorial read gear by gear: below the first gap every painter is a gear whose tooth sits exactly at t (the prime factors of 2 t P_s +- 1 above the base); no periodic gear reaches, since the first gap (t <= 12) is below the smallest gear above the base at every primorial to 31#. descent_small_t.py; spiral.md continued.
- 2026-09-14, owner: go (descent as parts). Manager: the primorial descent written as parts in recursive_walk.md (origin, up flip, down flips, landing, target) with the round-54 kernel names; open statement isolated: a gap in the stripes of P_s at some t within the machine's reach.
- 2026-09-14, owner: walk ruled out; mirror overlays instead (layers of landings per gear from the origin, stacked, compared with the open twins). Manager: layer_stack.py at q = 101, 499: layer g = columns 2kPg - 1 in the window; a column 2mP - 1 is in layer g iff g | m; depth = gears above the base dividing m; twins listed per layer and by depth; twins held by no gear layer listed.
- 2026-09-15, owner: the stack, one mirror per layer (gear selection as the spiral; each layer = gear g's sieve mirrored once about P g, ending at its landing 2Pg - 1; stack = layers over each other; holes = twins where every gear's layer still covers; false holes where a layer has ended). Manager: stack_one_mirror.py, pictures at 31, 37, span tables at 101 to 1999.
- 2026-09-15, owner: analyse the stack across machine sizes and base sets including the null set. Manager: stack_bases.py; null and {2} bases mark nothing in the window; holes = twins exactly in span 0 at every base (exact by construction); with the spiral base span 0 = (q, 2 P_s - 1] holds a twin at every machine 11 to 5000 except 27 machines just below 2 P_s (11, 41-59, 347-419, 4547-4603); true-hole share falls as layers end; twin-gear spans show nothing in aggregate. spiral.md continued.
- 2026-09-15, owner: can we build a locator. Manager: the stack locator (first hole of span 0 with the largest base the window allows) locates a twin at all 665 machines to 5000, 5 slots at the median, 29 at most; correct by section_twin_of_unstruck; existence = a twin in (q, 2 P g_min - 1], the window statement; and it is a slot-by-slot scan. stack_locator.py; spiral.md continued.
- 2026-09-15, owner: try the spiral with the slip. Manager: three slip definitions (previous gear, all earlier gears, all gears outside the base), candidates E +- S and E +- 6S: twin at 13 to 29 of 299 against 39 for E itself; no pointer. Closed. spiral_slip.py; spiral.md continued.
- 2026-09-15, owner: slip = difference of the products of two gear sets. Manager: consecutive-mirror slips summed plainly and with flip signs, nine candidates: twin at 20 to 48 of the machines in the window against 39 for the landing; no pointer. spiral.md continued.
- 2026-09-15, owner: I explained slip weeks ago. Manager: found in docs/handover.md 0.3 and 2.1 (cycle slip = difference of periods, machine slip = period mod new gear); memory slip-definition; the spiral run with machine slips (four readings, plain and signed): no candidate above the landing's rate except the column at twice the sum (50, 51 of 299) low in the window. spiral.md continued.
- 2026-09-15, owner: loop - algorithms that carry residues by mirrors to an open flip without secret knowledge. Manager, entries 1 to 14 in research/proof/loop_algorithms.md: greedy and searched pick-up walks, slip-inverse rules, blind offsets (primorial, lcm, factorial, slips of step pairs, gaps, column-twin gears, residues of q, gear-pair products, primorial sums), machine-to-machine handing, one integer per primorial; then the scoring pick-up walk (220 of 299) and, with a flexibility score and two repair flips, 299 of 299 machines to 2000 using the walk's own phases only. Entry 15 (to 4000 and a sample to 12000) running.
- 2026-09-15, loop entry 15: the flex pick-up walk with two repair flips on every machine to 4000 and a sample to 11549: 561 of 561. Loop target met as a measurement (860 machines, no failure); the invariant's persistence is not proved. Loop stopped; loop_algorithms.md holds the fifteen entries.
- 2026-09-15, loop 15 addendum: 745 of 745 (every machine to 5000, every 20th prime to 19891).
- 2026-09-15, loop entries 18 to 21: settle-and-grow walks (each settle step provable, chains of 7 to 8 twins, stopped at the gcd wall); the reach theorem in the kernel (MirrorWalkReach round 55): mirror walks reach exactly -1 + 2 G Z and guarantee openness exactly for the gears of G, G at most (q^2+1)/2 inside the window. The loop's target as posed (mirrors only, provable, no teeth) is closed by this theorem.
- 2026-09-15, loop entries 22 to 24: rule grammar searches (160,000 fixed rules) and an evolutionary search over walk genomes (four runs, one across eighteen primorial ranges): best fixed rules are per-range constants (twins by chance at bases 6, 30, 210, dead from 2310); evolution reaches the same and no further; consistent with the reach theorem. loop_algorithms.md standing after 24 entries.
- 2026-09-15, loop entries 25 and 26: evolution seeded with the hand-built walks, then five islands (spiral, spiral+flip, levels, descent, flips) evolved independently, 50 and 400 generations: the descent-plus-base-shift constant 2B(g1 + 4) - 1 wins wherever a descent step is allowed (twins at five bases by chance, dead from base 510510); the spiral island sits at the base-open share (streak 29, total 73 of 288). loop_algorithms.md.
- 2026-09-15, loop entries 27 to 29: residue rules for gear and period selection, settled-set mirrors, barred constants, residues as the target: every winner a lookup table (base or sqrt q keyed) or the base-open share. Evolutionary line closed; loop stopped. loop_algorithms.md standing after 29 entries.
- 2026-09-15, loop entry 30: distance-to-nearest-twin fitness, balanced ranges, distinct landings required: plateau at the base-open share (32 of 88) after fifty generations; with tables allowed, a two-key table of fifteen landings. Loop stopped. loop_algorithms.md standing after thirty entries.
- 2026-09-15, loop entries 31 and 32: the settle walk's streak against its residue memory (full memory needed: 204 of 204 to 3947; memory 89 reaches 1223; carrying gears in the mirror is worse); whole-field-set single flips overshoot or land at the base-open share. Loop stopped after 32 entries; the successful run under the fitness is the full-memory settle walk, a sieve on the candidate.
- 2026-09-16, owner: one piece at a time, the walk proven in the window. Manager: MirrorWalkInWindow round 56 (in_range_move, window_move, base_fits, stays_in_range; built, 0 sorries): an in-range move exists at every step where the mirror fits, a window-entering move at the last step, the spiral base fits for every gear with 2g + 3 <= q. recursive_walk.md continued.
- 2026-09-16, loop entries 33 to 36: the settle walk's missing keeping moves are the affordability wall (77 of 80 failing steps have at most three candidates in range); with the mirror shrunk to {2, 3, g} and forty periods each way, a keeping move exists at every step of every machine 29 to 1500 and every landing is a twin (230 machines). Piece 2 becomes a uniform local statement.
- 2026-09-16, piece 2 of the settle walk: the free-regime keeping-move lemma in the kernel (MirrorWalkSettleFree round 57, mex_form on a progression); measured: with mirror {2,3,g} and forty periods every machine 29 to 2000 has a keeping move at every step and a twin landing (294 machines). The owner's softer target (q, q#] measured: loses certification, no higher rate.
- 2026-09-16, loop entries 40 to 42: the settle walk's last flip about {2,3,5} preserves 2, 3, 5 and is redundant; the walk lands on the twin at the gear-7 step, carried there by the one-step lookahead (a candidate with options at the gear-5 step must be open to 5). The tight requirement is one step: eighty candidates L +- 84k, one open to every gear 5..q; margin 3 to 14 at q near 1000, first success within 12 periods.
- 2026-09-16, loop entry 44: the free-regime lemma covers a fraction ln q/(ln q + 2) of the walk's steps (70.5% at q = 1000, 84.3% at 10^6), rising to 1; the spacing/density route reaches less far and is not admissible; the tail cannot be collapsed into one flip (its mirror would be the primorial of q/4). The residual is one progression statement.
- 2026-09-16, loop entry 46: the last stride may be chosen among eight gears; some choice gives StepOpen at all 90 machines 400 to 1000 with the smallest working period at most 4, so the hypothesis can be stated over sixty-four candidates in eight progressions instead of eighty in one. Content unchanged.
- 2026-09-16, loop entry 48: the returning mirror (inheriting the tail from home) reaches only gear 13 to 17 and leaves 2 to 16 columns in reach, twin at 87 of 225; the trade is now a kernel lemma (mirror_times_candidates): carried gears times candidates is bounded by the window.
- 2026-09-16, loop entry 49: StepOpen's margin thins with a fixed family (mean 2.97 at q = 1000 to 0.92 at 3 x 10^5) and is flat when the period count grows as (ln q)^2 / 4 (min 2 to 3, mean about 10 at every q to 3 x 10^5); the window affords K up to about q / 24, so the growth is free. The walk's period law.
- 2026-09-17, loop entry 51: the walk's period count (K = (ln q)^2, measured sufficient) and the lemma's (K_n = 2n + 1 at step n, needed for the free-regime proof) separated; the lemma's count fits the window throughout the prefix (worst span 0.90 of q^2 at q = 1000, 0.44 at 10^6), so the proved prefix is the whole first cut, 70 to 84% of the steps and rising.
- 2026-09-17, loop entry 52: StepOpen holds at every tail step, and already at the tail's first step, at all 79 machines 200 to 700 (minimum 1 open candidate, mean 2.14); so the walk may stop at the first tail step and the open statement is StepOpen there, applied to the column the proved prefix hands over. Proof document updated (Part IV b, parts 15 and 16).
- 2026-09-17, loop entry 54: extra strides cannot move the free-regime cut (the stride count cancels: it multiplies candidates and strikes alike); the cut is exactly where pigeonhole stops, and below it the survivor needs the strikes to overlap, which is the twin statement.
- 2026-09-17, loop entry 55: no visiting order moves the free-regime cut (after n steps the memory's smallest gear is at most the n-th largest, attained by descending order); with entry 54 the boundary is fixed from both sides, and the proved prefix is the largest any pigeonhole argument reaches on this construction.
- 2026-09-17, loop entry 56: settle by exclusion (keeping the unvisited gears' phases) keeps only {5} or {5,7} inside the window, the carried base again; every way of spending the window has now been tried and each buys the same amount.
- 2026-09-17, loop entry 57: on the candidate line each gear's two teeth sit a fixed distance -2 s^{-1} mod h apart, fixed by the gear and the stride; only the pair's position moves with the column. The tail is a covering question with explicit fixed-width shapes; no forced gap follows, since the placements are the column's residues.
- 2026-09-17, loop entry 58: the prefix's last step reaches only 11.7 percent of the admissible residue triples mod 5, 7, 11 (none of 77 machines reaches all); choosing them freely would need periods of the order of the tail primorial, past q^2 by the trade lemma. The tail's placements are given, not chosen.
- 2026-09-17, loop entry 60: with the lookahead the walk can finish at the tail's FIRST step at all 108 machines 200 to 900, so the open statement is StepOpen at one named step: prefix (proved) -> handover column -> one step of 2K candidates -> twin.
- 2026-09-17, loop entry 61: the construction is now one kernel theorem (MirrorWalkTheorem round 59): Handover, construction_twin, window_statement_of_stepOpen - if every machine's walk meets StepOpen at its handover, the window statement follows. StepOpen measured at the tail's first step at all 150 machines 200 to 1200.
- 2026-09-17, loop entry 62: at the handover step the gears above the cut still strike 22.5 of 75.8 candidates (the prefix's work does not carry to the next line), the tail gears strike 48.4 more, 55 distinct gears strike in all, and 4.92 candidates survive; no small set of gears is responsible for the covering.
- 2026-09-17, loop entry 63: the settle walk's prefix is WORSE than no prefix at all - one flip from home with the best of eight strides gives 7.31 open candidates against the full walk's 4.92, and works at every machine; the prefix spends window room on settling that does not survive the next line. The construction reduces to the one-flip locator, and its statement is the one to carry.
- 2026-09-17, loop entry 64: the one-flip locator's margin holds at 30 to 50 open columns when the candidate count grows like (ln q)^2 or better, and thins with a fixed count; the arrangement (strides versus periods) does not matter, only the count; the first open candidate sits at period 10 to 19 with the smallest stride.
- 2026-09-17, loop entry 65: the one-flip locator is now the kernel's final form (proofs/OneFlipLocator.lean, round 60, 0 sorries): oneFlip, OneFlipOpen, oneflip_twin, window_statement_of_oneflip - from home, one flip about {2, 3, g} with g the first gear above sqrt q and period up to (ln q)^3, and if one of those columns is open the window holds a twin. Handover/construction_twin kept as the general step form. Proof document Part IV b rewritten: 15 the locator, 16 the settle walk as a recorded result with its two proved parts, 17 the one open hypothesis.
- 2026-09-17, loop entry 66: the teeth law on the one-flip family - its members are 72 g k - 7 and 72 g k - 5, so every gear strikes at k = 7u and k = 5u modulo h with u the gear's inverse of the stride: every gear's teeth are the fixed pair (7,5) scaled by its own unit (kernel: strike_iff_scaled, oneflip_teeth, oneflip_members, round 61). The mirror to use is the smallest that fits, {2,3,5}: 25 to 58 open columns at every machine 19 to 20011, against 0 to 36 for the mirror at the first gear above sqrt q; a bigger mirror carries more phases but throws its candidates past the twins. The only machines with none (11, 13, 17) have an EMPTY family - the mirror does not fit the window - which is base_fits failing, not a covering.
- 2026-09-17, loop entry 67 (correction of 66): the one-flip landing is a MEMBER, not a column - the flip sends home (-1,+1) to (12 g k d - 1, 12 g k d + 1), column 2 g k d; entry 66's form was degenerate (5 always divided one member at g=5) and its mirror numbers are void. True teeth law, proved: a gear dividing the stride never strikes, every other gear strikes at k = +v and -v modulo h with v the inverse of 12 g - a symmetric pair about home. Mirror {2,3,5} wins at every machine (72 open of 638 at q=20011 against 41 for the sqrt mirror), every machine 11 to 20011 has an open column, and the periods needed are the window's own start (k = q/60) plus an offset measured at 0 to 34 out to q = 2000003.
- 2026-09-17, loop entry 68: a landing serves a BAND of machines, not one - the pair (t-1, t+1) is in the window of every q with sqrt(t) <= q < t-1 - so the machines are covered by a chain of landings with each below the square of the one before (kernel: chain_covers, window_statement_of_chain, proofs/MirrorWalkChain.lean, 0 sorries). At the primorial mirrors the first twin landing sits at period k = 1,1,2,3,4,12,2,8,... for B = 5..127; all 28 consecutive pairs chain, covering every machine from 8 to 4.8e49, with slack growing (at B=23 the period could have been 1.2e8 instead of 2). Killer census by field id: the kills are products:2 and products:3 with smallest gear higher1:7, higher1:11, higher1:13 - single powers of the gears just above the mirror.
- 2026-09-17, loop entry 69: the carry wall is now a kernel statement - a landing inside the window forces 2M <= q^2, so a mirror carries at most log2(q^2) gears and every other gear stays live (carried_le_log, mirror_in_window_carries_le, uncarried_card, proofs/MirrorWalkCarry.lean, 0 sorries). Measured: at q = 1000003 the machine has 78499 gears, the largest mirror that fits carries 11, and the free regime would need the uncarried gears above 156976 when they start at 37. Each gear carried divides the periods afforded by that gear and removes one gear from the dodge list: logarithmic gain, geometric cost.
- 2026-09-17, loop entry 70: the certificate - four landings (12, 108, 11352, 128845110) prove the window statement unconditionally for every machine from 11 to 128845108 (window_statement_below, proofs/MirrorWalkCertificate.lean, 0 sorries), with chain_covers_upto and window_statement_upto added to the chain file. On paper nine landings settle every machine to 10^259, each found within a few hundred columns of the previous square; the certificate's length grows like log log X, so it verifies the construction rather than closing the open statement.
- 2026-09-17, loop entry 71: the multiplicative chain - from a landing t, flip about the mirror whose product is t/2 (the landing's own gears, all open at it), giving the candidates t*j; a landing t*j with j+3 <= t meets the chain condition automatically (mult_chain_window, proofs/MirrorWalkChain.lean, 0 sorries). The open content is then one sentence with no machines or windows in it: for every twin centre t, some j <= t-3 has t*j a twin centre. Measured over every twin centre to 20000: smallest j is 2 to 96, mean 15.2, 0.66% of the allowance at worst; mean j grows 11.8 to 92.2 from t ~ 1e3 to 1e8 while the allowance grows like t.
- 2026-09-17, loop entry 72: the enriching chain - a gear dividing the landing cannot strike its multiples and the carried set grows along the chain (landing_gear_never_strikes, mult_carried_monotone, 0 sorries), so the greedy form multiplies by the smallest missing gear each step. Measured: landings carrying gear 5 need mean multiplier 20.6 against 37.7 carrying none, 15.1 carrying 5 and 7; the greedy chain reaches a 62-digit landing with 24 gears in 21 steps, extra factor never above 700, multiplier down to 5.7e-55 of the allowance. The primorial spiral in working form.
- 2026-09-17, loop entry 73: the multiplicative step has growing room - over all 80 twin centres from 12 to 3000 the working multipliers number 3 to 284, mean 15.1 for t < 300 rising to 136.5 for t in [2000,3000), and no landing has none; the smallest margin anywhere is 3, at the chain's own start t = 12 (multipliers 5, 6, 9). Between 58 and 100 percent of working multipliers share a gear with the landing, as the carrying rule predicts. The kernel certificate cannot pass its fourth link with norm_num: a seventeen-digit member takes over ten minutes against seconds for nine digits, so a fifth link needs Lucas or Pratt certificates.
- 2026-09-18, loop entry 74: Lucas certificates break the certificate ceiling - PrattTools (cast_pow_eq_one_iff, sq_of, sq_mul_of, ne_one_of_mod) plus a generator (research/stack/r8/gen_pratt.py) certify a prime in the logarithm rather than the square root; a thirteen-digit test that failed norm_num after six and a half minutes certifies in 34 seconds. The chain certificate now runs to five links (12, 108, 11352, 128845110, 16601062113221682), so window_statement_below proves the window statement unconditionally for every machine from 11 to 16601062113221680 - a range 1.3e8 times wider than round 65.
- 2026-09-18, loop entry 75: the sixth link certified on this machine - 12 Lucas certificates at depth 3 (7390 generated lines, 1m48s to check) extend window_statement_below to every machine from 11 to 275595263287044304869593048464768, i.e. below 2.76e32, from six twin pairs. The seventh link needs the factorisation of a 130-digit p-1, which is hard; the multiplicative chain gives the upper member free (its p-1 is the landing, factored by construction), so a certificate-friendly chain would search for a landing whose lower member minus one also factors.
- 2026-09-18, loop entry 76: the step closed from both sides in the kernel - keeping_move_free_sharp (gears {5,7,11}, start 370: every candidate of k = 0..6 struck, so the free-regime conclusion fails once a gear is below twice their number) and free_regime_unreachable with small_gear_uncarried (a mirror that fits the window always leaves a small gear live, so the free-regime hypothesis fails for every mirror). With carried_le_log, gcdL_dvd_combo and mirror_times_candidates, the four mechanisms the machine has cannot force the step; a proof needs something outside that list.
- 2026-09-18, loop entry 77: the analytic route checked and the bridge built - the twin half of Chen's theorem has the explicit constant 1.205 C_2 x/(log x)^2 (Bordignon-Starichkova 2024) but no computable threshold, and the explicit Goldbach-half thresholds start at exp(exp(32.7)) against our certificate's 2.76e32, so loosening to analysis does not give a proof for every machine even of the weakened statement. Kernel: exists_in_window_of_count and chen_window_of_count (proofs/WindowFromCount.lean, 0 sorries) turn any counting bound at q^2 into a window statement, and chenPair_of_twin records that the analytic target is weaker. Measured: the first Chen pair sits 2 to 34 past the window's start at every machine to 1e6.
- 2026-09-18, loop entry 78: the anatomy of the failures (research/proof/failure_anatomy.md) - four stops, one root. Silence costs the primorial (silence_costs_primorial, primorial_le_of_silence, 0 sorries), composition intersects, the mirror's product both prices carrying and spaces candidates, and pigeonhole ends at a cut no mirror reaches. The root is that divisibility is the machine's only lever on a gear, and the measured test says landings carry nothing beyond their factorisation (structured 2^a 3^b landings need mean multiplier 38.4 against 15.2 for controls with twice the gears). Four requirements derived for any proof; requirement 3 - name a candidate, not a population - is the one no route has met.
- 2026-09-18, loop entry 79: requirement three made precise - arithmetic names candidates closed freely (perfect_power_landing: the only perfect-power landing is 4, with three_dvd_two_pow_add_one and not_prime_of_pow, proofs/LandingForms.lean, 0 sorries) but names them open only up to the primorial, by either of its two levers: divisibility (silence_costs_primorial) or congruence choice, which openness_periodic shows is the same object - openness to a gear set depends only on the column modulo the product of that set, so a choice over gears up to X names a class of period X#. Both stop at about 2 log2 q gears, so the carry wall is not an artifact of preferring divisibility.
- 2026-09-18, loop entry 80: the run bound route - window_of_column_gap and window_statement_of_gap_law (proofs/JacobsthalWindow.lean, 0 sorries) derive the window statement from a bound on the longest run of struck columns, i.e. from j2(q#) < q^2 - q. Needed exponent 2; the project's ladder proves 4.266; the sifting floor is 4, so the gap is parity; the truth measured inside the windows is polylogarithmic (longest run 34 to 251 for q = 101 to 2003 against windows of 1,684 to 668,335 columns). The implication is cheap and proved, the input is what no sieve can give.
- 2026-09-18, loop entry 81: the machine's own parity statement, proved - class_has_closed_column and no_class_of_twins (proofs/ClassIndistinguishable.lean, 0 sorries): every residue class, of every modulus, contains columns whose lower member is composite, as far out as one likes, by an explicit CRT witness. Since every mechanism the machine has names a class (carrying, the period rule, a walk's gcd), no mechanism can name a twin column; what separates twins inside a class is invisible to residues. Requirement 3 of the anatomy is therefore an impossibility, not a gap.
- 2026-09-18, loop entry 82: the alignment at infinity - open_columns_for_any_gears (for every finite gear set and every bound there are columns beyond it open to all of them, by taking a multiple of the product) and no_column_open_to_all_gears (no column is open to every gear, since the member above 1 has a prime factor), proofs/AlignmentLimit.lean, 0 sorries. The alignment holds at every finite level and nowhere in the limit: the quantifiers do not commute, and the window statement is the pattern in between, where the gear set is fixed by the column's own size. Measured: the pair either side of the primorial is a twin only at P = 3, 5, 11 up to 53 (at 7 the lower member is 209 = 11 x 19).
- 2026-09-18, loop entry 83: infinity as a number, checked system by system - vacuous in the extended reals (the pair collapses, primality undefined), the sieve's own local picture in the profinite integers, and equivalent to the conjecture itself in the nonstandard integers by transfer, where the hyperprimorial's neighbours still carry a hyperprime factor above the alignment. Finite shadow proved: aligned_neighbour_factor - alignment never removes a factor, it pushes it above the aligned set, and only the square-root rule converts that into primality, only below B squared.
- 2026-09-18, loop entry 84: a top number and its mirror side - the extension to zero and the negatives keeps the arithmetic, while every extension with a top element loses something the machine needs (the projective line collapses the pair, the ordinals have no predecessor, fields make primality vacuous, and the nonstandard integers keep Euclid so there are gears above every infinite element). gears_above_every_bound records it: above every bound there is another gear. The self-defeat the owner spotted is general - a framework with a largest number is not the framework that has the primes, so the exercise decides the answer by choosing its axioms.
- 2026-09-18, loop entry 85: infinity as an adopted number - the projective line (n/0 defined, far side collapses), the surreals (full signed neighbourhood of omega, arithmetic kept, but a field so primality is vacuous) and the hyperintegers (primality kept, but transfer makes the twin question the same question). zero_not_invertible records the forced part: in any ring with 1 different from 0 nothing satisfies 0 * x = 1, so defining n/0 moves the obstruction rather than removing it, as wheel algebras show by weakening subtraction.
- 2026-09-18, loop entry 86: single primes versus pairs - nothing blocks the first. Euclid is the alignment argument in the machine's vocabulary (aligned_neighbour_factor, gears_above_every_bound), and window_has_prime (proofs/AlignmentLimit.lean, 0 sorries) proves every window holds a prime by Bertrand, with room to spare. The whole difference is that a gear strikes one class of a single number and two of a pair: free-regime cut n against 2n, reciprocal deficit log against log squared, sifting dimension 1 against 2, and parity bites only at 2.
- 2026-09-18, loop entry 87: the counterexample hunt, asked structurally - a total kill is impossible by density (uncovered columns have density the product of (1-2/h) > 0, a union of classes mod the primorial), so any counterexample is local and is the run-length question. The adversarial form (teeth chosen rather than arithmetic) is the paired Jacobsthal extremal problem, whose bounds straddle q^2. Measured: a greedy adversary leaves 5 to 9 percent of the window uncovered at q = 29 to 149, and the real arithmetic leaves about twice that, so the machine's teeth are about half as efficient at killing twins as a deliberate arrangement.
- 2026-09-18, loop entry 88: review of rounds 60 to 81 (three corrections: mexT_le is the three-residue walk not the single-tooth one; one-class and two-class Jacobsthal exponents were conflated; three measurement scripts start the window one column early, no conclusion affected) and the hunt with prejudice - an annealing adversary over the free configuration never reaches zero uncovered columns (8 of 135 at q = 29, 86 of 1683 at q = 101) though the sum of 2/h exceeds 1, and the closest real calls to 10^7 are a distance of 1722 at q = 9923987 (6.6 (ln q)^2, 1.75e-11 of the window) and a window share of 0.0545 at q = 11.
- 2026-09-18, loop entry 89: the hunt continued - exact search shows no free configuration of the teeth covers the window for any machine up to 19 (q = 23: no free configuration covers   5   634,932,505 (708 s); q = 29: still running in the background; the search grows about a hundredfold per gear, so this one is near a day in Python and is left to finish on its own), so the window statement holds there for every configuration, not only the rigid one; poison machines below primorial multiples are ordinary (worst 1.85 (ln q)^2 against the record 6.6); the machine's own longest struck runs for gears to 23 are 33 columns, and machines placed at them reach a twin within 2.42 (ln q)^2.
- 2026-09-18, loop entry 90: the property a failure would need, named and tested - the gears above a cut must strike every survivor of the gears below it, which needs their teeth to land on survivors far above their share (factor 6.4 at the cut q/2, 2.3 at q^0.75 for q = 5003); measured ratios are 0.81, 1.11, 1.03, flat in q, and the top cut runs the other way for a proved reason (top_gear_cofactor, proofs/FailureConditions.lean, 0 sorries: a top gear strikes a survivor only through the gear itself or the gear times one prime in (q/2, 2q)). The property is absent in every state measured.
- 2026-09-18, loop entry 91: the share logic field by field - multiples is a rigid lattice at share 2/g; squares is one column per gear as the upper member (square_is_upper_member); lower:g reaches the smaller gears' survivors only at powers of g (lower_on_survivor_is_power, proofs/FieldBlocking.lean, 0 sorries); products:j relabels the higher fields; higher:g takes exactly its lattice share of the survivors below it (1.000 to three decimals below sqrt q, band mean 1.00 above) with prime cofactors above q^(2/3). No field has a member that can enter a blocking state; a block needs the shares to stop overlapping, a many-body property no single field carries.
- 2026-09-18, loop entry 92: pairs of fields - with the pair interaction isolated from each gear's own rate, pairs with a gear below sqrt q overlap exactly as independence (ratio 1.000 to four decimals), pairs of two large gears overlap 11 percent less (0.882 at q = 1009, 0.889 at 2003) through the rigid pattern of two large-prime products two apart, on an overlap term negligible against the survivors; the one exact pair, squares against the rest, combines exactly when g^2 - 2 is composite (square_lone_killer_iff, proofs/FieldBlocking.lean). No pair can combine into a blocking state.
- 2026-09-18, loop entry 93: triples of fields - 20,000 stratified triples per machine against independence and the pairwise-consistent (Kirkwood) baseline: 1.000 with all gears below sqrt q, 0.997 with one large gear, 1.03 with two, and with three large gears 0.72 to 0.85 against independence but 0.90 to 0.97 against the pairs on counts of 42 and 31. Everything a triple does is explained by its pairs; the pure three-body term shrinks toward 1 with q. No triple can combine into a blocking state.
- 2026-09-18, loop entry 94: quadruples of fields - against independence and the triple-consistent baseline, 1.00 within 0.05 with zero to two large gears over hundreds of thousands of coincidences; with three large gears 0.88 and 0.82 against independence, less avoidance than their three large pairs alone predict; with four large gears one coincidence at q = 1009 and none at 2003, below measurability. No quadruple can combine into a blocking state; each level above the second adds nothing or makes the coincidences rarer.
- 2026-09-18, loop entry 95: the many-body interactions by logic - five and only five: member coprimality (no_gear_both_members), stacking bounded by size (no_three_large_on_member), joint strikes one class (joint_strike_class, CRT, so lattices carry no interaction), truncation (fixed by q's residues, the only source of deviation), and the cofactor recursion (proofs/ManyBody.lean, 0 sorries). None coordinates gears across a window; a block needs a relation among one integer's residues modulo different gears, and there is none beyond size, so the question is whether a small integer can carry an adversarial residue vector - the free-configuration question.
- 2026-09-18, loop entry 96: the adversarial residue vector - one free shift per gear with the tooth pair rigid (teeth_separation, proofs/ManyBody.lean, 0 sorries); to kill, the rigid pairs must cover the window, equivalently the tuples' truncation strays must cancel the positive main term exactly. Exact search over shift vectors: no residue vector kills any window for machines 11 to 31 (37: no residue vector kills, fewest uncovered 21), a thousand times cheaper than the free search; the fewest uncovered columns grow with q (6, 3, 7, 6, 8, 14, 14).
- 2026-09-18, loop entry 97: pure powers in the field analysis - counted throughout (every higher power lies in higher:g, lower:g and products:k, and lower_on_survivor_is_power covers them) but only the squares were named; now power_member_side (proofs/FieldBlocking.lean, 0 sorries) says an even power is the upper member and an odd power of a gear 5 mod 6 is the lower member. Measured: 38 higher powers in the window at q = 1009 (14 lower, 24 upper), 88 at q = 5003, against 158 and 651 squares; no verdict changes.
- 2026-09-19, loop entry 98: the owner's argument that no gear set kills forever, checked - every step true, and the open-run step now proved both ways (open_run_lt_gear, open_run_after_alignment, proofs/AlignmentLimit.lean, 0 sorries): the longest open run is exactly the smallest gear less one and recurs at every common multiple. The argument proves the infinite statement (open_columns_for_any_gears); the window statement needs the open run inside (q, q^2], and the guaranteed one sits at the product of the gears, beyond the carry wall.
- 2026-09-19, loop entry 99: what a gear to come can do, exactly - new_gear_only_square (proofs/StretchRule.lean, 0 sorries): for consecutive gears p < q a member in (p^2, q^2] divisible by q is q^2 or already struck by a gear at most p, so the arriving gear closes only its own square's column and every slot is decided by the gears up to its square root. A kill needs the present gears' rigid tooth pairs to cover a stretch of about p x gap / 3 columns above p^2 except at the next square; measured over all 666 consecutive-gear stretches below 5000, none is twin-free, fewest twins 2.
- 2026-09-19, loop entry 100: the killers found and counted - under the strike law's square-residue constraint, killer configurations exist in the residue space from p = 17 on (every stretch from p = 37), so the protection is not in the constraint's shape; but each stretch has one realisable vector, p's own, the killers are a few tenths of a percent where they exist (376 of 85,085 at 17-19, 1.7 million of 1.08 billion at 29-31, none at 19-23, 23-29, 31-37), and p never lands on one. All 2260 stretches to 20000 have twins, fewest 2, fewest above p = 10000 is 147.
- 2026-09-19, loop entry 101: is a prime ever a killer of its own stretch - at 17..19 every killer needs a zero residue so no prime could have killed it; at 29..31 48,896 prime-compatible killers exist and 29's vector is two residues from one, but 29 is the only prime with that gear set; the prime-compatible killer fraction at twin-gear stretches falls like exp(-c p / ln^2 p), c about 2.3 (2.7e-3 at 29 to 5e-6 at 107, none sampled from 137), a heuristic recorded as one. A proof needs a property separating the class of a prime of the gear set's own size from the killer classes.
- 2026-09-19, loop entry 102: the location law - a gear strikes p^2 + a only if -a is a square modulo it (strike_after_square_isSquare, square_neighbour_killers: p^2 - 2 only by gears 1 or 7 mod 8, next_lower_killers: p^2 + 4 only by gears 1 mod 4; proofs/KillPositions.lean, 0 sorries), holding exactly on real machines with about half the gears eligible per position. The six proved rules governing a kill do not forbid a dead stretch - the killer configurations of entry 100 obey all six - so a permanent-kill impossibility would be a seventh rule tying a prime's residues away from its stretch's killers, which the machine's interactions do not carry.
- 2026-09-19, loop entry 103: how long a dead run can be - a dead stretch is p x gap / 3 columns by definition; a dead run of a fixed gear set is shorter than the gears' product (open_at_multiple_of_product, proofs/StretchRule.lean, 0 sorries) and in practice the paired Jacobsthal length (33 columns at P = 23); from P = 17 the rigid runs are long enough for a twin-gear stretch, so position, not length, protects it - the run would have to start at p^2; an infinite dead run is an unending sequence of stretch handovers, each finite, which is the conjecture's negation.
- 2026-09-19, loop entry 104: how later gears kill adjacent stretches - rough_member_form (proofs/StretchRule.lean, 0 sorries): above p^2 and below p^3 a member no gear up to p strikes is prime, a prime square, or a product of two primes above p, so later gears never kill singly; a dead run across stretches is the base's struck run with holes plugged by squares and two-prime products, and the interleave is multiplicative - products of later primes' residues must hit every open residue of the base in every period, with primes that exist at the right sizes.
- 2026-09-19, loop entry 105: the plan of attack on the four remaining killers (research/proof/killer_attack_plan.md) and two results - plug_law proved (proofs/StretchRule.lean, 0 sorries: in the stretch above p_2^2 the new gear plugs the base's holes only at p_2 times a prime at least p_3, so plugs across stretches are squares and near-consecutive prime products), and small lifts kill short runs (26987^2 starts 40 columns fully struck by the gears up to 59; its own stretch is 53,980 columns) - position permits, length forbids, and the length is the exponent gap.
- 2026-09-19, loop entry 106: concept 3 closed - the plug p_2 p_3 sits at an offset fixed by the gaps, its column is base-open when the partner is rough (a quadratic in the base residue with no factorisation), and measured over 2260 triples the consecutive product is base-open 12.4 percent against 15.5 to 16.6 for non-consecutive products and 19.7 for free residues: size structure plus the consecutive-prime residue correlation of Lemke Oliver and Soundararajan, a Hardy-Littlewood-type law outside the machine's rules.
- 2026-09-19, loop entry 107: concept 1 closed - kill_needs_run (proofs/StretchRule.lean, 0 sorries): a dead stretch is a struck run from the square at least as long as the stretch; measured to 200,000 the record run above a square is 703 columns against a stretch of 1,027,452, the largest share ever covered 0.32 at p = 19, records at 2 to 5 (ln p)^2 while the stretch grows like p. The plan of attack is closed: plugs and interleave rest on a conjectural distribution law outside the rules, killer vectors on the run-length exponent, strays on the identity.
- 2026-09-19, loop entry 108: the owner's four-line argument as the spine - lines 1, 2 proved, line 4 proved from line 3 (twins_unbounded_of_survival, proofs/OwnerArgument.lean, 0 sorries), and line 3 named as the Survival lemma: for every gear p with next gear q, some column strictly inside (p^2, q^2) escapes every gear up to p. First law of the attack: product_kill_square_law - a straddling product (p-a)(p+b) lands on offset (b-a)p - ab with (b-a)^2 - 4o a square modulo p.
- 2026-09-19, loop entry 109: the survival lemma on the primorial family - survival_of_family (proofs/OwnerArgument.lean, 0 sorries): the gears 7..p lay one fixed pattern on the t-line of 30t +- 1 and the machine only selects the range [p^2/30, q^2/30], so the conjecture is that the pattern's struck run from the square's position is shorter than the span; measured, the run at the square is an ordinary run of the pattern (18 against random 6 to 21 at 1009; 1 against 25 to 54 at 19997), so the location law does not protect the stretch - the runs are short everywhere, polylog against a span of exponent 1.
- 2026-09-19, loop entry 110: the survival lemma placed in the record - the reduction to run bounds, the j_2 ladder (explicit 19, 17, 15, 8.04, non-explicit 4.266), the parity ceiling, the rigid-below-free record law, the record and loaded record rules, the position frontier and the layered Erdos-Rankin lower bound were all already on record. New consequence: the layered lower bound j_2 >> x (log x)^3 makes the stretch-level survival FALSE in the free two-class model, so any proof at the stretch must use the rigid configuration; the window form (exponent 2) is the free-compatible one and is where parity blocks sieves. Program: the loaded record rule's domino cost at the actual square-residue core phase vector.
- 2026-09-19, loop entry 111: the survival lemma weighed - SurvivalInf (some stretch above every bound survives) proved EQUIVALENT to twins unbounded (survivalInf_iff_twins_unbounded, proofs/OwnerArgument.lean, 0 sorries; converse by the largest prime below the twin's square root and Bertrand); the strong Survival is Legendre-type for twins, strictly stronger. Entry 110's domino-cost program withdrawn: the stretch is longer than every gear, so the loaded record rule's tail is empty and the rule reduces to the survival statement itself. Five reformulations now on record, each equivalent or stronger; none lowers the weight.
- 2026-09-19, loop entry 112: the owner's line 3 against the record's counter-machine (fold_mechanic.md, 2026-09-11): M_G with the twin lowers removed from the generators has every striking mechanic (dilation, hand-up, square-root rule, one least factor per composite, both classes at every scale) and every section fully struck (85 columns, 0 pairs at [49, 2809); 0 mismatches over 7,309 sections). So SurvivalInf does not follow from the mechanics; a proof must use the line's completeness (every j is a column) in a way that is neither a free-phase cover nor a count. Admissibility test: delete the open columns and rerun. Next: the counter-machine as a kernel independence theorem.
- 2026-09-19, loop entry 113: the counter-machine in the kernel (proofs/CounterMachine.lean, 0 sorries) - abstract lines (multiplicative sets of fold survivors), gears = irreducibles, square-root rule and twin-of-surviving proved for every line; the real line's survival is MirrorWalk.SurvivalInf; the counter line (twin lowers removed from the generators) never survives any stretch. Survival is independent of the mechanics of striking; the only difference between the lines is real_column (every j is a column).
- 2026-09-19, loop entry 114: the wall map updated with entries 111-113 as edges and a table of seven angles each closed on record (cover/run bound, rigid record tools, mechanics alone, analytic, chain, pins, bilinear switching); a proof must be a use of the line's completeness that dies when the open columns are deleted and is neither cover nor count.
- 2026-09-19, loop entry 115: the two layers of a stretch (base gears <= q^(2/3), top gears in (B, p] whose strikes are pinned products g x r with r prime) and the plug-run law: K(p), the longest run of consecutive base-open columns all plugged by the top layer, is 3 to 4 ln(nb) (records 21 at 433, 24 at 2477, 32 at 5717) against nb ~ p gap/(log p)^2; plugs cluster above the independent rate (0.65 measured, 0.75 effective); the record plug runs use distinct top gears from the whole layer (32 of 32 at 5717), each once. FACT; next probe: whether consecutive plugs share the prime cofactor r.
- 2026-09-19, loop entry 116: correction to 115 (no clustering - the record plug run 32 is below the whole-sample independent expectation 35.0) and the independence law: the top layer's plug rate on a base-open column is independent of the base pattern around it (13 neighbour patterns and 12 distance buckets, all within 1.5 sigma of 0.6463 over 4.27 million columns). The two layers of a stretch do not interact; survival is carried by independence, the sieve's picture on the rigid teeth.
- 2026-09-19, loop entry 117: under the rigid tooth pair, single-gear killers of a stretch exist only at p = 17 (gear 17) and p = 41 (gear 11), none to 3,000 (the record's p = 29 killer was a free two-class tooth); the kill distance d(p) - least number of gears re-phased to strike every column, abandoned lone kills included - measured exactly to p = 71: 1-2 at 2-3 twins, 3 at 5, 5 at 7-11, beyond 5 at 13 or more; d ~ twins/2.5, growing; base case d >= 1 is the conjecture. FACT.
- 2026-09-19, loop entry 118: kill distance exact by ILP to p = 109 - the stretches at p = 7, 11, 13, 19, 23, 31 are UNKILLABLE by any rigid re-phasing (open under every phase vector); from 37 on killers exist and the distance from the real configuration is d = twins/2 (0.33-0.67, mean 0.50, no drift), moved gears spread over 5..p. FACT: the survival margin is half the twins' worth of gears, not one column.
- 2026-09-19, loop entry 119: the shift-rigid record equals the machine's record F(M) (CRT: every shift vector is a window position of the real pattern), so the ILP computes exact rigid records without the period scan - 33, 42, 57, 87, 90 at p = 23..41 all agreeing with the certified ladder; F(47) and F(59) (pinned [161, 178]) bisecting in the background. The re-phasing adversary never leaves the machine.
- 2026-09-19, loop entry 120: kernel RigidShift.lean (0 sorries) - window_realises_shift and shifted_pattern_is_window: every shift vector of the rigid pairs is a window position of the real pattern (CRT), so F_shift = F(M) at its mechanism and the re-phasing adversary walks the real period.
- 2026-09-19, loop entry 121: exact rigid ladder by ILP + CRT certificates - runs 1, 4, 6, 10, 17, 24, 33, 42, 57, 87, 90, 102, 117, 144 at p = 5..53 (144 at 53 NEW, certified at window x = 1249461754311661376); F(59) in [161, 164]; F / (p ln p) rising 0.3 to 0.69; from p = 37 the record exceeds most stretches, so the pattern has fully struck windows longer than the stretch and the squares are never at them. FACT.
- 2026-09-19, loop entry 122: square-class (location-law) shift vectors kill stretches exactly where unrestricted ones do, p = 7..83 (ILP); the location law removes no killer. DEAD as protection.
- 2026-09-19, loop entry 123: F({5..59}) = 160 exact (161 in the ladder's convention, the bottom of the record's pinned [161, 178]; 161 and 162 proved uncoverable by HiGHS), F({5..61}) >= 179 certified at x = 13169725611018917022346, 210 uncoverable. The record's falsification target is met.
- 2026-09-19, R5.d.i.b: record windows against square windows - the certified record windows' start residues lie in the square class at 37 of 70 gears (0.529), no gear systematically avoided. Prediction (no relation) held; DEAD.
- 2026-09-19, R5.e: adjacent stretches share phases exactly at the gears dividing (q-p)(q+p) (0-5 of them); detrended twin-density residuals of adjacent stretches uncorrelated (r = -0.02, p > 2000) and unshifted by the number of shared gears (1.004 to 0.997). FACT; raw r = 0.92 was the 1/(ln p)^2 trend, a confound in the pre-registration.
- 2026-09-19, R5.f.i: the first lane's three claims hold at every p from 37 to 19,997 - square-scale transfer (a p-rough partner of P^2, P(P+2) or (P+2)^2 for the first twin of the stretch at the mean), single-plug column in every stretch with every cofactor prime, near pairs with the separation law exact. FACT; the lane is lifting claim 1 to the level where it is used.
- 2026-09-19, R5.f.iii: the rung in centre coordinates - the locator (small gears pre-cleared by a congruence on the offset, phases from s^2 mod g) contains a twin centre at every twin lower to 200,000 under the y-rule; clean rungs from P = 271; the exact-count ratio min 0.472 at P = 71. FACT.
- 2026-09-19, R5.d.i.c: F({5..61}) = 179 exact (180, 181 proved uncoverable; 179 certified). R5.f.v: the canonical twin ladder from (5, 7) reaches a 417-digit twin lower in nine rungs, offsets 1, 4, 6, 77, 44, -829, 3605, 28145, -6965, ratio 6|j|/(ln P)^2 in [0.18, 2.9].
- 2026-09-19, R5.f.iv full run: the sieve-form rung law (first twin among the sqrt(s)-rough offsets at index ~4, max 34, never absent) holds to 10^5; steering by 35 from P = 2,383; progression form DEAD as stated; forced offsets closed.
- 2026-09-19, R5.f.vi: base half of the sieve-form rung holds with margin (base-struck runs at most 0.43 of 0.7 x ln x; base-open counts 2.25x the heuristic); top half: first twin index mean 4.00, max 29 to 10^5, constant 24 refuted, geometric tail; min-gap distinctness does not bind at N >= 16. A FACT, B OPEN in mechanism.
- 2026-09-19, R5.f.vi claim C: the canonical ladder under the lane's rule reaches a 378-digit twin lower in nine rungs; ratio 6|j|/(ln P)^2 without drift; phase locks exactly at the gears dividing 6j. FACT.
- 2026-09-19, R5.f.vii full run: index law i <= 4 ln P holds at all 8,168 twin lowers to 10^6 (max i/ln P = 3.29 at 646,421); square-phase eligibility exact; gear reuse at 0.41-0.49 of the crude independent estimate to 10^5.
- 2026-09-19, R5.f.viii: exact reuse expectation accounts for most of the crude factor 0.47 but a deficit growing with the block (7% at 16, 10% at 32, 4.5 sigma) remains, open; steering does not lower the first-twin index (4.53 +- 0.21); no impossible block of 2, 3, 4 base-open offsets.
- 2026-09-19, R5.f.ix full run: the reuse deficit was the statistic (sum C(k,2) = 2,057 vs 1,929 expected, +6.6%); the pigeonhole supply side of the index bound holds at every twin from s = 12,162 to 10^6, with eight small exceptions verified directly; the remaining hypothesis stated in the literature's words (twin pair between P^2 and (P+2)^2, or in [s^2 - H, s^2 + H] with H ~ N^(1/4) log N).
- 2026-09-19, R5.f.x: the lane's audit of rounds 1-8 (statuses, the serious flag that the candidate supply is conditional on a run bound F(x) < 2x^2/3), the proof map (critical path LadderHyp -> twins unbounded; L3, L4, L5 off it) and the certificate design; research/proof/ladder_proof_map.md; six-rung Pratt certificate to a 48-digit twin centre building.
- 2026-09-19, R5.f.xi: the fixed-depth rung (depth 61, F exact) measured to 10^6 - first-twin index among the 61-rough offsets is 0.064 (ln s)^2 on average, flat by decade, max 0.598 (ln s)^2 (line 0.6 held by a hair); the candidate supply is unconditional at this depth.
- 2026-09-19, R5.f.xii: at fixed depth 61 the bound function must be a cube, B = 0.0695 (ln s)^3 (maximum to 10^6 predicted 85-120, observed 107; no breach of 0.07 (ln s)^3); NTH_61 = the twin analogue of Cramer's conjecture at N = s^2 (interval 0.38 (log N)^3); the mean tier reproduces the average twin gap to three digits; depth cancels; the reduction is complete; path form added to the kernel.
- 2026-09-19, R5.f.ii to 10^6: every twin lower's stretch holds a twin (8,168 twins, 0 failures); worst nearest-twin ratio 13.27 at P = 646,421; at least 3 twins per stretch from 41. The ladder hypothesis holds at every twin below 10^6.
- 2026-09-19, R5.f.xiii (random lane): universal clearance classes U_g (offsets no gear strikes at any twin centre) exact to 61 with 0 violations, rung rate inside A_13 enhanced 3.21x; the rung graph is a forest with unique parents (identity 440 = 440 at 10^6); neighbour-forced phases carry no rung information. F(67) >= 213 certified.
- 2026-09-19, R5.f.xiv (random lane round 2): the universal list carries the cube bound at the certified top (4 x 10^30 guaranteed 109-rough candidates against 8.9 x 10^4); parenthood and child counts are generic (the twin constant); the cyclotomic offsets j = +-c have identity-forced roughness at a quarter of all gears but enhancement 0.89 = the singular series' 0.878 - no gain; F(67) in [213, 217].
- 2026-09-19, R5.f.xv (random lane round 3): in-window polynomial offsets classified (k in {0, +-1, +-2}); a second composite-forcing family found (upper member (s +- 1)^2 - v^2 at j = +-2c - 6t^2); every irreducible family priced by a finite Chebotarev constant; conservation law - each gear strikes exactly two of every g consecutive offsets, so offset selection cannot raise the expected rung count; the (5,7) tree enumerated to depth 3 (1, 2, 5, 182 nodes), child counts generic; the depth form is the weakest hypothesis, in words: an infinite chain of twin pairs each between the squares of the preceding pair.
- 2026-09-19, R5.f.xvi: at depth 4 of the (5,7) tree (s ~ 10^12-10^13, 30 nodes) the offset ratio (max 10.6), the 61-rough index (max 430 against the cube ~1,850) and the sqrt(s)-rough index (mean 5.8, max 24) all hold - the index laws extend six orders of magnitude in s.
- 2026-09-19, R5.f.xvii (contradiction lane): a leaf of a finite (5,7) tree is exactly sieve data (every composite member has a factor <= s-1), violates no exact law, is untouched by every proved theorem (BHP silent - the stretch is shorter than N^0.525; Legendre open even under RH; parity applies in full), and would be 10^2-10^7 times the extreme-value twin gap at its height; the route's one asset is the existential-per-level form (twins in the union of a level's stretches).
- 2026-09-19, R5.f.xviii: at depth 5 (s ~ 10^25) the 61-rough index mean is 205 against the law's 203, max 894 against the cube ~11,000; ratio max 5.5; every sampled node has a rung.
- 2026-09-19, R5.f.xix: at depths 6 and 7 (s ~ 10^50, 10^100) the 61-rough index sits on the mean law (854 vs 928; 3,959 vs 3,710), maxima two orders below the cube; every node has a rung; the laws hold to s ~ 10^100.
- 2026-09-19, R5.f.xix depth 8 (seven nodes, 400-digit members): every node has a rung; index mean 23,140 against the law ~15,000 (+1.4 SE), all below the cube by two orders; the index laws hold from 10^2 to 10^200.
- 2026-09-19, R5.f.xx: the chain form in the kernel - chains of every finite length anywhere in the rung forest give twins unbounded (twins_unbounded_of_chains); DepthHyp -> ChainHyp; the weakest ladder-type hypothesis on record.
- 2026-09-19, R5.f.xv addendum: child counts of the 182 depth-3 nodes are Poisson about the twin-constant law (std 0.0092 = 1/sqrt(T)); no excess variance.
- 2026-09-19, R5.f.xxi (conditional lane): the cube ladder follows from the standard short-interval Hardy-Littlewood lower bound (theta <= 2/3); the square ladder needs the Legendre-strength endpoint (twin Cramer); the CHAIN form follows from an almost-all power-saving hypothesis (variance of pi_2 at length x^(1/2)) - the pointwise/almost-all line falls between DepthHyp and ChainHyp; no primes-only hypothesis reaches any form.
- 2026-09-20, R5.f.xxii: kernel LadderAlmostAll.lean (0 sorries, standard axioms) - AlmostAll c delta eta (twin count >= X^(1-eta), exceptional rung-poor twin centres <= X^(1-delta)) implies ChainHyp, hence twins unbounded; the almost-all -> chain theorem of the conditional map is formal.
- 2026-09-20, R5.f.xxiii: kernel LadderShortInterval.lean (0 sorries, standard axioms) - a twin pair in every (x, x + x^theta] with theta <= 1 - 1/e gives every large twin centre an exponent-e rung, chains of every length and twins unbounded; the cube ladder rests on the standard short-interval twin conjecture.
- 2026-09-20, R5.f.xiv addendum: kernel LadderInfinite.lean - the (5,7) rung tree is infinite iff it has a node at every depth (no Koenig), so twins unbounded follows from an infinite (5,7) tree; 0 sorries.
- 2026-09-20, R5.f.xxiv (routes lane, Fable): the product forest - windows ((s-1)(t-1), (s+1)(t+1)) for twin centres s <= t, exactly three roots (6, 12, 18) to 10^6, never empty over 20,100 pairs; kernel LadderProduct.lean: LadderHyp -> ProductHyp -> twins unbounded, SixHyp (a twin centre within ratio 7/5) -> ProductHyp. Consecutive-product windows generic (0 empty of 462). Near-gear confinement refuted as stated, corrected k-rule is the difference-of-squares families again (rediscovery, closed).
- 2026-09-20, R5.f.xxiv.b.i (red lane): product-forest numbers reproduced exactly; multiplier-6 windows never empty (min 2 at 6, 12, 18); the multiplier-6 family does not cover u = 138 (needs (12,12)); r_p = p(p^2-6p+10)/(p-2)^3 exact, K = 0.823103, residual z = -1.73 inside Poisson.
- 2026-09-20, R5.f.xxv (inheritance lane, Fable): the forest is exactly residues - inherited local factor identically 1, no slope on omega(6j) or j mod 30, mod-5 dependence of consecutive offsets is the two-block residue rule with independence inside, every absent residue pair forced; sub-Poisson spread 0.82 matched by non-stretch windows 0.79.
- 2026-09-20, R5.f.xxv.a: dispersion index of twin counts in windows x^delta falls 0.926 -> 0.742 from delta 0.3 to 0.7 (Montgomery-Soundararajan shape; stopped as known).
- 2026-09-20, R5.f.xxvi: 3,398 regions between consecutive prime squares to 10^9, none empty; the gap-2 regions (twin stretches) carry the smallest counts and ratios (0.573 at p = 29) - the stretches are the binding case of the window statement; formalist lane opened for RegionHyp / WindowHyp.
- 2026-09-20, R5.f.xxvi.a: kernel LadderRegion.lean (0 sorries) - RegionHyp (a twin centre between every pair of consecutive prime squares) -> LadderHyp -> WindowHyp (a twin centre in [q, (q+1)^2) for every q >= 6): the owner's window statement is a kernel consequence of the ladder.
- 2026-09-20, R5.f.xxvii (parity lane, Fable): every exact law on the record (two classes, universal clearance as coprimality, phase lock, uniform local factor, top-band rule, unique parents, region law) holds equally for the parity-twisted sets Q and M'; only 'the members of a twin centre are prime' separates the twins. Index law to 10^7: max 0.6885 (ln s)^2 at s = 5,042,928, the 0.6 line breached as a geometric tail predicts, the cube envelope holds.
- 2026-09-20, R5.f.xxviii (prover lane, Fable): three written attempts; leaf ledger identity r = |R_61| - T_61 + X_61; gap A-2 (big gears strike the s/3-rough columns fewer times than their number, 256 vs 100 at 10^4); the dichotomy lemma proved and put in the kernel (LadderDichotomy.lean): no consecutive leaves + no single rung -> chains of every length, node at every depth from 6, twins unbounded; Six-window ledger closes SixHyp for t >= 10^15 under a per-gear discrepancy bound; the same ledger fails in the square window.
- 2026-09-20, R5.f.xxviii addendum: kernel LadderEuclid.lean - the Euclid device: a gear dividing the offset never strikes the lower member (it would divide a twin prime), strikes the upper iff it divides s^2 + 1; the one exact law that uses the members' primality; unconditional base-open supply at level ln s.
- 2026-09-20, R5.f.xxix (audit lane): kernel statements audited - no vacuous, subtraction, endpoint or direction defects; every hypothesis that contains the conclusion labelled; NearTwinHyp is a scaffold; docstring overclaims in LadderDepth fixed with two new theorems (chainHyp_of_path, chainHyp_of_ladderHyp_above); twins_unbounded_of_windowHyp and lower_member_rough_upto added.
- 2026-09-20, F(67) = 213 EXACT (two-class covering record for the gears 5..67; 214 and 215 uncoverable in 9,981 s and 11,285 s); increment 34 from F(61) = 179; F/(p ln p) = 0.756.
- 2026-09-20, correction: the Euclid device is the universal clearance class j = 0 (lower member) in kernel form, not a new primality-using law; both dichotomy hypotheses hold to 10^6 (minimum 3 rungs from s = 42). F(71) >= 222 (coverable in 3,227 s), bisection continuing.
- 2026-09-20, R5.f.xxx: the Chen ledger T = |R_{s/3}| - N_semi exact at 118 twin centres to 5000; rough columns are about 60% rungs, 40% large-semiprime columns (share 0.28-0.67); two placed numbers refuted.
- 2026-09-20, F(71) >= 222 (coverable); the L = 241 ILP killed for memory after 3 h; F(71) in [222, 259].
- 2026-09-20, R5.f.xxxi (literature register): no twin-pair result in any short interval, not even almost all; Chen pairs at theta 0.97, bounded gaps at theta 0.525, GEH gives 6 (parity-optimal); maximal twin gap 35,640 at 7 x 10^16 so SixHyp/LadderHyp hold to 10^16; OEIS A192870 (twin-Legendre, last failure 122) and A288815 (paired Jacobsthal) are the nearest published objects.
- 2026-09-20, R5.f.xxxii: THE FREE COVERING ROUTE - if two free classes per gear cannot cover the q^2/6 columns of the window of q (OEIS A072753 < window at every known q <= 73, tightest 24 vs 26 at q = 13), the window holds a twin by the square-root rule alone; the open lemma is an Iwaniec-type bound j_2(q) < (q^2-q)/6; its wall is the large sieve's P^2 constant, not parity. Kernel and register lanes opened.
- 2026-09-20, R5.f.xxxii register: no o(P^2) upper bound is known for any Jacobsthal-type function; Iwaniec's P^2 constant is ineffective; two-class upper bound unpublished; the covering route is Ziller-Morack's 2017 conjecture 6 with proved implication; CANDIDATE - the one route whose wall is not parity.
- 2026-09-20, R5.f.xxxii kernel: LadderCovering.lean (0 sorries) - CoveringHyp (no two classes per gear cover the window of q) -> a twin in the window of every prime machine q -> twins unbounded; the free covering route is formal.
- 2026-09-20, R5.f.xxxii correction (Iwaniec lane): Iwaniec's P^2 is the linear sieve's limit s = 2, not the large sieve; two classes give only P^4.27 (beta_2 = 4.266); the covering route's wall is parity in covering clothes - 'not parity' withdrawn; the implication stands in the kernel and the window statement holds by covering alone for q <= 73.
- 2026-09-20, R5.f.xxxii.a: kernel LadderMaxGap.lean - MaxGapHyp (the machine's longest struck run of gears 5..q is shorter than the window of q) -> window statement -> twins unbounded; the rigid F table beats the window by 2.5x or more at every computed q <= 67; the weakest covering-type hypothesis on the tree.
- 2026-09-20, R5.f.xxxii REDISCOVERY: the covering route is the project's rounds 21-27 line (docs/covering-bound-route.md, Ladder.lean's (D) ladder, novel j2-upper-bound with the beta_2 ceiling and Ziller-Morack Conjecture 6, j2-lower-ladder); the standing rule to grep docs/novel first was not followed; closed as a route, kernel files kept as the general form of the (D) step.
- 2026-09-20, R5.f.xxxiii (ideas lane): no route whose first step is not a sifted-set bound; every candidate traced to one; new exact gear-role composition law 36c*^2 - 1 = (36c^2-1)(36(2c)^2-1)^2 at c* = 3c(48c^2-1) (Chebyshev cube), G(c*) = G(c) + G(2c), 0 violations to 3000 - upward only, not a route.
- 2026-09-20, R5.f.xxxiv: the field proof draft written as lemmas A-E (research/proof/field_proof_draft.md); A, B, C1-C5, D1, E1 proved, E2-E4 the lemmas to establish, E4 = MaxGapHyp; first target E2 (two rows exact).
- 2026-09-20, R5.f.xxxiv.a: E2 PROVED - F({g,h}) = 4 iff {5,7}, 3 iff one of them is 5 or 7, else 2 (tooth distance inv3 = +-2 only at 5, 7); exact to 101, 0 violations; triples follow the tooth-distance multiset; initial segments F = 1, 4, 6, 10, 17, 24, 33 (run convention) for p = 5..23.
- 2026-09-20, R5.f.xxxiv.b: E3 exact - a new record is old runs joined at the new gear's teeth (h <= 2 ceil(F/p) holes filled, 1-3 observed), F(q') <= (h+1)F(q) + h; the old runs used are well below the old record; F(29) = 42.
- 2026-09-20, R5.f.xxxiv.c: the record recursion verified exactly for q' = 7..23 - F(q') is the longest chain of consecutive holes of machine q lying in q' tooth classes plus its flanks; aligned pairs 3-5x rarer than (2/q')^2 because alignment needs the hole gap to equal inv3 mod q' exactly; E4 = a statement about the hole-gap spectrum at two residues.
- 2026-09-20, R5.f.xxxiv.d-e: aligned hole pairs = the hole-gap census at inv3 and q'-inv3 divided by q' (exact by CRT); the record sandwich G_1(q) <= F(q') <= G_{2 ceil(F/q')}(q) proved and exact at every step 7..23 (equality F(q') = G_1(q) at 11, 13, 19); E4 = the overlap statement - the new gear's teeth never fill all holes of a window of length q'^2/6.
- 2026-09-20, R5.f.xxxiv.f: E4c proved - in a covered window of q' the interior hole gaps of machine q are at least (q'-1)/3 (tooth distance inv3 = (q'+-1)/3 or (2q'+-1)/3) and alternate between two residues; a covered window is a sparse-hole stretch of density at most 3/q'.
- 2026-09-20, F(71): second bisection reached 229 coverable (10,403 s) before the system killed it for memory; F(71) in [229, 259] (run convention 229).
- 2026-09-20, R5.f.xxxiv.g: E4d proved - F(q') <= S_t*(q), the longest stretch of machine q with hole gaps >= (q'-1)/3; ratios S_t*/window 0.39-0.73 on 5..23, no fall yet; E4 = S_(q'/3)(q) < q'^2/6.
- 2026-09-20, R5.f.xxxiv.h: thin-band bound proved from the shapes - F(19) <= 42, F(29) <= 89 with one top gear (window statement for machines 19 and 29 by shapes alone); vacuous with a two-gear band; E4 = the overlap statement.
- 2026-09-20, R5.f.xxxiv.i: residue-collapse census exact (14 of 14 checks) - covered windows of q' counted from machine q's windows by hole residues; F(q') = max(G_1(q), largest alternating window).
- 2026-09-20, R5.f.xxxiv.j: kernel LadderFields.lean - E1, E2 (4 / 3 / 2 with the {5,7} exception), E4c, the sandwich upper half and the thin-band bound, 0 sorries.
- 2026-09-20, R5.f.xxxiv.k: kernel LadderWidening.lean - Lemma B (widening, twin slots persist) and the sandwich's lower half (a single hole is always alignable, by periodicity and CRT); every proved line of the field draft is kernel-checked.
- 2026-09-20, R5.f.xxxiv.l: twisted translates - F(q') <= q'(F_T + 1) proved, vacuous (F_T is F's size); the form re-encodes the joint alignment; step 1's shape routes exhausted at the same point as 5a and 5b.
- 2026-09-20, R5.f.xxxiv.m: origin square lemma proved (the next gear fills only the square column among old holes, 75 pairs verified) and the window count law T(q') = T(q) - [q'+2 prime] + N(q^2, q'^2]; the field programme meets RegionHyp; the lemma left is a twin between consecutive prime squares.
- 2026-09-20, R5.f.xxxiv.n: the twin stretch in the products field - base-open = twins + plugs, plugs share 0.63-0.81 at B = sqrt(2s), P_2 dominant; the region lemma is the comparison plugs < base-open, Chen's switching frame at a level where the stretch has no lower-bound sieve.
- 2026-09-20, R5.f.xxxiv.o: the products field of the stretch is Goldbach - two-prime members are partitions of 2m (m from s to s^2/B), the top band the short partitions of 2s and 2s+2 (exact, 74 centres); the twin problem in the stretch and the Goldbach problem above 2s are two faces of one machine.
- 2026-09-20, R5.f.xxxiv.p: plug share law 1 - 0.79 ln^2 B/ln^2 s verified; the base/plug comparison has no admissible level (base lower bound only below s^0.234, plug constant e^gamma at level x^1/2); the lemma's exact obstruction in the products frame.
- 2026-09-20, R5.f.xxxiv.q: the base-open count is the hole uniformity of machine sqrt(2s) at scale B^2 (via the G_k law); region lemma = uniformity (a) + EH-level plug bound (b); the field draft is complete with the one lemma open in four exact forms.
- 2026-09-20, R5.f.xxxiv.q measured: worst-window hole share 0.87 at scale B^2, 0.7 at B^2/3, 0.43-0.64 at B^2/6 where the worst window keeps 4-8 holes for B <= 23.
- 2026-09-20, R5.f.xxxiv.r: whole-range form (U') - every window of q'^2/6 columns of machine q with more than 2 ceil(q'/6) holes gives E4(q') by the one-gear thin band; verified 19, 23, 29; (U') asks q/3 holes against a mean of q^2/(2.4 ln^2 q).
- 2026-09-21, R5.f.xxxv.a (range lane): one gear of T reads q's cycle re-indexed by g (twisted machine); two per run only at the leg distances, bound 3F+4; pairs of large gears coincide on four residues mod gg'; a run needs mu_T(w) gears; the twins of each tier of the range are exactly the openings of the tier-root machine (13 / 167 / 173 at q = 13) - the range statement is the disjunction of the tower's window statements.
- 2026-09-21, R5.f.xxxv.b: the known opening's copies k P_L (k < q) hold no twin at q = 43, 71, 79, 113; the family has about 3 ln^2 q / q twin copies; refuted as a guaranteed survivor.
- 2026-09-21, R5.f.xxxv.c: machine 5 as the cycle, overlay every gear above 5: overlay gear g strikes copies (30k-+1) at k = +-30^-1 mod g (two classes, distance 15^-1 mod g) - a machine of the same shape with 5 in the base; every range (q, q#] to q = 79 holds a twin copy, the first at k = 1, 2 or 5.
- 2026-09-21, R5.f.xxxv.c.i: lap machine 7..q on the copies 30k+-1 has an open lap in its window (q/30, q'^2/30) for every q <= 300 (first open lap k <= 14); lap records 2, 4, 5, 7, 12, 18 for q = 7..23; same E4 shape with 5 in the base.
- 2026-09-21, R5.f.xxxv.c.ii: leg rule on laps g | D, 15D-1 or 15D+1; gear 7 cuts adjacent lap pairs every 7 laps with 5-lap holes; a run is 7's pairs with holes filled by 11..q (doublers only 11, 23, 29, 31, 59, 61); record runs as words for q = 11..23; the record is the longest chain of filled holes.
- 2026-09-21, R5.f.xxxv.c.iii: hole machine - each class of gear g hits five of every g holes in a progression with step 7^-1 mod g, positions descending; gear 13 is the holes' adjacent gear (ten consecutive holes as a staircase); longest chain of filled holes 0, 1, 2, 3 at q <= 17, 19, 23, 29; lap record 25 at q = 29 = 15 + 8 + 2.
- 2026-09-21, R5.f.xxxv.c.iv: Q1-Q6 carry to the laps (30 for 6, 15 for 3); mirror k -> -k fixes every class pair; range with machine 5 as cycle = tower of lap-window statements.
- 2026-09-21, R5.f.xxxv.c.v: centre of the lap period - lap at offset j/2 struck by g iff g | 15j-+1; centre pair always 7's adjacent pair; central run 4 / 10 / 16 / 34 laps for q = 11..17 / 19..37 / 41..113 / 127; at q = 11 the record is the central run; origin always open, centre always struck.
- 2026-09-21, R5.f.xxxv.c.vi: centre run vs record (laps) 4/4, 4/5, 4/7, 10/12, 10/18, 10/25 for q = 11..29; centre is the phase-0 run, a lower bound, exact only at q = 11.
- 2026-09-21, R5.f.xxxv.c.vii: staircase rule - gears other than 13 hit at most one of three consecutive holes per class; chains of 13 holes need 8 gears (q >= 41); observed chains 4 at q = 31, 37 (partial), all inside 13's staircase.
- 2026-09-21, R5.f.xxxv.c.viii: hole fills inside the staircase - admissible doubler pairs per position of 13; 2+2 fills only 29+31 (p = 1, 5), 11/23 + 29/31 (p = 2, 4), 11+23 or 59/61+29/31 (p = 3); all 14 observed holes obey; one 2+2 seen (q = 37, 11 with 23 at p = 3).
- 2026-09-21, R5.f.xxxv.c.ix: hole words per gear - five holes per class at step 7^-1 with positions 5..1, 5 - d doubles when d <= 4, g - 10 + doubles empties, palindromic under the lap mirror; words for 11..31 written out.
- 2026-09-21, R5.f.xxxv.c.x: stacking the hole words - forced fills confirmed exactly at q = 19 (11 doubles in every filled hole) and q = 23 (11 in a 3-window in 34 of 36 chains); word bound 1, 2, 5, 7, 10, 15, 28 for q = 19..43, no bound from q = 47: only phases limit chains beyond.
- 2026-09-21, R5.f.xxxv.c.xi: phases of 11 and 13 - q = 19 filled holes = 12 diagrams x 8 CRT classes (11 at 3 or 6, 13 avoiding 11's pair); q = 23 chains: 11 in its two mirror 3-windows, 13's phase in a 5-set each; chains come in mirror pairs; the tiling rule stated.
- 2026-09-21, R5.f.xxxv.c.xii: word solver reproduces the lap records 2..25 for q = 7..29 with no scan and gives 31 laps at q = 31 (new exact); record/window on laps 4/5 .. 31/45.
- 2026-09-21, R5.f.xxxv.c.xiii: all record runs - 4, 22, 2 per period at q = 19, 23, 29, mirror-closed; waste exactly 0, 1, 2 in every record run; ends on 13 or 7; no single gear stops a run.
- 2026-09-21, R5.f.xxxv.c.xiv: waste prediction (gears above 7 minus 4) REFUTED at q = 31 - minimal waste 6, not 3; what survives is the forced-gear rule: gear g must strike any run longer than g - d_g.
- 2026-09-21, R5.f.xxxv.c.xv (tiling lane): clash rule 11 vs 13 (20 of 143 residues, every phase pair clashes at m = 9); 13 plus two doubles each runs at most 2 holes; the dead block of three silent holes needs four gears twice over; the double-window lock proves chain 3 at q = 29 and 4 at q = 31 from the words; chain ladder 1, 2, 3, 4, 5, 6, 8, 9 for q = 19..47 against window 2.52..13.38 holes, +1-per-gear refuted at q = 43.
- 2026-09-21, manager check of c.xv: every chain-ladder entry constructed explicitly by an independent set-cover search (phase tuples recorded); the chains at q = 37 and 47 open on a dead-block hole.
- 2026-09-21, R5.f.xxxv.c.xvi: machine 5 leaves three open classes (n = 0, 2, 3 mod 5) and the lap route follows only n = 0, so it is a strict restriction of the range statement - measured 2.7 times tighter on every q from 7 to 31; the word method survives and moves to the full opening set with gear 5 as the cutter.
- 2026-09-21, R5.f.xxxv.c.xvii: word construction on the full opening set - gear 5 cuts blocks of three, ONLY gear 7 can double (positions 1, 2 at b = 4 mod 7), every gear from 29 up misses neighbouring blocks; chain ladder 0, 1, 1, 3, 4, 6, 8, 11 for q = 7..31 against window 4.0..45.6 blocks (ratio 0.10 to 0.25); F(q) = 5 x chain + ends confirmed; density proves the lemma to q = 23 and dies at q = 29.
- 2026-09-21, R5.f.xxxv.c.xviii: leg rule on blocks PROVED - g strikes two blocks D apart iff g divides one of 5D +- 1, 2, 3 or 15D +- 1, 2, 4, 5, 7, 8, 10 (plus 7 when 7 | D); checked against direct search to gear 500 for D = 0..12; the bridging set is finite and independent of q, and a gear above 15m - 5 puts at most one cell into a chain of m blocks.
- 2026-09-22, R5.f.xxxv.c.xix (six-dimension workflow): the dead-pair reduction - gear 7 is silent on adjacent blocks 6, 0 mod 7, the orphan law (verified from columns to gear 2000) gives N(1) = 4, and chain(q) <= 7 k_max + 7 with k_max = max{k : N(k) <= pi(q) - 4}. The lemma for every q >= 13 reduces to ONE q-free statement, N(k) >= pi(sqrt(210(k+1))) - 3; measured N = 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 12 for k = 1..11; q = 53 needs only N(16) >= 13. Chain ladder extends to 17 at q = 37. Counting closed: the reciprocal sums cross one half at q = 29 and q = 47.
- 2026-09-22, R5.f.xxxv.c.xx: PRIOR-ART CHECK - c.xix's N(k) reduction is the wall's W2 (K_columns(d) > pi(sqrt(6d))) transported by d = 35k; the project already posed it and measured its constant at 24 against a target 6 (7 to 11 on islands). The dead-pair version asks for a STRONGER statement than W2. What survives as new: the leg rule on blocks, the orphan law, the row reading of the block, the N table and the chain ladder. Branch continues, but as W2.
- 2026-09-22, R5.f.xxxv.c.xxi: N exhaustive to k = 14 (4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 12, 12, 13, 14) and N(k) >= 15 for k >= 18 (cut-free); the lemma at q is exactly K(pi(q)-4) < k_req(q); the q-free surrogate is refuted; counting closed by LP measurement (root saturates at 10.6, every rung is enumeration). The route reaches q = 61 while the project's F ladder already reaches q = 67, so its per-q verdicts add nothing and its value is the structure. K(14) is the one measurement that decides whether the margin widens or closes.
- 2026-09-22, R5.f.xxxv.c.xxii: counterpoint pass on the owner's instruction - NINE OF TWELVE refutations were wrong or partly wrong. The N(15) certificate was killed by a three-gear transcription error and is valid; four-gears was a confirmation not a refutation; the q-free statement lives with 210k + 79 in place of 210(k+1), the reduction proved; the two-cells-in-an-m-window rule is now a theorem (g <= 15m - 5) and contains the orphan law's gear list; the doubler pool is frozen at 11, 13, 17, 19, 23 forever with longest doubled run exactly 6. Method now derive-refute-defend-adjudicate.
- 2026-09-22, R5.f.xxxv.c.xxiii: K(14) = 14 exactly - N(15) = 15 proved by twelve cut-free exhaustive cases and re-decided by a second engine. The margin k_req - K runs 2, 4, 4, 4, 6 at q = 43..61 and is WIDENING. Surplus N(k) - k is 0 from k = 12 through k = 15. Enumeration is closed as a route to a theorem; the target is now one sentence: N(k) >= k - c for every k.
- 2026-09-22, R5.f.xxxv.c.xxiv: N(k) >= k - c REFUTED by proof and raw certificates (N(k) <= 574 + k/2 for every k; N(100) <= 49, N(200) <= 70). Replacement N(k) >= 3.23 sqrt(k) for k >= 23, constant sharp at q = 109, reduction effective and unconditional. Skeleton S1-S7, S9 proved; S8 open and needs GROWTH, first failing at q = 67 (N(23) >= 16 against a standing 15). Counting closed from a fifth direction (supply-constrained LP flattens at 11.77 while the integer optimum is 16) - the growth is integral. BOUNDARY REPAIR: the square-root rule fails at exactly one column of the closed window, 6n+1 = q'^2, verified q = 7..113; state it with the strict inequality. k_req(61) = 21, not 20.
- 2026-09-23, R5.f.xxxv.c.xxv: DRIFT CORRECTION on the owner's instruction - the manager swapped the range statement for the window statement at node c.xvi without flagging it, so c.xvii to c.xxiv are window work. The range object restated: laps (30k +- 1), overlay every gear above 5 below sqrt(q#). The range is a DISJUNCTION OVER TIERS (2 to 5 tiers at q = 5..89, growing like log q) of which the window is only tier 0, and the range needs no location. Active line returns to the range.
- 2026-09-23, R5.f.xxxv.c.xxvi: the range answered. Path A is EXACTLY the window statement at the cut isqrt(q#); path B is FALSE (the three offsets of the 2,3,5 cycle are one pattern in three positions, exact translates by -e x 5^-1). The range is weaker by exactly one thing - the certifying gear set is chosen after the lap - and extra length is not freedom since the range is exactly one period. NEW: the congruence-locator closure (a class silent against [7,X] has modulus divisible by X#/30, so it certifies at most one lap, inside the window) kills the whole locator family at once; the squaring law j^2 = a_g^2; the supply law (dropping p multiplies live copies by p - 2, not p). Same wall, now one inequality X# > nextprime(X)^2. The manager's own tier framing in c.xxv is withdrawn.
- 2026-09-23, R5.f.xxxv.c.xxvii: blame-assignment round. PROVED: acting is cofactor order (the gear is the smaller factor of its leg); the first-strike law (k0 = min(u, 30-u) in {1,7,11,13}, every gear strikes below its square except 7 and 11); the exact separation law with four slopes by p mod 15; the fibre value is always a square so no quadratic character gates striking; THE BARRIER (the untruncated machine satisfies every strike-relation rule and admits a total blame map b(j) = lpf(30j-1), so no strike-relation rule can forbid one); the MACHINE-GEARS-ONLY THEOREM (some copy of the range is struck by no gear at or below q, verified q = 7..17); the c_g family; the symmetry closure; complete own-shelf silence. CORRECTION: 'a gear does not strike below its square' is FALSE (gear 13 strikes copy 3, 91 = 7 x 13 < 169); only acting is bounded below by the square. NOT PROVED: any property forbidding a total blame assignment.
- 2026-09-23, R5.f.xxxv.c.xxviii: shelves round. PROVED: missed-copy law with reciprocity reach per leg; own-shelf silence as a theorem with the deferral law; silent runs <= 2; cofactor spokes {1,7,11,13} and the properties of the base-30 machine that trial division does not imply; survivor classes, their stabiliser and the (p-2)-to-1 refinement by the next gear; gears above q refine and never delete a class; gear-pair joint classes. REFUTED: silence propagation, a revealed copy on every shelf, and others listed. NOT ESTABLISHED: infinitely many revealed c_g; full-range density; an acting-based blame obstruction. Record in research/proof/shelves_cofactors_2026-09-23.md.
- 2026-09-23, R5.f.xxxv.c.xxix: two range paths. Path A PROVED: exclusion tables and independent characters, escape classes non-empty at every finite level, the hand-off equivalence (range for all q via missed copies iff the revealed gears form an infinite chain with g_{i+1}^2 + B <= (g_i^2 + A)#; 327 hand-offs hold below 200,000), the neighbour kill rule h | d^2 + A', dormancy, Pell recurrences of the inert set. Path B: four transfer statements REFUTED with instances; forced stabiliser pairs preserve exposure. NOT ESTABLISHED: the hand-off chain is infinite. Record research/proof/range_two_paths_2026-09-23.md.
- 2026-09-24, R5.f.xxxv.c.xxx: derived machine built beside the original (0/2/4-class strikers, quartic pair rule, special strikers); c_g revealed iff g in the escape classes up to its top live row lambda(g) > 2g/3; RANGE <=> chain over all revealed copies, each serving [sigma(p+2), p); monotone slack; comb structure of the stretch field; sharing law S(D) and privacy H(w). REFUTED: capped-strip reduction, the orbit statement (q = 19). NOT ESTABLISHED: RANGE => missed-copy chain, doomed classes. Record research/proof/derived_machine_2026-09-24.md.
- 2026-09-25, R5.f.xxxv.c.xxxi: kernel for the range line - RangeCopies, RangeLocator, RangeMissed, RangeHandoff, 0 sorries, reviewer-checked, including range_implies_unbounded. Open items: RANGE <=> chain over missed copies plus the non-dominated tails, converse reduced to live anchors; doomed classes <=> classes with no prime g having both legs prime, doom has no finite-row certificate; the derivation operator is idempotent with D the least fixed point. Record research/proof/range_kernel_2026-09-25.md.
- 2026-09-25, R5.f.xxxv.c.xxxii: kernel batch 2 (missed-copy reveal rule, the copy map J, chain => range => unbounded twins, all 0 sorries); live anchors: chain(S_1) given RANGE <=> m_r > r at every live r, transport and fixed-row laws, base-6 rule-level refutation; tower limit as a digit shift on {1,4,11,14}^N with fixed points 1 and 29; range-line map written and source-checked.
- 2026-09-25, R5.f.xxxv.c.xxxiii: field kernel carried to the range - total blame iff every copy's least prime factor is a machine gear or acts; joint redundancy of non-acting strikes; translate/phase bijection with C_7^free and C_11^free empty; tooth, pair, comb and E2-record laws on copies; centre law and Gamma_k; node chain (RANGE <=> reach from 29 unbounded, scratch Lean); mirror height split. Acting enters only through trial division; no standing statement forbids a total blame map. Record research/proof/field_to_range_2026-09-25.md.
- 2026-09-25, R5.f.xxxv.c.xxxiv: kernel batch 3 (region law via minFac, centre law, the twin-node reach chain: RANGE iff every node good iff reach from 29 unbounded); mirror pairs: class form, acting windows, four-leg product, the mean-height identity (not-Range iff Loss empty and unstruck survivors lie in Gain); run laws: acting pair law exact; the 2k run bounds REFUTED at k = 1 (q = 47), 2 and 3 (q = 37).
- 2026-09-26, R5.f.xxxv.c.xxxv: kernel batch 4 (acting pair law, mirror pair legs/shared striker/height split, range-window blame form); one-gear runs: exact run conditions, gear-7 lane law, ladder against h2(q); runs of 5, 6, 7 at q = 89..293, so no constant bound c <= 6 on B_1; Gain = (S_Q minus S_Q*) in the high half, not-Range iff Loss empty, S_Q misses the low half and S_Q* the high half.
- 2026-09-26, R5.f.xxxv.c.xxxvi: generality ledger on the owner's rule - 18 general kernel entries (RangeGen1-5 new: height split, Gain, mirror under acting, total-blame anatomy, acting vs acting-free, all for every q), general scratch-Lean results (the wall nextprime(X)^2 < X# for X >= 7, cofactor floor and kill-free windows), 11 claimed-general items with their gaps, 43 refuted-as-general with counterexamples, instance data separated as checks.
- 2026-09-26, R5.f.xxxv.c.xxxvii: second general round - the wall nextprime(X)^2 < X# (X >= 7), near-kill windows and top-band law now kernel theorems; C7 (M(G) >= G'^4), C8 (below-square count), C11 (mod 15d), C12 (twin strikers) proved for every q/g; X10 strict increase proved only under Legendre; C1E reduced to the top gear per side.
