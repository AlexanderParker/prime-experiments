# Theory tree toward the proof (manager, started 2026-09-04)

Method (the human's): construct a theory, test it, observe, interpret the patterns, describe them,
make a theory about the patterns, repeat. Each repeat is a branch. Dead end: step back, try
another theory; step back further if needed and revisit earlier branches. Depth when a theory is
strong, breadth when building the base. Statuses: STRONG (tested, holds, mechanism visible),
OPEN (untested or partly tested), WEAK (holds but no mechanism), DEAD (refuted or proved unable),
with the pointer to the evidence. Vocabulary: docs/proof-search/alignment-rules.md section 0.

ROOT. Openings always align inside the window: for every machine {5..y} the longest opening-free
stretch stays below the window's growth, F(y) < W(y) - y/6. Accepted as true; the work is the proof.
Equivalent per-step form (theorem): the budget inequality F(M+q') <= F(M) + q' at every step.
Reduction (theorem, attainment identity): budget = PAIR statement AND CHAIN statement.

## Branch 1. The pair statement: F_2(M) <= F(M) + q'  [join cost <= long arc]
Status OPEN. Free while F(M) < q' (through m17). Content from m19. Measured slack 5..25, widening.
- 1a. Phase-shift / sole-coverer mechanism (manager_notes 3-4). Record stretches are near-perfect
  tilings; every gear is a sole coverer somewhere; the top gear alone covers 1-2 columns. Shifting
  one gear's phase moves the hole onto that gear's sole columns. OPEN: gives structure, no bound yet.
  Test next: does the top gear's sole count stay <= 2 in every record and one-hole record to m31?
  If yes, F_2(M) <= F_3(M^-) always (descent), and the question becomes hole costs one level down.
- 1b. Descent through the survivor generator (record 3.9): F_2 at M is layer 0 of the algebra one
  gear down. OPEN: recursion with no base; hole costs not monotone in J (m29: 12,10,5,15,5).
- 1c. One-class transfer (manager_notes 2): one-hole(P_k) = j(P_{k+1}) through k = 18, so the
  one-class pair statement IS the one-class increment statement, satisfied by Hagedorn's table with
  increments 0.1-0.3 p. OPEN: literature lane checking whether it is known or conjectured. If a
  one-class proof exists it should transfer; if it is open there, that is the classification.
- 1d. Data past the scan wall: coverability spectrum by SAT (instrument lane). Extends F, F_2 to
  m61..m97 to see whether the slack keeps widening. OPEN (building).
- 1e. Mirror / d_0: F_2 >= 2 d_0 with d_0 the first twin above p. Lower bound only; the family's
  one failure is this wrap pair. WEAK as an upper-bound tool.

## Branch 2. The chain statement: Q*_J(M) <= F(M) + q'  [extra kills <= short arc]
Status OPEN. Kernel at six literal steps as the increment form; padded case open (m31 event).
- 2a. Par trading as a theorem: each added letter is paid by the flank envelope; Delta_J measured
  in [-3, +4]. OPEN (prover B).
- 2b. Literal case from the middle-sum lemma + pair statement as black box. OPEN (prover B).
- 2c. Padded case by the record law one level down (a padded middle is an old gap >= q', itself a
  merge). OPEN (prover B).
- 2d. Survivor-algebra contraction across layers. OPEN (prover B).
- 2e. Spectrum-plus-depth (F_J only, no legality): DEAD as a uniform tool - fails at 29->31 and
  47->53 (A_kill >= 4). Legality must be used.

## Branch 3. The whole-window route directly: F(y) < y^2/6 by a covering bound
Status OPEN, likely DEAD by constant: Iwaniec 1978 gives the right order y^2 with a large constant;
an explicit constant below 1/6 in two classes would be a major theorem. Literature lane checking.

## Branch 4. Top-down: records recruit runner-ups (genealogy)
Status WEAK: exact at 8 steps (ancestor a runner-up by 2-14, largest gap merged one level down 7 of
8, 1-5 generations), no rule stated. Theory to build: the record at M+q' is assembled from a
bounded number of near-record pieces of M, each assembled the same way; if the assembly has a
bounded branching, the growth per step is bounded by the pieces' growth. Untested as a theory.

## Branch 5. "Made at the top": the near-perfect tiling
Status STRONG as an observation (manager_notes 3): overlap in a record stretch is tiny, the top
three or four gears do the work, the top gear alone covers one or two columns. Theory to build:
the record stretch of M+q' is a record-class stretch of M plus one or two kills by q'; the join
cost is the length of the M-stretch adjacent to a record-class stretch. Why adjacent stretches
are ordinary is the unexplained rule (record 9.3 item 15).

## Dead ends on record (do not re-enter; alignment-rules.md section 6 and 8)
Residue arithmetic at any bounded modulus (escape distance 1); gears 5,7 capping the padded depth
past 53->59 (CORRCAP infinite); fixed-depth counting (kills nothing); pairwise convexity / SDP
(stops at m19); capacity and overlap counting (nearly achievable, no slack); transfer matrices over
the gap word (refuted twice); symmetry levers beyond the mirror (group is Z/2); letter size as the
driver of L (refuted on the family); congruence-class potentials (certify nothing).

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
