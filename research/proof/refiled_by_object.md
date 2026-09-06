# The record re-filed by object (opening task of the R4 line, 2026-09-06)

The owner has split the problem into three objects (theory_tree.md, node R4, THE ANALOGY and
the CONSTRUCTION RULE / REFINEMENT of 2026-09-06):

- **THE MOTOR** - the bottom machine `{5..q}` on its own. Columns `k = (6k-1, 6k+1)`, the
  anchor 2, 3, 5 folded into the 6-fold ruler, gear `g` (prime `5 <= g <= q`) striking
  `k = +-6^{-1} (mod g)`, gear 5 the first turning gear. Period `P = prod g`, openings per
  period `N = prod (g-2)`, record `F(M)` the longest opening-free stretch, spectrum the
  multiset of gap sizes.
- **THE WHEELS** - the top machine on its own: the primes in `(q, Z]` on the RAW LINE, each
  striking its multiples, its twin object the pair `(n, n+2)`, its own anchor the wheel of its
  smallest gears. Written in its own coordinate, never in the bottom's.
- **THE CLUTCH** - the interaction: every column classified by (bottom state, top state);
  both open = twin; bottom-open/top-closed = a candidate killed by a top prime; bottom-closed/
  top-open; both closed.

This document is a **re-filing only**. Nothing here is new; every entry is a fact already on
the record, moved to the object it is about and cited by file and node. Where a fact belongs
to two objects that is said. The budget inequality `F(M+q') <= F(M) + q'` is a **target**,
certified rung by rung, never a law. Vocabulary is the tree's: openings not kills; window =
the certified range `(y, y^2]`; section = the window's new part; stretch = a sliding run.

**Status vocabulary** (the tree's evidence standards, plus the written corpus):
`KERNEL` = a Lean theorem in `proofs/` named in docs/proofs; `PROOF` = a written proof in
docs/proofs or a branch document; `EXACT` = full periods, phase reduction, SAT/LP/ILP or CRT
certificates, with the count; `MEASURED` = holds, no mechanism or no completeness;
`TARGET` = wanted, never proved.

Counts: **74 motor entries, 12 wheels entries, 36 clutch entries.**

---

# 1. THE MOTOR - the bottom machine `{5..q}` alone

## 1.1 Construction and the tooth rule

**M1. The column and the fold.** Column `k` is the pair `(6k-1, 6k+1)`; the anchor 2, 3, 5 is
one object (cycle 30) folded into the ruler, and 5 is the first gear that turns. KERNEL
(definitional) - docs/proofs/02; tree profile Vocabulary.

**M2. The tooth rule.** Gear `q` strikes `k` iff `k = +-6^{-1} (mod q)`: two teeth, two arcs,
alternating spacings `2u` and `q - 2u`; the shield; teeth are never adjacent; twin gears share
the tooth `(p+1)/6` and both strike the column of `p(p+2)`. KERNEL - `TwoTeeth.kill_spacing`,
`kill_period`, `teeth_letters`, `AnchorChain.neighbour_of_hit`, `Polignac.twin_product_slot`,
`twin_split_class_iff`, `own_slot_pin_gap_two`; docs/proofs/02. Re-verified 493,101,490 checks
in walk_path.md.

**M3. The separation is one rational at every gear.** `3 d_g = 1 (mod g)`, i.e.
`d_g = 3^{-1} (mod g)`; equivalently, in the coordinate `n = 6k` every gear's teeth sit at
`+-1`. EXACT, 1,200 of 1,200 cells - pinned_arithmetic.md. **Belongs to two objects**: see
Appendix B.1 - this is the wheels' separation 2 seen through the 6-fold.

**M4. Period and opening count.** `P = prod_{5<=g<=q} g` columns; exactly `N = prod (g-2)`
openings per period, by CRT. PROOF - period_scale.md section 2; tree R4.

**M5. Arcs, and why twin gears share one.** `3 a_g = g -+ 1`, so two gears share an arc iff
they are a twin prime pair; `{5..q}` therefore carries only `pi(q) - 2 - pi_2(q)` distinct
arcs and must buy both members of every twin pair. PROOF - gear_count.md; half_column.md
(fibre theorem).

**M6. Theorem (E), the effective machine.** For every column with `6k - 1 > q`: blocked under
`{5..q}` iff blocked under `{5..floor(sqrt(6k+1))}`. The exception set is exactly the twin
gear pairs striking their own home columns (7 of 7 count matches), all below `(q+1)/6`.
PROOF (one line) - position_frontier.md, node R2.e.i.

**M7. A gear exposes nothing below its own square.** KERNEL - `Gear.R_eq_zero_of_below_sq`,
`Layer.slot_cap`; docs/proofs/15. (The rest of the layer law is a clutch statement: C16.)

## 1.2 Always-open columns and the mirror

**M8. Column 0 and the antipode are always open; the mirror `k -> -k`; the symmetry group of
the opening set is exactly `Z/2`.** KERNEL - `Mirror.mirror_gear`, `antipode_open`,
`self_mirror_unique`; docs/proofs/03. This is face C1 of the wall.

**M9. Parity of counts.** Every gap length `>= 2` occurs an even number of times, so the
record never occurs exactly once; window counts are even. KERNEL -
`Mirror.even_card_involution`, `window_count_even`, `adjacent_equal_even`,
`none_of_at_most_one`; docs/proofs/03; register docs/novel/mirror-parity-laws.md.

**M10. The mirror lever is worth exactly one unit.** The full affine symmetry group is
`(Z/2)^m` (multiplication by `c = +-1` mod every gear) but only `c = +-1 (mod P)` preserves
adjacency; there is no mod-4 version. PROOF - mirror-parity-laws.md section 7.

**M11. The mirror is fixed-point-free on the word-legal family.** The self-mirror depth-`J`
window is never word-legal for `J >= 3` (odd `J`: its central middle is the antipodal gap of
length 1, illegal since `2u' = 3^{-1}`; even `J >= 4`: its two central middles are both `d_0`,
forbidden by T3). `J = 2` is the only depth needing a hypothesis, and there it is exactly
`d_0 != F`. PROOF, gated m11..m23, 185 assertions - mirror-parity-laws.md section 9.

**M12. The mirror at column 0 gives the pair `(d_0, d_0)`, hence `F_2 >= 2 d_0`.** THEOREM
(deletion ladder plus mirror) - pair_statement.md L3, docs/proofs/19; node 1e. This is the
obstruction: the pair statement at column 0 reads `2 d_0 <= F + q'`.

**M13. The frontier mirror.** `R_max = P - R_min - L + 1`, 88 of 88. EXACT -
position_frontier.md.

## 1.3 The alignment law and the dominoes

**M14. The alignment law.** The longest run of consecutive openings equals the long arc
`q_0 - 2u_0 - 1` of the smallest gear; with gear 5 in the machine the openings are isolated
points and dominoes, `prod (q-4)` dominoes. PROOF (full CRT proof, written; the record had a
check on 103 gear sets) - docs/proofs/04.

**M15. The all-teeth column.** The blocked run through a column struck by every gear is always
exactly 1 - both neighbours are open, since `6(k +- 1) = +-1 +- 6 (mod g)` vanishes only if
`g | 4` or `g | 8`. PROOF (one line) - tree log, manager scan 2026-09-06
(research/anchor235/r49/allteeth_record.py).

**M16. The record is not anchored at the alignment points.** The distance from every record and
near-record stretch to the nearest all-teeth column is random (median equal to the random
expectation at m11..m23). MEASURED - same scan; thin place 3, DEAD in its sharp reading.

## 1.4 Adding a gear: merge law, chain law, letters, legal words, caps, corridor

**M17. Adding a gear.** The `q'` copies realise every deletion phase exactly once and each
opening dies in exactly two of them; the hit law; the CHAIN LAW (`y - x = 0, +-d`); the MERGE
LAW; the grammar T1-T5 (alphabet `{a, b}`, residue necessity, strict alternation, spacing
`>= 2u`, fuel cap); legal words. KERNEL - `AnchorChain.copy_phase`, `phase_bijective`,
`chain_law`, `hop_zero`; `MergeLaw.interior_gap_mod`, `newgap_le_step`; `TwoTeeth.kills_gap_ge`,
`fuel_span_cap`; `WordLegal.legal_iff_noRepeat`, `killable_iff`; docs/proofs/05; register
merge-law, anchor-235-layer-laws.

**M18. Deletion spacing.** Merge deletions are `>= q - 1` apart, and tight. PROOF - register
deletion-spacing.

**M19. The letters.** An added gear's kill spacings lie in the two letter values
`{2u', q' - 2u'}` with exact `q'` padding, strictly alternating, minimum `2u'`. KERNEL
(T1-T5, `proofs/TwoTeeth.lean`, `MergeLaw.lean`); docs/proofs/02; register
two-teeth-kill-spacing.

**M20. The word reduction.** `Q*_J > -inf` iff `L(M) >= J - 2`, so `J_max = L + 2` and
`A_kill = L + 1`; chain iff legal word; the same-tooth lemma (middle span `= 0 mod q'` iff an
even number of non-padded middles, so a literal even-`J` chain starts and ends on the same
tooth). KERNEL - `WordLegal.chain_iff_word`, `qstar_iff_word`, `jmax`, `akill`, `same_tooth`,
`same_tooth_window`, `literal_even_span`; docs/proofs/10; register even-j-mechanism.

**M21. The bare-word cap - the first uniform cap on half of `L`.** `L_bare(M) <=
PSORD(q' mod 210) <= 5`; `PSORD in {1, 2, 3, 5}` with the 28-class set `S = {PSORD <= 2}`;
`PSORD = 4` is empty. KERNEL - `BareAlt.no_bare_run_ge`, `bareAlt_inadmissible_iff`, `S_card`,
`psord_le_five`, `psord_ne_four`; docs/proofs/12; register bare-word-uniform-cap. `L_pad` is
untouched and grows (0,0,0,1,1,1,2,2,2,2,3,3 at m11..m53).

**M22. Literal and Polignac caps.** A literal chain has at most `capC(q mod 210) <= 6` exposed
members, the table exact, no class of cap 5; over all even gaps the cap depends on
`gcd(e, 105)` and is at most 12. KERNEL - `LiteralCap.literal_chain_le_six`,
`cap_six_classes_sharp`, `LiteralCapTable.*`, `PolignacCap.capOf_le_twelve`; docs/proofs/13;
register literal-cap, polignac-cap.

**M23. The only bound on `L`.** `L(M) <= 2 floor((F(M+q') - 2)/q') + 1 <= 2 F(M+q')/q' + 1`,
with letter-aware and parity forms; so `L` is `O(F/q')`, not `O(1)`. PROOF (written, from
attainment and T3) - docs/proofs/11; register spectrum-bound-on-L.

**M24. The corridor.** `E_35` (15 residues); endpoint and adjacency laws (294 forbidden
pairs); tier A carriers; the completeness lemma `q <= 2n`; the 32-cap on prime-adjacent runs;
the adjacent-gap exclusion law mod 5 (6 classes, complete); the AP lemma; padding onset, count
and the 12 forbidden equal-padding classes. Partly KERNEL - `Corridor.exposed_iff_mem`,
`forbidden_pairs_count`, `prime_adjacent_run_le`, `TierA.*`; docs/proofs/14; register
corridor-law. Constrains where, never how big (face B1).

**M25. Phase saturation.** A gap word with exposed offsets `X` cannot occur at all if some gear
has no admissible phase; since the union has at most `2|X|` elements the whole content sits at
gears 5, 7, 11, giving a closed-form per-step ceiling on the pure alternation (6, 2, 2, 2, 5,
3, 3, 4 at 31->37 .. 61->67, attained at 47->53). EXACT - register phase-saturation-arity,
uniform-order-bound (`A_relax(M) <= 5` at every machine, `<= 4` off six classes mod 210).

**M26. Peel, triple and middle-sum.** Peel bound `Q*_J <= Q*_{J-1} + min flank`; the triple
inequality `g_L + w + g_R <= F_2 + min(g_L, g_R)`, hypothesis-free; the middle-sum lemma
(literal middles sum to `>= k q'` or `k q' + a`); even-`J` literal runs are never palindromes.
PROOF (written; the recorded per-`J` flank envelope's conditional step is flagged) -
docs/proofs/16; register per-j-window-analogues.

**M27. The legal-word length mechanism.** `L_g(M)` is predicted to within one unit by an
independent-letter model with the real class densities of the legal alphabet in `M`'s gap
histogram, while the COUNT of legal windows collapses at the top. MEASURED - register
legal-word-length-mechanism.

**M28. The exposure cap.** Exposure at word length `m` is decided by the gears `<= 2m + 2`
alone; `EXPCAP - L` is unbounded along the ladder (16, 11, 8, 18 at m37, m41, m43, m53) and
fixed-depth Bonferroni kills nothing among the exposure survivors. THEOREM + EXACT m11..m53 -
register cover-half-counter-ladder.

## 1.5 The record

**M29. The record law (phase reduction).** `F(M+g)` is the max over the `g` phases of the
largest gap of the phase-`r` sequence on ONE lower period; kernel-checked at both ends at
machine 17 (`F(17) = 18`); the nested next-opening formula `next_G = next_M^{k+1}` past a run
of `k` hits. KERNEL - `AnchorRecord17.surv_shift`, `record_max`, `F17_eq_18`,
`AnchorChain.hop_iter`, `hop_zero`, `hop_one`; docs/proofs/09; register
anchor-235-layer-laws (L3).

**M30. The attainment identity.** `F(M+q') = max(F_2(M), max_{J>=3} Q*_J(M; q'))`: a legal word
is always struck in full somewhere, and every new gap is a merged run with legal middles.
Partly KERNEL (`WordLegal.killable_iff`, `chain_iff_word`, `MergeLaw.newgap_le_step`,
`AnchorChain.phase_bijective`), the identity itself written - docs/proofs/08; register
kleene-generator, old-machine-spectrum. It splits the record into the PAIR statement (`J = 2`)
and the CHAIN statement (`J >= 3`); both are open.

**M31. The deletion ladder.** `F_{r+1}(M) <= F(M + r new gears)`, in particular
`F_2(M) <= F(M+q')`. PROOF (written CRT proof) - docs/proofs/07.

**M32. Saturation.** If `F(M) < 2u_q` (in particular if `3F(M) < q - 1`) then
`F(M+q) = F_2(M)` exactly. PROOF - docs/proofs/06; register saturation-theorem. Closed, but in
a regime disjoint from every rung.

**M33. THE FRONTIER.** `F(M + q') = max_a (a + Rest(a))` exactly, where `Rest(a)` is the most a
gap of size `a` gains by fusion. EXACT, 8 rungs, 0 exceptions - merge_forest.md, node 4.i. The
record is made at an interior `a / F_old = 0.58-0.70` at the top three rungs. The
strengthening `Rest(a) <= q'` would give the budget inequality in one line; it holds at rungs
7..29 and FAILS at 29 -> 31 (max rest 34 > 31).

**M34. The top law.** `Rest(F_old) = N(F_old)` if `F_old = 0` or `+-d (mod q')`, else
`n1(F_old)`; so the top's slack `s(F_old) = q' - Rest(F_old)` is an identity (4, 9, 10, 10, 12,
18, 24, 29). EXACT, 8 of 8 - frontier_collapse.md. Mechanism: the letter floor `a_L = q'/3`
against neighbour shortness.

**M35. The fusion-rate identity.** An old gap is fused in exactly 4, 3 or 2 of the `q'` copies
(generic / `+-d` / `0 mod q'`) and is an interior piece in 0, 1 or 2: junction availability
never collapses, only piece size does. PROOF (from docs/proofs/05), 137 cells -
frontier_collapse.md.

**M36. The availability gate.** `J = 3` needs a legal `a` or `hasM(a) > 0`; `J >= 4` needs
`hasM(a) > 0` always; the gate is ONE ROW of the level-2 dictionary, with the closed form
`a_hasM <= F_2(M) - a_L` proved in one line (68 of 68 including 60 family members). PROOF +
EXACT, 137 cells - availability_gate.md.

**M37. The short-letter row and the pinned letter.** The row `(a, a_L)` of the adjacent-pair
dictionary is empty above its realised top - proved scan-free by LP duality at m19 (above 20),
m23 (25), m29 (35, 270,070 exact operations). The pinned letter `F(M) <= a_L + r(a_L) <=
F(M) + 3` held 8 of 8 and out of sample at 31 -> 37 (`r(12) = 46`, `a_L + r = 58 = F`), and its
LOWER half is REFUTED out of sample at 37 -> 41 (`a_L + r(a_L) = 77` against `F = 88`); the
upper half survives 9 of 9 with slack 3, 1, 3, 0, 2, 1, 3, 14. PROOF (LP duality) + EXACT (CRT
row search) - short_letter_row.md, pinned_letter.md, pinned_arithmetic.md.

**M38. Record saturation.** At every one of 68 record occurrences of eight machines, every gear
of `M` is the sole striker of a column INSIDE the record gap itself; the configuration is
frozen, so `n1(F)` is a max over `2 m(F)` determined numbers. EXACT, exhaustive -
record_2run.md.

**M39. The spare-gear lemma.** If a 2-run has a gear that is neither obstructed at the middle
opening nor a sole striker inside the run, then `F(M) >= a + v`; contrapositive: the excess
`E(v) > 0` is exactly "no free gear". PROOF, 0 counterexamples in 13,616 runs; 133 of 133 runs
of span above `F` have no free gear - pinned_letter.md.

**M40. L4.** Every gear is a sole striker in any above-record stretch, teeth-free, in both
worlds; with the single-gear re-phasing certificate. PROOF - docs/proofs/19, pair_statement.md.

**M41. THE GEAR-5 LOCK.** Every maximal blocked stretch of every machine, at every length
(record, runner-up, window stretch, anywhere) has gear 5 at its coverage-maximal phase. PROOF
(five cases from the teeth `{+-1} mod 5` and the two flanking openings being non-teeth);
exhaustive to `L = 2000`; gated at all 62 records of m13..m31 and 1.7 million window stretches
at 295 rungs - gear5_lock.md, node 5g.

**M42. The slot rule.** `F = 1 (mod 5)` starts on slot 11|13, `F = 4` on 17|19, `F = 2` or `3`
on a mirror pair of slots, `F = 0` on any. EXACT at all eight full periods to m31 -
anchor_cycles.md, node 5e. This is M41 read at the stretch's start.

**M43. The allocation law at records.** Every gear of a record is at its coverage maximum
SUBJECT TO keeping the columns only it strikes: 340 of 348 gear-cells over all 62 records; the
gears below maximum are middle ones. EXACT - gear5_lock.md. Period records have 78% of
gear-cells at maximum and 2.3% free deficits; window stretches 30% and 63%; only the lock is
shared.

**M44. The record set is pinned.** `F(M minus g) < F(M)` for every `g` at m7..m23, and the
minimum blocking set of the period record is the whole machine; the record set has 2, 4, 12,
20, 20, 4, 2, 4 stretches at m7..m31; from m23 the record is ONE residue class mod the period
up to mirror (at m31 four stretches with every gear but 29 and 31 pinned). EXACT - node 5d,
record_frame.md, deletion_profile.md. Pinning says WHERE the record is, not that an opening is
forced into the window.

**M45. Records are made at the ends, of ordinary lower gaps.** A record is a row of ORDINARY
lower gaps whose junctions the top three gears strike: m29's `43 = 10 + 10 + 23` as gaps of
`{5..23}` (own record 34); m31's `58 = 23 + 10 + 25` as gaps of `{5..29}` (record 43); m23's
`34 = 4 + 8 + 15 + 7`. Junctions are closed by exactly three gears taking 3 + 2 + 2. The switch
is sharp at rung 19 -> 23 (largest-piece rank fraction 0.35, 0.30, 0.27, 0.33 after it). EXACT
- ends_or_middles.md (R3.h), record_2run.md. `F = flank + letters + flank`.

**M46. The junction theorem.** The junction condition is a congruence mod `q'` and the old
machine is periodic mod `P` with `gcd(P, q') = 1`, so over the period the flank pairs at
junctions are exactly the flank pairs at all old openings, each twice: a junction is an
ORDINARY opening and the maximum flank sum at junctions IS `F_2(M)`. PROOF - flank_walk.md
(closes weak point W5: the flank brick is the pair statement).

**M47. L6, made exact.** The left tiling is the negated right tiling gear by gear, equal iff
`g | x`; and `b_g^+ + b_g^- = a_g` or `g - a_g` at every opening and gear (0 exceptions in 10.3
million pairs), forcing exactly two things - a gear acts on both flanks only if `a_g <= S - 2`,
and a gear that misses a stretch has `g - a_g >= S + 2`. PROOF - docs/proofs/19, flank_walk.md.
Across a gap: `p_g + q_g = -v` or `-v +- d_g (mod g)`, 0 violations in 2.39 million pairs
(neighbour_profile.md).

**M48. Three gear bands at a flank.** Gears with `g - a_g < S + 2` strike at 100.00% (2.28
million cells); the middle band strikes at `0.796 +- 0.004`, constant over `q = 59..997`; the
top band falls 0.36 -> 0.18. The length is decided in the middle band. EXACT - flank_walk.md.

**M49. The flanks are coupled by the anchor.** `L^+ = 1 (mod 5)` forces `L^- in {0, 2, 4}`,
`L^+ = 4` forces `{0, 1, 3}`, `L^+ = 2` forbids 4, `L^+ = 3` forbids 1; with gear 7, 931 of
1,225 pair classes mod 35 are admissible. EXACT, 0 exceptions in 8.8 million openings -
flank_walk.md. Not by L6.

**M50. The branching identity.** `n_J = C_{J-1} - 2 C_J + C_{J+1}` with `C_0 = q' N`, proved in
general by run-length inversion; `C_r = W_{r-1} + Z_{r-1}` (legal words plus all-pad);
`max order = L + 2 = J_max`; the second moment in closed form; the size side
`m_{M+q'}(v) = sum_J sum eps_J` with `eps_J in {0, 1, 2}`; closure at depth `K_m <= m J_max`.
PROOF - branching_identity.md, merge_forest.md. At rung 31 the order distribution of 6.23
billion gaps comes out of five numbers.

**M51. What is teeth-free in the merge.** The mean merge order is exactly `q'/(q' - 2)` at 8
real rungs and 21 of 21 family members; the teeth live in the VARIANCE and nowhere lower (real
machine at percentile 0.19). PROOF + EXACT - merge_forest.md, branching_identity.md.

**M52. The ladder, and the ladder past the scan wall.** Certified `F` ladder 2, 5, 7, 11, 18,
25, 34, 43, 58; then `F(37) = 88` and `F(41) = 91` exactly from m23's period alone, every gate
exact, with the span-threshold prune as the tool. Budget slack along the extended ladder 14,
20, 16, 7, 38 at 23 -> 29 .. 37 -> 41, not monotone. EXACT (instrument) - ladder_closure.md,
merge_forest.md; register cov-sat-exact-spectra.

**M53. The position-length frontier.** `R_min(L) = 1` for every `L < d_0` and `R_min(L) >=
3.25 L` for every `L >= d_0`, 0 exceptions in 113 period cells (m7..m29) and 8,375 window cells
(`q = 23..19,997`); the frontier is bimodal with nothing between; theorem (E) delivers
`c = 1.25` unconditionally (1.54 for `L >= 6`). PROOF + EXACT - position_frontier.md (R2.e.i).
The frontier constant and the record constant are one number in two coordinates.

**M54. The certified rungs.** The increment law at the six literal steps 11->13 .. 29->31
(KERNEL `Increment.increment_law_literal_steps`, `IncCert23/29/31.F_le`; docs/proofs/17; false
at 31->37 as a general law); the case split 31->37, `F(37) <= 95`, 385 exhaustive held-phase
cases each an exact integer dual certificate (KERNEL `CaseCert37.F_le`; docs/proofs/18). Also
scan-free: the restricted-covering case split certifies every rung the project has to 41->43,
and the spectrum-depth certificate ratifies 41->43 and 43->47. Registers
restricted-covering-certificates, spectrum-depth-certificate.

**M55. The budget inequality is the TARGET.** `F(M+q') <= F(M) + q'`, certified rung by rung
(files 17, 18 and the `Machine*` ladders), never a law. docs/proofs/README "Not proved, and
said so"; tree profile.

## 1.6 The spectrum

**M56. The spectrum recursion.** `m_{M+q'}(v) = c_{q'}(v) m_M(v) + Merge(v)`, survival exact at
137 of 137 cells, reproducing the m31 spectrum in 30 s; `A(v) = prod c_q(v) >= prod (q-4)` with
equality iff `v` is uncoupled; `A(1) = m(1)`; no size is ever lost and every size is born a
merge (137/137, 55/55). EXACT - spectrum_sum_rule.md; register paired-holt-recursion (the
one-residue case is prior art).

**M57. Every spectrum hole is a phase hole, never a span hole** (7 of 7). EXACT -
spectrum_sum_rule.md.

**M58. Record isolation.** The record is isolated by 3 in the gap spectrum at m29 and m31 (no
41, 42 below 43; no 56, 57 below 58); at m37, 89 and 90 are certified empty with 13 certified
holes and the top band 88, 85, 77, 72, 71. EXACT - gear5_lock.md, record_2run.md.

**M59. Uncoupled sizes, classified.** For `v < y^2/3`, `v` is uncoupled in `{5..y}` iff `v` is
`y`-rough and its half-column is a twin column above `y` (5,505 of 5,505 cells, `y` prime
5..199): the even distances a machine cannot couple are exactly twice the twin columns above
it. The spectrum rule is dead both ways (24 occurs 1,180 times at m29) but the graded form is
exceptionless: an uncoupled size is depleted by a factor 12 to 128 against its coupled
neighbours (10 of 10), and the flip is exact (`v = 41` absent at m29, realised 134 times when
31 arrives). PROOF + EXACT - half_column.md.

**M60. The half-column map.** Both letters of a gear point at its home column (1,228 gears to
10,007); `Leg(v) = {g : g | 3v - 1 or 3v + 1}` (400 of 400); the FIBRE THEOREM (exactly three
distances have half-column `c`, namely `2c`, `4c - 1`, `4c + 1`, and they are the alphabet of
that column's gears - 2,000 columns, 0 exceptions); the FIXED-POINT THEOREM (a column is a
fixed point of the halving descent iff it is a twin column). PROOF - half_column.md,
separability.md. Records in column coordinates: m29's `10 + 10 + 23` and m31's `23 + 10 + 25`
live in columns 5 and 6 only; every record letter lands on the new gear's home column, 11 of
11.

**M61. `N(v) <= F_2(M)` for every realised gap size `v >= 6`,** exceptionless on full periods
to m31 (6.4 billion gaps), tight once (`N(7) = 55 = F_2` at m29); spikes only at `v <= 5`.
PROOF of mechanism (the GLUE LEMMA: re-phasing the right flank by CRT under any two-colouring
makes the glued middle column an opening, so the glued object is an adjacent PAIR, bounded by
`F_2` and never by `F`) - neighbour_profile.md. The `F + 1` law it replaced is DEAD (killed at
m29 by 4 at the letter, run `(18, 10, 30)`).

**M62. The top of the spectrum is pinned to `F_2`, not `F`.** The least valid `c` in
`v + n1(v) <= F + c` on `v >= 0.8 F` is 4, 5, 7, 6, 5, 12, 7; `D_top = F_2 - max` is 0 at six
machines and 3 at m31. And `max_a (a + Rest_2(a)) = F_2(M)` is an identity. EXACT -
record_2run.md, frontier_collapse.md.

**M63. The J-run outer law.** For `J` consecutive gaps with every middle `>= 6`,
`g_1 + g_J <= F_2`; 0 exceptions in 3,278,972 runs, `J = 3..8`, m13..m23, the maximum falling
with `J`. Drop the middle condition and it breaks at once. EXACT - glue_covering.md.

**M64. The shadow and move lemmas.** The covering instance of a 3-run has exactly two
single-sided columns (`x_1 - v`, `x_2 + v`), so the glue's whole content is buying one column
(min miss 1 at 178 of 178 failures); and recolouring a gear translates its strikes by `v`, so a
strike survives iff `v = 0` or `+-d_g (mod g)` - padded gears move free and never cover the
shadow, letter gears keep one tooth, all others lose everything. PROOF - glue_covering.md.

**M65. Adjacency repulsion / the suppression law.** Gaps next to a large gap are shorter than
independence gives: `F_2` actual 11..39 at m11..m23 against shuffled 12..55; the gap after a
gap `>= 0.7F` is below the mean at every machine; suppression is real beyond rarity (`n1` below
the rarity null at 34 of 35 top-band cells). MEASURED, structural in 95% of family members;
the rigorous side is the renewal ladder - node 5b, register suppression-law, renewal-ladder,
record_2run.md.

**M66. Position facts kept as breadth.** Corridor resonance (big gaps recur at slot separations
35, 70, 105 with left endpoints pinned to `{10, 12, 18} mod 35`) and the golden spectral gap
(gear 5's local frequency mode is `phi`, `phi/3` a machine-independent spectral gap). MEASURED
/ PROOF - node 5f, registers corridor-resonance, golden-spectral-gap. Both under the
escape-distance-1 ceiling (face B1).

## 1.7 The family: what is real-teeth and what is teeth-free

**M67. TEETH-FREE: the record law itself.** `max(F_2, max_J Q*_J) = F(M+q')` at all 27,570
counterfactual machines across five steps: the attainment identity survives moving the teeth
exactly where the budget inequality and the increment law do not. EXACT - register
tooth-counterfactual-percentile section 5B.

**M68. TEETH-FREE, the rest.** The mean merge order `q'/(q'-2)` (M51); L4 (M40); the `L1`
character bound (`sum_m |Shat|/P = prod S_q/q` is independent of the teeth, identical at all
30/180/1440 counterfactual tooth vectors while `F` spreads 1.83x-2.50x); the branching
identity's count side (M50). PROOF - merge_forest.md, docs/proofs/19,
walk-transform-pole-identity.

**M69. REAL-TEETH.** The increment law is not generic (violated by 13-22% of the family,
growing with the machine; pinning the new gear's tooth drops it to 0-6.5%); the pinned letter's
constant 3 (only 43 of 63 tooth-counterfactual members obey `0 <= E(a_L) <= 3`, range -6..+7,
and every step of the glue construction is tooth-invariant, so no glue argument can prove it);
the chain statement needs the real higher gears' teeth (2f REFUTED at 23 -> 29 by a member with
gears 5 and 7 real, no adjacent teeth and a pinned incoming tooth, giving 62 > 61); `L` bounded
is not structural (max `L` on the family 1, 3, 3, 3, 5 against the real 0, 1, 1, 1, 2, the teeth
entering through the mod-{5,7} admissibility of the bare alternation). MEASURED / EXACT -
tooth-counterfactual-percentile, pinned_letter.md, chain_statement.md, node 2f.

**M70. Where the real machine sits in its family.** `F` at the 17th-26th percentile of the
exhaustive tooth-counterfactual distribution at m11..m19 (the favourable direction, and it
strengthens with depth: m23 `F` 11.9%, `F_2` 3.1%); `F` at the 14th and 22nd percentile of
random symmetric spacings, so coherence explains nothing (node 6, face C2); the budget slack
places the twin at 59.0%/37.2%, undistinguished; twins at the 13.3rd percentile of their own
even-gap family. EXACT / COMPUTED - registers tooth-counterfactual-percentile, twin-percentile.

**M71. Twin gears help the record.** De-twinning LOWERS the record at every rung
(`F_real/F_detwinned` = 1.10, 1.29, 1.47, 1.70, 1.48, 1.66 at m13..m31) and mean `F` at fixed
gear count is monotone INCREASING in the number of duplicated arcs (5,383 sets, no exception):
a twin pair is the cheapest pair of small gears. EXACT - arc_multiset.md.

**M72. `A(K)`, the adversary with a free gear set.** `A(K) = 2, 5, 7, 16, 22, 28, 37, 45, 68,
88, 101, 115` at `K = 1..12` exactly over all primes; the open lemma
`A(K) < (p_{K+1}^2 - 1)/6` holds at every `K <= 12` with margin 2.7-3.8 and `A/W` flat
(0.26-0.37); PROVED for `K <= 10` by certified infeasibility of an exact 0/1 program, and
`A(K)` for `K <= 6` by reasoning (arc law, capacity bound, SPAN LEMMA, type lemma). KERNEL-free
certificates - docs/proofs/20, arc_multiset.md, gear_count.md, small_K_theorem.md. Strictly
stronger than the root; the real machine `{5..37}` is an optimal 10-gear blocker.

**M73. Collision laws for gear pairs.** The deficit is linear with slope `4/(gh)`,
`c(g, h; L + gh) = c(g, h; L) + 4` (248,334 real and 67,400 random instances); the SHARED-ARC
LAW (equal short arcs collide from `L = a + 1`, so every twin pair collides at `(g+4)/3`, the
earliest possible); the ARC FLOOR; the head collision (gears 5 and 7 cannot be simultaneously
maximal and disjoint); the block bound `L <= sum_B joint_max(B; L)` with the block-size ladder
1, 2, 2, 3, 4, 5, 6, 7, 8, 8 at `K = 3..12`. PROOF + certificates - docs/proofs/21,
collision_laws.md.

**M74. The one place the real teeth are atypical.** Gluability: the real teeth glue at 62.5%
against a pooled 9.4%, the 99.6th percentile of 223 comparable m19 members, not explained by
the count of letter gears; re-measured at matched `(v, slack)` cells the exception survives in
direction and shrinks from a factor 6.6 to about 2.4. MEASURED - glue_covering.md,
separability.md. This is face C's only exception.

---

# 2. THE WHEELS - the top machine alone

Little is on the record, and the owner's construction rule says why: until now the wheels were
inferred only from the motor's behaviour. Every entry below is marked
**[top-alone]** (a statement about the primes above `q` needing no bottom gear) or
**[measured only in the clutch]** (established inside the two-machine experiment, and stated
here only because its content is top-alone: the bottom machine enters only as the choice of
arena `[0, P)` and the cut `Z = isqrt(6P+1)`).

**W1. THE CONSTRUCTION RULE (owner, 2026-09-06).** The top machine is written on the RAW LINE:
gears = primes in `(q, Q]`, each striking its multiples; its twin object is a pair `(n, n+2)`,
so in pair coordinates each gear has teeth at `0` and `-2 (mod g)`, separation 2. Its own
anchor is the wheel of its smallest gears `q' q'' q'''` with `(q'-2)(q''-2)(q'''-2)` twin slots
per turn; the pair `(-1, +1)` straddling every wheel multiple is open for every gear (its
column 0); the reflection `n -> -n - 2` is its mirror; its interaction laws for pairs and
`n`-tuples carry the chain and merge laws with letters `{2, g-2}`. Whether a set of its lowest
gears structures left/right slots as 2, 3, 5 do is to be asked of the wheel in its own
residues. RULE, not yet worked - theory_tree.md node R4. **[top-alone, and it is the mandate,
not a result]**

**W2. The mirror about the origin.** The top machine is mirror-symmetric about column 0: the
state of column `-k` is the state of column `k` with the two members swapped. EXACT, 0
mismatches over 2,853,543 columns each side - period_scale.md 3.4, table 8.
**[top-alone; measured in the clutch]**

**W3. Column 0 is open in the top machine at every `q`** (its members are `-1` and `1`, and
every gear's teeth are at `+-u_g`, never 0). EXACT, 5 machines, 0 exceptions - period_scale.md
3.4, table 8. **[top-alone]** This is W1's `(-1, +1)` in the bottom's coordinate.

**W4. It is not periodic over the range, and it is not uniform.** Its density falls
monotonically across `[0, P)`: at `q = 23` the twenty block densities run 0.1727 down to
0.1224, a 41% fall, while the bottom machine's are constant to six figures. It agrees with its
own translate by any bottom subperiod (5, 35, 385, 5005, 85085, 1616615) at 0.7709-0.7752
against the independent expectation 0.7719 - exactly chance, no periodicity at any bottom
subperiod. EXACT - period_scale.md 3.1. **[top-alone; measured in the clutch]**

**W5. In-range density exceeds the independent-gear product by 6-9%** at every `q` (1.0655,
1.0590, 1.0928, 1.0921, 1.0754 at `q = 11..23`). Mechanism: a column is top-open iff neither
member has a prime factor in `(q, Z]`, and since the range is `Z^2` each member is then
`q`-smooth times at most one prime above `Z` - a one-dimensional-looking object with a Buchstab
excess, not a two-tooth sieve. EXACT - period_scale.md 3.1. **[top-alone; measured in the
clutch]**

**W6. Every top gear does exactly its fair share.** `|kills - 2P/g| < 1` at every gear and
every `q` (maxima 0.62-0.68 over 2,338 gears), so the kill count is `round(2P/g)`. There are no
"few-strike" gears at the period scale - that is a window phenomenon; over the period the
smallest and largest top gears differ in workload by a factor of 515, not by orders. EXACT, 0
exceptions - period_scale.md 3.1, table 8. **[top-alone; measured in the clutch]**

**W7. Its open runs are short.** The open-run spectrum at `q = 23` is 1: 3,804,784; 2: 504,459;
3: 64,997; 4: 7,972; 5: 952; 6: 102; 7: 19; 8: 2 - geometric decay by about 7.5 per step;
longest open run 7, 5, 6, 7, 8 at `q = 11..23`. EXACT - period_scale.md 3.1.
**[top-alone; measured in the clutch]**

**W8. CLOSED-RUN RECORDS, with the correction.** The period's longest closed run (7, 27, 114,
378, 1,376) sits INSIDE the placement prefix and is an artefact of the home strikes; the honest
top-machine number is the longest closed run above the prefix: **7, 24, 30, 58, 104** at
`q = 11..23`. EXACT - period_scale.md 3.1 and Dead ends. **[top-alone above the prefix; the
prefix version is a clutch artefact - Appendix B.4]**

**W9. Placement geometry.** A top gear is PLACED at its home column `h(g) = round(g/6)`; all
placements lie in the prefix `[1, ceil(Z/6)]` (2,490 columns of 37,182,145 at `q = 23`); a
placement is a strike but not a kill (the member it divides there is the prime `g` itself);
no column ever carries three top gears (2,338 placements, 0 exceptions, since a prime triple
`g, g+2, g+4` above 3 is impossible). EXACT - period_scale.md 3.5, table 8. **[top-alone]**

**W10. Double occupancy, and what it is worth.** Doubly occupied placements (a column carrying
two top gears) number 3, 9, 26, 78, 268 at `q = 11..23`, equal at every `q` to the number of
twin pairs in `(q, Z]`. PROVED: "for every `q` the top machine has a doubly occupied placement"
is EQUIVALENT to the infinitude of twin primes, strictly weaker than the window statement and
strictly stronger than R4's own survivor statement:
`window => placement => survivor`. PROOF - period_scale.md 4.4. **[top-alone in statement;
the equivalence is the conjecture itself]**

**W11. THE ZONES (owner, made exact by the manager).** On a range of `K` columns the top gears
split three ways: REPEATING gears `g <= sqrt(6K)` turn past their own square inside the range
and have exclusive kills; NON-REPEATING gears `sqrt(6K) < g <= 6K/q'` kill bottom-open columns
only at members `g m` with `m < g`; SILENT gears `g > 6K/q'` strike only their own home column.
So the effective top machine on a range is exactly `{q'..sqrt(6K)}`, a gear machine of the same
construction with NO anchor. Named, not yet formalised - theory_tree.md node R4.
**[top-alone in the repeating/silent split; the non-repeating clause is stated against
bottom-open columns, hence clutch]**

**W12. THE REDUNDANCY LEMMA (one line).** A non-repeating gear's kill of a bottom-open column
coincides with a smaller top gear's, because the smallest prime factor of the cofactor `m` is
itself a top gear below `g` striking the same column. Named as one of the two first things to
formalise when the line opens - theory_tree.md node R4. **[clutch as stated; its top-alone
core is "a composite multiple `g m` with `m < g` is also a multiple of a smaller top gear"]**

**Not top-alone, though sometimes filed as wheels facts** (they are in section 3): the
placement residue law (C9), the fraction of home columns that are bottom-open (C9), the
twisted copies (C6), the four cells (C4), the survivor curve (C8), the switching identity
(C14).

---

# 3. THE CLUTCH - the two machines together

## 3.1 The kernel equivalence and the window

**C1. The route.** Twin primes are infinite iff for every bound some machine `{5..y}` has an
opening in the window `(y, y^2]`; inside the window an opening IS a twin pair; and the gap
form. KERNEL - `BlockedSlots.twins_infinite_iff_survivor_in_window`, `survivor_iff_twin`,
`survivor_in_window_of_gap_bound`, `Horizon.exists_prime_factor_lt`; docs/proofs/01. This is
the clutch statement on which the whole project rests.

**C2. THE WINDOW IS THE CLUTCH'S ZERO-INTERACTION REGION.** A proper strike of a top gear on a
bottom-open column has member `g m` with `m` `q`-rough and `m > 1`, hence the member exceeds
`q q' > q^2`, so the top machine cannot kill anything in the window. Measured: the first proper
kill of a bottom-open column is at column 28, 60, 60, 140, 140 against window tops 20, 28, 48,
60, 88 (`q = 11..23`), above the window at every `q`. The window is the ONLY region of the
period where one machine decides alone, and it is 2.4e-6 of the period at `q = 23`. PROOF +
EXACT, 5 machines, 0 exceptions - period_scale.md 3.3, section 5 item 2.

**C3. Twins = both-open + home-only, exactly.** 0 mismatches over 38,889,216 columns; the
home-only sub-cell IS the set of doubly occupied placements, i.e. a home strike marks "this
candidate is a twin whose member the top machine happens to own". EXACT - period_scale.md 3.2,
3.5.

**C4. THE FOUR CELLS, and where the coupling is.** At `q = 23`: both open 895,791;
bottom-open/top-closed 7,056,384; bottom-closed/top-open 4,150,311; both closed 25,079,659.
Coupling of the both-open cell 1.0197, 0.9455, 0.8659, 0.8451, 0.8300 at `q = 11..23` -
negative and strengthening; the other three cells are within 5% of independence at every `q`.
**All the coupling of the clutch is in the both-open cell**, and that deficit is the classical
`s = 2` handicap seen as a correlation between two machines. EXACT - period_scale.md 3.2.
Sub-split of the bottom-open/top-closed cell: home strikes only 3, 9, 26, 78, 268; at least one
proper strike 68, 1,019, 17,642, 321,225, 7,056,116.

**C5. Runs inside the cells.** Both cells that need the bottom machine open have longest run
exactly 2 at every `q` - gear 5 alone forces it. The both-closed cell reaches the bottom
machine's own record run at `q = 13, 17, 19` (10, 17, 24) and falls short at `q = 23` (29
against 33): from `q = 23` the top machine leaves an opening inside the bottom machine's record
stretch. EXACT - period_scale.md 3.2.

## 3.2 The layer, the square gate, and the walk

**C6. THE TWISTED COPIES.** A top gear `g` strikes a bottom-open column iff, writing the struck
member as `g m`, the cofactor `m` is `q`-rough and the partner `g m -+ 2` is `q`-rough; in the
`m` coordinate that is exactly the opening set of a TWISTED bottom machine with teeth
`{0, -+2 g^{-1}}` at every bottom gear. So the top machine's action on the bottom machine's
openings is a union of `2(pi(Z) - pi(q))` coherent twisted copies of the bottom machine at
separation `2/g`, each with exactly `prod (h-2)` openings per `m`-period. EXACT - 4,676 copies,
17,035,903 cofactors, 0 mismatches; period_scale.md 3.6.

**C7. Level of distribution 1, exactly.** `|X_g - 2N/g| < 2 * 3^m` at 2,338 (`q`, gear) cells;
`|X_{g,h} - 4N/(gh)| < 2 * 3^m` at 19,956 pairs with `gh < P`; triples the same; and the true
growth is `2^m`, not `3^m` (the inclusion-exclusion term-count bound is loose by `(3/2)^m`).
The deviation is bounded by a constant of the bottom machine on BOTH sides of `gh = P`; what
changes is the size of the prediction. EXACT, 0 exceptions - period_scale.md 3.7.

**C8. The survivor curve.** `S(z)` against the independent-gear prediction: 1.0000 at
`s >= 4.27`, 1.0132 at `s = 3`, minimum 0.8603 at `s = 2.09`, 0.8926 at `s = 2` (`q = 23`); the
position of the minimum is a function of `s` alone across machines whose periods differ by a
factor 23; the `s = 2` ratio approaches `4 e^{-2 gamma}` as `0.79305 (1 + c/ln Z)` with `c` flat
at 1.21-1.39. EXACT - period_scale.md 3.8. The dimension-2 lower function `f_2(s)` is
identically zero for `s <= 4.2664`, precisely where the measured ratio is 1.0000.

**C9. THE PLACEMENT RESIDUE LAW (exact, new).** As `r` runs over the residues mod `6h` coprime
to 6 and nonzero mod `h`, the home column `k = (r -+ 1)/6 mod h` takes each of the `h-2`
non-tooth classes of bottom gear `h` exactly TWICE and each of `h`'s two tooth classes exactly
ONCE. Hence the placement density is exactly `prod (1 - 1/(h-1))`: placement is a DIMENSION-1
event, double occupancy the dimension-2 one, and that step is the parity barrier, named in the
machine's own terms. PROOF (one line) + EXACT, 25 (machine, bottom gear) pairs, all residues, 0
exceptions - period_scale.md 3.5, 4.4. **Filed as a wheels fact in the owner's opening list;
its content is residues modulo BOTTOM gears - Appendix B.3.**

**C10. A top gear's home column is never struck by a bottom gear on the gear's own side** (the
member there is the prime `g`), 0 violations in 15,549 (gear, bottom gear) checks - which is
why the placement condition is one forbidden class per bottom gear. EXACT - period_scale.md
3.5.

**C11. THE ORIGIN LAW.** Both machines are open at column 0 and both are mirror-symmetric about
it, so the clutch classification is an even function of `k` (0 mismatches). Inside the
placement prefix every top gear sits on its own home column, so the top machine runs at 3% of
its own average density there (0.0040 against 0.1357 at `q = 23`); consequently, exceptionless
from `q = 19`, **the longest both-open stretch of the entire period starts at column 0** and
ends at the first twin both of whose members exceed `Z` (520 and 2,523 columns against
`ceil(Z/6) = 519` and 2,490). EXACT - period_scale.md 3.4.

**C12. THE TWIN-FREE RECORD IS A JOINT OBJECT.** The longest twin-free run is 24, 82, 153, 254,
501 columns at `q = 11..23` - 1.85, 2.41, 3.26, 3.10, 3.66 times the SUM of the two machines'
own longest closed runs; and inside it the bottom machine is closed at 1.212, 1.072, 1.021,
1.010, 1.005 of its average rate, monotone to 1. The record is made entirely by the top machine
covering the bottom machine's ordinary leftovers. EXACT - period_scale.md 3.11.

**C13. Nothing at the period scale gives the window.** Twin-free stretches of 502 columns exist
inside the period against a window of 83, and the maximum twin gap per block is flat across the
period (287..502 over twenty blocks). What saves the window is only that it sits in the first
2.4e-6 of the period. EXACT - period_scale.md 3.11, Dead ends.

**C14. The bilinear `g-m` switching.** `E = 2D + Q` exactly at all five `q` (ordered
prime-cofactor strikes = twice the unordered both-prime pairs plus the squares); the
prime-cofactor fraction matches the sieve prediction to 0.01% at `q = 23`; the `m = 1` count is
exactly the number of bottom-open placements at every `q`. Both sides count the same set and
every dyadic split is exact, so switching gives an IDENTITY, not an inequality. EXACT -
period_scale.md 3.10.

**C15. Brun on the clutch.** The order-2 truncation error equals the order-3 term at 100%,
94.1%, 89.4%, 82.2%, 77.7% (`z = 40..400`, `q = 23`); order 1 is negative by `z = 400`.
Exactness buys nothing: face A is not "the error terms are too big" but "the main terms
alternate and do not converge at `s = 2`". EXACT - period_scale.md 3.9.

**C16. The layer law.** A composite in the layer `(y^2, y'^2)` has a prime factor below `y` or
is `y c` with `c` prime; one gear's ledger line counts partner primes exactly below `q^3`.
KERNEL - `Layer.layer_novelty`, `minFac_lt_or_eq`, `eq_mul_prime_of_minFac_eq`, `slot_cap`,
`Gear.mem_partners`; docs/proofs/15. (Its motor half is M7.)

**C17. THE SQUARE GATE.** The deepest layer of the walk from `q^2` that hops is the top gear
itself IFF `q^2 - 2` is prime: 0 exceptions in 667 walks (153 open, 514 shut). EXACT -
self_feeding.md W2; register walk-tooth-frame. Also: the only provably droppable gear set in
the window is the square gate `g^2 > 6 top + 1`, exact at all 165 rungs but explaining 11 of
143 zero drops (deletion_profile.md); and every gear makes an exclusive kill in the window at
`Q = 997`, so no proper subset of gears determines the window's openings (7d).

**C18. The walk from `q^2` and the top gear's single strike.** The walk starts ON a tooth of
the top gear (`6 k_0 = q^2 - 1`, so `k_0 = -6^{-1} mod q`) and the top gear strikes the whole
walk exactly once, at its first column; its next strike is `d = 2c mod q` columns on, and the
walk length `L` stays below `d` at every `q` above 53 (one exception, `q = 53`; worst
`L/d = 0.52` at `q = 137`, median 0.02). The top gear is INERT on its own walk (the smallest
striker of no path column but offset 0, 0 exceptions), and the `q^2` column is the unique tooth
of `q` in its window where `q` is the sole striker (0 of 337,011). EXACT, 667 and 2,260 walks -
self_feeding.md W1, walk_path.md; registers walk-tooth-frame, walk-path-parts,
walk-path-transforms. **Belongs to two objects - Appendix B.2: the path is a motor object; the
landing being a TWIN is clutch, via C2.**

**C19. THE NEAR-TWINS: at most three per rung.** Gear `q`'s contribution to the new section is
at most three isolated near-twins, spaced at least `(q-1)/3` slots apart, with new twins between
them; over 666 rungs gear `q` bites 0 spots at 502 rungs, 1 at 146, 2 at 15, 3 at 3, never more,
and the bitten fraction is below 0.047 at every `q >= 300`; at every rung `q >= 300` there is a
new twin between any two consecutive kills of gear `q`. So the new gear cannot join two of the
lower sieve's gaps into a block: a permanent blocking condition inside a window can never come
from the new gear. MEASURED, exhaustively listed - docs/proof-search/lower-sieve.md sections 2
and 4.

**C20. THE ISLAND WITNESS.** For every integer coprime to 30 above 2849 some island of `[1, d)`
is open (`d = 2u_q`, the top gear's next tooth): 0 exceptions in 17,748 primes and 52,574
integers to 200,000; every multiple of 5 fails, by a proved relocation law; powers of 5, 49 and
121 are the only prime-power failures; one class `i = 12 (mod 35)` suffices from `q = 5477`; the
free island sits inside `[1, 0.152 d)` for every prime above 20,000 and its absolute offset
never exceeds 2,392; the minimum count of open islands per band rises 2, 4, 12, 21, 57, 107.
Read as numbers: `q^2 + 6i - 2` and `q^2 + 6i` are a twin prime pair for some `i = 12 mod 35`
below 2,392. EXACT - island_witness.md, reachability.md; registers reachability-landscape,
island-witness-integers.

**C21. The reachability landscape (the island set itself).** `|Bar(g)| = (g + 1 - chi_g(2) -
chi_g(-2))/4`, so no gear reaches every offset (gear 5 reaches only offsets 1, 3, 4 mod 5); the
islands for bound `B` are exactly `prod |Bar(g)|` CRT classes (4 classes mod 35 at `B = 7`,
namely `{5, 10, 12, 17}`; 12, 48, 192, 960, 5,760 at `B = 11..23`); the DOUBLING LAW (a gear
strikes an offset for exactly `2 chi_g(i)` residue classes of `q`, never an odd number, so its
mean rate over offsets is exactly `2/g` and exactly 0 on a quarter of them); large gears strike
islands at exactly `2/g` (0.9956 of predicted over 103,899 sightings). PROOF + EXACT -
reachability.md. **The landscape is `q`-free and defined by the motor's teeth: a motor object
with a clutch target.**

**C22. THE COVER NUMBER `K(d)`.** Exact at 23 arcs to `d = 1,330` (3, 4, 5, ..., 22), every
value ILP-certified, growth `d/(ln d)^3` with `K (ln d)^3/d = 6.15 +- 0.20`, against a counting
requirement 2..11 that stalls. PROVED, no counting: a cover with phases is realised by exactly
`2^K` residue classes of `q` modulo the product of its gears (324 million residues checked, 0
exceptions), and that product exceeds `q^2` at every `d >= 70`, so a failure PINS `q^2` as an
integer. Why it does not close: about `2.7^m` covers (10^54 at `d = 1,120`) against a class
density 10^-30 - vacuous by 10^24. PROOF + EXACT - cover_number.md.

**C23. The real separation does not drive `K`.** `K_real` is the MODE of the random-separation
distribution at all six arcs (189 draws, 239 certified ILP rows; percentiles 0.46-0.75) and
coherent separations `c/r` give the same `K` at every arc. Mechanism, exact: two gears' four
struck residues are a translate of `{0, S_g, S_h, S_g + S_h} mod gh`, so the mean pairwise
overlap is exactly `4m/(gh)` for EVERY separation (72 checks, 0 exceptions). EXACT -
separation_drives_K.md.

**C24. The square phase vector is irrelevant.** Real vectors `q^2 mod g`, independent
locally-square vectors and random vectors fail the island witness at 0.029653 against 0.029700
(ratio 0.9984 +- 0.0033 over 6.3 million vectors of each kind on 30 machines), and every
derived statistic agrees to within 3%; 82 explicit failing locally-square vectors at `d = 954`,
0 of 82 with a perfect-square CRT lift, against 21 of 21 real failures with `R = q^2` exactly.
EXACT - square_vector.md (face C3). DEAD END recorded: the phase vector being a square.

## 3.3 What the two machines do to each other inside the window

**C25. Each gear's in-window take is one curve.** What a later gear removes follows a curve in
`t = ln g / ln Q'` alone - 1.000 of fair share for `t < 0.55`, 0.957 at `t = 0.62`, 1.87 as
`t -> 1` - the same for every anchor, with white residual (`z` mean 0.02, sd 1.004, max 4.02
over 105,919 gear-rows). Mechanism: where the multiplier columns `m = (6k -+ 1)/g` sit relative
to `g^2`. And the curve belongs to the RANGE, not to any family: all families agree to 2% in
every bin, the island family sits on the anchor family's curve, and no gear takes less than
`2/g` on any family. EXACT - anchor_window.md (7b), structured_families.md.

**C26. The anchor's rigidity in the window.** The openings of `{5..13}` sorted modulo any higher
gear miss their fair share by fewer than 30 in every window to `Q = 5000` (proved from the
interval discrepancy of the 180 re-toothed anchors; real teeth 7.54, worst 14.09; measured
`<= 11.4` at `W = 4.2e6` columns). The rigidity is EXHAUSTED at the first gear above the
anchor: after it the survivors are the lower machine's pattern, not the anchor's. PROOF +
EXACT, 400,000 gear-rows - anchor_window.md (7b).

**C27. The structured-families identity.** Every located family carries twins at the window's
own rate: on 661 disjoint sections (130,644 twins) the normalised excess is within 1.2 sigma of
1.000 for every family, and the islands are indistinguishable from ordinary corridor columns
(1.006 +- 0.005). MECHANISM: the singular series of any non-tooth residue family is `12 C_2`
divided by the small-gear factor the family "saves", exactly cancelling it - **a rule written
in the lower machine's residues cancels its own saving**; `survivors(F)/survivors(window) =
density(F)/prod (1 - 2/g) <= 1` with equality only for the whole opening set. A family defined
by residues modulo the lower machine's period can never beat existence, because that period is
invertible modulo every gear above it; only a rule whose definition involves the gears ABOVE
could. PROOF + EXACT - structured_families.md (R2.e.ii).

**C28. The position-length frontier reduces the window statement to `d_0 <= W`.** From
`q = 1427` on, the longest blocked run of the whole prefix `[1, W]` IS the initial run from
column 1 (2,038 of 2,038 rungs, 0 exceptions); the Pareto staircase collapses to the single
point `(1, d_0 - 1)`, with `(d_0 - 1)/(q/6)` in `[0.97, 1.35]`, median 1.005. So the window
statement is `R_min(W) > 1`, i.e. exactly `d_0 <= W`: all of `[q/6, W]` is provably removed
from suspicion, and by theorem (E) **the top machine is irrelevant inside the window**. PROOF +
EXACT - position_frontier.md (R2.e.i). The location is the bottom.

**C29. THE WINDOW HAS AT MOST TWO JUNCTIONS, and we know which.** 0 mismatches at 152 rungs:
the column of `q'` (iff `q'` is a twin member, 28 rungs) and the column of `q'^2` (iff
`q'^2 - 2` is prime, the square gate, 42 rungs). At the bottom junction `L^- = round(q'/6) =
d_0(M)` at all 28 - the twin-Bertrand quantity is literally one flank of the window's own
bottom junction; at the top junction the flanks under `{5..q}` are the two-sided walk from
`q'^2` under `{5..q'}` (0 mismatches of 42). EXACT - flank_walk.md (R3.h.i).

**C30. `d_0`.** `d_0` is the column of the first twin pair above the top gear at every level to
33,317, `d_0 <= q'`, inside the window by a factor 10-58; the mirror forces `F_2 >= 2 d_0` and
nothing more, the slack growing to 8x at m53. EXACT - node 1e.i, anchor_runs_zero.md. It is a
motor quantity and a twin statement at once - Appendix B.7.

**C31. The machine feeds on itself.** The next level's walk starts at `6k^2 - 2k`, exactly `2k`
below the pair's twin-product column, and both newest gears strike it once at distance
`2k = (g+1)/3`; and the LEVEL-FREE TRANSFER RULE - a gear striking column `k + j` beside a birth
column strikes column `i` of that pair's own walk iff it divides `(6j)^2 + 6i - 2`,
`(6j)^2 + 6i`, `(6j+2)^2 + 6i - 2` or `(6j+2)^2 + 6i`, with neither `k` nor `g` in the condition
(832,915 checks, 0 mismatches; at `j = +-1`, `i = 0` the admissible gears are exactly
`{7, 17, 31}`, 3,093 carry-overs of 50,906). EXACT - self_feeding.md W3, W4; register
walk-tooth-frame. The chain of landings has no rule, as pre-registered.

**C32. What holds up a window stretch, against what holds up a period record.** The period
record needs EVERY gear (minimum blocking set = the whole machine, m7..m23) while the window's
longest stretch needs a chosen fifth (32 of 166 gears at rung 997); the window profile is
ordered by column position, not gear size; zero-drop gears are individually redundant but
jointly essential (removing all of them destroys the window stretch at 157 of 165 rungs); the
nested-decreasing holder law (for a fixed stretch the set of gears holding it up can only
shrink as the machine grows, one-line proof); gear 5 holds every window record at 164 of 165
rungs. The top gear NEVER removes a survivor from the window's longest stretch (0 of 160), that
stretch is a two-piece fusion at every rung, and a fusion of four or more by one gear occurs
nowhere in any window while the m23 record is one. EXACT - deletion_profile.md,
ends_or_middles.md.

**C33. `F_W` is the largest twin gap in `(q, q'^2)`.** The window's longest stretch is a
twin-gap statement, not a machine statement; the 295 rungs to 1999 carry only 11 distinct window
stretches (the maximal twin gaps), so per-rung counts are not independent samples; `L*` (the
longest blocked run STARTING in the window) is 24 columns from `q = 23` to 43 and 27 from 47,
while `F - 1` climbs 33 to 144. EXACT - 7d, record_frame.md, gear5_lock.md, deletion_profile.md.

## 3.4 Moments over `q`, and the transfer

**C34. First and second moment for the island witness.** FIRST: the exact phase-vector model
reproduces a real integer's opening count to 0.03% at `s = 3.2` and is 26% high at `s = 2`, the
object's own configuration (Hardy-Littlewood over real 1.0021 at `q = 50,000`; model over real
1.2628 against the classical `4 e^{-2 gamma} = 1.2619`); with that correction the predicted
failures below `q = 6000` become 16.51 against 17 observed. SECOND: DEAD BY PROOF - any bound
strictly below 1 on the fraction of failing `q` implies the twin prime conjecture, and the
first moment is a twin count in `(q^2, (q+1)^2)` at `s = 2`, a lower-bound sieve, so the chain
never starts; measured `B(X) = 11.4 (ln X)^2/X`, flat to 6% over a 64-fold range. KEPT, exact:
the gears coupling two islands at separation `delta` are exactly the prime factors above 7 of
`delta`, `3 delta - 1` and `3 delta + 1` (0 exceptions in 359,712,683 cells, the same divisor
form as `Leg(v)`); the count of open islands is SUB-POISSON, variance over mean in
`[0.76, 0.81]` at all 42 exactly computed `q`, with an exact mechanism. PROOF + EXACT -
square_vector.md, second_moment.md.

**C35. THE FACES OF THE WALL, SORTED BY OBJECT** (the_wall.md sections 2 and 5a-5k):

| face | what it says | object |
|---|---|---|
| A1 sieve dimension | class-count-only bounds have their lower function vanishing below `s = 4.27` while the window sits at `s = 2` | **NEITHER** - a property of the method; R4.a shows it survives at the period scale with faces B, D, E removed |
| A2 counting on the record | the strike budget `sum 2/g` always covers the record stretch; the overlap the teeth force is nearly achieved | **MOTOR** |
| A3 counting through islands | large gears strike islands at exactly `2/g`, so the margin through islands equals the unrestricted one and crosses 1 at `q = 53` | **MOTOR** object, **CLUTCH** target |
| A4 rate to maximum | every rate is exact, and "the maximum does not exceed what the rate suggests" is never available | **NEITHER** - it recurs in both machines |
| A covering side | the collapse lemma: once the modulus exceeds the interval each fibre holds one column and the second moment collapses to the union bound | **NEITHER** (method) |
| B1-B4 position cannot see length | corridor, gear-5 lock, slot rule, record phase pinning, the hinge; escape distance 1 | **MOTOR** |
| B2 the zero mirror | the region past zero is thinner than the period mean (0.79) | **MOTOR**, with a clutch reading: any statement about `(0, W]` from tooth positions is a statement about the twins below `Q'^2` |
| C1-C2, C4 typicality | symmetry group `Z/2`; coherent spacings at the 14th-22nd percentile; the walk a typical tooth start | **MOTOR** |
| C3 the square phase vector | real, locally-square and random vectors fail alike | **CLUTCH** (the vector is `q^2 mod g`) |
| C5 the sifting level | at `s = 2` the real machine has 21% fewer openings than any phase model | **CLUTCH** |
| D1-D3 transfer | rare among all phase vectors is not never for real `q`; equidistribution beyond Bombieri-Vinogradov | **CLUTCH** |
| E1-E4 over-asking | the per-step ladder contains twin-Bertrand; a dead section is a twin gap of order `4 sqrt x`; `L < d` and the arc witness are twins within `q/3` of `q^2` | **CLUTCH** |

**C36. What the sorted faces leave.** Conditions 1, 4 and 5 of the wall's section 3 name one
object, the adversarial covering number - which is a MOTOR-family statement (`K_columns(W(q)) >
pi(q) - 3` with the gear set fixed IS `F(y) < y^2/6`, the root in covering language, wall 5a as
corrected by 5d); the transfer and the over-asking faces are CLUTCH and are exactly what the
adversarial form does not need. the_wall.md sections 3, 5a, 5d, 5e.

---

# Appendix A. Vocabulary, three columns

| what | MOTOR term (`{5..q}`) | WHEELS term (primes `> q`, raw line) | CLUTCH term |
|---|---|---|---|
| the unit | column `k = (6k-1, 6k+1)` | the pair `(n, n+2)` | the 6-fold map of the raw line into columns |
| a gear | prime `5 <= g <= q`, teeth at `+-6^{-1} (mod g)` | prime `g in (q, Z]` striking its multiples; in pair coordinates teeth at `0` and `-2 (mod g)` | a top gear meeting a bottom-open column |
| separation | `d_g = 3^{-1} (mod g)`, "one third", the same rational at every gear | `2` | the one third IS the top's 2 seen through the 6-fold (B.1) |
| anchor | 2, 3, 5 as one object, cycle 30 | the wheel of the smallest top gears `q' q'' q'''`, `(q'-2)(q''-2)(q'''-2)` slots per turn | - |
| origin | column 0 always open; mirror `k -> -k` | the pair `(-1, +1)` straddling every wheel multiple; mirror `n -> -n - 2` | the shared origin: open in both, clutch state an even function of `k` |
| a strike | a strike is a kill | a strike on the HOME column `h(g) = round(g/6)` is a PLACEMENT, not a kill; proper teeth are the other multiples | home strike = a twin with a small member; proper strike = a real kill |
| an opening | a column no gear of `{5..q}` strikes = a twin candidate | a column no prime in `(q, Z]` strikes properly | both open = twin |
| period | `P = prod g`; `N = prod (g-2)` openings per period | its period is far beyond the range: the in-range pattern is not periodic at all | the clutch's period is the primorial to `Z`; within `[0, P)` it never repeats, and its only exact self-similarity is the mirror |
| effective machine | `{5..sqrt(6k+1)}` at column `k` (theorem E) | `{q'..sqrt(6K)}` on a range of `K` columns (the zones) | non-repeating gears are redundant; silent gears only place |
| letters | `{2u', q' - 2u'}` plus padding `q'` | `{2, g-2}` (the construction rule) | - |
| the record | `F(M)`, the longest opening-free stretch (`F/W = 0.25` flat) | the longest closed run above the placement prefix: 7, 24, 30, 58, 104 | the longest twin-free run: 24, 82, 153, 254, 501, i.e. 1.85-3.66x the sum |
| window | `(y, y^2]`, the certified range | - | the zero-interaction region |
| section | `(p^2, q^2)`, the window's new part | - | where the new gear's near-twins live (at most three) |
| dimension | two teeth per gear: dimension 2 | - | placement is dimension 1, double occupancy dimension 2; that step is the parity barrier |

---

# Appendix B. Facts the re-filing found misfiled, or stated in the wrong object's coordinate

**B.1 The "one-third separation".** Filed throughout as the motor's defining arithmetic (node
6, separability.md's "the one-third separation maximises sharing", collision_laws.md, the
wall's W3). Per the construction rule it is the WHEELS' separation 2 seen through the bottom's
6-fold: a clutch fact wearing motor coordinates. The record already met it from the other side
without naming it - pinned_arithmetic.md's "the real-teeth input is ONE coordinate: in
`n = 6k` every gear's teeth sit at `+-1`", which that file itself flags as "the owner's
raw-line view of the top machine, met from the bottom's side". Re-filed: motor entry M3 with a
clutch cross-reference.

**B.2 The walk from `q^2`.** Named in the owner's opening list as a CLUTCH fact. In the R4 split
gear `q` belongs to the BOTTOM machine, so the path itself - the anchor slot, the gear-5
offsets, the quadratic-residue bar, the reachability landscape, the depth profile, the length
`L` - is entirely MOTOR. Only the landing being a TWIN is clutch, and it is clutch by C2 (the
first proper kill lies above). Splitting it that way is what makes the island witness legible:
a `q`-free motor object (C21) with a clutch target (C20).

**B.3 The placement residue law.** Named in the owner's opening list as a WHEELS fact. Its
content is the distribution of home columns modulo the BOTTOM gears (exactly 2 : 1, non-tooth
against tooth classes), so it is a clutch fact; the wheels-alone residue is only
`h(g) = round(g/6)` and that every placement lies in `[1, ceil(Z/6)]`. Re-filed as C9, with W9
keeping the geometry.

**B.4 The top machine's longest closed run.** First recorded as 7, 27, 114, 378, 1,376. The
period's longest sits INSIDE the placement prefix and is made of home strikes - i.e. of twins -
so it is a clutch artefact, not a wheels fact. The honest wheels number is the run above the
prefix, 7, 24, 30, 58, 104. Corrected already in period_scale.md's Dead ends; re-filed here as
W8 so the wheels' own record is not quoted with the artefact.

**B.5 Three different objects share the word "record".** The motor's `F` (`F/W = 0.25` flat at
`y = 7..53`, the wall's factor of four); the wheels' longest closed run above the prefix; and
the clutch's twin-free run (502 columns at `q = 23` against a window of 83). These are not
comparable, and the wall quotes the first and period_scale.md the third within a few lines of
each other. The clutch statement "the record is 1.85-3.66x the SUM of the two machines' own
records" is about the third object only.

**B.6 The coherent family `c/r` at separation `2/g`.** Introduced in separation_drives_K.md
(weak point W3) as a COUNTERFACTUAL family against which the real motor teeth were tested - a
motor-family object. period_scale.md 3.6 shows it is the top machine's natural coordinate on
the bottom machine's openings (4,676 copies, 17 million cofactors, 0 mismatches): the
"counterfactual family" was the clutch's own coordinate all along. Re-filed as C6, with the
motor reading kept in C23.

**B.7 `d_0`.** Carried in two coordinates at once: as the motor's first opening above the top
gear (the pair statement's `F_2 >= 2 d_0`, node 1e) and as "the column of the first twin pair
above `q`" (node 1e.i, prover A's verdict, the wall's E1). The two readings are the same number
only because the window is the clutch's zero-interaction region (C2); the identification is a
clutch fact, not a definition, and the twin-Bertrand strength of the per-step ladder rests on
it.

**B.8 Face A is not a face of the machine.** It is recorded on the wall beside faces B, C, D
and E as though all five were properties of the object. R4.a settles it: at the period scale,
where every moment is exact and faces B, D and E are absent, face A stands unchanged - the main
terms alternate and do not converge at `s = 2`. Face A (both its sieve and its covering side)
is a property of the METHOD; faces B and C are motor; faces D and E are clutch. Re-filed as the
table in C35.

**B.9 "Near-twins, at most three per rung".** Named as a clutch fact, and it is one - but it is
stated in the motor's coordinate (slots of the section) about the action of the machine's OWN
new top gear at a rung, not about the primes above `q`. It is a LADDER fact (`q` acting on the
section it opens), which is why it lives in docs/proof-search/lower-sieve.md and not in
period_scale.md. Kept as C19 with that reading made explicit.
