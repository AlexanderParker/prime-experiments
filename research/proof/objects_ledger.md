# The objects ledger (harvester, 2026-09-06)

The owner's rule of 2026-09-06: no clutch work until the motor, the wheels and the exhaust are
fully understood, and this ledger is the gate. Nothing here is new computation. Every line is a
fact already on the record, moved under the object it is about and cited by document; where a
document's own words state the law, those words are used.

**The three objects, in the owner's vocabulary.** The MOTOR is the bottom machine `{2..q}`, or
the anchored `{5..q}`: columns, gears, teeth, period, record. The WHEELS are the top machine,
the primes in `(q, q#]` on the raw line, written in their own pair coordinate and never in the
motor's. The EXHAUST is every tier above the wheels together: the stack, tier `k + 1` being the
primes from the top of tier `k` to tier `k`'s period, and the boundaries between them the CUTS.
The CLUTCH is the interaction of motor and wheels, and it is not worked here.

**Status vocabulary** (the tree's evidence standards). `KERNEL` = a Lean theorem in `proofs/`,
zero sorries, standard axioms, with its hypothesis stated. `PROOF` = a written proof in
`docs/proofs/` or a branch document. `EXACT` = full periods, exhaustive families, phase
reduction, CRT / LP / ILP certificates, with the count. `MEASURED` = holds over a stated range
with a stated count of exceptions, no mechanism or no completeness. `TARGET` = wanted, never
proved.

**Sources.** `research/proof/theory_tree.md` (whole tree, all verdicts); `the_wall.md`;
`docs/proof-search/human.md`; `top_machine_1.md` .. `top_machine_6.md`; `top_machine_lean.md`;
`docs/proofs/README.md` (22 written proofs); `docs/novel/README.md`; `refiled_by_object.md`
(the earlier refiling, 74 / 12 / 36); `position_frontier.md`; `anchor_window.md`.

**A numbering warning, carried from the record.** The top-machine documents restart their law
numbering: document 1 reaches L21, document 2 runs L22-L38, **document 3 runs L30-L45 and
therefore clashes with document 2 at L30-L38**, document 4 runs L46-L56, document 5 opens again
at L50 and runs to L59, document 6 runs L57-L66. Every citation below names the document.

**Two lanes are running as this is written. Their results are pending and are not counted
anywhere in this ledger.** Marked slots:

> ### SLOT A - PENDING: the exhaust in the kernel (Formalist lane)
> `proofs/MachineStack.lean` and `docs/proofs/23`. The file exists and carries declarations for
> tiers and cuts, stride containment, non-containment, the exhaust cap (home strike or echo),
> the quiet-zone clutch statement, the stack prefix as the primes up to a cut, and the wheels'
> smooth-zone and quiet-zone laws. **No lane report, no build gate and no axiom audit has been
> filed, so nothing from that file is entered as KERNEL below.** When the lane lands, its
> theorems belong in EXHAUST section 2 (structure and location) and in WHEELS section 2
> (counting: the zone laws), and the EXHAUST gate list shortens by its first item.

> ### SLOT B - LANDED 2026-09-06: the wheels' open laws (research/proof/top_machine_7.md, L67-L75)
> The loaded record rule PROVED both directions with no side hypothesis (L69: F_top = max{L :
> min over core phasings of the domino cost <= #{g > L + 1}}; 0 mismatches on 6,659 gear sets),
> so `L31` is a formula (O-W2 closed); the parity law and its sharp threshold derived as
> corollaries; the moment vanishing PROVED (L73) with r(d) = D(d - 1) (L74) and the d = 4
> exception absorbed (O-W1 closed); L18's multiplicities reproduced; the census law written in
> kernel shape (three lemmas, each verified). Remaining on the wheels alone: the complexity of
> min_U D_L(U); non-cancellation of M_{r(d)} (L75, 0 exceptions of 24, unproved); L22 in the
> kernel (round 36 did L67-L69 and the boundary corollary: proofs/TopMachineRecord.lean; round 37
> did L22 / W22: proofs/TopMachineCensus.lean, gears positive and coprime only; L70 and L73/L74 not
> attempted); a parity-refined capacity bound.

---

# MOTOR - the bottom machine `{5..q}` alone

## 1. DEFINITION

The motor is the machine on COLUMNS. Column `k` is the pair `(6k - 1, 6k + 1)`; the anchor 2, 3,
5 is one object of cycle 30 folded into the six-fold ruler, and gear 5 is the first gear that
turns. A gear is a prime `5 <= g <= q`; it has two TEETH, striking column `k` iff
`k = +-6^{-1} (mod g)`, so its struck residues are two classes at separation
`d_g = 2u_g = 3^{-1} (mod g)` - the same rational one third at every gear - leaving two ARCS of
lengths `2u_g - 1` and `g - 2u_g - 1` and `g - 2` open residues. The machine's PERIOD is
`P = prod g` columns with exactly `N = prod (g - 2)` openings per period (CRT); an OPENING is a
column no gear strikes; the RECORD `F(M)` is the longest opening-free stretch; the SPECTRUM is
the multiset of gap sizes; `F_2(M)` is the longest one-hole stretch. Adding a gear `q'` makes
`q'` copies of the period, each opening dying in exactly two of them, with LETTERS
`{2u', q' - 2u'}` and padding `q'`.

**Regimes.** The ANCHOR is 2, 3, 5 as one object, and by the anchor rescaling law (document 5
L56) putting it back multiplies a record by 6 and adds 5: `F(G + {2,3}) = 6 F_col(G) + 5`. The
WINDOW is the certified range `(y, y^2]`, never a sliding run; its new part is the SECTION
`(p^2, q^2)`; a sliding run is a STRETCH. The LADDER is the certified sequence of records
`F = 2, 5, 7, 11, 18, 25, 34, 43, 58` and, past the scan wall, `F(37) = 88`, `F(41) = 91`. Inside
the window the motor is EFFECTIVE only up to the square root: by theorem (E) a column with
`6k - 1 > q` is blocked under `{5..q}` iff blocked under `{5..floor(sqrt(6k + 1))}`.

## 2. PROVED

### Structure (teeth, dominoes, arcs, symmetry)

| law | statement | status | evidence |
|---|---|---|---|
| M1 the column and the fold | column `k = (6k-1, 6k+1)`; the anchor is one object of cycle 30 | KERNEL (definitional) | docs/proofs/02 |
| M2 the tooth rule | gear `q` strikes `k` iff `k = +-6^{-1} (mod q)`; two arcs; alternating spacings `2u`, `q - 2u`; the shield; teeth never adjacent; twin gears share the tooth `(p+1)/6` and both strike the column of `p(p+2)` | KERNEL | `TwoTeeth.kill_spacing`, `kill_period`, `teeth_letters`; `AnchorChain.neighbour_of_hit`; `Polignac.twin_product_slot`, `twin_split_class_iff`, `own_slot_pin_gap_two`; docs/proofs/02; re-verified 493,101,490 checks |
| M3 one rational at every gear | `3 d_g = 1 (mod g)`; in the coordinate `n = 6k` every gear's teeth sit at `+-1` | one line from `u_g = 6^{-1}`; the record marks it EXACT, 1,200 of 1,200 cells | pinned_arithmetic.md |
| M5 arcs, and why twin gears share one | `3 a_g = g -+ 1`, so two gears share an arc iff they are a twin prime pair; `{5..q}` carries only `pi(q) - 2 - pi_2(q)` distinct arcs | PROOF | gear_count.md, half_column.md |
| M8 always-open columns and the mirror | column 0 and the antipode `(P +- 1)/2` are always open; the mirror `k -> -k`; the symmetry group of the opening set is exactly `Z/2` | KERNEL | `Mirror.mirror_gear`, `antipode_open`, `self_mirror_unique`; docs/proofs/03 |
| M10 the mirror lever | the full affine group is `(Z/2)^m` but only `c = +-1 (mod P)` preserves adjacency; there is no mod-4 version | PROOF | mirror-parity-laws.md section 7 |
| M11 the mirror on the legal family | the self-mirror depth-`J` window is never word-legal for `J >= 3`; `J = 2` needs exactly `d_0 != F` | PROOF, gated m11..m23, 185 assertions | mirror-parity-laws.md section 9 |
| M14 the alignment law | the longest run of consecutive openings is the long arc `q_0 - 2u_0 - 1` of the smallest gear; with gear 5 openings are points and dominoes, `prod(q-4)` dominoes | PROOF (written CRT) | docs/proofs/04 |
| M15 the all-teeth column | the blocked run through a column struck by every gear is always exactly 1 | PROOF (one line) | tree log 2026-09-06 |
| M17 adding a gear | the `q'` copies realise every deletion phase once, each opening dies in exactly two; hit law; CHAIN LAW (`y - x = 0, +-d`); MERGE LAW; the grammar T1-T5 | KERNEL | `AnchorChain.copy_phase`, `phase_bijective`, `chain_law`, `hop_zero`; `MergeLaw.interior_gap_mod`, `newgap_le_step`; `TwoTeeth.kills_gap_ge`, `fuel_span_cap`; `WordLegal.legal_iff_noRepeat`, `killable_iff`; docs/proofs/05 |
| M18 deletion spacing | merge deletions are `>= q - 1` apart, and tight | PROOF | docs/novel/deletion-spacing |
| M19 the letters | an added gear's kill spacings are `{2u', q' - 2u'}` with exact `q'` padding, strictly alternating, minimum `2u'` | KERNEL (T1-T5) | `proofs/TwoTeeth.lean`, `MergeLaw.lean`; docs/proofs/02 |
| M6 theorem (E) | for `6k - 1 > q`: blocked under `{5..q}` iff blocked under `{5..floor(sqrt(6k+1))}`; the exception set is exactly the twin gear pairs on their own home columns, all below `(q+1)/6` | PROOF (one line) + EXACT (7 of 7 count matches) | position_frontier.md |
| M7 the square gate, motor half | a gear exposes nothing below its own square | KERNEL | `Gear.R_eq_zero_of_below_sq`, `Layer.slot_cap`; docs/proofs/15 |

### Counting (wheel counts, census, correlation)

| law | statement | status | evidence |
|---|---|---|---|
| M4 period and opening count | `P = prod g`; exactly `N = prod (g - 2)` openings per period | PROOF (CRT) | period_scale.md section 2 |
| M9 parity of counts | every gap length `>= 2` occurs an even number of times, so the record never occurs exactly once; window counts are even | KERNEL | `Mirror.even_card_involution`, `window_count_even`, `adjacent_equal_even`, `none_of_at_most_one`; docs/proofs/03 |
| M20 the word reduction | `Q*_J > -inf` iff `L(M) >= J - 2`, so `J_max = L + 2`, `A_kill = L + 1`; chain iff legal word; the same-tooth lemma | KERNEL | `WordLegal.chain_iff_word`, `qstar_iff_word`, `jmax`, `akill`, `same_tooth`, `same_tooth_window`, `literal_even_span`; docs/proofs/10 |
| M21 the bare-word cap | `L_bare(M) <= PSORD(q' mod 210) <= 5`; `PSORD in {1,2,3,5}`, the 28-class set `S`, `PSORD = 4` empty | KERNEL | `BareAlt.no_bare_run_ge`, `bareAlt_inadmissible_iff`, `S_card`, `psord_le_five`, `psord_ne_four`; docs/proofs/12 |
| M22 literal and Polignac caps | a literal chain has at most `capC(q mod 210) <= 6` exposed members, no class of cap 5; over all even gaps the cap depends on `gcd(e, 105)` and is `<= 12` | KERNEL | `LiteralCap.*`, `LiteralCapTable.*`, `PolignacCap.capOf_le_twelve`; docs/proofs/13 |
| M23 the only bound on `L` | `L(M) <= 2 floor((F(M+q') - 2)/q') + 1 <= 2 F(M+q')/q' + 1` | PROOF (written) | docs/proofs/11 |
| M25 phase saturation | a gap word whose exposed offsets leave some gear with no admissible phase cannot occur; the content sits at gears 5, 7, 11, giving a closed-form per-step ceiling (6, 2, 2, 2, 5, 3, 3, 4 at 31->37 .. 61->67) | EXACT | docs/novel/phase-saturation-arity, uniform-order-bound |
| M26 peel, triple, middle-sum | peel `Q*_J <= Q*_{J-1} + min flank`; the triple inequality `g_L + w + g_R <= F_2 + min(g_L, g_R)`, hypothesis-free; the middle-sum lemma; even-`J` literal runs are never palindromes | PROOF (written; the recorded per-`J` flank envelope's conditional step is flagged in the file) | docs/proofs/16 |
| M28 the exposure cap | exposure at word length `m` is decided by the gears `<= 2m + 2` alone | THEOREM + EXACT m11..m53 | docs/novel/cover-half-counter-ladder |
| M31 the deletion ladder | `F_{r+1}(M) <= F(M + r new gears)`, in particular `F_2(M) <= F(M+q')` | PROOF (written CRT) | docs/proofs/07 |
| M32 saturation | if `F(M) < 2u_q` (in particular `3F(M) < q - 1`) then `F(M+q) = F_2(M)` exactly | PROOF | docs/proofs/06 |
| M35 the fusion-rate identity | an old gap is fused in exactly 4, 3 or 2 of the `q'` copies and is an interior piece in 0, 1 or 2: junction availability never collapses, only piece size does | PROOF (from docs/proofs/05), 137 cells | frontier_collapse.md |
| M50 the branching identity | `n_J = C_{J-1} - 2C_J + C_{J+1}` with `C_0 = q' N`; `C_r = W_{r-1} + Z_{r-1}`; `max order = L + 2 = J_max`; the second moment in closed form; the size side; closure at depth `K_m <= m J_max` | PROOF | branching_identity.md, merge_forest.md |
| M51 what is teeth-free in the merge | the mean merge order is exactly `q'/(q' - 2)`; the teeth live in the variance and nowhere lower | PROOF + EXACT (8 real rungs, 42 family member-rungs) | merge_forest.md, branching_identity.md |
| M59 uncoupled sizes, classified | for `v < y^2/3`, `v` is uncoupled in `{5..y}` iff `v` is `y`-rough and its half-column is a twin column above `y` | PROOF + EXACT, 5,505 of 5,505 cells | half_column.md |
| M60 the half-column map | both letters of a gear point at its home column; `Leg(v) = {g : g | 3v-1 or 3v+1}`; the FIBRE THEOREM; the FIXED-POINT THEOREM (a column is a fixed point of halving iff it is a twin column) | PROOF | half_column.md, separability.md |
| M72 `A(K)`, the free-gear-set adversary | `A(K) = 2, 5, 7, 16, 22, 28` at `K = 1..6` by reasoning; no `K <= 10` primes at separation `3^{-1}` with any phase cover `(p_{K+1}^2-1)/6` columns | PROOF (K <= 6 by reasoning) + certified infeasibility (K <= 10) | docs/proofs/20, small_K_theorem.md |
| M73 collision laws | linear deficit `c(g,h;L+gh) = c(g,h;L) + 4`; the SHARED-ARC LAW (twin gears collide at `(g+4)/3`, the earliest possible); the arc floor; the head collision; the block bound and the block-size ladder | PROOF + certificates | docs/proofs/21, collision_laws.md |

### Metric (runs, chains, records, walks)

| law | statement | status | evidence |
|---|---|---|---|
| M29 the record law | `F(M+g)` is the max over `g` phases of the largest gap of the phase-`r` sequence on ONE lower period; the nested next-opening formula past a run of `k` hits | KERNEL | `AnchorRecord17.surv_shift`, `record_max`, `F17_eq_18`; `AnchorChain.hop_iter`, `hop_zero`, `hop_one`; docs/proofs/09 |
| M30 the attainment identity | `F(M+q') = max(F_2(M), max_{J>=3} Q*_J(M; q'))` | partly KERNEL, the identity written | `WordLegal.killable_iff`, `chain_iff_word`, `MergeLaw.newgap_le_step`, `AnchorChain.phase_bijective`; docs/proofs/08 |
| M39 the spare-gear lemma | if a 2-run has a gear neither obstructed at the middle opening nor a sole striker inside the run then `F(M) >= a + v`; contrapositive: `E(v) > 0` is exactly "no free gear" | PROOF, 0 counterexamples in 13,616 runs | pinned_letter.md |
| M40 L4 | every gear is a sole striker in any above-record stretch, teeth-free, in both worlds, with the single-gear re-phasing certificate | PROOF | docs/proofs/19, pair_statement.md |
| M41 the gear-5 lock | every maximal blocked stretch of every machine, at every length, has gear 5 at its coverage-maximal phase | PROOF (five cases) + exhaustive to `L = 2000`, 62 records, 1.7 million window stretches | gear5_lock.md |
| M46 the junction theorem | the junction condition is a congruence mod `q'` and the old machine is periodic mod `P` coprime to `q'`, so a junction is an ORDINARY opening and the maximum flank sum at junctions IS `F_2(M)` | PROOF | flank_walk.md |
| M47 L6 made exact | the left tiling is the negated right tiling gear by gear, equal iff `g | x`; `b_g^+ + b_g^- = a_g` or `g - a_g` at every opening and gear | PROOF; 0 exceptions in 10.3 million pairs (2.39 million across a gap) | docs/proofs/19, flank_walk.md, neighbour_profile.md |
| M36 the availability gate | `J = 3` needs a legal `a` or `hasM(a) > 0`; `J >= 4` needs `hasM(a) > 0`; the gate is one row of the level-2 dictionary, with `a_hasM <= F_2(M) - a_L` proved in one line | PROOF + EXACT, 137 cells, 68 of 68 | availability_gate.md |
| M37 the short-letter row | the row `(a, a_L)` of the adjacent-pair dictionary is empty above its realised top - proved scan-free by LP duality at m19 (above 20), m23 (25), m29 (35) | PROOF (LP duality) | short_letter_row.md |
| M54 the certified rungs | the increment law at the six literal steps 11->13 .. 29->31; the case split 31->37 giving `F(37) <= 95`, 385 exhaustive held-phase cases each an exact integer dual certificate | KERNEL | `Increment.increment_law_literal_steps`, `IncCert23/29/31.F_le`, `CaseCert37.F_le`; docs/proofs/17, 18 |

### Location (where records sit)

| law | statement | status | evidence |
|---|---|---|---|
| M12 the mirror at column 0 | the mirror gives the pair `(d_0, d_0)`, hence `F_2 >= 2 d_0` | THEOREM (deletion ladder plus mirror) | pair_statement.md L3, docs/proofs/19 |
| M13 the frontier mirror | `R_max = P - R_min - L + 1` | EXACT, 88 of 88 | position_frontier.md |
| M53 the position-length frontier | `R_min(L) >= ceil((y_L^2 - 1)/6) - L + 1`, an induction on the machine, unconditional for stretches whose top member is below `59^2` and delivering `c = 1.25` (1.54 for `L >= 6`) from the certified ladder | PROOF | position_frontier.md |
| M45 records are made at the ends | a record is a row of ORDINARY lower gaps whose junctions the top three gears strike (m29 `43 = 10+10+23`; m31 `58 = 23+10+25`; m23 `34 = 4+8+15+7`); `F = flank + letters + flank` | EXACT, 8 rungs | ends_or_middles.md, record_2run.md |
| M24 the corridor | `E_35` (15 residues); endpoint and adjacency laws (294 forbidden pairs); tier A carriers; the completeness lemma `q <= 2n`; the 32-cap; the adjacent-gap exclusion law mod 5; the AP lemma; padding onset and the 12 forbidden equal-padding classes | partly KERNEL | `Corridor.exposed_iff_mem`, `forbidden_pairs_count`, `prime_adjacent_run_le`, `TierA.*`; docs/proofs/14 |
| M42 the slot rule | `F = 1 (mod 5)` starts on slot 11\|13, `F = 4` on 17\|19, `F = 2` or `3` on a mirror pair, `F = 0` on any | EXACT at all eight full periods to m31 | anchor_cycles.md |
| M43 the allocation law at records | every gear of a record is at its coverage maximum subject to keeping the columns only it strikes | EXACT, 340 of 348 gear-cells over 62 records | gear5_lock.md |

### Transforms (spectral, bitwise, characters)

| law | statement | status | evidence |
|---|---|---|---|
| the machine DFT | the machine's transform is closed-form and real; gear 5's local frequency mode is `phi` and `phi/3` is a machine-independent spectral gap; the T3 law `3u = (q+1)/2` | PROOF + SCRIPT-VERIFIED; **prior art not yet checked** | docs/novel/golden-spectral-gap |
| M68 the `L1` character bound | `sum_m |Shat|/P = prod S_q/q` is independent of the teeth - identical at all 30/180/1440 counterfactual tooth vectors while `F` spreads 1.83x-2.50x | PROOF | docs/novel/walk-transform-pole-identity |
| the matrix formulation | the laws as one operating linear algebra; `charpoly(C_5) = (x-3)(x^2-x-1)^2` exact golden gap | SCRIPT-VERIFIED; prior-art checked per piece (CRT/Kronecker frame KNOWN; nilpotency-as-longest-run KNOWN technique, Jacobsthal application NOVEL*) | docs/novel/matrix-formulation |

## 3. MEASURED, NO PROOF

Each holds with 0 exceptions over the stated range unless the count says otherwise.

| # | statement | range and count | reading |
|---|---|---|---|
| 1 | `N(v) <= F_2(M)` for every realised gap size `v >= 6` | full periods to m31, 6.4 billion gaps, 0 exceptions, tight once (`N(7) = 55 = F_2` at m29) | **believed structural**; the GLUE LEMMA is proved as its mechanism, the law itself is not (neighbour_profile.md) |
| 2 | the J-run outer law `g_1 + g_J <= F_2` when every middle is `>= 6` | 3,278,972 runs, `J = 3..8`, m13..m23, 0 exceptions | believed structural; drop the middle condition and it breaks at once (glue_covering.md) |
| 3 | the pinned letter, upper half: `a_L + r(a_L) <= F(M) + 3` | 9 of 9 rungs, slack 3, 1, 3, 0, 2, 1, 3, 14 | **REAL-TEETH, not structural**: only 43 of 63 tooth-counterfactual members obey it, and every step of the glue construction is tooth-invariant, so no glue argument can prove the constant. Its LOWER half is REFUTED out of sample at 37->41 (`a_L + r = 77` against `F = 88`) (pinned_letter.md, pinned_arithmetic.md) |
| 4 | the frontier constant `R_min(L) >= 3.25 L` for `L >= d_0` | 113 period cells (m7..m29) and 8,375 window cells (`q = 23..19,997`), 0 exceptions | the PROVED constant is 1.25; the measured one is 3.25, and it is real-teeth (85th-90th percentile of its family) (position_frontier.md) |
| 5 | the initial run is the longest blocked run of `[1, W]` | 2,038 of 2,038 rungs, `q = 1427..19,997` | **the conjecture in disguise**: it makes the window statement exactly `d_0 <= W`, the first twin above `q` (position_frontier.md) |
| 6 | `d_0` is the column of the first twin pair above `q`, `d_0 <= q'`, inside the window by 10-58x | to level 33,317 | **the conjecture in disguise** (twin-Bertrand at scale `q`) (node 1e.i) |
| 7 | record saturation: every gear of `M` is the sole striker of a column INSIDE the record gap | 68 record occurrences of eight machines, exhaustive | structural; stronger than the spare-gear lemma (record_2run.md) |
| 8 | the record set is pinned: `F(M \ g) < F(M)` for every `g`; the minimum blocking set of the period record is the whole machine; from m23 the record is one residue class mod the period up to mirror | m7..m23 (blocking set), m7..m31 (record set sizes 2, 4, 12, 20, 20, 4, 2, 4) | structural; says WHERE the record is, not that an opening is forced into the window (node 5d) |
| 9 | record isolation: no 41, 42 below 43 at m29; no 56, 57 below 58 at m31; at m37, 89 and 90 certified empty with 13 certified holes | 3 machines | no mechanism (gear5_lock.md, record_2run.md) |
| 10 | every spectrum hole is a phase hole, never a span hole | 7 of 7 | no mechanism (spectrum_sum_rule.md) |
| 11 | the spectrum recursion `m_{M+q'}(v) = c_{q'}(v) m_M(v) + Merge(v)` | survival exact at 137 of 137 cells; reproduces the m31 spectrum | EXACT; the one-residue case is prior art (paired-holt-recursion) |
| 12 | the frontier `F(M+q') = max_a (a + Rest(a))`; the top law `Rest(F_old)` identity; the top slack 4, 9, 10, 10, 12, 18, 24, 29 | 8 rungs, 0 exceptions | EXACT; `Rest(a) <= q'` would give the budget in one line and FAILS at 29->31 (max rest 34 > 31) (merge_forest.md, frontier_collapse.md) |
| 13 | the top of the spectrum is pinned to `F_2`, not `F`; `max_a (a + Rest_2(a)) = F_2(M)` is an identity | 7 machines | EXACT (record_2run.md) |
| 14 | adjacency repulsion / the suppression law: gaps next to a large gap are shorter than independence gives | `F_2` 11..39 against shuffled 12..55, m11..m23; `n1` below the rarity null at 34 of 35 top-band cells | structural in 95% of family members; the rigorous side is the renewal ladder, and what stays heuristic is the rate-to-maximum step (node 5b) |
| 15 | three gear bands at a flank: `g - a_g < S + 2` strike at 100.00%; the middle band at `0.796 +- 0.004`, constant over `q = 59..997`; the top band falls 0.36 -> 0.18 | 2.28 million cells | the flank's length is decided in the middle band; no mechanism (flank_walk.md) |
| 16 | the flanks are coupled by the anchor's residue classes (`L^+ = 1 mod 5` forces `L^- in {0,2,4}`, and so on; 931 of 1,225 pair classes mod 35 admissible) | 8.8 million openings, 0 exceptions | not by the negation lemma; no mechanism (flank_walk.md) |
| 17 | the record is not anchored at the alignment points | m11..m23, distance to the nearest all-teeth column at the random median | thin place 3, DEAD in its sharp reading |
| 18 | the legal-word length mechanism: `L_g(M)` predicted to within one unit by an independent-letter model with the real class densities | measured | no mechanism for the collapse of the legal-window COUNT at the top (docs/novel/legal-word-length-mechanism) |
| 19 | twin gears help the record: de-twinning LOWERS `F` at every rung (1.10, 1.29, 1.47, 1.70, 1.48, 1.66 at m13..m31); mean `F` at fixed gear count increases in the number of duplicated arcs | 5,383 sets, no exception | EXACT, no mechanism (arc_multiset.md) |
| 20 | the real teeth are atypical in gluability: 62.5% against a pooled 9.4%, the 99.6th percentile of 223 comparable m19 members; at matched `(v, slack)` cells the factor shrinks from 6.6 to about 2.4 | m13..m31 | face C's only exception; a which-residues property, not followed (glue_covering.md, separability.md) |
| 21 | where the real machine sits in its family: `F` at the 17th-26th percentile of the exhaustive tooth-counterfactual distribution, strengthening with depth (m23: `F` 11.9%, `F_2` 3.1%); `F` at the 14th-22nd percentile of random symmetric spacings | m11..m23 | coherence explains nothing (face C2) |
| 22 | the increment law is not generic | violated by 13-22% of the family, growing with the machine; pinning the new gear's tooth drops it to 0-6.5% | real-teeth |
| 23 | corridor resonance and the golden spectral gap as position facts | big gaps recur at slot separations 35, 70, 105 with left endpoints pinned to `{10, 12, 18} mod 35` | under the escape-distance-1 ceiling (face B1) |
| 24 | `F/W = 0.25` flat at every computed machine | `y = 7..53` | the wall's factor of four; no mechanism |
| 25 | the ladder past the scan wall | `F(37) = 88`, `F(41) = 91` from m23's period alone, every gate exact; budget slack 14, 20, 16, 7, 38 | EXACT instrument, not a law (ladder_closure.md) |

## 4. OPEN

Structural questions about the motor alone that are neither proved nor the conjecture in
disguise. For each: the statement, what an attack looks like, and whether it blocks the clutch.

**O-M1. `L(M)` bounded, and `L_pad(M)` bounded.** `L(M)` is the length of the longest legal word;
`J_max = L + 2` and `A_kill = L + 1`, so every depth statement is a statement about `L`. The bare
half is capped uniformly (`L_bare <= 5`, KERNEL, docs/proofs/12); the padded half `L_pad` is
untouched and grows (0,0,0,1,1,1,2,2,2,2,3,3 at m11..m53). The only bound is
`L <= 2F(M+q')/q' + 1`, which grows with `F/q'`.
*Attack.* A uniform cap on `L_pad` of the shape of the bare cap - a residue obstruction on the
padded alternation - or a direct bound on the padded alphabet from the corridor's padding laws
(docs/proofs/14).
*Blocks the clutch?* **Yes, as understanding.** `L` is the only unbounded ingredient of the
motor's own grammar; every finite-depth statement about the motor (the closure, the dictionary
hierarchy, `J_max`) is conditional on it. It is a statement about `M` alone. Note the honest
counter-datum: `L` bounded is not structural on the family (max `L` on the family 1, 3, 3, 3, 5
against the real 0, 1, 1, 1, 2), so a proof must use the teeth.

**O-M2. The chain statement `Q*_J(M) <= F(M) + q'` for `J >= 3`, on the band.** Measured from
five sides, the budget's tightness inside the window is carried by 3- and 4-runs of ORDINARY old
gaps with a letter in the middle, in the band of old sizes `[15, 36]` at 29->31, and there is no
local certificate for that band. The gate now closes at the certified row top (12, 20, 25, 35,
46).
*Attack.* Extend the CRT row search / LP-duality certificate scheme from the short-letter row to
the whole band; or find the missing inequality on the extremes of the merge closure (O-M3).
*Blocks the clutch?* **Yes, as understanding.** It is the one object still open INSIDE the
window and it is a statement about `M` alone. Its ingredient list is known to require the real
higher gears' teeth (2f refuted at 23->29 by 62 > 61), so it is not soluble by any teeth-free
argument.

**O-M3. A monotone or contracting functional of the merge closure.** The closure is proved: the
depth-`m` dictionary of `M + q'` with multiplicity is determined by the depth-`K_m` dictionary of
`M`, `K_m <= m J_max`, and it is a deterministic finite-depth recursion on exact objects whose
extremes are the records. Nothing on the tree bounds a dictionary's extremes by its
predecessor's.
*Attack.* The candidates are named and never run (node 4.i.b.ii): legal-word density per opening
by depth, all-pad density, the order variance.
*Blocks the clutch?* **Yes, as understanding.** The owner's own correction stands: "nothing here
bounds it" is a brick, not a verdict. This is the single named unrun item on the motor.

**O-M4. The pinned letter as a law.** `a_L + r(a_L) <= F(M) + 3` at 9 of 9 rungs; the constant 3
is real-teeth and a proof must use `u_g = 6^{-1} mod g` and `3 a_L = q' -+ 1`.
*Attack.* The arithmetic route is named: the forced-gear law, the forced-cover count, the
twin-partner law and D3 are proved, and the residue obstruction explains 0, 1, 0, 3, 0, 0, 0, 1
of a deficit running 0, 3, 3, 3, 3, 4, 8, 9 and growing - so a per-letter CRT enumeration cannot
close it and the certificates are the answer.
*Blocks the clutch?* **No.** It is a sharpening of O-M2's certificate, not an independent
mechanism; the band is already a certified object without it.

**O-M5. The adversarial lemma `A(K) < (p_{K+1}^2 - 1)/6` for all `K`.** Proved for `K <= 10` by
certificate and `A(K)` exact to `K = 12`; margin 2.7-3.8, flat; no induction step exists.
*Attack.* The residual is a lower bound on the tiler function `h_S(L)`, the least holes `k`
primes leave in a run of `L` - a capacity statement with a gear count.
*Blocks the clutch?* **No, and it cannot be a gate item**: it is strictly STRONGER than the root
(it quantifies over gear sets as well as phases), so it is a statement about a family the motor
belongs to, not about the motor. The record says so at gear_count.md and arc_multiset.md.

**O-M6. `D_g = A_kill` bounded, and `Delta_J <= s_min` / `Delta_J = O(1)`.** Named in
docs/proofs/README "Not proved, and said so" and never closed; `Delta_J` measured in `[-3, +4]`
on the real machine and `eps in [-21, +15]` on the family, which killed par trading (2a).
*Attack.* None on the tree. The family measurement is the obstruction.
*Blocks the clutch?* **No.** These are ingredients of the per-step formulations, and the per-step
formulation is known to over-ask (face E1).

**Explicitly NOT open items of the motor, because they are the conjecture in disguise:** the
budget inequality `F(M+q') <= F(M) + q'` (TARGET, certified rung by rung, never a law); the pair
statement `F_2(M) <= F(M) + q'` (at column 0 it reads `2 d_0 <= F + q'`, and every route to it is
twin-Bertrand); `d_0 <= W`; the window statement `F(y) < y^2/6`; `K_columns(W(q)) > pi(q) - 3`
(the root in covering language, wall 5a as corrected by 5d).

---

# WHEELS - the top machine alone, on the raw line

## 1. DEFINITION

The wheels are written on the RAW LINE - the integers unfolded, no anchor, no six-fold, no
columns. The gears are the primes in `(q, Q]`, each striking its multiples. The object is a PAIR
`n = (n, n + 2)`, indexed by its lower member, so the pair coordinate IS the raw line, and gear
`g` strikes the pair `n` iff `g | n` or `g | n + 2`: **two TEETH, at `0` and `-2`, of separation
2**, leaving `g - 2` open residues in two ARCS of lengths `g - 3` and 1, the singleton being the
SHIELD `n = -1`. The WHEEL is `W = prod g`; a SLOT is an open residue class; a RUN is a maximal
block of consecutive open pairs; the RECORD `F_top` is the longest block of consecutive `n` with
no open pair; a GAP is the difference of consecutive open pairs; the LETTERS are `{2, g - 2}`; a
DOMINO is two adjacent open pairs. The MIRROR is `n -> -n - 2`, whose unique fixed point is the
shield. The ORIGIN CLUMP is the `2(q' - 3) + 1` forced slots around 0. The one-sentence identity:
**a strike never comes alone - its partner is exactly 2 away - so the wheels are a DOMINO
MACHINE**, and everything metric follows from that.

**Regimes.**
*Free wheels* - every gear above `2m`. Then the walk is the mex of `2m` residues, the record is
`F_top = 2m - (m mod 2)`, decided by the parity of the gear COUNT and not by the gears' sizes at
all, and removing a gear costs the same whichever gear leaves. The sharp threshold is
`q' >= 2m + 1` for even `m` and `q' >= 2m + 3` for odd `m` (document 5 L55); the sharp mex
criterion is `F_top(G) < q'` (document 5 L50).
*Loaded wheels* - small gears present. The CORE is `{g in G : g <= F_top + 1}` and the TAIL is
the rest; the record is a function of `m` and the core alone, the tail entering only by its
number `t` (documents 2 L31, 4 L53). In use the tail is EMPTY.
*On a range* the machine splits by height. The SMOOTH ZONE is `[1, Q]`: a pair `n <= Q - 2` is
open iff `n` and `n + 2` are both `q`-smooth, a finite Stormer list. The QUIET ZONE is
`(Q, Q^2]`: `n` is open iff `n = s P` with `s` `q`-smooth and `P` either 1 or a single prime
above `Q`; its lower edge in the machine's own vocabulary is `g_0^2`, the square of the first
gear whose square exceeds `Q`, and the rule dies at `p_1^2`, not `Q^2`. The wheels have NO
ANCHOR: folding needs a gear leaving at most two slots, i.e. `g <= 4`; what the smallest gear
fixes instead is the metric (run `q' - 3`, chain `q' - 2`, clump `2(q'-3)+1`) and what the gear
count fixes is the record, and neither fixes the other.

## 2. PROVED

### Structure (teeth, dominoes, arcs, symmetry)

| law | statement | status | hypothesis / evidence |
|---|---|---|---|
| d1 L1 teeth and count | two teeth at `0`, `-2`; `g - 2` slots | KERNEL | `strikesR_iff`, `card_open_residues`; `3 <= g` |
| d1 L2 arcs and shield | slots form arcs `(g-3, 1)`; the singleton is `n = -1` | KERNEL | `open_residues`, `not_strikes_neg_one`; `3 <= g` / `2 <= g` |
| d1 L3 the partner law | every strike has a partner strike at distance exactly 2; a gear's struck set is a disjoint union of dominoes `{x, x+2}` | KERNEL, **no hypothesis** | `partner`, `strikes_iff_domino`; 5.6 million struck residues, 8 wheels |
| d1 L4 the forbidden gap 4 | `n` and `n + 4` open implies `n + 2` open: a gap of exactly 4 is impossible | KERNEL, **no hypothesis** | `open_of_open_add_four`, `no_gap_four`; 0 in 12 wheels and 27 range machines to `10^7` |
| d1 L6 shield, antipodes, clump | `n = -1` open for every gear set; `n = 2` and `n = -4` open; every `n in [-(q'-1), q'-3]` open except `0` and `-2` | KERNEL | `shield_open`, `two_open`, `neg_four_open`, `clump_open`, `origin_clump`; gears `>= 2` / `>= 5` / `>= q' >= 3` |
| d1 L7 the mirror | `n -> -n - 2` preserves the open set; unique fixed point the shield | KERNEL, **no hypothesis** | `strikes_mirror`, `open_mirror`, `mirror_fixed_iff` |
| d1 L8 the symmetry group | the affine maps preserving the open set are exactly `n -> c(n+1) - 1` with `c = +-1` mod every gear: `(Z/2)^m`, adjacency subgroup `Z/2`; every sign vector realised; the count `2^m` | KERNEL | `symm_not_dvd_mul`, `isolate`, `affine_gear`, `affine_group` (prime gears, `5 <= g`), `affine_group_of_unit` (general, under invertibility), `exists_symmetry`, `sign_count` |
| d1 L12 the chain law | two openings `x < y` of `M` are both struck by a new gear `g` in some copy iff `y - x = 0, +2, -2 (mod g)` | KERNEL, **no hypothesis** | `chain_law`; 118,341 pairs |
| d1 L13 the merge law | every gap of `M + g` is a gap of `M` or a merge of consecutive gaps of `M` whose interior openings `g` strikes | KERNEL, **no hypothesis** | `merge_law`; 34,646 gaps |
| d3 L45 the holes | in the pair view the only gap hole below the record is `d = 4`; in the triple (twin-candidate) view the holes are exactly `d = 2` and `d = 3` | KERNEL, **no hypothesis** | `start_of_start_add_two/three`, `no_start_gap_two_three`, `no_start_gap`, `no_pair_gap_four` |
| d5 L59 the anchoring dichotomy | a gear folds the line iff `g - t_g = 1`, i.e. `g = 3` (two adjacent teeth) or `g = 2` (one tooth); an exact list of what each destroys and what survives | PROOF + EXACT (23 wheels) | top_machine_5.md |
| d2 L34 no top-machine anchor | an anchoring gear needs `g - 2 <= 2`, i.e. `g <= 4`; top gears leave `g - 2 >= 5`, so the smallest gears give a corridor of density `>= 0.58`, uniformly filled | PROOF (a count, not an accident) + EXACT (7 small wheels; uniform descent 5 of 5) | top_machine_2.md |

### Counting (wheel counts, census, correlation)

| law | statement | status | hypothesis / evidence |
|---|---|---|---|
| d1 L5 the wheel count | `prod (g - 2)` open pairs per wheel | KERNEL | `wheel_count`, engine `card_filter_crt`; gears `>= 3`, pairwise coprime; verified over a full `6.7e9` period |
| d3 L44 the correlation product | `B(d) = prod c_g(d)` with `c_g = g-2, g-3, g-4` for `d = 0, +-2, else`; hence `B(1) = prod(g-4)` and `B(2) = prod(g-3)` (document 1 L15) | KERNEL | `corr_prod`, `card_both_residues_eval`, `pair_corr`; `5 <= g`, pairwise coprime; 400 values, 0 mismatches |
| d2 L22 the gap census law | `N_d(G) = sum_{S subset [1,d-1]} (-1)^{|S|} prod_g (g - |E_g(S)|)`, `E_g(S) = ({0,-2,-d,-d-2} u {-j,-(j+2) : j in S}) mod g` | PROOF (CRT + inclusion-exclusion); **not in the kernel** | 15 wheels every gap length, and the full period of the eight-gear wheel to `d = 16`: 0 mismatches |
| d2 L24 W1 closed | `N_3` and `N_5` share the universal signature `prod(g-4) - 2prod(g-5) + prod(g-6)`; the gap-3 classes collapse only for `g | 3` or `5`, never a gear, the gap-5 classes for `g = 7`; so the counts are equal iff every gear exceeds 7, and `(3,5)` is the only coincident pair for `d <= 16` | PROOF | 16 wheels including the `6.7e9` full period |
| d2 L27 the odd census length | every gear is odd, so `N_d` is odd iff a signature sum over even `e` is odd, which happens only at `d = 1` | PROOF (second, independent proof of document 1 L9) | `d = 1..16` |
| d5 L51 the odd length, corrected | exactly one gap length has an odd count and it is the mirror-self-paired gap (`2a + d + 2 = 0 mod W`) | PROOF (from the mirror) + EXACT, 15 wheels | document 1's L9 is false as soon as `N_1 = 0`; L51 needs no hypothesis |
| d2 L28 the joint census | the residues struck by exactly `j` gears number `sum over j-subsets 2^j prod_{others}(g-2)`; all `m` gears strike on exactly `2^m` classes, of which exactly two (`n = 0`, `n = -2`) have every gear on the same tooth: **the origin is the machine's unique total collision, and the clump is its shadow** | PROOF (CRT) + EXACT, 14 tuples, deviation 0 | top_machine_2.md |
| d2 L29 the collision law | in a window with every gear `> L + 1`, no three distinct traces pairwise intersect: three gears share a position only if two do the same job | PROOF (one line) + EXACT, 1,354 triples, `L <= 12` | consequence: the record cover is a perfect tiling for even `m` and wastes exactly one unit for odd `m`, 11 of 11 |
| d5 L52 every counting law is a tooth-count law | replacing 2 by `t_g = |{0,-2} mod g|` carries the slot count, wheel count, run and chain counts, correlation, census, parity bias, transform and all-struck classes down to gear 2 | PROOF + EXACT | 23 wheels; 720 correlation values; 59,141 frequencies; 0 mismatches everywhere |
| d6 L61 the exact count | `#{admissible n <= X} = Psi(X, q) + sum_{s q-smooth <= X/p_1} (pi(X/s) - pi(Q))` for `X <= Q^2`; each family's count is a property of the integers, not of `q` | PROOF + EXACT, 13 machines, 0 mismatches | family `(1,1)` contributes `pi_2(Q^2) - pi_2(Q)` at every `q` |

### Metric (runs, chains, records, walks)

| law | statement | status | hypothesis / evidence |
|---|---|---|---|
| d1 L10 the two ceilings | the longest run of consecutive open pairs is exactly `q' - 3`; the longest step-2 chain is exactly `q' - 2`; both attained | KERNEL (both bounds and both attainments) | `no_long_run`, `run_lt`, `run_attained`, `no_long_chain2`, `chain2_lt`, `chain2_attained`; `5 <= q'`, `q'` odd, `q' in G` |
| d5 L53 the ceilings corrected | the run ceiling is `max(q' - 3, 1)`; the CHAIN ceiling is `q_odd - 2`, the smallest ODD gear minus two, because gear 2 is invisible to a step-2 chain | PROOF (mechanism) + EXACT, 14 of 14 | top_machine_5.md |
| d1 L16 the record is a cover | `F_top(G)` is the largest `L` such that `[0, L)` is covered by one phase per gear, each contributing a singleton, a domino `{x, x+2}`, or (if `g <= L+1`) the long letter `{x, x+g-2}` | PROOF (exact characterisation by CRT) | validated against the full-period scan 15 of 15 |
| d1 L17 the parity law | `F_top(G) = 2m - (m mod 2)` when every gear exceeds `2m + 1` | KERNEL, **as an equality** | `parity_core`, `parity_upper` (gears odd, `> 2m+1`), `parity_attained` (no size hypothesis), `parity_law` (`IsGreatest`) |
| d5 L55 the sharp parity threshold | the parity law holds iff the record cover is a tiling by free dominoes, and that happens exactly when `q' >= 2m + 1` (even `m`) or `q' >= 2m + 3` (odd `m`) | PROOF (the long letter `g-2` is odd and is the only parity-crossing piece; counted exactly) + EXACT, 7 boundary pairs, 0 exceptions; (b) equivalent to (c) 28 of 28 | document 1's `q' > 2m+1` is sufficient at both parities and NOT necessary at even `m` |
| d3 L30 the mex form | the next open pair after `x` is `x + mex{(-x) mod g, (-x-2) mod g : g in G}` | KERNEL, as an `IsLeast` | `mex_form`, `mexS_le`; hypothesis `2m < g` used only in the openness half; sharp (36 mismatches at `{7,11,13,17}`) |
| d3 L31 the location bound | `L(x) <= 2m - (m mod 2)` | KERNEL, by reuse of `parity_upper` | `mexS_le_parity`; gears odd, `2m + 1 < g` |
| d5 L50 the sharp mex criterion | `M(x) < q'` implies `L(x) = M(x)`; hence `F_top(G) < q'` makes the mex form exact everywhere | PROOF (one line) + EXACT | 94,774 positions across 18 wheels, 0 exceptions; `F < q'` predicts 0 failures at 26 of 26 gear sets; `{6,11,13}` separates it from `q' > 2m` |
| d3 L32 the general mex form | `L(x) = mex(union_g ({a_g,b_g} + g Z_{>=0}))`, truncatable at any proved bound `B` at `2 sum_g ceil((B+1)/g)` terms | PROOF | 24,000 in-use walks, 160-443 gears, 0 mismatches |
| d3 L34/L35 the triple machine | `R(x) = mex` of `3m` residues; and `F_3(G) = 3m` EXACTLY, with no parity defect, because the solid triomino tiles and the gapped domino does not | KERNEL, as an `IsGreatest` | `triple_mex_form`, `mexT_le`, `triple_upper`, `triple_attained` (no size hypothesis), `triple_law`; `3m < g` / `3m + 3 <= g`, pairwise coprime |
| d3 L33 the counting bound | `L <= 2m/(1 - 2H_S)`, `H_S = sum_{g <= L} 1/g`, valid when `H_S < 1/2` | PROOF | true at all twelve in-use machines; non-vacuous at two |
| d4 L50 where the counting bound dies | it is alive exactly while `F < L*(q) = exp exp(1/2 + sum_{p<=q} 1/p - M)`, `M` Mertens: `L* = 35, 61, 91, 130, 175, 231, 294, 359, 514` | PROOF (Mertens closed form) + EXACT, 35 of 36 machines correct | where alive it is 40-50x loose |
| d3 L38 the hop collapse | `g` hops at the landing `y` iff `y = 0` or `-2 (mod g)`; if `g > F_G + 3` the hop chain is at most 2, and a double hop occurs iff `y = -2 (mod g)` and the lower gap is exactly 2 - a one-line non-recursive layer | PROOF | 12 layers, 391,048 positions, 0 exceptions; sharp in the other order (gear 7 last gives chains of 3) |
| d5 L58 the hop collapse transfers | the chain bound `<= 2` survives the change of coordinate; the double-hop rule becomes "the lower gap equals the forward letter of the landing's tooth" | PROOF + EXACT | 18 pair layers, 4 column layers, 10,860 hits, 0 exceptions |
| d4 L46 the gear-zone identity | for `1 <= n <= Q - 2` the pair `n` is open iff `n` and `n + 2` are both `q`-smooth | PROOF (two lines) | 36 machines, 130,230 cells, 4,019 openings, 0 exceptions |
| d4 L47 the in-use record | the zone record is the largest gap of the finite `q`-smooth-pair list below `Q`, value AND position (the block starts one above the gap's lower end) | PROOF (from L46) | 36 machines + 24 sweep points, 0 exceptions |
| d4 L48 the proved lower bound | `F_range(N) >= max(maxgap of the list, Q - 2 - s_k)`; for `Q > s(q)`, `F_range(N) >= sqrt(N) - s(q) - 2`; no sieve estimate anywhere | PROOF (finiteness is Stormer 1897 / Lehmer 1964; the members used are computed exactly, so the bound is unconditional) | ratio truth/bound 1.000-2.000, exactly 1.000 at 17 of 36 |
| d4 L49 no bound in `(q', m)` | `Q ~ (m log m)/2`, so `F_range >= (m log m)/2 (1+o(1)) - s(q)`: no bound linear in `m` can hold; measured `F/2m` = 1.50, 2.60, 3.39, 4.02 and still climbing | PROOF + EXACT | 36 machines |
| d4 L51 the ceiling on union bounds | a union bound over covering patterns controls lengths only to `d <= 2 log N / log q'` | PROOF | the in-use record exceeds it by a factor 4 to 520 |
| d4 L53 the core/tail rule | `F_top(G) = max{L : min over core phases of the domino cost `D(U)` `<= t}` with `D` the sum of `ceil(run/2)` over step-2 runs per parity class | PROOF (from L16) | 0 mismatches on 13 known records; 89 consecutive-prime sets decided exactly |
| d4 L54 the additive form | `F_top = F_core + 2t - (t mod 2)` holds at exactly the sets with an EMPTY core (where it is the parity law) and fails at all 25 with a core | PROOF (mechanism: a core gear in a longer window contributes `2 ceil(L/g)`, not 2) + EXACT, 64 of 89 | pre-registered as refuted with `{13..41}` named in advance |
| d5 L56 the anchor rescaling law | `F(G+{2}) = 2F_2(G)+1`, `F(G+{3}) = 3F_3(G)+2`, `F(G+{2,3}) = 6 F_col(G) + 5` | PROOF (mechanism) + EXACT, 30 of 30 | the manager's note: it is the fold as a coordinate identity, exact and known, kept as FACT |
| d5 L57 the certified column mex | `M_B(x) = mex` over the gears' two COLUMN teeth WITH their recurrences; `M_B(x) <= B` certifies the next open column | PROOF (self-certifying) + EXACT | 890,501 walks, `{5..q}` to `q = 31`, 0 mismatches; the uncorrected two-per-gear form is wrong at 12% of positions |
| d6 L57 the zone rule | for every `n <= Q^2`: `n` admissible iff `n = sP` with `s` `q`-smooth and `P` either 1 or one prime `> Q` | PROOF (two lines) | 45 machines, `q = 5..17`, `Q = 50..10^4`, every cell of `[1, Q^2]`, 0 exceptions |
| d6 L58 the two edges | the first admissible non-smooth `n` is `p_1 = nextprime(Q)`; the rule first fails at `p_1^2`; there is NO transition band | PROOF | 45 of 45 for each |
| d6 L59 the stratification | at height `x` the smooth cofactor is at most `x/p_1`, attained in every stratum; in `(Q, 2Q]` admissible = prime or `q`-smooth | PROOF | every dyadic stratum of `(10^4, 10^8]`, 0 pairs violating among 5.4 million |
| d6 L60 the family decomposition | every open pair of the zone carries a label `(s, s')` of smooth parts with `gcd(s,s') | 2`, and the zone is the disjoint union over those labels of `{(sP, s'P') : s'P' - sP = 2}` | PROOF | 1,510 families at `q = 5`, `Q = 10^4`; 0 with a forbidden gcd among 5.4 million pairs |
| d6 L62 the walk in the zone | `nextadm(x) = min(least q-smooth >= x, min over q-smooth s of s * nextprime(max(Q, ceil(x/s))))`: the mex over residues becomes a minimum over smooth scalings of the next-prime function | PROOF (from L57) + EXACT | 11,489,920 positions, 0 mismatches; 600,000 pair walks by iteration, 0 mismatches, mean 2.3-3.5 iterations |
| d6 L63 alignments always occur | any prime gap in `(Q, 2Q]` free of `q`-smooth numbers is a run of struck pairs, so the zone record is at least the largest such gap | PROOF | 48 machines, 0 exceptions; truth 3.22 to 24.00 times the floor |
| d6 L66 L4 survives onto the range | no gap of distance 4 in the zone, because the argument is local | PROOF (inherited) + EXACT | 0 in 49,433,381 gaps over seven machines |

### Location (where records sit)

| law | statement | status | evidence |
|---|---|---|---|
| d1 L21 / d4 L46 | the in-use record always lies in the gear zone `[1, Q]`, immediately above the origin clump | PROOF (from the zone identity) | 18 of 18 range machines; at `q = 5`, `N = 10^9` the record run fills 99.8% of `[1, Z]` |
| d4 L55 no saturation | `F_range` is LINEAR in the largest gear (186, 858, 3,006, 9,846, 31,560, 99,990, 999,876 as `Q` runs 316 to `10^6`), because the gear zone is `[1, Q]` whatever `N` is; and above the zone added gears keep merging blocks (200 -> 1,511) | PROOF (two exact causes) + EXACT, 24 + 12 points | the expected saturation does not occur, in either form |
| d6 L58 | the zone's lower edge is `p_1 = nextprime(Q)`, named in the machine's own vocabulary by `g_0^2 = Q(1 + O(g_0/sqrt Q))` (measured 2.0%, 1.3%, 0.5%, 0.2%, 0.05% above `Q`); `Q - sqrt(Q)` is not an edge of anything | PROOF + EXACT | at `q = 5`, `Q = 10^4`, `Q - sqrt Q = 9,900` sits inside one struck block running 161 to 10,006 |

### Transforms (spectral, bitwise, characters)

| law | statement | status | evidence |
|---|---|---|---|
| d3 L40 the per-gear transform | `u_g^(0) = (g-2)/g`, `u_g^(a) = -(1 + omega^{2a})/g`, which in the SHIELD coordinate `n+1` is the real `-(2/g) cos(2 pi a/g)`: the top machine is the `u = 1` machine and the bottom's fold is replaced by one translation | PROOF + EXACT DFT | 5 wheels, 32,077 frequencies, max error 1.1e-16 |
| d3 L41 full spectral support | `O^(a) != 0` for every `a` (`cos = 0` needs `4a = g mod 2g`, impossible for odd `g`); the same for every nonempty run indicator; so the spectrum decides the run record and CANNOT decide `F_top`, whose positivity is that of an alternating sum | PROOF | minima 3.3e-07 to 3.1e-05, never 0 |
| d3 L42 the parity bias | the striker-parity XOR has `#even - #odd = prod(g - 4)` exactly - the same polynomial as the domino count | PROOF (one character sum per gear) | 10 wheels, exact |
| d3 L36 the C-identities | `#{L = j} = C(j) - C(j+1)`; `N_d = C(d-1) - 2C(d) + C(d+1)`; `F_top = max{j : C(j) > 0}`; `sum_x L(x) = sum_j C(j)` - the gap census is the second difference of the all-struck count, dual to L11's run spectrum | PROOF | all five identities, 10 wheels, 0 mismatches |
| d3 L37 the closed form of `C(j)` | `C(j) = sum_{k,e} (-1)^k T(j,k,e) prod_g (g - 2k + e)` with `T` a path convolution; and `C(j)` is provably NOT a product, because "all struck" is not a per-gear condition | PROOF | 6 wheels where the hypothesis holds, 0 mismatches; `T` verified to `j = 12` |
| d1 L11 the run spectrum | the number of maximal runs of exactly `L` open pairs is the second difference of `prod(g - 2 - L)`; an arithmetic progression of common difference exactly 6 for any three-gear wheel | PROOF (a per-gear count) + EXACT | 12 wheels, every `L`, 0 mismatches |

## 3. MEASURED, NO PROOF

| # | statement | range and count | reading |
|---|---|---|---|
| 1 | d2 L25 the degree law: `N_d` has degree `m - r(d)` in the gears and is gear-independent exactly when `r(d) = m`, value `(-1)^m M_m(d)` | 0 mismatches over `m = 3,4,5`, `d = 1..14`, three disjoint gear sets each | rests on the **unproved** vanishing of `M_k(d)` for `k < r(d)`, verified only to `d = 16` (see OPEN W-1) |
| 2 | d1 L18 universal record multiplicity 18, 24, 480, 720 for `m = 3,4,5,6` | 14 large-gear wheels, 0 exceptions | derived from L25, hence conditional on the same unproved vanishing |
| 3 | d2 L26 `r(d)` is the parity covering number, hence `F_top(m) = max{d : r(d) <= m} - 1` reproduces the parity law, and `d = 4` is the unique place the gap's closed-boundary cover differs from the record's free-boundary cover | `d = 1..16`, `d != 4`; census record = cover record 15 of 15 | verified, not proved in general |
| 4 | d2 L31 the sub-threshold reduction: `F_top` is a function of `m` and of `{g <= F_top + 1}` alone | 90 comparable cases, `m = 3..8`, 0 exceptions | now has a formula in document 4 L53 (PROVED); L31 itself is the measured statement it generalises. Pre-registered threshold `2m + 3` REFUTED (12 of 70) |
| 5 | d2 L30 the record of a triple or quadruple takes exactly two values, decided by one bit - is 7 a gear | exhaustive: 1,540 triples (1,330 / 210) and 7,315 quadruples (5,985 / 1,330), 0 exceptions | EXACT over a finite family, with the mechanism stated (only the odd long letter crosses parity); not a general theorem |
| 6 | d2 L32 the range record is a first hit on the wheel's census: `F_range(N) = max{d : W/c(d) <= N} - 1` | within one unit at 19 of 21 checkpoints, within two at the other two; first-occurrence positions within a factor ~3 of `W/c(d)` | the wheel record IS reached, at 0.005% to 10.9% of the period |
| 7 | d2 L33 the record blocks are pinned modulo the small gears | full period of `{7..31}`: 8 blocks in four mirror pairs summing to `W - 33`, two residues mod 1001, one mod 11, one mod 17 | measured on one wheel |
| 8 | d2 L35 no direction: the small wheel's cyclic gap word read from the shield is a palindrome | 6 of 6 wheels (23 wheels in document 5) | the mirror fixes the shield and reverses the cycle - a mechanism, not written as a proof |
| 9 | d2 L36 the metric anchor: run `q'-3`, chain `q'-2`, clump `2(q'-3)+1` are functions of `q'` alone; the record of `m` alone | 14 of 14 wheels, 0 exceptions | the ceilings are L6/L10 restated (proved); the separation from the record is the measured content |
| 10 | d2 L37/L38 the removal law and removal independence: `W` divides by `q'`, the open count by `q'-2`, dominoes by `q'-4`; ceilings grow; `F_top` falls by 3 (even `m`) or 1 (odd `m`); and `F_top(G \ g)` is the same whichever gear leaves | 16 steps over four chains; independence 9 of 9 in the large-gear regime, failing exactly outside it | EXACT; each ingredient but the `F` step is a restatement |
| 11 | d3 L43 the XOR run equals `F_top` at both parities, because a record block covered exactly once always exists | 10 of 10 wheels | the EXISTENCE of an exactly-covered record block is measured; it refuted the pre-registration that odd `m` would fall short |
| 12 | d4 L52 in use the tail is EMPTY: certified covers give `F_top >= Q`, 3 to 43 times `2m` | 26 of 27 in-use machines (the exception is the weakest certificate) | this is WHY the covering bound goes vacuous in use; pre-registration P3 refuted |
| 13 | d4 L56 the two regimes and the crossover: `F_range = max(F_zone, A)`, `A` fitted by the independent first-hit model to within 0.97-2.38; the zone wins in 26 of 36, and each `q` has one crossover `N` after which it wins for good | 36 machines; `A` from 24 to 419 | the pre-registered ceiling `A <= 400` is refuted (419) |
| 14 | d6 L64 the record profile is a U and the record is a competition between the two ends of the zone | stratum tables at 7 machines, positions at 48; bottom wins 28 of 48, losses narrow (200 against 193) | measured |
| 15 | d6 L66 the W1 near-equality survives onto the range: distance-3 and distance-5 counts agree to 0.02% exactly when 7 is not a gear | 7 full spectra, 49.4 million gaps | the wheel version is proved (L24); the range version is measured |
| 16 | d6 L65 the zone record has NO available upper bound | - | **the conjecture in disguise, and the record says so exactly**: the zone's bottom stratum is the family `(1,1)`, two primes two apart, so bounding the zone record above IS bounding the gaps between twin primes |
| 17 | d1 the budget analogue holds with vast slack: `F_top(M+g) - F_top(M)` never exceeds 7 over 69 exact ladder steps, against new gears up to 97 | 6 ladders | the record grows linearly in the gear COUNT, about 2 per gear, never with the size of a gear |
| 18 | d1 the range density of a FIXED gear set is flat to four or five figures over a range `10^4` times below its wheel; the IN-USE density exceeds the CRT product by 5-17% and falls monotonically | 27 range machines to `10^7` | non-periodicity belongs to the in-use machine, not to a gear machine as such; the excess is a Buchstab excess |

## 4. OPEN

**O-W1. The vanishing moments: `M_k(d) = 0` for `k < r(d)`.** Verified to `d = 16`, unproved.
*Attack.* A direct evaluation of the census signature `c_e(d)` as a moment sequence; the branch
says a proof "would probably also give `r(d)` in closed form and hence a second proof of the
parity law".
*Blocks the clutch?* **Yes, as understanding.** It is the single load-bearing gap in the wheels'
counting theory: the degree law (d2 L25), the universal record multiplicity (d1 L18) and the
census route to the parity law (d2 L26) all rest on it, and all three are currently MEASURED for
that reason. It is a statement about the wheels alone. **This is Slot B's first target.**

**O-W2. `L31` as a formula.** The record is a function of `m` and of the gears `<= F_top + 1`;
document 4 L53 turns that into an exact inequality (the core/tail rule with the domino cost), but
the function itself is still tabulated per core, not derived.
*Attack.* Named in document 2: a parity-refined covering bound per small gear, following the
mechanism that only the odd long letter `g - 2` crosses parity. Never attempted.
*Blocks the clutch?* **Yes, as understanding.** Without it the wheels' record is computed, not
known: we can decide any given gear set and cannot say what the record is as a function of the
split.

**O-W3. The census beyond `d = 20`.** Both known routes (inclusion-exclusion over uncovered
interior positions, and a transfer matrix over "which interior positions are covered") are
exponential in `d`, so the long tail of a wheel's census is obtainable only by scanning the
period.
*Attack.* None on the record.
*Blocks the clutch?* **No.** It is a computational reach limit, not a structural gap; the record
itself and its multiplicity live in the gear-independent regime, which the closed form reaches.

**O-W4. The in-use / above-zone record `A(q, N)`, as anything but a first-hit fit.** Document 4
left it open; document 6 gave it a mechanism (it IS the quiet-zone record, 8 of 8 by value and
position) and proved a floor made of prime gaps (L63) - but no upper bound.
*Attack.* None that is not the conjecture: L65 proves an upper bound is a twin-gap bound.
*Blocks the clutch?* **No, and it must not be a gate item.** By the owner's criterion this is not
hidden complexity we failed to understand; it is the target itself, named exactly, in the
wheels' own coordinate.

**O-W5. The kernel gap.** Formalised: L1-L8, L10, L12, L13, L17, L19 (documents 1, 2 as far as
the conjugacy), plus L30, L31, L34, L35, L44, L45 of document 3 - 158 declarations, zero sorries,
no `native_decide`, standard axioms. **Not attempted**, and named as such in the ledger: the L22
gap census; the general (in-use) mex `L32` with truncated progressions; the harmonic bound `L33`;
the distribution `C(j)`, its closed form, the hop law and the nested form (`L36-L39`); everything
spectral and bitwise (`L40-L43`), which needs a DFT nothing in the files has; and every law of
documents 4, 5, 6 (the zone laws, the core/tail rule, the anchor rescaling, the certified column
mex).
*Attack.* The cheapest are already named in order in document 3 and the Lean ledger.
*Blocks the clutch?* **Partly.** The kernel is the project's strictest evidence standard, and the
wheels' metric core is in it; what is not in it is the counting theory beyond the wheel count and
the correlation, and all of the in-use theory. A clutch built on the zone laws would rest on
written proofs, not kernel ones.

**O-W6. The small-`q'` regime of documents 1 and 2 is untested.** Document 5's own dead-end note:
document 2's L32 (the range record as a first hit) and L33 (the pinning of record blocks) and
document 1's L21 (the gear zone) were never tested at small `q'`, because they need whole-period
scans of wheels containing 2 and 3. "Left open, and noted as the only rows of the law table with
no measurement."
*Attack.* Whole-period scans of the small-`q'` wheels; the branch says they are as large as the
large-gear wheels already scanned and were out of budget.
*Blocks the clutch?* **No.** The clutch's wheels are the primes above `q`, so `q' >= 7` always;
the small-`q'` rows matter only for the conjugacy reading (the wheels at `q' = 5` ARE the motor).

**O-W7. Prior art for the wheels as an object.** Every top-machine document stops its own prior
art in a line - Jacobsthal 1961 / Iwaniec 1978 for the two-class Jacobsthal function; Schemmel /
Hardy-Littlewood for `prod(g-2)`, `prod(g-3)`, `prod(g-4)`; Stormer 1897 / Lehmer 1964 for the
smooth-pair finiteness; Bach-Peralta 1996 for semismooth numbers; Mertens for `L*(q)` - and
`docs/proofs/22` records "**Prior art not checked for the machine as an object**".
**There is no entry in `docs/novel/README.md` for any of the wheels' laws.** By the register's own
rule ("nothing here is announced as new until section 6 has a dated check"), every law of
documents 1-6 is UNCONFIRMED as novel.
*Attack.* A harvester prior-art pass and register entries.
*Blocks the clutch?* **No** for correctness; **yes** for the record's honesty, and it is cheap.

---

# EXHAUST - the stack above the wheels

## 1. DEFINITION

The exhaust is everything above the wheels, and it is a STACK. Tier 1 is the motor, the primes up
to `q`; its period is `cut q 1 = q#`. Tier 2 is the wheels, the primes in `(q, q#]`; its period
is `cut q 2`. In general **tier `k + 1` is the primes in `(cut q (k-1), cut q k]`**, so the CUTS
are `cut q 0 = q`, `cut q 1 = q#`, and `cut q k` = tier `k`'s period = the lower edge of tier
`k + 2`. Every tier is a machine of the same construction - gears striking their multiples, the
pair `(n, n+2)` as the object, teeth `0` and `-2`, no anchor - so every top-machine law, being
stated in `(smallest gear, gear count)` only, applies to it verbatim at its own split.

**Regimes.** On the QUIET ZONE `(C, C^2]` above a cut `C`, a strike by an exhaust gear `p > C` on
`n` leaves only two possibilities: `n = p`, a **HOME STRIKE** (the gear is the number, and the
number is prime), or `n` has a prime factor `<= C`, an **ECHO** of a gear at or below the cut.
SPAN is the stride relation: a gear SPANS a machine when its stride is at least that machine's
period, so it has at most one multiple - hence at most two struck pair positions - inside any
window of that period, while the machine's own pattern repeats in full there. STRIDE CONTAINMENT
is the tower's shape: every gear of tier `k + 2` exceeds tier `k`'s period, so it spans tier `k`;
and only two tiers down is there room - a gear never spans its own tier, and a gear of tier
`k + 2` never spans tier `k + 1`. The tower therefore turns the root into a LADDER OF WINDOWS at
the primorial rungs `q, q#, (q#)#, ...`, each rung asking "machines `1..k+1` leave an open pair
in `(P_k, P_k^2]`", with the same missing instrument at every rung.

## 2. PROVED

Everything in this section is on the record as FACT at tree node R4.b.viii, established by
reasoning from the tiers' definition and from the wheels' laws. **No branch document has been
written for the exhaust, and no computation has been run on it.** The kernel work is Slot A and
is not counted here.

| # | statement | status | evidence |
|---|---|---|---|
| X1 self-similarity | every top-machine law is stated in `(smallest gear, gear count)` only, so tier `k+1` is the top machine at split `(cut q (k-1), cut q k]` and obeys the same laws: the zone law below its top gear, smooth-times-one-prime on its quiet zone, and the mex and parity laws only where its smallest gear exceeds `2 m_k` (which fails from tier 3 on) | FACT (reasoning) | theory_tree.md R4.b.viii |
| X2 the removal law is the self-similarity, one gear at a time | document 2's L37/L38 (raising the split divides `W` by `q'`, the open count by `q'-2`, the dominoes by `q'-4`, and drops `F_top` by 3 or 1) is the stack's self-similarity read gear by gear | FACT | R4.b.viii, top_machine_2.md |
| X3 stride containment, the bottom half | every third-tier gear exceeds the motor's period `P`, so it strikes at most 2 positions per bottom period and the bottom's full twin-slot pattern sits inside every stride | FACT (one line) | R4.b.viii |
| X4 stride containment fails for the middle | the wheels' period `prod (q, Q]` is far above `Q`, so only third-tier gears above THAT period contain a full wheels period, and those are silent on ranges below it | FACT | R4.b.viii |
| X5 THE EXHAUST CAP (the owner's cap) | in `(Q, Q^2]` every third-gear strike is a home strike (`n = P`, a prime above `Q`) or a duplicate of a bottom or middle strike (`n = sP` with `s > 1` having a factor `<= Q`); **so an open pair of bottom + middle there is a twin prime, for every `q`, to infinity** - nothing above `Q` touches the window `(Q, Q^2]` | PROOF (the sieve-to-the-square-root fact in machine form) | R4.b.viii; it is the wheels' L57 seen from above |
| X6 the tower | with machine `k+1` = the gears from the top of machine `k` to machine `k`'s period, the lower cut of machine `k+2` IS machine `k`'s period, so every gear of machine `k+2` strides a full period of machine `k`, for every `k`, and machine `k+1` never fits except at its silent top; the redundancy cap repeats at every level | FACT | R4.b.viii |
| X7 no new interactions up the ladder | theorem (E) already says the effective machine at a column is exact, so the tower's shape is fixed - known machines + one in-use machine (smallest gear = the previous period) + clutch, on a zone of tranquillity - and the missing instrument is the same at every rung | FACT | R4.b.viii, position_frontier.md |
| X8 the stronger reading is settled negatively | "a bottom period with no third-gear strikes at all" exists by CRT only at heights where gears beyond the avoided set are active; exposure there is not twin-ness | FACT | R4.b.viii |

> ### SLOT A: the exhaust in the kernel - LANDED (round 35, manager-gated 2026-09-06)
> `proofs/MachineStack.lean`, 48 declarations, docs/proofs/23-stack-and-exhaust.md. Build of the
> five top-machine targets green at 2244 jobs; manager audit of `stride_containment`,
> `not_spans_below`, `exhaust_home_or_echo`, `open_iff_twin`, `wheels_open_iff_twin`,
> `stack_open_iff_twin`, `smooth_zone`, `quiet_zone`: propext, Classical.choice, Quot.sound;
> zero sorries. So X3 (`stride_containment`, `card_strikes_window_le_two`,
> `tier_pattern_repeats`), X4 (`not_spans_below`), X5 (`exhaust_home_or_echo`, `open_iff_twin`,
> `wheels_open_iff_twin` unconditional for `2 <= q`), X6 (`stack_eq_primesLE`,
> `stack_open_iff_twin`, under `CutMono`) and the wheels' zone laws (`smooth_zone`, `quiet_zone`)
> are KERNEL. Kernel finding: primality of the exhaust gear is never used in the cap. Gate item 1
> cleared; O-X1 (`CutMono`) stands exactly as written: proved for the first step only.

## 3. MEASURED, NO PROOF

**Nothing.** The exhaust has no measurement of its own on the record: no branch document, no
script directory, no exceptionless count. Every number ever quoted about a tier above the wheels
is a top-machine number read at a raised split (documents 1-6), and every statement in section 2
is reasoning from the tiers' definition. This is the honest state of the object and it is the
main reason for its gate verdict below.

## 4. OPEN

**O-X1. `CutMono`: is the cut sequence monotone?** `cut q k <= cut q (k+1)` for all `k`. The
first step is Bertrand (`cut q 0 <= cut q 1`, the primorial is at least its argument), and it is
**false at `q = 2, 3`** (`cut 3 2 = 6` while `cut 3 3 = 5`; the stack degenerates there). From
there on it is a statement about the density of primes in `(cut q k, cut q (k+1)]`, and the
kernel work carries it as an explicit hypothesis exactly where it is used.
*Attack.* From `q = 5` on the intervals are enormous (`cut q 1 = q#`), so the statement is far
weaker than any prime-gap result in print; the attack is to find the right elementary form (the
product of the primes in a huge interval exceeds its top) rather than to cite a gap theorem.
*Blocks the clutch?* **Yes, as understanding.** Without it, "the union of tiers `1..k+1` is
exactly the primes up to `cut q k`" - the statement that makes the stack a stack - is
conditional, and everything above tier 2 is stated under a hypothesis.

**O-X2. The zones and the redundancy lemma are named and not formalised.** The tree states them
(W11, W12 of the refiling): on a range of `K` columns, REPEATING gears `g <= sqrt(6K)` turn past
their own square and have exclusive kills; NON-REPEATING gears `sqrt(6K) < g <= 6K/q'` kill only
at members `gm` with `m < g`, and every such kill coincides with a smaller gear's (the redundancy
lemma, one line: the smallest prime factor of `m` is a smaller gear striking the same column);
SILENT gears `g > 6K/q'` strike only their own home column. The tree names the redundancy lemma
and the rigidity generalisation as "the first things to formalise when the line opens".
*Attack.* Both are one-line arguments; the work is stating them in the stack's own coordinate
rather than against bottom-open columns (the non-repeating clause as recorded is a clutch
statement, per the refiling's own note).
*Blocks the clutch?* **Yes, as understanding.** The zones are what say which gears of a tier do
any work at all on a given range; without them the exhaust is described only by its cap.

**O-X3. The exhaust has no in-use theory, because the wheels have none.** The tree's own words:
"the missing instrument is the same at every rung, an in-use bound, found once and carried up".
Every tier above the first is used far below its period, so tier `k`'s behaviour on a range is
governed by the wheels' in-use laws - the gear zone, the quiet zone, the first-hit record above
it - and the one thing those do not give is an upper bound.
*Attack.* Inherited from O-W4, and by L65 an upper bound there is the twin-gap problem.
*Blocks the clutch?* **No, and it must not be a gate item**, for the same reason as O-W4: it is
the target, named, not hidden complexity.

**O-X4. The tower's own record is undefined.** Three objects already share the word "record"
(the motor's `F`, the wheels' longest closed run above the placement prefix, the clutch's
twin-free run - Appendix B.5 of the refiling). A fourth is implied by the stack and has never
been stated: the longest twin-free run left by tiers `1..k+1` on the quiet zone `(P_k, P_k^2]`.
*Attack.* State it; by X5 it is exactly the twin gap there, so the honest outcome is likely that
it is the conjecture at a sparse set of rungs - which should be recorded rather than discovered
twice.
*Blocks the clutch?* **No**, but leaving it unstated invites a rediscovery, which the record has
already suffered twice (docs/novel/README.md, the two 2026-09-04 branches).

**O-X5. Prior art: not checked, and no register entry.** As with the wheels, there is no
`docs/novel/README.md` entry for the stack, the cuts, the exhaust cap or stride containment. The
nearest classical objects named anywhere on the record are Eratosthenes/Legendre (sieving
`[1, z^2]` by the primes up to `z` leaves the primes) - which the exhaust cap IS, in machine form
- and the primorial ladder.
*Blocks the clutch?* **No** for correctness; the cap should be recorded as the known fact it is,
so that nothing downstream treats it as new.

---

# THE GATE

The owner's criterion for "fully understood": no hidden complexity from some aspect we did not
properly understand. Read strictly, and separating structural gaps from the conjecture itself.

### MOTOR - **NO**, but by a short list, and every item is a statement about `M` alone.

1. `L(M)` bounded, and with it `L_pad` (O-M1). The only unbounded ingredient of the motor's own
   grammar; every finite-depth statement about the motor is conditional on it.
2. The chain statement at `J = 3, 4` on the band `[15, 36]` (O-M2). The one object still open
   inside the window; the instruments (the CRT row search, the LP-duality closure, the
   configuration enumerator) exist and the band is already a certified object.
3. A monotone or contracting functional of the merge closure (O-M3). Named at node 4.i.b.ii and
   never run - the single unrun item on the motor.

Everything else open about the motor is the conjecture in disguise (the budget inequality, the
pair statement, `d_0 <= W`, the window statement, the adversarial covering number) or strictly
stronger than the root (`A(K)`), and none of those can be gate items.

### WHEELS - **NO**, and the gaps are in the counting theory, not the metric one.

1. ~~The vanishing moments (O-W1)~~ CLOSED by top_machine_7.md L73/L74 (proved; verified to d = 26).
2. ~~`L31` as a formula (O-W2)~~ CLOSED by top_machine_7.md L69 (the loaded record rule, proved both
   directions). Open beneath it: the complexity of the core minimisation, and the non-cancellation
   of the top moment M_{r(d)} (L75, measured 0 of 24).
3. The kernel gap for the in-use theory and the counting theory (O-W5) - or an explicit decision
   that written proofs suffice for the clutch, recorded as such.
4. Prior art for the machine as an object, and register entries for documents 1-6 (O-W7).

NOT on the list, deliberately: an upper bound on the quiet-zone record. L65 proves it is the
twin-gap problem; it is the target, in the wheels' coordinate, and calling it a gate item would
put the conjecture inside the gate.

### EXHAUST - **NO**, and it is much the furthest behind: it is the only object with no
measurement and no branch document of its own.

1. ~~Slot A landing~~ CLEARED 2026-09-06: `proofs/MachineStack.lean` gated, X3-X6 and the zone
   laws are KERNEL.
2. ~~`CutMono` unconditional from `q = 5` (O-X1)~~ CLOSED by exhaust_1.md X12/X13 (written proof
   from dyadic Bertrand: cut_{k+1} > 4 cut_k for prime q >= 5; the base step fails at q = 2, 3, 4);
   kernel transcription DONE round 36: `cut_succ_gt_four_mul`, `cutMono_of_five_le`
   (proofs/MachineStack.lean); `stack_open_iff_twin` unconditional for prime `q >= 5`.
3. ~~The zones and the redundancy lemma (O-X2)~~ CLOSED by exhaust_1.md X14-X16 (range form: any g
   with g^2 > N, every multiple in [1, N] is g or has a prime factor below g; 0 exceptions in
   7,357,725 strikes).
4. ~~The tower's own record stated (O-X4)~~ CLOSED as ROOT by exhaust_1.md (family (1, 1) of a
   tier is the twin primes above its top gear; X24 the twin counts are identical across splits).
5. ~~One measurement pass of any kind~~ DONE: exhaust_1.md (X9-X24; 18,095,756 residues, 0
   exceptions on the self-similarity; the home/echo census; the regime law X20). Remaining on the
   exhaust: CutMono into the kernel; prior art for X10 and X13 (O-X5); the crossover height of
   X20 (O-X6, needs an upper bound on a prime gap, a known theorem).

**Gate verdict: the gate is NOT open.** The motor is close (three items, all instrumented); the
wheels are close in metric and short in counting (four items, one of them a live lane); the
exhaust is not yet an investigated object.

---

# INTERFACE OBJECTS ALREADY VISIBLE

For the future clutch, not to be worked now. Named only, with one line each.

**The named interfaces.**

- **The conjugacy `n -> 6^{-1}(n + 1)`** (document 1 L19, KERNEL `conjugacy`, `exists_column`,
  `conjugacy_census`): carries the wheels' open-pair set exactly onto the same gears' opening set
  in the motor's column coordinate, so every counting and symmetry law is common property and
  every metric law is the wheels' own.
- **The family decomposition `(s, s')` of the quiet zone** (document 6 L60): every open pair of
  `(Q, Q^2]` carries a label of smooth cofactors with `gcd | 2`, and the motor - whose gears are
  exactly the primes `<= q` - strikes every family but `(1, 1)`, which is the twin primes above
  `Q`.
- **The echo set**: on `(C, C^2]`, the numbers an exhaust gear strikes that a gear at or below the
  cut already strikes; the exhaust's whole action there apart from its home strikes.
- **The home strikes**: a top gear striking its own home column `h(g) = round(g/6)`, a strike that
  is not a kill (the member is the prime `g` itself); doubly occupied placements number the twin
  pairs in `(q, Z]`, and "twins = both-open + home-only" is exact.
- **The corridor** (docs/proofs/14, `Corridor.*`): the 15 residues `E_35` with its endpoint,
  adjacency and padding laws - the motor's positional interface, which constrains where and never
  how big.
- **The effective-machine theorem (E)** (position_frontier.md): a column above `q` is blocked
  under `{5..q}` iff blocked under `{5..floor(sqrt(6k+1))}`, with the exception set exactly the
  twin gear pairs on their home columns.

**The 36 clutch facts of `refiled_by_object.md`, by name.**

| # | name | one line |
|---|---|---|
| C1 | the route | twins infinite iff for every bound some `{5..y}` has an opening in `(y, y^2]`; inside the window an opening IS a twin pair |
| C2 | the window is the clutch's zero-interaction region | a top gear's proper strike on a bottom-open column has a member above `q^2`, so the top machine cannot kill anything in the window |
| C3 | twins = both-open + home-only | exact over 38,889,216 columns; the home-only cell IS the doubly occupied placements |
| C4 | the four cells | at `q = 23`: 895,791 / 7,056,384 / 4,150,311 / 25,079,659, with all the coupling in the both-open cell (0.83-1.02 of independence) |
| C5 | runs inside the cells | both cells needing the bottom open have longest run exactly 2 (gear 5 alone); the both-closed cell reaches the bottom record to `q = 19` and falls short at 23 |
| C6 | the twisted copies | the top machine's action on the bottom's openings is a union of coherent twisted copies of the bottom at separation `2/g` |
| C7 | level of distribution 1, exactly | `|X_g - 2N/g| < 2·3^m` at 2,338 cells, pairs and triples the same; true growth `2^m` |
| C8 | the survivor curve | 1.0000 at `s >= 4.27`, minimum 0.8603 at `s = 2.09`, `0.79305(1 + c/ln Z)` at `s = 2` |
| C9 | the placement residue law | home columns meet each non-tooth class of a bottom gear twice and each tooth class once, so placement is dimension 1 and double occupancy dimension 2 - the parity barrier, named |
| C10 | home columns are unstruck on the gear's own side | 0 violations in 15,549 checks |
| C11 | the origin law | both machines are mirror-symmetric about column 0, and from `q = 19` the longest both-open stretch of the whole period starts there |
| C12 | the twin-free record is a joint object | 24, 82, 153, 254, 501 columns, 1.85-3.66 times the sum of the two machines' own closed records |
| C13 | nothing at the period scale gives the window | twin-free stretches of 502 columns exist inside the period against a window of 83 |
| C14 | the bilinear `g-m` switching | `E = 2D + Q` exactly at all five `q`: switching gives an identity, not an inequality |
| C15 | Brun on the clutch | the order-2 truncation error equals the order-3 term; exactness buys nothing |
| C16 | the layer law | a composite in `(y^2, y'^2)` has a prime factor below `y` or is `y c` with `c` prime |
| C17 | the square gate | the deepest hopping layer of the walk from `q^2` is the top gear iff `q^2 - 2` is prime (153 open, 514 shut) |
| C18 | the walk from `q^2` and the top gear's single strike | the walk starts on a tooth of the top gear, which strikes it exactly once and is inert on it thereafter |
| C19 | the near-twins, at most three per rung | a new gear bites at most three isolated spots of its section, with new twins between them |
| C20 | the island witness | for every integer coprime to 30 above 2849 some island of `[1, d)` is open, 0 exceptions to 200,000 |
| C21 | the reachability landscape | the `q`-free set of gears that can reach each offset: bar size in closed form, islands as CRT classes, the doubling law |
| C22 | the cover number `K(d)` | exact at 23 arcs to `d = 1330`, growth `d/(ln d)^3`; a cover is `2^K` classes modulo a product above `q^2`, but there are `2.7^m` covers |
| C23 | the real separation does not drive `K` | `K_real` is the mode of the random-separation distribution at all six arcs |
| C24 | the square phase vector is irrelevant | real, locally-square and random vectors fail the island witness at the same rate (0.9984 ± 0.0033) |
| C25 | each gear's in-window take is one curve | the take follows a curve in `ln g / ln Q'` alone, with white residual, and belongs to the range, not to any family |
| C26 | the anchor's rigidity in the window | `{5..13}`'s openings miss their fair share modulo any higher gear by fewer than 30 in every window to `Q = 5000` - and it is exhausted at the first gear above the anchor |
| C27 | the structured-families identity | every located family carries twins at the window's own rate; a rule written in the lower machine's residues cancels its own saving |
| C28 | the frontier reduces the window statement to `d_0 <= W` | from `q = 1427` the longest blocked run of `[1, W]` is the initial run, and by (E) the top machine is irrelevant inside the window |
| C29 | the window has at most two junctions | the column of `q'` (iff `q'` is a twin member) and the column of `q'^2` (iff `q'^2 - 2` is prime), 0 mismatches at 152 rungs |
| C30 | `d_0` | the column of the first twin pair above the top gear at every level to 33,317, `d_0 <= q'`, inside the window by 10-58x |
| C31 | the machine feeds on itself | the next level's walk starts at `6k^2 - 2k`; the level-free transfer rule names the admissible gears with neither `k` nor `g` in the condition |
| C32 | what holds up a window stretch against a period record | the period record needs every gear; the window's longest stretch needs a chosen fifth, ordered by position not size |
| C33 | `F_W` is the largest twin gap in `(q, q'^2)` | the window's longest stretch is a twin-gap statement, not a machine statement |
| C34 | first and second moment for the island witness | the `s = 2` correction repairs the first moment (16.51 against 17); the second is dead by proof - a bound below 1 on the failing fraction is the conjecture |
| C35 | the faces of the wall, sorted by object | A and A4 belong to the method, B and C to the motor, D and E to the clutch |
| C36 | what the sorted faces leave | the adversarial covering number, a motor-family statement, is what the transfer and over-asking faces do not need |
