# Written proofs

One file per theorem, in the order the machine is built: the route, one gear, one machine,
adding a gear, the record of the bigger machine from the smaller, bounds on the alignment
depth, the corridor, the layer, and the rung-level certificates.  Each file opens with a
plain-words paragraph (no symbols), then vocabulary (column, gear, opening, window, record) with
its classical translation, a precise statement, a complete numbered proof, a Status line naming
the Lean kernel theorems where they exist and what is verified computationally, a
"Relationship to the conjecture" paragraph saying what the result does and does not establish
and whether anything measured enters, where the result is used, and its source on the record.
Nothing measured or conjectured is presented as proved; where a proof on record has a gap it is
marked in the file.  Every file now also carries a "Prior art, and what is new" section, placed
after its Status, separating what the proof leverages from the literature, what has no located
prior art, and what is a known result restated in the machine's vocabulary; where no prior-art
check has been run the section says so.

| # | File | Statement (one line) | Kernel | Bears on the conjecture how | Prior art |
|---|---|---|---|---|---|
| 1 | [01-the-route.md](01-the-route.md) | Twin primes are infinite iff for every bound some machine `{5..y}` has an opening in the window `(y, y^2]`; inside the window an opening is a twin pair; the gap form | yes: `BlockedSlots.twins_infinite_iff_survivor_in_window`, `survivor_iff_twin`, `survivor_in_window_of_gap_bound`; `Horizon.exists_prime_factor_lt` | The route itself, lossless; makes the record `F(M)` the object | Ziller-Morack 2017 Thm 4.1; Mercer 2018 |
| 2 | [02-tooth-rule.md](02-tooth-rule.md) | Gear `q` strikes `k` iff `k = +-6^{-1} (mod q)`; the two arcs and alternating spacings `2u`, `q-2u`; the shield; teeth never adjacent; twin gears share the tooth `(p+1)/6` and both strike the column of `p(p+2)` | yes: `TwoTeeth.kill_spacing`, `kill_period`, `teeth_letters`; `AnchorChain.neighbour_of_hit`; `Polignac.twin_product_slot`, `twin_split_class_iff`, `own_slot_pin_gap_two` (the iff of the tooth rule itself is definitional) | Bookkeeping: one gear, exactly | Standard CRT; Clement 1949 modulus |
| 3 | [03-always-open-columns.md](03-always-open-columns.md) | Column 0 and the antipode `(P+-1)/2` are always open; the mirror `k -> -k`; the symmetry group of the opening set is `Z/2`; every gap length `>= 2` occurs an even number of times, so the record never occurs exactly once | partly: `Mirror.mirror_gear`, `antipode_open`, `self_mirror_unique`, `even_card_involution`, `window_count_even`; symmetry group written | Bookkeeping; a parity lever worth exactly a factor 2 | Holt-Rudd remark (v), one class |
| 4 | [04-alignment-law.md](04-alignment-law.md) | The longest run of consecutive openings equals the long arc `q_0 - 2u_0 - 1` of the smallest gear; with gear 5 the openings are isolated points and dominoes, `prod(q-4)` dominoes | no (written CRT proof; verified on 103 gear sets) | Alignment somewhere in the period; nothing about spacing in the window | Standard CRT; HL local factor |
| 5 | [05-adding-a-gear.md](05-adding-a-gear.md) | The `q'` copies realise every deletion phase once and each opening dies in exactly two; hit law; chain law (`y - x = 0, +-d`); merge law; the grammar T1-T5 (alphabet `{a, b}`, residue necessity, strict alternation, spacing `>= 2u`, fuel cap); legal words | yes: `AnchorChain.copy_phase`, `phase_bijective`, `chain_law`, `hop_zero`; `MergeLaw.interior_gap_mod`, `newgap_le_step`; `TwoTeeth.kills_gap_ge`, `fuel_span_cap`; `WordLegal.legal_iff_noRepeat`, `killable_iff` | Exact machinery of one step; bounds nothing by itself | Holt-Rudd cycle recursion, one class |
| 6 | [06-saturation.md](06-saturation.md) | If `F(M) < 2u_q` (in particular if `3F(M) < q - 1`) then `F(M+q) = F_2(M)` exactly | no (written; ingredients kernel-checked) | Closed, but in a regime disjoint from every rung | None found; two-class threshold new |
| 7 | [07-deletion-ladder.md](07-deletion-ladder.md) | `F_{r+1}(M) <= F(M + any r new gears)`, in particular `F_2(M) <= F(M+q')` | no (written CRT proof) | Lower bound on the next record; circular as an induction step | Holt-Rudd Thm 2.3, one class |
| 8 | [08-attainment-identity.md](08-attainment-identity.md) | `F(M+q') = max(F_2(M), max_{J>=3} Q*_J(M; q'))`: a legal word is always struck in full somewhere (`>=`), and every new gap is a merged run with legal middles (`<=`) | partly: `WordLegal.killable_iff`, `chain_iff_word`, `MergeLaw.newgap_le_step` (qualifying-floor form), `AnchorChain.phase_bijective`; the identity itself written | Relocates the size statement to the old machine; no slack, no progress | Holt-Rudd Lemma 2.1 overlap; identity new |
| 9 | [09-record-law-and-nested-formula.md](09-record-law-and-nested-formula.md) | Phase reduction: `F(M+g)` is the max over `g` phases of the largest gap of the phase-`r` sequence on one lower period; kernel-checked at both ends at machine 17 (`F(17) = 18`); the nested next-opening formula `next_G = next_M^{k+1}` past a run of `k` hits | yes: `AnchorRecord17.surv_shift`, `record_max`, `F17_eq_18`; `AnchorChain.hop_iter`, `hop_zero`, `hop_one`; general phase reduction written | Computation of records; depth cap measured | Holt-Rudd copies; phase record law new |
| 10 | [10-word-reduction.md](10-word-reduction.md) | `Q*_J > -inf iff L(M) >= J - 2`, so `J_max = L + 2`, `A_kill = L + 1`; chain iff legal word; the same-tooth lemma (middle span `= 0 mod q'` iff an even number of non-padded middles) | yes: `WordLegal.chain_iff_word`, `qstar_iff_word`, `jmax`, `akill`, `same_tooth`, `same_tooth_window`, `literal_even_span` | Renames the open depth question as `L` bounded; does not answer it | Ziller 2020 D(k), one class, length one |
| 11 | [11-spectrum-bound-on-L.md](11-spectrum-bound-on-L.md) | `L(M) <= 2 floor((F(M+q') - 2)/q') + 1 <= 2F(M+q')/q' + 1`, with the letter-aware and parity forms | no (written; from attainment and T3) | The only bound on `L`; it grows with `F/q'`; closure needs the open padded constant | Prior art not checked |
| 12 | [12-bare-word-cap.md](12-bare-word-cap.md) | `L_bare(M) <= PSORD(q' mod 210) <= 5`; `PSORD in {1, 2, 3, 5}` with the 28-class set `S = {PSORD <= 2}`; never 4 | yes: `BareAlt.no_bare_run_ge`, `bareAlt_inadmissible_iff`, `S_card`, `psord_le_five`, `psord_ne_four`, `inadmissible_iff_capC` | Uniform cap on the bare half of `L`; `L_pad` untouched and unbounded on record | Prior art not checked |
| 13 | [13-literal-and-polignac-caps.md](13-literal-and-polignac-caps.md) | A literal chain has at most `capC(q mod 210) <= 6` exposed members, table exact, no class of cap 5; over all even gaps the cap depends on `gcd(e, 105)` and is at most 12 | yes: `LiteralCap.literal_chain_le_six`, `cap_six_classes_sharp`; `LiteralCapTable.cap_table_maximal`, `cap_table_realized`, `no_cap_five`; `PolignacCap.cap_gcd_*`, `capOf_le_twelve` (the `gcd` reduction written) | Same object as 12 from the gear's side; literal chains only | HL local factor known; caps new |
| 14 | [14-corridor.md](14-corridor.md) | `E_35` (15 residues); endpoint and adjacency laws (294 forbidden pairs); tier A carriers; the completeness lemma `q <= 2n`; the 32-cap on prime-adjacent runs; the adjacent-gap exclusion law mod 5 (6 classes, complete); the AP lemma; padding onset, count and the 12 forbidden equal-padding classes | partly: `Corridor.exposed_iff_mem`, `forbidden_pairs_count`, `prime_adjacent_run_le`; `TierA.no_chain_of_carrier_empty`, `equal_padding_forbidden_classes`, `padding_shape_dichotomy`, `onset_gate`; completeness, exclusion law, AP lemma written | Constrains where, never how big; pruning only | Standard CRT; classification new |
| 15 | [15-layer-and-shadow-laws.md](15-layer-and-shadow-laws.md) | A composite in the layer `(y^2, y'^2)` has a prime factor below `y` or is `y c` with `c` prime; a gear exposes nothing below its own square; one gear's ledger line counts partner primes exactly below `q^3` | yes: `Layer.layer_novelty`, `minFac_lt_or_eq`, `eq_mul_prime_of_minFac_eq`, `slot_cap`; `Gear.R_eq_zero_of_below_sq`, `sq_le_of_minFac_eq`, `mem_partners` | Bookkeeping per layer; no bearing on the record | Standard (Eratosthenes/Legendre) |
| 16 | [16-peel-triple-middle-sum.md](16-peel-triple-middle-sum.md) | Peel bound `Q*_J <= Q*_{J-1} + min flank`; triple inequality `g_L + w + g_R <= F_2 + min(g_L, g_R)` (hypothesis-free); middle-sum lemma (literal middles sum to `>= k q'` or `k q' + a`); even-`J` literal runs are never palindromes | no (written; a conditional step in the recorded consequence is flagged) | Reductions inside the per-`J` family; `Delta_J` bounds stay measured | Holt-Rudd Lemma 3.1 overlap |
| 17 | [17-increment-law-literal-steps.md](17-increment-law-literal-steps.md) | At the six literal steps 11->13 .. 29->31: an exhibited adjacent pair of the old machine and a bound on every gap of the new machine give `F(M+q') <= F_2(M) + s_min(q')`; not a general theorem, false at 31->37 | yes: `Increment.increment_law_literal_steps`, `f2_11 .. f2_29`, `IncCert23.F_le`, `IncCert29.F_le`, `IncCert31.F_le`, `f2_19_sharp`, `f2_23_sharp`, `f2_29_sharp` | Six certified rungs; no induction step; general law measured and refuted on the family | Unasked in print, either class count |
| 18 | [18-case-split-31-37.md](18-case-split-31-37.md) | Every 95 consecutive columns contain an opening of machine 37, so `F(37) <= 95 = F(31) + 37`: 385 exhaustive held-phase cases, each an exact integer dual certificate of the covering relaxation | yes: `CaseCert37.F_le`, `D_31_37_case`, `no_run`, `nocase0 .. nocase384`, `nocov0 .. nocov384`; `CaseSplit.lowest7`, `le_mxr`, `le_mxr2` | Rung ten of the budget inequality, hypothesis-free; no induction step | LP duality, Kounias, Sherali-Adams |
| 19 | [19-prover-lemmas.md](19-prover-lemmas.md) | Prover A: the trivial discharge `g_L + g_R <= F + min flank`; the column-0 equivalence PS at 0 iff `F >= 2d_0 - q'`; the re-phasing lemma and sole-striker corollary.  Prover C: (L) iff the incoming tooth is `round(q'/6)`; (T) iff separation `>= 2`; (T) pins gear 5 | no (written proofs only) | Bookkeeping around the open pair and chain statements; locates twin existence at column 0 | Prior art not checked |

Kernel names are cited as `Namespace.theorem`; the files live in `proofs/*.lean`
(`BareAlt.*` is `proofs/BareAlternation.lean`; `AnchorRecord17.record_max` is in
`proofs/AnchorRecord17Core.lean`; `Machine13.spectrum_one` and its siblings are in the
`Machine*Q.lean` files; the 385 case leaves are `proofs/CaseCert37C0.lean .. C384.lean` with
tiers `T0 .. T34`).

## Not proved, and said so

The budget inequality `F(M+q') <= F(M) + q'` is the target, certified rung by rung (files 17,
18 and the `Machine*` ladders), never a law.  `L(M)` bounded, `L_pad` bounded, `D_g = A_kill`
bounded, `Delta_J <= s_min` and `Delta_J = O(1)`, the pair statement and the chain statement are
open; the increment law is a theorem only at the six literal steps of file 17.  Of the nineteen
files, the ones that bear on the open size statement at all are 11 (a growing bound on `L`),
12-13 (a uniform cap on the bare half of `L`), and 17-18 (certified rungs); the rest is exact
machinery or bookkeeping, and each file's "Relationship to the conjecture" paragraph says which.

## Running the kernel audit

From the repository root:

    cd proofs
    lake env lean AxiomCheck.lean

`AxiomCheck.lean` imports every library of the corpus and runs `#print axioms` on the named
theorems.  Every line must list only `propext`, `Classical.choice` and `Quot.sound` (the
standard axioms); a line containing `sorryAx` would mean an unproved declaration and must not
appear.  The heavy per-case files (`CaseCert37C*`, `IncCert*`) are built by `lake build` per
module; the audit only reads their compiled results.  The lakefile's `defaultTargets` lists
every module; `proofs/DepAudit.lean` gates the independence claims (a bound derived on one
machine with the next machine's period nowhere in the derivation).
