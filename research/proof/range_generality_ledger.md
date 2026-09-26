# Range line: generality ledger

Date: 2026-09-26. This ledger covers the range line up to node R5.f.xxxv.c.xxxv. It includes the Lean kernel (the 13 Range modules and the new RangeGen1 to RangeGen5) and this round's adjudications of the claimed items C1 to C6.

**The owner's rule.** A result counts only if it holds for every machine size q, and for every gear g, copy j and gap d that the statement quantifies. A statement proved or checked only at particular q is instance data. Computation at particular q is used only to check a general statement or to refute one.

**Sources.** Every entry comes from one of these:
- the generality audit of 2026-09-26;
- this round's adjudications of C1 to C6;
- the Lean lane reports and the review of RangeGen1 to RangeGen5;
- `range_line_map.md`;
- the round records in this folder: `shelves_cofactors_2026-09-23.md`, `range_two_paths_2026-09-23.md`, `derived_machine_2026-09-24.md`, `range_kernel_2026-09-25.md`, `range_kernel2_2026-09-25.md`, `field_to_range_2026-09-25.md`, `mirror_runs_2026-09-25.md` and `runs_gain_2026-09-26.md`.

No entry is new.

## At a glance

| Section | Contents | Entries |
|---|---|---|
| 1 | Holds for every q and is a kernel theorem in Lean | 18 (5 of them new this round: RangeGen1 to RangeGen5) |
| 2 | Holds for every q, proved on paper or in a scratch Lean file, not yet in the kernel | 4 scratch-Lean groups, 13 paper groups, 1 list of counting or locating results |
| 3 | Stated for every q, not proved | 11 claimed statements, plus the open items recorded this round |
| 4 | False as a general statement, with a counterexample | 43 |
| 5 | Instance data: checks only, not proofs | 12 groups |

What moved this round:
- P1 to P5 (height split, gain, mirror pairs under acting, total-blame anatomy, acting against acting-free) moved from paper to the kernel.
- C4 and C6 moved from claimed to proved.
- C1, C2, C3 and C5 were split into proved parts, refuted parts and open parts.
- C7 to C12 were not examined this round and stay in section 3.

---

## Terms

**Numbers and the machine**
- **q#**: the product of all primes up to q. For example, 7# = 210.
- **Gear**: a prime ≥ 7. The primes 2, 3 and 5 lay out the copies and never divide a leg.
- **Machine q**: the gears 7..q. The *machine size* is q.
- **q'**: the next prime after q.
- **M** = q#/30, the product of the gears 7..q. M is odd for q ≥ 5.
- **a** = q#/2, and **ρ** = q#/2 + 1 = 15M + 1.

**Copies**
- **Copy j**: the pair (30j − 1, 30j + 1).
- **Legs**: the lower leg 30j − 1 and the upper leg 30j + 1.
- **Strike**: gear g strikes copy j when g divides a leg.
  - For j ≥ 1 this is j ≡ ±a_g (mod g), where a_g = 30⁻¹ mod g.
  - Equivalently, g divides 900j² − 1.
- **Teeth** of g: the copies that g strikes. They form two classes mod g, t_g = 15⁻¹ mod g = 2a_g apart.
  - **σ_g** is the least absolute residue of t_g.
  - A **lane** is a class of copies mod G.
- **Acts**: gear g acts on copy j when g² ≤ 30j + 1. For a struck leg g·c, this is the same as g ≤ c.
- **Home copy** of g: a copy with a leg equal to g. This is the one kind of strike by g that cannot act.
- **Revealed** (copy j ≥ 1): no acting gear strikes it. Equivalently, both legs are prime; such a copy is a *twin copy*.
- **P(j)**: the least prime factor of 900j² − 1 = (30j − 1)(30j + 1).
- **Survivor** of machine q: a copy struck by no gear 7..q.
  - **S_q** is the set of survivors.
  - **S_X** is the set of copies with no prime of [7, X] dividing a leg.
  - A class mod M made of survivors is a *survivor class*.
- **Upper gear** of machine q: a gear above q.
- **Range** of machine q: the copies j with q < 30j − 1 and 30j + 1 ≤ q#.
- **Window** of machine q: the range copies with 30j + 1 < q'².
- **Range(q)**, the range statement at q: some range copy is revealed.
- **RANGE**: Range(q) holds at every prime q ≥ 7.
- **RangeStatement q** (Lean): some twin pair (p, p + 2) has q < p and p + 2 ≤ q#. This also admits twin pairs that are not copies, such as (41, 43).

**Cut-offs and halves**
- **Q**: the largest prime ≤ √ρ.
- **Q\***: the largest prime ≤ √(q#).
- **U⁻**: the primes in (q, Q].
- **B**: the primes in (Q, Q\*].
- **U_q**: the primes in (q, isqrt(q#)]. For q ≥ 7, U_q = U⁻ ∪ B.
- **Low half**: copies 1 to (M − 1)/2.
- **High half**: copies (M + 1)/2 to M − 1.
- **Loss_q**: the twin copies whose lower leg lies in (q, Q].
- **Gain G_q**: the copies of S_Q in the high half that are not in S_{Q\*}.

**Mirror pairs**
- For odd d < M, the **low member** is s = (M − d)/2 and the **high member** is s' = (M + d)/2.
  - Their legs are L1 = 30s − 1, L2 = 30s + 1, H1 = 30s' − 1 and H2 = 30s' + 1.
  - The **mirror** of copy x is M − x.
- **Survivor pair**: both members are survivors.
- **Deleted**: struck and acted on by some upper gear. **Doubly deleted**: both members are deleted.
- **AB_q**: the primes g > q that strike and act on both members of some survivor pair.
- **Sole striker**: a gear that is the only acting upper striker of both members of a survivor pair.
- **E_g and m_g**: for a U⁻ prime g with q#/2 − s = g·m (s = ±1, so g divides q# − 2s), m_g = m and E_g = m − g.
- **Band** of g: the odd multiples d = gk with 15d ≤ ρ − g², equivalently 15k ≤ E_g. Write w_g = (ρ − g²)/15.

**Twin nodes and chains**
- **Twin node**: the lower member s of a twin pair (s, s + 2). **s⁺** is the next twin node above s.
- **Rung** of s: a twin node r with s < r and r + 2 ≤ s#.
- **Good** node: a node that has a rung, equivalently s⁺ + 2 ≤ s#.
- **Reach from 29**: the nodes reached from 29 by steps s → s⁺, where a step is allowed when s⁺ + 2 ≤ s#.
- **NS**: every good node s has s⁺⁺ + 2 ≤ s#.
- **Serves**: a revealed copy (p, p + 2) serves machine q when q < p and p + 2 ≤ q#.
- **σ(y)**: the least prime whose primorial is ≥ y.
- **chain(S)**: list S by lower leg, p_0 < p_1 < …. Then chain(S) holds when S is infinite, p_0 + 2 ≤ 210, and p_{k+1} + 2 ≤ p_k# for every k.

**Missed copies, shelves, rows**
- **Case** of gear g: case 1 when g² ≡ 1 (mod 30), and case 19 when g² ≡ 19.
- **Missed copy c_g**:
  - case 1: copy (g² + 29)/30, with legs g² + 28 and g² + 30;
  - case 19: copy (g² + 11)/30, with legs g² + 10 and g² + 12.
  - **A_g** and **B_g** are the two leg offsets.
  - **J_g** is the index of c_g.
  - A **revealed gear** is a gear whose missed copy is revealed.
- **Kill**: gear h kills c_g when h divides a leg of c_g.
- **R8(q, r)**: machine q's range holds a missed copy c_g that none of the r gears just below g kills.
- **Shelf(g)**: the copies [j_g, j_{g'}), where j_g = ⌈(g² − 1)/30⌉ and g' is the next gear.
- **Region** of g: [J_g, j_{g'}).
- **Row h**: a prime h ≥ 7, read as a striker of missed copies.
  - **E_h**: the residues of g mod h at which h strikes c_g.
  - **e_h**: the number of unit residues in E_h. A row with e_h = 0 is **inert**.
  - **D_h**: the dormancy set of derived_machine 2.2.
- **Live row** h on g: 7 ≤ h < g, e_h > 0, and g mod h is not in D_h.
- **λ(g)**: the largest live row on g.
- **d0(g)**: the least even d ≥ 2 with (d + 1)² ≥ 2g + 1 − B', where B' is the case constant of derived_machine 2.2.
- **Cap(P)**: the gears g > P with λ(g) ≤ P.
- **Escape class** at level X: a class of gears mod 30·∏_{7≤h≤X} h, all of one case, in which no row h ≤ X strikes c_g (for g > X).
  - A class is **doomed** when it holds no revealed gear.
  - **Desc_C(P')** is the set of escaping descendants of class C at level P'.

**Runs and struck stretches**
- **One-gear run** of an upper gear G at machine q: consecutive survivors x + r_0 < … < x + r_N (r_0 = 0), all struck by G, with G acting. The exact conditions are in 2B.3.
- **h(q + {G})**: the length of the longest stretch of consecutive copies each struck by some gear of {7..q} ∪ {G}.
- **h2(q)**: the same length for the gears 7..q alone.
- **Y_7(q)**: the longest stretch [1, Y] that can be covered by one freely chosen class for each prime in 7..q.
- **j(n)**: Jacobsthal's function.
- **B_1(q)**: the longest one-gear run at machine q.
- **f(K), K(n), T_G(N), E_7(G, N)**: the gear-7 lane counts, defined in 2B.10.

**Record symbols.** Entries 2B.4 to 2B.7 and some rows of sections 4 and 5 use symbols defined only in the record they cite, such as m_r, v(m), tail(r), Anc, S_1, Γ_q, exposure, lap, stretch and clearance class. They keep that record's meaning.

**Status words and flags**
- **GENERAL-PROVED**: the argument is valid for every q, and for every gear, copy and gap as the statement quantifies.
- **GENERAL-CLAIMED**: stated for every q, but the argument is incomplete.
- **INSTANCE**: holds at particular q only. It is a check, not a proof.
- Where each proof lives:
  - **[Lean kernel]**: a kernel module in `C:/dev/primes/proofs/`.
  - **[Lean scratch]**: a Lean file compiled this round in the scratch folder `C:/Users/Alex/AppData/Local/Temp/claude/C--dev-primes/a1c3a0ad-3acd-4fbe-b091-1e36256528ec/scratchpad/`, not added to the kernel. All of them compile with the standard axioms only.
  - **[paper]**: the argument is written in a record or an adjudication.
- **[imports X]**: the proof uses the outside theorem X.
- **[counting]** and **[locating]**: general statements of a type the working rules exclude as results. Counting means budget or density bounds. Locating means bounding the height of a revealed copy, reducing to a window, or using a tower of smaller machines.
- **[certificate]**: a finite computation that is one step of a general proof, not a sample.

---

## 1. General and in Lean (kernel theorems)

Every entry holds for all q from the stated lower bound. None has an upper bound on q or needs q to be prime.

The kernel checks:
- 0 sorry, admit, axiom or native_decide in the 18 Range modules.
- Axioms used: only propext, Classical.choice and Quot.sound.

The reviewer rebuilt RangeGen1 to RangeGen5 one module at a time, re-elaborated each from source, and checked axioms on all 59, 90, 90, 50 and 54 declarations. Each module is registered as a `[[lean_lib]]` block in `proofs/lakefile.toml`.

### 1.1 The range statement gives unbounded twins (RangeHandoff)
- If RangeStatement q holds at every prime q ≥ 7, then twin pairs exist above every N.
- A twin pair with q < p and p + 2 ≤ q# witnesses RangeStatement q.
- One pair with p + 2 ≤ q₁# and q₂ < p serves every q in [q₁, q₂].
- Lean: RangeStatement, range_implies_unbounded, twin_serves, serves_interval_of_le. serves_interval is the same as serves_interval_of_le with three unused hypotheses.

### 1.2 RANGE as reach along twin nodes (RangeReach)
- Three statements are equivalent:
  - RANGE (RangeStatement at every q ≥ 7, or at every prime q ≥ 7);
  - every twin node s ≥ 29 is good;
  - the reach from 29 is unbounded.
- For every s: good ⇔ RangeStatement s ⇔ s has a rung.
- The reachable nodes form an initial segment of the nodes.
- Lean: range_iff_all_good, rangePrime_iff_all_good, good_iff_rangeStatement, good_iff_exists_rung, range_iff_reach_unbounded, reach_unbounded_iff, goodReach_iff, GoodReach.initial, all_good_of_rangePrime, rangeAll_of_all_good.

### 1.3 Anatomy of a failure (RangeReach)
- A first bad node t has a good predecessor a whose only rung is t.
- Some node is bad ⇔ some node a ≥ 29 has a single rung t, and t is bad.
- Passing from a to a⁺ loses at most the rung a⁺.
- Rungs transport upward: Rung s r ⇔ Rung t r for s ≤ t < r with r + 2 ≤ s#.
- NS ⇔ every node ≥ 29 has two distinct rungs.
- NS ⇒ RANGE ⇒ twins unbounded.
- 149 is a rung of both 29 and 59, so a node's parent is not unique.
- Lean: first_bad, bad_iff_terminal, drop_at_most_one, rung_transport, overlap_agree, rungs_initial, rung_ge_next, ns_iff_two_rungs, ns_iff_no_single_rung, second_of_two_rungs, all_good_of_ns, range_of_ns, rangePrime_of_ns, twins_unbounded_of_ns, ns_of_nsAll, not_parent_unique.

### 1.4 Chain (RangeChain)
- Take a strictly increasing sequence of twin lower legs with p_0 = 29 and p_{k+1} + 2 ≤ p_k#.
- It gives RangeStatement q at every q ≥ 7, prime or not, and hence twin pairs above every N.
- Lean: chain_implies_range_all, chain_implies_range, chain_implies_unbounded, primorial_le_of_le.

### 1.5 Copy form of the range statement and blame by the least prime factor (RangeWindowForm, RangeRegion)
- For every copy j ≥ 1: both legs are prime ⇔ 30j + 1 < P(j)². The negation is P(j)² ≤ 30j + 1.
- For every bound N_max: no copy in [1, N_max) has both legs prime ⇔ every copy there has P(j)² ≤ 30j + 1.
- For every q, these are equivalent:
  - some copy j ≥ 1 with q < 30j − 1 and 30j + 1 ≤ q# has both legs prime;
  - some such copy has 30j + 1 < P(j)².
- The negation of the copy form is total blame by the least prime factor.
- The copy form implies RangeStatement q. The converse fails, because RangeStatement also admits pairs off the copies, such as (41, 43) and (59, 61).
- Lean: P, P_eq_minFac_legs, revealed_iff_minFac, blamed_iff_minFac, total_blame_iff, range_copy_iff, no_copy_blame, copy_range_implies_rangeStatement.
- total_blame_iff_primorial is true, but vacuous for q ≥ 7, because copy 1 = (29, 31) lies inside its bound.

### 1.6 Height split of the range (RangeGen1, new; was P1)
The statements hold for every natural q ≥ 7, prime or not.
- A range copy j in [1, M − 1] is revealed ⇔ it lies in Loss_q, or in S_Q ∩ low half, or in S_{Q\*} ∩ high half. The three parts are disjoint.
- Half rules:
  - a low-half copy is revealed ⇔ it is in Loss_q or in S_Q;
  - a high-half copy is revealed ⇔ it is in S_{Q\*}.
- Range(q) ⇔ one of the three parts is non-empty.
- Not-Range(q) ⇔ Loss_q is empty and every low-half survivor x satisfies both of these:
  - some U⁻ gear p strikes x, i.e. x ≡ ±a_p (mod p);
  - some gear g of U⁻ ∪ B strikes the mirror M − x, i.e. x ≡ M ± a_g (mod g).
- The acting cut-off is Q on the low half and Q\* on the high half.
- Growth lemma (from Bertrand): every prime p with 7 ≤ p ≤ q has 2p² < q#, so p ≤ Q ≤ Q\*.
- Consequences of the growth lemma:
  - U⁻ ∪ B is exactly the set of primes in (q, Q\*];
  - S_Q = S_q minus the copies struck by U⁻;
  - S_{Q\*} = S_q minus the copies struck by U⁻ ∪ B;
  - S_q is invariant under the mirror.
- Leg heights:
  - a low-half upper leg is below ρ;
  - high-half legs lie in [15M + 14, q#).
- A range copy is revealed ⇔ q < 30j − 1 and both legs are prime.
- RangeCopy q ⇒ RangeStatement q.
- Lean (46 theorems):
  - main theorems: revealed_iff_split, split_disjoint, range_iff_split, not_range_iff_empty, not_range_iff, not_range_iff_residue, revealed_low_iff, revealed_high_iff;
  - cut-offs and strike sets: two_mul_sq_lt_primorial, le_cutLo_iff, le_cutHi_iff, uminus_or_band_iff, clear_cutLo_iff, clear_cutHi_iff;
  - mirror and residues: clear_mirror, strikes_mirror, strikes_iff_gearRes, strikes_mirror_iff_gearRes;
  - link to the Lean range statement: revealed_iff_twin_above, rangeCopy_implies_rangeStatement.
- Correction made in the formalisation: Loss_q is defined by the lower leg lying in (q, Q].
  - Under the reading "both legs ≤ Q", a twin copy with 30j − 1 = Q and Q + 2 a prime above √ρ would fall in none of the three parts.

### 1.7 Gain structure (RangeGen2, new; was P2)
The statements hold for every q ≥ 7.

(i) The gain and the revealed set.
- G_q is exactly the set of copies of S_Q in the high half that are struck by some gear h of B, i.e. j ≡ ±a_h (mod h).
- The revealed range copies are Loss_q ∪ ((S_Q ∩ [1, M − 1]) \ G_q), and the union is disjoint.

(ii) Composite legs. Let L be a composite leg of a copy in S_Q ∩ high half. Then:
- L = h·c with h = minFac L in B, c prime and h ≤ c < 2h;
- h acts, and c does not act when c > h;
- such a leg is composite ⇔ some gear of B divides it.

(iii) Pair form. j ∈ G_q ⇔ j = (hc − ε)/30 for some h ∈ B, ε = ±1 and prime c meeting all of these:
- h ≤ c ≤ (q# − 29)/h;
- c ≡ εh⁻¹ (mod 30);
- c ≢ 0 and c ≢ 2εh⁻¹ (mod p) for every prime p with 7 ≤ p ≤ Q.

The pairs of j are exactly (minFac L, ε, L/minFac L), one for each composite leg L = 30j + ε. So j has two pairs when both legs are composite, and one otherwise.

(iv) Trichotomy. Every low-half survivor x satisfies exactly one of three cases; in particular this applies to x = M − j for j ∈ G_q.
- (a) No U⁻ gear strikes x ⇔ x is revealed and not in Loss_q.
- (b) U⁻ gears strike x, but none acts ⇔ x is revealed and in Loss_q. Here every U⁻ striker is a leg.
- (c) Some U⁻ gear strikes and acts on x ⇔ x is not revealed.

(v) Shared class. Take a prime h > Q with h | q# − 2, an odd d < M with h | d, and let s be the low member of the mirror pair at d. Then:
- the lower leg of s is h·c\* with c\* < h and c\*² < ρ;
- h divides the lower leg of the high member;
- if s is a survivor, every prime factor of c\* is a U⁻ gear that acts on s.

The case c\* = 1 is not excluded.

(vi) Four classes. For an odd prime p that does not divide h, the classes 0, 2εh⁻¹, h⁻¹N and h⁻¹(N + 2ε) mod p are distinct ⇔ p ∤ N(N − 2)(N + 2).
- For N = q# and p > q, this is p ∤ q# ± 2, the requested form.
- The requested form is false for p ≤ q (see section 4).

Lean (71 theorems):
- (i) and (ii): gain_iff_band_strike, gain_iff_residue, revealed_iff_gain, loss_not_clear_cutLo, high_leg_factor, high_leg_two_primes, high_leg_band_iff;
- (iii): gain_iff_pair, gainPair_iff_leg, pair_count, strike_iff_pair_class, mirror_strike_iff_pair_class;
- (iv): trichotomy_exactly_one, low_trichotomy, quiet_striker_is_leg, mirror_trichotomy;
- (v): shared_class, shared_class_band;
- (vi): four_classes_distinct_zmod, four_classes_distinct_iff, three_class_rule, three_class_rule_band.

### 1.8 Mirror pairs: sums, shared strikers, height split (RangeMirror)
The statements hold for q ≥ 5 and odd d < M.
- Leg sums:
  - (30s ∓ 1) + (30s' ± 1) = q#;
  - L1 + H1 = q# − 2;
  - L2 + H2 = q# + 2.
- Shared strikers:
  - a gear dividing a leg of each member divides q# − 2, q# or q# + 2;
  - a prime g > q divides q# − 2 or q# + 2.
- Height split:
  - (30s + 1) + (30s' + 1) = q# + 2, so a gear with q# + 2 < 2g² acts on at most one member;
  - q#/2 = 30s + 15d;
  - the four-leg product is (a² − (15d + 1)²)(a² − (15d − 1)²).
- Lean: mirror_sum, legs_sum, legs_diff, shared_striker_cases, shared_striker, shared_striker_off, shared_striker_copy, shared_striker_copy_off, shared_striker_above, height_split, acts_on_at_most_one, half_primorial_eq, four_leg_product, four_leg_product_nat, odd_mirrorM (M is odd), thirty_mul_mirrorM (30M = q#).

### 1.9 Mirror pairs under acting (RangeGen3, new; was P3)
The statements hold for every q ≥ 5 and odd d < M.

(i) Strikes and survival.
- For every prime g ≥ 7:
  - g strikes s ⇔ d ≡ M ∓ t_g (mod g);
  - g strikes s' ⇔ d ≡ −M ± t_g (mod g).
- A machine gear strikes a member ⇔ g divides 225d² − 1.
- Both members survive or neither does. They survive ⇔ gcd(225d² − 1, M) = 1.

(ii) Acting windows.
- 30s + 1 = ρ − 15d and 30s' + 1 = ρ + 15d.
- g acts on s ⇔ 15d ≤ ρ − g². g acts on s' ⇔ g² − ρ ≤ 15d.
- If an odd g acts on s and an odd h acts on s', then g² + h² < q# + 2 and gh < ρ.

(iii) Shared strikes.
- For any g coprime to 30:
  - g divides L1 and H1 ⇔ g | q# − 2 and g | d;
  - g divides L2 and H2 ⇔ g | q# + 2 and g | d.
- A prime g > q never divides both legs of a cross pair, (L1, H2) or (L2, H1).
- So a prime g > q strikes both members ⇔ g | (q# − 2)(q# + 2) and g | d, and the two struck legs are of one type.
- It deletes both members ⇔ in addition 15d ≤ ρ − g².
- Acting on s alone forces g² + 15 ≤ ρ, so g ∈ U⁻.

(iv) Sandwich and core.
- The acting set of s ⊆ U⁻ ⊆ the acting set of s'.
- On the core d ≤ R, both acting sets equal U⁻. Here:
  - R = min((ρ − g₋²)/15, (g₊² − ρ − 1)/15);
  - g₋ is the largest prime ≤ √ρ, and g₊ is the least prime > √ρ.

(v) Leg classes.
- For any n: n divides L1, L2, H1, H2 exactly when 15d ≡ a − 1, a + 1, 1 − a, −1 − a (mod n) respectively.
- For a prime g ≥ 7, the four root classes of L1·L2·H1·H2 are distinct ⇔ g ∤ (q# − 2)q#(q# + 2).
- For g > q this becomes g ∤ (q# − 2)(q# + 2).
- When g | (q# − 2)(q# + 2), the product is 225d²(15d − 2)(15d + 2) mod g, with 0 a double root and three distinct classes.

(vi) Empty band. Take odd g ≥ 3 with g | q# ∓ 2 and g² ≤ ρ. These are equivalent:
- the band holds no multiple of g;
- ρ − g² < 15g;
- q#/2 ∓ 1 = g(g + e) for some odd e with 1 ≤ e ≤ 13.

Also, q#/2 ∓ 1 = g(g + e) ⇔ 2q# ∓ 4 + e² = (2g + e)². The existential square form does not make g prime.

(vii) Range through mirror pairs.
- A survivor j ≥ 1 is deleted ⇔ it is not a twin copy.
- Every j in [1, M − 1] is a member of exactly one mirror pair; no copy is its own mirror.
- The copy-form range statement ⇔ some survivor pair is not doubly deleted.
- One gear deleting both members needs g | q# ∓ 2, g | d and g² ≤ 30s + 1.

Lean (78 theorems):
- (i) and (ii): strikes_lo_iff_class, strikes_hi_iff_class, survives_lo_iff, survives_pair, acting_windows, acts_lo_iff, acts_hi_iff, joint_acting_sq, joint_acting_mul;
- (iii): same_minus_iff, same_plus_iff, no_cross_strike, strikes_both_iff, strikes_both_leg_type, deletes_both_iff, deletes_both_mem_uMinus;
- (iv): sandwich, core_eq_of_le_coreR;
- (v): leg_classes, four_leg_roots, root_classes_nodup_iff, root_classes_nodup_iff_above, three_classes;
- (vi): band_empty_iff, band_empty_minus, band_empty_plus, band_square_minus, band_square_exists_minus;
- (vii): deleted_iff_not_twin, range_iff_pair, deleted_lo_iff, single_gear_deletion, rangeCopyWindow_rangeStatement.

### 1.10 Anatomy of total blame (RangeGen4, new; was P4)
The statements hold for every q. Only the gcd form needs q ≥ 5.

- Survivors: for j ≥ 1, j ∈ S_q ⇔ gcd(900j² − 1, M) = 1 ⇔ q < P(j).
- (i) Blame sets. The non-revealed survivors of [1, M) are the disjoint union, over g ∈ U_q, of F_g = {j : P(j) = g, j not a home copy of g}. Gear g acts on every member of F_g.
- (ii) Revealed survivors.
  - A survivor j in [1, M) is revealed ⇔ every U_q gear that strikes it is one of its legs.
  - For H_q < j < M, where H_q = ⌊(isqrt(q#) + 1)/30⌋: revealed ⇔ no U_q gear strikes it.
- (iii) Joint redundancy. Take every survivor j ≥ 1 and every set D of U_q gears closed downward inside U_q (in particular the first k gears, for every k). Then j is struck by a gear of D ⇔ j is struck by an acting gear of D or is a home copy of a gear of D.
- (iv) Acting on a leg. For every copy j ≥ 1 and every divisor h ≥ 2 of a leg L: h acts ⇔ h² ≤ L ⇔ h ≤ L/h. Each leg has at most one non-acting prime factor.
- (v) Low strata.
  - Survivors with 30j + 1 ≤ q² are twin copies.
  - Suppose no prime lies in (q, q'). A non-twin survivor with 30j + 1 ≤ q'² has j = (q'² − 1)/30, a prime lower leg, and P(j) = q'.
  - Such a survivor exists ⇔ q'² ≡ 1 (mod 30) and q'² − 2 is prime.
- (vi) The next gear q' strikes a survivor j ≥ 1 of q only by acting, or at its home copy.
- Lean (42 theorems):
  - survivors: surv_iff_coprime, surv_iff_lt_P;
  - (i): nonrevealed_iff_blameF, nonrevealed_eq_iUnion, blameF_pairwiseDisjoint, blameF_acts;
  - (ii): revealed_iff_strikers_home, revealed_iff_no_strike;
  - (iii): joint_redundancy, joint_redundancy_first, joint_redundancy_set;
  - (iv): acts_iff_sq_le_leg, acts_iff_le_cofactor, nonacting_factor_unique;
  - (v): revealed_of_sq_le, stratum_nonrevealed, stratum_exception, stratum_iff, stratum_iff_nextGear;
  - (vi): strike_acts_or_home, nextGear_strike_acts_or_home.

### 1.11 Acting against acting-free; the fixed cut (RangeGen5, new; was P5)
The statements hold for every q and every cut X. Notation:
- G_X = the primes in [7, X], and P_X = ∏G_X;
- E_X(k): no acting gear of G_X strikes copy k;
- F_X(k): no gear of G_X strikes copy k.

The laws:
- 30·P_X = X# for X ≥ 5.
- Every g ∈ G_X has 2g² < X# (Bertrand). So every gear of G_X acts on every copy k + P_X.
- F_X is P_X-periodic on k ≥ 1, and E_X(k + P_X) ⇔ F_X(k) for k ≥ 1.
- (a) E_X(k + N) ⇔ E_X(k) for every multiple N of P_X, wherever all gears of G_X act at k. X² ≤ 30k + 1 is enough; at X = isqrt(q#) this covers every k ≥ M.
- (b) E_X(k + N) ⇒ E_X(k), for every k.
- (c) For every k: E_X(k) and not E_X(k + P_X) ⇔ both legs of k are prime and 30k − 1 ≤ X.
- At copies k ≥ 1, the acting and acting-free deletion patterns differ exactly at the twin copies with a leg ≤ X.
- Below (X + 1)²:
  - E_X(k) ⇔ both legs are prime;
  - F_X(k) ⇔ both legs are prime and X < 30k − 1.
- At X = isqrt(q#) with 30j + 1 ≤ q#: F(j) ⇔ both legs are prime and both lie above √(q#).
- No period:
  - copy j + (30j − 1)N has lower leg (30j − 1)(1 + 30N);
  - "both legs prime" is not N-periodic for any N ≥ 1.
- Fixed cut, for every q, these are equivalent:
  - some range copy is struck by no gear of [7, isqrt(q#)];
  - some copy with 30j + 1 ≤ q# has both legs prime and lower leg above isqrt(q#).

Lean (47 theorems):
- periods: thirty_mul_gearProd, prime_two_sq_lt_primorial, freeClear_add_iff, actClear_add_gearProd_iff;
- (a) to (c): actClear_shift_iff_of_acts, actClear_period_range, actClear_of_actClear_add_range, actClear_jump_iff_range;
- patterns and revealed laws: patterns_differ_iff_range, actClear_iff_legsPrime, freeClear_iff, freeClear_iff_range;
- no period and fixed cut: lower_leg_shift, legsPrime_no_period, fixed_cut, fixed_cut_sq.

### 1.12 Single-gear laws on copies (RangeCopies)
- For g ≥ 2 and j ≥ 1: g strikes copy j ⇔ 30j ≡ ±1 (mod g).
- Leg rules, for prime g ≥ 7:
  - same-sign legs of j and j + D ⇔ g | D;
  - lower leg of j and upper leg of j + D ⇔ g | 15D + 1;
  - upper leg of j and lower leg of j + D ⇔ g | 15D − 1.
- Some pair (j, j + D) with j ≥ 1 is struck by g ⇔ g | D or g | 15D ± 1, and a witness exists with j < g.
- Every prime g ≥ 7 divides a lower leg and an upper leg of some copy with 1 ≤ j < g.
- Lean: copy_product, copy_strike_iff, copy_strike_iff_legs, minus_leg_iff_mod, plus_leg_iff_mod, copy_strike_class, leg_minus_minus, leg_plus_plus, leg_minus_plus, leg_plus_minus, copy_leg_rule, copy_leg_rule_converse, copy_pair_iff, exists_leg_minus, exists_leg_plus.

### 1.13 Acting pair law and distance law (RangeActPair)
- For prime g ≥ 7, j ≥ 1 and 0 < D < g, these are equivalent:
  - g strikes j and j + D, and g² ≤ 30j + 1;
  - g | 225D² − 1, g | 2j + D, g | 900j² − 1 and g² ≤ 30j + 1.
- A prime g ≥ 7 striking copies 1 ≤ x < y divides (y − x)(225(y − x)² − 1).
- Acting is monotone in the copy.
- Lean: Acts, acting_pair_law, distance_law, acts_mono.

### 1.14 Centre law (RangeCentre)
- For prime g ≥ 7 and 1 ≤ x < y, g strikes both x and y ⇔ one of these holds:
  - g | y − x and g | 900x² − 1;
  - g | x + y and g | 225(y − x)² − 1.
- The same holds over any field with 30 ≠ 0 and 4 ≠ 0.
- StrikesCopy g k ⇔ g | 900k² − 1, for every k.
- Lean: StrikesCopy, strikesCopy_iff_sq, strikesCopy_iff_zmod, centre_law_field, centre_law.

### 1.15 Locator closure (RangeLocator)
- For prime g ≥ 7 with g ∤ N and any r, some k < g gives j = r + kN ≥ 1 with g | 30j − 1. The residues r + kN (k < g) are distinct and cover Z/g.
- A class mod N that g never strikes forces g | N.
- A class struck by no prime in [7, X] has ∏_{7≤p≤X} p | N, and that product is ≤ N.
- Lean: class_meets_every_residue, class_residues_distinct, locator_closure_minus, locator_closure, locator_closure_contra, locator_modulus, locator_modulus_prod, locator_modulus_le.

### 1.16 The missed copy of each gear (RangeMissed, RangeMissedReveal)
The statements hold for every prime g ≥ 7.
- Exactly one of g² ≡ 1 and g² ≡ 19 (mod 30) holds.
- c_g has legs g² + 28 and g² + 30 (case 1), or g² + 10 and g² + 12 (case 19). g divides neither leg.
- No larger gear acts on c_g, because every prime h > g has h² > g² + 30.
- Both legs are coprime to 30.
- A composite leg has a prime factor h with 7 ≤ h < g.
- Both legs are prime ⇔ no prime h with 7 ≤ h < g divides a leg.
- Lean: prime_ge7_mod, sq_mod30_cases, sq_mod30_exactly_one, missed_copy_legs (with _case1 and _case19), gear_misses_own_copy, larger_gear_sq, no_larger_gear_acts, legs_coprime_30, composite_leg_small_factor, missed_copy_revealed_iff (with _case1 and _case19).

### 1.17 The derived copy map (RangeDerived)
- Column x (with x² ≡ 1 or 19 mod 30) carries copy J(x) = (x² + c)/30, where c = 29 in case 1 and c = 11 in case 19.
- For prime x, J(x) = J_x.
- J is strictly increasing.
- In case 1, J(x) = x exactly at x = 1 and x = 29. Case 19 has no fixed column.
- J(x) < x exactly at x = 11, 19 (case 1) and x = 7, 13, 17, 23 (case 19).
- Lean: J1_exact, J19_exact, copy_map_on_gear, J1_fixed_identity, J1_fixed_iff, J19_no_fixed, the strictMono lemmas, and the below-self lists.

### 1.18 Node bookkeeping (RangeReach)
- nextNode_unique, exists_nextNode, exists_prevNode.

**Kernel caveats**
- RangeStatement admits twin pairs off the copies. So the reach equivalences in 1.2 and 1.3 are about twin nodes, not copies. The copy form implies RangeStatement in one direction only (1.5).
- `small_cases` (RangeStatement at q = 7..23, with witness p = 29) is an instance, listed in section 5.
- RangeGen1 to RangeGen5 each declare their own notation. The reviewer checked that the separate declarations agree:
  - Clear, Survives and Surv agree;
  - Twin, TwinCopy and LegsPrime agree;
  - Revealed, Loss and Band agree across RangeGen1 and RangeGen2;
  - rho = mirrorRho;
  - Uminus = uMinusSet;
  - Uminus ∨ Band = InU for q ≥ 7;
  - RangeCopy = RangeCopyWindow for q ≥ 7.

---

## 2. General, proved on paper or in scratch Lean, not yet in the kernel

### 2A. Checked in scratch Lean files this round (GENERAL-PROVED; not in the kernel)

**2A.1 The wall inequality and its uses** (C6). [Lean scratch: `RangeWall.lean` and `C6Defence.lean`, recompiled as `adj_c6/C6AdjWall.lean` and `adj_c6/C6AdjDef.lean`]
- **Wall.** nextprime(X)² < X# for every natural X ≥ 7, and the inequality fails at every X ≤ 6.
  - For prime X ≥ 11, X# ≥ 15X(X + 1).
  - Proof: Bertrand (Mathlib) and divisibility by distinct primes.
  - Lean: wall_iff, wall_inequality_nat, wall_fails_below_seven, nextprime_sq_lt_primorial.
- **Doubled wall.** (2X)² < X# ⇔ X = 0, X = 7 or X ≥ 11, for every natural X. Lean: four_sq_lt_primorial_iff.
- **Acting from one period.** For every q ≥ 7 and every gear g ≤ q', g acts on every copy j ≥ M. In fact q'² < q#. Lean: acts_from_period.
- **Window and range top.**
  - Every j with 30j + 1 < q'² has j < M.
  - Such a j ≥ 1 is in the range ⇔ 30j − 1 > q.
  - The copies cut by the window below the range are 1 ≤ j ≤ ⌊(q + 1)/30⌋, and there are some ⇔ q ≥ 29.
  - Lean: window_below_primorial, window_mem_range_iff, below_range_iff, below_range_exists_iff.
- **Lap bound.** For every q ≥ 7 and residue r, let h(r) be the least j ≡ r (mod M) on which q' strikes and acts (field_to_range G1 row 13).
  - h(r) ≤ r + q'M, with q' dividing the lower leg.
  - h(r) ≤ r + (q' − 1)M, with q' dividing either leg.
  - The second bound is attained (see section 4, row 13).
  - Lean: lap_bound, lap_bound_sharp, locator_closure_plus.
- **Silent classes.** Take X ≥ 5, N ≥ 1 and a class mod N struck by no prime in [7, X].
  - ∏_{7≤p≤X} p divides N, so 30N ≥ X#. With the wall this gives N > X'²/30 for X ≥ 7, where X' is the next prime after X.
  - The class has at most one member j ≥ 1 with 30j ≤ X#, and at most one in the window band.
  - Lean: silent_class_period, silent_class_beyond_wall, silent_class_one_per_period, silent_class_one_in_band.

**2A.2 Near-neighbour kills: cofactor floor and kill-free windows** (C5). [Lean scratch: `c5/NearKill.lean`, `c5_defend/C5Defence.lean`]
- **Cofactor floor.** Take odd h < g, both prime to 30, with d = g − h and leg constant b.
  - h divides the leg g² + b ⇔ h divides d² + b.
  - Write d² + b = κh. Then κ is even and carries the whole {2, 3, 5}-part of d² + b.
  - The 2-adic classes are exact: v2(d² + 12) = 4; v2(d² + 28) ≥ 5 when d ≡ 2 (mod 4); v2 = 1 for b = 10 and b = 30.
  - No h divides both legs.
  - Lean: kill_floor and kill_floor4 (the 2h and 4h floors).
- **Kill-free window (W).** Case 1: d² + 30 < 2h ⇒ h misses both legs of c_g. Case 19: the same with d² + 10 < 2h. Lean: near_immune_case1, near_immune_case19.
- **Boundary (E).** Here h only needs to be odd.
  - Case 1: h divides a leg with d² + 30 ≤ 2h ⇔ 2g − 29 = t² with 5 ∤ t and h = g − t + 1. Then g ≡ 19 (mod 30), κ = 2, and only the upper leg is killed.
  - Case 19: h divides a leg with d² + 10 ≤ 2h ⇔ 2g − 9 = t² with 5 | t and h = g − t + 1. Then g ≡ 17 (mod 30), κ = 2, and the lower leg is killed.
  - The other square solutions give no kill: g ≡ 7 (mod 30) on leg 30, and g ≡ 29 (mod 30) on leg 10.
  - Lean: sq29_class, sq9_class, boundary30_iff, boundary10_iff, boundary30_upper_only.
- **Class windows (R).** For g ≡ c (mod 30), h misses both legs whenever d² + b_c < K_c·h. Each class window is sharp: there is a prime kill on the boundary in every class. Lean: cert_b_c, floor_b_c, window_c, sharp_c.

  | g mod 30 | 1 | 11 | 19 | 29 | 7 | 13 | 17 | 23 |
  |---|---|---|---|---|---|---|---|---|
  | b_c | 28 | 30 | 30 | 30 | 10 | 12 | 10 | 12 |
  | K_c | 4 | 6 | 2 | 22 | 22 | 12 | 2 | 4 |

- **Neighbour rule.** Take r ≥ 1 and n with p_{n−r} ≥ 7, and let c = p_n mod 30. If (p_n − p_{n−r})² + b_c < K_c·p_{n−r}, then none of p_{n−1}, …, p_{n−r} kills c_{p_n}. Lean: window_mono with window_c.

**2A.3 Top band of λ** (C4). [Lean scratch: `c4_0926/TopBand.lean`]
- Proved in Lean:
  - the algebra of the top-band law;
  - monotonicity of the d0 test;
  - the window inequality (window_form).
- The statements are in 2B.11.

**2A.4 Signed landing rule** (C2). [Lean scratch: `C2CoincideAx.lean`]
- For every modulus r coprime to 30, (d, e) and (d', e') land together (d a window offset, e = ±1 a leg) ⇔ one of:
  - e' = e and d ≡ d' (mod r);
  - e' = −e and 15(d − d') ≡ e (mod r).
- Equivalently, d − d' ≡ 2e·a_r.
- At gap level: the gaps d and d' give a cross-leg coincidence for some choice of legs ⇔ 15(d − d') ≡ ±1 (mod r).
- Lean: land_together_iff_zmod, gap_cross_iff.

### 2B. Proved on paper (GENERAL-PROVED; not in Lean)

**2B.1 Phase and translate form of the whole period** (P6; field_to_range G2 formulas (1)–(3), items 5, 6; G6 items 4, 6). For every q ≥ 7:
- (i) For k ≥ 1, the translate kM + [1, M − 1] is fully deleted by the acting gears 7..isqrt(q#) ⇔ the phase vector (−kM mod g) over g ∈ U_q lies in C_q^free.
  - C_q^free is the set of U_q phase vectors whose acting-free strikes cover every survivor of [1, M − 1].
  - k ↦ (−kM mod g)_g is a bijection mod ∏U_q.
- (ii) Range(q) ⇔ 0 ∉ C_q^act ⇔ [some s ∈ S_q with 30s − 1 ≤ g_max has both legs prime] or [0 ∉ C_q^free].
  - C_q^act is the same set with acting strikes.
  - An empty C_q^free is sufficient for Range(q), not necessary.
- (iii) Lift law:
  - g ∈ U_q strikes s + kM ⇔ k ≡ −s·M⁻¹ ± (q#)⁻¹ (mod g);
  - g acts on it ⇔ k ≥ 1 or g² ≤ 30s + 1.
- (iv) Coincidence law:
  - a clearance class meets a tooth ⇔ g | q# ∓ 1;
  - two teeth meet ⇔ g | q# ∓ 2.

**2B.2 Survivor structure of one period** (P7; shelves E1–E3, E5, E6, E8; field_to_range G6 item 3). For every q ≥ 7:
- |S_q| = ∏_{7≤g≤q}(g − 2) per period M.
- The range copies are exactly [⌊(q + 1)/30⌋ + 1, M − 1], and the omitted copies are never survivors. So:
  - survivor class 0 has no copy in the range;
  - each of the other ∏(g − 2) − 1 classes has exactly one copy in the range.
- The affine stabiliser of S_q is {j ↦ uj : u² ≡ 1 (mod M)}. It contains no translations.
- S_{q'} → S_q is onto and exactly (q' − 2)-to-1.
- Each g > q strikes exactly two of the g subclasses mod M·g of every class, one per leg, and acts on nothing below ⌈(g² − 1)/30⌉.
- Among the copies of S_q, the ones acted on by no gear above q are copy 0 and the twin copies with legs above q.
- Using the machine gears only: some range copy is struck by no gear ≤ q.
- In Lean already: j ∈ S_q ⇔ gcd(900j² − 1, M) = 1 ⇔ q < P(j) (RangeGen4.surv_iff_coprime, surv_iff_lt_P).

**2B.3 One upper gear on consecutive survivors** (P8; runs_gain A1–A6, B6; mirror_runs B2, B9; field_to_range G3 items 1–3, G4 items 3, 5). For every q:
- (i) Run characterisation: x + R, with 0 = r_0 < … < r_N, is a one-gear run of G ⇔ all of these hold:
  - G strikes every offset;
  - every p in 7..q avoids R with its pair of classes {u_p, u_p + t_p}, where u_p = −a_p − x;
  - every non-member of [0, r_N] is covered by some p;
  - G² ≤ 30x + 1;
  - 1 ≤ x ≤ M − 1 − r_N.
- (ii) Lane rule: p strikes x + c + kG ⇔ k ≡ G⁻¹(u_p − c) or G⁻¹(u_p + t_p − c) (mod p). These are two distinct slots.
- (iii) Cap by one machine gear:
  - a machine gear p alone caps a consecutive-teeth run at 2(p − 2), so gear 7 caps every such run at N ≤ 10;
  - N = 10 forces G ≡ ±1 and T ≡ 0 (mod 7).
- (iv) Gear-7 lane law: see 2B.10, R2.
- (v) Comb:
  - comb_g(L) = 2⌊L/g⌋ + min(L mod g, 1) + [L mod g > σ_g] is the largest number of g-teeth in L consecutive copies (written c_g(L) in the record);
  - τ_g(N) = ⌊(N − 1)/2⌋·g + [N even]·σ_g is the least span of N teeth;
  - three consecutive teeth span exactly g;
  - one gear striking three copies within a span below 2g has two of them exactly g apart.
- (vi) Tooth law:
  - 15σ_g = κ_g·g ± 1, with κ_g = 1, 2, 4, 7 for g ≡ ±1, ±7, ±11, ±13 (mod 30);
  - consecutive teeth σ_g apart differ in cofactor by 2κ_g;
  - on teeth, acting ⇔ cofactor ≥ g.
- In Lean already: acting_pair_law, distance_law, centre_law.

**2B.4 Converse reduction and chain forms** (P9; derived C2, C3; range_kernel 2.1 items 1, 3–8; range_kernel2 2A items 1–5, 7, 8 and F1–F6). These are set identities over the service intervals and hold for any sets as stated.
- Service and chain:
  - A revealed copy (p, p + 2) serves exactly the primes q ∈ [σ(p + 2), p).
  - RANGE ⇔ chain(T), where T is the set of all revealed copies.
  - chain is upward-closed.
- Domination and tails:
  - M dominates R ⇔ p_R ≤ p_M ≤ σ(p_R + 2)# − 2.
  - RANGE ⇔ chain(S_1 ∪ N), where N = ⋃_r tail(r).
- Crossing form: for sets of primes ≥ 29, chain(S) ⇔ S meets (r, r# − 2] at every prime r ≥ 7.
- Anchors:
  - Given RANGE, chain(S_1) ⇔ m_r > r at every live r.
  - Failure at r ⇔ g_r² + A ≤ r, and then m_r⁺ + 2 > r#.
  - RANGE ∧ ¬chain(S_1) ⇔ RANGE ∧ (S_1 is finite, or some consecutive (m, m⁺) has m < σ(m⁺ + 2) ≤ v(m)).
- Transport F4: c_h ∈ S_1 ⇔ some revealed p > h has p ≡ h² + A_h (mod (h−)#). For h > r (h ≥ 11), this class meets (r, r# − 2] only in c_h.
- Fixed-row criterion F6: row f kills every unit class of (x² + a)(x² + a + 2) ⇔ (f = 5, a ≡ 4) or (f = 3, a ≡ 0 or 2). This never happens in base 30.

**2B.5 Shelves, regions and position** (P10; shelves C1–C3, C5; derived C6, C10; range_kernel 2.1 items 2, 3). For all gears:
- Shelves:
  - The shelves tile j ≥ 2.
  - The acting set on every copy of shelf(g) is exactly the primes in [7, g].
- Regions:
  - The regions tile j ≥ 2, except the case-1 head copies (upper leg g²), which are never revealed.
  - Every revealed j ≥ 2 lies in the region of its top acting gear.
  - chain(revealed region copies) ⇔ RANGE.
- Position lemma: a copy at position n ≥ 2 of gear g has legs strictly between g² + 30 and g'².
- Region law: a copy in region g, and also c_{g'}, is revealed ⇔ no prime in 7..g divides a leg.
- Direction lemma, for R revealed at position ≥ 2 of region g:
  - neither c_g nor any missed copy with lower leg below p_R dominates R;
  - c_{g'} dominates R ⇔ c_{g'} is revealed and g'² + B' ≤ σ(p_R + 2)#.

**2B.6 Missed-copy route** (P11; two-paths A1.1, A1.2, A1.13, A1.14(ii), (iii); derived B1, B2, C1, C3, C4, C5, C7; range_kernel 2.3 item 17). For all gears.

Strikes on c_g:
- A row h < g strikes c_g ⇔ g² ≡ −A or −B (mod h), i.e. g mod h ∈ E_h.
- The four targets are distinct mod every h ≥ 7.
- |E_h| = 2 + (−A/h) + (−B/h), except for (7, case 1).

Range membership and hand-off:
- c_g lies in machine q's range ⇔ q − A_g < g² ≤ q# − B_g.
- Hand-off equivalence: every q ≥ 7 has a revealed missed copy in its range ⇔ the revealed gears 7 = g_0 < g_1 < … are infinite with g_{i+1}² + B ≤ (g_i² + A)# for all i. This target implies RANGE.
- One hand-off has four equivalent forms. With P = (g² + A_g)#:
  - g'² + B' ≤ P;
  - g'² ≤ P − 41;
  - J_{g'} < P/30;
  - σ(g'² + B') ≤ g² + A_g.
- X(g) = isqrt(P − 41) is non-decreasing.
- Slack step: u_{i+2} < p_{i+1}·u_{i+1} ⇒ Δ_{i+1} > Δ_i.

Escape classes:
- An integer n ≥ 2 is a revealed gear ⇔ n lies in a level-(n − 1) escape class.
- Cutoff criterion: every striker of c_g is live, so c_g is revealed ⇔ g mod 30·∏_{7≤h≤λ(g)} h is in a level-λ(g) escape class.
- The gears acting on c_m are exactly the primes 7..m. This holds for any integer m ≥ 0 except 6 and 10.

Forward arrows:
- chain(S_1) ⇒ chain(SQ_unit) ⇒ chain(SQ_all) ⇒ RANGE.
- chain(S_1) ⇒ chain(S_stretch) ⇒ chain(regions).

In Lean already: the missed-copy legs and the reveal rule (1.16).

**2B.7 Escape tree and doom** (P12; range_kernel 2.2 T1, T2, D1–D5, R; derived B1, B7, B8). For all levels.
- The children of an escape class at row h split into three kinds:
  - one zero child, holding at most the gear h;
  - e_h struck children;
  - h − 1 − e_h escaping children.
- Descendant sets are non-empty.
- A class is doomed ⇔ it holds no revealed gear ⇔ every prime g ≡ r_C (mod M_P) has a composite leg.
- Non-doom is inherited upward, and doom downward.
- x(x² + A)(x² + B) has no fixed prime divisor on any escape class, and both leg polynomials are irreducible.
- No finite row set certifies doom. [imports Dirichlet]
- With the acting bound removed, every class is doomed and the tree is unchanged.
- R: no class is doomed ⇔ every escape class holds a prime g with g² + A and g² + B both prime ⇔ every class holds infinitely many such g.
- Entering law: row λ(g) strikes c_g ⇔ g − λ is an even root of x² ≡ −A' (mod λ).

**2B.8 Mirror pairs: common divisors, the band, the survivor test** (C1 adjudication, items 1–4 and 6, and the repaired item 5).
- **Common divisors**, for every q:
  - gcd(L1, H1) = gcd(d, q# − 2) and gcd(L2, H2) = gcd(d, q# + 2);
  - gcd(L1, H2) and gcd(L2, H1) divide M.
  - So AB_q ⊆ {the primes of (q# − 2)(q# + 2) in U⁻}, and every sole striker lies in AB_q.
  - The divisor form is in Lean (RangeGen3.same_minus_iff, same_plus_iff, no_cross_strike, deletes_both_mem_uMinus). The gcd form is on paper.
- **Cofactor form**, for every q and every U⁻ prime g of q# ∓ 2:
  - g < m_g and E_g is odd;
  - ρ − g² = gE_g + 2 (for q# − 2) or gE_g (for q# + 2);
  - at d = gk (k odd), g acts on both members ⇔ 15k ≤ E_g;
  - the struck legs are g(m_g ∓ 15k);
  - the pair survives ⇔ gcd(m_g² − 225k², M) = 1;
  - every d in the band satisfies d ≤ M − 4, so the band lies in the range.
- **Residue law**:
  - For every odd prime p ≤ q, m_g is a root of x² − E_g·x + s (mod p). The roots are m_g and −g.
  - E_g² − 4s is a square or 0 mod p.
  - Mod 3: 3 ∤ E_g for q# − 2, and 3 | E_g for q# + 2.
- **No empty band for q ≥ 11.** 2q# ∓ 4 + e² is not a square for any odd e ≤ 13.
  - The proof is a complete case split by the least odd prime at which e² − 4s is a non-zero non-residue, for e = 1, 3, …, 13:
    - q# − 2: 5, 3, 11, 7, 3, 5, 17;
    - q# + 2: 3, 5, 3, 3, 11, 3, 3.
  - q = 11 and q = 13 are settled directly: 4785 and 60225 are not squares.
  - With RangeGen3.band_empty_minus/plus, every U⁻ prime of q# ∓ 2 has a non-empty band for q ≥ 11.
- **Band width** (repaired). [certificate]
  - Every factorisation q#/2 − s = g·m with g < m and odd E = m − g ≤ 10⁸ is one of exactly 43, all at q ≤ 41: 8 with g = 1, 22 with g even, and 12 with g an odd prime (listed in 5.4).
  - Hence E_g ≥ 51 for every q ≥ 11, and E_g > 10⁸ for every q ≥ 43.
  - Proof: for odd E, E² − 4s is never a square, so a least odd non-residue prime p_E exists, and it excludes E once q ≥ p_E. p_E ≤ 113 for every odd E ≤ 10⁸, and the primes q = 11..109 are checked completely.
  - The record's count of 40 is corrected to 43.
  - 10⁸ is a fixed constant; this gives no growth of E_g in q.
- **Survivor test from the gap.**
  - If no prime p in 7..q divides (15kE_g)² − (225k² + s)², the pair at d = gk is a survivor pair.
  - At k = 1 the test number is (15E − 226)(15E + 226) for q# − 2, and (15E − 224)(15E + 224) for q# + 2.

**2B.9 Aligning a covering pattern: box, edge and translation laws** (C2 adjudication, items 1, 3, 4, 6, 8). For every q ≥ 7.

Setup:
- The gears 7..q split into fixed gears and free gears; R is the product of the free gears and m = M/R.
- K is a set of n offsets inside the window [0, W].
- The fixed residues miss K (H1) and strike every other offset of [0, W] (H2). u is their CRT residue in [0, m).
- The upper gears g_j > q are distinct, of any size. G = ∏g_j, and B is a class mod G.

The laws:
- **Box law.** Consider the copies s in [1, M − 1 − W] with s ≡ u (mod m), s ≡ B (mod G), and the survivors of s + [0, W] exactly s + K.
  - These copies are exactly s = u + m(v0 + Gt), for 0 ≤ t < T, with t mod r ∉ Y_r for each free gear r, within the range bounds.
  - |Y_r| ≤ 2n.
  - H2 is needed: without it, the formula and the raw count differ in 387 of 400 check boxes.
- **Single landing.** In the B8 configuration (k + 1 free gears, k upper gears), T ≤ ⌈R/G⌉ ≤ r_min, so each (r, d, e) class excludes at most one t.
  - For k ≥ 1, R < r_0·G.
  - So B8's hypothesis R ≥ (2n(k + 1) + 3)G forces r_0 > 2n(k + 1) + 3, and r_0 ≥ 2n + 3 when k = 0.
- **Edge law.**
  - s = 0 ⇔ u = v = 0.
  - The top cut removes ⌈(1 + W + u)/m⌉ − 1 values of v.
  - For W < m that is at most one: v = R − 1, exactly when u + W ≥ m.
- **Translation.** A window [lo, hi] anywhere in the range reduces to [0, hi − lo] with shifted data. The box law, the edge law and the count carry over.
- **Rigid boxes.** When R < G there is at most one aligned copy, s = u + m·v0. It is present ⇔ v0 < R, the range bounds hold, and 0 ∉ Y_r for every free r.
- **Partition.** Branching on uncovered offsets partitions the covering tuples into boxes, and the capacity prune loses none.

**2B.10 Ceiling on one-gear runs** (C3 adjudication, R1 to R7).
- **R1 Containment**, for every q ≥ 7, every G and every N: a one-gear run of G with offsets 0 = r_0 < … < r_N has r_N + 1 ≤ h(q + {G}), and h(q + {G}) is finite.
- **R2 Gear-7 lane law**, for every G coprime to 7:
  - among K = 7a + r consecutive same-lane teeth, at most f(K) = K − 2a − max(0, r − 5) escape gear 7;
  - K(n) = n + 2⌊(n − 1)/5⌋ is the least K with f(K) ≥ n.
- **R3 Span floor**, for every G coprime to 105: any N teeth of G that escape gear 7 span at least T_G(N) = (K(⌈N/2⌉) − 1)·G + [N even]·σ_G. For N ≤ 10 this is τ_G(N).
- **R4 Exact gear-7 floor**, for every prime G ≥ 11:
  - the least span E_7(G, N) of N gear-7-escaping teeth is fixed for every N by G mod 7 and σ_G mod 7, through a table periodic in N (e_{i+10} = e_i + 7G);
  - E_7 = T_G for every N exactly when (G mod 7, σ_G mod 7) is (1, 0) or (6, 0).
- **R5 Monotone ceiling**, for every q and G: h(q + {G}) ≤ h2(G), with equality when G is the next prime after q.
- **R6 Ladder** (repaired), for every q ≥ 7: T_G(N) ≤ E_7(G, N) ≤ r_N ≤ h(q + {G}) − 1 ≤ h2(G) − 1.
  - So N ≤ N_max(H, G) for H = h(q + {G}) and for H = h2(G).
  - N ≤ N_max ⇔ T_G(N) ≤ H − 1. There is a closed form in ⌊(H − 1)/G⌋ and (H − 1) mod G.
  - Also N ≤ (10(H − 1)/G + 30)/7.
- **R7 Jacobsthal link**, for every q: h2(q) ≥ Y_7(q) ≥ ⌊(j(q#) − 1)/30⌋.

**2B.11 λ(g) and the top band** (C4 adjudication, items 1–4, 6–8).
- **Top-band law.** For every gear g and every row h with 2g/3 < h < g, in both cases: h is live on g ⇔ e_h > 0 and (g − h + 1)² ≥ 2g + 1 − B', i.e. g − h ≥ d0(g). The algebra is in Lean scratch (2A.3).
- **λ characterisation**, for every gear g ≥ 7 and every c in [2/3, 1): λ(g) > cg ⇔ some non-inert row lies in (cg, g − d0(g)].
  - If the largest non-inert prime h\* ≤ g − d0 exceeds 2g/3, then λ(g) = h\*.
  - A quarter of the unit classes are inert: 144 of 192 unit classes mod 840 are non-inert in case 1, and 24 of 32 mod 120 in case 19.
- **λ(g) > 2g/3 for every gear g ≥ 41.** [imports BMOR 2018, an explicit prime-count estimate in arithmetic progressions, for g ≥ 1.2·10¹⁰] [certificate: a chain of 112 certified non-inert primes covering g < 1.46·10¹⁰]
  - Each chain step is a general lemma covering every real g in [3p_i/2, 3p_{i+1}/2).
  - Gears 41, 43, 47 and 53 are done by hand.
- **Exceptions.** The gears with λ(g) ≤ 2g/3 are exactly 7, 11, 13, 23 and 37, with λ = 0, 0, 7, 13, 23.
- **Window form W**, for every level P' ≥ 23 and every escaping descendant class (Desc_C(P')):
  - the class has at most one member x in (P', max(37, 3P'/2)];
  - that member is a prime gear of the class's case;
  - x ∈ Cap(P') ⇔ no row h in (P', x) has e_h > 0 and (x − h)² ≥ 2h − B'.
  - This rests on the λ bound above.
- **Distance to the top row.** g − λ(g) ≥ d0(g) for every gear g ≥ 19.
- By the λ characterisation, the bound λ(g) > 2g/3 for g ≥ 41 is equivalent to this: every interval (2g/3, g − d0(g)] with g ≥ 41 holds a non-inert prime.

**2B.12 Near-neighbour kills: paper parts** (C5 adjudication, items 3–7).
- **Killer ordering**, for every gear g:
  - ordered by distance d = g − h, the killers of c_g have strictly increasing cofactor κ, over both legs together;
  - so the least-κ killer is the nearest.
- **First lap**, for every row h ≥ 7: a first-lap kill has
  - d even;
  - d² ≡ −b (mod h);
  - d² ≥ 2h − b;
  - d ≥ √(2h − 30).
  
  There is at most one even root per leg constant b, and at most 4 first-lap targets per h (an upper bound, not claimed sharp). The clause is vacuous for h = 7, 11 and 13.
- **Reduction** (repaired), for every q and r:
  - Take a gear g with √q < g ≤ √(q# − 32) and at least r lower gears, and let h_r be its r-th lower gear.
  - If g satisfies its window at h_r, then R8(q, r) holds. The window is (g − h_r)² + 30 < 2h_r in case 1 and (g − h_r)² + 10 < 2h_r in case 19; the class window (g − h_r)² + b_c < K_c·h_r also suffices.
  - The window is sufficient, not necessary.
- **Top-gear anchor**, for every prime q ≥ 7: q# ≥ 3q(q + 1) > q² + 32 (Bertrand), so c_q lies in machine q's range.
- **All-r equivalence**, for every gear g:
  - no lower gear kills c_g ⇔ both legs of c_g are prime;
  - so R8(q, r) at every r is equivalent to a revealed missed copy in the range, which gives RangeStatement q.

**2B.13 Further general items, lower bearing** (from the audit; sources as named).
- **Derivation operator** (kernel 2.3; kernel2 2B):
  - Der_int(O) = D;
  - idempotence;
  - the fixed points are the f_C, and D is the least fixed point;
  - tower P^k class counts 2^(2k+1);
  - diagonal acting, persistence and the next-digit law;
  - the limit L.
- **Barrier:** U admits total blame b(j) = lpf(30j − 1), so any blame-forbidding property must fail in U and cannot follow from T.
- **field_to_range G3:**
  - four-runs and the repaired (13);
  - the copy E2 record R(g, h) for all gear pairs: one gear takes at most 2 of any 5 consecutive copies, and distances 1, 2 and 3 force gear 7, gears 29/31, and gears 11/23 respectively.
- **field_to_range G4:**
  - the Γ_k formula, depth bounds, onset, and the (16')(a)–(d) laws;
  - the (15) doubled-pair decomposition;
  - the (14) lift law, which the record flags as a tower reading.
- **field_to_range G5:** transport and split.
- **field_to_range G6:**
  - palindrome: d\* = the least odd d with gcd(225d² − 1, M) = 1;
  - antipodes: legs q#/2 + 14 and q#/2 + 16, and 7 is the only machine striker, for all q (checked to q = 113).
- **Shelves and stretches:**
  - shelf silence, deferral and silent runs;
  - cofactor structure;
  - the p1/p2 formulas;
  - protected-stretch formulas (L = p1 only at g = 13; l ≥ 3 for g ≥ 31);
  - comb D1–D5.
- **D7 and D8.** [imports CRT and Shiu]
- **Kills:**
  - the kill rule, the distance law and the sharing law E7/E8;
  - inert-class characters and Pell recurrences;
  - whole-run inert pairs.
- **Gear pairs and Path B:**
  - joint strikes of gear pairs (F1, F2);
  - Path B translation B1.2, exact transfer B1.7, preservation of forced pairs, and orbit sizes F7.
- **First-strike separations** of twin, cousin and sexy pairs (F4, F5). The record only checks them; the audit derives them from the closed forms of a_g, for example p ≡ 11 gives F_q − F_p = (7 − 2p)/15.

### 2C. General, but counting or locating type (not results under the working rules)
- **B8 alignment count** (C2 item 5). [counting]
  - Hypotheses: the B8 configuration, H1 and H2, full allowed sets on the k + 1 free gears, distinct upper gears > q, and W < m.
  - Under these, at least ⌊R/G⌋ − 2n(k + 1) − 2 values of t give aligned copies.
  - For W ≥ m, the 2 is replaced by [u = 0] + ⌈(1 + W + u)/m⌉ − 1.
  - Nothing shows the hypothesis (a covering with spare free gears and R ≥ (2n(k + 1) + 3)G) is met at any general q.
- **h2(q)/q → ∞** (C3 R8). [imports Ford–Green–Konyagin–Maynard–Tao, arXiv:1412.5029, eqs. (1.2), (1.3)] [counting]
  - It follows from R7.
  - It counts only as the remark that the ladder's ceiling is not uniform in q. No run is produced.
- **Layered lower bound on the covered stretch** (C3 item 9). [counting]
  - For all large q, one stretch of y = K·q·A³C²/B⁴ struck copies exists (A = log q, B = log A, C = log B). K may be any value below a limit that tends to 1/(1080·c_1), where c_1 is a constant in the twin-prime upper bound π_2(t) ≤ c_1·t/(log t)².
  - The adjudication wrote out the o(1) assembly that the worker had left open.
  - It has no content at finite q: no parameter choice exists below log q ≈ 300.
- **Cap(P) ⊂ (P, max(37, 3P/2)]** for every level P ≥ 7, attained at P = 23 (C4 item 5). [locating]
- **λ(g) > 0.99g for every g ≥ 8.09·10⁹, and λ(g)/g → 1** (C4 item 8). [imports BMOR 2018] [locating]
- **RS block law** (C5 item 8, repaired). [imports Rosser–Schoenfeld (3.5), (3.6)] [counting]
  - Take X ≥ X\* = 255,271,782.94, any q with X² ≥ q and q# ≥ 4X² + 32, and any integer 0 ≤ r ≤ ρ_RS(X). Then R8(q, r) holds, witnessed by a gear in (X, 2X].
  - ρ_RS(X) = 0.7/(u + v) with u = ln(2X)/X and v = ln(2X)/√(2X − 30). It is strictly increasing on (15, ∞).
  - ρ_RS(10¹⁵) = 888,539.34 and ρ_RS(10¹⁶) = 2,637,438.23.
- **Bounded-r form of R8 for every q** (C5 item 9). [imports RS (3.5), (3.6), (3.16)] [counting] [locating]
  - It combines the RS block law, applied at X = X_max(q) = isqrt((q# − 32)/4) for each q ≥ 179, with the located witnesses c_13 and c_198437.
  - The recorded bound 3,585,137,033,388,283 lies below the exact floor of ρ_RS(X_max(179)) = 3,585,137,033,388,285.52.
  - The ceiling on r comes from the method: at q = 179, R8 holds for every r.
- **Fixed-H density ∏w_h.** [imports CRT and Dirichlet] [counting]
- **Density law 2^(−r_L).** [imports Dirichlet] [counting] range_line_map section 6 lists it as checked only for h ≤ 200000 and L ≤ 13.
- **Budget bounds:** B2(b), B2(c), B10(c), capacity, holes ≤ 2⌈L/g⌉, and R ≤ G_{2⌈R/g1⌉}. [counting]
- **"Range as a power"** (kernel 2.3 item 18) and the orbit-chain LOCATED item (kernel2 2B). [locating]

---

## 3. Claimed general, not proved

### 3A. Statements claimed for every q, with what is missing

**3.1 RANGE itself.** It is not established for all q.

**3.2 Acting-both set equals the primes of (q# − 2)(q# + 2) in U⁻** (C1, mirror_runs A8), for every q ≥ 11.
- Proved for all q:
  - the inclusion AB_q ⊆ {those primes};
  - equality ⇔ the existence clause below (2B.8 with RangeGen3).
- Missing: the existence clause. For every U⁻ prime g of q# ∓ 2, some odd k ≤ E_g/15 must have both cofactors m_g ∓ 15k free of the primes 7..q. Equivalently, the odd k in [1, E_g/15] must not all be covered by the two classes 15k ≡ ±m_g (mod p), p in 7..q.
- No residue-level argument was found. The band-width result gives only E_g > 10⁸, a constant, so the band length does not grow with q.
- No counterexample is known.
- Checked at q = 11..79, and this round at q ≤ 197 except q = 157, 163, 173, 181 and 191.

**3.3 Alignment of a relaxed covering pattern into the period** (C2, mirror_runs B8, runs_gain unresolved (v)).
- Missing: an argument that an aligned copy exists when a box has spare free gears. Such a copy exists exactly when the landing positions (at most 2n per free gear) plus at most two edge positions do not fill the line [0, T).
- The only argument given compares their numbers, which is a count (2C).
- Also missing:
  - when a pattern's coverings leave spare free gears (at q = 43, gear 53, every covering is rigid);
  - landings of different free gears at the same position.
- runs_gain records that no general lemma aligns a relaxed pattern into [1, M − 1].

**3.4 M(G) ≥ G'^4 for every shelf gear G ≥ 19** (C7; two-paths B1.3, B1.4(v); derived F2, F7). Here M(G) = ∏_{7≤g≤G} g and G' = nextprime(G). It is used in Path B.
- Only the base case M(19) = 323323 ≥ 23⁴ is recorded, with a computation over shelves ≤ 30000.
- Missing: the induction step, G'' ≤ G'^{5/4}.
- Route recorded by the audit, not adjudicated: Bertrand gives G'' < 2G' ≤ G'^{5/4} once G' ≥ 16.

**3.5 Below-square strike count** (C8; shelves A4, D1, D2).
- Claim: gear g strikes exactly 2⌊g/30⌋ + c(r) copies j ≥ 1 with 30j + 1 < g², where r = g mod 30 and c = 0, 0, 0, 1, 1, 1, 2, 1 for r = 1, 7, 11, 13, 17, 19, 23, 29. Hence every gear except 7 and 11 strikes below its own square.
- Missing: a written argument. The records give checks only, for gears ≤ 2000.
- Route recorded by the audit, not adjudicated: via shelves A1 and D1, a non-acting strike corresponds to a cofactor h ∈ [1, g − 1] with h ≡ ±u (mod 30), u = g⁻¹ mod 30, and that count is 2⌊g/30⌋ + [k0 < r] + [30 − k0 < r].

**3.6 Derived period and range** (C9; derived A11, N6).
- Claim: the derived period is q#, the derived range is x ∈ (√(q − A), √(q# − B)], and "each nonzero survivor class occurs once".
- Missing: the intended statement and its proof. The numbered text is not on disk, and the "occurs once" clause can only mean "at most once" or refer to the image of J.

**3.7 Derived-machine window thresholds** (C10; derived A14).
- Claim: in any span W of same-case lattice columns:
  - a striker h > W²/4 + B strikes at most 2 columns;
  - a striker h > W⁴/4 + (A + B)W²/2 + 1 strikes at most 1.
- Missing: a full argument. Only a sketch is recorded.

**3.8 Least split copy of a gear pair** (C11; shelves F3, F6, section 4 items 5–6).
- Claim: c is constant on each class of p mod 15d, and both gears act on the least split copy ⇔ c > d.
- Missing: the argument for mod 15d. The recorded argument gives constancy only mod 30d, and mod 15d is measured only for p < 6000.

**3.9 Twin strikers** (C12; derived A6, E7–E9).
- Claim: twin gears (g, g + 2) share leg strikers only from {11, 29, 31} (same case, g ≡ 29 mod 30) or {13, 23, 37} (cross case, g ≡ 11, 17 mod 30), and 7 never occurs.
- Missing: a case analysis of R_ab(2). The sharing law E7/E8 puts every shared factor in the finite set S(2), with H(2) = 37. The claim is checked on all twin gears ≤ 10⁶.

**3.10 Strict increase of X at every consecutive gear pair** (audit; range_line_map section 6).
- The iff "X increases strictly ⇔ a prime lies in (g² + A, g'² + A']" is elementary.
- Strict increase at every pair is proved only for revealed g.

**3.11 Type-C twins below about q²** (audit). The bound is stated imprecisely.

### 3B. Open items recorded by this round (no general statement claimed)
- **C1:**
  - The growth in q of min{E : E² − 4s is a square mod every odd p ≤ q}, a quantity of pseudosquare type. It is certified only as E_g > 10⁸ for q ≥ 43.
  - Checks not run: q = 157, 163, 173, 181, 191, and every q above 197.
- **C2:** h(43 + {53}) = 66 was not recomputed.
- **C3:**
  - There is no general explicit upper bound on h2(q); the trivial bound is h2(q) < ∏_{7≤p≤q} p.
  - Exact floors for gear sets larger than {7} were not tabulated.
  - Whether actual runs reach E_7 where E_7 > T_G was not tested.
- **C5:**
  - R8(q, r) for every q at a fixed r ≥ 1 by an argument without counting: not found, even for r = 1.
  - For every r at once, R8 is equivalent to the missed-copy hand-off target (R14, open).
  - Finiteness of the kill list at fixed r follows if the class condition (p_n − p_{n−r})² + b_c < K_c·p_{n−r} holds for all large n. That condition is open.
- **Lean:** the scratch files named in 2A are not in the kernel. The C1 parts, the chain-covering lemma and the bound d0 < √(2g) + 1 for C4, and the killer ordering, first lap, reduction, anchor and all-r equivalence for C5 are not formalised.
- The open questions of the earlier nodes are listed in `range_line_map.md` section 6.

---

## 4. Refuted as general

Each row is a statement that was stated or used for all q, or for all gears, together with the instance that makes it false.

**This round**

| # | Statement | Counterexample | Source |
|---|---|---|---|
| 1 | The sole strikers of survivor pairs are exactly the primes of (q# − 2)(q# + 2) in U⁻. | q = 11: q# + 2 = 2312 = 8·17², and AB_11 = {17}. At d = 17, A(low) = {17, 29}; at d = 51, A(high) = {17, 19}. The sole-striker set is empty. The inclusion holds for every q. | mirror_runs A8; C1 |
| 2 | B8 count ⌊R/G⌋ − 2n(k + 1) − 2 for a window that does not contain offset 0. | Every q ≥ 7: take window [L, L] with L = M − m, one free gear r_0 = q and k = 0. There is at most one aligned copy (none at q = 7), against the bound q − 4 ≥ 3. | C2 |
| 3 | B8 count with an upper gear above q#. | Every q ≥ 7: a prime g > q# strikes no range copy, so K = {0}, W = 0 assigned to g has relaxed classes but no representative in the range. | C2 |
| 4 | Ladder threshold: a relaxed pattern can be aligned into the range. | q = 43, gear 53: the relaxed pattern exists, but no class mod M lands in the range. Every covering is rigid: 23 covering tuples per pattern, 0 in range. | runs_gain A10; C2 |
| 5 | Unsigned landing rule: (d, e) and (d', e') land together ⇔ d ≡ d' or 15(d − d') ≡ ±1 (mod r). | Every modulus r ≥ 7. At r = 7 (a_7 = 4): (0, +1) lands at 4 and (1, −1) at 2, yet 15(0 − 1) ≡ −1. | C2 |
| 6 | "Equivalently" across the ladder chain (the E_7 rung adds nothing). | G = 11, H = 34: N_max = 7, but E_7(11, 7) = 36 > 33. | C3 |
| 7 | The condition for E_7 = T_G written with t_G mod 7 (record A4). | G = 223: G ≡ 6 and t ≡ 0 (mod 7), but σ ≡ 6, and E_7(223, 10) = 1011 > T = 996. | runs_gain A4; C3 |
| 8 | Kill-free window as an "iff" (the reverse direction without the case condition). | g = 127 (case 19): 113 divides 127² + 30 but misses c_127 = (16139, 16141). g = 29 (case 1): 23 divides 29² + 10 but c_29 = (869, 871) = (11·79, 13·67). | C5 |
| 9 | R8 is "settled in full by the window". | q = 7, r = 2: c_13 = (179, 181) is revealed, yet no gear of {7, 11, 13} meets the r = 2 window. At q = 7, R8 holds with no window witness at 49 of the 50 values r ≤ 50. | C5 |
| 10 | RS block law without the hypothesis X² ≥ q, or read at the fixed X = X_max(179). | When q > 4X² + 30, every block witness lies below q. At the fixed X_max(179), the witnesses' lower legs fall below q for q above about 7.45·10⁶⁹. | C5 |
| 11 | (2X)² < X# for every natural X ≥ 7. | X = 8, 9, 10: X# = 210 < 256, 324, 400. The exact domain is X = 0, X = 7 or X ≥ 11. | C6 |
| 12 | The window's lower end, read as every j ≥ 1 with 30j + 1 < q'². | Every q ≥ 29. For q = 31, q' = 37, j = 1: the legs 29 and 31 are ≤ q. | C6 |
| 13 | A uniform lap bound h(r) ≤ r + (q' − 2)M for every q. | q = 269, r = 262: h(r) = r + 270M. q = 277, r = 178: h(r) = r + 280M. | C6 |
| 14 | Window form W, clause 3, with "descendant" read as any class ≡ r_C (mod M_P). | 121 mod 210 at P' = 113 passes the criterion vacuously, but 121 = 11² is not a gear. With the record's Desc_C(P') the clause stands. | C4 |
| 15 | Four distinct leg classes ⇔ g ∤ (q# − 2)(q# + 2), for every prime g ≥ 7 (P2 (vi), P3 (v)). | Every machine gear g ≤ q: a ≡ 0 (mod g), so the classes collapse to ±t_g, and h⁻¹q# ≡ 0 meets class 0. The true forms are g ∤ (q# − 2)q#(q# + 2) for every prime g ≥ 7, and the literal form for g > q. | RangeGen2, RangeGen3 |
| 16 | Acting-free revealed ⇔ both legs are prime and above √(q#), without the range bound 30j + 1 ≤ q#. | q = 7, X = 14: copy 13 has legs 389 (prime) and 391 = 17·23, and none of 7, 11, 13 divides a leg. | RangeGen5 |

**Earlier rounds**

| # | Statement | Counterexample | Source |
|---|---|---|---|
| 17 | The 2k bound: k upper gears never strike 2k + 1 consecutive survivors. | k = 1 at q = 47: gear 53 strikes 4 consecutive survivors, 247507649335490 + {0, 7, 53, 60}. k = 2 at q = 37: 2749631283 + {0, 9, 29, 30, 41}, struck by gears 41 and 43. k = 3 at q = 37: three gears strike 7 consecutive survivors. | mirror_runs B5–B7 |
| 18 | A bound B_1(q) ≤ c for all q, with c ≤ 6. | q = 293: a one-gear acting run of 7 elements. | runs_gain |
| 19 | The unrestricted (16′) equality T(s_i) ∩ T(s_{i+1}) = A(s_i) ∩ A(s_{i+1}) for consecutive survivors (T: upper strikers; A: acting upper strikers). | q = 107..139: gear 241 strikes consecutive survivors 1197 and 1213 but acts on neither. | field_to_range |
| 20 | Revealed status is periodic with period P, the product of the gears of a cut. | P = 6685349671, the product of the gears 7..31: 29 and 31 act on copy 1 + P but not on copy 1. General form: "both legs prime" has no period N ≥ 1 (RangeGen5). | field_to_range |
| 21 | Every upper gear strikes some survivor. | q = 7: gears 11 and 13 strike none of the survivors 1, 2, 5, 6. | field_to_range |
| 22 | Consecutive g-struck survivors are spaced t_g or g − t_g apart. | q = 11, g = 13: 23 → 36. | field_to_range |
| 23 | Three consecutive g-struck survivors span exactly g. | q = 11, g = 13: survivors 16, 23, 36 span 20. | field_to_range |
| 24 | The high member's deletion always comes from a gear ≤ √ρ. | q = 13, d = 143: A(high) = {131}, and √ρ = 122.54. | mirror_runs A7 |
| 25 | No survivor pair has A(low) = A(high) = {g}. | q = 17, d = 3059: gear 23 on 209369 and 301139. More at q = 19 and 23. | mirror_runs A8 |
| 26 | The palindrome pair P_{d*} is doubly deleted. | Not doubly deleted at q = 7, 11, 17, 19. At q = 7 the legs are 59, 61 and 149, 151. | mirror_runs A10 |
| 27 | Acting pair law, literal ⇐ direction (record B9 form). | g = 41, j = 2749631283, D = 11: copy j + 11 is not struck. | mirror_runs B9 |
| 28 | Gear g strikes no copy j with 30j + 1 < g². | Gear 13 strikes copy 3 (91 = 7·13 < 169). There are 1043 counterexamples among gears < 500 with j < 3000. | map 5 |
| 29 | Every shelf contains a revealed copy. | Shelf 17 = {10, 11} and shelf 29 = {28, 29, 30, 31} contain none. | shelves |
| 30 | Every head stretch holds a revealed copy. | g = 37: every copy in 46..52 has a composite leg. Also g = 17, 29, 41 and 149. | two-paths |
| 31 | Each survivor class has exactly one copy in the range. | Class 0 has none (q = 7..19). The repaired form is 2B.2. | two-paths |
| 32 | R12: the ceiling system is a total cover. | Fails at every q = 7..31, with 3, 3, 6, 10, 19, 62, 168, 585 gears uncovered. | two-paths |
| 33 | Every revealed missed copy in machine q's range has λ(g) > q (derived (10)). | q = 13: c_7 = (59, 61) has λ = 0, c_11 = (149, 151) has λ = 0, and c_13 = (179, 181) has λ = 7. | derived |
| 34 | Reduction to the capped strip (derived (10), core). | Cap(13) = {17, 23}, and neither is revealed. The reduction fails at 50,139 of the 148,930 prime levels in [7, 2·10⁶]. | derived |
| 35 | Heredity of chain. | {c_7} ∪ {c_g : g ≥ 43,849,311,221} leaves machine 59 unserved. | derived |
| 36 | Every K consecutive gears include one with a revealed cell among the first N. | False for every N and K [imports CRT and Shiu]. | derived |
| 37 | V2: for every window exposure, some copy of the machine orbit mod M_q in the range is revealed. | q = 19, n = 33: the machine orbit of 33/5 mod 323,323 has 16 copies in the range, none revealed. | derived 3.F |
| 38 | S4: every exposure has an exact transfer into the range. | False for every q ≥ 29. For (2, 2), n = 12: 144 − 100 = 44 ≡ 2 (mod 7). | two-paths |
| 39 | The reveal status of a copy at position ≥ 2 determines the status of c_g or of c_{g'}. | All four combinations occur: copies 14, 184, 8 and 267. | kernel |
| 40 | A map from revealed copies to revealed missed copies that preserves service exists. | Copy 8 = (239, 241) serves [11, 239), and no revealed missed copy lies in [239, 2308]. | kernel |
| 41 | The anchor sentence of kernel2 (4). | r = 11: m_11 = 179 and Anc(179) = {7, 11}. | kernel2 |
| 42 | The written tower form P^(k+1) = {y ∈ L_k : 15 ∤ J^k(y)}. | y = 11, k = 1: J(11) = 5 is not a column. | kernel2 |
| 43 | log(top column)/log(period) = 2^(−k) at finite q. | q = 101: the ratios are 0.5000, 0.2612 and 0.1450 for k = 1, 2, 3. The identity holds only as a limit. | kernel |

Wordings refuted without a general statement at stake (miscounts, sign slips, literal readings) are listed in `range_line_map.md` section 5 and in the Refuted blocks of each record.

---

## 5. Instance data (checks only, not proofs)

**Nothing in this section is a proof.** Each item holds at the stated q or over the stated range. Items attached to a general statement are checks of it. Items attached to a false statement are counterexamples, and those are listed in section 4.

### 5.1 Range witnesses and counts
- Lean `small_cases`: RangeStatement q at every q in [7, 23], with witness p = 29.
- Least revealed copy above q:

  | q | Least revealed copy |
  |---|---|
  | ≤ 23 | copy 1 |
  | 29..53 | c_7 (copy 2) |
  | 59..139 | c_11 (copy 5) |
  | 149..173 | c_13 |
  | 179 | copy 8 |

- For every prime q in [7, 3000], the first twin copy above q lies below q², inside the window.
- Revealed range copies:

  | q | 7 | 11 | 13 | 17 | 19 | 23 | 29 |
  |---|---|---|---|---|---|---|---|
  | Count | 4 | 19 | 152 | 1517 | 19017 | 298408 | 6,155,197 |

- Machine 29 settles NS and Good at all 6,155,198 nodes ≤ 29# − 2.

### 5.2 Chains, hand-offs, anchors
- **Chain:** chain(T) holds, with |T| = 388,397 (j ≤ 10⁷).
- **Hand-offs below gear 200,000:**
  - all 327 missed-copy hand-offs hold (328 revealed gears: 233 case 1, 95 case 19);
  - the hand-off target holds at every prime q in [7, 39,377,242,978];
  - gear 198437 alone serves [31, 39,377,242,978].
- **Hand-offs below 10⁷:** 6,548 revealed gears; every hand-off is strong; the largest gap is 14,160.
- **C7:** the 14 gears ≤ 1200 meet (r, r# − 2] for every prime r in [7, 2·10⁵].
- **Near X(7) and X(13):**
  - the largest revealed gear ≤ X(7) is X(7) − 3069;
  - no revealed gear lies in (X(13) − 107,292, X(13)].
- No c_g is revealed for g in 17..73.
- **Tails and anchors:**
  - tail(7) is empty;
  - for r = 11..61, tail(r) is non-empty and m_r > r (exhaustive for r ≤ 23 only);
  - Anc(m_r) = {r} at r = 13..61.
- **Crossing pairs:** crossing pairs (t_r, t_r⁺) exist for r = 7..43, and t_r is not a missed copy for r = 11..43.
- **Measured bounds:** v(m_r) + 2 > r# and shelf(v(m_r)) > isqrt(r#) for r = 7..61. This was demoted from PROVED to MEASURED.
- **Direction-lemma brackets:** failures at 17#, 19# and 23#; none at 11# or 13#.
- **Base 6:** RANGE₆ witnesses and anchor rows for r ≤ 67.

### 5.3 Runs, struck stretches, h2
- **B_1(q):**

  | q | 11..31 | 37, 41, 43 | 47, 53 |
  |---|---|---|---|
  | B_1(q) | 2 | 3 | 4 |

  The upper bounds at q = 37..53 rest on the lane's exhaustive residue search.
- Long maximal acting runs: 5 elements at q = 89, 97, 101; 6 at q = 137; 7 at q = 293.
- **h2(q):**

  | q | 11 | 13 | 17 | 19 | 23 | 29 | 31 | 37 | 41 | 43 | 47 | 53 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | h2(q) | 4 | 5 | 7 | 12 | 18 | 25 | 31 | 38 | 49 | 59 | 69 | 86 |

  - Values at q = 43, 47 and 53 are lane-checked only.
  - Lower bounds are known at q = 59, 97, 127, 199 and 293.
  - This round recomputed h2 = 4, 5, 7, 12, 18, 25 at q = 11..29 by full-period scan. Y_7 = 2, 3, 4, 5, 6, 8 there, and Y = 13, 21, 25, 33, 39 at q = 11..23. At these q, ⌊Y/30⌋ ≤ 1, so R7's second inequality is only weakly tested.
- **No-5-run comparison** (C3 item 10):
  - needed H: 70, 64, 53 for 4-runs of gears 47, 43, 41, and 32 for a 3-run of gear 31;
  - against h2 = 69, 59, 49, 31.
  - T_G(5) = 2G for primes 11..2000, which is general since K(3) = 3.
  - h2(G) ≤ 2G at G = 11..29.
- There are 792 aligned (0, 4, 59, 63) starts of gear 59 at q = 53. f(K) is checked for K ≤ 60 (this round K ≤ 150), and K(n) for n ≤ 40 (this round n < 400).
- **Multi-gear records:**
  - Γ_q(5) = 10, 13, 19, 28, 36, 41, 47 at q = 11..31;
  - Γ_37(2..5), Γ_41(2), and Γ_29, Γ_31 at 9..12;
  - two- and three-gear record tables; the R0, R, G1 and G2 tables;
  - Γ_2 incidences at q = 7..31, with realised depth 2.
- The T∩T = A∩A equality holds at q = 7..31.
- **C2 at q = 43, gear 53:**
  - (0, 7, 53, 60) and (0, 46, 53, 99) have 0 coverings;
  - so a relaxed maximum of 3 needs only h(43 + {53}) ≤ 106;
  - the lane's value h = 66 was not recomputed.

### 5.4 Mirror and gain series
- **Series at q = 7..23:**
  - |Loss| = 0, 1, 2, 7, 19, 75;
  - |G_q| = 0, 2, 14, 73, 882, 11133;
  - also |Gain_full|, |B|, Q\*, the trichotomy tallies, and the A3 pair tallies at q = 23.
- **Acting-both sets:**
  - raw AB_q = {}, {17}, {}, {23, 31, 179}, {29, 109} at q = 7..19 (this round);
  - {587, 4801} at q = 23 (mirror_runs);
  - each equals the primes of q# ∓ 2 in U⁻.
- **Sole-striker equality** is measured at q = 13..23 and fails at q = 11.
- **Band existence clause:** holds at q = 11..79, and this round at q ≤ 197 except 157, 163, 173, 181, 191.
  - Example: 761 divides 79# + 2, E ≈ 2.114·10²⁷, and the first survivor is at k = 41.
- **Band-width certificate:** the 12 factorisations with g an odd prime and E ≤ 10⁸, as (q, side, E, g):
  - (11, q# + 2, 51, 17)
  - (17, q# − 2, 1247, 179)
  - (17, q# − 2, 8203, 31)
  - (17, q# − 2, 11075, 23)
  - (19, q# − 2, 167207, 29)
  - (19, q# + 2, 44385, 109)
  - (23, q# − 2, 18433, 4801)
  - (23, q# + 2, 189441, 587)
  - (29, q# + 2, 821931, 3917)
  - (31, q# − 2, 5220437, 19139)
  - (31, q# − 2, 16376863, 6121)
  - (41, q# + 2, 73694985, 2009461)

  The largest least non-residue primes are p_E = 113 (at E = 923663, q# − 2) and 103 (at E = 38850999, q# + 2).
- An empty-band square occurs, for q ≤ 3000, only at q = 7 (g = 8, which is not prime).
- **Palindrome and core:**
  - d\* = 3, 5, 5, 5, 11, 11 at q = 7..23, and d\* lies in the core at q = 11..23;
  - copy M − 1 is deleted by an acting upper gear at q = 11..23;
  - the c\* statuses and the central pairs are recorded.

### 5.5 Free covers of the upper set
- C_7^free = ∅, by exhausting 143 phases; the best cover takes 3 of 4.
- C_11^free = ∅, by branch-and-bound and HiGHS; the best cover takes 42 of 44, leaving {2, 27} uncovered, realised at k = 22370436767022.
- C_13^free is undecided, and q ≥ 17 was not attempted.

### 5.6 Escape classes, λ, doom
- Every escape class at levels 7, 11, 13, 17 and 19 holds a revealed gear. The largest least witness is 42,561,936,551. Level 23 was not run.
- Level-13 first-struck statistics: 1821 / 121 / 10.
- λ computed from its definition:
  - for g ≤ 10⁷, min λ/g = 0.9398;
  - this round, on every gear ≤ 2·10⁶, λ = h\* at all 148,925 gears with h\* > 2g/3, and no gear ≥ 41 has 3λ ≤ 2g.
- Cap(P) at P = 7, 11, 13, 17, 19, 23, 29, 31: {11, 13}, {13}, {17, 23}, {19, 23}, {23}, {29, 31, 37}, {31, 37}, {37, 41, 43}.
  - Among prime levels up to 59999, the Cap bound is attained only at P = 23.
- Chain certificate: the tightest step is 10729 → 15913.
- The shape model strikes every gear in 13..2·10⁶ before its cap.
- The κ = 4 entering kills up to 10⁷ are only 23→31, 67→83 and 199→227.
- |Esc| values are listed for levels 7..23; the counts themselves are exact formulas.

### 5.7 Kills, stretches, inert runs
- **Missed-copy counts:**
  - revealed c_g: 233 (case 1) and 95 (case 19) below 200,000;
  - n-th missed-copy counts for n ≤ 3.
- **Kill lists:**
  - complete for r ≤ 12 and g ≤ 10⁸; the largest killed gear is 4517;
  - this round recomputed them over all 5,761,453 gears below 10⁸, with r-step gap maxima 220, 248, 300, 342, 390, 396, 414, 462, 488, 504, 546, 564;
  - the conditional extension to 4·10¹⁸ holds for r ≤ 9.
- **Kills to 3·10⁵:** 92,774 kills among the 25,994 gears ≤ 3·10⁵.
  - The least κ equals K(b, c) in 15 of the 16 (leg, class) pairs.
  - There are 14 boundary kills, all with κ = 2, all in classes 19 and 17.
- **R8 truth tables** at q = 7, 11, 13, 17, 19, 23 for r ≤ 50 (|G(q)| = 3, 12, 37, 124, 440, 1745): 0 cases where the window holds but c_g is killed.
- **Block-law numbers** (RS): X\* = 255,271,782.94; ρ_RS(10¹⁵) = 888,539.34; ρ_RS(10¹⁶) = 2,637,438.23.
- **Witness c_198437:** legs 39377242979 and 39377242981 are prime; it serves 31 ≤ q ≤ 39,377,242,973.
- **Tables:** K(d) and K\*(d) for d ≤ 30; the F10 and F30 families to 10⁶; the H(w) table.
- **Stretches and regions:** the fully covered stretches are exactly those of 17, 29, 37, 41 and 149 (g ≤ 10⁸); the empty regions are 17, 29 and 41.
- **Other tables:** D_all/D_sq for y = 13..29. The inert-run maxima are 8 (case 1) and 13 (case 19) for h ≤ 2·10⁵.

### 5.8 Path B (offset columns)
- V1 orbit statement: exact at q ∈ [7, 61], BPSW-probable at q = 67..113.
- Forced pairs at q = 7..23: 3, 16, 65, 166, 539, 1709, with 0 counterexamples. The preservation itself is general.
- "No source leg strikes r" is measured for 187,237 exposures.
- Exact downward transfers reach at most copy 27.
- k\* has maximum 9.
- The X_fit, cost and B2.8 tables.

### 5.9 Derivation tower and limit
- W = L ∩ ℕ \ {1, 29} is empty below 2·15¹⁰.
- No column other than 1 has more than 2 consecutive revealed levels; 14 columns have exactly 2.
- Delta census: maximum 11 below 10⁷, and 22 below 2·15¹⁰.
- 37 persistent strikers below 3000.
- Z_h = ∅ for 107 primes.
- The seed-period lists stop at 2000, and the level-2 zero-class "exactly at" list covers h ≤ 3000.
- X(1) = 80434, X(7) = 43,849,291,330, and the P² reach endpoints are 1549 and 1,146,941.

### 5.10 Miscellaneous single-machine facts
- All 495 classes of machine 13 hold a revealed copy below 10⁶; the latest such copy is 35583.
- M\*_29 spares exactly copy 2.
- There are 20 {59, 61}-smooth numbers in (1, 29#].
- The shelf–survivor identity was enumerated only to machine 23.
- Per-gear strike minima at q = 11..23: 1, 5, 21, 76, 350.
- The escape-class checks in the (c′) mechanism resolved below 6·10⁷.
- The shelves of 7, 11 and 17 are the only ones of size ≤ 2 among gears whose next gear is ≤ 2·10⁶.

### 5.11 Lap maxima (C6)
The largest number of laps for q' to strike and act at q = 7..23:

| q | 7 | 11 | 13 | 17 | 19 | 23 |
|---|---|---|---|---|---|---|
| All r in [0, M) | 8 | 6 | 12 | 16 | 17 | 23 |
| Survivor classes r in [1, M] (the G1 record) | 6 | 6 | 12 | 16 | 16 | 23 |

- The difference between the rows is only the range of r.
- The exact all-r maxima at q = 29, 31, 37, 41, 43 are 26, 20, 27, 33, 39.
- Every value is below q' − 1.

### 5.12 Checks run this round on general statements
These checks confirm general statements; they do not prove them.

- **P1 split:** q = 7..19, 0 mismatches.
- **C8 count:** gears < 3000, 0 mismatches.
- **C6 wall:**
  - X < 4000: 0 failures;
  - natural X ≤ 4000: "nextprime(X)² < X# ⇔ X ≥ 7";
  - q'² ≤ q# + 1 for every q < 400;
  - silent classes at X = 7, 11, 13 (15, 135 and 1485 of them): 0 violations.
- **C1:**
  - gcd identities at q = 7..19;
  - cofactor, survivor and acting laws at q = 11..23 (692, 7,058 and 7,108 cases at q = 17, 19, 23);
  - root test: 52 of 52.
- **C2:**
  - 8,828 boxes at every prime q from 7 to 131, with 0 mismatches;
  - landing rule over all 80 moduli coprime to 30 in [7, 301];
  - 4,551 B8-configuration boxes.
- **C3:**
  - 489,692 maximal runs and 503,988 sub-runs at q = 7, 11, 13, 17 (G < 250);
  - E_7 against T in 9,400 cases (primes 11..1500, N ≤ 40);
  - the N_max closed form in 593,804 cases.
- **C4:**
  - top-band law on 765,392 pairs with g ≤ 20000;
  - Cap and window criterion at prime levels 7..59999.
- **C5:**
  - all laws on gears ≤ 3·10⁵;
  - the 16 κ certificates;
  - the 8 sharp witnesses.
