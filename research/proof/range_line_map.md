# Range line: map of results

This map covers theory-tree nodes R5.f.xxxv.c.xxv to R5.f.xxxv.c.xxxi (2026-09-23 to 2026-09-25), the four round records and the Lean kernel. Every statement and number is taken from those sources.

**Current status of every statement: see `range_generality_ledger.md`.** Rounds after this map (nodes c.xxxii to c.xxxvii, 2026-09-25 to 2026-09-26) are recorded there and in the tree. Under the owner's rule of 2026-09-26, only statements holding for every machine size count; the ledger sorts every statement of the range line into general-in-Lean (21 kernel modules), general on paper, claimed but not proved, refuted as general, and instance data.

Brackets give the source of each item:
- [shelves X]: `research/proof/shelves_cofactors_2026-09-23.md`, item X (node c.xxviii).
- [two-paths X]: `research/proof/range_two_paths_2026-09-23.md`, item X (node c.xxix).
- [derived X]: `research/proof/derived_machine_2026-09-24.md`, item X (node c.xxx).
- [kernel X]: `research/proof/range_kernel_2026-09-25.md`, item X (node c.xxxi).
- [c.xxv], [c.xxvi], [c.xxvii]: the node text in `research/proof/theory_tree.md`.

Status words:
- PROVED: the statement stands as proved in an adjudicated round record or tree node. "Imports X" names an outside theorem the proof uses.
- Stands: the statement stands in an adjudicated record, and the record gives checks rather than marking it proved.
- LEAN: proved in the Lean kernel (section 4).
- MEASURED: a computation over the stated range. It is not a proof.
- REFUTED: shown false, with an instance.
- OPEN: not established.

---

## 1. The range statement, and why it gives the twin prime conjecture

### 1.1 Terms

- **q#** (primorial): the product of all primes up to q. For example, 7# = 210.
- **Gear**: a prime. The gears laid over the copies are the primes above 5 [c.xxv]. The primes 2, 3 and 5 form machine 5, which lays out the copies.
- **Machine q**: the primes up to q, taken together.
- **Period**: the length after which a machine's pattern of divisibility repeats. Machine q has period q#. Machine 5 (2, 3 and 5) has period 30.
- **Column**: column n is the pair (6n − 1, 6n + 1).
- **Lap**: one period of machine 5, which is 5 columns. Lap j holds the columns n = 5j + e, where e = 0..4 is the offset. Machine 5 leaves offsets 0, 2 and 3 open.
- **Copy**: copy j is the pair (30j − 1, 30j + 1). It is the column at offset 0 of lap j. It is a copy of machine 5's known opening (−1, 1), carried to the multiple 30j.
- **Legs**: the two members of a copy. The lower (minus) leg is 30j − 1 and the upper (plus) leg is 30j + 1.
- **Strike**: gear g strikes copy j when g divides one of its legs. For j ≥ 1 this is 30j ≡ ±1 (mod g). Equivalently, j ≡ ±a_g (mod g), with a_g = 30⁻¹ mod g.
- **Acting**: gear g acts on copy j when g² ≤ 30j + 1. Write a struck leg as g·h. Then acting is exactly h ≥ g: the gear is the smaller factor of the leg it divides [c.xxvii; shelves A1].
- **Revealed**: copy j is revealed when no acting gear strikes it. For j ≥ 1 this holds exactly when both legs are prime [shelves A2]. Copy 0 is the one exception: nothing acts on it and nothing strikes it.
- **Range of machine q**: the copies j with 30j − 1 > q and 30j + 1 ≤ q#.
- **Window of machine q**: the copies of the range with 30j + 1 < q'², where q' is the next prime after q.

### 1.2 The statement

- **Range statement at q** [c.xxv]: some copy of machine q's range is revealed. That is: some j with 30j − 1 > q and 30j + 1 ≤ q# is struck by no gear at or below √(30j + 1).
- **RANGE**: the range statement at every prime q ≥ 7.
- **Lean form** (RangeHandoff.lean): `RangeStatement q := ∃ p, q < p ∧ p + 2 ≤ primorial q ∧ p.Prime ∧ (p + 2).Prime`. It asks for any twin pair (p, p + 2) with q < p and p + 2 ≤ q#. A revealed copy of machine q's range is such a pair, with p = 30j − 1; the Lean form also admits twin pairs off the copies.

### 1.3 Why it gives the twin prime conjecture

- **LEAN `range_implies_unbounded`**: if `RangeStatement q` holds at every prime q ≥ 7, then for every N there is a prime p > N with p + 2 prime.
- The proof picks a prime q ≥ N + 7. The pair it supplies lies above q, hence above N.
- **LEAN `small_cases`**: the range statement holds at every q in [7, 23]. The witness is p = 29, the pair (29, 31) = copy 1.

### 1.4 Range and window

- **Recorded difference** [c.xxvi]: the window fixes the cut and the interval in advance. The range certifies copy j with the gears in [7, √(30j + 1)], a set that depends on j.
- **Fixed cut, PROVED** [c.xxvi]: "for every q, some copy of the range is struck by no gear of [7, √(q#)]" is exactly the window statement at the cut X = isqrt(q#). It is proved both ways.
  - A copy surviving every gear up to X, with 30j + 1 ≤ q#, has its lower leg above X. So it is a twin above √(q#).
  - Enumerated whole at q = 13, 17, 19 (149, 1507 and 18991 survivors): zero non-twins, zero below the floor, zero twins above the floor missed.
- **One period** [c.xxvi; two-paths B3.2; c.xxix]: the range is one period of the machine's own gears.
  - A **survivor class** of machine q is a class of copies mod q#/30 that no gear from 7 to q strikes.
  - Survivor class 0 has no copy in the range. Each of the ∏(g − 2) − 1 nonzero survivor classes has exactly one copy in the range.
  - Checked exactly at q = 7..23 [c.xxvi]; re-run at q = 7..19 [two-paths B3.2].
- **MEASURED** [c.xxv, c.xxvi]: the first twin copy above q is copy 1 for q ≤ 23, copy 2 for q = 29..53 and copy 5 for q = 59..89. For every prime q in [7, 3000] it lies below q², inside the window.

---

## 2. Exact equivalent forms

### 2.1 Service intervals and the chain

- **Service**: a revealed copy with legs (p, p + 2) serves machine q when q < p and p + 2 ≤ q#.
- **σ(y)**: the least prime whose primorial is ≥ y.
- **Service interval, PROVED** [derived C2]: a copy with legs (p, p + 2) serves exactly the primes q in [σ(p + 2), p).
  - LEAN `twin_serves`: such a pair with q < p and p + 2 ≤ q# witnesses `RangeStatement q`.
  - LEAN `serves_interval_of_le`: one pair with p + 2 ≤ q₁# and q₂ < p serves every q in [q₁, q₂], because primorial is monotone.
- **Chain**: list a set S of revealed copies by lower leg, p_0 < p_1 < …. Then chain(S) holds when:
  - S is infinite,
  - p_0 + 2 ≤ 210, and
  - p_{k+1} + 2 ≤ p_k# for every k.
- **RANGE ⇔ chain(T), PROVED** [derived C2]. Here T is the set of all revealed copies.
- **Upward-closed, PROVED** [derived C3]: a set containing a chain is a chain.
- **Region form, PROVED** [derived C6].
  - **Shelf**: shelf(g) is the copies [j_g, j_{g'}), with j_g = ⌈(g² − 1)/30⌉ and g' the next gear. It runs from j_g up to, but not including, the first copy of the next gear's shelf.
  - **Region of g**: the copies [J_g, j_{g'}), with J_g = ⌈(g² + 1)/30⌉. In case 19 (defined in 2.2) the region is shelf(g). In case 1 it is shelf(g) minus the head copy j_g, whose upper leg is g².
  - **Position**: position n of gear g is copy J_g + n − 1. Position 1 is the missed copy c_g (defined in 2.2).
  - The regions do not overlap. They tile every j ≥ 2 except the case-1 head copies.
  - Every revealed j ≥ 2 lies in the region of its top acting gear. Copy 1 is revealed and lies below every region.
  - chain(revealed copies in regions) ⇔ RANGE.
- **Gear form** [c.xxvii]: "the gear set contains two gears differing by 2 whose midpoint is divisible by 30, the lower above q and the upper at most q#". The node records this as the range statement restated in gear terms.

### 2.2 The missed-copy chain (a sufficient form)

- **Cases** (LEAN `sq_mod30_cases`): a prime g ≥ 7 has g² ≡ 1 or 19 (mod 30), and exactly one of the two holds.
  - Case 1: g ≡ ±1, ±11 (mod 30). Set (A, B) = (28, 30).
  - Case 19: g ≡ ±7, ±13 (mod 30). Set (A, B) = (10, 12).
- **Missed copy c_g** [shelves B1]: the lowest copy at or above g² that g does not strike. It is copy J_g = ⌈(g² + 1)/30⌉, with legs g² + A and g² + B. Section 3.3 has its laws.
- **Revealed gear**: a gear g whose c_g is revealed. G is the set of revealed gears. S_1 is the set of revealed missed copies.
- **Hand-off**: for consecutive members g_i < g_{i+1} of a sequence of revealed gears, the hand-off g_i → g_{i+1} holds when g_{i+1}² + B ≤ (g_i² + A)#. Four equivalent forms of one hand-off are given below.
- **Hand-off equivalence, PROVED** [two-paths A1.14]:
  - "Every machine q ≥ 7 has a revealed missed copy in its range" holds for all q iff the revealed gears 7 = g_0 < g_1 < … form an infinite sequence with every hand-off holding, g_{i+1}² + B ≤ (g_i² + A)#.
  - This target implies RANGE and infinitely many revealed c_g.
  - The first interval [7, 59) is served by g_0 = 7, since 61 ≤ 210.
- **One hand-off, four forms, PROVED** [derived C1]. Put P = (g² + A_g)#. The record's constants are (A, B, B') = (28, 30, 30) in case 1 and (10, 12, 10) in case 19: A_g is taken for the case of g, and B' for the case of g' [derived 2.3]. Then:
  - g'² + B' ≤ P ⇔ g'² ≤ P − 41 ⇔ J_{g'} < P/30 ⇔ σ(g'² + B') ≤ g² + A_g.
  - X(g) = isqrt(P − 41). X(7) = 43,849,291,330. X(11), X(13) and X(79) have 29, 36 and 1341 digits.
  - X is non-decreasing along the gears.
  - X increases strictly from g to the next gear iff a prime lies in (g² + A, g'² + A']. This is PROVED when g is revealed, and MEASURED for g'² + A' ≤ 3·10⁸.
  - chain(G) ⇔ G meets (g, X(g)] for every g in G [derived 2.3; C3].
- **Slack step, PROVED** [derived C4; kernel 2.3 item 17].
  - Write p_i = g_i² + A_i for the lower leg of c_{g_i} and u_i = p_i + 2 for its upper leg. Put Δ_i = θ(p_i) − ln(u_{i+1}), with θ(x) = ln(x#). Then Δ_0 = ln(59#/151) = 43.9908.
  - If g_{i+2} ≤ p_{i+1} (a strong hand-off), then Δ_{i+1} > Δ_i.
  - More generally, u_{i+2} < p_{i+1}·u_{i+1} implies Δ_{i+1} > Δ_i.
- **Forward arrows, PROVED** [derived 2.3, C7, N3]. Only these directions are proved:
  - chain(S_1) = chain(G) ⇒ chain(S_{≤N}) and chain(S_stretch) ⇒ chain(revealed copies in regions) ⇔ RANGE.
  - The record uses S_{≤N} without defining it. On the map's reading it is the revealed copies at positions ≤ N of their gear. S_stretch is the revealed copies in stretches: the stretch of g is the run of copies from c_g that g does not strike, cut at the next gear's shelf (formulas in 3.4).
  - chain(S_1) ⇒ chain(SQ_unit) ⇒ chain(SQ_all) ⇒ RANGE.
  - SQ is the square family: the copies c_m = copy ⌈(m² + 1)/30⌉ for integers m, not only primes. SQ_unit takes the units m and SQ_all takes all m. The family bound is m ≥ 7.
  - The gears acting on c_m are exactly the primes 7..m, for every m ≥ 0 except m = 6 and m = 10 (PROVED).

### 2.3 The converse reduction [kernel 2.1]

Terms for this subsection:
- Rev: the lower legs of all revealed copies. S_1 is read as the lower legs of the revealed missed copies.
- x⁺: the next member after x in the set in question.
- m_r (the anchor at r): the largest m in S_1 with m + 2 ≤ r#.
- tail(r): the members of Rev in (m_r, r# − 2]. These are revealed copies above the anchor whose upper leg is still ≤ r#. An anchor value m is live when tail(r) is non-empty for some r with m_r = m (the reading of kernel 2.1 item 7).
- v(m): the largest member of Rev below m⁺.
- Copy M dominates copy R when M serves every machine that R serves.

Results:
1. **Domination criterion, PROVED.** With p_R and p_M the lower legs: M dominates R ⟺ p_R ≤ p_M ≤ σ(p_R + 2)# − 2. Checked on 21,918 pairs with 0 mismatches.
2. **Direction lemma, PROVED.** Let R be revealed at position ≥ 2 of region g.
   - Neither c_g nor any missed copy with lower leg below p_R dominates R.
   - c_{g'} dominates R ⟺ c_{g'} is revealed and g'² + B' ≤ σ(p_R + 2)#.
   - The bracket fails only when a primorial lies in [p_R + 2, g'² + B'). These intervals are disjoint across regions, so each primorial fails the bracket in at most one region.
   - MEASURED failures: region 709 (17#), region 3109 (19#), region 14929 (23#). There are none at 11# or 13#.
3. **Non-dominated set, PROVED.** Let N be the union of all tail(r). Then RANGE ⟺ chain(S_1 ∪ N).
   - Least tail copy by r: 7 none; 11: 239; 13: 17489; 17: 293999; 19: 9542369; 23: 197994299; 29: 6423863099; 31: 199288140509; 37: 7399721980769; 41: 303863920848119; 43: 13082663386240769.
4. **Two-copy cover law, PROVED.** For R in N with r = σ(p_R + 2):
   - R's service is covered by S_1 ⟺ it is covered by m_r and m_r⁺ ⟺ r ≤ m_r and m_r⁺ + 2 ≤ m_r#.
   - The hand-off m_r⁺ + 2 ≤ m_r# already forces m_r > r.
5. **Crossing form, PROVED** for sets of primes ≥ 29. Put t_r(S) = the largest s in S with s + 2 ≤ r#. Then chain(S) ⟺ s_0 + 2 ≤ 210 and t_r(S)⁺ + 2 ≤ t_r(S)# for every prime r ≥ 7.
   - The set {5, 43, 83} is outside the scope.
   - Crossing pairs (t_r, t_r⁺) for all revealed copies: 7: (179, 239); 11: (2129, 2309); 13: (29879, 30089); 17: (510449, 511109); 19: (9699509, 9701819); 23: (223091549, 223095149); 29: (6469693079, 6469694039); 31: (200560489859, 200560490549); 37: (7420738130759, 7420738140689); 41: (304250263523909, 304250263527479); 43: (13082761331668079, 13082761331670209).
   - MEASURED: t_r is a missed copy only at r = 7, and t_r⁺ ≠ m_r⁺ for every r = 7..43. "t_r is not a missed copy" is measured for r = 11..43 only.
6. **RANGE forces the S_1 hand-off at every anchor that is not live, PROVED.** Given RANGE, chain(S_1) ⟺ m_r⁺ + 2 ≤ m_r# at every r with a non-empty tail. A missing successor counts as false.
   - Anchors m_r → m_r⁺, with gears in brackets:

| r | m_r → m_r⁺ | gears |
|---|---|---|
| 7, 11 | 179 → 6269 | 13 → 79 |
| 13 | 17189 → 49739 | 131 → 223 |
| 17 | 292709 → 546149 | 541 → 739 |
| 19 | 9541949 → 11498909 | 3089 → 3391 |
| 23 | 197993069 → 247464389 | 14071 → 15731 |
| 29 | 6423862229 → 6509746499 | 80149 → 80683 |
| 31 | 199288137899 → 200812430669 | 446417 → 448121 |
| 37 | 7399721979059 → 7427625284699 | 2720243 → 2725367 |
| 41 | 303863920846259 → 304251655581749 | 17431693 → 17442811 |
| 43 | 13082663386239869 → 13083062802401099 | 114379471 → 114381217 |

   - MEASURED: tail(7) is empty, and tail(r) is non-empty for r = 11..43. So every anchor for r = 7..43 is live; 179 is live through r = 11.
7. **Converse-failure form, PROVED.** RANGE ∧ ¬chain(S_1) ⟺ RANGE ∧ (S_1 finite, or some consecutive pair (m, m⁺) in S_1 has m < σ(m⁺ + 2) ≤ v(m)).
   - An equivalent form replaces the second disjunct by m⁺ + 2 > m#.
   - v(m_r) for r = 7..43: 6089, 6089, 49529, 545789, 11498159, 247464209, 6509744609, 200812422119, 7427625283019, 304251655578329, 13083062802399299.
8. **Witnesses at a live anchor, PARTIAL.** At a live anchor, t_r witnesses every q in [m_r, t_r).
   - The only chain inequality of RANGE with m_r⁺ as successor is σ(m_r⁺ + 2) ≤ v(m_r), and v(m_r) > m_r.
   - So the S_1 hand-off σ(m_r⁺ + 2) ≤ m_r is not one of RANGE's chain inequalities.

### 2.4 Measured coverage

- Revealed copies in the range at q = 7, 11, 13, 17, 19, 23: 4, 19, 152, 1517, 19017, 298408. Machines q = 29..53 hold copy 2 = (59, 61) [two-paths B1.8].
- Witnesses, meaning the least revealed copy with lower leg > q [derived C8]:
  - copy 1 for q = 7..23;
  - c_7 for q = 29..53;
  - c_11 for q = 59..139;
  - c_13 for q = 149..173;
  - copy 8 = (239, 241) at q = 179.
  - The record adds that among the binding machines below 200 the witness is a missed copy only at q = 59 and q = 149. It does not define "binding machine".
- For j ≤ 10⁷: |T| = 388,397, and chain(T) holds [derived C2]. The record also gives a "tightest ratio" of 29/7, at 29 → 59, without saying what quantity the ratio is.
- For j ≤ 10⁷, these chains also hold [derived C9]:
  - chain(T − S_1) and chain(T − stretch copies);
  - with SQ taken over m ≥ m0 ≥ 6: chain(T − SQ) and chain(T − (S_1 ∪ stretch ∪ SQ)).
  - The revealed copies with upper leg ≤ 210 are 1, 2, 5 and 6.
- S_1 begins 59 (gear 7), 149 (11), 179 (13), 6269 (79), 7949 (89), 9419 (97). No c_g is revealed for g in 17..73 [kernel 2.4].
- Missed-copy hand-offs:
  - All 327 hand-offs below 200,000 hold [two-paths A1.14].
  - The target holds at every prime q in [7, 39,377,242,978] through the 328 revealed gears below 200,000 (233 in case 1, 95 in case 19).
  - Gear 198437 alone serves every prime q in [31, 39,377,242,978]. Its legs are 39377242979 and 39377242981.
- Below 10⁷ [derived C4]:
  - 6,548 revealed gears and 6,547 hand-offs; all hold and all are strong.
  - Maximum gap 14,160 (8,541,301 → 8,555,461).
  - The record also gives a "maximum ratio" of 79/13, without saying what quantity the ratio is.
- Near X(7) and X(13) [derived C4]:
  - The largest revealed gear ≤ X(7) is 43,849,288,261 = X(7) − 3069 (deterministic Miller–Rabin).
  - No revealed gear lies in (g*, X(13)], where g* = X(13) − 107,292. The rejections are exact; g* itself is BPSW-probable only.

---

## 3. Proved laws, grouped

### 3.1 A single gear on copies

- **Strike class, LEAN** (`copy_strike_class`, `copy_product`): the legs multiply to 900j² − 1. For g ≥ 2 and j ≥ 1, g strikes copy j iff 30j mod g is 1 or g − 1.
- **Squaring law, PROVED** [c.xxvi]: g strikes j iff j² ≡ a_g² (mod g). 0 failures for gears 7..5000, every residue.
  - Survival against any gear set depends on j only through j² mod their product.
  - The mirror j → −j needs no side condition.
- **Leg rule, LEAN** (`copy_pair_iff`): a prime g ≥ 7 strikes some pair of copies j and j + D (j ≥ 1) iff g | D, g | 15D − 1 or g | 15D + 1.
  - Legs of the same sign: g | D.
  - Lower leg of j, upper leg of j + D: g | 15D + 1.
  - Upper leg of j, lower leg of j + D: g | 15D − 1.
  - Consequences [two-paths A3.3]: two consecutive copies share a striker only if it is 7, and copies two apart only if it is 29 or 31. No single gear covers a stretch of length 3 or more.
- **Acting is cofactor order, PROVED** [c.xxvii]: g² ≤ 30j + 1 iff the cofactor h ≥ g. 42,529 strike incidences with 0 failures.
- **Revealed means both legs prime, PROVED** [shelves A2]: for j ≥ 1, no acting gear strikes j iff both legs are prime. The least prime factor p of a composite leg has p² ≤ leg, so p acts.
- **Shadowing, PROVED** [shelves A3]: every strike that does not act is shadowed. The least prime factor of that leg acts on the same copy.
- **First-strike law, PROVED** [c.xxvii; shelves A4, D2]:
  - The least cofactor is k0 = min(u, 30 − u), with u = g⁻¹ mod 30. So k0 is 1, 7, 11 or 13.
  - Over the first 2000 gears (7..17417), k0 = 1, 7, 11, 13 occur 488, 507, 497, 508 times.
  - Every gear strikes below its own square except 7 and 11.
- **Below-square strike count, Stands** [shelves A4]: the number of below-square strikes of g is 2⌊g/30⌋ + c(r), with r = g mod 30 and c(r) = 0, 0, 0, 1, 1, 1, 2, 1 for r = 1, 7, 11, 13, 17, 19, 23, 29. Checked for all gears ≤ 2000.
- **Cofactors, PROVED** [shelves D1–D3]:
  - Strikes of g are in bijection with cofactors h ≡ ±u (mod 30). The upper leg is struck iff h ≡ u.
  - (Z/30)*/{±1} is cyclic of order 4, generated by [7].
  - The struck copies are j = m·g ± j0, with j0 = (g·k0 − ε)/30 and ε = +1 if r < 15, −1 otherwise. The gaps alternate g − 2j0 and 2j0.
- **Separation, PROVED** [shelves D5]: 15·t_g = k·g + 1, with t_g = 15⁻¹ mod g and k = 14, 2, 4, 8, 7, 11, 13, 1 for g ≡ 1, 7, 11, 13, 17, 19, 23, 29 (mod 30). With σ_p = min(t_p, p − t_p): (p − 1)/15 ≤ σ_p ≤ (7p + 1)/15.
- **Fibre value is a square, PROVED** [c.xxvii]: a_g² is a square mod g for every gear, so its Legendre symbol is +1. No quadratic character gates which gears strike.
- **Symmetry closure, PROVED** [c.xxvii]: the affine symmetries of one machine's strike configuration are exactly j → ±j per gear, 2^k of them, all fixing the origin. None preserves acting, since 30j + 1 is injective.
- **Existence, LEAN** (`exists_leg_minus`, `exists_leg_plus`): every prime g ≥ 7 divides the lower leg of some copy 1 ≤ j < g, and the upper leg of some copy 1 ≤ j < g.

### 3.2 Pairs of gears

Gear pairs p < q = p + d, with d even and d < p [shelves F]. In this group q is the larger gear, not a machine.
- **Joint strikes, PROVED** (F1, F2):
  - Both gears strike copy j ⇔ (30j)² ≡ 1 (mod pq).
  - There are four joint classes: 30j ≡ ±1 (same leg: pq divides one leg) and 30j ≡ ±w (split: p and q divide different legs), with d·w ≡ p + q (mod pq). For twins, w = p + 1.
  - Per period pq: 4 copies struck by both, 2(q − 2) by p only, 2(p − 2) by q only, (p − 2)(q − 2) by neither.
- **Least split copy** (F3): both gears act iff c > d, where d·h_p = c·q ± 2. c is constant on each class of p mod 15d (checked for p < 6000).
- **Twins as gear pairs** (F4, F5):
  - Twins occur only for p ≡ 11, 17, 29 (mod 30).
  - First-strike copy: F_g = min{j ≥ 1 : g | 30j ± 1} = min(a_g, g − a_g). The record writes it J_g; F_g here keeps it apart from J_g = ⌈(g² + 1)/30⌉ of 2.2.
  - First-strike separations F_q − F_p: (7 − 2p)/15 for p ≡ 11, (2p + 11)/15 for p ≡ 17, and 0 for p ≡ 29.
  - The closed forms hold with 0 mismatches over 2158 twin, 2135 cousin (including (7, 11)) and 4294 sexy pairs, p in 7..199999.
- **Type-C twins** (F7; p ≡ 29 mod 30): exactly two joint copies below about q² are not acted on by both gears.
  - j* = (p + 1)/30, where neither gear acts.
  - The copy with leg pq, where p acts and q does not.
- **Partner strikes on the other gear's missed copy** (F8): p strikes c_q iff p divides d² + 10 or d² + 12 (when q² ≡ 19), or d² + 28 or d² + 30 (when q² ≡ 1). MEASURED over d ≤ 20, d < p, p ≤ 20000:
  - p strikes c_q exactly for (13,17), (17,19), (23,31), (29,41), (67,83);
  - q strikes c_p exactly for (17,23), (19,23), (31,43), (41,59), (83,103).
- **Twin strikers, stands** [derived A6, E9]: twin gears share leg strikers only from {11, 29, 31} (same case, g ≡ 29 mod 30) or {13, 23, 37} (cross case, g ≡ 11 or 17 mod 30). 7 never occurs. Checked on all twin gears ≤ 10⁶.

Neighbour kills. Gear h kills c_g when h < g and h strikes c_g. The r-th lower neighbour of g is the r-th gear below g.
- **Kill rule, PROVED** [two-paths A4.2; derived E1]: h kills c_{h+d} iff h divides d² + A', where A' is 28 or 30 when g = h + d ≡ ±1 (mod 5), and 10 or 12 when g ≡ ±2 (mod 5). The killer set K(d) lies in [7, (d² + 30)/2].
  - K*(d), the members of K(d) with h + d prime, for d = 2, 4, …, 30: {17}, {13}, {}, {23}, {7}, {11,29}, {}, {7,13,67}, {11}, {}, {31}, {293}, {353}, {199}, {7,13,29,31}.
  - The members of K(d) outside K*(d) for d ≤ 30 all have 3 | h + d.
- **Distance law, PROVED** [two-paths A4.5; derived E3]: (d + 1)² ≥ 2g − 29 in case 1, and (d + 1)² ≥ 2g − 9 in case 19.
  - Equality holds exactly on the families F30 and F10.
  - F30 up to 10⁶: 17→19, 353→379, 65537→65899, 76847→77239, 282767→283519, 679793→680959.
- **Dormancy, PROVED** [two-paths A1.8, A1.9]:
  - Write g = kh + ρ with |ρ| < h/2. Row h is silent on g when ρ² < h − B, and also when ρ² < 2h − B' for odd k (B' = 30 in case 1, 10 in case 19).
  - For h to strike the missed copy of the next gear h' needs (h' − h)² ≥ 2h − B'.
  - MEASURED: among consecutive gears up to 10⁷ (664575 pairs, maximum gap 154) this happens exactly twice: 13 strikes c_17 (leg 299 = 13·23), and 17 strikes c_19 (leg 391 = 17·23).
- **Complete kill lists, MEASURED** [two-paths A4.8]: kills by the r-th lower neighbour, r ≤ 12, g ≤ 10⁸. For example, r = 1: (13,17), (17,19); r = 2: (23,31). The largest killed g is 4517.
- **Not killed by near neighbours, PROVED** [two-paths A4.9]: every range (every q ≥ 7) holds a c_g that is not killed by its r nearest lower gears, for r ≤ 12.
  - q < 59: c_7 = (59, 61). 11 ≤ q ≤ 139: c_11 = (149, 151).
  - 149 ≤ q < 10¹⁵: any gear in (max(4517, √q), 10⁸].
  - q ≥ 10¹⁵: imports Rosser–Schoenfeld, with a pigeonhole on prime counts. This part covers r ≤ 2.6·10⁶.
  - The copy is not claimed revealed.
- **Sharing law, PROVED** [derived E7, E8]:
  - Every shared prime factor ≥ 7 of the legs of two gears D apart lies in a finite set S(D), whose members are < (D² + 58)²/4.
  - A prime above H(w) divides a leg of at most one gear in any set of gears of width w.
  - H(2) = 37, H(4..6) = 449, H(8..12) = 601, H(14) = 619, H(16..24) = 21,841, H(26) = 128,521, H(28) = 176,401, H(30..32) = 228,601, H(154) = 140,873,041.
- **Killed runs, PROVED** (imports Shiu) [derived E11]: gears ≡ 301 (mod 330) are case 1 with 11 | g² + 28. So there are arbitrarily long runs of consecutive gears whose missed copies are all killed.

### 3.3 The missed copy

**Location, LEAN and PROVED** [shelves B1–B3; RangeMissed]:
- c_g is copy J_g = ⌈(g² + 1)/30⌉. Its legs are (g² + 28, g² + 30) in case 1 and (g² + 10, g² + 12) in case 19 (`missed_copy_legs`). Re-checked for all gears < 50000, reading the height off either leg.
- g divides neither leg (`gear_misses_own_copy`).
- No gear larger than g acts on c_g. The next gear has g'² > g² + 30 (`larger_gear_sq`, `no_larger_gear_acts`).
- c_g = j_g in case 19 and c_g = j_g + 1 in case 1. c_g lies in shelf(g) [shelves C5].

**Which smaller gears strike c_g.** A row is a gear h, considered as a possible striker of the missed copies c_g of larger gears g.
- **Strike criterion, PROVED** [shelves B4; two-paths A1.1]:
  - Row h < g strikes c_g iff g² ≡ −A or −B (mod h). Equivalently, g mod h lies in E_h, the unit roots of x² ≡ −A or −B.
  - The four targets are distinct mod every h ≥ 7. No h divides both legs of one copy.
  - Checked for gears 7..30000: 10029 strike pairs with 0 mismatches.
- **Class counts, PROVED** [two-paths A1.2; derived A2]: the lattice count is |E_h| = 2 + (−A/h) + (−B/h), which is 0, 2 or 4 except at (7, case 1), where it is 1 (the class {0}) and e_7 = 0. Everywhere else the unit count e_h equals the lattice count.
- **Reach by reciprocity, PROVED** [shelves 3.A; two-paths A1.2]:
  - (−28/h) = (h/7).
  - (−12/h) = +1 iff h ≡ 1 (mod 3).
  - (−10/h) = +1 iff h mod 40 is in {1, 7, 9, 11, 13, 19, 23, 37}.
  - (−30/h) = +1 iff h mod 120 is in {1, 11, 13, 17, 23, 29, 31, 37, 43, 47, 49, 59, 67, 79, 101, 113}.
- **Inert rows** [shelves B5; two-paths A1.3, A2.1, A2.3]: a row is inert for a case when it can never strike c_g for any prime g > h of that case.
  - Each reach state (inert, A-only, B-only, both) is exactly a quarter of the unit classes.
  - The four characters χ₋₇, χ₋₃₀, χ₋₁₀ and χ₋₃ are independent.
  - Rows inert in both cases are exactly 12 classes mod 840: {83, 227, 311, 467, 479, 551, 563, 587, 671, 719, 803, 839}.
  - Gear 7 is inert in case 1. In case 19 it strikes c_g iff g ≡ ±2, ±3 (mod 7).
- **Free classes, PROVED** [two-paths A1.4]: E^(1) ∩ E^(19) = ∅. The free count h − 1 − e1 − e19 is at least h − 9, and at least 2 at every gear.
- **Revealed counts, MEASURED** [shelves B7]: c_g is revealed for 233 case-1 and 95 case-19 gears in 7..199999. The congruence count and the both-legs-prime count agree.
- **Fixed-H density, PROVED** (imports CRT and Dirichlet) [shelves B8]: for each fixed finite H, gears of a case that avoid the excluded classes for all 7 ≤ h ≤ H have relative density ∏ w_h, with w_h = 1 − |E_h|/(h − 1).
  - Products up to h < 200000: 0.026292 (case 1) and 0.010835 (case 19).
  - Summed predictions 298.94 and 125.08, against raw survivors 233 and 95.
- **Range membership, PROVED** [two-paths A1.13]: c_g lies in machine q's range iff q − A_g < g² ≤ q# − B_g.

**Above the square**
- **Strike law, PROVED** [shelves B9]: g strikes copy J_g + M iff M ≡ x_g or x_g + t_g (mod g), with x_g = −1 in case 1 and x_g = −2·5⁻¹ in case 19. So exactly g − 2 of every g consecutive copies above the square are missed.
- **Silent run p1, PROVED** [shelves 3.A; two-paths A3]: p1 is the length of the run of missed copies headed by c_g.
  - By g mod 30 = 1, 7, 11, 13, 17, 19, 23, 29: (14g−14)/15, (g−2)/5, (4g−14)/15, (g−1)/3, (g−2)/5, (11g−14)/15, (2g−1)/3, (g−14)/15.
  - p1 = 1, 2, 1 at g = 7, 11, 29, and p1 ≥ 3 for every other gear. So the 2nd and 3rd missed copies are J + 1 and J + 2 for every gear except 7, 11 and 29.
- **The n-th missed copy** [shelves B12; two-paths A2.4–A2.11]:
  - Copy J_g + n − 1 has legs g² + 2K_n and g² + 2K_n + 2, with K_n = 15n − 1 (case 1) or 15n − 10 (case 19).
  - Reach states have period h in n. The case-19 states are the case-1 states shifted by 3·5⁻¹.
  - All four states occur for every h ≥ 7. No gear is inert on every copy.
  - The inert set recurs along Pell families: 15s² − 14r² = 1 (case 1), with members n = 1, 3249, 10923025, 36723206689; and 5s² − 6y² = −1 (case 19), with n = 1, 177, 85009, 40973857.
  - Pairs (g, h) with h inert on the whole silent run of c_g are exactly (11,7), (29,7) and (29,19), all in case 1.
  - MEASURED: the longest inert runs for h ≤ 200000 are 8 in case 1 (at 14951 and 53231) and 13 in case 19 (at 38639).
  - If the upper leg of the n-th missed copy is below g'², then "no h < g strikes it" iff "both legs prime" [shelves B12].
  - Counts for n = 1, 2, 3 over gears 7..199999 (congruence / both legs prime): case 1: 233/233, 513/513, 167/167; case 19: 95/95, 157/157, 327/326. The single mismatch is at g = 17, legs (359, 361), 361 = 19².

**Escape classes** [two-paths A1.7, A1.10, A1.11]
- **Escape class**: a level-X escape class is a class of gears g mod 30·∏_{7≤h≤X} h, of one case, such that no row h ≤ X strikes c_g for the gears g > X in the class. Esc(X) is the set of these classes.
- **Counts, PROVED**:
  - At level X there are 4·∏_{7≤h≤X}(h − 1 − e_h) escape classes per case.
  - Passing row h refines each class (h − 1 − e_h)-to-1. No class is lost.
  - Case 1: 24, 144, 1440, 20160 classes at moduli 210, 2310, 30030, 510510. Case 19: 8, 64, 512, 8192.
- **Non-empty, PROVED** (imports Dirichlet): each class is a unit class and holds infinitely many gears. Every row of the level acts on each such gear g > X, and none of them strikes it.
- **Struck in every class, PROVED** (imports Dirichlet) [two-paths A1.11(b); derived B4]: for any modulus M, every unit class mod M holds infinitely many gears whose c_g is struck by an acting row h < g. With the case fixed, the class must be case-compatible.
- **Level-13 measurement**: 1952 escape classes mod 30030 (1440 + 512). In 1821 of them the class's first gear has its c_g struck. 121 classes have one revealed gear before the first struck gear, and 10 have two.
- **Escape characterisation of G, PROVED** [derived C5]: for every integer n ≥ 2, n is a revealed gear iff n lies in a level-(n−1) escape class.

**Escape against the cutoff** [derived B1–B9]
- **Dormancy set** D_h: the unit residues of g mod h whose centred value ρ (|ρ| < h/2) has ρ² < h − B, or has ρ even and ρ² < 2h − B'. D_h ∩ E_h = ∅ (PROVED).
- **Live row**: row h is live on g when 7 ≤ h < g, e_h > 0, and g mod h is not in D_h. **λ(g)** is the largest live row.
- **Cutoff criterion, PROVED** (B2): every striker of c_g is live. So c_g is revealed iff g mod 30·∏_{7≤h≤λ(g)} h lies in a level-λ(g) escape class.
  - The record's closed formula for λ [derived 2.2, B2]: let d0(g) be the least even d ≥ 2 with (d + 1)² ≥ 2g + 1 − B', and let h* be the largest non-inert prime ≤ g − d0. Then λ(g) = h* whenever h* > 2g/3.
  - The exceptions to λ(g) = h* are λ(7) = λ(11) = 0, λ(13) = 7, λ(23) = 13 and λ(37) = 23.
- **λ(g) > 2g/3 for every gear g ≥ 41, PROVED** (B5(d)). The proof pieces:
  - a k = 1 lemma;
  - the full definition of λ for 41 ≤ g ≤ 10⁷;
  - certified chains of non-inert primes on [6.6·10⁶, 1.26·10¹²];
  - imports BMOR 2018 for g ≥ 1.2·10¹⁰.
- **Capped set**: Cap(P) is the set of gears g > P with λ(g) ≤ P. PROVED: Cap(P) ⊂ (P, max(37, 3P/2)], with 37 attained at P = 23 (Cap(23) = {29, 31, 37}). There is at most one capped gear per class mod 30·∏(7..P).
- **Entering law, PROVED** (B7): row λ(g) strikes c_g iff g − λ is an even root of x² ≡ −A' (mod λ).
  - For a kill of c_g by h = g − d, the cofactor is κ = (d² + A')/h [two-paths A4.4].
  - On the κ = 2 kills, 4(g² + B') = (d² + B')((d + 2)² + B').
  - Up to 10⁷ there are 42 such kills by row λ(g): 39 with κ = 2, and 3 with κ = 4 (23→31, 67→83, 199→227).
- **Escape classes with revealed gears, MEASURED** (B9): every escape class at levels 7, 11, 13, 17 and 19, in both cases, holds a gear above the level whose c_g is revealed. The largest witness is 42,561,936,551.

**Doomed classes** [kernel 2.2]
- **Doomed**: an escape class C at level P is doomed when no escaping descendant at any level P' ≥ P meets Cap(P').
- **Children, PROVED** (T1): the children of C at the next row h are one zero child (it can hold only the gear h, which lies in Cap(P)), e_h struck children and h − 1 − e_h escaping children.
  - |Esc| at levels 7..23 is 24/144/1440/20160/362880/6531840 in case 1, and 8/64/512/8192/114688/2293760 in case 19.
- **Equivalences, PROVED** (D1): C is doomed ⟺ C holds no revealed gear ⟺ every prime g ≡ r_C (mod M_P) has a composite leg. Here r_C is the least residue of C and M_P = 30·∏_{7≤h≤P} h.
- **Inheritance, PROVED** (D2): non-doom is inherited upward and doom downward.
- **No fixed prime divisor, PROVED** (D3): x(x² + A)(x² + B) has no fixed prime divisor on any escape class, and both leg polynomials are irreducible.
- **No finite-row certificate, PROVED** (imports Dirichlet) (D4): for every finite row set H, some subclass free of H holds infinitely many gears. So doom has no finite-row certificate, and the rows striking a doomed class are unbounded.
- **R, PROVED**: no class is doomed ⟺ every escape class holds a prime g with g² + A and g² + B both prime ⟺ every class holds infinitely many such g.
- **MEASURED**: every class holds a revealed gear at level 11 (208 classes, below 3·10⁶), level 13 (1,952 classes) and level 17 (28,352 classes). The largest least witnesses are 1,553,537, 21,974,021 and 1,018,572,719.
- **Shape model, PARTIAL** (D6): a shape-only model up to 2·10⁶ (44,520 pairs) strikes every gear in 13..2·10⁶ before its cap.

### 3.4 Shelves and silence

- **Shelves tile, PROVED** [shelves C1–C3]:
  - j_g = ⌈(g² − 1)/30⌉ = (g² − 1 + 12χ(g))/30, with χ(g) = [g² ≡ 19 mod 30]. The shelves tile the copies from j = 2 on.
  - Copy 1 = (29, 31) is shelf(5). Gear 5 acts there but strikes nothing, so copy 1 is revealed.
  - Shelf size: N(g) = (G(2g + G) + 12(χ(g') − χ(g)))/30, with G = g' − g. Among gears whose next gear is ≤ 2·10⁶, the only shelves of size ≤ 2 are those of 7, 11 and 17 (each of size 2).
  - The acting set on every copy of shelf(g) is exactly the primes in [7, g].
- **Own-shelf silence, PROVED** [shelves C4]: gear g is silent when it strikes no copy of its own shelf. This happens iff (g, g + off/2 − 1] contains a prime, with off = 0, 6, 0, 10, 6, 0, 20, 0 for g mod 30 = 1, 7, 11, 13, 17, 19, 23, 29. So:
  - never for g ≡ 1, 7, 11, 19, 29 (mod 30);
  - for g ≡ 17: iff g + 2 is prime;
  - for g ≡ 13: iff g + 4 is prime;
  - for g ≡ 23: iff g + 6 or g + 8 is prime.
  - Checked to gear 300000. Below 30000: 3242 gears, 576 silent.
- **Deferral law, PROVED** [shelves C8]: a silent gear's first acting strike lands on the shelf of the largest prime in (g, g + off/2 − 1].
- **Silent runs, PROVED** [shelves C10]: runs of consecutive silent gears have length at most 2. The runs of length 2 are exactly those starting at g ≡ 13 (mod 30) with g + 4 and g + 6 both prime; there are 282 below 300000.
- **Twins at the gear** [shelves C7]: if g + 2 is prime, the own-shelf strike count of g is 1 for r = 11, 2 for r = 29 and 0 for r = 17.
- **Shelf and machine** [shelves C11, C13]: let g_i be the i-th gear and shelf i = shelf(g_i). Every acting strike on shelf(g_i) comes from a gear ≤ g_i. The least machine that reveals shelf i equals the maximum, over its copies with a composite leg, of the least prime factor of those legs.
- **Region law, PROVED** [kernel 2.1 item 2]: a copy in region g, and also c_{g'}, is revealed ⟺ no prime in 7..g divides either leg. Case-1 head copies are never revealed.
  - Checks: gears 7..3001 (302,422 copies, 0 mismatches). In the tiling of j in [2, 2·10⁶], the only uncovered j are the 484 case-1 heads.
- **Position lemma, PROVED** [derived C10]: a copy at position n ≥ 2 of gear g has legs strictly between g² + 30 and g'².

**Protected stretch** [two-paths A3; derived D]
- **Protected length**: L = j_{g'} − J_g, the copies from c_g up to the next gear's shelf. No gear larger than g acts there.
- **Stretch**: l = min(p1, L), the silent run of g cut at L.
- **Cell**: cell n of gear g is copy J_g + n − 1, the same as position n.
- **Formulas, PROVED**:
  - L > p1 ⟺ d > s, with d = g' − g and s = 14, 3, 4, 5, 3, 11, 10, 1 for g mod 30 = 1, 7, 11, 13, 17, 19, 23, 29.
  - L = p1 only at g = 13.
  - l ≤ 2 exactly at g = 7, 11, 17, 29. l ≥ 3 for every g ≥ 31, and l = 3 first at g = 59 (all gears 7..10⁷).
  - For g ≥ 31, l(g) ≥ (g − 14)/15, with equality only for g ≡ 29 (mod 30).
- **Field and comb, PROVED** [derived D1–D5]:
  - A striker's teeth on a stretch form a comb with gaps alternating m_h and h − m_h, where m_h = min(t_h, h − t_h) ≥ (h − 1)/15.
  - Three consecutive teeth span exactly h. Two teeth are D apart iff h | D(225D² − 1).
  - A striker takes between 2⌊l/h⌋ and 2⌈l/h⌉ teeth.
  - Two strikers cover three consecutive cells only if 7 takes an adjacent pair or 29/31 take the end cells. Otherwise three distinct strikers are needed. No striker takes all three cells.
  - Single-gear covers of a stretch occur exactly at g = 17 by h = 7 (301 = 7·43, 329 = 7·47) and at g = 29 by h = 11 and h = 13 (869 = 11·79, 871 = 13·67) [two-paths A3.3].
- **Covered stretches** [derived D6, D9]: a stretch is covered when every cell is struck by a striker below g. A stretch is covered iff it is comb-covered iff it contains no twin copy.
  - MEASURED: the fully covered stretches among gears 7..99,999,989 are exactly those of 17, 29, 37, 41 and 149 (l = 2, 1, 7, 5, 9). The only consecutive covered pair is {37, 41}.
  - Protected regions with no revealed copy up to g = 10⁶: 17, 29, 41 [two-paths A3.4].
- **No fixed-depth certificate, PROVED** (imports CRT and Shiu) [derived D7, D8]:
  - For every N and K, "every K consecutive gears include one with a revealed cell among its first N cells" is false.
  - No fixed finite striker set forces a revealed cell at fixed depth.

### 3.5 Survivor classes

- **Survivor set, PROVED** [shelves E1–E3]:
  - S_q = {j : gcd(900j² − 1, q#/30) = 1}, and |S_q| = ∏(g − 2). For q = 7, 11, 13, 17 this is 5, 45, 495, 7425.
  - Its affine stabiliser is exactly {j ↦ uj : u² ≡ 1 mod q#/30}, which is (Z/2)^r. There are no translations.
  - Adding the next gear p maps S_{q'} onto S_q exactly (p − 2)-to-1.
  - For j ≥ 1, j ∈ S_q iff q < P(j), the least prime factor of (30j − 1)(30j + 1).
- **Gears above q refine and never delete a class, PROVED** [shelves E5, E6]: each gear g > q strikes exactly two of the g subclasses of every class mod (q#/30)·g, one per leg, and removes (acts on) nothing below ⌈(g² − 1)/30⌉.
- **Machine-gears-only theorem, PROVED** [c.xxvii; shelves E8]: some copy of the range is struck by no gear ≤ q. For q = 7, 11, 13, 17 the survivors meet the range in 4, 44, 494, 7424 copies.
- **Check** [shelves E7]: all 495 classes of machine 13 hold a revealed copy with j < 10⁶. The latest first revealed copy is copy 35583, in class 548.
- **Supply law, PROVED** [c.xxvi]: the lower set is the primes that lay out the copies (2, 3 and 5 in the original machine). Dropping a prime p from the lower set into the gears multiplies the copies in one period by p and the surviving copies (struck by no gear) by exactly p − 2, since p rejoins as a gear and takes two classes. c.xxvi calls these the raw and the live copies.
- **Unit twist, PROVED** [c.xxvi; two-paths B2.3]: a rung y is a lower set of the primes up to y. The class geometry at rung y' is the geometry at rung y multiplied by a single unit mod g. The two-paths record writes this as conjugation by Q = y'#/y#, a change of index under CRT.
- **Offsets are translates, PROVED** [c.xxvi]: for any gear set with modulus M, the blocked sets, and the open sets, at the open offsets 0, 2 and 3 are exact translates of one another by −e·5⁻¹ mod M. Set equality was checked at M = 77, 1001, 17017, 323323.
  - Any map moving the known opening off offset 0 costs at least (M − 3)/5 laps.
- **Translation with acting built in** [two-paths B1]: M(j) is the product of the gears acting on copy j, and c(j, e) = j + e·5⁻¹ mod M(j). For g acting on j and e ∈ {2, 3}: g strikes column 5j + e iff g strikes copy c(j, e) (checked for laps < 20000).

Offset columns. An exposure (j, e) is a column n = 5j + e at offset 2 or 3 with both 6n − 1 and 6n + 1 prime.
- **Exact downward transfer, PROVED** [two-paths B1.7]: take an exposed column n and a copy j' with 5j' < n and (5j')² ≡ n² mod M(j'). Then j' is revealed and M(j') ≤ n² − 25j'². For n ≤ 5·10⁶ the targets reach at most copy 27.
- **Source legs** [two-paths B1, B1.4(v)]: let r = r(j, e) be the least copy of the class c(j, e) mod M(j). Both source legs act at r for every exposure with j ≥ 12. MEASURED: for all 187,237 exposures with 4 ≤ j < 2·10⁶, neither source leg strikes r.
- **Transfer cost floor, PROVED** [two-paths B2.5]: the minimum cost over nonzero open offsets is 6k*·M. Here k* is the least k with 6kM mod y# a nonzero open offset of the rung y.
- **Forced pairs, PROVED** [two-paths B3.7, B3.8; derived F3–F7]:
  - A pair of copies (j, j') is forced when j'² ≡ j² modulo the product of the gears 7..max(q, sh(j')). Here sh(j') is the top gear acting on j'.
  - On a forced pair, j revealed ⇒ j' revealed. Checked at q = 7..23 (3, 16, 65, 166, 539, 1709 forced pairs, 0 counterexamples).
  - A pair is forced iff every gear in (q, sh(j')] divides t, the record's pair parameter (t = −ab in its parametrisation).
  - Forced pairs never go upward.
  - A source with j² < M_q has no forced target.
- **Orbit sizes, PROVED** [derived F7]: F = {x : 25x² ≡ n² mod M_X}, with n = 5j + e, X = sh(j) and M_X the product of the gears 7..X, has |F| = 2^(k − w). Here k is the number of primes in [7, X] and w = ω(gcd(n, M_X)). Over machine q, F splits into ∏_{X<g≤q}(g + 1)/2 orbits. The machine orbit is one of these orbits, and is all of F iff X = q.
- **Orbit statement V1, MEASURED** [derived F7, F8, F10]: for every window exposure (j, e), some copy of F (mod M_X, X = sh(j)) in machine q's range is revealed. This is exact for q in [7, 61]. For q = 67..113 the witnesses are BPSW-probable. The record states V1 as equivalent to the fibre chain from max(7, sh(j)).
- **V2, REFUTED** (section 5) [derived 3.F]: for every window exposure, some copy of the machine orbit mod M_q in machine q's range is revealed. V2 uses the machine orbit, V1 all of F; they coincide when X = q.

### 3.6 The derived machine and the derivation operator

**Derived machine** [derived A; table 2.1]
- **Definition**: the columns are the integers x with x² ≡ 1 or 19 (mod 30). Column x carries copy J(x) = (x² + c)/30, with c = 29 in case 1 and 11 in case 19. Its legs are x² + A and x² + B.
- For prime x, J(x) = J_x, so the column x = g carries the missed copy c_g.
- A **striker** is a prime h ≥ 7 read on the columns. It strikes column x iff h divides (x² + A)(x² + B), i.e. iff x mod h lies in E_h.
- **Laws, PROVED**:
  - J is strictly increasing, and the images of the two cases are disjoint (A1).
  - |E_h| = 2 + (−A/h) + (−B/h), in {0, 2, 4} except at h = 7 in case 1, where the lattice count is 1 (A2).
  - Acting: h acts on column x iff h ≤ x, and h = x never strikes. This is exactly h² ≤ 30J(x) + 1 (A7). Composite columns can be revealed: 221, 451, 781, … (case 1) and 427, 517, 1477, … (case 19).
  - Pair rule on units (A4). N(D) is the ordered-pair count on units: the ordered pairs of struck classes (x, x + D), both in E_h, which the proof splits into identity, fold and cross pairs. With e_h the unit count, N(D) = e_h[h | D] + [h | D² + 4A] + [h | D² + 4B] + 2[h | D⁴ + 2(A + B)D² + 4]. It is exact except at (7, case 1) with D = 0.
  - Separation: for classes u, v struck by h, (v − u)(v + u) ≡ τ_v − τ_u (mod h), with τ = x² ≡ ε − c, where ε = +1 when h strikes the lower leg and −1 when it strikes the upper leg [A3; kernel 2.3 item 13]. The difference lies in {0, ±2} for a same-case pair and {±16, ±18, ±20} for a cross-case pair (A3).
  - Symmetry: for every non-inert striker, on units, the stabiliser of E_h is {x → ±x}, except at striker 29 in case 1, where {±1, ±12} swaps the legs. On the lattice, (7, case 1) is a further exception. No stabiliser contains a translation (A8).
  - ±1 is a struck class exactly at {29, 31} (case 1) and {11, 13} (case 19) (A9).
  - Joint classes for strikers h ≠ h' on one case: (n_A(h) + n_B(h))(n_A(h') + n_B(h')) jointly struck classes mod hh'. Here n_T(h) = 1 + χ_{−T}(h) = 1 + (−T/h) for T = A, B (the class count of leg T), except n_A(7) = 0 on units in case 1 (A10).
  - One acting striker can strike the missed copies of three consecutive same-case gears. First instances: 11 on 59, 61, 71 (case 1) and 7 on 17, 23, 37 (case 19) (A13).
- **Stands** [derived A11, A14, N6]:
  - The derived period is q#. The derived range is x ∈ (√(q − A), √(q# − B)], where each nonzero survivor class occurs once (A11).
  - Window thresholds: a striker h > W²/4 + B strikes at most 2 same-case columns in a span W, and h > W⁴/4 + (A + B)W²/2 + 1 strikes at most 1 (A14). The record lists it among items that stand as stated, with a refuter's rebuild to 3·10⁵, an assembly check (W = 2..8, h ≤ 4000) and an argument sketch.

**The derivation operator** [kernel 2.3]
- **Definition**: for a machine M, Der_int(M)(h) is the least copy c of M with 30c − 1 > h². O is the original machine and D the derived machine.
- **Idempotence, PROVED**:
  - Der_int(O) = D and Der_int(D) = D.
  - Der_int ∘ Der_int = Der_int for every machine.
- **Fixed points, PROVED**:
  - For a family C of copies, f_C(h) = min{c ∈ C : 30c − 1 > h²}. Der_int(M) = f_C with C the copies of M [kernel 2.3 item 3].
  - The fixed points are exactly the maps f_C, for unbounded copy families C.
  - D is the pointwise least fixed point. It is the unique fixed point whose lower leg at every column h lies in (h², h² + 30).
  - D is not the only fixed point. Der_int(P²) is another: column 7 carries copy 6 there, against D's copy 2.
- **Tower, PROVED**: P^k = (J^k)*O, in which column y carries copy J^k(y).
  - P^1 is 8 classes mod 30.
  - For k ≥ 2, P^k is 2^(2k+1) classes mod 2·15^k, all case 1, and J maps them 4-to-1 onto P^(k−1).
  - P² is 32 classes mod 450 and P³ is 128 classes mod 6750. P⁴ (512 classes mod 101250) comes from the refuter's recomputation.
- **Fixed columns, PROVED**: 30(J(y) − y) = (y − 1)(y − 29) in case 1, so the fixed columns are 1 and 29. J(y) < y exactly at y = 7, 11, 13, 17, 19, 23.
- **Class counts, PROVED**: the possible counts run {2} → {0, 2, 4} → {0, 2, 4, 6, 8} → {0, 2, …, 16}. They are always even and at most 2^(k+1).
- **Acting stays diagonal, PROVED**: h acts on column y of P^k iff h ≤ J^(k−1)(y). A prime h = J^(k−1)(y) never strikes that column. The only exception is gear 5 at column 1 of every level.
- **Persistence, PROVED**: k → E^(k)_h is eventually periodic, and h strikes at every level iff a seed lies on a J_29-cycle mod h. The seed-period-1 strikers are {11, 13, 29, 31, 67, 79} in branch 1 and {11, 13, 23, 37, 853} in branch 19.
- **Reach, PROVED**: X(1) = 80434 and X(7) = 43849291330. The P² reach ends at y' = 1549 for 29# and at y' = 1146941 for 59#.
- **Chain down the tower, PROVED**: chain is upward-closed (read as an unbounded sequence in which every member has a successor), and S_k ⊆ S_(k−1) for the record's level-k sets S_k.
- **Range as a power, PROVED**: the period of P^k is 15^(k−1)·q#. Every range column y of P^k satisfies y^(2^k) < 30^(2^k − 2)·q#.

### 3.7 The locator closure

- **LEAN `class_meets_every_residue`, `class_residues_distinct`**: if prime g does not divide N, the numbers r + kN, k < g, run over every residue mod g once.
- **LEAN `locator_closure`**: for a prime g ≥ 7 with g ∤ N, some copy j = r + kN (k < g, j ≥ 1) is struck by g.
- **LEAN `locator_closure_contra`**: if no copy j ≥ 1 of the class j ≡ r (mod N) is struck by g, then g | N.
- **LEAN `locator_modulus_prod`, `locator_modulus_le`**: if no copy of the class is struck by any prime in [7, X], then the product of those primes divides N, and is ≤ N when N ≥ 1.
- **Consequence, PROVED** [c.xxvi]: a class silent against [7, X] has N divisible by X#/30. So N exceeds X'²/30, and the class holds at most one member in the certified band.
  - Every congruence locator certifies at most one copy, and only inside the window. The node applies this to the known-opening copies of R5.f.xxxv.b and to the centre construction.
  - 0 escapes in 920 tested triples.
- **Inequality** [c.xxvi]: X# > nextprime(X)² for every prime X ≥ 7, verified to X = 4000.
- Related results: no fixed finite striker set forces a revealed cell at fixed depth (3.4), and doom has no finite-row certificate (3.3).

### 3.8 The barrier

- **Total blame assignment**: a map sending each copy of the range to a gear that strikes it and acts on it. If no copy of the range is revealed, such a map exists [c.xxvii].
- **Barrier, PROVED** [c.xxvii]:
  - Let U be the untruncated machine: the same gears and classes, with acting dropped.
  - Every rule in the round's proved list except acting is a statement about the strike relation alone. Each holds verbatim in U.
  - U admits a total blame assignment: b(j) = the least prime factor of 30j − 1, which is always a gear and always divides a leg (checked to j = 50000).
  - So no combination of strike-relation rules can forbid a total blame assignment.
- **Two-clause condition** [c.xxvii]: any property P that forbids a total blame assignment must:
  - (a) fail in U, and hence mention acting essentially; and
  - (b) not be a consequence of T, "a composite has a prime factor at or below its square root".
- **What T implies, PROVED** [shelves D6]:
  - T implies: acting ⇔ h ≥ g; revealed ⇔ both legs prime; below-square strikes are redundant.
  - T does not imply the spoke set {1, 7, 11, 13}, the first-strike exception set {7, 11}, the mirror, the location of c_g, or the constants of the silence criterion.
  - Each is refuted by a variant machine in which T holds: base 210 (M210), base 2310 (M2310) and a skewed base (SKEW). M210 has 24 values of k0 (max 103) and 13 first-strike exceptions (11 to 79). M2310 has 240 values of k0 (max 1153). SKEW has no mirror-symmetric gear ≥ 7.
- **Missed-copy form, PROVED** [two-paths A1.12(i)]: with all primes ≥ 7 as rows and no cutoff, every c_g is covered. The blame is b(g) = lpf(g² + A_g), which is never g and never below 7.
- **Doom form, PROVED** [kernel 2.2 D5]: with the acting bound removed, every class is doomed and the tree is unchanged. So the tree, as sets, does not determine doom.

---

## 4. Kernel table

Seven modules are in namespace `RangeLine`, in `C:/dev/primes/proofs/`: the four of the first kernel round, and RangeMissedReveal, RangeDerived and RangeChain from the second (rows at the end of the table; record `range_kernel2_2026-09-25.md`, Part 1). Status [kernel Part 1; kernel2 Part 1]:
- `lake build` succeeds for each module, and `lake env lean` exits 0 with no output.
- There is no sorry, admit, axiom, native_decide, unsafe, opaque, implemented_by, set_option, extern, macro or elab.
- The axioms used are only propext, Classical.choice and Quot.sound.
- Each module has a `[[lean_lib]]` block near the end of `proofs/lakefile.toml` (after them come RangeMissedReveal, RangeDerived and RangeChain). None is in defaultTargets.
- The reviewer found no false statement, no vacuous hypothesis and no off-by-one.
- "Strike" in the kernel is plain divisibility, so a leg equal to g counts: g = 29 strikes copy 1. Acting and revealed are not defined in these four files.

| Module | Theorem | What it says |
|---|---|---|
| RangeCopies | `copy_product` | (30j − 1)(30j + 1) = 900j² − 1, for every j |
| RangeCopies | `copy_strike_iff` | g divides the product of the legs iff g divides 900j² − 1 (any g, any j) |
| RangeCopies | `copy_strike_iff_legs` | for prime g, dividing 900j² − 1 is the same as dividing a leg |
| RangeCopies | `minus_leg_iff_mod`, `plus_leg_iff_mod` | g ≥ 2, j ≥ 1: g divides 30j − 1 iff 30j mod g = 1; g divides 30j + 1 iff 30j mod g = g − 1 |
| RangeCopies | `copy_strike_class` | g ≥ 2, j ≥ 1: g strikes copy j iff 30j ≡ ±1 (mod g) |
| RangeCopies | `leg_minus_minus`, `leg_plus_plus` | prime g ≥ 7 dividing one leg of copy j divides the same-sign leg of copy j + D iff g divides D |
| RangeCopies | `leg_minus_plus` | g divides 30j − 1 ⇒ (g divides 30(j + D) + 1 iff g divides 15D + 1) |
| RangeCopies | `leg_plus_minus` | g divides 30j + 1, D ≥ 1 ⇒ (g divides 30(j + D) − 1 iff g divides 15D − 1) |
| RangeCopies | `copy_leg_rule` | prime g ≥ 7 striking copies j and j + D ⇒ g divides D, 15D − 1 or 15D + 1 |
| RangeCopies | `exists_leg_minus`, `exists_leg_plus` | every prime g ≥ 7 divides the lower leg of some copy 1 ≤ j < g, and the upper leg of some copy 1 ≤ j < g |
| RangeCopies | `copy_leg_rule_converse` | each of the three conditions gives a copy 1 ≤ j < g with g striking copies j and j + D |
| RangeCopies | `copy_pair_iff` | the leg rule and its converse as one equivalence |
| RangeCopies | helpers | `not_dvd_two`, `not_dvd_thirty`, `int_two_mul_iff`, `int_thirty_mul_iff`, `int_step_iff` |
| RangeLocator | `class_meets_every_residue` | prime g not dividing N: r + kN, k < g, hits every residue mod g |
| RangeLocator | `class_residues_distinct` | the same g numbers are pairwise distinct mod g |
| RangeLocator | `thirty_ne_zero` | 30 is nonzero mod a prime g ≥ 7 |
| RangeLocator | `locator_closure_minus` | prime g ≥ 7 not dividing N: some k < g gives j = r + kN ≥ 1 with g dividing 30j − 1 |
| RangeLocator | `locator_closure` | the same, with g dividing (30j − 1)(30j + 1) |
| RangeLocator | `locator_closure_contra` | a class of copies mod N that g never strikes forces g to divide N |
| RangeLocator | `locator_modulus` | a class that no prime in [7, X] strikes: every such prime divides N |
| RangeLocator | `locator_modulus_prod` | the product of the primes in [7, X] divides N |
| RangeLocator | `locator_modulus_le` | that product is ≤ N when N ≥ 1 |
| RangeMissed | `prime_ge7_mod` | a prime g ≥ 7 is odd and divisible by neither 3 nor 5 |
| RangeMissed | `sq_mod30_cases`, `sq_mod30_exactly_one` | a prime g ≥ 7 has g² mod 30 equal to 1 or 19, and exactly one holds |
| RangeMissed | `missed_copy_legs_case1`, `missed_copy_legs_case19`, `missed_copy_legs` | case 1: J = (g² + 29)/30 has legs g² + 28, g² + 30; case 19: J = (g² + 11)/30 has legs g² + 10, g² + 12 (needs only g² mod 30) |
| RangeMissed | `dvd_of_dvd_sq_add` | g dividing g² + c implies g divides c |
| RangeMissed | `gear_misses_own_copy` | a prime g ≥ 7 divides neither leg of its own missed copy |
| RangeMissed | `larger_gear_sq` | primes 7 ≤ g < h: g² + 30 < h² |
| RangeMissed | `no_larger_gear_acts` | for such h, if h divides a leg L ≤ g² + 30 with h < L, then L/h < h, so h is not the smaller factor |
| RangeHandoff | `RangeStatement` (def) | ∃ p, q < p ∧ p + 2 ≤ primorial q ∧ p prime ∧ p + 2 prime |
| RangeHandoff | `range_implies_unbounded` | RangeStatement at every prime q ≥ 7 ⇒ twin pairs above every N |
| RangeHandoff | `twin_serves` | a twin pair with q < p and p + 2 ≤ primorial q witnesses RangeStatement q |
| RangeHandoff | `serves_interval_of_le`, `serves_interval` | one twin pair with p + 2 ≤ primorial q₁ and q₂ < p serves every q in [q₁, q₂] |
| RangeHandoff | `primorial_seven` | primorial 7 = 210, proved by kernel `decide` |
| RangeHandoff | `small_cases`, `small_case_7/11/13/17` | RangeStatement q for every q in [7, 23], witness p = 29 |
| RangeMissedReveal | `legs_coprime_30` | the legs g² + A, g² + B of a missed copy are coprime to 30 (needs only the offsets) |
| RangeMissedReveal | `composite_leg_small_factor` | prime g ≥ 7: a leg of c_g that is not prime has a prime factor h with 7 ≤ h < g |
| RangeMissedReveal | `missed_copy_revealed_iff` (and `_case1`, `_case19`) | both legs of c_g are prime iff no prime h with 7 ≤ h < g divides either leg |
| RangeDerived | `J1_exact`, `J19_exact`, `copy_map_on_gear` | 30·J1(y) = y² + 29 when y² ≡ 1 (mod 30); 30·J19(y) = y² + 11 when y² ≡ 19 |
| RangeDerived | `J1_fixed_identity`, `J1_fixed_iff` | 30(J1(y) − y) = (y − 1)(y − 29); in case 1, J1(y) = y iff y = 1 or y = 29 |
| RangeDerived | `sq_add_eleven_ne`, `J19_no_fixed` | y² + 11 ≠ 30y for every natural y; J19 has no fixed column |
| RangeDerived | `J1_strictMono(On)`, `J19_strictMono(On)` | J is strictly increasing on its residue set |
| RangeDerived | `J1_lt_self_iff`, `J1_lt_self_list`, `J19_lt_self_iff`, `J19_lt_self_list` | J moves a column below itself exactly at y ∈ {11, 19} (case 1) and y ∈ {7, 13, 17, 23} (case 19) |
| RangeChain | `primorial_le_of_le` | primorial is monotone |
| RangeChain | `chain_implies_range_all`, `chain_implies_range` | a strictly increasing sequence of twin lower legs p_k with p_0 = 29 and p_{k+1} + 2 ≤ primorial(p_k) gives RangeStatement q at every q ≥ 7 |
| RangeChain | `chain_implies_unbounded` | under the same chain hypotheses, twin pairs exist above every N |

Reviewer notes [kernel 1.3, 1.4]:
- No single theorem assembles the missed-copy law, and none was requested.
- `serves_interval` carries three unused hypotheses to match the requested wording; `serves_interval_of_le` is the stronger form.
- In `range_implies_unbounded`, the hypothesis is the open range statement, in the `RangeStatement` form of 1.2.

---

## 5. Refuted statements, with their instances

### Nodes c.xxv to c.xxvii (2026-09-23)

| Statement | Instance |
|---|---|
| Path B of c.xxvi: at a fixed gear set, the three open offsets of machine 5 carry different strike patterns to exploit | The three open offsets carry one pattern in three positions: exact translates by −e·5⁻¹ mod M (set equality at M = 77, 1001, 17017, 323323). |
| "Gear g strikes no copy j with 30j + 1 < g²" (the brief listed it as proved) | Gear 13 strikes copy 3 (91 = 7·13 < 169). 1043 counterexamples among gears < 500 with j < 3000. |

Withdrawn [c.xxv, c.xxvi]: the tier framing of c.xxv, which split the range into tiers, each a window statement for a larger machine. The owner withdrew it on 2026-09-23. c.xxvi proves the split is an exact covering identity.

### Node c.xxviii [shelves section 2]

| Statement | Instance |
|---|---|
| Case-1 inert rule written purely with characters | False at h = 7: 7 divides 28, so gear 7 is inert in case 1. |
| n = 3: 0 mismatches between congruence survival and both legs prime | g = 17, legs (359, 361), 361 = 19². |
| Protection bound M < (g'² − g² − 30)/30 in both cases | Fails in case 19. g = 13: copy 9, legs (269, 271), is protected (17² = 289), but the bound excludes M = 3. Also g = 43 and g = 47. |
| Silence propagates to the next gear | g = 17 is silent; gear 19 strikes leg 361 at copy 12, on its own shelf. |
| "Gears above p_i never strike shelf copies" | Gear 13 strikes copy 3, which lies in shelf(7) = [2, 4). The strike does not act. |
| Every shelf contains a revealed copy | Shelf 17 = {10, 11} and shelf 29 = {28, 29, 30, 31} have none. |
| Mirror argument using only "shelf ⊂ (0, P)" | Shelf 7 has top/P = 4/7. It needs the P/2 bound. |
| Silence criterion printed as (off − 2δ)g ≤ δ² | A sign slip; it is the exact negation of the true criterion. |
| Separation with t_g itself giving the four slopes | With t_g, k = 14, 2, 4, 8, 7, 11, 13, 1. The four slopes need min(t_g, g − t_g). |
| Cofactor window [g²/f, g'/f)² | The correct form is [g²/f, g'²/f). |
| d1/d2 value set without 2 and 8 | d2 = 8 for class 11 and d2 = 2 for class 29. |
| The first-strike exception set follows from T | M210 has 13 exceptions: 11, 13, …, 79. |
| Spoke set, k0 ≤ 13, mirror, c_g location and silence constants follow from T | M210, M2310, SKEW (section 3.8). |
| SKEW: "no gear ≥ 11 is mirror-symmetric" | Too weak. No gear ≥ 7 is, since symmetry needs g \| 2 or g \| 10. |
| M210 "seven offsets" | Six offsets, one per value of g² mod 210, plus a deviation at gear 11. |
| "h_lo ≠ g ⇒ c_g missed" | The clause alone does not prove it. |
| Type-C twins: non-acting joint copies = 2 | Correct only as "not both acting". With "neither acts" the count is 1. |
| isqrt(H) = g + off/2 − 1 for every silent gear | Fails at g = 23 (off = 20). The deferral conclusion is unaffected. |
| Cousin pairs, p in 7..199999: 2134 | The count is 2135, including (7, 11). |
| Least-c constancy on p mod 30 (sexy split copies) | Fails. Constancy is on p mod 15d. |

### Node c.xxix [two-paths section 3]

| Statement | Instance |
|---|---|
| Each survivor class has exactly one copy in the range | Class 0 has no copy in the range (q = 7..19). |
| R11(c) as originally stated | 7 is the only gear ≡ 7 mod 35 (and mod 210) up to 10⁶. For M = 5, 35, 55, every unit class incompatible with case 19 holds 0 case-19 gears (e.g. 1 mod 5). |
| R12: the ceiling system is a total cover | Not at any q = 7..31 (uncovered 3, 3, 6, 10, 19, 62, 168, 585). At q = 7 the uncovered gears are 7, 11, 13. |
| Dormancy with non-strict "within" | Both boundaries are attained: (31, 61) with ρ² = h − B, and (17, 19) with ρ² = 2h − B'. |
| The general clause, with reason "2K_n < 30g" | Fails at every copy g strikes: 330 failures for g < 1000. g = 11, n = 3: 209 = 11·19. g = 7, n = 2: 91 = 7·13. |
| "Divisor gears are always inert" | Case 1, n = 3, h = 11: 11 \| K_3 = 44 closes leg A, but −90 ≡ 9 mod 11 is a square. 45 such gears for n ≤ 30, h < 200. |
| "A recurrence of the inert set gains exactly the divisor gears" | n = 3249: 59 \| r but 59 stays B-only. n = 10923025: 11 goes to B-only. n = 177: 7 goes to A-only. |
| Stretch (1) "exactly", strike-and-eligible reading | g = 13, copy 10: legs 299 = 13·23 and 301 = 7·43. Gear 17 strikes neither, and 19² > 301. |
| Stretch (iv), n = 0 character times the un-normalised symbol | Off by (14/h) or (2/h). First failures at h = 17 and h = 11. |
| Every head stretch holds a revealed copy | g = 37, copies 46..52: 1379 = 7·197, 1411 = 17·83, 1441 = 11·131, 1469 = 13·113, 1501 = 19·79, 1529 = 11·139, 1561 = 7·223. Also g = 17, 29, 41, 149. |
| R6 header: sharp on all six families | 13→17 has d² = 16 while 2h − 30 = −4. 11→29 has κ = 32. |
| R5: "minimum attained only at 17→19" | Attained at all six F30 kills. |
| The R8(c) argument at q = 10¹⁵ | Reaches only r ≤ 888,539 (913,612 with the full form). |
| K(d) without the mod-5 case filter | Would wrongly add 7 to K(4), {11, 23} to K(6) and 113 to K(14). |
| S1: the least translate of an exposure is revealed | (6,2) → 607: 18209 = 131·139. (10,2) → 13624: 408721 has spf 113. Failure counts at q = 13..47: 3, 5, 7, 11, 13, 18, 24, 27, 36, 44. |
| S2: the class has a revealed copy in the range | q = 13, (6,2): the only class copy, 607, is struck. q = 41, (47,3): M = 247357937827, and all 41 class copies are struck. |
| S3: the full orbit has a revealed copy | q = 17, lap 14, offset 2: the only orbit copy is 7622. q = 29, lap 35, offset 2: 12 orbit copies, all struck. Both sources lie above the window. |
| S4: every exposure has an exact transfer into the range | False for q ≥ 29. (2,2), n = 12: 144 − 100 = 44 ≡ 2 mod 7. 87 sources in laps 2..399 have none. |
| "Exposure never passes upward" | Upward pairs (2,5), (3,5), (3,10), (3,25), (32,45). |
| (6) literal count N/M at laps 0 and 1 | One short: q = 7, lap 1 gives 7 against 6. |
| (3) side clause at lap 1 | The least copy, 0, lies below the source. |
| Rungs (4) read as a necessary condition | 42 → 60: A(42) = {}, A(60) = {7}. 16 instances. |
| Rungs (5b) half-lap bound in floor coordinates | Fails at 22 of 54 pairs, e.g. 378 → −1939560 (64664 laps against 64664.1). |
| Rungs (6a) as a statement about realised shelves | At y = 13, shelves 29 and 31 give no translate shorter than the range for q = 19 or 23. |
| Rungs (6b) y = 7 mean floor 0.2458 | The true minimum is 0.2286. |
| Rungs (7b) counting lower-set squares | Lap 0 at y = 5: 1 vs 2 (gears ≥ 5) and 2 vs 4 (all primes). |
| Rungs (11): a fit at y ≥ 7 | There is none. |
| Rungs (12): offset 18 shares the acting set | Lap 1: (47, 49) faces 7. This happens in 243 laps. |
| Stabiliser: "only u = 1 keeps height" | q = 11: u = 34 keeps 14 and 35; u = 43 keeps 22 and 44. |
| Stabiliser (9): "ties never carry exposure upward" | 3→2 (91 = 7·13), 3→5 (616 = 8·77), 32→9 (1001). |
| Stabiliser (11), pairwise: exposure transfers only under tie-containment | q = 11: 41→8 (u = 34), 27→6 (u = 43), 43→34 (u = −1), each with gear 13 eligible but untied and all four legs prime. |
| Stabiliser (12): a uniform status for non-forced lower images | 44 → 33: 989 = 23·43, with 23 untied. q = 23: 4576236 → 28, target (839, 841 = 29²). |

### Node c.xxx [derived section 3]

| Statement | Instance |
|---|---|
| 30ΔJ = v² − u² across cases | h = 37, u = 3 (case 1), v = 5 (case 19): 30ΔJ ≡ 35 but v² − u² = 16. It fails on all 3200 cross-case ordered pairs. |
| Gear-level sentence of (6): set {7} for case 19 | (17, 103), D = 86: 13 \| 299 and 13 \| 10621. |
| Original clauses of (15) | The product clause fails on the lattice at (7,1) pairs, e.g. (7,13) has 2 joint classes. The split clause gives 2 where the count is 0. |
| N(D) with e_h read as the lattice count | (7,1), D = 0: N = 1, formula 2. |
| (17) | Refuted by the incidences of A13 (text not on disk). |
| (10) conclusion: every revealed missed copy in machine q's range has λ(g) > q | q = 13: c_7 = (59, 61) has λ = 0, c_11 = (149, 151) has λ = 0, c_13 = (179, 181) has λ = 7. |
| (10) core: reduction to the capped strip | Cap(13) = {17, 23}, neither revealed (299 = 13·23, 301 = 7·43, 539 = 7²·11). It fails at 50,139 of the 148,930 prime levels in [7, 2·10⁶]. |
| Original (5)(f) class identity | At q = 7, case 19: Pull has 12 classes, Esc has 8. |
| H5 literal at n = 1 | 1 is a case-1 residue and c_1 = copy 1 is revealed, but 1 is not a gear. |
| Heredity of chain | {c_7} ∪ {c_g : g ≥ 43,849,311,221} leaves machine 59 unserved. |
| H7 with family bound m ≥ 16 | chain(SQ) fails at q = 7: the smallest upper leg is 271 > 210. |
| H8(i): "first time at q = 179" | Fails at q = 7, where the witness is copy 1. |
| H8(ii) with SQ over all m ≥ 0, or all units m ≥ 1 | q = 7 is left unserved. |
| "At most two teeth per stretch unless 7, 29, 31" | g = 43: striker 11 takes cells 1, 9, 12 (legs 1859, 2101, 2189). |
| "Every K consecutive gears include one with a revealed cell among the first N" | False for every N and K (CRT + Shiu). |
| Literal (6): K(d) ∩ K(d + D) ⊆ S(D) | (d, D) = (14, 2): 7 ∈ K(14) and 7 ∈ K(16), but 7 ∉ S(2). |
| Literal summary piece 5 | 11 and 151 with D = 140, p = 7. Without p ≥ 7: g = 3, D = 6 (39 = 3·13, 93 = 3·31). |
| Proof step: "a composite leg is < (g + 1)²" | False at g = 11 (151 > 144). |
| Row-by-row escape-class reading of the (8) residue sentence | Fails on the revealed gears ≤ H(w), e.g. {7, 11} at w = 4. |
| K(8) versus table A4.2 | K(8) = {19, 23}; the table has {23}. The table is K*. |
| V2: every window exposure has a revealed copy of the machine orbit mod M_q in the range (node wording: "every window exposure has a revealed orbit copy") | q = 19, n = 33 (exposure (6,3), legs 197, 199). The machine orbit of 33/5 mod 323,323 = M_19 has 16 copies, all in the range, none revealed. For example, 108130 has legs 3243899 = 199·16301 and 3243901 = 37·73·1201. |
| (7)(ii) literal, k counted over gears ≥ 5 | Off by a factor of 2. (2,2): legs 71, 73, X = 7, F = {1, 6}; the literal formula gives 4. |

Also recorded in that record, with statement texts as numbered there: B-(4) (the 360 struck pairs of B4), (5)(b) at g = 7, the (7) wording, and the (8) closing remark (struck as unproved).

### Node c.xxxi [kernel]

| Statement | Instance |
|---|---|
| The reveal status of a copy at position ≥ 2 determines c_g or c_{g'} | All four combinations occur (j ≤ 10⁷): copy 14 (no, no), copy 184 (no, yes), copy 8 (yes, no), copy 267 (yes, yes). |
| A service-preserving map from revealed copies to revealed missed copies exists | Copy 8 = (239, 241) serves [11, 239). No revealed missed copy lies in [239, 2308]. It is covered only by c_13 and c_79 together. |
| The witnesses are tail copies | At q = m_11 = 179, the missed copy c_79 = (6269, 6271) is a witness. |
| Comb sign as first written | Corrected and PROVED: with D = J_{g'} − j, h dividing the lower leg of c_{g'} strikes copy j iff D ≡ 0 or +15⁻¹ (mod h); on the upper leg, D ≡ 0 or −15⁻¹. |
| W, last clause: "only fixed-least-residue chains ever meet a window" | Case 19: 13 mod 2310 → 4633 → 64693 → 1085713 → 10785403 → 233878273 → …; 13 is in Cap(11). |
| "Doom depends only on the acting bound" | Withdrawn (D5). |
| Level-2 pair rule, ungated form (κ' is the case of a level-2 column) | h = 7, κ' = (1, 1), D = 2: the raw count is 0, the form gives 2. |
| Level-2 pair rule, indicator form | h = 53, κ' 1 then 19, D = 3 and D = 50: ν = 2 and the raw count is 2, while an indicator term is at most 1. |
| One seed-period-1 list {11, 13, 29, 31, 67, 79} for both branches | h = 23, branch 19, seed 6, with J_29(6) ≡ 6 (mod 23). |
| log(top column)/log(period) = 2^(−k) at finite q | It holds only as a limit. At q = 101 the ratios are 0.5000, 0.2612, 0.1450 for k = 1, 2, 3. |

Scope note [kernel 2.3 item 6, under Standing]: the worker's "exactly at" list for the level-2 zero class covers h ≤ 3000 only. Under the level-2 per-leg rule, (26041, κ' = 1) also has a struck zero class.

---

## 6. Not established (open questions)

**The range and the chain**
- RANGE itself. The kernel uses the weaker `RangeStatement` form (any twin pair (p, p + 2) with q < p and p + 2 ≤ q#) as the hypothesis of `range_implies_unbounded`; RANGE implies it.
- (a) Is the missed-copy hand-off sequence infinite, with every hand-off g_{i+1}² + B ≤ (g_i² + A)# holding (the R14 target) [two-paths 4, A1.14(ii)]?
- (b) Are there infinitely many gears g with c_g revealed, i.e. with g² + 28, g² + 30 (or g² + 10, g² + 12) both prime [shelves 4.1]?
- (a) implies (b) [two-paths A1.14(iii)].
- The converse RANGE ⇒ chain(S_1). It is reduced to the S_1 hand-offs at live anchors [kernel 2.1].
- Does RANGE exclude a finite S_1? The converse implies it.
- No statement was found that produces a revealed missed copy in (r# − 2, m_r# − 2] from revealed tail or head copies.
- That a revealed region copy is independent of the status of c_g and c_{g'} rests on four instances, not on a residue-level theorem. The fixed-K version (via D8 and Shiu) was not attempted.
- Empty tails beyond r = 61. Measured: tail(7) is empty, and tail(r) is non-empty at every r = 11..61 [kernel2 2A].
- The base-30 converse at live anchors (does RANGE force S_1 to meet (r, r# − 2] at every live anchor r). Refuted only at the level of rules that also hold in base 6, where S_1 = {29} and every tail holds (41, 43) [kernel2 2A].
- Whether the limit of the derivation tower contains natural numbers other than 1 and 29 (measured: none below 2·15^10), and whether any column of the limit is revealed at every level [kernel2 2B].
- The reverse arrows chain(regions) ⇒ chain(stretch) ⇒ chain(S_1), and chain(SQ) ⇒ chain(G) [derived 4.C].
- A property weaker than the hand-off that still implies it.
- Strict increase of X at every consecutive gear pair.
- A primality certificate for g* and its 71-digit legs.
- The lattice chain converse: do revealed composite columns shorten any hand-off?

**Missed copies, escape classes, doom**
- Is any escape class doomed? It reduces exactly to R. "Infinitely many primes g² + A with g prime" is not proved.
- "Non-integer paths meet windows only finitely often" is unresolved. It would imply that some class is doomed.
- The infinite shape model. It needs a Hall-type supply statement.
- Whether tree shape plus window are consistent with doom at all heights, and whether doom depends on E_h at a fixed acting bound.
- The leaf question beyond level 17: is there a class whose only revealed gear is P⁺?
- Is Cap(q) ∩ Esc(q) = ∅ at infinitely many q?
- Are the κ = 4 entering kills finite? Only g = 31, 83 and 227 occur up to 10⁷.
- Does the full-sieve survivor density equal ∏ w_h? Only the fixed-H statement is proved. The raw/product ratios at 200000 are 1.283 and 1.317.
- Closed forms, in L-values, for the two singular-series constants.
- Survivor censuses of the n-th missed copy for n ≥ 4.
- Is the inert run length bounded? Only the O(√h) least-non-residue bound is proved.
- The density law 2^(−r_L) was checked only for h ≤ 200000 and L ≤ 13.
- Gears above g acting on copies beyond the protected stretch are not analysed.
- Whether every c_g in a range can be killed by some gear.

**Shelves and stretches**
- Are there finitely many empty head stretches, empty regions or fully covered stretches? Measured to g = 10⁶ and to 10⁸.
- Full covering of three or more consecutive gears beyond 10⁸.
- Is D_sq(primes < g) ≥ l(g) for all g > 31? (D_sq is the covered-run length in the table of derived D10.)
- A version of D8 with N growing with g, and the whole-stretch S-free variant named in the record.
- The lengths of later g-silent runs inside the protected region when L > p1.
- Enumerated-class checks of the shelf–survivor identity beyond machine 23.

**Pairs of gears and kills**
- Are the equality families F10, F30, F12e, F28e, F12o, F28o infinite?
- Finiteness of the kill lists beyond 4·10¹⁸. The extension from 10⁸ rests on a published maximal-gap table that was not re-verified.
- The near-neighbour result (3.2) for r > 12 when 149 ≤ q < 10¹⁵.
- Is K(d) empty for infinitely many d?
- Is each realizable member of S(D) realized infinitely often? For example, 21841 (D = 16) and 128521 (D = 26) are not realized below 10⁶.
- ALL-KILLED on the hand-off intervals (g_i, G_i] [derived 4.E]. (g_i, G_i] is the interval in which E12's restatement of the hand-off equivalence places the next revealed gear after g_i [derived E12, 6.E]; the record does not give G_i in closed form. ALL-KILLED(W) is the record's term ("ALL-KILLED with r0(k)", derived 2.5); its defining text is not on disk. Result, derived E5: ALL-KILLED(W) holds iff every gear in W has a composite leg on its missed copy.
- Why the least-c table is constant on p mod 15d; the sexy-pair split-copy count beyond p < 6000.

**Derived machine and operator**
- Class counts at levels k ≥ 4. That every even value up to 2^(k+1) occurs is not proved.
- The pair rule at level ≥ 3 as Q-polynomials. The coincidence primes 53, 631 and 883 are not characterised.
- Persistence closed forms beyond seed periods 1 and 2.
- The converse arrows chain(S_(k−1)) ⇒ chain(S_k). S_3 has no gear member with y ≤ 2·10⁵ (BPSW).
- Der_int(P²): its class counts and pair rule.
- The O-shaped range on D's own period, and its relation to RANGE.
- Bridging strikers at distance D; the rational residues open under every striker (S-unit search not run); a derived lap-machine centre.

**Offset columns**
- V1, the orbit statement over F (mod M_X, X = sh(j)): measured exact to q = 61, not proved.
- A structural reason for the q = 19 failure of V2 (the machine orbit mod M_q). Control of the untied gears in (X, sh(x)]. Fibre chains other than via the range statement.
- S4-type existence for sources near n ≈ q#/6, and target-modulus transfer from above the range.
- "No source leg strikes r" is a measurement only (j < 2·10⁶).
- The partner condition of B1.6 was never recomputed.
- A closed bound on k* (the measured maximum is 9). Mean cost ≈ X#/4 is unproved.
- A structural rule for when the gears in (q, sh(j')] divide t. The necessity of p⁴ < 30q# (two-paths B3.6).

**The barrier**
- Which acting-based statement forbids a total blame assignment. The barrier shows any such statement must involve acting; none was produced.

**Runs not made**
- B9 at level 19 was not re-run, and level 23 (8,825,600 classes) was not run.
- Exact G_31 (3.3·10¹⁰ columns) was not attempted.
- D_sq at y = 31 was not computed.
- The all-pairs forced enumeration for q ≥ 29 was not run.
- The V1 witnesses for q ≥ 67 were not rerun; they are BPSW-probable.

---

## 7. Where the records are

**Theory tree**: `C:/dev/primes/research/proof/theory_tree.md`
- Nodes R5.f.xxxv.c.xxv to c.xxxi are at lines 4702–5053. Their one-line log entries are at lines 6637–6643.
- c.xxv: the drift correction. Nodes c.xvii to c.xxiv are window work, not range work, and stand as window results. Script: `research/stack/r8/range_tiers.py`.
- c.xxvi and c.xxvii name no separate record file; the node text is the record.

**Round records**, all in `C:/dev/primes/research/proof/`:
- `shelves_cofactors_2026-09-23.md` (c.xxviii): acting, missed copies, shelves, cofactors, survivor classes, gear pairs.
- `range_two_paths_2026-09-23.md` (c.xxix): Path A (the missed copies of the upper gears) and Path B (the translation law with acting built in).
- `derived_machine_2026-09-24.md` (c.xxx): derived machine, escape against the cutoff, hand-off chain, protected stretch, neighbour kills, forced pairs and orbits.
- `range_kernel_2026-09-25.md` (c.xxxi): the kernel status, the converse reduction, doomed classes and the derivation operator.
- `range_kernel2_2026-09-25.md` (c.xxxii): the second kernel batch, the converse at the live anchors (anchors measured to r = 61, the base-6 rule-level refutation, the fixed-row criterion), and the limit of the derivation tower.

**Lean kernel**: `C:/dev/primes/proofs/`
- `RangeCopies.lean`, `RangeLocator.lean`, `RangeMissed.lean`, `RangeHandoff.lean`, `RangeMissedReveal.lean`, `RangeDerived.lean`, `RangeChain.lean`.
- Registration: `[[lean_lib]]` blocks at the end of `proofs/lakefile.toml`. None is in defaultTargets.

**Scripts**: under `C:/Users/Alex/AppData/Local/Temp/claude/C--dev-primes/a1c3a0ad-3acd-4fbe-b091-1e36256528ec/scratchpad/`
- The top level holds `asm_lib.py` and `asm_1_missed.py` to `asm_4_protect.py` (shelves record).
- `rec2309/` (two-paths record reruns).
- `assemble_0924/` (derived-machine record).
- `converse_0925/`, `doom_0925/`, `adj_doom/`, `adj_derop/`, `defend_der/` and `assemble_0925/` (kernel record).
