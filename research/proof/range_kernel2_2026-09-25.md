# Round record, 2026-09-25: ONE MACHINE, ONE PERIOD

Scope: kernel additions, the converse at the live anchors, the derivation tower in the limit, and the range-line map. This assembly re-ran the maths checks listed in 2C. It did not rebuild Lean, and nothing under C:/dev/primes was modified.

Terms (as in the brief):
- **Columns.** Column n holds (6n − 1, 6n + 1).
- **Strikes.** Gear g (a prime ≥ 5) strikes column n iff g divides 6n − 1 or 6n + 1.
- **Machine q.** The gears 5..q. Its period is q# and its RANGE is that whole period.
- **Copies.** Copy j is (30j − 1, 30j + 1). Gear g strikes copy j iff j ≡ ±30⁻¹ (mod g).
- **Acting.** g acts on copy j iff g² ≤ 30j + 1.
- **Revealed.** A copy j ≥ 1 is revealed when both legs are prime.
- **Range statement at q.** Some copy with 30j − 1 > q and 30j + 1 ≤ q# is revealed. RANGE is the range statement at every q.
- **Cases.** (A, B) = (28, 30) in case 1 (g² ≡ 1 mod 30) and (10, 12) in case 19 (g² ≡ 19 mod 30).

---

## Part 1 — Kernel

All three modules are in namespace RangeLine under C:/dev/primes/proofs/. Each has a `[[lean_lib]]` block appended at the end of proofs/lakefile.toml, and none is in defaultTargets. All three files are untracked (git status `??`), and lakefile.toml shows `M`.

Checks run by this assembly (read-only, no rebuild):
- grep for sorry, admit, axiom and native_decide over all seven Range*.lean files: no matches.
- The lakefile ends with the blocks RangeHandoff, RangeMissedReveal, RangeDerived, RangeChain.
- The theorem counts in the files are 11, 15 and 4, matching the reports.

### RangeMissedReveal (imports RangeMissed)

**Definitions**
- `MissedOffsets g A B := (g^2 % 30 = 1 ∧ A = 28 ∧ B = 30) ∨ (g^2 % 30 = 19 ∧ A = 10 ∧ B = 12)`
- `missedA g := if g^2 % 30 = 1 then 28 else 10`
- `missedB g := if g^2 % 30 = 1 then 30 else 12`

**Main theorems**
- **(a) `legs_coprime_30 (g A B) (hAB : MissedOffsets g A B)`.** Both g² + A and g² + B are odd, have mod 3 ≠ 0, have mod 5 ≠ 0, and are Coprime to 30. It needs only the offsets, not g prime or g ≥ 7.
- **(b) `composite_leg_small_factor (g A B L) (hg : g.Prime) (h7 : 7 ≤ g) (hAB) (hLeg : L = g^2+A ∨ L = g^2+B) (hnp : ¬ L.Prime)`.** Concludes ∃ h, h.Prime ∧ 7 ≤ h ∧ h < g ∧ h ∣ L. The witness is L.minFac, and L > 1 is not needed because L ≥ 49.
- **(c) `missed_copy_revealed_iff (g A B) (hg) (h7) (hAB)`.** States ((g^2+A).Prime ∧ (g^2+B).Prime) ↔ ¬ ∃ h, h.Prime ∧ 7 ≤ h ∧ h < g ∧ (h ∣ g^2+A ∨ h ∣ g^2+B).
- **`missed_copy_revealed_iff_case1`** (h1 : g^2 % 30 = 1): statement (c) with 28 and 30.
- **`missed_copy_revealed_iff_case19`** (h19 : g^2 % 30 = 19): statement (c) with 10 and 12.

**Helpers**
- `missedOffsets_missed`
- `legs_mod30` (the legs are ≡ 29 and ≡ 1 mod 30)
- `mod30_unit_facts`
- `prime_dvd_ge7`
- `leg_le` (each leg ≤ g² + 30)
- `gear_not_dvd_leg`

**Build.** `lake build RangeMissedReveal`, run from proofs/ in the foreground, printed "Built RangeMissedReveal (141s)" and "Build completed successfully (3011 jobs)", with 0 errors and 0 warnings. The reviewer also compiled the file fresh with `lake env lean`: exit 0, no output.

**Sorry.** None (grep exit 1).

**Axioms.** All 11 theorems use only propext, Classical.choice and Quot.sound. prime_dvd_ge7 and leg_le use only propext and Quot.sound.

**Reviewer.** Builds, sorry-free, axioms correct, faithful. MissedOffsets is satisfiable for every prime g ≥ 7 (missedOffsets_missed), and the bounds 7 ≤ h < g are exactly as requested.

**Corrections**
- The statements are true as written. Two are stronger than asked: (a) drops "g prime, g ≥ 7", and (b) drops "L > 1".
- The brief's proof sketch for (b) had a wrong step. "g² + 30 < (g+1)²" needs g ≥ 15, so it is false at g = 7, 11 and 13. At g = 11 the leg 151 is at least 12² = 144.
- The Lean proof uses a different step: a prime h > g has h ≥ g + 2, so h² > g² + 30 ≥ L (by larger_gear_sq). Nat.minFac_sq_le_self gives h² ≤ L, a contradiction.
- The reviewer confirmed this correction.

### RangeDerived (imports Mathlib.Tactic, RangeMissed)

**Definitions**
- `J1 y := (y^2 + 29)/30`
- `J19 y := (y^2 + 11)/30`

**Theorems (15)**
- **(a)** `J1_exact` (y² % 30 = 1): 30·J1 y = y² + 29. `J19_exact` (y² % 30 = 19): 30·J19 y = y² + 11.
- **Extra.** `copy_map_on_gear` (g prime, g ≥ 7): (g² % 30 = 1 ∧ 30·J1 g = g² + 29) ∨ (g² % 30 = 19 ∧ 30·J19 g = g² + 11).
- **(b)** `J1_fixed_identity` (case 1): 30·((J1 y : ℤ) − y) = (y − 1)(y − 29). `J1_fixed_iff` (case 1): J1 y = y ↔ y = 1 ∨ y = 29.
- **(c)** `sq_add_eleven_ne`: y² + 11 ≠ 30y for every natural y. `J19_no_fixed` (case 19): J19 y ≠ y.
- **(d)** `J1_strictMono` and `J19_strictMono`: both points in the residue set and y < y' give J y < J y'. `J1_strictMonoOn` and `J19_strictMonoOn` state the same on {y | y² % 30 = 1} and {y | y² % 30 = 19}.
- **(e)**
  - `J19_lt_self_iff` (case 19): J19 y < y ↔ 1 ≤ y ∧ y² + 11 < 30y.
  - `J19_lt_self_list`: (y² % 30 = 19 ∧ J19 y < y) ↔ y ∈ {7, 13, 17, 23}.
  - `J1_lt_self_iff` (case 1): J1 y < y ↔ 1 < y ∧ y < 29.
  - `J1_lt_self_list`: (y² % 30 = 1 ∧ J1 y < y) ↔ y ∈ {11, 19}.

**Build.** "Built RangeDerived (13s)" and "Build completed successfully (3011 jobs)", with 0 errors and 0 warnings. Fresh `lake env lean` exited 0.

**Sorry.** None. The finite checks use interval_cases with omega or simp_all, and the concrete directions use plain decide.

**Axioms.** All 15 use [propext, Quot.sound] or [propext, Classical.choice, Quot.sound].

**Reviewer.** Faithful. The reviewer checked the two (e) lists independently from the residue classes (y ≡ ±7, ±13 and y ≡ ±1, ±11 mod 30). The one difference from the spec is a name: the spec's J_lt_self is split into J19_lt_self_iff and J1_lt_self_iff.

**Corrections.** None. Three results were added that were not asked for: copy_map_on_gear, J1_strictMonoOn and J19_strictMonoOn.

### RangeChain (imports RangeHandoff)

**Common hypotheses (C)**
- p : ℕ → ℕ with StrictMono p
- ∀ k, (p k).Prime and ∀ k, (p k + 2).Prime
- p 0 = 29
- ∀ k, p (k+1) + 2 ≤ primorial (p k)

**Theorems (4)**
- `primorial_le_of_le {a b} (h : a ≤ b) : primorial a ≤ primorial b`. This restates Mathlib's primorial_mono.
- `chain_implies_range_all (C) : ∀ q, 7 ≤ q → RangeStatement q`. The proof takes k = Nat.find (∃ k, q < p k); that k exists because p(q+1) ≥ q+1.
  - If k = 0: 31 ≤ 210 = primorial 7 ≤ primorial q.
  - If k = j+1: p j ≤ q, so p(j+1) + 2 ≤ primorial(p j) ≤ primorial q.
- `chain_implies_range (C) : ∀ q, q.Prime → 7 ≤ q → RangeStatement q`
- `chain_implies_unbounded (C) : ∀ N, ∃ r, N < r ∧ r.Prime ∧ (r+2).Prime`. It composes chain_implies_range with range_implies_unbounded.

**Build.** "Built RangeChain (12s)" and "Build completed successfully (3019 jobs)", with 0 errors and 0 warnings. Fresh `lake env lean` exited 0.

**Sorry.** None.

**Axioms.** All 4 use [propext, Classical.choice, Quot.sound].

**Reviewer.** Faithful: the hypotheses match the spec and the step runs in the requested direction. The reviewer checked for off-by-one errors in both Nat.find branches. `primorial` is Mathlib's global definition (∏ p ∈ range(n+1) with p prime), and no local primorial exists.

**Corrections.** None. chain_implies_range_all was added: it holds for every q ≥ 7, prime or not.

### Kernel, all modules
- **Kernel-proved:**
  - strike and leg rules on copies
  - the locator closure
  - the missed-copy legs
  - range_implies_unbounded, twin_serves, serves_interval
  - the missed-copy reveal rule (both legs prime ⇔ no prime 7 ≤ h < g divides a leg)
  - the J1/J19 fixed-point, monotonicity and below-self lists
  - chain ⇒ RANGE ⇒ twin primes unbounded
- **Remains a hypothesis in Lean:** RANGE itself, in its RangeStatement form (any twin pair in (q, q#]).

---

## Part 2 — Maths

### 2A. The converse at the live anchors

**Terms**
- **Rev**: the lower legs of revealed copies.
- **S_1**: the lower legs of revealed missed copies.
- **x⁺**: the next member of the set in question.
- **m_r** = max{m ∈ S_1 : m + 2 ≤ r#}, and g_r is its gear.
- **tail(r)** = Rev ∩ (m_r, r# − 2]. r is **live** when tail(r) ≠ ∅.
- **v(m)** = max{p ∈ Rev : p < m⁺}.
- **σ(y)**: the least prime with σ(y)# ≥ y.
- **Anc(m)**: the primes in [σ(m+2), σ(m⁺+2)).
- **t_r(S)** = max{s ∈ S : s + 2 ≤ r#}.
- **(h−)**: the prime before h.

**Formulas**

- **F1.** Given RANGE, chain(S_1) ⇔ m_r > r at every prime r ≥ 7 with tail(r) ≠ ∅. Equivalently:
  - S_1 meets (r, r# − 2] at each such r;
  - at each such r there is a prime h with r < h² + A_h and h² + B_h ≤ r#, such that no prime 7 ≤ f < h divides (h² + A_h)(h² + B_h).
- **F2 (pair form, consecutive S_1 gears g < g⁺).** The three conditions below are equivalent:
  - (g⁺)² + B⁺ ≤ (g² + A)#
  - g² + A ≥ σ((g⁺)² + B⁺)
  - g² + A > max{r : g² + B ≤ r# < (g⁺)² + B⁺}
- **F3.** Failure at r ⇔ g_r² + A ≤ r, and then (g_r⁺)² + B > r#. So m_r ≤ r and m_r⁺ + 2 > r#: the anchor pair straddles the whole of (r, r# − 2].
- **F4 (transport).** c_h ∈ S_1 ⇔ some (p, p+2) ∈ Rev has p > h and p ≡ h² + A_h (mod (h−)#). For h > r (h ≥ 11), this class meets (r, r# − 2] only in c_h.
- **F5 (base 6).** The missed column is (g² + 4, g² + 6), and 5 divides (g² + 4)(g² + 6) for every g ≥ 7. S_1⁽⁶⁾ = {29}. The anchor form fails at every r ≥ 29 while every tail is non-empty.
- **F6 (fixed row).** Row f kills every unit class of (x² + a)(x² + a + 2) ⇔ (f = 5, a ≡ 4) or (f = 3, a ≡ 0 or 2). f = 2, a odd, is excluded because 2 divides the base. This never applies in base 30.

**PROVED** (item numbers as the round text uses them)

1. **Crossing and pair forms,** for sets S of primes ≥ 29. chain(S) ⇔ S meets (r, r# − 2] at every prime r ≥ 7 ⇔ t_r(S) > r at every prime r ≥ 7. For consecutive (m, m⁺) in S:
   - m⁺ + 2 ≤ m# ⇔ m ≥ σ(m⁺ + 2) ⇔ Anc(m) = ∅ or m > max Anc(m).
   - The reason: max Anc(m) = prevprime(σ(m⁺ + 2)), and m is prime.
2. **Live anchors.**
   - RANGE(r) with tail(r) empty implies m_r > r.
   - Given RANGE, chain(S_1) ⇔ m_r > r at every live r.
   - m_r > r ⇔ the h-condition of F1. The legs are ≤ h² + 30 < (h+2)², h divides neither leg, and 2, 3 and 5 are excluded by the offsets.
3. **When m_r⁺ exists.**
   - The only chain inequality of RANGE with m_r⁺ as successor is v(m_r) ≥ σ(m_r⁺ + 2).
   - σ(m_r⁺ + 2) = nextprime(max Anc(m_r)). This equals r⁺ iff r = max Anc(m_r), the top anchor.
   - A failure at r is also a failure at every r' ∈ Anc(m_r) with r' ≥ r. Nothing makes the failing r the top anchor.
   - (g_r⁺)² + B = m_r⁺ + 2 > r# at every r, failing or not.
   - m_r = c_{g_r}, and failure ⇔ g_r² + A ≤ r.
   - The failure mode "S_1 finite" has no m_r⁺. It lies outside this anatomy and has to be treated on its own.
4. **Copies between the anchors.**
   - (a) Every revealed copy strictly between m_r and m_r⁺ sits at position ≥ 2 on a shelf h with g_r ≤ h < g_r⁺. The case-1 head of g_r⁺ lies between them and is never revealed.
   - (b) Those ≤ r# − 2 are exactly tail(r), on shelves in [g_r, isqrt(r#)].
   - (c) The rest lie in (r# − 2, m_r⁺).
     - At a top anchor, each of them has service [σ(p+2), p) ⊆ [r⁺, m_r⁺) = service(m_r⁺), so RANGE (coverage) neither needs them nor excludes them.
     - When r < r* = max Anc(m_r), those ≤ r*# − 2 are tail copies of the higher anchors, and the rest lie in service(m_r⁺) by the same argument at r*.
   - (d) v(m_r) ∈ (r# − 2, m_r⁺) iff that interval holds a revealed copy. Otherwise v(m_r) = max tail(r), or m_r when the tail is empty.
5. **Transport (F4).**
   - A revealed class member p > h has no prime f < h dividing p or p + 2. So no such f divides a leg of c_h, and those legs (< (h+2)², not divisible by h) are prime.
   - For h ≥ 11, c_h < (h−)# and (h−)# ≥ r#.
   - A class whose c_h is not revealed carries the striker on the same leg in every member, so it holds no revealed copy.
7. **Fixed row (F6),** by an exact residue argument: the (f−1)/2 nonzero squares must lie in {−a, −a−2}, which forces f ≤ 5.
8. **From (1) and (2).** Given RANGE, chain(S_1) ⇔ no live r has m_r ≤ r.

**MEASURED (exact ranges)**
- **Tails.** tail(7) = ∅: the only copy in (179, 208] is 209 = 11·19. tail(r) ≠ ∅ at every r = 11..61. m_r > r at r = 7..61.
- **Least tail copies at r = 47, 53, 59, 61:** 614889765550167089, 32589139221206666519, 1922760080996272415879, 117288377702574280788659. At r = 11..43 they equal the kernel record.
- **Anchors at r = 47..61** (recomputed by this assembly):

  | r | gears g_r → g_r⁺ | m_r → m_r⁺ | struck primes strictly between |
  |---|---|---|---|
  | 47 | 784149071 → 784149857 | 614889765550163069 → 614890998233120459 | 39 |
  | 53 | 5708689799 → 5708707861 | 32589139221206660429 → 32589345442243195349 | 799 |
  | 59 | 43849288261 → 43849311221 | 1922760080996272404149 → 1922762094556116510869 | 923 |
  | 61 | 342473908061 → 342473930599 | 117288377702574280779749 → 117288393139928668498829 | 848 |

  Intermediate-gear totals for r = 7..61: 15, 15, 15, 30, 35, 173, 49, 118, 329, 672, 92, 39, 799, 923, 848. These are check sizes, not results.
- **v(m_r) at r = 47..61:** 614890998233117399, 32589345442243194719, 1922762094556116507509, 117288393139928668498439. At every r = 7..61, v(m_r) + 2 > r# and shelf(v(m_r)) > isqrt(r#).
  - This clause was labelled PROVED in (4) and is now MEASURED.
  - shelf(v) is the largest prime g with g² ≤ v + 2.
- **Anchor pattern.**
  - Anc(179) = {7, 11}, the only two-anchor pair in r = 7..61.
  - r = max Anc(m_r) at r = 11..61; r = 7 is the only non-top anchor.
  - Anc(m_r) = {r} at r = 13..61.
  - σ(6271) = 13, not 11, at r = 7.
- **Log ratios ln m_r / ln r#:** 0.970129 at r = 7, 0.945885 to 0.999904 at r = 13..37, and ≥ 0.99996 at r ≥ 41. These are height measurements that no proof uses.
- **Exhaustive tails at r = 7, 11, 13, 17, 19, 23:**

  | r | 7 | 11 | 13 | 17 | 19 | 23 |
  |---|---|---|---|---|---|---|
  | revealed in (m_r, m_r⁺) | 43 | 43 | 125 | 644 | 3245 | 58958 |
  | tail | 0 | 15 | 50 | 548 | 285 | 30199 |
  | in (r# − 2, m_r⁺) | 43 | 28 | 75 | 96 | 2960 | 28759 |

  In all six: 0 tail shelves outside [g_r, isqrt r#], 0 position-1 or head copies, and 0 exceptions to service containment at the top anchors 11..23. Deleting all 31918 such copies leaves a witness for every q = 7..23.
- **S_1 and C7.**
  - Gears ≤ 2·10⁵ give 328 revealed gears (233 case 1, 95 case 19). The three pair forms agree on all 327 consecutive pairs.
  - C7: S_1 from the 14 gears ≤ 1200 (7, 11, 13, 79, 89, 97, 127, 131, 223, 241, 251, 409, 541, 739; largest copy 546149) meets (r, r# − 2] at every prime r ∈ [7, 2·10⁵].
- **Synthetic set** S = {59, 149, 179, c_h0}, with h0 = 88317167310118487253308991274873222475549:
  - h0 is prime and case 1, and both legs of c_h0 are prime.
  - 199# < c_h0 + 2 ≤ 211#.
  - The form fails at r = 179, 181, 191, 193, 197, 199.
  - σ(m_r⁺ + 2) = r⁺ only at r = 199.
- **Transport.** 525 (r, h) cases (r ≤ 61, h ∈ (r, 199]) gave 0 class members in (r, r# − 2] other than c_h. Among h = 7..43, revealed members exist exactly for h = 7, 11, 13. The first revealed member of c_7's class (29 mod 30) is 29.
- **Bases 2, 6, 30.**
  - Offsets: {(2,4)}, {(4,6)}, {(10,12), (28,30)}.
  - Revealed missed gears: 1 (3 → 11), 1 (5 → 29), 328.
  - 0 gears escape row 3 (base 2) or row 5 (base 6).
  - Base 6: m_r = 29, (41, 43) is in every tail, and the form fails at r = 5 and at r = 29..61.
  - Least RANGE₆ witnesses at r = 7..61: 11, 17, 17, 29, 29, 29, 41, 41, 41, 59, 59, 59, 59, 71, 71.
- **Fixed-row kills.** For primes f ≤ 397, the full kills are exactly (2,1), (3,0), (3,2), (5,4).

**REFUTED**
- **The anchor sentence of (4) as written.** Refuted at r = 11: m_11 = 179 and Anc(179) = {7, 11}. What stands of it is the measured pattern above.
- **"A non-empty tail(r) forces m_r > r."** This does not follow from any rules that also hold in base 6. Instance: in base 6, S_1⁽⁶⁾ = {29} (proved: g² ≡ ±1 mod 5 puts a leg of c_g in row 5), (41, 43) ∈ tail₆(r) at every r ≥ 7, and m_r = 29 ≤ r at every r ≥ 29.
- **Finite-horizon form** "(RANGE(r') for r' ≤ R) ⇒ anchor form at r ≤ R". Refuted in base 6 for R = 29..61: RANGE₆(r) holds at r = 7..61 while m_29 = 29.
- **Conditional on RANGE₆.** RANGE₆ is open, and RANGE implies it, since every base-30 copy is a base-6 twin pair in the same interval. Given RANGE₆, "RANGE ⇒ chain(S_1)" and "RANGE excludes a finite S_1" are refuted as consequences of rules that also hold in base 6.
  - Established without that condition: in base 6 both conclusions are false. So a derivation of either from base-6-valid rules would prove ¬RANGE₆, and with it ¬RANGE.
  - The rider "any derivation must use the offset arithmetic" carries the same condition.

**Not established or unfinished**
- The base-30 converse (whether RANGE forces S_1 to meet (r, r# − 2] at every live anchor) is neither proved nor refuted. No base-30 model obeying the rules, with no fixed row, in which RANGE holds and the form fails, was built.
- Whether RANGE excludes a finite S_1 in base 30. It is refuted only at the level of the rules, by bases 6 and 2.
- The anchors stop at r = 61, because deterministic Miller–Rabin is exact only below 3.3·10²⁴.
- "Every r ≥ 13 is the only anchor" is not established past r = 61. No other item of (4) depends on it.
- Tails were enumerated exhaustively only at r ≤ 23. At r = 29..61 only the least tail copy, the greatest tail copy and v(m_r) were located.
- Least killers above 2·10⁴ came from Pollard–Brent factorisation. They are explicit divisors, but the intermediate-gear list was not independently re-derived.

**Locating and counting residue**
- No proof step bounds the height of a revealed copy.
- The one locating clause that carried a PROVED label (v(m_r) on shelves above √(r#)) is now MEASURED.
- The height facts (log ratios, m_r > r to 61, the anchor pattern to 61, C7) are labelled measurements, and no proof uses them.
- The tail-shelf bound ≤ isqrt r# holds by definition from each copy's own height; it is not an existence bound.
- The transport reach is an exact comparison of c_h with (h−)#.
- Counts appear only as check tallies. The (7) pigeonhole is an exact residue argument, not a density.

**Scripts**
- Worker: scratchpad/anchor_conv_0925/: common.py, anchors.py (7.1 s, output anchors_61.txt), checks.py (1.6 s, checks_out.txt), column.py (3.2 s, column_out.txt).
- Adjudication: scratchpad/adj_live/: a_anchors.py, b_tails.py, c_rest.py.

### 2B. The derivation tower in the limit

**Terms**
- J(y) = (y² + 29)/30 on the case-1 columns, y mod 30 ∈ {1, 11, 19, 29}.
- L_0 is the odd integers, and L_k = {y : y, …, J^(k−1)(y) are all case-1 columns}.
- delta(y) is the least k with J^k(y) not a case-1 column.
- pdepth(y) is the largest m with y ∈ P^m.
- Z_h = {x mod h : x² ≡ −28 or −30 (mod h)}.

**Formulas**
- **Tower (repaired form):** P^(k+1) = {y ∈ L_k : gcd(J^k(y), 15) = 1}.
- **Limit:** L = ∩_k cl(P^k) = Z_2^× × ι⁻¹({1, 4, 11, 14}^ℕ) ⊂ Z_2 × Z_3 × Z_5.
  - ι(y) = (J^k(y) mod 15)_{k≥0}, and ι∘J = shift∘ι.
  - The levels interleave: P¹ ⊇ L_1 ⊇ P² ⊇ L_2 ⊇ ….
- **Next digit:** J^k(r + 15^k s) ≡ J^k(r) + s·∏_{i<k} J^i(r) (mod 15).
- **Inverse:** ι⁻¹(a_0 a_1 …) = lim_k s_{a_0}∘…∘s_{a_{k−1}}(0), with s_a(z) = √(30z − 29) taking the root ≡ a (mod 15).
- **Distance to the fixed point at each place:**
  - p = 3, 5 repel: |J(y) − 1|_p = p·min(|y − 1|_p, |y + 1|_p).
  - p = 2 is neutral on 1 + 4Z_2: |J(y) − 1|_2 = |y − 1|_2 and |J(y) − 29|_2 = |y − 29|_2.
- **Fixed points in L_15:** 1, 29, ω_19 and ω_11.
- **2-cycles at 3 and at 5:** −15 ± 8√−11, the roots of y² + 30y + 929 = 0.
- **Strikes:** h strikes level k iff J^k(y) ≡ ±30⁻¹ (mod h) iff J^(k−1)(y) mod h ∈ Z_h. E^(k)_h = J_29^(−(k−1))(Z_h).
- **Revealed at every level (y > 29):** for every h ≥ 7, J^(i_h)(y) mod h ∈ N_h, with i_h = min{i : J^i(y) ≥ h}.
- **Orbit chain — LOCATED** (belongs with RANGE and chain(S_1) consequences, not with the tower limit):
  - p_(k+1) + 2 = x_k² + 30 or x_k² + 12, which is ≤ ((p_k + 1)/30)² + 30 < p_k²/4 < p_k#. The last step uses Bertrand.
  - For an orbit revealed at every level, ∪_{k≥k0} [σ(p_k + 2), p_k) = [σ(30·J^(k0)(y) + 1), ∞).

**Established**
- **Tower membership, repaired gcd form** with L_0 = the odd integers. Raw P^(k+1) has 8, 32, 128, 512, 2048 classes for k = 0..4, and the gcd form equals raw at every k.
- **Periodicity and fibre (PROVED).**
  - (y + 2·15^m t)² ≡ y² (mod 4·15^m), so y mod 2·15^m determines J(y) mod 2·15^(m−1).
  - J: P^(m+1) → P^m is onto, with every fibre of size 4. The class count 2^(2k+1) follows and restates the record's count.
  - For m ≥ 2 every class is case 1. Case rule over all columns below 10⁶: 133,333 case-1 and 133,333 case-19 columns, 0 failures.
- **Next-digit law (PROVED).** J(y + 15^k s) ≡ J(y) + 15^(k−1)·y·s (mod 15^k), then induction.
  - There are exactly 4^k odd depth-k reps below 2·15^k, and each word of {1, 11, 19, 29}^k occurs once.
  - Precise form of the loose phrase: J^k maps each depth-k ball r + 15^k Z_15 bijectively onto Z_15, and it maps the cylinder's part of L onto L.
  - The size statements (Haar measure 0, dimension log 4/log 15, log 2/log 3, log 2/log 5) are counting and are not part of what stands.
- **Mixed fixed points.**
  - (1 ∈ Z_3, 29 ∈ Z_5) has word 19^∞, with reps 80029 (mod 15⁴) and 2881250029 (mod 15⁸).
  - (29 ∈ Z_3, 1 ∈ Z_5) has word 11^∞, with reps 21251 and 2244531251.
  - Exact deltas 4, 8, 4, 8, each followed by an exit. They do not approximate the 2-cycle.
  - The only real periodic points are 1 and 29: J(y) − y = (y − 1)(y − 29)/30, J(ℝ) ⊆ [29/30, ∞), and J is increasing there.
- **2-adic.**
  - Cycle lengths are powers of 2: an isometric permutation lifts each cycle mod 2^n to a cycle of equal or double length. So exact periods in Z_2 are powers of 2.
  - Φ_2 and Φ_4 have no 2-adic roots.
  - Periodic points must be odd elements of Z_2; an even y, or one with v_2(y) < 0, escapes.
- **Strikers (stands as stated).**
  - K_h = 1 + the longest backward chain when Z_h ≠ ∅.
  - Z_h = ∅ for 107 primes below 3000 (19, 41, 61, 73, 83, 89, 97, 103, 139, 173, …); for these K_h = 0, and h never strikes a limit column.
  - MEASURED below 3000: 37 persistent strikers, with struck-cycle lengths 11, 13, 29, 31, 67, 79 (1); 37 (2); 23, 107 (3); 1801 (4); 281, 1697 (5); 443 (7); 647, 883 (9); 337 (10); 163, 2281, 2711 (12); 257 (13); 59, 757 (14); 1123 (15); 269 (16); 1607 (18); 149 (21); 1453 (27); 571 (31); 1873, 2297 (37); 2633 (48); 1087 (50); 971 (56); 1747 (61); 2819 (71); 2203 (76); 1733 (93).
  - Values: K_7 = 1, K_17 = 2, K_43 = 4, K_109 = 13, K_179 = 15. The maximum, 138, occurs only at 2269.
- **N_h ≠ ∅ for every prime h ≥ 7 (PROVED).** 1 is struck iff h ∈ {29, 31} (copy 1 = (29, 31)). 29 is struck iff h ∈ {11, 13, 67, 79} (copy 29 = (869, 871) = (11·79, 13·67)). The two sets are disjoint.
- **Revealed criterion (PROVED; it restates ACTING).** For a column x ≥ 15, the next copy is revealed iff no prime 7 ≤ h ≤ x divides a leg. This is exact because √(x² + 30) < x + 1 iff x ≥ 15.
- **Depth measurements.**
  - Below 10⁷ (exact orbits): delta counts 1: 977777, 2: 260741, 3: 69527, …, 10: 9, 11: 2.
    - The maximum finite delta, 11, is reached by exactly 674,971 (exit 3 mod 30) and 8,645,779 (exit 27).
    - pdepth 11 is reached by 674971, 1305001, 5772971, 7812529, 8645779 and 8887529.
  - Below 2·15^10 (residue tree of 4^10 reps, delta by exact descent from 2·15^70):
    - The maximum finite delta, 22, occurs at 360,329,223,301 only (exit 27).
    - Delta 20 occurs at 816,167,552,029 (exit 27) and 435,189,446,551 (exit 13). No column has delta 21, and 7 columns reach 19.
    - Only 1 and 29 have infinite delta.
- **Revealed runs (MEASURED)** over columns 1..10⁶ (266,666 columns): runs of revealed levels {0: 262714, 1: 3937, 2: 14}, plus column 1, which is revealed at every level.
  - The 14 are 7229, 9371, 12421, 57269, 143701, 215129, 225181, 306991, 330221, 448121, 576029, 655201, 732601, 758441.
  - In 57269, 225181, 306991 and 758441 the second revealed step is a case-19 step.
- **Orbit-chain arithmetic.** It holds for every orbit, revealed or not (0 violations over 17,877 level pairs). The service intervals overlap:
  - 7229: [23, 52258469), [37, 3034386318629)
  - 9371: [23, 87815669), [41, 8568435441749)
  - 12421: [23, 154281269), [41, 26447455858709)

**REFUTED**
- **The written tower form** P^(k+1) = {y ∈ L_k : 15 ∤ J^k(y)}.
  - Instances: y = 11, k = 1 (J(11) = 5, not a column), and y = 3, k = 0.
  - The written form gives 14, 56, 224, 896, 3584 classes against the raw 8, 32, 128, 512, 2048.
  - Each of J(31) = 33, J(41) = 57, J(49) = 81, J(59) = 117, J²(251) = 147141 and J²(349) = 549725 is odd and not divisible by 15, yet shares a factor 3 or 5.
- **"674,971 is the unique deepest column (below 10⁷)."** 8,645,779 also has delta 11, and six columns have pdepth 11.

**Not established or unfinished**
1. Whether W is empty, that is, whether L ∩ ℕ is exactly {1, 29} or is infinite.
   - Measurement: W is empty below 2·15^10 (about 1.15·10¹²).
   - No finite congruence decides it, since every cylinder of every length holds integers.
2. Whether any column of W is revealed at every level. Such a column would give RANGE at every q ≥ σ(30J(y) + 1), and no finite striker set forbids it. Measured: no column ≤ 10⁶ other than 1 has more than 2 consecutive revealed levels.
3. 2-adic periodic points of period ≥ 8. Only periods ≤ 4 are certified. The l = 5 factorisation passed 5 minutes and was stopped, and the mod-2^n cycle data stop at n = 20.
4. Class counts at levels ≥ 4, the Galois group of the J-tower, and the Frobenius at h, which governs |E^(k)_h|. Established only: 3 and 5 split completely in every ℚ(J^(−k)(a)).
5. Rational non-integer points of L (denominators coprime to 30) were not examined.
6. Numerical coverage: the striker checks cover primes up to 3000, and the raw leg checks cover h < 400 and levels ≤ 4.

**Locating and counting residue**
- Locating: only the orbit-chain item is LOCATED. Under the hypothesis that the orbit is revealed, it places the next revealed copy in (p_k, p_k#] via Bertrand. It stands as arithmetic and moves to the RANGE and chain(S_1) consequences.
- Counting: the Haar and dimension bracket is excluded from what stands. The histograms, the depth tree, the 37 strikers and the run distribution are measurements with exact ranges, not results.

**Scripts**
- Worker: scratchpad/tower_limit/: a_structure.py, b_periodic.py, b2_twoadic.py (l ≤ 4 finished), c_integers.py, c_small.py, d_strikers.py, d2_list.py, e_revealed.py, f_misc.py.
- Adjudication: scratchpad/adj_limit/: s1_tower.py, s2_depth.py, s3_padic.py, s4_strikers.py, s5_revealed.py.

### 2C. Re-check run by this assembly

Scripts are in `C:/Users/Alex/AppData/Local/Temp/claude/C--dev-primes/a1c3a0ad-3acd-4fbe-b091-1e36256528ec/scratchpad/record_0925/`, with outputs in `*_out.txt`. All were run in the foreground with `uv run --directory C:/dev/primes`, from raw divisibility and primality: gmpy2 Miller–Rabin with 40 rounds (30 in c3), and sympy BPSW for h0 and for the levels of the 14 runs.

| Script | Time | Result |
|---|---|---|
| a_anchors.py | 2.5 s | S_1 to 2·10⁵ (328 = 233 + 95; first six 59, 149, 179, 6269, 7949, 9419), 327 pair forms, C7, anchors, least tails, v(m_r), Anc, top-anchor pattern, struck counts at r = 7..61, exhaustive tail counts at r = 7..23, synthetic h0. **0 mismatches against the record.** |
| b_bases.py | 0.5 s | Offsets in bases 2, 6, 30; rows 3 and 5; base-6 anchor rows to r = 67; RANGE₆ witnesses; fixed-row kills for f ≤ 397; transport classes h = 7..43 below 10⁸. All as stated. |
| c1_tower.py | 0.7 s | P^m counts m = 1..5 (8, 32, 128, 512, 2048; written form 14, …, 3584); 4-to-1 fibres; next-digit law (1800 tests, 0 failures); 4^k reps with distinct words for k ≤ 4; mixed fixed points (deltas 4, 8, 4, 8); named J values; 2-adic facts mod 2⁸, 2¹² and 2¹⁶. All as stated. |
| c2_depth.py | 19.4 s | Delta census below 10⁷ and the full 4^10 tree. Histogram, depth maxima and exits as stated. The tree reps below 10⁷ equal the exact census. |
| c3_runs.py | 0.9 s | Runs {0: 262714, 1: 3937, 2: 14} + column 1, with the 14 and the four case-19 second steps; criterion on x ∈ [15, 40000] (10,662 columns, 0 mismatches); service intervals. All as stated. |
| d_zh.py | — | Z_h = ∅ for 107 primes in 7..3000 (first ten as stated); Z_7 = {0}; the exception sets {29, 31} and {11, 13, 67, 79}. |

Notes:
- One line of c1_tower.py printed False on the 2-cycle discriminant. That was a wrong constant in my own check (64 in place of 256). The discriminant is −2816 = 256·(−11), which gives the roots −15 ± 8√−11 as stated.
- Why P^k is all case 1 for k ≥ 2 (a one-line reason noted by this assembly): for odd y, y² + 11 ≡ 4 (mod 8), so J19(y) is even and never a column.
- Not re-run here: the Lean builds (per the brief); the dynatomic factorisations; K_h and the persistence lists; the 525-case transport grid (a lighter scan was run instead); the cycle data at 2²⁰.

---

## Part 3 — Map

**File.** `C:/dev/primes/research/proof/range_line_map.md`, 785 lines on disk, untracked (not committed).
- It covers nodes R5.f.xxxv.c.xxv to c.xxxi, the four round records and the four-module kernel.
- Every item is tagged with its source and a status word: PROVED, Stands, LEAN, MEASURED, REFUTED or OPEN.
- Sections: 1 terms and the range statement; 2 exact equivalent forms; 3 proved laws; 4 kernel table; 5 refuted statements; 6 open questions; 7 where the records are.

**Checker.** The checker read all 777 lines of the draft against:
- the four round records;
- theory_tree.md, lines 4702–5053, and the log at 6637–6643;
- the four Lean files and lakefile.toml.

Every long number table it compared matched its source.

Findings: its list has 27 entries, although its summary says 26. No sentence judges prospects.

Five substantive findings:
- Line 680 treated "the hand-off sequence is infinite" as equivalent to "infinitely many revealed c_g". No source gives that equivalence.
- Line 239 used J_q − J_p in the first-strike sense, which clashes with the map's J_g = ⌈(g² + 1)/30⌉.
- Line 408 said gears "strike nothing" below ⌈(g² − 1)/30⌉. The record says "removes".
- V1 and V2 had the same wording, with neither orbit defined.
- Lines 44 and 679 equated RANGE with the Lean RangeStatement, which also admits twin pairs off the copies.

The other 22:
- The added "live anchor" definition contradicted line 161.
- PROVED labels on items the records only check: the below-square count, A11 and A14.
- Line 285 contradicted itself, and line 442 dropped the "non-inert" condition.
- Undefined terms: λ, p_i and shelf i, N(D), τ, n_A and n_B, f_C, binding machine, ratio, budgets, G_i, raw and live copies, Path B, S_{≤N}.
- Two rows sat in the refuted table without instances: the tier framing and the "exactly at" list.
- One wrong citation (line 107) and two additions not in a source (lines 102 and 227).
- The lakefile registration line (508) no longer matched disk.

**Fixes.** All 27 were fixed in the map, which is the only file changed; each fix was checked against its named source. Line numbers are for the edited file.

Items 1–20, from the fix report:
1. The hand-off item is split into open items (a) and (b), with (a) ⇒ (b) (lines 687–689).
2. F_g = min(a_g, g − a_g) is defined, and the separations read F_q − F_p (239–240).
3. Line 410 now reads "removes (acts on) nothing below ⌈(g² − 1)/30⌉".
4. V1 names F mod M_X and V2 names the machine orbit mod M_q (430–431, 659, 744–745).
5. RANGE and the Lean form are distinguished (44, 686, 558).
6. The live-anchor reading is taken from kernel 2.1 item 7 (125).
7. The below-square count is its own item, labelled Stands (219).
8. A11 and A14 are in a Stands block (449–451).
9. The lattice-count exception at (7, case 1) is stated (286). The same repair was made at 441.
10. Symmetry is restricted to non-inert strikers, with the lattice exception (445).
11. d0, h* and λ(g) = h* are stated before the exceptions (334–335).
12. g_i and shelf i are defined (376).
13. N(D), τ, n_A and n_B are defined (443, 444, 447).
14. f_C is defined (459).
15. The Path B claim is stated (568).
16. The tier framing is moved to a Withdrawn note (571).
17. The "exactly at" list is moved to a scope note (679).
18. A_g is taken for the case of g and B' for the case of g' (102).
19. The citation is changed to [derived 2.3; C3] (107).
20. The lakefile wording is now "near the end, followed by RangeMissedReveal, RangeDerived, RangeChain" (513, 779).

Items 21–27: the fix report was cut off after item 20, so this assembly checked them on disk.
- Line 114: S_{≤N} is flagged as the map's reading.
- Line 178: the record's "binding machine" is flagged as undefined in the record.
- Lines 179 and 192: the "tightest ratio" and "maximum ratio" are flagged as quantities the record does not state.
- Line 191: the budget percentages are dropped.
- Line 227: the distinctness addition is removed ("… the upper leg of some copy 1 ≤ j < g").
- Line 413: raw and live copies are defined in the supply-law sentence.
- Line 731: G_i and ALL-KILLED are explained as the record's terms, and E5's equivalence is given as a result.

**Not yet in the map.** The map covers c.xxv–c.xxxi, so this round's results are not in it:
- The three new modules' theorems. They are named only in the registration lines 513 and 779; the kernel table (section 4) lists four modules.
- The anchors at r = 47..61. Line 694 still reads "Only r = 11..43 were measured".
- The base-6 rule-level refutation and the fixed-row criterion.
- The tower-limit results and the repaired P^(k+1) form.