# Record: ONE MACHINE, ONE PERIOD (maths angles), assembled 2026-09-25

## PART 1: KERNEL

**Shared by all four modules**
- **Location:** namespace `RangeLine`, files in `C:/dev/primes/proofs/`.
- **Registration:** each module has a `[[lean_lib]]` block at the end of `C:/dev/primes/proofs/lakefile.toml`. The last four entries are RangeCopies, RangeLocator, RangeMissed and RangeHandoff. None is in defaultTargets.
- **Git:** the four .lean files are untracked and lakefile.toml is modified. Nothing was committed.
- **Reviewer's rerun of every gate:**
  - `lake build <Module>` succeeds for all four.
  - A direct `lake env lean <Module>.lean` exits 0 with no output, so 0 errors and 0 warnings.
  - Searches for `sorry|admit|axiom` and for `native_decide|unsafe|opaque|implemented_by|set_option|extern|macro|elab` find nothing.
  - Every declaration uses only axioms from {propext, Classical.choice, Quot.sound}.
  - All four modules import together with no name clashes.
  - It found no false statement, no vacuous hypothesis and no off-by-one, so no requested statement needed correcting.
- **This assembly:** Lean was not rebuilt. A grep of the four files for `sorry|admit|axiom|native_decide` gives 0 hits in each, and no TmpAxioms* file is left.

### 1.1 RangeCopies (`C:/dev/primes/proofs/RangeCopies.lean`)

**Theorems**

(a)
- `copy_product (j : ℕ) : (30 * j - 1) * (30 * j + 1) = 900 * j ^ 2 - 1`
- `copy_strike_iff (g j : ℕ) : g ∣ (30 * j - 1) * (30 * j + 1) ↔ g ∣ 900 * j ^ 2 - 1`
- `copy_strike_iff_legs {g j : ℕ} (hg : g.Prime) : g ∣ 900 * j ^ 2 - 1 ↔ g ∣ 30 * j - 1 ∨ g ∣ 30 * j + 1`

(b)
- `minus_leg_iff_mod {g j : ℕ} (hg : 2 ≤ g) (hj : 1 ≤ j) : g ∣ 30 * j - 1 ↔ 30 * j % g = 1`
- `plus_leg_iff_mod {g j : ℕ} (hg : 2 ≤ g) : g ∣ 30 * j + 1 ↔ 30 * j % g = g - 1`
- `copy_strike_class {g j : ℕ} (hg : 2 ≤ g) (hj : 1 ≤ j) : (g ∣ 30 * j - 1 ∨ g ∣ 30 * j + 1) ↔ (30 * j % g = 1 ∨ 30 * j % g = g - 1)`

The four leg rules (each is an iff; all take `hg : g.Prime`, `hg7 : 7 ≤ g`):
- `leg_minus_minus (hj : 1 ≤ j) (h : g ∣ 30 * j - 1) : g ∣ 30 * (j + D) - 1 ↔ g ∣ D`
- `leg_plus_plus (h : g ∣ 30 * j + 1) : g ∣ 30 * (j + D) + 1 ↔ g ∣ D`
- `leg_minus_plus (hj : 1 ≤ j) (h : g ∣ 30 * j - 1) : g ∣ 30 * (j + D) + 1 ↔ g ∣ 15 * D + 1`
- `leg_plus_minus (hD : 1 ≤ D) (h : g ∣ 30 * j + 1) : g ∣ 30 * (j + D) - 1 ↔ g ∣ 15 * D - 1`

(c)
- `copy_leg_rule {g j D : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g) (hj : 1 ≤ j) (h1 : g ∣ 30 * j - 1 ∨ g ∣ 30 * j + 1) (h2 : g ∣ 30 * (j + D) - 1 ∨ g ∣ 30 * (j + D) + 1) : g ∣ D ∨ g ∣ 15 * D - 1 ∨ g ∣ 15 * D + 1`

(d)
- `exists_leg_minus {g : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g) : ∃ j, 1 ≤ j ∧ j < g ∧ g ∣ 30 * j - 1`
- `exists_leg_plus {g : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g) : ∃ j, 1 ≤ j ∧ j < g ∧ g ∣ 30 * j + 1`
- `copy_leg_rule_converse {g D : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g) (h : g ∣ D ∨ g ∣ 15 * D - 1 ∨ g ∣ 15 * D + 1) : ∃ j, 1 ≤ j ∧ j < g ∧ (g ∣ 30 * j - 1 ∨ g ∣ 30 * j + 1) ∧ (g ∣ 30 * (j + D) - 1 ∨ g ∣ 30 * (j + D) + 1)`
- `copy_pair_iff {g D : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g) : (∃ j, 1 ≤ j ∧ (g ∣ 30 * j - 1 ∨ g ∣ 30 * j + 1) ∧ (g ∣ 30 * (j + D) - 1 ∨ g ∣ 30 * (j + D) + 1)) ↔ (g ∣ D ∨ g ∣ 15 * D - 1 ∨ g ∣ 15 * D + 1)`

Helper lemmas: `not_dvd_two`, `not_dvd_thirty`, `int_two_mul_iff`, `int_thirty_mul_iff`, `int_step_iff`.

**Status**
- **Build:** `lake build RangeCopies` printed "Build completed successfully (3010 jobs)". The first build gave two linter warnings (`haveI` should be `have`); both were fixed and the rebuild was clean.
- **Sorry:** none.
- **Axioms:** `copy_product`, `copy_strike_iff`, `plus_leg_iff_mod` and `int_step_iff` use [propext, Quot.sound]. Every other declaration uses [propext, Classical.choice, Quot.sound].
- **Reviewer:** faithful, and stronger than requested in every part.
  - j ≥ 1 in (b) and (c) is genuinely needed, because the minus leg is 0 in ℕ at j = 0.
  - At D = 0 in (c), the conclusion g ∣ D holds, so dropping 0 < D leaves no gap.
  - All four case splits were checked: same-sign legs give g | D; minus of j with plus of j+D gives 2(15D+1); plus of j with minus of j+D, D ≥ 1, gives 2(15D−1).
  - "Strike" is plain divisibility, as the spec defines it, so a leg equal to g counts: g = 29 strikes copy 1.
- **Corrections:** none of the requested statements is false. Some hypotheses were dropped, and each drop makes the theorem stronger:
  - in (a), prime g, 7 ≤ g and j ≥ 1;
  - in (b), everything except 2 ≤ g and 1 ≤ j;
  - in (c), 0 < D.
- **(d) is proved in full,** with the extra bound 1 ≤ j < g. The witnesses are:
  - j = 30⁻¹ mod g when g | D or g | 15D+1;
  - j = g − (30⁻¹ mod g) when g | 15D−1.

### 1.2 RangeLocator (`C:/dev/primes/proofs/RangeLocator.lean`)

A copy j is struck by g when g ∣ (30j−1)(30j+1).

**Theorems**
- (a) `class_meets_every_residue {g N : ℕ} (hg : g.Prime) (hgN : ¬ g ∣ N) (r t : ℕ) : ∃ k < g, (r + k * N) % g = t % g`
- (a, no residue is hit twice) `class_residues_distinct {g N : ℕ} (hg : g.Prime) (hgN : ¬ g ∣ N) (r : ℕ) {k k' : ℕ} (hk : k < g) (hk' : k' < g) (h : (r + k * N) % g = (r + k' * N) % g) : k = k'`
- (helper) `thirty_ne_zero {g : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g) : ((30 : ℕ) : ZMod g) ≠ 0`
- (b, sharp form) `locator_closure_minus {g N : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g) (hgN : ¬ g ∣ N) (r : ℕ) : ∃ k < g, 1 ≤ r + k * N ∧ g ∣ 30 * (r + k * N) - 1`
- (b) `locator_closure {g N : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g) (hgN : ¬ g ∣ N) (r : ℕ) : ∃ k < g, 1 ≤ r + k * N ∧ g ∣ (30 * (r + k * N) - 1) * (30 * (r + k * N) + 1)`
- (b, contrapositive) `locator_closure_contra {g N r : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g) (hclass : ∀ j, 1 ≤ j → j ≡ r [MOD N] → ¬ g ∣ (30 * j - 1) * (30 * j + 1)) : g ∣ N`
- (c) `locator_modulus {N r X : ℕ} (hclass : ∀ j, 1 ≤ j → j ≡ r [MOD N] → ∀ g, g.Prime → 7 ≤ g → g ≤ X → ¬ g ∣ (30 * j - 1) * (30 * j + 1)) : ∀ g, g.Prime → 7 ≤ g → g ≤ X → g ∣ N`
- (c, product form) `locator_modulus_prod {N r X : ℕ} (hclass : …same…) : (∏ p ∈ (Finset.Icc 7 X).filter Nat.Prime, p) ∣ N`
- (c, size form) `locator_modulus_le {N r X : ℕ} (hN : 1 ≤ N) (hclass : …same…) : (∏ p ∈ (Finset.Icc 7 X).filter Nat.Prime, p) ≤ N`

**Status**
- **Build:** `lake build RangeLocator` printed "Build completed successfully (3010 jobs)". `lake env lean RangeLocator.lean` exited 0 with no errors or warnings.
- **Sorry:** none.
- **Axioms:** all 9 declarations use [propext, Classical.choice, Quot.sound].
- **Reviewer:** faithful, and stronger than requested.
  - N ≥ 1 and r ≥ 1 are redundant: ¬ g ∣ N already rules out N = 0, and 30j ≡ 1 (mod g) forces j ≥ 1.
  - The direction of `j ≡ r [MOD N]` was checked against `Nat.add_mul_mod_self_right`.
  - The class hypothesis can actually be met. For g = N = 7, r = 0, X = 7, the reviewer type-checked that no copy j ≡ 0 (mod 7) is struck by 7, and `locator_closure_contra` then gives 7 ∣ 7.
  - N = 0 is a harmless degenerate case: the conclusion g ∣ 0 is trivially true.
- **Corrections:** none false. N ≥ 1 and r ≥ 1 were dropped, and j ≥ 1 is moved into the conclusion. Extras beyond the request: `class_residues_distinct`, `locator_closure_contra`, the product form and the size form. The resumed run verified the file and its lakefile block without editing either.

### 1.3 RangeMissed (`C:/dev/primes/proofs/RangeMissed.lean`)

**Theorems**
- `prime_ge7_mod (g : ℕ) (hg : g.Prime) (h7 : 7 ≤ g) : g % 2 = 1 ∧ g % 3 ≠ 0 ∧ g % 5 ≠ 0`
- `sq_mod30_cases (g : ℕ) (hg : g.Prime) (h7 : 7 ≤ g) : g ^ 2 % 30 = 1 ∨ g ^ 2 % 30 = 19`
- `sq_mod30_exactly_one (g : ℕ) (hg : g.Prime) (h7 : 7 ≤ g) : (g ^ 2 % 30 = 1 ∧ ¬ g ^ 2 % 30 = 19) ∨ (g ^ 2 % 30 = 19 ∧ ¬ g ^ 2 % 30 = 1)`
- `missed_copy_legs_case1 (g : ℕ) (h1 : g ^ 2 % 30 = 1) : 30 ∣ g ^ 2 + 29 ∧ 30 * ((g ^ 2 + 29) / 30) - 1 = g ^ 2 + 28 ∧ 30 * ((g ^ 2 + 29) / 30) + 1 = g ^ 2 + 30`
- `missed_copy_legs_case19 (g : ℕ) (h19 : g ^ 2 % 30 = 19) : 30 ∣ g ^ 2 + 11 ∧ 30 * ((g ^ 2 + 11) / 30) - 1 = g ^ 2 + 10 ∧ 30 * ((g ^ 2 + 11) / 30) + 1 = g ^ 2 + 12`
- `missed_copy_legs (g : ℕ)` is the conjunction of the two implications above.
- `dvd_of_dvd_sq_add (g c : ℕ) (h : g ∣ g ^ 2 + c) : g ∣ c`
- `gear_misses_own_copy (g : ℕ) (hg : g.Prime) (h7 : 7 ≤ g) : (g ^ 2 % 30 = 1 → ¬ g ∣ g ^ 2 + 28 ∧ ¬ g ∣ g ^ 2 + 30) ∧ (g ^ 2 % 30 = 19 → ¬ g ∣ g ^ 2 + 10 ∧ ¬ g ∣ g ^ 2 + 12)`
- `larger_gear_sq (g h : ℕ) (hg : g.Prime) (hh : h.Prime) (h7 : 7 ≤ g) (hgh : g < h) : g ^ 2 + 30 < h ^ 2`
- `no_larger_gear_acts (g h : ℕ) (hg : g.Prime) (hh : h.Prime) (h7 : 7 ≤ g) (hgh : g < h) : g ^ 2 + 30 < h ^ 2 ∧ ∀ L, h ∣ L → L ≤ g ^ 2 + 30 → h < L → L / h < h`

**Status**
- **Build:** `lake build RangeMissed` printed "Build completed successfully (3010 jobs)" with 0 errors and 0 warnings. The unused norm_num alternatives, the deprecated `Xor'` and the deprecated `push_neg` were removed before this build.
- **Sorry:** none.
- **Axioms:** `prime_ge7_mod` uses [propext, Quot.sound]. `dvd_of_dvd_sq_add` uses [propext]. All others use [propext, Classical.choice, Quot.sound].
- **Reviewer:** faithful. Check instance: g = 7 is case 19, J = 2, legs 59 and 61. Minor points, none a gap:
  - In `no_larger_gear_acts` the hypothesis h < L is unused, because L/h < h already follows from L ≤ g²+30 < h².
  - J ≥ 1 is not stated, but it is immediate.
  - Parts (a) to (d) are separate theorems; no single theorem assembles the missed-copy law, and none was requested.
- **Corrections:** none. "Exactly one" is written as an explicit disjunction because `Xor'` is deprecated. The case-1 and case-19 leg identities need only the value of g² % 30, not primality. Resume note: the file already existed; the resumed run appended the missing lakefile block and cleaned the warnings.

### 1.4 RangeHandoff (`C:/dev/primes/proofs/RangeHandoff.lean`)

**Definition**
- `def RangeStatement (q : ℕ) : Prop := ∃ p, q < p ∧ p + 2 ≤ primorial q ∧ p.Prime ∧ (p + 2).Prime`
- `primorial` is Mathlib's root-level primorial, the product of the primes ≤ q.

**Theorems**
- (a) `range_implies_unbounded (h : ∀ q : ℕ, q.Prime → 7 ≤ q → RangeStatement q) : ∀ N : ℕ, ∃ p, N < p ∧ p.Prime ∧ (p + 2).Prime`
- (b) `twin_serves {p q : ℕ} (hp : p.Prime) (hp2 : (p + 2).Prime) (hqp : q < p) (hle : p + 2 ≤ primorial q) : RangeStatement q`
- (c, form without the extra hypotheses) `serves_interval_of_le {p q₁ q₂ : ℕ} (hp : p.Prime) (hp2 : (p + 2).Prime) (hle : p + 2 ≤ primorial q₁) (hlt : q₂ < p) : ∀ q : ℕ, q₁ ≤ q → q ≤ q₂ → RangeStatement q`
- (c) `serves_interval {p q₁ q₂ : ℕ} (hp : p.Prime) (hp2 : (p + 2).Prime) (_hq₁ : q₁.Prime) (_hq₂ : q₂.Prime) (_h12 : q₁ ≤ q₂) (hle : p + 2 ≤ primorial q₁) (hlt : q₂ < p) : ∀ q : ℕ, q₁ ≤ q → q ≤ q₂ → RangeStatement q`
- `primorial_seven : primorial 7 = 210` (proved by kernel `decide`, not `native_decide`)
- (d) `small_cases : ∀ q : ℕ, 7 ≤ q → q ≤ 23 → RangeStatement q`, with witness p = 29
- `small_case_7`, `small_case_11`, `small_case_13`, `small_case_17 : RangeStatement _`

**Status**
- **Build:** `lake build RangeHandoff` printed "Build completed successfully (3018 jobs)" with 0 errors and 0 warnings. The first attempt left the goal 29 + 2 ≤ 210 open; adding `; norm_num` fixed it.
- **Sorry:** none.
- **Axioms:** all 10 declarations use [propext, Classical.choice, Quot.sound].
- **Reviewer:** faithful.
  - `RangeStatement` prints as the requested definition.
  - (a) is an implication whose hypothesis is the open range statement; this is intended, and the hypothesis is not vacuous in the Lean sense.
  - `serves_interval` carries three unused hypotheses (q₁ prime, q₂ prime, q₁ ≤ q₂) to match the requested wording; `serves_interval_of_le` is the strictly stronger form.
  - The same witness p = 29 would work up to q = 28.
  - There is no off-by-one: q < p is strict.
- **Corrections:** none.

## PART 2: MATHS

### 2.1 Does RANGE force the missed-copy chain

**Notation**
- Rev is the set of revealed lower legs. S_1 is the set of revealed missed copies.
- σ(y) is the least prime whose primorial is ≥ y.
- m_r = max{m ∈ S_1 : m + 2 ≤ r#}.
- t_r(S) = max{s ∈ S : s + 2 ≤ r#}, and x⁺ is the successor of x in the set.
- v(m) = max{p ∈ Rev : p < m⁺}.
- tail(r) = Rev ∩ (m_r, r# − 2].

**Formulas**
- Domination: service(M) ⊇ service(R) ⟺ p_R ≤ p_M ≤ σ(p_R+2)# − 2.
- Non-dominated set: N = ⋃_r tail(r). RANGE ⟺ chain(S_1 ∪ N).
- Crossing form: chain(S) ⟺ s_0 + 2 ≤ 210 and t_r(S)⁺ + 2 ≤ t_r(S)# for every prime r ≥ 7.
- Two-copy cover, for R ∈ N with r = σ(p_R+2): service(R) ⊆ ⋃_{S_1} service ⟺ service(R) ⊆ service(m_r) ∪ service(m_r⁺) ⟺ r ≤ m_r and m_r⁺ + 2 ≤ m_r#.
- Given RANGE: chain(S_1) ⟺ m_r⁺ + 2 ≤ m_r# at every r with tail(r) ≠ ∅.

**Standing**
1. **Domination criterion. PROVED.** service(R) is the primes in [σ(p_R+2), p_R), and σ is monotone. Check: all revealed R < 5000 against revealed M < 150000, 21,918 pairs, 0 mismatches.
2. **Region and c_{g'}. PROVED.** A copy j in region g = [J_g, sh(g′)), and also c_{g′}, is revealed ⟺ no prime in 7..g divides either leg. Case-1 heads are never revealed, because a head's upper leg is g². Checks:
   - gears 7..3001: 302,422 copies, 0 mismatches;
   - tiling of j in [2, 2·10⁶]: the only uncovered j are the 484 case-1 heads.
3. **Direction lemma. PROVED.** Let R be revealed at position ≥ 2 of region g.
   - Neither c_g nor any missed copy with lower leg below p_R dominates R.
   - c_{g′} dominates R ⟺ c_{g′} is revealed and g′² + B′ ≤ σ(p_R+2)#.
   - The bracket fails only when a primorial lies in [p_R+2, g′²+B′). These intervals are disjoint across regions, so each primorial fails the bracket in at most one region.
   - Measured failures: region 709 (17#), region 3109 (19#), region 14929 (23#). There are none at 11# or 13#.
4. **Non-dominated set. PROVED.** N = ⋃_r tail(r). The map φ(R) = least S_1 element ≥ p_R preserves service exactly off N.
   - Least tail copy by r: 7 none; 11: 239; 13: 17489; 17: 293999; 19: 9542369; 23: 197994299; 29: 6423863099; 31: 199288140509; 37: 7399721980769; 41: 303863920848119; 43: 13082663386240769.
   - "t_r is not a missed copy" is a MEASUREMENT for r = 11..43 only.
5. **Two-copy cover law. PROVED.** The hand-off m_r⁺ + 2 ≤ m_r# already forces m_r > r.
6. **Crossing form. PROVED** for sets of primes ≥ 29; {5, 43, 83} is outside the scope. Crossing pairs (t_r, t_r⁺):
   - 7: (179, 239)
   - 11: (2129, 2309)
   - 13: (29879, 30089)
   - 17: (510449, 511109)
   - 19: (9699509, 9701819)
   - 23: (223091549, 223095149)
   - 29: (6469693079, 6469694039)
   - 31: (200560489859, 200560490549)
   - 37: (7420738130759, 7420738140689)
   - 41: (304250263523909, 304250263527479)
   - 43: (13082761331668079, 13082761331670209)

   t_r is a missed copy only at r = 7, and t_r⁺ ≠ m_r⁺ for every r = 7..43 (measured).
7. **RANGE forces the S_1 hand-off at every non-live anchor. PROVED.** Given RANGE, chain(S_1) ⟺ m_r⁺ + 2 ≤ m_r# at every r with a non-empty tail; a missing successor counts as false. Anchors m_r → m_r⁺ (gears in brackets):
   - r = 7 and 11: 179 → 6269 (13 → 79)
   - 13: 17189 → 49739 (131 → 223)
   - 17: 292709 → 546149 (541 → 739)
   - 19: 9541949 → 11498909 (3089 → 3391)
   - 23: 197993069 → 247464389 (14071 → 15731)
   - 29: 6423862229 → 6509746499 (80149 → 80683)
   - 31: 199288137899 → 200812430669 (446417 → 448121)
   - 37: 7399721979059 → 7427625284699 (2720243 → 2725367)
   - 41: 303863920846259 → 304251655581749 (17431693 → 17442811)
   - 43: 13082663386239869 → 13083062802401099 (114379471 → 114381217)

   tail(7) is empty and tail(r) is non-empty for r = 11..43, so every anchor for r = 7..43 is live (179 through r = 11). Within J ≤ 10⁷, the live pairs are exactly the 5 straddling pairs.
8. **Repaired converse-failure form. PROVED.** RANGE ∧ ¬chain(S_1) ⟺ RANGE ∧ (S_1 finite ∨ ∃ consecutive (m, m⁺) in S_1 with m < σ(m⁺+2) ≤ v(m)). An equivalent form replaces the second disjunct by m⁺ + 2 > m#.
   - v(m_r) for r = 7..43: 6089, 6089, 49529, 545789, 11498159, 247464209, 6509744609, 200812422119, 7427625283019, 304251655578329, 13083062802399299.
9. **Witnesses at a live anchor. PARTIAL.** At a live anchor, t_r witnesses every q in [m_r, t_r).
   - The only RANGE chain inequality with m_r⁺ as successor is σ(m_r⁺+2) ≤ v(m_r), and v(m_r) > m_r.
   - So the S_1 hand-off σ(m_r⁺+2) ≤ m_r is not one of RANGE's chain inequalities.

**Refuted**
- **The reveal status of a copy at position ≥ 2 does not determine c_g or c_{g′}.** The smallest instance of each (c_g revealed?, c_{g′} revealed?) combination, j ≤ 10⁷:
  - (no, no): copy 14, legs (419, 421), g = 19, position 2; c_19 = (389, 391 = 17·23), c_23 = (539 = 7²·11, 541).
  - (no, yes): copy 184, legs (5519, 5521), g = 73, position 7; c_73 = (5339 = 19·281, 5341 = 7²·109), c_79 = (6269, 6271).
  - (yes, no): copy 8, legs (239, 241), g = 13, position 3; c_13 = (179, 181), c_17 = (299 = 13·23, 301 = 7·43).
  - (yes, yes): copy 267, legs (8009, 8011), g = 89, position 3; c_89 = (7949, 7951), c_97 = (9419, 9421).
- **Comb sign, fixed.** Take D = J_{g′} − j. If h divides the lower leg of c_{g′}, copy j is struck by h exactly when D ≡ 0 or +15⁻¹ (mod h). If h divides the upper leg, the rule is D ≡ 0 or −15⁻¹. (PROVED.)
- **No service-preserving map Rev → S_1 exists.** Copy 8 = (239, 241) has service [11, 239). There is no S_1 element in [239, 2308], since S_1 begins 59, 149, 179, 6269. Copy 8 is covered by c_13 = [7, 179) and c_79 = [13, 6269) together, and by neither alone.
- **"The witnesses are tail copies" is false.** At q = m_11 = 179, the missed copy c_79 = (6269, 6271) is a witness.

**Not established or unfinished**
- **The converse RANGE ⇒ chain(S_1) is open.** It is reduced to the S_1 hand-offs at live anchors. Whether RANGE excludes a finite S_1 is also open, and it is implied by the converse.
- **No structural bridge found.** No statement produces a revealed missed copy in (r# − 2, m_r# − 2] from revealed tail or head copies.
- **Independence rests on instances only.** That a revealed region copy is independent of the status of c_g and c_{g′} rests on the four instances above, not on a residue-level theorem. The fixed-K version (via D8 and Shiu) was not attempted.
- **Empty tails beyond r = 43 are undecided.** Only r = 11..43 were measured, and none has an empty tail.
- **Range of the raw checks.** The same-machine lemma was raw-checked for gears ≤ 883 only. The bracket tables come from the copy sieve (j ≤ 10⁷) for r ≤ 23 and from targeted trial division for r = 7..43; the two methods agree at r = 7..23.

**Locating and counting**
- **Locating:** none in what stands. The tails appear only as an exact set identity and as a live-or-not case condition. The tables of m_r, t_r, t_r⁺, v(m_r) and the least tail copy are exact positions, not bounds.
- **Counting:** none as a result. The figures 31,097 / 267,311, 21,918, 302,422, 388,323 and 72 (67 inside a bracket / 5 straddling) are sizes of checks.

**Scripts**
- In `scratchpad/converse_0925/`: `s1_region_missed.py` (1.5 s) and `s2_big_brackets.py` (about 4 s).
- Adjudicator scripts: `adj_small.py`, `adj_big.py`, `adj_check2.py`.

### 2.2 Doomed escape classes

**Formulas**
- Moduli: M_P = 30·∏_{7≤h≤P} h. The children of class C (least residue r_C) at row h = P⁺ are r_C + k·M_P for k = 0..h−1.
- Child types:
  - one zero child, k = −r_C·M_P⁻¹ mod h. It can hold only the gear h, which lies in Cap(P);
  - e_h struck children, k = (ε − r_C)·M_P⁻¹ mod h for ε ∈ E_h^k;
  - h−1−e_h escaping children.
- Strike-class counts: e_h^(1) = 2 + (−7/h) + (−30/h), with e_7^(1) = 0, and e_h^(19) = 2 + (−10/h) + (−3/h).
- Descendants: Desc_C(P′) = {n < M_{P′} : n ≡ r_C (mod M_P), and n mod h ∉ {0} ∪ E_h for every P < h ≤ P′}. Its size is ∏(h−1−e_h), the same for every class of the case (an exact class count, flagged).
- Window (P′ ≥ 23): let x = P′ + d_{P′}(D). Then D meets Cap(P′) iff all three hold:
  - x ≡ D (mod M_{P′});
  - x ≤ max(37, 3P′/2);
  - every prime h in (P′, x) has e_h = 0 or (x−h)² < 2h − B′, with B′ = 30 in case 1 and 10 in case 19.

  D then meets Cap(P′) in {x}, x is prime, and D escapes iff c_x is revealed.
- Doom: C is doomed ⟺ C ∩ G = ∅ ⟺ every prime g ≡ r_C (mod M_P) has g²+A or g²+B composite.
  - Some class is doomed ⟺ some class holds only finitely many revealed gears.
  - No class is doomed ⟺ G_{>P} → Esc(P) is onto for every P.

**Standing**
- **T1. PROVED.** The children take each residue mod h once, split 1 zero / e_h struck / h−1−e_h escaping. A struck child's gears are > h, and h strikes their c_g. The e_h formula holds with the single exception e_7^(1) = 0.
  - Correction to the worker's check sentence: 17 is in no case-19 escape class from level 7 on. 19 is in no level-17 class, but 19 mod 30030 is in Esc_1(13).
  - Class sizes: |Esc| = 24/144/1440/20160/362880/6531840 in case 1 and 8/64/512/8192/114688/2293760 in case 19, for levels 7..23.
- **T2. PROVED.** Escaping descendants are the residues avoiding {0} ∪ E_h on (P, P′]; the set is non-empty, and only non-emptiness is used downstream.
- **W. PROVED** from B2 and B5(d). W2 is redundant given W1 and W3. Measurements:
  - Cap(7..29) = {11, 13}, {13}, {17, 23}, {19, 23}, {23}, {29, 31, 37}, {31, 37}.
  - λ(g) > 2g/3 for 41 ≤ g ≤ 10⁷.
  - The formula exceptions are g = 7, 11, 13, 23 and 37, with λ = 0, 0, 7, 13 and 23.
- **D1. PROVED.** The following are equivalent:
  - C is doomed;
  - C holds no revealed gear;
  - every member n > P leaves the tree at a row in (P, λ(n)] (for composite n this row is lpf(n));
  - every prime g ≡ r_C (mod M_P) has a composite leg.

  A prime member lies in Cap(P′) at every prime P′ in [max(P, λ(n)), n) and has one outcome there. A member chain ends at a zero child or at a struck child, with no third ending.
- **D2. PROVED.** Non-doom is inherited upward and doom downward. Every class holds a revealed gear at levels 13 (1,952 classes) and 17 (28,352 classes). The largest least witnesses are 21,974,021 and 1,018,572,719; these are measurements.
- **D3. PROVED.** x(x²+A)(x²+B) has no fixed prime divisor on any escape class, and both leg polynomials are irreducible. Esc_k(P) is exactly the set of admissible case-k classes.
- **D4. PROVED** (imports Dirichlet). For every finite row set H there is a subclass free of H that holds infinitely many gears. So doom has no finite-row certificate, and the rows striking a doomed class are unbounded.
- **D5. PROVED.** If the acting bound is removed, every class is doomed and the tree is unchanged, so the tree (as sets) does not determine doom. "Doom depends only on the acting bound" is withdrawn.
- **D6. PARTIAL.** A shape-only model up to 2·10⁶ (44,520 pairs) strikes every gear in 13..2·10⁶ before its cap. So shape data cannot place a revealed gear in (11, 2·10⁶]. By König, an infinite model exists iff V_X ≠ ∅ for every X; this is the Hall-type supply statement, and it is not proved.
- **R. PROVED.** No class is doomed ⟺ every Esc_k(P) class contains a prime g with g²+A_k and g²+B_k both prime ⟺ every class contains infinitely many such g.

**Refuted**
- **W, last clause:** "Only fixed-least-residue chains ever meet a window; non-integer infinite paths never do."
  - Instance (case 19): 13 mod 2310 → 4633 → 64693 → 1085713 → 10785403 → 233878273 → ….
  - 13 is in Cap(11), since λ(13) = 7.
  - What is true: a class meets a window at most at its least residue, and the meeting certifies only that member.

**Not established or unfinished**
- **"No class is doomed" is not proved, and no doomed class was found.** It reduces exactly to R. The simplest part, infinitely many primes g²+A with g prime, is not proved here.
- **"Non-integer paths meet windows only finitely often" is unresolved.** It would imply that some class is doomed.
- **The infinite shape model is not proved.** It needs the Hall-type supply statement, which would be a counting argument.
- **Consistency with doom is unresolved.** Whether tree shape plus window are consistent with doom at all heights, and whether doom depends on E_h at a fixed acting bound, are both open.
- **B9 levels.** B9 was re-run at levels 7, 11, 13 and 17 only. Level 19 (477,568 classes) was not re-run and level 23 (8,825,600 classes) was not run.
- **The leaf question beyond level 17 was not examined:** is there a class whose only revealed gear is P⁺?
- **Runtime:** `d2_window.py` at N = 10⁷ took 319.8 s, over the 5-minute limit, and the harness moved it to the background. It was replaced by `d2b_window.py`, which ran in the foreground in 69.0 s with identical results.

**Locating and counting**
- **Locating:** the window form of W and the check heights (first-meet levels, least witnesses) are measurements only.
- **Counting:** the T1/T2 exact class counts are flagged and used only for non-emptiness. Dirichlet and B5(d) are imported.

**Scripts**
- In `scratchpad/doom_0925/`: `d1_tree.py`, `d2_window.py`, `d2b_window.py` (output in `d2b_out_1e7.txt`), `d3_model.py`, `d4_witness.py`.
- Adjudicator scripts: `scratchpad/adj_doom/a1_tree.py` to `a4_model.py`.

### 2.3 The derivation as an operator

**Formulas**
- Der_int(M)(h) = min{copy c of M : 30c − 1 > h²}, and Der_int ∘ Der_int = Der_int. Since m_O = J and m_D = id, Der_int(O) = Der_int(D) = D.
- Tower: P^k = (J^k)*O, with J(y) = (y²+c)/30 (c = 29 in case 1, c = 11 in case 19). Column y carries copy J^k(y).
- Columns:
  - P^1 is 8 classes mod 30.
  - For k ≥ 2, P^k is 2^(2k+1) classes mod 2·15^k, all case 1, and J maps them 4-to-1 onto P^(k−1).
  - P² = {y ≡ ±1, ±11 (mod 30); y ≢ ±4 (mod 9); y ≢ ±11 (mod 25)}, with κ′ = 1 iff y ≡ ±1, ±4 (mod 25).
  - The fixed columns are 1 and 29, because 30(J(y) − y) = (y−1)(y−29).
- Strike set: E^(k)_h = J_29^{−(k−1)}(J_c⁻¹{±30⁻¹}), with |J_c⁻¹(S)| = Σ_{s∈S}(1 + χ_h(30s − c)).
- Pair rule, for every k ≥ 1 and D ≠ 0: N(D) = Σ_{x∈E, x′∈E′} [h | D⁴ − 2(u+u′)D² + (u−u′)²], with u = 30x − c and u′ = 30x′ − c′; y = 0 and y = −D are dropped.
- Acting: g² ≤ 30j+1 (O) → h ≤ x (D) → h ≤ J^(k−1)(y) (P^k).
- Reach: X(g) = isqrt((g²+A_g)# − 41).
- Serving: [σ(p+2), p) on legs, at every level.

**Standing**
1. **Der_int(O) = D. PROVED.** For every lattice h, the least j with 30j−1 > h² is J(h).
2. **Der_int(D) = D. PROVED.** The class counts, the pair rule A4 and diagonal acting carry over unchanged.
3. **General form, repaired. PROVED.** Der_int(M) = f_C with f_C(h) = min{c ∈ C : 30c−1 > h²}, and Der_int ∘ Der_int = Der_int for every machine; the increasing-legs hypothesis is not needed.
   - (i) The fixed points are exactly the f_C for unbounded copy families C.
   - (ii) f_C has the identity missed-column map ⟺ f_C is strictly increasing ⟺ every (prev(h)², h²] contains a lower leg of C.
   - D is the pointwise least fixed point, not the only one. C_S = O minus {J(h) : h ∈ S} satisfies (ii) iff S contains neither 1 nor 11.
   - D is the unique fixed point whose lower leg at every column h lies in (h², h²+30).
   - Der_int(P²) is a fixed point different from D: column 7 carries copy 6 there, against D's copy 2.
4. **P^k columns. PROVED.** P² is 32 classes mod 450 and P³ is 128 classes mod 6750, with every fibre of J of size 4. P⁴ (512 classes mod 101250) comes from the refuter's recomputation.
5. **Fixed columns. PROVED.** In case 1, 30(J(y) − y) = (y−1)(y−29). In case 19 the discriminant 856 is not a square. J(y) < y exactly at y = 7, 11, 13, 17, 19, 23.
6. **Level-2 per-leg rule. PROVED as stated.** Each leg contributes 0, 2 or {0, 4}, decided by χ(−T), then χ(841+900T), then χ(30t−29) with t² = −T.
   - Factorisations: 26041 is prime, 27841 = 11·2531, 9841 = 13·757, 11641 = 7·1663.
   - The worker's "exactly at" list covers h ≤ 3000 only; under the iff, (26041, κ′ = 1) also has a struck zero class.
7. **Class-count sequence. PROVED as stated.** The possible counts go {2} → {0,2,4} → {0,2,4,6,8} → {0,2,…,16}, always even and at most 2^(k+1). First primes for each level-3 count:
   - branch 1: {0:7, 2:13, 4:23, 6:11, 8:67, 10:277, 12:2633, 14:71479, 16:196081}
   - branch 19: {0:7, 2:11, 4:13, 6:53, 8:103, 10:37, 12:3847, 14:26263, 16:239383}
8. **Pair rule, level form. PROVED.** Levels: U^(1) = {±1 − c}, U^(k+1) = {30y − 29 : y² ∈ U^(k)}. Check: 17,872 cases at levels 1 and 2, 0 failures.
9. **Level-2 pair rule over Q.** For the same κ′:
   - N(D) = Σ_T ( [h | (D²+116)² + 14400T] + 2[χ(−T) = 1][h | (D²+58)² − 4(841+900T)] ) + 2ν(D).
   - ν(D) = [χ(−A′) = χ(−B′) = 1][h | M_{A′B′}(D)], except at coincidence primes.
   - All six mixed norms M_{A′B′} have degree 16 and are irreducible over Q.
10. **Refutation (ungated form) stands.** Instance: h = 7, κ′ = (1, 1), D = 2. The raw count is 0 but the form without the [χ(−T) = 1] gate gives 2.
11. **Refutation (indicator form) stands.** Instance: h = 53, κ′ 1 then 19, D = 3 and D = 50, (T, T′) = (28, 10). Here ν = 2 and the raw count is 2 at both D, while an indicator term is at most 1.
12. **Acting. PROVED.** For strikers h ≥ 7, h acts on column y of P^k iff h ≤ J^(k−1)(y), and a prime h = J^(k−1)(y) never strikes that column. The only exception, counting gear 5, is gear 5 at column 1 of every level, where it acts and strikes nothing.
    - For a general f_C, diagonality and "the column's own gear never strikes" both fail: in Der_int(P²), gear 113 strikes on columns 31..109 and on its own column 113 (copy 953, legs 28589 = 11·23·113 and 28591), and 6301 divides the lower leg 45171869 of copy 1505729 at column 6301.
13. **Separation identities. PROVED.** On P², y′² − y² = 30(J(y′) − J(y)) exactly, with τ = ε − c_κ′ (ε = +1 when the lower leg is struck, −1 when the upper leg is struck).
14. **Persistence. PROVED.** The sequence k → E^(k)_h is eventually periodic, and h strikes at every level iff a seed lies on a J_29-cycle mod h. The seeds are ±30⁻¹ in branch 1 and the y with y² ≡ −10 or −12 in branch 19.
    - Seed period 1, branch 1: {11, 13, 29, 31, 67, 79} (the primes of 29·31·869·871).
    - Seed period 1, branch 19: {11, 13, 23, 37, 853}.
    - Seed period 2, branch 1: {37, 22573, 76091}.
    - Seed period 2, branch 19: {103, 8287, 851689}.
    - Any 2-cycle needs (−11/h) = 1.
    - The single list {11, 13, 29, 31, 67, 79} is the branch-1 set only. Instance: h = 23, branch 19, seed 6, with J_29(6) ≡ 6 (mod 23).
15. **Reach. PROVED.** J(y′) ≤ X(g) with X(1) = 80434 and X(7) = 43849291330. The P² reach ends at y′ = 1549 for 29# and at y′ = 1146941 for 59#.
16. **Chain is upward-closed. PROVED** (chain read as an unbounded sequence in which every member has a successor), and S_k ⊆ S_(k−1). The members and serving intervals in the check are flagged measurements.
17. **Slack step. PROVED.** u_{i+2} < p_{i+1}·u_{i+1} implies Δ_{i+1} > Δ_i. A D-column g ≤ p_{i+1} with p_{i+1} > 15 gives u_{i+2} < p_{i+1}(p_{i+1}+2).
18. **Range as a power, repaired. PROVED.** The period of P^k is 15^(k−1)·q#. Every range column y of P^k satisfies y^(2^k) < 30^(2^k − 2)·q#.
    - log(top column)/log(period) → 2^(−k) holds only as a limit in q. At finite q the ratio is not 2^(−k); for example, at q = 101 it is 0.5000, 0.2612 and 0.1450 for k = 1, 2, 3.
19. **Summary, as repaired.** The orbit is O → D → D, D is the least fixed point, and acting stays diagonal along that orbit.

**Not established or unfinished**
1. **Class counts at levels k ≥ 4** were not computed. That every even value up to 2^(k+1) occurs at every level is not proved; it needs the Galois group of the J-tower. Only the top value 2^(k+1) is proved to occur, for infinitely many h.
2. **Pair rule at level ≥ 3** is given only in level form, not as Q-polynomials in D. The coincidence primes of the level-2 mixed norms (53, 631, 883 below 1000, all cross-κ′) are not characterised.
3. **Persistence closed forms** exist only for seed periods 1 and 2, and the lists stop at 2000.
4. **The converse arrows chain(S_(k−1)) ⇒ chain(S_k)** were not addressed. S_3 has no gear member with y ≤ 2·10⁵ (BPSW).
5. **Der_int(P²):** its class counts and pair rule have not been derived.
6. **The O-shaped range on D's own period** (a revealed x with q < x ≤ q#) was only named. Relating it to RANGE is a window question and was not pursued.
7. **Level-3 raw checks are limited:** strike sets for h ≤ 139 and the pair lemma for h ≤ 79.

**Locating and counting**
- Items (1) to (14) and (19) are free of both.
- Item (15) is an exact coordinate change. Its endpoints 1549 and 1,146,941 are window positions that appear only as check values.
- Item (16) runs down the P^k tower, which the working rules list under locating. Its members and tallies are flagged measurement.
- Item (18) is a size statement, so it falls in the excluded counting class.

**Scripts**
- In `scratchpad/adj_derop/`: `a_fixed.py`, `b_acting.py`, `c_periods.py`, `d_range.py`, `e_exponent.py`, `f_pairs.py`, `g_norms.py`.
- In `scratchpad/defend_der/`: `d1_fixed.py`, `d2_acting.py`, `d3_periods.py`.

### 2.4 Spot checks rerun during this assembly

All three scripts were rebuilt from raw divisibility and trial division and ran in the foreground, taking under 4 s in total. Nothing under `C:/dev/primes` was modified.

**`spot_a_converse.py`**
- Revealed-by-definition equals both-legs-prime for j ≤ 20000, 0 mismatches.
- S_1 begins 59 (7), 149 (11), 179 (13), 6269 (79), 7949 (89), 9419 (97). No c_g is revealed for g in 17..73.
- Domination criterion: 4,966 pairs, 0 mismatches.
- σ(241) = 11, and no S_1 element lies in [239, 2308].
- No revealed j ≥ 2 lies outside the region of its top acting gear (j ≤ 20000).
- The four combination instances (copies 14, 184, 8, 267) reproduce with the stated factorisations.
- For r = 7..23, the following all reproduce exactly: crossing pairs, anchors and gear crossings, v(m_r), least tail copies, and "t_r missed only at r = 7".
- v(179) = 6089, and sigma(6271) = 13.
- c_79 serves q = 179.

**`spot_b_doom_op.py`**
- The e_h formula matches raw unit roots for all 427 rows 7..2999 in both cases, 0 mismatches.
- |Esc| at levels 7, 11, 13 is 24/144/1440 (case 1) and 8/64/512 (case 19), and it equals the admissible set in every case.
- λ(7, 11, 13, 23, 37) = 0, 0, 7, 13, 23. Cap(7..29) reproduces the list above.
- No gear 41 ≤ g < 3000 has λ ≤ 2g/3.
- Escape at level λ(g) equals "c_g revealed" for gears 7..2999, 0 mismatches.
- The refuted path: each step is in Esc_19, is the least residue at its level, and is congruent to the previous step. 13 is in Cap(11).
- Der_int(O) = J for the first 20000 lattice h.
- The J fixed points are {1, 29}, and J(y) < y exactly at {7, 11, 13, 17, 19, 23}.
- P² is 32 classes mod 450, all case 1. Its first copies are 1, 6, 29, 953, 1457, 2558.
- Der_int(P²)(7) = 6 against D(7) = 2. Lattice columns 31..169 carry 953. Der_int(P²)(6301) = 1505729, and 6301 divides 45171869.
- Period-1 and period-2 seed sets below 3000 match the lists above, as do all the factorisations.
- X(1) = 80434 and X(7) = 43849291330.

**`spot_c_level11.py`**
- All 208 level-11 escape classes (144 in case 1, 64 in case 19) hold a revealed gear below 3·10⁶.
- The largest least revealed gear is 1,553,537 (class 1217 mod 2310, case 19). This is a measurement only.

All three spot-check scripts are in `C:/Users/Alex/AppData/Local/Temp/claude/C--dev-primes/a1c3a0ad-3acd-4fbe-b091-1e36256528ec/scratchpad/assemble_0925/`:
- spot_a_converse.py
- spot_b_doom_op.py
- spot_c_level11.py