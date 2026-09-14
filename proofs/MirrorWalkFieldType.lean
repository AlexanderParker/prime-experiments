/-
MirrorWalkFieldType (round 52, 2026-09-14): ONE field type on the landing line, alone.

The field type is (gear `g`, left member, order 2) of the final step up, `L = E + 6h`.
Its rule on the high-gear line, in three parts, each its own lemma:
  1. the class: `g ∣ E + 6h` iff `h ≡ a (mod g)`, where `a` is the class with `6a ≡ -E (mod g)`
     (needs `g` coprime to 6, true for every gear from 5 up);
  2. the line: inside the class, `h = a + g t` and `E + 6h = g (c₀ + 6 t)` with `g c₀ = E + 6a`,
     so the cofactor runs along the column line `c₀ + 6t`;
  3. the shape: the type takes `h` iff the cofactor `c₀ + 6t` is prime (order 2 = `g` times one
     prime); a cofactor divisible by a smaller gear `p` is not this type's kill, and that happens
     exactly on the classes of `t` with `6t ≡ -c₀ (mod p)`.
Nothing here is about the band as a group; the same three lemmas apply to any gear's type.
-/
import MirrorWalkColumn

namespace MirrorWalk

/-- **Part 1, the class.**  With `6a ≡ -E (mod g)` and `g` coprime to 6, gear `g` divides the
left member `E + 6h` iff `h ≡ a (mod g)`. -/
theorem left_class_iff {g : ℕ} {E a h : ℤ} (hco : IsCoprime (g : ℤ) 6)
    (ha : 6 * a ≡ -E [ZMOD g]) : (g : ℤ) ∣ E + 6 * h ↔ h ≡ a [ZMOD g] := by
  constructor
  · intro hd
    have h1 : 6 * h ≡ -E [ZMOD g] := by
      rw [Int.modEq_iff_dvd]
      have : -E - 6 * h = -(E + 6 * h) := by ring
      rw [this, dvd_neg]; exact hd
    have h2 : 6 * h ≡ 6 * a [ZMOD g] := h1.trans ha.symm
    rw [Int.modEq_iff_dvd] at h2 ⊢
    have h3 : (g : ℤ) ∣ (a - h) * 6 := by
      have : (a - h) * 6 = 6 * a - 6 * h := by ring
      rw [this]; exact h2
    exact hco.dvd_of_dvd_mul_right h3
  · intro hh
    have h1 : 6 * h ≡ -E [ZMOD g] := (hh.mul_left 6).trans ha
    rw [Int.modEq_iff_dvd] at h1
    have : -E - 6 * h = -(E + 6 * h) := by ring
    rwa [this, dvd_neg] at h1

/-- **Part 2, the line.**  Inside the class `h = a + g t` the left member is `g (c₀ + 6 t)`. -/
theorem left_member_on_line {g : ℕ} {E a c₀ t : ℤ} (hc : (g : ℤ) * c₀ = E + 6 * a) :
    E + 6 * (a + g * t) = (g : ℤ) * (c₀ + 6 * t) := by
  linear_combination (-1 : ℤ) * hc

/-- **Part 3, the shape.**  The type (g, left, order 2) takes the column `h = a + g t` iff the
cofactor `c₀ + 6 t` is prime: then the left member is `g` times exactly one prime. -/
theorem left_order_two_iff {g : ℕ} {E a c₀ t : ℤ} (hc : (g : ℤ) * c₀ = E + 6 * a) :
    (∃ p : ℤ, Prime p ∧ E + 6 * (a + g * t) = (g : ℤ) * p ∧ p = c₀ + 6 * t) ↔ Prime (c₀ + 6 * t) := by
  constructor
  · rintro ⟨p, hp, -, rfl⟩; exact hp
  · intro hp; exact ⟨c₀ + 6 * t, hp, left_member_on_line hc, rfl⟩

/-- **Part 3, the smaller gear.**  A smaller gear `p` takes the cofactor exactly on the classes
`6 t ≡ -c₀ (mod p)`; on those `t` the kill is `p`'s, not this type's. -/
theorem cofactor_taken_by_iff {p : ℕ} {c₀ t : ℤ} :
    (p : ℤ) ∣ c₀ + 6 * t ↔ 6 * t ≡ -c₀ [ZMOD p] := by
  rw [Int.modEq_iff_dvd]
  have : -c₀ - 6 * t = -(c₀ + 6 * t) := by ring
  rw [this, dvd_neg]

/-- The order-2 shape and a smaller divisor exclude each other: a prime cofactor at least `g`
has no divisor `p` with `2 ≤ p < g`. -/
theorem order_two_excludes_smaller {g p : ℕ} {c : ℤ} (hp : 2 ≤ p) (hpg : p < g) (hgc : (g : ℤ) ≤ c)
    (hc : Prime c) : ¬ (p : ℤ) ∣ c := by
  rintro ⟨k, hk⟩
  have hpos : (0 : ℤ) < p := by exact_mod_cast (by omega : 0 < p)
  have hpc : (p : ℤ) < c := lt_of_lt_of_le (by exact_mod_cast hpg) hgc
  have hcd : c ∣ (p : ℤ) * k := ⟨1, by rw [← hk]; ring⟩
  rcases hc.dvd_or_dvd hcd with h | h
  · exact absurd (Int.le_of_dvd hpos h) (not_le.mpr hpc)
  · obtain ⟨m, hm⟩ := h
    have hc0 : c ≠ 0 := hc.ne_zero
    have e : c * 1 = c * ((p : ℤ) * m) := by
      calc c * 1 = c := mul_one c
        _ = (p : ℤ) * k := hk
        _ = (p : ℤ) * (c * m) := by rw [hm]
        _ = c * ((p : ℤ) * m) := by ring
    have h1 : (p : ℤ) * m = 1 := (mul_left_cancel₀ hc0 e).symm
    have hp1 : (p : ℤ) ∣ 1 := ⟨m, h1.symm⟩
    have h2 := Int.le_of_dvd one_pos hp1
    have h3 : (2 : ℤ) ≤ p := by exact_mod_cast hp
    omega

/-! ### The next type alone: (g, left member, order 3)

The cofactor `c₀ + 6t` is a product of two primes, both at least `g`.  Take the smaller, `p₁`:
it takes the cofactor on its class of `t` (cofactor_taken_by_iff), and inside that class
`t = t₁ + p₁ s` the quotient runs on its own column line `d₀ + 6 s`, where the type asks for a
prime: the order-2 shape one level down. -/

/-- **Order 3, the shape.**  The left member is `g` times two primes at least `g` iff some prime
`p₁ ≥ g` divides the cofactor with a prime quotient at least `g`. -/
theorem left_order_three_iff {g : ℕ} {E a c₀ t : ℤ} (hg : 0 < g) (hc : (g : ℤ) * c₀ = E + 6 * a) :
    (∃ p₁ p₂ : ℤ, Prime p₁ ∧ Prime p₂ ∧ (g : ℤ) ≤ p₁ ∧ (g : ℤ) ≤ p₂ ∧
        E + 6 * (a + g * t) = (g : ℤ) * (p₁ * p₂)) ↔
      (∃ p₁ : ℤ, Prime p₁ ∧ (g : ℤ) ≤ p₁ ∧ ∃ d, c₀ + 6 * t = p₁ * d ∧ Prime d ∧ (g : ℤ) ≤ d) := by
  have hg0 : (g : ℤ) ≠ 0 := by exact_mod_cast hg.ne'
  constructor
  · rintro ⟨p₁, p₂, h1, h2, g1, g2, he⟩
    rw [left_member_on_line hc] at he
    exact ⟨p₁, h1, g1, p₂, mul_left_cancel₀ hg0 he, h2, g2⟩
  · rintro ⟨p₁, h1, g1, d, hd, h2, g2⟩
    exact ⟨p₁, d, h1, h2, g1, g2, by rw [left_member_on_line hc, hd]⟩

/-- **Order 3, the line one level down.**  Inside `p₁`'s class `t = t₁ + p₁ s` the quotient of the
cofactor by `p₁` is `d₀ + 6 s`, with `p₁ d₀ = c₀ + 6 t₁`. -/
theorem quotient_on_line {p₁ c₀ t₁ d₀ s : ℤ} (hd : p₁ * d₀ = c₀ + 6 * t₁) :
    c₀ + 6 * (t₁ + p₁ * s) = p₁ * (d₀ + 6 * s) := by
  linear_combination (-1 : ℤ) * hd

/-! ### The right member, and the down step: the same three parts with the class moved. -/

/-- **Right member, the class.**  With `6b ≡ -(E + 2) (mod g)`, `g ∣ E + 6h + 2` iff `h ≡ b`. -/
theorem right_class_iff {g : ℕ} {E b h : ℤ} (hco : IsCoprime (g : ℤ) 6)
    (hb : 6 * b ≡ -(E + 2) [ZMOD g]) : (g : ℤ) ∣ E + 6 * h + 2 ↔ h ≡ b [ZMOD g] := by
  have := left_class_iff (E := E + 2) (h := h) hco hb
  rwa [show E + 2 + 6 * h = E + 6 * h + 2 by ring] at this

/-- **Right member, the line.**  `h = b + g t` gives `E + 6h + 2 = g (c₀ + 6t)`, `g c₀ = E + 2 + 6b`. -/
theorem right_member_on_line {g : ℕ} {E b c₀ t : ℤ} (hc : (g : ℤ) * c₀ = E + 2 + 6 * b) :
    E + 6 * (b + g * t) + 2 = (g : ℤ) * (c₀ + 6 * t) := by
  linear_combination (-1 : ℤ) * hc

/-- **Down step, the class.**  The landing is `E - 6h`; with `6a' ≡ E (mod g)`, `g ∣ E - 6h` iff
`h ≡ a'`. -/
theorem down_class_iff {g : ℕ} {E a' h : ℤ} (hco : IsCoprime (g : ℤ) 6)
    (ha : 6 * a' ≡ E [ZMOD g]) : (g : ℤ) ∣ E - 6 * h ↔ h ≡ a' [ZMOD g] := by
  have h1 : (g : ℤ) ∣ E - 6 * h ↔ (g : ℤ) ∣ -E + 6 * h := by
    rw [show -E + 6 * h = -(E - 6 * h) by ring, dvd_neg]
  rw [h1]
  apply left_class_iff hco
  rw [neg_neg]; exact ha

/-- **Down step, the line.**  `h = a' + g t` gives `E - 6h = g (c₀ - 6t)`, `g c₀ = E - 6a'`: the
cofactor line runs downward. -/
theorem down_member_on_line {g : ℕ} {E a' c₀ t : ℤ} (hc : (g : ℤ) * c₀ = E - 6 * a') :
    E - 6 * (a' + g * t) = (g : ℤ) * (c₀ - 6 * t) := by
  linear_combination (-1 : ℤ) * hc

/-! ### A high gear's type alone: above `√q` the order stops at 3. -/

/-- **High gear, order at most 3.**  If `g² > q`, the member `g c` lies at or below `q²`, and `c`
is a product of three primes each at least `g`, contradiction: `g⁴ ≤ g c ≤ q² < g⁴`. -/
theorem high_gear_no_order_four {g : ℕ} {q c p₁ p₂ p₃ : ℤ} (hq : 0 < q) (hg : q < (g : ℤ) ^ 2)
    (hle : (g : ℤ) * c ≤ q ^ 2) (hc : c = p₁ * p₂ * p₃)
    (h1 : (g : ℤ) ≤ p₁) (h2 : (g : ℤ) ≤ p₂) (h3 : (g : ℤ) ≤ p₃) : False := by
  have hg0 : (0 : ℤ) ≤ g := by exact_mod_cast Nat.zero_le g
  have a1 : (g : ℤ) * g ≤ g * p₁ := mul_le_mul_of_nonneg_left h1 hg0
  have b1 : (0 : ℤ) ≤ g * p₁ := le_trans (mul_nonneg hg0 hg0) a1
  have a2 : (g : ℤ) * g * g ≤ g * p₁ * p₂ := mul_le_mul a1 h2 hg0 b1
  have b2 : (0 : ℤ) ≤ g * p₁ * p₂ := le_trans (mul_nonneg (mul_nonneg hg0 hg0) hg0) a2
  have a3 : (g : ℤ) * g * g * g ≤ g * p₁ * p₂ * p₃ := mul_le_mul a2 h3 hg0 b2
  have hq2 : q ^ 2 < ((g : ℤ) ^ 2) ^ 2 := by
    rw [pow_two q, pow_two ((g : ℤ) ^ 2)]; exact mul_lt_mul'' hg hg hq.le hq.le
  have e : ((g : ℤ) ^ 2) ^ 2 = g * g * g * g := by ring
  have e2 : (g : ℤ) * (p₁ * p₂ * p₃) = g * p₁ * p₂ * p₃ := by ring
  rw [hc, e2] at hle
  rw [e] at hq2
  linarith

/-! ### Same-anchor pair: base gears 5 and 7 on column `h`, fixed teeth at every machine. -/

/-- Gear 5's teeth on the `h`-line: left member at `h ≡ 1`, right member at `h ≡ 4 (mod 5)`. -/
theorem teeth_five (h : ℤ) :
    ((5 : ℤ) ∣ 6 * h - 1 ↔ h % 5 = 1) ∧ ((5 : ℤ) ∣ 6 * h + 1 ↔ h % 5 = 4) := by
  constructor <;> omega

/-- Gear 7's teeth on the `h`-line: left member at `h ≡ 6`, right member at `h ≡ 1 (mod 7)`. -/
theorem teeth_seven (h : ℤ) :
    ((7 : ℤ) ∣ 6 * h - 1 ↔ h % 7 = 6) ∧ ((7 : ℤ) ∣ 6 * h + 1 ↔ h % 7 = 1) := by
  constructor <;> omega

/-- **The pair (5, 7) on column `h`.**  Column `h` is open to both iff `h` avoids `1, 4 (mod 5)`
and `1, 6 (mod 7)`: fifteen open classes of `h` modulo 35, the same at every machine. -/
theorem pair_five_seven (h : ℤ) :
    (OpenTo 5 (6 * h - 1) ∧ OpenTo 7 (6 * h - 1)) ↔
      (h % 5 ≠ 1 ∧ h % 5 ≠ 4 ∧ h % 7 ≠ 6 ∧ h % 7 ≠ 1) := by
  unfold OpenTo
  have e : 6 * h - 1 + 2 = 6 * h + 1 := by ring
  rw [e]
  push_cast
  constructor
  · rintro ⟨⟨a, b⟩, ⟨c, d⟩⟩
    exact ⟨fun x => a ((teeth_five h).1.mpr x), fun x => b ((teeth_five h).2.mpr x),
           fun x => c ((teeth_seven h).1.mpr x), fun x => d ((teeth_seven h).2.mpr x)⟩
  · rintro ⟨a, b, c, d⟩
    exact ⟨⟨fun x => a ((teeth_five h).1.mp x), fun x => b ((teeth_five h).2.mp x)⟩,
           ⟨fun x => c ((teeth_seven h).1.mp x), fun x => d ((teeth_seven h).2.mp x)⟩⟩

/-! ### The walk's own gear against its own landing. -/

/-- **Own landing.**  The gear `h` of the final step strikes its own landing `E + 6h` iff `h ∣ E`
(left member) or `h ∣ E + 2` (right member): the step's own gear only ever tests `E`. -/
theorem own_landing_iff (E : ℤ) (h : ℕ) :
    (((h : ℤ) ∣ E + 6 * h) ↔ (h : ℤ) ∣ E) ∧ (((h : ℤ) ∣ E + 6 * h + 2) ↔ (h : ℤ) ∣ E + 2) := by
  constructor
  · exact dvd_add_left (dvd_mul_left (h : ℤ) 6)
  · rw [show E + 6 * (h : ℤ) + 2 = (E + 2) + 6 * h by ring]
    exact dvd_add_left (dvd_mul_left (h : ℤ) 6)

/-! ### A triple: the fixed pair (5, 7) on column `h` with one `E`-anchored gear `g`. -/

/-- **Triple (5, 7, g).**  With `E ≡ -1 (mod 35)` (5 and 7 in the base) and `g` coprime to 6
with teeth `a, b` (`6a ≡ -E`, `6b ≡ -(E + 2) (mod g)`), the landing `E + 6h` is open to 5, 7
and `g` iff `h` avoids `1, 4 (mod 5)`, `6, 1 (mod 7)`, and `a, b (mod g)`. -/
theorem triple_five_seven_iff {g : ℕ} {E a b h : ℤ} (hE : E ≡ -1 [ZMOD 35])
    (hco : IsCoprime (g : ℤ) 6) (ha : 6 * a ≡ -E [ZMOD g]) (hb : 6 * b ≡ -(E + 2) [ZMOD g]) :
    (OpenTo 5 (E + 6 * h) ∧ OpenTo 7 (E + 6 * h) ∧ OpenTo g (E + 6 * h)) ↔
      (h % 5 ≠ 1 ∧ h % 5 ≠ 4 ∧ h % 7 ≠ 6 ∧ h % 7 ≠ 1 ∧ ¬ h ≡ a [ZMOD g] ∧ ¬ h ≡ b [ZMOD g]) := by
  have h5 := (landing_open_base_iff (h := h) hE (by norm_num : ((5 : ℕ) : ℤ) ∣ 35)).1
  have h7 := (landing_open_base_iff (h := h) hE (by norm_num : ((7 : ℕ) : ℤ) ∣ 35)).1
  have hg : OpenTo g (E + 6 * h) ↔ (¬ h ≡ a [ZMOD g] ∧ ¬ h ≡ b [ZMOD g]) := by
    unfold OpenTo
    rw [left_class_iff hco ha, right_class_iff hco hb]
  rw [h5, h7, hg, ← and_assoc, pair_five_seven]
  tauto

/-! ### A high gear's teeth are positions: one tooth holds at most one point per stretch of
length `g`. -/

/-- **One point per tooth per stretch.**  Two points of the same class modulo `g` closer than `g`
are the same point. -/
theorem tooth_unique {g : ℕ} {a h₁ h₂ : ℤ} (h1 : h₁ ≡ a [ZMOD g]) (h2 : h₂ ≡ a [ZMOD g])
    (hlt : |h₁ - h₂| < g) : h₁ = h₂ := by
  have hd : (g : ℤ) ∣ h₁ - h₂ := by
    have := (h1.trans h2.symm)
    rw [Int.modEq_iff_dvd] at this
    have e : h₁ - h₂ = -(h₂ - h₁) := by ring
    rw [e]; exact (dvd_neg).mpr this
  have := Int.eq_zero_of_abs_lt_dvd hd hlt
  linarith

/-- **A high gear above half the line has at most two points per tooth in reach**: any three
points of one tooth inside `(√q, q]` would put two of them closer than `g` when `2g > q - √q`.
Stated for a tooth's three points: not all distinct. -/
theorem tooth_at_most_two {g : ℕ} {q a h₁ h₂ h₃ : ℤ} (hg : q < 2 * g)
    (h1 : h₁ ≡ a [ZMOD g]) (h2 : h₂ ≡ a [ZMOD g]) (h3 : h₃ ≡ a [ZMOD g])
    (b1 : 0 < h₁ ∧ h₁ ≤ q) (b2 : 0 < h₂ ∧ h₂ ≤ q) (b3 : 0 < h₃ ∧ h₃ ≤ q) :
    h₁ = h₂ ∨ h₂ = h₃ ∨ h₁ = h₃ := by
  by_contra hne
  push_neg at hne
  obtain ⟨n12, n23, n13⟩ := hne
  have d12 : (g : ℤ) ∣ h₁ - h₂ := by
    have := (h1.trans h2.symm); rw [Int.modEq_iff_dvd] at this
    rw [show h₁ - h₂ = -(h₂ - h₁) by ring]; exact (dvd_neg).mpr this
  have d23 : (g : ℤ) ∣ h₂ - h₃ := by
    have := (h2.trans h3.symm); rw [Int.modEq_iff_dvd] at this
    rw [show h₂ - h₃ = -(h₃ - h₂) by ring]; exact (dvd_neg).mpr this
  have d13 : (g : ℤ) ∣ h₁ - h₃ := by
    have := (h1.trans h3.symm); rw [Int.modEq_iff_dvd] at this
    rw [show h₁ - h₃ = -(h₃ - h₁) by ring]; exact (dvd_neg).mpr this
  -- each nonzero difference has absolute value at least g
  have a12 : (g : ℤ) ≤ |h₁ - h₂| := Int.le_of_dvd (abs_pos.mpr (sub_ne_zero.mpr n12)) ((dvd_abs _ _).mpr d12)
  have a23 : (g : ℤ) ≤ |h₂ - h₃| := Int.le_of_dvd (abs_pos.mpr (sub_ne_zero.mpr n23)) ((dvd_abs _ _).mpr d23)
  have a13 : (g : ℤ) ≤ |h₁ - h₃| := Int.le_of_dvd (abs_pos.mpr (sub_ne_zero.mpr n13)) ((dvd_abs _ _).mpr d13)
  -- three points in (0, q] with pairwise gaps at least g force 2g ≤ q
  rcases le_abs.mp a12 with c12 | c12 <;> rcases le_abs.mp a23 with c23 | c23 <;>
    rcases le_abs.mp a13 with c13 | c13 <;> omega

end MirrorWalk
