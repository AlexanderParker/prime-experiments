import Mathlib.Tactic
import Mathlib.Data.Nat.Prime.Basic

/-!
# Near-neighbour kills of a missed copy: cofactor floors, kill-free windows, boundary cases

A missed copy of a gear `g` has legs `g² + a`, `g² + a + 2`, with `a = 28` (case 1,
`g ≡ ±1 mod 5`: legs `g² + 28`, `g² + 30`) or `a = 10` (case 19, `g ≡ ±2 mod 5`: legs `g² + 10`,
`g² + 12`). A lower gear `h < g` kills `c_g` when it divides a leg. Every statement below is for
all naturals `g`, `h`, `h'`, `a`, `b`, `t`, `K` as displayed; no machine size appears.

Ported from the scratch files `c5/NearKill.lean` and `c5_defend/C5Defence.lean`.

## The shift and the parity floors

* `dvd_shift` : for `h ≤ g`, `h ∣ g² + a ↔ h ∣ (g - h)² + a`.
* `kill_floor` : odd `g, h`, even `a > 0`, `h < g`, `h ∣ g² + a` give `2h ≤ (g - h)² + a`
  (the cofactor `κ = ((g - h)² + a)/h` is even, hence at least 2).
* `kill_floor4` : if moreover `4 ∣ a` then `4h ≤ (g - h)² + a`.

## The kill-free windows

* `near_immune` : if `(g - h')² + a + 2 < 2h'` then no odd `h` with `h' ≤ h < g` divides either
  leg `g² + a`, `g² + a + 2`.
* `near_immune_case1` : window `(g - h')² + 30 < 2h'` for legs `g² + 28`, `g² + 30`.
* `near_immune_case19` : window `(g - h')² + 10 < 2h'` for legs `g² + 10`, `g² + 12`.

## The exact boundary cases

* `sq29_class` / `sq9_class` : `t² + 29 = 2g` (resp. `t² + 9 = 2g`) with `g` prime to 30 forces
  `g ≡ 19 (30)` iff `5 ∤ t`, else `g ≡ 7` (resp. `g ≡ 17 (30)` iff `5 ∣ t`, else `g ≡ 29`).
* `boundary30_iff` / `boundary10_iff` : a case-1 (resp. case-19) kill on or inside the parity
  window happens exactly when `2g - 29 = t²` with `5 ∤ t` (resp. `2g - 9 = t²` with `5 ∣ t`) and
  `h = g - t + 1`.
* `boundary30_upper_only` : in the leg-30 boundary event `h` kills the upper leg only.

## The class-refined floors and windows

* `resOK`, `excluded`, `allExcluded` : the residue certificate mod `30κ`.
* `excluded_sound`, `floor_of_allExcluded` : a certificate excludes cofactor `κ`, and all
  cofactors below `K`, for every `g`, `h` with `h` prime to 30 and `g` in the class.
* `cert_b_c` : the sixteen certificates, checked by kernel reduction.
* `floor_b_c` : `K(b, c) · h ≤ (g - h)² + b` for every kill of leg `b` of a gear `g ≡ c (30)`.
* `window_c` : the class-refined kill-free windows (both legs of the class's case missed).
* `sharp_c` : every class window is sharp: a prime kill of a prime gear on its boundary.
* `window_mono` and `neighbour_c` : a window that holds at a lower neighbour `h'` holds at every
  `h` in `[h', g)`, so every `h` prime to 30 in `[h', g)` misses both legs of `c_g`.
-/

namespace RangeLine

/-! ## The shift and the parity floors -/

/-- The shift: for `h ≤ g`, `h` divides `g² + a` exactly when it divides `(g - h)² + a`. -/
lemma dvd_shift {g h a : ℕ} (hle : h ≤ g) :
    h ∣ g ^ 2 + a ↔ h ∣ (g - h) ^ 2 + a := by
  obtain ⟨d, rfl⟩ : ∃ d, g = h + d := ⟨g - h, by omega⟩
  have e : (h + d) ^ 2 + a = (d ^ 2 + a) + h * (h + 2 * d) := by ring
  rw [Nat.add_sub_cancel_left, e]
  exact Nat.dvd_add_left (dvd_mul_right h _)

/-- The parity floor: for odd `g, h`, even `a > 0`, `h < g` and `h ∣ g² + a`, the cofactor
`((g - h)² + a) / h` is even and nonzero, so `2h ≤ (g - h)² + a`. -/
theorem kill_floor {g h a : ℕ} (hg : Odd g) (hh : Odd h) (ha : Even a) (ha0 : 0 < a)
    (hlt : h < g) (hdvd : h ∣ g ^ 2 + a) : 2 * h ≤ (g - h) ^ 2 + a := by
  rw [dvd_shift hlt.le] at hdvd
  obtain ⟨k, hk⟩ := hdvd
  have hd : Even (g - h) := Nat.Odd.sub_odd hg hh
  have hN : Even ((g - h) ^ 2 + a) := (hd.pow_of_ne_zero two_ne_zero).add ha
  rw [hk] at hN
  have hk2 : Even k := by
    rcases Nat.even_mul.mp hN with h1 | h1
    · exact absurd h1 (Nat.not_even_iff_odd.mpr hh)
    · exact h1
  have hk0 : k ≠ 0 := by
    rintro rfl
    simp at hk
    omega
  obtain ⟨m, hm⟩ := hk2
  have : 1 ≤ m := by omega
  rw [hk, hm]
  nlinarith

/-- The floor for `4 ∣ a`: for odd `g, h`, `4 ∣ a`, `a > 0`, `h < g` and `h ∣ g² + a`, the
cofactor is a nonzero multiple of 4, so `4h ≤ (g - h)² + a`. -/
theorem kill_floor4 {g h a : ℕ} (hg : Odd g) (hh : Odd h) (ha : 4 ∣ a) (ha0 : 0 < a)
    (hlt : h < g) (hdvd : h ∣ g ^ 2 + a) : 4 * h ≤ (g - h) ^ 2 + a := by
  rw [dvd_shift hlt.le] at hdvd
  obtain ⟨k, hk⟩ := hdvd
  obtain ⟨t, ht⟩ : Even (g - h) := Nat.Odd.sub_odd hg hh
  have h4 : 4 ∣ (g - h) ^ 2 + a := by
    rw [ht]
    have : (t + t) ^ 2 = 4 * t ^ 2 := by ring
    rw [this]
    exact Nat.dvd_add (dvd_mul_right 4 _) ha
  rw [hk] at h4
  have hcop : Nat.Coprime 4 h := by
    have : Nat.Coprime 2 h := (Nat.coprime_two_left).mpr hh
    simpa using Nat.Coprime.pow_left 2 this
  have hk4 : 4 ∣ k := hcop.dvd_of_dvd_mul_left h4
  have hk0 : k ≠ 0 := by
    rintro rfl
    simp at hk
    omega
  obtain ⟨m, hm⟩ := hk4
  have : 1 ≤ m := by omega
  rw [hk, hm]
  nlinarith

/-! ## The kill-free windows -/

/-- Near-neighbour immunity: for odd `g`, even `a > 0`, if `(g - h')² + a + 2 < 2h'` then every
odd `h` with `h' ≤ h < g` misses both legs `g² + a` and `g² + a + 2` of `c_g`. -/
theorem near_immune {g h h' a : ℕ} (hg : Odd g) (ha : Even a) (ha0 : 0 < a)
    (hwin : (g - h') ^ 2 + a + 2 < 2 * h') (hh : Odd h) (hh' : h' ≤ h) (hlt : h < g) :
    ¬ h ∣ g ^ 2 + a ∧ ¬ h ∣ g ^ 2 + a + 2 := by
  have hsq : (g - h) ^ 2 ≤ (g - h') ^ 2 := Nat.pow_le_pow_left (by omega) 2
  constructor
  · intro hd
    have := kill_floor hg hh ha ha0 hlt hd
    omega
  · intro hd
    have := kill_floor hg hh (ha.add even_two) (by omega) hlt (by simpa [add_assoc] using hd)
    omega

/-- Kill-free window, case 1 (legs `g² + 28`, `g² + 30`): for odd `g`, if
`(g - h')² + 30 < 2h'` then every odd `h` with `h' ≤ h < g` misses both legs. -/
theorem near_immune_case1 {g h h' : ℕ} (hg : Odd g)
    (hwin : (g - h') ^ 2 + 30 < 2 * h') (hh : Odd h) (hh' : h' ≤ h) (hlt : h < g) :
    ¬ h ∣ g ^ 2 + 28 ∧ ¬ h ∣ g ^ 2 + 30 := by
  have hsq : (g - h) ^ 2 ≤ (g - h') ^ 2 := Nat.pow_le_pow_left (by omega) 2
  constructor
  · intro hd
    have := kill_floor4 hg hh (by norm_num) (by norm_num) hlt hd
    omega
  · intro hd
    have := kill_floor hg hh (by decide) (by norm_num) hlt hd
    omega

/-- Kill-free window, case 19 (legs `g² + 10`, `g² + 12`): for odd `g`, if
`(g - h')² + 10 < 2h'` then every odd `h` with `h' ≤ h < g` misses both legs. -/
theorem near_immune_case19 {g h h' : ℕ} (hg : Odd g)
    (hwin : (g - h') ^ 2 + 10 < 2 * h') (hh : Odd h) (hh' : h' ≤ h) (hlt : h < g) :
    ¬ h ∣ g ^ 2 + 10 ∧ ¬ h ∣ g ^ 2 + 12 := by
  have hsq : (g - h) ^ 2 ≤ (g - h') ^ 2 := Nat.pow_le_pow_left (by omega) 2
  have hpos : 1 ≤ h := hh.pos
  constructor
  · intro hd
    have := kill_floor hg hh (by decide) (by norm_num) hlt hd
    omega
  · intro hd
    have := kill_floor4 hg hh (by norm_num) (by norm_num) hlt hd
    omega

/-! ## The square families and their classes mod 30 -/

/-- The family `t² + 29 = 2g` with `g` prime to 30: either `g ≡ 19 (mod 30)` and `5 ∤ t`, or
`g ≡ 7 (mod 30)` and `5 ∣ t`. -/
theorem sq29_class {g t : ℕ} (h : t ^ 2 + 29 = 2 * g) (h2 : g % 2 = 1) (h3 : g % 3 ≠ 0)
    (h5 : g % 5 ≠ 0) :
    (g % 30 = 19 ∧ t % 5 ≠ 0) ∨ (g % 30 = 7 ∧ t % 5 = 0) := by
  obtain ⟨q, r, hr, rfl⟩ : ∃ q r, r < 60 ∧ t = 60 * q + r :=
    ⟨t / 60, t % 60, Nat.mod_lt _ (by norm_num), (Nat.div_add_mod t 60).symm⟩
  have e : (60 * q + r) ^ 2 = 60 * (60 * q ^ 2 + 2 * q * r) + r ^ 2 := by ring
  rw [e] at h
  generalize 60 * q ^ 2 + 2 * q * r = Q at h
  interval_cases r <;> omega

/-- The family `t² + 9 = 2g` with `g` prime to 30: either `g ≡ 17 (mod 30)` and `5 ∣ t`, or
`g ≡ 29 (mod 30)` and `5 ∤ t`. -/
theorem sq9_class {g t : ℕ} (h : t ^ 2 + 9 = 2 * g) (h2 : g % 2 = 1) (h3 : g % 3 ≠ 0)
    (h5 : g % 5 ≠ 0) :
    (g % 30 = 17 ∧ t % 5 = 0) ∨ (g % 30 = 29 ∧ t % 5 ≠ 0) := by
  obtain ⟨q, r, hr, rfl⟩ : ∃ q r, r < 60 ∧ t = 60 * q + r :=
    ⟨t / 60, t % 60, Nat.mod_lt _ (by norm_num), (Nat.div_add_mod t 60).symm⟩
  have e : (60 * q + r) ^ 2 = 60 * (60 * q ^ 2 + 2 * q * r) + r ^ 2 := by ring
  rw [e] at h
  generalize 60 * q ^ 2 + 2 * q * r = Q at h
  interval_cases r <;> omega

/-! ## The exact boundary cases -/

/-- Leg 30. For `g` prime to 30 and odd `h < g`: `g` is case 1, `h` divides a leg of
`c_g` and `(g - h)² + 30 ≤ 2h` (a kill on or inside the window) **iff** `2g - 29 = t²` with
`5 ∤ t` and `h = g - t + 1`. In that event `g ≡ 19 (mod 30)` (`sq29_class`). -/
theorem boundary30_iff {g h : ℕ} (h2 : g % 2 = 1) (h3 : g % 3 ≠ 0) (h5 : g % 5 ≠ 0)
    (hh : Odd h) (hlt : h < g) :
    ((g % 5 = 1 ∨ g % 5 = 4) ∧ (h ∣ g ^ 2 + 28 ∨ h ∣ g ^ 2 + 30) ∧ (g - h) ^ 2 + 30 ≤ 2 * h) ↔
      (∃ t, t ^ 2 + 29 = 2 * g ∧ h + t = g + 1 ∧ t % 5 ≠ 0) := by
  have hg : Odd g := Nat.odd_iff.mpr h2
  constructor
  · rintro ⟨hc, hk, hw⟩
    have heq : (g - h) ^ 2 + 30 = 2 * h := by
      rcases hk with hk | hk
      · have := kill_floor4 hg hh (by norm_num) (by norm_num) hlt hk
        omega
      · have := kill_floor hg hh (by decide) (by norm_num) hlt hk
        omega
    refine ⟨g - h + 1, ?_, by omega, ?_⟩
    · have e : (g - h + 1) ^ 2 + 29 = (g - h) ^ 2 + 30 + 2 * (g - h) := by ring
      rw [e, heq]
      omega
    · have e : (g - h + 1) ^ 2 + 29 = 2 * g := by
        have e' : (g - h + 1) ^ 2 + 29 = (g - h) ^ 2 + 30 + 2 * (g - h) := by ring
        rw [e', heq]
        omega
      rcases sq29_class e h2 h3 h5 with ⟨_, ht⟩ | ⟨h7, _⟩
      · exact ht
      · omega
  · rintro ⟨t, ht, hht, ht5⟩
    rcases sq29_class ht h2 h3 h5 with ⟨h19, _⟩ | ⟨_, ht0⟩
    · have hd : g - h + 1 = t := by omega
      have heq : (g - h) ^ 2 + 30 = 2 * h := by
        have e : (g - h + 1) ^ 2 + 29 = (g - h) ^ 2 + 30 + 2 * (g - h) := by ring
        rw [hd] at e
        omega
      refine ⟨by omega, Or.inr ?_, heq.le⟩
      rw [dvd_shift hlt.le, heq]
      exact dvd_mul_left h 2
    · exact absurd ht0 ht5

/-- In the leg-30 boundary event `(g - h)² + 30 = 2h` (with `h < g`), `h` kills the upper leg
`g² + 30` and misses the lower leg `g² + 28`. -/
theorem boundary30_upper_only {g h : ℕ} (hlt : h < g)
    (heq : (g - h) ^ 2 + 30 = 2 * h) : h ∣ g ^ 2 + 30 ∧ ¬ h ∣ g ^ 2 + 28 := by
  refine ⟨by rw [dvd_shift hlt.le, heq]; exact dvd_mul_left h 2, ?_⟩
  rw [dvd_shift hlt.le]
  intro hd
  have e : (g - h) ^ 2 + 28 + 2 = 2 * h := by omega
  have h2 : h ∣ 2 := by
    have : h ∣ 2 * h := dvd_mul_left h 2
    rw [← e] at this
    exact (Nat.dvd_add_right hd).mp this
  have := Nat.le_of_dvd (by norm_num) h2
  omega

/-- Leg 10, the mirror statement. For `g` prime to 30 and odd `h < g`: `g` is case 19,
`h` divides a leg of `c_g` and `(g - h)² + 10 ≤ 2h` **iff** `2g - 9 = t²` with `5 ∣ t` and
`h = g - t + 1`. In that event `g ≡ 17 (mod 30)` (`sq9_class`). -/
theorem boundary10_iff {g h : ℕ} (h2 : g % 2 = 1) (h3 : g % 3 ≠ 0) (h5 : g % 5 ≠ 0)
    (hh : Odd h) (hlt : h < g) :
    ((g % 5 = 2 ∨ g % 5 = 3) ∧ (h ∣ g ^ 2 + 10 ∨ h ∣ g ^ 2 + 12) ∧ (g - h) ^ 2 + 10 ≤ 2 * h) ↔
      (∃ t, t ^ 2 + 9 = 2 * g ∧ h + t = g + 1 ∧ t % 5 = 0) := by
  have hg : Odd g := Nat.odd_iff.mpr h2
  constructor
  · rintro ⟨hc, hk, hw⟩
    have heq : (g - h) ^ 2 + 10 = 2 * h := by
      rcases hk with hk | hk
      · have := kill_floor hg hh (by decide) (by norm_num) hlt hk
        omega
      · have := kill_floor4 hg hh (by norm_num) (by norm_num) hlt hk
        omega
    have e : (g - h + 1) ^ 2 + 9 = 2 * g := by
      have e' : (g - h + 1) ^ 2 + 9 = (g - h) ^ 2 + 10 + 2 * (g - h) := by ring
      rw [e', heq]
      omega
    refine ⟨g - h + 1, e, by omega, ?_⟩
    rcases sq9_class e h2 h3 h5 with ⟨_, ht⟩ | ⟨h29, _⟩
    · exact ht
    · omega
  · rintro ⟨t, ht, hht, ht5⟩
    rcases sq9_class ht h2 h3 h5 with ⟨h17, _⟩ | ⟨_, ht0⟩
    · have hd : g - h + 1 = t := by omega
      have heq : (g - h) ^ 2 + 10 = 2 * h := by
        have e : (g - h + 1) ^ 2 + 9 = (g - h) ^ 2 + 10 + 2 * (g - h) := by ring
        rw [hd] at e
        omega
      refine ⟨by omega, Or.inl ?_, heq.le⟩
      rw [dvd_shift hlt.le, heq]
      exact dvd_mul_left h 2
    · exact absurd ht5 ht0

/-! ## Class-refined floors by residue certificates -/

/-- Residue `r` of `d` mod `30κ` compatible with `κ h = d² + b`, `h` a unit mod 30 and
`h + d ≡ c (mod 30)`. -/
def resOK (κ b c r : ℕ) : Bool :=
  let m := (r ^ 2 + b) % (30 * κ)
  m % κ == 0 && Nat.gcd (m / κ) 30 == 1 && (m / κ + r) % 30 == c

/-- No residue mod `30κ` is compatible: cofactor `κ` is impossible for leg `b`, class `c`. -/
def excluded (κ b c : ℕ) : Bool := (List.range (30 * κ)).all (fun r => !resOK κ b c r)

/-- Every cofactor below `K` is impossible for leg `b`, class `c`. -/
def allExcluded (K b c : ℕ) : Bool := (List.range K).all (fun κ => κ == 0 || excluded κ b c)

/-- Soundness of one exclusion: if `excluded κ b c` holds, there are no `d`, `h`, `g` with
`κ h = d² + b`, `g = h + d`, `h` prime to 30 and `g ≡ c (mod 30)`. -/
theorem excluded_sound {κ b c d h g : ℕ} (hκ : 0 < κ) (hex : excluded κ b c = true)
    (hk : κ * h = d ^ 2 + b) (hg : g = h + d) (hcop : Nat.gcd h 30 = 1) (hc : g % 30 = c) :
    False := by
  unfold excluded at hex
  rw [List.all_eq_true] at hex
  have hn : 0 < 30 * κ := by omega
  have hr : d % (30 * κ) < 30 * κ := Nat.mod_lt _ hn
  have hx := hex (d % (30 * κ)) (List.mem_range.mpr hr)
  have hm1 : ((d % (30 * κ)) ^ 2 + b) % (30 * κ) = (d ^ 2 + b) % (30 * κ) :=
    ((Nat.mod_modEq d (30 * κ)).pow 2).add_right b
  have hm2 : (d ^ 2 + b) % (30 * κ) = κ * (h % 30) := by
    rw [← hk, mul_comm 30 κ]
    exact Nat.mul_mod_mul_left κ h 30
  have hm : ((d % (30 * κ)) ^ 2 + b) % (30 * κ) = κ * (h % 30) := hm1.trans hm2
  have hdiv : κ * (h % 30) / κ = h % 30 := Nat.mul_div_cancel_left _ hκ
  have hgcd : Nat.gcd (h % 30) 30 = 1 := by
    rw [← Nat.gcd_rec 30 h, Nat.gcd_comm]
    exact hcop
  have hmod30 : d % (30 * κ) % 30 = d % 30 := Nat.mod_mod_of_dvd d (Dvd.intro κ rfl)
  have hcls : (h % 30 + d % (30 * κ)) % 30 = c := by
    subst hg
    omega
  simp only [resOK, hm, hdiv, Nat.mul_mod_right, hgcd, hcls, beq_self_eq_true, Bool.and_self,
    Bool.not_true] at hx
  exact absurd hx (by decide)

/-- A certificate `allExcluded K b c` gives the floor `K h ≤ (g - h)² + b` for every kill
`h ∣ g² + b` with `h < g`, `h` prime to 30 and `g ≡ c (mod 30)`, for every `b > 0`. -/
theorem floor_of_allExcluded {K b c h g : ℕ} (hall : allExcluded K b c = true) (hb : 0 < b)
    (hlt : h < g) (hcop : Nat.gcd h 30 = 1) (hc : g % 30 = c) (hdvd : h ∣ g ^ 2 + b) :
    K * h ≤ (g - h) ^ 2 + b := by
  rw [dvd_shift hlt.le] at hdvd
  obtain ⟨κ, hκ⟩ := hdvd
  by_contra hcon
  push Not at hcon
  have hh0 : 0 < h := by
    rcases Nat.eq_zero_or_pos h with h0 | h0
    · subst h0
      simp at hcop
    · exact h0
  have hκK : κ < K := by
    by_contra hK
    push Not at hK
    have : K * h ≤ κ * h := Nat.mul_le_mul_right h hK
    rw [hκ] at hcon
    nlinarith
  have hκ0 : 0 < κ := by
    rcases Nat.eq_zero_or_pos κ with h0 | h0
    · subst h0
      simp at hκ
      omega
    · exact h0
  unfold allExcluded at hall
  rw [List.all_eq_true] at hall
  have hx := hall κ (List.mem_range.mpr hκK)
  have hex : excluded κ b c = true := by
    have hne : (κ == 0) = false := by simp; omega
    simpa [hne] using hx
  exact excluded_sound hκ0 hex (by rw [mul_comm, hκ]) (by omega) hcop hc

/-! ### The sixteen certificates `K(b, c)` (least cofactor not excluded mod `30κ`) -/

/-- Certificate: leg 28, class 1, every cofactor below 4 excluded. -/
theorem cert_28_1 : allExcluded 4 28 1 = true := by decide +kernel
/-- Certificate: leg 28, class 11, every cofactor below 32 excluded. -/
theorem cert_28_11 : allExcluded 32 28 11 = true := by decide +kernel
/-- Certificate: leg 28, class 19, every cofactor below 28 excluded. -/
theorem cert_28_19 : allExcluded 28 28 19 = true := by decide +kernel
/-- Certificate: leg 28, class 29, every cofactor below 32 excluded. -/
theorem cert_28_29 : allExcluded 32 28 29 = true := by decide +kernel
/-- Certificate: leg 30, class 1, every cofactor below 26 excluded. -/
theorem cert_30_1 : allExcluded 26 30 1 = true := by decide +kernel
/-- Certificate: leg 30, class 11, every cofactor below 6 excluded. -/
theorem cert_30_11 : allExcluded 6 30 11 = true := by decide +kernel
/-- Certificate: leg 30, class 19, every cofactor below 2 excluded. -/
theorem cert_30_19 : allExcluded 2 30 19 = true := by decide +kernel
/-- Certificate: leg 30, class 29, every cofactor below 22 excluded. -/
theorem cert_30_29 : allExcluded 22 30 29 = true := by decide +kernel
/-- Certificate: leg 10, class 7, every cofactor below 22 excluded. -/
theorem cert_10_7 : allExcluded 22 10 7 = true := by decide +kernel
/-- Certificate: leg 10, class 13, every cofactor below 70 excluded. -/
theorem cert_10_13 : allExcluded 70 10 13 = true := by decide +kernel
/-- Certificate: leg 10, class 17, every cofactor below 2 excluded. -/
theorem cert_10_17 : allExcluded 2 10 17 = true := by decide +kernel
/-- Certificate: leg 10, class 23, every cofactor below 14 excluded. -/
theorem cert_10_23 : allExcluded 14 10 23 = true := by decide +kernel
/-- Certificate: leg 12, class 7, every cofactor below 48 excluded. -/
theorem cert_12_7 : allExcluded 48 12 7 = true := by decide +kernel
/-- Certificate: leg 12, class 13, every cofactor below 12 excluded. -/
theorem cert_12_13 : allExcluded 12 12 13 = true := by decide +kernel
/-- Certificate: leg 12, class 17, every cofactor below 4 excluded. -/
theorem cert_12_17 : allExcluded 4 12 17 = true := by decide +kernel
/-- Certificate: leg 12, class 23, every cofactor below 4 excluded. -/
theorem cert_12_23 : allExcluded 4 12 23 = true := by decide +kernel

/-! ### Floors: every kill of leg `b` of a gear `g ≡ c (mod 30)` has cofactor at least `K(b,c)` -/

/-- Floor, leg 28, class 1: `4 h ≤ (g - h)² + 28`. -/
theorem floor_28_1 {g h : ℕ} (hlt : h < g) (hcop : Nat.gcd h 30 = 1) (hc : g % 30 = 1)
    (hd : h ∣ g ^ 2 + 28) : 4 * h ≤ (g - h) ^ 2 + 28 :=
  floor_of_allExcluded cert_28_1 (by norm_num) hlt hcop hc hd

/-- Floor, leg 28, class 11: `32 h ≤ (g - h)² + 28`. -/
theorem floor_28_11 {g h : ℕ} (hlt : h < g) (hcop : Nat.gcd h 30 = 1) (hc : g % 30 = 11)
    (hd : h ∣ g ^ 2 + 28) : 32 * h ≤ (g - h) ^ 2 + 28 :=
  floor_of_allExcluded cert_28_11 (by norm_num) hlt hcop hc hd

/-- Floor, leg 28, class 19: `28 h ≤ (g - h)² + 28`. -/
theorem floor_28_19 {g h : ℕ} (hlt : h < g) (hcop : Nat.gcd h 30 = 1) (hc : g % 30 = 19)
    (hd : h ∣ g ^ 2 + 28) : 28 * h ≤ (g - h) ^ 2 + 28 :=
  floor_of_allExcluded cert_28_19 (by norm_num) hlt hcop hc hd

/-- Floor, leg 28, class 29: `32 h ≤ (g - h)² + 28`. -/
theorem floor_28_29 {g h : ℕ} (hlt : h < g) (hcop : Nat.gcd h 30 = 1) (hc : g % 30 = 29)
    (hd : h ∣ g ^ 2 + 28) : 32 * h ≤ (g - h) ^ 2 + 28 :=
  floor_of_allExcluded cert_28_29 (by norm_num) hlt hcop hc hd

/-- Floor, leg 30, class 1: `26 h ≤ (g - h)² + 30`. -/
theorem floor_30_1 {g h : ℕ} (hlt : h < g) (hcop : Nat.gcd h 30 = 1) (hc : g % 30 = 1)
    (hd : h ∣ g ^ 2 + 30) : 26 * h ≤ (g - h) ^ 2 + 30 :=
  floor_of_allExcluded cert_30_1 (by norm_num) hlt hcop hc hd

/-- Floor, leg 30, class 11: `6 h ≤ (g - h)² + 30`. -/
theorem floor_30_11 {g h : ℕ} (hlt : h < g) (hcop : Nat.gcd h 30 = 1) (hc : g % 30 = 11)
    (hd : h ∣ g ^ 2 + 30) : 6 * h ≤ (g - h) ^ 2 + 30 :=
  floor_of_allExcluded cert_30_11 (by norm_num) hlt hcop hc hd

/-- Floor, leg 30, class 19: `2 h ≤ (g - h)² + 30`. -/
theorem floor_30_19 {g h : ℕ} (hlt : h < g) (hcop : Nat.gcd h 30 = 1) (hc : g % 30 = 19)
    (hd : h ∣ g ^ 2 + 30) : 2 * h ≤ (g - h) ^ 2 + 30 :=
  floor_of_allExcluded cert_30_19 (by norm_num) hlt hcop hc hd

/-- Floor, leg 30, class 29: `22 h ≤ (g - h)² + 30`. -/
theorem floor_30_29 {g h : ℕ} (hlt : h < g) (hcop : Nat.gcd h 30 = 1) (hc : g % 30 = 29)
    (hd : h ∣ g ^ 2 + 30) : 22 * h ≤ (g - h) ^ 2 + 30 :=
  floor_of_allExcluded cert_30_29 (by norm_num) hlt hcop hc hd

/-- Floor, leg 10, class 7: `22 h ≤ (g - h)² + 10`. -/
theorem floor_10_7 {g h : ℕ} (hlt : h < g) (hcop : Nat.gcd h 30 = 1) (hc : g % 30 = 7)
    (hd : h ∣ g ^ 2 + 10) : 22 * h ≤ (g - h) ^ 2 + 10 :=
  floor_of_allExcluded cert_10_7 (by norm_num) hlt hcop hc hd

/-- Floor, leg 10, class 13: `70 h ≤ (g - h)² + 10`. -/
theorem floor_10_13 {g h : ℕ} (hlt : h < g) (hcop : Nat.gcd h 30 = 1) (hc : g % 30 = 13)
    (hd : h ∣ g ^ 2 + 10) : 70 * h ≤ (g - h) ^ 2 + 10 :=
  floor_of_allExcluded cert_10_13 (by norm_num) hlt hcop hc hd

/-- Floor, leg 10, class 17: `2 h ≤ (g - h)² + 10`. -/
theorem floor_10_17 {g h : ℕ} (hlt : h < g) (hcop : Nat.gcd h 30 = 1) (hc : g % 30 = 17)
    (hd : h ∣ g ^ 2 + 10) : 2 * h ≤ (g - h) ^ 2 + 10 :=
  floor_of_allExcluded cert_10_17 (by norm_num) hlt hcop hc hd

/-- Floor, leg 10, class 23: `14 h ≤ (g - h)² + 10`. -/
theorem floor_10_23 {g h : ℕ} (hlt : h < g) (hcop : Nat.gcd h 30 = 1) (hc : g % 30 = 23)
    (hd : h ∣ g ^ 2 + 10) : 14 * h ≤ (g - h) ^ 2 + 10 :=
  floor_of_allExcluded cert_10_23 (by norm_num) hlt hcop hc hd

/-- Floor, leg 12, class 7: `48 h ≤ (g - h)² + 12`. -/
theorem floor_12_7 {g h : ℕ} (hlt : h < g) (hcop : Nat.gcd h 30 = 1) (hc : g % 30 = 7)
    (hd : h ∣ g ^ 2 + 12) : 48 * h ≤ (g - h) ^ 2 + 12 :=
  floor_of_allExcluded cert_12_7 (by norm_num) hlt hcop hc hd

/-- Floor, leg 12, class 13: `12 h ≤ (g - h)² + 12`. -/
theorem floor_12_13 {g h : ℕ} (hlt : h < g) (hcop : Nat.gcd h 30 = 1) (hc : g % 30 = 13)
    (hd : h ∣ g ^ 2 + 12) : 12 * h ≤ (g - h) ^ 2 + 12 :=
  floor_of_allExcluded cert_12_13 (by norm_num) hlt hcop hc hd

/-- Floor, leg 12, class 17: `4 h ≤ (g - h)² + 12`. -/
theorem floor_12_17 {g h : ℕ} (hlt : h < g) (hcop : Nat.gcd h 30 = 1) (hc : g % 30 = 17)
    (hd : h ∣ g ^ 2 + 12) : 4 * h ≤ (g - h) ^ 2 + 12 :=
  floor_of_allExcluded cert_12_17 (by norm_num) hlt hcop hc hd

/-- Floor, leg 12, class 23: `4 h ≤ (g - h)² + 12`. -/
theorem floor_12_23 {g h : ℕ} (hlt : h < g) (hcop : Nat.gcd h 30 = 1) (hc : g % 30 = 23)
    (hd : h ∣ g ^ 2 + 12) : 4 * h ≤ (g - h) ^ 2 + 12 :=
  floor_of_allExcluded cert_12_23 (by norm_num) hlt hcop hc hd

/-! ### Class-refined kill-free windows (both legs of the class's case missed) -/

/-- Window, class 1: if `(g - h)² + 28 < 4h` then `h` misses both legs `g² + 28`, `g² + 30`. -/
theorem window_1 {g h : ℕ} (hlt : h < g) (hcop : Nat.gcd h 30 = 1) (hc : g % 30 = 1)
    (hw : (g - h) ^ 2 + 28 < 4 * h) : ¬ h ∣ g ^ 2 + 28 ∧ ¬ h ∣ g ^ 2 + 30 :=
  ⟨fun hd => by have := floor_28_1 hlt hcop hc hd; omega,
   fun hd => by have := floor_30_1 hlt hcop hc hd; omega⟩

/-- Window, class 11: if `(g - h)² + 30 < 6h` then `h` misses both legs `g² + 28`, `g² + 30`. -/
theorem window_11 {g h : ℕ} (hlt : h < g) (hcop : Nat.gcd h 30 = 1) (hc : g % 30 = 11)
    (hw : (g - h) ^ 2 + 30 < 6 * h) : ¬ h ∣ g ^ 2 + 28 ∧ ¬ h ∣ g ^ 2 + 30 :=
  ⟨fun hd => by have := floor_28_11 hlt hcop hc hd; omega,
   fun hd => by have := floor_30_11 hlt hcop hc hd; omega⟩

/-- Window, class 19: if `(g - h)² + 30 < 2h` then `h` misses both legs `g² + 28`, `g² + 30`. -/
theorem window_19 {g h : ℕ} (hlt : h < g) (hcop : Nat.gcd h 30 = 1) (hc : g % 30 = 19)
    (hw : (g - h) ^ 2 + 30 < 2 * h) : ¬ h ∣ g ^ 2 + 28 ∧ ¬ h ∣ g ^ 2 + 30 :=
  ⟨fun hd => by have := floor_28_19 hlt hcop hc hd; omega,
   fun hd => by have := floor_30_19 hlt hcop hc hd; omega⟩

/-- Window, class 29: if `(g - h)² + 30 < 22h` then `h` misses both legs `g² + 28`, `g² + 30`. -/
theorem window_29 {g h : ℕ} (hlt : h < g) (hcop : Nat.gcd h 30 = 1) (hc : g % 30 = 29)
    (hw : (g - h) ^ 2 + 30 < 22 * h) : ¬ h ∣ g ^ 2 + 28 ∧ ¬ h ∣ g ^ 2 + 30 :=
  ⟨fun hd => by have := floor_28_29 hlt hcop hc hd; omega,
   fun hd => by have := floor_30_29 hlt hcop hc hd; omega⟩

/-- Window, class 7: if `(g - h)² + 10 < 22h` then `h` misses both legs `g² + 10`, `g² + 12`. -/
theorem window_7 {g h : ℕ} (hlt : h < g) (hcop : Nat.gcd h 30 = 1) (hc : g % 30 = 7)
    (hw : (g - h) ^ 2 + 10 < 22 * h) : ¬ h ∣ g ^ 2 + 10 ∧ ¬ h ∣ g ^ 2 + 12 :=
  ⟨fun hd => by have := floor_10_7 hlt hcop hc hd; omega,
   fun hd => by have := floor_12_7 hlt hcop hc hd; omega⟩

/-- Window, class 13: if `(g - h)² + 12 < 12h` then `h` misses both legs `g² + 10`, `g² + 12`. -/
theorem window_13 {g h : ℕ} (hlt : h < g) (hcop : Nat.gcd h 30 = 1) (hc : g % 30 = 13)
    (hw : (g - h) ^ 2 + 12 < 12 * h) : ¬ h ∣ g ^ 2 + 10 ∧ ¬ h ∣ g ^ 2 + 12 :=
  ⟨fun hd => by have := floor_10_13 hlt hcop hc hd; omega,
   fun hd => by have := floor_12_13 hlt hcop hc hd; omega⟩

/-- Window, class 17: if `(g - h)² + 10 < 2h` then `h` misses both legs `g² + 10`, `g² + 12`. -/
theorem window_17 {g h : ℕ} (hlt : h < g) (hcop : Nat.gcd h 30 = 1) (hc : g % 30 = 17)
    (hw : (g - h) ^ 2 + 10 < 2 * h) : ¬ h ∣ g ^ 2 + 10 ∧ ¬ h ∣ g ^ 2 + 12 :=
  ⟨fun hd => by have := floor_10_17 hlt hcop hc hd; omega,
   fun hd => by have := floor_12_17 hlt hcop hc hd; omega⟩

/-- Window, class 23: if `(g - h)² + 12 < 4h` then `h` misses both legs `g² + 10`, `g² + 12`. -/
theorem window_23 {g h : ℕ} (hlt : h < g) (hcop : Nat.gcd h 30 = 1) (hc : g % 30 = 23)
    (hw : (g - h) ^ 2 + 12 < 4 * h) : ¬ h ∣ g ^ 2 + 10 ∧ ¬ h ∣ g ^ 2 + 12 :=
  ⟨fun hd => by have := floor_10_23 hlt hcop hc hd; omega,
   fun hd => by have := floor_12_23 hlt hcop hc hd; omega⟩

/-! ### Each window is sharp: a prime kill on its boundary in every class -/

/-- Class 1 is sharp: `23 ∣ 31² + 28` with `(31 - 23)² + 28 = 4 · 23`. -/
theorem sharp_1 : Nat.Prime 23 ∧ Nat.Prime 31 ∧ 31 % 30 = 1 ∧ 23 ∣ 31 ^ 2 + 28 ∧
    (31 - 23) ^ 2 + 28 = 4 * 23 := by norm_num
/-- Class 11 is sharp: `29 ∣ 41² + 30` with `(41 - 29)² + 30 = 6 · 29`. -/
theorem sharp_11 : Nat.Prime 29 ∧ Nat.Prime 41 ∧ 41 % 30 = 11 ∧ 29 ∣ 41 ^ 2 + 30 ∧
    (41 - 29) ^ 2 + 30 = 6 * 29 := by norm_num
/-- Class 19 is sharp: `17 ∣ 19² + 30` with `(19 - 17)² + 30 = 2 · 17`. -/
theorem sharp_19 : Nat.Prime 17 ∧ Nat.Prime 19 ∧ 19 % 30 = 19 ∧ 17 ∣ 19 ^ 2 + 30 ∧
    (19 - 17) ^ 2 + 30 = 2 * 17 := by norm_num
/-- Class 29 is sharp: `13 ∣ 29² + 30` with `(29 - 13)² + 30 = 22 · 13`. -/
theorem sharp_29 : Nat.Prime 13 ∧ Nat.Prime 29 ∧ 29 % 30 = 29 ∧ 13 ∣ 29 ^ 2 + 30 ∧
    (29 - 13) ^ 2 + 30 = 22 * 13 := by norm_num
/-- Class 7 is sharp: `11093 ∣ 11587² + 10` with `(11587 - 11093)² + 10 = 22 · 11093`. -/
theorem sharp_7 : Nat.Prime 11093 ∧ Nat.Prime 11587 ∧ 11587 % 30 = 7 ∧ 11093 ∣ 11587 ^ 2 + 10 ∧
    (11587 - 11093) ^ 2 + 10 = 22 * 11093 := by norm_num
/-- Class 13 is sharp: `3469 ∣ 3673² + 12` with `(3673 - 3469)² + 12 = 12 · 3469`. -/
theorem sharp_13 : Nat.Prime 3469 ∧ Nat.Prime 3673 ∧ 3673 % 30 = 13 ∧ 3469 ∣ 3673 ^ 2 + 12 ∧
    (3673 - 3469) ^ 2 + 12 = 12 * 3469 := by norm_num
/-- Class 17 is sharp: `13 ∣ 17² + 10` with `(17 - 13)² + 10 = 2 · 13`. -/
theorem sharp_17 : Nat.Prime 13 ∧ Nat.Prime 17 ∧ 17 % 30 = 17 ∧ 13 ∣ 17 ^ 2 + 10 ∧
    (17 - 13) ^ 2 + 10 = 2 * 13 := by norm_num
/-- Class 23 is sharp: `67 ∣ 83² + 12` with `(83 - 67)² + 12 = 4 · 67`. -/
theorem sharp_23 : Nat.Prime 67 ∧ Nat.Prime 83 ∧ 83 % 30 = 23 ∧ 67 ∣ 83 ^ 2 + 12 ∧
    (83 - 67) ^ 2 + 12 = 4 * 67 := by norm_num

/-! ### The neighbour rule -/

/-- Monotone form: a window `(g - h')² + b < K h'` that holds at a lower neighbour `h'` holds at
every `h` in `[h', g)`, for all `b`, `K`. -/
theorem window_mono {g h h' b K : ℕ} (hh' : h' ≤ h) (hlt : h < g)
    (hw : (g - h') ^ 2 + b < K * h') : (g - h) ^ 2 + b < K * h := by
  have hsq : (g - h) ^ 2 ≤ (g - h') ^ 2 := Nat.pow_le_pow_left (by omega) 2
  have hK : K * h' ≤ K * h := Nat.mul_le_mul_left K hh'
  omega

/-- Neighbour rule, class 1: if `(g - h')² + 28 < 4h'`, every `h` prime to 30 in `[h', g)` misses
both legs `g² + 28`, `g² + 30`. -/
theorem neighbour_1 {g h h' : ℕ} (hc : g % 30 = 1) (hw : (g - h') ^ 2 + 28 < 4 * h')
    (hh' : h' ≤ h) (hlt : h < g) (hcop : Nat.gcd h 30 = 1) :
    ¬ h ∣ g ^ 2 + 28 ∧ ¬ h ∣ g ^ 2 + 30 :=
  window_1 hlt hcop hc (window_mono hh' hlt hw)

/-- Neighbour rule, class 11: if `(g - h')² + 30 < 6h'`, every `h` prime to 30 in `[h', g)`
misses both legs `g² + 28`, `g² + 30`. -/
theorem neighbour_11 {g h h' : ℕ} (hc : g % 30 = 11) (hw : (g - h') ^ 2 + 30 < 6 * h')
    (hh' : h' ≤ h) (hlt : h < g) (hcop : Nat.gcd h 30 = 1) :
    ¬ h ∣ g ^ 2 + 28 ∧ ¬ h ∣ g ^ 2 + 30 :=
  window_11 hlt hcop hc (window_mono hh' hlt hw)

/-- Neighbour rule, class 19: if `(g - h')² + 30 < 2h'`, every `h` prime to 30 in `[h', g)`
misses both legs `g² + 28`, `g² + 30`. -/
theorem neighbour_19 {g h h' : ℕ} (hc : g % 30 = 19) (hw : (g - h') ^ 2 + 30 < 2 * h')
    (hh' : h' ≤ h) (hlt : h < g) (hcop : Nat.gcd h 30 = 1) :
    ¬ h ∣ g ^ 2 + 28 ∧ ¬ h ∣ g ^ 2 + 30 :=
  window_19 hlt hcop hc (window_mono hh' hlt hw)

/-- Neighbour rule, class 29: if `(g - h')² + 30 < 22h'`, every `h` prime to 30 in `[h', g)`
misses both legs `g² + 28`, `g² + 30`. -/
theorem neighbour_29 {g h h' : ℕ} (hc : g % 30 = 29) (hw : (g - h') ^ 2 + 30 < 22 * h')
    (hh' : h' ≤ h) (hlt : h < g) (hcop : Nat.gcd h 30 = 1) :
    ¬ h ∣ g ^ 2 + 28 ∧ ¬ h ∣ g ^ 2 + 30 :=
  window_29 hlt hcop hc (window_mono hh' hlt hw)

/-- Neighbour rule, class 7: if `(g - h')² + 10 < 22h'`, every `h` prime to 30 in `[h', g)`
misses both legs `g² + 10`, `g² + 12`. -/
theorem neighbour_7 {g h h' : ℕ} (hc : g % 30 = 7) (hw : (g - h') ^ 2 + 10 < 22 * h')
    (hh' : h' ≤ h) (hlt : h < g) (hcop : Nat.gcd h 30 = 1) :
    ¬ h ∣ g ^ 2 + 10 ∧ ¬ h ∣ g ^ 2 + 12 :=
  window_7 hlt hcop hc (window_mono hh' hlt hw)

/-- Neighbour rule, class 13: if `(g - h')² + 12 < 12h'`, every `h` prime to 30 in `[h', g)`
misses both legs `g² + 10`, `g² + 12`. -/
theorem neighbour_13 {g h h' : ℕ} (hc : g % 30 = 13) (hw : (g - h') ^ 2 + 12 < 12 * h')
    (hh' : h' ≤ h) (hlt : h < g) (hcop : Nat.gcd h 30 = 1) :
    ¬ h ∣ g ^ 2 + 10 ∧ ¬ h ∣ g ^ 2 + 12 :=
  window_13 hlt hcop hc (window_mono hh' hlt hw)

/-- Neighbour rule, class 17: if `(g - h')² + 10 < 2h'`, every `h` prime to 30 in `[h', g)`
misses both legs `g² + 10`, `g² + 12`. -/
theorem neighbour_17 {g h h' : ℕ} (hc : g % 30 = 17) (hw : (g - h') ^ 2 + 10 < 2 * h')
    (hh' : h' ≤ h) (hlt : h < g) (hcop : Nat.gcd h 30 = 1) :
    ¬ h ∣ g ^ 2 + 10 ∧ ¬ h ∣ g ^ 2 + 12 :=
  window_17 hlt hcop hc (window_mono hh' hlt hw)

/-- Neighbour rule, class 23: if `(g - h')² + 12 < 4h'`, every `h` prime to 30 in `[h', g)`
misses both legs `g² + 10`, `g² + 12`. -/
theorem neighbour_23 {g h h' : ℕ} (hc : g % 30 = 23) (hw : (g - h') ^ 2 + 12 < 4 * h')
    (hh' : h' ≤ h) (hlt : h < g) (hcop : Nat.gcd h 30 = 1) :
    ¬ h ∣ g ^ 2 + 10 ∧ ¬ h ∣ g ^ 2 + 12 :=
  window_23 hlt hcop hc (window_mono hh' hlt hw)

end RangeLine
