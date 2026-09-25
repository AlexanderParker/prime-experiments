import Mathlib.NumberTheory.Primorial
import Mathlib.Tactic
import RangeCentre

/-!
# The mirror pair of copies

Copy `k` carries the pair `(30k - 1, 30k + 1)` (see `RangeCentre`).  Fix `q ≥ 5` and let
`Q = q#` be the primorial, so `30 ∣ Q`, and let `M = Q / 30`, the product of the primes
`7 ≤ p ≤ q`.  As `Q` is squarefree, `M` is odd.  For an odd offset `d < M` the *mirror pair* of
copies is

* `s  = (M - d) / 2` (`mirrorLo q d`), and
* `s' = (M + d) / 2` (`mirrorHi q d`),

so that `s + s' = M`, `s' = s + d` and `s ≥ 1`.  The two copies sit symmetrically about
`M / 2`, and their legs sit symmetrically about `Q / 2`:  `30s = Q/2 - 15d` and
`30s' = Q/2 + 15d`.

This file proves:

* (a) `legs_sum`: the cross legs sum to `Q`:
  `(30s - 1) + (30s' + 1) = Q` and `(30s + 1) + (30s' - 1) = Q`;
  `legs_diff`: the like legs sum to `Q ∓ 2`:
  `(30s' - 1) + (30s - 1) = Q - 2` and `(30s' + 1) + (30s + 1) = Q + 2`.
* (b) `shared_striker_cases`: a common divisor of a leg of `s` and a leg of `s'` divides the sum
  of the two legs, which is `Q - 2`, `Q` or `Q + 2` according to the two legs;
  `shared_striker`: the disjunction `g ∣ Q - 2 ∨ g ∣ Q ∨ g ∣ Q + 2`;
  `shared_striker_off`: if moreover `g ∤ Q` then `g ∣ (Q - 2)(Q + 2)`.  None of these needs
  `g` prime.  For a prime gear, `shared_striker_copy` / `shared_striker_copy_off` restate this for
  `StrikesCopy`, and `shared_striker_above` gives, for a prime `g > q` striking both copies,
  `g ∣ Q - 2 ∨ g ∣ Q + 2`.
* (c) `height_split`: `(30s + 1) + (30s' + 1) = Q + 2`, and `acts_on_at_most_one`: a gear `g`
  with `2g² > Q + 2` cannot have both `g² ≤ 30s + 1` and `g² ≤ 30s' + 1`.
* (d) `four_leg_product`: over `ℤ`, with `a = Q / 2`,
  `(30s - 1)(30s + 1)(30s' - 1)(30s' + 1) = (a² - (15d + 1)²)(a² - (15d - 1)²)`;
  `four_leg_product_nat` states the same for the natural-number product of the four legs.
-/

namespace RangeLine

/-- The mirror modulus `M = q# / 30`; for `q ≥ 5` it is the product of the primes `7 ≤ p ≤ q`. -/
def mirrorM (q : ℕ) : ℕ := primorial q / 30

/-- The lower copy of the mirror pair at offset `d`: `s = (M - d) / 2`. -/
def mirrorLo (q d : ℕ) : ℕ := (mirrorM q - d) / 2

/-- The upper copy of the mirror pair at offset `d`: `s' = (M + d) / 2`. -/
def mirrorHi (q d : ℕ) : ℕ := (mirrorM q + d) / 2

/-- The primorial of `5` is `2 · 3 · 5 = 30`. -/
theorem primorial_five : primorial 5 = 30 := by decide

/-- For `q ≥ 5`, `30` divides the primorial `q#`. -/
theorem thirty_dvd_primorial {q : ℕ} (hq : 5 ≤ q) : 30 ∣ primorial q :=
  primorial_five ▸ primorial_dvd_primorial hq

/-- For `q ≥ 5`, `30 · M = q#`. -/
theorem thirty_mul_mirrorM {q : ℕ} (hq : 5 ≤ q) : 30 * mirrorM q = primorial q :=
  Nat.mul_div_cancel' (thirty_dvd_primorial hq)

/-- For `q ≥ 5`, `M = q# / 30` is odd, since `q#` is squarefree and so `4 ∤ q#`. -/
theorem odd_mirrorM {q : ℕ} (hq : 5 ≤ q) : Odd (mirrorM q) := by
  rw [Nat.odd_iff]
  by_contra h
  have h2 : 2 ∣ mirrorM q := Nat.dvd_of_mod_eq_zero (by omega)
  obtain ⟨c, hc⟩ := h2
  have h4 : 2 * 2 ∣ primorial q := ⟨15 * c, by rw [← thirty_mul_mirrorM hq, hc]; ring⟩
  have hu := Nat.isUnit_iff.1 (squarefree_primorial q 2 h4)
  omega

/-- The two copies of the mirror pair sum to `M`: `s + s' = M` (for `q ≥ 5`, odd `d < M`). -/
theorem mirror_sum {q d : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q) :
    mirrorLo q d + mirrorHi q d = mirrorM q := by
  obtain ⟨m, hm⟩ := odd_mirrorM hq
  obtain ⟨e, he⟩ := hd
  unfold mirrorLo mirrorHi
  omega

/-- The upper copy is the lower copy shifted by the offset: `s' = s + d`. -/
theorem mirrorHi_eq {q d : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q) :
    mirrorHi q d = mirrorLo q d + d := by
  obtain ⟨m, hm⟩ := odd_mirrorM hq
  obtain ⟨e, he⟩ := hd
  unfold mirrorLo mirrorHi
  omega

/-- The lower copy is a genuine copy: `1 ≤ s` (as `M - d` is even and positive). -/
theorem one_le_mirrorLo {q d : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q) :
    1 ≤ mirrorLo q d := by
  obtain ⟨m, hm⟩ := odd_mirrorM hq
  obtain ⟨e, he⟩ := hd
  unfold mirrorLo
  omega

/-- The legs of the pair in terms of `Q`: `30s + 30s' = Q`. -/
theorem thirty_mirror_sum {q d : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q) :
    30 * mirrorLo q d + 30 * mirrorHi q d = primorial q := by
  rw [← thirty_mul_mirrorM hq, ← mirror_sum hq hd hdM]
  ring

/-- **(a) Cross legs sum to `Q`.**  `(30s - 1) + (30s' + 1) = Q` and
`(30s + 1) + (30s' - 1) = Q`. -/
theorem legs_sum {q d : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q) :
    (30 * mirrorLo q d - 1) + (30 * mirrorHi q d + 1) = primorial q ∧
      (30 * mirrorLo q d + 1) + (30 * mirrorHi q d - 1) = primorial q := by
  have h := thirty_mirror_sum hq hd hdM
  have h1 := one_le_mirrorLo hq hd hdM
  have h2 := mirrorHi_eq hq hd hdM
  omega

/-- **(a) Like legs sum to `Q ∓ 2`.**  `(30s' - 1) + (30s - 1) = Q - 2` and
`(30s' + 1) + (30s + 1) = Q + 2`. -/
theorem legs_diff {q d : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q) :
    (30 * mirrorHi q d - 1) + (30 * mirrorLo q d - 1) = primorial q - 2 ∧
      (30 * mirrorHi q d + 1) + (30 * mirrorLo q d + 1) = primorial q + 2 := by
  have h := thirty_mirror_sum hq hd hdM
  have h1 := one_le_mirrorLo hq hd hdM
  have h2 := mirrorHi_eq hq hd hdM
  omega

/-- **(b) Shared striker, case by case.**  A common divisor `g` of a leg of `s` and a leg of `s'`
divides the sum of the two legs: `Q - 2` for the two minus legs, `Q` for the two cross pairs,
`Q + 2` for the two plus legs. -/
theorem shared_striker_cases {q d g : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q) :
    (g ∣ 30 * mirrorLo q d - 1 → g ∣ 30 * mirrorHi q d - 1 → g ∣ primorial q - 2) ∧
      (g ∣ 30 * mirrorLo q d - 1 → g ∣ 30 * mirrorHi q d + 1 → g ∣ primorial q) ∧
      (g ∣ 30 * mirrorLo q d + 1 → g ∣ 30 * mirrorHi q d - 1 → g ∣ primorial q) ∧
      (g ∣ 30 * mirrorLo q d + 1 → g ∣ 30 * mirrorHi q d + 1 → g ∣ primorial q + 2) := by
  obtain ⟨hA, hB⟩ := legs_sum hq hd hdM
  obtain ⟨hC, hD⟩ := legs_diff hq hd hdM
  refine ⟨fun h1 h2 => ?_, fun h1 h2 => ?_, fun h1 h2 => ?_, fun h1 h2 => ?_⟩
  · rw [← hC]; exact dvd_add h2 h1
  · rw [← hA]; exact dvd_add h1 h2
  · rw [← hB]; exact dvd_add h1 h2
  · rw [← hD]; exact dvd_add h2 h1

/-- **(b) Shared striker.**  If `g` divides a leg of `s` and a leg of `s'`, then `g` divides
`Q - 2`, `Q` or `Q + 2`. -/
theorem shared_striker {q d g : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q)
    (hs : g ∣ 30 * mirrorLo q d - 1 ∨ g ∣ 30 * mirrorLo q d + 1)
    (hs' : g ∣ 30 * mirrorHi q d - 1 ∨ g ∣ 30 * mirrorHi q d + 1) :
    g ∣ primorial q - 2 ∨ g ∣ primorial q ∨ g ∣ primorial q + 2 := by
  obtain ⟨c1, c2, c3, c4⟩ := shared_striker_cases (g := g) hq hd hdM
  rcases hs with h1 | h1 <;> rcases hs' with h2 | h2
  · exact Or.inl (c1 h1 h2)
  · exact Or.inr (Or.inl (c2 h1 h2))
  · exact Or.inr (Or.inl (c3 h1 h2))
  · exact Or.inr (Or.inr (c4 h1 h2))

/-- **(b) Shared striker off `Q`.**  If `g` divides a leg of `s` and a leg of `s'` and `g ∤ Q`,
then `g ∣ (Q - 2)(Q + 2)`. -/
theorem shared_striker_off {q d g : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q)
    (hs : g ∣ 30 * mirrorLo q d - 1 ∨ g ∣ 30 * mirrorLo q d + 1)
    (hs' : g ∣ 30 * mirrorHi q d - 1 ∨ g ∣ 30 * mirrorHi q d + 1)
    (hgQ : ¬ g ∣ primorial q) :
    g ∣ (primorial q - 2) * (primorial q + 2) := by
  rcases shared_striker hq hd hdM hs hs' with h | h | h
  · exact Dvd.dvd.mul_right h _
  · exact absurd h hgQ
  · exact Dvd.dvd.mul_left h _

/-- A prime strikes copy `k` exactly when it divides one of the two legs. -/
theorem strikesCopy_iff_leg {g k : ℕ} (hg : g.Prime) :
    StrikesCopy g k ↔ g ∣ 30 * k - 1 ∨ g ∣ 30 * k + 1 :=
  hg.dvd_mul

/-- **(b) Shared striker for a prime gear.**  A prime `g` striking both copies of the mirror pair
divides `Q - 2`, `Q` or `Q + 2`. -/
theorem shared_striker_copy {q d g : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q)
    (hg : g.Prime) (hs : StrikesCopy g (mirrorLo q d)) (hs' : StrikesCopy g (mirrorHi q d)) :
    g ∣ primorial q - 2 ∨ g ∣ primorial q ∨ g ∣ primorial q + 2 :=
  shared_striker hq hd hdM ((strikesCopy_iff_leg hg).1 hs) ((strikesCopy_iff_leg hg).1 hs')

/-- **(b) Shared striker off `Q`, prime gear.**  A prime `g ∤ Q` striking both copies divides
`(Q - 2)(Q + 2)`, hence one of `Q - 2`, `Q + 2`. -/
theorem shared_striker_copy_off {q d g : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q)
    (hg : g.Prime) (hs : StrikesCopy g (mirrorLo q d)) (hs' : StrikesCopy g (mirrorHi q d))
    (hgQ : ¬ g ∣ primorial q) :
    g ∣ (primorial q - 2) * (primorial q + 2) ∧ (g ∣ primorial q - 2 ∨ g ∣ primorial q + 2) := by
  have h := shared_striker_off hq hd hdM ((strikesCopy_iff_leg hg).1 hs)
    ((strikesCopy_iff_leg hg).1 hs') hgQ
  exact ⟨h, hg.dvd_mul.1 h⟩

/-- **(b) Shared striker above `q`.**  A prime `g > q` does not divide `Q = q#`; so if it strikes
both copies of the mirror pair, it divides `Q - 2` or `Q + 2`. -/
theorem shared_striker_above {q d g : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q)
    (hg : g.Prime) (hgq : q < g) (hs : StrikesCopy g (mirrorLo q d))
    (hs' : StrikesCopy g (mirrorHi q d)) :
    g ∣ primorial q - 2 ∨ g ∣ primorial q + 2 := by
  have hgQ : ¬ g ∣ primorial q := by
    rw [hg.dvd_primorial_iff]
    omega
  exact (shared_striker_copy_off hq hd hdM hg hs hs' hgQ).2

/-- **(c) Height split.**  The two upper legs sum to `Q + 2`: `(30s + 1) + (30s' + 1) = Q + 2`. -/
theorem height_split {q d : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q) :
    (30 * mirrorLo q d + 1) + (30 * mirrorHi q d + 1) = primorial q + 2 := by
  have h := thirty_mirror_sum hq hd hdM
  omega

/-- **(c) A tall gear acts on at most one copy of the pair.**  If `2g² > Q + 2`, then `g²` does
not fit under both upper legs: `¬ (g² ≤ 30s + 1 ∧ g² ≤ 30s' + 1)`. -/
theorem acts_on_at_most_one {q d g : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q)
    (hg : primorial q + 2 < 2 * g ^ 2) :
    ¬ (g ^ 2 ≤ 30 * mirrorLo q d + 1 ∧ g ^ 2 ≤ 30 * mirrorHi q d + 1) := by
  rintro ⟨h1, h2⟩
  have h := height_split hq hd hdM
  omega

/-- The half-primorial as a natural number: `Q / 2 = 30s + 15d`. -/
theorem half_primorial_eq {q d : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q) :
    primorial q / 2 = 30 * mirrorLo q d + 15 * d := by
  have h := thirty_mul_mirrorM hq
  have hs := mirror_sum hq hd hdM
  have h2 := mirrorHi_eq hq hd hdM
  omega

/-- **(d) Four-leg product.**  Over `ℤ`, with `a = Q / 2`,
`(30s - 1)(30s + 1)(30s' - 1)(30s' + 1) = (a² - (15d + 1)²)(a² - (15d - 1)²)`. -/
theorem four_leg_product {q d : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q) :
    (30 * (mirrorLo q d : ℤ) - 1) * (30 * (mirrorLo q d : ℤ) + 1) *
        (30 * (mirrorHi q d : ℤ) - 1) * (30 * (mirrorHi q d : ℤ) + 1) =
      (((primorial q / 2 : ℕ) : ℤ) ^ 2 - (15 * (d : ℤ) + 1) ^ 2) *
        (((primorial q / 2 : ℕ) : ℤ) ^ 2 - (15 * (d : ℤ) - 1) ^ 2) := by
  have ea : ((primorial q / 2 : ℕ) : ℤ) = 30 * (mirrorLo q d : ℤ) + 15 * d := by
    exact_mod_cast half_primorial_eq hq hd hdM
  have eh : (mirrorHi q d : ℤ) = (mirrorLo q d : ℤ) + d := by
    exact_mod_cast mirrorHi_eq hq hd hdM
  rw [ea, eh]
  ring

/-- **(d) Four-leg product, natural-number legs.**  The product of the four legs, taken in `ℕ`
and cast to `ℤ`, equals `(a² - (15d + 1)²)(a² - (15d - 1)²)` with `a = Q / 2`. -/
theorem four_leg_product_nat {q d : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q) :
    (((30 * mirrorLo q d - 1) * (30 * mirrorLo q d + 1) *
        (30 * mirrorHi q d - 1) * (30 * mirrorHi q d + 1) : ℕ) : ℤ) =
      (((primorial q / 2 : ℕ) : ℤ) ^ 2 - (15 * (d : ℤ) + 1) ^ 2) *
        (((primorial q / 2 : ℕ) : ℤ) ^ 2 - (15 * (d : ℤ) - 1) ^ 2) := by
  have h1 : 1 ≤ 30 * mirrorLo q d := by
    have := one_le_mirrorLo hq hd hdM
    omega
  have h2 : 1 ≤ 30 * mirrorHi q d := by
    have := one_le_mirrorLo hq hd hdM
    have := mirrorHi_eq hq hd hdM
    omega
  push_cast [Nat.cast_sub h1, Nat.cast_sub h2]
  exact four_leg_product hq hd hdM

end RangeLine
