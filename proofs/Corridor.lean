/-
The (5,7) corridor: the 32-cap on prime-adjacent runs, and the twin-product
pin in slot coordinates.

**The 32-cap.** Gears 5 and 7 have split classes at `k ≡ 1` and `k ≡ 34
(mod 35)`: at `k ≡ 1`, 5 divides the lower member and 7 the upper; at
`k ≡ 34`, mirrored. Beyond the twin (5,7) itself (slot 1), both members are
composite at every slot of either class. The largest gap in the class set
`{1, 34} mod 35` is 33, so any 33 consecutive slots contain a class slot -
hence a slot with both members composite. A saturated run (every slot
carrying at least one prime member) therefore never exceeds 32 slots:
unconditional, at every scale, from two gears alone.

**The pin, unified.** `Polignac.twin_product_slot` places a twin pair's
product `p(p+2)` at slot `u(p+1)` (where `6u = p+1`), with both gears
striking its lower member. `Placement.slotOf` computes slots from members.
This file proves they agree - `slotOf (p*(p+2)) = u*(p+1) = 6u²` - so the
two files' objects are interchangeable in every downstream statement.
-/

import Mathlib.Tactic.Ring
import Placement
import Polignac

namespace Corridor

/-! ## The 32-cap -/

/-- The class set `{1, 34} mod 35` has no gap longer than 33: every 33
consecutive slots contain a class slot. -/
theorem exists_class_in_run (a : ℕ) :
    ∃ k, a ≤ k ∧ k < a + 33 ∧ (k % 35 = 1 ∨ k % 35 = 34) := by
  rcases Nat.lt_or_ge (a % 35) 2 with h | h
  · exact ⟨a + (1 - a % 35), by omega, by omega, by omega⟩
  · exact ⟨a + (34 - a % 35), by omega, by omega, by omega⟩

/-- A number with a proper divisor `≥ 2` is not prime. -/
theorem not_prime_of_proper_dvd {d m : ℕ} (h2 : 2 ≤ d) (hne : d ≠ m)
    (hdvd : d ∣ m) : ¬ m.Prime := by
  intro hp
  rcases hp.eq_one_or_self_of_dvd d hdvd with h | h <;> omega

/-- On the split classes of gears 5 and 7, both members are composite
(beyond slot 1, the twin (5,7) itself): at `k ≡ 1 (mod 35)` gear 5 takes the
lower member and gear 7 the upper; at `k ≡ 34` the mirror. -/
theorem both_composite_of_class {k : ℕ} (hk : 2 ≤ k)
    (h : k % 35 = 1 ∨ k % 35 = 34) :
    ¬ (Census.lo k).Prime ∧ ¬ (Census.hi k).Prime := by
  rcases h with h | h
  · constructor
    · refine not_prime_of_proper_dvd (d := 5) (by omega) ?_ ?_
      · simp only [Census.lo]; omega
      · simp only [Census.lo]; omega
    · refine not_prime_of_proper_dvd (d := 7) (by omega) ?_ ?_
      · simp only [Census.hi]; omega
      · simp only [Census.hi]; omega
  · constructor
    · refine not_prime_of_proper_dvd (d := 7) (by omega) ?_ ?_
      · simp only [Census.lo]; omega
      · simp only [Census.lo]; omega
    · refine not_prime_of_proper_dvd (d := 5) (by omega) ?_ ?_
      · simp only [Census.hi]; omega
      · simp only [Census.hi]; omega

/-- **The 32-cap, existence form.** Any 33 consecutive slots (from slot 2
on) contain a slot with both members composite. -/
theorem both_composite_in_run {a : ℕ} (ha : 2 ≤ a) :
    ∃ k, a ≤ k ∧ k < a + 33 ∧
      ¬ (Census.lo k).Prime ∧ ¬ (Census.hi k).Prime := by
  obtain ⟨k, h1, h2, h3⟩ := exists_class_in_run a
  obtain ⟨hlo, hhi⟩ := both_composite_of_class (by omega) h3
  exact ⟨k, h1, h2, hlo, hhi⟩

/-- Census form: every 33-slot window from slot 2 on holds a double slot. -/
theorem double_slot_in_run {a : ℕ} (ha : 2 ≤ a) :
    ∃ k, a ≤ k ∧ k < a + 33 ∧ Census.slotComps k = 2 := by
  obtain ⟨k, h1, h2, hlo, hhi⟩ := both_composite_in_run ha
  exact ⟨k, h1, h2, by simp [Census.slotComps, hlo, hhi]⟩

/-- **The 32-cap.** A prime-adjacent run - consecutive slots each carrying
at least one prime member - starting at slot 2 or later has length at most
32. Unconditional: gears 5 and 7 enforce it at every scale. -/
theorem prime_adjacent_run_le {a L : ℕ} (ha : 2 ≤ a)
    (hrun : ∀ k, a ≤ k → k < a + L →
      (Census.lo k).Prime ∨ (Census.hi k).Prime) :
    L ≤ 32 := by
  by_contra hL
  obtain ⟨k, h1, h2, hlo, hhi⟩ := both_composite_in_run ha
  rcases hrun k h1 (by omega) with h | h
  · exact hlo h
  · exact hhi h

/-! ## The twin-product pin in slot coordinates -/

/-- **Slot of the product.** Placement's slot function lands the twin
product exactly where Polignac's product-slot theorem puts it:
`slotOf (p*(p+2)) = u*(p+1)`. -/
theorem product_slotOf {p u : ℕ} (hu : 6 * u = p + 1) :
    Placement.slotOf (p * (p + 2)) = u * (p + 1) := by
  simp only [Placement.slotOf]
  have h2 : p * (p + 2) + 1 = (p + 1) * (p + 1) := by ring
  rw [h2, ← hu]
  rw [show (6 * u) * (6 * u) = 6 * (u * (6 * u)) by ring]
  exact Nat.mul_div_cancel_left _ (by omega)

/-- The square form: the product slot is `6u²` - six times the square of
the pair's own slot. -/
theorem product_slotOf_sq {p u : ℕ} (hu : 6 * u = p + 1) :
    Placement.slotOf (p * (p + 2)) = 6 * (u * u) := by
  rw [product_slotOf hu, ← hu]
  ring

/-- **The pin, unified.** The semiprime `p(p+2)` is the LOWER member of
slot `u(p+1) = slotOf (p*(p+2))`, and both gears of the pair strike it
there - Polignac's `twin_product_slot` re-expressed through Placement's
coordinates, making the two files' objects interchangeable. -/
theorem twin_product_pin {p u : ℕ} (hu : 6 * u = p + 1) :
    Placement.slotOf (p * (p + 2)) = u * (p + 1) ∧
      Census.lo (u * (p + 1)) = p * (p + 2) ∧
      p ∣ Census.lo (u * (p + 1)) ∧ (p + 2) ∣ Census.lo (u * (p + 1)) := by
  obtain ⟨e, hdp, hdp2⟩ := Polignac.twin_product_slot hu
  have hlo : Census.lo (u * (p + 1)) = p * (p + 2) := by
    simp only [Census.lo]; exact e
  refine ⟨product_slotOf hu, hlo, ?_, ?_⟩
  · simp only [Census.lo]; exact hdp
  · simp only [Census.lo]; exact hdp2

end Corridor
