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

/-! ## The exposed set and the endpoint law

A slot is EXPOSED to the (5,7) corridor when neither gear divides either
member. For `k ≥ 1` this is exactly a 15-residue condition mod 35 - the set
`E` below (3 free residues mod 5 x 5 free residues mod 7). Openings (gap
endpoints) are exposed slots, so both endpoints of any machine gap land in
`E`: the endpoint law. Verified against research/topgap_endpoint_law.py
(E, A(34) = {3,18,33}, forbidden-pair count 294) before formalising.
-/

/-- Exposed to gears 5 and 7: neither divides either member. -/
def Exposed (k : ℕ) : Prop :=
  ¬ (5 ∣ Census.lo k) ∧ ¬ (5 ∣ Census.hi k) ∧
    ¬ (7 ∣ Census.lo k) ∧ ¬ (7 ∣ Census.hi k)

instance (k : ℕ) : Decidable (Exposed k) := by unfold Exposed; infer_instance

/-- The 15-residue exposed set `E` mod 35. -/
def exposedSet : Finset ℕ :=
  {0, 2, 3, 5, 7, 10, 12, 17, 18, 23, 25, 28, 30, 32, 33}

/-- Exposure is a residue condition: for `k ≥ 1`, `k` is exposed exactly
when `k % 35 ∈ E`. -/
theorem exposed_iff_mem {k : ℕ} (hk : 1 ≤ k) :
    Exposed k ↔ k % 35 ∈ exposedSet := by
  have h5lo : (5 ∣ Census.lo k) ↔ k % 5 = 1 := by
    simp only [Census.lo]; omega
  have h5hi : (5 ∣ Census.hi k) ↔ k % 5 = 4 := by
    simp only [Census.hi]; omega
  have h7lo : (7 ∣ Census.lo k) ↔ k % 7 = 6 := by
    simp only [Census.lo]; omega
  have h7hi : (7 ∣ Census.hi k) ↔ k % 7 = 1 := by
    simp only [Census.hi]; omega
  simp only [Exposed]
  rw [h5lo, h5hi, h7lo, h7hi]
  simp only [exposedSet, Finset.mem_insert, Finset.mem_singleton]
  have e5 : k % 5 = k % 35 % 5 := by omega
  have e7 : k % 7 = k % 35 % 7 := by omega
  rw [e5, e7]
  obtain ⟨r, hr, hkr⟩ : ∃ r, r < 35 ∧ k % 35 = r :=
    ⟨k % 35, Nat.mod_lt _ (by omega), rfl⟩
  rw [hkr]
  interval_cases r <;> decide

/-- **The endpoint law.** A gap of length `G` between openings `a` and
`a + G` has its left endpoint in `A(G) = {r ∈ E : (r + G) % 35 ∈ E}`. -/
theorem endpoint_law {a G : ℕ} (ha : 1 ≤ a)
    (h1 : Exposed a) (h2 : Exposed (a + G)) :
    a % 35 ∈ exposedSet.filter fun r => (r + G) % 35 ∈ exposedSet := by
  rw [Finset.mem_filter]
  refine ⟨(exposed_iff_mem ha).mp h1, ?_⟩
  have e : (a % 35 + G) % 35 = (a + G) % 35 := by omega
  rw [e]
  exact (exposed_iff_mem (by omega)).mp h2

/-- The flagship instance: `G ≡ 34 (mod 35)` forces the left endpoint into
`{3, 18, 33}` - three residues out of fifteen. -/
theorem endpoint_law_34 {a G : ℕ} (ha : 1 ≤ a) (hG : G % 35 = 34)
    (h1 : Exposed a) (h2 : Exposed (a + G)) :
    a % 35 = 3 ∨ a % 35 = 18 ∨ a % 35 = 33 := by
  have e1 := (exposed_iff_mem ha).mp h1
  have e2 := (exposed_iff_mem (show 1 ≤ a + G by omega)).mp h2
  simp only [exposedSet, Finset.mem_insert, Finset.mem_singleton] at e1 e2
  omega

/-! ## The adjacency law

Adjacent gaps `(G1, G2)` force the opening chain `a, a+G1, a+G1+G2` into
`E`, so the left endpoint lies in `A3(G1, G2)`. When `A3` is empty the
adjacency is forbidden outright - 294 of the 1225 length-pairs mod 35,
from gears 5 and 7 alone.
-/

/-- The allowed left-endpoint residues for adjacent gaps `(g1, g2)`. -/
def allowed3 (g1 g2 : ℕ) : Finset ℕ :=
  exposedSet.filter fun r =>
    (r + g1) % 35 ∈ exposedSet ∧ (r + g1 + g2) % 35 ∈ exposedSet

/-- **The adjacency law.** Three chained openings put the left endpoint in
`A3(G1 % 35, G2 % 35)`. -/
theorem adjacency_law {a g1 g2 : ℕ} (ha : 1 ≤ a)
    (h1 : Exposed a) (h2 : Exposed (a + g1)) (h3 : Exposed (a + g1 + g2)) :
    a % 35 ∈ allowed3 (g1 % 35) (g2 % 35) := by
  rw [allowed3, Finset.mem_filter]
  refine ⟨(exposed_iff_mem ha).mp h1, ?_, ?_⟩
  · have e : (a % 35 + g1 % 35) % 35 = (a + g1) % 35 := by omega
    rw [e]
    exact (exposed_iff_mem (by omega)).mp h2
  · have e : (a % 35 + g1 % 35 + g2 % 35) % 35 = (a + g1 + g2) % 35 := by omega
    rw [e]
    exact (exposed_iff_mem (by omega)).mp h3

/-- A forbidden adjacency never occurs: no chain of three openings realises
a length-pair whose allowed set is empty. -/
theorem no_chain_of_forbidden {g1 g2 : ℕ}
    (hf : allowed3 (g1 % 35) (g2 % 35) = ∅) {a : ℕ} (ha : 1 ≤ a)
    (h1 : Exposed a) (h2 : Exposed (a + g1)) (h3 : Exposed (a + g1 + g2)) :
    False := by
  have h := adjacency_law ha h1 h2 h3
  rw [hf] at h
  exact Finset.notMem_empty _ h

/-- The first forbidden pairs, as recorded in the constructor log. -/
theorem forbidden_first_examples :
    allowed3 1 1 = ∅ ∧ allowed3 1 3 = ∅ ∧ allowed3 1 6 = ∅ := by decide

set_option maxRecDepth 8192 in
/-- **The count.** Exactly 294 of the 1225 length-pairs mod 35 are
forbidden. Verified by kernel computation of the full 35 x 35 table. -/
theorem forbidden_pairs_count :
    ((Finset.range 35 ×ˢ Finset.range 35).filter
      fun p => allowed3 p.1 p.2 = ∅).card = 294 := by decide +kernel

/-! ## The packing corollary: a floor on double slots -/

/-- **Packing.** Any `W` consecutive slots (from slot 2 on) contain at
least `W / 33` double slots: pack disjoint 33-windows and take the
guaranteed double slot of each. The demand side's doubles floor, from
gears 5 and 7 alone. -/
theorem n2_packing {a W : ℕ} (ha : 2 ≤ a) :
    W / 33 ≤ Census.n2 (Finset.Ico a (a + W)) := by
  have H : ∀ i : ℕ, ∃ k, a + 33 * i ≤ k ∧ k < a + 33 * i + 33 ∧
      Census.slotComps k = 2 := fun i => double_slot_in_run (by omega)
  choose f h1 h2 h3 using H
  have hcard : (Finset.range (W / 33)).card ≤
      ((Finset.Ico a (a + W)).filter fun k => Census.slotComps k = 2).card := by
    apply Finset.card_le_card_of_injOn f
    · intro i hi
      rw [Finset.mem_coe, Finset.mem_range] at hi
      rw [Finset.mem_coe, Finset.mem_filter, Finset.mem_Ico]
      have hi1 := h1 i
      have hi2 := h2 i
      exact ⟨⟨by omega, by omega⟩, h3 i⟩
    · intro i hi j hj heq
      have hi1 := h1 i
      have hi2 := h2 i
      have hj1 := h1 j
      have hj2 := h2 j
      omega
  rw [Finset.card_range] at hcard
  exact hcard

end Corridor
