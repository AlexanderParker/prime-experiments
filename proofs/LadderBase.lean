/-
LadderBase (round 115, 2026-09-19): the base half of the rung, from the record alone.

If the pattern of the base gears has no run of `F + 1` consecutive struck columns (`F` its record),
then every block of `F + 1` consecutive columns holds an open column; so a window of `L` columns
holds at least `L / (F + 1)` open columns, one in each full block.  No density, no count of
strikes: the record is a property of the gears alone, and the window's length is the twin's.
(Tree node R5.f.vi, claim A (i); the lane's pigeonhole.)
-/
import Mathlib.Tactic

namespace LadderBase

/-- **No long run, so an open column in every block.**  If no `F + 1` consecutive columns are all
struck, then for every start `a` and every block index `k` there is an open column in the block
`[a + k (F + 1), a + (k + 1) (F + 1))`. -/
theorem open_in_every_block (struck : ℕ → Prop) (F : ℕ)
    (hF : ∀ a : ℕ, ∃ i : ℕ, i ≤ F ∧ ¬ struck (a + i)) (a k : ℕ) :
    ∃ i : ℕ, k * (F + 1) ≤ i ∧ i < (k + 1) * (F + 1) ∧ ¬ struck (a + i) := by
  obtain ⟨t, ht, hopen⟩ := hF (a + k * (F + 1))
  refine ⟨k * (F + 1) + t, by omega, by nlinarith, ?_⟩
  have : a + (k * (F + 1) + t) = a + k * (F + 1) + t := by ring
  rw [this]; exact hopen

/-- **The window holds at least `L / (F + 1)` open columns**: the open columns picked one per full
block are pairwise distinct (they lie in disjoint blocks), so the finset of open columns of the
window has at least `L / (F + 1)` elements. -/
theorem card_open_ge (struck : ℕ → Prop) [DecidablePred struck] (F : ℕ)
    (hF : ∀ a : ℕ, ∃ i : ℕ, i ≤ F ∧ ¬ struck (a + i)) (a L : ℕ) :
    L / (F + 1) ≤ ((Finset.range L).filter (fun i => ¬ struck (a + i))).card := by
  classical
  -- choose one open column per full block
  have hpick : ∀ k : ℕ, ∃ i : ℕ, k * (F + 1) ≤ i ∧ i < (k + 1) * (F + 1) ∧ ¬ struck (a + i) :=
    fun k => open_in_every_block struck F hF a k
  choose f hf using hpick
  have hinj : Function.Injective f := by
    intro k₁ k₂ h
    have h1 := hf k₁; have h2 := hf k₂
    by_contra hne
    rcases Nat.lt_or_gt_of_ne hne with hlt | hlt
    · have : (k₁ + 1) * (F + 1) ≤ k₂ * (F + 1) := Nat.mul_le_mul_right _ hlt
      omega
    · have : (k₂ + 1) * (F + 1) ≤ k₁ * (F + 1) := Nat.mul_le_mul_right _ hlt
      omega
  have hsub : (Finset.range (L / (F + 1))).image f ⊆
      (Finset.range L).filter (fun i => ¬ struck (a + i)) := by
    intro i hi
    obtain ⟨k, hk, rfl⟩ := Finset.mem_image.mp hi
    have hkL : k < L / (F + 1) := Finset.mem_range.mp hk
    have h := hf k
    refine Finset.mem_filter.mpr ⟨Finset.mem_range.mpr ?_, h.2.2⟩
    have : (k + 1) * (F + 1) ≤ L := by
      have := Nat.div_mul_le_self L (F + 1)
      have h3 : (k + 1) ≤ L / (F + 1) := hkL
      calc (k + 1) * (F + 1) ≤ (L / (F + 1)) * (F + 1) := Nat.mul_le_mul_right _ h3
        _ ≤ L := Nat.div_mul_le_self L (F + 1)
    omega
  calc L / (F + 1) = (Finset.range (L / (F + 1))).card := (Finset.card_range _).symm
    _ = ((Finset.range (L / (F + 1))).image f).card := (Finset.card_image_of_injective _ hinj).symm
    _ ≤ _ := Finset.card_le_card hsub

end LadderBase
