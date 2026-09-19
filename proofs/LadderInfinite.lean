/-
LadderInfinite (2026-09-20): the (5,7) tree is infinite iff it has a node at every depth.

No König argument is needed in either direction: a node at depth `k` is at most `8^(2^k) - 1`,
so a tree with no node at some depth `n` has every node below `8^(2^n)` and is finite; and a
node at depth `n` is at least `6 + 2n`, so nodes at every depth are unbounded.  Hence the
twin prime conjecture follows from "the rung tree of (5, 7) is infinite" (tree node R5.f.xiv).
-/
import LadderDepth

namespace TwinLadder

/-- The nodes of the rung tree from `(5, 7)`. -/
def treeNodes : Set ℕ := {s | ∃ n, Depth n s}

/-- A node at depth `k` is below `8^(2^k)`. -/
theorem Depth.bound {k s : ℕ} (h : Depth k s) : s + 1 ≤ 8 ^ (2 ^ k) := by
  induction h with
  | root => norm_num
  | @step n s s' _ hr ih =>
    have h1 : s' + 1 < (s + 1) ^ 2 := hr.2.2
    have h2 : (s + 1) ^ 2 ≤ (8 ^ (2 ^ n)) ^ 2 := Nat.pow_le_pow_left ih 2
    have h3 : (8 ^ (2 ^ n)) ^ 2 = 8 ^ (2 ^ (n + 1)) := by
      rw [pow_succ 2 n, pow_mul]
    omega

/-- No node at depth `n` means no node at any depth `≥ n`. -/
theorem Depth.none_above {n : ℕ} (hn : ∀ s, ¬ Depth n s) :
    ∀ m, n ≤ m → ∀ s, ¬ Depth m s := by
  intro m hm
  induction m with
  | zero =>
    have h0 : n = 0 := by omega
    subst h0; exact hn
  | succ k ih =>
    intro s hs
    rcases Nat.lt_or_ge k n with hk | hk
    · have hnk : n = k + 1 := by omega
      subst hnk; exact hn s hs
    · cases hs with
      | step hd _ => exact ih hk _ hd

/-- **An infinite `(5, 7)` tree has a node at every depth.**  Its levels are finite, so a
missing level bounds the whole tree. -/
theorem depthHyp_of_infinite (h : treeNodes.Infinite) : DepthHyp := by
  by_contra hD
  simp only [DepthHyp, not_forall, not_exists] at hD
  obtain ⟨n, hn⟩ := hD
  apply h
  apply Set.Finite.subset (Set.finite_Iio (8 ^ (2 ^ n)))
  intro s hs
  obtain ⟨m, hm⟩ : ∃ m, Depth m s := hs
  rw [Set.mem_Iio]
  rcases Nat.lt_or_ge m n with hmn | hmn
  · have hb := hm.bound
    have hle : 8 ^ (2 ^ m) ≤ 8 ^ (2 ^ n) :=
      Nat.pow_le_pow_right (by norm_num) (Nat.pow_le_pow_right (by norm_num) hmn.le)
    omega
  · exact absurd hm (Depth.none_above hn m hmn s)

/-- A node at every depth makes the tree infinite (nodes at depth `n` are at least `6 + 2n`). -/
theorem infinite_of_depthHyp (hD : DepthHyp) : treeNodes.Infinite := by
  apply Set.infinite_of_not_bddAbove
  rintro ⟨B, hB⟩
  obtain ⟨s, hs⟩ := hD B
  have h1 : s ≤ B := hB ⟨B, hs⟩
  have h2 := hs.ge
  omega

/-- The two forms are equivalent, without König's lemma. -/
theorem infinite_iff_depthHyp : treeNodes.Infinite ↔ DepthHyp :=
  ⟨depthHyp_of_infinite, infinite_of_depthHyp⟩

/-- **Twins unbounded from an infinite `(5, 7)` tree.** -/
theorem twins_unbounded_of_infinite (h : treeNodes.Infinite) :
    ∀ N : ℕ, ∃ m : ℕ, N < 6 * m - 1 ∧ (6 * m - 1).Prime ∧ (6 * m + 1).Prime :=
  twins_unbounded_of_depth (depthHyp_of_infinite h)

end TwinLadder
