/-
LadderDepth (round 119, 2026-09-19): the weakest hypothesis - a node at every depth.

The rung tree from the twin (5, 7): depth 0 holds `6`; depth `n + 1` holds the rungs of the nodes
at depth `n`.  If the tree has a node at every depth, twin primes are unbounded - directly, since a
rung climbs by at least 2 so a node at depth `n` is a twin centre `≥ 6 + 2n`; no König argument is
needed.  This is weaker than the ladder hypothesis above any bound (a childless node does not
kill it) and, the tree being finitely branching, equivalent to the existence of an infinite path.
(Random lane round 2 (b); tree node R5.f.xiv.)
-/
import TwinLadderTheorem

namespace TwinLadder

/-- The nodes of the rung tree from `6`, by depth. -/
inductive Depth : ℕ → ℕ → Prop
  | root : Depth 0 6
  | step {n s s' : ℕ} : Depth n s → Rung s s' → Depth (n + 1) s'

/-- Every node is a twin centre. -/
theorem Depth.twinCentre {n s : ℕ} (h : Depth n s) : TwinCentre s := by
  induction h with
  | root => exact twinCentre_six
  | step _ hr _ => exact hr.1

/-- A node at depth `n` is at least `6 + 2n`. -/
theorem Depth.ge {n s : ℕ} (h : Depth n s) : 6 + 2 * n ≤ s := by
  induction h with
  | root => omega
  | step hd hr ih => have := rung_gt hd.twinCentre hr; omega

/-- **The depth hypothesis**: the rung tree from `(5, 7)` has a node at every depth. -/
def DepthHyp : Prop := ∀ n : ℕ, ∃ s : ℕ, Depth n s

/-- **Twins unbounded from a node at every depth.** -/
theorem twins_unbounded_of_depth (hD : DepthHyp) :
    ∀ N : ℕ, ∃ m : ℕ, N < 6 * m - 1 ∧ (6 * m - 1).Prime ∧ (6 * m + 1).Prime := by
  intro N
  obtain ⟨s, hs⟩ := hD N
  obtain ⟨⟨m, rfl⟩, hp1, hp2⟩ := hs.twinCentre
  have := hs.ge
  exact ⟨m, by omega, hp1, hp2⟩

/-- A path of rungs from `6` gives a node at every depth. -/
theorem depthHyp_of_path (f : ℕ → ℕ) (h0 : f 0 = 6)
    (hstep : ∀ k : ℕ, Rung (f k) (f (k + 1))) : DepthHyp := by
  intro n
  refine ⟨f n, ?_⟩
  induction n with
  | zero => rw [h0]; exact Depth.root
  | succ k ih => exact Depth.step ih (hstep k)

/-- The ladder hypothesis gives a node at every depth. -/
theorem depthHyp_of_ladderHyp (hL : LadderHyp) : DepthHyp := by
  intro n
  induction n with
  | zero => exact ⟨6, Depth.root⟩
  | succ k ih =>
    obtain ⟨s, hs⟩ := ih
    obtain ⟨s', hr⟩ := hL s hs.twinCentre
    exact ⟨s', Depth.step hs hr⟩

end TwinLadder
