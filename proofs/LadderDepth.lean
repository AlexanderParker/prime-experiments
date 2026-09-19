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

/-! ### The rung graph is a forest: parents are unique (random lane, angle 2) -/

/-- **Unique parent.**  Two twin centres with a common rung are equal: both lie in the interval
`(√s' - 1, √s' + 1)` of length 2, which holds at most one multiple of 6. -/
theorem parent_unique {s t s' : ℕ} (hs : TwinCentre s) (ht : TwinCentre t)
    (h1 : Rung s s') (h2 : Rung t s') : s = t := by
  have h6s := twinCentre_ge_six hs
  have h6t := twinCentre_ge_six ht
  obtain ⟨⟨a, rfl⟩, _, _⟩ := hs
  obtain ⟨⟨b, rfl⟩, _, _⟩ := ht
  by_contra hne
  rcases Nat.lt_or_gt_of_ne hne with hlt | hlt
  · -- 6a < 6b, so 6a + 1 ≤ 6b - 1, so (6a+1)² ≤ (6b-1)² < s' - 1 < s' + 1 < (6a+1)²
    have hle : 6 * a + 1 ≤ 6 * b - 1 := by omega
    have hsq : (6 * a + 1) ^ 2 ≤ (6 * b - 1) ^ 2 := Nat.pow_le_pow_left hle 2
    have := h1.2.2; have := h2.2.1
    omega
  · have hle : 6 * b + 1 ≤ 6 * a - 1 := by omega
    have hsq : (6 * b + 1) ^ 2 ≤ (6 * a - 1) ^ 2 := Nat.pow_le_pow_left hle 2
    have := h2.2.2; have := h1.2.1
    omega

/-! ### A leaf is sieve data (contradiction lane, M2) -/

/-- **Every composite member of a twin's stretch has a prime factor at most `s - 1`.**  A composite
`m < (s+1)²` has least prime factor `p` with `p² ≤ m < (s+1)²`, so `p ≤ s`; and `p ≠ s` because
`6 ∣ s`.  Hence a twin centre with no rung ("a leaf") is exactly a two-class covering of its window
by the gears `5..s-1`: there is no plug from above and no analytic remainder. -/
theorem leaf_is_sieve_data {s m : ℕ} (hs : TwinCentre s) (hm : m < (s + 1) ^ 2)
    (hcomp : ¬ m.Prime) (h2 : 2 ≤ m) :
    ∃ p : ℕ, p.Prime ∧ p ∣ m ∧ p ≤ s - 1 := by
  obtain ⟨⟨k, rfl⟩, _, _⟩ := hs
  have hmin : (Nat.minFac m).Prime := Nat.minFac_prime (by omega)
  have hsq : Nat.minFac m ^ 2 ≤ m := Nat.minFac_sq_le_self (by omega) hcomp
  refine ⟨Nat.minFac m, hmin, Nat.minFac_dvd m, ?_⟩
  have hlt : Nat.minFac m < 6 * k + 1 := by
    by_contra h; push_neg at h
    have : (6 * k + 1) ^ 2 ≤ Nat.minFac m ^ 2 := Nat.pow_le_pow_left h 2
    omega
  -- minFac m ≤ 6k and minFac m ≠ 6k (6k is not prime unless k = 0, excluded by 2 ≤ m < 1)
  have hne : Nat.minFac m ≠ 6 * k := by
    intro h
    have hp : (6 * k).Prime := h ▸ hmin
    have h2d : (2 : ℕ) ∣ 6 * k := ⟨3 * k, by ring⟩
    rcases hp.eq_one_or_self_of_dvd 2 h2d with h' | h' <;> omega
  omega

end TwinLadder
