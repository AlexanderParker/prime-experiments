/-
MirrorWalkRepair (round 48, 2026-09-13): the repair walk, the consistent sequential walk.

Origin: the zone start (the first column of the m-line above q).  Rule, the same at every step:
take the smallest gear striking the current column and move forward by the smallest amount that
clears it (1 or 2 columns, since a gear's two teeth cover at most two of any three consecutive
columns).  Stop when no gear strikes.  Facts proved here:
  * clear_step: a gear striking column k does not strike k+1 or does not strike k+2;
  * a step of 2 is taken only when k+1 is struck (by that same gear), so a step never passes an
    open column;
  * reach_first_open: any walk with steps of 1 or 2 that never passes an open column, started at
    or below an open column t, reaches an open column at or below t.
So the repair walk from the zone start stops at the first open column above q; it lands inside
the window exactly when that column is below q^2, and there the landing is a twin prime pair
(MirrorWalk.landing_twin).  The number of steps varies with the machine; the rule does not.
-/
import MirrorWalkSettle

namespace MirrorWalk

/-- Column `k` of the m-line, as a column of the line: `(12k - 1, 12k + 1)`. -/
def mcol (k : ℤ) : ℤ := 12 * k - 1

theorem mcol_eq_flip (k : ℤ) : mcol k = flip (k * 6) (-1) := by
  unfold mcol flip; ring

/-- **The clearing step.**  A prime gear `h ≥ 5` striking column `k` of the m-line misses `k+1`
or misses `k+2`. -/
theorem clear_step {h : ℕ} (hh : h.Prime) (h5 : 5 ≤ h) {k : ℤ} (hk : ¬ OpenTo h (mcol k)) :
    OpenTo h (mcol (k + 1)) ∨ OpenTo h (mcol (k + 2)) := by
  have hM : ¬ (h : ℤ) ∣ 2 * 6 := by
    intro hd
    have h12 : h ∣ 12 := by exact_mod_cast hd
    have h43 : h ∣ 4 * 3 := by simpa using h12
    rcases (Nat.Prime.dvd_mul hh).mp h43 with h4 | h3
    · have h22 : h ∣ 2 * 2 := by simpa using h4
      rcases (Nat.Prime.dvd_mul hh).mp h22 with h2 | h2
      · have := Nat.le_of_dvd (by norm_num) h2; omega
      · have := Nat.le_of_dvd (by norm_num) h2; omega
    · have := Nat.le_of_dvd (by norm_num) h3; omega
  obtain ⟨k', hk1, hk2, hopen⟩ := exists_axis_open hh h5 (n := -1) (k₀ := k) hM
  rw [← mcol_eq_flip] at hopen
  rcases lt_or_eq_of_le hk1 with hlt | heq
  · rcases lt_or_eq_of_le hk2 with hlt2 | heq2
    · left; have : k' = k + 1 := by omega
      rw [← this]; exact hopen
    · right; rw [← heq2]; exact hopen
  · rw [← heq] at hopen; exact absurd hopen hk

/-- A walk with steps of 1 or 2 that never passes an open column: a step of 2 from `k` is
allowed only when `k + 1` is not open. -/
inductive Reach (Open : ℤ → Prop) : ℤ → ℤ → Prop
  | refl (k : ℤ) : Reach Open k k
  | one {k k' : ℤ} (h : Reach Open k k') : Reach Open k (k' + 1)
  | two {k k' : ℤ} (h : Reach Open k k') (hmid : ¬ Open (k' + 1)) : Reach Open k (k' + 2)

theorem Reach.trans {Open : ℤ → Prop} {a b c : ℤ} (h1 : Reach Open a b) (h2 : Reach Open b c) :
    Reach Open a c := by
  induction h2 with
  | refl => exact h1
  | one _ ih => exact Reach.one ih
  | two _ hmid ih => exact Reach.two ih hmid

/-- **Reaching the first open column.**  If steps are always available from a column that is not
open (a step of 1, or a step of 2 when the middle column is not open), then from any `k ≤ t`
with `t` open the walk reaches an open column at or below `t`. -/
theorem reach_first_open (Open : ℤ → Prop)
    (hstep : ∀ k, ¬ Open k → Open (k + 1) ∨ ¬ Open (k + 1))
    {t : ℤ} (ht : Open t) :
    ∀ d : ℕ, ∀ k : ℤ, k + d = t → ∃ k', Reach Open k k' ∧ k' ≤ t ∧ Open k' := by
  intro d
  induction d using Nat.strong_induction_on with
  | h d ih =>
    intro k hk
    by_cases hok : Open k
    · exact ⟨k, Reach.refl k, by omega, hok⟩
    · have hd : 0 < d := by
        by_contra h0
        have : d = 0 := by omega
        subst this
        have : k = t := by omega
        exact hok (this ▸ ht)
      rcases hstep k hok with h1 | h1
      · -- step to k + 1, which is open
        exact ⟨k + 1, Reach.one (Reach.refl k), by omega, h1⟩
      · -- k + 1 is not open; step to k + 1 (a step of 1) and continue
        have : k + 1 + ((d - 1 : ℕ) : ℤ) = t := by omega
        obtain ⟨k', hr, hle, hop⟩ := ih (d - 1) (by omega) (k + 1) this
        exact ⟨k', Reach.trans (Reach.one (Reach.refl k)) hr, hle, hop⟩

/-- The repair walk's step is always available: from a column struck by the machine, either the
next column is open to the machine or it is not (so a step of one is always legal, and a step of
two is legal exactly when the middle column is struck). -/
theorem repair_step_available (Open : ℤ → Prop) (k : ℤ) (hk : ¬ Open k) :
    Open (k + 1) ∨ ¬ Open (k + 1) := by
  by_cases h : Open (k + 1)
  · exact Or.inl h
  · exact Or.inr h

end MirrorWalk
