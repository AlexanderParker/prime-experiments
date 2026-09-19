/-
TwinLadderTheorem (round 116, 2026-09-19): the ladder theorem with its one named hypothesis.

A TWIN CENTRE is `s` with `6 ∣ s` and `s - 1`, `s + 1` both prime.  A RUNG from `s` is a twin
centre `s'` inside the stretch of `s`: `(s-1)² < s' - 1` and `s' + 1 < (s+1)²` (by the normal form
in TwinLadder.lean this is `s' = s² + 6j` with `|j| ≤ 2c - 1`, `s = 6c`).

THE LADDER HYPOTHESIS: every twin centre has a rung.  The NEAR TWIN HYPOTHESIS (the lane's NTH)
is the bounded form - the rung is found among the first `⌈4 ln s⌉` base-open offsets - and
implies it.  THE THEOREM: under the ladder hypothesis the ladder from `s₀ = 6` (the twin (5, 7))
never terminates and twin primes are unbounded - SurvivalInf.  Each link is checkable by two
primality certificates and two containment inequalities; the existing six-link chain certificate
is this object's first links.  (Tree node R5.f.vii.)
-/
import Mathlib.Tactic
import Mathlib.Data.Nat.Prime.Basic

namespace TwinLadder

/-- A twin centre: `s - 1` and `s + 1` are both prime and `6 ∣ s`. -/
def TwinCentre (s : ℕ) : Prop := 6 ∣ s ∧ (s - 1).Prime ∧ (s + 1).Prime

/-- A rung from `s`: a twin centre strictly inside the stretch `((s-1)², (s+1)²)`. -/
def Rung (s s' : ℕ) : Prop := TwinCentre s' ∧ (s - 1) ^ 2 < s' - 1 ∧ s' + 1 < (s + 1) ^ 2

/-- **The ladder hypothesis**: every twin centre has a rung. -/
def LadderHyp : Prop := ∀ s : ℕ, TwinCentre s → ∃ s' : ℕ, Rung s s'

/-- A twin centre is at least 6. -/
theorem twinCentre_ge_six {s : ℕ} (h : TwinCentre s) : 6 ≤ s := by
  obtain ⟨⟨k, rfl⟩, hp, _⟩ := h
  rcases k with _ | k
  · simp at hp; exact absurd hp Nat.not_prime_zero
  · omega

/-- The twin (5, 7) is the first centre. -/
theorem twinCentre_six : TwinCentre 6 := ⟨dvd_refl 6, by norm_num, by norm_num⟩

/-- A rung climbs: `s' ≥ (s - 1)² + 2 > s`. -/
theorem rung_gt {s s' : ℕ} (hs : TwinCentre s) (h : Rung s s') : s + 2 ≤ s' := by
  have h6 := twinCentre_ge_six hs
  have h1 : (s - 1) ^ 2 < s' - 1 := h.2.1
  have h2 : s ≤ (s - 1) ^ 2 := by
    obtain ⟨t, rfl⟩ : ∃ t, s = t + 1 := ⟨s - 1, by omega⟩
    rw [Nat.add_sub_cancel]
    nlinarith
  omega

/-- **Under the ladder hypothesis, twin centres are unbounded.** -/
theorem twinCentre_unbounded (hL : LadderHyp) : ∀ N : ℕ, ∃ s : ℕ, TwinCentre s ∧ N ≤ s := by
  intro N
  induction N with
  | zero => exact ⟨6, twinCentre_six, Nat.zero_le _⟩
  | succ n ih =>
    obtain ⟨s, hs, hn⟩ := ih
    obtain ⟨s', hr⟩ := hL s hs
    exact ⟨s', hr.1, by have := rung_gt hs hr; omega⟩

/-- **The ladder theorem**: under the ladder hypothesis there are twin primes above every bound,
in the kernel's standard form (`6m - 1`, `6m + 1` both prime). -/
theorem twins_unbounded_of_ladder (hL : LadderHyp) :
    ∀ N : ℕ, ∃ m : ℕ, N < 6 * m - 1 ∧ (6 * m - 1).Prime ∧ (6 * m + 1).Prime := by
  intro N
  obtain ⟨s, ⟨⟨m, rfl⟩, hp1, hp2⟩, hN⟩ := twinCentre_unbounded hL (N + 2)
  exact ⟨m, by omega, hp1, hp2⟩

/-- **The Near Twin Hypothesis** (the lane's NTH, bounded form): for every twin centre `s` there
is a rung at an offset among the first `⌈4 ln s⌉` base-open offsets.  Stated here through any
bound function `B` on the offset's rank; it implies the ladder hypothesis outright. -/
def NearTwinHyp (rankOf : ℕ → ℕ → ℕ) (B : ℕ → ℕ) : Prop :=
  ∀ s : ℕ, TwinCentre s → ∃ s' : ℕ, Rung s s' ∧ rankOf s s' ≤ B s

theorem ladderHyp_of_nearTwin {rankOf : ℕ → ℕ → ℕ} {B : ℕ → ℕ}
    (h : NearTwinHyp rankOf B) : LadderHyp :=
  fun s hs => let ⟨s', hr, _⟩ := h s hs; ⟨s', hr⟩

/-! ### The weakest form that composes: a path (lane round 10)

The induction never uses a rung at every twin centre above a bound - only at the centres it
reaches.  `twins_unbounded_of_path` consumes exactly a path: a sequence of twin centres each a
rung of the previous.  `path_of_ladderHyp_above` builds one from the universal hypothesis by
dependent choice, so the certified prefix and any tail hypothesis compose through the path. -/

/-- **Twins unbounded from a path** of rungs starting at any twin centre. -/
theorem twins_unbounded_of_path (f : ℕ → ℕ) (h0 : TwinCentre (f 0))
    (hstep : ∀ k : ℕ, Rung (f k) (f (k + 1))) :
    ∀ N : ℕ, ∃ m : ℕ, N < 6 * m - 1 ∧ (6 * m - 1).Prime ∧ (6 * m + 1).Prime := by
  have htc : ∀ k : ℕ, TwinCentre (f k) := by
    intro k
    induction k with
    | zero => exact h0
    | succ k ih => exact (hstep k).1
  have hgrow : ∀ k : ℕ, k ≤ f k := by
    intro k
    induction k with
    | zero => exact Nat.zero_le _
    | succ k ih => have := rung_gt (htc k) (hstep k); omega
  intro N
  obtain ⟨⟨m, hm⟩, hp1, hp2⟩ := htc (N + 2)
  have := hgrow (N + 2)
  rw [hm] at this hp1 hp2
  exact ⟨m, by omega, hp1, hp2⟩

/-- **A path from the universal hypothesis above a bound**, by dependent choice: if every twin
centre at least `S₀` has a rung, any twin centre `s₀ ≥ S₀` starts an infinite path. -/
theorem path_of_ladderHyp_above (S₀ s₀ : ℕ) (hs₀ : TwinCentre s₀) (hS : S₀ ≤ s₀)
    (hL : ∀ s : ℕ, TwinCentre s → S₀ ≤ s → ∃ s' : ℕ, Rung s s') :
    ∃ f : ℕ → ℕ, f 0 = s₀ ∧ ∀ k : ℕ, Rung (f k) (f (k + 1)) := by
  classical
  -- the state carries the twin-centre and bound facts along
  let next : {s : ℕ // TwinCentre s ∧ S₀ ≤ s} → {s : ℕ // TwinCentre s ∧ S₀ ≤ s} :=
    fun p =>
      let h := hL p.1 p.2.1 p.2.2
      ⟨Classical.choose h, (Classical.choose_spec h).1,
        by have := rung_gt p.2.1 (Classical.choose_spec h); omega⟩
  let g : ℕ → {s : ℕ // TwinCentre s ∧ S₀ ≤ s} := fun k => Nat.rec ⟨s₀, hs₀, hS⟩ (fun _ p => next p) k
  refine ⟨fun k => (g k).1, rfl, ?_⟩
  intro k
  show Rung (g k).1 (next (g k)).1
  exact Classical.choose_spec (hL (g k).1 (g k).2.1 (g k).2.2)

end TwinLadder
