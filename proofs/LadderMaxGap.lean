/-
LadderMaxGap (2026-09-20): the machine's own record beats the window.

`F(q)`, the rigid record, is the longest run of consecutive columns every one of which some gear
`5..q` strikes, over the whole period of the pattern.  If every run of `L` columns holds an
unstruck column (`MaxGapBelow q L`), then the window of `q`, being such a run, holds one, and its
members are prime by the square-root rule.  This is the covering route of `LadderCovering.lean`
with the free classes replaced by the actual pattern: a weaker hypothesis (`F(q) ≤ j_2(q)`),
measured `F(67) = 213` against a window of `737` columns.  (Tree node R5.f.xxxii.a.)
-/
import LadderCovering

namespace TwinLadder

/-- **Every run of `L` columns (from a column `≥ 1`) holds an unstruck column**: the machine's
longest struck run is below `L`. -/
def MaxGapBelow (q L : ℕ) : Prop := ∀ a, 1 ≤ a → ∃ n, a ≤ n ∧ n < a + L ∧ ¬ Struck q n

/-- Free-uncoverability of every run gives the max-gap bound. -/
theorem maxGapBelow_of_freeUncoverable (q L : ℕ) (h : ∀ a, FreeUncoverable q a L) :
    MaxGapBelow q L :=
  fun a ha => exists_unstruck q a L ha (h a)

/-- **A run inside the window of `q` with an unstruck column holds a twin centre.** -/
theorem window_twin_of_maxGap (q : ℕ) (hq : q.Prime) (h5 : 5 ≤ q) (a L : ℕ)
    (ha : q < 6 * a - 1) (hL : 6 * (a + L - 1) + 1 ≤ q ^ 2) (hL0 : 1 ≤ L)
    (h : MaxGapBelow q L) :
    ∃ n, q < 6 * n - 1 ∧ 6 * n + 1 ≤ q ^ 2 ∧ TwinCentre (6 * n) := by
  obtain ⟨n, hn1, hn2, hns⟩ := h a (by omega)
  have hn : 1 ≤ n := by omega
  have hlt : q < 6 * n - 1 := by omega
  have hle : 6 * n + 1 ≤ q ^ 2 := by omega
  have hno1 : ∀ p, Gear q p → ¬ p ∣ 6 * n - 1 := fun p hp hd => hns ⟨p, hp, Or.inl hd⟩
  have hno2 : ∀ p, Gear q p → ¬ p ∣ 6 * n + 1 := fun p hp hd => hns ⟨p, hp, Or.inr hd⟩
  have hp1 : (6 * n - 1).Prime :=
    prime_of_unstruck_member hq h5 (coprime_sub_six hn) hlt (by omega) hno1
  have hp2 : (6 * n + 1).Prime :=
    prime_of_unstruck_member hq h5 (coprime_add_six n) (by omega) hle hno2
  exact ⟨n, hlt, hle, ⟨dvd_mul_right 6 n, hp1, hp2⟩⟩

/-- **The max-gap hypothesis**: for every machine `q` the longest struck run of the gears `5..q`
is shorter than the window `(q + 7)/6 .. (q² - 1)/6`. -/
def MaxGapHyp : Prop :=
  ∀ q, q.Prime → 5 ≤ q → MaxGapBelow q ((q ^ 2 - 1) / 6 - (q + 7) / 6 + 1)

/-- **The window statement from the max-gap hypothesis.** -/
theorem windowStatement_of_maxGapHyp (h : MaxGapHyp) :
    ∀ q, q.Prime → 5 ≤ q → ∃ n, q < 6 * n - 1 ∧ 6 * n + 1 ≤ q ^ 2 ∧ TwinCentre (6 * n) := by
  intro q hq h5
  have hqq : q + 8 ≤ q ^ 2 := by nlinarith
  have ht : 25 ≤ q ^ 2 := by nlinarith
  refine window_twin_of_maxGap q hq h5 ((q + 7) / 6) _ ?_ ?_ ?_ (h q hq h5)
  · omega
  · generalize q ^ 2 = t at *; omega
  · omega

/-- **Twins unbounded from the max-gap hypothesis.** -/
theorem twins_unbounded_of_maxGapHyp (h : MaxGapHyp) :
    ∀ N : ℕ, ∃ m : ℕ, N < 6 * m - 1 ∧ (6 * m - 1).Prime ∧ (6 * m + 1).Prime := by
  intro N
  obtain ⟨q, hNq, hq⟩ := Nat.exists_infinite_primes (N + 6)
  obtain ⟨n, hlt, -, -, hp1, hp2⟩ := windowStatement_of_maxGapHyp h q hq (by omega)
  exact ⟨n, by omega, hp1, hp2⟩

end TwinLadder
