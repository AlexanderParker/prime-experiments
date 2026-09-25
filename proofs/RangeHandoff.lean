import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Data.Nat.Prime.Infinite
import Mathlib.NumberTheory.Primorial
import Mathlib.Tactic

/-!
# The range statement and its hand-off form

For a bound `q`, write `primorial q` for the product of the primes `≤ q`.
The *range statement* at `q` says: there is a twin pair `(p, p + 2)` lying strictly above `q`
and with its upper member at most `primorial q`.

This file proves:

* `range_implies_unbounded`: if the range statement holds at every prime `q ≥ 7`, then
  twin pairs are unbounded: for every `N` there is a prime `p > N` with `p + 2` prime.
  (Pick a prime `q ≥ N + 7`; the pair it supplies sits above `q`, hence above `N`.)
* `twin_serves`: a twin pair `(p, p + 2)` with `q < p` and `p + 2 ≤ primorial q` is a witness
  for the range statement at `q` (this is the definition read backwards).
* `serves_interval` (and its hypothesis-light form `serves_interval_of_le`): one twin pair
  `(p, p + 2)` with `p + 2 ≤ primorial q₁` and `q₂ < p` serves every `q` with `q₁ ≤ q ≤ q₂`,
  because `primorial` is monotone, so `primorial q₁ ≤ primorial q`, while `q ≤ q₂ < p`.
  This is the hand-off: a pair serves a whole interval of bounds, and the next pair takes over.
* `primorial_seven`: `primorial 7 = 210`.
* `small_cases` (and `small_case_7`, `small_case_11`, `small_case_13`, `small_case_17`):
  the range statement holds at every `q` with `7 ≤ q ≤ 23`, in particular at `q = 7, 11, 13, 17`,
  all served by the single pair `(29, 31)`: `29, 31` are prime, `23 < 29`, `31 ≤ 210 = primorial 7`.
-/

namespace RangeLine

/-- The range statement at `q`: some twin pair `(p, p + 2)` has `q < p` and `p + 2 ≤ primorial q`. -/
def RangeStatement (q : ℕ) : Prop :=
  ∃ p, q < p ∧ p + 2 ≤ primorial q ∧ p.Prime ∧ (p + 2).Prime

/-- (a) If the range statement holds at every prime `q ≥ 7`, then for every `N` there is a
prime `p > N` with `p + 2` prime: twin pairs are unbounded. -/
theorem range_implies_unbounded
    (h : ∀ q : ℕ, q.Prime → 7 ≤ q → RangeStatement q) :
    ∀ N : ℕ, ∃ p, N < p ∧ p.Prime ∧ (p + 2).Prime := by
  intro N
  obtain ⟨q, hNq, hq⟩ := Nat.exists_infinite_primes (N + 7)
  obtain ⟨p, hqp, _, hp, hp2⟩ := h q hq (by omega)
  exact ⟨p, by omega, hp, hp2⟩

/-- (b) A twin pair `(p, p + 2)` with `q < p` and `p + 2 ≤ primorial q` witnesses the range
statement at `q`. -/
theorem twin_serves {p q : ℕ} (hp : p.Prime) (hp2 : (p + 2).Prime)
    (hqp : q < p) (hle : p + 2 ≤ primorial q) : RangeStatement q :=
  ⟨p, hqp, hle, hp, hp2⟩

/-- A twin pair `(p, p + 2)` with `p + 2 ≤ primorial q₁` and `q₂ < p` serves every bound `q`
with `q₁ ≤ q ≤ q₂` (no primality of `q₁`, `q₂` is needed). -/
theorem serves_interval_of_le {p q₁ q₂ : ℕ} (hp : p.Prime) (hp2 : (p + 2).Prime)
    (hle : p + 2 ≤ primorial q₁) (hlt : q₂ < p) :
    ∀ q : ℕ, q₁ ≤ q → q ≤ q₂ → RangeStatement q := by
  intro q h1 h2
  exact twin_serves hp hp2 (by omega) (le_trans hle (primorial_mono h1))

/-- (c) For a twin pair `(p, p + 2)` and primes `q₁ ≤ q₂` with `p + 2 ≤ primorial q₁` and
`q₂ < p`, the range statement holds at every `q` with `q₁ ≤ q ≤ q₂` (primorial is monotone). -/
theorem serves_interval {p q₁ q₂ : ℕ} (hp : p.Prime) (hp2 : (p + 2).Prime)
    (_hq₁ : q₁.Prime) (_hq₂ : q₂.Prime) (_h12 : q₁ ≤ q₂)
    (hle : p + 2 ≤ primorial q₁) (hlt : q₂ < p) :
    ∀ q : ℕ, q₁ ≤ q → q ≤ q₂ → RangeStatement q :=
  serves_interval_of_le hp hp2 hle hlt

/-- `primorial 7 = 2 * 3 * 5 * 7 = 210`. -/
theorem primorial_seven : primorial 7 = 210 := by
  decide

/-- (d) The range statement holds at every `q` with `7 ≤ q ≤ 23`, all served by the pair
`(29, 31)`: both prime, `23 < 29`, and `31 ≤ 210 = primorial 7`. -/
theorem small_cases : ∀ q : ℕ, 7 ≤ q → q ≤ 23 → RangeStatement q :=
  serves_interval (p := 29) (q₁ := 7) (q₂ := 23) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by rw [primorial_seven]; norm_num) (by norm_num)

/-- (d) The range statement at `q = 7`, witness `p = 29`. -/
theorem small_case_7 : RangeStatement 7 := small_cases 7 (by norm_num) (by norm_num)

/-- (d) The range statement at `q = 11`, witness `p = 29`. -/
theorem small_case_11 : RangeStatement 11 := small_cases 11 (by norm_num) (by norm_num)

/-- (d) The range statement at `q = 13`, witness `p = 29`. -/
theorem small_case_13 : RangeStatement 13 := small_cases 13 (by norm_num) (by norm_num)

/-- (d) The range statement at `q = 17`, witness `p = 29`. -/
theorem small_case_17 : RangeStatement 17 := small_cases 17 (by norm_num) (by norm_num)

end RangeLine
