import RangeHandoff

/-!
# A chain of twin pairs gives the range statement

Suppose we have a chain of twin pairs `(p 0, p 0 + 2), (p 1, p 1 + 2), ...` where

* the starting pair is `(29, 31)`,
* the lower members strictly increase, and
* each next pair fits under the primorial of the previous lower member:
  `p (k + 1) + 2 ≤ primorial (p k)`.

Then the range statement holds at every bound `q ≥ 7` (in particular at every prime `q ≥ 7`),
and therefore twin pairs are unbounded.

The argument: given `q`, take the first pair in the chain whose lower member is above `q`
(it exists because a strictly increasing sequence of naturals passes every bound).
If it is the first pair `(29, 31)`, then `31 ≤ 210 = primorial 7 ≤ primorial q`.
Otherwise the pair just before it has lower member `p (k - 1) ≤ q`, so
`p k + 2 ≤ primorial (p (k - 1)) ≤ primorial q`, because `primorial` is monotone.

This file proves:

* `primorial_le_of_le`: `a ≤ b` gives `primorial a ≤ primorial b` (Mathlib's `primorial_mono`,
  restated here).
* `chain_implies_range_all`: under the chain hypotheses, the range statement holds at every
  `q ≥ 7` (no primality of `q` needed).
* `chain_implies_range`: under the chain hypotheses, the range statement holds at every prime
  `q ≥ 7`.
* `chain_implies_unbounded`: under the chain hypotheses, for every `N` there is a prime `r > N`
  with `r + 2` prime.
-/

namespace RangeLine

/-- `primorial` is monotone: `a ≤ b` gives `primorial a ≤ primorial b`
(this is Mathlib's `primorial_mono`). -/
theorem primorial_le_of_le {a b : ℕ} (h : a ≤ b) : primorial a ≤ primorial b :=
  primorial_mono h

/-- A chain of twin pairs starting at `(29, 31)`, strictly increasing, with each next pair fitting
under the primorial of the previous lower member, gives the range statement at every `q ≥ 7`. -/
theorem chain_implies_range_all (p : ℕ → ℕ) (hmono : StrictMono p)
    (hprime : ∀ k, (p k).Prime) (hprime2 : ∀ k, (p k + 2).Prime)
    (h0 : p 0 = 29) (hstep : ∀ k, p (k + 1) + 2 ≤ primorial (p k)) :
    ∀ q : ℕ, 7 ≤ q → RangeStatement q := by
  intro q hq7
  classical
  have hex : ∃ k, q < p k :=
    ⟨q + 1, lt_of_lt_of_le (Nat.lt_succ_self q) (hmono.id_le (q + 1))⟩
  have hk : q < p (Nat.find hex) := Nat.find_spec hex
  have hmin : ∀ m, m < Nat.find hex → ¬ q < p m := fun m hm => Nat.find_min hex hm
  cases hfind : Nat.find hex with
  | zero =>
    rw [hfind] at hk
    refine twin_serves (hprime 0) (hprime2 0) hk ?_
    have h7 : primorial 7 ≤ primorial q := primorial_le_of_le hq7
    rw [primorial_seven] at h7
    rw [h0]
    omega
  | succ j =>
    rw [hfind] at hk hmin
    have hjq : p j ≤ q := Nat.le_of_not_lt (hmin j (Nat.lt_succ_self j))
    exact twin_serves (hprime (j + 1)) (hprime2 (j + 1)) hk
      (le_trans (hstep j) (primorial_le_of_le hjq))

/-- (a) A chain of twin pairs starting at `(29, 31)`, strictly increasing, with each next pair
fitting under the primorial of the previous lower member, gives the range statement at every
prime `q ≥ 7`. -/
theorem chain_implies_range (p : ℕ → ℕ) (hmono : StrictMono p)
    (hprime : ∀ k, (p k).Prime) (hprime2 : ∀ k, (p k + 2).Prime)
    (h0 : p 0 = 29) (hstep : ∀ k, p (k + 1) + 2 ≤ primorial (p k)) :
    ∀ q : ℕ, q.Prime → 7 ≤ q → RangeStatement q :=
  fun q _ hq7 => chain_implies_range_all p hmono hprime hprime2 h0 hstep q hq7

/-- (b) Under the same chain hypotheses, twin pairs are unbounded: for every `N` there is a
prime `r > N` with `r + 2` prime. -/
theorem chain_implies_unbounded (p : ℕ → ℕ) (hmono : StrictMono p)
    (hprime : ∀ k, (p k).Prime) (hprime2 : ∀ k, (p k + 2).Prime)
    (h0 : p 0 = 29) (hstep : ∀ k, p (k + 1) + 2 ≤ primorial (p k)) :
    ∀ N : ℕ, ∃ r, N < r ∧ r.Prime ∧ (r + 2).Prime :=
  range_implies_unbounded (chain_implies_range p hmono hprime hprime2 h0 hstep)

end RangeLine
