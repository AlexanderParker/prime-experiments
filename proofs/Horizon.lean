/-
Formalisation of the horizon theorem.

The session's statement: the divisors below `y` decide the open interior of the
window `(y, y^2)` exactly. Concretely, any composite in that range is exposed by
a prime factor strictly below `y`, because its least factor is at most its
square root and the square root is below `y`. So inside the window, "no prime
below `y` divides it" already means "prime" - the divisors below the horizon
see everything.

The pair corollary is the form the twin search uses: if neither member of the
pair `(m, m + 2)` has a prime factor below `y`, and the pair sits inside the
window, both members are prime.

This is the interior counterpart of `BlockedSlots.survivor_iff_twin`, with the
divisor bound strict (`p < y` rather than `q ≤ y`): the top gear itself is
never needed in the open interior.
-/

import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Tactic.Linarith

namespace Horizon

/-- **The horizon theorem.** A composite in the open window `(y, y * y)` has a
prime factor strictly below `y`. Its least prime factor works: `minFac m` is
prime, divides `m`, and `minFac m ^ 2 ≤ m < y ^ 2` forces `minFac m < y`. -/
theorem exists_prime_factor_lt {y m : ℕ} (hym : y < m) (hmyy : m < y * y)
    (hnp : ¬ m.Prime) : ∃ p, p.Prime ∧ p < y ∧ p ∣ m := by
  have hm1 : 1 < m := by
    rcases Nat.lt_or_ge 1 m with h | h
    · exact h
    · exfalso
      have hy0 : y = 0 := by omega
      subst hy0
      omega
  refine ⟨m.minFac, Nat.minFac_prime (by omega), ?_, Nat.minFac_dvd m⟩
  have hsq : m.minFac ^ 2 ≤ m := Nat.minFac_sq_le_self (by omega) hnp
  have hsq' : m.minFac * m.minFac ≤ m := by simpa [pow_two] using hsq
  by_contra hge
  have hge' : y ≤ m.minFac := Nat.le_of_not_lt hge
  have hyy : y * y ≤ m.minFac * m.minFac := Nat.mul_le_mul hge' hge'
  linarith

/-- The contrapositive reading: inside the open window, a number with no prime
factor below `y` is prime. -/
theorem prime_of_no_prime_factor_lt {y m : ℕ} (hym : y < m) (hmyy : m < y * y)
    (h : ∀ p, p.Prime → p < y → ¬ p ∣ m) : m.Prime := by
  by_contra hnp
  obtain ⟨p, hp, hpy, hpd⟩ := exists_prime_factor_lt hym hmyy hnp
  exact h p hp hpy hpd

/-- **Pair corollary.** If the pair `(m, m + 2)` sits inside the open window
`(y, y * y)` and no prime below `y` divides either member, both members are
prime - a twin pair. -/
theorem twin_of_no_prime_factor_lt {y m : ℕ} (hym : y < m) (hwin : m + 2 < y * y)
    (h : ∀ p, p.Prime → p < y → ¬ p ∣ m ∧ ¬ p ∣ (m + 2)) :
    m.Prime ∧ (m + 2).Prime := by
  constructor
  · exact prime_of_no_prime_factor_lt hym (by linarith) fun p hp hpy => (h p hp hpy).1
  · exact prime_of_no_prime_factor_lt (by omega) hwin fun p hp hpy => (h p hp hpy).2

end Horizon
