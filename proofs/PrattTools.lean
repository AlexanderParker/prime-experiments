/-
PrattTools (round 69, 2026-09-17): the pieces for certifying a large prime in the kernel.

`norm_num` proves primality by trial division, which costs the square root of the number: a
thirteen-digit prime already takes minutes and exceeds the recursion limit, so the certificate of
round 65 could not pass its fourth link.  A Lucas (Pratt) certificate costs the logarithm
instead: a witness `a` of order `p - 1` modulo `p`, checked by one square-and-multiply chain and
one check per prime factor of `p - 1`.

These are the tools the generated certificates use:
  * `cast_pow_eq_one_iff` moves the `ZMod p` statement of `lucas_primality` to arithmetic on ℕ;
  * `sq_step` and `sq_mul_step` are the two square-and-multiply steps, each one multiplication of
    numbers the size of `p` - so a seventeen-digit prime needs about fifty of them;
  * `ne_one_of_mod` discharges the order conditions.
-/
import Mathlib

namespace Pratt

/-- The `ZMod p` power condition as arithmetic on ℕ. -/
theorem cast_pow_eq_one_iff {p a n : ℕ} (hp : p ≠ 1) :
    (((a : ℕ) : ZMod p)) ^ n = 1 ↔ a ^ n % p = 1 % p := by
  rw [← Nat.cast_pow, ← Nat.cast_one, ZMod.natCast_eq_natCast_iff]
  rfl

/-- One squaring step of the exponentiation chain. -/
theorem sq_step {a p k r : ℕ} (h : a ^ k % p = r) : a ^ (2 * k) % p = r * r % p := by
  have e : a ^ (2 * k) = a ^ k * a ^ k := by rw [two_mul, pow_add]
  rw [e, Nat.mul_mod, h]

/-- One squaring-and-multiplying step. -/
theorem sq_mul_step {a p k r : ℕ} (h : a ^ k % p = r) :
    a ^ (2 * k + 1) % p = r * r % p * a % p := by
  have e : a ^ (2 * k + 1) = a ^ k * a ^ k * a := by rw [pow_succ, two_mul, pow_add]
  rw [e, Nat.mul_mod (a ^ k * a ^ k) a, Nat.mul_mod (a ^ k) (a ^ k), h]
  rw [Nat.mul_mod (r * r % p) a]
  simp [Nat.mod_mod_of_dvd, Nat.mod_mod]

/-- The order condition, from a computed residue. -/
theorem ne_one_of_mod {p a n r : ℕ} (hp : 1 < p) (h : a ^ n % p = r) (hr : r ≠ 1) :
    (((a : ℕ) : ZMod p)) ^ n ≠ 1 := by
  intro hcon
  rw [cast_pow_eq_one_iff (by omega : p ≠ 1), h, Nat.one_mod_eq_one.mpr (by omega)] at hcon
  exact hr hcon

/-- A squaring step with the exponent and residue given as literals, so a generated certificate
never has to rewrite inside the goal. -/
theorem sq_of {a p k r n s : ℕ} (h : a ^ k % p = r) (hn : n = 2 * k) (hs : r * r % p = s) :
    a ^ n % p = s := by
  subst hn; rw [sq_step h, hs]

/-- A squaring-and-multiplying step, same form. -/
theorem sq_mul_of {a p k r n s : ℕ} (h : a ^ k % p = r) (hn : n = 2 * k + 1)
    (hs : r * r % p * a % p = s) : a ^ n % p = s := by
  subst hn; rw [sq_mul_step h, hs]

end Pratt
