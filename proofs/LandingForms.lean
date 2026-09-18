/-
LandingForms (round 74, 2026-09-18): which shapes a landing can and cannot have.

The anatomy of the failures (research/proof/failure_anatomy.md) ends at one unmet requirement: a
proof must name a candidate rather than a population, and the reason must come from the
candidate's own arithmetic.  Arithmetic does supply reasons of exactly one kind - algebraic
factorisation - and this file records what they give here.

They are all negative.  A factorisation identity can say a named candidate is CLOSED; nothing of
that kind says one is open.  The sharpest instance:

  `perfect_power_landing`: the only landing that is a perfect power is 4, the pair (3, 5).

The proof is the shape of the whole family of such arguments.  If `N = x^k` with `k ≥ 2` then
`x - 1` divides `N - 1`, so the lower member is composite unless `x = 2`; and then `2^k - 1`
prime forces `k` prime, while for odd `k` the upper member `2^k + 1` is divisible by 3.  Only
`k = 2` survives.

The one positive reason arithmetic gives is the mirror itself: a candidate `2 M k ± 1` is coprime
to every gear dividing `M`, by construction and not by luck (`mirror_gear_never_strikes`).  That
is the primorial construction, and it reaches only the gears below the mirror - which is the carry
wall (`silence_costs_primorial`).  So arithmetic names candidates as closed freely, and as open
only up to the wall.
-/
import Mathlib

namespace MirrorWalk

/-- For odd `k`, the upper member `2^k + 1` is divisible by 3. -/
theorem three_dvd_two_pow_add_one {k : ℕ} (hk : Odd k) : 3 ∣ 2 ^ k + 1 := by
  have h : ((2 ^ k + 1 : ℕ) : ZMod 3) = 0 := by
    push_cast
    have h2 : (2 : ZMod 3) = -1 := by decide
    rw [h2, hk.neg_one_pow]
    ring
  exact (ZMod.natCast_eq_zero_iff _ 3).mp h

/-- **The only perfect-power landing is 4.**  If `x^k` is a twin centre with `k ≥ 2`, then
`x^k = 4` and the pair is `(3, 5)`. -/
theorem perfect_power_landing {x k : ℕ} (hk : 2 ≤ k)
    (hlo : (x ^ k - 1).Prime) (hhi : (x ^ k + 1).Prime) : x ^ k = 4 := by
  obtain ⟨hx2, hkp⟩ := Nat.prime_of_pow_sub_one_prime (by omega) hlo
  subst hx2
  rcases Nat.Prime.eq_two_or_odd' hkp with hk2 | hkodd
  · rw [hk2]; norm_num
  · exfalso
    have h3 : 3 ∣ 2 ^ k + 1 := three_dvd_two_pow_add_one hkodd
    have hbig : 3 < 2 ^ k + 1 := by
      have : 2 ^ 2 ≤ 2 ^ k := Nat.pow_le_pow_right (by norm_num) hk
      omega
    have := (Nat.Prime.eq_one_or_self_of_dvd hhi 3 h3)
    omega

/-- A landing one below a perfect power is impossible too, for the trivial reason: a proper power
is composite.  Stated for the record, since it removes the other obvious shape. -/
theorem not_prime_of_pow {x k : ℕ} (hx : 2 ≤ x) (hk : 2 ≤ k) : ¬ (x ^ k).Prime := by
  intro h
  have hdvd : x ∣ x ^ k := dvd_pow_self x (by omega)
  rcases (Nat.Prime.eq_one_or_self_of_dvd h x hdvd) with h1 | h2
  · omega
  · have : x ^ 2 ≤ x ^ k := Nat.pow_le_pow_right (by omega) hk
    nlinarith [this, h2]

end MirrorWalk
