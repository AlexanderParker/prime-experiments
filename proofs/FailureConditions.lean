/-
FailureConditions (round 84, 2026-09-18): what a failing window would need from its top gears.

Split the machine's gears at a cut.  The gears up to the cut leave survivors in the window; a
failure needs the gears above the cut to strike every survivor.  For the top cut, `q / 2`, the
machine itself constrains how a top gear can strike a survivor at all:

  `top_gear_cofactor`: if a gear `h` in `(q/2, q]` divides a member `n ≤ q²` that has no prime
  factor up to `q/2`, then `n = h` or `n = h p` with `p` a prime in `(q/2, 2q)`.

So a top gear strikes a survivor only when the cofactor is itself a single large prime - it cannot
strike survivors "freely" the way it strikes ordinary columns.  This is a rigid feature of the
residues, not a count: it says where the top gears' teeth CAN fall on the survivor set.

Measured (research/stack/r8/failure_conditions.py): at `q = 5003` the gears in `(q/2, q]` strike
12.6% of the survivors of the gears up to `q/2`, against 15.5% if their teeth fell on survivors
in proportion to their share - a ratio of 0.81 - while a failure would need 100%, a ratio of 6.4.
At the cuts `q^0.75` and `q^0.5` the ratios are 1.11 and 1.03 against needs of 2.3 and 1.35.  The
property a failure requires - top gears far MORE efficient on survivors than their share - is
absent, and at the top cut the real machine runs the other way.
-/
import Mathlib

namespace MirrorWalk

/-- **A top gear strikes a survivor only through a single large prime cofactor.** -/
theorem top_gear_cofactor {q h n : ℕ} (hq : 8 ≤ q) (hh : h.Prime) (hlo : q < 2 * h)
    (hhi : h ≤ q) (hn : n ≤ q ^ 2)
    (hrough : ∀ p : ℕ, p.Prime → 2 * p ≤ q → ¬ (p ∣ n)) (hdvd : h ∣ n) :
    n = h ∨ ∃ p : ℕ, p.Prime ∧ q < 2 * p ∧ p < 2 * q ∧ n = h * p := by
  obtain ⟨k, hk⟩ := hdvd
  have hn0 : n ≠ 0 := by
    intro h0
    exact hrough 2 Nat.prime_two (by omega) (h0 ▸ dvd_zero 2)
  have hk0 : k ≠ 0 := by
    intro h0; apply hn0; rw [hk, h0, mul_zero]
  have hh2 : 2 ≤ h := hh.two_le
  by_cases hk1 : k = 1
  · left; rw [hk, hk1, mul_one]
  · right
    obtain ⟨p, hp, hpk⟩ := Nat.exists_prime_and_dvd hk1
    obtain ⟨k', hk'⟩ := hpk
    have hpn : p ∣ n := ⟨h * k', by rw [hk, hk']; ring⟩
    have hpbig : q < 2 * p := by
      by_contra hle
      push_neg at hle
      exact hrough p hp hle hpn
    have hk'0 : k' ≠ 0 := by
      intro h0; apply hk0; rw [hk', h0, mul_zero]
    by_cases hk'1 : k' = 1
    · refine ⟨p, hp, hpbig, ?_, by rw [hk, hk', hk'1, mul_one]⟩
      -- h * p = n ≤ q², h > q/2, so p < 2q
      have hhp : h * p ≤ q ^ 2 := by rw [hk, hk', hk'1, mul_one] at hn; exact hn
      by_contra hge
      push_neg at hge
      have : h * p ≥ h * (2 * q) := Nat.mul_le_mul_left h hge
      nlinarith
    · exfalso
      -- a second large prime factor makes n exceed q²
      obtain ⟨p', hp', hp'k⟩ := Nat.exists_prime_and_dvd hk'1
      obtain ⟨k'', hk''⟩ := hp'k
      have hp'n : p' ∣ n := ⟨h * p * k'', by rw [hk, hk', hk'']; ring⟩
      have hp'big : q < 2 * p' := by
        by_contra hle
        push_neg at hle
        exact hrough p' hp' hle hp'n
      have hk''0 : 1 ≤ k'' := by
        rcases Nat.eq_zero_or_pos k'' with h0 | hpos
        · exfalso; apply hk'0; rw [hk'', h0, mul_zero]
        · exact hpos
      have hprod : h * p * p' ≤ n := by
        rw [hk, hk', hk'']
        calc h * p * p' = h * p * p' * 1 := by ring
          _ ≤ h * p * p' * k'' := Nat.mul_le_mul_left _ hk''0
          _ = h * (p * (p' * k'')) := by ring
      -- each of h, p, p' exceeds q/2, so the product exceeds q³/8 ≥ q² for q ≥ 8
      have h1 : q + 1 ≤ 2 * h := hlo
      have h2 : q + 1 ≤ 2 * p := hpbig
      have h3 : q + 1 ≤ 2 * p' := hp'big
      have : 8 * (h * p * p') ≥ (q + 1) * (q + 1) * (q + 1) := by
        calc 8 * (h * p * p') = (2 * h) * (2 * p) * (2 * p') := by ring
          _ ≥ (q + 1) * (q + 1) * (q + 1) := by
            apply Nat.mul_le_mul (Nat.mul_le_mul h1 h2) h3
      nlinarith

end MirrorWalk
