/-
StretchRule (round 93, 2026-09-19): what a gear to come can do, exactly.

Asked what properties a future gear would need to kill a twin slot.  The machine's own rules
answer it, and the answer is that a future gear can do exactly one thing in the stretch it opens.

Let `p < q` be consecutive gears.  The machine `q` extends the window from `p²` to `q²`, and the
new stretch `(p², q²]` is where the new gear could matter.  It cannot:

  `new_gear_only_square`: a member `n` in `(p², q²]` divisible by `q` is either `q²` itself or
  already divisible by some gear at most `p`.

Because `n = q k` with `k ≤ q`; `k = q` gives the square; `k = 1` gives `q` itself, which lies
below `p²`; and `2 ≤ k < q` gives `k` a prime factor below `q`, hence at most `p`.  So among the
columns of the new stretch that the gears up to `p` leave open, the new gear strikes only the
column of its own square.  Everything else the gears up to `p` left open there is a twin prime
pair of the machine `q` (`section_twin_of_unstruck`, the square-root rule).

Consequences.  A slot near `N` is decided by the gears up to `√N` and by nothing that arrives
later.  A twin-free window for the machine `q` would need every stretch `(p_i², p_{i+1}²]` inside
it to be twin-free, and in each of them the gears up to `p_i` would have to leave open nothing but
the square column - a stretch of about `p_i × gap / 3` columns covered by the gears below it with
a single exception.  Measured (research/stack/r8, entry 99): across the 666 consecutive-gear
stretches up to `q = 5000` none is twin-free; the fewest twins in a stretch is 2, at the
narrowest stretches.
-/
import Mathlib

namespace MirrorWalk

/-- **A new gear adds only its square.**  For consecutive gears `p < q` (no prime strictly
between them) and a member `n` with `p² < n ≤ q²`: if `q ∣ n` then `n = q²` or some prime at
most `p` divides `n`. -/
theorem new_gear_only_square {p q n : ℕ} (hq : q.Prime) (hpq : p < q)
    (hcons : ∀ r : ℕ, r.Prime → r < q → r ≤ p) (hqp : q ≤ p ^ 2)
    (hlo : p ^ 2 < n) (hhi : n ≤ q ^ 2) (hdvd : q ∣ n) :
    n = q ^ 2 ∨ ∃ r : ℕ, r.Prime ∧ r ≤ p ∧ r ∣ n := by
  obtain ⟨k, hk⟩ := hdvd
  have hq0 : 0 < q := hq.pos
  have hkq : k ≤ q := by
    by_contra h
    push_neg at h
    have : q * k > q * q := Nat.mul_lt_mul_of_pos_left h hq0
    rw [← hk] at this
    nlinarith
  rcases Nat.lt_trichotomy k q with hlt | heq | hgt
  · -- k < q
    rcases Nat.lt_or_ge k 2 with h1 | h2
    · -- k = 0 or 1
      interval_cases k
      · exfalso; rw [hk] at hlo; simp at hlo
      · exfalso; rw [hk, mul_one] at hlo; omega
    · right
      obtain ⟨r, hr, hrk⟩ := Nat.exists_prime_and_dvd (by omega : k ≠ 1)
      have hrk' : r ≤ k := Nat.le_of_dvd (by omega) hrk
      have hrq : r < q := by omega
      exact ⟨r, hr, hcons r hr hrq, by rw [hk]; exact Dvd.dvd.mul_left hrk q⟩
  · left; rw [hk, heq, pow_two]
  · exfalso; omega

end MirrorWalk
