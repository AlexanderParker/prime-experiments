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
import MirrorWalkConditional

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

/-- **A dead run of a fixed gear set is finite: shorter than the product of the gears.**  Every
column that is a multiple of the product `P` is open to every gear dividing `P` - the gear would
have to divide 1 - so no run of struck columns can span a full period.  (The measured runs are far
shorter: the paired Jacobsthal length, 33 columns for the gears to 23.) -/
theorem open_at_multiple_of_product {h P t : ℕ} (hh : 1 < h) (hP : h ∣ P) (ht : 1 ≤ t)
    (hP0 : 1 ≤ P) :
    ¬ (h ∣ 6 * (P * t) - 1) ∧ ¬ (h ∣ 6 * (P * t) + 1) := by
  have hdvd : h ∣ 6 * (P * t) := Dvd.dvd.mul_left (hP.mul_right t) 6
  have hpos : 1 ≤ 6 * (P * t) := by
    have : 1 ≤ P * t := Nat.one_le_iff_ne_zero.mpr (by positivity)
    omega
  constructor
  · intro hd
    have := Nat.dvd_sub hdvd hd
    rw [Nat.sub_sub_self hpos] at this
    have := Nat.le_of_dvd one_pos this
    omega
  · intro hd
    have h1 := Nat.dvd_sub hd hdvd
    have e : 6 * (P * t) + 1 - 6 * (P * t) = 1 := by omega
    rw [e] at h1
    have := Nat.le_of_dvd one_pos h1
    omega

/-- **How the later primes kill, exactly.**  A member above `p²` and below `p³` that no gear up
to `p` strikes is a prime, the square of a prime above `p`, or the product of two primes above `p`.
So in a run that the gears up to `p` cannot finish, every remaining kill is a square or a product
of two later primes: the later primes act only in pairs, never singly. -/
theorem rough_member_form {p n : ℕ} (hp : 1 ≤ p) (hlo : p ^ 2 < n) (hhi : n < p ^ 3)
    (hrough : ∀ r : ℕ, r.Prime → r ≤ p → ¬ (r ∣ n)) :
    n.Prime ∨ (∃ a b : ℕ, a.Prime ∧ b.Prime ∧ p < a ∧ p < b ∧ n = a * b) := by
  have hn1 : n ≠ 1 := by
    intro h; rw [h] at hlo; nlinarith
  have hn0 : n ≠ 0 := by
    intro h; rw [h] at hlo; omega
  set a := n.minFac with ha
  have hap : a.Prime := Nat.minFac_prime hn1
  have hadvd : a ∣ n := Nat.minFac_dvd n
  have hpa : p < a := by
    by_contra hle; push_neg at hle
    exact hrough a hap hle hadvd
  obtain ⟨k, hk⟩ := hadvd
  by_cases hk1 : k = 1
  · left; rw [hk, hk1, mul_one]; exact hap
  · right
    have hk0 : k ≠ 0 := by intro h; apply hn0; rw [hk, h, mul_zero]
    -- every prime factor of k is at least a, since a is the least prime factor of n
    have hkmin : ∀ r : ℕ, r.Prime → r ∣ k → a ≤ r := by
      intro r hr hrk
      have hrn : r ∣ n := ⟨a * (k / r), by
        obtain ⟨j, hj⟩ := hrk
        rw [hk, hj]; rw [Nat.mul_div_cancel_left j hr.pos]; ring⟩
      rw [ha]; exact Nat.minFac_le_of_dvd hr.two_le hrn
    by_cases hkp : k.Prime
    · exact ⟨a, k, hap, hkp, hpa, by have := hkmin k hkp (dvd_refl k); omega, hk⟩
    · exfalso
      -- k composite with all prime factors ≥ a gives k ≥ a², hence n ≥ a³ > p³
      obtain ⟨r, hr, hrk⟩ := Nat.exists_prime_and_dvd hk1
      obtain ⟨j, hj⟩ := hrk
      have hj1 : j ≠ 1 := by
        intro h; apply hkp; rw [hj, h, mul_one]; exact hr
      have hj0 : j ≠ 0 := by intro h; apply hk0; rw [hj, h, mul_zero]
      obtain ⟨s, hs, hsj⟩ := Nat.exists_prime_and_dvd hj1
      have har : a ≤ r := hkmin r hr ⟨j, hj⟩
      have has : a ≤ s := hkmin s hs (by
        obtain ⟨t, ht⟩ := hsj
        exact ⟨r * t, by rw [hj, ht]; ring⟩)
      have hsle : s ≤ j := Nat.le_of_dvd (by omega) hsj
      have hk2 : a * a ≤ k := by
        calc a * a ≤ r * s := Nat.mul_le_mul har has
          _ ≤ r * j := Nat.mul_le_mul_left r hsle
          _ = k := hj.symm
      have hn3 : a * (a * a) ≤ n := by rw [hk]; exact Nat.mul_le_mul_left a hk2
      have : p ^ 3 < a * (a * a) := by
        have h1 : p + 1 ≤ a := hpa
        calc p ^ 3 = p * (p * p) := by ring
          _ < (p + 1) * ((p + 1) * (p + 1)) := by nlinarith
          _ ≤ a * (a * a) := by
            apply Nat.mul_le_mul h1
            exact Nat.mul_le_mul h1 h1
      omega

/-- **The plug law.**  Let `p₁ < p₂ < p₃` be consecutive gears.  In the stretch `(p₂², p₃²]`, a
member that the gears up to `p₁` leave open and that `p₂` strikes is `p₂ k` with `k` a prime at
least `p₃`.  So the newly established gear plugs the base's holes only at the products of itself
with the next primes - a sparse set fixed in advance by the primes themselves. -/
theorem plug_law {p₁ p₂ p₃ n : ℕ} (hp₂ : p₂.Prime)
    (hcons₁ : ∀ r : ℕ, r.Prime → r < p₂ → r ≤ p₁)
    (hcons₂ : ∀ r : ℕ, r.Prime → p₂ < r → p₃ ≤ r)
    (hrough : ∀ r : ℕ, r.Prime → r ≤ p₁ → ¬ (r ∣ n))
    (hlo : p₂ ^ 2 < n) (hhi : n ≤ p₃ ^ 2) (hsize : p₃ ^ 2 < p₂ ^ 3) (hdvd : p₂ ∣ n) :
    ∃ k : ℕ, k.Prime ∧ p₃ ≤ k ∧ n = p₂ * k := by
  obtain ⟨k, hk⟩ := hdvd
  have hp0 : 0 < p₂ := hp₂.pos
  have hp2 : 2 ≤ p₂ := hp₂.two_le
  have hkbig : p₂ < k := by
    by_contra h; push_neg at h
    have : p₂ * k ≤ p₂ * p₂ := Nat.mul_le_mul_left _ h
    rw [← hk] at this; nlinarith
  -- every prime factor of k exceeds p₂
  have hfac : ∀ r : ℕ, r.Prime → r ∣ k → p₂ < r := by
    intro r hr hrk
    have hrn : r ∣ n := by rw [hk]; exact Dvd.dvd.mul_left hrk p₂
    rcases Nat.lt_trichotomy r p₂ with hlt | heq | hgt
    · exact absurd hrn (hrough r hr (hcons₁ r hr hlt))
    · exfalso
      rw [heq] at hrk
      obtain ⟨j, hj⟩ := hrk
      by_cases hj' : j = 1
      · rw [hj', mul_one] at hj; omega
      · obtain ⟨s, hs, hsj⟩ := Nat.exists_prime_and_dvd hj'
        have hsn : s ∣ n := by
          rw [hk, hj]
          exact Dvd.dvd.mul_left (Dvd.dvd.mul_left hsj p₂) p₂
        have hs_big : p₂ ≤ s := by
          by_contra hlt; push_neg at hlt
          exact hrough s hs (hcons₁ s hs hlt) hsn
        have hj0 : 0 < j := by
          rcases Nat.eq_zero_or_pos j with h0 | hpos
          · exfalso; rw [h0, mul_zero] at hj; omega
          · exact hpos
        have hsle : s ≤ j := Nat.le_of_dvd hj0 hsj
        have : p₂ * (p₂ * p₂) ≤ n := by
          calc p₂ * (p₂ * p₂) ≤ p₂ * (p₂ * j) := by
                apply Nat.mul_le_mul_left; apply Nat.mul_le_mul_left; omega
            _ = n := by rw [hk, hj]
        nlinarith
    · exact hgt
  -- k has no prime factor at most p₂ and is below p₃², so k is prime
  have hk1 : k ≠ 1 := by omega
  obtain ⟨a, ha, hak⟩ := Nat.exists_prime_and_dvd hk1
  have ha_big : p₂ < a := hfac a ha hak
  have ha3 : p₃ ≤ a := hcons₂ a ha ha_big
  obtain ⟨j, hj⟩ := hak
  by_cases hj1 : j = 1
  · refine ⟨k, ?_, ?_, hk⟩
    · rw [hj, hj1, mul_one]; exact ha
    · rw [hj, hj1, mul_one]; exact ha3
  · exfalso
    obtain ⟨b, hb, hbj⟩ := Nat.exists_prime_and_dvd hj1
    have hbk : b ∣ k := by rw [hj]; exact Dvd.dvd.mul_left hbj a
    have hb_big : p₂ < b := hfac b hb hbk
    have hb3 : p₃ ≤ b := hcons₂ b hb hb_big
    have hj0 : 0 < j := by
      rcases Nat.eq_zero_or_pos j with h0 | hpos
      · exfalso; rw [h0, mul_zero] at hj; omega
      · exact hpos
    have hble : b ≤ j := Nat.le_of_dvd hj0 hbj
    have hk3 : p₃ * p₃ ≤ k := by
      calc p₃ * p₃ ≤ a * b := Nat.mul_le_mul ha3 hb3
        _ ≤ a * j := Nat.mul_le_mul_left a hble
        _ = k := hj.symm
    have : p₂ * (p₃ * p₃) ≤ n := by rw [hk]; exact Nat.mul_le_mul_left p₂ hk3
    nlinarith

/-- **A kill needs the run to reach the stretch's end.**  If every column of the stretch is struck
by the machine's gears, then the least open column above `p²` lies beyond `q²`: the struck run
starting at the square spans the whole stretch.  With the stretch rule this is the reduction of
concept 1 of the attack plan: a dead stretch is a struck run above `p²` at least as long as the
stretch, and nothing else. -/
theorem kill_needs_run {G : Finset ℕ} {p q : ℕ}
    (hdead : ∀ m : ℕ, p ^ 2 < 6 * m - 1 → 6 * m + 1 ≤ q ^ 2 → SquareColumn.StruckBy G m)
    {m : ℕ} (hm : p ^ 2 < 6 * m - 1) (hopen : ¬ SquareColumn.StruckBy G m) :
    q ^ 2 < 6 * m + 1 := by
  by_contra h
  push_neg at h
  exact hopen (hdead m hm h)

end MirrorWalk
