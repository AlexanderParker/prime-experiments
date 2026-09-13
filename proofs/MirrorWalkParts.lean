/-
MirrorWalkParts: the walk taken apart (owner, 2026-09-13): the origin pair, the step rule, how
each landing relates to its last step, one gear at a time, and what a walk can carry into the
window.  Each part is its own theorem.
-/
import MirrorWalk
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.NumberTheory.Primorial
import Mathlib.Data.Nat.Prime.Int
import Mathlib.RingTheory.Int.Basic

namespace MirrorWalk

/-! ## Part 1: origins -/

/-- A gear pair `(g, g+2)` (both prime) is open to every prime gear other than its two members. -/
theorem pair_open {g h : ℕ} (hg : g.Prime) (hg2 : (g + 2).Prime) (hh : h.Prime)
    (h1 : h ≠ g) (h2 : h ≠ g + 2) : OpenTo h (g : ℤ) := by
  unfold OpenTo
  constructor
  · intro hd
    have : h ∣ g := by exact_mod_cast hd
    rcases (Nat.dvd_prime hg).mp this with h' | h'
    · exact hh.one_lt.ne' h'
    · exact h1 h'
  · intro hd
    have : h ∣ g + 2 := by exact_mod_cast hd
    rcases (Nat.dvd_prime hg2).mp this with h' | h'
    · exact hh.one_lt.ne' h'
    · exact h2 h'

/-! ## Part 3: how a landing relates to its last step, gear by gear -/

/-- **The landing law.**  Gear `h` strikes the landing of the flip about `a` from `n` iff the
axis product `2a` is congruent to a member of the origin column: `2a ≡ n + 2` (then `h` divides
the landing's left member) or `2a ≡ n` (its right member), modulo `h`. -/
theorem struck_flip_iff {h : ℕ} {a n : ℤ} :
    ((h : ℤ) ∣ flip a n ∨ (h : ℤ) ∣ flip a n + 2) ↔
      (2 * a ≡ n + 2 [ZMOD h] ∨ 2 * a ≡ n [ZMOD h]) := by
  have e1 : flip a n = -((n + 2) - 2 * a) := by unfold flip; ring
  have e2 : flip a n + 2 = -(n - 2 * a) := by unfold flip; ring
  rw [e2, e1, dvd_neg, dvd_neg, Int.modEq_iff_dvd, Int.modEq_iff_dvd]

/-- Two axes whose landings are both struck on the same side by `h` are congruent modulo `h`
whenever `h` is a prime not dividing `2M`: the step choice is decided by one residue class. -/
theorem same_class_of_struck {h : ℕ} (hh : h.Prime) {M n : ℤ} (hM : ¬ (h : ℤ) ∣ 2 * M)
    {k k' : ℤ} (hk : (h : ℤ) ∣ 2 * (k * M) - n) (hk' : (h : ℤ) ∣ 2 * (k' * M) - n) :
    (h : ℤ) ∣ k - k' := by
  have hd : (h : ℤ) ∣ (2 * M) * (k - k') := by
    have := dvd_sub hk hk'
    have e : 2 * (k * M) - n - (2 * (k' * M) - n) = (2 * M) * (k - k') := by ring
    rwa [e] at this
  rcases Int.Prime.dvd_mul' hh hd with h1 | h1
  · exact absurd h1 hM
  · exact h1

/-- **One gear, one step.**  For a prime gear `h ≥ 5` not dividing `2M`, among any three
consecutive multiples `k₀M, (k₀+1)M, (k₀+2)M` some axis lands the origin `n` on a column open
to `h`.  So a single gear never blocks a step: at most two of its residue classes are bad. -/
theorem exists_axis_open {h : ℕ} (hh : h.Prime) (h5 : 5 ≤ h) {M n k₀ : ℤ}
    (hM : ¬ (h : ℤ) ∣ 2 * M) :
    ∃ k, k₀ ≤ k ∧ k ≤ k₀ + 2 ∧ OpenTo h (flip (k * M) n) := by
  by_contra hcon
  push_neg at hcon
  -- every k in {k₀, k₀+1, k₀+2} is struck: h divides the left member (class L) or the right (class R)
  have bad : ∀ k, k₀ ≤ k → k ≤ k₀ + 2 →
      (h : ℤ) ∣ 2 * (k * M) - (n + 2) ∨ (h : ℤ) ∣ 2 * (k * M) - n := by
    intro k hk1 hk2
    have := hcon k hk1 hk2
    unfold OpenTo at this
    push_neg at this
    by_cases hL : (h : ℤ) ∣ flip (k * M) n
    · left
      have e : flip (k * M) n = 2 * (k * M) - (n + 2) := by unfold flip; ring
      rw [e] at hL; exact hL
    · right
      have hR := this hL
      have e : flip (k * M) n + 2 = 2 * (k * M) - n := by unfold flip; ring
      rw [e] at hR; exact hR
  have h0 := bad k₀ le_rfl (by linarith)
  have h1 := bad (k₀ + 1) (by linarith) (by linarith)
  have h2 := bad (k₀ + 2) (by linarith) (by linarith)
  have hle : (h : ℤ) ≤ 2 := by
    -- two of the three share a class; their difference (1 or 2) is then divisible by h
    have key : ∀ {x y : ℤ}, (h : ℤ) ∣ x - y → 0 < x - y → x - y ≤ 2 → (h : ℤ) ≤ 2 := by
      intro x y hd hpos hle
      exact le_trans (Int.le_of_dvd hpos hd) hle
    rcases h0 with h0 | h0 <;> rcases h1 with h1 | h1 <;> rcases h2 with h2 | h2
    · exact key (same_class_of_struck hh hM h1 h0) (by linarith) (by linarith)
    · exact key (same_class_of_struck hh hM h1 h0) (by linarith) (by linarith)
    · exact key (same_class_of_struck hh hM h2 h0) (by linarith) (by linarith)
    · exact key (same_class_of_struck hh hM h2 h1) (by linarith) (by linarith)
    · exact key (same_class_of_struck hh hM h2 h1) (by linarith) (by linarith)
    · exact key (same_class_of_struck hh hM h2 h0) (by linarith) (by linarith)
    · exact key (same_class_of_struck hh hM h1 h0) (by linarith) (by linarith)
    · exact key (same_class_of_struck hh hM h1 h0) (by linarith) (by linarith)
  have : (5 : ℤ) ≤ h := by exact_mod_cast h5
  linarith

/-! ## Part 5: what a walk can carry into the window -/

/-- **The carry cap.**  Distinct primes carried by a walk all divide the axis sum `A`, so their
product is at most `A`; a landing `(2A - 1, 2A + 1)` inside the window `(q, q²]` therefore
carries at most the primes whose product stays below `q²/2 + 1`. -/
theorem carried_product_le {s : Finset ℕ} {A : ℕ} (hA : 0 < A)
    (hs : ∀ p ∈ s, p.Prime) (hd : ∀ p ∈ s, p ∣ A) : ∏ p ∈ s, p ≤ A := by
  have : ∏ p ∈ s, p ∣ A :=
    Finset.prod_primes_dvd A (fun p hp => (hs p hp).prime) hd
  exact Nat.le_of_dvd hA this

theorem carried_product_le_window {s : Finset ℕ} {A q : ℕ}
    (hs : ∀ p ∈ s, p.Prime) (hd : ∀ p ∈ s, p ∣ A) (hA : 0 < A) (hwin : 2 * A + 1 ≤ q ^ 2) :
    2 * ∏ p ∈ s, p ≤ q ^ 2 := by
  have := carried_product_le hA hs hd
  omega

end MirrorWalk
