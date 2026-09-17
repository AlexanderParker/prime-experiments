/-
MirrorWalkCarry (round 64, 2026-09-17): how many gears a mirror can carry, exactly.

The mirror is what switches gears off: a gear dividing the mirror's product never strikes the
family (`mirror_gear_never_strikes`).  So the natural wish is to carry every gear.  The window
forbids it, and this file says by how much.

A mirror of product `M` lands no closer to home than `2 M`, so a landing inside the window
`(q, q²]` forces `2 M ≤ q²`.  Every carried gear is at least 2, so the product of `c` carried
gears is at least `2^c`:

    2 ^ (carried gears) ≤ M ≤ q² / 2,      hence   carried gears ≤ log₂ (q²).

That is `carried_le_log`: a mirror that fits the window carries at most about `2 log₂ q` gears,
against the `π(q)` gears the machine has.  Everything else must be dodged by the choice of
period, and `mirror_times_candidates` (round 56) bounds the periods the window affords.  The two
together are the trade in its sharpest form: the mirror buys gears logarithmically and the
window pays for them geometrically.
-/
import Mathlib

namespace MirrorWalk

/-- **The carried gears are at most logarithmic.**  If every gear of `S` is at least 2 and the
product of `S` is at most `N`, then `2 ^ |S| ≤ N`. -/
theorem two_pow_card_le {S : Finset ℕ} (hS : ∀ g ∈ S, 2 ≤ g) {N : ℕ}
    (hprod : ∏ g ∈ S, g ≤ N) : 2 ^ S.card ≤ N :=
  le_trans (Finset.pow_card_le_prod S _ 2 hS) hprod

/-- The same, read as a bound on the number of carried gears. -/
theorem carried_le_log {S : Finset ℕ} (hS : ∀ g ∈ S, 2 ≤ g) {N : ℕ} (hN : 0 < N)
    (hprod : ∏ g ∈ S, g ≤ N) : S.card ≤ Nat.log 2 N := by
  have h1 : 2 ^ S.card ≤ N := two_pow_card_le hS hprod
  exact Nat.le_log_of_pow_le (by norm_num) h1

/-- **The mirror that fits the window.**  A landing at `2 M k` inside `(q, q²]` forces
`2 M ≤ q²`, so the mirror carries at most `log₂ (q²)` gears. -/
theorem mirror_in_window_carries_le {S : Finset ℕ} (hS : ∀ g ∈ S, 2 ≤ g) {M k q : ℕ}
    (hM : ∏ g ∈ S, g = M) (hk : 1 ≤ k) (hq : 0 < q) (hwin : 2 * M * k ≤ q ^ 2) :
    S.card ≤ Nat.log 2 (q ^ 2) := by
  have hMk : M ≤ q ^ 2 := by nlinarith [hwin, hk]
  exact carried_le_log hS (by positivity) (by omega)

/-- **What is left to dodge.**  Of the machine's gears `G`, all but at most `log₂ (q²)` are
uncarried, and those are exactly the ones whose teeth the period has to miss. -/
theorem uncarried_card {G S : Finset ℕ} (hS : ∀ g ∈ S, 2 ≤ g) {M k q : ℕ}
    (hM : ∏ g ∈ S, g = M) (hk : 1 ≤ k) (hq : 0 < q) (hwin : 2 * M * k ≤ q ^ 2) :
    G.card - Nat.log 2 (q ^ 2) ≤ (G \ S).card := by
  have h1 : S.card ≤ Nat.log 2 (q ^ 2) := mirror_in_window_carries_le hS hM hk hq hwin
  have hsub : G ⊆ (G \ S) ∪ S := by
    intro x hx
    by_cases h : x ∈ S
    · exact Finset.mem_union_right _ h
    · exact Finset.mem_union_left _ (Finset.mem_sdiff.mpr ⟨hx, h⟩)
  have h2 := Finset.card_le_card hsub
  have h3 := Finset.card_union_le (G \ S) S
  omega

end MirrorWalk
