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

/-- **A small gear always stays live.**  If the mirror carries at most `L` gears and the machine
has `L + 1` gears at most `b`, then some gear at most `b` is uncarried. -/
theorem small_gear_uncarried {G S T : Finset ℕ} (hT : T ⊆ G) (hTS : T.card = S.card + 1)
    {b : ℕ} (hb : ∀ g ∈ T, g ≤ b) : ∃ g ∈ G \ S, g ≤ b := by
  classical
  have hne : ¬ (T ⊆ S) := by
    intro hsub
    have := Finset.card_le_card hsub
    omega
  obtain ⟨g, hgT, hgS⟩ := Finset.not_subset.mp hne
  exact ⟨g, Finset.mem_sdiff.mpr ⟨hT hgT, hgS⟩, hb g hgT⟩

/-- **Carrying cannot reach the free regime.**  The free-regime step lemma
(`keeping_move_free`) needs every live gear above twice their number.  A mirror that fits the
window carries only logarithmically many gears, so a small gear is always left live, and once the
live gears are more numerous than half that small gear the hypothesis fails - for every choice of
mirror.  (Its conclusion can fail too: `keeping_move_free_sharp`.) -/
theorem free_regime_unreachable {G S T : Finset ℕ} (hT : T ⊆ G) (hTS : T.card = S.card + 1)
    {b : ℕ} (hb : ∀ g ∈ T, g ≤ b) (hlive : b ≤ 2 * (G \ S).card) :
    ¬ (∀ g ∈ G \ S, 2 * (G \ S).card < g) := by
  intro hfree
  obtain ⟨g, hg, hgb⟩ := small_gear_uncarried hT hTS hb
  have := hfree g hg
  omega

/-- **Silencing the small gears costs the primorial.**  A mirror silences exactly the gears
dividing its product, so to leave no live gear at or below `X` the product must be divisible by
every prime up to `X`, hence by the primorial `X#`.  With the window's bound `2 M k ≤ q²` this is
the sharp form of the carry wall: the price of silence is exponential in `X` while the window
pays only quadratically in `q`. -/
theorem silence_costs_primorial {M X : ℕ} (hM : 0 < M)
    (hsil : ∀ p, p.Prime → p ≤ X → p ∣ M) : primorial X ∣ M := by
  classical
  unfold primorial
  refine Finset.prod_primes_dvd M ?_ ?_
  · intro p hp
    simp only [Finset.mem_filter, Finset.mem_range] at hp
    exact hp.2.prime
  · intro p hp
    simp only [Finset.mem_filter, Finset.mem_range] at hp
    exact hsil p hp.2 (by omega)

/-- The same, as a bound: silence up to `X` forces the primorial below the mirror. -/
theorem primorial_le_of_silence {M X : ℕ} (hM : 0 < M)
    (hsil : ∀ p, p.Prime → p ≤ X → p ∣ M) : primorial X ≤ M :=
  Nat.le_of_dvd hM (silence_costs_primorial hM hsil)

end MirrorWalk
