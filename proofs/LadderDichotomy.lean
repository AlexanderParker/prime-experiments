/-
LadderDichotomy (2026-09-20, prover lane attempt B): two local hypotheses give chains.

`NoConsecutiveLeaves`: two consecutive twin centres are never both rungless.  `NoSingleRung`: no
twin centre has exactly one rung.  Together they give a rung with a rung from any twin centre
with a rung: take two distinct rungs `a < b` of `s`, let `t` be the first twin centre after `a`
(so `t ≤ b` and `t` is a rung of `s`); `a` and `t` are consecutive twin centres, so one of them
has a rung.  Iterating gives chains of every length (`ChainHyp`), hence twins unbounded; from
`6` it gives a node at every depth.  (Tree node R5.f.xxviii.)
-/
import LadderDepth

namespace TwinLadder

/-- A twin centre with a rung. -/
def Good (s : ℕ) : Prop := TwinCentre s ∧ ∃ r, Rung s r

/-- **Leaves are never consecutive**: of two consecutive twin centres, one has a rung. -/
def NoConsecutiveLeaves : Prop :=
  ∀ s t, TwinCentre s → TwinCentre t → s < t → (∀ u, s < u → u < t → ¬ TwinCentre u) →
    (∃ r, Rung s r) ∨ (∃ r, Rung t r)

/-- **No single rung**: a twin centre with a rung has a second one. -/
def NoSingleRung : Prop :=
  ∀ s, TwinCentre s → ∀ r, Rung s r → ∃ r', Rung s r' ∧ r' ≠ r

/-- The step: a twin centre with a rung has a rung which itself has a rung. -/
theorem good_step (hNC : NoConsecutiveLeaves) (hD : NoSingleRung) {s : ℕ} (hs : Good s) :
    ∃ r, Rung s r ∧ Good r := by
  classical
  obtain ⟨hts, r₁, hr₁⟩ := hs
  obtain ⟨r₂, hr₂, hne⟩ := hD s hts r₁ hr₁
  have key : ∀ a b, Rung s a → Rung s b → a < b → ∃ r, Rung s r ∧ Good r := by
    intro a b ha hb hab
    have hex : ∃ u, a < u ∧ u ≤ b ∧ TwinCentre u := ⟨b, hab, le_rfl, hb.1⟩
    have ht : a < Nat.find hex ∧ Nat.find hex ≤ b ∧ TwinCentre (Nat.find hex) := Nat.find_spec hex
    have hmin : ∀ u, a < u → u < Nat.find hex → ¬ TwinCentre u := by
      intro u hau hut hu
      exact Nat.find_min hex hut ⟨hau, by omega, hu⟩
    have hrt : Rung s (Nat.find hex) :=
      ⟨ht.2.2, by have := ha.2.1; omega, by have := hb.2.2; omega⟩
    rcases hNC a (Nat.find hex) ha.1 ht.2.2 ht.1 hmin with ⟨r, hr⟩ | ⟨r, hr⟩
    · exact ⟨a, ha, ha.1, r, hr⟩
    · exact ⟨Nat.find hex, hrt, ht.2.2, r, hr⟩
  rcases Nat.lt_or_gt_of_ne hne with h | h
  · exact key r₂ r₁ hr₂ hr₁ h
  · exact key r₁ r₂ hr₁ hr₂ h

/-- Chains of every length from any twin centre with a rung. -/
theorem chain_of_good (hNC : NoConsecutiveLeaves) (hD : NoSingleRung) {s₀ : ℕ} (h₀ : Good s₀) :
    ∀ n, ∃ f : ℕ → ℕ, Chain n f ∧ f 0 = s₀ ∧ Good (f n) := by
  intro n
  induction n with
  | zero => exact ⟨fun _ => s₀, ⟨h₀.1, fun k hk => by omega⟩, rfl, h₀⟩
  | succ k ih =>
    obtain ⟨f, hf, hf0, hg⟩ := ih
    obtain ⟨r, hr, hgr⟩ := good_step hNC hD hg
    refine ⟨fun i => if i ≤ k then f i else r, ⟨?_, ?_⟩, ?_, ?_⟩
    · show TwinCentre (if 0 ≤ k then f 0 else r)
      rw [if_pos (Nat.zero_le k)]; exact hf.1
    · intro i hi
      by_cases hik : i < k
      · have h1 : i ≤ k := by omega
        have h2 : i + 1 ≤ k := by omega
        show Rung (if i ≤ k then f i else r) (if i + 1 ≤ k then f (i + 1) else r)
        rw [if_pos h1, if_pos h2]
        exact hf.2 i hik
      · have hik' : i = k := by omega
        subst hik'
        show Rung (if i ≤ i then f i else r) (if i + 1 ≤ i then f (i + 1) else r)
        rw [if_pos le_rfl, if_neg (by omega)]
        exact hr
    · show (if 0 ≤ k then f 0 else r) = s₀
      rw [if_pos (Nat.zero_le k)]; exact hf0
    · show Good (if k + 1 ≤ k then f (k + 1) else r)
      rw [if_neg (by omega)]; exact hgr

/-- **The dichotomy theorem**: no consecutive leaves, no single rung, and one twin centre with a
rung give chains of every length. -/
theorem chainHyp_of_dichotomy (hNC : NoConsecutiveLeaves) (hD : NoSingleRung)
    {s₀ : ℕ} (h₀ : Good s₀) : ChainHyp := by
  intro n
  obtain ⟨f, hf, _, _⟩ := chain_of_good hNC hD h₀ n
  exact ⟨f, hf⟩

/-- `6` has the rung `30`. -/
theorem good_six : Good 6 :=
  ⟨twinCentre_six, 30, ⟨⟨⟨5, rfl⟩, by norm_num, by norm_num⟩, by norm_num, by norm_num⟩⟩

/-- **Twins unbounded from the two local hypotheses** (rooted at `(5, 7)`). -/
theorem twins_unbounded_of_dichotomy (hNC : NoConsecutiveLeaves) (hD : NoSingleRung) :
    ∀ N : ℕ, ∃ m : ℕ, N < 6 * m - 1 ∧ (6 * m - 1).Prime ∧ (6 * m + 1).Prime :=
  twins_unbounded_of_chains (chainHyp_of_dichotomy hNC hD good_six)

/-- From `6` the two hypotheses give a node at every depth of the `(5, 7)` tree. -/
theorem depthHyp_of_dichotomy (hNC : NoConsecutiveLeaves) (hD : NoSingleRung) : DepthHyp := by
  intro n
  obtain ⟨f, hf, hf0, _⟩ := chain_of_good hNC hD good_six n
  have key : ∀ k, k ≤ n → Depth k (f k) := by
    intro k
    induction k with
    | zero => intro _; rw [hf0]; exact Depth.root
    | succ k ih => intro hk; exact Depth.step (ih (by omega)) (hf.2 k (by omega))
  exact ⟨f n, key n le_rfl⟩

end TwinLadder
