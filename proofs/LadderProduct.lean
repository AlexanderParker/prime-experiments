/-
LadderProduct (2026-09-20, routes lane route 2): the product forest.

For twin centres `s ≤ t` the product window is the open interval `((s-1)(t-1), (s+1)(t+1))`; a
twin centre inside it is a product-rung of the pair.  With `s = t` this is the square stretch
and the rung of `TwinLadderTheorem`; with `s = 6` it is the window `(5(t-1), 7(t+1))`, a
constant-ratio window.  ProductHyp - every twin centre is the larger factor of a pair with a
product-rung - is weaker than LadderHyp and still gives twins above every bound, since a
product-rung exceeds `5t - 4 ≥ t + 2`.  (Tree node R5.f.xxiv.b.)
-/
import TwinLadderTheorem

namespace TwinLadder

/-- A product-rung of the pair `(s, t)`: a twin centre strictly inside `((s-1)(t-1), (s+1)(t+1))`. -/
def ProductRung (s t u : ℕ) : Prop :=
  TwinCentre u ∧ (s - 1) * (t - 1) < u - 1 ∧ u + 1 < (s + 1) * (t + 1)

/-- **The product hypothesis**: every twin centre `t` has a product-rung with some twin centre
`s ≤ t`. -/
def ProductHyp : Prop :=
  ∀ t : ℕ, TwinCentre t → ∃ s u : ℕ, TwinCentre s ∧ s ≤ t ∧ ProductRung s t u

/-- A product-rung climbs: `u ≥ t + 2`. -/
theorem productRung_gt {s t u : ℕ} (hs : TwinCentre s) (ht : TwinCentre t)
    (h : ProductRung s t u) : t + 2 ≤ u := by
  have h6s := twinCentre_ge_six hs
  have h6t := twinCentre_ge_six ht
  have h1 : (s - 1) * (t - 1) < u - 1 := h.2.1
  have h2 : t ≤ (s - 1) * (t - 1) := by
    obtain ⟨a, rfl⟩ : ∃ a, s = a + 1 := ⟨s - 1, by omega⟩
    obtain ⟨b, rfl⟩ : ∃ b, t = b + 1 := ⟨t - 1, by omega⟩
    simp only [Nat.add_sub_cancel]
    nlinarith
  omega

/-- The square rung is the diagonal product-rung. -/
theorem productRung_of_rung {s u : ℕ} (h : Rung s u) : ProductRung s s u := by
  refine ⟨h.1, ?_, ?_⟩
  · have := h.2.1; rwa [sq] at this
  · have := h.2.2; rwa [sq] at this

/-- The ladder hypothesis implies the product hypothesis (take `s = t`). -/
theorem productHyp_of_ladderHyp (hL : LadderHyp) : ProductHyp := by
  intro t ht
  obtain ⟨u, hu⟩ := hL t ht
  exact ⟨t, u, ht, le_rfl, productRung_of_rung hu⟩

/-- Under the product hypothesis twin centres are unbounded. -/
theorem twinCentre_unbounded_of_product (hP : ProductHyp) :
    ∀ N : ℕ, ∃ s : ℕ, TwinCentre s ∧ N ≤ s := by
  intro N
  induction N with
  | zero => exact ⟨6, twinCentre_six, Nat.zero_le _⟩
  | succ n ih =>
    obtain ⟨t, ht, hn⟩ := ih
    obtain ⟨s, u, hs, _, hu⟩ := hP t ht
    exact ⟨u, hu.1, by have := productRung_gt hs ht hu; omega⟩

/-- **Twins unbounded from the product hypothesis.** -/
theorem twins_unbounded_of_product (hP : ProductHyp) :
    ∀ N : ℕ, ∃ m : ℕ, N < 6 * m - 1 ∧ (6 * m - 1).Prime ∧ (6 * m + 1).Prime := by
  intro N
  obtain ⟨s, ⟨⟨m, rfl⟩, hp1, hp2⟩, hN⟩ := twinCentre_unbounded_of_product hP (N + 2)
  exact ⟨m, by omega, hp1, hp2⟩

/-- The multiplier-6 form: a twin centre in `(5(t-1), 7(t+1))` for every twin centre `t`. -/
def SixHyp : Prop := ∀ t : ℕ, TwinCentre t → ∃ u : ℕ, ProductRung 6 t u

theorem productHyp_of_sixHyp (h6 : SixHyp) : ProductHyp := by
  intro t ht
  obtain ⟨u, hu⟩ := h6 t ht
  exact ⟨6, u, twinCentre_six, twinCentre_ge_six ht, hu⟩

end TwinLadder
