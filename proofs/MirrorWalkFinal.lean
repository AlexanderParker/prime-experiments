/-
MirrorWalkFinal (round 50, 2026-09-14): the final step of the locator as residue avoidance.

From a column `E` (the primorial spiral's landing), the step about the mirror `{3, h}` lands at
`L = E + 6 d h` (`d = 1` up, `d = -1` down).  For a gear `g`, `g` divides the landing's left
member iff `6 d h ≡ -E (mod g)`, and its right member iff `6 d h ≡ -(E + 2) (mod g)`: two
forbidden residue classes of `h` per gear, all computed from `E`.  A high gear `h` in no
forbidden class of any gear up to `q` lands, below `q²`, on a twin prime pair (the square-root
rule).  No primality is tested; residues are.
-/
import MirrorWalkSpiral

namespace MirrorWalk

open SquareColumn

/-- **The forbidden classes.**  Gear `g` strikes the landing `E + 6 d h` iff `6 d h` is congruent
to `-E` or to `-(E + 2)` modulo `g`. -/
theorem strikes_landing_iff {g : ℕ} {E h d : ℤ} :
    ((g : ℤ) ∣ E + 6 * d * h ∨ (g : ℤ) ∣ E + 6 * d * h + 2) ↔
      (6 * d * h ≡ -E [ZMOD g] ∨ 6 * d * h ≡ -(E + 2) [ZMOD g]) := by
  rw [Int.modEq_iff_dvd, Int.modEq_iff_dvd]
  have e1 : -E - 6 * d * h = -(E + 6 * d * h) := by ring
  have e2 : -(E + 2) - 6 * d * h = -(E + 6 * d * h + 2) := by ring
  rw [e1, e2, dvd_neg, dvd_neg]

/-- Avoiding both classes is exactly openness of the landing to `g`. -/
theorem avoid_iff_open {g : ℕ} {E h d : ℤ} :
    (¬ 6 * d * h ≡ -E [ZMOD g] ∧ ¬ 6 * d * h ≡ -(E + 2) [ZMOD g]) ↔ OpenTo g (E + 6 * d * h) := by
  unfold OpenTo
  have := @strikes_landing_iff g E h d
  constructor
  · rintro ⟨h1, h2⟩
    refine ⟨fun hd => ?_, fun hd => ?_⟩
    · rcases this.mp (Or.inl hd) with h' | h' <;> contradiction
    · rcases this.mp (Or.inr hd) with h' | h' <;> contradiction
  · rintro ⟨h1, h2⟩
    refine ⟨fun hc => ?_, fun hc => ?_⟩
    · rcases this.mpr (Or.inl hc) with h' | h' <;> contradiction
    · rcases this.mpr (Or.inr hc) with h' | h' <;> contradiction

/-- **The final step lands on a twin.**  If the landing `E + 6 d h` is the column `k` of the line
(`6k - 1 = E + 6 d h`), lies below `P²`, and every gear of a set `G` holding every prime of
`[5, P)` avoids its two forbidden classes, then the landing is a twin prime pair. -/
theorem final_step_twin {G : Finset ℕ} {P k : ℕ} {E h d : ℤ}
    (hfull : ∀ r, r.Prime → 5 ≤ r → r < P → r ∈ G) (hk : 1 ≤ k) (hlt : 6 * k + 1 < P ^ 2)
    (hL : ((6 * k - 1 : ℕ) : ℤ) = E + 6 * d * h)
    (havoid : ∀ g ∈ G, ¬ 6 * d * h ≡ -E [ZMOD g] ∧ ¬ 6 * d * h ≡ -(E + 2) [ZMOD g]) :
    (6 * k - 1).Prime ∧ (6 * k + 1).Prime := by
  apply section_twin_of_unstruck hfull hk hlt
  rintro ⟨g, hg, hdvd | hdvd⟩
  · have ho := (avoid_iff_open).mp (havoid g hg)
    apply ho.1
    rw [← hL]; exact_mod_cast hdvd
  · have ho := (avoid_iff_open).mp (havoid g hg)
    apply ho.2
    have e : ((6 * k + 1 : ℕ) : ℤ) = ((6 * k - 1 : ℕ) : ℤ) + 2 := by
      have : 6 * k + 1 = (6 * k - 1) + 2 := by omega
      rw [this]; push_cast; ring
    rw [← hL, ← e]; exact_mod_cast hdvd

end MirrorWalk
