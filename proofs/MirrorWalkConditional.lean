/-
MirrorWalkConditional (round 58, 2026-09-16): the walk's theorem, with its one hypothesis named.

The settle walk is proved except for one statement.  This file states the implication in full:
IF at the walk's last step some candidate of its progression is open to every gear of the
machine, THEN the walk lands on a twin prime pair inside the window.

The walk's last step goes from the column `c₀` by the stride `s` (`s = 2 g` columns for the
mirror `{2, 3, g}`, either sign), so its candidates are the columns `c₀ + s k`, `k = 1 … K`.
`StepOpen` says one of them is a column of the window that no gear of `G` strikes.

The other pieces are already proved: the walk stays in the window (`MirrorWalkInWindow`), the
early steps always have a keeping move (`keeping_move_free`), and a window column struck by no
gear up to `q` is a twin prime pair (`section_twin_of_unstruck`).  `StepOpen` is what the
measurement meets with a margin of four to fourteen candidates of eighty at every machine
tested, and it is the window statement on one arithmetic progression.
-/
import MirrorWalkSettleFree
import MirrorWalkInWindow

namespace MirrorWalk

open SquareColumn

/-- The candidate columns of a step: `c₀ + s k`. -/
def cand (c₀ s : ℤ) (k : ℕ) : ℤ := c₀ + s * k

/-- **The step hypothesis.**  Some candidate within `K` periods is a column of the window
(`1 ≤ m`, `6m + 1 < P²`) that no gear of `G` strikes. -/
def StepOpen (G : Finset ℕ) (c₀ s : ℤ) (K P : ℕ) : Prop :=
  ∃ (k : ℕ) (m : ℕ), 1 ≤ k ∧ k ≤ K ∧ (m : ℤ) = cand c₀ s k ∧ 1 ≤ m ∧ 6 * m + 1 < P ^ 2 ∧
    ∀ g ∈ G, ¬ (g ∣ 6 * m - 1) ∧ ¬ (g ∣ 6 * m + 1)

/-- **The walk's theorem.**  With `G` holding every prime from 5 below `P`, the step hypothesis
gives a twin prime pair inside the window `(P, P²]`. -/
theorem walk_twin_of_stepOpen {G : Finset ℕ} {c₀ s : ℤ} {K P : ℕ}
    (hfull : ∀ r, r.Prime → 5 ≤ r → r < P → r ∈ G)
    (hstep : StepOpen G c₀ s K P) :
    ∃ m : ℕ, 1 ≤ m ∧ 6 * m + 1 < P ^ 2 ∧ (6 * m - 1).Prime ∧ (6 * m + 1).Prime := by
  obtain ⟨k, m, hk1, hkK, hcast, hm1, hmlt, hopen⟩ := hstep
  have hns : ¬ StruckBy G m := by
    rintro ⟨g, hg, hd | hd⟩
    · exact (hopen g hg).1 hd
    · exact (hopen g hg).2 hd
  obtain ⟨hp1, hp2⟩ := section_twin_of_unstruck hfull hm1 hmlt hns
  exact ⟨m, hm1, hmlt, hp1, hp2⟩

end MirrorWalk
