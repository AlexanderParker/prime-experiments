import RangeCentre

/-!
# Acting and the pair law on copies

Copies are indexed by naturals `j ≥ 1`; copy `j` carries the pair `(30j - 1, 30j + 1)`, and a
gear `g` strikes copy `j` when `g ∣ (30j - 1)(30j + 1)` (`StrikesCopy`, from `RangeCentre`).

A gear `g` *acts* at copy `j` when `g² ≤ 30j + 1`, i.e. `g` is at most the square root of the
top member of the pair.

This file proves:

* `distance_law`: for a prime `g ≥ 7` and copies `1 ≤ x < y`, if `g` strikes both copies then
  `g ∣ (y - x) · (225(y - x)² - 1)`. By the centre law either `g ∣ y - x` (same sign) or
  `g ∣ 225(y - x)² - 1` (opposite signs).
* `acting_pair_law`: for a prime `g ≥ 7`, `j ≥ 1` and a distance `0 < D < g`,
  `g` strikes copies `j` and `j + D` and acts at `j` iff
  `g ∣ 225D² - 1`, `g ∣ 2j + D`, `g ∣ 900j² - 1` and `g` acts at `j`.
  Since `0 < D < g` rules out `g ∣ D`, the centre law leaves only the sum branch
  (`g ∣ j + (j + D) = 2j + D`); the condition `g ∣ 900j² - 1` is the strike on `j` itself.
* `acts_mono`: acting is monotone in the copy index.
-/

namespace RangeLine

/-- The gear `g` *acts* at copy `j`: `g² ≤ 30j + 1`. -/
def Acts (g j : ℕ) : Prop := g ^ 2 ≤ 30 * j + 1

/-- **Distance law.** For a prime `g ≥ 7` and copies `1 ≤ x < y`, if `g` strikes both copies
then `g ∣ (y - x) · (225(y - x)² - 1)`. -/
theorem distance_law (g x y : ℕ) (hg : g.Prime) (hg7 : 7 ≤ g) (hx : 1 ≤ x) (hxy : x < y)
    (hsx : StrikesCopy g x) (hsy : StrikesCopy g y) :
    g ∣ (y - x) * (225 * (y - x) ^ 2 - 1) := by
  rcases (centre_law g x y hg hg7 hx hxy).1 ⟨hsx, hsy⟩ with ⟨hd, _⟩ | ⟨_, hr⟩
  · exact Dvd.dvd.mul_right hd _
  · exact Dvd.dvd.mul_left hr _

/-- **Acting pair law.** For a prime `g ≥ 7`, a copy `j ≥ 1` and a distance `0 < D < g`,
`g` strikes copies `j` and `j + D` and acts at `j` iff `g ∣ 225D² - 1`, `g ∣ 2j + D`,
`g ∣ 900j² - 1` and `g` acts at `j`. -/
theorem acting_pair_law (g j D : ℕ) (hg : g.Prime) (hg7 : 7 ≤ g) (hj : 1 ≤ j)
    (hD0 : 0 < D) (hDg : D < g) :
    (StrikesCopy g j ∧ StrikesCopy g (j + D) ∧ Acts g j) ↔
      (g ∣ 225 * D ^ 2 - 1 ∧ g ∣ 2 * j + D ∧ g ∣ 900 * j ^ 2 - 1 ∧ Acts g j) := by
  have hc := centre_law g j (j + D) hg hg7 hj (by omega)
  have e1 : j + D - j = D := by omega
  have e2 : j + (j + D) = 2 * j + D := by omega
  rw [e1, e2] at hc
  have hnD : ¬ g ∣ D := fun h => absurd (Nat.le_of_dvd hD0 h) (by omega)
  constructor
  · rintro ⟨hsj, hsjD, ha⟩
    rcases hc.1 ⟨hsj, hsjD⟩ with ⟨hd, _⟩ | ⟨hs, hr⟩
    · exact absurd hd hnD
    · exact ⟨hr, hs, (strikesCopy_iff_sq g j).1 hsj, ha⟩
  · rintro ⟨hr, hs, _, ha⟩
    obtain ⟨hsj, hsjD⟩ := hc.2 (Or.inr ⟨hs, hr⟩)
    exact ⟨hsj, hsjD, ha⟩

/-- **Acting is monotone.** If `g` acts at copy `j` and `j ≤ j'`, then `g` acts at copy `j'`. -/
theorem acts_mono {g j j' : ℕ} (h : Acts g j) (hjj : j ≤ j') : Acts g j' := by
  unfold Acts at *
  omega

end RangeLine
