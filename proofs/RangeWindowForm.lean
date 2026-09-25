import Mathlib.Data.Nat.Prime.Basic
import Mathlib.NumberTheory.Primorial
import Mathlib.Tactic
import RangeRegion
import RangeHandoff

/-!
# The range-window form of the blame test

`RangeRegion.total_blame_iff_primorial` states the blame test over all copies `j` with
`1 ≤ j < primorial q / 30`. That window contains copy `1` (legs `29`, `31`, both prime) as soon as
`primorial q ≥ 60`, so its left side is false there and the statement carries no information about
the range. The range statement at `q` (`RangeHandoff.RangeStatement`) asks for a twin pair strictly
above `q` with upper member at most `primorial q`; the matching window of copies is

  `q < 30 j - 1` and `30 j + 1 ≤ primorial q`.

This file states the blame test on that window, with `P j = Nat.minFac (900 j² - 1)`
(`RangeRegion.P`):

* `range_copy_iff` (a): some copy `j ≥ 1` in the window has both legs prime exactly when some copy
  `j ≥ 1` in the window has `30 j + 1 < (P j)²` (region law `revealed_iff_minFac`, copy by copy).
* `copy_range_implies_rangeStatement` (b): a copy `j ≥ 1` in the window with both legs prime is a
  witness for `RangeStatement q`, with `p = 30 j - 1` and `p + 2 = 30 j + 1`.
* `no_copy_blame` (c): no copy `j ≥ 1` in the window has both legs prime exactly when every copy
  `j ≥ 1` in the window has `(P j)² ≤ 30 j + 1` (failure form `blamed_iff_minFac`, copy by copy).
-/

namespace RangeLine

/-- (a) Range-window region law: for any `q`, some copy `j ≥ 1` with `q < 30 j - 1` and
`30 j + 1 ≤ primorial q` has both legs `30 j - 1`, `30 j + 1` prime exactly when some copy `j ≥ 1`
in the same window has `30 j + 1 < (P j)²`. -/
theorem range_copy_iff (q : ℕ) :
    (∃ j, 1 ≤ j ∧ q < 30 * j - 1 ∧ 30 * j + 1 ≤ primorial q ∧
        (30 * j - 1).Prime ∧ (30 * j + 1).Prime) ↔
      (∃ j, 1 ≤ j ∧ q < 30 * j - 1 ∧ 30 * j + 1 ≤ primorial q ∧ 30 * j + 1 < (P j) ^ 2) := by
  constructor
  · rintro ⟨j, hj, hq, hle, hp1, hp2⟩
    exact ⟨j, hj, hq, hle, (revealed_iff_minFac hj).mp ⟨hp1, hp2⟩⟩
  · rintro ⟨j, hj, hq, hle, hlt⟩
    obtain ⟨hp1, hp2⟩ := (revealed_iff_minFac hj).mpr hlt
    exact ⟨j, hj, hq, hle, hp1, hp2⟩

/-- (b) A copy `j ≥ 1` with `q < 30 j - 1`, `30 j + 1 ≤ primorial q` and both legs prime
witnesses the range statement at `q`, with `p = 30 j - 1` (so `p + 2 = 30 j + 1`). -/
theorem copy_range_implies_rangeStatement {q : ℕ}
    (h : ∃ j, 1 ≤ j ∧ q < 30 * j - 1 ∧ 30 * j + 1 ≤ primorial q ∧
        (30 * j - 1).Prime ∧ (30 * j + 1).Prime) :
    RangeStatement q := by
  obtain ⟨j, hj, hq, hle, hp1, hp2⟩ := h
  have hsum : 30 * j - 1 + 2 = 30 * j + 1 := by omega
  refine ⟨30 * j - 1, hq, ?_, hp1, ?_⟩
  · rw [hsum]
    exact hle
  · rw [hsum]
    exact hp2

/-- (c) Range-window blame test: no copy `j ≥ 1` with `q < 30 j - 1` and `30 j + 1 ≤ primorial q`
has both legs prime exactly when every copy `j ≥ 1` in that window has `(P j)² ≤ 30 j + 1`. -/
theorem no_copy_blame (q : ℕ) :
    (¬ ∃ j, 1 ≤ j ∧ q < 30 * j - 1 ∧ 30 * j + 1 ≤ primorial q ∧
        (30 * j - 1).Prime ∧ (30 * j + 1).Prime) ↔
      ∀ j, 1 ≤ j → q < 30 * j - 1 → 30 * j + 1 ≤ primorial q → (P j) ^ 2 ≤ 30 * j + 1 := by
  constructor
  · intro h j hj hq hle
    exact (blamed_iff_minFac hj).mp (fun hp => h ⟨j, hj, hq, hle, hp⟩)
  · rintro h ⟨j, hj, hq, hle, hp⟩
    exact (blamed_iff_minFac hj).mpr (h j hj hq hle) hp

end RangeLine
