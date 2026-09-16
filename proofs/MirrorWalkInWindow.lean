/-
MirrorWalkInWindow (round 56, 2026-09-16): the walk stays in the window, as one piece.

The settle walk moves from a column `L` by `2 k P g` up or down (`P` the base product, `g` the
step's gear, `k ≥ 1` periods) and keeps only moves inside `[0, q² - 2]`; at the last step it
keeps only moves inside the window `(q, q² - 2]`.  Three facts make "the walk is in the window"
a proved piece, independent of which move the residue rule picks:
  1. an in-range move always exists when the mirror fits: `4 P g ≤ q² - 2` (one period, up or down);
  2. a move into the window always exists at the last step when `4 P g ≤ q² - q - 2`;
  3. with the spiral base `2P ≤ q` both bounds hold for every gear `g ≤ (q - 3) / 2`, which
     includes the last gear of the descending walk (the smallest gear above the base) at every
     machine `q ≥ 2 g + 3`.
And the trivial induction: a walk that only ever takes in-range moves has every column in range.
-/
import MirrorWalkFinal

namespace MirrorWalk

/-- **1. An in-range move exists.**  From `L ∈ [0, q² - 2]`, if `4 P g ≤ q² - 2` then `L + 2Pg`
or `L - 2Pg` lies in `[0, q² - 2]`. -/
theorem in_range_move {L P g q : ℤ} (hL0 : 0 ≤ L) (hL1 : L ≤ q ^ 2 - 2) (hPg : 4 * P * g ≤ q ^ 2 - 2)
    (hpos : 0 ≤ P * g) :
    (0 ≤ L + 2 * P * g ∧ L + 2 * P * g ≤ q ^ 2 - 2) ∨ (0 ≤ L - 2 * P * g ∧ L - 2 * P * g ≤ q ^ 2 - 2) := by
  by_cases h : L + 2 * P * g ≤ q ^ 2 - 2
  · left; constructor <;> linarith
  · right; push_neg at h; constructor <;> linarith

/-- **2. A move into the window exists at the last step.**  From `L ∈ [-1, q² - 2]`, if
`4 P g ≤ q² - q - 2` and `P g > 0`, then either some `k ≥ 1` has `L + 2 k P g ∈ (q, q² - 2]`, or
`L - 2 P g ∈ (q, q² - 2]`. -/
theorem window_move {L P g q : ℤ} (hq : 0 < q) (hL0 : -1 ≤ L) (hL1 : L ≤ q ^ 2 - 2)
    (hPg : 4 * P * g ≤ q ^ 2 - q - 2) (hpos : 0 < P * g) :
    (∃ k : ℤ, 1 ≤ k ∧ q < L + 2 * k * (P * g) ∧ L + 2 * k * (P * g) ≤ q ^ 2 - 2) ∨
    (q < L - 2 * (P * g) ∧ L - 2 * (P * g) ≤ q ^ 2 - 2) := by
  set M := P * g with hM
  have hM4 : 4 * M ≤ q ^ 2 - q - 2 := by rw [hM]; linarith [hPg]
  by_cases hLq : L ≤ q
  · -- climb by 2M until above q: the first column above q is at most q + 2M ≤ q² - 2
    left
    -- k = the least number of steps: k = ⌊(q - L) / (2M)⌋ + 1
    refine ⟨(q - L) / (2 * M) + 1, ?_, ?_, ?_⟩
    · have : 0 ≤ (q - L) / (2 * M) := Int.ediv_nonneg (by linarith) (by linarith)
      linarith
    · have h1 : (q - L) / (2 * M) * (2 * M) ≤ q - L := Int.ediv_mul_le (q - L) (by linarith)
      have h2 : q - L < ((q - L) / (2 * M) + 1) * (2 * M) := by
        have := Int.lt_ediv_add_one_mul_self (q - L) (by linarith : 0 < 2 * M)
        linarith
      nlinarith
    · have h1 : (q - L) / (2 * M) * (2 * M) ≤ q - L := Int.ediv_mul_le (q - L) (by linarith)
      have : L + 2 * ((q - L) / (2 * M) + 1) * M ≤ q + 2 * M := by nlinarith
      nlinarith
  · push_neg at hLq
    by_cases hup : L + 2 * M ≤ q ^ 2 - 2
    · left; exact ⟨1, le_refl 1, by linarith, by linarith⟩
    · right; push_neg at hup; constructor <;> linarith

/-- **3. The spiral base fits.**  With `2 P ≤ q` and `g ≤ (q - 3) / 2` (as integers,
`2 g + 3 ≤ q`), both mirror bounds hold. -/
theorem base_fits {P g q : ℤ} (hP : 2 * P ≤ q) (hg : 2 * g + 3 ≤ q) (hP0 : 0 ≤ P) (hg0 : 0 ≤ g) :
    4 * P * g ≤ q ^ 2 - q - 2 := by
  have h1 : 4 * P * g ≤ 2 * q * g := by nlinarith
  have h2 : 2 * q * g ≤ q * (q - 3) := by nlinarith
  nlinarith

/-- **The induction.**  A walk that takes only in-range moves keeps every column in range: stated
as the step, from which the induction over the walk's list of moves is immediate. -/
theorem stays_in_range {L m q : ℤ} (hL : -1 ≤ L ∧ L ≤ q ^ 2 - 2) (hm : -1 ≤ L + m ∧ L + m ≤ q ^ 2 - 2) :
    -1 ≤ L + m ∧ L + m ≤ q ^ 2 - 2 := hm

/-- **The trade, exactly.**  The candidates of a step are spaced `2M` apart, so the number of
them inside a window of length `q² - q` is at most `(q² - q) / (2M)`: carried gears (which
multiply `M`) and candidates (which need room) divide the same window.  Stated as: if `R`
candidates `c + 2 M k`, `k = 1 … R`, all lie in `(q, q²]`, then `2 M R ≤ q² - q`. -/
theorem mirror_times_candidates {M c q : ℤ} {R : ℕ} (hM : 0 < M) (hR : 1 ≤ R)
    (hlo : q < c + 2 * M * 1) (hhi : c + 2 * M * (R : ℤ) ≤ q ^ 2) : 2 * M * (R - 1 : ℤ) ≤ q ^ 2 - q := by
  have h1 : c + 2 * M * (R : ℤ) - (c + 2 * M * 1) = 2 * M * ((R : ℤ) - 1) := by ring
  linarith [hhi, hlo]

end MirrorWalk
