/-
TwinLadder (round 114, 2026-09-19): the normal form of a twin's own stretch.

For a twin prime pair (P, P+2) with P = 6c - 1, the stretch of P - the columns k with
P² < 6k - 1 and 6k + 1 < (P+2)² - is exactly the 4c - 1 columns k = 6c² + j, |j| ≤ 2c - 1,
centred on the column 6c² whose lower member is P(P+2) = 36c² - 1.  Column 6c² + j has members
P(P+2) + 6j and P(P+2) + 6j + 2.  The two twin gears P and P + 2 strike the stretch ONLY at its
centre, on the member P(P+2): a multiple of P inside the stretch differs from P(P+2) by 6j + e,
|6j + e| ≤ 2P - 2 with e ∈ {0, 2}, and the only multiples of P in that range are 0 and ±P, of which
±P ≡ ±5 (mod 6) is not ≡ 0 or 2.  So a twin hands up a fully specified stretch: position, length,
member parametrisation and its top gears' whole action.  (Tree node R5.f.ii; lane round 2.)
-/
import Mathlib.Tactic

namespace TwinLadder

/-- **The stretch of a twin, in normal form.**  With `P = 6c - 1` and `q = P + 2 = 6c + 1`, column
`k` lies in the stretch `(P², q²)` iff `k = 6c² + j` with `-(2c - 1) ≤ j ≤ 2c - 1` (as integers). -/
theorem stretch_normal_form (c k : ℤ) (hc : 1 ≤ c) :
    ((6 * c - 1) ^ 2 < 6 * k - 1 ∧ 6 * k + 1 < (6 * c + 1) ^ 2) ↔
      (6 * c ^ 2 - (2 * c - 1) ≤ k ∧ k ≤ 6 * c ^ 2 + (2 * c - 1)) := by
  have e1 : (6 * c - 1) ^ 2 = 36 * c ^ 2 - 12 * c + 1 := by ring
  have e2 : (6 * c + 1) ^ 2 = 36 * c ^ 2 + 12 * c + 1 := by ring
  rw [e1, e2]
  constructor <;> intro h <;> constructor <;> omega

/-- **The members of the stretch's columns.**  Column `6c² + j` has lower member `P(P+2) + 6j` and
upper member `P(P+2) + 6j + 2`, where `P(P+2) = 36c² - 1`. -/
theorem members_of_column (c j : ℤ) :
    6 * (6 * c ^ 2 + j) - 1 = (6 * c - 1) * (6 * c + 1) + 6 * j ∧
    6 * (6 * c ^ 2 + j) + 1 = (6 * c - 1) * (6 * c + 1) + 6 * j + 2 := by
  constructor <;> ring

/-- **A twin gear strikes its own stretch only at the centre.**  Let `g` be `P = 6c - 1` or
`P + 2 = 6c + 1` and let `m = P(P+2) + 6j + e` with `e ∈ {0, 2}` be a member of a column of the
stretch (`|j| ≤ 2c - 1`).  If `g ∣ m` then `j = 0` and `e = 0`: the member is `P(P+2)` itself. -/
theorem twin_gear_strikes_centre_only (c j e : ℤ) (hc : 1 ≤ c)
    (hj : -(2 * c - 1) ≤ j ∧ j ≤ 2 * c - 1) (he : e = 0 ∨ e = 2)
    (g : ℤ) (hg : g = 6 * c - 1 ∨ g = 6 * c + 1)
    (hdvd : g ∣ (6 * c - 1) * (6 * c + 1) + 6 * j + e) :
    j = 0 ∧ e = 0 := by
  -- g divides the product, hence the offset 6j + e
  have hprod : g ∣ (6 * c - 1) * (6 * c + 1) := by
    rcases hg with rfl | rfl
    · exact Dvd.intro _ rfl
    · exact Dvd.intro_left _ rfl
  have hoff : g ∣ 6 * j + e := by
    have := (Int.dvd_add_right hprod).mp (by rw [← add_assoc]; exact hdvd)
    exact this
  obtain ⟨t, ht⟩ := hoff
  -- |6j + e| ≤ 2P - 2 < 2g, so t ∈ {-1, 0, 1}
  have hbound : -(12 * c - 6) ≤ 6 * j + e ∧ 6 * j + e ≤ 12 * c - 4 := by omega
  have ht3 : t = -1 ∨ t = 0 ∨ t = 1 := by
    rcases hg with rfl | rfl
    · -- (6c - 1) * t between -(12c - 6) and 12c - 4 forces |t| ≤ 1
      have h1 : (6 * c - 1) * t ≤ 12 * c - 4 := by omega
      have h2 : -(12 * c - 6) ≤ (6 * c - 1) * t := by omega
      have : t ≤ 1 := by nlinarith
      have : -1 ≤ t := by nlinarith
      omega
    · have h1 : (6 * c + 1) * t ≤ 12 * c - 4 := by omega
      have h2 : -(12 * c - 6) ≤ (6 * c + 1) * t := by omega
      have : t ≤ 1 := by nlinarith
      have : -1 ≤ t := by nlinarith
      omega
  -- t = ±1 is impossible mod 6: 6j + e ≡ e ∈ {0, 2}, while ±g ≡ ±1 or ±5
  rcases ht3 with rfl | rfl | rfl
  · exfalso
    rcases hg with rfl | rfl <;> rcases he with rfl | rfl <;> omega
  · constructor <;> rcases he with rfl | rfl <;> omega
  · exfalso
    rcases hg with rfl | rfl <;> rcases he with rfl | rfl <;> omega

/-- **The two composite-forcing offset families** (random lane round 3).  With `s = 6c`, the lower
member at `j = -(u² - 1)/6` is `s² - u²` (`u ≡ ±1 mod 6`, the known difference of squares), and the
upper member at `j = ±2c - 6t²` is `(s ± 1)² - (6t)²` - a second family, previously unrecorded.  Both
force a composite member, so both offsets are excluded from any rung. -/
theorem upper_member_difference_of_squares (c t : ℤ) :
    (6 * c) ^ 2 + 6 * (2 * c - 6 * t ^ 2) + 1 = (6 * c + 1 - 6 * t) * (6 * c + 1 + 6 * t) ∧
    (6 * c) ^ 2 + 6 * (-(2 * c) - 6 * t ^ 2) + 1 = (6 * c - 1 - 6 * t) * (6 * c - 1 + 6 * t) := by
  constructor <;> ring

theorem lower_member_difference_of_squares (c u : ℤ) (hu : (u ^ 2 - 1) % 6 = 0) :
    (6 * c) ^ 2 + 6 * (-((u ^ 2 - 1) / 6)) - 1 = (6 * c - u) * (6 * c + u) := by
  have h : 6 * ((u ^ 2 - 1) / 6) = u ^ 2 - 1 := Int.mul_ediv_cancel' (Int.dvd_of_emod_eq_zero hu)
  linear_combination -h

end TwinLadder
