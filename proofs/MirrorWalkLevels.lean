/-
MirrorWalkLevels (round 53, 2026-09-14): the spiral with a general base, and the levels.

A spiral with base product `P` over the gears `gs` flips once per gear about the mirror
`{base, g}` (size `P g`), alternating up and down: each flip moves the column by `2 P g` in its
direction.  From column `n` the spiral ends at `n + 2 P d altSum gs`.  Stacking levels: level `k`
runs its spiral from the landing of level `k+1`, so the landing is congruent to the previous
landing modulo `P_k`; with `6 ∣ P_k` every landing stays a left member (`≡ 5 (mod 6)`), which is
what the final flip needs.  The final flip about `{3, h}` from a landing at most `q` with a gear
`h` at most `q` stays below `q²` once `q ≥ 8`.
-/
import MirrorWalkSpiral

namespace MirrorWalk

/-- The spiral with base product `P`: one flip per gear, alternating, each moving `2 P g`. -/
def spiralP (P : ℤ) : List ℤ → ℤ → ℤ → ℤ
  | [], _, n => n
  | g :: rest, d, n => spiralP P rest (-d) (n + 2 * P * d * g)

theorem spiralP_eq_dir (P : ℤ) : ∀ (gs : List ℤ) (d n : ℤ),
    spiralP P gs d n = n + 2 * P * d * altSum gs := by
  intro gs
  induction gs with
  | nil => intro d n; simp [spiralP, altSum]
  | cons g rest ih =>
    intro d n
    simp only [spiralP, altSum]
    rw [ih]; ring

/-- **Level landing.**  Level `k` from the previous landing `n` ends at `n + 2 P_k A_k`. -/
theorem spiralP_eq (P : ℤ) (gs : List ℤ) (n : ℤ) : spiralP P gs 1 n = n + 2 * P * altSum gs := by
  rw [spiralP_eq_dir]; ring

/-- The landing is congruent to the previous landing modulo the level's base product. -/
theorem spiralP_modEq (P : ℤ) (gs : List ℤ) (n : ℤ) : spiralP P gs 1 n ≡ n [ZMOD P] := by
  rw [spiralP_eq, Int.modEq_iff_dvd]
  exact ⟨-(2 * altSum gs), by ring⟩

/-- **Alignment.**  With `6 ∣ P` a left member (`n ≡ 5 (mod 6)`) lands on a left member. -/
theorem spiralP_left_member {P : ℤ} (gs : List ℤ) {n : ℤ} (h6 : (6 : ℤ) ∣ P) (hn : n ≡ 5 [ZMOD 6]) :
    spiralP P gs 1 n ≡ 5 [ZMOD 6] := by
  have := (spiralP_modEq P gs n).of_dvd h6
  exact this.trans hn

/-- The primorial spiral from home is the base-`P` spiral from `-1`: `E = -1 + 2 P A`. -/
theorem spiralP_home (P : ℤ) (gs : List ℤ) : spiralP P gs 1 (-1) = -1 + 2 * P * altSum gs := by
  rw [spiralP_eq]

/-- **Ceiling of a level.**  With the gears positive and strictly descending, `A ≤ head`, so the
landing is at most `n + 2 P g₁`. -/
theorem spiralP_le (P : ℤ) (hP : 0 ≤ P) (gs : List ℤ) (hs : List.Pairwise (· > ·) gs)
    (hpos : ∀ x ∈ gs, 0 < x) (n : ℤ) : spiralP P gs 1 n ≤ n + 2 * P * gs.headD 0 := by
  rw [spiralP_eq]
  have := (altSum_le_head gs hs hpos).1
  nlinarith

/-- **The final flip stays below the window's top.**  A landing `E ≤ q` and a gear `h ≤ q` give
`E + 6h + 2 ≤ q²` once `q ≥ 8`. -/
theorem final_flip_below_square {E h q : ℤ} (hE : E ≤ q) (hh : h ≤ q) (hq : 8 ≤ q) :
    E + 6 * h + 2 ≤ q ^ 2 := by
  nlinarith

/-- **The final flip enters the window** iff `6h > q - E`. -/
theorem final_flip_above_q {E h q : ℤ} : q < E + 6 * h ↔ q - E < 6 * h := by
  constructor <;> intro hlt <;> linarith

end MirrorWalk
