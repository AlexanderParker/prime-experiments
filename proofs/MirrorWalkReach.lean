/-
MirrorWalkReach (round 55, 2026-09-15): what a mirror walk can carry, exactly.

A walk from home (-1, 1) on mirrors `M₁, …, Mₙ` (any periods, directions, repeats, order)
lands at `-1 + 2 D` where `D` is an integer combination of the mirrors.  Facts:
  1. `D` is a multiple of `G = gcd(M₁, …, Mₙ)`, and every multiple of `G` is such a `D`
     (Bezout): the reachable columns are exactly `-1 + 2 G ℤ`.
  2. The landing is open by construction to every gear dividing `G` (residue `-1` carried).
  3. For a prime gear `g` not dividing `2G`, some reachable column is struck by `g`: nothing
     about `g` is guaranteed by the walk.
  4. A landing inside the window `(q, q²]` forces `2G ≤ q² + 1`.
Together: a mirror walk guarantees openness to exactly the gears of its gcd, and inside the
window that gcd is at most `(q² + 1)/2`.  Chains of windows change nothing: every landing of
every chain lies in the same set `-1 + 2 G ℤ` for the gcd of all mirrors used.
-/
import MirrorWalkFinal

namespace MirrorWalk

/-- Integer combinations of a list of mirrors. -/
def combo : List ℤ → List ℤ → ℤ
  | [], _ => 0
  | _, [] => 0
  | M :: Ms, c :: cs => c * M + combo Ms cs

/-- The gcd of a list of mirrors. -/
def gcdL : List ℤ → ℤ
  | [] => 0
  | M :: Ms => ((Int.gcd M (gcdL Ms) : ℕ) : ℤ)

theorem gcdL_dvd_head (M : ℤ) (Ms : List ℤ) : gcdL (M :: Ms) ∣ M := by
  show ((Int.gcd M (gcdL Ms) : ℕ) : ℤ) ∣ M
  exact Int.gcd_dvd_left M (gcdL Ms)

theorem gcdL_dvd_tail (M : ℤ) (Ms : List ℤ) : gcdL (M :: Ms) ∣ gcdL Ms := by
  show ((Int.gcd M (gcdL Ms) : ℕ) : ℤ) ∣ gcdL Ms
  exact Int.gcd_dvd_right M (gcdL Ms)

/-- **1a. Every combination is a multiple of the gcd.** -/
theorem gcdL_dvd_combo : ∀ (Ms cs : List ℤ), gcdL Ms ∣ combo Ms cs := by
  intro Ms
  induction Ms with
  | nil => intro cs; simp [combo]
  | cons M rest ih =>
    intro cs
    cases cs with
    | nil => simp [combo]
    | cons c cs' =>
      simp only [combo]
      apply dvd_add
      · exact Dvd.dvd.mul_left (gcdL_dvd_head M rest) c
      · exact dvd_trans (gcdL_dvd_tail M rest) (ih cs')

/-- **1b. The gcd itself is a combination (Bezout for a list).** -/
theorem gcdL_is_combo : ∀ (Ms : List ℤ), ∃ cs : List ℤ, combo Ms cs = gcdL Ms := by
  intro Ms
  induction Ms with
  | nil => exact ⟨[], by simp [combo, gcdL]⟩
  | cons M rest ih =>
    obtain ⟨cs, hcs⟩ := ih
    -- Int.gcd M g = M * gcdA + g * gcdB
    have hb := Int.gcd_eq_gcd_ab M (gcdL rest)
    refine ⟨Int.gcdA M (gcdL rest) :: cs.map (fun c => c * Int.gcdB M (gcdL rest)), ?_⟩
    simp only [combo, gcdL]
    have hscale : ∀ (Ns cs : List ℤ) (b : ℤ), combo Ns (cs.map (fun c => c * b)) = combo Ns cs * b := by
      intro Ns
      induction Ns with
      | nil => intro cs b; simp [combo]
      | cons N Ns' ihN =>
        intro cs b
        cases cs with
        | nil => simp [combo]
        | cons c cs' => simp only [combo, List.map_cons]; rw [ihN]; ring
    rw [hscale, hcs, hb]; ring

/-- **1c. Every multiple of the gcd is reachable.** -/
theorem multiple_reachable (Ms : List ℤ) (t : ℤ) : ∃ cs : List ℤ, combo Ms cs = t * gcdL Ms := by
  obtain ⟨cs, hcs⟩ := gcdL_is_combo Ms
  have hscale : ∀ (Ns cs : List ℤ) (b : ℤ), combo Ns (cs.map (fun c => c * b)) = combo Ns cs * b := by
    intro Ns
    induction Ns with
    | nil => intro cs b; simp [combo]
    | cons N Ns' ihN =>
      intro cs b
      cases cs with
      | nil => simp [combo]
      | cons c cs' => simp only [combo, List.map_cons]; rw [ihN]; ring
  refine ⟨cs.map (fun c => c * t), ?_⟩
  rw [hscale, hcs]; ring

/-- **2. Carried residues.**  The landing `-1 + 2 D` is open to every gear `g ≥ 2` dividing `D`. -/
theorem landing_open_of_dvd_combo {g : ℕ} {D : ℤ} (hg : 2 ≤ g) (hd : (g : ℤ) ∣ D) :
    OpenTo g (-1 + 2 * D) := by
  have h2 : (g : ℤ) ∣ 2 * D := Dvd.dvd.mul_left hd 2
  have hg1 : ¬ (g : ℤ) ∣ 1 := by
    intro h
    have h1 := Int.le_of_dvd one_pos h
    have h2' : (2 : ℤ) ≤ g := by exact_mod_cast hg
    omega
  constructor
  · intro h
    have : (g : ℤ) ∣ 2 * D - (-1 + 2 * D) := Int.dvd_sub h2 h
    simp at this; exact hg1 this
  · intro h
    have e : -1 + 2 * D + 2 = 2 * D + 1 := by ring
    rw [e] at h
    have : (g : ℤ) ∣ (2 * D + 1) - 2 * D := Int.dvd_sub h h2
    simp at this; exact hg1 this

/-- **3. Nothing is guaranteed for a gear outside the gcd.**  If `g` is coprime to `2G`, some
multiple `t G` of the gcd has `g ∣ -1 + 2 t G`: a reachable column struck by `g`. -/
theorem struck_reachable {g : ℕ} {G : ℤ} (hco : IsCoprime (2 * G) (g : ℤ)) :
    ∃ t : ℤ, (g : ℤ) ∣ -1 + 2 * t * G := by
  obtain ⟨u, v, huv⟩ := hco
  refine ⟨u, -v, ?_⟩
  linear_combination huv

/-- **4. A landing in the window bounds the gcd.**  If `-1 + 2 t G` lies in `(q, q²]` with
`G > 0`, then `2 G ≤ q² + 1`. -/
theorem gcd_le_of_in_window {G t q : ℤ} (hG : 0 < G) (hq : 0 < q)
    (hlo : q < -1 + 2 * t * G) (hhi : -1 + 2 * t * G ≤ q ^ 2) : 2 * G ≤ q ^ 2 + 1 := by
  have ht : 1 ≤ t := by
    by_contra h
    push_neg at h
    have : t ≤ 0 := by omega
    have : 2 * t * G ≤ 0 := by nlinarith
    linarith
  nlinarith

end MirrorWalk
