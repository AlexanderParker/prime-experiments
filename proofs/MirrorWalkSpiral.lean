/-
MirrorWalkSpiral (round 49, 2026-09-13): the owner's spiral, formalised.

The spiral of machine q: from home (-1, 1), one flip per odd gear g = q, p', p'', ..., 3 (the odd
primes up to q, descending), each about the axis one period of {2, g} from the current column
(axis = current centre + d·2g), the direction d alternating up, down, up, ...  Each such flip
moves the column by 4 d g, so the spiral ends at

    spiralEnd = -1 + 4 · (q - p' + p'' - ... ± 3),

four times the alternating sum of the odd gears.  Proved here: the closed form of the endpoint
for any list of gears; the endpoint is open to every gear dividing the alternating sum (the
walk carries it from home); and the endpoint lies below 4q, hence below q² from q = 5 on, so
the spiral never overshoots the window.  Whether it lands above q (measured: between 1.4q and
2.7q at every machine to 20000) is the alternating sum exceeding (q+1)/4, which is about the
gaps between the gears and is not proved here.
-/
import MirrorWalkRepair

namespace MirrorWalk

/-- Alternating sum of a list, first element positive. -/
def altSum : List ℤ → ℤ
  | [] => 0
  | g :: rest => g - altSum rest

/-- The spiral over a list of gears from the column `n`: flip about `n + 1 + d·2g` with `d`
alternating, starting with the given direction. -/
def spiral : List ℤ → ℤ → ℤ → ℤ
  | [], _, n => n
  | g :: rest, d, n => spiral rest (-d) (flip (n + 1 + d * (2 * g)) n)

/-- One spiral step moves the column by `4 d g`. -/
theorem spiral_step (g d n : ℤ) : flip (n + 1 + d * (2 * g)) n = n + 4 * d * g := by
  unfold flip; ring

theorem spiral_eq_dir : ∀ (gs : List ℤ) (d n : ℤ), spiral gs d n = n + 4 * d * altSum gs := by
  intro gs
  induction gs with
  | nil => intro d n; simp [spiral, altSum]
  | cons g rest ih =>
    intro d n
    simp only [spiral, altSum]
    rw [spiral_step, ih]
    ring

/-- **Closed form of the spiral.**  Starting upward from `n`, the spiral over the gears `gs`
ends at `n + 4 · altSum gs`. -/
theorem spiral_eq (gs : List ℤ) (n : ℤ) : spiral gs 1 n = n + 4 * altSum gs := by
  rw [spiral_eq_dir]; ring

/-- The spiral's endpoint from home. -/
def spiralEnd (gs : List ℤ) : ℤ := -1 + 4 * altSum gs

theorem spiral_home (gs : List ℤ) : spiral gs 1 (-1) = spiralEnd gs := by
  unfold spiralEnd; rw [spiral_eq]

/-- **What the spiral carries.**  The endpoint is open to every gear dividing the alternating
sum of the gears (the endpoint is `-1 + 4A`, so such a gear sees it as home). -/
theorem spiralEnd_open_of_dvd {h : ℕ} (hg : 2 ≤ h) (gs : List ℤ) (hd : (h : ℤ) ∣ altSum gs) :
    OpenTo h (spiralEnd gs) := by
  show OpenTo h ((-1) + 4 * altSum gs)
  exact (openTo_add_of_dvd (dvd_mul_of_dvd_right hd 4)).mpr (home_open hg)

/-- The alternating sum of a strictly decreasing list of positive integers lies between `0` and
its first element. -/
theorem altSum_le_head : ∀ (gs : List ℤ), List.Pairwise (· > ·) gs → (∀ x ∈ gs, 0 < x) →
    altSum gs ≤ gs.headD 0 ∧ 0 ≤ altSum gs := by
  intro gs
  induction gs with
  | nil => intro _ _; simp [altSum]
  | cons g rest ih =>
    intro hs hp
    obtain ⟨hgt, hs'⟩ := List.pairwise_cons.mp hs
    have hp' : ∀ x ∈ rest, 0 < x := fun x hx => hp x (by simp [hx])
    obtain ⟨h1, h2⟩ := ih hs' hp'
    have hg0 : 0 < g := hp g (by simp)
    simp only [altSum, List.headD_cons]
    constructor
    · linarith
    · cases rest with
      | nil => simp [altSum]; exact le_of_lt hg0
      | cons g' rest' =>
        have hlt : g' < g := hgt g' (by simp)
        simp only [List.headD_cons] at h1
        linarith

/-- **The spiral stays below `4q`**, hence below `q²` once `q ≥ 5`: it never overshoots the
window. -/
theorem spiralEnd_lt {gs : List ℤ} {q : ℤ} (hs : List.Pairwise (· > ·) gs) (hp : ∀ x ∈ gs, 0 < x)
    (hhead : gs.headD 0 ≤ q) (hq : 5 ≤ q) : spiralEnd gs < q ^ 2 := by
  obtain ⟨h1, _⟩ := altSum_le_head gs hs hp
  unfold spiralEnd
  nlinarith

end MirrorWalk
