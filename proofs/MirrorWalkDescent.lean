/-
MirrorWalkDescent (round 54, 2026-09-14): the primorial descent.

From home flip up on the full primorial `Q` (one period), then down on partial primorials
`P₁, P₂, …` with `k₁, k₂, …` periods.  Every flip is a real primorial axis.  The landing is
`-1 + 2 (Q - Σ kᵢ Pᵢ)`; when the descent stops at `P_s` with every partial primorial a multiple of
`P_s`, the landing is `2 t P_s - 1` with `t = (Q - Σ kᵢ Pᵢ) / P_s`.

Facts, each its own lemma:
  * the landing formula for a list of (partial primorial, periods) pairs;
  * the landing is open to every gear dividing `P_s` (residue `-1` carried);
  * a gear `g` outside the base strikes the left member iff `t ≡ (2 P_s)⁻¹ (mod g)` and the right
    member iff `t ≡ -(2 P_s)⁻¹ (mod g)`, stated as `2 P_s t ≡ 1` and `2 P_s t ≡ -1`: two classes
    of `t` fixed by `g` and `P_s` alone;
  * the landing lies in the window iff `q + 1 ≤ 2 t P_s ≤ q² - 1`.
-/
import MirrorWalkFinal

namespace MirrorWalk

/-- Total displacement of the descent: up `2Q`, then down `2 kᵢ Pᵢ` for each pair. -/
def descentEnd (Q : ℤ) : List (ℤ × ℤ) → ℤ
  | [] => -1 + 2 * Q
  | (P, k) :: rest => descentEnd Q rest - 2 * k * P

theorem descentEnd_eq (Q : ℤ) : ∀ (l : List (ℤ × ℤ)),
    descentEnd Q l = -1 + 2 * (Q - (l.map (fun x => x.2 * x.1)).sum) := by
  intro l
  induction l with
  | nil => simp [descentEnd]
  | cons x rest ih =>
    simp only [descentEnd, List.map_cons, List.sum_cons]
    rw [ih]; ring

/-- **Landing of the descent.**  If every partial primorial is a multiple of `P_s` and so is `Q`,
the landing is `2 t P_s - 1` with `t = (Q - Σ kᵢ Pᵢ) / P_s`. -/
theorem descentEnd_form {Q Ps : ℤ} (l : List (ℤ × ℤ)) (hQ : Ps ∣ Q)
    (hl : ∀ x ∈ l, Ps ∣ x.1) : ∃ t : ℤ, descentEnd Q l = 2 * t * Ps - 1 := by
  rw [descentEnd_eq]
  have hsum : Ps ∣ (l.map (fun x => x.2 * x.1)).sum := by
    apply List.dvd_sum
    intro y hy
    rw [List.mem_map] at hy
    obtain ⟨x, hx, rfl⟩ := hy
    exact Dvd.dvd.mul_left (hl x hx) _
  obtain ⟨a, ha⟩ := hQ
  obtain ⟨b, hb⟩ := hsum
  refine ⟨a - b, ?_⟩
  rw [ha, hb]; ring

/-- **Carried residues.**  The landing `2 t P_s - 1` is open to every gear dividing `P_s`
(with `g ≥ 3`, so that `g ∤ 1` and `g ∤ -1`... stated as: `g ∤ 2tP_s - 1` and `g ∤ 2tP_s + 1`). -/
theorem descent_open_base {g : ℕ} {t Ps : ℤ} (hg : 2 ≤ g) (hd : (g : ℤ) ∣ Ps) :
    OpenTo g (2 * t * Ps - 1) := by
  have h2 : (g : ℤ) ∣ 2 * t * Ps := Dvd.dvd.mul_left hd _
  have hg1 : ¬ (g : ℤ) ∣ 1 := by
    intro h
    have h1 := Int.le_of_dvd one_pos h
    have h2 : (2 : ℤ) ≤ g := (by exact_mod_cast hg)
    omega
  constructor
  · intro h
    have : (g : ℤ) ∣ 2 * t * Ps - (2 * t * Ps - 1) := Int.dvd_sub h2 h
    simp at this; exact hg1 this
  · intro h
    have e : 2 * t * Ps - 1 + 2 = 2 * t * Ps + 1 := by ring
    rw [e] at h
    have : (g : ℤ) ∣ (2 * t * Ps + 1) - 2 * t * Ps := Int.dvd_sub h h2
    simp at this; exact hg1 this

/-- **Fixed classes of `t`.**  A gear `g` strikes the left member iff `2 P_s t ≡ 1 (mod g)` and the
right member iff `2 P_s t ≡ -1 (mod g)`; both classes depend on `g` and `P_s` only. -/
theorem descent_strikes_iff {g : ℕ} {t Ps : ℤ} :
    ((g : ℤ) ∣ 2 * t * Ps - 1 ↔ 2 * Ps * t ≡ 1 [ZMOD g]) ∧
    ((g : ℤ) ∣ 2 * t * Ps - 1 + 2 ↔ 2 * Ps * t ≡ -1 [ZMOD g]) := by
  constructor
  · rw [Int.modEq_iff_dvd]
    have e : 1 - 2 * Ps * t = -(2 * t * Ps - 1) := by ring
    rw [e, dvd_neg]
  · rw [Int.modEq_iff_dvd]
    have e : -1 - 2 * Ps * t = -(2 * t * Ps - 1 + 2) := by ring
    rw [e, dvd_neg]

/-- **In the window** iff `q + 1 ≤ 2 t P_s ≤ q² - 1`. -/
theorem descent_in_window {q t Ps : ℤ} :
    (q < 2 * t * Ps - 1 ∧ 2 * t * Ps - 1 + 2 ≤ q ^ 2) ↔ (q + 1 < 2 * t * Ps ∧ 2 * t * Ps + 1 ≤ q ^ 2) := by
  constructor <;> rintro ⟨a, b⟩ <;> constructor <;> linarith

/-- **The teeth are symmetric.**  Gear `g` strikes the left member at `t` iff it strikes the right
member at `-t`: the two teeth of every gear on the `t`-line are `t₀` and `-t₀`. -/
theorem descent_teeth_symmetric {g : ℕ} {t Ps : ℤ} :
    (g : ℤ) ∣ 2 * t * Ps - 1 ↔ (g : ℤ) ∣ 2 * (-t) * Ps - 1 + 2 := by
  have e : 2 * (-t) * Ps - 1 + 2 = -(2 * t * Ps - 1) := by ring
  rw [e, dvd_neg]

/-- The landing family is a mirror family: the landing at `-t` is the reflection of the landing
at `t` about the home pair, member for member (`2(-t)P_s - 1 = -(2tP_s + 1)`). -/
theorem descent_reflect (t Ps : ℤ) :
    2 * (-t) * Ps - 1 = -(2 * t * Ps - 1 + 2) ∧ 2 * (-t) * Ps - 1 + 2 = -(2 * t * Ps - 1) := by
  constructor <;> ring

end MirrorWalk
