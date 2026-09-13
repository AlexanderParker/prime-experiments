/-
MirrorWalkSettle (round 47, 2026-09-13): settling gears one after another, and the origin lever.

Settling: a gear is settled by a step when the landing is open to it.  One gear is settled within
three consecutive multiples of the axis unit (MirrorWalkParts.exists_axis_open).  Here: a second
gear is settled by moving in strides of the first (the first stays settled, since sliding by a
multiple of it keeps its openness), within 2 + 2h further multiples; the same step adds any new
gear at the cost of a stride equal to the product of the gears already settled.
The origin lever: a landing L is certified open to h by any known-open column v with
h dividing L + v + 2, because L is then the flip of v about (L + v + 2)/2, an axis h divides.
-/
import MirrorWalkParts

namespace MirrorWalk

/-- Sliding a column by a multiple of `h` keeps its openness to `h`. -/
theorem openTo_add_of_dvd {h : ℕ} {x c : ℤ} (hc : (h : ℤ) ∣ c) :
    OpenTo h (x + c) ↔ OpenTo h x := by
  unfold OpenTo
  have e : x + c + 2 = (x + 2) + c := by ring
  rw [e]
  constructor
  · rintro ⟨h1, h2⟩
    refine ⟨fun hd => h1 ?_, fun hd => h2 ?_⟩
    · have := dvd_add hd hc; exact this
    · have := dvd_add hd hc; exact this
  · rintro ⟨h1, h2⟩
    refine ⟨fun hd => h1 ?_, fun hd => h2 ?_⟩
    · have := dvd_sub hd hc; simpa using this
    · have := dvd_sub hd hc; simpa using this

/-- Moving the axis by `j D` units moves the landing by `2 j D M`. -/
theorem flip_stride (k j D M n : ℤ) :
    flip ((k + j * D) * M) n = flip (k * M) n + 2 * j * D * M := by
  unfold flip; ring

/-- Two axes in the same stride family whose landings `h` strikes on the same side are congruent
in the stride index, when `h` is a prime not dividing `2 M D`. -/
theorem same_class_stride {h : ℕ} (hh : h.Prime) {M D n k₁ : ℤ} (hM : ¬ (h : ℤ) ∣ 2 * M * D)
    {j j' : ℤ} (hj : (h : ℤ) ∣ 2 * ((k₁ + j * D) * M) - n)
    (hj' : (h : ℤ) ∣ 2 * ((k₁ + j' * D) * M) - n) : (h : ℤ) ∣ j - j' := by
  have hd : (h : ℤ) ∣ (2 * M * D) * (j - j') := by
    have := dvd_sub hj hj'
    have e : 2 * ((k₁ + j * D) * M) - n - (2 * ((k₁ + j' * D) * M) - n) = (2 * M * D) * (j - j') := by
      ring
    rwa [e] at this
  rcases Int.Prime.dvd_mul' hh hd with h1 | h1
  · exact absurd h1 hM
  · exact h1

/-- **One gear, one step, in strides.**  For a prime `h ≥ 5` not dividing `2 M D`, among the
axes `(k₁ + j D) M`, `j = 0, 1, 2`, some landing is open to `h`. -/
theorem exists_axis_open_stride {h : ℕ} (hh : h.Prime) (h5 : 5 ≤ h) {M D n k₁ : ℤ}
    (hM : ¬ (h : ℤ) ∣ 2 * M * D) :
    ∃ j : ℤ, 0 ≤ j ∧ j ≤ 2 ∧ OpenTo h (flip ((k₁ + j * D) * M) n) := by
  by_contra hcon
  push_neg at hcon
  have bad : ∀ j : ℤ, 0 ≤ j → j ≤ 2 →
      (h : ℤ) ∣ 2 * ((k₁ + j * D) * M) - (n + 2) ∨ (h : ℤ) ∣ 2 * ((k₁ + j * D) * M) - n := by
    intro j hj1 hj2
    have := hcon j hj1 hj2
    unfold OpenTo at this
    push_neg at this
    by_cases hL : (h : ℤ) ∣ flip ((k₁ + j * D) * M) n
    · left
      have e : flip ((k₁ + j * D) * M) n = 2 * ((k₁ + j * D) * M) - (n + 2) := by unfold flip; ring
      rw [e] at hL; exact hL
    · right
      have hR := this hL
      have e : flip ((k₁ + j * D) * M) n + 2 = 2 * ((k₁ + j * D) * M) - n := by unfold flip; ring
      rw [e] at hR; exact hR
  have h0 := bad 0 le_rfl (by norm_num)
  have h1 := bad 1 (by norm_num) (by norm_num)
  have h2 := bad 2 (by norm_num) (by norm_num)
  have key : ∀ {x y : ℤ}, (h : ℤ) ∣ x - y → 0 < x - y → x - y ≤ 2 → (h : ℤ) ≤ 2 := by
    intro x y hd hpos hle
    exact le_trans (Int.le_of_dvd hpos hd) hle
  have hle : (h : ℤ) ≤ 2 := by
    rcases h0 with h0 | h0 <;> rcases h1 with h1 | h1 <;> rcases h2 with h2 | h2
    · exact key (same_class_stride hh hM h1 h0) (by norm_num) (by norm_num)
    · exact key (same_class_stride hh hM h1 h0) (by norm_num) (by norm_num)
    · exact key (same_class_stride hh hM h2 h0) (by norm_num) (by norm_num)
    · exact key (same_class_stride hh hM h2 h1) (by norm_num) (by norm_num)
    · exact key (same_class_stride hh hM h2 h1) (by norm_num) (by norm_num)
    · exact key (same_class_stride hh hM h2 h0) (by norm_num) (by norm_num)
    · exact key (same_class_stride hh hM h1 h0) (by norm_num) (by norm_num)
    · exact key (same_class_stride hh hM h1 h0) (by norm_num) (by norm_num)
  have : (5 : ℤ) ≤ h := by exact_mod_cast h5
  linarith

/-- **Two gears, one step.**  Distinct primes `h, h' ≥ 5`, neither dividing `2M`: some axis
`k M` with `k₀ ≤ k ≤ k₀ + 2 + 2h` lands the origin `n` on a column open to both.  The first
gear is settled within three multiples, the second by strides of the first. -/
theorem settle_two {h h' : ℕ} (hh : h.Prime) (hh' : h'.Prime) (h5 : 5 ≤ h) (h5' : 5 ≤ h')
    (hne : h ≠ h') {M n k₀ : ℤ} (hM : ¬ (h : ℤ) ∣ 2 * M) (hM' : ¬ (h' : ℤ) ∣ 2 * M) :
    ∃ k, k₀ ≤ k ∧ k ≤ k₀ + 2 + 2 * h ∧ OpenTo h (flip (k * M) n) ∧ OpenTo h' (flip (k * M) n) := by
  obtain ⟨k₁, hk1, hk2, hopen⟩ := exists_axis_open hh h5 (n := n) (k₀ := k₀) hM
  have hMD : ¬ (h' : ℤ) ∣ 2 * M * (h : ℤ) := by
    intro hd
    rcases Int.Prime.dvd_mul' hh' hd with h1 | h1
    · exact hM' h1
    · have : h' ∣ h := by exact_mod_cast h1
      rcases (Nat.dvd_prime hh).mp this with e | e
      · exact hh'.one_lt.ne' e
      · exact hne e.symm
  obtain ⟨j, hj0, hj2, hopen'⟩ := exists_axis_open_stride hh' h5' (n := n) (k₁ := k₁) hMD
  refine ⟨k₁ + j * h, by nlinarith, by nlinarith, ?_, hopen'⟩
  rw [flip_stride, openTo_add_of_dvd]
  · exact hopen
  · exact ⟨2 * j * M, by ring⟩

/-- **The origin lever.**  A landing `L` is open to `h` if some column `v` open to `h` has
`h` dividing `L + v + 2`: then `L` is the flip of `v` about the axis `(L + v + 2)/2`, which `h`
divides.  Every known twin, gear pair or home is such a `v` for the gears it is open to. -/
theorem anchor_certifies {h : ℕ} {v L a : ℤ} (ha : L + v + 2 = 2 * a) (hv : OpenTo h v)
    (hd : (h : ℤ) ∣ 2 * a) : OpenTo h L := by
  have e : L = flip a v := by unfold flip; linarith
  rw [e]
  exact (openTo_flip_iff hd).mpr hv

end MirrorWalk
