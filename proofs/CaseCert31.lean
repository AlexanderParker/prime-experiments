/-
THE 29->31 RUNG BY CASE-SPLIT LP DUALITY (GENERATED root).

Every configuration of machine 31 has its held gears [5, 7] at exactly
one of the 35 phase tuples, and each of those cases carries an exact
dual certificate of the restricted level-2 covering relaxation.  So
no window of 74 consecutive slots of machine 31 is fully blocked.

NO CENSUS HYPOTHESIS, NO PERIOD SCAN: the only inputs are the primes
up to 31 and 892 integers per case.
-/
import CaseCert31C0
import CaseCert31C1
import CaseCert31C2
import CaseCert31C3
import CaseCert31C4
import CaseCert31C5
import CaseCert31C6
import CaseCert31C7
import CaseCert31C8
import CaseCert31C9
import CaseCert31C10
import CaseCert31C11
import CaseCert31C12
import CaseCert31C13
import CaseCert31C14
import CaseCert31C15
import CaseCert31C16
import CaseCert31C17
import CaseCert31C18
import CaseCert31C19
import CaseCert31C20
import CaseCert31C21
import CaseCert31C22
import CaseCert31C23
import CaseCert31C24
import CaseCert31C25
import CaseCert31C26
import CaseCert31C27
import CaseCert31C28
import CaseCert31C29
import CaseCert31C30
import CaseCert31C31
import CaseCert31C32
import CaseCert31C33
import CaseCert31C34
import Machine31

namespace CaseCert31

set_option maxHeartbeats 4000000

/-- A slot that is not an opening of machine 31 is blocked by one
of its gears, in the certificate's (phase, offset) coordinates. -/
theorem blocked {p i : ℕ} (hp : 1 ≤ p) (h : ¬ Machine31.Exposed31 (p + i)) :
    (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true := by
  have e5 : (p % 5 + i) % 5 = (p + i) % 5 := by omega
  have e7 : (p % 7 + i) % 7 = (p + i) % 7 := by omega
  have e11 : (p % 11 + i) % 11 = (p + i) % 11 := by omega
  have e13 : (p % 13 + i) % 13 = (p + i) % 13 := by omega
  have e17 : (p % 17 + i) % 17 = (p + i) % 17 := by omega
  have e19 : (p % 19 + i) % 19 = (p + i) % 19 := by omega
  have e23 : (p % 23 + i) % 23 = (p + i) % 23 := by omega
  have e29 : (p % 29 + i) % 29 = (p + i) % 29 := by omega
  have e31 : (p % 31 + i) % 31 = (p + i) % 31 := by omega
  simp only [gb5, gb7, gb11, gb13, gb17, gb19, gb23, gb29, gb31, e5, e7, e11, e13, e17, e19, e23, e29, e31, Bool.or_eq_true, beq_iff_eq]
  by_contra hcon
  push Not at hcon
  apply h
  refine Machine31.exposed31_of (show 1 ≤ p + i by omega) (Machine29.exposed29_of (show 1 ≤ p + i by omega) (Machine23.exposed23_of (show 1 ≤ p + i by omega) (?_) ?_) ?_) ?_
  · rw [Machine19.exposed19_iff (show 1 ≤ p + i by omega)]
    simp only [Machine19.expT, Bool.and_eq_true, bne_iff_ne, ne_eq, and_assoc]
    tauto
  · unfold Machine23.Killed23
    omega
  · unfold Machine29.Killed29
    omega
  · unfold Machine31.Killed31
    omega

theorem nocase0 {p : ℕ} (e5 : p % 5 = 0) (e7 : p % 7 = 0)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov0 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q0 t) (plt0 t ht)
  rw [e5, e7] at h3
  simp only [pfree0_5 t ht, pfree0_7 t ht, Bool.false_or] at h3
  simpa only [c0_0, c0_1, c0_2, c0_3, c0_4, c0_5, c0_6] using h3

theorem nocase1 {p : ℕ} (e5 : p % 5 = 0) (e7 : p % 7 = 1)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov1 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q1 t) (plt1 t ht)
  rw [e5, e7] at h3
  simp only [pfree1_5 t ht, pfree1_7 t ht, Bool.false_or] at h3
  simpa only [c1_0, c1_1, c1_2, c1_3, c1_4, c1_5, c1_6] using h3

theorem nocase2 {p : ℕ} (e5 : p % 5 = 0) (e7 : p % 7 = 2)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov2 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q2 t) (plt2 t ht)
  rw [e5, e7] at h3
  simp only [pfree2_5 t ht, pfree2_7 t ht, Bool.false_or] at h3
  simpa only [c2_0, c2_1, c2_2, c2_3, c2_4, c2_5, c2_6] using h3

theorem nocase3 {p : ℕ} (e5 : p % 5 = 0) (e7 : p % 7 = 3)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov3 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q3 t) (plt3 t ht)
  rw [e5, e7] at h3
  simp only [pfree3_5 t ht, pfree3_7 t ht, Bool.false_or] at h3
  simpa only [c3_0, c3_1, c3_2, c3_3, c3_4, c3_5, c3_6] using h3

theorem nocase4 {p : ℕ} (e5 : p % 5 = 0) (e7 : p % 7 = 4)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov4 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q4 t) (plt4 t ht)
  rw [e5, e7] at h3
  simp only [pfree4_5 t ht, pfree4_7 t ht, Bool.false_or] at h3
  simpa only [c4_0, c4_1, c4_2, c4_3, c4_4, c4_5, c4_6] using h3

theorem nocase5 {p : ℕ} (e5 : p % 5 = 0) (e7 : p % 7 = 5)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov5 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q5 t) (plt5 t ht)
  rw [e5, e7] at h3
  simp only [pfree5_5 t ht, pfree5_7 t ht, Bool.false_or] at h3
  simpa only [c5_0, c5_1, c5_2, c5_3, c5_4, c5_5, c5_6] using h3

theorem nocase6 {p : ℕ} (e5 : p % 5 = 0) (e7 : p % 7 = 6)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov6 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q6 t) (plt6 t ht)
  rw [e5, e7] at h3
  simp only [pfree6_5 t ht, pfree6_7 t ht, Bool.false_or] at h3
  simpa only [c6_0, c6_1, c6_2, c6_3, c6_4, c6_5, c6_6] using h3

theorem nocase7 {p : ℕ} (e5 : p % 5 = 1) (e7 : p % 7 = 0)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov7 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q7 t) (plt7 t ht)
  rw [e5, e7] at h3
  simp only [pfree7_5 t ht, pfree7_7 t ht, Bool.false_or] at h3
  simpa only [c7_0, c7_1, c7_2, c7_3, c7_4, c7_5, c7_6] using h3

theorem nocase8 {p : ℕ} (e5 : p % 5 = 1) (e7 : p % 7 = 1)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov8 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q8 t) (plt8 t ht)
  rw [e5, e7] at h3
  simp only [pfree8_5 t ht, pfree8_7 t ht, Bool.false_or] at h3
  simpa only [c8_0, c8_1, c8_2, c8_3, c8_4, c8_5, c8_6] using h3

theorem nocase9 {p : ℕ} (e5 : p % 5 = 1) (e7 : p % 7 = 2)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov9 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q9 t) (plt9 t ht)
  rw [e5, e7] at h3
  simp only [pfree9_5 t ht, pfree9_7 t ht, Bool.false_or] at h3
  simpa only [c9_0, c9_1, c9_2, c9_3, c9_4, c9_5, c9_6] using h3

theorem nocase10 {p : ℕ} (e5 : p % 5 = 1) (e7 : p % 7 = 3)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov10 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q10 t) (plt10 t ht)
  rw [e5, e7] at h3
  simp only [pfree10_5 t ht, pfree10_7 t ht, Bool.false_or] at h3
  simpa only [c10_0, c10_1, c10_2, c10_3, c10_4, c10_5, c10_6] using h3

theorem nocase11 {p : ℕ} (e5 : p % 5 = 1) (e7 : p % 7 = 4)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov11 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q11 t) (plt11 t ht)
  rw [e5, e7] at h3
  simp only [pfree11_5 t ht, pfree11_7 t ht, Bool.false_or] at h3
  simpa only [c11_0, c11_1, c11_2, c11_3, c11_4, c11_5, c11_6] using h3

theorem nocase12 {p : ℕ} (e5 : p % 5 = 1) (e7 : p % 7 = 5)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov12 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q12 t) (plt12 t ht)
  rw [e5, e7] at h3
  simp only [pfree12_5 t ht, pfree12_7 t ht, Bool.false_or] at h3
  simpa only [c12_0, c12_1, c12_2, c12_3, c12_4, c12_5, c12_6] using h3

theorem nocase13 {p : ℕ} (e5 : p % 5 = 1) (e7 : p % 7 = 6)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov13 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q13 t) (plt13 t ht)
  rw [e5, e7] at h3
  simp only [pfree13_5 t ht, pfree13_7 t ht, Bool.false_or] at h3
  simpa only [c13_0, c13_1, c13_2, c13_3, c13_4, c13_5, c13_6] using h3

theorem nocase14 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 0)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov14 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q14 t) (plt14 t ht)
  rw [e5, e7] at h3
  simp only [pfree14_5 t ht, pfree14_7 t ht, Bool.false_or] at h3
  simpa only [c14_0, c14_1, c14_2, c14_3, c14_4, c14_5, c14_6] using h3

theorem nocase15 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 1)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov15 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q15 t) (plt15 t ht)
  rw [e5, e7] at h3
  simp only [pfree15_5 t ht, pfree15_7 t ht, Bool.false_or] at h3
  simpa only [c15_0, c15_1, c15_2, c15_3, c15_4, c15_5, c15_6] using h3

theorem nocase16 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 2)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov16 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q16 t) (plt16 t ht)
  rw [e5, e7] at h3
  simp only [pfree16_5 t ht, pfree16_7 t ht, Bool.false_or] at h3
  simpa only [c16_0, c16_1, c16_2, c16_3, c16_4, c16_5, c16_6] using h3

theorem nocase17 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 3)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov17 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q17 t) (plt17 t ht)
  rw [e5, e7] at h3
  simp only [pfree17_5 t ht, pfree17_7 t ht, Bool.false_or] at h3
  simpa only [c17_0, c17_1, c17_2, c17_3, c17_4, c17_5, c17_6] using h3

theorem nocase18 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 4)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov18 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q18 t) (plt18 t ht)
  rw [e5, e7] at h3
  simp only [pfree18_5 t ht, pfree18_7 t ht, Bool.false_or] at h3
  simpa only [c18_0, c18_1, c18_2, c18_3, c18_4, c18_5, c18_6] using h3

theorem nocase19 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 5)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov19 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q19 t) (plt19 t ht)
  rw [e5, e7] at h3
  simp only [pfree19_5 t ht, pfree19_7 t ht, Bool.false_or] at h3
  simpa only [c19_0, c19_1, c19_2, c19_3, c19_4, c19_5, c19_6] using h3

theorem nocase20 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 6)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov20 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q20 t) (plt20 t ht)
  rw [e5, e7] at h3
  simp only [pfree20_5 t ht, pfree20_7 t ht, Bool.false_or] at h3
  simpa only [c20_0, c20_1, c20_2, c20_3, c20_4, c20_5, c20_6] using h3

theorem nocase21 {p : ℕ} (e5 : p % 5 = 3) (e7 : p % 7 = 0)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov21 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q21 t) (plt21 t ht)
  rw [e5, e7] at h3
  simp only [pfree21_5 t ht, pfree21_7 t ht, Bool.false_or] at h3
  simpa only [c21_0, c21_1, c21_2, c21_3, c21_4, c21_5, c21_6] using h3

theorem nocase22 {p : ℕ} (e5 : p % 5 = 3) (e7 : p % 7 = 1)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov22 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q22 t) (plt22 t ht)
  rw [e5, e7] at h3
  simp only [pfree22_5 t ht, pfree22_7 t ht, Bool.false_or] at h3
  simpa only [c22_0, c22_1, c22_2, c22_3, c22_4, c22_5, c22_6] using h3

theorem nocase23 {p : ℕ} (e5 : p % 5 = 3) (e7 : p % 7 = 2)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov23 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q23 t) (plt23 t ht)
  rw [e5, e7] at h3
  simp only [pfree23_5 t ht, pfree23_7 t ht, Bool.false_or] at h3
  simpa only [c23_0, c23_1, c23_2, c23_3, c23_4, c23_5, c23_6] using h3

theorem nocase24 {p : ℕ} (e5 : p % 5 = 3) (e7 : p % 7 = 3)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov24 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q24 t) (plt24 t ht)
  rw [e5, e7] at h3
  simp only [pfree24_5 t ht, pfree24_7 t ht, Bool.false_or] at h3
  simpa only [c24_0, c24_1, c24_2, c24_3, c24_4, c24_5, c24_6] using h3

theorem nocase25 {p : ℕ} (e5 : p % 5 = 3) (e7 : p % 7 = 4)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov25 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q25 t) (plt25 t ht)
  rw [e5, e7] at h3
  simp only [pfree25_5 t ht, pfree25_7 t ht, Bool.false_or] at h3
  simpa only [c25_0, c25_1, c25_2, c25_3, c25_4, c25_5, c25_6] using h3

theorem nocase26 {p : ℕ} (e5 : p % 5 = 3) (e7 : p % 7 = 5)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov26 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q26 t) (plt26 t ht)
  rw [e5, e7] at h3
  simp only [pfree26_5 t ht, pfree26_7 t ht, Bool.false_or] at h3
  simpa only [c26_0, c26_1, c26_2, c26_3, c26_4, c26_5, c26_6] using h3

theorem nocase27 {p : ℕ} (e5 : p % 5 = 3) (e7 : p % 7 = 6)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov27 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q27 t) (plt27 t ht)
  rw [e5, e7] at h3
  simp only [pfree27_5 t ht, pfree27_7 t ht, Bool.false_or] at h3
  simpa only [c27_0, c27_1, c27_2, c27_3, c27_4, c27_5, c27_6] using h3

theorem nocase28 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 0)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov28 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q28 t) (plt28 t ht)
  rw [e5, e7] at h3
  simp only [pfree28_5 t ht, pfree28_7 t ht, Bool.false_or] at h3
  simpa only [c28_0, c28_1, c28_2, c28_3, c28_4, c28_5, c28_6] using h3

theorem nocase29 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 1)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov29 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q29 t) (plt29 t ht)
  rw [e5, e7] at h3
  simp only [pfree29_5 t ht, pfree29_7 t ht, Bool.false_or] at h3
  simpa only [c29_0, c29_1, c29_2, c29_3, c29_4, c29_5, c29_6] using h3

theorem nocase30 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 2)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov30 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q30 t) (plt30 t ht)
  rw [e5, e7] at h3
  simp only [pfree30_5 t ht, pfree30_7 t ht, Bool.false_or] at h3
  simpa only [c30_0, c30_1, c30_2, c30_3, c30_4, c30_5, c30_6] using h3

theorem nocase31 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 3)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov31 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q31 t) (plt31 t ht)
  rw [e5, e7] at h3
  simp only [pfree31_5 t ht, pfree31_7 t ht, Bool.false_or] at h3
  simpa only [c31_0, c31_1, c31_2, c31_3, c31_4, c31_5, c31_6] using h3

theorem nocase32 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 4)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov32 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q32 t) (plt32 t ht)
  rw [e5, e7] at h3
  simp only [pfree32_5 t ht, pfree32_7 t ht, Bool.false_or] at h3
  simpa only [c32_0, c32_1, c32_2, c32_3, c32_4, c32_5, c32_6] using h3

theorem nocase33 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 5)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov33 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q33 t) (plt33 t ht)
  rw [e5, e7] at h3
  simp only [pfree33_5 t ht, pfree33_7 t ht, Bool.false_or] at h3
  simpa only [c33_0, c33_1, c33_2, c33_3, c33_4, c33_5, c33_6] using h3

theorem nocase34 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 6)
    (hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true) : False := by
  refine nocov34 (r0 := p % 11) (r1 := p % 13) (r2 := p % 17) (r3 := p % 19) (r4 := p % 23) (r5 := p % 29) (r6 := p % 31) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q34 t) (plt34 t ht)
  rw [e5, e7] at h3
  simp only [pfree34_5 t ht, pfree34_7 t ht, Bool.false_or] at h3
  simpa only [c34_0, c34_1, c34_2, c34_3, c34_4, c34_5, c34_6] using h3

/-- **`F(31) <= 74` by the case split**: every window of 74
consecutive slots contains an opening of machine 31. -/
theorem no_run {p : ℕ} (hp : 1 ≤ p) :
    ∃ i < 74, Machine31.Exposed31 (p + i) := by
  by_contra hc
  push Not at hc
  have hall : ∀ i, i < 74 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i) = true :=
    fun i hi => blocked hp (hc i hi)
  have d5 : p % 5 = 0 ∨ p % 5 = 1 ∨ p % 5 = 2 ∨ p % 5 = 3 ∨ p % 5 = 4 := by omega
  have d7 : p % 7 = 0 ∨ p % 7 = 1 ∨ p % 7 = 2 ∨ p % 7 = 3 ∨ p % 7 = 4 ∨ p % 7 = 5 ∨ p % 7 = 6 := by omega
  rcases d5 with e5 | e5 | e5 | e5 | e5
  · skip
    rcases d7 with e7 | e7 | e7 | e7 | e7 | e7 | e7
    · exact nocase0 e5 e7 hall
    · exact nocase1 e5 e7 hall
    · exact nocase2 e5 e7 hall
    · exact nocase3 e5 e7 hall
    · exact nocase4 e5 e7 hall
    · exact nocase5 e5 e7 hall
    · exact nocase6 e5 e7 hall
  · skip
    rcases d7 with e7 | e7 | e7 | e7 | e7 | e7 | e7
    · exact nocase7 e5 e7 hall
    · exact nocase8 e5 e7 hall
    · exact nocase9 e5 e7 hall
    · exact nocase10 e5 e7 hall
    · exact nocase11 e5 e7 hall
    · exact nocase12 e5 e7 hall
    · exact nocase13 e5 e7 hall
  · skip
    rcases d7 with e7 | e7 | e7 | e7 | e7 | e7 | e7
    · exact nocase14 e5 e7 hall
    · exact nocase15 e5 e7 hall
    · exact nocase16 e5 e7 hall
    · exact nocase17 e5 e7 hall
    · exact nocase18 e5 e7 hall
    · exact nocase19 e5 e7 hall
    · exact nocase20 e5 e7 hall
  · skip
    rcases d7 with e7 | e7 | e7 | e7 | e7 | e7 | e7
    · exact nocase21 e5 e7 hall
    · exact nocase22 e5 e7 hall
    · exact nocase23 e5 e7 hall
    · exact nocase24 e5 e7 hall
    · exact nocase25 e5 e7 hall
    · exact nocase26 e5 e7 hall
    · exact nocase27 e5 e7 hall
  · skip
    rcases d7 with e7 | e7 | e7 | e7 | e7 | e7 | e7
    · exact nocase28 e5 e7 hall
    · exact nocase29 e5 e7 hall
    · exact nocase30 e5 e7 hall
    · exact nocase31 e5 e7 hall
    · exact nocase32 e5 e7 hall
    · exact nocase33 e5 e7 hall
    · exact nocase34 e5 e7 hall

theorem F_le (n : ℕ) : Machine31.g31 n ≤ 74 := by
  by_contra hcon
  obtain ⟨i, hi, hE⟩ := no_run (p := Machine31.opSeq31 n + 1)
    (by have := Machine31.opSeq31_pos n; omega)
  have hgap : Machine31.g31 n = Machine31.opSeq31 (n + 1) - Machine31.opSeq31 n := rfl
  have hlt := Machine31.opSeq31_lt_succ n
  exact Machine31.opSeq31_gap_empty n (Machine31.opSeq31 n + 1 + i)
    (by omega) (by omega) hE

/-- **(D) at alpha = 3 at the 29->31 step, BY CASE-SPLIT LP
DUALITY**: every gap of machine 31 is at most `F(29) + 31 = 74`.
No census hypothesis, no period scan - only the primes up to 31
and the 35 case certificates. -/
theorem D_29_31_case (n : ℕ) : Machine31.g31 n ≤ 43 + 31 :=
  F_le n

end CaseCert31
