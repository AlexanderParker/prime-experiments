/-
THE 19->23 RUNG BY CASE-SPLIT LP DUALITY (GENERATED root).

Every configuration of machine 23 has its held gears [5] at exactly
one of the 5 phase tuples, and each of those cases carries an exact
dual certificate of the restricted level-2 covering relaxation.  So
no window of 48 consecutive slots of machine 23 is fully blocked.

NO CENSUS HYPOTHESIS, NO PERIOD SCAN: the only inputs are the primes
up to 23 and 480 integers per case.
-/
import CaseCert23C0
import CaseCert23C1
import CaseCert23C2
import CaseCert23C3
import CaseCert23C4
import Machine23

namespace CaseCert23

set_option maxHeartbeats 4000000

/-- A slot that is not an opening of machine 23 is blocked by one
of its gears, in the certificate's (phase, offset) coordinates. -/
theorem blocked {p i : ℕ} (hp : 1 ≤ p) (h : ¬ Machine23.Exposed23 (p + i)) :
    (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i) = true := by
  have e5 : (p % 5 + i) % 5 = (p + i) % 5 := by omega
  have e7 : (p % 7 + i) % 7 = (p + i) % 7 := by omega
  have e11 : (p % 11 + i) % 11 = (p + i) % 11 := by omega
  have e13 : (p % 13 + i) % 13 = (p + i) % 13 := by omega
  have e17 : (p % 17 + i) % 17 = (p + i) % 17 := by omega
  have e19 : (p % 19 + i) % 19 = (p + i) % 19 := by omega
  have e23 : (p % 23 + i) % 23 = (p + i) % 23 := by omega
  simp only [gb5, gb7, gb11, gb13, gb17, gb19, gb23, e5, e7, e11, e13, e17, e19, e23, Bool.or_eq_true, beq_iff_eq]
  by_contra hcon
  push Not at hcon
  apply h
  refine Machine23.exposed23_of (show 1 ≤ p + i by omega) (?_) ?_
  · rw [Machine19.exposed19_iff (show 1 ≤ p + i by omega)]
    simp only [Machine19.expT, Bool.and_eq_true, bne_iff_ne, ne_eq, and_assoc]
    tauto
  · unfold Machine23.Killed23
    omega

theorem nocase0 {p : ℕ} (e5 : p % 5 = 0)
    (hall : ∀ i, i < 48 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i) = true) : False := by
  refine nocov0 (r0 := p % 7) (r1 := p % 11) (r2 := p % 13) (r3 := p % 17) (r4 := p % 19) (r5 := p % 23) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q0 t) (plt0 t ht)
  rw [e5] at h3
  simp only [pfree0_5 t ht, Bool.false_or] at h3
  simpa only [c0_0, c0_1, c0_2, c0_3, c0_4, c0_5] using h3

theorem nocase1 {p : ℕ} (e5 : p % 5 = 1)
    (hall : ∀ i, i < 48 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i) = true) : False := by
  refine nocov1 (r0 := p % 7) (r1 := p % 11) (r2 := p % 13) (r3 := p % 17) (r4 := p % 19) (r5 := p % 23) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q1 t) (plt1 t ht)
  rw [e5] at h3
  simp only [pfree1_5 t ht, Bool.false_or] at h3
  simpa only [c1_0, c1_1, c1_2, c1_3, c1_4, c1_5] using h3

theorem nocase2 {p : ℕ} (e5 : p % 5 = 2)
    (hall : ∀ i, i < 48 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i) = true) : False := by
  refine nocov2 (r0 := p % 7) (r1 := p % 11) (r2 := p % 13) (r3 := p % 17) (r4 := p % 19) (r5 := p % 23) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q2 t) (plt2 t ht)
  rw [e5] at h3
  simp only [pfree2_5 t ht, Bool.false_or] at h3
  simpa only [c2_0, c2_1, c2_2, c2_3, c2_4, c2_5] using h3

theorem nocase3 {p : ℕ} (e5 : p % 5 = 3)
    (hall : ∀ i, i < 48 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i) = true) : False := by
  refine nocov3 (r0 := p % 7) (r1 := p % 11) (r2 := p % 13) (r3 := p % 17) (r4 := p % 19) (r5 := p % 23) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q3 t) (plt3 t ht)
  rw [e5] at h3
  simp only [pfree3_5 t ht, Bool.false_or] at h3
  simpa only [c3_0, c3_1, c3_2, c3_3, c3_4, c3_5] using h3

theorem nocase4 {p : ℕ} (e5 : p % 5 = 4)
    (hall : ∀ i, i < 48 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i) = true) : False := by
  refine nocov4 (r0 := p % 7) (r1 := p % 11) (r2 := p % 13) (r3 := p % 17) (r4 := p % 19) (r5 := p % 23) (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) ?_
  intro t ht
  have h3 := hall (q4 t) (plt4 t ht)
  rw [e5] at h3
  simp only [pfree4_5 t ht, Bool.false_or] at h3
  simpa only [c4_0, c4_1, c4_2, c4_3, c4_4, c4_5] using h3

/-- **`F(23) <= 48` by the case split**: every window of 48
consecutive slots contains an opening of machine 23. -/
theorem no_run {p : ℕ} (hp : 1 ≤ p) :
    ∃ i < 48, Machine23.Exposed23 (p + i) := by
  by_contra hc
  push Not at hc
  have hall : ∀ i, i < 48 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i) = true :=
    fun i hi => blocked hp (hc i hi)
  have d5 : p % 5 = 0 ∨ p % 5 = 1 ∨ p % 5 = 2 ∨ p % 5 = 3 ∨ p % 5 = 4 := by omega
  rcases d5 with e5 | e5 | e5 | e5 | e5
  · exact nocase0 e5 hall
  · exact nocase1 e5 hall
  · exact nocase2 e5 hall
  · exact nocase3 e5 hall
  · exact nocase4 e5 hall

theorem F_le (n : ℕ) : Machine23.g23 n ≤ 48 := by
  by_contra hcon
  obtain ⟨i, hi, hE⟩ := no_run (p := Machine23.opSeq23 n + 1)
    (by have := Machine23.opSeq23_pos n; omega)
  have hgap : Machine23.g23 n = Machine23.opSeq23 (n + 1) - Machine23.opSeq23 n := rfl
  have hlt := Machine23.opSeq23_lt_succ n
  exact Machine23.opSeq23_gap_empty n (Machine23.opSeq23 n + 1 + i)
    (by omega) (by omega) hE

/-- **(D) at alpha = 3 at the 19->23 step, BY CASE-SPLIT LP
DUALITY**: every gap of machine 23 is at most `F(19) + 23 = 48`.
No census hypothesis, no period scan - only the primes up to 23
and the 5 case certificates. -/
theorem D_19_23_case (n : ℕ) : Machine23.g23 n ≤ 25 + 23 :=
  F_le n

end CaseCert23
