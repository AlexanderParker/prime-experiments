/-
THE 31->37 RUNG BY CASE-SPLIT LP DUALITY (GENERATED TIERED root).

Every configuration of machine 37 has its held gears [5, 7, 11] at exactly
one of the 385 phase tuples, and each of those cases carries an exact
dual certificate of the restricted level-2 covering relaxation.  So
no window of 95 consecutive slots of machine 37 is fully blocked.

NO CENSUS HYPOTHESIS, NO PERIOD SCAN: the only inputs are the primes
up to 37 and 1049 integers per case.

TIERED (round 30): the 385 cases are assembled through 35 sub-roots
CaseCert37T0 .. T34, one per residue tuple of the held gears [5, 7],
so that no single lean process elaborates more than 11 case bridges
(the flat root reached 53.7 GB and crashed the machine, R29.5).
-/
import CaseCert37T0
import CaseCert37T1
import CaseCert37T2
import CaseCert37T3
import CaseCert37T4
import CaseCert37T5
import CaseCert37T6
import CaseCert37T7
import CaseCert37T8
import CaseCert37T9
import CaseCert37T10
import CaseCert37T11
import CaseCert37T12
import CaseCert37T13
import CaseCert37T14
import CaseCert37T15
import CaseCert37T16
import CaseCert37T17
import CaseCert37T18
import CaseCert37T19
import CaseCert37T20
import CaseCert37T21
import CaseCert37T22
import CaseCert37T23
import CaseCert37T24
import CaseCert37T25
import CaseCert37T26
import CaseCert37T27
import CaseCert37T28
import CaseCert37T29
import CaseCert37T30
import CaseCert37T31
import CaseCert37T32
import CaseCert37T33
import CaseCert37T34
import Machine37

namespace CaseCert37

set_option maxHeartbeats 4000000

/-- A slot that is not an opening of machine 37 is blocked by one
of its gears, in the certificate's (phase, offset) coordinates. -/
theorem blocked {p i : ℕ} (hp : 1 ≤ p) (h : ¬ Machine37.Exposed37 (p + i)) :
    (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true := by
  have e5 : (p % 5 + i) % 5 = (p + i) % 5 := by omega
  have e7 : (p % 7 + i) % 7 = (p + i) % 7 := by omega
  have e11 : (p % 11 + i) % 11 = (p + i) % 11 := by omega
  have e13 : (p % 13 + i) % 13 = (p + i) % 13 := by omega
  have e17 : (p % 17 + i) % 17 = (p + i) % 17 := by omega
  have e19 : (p % 19 + i) % 19 = (p + i) % 19 := by omega
  have e23 : (p % 23 + i) % 23 = (p + i) % 23 := by omega
  have e29 : (p % 29 + i) % 29 = (p + i) % 29 := by omega
  have e31 : (p % 31 + i) % 31 = (p + i) % 31 := by omega
  have e37 : (p % 37 + i) % 37 = (p + i) % 37 := by omega
  simp only [gb5, gb7, gb11, gb13, gb17, gb19, gb23, gb29, gb31, gb37, e5, e7, e11, e13, e17, e19, e23, e29, e31, e37, Bool.or_eq_true, beq_iff_eq]
  by_contra hcon
  push Not at hcon
  apply h
  refine Machine37.exposed37_of (show 1 ≤ p + i by omega) (Machine31.exposed31_of (show 1 ≤ p + i by omega) (Machine29.exposed29_of (show 1 ≤ p + i by omega) (Machine23.exposed23_of (show 1 ≤ p + i by omega) (?_) ?_) ?_) ?_) ?_
  · rw [Machine19.exposed19_iff (show 1 ≤ p + i by omega)]
    simp only [Machine19.expT, Bool.and_eq_true, bne_iff_ne, ne_eq, and_assoc]
    tauto
  · unfold Machine23.Killed23
    omega
  · unfold Machine29.Killed29
    omega
  · unfold Machine31.Killed31
    omega
  · unfold Machine37.Killed37
    omega

/-- **`F(37) <= 95` by the case split**: every window of 95
consecutive slots contains an opening of machine 37. -/
theorem no_run {p : ℕ} (hp : 1 ≤ p) :
    ∃ i < 95, Machine37.Exposed37 (p + i) := by
  by_contra hc
  push Not at hc
  have hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true :=
    fun i hi => blocked hp (hc i hi)
  have d5 : p % 5 = 0 ∨ p % 5 = 1 ∨ p % 5 = 2 ∨ p % 5 = 3 ∨ p % 5 = 4 := by omega
  have d7 : p % 7 = 0 ∨ p % 7 = 1 ∨ p % 7 = 2 ∨ p % 7 = 3 ∨ p % 7 = 4 ∨ p % 7 = 5 ∨ p % 7 = 6 := by omega
  rcases d5 with e5 | e5 | e5 | e5 | e5
  · skip
    rcases d7 with e7 | e7 | e7 | e7 | e7 | e7 | e7
    · exact nopair0 e5 e7 hall
    · exact nopair1 e5 e7 hall
    · exact nopair2 e5 e7 hall
    · exact nopair3 e5 e7 hall
    · exact nopair4 e5 e7 hall
    · exact nopair5 e5 e7 hall
    · exact nopair6 e5 e7 hall
  · skip
    rcases d7 with e7 | e7 | e7 | e7 | e7 | e7 | e7
    · exact nopair7 e5 e7 hall
    · exact nopair8 e5 e7 hall
    · exact nopair9 e5 e7 hall
    · exact nopair10 e5 e7 hall
    · exact nopair11 e5 e7 hall
    · exact nopair12 e5 e7 hall
    · exact nopair13 e5 e7 hall
  · skip
    rcases d7 with e7 | e7 | e7 | e7 | e7 | e7 | e7
    · exact nopair14 e5 e7 hall
    · exact nopair15 e5 e7 hall
    · exact nopair16 e5 e7 hall
    · exact nopair17 e5 e7 hall
    · exact nopair18 e5 e7 hall
    · exact nopair19 e5 e7 hall
    · exact nopair20 e5 e7 hall
  · skip
    rcases d7 with e7 | e7 | e7 | e7 | e7 | e7 | e7
    · exact nopair21 e5 e7 hall
    · exact nopair22 e5 e7 hall
    · exact nopair23 e5 e7 hall
    · exact nopair24 e5 e7 hall
    · exact nopair25 e5 e7 hall
    · exact nopair26 e5 e7 hall
    · exact nopair27 e5 e7 hall
  · skip
    rcases d7 with e7 | e7 | e7 | e7 | e7 | e7 | e7
    · exact nopair28 e5 e7 hall
    · exact nopair29 e5 e7 hall
    · exact nopair30 e5 e7 hall
    · exact nopair31 e5 e7 hall
    · exact nopair32 e5 e7 hall
    · exact nopair33 e5 e7 hall
    · exact nopair34 e5 e7 hall

theorem F_le (n : ℕ) : Machine37.g37 n ≤ 95 := by
  by_contra hcon
  obtain ⟨i, hi, hE⟩ := no_run (p := Machine37.opSeq37 n + 1)
    (by have := Machine37.opSeq37_pos n; omega)
  have hgap : Machine37.g37 n = Machine37.opSeq37 (n + 1) - Machine37.opSeq37 n := rfl
  have hlt := Machine37.opSeq37_lt_succ n
  exact Machine37.opSeq37_gap_empty n (Machine37.opSeq37 n + 1 + i)
    (by omega) (by omega) hE

/-- **(D) at alpha = 3 at the 31->37 step, BY CASE-SPLIT LP
DUALITY**: every gap of machine 37 is at most `F(31) + 37 = 95`.
No census hypothesis, no period scan - only the primes up to 37
and the 385 case certificates (via 35 tiers). -/
theorem D_31_37_case (n : ℕ) : Machine37.g37 n ≤ 58 + 37 :=
  F_le n

end CaseCert37
