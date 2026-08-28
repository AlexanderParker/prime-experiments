/-
THE DICTIONARY VEHICLE AT MACHINE 31 - the input of the SEVENTH (D) rung
(round 25).

Identical in shape to `Machine29Dict.lean` one gear up, which is the point:
the vehicle is a template, and a rung is now a census plus a transcription.
Gear 37's teeth are `{6, 31}` (`6 * 6 = 36 = 37 - 1`, `6 * 31 = 186 =
5 * 37 + 1`), so the qualifying floor on machine 31's gap word is
`2u'' = 12`, and the budget is `F(31) + 37 = 58 + 37 = 95`.

    j        2      3      4      5      6      7      8
    |D_j|  1,253  8,155 18,566 13,049  2,120     42      0
    Q_j       68     85     90     91     90     88      -    max 91 <= 95

43,185 tuples against a period of 33,426,748,355 slots.

TWO THINGS WORTH NOTING IN THAT TABLE.  First, `Q_j` is NOT MONOTONE in `j`
here - it rises 68, 85, 90, 91 and then FALLS BACK to 90, 88.  Machine 31 is
the first machine in this ledger where the qualifying spectrum turns over
before it goes vacuous; at machines 19, 23 and 29 it was non-decreasing and
then saturated.  Second, the maximum is at `j = 5`, so the binding
constraint on this rung is a five-gap window with three qualifying interiors,
not the two-gap statement.

WHAT IS KERNEL-CHECKED AND WHAT IS NOT - as at machine 29:

  * KERNEL-CHECKED (`Machine31D2..D7`, `decide +kernel`, no `native_decide`):
    every window in `Dj` sums to at most `Q_j(31; 12)`.
  * KERNEL-CHECKED (this file): those list facts plus `Census31` give
    `SpectrumBound g31 2 68` and `∀ j ≥ 3, QualBound g31 6 j 91`.
  * A HYPOTHESIS, NOT PROVED HERE: `Census31`.  A full-period claim about
    33,426,748,355 slots, measured by `research/qual_dict.py` and gated by
    `research/qual_dict_gate31.py` (whole-period rescan at an unrelated chunk
    size, gap count asserted equal to `prod (q-2) = 6,226,553,025`, `F(31) =
    58` against the corpus, and transcription against these very files).

The measured spectrum also supplies `F_2(31) = 68` exactly, which is the
value Constructor's `A_5(29)` survivor closure and Mechanic's pair census
both report - a third independent route to that integer.
-/

import Machine31D2
import Machine31D3
import Machine31D4
import Machine31D5
import Machine31D6
import Machine31D7
import Machine31Q

namespace Machine31

/-- **The machine-31 census input** - the ONLY ingredient of the 31->37 rung
that is not kernel-checked.  Each `Ej` says the depth-`j` dictionary CONTAINS
every realised window of `j` consecutive gaps whose interiors all reach the
floor `12 = 2u''` of gear 37; `run` says no six consecutive gaps reach it. -/
structure Census31 : Prop where
  E2 : ∀ n, (g31 n, g31 (n + 1)) ∈ D2
  E3 : ∀ n, 12 ≤ g31 (n + 1) → (g31 n, g31 (n + 1), g31 (n + 2)) ∈ D3
  E4 : ∀ n, 12 ≤ g31 (n + 1) → 12 ≤ g31 (n + 2) →
    (g31 n, g31 (n + 1), g31 (n + 2), g31 (n + 3)) ∈ D4
  E5 : ∀ n, 12 ≤ g31 (n + 1) → 12 ≤ g31 (n + 2) → 12 ≤ g31 (n + 3) →
    (g31 n, g31 (n + 1), g31 (n + 2), g31 (n + 3), g31 (n + 4)) ∈ D5
  E6 : ∀ n, 12 ≤ g31 (n + 1) → 12 ≤ g31 (n + 2) → 12 ≤ g31 (n + 3) →
    12 ≤ g31 (n + 4) →
    (g31 n, g31 (n + 1), g31 (n + 2), g31 (n + 3), g31 (n + 4),
      g31 (n + 5)) ∈ D6
  E7 : ∀ n, 12 ≤ g31 (n + 1) → 12 ≤ g31 (n + 2) → 12 ≤ g31 (n + 3) →
    12 ≤ g31 (n + 4) → 12 ≤ g31 (n + 5) →
    (g31 n, g31 (n + 1), g31 (n + 2), g31 (n + 3), g31 (n + 4),
      g31 (n + 5), g31 (n + 6)) ∈ D7
  run : ∀ n, ¬ (12 ≤ g31 (n + 1) ∧ 12 ≤ g31 (n + 2) ∧ 12 ≤ g31 (n + 3) ∧
    12 ≤ g31 (n + 4) ∧ 12 ≤ g31 (n + 5) ∧ 12 ≤ g31 (n + 6))

/-! ## Reading the window sums off the dictionaries -/

theorem w2 (a : ℕ) : Spectrum.windowSum g31 a 2 = g31 a + g31 (a + 1) := by
  simp [Spectrum.windowSum, Finset.sum_range_succ]

theorem w3 (a : ℕ) :
    Spectrum.windowSum g31 a 3 = g31 a + g31 (a + 1) + g31 (a + 2) := by
  simp [Spectrum.windowSum, Finset.sum_range_succ]

theorem w4 (a : ℕ) : Spectrum.windowSum g31 a 4 =
    g31 a + g31 (a + 1) + g31 (a + 2) + g31 (a + 3) := by
  simp [Spectrum.windowSum, Finset.sum_range_succ]

theorem w5 (a : ℕ) : Spectrum.windowSum g31 a 5 =
    g31 a + g31 (a + 1) + g31 (a + 2) + g31 (a + 3) + g31 (a + 4) := by
  simp [Spectrum.windowSum, Finset.sum_range_succ]

theorem w6 (a : ℕ) : Spectrum.windowSum g31 a 6 =
    g31 a + g31 (a + 1) + g31 (a + 2) + g31 (a + 3) + g31 (a + 4)
      + g31 (a + 5) := by
  simp [Spectrum.windowSum, Finset.sum_range_succ]

theorem w7 (a : ℕ) : Spectrum.windowSum g31 a 7 =
    g31 a + g31 (a + 1) + g31 (a + 2) + g31 (a + 3) + g31 (a + 4)
      + g31 (a + 5) + g31 (a + 6) := by
  simp [Spectrum.windowSum, Finset.sum_range_succ]

/-- **`F_2(31) <= 68`** from the pair dictionary. -/
theorem spectrum31_two (h : Census31) : Spectrum.SpectrumBound g31 2 68 := by
  intro a
  have hm := List.all_eq_true.mp D2_ok _ (h.E2 a)
  simp only [Nat.ble_eq] at hm
  rw [w2]
  exact hm

/-- `Q_3(31; 12) <= 85`. -/
theorem qual31_three (h : Census31) : Spectrum.QualBound g31 6 3 85 := by
  intro a hq
  have h1 : 12 ≤ g31 (a + 1) := hq 1 (by omega) (by omega)
  have hm := List.all_eq_true.mp D3_ok _ (h.E3 a h1)
  simp only [Nat.ble_eq] at hm
  rw [w3]
  exact hm

/-- `Q_4(31; 12) <= 90`. -/
theorem qual31_four (h : Census31) : Spectrum.QualBound g31 6 4 90 := by
  intro a hq
  have h1 : 12 ≤ g31 (a + 1) := hq 1 (by omega) (by omega)
  have h2 : 12 ≤ g31 (a + 2) := hq 2 (by omega) (by omega)
  have hm := List.all_eq_true.mp D4_ok _ (h.E4 a h1 h2)
  simp only [Nat.ble_eq] at hm
  rw [w4]
  exact hm

/-- `Q_5(31; 12) <= 91` - the LARGEST entry of the ladder, and the constraint
that actually binds this rung. -/
theorem qual31_five (h : Census31) : Spectrum.QualBound g31 6 5 91 := by
  intro a hq
  have h1 : 12 ≤ g31 (a + 1) := hq 1 (by omega) (by omega)
  have h2 : 12 ≤ g31 (a + 2) := hq 2 (by omega) (by omega)
  have h3 : 12 ≤ g31 (a + 3) := hq 3 (by omega) (by omega)
  have hm := List.all_eq_true.mp D5_ok _ (h.E5 a h1 h2 h3)
  simp only [Nat.ble_eq] at hm
  rw [w5]
  exact hm

/-- `Q_6(31; 12) <= 90` - the spectrum has TURNED OVER. -/
theorem qual31_six (h : Census31) : Spectrum.QualBound g31 6 6 90 := by
  intro a hq
  have h1 : 12 ≤ g31 (a + 1) := hq 1 (by omega) (by omega)
  have h2 : 12 ≤ g31 (a + 2) := hq 2 (by omega) (by omega)
  have h3 : 12 ≤ g31 (a + 3) := hq 3 (by omega) (by omega)
  have h4 : 12 ≤ g31 (a + 4) := hq 4 (by omega) (by omega)
  have hm := List.all_eq_true.mp D6_ok _ (h.E6 a h1 h2 h3 h4)
  simp only [Nat.ble_eq] at hm
  rw [w6]
  exact hm

/-- `Q_7(31; 12) <= 88`. -/
theorem qual31_seven (h : Census31) : Spectrum.QualBound g31 6 7 88 := by
  intro a hq
  have h1 : 12 ≤ g31 (a + 1) := hq 1 (by omega) (by omega)
  have h2 : 12 ≤ g31 (a + 2) := hq 2 (by omega) (by omega)
  have h3 : 12 ≤ g31 (a + 3) := hq 3 (by omega) (by omega)
  have h4 : 12 ≤ g31 (a + 4) := hq 4 (by omega) (by omega)
  have h5 : 12 ≤ g31 (a + 5) := hq 5 (by omega) (by omega)
  have hm := List.all_eq_true.mp D7_ok _ (h.E7 a h1 h2 h3 h4 h5)
  simp only [Nat.ble_eq] at hm
  rw [w7]
  exact hm

/-- **Every depth at once**: `Q_j(31; 12) <= 91` for every `j >= 3`.
Depths 3..7 from the dictionaries; depth `j >= 8` is VACUOUS, because a
qualifying window of `j >= 8` gaps has six consecutive qualifying interiors,
which `Census31.run` forbids. -/
theorem qual31_all (h : Census31) :
    ∀ j, 3 ≤ j → Spectrum.QualBound g31 6 j 91 := by
  intro j hj a hq
  by_cases h8 : 8 ≤ j
  · exact absurd
      ⟨hq 1 (by omega) (by omega), hq 2 (by omega) (by omega),
        hq 3 (by omega) (by omega), hq 4 (by omega) (by omega),
        hq 5 (by omega) (by omega), hq 6 (by omega) (by omega)⟩ (h.run a)
  · interval_cases j
    · exact le_trans (qual31_three h a hq) (by omega)
    · exact le_trans (qual31_four h a hq) (by omega)
    · exact qual31_five h a hq
    · exact le_trans (qual31_six h a hq) (by omega)
    · exact le_trans (qual31_seven h a hq) (by omega)

/-- **The criterion at 31->37**, as integers:
`max (F_2(31), max_j Q_j(31; 12)) = max (68, 91) = 91 <= 95 = F(31) + 37`,
margin 4 - the widest margin of any rung so far. -/
theorem criterion_31_37 : max 68 91 ≤ 58 + 37 := by decide

end Machine31
