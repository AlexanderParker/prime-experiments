/-
THE DICTIONARY VEHICLE: machine 29's whole qualifying spectrum from an
EXPLICIT FINITE LIST, with the census as one named hypothesis (round 25).

Round-24 verdict 17 measured the per-rung period-scan vehicle dead past
23->29 (~170 h at 29->31), and round-24 verdict 15 named what replaces it:

    "What CAN be [kernel-checked]: the longest-path value over an EXPLICIT
     edge set, with `E contains every realised tuple` as a named hypothesis
     a census discharges."

This file is that object, in the sharpest form the merge law can consume.
`MergeLaw.newgap_le_step` needs exactly two facts about the OLD machine's
own gap word:

    (i)  `Spectrum.SpectrumBound g29 2 F2`     with `F2 <= 43 + 31`
    (ii) `∀ j ≥ 3, Spectrum.QualBound g29 5 j (Q j)`  with `Q j <= 43 + 31`

and `Spectrum.QualBound g29 5 j Qj` quantifies only over windows whose
INTERIOR gaps all reach the floor `2u'' = 10`.  So the entire input is the
family of QUALIFYING WINDOW DICTIONARIES `D2 .. D7` - the realised windows
of `j` consecutive machine-29 gaps with qualifying interiors - together with
the fact that no SIX consecutive gaps qualify, which makes every depth
`j ≥ 8` vacuous.  Those dictionaries are small (730, 3692, 6688, 3915, 789,
46 tuples; 15,860 in all) where the period is 1,078,282,205 slots.

WHAT IS KERNEL-CHECKED AND WHAT IS NOT - stated plainly, because it is the
whole point of the vehicle:

  * KERNEL-CHECKED (`Machine29D2..D7`, `decide +kernel`, no `native_decide`):
    every window in `Dj` sums to at most `Q_j(29; 10)`, the values
    55, 65, 68, 71, 71, 71 for `j = 2 .. 7`.
  * KERNEL-CHECKED (this file): those list facts plus `Census29` give
    `SpectrumBound g29 2 55` and `∀ j ≥ 3, QualBound g29 5 j 71`, hence
    (by `Machine31.lean`) the 29->31 rung of the (D) ladder.
  * A HYPOTHESIS, NOT PROVED HERE: `Census29` - that the lists CONTAIN every
    realised qualifying window, and that no six consecutive gaps qualify.
    That is a full-period claim about 1,078,282,205 slots.  It was measured
    exactly by `research/qual_dict.py`, whose output is gate-checked against
    the corpus ladder at machines 19 and 23 (where it reproduces
    `F_j(19) = 25,31,35,38`, `Q_j(19;8) = 31,35,37,38`, `F_j(23) =
    34,39,50,58,65,77,83,88` and `Q_j(23;10) = 39,43,50,55,60` - all
    kernel-checked values in this ledger - and `F(29) = 43`).  No claim is
    made that a Lean kernel has seen machine 29's period.

The measured spectrum this file consumes ALSO reproduces, independently and
exactly, two published numbers of the project: `F_2(29) = 55` (Constructor's
`A_5(23)` survivor closure, and Mechanic's pair census) and the CORRECTED
marked spectrum `Q_J(29) = 55, 65, 68, 71, 71, 71` of round-24 verdict 12c.
-/

import Machine29D2
import Machine29D3
import Machine29D4
import Machine29D5
import Machine29D6
import Machine29D7
import Machine29Q

namespace Machine29

/-- **The machine-29 census input** - the ONLY ingredient of the 29->31 rung
that is not kernel-checked.  Each `Ej` says the depth-`j` dictionary CONTAINS
every realised window of `j` consecutive gaps whose interiors all reach the
floor `10 = 2u''` of gear 31; `run` says no six consecutive gaps reach it. -/
structure Census29 : Prop where
  E2 : ∀ n, (g29 n, g29 (n + 1)) ∈ D2
  E3 : ∀ n, 10 ≤ g29 (n + 1) → (g29 n, g29 (n + 1), g29 (n + 2)) ∈ D3
  E4 : ∀ n, 10 ≤ g29 (n + 1) → 10 ≤ g29 (n + 2) →
    (g29 n, g29 (n + 1), g29 (n + 2), g29 (n + 3)) ∈ D4
  E5 : ∀ n, 10 ≤ g29 (n + 1) → 10 ≤ g29 (n + 2) → 10 ≤ g29 (n + 3) →
    (g29 n, g29 (n + 1), g29 (n + 2), g29 (n + 3), g29 (n + 4)) ∈ D5
  E6 : ∀ n, 10 ≤ g29 (n + 1) → 10 ≤ g29 (n + 2) → 10 ≤ g29 (n + 3) →
    10 ≤ g29 (n + 4) →
    (g29 n, g29 (n + 1), g29 (n + 2), g29 (n + 3), g29 (n + 4),
      g29 (n + 5)) ∈ D6
  E7 : ∀ n, 10 ≤ g29 (n + 1) → 10 ≤ g29 (n + 2) → 10 ≤ g29 (n + 3) →
    10 ≤ g29 (n + 4) → 10 ≤ g29 (n + 5) →
    (g29 n, g29 (n + 1), g29 (n + 2), g29 (n + 3), g29 (n + 4),
      g29 (n + 5), g29 (n + 6)) ∈ D7
  run : ∀ n, ¬ (10 ≤ g29 (n + 1) ∧ 10 ≤ g29 (n + 2) ∧ 10 ≤ g29 (n + 3) ∧
    10 ≤ g29 (n + 4) ∧ 10 ≤ g29 (n + 5) ∧ 10 ≤ g29 (n + 6))

/-! ## Reading the window sums off the dictionaries -/

theorem ws2 (a : ℕ) : Spectrum.windowSum g29 a 2 = g29 a + g29 (a + 1) := by
  simp [Spectrum.windowSum, Finset.sum_range_succ]

theorem ws3 (a : ℕ) :
    Spectrum.windowSum g29 a 3 = g29 a + g29 (a + 1) + g29 (a + 2) := by
  simp [Spectrum.windowSum, Finset.sum_range_succ]

theorem ws4 (a : ℕ) : Spectrum.windowSum g29 a 4 =
    g29 a + g29 (a + 1) + g29 (a + 2) + g29 (a + 3) := by
  simp [Spectrum.windowSum, Finset.sum_range_succ]

theorem ws5 (a : ℕ) : Spectrum.windowSum g29 a 5 =
    g29 a + g29 (a + 1) + g29 (a + 2) + g29 (a + 3) + g29 (a + 4) := by
  simp [Spectrum.windowSum, Finset.sum_range_succ]

theorem ws6 (a : ℕ) : Spectrum.windowSum g29 a 6 =
    g29 a + g29 (a + 1) + g29 (a + 2) + g29 (a + 3) + g29 (a + 4)
      + g29 (a + 5) := by
  simp [Spectrum.windowSum, Finset.sum_range_succ]

theorem ws7 (a : ℕ) : Spectrum.windowSum g29 a 7 =
    g29 a + g29 (a + 1) + g29 (a + 2) + g29 (a + 3) + g29 (a + 4)
      + g29 (a + 5) + g29 (a + 6) := by
  simp [Spectrum.windowSum, Finset.sum_range_succ]

/-- **`F_2(29) <= 55`** from the pair dictionary: the literal integer
Constructor's `A_5(23)` survivor closure produces with no machine-29 scan,
here derived instead from machine 29's own realised pairs. -/
theorem spectrum29_two (h : Census29) : Spectrum.SpectrumBound g29 2 55 := by
  intro a
  have hm := List.all_eq_true.mp D2_ok _ (h.E2 a)
  simp only [Nat.ble_eq] at hm
  rw [ws2]
  exact hm

/-- `Q_3(29; 10) <= 65`. -/
theorem qual29_three (h : Census29) : Spectrum.QualBound g29 5 3 65 := by
  intro a hq
  have h1 : 10 ≤ g29 (a + 1) := hq 1 (by omega) (by omega)
  have hm := List.all_eq_true.mp D3_ok _ (h.E3 a h1)
  simp only [Nat.ble_eq] at hm
  rw [ws3]
  exact hm

/-- `Q_4(29; 10) <= 68`. -/
theorem qual29_four (h : Census29) : Spectrum.QualBound g29 5 4 68 := by
  intro a hq
  have h1 : 10 ≤ g29 (a + 1) := hq 1 (by omega) (by omega)
  have h2 : 10 ≤ g29 (a + 2) := hq 2 (by omega) (by omega)
  have hm := List.all_eq_true.mp D4_ok _ (h.E4 a h1 h2)
  simp only [Nat.ble_eq] at hm
  rw [ws4]
  exact hm

/-- `Q_5(29; 10) <= 71`. -/
theorem qual29_five (h : Census29) : Spectrum.QualBound g29 5 5 71 := by
  intro a hq
  have h1 : 10 ≤ g29 (a + 1) := hq 1 (by omega) (by omega)
  have h2 : 10 ≤ g29 (a + 2) := hq 2 (by omega) (by omega)
  have h3 : 10 ≤ g29 (a + 3) := hq 3 (by omega) (by omega)
  have hm := List.all_eq_true.mp D5_ok _ (h.E5 a h1 h2 h3)
  simp only [Nat.ble_eq] at hm
  rw [ws5]
  exact hm

/-- `Q_6(29; 10) <= 71`. -/
theorem qual29_six (h : Census29) : Spectrum.QualBound g29 5 6 71 := by
  intro a hq
  have h1 : 10 ≤ g29 (a + 1) := hq 1 (by omega) (by omega)
  have h2 : 10 ≤ g29 (a + 2) := hq 2 (by omega) (by omega)
  have h3 : 10 ≤ g29 (a + 3) := hq 3 (by omega) (by omega)
  have h4 : 10 ≤ g29 (a + 4) := hq 4 (by omega) (by omega)
  have hm := List.all_eq_true.mp D6_ok _ (h.E6 a h1 h2 h3 h4)
  simp only [Nat.ble_eq] at hm
  rw [ws6]
  exact hm

/-- `Q_7(29; 10) <= 71`. -/
theorem qual29_seven (h : Census29) : Spectrum.QualBound g29 5 7 71 := by
  intro a hq
  have h1 : 10 ≤ g29 (a + 1) := hq 1 (by omega) (by omega)
  have h2 : 10 ≤ g29 (a + 2) := hq 2 (by omega) (by omega)
  have h3 : 10 ≤ g29 (a + 3) := hq 3 (by omega) (by omega)
  have h4 : 10 ≤ g29 (a + 4) := hq 4 (by omega) (by omega)
  have h5 : 10 ≤ g29 (a + 5) := hq 5 (by omega) (by omega)
  have hm := List.all_eq_true.mp D7_ok _ (h.E7 a h1 h2 h3 h4 h5)
  simp only [Nat.ble_eq] at hm
  rw [ws7]
  exact hm

/-- **Every depth at once**: `Q_j(29; 10) <= 71` for every `j >= 3`.
Depths 3..7 come from the dictionaries; depth `j >= 8` is VACUOUS, because a
qualifying window of `j >= 8` gaps has six consecutive qualifying interiors,
which `Census29.run` forbids.  This is machine 29's `no_big_run`, and it is
the same shape that terminates the potential at every earlier machine. -/
theorem qual29_all (h : Census29) :
    ∀ j, 3 ≤ j → Spectrum.QualBound g29 5 j 71 := by
  intro j hj a hq
  by_cases h8 : 8 ≤ j
  · exact absurd
      ⟨hq 1 (by omega) (by omega), hq 2 (by omega) (by omega),
        hq 3 (by omega) (by omega), hq 4 (by omega) (by omega),
        hq 5 (by omega) (by omega), hq 6 (by omega) (by omega)⟩ (h.run a)
  · interval_cases j
    · exact le_trans (qual29_three h a hq) (by omega)
    · exact le_trans (qual29_four h a hq) (by omega)
    · exact qual29_five h a hq
    · exact qual29_six h a hq
    · exact qual29_seven h a hq

/-- **The criterion at 29->31**, as integers:
`max (F_2(29), max_j Q_j(29; 10)) = max (55, 71) = 71 <= 74 = F(29) + 31`,
margin 3 - the same margin the 23->29 rung had. -/
theorem criterion_29_31 : max 55 71 ≤ 43 + 31 := by decide

end Machine29
