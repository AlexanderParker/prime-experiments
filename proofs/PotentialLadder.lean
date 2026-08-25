/-
THE POTENTIAL FORM OF (D) AT EVERY SCANNED RUNG (round 23).

Round 22 exhibited the first potential (`Potential19.h19`, at 19->23) and
recorded the recipe: at a machine whose qualifying runs are bounded - which is
exactly what `Q_J = 0` says, and every machine scanned so far has such a `J` -
THE QUALIFYING TAIL, unfolded to depth `J - 2`, IS a potential; (C2)'s deepest
branch is the machine's `no_big_run` refutation and (C3)'s cases are the
machine's own spectrum ladder. This file runs that recipe at the three
remaining scanned rungs, so the DEPTH-QUANTIFIER-FREE form of (D) is now
available at every rung of the ladder, not just the top one:

    rung      potential   floor 2u'   tail depth   budget F + q'
    11 -> 13  h11             4           4          7 + 13 = 20
    13 -> 17  h13             6           3         11 + 17 = 28
    17 -> 19  h17             6           5         18 + 19 = 37
    19 -> 23  h19 (r22)       8           4         25 + 23 = 48

Each `D_of_word_*` below re-proves that rung's (D) statement through
`Potential.merged_le_of_potential`, whose hypotheses contain NO quantifier
over the word length `l` while its conclusion holds for every `l`.

The tail depths are the interesting column: they are `J - 2` for the machine's
own `Q_J = 0`, and they do NOT increase with the machine (4, 3, 5, 4). What is
still not known - and what Constructor's 29->31 negative is about - is a
potential valid at every machine at once; these are four separate finite
objects, one per machine.
-/

import Machine11
import Machine13Q
import Machine17Q
import Potential

namespace PotentialLadder

/-! ## Window sums as explicit gap sums -/

theorem ws2 {g : ℕ → ℕ} (a : ℕ) : Spectrum.windowSum g a 2 = g a + g (a + 1) := by
  simp [Spectrum.windowSum, Finset.sum_range_succ]

theorem ws3 {g : ℕ → ℕ} (a : ℕ) :
    Spectrum.windowSum g a 3 = g a + g (a + 1) + g (a + 2) := by
  simp [Spectrum.windowSum, Finset.sum_range_succ]

theorem ws4 {g : ℕ → ℕ} (a : ℕ) :
    Spectrum.windowSum g a 4 = g a + g (a + 1) + g (a + 2) + g (a + 3) := by
  simp [Spectrum.windowSum, Finset.sum_range_succ]

theorem ws5 {g : ℕ → ℕ} (a : ℕ) :
    Spectrum.windowSum g a 5
      = g a + g (a + 1) + g (a + 2) + g (a + 3) + g (a + 4) := by
  simp [Spectrum.windowSum, Finset.sum_range_succ]

theorem ws6 {g : ℕ → ℕ} (a : ℕ) :
    Spectrum.windowSum g a 6
      = g a + g (a + 1) + g (a + 2) + g (a + 3) + g (a + 4) + g (a + 5) := by
  simp [Spectrum.windowSum, Finset.sum_range_succ]

/-! ## Rung 11 -> 13: floor 4, budget 20 -/

open Machine11 in
/-- The qualifying tail of machine 11 (floor `2u' = 4`, gear 13). It extends
at most three times: `Machine11.no_big_run` forbids four consecutive gaps
`>= 4`. -/
def h11 (i : ℕ) : ℕ :=
  if 4 ≤ g11 i then
    (if 4 ≤ g11 (i + 1) then
      (if 4 ≤ g11 (i + 2) then g11 i + g11 (i + 1) + g11 (i + 2) + g11 (i + 3)
        else g11 i + g11 (i + 1) + g11 (i + 2))
      else g11 i + g11 (i + 1))
    else g11 i

theorem h11_C1 (i : ℕ) : Machine11.g11 i ≤ h11 i := by
  unfold h11; split_ifs <;> omega

theorem h11_C2 (i : ℕ) (hq : 4 ≤ Machine11.g11 i) :
    Machine11.g11 i + h11 (i + 1) ≤ h11 i := by
  have hnb := Machine11.no_big_run i
  have e1 : i + 1 + 1 = i + 2 := by omega
  have e2 : i + 1 + 2 = i + 3 := by omega
  simp only [h11, e1, e2]
  split_ifs <;> omega

theorem h11_C3 (i : ℕ) : Machine11.g11 i + h11 (i + 1) ≤ 7 + 13 := by
  have b2 : Machine11.g11 i + Machine11.g11 (i + 1) ≤ 11 := by
    have := Machine11.spectrum_two i; rwa [ws2] at this
  have b3 : Machine11.g11 i + Machine11.g11 (i + 1) + Machine11.g11 (i + 2) ≤ 16 := by
    have := Machine11.spectrum_three i; rwa [ws3] at this
  have b4 : Machine11.g11 i + Machine11.g11 (i + 1) + Machine11.g11 (i + 2)
      + Machine11.g11 (i + 3) ≤ 18 := by
    have := Machine11.spectrum_four i; rwa [ws4] at this
  have b5 : (4 ≤ Machine11.g11 (i + 1) → 4 ≤ Machine11.g11 (i + 2) →
      4 ≤ Machine11.g11 (i + 3) →
      Machine11.g11 i + Machine11.g11 (i + 1) + Machine11.g11 (i + 2)
        + Machine11.g11 (i + 3) + Machine11.g11 (i + 4) ≤ 20) := by
    intro k1 k2 k3
    have h := (Machine11.chain_facts i).2.2.2.2.1 ⟨k1, k2, k3⟩
    have hw := Machine11.windowSum_g11 i 5
    rw [ws5] at hw
    omega
  have e1 : i + 1 + 1 = i + 2 := by omega
  have e2 : i + 1 + 2 = i + 3 := by omega
  have e3 : i + 1 + 3 = i + 4 := by omega
  simp only [h11, e1, e2, e3]
  split_ifs <;> omega

/-- **(D) at the 11->13 step from a potential** - no depth quantifier. -/
theorem D_of_word_11 {a l : ℕ} (hw : ∀ i < l, 4 ≤ Machine11.g11 (a + 1 + i)) :
    Machine11.g11 a + Spectrum.windowSum Machine11.g11 (a + 1) l
      + Machine11.g11 (a + l + 1) ≤ 7 + 13 := by
  refine Potential.merged_le_of_potential (u := 2) (h := h11) h11_C1 ?_ h11_C3 ?_
  · intro i hqi; exact h11_C2 i (by omega)
  · intro i hi; have := hw i hi; omega

/-! ## Rung 13 -> 17: floor 6, budget 28 -/

open Machine13 in
/-- The qualifying tail of machine 13 (floor `2u' = 6`, gear 17). It extends
at most twice: `Machine13.no_big_run` forbids three consecutive gaps `>= 6`. -/
def h13 (i : ℕ) : ℕ :=
  if 6 ≤ g13 i then
    (if 6 ≤ g13 (i + 1) then g13 i + g13 (i + 1) + g13 (i + 2)
      else g13 i + g13 (i + 1))
    else g13 i

theorem h13_C1 (i : ℕ) : Machine13.g13 i ≤ h13 i := by
  unfold h13; split_ifs <;> omega

theorem h13_C2 (i : ℕ) (hq : 6 ≤ Machine13.g13 i) :
    Machine13.g13 i + h13 (i + 1) ≤ h13 i := by
  have hnb := Machine13.no_big_run i
  have e1 : i + 1 + 1 = i + 2 := by omega
  simp only [h13, e1]
  split_ifs <;> omega

theorem h13_C3 (i : ℕ) : Machine13.g13 i + h13 (i + 1) ≤ 11 + 17 := by
  have b2 : Machine13.g13 i + Machine13.g13 (i + 1) ≤ 16 := by
    have := Machine13.spectrum_two i; rwa [ws2] at this
  have b3 : Machine13.g13 i + Machine13.g13 (i + 1) + Machine13.g13 (i + 2) ≤ 23 := by
    have := Machine13.spectrum_three i; rwa [ws3] at this
  have b4 : Machine13.g13 i + Machine13.g13 (i + 1) + Machine13.g13 (i + 2)
      + Machine13.g13 (i + 3) ≤ 26 := by
    have := Machine13.spectrum_four i; rwa [ws4] at this
  have e1 : i + 1 + 1 = i + 2 := by omega
  have e2 : i + 1 + 2 = i + 3 := by omega
  simp only [h13, e1, e2]
  split_ifs <;> omega

/-- **(D) at the 13->17 step from a potential** - no depth quantifier. -/
theorem D_of_word_13 {a l : ℕ} (hw : ∀ i < l, 6 ≤ Machine13.g13 (a + 1 + i)) :
    Machine13.g13 a + Spectrum.windowSum Machine13.g13 (a + 1) l
      + Machine13.g13 (a + l + 1) ≤ 11 + 17 := by
  refine Potential.merged_le_of_potential (u := 3) (h := h13) h13_C1 ?_ h13_C3 ?_
  · intro i hqi; exact h13_C2 i (by omega)
  · intro i hi; have := hw i hi; omega

/-! ## Rung 17 -> 19: floor 6, budget 37 -/

open Machine17 in
/-- The qualifying tail of machine 17 (floor `2u' = 6`, gear 19). It extends
at most four times: `Machine17.no_big_run` forbids five consecutive gaps
`>= 6`. -/
def h17 (i : ℕ) : ℕ :=
  if 6 ≤ g17 i then
    (if 6 ≤ g17 (i + 1) then
      (if 6 ≤ g17 (i + 2) then
        (if 6 ≤ g17 (i + 3) then
          g17 i + g17 (i + 1) + g17 (i + 2) + g17 (i + 3) + g17 (i + 4)
          else g17 i + g17 (i + 1) + g17 (i + 2) + g17 (i + 3))
        else g17 i + g17 (i + 1) + g17 (i + 2))
      else g17 i + g17 (i + 1))
    else g17 i

theorem h17_C1 (i : ℕ) : Machine17.g17 i ≤ h17 i := by
  unfold h17; split_ifs <;> omega

theorem h17_C2 (i : ℕ) (hq : 6 ≤ Machine17.g17 i) :
    Machine17.g17 i + h17 (i + 1) ≤ h17 i := by
  have hnb := Machine17.no_big_run i
  have e1 : i + 1 + 1 = i + 2 := by omega
  have e2 : i + 1 + 2 = i + 3 := by omega
  have e3 : i + 1 + 3 = i + 4 := by omega
  simp only [h17, e1, e2, e3]
  split_ifs <;> omega

set_option maxHeartbeats 1000000 in
theorem h17_C3 (i : ℕ) : Machine17.g17 i + h17 (i + 1) ≤ 18 + 19 := by
  have b2 : Machine17.g17 i + Machine17.g17 (i + 1) ≤ 25 := by
    have := Machine17.spectrum_two i; rwa [ws2] at this
  have b3 : Machine17.g17 i + Machine17.g17 (i + 1) + Machine17.g17 (i + 2) ≤ 28 := by
    have := Machine17.spectrum_three i; rwa [ws3] at this
  have b4 : Machine17.g17 i + Machine17.g17 (i + 1) + Machine17.g17 (i + 2)
      + Machine17.g17 (i + 3) ≤ 33 := by
    have := Machine17.spectrum_four i; rwa [ws4] at this
  have b5 : Machine17.g17 i + Machine17.g17 (i + 1) + Machine17.g17 (i + 2)
      + Machine17.g17 (i + 3) + Machine17.g17 (i + 4) ≤ 35 := by
    have := Machine17.spectrum_five i; rwa [ws5] at this
  have b6 : (6 ≤ Machine17.g17 (i + 1) → 6 ≤ Machine17.g17 (i + 2) →
      6 ≤ Machine17.g17 (i + 3) → 6 ≤ Machine17.g17 (i + 4) →
      Machine17.g17 i + Machine17.g17 (i + 1) + Machine17.g17 (i + 2)
        + Machine17.g17 (i + 3) + Machine17.g17 (i + 4)
        + Machine17.g17 (i + 5) ≤ 34) := by
    intro k1 k2 k3 k4
    have h := (Machine17.chain_facts i).2.2.2.2.1 ⟨k1, k2, k3, k4⟩
    have hw := Machine17.windowSum_g17 i 6
    rw [ws6] at hw
    omega
  have e1 : i + 1 + 1 = i + 2 := by omega
  have e2 : i + 1 + 2 = i + 3 := by omega
  have e3 : i + 1 + 3 = i + 4 := by omega
  have e4 : i + 1 + 4 = i + 5 := by omega
  simp only [h17, e1, e2, e3, e4]
  split_ifs <;> omega

/-- **(D) at the 17->19 step from a potential** - no depth quantifier. -/
theorem D_of_word_17 {a l : ℕ} (hw : ∀ i < l, 6 ≤ Machine17.g17 (a + 1 + i)) :
    Machine17.g17 a + Spectrum.windowSum Machine17.g17 (a + 1) l
      + Machine17.g17 (a + l + 1) ≤ 18 + 19 := by
  refine Potential.merged_le_of_potential (u := 3) (h := h17) h17_C1 ?_ h17_C3 ?_
  · intro i hqi; exact h17_C2 i (by omega)
  · intro i hi; have := hw i hi; omega

/-- **THE POTENTIAL LADDER**: (D) at `alpha = 3` at all four scanned rungs,
each through a finite potential with no quantifier over depth in its
hypotheses. -/
theorem potential_ladder :
    (∀ a l, (∀ i < l, 4 ≤ Machine11.g11 (a + 1 + i)) →
      Machine11.g11 a + Spectrum.windowSum Machine11.g11 (a + 1) l
        + Machine11.g11 (a + l + 1) ≤ 7 + 13) ∧
    (∀ a l, (∀ i < l, 6 ≤ Machine13.g13 (a + 1 + i)) →
      Machine13.g13 a + Spectrum.windowSum Machine13.g13 (a + 1) l
        + Machine13.g13 (a + l + 1) ≤ 11 + 17) ∧
    (∀ a l, (∀ i < l, 6 ≤ Machine17.g17 (a + 1 + i)) →
      Machine17.g17 a + Spectrum.windowSum Machine17.g17 (a + 1) l
        + Machine17.g17 (a + l + 1) ≤ 18 + 19) :=
  ⟨fun _ _ hw => D_of_word_11 hw, fun _ _ hw => D_of_word_13 hw,
    fun _ _ hw => D_of_word_17 hw⟩

end PotentialLadder
