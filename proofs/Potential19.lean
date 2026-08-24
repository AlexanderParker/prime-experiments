/-
THE FIRST EXHIBITED POTENTIAL: (D) at 19->23 from three one-step inequalities
(round 22).

`Potential.lean` proves that a potential `h` certifies (D) with no quantifier
over depth. This file EXHIBITS one, at the step where (D) is already known
(`Machine23.D_at_19_23`), so the certificate form is not just a definition:

    h19 i = the qualifying tail from i - `g19 i`, extended by the next gap for
            as long as the gaps keep meeting the floor 8, and at most three
            times because machine 19 has no four consecutive gaps `>= 8`.

The three clauses are then discharged by machine 19's kernel ladder alone:

* (C1) `g19 i <= h19 i`             - every branch of `h19` contains `g19 i`;
* (C2) `8 <= g19 i -> g19 i + h19 (i+1) <= h19 i` - EQUALITY in every branch;
       the deepest branch needs exactly `Machine19.no_big_run` (`Q_6 = 0`) to
       know the tail stops;
* (C3) `g19 i + h19 (i+1) <= 25 + 23` - the four cases are exactly the four
       rungs of the ladder `F_2, F_3, F_4, F_5 <= 31, 35, 38, 47`, all under
       the budget 48.

So the depth-indexed criterion `Q_j <= F + q' at every j` and the
depth-quantifier-free criterion "a potential exists" are BOTH satisfied at
19->23, and the second is a finite object one can write down. The construction
is generic: at any machine whose qualifying runs are bounded (which is what
`Q_J = 0` says), the tail function is a potential, and (C3)'s cases are exactly
the machine's spectrum ladder. That is the recipe a future rung can reuse -
what is NOT known is a potential valid at every machine at once (Constructor's
bounded-state certificates fail at 29->31).
-/

import Machine19Q
import Potential

namespace Potential19

open Machine19

/-! ## Window sums as explicit gap sums -/

theorem ws2 (a : ℕ) : Spectrum.windowSum g19 a 2 = g19 a + g19 (a + 1) := by
  simp [Spectrum.windowSum, Finset.sum_range_succ]

theorem ws3 (a : ℕ) :
    Spectrum.windowSum g19 a 3 = g19 a + g19 (a + 1) + g19 (a + 2) := by
  simp [Spectrum.windowSum, Finset.sum_range_succ]

theorem ws4 (a : ℕ) :
    Spectrum.windowSum g19 a 4
      = g19 a + g19 (a + 1) + g19 (a + 2) + g19 (a + 3) := by
  simp [Spectrum.windowSum, Finset.sum_range_succ]

theorem ws5 (a : ℕ) :
    Spectrum.windowSum g19 a 5
      = g19 a + g19 (a + 1) + g19 (a + 2) + g19 (a + 3) + g19 (a + 4) := by
  simp [Spectrum.windowSum, Finset.sum_range_succ]

/-! ## The potential -/

/-- **The qualifying tail of machine 19 at opening `i`** - the potential. It
extends while the gaps meet the floor `2u' = 8`, and never more than three
times (`no_big_run`). -/
def h19 (i : ℕ) : ℕ :=
  if 8 ≤ g19 i then
    (if 8 ≤ g19 (i + 1) then
      (if 8 ≤ g19 (i + 2) then g19 i + g19 (i + 1) + g19 (i + 2) + g19 (i + 3)
        else g19 i + g19 (i + 1) + g19 (i + 2))
      else g19 i + g19 (i + 1))
    else g19 i

/-- **(C1)**: the potential dominates the gap at its own state. -/
theorem h19_C1 (i : ℕ) : g19 i ≤ h19 i := by
  unfold h19
  split_ifs <;> omega

/-- **(C2)**: along a qualifying step the potential drops by exactly the gap.
The deepest branch is where `Q_6(19) = 0` does its work: with three floor gaps
in a row the fourth cannot also qualify, so the tail terminates. -/
theorem h19_C2 (i : ℕ) (hq : 8 ≤ g19 i) : g19 i + h19 (i + 1) ≤ h19 i := by
  have hnb := no_big_run i
  have e1 : i + 1 + 1 = i + 2 := by omega
  have e2 : i + 1 + 2 = i + 3 := by omega
  simp only [h19, e1, e2]
  split_ifs <;> omega

/-- **(C3)**: a flank plus the potential fits the tolerance budget. The four
cases are the four rungs `F_2, F_3, F_4, F_5 <= 31, 35, 38, 47` of machine
19's kernel-fed ladder, all at most `F + q' = 48`. -/
theorem h19_C3 (i : ℕ) : g19 i + h19 (i + 1) ≤ 25 + 23 := by
  have b2 : g19 i + g19 (i + 1) ≤ 31 := by
    have := spectrum_two i; rwa [ws2] at this
  have b3 : g19 i + g19 (i + 1) + g19 (i + 2) ≤ 35 := by
    have := spectrum_three i; rwa [ws3] at this
  have b4 : g19 i + g19 (i + 1) + g19 (i + 2) + g19 (i + 3) ≤ 38 := by
    have := spectrum_four i; rwa [ws4] at this
  have b5 : g19 i + g19 (i + 1) + g19 (i + 2) + g19 (i + 3) + g19 (i + 4) ≤ 47 := by
    have := spectrum_five i; rwa [ws5] at this
  have e1 : i + 1 + 1 = i + 2 := by omega
  have e2 : i + 1 + 2 = i + 3 := by omega
  have e3 : i + 1 + 3 = i + 4 := by omega
  simp only [h19, e1, e2, e3]
  split_ifs <;> omega

/-- **(D) at `alpha = 3` at machine 19, from the potential** - the same
conclusion as `Machine19.D_of_word`, but reached through three one-step
inequalities with NO quantifier over the word's length anywhere in the
hypotheses. -/
theorem D_of_word_potential {a l : ℕ} (hw : ∀ i < l, 8 ≤ g19 (a + 1 + i)) :
    g19 a + Spectrum.windowSum g19 (a + 1) l + g19 (a + l + 1) ≤ 25 + 23 := by
  refine Potential.merged_le_of_potential (u := 4) (h := h19) h19_C1 ?_ h19_C3 ?_
  · intro i hqi
    exact h19_C2 i (by omega)
  · intro i hi
    have := hw i hi
    omega

end Potential19
