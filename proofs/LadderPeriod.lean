/-
THE LADDER'S TOP TWO RUNGS ON THE SHRUNKEN HYPOTHESIS (round 26).

Round-25 verdict 21: "anyone quoting a rung must quote the hypothesis".
Round 26 does not remove the hypothesis - it SHRINKS it.  `Census29P` and
`Census31P` say what `Census29` and `Census31` say, but only for the indices
of ONE PERIOD, which is exactly the finite object `research/qual_dict.py` and
`qual_dict_gate.py` verify.  The step from one period to every index used to
be an unstated assumption; it is now `Periodic.index_reduce`.

    D_29_31_period (h : Census29P) (n) : g31 n <= 43 + 31
    D_31_37_period (h : Census31P) (n) : g37 n <= 58 + 37

Nothing else about the rungs changes: the same dictionaries, the same
`decide +kernel` list facts, the same merge law.
-/

import Machine29Cen
import Machine31Cen
import Machine31
import Machine37

namespace LadderPeriod

/-- **The sixth rung, on the one-period census.** -/
theorem D_29_31_period (h : Machine29.Census29P) (n : ℕ) :
    Machine31.g31 n ≤ 43 + 31 :=
  Machine31.D_29_31 (Machine29.census29_of_period h) n

/-- R39's own form at 29->31, on the one-period census. -/
theorem g31_le_of_period (h : Machine29.Census29P) (n : ℕ) :
    Machine31.g31 n ≤ 71 :=
  Machine31.g31_le_of_census (Machine29.census29_of_period h) n

/-- **The seventh rung, on the one-period census.** -/
theorem D_31_37_period (h : Machine31.Census31P) (n : ℕ) :
    Machine37.g37 n ≤ 58 + 37 :=
  Machine37.D_31_37 (Machine31.census31_of_period h) n

/-- R39's own form at 31->37, on the one-period census. -/
theorem g37_le_of_period (h : Machine31.Census31P) (n : ℕ) :
    Machine37.g37 n ≤ 91 :=
  Machine37.g37_le_of_census (Machine31.census31_of_period h) n

end LadderPeriod
