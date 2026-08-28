/-
Machine 23 position-indexed chain scan (round 24), slice family e = 5
(all f < 19): 19 kernel checks of 5005 CRT tuples x 23 gear-23 phases
each.  See Machine23Idx.lean.
-/

import Machine23Idx

namespace Machine23

set_option maxRecDepth 40000 in
theorem is5_0 : qsliceIdx 5 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is5_1 : qsliceIdx 5 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is5_2 : qsliceIdx 5 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is5_3 : qsliceIdx 5 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is5_4 : qsliceIdx 5 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is5_5 : qsliceIdx 5 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is5_6 : qsliceIdx 5 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is5_7 : qsliceIdx 5 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is5_8 : qsliceIdx 5 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is5_9 : qsliceIdx 5 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is5_10 : qsliceIdx 5 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is5_11 : qsliceIdx 5 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is5_12 : qsliceIdx 5 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is5_13 : qsliceIdx 5 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is5_14 : qsliceIdx 5 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is5_15 : qsliceIdx 5 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is5_16 : qsliceIdx 5 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is5_17 : qsliceIdx 5 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is5_18 : qsliceIdx 5 18 = true := by decide +kernel

/-- All 19 slices at e = 5. -/
theorem iasm5 : ∀ f < 19, qsliceIdx 5 f = true := by
  intro f hf
  interval_cases f
  exacts [is5_0, is5_1, is5_2, is5_3, is5_4, is5_5, is5_6, is5_7, is5_8, is5_9, is5_10, is5_11, is5_12, is5_13, is5_14, is5_15, is5_16, is5_17, is5_18]

end Machine23