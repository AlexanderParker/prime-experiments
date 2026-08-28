/-
Machine 23 position-indexed chain scan (round 24), slice family e = 15
(all f < 19): 19 kernel checks of 5005 CRT tuples x 23 gear-23 phases
each.  See Machine23Idx.lean.
-/

import Machine23Idx

namespace Machine23

set_option maxRecDepth 40000 in
theorem is15_0 : qsliceIdx 15 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is15_1 : qsliceIdx 15 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is15_2 : qsliceIdx 15 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is15_3 : qsliceIdx 15 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is15_4 : qsliceIdx 15 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is15_5 : qsliceIdx 15 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is15_6 : qsliceIdx 15 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is15_7 : qsliceIdx 15 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is15_8 : qsliceIdx 15 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is15_9 : qsliceIdx 15 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is15_10 : qsliceIdx 15 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is15_11 : qsliceIdx 15 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is15_12 : qsliceIdx 15 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is15_13 : qsliceIdx 15 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is15_14 : qsliceIdx 15 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is15_15 : qsliceIdx 15 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is15_16 : qsliceIdx 15 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is15_17 : qsliceIdx 15 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is15_18 : qsliceIdx 15 18 = true := by decide +kernel

/-- All 19 slices at e = 15. -/
theorem iasm15 : ∀ f < 19, qsliceIdx 15 f = true := by
  intro f hf
  interval_cases f
  exacts [is15_0, is15_1, is15_2, is15_3, is15_4, is15_5, is15_6, is15_7, is15_8, is15_9, is15_10, is15_11, is15_12, is15_13, is15_14, is15_15, is15_16, is15_17, is15_18]

end Machine23