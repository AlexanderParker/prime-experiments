/-
Machine 23 position-indexed chain scan (round 24), slice family e = 14
(all f < 19): 19 kernel checks of 5005 CRT tuples x 23 gear-23 phases
each.  See Machine23Idx.lean.
-/

import Machine23Idx

namespace Machine23

set_option maxRecDepth 40000 in
theorem is14_0 : qsliceIdx 14 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is14_1 : qsliceIdx 14 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is14_2 : qsliceIdx 14 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is14_3 : qsliceIdx 14 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is14_4 : qsliceIdx 14 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is14_5 : qsliceIdx 14 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is14_6 : qsliceIdx 14 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is14_7 : qsliceIdx 14 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is14_8 : qsliceIdx 14 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is14_9 : qsliceIdx 14 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is14_10 : qsliceIdx 14 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is14_11 : qsliceIdx 14 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is14_12 : qsliceIdx 14 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is14_13 : qsliceIdx 14 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is14_14 : qsliceIdx 14 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is14_15 : qsliceIdx 14 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is14_16 : qsliceIdx 14 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is14_17 : qsliceIdx 14 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is14_18 : qsliceIdx 14 18 = true := by decide +kernel

/-- All 19 slices at e = 14. -/
theorem iasm14 : ∀ f < 19, qsliceIdx 14 f = true := by
  intro f hf
  interval_cases f
  exacts [is14_0, is14_1, is14_2, is14_3, is14_4, is14_5, is14_6, is14_7, is14_8, is14_9, is14_10, is14_11, is14_12, is14_13, is14_14, is14_15, is14_16, is14_17, is14_18]

end Machine23