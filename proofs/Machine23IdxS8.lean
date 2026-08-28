/-
Machine 23 position-indexed chain scan (round 24), slice family e = 8
(all f < 19): 19 kernel checks of 5005 CRT tuples x 23 gear-23 phases
each.  See Machine23Idx.lean.
-/

import Machine23Idx

namespace Machine23

set_option maxRecDepth 40000 in
theorem is8_0 : qsliceIdx 8 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is8_1 : qsliceIdx 8 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is8_2 : qsliceIdx 8 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is8_3 : qsliceIdx 8 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is8_4 : qsliceIdx 8 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is8_5 : qsliceIdx 8 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is8_6 : qsliceIdx 8 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is8_7 : qsliceIdx 8 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is8_8 : qsliceIdx 8 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is8_9 : qsliceIdx 8 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is8_10 : qsliceIdx 8 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is8_11 : qsliceIdx 8 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is8_12 : qsliceIdx 8 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is8_13 : qsliceIdx 8 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is8_14 : qsliceIdx 8 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is8_15 : qsliceIdx 8 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is8_16 : qsliceIdx 8 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is8_17 : qsliceIdx 8 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is8_18 : qsliceIdx 8 18 = true := by decide +kernel

/-- All 19 slices at e = 8. -/
theorem iasm8 : ∀ f < 19, qsliceIdx 8 f = true := by
  intro f hf
  interval_cases f
  exacts [is8_0, is8_1, is8_2, is8_3, is8_4, is8_5, is8_6, is8_7, is8_8, is8_9, is8_10, is8_11, is8_12, is8_13, is8_14, is8_15, is8_16, is8_17, is8_18]

end Machine23