/-
Machine 23 position-indexed chain scan (round 24), slice family e = 16
(all f < 19): 19 kernel checks of 5005 CRT tuples x 23 gear-23 phases
each.  See Machine23Idx.lean.
-/

import Machine23Idx

namespace Machine23

set_option maxRecDepth 40000 in
theorem is16_0 : qsliceIdx 16 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is16_1 : qsliceIdx 16 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is16_2 : qsliceIdx 16 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is16_3 : qsliceIdx 16 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is16_4 : qsliceIdx 16 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is16_5 : qsliceIdx 16 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is16_6 : qsliceIdx 16 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is16_7 : qsliceIdx 16 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is16_8 : qsliceIdx 16 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is16_9 : qsliceIdx 16 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is16_10 : qsliceIdx 16 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is16_11 : qsliceIdx 16 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is16_12 : qsliceIdx 16 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is16_13 : qsliceIdx 16 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is16_14 : qsliceIdx 16 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is16_15 : qsliceIdx 16 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is16_16 : qsliceIdx 16 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is16_17 : qsliceIdx 16 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is16_18 : qsliceIdx 16 18 = true := by decide +kernel

/-- All 19 slices at e = 16. -/
theorem iasm16 : ∀ f < 19, qsliceIdx 16 f = true := by
  intro f hf
  interval_cases f
  exacts [is16_0, is16_1, is16_2, is16_3, is16_4, is16_5, is16_6, is16_7, is16_8, is16_9, is16_10, is16_11, is16_12, is16_13, is16_14, is16_15, is16_16, is16_17, is16_18]

end Machine23