/-
Machine 23 position-indexed chain scan (round 24), slice family e = 2
(all f < 19): 19 kernel checks of 5005 CRT tuples x 23 gear-23 phases
each.  See Machine23Idx.lean.
-/

import Machine23Idx

namespace Machine23

set_option maxRecDepth 40000 in
theorem is2_0 : qsliceIdx 2 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is2_1 : qsliceIdx 2 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is2_2 : qsliceIdx 2 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is2_3 : qsliceIdx 2 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is2_4 : qsliceIdx 2 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is2_5 : qsliceIdx 2 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is2_6 : qsliceIdx 2 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is2_7 : qsliceIdx 2 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is2_8 : qsliceIdx 2 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is2_9 : qsliceIdx 2 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is2_10 : qsliceIdx 2 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is2_11 : qsliceIdx 2 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is2_12 : qsliceIdx 2 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is2_13 : qsliceIdx 2 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is2_14 : qsliceIdx 2 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is2_15 : qsliceIdx 2 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is2_16 : qsliceIdx 2 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is2_17 : qsliceIdx 2 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is2_18 : qsliceIdx 2 18 = true := by decide +kernel

/-- All 19 slices at e = 2. -/
theorem iasm2 : ∀ f < 19, qsliceIdx 2 f = true := by
  intro f hf
  interval_cases f
  exacts [is2_0, is2_1, is2_2, is2_3, is2_4, is2_5, is2_6, is2_7, is2_8, is2_9, is2_10, is2_11, is2_12, is2_13, is2_14, is2_15, is2_16, is2_17, is2_18]

end Machine23