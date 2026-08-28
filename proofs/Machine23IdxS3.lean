/-
Machine 23 position-indexed chain scan (round 24), slice family e = 3
(all f < 19): 19 kernel checks of 5005 CRT tuples x 23 gear-23 phases
each.  See Machine23Idx.lean.
-/

import Machine23Idx

namespace Machine23

set_option maxRecDepth 40000 in
theorem is3_0 : qsliceIdx 3 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is3_1 : qsliceIdx 3 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is3_2 : qsliceIdx 3 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is3_3 : qsliceIdx 3 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is3_4 : qsliceIdx 3 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is3_5 : qsliceIdx 3 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is3_6 : qsliceIdx 3 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is3_7 : qsliceIdx 3 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is3_8 : qsliceIdx 3 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is3_9 : qsliceIdx 3 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is3_10 : qsliceIdx 3 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is3_11 : qsliceIdx 3 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is3_12 : qsliceIdx 3 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is3_13 : qsliceIdx 3 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is3_14 : qsliceIdx 3 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is3_15 : qsliceIdx 3 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is3_16 : qsliceIdx 3 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is3_17 : qsliceIdx 3 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is3_18 : qsliceIdx 3 18 = true := by decide +kernel

/-- All 19 slices at e = 3. -/
theorem iasm3 : ∀ f < 19, qsliceIdx 3 f = true := by
  intro f hf
  interval_cases f
  exacts [is3_0, is3_1, is3_2, is3_3, is3_4, is3_5, is3_6, is3_7, is3_8, is3_9, is3_10, is3_11, is3_12, is3_13, is3_14, is3_15, is3_16, is3_17, is3_18]

end Machine23