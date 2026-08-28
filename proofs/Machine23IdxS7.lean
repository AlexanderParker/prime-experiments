/-
Machine 23 position-indexed chain scan (round 24), slice family e = 7
(all f < 19): 19 kernel checks of 5005 CRT tuples x 23 gear-23 phases
each.  See Machine23Idx.lean.
-/

import Machine23Idx

namespace Machine23

set_option maxRecDepth 40000 in
theorem is7_0 : qsliceIdx 7 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is7_1 : qsliceIdx 7 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is7_2 : qsliceIdx 7 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is7_3 : qsliceIdx 7 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is7_4 : qsliceIdx 7 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is7_5 : qsliceIdx 7 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is7_6 : qsliceIdx 7 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is7_7 : qsliceIdx 7 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is7_8 : qsliceIdx 7 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is7_9 : qsliceIdx 7 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is7_10 : qsliceIdx 7 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is7_11 : qsliceIdx 7 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is7_12 : qsliceIdx 7 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is7_13 : qsliceIdx 7 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is7_14 : qsliceIdx 7 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is7_15 : qsliceIdx 7 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is7_16 : qsliceIdx 7 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is7_17 : qsliceIdx 7 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is7_18 : qsliceIdx 7 18 = true := by decide +kernel

/-- All 19 slices at e = 7. -/
theorem iasm7 : ∀ f < 19, qsliceIdx 7 f = true := by
  intro f hf
  interval_cases f
  exacts [is7_0, is7_1, is7_2, is7_3, is7_4, is7_5, is7_6, is7_7, is7_8, is7_9, is7_10, is7_11, is7_12, is7_13, is7_14, is7_15, is7_16, is7_17, is7_18]

end Machine23