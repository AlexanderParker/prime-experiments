/-
Machine 23 position-indexed chain scan (round 24), slice family e = 13
(all f < 19): 19 kernel checks of 5005 CRT tuples x 23 gear-23 phases
each.  See Machine23Idx.lean.
-/

import Machine23Idx

namespace Machine23

set_option maxRecDepth 40000 in
theorem is13_0 : qsliceIdx 13 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is13_1 : qsliceIdx 13 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is13_2 : qsliceIdx 13 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is13_3 : qsliceIdx 13 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is13_4 : qsliceIdx 13 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is13_5 : qsliceIdx 13 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is13_6 : qsliceIdx 13 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is13_7 : qsliceIdx 13 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is13_8 : qsliceIdx 13 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is13_9 : qsliceIdx 13 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is13_10 : qsliceIdx 13 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is13_11 : qsliceIdx 13 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is13_12 : qsliceIdx 13 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is13_13 : qsliceIdx 13 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is13_14 : qsliceIdx 13 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is13_15 : qsliceIdx 13 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is13_16 : qsliceIdx 13 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is13_17 : qsliceIdx 13 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is13_18 : qsliceIdx 13 18 = true := by decide +kernel

/-- All 19 slices at e = 13. -/
theorem iasm13 : ∀ f < 19, qsliceIdx 13 f = true := by
  intro f hf
  interval_cases f
  exacts [is13_0, is13_1, is13_2, is13_3, is13_4, is13_5, is13_6, is13_7, is13_8, is13_9, is13_10, is13_11, is13_12, is13_13, is13_14, is13_15, is13_16, is13_17, is13_18]

end Machine23