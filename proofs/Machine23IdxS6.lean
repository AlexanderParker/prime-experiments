/-
Machine 23 position-indexed chain scan (round 24), slice family e = 6
(all f < 19): 19 kernel checks of 5005 CRT tuples x 23 gear-23 phases
each.  See Machine23Idx.lean.
-/

import Machine23Idx

namespace Machine23

set_option maxRecDepth 40000 in
theorem is6_0 : qsliceIdx 6 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is6_1 : qsliceIdx 6 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is6_2 : qsliceIdx 6 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is6_3 : qsliceIdx 6 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is6_4 : qsliceIdx 6 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is6_5 : qsliceIdx 6 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is6_6 : qsliceIdx 6 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is6_7 : qsliceIdx 6 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is6_8 : qsliceIdx 6 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is6_9 : qsliceIdx 6 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is6_10 : qsliceIdx 6 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is6_11 : qsliceIdx 6 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is6_12 : qsliceIdx 6 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is6_13 : qsliceIdx 6 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is6_14 : qsliceIdx 6 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is6_15 : qsliceIdx 6 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is6_16 : qsliceIdx 6 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is6_17 : qsliceIdx 6 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is6_18 : qsliceIdx 6 18 = true := by decide +kernel

/-- All 19 slices at e = 6. -/
theorem iasm6 : ∀ f < 19, qsliceIdx 6 f = true := by
  intro f hf
  interval_cases f
  exacts [is6_0, is6_1, is6_2, is6_3, is6_4, is6_5, is6_6, is6_7, is6_8, is6_9, is6_10, is6_11, is6_12, is6_13, is6_14, is6_15, is6_16, is6_17, is6_18]

end Machine23