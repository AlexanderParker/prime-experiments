/-
Machine 23 position-indexed chain scan (round 24), slice family e = 11
(all f < 19): 19 kernel checks of 5005 CRT tuples x 23 gear-23 phases
each.  See Machine23Idx.lean.
-/

import Machine23Idx

namespace Machine23

set_option maxRecDepth 40000 in
theorem is11_0 : qsliceIdx 11 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is11_1 : qsliceIdx 11 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is11_2 : qsliceIdx 11 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is11_3 : qsliceIdx 11 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is11_4 : qsliceIdx 11 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is11_5 : qsliceIdx 11 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is11_6 : qsliceIdx 11 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is11_7 : qsliceIdx 11 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is11_8 : qsliceIdx 11 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is11_9 : qsliceIdx 11 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is11_10 : qsliceIdx 11 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is11_11 : qsliceIdx 11 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is11_12 : qsliceIdx 11 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is11_13 : qsliceIdx 11 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is11_14 : qsliceIdx 11 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is11_15 : qsliceIdx 11 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is11_16 : qsliceIdx 11 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is11_17 : qsliceIdx 11 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is11_18 : qsliceIdx 11 18 = true := by decide +kernel

/-- All 19 slices at e = 11. -/
theorem iasm11 : ∀ f < 19, qsliceIdx 11 f = true := by
  intro f hf
  interval_cases f
  exacts [is11_0, is11_1, is11_2, is11_3, is11_4, is11_5, is11_6, is11_7, is11_8, is11_9, is11_10, is11_11, is11_12, is11_13, is11_14, is11_15, is11_16, is11_17, is11_18]

end Machine23