/-
Machine 23 position-indexed chain scan (round 24), slice family e = 0
(all f < 19): 19 kernel checks of 5005 CRT tuples x 23 gear-23 phases
each.  See Machine23Idx.lean.
-/

import Machine23Idx

namespace Machine23

set_option maxRecDepth 40000 in
theorem is0_0 : qsliceIdx 0 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is0_1 : qsliceIdx 0 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is0_2 : qsliceIdx 0 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is0_3 : qsliceIdx 0 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is0_4 : qsliceIdx 0 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is0_5 : qsliceIdx 0 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is0_6 : qsliceIdx 0 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is0_7 : qsliceIdx 0 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is0_8 : qsliceIdx 0 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is0_9 : qsliceIdx 0 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is0_10 : qsliceIdx 0 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is0_11 : qsliceIdx 0 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is0_12 : qsliceIdx 0 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is0_13 : qsliceIdx 0 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is0_14 : qsliceIdx 0 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is0_15 : qsliceIdx 0 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is0_16 : qsliceIdx 0 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is0_17 : qsliceIdx 0 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is0_18 : qsliceIdx 0 18 = true := by decide +kernel

/-- All 19 slices at e = 0. -/
theorem iasm0 : ∀ f < 19, qsliceIdx 0 f = true := by
  intro f hf
  interval_cases f
  exacts [is0_0, is0_1, is0_2, is0_3, is0_4, is0_5, is0_6, is0_7, is0_8, is0_9, is0_10, is0_11, is0_12, is0_13, is0_14, is0_15, is0_16, is0_17, is0_18]

end Machine23