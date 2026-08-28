/-
Machine 23 position-indexed chain scan (round 24), slice family e = 1
(all f < 19): 19 kernel checks of 5005 CRT tuples x 23 gear-23 phases
each.  See Machine23Idx.lean.
-/

import Machine23Idx

namespace Machine23

set_option maxRecDepth 40000 in
theorem is1_0 : qsliceIdx 1 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is1_1 : qsliceIdx 1 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is1_2 : qsliceIdx 1 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is1_3 : qsliceIdx 1 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is1_4 : qsliceIdx 1 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is1_5 : qsliceIdx 1 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is1_6 : qsliceIdx 1 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is1_7 : qsliceIdx 1 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is1_8 : qsliceIdx 1 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is1_9 : qsliceIdx 1 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is1_10 : qsliceIdx 1 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is1_11 : qsliceIdx 1 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is1_12 : qsliceIdx 1 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is1_13 : qsliceIdx 1 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is1_14 : qsliceIdx 1 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is1_15 : qsliceIdx 1 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is1_16 : qsliceIdx 1 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is1_17 : qsliceIdx 1 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is1_18 : qsliceIdx 1 18 = true := by decide +kernel

/-- All 19 slices at e = 1. -/
theorem iasm1 : ∀ f < 19, qsliceIdx 1 f = true := by
  intro f hf
  interval_cases f
  exacts [is1_0, is1_1, is1_2, is1_3, is1_4, is1_5, is1_6, is1_7, is1_8, is1_9, is1_10, is1_11, is1_12, is1_13, is1_14, is1_15, is1_16, is1_17, is1_18]

end Machine23