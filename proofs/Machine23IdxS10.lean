/-
Machine 23 position-indexed chain scan (round 24), slice family e = 10
(all f < 19): 19 kernel checks of 5005 CRT tuples x 23 gear-23 phases
each.  See Machine23Idx.lean.
-/

import Machine23Idx

namespace Machine23

set_option maxRecDepth 40000 in
theorem is10_0 : qsliceIdx 10 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is10_1 : qsliceIdx 10 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is10_2 : qsliceIdx 10 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is10_3 : qsliceIdx 10 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is10_4 : qsliceIdx 10 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is10_5 : qsliceIdx 10 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is10_6 : qsliceIdx 10 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is10_7 : qsliceIdx 10 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is10_8 : qsliceIdx 10 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is10_9 : qsliceIdx 10 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is10_10 : qsliceIdx 10 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is10_11 : qsliceIdx 10 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is10_12 : qsliceIdx 10 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is10_13 : qsliceIdx 10 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is10_14 : qsliceIdx 10 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is10_15 : qsliceIdx 10 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is10_16 : qsliceIdx 10 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is10_17 : qsliceIdx 10 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is10_18 : qsliceIdx 10 18 = true := by decide +kernel

/-- All 19 slices at e = 10. -/
theorem iasm10 : ∀ f < 19, qsliceIdx 10 f = true := by
  intro f hf
  interval_cases f
  exacts [is10_0, is10_1, is10_2, is10_3, is10_4, is10_5, is10_6, is10_7, is10_8, is10_9, is10_10, is10_11, is10_12, is10_13, is10_14, is10_15, is10_16, is10_17, is10_18]

end Machine23