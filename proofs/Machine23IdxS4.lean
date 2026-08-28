/-
Machine 23 position-indexed chain scan (round 24), slice family e = 4
(all f < 19): 19 kernel checks of 5005 CRT tuples x 23 gear-23 phases
each.  See Machine23Idx.lean.
-/

import Machine23Idx

namespace Machine23

set_option maxRecDepth 40000 in
theorem is4_0 : qsliceIdx 4 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is4_1 : qsliceIdx 4 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is4_2 : qsliceIdx 4 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is4_3 : qsliceIdx 4 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is4_4 : qsliceIdx 4 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is4_5 : qsliceIdx 4 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is4_6 : qsliceIdx 4 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is4_7 : qsliceIdx 4 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is4_8 : qsliceIdx 4 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is4_9 : qsliceIdx 4 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is4_10 : qsliceIdx 4 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is4_11 : qsliceIdx 4 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is4_12 : qsliceIdx 4 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is4_13 : qsliceIdx 4 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is4_14 : qsliceIdx 4 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is4_15 : qsliceIdx 4 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is4_16 : qsliceIdx 4 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is4_17 : qsliceIdx 4 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is4_18 : qsliceIdx 4 18 = true := by decide +kernel

/-- All 19 slices at e = 4. -/
theorem iasm4 : ∀ f < 19, qsliceIdx 4 f = true := by
  intro f hf
  interval_cases f
  exacts [is4_0, is4_1, is4_2, is4_3, is4_4, is4_5, is4_6, is4_7, is4_8, is4_9, is4_10, is4_11, is4_12, is4_13, is4_14, is4_15, is4_16, is4_17, is4_18]

end Machine23