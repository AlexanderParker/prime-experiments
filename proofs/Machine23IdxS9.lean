/-
Machine 23 position-indexed chain scan (round 24), slice family e = 9
(all f < 19): 19 kernel checks of 5005 CRT tuples x 23 gear-23 phases
each.  See Machine23Idx.lean.
-/

import Machine23Idx

namespace Machine23

set_option maxRecDepth 40000 in
theorem is9_0 : qsliceIdx 9 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is9_1 : qsliceIdx 9 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is9_2 : qsliceIdx 9 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is9_3 : qsliceIdx 9 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is9_4 : qsliceIdx 9 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is9_5 : qsliceIdx 9 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is9_6 : qsliceIdx 9 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is9_7 : qsliceIdx 9 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is9_8 : qsliceIdx 9 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is9_9 : qsliceIdx 9 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is9_10 : qsliceIdx 9 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is9_11 : qsliceIdx 9 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is9_12 : qsliceIdx 9 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is9_13 : qsliceIdx 9 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is9_14 : qsliceIdx 9 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is9_15 : qsliceIdx 9 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is9_16 : qsliceIdx 9 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is9_17 : qsliceIdx 9 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is9_18 : qsliceIdx 9 18 = true := by decide +kernel

/-- All 19 slices at e = 9. -/
theorem iasm9 : ∀ f < 19, qsliceIdx 9 f = true := by
  intro f hf
  interval_cases f
  exacts [is9_0, is9_1, is9_2, is9_3, is9_4, is9_5, is9_6, is9_7, is9_8, is9_9, is9_10, is9_11, is9_12, is9_13, is9_14, is9_15, is9_16, is9_17, is9_18]

end Machine23