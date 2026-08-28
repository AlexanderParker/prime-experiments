/-
Machine 23 position-indexed chain scan (round 24), slice family e = 12
(all f < 19): 19 kernel checks of 5005 CRT tuples x 23 gear-23 phases
each.  See Machine23Idx.lean.
-/

import Machine23Idx

namespace Machine23

set_option maxRecDepth 40000 in
theorem is12_0 : qsliceIdx 12 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is12_1 : qsliceIdx 12 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is12_2 : qsliceIdx 12 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is12_3 : qsliceIdx 12 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is12_4 : qsliceIdx 12 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is12_5 : qsliceIdx 12 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is12_6 : qsliceIdx 12 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is12_7 : qsliceIdx 12 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is12_8 : qsliceIdx 12 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is12_9 : qsliceIdx 12 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is12_10 : qsliceIdx 12 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is12_11 : qsliceIdx 12 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is12_12 : qsliceIdx 12 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is12_13 : qsliceIdx 12 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is12_14 : qsliceIdx 12 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is12_15 : qsliceIdx 12 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is12_16 : qsliceIdx 12 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is12_17 : qsliceIdx 12 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem is12_18 : qsliceIdx 12 18 = true := by decide +kernel

/-- All 19 slices at e = 12. -/
theorem iasm12 : ∀ f < 19, qsliceIdx 12 f = true := by
  intro f hf
  interval_cases f
  exacts [is12_0, is12_1, is12_2, is12_3, is12_4, is12_5, is12_6, is12_7, is12_8, is12_9, is12_10, is12_11, is12_12, is12_13, is12_14, is12_15, is12_16, is12_17, is12_18]

end Machine23