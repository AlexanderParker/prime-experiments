/-
Machine 19 period scan, slice family e = 15 (all f < 19): 19 kernel
checks of 5005 CRT tuples each. See Machine19Core.lean.
-/

import Machine19Core

namespace Machine19

set_option maxRecDepth 40000 in
theorem s15_0 : slice 15 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s15_1 : slice 15 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s15_2 : slice 15 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s15_3 : slice 15 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s15_4 : slice 15 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s15_5 : slice 15 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s15_6 : slice 15 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s15_7 : slice 15 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s15_8 : slice 15 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s15_9 : slice 15 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s15_10 : slice 15 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s15_11 : slice 15 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s15_12 : slice 15 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s15_13 : slice 15 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s15_14 : slice 15 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s15_15 : slice 15 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s15_16 : slice 15 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s15_17 : slice 15 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s15_18 : slice 15 18 = true := by decide +kernel

/-- All 19 slices at e = 15. -/
theorem asm15 : ∀ f < 19, slice 15 f = true := by
  intro f hf
  interval_cases f
  exacts [s15_0, s15_1, s15_2, s15_3, s15_4, s15_5, s15_6, s15_7, s15_8, s15_9, s15_10, s15_11, s15_12, s15_13, s15_14, s15_15, s15_16, s15_17, s15_18]

end Machine19
