/-
Machine 19 period scan, slice family e = 10 (all f < 19): 19 kernel
checks of 5005 CRT tuples each. See Machine19Core.lean.
-/

import Machine19Core

namespace Machine19

set_option maxRecDepth 40000 in
theorem s10_0 : slice 10 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s10_1 : slice 10 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s10_2 : slice 10 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s10_3 : slice 10 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s10_4 : slice 10 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s10_5 : slice 10 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s10_6 : slice 10 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s10_7 : slice 10 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s10_8 : slice 10 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s10_9 : slice 10 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s10_10 : slice 10 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s10_11 : slice 10 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s10_12 : slice 10 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s10_13 : slice 10 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s10_14 : slice 10 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s10_15 : slice 10 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s10_16 : slice 10 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s10_17 : slice 10 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s10_18 : slice 10 18 = true := by decide +kernel

/-- All 19 slices at e = 10. -/
theorem asm10 : ∀ f < 19, slice 10 f = true := by
  intro f hf
  interval_cases f
  exacts [s10_0, s10_1, s10_2, s10_3, s10_4, s10_5, s10_6, s10_7, s10_8, s10_9, s10_10, s10_11, s10_12, s10_13, s10_14, s10_15, s10_16, s10_17, s10_18]

end Machine19
