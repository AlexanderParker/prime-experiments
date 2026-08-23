/-
Machine 19 period scan, slice family e = 16 (all f < 19): 19 kernel
checks of 5005 CRT tuples each. See Machine19Core.lean.
-/

import Machine19Core

namespace Machine19

set_option maxRecDepth 40000 in
theorem s16_0 : slice 16 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s16_1 : slice 16 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s16_2 : slice 16 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s16_3 : slice 16 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s16_4 : slice 16 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s16_5 : slice 16 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s16_6 : slice 16 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s16_7 : slice 16 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s16_8 : slice 16 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s16_9 : slice 16 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s16_10 : slice 16 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s16_11 : slice 16 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s16_12 : slice 16 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s16_13 : slice 16 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s16_14 : slice 16 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s16_15 : slice 16 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s16_16 : slice 16 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s16_17 : slice 16 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s16_18 : slice 16 18 = true := by decide +kernel

/-- All 19 slices at e = 16. -/
theorem asm16 : ∀ f < 19, slice 16 f = true := by
  intro f hf
  interval_cases f
  exacts [s16_0, s16_1, s16_2, s16_3, s16_4, s16_5, s16_6, s16_7, s16_8, s16_9, s16_10, s16_11, s16_12, s16_13, s16_14, s16_15, s16_16, s16_17, s16_18]

end Machine19
