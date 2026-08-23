/-
Machine 19 period scan, slice family e = 8 (all f < 19): 19 kernel
checks of 5005 CRT tuples each. See Machine19Core.lean.
-/

import Machine19Core

namespace Machine19

set_option maxRecDepth 40000 in
theorem s8_0 : slice 8 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s8_1 : slice 8 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s8_2 : slice 8 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s8_3 : slice 8 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s8_4 : slice 8 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s8_5 : slice 8 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s8_6 : slice 8 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s8_7 : slice 8 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s8_8 : slice 8 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s8_9 : slice 8 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s8_10 : slice 8 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s8_11 : slice 8 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s8_12 : slice 8 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s8_13 : slice 8 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s8_14 : slice 8 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s8_15 : slice 8 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s8_16 : slice 8 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s8_17 : slice 8 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s8_18 : slice 8 18 = true := by decide +kernel

/-- All 19 slices at e = 8. -/
theorem asm8 : ∀ f < 19, slice 8 f = true := by
  intro f hf
  interval_cases f
  exacts [s8_0, s8_1, s8_2, s8_3, s8_4, s8_5, s8_6, s8_7, s8_8, s8_9, s8_10, s8_11, s8_12, s8_13, s8_14, s8_15, s8_16, s8_17, s8_18]

end Machine19
