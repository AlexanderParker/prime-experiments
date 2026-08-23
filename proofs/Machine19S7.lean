/-
Machine 19 period scan, slice family e = 7 (all f < 19): 19 kernel
checks of 5005 CRT tuples each. See Machine19Core.lean.
-/

import Machine19Core

namespace Machine19

set_option maxRecDepth 40000 in
theorem s7_0 : slice 7 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s7_1 : slice 7 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s7_2 : slice 7 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s7_3 : slice 7 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s7_4 : slice 7 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s7_5 : slice 7 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s7_6 : slice 7 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s7_7 : slice 7 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s7_8 : slice 7 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s7_9 : slice 7 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s7_10 : slice 7 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s7_11 : slice 7 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s7_12 : slice 7 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s7_13 : slice 7 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s7_14 : slice 7 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s7_15 : slice 7 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s7_16 : slice 7 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s7_17 : slice 7 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s7_18 : slice 7 18 = true := by decide +kernel

/-- All 19 slices at e = 7. -/
theorem asm7 : ∀ f < 19, slice 7 f = true := by
  intro f hf
  interval_cases f
  exacts [s7_0, s7_1, s7_2, s7_3, s7_4, s7_5, s7_6, s7_7, s7_8, s7_9, s7_10, s7_11, s7_12, s7_13, s7_14, s7_15, s7_16, s7_17, s7_18]

end Machine19
