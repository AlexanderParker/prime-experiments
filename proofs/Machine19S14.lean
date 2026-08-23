/-
Machine 19 period scan, slice family e = 14 (all f < 19): 19 kernel
checks of 5005 CRT tuples each. See Machine19Core.lean.
-/

import Machine19Core

namespace Machine19

set_option maxRecDepth 40000 in
theorem s14_0 : slice 14 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s14_1 : slice 14 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s14_2 : slice 14 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s14_3 : slice 14 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s14_4 : slice 14 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s14_5 : slice 14 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s14_6 : slice 14 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s14_7 : slice 14 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s14_8 : slice 14 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s14_9 : slice 14 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s14_10 : slice 14 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s14_11 : slice 14 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s14_12 : slice 14 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s14_13 : slice 14 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s14_14 : slice 14 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s14_15 : slice 14 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s14_16 : slice 14 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s14_17 : slice 14 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s14_18 : slice 14 18 = true := by decide +kernel

/-- All 19 slices at e = 14. -/
theorem asm14 : ∀ f < 19, slice 14 f = true := by
  intro f hf
  interval_cases f
  exacts [s14_0, s14_1, s14_2, s14_3, s14_4, s14_5, s14_6, s14_7, s14_8, s14_9, s14_10, s14_11, s14_12, s14_13, s14_14, s14_15, s14_16, s14_17, s14_18]

end Machine19
