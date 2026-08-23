/-
Machine 19 period scan, slice family e = 1 (all f < 19): 19 kernel
checks of 5005 CRT tuples each. See Machine19Core.lean.
-/

import Machine19Core

namespace Machine19

set_option maxRecDepth 40000 in
theorem s1_0 : slice 1 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s1_1 : slice 1 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s1_2 : slice 1 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s1_3 : slice 1 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s1_4 : slice 1 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s1_5 : slice 1 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s1_6 : slice 1 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s1_7 : slice 1 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s1_8 : slice 1 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s1_9 : slice 1 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s1_10 : slice 1 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s1_11 : slice 1 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s1_12 : slice 1 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s1_13 : slice 1 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s1_14 : slice 1 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s1_15 : slice 1 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s1_16 : slice 1 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s1_17 : slice 1 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s1_18 : slice 1 18 = true := by decide +kernel

/-- All 19 slices at e = 1. -/
theorem asm1 : ∀ f < 19, slice 1 f = true := by
  intro f hf
  interval_cases f
  exacts [s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7, s1_8, s1_9, s1_10, s1_11, s1_12, s1_13, s1_14, s1_15, s1_16, s1_17, s1_18]

end Machine19
