/-
Machine 19 period scan, slice family e = 2 (all f < 19): 19 kernel
checks of 5005 CRT tuples each. See Machine19Core.lean.
-/

import Machine19Core

namespace Machine19

set_option maxRecDepth 40000 in
theorem s2_0 : slice 2 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s2_1 : slice 2 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s2_2 : slice 2 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s2_3 : slice 2 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s2_4 : slice 2 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s2_5 : slice 2 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s2_6 : slice 2 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s2_7 : slice 2 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s2_8 : slice 2 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s2_9 : slice 2 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s2_10 : slice 2 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s2_11 : slice 2 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s2_12 : slice 2 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s2_13 : slice 2 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s2_14 : slice 2 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s2_15 : slice 2 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s2_16 : slice 2 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s2_17 : slice 2 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s2_18 : slice 2 18 = true := by decide +kernel

/-- All 19 slices at e = 2. -/
theorem asm2 : ∀ f < 19, slice 2 f = true := by
  intro f hf
  interval_cases f
  exacts [s2_0, s2_1, s2_2, s2_3, s2_4, s2_5, s2_6, s2_7, s2_8, s2_9, s2_10, s2_11, s2_12, s2_13, s2_14, s2_15, s2_16, s2_17, s2_18]

end Machine19
