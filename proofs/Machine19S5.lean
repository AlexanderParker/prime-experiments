/-
Machine 19 period scan, slice family e = 5 (all f < 19): 19 kernel
checks of 5005 CRT tuples each. See Machine19Core.lean.
-/

import Machine19Core

namespace Machine19

set_option maxRecDepth 40000 in
theorem s5_0 : slice 5 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s5_1 : slice 5 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s5_2 : slice 5 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s5_3 : slice 5 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s5_4 : slice 5 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s5_5 : slice 5 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s5_6 : slice 5 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s5_7 : slice 5 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s5_8 : slice 5 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s5_9 : slice 5 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s5_10 : slice 5 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s5_11 : slice 5 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s5_12 : slice 5 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s5_13 : slice 5 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s5_14 : slice 5 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s5_15 : slice 5 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s5_16 : slice 5 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s5_17 : slice 5 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s5_18 : slice 5 18 = true := by decide +kernel

/-- All 19 slices at e = 5. -/
theorem asm5 : ∀ f < 19, slice 5 f = true := by
  intro f hf
  interval_cases f
  exacts [s5_0, s5_1, s5_2, s5_3, s5_4, s5_5, s5_6, s5_7, s5_8, s5_9, s5_10, s5_11, s5_12, s5_13, s5_14, s5_15, s5_16, s5_17, s5_18]

end Machine19
