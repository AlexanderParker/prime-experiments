/-
Machine 19 period scan, slice family e = 9 (all f < 19): 19 kernel
checks of 5005 CRT tuples each. See Machine19Core.lean.
-/

import Machine19Core

namespace Machine19

set_option maxRecDepth 40000 in
theorem s9_0 : slice 9 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s9_1 : slice 9 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s9_2 : slice 9 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s9_3 : slice 9 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s9_4 : slice 9 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s9_5 : slice 9 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s9_6 : slice 9 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s9_7 : slice 9 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s9_8 : slice 9 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s9_9 : slice 9 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s9_10 : slice 9 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s9_11 : slice 9 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s9_12 : slice 9 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s9_13 : slice 9 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s9_14 : slice 9 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s9_15 : slice 9 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s9_16 : slice 9 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s9_17 : slice 9 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s9_18 : slice 9 18 = true := by decide +kernel

/-- All 19 slices at e = 9. -/
theorem asm9 : ∀ f < 19, slice 9 f = true := by
  intro f hf
  interval_cases f
  exacts [s9_0, s9_1, s9_2, s9_3, s9_4, s9_5, s9_6, s9_7, s9_8, s9_9, s9_10, s9_11, s9_12, s9_13, s9_14, s9_15, s9_16, s9_17, s9_18]

end Machine19
