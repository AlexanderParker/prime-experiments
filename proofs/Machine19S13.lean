/-
Machine 19 period scan, slice family e = 13 (all f < 19): 19 kernel
checks of 5005 CRT tuples each. See Machine19Core.lean.
-/

import Machine19Core

namespace Machine19

set_option maxRecDepth 40000 in
theorem s13_0 : slice 13 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s13_1 : slice 13 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s13_2 : slice 13 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s13_3 : slice 13 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s13_4 : slice 13 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s13_5 : slice 13 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s13_6 : slice 13 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s13_7 : slice 13 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s13_8 : slice 13 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s13_9 : slice 13 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s13_10 : slice 13 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s13_11 : slice 13 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s13_12 : slice 13 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s13_13 : slice 13 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s13_14 : slice 13 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s13_15 : slice 13 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s13_16 : slice 13 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s13_17 : slice 13 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s13_18 : slice 13 18 = true := by decide +kernel

/-- All 19 slices at e = 13. -/
theorem asm13 : ∀ f < 19, slice 13 f = true := by
  intro f hf
  interval_cases f
  exacts [s13_0, s13_1, s13_2, s13_3, s13_4, s13_5, s13_6, s13_7, s13_8, s13_9, s13_10, s13_11, s13_12, s13_13, s13_14, s13_15, s13_16, s13_17, s13_18]

end Machine19
