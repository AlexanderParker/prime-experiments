/-
Machine 19 period scan, slice family e = 3 (all f < 19): 19 kernel
checks of 5005 CRT tuples each. See Machine19Core.lean.
-/

import Machine19Core

namespace Machine19

set_option maxRecDepth 40000 in
theorem s3_0 : slice 3 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s3_1 : slice 3 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s3_2 : slice 3 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s3_3 : slice 3 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s3_4 : slice 3 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s3_5 : slice 3 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s3_6 : slice 3 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s3_7 : slice 3 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s3_8 : slice 3 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s3_9 : slice 3 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s3_10 : slice 3 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s3_11 : slice 3 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s3_12 : slice 3 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s3_13 : slice 3 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s3_14 : slice 3 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s3_15 : slice 3 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s3_16 : slice 3 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s3_17 : slice 3 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s3_18 : slice 3 18 = true := by decide +kernel

/-- All 19 slices at e = 3. -/
theorem asm3 : ∀ f < 19, slice 3 f = true := by
  intro f hf
  interval_cases f
  exacts [s3_0, s3_1, s3_2, s3_3, s3_4, s3_5, s3_6, s3_7, s3_8, s3_9, s3_10, s3_11, s3_12, s3_13, s3_14, s3_15, s3_16, s3_17, s3_18]

end Machine19
