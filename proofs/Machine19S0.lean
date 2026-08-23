/-
Machine 19 period scan, slice family e = 0 (all f < 19): 19 kernel
checks of 5005 CRT tuples each. See Machine19Core.lean.
-/

import Machine19Core

namespace Machine19

set_option maxRecDepth 40000 in
theorem s0_0 : slice 0 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s0_1 : slice 0 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s0_2 : slice 0 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s0_3 : slice 0 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s0_4 : slice 0 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s0_5 : slice 0 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s0_6 : slice 0 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s0_7 : slice 0 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s0_8 : slice 0 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s0_9 : slice 0 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s0_10 : slice 0 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s0_11 : slice 0 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s0_12 : slice 0 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s0_13 : slice 0 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s0_14 : slice 0 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s0_15 : slice 0 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s0_16 : slice 0 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s0_17 : slice 0 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s0_18 : slice 0 18 = true := by decide +kernel

/-- All 19 slices at e = 0. -/
theorem asm0 : ∀ f < 19, slice 0 f = true := by
  intro f hf
  interval_cases f
  exacts [s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7, s0_8, s0_9, s0_10, s0_11, s0_12, s0_13, s0_14, s0_15, s0_16, s0_17, s0_18]

end Machine19
