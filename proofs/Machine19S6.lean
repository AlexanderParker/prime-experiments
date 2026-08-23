/-
Machine 19 period scan, slice family e = 6 (all f < 19): 19 kernel
checks of 5005 CRT tuples each. See Machine19Core.lean.
-/

import Machine19Core

namespace Machine19

set_option maxRecDepth 40000 in
theorem s6_0 : slice 6 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s6_1 : slice 6 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s6_2 : slice 6 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s6_3 : slice 6 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s6_4 : slice 6 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s6_5 : slice 6 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s6_6 : slice 6 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s6_7 : slice 6 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s6_8 : slice 6 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s6_9 : slice 6 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s6_10 : slice 6 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s6_11 : slice 6 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s6_12 : slice 6 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s6_13 : slice 6 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s6_14 : slice 6 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s6_15 : slice 6 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s6_16 : slice 6 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s6_17 : slice 6 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s6_18 : slice 6 18 = true := by decide +kernel

/-- All 19 slices at e = 6. -/
theorem asm6 : ∀ f < 19, slice 6 f = true := by
  intro f hf
  interval_cases f
  exacts [s6_0, s6_1, s6_2, s6_3, s6_4, s6_5, s6_6, s6_7, s6_8, s6_9, s6_10, s6_11, s6_12, s6_13, s6_14, s6_15, s6_16, s6_17, s6_18]

end Machine19
