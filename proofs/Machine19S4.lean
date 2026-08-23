/-
Machine 19 period scan, slice family e = 4 (all f < 19): 19 kernel
checks of 5005 CRT tuples each. See Machine19Core.lean.
-/

import Machine19Core

namespace Machine19

set_option maxRecDepth 40000 in
theorem s4_0 : slice 4 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s4_1 : slice 4 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s4_2 : slice 4 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s4_3 : slice 4 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s4_4 : slice 4 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s4_5 : slice 4 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s4_6 : slice 4 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s4_7 : slice 4 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s4_8 : slice 4 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s4_9 : slice 4 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s4_10 : slice 4 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s4_11 : slice 4 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s4_12 : slice 4 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s4_13 : slice 4 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s4_14 : slice 4 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s4_15 : slice 4 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s4_16 : slice 4 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s4_17 : slice 4 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s4_18 : slice 4 18 = true := by decide +kernel

/-- All 19 slices at e = 4. -/
theorem asm4 : ∀ f < 19, slice 4 f = true := by
  intro f hf
  interval_cases f
  exacts [s4_0, s4_1, s4_2, s4_3, s4_4, s4_5, s4_6, s4_7, s4_8, s4_9, s4_10, s4_11, s4_12, s4_13, s4_14, s4_15, s4_16, s4_17, s4_18]

end Machine19
