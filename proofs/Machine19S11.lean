/-
Machine 19 period scan, slice family e = 11 (all f < 19): 19 kernel
checks of 5005 CRT tuples each. See Machine19Core.lean.
-/

import Machine19Core

namespace Machine19

set_option maxRecDepth 40000 in
theorem s11_0 : slice 11 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s11_1 : slice 11 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s11_2 : slice 11 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s11_3 : slice 11 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s11_4 : slice 11 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s11_5 : slice 11 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s11_6 : slice 11 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s11_7 : slice 11 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s11_8 : slice 11 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s11_9 : slice 11 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s11_10 : slice 11 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s11_11 : slice 11 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s11_12 : slice 11 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s11_13 : slice 11 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s11_14 : slice 11 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s11_15 : slice 11 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s11_16 : slice 11 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s11_17 : slice 11 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s11_18 : slice 11 18 = true := by decide +kernel

/-- All 19 slices at e = 11. -/
theorem asm11 : ∀ f < 19, slice 11 f = true := by
  intro f hf
  interval_cases f
  exacts [s11_0, s11_1, s11_2, s11_3, s11_4, s11_5, s11_6, s11_7, s11_8, s11_9, s11_10, s11_11, s11_12, s11_13, s11_14, s11_15, s11_16, s11_17, s11_18]

end Machine19
