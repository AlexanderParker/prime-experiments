/-
Machine 19 period scan, slice family e = 12 (all f < 19): 19 kernel
checks of 5005 CRT tuples each. See Machine19Core.lean.
-/

import Machine19Core

namespace Machine19

set_option maxRecDepth 40000 in
theorem s12_0 : slice 12 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s12_1 : slice 12 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s12_2 : slice 12 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s12_3 : slice 12 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s12_4 : slice 12 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s12_5 : slice 12 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s12_6 : slice 12 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s12_7 : slice 12 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s12_8 : slice 12 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s12_9 : slice 12 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s12_10 : slice 12 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s12_11 : slice 12 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s12_12 : slice 12 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s12_13 : slice 12 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s12_14 : slice 12 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s12_15 : slice 12 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s12_16 : slice 12 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s12_17 : slice 12 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s12_18 : slice 12 18 = true := by decide +kernel

/-- All 19 slices at e = 12. -/
theorem asm12 : ∀ f < 19, slice 12 f = true := by
  intro f hf
  interval_cases f
  exacts [s12_0, s12_1, s12_2, s12_3, s12_4, s12_5, s12_6, s12_7, s12_8, s12_9, s12_10, s12_11, s12_12, s12_13, s12_14, s12_15, s12_16, s12_17, s12_18]

end Machine19
