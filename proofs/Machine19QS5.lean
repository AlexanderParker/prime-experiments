/-
Machine 19 qualifying scan (round 21), slice family e = 5 (all f < 19): 19
kernel checks of 5005 CRT tuples each. See Machine19QCore.lean.
-/

import Machine19QCore

namespace Machine19

set_option maxRecDepth 40000 in
theorem qs5_0 : qslice 5 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs5_1 : qslice 5 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs5_2 : qslice 5 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs5_3 : qslice 5 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs5_4 : qslice 5 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs5_5 : qslice 5 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs5_6 : qslice 5 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs5_7 : qslice 5 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs5_8 : qslice 5 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs5_9 : qslice 5 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs5_10 : qslice 5 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs5_11 : qslice 5 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs5_12 : qslice 5 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs5_13 : qslice 5 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs5_14 : qslice 5 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs5_15 : qslice 5 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs5_16 : qslice 5 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs5_17 : qslice 5 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs5_18 : qslice 5 18 = true := by decide +kernel

/-- All 19 slices at e = 5. -/
theorem qasm5 : ∀ f < 19, qslice 5 f = true := by
  intro f hf
  interval_cases f
  exacts [qs5_0, qs5_1, qs5_2, qs5_3, qs5_4, qs5_5, qs5_6, qs5_7, qs5_8, qs5_9, qs5_10, qs5_11, qs5_12, qs5_13, qs5_14, qs5_15, qs5_16, qs5_17, qs5_18]

end Machine19
