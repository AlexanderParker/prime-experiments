/-
Machine 19 qualifying scan (round 21), slice family e = 7 (all f < 19): 19
kernel checks of 5005 CRT tuples each. See Machine19QCore.lean.
-/

import Machine19QCore

namespace Machine19

set_option maxRecDepth 40000 in
theorem qs7_0 : qslice 7 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs7_1 : qslice 7 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs7_2 : qslice 7 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs7_3 : qslice 7 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs7_4 : qslice 7 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs7_5 : qslice 7 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs7_6 : qslice 7 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs7_7 : qslice 7 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs7_8 : qslice 7 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs7_9 : qslice 7 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs7_10 : qslice 7 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs7_11 : qslice 7 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs7_12 : qslice 7 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs7_13 : qslice 7 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs7_14 : qslice 7 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs7_15 : qslice 7 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs7_16 : qslice 7 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs7_17 : qslice 7 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs7_18 : qslice 7 18 = true := by decide +kernel

/-- All 19 slices at e = 7. -/
theorem qasm7 : ∀ f < 19, qslice 7 f = true := by
  intro f hf
  interval_cases f
  exacts [qs7_0, qs7_1, qs7_2, qs7_3, qs7_4, qs7_5, qs7_6, qs7_7, qs7_8, qs7_9, qs7_10, qs7_11, qs7_12, qs7_13, qs7_14, qs7_15, qs7_16, qs7_17, qs7_18]

end Machine19
