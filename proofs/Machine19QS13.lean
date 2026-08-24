/-
Machine 19 qualifying scan (round 21), slice family e = 13 (all f < 19): 19
kernel checks of 5005 CRT tuples each. See Machine19QCore.lean.
-/

import Machine19QCore

namespace Machine19

set_option maxRecDepth 40000 in
theorem qs13_0 : qslice 13 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs13_1 : qslice 13 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs13_2 : qslice 13 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs13_3 : qslice 13 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs13_4 : qslice 13 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs13_5 : qslice 13 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs13_6 : qslice 13 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs13_7 : qslice 13 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs13_8 : qslice 13 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs13_9 : qslice 13 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs13_10 : qslice 13 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs13_11 : qslice 13 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs13_12 : qslice 13 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs13_13 : qslice 13 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs13_14 : qslice 13 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs13_15 : qslice 13 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs13_16 : qslice 13 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs13_17 : qslice 13 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs13_18 : qslice 13 18 = true := by decide +kernel

/-- All 19 slices at e = 13. -/
theorem qasm13 : ∀ f < 19, qslice 13 f = true := by
  intro f hf
  interval_cases f
  exacts [qs13_0, qs13_1, qs13_2, qs13_3, qs13_4, qs13_5, qs13_6, qs13_7, qs13_8, qs13_9, qs13_10, qs13_11, qs13_12, qs13_13, qs13_14, qs13_15, qs13_16, qs13_17, qs13_18]

end Machine19
