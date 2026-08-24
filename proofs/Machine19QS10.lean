/-
Machine 19 qualifying scan (round 21), slice family e = 10 (all f < 19): 19
kernel checks of 5005 CRT tuples each. See Machine19QCore.lean.
-/

import Machine19QCore

namespace Machine19

set_option maxRecDepth 40000 in
theorem qs10_0 : qslice 10 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs10_1 : qslice 10 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs10_2 : qslice 10 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs10_3 : qslice 10 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs10_4 : qslice 10 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs10_5 : qslice 10 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs10_6 : qslice 10 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs10_7 : qslice 10 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs10_8 : qslice 10 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs10_9 : qslice 10 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs10_10 : qslice 10 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs10_11 : qslice 10 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs10_12 : qslice 10 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs10_13 : qslice 10 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs10_14 : qslice 10 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs10_15 : qslice 10 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs10_16 : qslice 10 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs10_17 : qslice 10 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs10_18 : qslice 10 18 = true := by decide +kernel

/-- All 19 slices at e = 10. -/
theorem qasm10 : ∀ f < 19, qslice 10 f = true := by
  intro f hf
  interval_cases f
  exacts [qs10_0, qs10_1, qs10_2, qs10_3, qs10_4, qs10_5, qs10_6, qs10_7, qs10_8, qs10_9, qs10_10, qs10_11, qs10_12, qs10_13, qs10_14, qs10_15, qs10_16, qs10_17, qs10_18]

end Machine19
