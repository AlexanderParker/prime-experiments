/-
Machine 19 qualifying scan (round 21), slice family e = 1 (all f < 19): 19
kernel checks of 5005 CRT tuples each. See Machine19QCore.lean.
-/

import Machine19QCore

namespace Machine19

set_option maxRecDepth 40000 in
theorem qs1_0 : qslice 1 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs1_1 : qslice 1 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs1_2 : qslice 1 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs1_3 : qslice 1 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs1_4 : qslice 1 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs1_5 : qslice 1 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs1_6 : qslice 1 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs1_7 : qslice 1 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs1_8 : qslice 1 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs1_9 : qslice 1 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs1_10 : qslice 1 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs1_11 : qslice 1 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs1_12 : qslice 1 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs1_13 : qslice 1 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs1_14 : qslice 1 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs1_15 : qslice 1 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs1_16 : qslice 1 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs1_17 : qslice 1 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs1_18 : qslice 1 18 = true := by decide +kernel

/-- All 19 slices at e = 1. -/
theorem qasm1 : ∀ f < 19, qslice 1 f = true := by
  intro f hf
  interval_cases f
  exacts [qs1_0, qs1_1, qs1_2, qs1_3, qs1_4, qs1_5, qs1_6, qs1_7, qs1_8, qs1_9, qs1_10, qs1_11, qs1_12, qs1_13, qs1_14, qs1_15, qs1_16, qs1_17, qs1_18]

end Machine19
