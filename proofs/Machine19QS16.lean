/-
Machine 19 qualifying scan (round 21), slice family e = 16 (all f < 19): 19
kernel checks of 5005 CRT tuples each. See Machine19QCore.lean.
-/

import Machine19QCore

namespace Machine19

set_option maxRecDepth 40000 in
theorem qs16_0 : qslice 16 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs16_1 : qslice 16 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs16_2 : qslice 16 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs16_3 : qslice 16 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs16_4 : qslice 16 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs16_5 : qslice 16 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs16_6 : qslice 16 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs16_7 : qslice 16 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs16_8 : qslice 16 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs16_9 : qslice 16 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs16_10 : qslice 16 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs16_11 : qslice 16 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs16_12 : qslice 16 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs16_13 : qslice 16 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs16_14 : qslice 16 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs16_15 : qslice 16 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs16_16 : qslice 16 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs16_17 : qslice 16 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs16_18 : qslice 16 18 = true := by decide +kernel

/-- All 19 slices at e = 16. -/
theorem qasm16 : ∀ f < 19, qslice 16 f = true := by
  intro f hf
  interval_cases f
  exacts [qs16_0, qs16_1, qs16_2, qs16_3, qs16_4, qs16_5, qs16_6, qs16_7, qs16_8, qs16_9, qs16_10, qs16_11, qs16_12, qs16_13, qs16_14, qs16_15, qs16_16, qs16_17, qs16_18]

end Machine19
