/-
Machine 19 qualifying scan (round 21), slice family e = 8 (all f < 19): 19
kernel checks of 5005 CRT tuples each. See Machine19QCore.lean.
-/

import Machine19QCore

namespace Machine19

set_option maxRecDepth 40000 in
theorem qs8_0 : qslice 8 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs8_1 : qslice 8 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs8_2 : qslice 8 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs8_3 : qslice 8 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs8_4 : qslice 8 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs8_5 : qslice 8 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs8_6 : qslice 8 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs8_7 : qslice 8 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs8_8 : qslice 8 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs8_9 : qslice 8 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs8_10 : qslice 8 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs8_11 : qslice 8 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs8_12 : qslice 8 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs8_13 : qslice 8 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs8_14 : qslice 8 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs8_15 : qslice 8 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs8_16 : qslice 8 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs8_17 : qslice 8 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs8_18 : qslice 8 18 = true := by decide +kernel

/-- All 19 slices at e = 8. -/
theorem qasm8 : ∀ f < 19, qslice 8 f = true := by
  intro f hf
  interval_cases f
  exacts [qs8_0, qs8_1, qs8_2, qs8_3, qs8_4, qs8_5, qs8_6, qs8_7, qs8_8, qs8_9, qs8_10, qs8_11, qs8_12, qs8_13, qs8_14, qs8_15, qs8_16, qs8_17, qs8_18]

end Machine19
