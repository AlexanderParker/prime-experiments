/-
Machine 19 qualifying scan (round 21), slice family e = 15 (all f < 19): 19
kernel checks of 5005 CRT tuples each. See Machine19QCore.lean.
-/

import Machine19QCore

namespace Machine19

set_option maxRecDepth 40000 in
theorem qs15_0 : qslice 15 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs15_1 : qslice 15 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs15_2 : qslice 15 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs15_3 : qslice 15 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs15_4 : qslice 15 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs15_5 : qslice 15 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs15_6 : qslice 15 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs15_7 : qslice 15 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs15_8 : qslice 15 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs15_9 : qslice 15 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs15_10 : qslice 15 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs15_11 : qslice 15 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs15_12 : qslice 15 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs15_13 : qslice 15 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs15_14 : qslice 15 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs15_15 : qslice 15 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs15_16 : qslice 15 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs15_17 : qslice 15 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs15_18 : qslice 15 18 = true := by decide +kernel

/-- All 19 slices at e = 15. -/
theorem qasm15 : ∀ f < 19, qslice 15 f = true := by
  intro f hf
  interval_cases f
  exacts [qs15_0, qs15_1, qs15_2, qs15_3, qs15_4, qs15_5, qs15_6, qs15_7, qs15_8, qs15_9, qs15_10, qs15_11, qs15_12, qs15_13, qs15_14, qs15_15, qs15_16, qs15_17, qs15_18]

end Machine19
