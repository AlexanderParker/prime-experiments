/-
Machine 19 qualifying scan (round 21), slice family e = 4 (all f < 19): 19
kernel checks of 5005 CRT tuples each. See Machine19QCore.lean.
-/

import Machine19QCore

namespace Machine19

set_option maxRecDepth 40000 in
theorem qs4_0 : qslice 4 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs4_1 : qslice 4 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs4_2 : qslice 4 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs4_3 : qslice 4 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs4_4 : qslice 4 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs4_5 : qslice 4 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs4_6 : qslice 4 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs4_7 : qslice 4 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs4_8 : qslice 4 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs4_9 : qslice 4 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs4_10 : qslice 4 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs4_11 : qslice 4 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs4_12 : qslice 4 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs4_13 : qslice 4 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs4_14 : qslice 4 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs4_15 : qslice 4 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs4_16 : qslice 4 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs4_17 : qslice 4 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs4_18 : qslice 4 18 = true := by decide +kernel

/-- All 19 slices at e = 4. -/
theorem qasm4 : ∀ f < 19, qslice 4 f = true := by
  intro f hf
  interval_cases f
  exacts [qs4_0, qs4_1, qs4_2, qs4_3, qs4_4, qs4_5, qs4_6, qs4_7, qs4_8, qs4_9, qs4_10, qs4_11, qs4_12, qs4_13, qs4_14, qs4_15, qs4_16, qs4_17, qs4_18]

end Machine19
