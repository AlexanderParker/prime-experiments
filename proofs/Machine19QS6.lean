/-
Machine 19 qualifying scan (round 21), slice family e = 6 (all f < 19): 19
kernel checks of 5005 CRT tuples each. See Machine19QCore.lean.
-/

import Machine19QCore

namespace Machine19

set_option maxRecDepth 40000 in
theorem qs6_0 : qslice 6 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs6_1 : qslice 6 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs6_2 : qslice 6 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs6_3 : qslice 6 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs6_4 : qslice 6 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs6_5 : qslice 6 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs6_6 : qslice 6 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs6_7 : qslice 6 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs6_8 : qslice 6 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs6_9 : qslice 6 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs6_10 : qslice 6 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs6_11 : qslice 6 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs6_12 : qslice 6 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs6_13 : qslice 6 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs6_14 : qslice 6 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs6_15 : qslice 6 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs6_16 : qslice 6 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs6_17 : qslice 6 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs6_18 : qslice 6 18 = true := by decide +kernel

/-- All 19 slices at e = 6. -/
theorem qasm6 : ∀ f < 19, qslice 6 f = true := by
  intro f hf
  interval_cases f
  exacts [qs6_0, qs6_1, qs6_2, qs6_3, qs6_4, qs6_5, qs6_6, qs6_7, qs6_8, qs6_9, qs6_10, qs6_11, qs6_12, qs6_13, qs6_14, qs6_15, qs6_16, qs6_17, qs6_18]

end Machine19
