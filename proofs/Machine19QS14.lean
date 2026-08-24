/-
Machine 19 qualifying scan (round 21), slice family e = 14 (all f < 19): 19
kernel checks of 5005 CRT tuples each. See Machine19QCore.lean.
-/

import Machine19QCore

namespace Machine19

set_option maxRecDepth 40000 in
theorem qs14_0 : qslice 14 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs14_1 : qslice 14 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs14_2 : qslice 14 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs14_3 : qslice 14 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs14_4 : qslice 14 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs14_5 : qslice 14 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs14_6 : qslice 14 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs14_7 : qslice 14 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs14_8 : qslice 14 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs14_9 : qslice 14 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs14_10 : qslice 14 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs14_11 : qslice 14 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs14_12 : qslice 14 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs14_13 : qslice 14 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs14_14 : qslice 14 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs14_15 : qslice 14 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs14_16 : qslice 14 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs14_17 : qslice 14 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs14_18 : qslice 14 18 = true := by decide +kernel

/-- All 19 slices at e = 14. -/
theorem qasm14 : ∀ f < 19, qslice 14 f = true := by
  intro f hf
  interval_cases f
  exacts [qs14_0, qs14_1, qs14_2, qs14_3, qs14_4, qs14_5, qs14_6, qs14_7, qs14_8, qs14_9, qs14_10, qs14_11, qs14_12, qs14_13, qs14_14, qs14_15, qs14_16, qs14_17, qs14_18]

end Machine19
