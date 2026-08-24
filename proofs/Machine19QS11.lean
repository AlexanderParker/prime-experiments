/-
Machine 19 qualifying scan (round 21), slice family e = 11 (all f < 19): 19
kernel checks of 5005 CRT tuples each. See Machine19QCore.lean.
-/

import Machine19QCore

namespace Machine19

set_option maxRecDepth 40000 in
theorem qs11_0 : qslice 11 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs11_1 : qslice 11 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs11_2 : qslice 11 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs11_3 : qslice 11 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs11_4 : qslice 11 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs11_5 : qslice 11 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs11_6 : qslice 11 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs11_7 : qslice 11 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs11_8 : qslice 11 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs11_9 : qslice 11 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs11_10 : qslice 11 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs11_11 : qslice 11 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs11_12 : qslice 11 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs11_13 : qslice 11 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs11_14 : qslice 11 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs11_15 : qslice 11 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs11_16 : qslice 11 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs11_17 : qslice 11 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs11_18 : qslice 11 18 = true := by decide +kernel

/-- All 19 slices at e = 11. -/
theorem qasm11 : ∀ f < 19, qslice 11 f = true := by
  intro f hf
  interval_cases f
  exacts [qs11_0, qs11_1, qs11_2, qs11_3, qs11_4, qs11_5, qs11_6, qs11_7, qs11_8, qs11_9, qs11_10, qs11_11, qs11_12, qs11_13, qs11_14, qs11_15, qs11_16, qs11_17, qs11_18]

end Machine19
