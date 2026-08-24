/-
Machine 19 qualifying scan (round 21), slice family e = 9 (all f < 19): 19
kernel checks of 5005 CRT tuples each. See Machine19QCore.lean.
-/

import Machine19QCore

namespace Machine19

set_option maxRecDepth 40000 in
theorem qs9_0 : qslice 9 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs9_1 : qslice 9 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs9_2 : qslice 9 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs9_3 : qslice 9 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs9_4 : qslice 9 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs9_5 : qslice 9 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs9_6 : qslice 9 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs9_7 : qslice 9 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs9_8 : qslice 9 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs9_9 : qslice 9 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs9_10 : qslice 9 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs9_11 : qslice 9 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs9_12 : qslice 9 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs9_13 : qslice 9 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs9_14 : qslice 9 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs9_15 : qslice 9 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs9_16 : qslice 9 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs9_17 : qslice 9 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs9_18 : qslice 9 18 = true := by decide +kernel

/-- All 19 slices at e = 9. -/
theorem qasm9 : ∀ f < 19, qslice 9 f = true := by
  intro f hf
  interval_cases f
  exacts [qs9_0, qs9_1, qs9_2, qs9_3, qs9_4, qs9_5, qs9_6, qs9_7, qs9_8, qs9_9, qs9_10, qs9_11, qs9_12, qs9_13, qs9_14, qs9_15, qs9_16, qs9_17, qs9_18]

end Machine19
