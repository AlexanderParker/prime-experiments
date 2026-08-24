/-
Machine 19 qualifying scan (round 21), slice family e = 2 (all f < 19): 19
kernel checks of 5005 CRT tuples each. See Machine19QCore.lean.
-/

import Machine19QCore

namespace Machine19

set_option maxRecDepth 40000 in
theorem qs2_0 : qslice 2 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs2_1 : qslice 2 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs2_2 : qslice 2 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs2_3 : qslice 2 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs2_4 : qslice 2 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs2_5 : qslice 2 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs2_6 : qslice 2 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs2_7 : qslice 2 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs2_8 : qslice 2 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs2_9 : qslice 2 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs2_10 : qslice 2 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs2_11 : qslice 2 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs2_12 : qslice 2 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs2_13 : qslice 2 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs2_14 : qslice 2 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs2_15 : qslice 2 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs2_16 : qslice 2 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs2_17 : qslice 2 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs2_18 : qslice 2 18 = true := by decide +kernel

/-- All 19 slices at e = 2. -/
theorem qasm2 : ∀ f < 19, qslice 2 f = true := by
  intro f hf
  interval_cases f
  exacts [qs2_0, qs2_1, qs2_2, qs2_3, qs2_4, qs2_5, qs2_6, qs2_7, qs2_8, qs2_9, qs2_10, qs2_11, qs2_12, qs2_13, qs2_14, qs2_15, qs2_16, qs2_17, qs2_18]

end Machine19
