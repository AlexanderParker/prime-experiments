/-
Machine 19 qualifying scan (round 21), slice family e = 3 (all f < 19): 19
kernel checks of 5005 CRT tuples each. See Machine19QCore.lean.
-/

import Machine19QCore

namespace Machine19

set_option maxRecDepth 40000 in
theorem qs3_0 : qslice 3 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs3_1 : qslice 3 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs3_2 : qslice 3 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs3_3 : qslice 3 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs3_4 : qslice 3 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs3_5 : qslice 3 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs3_6 : qslice 3 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs3_7 : qslice 3 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs3_8 : qslice 3 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs3_9 : qslice 3 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs3_10 : qslice 3 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs3_11 : qslice 3 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs3_12 : qslice 3 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs3_13 : qslice 3 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs3_14 : qslice 3 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs3_15 : qslice 3 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs3_16 : qslice 3 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs3_17 : qslice 3 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs3_18 : qslice 3 18 = true := by decide +kernel

/-- All 19 slices at e = 3. -/
theorem qasm3 : ∀ f < 19, qslice 3 f = true := by
  intro f hf
  interval_cases f
  exacts [qs3_0, qs3_1, qs3_2, qs3_3, qs3_4, qs3_5, qs3_6, qs3_7, qs3_8, qs3_9, qs3_10, qs3_11, qs3_12, qs3_13, qs3_14, qs3_15, qs3_16, qs3_17, qs3_18]

end Machine19
