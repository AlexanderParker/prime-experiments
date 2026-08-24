/-
Machine 19 qualifying scan (round 21), slice family e = 12 (all f < 19): 19
kernel checks of 5005 CRT tuples each. See Machine19QCore.lean.
-/

import Machine19QCore

namespace Machine19

set_option maxRecDepth 40000 in
theorem qs12_0 : qslice 12 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs12_1 : qslice 12 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs12_2 : qslice 12 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs12_3 : qslice 12 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs12_4 : qslice 12 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs12_5 : qslice 12 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs12_6 : qslice 12 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs12_7 : qslice 12 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs12_8 : qslice 12 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs12_9 : qslice 12 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs12_10 : qslice 12 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs12_11 : qslice 12 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs12_12 : qslice 12 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs12_13 : qslice 12 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs12_14 : qslice 12 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs12_15 : qslice 12 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs12_16 : qslice 12 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs12_17 : qslice 12 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs12_18 : qslice 12 18 = true := by decide +kernel

/-- All 19 slices at e = 12. -/
theorem qasm12 : ∀ f < 19, qslice 12 f = true := by
  intro f hf
  interval_cases f
  exacts [qs12_0, qs12_1, qs12_2, qs12_3, qs12_4, qs12_5, qs12_6, qs12_7, qs12_8, qs12_9, qs12_10, qs12_11, qs12_12, qs12_13, qs12_14, qs12_15, qs12_16, qs12_17, qs12_18]

end Machine19
