/-
Machine 19 qualifying scan (round 21), slice family e = 0 (all f < 19): 19
kernel checks of 5005 CRT tuples each. See Machine19QCore.lean.
-/

import Machine19QCore

namespace Machine19

set_option maxRecDepth 40000 in
theorem qs0_0 : qslice 0 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs0_1 : qslice 0 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs0_2 : qslice 0 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs0_3 : qslice 0 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs0_4 : qslice 0 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs0_5 : qslice 0 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs0_6 : qslice 0 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs0_7 : qslice 0 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs0_8 : qslice 0 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs0_9 : qslice 0 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs0_10 : qslice 0 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs0_11 : qslice 0 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs0_12 : qslice 0 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs0_13 : qslice 0 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs0_14 : qslice 0 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs0_15 : qslice 0 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs0_16 : qslice 0 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs0_17 : qslice 0 17 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs0_18 : qslice 0 18 = true := by decide +kernel

/-- All 19 slices at e = 0. -/
theorem qasm0 : ∀ f < 19, qslice 0 f = true := by
  intro f hf
  interval_cases f
  exacts [qs0_0, qs0_1, qs0_2, qs0_3, qs0_4, qs0_5, qs0_6, qs0_7, qs0_8, qs0_9, qs0_10, qs0_11, qs0_12, qs0_13, qs0_14, qs0_15, qs0_16, qs0_17, qs0_18]

end Machine19
