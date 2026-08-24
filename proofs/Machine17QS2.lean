/-
Machine 17 qualifying scan (round 22), slices e = 12 .. 16: kernel
checks of 5005 CRT tuples each. See Machine17QCore.lean.
-/

import Machine17QCore

namespace Machine17

set_option maxRecDepth 40000 in
theorem qs_12 : qslice 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs_13 : qslice 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs_14 : qslice 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs_15 : qslice 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs_16 : qslice 16 = true := by decide +kernel

/-- The slices of this family. -/
theorem qasm2 : ∀ e, 12 ≤ e → e ≤ 16 → qslice e = true := by
  intro e h1 h2
  interval_cases e
  exacts [qs_12, qs_13, qs_14, qs_15, qs_16]

end Machine17
