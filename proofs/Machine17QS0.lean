/-
Machine 17 qualifying scan (round 22), slices e = 0 .. 5: kernel
checks of 5005 CRT tuples each. See Machine17QCore.lean.
-/

import Machine17QCore

namespace Machine17

set_option maxRecDepth 40000 in
theorem qs_0 : qslice 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs_1 : qslice 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs_2 : qslice 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs_3 : qslice 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs_4 : qslice 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs_5 : qslice 5 = true := by decide +kernel

/-- The slices of this family. -/
theorem qasm0 : ∀ e, 0 ≤ e → e ≤ 5 → qslice e = true := by
  intro e h1 h2
  interval_cases e
  exacts [qs_0, qs_1, qs_2, qs_3, qs_4, qs_5]

end Machine17
