/-
Machine 17 qualifying scan (round 22), slices e = 6 .. 11: kernel
checks of 5005 CRT tuples each. See Machine17QCore.lean.
-/

import Machine17QCore

namespace Machine17

set_option maxRecDepth 40000 in
theorem qs_6 : qslice 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs_7 : qslice 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs_8 : qslice 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs_9 : qslice 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs_10 : qslice 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem qs_11 : qslice 11 = true := by decide +kernel

/-- The slices of this family. -/
theorem qasm1 : ∀ e, 6 ≤ e → e ≤ 11 → qslice e = true := by
  intro e h1 h2
  interval_cases e
  exacts [qs_6, qs_7, qs_8, qs_9, qs_10, qs_11]

end Machine17
