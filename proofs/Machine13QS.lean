/-
Machine 13 qualifying scan (round 22): the whole 5005-tuple period in one
kernel check. See Machine13QCore.lean.
-/

import Machine13QCore

namespace Machine13

set_option maxRecDepth 40000 in
theorem qasm : qslice = true := by decide +kernel

end Machine13
