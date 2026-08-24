/- Timing probe for the round-21 qualifying scan: one 5005-tuple slice. -/

import Machine19QCore

namespace Machine19

set_option maxRecDepth 40000 in
theorem qprobe_0_0 : qslice 0 0 = true := by decide +kernel

end Machine19
