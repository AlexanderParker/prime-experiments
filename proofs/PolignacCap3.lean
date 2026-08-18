import PolignacCapCore

namespace PolignacCap

set_option maxRecDepth 40000 in
/-- `gcd(e,105) = 3`, i.e. `d = 0 (mod 6)` - the densest Polignac gaps, and the case excluded from the original mod-35 treatment. Still capped at 6. -/
theorem cap_gcd_3 : capOK 3 6 26 = true := by decide +kernel

end PolignacCap
