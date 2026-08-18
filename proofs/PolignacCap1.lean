import PolignacCapCore

namespace PolignacCap

set_option maxRecDepth 40000 in
/-- `gcd(e,105) = 1` - the twin case `d = 2` among others. -/
theorem cap_gcd_1 : capOK 1 6 26 = true := by decide +kernel

end PolignacCap
