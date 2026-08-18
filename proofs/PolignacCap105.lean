import PolignacCapCore

namespace PolignacCap

set_option maxRecDepth 40000 in
/-- `gcd(e,105) = 105` - the extreme class, and the absolute ceiling. -/
theorem cap_gcd_105 : capOK 105 12 26 = true := by decide +kernel

end PolignacCap
