import PolignacCapCore

namespace PolignacCap

set_option maxRecDepth 40000 in
/-- `gcd(e,105) = 21`. -/
theorem cap_gcd_21 : capOK 21 6 26 = true := by decide +kernel

end PolignacCap
