import PolignacCapCore

namespace PolignacCap

set_option maxRecDepth 40000 in
/-- `gcd(e,105) = 35`. -/
theorem cap_gcd_35 : capOK 35 6 26 = true := by decide +kernel

end PolignacCap
