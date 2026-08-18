import PolignacCapCore

namespace PolignacCap

set_option maxRecDepth 40000 in
/-- `gcd(e,105) = 5`. -/
theorem cap_gcd_5 : capOK 5 6 26 = true := by decide +kernel

end PolignacCap
