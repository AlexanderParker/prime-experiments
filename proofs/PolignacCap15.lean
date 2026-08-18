import PolignacCapCore

namespace PolignacCap

set_option maxRecDepth 40000 in
/-- `gcd(e,105) = 15` - the ceiling first breaks here. -/
theorem cap_gcd_15 : capOK 15 10 26 = true := by decide +kernel

end PolignacCap
