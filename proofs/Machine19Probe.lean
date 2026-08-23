import Machine19Core
namespace Machine19
set_option maxRecDepth 40000 in
theorem probe_1_1 : slice 1 1 = true := by decide +kernel
end Machine19
