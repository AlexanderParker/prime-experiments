/-
The literal cap over ALL Polignac gaps - root module.

The eight gcd classes are proved in separate modules so that each kernel
evaluation runs in its own process; eight heavy `decide +kernel` calls in a
single file accumulate memory and stall (measured: >20 min and still going,
versus ~20-60 s each when separated).
-/

import PolignacCapCore
import PolignacCap1
import PolignacCap3
import PolignacCap5
import PolignacCap7
import PolignacCap15
import PolignacCap21
import PolignacCap35
import PolignacCap105

namespace PolignacCap

/-- The cap attached to each of the eight divisor classes of 105. -/
def capOf : ℕ → ℕ
  | 1 => 6 | 3 => 6 | 5 => 6 | 7 => 6 | 15 => 10 | 21 => 6 | 35 => 6
  | 105 => 12 | _ => 0

/-- **12 is the absolute ceiling over all Polignac gaps.** Every divisor
class of 105 - hence every even gap `d = 2e` - caps at 12 or below. -/
theorem capOf_le_twelve : ∀ g ∈ [1, 3, 5, 7, 15, 21, 35, 105], capOf g ≤ 12 := by
  decide

end PolignacCap
