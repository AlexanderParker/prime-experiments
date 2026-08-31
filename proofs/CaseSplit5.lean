/-
THE CASE-SPLIT VEHICLE - THE FIVE-FREE-GEAR ARITY.

Round 28.  `CaseSplit.lean` carries the lowest-blocker inequality at six and
seven free gears (the arities the (D) rungs 19->23 and 29->31 need).  The
INCREMENT-WIDTH rung at 19->23 holds TWO gears rather than one, leaving five
free, so it needs the same statement one gear narrower.

It is kept in its own module rather than appended to `CaseSplit.lean` for a
purely mechanical reason worth recording: lake keys on CONTENT HASHES, so
touching `CaseSplit.lean` would invalidate all 75 existing case modules and cost
about an hour of kernel to rebuild artefacts that did not change.  A new module
in the same namespace costs nothing.
-/

import CaseSplit

namespace CaseSplit

/-- **Five free gears.**  `c a` says gear `a` blocks the position.  The pair term
for `(a, b)` fires only when `a` is the LOWEST blocker, so the pairs counted are
exactly `(lowest, other)` and there are `#blockers - 1` of them. -/
theorem lowest5 (c0 c1 c2 c3 c4 : Bool)
    (h : (c0 || c1 || c2 || c3 || c4) = true) :
    (1 : ℤ) +
      ((if c0 && c1 then 1 else 0) + (if c0 && c2 then 1 else 0) +
       (if c0 && c3 then 1 else 0) + (if c0 && c4 then 1 else 0) +
       (if !c0 && c1 && c2 then 1 else 0) + (if !c0 && c1 && c3 then 1 else 0) +
       (if !c0 && c1 && c4 then 1 else 0) +
       (if !c0 && !c1 && c2 && c3 then 1 else 0) +
       (if !c0 && !c1 && c2 && c4 then 1 else 0) +
       (if !c0 && !c1 && !c2 && c3 && c4 then 1 else 0))
    ≤ (if c0 then 1 else 0) + (if c1 then 1 else 0) + (if c2 then 1 else 0) +
      (if c3 then 1 else 0) + (if c4 then 1 else 0) := by
  revert h
  cases c0 <;> cases c1 <;> cases c2 <;> cases c3 <;> cases c4 <;> decide

/-- A blocked position has at least one blocker (five free gears). -/
theorem degpos5 (c0 c1 c2 c3 c4 : Bool)
    (h : (c0 || c1 || c2 || c3 || c4) = true) :
    (1 : ℤ) ≤ (if c0 then 1 else 0) + (if c1 then 1 else 0) +
      (if c2 then 1 else 0) + (if c3 then 1 else 0) + (if c4 then 1 else 0) := by
  revert h
  cases c0 <;> cases c1 <;> cases c2 <;> cases c3 <;> cases c4 <;> decide

end CaseSplit
