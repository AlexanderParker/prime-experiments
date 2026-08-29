/-
THE CASE-SPLIT COVERING VEHICLE (`RelaxStar`), the reusable half.

Round 27.  The LP-duality thread's round-26 vehicle certifies a (D) rung by
HOLDING the phases of the k smallest gears and producing, in EVERY case, an
exact rational dual certificate of the level-2 composed covering relaxation
restricted to the positions the held gears leave uncovered.  This file carries
the two ingredients that do not depend on which rung is being proved:

  * `mxr` / `mxr2` - the block maxima of the dual's aggregated coefficient
    vector, with the only fact the soundness proof needs about them
    (`le_mxr`, `le_mxr2`: every member of a block is at most the block max);

  * `lowest6` / `lowest7` - THE LOWEST-BLOCKER INEQUALITY, pointwise.  This is
    the combinatorial content of the vehicle's RECURSION ROW.  If position `x`
    is blocked by at least one free gear, and for each ORDERED pair `a < b` we
    count `x` once when both gears block it AND no gear below `a` does, then

        1 + #{such pairs}  =  #{gears blocking x}

    - because only the LOWEST blocker can be the `a` of such a pair, and it
    pairs with each of the other blockers exactly once.  Summed over the
    position set this is `sum_a |A_a| >= |pos| + sum_{a<b} n_ab` whenever
    `n_ab` is at most the number of positions where `a` is the lowest blocker
    and `b` also blocks - which is exactly what the vehicle's `n_ab` is
    (a MINIMUM over the lower gears' phases of that count).

Nothing here mentions a machine, a gear or a certificate; the per-rung files
supply those.
-/

import Machine19
import Mathlib.Algebra.BigOperators.Ring.Finset

namespace CaseSplit

/-! ## Block maxima -/

/-- `mxr f n = max over `s <= n` of `f s`, as a fold the kernel can evaluate. -/
def mxr (f : ℕ → ℤ) : ℕ → ℤ
  | 0 => f 0
  | n + 1 => max (f (n + 1)) (mxr f n)

theorem le_mxr (f : ℕ → ℤ) : ∀ n s, s ≤ n → f s ≤ mxr f n := by
  intro n
  induction n with
  | zero =>
      intro s hs
      have : s = 0 := Nat.le_zero.mp hs
      subst this
      exact le_refl _
  | succ m ih =>
      intro s hs
      simp only [mxr]
      rcases Nat.eq_or_lt_of_le hs with h | h
      · subst h; exact le_max_left _ _
      · exact le_trans (ih s (by omega)) (le_max_right _ _)

/-- The two-index block maximum. -/
def mxr2 (g : ℕ → ℕ → ℤ) (m n : ℕ) : ℤ := mxr (fun a => mxr (g a) n) m

theorem le_mxr2 (g : ℕ → ℕ → ℤ) (m n a b : ℕ) (ha : a ≤ m) (hb : b ≤ n) :
    g a b ≤ mxr2 g m n :=
  le_trans (le_mxr (g a) n b hb) (le_mxr (fun a => mxr (g a) n) m a ha)

/-! ## The lowest-blocker inequality -/

/-- **Six free gears.** `c a` says gear `a` blocks the position.  The pair term
for `(a, b)` fires only when `a` is the LOWEST blocker, so the pairs counted
are exactly `(lowest, other)` and there are `#blockers - 1` of them. -/
theorem lowest6 (c0 c1 c2 c3 c4 c5 : Bool)
    (h : (c0 || c1 || c2 || c3 || c4 || c5) = true) :
    (1 : ℤ) +
      ((if c0 && c1 then 1 else 0) + (if c0 && c2 then 1 else 0) +
       (if c0 && c3 then 1 else 0) + (if c0 && c4 then 1 else 0) +
       (if c0 && c5 then 1 else 0) +
       (if !c0 && c1 && c2 then 1 else 0) + (if !c0 && c1 && c3 then 1 else 0) +
       (if !c0 && c1 && c4 then 1 else 0) + (if !c0 && c1 && c5 then 1 else 0) +
       (if !c0 && !c1 && c2 && c3 then 1 else 0) +
       (if !c0 && !c1 && c2 && c4 then 1 else 0) +
       (if !c0 && !c1 && c2 && c5 then 1 else 0) +
       (if !c0 && !c1 && !c2 && c3 && c4 then 1 else 0) +
       (if !c0 && !c1 && !c2 && c3 && c5 then 1 else 0) +
       (if !c0 && !c1 && !c2 && !c3 && c4 && c5 then 1 else 0))
    ≤ (if c0 then 1 else 0) + (if c1 then 1 else 0) + (if c2 then 1 else 0) +
      (if c3 then 1 else 0) + (if c4 then 1 else 0) + (if c5 then 1 else 0) := by
  revert h
  cases c0 <;> cases c1 <;> cases c2 <;> cases c3 <;> cases c4 <;> cases c5 <;>
    decide

/-- **Seven free gears** - the same statement one gear wider (the 29->31 rung
holds two gears, leaving seven). -/
theorem lowest7 (c0 c1 c2 c3 c4 c5 c6 : Bool)
    (h : (c0 || c1 || c2 || c3 || c4 || c5 || c6) = true) :
    (1 : ℤ) +
      ((if c0 && c1 then 1 else 0) + (if c0 && c2 then 1 else 0) +
       (if c0 && c3 then 1 else 0) + (if c0 && c4 then 1 else 0) +
       (if c0 && c5 then 1 else 0) + (if c0 && c6 then 1 else 0) +
       (if !c0 && c1 && c2 then 1 else 0) + (if !c0 && c1 && c3 then 1 else 0) +
       (if !c0 && c1 && c4 then 1 else 0) + (if !c0 && c1 && c5 then 1 else 0) +
       (if !c0 && c1 && c6 then 1 else 0) +
       (if !c0 && !c1 && c2 && c3 then 1 else 0) +
       (if !c0 && !c1 && c2 && c4 then 1 else 0) +
       (if !c0 && !c1 && c2 && c5 then 1 else 0) +
       (if !c0 && !c1 && c2 && c6 then 1 else 0) +
       (if !c0 && !c1 && !c2 && c3 && c4 then 1 else 0) +
       (if !c0 && !c1 && !c2 && c3 && c5 then 1 else 0) +
       (if !c0 && !c1 && !c2 && c3 && c6 then 1 else 0) +
       (if !c0 && !c1 && !c2 && !c3 && c4 && c5 then 1 else 0) +
       (if !c0 && !c1 && !c2 && !c3 && c4 && c6 then 1 else 0) +
       (if !c0 && !c1 && !c2 && !c3 && !c4 && c5 && c6 then 1 else 0))
    ≤ (if c0 then 1 else 0) + (if c1 then 1 else 0) + (if c2 then 1 else 0) +
      (if c3 then 1 else 0) + (if c4 then 1 else 0) + (if c5 then 1 else 0) +
      (if c6 then 1 else 0) := by
  revert h
  cases c0 <;> cases c1 <;> cases c2 <;> cases c3 <;> cases c4 <;> cases c5 <;>
    cases c6 <;> decide

/-! ## Degree positivity -/

/-- A blocked position has at least one blocker. -/
theorem degpos6 (c0 c1 c2 c3 c4 c5 : Bool)
    (h : (c0 || c1 || c2 || c3 || c4 || c5) = true) :
    (1 : ℤ) ≤ (if c0 then 1 else 0) + (if c1 then 1 else 0) +
      (if c2 then 1 else 0) + (if c3 then 1 else 0) + (if c4 then 1 else 0) +
      (if c5 then 1 else 0) := by
  revert h
  cases c0 <;> cases c1 <;> cases c2 <;> cases c3 <;> cases c4 <;> cases c5 <;>
    decide

theorem degpos7 (c0 c1 c2 c3 c4 c5 c6 : Bool)
    (h : (c0 || c1 || c2 || c3 || c4 || c5 || c6) = true) :
    (1 : ℤ) ≤ (if c0 then 1 else 0) + (if c1 then 1 else 0) +
      (if c2 then 1 else 0) + (if c3 then 1 else 0) + (if c4 then 1 else 0) +
      (if c5 then 1 else 0) + (if c6 then 1 else 0) := by
  revert h
  cases c0 <;> cases c1 <;> cases c2 <;> cases c3 <;> cases c4 <;> cases c5 <;>
    cases c6 <;> decide

/-! ## Splitting an "and not" indicator -/

/-- `[not A and B and C] = [B and C] - [B and C and A]` - the step that turns
the count of positions where the second gear is the LOWEST blocker into
`|P| - (what the one gear below covers)`. -/
theorem ind_low2 (A B C : Bool) :
    (if !A && B && C then (1 : ℤ) else 0)
      = (if B && C then 1 else 0) - (if B && C && A then 1 else 0) := by
  cases A <;> cases B <;> cases C <;> decide

theorem ind_nonneg (b : Bool) : (0 : ℤ) ≤ if b then 1 else 0 := by
  cases b <;> decide

end CaseSplit
