/-
INCREMENT-WIDTH CERTIFICATE, step 23->29, case 31 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_23_29.json, which re-derives every number
from the primes alone).

Machine 29, INCREMENT width 49 = F_2(23) + s_min(29) = 39 + 10,
held gears [5, 7] at phases [4, 3].  Free gears [11, 13, 17, 19, 23, 29].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 2.
-/
import IncCert29B

namespace IncCert29

/-! ### case 31: held gears at phases [4, 3] -/

def p31 : List ℕ := [1, 4, 6, 8, 9, 11, 13, 14, 16, 18, 21, 23, 28, 29, 34, 36, 39, 41, 43, 44, 46, 48]
def q31 (t : ℕ) : ℕ := p31.getD t 0
def n31 : ℕ := 22
def yl31 : List ℤ := [0, 0, 1, 0, 1, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0]
def w31 (t : ℕ) : ℤ := yl31.getD t 0
def ul31 : List ℤ := [0, (-1), (-1), (-1), (-1), (-2), (-1), 0, (-2), (-1), (-1), (-2), (-1), 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, (-3), (-2), (-1), (-2), (-3), (-3), (-3), (-2), 0, (-2), (-3), (-3), (-3), (-3), (-2), (-2), (-3), 2, 2, 2, 1, (-1), 0, 2, 3, 2, 0, 0, 0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, 0, (-1), (-1), (-1), 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, (-2), (-2), (-1), 0, (-2), (-2), (-2), 0, (-2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 9, 10, 10, 10, 9, 10, 8, 10, 10, 7, 7, (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), 6, 5, 7, 3, 3, 8, 8, 8, 8, 3, 8, 8, 8, 8, 7, 5, 6, 5, 7, (-8), (-8), (-8), (-8), (-8), (-8), (-8), (-8), (-8), (-8), (-8), (-8), (-8), 4, 4, 0, 4, 2, 0, 4, 0, 3, 4, 0, 4, 0, 4, 4, 4, 3, 0, 2, 4, 0, 4, 4, (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 0, 2, 2, 4, 4, 3, 3, 2, 4, 4, 3, 6, 2, 6, 4, 6, 6, 6, 6, 2, 3, 6, 4, 5, 2, 3, 3, (-2), 1, 2, 3, 3, 2, 3, 2, 3, 3, 3, 3, 0, 3, 3, (-2), 3, 2, 3, 3, 0, 3, 0]
def u31 (k : ℕ) : ℤ := ul31.getD k 0

def c31_0 (r t : ℕ) : Bool := gb11 r (q31 t)
def c31_1 (r t : ℕ) : Bool := gb13 r (q31 t)
def c31_2 (r t : ℕ) : Bool := gb17 r (q31 t)
def c31_3 (r t : ℕ) : Bool := gb19 r (q31 t)
def c31_4 (r t : ℕ) : Bool := gb23 r (q31 t)
def c31_5 (r t : ℕ) : Bool := gb29 r (q31 t)

def S31_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (w31 t + 2) * (if c31_0 r t then 1 else 0)
def S31_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (w31 t + 2) * (if c31_1 r t then 1 else 0)
def S31_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (w31 t + 2) * (if c31_2 r t then 1 else 0)
def S31_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (w31 t + 2) * (if c31_3 r t then 1 else 0)
def S31_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (w31 t + 2) * (if c31_4 r t then 1 else 0)
def S31_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (w31 t + 2) * (if c31_5 r t then 1 else 0)

def L31_0 (r : ℕ) : ℤ := u31 (13 + r) + u31 (41 + r) + u31 (71 + r) + u31 (105 + r) + u31 (145 + r)
def L31_1 (r : ℕ) : ℤ := u31 (0 + r) + u31 (173 + r) + u31 (205 + r) + u31 (241 + r) + u31 (283 + r)
def L31_2 (r : ℕ) : ℤ := u31 (24 + r) + u31 (156 + r) + u31 (315 + r) + u31 (355 + r) + u31 (401 + r)
def L31_3 (r : ℕ) : ℤ := u31 (52 + r) + u31 (186 + r) + u31 (296 + r) + u31 (441 + r) + u31 (489 + r)
def L31_4 (r : ℕ) : ℤ := u31 (82 + r) + u31 (218 + r) + u31 (332 + r) + u31 (418 + r) + u31 (537 + r)
def L31_5 (r : ℕ) : ℤ := u31 (116 + r) + u31 (254 + r) + u31 (372 + r) + u31 (460 + r) + u31 (508 + r)

def aS31_0 (r : ℕ) : ℤ := S31_0 r - L31_0 r
def MS31_0 : ℤ := CaseSplit.mxr (aS31_0) 10
def aS31_1 (r : ℕ) : ℤ := S31_1 r - L31_1 r
def MS31_1 : ℤ := CaseSplit.mxr (aS31_1) 12
def aS31_2 (r : ℕ) : ℤ := S31_2 r - L31_2 r
def MS31_2 : ℤ := CaseSplit.mxr (aS31_2) 16
def aS31_3 (r : ℕ) : ℤ := S31_3 r - L31_3 r
def MS31_3 : ℤ := CaseSplit.mxr (aS31_3) 18
def aS31_4 (r : ℕ) : ℤ := S31_4 r - L31_4 r
def MS31_4 : ℤ := CaseSplit.mxr (aS31_4) 22
def aS31_5 (r : ℕ) : ℤ := S31_5 r - L31_5 r
def MS31_5 : ℤ := CaseSplit.mxr (aS31_5) 28

def N31_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_0 ra t && c31_1 rb t then 1 else 0)
def aP31_0 (ra rb : ℕ) : ℤ := -(2) * N31_0 ra rb + u31 (0 + rb) + u31 (13 + ra)
def MP31_0 : ℤ := CaseSplit.mxr2 (aP31_0) 10 12
def N31_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_0 ra t && c31_2 rb t then 1 else 0)
def aP31_1 (ra rb : ℕ) : ℤ := -(2) * N31_1 ra rb + u31 (24 + rb) + u31 (41 + ra)
def MP31_1 : ℤ := CaseSplit.mxr2 (aP31_1) 10 16
def N31_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_0 ra t && c31_3 rb t then 1 else 0)
def aP31_2 (ra rb : ℕ) : ℤ := -(2) * N31_2 ra rb + u31 (52 + rb) + u31 (71 + ra)
def MP31_2 : ℤ := CaseSplit.mxr2 (aP31_2) 10 18
def N31_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_0 ra t && c31_4 rb t then 1 else 0)
def aP31_3 (ra rb : ℕ) : ℤ := -(2) * N31_3 ra rb + u31 (82 + rb) + u31 (105 + ra)
def MP31_3 : ℤ := CaseSplit.mxr2 (aP31_3) 10 22
def N31_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_0 ra t && c31_5 rb t then 1 else 0)
def aP31_4 (ra rb : ℕ) : ℤ := -(2) * N31_4 ra rb + u31 (116 + rb) + u31 (145 + ra)
def MP31_4 : ℤ := CaseSplit.mxr2 (aP31_4) 10 28
def P31_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_1 ra t && c31_2 rb t then 1 else 0)
def C31_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_1 ra t && c31_2 rb t && c31_0 s t then 1 else 0)
def M31_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C31_5 ra rb) 10
def E31_5 : List ℕ := [3, 9, 93, 99, 129, 135, 138, 144, 172, 183]
def N31_5 (ra rb : ℕ) : ℤ := if E31_5.contains (ra * 17 + rb) = true then P31_5 ra rb - M31_5 ra rb else 0
def aP31_5 (ra rb : ℕ) : ℤ := -(2) * N31_5 ra rb + u31 (156 + rb) + u31 (173 + ra)
def MP31_5 : ℤ := CaseSplit.mxr2 (aP31_5) 12 16
def P31_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_1 ra t && c31_3 rb t then 1 else 0)
def C31_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_1 ra t && c31_3 rb t && c31_0 s t then 1 else 0)
def M31_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C31_6 ra rb) 10
def E31_6 : List ℕ := [13, 21, 37, 58, 71, 113, 134, 147, 158, 184, 192, 234]
def N31_6 (ra rb : ℕ) : ℤ := if E31_6.contains (ra * 19 + rb) = true then P31_6 ra rb - M31_6 ra rb else 0
def aP31_6 (ra rb : ℕ) : ℤ := -(2) * N31_6 ra rb + u31 (186 + rb) + u31 (205 + ra)
def MP31_6 : ℤ := CaseSplit.mxr2 (aP31_6) 12 18
def P31_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_1 ra t && c31_4 rb t then 1 else 0)
def C31_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_1 ra t && c31_4 rb t && c31_0 s t then 1 else 0)
def M31_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C31_7 ra rb) 10
def E31_7 : List ℕ := []
def N31_7 (ra rb : ℕ) : ℤ := if E31_7.contains (ra * 23 + rb) = true then P31_7 ra rb - M31_7 ra rb else 0
def aP31_7 (ra rb : ℕ) : ℤ := -(2) * N31_7 ra rb + u31 (218 + rb) + u31 (241 + ra)
def MP31_7 : ℤ := CaseSplit.mxr2 (aP31_7) 12 22
def P31_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_1 ra t && c31_5 rb t then 1 else 0)
def C31_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_1 ra t && c31_5 rb t && c31_0 s t then 1 else 0)
def M31_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C31_8 ra rb) 10
def E31_8 : List ℕ := [73, 189, 223, 339]
def N31_8 (ra rb : ℕ) : ℤ := if E31_8.contains (ra * 29 + rb) = true then P31_8 ra rb - M31_8 ra rb else 0
def aP31_8 (ra rb : ℕ) : ℤ := -(2) * N31_8 ra rb + u31 (254 + rb) + u31 (283 + ra)
def MP31_8 : ℤ := CaseSplit.mxr2 (aP31_8) 12 28
def N31_9 (_ra _rb : ℕ) : ℤ := 0
def aP31_9 (ra rb : ℕ) : ℤ := -(2) * N31_9 ra rb + u31 (296 + rb) + u31 (315 + ra)
def MP31_9 : ℤ := CaseSplit.mxr2 (aP31_9) 16 18
def N31_10 (_ra _rb : ℕ) : ℤ := 0
def aP31_10 (ra rb : ℕ) : ℤ := -(2) * N31_10 ra rb + u31 (332 + rb) + u31 (355 + ra)
def MP31_10 : ℤ := CaseSplit.mxr2 (aP31_10) 16 22
def N31_11 (_ra _rb : ℕ) : ℤ := 0
def aP31_11 (ra rb : ℕ) : ℤ := -(2) * N31_11 ra rb + u31 (372 + rb) + u31 (401 + ra)
def MP31_11 : ℤ := CaseSplit.mxr2 (aP31_11) 16 28
def N31_12 (_ra _rb : ℕ) : ℤ := 0
def aP31_12 (ra rb : ℕ) : ℤ := -(2) * N31_12 ra rb + u31 (418 + rb) + u31 (441 + ra)
def MP31_12 : ℤ := CaseSplit.mxr2 (aP31_12) 18 22
def N31_13 (_ra _rb : ℕ) : ℤ := 0
def aP31_13 (ra rb : ℕ) : ℤ := -(2) * N31_13 ra rb + u31 (460 + rb) + u31 (489 + ra)
def MP31_13 : ℤ := CaseSplit.mxr2 (aP31_13) 18 28
def N31_14 (_ra _rb : ℕ) : ℤ := 0
def aP31_14 (ra rb : ℕ) : ℤ := -(2) * N31_14 ra rb + u31 (508 + rb) + u31 (537 + ra)
def MP31_14 : ℤ := CaseSplit.mxr2 (aP31_14) 22 28

def rhs31 : ℤ := (∑ t ∈ Finset.range n31, w31 t) + 2 * (n31 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn31 : ∀ t, t < n31 → (0 : ℤ) ≤ w31 t := by decide
theorem plt31 : ∀ t, t < n31 → q31 t < 49 := by decide
theorem pfree31_5 : ∀ t, t < n31 → gb5 4 (q31 t) = false := by decide
theorem pfree31_7 : ∀ t, t < n31 → gb7 3 (q31 t) = false := by decide
theorem MSv31_0 : MS31_0 = 9 := by decide +kernel
theorem MSv31_1 : MS31_1 = 32 := by decide +kernel
theorem MSv31_2 : MS31_2 = 0 := by decide +kernel
theorem MSv31_3 : MS31_3 = 0 := by decide +kernel
theorem MSv31_4 : MS31_4 = 0 := by decide +kernel
theorem MSv31_5 : MS31_5 = 0 := by decide +kernel
theorem MPv31_0 : MP31_0 = 0 := by decide +kernel
theorem MPv31_1 : MP31_1 = 0 := by decide +kernel
theorem MPv31_2 : MP31_2 = 0 := by decide +kernel
theorem MPv31_3 : MP31_3 = 0 := by decide +kernel
theorem MPv31_4 : MP31_4 = 0 := by decide +kernel
theorem MPv31_5 : MP31_5 = 0 := by decide +kernel
theorem MPv31_6 : MP31_6 = 0 := by decide +kernel
theorem MPv31_7 : MP31_7 = 0 := by decide +kernel
theorem MPv31_8 : MP31_8 = 0 := by decide +kernel
theorem MPv31_9 : MP31_9 = 0 := by decide +kernel
theorem MPv31_10 : MP31_10 = 0 := by decide +kernel
theorem MPv31_11 : MP31_11 = 0 := by decide +kernel
theorem MPv31_12 : MP31_12 = 0 := by decide +kernel
theorem MPv31_13 : MP31_13 = 0 := by decide +kernel
theorem MPv31_14 : MP31_14 = 9 := by decide +kernel
theorem rhsv31 : rhs31 = 52 := by decide +kernel

/-- **The case-31 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 2/2.
    (Scaled by the common denominator 2: 50 < 52.) -/
theorem cert31 : MS31_0 + MS31_1 + MS31_2 + MS31_3 + MS31_4 + MS31_5 + MP31_0 + MP31_1 + MP31_2 + MP31_3 + MP31_4 + MP31_5 + MP31_6 + MP31_7 + MP31_8 + MP31_9 + MP31_10 + MP31_11 + MP31_12 + MP31_13 + MP31_14 < rhs31 := by
  rw [MSv31_0, MSv31_1, MSv31_2, MSv31_3, MSv31_4, MSv31_5, MPv31_0, MPv31_1, MPv31_2, MPv31_3, MPv31_4, MPv31_5, MPv31_6, MPv31_7, MPv31_8, MPv31_9, MPv31_10, MPv31_11, MPv31_12, MPv31_13, MPv31_14, rhsv31]
  decide

def Dg31 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := (if c31_0 r0 t then 1 else 0) + (if c31_1 r1 t then 1 else 0) + (if c31_2 r2 t then 1 else 0) + (if c31_3 r3 t then 1 else 0) + (if c31_4 r4 t then 1 else 0) + (if c31_5 r5 t then 1 else 0)
def Wl31_0 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c31_0 r0 t && c31_1 r1 t then 1 else 0
def Wl31_1 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c31_0 r0 t && c31_2 r2 t then 1 else 0
def Wl31_2 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c31_0 r0 t && c31_3 r3 t then 1 else 0
def Wl31_3 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c31_0 r0 t && c31_4 r4 t then 1 else 0
def Wl31_4 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c31_0 r0 t && c31_5 r5 t then 1 else 0
def Wl31_5 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c31_0 r0 t && c31_1 r1 t && c31_2 r2 t then 1 else 0
def Wl31_6 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c31_0 r0 t && c31_1 r1 t && c31_3 r3 t then 1 else 0
def Wl31_7 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c31_0 r0 t && c31_1 r1 t && c31_4 r4 t then 1 else 0
def Wl31_8 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c31_0 r0 t && c31_1 r1 t && c31_5 r5 t then 1 else 0
def Wl31_9 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c31_0 r0 t && !c31_1 r1 t && c31_2 r2 t && c31_3 r3 t then 1 else 0
def Wl31_10 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c31_0 r0 t && !c31_1 r1 t && c31_2 r2 t && c31_4 r4 t then 1 else 0
def Wl31_11 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c31_0 r0 t && !c31_1 r1 t && c31_2 r2 t && c31_5 r5 t then 1 else 0
def Wl31_12 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c31_0 r0 t && !c31_1 r1 t && !c31_2 r2 t && c31_3 r3 t && c31_4 r4 t then 1 else 0
def Wl31_13 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c31_0 r0 t && !c31_1 r1 t && !c31_2 r2 t && c31_3 r3 t && c31_5 r5 t then 1 else 0
def Wl31_14 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c31_0 r0 t && !c31_1 r1 t && !c31_2 r2 t && !c31_3 r3 t && c31_4 r4 t && c31_5 r5 t then 1 else 0

/-- **No configuration blocks the whole window in case 31.** -/
theorem nocov31 {r0 r1 r2 r3 r4 r5 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29)
    (hcov : ∀ t, t < n31 → (c31_0 r0 t || c31_1 r1 t || c31_2 r2 t || c31_3 r3 t || c31_4 r4 t || c31_5 r5 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n31, (1 : ℤ) + (Wl31_0 r0 r1 r2 r3 r4 r5 t + Wl31_1 r0 r1 r2 r3 r4 r5 t + Wl31_2 r0 r1 r2 r3 r4 r5 t + Wl31_3 r0 r1 r2 r3 r4 r5 t + Wl31_4 r0 r1 r2 r3 r4 r5 t + Wl31_5 r0 r1 r2 r3 r4 r5 t + Wl31_6 r0 r1 r2 r3 r4 r5 t + Wl31_7 r0 r1 r2 r3 r4 r5 t + Wl31_8 r0 r1 r2 r3 r4 r5 t + Wl31_9 r0 r1 r2 r3 r4 r5 t + Wl31_10 r0 r1 r2 r3 r4 r5 t + Wl31_11 r0 r1 r2 r3 r4 r5 t + Wl31_12 r0 r1 r2 r3 r4 r5 t + Wl31_13 r0 r1 r2 r3 r4 r5 t + Wl31_14 r0 r1 r2 r3 r4 r5 t) ≤ Dg31 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Wl31_0, Wl31_1, Wl31_2, Wl31_3, Wl31_4, Wl31_5, Wl31_6, Wl31_7, Wl31_8, Wl31_9, Wl31_10, Wl31_11, Wl31_12, Wl31_13, Wl31_14, Dg31]
    exact CaseSplit.lowest6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n31, (1 : ℤ) ≤ Dg31 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Dg31]
    exact CaseSplit.degpos6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n31 : ℤ) + ((∑ t ∈ Finset.range n31, Wl31_0 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n31, Wl31_1 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n31, Wl31_2 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n31, Wl31_3 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n31, Wl31_4 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n31, Wl31_5 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n31, Wl31_6 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n31, Wl31_7 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n31, Wl31_8 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n31, Wl31_9 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n31, Wl31_10 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n31, Wl31_11 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n31, Wl31_12 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n31, Wl31_13 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n31, Wl31_14 r0 r1 r2 r3 r4 r5 t)) ≤ ∑ t ∈ Finset.range n31, Dg31 r0 r1 r2 r3 r4 r5 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N31_0 r0 r1 ≤ ∑ t ∈ Finset.range n31, Wl31_0 r0 r1 r2 r3 r4 r5 t := by
    simp only [N31_0, Wl31_0, le_refl]
  have hn1 : N31_1 r0 r2 ≤ ∑ t ∈ Finset.range n31, Wl31_1 r0 r1 r2 r3 r4 r5 t := by
    simp only [N31_1, Wl31_1, le_refl]
  have hn2 : N31_2 r0 r3 ≤ ∑ t ∈ Finset.range n31, Wl31_2 r0 r1 r2 r3 r4 r5 t := by
    simp only [N31_2, Wl31_2, le_refl]
  have hn3 : N31_3 r0 r4 ≤ ∑ t ∈ Finset.range n31, Wl31_3 r0 r1 r2 r3 r4 r5 t := by
    simp only [N31_3, Wl31_3, le_refl]
  have hn4 : N31_4 r0 r5 ≤ ∑ t ∈ Finset.range n31, Wl31_4 r0 r1 r2 r3 r4 r5 t := by
    simp only [N31_4, Wl31_4, le_refl]
  have hn5 : N31_5 r1 r2 ≤ ∑ t ∈ Finset.range n31, Wl31_5 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n31, Wl31_5 r0 r1 r2 r3 r4 r5 t
        = (if c31_1 r1 t && c31_2 r2 t then (1:ℤ) else 0)
          - (if c31_1 r1 t && c31_2 r2 t && c31_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl31_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n31, Wl31_5 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl31_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n31, Wl31_5 r0 r1 r2 r3 r4 r5 t
        = P31_5 r1 r2 - C31_5 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P31_5, C31_5]
    have hm : C31_5 r1 r2 r0 ≤ M31_5 r1 r2 :=
      CaseSplit.le_mxr (C31_5 r1 r2) 10 r0 (by omega)
    simp only [N31_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N31_6 r1 r3 ≤ ∑ t ∈ Finset.range n31, Wl31_6 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n31, Wl31_6 r0 r1 r2 r3 r4 r5 t
        = (if c31_1 r1 t && c31_3 r3 t then (1:ℤ) else 0)
          - (if c31_1 r1 t && c31_3 r3 t && c31_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl31_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n31, Wl31_6 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl31_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n31, Wl31_6 r0 r1 r2 r3 r4 r5 t
        = P31_6 r1 r3 - C31_6 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P31_6, C31_6]
    have hm : C31_6 r1 r3 r0 ≤ M31_6 r1 r3 :=
      CaseSplit.le_mxr (C31_6 r1 r3) 10 r0 (by omega)
    simp only [N31_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N31_7 r1 r4 ≤ ∑ t ∈ Finset.range n31, Wl31_7 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n31, Wl31_7 r0 r1 r2 r3 r4 r5 t
        = (if c31_1 r1 t && c31_4 r4 t then (1:ℤ) else 0)
          - (if c31_1 r1 t && c31_4 r4 t && c31_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl31_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n31, Wl31_7 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl31_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n31, Wl31_7 r0 r1 r2 r3 r4 r5 t
        = P31_7 r1 r4 - C31_7 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P31_7, C31_7]
    have hm : C31_7 r1 r4 r0 ≤ M31_7 r1 r4 :=
      CaseSplit.le_mxr (C31_7 r1 r4) 10 r0 (by omega)
    simp only [N31_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N31_8 r1 r5 ≤ ∑ t ∈ Finset.range n31, Wl31_8 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n31, Wl31_8 r0 r1 r2 r3 r4 r5 t
        = (if c31_1 r1 t && c31_5 r5 t then (1:ℤ) else 0)
          - (if c31_1 r1 t && c31_5 r5 t && c31_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl31_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n31, Wl31_8 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl31_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n31, Wl31_8 r0 r1 r2 r3 r4 r5 t
        = P31_8 r1 r5 - C31_8 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P31_8, C31_8]
    have hm : C31_8 r1 r5 r0 ≤ M31_8 r1 r5 :=
      CaseSplit.le_mxr (C31_8 r1 r5) 10 r0 (by omega)
    simp only [N31_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N31_9 r2 r3 ≤ ∑ t ∈ Finset.range n31, Wl31_9 r0 r1 r2 r3 r4 r5 t := by
    simp only [N31_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl31_9]
    exact CaseSplit.ind_nonneg _
  have hn10 : N31_10 r2 r4 ≤ ∑ t ∈ Finset.range n31, Wl31_10 r0 r1 r2 r3 r4 r5 t := by
    simp only [N31_10]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl31_10]
    exact CaseSplit.ind_nonneg _
  have hn11 : N31_11 r2 r5 ≤ ∑ t ∈ Finset.range n31, Wl31_11 r0 r1 r2 r3 r4 r5 t := by
    simp only [N31_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl31_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N31_12 r3 r4 ≤ ∑ t ∈ Finset.range n31, Wl31_12 r0 r1 r2 r3 r4 r5 t := by
    simp only [N31_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl31_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N31_13 r3 r5 ≤ ∑ t ∈ Finset.range n31, Wl31_13 r0 r1 r2 r3 r4 r5 t := by
    simp only [N31_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl31_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N31_14 r4 r5 ≤ ∑ t ∈ Finset.range n31, Wl31_14 r0 r1 r2 r3 r4 r5 t := by
    simp only [N31_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl31_14]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n31, (w31 t + 2) * Dg31 r0 r1 r2 r3 r4 r5 t = S31_0 r0 + S31_1 r1 + S31_2 r2 + S31_3 r3 + S31_4 r4 + S31_5 r5 := by
    simp only [S31_0, S31_1, S31_2, S31_3, S31_4, S31_5, Dg31, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n31, (w31 t + 2) * Dg31 r0 r1 r2 r3 r4 r5 t
      = (∑ t ∈ Finset.range n31, w31 t * Dg31 r0 r1 r2 r3 r4 r5 t)
        + 2 * (∑ t ∈ Finset.range n31, Dg31 r0 r1 r2 r3 r4 r5 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n31, w31 t)
      ≤ ∑ t ∈ Finset.range n31, w31 t * Dg31 r0 r1 r2 r3 r4 r5 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg31 r0 r1 r2 r3 r4 r5 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w31 t := wnn31 t (Finset.mem_range.mp ht)
    calc w31 t = w31 t * 1 := (mul_one _).symm
      _ ≤ w31 t * Dg31 r0 r1 r2 r3 r4 r5 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS31_0 r0 + aS31_1 r1 + aS31_2 r2 + aS31_3 r3 + aS31_4 r4 + aS31_5 r5) + (aP31_0 r0 r1 + aP31_1 r0 r2 + aP31_2 r0 r3 + aP31_3 r0 r4 + aP31_4 r0 r5 + aP31_5 r1 r2 + aP31_6 r1 r3 + aP31_7 r1 r4 + aP31_8 r1 r5 + aP31_9 r2 r3 + aP31_10 r2 r4 + aP31_11 r2 r5 + aP31_12 r3 r4 + aP31_13 r3 r5 + aP31_14 r4 r5) = (S31_0 r0 + S31_1 r1 + S31_2 r2 + S31_3 r3 + S31_4 r4 + S31_5 r5) - 2 * (N31_0 r0 r1 + N31_1 r0 r2 + N31_2 r0 r3 + N31_3 r0 r4 + N31_4 r0 r5 + N31_5 r1 r2 + N31_6 r1 r3 + N31_7 r1 r4 + N31_8 r1 r5 + N31_9 r2 r3 + N31_10 r2 r4 + N31_11 r2 r5 + N31_12 r3 r4 + N31_13 r3 r5 + N31_14 r4 r5) := by
    simp only [aS31_0, aS31_1, aS31_2, aS31_3, aS31_4, aS31_5, aP31_0, aP31_1, aP31_2, aP31_3, aP31_4, aP31_5, aP31_6, aP31_7, aP31_8, aP31_9, aP31_10, aP31_11, aP31_12, aP31_13, aP31_14, L31_0, L31_1, L31_2, L31_3, L31_4, L31_5]
    ring
  have bS0 : aS31_0 r0 ≤ MS31_0 := CaseSplit.le_mxr (aS31_0) 10 r0 (by omega)
  have bS1 : aS31_1 r1 ≤ MS31_1 := CaseSplit.le_mxr (aS31_1) 12 r1 (by omega)
  have bS2 : aS31_2 r2 ≤ MS31_2 := CaseSplit.le_mxr (aS31_2) 16 r2 (by omega)
  have bS3 : aS31_3 r3 ≤ MS31_3 := CaseSplit.le_mxr (aS31_3) 18 r3 (by omega)
  have bS4 : aS31_4 r4 ≤ MS31_4 := CaseSplit.le_mxr (aS31_4) 22 r4 (by omega)
  have bS5 : aS31_5 r5 ≤ MS31_5 := CaseSplit.le_mxr (aS31_5) 28 r5 (by omega)
  have bP0 : aP31_0 r0 r1 ≤ MP31_0 := CaseSplit.le_mxr2 (aP31_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP31_1 r0 r2 ≤ MP31_1 := CaseSplit.le_mxr2 (aP31_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP31_2 r0 r3 ≤ MP31_2 := CaseSplit.le_mxr2 (aP31_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP31_3 r0 r4 ≤ MP31_3 := CaseSplit.le_mxr2 (aP31_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP31_4 r0 r5 ≤ MP31_4 := CaseSplit.le_mxr2 (aP31_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP31_5 r1 r2 ≤ MP31_5 := CaseSplit.le_mxr2 (aP31_5) 12 16 r1 r2 (by omega) (by omega)
  have bP6 : aP31_6 r1 r3 ≤ MP31_6 := CaseSplit.le_mxr2 (aP31_6) 12 18 r1 r3 (by omega) (by omega)
  have bP7 : aP31_7 r1 r4 ≤ MP31_7 := CaseSplit.le_mxr2 (aP31_7) 12 22 r1 r4 (by omega) (by omega)
  have bP8 : aP31_8 r1 r5 ≤ MP31_8 := CaseSplit.le_mxr2 (aP31_8) 12 28 r1 r5 (by omega) (by omega)
  have bP9 : aP31_9 r2 r3 ≤ MP31_9 := CaseSplit.le_mxr2 (aP31_9) 16 18 r2 r3 (by omega) (by omega)
  have bP10 : aP31_10 r2 r4 ≤ MP31_10 := CaseSplit.le_mxr2 (aP31_10) 16 22 r2 r4 (by omega) (by omega)
  have bP11 : aP31_11 r2 r5 ≤ MP31_11 := CaseSplit.le_mxr2 (aP31_11) 16 28 r2 r5 (by omega) (by omega)
  have bP12 : aP31_12 r3 r4 ≤ MP31_12 := CaseSplit.le_mxr2 (aP31_12) 18 22 r3 r4 (by omega) (by omega)
  have bP13 : aP31_13 r3 r5 ≤ MP31_13 := CaseSplit.le_mxr2 (aP31_13) 18 28 r3 r5 (by omega) (by omega)
  have bP14 : aP31_14 r4 r5 ≤ MP31_14 := CaseSplit.le_mxr2 (aP31_14) 22 28 r4 r5 (by omega) (by omega)
  have hrhs : rhs31 = (∑ t ∈ Finset.range n31, w31 t) + 2 * (n31 : ℤ) := rfl
  have hc := cert31
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, bS0, bS1, bS2, bS3, bS4, bS5, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14]

end IncCert29
