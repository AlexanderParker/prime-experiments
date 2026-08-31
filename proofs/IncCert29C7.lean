/-
INCREMENT-WIDTH CERTIFICATE, step 23->29, case 7 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_23_29.json, which re-derives every number
from the primes alone).

Machine 29, INCREMENT width 49 = F_2(23) + s_min(29) = 39 + 10,
held gears [5, 7] at phases [1, 0].  Free gears [11, 13, 17, 19, 23, 29].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 6.
-/
import IncCert29B

namespace IncCert29

/-! ### case 7: held gears at phases [1, 0] -/

def p7 : List ℕ := [2, 4, 7, 9, 11, 12, 14, 16, 17, 19, 21, 24, 26, 31, 32, 37, 39, 42, 44, 46, 47]
def q7 (t : ℕ) : ℕ := p7.getD t 0
def n7 : ℕ := 21
def yl7 : List ℤ := [0, 0, 0, 4, 0, 3, 5, 4, 5, 5, 2, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0]
def w7 (t : ℕ) : ℤ := yl7.getD t 0
def ul7 : List ℤ := [(-1), (-5), 0, (-5), (-1), (-5), (-5), 0, (-4), (-4), (-5), 0, (-5), 0, 0, 0, 1, 0, 4, 0, 5, 0, 4, 0, 0, 0, 0, (-2), (-2), 1, 0, 0, 0, (-2), (-2), 0, 0, 0, (-2), (-2), (-2), 2, (-1), (-1), 0, 0, 0, (-1), (-1), (-1), 0, 0, (-3), (-3), (-3), (-3), 0, (-3), (-3), (-3), (-3), (-3), 0, (-1), 0, (-3), (-3), (-3), (-3), (-1), 0, 3, 0, (-5), 0, 1, 0, 0, 0, (-6), 3, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, (-6), (-6), 0, 0, (-6), (-6), 0, (-6), (-6), 0, 0, (-4), (-4), 0, (-4), (-4), (-4), 0, (-4), (-4), 0, 0, (-4), (-4), (-4), 0, (-4), 0, (-4), (-4), (-4), 0, 0, (-4), (-4), 0, (-4), (-4), (-4), 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 22, 15, 18, 18, 22, 22, 22, 15, 15, 22, 22, 22, 12, 12, 22, 22, (-22), (-22), (-22), (-22), (-22), (-22), (-22), (-22), (-22), (-22), (-22), (-22), (-22), 12, 23, 21, 20, 21, 20, 23, 19, 20, 22, 23, 23, 12, 23, 23, 23, 21, 23, 23, (-23), (-23), (-23), (-23), (-23), (-23), (-23), (-23), (-23), (-23), (-23), (-23), (-23), 8, 8, 8, 8, 8, 2, 0, 0, 8, 0, 7, 8, 8, 8, 0, 8, (-2), 0, 8, 8, 0, 0, 0, (-8), (-8), (-8), (-8), (-8), (-8), (-8), (-8), (-8), (-8), (-8), (-8), (-8), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 12, 12, 23, 6, 17, 12, 18, 22, 13, 18, 7, 15, 20, 13, 23, 18, 23, 15, 13, 23, 14, 23, 12, 13, 12, 12, 18, 6, 15, (-1), 9, 15, (-2), 15, 15, 10, 15, 0, 15, 15, (-2), 9, 0, 7, 15, 6, 15, 4, 6, 15, 0]
def u7 (k : ℕ) : ℤ := ul7.getD k 0

def c7_0 (r t : ℕ) : Bool := gb11 r (q7 t)
def c7_1 (r t : ℕ) : Bool := gb13 r (q7 t)
def c7_2 (r t : ℕ) : Bool := gb17 r (q7 t)
def c7_3 (r t : ℕ) : Bool := gb19 r (q7 t)
def c7_4 (r t : ℕ) : Bool := gb23 r (q7 t)
def c7_5 (r t : ℕ) : Bool := gb29 r (q7 t)

def S7_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (w7 t + 6) * (if c7_0 r t then 1 else 0)
def S7_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (w7 t + 6) * (if c7_1 r t then 1 else 0)
def S7_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (w7 t + 6) * (if c7_2 r t then 1 else 0)
def S7_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (w7 t + 6) * (if c7_3 r t then 1 else 0)
def S7_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (w7 t + 6) * (if c7_4 r t then 1 else 0)
def S7_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (w7 t + 6) * (if c7_5 r t then 1 else 0)

def L7_0 (r : ℕ) : ℤ := u7 (13 + r) + u7 (41 + r) + u7 (71 + r) + u7 (105 + r) + u7 (145 + r)
def L7_1 (r : ℕ) : ℤ := u7 (0 + r) + u7 (173 + r) + u7 (205 + r) + u7 (241 + r) + u7 (283 + r)
def L7_2 (r : ℕ) : ℤ := u7 (24 + r) + u7 (156 + r) + u7 (315 + r) + u7 (355 + r) + u7 (401 + r)
def L7_3 (r : ℕ) : ℤ := u7 (52 + r) + u7 (186 + r) + u7 (296 + r) + u7 (441 + r) + u7 (489 + r)
def L7_4 (r : ℕ) : ℤ := u7 (82 + r) + u7 (218 + r) + u7 (332 + r) + u7 (418 + r) + u7 (537 + r)
def L7_5 (r : ℕ) : ℤ := u7 (116 + r) + u7 (254 + r) + u7 (372 + r) + u7 (460 + r) + u7 (508 + r)

def aS7_0 (r : ℕ) : ℤ := S7_0 r - L7_0 r
def MS7_0 : ℤ := CaseSplit.mxr (aS7_0) 10
def aS7_1 (r : ℕ) : ℤ := S7_1 r - L7_1 r
def MS7_1 : ℤ := CaseSplit.mxr (aS7_1) 12
def aS7_2 (r : ℕ) : ℤ := S7_2 r - L7_2 r
def MS7_2 : ℤ := CaseSplit.mxr (aS7_2) 16
def aS7_3 (r : ℕ) : ℤ := S7_3 r - L7_3 r
def MS7_3 : ℤ := CaseSplit.mxr (aS7_3) 18
def aS7_4 (r : ℕ) : ℤ := S7_4 r - L7_4 r
def MS7_4 : ℤ := CaseSplit.mxr (aS7_4) 22
def aS7_5 (r : ℕ) : ℤ := S7_5 r - L7_5 r
def MS7_5 : ℤ := CaseSplit.mxr (aS7_5) 28

def N7_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (if c7_0 ra t && c7_1 rb t then 1 else 0)
def aP7_0 (ra rb : ℕ) : ℤ := -(6) * N7_0 ra rb + u7 (0 + rb) + u7 (13 + ra)
def MP7_0 : ℤ := CaseSplit.mxr2 (aP7_0) 10 12
def N7_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (if c7_0 ra t && c7_2 rb t then 1 else 0)
def aP7_1 (ra rb : ℕ) : ℤ := -(6) * N7_1 ra rb + u7 (24 + rb) + u7 (41 + ra)
def MP7_1 : ℤ := CaseSplit.mxr2 (aP7_1) 10 16
def N7_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (if c7_0 ra t && c7_3 rb t then 1 else 0)
def aP7_2 (ra rb : ℕ) : ℤ := -(6) * N7_2 ra rb + u7 (52 + rb) + u7 (71 + ra)
def MP7_2 : ℤ := CaseSplit.mxr2 (aP7_2) 10 18
def N7_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (if c7_0 ra t && c7_4 rb t then 1 else 0)
def aP7_3 (ra rb : ℕ) : ℤ := -(6) * N7_3 ra rb + u7 (82 + rb) + u7 (105 + ra)
def MP7_3 : ℤ := CaseSplit.mxr2 (aP7_3) 10 22
def N7_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (if c7_0 ra t && c7_5 rb t then 1 else 0)
def aP7_4 (ra rb : ℕ) : ℤ := -(6) * N7_4 ra rb + u7 (116 + rb) + u7 (145 + ra)
def MP7_4 : ℤ := CaseSplit.mxr2 (aP7_4) 10 28
def P7_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (if c7_1 ra t && c7_2 rb t then 1 else 0)
def C7_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (if c7_1 ra t && c7_2 rb t && c7_0 s t then 1 else 0)
def M7_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C7_5 ra rb) 10
def E7_5 : List ℕ := [39, 45, 75, 81, 129, 135, 154, 165, 170, 176]
def N7_5 (ra rb : ℕ) : ℤ := if E7_5.contains (ra * 17 + rb) = true then P7_5 ra rb - M7_5 ra rb else 0
def aP7_5 (ra rb : ℕ) : ℤ := -(6) * N7_5 ra rb + u7 (156 + rb) + u7 (173 + ra)
def MP7_5 : ℤ := CaseSplit.mxr2 (aP7_5) 12 16
def P7_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (if c7_1 ra t && c7_3 rb t then 1 else 0)
def C7_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (if c7_1 ra t && c7_3 rb t && c7_0 s t then 1 else 0)
def M7_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C7_6 ra rb) 10
def E7_6 : List ℕ := [11, 17, 53, 87, 93, 98, 124, 151, 174, 200, 224, 227]
def N7_6 (ra rb : ℕ) : ℤ := if E7_6.contains (ra * 19 + rb) = true then P7_6 ra rb - M7_6 ra rb else 0
def aP7_6 (ra rb : ℕ) : ℤ := -(6) * N7_6 ra rb + u7 (186 + rb) + u7 (205 + ra)
def MP7_6 : ℤ := CaseSplit.mxr2 (aP7_6) 12 18
def P7_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (if c7_1 ra t && c7_4 rb t then 1 else 0)
def C7_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (if c7_1 ra t && c7_4 rb t && c7_0 s t then 1 else 0)
def M7_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C7_7 ra rb) 10
def E7_7 : List ℕ := []
def N7_7 (ra rb : ℕ) : ℤ := if E7_7.contains (ra * 23 + rb) = true then P7_7 ra rb - M7_7 ra rb else 0
def aP7_7 (ra rb : ℕ) : ℤ := -(6) * N7_7 ra rb + u7 (218 + rb) + u7 (241 + ra)
def MP7_7 : ℤ := CaseSplit.mxr2 (aP7_7) 12 22
def P7_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (if c7_1 ra t && c7_5 rb t then 1 else 0)
def C7_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (if c7_1 ra t && c7_5 rb t && c7_0 s t then 1 else 0)
def M7_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C7_8 ra rb) 10
def E7_8 : List ℕ := [133, 249]
def N7_8 (ra rb : ℕ) : ℤ := if E7_8.contains (ra * 29 + rb) = true then P7_8 ra rb - M7_8 ra rb else 0
def aP7_8 (ra rb : ℕ) : ℤ := -(6) * N7_8 ra rb + u7 (254 + rb) + u7 (283 + ra)
def MP7_8 : ℤ := CaseSplit.mxr2 (aP7_8) 12 28
def N7_9 (_ra _rb : ℕ) : ℤ := 0
def aP7_9 (ra rb : ℕ) : ℤ := -(6) * N7_9 ra rb + u7 (296 + rb) + u7 (315 + ra)
def MP7_9 : ℤ := CaseSplit.mxr2 (aP7_9) 16 18
def N7_10 (_ra _rb : ℕ) : ℤ := 0
def aP7_10 (ra rb : ℕ) : ℤ := -(6) * N7_10 ra rb + u7 (332 + rb) + u7 (355 + ra)
def MP7_10 : ℤ := CaseSplit.mxr2 (aP7_10) 16 22
def N7_11 (_ra _rb : ℕ) : ℤ := 0
def aP7_11 (ra rb : ℕ) : ℤ := -(6) * N7_11 ra rb + u7 (372 + rb) + u7 (401 + ra)
def MP7_11 : ℤ := CaseSplit.mxr2 (aP7_11) 16 28
def N7_12 (_ra _rb : ℕ) : ℤ := 0
def aP7_12 (ra rb : ℕ) : ℤ := -(6) * N7_12 ra rb + u7 (418 + rb) + u7 (441 + ra)
def MP7_12 : ℤ := CaseSplit.mxr2 (aP7_12) 18 22
def N7_13 (_ra _rb : ℕ) : ℤ := 0
def aP7_13 (ra rb : ℕ) : ℤ := -(6) * N7_13 ra rb + u7 (460 + rb) + u7 (489 + ra)
def MP7_13 : ℤ := CaseSplit.mxr2 (aP7_13) 18 28
def N7_14 (_ra _rb : ℕ) : ℤ := 0
def aP7_14 (ra rb : ℕ) : ℤ := -(6) * N7_14 ra rb + u7 (508 + rb) + u7 (537 + ra)
def MP7_14 : ℤ := CaseSplit.mxr2 (aP7_14) 22 28

def rhs7 : ℤ := (∑ t ∈ Finset.range n7, w7 t) + 6 * (n7 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn7 : ∀ t, t < n7 → (0 : ℤ) ≤ w7 t := by decide
theorem plt7 : ∀ t, t < n7 → q7 t < 49 := by decide
theorem pfree7_5 : ∀ t, t < n7 → gb5 1 (q7 t) = false := by decide
theorem pfree7_7 : ∀ t, t < n7 → gb7 0 (q7 t) = false := by decide
theorem MSv7_0 : MS7_0 = 32 := by decide +kernel
theorem MSv7_1 : MS7_1 = 84 := by decide +kernel
theorem MSv7_2 : MS7_2 = 1 := by decide +kernel
theorem MSv7_3 : MS7_3 = 1 := by decide +kernel
theorem MSv7_4 : MS7_4 = 0 := by decide +kernel
theorem MSv7_5 : MS7_5 = 0 := by decide +kernel
theorem MPv7_0 : MP7_0 = 0 := by decide +kernel
theorem MPv7_1 : MP7_1 = 0 := by decide +kernel
theorem MPv7_2 : MP7_2 = 0 := by decide +kernel
theorem MPv7_3 : MP7_3 = 0 := by decide +kernel
theorem MPv7_4 : MP7_4 = 0 := by decide +kernel
theorem MPv7_5 : MP7_5 = 0 := by decide +kernel
theorem MPv7_6 : MP7_6 = 0 := by decide +kernel
theorem MPv7_7 : MP7_7 = 0 := by decide +kernel
theorem MPv7_8 : MP7_8 = 0 := by decide +kernel
theorem MPv7_9 : MP7_9 = 0 := by decide +kernel
theorem MPv7_10 : MP7_10 = 0 := by decide +kernel
theorem MPv7_11 : MP7_11 = 0 := by decide +kernel
theorem MPv7_12 : MP7_12 = 0 := by decide +kernel
theorem MPv7_13 : MP7_13 = 0 := by decide +kernel
theorem MPv7_14 : MP7_14 = 38 := by decide +kernel
theorem rhsv7 : rhs7 = 157 := by decide +kernel

/-- **The case-7 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/6.
    (Scaled by the common denominator 6: 156 < 157.) -/
theorem cert7 : MS7_0 + MS7_1 + MS7_2 + MS7_3 + MS7_4 + MS7_5 + MP7_0 + MP7_1 + MP7_2 + MP7_3 + MP7_4 + MP7_5 + MP7_6 + MP7_7 + MP7_8 + MP7_9 + MP7_10 + MP7_11 + MP7_12 + MP7_13 + MP7_14 < rhs7 := by
  rw [MSv7_0, MSv7_1, MSv7_2, MSv7_3, MSv7_4, MSv7_5, MPv7_0, MPv7_1, MPv7_2, MPv7_3, MPv7_4, MPv7_5, MPv7_6, MPv7_7, MPv7_8, MPv7_9, MPv7_10, MPv7_11, MPv7_12, MPv7_13, MPv7_14, rhsv7]
  decide

def Dg7 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := (if c7_0 r0 t then 1 else 0) + (if c7_1 r1 t then 1 else 0) + (if c7_2 r2 t then 1 else 0) + (if c7_3 r3 t then 1 else 0) + (if c7_4 r4 t then 1 else 0) + (if c7_5 r5 t then 1 else 0)
def Wl7_0 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c7_0 r0 t && c7_1 r1 t then 1 else 0
def Wl7_1 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c7_0 r0 t && c7_2 r2 t then 1 else 0
def Wl7_2 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c7_0 r0 t && c7_3 r3 t then 1 else 0
def Wl7_3 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c7_0 r0 t && c7_4 r4 t then 1 else 0
def Wl7_4 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c7_0 r0 t && c7_5 r5 t then 1 else 0
def Wl7_5 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c7_0 r0 t && c7_1 r1 t && c7_2 r2 t then 1 else 0
def Wl7_6 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c7_0 r0 t && c7_1 r1 t && c7_3 r3 t then 1 else 0
def Wl7_7 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c7_0 r0 t && c7_1 r1 t && c7_4 r4 t then 1 else 0
def Wl7_8 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c7_0 r0 t && c7_1 r1 t && c7_5 r5 t then 1 else 0
def Wl7_9 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c7_0 r0 t && !c7_1 r1 t && c7_2 r2 t && c7_3 r3 t then 1 else 0
def Wl7_10 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c7_0 r0 t && !c7_1 r1 t && c7_2 r2 t && c7_4 r4 t then 1 else 0
def Wl7_11 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c7_0 r0 t && !c7_1 r1 t && c7_2 r2 t && c7_5 r5 t then 1 else 0
def Wl7_12 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c7_0 r0 t && !c7_1 r1 t && !c7_2 r2 t && c7_3 r3 t && c7_4 r4 t then 1 else 0
def Wl7_13 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c7_0 r0 t && !c7_1 r1 t && !c7_2 r2 t && c7_3 r3 t && c7_5 r5 t then 1 else 0
def Wl7_14 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c7_0 r0 t && !c7_1 r1 t && !c7_2 r2 t && !c7_3 r3 t && c7_4 r4 t && c7_5 r5 t then 1 else 0

/-- **No configuration blocks the whole window in case 7.** -/
theorem nocov7 {r0 r1 r2 r3 r4 r5 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29)
    (hcov : ∀ t, t < n7 → (c7_0 r0 t || c7_1 r1 t || c7_2 r2 t || c7_3 r3 t || c7_4 r4 t || c7_5 r5 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n7, (1 : ℤ) + (Wl7_0 r0 r1 r2 r3 r4 r5 t + Wl7_1 r0 r1 r2 r3 r4 r5 t + Wl7_2 r0 r1 r2 r3 r4 r5 t + Wl7_3 r0 r1 r2 r3 r4 r5 t + Wl7_4 r0 r1 r2 r3 r4 r5 t + Wl7_5 r0 r1 r2 r3 r4 r5 t + Wl7_6 r0 r1 r2 r3 r4 r5 t + Wl7_7 r0 r1 r2 r3 r4 r5 t + Wl7_8 r0 r1 r2 r3 r4 r5 t + Wl7_9 r0 r1 r2 r3 r4 r5 t + Wl7_10 r0 r1 r2 r3 r4 r5 t + Wl7_11 r0 r1 r2 r3 r4 r5 t + Wl7_12 r0 r1 r2 r3 r4 r5 t + Wl7_13 r0 r1 r2 r3 r4 r5 t + Wl7_14 r0 r1 r2 r3 r4 r5 t) ≤ Dg7 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Wl7_0, Wl7_1, Wl7_2, Wl7_3, Wl7_4, Wl7_5, Wl7_6, Wl7_7, Wl7_8, Wl7_9, Wl7_10, Wl7_11, Wl7_12, Wl7_13, Wl7_14, Dg7]
    exact CaseSplit.lowest6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n7, (1 : ℤ) ≤ Dg7 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Dg7]
    exact CaseSplit.degpos6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n7 : ℤ) + ((∑ t ∈ Finset.range n7, Wl7_0 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n7, Wl7_1 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n7, Wl7_2 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n7, Wl7_3 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n7, Wl7_4 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n7, Wl7_5 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n7, Wl7_6 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n7, Wl7_7 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n7, Wl7_8 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n7, Wl7_9 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n7, Wl7_10 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n7, Wl7_11 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n7, Wl7_12 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n7, Wl7_13 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n7, Wl7_14 r0 r1 r2 r3 r4 r5 t)) ≤ ∑ t ∈ Finset.range n7, Dg7 r0 r1 r2 r3 r4 r5 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N7_0 r0 r1 ≤ ∑ t ∈ Finset.range n7, Wl7_0 r0 r1 r2 r3 r4 r5 t := by
    simp only [N7_0, Wl7_0, le_refl]
  have hn1 : N7_1 r0 r2 ≤ ∑ t ∈ Finset.range n7, Wl7_1 r0 r1 r2 r3 r4 r5 t := by
    simp only [N7_1, Wl7_1, le_refl]
  have hn2 : N7_2 r0 r3 ≤ ∑ t ∈ Finset.range n7, Wl7_2 r0 r1 r2 r3 r4 r5 t := by
    simp only [N7_2, Wl7_2, le_refl]
  have hn3 : N7_3 r0 r4 ≤ ∑ t ∈ Finset.range n7, Wl7_3 r0 r1 r2 r3 r4 r5 t := by
    simp only [N7_3, Wl7_3, le_refl]
  have hn4 : N7_4 r0 r5 ≤ ∑ t ∈ Finset.range n7, Wl7_4 r0 r1 r2 r3 r4 r5 t := by
    simp only [N7_4, Wl7_4, le_refl]
  have hn5 : N7_5 r1 r2 ≤ ∑ t ∈ Finset.range n7, Wl7_5 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n7, Wl7_5 r0 r1 r2 r3 r4 r5 t
        = (if c7_1 r1 t && c7_2 r2 t then (1:ℤ) else 0)
          - (if c7_1 r1 t && c7_2 r2 t && c7_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl7_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n7, Wl7_5 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl7_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n7, Wl7_5 r0 r1 r2 r3 r4 r5 t
        = P7_5 r1 r2 - C7_5 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P7_5, C7_5]
    have hm : C7_5 r1 r2 r0 ≤ M7_5 r1 r2 :=
      CaseSplit.le_mxr (C7_5 r1 r2) 10 r0 (by omega)
    simp only [N7_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N7_6 r1 r3 ≤ ∑ t ∈ Finset.range n7, Wl7_6 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n7, Wl7_6 r0 r1 r2 r3 r4 r5 t
        = (if c7_1 r1 t && c7_3 r3 t then (1:ℤ) else 0)
          - (if c7_1 r1 t && c7_3 r3 t && c7_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl7_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n7, Wl7_6 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl7_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n7, Wl7_6 r0 r1 r2 r3 r4 r5 t
        = P7_6 r1 r3 - C7_6 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P7_6, C7_6]
    have hm : C7_6 r1 r3 r0 ≤ M7_6 r1 r3 :=
      CaseSplit.le_mxr (C7_6 r1 r3) 10 r0 (by omega)
    simp only [N7_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N7_7 r1 r4 ≤ ∑ t ∈ Finset.range n7, Wl7_7 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n7, Wl7_7 r0 r1 r2 r3 r4 r5 t
        = (if c7_1 r1 t && c7_4 r4 t then (1:ℤ) else 0)
          - (if c7_1 r1 t && c7_4 r4 t && c7_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl7_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n7, Wl7_7 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl7_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n7, Wl7_7 r0 r1 r2 r3 r4 r5 t
        = P7_7 r1 r4 - C7_7 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P7_7, C7_7]
    have hm : C7_7 r1 r4 r0 ≤ M7_7 r1 r4 :=
      CaseSplit.le_mxr (C7_7 r1 r4) 10 r0 (by omega)
    simp only [N7_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N7_8 r1 r5 ≤ ∑ t ∈ Finset.range n7, Wl7_8 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n7, Wl7_8 r0 r1 r2 r3 r4 r5 t
        = (if c7_1 r1 t && c7_5 r5 t then (1:ℤ) else 0)
          - (if c7_1 r1 t && c7_5 r5 t && c7_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl7_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n7, Wl7_8 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl7_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n7, Wl7_8 r0 r1 r2 r3 r4 r5 t
        = P7_8 r1 r5 - C7_8 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P7_8, C7_8]
    have hm : C7_8 r1 r5 r0 ≤ M7_8 r1 r5 :=
      CaseSplit.le_mxr (C7_8 r1 r5) 10 r0 (by omega)
    simp only [N7_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N7_9 r2 r3 ≤ ∑ t ∈ Finset.range n7, Wl7_9 r0 r1 r2 r3 r4 r5 t := by
    simp only [N7_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl7_9]
    exact CaseSplit.ind_nonneg _
  have hn10 : N7_10 r2 r4 ≤ ∑ t ∈ Finset.range n7, Wl7_10 r0 r1 r2 r3 r4 r5 t := by
    simp only [N7_10]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl7_10]
    exact CaseSplit.ind_nonneg _
  have hn11 : N7_11 r2 r5 ≤ ∑ t ∈ Finset.range n7, Wl7_11 r0 r1 r2 r3 r4 r5 t := by
    simp only [N7_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl7_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N7_12 r3 r4 ≤ ∑ t ∈ Finset.range n7, Wl7_12 r0 r1 r2 r3 r4 r5 t := by
    simp only [N7_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl7_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N7_13 r3 r5 ≤ ∑ t ∈ Finset.range n7, Wl7_13 r0 r1 r2 r3 r4 r5 t := by
    simp only [N7_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl7_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N7_14 r4 r5 ≤ ∑ t ∈ Finset.range n7, Wl7_14 r0 r1 r2 r3 r4 r5 t := by
    simp only [N7_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl7_14]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n7, (w7 t + 6) * Dg7 r0 r1 r2 r3 r4 r5 t = S7_0 r0 + S7_1 r1 + S7_2 r2 + S7_3 r3 + S7_4 r4 + S7_5 r5 := by
    simp only [S7_0, S7_1, S7_2, S7_3, S7_4, S7_5, Dg7, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n7, (w7 t + 6) * Dg7 r0 r1 r2 r3 r4 r5 t
      = (∑ t ∈ Finset.range n7, w7 t * Dg7 r0 r1 r2 r3 r4 r5 t)
        + 6 * (∑ t ∈ Finset.range n7, Dg7 r0 r1 r2 r3 r4 r5 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n7, w7 t)
      ≤ ∑ t ∈ Finset.range n7, w7 t * Dg7 r0 r1 r2 r3 r4 r5 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg7 r0 r1 r2 r3 r4 r5 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w7 t := wnn7 t (Finset.mem_range.mp ht)
    calc w7 t = w7 t * 1 := (mul_one _).symm
      _ ≤ w7 t * Dg7 r0 r1 r2 r3 r4 r5 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS7_0 r0 + aS7_1 r1 + aS7_2 r2 + aS7_3 r3 + aS7_4 r4 + aS7_5 r5) + (aP7_0 r0 r1 + aP7_1 r0 r2 + aP7_2 r0 r3 + aP7_3 r0 r4 + aP7_4 r0 r5 + aP7_5 r1 r2 + aP7_6 r1 r3 + aP7_7 r1 r4 + aP7_8 r1 r5 + aP7_9 r2 r3 + aP7_10 r2 r4 + aP7_11 r2 r5 + aP7_12 r3 r4 + aP7_13 r3 r5 + aP7_14 r4 r5) = (S7_0 r0 + S7_1 r1 + S7_2 r2 + S7_3 r3 + S7_4 r4 + S7_5 r5) - 6 * (N7_0 r0 r1 + N7_1 r0 r2 + N7_2 r0 r3 + N7_3 r0 r4 + N7_4 r0 r5 + N7_5 r1 r2 + N7_6 r1 r3 + N7_7 r1 r4 + N7_8 r1 r5 + N7_9 r2 r3 + N7_10 r2 r4 + N7_11 r2 r5 + N7_12 r3 r4 + N7_13 r3 r5 + N7_14 r4 r5) := by
    simp only [aS7_0, aS7_1, aS7_2, aS7_3, aS7_4, aS7_5, aP7_0, aP7_1, aP7_2, aP7_3, aP7_4, aP7_5, aP7_6, aP7_7, aP7_8, aP7_9, aP7_10, aP7_11, aP7_12, aP7_13, aP7_14, L7_0, L7_1, L7_2, L7_3, L7_4, L7_5]
    ring
  have bS0 : aS7_0 r0 ≤ MS7_0 := CaseSplit.le_mxr (aS7_0) 10 r0 (by omega)
  have bS1 : aS7_1 r1 ≤ MS7_1 := CaseSplit.le_mxr (aS7_1) 12 r1 (by omega)
  have bS2 : aS7_2 r2 ≤ MS7_2 := CaseSplit.le_mxr (aS7_2) 16 r2 (by omega)
  have bS3 : aS7_3 r3 ≤ MS7_3 := CaseSplit.le_mxr (aS7_3) 18 r3 (by omega)
  have bS4 : aS7_4 r4 ≤ MS7_4 := CaseSplit.le_mxr (aS7_4) 22 r4 (by omega)
  have bS5 : aS7_5 r5 ≤ MS7_5 := CaseSplit.le_mxr (aS7_5) 28 r5 (by omega)
  have bP0 : aP7_0 r0 r1 ≤ MP7_0 := CaseSplit.le_mxr2 (aP7_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP7_1 r0 r2 ≤ MP7_1 := CaseSplit.le_mxr2 (aP7_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP7_2 r0 r3 ≤ MP7_2 := CaseSplit.le_mxr2 (aP7_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP7_3 r0 r4 ≤ MP7_3 := CaseSplit.le_mxr2 (aP7_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP7_4 r0 r5 ≤ MP7_4 := CaseSplit.le_mxr2 (aP7_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP7_5 r1 r2 ≤ MP7_5 := CaseSplit.le_mxr2 (aP7_5) 12 16 r1 r2 (by omega) (by omega)
  have bP6 : aP7_6 r1 r3 ≤ MP7_6 := CaseSplit.le_mxr2 (aP7_6) 12 18 r1 r3 (by omega) (by omega)
  have bP7 : aP7_7 r1 r4 ≤ MP7_7 := CaseSplit.le_mxr2 (aP7_7) 12 22 r1 r4 (by omega) (by omega)
  have bP8 : aP7_8 r1 r5 ≤ MP7_8 := CaseSplit.le_mxr2 (aP7_8) 12 28 r1 r5 (by omega) (by omega)
  have bP9 : aP7_9 r2 r3 ≤ MP7_9 := CaseSplit.le_mxr2 (aP7_9) 16 18 r2 r3 (by omega) (by omega)
  have bP10 : aP7_10 r2 r4 ≤ MP7_10 := CaseSplit.le_mxr2 (aP7_10) 16 22 r2 r4 (by omega) (by omega)
  have bP11 : aP7_11 r2 r5 ≤ MP7_11 := CaseSplit.le_mxr2 (aP7_11) 16 28 r2 r5 (by omega) (by omega)
  have bP12 : aP7_12 r3 r4 ≤ MP7_12 := CaseSplit.le_mxr2 (aP7_12) 18 22 r3 r4 (by omega) (by omega)
  have bP13 : aP7_13 r3 r5 ≤ MP7_13 := CaseSplit.le_mxr2 (aP7_13) 18 28 r3 r5 (by omega) (by omega)
  have bP14 : aP7_14 r4 r5 ≤ MP7_14 := CaseSplit.le_mxr2 (aP7_14) 22 28 r4 r5 (by omega) (by omega)
  have hrhs : rhs7 = (∑ t ∈ Finset.range n7, w7 t) + 6 * (n7 : ℤ) := rfl
  have hc := cert7
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, bS0, bS1, bS2, bS3, bS4, bS5, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14]

end IncCert29
