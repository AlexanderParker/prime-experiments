/-
INCREMENT-WIDTH CERTIFICATE, step 23->29, case 6 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_23_29.json, which re-derives every number
from the primes alone).

Machine 29, INCREMENT width 49 = F_2(23) + s_min(29) = 39 + 10,
held gears [5, 7] at phases [0, 6].  Free gears [11, 13, 17, 19, 23, 29].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 5.
-/
import IncCert29B

namespace IncCert29

/-! ### case 6: held gears at phases [0, 6] -/

def p6 : List ℕ := [3, 5, 8, 10, 12, 13, 15, 17, 18, 20, 22, 25, 27, 32, 33, 38, 40, 43, 45, 47, 48]
def q6 (t : ℕ) : ℕ := p6.getD t 0
def n6 : ℕ := 21
def yl6 : List ℤ := [0, 0, 0, 4, 0, 3, 4, 3, 4, 4, 2, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0]
def w6 (t : ℕ) : ℤ := yl6.getD t 0
def ul6 : List ℤ := [(-4), 0, (-4), (-1), (-4), (-4), 0, (-3), (-6), (-4), 0, (-4), (-1), 0, 0, 1, 0, 3, 0, 4, 0, 3, 0, 0, 2, 2, 0, 0, 2, 2, 2, 2, 0, 0, 2, 2, 2, 0, 0, 0, 2, (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), 0, 0, 0, (-2), 2, 0, 0, 0, 0, 0, 2, 2, 2, (-3), 0, 0, 0, 2, 2, 0, (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), 0, (-2), 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 0, (-14), 0, 0, 0, (-5), (-5), 0, 0, (-5), (-5), 0, (-5), (-5), 0, 0, (-9), 0, 4, 0, 0, 0, 4, 0, (-8), 4, 4, 0, 0, 0, 4, 0, 4, 0, (-8), 0, 4, 4, 0, 0, 4, 0, 0, 0, 4, (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), 0, 17, 11, 14, 14, 17, 17, 13, 11, 12, 17, 17, 17, 8, 9, 14, 17, 17, (-17), (-17), (-17), (-17), (-17), (-17), (-17), (-17), (-17), (-17), (-17), (-17), (-17), 12, 15, 17, 16, 14, 8, 17, 17, 16, 17, 9, 8, 17, 17, 17, 15, 17, 17, 8, (-17), (-17), (-17), (-17), (-17), (-17), (-17), (-17), (-17), (-17), (-17), (-17), (-21), 5, 13, 19, 5, 14, 19, 8, 19, 0, 18, 19, 19, 14, 0, 19, 10, 19, 19, 10, 19, 19, 0, 19, (-19), (-19), (-19), (-19), (-19), (-19), (-19), (-19), (-19), (-19), (-19), (-19), (-19), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-9), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-8), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 7, 15, 1, 10, 6, 11, 14, 15, 11, 2, 9, 15, 15, 15, 15, 15, 9, 15, 15, 8, 15, 6, 7, 6, 6, 15, 1, 2, 0, 0, 0, 0, 0, (-7), 0, 0, 0, 0, 0, 0, 0, 0, (-6), 0, 0, 0, 0, 0, (-7), 0, 0]
def u6 (k : ℕ) : ℤ := ul6.getD k 0

def c6_0 (r t : ℕ) : Bool := gb11 r (q6 t)
def c6_1 (r t : ℕ) : Bool := gb13 r (q6 t)
def c6_2 (r t : ℕ) : Bool := gb17 r (q6 t)
def c6_3 (r t : ℕ) : Bool := gb19 r (q6 t)
def c6_4 (r t : ℕ) : Bool := gb23 r (q6 t)
def c6_5 (r t : ℕ) : Bool := gb29 r (q6 t)

def S6_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n6, (w6 t + 5) * (if c6_0 r t then 1 else 0)
def S6_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n6, (w6 t + 5) * (if c6_1 r t then 1 else 0)
def S6_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n6, (w6 t + 5) * (if c6_2 r t then 1 else 0)
def S6_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n6, (w6 t + 5) * (if c6_3 r t then 1 else 0)
def S6_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n6, (w6 t + 5) * (if c6_4 r t then 1 else 0)
def S6_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n6, (w6 t + 5) * (if c6_5 r t then 1 else 0)

def L6_0 (r : ℕ) : ℤ := u6 (13 + r) + u6 (41 + r) + u6 (71 + r) + u6 (105 + r) + u6 (145 + r)
def L6_1 (r : ℕ) : ℤ := u6 (0 + r) + u6 (173 + r) + u6 (205 + r) + u6 (241 + r) + u6 (283 + r)
def L6_2 (r : ℕ) : ℤ := u6 (24 + r) + u6 (156 + r) + u6 (315 + r) + u6 (355 + r) + u6 (401 + r)
def L6_3 (r : ℕ) : ℤ := u6 (52 + r) + u6 (186 + r) + u6 (296 + r) + u6 (441 + r) + u6 (489 + r)
def L6_4 (r : ℕ) : ℤ := u6 (82 + r) + u6 (218 + r) + u6 (332 + r) + u6 (418 + r) + u6 (537 + r)
def L6_5 (r : ℕ) : ℤ := u6 (116 + r) + u6 (254 + r) + u6 (372 + r) + u6 (460 + r) + u6 (508 + r)

def aS6_0 (r : ℕ) : ℤ := S6_0 r - L6_0 r
def MS6_0 : ℤ := CaseSplit.mxr (aS6_0) 10
def aS6_1 (r : ℕ) : ℤ := S6_1 r - L6_1 r
def MS6_1 : ℤ := CaseSplit.mxr (aS6_1) 12
def aS6_2 (r : ℕ) : ℤ := S6_2 r - L6_2 r
def MS6_2 : ℤ := CaseSplit.mxr (aS6_2) 16
def aS6_3 (r : ℕ) : ℤ := S6_3 r - L6_3 r
def MS6_3 : ℤ := CaseSplit.mxr (aS6_3) 18
def aS6_4 (r : ℕ) : ℤ := S6_4 r - L6_4 r
def MS6_4 : ℤ := CaseSplit.mxr (aS6_4) 22
def aS6_5 (r : ℕ) : ℤ := S6_5 r - L6_5 r
def MS6_5 : ℤ := CaseSplit.mxr (aS6_5) 28

def N6_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n6, (if c6_0 ra t && c6_1 rb t then 1 else 0)
def aP6_0 (ra rb : ℕ) : ℤ := -(5) * N6_0 ra rb + u6 (0 + rb) + u6 (13 + ra)
def MP6_0 : ℤ := CaseSplit.mxr2 (aP6_0) 10 12
def N6_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n6, (if c6_0 ra t && c6_2 rb t then 1 else 0)
def aP6_1 (ra rb : ℕ) : ℤ := -(5) * N6_1 ra rb + u6 (24 + rb) + u6 (41 + ra)
def MP6_1 : ℤ := CaseSplit.mxr2 (aP6_1) 10 16
def N6_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n6, (if c6_0 ra t && c6_3 rb t then 1 else 0)
def aP6_2 (ra rb : ℕ) : ℤ := -(5) * N6_2 ra rb + u6 (52 + rb) + u6 (71 + ra)
def MP6_2 : ℤ := CaseSplit.mxr2 (aP6_2) 10 18
def N6_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n6, (if c6_0 ra t && c6_4 rb t then 1 else 0)
def aP6_3 (ra rb : ℕ) : ℤ := -(5) * N6_3 ra rb + u6 (82 + rb) + u6 (105 + ra)
def MP6_3 : ℤ := CaseSplit.mxr2 (aP6_3) 10 22
def N6_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n6, (if c6_0 ra t && c6_5 rb t then 1 else 0)
def aP6_4 (ra rb : ℕ) : ℤ := -(5) * N6_4 ra rb + u6 (116 + rb) + u6 (145 + ra)
def MP6_4 : ℤ := CaseSplit.mxr2 (aP6_4) 10 28
def P6_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n6, (if c6_1 ra t && c6_2 rb t then 1 else 0)
def C6_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n6, (if c6_1 ra t && c6_2 rb t && c6_0 s t then 1 else 0)
def M6_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C6_5 ra rb) 10
def E6_5 : List ℕ := [21, 27, 57, 63, 111, 117, 136, 147, 158, 169]
def N6_5 (ra rb : ℕ) : ℤ := if E6_5.contains (ra * 17 + rb) = true then P6_5 ra rb - M6_5 ra rb else 0
def aP6_5 (ra rb : ℕ) : ℤ := -(5) * N6_5 ra rb + u6 (156 + rb) + u6 (173 + ra)
def MP6_5 : ℤ := CaseSplit.mxr2 (aP6_5) 12 16
def P6_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n6, (if c6_1 ra t && c6_3 rb t then 1 else 0)
def C6_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n6, (if c6_1 ra t && c6_3 rb t && c6_0 s t then 1 else 0)
def M6_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C6_6 ra rb) 10
def E6_6 : List ℕ := [33, 67, 73, 78, 104, 131, 154, 180, 204, 207, 238, 244]
def N6_6 (ra rb : ℕ) : ℤ := if E6_6.contains (ra * 19 + rb) = true then P6_6 ra rb - M6_6 ra rb else 0
def aP6_6 (ra rb : ℕ) : ℤ := -(5) * N6_6 ra rb + u6 (186 + rb) + u6 (205 + ra)
def MP6_6 : ℤ := CaseSplit.mxr2 (aP6_6) 12 18
def P6_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n6, (if c6_1 ra t && c6_4 rb t then 1 else 0)
def C6_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n6, (if c6_1 ra t && c6_4 rb t && c6_0 s t then 1 else 0)
def M6_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C6_7 ra rb) 10
def E6_7 : List ℕ := []
def N6_7 (ra rb : ℕ) : ℤ := if E6_7.contains (ra * 23 + rb) = true then P6_7 ra rb - M6_7 ra rb else 0
def aP6_7 (ra rb : ℕ) : ℤ := -(5) * N6_7 ra rb + u6 (218 + rb) + u6 (241 + ra)
def MP6_7 : ℤ := CaseSplit.mxr2 (aP6_7) 12 22
def P6_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n6, (if c6_1 ra t && c6_5 rb t then 1 else 0)
def C6_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n6, (if c6_1 ra t && c6_5 rb t && c6_0 s t then 1 else 0)
def M6_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C6_8 ra rb) 10
def E6_8 : List ℕ := [103, 219]
def N6_8 (ra rb : ℕ) : ℤ := if E6_8.contains (ra * 29 + rb) = true then P6_8 ra rb - M6_8 ra rb else 0
def aP6_8 (ra rb : ℕ) : ℤ := -(5) * N6_8 ra rb + u6 (254 + rb) + u6 (283 + ra)
def MP6_8 : ℤ := CaseSplit.mxr2 (aP6_8) 12 28
def N6_9 (_ra _rb : ℕ) : ℤ := 0
def aP6_9 (ra rb : ℕ) : ℤ := -(5) * N6_9 ra rb + u6 (296 + rb) + u6 (315 + ra)
def MP6_9 : ℤ := CaseSplit.mxr2 (aP6_9) 16 18
def N6_10 (_ra _rb : ℕ) : ℤ := 0
def aP6_10 (ra rb : ℕ) : ℤ := -(5) * N6_10 ra rb + u6 (332 + rb) + u6 (355 + ra)
def MP6_10 : ℤ := CaseSplit.mxr2 (aP6_10) 16 22
def N6_11 (_ra _rb : ℕ) : ℤ := 0
def aP6_11 (ra rb : ℕ) : ℤ := -(5) * N6_11 ra rb + u6 (372 + rb) + u6 (401 + ra)
def MP6_11 : ℤ := CaseSplit.mxr2 (aP6_11) 16 28
def N6_12 (_ra _rb : ℕ) : ℤ := 0
def aP6_12 (ra rb : ℕ) : ℤ := -(5) * N6_12 ra rb + u6 (418 + rb) + u6 (441 + ra)
def MP6_12 : ℤ := CaseSplit.mxr2 (aP6_12) 18 22
def N6_13 (_ra _rb : ℕ) : ℤ := 0
def aP6_13 (ra rb : ℕ) : ℤ := -(5) * N6_13 ra rb + u6 (460 + rb) + u6 (489 + ra)
def MP6_13 : ℤ := CaseSplit.mxr2 (aP6_13) 18 28
def N6_14 (_ra _rb : ℕ) : ℤ := 0
def aP6_14 (ra rb : ℕ) : ℤ := -(5) * N6_14 ra rb + u6 (508 + rb) + u6 (537 + ra)
def MP6_14 : ℤ := CaseSplit.mxr2 (aP6_14) 22 28

def rhs6 : ℤ := (∑ t ∈ Finset.range n6, w6 t) + 5 * (n6 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn6 : ∀ t, t < n6 → (0 : ℤ) ≤ w6 t := by decide
theorem plt6 : ∀ t, t < n6 → q6 t < 49 := by decide
theorem pfree6_5 : ∀ t, t < n6 → gb5 0 (q6 t) = false := by decide
theorem pfree6_7 : ∀ t, t < n6 → gb7 6 (q6 t) = false := by decide
theorem MSv6_0 : MS6_0 = 35 := by decide +kernel
theorem MSv6_1 : MS6_1 = 79 := by decide +kernel
theorem MSv6_2 : MS6_2 = 1 := by decide +kernel
theorem MSv6_3 : MS6_3 = 0 := by decide +kernel
theorem MSv6_4 : MS6_4 = 1 := by decide +kernel
theorem MSv6_5 : MS6_5 = 0 := by decide +kernel
theorem MPv6_0 : MP6_0 = 0 := by decide +kernel
theorem MPv6_1 : MP6_1 = 0 := by decide +kernel
theorem MPv6_2 : MP6_2 = 0 := by decide +kernel
theorem MPv6_3 : MP6_3 = 0 := by decide +kernel
theorem MPv6_4 : MP6_4 = 0 := by decide +kernel
theorem MPv6_5 : MP6_5 = 0 := by decide +kernel
theorem MPv6_6 : MP6_6 = 0 := by decide +kernel
theorem MPv6_7 : MP6_7 = 0 := by decide +kernel
theorem MPv6_8 : MP6_8 = 0 := by decide +kernel
theorem MPv6_9 : MP6_9 = 0 := by decide +kernel
theorem MPv6_10 : MP6_10 = 0 := by decide +kernel
theorem MPv6_11 : MP6_11 = 0 := by decide +kernel
theorem MPv6_12 : MP6_12 = 0 := by decide +kernel
theorem MPv6_13 : MP6_13 = 0 := by decide +kernel
theorem MPv6_14 : MP6_14 = 15 := by decide +kernel
theorem rhsv6 : rhs6 = 132 := by decide +kernel

/-- **The case-6 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/5.
    (Scaled by the common denominator 5: 131 < 132.) -/
theorem cert6 : MS6_0 + MS6_1 + MS6_2 + MS6_3 + MS6_4 + MS6_5 + MP6_0 + MP6_1 + MP6_2 + MP6_3 + MP6_4 + MP6_5 + MP6_6 + MP6_7 + MP6_8 + MP6_9 + MP6_10 + MP6_11 + MP6_12 + MP6_13 + MP6_14 < rhs6 := by
  rw [MSv6_0, MSv6_1, MSv6_2, MSv6_3, MSv6_4, MSv6_5, MPv6_0, MPv6_1, MPv6_2, MPv6_3, MPv6_4, MPv6_5, MPv6_6, MPv6_7, MPv6_8, MPv6_9, MPv6_10, MPv6_11, MPv6_12, MPv6_13, MPv6_14, rhsv6]
  decide

def Dg6 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := (if c6_0 r0 t then 1 else 0) + (if c6_1 r1 t then 1 else 0) + (if c6_2 r2 t then 1 else 0) + (if c6_3 r3 t then 1 else 0) + (if c6_4 r4 t then 1 else 0) + (if c6_5 r5 t then 1 else 0)
def Wl6_0 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c6_0 r0 t && c6_1 r1 t then 1 else 0
def Wl6_1 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c6_0 r0 t && c6_2 r2 t then 1 else 0
def Wl6_2 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c6_0 r0 t && c6_3 r3 t then 1 else 0
def Wl6_3 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c6_0 r0 t && c6_4 r4 t then 1 else 0
def Wl6_4 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c6_0 r0 t && c6_5 r5 t then 1 else 0
def Wl6_5 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c6_0 r0 t && c6_1 r1 t && c6_2 r2 t then 1 else 0
def Wl6_6 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c6_0 r0 t && c6_1 r1 t && c6_3 r3 t then 1 else 0
def Wl6_7 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c6_0 r0 t && c6_1 r1 t && c6_4 r4 t then 1 else 0
def Wl6_8 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c6_0 r0 t && c6_1 r1 t && c6_5 r5 t then 1 else 0
def Wl6_9 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c6_0 r0 t && !c6_1 r1 t && c6_2 r2 t && c6_3 r3 t then 1 else 0
def Wl6_10 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c6_0 r0 t && !c6_1 r1 t && c6_2 r2 t && c6_4 r4 t then 1 else 0
def Wl6_11 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c6_0 r0 t && !c6_1 r1 t && c6_2 r2 t && c6_5 r5 t then 1 else 0
def Wl6_12 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c6_0 r0 t && !c6_1 r1 t && !c6_2 r2 t && c6_3 r3 t && c6_4 r4 t then 1 else 0
def Wl6_13 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c6_0 r0 t && !c6_1 r1 t && !c6_2 r2 t && c6_3 r3 t && c6_5 r5 t then 1 else 0
def Wl6_14 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c6_0 r0 t && !c6_1 r1 t && !c6_2 r2 t && !c6_3 r3 t && c6_4 r4 t && c6_5 r5 t then 1 else 0

/-- **No configuration blocks the whole window in case 6.** -/
theorem nocov6 {r0 r1 r2 r3 r4 r5 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29)
    (hcov : ∀ t, t < n6 → (c6_0 r0 t || c6_1 r1 t || c6_2 r2 t || c6_3 r3 t || c6_4 r4 t || c6_5 r5 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n6, (1 : ℤ) + (Wl6_0 r0 r1 r2 r3 r4 r5 t + Wl6_1 r0 r1 r2 r3 r4 r5 t + Wl6_2 r0 r1 r2 r3 r4 r5 t + Wl6_3 r0 r1 r2 r3 r4 r5 t + Wl6_4 r0 r1 r2 r3 r4 r5 t + Wl6_5 r0 r1 r2 r3 r4 r5 t + Wl6_6 r0 r1 r2 r3 r4 r5 t + Wl6_7 r0 r1 r2 r3 r4 r5 t + Wl6_8 r0 r1 r2 r3 r4 r5 t + Wl6_9 r0 r1 r2 r3 r4 r5 t + Wl6_10 r0 r1 r2 r3 r4 r5 t + Wl6_11 r0 r1 r2 r3 r4 r5 t + Wl6_12 r0 r1 r2 r3 r4 r5 t + Wl6_13 r0 r1 r2 r3 r4 r5 t + Wl6_14 r0 r1 r2 r3 r4 r5 t) ≤ Dg6 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Wl6_0, Wl6_1, Wl6_2, Wl6_3, Wl6_4, Wl6_5, Wl6_6, Wl6_7, Wl6_8, Wl6_9, Wl6_10, Wl6_11, Wl6_12, Wl6_13, Wl6_14, Dg6]
    exact CaseSplit.lowest6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n6, (1 : ℤ) ≤ Dg6 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Dg6]
    exact CaseSplit.degpos6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n6 : ℤ) + ((∑ t ∈ Finset.range n6, Wl6_0 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n6, Wl6_1 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n6, Wl6_2 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n6, Wl6_3 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n6, Wl6_4 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n6, Wl6_5 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n6, Wl6_6 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n6, Wl6_7 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n6, Wl6_8 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n6, Wl6_9 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n6, Wl6_10 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n6, Wl6_11 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n6, Wl6_12 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n6, Wl6_13 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n6, Wl6_14 r0 r1 r2 r3 r4 r5 t)) ≤ ∑ t ∈ Finset.range n6, Dg6 r0 r1 r2 r3 r4 r5 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N6_0 r0 r1 ≤ ∑ t ∈ Finset.range n6, Wl6_0 r0 r1 r2 r3 r4 r5 t := by
    simp only [N6_0, Wl6_0, le_refl]
  have hn1 : N6_1 r0 r2 ≤ ∑ t ∈ Finset.range n6, Wl6_1 r0 r1 r2 r3 r4 r5 t := by
    simp only [N6_1, Wl6_1, le_refl]
  have hn2 : N6_2 r0 r3 ≤ ∑ t ∈ Finset.range n6, Wl6_2 r0 r1 r2 r3 r4 r5 t := by
    simp only [N6_2, Wl6_2, le_refl]
  have hn3 : N6_3 r0 r4 ≤ ∑ t ∈ Finset.range n6, Wl6_3 r0 r1 r2 r3 r4 r5 t := by
    simp only [N6_3, Wl6_3, le_refl]
  have hn4 : N6_4 r0 r5 ≤ ∑ t ∈ Finset.range n6, Wl6_4 r0 r1 r2 r3 r4 r5 t := by
    simp only [N6_4, Wl6_4, le_refl]
  have hn5 : N6_5 r1 r2 ≤ ∑ t ∈ Finset.range n6, Wl6_5 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n6, Wl6_5 r0 r1 r2 r3 r4 r5 t
        = (if c6_1 r1 t && c6_2 r2 t then (1:ℤ) else 0)
          - (if c6_1 r1 t && c6_2 r2 t && c6_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl6_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n6, Wl6_5 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl6_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n6, Wl6_5 r0 r1 r2 r3 r4 r5 t
        = P6_5 r1 r2 - C6_5 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P6_5, C6_5]
    have hm : C6_5 r1 r2 r0 ≤ M6_5 r1 r2 :=
      CaseSplit.le_mxr (C6_5 r1 r2) 10 r0 (by omega)
    simp only [N6_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N6_6 r1 r3 ≤ ∑ t ∈ Finset.range n6, Wl6_6 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n6, Wl6_6 r0 r1 r2 r3 r4 r5 t
        = (if c6_1 r1 t && c6_3 r3 t then (1:ℤ) else 0)
          - (if c6_1 r1 t && c6_3 r3 t && c6_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl6_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n6, Wl6_6 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl6_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n6, Wl6_6 r0 r1 r2 r3 r4 r5 t
        = P6_6 r1 r3 - C6_6 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P6_6, C6_6]
    have hm : C6_6 r1 r3 r0 ≤ M6_6 r1 r3 :=
      CaseSplit.le_mxr (C6_6 r1 r3) 10 r0 (by omega)
    simp only [N6_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N6_7 r1 r4 ≤ ∑ t ∈ Finset.range n6, Wl6_7 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n6, Wl6_7 r0 r1 r2 r3 r4 r5 t
        = (if c6_1 r1 t && c6_4 r4 t then (1:ℤ) else 0)
          - (if c6_1 r1 t && c6_4 r4 t && c6_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl6_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n6, Wl6_7 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl6_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n6, Wl6_7 r0 r1 r2 r3 r4 r5 t
        = P6_7 r1 r4 - C6_7 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P6_7, C6_7]
    have hm : C6_7 r1 r4 r0 ≤ M6_7 r1 r4 :=
      CaseSplit.le_mxr (C6_7 r1 r4) 10 r0 (by omega)
    simp only [N6_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N6_8 r1 r5 ≤ ∑ t ∈ Finset.range n6, Wl6_8 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n6, Wl6_8 r0 r1 r2 r3 r4 r5 t
        = (if c6_1 r1 t && c6_5 r5 t then (1:ℤ) else 0)
          - (if c6_1 r1 t && c6_5 r5 t && c6_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl6_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n6, Wl6_8 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl6_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n6, Wl6_8 r0 r1 r2 r3 r4 r5 t
        = P6_8 r1 r5 - C6_8 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P6_8, C6_8]
    have hm : C6_8 r1 r5 r0 ≤ M6_8 r1 r5 :=
      CaseSplit.le_mxr (C6_8 r1 r5) 10 r0 (by omega)
    simp only [N6_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N6_9 r2 r3 ≤ ∑ t ∈ Finset.range n6, Wl6_9 r0 r1 r2 r3 r4 r5 t := by
    simp only [N6_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl6_9]
    exact CaseSplit.ind_nonneg _
  have hn10 : N6_10 r2 r4 ≤ ∑ t ∈ Finset.range n6, Wl6_10 r0 r1 r2 r3 r4 r5 t := by
    simp only [N6_10]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl6_10]
    exact CaseSplit.ind_nonneg _
  have hn11 : N6_11 r2 r5 ≤ ∑ t ∈ Finset.range n6, Wl6_11 r0 r1 r2 r3 r4 r5 t := by
    simp only [N6_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl6_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N6_12 r3 r4 ≤ ∑ t ∈ Finset.range n6, Wl6_12 r0 r1 r2 r3 r4 r5 t := by
    simp only [N6_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl6_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N6_13 r3 r5 ≤ ∑ t ∈ Finset.range n6, Wl6_13 r0 r1 r2 r3 r4 r5 t := by
    simp only [N6_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl6_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N6_14 r4 r5 ≤ ∑ t ∈ Finset.range n6, Wl6_14 r0 r1 r2 r3 r4 r5 t := by
    simp only [N6_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl6_14]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n6, (w6 t + 5) * Dg6 r0 r1 r2 r3 r4 r5 t = S6_0 r0 + S6_1 r1 + S6_2 r2 + S6_3 r3 + S6_4 r4 + S6_5 r5 := by
    simp only [S6_0, S6_1, S6_2, S6_3, S6_4, S6_5, Dg6, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n6, (w6 t + 5) * Dg6 r0 r1 r2 r3 r4 r5 t
      = (∑ t ∈ Finset.range n6, w6 t * Dg6 r0 r1 r2 r3 r4 r5 t)
        + 5 * (∑ t ∈ Finset.range n6, Dg6 r0 r1 r2 r3 r4 r5 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n6, w6 t)
      ≤ ∑ t ∈ Finset.range n6, w6 t * Dg6 r0 r1 r2 r3 r4 r5 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg6 r0 r1 r2 r3 r4 r5 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w6 t := wnn6 t (Finset.mem_range.mp ht)
    calc w6 t = w6 t * 1 := (mul_one _).symm
      _ ≤ w6 t * Dg6 r0 r1 r2 r3 r4 r5 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS6_0 r0 + aS6_1 r1 + aS6_2 r2 + aS6_3 r3 + aS6_4 r4 + aS6_5 r5) + (aP6_0 r0 r1 + aP6_1 r0 r2 + aP6_2 r0 r3 + aP6_3 r0 r4 + aP6_4 r0 r5 + aP6_5 r1 r2 + aP6_6 r1 r3 + aP6_7 r1 r4 + aP6_8 r1 r5 + aP6_9 r2 r3 + aP6_10 r2 r4 + aP6_11 r2 r5 + aP6_12 r3 r4 + aP6_13 r3 r5 + aP6_14 r4 r5) = (S6_0 r0 + S6_1 r1 + S6_2 r2 + S6_3 r3 + S6_4 r4 + S6_5 r5) - 5 * (N6_0 r0 r1 + N6_1 r0 r2 + N6_2 r0 r3 + N6_3 r0 r4 + N6_4 r0 r5 + N6_5 r1 r2 + N6_6 r1 r3 + N6_7 r1 r4 + N6_8 r1 r5 + N6_9 r2 r3 + N6_10 r2 r4 + N6_11 r2 r5 + N6_12 r3 r4 + N6_13 r3 r5 + N6_14 r4 r5) := by
    simp only [aS6_0, aS6_1, aS6_2, aS6_3, aS6_4, aS6_5, aP6_0, aP6_1, aP6_2, aP6_3, aP6_4, aP6_5, aP6_6, aP6_7, aP6_8, aP6_9, aP6_10, aP6_11, aP6_12, aP6_13, aP6_14, L6_0, L6_1, L6_2, L6_3, L6_4, L6_5]
    ring
  have bS0 : aS6_0 r0 ≤ MS6_0 := CaseSplit.le_mxr (aS6_0) 10 r0 (by omega)
  have bS1 : aS6_1 r1 ≤ MS6_1 := CaseSplit.le_mxr (aS6_1) 12 r1 (by omega)
  have bS2 : aS6_2 r2 ≤ MS6_2 := CaseSplit.le_mxr (aS6_2) 16 r2 (by omega)
  have bS3 : aS6_3 r3 ≤ MS6_3 := CaseSplit.le_mxr (aS6_3) 18 r3 (by omega)
  have bS4 : aS6_4 r4 ≤ MS6_4 := CaseSplit.le_mxr (aS6_4) 22 r4 (by omega)
  have bS5 : aS6_5 r5 ≤ MS6_5 := CaseSplit.le_mxr (aS6_5) 28 r5 (by omega)
  have bP0 : aP6_0 r0 r1 ≤ MP6_0 := CaseSplit.le_mxr2 (aP6_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP6_1 r0 r2 ≤ MP6_1 := CaseSplit.le_mxr2 (aP6_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP6_2 r0 r3 ≤ MP6_2 := CaseSplit.le_mxr2 (aP6_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP6_3 r0 r4 ≤ MP6_3 := CaseSplit.le_mxr2 (aP6_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP6_4 r0 r5 ≤ MP6_4 := CaseSplit.le_mxr2 (aP6_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP6_5 r1 r2 ≤ MP6_5 := CaseSplit.le_mxr2 (aP6_5) 12 16 r1 r2 (by omega) (by omega)
  have bP6 : aP6_6 r1 r3 ≤ MP6_6 := CaseSplit.le_mxr2 (aP6_6) 12 18 r1 r3 (by omega) (by omega)
  have bP7 : aP6_7 r1 r4 ≤ MP6_7 := CaseSplit.le_mxr2 (aP6_7) 12 22 r1 r4 (by omega) (by omega)
  have bP8 : aP6_8 r1 r5 ≤ MP6_8 := CaseSplit.le_mxr2 (aP6_8) 12 28 r1 r5 (by omega) (by omega)
  have bP9 : aP6_9 r2 r3 ≤ MP6_9 := CaseSplit.le_mxr2 (aP6_9) 16 18 r2 r3 (by omega) (by omega)
  have bP10 : aP6_10 r2 r4 ≤ MP6_10 := CaseSplit.le_mxr2 (aP6_10) 16 22 r2 r4 (by omega) (by omega)
  have bP11 : aP6_11 r2 r5 ≤ MP6_11 := CaseSplit.le_mxr2 (aP6_11) 16 28 r2 r5 (by omega) (by omega)
  have bP12 : aP6_12 r3 r4 ≤ MP6_12 := CaseSplit.le_mxr2 (aP6_12) 18 22 r3 r4 (by omega) (by omega)
  have bP13 : aP6_13 r3 r5 ≤ MP6_13 := CaseSplit.le_mxr2 (aP6_13) 18 28 r3 r5 (by omega) (by omega)
  have bP14 : aP6_14 r4 r5 ≤ MP6_14 := CaseSplit.le_mxr2 (aP6_14) 22 28 r4 r5 (by omega) (by omega)
  have hrhs : rhs6 = (∑ t ∈ Finset.range n6, w6 t) + 5 * (n6 : ℤ) := rfl
  have hc := cert6
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, bS0, bS1, bS2, bS3, bS4, bS5, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14]

end IncCert29
