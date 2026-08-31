/-
INCREMENT-WIDTH CERTIFICATE, step 23->29, case 1 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_23_29.json, which re-derives every number
from the primes alone).

Machine 29, INCREMENT width 49 = F_2(23) + s_min(29) = 39 + 10,
held gears [5, 7] at phases [0, 1].  Free gears [11, 13, 17, 19, 23, 29].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 8.
-/
import IncCert29B

namespace IncCert29

/-! ### case 1: held gears at phases [0, 1] -/

def p1 : List ℕ := [2, 3, 8, 10, 13, 15, 17, 18, 20, 22, 23, 25, 27, 30, 32, 37, 38, 43, 45, 48]
def q1 (t : ℕ) : ℕ := p1.getD t 0
def n1 : ℕ := 20
def yl1 : List ℤ := [1, 0, 2, 3, 5, 7, 3, 6, 8, 7, 4, 4, 5, 2, 1, 0, 0, 0, 2, 0]
def w1 (t : ℕ) : ℤ := yl1.getD t 0
def ul1 : List ℤ := [0, 0, 0, (-6), 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, (-1), 0, (-1), 0, (-1), 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, (-3), (-3), (-3), 0, 0, 0, (-3), (-3), 0, 0, (-3), (-3), (-3), (-2), (-3), (-3), (-3), (-3), 0, 0, (-3), (-3), 0, (-3), 0, 0, (-7), (-3), 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 0, 0, (-3), (-3), 0, (-3), 0, 0, (-3), (-3), (-3), (-3), (-3), (-3), (-2), 0, (-3), (-3), (-3), 0, (-3), (-3), 0, (-3), (-3), 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 0, (-13), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 16, 16, 10, 16, 16, 16, 12, 16, 16, 16, 16, 15, 16, 12, 10, 16, (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), 15, 15, 16, 16, 16, 16, 11, 12, 16, 16, 9, 16, 15, 15, 16, 16, 16, 16, 13, (-22), (-16), (-16), (-16), (-16), (-16), (-16), (-19), (-16), (-16), (-21), (-16), (-25), 16, 16, 16, 3, 16, 14, 16, 13, 3, 16, 9, 5, 16, 0, 16, 16, 5, 16, 16, 16, 16, 0, 16, (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-7), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-7), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-9), (-2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 16, 16, 16, 3, 8, 13, 5, 16, 3, 15, 10, 0, 16, 16, 16, 6, 5, 9, 3, 14, 3, 5, 6, 3, 16, 0, 7, (-6), 3, 3, 0, 3, 3, (-5), 3, 0, 3, 3, 3, 3, 3, (-5), 3, 0, 3, (-9), (-1), 3, 3, 0]
def u1 (k : ℕ) : ℤ := ul1.getD k 0

def c1_0 (r t : ℕ) : Bool := gb11 r (q1 t)
def c1_1 (r t : ℕ) : Bool := gb13 r (q1 t)
def c1_2 (r t : ℕ) : Bool := gb17 r (q1 t)
def c1_3 (r t : ℕ) : Bool := gb19 r (q1 t)
def c1_4 (r t : ℕ) : Bool := gb23 r (q1 t)
def c1_5 (r t : ℕ) : Bool := gb29 r (q1 t)

def S1_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (w1 t + 3) * (if c1_0 r t then 1 else 0)
def S1_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (w1 t + 3) * (if c1_1 r t then 1 else 0)
def S1_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (w1 t + 3) * (if c1_2 r t then 1 else 0)
def S1_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (w1 t + 3) * (if c1_3 r t then 1 else 0)
def S1_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (w1 t + 3) * (if c1_4 r t then 1 else 0)
def S1_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (w1 t + 3) * (if c1_5 r t then 1 else 0)

def L1_0 (r : ℕ) : ℤ := u1 (13 + r) + u1 (41 + r) + u1 (71 + r) + u1 (105 + r) + u1 (145 + r)
def L1_1 (r : ℕ) : ℤ := u1 (0 + r) + u1 (173 + r) + u1 (205 + r) + u1 (241 + r) + u1 (283 + r)
def L1_2 (r : ℕ) : ℤ := u1 (24 + r) + u1 (156 + r) + u1 (315 + r) + u1 (355 + r) + u1 (401 + r)
def L1_3 (r : ℕ) : ℤ := u1 (52 + r) + u1 (186 + r) + u1 (296 + r) + u1 (441 + r) + u1 (489 + r)
def L1_4 (r : ℕ) : ℤ := u1 (82 + r) + u1 (218 + r) + u1 (332 + r) + u1 (418 + r) + u1 (537 + r)
def L1_5 (r : ℕ) : ℤ := u1 (116 + r) + u1 (254 + r) + u1 (372 + r) + u1 (460 + r) + u1 (508 + r)

def aS1_0 (r : ℕ) : ℤ := S1_0 r - L1_0 r
def MS1_0 : ℤ := CaseSplit.mxr (aS1_0) 10
def aS1_1 (r : ℕ) : ℤ := S1_1 r - L1_1 r
def MS1_1 : ℤ := CaseSplit.mxr (aS1_1) 12
def aS1_2 (r : ℕ) : ℤ := S1_2 r - L1_2 r
def MS1_2 : ℤ := CaseSplit.mxr (aS1_2) 16
def aS1_3 (r : ℕ) : ℤ := S1_3 r - L1_3 r
def MS1_3 : ℤ := CaseSplit.mxr (aS1_3) 18
def aS1_4 (r : ℕ) : ℤ := S1_4 r - L1_4 r
def MS1_4 : ℤ := CaseSplit.mxr (aS1_4) 22
def aS1_5 (r : ℕ) : ℤ := S1_5 r - L1_5 r
def MS1_5 : ℤ := CaseSplit.mxr (aS1_5) 28

def N1_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (if c1_0 ra t && c1_1 rb t then 1 else 0)
def aP1_0 (ra rb : ℕ) : ℤ := -(3) * N1_0 ra rb + u1 (0 + rb) + u1 (13 + ra)
def MP1_0 : ℤ := CaseSplit.mxr2 (aP1_0) 10 12
def N1_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (if c1_0 ra t && c1_2 rb t then 1 else 0)
def aP1_1 (ra rb : ℕ) : ℤ := -(3) * N1_1 ra rb + u1 (24 + rb) + u1 (41 + ra)
def MP1_1 : ℤ := CaseSplit.mxr2 (aP1_1) 10 16
def N1_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (if c1_0 ra t && c1_3 rb t then 1 else 0)
def aP1_2 (ra rb : ℕ) : ℤ := -(3) * N1_2 ra rb + u1 (52 + rb) + u1 (71 + ra)
def MP1_2 : ℤ := CaseSplit.mxr2 (aP1_2) 10 18
def N1_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (if c1_0 ra t && c1_4 rb t then 1 else 0)
def aP1_3 (ra rb : ℕ) : ℤ := -(3) * N1_3 ra rb + u1 (82 + rb) + u1 (105 + ra)
def MP1_3 : ℤ := CaseSplit.mxr2 (aP1_3) 10 22
def N1_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (if c1_0 ra t && c1_5 rb t then 1 else 0)
def aP1_4 (ra rb : ℕ) : ℤ := -(3) * N1_4 ra rb + u1 (116 + rb) + u1 (145 + ra)
def MP1_4 : ℤ := CaseSplit.mxr2 (aP1_4) 10 28
def P1_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (if c1_1 ra t && c1_2 rb t then 1 else 0)
def C1_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (if c1_1 ra t && c1_2 rb t && c1_0 s t then 1 else 0)
def M1_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C1_5 ra rb) 10
def E1_5 : List ℕ := [21, 27, 57, 63, 68, 79, 136, 147, 158, 169, 188, 194]
def N1_5 (ra rb : ℕ) : ℤ := if E1_5.contains (ra * 17 + rb) = true then P1_5 ra rb - M1_5 ra rb else 0
def aP1_5 (ra rb : ℕ) : ℤ := -(3) * N1_5 ra rb + u1 (156 + rb) + u1 (173 + ra)
def MP1_5 : ℤ := CaseSplit.mxr2 (aP1_5) 12 16
def P1_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (if c1_1 ra t && c1_3 rb t then 1 else 0)
def C1_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (if c1_1 ra t && c1_3 rb t && c1_0 s t then 1 else 0)
def M1_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C1_6 ra rb) 10
def E1_6 : List ℕ := [1, 31, 73, 104, 107, 138, 144, 172, 180, 214, 220, 244]
def N1_6 (ra rb : ℕ) : ℤ := if E1_6.contains (ra * 19 + rb) = true then P1_6 ra rb - M1_6 ra rb else 0
def aP1_6 (ra rb : ℕ) : ℤ := -(3) * N1_6 ra rb + u1 (186 + rb) + u1 (205 + ra)
def MP1_6 : ℤ := CaseSplit.mxr2 (aP1_6) 12 18
def P1_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (if c1_1 ra t && c1_4 rb t then 1 else 0)
def C1_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (if c1_1 ra t && c1_4 rb t && c1_0 s t then 1 else 0)
def M1_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C1_7 ra rb) 10
def E1_7 : List ℕ := []
def N1_7 (ra rb : ℕ) : ℤ := if E1_7.contains (ra * 23 + rb) = true then P1_7 ra rb - M1_7 ra rb else 0
def aP1_7 (ra rb : ℕ) : ℤ := -(3) * N1_7 ra rb + u1 (218 + rb) + u1 (241 + ra)
def MP1_7 : ℤ := CaseSplit.mxr2 (aP1_7) 12 22
def P1_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (if c1_1 ra t && c1_5 rb t then 1 else 0)
def C1_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (if c1_1 ra t && c1_5 rb t && c1_0 s t then 1 else 0)
def M1_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C1_8 ra rb) 10
def E1_8 : List ℕ := []
def N1_8 (ra rb : ℕ) : ℤ := if E1_8.contains (ra * 29 + rb) = true then P1_8 ra rb - M1_8 ra rb else 0
def aP1_8 (ra rb : ℕ) : ℤ := -(3) * N1_8 ra rb + u1 (254 + rb) + u1 (283 + ra)
def MP1_8 : ℤ := CaseSplit.mxr2 (aP1_8) 12 28
def N1_9 (_ra _rb : ℕ) : ℤ := 0
def aP1_9 (ra rb : ℕ) : ℤ := -(3) * N1_9 ra rb + u1 (296 + rb) + u1 (315 + ra)
def MP1_9 : ℤ := CaseSplit.mxr2 (aP1_9) 16 18
def N1_10 (_ra _rb : ℕ) : ℤ := 0
def aP1_10 (ra rb : ℕ) : ℤ := -(3) * N1_10 ra rb + u1 (332 + rb) + u1 (355 + ra)
def MP1_10 : ℤ := CaseSplit.mxr2 (aP1_10) 16 22
def N1_11 (_ra _rb : ℕ) : ℤ := 0
def aP1_11 (ra rb : ℕ) : ℤ := -(3) * N1_11 ra rb + u1 (372 + rb) + u1 (401 + ra)
def MP1_11 : ℤ := CaseSplit.mxr2 (aP1_11) 16 28
def N1_12 (_ra _rb : ℕ) : ℤ := 0
def aP1_12 (ra rb : ℕ) : ℤ := -(3) * N1_12 ra rb + u1 (418 + rb) + u1 (441 + ra)
def MP1_12 : ℤ := CaseSplit.mxr2 (aP1_12) 18 22
def N1_13 (_ra _rb : ℕ) : ℤ := 0
def aP1_13 (ra rb : ℕ) : ℤ := -(3) * N1_13 ra rb + u1 (460 + rb) + u1 (489 + ra)
def MP1_13 : ℤ := CaseSplit.mxr2 (aP1_13) 18 28
def N1_14 (_ra _rb : ℕ) : ℤ := 0
def aP1_14 (ra rb : ℕ) : ℤ := -(3) * N1_14 ra rb + u1 (508 + rb) + u1 (537 + ra)
def MP1_14 : ℤ := CaseSplit.mxr2 (aP1_14) 22 28

def rhs1 : ℤ := (∑ t ∈ Finset.range n1, w1 t) + 3 * (n1 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn1 : ∀ t, t < n1 → (0 : ℤ) ≤ w1 t := by decide
theorem plt1 : ∀ t, t < n1 → q1 t < 49 := by decide
theorem pfree1_5 : ∀ t, t < n1 → gb5 0 (q1 t) = false := by decide
theorem pfree1_7 : ∀ t, t < n1 → gb7 1 (q1 t) = false := by decide
theorem MSv1_0 : MS1_0 = 23 := by decide +kernel
theorem MSv1_1 : MS1_1 = 71 := by decide +kernel
theorem MSv1_2 : MS1_2 = 1 := by decide +kernel
theorem MSv1_3 : MS1_3 = 2 := by decide +kernel
theorem MSv1_4 : MS1_4 = 1 := by decide +kernel
theorem MSv1_5 : MS1_5 = 1 := by decide +kernel
theorem MPv1_0 : MP1_0 = 0 := by decide +kernel
theorem MPv1_1 : MP1_1 = 0 := by decide +kernel
theorem MPv1_2 : MP1_2 = 0 := by decide +kernel
theorem MPv1_3 : MP1_3 = 0 := by decide +kernel
theorem MPv1_4 : MP1_4 = 0 := by decide +kernel
theorem MPv1_5 : MP1_5 = 0 := by decide +kernel
theorem MPv1_6 : MP1_6 = 0 := by decide +kernel
theorem MPv1_7 : MP1_7 = 0 := by decide +kernel
theorem MPv1_8 : MP1_8 = 0 := by decide +kernel
theorem MPv1_9 : MP1_9 = 0 := by decide +kernel
theorem MPv1_10 : MP1_10 = 0 := by decide +kernel
theorem MPv1_11 : MP1_11 = 0 := by decide +kernel
theorem MPv1_12 : MP1_12 = 0 := by decide +kernel
theorem MPv1_13 : MP1_13 = 0 := by decide +kernel
theorem MPv1_14 : MP1_14 = 19 := by decide +kernel
theorem rhsv1 : rhs1 = 120 := by decide +kernel

/-- **The case-1 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 2/8.
    (Scaled by the common denominator 8: 118 < 120.) -/
theorem cert1 : MS1_0 + MS1_1 + MS1_2 + MS1_3 + MS1_4 + MS1_5 + MP1_0 + MP1_1 + MP1_2 + MP1_3 + MP1_4 + MP1_5 + MP1_6 + MP1_7 + MP1_8 + MP1_9 + MP1_10 + MP1_11 + MP1_12 + MP1_13 + MP1_14 < rhs1 := by
  rw [MSv1_0, MSv1_1, MSv1_2, MSv1_3, MSv1_4, MSv1_5, MPv1_0, MPv1_1, MPv1_2, MPv1_3, MPv1_4, MPv1_5, MPv1_6, MPv1_7, MPv1_8, MPv1_9, MPv1_10, MPv1_11, MPv1_12, MPv1_13, MPv1_14, rhsv1]
  decide

def Dg1 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := (if c1_0 r0 t then 1 else 0) + (if c1_1 r1 t then 1 else 0) + (if c1_2 r2 t then 1 else 0) + (if c1_3 r3 t then 1 else 0) + (if c1_4 r4 t then 1 else 0) + (if c1_5 r5 t then 1 else 0)
def Wl1_0 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c1_0 r0 t && c1_1 r1 t then 1 else 0
def Wl1_1 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c1_0 r0 t && c1_2 r2 t then 1 else 0
def Wl1_2 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c1_0 r0 t && c1_3 r3 t then 1 else 0
def Wl1_3 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c1_0 r0 t && c1_4 r4 t then 1 else 0
def Wl1_4 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c1_0 r0 t && c1_5 r5 t then 1 else 0
def Wl1_5 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c1_0 r0 t && c1_1 r1 t && c1_2 r2 t then 1 else 0
def Wl1_6 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c1_0 r0 t && c1_1 r1 t && c1_3 r3 t then 1 else 0
def Wl1_7 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c1_0 r0 t && c1_1 r1 t && c1_4 r4 t then 1 else 0
def Wl1_8 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c1_0 r0 t && c1_1 r1 t && c1_5 r5 t then 1 else 0
def Wl1_9 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c1_0 r0 t && !c1_1 r1 t && c1_2 r2 t && c1_3 r3 t then 1 else 0
def Wl1_10 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c1_0 r0 t && !c1_1 r1 t && c1_2 r2 t && c1_4 r4 t then 1 else 0
def Wl1_11 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c1_0 r0 t && !c1_1 r1 t && c1_2 r2 t && c1_5 r5 t then 1 else 0
def Wl1_12 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c1_0 r0 t && !c1_1 r1 t && !c1_2 r2 t && c1_3 r3 t && c1_4 r4 t then 1 else 0
def Wl1_13 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c1_0 r0 t && !c1_1 r1 t && !c1_2 r2 t && c1_3 r3 t && c1_5 r5 t then 1 else 0
def Wl1_14 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c1_0 r0 t && !c1_1 r1 t && !c1_2 r2 t && !c1_3 r3 t && c1_4 r4 t && c1_5 r5 t then 1 else 0

/-- **No configuration blocks the whole window in case 1.** -/
theorem nocov1 {r0 r1 r2 r3 r4 r5 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29)
    (hcov : ∀ t, t < n1 → (c1_0 r0 t || c1_1 r1 t || c1_2 r2 t || c1_3 r3 t || c1_4 r4 t || c1_5 r5 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n1, (1 : ℤ) + (Wl1_0 r0 r1 r2 r3 r4 r5 t + Wl1_1 r0 r1 r2 r3 r4 r5 t + Wl1_2 r0 r1 r2 r3 r4 r5 t + Wl1_3 r0 r1 r2 r3 r4 r5 t + Wl1_4 r0 r1 r2 r3 r4 r5 t + Wl1_5 r0 r1 r2 r3 r4 r5 t + Wl1_6 r0 r1 r2 r3 r4 r5 t + Wl1_7 r0 r1 r2 r3 r4 r5 t + Wl1_8 r0 r1 r2 r3 r4 r5 t + Wl1_9 r0 r1 r2 r3 r4 r5 t + Wl1_10 r0 r1 r2 r3 r4 r5 t + Wl1_11 r0 r1 r2 r3 r4 r5 t + Wl1_12 r0 r1 r2 r3 r4 r5 t + Wl1_13 r0 r1 r2 r3 r4 r5 t + Wl1_14 r0 r1 r2 r3 r4 r5 t) ≤ Dg1 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Wl1_0, Wl1_1, Wl1_2, Wl1_3, Wl1_4, Wl1_5, Wl1_6, Wl1_7, Wl1_8, Wl1_9, Wl1_10, Wl1_11, Wl1_12, Wl1_13, Wl1_14, Dg1]
    exact CaseSplit.lowest6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n1, (1 : ℤ) ≤ Dg1 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Dg1]
    exact CaseSplit.degpos6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n1 : ℤ) + ((∑ t ∈ Finset.range n1, Wl1_0 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n1, Wl1_1 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n1, Wl1_2 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n1, Wl1_3 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n1, Wl1_4 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n1, Wl1_5 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n1, Wl1_6 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n1, Wl1_7 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n1, Wl1_8 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n1, Wl1_9 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n1, Wl1_10 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n1, Wl1_11 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n1, Wl1_12 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n1, Wl1_13 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n1, Wl1_14 r0 r1 r2 r3 r4 r5 t)) ≤ ∑ t ∈ Finset.range n1, Dg1 r0 r1 r2 r3 r4 r5 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N1_0 r0 r1 ≤ ∑ t ∈ Finset.range n1, Wl1_0 r0 r1 r2 r3 r4 r5 t := by
    simp only [N1_0, Wl1_0, le_refl]
  have hn1 : N1_1 r0 r2 ≤ ∑ t ∈ Finset.range n1, Wl1_1 r0 r1 r2 r3 r4 r5 t := by
    simp only [N1_1, Wl1_1, le_refl]
  have hn2 : N1_2 r0 r3 ≤ ∑ t ∈ Finset.range n1, Wl1_2 r0 r1 r2 r3 r4 r5 t := by
    simp only [N1_2, Wl1_2, le_refl]
  have hn3 : N1_3 r0 r4 ≤ ∑ t ∈ Finset.range n1, Wl1_3 r0 r1 r2 r3 r4 r5 t := by
    simp only [N1_3, Wl1_3, le_refl]
  have hn4 : N1_4 r0 r5 ≤ ∑ t ∈ Finset.range n1, Wl1_4 r0 r1 r2 r3 r4 r5 t := by
    simp only [N1_4, Wl1_4, le_refl]
  have hn5 : N1_5 r1 r2 ≤ ∑ t ∈ Finset.range n1, Wl1_5 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n1, Wl1_5 r0 r1 r2 r3 r4 r5 t
        = (if c1_1 r1 t && c1_2 r2 t then (1:ℤ) else 0)
          - (if c1_1 r1 t && c1_2 r2 t && c1_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl1_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n1, Wl1_5 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl1_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n1, Wl1_5 r0 r1 r2 r3 r4 r5 t
        = P1_5 r1 r2 - C1_5 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P1_5, C1_5]
    have hm : C1_5 r1 r2 r0 ≤ M1_5 r1 r2 :=
      CaseSplit.le_mxr (C1_5 r1 r2) 10 r0 (by omega)
    simp only [N1_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N1_6 r1 r3 ≤ ∑ t ∈ Finset.range n1, Wl1_6 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n1, Wl1_6 r0 r1 r2 r3 r4 r5 t
        = (if c1_1 r1 t && c1_3 r3 t then (1:ℤ) else 0)
          - (if c1_1 r1 t && c1_3 r3 t && c1_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl1_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n1, Wl1_6 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl1_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n1, Wl1_6 r0 r1 r2 r3 r4 r5 t
        = P1_6 r1 r3 - C1_6 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P1_6, C1_6]
    have hm : C1_6 r1 r3 r0 ≤ M1_6 r1 r3 :=
      CaseSplit.le_mxr (C1_6 r1 r3) 10 r0 (by omega)
    simp only [N1_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N1_7 r1 r4 ≤ ∑ t ∈ Finset.range n1, Wl1_7 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n1, Wl1_7 r0 r1 r2 r3 r4 r5 t
        = (if c1_1 r1 t && c1_4 r4 t then (1:ℤ) else 0)
          - (if c1_1 r1 t && c1_4 r4 t && c1_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl1_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n1, Wl1_7 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl1_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n1, Wl1_7 r0 r1 r2 r3 r4 r5 t
        = P1_7 r1 r4 - C1_7 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P1_7, C1_7]
    have hm : C1_7 r1 r4 r0 ≤ M1_7 r1 r4 :=
      CaseSplit.le_mxr (C1_7 r1 r4) 10 r0 (by omega)
    simp only [N1_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N1_8 r1 r5 ≤ ∑ t ∈ Finset.range n1, Wl1_8 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n1, Wl1_8 r0 r1 r2 r3 r4 r5 t
        = (if c1_1 r1 t && c1_5 r5 t then (1:ℤ) else 0)
          - (if c1_1 r1 t && c1_5 r5 t && c1_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl1_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n1, Wl1_8 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl1_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n1, Wl1_8 r0 r1 r2 r3 r4 r5 t
        = P1_8 r1 r5 - C1_8 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P1_8, C1_8]
    have hm : C1_8 r1 r5 r0 ≤ M1_8 r1 r5 :=
      CaseSplit.le_mxr (C1_8 r1 r5) 10 r0 (by omega)
    simp only [N1_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N1_9 r2 r3 ≤ ∑ t ∈ Finset.range n1, Wl1_9 r0 r1 r2 r3 r4 r5 t := by
    simp only [N1_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl1_9]
    exact CaseSplit.ind_nonneg _
  have hn10 : N1_10 r2 r4 ≤ ∑ t ∈ Finset.range n1, Wl1_10 r0 r1 r2 r3 r4 r5 t := by
    simp only [N1_10]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl1_10]
    exact CaseSplit.ind_nonneg _
  have hn11 : N1_11 r2 r5 ≤ ∑ t ∈ Finset.range n1, Wl1_11 r0 r1 r2 r3 r4 r5 t := by
    simp only [N1_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl1_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N1_12 r3 r4 ≤ ∑ t ∈ Finset.range n1, Wl1_12 r0 r1 r2 r3 r4 r5 t := by
    simp only [N1_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl1_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N1_13 r3 r5 ≤ ∑ t ∈ Finset.range n1, Wl1_13 r0 r1 r2 r3 r4 r5 t := by
    simp only [N1_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl1_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N1_14 r4 r5 ≤ ∑ t ∈ Finset.range n1, Wl1_14 r0 r1 r2 r3 r4 r5 t := by
    simp only [N1_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl1_14]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n1, (w1 t + 3) * Dg1 r0 r1 r2 r3 r4 r5 t = S1_0 r0 + S1_1 r1 + S1_2 r2 + S1_3 r3 + S1_4 r4 + S1_5 r5 := by
    simp only [S1_0, S1_1, S1_2, S1_3, S1_4, S1_5, Dg1, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n1, (w1 t + 3) * Dg1 r0 r1 r2 r3 r4 r5 t
      = (∑ t ∈ Finset.range n1, w1 t * Dg1 r0 r1 r2 r3 r4 r5 t)
        + 3 * (∑ t ∈ Finset.range n1, Dg1 r0 r1 r2 r3 r4 r5 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n1, w1 t)
      ≤ ∑ t ∈ Finset.range n1, w1 t * Dg1 r0 r1 r2 r3 r4 r5 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg1 r0 r1 r2 r3 r4 r5 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w1 t := wnn1 t (Finset.mem_range.mp ht)
    calc w1 t = w1 t * 1 := (mul_one _).symm
      _ ≤ w1 t * Dg1 r0 r1 r2 r3 r4 r5 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS1_0 r0 + aS1_1 r1 + aS1_2 r2 + aS1_3 r3 + aS1_4 r4 + aS1_5 r5) + (aP1_0 r0 r1 + aP1_1 r0 r2 + aP1_2 r0 r3 + aP1_3 r0 r4 + aP1_4 r0 r5 + aP1_5 r1 r2 + aP1_6 r1 r3 + aP1_7 r1 r4 + aP1_8 r1 r5 + aP1_9 r2 r3 + aP1_10 r2 r4 + aP1_11 r2 r5 + aP1_12 r3 r4 + aP1_13 r3 r5 + aP1_14 r4 r5) = (S1_0 r0 + S1_1 r1 + S1_2 r2 + S1_3 r3 + S1_4 r4 + S1_5 r5) - 3 * (N1_0 r0 r1 + N1_1 r0 r2 + N1_2 r0 r3 + N1_3 r0 r4 + N1_4 r0 r5 + N1_5 r1 r2 + N1_6 r1 r3 + N1_7 r1 r4 + N1_8 r1 r5 + N1_9 r2 r3 + N1_10 r2 r4 + N1_11 r2 r5 + N1_12 r3 r4 + N1_13 r3 r5 + N1_14 r4 r5) := by
    simp only [aS1_0, aS1_1, aS1_2, aS1_3, aS1_4, aS1_5, aP1_0, aP1_1, aP1_2, aP1_3, aP1_4, aP1_5, aP1_6, aP1_7, aP1_8, aP1_9, aP1_10, aP1_11, aP1_12, aP1_13, aP1_14, L1_0, L1_1, L1_2, L1_3, L1_4, L1_5]
    ring
  have bS0 : aS1_0 r0 ≤ MS1_0 := CaseSplit.le_mxr (aS1_0) 10 r0 (by omega)
  have bS1 : aS1_1 r1 ≤ MS1_1 := CaseSplit.le_mxr (aS1_1) 12 r1 (by omega)
  have bS2 : aS1_2 r2 ≤ MS1_2 := CaseSplit.le_mxr (aS1_2) 16 r2 (by omega)
  have bS3 : aS1_3 r3 ≤ MS1_3 := CaseSplit.le_mxr (aS1_3) 18 r3 (by omega)
  have bS4 : aS1_4 r4 ≤ MS1_4 := CaseSplit.le_mxr (aS1_4) 22 r4 (by omega)
  have bS5 : aS1_5 r5 ≤ MS1_5 := CaseSplit.le_mxr (aS1_5) 28 r5 (by omega)
  have bP0 : aP1_0 r0 r1 ≤ MP1_0 := CaseSplit.le_mxr2 (aP1_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP1_1 r0 r2 ≤ MP1_1 := CaseSplit.le_mxr2 (aP1_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP1_2 r0 r3 ≤ MP1_2 := CaseSplit.le_mxr2 (aP1_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP1_3 r0 r4 ≤ MP1_3 := CaseSplit.le_mxr2 (aP1_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP1_4 r0 r5 ≤ MP1_4 := CaseSplit.le_mxr2 (aP1_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP1_5 r1 r2 ≤ MP1_5 := CaseSplit.le_mxr2 (aP1_5) 12 16 r1 r2 (by omega) (by omega)
  have bP6 : aP1_6 r1 r3 ≤ MP1_6 := CaseSplit.le_mxr2 (aP1_6) 12 18 r1 r3 (by omega) (by omega)
  have bP7 : aP1_7 r1 r4 ≤ MP1_7 := CaseSplit.le_mxr2 (aP1_7) 12 22 r1 r4 (by omega) (by omega)
  have bP8 : aP1_8 r1 r5 ≤ MP1_8 := CaseSplit.le_mxr2 (aP1_8) 12 28 r1 r5 (by omega) (by omega)
  have bP9 : aP1_9 r2 r3 ≤ MP1_9 := CaseSplit.le_mxr2 (aP1_9) 16 18 r2 r3 (by omega) (by omega)
  have bP10 : aP1_10 r2 r4 ≤ MP1_10 := CaseSplit.le_mxr2 (aP1_10) 16 22 r2 r4 (by omega) (by omega)
  have bP11 : aP1_11 r2 r5 ≤ MP1_11 := CaseSplit.le_mxr2 (aP1_11) 16 28 r2 r5 (by omega) (by omega)
  have bP12 : aP1_12 r3 r4 ≤ MP1_12 := CaseSplit.le_mxr2 (aP1_12) 18 22 r3 r4 (by omega) (by omega)
  have bP13 : aP1_13 r3 r5 ≤ MP1_13 := CaseSplit.le_mxr2 (aP1_13) 18 28 r3 r5 (by omega) (by omega)
  have bP14 : aP1_14 r4 r5 ≤ MP1_14 := CaseSplit.le_mxr2 (aP1_14) 22 28 r4 r5 (by omega) (by omega)
  have hrhs : rhs1 = (∑ t ∈ Finset.range n1, w1 t) + 3 * (n1 : ℤ) := rfl
  have hc := cert1
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, bS0, bS1, bS2, bS3, bS4, bS5, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14]

end IncCert29
