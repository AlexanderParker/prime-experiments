/-
INCREMENT-WIDTH CERTIFICATE, step 23->29, case 27 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_23_29.json, which re-derives every number
from the primes alone).

Machine 29, INCREMENT width 49 = F_2(23) + s_min(29) = 39 + 10,
held gears [5, 7] at phases [3, 6].  Free gears [11, 13, 17, 19, 23, 29].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 7.
-/
import IncCert29B

namespace IncCert29

/-! ### case 27: held gears at phases [3, 6] -/

def p27 : List ℕ := [4, 5, 10, 12, 15, 17, 19, 20, 22, 24, 25, 27, 29, 32, 34, 39, 40, 45, 47]
def q27 (t : ℕ) : ℕ := p27.getD t 0
def n27 : ℕ := 19
def yl27 : List ℤ := [0, 0, 3, 4, 4, 5, 4, 6, 7, 7, 4, 6, 4, 2, 1, 0, 1, 0, 1]
def w27 (t : ℕ) : ℤ := yl27.getD t 0
def ul27 : List ℤ := [0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0, 2, (-2), 0, (-2), (-2), (-2), (-2), (-2), (-2), 0, (-2), 0, (-3), (-3), (-1), (-1), (-3), (-3), (-3), (-3), (-1), 0, (-1), (-3), (-3), (-3), (-1), (-1), (-3), 0, (-1), 1, 1, 1, (-3), 0, 1, 3, 1, 0, (-1), 0, (-1), (-1), (-1), (-1), 0, 0, (-1), (-1), 0, (-1), 0, 0, 0, (-1), 0, (-1), (-1), 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 13, 14, 16, 14, 9, 16, 16, 16, 16, 11, 15, 13, 15, 15, 16, (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), 10, 13, 11, 13, 7, 8, 11, 13, 13, 13, 13, 13, 13, 13, 11, 13, 12, 13, 10, (-13), (-13), (-13), (-13), (-13), (-13), (-13), (-13), (-13), (-13), (-13), (-13), (-13), 10, 0, 10, 10, 1, 6, 0, 8, 4, 0, 10, 2, 6, 6, 2, 10, 0, 5, 10, 0, 10, 6, 10, (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 2, 13, 0, 8, 12, 3, 15, 2, 12, 9, 0, 15, 3, 15, 6, 3, 7, 2, 12, 2, 4, 6, 3, 13, 0, 8, 0, 6, 6, 0, 6, 4, 6, 6, 0, 6, 6, 5, 4, 0, 0, 0, 0, 6, 4, 6, 6, 0, 1, 0, 0]
def u27 (k : ℕ) : ℤ := ul27.getD k 0

def c27_0 (r t : ℕ) : Bool := gb11 r (q27 t)
def c27_1 (r t : ℕ) : Bool := gb13 r (q27 t)
def c27_2 (r t : ℕ) : Bool := gb17 r (q27 t)
def c27_3 (r t : ℕ) : Bool := gb19 r (q27 t)
def c27_4 (r t : ℕ) : Bool := gb23 r (q27 t)
def c27_5 (r t : ℕ) : Bool := gb29 r (q27 t)

def S27_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (w27 t + 2) * (if c27_0 r t then 1 else 0)
def S27_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (w27 t + 2) * (if c27_1 r t then 1 else 0)
def S27_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (w27 t + 2) * (if c27_2 r t then 1 else 0)
def S27_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (w27 t + 2) * (if c27_3 r t then 1 else 0)
def S27_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (w27 t + 2) * (if c27_4 r t then 1 else 0)
def S27_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (w27 t + 2) * (if c27_5 r t then 1 else 0)

def L27_0 (r : ℕ) : ℤ := u27 (13 + r) + u27 (41 + r) + u27 (71 + r) + u27 (105 + r) + u27 (145 + r)
def L27_1 (r : ℕ) : ℤ := u27 (0 + r) + u27 (173 + r) + u27 (205 + r) + u27 (241 + r) + u27 (283 + r)
def L27_2 (r : ℕ) : ℤ := u27 (24 + r) + u27 (156 + r) + u27 (315 + r) + u27 (355 + r) + u27 (401 + r)
def L27_3 (r : ℕ) : ℤ := u27 (52 + r) + u27 (186 + r) + u27 (296 + r) + u27 (441 + r) + u27 (489 + r)
def L27_4 (r : ℕ) : ℤ := u27 (82 + r) + u27 (218 + r) + u27 (332 + r) + u27 (418 + r) + u27 (537 + r)
def L27_5 (r : ℕ) : ℤ := u27 (116 + r) + u27 (254 + r) + u27 (372 + r) + u27 (460 + r) + u27 (508 + r)

def aS27_0 (r : ℕ) : ℤ := S27_0 r - L27_0 r
def MS27_0 : ℤ := CaseSplit.mxr (aS27_0) 10
def aS27_1 (r : ℕ) : ℤ := S27_1 r - L27_1 r
def MS27_1 : ℤ := CaseSplit.mxr (aS27_1) 12
def aS27_2 (r : ℕ) : ℤ := S27_2 r - L27_2 r
def MS27_2 : ℤ := CaseSplit.mxr (aS27_2) 16
def aS27_3 (r : ℕ) : ℤ := S27_3 r - L27_3 r
def MS27_3 : ℤ := CaseSplit.mxr (aS27_3) 18
def aS27_4 (r : ℕ) : ℤ := S27_4 r - L27_4 r
def MS27_4 : ℤ := CaseSplit.mxr (aS27_4) 22
def aS27_5 (r : ℕ) : ℤ := S27_5 r - L27_5 r
def MS27_5 : ℤ := CaseSplit.mxr (aS27_5) 28

def N27_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_0 ra t && c27_1 rb t then 1 else 0)
def aP27_0 (ra rb : ℕ) : ℤ := -(2) * N27_0 ra rb + u27 (0 + rb) + u27 (13 + ra)
def MP27_0 : ℤ := CaseSplit.mxr2 (aP27_0) 10 12
def N27_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_0 ra t && c27_2 rb t then 1 else 0)
def aP27_1 (ra rb : ℕ) : ℤ := -(2) * N27_1 ra rb + u27 (24 + rb) + u27 (41 + ra)
def MP27_1 : ℤ := CaseSplit.mxr2 (aP27_1) 10 16
def N27_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_0 ra t && c27_3 rb t then 1 else 0)
def aP27_2 (ra rb : ℕ) : ℤ := -(2) * N27_2 ra rb + u27 (52 + rb) + u27 (71 + ra)
def MP27_2 : ℤ := CaseSplit.mxr2 (aP27_2) 10 18
def N27_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_0 ra t && c27_4 rb t then 1 else 0)
def aP27_3 (ra rb : ℕ) : ℤ := -(2) * N27_3 ra rb + u27 (82 + rb) + u27 (105 + ra)
def MP27_3 : ℤ := CaseSplit.mxr2 (aP27_3) 10 22
def N27_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_0 ra t && c27_5 rb t then 1 else 0)
def aP27_4 (ra rb : ℕ) : ℤ := -(2) * N27_4 ra rb + u27 (116 + rb) + u27 (145 + ra)
def MP27_4 : ℤ := CaseSplit.mxr2 (aP27_4) 10 28
def P27_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_1 ra t && c27_2 rb t then 1 else 0)
def C27_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_1 ra t && c27_2 rb t && c27_0 s t then 1 else 0)
def M27_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C27_5 ra rb) 10
def E27_5 : List ℕ := [21, 27, 43, 49, 111, 117, 122, 133, 158, 169, 206, 212]
def N27_5 (ra rb : ℕ) : ℤ := if E27_5.contains (ra * 17 + rb) = true then P27_5 ra rb - M27_5 ra rb else 0
def aP27_5 (ra rb : ℕ) : ℤ := -(2) * N27_5 ra rb + u27 (156 + rb) + u27 (173 + ra)
def MP27_5 : ℤ := CaseSplit.mxr2 (aP27_5) 12 16
def P27_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_1 ra t && c27_3 rb t then 1 else 0)
def C27_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_1 ra t && c27_3 rb t && c27_0 s t then 1 else 0)
def M27_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C27_6 ra rb) 10
def E27_6 : List ℕ := [33, 64, 67, 98, 104, 140, 151, 174, 180, 204, 227, 238]
def N27_6 (ra rb : ℕ) : ℤ := if E27_6.contains (ra * 19 + rb) = true then P27_6 ra rb - M27_6 ra rb else 0
def aP27_6 (ra rb : ℕ) : ℤ := -(2) * N27_6 ra rb + u27 (186 + rb) + u27 (205 + ra)
def MP27_6 : ℤ := CaseSplit.mxr2 (aP27_6) 12 18
def P27_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_1 ra t && c27_4 rb t then 1 else 0)
def C27_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_1 ra t && c27_4 rb t && c27_0 s t then 1 else 0)
def M27_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C27_7 ra rb) 10
def E27_7 : List ℕ := []
def N27_7 (ra rb : ℕ) : ℤ := if E27_7.contains (ra * 23 + rb) = true then P27_7 ra rb - M27_7 ra rb else 0
def aP27_7 (ra rb : ℕ) : ℤ := -(2) * N27_7 ra rb + u27 (218 + rb) + u27 (241 + ra)
def MP27_7 : ℤ := CaseSplit.mxr2 (aP27_7) 12 22
def P27_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_1 ra t && c27_5 rb t then 1 else 0)
def C27_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_1 ra t && c27_5 rb t && c27_0 s t then 1 else 0)
def M27_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C27_8 ra rb) 10
def E27_8 : List ℕ := []
def N27_8 (ra rb : ℕ) : ℤ := if E27_8.contains (ra * 29 + rb) = true then P27_8 ra rb - M27_8 ra rb else 0
def aP27_8 (ra rb : ℕ) : ℤ := -(2) * N27_8 ra rb + u27 (254 + rb) + u27 (283 + ra)
def MP27_8 : ℤ := CaseSplit.mxr2 (aP27_8) 12 28
def N27_9 (_ra _rb : ℕ) : ℤ := 0
def aP27_9 (ra rb : ℕ) : ℤ := -(2) * N27_9 ra rb + u27 (296 + rb) + u27 (315 + ra)
def MP27_9 : ℤ := CaseSplit.mxr2 (aP27_9) 16 18
def N27_10 (_ra _rb : ℕ) : ℤ := 0
def aP27_10 (ra rb : ℕ) : ℤ := -(2) * N27_10 ra rb + u27 (332 + rb) + u27 (355 + ra)
def MP27_10 : ℤ := CaseSplit.mxr2 (aP27_10) 16 22
def N27_11 (_ra _rb : ℕ) : ℤ := 0
def aP27_11 (ra rb : ℕ) : ℤ := -(2) * N27_11 ra rb + u27 (372 + rb) + u27 (401 + ra)
def MP27_11 : ℤ := CaseSplit.mxr2 (aP27_11) 16 28
def N27_12 (_ra _rb : ℕ) : ℤ := 0
def aP27_12 (ra rb : ℕ) : ℤ := -(2) * N27_12 ra rb + u27 (418 + rb) + u27 (441 + ra)
def MP27_12 : ℤ := CaseSplit.mxr2 (aP27_12) 18 22
def N27_13 (_ra _rb : ℕ) : ℤ := 0
def aP27_13 (ra rb : ℕ) : ℤ := -(2) * N27_13 ra rb + u27 (460 + rb) + u27 (489 + ra)
def MP27_13 : ℤ := CaseSplit.mxr2 (aP27_13) 18 28
def N27_14 (_ra _rb : ℕ) : ℤ := 0
def aP27_14 (ra rb : ℕ) : ℤ := -(2) * N27_14 ra rb + u27 (508 + rb) + u27 (537 + ra)
def MP27_14 : ℤ := CaseSplit.mxr2 (aP27_14) 22 28

def rhs27 : ℤ := (∑ t ∈ Finset.range n27, w27 t) + 2 * (n27 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn27 : ∀ t, t < n27 → (0 : ℤ) ≤ w27 t := by decide
theorem plt27 : ∀ t, t < n27 → q27 t < 49 := by decide
theorem pfree27_5 : ∀ t, t < n27 → gb5 3 (q27 t) = false := by decide
theorem pfree27_7 : ∀ t, t < n27 → gb7 6 (q27 t) = false := by decide
theorem MSv27_0 : MS27_0 = 19 := by decide +kernel
theorem MSv27_1 : MS27_1 = 56 := by decide +kernel
theorem MSv27_2 : MS27_2 = 0 := by decide +kernel
theorem MSv27_3 : MS27_3 = 0 := by decide +kernel
theorem MSv27_4 : MS27_4 = 0 := by decide +kernel
theorem MSv27_5 : MS27_5 = 0 := by decide +kernel
theorem MPv27_0 : MP27_0 = 0 := by decide +kernel
theorem MPv27_1 : MP27_1 = 0 := by decide +kernel
theorem MPv27_2 : MP27_2 = 0 := by decide +kernel
theorem MPv27_3 : MP27_3 = 0 := by decide +kernel
theorem MPv27_4 : MP27_4 = 0 := by decide +kernel
theorem MPv27_5 : MP27_5 = 0 := by decide +kernel
theorem MPv27_6 : MP27_6 = 0 := by decide +kernel
theorem MPv27_7 : MP27_7 = 0 := by decide +kernel
theorem MPv27_8 : MP27_8 = 0 := by decide +kernel
theorem MPv27_9 : MP27_9 = 0 := by decide +kernel
theorem MPv27_10 : MP27_10 = 0 := by decide +kernel
theorem MPv27_11 : MP27_11 = 0 := by decide +kernel
theorem MPv27_12 : MP27_12 = 0 := by decide +kernel
theorem MPv27_13 : MP27_13 = 0 := by decide +kernel
theorem MPv27_14 : MP27_14 = 21 := by decide +kernel
theorem rhsv27 : rhs27 = 97 := by decide +kernel

/-- **The case-27 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/7.
    (Scaled by the common denominator 7: 96 < 97.) -/
theorem cert27 : MS27_0 + MS27_1 + MS27_2 + MS27_3 + MS27_4 + MS27_5 + MP27_0 + MP27_1 + MP27_2 + MP27_3 + MP27_4 + MP27_5 + MP27_6 + MP27_7 + MP27_8 + MP27_9 + MP27_10 + MP27_11 + MP27_12 + MP27_13 + MP27_14 < rhs27 := by
  rw [MSv27_0, MSv27_1, MSv27_2, MSv27_3, MSv27_4, MSv27_5, MPv27_0, MPv27_1, MPv27_2, MPv27_3, MPv27_4, MPv27_5, MPv27_6, MPv27_7, MPv27_8, MPv27_9, MPv27_10, MPv27_11, MPv27_12, MPv27_13, MPv27_14, rhsv27]
  decide

def Dg27 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := (if c27_0 r0 t then 1 else 0) + (if c27_1 r1 t then 1 else 0) + (if c27_2 r2 t then 1 else 0) + (if c27_3 r3 t then 1 else 0) + (if c27_4 r4 t then 1 else 0) + (if c27_5 r5 t then 1 else 0)
def Wl27_0 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c27_0 r0 t && c27_1 r1 t then 1 else 0
def Wl27_1 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c27_0 r0 t && c27_2 r2 t then 1 else 0
def Wl27_2 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c27_0 r0 t && c27_3 r3 t then 1 else 0
def Wl27_3 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c27_0 r0 t && c27_4 r4 t then 1 else 0
def Wl27_4 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c27_0 r0 t && c27_5 r5 t then 1 else 0
def Wl27_5 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c27_0 r0 t && c27_1 r1 t && c27_2 r2 t then 1 else 0
def Wl27_6 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c27_0 r0 t && c27_1 r1 t && c27_3 r3 t then 1 else 0
def Wl27_7 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c27_0 r0 t && c27_1 r1 t && c27_4 r4 t then 1 else 0
def Wl27_8 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c27_0 r0 t && c27_1 r1 t && c27_5 r5 t then 1 else 0
def Wl27_9 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c27_0 r0 t && !c27_1 r1 t && c27_2 r2 t && c27_3 r3 t then 1 else 0
def Wl27_10 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c27_0 r0 t && !c27_1 r1 t && c27_2 r2 t && c27_4 r4 t then 1 else 0
def Wl27_11 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c27_0 r0 t && !c27_1 r1 t && c27_2 r2 t && c27_5 r5 t then 1 else 0
def Wl27_12 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c27_0 r0 t && !c27_1 r1 t && !c27_2 r2 t && c27_3 r3 t && c27_4 r4 t then 1 else 0
def Wl27_13 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c27_0 r0 t && !c27_1 r1 t && !c27_2 r2 t && c27_3 r3 t && c27_5 r5 t then 1 else 0
def Wl27_14 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c27_0 r0 t && !c27_1 r1 t && !c27_2 r2 t && !c27_3 r3 t && c27_4 r4 t && c27_5 r5 t then 1 else 0

/-- **No configuration blocks the whole window in case 27.** -/
theorem nocov27 {r0 r1 r2 r3 r4 r5 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29)
    (hcov : ∀ t, t < n27 → (c27_0 r0 t || c27_1 r1 t || c27_2 r2 t || c27_3 r3 t || c27_4 r4 t || c27_5 r5 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n27, (1 : ℤ) + (Wl27_0 r0 r1 r2 r3 r4 r5 t + Wl27_1 r0 r1 r2 r3 r4 r5 t + Wl27_2 r0 r1 r2 r3 r4 r5 t + Wl27_3 r0 r1 r2 r3 r4 r5 t + Wl27_4 r0 r1 r2 r3 r4 r5 t + Wl27_5 r0 r1 r2 r3 r4 r5 t + Wl27_6 r0 r1 r2 r3 r4 r5 t + Wl27_7 r0 r1 r2 r3 r4 r5 t + Wl27_8 r0 r1 r2 r3 r4 r5 t + Wl27_9 r0 r1 r2 r3 r4 r5 t + Wl27_10 r0 r1 r2 r3 r4 r5 t + Wl27_11 r0 r1 r2 r3 r4 r5 t + Wl27_12 r0 r1 r2 r3 r4 r5 t + Wl27_13 r0 r1 r2 r3 r4 r5 t + Wl27_14 r0 r1 r2 r3 r4 r5 t) ≤ Dg27 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Wl27_0, Wl27_1, Wl27_2, Wl27_3, Wl27_4, Wl27_5, Wl27_6, Wl27_7, Wl27_8, Wl27_9, Wl27_10, Wl27_11, Wl27_12, Wl27_13, Wl27_14, Dg27]
    exact CaseSplit.lowest6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n27, (1 : ℤ) ≤ Dg27 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Dg27]
    exact CaseSplit.degpos6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n27 : ℤ) + ((∑ t ∈ Finset.range n27, Wl27_0 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n27, Wl27_1 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n27, Wl27_2 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n27, Wl27_3 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n27, Wl27_4 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n27, Wl27_5 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n27, Wl27_6 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n27, Wl27_7 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n27, Wl27_8 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n27, Wl27_9 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n27, Wl27_10 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n27, Wl27_11 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n27, Wl27_12 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n27, Wl27_13 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n27, Wl27_14 r0 r1 r2 r3 r4 r5 t)) ≤ ∑ t ∈ Finset.range n27, Dg27 r0 r1 r2 r3 r4 r5 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N27_0 r0 r1 ≤ ∑ t ∈ Finset.range n27, Wl27_0 r0 r1 r2 r3 r4 r5 t := by
    simp only [N27_0, Wl27_0, le_refl]
  have hn1 : N27_1 r0 r2 ≤ ∑ t ∈ Finset.range n27, Wl27_1 r0 r1 r2 r3 r4 r5 t := by
    simp only [N27_1, Wl27_1, le_refl]
  have hn2 : N27_2 r0 r3 ≤ ∑ t ∈ Finset.range n27, Wl27_2 r0 r1 r2 r3 r4 r5 t := by
    simp only [N27_2, Wl27_2, le_refl]
  have hn3 : N27_3 r0 r4 ≤ ∑ t ∈ Finset.range n27, Wl27_3 r0 r1 r2 r3 r4 r5 t := by
    simp only [N27_3, Wl27_3, le_refl]
  have hn4 : N27_4 r0 r5 ≤ ∑ t ∈ Finset.range n27, Wl27_4 r0 r1 r2 r3 r4 r5 t := by
    simp only [N27_4, Wl27_4, le_refl]
  have hn5 : N27_5 r1 r2 ≤ ∑ t ∈ Finset.range n27, Wl27_5 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n27, Wl27_5 r0 r1 r2 r3 r4 r5 t
        = (if c27_1 r1 t && c27_2 r2 t then (1:ℤ) else 0)
          - (if c27_1 r1 t && c27_2 r2 t && c27_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl27_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n27, Wl27_5 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl27_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n27, Wl27_5 r0 r1 r2 r3 r4 r5 t
        = P27_5 r1 r2 - C27_5 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P27_5, C27_5]
    have hm : C27_5 r1 r2 r0 ≤ M27_5 r1 r2 :=
      CaseSplit.le_mxr (C27_5 r1 r2) 10 r0 (by omega)
    simp only [N27_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N27_6 r1 r3 ≤ ∑ t ∈ Finset.range n27, Wl27_6 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n27, Wl27_6 r0 r1 r2 r3 r4 r5 t
        = (if c27_1 r1 t && c27_3 r3 t then (1:ℤ) else 0)
          - (if c27_1 r1 t && c27_3 r3 t && c27_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl27_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n27, Wl27_6 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl27_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n27, Wl27_6 r0 r1 r2 r3 r4 r5 t
        = P27_6 r1 r3 - C27_6 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P27_6, C27_6]
    have hm : C27_6 r1 r3 r0 ≤ M27_6 r1 r3 :=
      CaseSplit.le_mxr (C27_6 r1 r3) 10 r0 (by omega)
    simp only [N27_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N27_7 r1 r4 ≤ ∑ t ∈ Finset.range n27, Wl27_7 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n27, Wl27_7 r0 r1 r2 r3 r4 r5 t
        = (if c27_1 r1 t && c27_4 r4 t then (1:ℤ) else 0)
          - (if c27_1 r1 t && c27_4 r4 t && c27_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl27_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n27, Wl27_7 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl27_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n27, Wl27_7 r0 r1 r2 r3 r4 r5 t
        = P27_7 r1 r4 - C27_7 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P27_7, C27_7]
    have hm : C27_7 r1 r4 r0 ≤ M27_7 r1 r4 :=
      CaseSplit.le_mxr (C27_7 r1 r4) 10 r0 (by omega)
    simp only [N27_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N27_8 r1 r5 ≤ ∑ t ∈ Finset.range n27, Wl27_8 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n27, Wl27_8 r0 r1 r2 r3 r4 r5 t
        = (if c27_1 r1 t && c27_5 r5 t then (1:ℤ) else 0)
          - (if c27_1 r1 t && c27_5 r5 t && c27_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl27_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n27, Wl27_8 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl27_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n27, Wl27_8 r0 r1 r2 r3 r4 r5 t
        = P27_8 r1 r5 - C27_8 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P27_8, C27_8]
    have hm : C27_8 r1 r5 r0 ≤ M27_8 r1 r5 :=
      CaseSplit.le_mxr (C27_8 r1 r5) 10 r0 (by omega)
    simp only [N27_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N27_9 r2 r3 ≤ ∑ t ∈ Finset.range n27, Wl27_9 r0 r1 r2 r3 r4 r5 t := by
    simp only [N27_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl27_9]
    exact CaseSplit.ind_nonneg _
  have hn10 : N27_10 r2 r4 ≤ ∑ t ∈ Finset.range n27, Wl27_10 r0 r1 r2 r3 r4 r5 t := by
    simp only [N27_10]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl27_10]
    exact CaseSplit.ind_nonneg _
  have hn11 : N27_11 r2 r5 ≤ ∑ t ∈ Finset.range n27, Wl27_11 r0 r1 r2 r3 r4 r5 t := by
    simp only [N27_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl27_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N27_12 r3 r4 ≤ ∑ t ∈ Finset.range n27, Wl27_12 r0 r1 r2 r3 r4 r5 t := by
    simp only [N27_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl27_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N27_13 r3 r5 ≤ ∑ t ∈ Finset.range n27, Wl27_13 r0 r1 r2 r3 r4 r5 t := by
    simp only [N27_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl27_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N27_14 r4 r5 ≤ ∑ t ∈ Finset.range n27, Wl27_14 r0 r1 r2 r3 r4 r5 t := by
    simp only [N27_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl27_14]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n27, (w27 t + 2) * Dg27 r0 r1 r2 r3 r4 r5 t = S27_0 r0 + S27_1 r1 + S27_2 r2 + S27_3 r3 + S27_4 r4 + S27_5 r5 := by
    simp only [S27_0, S27_1, S27_2, S27_3, S27_4, S27_5, Dg27, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n27, (w27 t + 2) * Dg27 r0 r1 r2 r3 r4 r5 t
      = (∑ t ∈ Finset.range n27, w27 t * Dg27 r0 r1 r2 r3 r4 r5 t)
        + 2 * (∑ t ∈ Finset.range n27, Dg27 r0 r1 r2 r3 r4 r5 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n27, w27 t)
      ≤ ∑ t ∈ Finset.range n27, w27 t * Dg27 r0 r1 r2 r3 r4 r5 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg27 r0 r1 r2 r3 r4 r5 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w27 t := wnn27 t (Finset.mem_range.mp ht)
    calc w27 t = w27 t * 1 := (mul_one _).symm
      _ ≤ w27 t * Dg27 r0 r1 r2 r3 r4 r5 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS27_0 r0 + aS27_1 r1 + aS27_2 r2 + aS27_3 r3 + aS27_4 r4 + aS27_5 r5) + (aP27_0 r0 r1 + aP27_1 r0 r2 + aP27_2 r0 r3 + aP27_3 r0 r4 + aP27_4 r0 r5 + aP27_5 r1 r2 + aP27_6 r1 r3 + aP27_7 r1 r4 + aP27_8 r1 r5 + aP27_9 r2 r3 + aP27_10 r2 r4 + aP27_11 r2 r5 + aP27_12 r3 r4 + aP27_13 r3 r5 + aP27_14 r4 r5) = (S27_0 r0 + S27_1 r1 + S27_2 r2 + S27_3 r3 + S27_4 r4 + S27_5 r5) - 2 * (N27_0 r0 r1 + N27_1 r0 r2 + N27_2 r0 r3 + N27_3 r0 r4 + N27_4 r0 r5 + N27_5 r1 r2 + N27_6 r1 r3 + N27_7 r1 r4 + N27_8 r1 r5 + N27_9 r2 r3 + N27_10 r2 r4 + N27_11 r2 r5 + N27_12 r3 r4 + N27_13 r3 r5 + N27_14 r4 r5) := by
    simp only [aS27_0, aS27_1, aS27_2, aS27_3, aS27_4, aS27_5, aP27_0, aP27_1, aP27_2, aP27_3, aP27_4, aP27_5, aP27_6, aP27_7, aP27_8, aP27_9, aP27_10, aP27_11, aP27_12, aP27_13, aP27_14, L27_0, L27_1, L27_2, L27_3, L27_4, L27_5]
    ring
  have bS0 : aS27_0 r0 ≤ MS27_0 := CaseSplit.le_mxr (aS27_0) 10 r0 (by omega)
  have bS1 : aS27_1 r1 ≤ MS27_1 := CaseSplit.le_mxr (aS27_1) 12 r1 (by omega)
  have bS2 : aS27_2 r2 ≤ MS27_2 := CaseSplit.le_mxr (aS27_2) 16 r2 (by omega)
  have bS3 : aS27_3 r3 ≤ MS27_3 := CaseSplit.le_mxr (aS27_3) 18 r3 (by omega)
  have bS4 : aS27_4 r4 ≤ MS27_4 := CaseSplit.le_mxr (aS27_4) 22 r4 (by omega)
  have bS5 : aS27_5 r5 ≤ MS27_5 := CaseSplit.le_mxr (aS27_5) 28 r5 (by omega)
  have bP0 : aP27_0 r0 r1 ≤ MP27_0 := CaseSplit.le_mxr2 (aP27_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP27_1 r0 r2 ≤ MP27_1 := CaseSplit.le_mxr2 (aP27_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP27_2 r0 r3 ≤ MP27_2 := CaseSplit.le_mxr2 (aP27_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP27_3 r0 r4 ≤ MP27_3 := CaseSplit.le_mxr2 (aP27_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP27_4 r0 r5 ≤ MP27_4 := CaseSplit.le_mxr2 (aP27_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP27_5 r1 r2 ≤ MP27_5 := CaseSplit.le_mxr2 (aP27_5) 12 16 r1 r2 (by omega) (by omega)
  have bP6 : aP27_6 r1 r3 ≤ MP27_6 := CaseSplit.le_mxr2 (aP27_6) 12 18 r1 r3 (by omega) (by omega)
  have bP7 : aP27_7 r1 r4 ≤ MP27_7 := CaseSplit.le_mxr2 (aP27_7) 12 22 r1 r4 (by omega) (by omega)
  have bP8 : aP27_8 r1 r5 ≤ MP27_8 := CaseSplit.le_mxr2 (aP27_8) 12 28 r1 r5 (by omega) (by omega)
  have bP9 : aP27_9 r2 r3 ≤ MP27_9 := CaseSplit.le_mxr2 (aP27_9) 16 18 r2 r3 (by omega) (by omega)
  have bP10 : aP27_10 r2 r4 ≤ MP27_10 := CaseSplit.le_mxr2 (aP27_10) 16 22 r2 r4 (by omega) (by omega)
  have bP11 : aP27_11 r2 r5 ≤ MP27_11 := CaseSplit.le_mxr2 (aP27_11) 16 28 r2 r5 (by omega) (by omega)
  have bP12 : aP27_12 r3 r4 ≤ MP27_12 := CaseSplit.le_mxr2 (aP27_12) 18 22 r3 r4 (by omega) (by omega)
  have bP13 : aP27_13 r3 r5 ≤ MP27_13 := CaseSplit.le_mxr2 (aP27_13) 18 28 r3 r5 (by omega) (by omega)
  have bP14 : aP27_14 r4 r5 ≤ MP27_14 := CaseSplit.le_mxr2 (aP27_14) 22 28 r4 r5 (by omega) (by omega)
  have hrhs : rhs27 = (∑ t ∈ Finset.range n27, w27 t) + 2 * (n27 : ℤ) := rfl
  have hc := cert27
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, bS0, bS1, bS2, bS3, bS4, bS5, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14]

end IncCert29
