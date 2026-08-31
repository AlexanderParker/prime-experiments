/-
INCREMENT-WIDTH CERTIFICATE, step 23->29, case 30 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_23_29.json, which re-derives every number
from the primes alone).

Machine 29, INCREMENT width 49 = F_2(23) + s_min(29) = 39 + 10,
held gears [5, 7] at phases [4, 2].  Free gears [11, 13, 17, 19, 23, 29].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 8.
-/
import IncCert29B

namespace IncCert29

/-! ### case 30: held gears at phases [4, 2] -/

def p30 : List ℕ := [1, 3, 8, 9, 14, 16, 19, 21, 23, 24, 26, 28, 29, 31, 33, 36, 38, 43, 44]
def q30 (t : ℕ) : ℕ := p30.getD t 0
def n30 : ℕ := 19
def yl30 : List ℤ := [1, 0, 1, 0, 2, 2, 4, 6, 4, 8, 8, 7, 4, 6, 5, 4, 3, 0, 0]
def w30 (t : ℕ) : ℤ := yl30.getD t 0
def ul30 : List ℤ := [0, 2, 0, 0, 0, 2, 0, 0, 2, 0, (-1), 0, 2, (-2), (-2), (-2), (-2), (-2), (-2), 0, (-2), 0, (-2), 0, 3, 3, 0, 0, 0, 3, 2, 0, 0, 0, 3, 4, 3, 0, 0, 0, 3, (-3), (-4), (-4), (-3), (-3), (-3), (-3), (-4), (-4), (-3), 0, (-8), (-6), 0, 0, (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), 0, (-1), 0, 0, 0, (-1), 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-3), 0, 0, 0, 0, 0, 0, 0, 0, 15, 11, 10, 9, 15, 13, 15, 15, 14, 9, 15, 15, 15, 15, 7, 12, 15, (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), 15, 15, 15, 15, 9, 7, 15, 12, 15, 11, 12, 14, 14, 15, 15, 14, 15, 14, 15, (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-19), (-15), (-15), (-15), (-15), (-15), 1, 7, 1, 12, 12, 12, 12, 12, 1, 12, 0, 12, 12, 12, 12, 1, 7, 1, 10, 12, 1, 12, 12, (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-11), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 14, 3, 17, 4, 14, 9, 0, 15, 3, 17, 7, 0, 9, 0, 17, 4, 7, 5, 3, 14, 17, 8, 4, 17, 17, 4, 17, 0, 6, 6, 0, 6, 1, (-8), 6, (-9), 6, (-5), 3, 4, (-6), (-1), 4, 0, 6, 6, 6, 6, 0, 6, 0]
def u30 (k : ℕ) : ℤ := ul30.getD k 0

def c30_0 (r t : ℕ) : Bool := gb11 r (q30 t)
def c30_1 (r t : ℕ) : Bool := gb13 r (q30 t)
def c30_2 (r t : ℕ) : Bool := gb17 r (q30 t)
def c30_3 (r t : ℕ) : Bool := gb19 r (q30 t)
def c30_4 (r t : ℕ) : Bool := gb23 r (q30 t)
def c30_5 (r t : ℕ) : Bool := gb29 r (q30 t)

def S30_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (w30 t + 3) * (if c30_0 r t then 1 else 0)
def S30_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (w30 t + 3) * (if c30_1 r t then 1 else 0)
def S30_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (w30 t + 3) * (if c30_2 r t then 1 else 0)
def S30_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (w30 t + 3) * (if c30_3 r t then 1 else 0)
def S30_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (w30 t + 3) * (if c30_4 r t then 1 else 0)
def S30_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (w30 t + 3) * (if c30_5 r t then 1 else 0)

def L30_0 (r : ℕ) : ℤ := u30 (13 + r) + u30 (41 + r) + u30 (71 + r) + u30 (105 + r) + u30 (145 + r)
def L30_1 (r : ℕ) : ℤ := u30 (0 + r) + u30 (173 + r) + u30 (205 + r) + u30 (241 + r) + u30 (283 + r)
def L30_2 (r : ℕ) : ℤ := u30 (24 + r) + u30 (156 + r) + u30 (315 + r) + u30 (355 + r) + u30 (401 + r)
def L30_3 (r : ℕ) : ℤ := u30 (52 + r) + u30 (186 + r) + u30 (296 + r) + u30 (441 + r) + u30 (489 + r)
def L30_4 (r : ℕ) : ℤ := u30 (82 + r) + u30 (218 + r) + u30 (332 + r) + u30 (418 + r) + u30 (537 + r)
def L30_5 (r : ℕ) : ℤ := u30 (116 + r) + u30 (254 + r) + u30 (372 + r) + u30 (460 + r) + u30 (508 + r)

def aS30_0 (r : ℕ) : ℤ := S30_0 r - L30_0 r
def MS30_0 : ℤ := CaseSplit.mxr (aS30_0) 10
def aS30_1 (r : ℕ) : ℤ := S30_1 r - L30_1 r
def MS30_1 : ℤ := CaseSplit.mxr (aS30_1) 12
def aS30_2 (r : ℕ) : ℤ := S30_2 r - L30_2 r
def MS30_2 : ℤ := CaseSplit.mxr (aS30_2) 16
def aS30_3 (r : ℕ) : ℤ := S30_3 r - L30_3 r
def MS30_3 : ℤ := CaseSplit.mxr (aS30_3) 18
def aS30_4 (r : ℕ) : ℤ := S30_4 r - L30_4 r
def MS30_4 : ℤ := CaseSplit.mxr (aS30_4) 22
def aS30_5 (r : ℕ) : ℤ := S30_5 r - L30_5 r
def MS30_5 : ℤ := CaseSplit.mxr (aS30_5) 28

def N30_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (if c30_0 ra t && c30_1 rb t then 1 else 0)
def aP30_0 (ra rb : ℕ) : ℤ := -(3) * N30_0 ra rb + u30 (0 + rb) + u30 (13 + ra)
def MP30_0 : ℤ := CaseSplit.mxr2 (aP30_0) 10 12
def N30_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (if c30_0 ra t && c30_2 rb t then 1 else 0)
def aP30_1 (ra rb : ℕ) : ℤ := -(3) * N30_1 ra rb + u30 (24 + rb) + u30 (41 + ra)
def MP30_1 : ℤ := CaseSplit.mxr2 (aP30_1) 10 16
def N30_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (if c30_0 ra t && c30_3 rb t then 1 else 0)
def aP30_2 (ra rb : ℕ) : ℤ := -(3) * N30_2 ra rb + u30 (52 + rb) + u30 (71 + ra)
def MP30_2 : ℤ := CaseSplit.mxr2 (aP30_2) 10 18
def N30_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (if c30_0 ra t && c30_4 rb t then 1 else 0)
def aP30_3 (ra rb : ℕ) : ℤ := -(3) * N30_3 ra rb + u30 (82 + rb) + u30 (105 + ra)
def MP30_3 : ℤ := CaseSplit.mxr2 (aP30_3) 10 22
def N30_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (if c30_0 ra t && c30_5 rb t then 1 else 0)
def aP30_4 (ra rb : ℕ) : ℤ := -(3) * N30_4 ra rb + u30 (116 + rb) + u30 (145 + ra)
def MP30_4 : ℤ := CaseSplit.mxr2 (aP30_4) 10 28
def P30_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (if c30_1 ra t && c30_2 rb t then 1 else 0)
def C30_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (if c30_1 ra t && c30_2 rb t && c30_0 s t then 1 else 0)
def M30_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C30_5 ra rb) 10
def E30_5 : List ℕ := [39, 45, 61, 67, 86, 97, 140, 151, 170, 176, 192, 198]
def N30_5 (ra rb : ℕ) : ℤ := if E30_5.contains (ra * 17 + rb) = true then P30_5 ra rb - M30_5 ra rb else 0
def aP30_5 (ra rb : ℕ) : ℤ := -(3) * N30_5 ra rb + u30 (156 + rb) + u30 (173 + ra)
def MP30_5 : ℤ := CaseSplit.mxr2 (aP30_5) 12 16
def P30_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (if c30_1 ra t && c30_3 rb t then 1 else 0)
def C30_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (if c30_1 ra t && c30_3 rb t && c30_0 s t then 1 else 0)
def M30_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C30_6 ra rb) 10
def E30_6 : List ℕ := [21, 37, 71, 113, 124, 147, 152, 158, 192, 200, 228, 234]
def N30_6 (ra rb : ℕ) : ℤ := if E30_6.contains (ra * 19 + rb) = true then P30_6 ra rb - M30_6 ra rb else 0
def aP30_6 (ra rb : ℕ) : ℤ := -(3) * N30_6 ra rb + u30 (186 + rb) + u30 (205 + ra)
def MP30_6 : ℤ := CaseSplit.mxr2 (aP30_6) 12 18
def P30_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (if c30_1 ra t && c30_4 rb t then 1 else 0)
def C30_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (if c30_1 ra t && c30_4 rb t && c30_0 s t then 1 else 0)
def M30_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C30_7 ra rb) 10
def E30_7 : List ℕ := []
def N30_7 (ra rb : ℕ) : ℤ := if E30_7.contains (ra * 23 + rb) = true then P30_7 ra rb - M30_7 ra rb else 0
def aP30_7 (ra rb : ℕ) : ℤ := -(3) * N30_7 ra rb + u30 (218 + rb) + u30 (241 + ra)
def MP30_7 : ℤ := CaseSplit.mxr2 (aP30_7) 12 22
def P30_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (if c30_1 ra t && c30_5 rb t then 1 else 0)
def C30_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (if c30_1 ra t && c30_5 rb t && c30_0 s t then 1 else 0)
def M30_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C30_8 ra rb) 10
def E30_8 : List ℕ := []
def N30_8 (ra rb : ℕ) : ℤ := if E30_8.contains (ra * 29 + rb) = true then P30_8 ra rb - M30_8 ra rb else 0
def aP30_8 (ra rb : ℕ) : ℤ := -(3) * N30_8 ra rb + u30 (254 + rb) + u30 (283 + ra)
def MP30_8 : ℤ := CaseSplit.mxr2 (aP30_8) 12 28
def N30_9 (_ra _rb : ℕ) : ℤ := 0
def aP30_9 (ra rb : ℕ) : ℤ := -(3) * N30_9 ra rb + u30 (296 + rb) + u30 (315 + ra)
def MP30_9 : ℤ := CaseSplit.mxr2 (aP30_9) 16 18
def N30_10 (_ra _rb : ℕ) : ℤ := 0
def aP30_10 (ra rb : ℕ) : ℤ := -(3) * N30_10 ra rb + u30 (332 + rb) + u30 (355 + ra)
def MP30_10 : ℤ := CaseSplit.mxr2 (aP30_10) 16 22
def N30_11 (_ra _rb : ℕ) : ℤ := 0
def aP30_11 (ra rb : ℕ) : ℤ := -(3) * N30_11 ra rb + u30 (372 + rb) + u30 (401 + ra)
def MP30_11 : ℤ := CaseSplit.mxr2 (aP30_11) 16 28
def N30_12 (_ra _rb : ℕ) : ℤ := 0
def aP30_12 (ra rb : ℕ) : ℤ := -(3) * N30_12 ra rb + u30 (418 + rb) + u30 (441 + ra)
def MP30_12 : ℤ := CaseSplit.mxr2 (aP30_12) 18 22
def N30_13 (_ra _rb : ℕ) : ℤ := 0
def aP30_13 (ra rb : ℕ) : ℤ := -(3) * N30_13 ra rb + u30 (460 + rb) + u30 (489 + ra)
def MP30_13 : ℤ := CaseSplit.mxr2 (aP30_13) 18 28
def N30_14 (_ra _rb : ℕ) : ℤ := 0
def aP30_14 (ra rb : ℕ) : ℤ := -(3) * N30_14 ra rb + u30 (508 + rb) + u30 (537 + ra)
def MP30_14 : ℤ := CaseSplit.mxr2 (aP30_14) 22 28

def rhs30 : ℤ := (∑ t ∈ Finset.range n30, w30 t) + 3 * (n30 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn30 : ∀ t, t < n30 → (0 : ℤ) ≤ w30 t := by decide
theorem plt30 : ∀ t, t < n30 → q30 t < 49 := by decide
theorem pfree30_5 : ∀ t, t < n30 → gb5 4 (q30 t) = false := by decide
theorem pfree30_7 : ∀ t, t < n30 → gb7 2 (q30 t) = false := by decide
theorem MSv30_0 : MS30_0 = 27 := by decide +kernel
theorem MSv30_1 : MS30_1 = 64 := by decide +kernel
theorem MSv30_2 : MS30_2 = 1 := by decide +kernel
theorem MSv30_3 : MS30_3 = 2 := by decide +kernel
theorem MSv30_4 : MS30_4 = 2 := by decide +kernel
theorem MSv30_5 : MS30_5 = 2 := by decide +kernel
theorem MPv30_0 : MP30_0 = 0 := by decide +kernel
theorem MPv30_1 : MP30_1 = 0 := by decide +kernel
theorem MPv30_2 : MP30_2 = 0 := by decide +kernel
theorem MPv30_3 : MP30_3 = 0 := by decide +kernel
theorem MPv30_4 : MP30_4 = 0 := by decide +kernel
theorem MPv30_5 : MP30_5 = 0 := by decide +kernel
theorem MPv30_6 : MP30_6 = 0 := by decide +kernel
theorem MPv30_7 : MP30_7 = 0 := by decide +kernel
theorem MPv30_8 : MP30_8 = 0 := by decide +kernel
theorem MPv30_9 : MP30_9 = 0 := by decide +kernel
theorem MPv30_10 : MP30_10 = 0 := by decide +kernel
theorem MPv30_11 : MP30_11 = 0 := by decide +kernel
theorem MPv30_12 : MP30_12 = 0 := by decide +kernel
theorem MPv30_13 : MP30_13 = 0 := by decide +kernel
theorem MPv30_14 : MP30_14 = 23 := by decide +kernel
theorem rhsv30 : rhs30 = 122 := by decide +kernel

/-- **The case-30 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/8.
    (Scaled by the common denominator 8: 121 < 122.) -/
theorem cert30 : MS30_0 + MS30_1 + MS30_2 + MS30_3 + MS30_4 + MS30_5 + MP30_0 + MP30_1 + MP30_2 + MP30_3 + MP30_4 + MP30_5 + MP30_6 + MP30_7 + MP30_8 + MP30_9 + MP30_10 + MP30_11 + MP30_12 + MP30_13 + MP30_14 < rhs30 := by
  rw [MSv30_0, MSv30_1, MSv30_2, MSv30_3, MSv30_4, MSv30_5, MPv30_0, MPv30_1, MPv30_2, MPv30_3, MPv30_4, MPv30_5, MPv30_6, MPv30_7, MPv30_8, MPv30_9, MPv30_10, MPv30_11, MPv30_12, MPv30_13, MPv30_14, rhsv30]
  decide

def Dg30 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := (if c30_0 r0 t then 1 else 0) + (if c30_1 r1 t then 1 else 0) + (if c30_2 r2 t then 1 else 0) + (if c30_3 r3 t then 1 else 0) + (if c30_4 r4 t then 1 else 0) + (if c30_5 r5 t then 1 else 0)
def Wl30_0 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c30_0 r0 t && c30_1 r1 t then 1 else 0
def Wl30_1 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c30_0 r0 t && c30_2 r2 t then 1 else 0
def Wl30_2 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c30_0 r0 t && c30_3 r3 t then 1 else 0
def Wl30_3 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c30_0 r0 t && c30_4 r4 t then 1 else 0
def Wl30_4 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c30_0 r0 t && c30_5 r5 t then 1 else 0
def Wl30_5 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c30_0 r0 t && c30_1 r1 t && c30_2 r2 t then 1 else 0
def Wl30_6 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c30_0 r0 t && c30_1 r1 t && c30_3 r3 t then 1 else 0
def Wl30_7 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c30_0 r0 t && c30_1 r1 t && c30_4 r4 t then 1 else 0
def Wl30_8 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c30_0 r0 t && c30_1 r1 t && c30_5 r5 t then 1 else 0
def Wl30_9 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c30_0 r0 t && !c30_1 r1 t && c30_2 r2 t && c30_3 r3 t then 1 else 0
def Wl30_10 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c30_0 r0 t && !c30_1 r1 t && c30_2 r2 t && c30_4 r4 t then 1 else 0
def Wl30_11 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c30_0 r0 t && !c30_1 r1 t && c30_2 r2 t && c30_5 r5 t then 1 else 0
def Wl30_12 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c30_0 r0 t && !c30_1 r1 t && !c30_2 r2 t && c30_3 r3 t && c30_4 r4 t then 1 else 0
def Wl30_13 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c30_0 r0 t && !c30_1 r1 t && !c30_2 r2 t && c30_3 r3 t && c30_5 r5 t then 1 else 0
def Wl30_14 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c30_0 r0 t && !c30_1 r1 t && !c30_2 r2 t && !c30_3 r3 t && c30_4 r4 t && c30_5 r5 t then 1 else 0

/-- **No configuration blocks the whole window in case 30.** -/
theorem nocov30 {r0 r1 r2 r3 r4 r5 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29)
    (hcov : ∀ t, t < n30 → (c30_0 r0 t || c30_1 r1 t || c30_2 r2 t || c30_3 r3 t || c30_4 r4 t || c30_5 r5 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n30, (1 : ℤ) + (Wl30_0 r0 r1 r2 r3 r4 r5 t + Wl30_1 r0 r1 r2 r3 r4 r5 t + Wl30_2 r0 r1 r2 r3 r4 r5 t + Wl30_3 r0 r1 r2 r3 r4 r5 t + Wl30_4 r0 r1 r2 r3 r4 r5 t + Wl30_5 r0 r1 r2 r3 r4 r5 t + Wl30_6 r0 r1 r2 r3 r4 r5 t + Wl30_7 r0 r1 r2 r3 r4 r5 t + Wl30_8 r0 r1 r2 r3 r4 r5 t + Wl30_9 r0 r1 r2 r3 r4 r5 t + Wl30_10 r0 r1 r2 r3 r4 r5 t + Wl30_11 r0 r1 r2 r3 r4 r5 t + Wl30_12 r0 r1 r2 r3 r4 r5 t + Wl30_13 r0 r1 r2 r3 r4 r5 t + Wl30_14 r0 r1 r2 r3 r4 r5 t) ≤ Dg30 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Wl30_0, Wl30_1, Wl30_2, Wl30_3, Wl30_4, Wl30_5, Wl30_6, Wl30_7, Wl30_8, Wl30_9, Wl30_10, Wl30_11, Wl30_12, Wl30_13, Wl30_14, Dg30]
    exact CaseSplit.lowest6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n30, (1 : ℤ) ≤ Dg30 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Dg30]
    exact CaseSplit.degpos6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n30 : ℤ) + ((∑ t ∈ Finset.range n30, Wl30_0 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n30, Wl30_1 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n30, Wl30_2 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n30, Wl30_3 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n30, Wl30_4 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n30, Wl30_5 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n30, Wl30_6 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n30, Wl30_7 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n30, Wl30_8 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n30, Wl30_9 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n30, Wl30_10 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n30, Wl30_11 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n30, Wl30_12 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n30, Wl30_13 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n30, Wl30_14 r0 r1 r2 r3 r4 r5 t)) ≤ ∑ t ∈ Finset.range n30, Dg30 r0 r1 r2 r3 r4 r5 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N30_0 r0 r1 ≤ ∑ t ∈ Finset.range n30, Wl30_0 r0 r1 r2 r3 r4 r5 t := by
    simp only [N30_0, Wl30_0, le_refl]
  have hn1 : N30_1 r0 r2 ≤ ∑ t ∈ Finset.range n30, Wl30_1 r0 r1 r2 r3 r4 r5 t := by
    simp only [N30_1, Wl30_1, le_refl]
  have hn2 : N30_2 r0 r3 ≤ ∑ t ∈ Finset.range n30, Wl30_2 r0 r1 r2 r3 r4 r5 t := by
    simp only [N30_2, Wl30_2, le_refl]
  have hn3 : N30_3 r0 r4 ≤ ∑ t ∈ Finset.range n30, Wl30_3 r0 r1 r2 r3 r4 r5 t := by
    simp only [N30_3, Wl30_3, le_refl]
  have hn4 : N30_4 r0 r5 ≤ ∑ t ∈ Finset.range n30, Wl30_4 r0 r1 r2 r3 r4 r5 t := by
    simp only [N30_4, Wl30_4, le_refl]
  have hn5 : N30_5 r1 r2 ≤ ∑ t ∈ Finset.range n30, Wl30_5 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n30, Wl30_5 r0 r1 r2 r3 r4 r5 t
        = (if c30_1 r1 t && c30_2 r2 t then (1:ℤ) else 0)
          - (if c30_1 r1 t && c30_2 r2 t && c30_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl30_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n30, Wl30_5 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl30_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n30, Wl30_5 r0 r1 r2 r3 r4 r5 t
        = P30_5 r1 r2 - C30_5 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P30_5, C30_5]
    have hm : C30_5 r1 r2 r0 ≤ M30_5 r1 r2 :=
      CaseSplit.le_mxr (C30_5 r1 r2) 10 r0 (by omega)
    simp only [N30_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N30_6 r1 r3 ≤ ∑ t ∈ Finset.range n30, Wl30_6 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n30, Wl30_6 r0 r1 r2 r3 r4 r5 t
        = (if c30_1 r1 t && c30_3 r3 t then (1:ℤ) else 0)
          - (if c30_1 r1 t && c30_3 r3 t && c30_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl30_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n30, Wl30_6 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl30_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n30, Wl30_6 r0 r1 r2 r3 r4 r5 t
        = P30_6 r1 r3 - C30_6 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P30_6, C30_6]
    have hm : C30_6 r1 r3 r0 ≤ M30_6 r1 r3 :=
      CaseSplit.le_mxr (C30_6 r1 r3) 10 r0 (by omega)
    simp only [N30_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N30_7 r1 r4 ≤ ∑ t ∈ Finset.range n30, Wl30_7 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n30, Wl30_7 r0 r1 r2 r3 r4 r5 t
        = (if c30_1 r1 t && c30_4 r4 t then (1:ℤ) else 0)
          - (if c30_1 r1 t && c30_4 r4 t && c30_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl30_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n30, Wl30_7 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl30_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n30, Wl30_7 r0 r1 r2 r3 r4 r5 t
        = P30_7 r1 r4 - C30_7 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P30_7, C30_7]
    have hm : C30_7 r1 r4 r0 ≤ M30_7 r1 r4 :=
      CaseSplit.le_mxr (C30_7 r1 r4) 10 r0 (by omega)
    simp only [N30_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N30_8 r1 r5 ≤ ∑ t ∈ Finset.range n30, Wl30_8 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n30, Wl30_8 r0 r1 r2 r3 r4 r5 t
        = (if c30_1 r1 t && c30_5 r5 t then (1:ℤ) else 0)
          - (if c30_1 r1 t && c30_5 r5 t && c30_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl30_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n30, Wl30_8 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl30_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n30, Wl30_8 r0 r1 r2 r3 r4 r5 t
        = P30_8 r1 r5 - C30_8 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P30_8, C30_8]
    have hm : C30_8 r1 r5 r0 ≤ M30_8 r1 r5 :=
      CaseSplit.le_mxr (C30_8 r1 r5) 10 r0 (by omega)
    simp only [N30_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N30_9 r2 r3 ≤ ∑ t ∈ Finset.range n30, Wl30_9 r0 r1 r2 r3 r4 r5 t := by
    simp only [N30_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl30_9]
    exact CaseSplit.ind_nonneg _
  have hn10 : N30_10 r2 r4 ≤ ∑ t ∈ Finset.range n30, Wl30_10 r0 r1 r2 r3 r4 r5 t := by
    simp only [N30_10]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl30_10]
    exact CaseSplit.ind_nonneg _
  have hn11 : N30_11 r2 r5 ≤ ∑ t ∈ Finset.range n30, Wl30_11 r0 r1 r2 r3 r4 r5 t := by
    simp only [N30_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl30_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N30_12 r3 r4 ≤ ∑ t ∈ Finset.range n30, Wl30_12 r0 r1 r2 r3 r4 r5 t := by
    simp only [N30_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl30_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N30_13 r3 r5 ≤ ∑ t ∈ Finset.range n30, Wl30_13 r0 r1 r2 r3 r4 r5 t := by
    simp only [N30_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl30_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N30_14 r4 r5 ≤ ∑ t ∈ Finset.range n30, Wl30_14 r0 r1 r2 r3 r4 r5 t := by
    simp only [N30_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl30_14]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n30, (w30 t + 3) * Dg30 r0 r1 r2 r3 r4 r5 t = S30_0 r0 + S30_1 r1 + S30_2 r2 + S30_3 r3 + S30_4 r4 + S30_5 r5 := by
    simp only [S30_0, S30_1, S30_2, S30_3, S30_4, S30_5, Dg30, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n30, (w30 t + 3) * Dg30 r0 r1 r2 r3 r4 r5 t
      = (∑ t ∈ Finset.range n30, w30 t * Dg30 r0 r1 r2 r3 r4 r5 t)
        + 3 * (∑ t ∈ Finset.range n30, Dg30 r0 r1 r2 r3 r4 r5 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n30, w30 t)
      ≤ ∑ t ∈ Finset.range n30, w30 t * Dg30 r0 r1 r2 r3 r4 r5 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg30 r0 r1 r2 r3 r4 r5 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w30 t := wnn30 t (Finset.mem_range.mp ht)
    calc w30 t = w30 t * 1 := (mul_one _).symm
      _ ≤ w30 t * Dg30 r0 r1 r2 r3 r4 r5 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS30_0 r0 + aS30_1 r1 + aS30_2 r2 + aS30_3 r3 + aS30_4 r4 + aS30_5 r5) + (aP30_0 r0 r1 + aP30_1 r0 r2 + aP30_2 r0 r3 + aP30_3 r0 r4 + aP30_4 r0 r5 + aP30_5 r1 r2 + aP30_6 r1 r3 + aP30_7 r1 r4 + aP30_8 r1 r5 + aP30_9 r2 r3 + aP30_10 r2 r4 + aP30_11 r2 r5 + aP30_12 r3 r4 + aP30_13 r3 r5 + aP30_14 r4 r5) = (S30_0 r0 + S30_1 r1 + S30_2 r2 + S30_3 r3 + S30_4 r4 + S30_5 r5) - 3 * (N30_0 r0 r1 + N30_1 r0 r2 + N30_2 r0 r3 + N30_3 r0 r4 + N30_4 r0 r5 + N30_5 r1 r2 + N30_6 r1 r3 + N30_7 r1 r4 + N30_8 r1 r5 + N30_9 r2 r3 + N30_10 r2 r4 + N30_11 r2 r5 + N30_12 r3 r4 + N30_13 r3 r5 + N30_14 r4 r5) := by
    simp only [aS30_0, aS30_1, aS30_2, aS30_3, aS30_4, aS30_5, aP30_0, aP30_1, aP30_2, aP30_3, aP30_4, aP30_5, aP30_6, aP30_7, aP30_8, aP30_9, aP30_10, aP30_11, aP30_12, aP30_13, aP30_14, L30_0, L30_1, L30_2, L30_3, L30_4, L30_5]
    ring
  have bS0 : aS30_0 r0 ≤ MS30_0 := CaseSplit.le_mxr (aS30_0) 10 r0 (by omega)
  have bS1 : aS30_1 r1 ≤ MS30_1 := CaseSplit.le_mxr (aS30_1) 12 r1 (by omega)
  have bS2 : aS30_2 r2 ≤ MS30_2 := CaseSplit.le_mxr (aS30_2) 16 r2 (by omega)
  have bS3 : aS30_3 r3 ≤ MS30_3 := CaseSplit.le_mxr (aS30_3) 18 r3 (by omega)
  have bS4 : aS30_4 r4 ≤ MS30_4 := CaseSplit.le_mxr (aS30_4) 22 r4 (by omega)
  have bS5 : aS30_5 r5 ≤ MS30_5 := CaseSplit.le_mxr (aS30_5) 28 r5 (by omega)
  have bP0 : aP30_0 r0 r1 ≤ MP30_0 := CaseSplit.le_mxr2 (aP30_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP30_1 r0 r2 ≤ MP30_1 := CaseSplit.le_mxr2 (aP30_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP30_2 r0 r3 ≤ MP30_2 := CaseSplit.le_mxr2 (aP30_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP30_3 r0 r4 ≤ MP30_3 := CaseSplit.le_mxr2 (aP30_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP30_4 r0 r5 ≤ MP30_4 := CaseSplit.le_mxr2 (aP30_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP30_5 r1 r2 ≤ MP30_5 := CaseSplit.le_mxr2 (aP30_5) 12 16 r1 r2 (by omega) (by omega)
  have bP6 : aP30_6 r1 r3 ≤ MP30_6 := CaseSplit.le_mxr2 (aP30_6) 12 18 r1 r3 (by omega) (by omega)
  have bP7 : aP30_7 r1 r4 ≤ MP30_7 := CaseSplit.le_mxr2 (aP30_7) 12 22 r1 r4 (by omega) (by omega)
  have bP8 : aP30_8 r1 r5 ≤ MP30_8 := CaseSplit.le_mxr2 (aP30_8) 12 28 r1 r5 (by omega) (by omega)
  have bP9 : aP30_9 r2 r3 ≤ MP30_9 := CaseSplit.le_mxr2 (aP30_9) 16 18 r2 r3 (by omega) (by omega)
  have bP10 : aP30_10 r2 r4 ≤ MP30_10 := CaseSplit.le_mxr2 (aP30_10) 16 22 r2 r4 (by omega) (by omega)
  have bP11 : aP30_11 r2 r5 ≤ MP30_11 := CaseSplit.le_mxr2 (aP30_11) 16 28 r2 r5 (by omega) (by omega)
  have bP12 : aP30_12 r3 r4 ≤ MP30_12 := CaseSplit.le_mxr2 (aP30_12) 18 22 r3 r4 (by omega) (by omega)
  have bP13 : aP30_13 r3 r5 ≤ MP30_13 := CaseSplit.le_mxr2 (aP30_13) 18 28 r3 r5 (by omega) (by omega)
  have bP14 : aP30_14 r4 r5 ≤ MP30_14 := CaseSplit.le_mxr2 (aP30_14) 22 28 r4 r5 (by omega) (by omega)
  have hrhs : rhs30 = (∑ t ∈ Finset.range n30, w30 t) + 3 * (n30 : ℤ) := rfl
  have hc := cert30
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, bS0, bS1, bS2, bS3, bS4, bS5, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14]

end IncCert29
