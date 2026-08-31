/-
INCREMENT-WIDTH CERTIFICATE, step 23->29, case 14 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_23_29.json, which re-derives every number
from the primes alone).

Machine 29, INCREMENT width 49 = F_2(23) + s_min(29) = 39 + 10,
held gears [5, 7] at phases [2, 0].  Free gears [11, 13, 17, 19, 23, 29].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 8.
-/
import IncCert29B

namespace IncCert29

/-! ### case 14: held gears at phases [2, 0] -/

def p14 : List ℕ := [0, 3, 5, 10, 11, 16, 18, 21, 23, 25, 26, 28, 30, 31, 33, 35, 38, 40, 45, 46]
def q14 (t : ℕ) : ℕ := p14.getD t 0
def n14 : ℕ := 20
def yl14 : List ℤ := [0, 2, 0, 0, 0, 1, 2, 5, 4, 4, 7, 8, 6, 3, 7, 5, 3, 2, 0, 1]
def w14 (t : ℕ) : ℤ := yl14.getD t 0
def ul14 : List ℤ := [0, 0, (-1), 0, (-5), (-9), 0, 0, 1, 0, (-3), 0, 0, 0, (-1), 0, (-1), 0, (-1), 0, 0, 0, 0, 0, (-3), (-3), (-3), 0, (-3), (-3), (-3), (-3), (-3), 0, (-3), (-3), (-3), (-3), (-3), (-3), (-3), 0, 0, 3, 3, 0, 0, 0, 3, 3, 3, 0, 3, 1, 0, 0, 0, 0, 1, 0, 0, 0, (-1), 0, (-4), 1, 3, 0, 1, 0, 0, (-1), (-3), (-3), (-3), (-3), (-3), (-1), (-1), (-3), (-3), 0, (-3), 0, (-3), (-3), 0, (-3), (-3), (-3), 0, 0, (-3), (-3), (-3), (-3), (-3), (-3), 0, 0, (-3), 0, (-3), (-3), (-3), 0, 0, 3, 0, (-1), 0, 0, (-1), 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-9), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-7), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 18, 18, 18, 18, 18, 15, 18, 18, 18, 18, 17, 9, 15, 18, 18, 18, (-18), (-18), (-18), (-18), (-18), (-18), (-18), (-18), (-18), (-18), (-18), (-18), (-18), 14, 14, 14, 8, 14, 14, 14, 11, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 7, (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), 3, 16, 16, 16, 16, 2, 16, 8, 16, 14, 16, 16, 14, 0, 13, 16, 14, 16, 0, 16, 16, 8, 16, (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-12), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-3), 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, (-6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-7), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-7), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 16, 16, 16, 8, 3, 15, 3, 16, 16, 0, 7, 0, 16, 3, 6, 5, 3, 14, 3, 16, 16, 6, 16, 16, 16, 0, 10, 15, 0, 3, 3, (-9), 3, 3, (-6), 3, (-13), 3, (-8), (-4), 3, 3, 3, (-5), 3, 3, 3, 3, 3, 3, 0]
def u14 (k : ℕ) : ℤ := ul14.getD k 0

def c14_0 (r t : ℕ) : Bool := gb11 r (q14 t)
def c14_1 (r t : ℕ) : Bool := gb13 r (q14 t)
def c14_2 (r t : ℕ) : Bool := gb17 r (q14 t)
def c14_3 (r t : ℕ) : Bool := gb19 r (q14 t)
def c14_4 (r t : ℕ) : Bool := gb23 r (q14 t)
def c14_5 (r t : ℕ) : Bool := gb29 r (q14 t)

def S14_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (w14 t + 3) * (if c14_0 r t then 1 else 0)
def S14_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (w14 t + 3) * (if c14_1 r t then 1 else 0)
def S14_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (w14 t + 3) * (if c14_2 r t then 1 else 0)
def S14_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (w14 t + 3) * (if c14_3 r t then 1 else 0)
def S14_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (w14 t + 3) * (if c14_4 r t then 1 else 0)
def S14_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (w14 t + 3) * (if c14_5 r t then 1 else 0)

def L14_0 (r : ℕ) : ℤ := u14 (13 + r) + u14 (41 + r) + u14 (71 + r) + u14 (105 + r) + u14 (145 + r)
def L14_1 (r : ℕ) : ℤ := u14 (0 + r) + u14 (173 + r) + u14 (205 + r) + u14 (241 + r) + u14 (283 + r)
def L14_2 (r : ℕ) : ℤ := u14 (24 + r) + u14 (156 + r) + u14 (315 + r) + u14 (355 + r) + u14 (401 + r)
def L14_3 (r : ℕ) : ℤ := u14 (52 + r) + u14 (186 + r) + u14 (296 + r) + u14 (441 + r) + u14 (489 + r)
def L14_4 (r : ℕ) : ℤ := u14 (82 + r) + u14 (218 + r) + u14 (332 + r) + u14 (418 + r) + u14 (537 + r)
def L14_5 (r : ℕ) : ℤ := u14 (116 + r) + u14 (254 + r) + u14 (372 + r) + u14 (460 + r) + u14 (508 + r)

def aS14_0 (r : ℕ) : ℤ := S14_0 r - L14_0 r
def MS14_0 : ℤ := CaseSplit.mxr (aS14_0) 10
def aS14_1 (r : ℕ) : ℤ := S14_1 r - L14_1 r
def MS14_1 : ℤ := CaseSplit.mxr (aS14_1) 12
def aS14_2 (r : ℕ) : ℤ := S14_2 r - L14_2 r
def MS14_2 : ℤ := CaseSplit.mxr (aS14_2) 16
def aS14_3 (r : ℕ) : ℤ := S14_3 r - L14_3 r
def MS14_3 : ℤ := CaseSplit.mxr (aS14_3) 18
def aS14_4 (r : ℕ) : ℤ := S14_4 r - L14_4 r
def MS14_4 : ℤ := CaseSplit.mxr (aS14_4) 22
def aS14_5 (r : ℕ) : ℤ := S14_5 r - L14_5 r
def MS14_5 : ℤ := CaseSplit.mxr (aS14_5) 28

def N14_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (if c14_0 ra t && c14_1 rb t then 1 else 0)
def aP14_0 (ra rb : ℕ) : ℤ := -(3) * N14_0 ra rb + u14 (0 + rb) + u14 (13 + ra)
def MP14_0 : ℤ := CaseSplit.mxr2 (aP14_0) 10 12
def N14_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (if c14_0 ra t && c14_2 rb t then 1 else 0)
def aP14_1 (ra rb : ℕ) : ℤ := -(3) * N14_1 ra rb + u14 (24 + rb) + u14 (41 + ra)
def MP14_1 : ℤ := CaseSplit.mxr2 (aP14_1) 10 16
def N14_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (if c14_0 ra t && c14_3 rb t then 1 else 0)
def aP14_2 (ra rb : ℕ) : ℤ := -(3) * N14_2 ra rb + u14 (52 + rb) + u14 (71 + ra)
def MP14_2 : ℤ := CaseSplit.mxr2 (aP14_2) 10 18
def N14_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (if c14_0 ra t && c14_4 rb t then 1 else 0)
def aP14_3 (ra rb : ℕ) : ℤ := -(3) * N14_3 ra rb + u14 (82 + rb) + u14 (105 + ra)
def MP14_3 : ℤ := CaseSplit.mxr2 (aP14_3) 10 22
def N14_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (if c14_0 ra t && c14_5 rb t then 1 else 0)
def aP14_4 (ra rb : ℕ) : ℤ := -(3) * N14_4 ra rb + u14 (116 + rb) + u14 (145 + ra)
def MP14_4 : ℤ := CaseSplit.mxr2 (aP14_4) 10 28
def P14_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (if c14_1 ra t && c14_2 rb t then 1 else 0)
def C14_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (if c14_1 ra t && c14_2 rb t && c14_0 s t then 1 else 0)
def M14_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C14_5 ra rb) 10
def E14_5 : List ℕ := [3, 9, 25, 31, 61, 67, 104, 115, 140, 151, 156, 162]
def N14_5 (ra rb : ℕ) : ℤ := if E14_5.contains (ra * 17 + rb) = true then P14_5 ra rb - M14_5 ra rb else 0
def aP14_5 (ra rb : ℕ) : ℤ := -(3) * N14_5 ra rb + u14 (156 + rb) + u14 (173 + ra)
def MP14_5 : ℤ := CaseSplit.mxr2 (aP14_5) 12 16
def P14_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (if c14_1 ra t && c14_3 rb t then 1 else 0)
def C14_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (if c14_1 ra t && c14_3 rb t && c14_0 s t then 1 else 0)
def M14_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C14_6 ra rb) 10
def E14_6 : List ℕ := [31, 73, 84, 107, 118, 131, 152, 160, 194, 207, 228, 244]
def N14_6 (ra rb : ℕ) : ℤ := if E14_6.contains (ra * 19 + rb) = true then P14_6 ra rb - M14_6 ra rb else 0
def aP14_6 (ra rb : ℕ) : ℤ := -(3) * N14_6 ra rb + u14 (186 + rb) + u14 (205 + ra)
def MP14_6 : ℤ := CaseSplit.mxr2 (aP14_6) 12 18
def P14_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (if c14_1 ra t && c14_4 rb t then 1 else 0)
def C14_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (if c14_1 ra t && c14_4 rb t && c14_0 s t then 1 else 0)
def M14_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C14_7 ra rb) 10
def E14_7 : List ℕ := []
def N14_7 (ra rb : ℕ) : ℤ := if E14_7.contains (ra * 23 + rb) = true then P14_7 ra rb - M14_7 ra rb else 0
def aP14_7 (ra rb : ℕ) : ℤ := -(3) * N14_7 ra rb + u14 (218 + rb) + u14 (241 + ra)
def MP14_7 : ℤ := CaseSplit.mxr2 (aP14_7) 12 22
def P14_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (if c14_1 ra t && c14_5 rb t then 1 else 0)
def C14_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (if c14_1 ra t && c14_5 rb t && c14_0 s t then 1 else 0)
def M14_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C14_8 ra rb) 10
def E14_8 : List ℕ := []
def N14_8 (ra rb : ℕ) : ℤ := if E14_8.contains (ra * 29 + rb) = true then P14_8 ra rb - M14_8 ra rb else 0
def aP14_8 (ra rb : ℕ) : ℤ := -(3) * N14_8 ra rb + u14 (254 + rb) + u14 (283 + ra)
def MP14_8 : ℤ := CaseSplit.mxr2 (aP14_8) 12 28
def N14_9 (_ra _rb : ℕ) : ℤ := 0
def aP14_9 (ra rb : ℕ) : ℤ := -(3) * N14_9 ra rb + u14 (296 + rb) + u14 (315 + ra)
def MP14_9 : ℤ := CaseSplit.mxr2 (aP14_9) 16 18
def N14_10 (_ra _rb : ℕ) : ℤ := 0
def aP14_10 (ra rb : ℕ) : ℤ := -(3) * N14_10 ra rb + u14 (332 + rb) + u14 (355 + ra)
def MP14_10 : ℤ := CaseSplit.mxr2 (aP14_10) 16 22
def N14_11 (_ra _rb : ℕ) : ℤ := 0
def aP14_11 (ra rb : ℕ) : ℤ := -(3) * N14_11 ra rb + u14 (372 + rb) + u14 (401 + ra)
def MP14_11 : ℤ := CaseSplit.mxr2 (aP14_11) 16 28
def N14_12 (_ra _rb : ℕ) : ℤ := 0
def aP14_12 (ra rb : ℕ) : ℤ := -(3) * N14_12 ra rb + u14 (418 + rb) + u14 (441 + ra)
def MP14_12 : ℤ := CaseSplit.mxr2 (aP14_12) 18 22
def N14_13 (_ra _rb : ℕ) : ℤ := 0
def aP14_13 (ra rb : ℕ) : ℤ := -(3) * N14_13 ra rb + u14 (460 + rb) + u14 (489 + ra)
def MP14_13 : ℤ := CaseSplit.mxr2 (aP14_13) 18 28
def N14_14 (_ra _rb : ℕ) : ℤ := 0
def aP14_14 (ra rb : ℕ) : ℤ := -(3) * N14_14 ra rb + u14 (508 + rb) + u14 (537 + ra)
def MP14_14 : ℤ := CaseSplit.mxr2 (aP14_14) 22 28

def rhs14 : ℤ := (∑ t ∈ Finset.range n14, w14 t) + 3 * (n14 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn14 : ∀ t, t < n14 → (0 : ℤ) ≤ w14 t := by decide
theorem plt14 : ∀ t, t < n14 → q14 t < 49 := by decide
theorem pfree14_5 : ∀ t, t < n14 → gb5 2 (q14 t) = false := by decide
theorem pfree14_7 : ∀ t, t < n14 → gb7 0 (q14 t) = false := by decide
theorem MSv14_0 : MS14_0 = 23 := by decide +kernel
theorem MSv14_1 : MS14_1 = 70 := by decide +kernel
theorem MSv14_2 : MS14_2 = 2 := by decide +kernel
theorem MSv14_3 : MS14_3 = 1 := by decide +kernel
theorem MSv14_4 : MS14_4 = 1 := by decide +kernel
theorem MSv14_5 : MS14_5 = 1 := by decide +kernel
theorem MPv14_0 : MP14_0 = 0 := by decide +kernel
theorem MPv14_1 : MP14_1 = 0 := by decide +kernel
theorem MPv14_2 : MP14_2 = 0 := by decide +kernel
theorem MPv14_3 : MP14_3 = 0 := by decide +kernel
theorem MPv14_4 : MP14_4 = 0 := by decide +kernel
theorem MPv14_5 : MP14_5 = 0 := by decide +kernel
theorem MPv14_6 : MP14_6 = 0 := by decide +kernel
theorem MPv14_7 : MP14_7 = 0 := by decide +kernel
theorem MPv14_8 : MP14_8 = 0 := by decide +kernel
theorem MPv14_9 : MP14_9 = 0 := by decide +kernel
theorem MPv14_10 : MP14_10 = 0 := by decide +kernel
theorem MPv14_11 : MP14_11 = 0 := by decide +kernel
theorem MPv14_12 : MP14_12 = 0 := by decide +kernel
theorem MPv14_13 : MP14_13 = 0 := by decide +kernel
theorem MPv14_14 : MP14_14 = 19 := by decide +kernel
theorem rhsv14 : rhs14 = 120 := by decide +kernel

/-- **The case-14 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 3/8.
    (Scaled by the common denominator 8: 117 < 120.) -/
theorem cert14 : MS14_0 + MS14_1 + MS14_2 + MS14_3 + MS14_4 + MS14_5 + MP14_0 + MP14_1 + MP14_2 + MP14_3 + MP14_4 + MP14_5 + MP14_6 + MP14_7 + MP14_8 + MP14_9 + MP14_10 + MP14_11 + MP14_12 + MP14_13 + MP14_14 < rhs14 := by
  rw [MSv14_0, MSv14_1, MSv14_2, MSv14_3, MSv14_4, MSv14_5, MPv14_0, MPv14_1, MPv14_2, MPv14_3, MPv14_4, MPv14_5, MPv14_6, MPv14_7, MPv14_8, MPv14_9, MPv14_10, MPv14_11, MPv14_12, MPv14_13, MPv14_14, rhsv14]
  decide

def Dg14 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := (if c14_0 r0 t then 1 else 0) + (if c14_1 r1 t then 1 else 0) + (if c14_2 r2 t then 1 else 0) + (if c14_3 r3 t then 1 else 0) + (if c14_4 r4 t then 1 else 0) + (if c14_5 r5 t then 1 else 0)
def Wl14_0 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c14_0 r0 t && c14_1 r1 t then 1 else 0
def Wl14_1 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c14_0 r0 t && c14_2 r2 t then 1 else 0
def Wl14_2 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c14_0 r0 t && c14_3 r3 t then 1 else 0
def Wl14_3 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c14_0 r0 t && c14_4 r4 t then 1 else 0
def Wl14_4 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c14_0 r0 t && c14_5 r5 t then 1 else 0
def Wl14_5 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c14_0 r0 t && c14_1 r1 t && c14_2 r2 t then 1 else 0
def Wl14_6 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c14_0 r0 t && c14_1 r1 t && c14_3 r3 t then 1 else 0
def Wl14_7 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c14_0 r0 t && c14_1 r1 t && c14_4 r4 t then 1 else 0
def Wl14_8 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c14_0 r0 t && c14_1 r1 t && c14_5 r5 t then 1 else 0
def Wl14_9 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c14_0 r0 t && !c14_1 r1 t && c14_2 r2 t && c14_3 r3 t then 1 else 0
def Wl14_10 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c14_0 r0 t && !c14_1 r1 t && c14_2 r2 t && c14_4 r4 t then 1 else 0
def Wl14_11 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c14_0 r0 t && !c14_1 r1 t && c14_2 r2 t && c14_5 r5 t then 1 else 0
def Wl14_12 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c14_0 r0 t && !c14_1 r1 t && !c14_2 r2 t && c14_3 r3 t && c14_4 r4 t then 1 else 0
def Wl14_13 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c14_0 r0 t && !c14_1 r1 t && !c14_2 r2 t && c14_3 r3 t && c14_5 r5 t then 1 else 0
def Wl14_14 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c14_0 r0 t && !c14_1 r1 t && !c14_2 r2 t && !c14_3 r3 t && c14_4 r4 t && c14_5 r5 t then 1 else 0

/-- **No configuration blocks the whole window in case 14.** -/
theorem nocov14 {r0 r1 r2 r3 r4 r5 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29)
    (hcov : ∀ t, t < n14 → (c14_0 r0 t || c14_1 r1 t || c14_2 r2 t || c14_3 r3 t || c14_4 r4 t || c14_5 r5 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n14, (1 : ℤ) + (Wl14_0 r0 r1 r2 r3 r4 r5 t + Wl14_1 r0 r1 r2 r3 r4 r5 t + Wl14_2 r0 r1 r2 r3 r4 r5 t + Wl14_3 r0 r1 r2 r3 r4 r5 t + Wl14_4 r0 r1 r2 r3 r4 r5 t + Wl14_5 r0 r1 r2 r3 r4 r5 t + Wl14_6 r0 r1 r2 r3 r4 r5 t + Wl14_7 r0 r1 r2 r3 r4 r5 t + Wl14_8 r0 r1 r2 r3 r4 r5 t + Wl14_9 r0 r1 r2 r3 r4 r5 t + Wl14_10 r0 r1 r2 r3 r4 r5 t + Wl14_11 r0 r1 r2 r3 r4 r5 t + Wl14_12 r0 r1 r2 r3 r4 r5 t + Wl14_13 r0 r1 r2 r3 r4 r5 t + Wl14_14 r0 r1 r2 r3 r4 r5 t) ≤ Dg14 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Wl14_0, Wl14_1, Wl14_2, Wl14_3, Wl14_4, Wl14_5, Wl14_6, Wl14_7, Wl14_8, Wl14_9, Wl14_10, Wl14_11, Wl14_12, Wl14_13, Wl14_14, Dg14]
    exact CaseSplit.lowest6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n14, (1 : ℤ) ≤ Dg14 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Dg14]
    exact CaseSplit.degpos6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n14 : ℤ) + ((∑ t ∈ Finset.range n14, Wl14_0 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n14, Wl14_1 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n14, Wl14_2 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n14, Wl14_3 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n14, Wl14_4 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n14, Wl14_5 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n14, Wl14_6 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n14, Wl14_7 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n14, Wl14_8 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n14, Wl14_9 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n14, Wl14_10 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n14, Wl14_11 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n14, Wl14_12 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n14, Wl14_13 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n14, Wl14_14 r0 r1 r2 r3 r4 r5 t)) ≤ ∑ t ∈ Finset.range n14, Dg14 r0 r1 r2 r3 r4 r5 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N14_0 r0 r1 ≤ ∑ t ∈ Finset.range n14, Wl14_0 r0 r1 r2 r3 r4 r5 t := by
    simp only [N14_0, Wl14_0, le_refl]
  have hn1 : N14_1 r0 r2 ≤ ∑ t ∈ Finset.range n14, Wl14_1 r0 r1 r2 r3 r4 r5 t := by
    simp only [N14_1, Wl14_1, le_refl]
  have hn2 : N14_2 r0 r3 ≤ ∑ t ∈ Finset.range n14, Wl14_2 r0 r1 r2 r3 r4 r5 t := by
    simp only [N14_2, Wl14_2, le_refl]
  have hn3 : N14_3 r0 r4 ≤ ∑ t ∈ Finset.range n14, Wl14_3 r0 r1 r2 r3 r4 r5 t := by
    simp only [N14_3, Wl14_3, le_refl]
  have hn4 : N14_4 r0 r5 ≤ ∑ t ∈ Finset.range n14, Wl14_4 r0 r1 r2 r3 r4 r5 t := by
    simp only [N14_4, Wl14_4, le_refl]
  have hn5 : N14_5 r1 r2 ≤ ∑ t ∈ Finset.range n14, Wl14_5 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n14, Wl14_5 r0 r1 r2 r3 r4 r5 t
        = (if c14_1 r1 t && c14_2 r2 t then (1:ℤ) else 0)
          - (if c14_1 r1 t && c14_2 r2 t && c14_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl14_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n14, Wl14_5 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl14_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n14, Wl14_5 r0 r1 r2 r3 r4 r5 t
        = P14_5 r1 r2 - C14_5 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P14_5, C14_5]
    have hm : C14_5 r1 r2 r0 ≤ M14_5 r1 r2 :=
      CaseSplit.le_mxr (C14_5 r1 r2) 10 r0 (by omega)
    simp only [N14_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N14_6 r1 r3 ≤ ∑ t ∈ Finset.range n14, Wl14_6 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n14, Wl14_6 r0 r1 r2 r3 r4 r5 t
        = (if c14_1 r1 t && c14_3 r3 t then (1:ℤ) else 0)
          - (if c14_1 r1 t && c14_3 r3 t && c14_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl14_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n14, Wl14_6 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl14_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n14, Wl14_6 r0 r1 r2 r3 r4 r5 t
        = P14_6 r1 r3 - C14_6 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P14_6, C14_6]
    have hm : C14_6 r1 r3 r0 ≤ M14_6 r1 r3 :=
      CaseSplit.le_mxr (C14_6 r1 r3) 10 r0 (by omega)
    simp only [N14_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N14_7 r1 r4 ≤ ∑ t ∈ Finset.range n14, Wl14_7 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n14, Wl14_7 r0 r1 r2 r3 r4 r5 t
        = (if c14_1 r1 t && c14_4 r4 t then (1:ℤ) else 0)
          - (if c14_1 r1 t && c14_4 r4 t && c14_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl14_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n14, Wl14_7 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl14_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n14, Wl14_7 r0 r1 r2 r3 r4 r5 t
        = P14_7 r1 r4 - C14_7 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P14_7, C14_7]
    have hm : C14_7 r1 r4 r0 ≤ M14_7 r1 r4 :=
      CaseSplit.le_mxr (C14_7 r1 r4) 10 r0 (by omega)
    simp only [N14_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N14_8 r1 r5 ≤ ∑ t ∈ Finset.range n14, Wl14_8 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n14, Wl14_8 r0 r1 r2 r3 r4 r5 t
        = (if c14_1 r1 t && c14_5 r5 t then (1:ℤ) else 0)
          - (if c14_1 r1 t && c14_5 r5 t && c14_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl14_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n14, Wl14_8 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl14_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n14, Wl14_8 r0 r1 r2 r3 r4 r5 t
        = P14_8 r1 r5 - C14_8 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P14_8, C14_8]
    have hm : C14_8 r1 r5 r0 ≤ M14_8 r1 r5 :=
      CaseSplit.le_mxr (C14_8 r1 r5) 10 r0 (by omega)
    simp only [N14_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N14_9 r2 r3 ≤ ∑ t ∈ Finset.range n14, Wl14_9 r0 r1 r2 r3 r4 r5 t := by
    simp only [N14_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl14_9]
    exact CaseSplit.ind_nonneg _
  have hn10 : N14_10 r2 r4 ≤ ∑ t ∈ Finset.range n14, Wl14_10 r0 r1 r2 r3 r4 r5 t := by
    simp only [N14_10]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl14_10]
    exact CaseSplit.ind_nonneg _
  have hn11 : N14_11 r2 r5 ≤ ∑ t ∈ Finset.range n14, Wl14_11 r0 r1 r2 r3 r4 r5 t := by
    simp only [N14_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl14_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N14_12 r3 r4 ≤ ∑ t ∈ Finset.range n14, Wl14_12 r0 r1 r2 r3 r4 r5 t := by
    simp only [N14_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl14_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N14_13 r3 r5 ≤ ∑ t ∈ Finset.range n14, Wl14_13 r0 r1 r2 r3 r4 r5 t := by
    simp only [N14_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl14_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N14_14 r4 r5 ≤ ∑ t ∈ Finset.range n14, Wl14_14 r0 r1 r2 r3 r4 r5 t := by
    simp only [N14_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl14_14]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n14, (w14 t + 3) * Dg14 r0 r1 r2 r3 r4 r5 t = S14_0 r0 + S14_1 r1 + S14_2 r2 + S14_3 r3 + S14_4 r4 + S14_5 r5 := by
    simp only [S14_0, S14_1, S14_2, S14_3, S14_4, S14_5, Dg14, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n14, (w14 t + 3) * Dg14 r0 r1 r2 r3 r4 r5 t
      = (∑ t ∈ Finset.range n14, w14 t * Dg14 r0 r1 r2 r3 r4 r5 t)
        + 3 * (∑ t ∈ Finset.range n14, Dg14 r0 r1 r2 r3 r4 r5 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n14, w14 t)
      ≤ ∑ t ∈ Finset.range n14, w14 t * Dg14 r0 r1 r2 r3 r4 r5 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg14 r0 r1 r2 r3 r4 r5 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w14 t := wnn14 t (Finset.mem_range.mp ht)
    calc w14 t = w14 t * 1 := (mul_one _).symm
      _ ≤ w14 t * Dg14 r0 r1 r2 r3 r4 r5 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS14_0 r0 + aS14_1 r1 + aS14_2 r2 + aS14_3 r3 + aS14_4 r4 + aS14_5 r5) + (aP14_0 r0 r1 + aP14_1 r0 r2 + aP14_2 r0 r3 + aP14_3 r0 r4 + aP14_4 r0 r5 + aP14_5 r1 r2 + aP14_6 r1 r3 + aP14_7 r1 r4 + aP14_8 r1 r5 + aP14_9 r2 r3 + aP14_10 r2 r4 + aP14_11 r2 r5 + aP14_12 r3 r4 + aP14_13 r3 r5 + aP14_14 r4 r5) = (S14_0 r0 + S14_1 r1 + S14_2 r2 + S14_3 r3 + S14_4 r4 + S14_5 r5) - 3 * (N14_0 r0 r1 + N14_1 r0 r2 + N14_2 r0 r3 + N14_3 r0 r4 + N14_4 r0 r5 + N14_5 r1 r2 + N14_6 r1 r3 + N14_7 r1 r4 + N14_8 r1 r5 + N14_9 r2 r3 + N14_10 r2 r4 + N14_11 r2 r5 + N14_12 r3 r4 + N14_13 r3 r5 + N14_14 r4 r5) := by
    simp only [aS14_0, aS14_1, aS14_2, aS14_3, aS14_4, aS14_5, aP14_0, aP14_1, aP14_2, aP14_3, aP14_4, aP14_5, aP14_6, aP14_7, aP14_8, aP14_9, aP14_10, aP14_11, aP14_12, aP14_13, aP14_14, L14_0, L14_1, L14_2, L14_3, L14_4, L14_5]
    ring
  have bS0 : aS14_0 r0 ≤ MS14_0 := CaseSplit.le_mxr (aS14_0) 10 r0 (by omega)
  have bS1 : aS14_1 r1 ≤ MS14_1 := CaseSplit.le_mxr (aS14_1) 12 r1 (by omega)
  have bS2 : aS14_2 r2 ≤ MS14_2 := CaseSplit.le_mxr (aS14_2) 16 r2 (by omega)
  have bS3 : aS14_3 r3 ≤ MS14_3 := CaseSplit.le_mxr (aS14_3) 18 r3 (by omega)
  have bS4 : aS14_4 r4 ≤ MS14_4 := CaseSplit.le_mxr (aS14_4) 22 r4 (by omega)
  have bS5 : aS14_5 r5 ≤ MS14_5 := CaseSplit.le_mxr (aS14_5) 28 r5 (by omega)
  have bP0 : aP14_0 r0 r1 ≤ MP14_0 := CaseSplit.le_mxr2 (aP14_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP14_1 r0 r2 ≤ MP14_1 := CaseSplit.le_mxr2 (aP14_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP14_2 r0 r3 ≤ MP14_2 := CaseSplit.le_mxr2 (aP14_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP14_3 r0 r4 ≤ MP14_3 := CaseSplit.le_mxr2 (aP14_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP14_4 r0 r5 ≤ MP14_4 := CaseSplit.le_mxr2 (aP14_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP14_5 r1 r2 ≤ MP14_5 := CaseSplit.le_mxr2 (aP14_5) 12 16 r1 r2 (by omega) (by omega)
  have bP6 : aP14_6 r1 r3 ≤ MP14_6 := CaseSplit.le_mxr2 (aP14_6) 12 18 r1 r3 (by omega) (by omega)
  have bP7 : aP14_7 r1 r4 ≤ MP14_7 := CaseSplit.le_mxr2 (aP14_7) 12 22 r1 r4 (by omega) (by omega)
  have bP8 : aP14_8 r1 r5 ≤ MP14_8 := CaseSplit.le_mxr2 (aP14_8) 12 28 r1 r5 (by omega) (by omega)
  have bP9 : aP14_9 r2 r3 ≤ MP14_9 := CaseSplit.le_mxr2 (aP14_9) 16 18 r2 r3 (by omega) (by omega)
  have bP10 : aP14_10 r2 r4 ≤ MP14_10 := CaseSplit.le_mxr2 (aP14_10) 16 22 r2 r4 (by omega) (by omega)
  have bP11 : aP14_11 r2 r5 ≤ MP14_11 := CaseSplit.le_mxr2 (aP14_11) 16 28 r2 r5 (by omega) (by omega)
  have bP12 : aP14_12 r3 r4 ≤ MP14_12 := CaseSplit.le_mxr2 (aP14_12) 18 22 r3 r4 (by omega) (by omega)
  have bP13 : aP14_13 r3 r5 ≤ MP14_13 := CaseSplit.le_mxr2 (aP14_13) 18 28 r3 r5 (by omega) (by omega)
  have bP14 : aP14_14 r4 r5 ≤ MP14_14 := CaseSplit.le_mxr2 (aP14_14) 22 28 r4 r5 (by omega) (by omega)
  have hrhs : rhs14 = (∑ t ∈ Finset.range n14, w14 t) + 3 * (n14 : ℤ) := rfl
  have hc := cert14
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, bS0, bS1, bS2, bS3, bS4, bS5, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14]

end IncCert29
