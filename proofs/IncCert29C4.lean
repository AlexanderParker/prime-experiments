/-
INCREMENT-WIDTH CERTIFICATE, step 23->29, case 4 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_23_29.json, which re-derives every number
from the primes alone).

Machine 29, INCREMENT width 49 = F_2(23) + s_min(29) = 39 + 10,
held gears [5, 7] at phases [0, 4].  Free gears [11, 13, 17, 19, 23, 29].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 3.
-/
import IncCert29B

namespace IncCert29

/-! ### case 4: held gears at phases [0, 4] -/

def p4 : List ℕ := [0, 3, 5, 7, 8, 10, 12, 13, 15, 17, 20, 22, 27, 28, 33, 35, 38, 40, 42, 43, 45, 47, 48]
def q4 (t : ℕ) : ℕ := p4.getD t 0
def n4 : ℕ := 23
def yl4 : List ℤ := [0, 0, 0, 1, 1, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0]
def w4 (t : ℕ) : ℤ := yl4.getD t 0
def ul4 : List ℤ := [(-7), (-2), 0, (-2), 0, (-2), (-3), (-2), 0, (-4), (-2), 0, (-2), 2, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, (-2), (-2), (-2), 0, 0, (-2), (-2), (-2), (-2), 0, (-2), (-3), (-2), (-2), (-2), (-1), (-2), 0, 0, 2, 1, 0, 0, 0, 0, 2, 2, 0, 0, 0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), 0, 0, 0, (-1), (-1), (-1), 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, (-1), (-1), (-1), 0, (-1), (-1), 0, (-1), (-1), (-1), (-1), 0, (-1), 0, 0, (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, (-1), (-1), (-1), 0, (-1), 0, 0, (-1), (-1), 0, 12, 8, 12, 12, 11, 10, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 10, (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), 12, 9, 7, 12, 4, 4, 12, 12, 11, 11, 4, 7, 12, 12, 12, 11, 7, 7, 12, (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-15), (-12), 12, 12, 12, 12, 12, 8, 4, 12, 12, 9, 12, 12, 12, 1, 9, 9, 4, 12, 4, 8, 12, 1, 12, (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-4), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 6, 0, 3, 6, 6, 6, 4, 5, 3, 6, 6, 5, 8, 6, 8, 7, 8, 8, 6, 8, 3, 5, 8, 6, 7, 4, 3, (-6), (-8), 0, (-8), 0, 0, 0, 0, (-8), 0, (-3), (-7), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def u4 (k : ℕ) : ℤ := ul4.getD k 0

def c4_0 (r t : ℕ) : Bool := gb11 r (q4 t)
def c4_1 (r t : ℕ) : Bool := gb13 r (q4 t)
def c4_2 (r t : ℕ) : Bool := gb17 r (q4 t)
def c4_3 (r t : ℕ) : Bool := gb19 r (q4 t)
def c4_4 (r t : ℕ) : Bool := gb23 r (q4 t)
def c4_5 (r t : ℕ) : Bool := gb29 r (q4 t)

def S4_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (w4 t + 3) * (if c4_0 r t then 1 else 0)
def S4_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (w4 t + 3) * (if c4_1 r t then 1 else 0)
def S4_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (w4 t + 3) * (if c4_2 r t then 1 else 0)
def S4_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (w4 t + 3) * (if c4_3 r t then 1 else 0)
def S4_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (w4 t + 3) * (if c4_4 r t then 1 else 0)
def S4_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (w4 t + 3) * (if c4_5 r t then 1 else 0)

def L4_0 (r : ℕ) : ℤ := u4 (13 + r) + u4 (41 + r) + u4 (71 + r) + u4 (105 + r) + u4 (145 + r)
def L4_1 (r : ℕ) : ℤ := u4 (0 + r) + u4 (173 + r) + u4 (205 + r) + u4 (241 + r) + u4 (283 + r)
def L4_2 (r : ℕ) : ℤ := u4 (24 + r) + u4 (156 + r) + u4 (315 + r) + u4 (355 + r) + u4 (401 + r)
def L4_3 (r : ℕ) : ℤ := u4 (52 + r) + u4 (186 + r) + u4 (296 + r) + u4 (441 + r) + u4 (489 + r)
def L4_4 (r : ℕ) : ℤ := u4 (82 + r) + u4 (218 + r) + u4 (332 + r) + u4 (418 + r) + u4 (537 + r)
def L4_5 (r : ℕ) : ℤ := u4 (116 + r) + u4 (254 + r) + u4 (372 + r) + u4 (460 + r) + u4 (508 + r)

def aS4_0 (r : ℕ) : ℤ := S4_0 r - L4_0 r
def MS4_0 : ℤ := CaseSplit.mxr (aS4_0) 10
def aS4_1 (r : ℕ) : ℤ := S4_1 r - L4_1 r
def MS4_1 : ℤ := CaseSplit.mxr (aS4_1) 12
def aS4_2 (r : ℕ) : ℤ := S4_2 r - L4_2 r
def MS4_2 : ℤ := CaseSplit.mxr (aS4_2) 16
def aS4_3 (r : ℕ) : ℤ := S4_3 r - L4_3 r
def MS4_3 : ℤ := CaseSplit.mxr (aS4_3) 18
def aS4_4 (r : ℕ) : ℤ := S4_4 r - L4_4 r
def MS4_4 : ℤ := CaseSplit.mxr (aS4_4) 22
def aS4_5 (r : ℕ) : ℤ := S4_5 r - L4_5 r
def MS4_5 : ℤ := CaseSplit.mxr (aS4_5) 28

def N4_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_0 ra t && c4_1 rb t then 1 else 0)
def aP4_0 (ra rb : ℕ) : ℤ := -(3) * N4_0 ra rb + u4 (0 + rb) + u4 (13 + ra)
def MP4_0 : ℤ := CaseSplit.mxr2 (aP4_0) 10 12
def N4_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_0 ra t && c4_2 rb t then 1 else 0)
def aP4_1 (ra rb : ℕ) : ℤ := -(3) * N4_1 ra rb + u4 (24 + rb) + u4 (41 + ra)
def MP4_1 : ℤ := CaseSplit.mxr2 (aP4_1) 10 16
def N4_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_0 ra t && c4_3 rb t then 1 else 0)
def aP4_2 (ra rb : ℕ) : ℤ := -(3) * N4_2 ra rb + u4 (52 + rb) + u4 (71 + ra)
def MP4_2 : ℤ := CaseSplit.mxr2 (aP4_2) 10 18
def N4_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_0 ra t && c4_4 rb t then 1 else 0)
def aP4_3 (ra rb : ℕ) : ℤ := -(3) * N4_3 ra rb + u4 (82 + rb) + u4 (105 + ra)
def MP4_3 : ℤ := CaseSplit.mxr2 (aP4_3) 10 22
def N4_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_0 ra t && c4_5 rb t then 1 else 0)
def aP4_4 (ra rb : ℕ) : ℤ := -(3) * N4_4 ra rb + u4 (116 + rb) + u4 (145 + ra)
def MP4_4 : ℤ := CaseSplit.mxr2 (aP4_4) 10 28
def P4_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_1 ra t && c4_2 rb t then 1 else 0)
def C4_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_1 ra t && c4_2 rb t && c4_0 s t then 1 else 0)
def M4_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C4_5 ra rb) 10
def E4_5 : List ℕ := [21, 27, 111, 117, 136, 147, 156, 162, 190, 201]
def N4_5 (ra rb : ℕ) : ℤ := if E4_5.contains (ra * 17 + rb) = true then P4_5 ra rb - M4_5 ra rb else 0
def aP4_5 (ra rb : ℕ) : ℤ := -(3) * N4_5 ra rb + u4 (156 + rb) + u4 (173 + ra)
def MP4_5 : ℤ := CaseSplit.mxr2 (aP4_5) 12 16
def P4_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_1 ra t && c4_3 rb t then 1 else 0)
def C4_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_1 ra t && c4_3 rb t && c4_0 s t then 1 else 0)
def M4_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C4_6 ra rb) 10
def E4_6 : List ℕ := [7, 33, 38, 41, 44, 78, 91, 114, 120, 154, 167, 178, 204, 212]
def N4_6 (ra rb : ℕ) : ℤ := if E4_6.contains (ra * 19 + rb) = true then P4_6 ra rb - M4_6 ra rb else 0
def aP4_6 (ra rb : ℕ) : ℤ := -(3) * N4_6 ra rb + u4 (186 + rb) + u4 (205 + ra)
def MP4_6 : ℤ := CaseSplit.mxr2 (aP4_6) 12 18
def P4_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_1 ra t && c4_4 rb t then 1 else 0)
def C4_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_1 ra t && c4_4 rb t && c4_0 s t then 1 else 0)
def M4_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C4_7 ra rb) 10
def E4_7 : List ℕ := []
def N4_7 (ra rb : ℕ) : ℤ := if E4_7.contains (ra * 23 + rb) = true then P4_7 ra rb - M4_7 ra rb else 0
def aP4_7 (ra rb : ℕ) : ℤ := -(3) * N4_7 ra rb + u4 (218 + rb) + u4 (241 + ra)
def MP4_7 : ℤ := CaseSplit.mxr2 (aP4_7) 12 22
def P4_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_1 ra t && c4_5 rb t then 1 else 0)
def C4_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_1 ra t && c4_5 rb t && c4_0 s t then 1 else 0)
def M4_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C4_8 ra rb) 10
def E4_8 : List ℕ := [103, 219, 253, 369]
def N4_8 (ra rb : ℕ) : ℤ := if E4_8.contains (ra * 29 + rb) = true then P4_8 ra rb - M4_8 ra rb else 0
def aP4_8 (ra rb : ℕ) : ℤ := -(3) * N4_8 ra rb + u4 (254 + rb) + u4 (283 + ra)
def MP4_8 : ℤ := CaseSplit.mxr2 (aP4_8) 12 28
def N4_9 (_ra _rb : ℕ) : ℤ := 0
def aP4_9 (ra rb : ℕ) : ℤ := -(3) * N4_9 ra rb + u4 (296 + rb) + u4 (315 + ra)
def MP4_9 : ℤ := CaseSplit.mxr2 (aP4_9) 16 18
def N4_10 (_ra _rb : ℕ) : ℤ := 0
def aP4_10 (ra rb : ℕ) : ℤ := -(3) * N4_10 ra rb + u4 (332 + rb) + u4 (355 + ra)
def MP4_10 : ℤ := CaseSplit.mxr2 (aP4_10) 16 22
def N4_11 (_ra _rb : ℕ) : ℤ := 0
def aP4_11 (ra rb : ℕ) : ℤ := -(3) * N4_11 ra rb + u4 (372 + rb) + u4 (401 + ra)
def MP4_11 : ℤ := CaseSplit.mxr2 (aP4_11) 16 28
def N4_12 (_ra _rb : ℕ) : ℤ := 0
def aP4_12 (ra rb : ℕ) : ℤ := -(3) * N4_12 ra rb + u4 (418 + rb) + u4 (441 + ra)
def MP4_12 : ℤ := CaseSplit.mxr2 (aP4_12) 18 22
def N4_13 (_ra _rb : ℕ) : ℤ := 0
def aP4_13 (ra rb : ℕ) : ℤ := -(3) * N4_13 ra rb + u4 (460 + rb) + u4 (489 + ra)
def MP4_13 : ℤ := CaseSplit.mxr2 (aP4_13) 18 28
def N4_14 (_ra _rb : ℕ) : ℤ := 0
def aP4_14 (ra rb : ℕ) : ℤ := -(3) * N4_14 ra rb + u4 (508 + rb) + u4 (537 + ra)
def MP4_14 : ℤ := CaseSplit.mxr2 (aP4_14) 22 28

def rhs4 : ℤ := (∑ t ∈ Finset.range n4, w4 t) + 3 * (n4 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn4 : ∀ t, t < n4 → (0 : ℤ) ≤ w4 t := by decide
theorem plt4 : ∀ t, t < n4 → q4 t < 49 := by decide
theorem pfree4_5 : ∀ t, t < n4 → gb5 0 (q4 t) = false := by decide
theorem pfree4_7 : ∀ t, t < n4 → gb7 4 (q4 t) = false := by decide
theorem MSv4_0 : MS4_0 = 14 := by decide +kernel
theorem MSv4_1 : MS4_1 = 52 := by decide +kernel
theorem MSv4_2 : MS4_2 = 1 := by decide +kernel
theorem MSv4_3 : MS4_3 = 1 := by decide +kernel
theorem MSv4_4 : MS4_4 = 0 := by decide +kernel
theorem MSv4_5 : MS4_5 = 0 := by decide +kernel
theorem MPv4_0 : MP4_0 = 0 := by decide +kernel
theorem MPv4_1 : MP4_1 = 0 := by decide +kernel
theorem MPv4_2 : MP4_2 = 0 := by decide +kernel
theorem MPv4_3 : MP4_3 = 0 := by decide +kernel
theorem MPv4_4 : MP4_4 = 0 := by decide +kernel
theorem MPv4_5 : MP4_5 = 0 := by decide +kernel
theorem MPv4_6 : MP4_6 = 0 := by decide +kernel
theorem MPv4_7 : MP4_7 = 0 := by decide +kernel
theorem MPv4_8 : MP4_8 = 0 := by decide +kernel
theorem MPv4_9 : MP4_9 = 0 := by decide +kernel
theorem MPv4_10 : MP4_10 = 0 := by decide +kernel
theorem MPv4_11 : MP4_11 = 0 := by decide +kernel
theorem MPv4_12 : MP4_12 = 0 := by decide +kernel
theorem MPv4_13 : MP4_13 = 0 := by decide +kernel
theorem MPv4_14 : MP4_14 = 8 := by decide +kernel
theorem rhsv4 : rhs4 = 77 := by decide +kernel

/-- **The case-4 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/3.
    (Scaled by the common denominator 3: 76 < 77.) -/
theorem cert4 : MS4_0 + MS4_1 + MS4_2 + MS4_3 + MS4_4 + MS4_5 + MP4_0 + MP4_1 + MP4_2 + MP4_3 + MP4_4 + MP4_5 + MP4_6 + MP4_7 + MP4_8 + MP4_9 + MP4_10 + MP4_11 + MP4_12 + MP4_13 + MP4_14 < rhs4 := by
  rw [MSv4_0, MSv4_1, MSv4_2, MSv4_3, MSv4_4, MSv4_5, MPv4_0, MPv4_1, MPv4_2, MPv4_3, MPv4_4, MPv4_5, MPv4_6, MPv4_7, MPv4_8, MPv4_9, MPv4_10, MPv4_11, MPv4_12, MPv4_13, MPv4_14, rhsv4]
  decide

def Dg4 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := (if c4_0 r0 t then 1 else 0) + (if c4_1 r1 t then 1 else 0) + (if c4_2 r2 t then 1 else 0) + (if c4_3 r3 t then 1 else 0) + (if c4_4 r4 t then 1 else 0) + (if c4_5 r5 t then 1 else 0)
def Wl4_0 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c4_0 r0 t && c4_1 r1 t then 1 else 0
def Wl4_1 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c4_0 r0 t && c4_2 r2 t then 1 else 0
def Wl4_2 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c4_0 r0 t && c4_3 r3 t then 1 else 0
def Wl4_3 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c4_0 r0 t && c4_4 r4 t then 1 else 0
def Wl4_4 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c4_0 r0 t && c4_5 r5 t then 1 else 0
def Wl4_5 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c4_0 r0 t && c4_1 r1 t && c4_2 r2 t then 1 else 0
def Wl4_6 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c4_0 r0 t && c4_1 r1 t && c4_3 r3 t then 1 else 0
def Wl4_7 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c4_0 r0 t && c4_1 r1 t && c4_4 r4 t then 1 else 0
def Wl4_8 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c4_0 r0 t && c4_1 r1 t && c4_5 r5 t then 1 else 0
def Wl4_9 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c4_0 r0 t && !c4_1 r1 t && c4_2 r2 t && c4_3 r3 t then 1 else 0
def Wl4_10 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c4_0 r0 t && !c4_1 r1 t && c4_2 r2 t && c4_4 r4 t then 1 else 0
def Wl4_11 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c4_0 r0 t && !c4_1 r1 t && c4_2 r2 t && c4_5 r5 t then 1 else 0
def Wl4_12 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c4_0 r0 t && !c4_1 r1 t && !c4_2 r2 t && c4_3 r3 t && c4_4 r4 t then 1 else 0
def Wl4_13 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c4_0 r0 t && !c4_1 r1 t && !c4_2 r2 t && c4_3 r3 t && c4_5 r5 t then 1 else 0
def Wl4_14 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c4_0 r0 t && !c4_1 r1 t && !c4_2 r2 t && !c4_3 r3 t && c4_4 r4 t && c4_5 r5 t then 1 else 0

/-- **No configuration blocks the whole window in case 4.** -/
theorem nocov4 {r0 r1 r2 r3 r4 r5 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29)
    (hcov : ∀ t, t < n4 → (c4_0 r0 t || c4_1 r1 t || c4_2 r2 t || c4_3 r3 t || c4_4 r4 t || c4_5 r5 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n4, (1 : ℤ) + (Wl4_0 r0 r1 r2 r3 r4 r5 t + Wl4_1 r0 r1 r2 r3 r4 r5 t + Wl4_2 r0 r1 r2 r3 r4 r5 t + Wl4_3 r0 r1 r2 r3 r4 r5 t + Wl4_4 r0 r1 r2 r3 r4 r5 t + Wl4_5 r0 r1 r2 r3 r4 r5 t + Wl4_6 r0 r1 r2 r3 r4 r5 t + Wl4_7 r0 r1 r2 r3 r4 r5 t + Wl4_8 r0 r1 r2 r3 r4 r5 t + Wl4_9 r0 r1 r2 r3 r4 r5 t + Wl4_10 r0 r1 r2 r3 r4 r5 t + Wl4_11 r0 r1 r2 r3 r4 r5 t + Wl4_12 r0 r1 r2 r3 r4 r5 t + Wl4_13 r0 r1 r2 r3 r4 r5 t + Wl4_14 r0 r1 r2 r3 r4 r5 t) ≤ Dg4 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Wl4_0, Wl4_1, Wl4_2, Wl4_3, Wl4_4, Wl4_5, Wl4_6, Wl4_7, Wl4_8, Wl4_9, Wl4_10, Wl4_11, Wl4_12, Wl4_13, Wl4_14, Dg4]
    exact CaseSplit.lowest6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n4, (1 : ℤ) ≤ Dg4 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Dg4]
    exact CaseSplit.degpos6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n4 : ℤ) + ((∑ t ∈ Finset.range n4, Wl4_0 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n4, Wl4_1 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n4, Wl4_2 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n4, Wl4_3 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n4, Wl4_4 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n4, Wl4_5 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n4, Wl4_6 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n4, Wl4_7 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n4, Wl4_8 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n4, Wl4_9 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n4, Wl4_10 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n4, Wl4_11 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n4, Wl4_12 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n4, Wl4_13 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n4, Wl4_14 r0 r1 r2 r3 r4 r5 t)) ≤ ∑ t ∈ Finset.range n4, Dg4 r0 r1 r2 r3 r4 r5 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N4_0 r0 r1 ≤ ∑ t ∈ Finset.range n4, Wl4_0 r0 r1 r2 r3 r4 r5 t := by
    simp only [N4_0, Wl4_0, le_refl]
  have hn1 : N4_1 r0 r2 ≤ ∑ t ∈ Finset.range n4, Wl4_1 r0 r1 r2 r3 r4 r5 t := by
    simp only [N4_1, Wl4_1, le_refl]
  have hn2 : N4_2 r0 r3 ≤ ∑ t ∈ Finset.range n4, Wl4_2 r0 r1 r2 r3 r4 r5 t := by
    simp only [N4_2, Wl4_2, le_refl]
  have hn3 : N4_3 r0 r4 ≤ ∑ t ∈ Finset.range n4, Wl4_3 r0 r1 r2 r3 r4 r5 t := by
    simp only [N4_3, Wl4_3, le_refl]
  have hn4 : N4_4 r0 r5 ≤ ∑ t ∈ Finset.range n4, Wl4_4 r0 r1 r2 r3 r4 r5 t := by
    simp only [N4_4, Wl4_4, le_refl]
  have hn5 : N4_5 r1 r2 ≤ ∑ t ∈ Finset.range n4, Wl4_5 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n4, Wl4_5 r0 r1 r2 r3 r4 r5 t
        = (if c4_1 r1 t && c4_2 r2 t then (1:ℤ) else 0)
          - (if c4_1 r1 t && c4_2 r2 t && c4_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl4_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n4, Wl4_5 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl4_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n4, Wl4_5 r0 r1 r2 r3 r4 r5 t
        = P4_5 r1 r2 - C4_5 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P4_5, C4_5]
    have hm : C4_5 r1 r2 r0 ≤ M4_5 r1 r2 :=
      CaseSplit.le_mxr (C4_5 r1 r2) 10 r0 (by omega)
    simp only [N4_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N4_6 r1 r3 ≤ ∑ t ∈ Finset.range n4, Wl4_6 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n4, Wl4_6 r0 r1 r2 r3 r4 r5 t
        = (if c4_1 r1 t && c4_3 r3 t then (1:ℤ) else 0)
          - (if c4_1 r1 t && c4_3 r3 t && c4_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl4_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n4, Wl4_6 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl4_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n4, Wl4_6 r0 r1 r2 r3 r4 r5 t
        = P4_6 r1 r3 - C4_6 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P4_6, C4_6]
    have hm : C4_6 r1 r3 r0 ≤ M4_6 r1 r3 :=
      CaseSplit.le_mxr (C4_6 r1 r3) 10 r0 (by omega)
    simp only [N4_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N4_7 r1 r4 ≤ ∑ t ∈ Finset.range n4, Wl4_7 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n4, Wl4_7 r0 r1 r2 r3 r4 r5 t
        = (if c4_1 r1 t && c4_4 r4 t then (1:ℤ) else 0)
          - (if c4_1 r1 t && c4_4 r4 t && c4_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl4_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n4, Wl4_7 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl4_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n4, Wl4_7 r0 r1 r2 r3 r4 r5 t
        = P4_7 r1 r4 - C4_7 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P4_7, C4_7]
    have hm : C4_7 r1 r4 r0 ≤ M4_7 r1 r4 :=
      CaseSplit.le_mxr (C4_7 r1 r4) 10 r0 (by omega)
    simp only [N4_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N4_8 r1 r5 ≤ ∑ t ∈ Finset.range n4, Wl4_8 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n4, Wl4_8 r0 r1 r2 r3 r4 r5 t
        = (if c4_1 r1 t && c4_5 r5 t then (1:ℤ) else 0)
          - (if c4_1 r1 t && c4_5 r5 t && c4_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl4_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n4, Wl4_8 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl4_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n4, Wl4_8 r0 r1 r2 r3 r4 r5 t
        = P4_8 r1 r5 - C4_8 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P4_8, C4_8]
    have hm : C4_8 r1 r5 r0 ≤ M4_8 r1 r5 :=
      CaseSplit.le_mxr (C4_8 r1 r5) 10 r0 (by omega)
    simp only [N4_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N4_9 r2 r3 ≤ ∑ t ∈ Finset.range n4, Wl4_9 r0 r1 r2 r3 r4 r5 t := by
    simp only [N4_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl4_9]
    exact CaseSplit.ind_nonneg _
  have hn10 : N4_10 r2 r4 ≤ ∑ t ∈ Finset.range n4, Wl4_10 r0 r1 r2 r3 r4 r5 t := by
    simp only [N4_10]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl4_10]
    exact CaseSplit.ind_nonneg _
  have hn11 : N4_11 r2 r5 ≤ ∑ t ∈ Finset.range n4, Wl4_11 r0 r1 r2 r3 r4 r5 t := by
    simp only [N4_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl4_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N4_12 r3 r4 ≤ ∑ t ∈ Finset.range n4, Wl4_12 r0 r1 r2 r3 r4 r5 t := by
    simp only [N4_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl4_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N4_13 r3 r5 ≤ ∑ t ∈ Finset.range n4, Wl4_13 r0 r1 r2 r3 r4 r5 t := by
    simp only [N4_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl4_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N4_14 r4 r5 ≤ ∑ t ∈ Finset.range n4, Wl4_14 r0 r1 r2 r3 r4 r5 t := by
    simp only [N4_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl4_14]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n4, (w4 t + 3) * Dg4 r0 r1 r2 r3 r4 r5 t = S4_0 r0 + S4_1 r1 + S4_2 r2 + S4_3 r3 + S4_4 r4 + S4_5 r5 := by
    simp only [S4_0, S4_1, S4_2, S4_3, S4_4, S4_5, Dg4, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n4, (w4 t + 3) * Dg4 r0 r1 r2 r3 r4 r5 t
      = (∑ t ∈ Finset.range n4, w4 t * Dg4 r0 r1 r2 r3 r4 r5 t)
        + 3 * (∑ t ∈ Finset.range n4, Dg4 r0 r1 r2 r3 r4 r5 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n4, w4 t)
      ≤ ∑ t ∈ Finset.range n4, w4 t * Dg4 r0 r1 r2 r3 r4 r5 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg4 r0 r1 r2 r3 r4 r5 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w4 t := wnn4 t (Finset.mem_range.mp ht)
    calc w4 t = w4 t * 1 := (mul_one _).symm
      _ ≤ w4 t * Dg4 r0 r1 r2 r3 r4 r5 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS4_0 r0 + aS4_1 r1 + aS4_2 r2 + aS4_3 r3 + aS4_4 r4 + aS4_5 r5) + (aP4_0 r0 r1 + aP4_1 r0 r2 + aP4_2 r0 r3 + aP4_3 r0 r4 + aP4_4 r0 r5 + aP4_5 r1 r2 + aP4_6 r1 r3 + aP4_7 r1 r4 + aP4_8 r1 r5 + aP4_9 r2 r3 + aP4_10 r2 r4 + aP4_11 r2 r5 + aP4_12 r3 r4 + aP4_13 r3 r5 + aP4_14 r4 r5) = (S4_0 r0 + S4_1 r1 + S4_2 r2 + S4_3 r3 + S4_4 r4 + S4_5 r5) - 3 * (N4_0 r0 r1 + N4_1 r0 r2 + N4_2 r0 r3 + N4_3 r0 r4 + N4_4 r0 r5 + N4_5 r1 r2 + N4_6 r1 r3 + N4_7 r1 r4 + N4_8 r1 r5 + N4_9 r2 r3 + N4_10 r2 r4 + N4_11 r2 r5 + N4_12 r3 r4 + N4_13 r3 r5 + N4_14 r4 r5) := by
    simp only [aS4_0, aS4_1, aS4_2, aS4_3, aS4_4, aS4_5, aP4_0, aP4_1, aP4_2, aP4_3, aP4_4, aP4_5, aP4_6, aP4_7, aP4_8, aP4_9, aP4_10, aP4_11, aP4_12, aP4_13, aP4_14, L4_0, L4_1, L4_2, L4_3, L4_4, L4_5]
    ring
  have bS0 : aS4_0 r0 ≤ MS4_0 := CaseSplit.le_mxr (aS4_0) 10 r0 (by omega)
  have bS1 : aS4_1 r1 ≤ MS4_1 := CaseSplit.le_mxr (aS4_1) 12 r1 (by omega)
  have bS2 : aS4_2 r2 ≤ MS4_2 := CaseSplit.le_mxr (aS4_2) 16 r2 (by omega)
  have bS3 : aS4_3 r3 ≤ MS4_3 := CaseSplit.le_mxr (aS4_3) 18 r3 (by omega)
  have bS4 : aS4_4 r4 ≤ MS4_4 := CaseSplit.le_mxr (aS4_4) 22 r4 (by omega)
  have bS5 : aS4_5 r5 ≤ MS4_5 := CaseSplit.le_mxr (aS4_5) 28 r5 (by omega)
  have bP0 : aP4_0 r0 r1 ≤ MP4_0 := CaseSplit.le_mxr2 (aP4_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP4_1 r0 r2 ≤ MP4_1 := CaseSplit.le_mxr2 (aP4_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP4_2 r0 r3 ≤ MP4_2 := CaseSplit.le_mxr2 (aP4_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP4_3 r0 r4 ≤ MP4_3 := CaseSplit.le_mxr2 (aP4_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP4_4 r0 r5 ≤ MP4_4 := CaseSplit.le_mxr2 (aP4_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP4_5 r1 r2 ≤ MP4_5 := CaseSplit.le_mxr2 (aP4_5) 12 16 r1 r2 (by omega) (by omega)
  have bP6 : aP4_6 r1 r3 ≤ MP4_6 := CaseSplit.le_mxr2 (aP4_6) 12 18 r1 r3 (by omega) (by omega)
  have bP7 : aP4_7 r1 r4 ≤ MP4_7 := CaseSplit.le_mxr2 (aP4_7) 12 22 r1 r4 (by omega) (by omega)
  have bP8 : aP4_8 r1 r5 ≤ MP4_8 := CaseSplit.le_mxr2 (aP4_8) 12 28 r1 r5 (by omega) (by omega)
  have bP9 : aP4_9 r2 r3 ≤ MP4_9 := CaseSplit.le_mxr2 (aP4_9) 16 18 r2 r3 (by omega) (by omega)
  have bP10 : aP4_10 r2 r4 ≤ MP4_10 := CaseSplit.le_mxr2 (aP4_10) 16 22 r2 r4 (by omega) (by omega)
  have bP11 : aP4_11 r2 r5 ≤ MP4_11 := CaseSplit.le_mxr2 (aP4_11) 16 28 r2 r5 (by omega) (by omega)
  have bP12 : aP4_12 r3 r4 ≤ MP4_12 := CaseSplit.le_mxr2 (aP4_12) 18 22 r3 r4 (by omega) (by omega)
  have bP13 : aP4_13 r3 r5 ≤ MP4_13 := CaseSplit.le_mxr2 (aP4_13) 18 28 r3 r5 (by omega) (by omega)
  have bP14 : aP4_14 r4 r5 ≤ MP4_14 := CaseSplit.le_mxr2 (aP4_14) 22 28 r4 r5 (by omega) (by omega)
  have hrhs : rhs4 = (∑ t ∈ Finset.range n4, w4 t) + 3 * (n4 : ℤ) := rfl
  have hc := cert4
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, bS0, bS1, bS2, bS3, bS4, bS5, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14]

end IncCert29
