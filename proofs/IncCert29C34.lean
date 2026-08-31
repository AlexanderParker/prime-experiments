/-
INCREMENT-WIDTH CERTIFICATE, step 23->29, case 34 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_23_29.json, which re-derives every number
from the primes alone).

Machine 29, INCREMENT width 49 = F_2(23) + s_min(29) = 39 + 10,
held gears [5, 7] at phases [4, 6].  Free gears [11, 13, 17, 19, 23, 29].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 4.
-/
import IncCert29B

namespace IncCert29

/-! ### case 34: held gears at phases [4, 6] -/

def p34 : List ℕ := [1, 3, 4, 6, 8, 11, 13, 18, 19, 24, 26, 29, 31, 33, 34, 36, 38, 39, 41, 43, 46, 48]
def q34 (t : ℕ) : ℕ := p34.getD t 0
def n34 : ℕ := 22
def yl34 : List ℤ := [1, 0, 0, 2, 0, 1, 1, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 1, 0, 0, 0]
def w34 (t : ℕ) : ℤ := yl34.getD t 0
def ul34 : List ℤ := [0, (-1), 0, (-1), 0, (-1), (-1), (-1), 0, (-1), (-1), 4, (-1), 0, (-4), (-4), 0, (-4), 0, (-4), 1, (-4), 1, 0, (-3), (-2), (-1), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), 0, (-3), (-3), (-3), 3, 3, 2, 0, 0, 1, 3, 3, 2, 0, 0, (-3), (-3), (-1), 0, (-3), (-3), (-3), (-3), (-2), (-2), (-2), (-2), (-3), (-3), (-2), 0, 0, (-2), (-3), 0, 2, 0, 0, 0, 1, 0, 3, 2, 0, 0, 0, 3, (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, (-1), 0, (-1), (-1), (-1), (-1), 3, (-1), (-1), 0, (-1), (-1), (-1), (-3), 1, 0, (-3), (-3), 1, 0, (-3), (-6), 0, 0, (-2), 0, (-2), (-2), (-2), 0, (-2), (-2), 0, (-2), (-2), (-2), (-1), (-2), (-2), 0, (-1), (-2), (-2), (-2), 0, (-2), (-1), (-2), (-2), (-2), (-1), 0, (-2), 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 17, 17, 17, 12, 8, 11, 17, 17, 17, 17, 12, 15, 17, 17, 17, 17, 12, (-17), (-17), (-17), (-17), (-17), (-17), (-17), (-17), (-17), (-17), (-17), (-17), (-17), 11, 7, 15, 13, 15, 13, 11, 7, 15, 11, 14, 15, 15, 11, 10, 13, 15, 15, 15, (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), 5, 5, 5, 5, 1, 1, 5, 5, 4, 5, 1, 5, 0, 5, 5, 0, 5, 5, 5, 5, 1, 5, 5, (-5), (-5), (-5), (-5), (-5), (-5), (-5), (-5), (-5), (-5), (-5), (-5), (-5), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 8, 6, 8, 7, 12, 6, 6, 4, 2, 10, 7, 6, 7, 6, 12, 12, 12, 8, 6, 12, 11, 12, 12, 10, 7, 5, 10, 8, 3, 9, 0, 9, 9, 0, 7, 0, 9, 8, 0, 9, 6, 6, 7, 5, 9, 4, 5, 9, 0, 6, 0]
def u34 (k : ℕ) : ℤ := ul34.getD k 0

def c34_0 (r t : ℕ) : Bool := gb11 r (q34 t)
def c34_1 (r t : ℕ) : Bool := gb13 r (q34 t)
def c34_2 (r t : ℕ) : Bool := gb17 r (q34 t)
def c34_3 (r t : ℕ) : Bool := gb19 r (q34 t)
def c34_4 (r t : ℕ) : Bool := gb23 r (q34 t)
def c34_5 (r t : ℕ) : Bool := gb29 r (q34 t)

def S34_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n34, (w34 t + 4) * (if c34_0 r t then 1 else 0)
def S34_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n34, (w34 t + 4) * (if c34_1 r t then 1 else 0)
def S34_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n34, (w34 t + 4) * (if c34_2 r t then 1 else 0)
def S34_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n34, (w34 t + 4) * (if c34_3 r t then 1 else 0)
def S34_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n34, (w34 t + 4) * (if c34_4 r t then 1 else 0)
def S34_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n34, (w34 t + 4) * (if c34_5 r t then 1 else 0)

def L34_0 (r : ℕ) : ℤ := u34 (13 + r) + u34 (41 + r) + u34 (71 + r) + u34 (105 + r) + u34 (145 + r)
def L34_1 (r : ℕ) : ℤ := u34 (0 + r) + u34 (173 + r) + u34 (205 + r) + u34 (241 + r) + u34 (283 + r)
def L34_2 (r : ℕ) : ℤ := u34 (24 + r) + u34 (156 + r) + u34 (315 + r) + u34 (355 + r) + u34 (401 + r)
def L34_3 (r : ℕ) : ℤ := u34 (52 + r) + u34 (186 + r) + u34 (296 + r) + u34 (441 + r) + u34 (489 + r)
def L34_4 (r : ℕ) : ℤ := u34 (82 + r) + u34 (218 + r) + u34 (332 + r) + u34 (418 + r) + u34 (537 + r)
def L34_5 (r : ℕ) : ℤ := u34 (116 + r) + u34 (254 + r) + u34 (372 + r) + u34 (460 + r) + u34 (508 + r)

def aS34_0 (r : ℕ) : ℤ := S34_0 r - L34_0 r
def MS34_0 : ℤ := CaseSplit.mxr (aS34_0) 10
def aS34_1 (r : ℕ) : ℤ := S34_1 r - L34_1 r
def MS34_1 : ℤ := CaseSplit.mxr (aS34_1) 12
def aS34_2 (r : ℕ) : ℤ := S34_2 r - L34_2 r
def MS34_2 : ℤ := CaseSplit.mxr (aS34_2) 16
def aS34_3 (r : ℕ) : ℤ := S34_3 r - L34_3 r
def MS34_3 : ℤ := CaseSplit.mxr (aS34_3) 18
def aS34_4 (r : ℕ) : ℤ := S34_4 r - L34_4 r
def MS34_4 : ℤ := CaseSplit.mxr (aS34_4) 22
def aS34_5 (r : ℕ) : ℤ := S34_5 r - L34_5 r
def MS34_5 : ℤ := CaseSplit.mxr (aS34_5) 28

def N34_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n34, (if c34_0 ra t && c34_1 rb t then 1 else 0)
def aP34_0 (ra rb : ℕ) : ℤ := -(4) * N34_0 ra rb + u34 (0 + rb) + u34 (13 + ra)
def MP34_0 : ℤ := CaseSplit.mxr2 (aP34_0) 10 12
def N34_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n34, (if c34_0 ra t && c34_2 rb t then 1 else 0)
def aP34_1 (ra rb : ℕ) : ℤ := -(4) * N34_1 ra rb + u34 (24 + rb) + u34 (41 + ra)
def MP34_1 : ℤ := CaseSplit.mxr2 (aP34_1) 10 16
def N34_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n34, (if c34_0 ra t && c34_3 rb t then 1 else 0)
def aP34_2 (ra rb : ℕ) : ℤ := -(4) * N34_2 ra rb + u34 (52 + rb) + u34 (71 + ra)
def MP34_2 : ℤ := CaseSplit.mxr2 (aP34_2) 10 18
def N34_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n34, (if c34_0 ra t && c34_4 rb t then 1 else 0)
def aP34_3 (ra rb : ℕ) : ℤ := -(4) * N34_3 ra rb + u34 (82 + rb) + u34 (105 + ra)
def MP34_3 : ℤ := CaseSplit.mxr2 (aP34_3) 10 22
def N34_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n34, (if c34_0 ra t && c34_5 rb t then 1 else 0)
def aP34_4 (ra rb : ℕ) : ℤ := -(4) * N34_4 ra rb + u34 (116 + rb) + u34 (145 + ra)
def MP34_4 : ℤ := CaseSplit.mxr2 (aP34_4) 10 28
def P34_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n34, (if c34_1 ra t && c34_2 rb t then 1 else 0)
def C34_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n34, (if c34_1 ra t && c34_2 rb t && c34_0 s t then 1 else 0)
def M34_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C34_5 ra rb) 10
def E34_5 : List ℕ := [7, 13, 86, 97, 102, 108, 138, 144, 172, 183, 192, 198]
def N34_5 (ra rb : ℕ) : ℤ := if E34_5.contains (ra * 17 + rb) = true then P34_5 ra rb - M34_5 ra rb else 0
def aP34_5 (ra rb : ℕ) : ℤ := -(4) * N34_5 ra rb + u34 (156 + rb) + u34 (173 + ra)
def MP34_5 : ℤ := CaseSplit.mxr2 (aP34_5) 12 16
def P34_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n34, (if c34_1 ra t && c34_3 rb t then 1 else 0)
def C34_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n34, (if c34_1 ra t && c34_3 rb t && c34_0 s t then 1 else 0)
def M34_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C34_6 ra rb) 10
def E34_6 : List ℕ := [11, 47, 53, 84, 87, 111, 118, 160, 187, 194, 218, 224]
def N34_6 (ra rb : ℕ) : ℤ := if E34_6.contains (ra * 19 + rb) = true then P34_6 ra rb - M34_6 ra rb else 0
def aP34_6 (ra rb : ℕ) : ℤ := -(4) * N34_6 ra rb + u34 (186 + rb) + u34 (205 + ra)
def MP34_6 : ℤ := CaseSplit.mxr2 (aP34_6) 12 18
def P34_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n34, (if c34_1 ra t && c34_4 rb t then 1 else 0)
def C34_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n34, (if c34_1 ra t && c34_4 rb t && c34_0 s t then 1 else 0)
def M34_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C34_7 ra rb) 10
def E34_7 : List ℕ := []
def N34_7 (ra rb : ℕ) : ℤ := if E34_7.contains (ra * 23 + rb) = true then P34_7 ra rb - M34_7 ra rb else 0
def aP34_7 (ra rb : ℕ) : ℤ := -(4) * N34_7 ra rb + u34 (218 + rb) + u34 (241 + ra)
def MP34_7 : ℤ := CaseSplit.mxr2 (aP34_7) 12 22
def P34_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n34, (if c34_1 ra t && c34_5 rb t then 1 else 0)
def C34_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n34, (if c34_1 ra t && c34_5 rb t && c34_0 s t then 1 else 0)
def M34_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C34_8 ra rb) 10
def E34_8 : List ℕ := [223, 339]
def N34_8 (ra rb : ℕ) : ℤ := if E34_8.contains (ra * 29 + rb) = true then P34_8 ra rb - M34_8 ra rb else 0
def aP34_8 (ra rb : ℕ) : ℤ := -(4) * N34_8 ra rb + u34 (254 + rb) + u34 (283 + ra)
def MP34_8 : ℤ := CaseSplit.mxr2 (aP34_8) 12 28
def N34_9 (_ra _rb : ℕ) : ℤ := 0
def aP34_9 (ra rb : ℕ) : ℤ := -(4) * N34_9 ra rb + u34 (296 + rb) + u34 (315 + ra)
def MP34_9 : ℤ := CaseSplit.mxr2 (aP34_9) 16 18
def N34_10 (_ra _rb : ℕ) : ℤ := 0
def aP34_10 (ra rb : ℕ) : ℤ := -(4) * N34_10 ra rb + u34 (332 + rb) + u34 (355 + ra)
def MP34_10 : ℤ := CaseSplit.mxr2 (aP34_10) 16 22
def N34_11 (_ra _rb : ℕ) : ℤ := 0
def aP34_11 (ra rb : ℕ) : ℤ := -(4) * N34_11 ra rb + u34 (372 + rb) + u34 (401 + ra)
def MP34_11 : ℤ := CaseSplit.mxr2 (aP34_11) 16 28
def N34_12 (_ra _rb : ℕ) : ℤ := 0
def aP34_12 (ra rb : ℕ) : ℤ := -(4) * N34_12 ra rb + u34 (418 + rb) + u34 (441 + ra)
def MP34_12 : ℤ := CaseSplit.mxr2 (aP34_12) 18 22
def N34_13 (_ra _rb : ℕ) : ℤ := 0
def aP34_13 (ra rb : ℕ) : ℤ := -(4) * N34_13 ra rb + u34 (460 + rb) + u34 (489 + ra)
def MP34_13 : ℤ := CaseSplit.mxr2 (aP34_13) 18 28
def N34_14 (_ra _rb : ℕ) : ℤ := 0
def aP34_14 (ra rb : ℕ) : ℤ := -(4) * N34_14 ra rb + u34 (508 + rb) + u34 (537 + ra)
def MP34_14 : ℤ := CaseSplit.mxr2 (aP34_14) 22 28

def rhs34 : ℤ := (∑ t ∈ Finset.range n34, w34 t) + 4 * (n34 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn34 : ∀ t, t < n34 → (0 : ℤ) ≤ w34 t := by decide
theorem plt34 : ∀ t, t < n34 → q34 t < 49 := by decide
theorem pfree34_5 : ∀ t, t < n34 → gb5 4 (q34 t) = false := by decide
theorem pfree34_7 : ∀ t, t < n34 → gb7 6 (q34 t) = false := by decide
theorem MSv34_0 : MS34_0 = 20 := by decide +kernel
theorem MSv34_1 : MS34_1 = 54 := by decide +kernel
theorem MSv34_2 : MS34_2 = 1 := by decide +kernel
theorem MSv34_3 : MS34_3 = 1 := by decide +kernel
theorem MSv34_4 : MS34_4 = 0 := by decide +kernel
theorem MSv34_5 : MS34_5 = 0 := by decide +kernel
theorem MPv34_0 : MP34_0 = 0 := by decide +kernel
theorem MPv34_1 : MP34_1 = 0 := by decide +kernel
theorem MPv34_2 : MP34_2 = 0 := by decide +kernel
theorem MPv34_3 : MP34_3 = 0 := by decide +kernel
theorem MPv34_4 : MP34_4 = 0 := by decide +kernel
theorem MPv34_5 : MP34_5 = 0 := by decide +kernel
theorem MPv34_6 : MP34_6 = 0 := by decide +kernel
theorem MPv34_7 : MP34_7 = 0 := by decide +kernel
theorem MPv34_8 : MP34_8 = 0 := by decide +kernel
theorem MPv34_9 : MP34_9 = 0 := by decide +kernel
theorem MPv34_10 : MP34_10 = 0 := by decide +kernel
theorem MPv34_11 : MP34_11 = 0 := by decide +kernel
theorem MPv34_12 : MP34_12 = 0 := by decide +kernel
theorem MPv34_13 : MP34_13 = 0 := by decide +kernel
theorem MPv34_14 : MP34_14 = 21 := by decide +kernel
theorem rhsv34 : rhs34 = 98 := by decide +kernel

/-- **The case-34 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/4.
    (Scaled by the common denominator 4: 97 < 98.) -/
theorem cert34 : MS34_0 + MS34_1 + MS34_2 + MS34_3 + MS34_4 + MS34_5 + MP34_0 + MP34_1 + MP34_2 + MP34_3 + MP34_4 + MP34_5 + MP34_6 + MP34_7 + MP34_8 + MP34_9 + MP34_10 + MP34_11 + MP34_12 + MP34_13 + MP34_14 < rhs34 := by
  rw [MSv34_0, MSv34_1, MSv34_2, MSv34_3, MSv34_4, MSv34_5, MPv34_0, MPv34_1, MPv34_2, MPv34_3, MPv34_4, MPv34_5, MPv34_6, MPv34_7, MPv34_8, MPv34_9, MPv34_10, MPv34_11, MPv34_12, MPv34_13, MPv34_14, rhsv34]
  decide

def Dg34 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := (if c34_0 r0 t then 1 else 0) + (if c34_1 r1 t then 1 else 0) + (if c34_2 r2 t then 1 else 0) + (if c34_3 r3 t then 1 else 0) + (if c34_4 r4 t then 1 else 0) + (if c34_5 r5 t then 1 else 0)
def Wl34_0 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c34_0 r0 t && c34_1 r1 t then 1 else 0
def Wl34_1 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c34_0 r0 t && c34_2 r2 t then 1 else 0
def Wl34_2 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c34_0 r0 t && c34_3 r3 t then 1 else 0
def Wl34_3 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c34_0 r0 t && c34_4 r4 t then 1 else 0
def Wl34_4 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c34_0 r0 t && c34_5 r5 t then 1 else 0
def Wl34_5 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c34_0 r0 t && c34_1 r1 t && c34_2 r2 t then 1 else 0
def Wl34_6 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c34_0 r0 t && c34_1 r1 t && c34_3 r3 t then 1 else 0
def Wl34_7 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c34_0 r0 t && c34_1 r1 t && c34_4 r4 t then 1 else 0
def Wl34_8 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c34_0 r0 t && c34_1 r1 t && c34_5 r5 t then 1 else 0
def Wl34_9 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c34_0 r0 t && !c34_1 r1 t && c34_2 r2 t && c34_3 r3 t then 1 else 0
def Wl34_10 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c34_0 r0 t && !c34_1 r1 t && c34_2 r2 t && c34_4 r4 t then 1 else 0
def Wl34_11 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c34_0 r0 t && !c34_1 r1 t && c34_2 r2 t && c34_5 r5 t then 1 else 0
def Wl34_12 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c34_0 r0 t && !c34_1 r1 t && !c34_2 r2 t && c34_3 r3 t && c34_4 r4 t then 1 else 0
def Wl34_13 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c34_0 r0 t && !c34_1 r1 t && !c34_2 r2 t && c34_3 r3 t && c34_5 r5 t then 1 else 0
def Wl34_14 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c34_0 r0 t && !c34_1 r1 t && !c34_2 r2 t && !c34_3 r3 t && c34_4 r4 t && c34_5 r5 t then 1 else 0

/-- **No configuration blocks the whole window in case 34.** -/
theorem nocov34 {r0 r1 r2 r3 r4 r5 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29)
    (hcov : ∀ t, t < n34 → (c34_0 r0 t || c34_1 r1 t || c34_2 r2 t || c34_3 r3 t || c34_4 r4 t || c34_5 r5 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n34, (1 : ℤ) + (Wl34_0 r0 r1 r2 r3 r4 r5 t + Wl34_1 r0 r1 r2 r3 r4 r5 t + Wl34_2 r0 r1 r2 r3 r4 r5 t + Wl34_3 r0 r1 r2 r3 r4 r5 t + Wl34_4 r0 r1 r2 r3 r4 r5 t + Wl34_5 r0 r1 r2 r3 r4 r5 t + Wl34_6 r0 r1 r2 r3 r4 r5 t + Wl34_7 r0 r1 r2 r3 r4 r5 t + Wl34_8 r0 r1 r2 r3 r4 r5 t + Wl34_9 r0 r1 r2 r3 r4 r5 t + Wl34_10 r0 r1 r2 r3 r4 r5 t + Wl34_11 r0 r1 r2 r3 r4 r5 t + Wl34_12 r0 r1 r2 r3 r4 r5 t + Wl34_13 r0 r1 r2 r3 r4 r5 t + Wl34_14 r0 r1 r2 r3 r4 r5 t) ≤ Dg34 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Wl34_0, Wl34_1, Wl34_2, Wl34_3, Wl34_4, Wl34_5, Wl34_6, Wl34_7, Wl34_8, Wl34_9, Wl34_10, Wl34_11, Wl34_12, Wl34_13, Wl34_14, Dg34]
    exact CaseSplit.lowest6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n34, (1 : ℤ) ≤ Dg34 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Dg34]
    exact CaseSplit.degpos6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n34 : ℤ) + ((∑ t ∈ Finset.range n34, Wl34_0 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n34, Wl34_1 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n34, Wl34_2 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n34, Wl34_3 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n34, Wl34_4 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n34, Wl34_5 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n34, Wl34_6 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n34, Wl34_7 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n34, Wl34_8 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n34, Wl34_9 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n34, Wl34_10 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n34, Wl34_11 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n34, Wl34_12 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n34, Wl34_13 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n34, Wl34_14 r0 r1 r2 r3 r4 r5 t)) ≤ ∑ t ∈ Finset.range n34, Dg34 r0 r1 r2 r3 r4 r5 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N34_0 r0 r1 ≤ ∑ t ∈ Finset.range n34, Wl34_0 r0 r1 r2 r3 r4 r5 t := by
    simp only [N34_0, Wl34_0, le_refl]
  have hn1 : N34_1 r0 r2 ≤ ∑ t ∈ Finset.range n34, Wl34_1 r0 r1 r2 r3 r4 r5 t := by
    simp only [N34_1, Wl34_1, le_refl]
  have hn2 : N34_2 r0 r3 ≤ ∑ t ∈ Finset.range n34, Wl34_2 r0 r1 r2 r3 r4 r5 t := by
    simp only [N34_2, Wl34_2, le_refl]
  have hn3 : N34_3 r0 r4 ≤ ∑ t ∈ Finset.range n34, Wl34_3 r0 r1 r2 r3 r4 r5 t := by
    simp only [N34_3, Wl34_3, le_refl]
  have hn4 : N34_4 r0 r5 ≤ ∑ t ∈ Finset.range n34, Wl34_4 r0 r1 r2 r3 r4 r5 t := by
    simp only [N34_4, Wl34_4, le_refl]
  have hn5 : N34_5 r1 r2 ≤ ∑ t ∈ Finset.range n34, Wl34_5 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n34, Wl34_5 r0 r1 r2 r3 r4 r5 t
        = (if c34_1 r1 t && c34_2 r2 t then (1:ℤ) else 0)
          - (if c34_1 r1 t && c34_2 r2 t && c34_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl34_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n34, Wl34_5 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl34_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n34, Wl34_5 r0 r1 r2 r3 r4 r5 t
        = P34_5 r1 r2 - C34_5 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P34_5, C34_5]
    have hm : C34_5 r1 r2 r0 ≤ M34_5 r1 r2 :=
      CaseSplit.le_mxr (C34_5 r1 r2) 10 r0 (by omega)
    simp only [N34_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N34_6 r1 r3 ≤ ∑ t ∈ Finset.range n34, Wl34_6 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n34, Wl34_6 r0 r1 r2 r3 r4 r5 t
        = (if c34_1 r1 t && c34_3 r3 t then (1:ℤ) else 0)
          - (if c34_1 r1 t && c34_3 r3 t && c34_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl34_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n34, Wl34_6 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl34_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n34, Wl34_6 r0 r1 r2 r3 r4 r5 t
        = P34_6 r1 r3 - C34_6 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P34_6, C34_6]
    have hm : C34_6 r1 r3 r0 ≤ M34_6 r1 r3 :=
      CaseSplit.le_mxr (C34_6 r1 r3) 10 r0 (by omega)
    simp only [N34_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N34_7 r1 r4 ≤ ∑ t ∈ Finset.range n34, Wl34_7 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n34, Wl34_7 r0 r1 r2 r3 r4 r5 t
        = (if c34_1 r1 t && c34_4 r4 t then (1:ℤ) else 0)
          - (if c34_1 r1 t && c34_4 r4 t && c34_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl34_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n34, Wl34_7 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl34_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n34, Wl34_7 r0 r1 r2 r3 r4 r5 t
        = P34_7 r1 r4 - C34_7 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P34_7, C34_7]
    have hm : C34_7 r1 r4 r0 ≤ M34_7 r1 r4 :=
      CaseSplit.le_mxr (C34_7 r1 r4) 10 r0 (by omega)
    simp only [N34_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N34_8 r1 r5 ≤ ∑ t ∈ Finset.range n34, Wl34_8 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n34, Wl34_8 r0 r1 r2 r3 r4 r5 t
        = (if c34_1 r1 t && c34_5 r5 t then (1:ℤ) else 0)
          - (if c34_1 r1 t && c34_5 r5 t && c34_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl34_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n34, Wl34_8 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl34_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n34, Wl34_8 r0 r1 r2 r3 r4 r5 t
        = P34_8 r1 r5 - C34_8 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P34_8, C34_8]
    have hm : C34_8 r1 r5 r0 ≤ M34_8 r1 r5 :=
      CaseSplit.le_mxr (C34_8 r1 r5) 10 r0 (by omega)
    simp only [N34_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N34_9 r2 r3 ≤ ∑ t ∈ Finset.range n34, Wl34_9 r0 r1 r2 r3 r4 r5 t := by
    simp only [N34_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl34_9]
    exact CaseSplit.ind_nonneg _
  have hn10 : N34_10 r2 r4 ≤ ∑ t ∈ Finset.range n34, Wl34_10 r0 r1 r2 r3 r4 r5 t := by
    simp only [N34_10]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl34_10]
    exact CaseSplit.ind_nonneg _
  have hn11 : N34_11 r2 r5 ≤ ∑ t ∈ Finset.range n34, Wl34_11 r0 r1 r2 r3 r4 r5 t := by
    simp only [N34_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl34_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N34_12 r3 r4 ≤ ∑ t ∈ Finset.range n34, Wl34_12 r0 r1 r2 r3 r4 r5 t := by
    simp only [N34_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl34_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N34_13 r3 r5 ≤ ∑ t ∈ Finset.range n34, Wl34_13 r0 r1 r2 r3 r4 r5 t := by
    simp only [N34_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl34_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N34_14 r4 r5 ≤ ∑ t ∈ Finset.range n34, Wl34_14 r0 r1 r2 r3 r4 r5 t := by
    simp only [N34_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl34_14]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n34, (w34 t + 4) * Dg34 r0 r1 r2 r3 r4 r5 t = S34_0 r0 + S34_1 r1 + S34_2 r2 + S34_3 r3 + S34_4 r4 + S34_5 r5 := by
    simp only [S34_0, S34_1, S34_2, S34_3, S34_4, S34_5, Dg34, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n34, (w34 t + 4) * Dg34 r0 r1 r2 r3 r4 r5 t
      = (∑ t ∈ Finset.range n34, w34 t * Dg34 r0 r1 r2 r3 r4 r5 t)
        + 4 * (∑ t ∈ Finset.range n34, Dg34 r0 r1 r2 r3 r4 r5 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n34, w34 t)
      ≤ ∑ t ∈ Finset.range n34, w34 t * Dg34 r0 r1 r2 r3 r4 r5 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg34 r0 r1 r2 r3 r4 r5 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w34 t := wnn34 t (Finset.mem_range.mp ht)
    calc w34 t = w34 t * 1 := (mul_one _).symm
      _ ≤ w34 t * Dg34 r0 r1 r2 r3 r4 r5 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS34_0 r0 + aS34_1 r1 + aS34_2 r2 + aS34_3 r3 + aS34_4 r4 + aS34_5 r5) + (aP34_0 r0 r1 + aP34_1 r0 r2 + aP34_2 r0 r3 + aP34_3 r0 r4 + aP34_4 r0 r5 + aP34_5 r1 r2 + aP34_6 r1 r3 + aP34_7 r1 r4 + aP34_8 r1 r5 + aP34_9 r2 r3 + aP34_10 r2 r4 + aP34_11 r2 r5 + aP34_12 r3 r4 + aP34_13 r3 r5 + aP34_14 r4 r5) = (S34_0 r0 + S34_1 r1 + S34_2 r2 + S34_3 r3 + S34_4 r4 + S34_5 r5) - 4 * (N34_0 r0 r1 + N34_1 r0 r2 + N34_2 r0 r3 + N34_3 r0 r4 + N34_4 r0 r5 + N34_5 r1 r2 + N34_6 r1 r3 + N34_7 r1 r4 + N34_8 r1 r5 + N34_9 r2 r3 + N34_10 r2 r4 + N34_11 r2 r5 + N34_12 r3 r4 + N34_13 r3 r5 + N34_14 r4 r5) := by
    simp only [aS34_0, aS34_1, aS34_2, aS34_3, aS34_4, aS34_5, aP34_0, aP34_1, aP34_2, aP34_3, aP34_4, aP34_5, aP34_6, aP34_7, aP34_8, aP34_9, aP34_10, aP34_11, aP34_12, aP34_13, aP34_14, L34_0, L34_1, L34_2, L34_3, L34_4, L34_5]
    ring
  have bS0 : aS34_0 r0 ≤ MS34_0 := CaseSplit.le_mxr (aS34_0) 10 r0 (by omega)
  have bS1 : aS34_1 r1 ≤ MS34_1 := CaseSplit.le_mxr (aS34_1) 12 r1 (by omega)
  have bS2 : aS34_2 r2 ≤ MS34_2 := CaseSplit.le_mxr (aS34_2) 16 r2 (by omega)
  have bS3 : aS34_3 r3 ≤ MS34_3 := CaseSplit.le_mxr (aS34_3) 18 r3 (by omega)
  have bS4 : aS34_4 r4 ≤ MS34_4 := CaseSplit.le_mxr (aS34_4) 22 r4 (by omega)
  have bS5 : aS34_5 r5 ≤ MS34_5 := CaseSplit.le_mxr (aS34_5) 28 r5 (by omega)
  have bP0 : aP34_0 r0 r1 ≤ MP34_0 := CaseSplit.le_mxr2 (aP34_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP34_1 r0 r2 ≤ MP34_1 := CaseSplit.le_mxr2 (aP34_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP34_2 r0 r3 ≤ MP34_2 := CaseSplit.le_mxr2 (aP34_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP34_3 r0 r4 ≤ MP34_3 := CaseSplit.le_mxr2 (aP34_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP34_4 r0 r5 ≤ MP34_4 := CaseSplit.le_mxr2 (aP34_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP34_5 r1 r2 ≤ MP34_5 := CaseSplit.le_mxr2 (aP34_5) 12 16 r1 r2 (by omega) (by omega)
  have bP6 : aP34_6 r1 r3 ≤ MP34_6 := CaseSplit.le_mxr2 (aP34_6) 12 18 r1 r3 (by omega) (by omega)
  have bP7 : aP34_7 r1 r4 ≤ MP34_7 := CaseSplit.le_mxr2 (aP34_7) 12 22 r1 r4 (by omega) (by omega)
  have bP8 : aP34_8 r1 r5 ≤ MP34_8 := CaseSplit.le_mxr2 (aP34_8) 12 28 r1 r5 (by omega) (by omega)
  have bP9 : aP34_9 r2 r3 ≤ MP34_9 := CaseSplit.le_mxr2 (aP34_9) 16 18 r2 r3 (by omega) (by omega)
  have bP10 : aP34_10 r2 r4 ≤ MP34_10 := CaseSplit.le_mxr2 (aP34_10) 16 22 r2 r4 (by omega) (by omega)
  have bP11 : aP34_11 r2 r5 ≤ MP34_11 := CaseSplit.le_mxr2 (aP34_11) 16 28 r2 r5 (by omega) (by omega)
  have bP12 : aP34_12 r3 r4 ≤ MP34_12 := CaseSplit.le_mxr2 (aP34_12) 18 22 r3 r4 (by omega) (by omega)
  have bP13 : aP34_13 r3 r5 ≤ MP34_13 := CaseSplit.le_mxr2 (aP34_13) 18 28 r3 r5 (by omega) (by omega)
  have bP14 : aP34_14 r4 r5 ≤ MP34_14 := CaseSplit.le_mxr2 (aP34_14) 22 28 r4 r5 (by omega) (by omega)
  have hrhs : rhs34 = (∑ t ∈ Finset.range n34, w34 t) + 4 * (n34 : ℤ) := rfl
  have hc := cert34
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, bS0, bS1, bS2, bS3, bS4, bS5, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14]

end IncCert29
