/-
INCREMENT-WIDTH CERTIFICATE, step 23->29, case 28 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_23_29.json, which re-derives every number
from the primes alone).

Machine 29, INCREMENT width 49 = F_2(23) + s_min(29) = 39 + 10,
held gears [5, 7] at phases [4, 0].  Free gears [11, 13, 17, 19, 23, 29].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 8.
-/
import IncCert29B

namespace IncCert29

/-! ### case 28: held gears at phases [4, 0] -/

def p28 : List ℕ := [3, 4, 9, 11, 14, 16, 18, 19, 21, 23, 24, 26, 28, 31, 33, 38, 39, 44, 46]
def q28 (t : ℕ) : ℕ := p28.getD t 0
def n28 : ℕ := 19
def yl28 : List ℤ := [0, 0, 3, 4, 5, 6, 4, 7, 8, 8, 4, 6, 4, 2, 2, 0, 1, 0, 1]
def w28 (t : ℕ) : ℤ := yl28.getD t 0
def ul28 : List ℤ := [(-1), (-2), (-2), (-2), 0, (-2), 0, (-2), 0, (-2), 0, (-6), (-2), 2, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, (-3), (-3), (-3), 0, 0, 0, (-3), (-3), (-3), 0, 1, 0, (-3), (-3), (-3), 0, 0, (-2), (-1), 0, 0, 0, 0, (-1), (-1), 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), 0, (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-14), 0, 0, 0, (-15), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 12, 17, 14, 15, 15, 15, 9, 17, 17, 17, 17, 12, 16, 13, 16, 16, (-17), (-17), (-19), (-17), (-17), (-17), (-17), (-23), (-17), (-17), (-17), (-17), (-17), 14, 11, 14, 12, 12, 7, 14, 12, 14, 9, 14, 14, 13, 14, 13, 14, 14, 14, 14, (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), 7, 7, 1, 7, 7, 0, 7, 1, 5, 0, 6, 7, 0, 0, 7, 7, 7, 5, 7, 7, 1, 7, 7, (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 17, 3, 15, 0, 9, 17, 4, 17, 3, 14, 10, 0, 17, 4, 17, 17, 17, 8, 3, 14, 17, 5, 7, 4, 15, 0, 9, 0, 5, 12, 0, 12, 9, 8, 6, 0, 12, 12, 0, 10, 3, 8, 0, (-4), 12, 0, 6, 12, 0, 6, 0]
def u28 (k : ℕ) : ℤ := ul28.getD k 0

def c28_0 (r t : ℕ) : Bool := gb11 r (q28 t)
def c28_1 (r t : ℕ) : Bool := gb13 r (q28 t)
def c28_2 (r t : ℕ) : Bool := gb17 r (q28 t)
def c28_3 (r t : ℕ) : Bool := gb19 r (q28 t)
def c28_4 (r t : ℕ) : Bool := gb23 r (q28 t)
def c28_5 (r t : ℕ) : Bool := gb29 r (q28 t)

def S28_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n28, (w28 t + 3) * (if c28_0 r t then 1 else 0)
def S28_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n28, (w28 t + 3) * (if c28_1 r t then 1 else 0)
def S28_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n28, (w28 t + 3) * (if c28_2 r t then 1 else 0)
def S28_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n28, (w28 t + 3) * (if c28_3 r t then 1 else 0)
def S28_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n28, (w28 t + 3) * (if c28_4 r t then 1 else 0)
def S28_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n28, (w28 t + 3) * (if c28_5 r t then 1 else 0)

def L28_0 (r : ℕ) : ℤ := u28 (13 + r) + u28 (41 + r) + u28 (71 + r) + u28 (105 + r) + u28 (145 + r)
def L28_1 (r : ℕ) : ℤ := u28 (0 + r) + u28 (173 + r) + u28 (205 + r) + u28 (241 + r) + u28 (283 + r)
def L28_2 (r : ℕ) : ℤ := u28 (24 + r) + u28 (156 + r) + u28 (315 + r) + u28 (355 + r) + u28 (401 + r)
def L28_3 (r : ℕ) : ℤ := u28 (52 + r) + u28 (186 + r) + u28 (296 + r) + u28 (441 + r) + u28 (489 + r)
def L28_4 (r : ℕ) : ℤ := u28 (82 + r) + u28 (218 + r) + u28 (332 + r) + u28 (418 + r) + u28 (537 + r)
def L28_5 (r : ℕ) : ℤ := u28 (116 + r) + u28 (254 + r) + u28 (372 + r) + u28 (460 + r) + u28 (508 + r)

def aS28_0 (r : ℕ) : ℤ := S28_0 r - L28_0 r
def MS28_0 : ℤ := CaseSplit.mxr (aS28_0) 10
def aS28_1 (r : ℕ) : ℤ := S28_1 r - L28_1 r
def MS28_1 : ℤ := CaseSplit.mxr (aS28_1) 12
def aS28_2 (r : ℕ) : ℤ := S28_2 r - L28_2 r
def MS28_2 : ℤ := CaseSplit.mxr (aS28_2) 16
def aS28_3 (r : ℕ) : ℤ := S28_3 r - L28_3 r
def MS28_3 : ℤ := CaseSplit.mxr (aS28_3) 18
def aS28_4 (r : ℕ) : ℤ := S28_4 r - L28_4 r
def MS28_4 : ℤ := CaseSplit.mxr (aS28_4) 22
def aS28_5 (r : ℕ) : ℤ := S28_5 r - L28_5 r
def MS28_5 : ℤ := CaseSplit.mxr (aS28_5) 28

def N28_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n28, (if c28_0 ra t && c28_1 rb t then 1 else 0)
def aP28_0 (ra rb : ℕ) : ℤ := -(3) * N28_0 ra rb + u28 (0 + rb) + u28 (13 + ra)
def MP28_0 : ℤ := CaseSplit.mxr2 (aP28_0) 10 12
def N28_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n28, (if c28_0 ra t && c28_2 rb t then 1 else 0)
def aP28_1 (ra rb : ℕ) : ℤ := -(3) * N28_1 ra rb + u28 (24 + rb) + u28 (41 + ra)
def MP28_1 : ℤ := CaseSplit.mxr2 (aP28_1) 10 16
def N28_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n28, (if c28_0 ra t && c28_3 rb t then 1 else 0)
def aP28_2 (ra rb : ℕ) : ℤ := -(3) * N28_2 ra rb + u28 (52 + rb) + u28 (71 + ra)
def MP28_2 : ℤ := CaseSplit.mxr2 (aP28_2) 10 18
def N28_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n28, (if c28_0 ra t && c28_4 rb t then 1 else 0)
def aP28_3 (ra rb : ℕ) : ℤ := -(3) * N28_3 ra rb + u28 (82 + rb) + u28 (105 + ra)
def MP28_3 : ℤ := CaseSplit.mxr2 (aP28_3) 10 22
def N28_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n28, (if c28_0 ra t && c28_5 rb t then 1 else 0)
def aP28_4 (ra rb : ℕ) : ℤ := -(3) * N28_4 ra rb + u28 (116 + rb) + u28 (145 + ra)
def MP28_4 : ℤ := CaseSplit.mxr2 (aP28_4) 10 28
def P28_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n28, (if c28_1 ra t && c28_2 rb t then 1 else 0)
def C28_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n28, (if c28_1 ra t && c28_2 rb t && c28_0 s t then 1 else 0)
def M28_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C28_5 ra rb) 10
def E28_5 : List ℕ := [3, 9, 39, 45, 61, 67, 129, 135, 140, 151, 170, 176]
def N28_5 (ra rb : ℕ) : ℤ := if E28_5.contains (ra * 17 + rb) = true then P28_5 ra rb - M28_5 ra rb else 0
def aP28_5 (ra rb : ℕ) : ℤ := -(3) * N28_5 ra rb + u28 (156 + rb) + u28 (173 + ra)
def MP28_5 : ℤ := CaseSplit.mxr2 (aP28_5) 12 16
def P28_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n28, (if c28_1 ra t && c28_3 rb t then 1 else 0)
def C28_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n28, (if c28_1 ra t && c28_3 rb t && c28_0 s t then 1 else 0)
def M28_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C28_6 ra rb) 10
def E28_6 : List ℕ := [11, 53, 84, 87, 118, 124, 152, 160, 194, 200, 224, 228]
def N28_6 (ra rb : ℕ) : ℤ := if E28_6.contains (ra * 19 + rb) = true then P28_6 ra rb - M28_6 ra rb else 0
def aP28_6 (ra rb : ℕ) : ℤ := -(3) * N28_6 ra rb + u28 (186 + rb) + u28 (205 + ra)
def MP28_6 : ℤ := CaseSplit.mxr2 (aP28_6) 12 18
def P28_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n28, (if c28_1 ra t && c28_4 rb t then 1 else 0)
def C28_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n28, (if c28_1 ra t && c28_4 rb t && c28_0 s t then 1 else 0)
def M28_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C28_7 ra rb) 10
def E28_7 : List ℕ := []
def N28_7 (ra rb : ℕ) : ℤ := if E28_7.contains (ra * 23 + rb) = true then P28_7 ra rb - M28_7 ra rb else 0
def aP28_7 (ra rb : ℕ) : ℤ := -(3) * N28_7 ra rb + u28 (218 + rb) + u28 (241 + ra)
def MP28_7 : ℤ := CaseSplit.mxr2 (aP28_7) 12 22
def P28_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n28, (if c28_1 ra t && c28_5 rb t then 1 else 0)
def C28_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n28, (if c28_1 ra t && c28_5 rb t && c28_0 s t then 1 else 0)
def M28_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C28_8 ra rb) 10
def E28_8 : List ℕ := []
def N28_8 (ra rb : ℕ) : ℤ := if E28_8.contains (ra * 29 + rb) = true then P28_8 ra rb - M28_8 ra rb else 0
def aP28_8 (ra rb : ℕ) : ℤ := -(3) * N28_8 ra rb + u28 (254 + rb) + u28 (283 + ra)
def MP28_8 : ℤ := CaseSplit.mxr2 (aP28_8) 12 28
def N28_9 (_ra _rb : ℕ) : ℤ := 0
def aP28_9 (ra rb : ℕ) : ℤ := -(3) * N28_9 ra rb + u28 (296 + rb) + u28 (315 + ra)
def MP28_9 : ℤ := CaseSplit.mxr2 (aP28_9) 16 18
def N28_10 (_ra _rb : ℕ) : ℤ := 0
def aP28_10 (ra rb : ℕ) : ℤ := -(3) * N28_10 ra rb + u28 (332 + rb) + u28 (355 + ra)
def MP28_10 : ℤ := CaseSplit.mxr2 (aP28_10) 16 22
def N28_11 (_ra _rb : ℕ) : ℤ := 0
def aP28_11 (ra rb : ℕ) : ℤ := -(3) * N28_11 ra rb + u28 (372 + rb) + u28 (401 + ra)
def MP28_11 : ℤ := CaseSplit.mxr2 (aP28_11) 16 28
def N28_12 (_ra _rb : ℕ) : ℤ := 0
def aP28_12 (ra rb : ℕ) : ℤ := -(3) * N28_12 ra rb + u28 (418 + rb) + u28 (441 + ra)
def MP28_12 : ℤ := CaseSplit.mxr2 (aP28_12) 18 22
def N28_13 (_ra _rb : ℕ) : ℤ := 0
def aP28_13 (ra rb : ℕ) : ℤ := -(3) * N28_13 ra rb + u28 (460 + rb) + u28 (489 + ra)
def MP28_13 : ℤ := CaseSplit.mxr2 (aP28_13) 18 28
def N28_14 (_ra _rb : ℕ) : ℤ := 0
def aP28_14 (ra rb : ℕ) : ℤ := -(3) * N28_14 ra rb + u28 (508 + rb) + u28 (537 + ra)
def MP28_14 : ℤ := CaseSplit.mxr2 (aP28_14) 22 28

def rhs28 : ℤ := (∑ t ∈ Finset.range n28, w28 t) + 3 * (n28 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn28 : ∀ t, t < n28 → (0 : ℤ) ≤ w28 t := by decide
theorem plt28 : ∀ t, t < n28 → q28 t < 49 := by decide
theorem pfree28_5 : ∀ t, t < n28 → gb5 4 (q28 t) = false := by decide
theorem pfree28_7 : ∀ t, t < n28 → gb7 0 (q28 t) = false := by decide
theorem MSv28_0 : MS28_0 = 23 := by decide +kernel
theorem MSv28_1 : MS28_1 = 61 := by decide +kernel
theorem MSv28_2 : MS28_2 = 2 := by decide +kernel
theorem MSv28_3 : MS28_3 = 2 := by decide +kernel
theorem MSv28_4 : MS28_4 = 2 := by decide +kernel
theorem MSv28_5 : MS28_5 = 2 := by decide +kernel
theorem MPv28_0 : MP28_0 = 0 := by decide +kernel
theorem MPv28_1 : MP28_1 = 0 := by decide +kernel
theorem MPv28_2 : MP28_2 = 0 := by decide +kernel
theorem MPv28_3 : MP28_3 = 0 := by decide +kernel
theorem MPv28_4 : MP28_4 = 0 := by decide +kernel
theorem MPv28_5 : MP28_5 = 0 := by decide +kernel
theorem MPv28_6 : MP28_6 = 0 := by decide +kernel
theorem MPv28_7 : MP28_7 = 0 := by decide +kernel
theorem MPv28_8 : MP28_8 = 0 := by decide +kernel
theorem MPv28_9 : MP28_9 = 0 := by decide +kernel
theorem MPv28_10 : MP28_10 = 0 := by decide +kernel
theorem MPv28_11 : MP28_11 = 0 := by decide +kernel
theorem MPv28_12 : MP28_12 = 0 := by decide +kernel
theorem MPv28_13 : MP28_13 = 0 := by decide +kernel
theorem MPv28_14 : MP28_14 = 29 := by decide +kernel
theorem rhsv28 : rhs28 = 122 := by decide +kernel

/-- **The case-28 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/8.
    (Scaled by the common denominator 8: 121 < 122.) -/
theorem cert28 : MS28_0 + MS28_1 + MS28_2 + MS28_3 + MS28_4 + MS28_5 + MP28_0 + MP28_1 + MP28_2 + MP28_3 + MP28_4 + MP28_5 + MP28_6 + MP28_7 + MP28_8 + MP28_9 + MP28_10 + MP28_11 + MP28_12 + MP28_13 + MP28_14 < rhs28 := by
  rw [MSv28_0, MSv28_1, MSv28_2, MSv28_3, MSv28_4, MSv28_5, MPv28_0, MPv28_1, MPv28_2, MPv28_3, MPv28_4, MPv28_5, MPv28_6, MPv28_7, MPv28_8, MPv28_9, MPv28_10, MPv28_11, MPv28_12, MPv28_13, MPv28_14, rhsv28]
  decide

def Dg28 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := (if c28_0 r0 t then 1 else 0) + (if c28_1 r1 t then 1 else 0) + (if c28_2 r2 t then 1 else 0) + (if c28_3 r3 t then 1 else 0) + (if c28_4 r4 t then 1 else 0) + (if c28_5 r5 t then 1 else 0)
def Wl28_0 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c28_0 r0 t && c28_1 r1 t then 1 else 0
def Wl28_1 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c28_0 r0 t && c28_2 r2 t then 1 else 0
def Wl28_2 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c28_0 r0 t && c28_3 r3 t then 1 else 0
def Wl28_3 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c28_0 r0 t && c28_4 r4 t then 1 else 0
def Wl28_4 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c28_0 r0 t && c28_5 r5 t then 1 else 0
def Wl28_5 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c28_0 r0 t && c28_1 r1 t && c28_2 r2 t then 1 else 0
def Wl28_6 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c28_0 r0 t && c28_1 r1 t && c28_3 r3 t then 1 else 0
def Wl28_7 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c28_0 r0 t && c28_1 r1 t && c28_4 r4 t then 1 else 0
def Wl28_8 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c28_0 r0 t && c28_1 r1 t && c28_5 r5 t then 1 else 0
def Wl28_9 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c28_0 r0 t && !c28_1 r1 t && c28_2 r2 t && c28_3 r3 t then 1 else 0
def Wl28_10 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c28_0 r0 t && !c28_1 r1 t && c28_2 r2 t && c28_4 r4 t then 1 else 0
def Wl28_11 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c28_0 r0 t && !c28_1 r1 t && c28_2 r2 t && c28_5 r5 t then 1 else 0
def Wl28_12 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c28_0 r0 t && !c28_1 r1 t && !c28_2 r2 t && c28_3 r3 t && c28_4 r4 t then 1 else 0
def Wl28_13 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c28_0 r0 t && !c28_1 r1 t && !c28_2 r2 t && c28_3 r3 t && c28_5 r5 t then 1 else 0
def Wl28_14 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c28_0 r0 t && !c28_1 r1 t && !c28_2 r2 t && !c28_3 r3 t && c28_4 r4 t && c28_5 r5 t then 1 else 0

/-- **No configuration blocks the whole window in case 28.** -/
theorem nocov28 {r0 r1 r2 r3 r4 r5 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29)
    (hcov : ∀ t, t < n28 → (c28_0 r0 t || c28_1 r1 t || c28_2 r2 t || c28_3 r3 t || c28_4 r4 t || c28_5 r5 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n28, (1 : ℤ) + (Wl28_0 r0 r1 r2 r3 r4 r5 t + Wl28_1 r0 r1 r2 r3 r4 r5 t + Wl28_2 r0 r1 r2 r3 r4 r5 t + Wl28_3 r0 r1 r2 r3 r4 r5 t + Wl28_4 r0 r1 r2 r3 r4 r5 t + Wl28_5 r0 r1 r2 r3 r4 r5 t + Wl28_6 r0 r1 r2 r3 r4 r5 t + Wl28_7 r0 r1 r2 r3 r4 r5 t + Wl28_8 r0 r1 r2 r3 r4 r5 t + Wl28_9 r0 r1 r2 r3 r4 r5 t + Wl28_10 r0 r1 r2 r3 r4 r5 t + Wl28_11 r0 r1 r2 r3 r4 r5 t + Wl28_12 r0 r1 r2 r3 r4 r5 t + Wl28_13 r0 r1 r2 r3 r4 r5 t + Wl28_14 r0 r1 r2 r3 r4 r5 t) ≤ Dg28 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Wl28_0, Wl28_1, Wl28_2, Wl28_3, Wl28_4, Wl28_5, Wl28_6, Wl28_7, Wl28_8, Wl28_9, Wl28_10, Wl28_11, Wl28_12, Wl28_13, Wl28_14, Dg28]
    exact CaseSplit.lowest6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n28, (1 : ℤ) ≤ Dg28 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Dg28]
    exact CaseSplit.degpos6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n28 : ℤ) + ((∑ t ∈ Finset.range n28, Wl28_0 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n28, Wl28_1 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n28, Wl28_2 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n28, Wl28_3 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n28, Wl28_4 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n28, Wl28_5 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n28, Wl28_6 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n28, Wl28_7 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n28, Wl28_8 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n28, Wl28_9 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n28, Wl28_10 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n28, Wl28_11 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n28, Wl28_12 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n28, Wl28_13 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n28, Wl28_14 r0 r1 r2 r3 r4 r5 t)) ≤ ∑ t ∈ Finset.range n28, Dg28 r0 r1 r2 r3 r4 r5 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N28_0 r0 r1 ≤ ∑ t ∈ Finset.range n28, Wl28_0 r0 r1 r2 r3 r4 r5 t := by
    simp only [N28_0, Wl28_0, le_refl]
  have hn1 : N28_1 r0 r2 ≤ ∑ t ∈ Finset.range n28, Wl28_1 r0 r1 r2 r3 r4 r5 t := by
    simp only [N28_1, Wl28_1, le_refl]
  have hn2 : N28_2 r0 r3 ≤ ∑ t ∈ Finset.range n28, Wl28_2 r0 r1 r2 r3 r4 r5 t := by
    simp only [N28_2, Wl28_2, le_refl]
  have hn3 : N28_3 r0 r4 ≤ ∑ t ∈ Finset.range n28, Wl28_3 r0 r1 r2 r3 r4 r5 t := by
    simp only [N28_3, Wl28_3, le_refl]
  have hn4 : N28_4 r0 r5 ≤ ∑ t ∈ Finset.range n28, Wl28_4 r0 r1 r2 r3 r4 r5 t := by
    simp only [N28_4, Wl28_4, le_refl]
  have hn5 : N28_5 r1 r2 ≤ ∑ t ∈ Finset.range n28, Wl28_5 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n28, Wl28_5 r0 r1 r2 r3 r4 r5 t
        = (if c28_1 r1 t && c28_2 r2 t then (1:ℤ) else 0)
          - (if c28_1 r1 t && c28_2 r2 t && c28_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl28_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n28, Wl28_5 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl28_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n28, Wl28_5 r0 r1 r2 r3 r4 r5 t
        = P28_5 r1 r2 - C28_5 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P28_5, C28_5]
    have hm : C28_5 r1 r2 r0 ≤ M28_5 r1 r2 :=
      CaseSplit.le_mxr (C28_5 r1 r2) 10 r0 (by omega)
    simp only [N28_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N28_6 r1 r3 ≤ ∑ t ∈ Finset.range n28, Wl28_6 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n28, Wl28_6 r0 r1 r2 r3 r4 r5 t
        = (if c28_1 r1 t && c28_3 r3 t then (1:ℤ) else 0)
          - (if c28_1 r1 t && c28_3 r3 t && c28_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl28_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n28, Wl28_6 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl28_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n28, Wl28_6 r0 r1 r2 r3 r4 r5 t
        = P28_6 r1 r3 - C28_6 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P28_6, C28_6]
    have hm : C28_6 r1 r3 r0 ≤ M28_6 r1 r3 :=
      CaseSplit.le_mxr (C28_6 r1 r3) 10 r0 (by omega)
    simp only [N28_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N28_7 r1 r4 ≤ ∑ t ∈ Finset.range n28, Wl28_7 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n28, Wl28_7 r0 r1 r2 r3 r4 r5 t
        = (if c28_1 r1 t && c28_4 r4 t then (1:ℤ) else 0)
          - (if c28_1 r1 t && c28_4 r4 t && c28_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl28_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n28, Wl28_7 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl28_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n28, Wl28_7 r0 r1 r2 r3 r4 r5 t
        = P28_7 r1 r4 - C28_7 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P28_7, C28_7]
    have hm : C28_7 r1 r4 r0 ≤ M28_7 r1 r4 :=
      CaseSplit.le_mxr (C28_7 r1 r4) 10 r0 (by omega)
    simp only [N28_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N28_8 r1 r5 ≤ ∑ t ∈ Finset.range n28, Wl28_8 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n28, Wl28_8 r0 r1 r2 r3 r4 r5 t
        = (if c28_1 r1 t && c28_5 r5 t then (1:ℤ) else 0)
          - (if c28_1 r1 t && c28_5 r5 t && c28_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl28_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n28, Wl28_8 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl28_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n28, Wl28_8 r0 r1 r2 r3 r4 r5 t
        = P28_8 r1 r5 - C28_8 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P28_8, C28_8]
    have hm : C28_8 r1 r5 r0 ≤ M28_8 r1 r5 :=
      CaseSplit.le_mxr (C28_8 r1 r5) 10 r0 (by omega)
    simp only [N28_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N28_9 r2 r3 ≤ ∑ t ∈ Finset.range n28, Wl28_9 r0 r1 r2 r3 r4 r5 t := by
    simp only [N28_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl28_9]
    exact CaseSplit.ind_nonneg _
  have hn10 : N28_10 r2 r4 ≤ ∑ t ∈ Finset.range n28, Wl28_10 r0 r1 r2 r3 r4 r5 t := by
    simp only [N28_10]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl28_10]
    exact CaseSplit.ind_nonneg _
  have hn11 : N28_11 r2 r5 ≤ ∑ t ∈ Finset.range n28, Wl28_11 r0 r1 r2 r3 r4 r5 t := by
    simp only [N28_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl28_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N28_12 r3 r4 ≤ ∑ t ∈ Finset.range n28, Wl28_12 r0 r1 r2 r3 r4 r5 t := by
    simp only [N28_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl28_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N28_13 r3 r5 ≤ ∑ t ∈ Finset.range n28, Wl28_13 r0 r1 r2 r3 r4 r5 t := by
    simp only [N28_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl28_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N28_14 r4 r5 ≤ ∑ t ∈ Finset.range n28, Wl28_14 r0 r1 r2 r3 r4 r5 t := by
    simp only [N28_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl28_14]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n28, (w28 t + 3) * Dg28 r0 r1 r2 r3 r4 r5 t = S28_0 r0 + S28_1 r1 + S28_2 r2 + S28_3 r3 + S28_4 r4 + S28_5 r5 := by
    simp only [S28_0, S28_1, S28_2, S28_3, S28_4, S28_5, Dg28, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n28, (w28 t + 3) * Dg28 r0 r1 r2 r3 r4 r5 t
      = (∑ t ∈ Finset.range n28, w28 t * Dg28 r0 r1 r2 r3 r4 r5 t)
        + 3 * (∑ t ∈ Finset.range n28, Dg28 r0 r1 r2 r3 r4 r5 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n28, w28 t)
      ≤ ∑ t ∈ Finset.range n28, w28 t * Dg28 r0 r1 r2 r3 r4 r5 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg28 r0 r1 r2 r3 r4 r5 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w28 t := wnn28 t (Finset.mem_range.mp ht)
    calc w28 t = w28 t * 1 := (mul_one _).symm
      _ ≤ w28 t * Dg28 r0 r1 r2 r3 r4 r5 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS28_0 r0 + aS28_1 r1 + aS28_2 r2 + aS28_3 r3 + aS28_4 r4 + aS28_5 r5) + (aP28_0 r0 r1 + aP28_1 r0 r2 + aP28_2 r0 r3 + aP28_3 r0 r4 + aP28_4 r0 r5 + aP28_5 r1 r2 + aP28_6 r1 r3 + aP28_7 r1 r4 + aP28_8 r1 r5 + aP28_9 r2 r3 + aP28_10 r2 r4 + aP28_11 r2 r5 + aP28_12 r3 r4 + aP28_13 r3 r5 + aP28_14 r4 r5) = (S28_0 r0 + S28_1 r1 + S28_2 r2 + S28_3 r3 + S28_4 r4 + S28_5 r5) - 3 * (N28_0 r0 r1 + N28_1 r0 r2 + N28_2 r0 r3 + N28_3 r0 r4 + N28_4 r0 r5 + N28_5 r1 r2 + N28_6 r1 r3 + N28_7 r1 r4 + N28_8 r1 r5 + N28_9 r2 r3 + N28_10 r2 r4 + N28_11 r2 r5 + N28_12 r3 r4 + N28_13 r3 r5 + N28_14 r4 r5) := by
    simp only [aS28_0, aS28_1, aS28_2, aS28_3, aS28_4, aS28_5, aP28_0, aP28_1, aP28_2, aP28_3, aP28_4, aP28_5, aP28_6, aP28_7, aP28_8, aP28_9, aP28_10, aP28_11, aP28_12, aP28_13, aP28_14, L28_0, L28_1, L28_2, L28_3, L28_4, L28_5]
    ring
  have bS0 : aS28_0 r0 ≤ MS28_0 := CaseSplit.le_mxr (aS28_0) 10 r0 (by omega)
  have bS1 : aS28_1 r1 ≤ MS28_1 := CaseSplit.le_mxr (aS28_1) 12 r1 (by omega)
  have bS2 : aS28_2 r2 ≤ MS28_2 := CaseSplit.le_mxr (aS28_2) 16 r2 (by omega)
  have bS3 : aS28_3 r3 ≤ MS28_3 := CaseSplit.le_mxr (aS28_3) 18 r3 (by omega)
  have bS4 : aS28_4 r4 ≤ MS28_4 := CaseSplit.le_mxr (aS28_4) 22 r4 (by omega)
  have bS5 : aS28_5 r5 ≤ MS28_5 := CaseSplit.le_mxr (aS28_5) 28 r5 (by omega)
  have bP0 : aP28_0 r0 r1 ≤ MP28_0 := CaseSplit.le_mxr2 (aP28_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP28_1 r0 r2 ≤ MP28_1 := CaseSplit.le_mxr2 (aP28_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP28_2 r0 r3 ≤ MP28_2 := CaseSplit.le_mxr2 (aP28_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP28_3 r0 r4 ≤ MP28_3 := CaseSplit.le_mxr2 (aP28_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP28_4 r0 r5 ≤ MP28_4 := CaseSplit.le_mxr2 (aP28_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP28_5 r1 r2 ≤ MP28_5 := CaseSplit.le_mxr2 (aP28_5) 12 16 r1 r2 (by omega) (by omega)
  have bP6 : aP28_6 r1 r3 ≤ MP28_6 := CaseSplit.le_mxr2 (aP28_6) 12 18 r1 r3 (by omega) (by omega)
  have bP7 : aP28_7 r1 r4 ≤ MP28_7 := CaseSplit.le_mxr2 (aP28_7) 12 22 r1 r4 (by omega) (by omega)
  have bP8 : aP28_8 r1 r5 ≤ MP28_8 := CaseSplit.le_mxr2 (aP28_8) 12 28 r1 r5 (by omega) (by omega)
  have bP9 : aP28_9 r2 r3 ≤ MP28_9 := CaseSplit.le_mxr2 (aP28_9) 16 18 r2 r3 (by omega) (by omega)
  have bP10 : aP28_10 r2 r4 ≤ MP28_10 := CaseSplit.le_mxr2 (aP28_10) 16 22 r2 r4 (by omega) (by omega)
  have bP11 : aP28_11 r2 r5 ≤ MP28_11 := CaseSplit.le_mxr2 (aP28_11) 16 28 r2 r5 (by omega) (by omega)
  have bP12 : aP28_12 r3 r4 ≤ MP28_12 := CaseSplit.le_mxr2 (aP28_12) 18 22 r3 r4 (by omega) (by omega)
  have bP13 : aP28_13 r3 r5 ≤ MP28_13 := CaseSplit.le_mxr2 (aP28_13) 18 28 r3 r5 (by omega) (by omega)
  have bP14 : aP28_14 r4 r5 ≤ MP28_14 := CaseSplit.le_mxr2 (aP28_14) 22 28 r4 r5 (by omega) (by omega)
  have hrhs : rhs28 = (∑ t ∈ Finset.range n28, w28 t) + 3 * (n28 : ℤ) := rfl
  have hc := cert28
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, bS0, bS1, bS2, bS3, bS4, bS5, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14]

end IncCert29
