/-
INCREMENT-WIDTH CERTIFICATE, step 23->29, case 10 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_23_29.json, which re-derives every number
from the primes alone).

Machine 29, INCREMENT width 49 = F_2(23) + s_min(29) = 39 + 10,
held gears [5, 7] at phases [1, 3].  Free gears [11, 13, 17, 19, 23, 29].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 5.
-/
import IncCert29B

namespace IncCert29

/-! ### case 10: held gears at phases [1, 3] -/

def p10 : List ℕ := [1, 2, 4, 6, 7, 9, 11, 14, 16, 21, 22, 27, 29, 32, 34, 36, 37, 39, 41, 42, 44, 46]
def q10 (t : ℕ) : ℕ := p10.getD t 0
def n10 : ℕ := 22
def yl10 : List ℤ := [0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 4, 0, 0, 4, 0, 0, 0, 0]
def w10 (t : ℕ) : ℤ := yl10.getD t 0
def ul10 : List ℤ := [(-1), (-2), (-1), (-2), (-1), (-2), (-2), (-2), 0, (-1), (-2), (-2), (-2), 2, 0, 1, 0, 1, (-5), 1, 2, 1, 0, 0, (-2), (-4), (-4), (-2), (-2), (-4), (-4), (-4), (-4), 0, 2, (-2), (-4), (-4), (-4), 0, 0, (-2), (-2), 0, 2, 0, 0, (-2), (-2), 0, 4, 0, (-1), 0, 0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 1, 0, (-1), (-1), (-1), 0, (-1), (-1), 1, 0, (-1), (-1), 1, 1, (-1), 0, 0, 0, (-9), 0, 0, 0, 1, 0, 0, (-14), 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, (-1), (-1), 0, 0, (-1), (-1), 0, (-1), (-1), 0, 0, (-2), (-2), 0, (-2), (-2), 0, (-2), (-2), (-2), 0, (-2), (-2), 0, 0, (-5), (-2), (-2), 0, (-2), 0, (-2), (-2), (-2), 0, 0, (-2), (-2), 0, (-2), 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 12, 18, 18, 18, 18, 17, 13, 13, 18, 18, 18, 14, 18, 18, 18, 18, 18, (-18), (-18), (-18), (-18), (-18), (-18), (-18), (-18), (-18), (-18), (-18), (-18), (-18), 19, 19, 19, 8, 6, 12, 19, 18, 19, 19, 11, 6, 17, 19, 19, 19, 11, 6, 17, (-19), (-19), (-19), (-25), (-19), (-19), (-19), (-19), (-19), (-19), (-19), (-19), (-19), 15, 15, 15, 14, 6, 10, 15, 1, 15, 15, 15, 15, 0, 15, 15, 6, 10, 0, 15, 6, 15, 15, 1, (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), 0, (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 16, 16, 14, 9, 7, 4, 14, 9, 7, 9, 9, 13, 12, 16, 11, 9, 16, 14, 16, 15, 16, 14, 12, 16, 11, 16, 13, 9, 6, (-8), 0, 6, 0, 6, 1, 0, 6, 6, 1, 0, 6, 6, (-8), 6, 6, 6, 6, 0, (-4), 5, 0]
def u10 (k : ℕ) : ℤ := ul10.getD k 0

def c10_0 (r t : ℕ) : Bool := gb11 r (q10 t)
def c10_1 (r t : ℕ) : Bool := gb13 r (q10 t)
def c10_2 (r t : ℕ) : Bool := gb17 r (q10 t)
def c10_3 (r t : ℕ) : Bool := gb19 r (q10 t)
def c10_4 (r t : ℕ) : Bool := gb23 r (q10 t)
def c10_5 (r t : ℕ) : Bool := gb29 r (q10 t)

def S10_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (w10 t + 5) * (if c10_0 r t then 1 else 0)
def S10_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (w10 t + 5) * (if c10_1 r t then 1 else 0)
def S10_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (w10 t + 5) * (if c10_2 r t then 1 else 0)
def S10_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (w10 t + 5) * (if c10_3 r t then 1 else 0)
def S10_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (w10 t + 5) * (if c10_4 r t then 1 else 0)
def S10_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (w10 t + 5) * (if c10_5 r t then 1 else 0)

def L10_0 (r : ℕ) : ℤ := u10 (13 + r) + u10 (41 + r) + u10 (71 + r) + u10 (105 + r) + u10 (145 + r)
def L10_1 (r : ℕ) : ℤ := u10 (0 + r) + u10 (173 + r) + u10 (205 + r) + u10 (241 + r) + u10 (283 + r)
def L10_2 (r : ℕ) : ℤ := u10 (24 + r) + u10 (156 + r) + u10 (315 + r) + u10 (355 + r) + u10 (401 + r)
def L10_3 (r : ℕ) : ℤ := u10 (52 + r) + u10 (186 + r) + u10 (296 + r) + u10 (441 + r) + u10 (489 + r)
def L10_4 (r : ℕ) : ℤ := u10 (82 + r) + u10 (218 + r) + u10 (332 + r) + u10 (418 + r) + u10 (537 + r)
def L10_5 (r : ℕ) : ℤ := u10 (116 + r) + u10 (254 + r) + u10 (372 + r) + u10 (460 + r) + u10 (508 + r)

def aS10_0 (r : ℕ) : ℤ := S10_0 r - L10_0 r
def MS10_0 : ℤ := CaseSplit.mxr (aS10_0) 10
def aS10_1 (r : ℕ) : ℤ := S10_1 r - L10_1 r
def MS10_1 : ℤ := CaseSplit.mxr (aS10_1) 12
def aS10_2 (r : ℕ) : ℤ := S10_2 r - L10_2 r
def MS10_2 : ℤ := CaseSplit.mxr (aS10_2) 16
def aS10_3 (r : ℕ) : ℤ := S10_3 r - L10_3 r
def MS10_3 : ℤ := CaseSplit.mxr (aS10_3) 18
def aS10_4 (r : ℕ) : ℤ := S10_4 r - L10_4 r
def MS10_4 : ℤ := CaseSplit.mxr (aS10_4) 22
def aS10_5 (r : ℕ) : ℤ := S10_5 r - L10_5 r
def MS10_5 : ℤ := CaseSplit.mxr (aS10_5) 28

def N10_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_0 ra t && c10_1 rb t then 1 else 0)
def aP10_0 (ra rb : ℕ) : ℤ := -(5) * N10_0 ra rb + u10 (0 + rb) + u10 (13 + ra)
def MP10_0 : ℤ := CaseSplit.mxr2 (aP10_0) 10 12
def N10_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_0 ra t && c10_2 rb t then 1 else 0)
def aP10_1 (ra rb : ℕ) : ℤ := -(5) * N10_1 ra rb + u10 (24 + rb) + u10 (41 + ra)
def MP10_1 : ℤ := CaseSplit.mxr2 (aP10_1) 10 16
def N10_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_0 ra t && c10_3 rb t then 1 else 0)
def aP10_2 (ra rb : ℕ) : ℤ := -(5) * N10_2 ra rb + u10 (52 + rb) + u10 (71 + ra)
def MP10_2 : ℤ := CaseSplit.mxr2 (aP10_2) 10 18
def N10_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_0 ra t && c10_4 rb t then 1 else 0)
def aP10_3 (ra rb : ℕ) : ℤ := -(5) * N10_3 ra rb + u10 (82 + rb) + u10 (105 + ra)
def MP10_3 : ℤ := CaseSplit.mxr2 (aP10_3) 10 22
def N10_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_0 ra t && c10_5 rb t then 1 else 0)
def aP10_4 (ra rb : ℕ) : ℤ := -(5) * N10_4 ra rb + u10 (116 + rb) + u10 (145 + ra)
def MP10_4 : ℤ := CaseSplit.mxr2 (aP10_4) 10 28
def P10_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_1 ra t && c10_2 rb t then 1 else 0)
def C10_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_1 ra t && c10_2 rb t && c10_0 s t then 1 else 0)
def M10_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C10_5 ra rb) 10
def E10_5 : List ℕ := [43, 49, 129, 135, 138, 144, 174, 180]
def N10_5 (ra rb : ℕ) : ℤ := if E10_5.contains (ra * 17 + rb) = true then P10_5 ra rb - M10_5 ra rb else 0
def aP10_5 (ra rb : ℕ) : ℤ := -(5) * N10_5 ra rb + u10 (156 + rb) + u10 (173 + ra)
def MP10_5 : ℤ := CaseSplit.mxr2 (aP10_5) 12 16
def P10_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_1 ra t && c10_3 rb t then 1 else 0)
def C10_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_1 ra t && c10_3 rb t && c10_0 s t then 1 else 0)
def M10_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C10_6 ra rb) 10
def E10_6 : List ℕ := [21, 27, 51, 58, 127, 134, 158, 164, 192, 198, 234, 240]
def N10_6 (ra rb : ℕ) : ℤ := if E10_6.contains (ra * 19 + rb) = true then P10_6 ra rb - M10_6 ra rb else 0
def aP10_6 (ra rb : ℕ) : ℤ := -(5) * N10_6 ra rb + u10 (186 + rb) + u10 (205 + ra)
def MP10_6 : ℤ := CaseSplit.mxr2 (aP10_6) 12 18
def P10_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_1 ra t && c10_4 rb t then 1 else 0)
def C10_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_1 ra t && c10_4 rb t && c10_0 s t then 1 else 0)
def M10_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C10_7 ra rb) 10
def E10_7 : List ℕ := []
def N10_7 (ra rb : ℕ) : ℤ := if E10_7.contains (ra * 23 + rb) = true then P10_7 ra rb - M10_7 ra rb else 0
def aP10_7 (ra rb : ℕ) : ℤ := -(5) * N10_7 ra rb + u10 (218 + rb) + u10 (241 + ra)
def MP10_7 : ℤ := CaseSplit.mxr2 (aP10_7) 12 22
def P10_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_1 ra t && c10_5 rb t then 1 else 0)
def C10_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_1 ra t && c10_5 rb t && c10_0 s t then 1 else 0)
def M10_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C10_8 ra rb) 10
def E10_8 : List ℕ := [22, 133, 249, 283]
def N10_8 (ra rb : ℕ) : ℤ := if E10_8.contains (ra * 29 + rb) = true then P10_8 ra rb - M10_8 ra rb else 0
def aP10_8 (ra rb : ℕ) : ℤ := -(5) * N10_8 ra rb + u10 (254 + rb) + u10 (283 + ra)
def MP10_8 : ℤ := CaseSplit.mxr2 (aP10_8) 12 28
def N10_9 (_ra _rb : ℕ) : ℤ := 0
def aP10_9 (ra rb : ℕ) : ℤ := -(5) * N10_9 ra rb + u10 (296 + rb) + u10 (315 + ra)
def MP10_9 : ℤ := CaseSplit.mxr2 (aP10_9) 16 18
def N10_10 (_ra _rb : ℕ) : ℤ := 0
def aP10_10 (ra rb : ℕ) : ℤ := -(5) * N10_10 ra rb + u10 (332 + rb) + u10 (355 + ra)
def MP10_10 : ℤ := CaseSplit.mxr2 (aP10_10) 16 22
def N10_11 (_ra _rb : ℕ) : ℤ := 0
def aP10_11 (ra rb : ℕ) : ℤ := -(5) * N10_11 ra rb + u10 (372 + rb) + u10 (401 + ra)
def MP10_11 : ℤ := CaseSplit.mxr2 (aP10_11) 16 28
def N10_12 (_ra _rb : ℕ) : ℤ := 0
def aP10_12 (ra rb : ℕ) : ℤ := -(5) * N10_12 ra rb + u10 (418 + rb) + u10 (441 + ra)
def MP10_12 : ℤ := CaseSplit.mxr2 (aP10_12) 18 22
def N10_13 (_ra _rb : ℕ) : ℤ := 0
def aP10_13 (ra rb : ℕ) : ℤ := -(5) * N10_13 ra rb + u10 (460 + rb) + u10 (489 + ra)
def MP10_13 : ℤ := CaseSplit.mxr2 (aP10_13) 18 28
def N10_14 (_ra _rb : ℕ) : ℤ := 0
def aP10_14 (ra rb : ℕ) : ℤ := -(5) * N10_14 ra rb + u10 (508 + rb) + u10 (537 + ra)
def MP10_14 : ℤ := CaseSplit.mxr2 (aP10_14) 22 28

def rhs10 : ℤ := (∑ t ∈ Finset.range n10, w10 t) + 5 * (n10 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn10 : ∀ t, t < n10 → (0 : ℤ) ≤ w10 t := by decide
theorem plt10 : ∀ t, t < n10 → q10 t < 49 := by decide
theorem pfree10_5 : ∀ t, t < n10 → gb5 1 (q10 t) = false := by decide
theorem pfree10_7 : ∀ t, t < n10 → gb7 3 (q10 t) = false := by decide
theorem MSv10_0 : MS10_0 = 24 := by decide +kernel
theorem MSv10_1 : MS10_1 = 75 := by decide +kernel
theorem MSv10_2 : MS10_2 = 1 := by decide +kernel
theorem MSv10_3 : MS10_3 = 0 := by decide +kernel
theorem MSv10_4 : MS10_4 = 0 := by decide +kernel
theorem MSv10_5 : MS10_5 = 0 := by decide +kernel
theorem MPv10_0 : MP10_0 = 0 := by decide +kernel
theorem MPv10_1 : MP10_1 = 0 := by decide +kernel
theorem MPv10_2 : MP10_2 = 0 := by decide +kernel
theorem MPv10_3 : MP10_3 = 0 := by decide +kernel
theorem MPv10_4 : MP10_4 = 0 := by decide +kernel
theorem MPv10_5 : MP10_5 = 0 := by decide +kernel
theorem MPv10_6 : MP10_6 = 0 := by decide +kernel
theorem MPv10_7 : MP10_7 = 0 := by decide +kernel
theorem MPv10_8 : MP10_8 = 0 := by decide +kernel
theorem MPv10_9 : MP10_9 = 0 := by decide +kernel
theorem MPv10_10 : MP10_10 = 0 := by decide +kernel
theorem MPv10_11 : MP10_11 = 0 := by decide +kernel
theorem MPv10_12 : MP10_12 = 0 := by decide +kernel
theorem MPv10_13 : MP10_13 = 0 := by decide +kernel
theorem MPv10_14 : MP10_14 = 22 := by decide +kernel
theorem rhsv10 : rhs10 = 123 := by decide +kernel

/-- **The case-10 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/5.
    (Scaled by the common denominator 5: 122 < 123.) -/
theorem cert10 : MS10_0 + MS10_1 + MS10_2 + MS10_3 + MS10_4 + MS10_5 + MP10_0 + MP10_1 + MP10_2 + MP10_3 + MP10_4 + MP10_5 + MP10_6 + MP10_7 + MP10_8 + MP10_9 + MP10_10 + MP10_11 + MP10_12 + MP10_13 + MP10_14 < rhs10 := by
  rw [MSv10_0, MSv10_1, MSv10_2, MSv10_3, MSv10_4, MSv10_5, MPv10_0, MPv10_1, MPv10_2, MPv10_3, MPv10_4, MPv10_5, MPv10_6, MPv10_7, MPv10_8, MPv10_9, MPv10_10, MPv10_11, MPv10_12, MPv10_13, MPv10_14, rhsv10]
  decide

def Dg10 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := (if c10_0 r0 t then 1 else 0) + (if c10_1 r1 t then 1 else 0) + (if c10_2 r2 t then 1 else 0) + (if c10_3 r3 t then 1 else 0) + (if c10_4 r4 t then 1 else 0) + (if c10_5 r5 t then 1 else 0)
def Wl10_0 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c10_0 r0 t && c10_1 r1 t then 1 else 0
def Wl10_1 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c10_0 r0 t && c10_2 r2 t then 1 else 0
def Wl10_2 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c10_0 r0 t && c10_3 r3 t then 1 else 0
def Wl10_3 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c10_0 r0 t && c10_4 r4 t then 1 else 0
def Wl10_4 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c10_0 r0 t && c10_5 r5 t then 1 else 0
def Wl10_5 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c10_0 r0 t && c10_1 r1 t && c10_2 r2 t then 1 else 0
def Wl10_6 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c10_0 r0 t && c10_1 r1 t && c10_3 r3 t then 1 else 0
def Wl10_7 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c10_0 r0 t && c10_1 r1 t && c10_4 r4 t then 1 else 0
def Wl10_8 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c10_0 r0 t && c10_1 r1 t && c10_5 r5 t then 1 else 0
def Wl10_9 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c10_0 r0 t && !c10_1 r1 t && c10_2 r2 t && c10_3 r3 t then 1 else 0
def Wl10_10 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c10_0 r0 t && !c10_1 r1 t && c10_2 r2 t && c10_4 r4 t then 1 else 0
def Wl10_11 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c10_0 r0 t && !c10_1 r1 t && c10_2 r2 t && c10_5 r5 t then 1 else 0
def Wl10_12 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c10_0 r0 t && !c10_1 r1 t && !c10_2 r2 t && c10_3 r3 t && c10_4 r4 t then 1 else 0
def Wl10_13 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c10_0 r0 t && !c10_1 r1 t && !c10_2 r2 t && c10_3 r3 t && c10_5 r5 t then 1 else 0
def Wl10_14 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c10_0 r0 t && !c10_1 r1 t && !c10_2 r2 t && !c10_3 r3 t && c10_4 r4 t && c10_5 r5 t then 1 else 0

/-- **No configuration blocks the whole window in case 10.** -/
theorem nocov10 {r0 r1 r2 r3 r4 r5 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29)
    (hcov : ∀ t, t < n10 → (c10_0 r0 t || c10_1 r1 t || c10_2 r2 t || c10_3 r3 t || c10_4 r4 t || c10_5 r5 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n10, (1 : ℤ) + (Wl10_0 r0 r1 r2 r3 r4 r5 t + Wl10_1 r0 r1 r2 r3 r4 r5 t + Wl10_2 r0 r1 r2 r3 r4 r5 t + Wl10_3 r0 r1 r2 r3 r4 r5 t + Wl10_4 r0 r1 r2 r3 r4 r5 t + Wl10_5 r0 r1 r2 r3 r4 r5 t + Wl10_6 r0 r1 r2 r3 r4 r5 t + Wl10_7 r0 r1 r2 r3 r4 r5 t + Wl10_8 r0 r1 r2 r3 r4 r5 t + Wl10_9 r0 r1 r2 r3 r4 r5 t + Wl10_10 r0 r1 r2 r3 r4 r5 t + Wl10_11 r0 r1 r2 r3 r4 r5 t + Wl10_12 r0 r1 r2 r3 r4 r5 t + Wl10_13 r0 r1 r2 r3 r4 r5 t + Wl10_14 r0 r1 r2 r3 r4 r5 t) ≤ Dg10 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Wl10_0, Wl10_1, Wl10_2, Wl10_3, Wl10_4, Wl10_5, Wl10_6, Wl10_7, Wl10_8, Wl10_9, Wl10_10, Wl10_11, Wl10_12, Wl10_13, Wl10_14, Dg10]
    exact CaseSplit.lowest6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n10, (1 : ℤ) ≤ Dg10 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Dg10]
    exact CaseSplit.degpos6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n10 : ℤ) + ((∑ t ∈ Finset.range n10, Wl10_0 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n10, Wl10_1 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n10, Wl10_2 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n10, Wl10_3 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n10, Wl10_4 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n10, Wl10_5 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n10, Wl10_6 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n10, Wl10_7 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n10, Wl10_8 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n10, Wl10_9 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n10, Wl10_10 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n10, Wl10_11 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n10, Wl10_12 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n10, Wl10_13 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n10, Wl10_14 r0 r1 r2 r3 r4 r5 t)) ≤ ∑ t ∈ Finset.range n10, Dg10 r0 r1 r2 r3 r4 r5 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N10_0 r0 r1 ≤ ∑ t ∈ Finset.range n10, Wl10_0 r0 r1 r2 r3 r4 r5 t := by
    simp only [N10_0, Wl10_0, le_refl]
  have hn1 : N10_1 r0 r2 ≤ ∑ t ∈ Finset.range n10, Wl10_1 r0 r1 r2 r3 r4 r5 t := by
    simp only [N10_1, Wl10_1, le_refl]
  have hn2 : N10_2 r0 r3 ≤ ∑ t ∈ Finset.range n10, Wl10_2 r0 r1 r2 r3 r4 r5 t := by
    simp only [N10_2, Wl10_2, le_refl]
  have hn3 : N10_3 r0 r4 ≤ ∑ t ∈ Finset.range n10, Wl10_3 r0 r1 r2 r3 r4 r5 t := by
    simp only [N10_3, Wl10_3, le_refl]
  have hn4 : N10_4 r0 r5 ≤ ∑ t ∈ Finset.range n10, Wl10_4 r0 r1 r2 r3 r4 r5 t := by
    simp only [N10_4, Wl10_4, le_refl]
  have hn5 : N10_5 r1 r2 ≤ ∑ t ∈ Finset.range n10, Wl10_5 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n10, Wl10_5 r0 r1 r2 r3 r4 r5 t
        = (if c10_1 r1 t && c10_2 r2 t then (1:ℤ) else 0)
          - (if c10_1 r1 t && c10_2 r2 t && c10_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl10_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n10, Wl10_5 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl10_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n10, Wl10_5 r0 r1 r2 r3 r4 r5 t
        = P10_5 r1 r2 - C10_5 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P10_5, C10_5]
    have hm : C10_5 r1 r2 r0 ≤ M10_5 r1 r2 :=
      CaseSplit.le_mxr (C10_5 r1 r2) 10 r0 (by omega)
    simp only [N10_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N10_6 r1 r3 ≤ ∑ t ∈ Finset.range n10, Wl10_6 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n10, Wl10_6 r0 r1 r2 r3 r4 r5 t
        = (if c10_1 r1 t && c10_3 r3 t then (1:ℤ) else 0)
          - (if c10_1 r1 t && c10_3 r3 t && c10_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl10_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n10, Wl10_6 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl10_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n10, Wl10_6 r0 r1 r2 r3 r4 r5 t
        = P10_6 r1 r3 - C10_6 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P10_6, C10_6]
    have hm : C10_6 r1 r3 r0 ≤ M10_6 r1 r3 :=
      CaseSplit.le_mxr (C10_6 r1 r3) 10 r0 (by omega)
    simp only [N10_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N10_7 r1 r4 ≤ ∑ t ∈ Finset.range n10, Wl10_7 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n10, Wl10_7 r0 r1 r2 r3 r4 r5 t
        = (if c10_1 r1 t && c10_4 r4 t then (1:ℤ) else 0)
          - (if c10_1 r1 t && c10_4 r4 t && c10_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl10_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n10, Wl10_7 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl10_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n10, Wl10_7 r0 r1 r2 r3 r4 r5 t
        = P10_7 r1 r4 - C10_7 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P10_7, C10_7]
    have hm : C10_7 r1 r4 r0 ≤ M10_7 r1 r4 :=
      CaseSplit.le_mxr (C10_7 r1 r4) 10 r0 (by omega)
    simp only [N10_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N10_8 r1 r5 ≤ ∑ t ∈ Finset.range n10, Wl10_8 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n10, Wl10_8 r0 r1 r2 r3 r4 r5 t
        = (if c10_1 r1 t && c10_5 r5 t then (1:ℤ) else 0)
          - (if c10_1 r1 t && c10_5 r5 t && c10_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl10_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n10, Wl10_8 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl10_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n10, Wl10_8 r0 r1 r2 r3 r4 r5 t
        = P10_8 r1 r5 - C10_8 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P10_8, C10_8]
    have hm : C10_8 r1 r5 r0 ≤ M10_8 r1 r5 :=
      CaseSplit.le_mxr (C10_8 r1 r5) 10 r0 (by omega)
    simp only [N10_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N10_9 r2 r3 ≤ ∑ t ∈ Finset.range n10, Wl10_9 r0 r1 r2 r3 r4 r5 t := by
    simp only [N10_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl10_9]
    exact CaseSplit.ind_nonneg _
  have hn10 : N10_10 r2 r4 ≤ ∑ t ∈ Finset.range n10, Wl10_10 r0 r1 r2 r3 r4 r5 t := by
    simp only [N10_10]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl10_10]
    exact CaseSplit.ind_nonneg _
  have hn11 : N10_11 r2 r5 ≤ ∑ t ∈ Finset.range n10, Wl10_11 r0 r1 r2 r3 r4 r5 t := by
    simp only [N10_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl10_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N10_12 r3 r4 ≤ ∑ t ∈ Finset.range n10, Wl10_12 r0 r1 r2 r3 r4 r5 t := by
    simp only [N10_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl10_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N10_13 r3 r5 ≤ ∑ t ∈ Finset.range n10, Wl10_13 r0 r1 r2 r3 r4 r5 t := by
    simp only [N10_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl10_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N10_14 r4 r5 ≤ ∑ t ∈ Finset.range n10, Wl10_14 r0 r1 r2 r3 r4 r5 t := by
    simp only [N10_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl10_14]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n10, (w10 t + 5) * Dg10 r0 r1 r2 r3 r4 r5 t = S10_0 r0 + S10_1 r1 + S10_2 r2 + S10_3 r3 + S10_4 r4 + S10_5 r5 := by
    simp only [S10_0, S10_1, S10_2, S10_3, S10_4, S10_5, Dg10, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n10, (w10 t + 5) * Dg10 r0 r1 r2 r3 r4 r5 t
      = (∑ t ∈ Finset.range n10, w10 t * Dg10 r0 r1 r2 r3 r4 r5 t)
        + 5 * (∑ t ∈ Finset.range n10, Dg10 r0 r1 r2 r3 r4 r5 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n10, w10 t)
      ≤ ∑ t ∈ Finset.range n10, w10 t * Dg10 r0 r1 r2 r3 r4 r5 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg10 r0 r1 r2 r3 r4 r5 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w10 t := wnn10 t (Finset.mem_range.mp ht)
    calc w10 t = w10 t * 1 := (mul_one _).symm
      _ ≤ w10 t * Dg10 r0 r1 r2 r3 r4 r5 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS10_0 r0 + aS10_1 r1 + aS10_2 r2 + aS10_3 r3 + aS10_4 r4 + aS10_5 r5) + (aP10_0 r0 r1 + aP10_1 r0 r2 + aP10_2 r0 r3 + aP10_3 r0 r4 + aP10_4 r0 r5 + aP10_5 r1 r2 + aP10_6 r1 r3 + aP10_7 r1 r4 + aP10_8 r1 r5 + aP10_9 r2 r3 + aP10_10 r2 r4 + aP10_11 r2 r5 + aP10_12 r3 r4 + aP10_13 r3 r5 + aP10_14 r4 r5) = (S10_0 r0 + S10_1 r1 + S10_2 r2 + S10_3 r3 + S10_4 r4 + S10_5 r5) - 5 * (N10_0 r0 r1 + N10_1 r0 r2 + N10_2 r0 r3 + N10_3 r0 r4 + N10_4 r0 r5 + N10_5 r1 r2 + N10_6 r1 r3 + N10_7 r1 r4 + N10_8 r1 r5 + N10_9 r2 r3 + N10_10 r2 r4 + N10_11 r2 r5 + N10_12 r3 r4 + N10_13 r3 r5 + N10_14 r4 r5) := by
    simp only [aS10_0, aS10_1, aS10_2, aS10_3, aS10_4, aS10_5, aP10_0, aP10_1, aP10_2, aP10_3, aP10_4, aP10_5, aP10_6, aP10_7, aP10_8, aP10_9, aP10_10, aP10_11, aP10_12, aP10_13, aP10_14, L10_0, L10_1, L10_2, L10_3, L10_4, L10_5]
    ring
  have bS0 : aS10_0 r0 ≤ MS10_0 := CaseSplit.le_mxr (aS10_0) 10 r0 (by omega)
  have bS1 : aS10_1 r1 ≤ MS10_1 := CaseSplit.le_mxr (aS10_1) 12 r1 (by omega)
  have bS2 : aS10_2 r2 ≤ MS10_2 := CaseSplit.le_mxr (aS10_2) 16 r2 (by omega)
  have bS3 : aS10_3 r3 ≤ MS10_3 := CaseSplit.le_mxr (aS10_3) 18 r3 (by omega)
  have bS4 : aS10_4 r4 ≤ MS10_4 := CaseSplit.le_mxr (aS10_4) 22 r4 (by omega)
  have bS5 : aS10_5 r5 ≤ MS10_5 := CaseSplit.le_mxr (aS10_5) 28 r5 (by omega)
  have bP0 : aP10_0 r0 r1 ≤ MP10_0 := CaseSplit.le_mxr2 (aP10_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP10_1 r0 r2 ≤ MP10_1 := CaseSplit.le_mxr2 (aP10_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP10_2 r0 r3 ≤ MP10_2 := CaseSplit.le_mxr2 (aP10_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP10_3 r0 r4 ≤ MP10_3 := CaseSplit.le_mxr2 (aP10_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP10_4 r0 r5 ≤ MP10_4 := CaseSplit.le_mxr2 (aP10_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP10_5 r1 r2 ≤ MP10_5 := CaseSplit.le_mxr2 (aP10_5) 12 16 r1 r2 (by omega) (by omega)
  have bP6 : aP10_6 r1 r3 ≤ MP10_6 := CaseSplit.le_mxr2 (aP10_6) 12 18 r1 r3 (by omega) (by omega)
  have bP7 : aP10_7 r1 r4 ≤ MP10_7 := CaseSplit.le_mxr2 (aP10_7) 12 22 r1 r4 (by omega) (by omega)
  have bP8 : aP10_8 r1 r5 ≤ MP10_8 := CaseSplit.le_mxr2 (aP10_8) 12 28 r1 r5 (by omega) (by omega)
  have bP9 : aP10_9 r2 r3 ≤ MP10_9 := CaseSplit.le_mxr2 (aP10_9) 16 18 r2 r3 (by omega) (by omega)
  have bP10 : aP10_10 r2 r4 ≤ MP10_10 := CaseSplit.le_mxr2 (aP10_10) 16 22 r2 r4 (by omega) (by omega)
  have bP11 : aP10_11 r2 r5 ≤ MP10_11 := CaseSplit.le_mxr2 (aP10_11) 16 28 r2 r5 (by omega) (by omega)
  have bP12 : aP10_12 r3 r4 ≤ MP10_12 := CaseSplit.le_mxr2 (aP10_12) 18 22 r3 r4 (by omega) (by omega)
  have bP13 : aP10_13 r3 r5 ≤ MP10_13 := CaseSplit.le_mxr2 (aP10_13) 18 28 r3 r5 (by omega) (by omega)
  have bP14 : aP10_14 r4 r5 ≤ MP10_14 := CaseSplit.le_mxr2 (aP10_14) 22 28 r4 r5 (by omega) (by omega)
  have hrhs : rhs10 = (∑ t ∈ Finset.range n10, w10 t) + 5 * (n10 : ℤ) := rfl
  have hc := cert10
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, bS0, bS1, bS2, bS3, bS4, bS5, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14]

end IncCert29
