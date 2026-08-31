/-
INCREMENT-WIDTH CERTIFICATE, step 23->29, case 33 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_23_29.json, which re-derives every number
from the primes alone).

Machine 29, INCREMENT width 49 = F_2(23) + s_min(29) = 39 + 10,
held gears [5, 7] at phases [4, 5].  Free gears [11, 13, 17, 19, 23, 29].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 7.
-/
import IncCert29B

namespace IncCert29

/-! ### case 33: held gears at phases [4, 5] -/

def p33 : List ℕ := [4, 6, 9, 11, 13, 14, 16, 18, 19, 21, 23, 26, 28, 33, 34, 39, 41, 44, 46, 48]
def q33 (t : ℕ) : ℕ := p33.getD t 0
def n33 : ℕ := 20
def yl33 : List ℤ := [1, 2, 3, 3, 2, 5, 7, 6, 4, 6, 5, 2, 3, 0, 1, 0, 0, 0, 1, 0]
def w33 (t : ℕ) : ℤ := yl33.getD t 0
def ul33 : List ℤ := [0, (-1), 0, (-1), (-1), 0, (-1), (-4), (-1), 0, (-1), 0, (-1), 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, (-2), (-2), (-2), 0, (-1), (-2), (-2), (-2), 0, (-2), (-2), (-2), (-2), (-2), 0, (-1), (-2), 0, 1, 0, 1, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, (-9), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, (-1), (-3), (-3), (-3), 0, (-3), 0, 0, (-3), (-3), (-3), (-3), (-3), (-1), 0, 0, (-3), (-3), (-3), 0, (-3), 0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 13, 16, 15, 16, 16, 12, 16, 16, 16, 16, 11, 8, 16, 16, 16, 14, (-21), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), 13, 12, 14, 11, 9, 15, 13, 15, 15, 15, 15, 6, 15, 15, 12, 7, 14, 9, 12, (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), 8, 8, (-4), 8, 7, 3, 8, 3, 8, 8, 1, 8, 3, 8, 8, 0, 8, 0, 8, 7, 0, 8, 8, (-8), (-8), (-8), (-8), (-8), (-8), (-8), (-8), (-8), (-8), (-8), (-8), (-8), 0, 0, 0, 0, 0, 0, 0, (-11), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-5), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-7), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 14, 0, 8, 0, 9, 14, 14, 14, 3, 8, 14, 14, 14, 3, 14, 9, 14, 14, 6, 14, 5, 3, 14, 3, 11, 0, 4, 5, 2, 7, 7, 7, 7, 7, 7, 0, 7, 7, 7, 7, 0, 7, 5, 4, 7, 6, 0, 7, 3, 7, 0]
def u33 (k : ℕ) : ℤ := ul33.getD k 0

def c33_0 (r t : ℕ) : Bool := gb11 r (q33 t)
def c33_1 (r t : ℕ) : Bool := gb13 r (q33 t)
def c33_2 (r t : ℕ) : Bool := gb17 r (q33 t)
def c33_3 (r t : ℕ) : Bool := gb19 r (q33 t)
def c33_4 (r t : ℕ) : Bool := gb23 r (q33 t)
def c33_5 (r t : ℕ) : Bool := gb29 r (q33 t)

def S33_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (w33 t + 3) * (if c33_0 r t then 1 else 0)
def S33_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (w33 t + 3) * (if c33_1 r t then 1 else 0)
def S33_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (w33 t + 3) * (if c33_2 r t then 1 else 0)
def S33_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (w33 t + 3) * (if c33_3 r t then 1 else 0)
def S33_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (w33 t + 3) * (if c33_4 r t then 1 else 0)
def S33_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (w33 t + 3) * (if c33_5 r t then 1 else 0)

def L33_0 (r : ℕ) : ℤ := u33 (13 + r) + u33 (41 + r) + u33 (71 + r) + u33 (105 + r) + u33 (145 + r)
def L33_1 (r : ℕ) : ℤ := u33 (0 + r) + u33 (173 + r) + u33 (205 + r) + u33 (241 + r) + u33 (283 + r)
def L33_2 (r : ℕ) : ℤ := u33 (24 + r) + u33 (156 + r) + u33 (315 + r) + u33 (355 + r) + u33 (401 + r)
def L33_3 (r : ℕ) : ℤ := u33 (52 + r) + u33 (186 + r) + u33 (296 + r) + u33 (441 + r) + u33 (489 + r)
def L33_4 (r : ℕ) : ℤ := u33 (82 + r) + u33 (218 + r) + u33 (332 + r) + u33 (418 + r) + u33 (537 + r)
def L33_5 (r : ℕ) : ℤ := u33 (116 + r) + u33 (254 + r) + u33 (372 + r) + u33 (460 + r) + u33 (508 + r)

def aS33_0 (r : ℕ) : ℤ := S33_0 r - L33_0 r
def MS33_0 : ℤ := CaseSplit.mxr (aS33_0) 10
def aS33_1 (r : ℕ) : ℤ := S33_1 r - L33_1 r
def MS33_1 : ℤ := CaseSplit.mxr (aS33_1) 12
def aS33_2 (r : ℕ) : ℤ := S33_2 r - L33_2 r
def MS33_2 : ℤ := CaseSplit.mxr (aS33_2) 16
def aS33_3 (r : ℕ) : ℤ := S33_3 r - L33_3 r
def MS33_3 : ℤ := CaseSplit.mxr (aS33_3) 18
def aS33_4 (r : ℕ) : ℤ := S33_4 r - L33_4 r
def MS33_4 : ℤ := CaseSplit.mxr (aS33_4) 22
def aS33_5 (r : ℕ) : ℤ := S33_5 r - L33_5 r
def MS33_5 : ℤ := CaseSplit.mxr (aS33_5) 28

def N33_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (if c33_0 ra t && c33_1 rb t then 1 else 0)
def aP33_0 (ra rb : ℕ) : ℤ := -(3) * N33_0 ra rb + u33 (0 + rb) + u33 (13 + ra)
def MP33_0 : ℤ := CaseSplit.mxr2 (aP33_0) 10 12
def N33_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (if c33_0 ra t && c33_2 rb t then 1 else 0)
def aP33_1 (ra rb : ℕ) : ℤ := -(3) * N33_1 ra rb + u33 (24 + rb) + u33 (41 + ra)
def MP33_1 : ℤ := CaseSplit.mxr2 (aP33_1) 10 16
def N33_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (if c33_0 ra t && c33_3 rb t then 1 else 0)
def aP33_2 (ra rb : ℕ) : ℤ := -(3) * N33_2 ra rb + u33 (52 + rb) + u33 (71 + ra)
def MP33_2 : ℤ := CaseSplit.mxr2 (aP33_2) 10 18
def N33_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (if c33_0 ra t && c33_4 rb t then 1 else 0)
def aP33_3 (ra rb : ℕ) : ℤ := -(3) * N33_3 ra rb + u33 (82 + rb) + u33 (105 + ra)
def MP33_3 : ℤ := CaseSplit.mxr2 (aP33_3) 10 22
def N33_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (if c33_0 ra t && c33_5 rb t then 1 else 0)
def aP33_4 (ra rb : ℕ) : ℤ := -(3) * N33_4 ra rb + u33 (116 + rb) + u33 (145 + ra)
def MP33_4 : ℤ := CaseSplit.mxr2 (aP33_4) 10 28
def P33_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (if c33_1 ra t && c33_2 rb t then 1 else 0)
def C33_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (if c33_1 ra t && c33_2 rb t && c33_0 s t then 1 else 0)
def M33_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C33_5 ra rb) 10
def E33_5 : List ℕ := [3, 9, 39, 45, 93, 99, 129, 135, 140, 151]
def N33_5 (ra rb : ℕ) : ℤ := if E33_5.contains (ra * 17 + rb) = true then P33_5 ra rb - M33_5 ra rb else 0
def aP33_5 (ra rb : ℕ) : ℤ := -(3) * N33_5 ra rb + u33 (156 + rb) + u33 (173 + ra)
def MP33_5 : ℤ := CaseSplit.mxr2 (aP33_5) 12 16
def P33_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (if c33_1 ra t && c33_3 rb t then 1 else 0)
def C33_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (if c33_1 ra t && c33_3 rb t && c33_0 s t then 1 else 0)
def M33_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C33_6 ra rb) 10
def E33_6 : List ℕ := [13, 47, 53, 58, 84, 111, 134, 160, 184, 187, 218, 224]
def N33_6 (ra rb : ℕ) : ℤ := if E33_6.contains (ra * 19 + rb) = true then P33_6 ra rb - M33_6 ra rb else 0
def aP33_6 (ra rb : ℕ) : ℤ := -(3) * N33_6 ra rb + u33 (186 + rb) + u33 (205 + ra)
def MP33_6 : ℤ := CaseSplit.mxr2 (aP33_6) 12 18
def P33_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (if c33_1 ra t && c33_4 rb t then 1 else 0)
def C33_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (if c33_1 ra t && c33_4 rb t && c33_0 s t then 1 else 0)
def M33_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C33_7 ra rb) 10
def E33_7 : List ℕ := []
def N33_7 (ra rb : ℕ) : ℤ := if E33_7.contains (ra * 23 + rb) = true then P33_7 ra rb - M33_7 ra rb else 0
def aP33_7 (ra rb : ℕ) : ℤ := -(3) * N33_7 ra rb + u33 (218 + rb) + u33 (241 + ra)
def MP33_7 : ℤ := CaseSplit.mxr2 (aP33_7) 12 22
def P33_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (if c33_1 ra t && c33_5 rb t then 1 else 0)
def C33_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (if c33_1 ra t && c33_5 rb t && c33_0 s t then 1 else 0)
def M33_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C33_8 ra rb) 10
def E33_8 : List ℕ := [73, 189]
def N33_8 (ra rb : ℕ) : ℤ := if E33_8.contains (ra * 29 + rb) = true then P33_8 ra rb - M33_8 ra rb else 0
def aP33_8 (ra rb : ℕ) : ℤ := -(3) * N33_8 ra rb + u33 (254 + rb) + u33 (283 + ra)
def MP33_8 : ℤ := CaseSplit.mxr2 (aP33_8) 12 28
def N33_9 (_ra _rb : ℕ) : ℤ := 0
def aP33_9 (ra rb : ℕ) : ℤ := -(3) * N33_9 ra rb + u33 (296 + rb) + u33 (315 + ra)
def MP33_9 : ℤ := CaseSplit.mxr2 (aP33_9) 16 18
def N33_10 (_ra _rb : ℕ) : ℤ := 0
def aP33_10 (ra rb : ℕ) : ℤ := -(3) * N33_10 ra rb + u33 (332 + rb) + u33 (355 + ra)
def MP33_10 : ℤ := CaseSplit.mxr2 (aP33_10) 16 22
def N33_11 (_ra _rb : ℕ) : ℤ := 0
def aP33_11 (ra rb : ℕ) : ℤ := -(3) * N33_11 ra rb + u33 (372 + rb) + u33 (401 + ra)
def MP33_11 : ℤ := CaseSplit.mxr2 (aP33_11) 16 28
def N33_12 (_ra _rb : ℕ) : ℤ := 0
def aP33_12 (ra rb : ℕ) : ℤ := -(3) * N33_12 ra rb + u33 (418 + rb) + u33 (441 + ra)
def MP33_12 : ℤ := CaseSplit.mxr2 (aP33_12) 18 22
def N33_13 (_ra _rb : ℕ) : ℤ := 0
def aP33_13 (ra rb : ℕ) : ℤ := -(3) * N33_13 ra rb + u33 (460 + rb) + u33 (489 + ra)
def MP33_13 : ℤ := CaseSplit.mxr2 (aP33_13) 18 28
def N33_14 (_ra _rb : ℕ) : ℤ := 0
def aP33_14 (ra rb : ℕ) : ℤ := -(3) * N33_14 ra rb + u33 (508 + rb) + u33 (537 + ra)
def MP33_14 : ℤ := CaseSplit.mxr2 (aP33_14) 22 28

def rhs33 : ℤ := (∑ t ∈ Finset.range n33, w33 t) + 3 * (n33 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn33 : ∀ t, t < n33 → (0 : ℤ) ≤ w33 t := by decide
theorem plt33 : ∀ t, t < n33 → q33 t < 49 := by decide
theorem pfree33_5 : ∀ t, t < n33 → gb5 4 (q33 t) = false := by decide
theorem pfree33_7 : ∀ t, t < n33 → gb7 5 (q33 t) = false := by decide
theorem MSv33_0 : MS33_0 = 21 := by decide +kernel
theorem MSv33_1 : MS33_1 = 61 := by decide +kernel
theorem MSv33_2 : MS33_2 = 2 := by decide +kernel
theorem MSv33_3 : MS33_3 = 1 := by decide +kernel
theorem MSv33_4 : MS33_4 = 2 := by decide +kernel
theorem MSv33_5 : MS33_5 = 2 := by decide +kernel
theorem MPv33_0 : MP33_0 = 0 := by decide +kernel
theorem MPv33_1 : MP33_1 = 0 := by decide +kernel
theorem MPv33_2 : MP33_2 = 0 := by decide +kernel
theorem MPv33_3 : MP33_3 = 0 := by decide +kernel
theorem MPv33_4 : MP33_4 = 0 := by decide +kernel
theorem MPv33_5 : MP33_5 = 0 := by decide +kernel
theorem MPv33_6 : MP33_6 = 0 := by decide +kernel
theorem MPv33_7 : MP33_7 = 0 := by decide +kernel
theorem MPv33_8 : MP33_8 = 0 := by decide +kernel
theorem MPv33_9 : MP33_9 = 0 := by decide +kernel
theorem MPv33_10 : MP33_10 = 0 := by decide +kernel
theorem MPv33_11 : MP33_11 = 0 := by decide +kernel
theorem MPv33_12 : MP33_12 = 0 := by decide +kernel
theorem MPv33_13 : MP33_13 = 0 := by decide +kernel
theorem MPv33_14 : MP33_14 = 21 := by decide +kernel
theorem rhsv33 : rhs33 = 111 := by decide +kernel

/-- **The case-33 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/7.
    (Scaled by the common denominator 7: 110 < 111.) -/
theorem cert33 : MS33_0 + MS33_1 + MS33_2 + MS33_3 + MS33_4 + MS33_5 + MP33_0 + MP33_1 + MP33_2 + MP33_3 + MP33_4 + MP33_5 + MP33_6 + MP33_7 + MP33_8 + MP33_9 + MP33_10 + MP33_11 + MP33_12 + MP33_13 + MP33_14 < rhs33 := by
  rw [MSv33_0, MSv33_1, MSv33_2, MSv33_3, MSv33_4, MSv33_5, MPv33_0, MPv33_1, MPv33_2, MPv33_3, MPv33_4, MPv33_5, MPv33_6, MPv33_7, MPv33_8, MPv33_9, MPv33_10, MPv33_11, MPv33_12, MPv33_13, MPv33_14, rhsv33]
  decide

def Dg33 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := (if c33_0 r0 t then 1 else 0) + (if c33_1 r1 t then 1 else 0) + (if c33_2 r2 t then 1 else 0) + (if c33_3 r3 t then 1 else 0) + (if c33_4 r4 t then 1 else 0) + (if c33_5 r5 t then 1 else 0)
def Wl33_0 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c33_0 r0 t && c33_1 r1 t then 1 else 0
def Wl33_1 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c33_0 r0 t && c33_2 r2 t then 1 else 0
def Wl33_2 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c33_0 r0 t && c33_3 r3 t then 1 else 0
def Wl33_3 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c33_0 r0 t && c33_4 r4 t then 1 else 0
def Wl33_4 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c33_0 r0 t && c33_5 r5 t then 1 else 0
def Wl33_5 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c33_0 r0 t && c33_1 r1 t && c33_2 r2 t then 1 else 0
def Wl33_6 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c33_0 r0 t && c33_1 r1 t && c33_3 r3 t then 1 else 0
def Wl33_7 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c33_0 r0 t && c33_1 r1 t && c33_4 r4 t then 1 else 0
def Wl33_8 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c33_0 r0 t && c33_1 r1 t && c33_5 r5 t then 1 else 0
def Wl33_9 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c33_0 r0 t && !c33_1 r1 t && c33_2 r2 t && c33_3 r3 t then 1 else 0
def Wl33_10 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c33_0 r0 t && !c33_1 r1 t && c33_2 r2 t && c33_4 r4 t then 1 else 0
def Wl33_11 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c33_0 r0 t && !c33_1 r1 t && c33_2 r2 t && c33_5 r5 t then 1 else 0
def Wl33_12 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c33_0 r0 t && !c33_1 r1 t && !c33_2 r2 t && c33_3 r3 t && c33_4 r4 t then 1 else 0
def Wl33_13 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c33_0 r0 t && !c33_1 r1 t && !c33_2 r2 t && c33_3 r3 t && c33_5 r5 t then 1 else 0
def Wl33_14 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c33_0 r0 t && !c33_1 r1 t && !c33_2 r2 t && !c33_3 r3 t && c33_4 r4 t && c33_5 r5 t then 1 else 0

/-- **No configuration blocks the whole window in case 33.** -/
theorem nocov33 {r0 r1 r2 r3 r4 r5 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29)
    (hcov : ∀ t, t < n33 → (c33_0 r0 t || c33_1 r1 t || c33_2 r2 t || c33_3 r3 t || c33_4 r4 t || c33_5 r5 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n33, (1 : ℤ) + (Wl33_0 r0 r1 r2 r3 r4 r5 t + Wl33_1 r0 r1 r2 r3 r4 r5 t + Wl33_2 r0 r1 r2 r3 r4 r5 t + Wl33_3 r0 r1 r2 r3 r4 r5 t + Wl33_4 r0 r1 r2 r3 r4 r5 t + Wl33_5 r0 r1 r2 r3 r4 r5 t + Wl33_6 r0 r1 r2 r3 r4 r5 t + Wl33_7 r0 r1 r2 r3 r4 r5 t + Wl33_8 r0 r1 r2 r3 r4 r5 t + Wl33_9 r0 r1 r2 r3 r4 r5 t + Wl33_10 r0 r1 r2 r3 r4 r5 t + Wl33_11 r0 r1 r2 r3 r4 r5 t + Wl33_12 r0 r1 r2 r3 r4 r5 t + Wl33_13 r0 r1 r2 r3 r4 r5 t + Wl33_14 r0 r1 r2 r3 r4 r5 t) ≤ Dg33 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Wl33_0, Wl33_1, Wl33_2, Wl33_3, Wl33_4, Wl33_5, Wl33_6, Wl33_7, Wl33_8, Wl33_9, Wl33_10, Wl33_11, Wl33_12, Wl33_13, Wl33_14, Dg33]
    exact CaseSplit.lowest6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n33, (1 : ℤ) ≤ Dg33 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Dg33]
    exact CaseSplit.degpos6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n33 : ℤ) + ((∑ t ∈ Finset.range n33, Wl33_0 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n33, Wl33_1 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n33, Wl33_2 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n33, Wl33_3 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n33, Wl33_4 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n33, Wl33_5 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n33, Wl33_6 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n33, Wl33_7 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n33, Wl33_8 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n33, Wl33_9 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n33, Wl33_10 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n33, Wl33_11 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n33, Wl33_12 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n33, Wl33_13 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n33, Wl33_14 r0 r1 r2 r3 r4 r5 t)) ≤ ∑ t ∈ Finset.range n33, Dg33 r0 r1 r2 r3 r4 r5 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N33_0 r0 r1 ≤ ∑ t ∈ Finset.range n33, Wl33_0 r0 r1 r2 r3 r4 r5 t := by
    simp only [N33_0, Wl33_0, le_refl]
  have hn1 : N33_1 r0 r2 ≤ ∑ t ∈ Finset.range n33, Wl33_1 r0 r1 r2 r3 r4 r5 t := by
    simp only [N33_1, Wl33_1, le_refl]
  have hn2 : N33_2 r0 r3 ≤ ∑ t ∈ Finset.range n33, Wl33_2 r0 r1 r2 r3 r4 r5 t := by
    simp only [N33_2, Wl33_2, le_refl]
  have hn3 : N33_3 r0 r4 ≤ ∑ t ∈ Finset.range n33, Wl33_3 r0 r1 r2 r3 r4 r5 t := by
    simp only [N33_3, Wl33_3, le_refl]
  have hn4 : N33_4 r0 r5 ≤ ∑ t ∈ Finset.range n33, Wl33_4 r0 r1 r2 r3 r4 r5 t := by
    simp only [N33_4, Wl33_4, le_refl]
  have hn5 : N33_5 r1 r2 ≤ ∑ t ∈ Finset.range n33, Wl33_5 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n33, Wl33_5 r0 r1 r2 r3 r4 r5 t
        = (if c33_1 r1 t && c33_2 r2 t then (1:ℤ) else 0)
          - (if c33_1 r1 t && c33_2 r2 t && c33_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl33_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n33, Wl33_5 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl33_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n33, Wl33_5 r0 r1 r2 r3 r4 r5 t
        = P33_5 r1 r2 - C33_5 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P33_5, C33_5]
    have hm : C33_5 r1 r2 r0 ≤ M33_5 r1 r2 :=
      CaseSplit.le_mxr (C33_5 r1 r2) 10 r0 (by omega)
    simp only [N33_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N33_6 r1 r3 ≤ ∑ t ∈ Finset.range n33, Wl33_6 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n33, Wl33_6 r0 r1 r2 r3 r4 r5 t
        = (if c33_1 r1 t && c33_3 r3 t then (1:ℤ) else 0)
          - (if c33_1 r1 t && c33_3 r3 t && c33_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl33_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n33, Wl33_6 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl33_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n33, Wl33_6 r0 r1 r2 r3 r4 r5 t
        = P33_6 r1 r3 - C33_6 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P33_6, C33_6]
    have hm : C33_6 r1 r3 r0 ≤ M33_6 r1 r3 :=
      CaseSplit.le_mxr (C33_6 r1 r3) 10 r0 (by omega)
    simp only [N33_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N33_7 r1 r4 ≤ ∑ t ∈ Finset.range n33, Wl33_7 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n33, Wl33_7 r0 r1 r2 r3 r4 r5 t
        = (if c33_1 r1 t && c33_4 r4 t then (1:ℤ) else 0)
          - (if c33_1 r1 t && c33_4 r4 t && c33_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl33_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n33, Wl33_7 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl33_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n33, Wl33_7 r0 r1 r2 r3 r4 r5 t
        = P33_7 r1 r4 - C33_7 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P33_7, C33_7]
    have hm : C33_7 r1 r4 r0 ≤ M33_7 r1 r4 :=
      CaseSplit.le_mxr (C33_7 r1 r4) 10 r0 (by omega)
    simp only [N33_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N33_8 r1 r5 ≤ ∑ t ∈ Finset.range n33, Wl33_8 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n33, Wl33_8 r0 r1 r2 r3 r4 r5 t
        = (if c33_1 r1 t && c33_5 r5 t then (1:ℤ) else 0)
          - (if c33_1 r1 t && c33_5 r5 t && c33_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl33_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n33, Wl33_8 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl33_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n33, Wl33_8 r0 r1 r2 r3 r4 r5 t
        = P33_8 r1 r5 - C33_8 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P33_8, C33_8]
    have hm : C33_8 r1 r5 r0 ≤ M33_8 r1 r5 :=
      CaseSplit.le_mxr (C33_8 r1 r5) 10 r0 (by omega)
    simp only [N33_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N33_9 r2 r3 ≤ ∑ t ∈ Finset.range n33, Wl33_9 r0 r1 r2 r3 r4 r5 t := by
    simp only [N33_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl33_9]
    exact CaseSplit.ind_nonneg _
  have hn10 : N33_10 r2 r4 ≤ ∑ t ∈ Finset.range n33, Wl33_10 r0 r1 r2 r3 r4 r5 t := by
    simp only [N33_10]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl33_10]
    exact CaseSplit.ind_nonneg _
  have hn11 : N33_11 r2 r5 ≤ ∑ t ∈ Finset.range n33, Wl33_11 r0 r1 r2 r3 r4 r5 t := by
    simp only [N33_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl33_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N33_12 r3 r4 ≤ ∑ t ∈ Finset.range n33, Wl33_12 r0 r1 r2 r3 r4 r5 t := by
    simp only [N33_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl33_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N33_13 r3 r5 ≤ ∑ t ∈ Finset.range n33, Wl33_13 r0 r1 r2 r3 r4 r5 t := by
    simp only [N33_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl33_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N33_14 r4 r5 ≤ ∑ t ∈ Finset.range n33, Wl33_14 r0 r1 r2 r3 r4 r5 t := by
    simp only [N33_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl33_14]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n33, (w33 t + 3) * Dg33 r0 r1 r2 r3 r4 r5 t = S33_0 r0 + S33_1 r1 + S33_2 r2 + S33_3 r3 + S33_4 r4 + S33_5 r5 := by
    simp only [S33_0, S33_1, S33_2, S33_3, S33_4, S33_5, Dg33, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n33, (w33 t + 3) * Dg33 r0 r1 r2 r3 r4 r5 t
      = (∑ t ∈ Finset.range n33, w33 t * Dg33 r0 r1 r2 r3 r4 r5 t)
        + 3 * (∑ t ∈ Finset.range n33, Dg33 r0 r1 r2 r3 r4 r5 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n33, w33 t)
      ≤ ∑ t ∈ Finset.range n33, w33 t * Dg33 r0 r1 r2 r3 r4 r5 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg33 r0 r1 r2 r3 r4 r5 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w33 t := wnn33 t (Finset.mem_range.mp ht)
    calc w33 t = w33 t * 1 := (mul_one _).symm
      _ ≤ w33 t * Dg33 r0 r1 r2 r3 r4 r5 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS33_0 r0 + aS33_1 r1 + aS33_2 r2 + aS33_3 r3 + aS33_4 r4 + aS33_5 r5) + (aP33_0 r0 r1 + aP33_1 r0 r2 + aP33_2 r0 r3 + aP33_3 r0 r4 + aP33_4 r0 r5 + aP33_5 r1 r2 + aP33_6 r1 r3 + aP33_7 r1 r4 + aP33_8 r1 r5 + aP33_9 r2 r3 + aP33_10 r2 r4 + aP33_11 r2 r5 + aP33_12 r3 r4 + aP33_13 r3 r5 + aP33_14 r4 r5) = (S33_0 r0 + S33_1 r1 + S33_2 r2 + S33_3 r3 + S33_4 r4 + S33_5 r5) - 3 * (N33_0 r0 r1 + N33_1 r0 r2 + N33_2 r0 r3 + N33_3 r0 r4 + N33_4 r0 r5 + N33_5 r1 r2 + N33_6 r1 r3 + N33_7 r1 r4 + N33_8 r1 r5 + N33_9 r2 r3 + N33_10 r2 r4 + N33_11 r2 r5 + N33_12 r3 r4 + N33_13 r3 r5 + N33_14 r4 r5) := by
    simp only [aS33_0, aS33_1, aS33_2, aS33_3, aS33_4, aS33_5, aP33_0, aP33_1, aP33_2, aP33_3, aP33_4, aP33_5, aP33_6, aP33_7, aP33_8, aP33_9, aP33_10, aP33_11, aP33_12, aP33_13, aP33_14, L33_0, L33_1, L33_2, L33_3, L33_4, L33_5]
    ring
  have bS0 : aS33_0 r0 ≤ MS33_0 := CaseSplit.le_mxr (aS33_0) 10 r0 (by omega)
  have bS1 : aS33_1 r1 ≤ MS33_1 := CaseSplit.le_mxr (aS33_1) 12 r1 (by omega)
  have bS2 : aS33_2 r2 ≤ MS33_2 := CaseSplit.le_mxr (aS33_2) 16 r2 (by omega)
  have bS3 : aS33_3 r3 ≤ MS33_3 := CaseSplit.le_mxr (aS33_3) 18 r3 (by omega)
  have bS4 : aS33_4 r4 ≤ MS33_4 := CaseSplit.le_mxr (aS33_4) 22 r4 (by omega)
  have bS5 : aS33_5 r5 ≤ MS33_5 := CaseSplit.le_mxr (aS33_5) 28 r5 (by omega)
  have bP0 : aP33_0 r0 r1 ≤ MP33_0 := CaseSplit.le_mxr2 (aP33_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP33_1 r0 r2 ≤ MP33_1 := CaseSplit.le_mxr2 (aP33_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP33_2 r0 r3 ≤ MP33_2 := CaseSplit.le_mxr2 (aP33_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP33_3 r0 r4 ≤ MP33_3 := CaseSplit.le_mxr2 (aP33_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP33_4 r0 r5 ≤ MP33_4 := CaseSplit.le_mxr2 (aP33_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP33_5 r1 r2 ≤ MP33_5 := CaseSplit.le_mxr2 (aP33_5) 12 16 r1 r2 (by omega) (by omega)
  have bP6 : aP33_6 r1 r3 ≤ MP33_6 := CaseSplit.le_mxr2 (aP33_6) 12 18 r1 r3 (by omega) (by omega)
  have bP7 : aP33_7 r1 r4 ≤ MP33_7 := CaseSplit.le_mxr2 (aP33_7) 12 22 r1 r4 (by omega) (by omega)
  have bP8 : aP33_8 r1 r5 ≤ MP33_8 := CaseSplit.le_mxr2 (aP33_8) 12 28 r1 r5 (by omega) (by omega)
  have bP9 : aP33_9 r2 r3 ≤ MP33_9 := CaseSplit.le_mxr2 (aP33_9) 16 18 r2 r3 (by omega) (by omega)
  have bP10 : aP33_10 r2 r4 ≤ MP33_10 := CaseSplit.le_mxr2 (aP33_10) 16 22 r2 r4 (by omega) (by omega)
  have bP11 : aP33_11 r2 r5 ≤ MP33_11 := CaseSplit.le_mxr2 (aP33_11) 16 28 r2 r5 (by omega) (by omega)
  have bP12 : aP33_12 r3 r4 ≤ MP33_12 := CaseSplit.le_mxr2 (aP33_12) 18 22 r3 r4 (by omega) (by omega)
  have bP13 : aP33_13 r3 r5 ≤ MP33_13 := CaseSplit.le_mxr2 (aP33_13) 18 28 r3 r5 (by omega) (by omega)
  have bP14 : aP33_14 r4 r5 ≤ MP33_14 := CaseSplit.le_mxr2 (aP33_14) 22 28 r4 r5 (by omega) (by omega)
  have hrhs : rhs33 = (∑ t ∈ Finset.range n33, w33 t) + 3 * (n33 : ℤ) := rfl
  have hc := cert33
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, bS0, bS1, bS2, bS3, bS4, bS5, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14]

end IncCert29
