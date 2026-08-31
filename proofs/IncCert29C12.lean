/-
INCREMENT-WIDTH CERTIFICATE, step 23->29, case 12 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_23_29.json, which re-derives every number
from the primes alone).

Machine 29, INCREMENT width 49 = F_2(23) + s_min(29) = 39 + 10,
held gears [5, 7] at phases [1, 5].  Free gears [11, 13, 17, 19, 23, 29].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 4.
-/
import IncCert29B

namespace IncCert29

/-! ### case 12: held gears at phases [1, 5] -/

def p12 : List ℕ := [2, 4, 6, 7, 9, 11, 12, 14, 16, 19, 21, 26, 27, 32, 34, 37, 39, 41, 42, 44, 46, 47]
def q12 (t : ℕ) : ℕ := p12.getD t 0
def n12 : ℕ := 22
def yl12 : List ℤ := [0, 0, 0, 0, 3, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0]
def w12 (t : ℕ) : ℤ := yl12.getD t 0
def ul12 : List ℤ := [0, (-7), 0, (-1), 0, (-1), (-1), (-1), 0, 1, (-1), (-1), (-1), 1, 0, (-1), 0, (-1), 0, (-1), 1, (-1), (-1), 0, (-2), (-3), (-7), (-2), (-1), 0, (-3), (-3), (-3), (-2), 1, 0, (-3), (-3), (-3), (-3), (-2), (-1), (-1), 0, 0, 2, 1, (-1), (-1), 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, (-2), (-2), (-1), 0, (-2), (-2), (-1), (-1), (-2), 0, 0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-5), (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, (-1), 0, (-1), (-1), (-1), 0, (-1), (-1), 0, (-1), (-1), (-1), 0, (-1), (-1), 0, 0, (-1), (-1), (-1), 0, (-1), 0, (-1), (-1), (-1), 0, 0, (-1), 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 13, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 11, 11, 14, 14, (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), 10, 14, 13, 8, 14, 4, 6, 14, 14, 12, 13, 4, 8, 14, 14, 14, 13, 4, 8, (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), (-2), 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-11), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 5, 6, 10, 2, 5, 6, 10, 11, 7, 10, 6, 10, 10, 8, 12, 9, 10, 12, 11, 12, 10, 12, 6, 8, 9, 10, 10, 6, 15, 8, 5, 16, 4, 12, 9, 5, 16, 5, 12, 16, 5, 16, 0, 12, 13, 4, 15, 16, 4, 13, 0]
def u12 (k : ℕ) : ℤ := ul12.getD k 0

def c12_0 (r t : ℕ) : Bool := gb11 r (q12 t)
def c12_1 (r t : ℕ) : Bool := gb13 r (q12 t)
def c12_2 (r t : ℕ) : Bool := gb17 r (q12 t)
def c12_3 (r t : ℕ) : Bool := gb19 r (q12 t)
def c12_4 (r t : ℕ) : Bool := gb23 r (q12 t)
def c12_5 (r t : ℕ) : Bool := gb29 r (q12 t)

def S12_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (w12 t + 4) * (if c12_0 r t then 1 else 0)
def S12_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (w12 t + 4) * (if c12_1 r t then 1 else 0)
def S12_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (w12 t + 4) * (if c12_2 r t then 1 else 0)
def S12_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (w12 t + 4) * (if c12_3 r t then 1 else 0)
def S12_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (w12 t + 4) * (if c12_4 r t then 1 else 0)
def S12_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (w12 t + 4) * (if c12_5 r t then 1 else 0)

def L12_0 (r : ℕ) : ℤ := u12 (13 + r) + u12 (41 + r) + u12 (71 + r) + u12 (105 + r) + u12 (145 + r)
def L12_1 (r : ℕ) : ℤ := u12 (0 + r) + u12 (173 + r) + u12 (205 + r) + u12 (241 + r) + u12 (283 + r)
def L12_2 (r : ℕ) : ℤ := u12 (24 + r) + u12 (156 + r) + u12 (315 + r) + u12 (355 + r) + u12 (401 + r)
def L12_3 (r : ℕ) : ℤ := u12 (52 + r) + u12 (186 + r) + u12 (296 + r) + u12 (441 + r) + u12 (489 + r)
def L12_4 (r : ℕ) : ℤ := u12 (82 + r) + u12 (218 + r) + u12 (332 + r) + u12 (418 + r) + u12 (537 + r)
def L12_5 (r : ℕ) : ℤ := u12 (116 + r) + u12 (254 + r) + u12 (372 + r) + u12 (460 + r) + u12 (508 + r)

def aS12_0 (r : ℕ) : ℤ := S12_0 r - L12_0 r
def MS12_0 : ℤ := CaseSplit.mxr (aS12_0) 10
def aS12_1 (r : ℕ) : ℤ := S12_1 r - L12_1 r
def MS12_1 : ℤ := CaseSplit.mxr (aS12_1) 12
def aS12_2 (r : ℕ) : ℤ := S12_2 r - L12_2 r
def MS12_2 : ℤ := CaseSplit.mxr (aS12_2) 16
def aS12_3 (r : ℕ) : ℤ := S12_3 r - L12_3 r
def MS12_3 : ℤ := CaseSplit.mxr (aS12_3) 18
def aS12_4 (r : ℕ) : ℤ := S12_4 r - L12_4 r
def MS12_4 : ℤ := CaseSplit.mxr (aS12_4) 22
def aS12_5 (r : ℕ) : ℤ := S12_5 r - L12_5 r
def MS12_5 : ℤ := CaseSplit.mxr (aS12_5) 28

def N12_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (if c12_0 ra t && c12_1 rb t then 1 else 0)
def aP12_0 (ra rb : ℕ) : ℤ := -(4) * N12_0 ra rb + u12 (0 + rb) + u12 (13 + ra)
def MP12_0 : ℤ := CaseSplit.mxr2 (aP12_0) 10 12
def N12_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (if c12_0 ra t && c12_2 rb t then 1 else 0)
def aP12_1 (ra rb : ℕ) : ℤ := -(4) * N12_1 ra rb + u12 (24 + rb) + u12 (41 + ra)
def MP12_1 : ℤ := CaseSplit.mxr2 (aP12_1) 10 16
def N12_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (if c12_0 ra t && c12_3 rb t then 1 else 0)
def aP12_2 (ra rb : ℕ) : ℤ := -(4) * N12_2 ra rb + u12 (52 + rb) + u12 (71 + ra)
def MP12_2 : ℤ := CaseSplit.mxr2 (aP12_2) 10 18
def N12_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (if c12_0 ra t && c12_4 rb t then 1 else 0)
def aP12_3 (ra rb : ℕ) : ℤ := -(4) * N12_3 ra rb + u12 (82 + rb) + u12 (105 + ra)
def MP12_3 : ℤ := CaseSplit.mxr2 (aP12_3) 10 22
def N12_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (if c12_0 ra t && c12_5 rb t then 1 else 0)
def aP12_4 (ra rb : ℕ) : ℤ := -(4) * N12_4 ra rb + u12 (116 + rb) + u12 (145 + ra)
def MP12_4 : ℤ := CaseSplit.mxr2 (aP12_4) 10 28
def P12_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (if c12_1 ra t && c12_2 rb t then 1 else 0)
def C12_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (if c12_1 ra t && c12_2 rb t && c12_0 s t then 1 else 0)
def M12_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C12_5 ra rb) 10
def E12_5 : List ℕ := [39, 45, 129, 135, 154, 165, 174, 180]
def N12_5 (ra rb : ℕ) : ℤ := if E12_5.contains (ra * 17 + rb) = true then P12_5 ra rb - M12_5 ra rb else 0
def aP12_5 (ra rb : ℕ) : ℤ := -(4) * N12_5 ra rb + u12 (156 + rb) + u12 (173 + ra)
def MP12_5 : ℤ := CaseSplit.mxr2 (aP12_5) 12 16
def P12_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (if c12_1 ra t && c12_3 rb t then 1 else 0)
def C12_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (if c12_1 ra t && c12_3 rb t && c12_0 s t then 1 else 0)
def M12_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C12_6 ra rb) 10
def E12_6 : List ℕ := [27, 53, 58, 64, 98, 111, 134, 140, 174, 187, 198, 224]
def N12_6 (ra rb : ℕ) : ℤ := if E12_6.contains (ra * 19 + rb) = true then P12_6 ra rb - M12_6 ra rb else 0
def aP12_6 (ra rb : ℕ) : ℤ := -(4) * N12_6 ra rb + u12 (186 + rb) + u12 (205 + ra)
def MP12_6 : ℤ := CaseSplit.mxr2 (aP12_6) 12 18
def P12_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (if c12_1 ra t && c12_4 rb t then 1 else 0)
def C12_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (if c12_1 ra t && c12_4 rb t && c12_0 s t then 1 else 0)
def M12_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C12_7 ra rb) 10
def E12_7 : List ℕ := []
def N12_7 (ra rb : ℕ) : ℤ := if E12_7.contains (ra * 23 + rb) = true then P12_7 ra rb - M12_7 ra rb else 0
def aP12_7 (ra rb : ℕ) : ℤ := -(4) * N12_7 ra rb + u12 (218 + rb) + u12 (241 + ra)
def MP12_7 : ℤ := CaseSplit.mxr2 (aP12_7) 12 22
def P12_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (if c12_1 ra t && c12_5 rb t then 1 else 0)
def C12_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (if c12_1 ra t && c12_5 rb t && c12_0 s t then 1 else 0)
def M12_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C12_8 ra rb) 10
def E12_8 : List ℕ := [22, 133, 249, 283]
def N12_8 (ra rb : ℕ) : ℤ := if E12_8.contains (ra * 29 + rb) = true then P12_8 ra rb - M12_8 ra rb else 0
def aP12_8 (ra rb : ℕ) : ℤ := -(4) * N12_8 ra rb + u12 (254 + rb) + u12 (283 + ra)
def MP12_8 : ℤ := CaseSplit.mxr2 (aP12_8) 12 28
def N12_9 (_ra _rb : ℕ) : ℤ := 0
def aP12_9 (ra rb : ℕ) : ℤ := -(4) * N12_9 ra rb + u12 (296 + rb) + u12 (315 + ra)
def MP12_9 : ℤ := CaseSplit.mxr2 (aP12_9) 16 18
def N12_10 (_ra _rb : ℕ) : ℤ := 0
def aP12_10 (ra rb : ℕ) : ℤ := -(4) * N12_10 ra rb + u12 (332 + rb) + u12 (355 + ra)
def MP12_10 : ℤ := CaseSplit.mxr2 (aP12_10) 16 22
def N12_11 (_ra _rb : ℕ) : ℤ := 0
def aP12_11 (ra rb : ℕ) : ℤ := -(4) * N12_11 ra rb + u12 (372 + rb) + u12 (401 + ra)
def MP12_11 : ℤ := CaseSplit.mxr2 (aP12_11) 16 28
def N12_12 (_ra _rb : ℕ) : ℤ := 0
def aP12_12 (ra rb : ℕ) : ℤ := -(4) * N12_12 ra rb + u12 (418 + rb) + u12 (441 + ra)
def MP12_12 : ℤ := CaseSplit.mxr2 (aP12_12) 18 22
def N12_13 (_ra _rb : ℕ) : ℤ := 0
def aP12_13 (ra rb : ℕ) : ℤ := -(4) * N12_13 ra rb + u12 (460 + rb) + u12 (489 + ra)
def MP12_13 : ℤ := CaseSplit.mxr2 (aP12_13) 18 28
def N12_14 (_ra _rb : ℕ) : ℤ := 0
def aP12_14 (ra rb : ℕ) : ℤ := -(4) * N12_14 ra rb + u12 (508 + rb) + u12 (537 + ra)
def MP12_14 : ℤ := CaseSplit.mxr2 (aP12_14) 22 28

def rhs12 : ℤ := (∑ t ∈ Finset.range n12, w12 t) + 4 * (n12 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn12 : ∀ t, t < n12 → (0 : ℤ) ≤ w12 t := by decide
theorem plt12 : ∀ t, t < n12 → q12 t < 49 := by decide
theorem pfree12_5 : ∀ t, t < n12 → gb5 1 (q12 t) = false := by decide
theorem pfree12_7 : ∀ t, t < n12 → gb7 5 (q12 t) = false := by decide
theorem MSv12_0 : MS12_0 = 20 := by decide +kernel
theorem MSv12_1 : MS12_1 = 48 := by decide +kernel
theorem MSv12_2 : MS12_2 = 1 := by decide +kernel
theorem MSv12_3 : MS12_3 = 0 := by decide +kernel
theorem MSv12_4 : MS12_4 = 1 := by decide +kernel
theorem MSv12_5 : MS12_5 = 0 := by decide +kernel
theorem MPv12_0 : MP12_0 = 0 := by decide +kernel
theorem MPv12_1 : MP12_1 = 0 := by decide +kernel
theorem MPv12_2 : MP12_2 = 0 := by decide +kernel
theorem MPv12_3 : MP12_3 = 0 := by decide +kernel
theorem MPv12_4 : MP12_4 = 0 := by decide +kernel
theorem MPv12_5 : MP12_5 = 0 := by decide +kernel
theorem MPv12_6 : MP12_6 = 0 := by decide +kernel
theorem MPv12_7 : MP12_7 = 0 := by decide +kernel
theorem MPv12_8 : MP12_8 = 0 := by decide +kernel
theorem MPv12_9 : MP12_9 = 0 := by decide +kernel
theorem MPv12_10 : MP12_10 = 0 := by decide +kernel
theorem MPv12_11 : MP12_11 = 0 := by decide +kernel
theorem MPv12_12 : MP12_12 = 0 := by decide +kernel
theorem MPv12_13 : MP12_13 = 0 := by decide +kernel
theorem MPv12_14 : MP12_14 = 28 := by decide +kernel
theorem rhsv12 : rhs12 = 99 := by decide +kernel

/-- **The case-12 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/4.
    (Scaled by the common denominator 4: 98 < 99.) -/
theorem cert12 : MS12_0 + MS12_1 + MS12_2 + MS12_3 + MS12_4 + MS12_5 + MP12_0 + MP12_1 + MP12_2 + MP12_3 + MP12_4 + MP12_5 + MP12_6 + MP12_7 + MP12_8 + MP12_9 + MP12_10 + MP12_11 + MP12_12 + MP12_13 + MP12_14 < rhs12 := by
  rw [MSv12_0, MSv12_1, MSv12_2, MSv12_3, MSv12_4, MSv12_5, MPv12_0, MPv12_1, MPv12_2, MPv12_3, MPv12_4, MPv12_5, MPv12_6, MPv12_7, MPv12_8, MPv12_9, MPv12_10, MPv12_11, MPv12_12, MPv12_13, MPv12_14, rhsv12]
  decide

def Dg12 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := (if c12_0 r0 t then 1 else 0) + (if c12_1 r1 t then 1 else 0) + (if c12_2 r2 t then 1 else 0) + (if c12_3 r3 t then 1 else 0) + (if c12_4 r4 t then 1 else 0) + (if c12_5 r5 t then 1 else 0)
def Wl12_0 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c12_0 r0 t && c12_1 r1 t then 1 else 0
def Wl12_1 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c12_0 r0 t && c12_2 r2 t then 1 else 0
def Wl12_2 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c12_0 r0 t && c12_3 r3 t then 1 else 0
def Wl12_3 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c12_0 r0 t && c12_4 r4 t then 1 else 0
def Wl12_4 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c12_0 r0 t && c12_5 r5 t then 1 else 0
def Wl12_5 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c12_0 r0 t && c12_1 r1 t && c12_2 r2 t then 1 else 0
def Wl12_6 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c12_0 r0 t && c12_1 r1 t && c12_3 r3 t then 1 else 0
def Wl12_7 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c12_0 r0 t && c12_1 r1 t && c12_4 r4 t then 1 else 0
def Wl12_8 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c12_0 r0 t && c12_1 r1 t && c12_5 r5 t then 1 else 0
def Wl12_9 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c12_0 r0 t && !c12_1 r1 t && c12_2 r2 t && c12_3 r3 t then 1 else 0
def Wl12_10 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c12_0 r0 t && !c12_1 r1 t && c12_2 r2 t && c12_4 r4 t then 1 else 0
def Wl12_11 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c12_0 r0 t && !c12_1 r1 t && c12_2 r2 t && c12_5 r5 t then 1 else 0
def Wl12_12 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c12_0 r0 t && !c12_1 r1 t && !c12_2 r2 t && c12_3 r3 t && c12_4 r4 t then 1 else 0
def Wl12_13 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c12_0 r0 t && !c12_1 r1 t && !c12_2 r2 t && c12_3 r3 t && c12_5 r5 t then 1 else 0
def Wl12_14 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c12_0 r0 t && !c12_1 r1 t && !c12_2 r2 t && !c12_3 r3 t && c12_4 r4 t && c12_5 r5 t then 1 else 0

/-- **No configuration blocks the whole window in case 12.** -/
theorem nocov12 {r0 r1 r2 r3 r4 r5 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29)
    (hcov : ∀ t, t < n12 → (c12_0 r0 t || c12_1 r1 t || c12_2 r2 t || c12_3 r3 t || c12_4 r4 t || c12_5 r5 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n12, (1 : ℤ) + (Wl12_0 r0 r1 r2 r3 r4 r5 t + Wl12_1 r0 r1 r2 r3 r4 r5 t + Wl12_2 r0 r1 r2 r3 r4 r5 t + Wl12_3 r0 r1 r2 r3 r4 r5 t + Wl12_4 r0 r1 r2 r3 r4 r5 t + Wl12_5 r0 r1 r2 r3 r4 r5 t + Wl12_6 r0 r1 r2 r3 r4 r5 t + Wl12_7 r0 r1 r2 r3 r4 r5 t + Wl12_8 r0 r1 r2 r3 r4 r5 t + Wl12_9 r0 r1 r2 r3 r4 r5 t + Wl12_10 r0 r1 r2 r3 r4 r5 t + Wl12_11 r0 r1 r2 r3 r4 r5 t + Wl12_12 r0 r1 r2 r3 r4 r5 t + Wl12_13 r0 r1 r2 r3 r4 r5 t + Wl12_14 r0 r1 r2 r3 r4 r5 t) ≤ Dg12 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Wl12_0, Wl12_1, Wl12_2, Wl12_3, Wl12_4, Wl12_5, Wl12_6, Wl12_7, Wl12_8, Wl12_9, Wl12_10, Wl12_11, Wl12_12, Wl12_13, Wl12_14, Dg12]
    exact CaseSplit.lowest6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n12, (1 : ℤ) ≤ Dg12 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Dg12]
    exact CaseSplit.degpos6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n12 : ℤ) + ((∑ t ∈ Finset.range n12, Wl12_0 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n12, Wl12_1 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n12, Wl12_2 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n12, Wl12_3 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n12, Wl12_4 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n12, Wl12_5 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n12, Wl12_6 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n12, Wl12_7 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n12, Wl12_8 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n12, Wl12_9 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n12, Wl12_10 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n12, Wl12_11 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n12, Wl12_12 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n12, Wl12_13 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n12, Wl12_14 r0 r1 r2 r3 r4 r5 t)) ≤ ∑ t ∈ Finset.range n12, Dg12 r0 r1 r2 r3 r4 r5 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N12_0 r0 r1 ≤ ∑ t ∈ Finset.range n12, Wl12_0 r0 r1 r2 r3 r4 r5 t := by
    simp only [N12_0, Wl12_0, le_refl]
  have hn1 : N12_1 r0 r2 ≤ ∑ t ∈ Finset.range n12, Wl12_1 r0 r1 r2 r3 r4 r5 t := by
    simp only [N12_1, Wl12_1, le_refl]
  have hn2 : N12_2 r0 r3 ≤ ∑ t ∈ Finset.range n12, Wl12_2 r0 r1 r2 r3 r4 r5 t := by
    simp only [N12_2, Wl12_2, le_refl]
  have hn3 : N12_3 r0 r4 ≤ ∑ t ∈ Finset.range n12, Wl12_3 r0 r1 r2 r3 r4 r5 t := by
    simp only [N12_3, Wl12_3, le_refl]
  have hn4 : N12_4 r0 r5 ≤ ∑ t ∈ Finset.range n12, Wl12_4 r0 r1 r2 r3 r4 r5 t := by
    simp only [N12_4, Wl12_4, le_refl]
  have hn5 : N12_5 r1 r2 ≤ ∑ t ∈ Finset.range n12, Wl12_5 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n12, Wl12_5 r0 r1 r2 r3 r4 r5 t
        = (if c12_1 r1 t && c12_2 r2 t then (1:ℤ) else 0)
          - (if c12_1 r1 t && c12_2 r2 t && c12_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl12_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n12, Wl12_5 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl12_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n12, Wl12_5 r0 r1 r2 r3 r4 r5 t
        = P12_5 r1 r2 - C12_5 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P12_5, C12_5]
    have hm : C12_5 r1 r2 r0 ≤ M12_5 r1 r2 :=
      CaseSplit.le_mxr (C12_5 r1 r2) 10 r0 (by omega)
    simp only [N12_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N12_6 r1 r3 ≤ ∑ t ∈ Finset.range n12, Wl12_6 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n12, Wl12_6 r0 r1 r2 r3 r4 r5 t
        = (if c12_1 r1 t && c12_3 r3 t then (1:ℤ) else 0)
          - (if c12_1 r1 t && c12_3 r3 t && c12_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl12_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n12, Wl12_6 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl12_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n12, Wl12_6 r0 r1 r2 r3 r4 r5 t
        = P12_6 r1 r3 - C12_6 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P12_6, C12_6]
    have hm : C12_6 r1 r3 r0 ≤ M12_6 r1 r3 :=
      CaseSplit.le_mxr (C12_6 r1 r3) 10 r0 (by omega)
    simp only [N12_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N12_7 r1 r4 ≤ ∑ t ∈ Finset.range n12, Wl12_7 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n12, Wl12_7 r0 r1 r2 r3 r4 r5 t
        = (if c12_1 r1 t && c12_4 r4 t then (1:ℤ) else 0)
          - (if c12_1 r1 t && c12_4 r4 t && c12_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl12_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n12, Wl12_7 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl12_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n12, Wl12_7 r0 r1 r2 r3 r4 r5 t
        = P12_7 r1 r4 - C12_7 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P12_7, C12_7]
    have hm : C12_7 r1 r4 r0 ≤ M12_7 r1 r4 :=
      CaseSplit.le_mxr (C12_7 r1 r4) 10 r0 (by omega)
    simp only [N12_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N12_8 r1 r5 ≤ ∑ t ∈ Finset.range n12, Wl12_8 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n12, Wl12_8 r0 r1 r2 r3 r4 r5 t
        = (if c12_1 r1 t && c12_5 r5 t then (1:ℤ) else 0)
          - (if c12_1 r1 t && c12_5 r5 t && c12_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl12_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n12, Wl12_8 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl12_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n12, Wl12_8 r0 r1 r2 r3 r4 r5 t
        = P12_8 r1 r5 - C12_8 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P12_8, C12_8]
    have hm : C12_8 r1 r5 r0 ≤ M12_8 r1 r5 :=
      CaseSplit.le_mxr (C12_8 r1 r5) 10 r0 (by omega)
    simp only [N12_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N12_9 r2 r3 ≤ ∑ t ∈ Finset.range n12, Wl12_9 r0 r1 r2 r3 r4 r5 t := by
    simp only [N12_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl12_9]
    exact CaseSplit.ind_nonneg _
  have hn10 : N12_10 r2 r4 ≤ ∑ t ∈ Finset.range n12, Wl12_10 r0 r1 r2 r3 r4 r5 t := by
    simp only [N12_10]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl12_10]
    exact CaseSplit.ind_nonneg _
  have hn11 : N12_11 r2 r5 ≤ ∑ t ∈ Finset.range n12, Wl12_11 r0 r1 r2 r3 r4 r5 t := by
    simp only [N12_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl12_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N12_12 r3 r4 ≤ ∑ t ∈ Finset.range n12, Wl12_12 r0 r1 r2 r3 r4 r5 t := by
    simp only [N12_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl12_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N12_13 r3 r5 ≤ ∑ t ∈ Finset.range n12, Wl12_13 r0 r1 r2 r3 r4 r5 t := by
    simp only [N12_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl12_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N12_14 r4 r5 ≤ ∑ t ∈ Finset.range n12, Wl12_14 r0 r1 r2 r3 r4 r5 t := by
    simp only [N12_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl12_14]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n12, (w12 t + 4) * Dg12 r0 r1 r2 r3 r4 r5 t = S12_0 r0 + S12_1 r1 + S12_2 r2 + S12_3 r3 + S12_4 r4 + S12_5 r5 := by
    simp only [S12_0, S12_1, S12_2, S12_3, S12_4, S12_5, Dg12, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n12, (w12 t + 4) * Dg12 r0 r1 r2 r3 r4 r5 t
      = (∑ t ∈ Finset.range n12, w12 t * Dg12 r0 r1 r2 r3 r4 r5 t)
        + 4 * (∑ t ∈ Finset.range n12, Dg12 r0 r1 r2 r3 r4 r5 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n12, w12 t)
      ≤ ∑ t ∈ Finset.range n12, w12 t * Dg12 r0 r1 r2 r3 r4 r5 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg12 r0 r1 r2 r3 r4 r5 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w12 t := wnn12 t (Finset.mem_range.mp ht)
    calc w12 t = w12 t * 1 := (mul_one _).symm
      _ ≤ w12 t * Dg12 r0 r1 r2 r3 r4 r5 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS12_0 r0 + aS12_1 r1 + aS12_2 r2 + aS12_3 r3 + aS12_4 r4 + aS12_5 r5) + (aP12_0 r0 r1 + aP12_1 r0 r2 + aP12_2 r0 r3 + aP12_3 r0 r4 + aP12_4 r0 r5 + aP12_5 r1 r2 + aP12_6 r1 r3 + aP12_7 r1 r4 + aP12_8 r1 r5 + aP12_9 r2 r3 + aP12_10 r2 r4 + aP12_11 r2 r5 + aP12_12 r3 r4 + aP12_13 r3 r5 + aP12_14 r4 r5) = (S12_0 r0 + S12_1 r1 + S12_2 r2 + S12_3 r3 + S12_4 r4 + S12_5 r5) - 4 * (N12_0 r0 r1 + N12_1 r0 r2 + N12_2 r0 r3 + N12_3 r0 r4 + N12_4 r0 r5 + N12_5 r1 r2 + N12_6 r1 r3 + N12_7 r1 r4 + N12_8 r1 r5 + N12_9 r2 r3 + N12_10 r2 r4 + N12_11 r2 r5 + N12_12 r3 r4 + N12_13 r3 r5 + N12_14 r4 r5) := by
    simp only [aS12_0, aS12_1, aS12_2, aS12_3, aS12_4, aS12_5, aP12_0, aP12_1, aP12_2, aP12_3, aP12_4, aP12_5, aP12_6, aP12_7, aP12_8, aP12_9, aP12_10, aP12_11, aP12_12, aP12_13, aP12_14, L12_0, L12_1, L12_2, L12_3, L12_4, L12_5]
    ring
  have bS0 : aS12_0 r0 ≤ MS12_0 := CaseSplit.le_mxr (aS12_0) 10 r0 (by omega)
  have bS1 : aS12_1 r1 ≤ MS12_1 := CaseSplit.le_mxr (aS12_1) 12 r1 (by omega)
  have bS2 : aS12_2 r2 ≤ MS12_2 := CaseSplit.le_mxr (aS12_2) 16 r2 (by omega)
  have bS3 : aS12_3 r3 ≤ MS12_3 := CaseSplit.le_mxr (aS12_3) 18 r3 (by omega)
  have bS4 : aS12_4 r4 ≤ MS12_4 := CaseSplit.le_mxr (aS12_4) 22 r4 (by omega)
  have bS5 : aS12_5 r5 ≤ MS12_5 := CaseSplit.le_mxr (aS12_5) 28 r5 (by omega)
  have bP0 : aP12_0 r0 r1 ≤ MP12_0 := CaseSplit.le_mxr2 (aP12_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP12_1 r0 r2 ≤ MP12_1 := CaseSplit.le_mxr2 (aP12_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP12_2 r0 r3 ≤ MP12_2 := CaseSplit.le_mxr2 (aP12_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP12_3 r0 r4 ≤ MP12_3 := CaseSplit.le_mxr2 (aP12_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP12_4 r0 r5 ≤ MP12_4 := CaseSplit.le_mxr2 (aP12_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP12_5 r1 r2 ≤ MP12_5 := CaseSplit.le_mxr2 (aP12_5) 12 16 r1 r2 (by omega) (by omega)
  have bP6 : aP12_6 r1 r3 ≤ MP12_6 := CaseSplit.le_mxr2 (aP12_6) 12 18 r1 r3 (by omega) (by omega)
  have bP7 : aP12_7 r1 r4 ≤ MP12_7 := CaseSplit.le_mxr2 (aP12_7) 12 22 r1 r4 (by omega) (by omega)
  have bP8 : aP12_8 r1 r5 ≤ MP12_8 := CaseSplit.le_mxr2 (aP12_8) 12 28 r1 r5 (by omega) (by omega)
  have bP9 : aP12_9 r2 r3 ≤ MP12_9 := CaseSplit.le_mxr2 (aP12_9) 16 18 r2 r3 (by omega) (by omega)
  have bP10 : aP12_10 r2 r4 ≤ MP12_10 := CaseSplit.le_mxr2 (aP12_10) 16 22 r2 r4 (by omega) (by omega)
  have bP11 : aP12_11 r2 r5 ≤ MP12_11 := CaseSplit.le_mxr2 (aP12_11) 16 28 r2 r5 (by omega) (by omega)
  have bP12 : aP12_12 r3 r4 ≤ MP12_12 := CaseSplit.le_mxr2 (aP12_12) 18 22 r3 r4 (by omega) (by omega)
  have bP13 : aP12_13 r3 r5 ≤ MP12_13 := CaseSplit.le_mxr2 (aP12_13) 18 28 r3 r5 (by omega) (by omega)
  have bP14 : aP12_14 r4 r5 ≤ MP12_14 := CaseSplit.le_mxr2 (aP12_14) 22 28 r4 r5 (by omega) (by omega)
  have hrhs : rhs12 = (∑ t ∈ Finset.range n12, w12 t) + 4 * (n12 : ℤ) := rfl
  have hc := cert12
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, bS0, bS1, bS2, bS3, bS4, bS5, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14]

end IncCert29
