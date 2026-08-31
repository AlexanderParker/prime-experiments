/-
INCREMENT-WIDTH CERTIFICATE, step 23->29, case 15 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_23_29.json, which re-derives every number
from the primes alone).

Machine 29, INCREMENT width 49 = F_2(23) + s_min(29) = 39 + 10,
held gears [5, 7] at phases [2, 1].  Free gears [11, 13, 17, 19, 23, 29].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 4.
-/
import IncCert29B

namespace IncCert29

/-! ### case 15: held gears at phases [2, 1] -/

def p15 : List ℕ := [1, 3, 6, 8, 10, 11, 13, 15, 16, 18, 20, 23, 25, 30, 31, 36, 38, 41, 43, 45, 46, 48]
def q15 (t : ℕ) : ℕ := p15.getD t 0
def n15 : ℕ := 22
def yl15 : List ℤ := [0, 0, 0, 1, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 0, 2, 0, 0, 1]
def w15 (t : ℕ) : ℤ := yl15.getD t 0
def ul15 : List ℤ := [0, 1, 0, 1, 0, 5, 0, 0, 0, 0, 0, 0, 1, (-5), (-1), (-5), (-1), (-5), (-5), (-1), (-1), 0, (-5), 0, 3, 1, 0, (-6), (-6), 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-5), 0, (-3), (-3), (-3), (-1), 0, 0, (-3), (-3), (-1), 0, 3, 0, (-4), 0, 0, 3, 1, 0, 0, 0, 1, 3, 3, 3, 0, 0, 1, 1, 1, (-3), (-1), (-3), (-3), (-3), (-1), (-3), (-3), (-3), (-3), 0, (-4), (-3), (-4), (-4), 0, (-4), (-4), (-4), (-4), (-3), (-4), 0, (-3), (-4), (-4), (-4), (-4), (-4), (-4), 0, (-6), (-4), (-4), 3, 4, 0, 0, 3, 4, 0, 0, 3, 0, 0, (-2), 0, (-2), (-2), 0, (-2), (-2), (-2), 0, (-2), (-2), 0, (-1), (-2), (-2), (-2), (-1), (-2), 0, (-2), (-2), (-2), (-4), 0, (-2), (-2), (-1), (-2), (-2), 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 9, 14, 14, 14, 8, 4, 14, (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), 10, 4, 12, 11, 10, 11, 12, 4, 12, 12, 11, 12, 12, 12, 12, 12, 12, 12, 12, (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), 1, 14, 14, 14, 14, 5, 13, 12, 6, 14, 4, 11, 14, 5, 13, 4, 10, 14, 5, 14, 14, 8, 14, (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), (-14), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-4), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-3), (-2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 12, 6, 6, 12, 7, 8, 12, 8, 12, 8, 12, 5, 7, 10, 12, 12, 12, 12, 6, 8, 12, 12, 12, 6, 7, 6, 7, 10, 3, 3, 3, (-2), 3, 3, 3, 3, 3, 3, 0, 3, 1, 3, 3, 0, 3, 3, 3, 3, 0, 0, 0]
def u15 (k : ℕ) : ℤ := ul15.getD k 0

def c15_0 (r t : ℕ) : Bool := gb11 r (q15 t)
def c15_1 (r t : ℕ) : Bool := gb13 r (q15 t)
def c15_2 (r t : ℕ) : Bool := gb17 r (q15 t)
def c15_3 (r t : ℕ) : Bool := gb19 r (q15 t)
def c15_4 (r t : ℕ) : Bool := gb23 r (q15 t)
def c15_5 (r t : ℕ) : Bool := gb29 r (q15 t)

def S15_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (w15 t + 4) * (if c15_0 r t then 1 else 0)
def S15_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (w15 t + 4) * (if c15_1 r t then 1 else 0)
def S15_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (w15 t + 4) * (if c15_2 r t then 1 else 0)
def S15_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (w15 t + 4) * (if c15_3 r t then 1 else 0)
def S15_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (w15 t + 4) * (if c15_4 r t then 1 else 0)
def S15_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (w15 t + 4) * (if c15_5 r t then 1 else 0)

def L15_0 (r : ℕ) : ℤ := u15 (13 + r) + u15 (41 + r) + u15 (71 + r) + u15 (105 + r) + u15 (145 + r)
def L15_1 (r : ℕ) : ℤ := u15 (0 + r) + u15 (173 + r) + u15 (205 + r) + u15 (241 + r) + u15 (283 + r)
def L15_2 (r : ℕ) : ℤ := u15 (24 + r) + u15 (156 + r) + u15 (315 + r) + u15 (355 + r) + u15 (401 + r)
def L15_3 (r : ℕ) : ℤ := u15 (52 + r) + u15 (186 + r) + u15 (296 + r) + u15 (441 + r) + u15 (489 + r)
def L15_4 (r : ℕ) : ℤ := u15 (82 + r) + u15 (218 + r) + u15 (332 + r) + u15 (418 + r) + u15 (537 + r)
def L15_5 (r : ℕ) : ℤ := u15 (116 + r) + u15 (254 + r) + u15 (372 + r) + u15 (460 + r) + u15 (508 + r)

def aS15_0 (r : ℕ) : ℤ := S15_0 r - L15_0 r
def MS15_0 : ℤ := CaseSplit.mxr (aS15_0) 10
def aS15_1 (r : ℕ) : ℤ := S15_1 r - L15_1 r
def MS15_1 : ℤ := CaseSplit.mxr (aS15_1) 12
def aS15_2 (r : ℕ) : ℤ := S15_2 r - L15_2 r
def MS15_2 : ℤ := CaseSplit.mxr (aS15_2) 16
def aS15_3 (r : ℕ) : ℤ := S15_3 r - L15_3 r
def MS15_3 : ℤ := CaseSplit.mxr (aS15_3) 18
def aS15_4 (r : ℕ) : ℤ := S15_4 r - L15_4 r
def MS15_4 : ℤ := CaseSplit.mxr (aS15_4) 22
def aS15_5 (r : ℕ) : ℤ := S15_5 r - L15_5 r
def MS15_5 : ℤ := CaseSplit.mxr (aS15_5) 28

def N15_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (if c15_0 ra t && c15_1 rb t then 1 else 0)
def aP15_0 (ra rb : ℕ) : ℤ := -(4) * N15_0 ra rb + u15 (0 + rb) + u15 (13 + ra)
def MP15_0 : ℤ := CaseSplit.mxr2 (aP15_0) 10 12
def N15_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (if c15_0 ra t && c15_2 rb t then 1 else 0)
def aP15_1 (ra rb : ℕ) : ℤ := -(4) * N15_1 ra rb + u15 (24 + rb) + u15 (41 + ra)
def MP15_1 : ℤ := CaseSplit.mxr2 (aP15_1) 10 16
def N15_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (if c15_0 ra t && c15_3 rb t then 1 else 0)
def aP15_2 (ra rb : ℕ) : ℤ := -(4) * N15_2 ra rb + u15 (52 + rb) + u15 (71 + ra)
def MP15_2 : ℤ := CaseSplit.mxr2 (aP15_2) 10 18
def N15_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (if c15_0 ra t && c15_4 rb t then 1 else 0)
def aP15_3 (ra rb : ℕ) : ℤ := -(4) * N15_3 ra rb + u15 (82 + rb) + u15 (105 + ra)
def MP15_3 : ℤ := CaseSplit.mxr2 (aP15_3) 10 22
def N15_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (if c15_0 ra t && c15_5 rb t then 1 else 0)
def aP15_4 (ra rb : ℕ) : ℤ := -(4) * N15_4 ra rb + u15 (116 + rb) + u15 (145 + ra)
def MP15_4 : ℤ := CaseSplit.mxr2 (aP15_4) 10 28
def P15_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (if c15_1 ra t && c15_2 rb t then 1 else 0)
def C15_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (if c15_1 ra t && c15_2 rb t && c15_0 s t then 1 else 0)
def M15_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C15_5 ra rb) 10
def E15_5 : List ℕ := [57, 63, 93, 99, 102, 108, 136, 147, 172, 183, 188, 194]
def N15_5 (ra rb : ℕ) : ℤ := if E15_5.contains (ra * 17 + rb) = true then P15_5 ra rb - M15_5 ra rb else 0
def aP15_5 (ra rb : ℕ) : ℤ := -(4) * N15_5 ra rb + u15 (156 + rb) + u15 (173 + ra)
def MP15_5 : ℤ := CaseSplit.mxr2 (aP15_5) 12 16
def P15_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (if c15_1 ra t && c15_3 rb t then 1 else 0)
def C15_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (if c15_1 ra t && c15_3 rb t && c15_0 s t then 1 else 0)
def M15_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C15_6 ra rb) 10
def E15_6 : List ℕ := [31, 37, 73, 107, 113, 118, 144, 152, 194, 220, 228, 244]
def N15_6 (ra rb : ℕ) : ℤ := if E15_6.contains (ra * 19 + rb) = true then P15_6 ra rb - M15_6 ra rb else 0
def aP15_6 (ra rb : ℕ) : ℤ := -(4) * N15_6 ra rb + u15 (186 + rb) + u15 (205 + ra)
def MP15_6 : ℤ := CaseSplit.mxr2 (aP15_6) 12 18
def P15_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (if c15_1 ra t && c15_4 rb t then 1 else 0)
def C15_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (if c15_1 ra t && c15_4 rb t && c15_0 s t then 1 else 0)
def M15_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C15_7 ra rb) 10
def E15_7 : List ℕ := []
def N15_7 (ra rb : ℕ) : ℤ := if E15_7.contains (ra * 23 + rb) = true then P15_7 ra rb - M15_7 ra rb else 0
def aP15_7 (ra rb : ℕ) : ℤ := -(4) * N15_7 ra rb + u15 (218 + rb) + u15 (241 + ra)
def MP15_7 : ℤ := CaseSplit.mxr2 (aP15_7) 12 22
def P15_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (if c15_1 ra t && c15_5 rb t then 1 else 0)
def C15_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (if c15_1 ra t && c15_5 rb t && c15_0 s t then 1 else 0)
def M15_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C15_8 ra rb) 10
def E15_8 : List ℕ := [163, 279]
def N15_8 (ra rb : ℕ) : ℤ := if E15_8.contains (ra * 29 + rb) = true then P15_8 ra rb - M15_8 ra rb else 0
def aP15_8 (ra rb : ℕ) : ℤ := -(4) * N15_8 ra rb + u15 (254 + rb) + u15 (283 + ra)
def MP15_8 : ℤ := CaseSplit.mxr2 (aP15_8) 12 28
def N15_9 (_ra _rb : ℕ) : ℤ := 0
def aP15_9 (ra rb : ℕ) : ℤ := -(4) * N15_9 ra rb + u15 (296 + rb) + u15 (315 + ra)
def MP15_9 : ℤ := CaseSplit.mxr2 (aP15_9) 16 18
def N15_10 (_ra _rb : ℕ) : ℤ := 0
def aP15_10 (ra rb : ℕ) : ℤ := -(4) * N15_10 ra rb + u15 (332 + rb) + u15 (355 + ra)
def MP15_10 : ℤ := CaseSplit.mxr2 (aP15_10) 16 22
def N15_11 (_ra _rb : ℕ) : ℤ := 0
def aP15_11 (ra rb : ℕ) : ℤ := -(4) * N15_11 ra rb + u15 (372 + rb) + u15 (401 + ra)
def MP15_11 : ℤ := CaseSplit.mxr2 (aP15_11) 16 28
def N15_12 (_ra _rb : ℕ) : ℤ := 0
def aP15_12 (ra rb : ℕ) : ℤ := -(4) * N15_12 ra rb + u15 (418 + rb) + u15 (441 + ra)
def MP15_12 : ℤ := CaseSplit.mxr2 (aP15_12) 18 22
def N15_13 (_ra _rb : ℕ) : ℤ := 0
def aP15_13 (ra rb : ℕ) : ℤ := -(4) * N15_13 ra rb + u15 (460 + rb) + u15 (489 + ra)
def MP15_13 : ℤ := CaseSplit.mxr2 (aP15_13) 18 28
def N15_14 (_ra _rb : ℕ) : ℤ := 0
def aP15_14 (ra rb : ℕ) : ℤ := -(4) * N15_14 ra rb + u15 (508 + rb) + u15 (537 + ra)
def MP15_14 : ℤ := CaseSplit.mxr2 (aP15_14) 22 28

def rhs15 : ℤ := (∑ t ∈ Finset.range n15, w15 t) + 4 * (n15 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn15 : ∀ t, t < n15 → (0 : ℤ) ≤ w15 t := by decide
theorem plt15 : ∀ t, t < n15 → q15 t < 49 := by decide
theorem pfree15_5 : ∀ t, t < n15 → gb5 2 (q15 t) = false := by decide
theorem pfree15_7 : ∀ t, t < n15 → gb7 1 (q15 t) = false := by decide
theorem MSv15_0 : MS15_0 = 24 := by decide +kernel
theorem MSv15_1 : MS15_1 = 56 := by decide +kernel
theorem MSv15_2 : MS15_2 = 1 := by decide +kernel
theorem MSv15_3 : MS15_3 = 0 := by decide +kernel
theorem MSv15_4 : MS15_4 = 1 := by decide +kernel
theorem MSv15_5 : MS15_5 = 0 := by decide +kernel
theorem MPv15_0 : MP15_0 = 0 := by decide +kernel
theorem MPv15_1 : MP15_1 = 0 := by decide +kernel
theorem MPv15_2 : MP15_2 = 0 := by decide +kernel
theorem MPv15_3 : MP15_3 = 0 := by decide +kernel
theorem MPv15_4 : MP15_4 = 0 := by decide +kernel
theorem MPv15_5 : MP15_5 = 0 := by decide +kernel
theorem MPv15_6 : MP15_6 = 0 := by decide +kernel
theorem MPv15_7 : MP15_7 = 0 := by decide +kernel
theorem MPv15_8 : MP15_8 = 0 := by decide +kernel
theorem MPv15_9 : MP15_9 = 0 := by decide +kernel
theorem MPv15_10 : MP15_10 = 0 := by decide +kernel
theorem MPv15_11 : MP15_11 = 0 := by decide +kernel
theorem MPv15_12 : MP15_12 = 0 := by decide +kernel
theorem MPv15_13 : MP15_13 = 0 := by decide +kernel
theorem MPv15_14 : MP15_14 = 15 := by decide +kernel
theorem rhsv15 : rhs15 = 98 := by decide +kernel

/-- **The case-15 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/4.
    (Scaled by the common denominator 4: 97 < 98.) -/
theorem cert15 : MS15_0 + MS15_1 + MS15_2 + MS15_3 + MS15_4 + MS15_5 + MP15_0 + MP15_1 + MP15_2 + MP15_3 + MP15_4 + MP15_5 + MP15_6 + MP15_7 + MP15_8 + MP15_9 + MP15_10 + MP15_11 + MP15_12 + MP15_13 + MP15_14 < rhs15 := by
  rw [MSv15_0, MSv15_1, MSv15_2, MSv15_3, MSv15_4, MSv15_5, MPv15_0, MPv15_1, MPv15_2, MPv15_3, MPv15_4, MPv15_5, MPv15_6, MPv15_7, MPv15_8, MPv15_9, MPv15_10, MPv15_11, MPv15_12, MPv15_13, MPv15_14, rhsv15]
  decide

def Dg15 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := (if c15_0 r0 t then 1 else 0) + (if c15_1 r1 t then 1 else 0) + (if c15_2 r2 t then 1 else 0) + (if c15_3 r3 t then 1 else 0) + (if c15_4 r4 t then 1 else 0) + (if c15_5 r5 t then 1 else 0)
def Wl15_0 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c15_0 r0 t && c15_1 r1 t then 1 else 0
def Wl15_1 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c15_0 r0 t && c15_2 r2 t then 1 else 0
def Wl15_2 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c15_0 r0 t && c15_3 r3 t then 1 else 0
def Wl15_3 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c15_0 r0 t && c15_4 r4 t then 1 else 0
def Wl15_4 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c15_0 r0 t && c15_5 r5 t then 1 else 0
def Wl15_5 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c15_0 r0 t && c15_1 r1 t && c15_2 r2 t then 1 else 0
def Wl15_6 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c15_0 r0 t && c15_1 r1 t && c15_3 r3 t then 1 else 0
def Wl15_7 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c15_0 r0 t && c15_1 r1 t && c15_4 r4 t then 1 else 0
def Wl15_8 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c15_0 r0 t && c15_1 r1 t && c15_5 r5 t then 1 else 0
def Wl15_9 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c15_0 r0 t && !c15_1 r1 t && c15_2 r2 t && c15_3 r3 t then 1 else 0
def Wl15_10 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c15_0 r0 t && !c15_1 r1 t && c15_2 r2 t && c15_4 r4 t then 1 else 0
def Wl15_11 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c15_0 r0 t && !c15_1 r1 t && c15_2 r2 t && c15_5 r5 t then 1 else 0
def Wl15_12 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c15_0 r0 t && !c15_1 r1 t && !c15_2 r2 t && c15_3 r3 t && c15_4 r4 t then 1 else 0
def Wl15_13 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c15_0 r0 t && !c15_1 r1 t && !c15_2 r2 t && c15_3 r3 t && c15_5 r5 t then 1 else 0
def Wl15_14 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c15_0 r0 t && !c15_1 r1 t && !c15_2 r2 t && !c15_3 r3 t && c15_4 r4 t && c15_5 r5 t then 1 else 0

/-- **No configuration blocks the whole window in case 15.** -/
theorem nocov15 {r0 r1 r2 r3 r4 r5 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29)
    (hcov : ∀ t, t < n15 → (c15_0 r0 t || c15_1 r1 t || c15_2 r2 t || c15_3 r3 t || c15_4 r4 t || c15_5 r5 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n15, (1 : ℤ) + (Wl15_0 r0 r1 r2 r3 r4 r5 t + Wl15_1 r0 r1 r2 r3 r4 r5 t + Wl15_2 r0 r1 r2 r3 r4 r5 t + Wl15_3 r0 r1 r2 r3 r4 r5 t + Wl15_4 r0 r1 r2 r3 r4 r5 t + Wl15_5 r0 r1 r2 r3 r4 r5 t + Wl15_6 r0 r1 r2 r3 r4 r5 t + Wl15_7 r0 r1 r2 r3 r4 r5 t + Wl15_8 r0 r1 r2 r3 r4 r5 t + Wl15_9 r0 r1 r2 r3 r4 r5 t + Wl15_10 r0 r1 r2 r3 r4 r5 t + Wl15_11 r0 r1 r2 r3 r4 r5 t + Wl15_12 r0 r1 r2 r3 r4 r5 t + Wl15_13 r0 r1 r2 r3 r4 r5 t + Wl15_14 r0 r1 r2 r3 r4 r5 t) ≤ Dg15 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Wl15_0, Wl15_1, Wl15_2, Wl15_3, Wl15_4, Wl15_5, Wl15_6, Wl15_7, Wl15_8, Wl15_9, Wl15_10, Wl15_11, Wl15_12, Wl15_13, Wl15_14, Dg15]
    exact CaseSplit.lowest6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n15, (1 : ℤ) ≤ Dg15 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Dg15]
    exact CaseSplit.degpos6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n15 : ℤ) + ((∑ t ∈ Finset.range n15, Wl15_0 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n15, Wl15_1 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n15, Wl15_2 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n15, Wl15_3 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n15, Wl15_4 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n15, Wl15_5 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n15, Wl15_6 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n15, Wl15_7 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n15, Wl15_8 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n15, Wl15_9 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n15, Wl15_10 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n15, Wl15_11 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n15, Wl15_12 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n15, Wl15_13 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n15, Wl15_14 r0 r1 r2 r3 r4 r5 t)) ≤ ∑ t ∈ Finset.range n15, Dg15 r0 r1 r2 r3 r4 r5 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N15_0 r0 r1 ≤ ∑ t ∈ Finset.range n15, Wl15_0 r0 r1 r2 r3 r4 r5 t := by
    simp only [N15_0, Wl15_0, le_refl]
  have hn1 : N15_1 r0 r2 ≤ ∑ t ∈ Finset.range n15, Wl15_1 r0 r1 r2 r3 r4 r5 t := by
    simp only [N15_1, Wl15_1, le_refl]
  have hn2 : N15_2 r0 r3 ≤ ∑ t ∈ Finset.range n15, Wl15_2 r0 r1 r2 r3 r4 r5 t := by
    simp only [N15_2, Wl15_2, le_refl]
  have hn3 : N15_3 r0 r4 ≤ ∑ t ∈ Finset.range n15, Wl15_3 r0 r1 r2 r3 r4 r5 t := by
    simp only [N15_3, Wl15_3, le_refl]
  have hn4 : N15_4 r0 r5 ≤ ∑ t ∈ Finset.range n15, Wl15_4 r0 r1 r2 r3 r4 r5 t := by
    simp only [N15_4, Wl15_4, le_refl]
  have hn5 : N15_5 r1 r2 ≤ ∑ t ∈ Finset.range n15, Wl15_5 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n15, Wl15_5 r0 r1 r2 r3 r4 r5 t
        = (if c15_1 r1 t && c15_2 r2 t then (1:ℤ) else 0)
          - (if c15_1 r1 t && c15_2 r2 t && c15_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl15_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n15, Wl15_5 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl15_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n15, Wl15_5 r0 r1 r2 r3 r4 r5 t
        = P15_5 r1 r2 - C15_5 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P15_5, C15_5]
    have hm : C15_5 r1 r2 r0 ≤ M15_5 r1 r2 :=
      CaseSplit.le_mxr (C15_5 r1 r2) 10 r0 (by omega)
    simp only [N15_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N15_6 r1 r3 ≤ ∑ t ∈ Finset.range n15, Wl15_6 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n15, Wl15_6 r0 r1 r2 r3 r4 r5 t
        = (if c15_1 r1 t && c15_3 r3 t then (1:ℤ) else 0)
          - (if c15_1 r1 t && c15_3 r3 t && c15_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl15_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n15, Wl15_6 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl15_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n15, Wl15_6 r0 r1 r2 r3 r4 r5 t
        = P15_6 r1 r3 - C15_6 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P15_6, C15_6]
    have hm : C15_6 r1 r3 r0 ≤ M15_6 r1 r3 :=
      CaseSplit.le_mxr (C15_6 r1 r3) 10 r0 (by omega)
    simp only [N15_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N15_7 r1 r4 ≤ ∑ t ∈ Finset.range n15, Wl15_7 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n15, Wl15_7 r0 r1 r2 r3 r4 r5 t
        = (if c15_1 r1 t && c15_4 r4 t then (1:ℤ) else 0)
          - (if c15_1 r1 t && c15_4 r4 t && c15_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl15_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n15, Wl15_7 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl15_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n15, Wl15_7 r0 r1 r2 r3 r4 r5 t
        = P15_7 r1 r4 - C15_7 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P15_7, C15_7]
    have hm : C15_7 r1 r4 r0 ≤ M15_7 r1 r4 :=
      CaseSplit.le_mxr (C15_7 r1 r4) 10 r0 (by omega)
    simp only [N15_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N15_8 r1 r5 ≤ ∑ t ∈ Finset.range n15, Wl15_8 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n15, Wl15_8 r0 r1 r2 r3 r4 r5 t
        = (if c15_1 r1 t && c15_5 r5 t then (1:ℤ) else 0)
          - (if c15_1 r1 t && c15_5 r5 t && c15_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl15_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n15, Wl15_8 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl15_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n15, Wl15_8 r0 r1 r2 r3 r4 r5 t
        = P15_8 r1 r5 - C15_8 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P15_8, C15_8]
    have hm : C15_8 r1 r5 r0 ≤ M15_8 r1 r5 :=
      CaseSplit.le_mxr (C15_8 r1 r5) 10 r0 (by omega)
    simp only [N15_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N15_9 r2 r3 ≤ ∑ t ∈ Finset.range n15, Wl15_9 r0 r1 r2 r3 r4 r5 t := by
    simp only [N15_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl15_9]
    exact CaseSplit.ind_nonneg _
  have hn10 : N15_10 r2 r4 ≤ ∑ t ∈ Finset.range n15, Wl15_10 r0 r1 r2 r3 r4 r5 t := by
    simp only [N15_10]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl15_10]
    exact CaseSplit.ind_nonneg _
  have hn11 : N15_11 r2 r5 ≤ ∑ t ∈ Finset.range n15, Wl15_11 r0 r1 r2 r3 r4 r5 t := by
    simp only [N15_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl15_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N15_12 r3 r4 ≤ ∑ t ∈ Finset.range n15, Wl15_12 r0 r1 r2 r3 r4 r5 t := by
    simp only [N15_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl15_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N15_13 r3 r5 ≤ ∑ t ∈ Finset.range n15, Wl15_13 r0 r1 r2 r3 r4 r5 t := by
    simp only [N15_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl15_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N15_14 r4 r5 ≤ ∑ t ∈ Finset.range n15, Wl15_14 r0 r1 r2 r3 r4 r5 t := by
    simp only [N15_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl15_14]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n15, (w15 t + 4) * Dg15 r0 r1 r2 r3 r4 r5 t = S15_0 r0 + S15_1 r1 + S15_2 r2 + S15_3 r3 + S15_4 r4 + S15_5 r5 := by
    simp only [S15_0, S15_1, S15_2, S15_3, S15_4, S15_5, Dg15, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n15, (w15 t + 4) * Dg15 r0 r1 r2 r3 r4 r5 t
      = (∑ t ∈ Finset.range n15, w15 t * Dg15 r0 r1 r2 r3 r4 r5 t)
        + 4 * (∑ t ∈ Finset.range n15, Dg15 r0 r1 r2 r3 r4 r5 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n15, w15 t)
      ≤ ∑ t ∈ Finset.range n15, w15 t * Dg15 r0 r1 r2 r3 r4 r5 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg15 r0 r1 r2 r3 r4 r5 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w15 t := wnn15 t (Finset.mem_range.mp ht)
    calc w15 t = w15 t * 1 := (mul_one _).symm
      _ ≤ w15 t * Dg15 r0 r1 r2 r3 r4 r5 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS15_0 r0 + aS15_1 r1 + aS15_2 r2 + aS15_3 r3 + aS15_4 r4 + aS15_5 r5) + (aP15_0 r0 r1 + aP15_1 r0 r2 + aP15_2 r0 r3 + aP15_3 r0 r4 + aP15_4 r0 r5 + aP15_5 r1 r2 + aP15_6 r1 r3 + aP15_7 r1 r4 + aP15_8 r1 r5 + aP15_9 r2 r3 + aP15_10 r2 r4 + aP15_11 r2 r5 + aP15_12 r3 r4 + aP15_13 r3 r5 + aP15_14 r4 r5) = (S15_0 r0 + S15_1 r1 + S15_2 r2 + S15_3 r3 + S15_4 r4 + S15_5 r5) - 4 * (N15_0 r0 r1 + N15_1 r0 r2 + N15_2 r0 r3 + N15_3 r0 r4 + N15_4 r0 r5 + N15_5 r1 r2 + N15_6 r1 r3 + N15_7 r1 r4 + N15_8 r1 r5 + N15_9 r2 r3 + N15_10 r2 r4 + N15_11 r2 r5 + N15_12 r3 r4 + N15_13 r3 r5 + N15_14 r4 r5) := by
    simp only [aS15_0, aS15_1, aS15_2, aS15_3, aS15_4, aS15_5, aP15_0, aP15_1, aP15_2, aP15_3, aP15_4, aP15_5, aP15_6, aP15_7, aP15_8, aP15_9, aP15_10, aP15_11, aP15_12, aP15_13, aP15_14, L15_0, L15_1, L15_2, L15_3, L15_4, L15_5]
    ring
  have bS0 : aS15_0 r0 ≤ MS15_0 := CaseSplit.le_mxr (aS15_0) 10 r0 (by omega)
  have bS1 : aS15_1 r1 ≤ MS15_1 := CaseSplit.le_mxr (aS15_1) 12 r1 (by omega)
  have bS2 : aS15_2 r2 ≤ MS15_2 := CaseSplit.le_mxr (aS15_2) 16 r2 (by omega)
  have bS3 : aS15_3 r3 ≤ MS15_3 := CaseSplit.le_mxr (aS15_3) 18 r3 (by omega)
  have bS4 : aS15_4 r4 ≤ MS15_4 := CaseSplit.le_mxr (aS15_4) 22 r4 (by omega)
  have bS5 : aS15_5 r5 ≤ MS15_5 := CaseSplit.le_mxr (aS15_5) 28 r5 (by omega)
  have bP0 : aP15_0 r0 r1 ≤ MP15_0 := CaseSplit.le_mxr2 (aP15_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP15_1 r0 r2 ≤ MP15_1 := CaseSplit.le_mxr2 (aP15_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP15_2 r0 r3 ≤ MP15_2 := CaseSplit.le_mxr2 (aP15_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP15_3 r0 r4 ≤ MP15_3 := CaseSplit.le_mxr2 (aP15_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP15_4 r0 r5 ≤ MP15_4 := CaseSplit.le_mxr2 (aP15_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP15_5 r1 r2 ≤ MP15_5 := CaseSplit.le_mxr2 (aP15_5) 12 16 r1 r2 (by omega) (by omega)
  have bP6 : aP15_6 r1 r3 ≤ MP15_6 := CaseSplit.le_mxr2 (aP15_6) 12 18 r1 r3 (by omega) (by omega)
  have bP7 : aP15_7 r1 r4 ≤ MP15_7 := CaseSplit.le_mxr2 (aP15_7) 12 22 r1 r4 (by omega) (by omega)
  have bP8 : aP15_8 r1 r5 ≤ MP15_8 := CaseSplit.le_mxr2 (aP15_8) 12 28 r1 r5 (by omega) (by omega)
  have bP9 : aP15_9 r2 r3 ≤ MP15_9 := CaseSplit.le_mxr2 (aP15_9) 16 18 r2 r3 (by omega) (by omega)
  have bP10 : aP15_10 r2 r4 ≤ MP15_10 := CaseSplit.le_mxr2 (aP15_10) 16 22 r2 r4 (by omega) (by omega)
  have bP11 : aP15_11 r2 r5 ≤ MP15_11 := CaseSplit.le_mxr2 (aP15_11) 16 28 r2 r5 (by omega) (by omega)
  have bP12 : aP15_12 r3 r4 ≤ MP15_12 := CaseSplit.le_mxr2 (aP15_12) 18 22 r3 r4 (by omega) (by omega)
  have bP13 : aP15_13 r3 r5 ≤ MP15_13 := CaseSplit.le_mxr2 (aP15_13) 18 28 r3 r5 (by omega) (by omega)
  have bP14 : aP15_14 r4 r5 ≤ MP15_14 := CaseSplit.le_mxr2 (aP15_14) 22 28 r4 r5 (by omega) (by omega)
  have hrhs : rhs15 = (∑ t ∈ Finset.range n15, w15 t) + 4 * (n15 : ℤ) := rfl
  have hc := cert15
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, bS0, bS1, bS2, bS3, bS4, bS5, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14]

end IncCert29
