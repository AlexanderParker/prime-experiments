/-
INCREMENT-WIDTH CERTIFICATE, step 23->29, case 22 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_23_29.json, which re-derives every number
from the primes alone).

Machine 29, INCREMENT width 49 = F_2(23) + s_min(29) = 39 + 10,
held gears [5, 7] at phases [3, 1].  Free gears [11, 13, 17, 19, 23, 29].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 7.
-/
import IncCert29B

namespace IncCert29

/-! ### case 22: held gears at phases [3, 1] -/

def p22 : List ℕ := [2, 4, 9, 10, 15, 17, 20, 22, 24, 25, 27, 29, 30, 32, 34, 37, 39, 44, 45]
def q22 (t : ℕ) : ℕ := p22.getD t 0
def n22 : ℕ := 19
def yl22 : List ℤ := [1, 0, 1, 0, 1, 2, 4, 6, 4, 7, 7, 6, 4, 5, 4, 4, 3, 0, 0]
def w22 (t : ℕ) : ℤ := yl22.getD t 0
def ul22 : List ℤ := [0, (-2), (-3), (-2), 0, (-7), (-5), 0, (-2), 0, (-7), 0, (-2), 0, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0, (-2), (-2), (-2), 0, 0, (-2), (-2), (-2), 0, 1, 0, (-2), (-2), (-2), 0, 0, (-1), (-1), 0, 0, 0, 0, (-1), (-1), 0, 2, 0, (-1), 0, 0, (-1), (-1), (-1), (-1), 0, (-1), (-3), (-1), 0, (-1), 0, 0, 0, (-1), 0, (-1), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-2), 0, 0, 0, 0, 0, 12, 11, 10, 15, 14, 14, 11, 14, 10, 15, 15, 15, 15, 8, 13, 13, 15, (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), 13, 13, 11, 8, 7, 11, 11, 13, 13, 13, 13, 13, 13, 13, 12, 13, 13, 12, 13, (-13), (-13), (-13), (-13), (-13), (-13), (-13), (-13), (-13), (-13), (-13), (-13), (-13), 6, 0, 6, 1, 6, 6, 0, 6, 6, 6, 6, 5, (-1), 6, 0, 6, 6, 6, 6, 0, 6, 6, 6, (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-8), (-6), (-6), (-6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 2, 15, 3, 12, 8, 0, 13, 2, 15, 6, 0, 8, 0, 13, 3, 6, 4, 15, 12, 2, 7, 3, 6, 15, 3, 15, 0, 9, 6, 0, 10, 10, (-2), 10, 2, 0, 0, 0, 8, 0, 10, 8, 0, 6, 0, 8, 10, 0, 10, 5, 0]
def u22 (k : ℕ) : ℤ := ul22.getD k 0

def c22_0 (r t : ℕ) : Bool := gb11 r (q22 t)
def c22_1 (r t : ℕ) : Bool := gb13 r (q22 t)
def c22_2 (r t : ℕ) : Bool := gb17 r (q22 t)
def c22_3 (r t : ℕ) : Bool := gb19 r (q22 t)
def c22_4 (r t : ℕ) : Bool := gb23 r (q22 t)
def c22_5 (r t : ℕ) : Bool := gb29 r (q22 t)

def S22_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n22, (w22 t + 2) * (if c22_0 r t then 1 else 0)
def S22_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n22, (w22 t + 2) * (if c22_1 r t then 1 else 0)
def S22_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n22, (w22 t + 2) * (if c22_2 r t then 1 else 0)
def S22_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n22, (w22 t + 2) * (if c22_3 r t then 1 else 0)
def S22_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n22, (w22 t + 2) * (if c22_4 r t then 1 else 0)
def S22_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n22, (w22 t + 2) * (if c22_5 r t then 1 else 0)

def L22_0 (r : ℕ) : ℤ := u22 (13 + r) + u22 (41 + r) + u22 (71 + r) + u22 (105 + r) + u22 (145 + r)
def L22_1 (r : ℕ) : ℤ := u22 (0 + r) + u22 (173 + r) + u22 (205 + r) + u22 (241 + r) + u22 (283 + r)
def L22_2 (r : ℕ) : ℤ := u22 (24 + r) + u22 (156 + r) + u22 (315 + r) + u22 (355 + r) + u22 (401 + r)
def L22_3 (r : ℕ) : ℤ := u22 (52 + r) + u22 (186 + r) + u22 (296 + r) + u22 (441 + r) + u22 (489 + r)
def L22_4 (r : ℕ) : ℤ := u22 (82 + r) + u22 (218 + r) + u22 (332 + r) + u22 (418 + r) + u22 (537 + r)
def L22_5 (r : ℕ) : ℤ := u22 (116 + r) + u22 (254 + r) + u22 (372 + r) + u22 (460 + r) + u22 (508 + r)

def aS22_0 (r : ℕ) : ℤ := S22_0 r - L22_0 r
def MS22_0 : ℤ := CaseSplit.mxr (aS22_0) 10
def aS22_1 (r : ℕ) : ℤ := S22_1 r - L22_1 r
def MS22_1 : ℤ := CaseSplit.mxr (aS22_1) 12
def aS22_2 (r : ℕ) : ℤ := S22_2 r - L22_2 r
def MS22_2 : ℤ := CaseSplit.mxr (aS22_2) 16
def aS22_3 (r : ℕ) : ℤ := S22_3 r - L22_3 r
def MS22_3 : ℤ := CaseSplit.mxr (aS22_3) 18
def aS22_4 (r : ℕ) : ℤ := S22_4 r - L22_4 r
def MS22_4 : ℤ := CaseSplit.mxr (aS22_4) 22
def aS22_5 (r : ℕ) : ℤ := S22_5 r - L22_5 r
def MS22_5 : ℤ := CaseSplit.mxr (aS22_5) 28

def N22_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n22, (if c22_0 ra t && c22_1 rb t then 1 else 0)
def aP22_0 (ra rb : ℕ) : ℤ := -(2) * N22_0 ra rb + u22 (0 + rb) + u22 (13 + ra)
def MP22_0 : ℤ := CaseSplit.mxr2 (aP22_0) 10 12
def N22_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n22, (if c22_0 ra t && c22_2 rb t then 1 else 0)
def aP22_1 (ra rb : ℕ) : ℤ := -(2) * N22_1 ra rb + u22 (24 + rb) + u22 (41 + ra)
def MP22_1 : ℤ := CaseSplit.mxr2 (aP22_1) 10 16
def N22_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n22, (if c22_0 ra t && c22_3 rb t then 1 else 0)
def aP22_2 (ra rb : ℕ) : ℤ := -(2) * N22_2 ra rb + u22 (52 + rb) + u22 (71 + ra)
def MP22_2 : ℤ := CaseSplit.mxr2 (aP22_2) 10 18
def N22_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n22, (if c22_0 ra t && c22_4 rb t then 1 else 0)
def aP22_3 (ra rb : ℕ) : ℤ := -(2) * N22_3 ra rb + u22 (82 + rb) + u22 (105 + ra)
def MP22_3 : ℤ := CaseSplit.mxr2 (aP22_3) 10 22
def N22_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n22, (if c22_0 ra t && c22_5 rb t then 1 else 0)
def aP22_4 (ra rb : ℕ) : ℤ := -(2) * N22_4 ra rb + u22 (116 + rb) + u22 (145 + ra)
def MP22_4 : ℤ := CaseSplit.mxr2 (aP22_4) 10 28
def P22_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n22, (if c22_1 ra t && c22_2 rb t then 1 else 0)
def C22_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n22, (if c22_1 ra t && c22_2 rb t && c22_0 s t then 1 else 0)
def M22_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C22_5 ra rb) 10
def E22_5 : List ℕ := [21, 27, 43, 49, 68, 79, 122, 133, 158, 169, 174, 180]
def N22_5 (ra rb : ℕ) : ℤ := if E22_5.contains (ra * 17 + rb) = true then P22_5 ra rb - M22_5 ra rb else 0
def aP22_5 (ra rb : ℕ) : ℤ := -(2) * N22_5 ra rb + u22 (156 + rb) + u22 (173 + ra)
def MP22_5 : ℤ := CaseSplit.mxr2 (aP22_5) 12 16
def P22_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n22, (if c22_1 ra t && c22_3 rb t then 1 else 0)
def C22_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n22, (if c22_1 ra t && c22_3 rb t && c22_0 s t then 1 else 0)
def M22_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C22_6 ra rb) 10
def E22_6 : List ℕ := [1, 17, 51, 93, 104, 127, 138, 151, 172, 180, 214, 227]
def N22_6 (ra rb : ℕ) : ℤ := if E22_6.contains (ra * 19 + rb) = true then P22_6 ra rb - M22_6 ra rb else 0
def aP22_6 (ra rb : ℕ) : ℤ := -(2) * N22_6 ra rb + u22 (186 + rb) + u22 (205 + ra)
def MP22_6 : ℤ := CaseSplit.mxr2 (aP22_6) 12 18
def P22_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n22, (if c22_1 ra t && c22_4 rb t then 1 else 0)
def C22_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n22, (if c22_1 ra t && c22_4 rb t && c22_0 s t then 1 else 0)
def M22_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C22_7 ra rb) 10
def E22_7 : List ℕ := []
def N22_7 (ra rb : ℕ) : ℤ := if E22_7.contains (ra * 23 + rb) = true then P22_7 ra rb - M22_7 ra rb else 0
def aP22_7 (ra rb : ℕ) : ℤ := -(2) * N22_7 ra rb + u22 (218 + rb) + u22 (241 + ra)
def MP22_7 : ℤ := CaseSplit.mxr2 (aP22_7) 12 22
def P22_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n22, (if c22_1 ra t && c22_5 rb t then 1 else 0)
def C22_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n22, (if c22_1 ra t && c22_5 rb t && c22_0 s t then 1 else 0)
def M22_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C22_8 ra rb) 10
def E22_8 : List ℕ := []
def N22_8 (ra rb : ℕ) : ℤ := if E22_8.contains (ra * 29 + rb) = true then P22_8 ra rb - M22_8 ra rb else 0
def aP22_8 (ra rb : ℕ) : ℤ := -(2) * N22_8 ra rb + u22 (254 + rb) + u22 (283 + ra)
def MP22_8 : ℤ := CaseSplit.mxr2 (aP22_8) 12 28
def N22_9 (_ra _rb : ℕ) : ℤ := 0
def aP22_9 (ra rb : ℕ) : ℤ := -(2) * N22_9 ra rb + u22 (296 + rb) + u22 (315 + ra)
def MP22_9 : ℤ := CaseSplit.mxr2 (aP22_9) 16 18
def N22_10 (_ra _rb : ℕ) : ℤ := 0
def aP22_10 (ra rb : ℕ) : ℤ := -(2) * N22_10 ra rb + u22 (332 + rb) + u22 (355 + ra)
def MP22_10 : ℤ := CaseSplit.mxr2 (aP22_10) 16 22
def N22_11 (_ra _rb : ℕ) : ℤ := 0
def aP22_11 (ra rb : ℕ) : ℤ := -(2) * N22_11 ra rb + u22 (372 + rb) + u22 (401 + ra)
def MP22_11 : ℤ := CaseSplit.mxr2 (aP22_11) 16 28
def N22_12 (_ra _rb : ℕ) : ℤ := 0
def aP22_12 (ra rb : ℕ) : ℤ := -(2) * N22_12 ra rb + u22 (418 + rb) + u22 (441 + ra)
def MP22_12 : ℤ := CaseSplit.mxr2 (aP22_12) 18 22
def N22_13 (_ra _rb : ℕ) : ℤ := 0
def aP22_13 (ra rb : ℕ) : ℤ := -(2) * N22_13 ra rb + u22 (460 + rb) + u22 (489 + ra)
def MP22_13 : ℤ := CaseSplit.mxr2 (aP22_13) 18 28
def N22_14 (_ra _rb : ℕ) : ℤ := 0
def aP22_14 (ra rb : ℕ) : ℤ := -(2) * N22_14 ra rb + u22 (508 + rb) + u22 (537 + ra)
def MP22_14 : ℤ := CaseSplit.mxr2 (aP22_14) 22 28

def rhs22 : ℤ := (∑ t ∈ Finset.range n22, w22 t) + 2 * (n22 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn22 : ∀ t, t < n22 → (0 : ℤ) ≤ w22 t := by decide
theorem plt22 : ∀ t, t < n22 → q22 t < 49 := by decide
theorem pfree22_5 : ∀ t, t < n22 → gb5 3 (q22 t) = false := by decide
theorem pfree22_7 : ∀ t, t < n22 → gb7 1 (q22 t) = false := by decide
theorem MSv22_0 : MS22_0 = 18 := by decide +kernel
theorem MSv22_1 : MS22_1 = 53 := by decide +kernel
theorem MSv22_2 : MS22_2 = 0 := by decide +kernel
theorem MSv22_3 : MS22_3 = 0 := by decide +kernel
theorem MSv22_4 : MS22_4 = 0 := by decide +kernel
theorem MSv22_5 : MS22_5 = 0 := by decide +kernel
theorem MPv22_0 : MP22_0 = 0 := by decide +kernel
theorem MPv22_1 : MP22_1 = 0 := by decide +kernel
theorem MPv22_2 : MP22_2 = 0 := by decide +kernel
theorem MPv22_3 : MP22_3 = 0 := by decide +kernel
theorem MPv22_4 : MP22_4 = 0 := by decide +kernel
theorem MPv22_5 : MP22_5 = 0 := by decide +kernel
theorem MPv22_6 : MP22_6 = 0 := by decide +kernel
theorem MPv22_7 : MP22_7 = 0 := by decide +kernel
theorem MPv22_8 : MP22_8 = 0 := by decide +kernel
theorem MPv22_9 : MP22_9 = 0 := by decide +kernel
theorem MPv22_10 : MP22_10 = 0 := by decide +kernel
theorem MPv22_11 : MP22_11 = 0 := by decide +kernel
theorem MPv22_12 : MP22_12 = 0 := by decide +kernel
theorem MPv22_13 : MP22_13 = 0 := by decide +kernel
theorem MPv22_14 : MP22_14 = 25 := by decide +kernel
theorem rhsv22 : rhs22 = 97 := by decide +kernel

/-- **The case-22 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/7.
    (Scaled by the common denominator 7: 96 < 97.) -/
theorem cert22 : MS22_0 + MS22_1 + MS22_2 + MS22_3 + MS22_4 + MS22_5 + MP22_0 + MP22_1 + MP22_2 + MP22_3 + MP22_4 + MP22_5 + MP22_6 + MP22_7 + MP22_8 + MP22_9 + MP22_10 + MP22_11 + MP22_12 + MP22_13 + MP22_14 < rhs22 := by
  rw [MSv22_0, MSv22_1, MSv22_2, MSv22_3, MSv22_4, MSv22_5, MPv22_0, MPv22_1, MPv22_2, MPv22_3, MPv22_4, MPv22_5, MPv22_6, MPv22_7, MPv22_8, MPv22_9, MPv22_10, MPv22_11, MPv22_12, MPv22_13, MPv22_14, rhsv22]
  decide

def Dg22 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := (if c22_0 r0 t then 1 else 0) + (if c22_1 r1 t then 1 else 0) + (if c22_2 r2 t then 1 else 0) + (if c22_3 r3 t then 1 else 0) + (if c22_4 r4 t then 1 else 0) + (if c22_5 r5 t then 1 else 0)
def Wl22_0 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c22_0 r0 t && c22_1 r1 t then 1 else 0
def Wl22_1 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c22_0 r0 t && c22_2 r2 t then 1 else 0
def Wl22_2 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c22_0 r0 t && c22_3 r3 t then 1 else 0
def Wl22_3 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c22_0 r0 t && c22_4 r4 t then 1 else 0
def Wl22_4 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c22_0 r0 t && c22_5 r5 t then 1 else 0
def Wl22_5 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c22_0 r0 t && c22_1 r1 t && c22_2 r2 t then 1 else 0
def Wl22_6 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c22_0 r0 t && c22_1 r1 t && c22_3 r3 t then 1 else 0
def Wl22_7 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c22_0 r0 t && c22_1 r1 t && c22_4 r4 t then 1 else 0
def Wl22_8 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c22_0 r0 t && c22_1 r1 t && c22_5 r5 t then 1 else 0
def Wl22_9 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c22_0 r0 t && !c22_1 r1 t && c22_2 r2 t && c22_3 r3 t then 1 else 0
def Wl22_10 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c22_0 r0 t && !c22_1 r1 t && c22_2 r2 t && c22_4 r4 t then 1 else 0
def Wl22_11 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c22_0 r0 t && !c22_1 r1 t && c22_2 r2 t && c22_5 r5 t then 1 else 0
def Wl22_12 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c22_0 r0 t && !c22_1 r1 t && !c22_2 r2 t && c22_3 r3 t && c22_4 r4 t then 1 else 0
def Wl22_13 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c22_0 r0 t && !c22_1 r1 t && !c22_2 r2 t && c22_3 r3 t && c22_5 r5 t then 1 else 0
def Wl22_14 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c22_0 r0 t && !c22_1 r1 t && !c22_2 r2 t && !c22_3 r3 t && c22_4 r4 t && c22_5 r5 t then 1 else 0

/-- **No configuration blocks the whole window in case 22.** -/
theorem nocov22 {r0 r1 r2 r3 r4 r5 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29)
    (hcov : ∀ t, t < n22 → (c22_0 r0 t || c22_1 r1 t || c22_2 r2 t || c22_3 r3 t || c22_4 r4 t || c22_5 r5 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n22, (1 : ℤ) + (Wl22_0 r0 r1 r2 r3 r4 r5 t + Wl22_1 r0 r1 r2 r3 r4 r5 t + Wl22_2 r0 r1 r2 r3 r4 r5 t + Wl22_3 r0 r1 r2 r3 r4 r5 t + Wl22_4 r0 r1 r2 r3 r4 r5 t + Wl22_5 r0 r1 r2 r3 r4 r5 t + Wl22_6 r0 r1 r2 r3 r4 r5 t + Wl22_7 r0 r1 r2 r3 r4 r5 t + Wl22_8 r0 r1 r2 r3 r4 r5 t + Wl22_9 r0 r1 r2 r3 r4 r5 t + Wl22_10 r0 r1 r2 r3 r4 r5 t + Wl22_11 r0 r1 r2 r3 r4 r5 t + Wl22_12 r0 r1 r2 r3 r4 r5 t + Wl22_13 r0 r1 r2 r3 r4 r5 t + Wl22_14 r0 r1 r2 r3 r4 r5 t) ≤ Dg22 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Wl22_0, Wl22_1, Wl22_2, Wl22_3, Wl22_4, Wl22_5, Wl22_6, Wl22_7, Wl22_8, Wl22_9, Wl22_10, Wl22_11, Wl22_12, Wl22_13, Wl22_14, Dg22]
    exact CaseSplit.lowest6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n22, (1 : ℤ) ≤ Dg22 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Dg22]
    exact CaseSplit.degpos6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n22 : ℤ) + ((∑ t ∈ Finset.range n22, Wl22_0 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n22, Wl22_1 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n22, Wl22_2 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n22, Wl22_3 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n22, Wl22_4 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n22, Wl22_5 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n22, Wl22_6 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n22, Wl22_7 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n22, Wl22_8 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n22, Wl22_9 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n22, Wl22_10 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n22, Wl22_11 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n22, Wl22_12 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n22, Wl22_13 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n22, Wl22_14 r0 r1 r2 r3 r4 r5 t)) ≤ ∑ t ∈ Finset.range n22, Dg22 r0 r1 r2 r3 r4 r5 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N22_0 r0 r1 ≤ ∑ t ∈ Finset.range n22, Wl22_0 r0 r1 r2 r3 r4 r5 t := by
    simp only [N22_0, Wl22_0, le_refl]
  have hn1 : N22_1 r0 r2 ≤ ∑ t ∈ Finset.range n22, Wl22_1 r0 r1 r2 r3 r4 r5 t := by
    simp only [N22_1, Wl22_1, le_refl]
  have hn2 : N22_2 r0 r3 ≤ ∑ t ∈ Finset.range n22, Wl22_2 r0 r1 r2 r3 r4 r5 t := by
    simp only [N22_2, Wl22_2, le_refl]
  have hn3 : N22_3 r0 r4 ≤ ∑ t ∈ Finset.range n22, Wl22_3 r0 r1 r2 r3 r4 r5 t := by
    simp only [N22_3, Wl22_3, le_refl]
  have hn4 : N22_4 r0 r5 ≤ ∑ t ∈ Finset.range n22, Wl22_4 r0 r1 r2 r3 r4 r5 t := by
    simp only [N22_4, Wl22_4, le_refl]
  have hn5 : N22_5 r1 r2 ≤ ∑ t ∈ Finset.range n22, Wl22_5 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n22, Wl22_5 r0 r1 r2 r3 r4 r5 t
        = (if c22_1 r1 t && c22_2 r2 t then (1:ℤ) else 0)
          - (if c22_1 r1 t && c22_2 r2 t && c22_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl22_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n22, Wl22_5 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl22_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n22, Wl22_5 r0 r1 r2 r3 r4 r5 t
        = P22_5 r1 r2 - C22_5 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P22_5, C22_5]
    have hm : C22_5 r1 r2 r0 ≤ M22_5 r1 r2 :=
      CaseSplit.le_mxr (C22_5 r1 r2) 10 r0 (by omega)
    simp only [N22_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N22_6 r1 r3 ≤ ∑ t ∈ Finset.range n22, Wl22_6 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n22, Wl22_6 r0 r1 r2 r3 r4 r5 t
        = (if c22_1 r1 t && c22_3 r3 t then (1:ℤ) else 0)
          - (if c22_1 r1 t && c22_3 r3 t && c22_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl22_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n22, Wl22_6 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl22_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n22, Wl22_6 r0 r1 r2 r3 r4 r5 t
        = P22_6 r1 r3 - C22_6 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P22_6, C22_6]
    have hm : C22_6 r1 r3 r0 ≤ M22_6 r1 r3 :=
      CaseSplit.le_mxr (C22_6 r1 r3) 10 r0 (by omega)
    simp only [N22_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N22_7 r1 r4 ≤ ∑ t ∈ Finset.range n22, Wl22_7 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n22, Wl22_7 r0 r1 r2 r3 r4 r5 t
        = (if c22_1 r1 t && c22_4 r4 t then (1:ℤ) else 0)
          - (if c22_1 r1 t && c22_4 r4 t && c22_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl22_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n22, Wl22_7 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl22_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n22, Wl22_7 r0 r1 r2 r3 r4 r5 t
        = P22_7 r1 r4 - C22_7 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P22_7, C22_7]
    have hm : C22_7 r1 r4 r0 ≤ M22_7 r1 r4 :=
      CaseSplit.le_mxr (C22_7 r1 r4) 10 r0 (by omega)
    simp only [N22_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N22_8 r1 r5 ≤ ∑ t ∈ Finset.range n22, Wl22_8 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n22, Wl22_8 r0 r1 r2 r3 r4 r5 t
        = (if c22_1 r1 t && c22_5 r5 t then (1:ℤ) else 0)
          - (if c22_1 r1 t && c22_5 r5 t && c22_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl22_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n22, Wl22_8 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl22_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n22, Wl22_8 r0 r1 r2 r3 r4 r5 t
        = P22_8 r1 r5 - C22_8 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P22_8, C22_8]
    have hm : C22_8 r1 r5 r0 ≤ M22_8 r1 r5 :=
      CaseSplit.le_mxr (C22_8 r1 r5) 10 r0 (by omega)
    simp only [N22_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N22_9 r2 r3 ≤ ∑ t ∈ Finset.range n22, Wl22_9 r0 r1 r2 r3 r4 r5 t := by
    simp only [N22_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl22_9]
    exact CaseSplit.ind_nonneg _
  have hn10 : N22_10 r2 r4 ≤ ∑ t ∈ Finset.range n22, Wl22_10 r0 r1 r2 r3 r4 r5 t := by
    simp only [N22_10]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl22_10]
    exact CaseSplit.ind_nonneg _
  have hn11 : N22_11 r2 r5 ≤ ∑ t ∈ Finset.range n22, Wl22_11 r0 r1 r2 r3 r4 r5 t := by
    simp only [N22_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl22_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N22_12 r3 r4 ≤ ∑ t ∈ Finset.range n22, Wl22_12 r0 r1 r2 r3 r4 r5 t := by
    simp only [N22_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl22_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N22_13 r3 r5 ≤ ∑ t ∈ Finset.range n22, Wl22_13 r0 r1 r2 r3 r4 r5 t := by
    simp only [N22_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl22_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N22_14 r4 r5 ≤ ∑ t ∈ Finset.range n22, Wl22_14 r0 r1 r2 r3 r4 r5 t := by
    simp only [N22_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl22_14]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n22, (w22 t + 2) * Dg22 r0 r1 r2 r3 r4 r5 t = S22_0 r0 + S22_1 r1 + S22_2 r2 + S22_3 r3 + S22_4 r4 + S22_5 r5 := by
    simp only [S22_0, S22_1, S22_2, S22_3, S22_4, S22_5, Dg22, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n22, (w22 t + 2) * Dg22 r0 r1 r2 r3 r4 r5 t
      = (∑ t ∈ Finset.range n22, w22 t * Dg22 r0 r1 r2 r3 r4 r5 t)
        + 2 * (∑ t ∈ Finset.range n22, Dg22 r0 r1 r2 r3 r4 r5 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n22, w22 t)
      ≤ ∑ t ∈ Finset.range n22, w22 t * Dg22 r0 r1 r2 r3 r4 r5 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg22 r0 r1 r2 r3 r4 r5 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w22 t := wnn22 t (Finset.mem_range.mp ht)
    calc w22 t = w22 t * 1 := (mul_one _).symm
      _ ≤ w22 t * Dg22 r0 r1 r2 r3 r4 r5 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS22_0 r0 + aS22_1 r1 + aS22_2 r2 + aS22_3 r3 + aS22_4 r4 + aS22_5 r5) + (aP22_0 r0 r1 + aP22_1 r0 r2 + aP22_2 r0 r3 + aP22_3 r0 r4 + aP22_4 r0 r5 + aP22_5 r1 r2 + aP22_6 r1 r3 + aP22_7 r1 r4 + aP22_8 r1 r5 + aP22_9 r2 r3 + aP22_10 r2 r4 + aP22_11 r2 r5 + aP22_12 r3 r4 + aP22_13 r3 r5 + aP22_14 r4 r5) = (S22_0 r0 + S22_1 r1 + S22_2 r2 + S22_3 r3 + S22_4 r4 + S22_5 r5) - 2 * (N22_0 r0 r1 + N22_1 r0 r2 + N22_2 r0 r3 + N22_3 r0 r4 + N22_4 r0 r5 + N22_5 r1 r2 + N22_6 r1 r3 + N22_7 r1 r4 + N22_8 r1 r5 + N22_9 r2 r3 + N22_10 r2 r4 + N22_11 r2 r5 + N22_12 r3 r4 + N22_13 r3 r5 + N22_14 r4 r5) := by
    simp only [aS22_0, aS22_1, aS22_2, aS22_3, aS22_4, aS22_5, aP22_0, aP22_1, aP22_2, aP22_3, aP22_4, aP22_5, aP22_6, aP22_7, aP22_8, aP22_9, aP22_10, aP22_11, aP22_12, aP22_13, aP22_14, L22_0, L22_1, L22_2, L22_3, L22_4, L22_5]
    ring
  have bS0 : aS22_0 r0 ≤ MS22_0 := CaseSplit.le_mxr (aS22_0) 10 r0 (by omega)
  have bS1 : aS22_1 r1 ≤ MS22_1 := CaseSplit.le_mxr (aS22_1) 12 r1 (by omega)
  have bS2 : aS22_2 r2 ≤ MS22_2 := CaseSplit.le_mxr (aS22_2) 16 r2 (by omega)
  have bS3 : aS22_3 r3 ≤ MS22_3 := CaseSplit.le_mxr (aS22_3) 18 r3 (by omega)
  have bS4 : aS22_4 r4 ≤ MS22_4 := CaseSplit.le_mxr (aS22_4) 22 r4 (by omega)
  have bS5 : aS22_5 r5 ≤ MS22_5 := CaseSplit.le_mxr (aS22_5) 28 r5 (by omega)
  have bP0 : aP22_0 r0 r1 ≤ MP22_0 := CaseSplit.le_mxr2 (aP22_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP22_1 r0 r2 ≤ MP22_1 := CaseSplit.le_mxr2 (aP22_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP22_2 r0 r3 ≤ MP22_2 := CaseSplit.le_mxr2 (aP22_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP22_3 r0 r4 ≤ MP22_3 := CaseSplit.le_mxr2 (aP22_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP22_4 r0 r5 ≤ MP22_4 := CaseSplit.le_mxr2 (aP22_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP22_5 r1 r2 ≤ MP22_5 := CaseSplit.le_mxr2 (aP22_5) 12 16 r1 r2 (by omega) (by omega)
  have bP6 : aP22_6 r1 r3 ≤ MP22_6 := CaseSplit.le_mxr2 (aP22_6) 12 18 r1 r3 (by omega) (by omega)
  have bP7 : aP22_7 r1 r4 ≤ MP22_7 := CaseSplit.le_mxr2 (aP22_7) 12 22 r1 r4 (by omega) (by omega)
  have bP8 : aP22_8 r1 r5 ≤ MP22_8 := CaseSplit.le_mxr2 (aP22_8) 12 28 r1 r5 (by omega) (by omega)
  have bP9 : aP22_9 r2 r3 ≤ MP22_9 := CaseSplit.le_mxr2 (aP22_9) 16 18 r2 r3 (by omega) (by omega)
  have bP10 : aP22_10 r2 r4 ≤ MP22_10 := CaseSplit.le_mxr2 (aP22_10) 16 22 r2 r4 (by omega) (by omega)
  have bP11 : aP22_11 r2 r5 ≤ MP22_11 := CaseSplit.le_mxr2 (aP22_11) 16 28 r2 r5 (by omega) (by omega)
  have bP12 : aP22_12 r3 r4 ≤ MP22_12 := CaseSplit.le_mxr2 (aP22_12) 18 22 r3 r4 (by omega) (by omega)
  have bP13 : aP22_13 r3 r5 ≤ MP22_13 := CaseSplit.le_mxr2 (aP22_13) 18 28 r3 r5 (by omega) (by omega)
  have bP14 : aP22_14 r4 r5 ≤ MP22_14 := CaseSplit.le_mxr2 (aP22_14) 22 28 r4 r5 (by omega) (by omega)
  have hrhs : rhs22 = (∑ t ∈ Finset.range n22, w22 t) + 2 * (n22 : ℤ) := rfl
  have hc := cert22
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, bS0, bS1, bS2, bS3, bS4, bS5, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14]

end IncCert29
