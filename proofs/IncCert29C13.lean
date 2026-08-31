/-
INCREMENT-WIDTH CERTIFICATE, step 23->29, case 13 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_23_29.json, which re-derives every number
from the primes alone).

Machine 29, INCREMENT width 49 = F_2(23) + s_min(29) = 39 + 10,
held gears [5, 7] at phases [1, 6].  Free gears [11, 13, 17, 19, 23, 29].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 8.
-/
import IncCert29B

namespace IncCert29

/-! ### case 13: held gears at phases [1, 6] -/

def p13 : List ℕ := [1, 4, 6, 11, 12, 17, 19, 22, 24, 26, 27, 29, 31, 32, 34, 36, 39, 41, 46, 47]
def q13 (t : ℕ) : ℕ := p13.getD t 0
def n13 : ℕ := 20
def yl13 : List ℤ := [0, 2, 0, 0, 0, 1, 2, 5, 4, 4, 7, 8, 6, 3, 7, 5, 3, 2, 0, 1]
def w13 (t : ℕ) : ℤ := yl13.getD t 0
def ul13 : List ℤ := [0, (-1), 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, (-1), 0, (-1), 0, (-1), 0, 0, 0, 0, 0, 0, (-3), (-3), 0, (-3), (-3), (-3), (-3), (-3), 0, (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), 0, 3, 3, 0, 0, 0, 3, 3, 3, 0, 0, 0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 1, (-1), (-1), 0, 1, (-1), 0, (-1), (-1), 1, (-1), (-1), (-1), (-1), (-1), 1, (-1), (-1), (-1), 1, 0, 0, (-3), (-3), 0, (-3), (-3), (-3), 0, 0, (-3), (-3), (-3), (-3), (-3), (-3), 0, 0, (-3), 0, (-3), (-3), (-3), (-3), 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 11, 18, 18, 18, 12, 15, 11, 18, 18, 18, 17, 9, 15, 18, 18, 18, 18, (-18), (-18), (-18), (-18), (-18), (-18), (-18), (-18), (-18), (-18), (-18), (-18), (-18), 13, 11, 10, 15, 15, 15, 12, 14, 15, 12, 15, 15, 15, 15, 14, 15, 15, 8, 15, (-15), (-15), (-15), (-20), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), (-15), 5, 1, (-7), 5, 5, 5, 5, 5, 5, 0, 5, 5, 3, 5, 5, 5, 5, 0, 5, 0, 5, 2, 3, (-5), (-5), (-5), (-5), (-5), (-5), (-5), (-5), (-5), (-5), (-5), (-5), (-5), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 5, 16, 8, 3, 15, 3, 16, 7, 0, 7, 0, 16, 3, 6, 5, 3, 14, 3, 9, 5, 6, 16, 5, 16, 0, 10, 15, 3, 14, 14, 14, 14, 0, 5, 6, (-2), 14, 14, 7, 14, 0, 11, 6, 12, 14, 3, 14, 14, 5, 14, 0]
def u13 (k : ℕ) : ℤ := ul13.getD k 0

def c13_0 (r t : ℕ) : Bool := gb11 r (q13 t)
def c13_1 (r t : ℕ) : Bool := gb13 r (q13 t)
def c13_2 (r t : ℕ) : Bool := gb17 r (q13 t)
def c13_3 (r t : ℕ) : Bool := gb19 r (q13 t)
def c13_4 (r t : ℕ) : Bool := gb23 r (q13 t)
def c13_5 (r t : ℕ) : Bool := gb29 r (q13 t)

def S13_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (w13 t + 3) * (if c13_0 r t then 1 else 0)
def S13_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (w13 t + 3) * (if c13_1 r t then 1 else 0)
def S13_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (w13 t + 3) * (if c13_2 r t then 1 else 0)
def S13_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (w13 t + 3) * (if c13_3 r t then 1 else 0)
def S13_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (w13 t + 3) * (if c13_4 r t then 1 else 0)
def S13_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (w13 t + 3) * (if c13_5 r t then 1 else 0)

def L13_0 (r : ℕ) : ℤ := u13 (13 + r) + u13 (41 + r) + u13 (71 + r) + u13 (105 + r) + u13 (145 + r)
def L13_1 (r : ℕ) : ℤ := u13 (0 + r) + u13 (173 + r) + u13 (205 + r) + u13 (241 + r) + u13 (283 + r)
def L13_2 (r : ℕ) : ℤ := u13 (24 + r) + u13 (156 + r) + u13 (315 + r) + u13 (355 + r) + u13 (401 + r)
def L13_3 (r : ℕ) : ℤ := u13 (52 + r) + u13 (186 + r) + u13 (296 + r) + u13 (441 + r) + u13 (489 + r)
def L13_4 (r : ℕ) : ℤ := u13 (82 + r) + u13 (218 + r) + u13 (332 + r) + u13 (418 + r) + u13 (537 + r)
def L13_5 (r : ℕ) : ℤ := u13 (116 + r) + u13 (254 + r) + u13 (372 + r) + u13 (460 + r) + u13 (508 + r)

def aS13_0 (r : ℕ) : ℤ := S13_0 r - L13_0 r
def MS13_0 : ℤ := CaseSplit.mxr (aS13_0) 10
def aS13_1 (r : ℕ) : ℤ := S13_1 r - L13_1 r
def MS13_1 : ℤ := CaseSplit.mxr (aS13_1) 12
def aS13_2 (r : ℕ) : ℤ := S13_2 r - L13_2 r
def MS13_2 : ℤ := CaseSplit.mxr (aS13_2) 16
def aS13_3 (r : ℕ) : ℤ := S13_3 r - L13_3 r
def MS13_3 : ℤ := CaseSplit.mxr (aS13_3) 18
def aS13_4 (r : ℕ) : ℤ := S13_4 r - L13_4 r
def MS13_4 : ℤ := CaseSplit.mxr (aS13_4) 22
def aS13_5 (r : ℕ) : ℤ := S13_5 r - L13_5 r
def MS13_5 : ℤ := CaseSplit.mxr (aS13_5) 28

def N13_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (if c13_0 ra t && c13_1 rb t then 1 else 0)
def aP13_0 (ra rb : ℕ) : ℤ := -(3) * N13_0 ra rb + u13 (0 + rb) + u13 (13 + ra)
def MP13_0 : ℤ := CaseSplit.mxr2 (aP13_0) 10 12
def N13_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (if c13_0 ra t && c13_2 rb t then 1 else 0)
def aP13_1 (ra rb : ℕ) : ℤ := -(3) * N13_1 ra rb + u13 (24 + rb) + u13 (41 + ra)
def MP13_1 : ℤ := CaseSplit.mxr2 (aP13_1) 10 16
def N13_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (if c13_0 ra t && c13_3 rb t then 1 else 0)
def aP13_2 (ra rb : ℕ) : ℤ := -(3) * N13_2 ra rb + u13 (52 + rb) + u13 (71 + ra)
def MP13_2 : ℤ := CaseSplit.mxr2 (aP13_2) 10 18
def N13_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (if c13_0 ra t && c13_4 rb t then 1 else 0)
def aP13_3 (ra rb : ℕ) : ℤ := -(3) * N13_3 ra rb + u13 (82 + rb) + u13 (105 + ra)
def MP13_3 : ℤ := CaseSplit.mxr2 (aP13_3) 10 22
def N13_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (if c13_0 ra t && c13_5 rb t then 1 else 0)
def aP13_4 (ra rb : ℕ) : ℤ := -(3) * N13_4 ra rb + u13 (116 + rb) + u13 (145 + ra)
def MP13_4 : ℤ := CaseSplit.mxr2 (aP13_4) 10 28
def P13_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (if c13_1 ra t && c13_2 rb t then 1 else 0)
def C13_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (if c13_1 ra t && c13_2 rb t && c13_0 s t then 1 else 0)
def M13_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C13_5 ra rb) 10
def E13_5 : List ℕ := [7, 13, 43, 49, 86, 97, 122, 133, 138, 144, 206, 212]
def N13_5 (ra rb : ℕ) : ℤ := if E13_5.contains (ra * 17 + rb) = true then P13_5 ra rb - M13_5 ra rb else 0
def aP13_5 (ra rb : ℕ) : ℤ := -(3) * N13_5 ra rb + u13 (156 + rb) + u13 (173 + ra)
def MP13_5 : ℤ := CaseSplit.mxr2 (aP13_5) 12 16
def P13_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (if c13_1 ra t && c13_3 rb t then 1 else 0)
def C13_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (if c13_1 ra t && c13_3 rb t && c13_0 s t then 1 else 0)
def M13_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C13_6 ra rb) 10
def E13_6 : List ℕ := [11, 53, 64, 87, 98, 111, 140, 151, 174, 187, 224, 227]
def N13_6 (ra rb : ℕ) : ℤ := if E13_6.contains (ra * 19 + rb) = true then P13_6 ra rb - M13_6 ra rb else 0
def aP13_6 (ra rb : ℕ) : ℤ := -(3) * N13_6 ra rb + u13 (186 + rb) + u13 (205 + ra)
def MP13_6 : ℤ := CaseSplit.mxr2 (aP13_6) 12 18
def P13_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (if c13_1 ra t && c13_4 rb t then 1 else 0)
def C13_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (if c13_1 ra t && c13_4 rb t && c13_0 s t then 1 else 0)
def M13_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C13_7 ra rb) 10
def E13_7 : List ℕ := []
def N13_7 (ra rb : ℕ) : ℤ := if E13_7.contains (ra * 23 + rb) = true then P13_7 ra rb - M13_7 ra rb else 0
def aP13_7 (ra rb : ℕ) : ℤ := -(3) * N13_7 ra rb + u13 (218 + rb) + u13 (241 + ra)
def MP13_7 : ℤ := CaseSplit.mxr2 (aP13_7) 12 22
def P13_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (if c13_1 ra t && c13_5 rb t then 1 else 0)
def C13_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (if c13_1 ra t && c13_5 rb t && c13_0 s t then 1 else 0)
def M13_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C13_8 ra rb) 10
def E13_8 : List ℕ := []
def N13_8 (ra rb : ℕ) : ℤ := if E13_8.contains (ra * 29 + rb) = true then P13_8 ra rb - M13_8 ra rb else 0
def aP13_8 (ra rb : ℕ) : ℤ := -(3) * N13_8 ra rb + u13 (254 + rb) + u13 (283 + ra)
def MP13_8 : ℤ := CaseSplit.mxr2 (aP13_8) 12 28
def N13_9 (_ra _rb : ℕ) : ℤ := 0
def aP13_9 (ra rb : ℕ) : ℤ := -(3) * N13_9 ra rb + u13 (296 + rb) + u13 (315 + ra)
def MP13_9 : ℤ := CaseSplit.mxr2 (aP13_9) 16 18
def N13_10 (_ra _rb : ℕ) : ℤ := 0
def aP13_10 (ra rb : ℕ) : ℤ := -(3) * N13_10 ra rb + u13 (332 + rb) + u13 (355 + ra)
def MP13_10 : ℤ := CaseSplit.mxr2 (aP13_10) 16 22
def N13_11 (_ra _rb : ℕ) : ℤ := 0
def aP13_11 (ra rb : ℕ) : ℤ := -(3) * N13_11 ra rb + u13 (372 + rb) + u13 (401 + ra)
def MP13_11 : ℤ := CaseSplit.mxr2 (aP13_11) 16 28
def N13_12 (_ra _rb : ℕ) : ℤ := 0
def aP13_12 (ra rb : ℕ) : ℤ := -(3) * N13_12 ra rb + u13 (418 + rb) + u13 (441 + ra)
def MP13_12 : ℤ := CaseSplit.mxr2 (aP13_12) 18 22
def N13_13 (_ra _rb : ℕ) : ℤ := 0
def aP13_13 (ra rb : ℕ) : ℤ := -(3) * N13_13 ra rb + u13 (460 + rb) + u13 (489 + ra)
def MP13_13 : ℤ := CaseSplit.mxr2 (aP13_13) 18 28
def N13_14 (_ra _rb : ℕ) : ℤ := 0
def aP13_14 (ra rb : ℕ) : ℤ := -(3) * N13_14 ra rb + u13 (508 + rb) + u13 (537 + ra)
def MP13_14 : ℤ := CaseSplit.mxr2 (aP13_14) 22 28

def rhs13 : ℤ := (∑ t ∈ Finset.range n13, w13 t) + 3 * (n13 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn13 : ∀ t, t < n13 → (0 : ℤ) ≤ w13 t := by decide
theorem plt13 : ∀ t, t < n13 → q13 t < 49 := by decide
theorem pfree13_5 : ∀ t, t < n13 → gb5 1 (q13 t) = false := by decide
theorem pfree13_7 : ∀ t, t < n13 → gb7 6 (q13 t) = false := by decide
theorem MSv13_0 : MS13_0 = 22 := by decide +kernel
theorem MSv13_1 : MS13_1 = 60 := by decide +kernel
theorem MSv13_2 : MS13_2 = 2 := by decide +kernel
theorem MSv13_3 : MS13_3 = 1 := by decide +kernel
theorem MSv13_4 : MS13_4 = 1 := by decide +kernel
theorem MSv13_5 : MS13_5 = 1 := by decide +kernel
theorem MPv13_0 : MP13_0 = 0 := by decide +kernel
theorem MPv13_1 : MP13_1 = 0 := by decide +kernel
theorem MPv13_2 : MP13_2 = 0 := by decide +kernel
theorem MPv13_3 : MP13_3 = 0 := by decide +kernel
theorem MPv13_4 : MP13_4 = 0 := by decide +kernel
theorem MPv13_5 : MP13_5 = 0 := by decide +kernel
theorem MPv13_6 : MP13_6 = 0 := by decide +kernel
theorem MPv13_7 : MP13_7 = 0 := by decide +kernel
theorem MPv13_8 : MP13_8 = 0 := by decide +kernel
theorem MPv13_9 : MP13_9 = 0 := by decide +kernel
theorem MPv13_10 : MP13_10 = 0 := by decide +kernel
theorem MPv13_11 : MP13_11 = 0 := by decide +kernel
theorem MPv13_12 : MP13_12 = 0 := by decide +kernel
theorem MPv13_13 : MP13_13 = 0 := by decide +kernel
theorem MPv13_14 : MP13_14 = 30 := by decide +kernel
theorem rhsv13 : rhs13 = 120 := by decide +kernel

/-- **The case-13 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 3/8.
    (Scaled by the common denominator 8: 117 < 120.) -/
theorem cert13 : MS13_0 + MS13_1 + MS13_2 + MS13_3 + MS13_4 + MS13_5 + MP13_0 + MP13_1 + MP13_2 + MP13_3 + MP13_4 + MP13_5 + MP13_6 + MP13_7 + MP13_8 + MP13_9 + MP13_10 + MP13_11 + MP13_12 + MP13_13 + MP13_14 < rhs13 := by
  rw [MSv13_0, MSv13_1, MSv13_2, MSv13_3, MSv13_4, MSv13_5, MPv13_0, MPv13_1, MPv13_2, MPv13_3, MPv13_4, MPv13_5, MPv13_6, MPv13_7, MPv13_8, MPv13_9, MPv13_10, MPv13_11, MPv13_12, MPv13_13, MPv13_14, rhsv13]
  decide

def Dg13 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := (if c13_0 r0 t then 1 else 0) + (if c13_1 r1 t then 1 else 0) + (if c13_2 r2 t then 1 else 0) + (if c13_3 r3 t then 1 else 0) + (if c13_4 r4 t then 1 else 0) + (if c13_5 r5 t then 1 else 0)
def Wl13_0 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c13_0 r0 t && c13_1 r1 t then 1 else 0
def Wl13_1 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c13_0 r0 t && c13_2 r2 t then 1 else 0
def Wl13_2 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c13_0 r0 t && c13_3 r3 t then 1 else 0
def Wl13_3 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c13_0 r0 t && c13_4 r4 t then 1 else 0
def Wl13_4 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c13_0 r0 t && c13_5 r5 t then 1 else 0
def Wl13_5 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c13_0 r0 t && c13_1 r1 t && c13_2 r2 t then 1 else 0
def Wl13_6 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c13_0 r0 t && c13_1 r1 t && c13_3 r3 t then 1 else 0
def Wl13_7 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c13_0 r0 t && c13_1 r1 t && c13_4 r4 t then 1 else 0
def Wl13_8 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c13_0 r0 t && c13_1 r1 t && c13_5 r5 t then 1 else 0
def Wl13_9 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c13_0 r0 t && !c13_1 r1 t && c13_2 r2 t && c13_3 r3 t then 1 else 0
def Wl13_10 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c13_0 r0 t && !c13_1 r1 t && c13_2 r2 t && c13_4 r4 t then 1 else 0
def Wl13_11 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c13_0 r0 t && !c13_1 r1 t && c13_2 r2 t && c13_5 r5 t then 1 else 0
def Wl13_12 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c13_0 r0 t && !c13_1 r1 t && !c13_2 r2 t && c13_3 r3 t && c13_4 r4 t then 1 else 0
def Wl13_13 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c13_0 r0 t && !c13_1 r1 t && !c13_2 r2 t && c13_3 r3 t && c13_5 r5 t then 1 else 0
def Wl13_14 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c13_0 r0 t && !c13_1 r1 t && !c13_2 r2 t && !c13_3 r3 t && c13_4 r4 t && c13_5 r5 t then 1 else 0

/-- **No configuration blocks the whole window in case 13.** -/
theorem nocov13 {r0 r1 r2 r3 r4 r5 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29)
    (hcov : ∀ t, t < n13 → (c13_0 r0 t || c13_1 r1 t || c13_2 r2 t || c13_3 r3 t || c13_4 r4 t || c13_5 r5 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n13, (1 : ℤ) + (Wl13_0 r0 r1 r2 r3 r4 r5 t + Wl13_1 r0 r1 r2 r3 r4 r5 t + Wl13_2 r0 r1 r2 r3 r4 r5 t + Wl13_3 r0 r1 r2 r3 r4 r5 t + Wl13_4 r0 r1 r2 r3 r4 r5 t + Wl13_5 r0 r1 r2 r3 r4 r5 t + Wl13_6 r0 r1 r2 r3 r4 r5 t + Wl13_7 r0 r1 r2 r3 r4 r5 t + Wl13_8 r0 r1 r2 r3 r4 r5 t + Wl13_9 r0 r1 r2 r3 r4 r5 t + Wl13_10 r0 r1 r2 r3 r4 r5 t + Wl13_11 r0 r1 r2 r3 r4 r5 t + Wl13_12 r0 r1 r2 r3 r4 r5 t + Wl13_13 r0 r1 r2 r3 r4 r5 t + Wl13_14 r0 r1 r2 r3 r4 r5 t) ≤ Dg13 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Wl13_0, Wl13_1, Wl13_2, Wl13_3, Wl13_4, Wl13_5, Wl13_6, Wl13_7, Wl13_8, Wl13_9, Wl13_10, Wl13_11, Wl13_12, Wl13_13, Wl13_14, Dg13]
    exact CaseSplit.lowest6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n13, (1 : ℤ) ≤ Dg13 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Dg13]
    exact CaseSplit.degpos6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n13 : ℤ) + ((∑ t ∈ Finset.range n13, Wl13_0 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n13, Wl13_1 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n13, Wl13_2 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n13, Wl13_3 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n13, Wl13_4 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n13, Wl13_5 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n13, Wl13_6 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n13, Wl13_7 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n13, Wl13_8 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n13, Wl13_9 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n13, Wl13_10 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n13, Wl13_11 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n13, Wl13_12 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n13, Wl13_13 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n13, Wl13_14 r0 r1 r2 r3 r4 r5 t)) ≤ ∑ t ∈ Finset.range n13, Dg13 r0 r1 r2 r3 r4 r5 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N13_0 r0 r1 ≤ ∑ t ∈ Finset.range n13, Wl13_0 r0 r1 r2 r3 r4 r5 t := by
    simp only [N13_0, Wl13_0, le_refl]
  have hn1 : N13_1 r0 r2 ≤ ∑ t ∈ Finset.range n13, Wl13_1 r0 r1 r2 r3 r4 r5 t := by
    simp only [N13_1, Wl13_1, le_refl]
  have hn2 : N13_2 r0 r3 ≤ ∑ t ∈ Finset.range n13, Wl13_2 r0 r1 r2 r3 r4 r5 t := by
    simp only [N13_2, Wl13_2, le_refl]
  have hn3 : N13_3 r0 r4 ≤ ∑ t ∈ Finset.range n13, Wl13_3 r0 r1 r2 r3 r4 r5 t := by
    simp only [N13_3, Wl13_3, le_refl]
  have hn4 : N13_4 r0 r5 ≤ ∑ t ∈ Finset.range n13, Wl13_4 r0 r1 r2 r3 r4 r5 t := by
    simp only [N13_4, Wl13_4, le_refl]
  have hn5 : N13_5 r1 r2 ≤ ∑ t ∈ Finset.range n13, Wl13_5 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n13, Wl13_5 r0 r1 r2 r3 r4 r5 t
        = (if c13_1 r1 t && c13_2 r2 t then (1:ℤ) else 0)
          - (if c13_1 r1 t && c13_2 r2 t && c13_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl13_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n13, Wl13_5 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl13_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n13, Wl13_5 r0 r1 r2 r3 r4 r5 t
        = P13_5 r1 r2 - C13_5 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P13_5, C13_5]
    have hm : C13_5 r1 r2 r0 ≤ M13_5 r1 r2 :=
      CaseSplit.le_mxr (C13_5 r1 r2) 10 r0 (by omega)
    simp only [N13_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N13_6 r1 r3 ≤ ∑ t ∈ Finset.range n13, Wl13_6 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n13, Wl13_6 r0 r1 r2 r3 r4 r5 t
        = (if c13_1 r1 t && c13_3 r3 t then (1:ℤ) else 0)
          - (if c13_1 r1 t && c13_3 r3 t && c13_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl13_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n13, Wl13_6 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl13_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n13, Wl13_6 r0 r1 r2 r3 r4 r5 t
        = P13_6 r1 r3 - C13_6 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P13_6, C13_6]
    have hm : C13_6 r1 r3 r0 ≤ M13_6 r1 r3 :=
      CaseSplit.le_mxr (C13_6 r1 r3) 10 r0 (by omega)
    simp only [N13_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N13_7 r1 r4 ≤ ∑ t ∈ Finset.range n13, Wl13_7 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n13, Wl13_7 r0 r1 r2 r3 r4 r5 t
        = (if c13_1 r1 t && c13_4 r4 t then (1:ℤ) else 0)
          - (if c13_1 r1 t && c13_4 r4 t && c13_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl13_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n13, Wl13_7 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl13_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n13, Wl13_7 r0 r1 r2 r3 r4 r5 t
        = P13_7 r1 r4 - C13_7 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P13_7, C13_7]
    have hm : C13_7 r1 r4 r0 ≤ M13_7 r1 r4 :=
      CaseSplit.le_mxr (C13_7 r1 r4) 10 r0 (by omega)
    simp only [N13_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N13_8 r1 r5 ≤ ∑ t ∈ Finset.range n13, Wl13_8 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n13, Wl13_8 r0 r1 r2 r3 r4 r5 t
        = (if c13_1 r1 t && c13_5 r5 t then (1:ℤ) else 0)
          - (if c13_1 r1 t && c13_5 r5 t && c13_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl13_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n13, Wl13_8 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl13_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n13, Wl13_8 r0 r1 r2 r3 r4 r5 t
        = P13_8 r1 r5 - C13_8 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P13_8, C13_8]
    have hm : C13_8 r1 r5 r0 ≤ M13_8 r1 r5 :=
      CaseSplit.le_mxr (C13_8 r1 r5) 10 r0 (by omega)
    simp only [N13_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N13_9 r2 r3 ≤ ∑ t ∈ Finset.range n13, Wl13_9 r0 r1 r2 r3 r4 r5 t := by
    simp only [N13_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl13_9]
    exact CaseSplit.ind_nonneg _
  have hn10 : N13_10 r2 r4 ≤ ∑ t ∈ Finset.range n13, Wl13_10 r0 r1 r2 r3 r4 r5 t := by
    simp only [N13_10]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl13_10]
    exact CaseSplit.ind_nonneg _
  have hn11 : N13_11 r2 r5 ≤ ∑ t ∈ Finset.range n13, Wl13_11 r0 r1 r2 r3 r4 r5 t := by
    simp only [N13_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl13_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N13_12 r3 r4 ≤ ∑ t ∈ Finset.range n13, Wl13_12 r0 r1 r2 r3 r4 r5 t := by
    simp only [N13_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl13_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N13_13 r3 r5 ≤ ∑ t ∈ Finset.range n13, Wl13_13 r0 r1 r2 r3 r4 r5 t := by
    simp only [N13_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl13_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N13_14 r4 r5 ≤ ∑ t ∈ Finset.range n13, Wl13_14 r0 r1 r2 r3 r4 r5 t := by
    simp only [N13_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl13_14]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n13, (w13 t + 3) * Dg13 r0 r1 r2 r3 r4 r5 t = S13_0 r0 + S13_1 r1 + S13_2 r2 + S13_3 r3 + S13_4 r4 + S13_5 r5 := by
    simp only [S13_0, S13_1, S13_2, S13_3, S13_4, S13_5, Dg13, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n13, (w13 t + 3) * Dg13 r0 r1 r2 r3 r4 r5 t
      = (∑ t ∈ Finset.range n13, w13 t * Dg13 r0 r1 r2 r3 r4 r5 t)
        + 3 * (∑ t ∈ Finset.range n13, Dg13 r0 r1 r2 r3 r4 r5 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n13, w13 t)
      ≤ ∑ t ∈ Finset.range n13, w13 t * Dg13 r0 r1 r2 r3 r4 r5 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg13 r0 r1 r2 r3 r4 r5 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w13 t := wnn13 t (Finset.mem_range.mp ht)
    calc w13 t = w13 t * 1 := (mul_one _).symm
      _ ≤ w13 t * Dg13 r0 r1 r2 r3 r4 r5 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS13_0 r0 + aS13_1 r1 + aS13_2 r2 + aS13_3 r3 + aS13_4 r4 + aS13_5 r5) + (aP13_0 r0 r1 + aP13_1 r0 r2 + aP13_2 r0 r3 + aP13_3 r0 r4 + aP13_4 r0 r5 + aP13_5 r1 r2 + aP13_6 r1 r3 + aP13_7 r1 r4 + aP13_8 r1 r5 + aP13_9 r2 r3 + aP13_10 r2 r4 + aP13_11 r2 r5 + aP13_12 r3 r4 + aP13_13 r3 r5 + aP13_14 r4 r5) = (S13_0 r0 + S13_1 r1 + S13_2 r2 + S13_3 r3 + S13_4 r4 + S13_5 r5) - 3 * (N13_0 r0 r1 + N13_1 r0 r2 + N13_2 r0 r3 + N13_3 r0 r4 + N13_4 r0 r5 + N13_5 r1 r2 + N13_6 r1 r3 + N13_7 r1 r4 + N13_8 r1 r5 + N13_9 r2 r3 + N13_10 r2 r4 + N13_11 r2 r5 + N13_12 r3 r4 + N13_13 r3 r5 + N13_14 r4 r5) := by
    simp only [aS13_0, aS13_1, aS13_2, aS13_3, aS13_4, aS13_5, aP13_0, aP13_1, aP13_2, aP13_3, aP13_4, aP13_5, aP13_6, aP13_7, aP13_8, aP13_9, aP13_10, aP13_11, aP13_12, aP13_13, aP13_14, L13_0, L13_1, L13_2, L13_3, L13_4, L13_5]
    ring
  have bS0 : aS13_0 r0 ≤ MS13_0 := CaseSplit.le_mxr (aS13_0) 10 r0 (by omega)
  have bS1 : aS13_1 r1 ≤ MS13_1 := CaseSplit.le_mxr (aS13_1) 12 r1 (by omega)
  have bS2 : aS13_2 r2 ≤ MS13_2 := CaseSplit.le_mxr (aS13_2) 16 r2 (by omega)
  have bS3 : aS13_3 r3 ≤ MS13_3 := CaseSplit.le_mxr (aS13_3) 18 r3 (by omega)
  have bS4 : aS13_4 r4 ≤ MS13_4 := CaseSplit.le_mxr (aS13_4) 22 r4 (by omega)
  have bS5 : aS13_5 r5 ≤ MS13_5 := CaseSplit.le_mxr (aS13_5) 28 r5 (by omega)
  have bP0 : aP13_0 r0 r1 ≤ MP13_0 := CaseSplit.le_mxr2 (aP13_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP13_1 r0 r2 ≤ MP13_1 := CaseSplit.le_mxr2 (aP13_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP13_2 r0 r3 ≤ MP13_2 := CaseSplit.le_mxr2 (aP13_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP13_3 r0 r4 ≤ MP13_3 := CaseSplit.le_mxr2 (aP13_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP13_4 r0 r5 ≤ MP13_4 := CaseSplit.le_mxr2 (aP13_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP13_5 r1 r2 ≤ MP13_5 := CaseSplit.le_mxr2 (aP13_5) 12 16 r1 r2 (by omega) (by omega)
  have bP6 : aP13_6 r1 r3 ≤ MP13_6 := CaseSplit.le_mxr2 (aP13_6) 12 18 r1 r3 (by omega) (by omega)
  have bP7 : aP13_7 r1 r4 ≤ MP13_7 := CaseSplit.le_mxr2 (aP13_7) 12 22 r1 r4 (by omega) (by omega)
  have bP8 : aP13_8 r1 r5 ≤ MP13_8 := CaseSplit.le_mxr2 (aP13_8) 12 28 r1 r5 (by omega) (by omega)
  have bP9 : aP13_9 r2 r3 ≤ MP13_9 := CaseSplit.le_mxr2 (aP13_9) 16 18 r2 r3 (by omega) (by omega)
  have bP10 : aP13_10 r2 r4 ≤ MP13_10 := CaseSplit.le_mxr2 (aP13_10) 16 22 r2 r4 (by omega) (by omega)
  have bP11 : aP13_11 r2 r5 ≤ MP13_11 := CaseSplit.le_mxr2 (aP13_11) 16 28 r2 r5 (by omega) (by omega)
  have bP12 : aP13_12 r3 r4 ≤ MP13_12 := CaseSplit.le_mxr2 (aP13_12) 18 22 r3 r4 (by omega) (by omega)
  have bP13 : aP13_13 r3 r5 ≤ MP13_13 := CaseSplit.le_mxr2 (aP13_13) 18 28 r3 r5 (by omega) (by omega)
  have bP14 : aP13_14 r4 r5 ≤ MP13_14 := CaseSplit.le_mxr2 (aP13_14) 22 28 r4 r5 (by omega) (by omega)
  have hrhs : rhs13 = (∑ t ∈ Finset.range n13, w13 t) + 3 * (n13 : ℤ) := rfl
  have hc := cert13
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, bS0, bS1, bS2, bS3, bS4, bS5, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14]

end IncCert29
