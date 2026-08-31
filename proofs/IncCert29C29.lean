/-
INCREMENT-WIDTH CERTIFICATE, step 23->29, case 29 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_23_29.json, which re-derives every number
from the primes alone).

Machine 29, INCREMENT width 49 = F_2(23) + s_min(29) = 39 + 10,
held gears [5, 7] at phases [4, 1].  Free gears [11, 13, 17, 19, 23, 29].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 1.
-/
import IncCert29B

namespace IncCert29

/-! ### case 29: held gears at phases [4, 1] -/

def p29 : List ℕ := [1, 3, 4, 6, 8, 9, 11, 13, 16, 18, 23, 24, 29, 31, 34, 36, 38, 39, 41, 43, 44, 46, 48]
def q29 (t : ℕ) : ℕ := p29.getD t 0
def n29 : ℕ := 23
def yl29 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def w29 (t : ℕ) : ℤ := yl29.getD t 0
def ul29 : List ℤ := [0, (-1), 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 4, 4, 3, 3, 4, 4, 4, 3, 4, 4, 4, 4, 4, 3, 4, (-5), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-6), (-4), (-4), (-4), 4, 2, 4, 4, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 3, 3, 4, 4, 3, (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), 2, 2, 2, 2, 1, 2, 2, 0, 1, 1, 0, 2, 2, 1, 1, 2, 2, 2, 2, 2, 0, 2, 2, (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 2, 3, 2, 3, 2, 2, 2, 2, 3, 3, 2, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 0, 2, 0, 2, 2, (-1), 2, 2, 2, 2, 2, 2, 0, 2, 2, 0, 2, 0, 1, 2, 1, 2, 0]
def u29 (k : ℕ) : ℤ := ul29.getD k 0

def c29_0 (r t : ℕ) : Bool := gb11 r (q29 t)
def c29_1 (r t : ℕ) : Bool := gb13 r (q29 t)
def c29_2 (r t : ℕ) : Bool := gb17 r (q29 t)
def c29_3 (r t : ℕ) : Bool := gb19 r (q29 t)
def c29_4 (r t : ℕ) : Bool := gb23 r (q29 t)
def c29_5 (r t : ℕ) : Bool := gb29 r (q29 t)

def S29_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (w29 t + 1) * (if c29_0 r t then 1 else 0)
def S29_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (w29 t + 1) * (if c29_1 r t then 1 else 0)
def S29_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (w29 t + 1) * (if c29_2 r t then 1 else 0)
def S29_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (w29 t + 1) * (if c29_3 r t then 1 else 0)
def S29_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (w29 t + 1) * (if c29_4 r t then 1 else 0)
def S29_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (w29 t + 1) * (if c29_5 r t then 1 else 0)

def L29_0 (r : ℕ) : ℤ := u29 (13 + r) + u29 (41 + r) + u29 (71 + r) + u29 (105 + r) + u29 (145 + r)
def L29_1 (r : ℕ) : ℤ := u29 (0 + r) + u29 (173 + r) + u29 (205 + r) + u29 (241 + r) + u29 (283 + r)
def L29_2 (r : ℕ) : ℤ := u29 (24 + r) + u29 (156 + r) + u29 (315 + r) + u29 (355 + r) + u29 (401 + r)
def L29_3 (r : ℕ) : ℤ := u29 (52 + r) + u29 (186 + r) + u29 (296 + r) + u29 (441 + r) + u29 (489 + r)
def L29_4 (r : ℕ) : ℤ := u29 (82 + r) + u29 (218 + r) + u29 (332 + r) + u29 (418 + r) + u29 (537 + r)
def L29_5 (r : ℕ) : ℤ := u29 (116 + r) + u29 (254 + r) + u29 (372 + r) + u29 (460 + r) + u29 (508 + r)

def aS29_0 (r : ℕ) : ℤ := S29_0 r - L29_0 r
def MS29_0 : ℤ := CaseSplit.mxr (aS29_0) 10
def aS29_1 (r : ℕ) : ℤ := S29_1 r - L29_1 r
def MS29_1 : ℤ := CaseSplit.mxr (aS29_1) 12
def aS29_2 (r : ℕ) : ℤ := S29_2 r - L29_2 r
def MS29_2 : ℤ := CaseSplit.mxr (aS29_2) 16
def aS29_3 (r : ℕ) : ℤ := S29_3 r - L29_3 r
def MS29_3 : ℤ := CaseSplit.mxr (aS29_3) 18
def aS29_4 (r : ℕ) : ℤ := S29_4 r - L29_4 r
def MS29_4 : ℤ := CaseSplit.mxr (aS29_4) 22
def aS29_5 (r : ℕ) : ℤ := S29_5 r - L29_5 r
def MS29_5 : ℤ := CaseSplit.mxr (aS29_5) 28

def N29_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_0 ra t && c29_1 rb t then 1 else 0)
def aP29_0 (ra rb : ℕ) : ℤ := -(1) * N29_0 ra rb + u29 (0 + rb) + u29 (13 + ra)
def MP29_0 : ℤ := CaseSplit.mxr2 (aP29_0) 10 12
def N29_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_0 ra t && c29_2 rb t then 1 else 0)
def aP29_1 (ra rb : ℕ) : ℤ := -(1) * N29_1 ra rb + u29 (24 + rb) + u29 (41 + ra)
def MP29_1 : ℤ := CaseSplit.mxr2 (aP29_1) 10 16
def N29_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_0 ra t && c29_3 rb t then 1 else 0)
def aP29_2 (ra rb : ℕ) : ℤ := -(1) * N29_2 ra rb + u29 (52 + rb) + u29 (71 + ra)
def MP29_2 : ℤ := CaseSplit.mxr2 (aP29_2) 10 18
def N29_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_0 ra t && c29_4 rb t then 1 else 0)
def aP29_3 (ra rb : ℕ) : ℤ := -(1) * N29_3 ra rb + u29 (82 + rb) + u29 (105 + ra)
def MP29_3 : ℤ := CaseSplit.mxr2 (aP29_3) 10 22
def N29_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_0 ra t && c29_5 rb t then 1 else 0)
def aP29_4 (ra rb : ℕ) : ℤ := -(1) * N29_4 ra rb + u29 (116 + rb) + u29 (145 + ra)
def MP29_4 : ℤ := CaseSplit.mxr2 (aP29_4) 10 28
def P29_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_1 ra t && c29_2 rb t then 1 else 0)
def C29_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_1 ra t && c29_2 rb t && c29_0 s t then 1 else 0)
def M29_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C29_5 ra rb) 10
def E29_5 : List ℕ := [7, 13, 93, 99, 102, 108, 138, 144, 172, 183]
def N29_5 (ra rb : ℕ) : ℤ := if E29_5.contains (ra * 17 + rb) = true then P29_5 ra rb - M29_5 ra rb else 0
def aP29_5 (ra rb : ℕ) : ℤ := -(1) * N29_5 ra rb + u29 (156 + rb) + u29 (173 + ra)
def MP29_5 : ℤ := CaseSplit.mxr2 (aP29_5) 12 16
def P29_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_1 ra t && c29_3 rb t then 1 else 0)
def C29_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_1 ra t && c29_3 rb t && c29_0 s t then 1 else 0)
def M29_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C29_6 ra rb) 10
def E29_6 : List ℕ := [11, 37, 87, 113, 118, 124, 152, 158, 194, 200, 228, 234]
def N29_6 (ra rb : ℕ) : ℤ := if E29_6.contains (ra * 19 + rb) = true then P29_6 ra rb - M29_6 ra rb else 0
def aP29_6 (ra rb : ℕ) : ℤ := -(1) * N29_6 ra rb + u29 (186 + rb) + u29 (205 + ra)
def MP29_6 : ℤ := CaseSplit.mxr2 (aP29_6) 12 18
def P29_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_1 ra t && c29_4 rb t then 1 else 0)
def C29_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_1 ra t && c29_4 rb t && c29_0 s t then 1 else 0)
def M29_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C29_7 ra rb) 10
def E29_7 : List ℕ := []
def N29_7 (ra rb : ℕ) : ℤ := if E29_7.contains (ra * 23 + rb) = true then P29_7 ra rb - M29_7 ra rb else 0
def aP29_7 (ra rb : ℕ) : ℤ := -(1) * N29_7 ra rb + u29 (218 + rb) + u29 (241 + ra)
def MP29_7 : ℤ := CaseSplit.mxr2 (aP29_7) 12 22
def P29_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_1 ra t && c29_5 rb t then 1 else 0)
def C29_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_1 ra t && c29_5 rb t && c29_0 s t then 1 else 0)
def M29_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C29_8 ra rb) 10
def E29_8 : List ℕ := [73, 189, 223, 339]
def N29_8 (ra rb : ℕ) : ℤ := if E29_8.contains (ra * 29 + rb) = true then P29_8 ra rb - M29_8 ra rb else 0
def aP29_8 (ra rb : ℕ) : ℤ := -(1) * N29_8 ra rb + u29 (254 + rb) + u29 (283 + ra)
def MP29_8 : ℤ := CaseSplit.mxr2 (aP29_8) 12 28
def N29_9 (_ra _rb : ℕ) : ℤ := 0
def aP29_9 (ra rb : ℕ) : ℤ := -(1) * N29_9 ra rb + u29 (296 + rb) + u29 (315 + ra)
def MP29_9 : ℤ := CaseSplit.mxr2 (aP29_9) 16 18
def N29_10 (_ra _rb : ℕ) : ℤ := 0
def aP29_10 (ra rb : ℕ) : ℤ := -(1) * N29_10 ra rb + u29 (332 + rb) + u29 (355 + ra)
def MP29_10 : ℤ := CaseSplit.mxr2 (aP29_10) 16 22
def N29_11 (_ra _rb : ℕ) : ℤ := 0
def aP29_11 (ra rb : ℕ) : ℤ := -(1) * N29_11 ra rb + u29 (372 + rb) + u29 (401 + ra)
def MP29_11 : ℤ := CaseSplit.mxr2 (aP29_11) 16 28
def N29_12 (_ra _rb : ℕ) : ℤ := 0
def aP29_12 (ra rb : ℕ) : ℤ := -(1) * N29_12 ra rb + u29 (418 + rb) + u29 (441 + ra)
def MP29_12 : ℤ := CaseSplit.mxr2 (aP29_12) 18 22
def N29_13 (_ra _rb : ℕ) : ℤ := 0
def aP29_13 (ra rb : ℕ) : ℤ := -(1) * N29_13 ra rb + u29 (460 + rb) + u29 (489 + ra)
def MP29_13 : ℤ := CaseSplit.mxr2 (aP29_13) 18 28
def N29_14 (_ra _rb : ℕ) : ℤ := 0
def aP29_14 (ra rb : ℕ) : ℤ := -(1) * N29_14 ra rb + u29 (508 + rb) + u29 (537 + ra)
def MP29_14 : ℤ := CaseSplit.mxr2 (aP29_14) 22 28

def rhs29 : ℤ := (∑ t ∈ Finset.range n29, w29 t) + 1 * (n29 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn29 : ∀ t, t < n29 → (0 : ℤ) ≤ w29 t := by decide
theorem plt29 : ∀ t, t < n29 → q29 t < 49 := by decide
theorem pfree29_5 : ∀ t, t < n29 → gb5 4 (q29 t) = false := by decide
theorem pfree29_7 : ∀ t, t < n29 → gb7 1 (q29 t) = false := by decide
theorem MSv29_0 : MS29_0 = 3 := by decide +kernel
theorem MSv29_1 : MS29_1 = 14 := by decide +kernel
theorem MSv29_2 : MS29_2 = 0 := by decide +kernel
theorem MSv29_3 : MS29_3 = 0 := by decide +kernel
theorem MSv29_4 : MS29_4 = 0 := by decide +kernel
theorem MSv29_5 : MS29_5 = 0 := by decide +kernel
theorem MPv29_0 : MP29_0 = 0 := by decide +kernel
theorem MPv29_1 : MP29_1 = 0 := by decide +kernel
theorem MPv29_2 : MP29_2 = 0 := by decide +kernel
theorem MPv29_3 : MP29_3 = 0 := by decide +kernel
theorem MPv29_4 : MP29_4 = 0 := by decide +kernel
theorem MPv29_5 : MP29_5 = 0 := by decide +kernel
theorem MPv29_6 : MP29_6 = 0 := by decide +kernel
theorem MPv29_7 : MP29_7 = 0 := by decide +kernel
theorem MPv29_8 : MP29_8 = 0 := by decide +kernel
theorem MPv29_9 : MP29_9 = 0 := by decide +kernel
theorem MPv29_10 : MP29_10 = 0 := by decide +kernel
theorem MPv29_11 : MP29_11 = 0 := by decide +kernel
theorem MPv29_12 : MP29_12 = 0 := by decide +kernel
theorem MPv29_13 : MP29_13 = 0 := by decide +kernel
theorem MPv29_14 : MP29_14 = 5 := by decide +kernel
theorem rhsv29 : rhs29 = 23 := by decide +kernel

/-- **The case-29 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/1.
    (Scaled by the common denominator 1: 22 < 23.) -/
theorem cert29 : MS29_0 + MS29_1 + MS29_2 + MS29_3 + MS29_4 + MS29_5 + MP29_0 + MP29_1 + MP29_2 + MP29_3 + MP29_4 + MP29_5 + MP29_6 + MP29_7 + MP29_8 + MP29_9 + MP29_10 + MP29_11 + MP29_12 + MP29_13 + MP29_14 < rhs29 := by
  rw [MSv29_0, MSv29_1, MSv29_2, MSv29_3, MSv29_4, MSv29_5, MPv29_0, MPv29_1, MPv29_2, MPv29_3, MPv29_4, MPv29_5, MPv29_6, MPv29_7, MPv29_8, MPv29_9, MPv29_10, MPv29_11, MPv29_12, MPv29_13, MPv29_14, rhsv29]
  decide

def Dg29 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := (if c29_0 r0 t then 1 else 0) + (if c29_1 r1 t then 1 else 0) + (if c29_2 r2 t then 1 else 0) + (if c29_3 r3 t then 1 else 0) + (if c29_4 r4 t then 1 else 0) + (if c29_5 r5 t then 1 else 0)
def Wl29_0 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c29_0 r0 t && c29_1 r1 t then 1 else 0
def Wl29_1 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c29_0 r0 t && c29_2 r2 t then 1 else 0
def Wl29_2 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c29_0 r0 t && c29_3 r3 t then 1 else 0
def Wl29_3 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c29_0 r0 t && c29_4 r4 t then 1 else 0
def Wl29_4 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c29_0 r0 t && c29_5 r5 t then 1 else 0
def Wl29_5 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c29_0 r0 t && c29_1 r1 t && c29_2 r2 t then 1 else 0
def Wl29_6 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c29_0 r0 t && c29_1 r1 t && c29_3 r3 t then 1 else 0
def Wl29_7 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c29_0 r0 t && c29_1 r1 t && c29_4 r4 t then 1 else 0
def Wl29_8 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c29_0 r0 t && c29_1 r1 t && c29_5 r5 t then 1 else 0
def Wl29_9 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c29_0 r0 t && !c29_1 r1 t && c29_2 r2 t && c29_3 r3 t then 1 else 0
def Wl29_10 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c29_0 r0 t && !c29_1 r1 t && c29_2 r2 t && c29_4 r4 t then 1 else 0
def Wl29_11 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c29_0 r0 t && !c29_1 r1 t && c29_2 r2 t && c29_5 r5 t then 1 else 0
def Wl29_12 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c29_0 r0 t && !c29_1 r1 t && !c29_2 r2 t && c29_3 r3 t && c29_4 r4 t then 1 else 0
def Wl29_13 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c29_0 r0 t && !c29_1 r1 t && !c29_2 r2 t && c29_3 r3 t && c29_5 r5 t then 1 else 0
def Wl29_14 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c29_0 r0 t && !c29_1 r1 t && !c29_2 r2 t && !c29_3 r3 t && c29_4 r4 t && c29_5 r5 t then 1 else 0

/-- **No configuration blocks the whole window in case 29.** -/
theorem nocov29 {r0 r1 r2 r3 r4 r5 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29)
    (hcov : ∀ t, t < n29 → (c29_0 r0 t || c29_1 r1 t || c29_2 r2 t || c29_3 r3 t || c29_4 r4 t || c29_5 r5 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n29, (1 : ℤ) + (Wl29_0 r0 r1 r2 r3 r4 r5 t + Wl29_1 r0 r1 r2 r3 r4 r5 t + Wl29_2 r0 r1 r2 r3 r4 r5 t + Wl29_3 r0 r1 r2 r3 r4 r5 t + Wl29_4 r0 r1 r2 r3 r4 r5 t + Wl29_5 r0 r1 r2 r3 r4 r5 t + Wl29_6 r0 r1 r2 r3 r4 r5 t + Wl29_7 r0 r1 r2 r3 r4 r5 t + Wl29_8 r0 r1 r2 r3 r4 r5 t + Wl29_9 r0 r1 r2 r3 r4 r5 t + Wl29_10 r0 r1 r2 r3 r4 r5 t + Wl29_11 r0 r1 r2 r3 r4 r5 t + Wl29_12 r0 r1 r2 r3 r4 r5 t + Wl29_13 r0 r1 r2 r3 r4 r5 t + Wl29_14 r0 r1 r2 r3 r4 r5 t) ≤ Dg29 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Wl29_0, Wl29_1, Wl29_2, Wl29_3, Wl29_4, Wl29_5, Wl29_6, Wl29_7, Wl29_8, Wl29_9, Wl29_10, Wl29_11, Wl29_12, Wl29_13, Wl29_14, Dg29]
    exact CaseSplit.lowest6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n29, (1 : ℤ) ≤ Dg29 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Dg29]
    exact CaseSplit.degpos6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n29 : ℤ) + ((∑ t ∈ Finset.range n29, Wl29_0 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n29, Wl29_1 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n29, Wl29_2 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n29, Wl29_3 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n29, Wl29_4 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n29, Wl29_5 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n29, Wl29_6 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n29, Wl29_7 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n29, Wl29_8 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n29, Wl29_9 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n29, Wl29_10 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n29, Wl29_11 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n29, Wl29_12 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n29, Wl29_13 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n29, Wl29_14 r0 r1 r2 r3 r4 r5 t)) ≤ ∑ t ∈ Finset.range n29, Dg29 r0 r1 r2 r3 r4 r5 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N29_0 r0 r1 ≤ ∑ t ∈ Finset.range n29, Wl29_0 r0 r1 r2 r3 r4 r5 t := by
    simp only [N29_0, Wl29_0, le_refl]
  have hn1 : N29_1 r0 r2 ≤ ∑ t ∈ Finset.range n29, Wl29_1 r0 r1 r2 r3 r4 r5 t := by
    simp only [N29_1, Wl29_1, le_refl]
  have hn2 : N29_2 r0 r3 ≤ ∑ t ∈ Finset.range n29, Wl29_2 r0 r1 r2 r3 r4 r5 t := by
    simp only [N29_2, Wl29_2, le_refl]
  have hn3 : N29_3 r0 r4 ≤ ∑ t ∈ Finset.range n29, Wl29_3 r0 r1 r2 r3 r4 r5 t := by
    simp only [N29_3, Wl29_3, le_refl]
  have hn4 : N29_4 r0 r5 ≤ ∑ t ∈ Finset.range n29, Wl29_4 r0 r1 r2 r3 r4 r5 t := by
    simp only [N29_4, Wl29_4, le_refl]
  have hn5 : N29_5 r1 r2 ≤ ∑ t ∈ Finset.range n29, Wl29_5 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n29, Wl29_5 r0 r1 r2 r3 r4 r5 t
        = (if c29_1 r1 t && c29_2 r2 t then (1:ℤ) else 0)
          - (if c29_1 r1 t && c29_2 r2 t && c29_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl29_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n29, Wl29_5 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl29_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n29, Wl29_5 r0 r1 r2 r3 r4 r5 t
        = P29_5 r1 r2 - C29_5 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P29_5, C29_5]
    have hm : C29_5 r1 r2 r0 ≤ M29_5 r1 r2 :=
      CaseSplit.le_mxr (C29_5 r1 r2) 10 r0 (by omega)
    simp only [N29_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N29_6 r1 r3 ≤ ∑ t ∈ Finset.range n29, Wl29_6 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n29, Wl29_6 r0 r1 r2 r3 r4 r5 t
        = (if c29_1 r1 t && c29_3 r3 t then (1:ℤ) else 0)
          - (if c29_1 r1 t && c29_3 r3 t && c29_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl29_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n29, Wl29_6 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl29_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n29, Wl29_6 r0 r1 r2 r3 r4 r5 t
        = P29_6 r1 r3 - C29_6 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P29_6, C29_6]
    have hm : C29_6 r1 r3 r0 ≤ M29_6 r1 r3 :=
      CaseSplit.le_mxr (C29_6 r1 r3) 10 r0 (by omega)
    simp only [N29_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N29_7 r1 r4 ≤ ∑ t ∈ Finset.range n29, Wl29_7 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n29, Wl29_7 r0 r1 r2 r3 r4 r5 t
        = (if c29_1 r1 t && c29_4 r4 t then (1:ℤ) else 0)
          - (if c29_1 r1 t && c29_4 r4 t && c29_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl29_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n29, Wl29_7 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl29_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n29, Wl29_7 r0 r1 r2 r3 r4 r5 t
        = P29_7 r1 r4 - C29_7 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P29_7, C29_7]
    have hm : C29_7 r1 r4 r0 ≤ M29_7 r1 r4 :=
      CaseSplit.le_mxr (C29_7 r1 r4) 10 r0 (by omega)
    simp only [N29_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N29_8 r1 r5 ≤ ∑ t ∈ Finset.range n29, Wl29_8 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n29, Wl29_8 r0 r1 r2 r3 r4 r5 t
        = (if c29_1 r1 t && c29_5 r5 t then (1:ℤ) else 0)
          - (if c29_1 r1 t && c29_5 r5 t && c29_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl29_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n29, Wl29_8 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl29_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n29, Wl29_8 r0 r1 r2 r3 r4 r5 t
        = P29_8 r1 r5 - C29_8 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P29_8, C29_8]
    have hm : C29_8 r1 r5 r0 ≤ M29_8 r1 r5 :=
      CaseSplit.le_mxr (C29_8 r1 r5) 10 r0 (by omega)
    simp only [N29_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N29_9 r2 r3 ≤ ∑ t ∈ Finset.range n29, Wl29_9 r0 r1 r2 r3 r4 r5 t := by
    simp only [N29_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl29_9]
    exact CaseSplit.ind_nonneg _
  have hn10 : N29_10 r2 r4 ≤ ∑ t ∈ Finset.range n29, Wl29_10 r0 r1 r2 r3 r4 r5 t := by
    simp only [N29_10]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl29_10]
    exact CaseSplit.ind_nonneg _
  have hn11 : N29_11 r2 r5 ≤ ∑ t ∈ Finset.range n29, Wl29_11 r0 r1 r2 r3 r4 r5 t := by
    simp only [N29_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl29_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N29_12 r3 r4 ≤ ∑ t ∈ Finset.range n29, Wl29_12 r0 r1 r2 r3 r4 r5 t := by
    simp only [N29_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl29_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N29_13 r3 r5 ≤ ∑ t ∈ Finset.range n29, Wl29_13 r0 r1 r2 r3 r4 r5 t := by
    simp only [N29_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl29_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N29_14 r4 r5 ≤ ∑ t ∈ Finset.range n29, Wl29_14 r0 r1 r2 r3 r4 r5 t := by
    simp only [N29_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl29_14]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n29, (w29 t + 1) * Dg29 r0 r1 r2 r3 r4 r5 t = S29_0 r0 + S29_1 r1 + S29_2 r2 + S29_3 r3 + S29_4 r4 + S29_5 r5 := by
    simp only [S29_0, S29_1, S29_2, S29_3, S29_4, S29_5, Dg29, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n29, (w29 t + 1) * Dg29 r0 r1 r2 r3 r4 r5 t
      = (∑ t ∈ Finset.range n29, w29 t * Dg29 r0 r1 r2 r3 r4 r5 t)
        + 1 * (∑ t ∈ Finset.range n29, Dg29 r0 r1 r2 r3 r4 r5 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n29, w29 t)
      ≤ ∑ t ∈ Finset.range n29, w29 t * Dg29 r0 r1 r2 r3 r4 r5 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg29 r0 r1 r2 r3 r4 r5 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w29 t := wnn29 t (Finset.mem_range.mp ht)
    calc w29 t = w29 t * 1 := (mul_one _).symm
      _ ≤ w29 t * Dg29 r0 r1 r2 r3 r4 r5 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS29_0 r0 + aS29_1 r1 + aS29_2 r2 + aS29_3 r3 + aS29_4 r4 + aS29_5 r5) + (aP29_0 r0 r1 + aP29_1 r0 r2 + aP29_2 r0 r3 + aP29_3 r0 r4 + aP29_4 r0 r5 + aP29_5 r1 r2 + aP29_6 r1 r3 + aP29_7 r1 r4 + aP29_8 r1 r5 + aP29_9 r2 r3 + aP29_10 r2 r4 + aP29_11 r2 r5 + aP29_12 r3 r4 + aP29_13 r3 r5 + aP29_14 r4 r5) = (S29_0 r0 + S29_1 r1 + S29_2 r2 + S29_3 r3 + S29_4 r4 + S29_5 r5) - 1 * (N29_0 r0 r1 + N29_1 r0 r2 + N29_2 r0 r3 + N29_3 r0 r4 + N29_4 r0 r5 + N29_5 r1 r2 + N29_6 r1 r3 + N29_7 r1 r4 + N29_8 r1 r5 + N29_9 r2 r3 + N29_10 r2 r4 + N29_11 r2 r5 + N29_12 r3 r4 + N29_13 r3 r5 + N29_14 r4 r5) := by
    simp only [aS29_0, aS29_1, aS29_2, aS29_3, aS29_4, aS29_5, aP29_0, aP29_1, aP29_2, aP29_3, aP29_4, aP29_5, aP29_6, aP29_7, aP29_8, aP29_9, aP29_10, aP29_11, aP29_12, aP29_13, aP29_14, L29_0, L29_1, L29_2, L29_3, L29_4, L29_5]
    ring
  have bS0 : aS29_0 r0 ≤ MS29_0 := CaseSplit.le_mxr (aS29_0) 10 r0 (by omega)
  have bS1 : aS29_1 r1 ≤ MS29_1 := CaseSplit.le_mxr (aS29_1) 12 r1 (by omega)
  have bS2 : aS29_2 r2 ≤ MS29_2 := CaseSplit.le_mxr (aS29_2) 16 r2 (by omega)
  have bS3 : aS29_3 r3 ≤ MS29_3 := CaseSplit.le_mxr (aS29_3) 18 r3 (by omega)
  have bS4 : aS29_4 r4 ≤ MS29_4 := CaseSplit.le_mxr (aS29_4) 22 r4 (by omega)
  have bS5 : aS29_5 r5 ≤ MS29_5 := CaseSplit.le_mxr (aS29_5) 28 r5 (by omega)
  have bP0 : aP29_0 r0 r1 ≤ MP29_0 := CaseSplit.le_mxr2 (aP29_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP29_1 r0 r2 ≤ MP29_1 := CaseSplit.le_mxr2 (aP29_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP29_2 r0 r3 ≤ MP29_2 := CaseSplit.le_mxr2 (aP29_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP29_3 r0 r4 ≤ MP29_3 := CaseSplit.le_mxr2 (aP29_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP29_4 r0 r5 ≤ MP29_4 := CaseSplit.le_mxr2 (aP29_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP29_5 r1 r2 ≤ MP29_5 := CaseSplit.le_mxr2 (aP29_5) 12 16 r1 r2 (by omega) (by omega)
  have bP6 : aP29_6 r1 r3 ≤ MP29_6 := CaseSplit.le_mxr2 (aP29_6) 12 18 r1 r3 (by omega) (by omega)
  have bP7 : aP29_7 r1 r4 ≤ MP29_7 := CaseSplit.le_mxr2 (aP29_7) 12 22 r1 r4 (by omega) (by omega)
  have bP8 : aP29_8 r1 r5 ≤ MP29_8 := CaseSplit.le_mxr2 (aP29_8) 12 28 r1 r5 (by omega) (by omega)
  have bP9 : aP29_9 r2 r3 ≤ MP29_9 := CaseSplit.le_mxr2 (aP29_9) 16 18 r2 r3 (by omega) (by omega)
  have bP10 : aP29_10 r2 r4 ≤ MP29_10 := CaseSplit.le_mxr2 (aP29_10) 16 22 r2 r4 (by omega) (by omega)
  have bP11 : aP29_11 r2 r5 ≤ MP29_11 := CaseSplit.le_mxr2 (aP29_11) 16 28 r2 r5 (by omega) (by omega)
  have bP12 : aP29_12 r3 r4 ≤ MP29_12 := CaseSplit.le_mxr2 (aP29_12) 18 22 r3 r4 (by omega) (by omega)
  have bP13 : aP29_13 r3 r5 ≤ MP29_13 := CaseSplit.le_mxr2 (aP29_13) 18 28 r3 r5 (by omega) (by omega)
  have bP14 : aP29_14 r4 r5 ≤ MP29_14 := CaseSplit.le_mxr2 (aP29_14) 22 28 r4 r5 (by omega) (by omega)
  have hrhs : rhs29 = (∑ t ∈ Finset.range n29, w29 t) + 1 * (n29 : ℤ) := rfl
  have hc := cert29
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, bS0, bS1, bS2, bS3, bS4, bS5, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14]

end IncCert29
