/-
INCREMENT-WIDTH CERTIFICATE, step 23->29, case 20 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_23_29.json, which re-derives every number
from the primes alone).

Machine 29, INCREMENT width 49 = F_2(23) + s_min(29) = 39 + 10,
held gears [5, 7] at phases [2, 6].  Free gears [11, 13, 17, 19, 23, 29].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 1.
-/
import IncCert29B

namespace IncCert29

/-! ### case 20: held gears at phases [2, 6] -/

def p20 : List ℕ := [1, 3, 5, 6, 8, 10, 11, 13, 15, 18, 20, 25, 26, 31, 33, 36, 38, 40, 41, 43, 45, 46, 48]
def q20 (t : ℕ) : ℕ := p20.getD t 0
def n20 : ℕ := 23
def yl20 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def w20 (t : ℕ) : ℤ := yl20.getD t 0
def ul20 : List ℤ := [(-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, (-1), 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, (-1), (-1), (-1), 0, 0, 0, (-1), (-1), 0, 0, (-1), (-1), (-1), (-1), (-1), (-1), (-2), (-1), (-1), 0, (-1), (-1), (-1), (-2), (-1), 0, (-1), (-1), (-3), 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, (-1), (-1), 0, (-1), (-1), (-1), 0, (-1), (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 3, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-4), 3, 3, 4, 3, 3, 3, 4, 2, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), 0, 3, 3, 3, 3, 3, 2, 3, 1, 3, 1, 2, 2, 1, 3, 0, 2, 3, 3, 3, 1, 3, 3, (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 3, 3, 0, 0, 0, (-2), 0, (-2), 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-2), 0, 0, (-2), 0]
def u20 (k : ℕ) : ℤ := ul20.getD k 0

def c20_0 (r t : ℕ) : Bool := gb11 r (q20 t)
def c20_1 (r t : ℕ) : Bool := gb13 r (q20 t)
def c20_2 (r t : ℕ) : Bool := gb17 r (q20 t)
def c20_3 (r t : ℕ) : Bool := gb19 r (q20 t)
def c20_4 (r t : ℕ) : Bool := gb23 r (q20 t)
def c20_5 (r t : ℕ) : Bool := gb29 r (q20 t)

def S20_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n20, (w20 t + 1) * (if c20_0 r t then 1 else 0)
def S20_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n20, (w20 t + 1) * (if c20_1 r t then 1 else 0)
def S20_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n20, (w20 t + 1) * (if c20_2 r t then 1 else 0)
def S20_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n20, (w20 t + 1) * (if c20_3 r t then 1 else 0)
def S20_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n20, (w20 t + 1) * (if c20_4 r t then 1 else 0)
def S20_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n20, (w20 t + 1) * (if c20_5 r t then 1 else 0)

def L20_0 (r : ℕ) : ℤ := u20 (13 + r) + u20 (41 + r) + u20 (71 + r) + u20 (105 + r) + u20 (145 + r)
def L20_1 (r : ℕ) : ℤ := u20 (0 + r) + u20 (173 + r) + u20 (205 + r) + u20 (241 + r) + u20 (283 + r)
def L20_2 (r : ℕ) : ℤ := u20 (24 + r) + u20 (156 + r) + u20 (315 + r) + u20 (355 + r) + u20 (401 + r)
def L20_3 (r : ℕ) : ℤ := u20 (52 + r) + u20 (186 + r) + u20 (296 + r) + u20 (441 + r) + u20 (489 + r)
def L20_4 (r : ℕ) : ℤ := u20 (82 + r) + u20 (218 + r) + u20 (332 + r) + u20 (418 + r) + u20 (537 + r)
def L20_5 (r : ℕ) : ℤ := u20 (116 + r) + u20 (254 + r) + u20 (372 + r) + u20 (460 + r) + u20 (508 + r)

def aS20_0 (r : ℕ) : ℤ := S20_0 r - L20_0 r
def MS20_0 : ℤ := CaseSplit.mxr (aS20_0) 10
def aS20_1 (r : ℕ) : ℤ := S20_1 r - L20_1 r
def MS20_1 : ℤ := CaseSplit.mxr (aS20_1) 12
def aS20_2 (r : ℕ) : ℤ := S20_2 r - L20_2 r
def MS20_2 : ℤ := CaseSplit.mxr (aS20_2) 16
def aS20_3 (r : ℕ) : ℤ := S20_3 r - L20_3 r
def MS20_3 : ℤ := CaseSplit.mxr (aS20_3) 18
def aS20_4 (r : ℕ) : ℤ := S20_4 r - L20_4 r
def MS20_4 : ℤ := CaseSplit.mxr (aS20_4) 22
def aS20_5 (r : ℕ) : ℤ := S20_5 r - L20_5 r
def MS20_5 : ℤ := CaseSplit.mxr (aS20_5) 28

def N20_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n20, (if c20_0 ra t && c20_1 rb t then 1 else 0)
def aP20_0 (ra rb : ℕ) : ℤ := -(1) * N20_0 ra rb + u20 (0 + rb) + u20 (13 + ra)
def MP20_0 : ℤ := CaseSplit.mxr2 (aP20_0) 10 12
def N20_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n20, (if c20_0 ra t && c20_2 rb t then 1 else 0)
def aP20_1 (ra rb : ℕ) : ℤ := -(1) * N20_1 ra rb + u20 (24 + rb) + u20 (41 + ra)
def MP20_1 : ℤ := CaseSplit.mxr2 (aP20_1) 10 16
def N20_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n20, (if c20_0 ra t && c20_3 rb t then 1 else 0)
def aP20_2 (ra rb : ℕ) : ℤ := -(1) * N20_2 ra rb + u20 (52 + rb) + u20 (71 + ra)
def MP20_2 : ℤ := CaseSplit.mxr2 (aP20_2) 10 18
def N20_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n20, (if c20_0 ra t && c20_4 rb t then 1 else 0)
def aP20_3 (ra rb : ℕ) : ℤ := -(1) * N20_3 ra rb + u20 (82 + rb) + u20 (105 + ra)
def MP20_3 : ℤ := CaseSplit.mxr2 (aP20_3) 10 22
def N20_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n20, (if c20_0 ra t && c20_5 rb t then 1 else 0)
def aP20_4 (ra rb : ℕ) : ℤ := -(1) * N20_4 ra rb + u20 (116 + rb) + u20 (145 + ra)
def MP20_4 : ℤ := CaseSplit.mxr2 (aP20_4) 10 28
def P20_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n20, (if c20_1 ra t && c20_2 rb t then 1 else 0)
def C20_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n20, (if c20_1 ra t && c20_2 rb t && c20_0 s t then 1 else 0)
def M20_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C20_5 ra rb) 10
def E20_5 : List ℕ := [57, 63, 102, 108, 136, 147, 172, 183, 192, 198]
def N20_5 (ra rb : ℕ) : ℤ := if E20_5.contains (ra * 17 + rb) = true then P20_5 ra rb - M20_5 ra rb else 0
def aP20_5 (ra rb : ℕ) : ℤ := -(1) * N20_5 ra rb + u20 (156 + rb) + u20 (173 + ra)
def MP20_5 : ℤ := CaseSplit.mxr2 (aP20_5) 12 16
def P20_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n20, (if c20_1 ra t && c20_3 rb t then 1 else 0)
def C20_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n20, (if c20_1 ra t && c20_3 rb t && c20_0 s t then 1 else 0)
def M20_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C20_6 ra rb) 10
def E20_6 : List ℕ := [47, 73, 78, 84, 118, 131, 154, 160, 194, 207, 218, 244]
def N20_6 (ra rb : ℕ) : ℤ := if E20_6.contains (ra * 19 + rb) = true then P20_6 ra rb - M20_6 ra rb else 0
def aP20_6 (ra rb : ℕ) : ℤ := -(1) * N20_6 ra rb + u20 (186 + rb) + u20 (205 + ra)
def MP20_6 : ℤ := CaseSplit.mxr2 (aP20_6) 12 18
def P20_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n20, (if c20_1 ra t && c20_4 rb t then 1 else 0)
def C20_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n20, (if c20_1 ra t && c20_4 rb t && c20_0 s t then 1 else 0)
def M20_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C20_7 ra rb) 10
def E20_7 : List ℕ := []
def N20_7 (ra rb : ℕ) : ℤ := if E20_7.contains (ra * 23 + rb) = true then P20_7 ra rb - M20_7 ra rb else 0
def aP20_7 (ra rb : ℕ) : ℤ := -(1) * N20_7 ra rb + u20 (218 + rb) + u20 (241 + ra)
def MP20_7 : ℤ := CaseSplit.mxr2 (aP20_7) 12 22
def P20_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n20, (if c20_1 ra t && c20_5 rb t then 1 else 0)
def C20_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n20, (if c20_1 ra t && c20_5 rb t && c20_0 s t then 1 else 0)
def M20_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C20_8 ra rb) 10
def E20_8 : List ℕ := [52, 163, 279, 313]
def N20_8 (ra rb : ℕ) : ℤ := if E20_8.contains (ra * 29 + rb) = true then P20_8 ra rb - M20_8 ra rb else 0
def aP20_8 (ra rb : ℕ) : ℤ := -(1) * N20_8 ra rb + u20 (254 + rb) + u20 (283 + ra)
def MP20_8 : ℤ := CaseSplit.mxr2 (aP20_8) 12 28
def N20_9 (_ra _rb : ℕ) : ℤ := 0
def aP20_9 (ra rb : ℕ) : ℤ := -(1) * N20_9 ra rb + u20 (296 + rb) + u20 (315 + ra)
def MP20_9 : ℤ := CaseSplit.mxr2 (aP20_9) 16 18
def N20_10 (_ra _rb : ℕ) : ℤ := 0
def aP20_10 (ra rb : ℕ) : ℤ := -(1) * N20_10 ra rb + u20 (332 + rb) + u20 (355 + ra)
def MP20_10 : ℤ := CaseSplit.mxr2 (aP20_10) 16 22
def N20_11 (_ra _rb : ℕ) : ℤ := 0
def aP20_11 (ra rb : ℕ) : ℤ := -(1) * N20_11 ra rb + u20 (372 + rb) + u20 (401 + ra)
def MP20_11 : ℤ := CaseSplit.mxr2 (aP20_11) 16 28
def N20_12 (_ra _rb : ℕ) : ℤ := 0
def aP20_12 (ra rb : ℕ) : ℤ := -(1) * N20_12 ra rb + u20 (418 + rb) + u20 (441 + ra)
def MP20_12 : ℤ := CaseSplit.mxr2 (aP20_12) 18 22
def N20_13 (_ra _rb : ℕ) : ℤ := 0
def aP20_13 (ra rb : ℕ) : ℤ := -(1) * N20_13 ra rb + u20 (460 + rb) + u20 (489 + ra)
def MP20_13 : ℤ := CaseSplit.mxr2 (aP20_13) 18 28
def N20_14 (_ra _rb : ℕ) : ℤ := 0
def aP20_14 (ra rb : ℕ) : ℤ := -(1) * N20_14 ra rb + u20 (508 + rb) + u20 (537 + ra)
def MP20_14 : ℤ := CaseSplit.mxr2 (aP20_14) 22 28

def rhs20 : ℤ := (∑ t ∈ Finset.range n20, w20 t) + 1 * (n20 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn20 : ∀ t, t < n20 → (0 : ℤ) ≤ w20 t := by decide
theorem plt20 : ∀ t, t < n20 → q20 t < 49 := by decide
theorem pfree20_5 : ∀ t, t < n20 → gb5 2 (q20 t) = false := by decide
theorem pfree20_7 : ∀ t, t < n20 → gb7 6 (q20 t) = false := by decide
theorem MSv20_0 : MS20_0 = 5 := by decide +kernel
theorem MSv20_1 : MS20_1 = 14 := by decide +kernel
theorem MSv20_2 : MS20_2 = 0 := by decide +kernel
theorem MSv20_3 : MS20_3 = 0 := by decide +kernel
theorem MSv20_4 : MS20_4 = 0 := by decide +kernel
theorem MSv20_5 : MS20_5 = 0 := by decide +kernel
theorem MPv20_0 : MP20_0 = 0 := by decide +kernel
theorem MPv20_1 : MP20_1 = 0 := by decide +kernel
theorem MPv20_2 : MP20_2 = 0 := by decide +kernel
theorem MPv20_3 : MP20_3 = 0 := by decide +kernel
theorem MPv20_4 : MP20_4 = 0 := by decide +kernel
theorem MPv20_5 : MP20_5 = 0 := by decide +kernel
theorem MPv20_6 : MP20_6 = 0 := by decide +kernel
theorem MPv20_7 : MP20_7 = 0 := by decide +kernel
theorem MPv20_8 : MP20_8 = 0 := by decide +kernel
theorem MPv20_9 : MP20_9 = 0 := by decide +kernel
theorem MPv20_10 : MP20_10 = 0 := by decide +kernel
theorem MPv20_11 : MP20_11 = 0 := by decide +kernel
theorem MPv20_12 : MP20_12 = 0 := by decide +kernel
theorem MPv20_13 : MP20_13 = 0 := by decide +kernel
theorem MPv20_14 : MP20_14 = 3 := by decide +kernel
theorem rhsv20 : rhs20 = 23 := by decide +kernel

/-- **The case-20 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/1.
    (Scaled by the common denominator 1: 22 < 23.) -/
theorem cert20 : MS20_0 + MS20_1 + MS20_2 + MS20_3 + MS20_4 + MS20_5 + MP20_0 + MP20_1 + MP20_2 + MP20_3 + MP20_4 + MP20_5 + MP20_6 + MP20_7 + MP20_8 + MP20_9 + MP20_10 + MP20_11 + MP20_12 + MP20_13 + MP20_14 < rhs20 := by
  rw [MSv20_0, MSv20_1, MSv20_2, MSv20_3, MSv20_4, MSv20_5, MPv20_0, MPv20_1, MPv20_2, MPv20_3, MPv20_4, MPv20_5, MPv20_6, MPv20_7, MPv20_8, MPv20_9, MPv20_10, MPv20_11, MPv20_12, MPv20_13, MPv20_14, rhsv20]
  decide

def Dg20 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := (if c20_0 r0 t then 1 else 0) + (if c20_1 r1 t then 1 else 0) + (if c20_2 r2 t then 1 else 0) + (if c20_3 r3 t then 1 else 0) + (if c20_4 r4 t then 1 else 0) + (if c20_5 r5 t then 1 else 0)
def Wl20_0 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c20_0 r0 t && c20_1 r1 t then 1 else 0
def Wl20_1 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c20_0 r0 t && c20_2 r2 t then 1 else 0
def Wl20_2 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c20_0 r0 t && c20_3 r3 t then 1 else 0
def Wl20_3 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c20_0 r0 t && c20_4 r4 t then 1 else 0
def Wl20_4 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c20_0 r0 t && c20_5 r5 t then 1 else 0
def Wl20_5 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c20_0 r0 t && c20_1 r1 t && c20_2 r2 t then 1 else 0
def Wl20_6 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c20_0 r0 t && c20_1 r1 t && c20_3 r3 t then 1 else 0
def Wl20_7 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c20_0 r0 t && c20_1 r1 t && c20_4 r4 t then 1 else 0
def Wl20_8 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c20_0 r0 t && c20_1 r1 t && c20_5 r5 t then 1 else 0
def Wl20_9 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c20_0 r0 t && !c20_1 r1 t && c20_2 r2 t && c20_3 r3 t then 1 else 0
def Wl20_10 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c20_0 r0 t && !c20_1 r1 t && c20_2 r2 t && c20_4 r4 t then 1 else 0
def Wl20_11 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c20_0 r0 t && !c20_1 r1 t && c20_2 r2 t && c20_5 r5 t then 1 else 0
def Wl20_12 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c20_0 r0 t && !c20_1 r1 t && !c20_2 r2 t && c20_3 r3 t && c20_4 r4 t then 1 else 0
def Wl20_13 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c20_0 r0 t && !c20_1 r1 t && !c20_2 r2 t && c20_3 r3 t && c20_5 r5 t then 1 else 0
def Wl20_14 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c20_0 r0 t && !c20_1 r1 t && !c20_2 r2 t && !c20_3 r3 t && c20_4 r4 t && c20_5 r5 t then 1 else 0

/-- **No configuration blocks the whole window in case 20.** -/
theorem nocov20 {r0 r1 r2 r3 r4 r5 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29)
    (hcov : ∀ t, t < n20 → (c20_0 r0 t || c20_1 r1 t || c20_2 r2 t || c20_3 r3 t || c20_4 r4 t || c20_5 r5 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n20, (1 : ℤ) + (Wl20_0 r0 r1 r2 r3 r4 r5 t + Wl20_1 r0 r1 r2 r3 r4 r5 t + Wl20_2 r0 r1 r2 r3 r4 r5 t + Wl20_3 r0 r1 r2 r3 r4 r5 t + Wl20_4 r0 r1 r2 r3 r4 r5 t + Wl20_5 r0 r1 r2 r3 r4 r5 t + Wl20_6 r0 r1 r2 r3 r4 r5 t + Wl20_7 r0 r1 r2 r3 r4 r5 t + Wl20_8 r0 r1 r2 r3 r4 r5 t + Wl20_9 r0 r1 r2 r3 r4 r5 t + Wl20_10 r0 r1 r2 r3 r4 r5 t + Wl20_11 r0 r1 r2 r3 r4 r5 t + Wl20_12 r0 r1 r2 r3 r4 r5 t + Wl20_13 r0 r1 r2 r3 r4 r5 t + Wl20_14 r0 r1 r2 r3 r4 r5 t) ≤ Dg20 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Wl20_0, Wl20_1, Wl20_2, Wl20_3, Wl20_4, Wl20_5, Wl20_6, Wl20_7, Wl20_8, Wl20_9, Wl20_10, Wl20_11, Wl20_12, Wl20_13, Wl20_14, Dg20]
    exact CaseSplit.lowest6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n20, (1 : ℤ) ≤ Dg20 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Dg20]
    exact CaseSplit.degpos6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n20 : ℤ) + ((∑ t ∈ Finset.range n20, Wl20_0 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n20, Wl20_1 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n20, Wl20_2 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n20, Wl20_3 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n20, Wl20_4 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n20, Wl20_5 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n20, Wl20_6 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n20, Wl20_7 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n20, Wl20_8 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n20, Wl20_9 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n20, Wl20_10 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n20, Wl20_11 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n20, Wl20_12 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n20, Wl20_13 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n20, Wl20_14 r0 r1 r2 r3 r4 r5 t)) ≤ ∑ t ∈ Finset.range n20, Dg20 r0 r1 r2 r3 r4 r5 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N20_0 r0 r1 ≤ ∑ t ∈ Finset.range n20, Wl20_0 r0 r1 r2 r3 r4 r5 t := by
    simp only [N20_0, Wl20_0, le_refl]
  have hn1 : N20_1 r0 r2 ≤ ∑ t ∈ Finset.range n20, Wl20_1 r0 r1 r2 r3 r4 r5 t := by
    simp only [N20_1, Wl20_1, le_refl]
  have hn2 : N20_2 r0 r3 ≤ ∑ t ∈ Finset.range n20, Wl20_2 r0 r1 r2 r3 r4 r5 t := by
    simp only [N20_2, Wl20_2, le_refl]
  have hn3 : N20_3 r0 r4 ≤ ∑ t ∈ Finset.range n20, Wl20_3 r0 r1 r2 r3 r4 r5 t := by
    simp only [N20_3, Wl20_3, le_refl]
  have hn4 : N20_4 r0 r5 ≤ ∑ t ∈ Finset.range n20, Wl20_4 r0 r1 r2 r3 r4 r5 t := by
    simp only [N20_4, Wl20_4, le_refl]
  have hn5 : N20_5 r1 r2 ≤ ∑ t ∈ Finset.range n20, Wl20_5 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n20, Wl20_5 r0 r1 r2 r3 r4 r5 t
        = (if c20_1 r1 t && c20_2 r2 t then (1:ℤ) else 0)
          - (if c20_1 r1 t && c20_2 r2 t && c20_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl20_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n20, Wl20_5 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl20_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n20, Wl20_5 r0 r1 r2 r3 r4 r5 t
        = P20_5 r1 r2 - C20_5 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P20_5, C20_5]
    have hm : C20_5 r1 r2 r0 ≤ M20_5 r1 r2 :=
      CaseSplit.le_mxr (C20_5 r1 r2) 10 r0 (by omega)
    simp only [N20_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N20_6 r1 r3 ≤ ∑ t ∈ Finset.range n20, Wl20_6 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n20, Wl20_6 r0 r1 r2 r3 r4 r5 t
        = (if c20_1 r1 t && c20_3 r3 t then (1:ℤ) else 0)
          - (if c20_1 r1 t && c20_3 r3 t && c20_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl20_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n20, Wl20_6 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl20_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n20, Wl20_6 r0 r1 r2 r3 r4 r5 t
        = P20_6 r1 r3 - C20_6 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P20_6, C20_6]
    have hm : C20_6 r1 r3 r0 ≤ M20_6 r1 r3 :=
      CaseSplit.le_mxr (C20_6 r1 r3) 10 r0 (by omega)
    simp only [N20_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N20_7 r1 r4 ≤ ∑ t ∈ Finset.range n20, Wl20_7 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n20, Wl20_7 r0 r1 r2 r3 r4 r5 t
        = (if c20_1 r1 t && c20_4 r4 t then (1:ℤ) else 0)
          - (if c20_1 r1 t && c20_4 r4 t && c20_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl20_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n20, Wl20_7 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl20_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n20, Wl20_7 r0 r1 r2 r3 r4 r5 t
        = P20_7 r1 r4 - C20_7 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P20_7, C20_7]
    have hm : C20_7 r1 r4 r0 ≤ M20_7 r1 r4 :=
      CaseSplit.le_mxr (C20_7 r1 r4) 10 r0 (by omega)
    simp only [N20_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N20_8 r1 r5 ≤ ∑ t ∈ Finset.range n20, Wl20_8 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n20, Wl20_8 r0 r1 r2 r3 r4 r5 t
        = (if c20_1 r1 t && c20_5 r5 t then (1:ℤ) else 0)
          - (if c20_1 r1 t && c20_5 r5 t && c20_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl20_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n20, Wl20_8 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl20_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n20, Wl20_8 r0 r1 r2 r3 r4 r5 t
        = P20_8 r1 r5 - C20_8 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P20_8, C20_8]
    have hm : C20_8 r1 r5 r0 ≤ M20_8 r1 r5 :=
      CaseSplit.le_mxr (C20_8 r1 r5) 10 r0 (by omega)
    simp only [N20_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N20_9 r2 r3 ≤ ∑ t ∈ Finset.range n20, Wl20_9 r0 r1 r2 r3 r4 r5 t := by
    simp only [N20_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl20_9]
    exact CaseSplit.ind_nonneg _
  have hn10 : N20_10 r2 r4 ≤ ∑ t ∈ Finset.range n20, Wl20_10 r0 r1 r2 r3 r4 r5 t := by
    simp only [N20_10]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl20_10]
    exact CaseSplit.ind_nonneg _
  have hn11 : N20_11 r2 r5 ≤ ∑ t ∈ Finset.range n20, Wl20_11 r0 r1 r2 r3 r4 r5 t := by
    simp only [N20_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl20_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N20_12 r3 r4 ≤ ∑ t ∈ Finset.range n20, Wl20_12 r0 r1 r2 r3 r4 r5 t := by
    simp only [N20_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl20_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N20_13 r3 r5 ≤ ∑ t ∈ Finset.range n20, Wl20_13 r0 r1 r2 r3 r4 r5 t := by
    simp only [N20_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl20_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N20_14 r4 r5 ≤ ∑ t ∈ Finset.range n20, Wl20_14 r0 r1 r2 r3 r4 r5 t := by
    simp only [N20_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl20_14]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n20, (w20 t + 1) * Dg20 r0 r1 r2 r3 r4 r5 t = S20_0 r0 + S20_1 r1 + S20_2 r2 + S20_3 r3 + S20_4 r4 + S20_5 r5 := by
    simp only [S20_0, S20_1, S20_2, S20_3, S20_4, S20_5, Dg20, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n20, (w20 t + 1) * Dg20 r0 r1 r2 r3 r4 r5 t
      = (∑ t ∈ Finset.range n20, w20 t * Dg20 r0 r1 r2 r3 r4 r5 t)
        + 1 * (∑ t ∈ Finset.range n20, Dg20 r0 r1 r2 r3 r4 r5 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n20, w20 t)
      ≤ ∑ t ∈ Finset.range n20, w20 t * Dg20 r0 r1 r2 r3 r4 r5 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg20 r0 r1 r2 r3 r4 r5 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w20 t := wnn20 t (Finset.mem_range.mp ht)
    calc w20 t = w20 t * 1 := (mul_one _).symm
      _ ≤ w20 t * Dg20 r0 r1 r2 r3 r4 r5 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS20_0 r0 + aS20_1 r1 + aS20_2 r2 + aS20_3 r3 + aS20_4 r4 + aS20_5 r5) + (aP20_0 r0 r1 + aP20_1 r0 r2 + aP20_2 r0 r3 + aP20_3 r0 r4 + aP20_4 r0 r5 + aP20_5 r1 r2 + aP20_6 r1 r3 + aP20_7 r1 r4 + aP20_8 r1 r5 + aP20_9 r2 r3 + aP20_10 r2 r4 + aP20_11 r2 r5 + aP20_12 r3 r4 + aP20_13 r3 r5 + aP20_14 r4 r5) = (S20_0 r0 + S20_1 r1 + S20_2 r2 + S20_3 r3 + S20_4 r4 + S20_5 r5) - 1 * (N20_0 r0 r1 + N20_1 r0 r2 + N20_2 r0 r3 + N20_3 r0 r4 + N20_4 r0 r5 + N20_5 r1 r2 + N20_6 r1 r3 + N20_7 r1 r4 + N20_8 r1 r5 + N20_9 r2 r3 + N20_10 r2 r4 + N20_11 r2 r5 + N20_12 r3 r4 + N20_13 r3 r5 + N20_14 r4 r5) := by
    simp only [aS20_0, aS20_1, aS20_2, aS20_3, aS20_4, aS20_5, aP20_0, aP20_1, aP20_2, aP20_3, aP20_4, aP20_5, aP20_6, aP20_7, aP20_8, aP20_9, aP20_10, aP20_11, aP20_12, aP20_13, aP20_14, L20_0, L20_1, L20_2, L20_3, L20_4, L20_5]
    ring
  have bS0 : aS20_0 r0 ≤ MS20_0 := CaseSplit.le_mxr (aS20_0) 10 r0 (by omega)
  have bS1 : aS20_1 r1 ≤ MS20_1 := CaseSplit.le_mxr (aS20_1) 12 r1 (by omega)
  have bS2 : aS20_2 r2 ≤ MS20_2 := CaseSplit.le_mxr (aS20_2) 16 r2 (by omega)
  have bS3 : aS20_3 r3 ≤ MS20_3 := CaseSplit.le_mxr (aS20_3) 18 r3 (by omega)
  have bS4 : aS20_4 r4 ≤ MS20_4 := CaseSplit.le_mxr (aS20_4) 22 r4 (by omega)
  have bS5 : aS20_5 r5 ≤ MS20_5 := CaseSplit.le_mxr (aS20_5) 28 r5 (by omega)
  have bP0 : aP20_0 r0 r1 ≤ MP20_0 := CaseSplit.le_mxr2 (aP20_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP20_1 r0 r2 ≤ MP20_1 := CaseSplit.le_mxr2 (aP20_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP20_2 r0 r3 ≤ MP20_2 := CaseSplit.le_mxr2 (aP20_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP20_3 r0 r4 ≤ MP20_3 := CaseSplit.le_mxr2 (aP20_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP20_4 r0 r5 ≤ MP20_4 := CaseSplit.le_mxr2 (aP20_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP20_5 r1 r2 ≤ MP20_5 := CaseSplit.le_mxr2 (aP20_5) 12 16 r1 r2 (by omega) (by omega)
  have bP6 : aP20_6 r1 r3 ≤ MP20_6 := CaseSplit.le_mxr2 (aP20_6) 12 18 r1 r3 (by omega) (by omega)
  have bP7 : aP20_7 r1 r4 ≤ MP20_7 := CaseSplit.le_mxr2 (aP20_7) 12 22 r1 r4 (by omega) (by omega)
  have bP8 : aP20_8 r1 r5 ≤ MP20_8 := CaseSplit.le_mxr2 (aP20_8) 12 28 r1 r5 (by omega) (by omega)
  have bP9 : aP20_9 r2 r3 ≤ MP20_9 := CaseSplit.le_mxr2 (aP20_9) 16 18 r2 r3 (by omega) (by omega)
  have bP10 : aP20_10 r2 r4 ≤ MP20_10 := CaseSplit.le_mxr2 (aP20_10) 16 22 r2 r4 (by omega) (by omega)
  have bP11 : aP20_11 r2 r5 ≤ MP20_11 := CaseSplit.le_mxr2 (aP20_11) 16 28 r2 r5 (by omega) (by omega)
  have bP12 : aP20_12 r3 r4 ≤ MP20_12 := CaseSplit.le_mxr2 (aP20_12) 18 22 r3 r4 (by omega) (by omega)
  have bP13 : aP20_13 r3 r5 ≤ MP20_13 := CaseSplit.le_mxr2 (aP20_13) 18 28 r3 r5 (by omega) (by omega)
  have bP14 : aP20_14 r4 r5 ≤ MP20_14 := CaseSplit.le_mxr2 (aP20_14) 22 28 r4 r5 (by omega) (by omega)
  have hrhs : rhs20 = (∑ t ∈ Finset.range n20, w20 t) + 1 * (n20 : ℤ) := rfl
  have hc := cert20
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, bS0, bS1, bS2, bS3, bS4, bS5, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14]

end IncCert29
