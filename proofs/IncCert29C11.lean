/-
INCREMENT-WIDTH CERTIFICATE, step 23->29, case 11 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_23_29.json, which re-derives every number
from the primes alone).

Machine 29, INCREMENT width 49 = F_2(23) + s_min(29) = 39 + 10,
held gears [5, 7] at phases [1, 4].  Free gears [11, 13, 17, 19, 23, 29].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 6.
-/
import IncCert29B

namespace IncCert29

/-! ### case 11: held gears at phases [1, 4] -/

def p11 : List ℕ := [1, 6, 7, 12, 14, 17, 19, 21, 22, 24, 26, 27, 29, 31, 34, 36, 41, 42, 47]
def q11 (t : ℕ) : ℕ := p11.getD t 0
def n11 : ℕ := 19
def yl11 : List ℤ := [0, 0, 0, 1, 3, 3, 4, 3, 6, 6, 6, 3, 4, 3, 3, 1, 0, 0, 0]
def w11 (t : ℕ) : ℤ := yl11.getD t 0
def ul11 : List ℤ := [0, 3, 0, 3, 0, 0, 0, 3, 0, 0, 3, 0, 0, (-3), 0, (-3), (-3), (-3), (-3), 0, (-3), 0, (-3), 0, (-3), (-3), (-3), (-3), (-3), (-3), (-3), 0, (-3), (-3), (-3), (-3), (-3), 0, (-3), (-3), (-3), 3, 3, 2, 0, 0, 2, 3, 3, 0, 0, 0, (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-3), (-2), (-2), (-2), (-2), 0, (-2), 0, (-2), (-2), (-2), 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 17, 16, 16, 14, 8, 11, 17, 17, 16, 11, 14, 11, 16, 17, 17, 11, 8, (-17), (-17), (-17), (-17), (-17), (-17), (-17), (-17), (-17), (-17), (-17), (-20), (-17), 13, 14, 10, 14, 14, 11, 9, 10, 14, 13, 14, 10, 14, 14, 13, 14, 14, 10, 14, (-14), (-16), (-14), (-14), (-14), (-14), (-17), (-14), (-14), (-14), (-14), (-14), (-14), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-9), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 1, 9, 15, 3, 15, 3, 12, 9, 1, 15, 3, 15, 6, 1, 8, 3, 13, 3, 6, 6, 3, 13, 3, 8, 1, 6, 15, 3, 15, 15, 6, 14, 0, 14, 9, 4, 15, 3, 6, 15, 3, 15, 4, 9, 14, 0, 14, 15, 11, 15, 0]
def u11 (k : ℕ) : ℤ := ul11.getD k 0

def c11_0 (r t : ℕ) : Bool := gb11 r (q11 t)
def c11_1 (r t : ℕ) : Bool := gb13 r (q11 t)
def c11_2 (r t : ℕ) : Bool := gb17 r (q11 t)
def c11_3 (r t : ℕ) : Bool := gb19 r (q11 t)
def c11_4 (r t : ℕ) : Bool := gb23 r (q11 t)
def c11_5 (r t : ℕ) : Bool := gb29 r (q11 t)

def S11_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (w11 t + 3) * (if c11_0 r t then 1 else 0)
def S11_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (w11 t + 3) * (if c11_1 r t then 1 else 0)
def S11_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (w11 t + 3) * (if c11_2 r t then 1 else 0)
def S11_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (w11 t + 3) * (if c11_3 r t then 1 else 0)
def S11_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (w11 t + 3) * (if c11_4 r t then 1 else 0)
def S11_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (w11 t + 3) * (if c11_5 r t then 1 else 0)

def L11_0 (r : ℕ) : ℤ := u11 (13 + r) + u11 (41 + r) + u11 (71 + r) + u11 (105 + r) + u11 (145 + r)
def L11_1 (r : ℕ) : ℤ := u11 (0 + r) + u11 (173 + r) + u11 (205 + r) + u11 (241 + r) + u11 (283 + r)
def L11_2 (r : ℕ) : ℤ := u11 (24 + r) + u11 (156 + r) + u11 (315 + r) + u11 (355 + r) + u11 (401 + r)
def L11_3 (r : ℕ) : ℤ := u11 (52 + r) + u11 (186 + r) + u11 (296 + r) + u11 (441 + r) + u11 (489 + r)
def L11_4 (r : ℕ) : ℤ := u11 (82 + r) + u11 (218 + r) + u11 (332 + r) + u11 (418 + r) + u11 (537 + r)
def L11_5 (r : ℕ) : ℤ := u11 (116 + r) + u11 (254 + r) + u11 (372 + r) + u11 (460 + r) + u11 (508 + r)

def aS11_0 (r : ℕ) : ℤ := S11_0 r - L11_0 r
def MS11_0 : ℤ := CaseSplit.mxr (aS11_0) 10
def aS11_1 (r : ℕ) : ℤ := S11_1 r - L11_1 r
def MS11_1 : ℤ := CaseSplit.mxr (aS11_1) 12
def aS11_2 (r : ℕ) : ℤ := S11_2 r - L11_2 r
def MS11_2 : ℤ := CaseSplit.mxr (aS11_2) 16
def aS11_3 (r : ℕ) : ℤ := S11_3 r - L11_3 r
def MS11_3 : ℤ := CaseSplit.mxr (aS11_3) 18
def aS11_4 (r : ℕ) : ℤ := S11_4 r - L11_4 r
def MS11_4 : ℤ := CaseSplit.mxr (aS11_4) 22
def aS11_5 (r : ℕ) : ℤ := S11_5 r - L11_5 r
def MS11_5 : ℤ := CaseSplit.mxr (aS11_5) 28

def N11_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (if c11_0 ra t && c11_1 rb t then 1 else 0)
def aP11_0 (ra rb : ℕ) : ℤ := -(3) * N11_0 ra rb + u11 (0 + rb) + u11 (13 + ra)
def MP11_0 : ℤ := CaseSplit.mxr2 (aP11_0) 10 12
def N11_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (if c11_0 ra t && c11_2 rb t then 1 else 0)
def aP11_1 (ra rb : ℕ) : ℤ := -(3) * N11_1 ra rb + u11 (24 + rb) + u11 (41 + ra)
def MP11_1 : ℤ := CaseSplit.mxr2 (aP11_1) 10 16
def N11_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (if c11_0 ra t && c11_3 rb t then 1 else 0)
def aP11_2 (ra rb : ℕ) : ℤ := -(3) * N11_2 ra rb + u11 (52 + rb) + u11 (71 + ra)
def MP11_2 : ℤ := CaseSplit.mxr2 (aP11_2) 10 18
def N11_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (if c11_0 ra t && c11_4 rb t then 1 else 0)
def aP11_3 (ra rb : ℕ) : ℤ := -(3) * N11_3 ra rb + u11 (82 + rb) + u11 (105 + ra)
def MP11_3 : ℤ := CaseSplit.mxr2 (aP11_3) 10 22
def N11_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (if c11_0 ra t && c11_5 rb t then 1 else 0)
def aP11_4 (ra rb : ℕ) : ℤ := -(3) * N11_4 ra rb + u11 (116 + rb) + u11 (145 + ra)
def MP11_4 : ℤ := CaseSplit.mxr2 (aP11_4) 10 28
def P11_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (if c11_1 ra t && c11_2 rb t then 1 else 0)
def C11_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (if c11_1 ra t && c11_2 rb t && c11_0 s t then 1 else 0)
def M11_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C11_5 ra rb) 10
def E11_5 : List ℕ := [7, 13, 75, 81, 86, 97, 122, 133, 170, 176, 206, 212]
def N11_5 (ra rb : ℕ) : ℤ := if E11_5.contains (ra * 17 + rb) = true then P11_5 ra rb - M11_5 ra rb else 0
def aP11_5 (ra rb : ℕ) : ℤ := -(3) * N11_5 ra rb + u11 (156 + rb) + u11 (173 + ra)
def MP11_5 : ℤ := CaseSplit.mxr2 (aP11_5) 12 16
def P11_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (if c11_1 ra t && c11_3 rb t then 1 else 0)
def C11_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (if c11_1 ra t && c11_3 rb t && c11_0 s t then 1 else 0)
def M11_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C11_6 ra rb) 10
def E11_6 : List ℕ := [21, 27, 58, 64, 111, 134, 140, 164, 187, 192, 198, 240]
def N11_6 (ra rb : ℕ) : ℤ := if E11_6.contains (ra * 19 + rb) = true then P11_6 ra rb - M11_6 ra rb else 0
def aP11_6 (ra rb : ℕ) : ℤ := -(3) * N11_6 ra rb + u11 (186 + rb) + u11 (205 + ra)
def MP11_6 : ℤ := CaseSplit.mxr2 (aP11_6) 12 18
def P11_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (if c11_1 ra t && c11_4 rb t then 1 else 0)
def C11_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (if c11_1 ra t && c11_4 rb t && c11_0 s t then 1 else 0)
def M11_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C11_7 ra rb) 10
def E11_7 : List ℕ := []
def N11_7 (ra rb : ℕ) : ℤ := if E11_7.contains (ra * 23 + rb) = true then P11_7 ra rb - M11_7 ra rb else 0
def aP11_7 (ra rb : ℕ) : ℤ := -(3) * N11_7 ra rb + u11 (218 + rb) + u11 (241 + ra)
def MP11_7 : ℤ := CaseSplit.mxr2 (aP11_7) 12 22
def P11_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (if c11_1 ra t && c11_5 rb t then 1 else 0)
def C11_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (if c11_1 ra t && c11_5 rb t && c11_0 s t then 1 else 0)
def M11_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C11_8 ra rb) 10
def E11_8 : List ℕ := []
def N11_8 (ra rb : ℕ) : ℤ := if E11_8.contains (ra * 29 + rb) = true then P11_8 ra rb - M11_8 ra rb else 0
def aP11_8 (ra rb : ℕ) : ℤ := -(3) * N11_8 ra rb + u11 (254 + rb) + u11 (283 + ra)
def MP11_8 : ℤ := CaseSplit.mxr2 (aP11_8) 12 28
def N11_9 (_ra _rb : ℕ) : ℤ := 0
def aP11_9 (ra rb : ℕ) : ℤ := -(3) * N11_9 ra rb + u11 (296 + rb) + u11 (315 + ra)
def MP11_9 : ℤ := CaseSplit.mxr2 (aP11_9) 16 18
def N11_10 (_ra _rb : ℕ) : ℤ := 0
def aP11_10 (ra rb : ℕ) : ℤ := -(3) * N11_10 ra rb + u11 (332 + rb) + u11 (355 + ra)
def MP11_10 : ℤ := CaseSplit.mxr2 (aP11_10) 16 22
def N11_11 (_ra _rb : ℕ) : ℤ := 0
def aP11_11 (ra rb : ℕ) : ℤ := -(3) * N11_11 ra rb + u11 (372 + rb) + u11 (401 + ra)
def MP11_11 : ℤ := CaseSplit.mxr2 (aP11_11) 16 28
def N11_12 (_ra _rb : ℕ) : ℤ := 0
def aP11_12 (ra rb : ℕ) : ℤ := -(3) * N11_12 ra rb + u11 (418 + rb) + u11 (441 + ra)
def MP11_12 : ℤ := CaseSplit.mxr2 (aP11_12) 18 22
def N11_13 (_ra _rb : ℕ) : ℤ := 0
def aP11_13 (ra rb : ℕ) : ℤ := -(3) * N11_13 ra rb + u11 (460 + rb) + u11 (489 + ra)
def MP11_13 : ℤ := CaseSplit.mxr2 (aP11_13) 18 28
def N11_14 (_ra _rb : ℕ) : ℤ := 0
def aP11_14 (ra rb : ℕ) : ℤ := -(3) * N11_14 ra rb + u11 (508 + rb) + u11 (537 + ra)
def MP11_14 : ℤ := CaseSplit.mxr2 (aP11_14) 22 28

def rhs11 : ℤ := (∑ t ∈ Finset.range n11, w11 t) + 3 * (n11 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn11 : ∀ t, t < n11 → (0 : ℤ) ≤ w11 t := by decide
theorem plt11 : ∀ t, t < n11 → q11 t < 49 := by decide
theorem pfree11_5 : ∀ t, t < n11 → gb5 1 (q11 t) = false := by decide
theorem pfree11_7 : ∀ t, t < n11 → gb7 4 (q11 t) = false := by decide
theorem MSv11_0 : MS11_0 = 19 := by decide +kernel
theorem MSv11_1 : MS11_1 = 49 := by decide +kernel
theorem MSv11_2 : MS11_2 = 1 := by decide +kernel
theorem MSv11_3 : MS11_3 = 1 := by decide +kernel
theorem MSv11_4 : MS11_4 = 1 := by decide +kernel
theorem MSv11_5 : MS11_5 = 1 := by decide +kernel
theorem MPv11_0 : MP11_0 = 0 := by decide +kernel
theorem MPv11_1 : MP11_1 = 0 := by decide +kernel
theorem MPv11_2 : MP11_2 = 0 := by decide +kernel
theorem MPv11_3 : MP11_3 = 0 := by decide +kernel
theorem MPv11_4 : MP11_4 = 0 := by decide +kernel
theorem MPv11_5 : MP11_5 = 0 := by decide +kernel
theorem MPv11_6 : MP11_6 = 0 := by decide +kernel
theorem MPv11_7 : MP11_7 = 0 := by decide +kernel
theorem MPv11_8 : MP11_8 = 0 := by decide +kernel
theorem MPv11_9 : MP11_9 = 0 := by decide +kernel
theorem MPv11_10 : MP11_10 = 0 := by decide +kernel
theorem MPv11_11 : MP11_11 = 0 := by decide +kernel
theorem MPv11_12 : MP11_12 = 0 := by decide +kernel
theorem MPv11_13 : MP11_13 = 0 := by decide +kernel
theorem MPv11_14 : MP11_14 = 30 := by decide +kernel
theorem rhsv11 : rhs11 = 103 := by decide +kernel

/-- **The case-11 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/6.
    (Scaled by the common denominator 6: 102 < 103.) -/
theorem cert11 : MS11_0 + MS11_1 + MS11_2 + MS11_3 + MS11_4 + MS11_5 + MP11_0 + MP11_1 + MP11_2 + MP11_3 + MP11_4 + MP11_5 + MP11_6 + MP11_7 + MP11_8 + MP11_9 + MP11_10 + MP11_11 + MP11_12 + MP11_13 + MP11_14 < rhs11 := by
  rw [MSv11_0, MSv11_1, MSv11_2, MSv11_3, MSv11_4, MSv11_5, MPv11_0, MPv11_1, MPv11_2, MPv11_3, MPv11_4, MPv11_5, MPv11_6, MPv11_7, MPv11_8, MPv11_9, MPv11_10, MPv11_11, MPv11_12, MPv11_13, MPv11_14, rhsv11]
  decide

def Dg11 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := (if c11_0 r0 t then 1 else 0) + (if c11_1 r1 t then 1 else 0) + (if c11_2 r2 t then 1 else 0) + (if c11_3 r3 t then 1 else 0) + (if c11_4 r4 t then 1 else 0) + (if c11_5 r5 t then 1 else 0)
def Wl11_0 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c11_0 r0 t && c11_1 r1 t then 1 else 0
def Wl11_1 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c11_0 r0 t && c11_2 r2 t then 1 else 0
def Wl11_2 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c11_0 r0 t && c11_3 r3 t then 1 else 0
def Wl11_3 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c11_0 r0 t && c11_4 r4 t then 1 else 0
def Wl11_4 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c11_0 r0 t && c11_5 r5 t then 1 else 0
def Wl11_5 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c11_0 r0 t && c11_1 r1 t && c11_2 r2 t then 1 else 0
def Wl11_6 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c11_0 r0 t && c11_1 r1 t && c11_3 r3 t then 1 else 0
def Wl11_7 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c11_0 r0 t && c11_1 r1 t && c11_4 r4 t then 1 else 0
def Wl11_8 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c11_0 r0 t && c11_1 r1 t && c11_5 r5 t then 1 else 0
def Wl11_9 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c11_0 r0 t && !c11_1 r1 t && c11_2 r2 t && c11_3 r3 t then 1 else 0
def Wl11_10 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c11_0 r0 t && !c11_1 r1 t && c11_2 r2 t && c11_4 r4 t then 1 else 0
def Wl11_11 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c11_0 r0 t && !c11_1 r1 t && c11_2 r2 t && c11_5 r5 t then 1 else 0
def Wl11_12 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c11_0 r0 t && !c11_1 r1 t && !c11_2 r2 t && c11_3 r3 t && c11_4 r4 t then 1 else 0
def Wl11_13 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c11_0 r0 t && !c11_1 r1 t && !c11_2 r2 t && c11_3 r3 t && c11_5 r5 t then 1 else 0
def Wl11_14 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c11_0 r0 t && !c11_1 r1 t && !c11_2 r2 t && !c11_3 r3 t && c11_4 r4 t && c11_5 r5 t then 1 else 0

/-- **No configuration blocks the whole window in case 11.** -/
theorem nocov11 {r0 r1 r2 r3 r4 r5 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29)
    (hcov : ∀ t, t < n11 → (c11_0 r0 t || c11_1 r1 t || c11_2 r2 t || c11_3 r3 t || c11_4 r4 t || c11_5 r5 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n11, (1 : ℤ) + (Wl11_0 r0 r1 r2 r3 r4 r5 t + Wl11_1 r0 r1 r2 r3 r4 r5 t + Wl11_2 r0 r1 r2 r3 r4 r5 t + Wl11_3 r0 r1 r2 r3 r4 r5 t + Wl11_4 r0 r1 r2 r3 r4 r5 t + Wl11_5 r0 r1 r2 r3 r4 r5 t + Wl11_6 r0 r1 r2 r3 r4 r5 t + Wl11_7 r0 r1 r2 r3 r4 r5 t + Wl11_8 r0 r1 r2 r3 r4 r5 t + Wl11_9 r0 r1 r2 r3 r4 r5 t + Wl11_10 r0 r1 r2 r3 r4 r5 t + Wl11_11 r0 r1 r2 r3 r4 r5 t + Wl11_12 r0 r1 r2 r3 r4 r5 t + Wl11_13 r0 r1 r2 r3 r4 r5 t + Wl11_14 r0 r1 r2 r3 r4 r5 t) ≤ Dg11 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Wl11_0, Wl11_1, Wl11_2, Wl11_3, Wl11_4, Wl11_5, Wl11_6, Wl11_7, Wl11_8, Wl11_9, Wl11_10, Wl11_11, Wl11_12, Wl11_13, Wl11_14, Dg11]
    exact CaseSplit.lowest6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n11, (1 : ℤ) ≤ Dg11 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Dg11]
    exact CaseSplit.degpos6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n11 : ℤ) + ((∑ t ∈ Finset.range n11, Wl11_0 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n11, Wl11_1 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n11, Wl11_2 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n11, Wl11_3 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n11, Wl11_4 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n11, Wl11_5 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n11, Wl11_6 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n11, Wl11_7 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n11, Wl11_8 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n11, Wl11_9 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n11, Wl11_10 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n11, Wl11_11 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n11, Wl11_12 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n11, Wl11_13 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n11, Wl11_14 r0 r1 r2 r3 r4 r5 t)) ≤ ∑ t ∈ Finset.range n11, Dg11 r0 r1 r2 r3 r4 r5 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N11_0 r0 r1 ≤ ∑ t ∈ Finset.range n11, Wl11_0 r0 r1 r2 r3 r4 r5 t := by
    simp only [N11_0, Wl11_0, le_refl]
  have hn1 : N11_1 r0 r2 ≤ ∑ t ∈ Finset.range n11, Wl11_1 r0 r1 r2 r3 r4 r5 t := by
    simp only [N11_1, Wl11_1, le_refl]
  have hn2 : N11_2 r0 r3 ≤ ∑ t ∈ Finset.range n11, Wl11_2 r0 r1 r2 r3 r4 r5 t := by
    simp only [N11_2, Wl11_2, le_refl]
  have hn3 : N11_3 r0 r4 ≤ ∑ t ∈ Finset.range n11, Wl11_3 r0 r1 r2 r3 r4 r5 t := by
    simp only [N11_3, Wl11_3, le_refl]
  have hn4 : N11_4 r0 r5 ≤ ∑ t ∈ Finset.range n11, Wl11_4 r0 r1 r2 r3 r4 r5 t := by
    simp only [N11_4, Wl11_4, le_refl]
  have hn5 : N11_5 r1 r2 ≤ ∑ t ∈ Finset.range n11, Wl11_5 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n11, Wl11_5 r0 r1 r2 r3 r4 r5 t
        = (if c11_1 r1 t && c11_2 r2 t then (1:ℤ) else 0)
          - (if c11_1 r1 t && c11_2 r2 t && c11_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl11_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n11, Wl11_5 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl11_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n11, Wl11_5 r0 r1 r2 r3 r4 r5 t
        = P11_5 r1 r2 - C11_5 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P11_5, C11_5]
    have hm : C11_5 r1 r2 r0 ≤ M11_5 r1 r2 :=
      CaseSplit.le_mxr (C11_5 r1 r2) 10 r0 (by omega)
    simp only [N11_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N11_6 r1 r3 ≤ ∑ t ∈ Finset.range n11, Wl11_6 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n11, Wl11_6 r0 r1 r2 r3 r4 r5 t
        = (if c11_1 r1 t && c11_3 r3 t then (1:ℤ) else 0)
          - (if c11_1 r1 t && c11_3 r3 t && c11_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl11_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n11, Wl11_6 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl11_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n11, Wl11_6 r0 r1 r2 r3 r4 r5 t
        = P11_6 r1 r3 - C11_6 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P11_6, C11_6]
    have hm : C11_6 r1 r3 r0 ≤ M11_6 r1 r3 :=
      CaseSplit.le_mxr (C11_6 r1 r3) 10 r0 (by omega)
    simp only [N11_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N11_7 r1 r4 ≤ ∑ t ∈ Finset.range n11, Wl11_7 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n11, Wl11_7 r0 r1 r2 r3 r4 r5 t
        = (if c11_1 r1 t && c11_4 r4 t then (1:ℤ) else 0)
          - (if c11_1 r1 t && c11_4 r4 t && c11_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl11_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n11, Wl11_7 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl11_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n11, Wl11_7 r0 r1 r2 r3 r4 r5 t
        = P11_7 r1 r4 - C11_7 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P11_7, C11_7]
    have hm : C11_7 r1 r4 r0 ≤ M11_7 r1 r4 :=
      CaseSplit.le_mxr (C11_7 r1 r4) 10 r0 (by omega)
    simp only [N11_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N11_8 r1 r5 ≤ ∑ t ∈ Finset.range n11, Wl11_8 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n11, Wl11_8 r0 r1 r2 r3 r4 r5 t
        = (if c11_1 r1 t && c11_5 r5 t then (1:ℤ) else 0)
          - (if c11_1 r1 t && c11_5 r5 t && c11_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl11_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n11, Wl11_8 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl11_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n11, Wl11_8 r0 r1 r2 r3 r4 r5 t
        = P11_8 r1 r5 - C11_8 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P11_8, C11_8]
    have hm : C11_8 r1 r5 r0 ≤ M11_8 r1 r5 :=
      CaseSplit.le_mxr (C11_8 r1 r5) 10 r0 (by omega)
    simp only [N11_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N11_9 r2 r3 ≤ ∑ t ∈ Finset.range n11, Wl11_9 r0 r1 r2 r3 r4 r5 t := by
    simp only [N11_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl11_9]
    exact CaseSplit.ind_nonneg _
  have hn10 : N11_10 r2 r4 ≤ ∑ t ∈ Finset.range n11, Wl11_10 r0 r1 r2 r3 r4 r5 t := by
    simp only [N11_10]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl11_10]
    exact CaseSplit.ind_nonneg _
  have hn11 : N11_11 r2 r5 ≤ ∑ t ∈ Finset.range n11, Wl11_11 r0 r1 r2 r3 r4 r5 t := by
    simp only [N11_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl11_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N11_12 r3 r4 ≤ ∑ t ∈ Finset.range n11, Wl11_12 r0 r1 r2 r3 r4 r5 t := by
    simp only [N11_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl11_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N11_13 r3 r5 ≤ ∑ t ∈ Finset.range n11, Wl11_13 r0 r1 r2 r3 r4 r5 t := by
    simp only [N11_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl11_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N11_14 r4 r5 ≤ ∑ t ∈ Finset.range n11, Wl11_14 r0 r1 r2 r3 r4 r5 t := by
    simp only [N11_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl11_14]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n11, (w11 t + 3) * Dg11 r0 r1 r2 r3 r4 r5 t = S11_0 r0 + S11_1 r1 + S11_2 r2 + S11_3 r3 + S11_4 r4 + S11_5 r5 := by
    simp only [S11_0, S11_1, S11_2, S11_3, S11_4, S11_5, Dg11, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n11, (w11 t + 3) * Dg11 r0 r1 r2 r3 r4 r5 t
      = (∑ t ∈ Finset.range n11, w11 t * Dg11 r0 r1 r2 r3 r4 r5 t)
        + 3 * (∑ t ∈ Finset.range n11, Dg11 r0 r1 r2 r3 r4 r5 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n11, w11 t)
      ≤ ∑ t ∈ Finset.range n11, w11 t * Dg11 r0 r1 r2 r3 r4 r5 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg11 r0 r1 r2 r3 r4 r5 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w11 t := wnn11 t (Finset.mem_range.mp ht)
    calc w11 t = w11 t * 1 := (mul_one _).symm
      _ ≤ w11 t * Dg11 r0 r1 r2 r3 r4 r5 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS11_0 r0 + aS11_1 r1 + aS11_2 r2 + aS11_3 r3 + aS11_4 r4 + aS11_5 r5) + (aP11_0 r0 r1 + aP11_1 r0 r2 + aP11_2 r0 r3 + aP11_3 r0 r4 + aP11_4 r0 r5 + aP11_5 r1 r2 + aP11_6 r1 r3 + aP11_7 r1 r4 + aP11_8 r1 r5 + aP11_9 r2 r3 + aP11_10 r2 r4 + aP11_11 r2 r5 + aP11_12 r3 r4 + aP11_13 r3 r5 + aP11_14 r4 r5) = (S11_0 r0 + S11_1 r1 + S11_2 r2 + S11_3 r3 + S11_4 r4 + S11_5 r5) - 3 * (N11_0 r0 r1 + N11_1 r0 r2 + N11_2 r0 r3 + N11_3 r0 r4 + N11_4 r0 r5 + N11_5 r1 r2 + N11_6 r1 r3 + N11_7 r1 r4 + N11_8 r1 r5 + N11_9 r2 r3 + N11_10 r2 r4 + N11_11 r2 r5 + N11_12 r3 r4 + N11_13 r3 r5 + N11_14 r4 r5) := by
    simp only [aS11_0, aS11_1, aS11_2, aS11_3, aS11_4, aS11_5, aP11_0, aP11_1, aP11_2, aP11_3, aP11_4, aP11_5, aP11_6, aP11_7, aP11_8, aP11_9, aP11_10, aP11_11, aP11_12, aP11_13, aP11_14, L11_0, L11_1, L11_2, L11_3, L11_4, L11_5]
    ring
  have bS0 : aS11_0 r0 ≤ MS11_0 := CaseSplit.le_mxr (aS11_0) 10 r0 (by omega)
  have bS1 : aS11_1 r1 ≤ MS11_1 := CaseSplit.le_mxr (aS11_1) 12 r1 (by omega)
  have bS2 : aS11_2 r2 ≤ MS11_2 := CaseSplit.le_mxr (aS11_2) 16 r2 (by omega)
  have bS3 : aS11_3 r3 ≤ MS11_3 := CaseSplit.le_mxr (aS11_3) 18 r3 (by omega)
  have bS4 : aS11_4 r4 ≤ MS11_4 := CaseSplit.le_mxr (aS11_4) 22 r4 (by omega)
  have bS5 : aS11_5 r5 ≤ MS11_5 := CaseSplit.le_mxr (aS11_5) 28 r5 (by omega)
  have bP0 : aP11_0 r0 r1 ≤ MP11_0 := CaseSplit.le_mxr2 (aP11_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP11_1 r0 r2 ≤ MP11_1 := CaseSplit.le_mxr2 (aP11_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP11_2 r0 r3 ≤ MP11_2 := CaseSplit.le_mxr2 (aP11_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP11_3 r0 r4 ≤ MP11_3 := CaseSplit.le_mxr2 (aP11_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP11_4 r0 r5 ≤ MP11_4 := CaseSplit.le_mxr2 (aP11_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP11_5 r1 r2 ≤ MP11_5 := CaseSplit.le_mxr2 (aP11_5) 12 16 r1 r2 (by omega) (by omega)
  have bP6 : aP11_6 r1 r3 ≤ MP11_6 := CaseSplit.le_mxr2 (aP11_6) 12 18 r1 r3 (by omega) (by omega)
  have bP7 : aP11_7 r1 r4 ≤ MP11_7 := CaseSplit.le_mxr2 (aP11_7) 12 22 r1 r4 (by omega) (by omega)
  have bP8 : aP11_8 r1 r5 ≤ MP11_8 := CaseSplit.le_mxr2 (aP11_8) 12 28 r1 r5 (by omega) (by omega)
  have bP9 : aP11_9 r2 r3 ≤ MP11_9 := CaseSplit.le_mxr2 (aP11_9) 16 18 r2 r3 (by omega) (by omega)
  have bP10 : aP11_10 r2 r4 ≤ MP11_10 := CaseSplit.le_mxr2 (aP11_10) 16 22 r2 r4 (by omega) (by omega)
  have bP11 : aP11_11 r2 r5 ≤ MP11_11 := CaseSplit.le_mxr2 (aP11_11) 16 28 r2 r5 (by omega) (by omega)
  have bP12 : aP11_12 r3 r4 ≤ MP11_12 := CaseSplit.le_mxr2 (aP11_12) 18 22 r3 r4 (by omega) (by omega)
  have bP13 : aP11_13 r3 r5 ≤ MP11_13 := CaseSplit.le_mxr2 (aP11_13) 18 28 r3 r5 (by omega) (by omega)
  have bP14 : aP11_14 r4 r5 ≤ MP11_14 := CaseSplit.le_mxr2 (aP11_14) 22 28 r4 r5 (by omega) (by omega)
  have hrhs : rhs11 = (∑ t ∈ Finset.range n11, w11 t) + 3 * (n11 : ℤ) := rfl
  have hc := cert11
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, bS0, bS1, bS2, bS3, bS4, bS5, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14]

end IncCert29
