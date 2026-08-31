/-
INCREMENT-WIDTH CERTIFICATE, step 23->29, case 18 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_23_29.json, which re-derives every number
from the primes alone).

Machine 29, INCREMENT width 49 = F_2(23) + s_min(29) = 39 + 10,
held gears [5, 7] at phases [2, 4].  Free gears [11, 13, 17, 19, 23, 29].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 2.
-/
import IncCert29B

namespace IncCert29

/-! ### case 18: held gears at phases [2, 4] -/

def p18 : List ℕ := [0, 1, 3, 5, 6, 8, 10, 13, 15, 20, 21, 26, 28, 31, 33, 35, 36, 38, 40, 41, 43, 45, 48]
def q18 (t : ℕ) : ℕ := p18.getD t 0
def n18 : ℕ := 23
def yl18 : List ℤ := [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0]
def w18 (t : ℕ) : ℤ := yl18.getD t 0
def ul18 : List ℤ := [(-4), (-2), 0, (-2), (-2), (-2), (-2), (-2), (-2), 0, (-2), 0, (-2), 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, (-1), (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), 0, 0, 0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), 0, (-1), (-1), 0, (-1), (-1), (-1), 0, (-1), (-1), 0, 0, (-1), (-1), (-1), 0, (-1), 0, (-1), (-1), (-1), 0, 0, (-1), (-1), 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 7, 4, 4, 6, 5, 7, 7, 5, 5, 6, 7, 7, 5, 7, 7, 5, 6, (-7), (-7), (-7), (-7), (-7), (-8), (-7), (-7), (-7), (-7), (-7), (-9), (-7), 7, 8, 8, 8, 3, 3, 7, 8, 7, 8, 5, 5, 5, 8, 8, 8, 8, 8, 3, (-8), (-8), (-8), (-8), (-8), (-8), (-8), (-8), (-9), (-8), (-8), (-8), (-8), 0, 7, 5, 2, 7, 7, 7, 5, 7, 7, 7, 7, 5, 7, 7, 7, 4, 6, 7, 7, 7, 3, 7, (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-5), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 3, 5, 5, 5, 3, 1, 5, 3, 3, 3, 4, 5, 5, 6, 4, 3, 6, 5, 6, 5, 6, 6, 5, 6, 4, 5, 5, 0, 0, 0, 0, 0, (-5), (-2), 0, 0, 0, 0, (-2), 0, 0, 0, (-5), 0, 0, 0, 0, (-5), 0, 0]
def u18 (k : ℕ) : ℤ := ul18.getD k 0

def c18_0 (r t : ℕ) : Bool := gb11 r (q18 t)
def c18_1 (r t : ℕ) : Bool := gb13 r (q18 t)
def c18_2 (r t : ℕ) : Bool := gb17 r (q18 t)
def c18_3 (r t : ℕ) : Bool := gb19 r (q18 t)
def c18_4 (r t : ℕ) : Bool := gb23 r (q18 t)
def c18_5 (r t : ℕ) : Bool := gb29 r (q18 t)

def S18_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (w18 t + 2) * (if c18_0 r t then 1 else 0)
def S18_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (w18 t + 2) * (if c18_1 r t then 1 else 0)
def S18_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (w18 t + 2) * (if c18_2 r t then 1 else 0)
def S18_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (w18 t + 2) * (if c18_3 r t then 1 else 0)
def S18_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (w18 t + 2) * (if c18_4 r t then 1 else 0)
def S18_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (w18 t + 2) * (if c18_5 r t then 1 else 0)

def L18_0 (r : ℕ) : ℤ := u18 (13 + r) + u18 (41 + r) + u18 (71 + r) + u18 (105 + r) + u18 (145 + r)
def L18_1 (r : ℕ) : ℤ := u18 (0 + r) + u18 (173 + r) + u18 (205 + r) + u18 (241 + r) + u18 (283 + r)
def L18_2 (r : ℕ) : ℤ := u18 (24 + r) + u18 (156 + r) + u18 (315 + r) + u18 (355 + r) + u18 (401 + r)
def L18_3 (r : ℕ) : ℤ := u18 (52 + r) + u18 (186 + r) + u18 (296 + r) + u18 (441 + r) + u18 (489 + r)
def L18_4 (r : ℕ) : ℤ := u18 (82 + r) + u18 (218 + r) + u18 (332 + r) + u18 (418 + r) + u18 (537 + r)
def L18_5 (r : ℕ) : ℤ := u18 (116 + r) + u18 (254 + r) + u18 (372 + r) + u18 (460 + r) + u18 (508 + r)

def aS18_0 (r : ℕ) : ℤ := S18_0 r - L18_0 r
def MS18_0 : ℤ := CaseSplit.mxr (aS18_0) 10
def aS18_1 (r : ℕ) : ℤ := S18_1 r - L18_1 r
def MS18_1 : ℤ := CaseSplit.mxr (aS18_1) 12
def aS18_2 (r : ℕ) : ℤ := S18_2 r - L18_2 r
def MS18_2 : ℤ := CaseSplit.mxr (aS18_2) 16
def aS18_3 (r : ℕ) : ℤ := S18_3 r - L18_3 r
def MS18_3 : ℤ := CaseSplit.mxr (aS18_3) 18
def aS18_4 (r : ℕ) : ℤ := S18_4 r - L18_4 r
def MS18_4 : ℤ := CaseSplit.mxr (aS18_4) 22
def aS18_5 (r : ℕ) : ℤ := S18_5 r - L18_5 r
def MS18_5 : ℤ := CaseSplit.mxr (aS18_5) 28

def N18_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (if c18_0 ra t && c18_1 rb t then 1 else 0)
def aP18_0 (ra rb : ℕ) : ℤ := -(2) * N18_0 ra rb + u18 (0 + rb) + u18 (13 + ra)
def MP18_0 : ℤ := CaseSplit.mxr2 (aP18_0) 10 12
def N18_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (if c18_0 ra t && c18_2 rb t then 1 else 0)
def aP18_1 (ra rb : ℕ) : ℤ := -(2) * N18_1 ra rb + u18 (24 + rb) + u18 (41 + ra)
def MP18_1 : ℤ := CaseSplit.mxr2 (aP18_1) 10 16
def N18_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (if c18_0 ra t && c18_3 rb t then 1 else 0)
def aP18_2 (ra rb : ℕ) : ℤ := -(2) * N18_2 ra rb + u18 (52 + rb) + u18 (71 + ra)
def MP18_2 : ℤ := CaseSplit.mxr2 (aP18_2) 10 18
def N18_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (if c18_0 ra t && c18_4 rb t then 1 else 0)
def aP18_3 (ra rb : ℕ) : ℤ := -(2) * N18_3 ra rb + u18 (82 + rb) + u18 (105 + ra)
def MP18_3 : ℤ := CaseSplit.mxr2 (aP18_3) 10 22
def N18_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (if c18_0 ra t && c18_5 rb t then 1 else 0)
def aP18_4 (ra rb : ℕ) : ℤ := -(2) * N18_4 ra rb + u18 (116 + rb) + u18 (145 + ra)
def MP18_4 : ℤ := CaseSplit.mxr2 (aP18_4) 10 28
def P18_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (if c18_1 ra t && c18_2 rb t then 1 else 0)
def C18_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (if c18_1 ra t && c18_2 rb t && c18_0 s t then 1 else 0)
def M18_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C18_5 ra rb) 10
def E18_5 : List ℕ := [61, 67, 102, 108, 136, 147, 156, 162, 192, 198]
def N18_5 (ra rb : ℕ) : ℤ := if E18_5.contains (ra * 17 + rb) = true then P18_5 ra rb - M18_5 ra rb else 0
def aP18_5 (ra rb : ℕ) : ℤ := -(2) * N18_5 ra rb + u18 (156 + rb) + u18 (173 + ra)
def MP18_5 : ℤ := CaseSplit.mxr2 (aP18_5) 12 16
def P18_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (if c18_1 ra t && c18_3 rb t then 1 else 0)
def C18_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (if c18_1 ra t && c18_3 rb t && c18_0 s t then 1 else 0)
def M18_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C18_6 ra rb) 10
def E18_6 : List ℕ := [7, 13, 41, 44, 47, 71, 78, 120, 147, 154, 178, 184, 212, 218]
def N18_6 (ra rb : ℕ) : ℤ := if E18_6.contains (ra * 19 + rb) = true then P18_6 ra rb - M18_6 ra rb else 0
def aP18_6 (ra rb : ℕ) : ℤ := -(2) * N18_6 ra rb + u18 (186 + rb) + u18 (205 + ra)
def MP18_6 : ℤ := CaseSplit.mxr2 (aP18_6) 12 18
def P18_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (if c18_1 ra t && c18_4 rb t then 1 else 0)
def C18_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (if c18_1 ra t && c18_4 rb t && c18_0 s t then 1 else 0)
def M18_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C18_7 ra rb) 10
def E18_7 : List ℕ := []
def N18_7 (ra rb : ℕ) : ℤ := if E18_7.contains (ra * 23 + rb) = true then P18_7 ra rb - M18_7 ra rb else 0
def aP18_7 (ra rb : ℕ) : ℤ := -(2) * N18_7 ra rb + u18 (218 + rb) + u18 (241 + ra)
def MP18_7 : ℤ := CaseSplit.mxr2 (aP18_7) 12 22
def P18_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (if c18_1 ra t && c18_5 rb t then 1 else 0)
def C18_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (if c18_1 ra t && c18_5 rb t && c18_0 s t then 1 else 0)
def M18_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C18_8 ra rb) 10
def E18_8 : List ℕ := [52, 163, 279, 313]
def N18_8 (ra rb : ℕ) : ℤ := if E18_8.contains (ra * 29 + rb) = true then P18_8 ra rb - M18_8 ra rb else 0
def aP18_8 (ra rb : ℕ) : ℤ := -(2) * N18_8 ra rb + u18 (254 + rb) + u18 (283 + ra)
def MP18_8 : ℤ := CaseSplit.mxr2 (aP18_8) 12 28
def N18_9 (_ra _rb : ℕ) : ℤ := 0
def aP18_9 (ra rb : ℕ) : ℤ := -(2) * N18_9 ra rb + u18 (296 + rb) + u18 (315 + ra)
def MP18_9 : ℤ := CaseSplit.mxr2 (aP18_9) 16 18
def N18_10 (_ra _rb : ℕ) : ℤ := 0
def aP18_10 (ra rb : ℕ) : ℤ := -(2) * N18_10 ra rb + u18 (332 + rb) + u18 (355 + ra)
def MP18_10 : ℤ := CaseSplit.mxr2 (aP18_10) 16 22
def N18_11 (_ra _rb : ℕ) : ℤ := 0
def aP18_11 (ra rb : ℕ) : ℤ := -(2) * N18_11 ra rb + u18 (372 + rb) + u18 (401 + ra)
def MP18_11 : ℤ := CaseSplit.mxr2 (aP18_11) 16 28
def N18_12 (_ra _rb : ℕ) : ℤ := 0
def aP18_12 (ra rb : ℕ) : ℤ := -(2) * N18_12 ra rb + u18 (418 + rb) + u18 (441 + ra)
def MP18_12 : ℤ := CaseSplit.mxr2 (aP18_12) 18 22
def N18_13 (_ra _rb : ℕ) : ℤ := 0
def aP18_13 (ra rb : ℕ) : ℤ := -(2) * N18_13 ra rb + u18 (460 + rb) + u18 (489 + ra)
def MP18_13 : ℤ := CaseSplit.mxr2 (aP18_13) 18 28
def N18_14 (_ra _rb : ℕ) : ℤ := 0
def aP18_14 (ra rb : ℕ) : ℤ := -(2) * N18_14 ra rb + u18 (508 + rb) + u18 (537 + ra)
def MP18_14 : ℤ := CaseSplit.mxr2 (aP18_14) 22 28

def rhs18 : ℤ := (∑ t ∈ Finset.range n18, w18 t) + 2 * (n18 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn18 : ∀ t, t < n18 → (0 : ℤ) ≤ w18 t := by decide
theorem plt18 : ∀ t, t < n18 → q18 t < 49 := by decide
theorem pfree18_5 : ∀ t, t < n18 → gb5 2 (q18 t) = false := by decide
theorem pfree18_7 : ∀ t, t < n18 → gb7 4 (q18 t) = false := by decide
theorem MSv18_0 : MS18_0 = 11 := by decide +kernel
theorem MSv18_1 : MS18_1 = 33 := by decide +kernel
theorem MSv18_2 : MS18_2 = 0 := by decide +kernel
theorem MSv18_3 : MS18_3 = 0 := by decide +kernel
theorem MSv18_4 : MS18_4 = 0 := by decide +kernel
theorem MSv18_5 : MS18_5 = 0 := by decide +kernel
theorem MPv18_0 : MP18_0 = 0 := by decide +kernel
theorem MPv18_1 : MP18_1 = 0 := by decide +kernel
theorem MPv18_2 : MP18_2 = 0 := by decide +kernel
theorem MPv18_3 : MP18_3 = 0 := by decide +kernel
theorem MPv18_4 : MP18_4 = 0 := by decide +kernel
theorem MPv18_5 : MP18_5 = 0 := by decide +kernel
theorem MPv18_6 : MP18_6 = 0 := by decide +kernel
theorem MPv18_7 : MP18_7 = 0 := by decide +kernel
theorem MPv18_8 : MP18_8 = 0 := by decide +kernel
theorem MPv18_9 : MP18_9 = 0 := by decide +kernel
theorem MPv18_10 : MP18_10 = 0 := by decide +kernel
theorem MPv18_11 : MP18_11 = 0 := by decide +kernel
theorem MPv18_12 : MP18_12 = 0 := by decide +kernel
theorem MPv18_13 : MP18_13 = 0 := by decide +kernel
theorem MPv18_14 : MP18_14 = 6 := by decide +kernel
theorem rhsv18 : rhs18 = 51 := by decide +kernel

/-- **The case-18 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/2.
    (Scaled by the common denominator 2: 50 < 51.) -/
theorem cert18 : MS18_0 + MS18_1 + MS18_2 + MS18_3 + MS18_4 + MS18_5 + MP18_0 + MP18_1 + MP18_2 + MP18_3 + MP18_4 + MP18_5 + MP18_6 + MP18_7 + MP18_8 + MP18_9 + MP18_10 + MP18_11 + MP18_12 + MP18_13 + MP18_14 < rhs18 := by
  rw [MSv18_0, MSv18_1, MSv18_2, MSv18_3, MSv18_4, MSv18_5, MPv18_0, MPv18_1, MPv18_2, MPv18_3, MPv18_4, MPv18_5, MPv18_6, MPv18_7, MPv18_8, MPv18_9, MPv18_10, MPv18_11, MPv18_12, MPv18_13, MPv18_14, rhsv18]
  decide

def Dg18 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := (if c18_0 r0 t then 1 else 0) + (if c18_1 r1 t then 1 else 0) + (if c18_2 r2 t then 1 else 0) + (if c18_3 r3 t then 1 else 0) + (if c18_4 r4 t then 1 else 0) + (if c18_5 r5 t then 1 else 0)
def Wl18_0 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c18_0 r0 t && c18_1 r1 t then 1 else 0
def Wl18_1 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c18_0 r0 t && c18_2 r2 t then 1 else 0
def Wl18_2 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c18_0 r0 t && c18_3 r3 t then 1 else 0
def Wl18_3 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c18_0 r0 t && c18_4 r4 t then 1 else 0
def Wl18_4 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if c18_0 r0 t && c18_5 r5 t then 1 else 0
def Wl18_5 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c18_0 r0 t && c18_1 r1 t && c18_2 r2 t then 1 else 0
def Wl18_6 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c18_0 r0 t && c18_1 r1 t && c18_3 r3 t then 1 else 0
def Wl18_7 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c18_0 r0 t && c18_1 r1 t && c18_4 r4 t then 1 else 0
def Wl18_8 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c18_0 r0 t && c18_1 r1 t && c18_5 r5 t then 1 else 0
def Wl18_9 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c18_0 r0 t && !c18_1 r1 t && c18_2 r2 t && c18_3 r3 t then 1 else 0
def Wl18_10 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c18_0 r0 t && !c18_1 r1 t && c18_2 r2 t && c18_4 r4 t then 1 else 0
def Wl18_11 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c18_0 r0 t && !c18_1 r1 t && c18_2 r2 t && c18_5 r5 t then 1 else 0
def Wl18_12 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c18_0 r0 t && !c18_1 r1 t && !c18_2 r2 t && c18_3 r3 t && c18_4 r4 t then 1 else 0
def Wl18_13 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c18_0 r0 t && !c18_1 r1 t && !c18_2 r2 t && c18_3 r3 t && c18_5 r5 t then 1 else 0
def Wl18_14 (r0 r1 r2 r3 r4 r5 t : ℕ) : ℤ := if !c18_0 r0 t && !c18_1 r1 t && !c18_2 r2 t && !c18_3 r3 t && c18_4 r4 t && c18_5 r5 t then 1 else 0

/-- **No configuration blocks the whole window in case 18.** -/
theorem nocov18 {r0 r1 r2 r3 r4 r5 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29)
    (hcov : ∀ t, t < n18 → (c18_0 r0 t || c18_1 r1 t || c18_2 r2 t || c18_3 r3 t || c18_4 r4 t || c18_5 r5 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n18, (1 : ℤ) + (Wl18_0 r0 r1 r2 r3 r4 r5 t + Wl18_1 r0 r1 r2 r3 r4 r5 t + Wl18_2 r0 r1 r2 r3 r4 r5 t + Wl18_3 r0 r1 r2 r3 r4 r5 t + Wl18_4 r0 r1 r2 r3 r4 r5 t + Wl18_5 r0 r1 r2 r3 r4 r5 t + Wl18_6 r0 r1 r2 r3 r4 r5 t + Wl18_7 r0 r1 r2 r3 r4 r5 t + Wl18_8 r0 r1 r2 r3 r4 r5 t + Wl18_9 r0 r1 r2 r3 r4 r5 t + Wl18_10 r0 r1 r2 r3 r4 r5 t + Wl18_11 r0 r1 r2 r3 r4 r5 t + Wl18_12 r0 r1 r2 r3 r4 r5 t + Wl18_13 r0 r1 r2 r3 r4 r5 t + Wl18_14 r0 r1 r2 r3 r4 r5 t) ≤ Dg18 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Wl18_0, Wl18_1, Wl18_2, Wl18_3, Wl18_4, Wl18_5, Wl18_6, Wl18_7, Wl18_8, Wl18_9, Wl18_10, Wl18_11, Wl18_12, Wl18_13, Wl18_14, Dg18]
    exact CaseSplit.lowest6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n18, (1 : ℤ) ≤ Dg18 r0 r1 r2 r3 r4 r5 t := by
    intro t ht
    simp only [Dg18]
    exact CaseSplit.degpos6 _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n18 : ℤ) + ((∑ t ∈ Finset.range n18, Wl18_0 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n18, Wl18_1 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n18, Wl18_2 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n18, Wl18_3 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n18, Wl18_4 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n18, Wl18_5 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n18, Wl18_6 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n18, Wl18_7 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n18, Wl18_8 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n18, Wl18_9 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n18, Wl18_10 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n18, Wl18_11 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n18, Wl18_12 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n18, Wl18_13 r0 r1 r2 r3 r4 r5 t) + (∑ t ∈ Finset.range n18, Wl18_14 r0 r1 r2 r3 r4 r5 t)) ≤ ∑ t ∈ Finset.range n18, Dg18 r0 r1 r2 r3 r4 r5 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N18_0 r0 r1 ≤ ∑ t ∈ Finset.range n18, Wl18_0 r0 r1 r2 r3 r4 r5 t := by
    simp only [N18_0, Wl18_0, le_refl]
  have hn1 : N18_1 r0 r2 ≤ ∑ t ∈ Finset.range n18, Wl18_1 r0 r1 r2 r3 r4 r5 t := by
    simp only [N18_1, Wl18_1, le_refl]
  have hn2 : N18_2 r0 r3 ≤ ∑ t ∈ Finset.range n18, Wl18_2 r0 r1 r2 r3 r4 r5 t := by
    simp only [N18_2, Wl18_2, le_refl]
  have hn3 : N18_3 r0 r4 ≤ ∑ t ∈ Finset.range n18, Wl18_3 r0 r1 r2 r3 r4 r5 t := by
    simp only [N18_3, Wl18_3, le_refl]
  have hn4 : N18_4 r0 r5 ≤ ∑ t ∈ Finset.range n18, Wl18_4 r0 r1 r2 r3 r4 r5 t := by
    simp only [N18_4, Wl18_4, le_refl]
  have hn5 : N18_5 r1 r2 ≤ ∑ t ∈ Finset.range n18, Wl18_5 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n18, Wl18_5 r0 r1 r2 r3 r4 r5 t
        = (if c18_1 r1 t && c18_2 r2 t then (1:ℤ) else 0)
          - (if c18_1 r1 t && c18_2 r2 t && c18_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl18_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n18, Wl18_5 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl18_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n18, Wl18_5 r0 r1 r2 r3 r4 r5 t
        = P18_5 r1 r2 - C18_5 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P18_5, C18_5]
    have hm : C18_5 r1 r2 r0 ≤ M18_5 r1 r2 :=
      CaseSplit.le_mxr (C18_5 r1 r2) 10 r0 (by omega)
    simp only [N18_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N18_6 r1 r3 ≤ ∑ t ∈ Finset.range n18, Wl18_6 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n18, Wl18_6 r0 r1 r2 r3 r4 r5 t
        = (if c18_1 r1 t && c18_3 r3 t then (1:ℤ) else 0)
          - (if c18_1 r1 t && c18_3 r3 t && c18_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl18_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n18, Wl18_6 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl18_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n18, Wl18_6 r0 r1 r2 r3 r4 r5 t
        = P18_6 r1 r3 - C18_6 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P18_6, C18_6]
    have hm : C18_6 r1 r3 r0 ≤ M18_6 r1 r3 :=
      CaseSplit.le_mxr (C18_6 r1 r3) 10 r0 (by omega)
    simp only [N18_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N18_7 r1 r4 ≤ ∑ t ∈ Finset.range n18, Wl18_7 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n18, Wl18_7 r0 r1 r2 r3 r4 r5 t
        = (if c18_1 r1 t && c18_4 r4 t then (1:ℤ) else 0)
          - (if c18_1 r1 t && c18_4 r4 t && c18_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl18_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n18, Wl18_7 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl18_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n18, Wl18_7 r0 r1 r2 r3 r4 r5 t
        = P18_7 r1 r4 - C18_7 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P18_7, C18_7]
    have hm : C18_7 r1 r4 r0 ≤ M18_7 r1 r4 :=
      CaseSplit.le_mxr (C18_7 r1 r4) 10 r0 (by omega)
    simp only [N18_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N18_8 r1 r5 ≤ ∑ t ∈ Finset.range n18, Wl18_8 r0 r1 r2 r3 r4 r5 t := by
    have hsp : ∀ t ∈ Finset.range n18, Wl18_8 r0 r1 r2 r3 r4 r5 t
        = (if c18_1 r1 t && c18_5 r5 t then (1:ℤ) else 0)
          - (if c18_1 r1 t && c18_5 r5 t && c18_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl18_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n18, Wl18_8 r0 r1 r2 r3 r4 r5 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl18_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n18, Wl18_8 r0 r1 r2 r3 r4 r5 t
        = P18_8 r1 r5 - C18_8 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P18_8, C18_8]
    have hm : C18_8 r1 r5 r0 ≤ M18_8 r1 r5 :=
      CaseSplit.le_mxr (C18_8 r1 r5) 10 r0 (by omega)
    simp only [N18_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N18_9 r2 r3 ≤ ∑ t ∈ Finset.range n18, Wl18_9 r0 r1 r2 r3 r4 r5 t := by
    simp only [N18_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl18_9]
    exact CaseSplit.ind_nonneg _
  have hn10 : N18_10 r2 r4 ≤ ∑ t ∈ Finset.range n18, Wl18_10 r0 r1 r2 r3 r4 r5 t := by
    simp only [N18_10]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl18_10]
    exact CaseSplit.ind_nonneg _
  have hn11 : N18_11 r2 r5 ≤ ∑ t ∈ Finset.range n18, Wl18_11 r0 r1 r2 r3 r4 r5 t := by
    simp only [N18_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl18_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N18_12 r3 r4 ≤ ∑ t ∈ Finset.range n18, Wl18_12 r0 r1 r2 r3 r4 r5 t := by
    simp only [N18_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl18_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N18_13 r3 r5 ≤ ∑ t ∈ Finset.range n18, Wl18_13 r0 r1 r2 r3 r4 r5 t := by
    simp only [N18_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl18_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N18_14 r4 r5 ≤ ∑ t ∈ Finset.range n18, Wl18_14 r0 r1 r2 r3 r4 r5 t := by
    simp only [N18_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl18_14]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n18, (w18 t + 2) * Dg18 r0 r1 r2 r3 r4 r5 t = S18_0 r0 + S18_1 r1 + S18_2 r2 + S18_3 r3 + S18_4 r4 + S18_5 r5 := by
    simp only [S18_0, S18_1, S18_2, S18_3, S18_4, S18_5, Dg18, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n18, (w18 t + 2) * Dg18 r0 r1 r2 r3 r4 r5 t
      = (∑ t ∈ Finset.range n18, w18 t * Dg18 r0 r1 r2 r3 r4 r5 t)
        + 2 * (∑ t ∈ Finset.range n18, Dg18 r0 r1 r2 r3 r4 r5 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n18, w18 t)
      ≤ ∑ t ∈ Finset.range n18, w18 t * Dg18 r0 r1 r2 r3 r4 r5 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg18 r0 r1 r2 r3 r4 r5 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w18 t := wnn18 t (Finset.mem_range.mp ht)
    calc w18 t = w18 t * 1 := (mul_one _).symm
      _ ≤ w18 t * Dg18 r0 r1 r2 r3 r4 r5 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS18_0 r0 + aS18_1 r1 + aS18_2 r2 + aS18_3 r3 + aS18_4 r4 + aS18_5 r5) + (aP18_0 r0 r1 + aP18_1 r0 r2 + aP18_2 r0 r3 + aP18_3 r0 r4 + aP18_4 r0 r5 + aP18_5 r1 r2 + aP18_6 r1 r3 + aP18_7 r1 r4 + aP18_8 r1 r5 + aP18_9 r2 r3 + aP18_10 r2 r4 + aP18_11 r2 r5 + aP18_12 r3 r4 + aP18_13 r3 r5 + aP18_14 r4 r5) = (S18_0 r0 + S18_1 r1 + S18_2 r2 + S18_3 r3 + S18_4 r4 + S18_5 r5) - 2 * (N18_0 r0 r1 + N18_1 r0 r2 + N18_2 r0 r3 + N18_3 r0 r4 + N18_4 r0 r5 + N18_5 r1 r2 + N18_6 r1 r3 + N18_7 r1 r4 + N18_8 r1 r5 + N18_9 r2 r3 + N18_10 r2 r4 + N18_11 r2 r5 + N18_12 r3 r4 + N18_13 r3 r5 + N18_14 r4 r5) := by
    simp only [aS18_0, aS18_1, aS18_2, aS18_3, aS18_4, aS18_5, aP18_0, aP18_1, aP18_2, aP18_3, aP18_4, aP18_5, aP18_6, aP18_7, aP18_8, aP18_9, aP18_10, aP18_11, aP18_12, aP18_13, aP18_14, L18_0, L18_1, L18_2, L18_3, L18_4, L18_5]
    ring
  have bS0 : aS18_0 r0 ≤ MS18_0 := CaseSplit.le_mxr (aS18_0) 10 r0 (by omega)
  have bS1 : aS18_1 r1 ≤ MS18_1 := CaseSplit.le_mxr (aS18_1) 12 r1 (by omega)
  have bS2 : aS18_2 r2 ≤ MS18_2 := CaseSplit.le_mxr (aS18_2) 16 r2 (by omega)
  have bS3 : aS18_3 r3 ≤ MS18_3 := CaseSplit.le_mxr (aS18_3) 18 r3 (by omega)
  have bS4 : aS18_4 r4 ≤ MS18_4 := CaseSplit.le_mxr (aS18_4) 22 r4 (by omega)
  have bS5 : aS18_5 r5 ≤ MS18_5 := CaseSplit.le_mxr (aS18_5) 28 r5 (by omega)
  have bP0 : aP18_0 r0 r1 ≤ MP18_0 := CaseSplit.le_mxr2 (aP18_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP18_1 r0 r2 ≤ MP18_1 := CaseSplit.le_mxr2 (aP18_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP18_2 r0 r3 ≤ MP18_2 := CaseSplit.le_mxr2 (aP18_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP18_3 r0 r4 ≤ MP18_3 := CaseSplit.le_mxr2 (aP18_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP18_4 r0 r5 ≤ MP18_4 := CaseSplit.le_mxr2 (aP18_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP18_5 r1 r2 ≤ MP18_5 := CaseSplit.le_mxr2 (aP18_5) 12 16 r1 r2 (by omega) (by omega)
  have bP6 : aP18_6 r1 r3 ≤ MP18_6 := CaseSplit.le_mxr2 (aP18_6) 12 18 r1 r3 (by omega) (by omega)
  have bP7 : aP18_7 r1 r4 ≤ MP18_7 := CaseSplit.le_mxr2 (aP18_7) 12 22 r1 r4 (by omega) (by omega)
  have bP8 : aP18_8 r1 r5 ≤ MP18_8 := CaseSplit.le_mxr2 (aP18_8) 12 28 r1 r5 (by omega) (by omega)
  have bP9 : aP18_9 r2 r3 ≤ MP18_9 := CaseSplit.le_mxr2 (aP18_9) 16 18 r2 r3 (by omega) (by omega)
  have bP10 : aP18_10 r2 r4 ≤ MP18_10 := CaseSplit.le_mxr2 (aP18_10) 16 22 r2 r4 (by omega) (by omega)
  have bP11 : aP18_11 r2 r5 ≤ MP18_11 := CaseSplit.le_mxr2 (aP18_11) 16 28 r2 r5 (by omega) (by omega)
  have bP12 : aP18_12 r3 r4 ≤ MP18_12 := CaseSplit.le_mxr2 (aP18_12) 18 22 r3 r4 (by omega) (by omega)
  have bP13 : aP18_13 r3 r5 ≤ MP18_13 := CaseSplit.le_mxr2 (aP18_13) 18 28 r3 r5 (by omega) (by omega)
  have bP14 : aP18_14 r4 r5 ≤ MP18_14 := CaseSplit.le_mxr2 (aP18_14) 22 28 r4 r5 (by omega) (by omega)
  have hrhs : rhs18 = (∑ t ∈ Finset.range n18, w18 t) + 2 * (n18 : ℤ) := rfl
  have hc := cert18
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, bS0, bS1, bS2, bS3, bS4, bS5, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14]

end IncCert29
