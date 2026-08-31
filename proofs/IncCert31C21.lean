/-
INCREMENT-WIDTH CERTIFICATE, step 29->31, case 21 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_29_31.json, which re-derives every number
from the primes alone).

Machine 31, INCREMENT width 65 = F_2(29) + s_min(31) = 55 + 10,
held gears [5, 7] at phases [3, 0].  Free gears [11, 13, 17, 19, 23, 29, 31].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 10.
-/
import IncCert31B

namespace IncCert31

/-! ### case 21: held gears at phases [3, 0] -/

def p21 : List ℕ := [0, 2, 4, 5, 7, 9, 10, 12, 14, 17, 19, 24, 25, 30, 32, 35, 37, 39, 40, 42, 44, 45, 47, 49, 52, 54, 59, 60]
def q21 (t : ℕ) : ℕ := p21.getD t 0
def n21 : ℕ := 28
def yl21 : List ℤ := [3, 0, 0, 7, 10, 2, 5, 8, 2, 0, 0, 0, 0, 0, 3, 3, 7, 4, 6, 9, 6, 0, 6, 2, 2, 6, 2, 0]
def w21 (t : ℕ) : ℤ := yl21.getD t 0
def ul21 : List ℤ := [(-4), (-4), 0, (-4), 0, (-4), 0, (-4), 0, (-4), 1, 0, (-4), (-1), (-1), 4, (-1), 0, (-1), 0, (-1), 0, 4, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, (-1), (-1), (-1), 0, 0, 0, (-1), (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, (-5), (-5), (-5), (-5), 0, (-5), 0, (-5), (-5), (-5), 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 4, 0, 0, 0, 3, (-4), (-4), (-3), 0, (-4), (-4), (-4), (-4), (-4), (-3), 0, (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, 0, (-1), (-1), (-1), (-1), 0, (-1), 0, 0, (-1), 2, (-1), (-1), (-1), (-1), 0, (-1), 0, (-1), 0, 0, (-2), (-2), (-2), 0, 1, 0, 0, 0, 0, 35, 35, 28, 18, 21, 35, 35, 35, 23, 26, 26, 35, 35, 35, 24, 21, 20, (-35), (-35), (-35), (-35), (-35), (-35), (-35), (-35), (-35), (-35), (-35), (-35), (-35), 24, 23, 22, 29, 24, 17, 29, 18, 29, 28, 28, 29, 29, 23, 23, 29, 29, 29, 29, (-29), (-29), (-29), (-29), (-29), (-29), (-29), (-29), (-29), (-29), (-29), (-29), (-29), 24, 7, 25, 25, 8, 25, 18, 21, 25, 10, 25, 25, 20, 25, 25, 25, 7, 25, 25, 25, 25, 18, 12, (-25), (-25), (-25), (-25), (-25), (-25), (-25), (-25), (-25), (-25), (-25), (-25), (-25), 21, 21, 5, 21, 21, 21, 11, 3, 21, 21, 21, 19, 11, 9, 21, 21, 21, 21, 21, 21, 10, 21, 21, 21, 21, 5, 12, 21, 21, (-21), (-21), (-21), (-21), (-21), (-21), (-21), (-21), (-21), (-21), (-21), (-21), (-21), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-10), 0, 0, 0, (-8), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-7), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-10), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 18, 6, 23, 9, 16, 6, 10, 14, 6, 11, 5, 23, 22, 13, 23, 10, 23, 16, 20, 23, 12, 23, 11, 18, 22, 18, 23, 14, 22, 12, (-4), 2, 0, 0, (-2), (-8), 0, 2, 0, 2, (-9), 2, 2, 2, 2, 2, 2, 2, (-8), 2, 2, (-2), 2, 2, 2, 2, 0, (-6), 0]
def u21 (k : ℕ) : ℤ := ul21.getD k 0

def c21_0 (r t : ℕ) : Bool := gb11 r (q21 t)
def c21_1 (r t : ℕ) : Bool := gb13 r (q21 t)
def c21_2 (r t : ℕ) : Bool := gb17 r (q21 t)
def c21_3 (r t : ℕ) : Bool := gb19 r (q21 t)
def c21_4 (r t : ℕ) : Bool := gb23 r (q21 t)
def c21_5 (r t : ℕ) : Bool := gb29 r (q21 t)
def c21_6 (r t : ℕ) : Bool := gb31 r (q21 t)

def S21_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (w21 t + 5) * (if c21_0 r t then 1 else 0)
def S21_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (w21 t + 5) * (if c21_1 r t then 1 else 0)
def S21_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (w21 t + 5) * (if c21_2 r t then 1 else 0)
def S21_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (w21 t + 5) * (if c21_3 r t then 1 else 0)
def S21_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (w21 t + 5) * (if c21_4 r t then 1 else 0)
def S21_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (w21 t + 5) * (if c21_5 r t then 1 else 0)
def S21_6 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (w21 t + 5) * (if c21_6 r t then 1 else 0)

def L21_0 (r : ℕ) : ℤ := u21 (13 + r) + u21 (41 + r) + u21 (71 + r) + u21 (105 + r) + u21 (145 + r) + u21 (187 + r)
def L21_1 (r : ℕ) : ℤ := u21 (0 + r) + u21 (215 + r) + u21 (247 + r) + u21 (283 + r) + u21 (325 + r) + u21 (369 + r)
def L21_2 (r : ℕ) : ℤ := u21 (24 + r) + u21 (198 + r) + u21 (401 + r) + u21 (441 + r) + u21 (487 + r) + u21 (535 + r)
def L21_3 (r : ℕ) : ℤ := u21 (52 + r) + u21 (228 + r) + u21 (382 + r) + u21 (575 + r) + u21 (623 + r) + u21 (673 + r)
def L21_4 (r : ℕ) : ℤ := u21 (82 + r) + u21 (260 + r) + u21 (418 + r) + u21 (552 + r) + u21 (721 + r) + u21 (775 + r)
def L21_5 (r : ℕ) : ℤ := u21 (116 + r) + u21 (296 + r) + u21 (458 + r) + u21 (594 + r) + u21 (692 + r) + u21 (829 + r)
def L21_6 (r : ℕ) : ℤ := u21 (156 + r) + u21 (338 + r) + u21 (504 + r) + u21 (642 + r) + u21 (744 + r) + u21 (798 + r)

def aS21_0 (r : ℕ) : ℤ := S21_0 r - L21_0 r
def MS21_0 : ℤ := CaseSplit.mxr (aS21_0) 10
def aS21_1 (r : ℕ) : ℤ := S21_1 r - L21_1 r
def MS21_1 : ℤ := CaseSplit.mxr (aS21_1) 12
def aS21_2 (r : ℕ) : ℤ := S21_2 r - L21_2 r
def MS21_2 : ℤ := CaseSplit.mxr (aS21_2) 16
def aS21_3 (r : ℕ) : ℤ := S21_3 r - L21_3 r
def MS21_3 : ℤ := CaseSplit.mxr (aS21_3) 18
def aS21_4 (r : ℕ) : ℤ := S21_4 r - L21_4 r
def MS21_4 : ℤ := CaseSplit.mxr (aS21_4) 22
def aS21_5 (r : ℕ) : ℤ := S21_5 r - L21_5 r
def MS21_5 : ℤ := CaseSplit.mxr (aS21_5) 28
def aS21_6 (r : ℕ) : ℤ := S21_6 r - L21_6 r
def MS21_6 : ℤ := CaseSplit.mxr (aS21_6) 30

def N21_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (if c21_0 ra t && c21_1 rb t then 1 else 0)
def aP21_0 (ra rb : ℕ) : ℤ := -(5) * N21_0 ra rb + u21 (0 + rb) + u21 (13 + ra)
def MP21_0 : ℤ := CaseSplit.mxr2 (aP21_0) 10 12
def N21_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (if c21_0 ra t && c21_2 rb t then 1 else 0)
def aP21_1 (ra rb : ℕ) : ℤ := -(5) * N21_1 ra rb + u21 (24 + rb) + u21 (41 + ra)
def MP21_1 : ℤ := CaseSplit.mxr2 (aP21_1) 10 16
def N21_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (if c21_0 ra t && c21_3 rb t then 1 else 0)
def aP21_2 (ra rb : ℕ) : ℤ := -(5) * N21_2 ra rb + u21 (52 + rb) + u21 (71 + ra)
def MP21_2 : ℤ := CaseSplit.mxr2 (aP21_2) 10 18
def N21_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (if c21_0 ra t && c21_4 rb t then 1 else 0)
def aP21_3 (ra rb : ℕ) : ℤ := -(5) * N21_3 ra rb + u21 (82 + rb) + u21 (105 + ra)
def MP21_3 : ℤ := CaseSplit.mxr2 (aP21_3) 10 22
def N21_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (if c21_0 ra t && c21_5 rb t then 1 else 0)
def aP21_4 (ra rb : ℕ) : ℤ := -(5) * N21_4 ra rb + u21 (116 + rb) + u21 (145 + ra)
def MP21_4 : ℤ := CaseSplit.mxr2 (aP21_4) 10 28
def N21_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (if c21_0 ra t && c21_6 rb t then 1 else 0)
def aP21_5 (ra rb : ℕ) : ℤ := -(5) * N21_5 ra rb + u21 (156 + rb) + u21 (187 + ra)
def MP21_5 : ℤ := CaseSplit.mxr2 (aP21_5) 10 30
def P21_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (if c21_1 ra t && c21_2 rb t then 1 else 0)
def C21_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (if c21_1 ra t && c21_2 rb t && c21_0 s t then 1 else 0)
def M21_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C21_6 ra rb) 10
def E21_6 : List ℕ := [0, 11, 36, 47, 75, 81, 90, 101, 120, 126, 142, 148, 154, 165, 190, 201, 210, 216]
def N21_6 (ra rb : ℕ) : ℤ := if E21_6.contains (ra * 17 + rb) = true then P21_6 ra rb - M21_6 ra rb else 0
def aP21_6 (ra rb : ℕ) : ℤ := -(5) * N21_6 ra rb + u21 (198 + rb) + u21 (215 + ra)
def MP21_6 : ℤ := CaseSplit.mxr2 (aP21_6) 12 16
def P21_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (if c21_1 ra t && c21_3 rb t then 1 else 0)
def C21_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (if c21_1 ra t && c21_3 rb t && c21_0 s t then 1 else 0)
def M21_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C21_7 ra rb) 10
def E21_7 : List ℕ := [17, 40, 67, 70, 93, 98, 104, 138, 146, 151, 174, 180, 211, 214, 227, 238]
def N21_7 (ra rb : ℕ) : ℤ := if E21_7.contains (ra * 19 + rb) = true then P21_7 ra rb - M21_7 ra rb else 0
def aP21_7 (ra rb : ℕ) : ℤ := -(5) * N21_7 ra rb + u21 (228 + rb) + u21 (247 + ra)
def MP21_7 : ℤ := CaseSplit.mxr2 (aP21_7) 12 18
def P21_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (if c21_1 ra t && c21_4 rb t then 1 else 0)
def C21_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (if c21_1 ra t && c21_4 rb t && c21_0 s t then 1 else 0)
def M21_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C21_8 ra rb) 10
def E21_8 : List ℕ := []
def N21_8 (ra rb : ℕ) : ℤ := if E21_8.contains (ra * 23 + rb) = true then P21_8 ra rb - M21_8 ra rb else 0
def aP21_8 (ra rb : ℕ) : ℤ := -(5) * N21_8 ra rb + u21 (260 + rb) + u21 (283 + ra)
def MP21_8 : ℤ := CaseSplit.mxr2 (aP21_8) 12 22
def P21_9 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (if c21_1 ra t && c21_5 rb t then 1 else 0)
def C21_9 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (if c21_1 ra t && c21_5 rb t && c21_0 s t then 1 else 0)
def M21_9 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C21_9 ra rb) 10
def E21_9 : List ℕ := [43, 82, 159, 193, 309, 343]
def N21_9 (ra rb : ℕ) : ℤ := if E21_9.contains (ra * 29 + rb) = true then P21_9 ra rb - M21_9 ra rb else 0
def aP21_9 (ra rb : ℕ) : ℤ := -(5) * N21_9 ra rb + u21 (296 + rb) + u21 (325 + ra)
def MP21_9 : ℤ := CaseSplit.mxr2 (aP21_9) 12 28
def P21_10 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (if c21_1 ra t && c21_6 rb t then 1 else 0)
def C21_10 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (if c21_1 ra t && c21_6 rb t && c21_0 s t then 1 else 0)
def M21_10 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C21_10 ra rb) 10
def E21_10 : List ℕ := [3, 67, 153, 277, 282, 346]
def N21_10 (ra rb : ℕ) : ℤ := if E21_10.contains (ra * 31 + rb) = true then P21_10 ra rb - M21_10 ra rb else 0
def aP21_10 (ra rb : ℕ) : ℤ := -(5) * N21_10 ra rb + u21 (338 + rb) + u21 (369 + ra)
def MP21_10 : ℤ := CaseSplit.mxr2 (aP21_10) 12 30
def N21_11 (_ra _rb : ℕ) : ℤ := 0
def aP21_11 (ra rb : ℕ) : ℤ := -(5) * N21_11 ra rb + u21 (382 + rb) + u21 (401 + ra)
def MP21_11 : ℤ := CaseSplit.mxr2 (aP21_11) 16 18
def N21_12 (_ra _rb : ℕ) : ℤ := 0
def aP21_12 (ra rb : ℕ) : ℤ := -(5) * N21_12 ra rb + u21 (418 + rb) + u21 (441 + ra)
def MP21_12 : ℤ := CaseSplit.mxr2 (aP21_12) 16 22
def N21_13 (_ra _rb : ℕ) : ℤ := 0
def aP21_13 (ra rb : ℕ) : ℤ := -(5) * N21_13 ra rb + u21 (458 + rb) + u21 (487 + ra)
def MP21_13 : ℤ := CaseSplit.mxr2 (aP21_13) 16 28
def N21_14 (_ra _rb : ℕ) : ℤ := 0
def aP21_14 (ra rb : ℕ) : ℤ := -(5) * N21_14 ra rb + u21 (504 + rb) + u21 (535 + ra)
def MP21_14 : ℤ := CaseSplit.mxr2 (aP21_14) 16 30
def N21_15 (_ra _rb : ℕ) : ℤ := 0
def aP21_15 (ra rb : ℕ) : ℤ := -(5) * N21_15 ra rb + u21 (552 + rb) + u21 (575 + ra)
def MP21_15 : ℤ := CaseSplit.mxr2 (aP21_15) 18 22
def N21_16 (_ra _rb : ℕ) : ℤ := 0
def aP21_16 (ra rb : ℕ) : ℤ := -(5) * N21_16 ra rb + u21 (594 + rb) + u21 (623 + ra)
def MP21_16 : ℤ := CaseSplit.mxr2 (aP21_16) 18 28
def N21_17 (_ra _rb : ℕ) : ℤ := 0
def aP21_17 (ra rb : ℕ) : ℤ := -(5) * N21_17 ra rb + u21 (642 + rb) + u21 (673 + ra)
def MP21_17 : ℤ := CaseSplit.mxr2 (aP21_17) 18 30
def N21_18 (_ra _rb : ℕ) : ℤ := 0
def aP21_18 (ra rb : ℕ) : ℤ := -(5) * N21_18 ra rb + u21 (692 + rb) + u21 (721 + ra)
def MP21_18 : ℤ := CaseSplit.mxr2 (aP21_18) 22 28
def N21_19 (_ra _rb : ℕ) : ℤ := 0
def aP21_19 (ra rb : ℕ) : ℤ := -(5) * N21_19 ra rb + u21 (744 + rb) + u21 (775 + ra)
def MP21_19 : ℤ := CaseSplit.mxr2 (aP21_19) 22 30
def N21_20 (_ra _rb : ℕ) : ℤ := 0
def aP21_20 (ra rb : ℕ) : ℤ := -(5) * N21_20 ra rb + u21 (798 + rb) + u21 (829 + ra)
def MP21_20 : ℤ := CaseSplit.mxr2 (aP21_20) 28 30

def rhs21 : ℤ := (∑ t ∈ Finset.range n21, w21 t) + 5 * (n21 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn21 : ∀ t, t < n21 → (0 : ℤ) ≤ w21 t := by decide
theorem plt21 : ∀ t, t < n21 → q21 t < 65 := by decide
theorem pfree21_5 : ∀ t, t < n21 → gb5 3 (q21 t) = false := by decide
theorem pfree21_7 : ∀ t, t < n21 → gb7 0 (q21 t) = false := by decide
theorem MSv21_0 : MS21_0 = 52 := by decide +kernel
theorem MSv21_1 : MS21_1 = 150 := by decide +kernel
theorem MSv21_2 : MS21_2 = 1 := by decide +kernel
theorem MSv21_3 : MS21_3 = 1 := by decide +kernel
theorem MSv21_4 : MS21_4 = 1 := by decide +kernel
theorem MSv21_5 : MS21_5 = 0 := by decide +kernel
theorem MSv21_6 : MS21_6 = 1 := by decide +kernel
theorem MPv21_0 : MP21_0 = 0 := by decide +kernel
theorem MPv21_1 : MP21_1 = 0 := by decide +kernel
theorem MPv21_2 : MP21_2 = 0 := by decide +kernel
theorem MPv21_3 : MP21_3 = 0 := by decide +kernel
theorem MPv21_4 : MP21_4 = 0 := by decide +kernel
theorem MPv21_5 : MP21_5 = 0 := by decide +kernel
theorem MPv21_6 : MP21_6 = 0 := by decide +kernel
theorem MPv21_7 : MP21_7 = 0 := by decide +kernel
theorem MPv21_8 : MP21_8 = 0 := by decide +kernel
theorem MPv21_9 : MP21_9 = 0 := by decide +kernel
theorem MPv21_10 : MP21_10 = 0 := by decide +kernel
theorem MPv21_11 : MP21_11 = 0 := by decide +kernel
theorem MPv21_12 : MP21_12 = 0 := by decide +kernel
theorem MPv21_13 : MP21_13 = 0 := by decide +kernel
theorem MPv21_14 : MP21_14 = 0 := by decide +kernel
theorem MPv21_15 : MP21_15 = 0 := by decide +kernel
theorem MPv21_16 : MP21_16 = 0 := by decide +kernel
theorem MPv21_17 : MP21_17 = 0 := by decide +kernel
theorem MPv21_18 : MP21_18 = 0 := by decide +kernel
theorem MPv21_19 : MP21_19 = 0 := by decide +kernel
theorem MPv21_20 : MP21_20 = 25 := by decide +kernel
theorem rhsv21 : rhs21 = 233 := by decide +kernel

/-- **The case-21 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 2/10.
    (Scaled by the common denominator 10: 231 < 233.) -/
theorem cert21 : MS21_0 + MS21_1 + MS21_2 + MS21_3 + MS21_4 + MS21_5 + MS21_6 + MP21_0 + MP21_1 + MP21_2 + MP21_3 + MP21_4 + MP21_5 + MP21_6 + MP21_7 + MP21_8 + MP21_9 + MP21_10 + MP21_11 + MP21_12 + MP21_13 + MP21_14 + MP21_15 + MP21_16 + MP21_17 + MP21_18 + MP21_19 + MP21_20 < rhs21 := by
  rw [MSv21_0, MSv21_1, MSv21_2, MSv21_3, MSv21_4, MSv21_5, MSv21_6, MPv21_0, MPv21_1, MPv21_2, MPv21_3, MPv21_4, MPv21_5, MPv21_6, MPv21_7, MPv21_8, MPv21_9, MPv21_10, MPv21_11, MPv21_12, MPv21_13, MPv21_14, MPv21_15, MPv21_16, MPv21_17, MPv21_18, MPv21_19, MPv21_20, rhsv21]
  decide

def Dg21 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := (if c21_0 r0 t then 1 else 0) + (if c21_1 r1 t then 1 else 0) + (if c21_2 r2 t then 1 else 0) + (if c21_3 r3 t then 1 else 0) + (if c21_4 r4 t then 1 else 0) + (if c21_5 r5 t then 1 else 0) + (if c21_6 r6 t then 1 else 0)
def Wl21_0 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c21_0 r0 t && c21_1 r1 t then 1 else 0
def Wl21_1 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c21_0 r0 t && c21_2 r2 t then 1 else 0
def Wl21_2 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c21_0 r0 t && c21_3 r3 t then 1 else 0
def Wl21_3 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c21_0 r0 t && c21_4 r4 t then 1 else 0
def Wl21_4 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c21_0 r0 t && c21_5 r5 t then 1 else 0
def Wl21_5 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c21_0 r0 t && c21_6 r6 t then 1 else 0
def Wl21_6 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c21_0 r0 t && c21_1 r1 t && c21_2 r2 t then 1 else 0
def Wl21_7 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c21_0 r0 t && c21_1 r1 t && c21_3 r3 t then 1 else 0
def Wl21_8 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c21_0 r0 t && c21_1 r1 t && c21_4 r4 t then 1 else 0
def Wl21_9 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c21_0 r0 t && c21_1 r1 t && c21_5 r5 t then 1 else 0
def Wl21_10 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c21_0 r0 t && c21_1 r1 t && c21_6 r6 t then 1 else 0
def Wl21_11 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c21_0 r0 t && !c21_1 r1 t && c21_2 r2 t && c21_3 r3 t then 1 else 0
def Wl21_12 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c21_0 r0 t && !c21_1 r1 t && c21_2 r2 t && c21_4 r4 t then 1 else 0
def Wl21_13 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c21_0 r0 t && !c21_1 r1 t && c21_2 r2 t && c21_5 r5 t then 1 else 0
def Wl21_14 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c21_0 r0 t && !c21_1 r1 t && c21_2 r2 t && c21_6 r6 t then 1 else 0
def Wl21_15 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c21_0 r0 t && !c21_1 r1 t && !c21_2 r2 t && c21_3 r3 t && c21_4 r4 t then 1 else 0
def Wl21_16 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c21_0 r0 t && !c21_1 r1 t && !c21_2 r2 t && c21_3 r3 t && c21_5 r5 t then 1 else 0
def Wl21_17 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c21_0 r0 t && !c21_1 r1 t && !c21_2 r2 t && c21_3 r3 t && c21_6 r6 t then 1 else 0
def Wl21_18 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c21_0 r0 t && !c21_1 r1 t && !c21_2 r2 t && !c21_3 r3 t && c21_4 r4 t && c21_5 r5 t then 1 else 0
def Wl21_19 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c21_0 r0 t && !c21_1 r1 t && !c21_2 r2 t && !c21_3 r3 t && c21_4 r4 t && c21_6 r6 t then 1 else 0
def Wl21_20 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c21_0 r0 t && !c21_1 r1 t && !c21_2 r2 t && !c21_3 r3 t && !c21_4 r4 t && c21_5 r5 t && c21_6 r6 t then 1 else 0

/-- **No configuration blocks the whole window in case 21.** -/
theorem nocov21 {r0 r1 r2 r3 r4 r5 r6 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29) (h6 : r6 < 31)
    (hcov : ∀ t, t < n21 → (c21_0 r0 t || c21_1 r1 t || c21_2 r2 t || c21_3 r3 t || c21_4 r4 t || c21_5 r5 t || c21_6 r6 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n21, (1 : ℤ) + (Wl21_0 r0 r1 r2 r3 r4 r5 r6 t + Wl21_1 r0 r1 r2 r3 r4 r5 r6 t + Wl21_2 r0 r1 r2 r3 r4 r5 r6 t + Wl21_3 r0 r1 r2 r3 r4 r5 r6 t + Wl21_4 r0 r1 r2 r3 r4 r5 r6 t + Wl21_5 r0 r1 r2 r3 r4 r5 r6 t + Wl21_6 r0 r1 r2 r3 r4 r5 r6 t + Wl21_7 r0 r1 r2 r3 r4 r5 r6 t + Wl21_8 r0 r1 r2 r3 r4 r5 r6 t + Wl21_9 r0 r1 r2 r3 r4 r5 r6 t + Wl21_10 r0 r1 r2 r3 r4 r5 r6 t + Wl21_11 r0 r1 r2 r3 r4 r5 r6 t + Wl21_12 r0 r1 r2 r3 r4 r5 r6 t + Wl21_13 r0 r1 r2 r3 r4 r5 r6 t + Wl21_14 r0 r1 r2 r3 r4 r5 r6 t + Wl21_15 r0 r1 r2 r3 r4 r5 r6 t + Wl21_16 r0 r1 r2 r3 r4 r5 r6 t + Wl21_17 r0 r1 r2 r3 r4 r5 r6 t + Wl21_18 r0 r1 r2 r3 r4 r5 r6 t + Wl21_19 r0 r1 r2 r3 r4 r5 r6 t + Wl21_20 r0 r1 r2 r3 r4 r5 r6 t) ≤ Dg21 r0 r1 r2 r3 r4 r5 r6 t := by
    intro t ht
    simp only [Wl21_0, Wl21_1, Wl21_2, Wl21_3, Wl21_4, Wl21_5, Wl21_6, Wl21_7, Wl21_8, Wl21_9, Wl21_10, Wl21_11, Wl21_12, Wl21_13, Wl21_14, Wl21_15, Wl21_16, Wl21_17, Wl21_18, Wl21_19, Wl21_20, Dg21]
    exact CaseSplit.lowest7 _ _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n21, (1 : ℤ) ≤ Dg21 r0 r1 r2 r3 r4 r5 r6 t := by
    intro t ht
    simp only [Dg21]
    exact CaseSplit.degpos7 _ _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n21 : ℤ) + ((∑ t ∈ Finset.range n21, Wl21_0 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n21, Wl21_1 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n21, Wl21_2 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n21, Wl21_3 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n21, Wl21_4 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n21, Wl21_5 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n21, Wl21_6 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n21, Wl21_7 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n21, Wl21_8 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n21, Wl21_9 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n21, Wl21_10 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n21, Wl21_11 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n21, Wl21_12 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n21, Wl21_13 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n21, Wl21_14 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n21, Wl21_15 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n21, Wl21_16 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n21, Wl21_17 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n21, Wl21_18 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n21, Wl21_19 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n21, Wl21_20 r0 r1 r2 r3 r4 r5 r6 t)) ≤ ∑ t ∈ Finset.range n21, Dg21 r0 r1 r2 r3 r4 r5 r6 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N21_0 r0 r1 ≤ ∑ t ∈ Finset.range n21, Wl21_0 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N21_0, Wl21_0, le_refl]
  have hn1 : N21_1 r0 r2 ≤ ∑ t ∈ Finset.range n21, Wl21_1 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N21_1, Wl21_1, le_refl]
  have hn2 : N21_2 r0 r3 ≤ ∑ t ∈ Finset.range n21, Wl21_2 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N21_2, Wl21_2, le_refl]
  have hn3 : N21_3 r0 r4 ≤ ∑ t ∈ Finset.range n21, Wl21_3 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N21_3, Wl21_3, le_refl]
  have hn4 : N21_4 r0 r5 ≤ ∑ t ∈ Finset.range n21, Wl21_4 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N21_4, Wl21_4, le_refl]
  have hn5 : N21_5 r0 r6 ≤ ∑ t ∈ Finset.range n21, Wl21_5 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N21_5, Wl21_5, le_refl]
  have hn6 : N21_6 r1 r2 ≤ ∑ t ∈ Finset.range n21, Wl21_6 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n21, Wl21_6 r0 r1 r2 r3 r4 r5 r6 t
        = (if c21_1 r1 t && c21_2 r2 t then (1:ℤ) else 0)
          - (if c21_1 r1 t && c21_2 r2 t && c21_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl21_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n21, Wl21_6 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl21_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n21, Wl21_6 r0 r1 r2 r3 r4 r5 r6 t
        = P21_6 r1 r2 - C21_6 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P21_6, C21_6]
    have hm : C21_6 r1 r2 r0 ≤ M21_6 r1 r2 :=
      CaseSplit.le_mxr (C21_6 r1 r2) 10 r0 (by omega)
    simp only [N21_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N21_7 r1 r3 ≤ ∑ t ∈ Finset.range n21, Wl21_7 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n21, Wl21_7 r0 r1 r2 r3 r4 r5 r6 t
        = (if c21_1 r1 t && c21_3 r3 t then (1:ℤ) else 0)
          - (if c21_1 r1 t && c21_3 r3 t && c21_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl21_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n21, Wl21_7 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl21_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n21, Wl21_7 r0 r1 r2 r3 r4 r5 r6 t
        = P21_7 r1 r3 - C21_7 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P21_7, C21_7]
    have hm : C21_7 r1 r3 r0 ≤ M21_7 r1 r3 :=
      CaseSplit.le_mxr (C21_7 r1 r3) 10 r0 (by omega)
    simp only [N21_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N21_8 r1 r4 ≤ ∑ t ∈ Finset.range n21, Wl21_8 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n21, Wl21_8 r0 r1 r2 r3 r4 r5 r6 t
        = (if c21_1 r1 t && c21_4 r4 t then (1:ℤ) else 0)
          - (if c21_1 r1 t && c21_4 r4 t && c21_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl21_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n21, Wl21_8 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl21_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n21, Wl21_8 r0 r1 r2 r3 r4 r5 r6 t
        = P21_8 r1 r4 - C21_8 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P21_8, C21_8]
    have hm : C21_8 r1 r4 r0 ≤ M21_8 r1 r4 :=
      CaseSplit.le_mxr (C21_8 r1 r4) 10 r0 (by omega)
    simp only [N21_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N21_9 r1 r5 ≤ ∑ t ∈ Finset.range n21, Wl21_9 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n21, Wl21_9 r0 r1 r2 r3 r4 r5 r6 t
        = (if c21_1 r1 t && c21_5 r5 t then (1:ℤ) else 0)
          - (if c21_1 r1 t && c21_5 r5 t && c21_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl21_9]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n21, Wl21_9 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl21_9]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n21, Wl21_9 r0 r1 r2 r3 r4 r5 r6 t
        = P21_9 r1 r5 - C21_9 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P21_9, C21_9]
    have hm : C21_9 r1 r5 r0 ≤ M21_9 r1 r5 :=
      CaseSplit.le_mxr (C21_9 r1 r5) 10 r0 (by omega)
    simp only [N21_9]
    split
    · rw [hL]; omega
    · exact hnn
  have hn10 : N21_10 r1 r6 ≤ ∑ t ∈ Finset.range n21, Wl21_10 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n21, Wl21_10 r0 r1 r2 r3 r4 r5 r6 t
        = (if c21_1 r1 t && c21_6 r6 t then (1:ℤ) else 0)
          - (if c21_1 r1 t && c21_6 r6 t && c21_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl21_10]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n21, Wl21_10 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl21_10]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n21, Wl21_10 r0 r1 r2 r3 r4 r5 r6 t
        = P21_10 r1 r6 - C21_10 r1 r6 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P21_10, C21_10]
    have hm : C21_10 r1 r6 r0 ≤ M21_10 r1 r6 :=
      CaseSplit.le_mxr (C21_10 r1 r6) 10 r0 (by omega)
    simp only [N21_10]
    split
    · rw [hL]; omega
    · exact hnn
  have hn11 : N21_11 r2 r3 ≤ ∑ t ∈ Finset.range n21, Wl21_11 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N21_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl21_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N21_12 r2 r4 ≤ ∑ t ∈ Finset.range n21, Wl21_12 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N21_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl21_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N21_13 r2 r5 ≤ ∑ t ∈ Finset.range n21, Wl21_13 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N21_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl21_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N21_14 r2 r6 ≤ ∑ t ∈ Finset.range n21, Wl21_14 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N21_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl21_14]
    exact CaseSplit.ind_nonneg _
  have hn15 : N21_15 r3 r4 ≤ ∑ t ∈ Finset.range n21, Wl21_15 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N21_15]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl21_15]
    exact CaseSplit.ind_nonneg _
  have hn16 : N21_16 r3 r5 ≤ ∑ t ∈ Finset.range n21, Wl21_16 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N21_16]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl21_16]
    exact CaseSplit.ind_nonneg _
  have hn17 : N21_17 r3 r6 ≤ ∑ t ∈ Finset.range n21, Wl21_17 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N21_17]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl21_17]
    exact CaseSplit.ind_nonneg _
  have hn18 : N21_18 r4 r5 ≤ ∑ t ∈ Finset.range n21, Wl21_18 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N21_18]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl21_18]
    exact CaseSplit.ind_nonneg _
  have hn19 : N21_19 r4 r6 ≤ ∑ t ∈ Finset.range n21, Wl21_19 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N21_19]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl21_19]
    exact CaseSplit.ind_nonneg _
  have hn20 : N21_20 r5 r6 ≤ ∑ t ∈ Finset.range n21, Wl21_20 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N21_20]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl21_20]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n21, (w21 t + 5) * Dg21 r0 r1 r2 r3 r4 r5 r6 t = S21_0 r0 + S21_1 r1 + S21_2 r2 + S21_3 r3 + S21_4 r4 + S21_5 r5 + S21_6 r6 := by
    simp only [S21_0, S21_1, S21_2, S21_3, S21_4, S21_5, S21_6, Dg21, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n21, (w21 t + 5) * Dg21 r0 r1 r2 r3 r4 r5 r6 t
      = (∑ t ∈ Finset.range n21, w21 t * Dg21 r0 r1 r2 r3 r4 r5 r6 t)
        + 5 * (∑ t ∈ Finset.range n21, Dg21 r0 r1 r2 r3 r4 r5 r6 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n21, w21 t)
      ≤ ∑ t ∈ Finset.range n21, w21 t * Dg21 r0 r1 r2 r3 r4 r5 r6 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg21 r0 r1 r2 r3 r4 r5 r6 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w21 t := wnn21 t (Finset.mem_range.mp ht)
    calc w21 t = w21 t * 1 := (mul_one _).symm
      _ ≤ w21 t * Dg21 r0 r1 r2 r3 r4 r5 r6 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS21_0 r0 + aS21_1 r1 + aS21_2 r2 + aS21_3 r3 + aS21_4 r4 + aS21_5 r5 + aS21_6 r6) + (aP21_0 r0 r1 + aP21_1 r0 r2 + aP21_2 r0 r3 + aP21_3 r0 r4 + aP21_4 r0 r5 + aP21_5 r0 r6 + aP21_6 r1 r2 + aP21_7 r1 r3 + aP21_8 r1 r4 + aP21_9 r1 r5 + aP21_10 r1 r6 + aP21_11 r2 r3 + aP21_12 r2 r4 + aP21_13 r2 r5 + aP21_14 r2 r6 + aP21_15 r3 r4 + aP21_16 r3 r5 + aP21_17 r3 r6 + aP21_18 r4 r5 + aP21_19 r4 r6 + aP21_20 r5 r6) = (S21_0 r0 + S21_1 r1 + S21_2 r2 + S21_3 r3 + S21_4 r4 + S21_5 r5 + S21_6 r6) - 5 * (N21_0 r0 r1 + N21_1 r0 r2 + N21_2 r0 r3 + N21_3 r0 r4 + N21_4 r0 r5 + N21_5 r0 r6 + N21_6 r1 r2 + N21_7 r1 r3 + N21_8 r1 r4 + N21_9 r1 r5 + N21_10 r1 r6 + N21_11 r2 r3 + N21_12 r2 r4 + N21_13 r2 r5 + N21_14 r2 r6 + N21_15 r3 r4 + N21_16 r3 r5 + N21_17 r3 r6 + N21_18 r4 r5 + N21_19 r4 r6 + N21_20 r5 r6) := by
    simp only [aS21_0, aS21_1, aS21_2, aS21_3, aS21_4, aS21_5, aS21_6, aP21_0, aP21_1, aP21_2, aP21_3, aP21_4, aP21_5, aP21_6, aP21_7, aP21_8, aP21_9, aP21_10, aP21_11, aP21_12, aP21_13, aP21_14, aP21_15, aP21_16, aP21_17, aP21_18, aP21_19, aP21_20, L21_0, L21_1, L21_2, L21_3, L21_4, L21_5, L21_6]
    ring
  have bS0 : aS21_0 r0 ≤ MS21_0 := CaseSplit.le_mxr (aS21_0) 10 r0 (by omega)
  have bS1 : aS21_1 r1 ≤ MS21_1 := CaseSplit.le_mxr (aS21_1) 12 r1 (by omega)
  have bS2 : aS21_2 r2 ≤ MS21_2 := CaseSplit.le_mxr (aS21_2) 16 r2 (by omega)
  have bS3 : aS21_3 r3 ≤ MS21_3 := CaseSplit.le_mxr (aS21_3) 18 r3 (by omega)
  have bS4 : aS21_4 r4 ≤ MS21_4 := CaseSplit.le_mxr (aS21_4) 22 r4 (by omega)
  have bS5 : aS21_5 r5 ≤ MS21_5 := CaseSplit.le_mxr (aS21_5) 28 r5 (by omega)
  have bS6 : aS21_6 r6 ≤ MS21_6 := CaseSplit.le_mxr (aS21_6) 30 r6 (by omega)
  have bP0 : aP21_0 r0 r1 ≤ MP21_0 := CaseSplit.le_mxr2 (aP21_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP21_1 r0 r2 ≤ MP21_1 := CaseSplit.le_mxr2 (aP21_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP21_2 r0 r3 ≤ MP21_2 := CaseSplit.le_mxr2 (aP21_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP21_3 r0 r4 ≤ MP21_3 := CaseSplit.le_mxr2 (aP21_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP21_4 r0 r5 ≤ MP21_4 := CaseSplit.le_mxr2 (aP21_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP21_5 r0 r6 ≤ MP21_5 := CaseSplit.le_mxr2 (aP21_5) 10 30 r0 r6 (by omega) (by omega)
  have bP6 : aP21_6 r1 r2 ≤ MP21_6 := CaseSplit.le_mxr2 (aP21_6) 12 16 r1 r2 (by omega) (by omega)
  have bP7 : aP21_7 r1 r3 ≤ MP21_7 := CaseSplit.le_mxr2 (aP21_7) 12 18 r1 r3 (by omega) (by omega)
  have bP8 : aP21_8 r1 r4 ≤ MP21_8 := CaseSplit.le_mxr2 (aP21_8) 12 22 r1 r4 (by omega) (by omega)
  have bP9 : aP21_9 r1 r5 ≤ MP21_9 := CaseSplit.le_mxr2 (aP21_9) 12 28 r1 r5 (by omega) (by omega)
  have bP10 : aP21_10 r1 r6 ≤ MP21_10 := CaseSplit.le_mxr2 (aP21_10) 12 30 r1 r6 (by omega) (by omega)
  have bP11 : aP21_11 r2 r3 ≤ MP21_11 := CaseSplit.le_mxr2 (aP21_11) 16 18 r2 r3 (by omega) (by omega)
  have bP12 : aP21_12 r2 r4 ≤ MP21_12 := CaseSplit.le_mxr2 (aP21_12) 16 22 r2 r4 (by omega) (by omega)
  have bP13 : aP21_13 r2 r5 ≤ MP21_13 := CaseSplit.le_mxr2 (aP21_13) 16 28 r2 r5 (by omega) (by omega)
  have bP14 : aP21_14 r2 r6 ≤ MP21_14 := CaseSplit.le_mxr2 (aP21_14) 16 30 r2 r6 (by omega) (by omega)
  have bP15 : aP21_15 r3 r4 ≤ MP21_15 := CaseSplit.le_mxr2 (aP21_15) 18 22 r3 r4 (by omega) (by omega)
  have bP16 : aP21_16 r3 r5 ≤ MP21_16 := CaseSplit.le_mxr2 (aP21_16) 18 28 r3 r5 (by omega) (by omega)
  have bP17 : aP21_17 r3 r6 ≤ MP21_17 := CaseSplit.le_mxr2 (aP21_17) 18 30 r3 r6 (by omega) (by omega)
  have bP18 : aP21_18 r4 r5 ≤ MP21_18 := CaseSplit.le_mxr2 (aP21_18) 22 28 r4 r5 (by omega) (by omega)
  have bP19 : aP21_19 r4 r6 ≤ MP21_19 := CaseSplit.le_mxr2 (aP21_19) 22 30 r4 r6 (by omega) (by omega)
  have bP20 : aP21_20 r5 r6 ≤ MP21_20 := CaseSplit.le_mxr2 (aP21_20) 28 30 r5 r6 (by omega) (by omega)
  have hrhs : rhs21 = (∑ t ∈ Finset.range n21, w21 t) + 5 * (n21 : ℤ) := rfl
  have hc := cert21
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, hn15, hn16, hn17, hn18, hn19, hn20, bS0, bS1, bS2, bS3, bS4, bS5, bS6, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14, bP15, bP16, bP17, bP18, bP19, bP20]

end IncCert31
