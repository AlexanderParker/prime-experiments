/-
INCREMENT-WIDTH CERTIFICATE, step 29->31, case 27 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_29_31.json, which re-derives every number
from the primes alone).

Machine 31, INCREMENT width 65 = F_2(29) + s_min(31) = 55 + 10,
held gears [5, 7] at phases [3, 6].  Free gears [11, 13, 17, 19, 23, 29, 31].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 10.
-/
import IncCert31B

namespace IncCert31

/-! ### case 27: held gears at phases [3, 6] -/

def p27 : List ℕ := [4, 5, 10, 12, 15, 17, 19, 20, 22, 24, 25, 27, 29, 32, 34, 39, 40, 45, 47, 50, 52, 54, 55, 57, 59, 60, 62, 64]
def q27 (t : ℕ) : ℕ := p27.getD t 0
def n27 : ℕ := 28
def yl27 : List ℤ := [0, 2, 6, 2, 2, 6, 0, 6, 9, 6, 4, 7, 3, 3, 0, 0, 0, 0, 0, 2, 8, 5, 2, 10, 7, 0, 0, 3]
def w27 (t : ℕ) : ℤ := yl27.getD t 0
def ul27 : List ℤ := [(-5), (-5), (-5), (-1), 0, (-6), (-1), (-5), (-1), (-5), (-1), (-5), (-1), 5, 0, 0, 1, 5, 1, (-9), 1, 0, 1, 0, (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), 0, 0, (-1), 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-5), (-5), (-5), (-5), (-5), (-5), (-5), (-5), (-5), 0, (-5), (-5), (-5), (-5), 0, (-5), (-5), 0, (-5), (-12), (-5), (-5), 0, 0, 0, 5, 0, 0, 0, 5, 0, 5, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 4, (-3), (-4), (-4), 0, (-3), (-4), (-4), (-4), (-4), (-4), 0, (-2), (-2), (-2), (-2), (-2), (-2), (-2), 0, (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), 0, 2, 0, 2, 2, 2, 2, 2, 2, 0, 0, 22, 19, 29, 35, 35, 21, 21, 24, 35, 35, 35, 27, 35, 35, 35, 32, 35, (-35), (-35), (-35), (-35), (-35), (-35), (-35), (-35), (-35), (-35), (-35), (-35), (-35), 29, 29, 28, 28, 14, 18, 29, 17, 24, 21, 22, 29, 29, 29, 29, 29, 29, 29, 29, (-29), (-29), (-29), (-29), (-29), (-29), (-29), (-29), (-29), (-29), (-29), (-29), (-29), 30, 13, 29, 30, 12, 30, 17, 23, 30, 24, 30, 25, 12, 30, 30, 30, 30, 24, 30, 15, 30, 30, 23, (-30), (-30), (-34), (-30), (-30), (-30), (-30), (-30), (-30), (-30), (-30), (-30), (-30), 23, 23, 19, 23, 23, 23, 20, 23, 23, 23, 11, 23, 23, 23, 23, 23, 5, 23, 23, 19, 23, 23, 12, 23, 23, 23, 12, 7, 23, (-23), (-23), (-23), (-23), (-23), (-23), (-23), (-23), (-23), (-23), (-23), (-23), (-23), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-5), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-6), 0, 0, 0, 0, 0, 0, 24, 16, 24, 20, 24, 20, 13, 24, 14, 24, 22, 18, 24, 12, 23, 15, 24, 18, 7, 13, 7, 16, 12, 7, 17, 10, 24, 7, 20, 14, 14, 0, 0, 0, 0, 0, (-10), 0, 0, (-11), 0, 0, (-10), (-2), 0, 0, (-18), 0, 0, 0, 0, (-13), (-18), 0, (-6), 0, (-8), 0, 0, 0]
def u27 (k : ℕ) : ℤ := ul27.getD k 0

def c27_0 (r t : ℕ) : Bool := gb11 r (q27 t)
def c27_1 (r t : ℕ) : Bool := gb13 r (q27 t)
def c27_2 (r t : ℕ) : Bool := gb17 r (q27 t)
def c27_3 (r t : ℕ) : Bool := gb19 r (q27 t)
def c27_4 (r t : ℕ) : Bool := gb23 r (q27 t)
def c27_5 (r t : ℕ) : Bool := gb29 r (q27 t)
def c27_6 (r t : ℕ) : Bool := gb31 r (q27 t)

def S27_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (w27 t + 5) * (if c27_0 r t then 1 else 0)
def S27_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (w27 t + 5) * (if c27_1 r t then 1 else 0)
def S27_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (w27 t + 5) * (if c27_2 r t then 1 else 0)
def S27_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (w27 t + 5) * (if c27_3 r t then 1 else 0)
def S27_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (w27 t + 5) * (if c27_4 r t then 1 else 0)
def S27_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (w27 t + 5) * (if c27_5 r t then 1 else 0)
def S27_6 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (w27 t + 5) * (if c27_6 r t then 1 else 0)

def L27_0 (r : ℕ) : ℤ := u27 (13 + r) + u27 (41 + r) + u27 (71 + r) + u27 (105 + r) + u27 (145 + r) + u27 (187 + r)
def L27_1 (r : ℕ) : ℤ := u27 (0 + r) + u27 (215 + r) + u27 (247 + r) + u27 (283 + r) + u27 (325 + r) + u27 (369 + r)
def L27_2 (r : ℕ) : ℤ := u27 (24 + r) + u27 (198 + r) + u27 (401 + r) + u27 (441 + r) + u27 (487 + r) + u27 (535 + r)
def L27_3 (r : ℕ) : ℤ := u27 (52 + r) + u27 (228 + r) + u27 (382 + r) + u27 (575 + r) + u27 (623 + r) + u27 (673 + r)
def L27_4 (r : ℕ) : ℤ := u27 (82 + r) + u27 (260 + r) + u27 (418 + r) + u27 (552 + r) + u27 (721 + r) + u27 (775 + r)
def L27_5 (r : ℕ) : ℤ := u27 (116 + r) + u27 (296 + r) + u27 (458 + r) + u27 (594 + r) + u27 (692 + r) + u27 (829 + r)
def L27_6 (r : ℕ) : ℤ := u27 (156 + r) + u27 (338 + r) + u27 (504 + r) + u27 (642 + r) + u27 (744 + r) + u27 (798 + r)

def aS27_0 (r : ℕ) : ℤ := S27_0 r - L27_0 r
def MS27_0 : ℤ := CaseSplit.mxr (aS27_0) 10
def aS27_1 (r : ℕ) : ℤ := S27_1 r - L27_1 r
def MS27_1 : ℤ := CaseSplit.mxr (aS27_1) 12
def aS27_2 (r : ℕ) : ℤ := S27_2 r - L27_2 r
def MS27_2 : ℤ := CaseSplit.mxr (aS27_2) 16
def aS27_3 (r : ℕ) : ℤ := S27_3 r - L27_3 r
def MS27_3 : ℤ := CaseSplit.mxr (aS27_3) 18
def aS27_4 (r : ℕ) : ℤ := S27_4 r - L27_4 r
def MS27_4 : ℤ := CaseSplit.mxr (aS27_4) 22
def aS27_5 (r : ℕ) : ℤ := S27_5 r - L27_5 r
def MS27_5 : ℤ := CaseSplit.mxr (aS27_5) 28
def aS27_6 (r : ℕ) : ℤ := S27_6 r - L27_6 r
def MS27_6 : ℤ := CaseSplit.mxr (aS27_6) 30

def N27_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_0 ra t && c27_1 rb t then 1 else 0)
def aP27_0 (ra rb : ℕ) : ℤ := -(5) * N27_0 ra rb + u27 (0 + rb) + u27 (13 + ra)
def MP27_0 : ℤ := CaseSplit.mxr2 (aP27_0) 10 12
def N27_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_0 ra t && c27_2 rb t then 1 else 0)
def aP27_1 (ra rb : ℕ) : ℤ := -(5) * N27_1 ra rb + u27 (24 + rb) + u27 (41 + ra)
def MP27_1 : ℤ := CaseSplit.mxr2 (aP27_1) 10 16
def N27_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_0 ra t && c27_3 rb t then 1 else 0)
def aP27_2 (ra rb : ℕ) : ℤ := -(5) * N27_2 ra rb + u27 (52 + rb) + u27 (71 + ra)
def MP27_2 : ℤ := CaseSplit.mxr2 (aP27_2) 10 18
def N27_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_0 ra t && c27_4 rb t then 1 else 0)
def aP27_3 (ra rb : ℕ) : ℤ := -(5) * N27_3 ra rb + u27 (82 + rb) + u27 (105 + ra)
def MP27_3 : ℤ := CaseSplit.mxr2 (aP27_3) 10 22
def N27_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_0 ra t && c27_5 rb t then 1 else 0)
def aP27_4 (ra rb : ℕ) : ℤ := -(5) * N27_4 ra rb + u27 (116 + rb) + u27 (145 + ra)
def MP27_4 : ℤ := CaseSplit.mxr2 (aP27_4) 10 28
def N27_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_0 ra t && c27_6 rb t then 1 else 0)
def aP27_5 (ra rb : ℕ) : ℤ := -(5) * N27_5 ra rb + u27 (156 + rb) + u27 (187 + ra)
def MP27_5 : ℤ := CaseSplit.mxr2 (aP27_5) 10 30
def P27_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_1 ra t && c27_2 rb t then 1 else 0)
def C27_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_1 ra t && c27_2 rb t && c27_0 s t then 1 else 0)
def M27_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C27_6 ra rb) 10
def E27_6 : List ℕ := [21, 27, 43, 49, 52, 58, 88, 94, 111, 117, 122, 133, 158, 169, 178, 184, 206, 212]
def N27_6 (ra rb : ℕ) : ℤ := if E27_6.contains (ra * 17 + rb) = true then P27_6 ra rb - M27_6 ra rb else 0
def aP27_6 (ra rb : ℕ) : ℤ := -(5) * N27_6 ra rb + u27 (198 + rb) + u27 (215 + ra)
def MP27_6 : ℤ := CaseSplit.mxr2 (aP27_6) 12 16
def P27_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_1 ra t && c27_3 rb t then 1 else 0)
def C27_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_1 ra t && c27_3 rb t && c27_0 s t then 1 else 0)
def M27_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C27_7 ra rb) 10
def E27_7 : List ℕ := [33, 40, 64, 67, 70, 98, 104, 140, 146, 151, 174, 180, 204, 211, 227, 238]
def N27_7 (ra rb : ℕ) : ℤ := if E27_7.contains (ra * 19 + rb) = true then P27_7 ra rb - M27_7 ra rb else 0
def aP27_7 (ra rb : ℕ) : ℤ := -(5) * N27_7 ra rb + u27 (228 + rb) + u27 (247 + ra)
def MP27_7 : ℤ := CaseSplit.mxr2 (aP27_7) 12 18
def P27_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_1 ra t && c27_4 rb t then 1 else 0)
def C27_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_1 ra t && c27_4 rb t && c27_0 s t then 1 else 0)
def M27_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C27_8 ra rb) 10
def E27_8 : List ℕ := []
def N27_8 (ra rb : ℕ) : ℤ := if E27_8.contains (ra * 23 + rb) = true then P27_8 ra rb - M27_8 ra rb else 0
def aP27_8 (ra rb : ℕ) : ℤ := -(5) * N27_8 ra rb + u27 (260 + rb) + u27 (283 + ra)
def MP27_8 : ℤ := CaseSplit.mxr2 (aP27_8) 12 22
def P27_9 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_1 ra t && c27_5 rb t then 1 else 0)
def C27_9 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_1 ra t && c27_5 rb t && c27_0 s t then 1 else 0)
def M27_9 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C27_9 ra rb) 10
def E27_9 : List ℕ := [9, 115, 120, 236, 270, 376]
def N27_9 (ra rb : ℕ) : ℤ := if E27_9.contains (ra * 29 + rb) = true then P27_9 ra rb - M27_9 ra rb else 0
def aP27_9 (ra rb : ℕ) : ℤ := -(5) * N27_9 ra rb + u27 (296 + rb) + u27 (325 + ra)
def MP27_9 : ℤ := CaseSplit.mxr2 (aP27_9) 12 28
def P27_10 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_1 ra t && c27_6 rb t then 1 else 0)
def C27_10 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n27, (if c27_1 ra t && c27_6 rb t && c27_0 s t then 1 else 0)
def M27_10 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C27_10 ra rb) 10
def E27_10 : List ℕ := [57, 117, 181, 186, 310, 396]
def N27_10 (ra rb : ℕ) : ℤ := if E27_10.contains (ra * 31 + rb) = true then P27_10 ra rb - M27_10 ra rb else 0
def aP27_10 (ra rb : ℕ) : ℤ := -(5) * N27_10 ra rb + u27 (338 + rb) + u27 (369 + ra)
def MP27_10 : ℤ := CaseSplit.mxr2 (aP27_10) 12 30
def N27_11 (_ra _rb : ℕ) : ℤ := 0
def aP27_11 (ra rb : ℕ) : ℤ := -(5) * N27_11 ra rb + u27 (382 + rb) + u27 (401 + ra)
def MP27_11 : ℤ := CaseSplit.mxr2 (aP27_11) 16 18
def N27_12 (_ra _rb : ℕ) : ℤ := 0
def aP27_12 (ra rb : ℕ) : ℤ := -(5) * N27_12 ra rb + u27 (418 + rb) + u27 (441 + ra)
def MP27_12 : ℤ := CaseSplit.mxr2 (aP27_12) 16 22
def N27_13 (_ra _rb : ℕ) : ℤ := 0
def aP27_13 (ra rb : ℕ) : ℤ := -(5) * N27_13 ra rb + u27 (458 + rb) + u27 (487 + ra)
def MP27_13 : ℤ := CaseSplit.mxr2 (aP27_13) 16 28
def N27_14 (_ra _rb : ℕ) : ℤ := 0
def aP27_14 (ra rb : ℕ) : ℤ := -(5) * N27_14 ra rb + u27 (504 + rb) + u27 (535 + ra)
def MP27_14 : ℤ := CaseSplit.mxr2 (aP27_14) 16 30
def N27_15 (_ra _rb : ℕ) : ℤ := 0
def aP27_15 (ra rb : ℕ) : ℤ := -(5) * N27_15 ra rb + u27 (552 + rb) + u27 (575 + ra)
def MP27_15 : ℤ := CaseSplit.mxr2 (aP27_15) 18 22
def N27_16 (_ra _rb : ℕ) : ℤ := 0
def aP27_16 (ra rb : ℕ) : ℤ := -(5) * N27_16 ra rb + u27 (594 + rb) + u27 (623 + ra)
def MP27_16 : ℤ := CaseSplit.mxr2 (aP27_16) 18 28
def N27_17 (_ra _rb : ℕ) : ℤ := 0
def aP27_17 (ra rb : ℕ) : ℤ := -(5) * N27_17 ra rb + u27 (642 + rb) + u27 (673 + ra)
def MP27_17 : ℤ := CaseSplit.mxr2 (aP27_17) 18 30
def N27_18 (_ra _rb : ℕ) : ℤ := 0
def aP27_18 (ra rb : ℕ) : ℤ := -(5) * N27_18 ra rb + u27 (692 + rb) + u27 (721 + ra)
def MP27_18 : ℤ := CaseSplit.mxr2 (aP27_18) 22 28
def N27_19 (_ra _rb : ℕ) : ℤ := 0
def aP27_19 (ra rb : ℕ) : ℤ := -(5) * N27_19 ra rb + u27 (744 + rb) + u27 (775 + ra)
def MP27_19 : ℤ := CaseSplit.mxr2 (aP27_19) 22 30
def N27_20 (_ra _rb : ℕ) : ℤ := 0
def aP27_20 (ra rb : ℕ) : ℤ := -(5) * N27_20 ra rb + u27 (798 + rb) + u27 (829 + ra)
def MP27_20 : ℤ := CaseSplit.mxr2 (aP27_20) 28 30

def rhs27 : ℤ := (∑ t ∈ Finset.range n27, w27 t) + 5 * (n27 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn27 : ∀ t, t < n27 → (0 : ℤ) ≤ w27 t := by decide
theorem plt27 : ∀ t, t < n27 → q27 t < 65 := by decide
theorem pfree27_5 : ∀ t, t < n27 → gb5 3 (q27 t) = false := by decide
theorem pfree27_7 : ∀ t, t < n27 → gb7 6 (q27 t) = false := by decide
theorem MSv27_0 : MS27_0 = 44 := by decide +kernel
theorem MSv27_1 : MS27_1 = 158 := by decide +kernel
theorem MSv27_2 : MS27_2 = 1 := by decide +kernel
theorem MSv27_3 : MS27_3 = 1 := by decide +kernel
theorem MSv27_4 : MS27_4 = 1 := by decide +kernel
theorem MSv27_5 : MS27_5 = 0 := by decide +kernel
theorem MSv27_6 : MS27_6 = 1 := by decide +kernel
theorem MPv27_0 : MP27_0 = 0 := by decide +kernel
theorem MPv27_1 : MP27_1 = 0 := by decide +kernel
theorem MPv27_2 : MP27_2 = 0 := by decide +kernel
theorem MPv27_3 : MP27_3 = 0 := by decide +kernel
theorem MPv27_4 : MP27_4 = 0 := by decide +kernel
theorem MPv27_5 : MP27_5 = 0 := by decide +kernel
theorem MPv27_6 : MP27_6 = 0 := by decide +kernel
theorem MPv27_7 : MP27_7 = 0 := by decide +kernel
theorem MPv27_8 : MP27_8 = 0 := by decide +kernel
theorem MPv27_9 : MP27_9 = 0 := by decide +kernel
theorem MPv27_10 : MP27_10 = 0 := by decide +kernel
theorem MPv27_11 : MP27_11 = 0 := by decide +kernel
theorem MPv27_12 : MP27_12 = 0 := by decide +kernel
theorem MPv27_13 : MP27_13 = 0 := by decide +kernel
theorem MPv27_14 : MP27_14 = 0 := by decide +kernel
theorem MPv27_15 : MP27_15 = 0 := by decide +kernel
theorem MPv27_16 : MP27_16 = 0 := by decide +kernel
theorem MPv27_17 : MP27_17 = 0 := by decide +kernel
theorem MPv27_18 : MP27_18 = 0 := by decide +kernel
theorem MPv27_19 : MP27_19 = 0 := by decide +kernel
theorem MPv27_20 : MP27_20 = 24 := by decide +kernel
theorem rhsv27 : rhs27 = 233 := by decide +kernel

/-- **The case-27 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 3/10.
    (Scaled by the common denominator 10: 230 < 233.) -/
theorem cert27 : MS27_0 + MS27_1 + MS27_2 + MS27_3 + MS27_4 + MS27_5 + MS27_6 + MP27_0 + MP27_1 + MP27_2 + MP27_3 + MP27_4 + MP27_5 + MP27_6 + MP27_7 + MP27_8 + MP27_9 + MP27_10 + MP27_11 + MP27_12 + MP27_13 + MP27_14 + MP27_15 + MP27_16 + MP27_17 + MP27_18 + MP27_19 + MP27_20 < rhs27 := by
  rw [MSv27_0, MSv27_1, MSv27_2, MSv27_3, MSv27_4, MSv27_5, MSv27_6, MPv27_0, MPv27_1, MPv27_2, MPv27_3, MPv27_4, MPv27_5, MPv27_6, MPv27_7, MPv27_8, MPv27_9, MPv27_10, MPv27_11, MPv27_12, MPv27_13, MPv27_14, MPv27_15, MPv27_16, MPv27_17, MPv27_18, MPv27_19, MPv27_20, rhsv27]
  decide

def Dg27 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := (if c27_0 r0 t then 1 else 0) + (if c27_1 r1 t then 1 else 0) + (if c27_2 r2 t then 1 else 0) + (if c27_3 r3 t then 1 else 0) + (if c27_4 r4 t then 1 else 0) + (if c27_5 r5 t then 1 else 0) + (if c27_6 r6 t then 1 else 0)
def Wl27_0 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c27_0 r0 t && c27_1 r1 t then 1 else 0
def Wl27_1 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c27_0 r0 t && c27_2 r2 t then 1 else 0
def Wl27_2 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c27_0 r0 t && c27_3 r3 t then 1 else 0
def Wl27_3 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c27_0 r0 t && c27_4 r4 t then 1 else 0
def Wl27_4 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c27_0 r0 t && c27_5 r5 t then 1 else 0
def Wl27_5 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c27_0 r0 t && c27_6 r6 t then 1 else 0
def Wl27_6 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c27_0 r0 t && c27_1 r1 t && c27_2 r2 t then 1 else 0
def Wl27_7 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c27_0 r0 t && c27_1 r1 t && c27_3 r3 t then 1 else 0
def Wl27_8 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c27_0 r0 t && c27_1 r1 t && c27_4 r4 t then 1 else 0
def Wl27_9 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c27_0 r0 t && c27_1 r1 t && c27_5 r5 t then 1 else 0
def Wl27_10 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c27_0 r0 t && c27_1 r1 t && c27_6 r6 t then 1 else 0
def Wl27_11 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c27_0 r0 t && !c27_1 r1 t && c27_2 r2 t && c27_3 r3 t then 1 else 0
def Wl27_12 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c27_0 r0 t && !c27_1 r1 t && c27_2 r2 t && c27_4 r4 t then 1 else 0
def Wl27_13 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c27_0 r0 t && !c27_1 r1 t && c27_2 r2 t && c27_5 r5 t then 1 else 0
def Wl27_14 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c27_0 r0 t && !c27_1 r1 t && c27_2 r2 t && c27_6 r6 t then 1 else 0
def Wl27_15 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c27_0 r0 t && !c27_1 r1 t && !c27_2 r2 t && c27_3 r3 t && c27_4 r4 t then 1 else 0
def Wl27_16 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c27_0 r0 t && !c27_1 r1 t && !c27_2 r2 t && c27_3 r3 t && c27_5 r5 t then 1 else 0
def Wl27_17 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c27_0 r0 t && !c27_1 r1 t && !c27_2 r2 t && c27_3 r3 t && c27_6 r6 t then 1 else 0
def Wl27_18 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c27_0 r0 t && !c27_1 r1 t && !c27_2 r2 t && !c27_3 r3 t && c27_4 r4 t && c27_5 r5 t then 1 else 0
def Wl27_19 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c27_0 r0 t && !c27_1 r1 t && !c27_2 r2 t && !c27_3 r3 t && c27_4 r4 t && c27_6 r6 t then 1 else 0
def Wl27_20 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c27_0 r0 t && !c27_1 r1 t && !c27_2 r2 t && !c27_3 r3 t && !c27_4 r4 t && c27_5 r5 t && c27_6 r6 t then 1 else 0

/-- **No configuration blocks the whole window in case 27.** -/
theorem nocov27 {r0 r1 r2 r3 r4 r5 r6 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29) (h6 : r6 < 31)
    (hcov : ∀ t, t < n27 → (c27_0 r0 t || c27_1 r1 t || c27_2 r2 t || c27_3 r3 t || c27_4 r4 t || c27_5 r5 t || c27_6 r6 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n27, (1 : ℤ) + (Wl27_0 r0 r1 r2 r3 r4 r5 r6 t + Wl27_1 r0 r1 r2 r3 r4 r5 r6 t + Wl27_2 r0 r1 r2 r3 r4 r5 r6 t + Wl27_3 r0 r1 r2 r3 r4 r5 r6 t + Wl27_4 r0 r1 r2 r3 r4 r5 r6 t + Wl27_5 r0 r1 r2 r3 r4 r5 r6 t + Wl27_6 r0 r1 r2 r3 r4 r5 r6 t + Wl27_7 r0 r1 r2 r3 r4 r5 r6 t + Wl27_8 r0 r1 r2 r3 r4 r5 r6 t + Wl27_9 r0 r1 r2 r3 r4 r5 r6 t + Wl27_10 r0 r1 r2 r3 r4 r5 r6 t + Wl27_11 r0 r1 r2 r3 r4 r5 r6 t + Wl27_12 r0 r1 r2 r3 r4 r5 r6 t + Wl27_13 r0 r1 r2 r3 r4 r5 r6 t + Wl27_14 r0 r1 r2 r3 r4 r5 r6 t + Wl27_15 r0 r1 r2 r3 r4 r5 r6 t + Wl27_16 r0 r1 r2 r3 r4 r5 r6 t + Wl27_17 r0 r1 r2 r3 r4 r5 r6 t + Wl27_18 r0 r1 r2 r3 r4 r5 r6 t + Wl27_19 r0 r1 r2 r3 r4 r5 r6 t + Wl27_20 r0 r1 r2 r3 r4 r5 r6 t) ≤ Dg27 r0 r1 r2 r3 r4 r5 r6 t := by
    intro t ht
    simp only [Wl27_0, Wl27_1, Wl27_2, Wl27_3, Wl27_4, Wl27_5, Wl27_6, Wl27_7, Wl27_8, Wl27_9, Wl27_10, Wl27_11, Wl27_12, Wl27_13, Wl27_14, Wl27_15, Wl27_16, Wl27_17, Wl27_18, Wl27_19, Wl27_20, Dg27]
    exact CaseSplit.lowest7 _ _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n27, (1 : ℤ) ≤ Dg27 r0 r1 r2 r3 r4 r5 r6 t := by
    intro t ht
    simp only [Dg27]
    exact CaseSplit.degpos7 _ _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n27 : ℤ) + ((∑ t ∈ Finset.range n27, Wl27_0 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n27, Wl27_1 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n27, Wl27_2 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n27, Wl27_3 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n27, Wl27_4 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n27, Wl27_5 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n27, Wl27_6 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n27, Wl27_7 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n27, Wl27_8 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n27, Wl27_9 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n27, Wl27_10 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n27, Wl27_11 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n27, Wl27_12 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n27, Wl27_13 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n27, Wl27_14 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n27, Wl27_15 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n27, Wl27_16 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n27, Wl27_17 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n27, Wl27_18 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n27, Wl27_19 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n27, Wl27_20 r0 r1 r2 r3 r4 r5 r6 t)) ≤ ∑ t ∈ Finset.range n27, Dg27 r0 r1 r2 r3 r4 r5 r6 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N27_0 r0 r1 ≤ ∑ t ∈ Finset.range n27, Wl27_0 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N27_0, Wl27_0, le_refl]
  have hn1 : N27_1 r0 r2 ≤ ∑ t ∈ Finset.range n27, Wl27_1 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N27_1, Wl27_1, le_refl]
  have hn2 : N27_2 r0 r3 ≤ ∑ t ∈ Finset.range n27, Wl27_2 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N27_2, Wl27_2, le_refl]
  have hn3 : N27_3 r0 r4 ≤ ∑ t ∈ Finset.range n27, Wl27_3 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N27_3, Wl27_3, le_refl]
  have hn4 : N27_4 r0 r5 ≤ ∑ t ∈ Finset.range n27, Wl27_4 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N27_4, Wl27_4, le_refl]
  have hn5 : N27_5 r0 r6 ≤ ∑ t ∈ Finset.range n27, Wl27_5 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N27_5, Wl27_5, le_refl]
  have hn6 : N27_6 r1 r2 ≤ ∑ t ∈ Finset.range n27, Wl27_6 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n27, Wl27_6 r0 r1 r2 r3 r4 r5 r6 t
        = (if c27_1 r1 t && c27_2 r2 t then (1:ℤ) else 0)
          - (if c27_1 r1 t && c27_2 r2 t && c27_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl27_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n27, Wl27_6 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl27_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n27, Wl27_6 r0 r1 r2 r3 r4 r5 r6 t
        = P27_6 r1 r2 - C27_6 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P27_6, C27_6]
    have hm : C27_6 r1 r2 r0 ≤ M27_6 r1 r2 :=
      CaseSplit.le_mxr (C27_6 r1 r2) 10 r0 (by omega)
    simp only [N27_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N27_7 r1 r3 ≤ ∑ t ∈ Finset.range n27, Wl27_7 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n27, Wl27_7 r0 r1 r2 r3 r4 r5 r6 t
        = (if c27_1 r1 t && c27_3 r3 t then (1:ℤ) else 0)
          - (if c27_1 r1 t && c27_3 r3 t && c27_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl27_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n27, Wl27_7 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl27_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n27, Wl27_7 r0 r1 r2 r3 r4 r5 r6 t
        = P27_7 r1 r3 - C27_7 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P27_7, C27_7]
    have hm : C27_7 r1 r3 r0 ≤ M27_7 r1 r3 :=
      CaseSplit.le_mxr (C27_7 r1 r3) 10 r0 (by omega)
    simp only [N27_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N27_8 r1 r4 ≤ ∑ t ∈ Finset.range n27, Wl27_8 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n27, Wl27_8 r0 r1 r2 r3 r4 r5 r6 t
        = (if c27_1 r1 t && c27_4 r4 t then (1:ℤ) else 0)
          - (if c27_1 r1 t && c27_4 r4 t && c27_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl27_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n27, Wl27_8 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl27_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n27, Wl27_8 r0 r1 r2 r3 r4 r5 r6 t
        = P27_8 r1 r4 - C27_8 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P27_8, C27_8]
    have hm : C27_8 r1 r4 r0 ≤ M27_8 r1 r4 :=
      CaseSplit.le_mxr (C27_8 r1 r4) 10 r0 (by omega)
    simp only [N27_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N27_9 r1 r5 ≤ ∑ t ∈ Finset.range n27, Wl27_9 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n27, Wl27_9 r0 r1 r2 r3 r4 r5 r6 t
        = (if c27_1 r1 t && c27_5 r5 t then (1:ℤ) else 0)
          - (if c27_1 r1 t && c27_5 r5 t && c27_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl27_9]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n27, Wl27_9 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl27_9]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n27, Wl27_9 r0 r1 r2 r3 r4 r5 r6 t
        = P27_9 r1 r5 - C27_9 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P27_9, C27_9]
    have hm : C27_9 r1 r5 r0 ≤ M27_9 r1 r5 :=
      CaseSplit.le_mxr (C27_9 r1 r5) 10 r0 (by omega)
    simp only [N27_9]
    split
    · rw [hL]; omega
    · exact hnn
  have hn10 : N27_10 r1 r6 ≤ ∑ t ∈ Finset.range n27, Wl27_10 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n27, Wl27_10 r0 r1 r2 r3 r4 r5 r6 t
        = (if c27_1 r1 t && c27_6 r6 t then (1:ℤ) else 0)
          - (if c27_1 r1 t && c27_6 r6 t && c27_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl27_10]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n27, Wl27_10 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl27_10]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n27, Wl27_10 r0 r1 r2 r3 r4 r5 r6 t
        = P27_10 r1 r6 - C27_10 r1 r6 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P27_10, C27_10]
    have hm : C27_10 r1 r6 r0 ≤ M27_10 r1 r6 :=
      CaseSplit.le_mxr (C27_10 r1 r6) 10 r0 (by omega)
    simp only [N27_10]
    split
    · rw [hL]; omega
    · exact hnn
  have hn11 : N27_11 r2 r3 ≤ ∑ t ∈ Finset.range n27, Wl27_11 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N27_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl27_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N27_12 r2 r4 ≤ ∑ t ∈ Finset.range n27, Wl27_12 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N27_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl27_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N27_13 r2 r5 ≤ ∑ t ∈ Finset.range n27, Wl27_13 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N27_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl27_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N27_14 r2 r6 ≤ ∑ t ∈ Finset.range n27, Wl27_14 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N27_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl27_14]
    exact CaseSplit.ind_nonneg _
  have hn15 : N27_15 r3 r4 ≤ ∑ t ∈ Finset.range n27, Wl27_15 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N27_15]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl27_15]
    exact CaseSplit.ind_nonneg _
  have hn16 : N27_16 r3 r5 ≤ ∑ t ∈ Finset.range n27, Wl27_16 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N27_16]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl27_16]
    exact CaseSplit.ind_nonneg _
  have hn17 : N27_17 r3 r6 ≤ ∑ t ∈ Finset.range n27, Wl27_17 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N27_17]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl27_17]
    exact CaseSplit.ind_nonneg _
  have hn18 : N27_18 r4 r5 ≤ ∑ t ∈ Finset.range n27, Wl27_18 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N27_18]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl27_18]
    exact CaseSplit.ind_nonneg _
  have hn19 : N27_19 r4 r6 ≤ ∑ t ∈ Finset.range n27, Wl27_19 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N27_19]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl27_19]
    exact CaseSplit.ind_nonneg _
  have hn20 : N27_20 r5 r6 ≤ ∑ t ∈ Finset.range n27, Wl27_20 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N27_20]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl27_20]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n27, (w27 t + 5) * Dg27 r0 r1 r2 r3 r4 r5 r6 t = S27_0 r0 + S27_1 r1 + S27_2 r2 + S27_3 r3 + S27_4 r4 + S27_5 r5 + S27_6 r6 := by
    simp only [S27_0, S27_1, S27_2, S27_3, S27_4, S27_5, S27_6, Dg27, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n27, (w27 t + 5) * Dg27 r0 r1 r2 r3 r4 r5 r6 t
      = (∑ t ∈ Finset.range n27, w27 t * Dg27 r0 r1 r2 r3 r4 r5 r6 t)
        + 5 * (∑ t ∈ Finset.range n27, Dg27 r0 r1 r2 r3 r4 r5 r6 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n27, w27 t)
      ≤ ∑ t ∈ Finset.range n27, w27 t * Dg27 r0 r1 r2 r3 r4 r5 r6 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg27 r0 r1 r2 r3 r4 r5 r6 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w27 t := wnn27 t (Finset.mem_range.mp ht)
    calc w27 t = w27 t * 1 := (mul_one _).symm
      _ ≤ w27 t * Dg27 r0 r1 r2 r3 r4 r5 r6 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS27_0 r0 + aS27_1 r1 + aS27_2 r2 + aS27_3 r3 + aS27_4 r4 + aS27_5 r5 + aS27_6 r6) + (aP27_0 r0 r1 + aP27_1 r0 r2 + aP27_2 r0 r3 + aP27_3 r0 r4 + aP27_4 r0 r5 + aP27_5 r0 r6 + aP27_6 r1 r2 + aP27_7 r1 r3 + aP27_8 r1 r4 + aP27_9 r1 r5 + aP27_10 r1 r6 + aP27_11 r2 r3 + aP27_12 r2 r4 + aP27_13 r2 r5 + aP27_14 r2 r6 + aP27_15 r3 r4 + aP27_16 r3 r5 + aP27_17 r3 r6 + aP27_18 r4 r5 + aP27_19 r4 r6 + aP27_20 r5 r6) = (S27_0 r0 + S27_1 r1 + S27_2 r2 + S27_3 r3 + S27_4 r4 + S27_5 r5 + S27_6 r6) - 5 * (N27_0 r0 r1 + N27_1 r0 r2 + N27_2 r0 r3 + N27_3 r0 r4 + N27_4 r0 r5 + N27_5 r0 r6 + N27_6 r1 r2 + N27_7 r1 r3 + N27_8 r1 r4 + N27_9 r1 r5 + N27_10 r1 r6 + N27_11 r2 r3 + N27_12 r2 r4 + N27_13 r2 r5 + N27_14 r2 r6 + N27_15 r3 r4 + N27_16 r3 r5 + N27_17 r3 r6 + N27_18 r4 r5 + N27_19 r4 r6 + N27_20 r5 r6) := by
    simp only [aS27_0, aS27_1, aS27_2, aS27_3, aS27_4, aS27_5, aS27_6, aP27_0, aP27_1, aP27_2, aP27_3, aP27_4, aP27_5, aP27_6, aP27_7, aP27_8, aP27_9, aP27_10, aP27_11, aP27_12, aP27_13, aP27_14, aP27_15, aP27_16, aP27_17, aP27_18, aP27_19, aP27_20, L27_0, L27_1, L27_2, L27_3, L27_4, L27_5, L27_6]
    ring
  have bS0 : aS27_0 r0 ≤ MS27_0 := CaseSplit.le_mxr (aS27_0) 10 r0 (by omega)
  have bS1 : aS27_1 r1 ≤ MS27_1 := CaseSplit.le_mxr (aS27_1) 12 r1 (by omega)
  have bS2 : aS27_2 r2 ≤ MS27_2 := CaseSplit.le_mxr (aS27_2) 16 r2 (by omega)
  have bS3 : aS27_3 r3 ≤ MS27_3 := CaseSplit.le_mxr (aS27_3) 18 r3 (by omega)
  have bS4 : aS27_4 r4 ≤ MS27_4 := CaseSplit.le_mxr (aS27_4) 22 r4 (by omega)
  have bS5 : aS27_5 r5 ≤ MS27_5 := CaseSplit.le_mxr (aS27_5) 28 r5 (by omega)
  have bS6 : aS27_6 r6 ≤ MS27_6 := CaseSplit.le_mxr (aS27_6) 30 r6 (by omega)
  have bP0 : aP27_0 r0 r1 ≤ MP27_0 := CaseSplit.le_mxr2 (aP27_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP27_1 r0 r2 ≤ MP27_1 := CaseSplit.le_mxr2 (aP27_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP27_2 r0 r3 ≤ MP27_2 := CaseSplit.le_mxr2 (aP27_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP27_3 r0 r4 ≤ MP27_3 := CaseSplit.le_mxr2 (aP27_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP27_4 r0 r5 ≤ MP27_4 := CaseSplit.le_mxr2 (aP27_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP27_5 r0 r6 ≤ MP27_5 := CaseSplit.le_mxr2 (aP27_5) 10 30 r0 r6 (by omega) (by omega)
  have bP6 : aP27_6 r1 r2 ≤ MP27_6 := CaseSplit.le_mxr2 (aP27_6) 12 16 r1 r2 (by omega) (by omega)
  have bP7 : aP27_7 r1 r3 ≤ MP27_7 := CaseSplit.le_mxr2 (aP27_7) 12 18 r1 r3 (by omega) (by omega)
  have bP8 : aP27_8 r1 r4 ≤ MP27_8 := CaseSplit.le_mxr2 (aP27_8) 12 22 r1 r4 (by omega) (by omega)
  have bP9 : aP27_9 r1 r5 ≤ MP27_9 := CaseSplit.le_mxr2 (aP27_9) 12 28 r1 r5 (by omega) (by omega)
  have bP10 : aP27_10 r1 r6 ≤ MP27_10 := CaseSplit.le_mxr2 (aP27_10) 12 30 r1 r6 (by omega) (by omega)
  have bP11 : aP27_11 r2 r3 ≤ MP27_11 := CaseSplit.le_mxr2 (aP27_11) 16 18 r2 r3 (by omega) (by omega)
  have bP12 : aP27_12 r2 r4 ≤ MP27_12 := CaseSplit.le_mxr2 (aP27_12) 16 22 r2 r4 (by omega) (by omega)
  have bP13 : aP27_13 r2 r5 ≤ MP27_13 := CaseSplit.le_mxr2 (aP27_13) 16 28 r2 r5 (by omega) (by omega)
  have bP14 : aP27_14 r2 r6 ≤ MP27_14 := CaseSplit.le_mxr2 (aP27_14) 16 30 r2 r6 (by omega) (by omega)
  have bP15 : aP27_15 r3 r4 ≤ MP27_15 := CaseSplit.le_mxr2 (aP27_15) 18 22 r3 r4 (by omega) (by omega)
  have bP16 : aP27_16 r3 r5 ≤ MP27_16 := CaseSplit.le_mxr2 (aP27_16) 18 28 r3 r5 (by omega) (by omega)
  have bP17 : aP27_17 r3 r6 ≤ MP27_17 := CaseSplit.le_mxr2 (aP27_17) 18 30 r3 r6 (by omega) (by omega)
  have bP18 : aP27_18 r4 r5 ≤ MP27_18 := CaseSplit.le_mxr2 (aP27_18) 22 28 r4 r5 (by omega) (by omega)
  have bP19 : aP27_19 r4 r6 ≤ MP27_19 := CaseSplit.le_mxr2 (aP27_19) 22 30 r4 r6 (by omega) (by omega)
  have bP20 : aP27_20 r5 r6 ≤ MP27_20 := CaseSplit.le_mxr2 (aP27_20) 28 30 r5 r6 (by omega) (by omega)
  have hrhs : rhs27 = (∑ t ∈ Finset.range n27, w27 t) + 5 * (n27 : ℤ) := rfl
  have hc := cert27
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, hn15, hn16, hn17, hn18, hn19, hn20, bS0, bS1, bS2, bS3, bS4, bS5, bS6, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14, bP15, bP16, bP17, bP18, bP19, bP20]

end IncCert31
