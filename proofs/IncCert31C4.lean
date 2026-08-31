/-
INCREMENT-WIDTH CERTIFICATE, step 29->31, case 4 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_29_31.json, which re-derives every number
from the primes alone).

Machine 31, INCREMENT width 65 = F_2(29) + s_min(31) = 55 + 10,
held gears [5, 7] at phases [0, 4].  Free gears [11, 13, 17, 19, 23, 29, 31].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 7.
-/
import IncCert31B

namespace IncCert31

/-! ### case 4: held gears at phases [0, 4] -/

def p4 : List ℕ := [0, 3, 5, 7, 8, 10, 12, 13, 15, 17, 20, 22, 27, 28, 33, 35, 38, 40, 42, 43, 45, 47, 48, 50, 52, 55, 57, 62, 63]
def q4 (t : ℕ) : ℕ := p4.getD t 0
def n4 : ℕ := 29
def yl4 : List ℤ := [0, 0, 0, 0, 2, 6, 0, 2, 6, 0, 0, 0, 1, 0, 0, 1, 2, 6, 5, 6, 7, 4, 0, 5, 0, 1, 2, 0, 0]
def w4 (t : ℕ) : ℤ := yl4.getD t 0
def ul4 : List ℤ := [0, 1, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 3, (-1), (-3), 0, (-3), (-1), (-3), 0, (-3), (-3), (-1), 0, (-5), (-5), (-5), 0, 0, (-5), (-5), (-3), (-3), 0, (-2), (-5), (-5), (-3), (-2), 0, (-3), 0, 0, 5, 3, 0, 0, 0, 2, 2, 2, 0, (-9), 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, (-1), (-1), (-1), 0, (-3), 0, (-1), (-1), (-1), 0, 0, (-2), (-5), 0, (-5), (-5), (-5), (-5), 0, (-5), (-5), 0, (-5), (-5), (-22), (-2), 0, (-5), (-5), (-5), (-5), (-5), (-5), (-2), 0, 2, 0, 5, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-2), (-2), 0, 0, 0, 0, (-2), (-2), (-2), (-2), (-2), (-2), 0, (-2), 0, (-2), (-2), 0, (-2), 0, (-2), (-2), (-2), (-2), 0, (-2), 0, 0, (-2), 0, (-2), 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 15, 20, 21, 27, 27, 22, 22, 20, 27, 27, 27, 27, 17, 19, 25, 27, 27, (-27), (-27), (-27), (-27), (-27), (-27), (-27), (-27), (-27), (-27), (-27), (-27), (-27), 24, 22, 14, 24, 15, 11, 21, 24, 23, 23, 24, 24, 24, 24, 24, 21, 23, 24, 24, (-24), (-24), (-24), (-24), (-27), (-24), (-24), (-24), (-24), (-24), (-24), (-24), (-24), 31, 14, 30, 23, 31, 25, 15, 31, 31, 31, 31, 31, 31, 31, 22, 31, 31, 29, 31, 31, 31, 14, 31, (-31), (-31), (-31), (-31), (-31), (-31), (-31), (-31), (-31), (-31), (-31), (-31), (-31), 11, 11, 0, 0, 0, 11, 11, 11, 11, 11, 11, 11, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 0, 11, 11, 11, 4, 11, 6, (-11), (-11), (-11), (-15), (-11), (-11), (-11), (-11), (-11), (-11), (-11), (-11), (-11), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-14), (-4), (-4), (-4), 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-4), 0, (-7), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 8, 21, 5, 10, 21, 7, 11, 7, 18, 21, 13, 18, 8, 21, 16, 18, 21, 9, 21, 11, 18, 20, 13, 21, 16, 21, 11, 9, 16, 8, (-2), 4, 10, 10, 5, (-2), 10, 0, 7, 0, 0, 10, 0, 10, 4, 0, 10, (-2), 7, 10, 10, 10, 5, 0, 4, 7, 10, 0, 0]
def u4 (k : ℕ) : ℤ := ul4.getD k 0

def c4_0 (r t : ℕ) : Bool := gb11 r (q4 t)
def c4_1 (r t : ℕ) : Bool := gb13 r (q4 t)
def c4_2 (r t : ℕ) : Bool := gb17 r (q4 t)
def c4_3 (r t : ℕ) : Bool := gb19 r (q4 t)
def c4_4 (r t : ℕ) : Bool := gb23 r (q4 t)
def c4_5 (r t : ℕ) : Bool := gb29 r (q4 t)
def c4_6 (r t : ℕ) : Bool := gb31 r (q4 t)

def S4_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (w4 t + 5) * (if c4_0 r t then 1 else 0)
def S4_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (w4 t + 5) * (if c4_1 r t then 1 else 0)
def S4_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (w4 t + 5) * (if c4_2 r t then 1 else 0)
def S4_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (w4 t + 5) * (if c4_3 r t then 1 else 0)
def S4_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (w4 t + 5) * (if c4_4 r t then 1 else 0)
def S4_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (w4 t + 5) * (if c4_5 r t then 1 else 0)
def S4_6 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (w4 t + 5) * (if c4_6 r t then 1 else 0)

def L4_0 (r : ℕ) : ℤ := u4 (13 + r) + u4 (41 + r) + u4 (71 + r) + u4 (105 + r) + u4 (145 + r) + u4 (187 + r)
def L4_1 (r : ℕ) : ℤ := u4 (0 + r) + u4 (215 + r) + u4 (247 + r) + u4 (283 + r) + u4 (325 + r) + u4 (369 + r)
def L4_2 (r : ℕ) : ℤ := u4 (24 + r) + u4 (198 + r) + u4 (401 + r) + u4 (441 + r) + u4 (487 + r) + u4 (535 + r)
def L4_3 (r : ℕ) : ℤ := u4 (52 + r) + u4 (228 + r) + u4 (382 + r) + u4 (575 + r) + u4 (623 + r) + u4 (673 + r)
def L4_4 (r : ℕ) : ℤ := u4 (82 + r) + u4 (260 + r) + u4 (418 + r) + u4 (552 + r) + u4 (721 + r) + u4 (775 + r)
def L4_5 (r : ℕ) : ℤ := u4 (116 + r) + u4 (296 + r) + u4 (458 + r) + u4 (594 + r) + u4 (692 + r) + u4 (829 + r)
def L4_6 (r : ℕ) : ℤ := u4 (156 + r) + u4 (338 + r) + u4 (504 + r) + u4 (642 + r) + u4 (744 + r) + u4 (798 + r)

def aS4_0 (r : ℕ) : ℤ := S4_0 r - L4_0 r
def MS4_0 : ℤ := CaseSplit.mxr (aS4_0) 10
def aS4_1 (r : ℕ) : ℤ := S4_1 r - L4_1 r
def MS4_1 : ℤ := CaseSplit.mxr (aS4_1) 12
def aS4_2 (r : ℕ) : ℤ := S4_2 r - L4_2 r
def MS4_2 : ℤ := CaseSplit.mxr (aS4_2) 16
def aS4_3 (r : ℕ) : ℤ := S4_3 r - L4_3 r
def MS4_3 : ℤ := CaseSplit.mxr (aS4_3) 18
def aS4_4 (r : ℕ) : ℤ := S4_4 r - L4_4 r
def MS4_4 : ℤ := CaseSplit.mxr (aS4_4) 22
def aS4_5 (r : ℕ) : ℤ := S4_5 r - L4_5 r
def MS4_5 : ℤ := CaseSplit.mxr (aS4_5) 28
def aS4_6 (r : ℕ) : ℤ := S4_6 r - L4_6 r
def MS4_6 : ℤ := CaseSplit.mxr (aS4_6) 30

def N4_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_0 ra t && c4_1 rb t then 1 else 0)
def aP4_0 (ra rb : ℕ) : ℤ := -(5) * N4_0 ra rb + u4 (0 + rb) + u4 (13 + ra)
def MP4_0 : ℤ := CaseSplit.mxr2 (aP4_0) 10 12
def N4_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_0 ra t && c4_2 rb t then 1 else 0)
def aP4_1 (ra rb : ℕ) : ℤ := -(5) * N4_1 ra rb + u4 (24 + rb) + u4 (41 + ra)
def MP4_1 : ℤ := CaseSplit.mxr2 (aP4_1) 10 16
def N4_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_0 ra t && c4_3 rb t then 1 else 0)
def aP4_2 (ra rb : ℕ) : ℤ := -(5) * N4_2 ra rb + u4 (52 + rb) + u4 (71 + ra)
def MP4_2 : ℤ := CaseSplit.mxr2 (aP4_2) 10 18
def N4_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_0 ra t && c4_4 rb t then 1 else 0)
def aP4_3 (ra rb : ℕ) : ℤ := -(5) * N4_3 ra rb + u4 (82 + rb) + u4 (105 + ra)
def MP4_3 : ℤ := CaseSplit.mxr2 (aP4_3) 10 22
def N4_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_0 ra t && c4_5 rb t then 1 else 0)
def aP4_4 (ra rb : ℕ) : ℤ := -(5) * N4_4 ra rb + u4 (116 + rb) + u4 (145 + ra)
def MP4_4 : ℤ := CaseSplit.mxr2 (aP4_4) 10 28
def N4_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_0 ra t && c4_6 rb t then 1 else 0)
def aP4_5 (ra rb : ℕ) : ℤ := -(5) * N4_5 ra rb + u4 (156 + rb) + u4 (187 + ra)
def MP4_5 : ℤ := CaseSplit.mxr2 (aP4_5) 10 30
def P4_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_1 ra t && c4_2 rb t then 1 else 0)
def C4_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_1 ra t && c4_2 rb t && c4_0 s t then 1 else 0)
def M4_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C4_6 ra rb) 10
def E4_6 : List ℕ := [21, 27, 36, 47, 72, 83, 88, 94, 111, 117, 136, 147, 156, 162, 178, 184, 190, 201, 214, 220]
def N4_6 (ra rb : ℕ) : ℤ := if E4_6.contains (ra * 17 + rb) = true then P4_6 ra rb - M4_6 ra rb else 0
def aP4_6 (ra rb : ℕ) : ℤ := -(5) * N4_6 ra rb + u4 (198 + rb) + u4 (215 + ra)
def MP4_6 : ℤ := CaseSplit.mxr2 (aP4_6) 12 16
def P4_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_1 ra t && c4_3 rb t then 1 else 0)
def C4_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_1 ra t && c4_3 rb t && c4_0 s t then 1 else 0)
def M4_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C4_7 ra rb) 10
def E4_7 : List ℕ := [7, 10, 33, 38, 41, 44, 78, 86, 91, 114, 120, 154, 167, 170, 178, 204, 212, 246]
def N4_7 (ra rb : ℕ) : ℤ := if E4_7.contains (ra * 19 + rb) = true then P4_7 ra rb - M4_7 ra rb else 0
def aP4_7 (ra rb : ℕ) : ℤ := -(5) * N4_7 ra rb + u4 (228 + rb) + u4 (247 + ra)
def MP4_7 : ℤ := CaseSplit.mxr2 (aP4_7) 12 18
def P4_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_1 ra t && c4_4 rb t then 1 else 0)
def C4_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_1 ra t && c4_4 rb t && c4_0 s t then 1 else 0)
def M4_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C4_8 ra rb) 10
def E4_8 : List ℕ := []
def N4_8 (ra rb : ℕ) : ℤ := if E4_8.contains (ra * 23 + rb) = true then P4_8 ra rb - M4_8 ra rb else 0
def aP4_8 (ra rb : ℕ) : ℤ := -(5) * N4_8 ra rb + u4 (260 + rb) + u4 (283 + ra)
def MP4_8 : ℤ := CaseSplit.mxr2 (aP4_8) 12 22
def P4_9 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_1 ra t && c4_5 rb t then 1 else 0)
def C4_9 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_1 ra t && c4_5 rb t && c4_0 s t then 1 else 0)
def M4_9 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C4_9 ra rb) 10
def E4_9 : List ℕ := [69, 103, 219, 253, 330, 369]
def N4_9 (ra rb : ℕ) : ℤ := if E4_9.contains (ra * 29 + rb) = true then P4_9 ra rb - M4_9 ra rb else 0
def aP4_9 (ra rb : ℕ) : ℤ := -(5) * N4_9 ra rb + u4 (296 + rb) + u4 (325 + ra)
def MP4_9 : ℤ := CaseSplit.mxr2 (aP4_9) 12 28
def P4_10 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_1 ra t && c4_6 rb t then 1 else 0)
def C4_10 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_1 ra t && c4_6 rb t && c4_0 s t then 1 else 0)
def M4_10 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C4_10 ra rb) 10
def E4_10 : List ℕ := [57, 67, 181, 186, 250, 310, 346, 374]
def N4_10 (ra rb : ℕ) : ℤ := if E4_10.contains (ra * 31 + rb) = true then P4_10 ra rb - M4_10 ra rb else 0
def aP4_10 (ra rb : ℕ) : ℤ := -(5) * N4_10 ra rb + u4 (338 + rb) + u4 (369 + ra)
def MP4_10 : ℤ := CaseSplit.mxr2 (aP4_10) 12 30
def N4_11 (_ra _rb : ℕ) : ℤ := 0
def aP4_11 (ra rb : ℕ) : ℤ := -(5) * N4_11 ra rb + u4 (382 + rb) + u4 (401 + ra)
def MP4_11 : ℤ := CaseSplit.mxr2 (aP4_11) 16 18
def N4_12 (_ra _rb : ℕ) : ℤ := 0
def aP4_12 (ra rb : ℕ) : ℤ := -(5) * N4_12 ra rb + u4 (418 + rb) + u4 (441 + ra)
def MP4_12 : ℤ := CaseSplit.mxr2 (aP4_12) 16 22
def N4_13 (_ra _rb : ℕ) : ℤ := 0
def aP4_13 (ra rb : ℕ) : ℤ := -(5) * N4_13 ra rb + u4 (458 + rb) + u4 (487 + ra)
def MP4_13 : ℤ := CaseSplit.mxr2 (aP4_13) 16 28
def N4_14 (_ra _rb : ℕ) : ℤ := 0
def aP4_14 (ra rb : ℕ) : ℤ := -(5) * N4_14 ra rb + u4 (504 + rb) + u4 (535 + ra)
def MP4_14 : ℤ := CaseSplit.mxr2 (aP4_14) 16 30
def N4_15 (_ra _rb : ℕ) : ℤ := 0
def aP4_15 (ra rb : ℕ) : ℤ := -(5) * N4_15 ra rb + u4 (552 + rb) + u4 (575 + ra)
def MP4_15 : ℤ := CaseSplit.mxr2 (aP4_15) 18 22
def N4_16 (_ra _rb : ℕ) : ℤ := 0
def aP4_16 (ra rb : ℕ) : ℤ := -(5) * N4_16 ra rb + u4 (594 + rb) + u4 (623 + ra)
def MP4_16 : ℤ := CaseSplit.mxr2 (aP4_16) 18 28
def N4_17 (_ra _rb : ℕ) : ℤ := 0
def aP4_17 (ra rb : ℕ) : ℤ := -(5) * N4_17 ra rb + u4 (642 + rb) + u4 (673 + ra)
def MP4_17 : ℤ := CaseSplit.mxr2 (aP4_17) 18 30
def N4_18 (_ra _rb : ℕ) : ℤ := 0
def aP4_18 (ra rb : ℕ) : ℤ := -(5) * N4_18 ra rb + u4 (692 + rb) + u4 (721 + ra)
def MP4_18 : ℤ := CaseSplit.mxr2 (aP4_18) 22 28
def N4_19 (_ra _rb : ℕ) : ℤ := 0
def aP4_19 (ra rb : ℕ) : ℤ := -(5) * N4_19 ra rb + u4 (744 + rb) + u4 (775 + ra)
def MP4_19 : ℤ := CaseSplit.mxr2 (aP4_19) 22 30
def N4_20 (_ra _rb : ℕ) : ℤ := 0
def aP4_20 (ra rb : ℕ) : ℤ := -(5) * N4_20 ra rb + u4 (798 + rb) + u4 (829 + ra)
def MP4_20 : ℤ := CaseSplit.mxr2 (aP4_20) 28 30

def rhs4 : ℤ := (∑ t ∈ Finset.range n4, w4 t) + 5 * (n4 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn4 : ∀ t, t < n4 → (0 : ℤ) ≤ w4 t := by decide
theorem plt4 : ∀ t, t < n4 → q4 t < 65 := by decide
theorem pfree4_5 : ∀ t, t < n4 → gb5 0 (q4 t) = false := by decide
theorem pfree4_7 : ∀ t, t < n4 → gb7 4 (q4 t) = false := by decide
theorem MSv4_0 : MS4_0 = 37 := by decide +kernel
theorem MSv4_1 : MS4_1 = 126 := by decide +kernel
theorem MSv4_2 : MS4_2 = 1 := by decide +kernel
theorem MSv4_3 : MS4_3 = 2 := by decide +kernel
theorem MSv4_4 : MS4_4 = 1 := by decide +kernel
theorem MSv4_5 : MS4_5 = 1 := by decide +kernel
theorem MSv4_6 : MS4_6 = 1 := by decide +kernel
theorem MPv4_0 : MP4_0 = 0 := by decide +kernel
theorem MPv4_1 : MP4_1 = 0 := by decide +kernel
theorem MPv4_2 : MP4_2 = 0 := by decide +kernel
theorem MPv4_3 : MP4_3 = 0 := by decide +kernel
theorem MPv4_4 : MP4_4 = 0 := by decide +kernel
theorem MPv4_5 : MP4_5 = 0 := by decide +kernel
theorem MPv4_6 : MP4_6 = 0 := by decide +kernel
theorem MPv4_7 : MP4_7 = 0 := by decide +kernel
theorem MPv4_8 : MP4_8 = 0 := by decide +kernel
theorem MPv4_9 : MP4_9 = 0 := by decide +kernel
theorem MPv4_10 : MP4_10 = 0 := by decide +kernel
theorem MPv4_11 : MP4_11 = 0 := by decide +kernel
theorem MPv4_12 : MP4_12 = 0 := by decide +kernel
theorem MPv4_13 : MP4_13 = 0 := by decide +kernel
theorem MPv4_14 : MP4_14 = 0 := by decide +kernel
theorem MPv4_15 : MP4_15 = 0 := by decide +kernel
theorem MPv4_16 : MP4_16 = 0 := by decide +kernel
theorem MPv4_17 : MP4_17 = 0 := by decide +kernel
theorem MPv4_18 : MP4_18 = 0 := by decide +kernel
theorem MPv4_19 : MP4_19 = 0 := by decide +kernel
theorem MPv4_20 : MP4_20 = 31 := by decide +kernel
theorem rhsv4 : rhs4 = 201 := by decide +kernel

/-- **The case-4 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/7.
    (Scaled by the common denominator 7: 200 < 201.) -/
theorem cert4 : MS4_0 + MS4_1 + MS4_2 + MS4_3 + MS4_4 + MS4_5 + MS4_6 + MP4_0 + MP4_1 + MP4_2 + MP4_3 + MP4_4 + MP4_5 + MP4_6 + MP4_7 + MP4_8 + MP4_9 + MP4_10 + MP4_11 + MP4_12 + MP4_13 + MP4_14 + MP4_15 + MP4_16 + MP4_17 + MP4_18 + MP4_19 + MP4_20 < rhs4 := by
  rw [MSv4_0, MSv4_1, MSv4_2, MSv4_3, MSv4_4, MSv4_5, MSv4_6, MPv4_0, MPv4_1, MPv4_2, MPv4_3, MPv4_4, MPv4_5, MPv4_6, MPv4_7, MPv4_8, MPv4_9, MPv4_10, MPv4_11, MPv4_12, MPv4_13, MPv4_14, MPv4_15, MPv4_16, MPv4_17, MPv4_18, MPv4_19, MPv4_20, rhsv4]
  decide

def Dg4 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := (if c4_0 r0 t then 1 else 0) + (if c4_1 r1 t then 1 else 0) + (if c4_2 r2 t then 1 else 0) + (if c4_3 r3 t then 1 else 0) + (if c4_4 r4 t then 1 else 0) + (if c4_5 r5 t then 1 else 0) + (if c4_6 r6 t then 1 else 0)
def Wl4_0 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c4_0 r0 t && c4_1 r1 t then 1 else 0
def Wl4_1 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c4_0 r0 t && c4_2 r2 t then 1 else 0
def Wl4_2 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c4_0 r0 t && c4_3 r3 t then 1 else 0
def Wl4_3 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c4_0 r0 t && c4_4 r4 t then 1 else 0
def Wl4_4 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c4_0 r0 t && c4_5 r5 t then 1 else 0
def Wl4_5 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c4_0 r0 t && c4_6 r6 t then 1 else 0
def Wl4_6 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c4_0 r0 t && c4_1 r1 t && c4_2 r2 t then 1 else 0
def Wl4_7 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c4_0 r0 t && c4_1 r1 t && c4_3 r3 t then 1 else 0
def Wl4_8 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c4_0 r0 t && c4_1 r1 t && c4_4 r4 t then 1 else 0
def Wl4_9 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c4_0 r0 t && c4_1 r1 t && c4_5 r5 t then 1 else 0
def Wl4_10 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c4_0 r0 t && c4_1 r1 t && c4_6 r6 t then 1 else 0
def Wl4_11 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c4_0 r0 t && !c4_1 r1 t && c4_2 r2 t && c4_3 r3 t then 1 else 0
def Wl4_12 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c4_0 r0 t && !c4_1 r1 t && c4_2 r2 t && c4_4 r4 t then 1 else 0
def Wl4_13 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c4_0 r0 t && !c4_1 r1 t && c4_2 r2 t && c4_5 r5 t then 1 else 0
def Wl4_14 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c4_0 r0 t && !c4_1 r1 t && c4_2 r2 t && c4_6 r6 t then 1 else 0
def Wl4_15 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c4_0 r0 t && !c4_1 r1 t && !c4_2 r2 t && c4_3 r3 t && c4_4 r4 t then 1 else 0
def Wl4_16 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c4_0 r0 t && !c4_1 r1 t && !c4_2 r2 t && c4_3 r3 t && c4_5 r5 t then 1 else 0
def Wl4_17 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c4_0 r0 t && !c4_1 r1 t && !c4_2 r2 t && c4_3 r3 t && c4_6 r6 t then 1 else 0
def Wl4_18 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c4_0 r0 t && !c4_1 r1 t && !c4_2 r2 t && !c4_3 r3 t && c4_4 r4 t && c4_5 r5 t then 1 else 0
def Wl4_19 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c4_0 r0 t && !c4_1 r1 t && !c4_2 r2 t && !c4_3 r3 t && c4_4 r4 t && c4_6 r6 t then 1 else 0
def Wl4_20 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c4_0 r0 t && !c4_1 r1 t && !c4_2 r2 t && !c4_3 r3 t && !c4_4 r4 t && c4_5 r5 t && c4_6 r6 t then 1 else 0

/-- **No configuration blocks the whole window in case 4.** -/
theorem nocov4 {r0 r1 r2 r3 r4 r5 r6 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29) (h6 : r6 < 31)
    (hcov : ∀ t, t < n4 → (c4_0 r0 t || c4_1 r1 t || c4_2 r2 t || c4_3 r3 t || c4_4 r4 t || c4_5 r5 t || c4_6 r6 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n4, (1 : ℤ) + (Wl4_0 r0 r1 r2 r3 r4 r5 r6 t + Wl4_1 r0 r1 r2 r3 r4 r5 r6 t + Wl4_2 r0 r1 r2 r3 r4 r5 r6 t + Wl4_3 r0 r1 r2 r3 r4 r5 r6 t + Wl4_4 r0 r1 r2 r3 r4 r5 r6 t + Wl4_5 r0 r1 r2 r3 r4 r5 r6 t + Wl4_6 r0 r1 r2 r3 r4 r5 r6 t + Wl4_7 r0 r1 r2 r3 r4 r5 r6 t + Wl4_8 r0 r1 r2 r3 r4 r5 r6 t + Wl4_9 r0 r1 r2 r3 r4 r5 r6 t + Wl4_10 r0 r1 r2 r3 r4 r5 r6 t + Wl4_11 r0 r1 r2 r3 r4 r5 r6 t + Wl4_12 r0 r1 r2 r3 r4 r5 r6 t + Wl4_13 r0 r1 r2 r3 r4 r5 r6 t + Wl4_14 r0 r1 r2 r3 r4 r5 r6 t + Wl4_15 r0 r1 r2 r3 r4 r5 r6 t + Wl4_16 r0 r1 r2 r3 r4 r5 r6 t + Wl4_17 r0 r1 r2 r3 r4 r5 r6 t + Wl4_18 r0 r1 r2 r3 r4 r5 r6 t + Wl4_19 r0 r1 r2 r3 r4 r5 r6 t + Wl4_20 r0 r1 r2 r3 r4 r5 r6 t) ≤ Dg4 r0 r1 r2 r3 r4 r5 r6 t := by
    intro t ht
    simp only [Wl4_0, Wl4_1, Wl4_2, Wl4_3, Wl4_4, Wl4_5, Wl4_6, Wl4_7, Wl4_8, Wl4_9, Wl4_10, Wl4_11, Wl4_12, Wl4_13, Wl4_14, Wl4_15, Wl4_16, Wl4_17, Wl4_18, Wl4_19, Wl4_20, Dg4]
    exact CaseSplit.lowest7 _ _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n4, (1 : ℤ) ≤ Dg4 r0 r1 r2 r3 r4 r5 r6 t := by
    intro t ht
    simp only [Dg4]
    exact CaseSplit.degpos7 _ _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n4 : ℤ) + ((∑ t ∈ Finset.range n4, Wl4_0 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n4, Wl4_1 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n4, Wl4_2 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n4, Wl4_3 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n4, Wl4_4 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n4, Wl4_5 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n4, Wl4_6 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n4, Wl4_7 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n4, Wl4_8 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n4, Wl4_9 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n4, Wl4_10 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n4, Wl4_11 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n4, Wl4_12 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n4, Wl4_13 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n4, Wl4_14 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n4, Wl4_15 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n4, Wl4_16 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n4, Wl4_17 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n4, Wl4_18 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n4, Wl4_19 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n4, Wl4_20 r0 r1 r2 r3 r4 r5 r6 t)) ≤ ∑ t ∈ Finset.range n4, Dg4 r0 r1 r2 r3 r4 r5 r6 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N4_0 r0 r1 ≤ ∑ t ∈ Finset.range n4, Wl4_0 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N4_0, Wl4_0, le_refl]
  have hn1 : N4_1 r0 r2 ≤ ∑ t ∈ Finset.range n4, Wl4_1 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N4_1, Wl4_1, le_refl]
  have hn2 : N4_2 r0 r3 ≤ ∑ t ∈ Finset.range n4, Wl4_2 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N4_2, Wl4_2, le_refl]
  have hn3 : N4_3 r0 r4 ≤ ∑ t ∈ Finset.range n4, Wl4_3 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N4_3, Wl4_3, le_refl]
  have hn4 : N4_4 r0 r5 ≤ ∑ t ∈ Finset.range n4, Wl4_4 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N4_4, Wl4_4, le_refl]
  have hn5 : N4_5 r0 r6 ≤ ∑ t ∈ Finset.range n4, Wl4_5 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N4_5, Wl4_5, le_refl]
  have hn6 : N4_6 r1 r2 ≤ ∑ t ∈ Finset.range n4, Wl4_6 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n4, Wl4_6 r0 r1 r2 r3 r4 r5 r6 t
        = (if c4_1 r1 t && c4_2 r2 t then (1:ℤ) else 0)
          - (if c4_1 r1 t && c4_2 r2 t && c4_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl4_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n4, Wl4_6 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl4_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n4, Wl4_6 r0 r1 r2 r3 r4 r5 r6 t
        = P4_6 r1 r2 - C4_6 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P4_6, C4_6]
    have hm : C4_6 r1 r2 r0 ≤ M4_6 r1 r2 :=
      CaseSplit.le_mxr (C4_6 r1 r2) 10 r0 (by omega)
    simp only [N4_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N4_7 r1 r3 ≤ ∑ t ∈ Finset.range n4, Wl4_7 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n4, Wl4_7 r0 r1 r2 r3 r4 r5 r6 t
        = (if c4_1 r1 t && c4_3 r3 t then (1:ℤ) else 0)
          - (if c4_1 r1 t && c4_3 r3 t && c4_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl4_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n4, Wl4_7 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl4_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n4, Wl4_7 r0 r1 r2 r3 r4 r5 r6 t
        = P4_7 r1 r3 - C4_7 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P4_7, C4_7]
    have hm : C4_7 r1 r3 r0 ≤ M4_7 r1 r3 :=
      CaseSplit.le_mxr (C4_7 r1 r3) 10 r0 (by omega)
    simp only [N4_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N4_8 r1 r4 ≤ ∑ t ∈ Finset.range n4, Wl4_8 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n4, Wl4_8 r0 r1 r2 r3 r4 r5 r6 t
        = (if c4_1 r1 t && c4_4 r4 t then (1:ℤ) else 0)
          - (if c4_1 r1 t && c4_4 r4 t && c4_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl4_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n4, Wl4_8 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl4_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n4, Wl4_8 r0 r1 r2 r3 r4 r5 r6 t
        = P4_8 r1 r4 - C4_8 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P4_8, C4_8]
    have hm : C4_8 r1 r4 r0 ≤ M4_8 r1 r4 :=
      CaseSplit.le_mxr (C4_8 r1 r4) 10 r0 (by omega)
    simp only [N4_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N4_9 r1 r5 ≤ ∑ t ∈ Finset.range n4, Wl4_9 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n4, Wl4_9 r0 r1 r2 r3 r4 r5 r6 t
        = (if c4_1 r1 t && c4_5 r5 t then (1:ℤ) else 0)
          - (if c4_1 r1 t && c4_5 r5 t && c4_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl4_9]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n4, Wl4_9 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl4_9]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n4, Wl4_9 r0 r1 r2 r3 r4 r5 r6 t
        = P4_9 r1 r5 - C4_9 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P4_9, C4_9]
    have hm : C4_9 r1 r5 r0 ≤ M4_9 r1 r5 :=
      CaseSplit.le_mxr (C4_9 r1 r5) 10 r0 (by omega)
    simp only [N4_9]
    split
    · rw [hL]; omega
    · exact hnn
  have hn10 : N4_10 r1 r6 ≤ ∑ t ∈ Finset.range n4, Wl4_10 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n4, Wl4_10 r0 r1 r2 r3 r4 r5 r6 t
        = (if c4_1 r1 t && c4_6 r6 t then (1:ℤ) else 0)
          - (if c4_1 r1 t && c4_6 r6 t && c4_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl4_10]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n4, Wl4_10 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl4_10]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n4, Wl4_10 r0 r1 r2 r3 r4 r5 r6 t
        = P4_10 r1 r6 - C4_10 r1 r6 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P4_10, C4_10]
    have hm : C4_10 r1 r6 r0 ≤ M4_10 r1 r6 :=
      CaseSplit.le_mxr (C4_10 r1 r6) 10 r0 (by omega)
    simp only [N4_10]
    split
    · rw [hL]; omega
    · exact hnn
  have hn11 : N4_11 r2 r3 ≤ ∑ t ∈ Finset.range n4, Wl4_11 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N4_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl4_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N4_12 r2 r4 ≤ ∑ t ∈ Finset.range n4, Wl4_12 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N4_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl4_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N4_13 r2 r5 ≤ ∑ t ∈ Finset.range n4, Wl4_13 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N4_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl4_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N4_14 r2 r6 ≤ ∑ t ∈ Finset.range n4, Wl4_14 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N4_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl4_14]
    exact CaseSplit.ind_nonneg _
  have hn15 : N4_15 r3 r4 ≤ ∑ t ∈ Finset.range n4, Wl4_15 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N4_15]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl4_15]
    exact CaseSplit.ind_nonneg _
  have hn16 : N4_16 r3 r5 ≤ ∑ t ∈ Finset.range n4, Wl4_16 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N4_16]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl4_16]
    exact CaseSplit.ind_nonneg _
  have hn17 : N4_17 r3 r6 ≤ ∑ t ∈ Finset.range n4, Wl4_17 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N4_17]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl4_17]
    exact CaseSplit.ind_nonneg _
  have hn18 : N4_18 r4 r5 ≤ ∑ t ∈ Finset.range n4, Wl4_18 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N4_18]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl4_18]
    exact CaseSplit.ind_nonneg _
  have hn19 : N4_19 r4 r6 ≤ ∑ t ∈ Finset.range n4, Wl4_19 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N4_19]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl4_19]
    exact CaseSplit.ind_nonneg _
  have hn20 : N4_20 r5 r6 ≤ ∑ t ∈ Finset.range n4, Wl4_20 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N4_20]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl4_20]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n4, (w4 t + 5) * Dg4 r0 r1 r2 r3 r4 r5 r6 t = S4_0 r0 + S4_1 r1 + S4_2 r2 + S4_3 r3 + S4_4 r4 + S4_5 r5 + S4_6 r6 := by
    simp only [S4_0, S4_1, S4_2, S4_3, S4_4, S4_5, S4_6, Dg4, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n4, (w4 t + 5) * Dg4 r0 r1 r2 r3 r4 r5 r6 t
      = (∑ t ∈ Finset.range n4, w4 t * Dg4 r0 r1 r2 r3 r4 r5 r6 t)
        + 5 * (∑ t ∈ Finset.range n4, Dg4 r0 r1 r2 r3 r4 r5 r6 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n4, w4 t)
      ≤ ∑ t ∈ Finset.range n4, w4 t * Dg4 r0 r1 r2 r3 r4 r5 r6 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg4 r0 r1 r2 r3 r4 r5 r6 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w4 t := wnn4 t (Finset.mem_range.mp ht)
    calc w4 t = w4 t * 1 := (mul_one _).symm
      _ ≤ w4 t * Dg4 r0 r1 r2 r3 r4 r5 r6 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS4_0 r0 + aS4_1 r1 + aS4_2 r2 + aS4_3 r3 + aS4_4 r4 + aS4_5 r5 + aS4_6 r6) + (aP4_0 r0 r1 + aP4_1 r0 r2 + aP4_2 r0 r3 + aP4_3 r0 r4 + aP4_4 r0 r5 + aP4_5 r0 r6 + aP4_6 r1 r2 + aP4_7 r1 r3 + aP4_8 r1 r4 + aP4_9 r1 r5 + aP4_10 r1 r6 + aP4_11 r2 r3 + aP4_12 r2 r4 + aP4_13 r2 r5 + aP4_14 r2 r6 + aP4_15 r3 r4 + aP4_16 r3 r5 + aP4_17 r3 r6 + aP4_18 r4 r5 + aP4_19 r4 r6 + aP4_20 r5 r6) = (S4_0 r0 + S4_1 r1 + S4_2 r2 + S4_3 r3 + S4_4 r4 + S4_5 r5 + S4_6 r6) - 5 * (N4_0 r0 r1 + N4_1 r0 r2 + N4_2 r0 r3 + N4_3 r0 r4 + N4_4 r0 r5 + N4_5 r0 r6 + N4_6 r1 r2 + N4_7 r1 r3 + N4_8 r1 r4 + N4_9 r1 r5 + N4_10 r1 r6 + N4_11 r2 r3 + N4_12 r2 r4 + N4_13 r2 r5 + N4_14 r2 r6 + N4_15 r3 r4 + N4_16 r3 r5 + N4_17 r3 r6 + N4_18 r4 r5 + N4_19 r4 r6 + N4_20 r5 r6) := by
    simp only [aS4_0, aS4_1, aS4_2, aS4_3, aS4_4, aS4_5, aS4_6, aP4_0, aP4_1, aP4_2, aP4_3, aP4_4, aP4_5, aP4_6, aP4_7, aP4_8, aP4_9, aP4_10, aP4_11, aP4_12, aP4_13, aP4_14, aP4_15, aP4_16, aP4_17, aP4_18, aP4_19, aP4_20, L4_0, L4_1, L4_2, L4_3, L4_4, L4_5, L4_6]
    ring
  have bS0 : aS4_0 r0 ≤ MS4_0 := CaseSplit.le_mxr (aS4_0) 10 r0 (by omega)
  have bS1 : aS4_1 r1 ≤ MS4_1 := CaseSplit.le_mxr (aS4_1) 12 r1 (by omega)
  have bS2 : aS4_2 r2 ≤ MS4_2 := CaseSplit.le_mxr (aS4_2) 16 r2 (by omega)
  have bS3 : aS4_3 r3 ≤ MS4_3 := CaseSplit.le_mxr (aS4_3) 18 r3 (by omega)
  have bS4 : aS4_4 r4 ≤ MS4_4 := CaseSplit.le_mxr (aS4_4) 22 r4 (by omega)
  have bS5 : aS4_5 r5 ≤ MS4_5 := CaseSplit.le_mxr (aS4_5) 28 r5 (by omega)
  have bS6 : aS4_6 r6 ≤ MS4_6 := CaseSplit.le_mxr (aS4_6) 30 r6 (by omega)
  have bP0 : aP4_0 r0 r1 ≤ MP4_0 := CaseSplit.le_mxr2 (aP4_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP4_1 r0 r2 ≤ MP4_1 := CaseSplit.le_mxr2 (aP4_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP4_2 r0 r3 ≤ MP4_2 := CaseSplit.le_mxr2 (aP4_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP4_3 r0 r4 ≤ MP4_3 := CaseSplit.le_mxr2 (aP4_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP4_4 r0 r5 ≤ MP4_4 := CaseSplit.le_mxr2 (aP4_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP4_5 r0 r6 ≤ MP4_5 := CaseSplit.le_mxr2 (aP4_5) 10 30 r0 r6 (by omega) (by omega)
  have bP6 : aP4_6 r1 r2 ≤ MP4_6 := CaseSplit.le_mxr2 (aP4_6) 12 16 r1 r2 (by omega) (by omega)
  have bP7 : aP4_7 r1 r3 ≤ MP4_7 := CaseSplit.le_mxr2 (aP4_7) 12 18 r1 r3 (by omega) (by omega)
  have bP8 : aP4_8 r1 r4 ≤ MP4_8 := CaseSplit.le_mxr2 (aP4_8) 12 22 r1 r4 (by omega) (by omega)
  have bP9 : aP4_9 r1 r5 ≤ MP4_9 := CaseSplit.le_mxr2 (aP4_9) 12 28 r1 r5 (by omega) (by omega)
  have bP10 : aP4_10 r1 r6 ≤ MP4_10 := CaseSplit.le_mxr2 (aP4_10) 12 30 r1 r6 (by omega) (by omega)
  have bP11 : aP4_11 r2 r3 ≤ MP4_11 := CaseSplit.le_mxr2 (aP4_11) 16 18 r2 r3 (by omega) (by omega)
  have bP12 : aP4_12 r2 r4 ≤ MP4_12 := CaseSplit.le_mxr2 (aP4_12) 16 22 r2 r4 (by omega) (by omega)
  have bP13 : aP4_13 r2 r5 ≤ MP4_13 := CaseSplit.le_mxr2 (aP4_13) 16 28 r2 r5 (by omega) (by omega)
  have bP14 : aP4_14 r2 r6 ≤ MP4_14 := CaseSplit.le_mxr2 (aP4_14) 16 30 r2 r6 (by omega) (by omega)
  have bP15 : aP4_15 r3 r4 ≤ MP4_15 := CaseSplit.le_mxr2 (aP4_15) 18 22 r3 r4 (by omega) (by omega)
  have bP16 : aP4_16 r3 r5 ≤ MP4_16 := CaseSplit.le_mxr2 (aP4_16) 18 28 r3 r5 (by omega) (by omega)
  have bP17 : aP4_17 r3 r6 ≤ MP4_17 := CaseSplit.le_mxr2 (aP4_17) 18 30 r3 r6 (by omega) (by omega)
  have bP18 : aP4_18 r4 r5 ≤ MP4_18 := CaseSplit.le_mxr2 (aP4_18) 22 28 r4 r5 (by omega) (by omega)
  have bP19 : aP4_19 r4 r6 ≤ MP4_19 := CaseSplit.le_mxr2 (aP4_19) 22 30 r4 r6 (by omega) (by omega)
  have bP20 : aP4_20 r5 r6 ≤ MP4_20 := CaseSplit.le_mxr2 (aP4_20) 28 30 r5 r6 (by omega) (by omega)
  have hrhs : rhs4 = (∑ t ∈ Finset.range n4, w4 t) + 5 * (n4 : ℤ) := rfl
  have hc := cert4
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, hn15, hn16, hn17, hn18, hn19, hn20, bS0, bS1, bS2, bS3, bS4, bS5, bS6, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14, bP15, bP16, bP17, bP18, bP19, bP20]

end IncCert31
