/-
INCREMENT-WIDTH CERTIFICATE, step 29->31, case 25 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_29_31.json, which re-derives every number
from the primes alone).

Machine 31, INCREMENT width 65 = F_2(29) + s_min(31) = 55 + 10,
held gears [5, 7] at phases [3, 4].  Free gears [11, 13, 17, 19, 23, 29, 31].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 4.
-/
import IncCert31B

namespace IncCert31

/-! ### case 25: held gears at phases [3, 4] -/

def p25 : List ℕ := [0, 5, 7, 10, 12, 14, 15, 17, 19, 20, 22, 24, 27, 29, 34, 35, 40, 42, 45, 47, 49, 50, 52, 54, 55, 57, 59, 62, 64]
def q25 (t : ℕ) : ℕ := p25.getD t 0
def n25 : ℕ := 29
def yl25 : List ℤ := [0, 1, 0, 0, 1, 0, 0, 3, 1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 3, 1, 0, 0, 0, 0, 0]
def w25 (t : ℕ) : ℤ := yl25.getD t 0
def ul25 : List ℤ := [(-3), (-1), (-2), (-3), 0, (-1), (-2), (-3), 0, (-3), (-1), (-2), (-1), 1, 0, 3, 0, 2, 1, 2, 0, 0, 1, 0, (-5), (-5), (-2), 0, (-5), (-5), (-5), (-2), (-2), (-1), (-4), (-5), (-5), (-2), 0, (-4), (-4), 0, 2, 5, 1, 0, 0, 4, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-3), (-3), (-3), (-3), (-3), (-3), (-1), (-1), (-3), (-3), (-3), (-3), (-3), (-3), 0, (-1), (-14), (-3), (-3), (-3), (-3), (-3), 3, 3, 0, 1, 3, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-3), (-3), (-1), (-3), (-1), 0, (-3), 0, (-3), (-3), (-3), (-3), 1, (-3), (-1), (-3), (-3), 0, (-3), 0, (-3), (-3), (-3), (-3), (-3), (-3), 0, 0, 0, 0, (-3), 0, 0, 3, (-1), (-1), (-1), 1, 0, 0, 0, 0, 17, 22, 21, 22, 18, 13, 22, 22, 22, 20, 22, 13, 18, 22, 22, 21, 16, (-22), (-22), (-22), (-22), (-22), (-22), (-22), (-22), (-22), (-22), (-22), (-22), (-22), 16, 16, 14, 12, 11, 14, 16, 16, 14, 12, 11, 16, 12, 15, 12, 16, 16, 16, 16, (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), (-16), 18, 11, 13, 18, 11, 18, 7, 18, 17, 18, 17, 11, 11, 17, 18, 18, 12, 18, 18, 11, 18, 18, 11, (-18), (-18), (-18), (-18), (-20), (-18), (-18), (-18), (-18), (-18), (-18), (-18), (-18), 12, 12, 7, 6, 12, 12, 12, 11, 8, 9, 8, 11, 12, 9, 12, 5, 8, 11, 12, 12, 12, 4, 12, 12, 12, 4, 4, 12, 12, (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-12), (-3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-5), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-5), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 6, 13, 11, 9, 14, 6, 14, 10, 13, 14, 7, 14, 7, 14, 13, 10, 14, 6, 11, 10, 14, 14, 6, 14, 7, 14, 4, 4, 8, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, (-5), 0, (-7), 0, 0, 0, 0, 0, 0]
def u25 (k : ℕ) : ℤ := ul25.getD k 0

def c25_0 (r t : ℕ) : Bool := gb11 r (q25 t)
def c25_1 (r t : ℕ) : Bool := gb13 r (q25 t)
def c25_2 (r t : ℕ) : Bool := gb17 r (q25 t)
def c25_3 (r t : ℕ) : Bool := gb19 r (q25 t)
def c25_4 (r t : ℕ) : Bool := gb23 r (q25 t)
def c25_5 (r t : ℕ) : Bool := gb29 r (q25 t)
def c25_6 (r t : ℕ) : Bool := gb31 r (q25 t)

def S25_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (w25 t + 4) * (if c25_0 r t then 1 else 0)
def S25_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (w25 t + 4) * (if c25_1 r t then 1 else 0)
def S25_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (w25 t + 4) * (if c25_2 r t then 1 else 0)
def S25_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (w25 t + 4) * (if c25_3 r t then 1 else 0)
def S25_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (w25 t + 4) * (if c25_4 r t then 1 else 0)
def S25_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (w25 t + 4) * (if c25_5 r t then 1 else 0)
def S25_6 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (w25 t + 4) * (if c25_6 r t then 1 else 0)

def L25_0 (r : ℕ) : ℤ := u25 (13 + r) + u25 (41 + r) + u25 (71 + r) + u25 (105 + r) + u25 (145 + r) + u25 (187 + r)
def L25_1 (r : ℕ) : ℤ := u25 (0 + r) + u25 (215 + r) + u25 (247 + r) + u25 (283 + r) + u25 (325 + r) + u25 (369 + r)
def L25_2 (r : ℕ) : ℤ := u25 (24 + r) + u25 (198 + r) + u25 (401 + r) + u25 (441 + r) + u25 (487 + r) + u25 (535 + r)
def L25_3 (r : ℕ) : ℤ := u25 (52 + r) + u25 (228 + r) + u25 (382 + r) + u25 (575 + r) + u25 (623 + r) + u25 (673 + r)
def L25_4 (r : ℕ) : ℤ := u25 (82 + r) + u25 (260 + r) + u25 (418 + r) + u25 (552 + r) + u25 (721 + r) + u25 (775 + r)
def L25_5 (r : ℕ) : ℤ := u25 (116 + r) + u25 (296 + r) + u25 (458 + r) + u25 (594 + r) + u25 (692 + r) + u25 (829 + r)
def L25_6 (r : ℕ) : ℤ := u25 (156 + r) + u25 (338 + r) + u25 (504 + r) + u25 (642 + r) + u25 (744 + r) + u25 (798 + r)

def aS25_0 (r : ℕ) : ℤ := S25_0 r - L25_0 r
def MS25_0 : ℤ := CaseSplit.mxr (aS25_0) 10
def aS25_1 (r : ℕ) : ℤ := S25_1 r - L25_1 r
def MS25_1 : ℤ := CaseSplit.mxr (aS25_1) 12
def aS25_2 (r : ℕ) : ℤ := S25_2 r - L25_2 r
def MS25_2 : ℤ := CaseSplit.mxr (aS25_2) 16
def aS25_3 (r : ℕ) : ℤ := S25_3 r - L25_3 r
def MS25_3 : ℤ := CaseSplit.mxr (aS25_3) 18
def aS25_4 (r : ℕ) : ℤ := S25_4 r - L25_4 r
def MS25_4 : ℤ := CaseSplit.mxr (aS25_4) 22
def aS25_5 (r : ℕ) : ℤ := S25_5 r - L25_5 r
def MS25_5 : ℤ := CaseSplit.mxr (aS25_5) 28
def aS25_6 (r : ℕ) : ℤ := S25_6 r - L25_6 r
def MS25_6 : ℤ := CaseSplit.mxr (aS25_6) 30

def N25_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (if c25_0 ra t && c25_1 rb t then 1 else 0)
def aP25_0 (ra rb : ℕ) : ℤ := -(4) * N25_0 ra rb + u25 (0 + rb) + u25 (13 + ra)
def MP25_0 : ℤ := CaseSplit.mxr2 (aP25_0) 10 12
def N25_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (if c25_0 ra t && c25_2 rb t then 1 else 0)
def aP25_1 (ra rb : ℕ) : ℤ := -(4) * N25_1 ra rb + u25 (24 + rb) + u25 (41 + ra)
def MP25_1 : ℤ := CaseSplit.mxr2 (aP25_1) 10 16
def N25_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (if c25_0 ra t && c25_3 rb t then 1 else 0)
def aP25_2 (ra rb : ℕ) : ℤ := -(4) * N25_2 ra rb + u25 (52 + rb) + u25 (71 + ra)
def MP25_2 : ℤ := CaseSplit.mxr2 (aP25_2) 10 18
def N25_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (if c25_0 ra t && c25_4 rb t then 1 else 0)
def aP25_3 (ra rb : ℕ) : ℤ := -(4) * N25_3 ra rb + u25 (82 + rb) + u25 (105 + ra)
def MP25_3 : ℤ := CaseSplit.mxr2 (aP25_3) 10 22
def N25_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (if c25_0 ra t && c25_5 rb t then 1 else 0)
def aP25_4 (ra rb : ℕ) : ℤ := -(4) * N25_4 ra rb + u25 (116 + rb) + u25 (145 + ra)
def MP25_4 : ℤ := CaseSplit.mxr2 (aP25_4) 10 28
def N25_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (if c25_0 ra t && c25_6 rb t then 1 else 0)
def aP25_5 (ra rb : ℕ) : ℤ := -(4) * N25_5 ra rb + u25 (156 + rb) + u25 (187 + ra)
def MP25_5 : ℤ := CaseSplit.mxr2 (aP25_5) 10 30
def P25_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (if c25_1 ra t && c25_2 rb t then 1 else 0)
def C25_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (if c25_1 ra t && c25_2 rb t && c25_0 s t then 1 else 0)
def M25_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C25_6 ra rb) 10
def E25_6 : List ℕ := [21, 27, 36, 47, 52, 58, 75, 81, 88, 94, 111, 117, 122, 133, 142, 148, 178, 184, 190, 201, 206, 212]
def N25_6 (ra rb : ℕ) : ℤ := if E25_6.contains (ra * 17 + rb) = true then P25_6 ra rb - M25_6 ra rb else 0
def aP25_6 (ra rb : ℕ) : ℤ := -(4) * N25_6 ra rb + u25 (198 + rb) + u25 (215 + ra)
def MP25_6 : ℤ := CaseSplit.mxr2 (aP25_6) 12 16
def P25_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (if c25_1 ra t && c25_3 rb t then 1 else 0)
def C25_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (if c25_1 ra t && c25_3 rb t && c25_0 s t then 1 else 0)
def M25_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C25_7 ra rb) 10
def E25_7 : List ℕ := [27, 30, 33, 38, 64, 91, 106, 114, 140, 164, 167, 170, 198, 204, 240, 246]
def N25_7 (ra rb : ℕ) : ℤ := if E25_7.contains (ra * 19 + rb) = true then P25_7 ra rb - M25_7 ra rb else 0
def aP25_7 (ra rb : ℕ) : ℤ := -(4) * N25_7 ra rb + u25 (228 + rb) + u25 (247 + ra)
def MP25_7 : ℤ := CaseSplit.mxr2 (aP25_7) 12 18
def P25_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (if c25_1 ra t && c25_4 rb t then 1 else 0)
def C25_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (if c25_1 ra t && c25_4 rb t && c25_0 s t then 1 else 0)
def M25_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C25_8 ra rb) 10
def E25_8 : List ℕ := []
def N25_8 (ra rb : ℕ) : ℤ := if E25_8.contains (ra * 23 + rb) = true then P25_8 ra rb - M25_8 ra rb else 0
def aP25_8 (ra rb : ℕ) : ℤ := -(4) * N25_8 ra rb + u25 (260 + rb) + u25 (283 + ra)
def MP25_8 : ℤ := CaseSplit.mxr2 (aP25_8) 12 22
def P25_9 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (if c25_1 ra t && c25_5 rb t then 1 else 0)
def C25_9 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (if c25_1 ra t && c25_5 rb t && c25_0 s t then 1 else 0)
def M25_9 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C25_9 ra rb) 10
def E25_9 : List ℕ := [9, 43, 120, 159, 236, 270]
def N25_9 (ra rb : ℕ) : ℤ := if E25_9.contains (ra * 29 + rb) = true then P25_9 ra rb - M25_9 ra rb else 0
def aP25_9 (ra rb : ℕ) : ℤ := -(4) * N25_9 ra rb + u25 (296 + rb) + u25 (325 + ra)
def MP25_9 : ℤ := CaseSplit.mxr2 (aP25_9) 12 28
def P25_10 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (if c25_1 ra t && c25_6 rb t then 1 else 0)
def C25_10 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (if c25_1 ra t && c25_6 rb t && c25_0 s t then 1 else 0)
def M25_10 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C25_10 ra rb) 10
def E25_10 : List ℕ := [57, 67, 117, 153, 181, 186, 277, 310, 346, 396]
def N25_10 (ra rb : ℕ) : ℤ := if E25_10.contains (ra * 31 + rb) = true then P25_10 ra rb - M25_10 ra rb else 0
def aP25_10 (ra rb : ℕ) : ℤ := -(4) * N25_10 ra rb + u25 (338 + rb) + u25 (369 + ra)
def MP25_10 : ℤ := CaseSplit.mxr2 (aP25_10) 12 30
def N25_11 (_ra _rb : ℕ) : ℤ := 0
def aP25_11 (ra rb : ℕ) : ℤ := -(4) * N25_11 ra rb + u25 (382 + rb) + u25 (401 + ra)
def MP25_11 : ℤ := CaseSplit.mxr2 (aP25_11) 16 18
def N25_12 (_ra _rb : ℕ) : ℤ := 0
def aP25_12 (ra rb : ℕ) : ℤ := -(4) * N25_12 ra rb + u25 (418 + rb) + u25 (441 + ra)
def MP25_12 : ℤ := CaseSplit.mxr2 (aP25_12) 16 22
def N25_13 (_ra _rb : ℕ) : ℤ := 0
def aP25_13 (ra rb : ℕ) : ℤ := -(4) * N25_13 ra rb + u25 (458 + rb) + u25 (487 + ra)
def MP25_13 : ℤ := CaseSplit.mxr2 (aP25_13) 16 28
def N25_14 (_ra _rb : ℕ) : ℤ := 0
def aP25_14 (ra rb : ℕ) : ℤ := -(4) * N25_14 ra rb + u25 (504 + rb) + u25 (535 + ra)
def MP25_14 : ℤ := CaseSplit.mxr2 (aP25_14) 16 30
def N25_15 (_ra _rb : ℕ) : ℤ := 0
def aP25_15 (ra rb : ℕ) : ℤ := -(4) * N25_15 ra rb + u25 (552 + rb) + u25 (575 + ra)
def MP25_15 : ℤ := CaseSplit.mxr2 (aP25_15) 18 22
def N25_16 (_ra _rb : ℕ) : ℤ := 0
def aP25_16 (ra rb : ℕ) : ℤ := -(4) * N25_16 ra rb + u25 (594 + rb) + u25 (623 + ra)
def MP25_16 : ℤ := CaseSplit.mxr2 (aP25_16) 18 28
def N25_17 (_ra _rb : ℕ) : ℤ := 0
def aP25_17 (ra rb : ℕ) : ℤ := -(4) * N25_17 ra rb + u25 (642 + rb) + u25 (673 + ra)
def MP25_17 : ℤ := CaseSplit.mxr2 (aP25_17) 18 30
def N25_18 (_ra _rb : ℕ) : ℤ := 0
def aP25_18 (ra rb : ℕ) : ℤ := -(4) * N25_18 ra rb + u25 (692 + rb) + u25 (721 + ra)
def MP25_18 : ℤ := CaseSplit.mxr2 (aP25_18) 22 28
def N25_19 (_ra _rb : ℕ) : ℤ := 0
def aP25_19 (ra rb : ℕ) : ℤ := -(4) * N25_19 ra rb + u25 (744 + rb) + u25 (775 + ra)
def MP25_19 : ℤ := CaseSplit.mxr2 (aP25_19) 22 30
def N25_20 (_ra _rb : ℕ) : ℤ := 0
def aP25_20 (ra rb : ℕ) : ℤ := -(4) * N25_20 ra rb + u25 (798 + rb) + u25 (829 + ra)
def MP25_20 : ℤ := CaseSplit.mxr2 (aP25_20) 28 30

def rhs25 : ℤ := (∑ t ∈ Finset.range n25, w25 t) + 4 * (n25 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn25 : ∀ t, t < n25 → (0 : ℤ) ≤ w25 t := by decide
theorem plt25 : ∀ t, t < n25 → q25 t < 65 := by decide
theorem pfree25_5 : ∀ t, t < n25 → gb5 3 (q25 t) = false := by decide
theorem pfree25_7 : ∀ t, t < n25 → gb7 4 (q25 t) = false := by decide
theorem MSv25_0 : MS25_0 = 21 := by decide +kernel
theorem MSv25_1 : MS25_1 = 93 := by decide +kernel
theorem MSv25_2 : MS25_2 = 2 := by decide +kernel
theorem MSv25_3 : MS25_3 = 1 := by decide +kernel
theorem MSv25_4 : MS25_4 = 1 := by decide +kernel
theorem MSv25_5 : MS25_5 = 1 := by decide +kernel
theorem MSv25_6 : MS25_6 = 1 := by decide +kernel
theorem MPv25_0 : MP25_0 = 0 := by decide +kernel
theorem MPv25_1 : MP25_1 = 0 := by decide +kernel
theorem MPv25_2 : MP25_2 = 0 := by decide +kernel
theorem MPv25_3 : MP25_3 = 0 := by decide +kernel
theorem MPv25_4 : MP25_4 = 0 := by decide +kernel
theorem MPv25_5 : MP25_5 = 0 := by decide +kernel
theorem MPv25_6 : MP25_6 = 0 := by decide +kernel
theorem MPv25_7 : MP25_7 = 0 := by decide +kernel
theorem MPv25_8 : MP25_8 = 0 := by decide +kernel
theorem MPv25_9 : MP25_9 = 0 := by decide +kernel
theorem MPv25_10 : MP25_10 = 0 := by decide +kernel
theorem MPv25_11 : MP25_11 = 0 := by decide +kernel
theorem MPv25_12 : MP25_12 = 0 := by decide +kernel
theorem MPv25_13 : MP25_13 = 0 := by decide +kernel
theorem MPv25_14 : MP25_14 = 0 := by decide +kernel
theorem MPv25_15 : MP25_15 = 0 := by decide +kernel
theorem MPv25_16 : MP25_16 = 0 := by decide +kernel
theorem MPv25_17 : MP25_17 = 0 := by decide +kernel
theorem MPv25_18 : MP25_18 = 0 := by decide +kernel
theorem MPv25_19 : MP25_19 = 0 := by decide +kernel
theorem MPv25_20 : MP25_20 = 14 := by decide +kernel
theorem rhsv25 : rhs25 = 136 := by decide +kernel

/-- **The case-25 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 2/4.
    (Scaled by the common denominator 4: 134 < 136.) -/
theorem cert25 : MS25_0 + MS25_1 + MS25_2 + MS25_3 + MS25_4 + MS25_5 + MS25_6 + MP25_0 + MP25_1 + MP25_2 + MP25_3 + MP25_4 + MP25_5 + MP25_6 + MP25_7 + MP25_8 + MP25_9 + MP25_10 + MP25_11 + MP25_12 + MP25_13 + MP25_14 + MP25_15 + MP25_16 + MP25_17 + MP25_18 + MP25_19 + MP25_20 < rhs25 := by
  rw [MSv25_0, MSv25_1, MSv25_2, MSv25_3, MSv25_4, MSv25_5, MSv25_6, MPv25_0, MPv25_1, MPv25_2, MPv25_3, MPv25_4, MPv25_5, MPv25_6, MPv25_7, MPv25_8, MPv25_9, MPv25_10, MPv25_11, MPv25_12, MPv25_13, MPv25_14, MPv25_15, MPv25_16, MPv25_17, MPv25_18, MPv25_19, MPv25_20, rhsv25]
  decide

def Dg25 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := (if c25_0 r0 t then 1 else 0) + (if c25_1 r1 t then 1 else 0) + (if c25_2 r2 t then 1 else 0) + (if c25_3 r3 t then 1 else 0) + (if c25_4 r4 t then 1 else 0) + (if c25_5 r5 t then 1 else 0) + (if c25_6 r6 t then 1 else 0)
def Wl25_0 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c25_0 r0 t && c25_1 r1 t then 1 else 0
def Wl25_1 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c25_0 r0 t && c25_2 r2 t then 1 else 0
def Wl25_2 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c25_0 r0 t && c25_3 r3 t then 1 else 0
def Wl25_3 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c25_0 r0 t && c25_4 r4 t then 1 else 0
def Wl25_4 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c25_0 r0 t && c25_5 r5 t then 1 else 0
def Wl25_5 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c25_0 r0 t && c25_6 r6 t then 1 else 0
def Wl25_6 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c25_0 r0 t && c25_1 r1 t && c25_2 r2 t then 1 else 0
def Wl25_7 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c25_0 r0 t && c25_1 r1 t && c25_3 r3 t then 1 else 0
def Wl25_8 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c25_0 r0 t && c25_1 r1 t && c25_4 r4 t then 1 else 0
def Wl25_9 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c25_0 r0 t && c25_1 r1 t && c25_5 r5 t then 1 else 0
def Wl25_10 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c25_0 r0 t && c25_1 r1 t && c25_6 r6 t then 1 else 0
def Wl25_11 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c25_0 r0 t && !c25_1 r1 t && c25_2 r2 t && c25_3 r3 t then 1 else 0
def Wl25_12 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c25_0 r0 t && !c25_1 r1 t && c25_2 r2 t && c25_4 r4 t then 1 else 0
def Wl25_13 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c25_0 r0 t && !c25_1 r1 t && c25_2 r2 t && c25_5 r5 t then 1 else 0
def Wl25_14 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c25_0 r0 t && !c25_1 r1 t && c25_2 r2 t && c25_6 r6 t then 1 else 0
def Wl25_15 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c25_0 r0 t && !c25_1 r1 t && !c25_2 r2 t && c25_3 r3 t && c25_4 r4 t then 1 else 0
def Wl25_16 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c25_0 r0 t && !c25_1 r1 t && !c25_2 r2 t && c25_3 r3 t && c25_5 r5 t then 1 else 0
def Wl25_17 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c25_0 r0 t && !c25_1 r1 t && !c25_2 r2 t && c25_3 r3 t && c25_6 r6 t then 1 else 0
def Wl25_18 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c25_0 r0 t && !c25_1 r1 t && !c25_2 r2 t && !c25_3 r3 t && c25_4 r4 t && c25_5 r5 t then 1 else 0
def Wl25_19 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c25_0 r0 t && !c25_1 r1 t && !c25_2 r2 t && !c25_3 r3 t && c25_4 r4 t && c25_6 r6 t then 1 else 0
def Wl25_20 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c25_0 r0 t && !c25_1 r1 t && !c25_2 r2 t && !c25_3 r3 t && !c25_4 r4 t && c25_5 r5 t && c25_6 r6 t then 1 else 0

/-- **No configuration blocks the whole window in case 25.** -/
theorem nocov25 {r0 r1 r2 r3 r4 r5 r6 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29) (h6 : r6 < 31)
    (hcov : ∀ t, t < n25 → (c25_0 r0 t || c25_1 r1 t || c25_2 r2 t || c25_3 r3 t || c25_4 r4 t || c25_5 r5 t || c25_6 r6 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n25, (1 : ℤ) + (Wl25_0 r0 r1 r2 r3 r4 r5 r6 t + Wl25_1 r0 r1 r2 r3 r4 r5 r6 t + Wl25_2 r0 r1 r2 r3 r4 r5 r6 t + Wl25_3 r0 r1 r2 r3 r4 r5 r6 t + Wl25_4 r0 r1 r2 r3 r4 r5 r6 t + Wl25_5 r0 r1 r2 r3 r4 r5 r6 t + Wl25_6 r0 r1 r2 r3 r4 r5 r6 t + Wl25_7 r0 r1 r2 r3 r4 r5 r6 t + Wl25_8 r0 r1 r2 r3 r4 r5 r6 t + Wl25_9 r0 r1 r2 r3 r4 r5 r6 t + Wl25_10 r0 r1 r2 r3 r4 r5 r6 t + Wl25_11 r0 r1 r2 r3 r4 r5 r6 t + Wl25_12 r0 r1 r2 r3 r4 r5 r6 t + Wl25_13 r0 r1 r2 r3 r4 r5 r6 t + Wl25_14 r0 r1 r2 r3 r4 r5 r6 t + Wl25_15 r0 r1 r2 r3 r4 r5 r6 t + Wl25_16 r0 r1 r2 r3 r4 r5 r6 t + Wl25_17 r0 r1 r2 r3 r4 r5 r6 t + Wl25_18 r0 r1 r2 r3 r4 r5 r6 t + Wl25_19 r0 r1 r2 r3 r4 r5 r6 t + Wl25_20 r0 r1 r2 r3 r4 r5 r6 t) ≤ Dg25 r0 r1 r2 r3 r4 r5 r6 t := by
    intro t ht
    simp only [Wl25_0, Wl25_1, Wl25_2, Wl25_3, Wl25_4, Wl25_5, Wl25_6, Wl25_7, Wl25_8, Wl25_9, Wl25_10, Wl25_11, Wl25_12, Wl25_13, Wl25_14, Wl25_15, Wl25_16, Wl25_17, Wl25_18, Wl25_19, Wl25_20, Dg25]
    exact CaseSplit.lowest7 _ _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n25, (1 : ℤ) ≤ Dg25 r0 r1 r2 r3 r4 r5 r6 t := by
    intro t ht
    simp only [Dg25]
    exact CaseSplit.degpos7 _ _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n25 : ℤ) + ((∑ t ∈ Finset.range n25, Wl25_0 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n25, Wl25_1 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n25, Wl25_2 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n25, Wl25_3 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n25, Wl25_4 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n25, Wl25_5 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n25, Wl25_6 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n25, Wl25_7 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n25, Wl25_8 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n25, Wl25_9 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n25, Wl25_10 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n25, Wl25_11 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n25, Wl25_12 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n25, Wl25_13 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n25, Wl25_14 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n25, Wl25_15 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n25, Wl25_16 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n25, Wl25_17 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n25, Wl25_18 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n25, Wl25_19 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n25, Wl25_20 r0 r1 r2 r3 r4 r5 r6 t)) ≤ ∑ t ∈ Finset.range n25, Dg25 r0 r1 r2 r3 r4 r5 r6 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N25_0 r0 r1 ≤ ∑ t ∈ Finset.range n25, Wl25_0 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N25_0, Wl25_0, le_refl]
  have hn1 : N25_1 r0 r2 ≤ ∑ t ∈ Finset.range n25, Wl25_1 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N25_1, Wl25_1, le_refl]
  have hn2 : N25_2 r0 r3 ≤ ∑ t ∈ Finset.range n25, Wl25_2 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N25_2, Wl25_2, le_refl]
  have hn3 : N25_3 r0 r4 ≤ ∑ t ∈ Finset.range n25, Wl25_3 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N25_3, Wl25_3, le_refl]
  have hn4 : N25_4 r0 r5 ≤ ∑ t ∈ Finset.range n25, Wl25_4 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N25_4, Wl25_4, le_refl]
  have hn5 : N25_5 r0 r6 ≤ ∑ t ∈ Finset.range n25, Wl25_5 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N25_5, Wl25_5, le_refl]
  have hn6 : N25_6 r1 r2 ≤ ∑ t ∈ Finset.range n25, Wl25_6 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n25, Wl25_6 r0 r1 r2 r3 r4 r5 r6 t
        = (if c25_1 r1 t && c25_2 r2 t then (1:ℤ) else 0)
          - (if c25_1 r1 t && c25_2 r2 t && c25_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl25_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n25, Wl25_6 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl25_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n25, Wl25_6 r0 r1 r2 r3 r4 r5 r6 t
        = P25_6 r1 r2 - C25_6 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P25_6, C25_6]
    have hm : C25_6 r1 r2 r0 ≤ M25_6 r1 r2 :=
      CaseSplit.le_mxr (C25_6 r1 r2) 10 r0 (by omega)
    simp only [N25_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N25_7 r1 r3 ≤ ∑ t ∈ Finset.range n25, Wl25_7 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n25, Wl25_7 r0 r1 r2 r3 r4 r5 r6 t
        = (if c25_1 r1 t && c25_3 r3 t then (1:ℤ) else 0)
          - (if c25_1 r1 t && c25_3 r3 t && c25_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl25_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n25, Wl25_7 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl25_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n25, Wl25_7 r0 r1 r2 r3 r4 r5 r6 t
        = P25_7 r1 r3 - C25_7 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P25_7, C25_7]
    have hm : C25_7 r1 r3 r0 ≤ M25_7 r1 r3 :=
      CaseSplit.le_mxr (C25_7 r1 r3) 10 r0 (by omega)
    simp only [N25_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N25_8 r1 r4 ≤ ∑ t ∈ Finset.range n25, Wl25_8 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n25, Wl25_8 r0 r1 r2 r3 r4 r5 r6 t
        = (if c25_1 r1 t && c25_4 r4 t then (1:ℤ) else 0)
          - (if c25_1 r1 t && c25_4 r4 t && c25_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl25_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n25, Wl25_8 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl25_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n25, Wl25_8 r0 r1 r2 r3 r4 r5 r6 t
        = P25_8 r1 r4 - C25_8 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P25_8, C25_8]
    have hm : C25_8 r1 r4 r0 ≤ M25_8 r1 r4 :=
      CaseSplit.le_mxr (C25_8 r1 r4) 10 r0 (by omega)
    simp only [N25_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N25_9 r1 r5 ≤ ∑ t ∈ Finset.range n25, Wl25_9 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n25, Wl25_9 r0 r1 r2 r3 r4 r5 r6 t
        = (if c25_1 r1 t && c25_5 r5 t then (1:ℤ) else 0)
          - (if c25_1 r1 t && c25_5 r5 t && c25_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl25_9]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n25, Wl25_9 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl25_9]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n25, Wl25_9 r0 r1 r2 r3 r4 r5 r6 t
        = P25_9 r1 r5 - C25_9 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P25_9, C25_9]
    have hm : C25_9 r1 r5 r0 ≤ M25_9 r1 r5 :=
      CaseSplit.le_mxr (C25_9 r1 r5) 10 r0 (by omega)
    simp only [N25_9]
    split
    · rw [hL]; omega
    · exact hnn
  have hn10 : N25_10 r1 r6 ≤ ∑ t ∈ Finset.range n25, Wl25_10 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n25, Wl25_10 r0 r1 r2 r3 r4 r5 r6 t
        = (if c25_1 r1 t && c25_6 r6 t then (1:ℤ) else 0)
          - (if c25_1 r1 t && c25_6 r6 t && c25_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl25_10]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n25, Wl25_10 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl25_10]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n25, Wl25_10 r0 r1 r2 r3 r4 r5 r6 t
        = P25_10 r1 r6 - C25_10 r1 r6 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P25_10, C25_10]
    have hm : C25_10 r1 r6 r0 ≤ M25_10 r1 r6 :=
      CaseSplit.le_mxr (C25_10 r1 r6) 10 r0 (by omega)
    simp only [N25_10]
    split
    · rw [hL]; omega
    · exact hnn
  have hn11 : N25_11 r2 r3 ≤ ∑ t ∈ Finset.range n25, Wl25_11 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N25_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl25_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N25_12 r2 r4 ≤ ∑ t ∈ Finset.range n25, Wl25_12 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N25_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl25_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N25_13 r2 r5 ≤ ∑ t ∈ Finset.range n25, Wl25_13 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N25_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl25_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N25_14 r2 r6 ≤ ∑ t ∈ Finset.range n25, Wl25_14 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N25_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl25_14]
    exact CaseSplit.ind_nonneg _
  have hn15 : N25_15 r3 r4 ≤ ∑ t ∈ Finset.range n25, Wl25_15 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N25_15]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl25_15]
    exact CaseSplit.ind_nonneg _
  have hn16 : N25_16 r3 r5 ≤ ∑ t ∈ Finset.range n25, Wl25_16 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N25_16]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl25_16]
    exact CaseSplit.ind_nonneg _
  have hn17 : N25_17 r3 r6 ≤ ∑ t ∈ Finset.range n25, Wl25_17 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N25_17]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl25_17]
    exact CaseSplit.ind_nonneg _
  have hn18 : N25_18 r4 r5 ≤ ∑ t ∈ Finset.range n25, Wl25_18 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N25_18]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl25_18]
    exact CaseSplit.ind_nonneg _
  have hn19 : N25_19 r4 r6 ≤ ∑ t ∈ Finset.range n25, Wl25_19 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N25_19]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl25_19]
    exact CaseSplit.ind_nonneg _
  have hn20 : N25_20 r5 r6 ≤ ∑ t ∈ Finset.range n25, Wl25_20 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N25_20]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl25_20]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n25, (w25 t + 4) * Dg25 r0 r1 r2 r3 r4 r5 r6 t = S25_0 r0 + S25_1 r1 + S25_2 r2 + S25_3 r3 + S25_4 r4 + S25_5 r5 + S25_6 r6 := by
    simp only [S25_0, S25_1, S25_2, S25_3, S25_4, S25_5, S25_6, Dg25, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n25, (w25 t + 4) * Dg25 r0 r1 r2 r3 r4 r5 r6 t
      = (∑ t ∈ Finset.range n25, w25 t * Dg25 r0 r1 r2 r3 r4 r5 r6 t)
        + 4 * (∑ t ∈ Finset.range n25, Dg25 r0 r1 r2 r3 r4 r5 r6 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n25, w25 t)
      ≤ ∑ t ∈ Finset.range n25, w25 t * Dg25 r0 r1 r2 r3 r4 r5 r6 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg25 r0 r1 r2 r3 r4 r5 r6 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w25 t := wnn25 t (Finset.mem_range.mp ht)
    calc w25 t = w25 t * 1 := (mul_one _).symm
      _ ≤ w25 t * Dg25 r0 r1 r2 r3 r4 r5 r6 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS25_0 r0 + aS25_1 r1 + aS25_2 r2 + aS25_3 r3 + aS25_4 r4 + aS25_5 r5 + aS25_6 r6) + (aP25_0 r0 r1 + aP25_1 r0 r2 + aP25_2 r0 r3 + aP25_3 r0 r4 + aP25_4 r0 r5 + aP25_5 r0 r6 + aP25_6 r1 r2 + aP25_7 r1 r3 + aP25_8 r1 r4 + aP25_9 r1 r5 + aP25_10 r1 r6 + aP25_11 r2 r3 + aP25_12 r2 r4 + aP25_13 r2 r5 + aP25_14 r2 r6 + aP25_15 r3 r4 + aP25_16 r3 r5 + aP25_17 r3 r6 + aP25_18 r4 r5 + aP25_19 r4 r6 + aP25_20 r5 r6) = (S25_0 r0 + S25_1 r1 + S25_2 r2 + S25_3 r3 + S25_4 r4 + S25_5 r5 + S25_6 r6) - 4 * (N25_0 r0 r1 + N25_1 r0 r2 + N25_2 r0 r3 + N25_3 r0 r4 + N25_4 r0 r5 + N25_5 r0 r6 + N25_6 r1 r2 + N25_7 r1 r3 + N25_8 r1 r4 + N25_9 r1 r5 + N25_10 r1 r6 + N25_11 r2 r3 + N25_12 r2 r4 + N25_13 r2 r5 + N25_14 r2 r6 + N25_15 r3 r4 + N25_16 r3 r5 + N25_17 r3 r6 + N25_18 r4 r5 + N25_19 r4 r6 + N25_20 r5 r6) := by
    simp only [aS25_0, aS25_1, aS25_2, aS25_3, aS25_4, aS25_5, aS25_6, aP25_0, aP25_1, aP25_2, aP25_3, aP25_4, aP25_5, aP25_6, aP25_7, aP25_8, aP25_9, aP25_10, aP25_11, aP25_12, aP25_13, aP25_14, aP25_15, aP25_16, aP25_17, aP25_18, aP25_19, aP25_20, L25_0, L25_1, L25_2, L25_3, L25_4, L25_5, L25_6]
    ring
  have bS0 : aS25_0 r0 ≤ MS25_0 := CaseSplit.le_mxr (aS25_0) 10 r0 (by omega)
  have bS1 : aS25_1 r1 ≤ MS25_1 := CaseSplit.le_mxr (aS25_1) 12 r1 (by omega)
  have bS2 : aS25_2 r2 ≤ MS25_2 := CaseSplit.le_mxr (aS25_2) 16 r2 (by omega)
  have bS3 : aS25_3 r3 ≤ MS25_3 := CaseSplit.le_mxr (aS25_3) 18 r3 (by omega)
  have bS4 : aS25_4 r4 ≤ MS25_4 := CaseSplit.le_mxr (aS25_4) 22 r4 (by omega)
  have bS5 : aS25_5 r5 ≤ MS25_5 := CaseSplit.le_mxr (aS25_5) 28 r5 (by omega)
  have bS6 : aS25_6 r6 ≤ MS25_6 := CaseSplit.le_mxr (aS25_6) 30 r6 (by omega)
  have bP0 : aP25_0 r0 r1 ≤ MP25_0 := CaseSplit.le_mxr2 (aP25_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP25_1 r0 r2 ≤ MP25_1 := CaseSplit.le_mxr2 (aP25_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP25_2 r0 r3 ≤ MP25_2 := CaseSplit.le_mxr2 (aP25_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP25_3 r0 r4 ≤ MP25_3 := CaseSplit.le_mxr2 (aP25_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP25_4 r0 r5 ≤ MP25_4 := CaseSplit.le_mxr2 (aP25_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP25_5 r0 r6 ≤ MP25_5 := CaseSplit.le_mxr2 (aP25_5) 10 30 r0 r6 (by omega) (by omega)
  have bP6 : aP25_6 r1 r2 ≤ MP25_6 := CaseSplit.le_mxr2 (aP25_6) 12 16 r1 r2 (by omega) (by omega)
  have bP7 : aP25_7 r1 r3 ≤ MP25_7 := CaseSplit.le_mxr2 (aP25_7) 12 18 r1 r3 (by omega) (by omega)
  have bP8 : aP25_8 r1 r4 ≤ MP25_8 := CaseSplit.le_mxr2 (aP25_8) 12 22 r1 r4 (by omega) (by omega)
  have bP9 : aP25_9 r1 r5 ≤ MP25_9 := CaseSplit.le_mxr2 (aP25_9) 12 28 r1 r5 (by omega) (by omega)
  have bP10 : aP25_10 r1 r6 ≤ MP25_10 := CaseSplit.le_mxr2 (aP25_10) 12 30 r1 r6 (by omega) (by omega)
  have bP11 : aP25_11 r2 r3 ≤ MP25_11 := CaseSplit.le_mxr2 (aP25_11) 16 18 r2 r3 (by omega) (by omega)
  have bP12 : aP25_12 r2 r4 ≤ MP25_12 := CaseSplit.le_mxr2 (aP25_12) 16 22 r2 r4 (by omega) (by omega)
  have bP13 : aP25_13 r2 r5 ≤ MP25_13 := CaseSplit.le_mxr2 (aP25_13) 16 28 r2 r5 (by omega) (by omega)
  have bP14 : aP25_14 r2 r6 ≤ MP25_14 := CaseSplit.le_mxr2 (aP25_14) 16 30 r2 r6 (by omega) (by omega)
  have bP15 : aP25_15 r3 r4 ≤ MP25_15 := CaseSplit.le_mxr2 (aP25_15) 18 22 r3 r4 (by omega) (by omega)
  have bP16 : aP25_16 r3 r5 ≤ MP25_16 := CaseSplit.le_mxr2 (aP25_16) 18 28 r3 r5 (by omega) (by omega)
  have bP17 : aP25_17 r3 r6 ≤ MP25_17 := CaseSplit.le_mxr2 (aP25_17) 18 30 r3 r6 (by omega) (by omega)
  have bP18 : aP25_18 r4 r5 ≤ MP25_18 := CaseSplit.le_mxr2 (aP25_18) 22 28 r4 r5 (by omega) (by omega)
  have bP19 : aP25_19 r4 r6 ≤ MP25_19 := CaseSplit.le_mxr2 (aP25_19) 22 30 r4 r6 (by omega) (by omega)
  have bP20 : aP25_20 r5 r6 ≤ MP25_20 := CaseSplit.le_mxr2 (aP25_20) 28 30 r5 r6 (by omega) (by omega)
  have hrhs : rhs25 = (∑ t ∈ Finset.range n25, w25 t) + 4 * (n25 : ℤ) := rfl
  have hc := cert25
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, hn15, hn16, hn17, hn18, hn19, hn20, bS0, bS1, bS2, bS3, bS4, bS5, bS6, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14, bP15, bP16, bP17, bP18, bP19, bP20]

end IncCert31
