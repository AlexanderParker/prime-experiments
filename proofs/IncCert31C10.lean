/-
INCREMENT-WIDTH CERTIFICATE, step 29->31, case 10 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_29_31.json, which re-derives every number
from the primes alone).

Machine 31, INCREMENT width 65 = F_2(29) + s_min(31) = 55 + 10,
held gears [5, 7] at phases [1, 3].  Free gears [11, 13, 17, 19, 23, 29, 31].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 12.
-/
import IncCert31B

namespace IncCert31

/-! ### case 10: held gears at phases [1, 3] -/

def p10 : List ℕ := [1, 2, 4, 6, 7, 9, 11, 14, 16, 21, 22, 27, 29, 32, 34, 36, 37, 39, 41, 42, 44, 46, 49, 51, 56, 57, 62, 64]
def q10 (t : ℕ) : ℕ := p10.getD t 0
def n10 : ℕ := 28
def yl10 : List ℤ := [0, 9, 12, 0, 4, 12, 0, 2, 6, 0, 0, 1, 8, 8, 10, 10, 8, 11, 10, 0, 10, 6, 3, 4, 5, 5, 0, 1]
def w10 (t : ℕ) : ℤ := yl10.getD t 0
def ul10 : List ℤ := [(-2), (-2), (-2), 0, (-2), (-2), (-2), (-2), 0, (-2), 0, (-2), 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0, (-4), (-4), (-4), 0, 0, (-4), (-4), (-4), (-4), 0, 0, (-4), (-4), (-4), (-4), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, (-6), (-1), (-6), (-1), (-6), (-6), (-6), 0, 0, 0, (-6), (-6), 0, 0, 0, 0, (-1), (-6), 0, 6, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, (-4), (-4), (-4), 0, (-4), (-1), (-1), (-4), 0, (-4), (-4), 0, (-1), 0, (-1), (-4), 0, (-4), (-4), (-4), (-1), 0, (-1), 0, 0, 4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-20), 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-6), 0, (-6), (-6), (-6), 0, 0, (-6), (-6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 46, 36, 38, 34, 46, 18, 46, 46, 46, 46, 35, 41, 46, 46, 46, 46, (-46), (-46), (-46), (-46), (-46), (-46), (-46), (-46), (-46), (-46), (-46), (-46), (-46), 40, 40, 37, 35, 31, 40, 32, 34, 40, 40, 27, 40, 38, 40, 40, 40, 40, 40, 40, (-40), (-40), (-40), (-40), (-40), (-40), (-40), (-40), (-40), (-40), (-40), (-40), (-40), 41, 41, 19, 41, 16, 28, 38, 4, 41, 41, 35, 35, 10, 41, 26, 28, 41, 41, 41, 16, 16, 41, 10, (-41), (-41), (-41), (-41), (-41), (-41), (-41), (-41), (-41), (-41), (-41), (-41), (-41), 16, 32, 29, 32, 32, 14, 11, 30, 12, 15, 32, 6, 32, 32, 32, 18, 14, 32, 32, 32, 32, 20, 32, 12, 31, 32, 32, 32, 32, (-32), (-32), (-32), (-32), (-32), (-32), (-32), (-32), (-32), (-32), (-32), (-32), (-32), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-11), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-7), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 29, 17, 23, 26, 12, 10, 14, 9, 7, 23, 23, 8, 15, 6, 18, 26, 18, 26, 10, 32, 28, 32, 32, 23, 32, 22, 18, 32, 10, 27, 0, (-8), 0, (-10), (-17), 0, 0, 0, 0, 0, (-24), 0, 0, 0, (-6), 0, 0, 0, (-7), 0, 0, 0, 0, 0, 0, 0, 0, (-6), 0]
def u10 (k : ℕ) : ℤ := ul10.getD k 0

def c10_0 (r t : ℕ) : Bool := gb11 r (q10 t)
def c10_1 (r t : ℕ) : Bool := gb13 r (q10 t)
def c10_2 (r t : ℕ) : Bool := gb17 r (q10 t)
def c10_3 (r t : ℕ) : Bool := gb19 r (q10 t)
def c10_4 (r t : ℕ) : Bool := gb23 r (q10 t)
def c10_5 (r t : ℕ) : Bool := gb29 r (q10 t)
def c10_6 (r t : ℕ) : Bool := gb31 r (q10 t)

def S10_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (w10 t + 6) * (if c10_0 r t then 1 else 0)
def S10_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (w10 t + 6) * (if c10_1 r t then 1 else 0)
def S10_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (w10 t + 6) * (if c10_2 r t then 1 else 0)
def S10_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (w10 t + 6) * (if c10_3 r t then 1 else 0)
def S10_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (w10 t + 6) * (if c10_4 r t then 1 else 0)
def S10_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (w10 t + 6) * (if c10_5 r t then 1 else 0)
def S10_6 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (w10 t + 6) * (if c10_6 r t then 1 else 0)

def L10_0 (r : ℕ) : ℤ := u10 (13 + r) + u10 (41 + r) + u10 (71 + r) + u10 (105 + r) + u10 (145 + r) + u10 (187 + r)
def L10_1 (r : ℕ) : ℤ := u10 (0 + r) + u10 (215 + r) + u10 (247 + r) + u10 (283 + r) + u10 (325 + r) + u10 (369 + r)
def L10_2 (r : ℕ) : ℤ := u10 (24 + r) + u10 (198 + r) + u10 (401 + r) + u10 (441 + r) + u10 (487 + r) + u10 (535 + r)
def L10_3 (r : ℕ) : ℤ := u10 (52 + r) + u10 (228 + r) + u10 (382 + r) + u10 (575 + r) + u10 (623 + r) + u10 (673 + r)
def L10_4 (r : ℕ) : ℤ := u10 (82 + r) + u10 (260 + r) + u10 (418 + r) + u10 (552 + r) + u10 (721 + r) + u10 (775 + r)
def L10_5 (r : ℕ) : ℤ := u10 (116 + r) + u10 (296 + r) + u10 (458 + r) + u10 (594 + r) + u10 (692 + r) + u10 (829 + r)
def L10_6 (r : ℕ) : ℤ := u10 (156 + r) + u10 (338 + r) + u10 (504 + r) + u10 (642 + r) + u10 (744 + r) + u10 (798 + r)

def aS10_0 (r : ℕ) : ℤ := S10_0 r - L10_0 r
def MS10_0 : ℤ := CaseSplit.mxr (aS10_0) 10
def aS10_1 (r : ℕ) : ℤ := S10_1 r - L10_1 r
def MS10_1 : ℤ := CaseSplit.mxr (aS10_1) 12
def aS10_2 (r : ℕ) : ℤ := S10_2 r - L10_2 r
def MS10_2 : ℤ := CaseSplit.mxr (aS10_2) 16
def aS10_3 (r : ℕ) : ℤ := S10_3 r - L10_3 r
def MS10_3 : ℤ := CaseSplit.mxr (aS10_3) 18
def aS10_4 (r : ℕ) : ℤ := S10_4 r - L10_4 r
def MS10_4 : ℤ := CaseSplit.mxr (aS10_4) 22
def aS10_5 (r : ℕ) : ℤ := S10_5 r - L10_5 r
def MS10_5 : ℤ := CaseSplit.mxr (aS10_5) 28
def aS10_6 (r : ℕ) : ℤ := S10_6 r - L10_6 r
def MS10_6 : ℤ := CaseSplit.mxr (aS10_6) 30

def N10_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_0 ra t && c10_1 rb t then 1 else 0)
def aP10_0 (ra rb : ℕ) : ℤ := -(6) * N10_0 ra rb + u10 (0 + rb) + u10 (13 + ra)
def MP10_0 : ℤ := CaseSplit.mxr2 (aP10_0) 10 12
def N10_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_0 ra t && c10_2 rb t then 1 else 0)
def aP10_1 (ra rb : ℕ) : ℤ := -(6) * N10_1 ra rb + u10 (24 + rb) + u10 (41 + ra)
def MP10_1 : ℤ := CaseSplit.mxr2 (aP10_1) 10 16
def N10_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_0 ra t && c10_3 rb t then 1 else 0)
def aP10_2 (ra rb : ℕ) : ℤ := -(6) * N10_2 ra rb + u10 (52 + rb) + u10 (71 + ra)
def MP10_2 : ℤ := CaseSplit.mxr2 (aP10_2) 10 18
def N10_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_0 ra t && c10_4 rb t then 1 else 0)
def aP10_3 (ra rb : ℕ) : ℤ := -(6) * N10_3 ra rb + u10 (82 + rb) + u10 (105 + ra)
def MP10_3 : ℤ := CaseSplit.mxr2 (aP10_3) 10 22
def N10_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_0 ra t && c10_5 rb t then 1 else 0)
def aP10_4 (ra rb : ℕ) : ℤ := -(6) * N10_4 ra rb + u10 (116 + rb) + u10 (145 + ra)
def MP10_4 : ℤ := CaseSplit.mxr2 (aP10_4) 10 28
def N10_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_0 ra t && c10_6 rb t then 1 else 0)
def aP10_5 (ra rb : ℕ) : ℤ := -(6) * N10_5 ra rb + u10 (156 + rb) + u10 (187 + ra)
def MP10_5 : ℤ := CaseSplit.mxr2 (aP10_5) 10 30
def P10_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_1 ra t && c10_2 rb t then 1 else 0)
def C10_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_1 ra t && c10_2 rb t && c10_0 s t then 1 else 0)
def M10_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C10_6 ra rb) 10
def E10_6 : List ℕ := [43, 49, 54, 65, 90, 101, 129, 135, 138, 144, 174, 180, 196, 202]
def N10_6 (ra rb : ℕ) : ℤ := if E10_6.contains (ra * 17 + rb) = true then P10_6 ra rb - M10_6 ra rb else 0
def aP10_6 (ra rb : ℕ) : ℤ := -(6) * N10_6 ra rb + u10 (198 + rb) + u10 (215 + ra)
def MP10_6 : ℤ := CaseSplit.mxr2 (aP10_6) 12 16
def P10_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_1 ra t && c10_3 rb t then 1 else 0)
def C10_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_1 ra t && c10_3 rb t && c10_0 s t then 1 else 0)
def M10_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C10_7 ra rb) 10
def E10_7 : List ℕ := [21, 24, 27, 30, 51, 58, 66, 100, 106, 127, 130, 134, 158, 164, 192, 198, 206, 234, 237, 240]
def N10_7 (ra rb : ℕ) : ℤ := if E10_7.contains (ra * 19 + rb) = true then P10_7 ra rb - M10_7 ra rb else 0
def aP10_7 (ra rb : ℕ) : ℤ := -(6) * N10_7 ra rb + u10 (228 + rb) + u10 (247 + ra)
def MP10_7 : ℤ := CaseSplit.mxr2 (aP10_7) 12 18
def P10_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_1 ra t && c10_4 rb t then 1 else 0)
def C10_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_1 ra t && c10_4 rb t && c10_0 s t then 1 else 0)
def M10_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C10_8 ra rb) 10
def E10_8 : List ℕ := [26]
def N10_8 (ra rb : ℕ) : ℤ := if E10_8.contains (ra * 23 + rb) = true then P10_8 ra rb - M10_8 ra rb else 0
def aP10_8 (ra rb : ℕ) : ℤ := -(6) * N10_8 ra rb + u10 (260 + rb) + u10 (283 + ra)
def MP10_8 : ℤ := CaseSplit.mxr2 (aP10_8) 12 22
def P10_9 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_1 ra t && c10_5 rb t then 1 else 0)
def C10_9 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_1 ra t && c10_5 rb t && c10_0 s t then 1 else 0)
def M10_9 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C10_9 ra rb) 10
def E10_9 : List ℕ := [22, 133, 249, 283]
def N10_9 (ra rb : ℕ) : ℤ := if E10_9.contains (ra * 29 + rb) = true then P10_9 ra rb - M10_9 ra rb else 0
def aP10_9 (ra rb : ℕ) : ℤ := -(6) * N10_9 ra rb + u10 (296 + rb) + u10 (325 + ra)
def MP10_9 : ℤ := CaseSplit.mxr2 (aP10_9) 12 28
def P10_10 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_1 ra t && c10_6 rb t then 1 else 0)
def C10_10 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n10, (if c10_1 ra t && c10_6 rb t && c10_0 s t then 1 else 0)
def M10_10 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C10_10 ra rb) 10
def E10_10 : List ℕ := [218, 342]
def N10_10 (ra rb : ℕ) : ℤ := if E10_10.contains (ra * 31 + rb) = true then P10_10 ra rb - M10_10 ra rb else 0
def aP10_10 (ra rb : ℕ) : ℤ := -(6) * N10_10 ra rb + u10 (338 + rb) + u10 (369 + ra)
def MP10_10 : ℤ := CaseSplit.mxr2 (aP10_10) 12 30
def N10_11 (_ra _rb : ℕ) : ℤ := 0
def aP10_11 (ra rb : ℕ) : ℤ := -(6) * N10_11 ra rb + u10 (382 + rb) + u10 (401 + ra)
def MP10_11 : ℤ := CaseSplit.mxr2 (aP10_11) 16 18
def N10_12 (_ra _rb : ℕ) : ℤ := 0
def aP10_12 (ra rb : ℕ) : ℤ := -(6) * N10_12 ra rb + u10 (418 + rb) + u10 (441 + ra)
def MP10_12 : ℤ := CaseSplit.mxr2 (aP10_12) 16 22
def N10_13 (_ra _rb : ℕ) : ℤ := 0
def aP10_13 (ra rb : ℕ) : ℤ := -(6) * N10_13 ra rb + u10 (458 + rb) + u10 (487 + ra)
def MP10_13 : ℤ := CaseSplit.mxr2 (aP10_13) 16 28
def N10_14 (_ra _rb : ℕ) : ℤ := 0
def aP10_14 (ra rb : ℕ) : ℤ := -(6) * N10_14 ra rb + u10 (504 + rb) + u10 (535 + ra)
def MP10_14 : ℤ := CaseSplit.mxr2 (aP10_14) 16 30
def N10_15 (_ra _rb : ℕ) : ℤ := 0
def aP10_15 (ra rb : ℕ) : ℤ := -(6) * N10_15 ra rb + u10 (552 + rb) + u10 (575 + ra)
def MP10_15 : ℤ := CaseSplit.mxr2 (aP10_15) 18 22
def N10_16 (_ra _rb : ℕ) : ℤ := 0
def aP10_16 (ra rb : ℕ) : ℤ := -(6) * N10_16 ra rb + u10 (594 + rb) + u10 (623 + ra)
def MP10_16 : ℤ := CaseSplit.mxr2 (aP10_16) 18 28
def N10_17 (_ra _rb : ℕ) : ℤ := 0
def aP10_17 (ra rb : ℕ) : ℤ := -(6) * N10_17 ra rb + u10 (642 + rb) + u10 (673 + ra)
def MP10_17 : ℤ := CaseSplit.mxr2 (aP10_17) 18 30
def N10_18 (_ra _rb : ℕ) : ℤ := 0
def aP10_18 (ra rb : ℕ) : ℤ := -(6) * N10_18 ra rb + u10 (692 + rb) + u10 (721 + ra)
def MP10_18 : ℤ := CaseSplit.mxr2 (aP10_18) 22 28
def N10_19 (_ra _rb : ℕ) : ℤ := 0
def aP10_19 (ra rb : ℕ) : ℤ := -(6) * N10_19 ra rb + u10 (744 + rb) + u10 (775 + ra)
def MP10_19 : ℤ := CaseSplit.mxr2 (aP10_19) 22 30
def N10_20 (_ra _rb : ℕ) : ℤ := 0
def aP10_20 (ra rb : ℕ) : ℤ := -(6) * N10_20 ra rb + u10 (798 + rb) + u10 (829 + ra)
def MP10_20 : ℤ := CaseSplit.mxr2 (aP10_20) 28 30

def rhs10 : ℤ := (∑ t ∈ Finset.range n10, w10 t) + 6 * (n10 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn10 : ∀ t, t < n10 → (0 : ℤ) ≤ w10 t := by decide
theorem plt10 : ∀ t, t < n10 → q10 t < 65 := by decide
theorem pfree10_5 : ∀ t, t < n10 → gb5 1 (q10 t) = false := by decide
theorem pfree10_7 : ∀ t, t < n10 → gb7 3 (q10 t) = false := by decide
theorem MSv10_0 : MS10_0 = 62 := by decide +kernel
theorem MSv10_1 : MS10_1 = 213 := by decide +kernel
theorem MSv10_2 : MS10_2 = 1 := by decide +kernel
theorem MSv10_3 : MS10_3 = 1 := by decide +kernel
theorem MSv10_4 : MS10_4 = 1 := by decide +kernel
theorem MSv10_5 : MS10_5 = 1 := by decide +kernel
theorem MSv10_6 : MS10_6 = 1 := by decide +kernel
theorem MPv10_0 : MP10_0 = 0 := by decide +kernel
theorem MPv10_1 : MP10_1 = 0 := by decide +kernel
theorem MPv10_2 : MP10_2 = 0 := by decide +kernel
theorem MPv10_3 : MP10_3 = 0 := by decide +kernel
theorem MPv10_4 : MP10_4 = 0 := by decide +kernel
theorem MPv10_5 : MP10_5 = 0 := by decide +kernel
theorem MPv10_6 : MP10_6 = 0 := by decide +kernel
theorem MPv10_7 : MP10_7 = 0 := by decide +kernel
theorem MPv10_8 : MP10_8 = 0 := by decide +kernel
theorem MPv10_9 : MP10_9 = 0 := by decide +kernel
theorem MPv10_10 : MP10_10 = 0 := by decide +kernel
theorem MPv10_11 : MP10_11 = 0 := by decide +kernel
theorem MPv10_12 : MP10_12 = 0 := by decide +kernel
theorem MPv10_13 : MP10_13 = 0 := by decide +kernel
theorem MPv10_14 : MP10_14 = 0 := by decide +kernel
theorem MPv10_15 : MP10_15 = 0 := by decide +kernel
theorem MPv10_16 : MP10_16 = 0 := by decide +kernel
theorem MPv10_17 : MP10_17 = 0 := by decide +kernel
theorem MPv10_18 : MP10_18 = 0 := by decide +kernel
theorem MPv10_19 : MP10_19 = 0 := by decide +kernel
theorem MPv10_20 : MP10_20 = 32 := by decide +kernel
theorem rhsv10 : rhs10 = 313 := by decide +kernel

/-- **The case-10 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/12.
    (Scaled by the common denominator 12: 312 < 313.) -/
theorem cert10 : MS10_0 + MS10_1 + MS10_2 + MS10_3 + MS10_4 + MS10_5 + MS10_6 + MP10_0 + MP10_1 + MP10_2 + MP10_3 + MP10_4 + MP10_5 + MP10_6 + MP10_7 + MP10_8 + MP10_9 + MP10_10 + MP10_11 + MP10_12 + MP10_13 + MP10_14 + MP10_15 + MP10_16 + MP10_17 + MP10_18 + MP10_19 + MP10_20 < rhs10 := by
  rw [MSv10_0, MSv10_1, MSv10_2, MSv10_3, MSv10_4, MSv10_5, MSv10_6, MPv10_0, MPv10_1, MPv10_2, MPv10_3, MPv10_4, MPv10_5, MPv10_6, MPv10_7, MPv10_8, MPv10_9, MPv10_10, MPv10_11, MPv10_12, MPv10_13, MPv10_14, MPv10_15, MPv10_16, MPv10_17, MPv10_18, MPv10_19, MPv10_20, rhsv10]
  decide

def Dg10 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := (if c10_0 r0 t then 1 else 0) + (if c10_1 r1 t then 1 else 0) + (if c10_2 r2 t then 1 else 0) + (if c10_3 r3 t then 1 else 0) + (if c10_4 r4 t then 1 else 0) + (if c10_5 r5 t then 1 else 0) + (if c10_6 r6 t then 1 else 0)
def Wl10_0 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c10_0 r0 t && c10_1 r1 t then 1 else 0
def Wl10_1 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c10_0 r0 t && c10_2 r2 t then 1 else 0
def Wl10_2 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c10_0 r0 t && c10_3 r3 t then 1 else 0
def Wl10_3 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c10_0 r0 t && c10_4 r4 t then 1 else 0
def Wl10_4 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c10_0 r0 t && c10_5 r5 t then 1 else 0
def Wl10_5 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c10_0 r0 t && c10_6 r6 t then 1 else 0
def Wl10_6 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c10_0 r0 t && c10_1 r1 t && c10_2 r2 t then 1 else 0
def Wl10_7 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c10_0 r0 t && c10_1 r1 t && c10_3 r3 t then 1 else 0
def Wl10_8 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c10_0 r0 t && c10_1 r1 t && c10_4 r4 t then 1 else 0
def Wl10_9 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c10_0 r0 t && c10_1 r1 t && c10_5 r5 t then 1 else 0
def Wl10_10 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c10_0 r0 t && c10_1 r1 t && c10_6 r6 t then 1 else 0
def Wl10_11 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c10_0 r0 t && !c10_1 r1 t && c10_2 r2 t && c10_3 r3 t then 1 else 0
def Wl10_12 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c10_0 r0 t && !c10_1 r1 t && c10_2 r2 t && c10_4 r4 t then 1 else 0
def Wl10_13 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c10_0 r0 t && !c10_1 r1 t && c10_2 r2 t && c10_5 r5 t then 1 else 0
def Wl10_14 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c10_0 r0 t && !c10_1 r1 t && c10_2 r2 t && c10_6 r6 t then 1 else 0
def Wl10_15 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c10_0 r0 t && !c10_1 r1 t && !c10_2 r2 t && c10_3 r3 t && c10_4 r4 t then 1 else 0
def Wl10_16 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c10_0 r0 t && !c10_1 r1 t && !c10_2 r2 t && c10_3 r3 t && c10_5 r5 t then 1 else 0
def Wl10_17 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c10_0 r0 t && !c10_1 r1 t && !c10_2 r2 t && c10_3 r3 t && c10_6 r6 t then 1 else 0
def Wl10_18 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c10_0 r0 t && !c10_1 r1 t && !c10_2 r2 t && !c10_3 r3 t && c10_4 r4 t && c10_5 r5 t then 1 else 0
def Wl10_19 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c10_0 r0 t && !c10_1 r1 t && !c10_2 r2 t && !c10_3 r3 t && c10_4 r4 t && c10_6 r6 t then 1 else 0
def Wl10_20 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c10_0 r0 t && !c10_1 r1 t && !c10_2 r2 t && !c10_3 r3 t && !c10_4 r4 t && c10_5 r5 t && c10_6 r6 t then 1 else 0

/-- **No configuration blocks the whole window in case 10.** -/
theorem nocov10 {r0 r1 r2 r3 r4 r5 r6 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29) (h6 : r6 < 31)
    (hcov : ∀ t, t < n10 → (c10_0 r0 t || c10_1 r1 t || c10_2 r2 t || c10_3 r3 t || c10_4 r4 t || c10_5 r5 t || c10_6 r6 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n10, (1 : ℤ) + (Wl10_0 r0 r1 r2 r3 r4 r5 r6 t + Wl10_1 r0 r1 r2 r3 r4 r5 r6 t + Wl10_2 r0 r1 r2 r3 r4 r5 r6 t + Wl10_3 r0 r1 r2 r3 r4 r5 r6 t + Wl10_4 r0 r1 r2 r3 r4 r5 r6 t + Wl10_5 r0 r1 r2 r3 r4 r5 r6 t + Wl10_6 r0 r1 r2 r3 r4 r5 r6 t + Wl10_7 r0 r1 r2 r3 r4 r5 r6 t + Wl10_8 r0 r1 r2 r3 r4 r5 r6 t + Wl10_9 r0 r1 r2 r3 r4 r5 r6 t + Wl10_10 r0 r1 r2 r3 r4 r5 r6 t + Wl10_11 r0 r1 r2 r3 r4 r5 r6 t + Wl10_12 r0 r1 r2 r3 r4 r5 r6 t + Wl10_13 r0 r1 r2 r3 r4 r5 r6 t + Wl10_14 r0 r1 r2 r3 r4 r5 r6 t + Wl10_15 r0 r1 r2 r3 r4 r5 r6 t + Wl10_16 r0 r1 r2 r3 r4 r5 r6 t + Wl10_17 r0 r1 r2 r3 r4 r5 r6 t + Wl10_18 r0 r1 r2 r3 r4 r5 r6 t + Wl10_19 r0 r1 r2 r3 r4 r5 r6 t + Wl10_20 r0 r1 r2 r3 r4 r5 r6 t) ≤ Dg10 r0 r1 r2 r3 r4 r5 r6 t := by
    intro t ht
    simp only [Wl10_0, Wl10_1, Wl10_2, Wl10_3, Wl10_4, Wl10_5, Wl10_6, Wl10_7, Wl10_8, Wl10_9, Wl10_10, Wl10_11, Wl10_12, Wl10_13, Wl10_14, Wl10_15, Wl10_16, Wl10_17, Wl10_18, Wl10_19, Wl10_20, Dg10]
    exact CaseSplit.lowest7 _ _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n10, (1 : ℤ) ≤ Dg10 r0 r1 r2 r3 r4 r5 r6 t := by
    intro t ht
    simp only [Dg10]
    exact CaseSplit.degpos7 _ _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n10 : ℤ) + ((∑ t ∈ Finset.range n10, Wl10_0 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n10, Wl10_1 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n10, Wl10_2 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n10, Wl10_3 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n10, Wl10_4 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n10, Wl10_5 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n10, Wl10_6 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n10, Wl10_7 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n10, Wl10_8 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n10, Wl10_9 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n10, Wl10_10 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n10, Wl10_11 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n10, Wl10_12 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n10, Wl10_13 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n10, Wl10_14 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n10, Wl10_15 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n10, Wl10_16 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n10, Wl10_17 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n10, Wl10_18 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n10, Wl10_19 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n10, Wl10_20 r0 r1 r2 r3 r4 r5 r6 t)) ≤ ∑ t ∈ Finset.range n10, Dg10 r0 r1 r2 r3 r4 r5 r6 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N10_0 r0 r1 ≤ ∑ t ∈ Finset.range n10, Wl10_0 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N10_0, Wl10_0, le_refl]
  have hn1 : N10_1 r0 r2 ≤ ∑ t ∈ Finset.range n10, Wl10_1 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N10_1, Wl10_1, le_refl]
  have hn2 : N10_2 r0 r3 ≤ ∑ t ∈ Finset.range n10, Wl10_2 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N10_2, Wl10_2, le_refl]
  have hn3 : N10_3 r0 r4 ≤ ∑ t ∈ Finset.range n10, Wl10_3 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N10_3, Wl10_3, le_refl]
  have hn4 : N10_4 r0 r5 ≤ ∑ t ∈ Finset.range n10, Wl10_4 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N10_4, Wl10_4, le_refl]
  have hn5 : N10_5 r0 r6 ≤ ∑ t ∈ Finset.range n10, Wl10_5 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N10_5, Wl10_5, le_refl]
  have hn6 : N10_6 r1 r2 ≤ ∑ t ∈ Finset.range n10, Wl10_6 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n10, Wl10_6 r0 r1 r2 r3 r4 r5 r6 t
        = (if c10_1 r1 t && c10_2 r2 t then (1:ℤ) else 0)
          - (if c10_1 r1 t && c10_2 r2 t && c10_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl10_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n10, Wl10_6 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl10_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n10, Wl10_6 r0 r1 r2 r3 r4 r5 r6 t
        = P10_6 r1 r2 - C10_6 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P10_6, C10_6]
    have hm : C10_6 r1 r2 r0 ≤ M10_6 r1 r2 :=
      CaseSplit.le_mxr (C10_6 r1 r2) 10 r0 (by omega)
    simp only [N10_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N10_7 r1 r3 ≤ ∑ t ∈ Finset.range n10, Wl10_7 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n10, Wl10_7 r0 r1 r2 r3 r4 r5 r6 t
        = (if c10_1 r1 t && c10_3 r3 t then (1:ℤ) else 0)
          - (if c10_1 r1 t && c10_3 r3 t && c10_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl10_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n10, Wl10_7 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl10_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n10, Wl10_7 r0 r1 r2 r3 r4 r5 r6 t
        = P10_7 r1 r3 - C10_7 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P10_7, C10_7]
    have hm : C10_7 r1 r3 r0 ≤ M10_7 r1 r3 :=
      CaseSplit.le_mxr (C10_7 r1 r3) 10 r0 (by omega)
    simp only [N10_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N10_8 r1 r4 ≤ ∑ t ∈ Finset.range n10, Wl10_8 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n10, Wl10_8 r0 r1 r2 r3 r4 r5 r6 t
        = (if c10_1 r1 t && c10_4 r4 t then (1:ℤ) else 0)
          - (if c10_1 r1 t && c10_4 r4 t && c10_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl10_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n10, Wl10_8 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl10_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n10, Wl10_8 r0 r1 r2 r3 r4 r5 r6 t
        = P10_8 r1 r4 - C10_8 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P10_8, C10_8]
    have hm : C10_8 r1 r4 r0 ≤ M10_8 r1 r4 :=
      CaseSplit.le_mxr (C10_8 r1 r4) 10 r0 (by omega)
    simp only [N10_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N10_9 r1 r5 ≤ ∑ t ∈ Finset.range n10, Wl10_9 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n10, Wl10_9 r0 r1 r2 r3 r4 r5 r6 t
        = (if c10_1 r1 t && c10_5 r5 t then (1:ℤ) else 0)
          - (if c10_1 r1 t && c10_5 r5 t && c10_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl10_9]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n10, Wl10_9 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl10_9]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n10, Wl10_9 r0 r1 r2 r3 r4 r5 r6 t
        = P10_9 r1 r5 - C10_9 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P10_9, C10_9]
    have hm : C10_9 r1 r5 r0 ≤ M10_9 r1 r5 :=
      CaseSplit.le_mxr (C10_9 r1 r5) 10 r0 (by omega)
    simp only [N10_9]
    split
    · rw [hL]; omega
    · exact hnn
  have hn10 : N10_10 r1 r6 ≤ ∑ t ∈ Finset.range n10, Wl10_10 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n10, Wl10_10 r0 r1 r2 r3 r4 r5 r6 t
        = (if c10_1 r1 t && c10_6 r6 t then (1:ℤ) else 0)
          - (if c10_1 r1 t && c10_6 r6 t && c10_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl10_10]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n10, Wl10_10 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl10_10]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n10, Wl10_10 r0 r1 r2 r3 r4 r5 r6 t
        = P10_10 r1 r6 - C10_10 r1 r6 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P10_10, C10_10]
    have hm : C10_10 r1 r6 r0 ≤ M10_10 r1 r6 :=
      CaseSplit.le_mxr (C10_10 r1 r6) 10 r0 (by omega)
    simp only [N10_10]
    split
    · rw [hL]; omega
    · exact hnn
  have hn11 : N10_11 r2 r3 ≤ ∑ t ∈ Finset.range n10, Wl10_11 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N10_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl10_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N10_12 r2 r4 ≤ ∑ t ∈ Finset.range n10, Wl10_12 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N10_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl10_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N10_13 r2 r5 ≤ ∑ t ∈ Finset.range n10, Wl10_13 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N10_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl10_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N10_14 r2 r6 ≤ ∑ t ∈ Finset.range n10, Wl10_14 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N10_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl10_14]
    exact CaseSplit.ind_nonneg _
  have hn15 : N10_15 r3 r4 ≤ ∑ t ∈ Finset.range n10, Wl10_15 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N10_15]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl10_15]
    exact CaseSplit.ind_nonneg _
  have hn16 : N10_16 r3 r5 ≤ ∑ t ∈ Finset.range n10, Wl10_16 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N10_16]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl10_16]
    exact CaseSplit.ind_nonneg _
  have hn17 : N10_17 r3 r6 ≤ ∑ t ∈ Finset.range n10, Wl10_17 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N10_17]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl10_17]
    exact CaseSplit.ind_nonneg _
  have hn18 : N10_18 r4 r5 ≤ ∑ t ∈ Finset.range n10, Wl10_18 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N10_18]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl10_18]
    exact CaseSplit.ind_nonneg _
  have hn19 : N10_19 r4 r6 ≤ ∑ t ∈ Finset.range n10, Wl10_19 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N10_19]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl10_19]
    exact CaseSplit.ind_nonneg _
  have hn20 : N10_20 r5 r6 ≤ ∑ t ∈ Finset.range n10, Wl10_20 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N10_20]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl10_20]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n10, (w10 t + 6) * Dg10 r0 r1 r2 r3 r4 r5 r6 t = S10_0 r0 + S10_1 r1 + S10_2 r2 + S10_3 r3 + S10_4 r4 + S10_5 r5 + S10_6 r6 := by
    simp only [S10_0, S10_1, S10_2, S10_3, S10_4, S10_5, S10_6, Dg10, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n10, (w10 t + 6) * Dg10 r0 r1 r2 r3 r4 r5 r6 t
      = (∑ t ∈ Finset.range n10, w10 t * Dg10 r0 r1 r2 r3 r4 r5 r6 t)
        + 6 * (∑ t ∈ Finset.range n10, Dg10 r0 r1 r2 r3 r4 r5 r6 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n10, w10 t)
      ≤ ∑ t ∈ Finset.range n10, w10 t * Dg10 r0 r1 r2 r3 r4 r5 r6 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg10 r0 r1 r2 r3 r4 r5 r6 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w10 t := wnn10 t (Finset.mem_range.mp ht)
    calc w10 t = w10 t * 1 := (mul_one _).symm
      _ ≤ w10 t * Dg10 r0 r1 r2 r3 r4 r5 r6 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS10_0 r0 + aS10_1 r1 + aS10_2 r2 + aS10_3 r3 + aS10_4 r4 + aS10_5 r5 + aS10_6 r6) + (aP10_0 r0 r1 + aP10_1 r0 r2 + aP10_2 r0 r3 + aP10_3 r0 r4 + aP10_4 r0 r5 + aP10_5 r0 r6 + aP10_6 r1 r2 + aP10_7 r1 r3 + aP10_8 r1 r4 + aP10_9 r1 r5 + aP10_10 r1 r6 + aP10_11 r2 r3 + aP10_12 r2 r4 + aP10_13 r2 r5 + aP10_14 r2 r6 + aP10_15 r3 r4 + aP10_16 r3 r5 + aP10_17 r3 r6 + aP10_18 r4 r5 + aP10_19 r4 r6 + aP10_20 r5 r6) = (S10_0 r0 + S10_1 r1 + S10_2 r2 + S10_3 r3 + S10_4 r4 + S10_5 r5 + S10_6 r6) - 6 * (N10_0 r0 r1 + N10_1 r0 r2 + N10_2 r0 r3 + N10_3 r0 r4 + N10_4 r0 r5 + N10_5 r0 r6 + N10_6 r1 r2 + N10_7 r1 r3 + N10_8 r1 r4 + N10_9 r1 r5 + N10_10 r1 r6 + N10_11 r2 r3 + N10_12 r2 r4 + N10_13 r2 r5 + N10_14 r2 r6 + N10_15 r3 r4 + N10_16 r3 r5 + N10_17 r3 r6 + N10_18 r4 r5 + N10_19 r4 r6 + N10_20 r5 r6) := by
    simp only [aS10_0, aS10_1, aS10_2, aS10_3, aS10_4, aS10_5, aS10_6, aP10_0, aP10_1, aP10_2, aP10_3, aP10_4, aP10_5, aP10_6, aP10_7, aP10_8, aP10_9, aP10_10, aP10_11, aP10_12, aP10_13, aP10_14, aP10_15, aP10_16, aP10_17, aP10_18, aP10_19, aP10_20, L10_0, L10_1, L10_2, L10_3, L10_4, L10_5, L10_6]
    ring
  have bS0 : aS10_0 r0 ≤ MS10_0 := CaseSplit.le_mxr (aS10_0) 10 r0 (by omega)
  have bS1 : aS10_1 r1 ≤ MS10_1 := CaseSplit.le_mxr (aS10_1) 12 r1 (by omega)
  have bS2 : aS10_2 r2 ≤ MS10_2 := CaseSplit.le_mxr (aS10_2) 16 r2 (by omega)
  have bS3 : aS10_3 r3 ≤ MS10_3 := CaseSplit.le_mxr (aS10_3) 18 r3 (by omega)
  have bS4 : aS10_4 r4 ≤ MS10_4 := CaseSplit.le_mxr (aS10_4) 22 r4 (by omega)
  have bS5 : aS10_5 r5 ≤ MS10_5 := CaseSplit.le_mxr (aS10_5) 28 r5 (by omega)
  have bS6 : aS10_6 r6 ≤ MS10_6 := CaseSplit.le_mxr (aS10_6) 30 r6 (by omega)
  have bP0 : aP10_0 r0 r1 ≤ MP10_0 := CaseSplit.le_mxr2 (aP10_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP10_1 r0 r2 ≤ MP10_1 := CaseSplit.le_mxr2 (aP10_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP10_2 r0 r3 ≤ MP10_2 := CaseSplit.le_mxr2 (aP10_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP10_3 r0 r4 ≤ MP10_3 := CaseSplit.le_mxr2 (aP10_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP10_4 r0 r5 ≤ MP10_4 := CaseSplit.le_mxr2 (aP10_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP10_5 r0 r6 ≤ MP10_5 := CaseSplit.le_mxr2 (aP10_5) 10 30 r0 r6 (by omega) (by omega)
  have bP6 : aP10_6 r1 r2 ≤ MP10_6 := CaseSplit.le_mxr2 (aP10_6) 12 16 r1 r2 (by omega) (by omega)
  have bP7 : aP10_7 r1 r3 ≤ MP10_7 := CaseSplit.le_mxr2 (aP10_7) 12 18 r1 r3 (by omega) (by omega)
  have bP8 : aP10_8 r1 r4 ≤ MP10_8 := CaseSplit.le_mxr2 (aP10_8) 12 22 r1 r4 (by omega) (by omega)
  have bP9 : aP10_9 r1 r5 ≤ MP10_9 := CaseSplit.le_mxr2 (aP10_9) 12 28 r1 r5 (by omega) (by omega)
  have bP10 : aP10_10 r1 r6 ≤ MP10_10 := CaseSplit.le_mxr2 (aP10_10) 12 30 r1 r6 (by omega) (by omega)
  have bP11 : aP10_11 r2 r3 ≤ MP10_11 := CaseSplit.le_mxr2 (aP10_11) 16 18 r2 r3 (by omega) (by omega)
  have bP12 : aP10_12 r2 r4 ≤ MP10_12 := CaseSplit.le_mxr2 (aP10_12) 16 22 r2 r4 (by omega) (by omega)
  have bP13 : aP10_13 r2 r5 ≤ MP10_13 := CaseSplit.le_mxr2 (aP10_13) 16 28 r2 r5 (by omega) (by omega)
  have bP14 : aP10_14 r2 r6 ≤ MP10_14 := CaseSplit.le_mxr2 (aP10_14) 16 30 r2 r6 (by omega) (by omega)
  have bP15 : aP10_15 r3 r4 ≤ MP10_15 := CaseSplit.le_mxr2 (aP10_15) 18 22 r3 r4 (by omega) (by omega)
  have bP16 : aP10_16 r3 r5 ≤ MP10_16 := CaseSplit.le_mxr2 (aP10_16) 18 28 r3 r5 (by omega) (by omega)
  have bP17 : aP10_17 r3 r6 ≤ MP10_17 := CaseSplit.le_mxr2 (aP10_17) 18 30 r3 r6 (by omega) (by omega)
  have bP18 : aP10_18 r4 r5 ≤ MP10_18 := CaseSplit.le_mxr2 (aP10_18) 22 28 r4 r5 (by omega) (by omega)
  have bP19 : aP10_19 r4 r6 ≤ MP10_19 := CaseSplit.le_mxr2 (aP10_19) 22 30 r4 r6 (by omega) (by omega)
  have bP20 : aP10_20 r5 r6 ≤ MP10_20 := CaseSplit.le_mxr2 (aP10_20) 28 30 r5 r6 (by omega) (by omega)
  have hrhs : rhs10 = (∑ t ∈ Finset.range n10, w10 t) + 6 * (n10 : ℤ) := rfl
  have hc := cert10
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, hn15, hn16, hn17, hn18, hn19, hn20, bS0, bS1, bS2, bS3, bS4, bS5, bS6, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14, bP15, bP16, bP17, bP18, bP19, bP20]

end IncCert31
