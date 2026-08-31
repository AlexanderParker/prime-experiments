/-
INCREMENT-WIDTH CERTIFICATE, step 29->31, case 29 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_29_31.json, which re-derives every number
from the primes alone).

Machine 31, INCREMENT width 65 = F_2(29) + s_min(31) = 55 + 10,
held gears [5, 7] at phases [4, 1].  Free gears [11, 13, 17, 19, 23, 29, 31].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 16.
-/
import IncCert31B

namespace IncCert31

/-! ### case 29: held gears at phases [4, 1] -/

def p29 : List ℕ := [1, 3, 4, 6, 8, 9, 11, 13, 16, 18, 23, 24, 29, 31, 34, 36, 38, 39, 41, 43, 44, 46, 48, 51, 53, 58, 59, 64]
def q29 (t : ℕ) : ℕ := p29.getD t 0
def n29 : ℕ := 28
def yl29 : List ℤ := [1, 0, 8, 16, 2, 7, 13, 0, 3, 3, 0, 0, 2, 6, 8, 11, 8, 10, 16, 13, 2, 13, 9, 6, 8, 5, 3, 0]
def w29 (t : ℕ) : ℤ := yl29.getD t 0
def ul29 : List ℤ := [(-2), (-2), (-2), (-3), (-2), (-2), 0, (-2), 0, (-3), (-2), (-2), (-2), 2, 0, 2, 0, 2, 2, 2, 0, 0, 2, 0, (-7), (-1), 0, (-7), (-7), (-7), (-7), 0, 0, (-1), (-7), (-7), (-7), 0, (-1), (-1), (-7), 1, 1, 0, 0, 0, 0, 1, 7, 0, 0, 0, (-4), 0, (-4), (-4), (-4), 0, 0, (-4), (-4), (-4), 0, 0, 0, (-4), 0, (-4), 0, (-4), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, (-7), (-2), (-7), (-7), (-7), (-7), (-1), (-7), (-7), (-2), (-7), 0, (-7), (-7), (-1), (-7), (-7), (-7), (-7), 0, (-33), (-7), (-7), 7, 1, 0, 0, 1, 7, 0, 2, (-11), 0, 0, (-2), (-7), (-7), (-2), (-7), (-7), (-7), (-7), (-7), (-7), (-2), (-7), (-7), (-7), (-7), 0, (-7), (-7), (-2), (-7), (-7), (-7), (-2), (-7), (-7), (-2), (-7), (-7), (-2), 7, 0, 0, 0, 7, 2, 0, 0, 2, 2, 0, 0, 0, 2, 0, (-12), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, (-2), (-2), (-2), (-2), (-2), 0, 0, (-2), (-2), 0, 58, 47, 58, 58, 25, 40, 58, 54, 58, 58, 58, 47, 58, 58, 58, 58, 36, (-58), (-58), (-58), (-58), (-61), (-58), (-58), (-58), (-58), (-58), (-58), (-58), (-58), 39, 39, 39, 39, 39, 39, 27, 39, 32, 23, 39, 39, 39, 39, 39, 39, 39, 39, 39, (-39), (-39), (-39), (-39), (-39), (-39), (-39), (-39), (-39), (-39), (-39), (-39), (-39), 15, 41, 16, 41, 41, 0, 41, 31, 34, 34, 20, 41, 29, 32, 41, 33, 41, 16, 28, 41, 41, 41, 39, (-41), (-41), (-41), (-41), (-41), (-41), (-41), (-41), (-41), (-41), (-41), (-41), (-41), 50, 50, 50, 21, 38, 50, 29, 50, 29, 50, 50, 50, 50, 39, 37, 50, 38, 50, 50, 50, 50, 33, 50, 50, 50, 50, 29, 38, 45, (-50), (-50), (-50), (-50), (-50), (-50), (-50), (-50), (-50), (-50), (-50), (-50), (-50), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-21), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-9), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), 3, 7, 0, 7, 7, (-3), 7, 7, 0, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0, 7, 7, 7, 7, 0, 7, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 15, 26, 13, 35, 12, 12, 9, 19, 27, 9, 19, 7, 22, 35, 19, 35, 13, 35, 30, 32, 35, 15, 35, 27, 27, 35, 13, 35, 24, 35, 1, 0, 7, 7, 0, 6, 0, (-11), 0, (-21), 7, 0, 5, 0, 0, 7, 0, 7, 4, (-7), 7, 0, 0, 7, 7, (-7), 0, 0, 0]
def u29 (k : ℕ) : ℤ := ul29.getD k 0

def c29_0 (r t : ℕ) : Bool := gb11 r (q29 t)
def c29_1 (r t : ℕ) : Bool := gb13 r (q29 t)
def c29_2 (r t : ℕ) : Bool := gb17 r (q29 t)
def c29_3 (r t : ℕ) : Bool := gb19 r (q29 t)
def c29_4 (r t : ℕ) : Bool := gb23 r (q29 t)
def c29_5 (r t : ℕ) : Bool := gb29 r (q29 t)
def c29_6 (r t : ℕ) : Bool := gb31 r (q29 t)

def S29_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (w29 t + 7) * (if c29_0 r t then 1 else 0)
def S29_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (w29 t + 7) * (if c29_1 r t then 1 else 0)
def S29_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (w29 t + 7) * (if c29_2 r t then 1 else 0)
def S29_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (w29 t + 7) * (if c29_3 r t then 1 else 0)
def S29_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (w29 t + 7) * (if c29_4 r t then 1 else 0)
def S29_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (w29 t + 7) * (if c29_5 r t then 1 else 0)
def S29_6 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (w29 t + 7) * (if c29_6 r t then 1 else 0)

def L29_0 (r : ℕ) : ℤ := u29 (13 + r) + u29 (41 + r) + u29 (71 + r) + u29 (105 + r) + u29 (145 + r) + u29 (187 + r)
def L29_1 (r : ℕ) : ℤ := u29 (0 + r) + u29 (215 + r) + u29 (247 + r) + u29 (283 + r) + u29 (325 + r) + u29 (369 + r)
def L29_2 (r : ℕ) : ℤ := u29 (24 + r) + u29 (198 + r) + u29 (401 + r) + u29 (441 + r) + u29 (487 + r) + u29 (535 + r)
def L29_3 (r : ℕ) : ℤ := u29 (52 + r) + u29 (228 + r) + u29 (382 + r) + u29 (575 + r) + u29 (623 + r) + u29 (673 + r)
def L29_4 (r : ℕ) : ℤ := u29 (82 + r) + u29 (260 + r) + u29 (418 + r) + u29 (552 + r) + u29 (721 + r) + u29 (775 + r)
def L29_5 (r : ℕ) : ℤ := u29 (116 + r) + u29 (296 + r) + u29 (458 + r) + u29 (594 + r) + u29 (692 + r) + u29 (829 + r)
def L29_6 (r : ℕ) : ℤ := u29 (156 + r) + u29 (338 + r) + u29 (504 + r) + u29 (642 + r) + u29 (744 + r) + u29 (798 + r)

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
def aS29_6 (r : ℕ) : ℤ := S29_6 r - L29_6 r
def MS29_6 : ℤ := CaseSplit.mxr (aS29_6) 30

def N29_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_0 ra t && c29_1 rb t then 1 else 0)
def aP29_0 (ra rb : ℕ) : ℤ := -(7) * N29_0 ra rb + u29 (0 + rb) + u29 (13 + ra)
def MP29_0 : ℤ := CaseSplit.mxr2 (aP29_0) 10 12
def N29_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_0 ra t && c29_2 rb t then 1 else 0)
def aP29_1 (ra rb : ℕ) : ℤ := -(7) * N29_1 ra rb + u29 (24 + rb) + u29 (41 + ra)
def MP29_1 : ℤ := CaseSplit.mxr2 (aP29_1) 10 16
def N29_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_0 ra t && c29_3 rb t then 1 else 0)
def aP29_2 (ra rb : ℕ) : ℤ := -(7) * N29_2 ra rb + u29 (52 + rb) + u29 (71 + ra)
def MP29_2 : ℤ := CaseSplit.mxr2 (aP29_2) 10 18
def N29_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_0 ra t && c29_4 rb t then 1 else 0)
def aP29_3 (ra rb : ℕ) : ℤ := -(7) * N29_3 ra rb + u29 (82 + rb) + u29 (105 + ra)
def MP29_3 : ℤ := CaseSplit.mxr2 (aP29_3) 10 22
def N29_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_0 ra t && c29_5 rb t then 1 else 0)
def aP29_4 (ra rb : ℕ) : ℤ := -(7) * N29_4 ra rb + u29 (116 + rb) + u29 (145 + ra)
def MP29_4 : ℤ := CaseSplit.mxr2 (aP29_4) 10 28
def N29_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_0 ra t && c29_6 rb t then 1 else 0)
def aP29_5 (ra rb : ℕ) : ℤ := -(7) * N29_5 ra rb + u29 (156 + rb) + u29 (187 + ra)
def MP29_5 : ℤ := CaseSplit.mxr2 (aP29_5) 10 30
def P29_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_1 ra t && c29_2 rb t then 1 else 0)
def C29_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_1 ra t && c29_2 rb t && c29_0 s t then 1 else 0)
def M29_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C29_6 ra rb) 10
def E29_6 : List ℕ := [7, 13, 18, 29, 54, 65, 93, 99, 102, 108, 138, 144, 160, 166, 172, 183]
def N29_6 (ra rb : ℕ) : ℤ := if E29_6.contains (ra * 17 + rb) = true then P29_6 ra rb - M29_6 ra rb else 0
def aP29_6 (ra rb : ℕ) : ℤ := -(7) * N29_6 ra rb + u29 (198 + rb) + u29 (215 + ra)
def MP29_6 : ℤ := CaseSplit.mxr2 (aP29_6) 12 16
def P29_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_1 ra t && c29_3 rb t then 1 else 0)
def C29_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_1 ra t && c29_3 rb t && c29_0 s t then 1 else 0)
def M29_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C29_7 ra rb) 10
def E29_7 : List ℕ := [11, 37, 60, 66, 87, 90, 113, 118, 124, 152, 158, 166, 194, 200, 228, 231, 234, 237]
def N29_7 (ra rb : ℕ) : ℤ := if E29_7.contains (ra * 19 + rb) = true then P29_7 ra rb - M29_7 ra rb else 0
def aP29_7 (ra rb : ℕ) : ℤ := -(7) * N29_7 ra rb + u29 (228 + rb) + u29 (247 + ra)
def MP29_7 : ℤ := CaseSplit.mxr2 (aP29_7) 12 18
def P29_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_1 ra t && c29_4 rb t then 1 else 0)
def C29_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_1 ra t && c29_4 rb t && c29_0 s t then 1 else 0)
def M29_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C29_8 ra rb) 10
def E29_8 : List ℕ := [277]
def N29_8 (ra rb : ℕ) : ℤ := if E29_8.contains (ra * 23 + rb) = true then P29_8 ra rb - M29_8 ra rb else 0
def aP29_8 (ra rb : ℕ) : ℤ := -(7) * N29_8 ra rb + u29 (260 + rb) + u29 (283 + ra)
def MP29_8 : ℤ := CaseSplit.mxr2 (aP29_8) 12 22
def P29_9 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_1 ra t && c29_5 rb t then 1 else 0)
def C29_9 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_1 ra t && c29_5 rb t && c29_0 s t then 1 else 0)
def M29_9 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C29_9 ra rb) 10
def E29_9 : List ℕ := [73, 189, 223, 339]
def N29_9 (ra rb : ℕ) : ℤ := if E29_9.contains (ra * 29 + rb) = true then P29_9 ra rb - M29_9 ra rb else 0
def aP29_9 (ra rb : ℕ) : ℤ := -(7) * N29_9 ra rb + u29 (296 + rb) + u29 (325 + ra)
def MP29_9 : ℤ := CaseSplit.mxr2 (aP29_9) 12 28
def P29_10 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_1 ra t && c29_6 rb t then 1 else 0)
def C29_10 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_1 ra t && c29_6 rb t && c29_0 s t then 1 else 0)
def M29_10 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C29_10 ra rb) 10
def E29_10 : List ℕ := [35, 185, 309, 314]
def N29_10 (ra rb : ℕ) : ℤ := if E29_10.contains (ra * 31 + rb) = true then P29_10 ra rb - M29_10 ra rb else 0
def aP29_10 (ra rb : ℕ) : ℤ := -(7) * N29_10 ra rb + u29 (338 + rb) + u29 (369 + ra)
def MP29_10 : ℤ := CaseSplit.mxr2 (aP29_10) 12 30
def N29_11 (_ra _rb : ℕ) : ℤ := 0
def aP29_11 (ra rb : ℕ) : ℤ := -(7) * N29_11 ra rb + u29 (382 + rb) + u29 (401 + ra)
def MP29_11 : ℤ := CaseSplit.mxr2 (aP29_11) 16 18
def N29_12 (_ra _rb : ℕ) : ℤ := 0
def aP29_12 (ra rb : ℕ) : ℤ := -(7) * N29_12 ra rb + u29 (418 + rb) + u29 (441 + ra)
def MP29_12 : ℤ := CaseSplit.mxr2 (aP29_12) 16 22
def N29_13 (_ra _rb : ℕ) : ℤ := 0
def aP29_13 (ra rb : ℕ) : ℤ := -(7) * N29_13 ra rb + u29 (458 + rb) + u29 (487 + ra)
def MP29_13 : ℤ := CaseSplit.mxr2 (aP29_13) 16 28
def N29_14 (_ra _rb : ℕ) : ℤ := 0
def aP29_14 (ra rb : ℕ) : ℤ := -(7) * N29_14 ra rb + u29 (504 + rb) + u29 (535 + ra)
def MP29_14 : ℤ := CaseSplit.mxr2 (aP29_14) 16 30
def N29_15 (_ra _rb : ℕ) : ℤ := 0
def aP29_15 (ra rb : ℕ) : ℤ := -(7) * N29_15 ra rb + u29 (552 + rb) + u29 (575 + ra)
def MP29_15 : ℤ := CaseSplit.mxr2 (aP29_15) 18 22
def N29_16 (_ra _rb : ℕ) : ℤ := 0
def aP29_16 (ra rb : ℕ) : ℤ := -(7) * N29_16 ra rb + u29 (594 + rb) + u29 (623 + ra)
def MP29_16 : ℤ := CaseSplit.mxr2 (aP29_16) 18 28
def N29_17 (_ra _rb : ℕ) : ℤ := 0
def aP29_17 (ra rb : ℕ) : ℤ := -(7) * N29_17 ra rb + u29 (642 + rb) + u29 (673 + ra)
def MP29_17 : ℤ := CaseSplit.mxr2 (aP29_17) 18 30
def N29_18 (_ra _rb : ℕ) : ℤ := 0
def aP29_18 (ra rb : ℕ) : ℤ := -(7) * N29_18 ra rb + u29 (692 + rb) + u29 (721 + ra)
def MP29_18 : ℤ := CaseSplit.mxr2 (aP29_18) 22 28
def N29_19 (_ra _rb : ℕ) : ℤ := 0
def aP29_19 (ra rb : ℕ) : ℤ := -(7) * N29_19 ra rb + u29 (744 + rb) + u29 (775 + ra)
def MP29_19 : ℤ := CaseSplit.mxr2 (aP29_19) 22 30
def N29_20 (_ra _rb : ℕ) : ℤ := 0
def aP29_20 (ra rb : ℕ) : ℤ := -(7) * N29_20 ra rb + u29 (798 + rb) + u29 (829 + ra)
def MP29_20 : ℤ := CaseSplit.mxr2 (aP29_20) 28 30

def rhs29 : ℤ := (∑ t ∈ Finset.range n29, w29 t) + 7 * (n29 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn29 : ∀ t, t < n29 → (0 : ℤ) ≤ w29 t := by decide
theorem plt29 : ∀ t, t < n29 → q29 t < 65 := by decide
theorem pfree29_5 : ∀ t, t < n29 → gb5 4 (q29 t) = false := by decide
theorem pfree29_7 : ∀ t, t < n29 → gb7 1 (q29 t) = false := by decide
theorem MSv29_0 : MS29_0 = 66 := by decide +kernel
theorem MSv29_1 : MS29_1 = 251 := by decide +kernel
theorem MSv29_2 : MS29_2 = 2 := by decide +kernel
theorem MSv29_3 : MS29_3 = 2 := by decide +kernel
theorem MSv29_4 : MS29_4 = 1 := by decide +kernel
theorem MSv29_5 : MS29_5 = 2 := by decide +kernel
theorem MSv29_6 : MS29_6 = 1 := by decide +kernel
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
theorem MPv29_14 : MP29_14 = 0 := by decide +kernel
theorem MPv29_15 : MP29_15 = 0 := by decide +kernel
theorem MPv29_16 : MP29_16 = 0 := by decide +kernel
theorem MPv29_17 : MP29_17 = 0 := by decide +kernel
theorem MPv29_18 : MP29_18 = 0 := by decide +kernel
theorem MPv29_19 : MP29_19 = 0 := by decide +kernel
theorem MPv29_20 : MP29_20 = 42 := by decide +kernel
theorem rhsv29 : rhs29 = 369 := by decide +kernel

/-- **The case-29 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 2/16.
    (Scaled by the common denominator 16: 367 < 369.) -/
theorem cert29 : MS29_0 + MS29_1 + MS29_2 + MS29_3 + MS29_4 + MS29_5 + MS29_6 + MP29_0 + MP29_1 + MP29_2 + MP29_3 + MP29_4 + MP29_5 + MP29_6 + MP29_7 + MP29_8 + MP29_9 + MP29_10 + MP29_11 + MP29_12 + MP29_13 + MP29_14 + MP29_15 + MP29_16 + MP29_17 + MP29_18 + MP29_19 + MP29_20 < rhs29 := by
  rw [MSv29_0, MSv29_1, MSv29_2, MSv29_3, MSv29_4, MSv29_5, MSv29_6, MPv29_0, MPv29_1, MPv29_2, MPv29_3, MPv29_4, MPv29_5, MPv29_6, MPv29_7, MPv29_8, MPv29_9, MPv29_10, MPv29_11, MPv29_12, MPv29_13, MPv29_14, MPv29_15, MPv29_16, MPv29_17, MPv29_18, MPv29_19, MPv29_20, rhsv29]
  decide

def Dg29 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := (if c29_0 r0 t then 1 else 0) + (if c29_1 r1 t then 1 else 0) + (if c29_2 r2 t then 1 else 0) + (if c29_3 r3 t then 1 else 0) + (if c29_4 r4 t then 1 else 0) + (if c29_5 r5 t then 1 else 0) + (if c29_6 r6 t then 1 else 0)
def Wl29_0 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c29_0 r0 t && c29_1 r1 t then 1 else 0
def Wl29_1 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c29_0 r0 t && c29_2 r2 t then 1 else 0
def Wl29_2 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c29_0 r0 t && c29_3 r3 t then 1 else 0
def Wl29_3 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c29_0 r0 t && c29_4 r4 t then 1 else 0
def Wl29_4 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c29_0 r0 t && c29_5 r5 t then 1 else 0
def Wl29_5 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c29_0 r0 t && c29_6 r6 t then 1 else 0
def Wl29_6 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c29_0 r0 t && c29_1 r1 t && c29_2 r2 t then 1 else 0
def Wl29_7 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c29_0 r0 t && c29_1 r1 t && c29_3 r3 t then 1 else 0
def Wl29_8 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c29_0 r0 t && c29_1 r1 t && c29_4 r4 t then 1 else 0
def Wl29_9 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c29_0 r0 t && c29_1 r1 t && c29_5 r5 t then 1 else 0
def Wl29_10 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c29_0 r0 t && c29_1 r1 t && c29_6 r6 t then 1 else 0
def Wl29_11 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c29_0 r0 t && !c29_1 r1 t && c29_2 r2 t && c29_3 r3 t then 1 else 0
def Wl29_12 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c29_0 r0 t && !c29_1 r1 t && c29_2 r2 t && c29_4 r4 t then 1 else 0
def Wl29_13 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c29_0 r0 t && !c29_1 r1 t && c29_2 r2 t && c29_5 r5 t then 1 else 0
def Wl29_14 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c29_0 r0 t && !c29_1 r1 t && c29_2 r2 t && c29_6 r6 t then 1 else 0
def Wl29_15 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c29_0 r0 t && !c29_1 r1 t && !c29_2 r2 t && c29_3 r3 t && c29_4 r4 t then 1 else 0
def Wl29_16 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c29_0 r0 t && !c29_1 r1 t && !c29_2 r2 t && c29_3 r3 t && c29_5 r5 t then 1 else 0
def Wl29_17 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c29_0 r0 t && !c29_1 r1 t && !c29_2 r2 t && c29_3 r3 t && c29_6 r6 t then 1 else 0
def Wl29_18 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c29_0 r0 t && !c29_1 r1 t && !c29_2 r2 t && !c29_3 r3 t && c29_4 r4 t && c29_5 r5 t then 1 else 0
def Wl29_19 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c29_0 r0 t && !c29_1 r1 t && !c29_2 r2 t && !c29_3 r3 t && c29_4 r4 t && c29_6 r6 t then 1 else 0
def Wl29_20 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c29_0 r0 t && !c29_1 r1 t && !c29_2 r2 t && !c29_3 r3 t && !c29_4 r4 t && c29_5 r5 t && c29_6 r6 t then 1 else 0

/-- **No configuration blocks the whole window in case 29.** -/
theorem nocov29 {r0 r1 r2 r3 r4 r5 r6 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29) (h6 : r6 < 31)
    (hcov : ∀ t, t < n29 → (c29_0 r0 t || c29_1 r1 t || c29_2 r2 t || c29_3 r3 t || c29_4 r4 t || c29_5 r5 t || c29_6 r6 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n29, (1 : ℤ) + (Wl29_0 r0 r1 r2 r3 r4 r5 r6 t + Wl29_1 r0 r1 r2 r3 r4 r5 r6 t + Wl29_2 r0 r1 r2 r3 r4 r5 r6 t + Wl29_3 r0 r1 r2 r3 r4 r5 r6 t + Wl29_4 r0 r1 r2 r3 r4 r5 r6 t + Wl29_5 r0 r1 r2 r3 r4 r5 r6 t + Wl29_6 r0 r1 r2 r3 r4 r5 r6 t + Wl29_7 r0 r1 r2 r3 r4 r5 r6 t + Wl29_8 r0 r1 r2 r3 r4 r5 r6 t + Wl29_9 r0 r1 r2 r3 r4 r5 r6 t + Wl29_10 r0 r1 r2 r3 r4 r5 r6 t + Wl29_11 r0 r1 r2 r3 r4 r5 r6 t + Wl29_12 r0 r1 r2 r3 r4 r5 r6 t + Wl29_13 r0 r1 r2 r3 r4 r5 r6 t + Wl29_14 r0 r1 r2 r3 r4 r5 r6 t + Wl29_15 r0 r1 r2 r3 r4 r5 r6 t + Wl29_16 r0 r1 r2 r3 r4 r5 r6 t + Wl29_17 r0 r1 r2 r3 r4 r5 r6 t + Wl29_18 r0 r1 r2 r3 r4 r5 r6 t + Wl29_19 r0 r1 r2 r3 r4 r5 r6 t + Wl29_20 r0 r1 r2 r3 r4 r5 r6 t) ≤ Dg29 r0 r1 r2 r3 r4 r5 r6 t := by
    intro t ht
    simp only [Wl29_0, Wl29_1, Wl29_2, Wl29_3, Wl29_4, Wl29_5, Wl29_6, Wl29_7, Wl29_8, Wl29_9, Wl29_10, Wl29_11, Wl29_12, Wl29_13, Wl29_14, Wl29_15, Wl29_16, Wl29_17, Wl29_18, Wl29_19, Wl29_20, Dg29]
    exact CaseSplit.lowest7 _ _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n29, (1 : ℤ) ≤ Dg29 r0 r1 r2 r3 r4 r5 r6 t := by
    intro t ht
    simp only [Dg29]
    exact CaseSplit.degpos7 _ _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n29 : ℤ) + ((∑ t ∈ Finset.range n29, Wl29_0 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n29, Wl29_1 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n29, Wl29_2 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n29, Wl29_3 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n29, Wl29_4 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n29, Wl29_5 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n29, Wl29_6 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n29, Wl29_7 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n29, Wl29_8 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n29, Wl29_9 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n29, Wl29_10 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n29, Wl29_11 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n29, Wl29_12 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n29, Wl29_13 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n29, Wl29_14 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n29, Wl29_15 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n29, Wl29_16 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n29, Wl29_17 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n29, Wl29_18 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n29, Wl29_19 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n29, Wl29_20 r0 r1 r2 r3 r4 r5 r6 t)) ≤ ∑ t ∈ Finset.range n29, Dg29 r0 r1 r2 r3 r4 r5 r6 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N29_0 r0 r1 ≤ ∑ t ∈ Finset.range n29, Wl29_0 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N29_0, Wl29_0, le_refl]
  have hn1 : N29_1 r0 r2 ≤ ∑ t ∈ Finset.range n29, Wl29_1 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N29_1, Wl29_1, le_refl]
  have hn2 : N29_2 r0 r3 ≤ ∑ t ∈ Finset.range n29, Wl29_2 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N29_2, Wl29_2, le_refl]
  have hn3 : N29_3 r0 r4 ≤ ∑ t ∈ Finset.range n29, Wl29_3 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N29_3, Wl29_3, le_refl]
  have hn4 : N29_4 r0 r5 ≤ ∑ t ∈ Finset.range n29, Wl29_4 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N29_4, Wl29_4, le_refl]
  have hn5 : N29_5 r0 r6 ≤ ∑ t ∈ Finset.range n29, Wl29_5 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N29_5, Wl29_5, le_refl]
  have hn6 : N29_6 r1 r2 ≤ ∑ t ∈ Finset.range n29, Wl29_6 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n29, Wl29_6 r0 r1 r2 r3 r4 r5 r6 t
        = (if c29_1 r1 t && c29_2 r2 t then (1:ℤ) else 0)
          - (if c29_1 r1 t && c29_2 r2 t && c29_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl29_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n29, Wl29_6 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl29_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n29, Wl29_6 r0 r1 r2 r3 r4 r5 r6 t
        = P29_6 r1 r2 - C29_6 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P29_6, C29_6]
    have hm : C29_6 r1 r2 r0 ≤ M29_6 r1 r2 :=
      CaseSplit.le_mxr (C29_6 r1 r2) 10 r0 (by omega)
    simp only [N29_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N29_7 r1 r3 ≤ ∑ t ∈ Finset.range n29, Wl29_7 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n29, Wl29_7 r0 r1 r2 r3 r4 r5 r6 t
        = (if c29_1 r1 t && c29_3 r3 t then (1:ℤ) else 0)
          - (if c29_1 r1 t && c29_3 r3 t && c29_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl29_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n29, Wl29_7 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl29_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n29, Wl29_7 r0 r1 r2 r3 r4 r5 r6 t
        = P29_7 r1 r3 - C29_7 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P29_7, C29_7]
    have hm : C29_7 r1 r3 r0 ≤ M29_7 r1 r3 :=
      CaseSplit.le_mxr (C29_7 r1 r3) 10 r0 (by omega)
    simp only [N29_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N29_8 r1 r4 ≤ ∑ t ∈ Finset.range n29, Wl29_8 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n29, Wl29_8 r0 r1 r2 r3 r4 r5 r6 t
        = (if c29_1 r1 t && c29_4 r4 t then (1:ℤ) else 0)
          - (if c29_1 r1 t && c29_4 r4 t && c29_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl29_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n29, Wl29_8 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl29_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n29, Wl29_8 r0 r1 r2 r3 r4 r5 r6 t
        = P29_8 r1 r4 - C29_8 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P29_8, C29_8]
    have hm : C29_8 r1 r4 r0 ≤ M29_8 r1 r4 :=
      CaseSplit.le_mxr (C29_8 r1 r4) 10 r0 (by omega)
    simp only [N29_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N29_9 r1 r5 ≤ ∑ t ∈ Finset.range n29, Wl29_9 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n29, Wl29_9 r0 r1 r2 r3 r4 r5 r6 t
        = (if c29_1 r1 t && c29_5 r5 t then (1:ℤ) else 0)
          - (if c29_1 r1 t && c29_5 r5 t && c29_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl29_9]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n29, Wl29_9 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl29_9]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n29, Wl29_9 r0 r1 r2 r3 r4 r5 r6 t
        = P29_9 r1 r5 - C29_9 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P29_9, C29_9]
    have hm : C29_9 r1 r5 r0 ≤ M29_9 r1 r5 :=
      CaseSplit.le_mxr (C29_9 r1 r5) 10 r0 (by omega)
    simp only [N29_9]
    split
    · rw [hL]; omega
    · exact hnn
  have hn10 : N29_10 r1 r6 ≤ ∑ t ∈ Finset.range n29, Wl29_10 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n29, Wl29_10 r0 r1 r2 r3 r4 r5 r6 t
        = (if c29_1 r1 t && c29_6 r6 t then (1:ℤ) else 0)
          - (if c29_1 r1 t && c29_6 r6 t && c29_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl29_10]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n29, Wl29_10 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl29_10]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n29, Wl29_10 r0 r1 r2 r3 r4 r5 r6 t
        = P29_10 r1 r6 - C29_10 r1 r6 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P29_10, C29_10]
    have hm : C29_10 r1 r6 r0 ≤ M29_10 r1 r6 :=
      CaseSplit.le_mxr (C29_10 r1 r6) 10 r0 (by omega)
    simp only [N29_10]
    split
    · rw [hL]; omega
    · exact hnn
  have hn11 : N29_11 r2 r3 ≤ ∑ t ∈ Finset.range n29, Wl29_11 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N29_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl29_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N29_12 r2 r4 ≤ ∑ t ∈ Finset.range n29, Wl29_12 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N29_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl29_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N29_13 r2 r5 ≤ ∑ t ∈ Finset.range n29, Wl29_13 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N29_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl29_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N29_14 r2 r6 ≤ ∑ t ∈ Finset.range n29, Wl29_14 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N29_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl29_14]
    exact CaseSplit.ind_nonneg _
  have hn15 : N29_15 r3 r4 ≤ ∑ t ∈ Finset.range n29, Wl29_15 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N29_15]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl29_15]
    exact CaseSplit.ind_nonneg _
  have hn16 : N29_16 r3 r5 ≤ ∑ t ∈ Finset.range n29, Wl29_16 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N29_16]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl29_16]
    exact CaseSplit.ind_nonneg _
  have hn17 : N29_17 r3 r6 ≤ ∑ t ∈ Finset.range n29, Wl29_17 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N29_17]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl29_17]
    exact CaseSplit.ind_nonneg _
  have hn18 : N29_18 r4 r5 ≤ ∑ t ∈ Finset.range n29, Wl29_18 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N29_18]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl29_18]
    exact CaseSplit.ind_nonneg _
  have hn19 : N29_19 r4 r6 ≤ ∑ t ∈ Finset.range n29, Wl29_19 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N29_19]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl29_19]
    exact CaseSplit.ind_nonneg _
  have hn20 : N29_20 r5 r6 ≤ ∑ t ∈ Finset.range n29, Wl29_20 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N29_20]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl29_20]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n29, (w29 t + 7) * Dg29 r0 r1 r2 r3 r4 r5 r6 t = S29_0 r0 + S29_1 r1 + S29_2 r2 + S29_3 r3 + S29_4 r4 + S29_5 r5 + S29_6 r6 := by
    simp only [S29_0, S29_1, S29_2, S29_3, S29_4, S29_5, S29_6, Dg29, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n29, (w29 t + 7) * Dg29 r0 r1 r2 r3 r4 r5 r6 t
      = (∑ t ∈ Finset.range n29, w29 t * Dg29 r0 r1 r2 r3 r4 r5 r6 t)
        + 7 * (∑ t ∈ Finset.range n29, Dg29 r0 r1 r2 r3 r4 r5 r6 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n29, w29 t)
      ≤ ∑ t ∈ Finset.range n29, w29 t * Dg29 r0 r1 r2 r3 r4 r5 r6 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg29 r0 r1 r2 r3 r4 r5 r6 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w29 t := wnn29 t (Finset.mem_range.mp ht)
    calc w29 t = w29 t * 1 := (mul_one _).symm
      _ ≤ w29 t * Dg29 r0 r1 r2 r3 r4 r5 r6 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS29_0 r0 + aS29_1 r1 + aS29_2 r2 + aS29_3 r3 + aS29_4 r4 + aS29_5 r5 + aS29_6 r6) + (aP29_0 r0 r1 + aP29_1 r0 r2 + aP29_2 r0 r3 + aP29_3 r0 r4 + aP29_4 r0 r5 + aP29_5 r0 r6 + aP29_6 r1 r2 + aP29_7 r1 r3 + aP29_8 r1 r4 + aP29_9 r1 r5 + aP29_10 r1 r6 + aP29_11 r2 r3 + aP29_12 r2 r4 + aP29_13 r2 r5 + aP29_14 r2 r6 + aP29_15 r3 r4 + aP29_16 r3 r5 + aP29_17 r3 r6 + aP29_18 r4 r5 + aP29_19 r4 r6 + aP29_20 r5 r6) = (S29_0 r0 + S29_1 r1 + S29_2 r2 + S29_3 r3 + S29_4 r4 + S29_5 r5 + S29_6 r6) - 7 * (N29_0 r0 r1 + N29_1 r0 r2 + N29_2 r0 r3 + N29_3 r0 r4 + N29_4 r0 r5 + N29_5 r0 r6 + N29_6 r1 r2 + N29_7 r1 r3 + N29_8 r1 r4 + N29_9 r1 r5 + N29_10 r1 r6 + N29_11 r2 r3 + N29_12 r2 r4 + N29_13 r2 r5 + N29_14 r2 r6 + N29_15 r3 r4 + N29_16 r3 r5 + N29_17 r3 r6 + N29_18 r4 r5 + N29_19 r4 r6 + N29_20 r5 r6) := by
    simp only [aS29_0, aS29_1, aS29_2, aS29_3, aS29_4, aS29_5, aS29_6, aP29_0, aP29_1, aP29_2, aP29_3, aP29_4, aP29_5, aP29_6, aP29_7, aP29_8, aP29_9, aP29_10, aP29_11, aP29_12, aP29_13, aP29_14, aP29_15, aP29_16, aP29_17, aP29_18, aP29_19, aP29_20, L29_0, L29_1, L29_2, L29_3, L29_4, L29_5, L29_6]
    ring
  have bS0 : aS29_0 r0 ≤ MS29_0 := CaseSplit.le_mxr (aS29_0) 10 r0 (by omega)
  have bS1 : aS29_1 r1 ≤ MS29_1 := CaseSplit.le_mxr (aS29_1) 12 r1 (by omega)
  have bS2 : aS29_2 r2 ≤ MS29_2 := CaseSplit.le_mxr (aS29_2) 16 r2 (by omega)
  have bS3 : aS29_3 r3 ≤ MS29_3 := CaseSplit.le_mxr (aS29_3) 18 r3 (by omega)
  have bS4 : aS29_4 r4 ≤ MS29_4 := CaseSplit.le_mxr (aS29_4) 22 r4 (by omega)
  have bS5 : aS29_5 r5 ≤ MS29_5 := CaseSplit.le_mxr (aS29_5) 28 r5 (by omega)
  have bS6 : aS29_6 r6 ≤ MS29_6 := CaseSplit.le_mxr (aS29_6) 30 r6 (by omega)
  have bP0 : aP29_0 r0 r1 ≤ MP29_0 := CaseSplit.le_mxr2 (aP29_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP29_1 r0 r2 ≤ MP29_1 := CaseSplit.le_mxr2 (aP29_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP29_2 r0 r3 ≤ MP29_2 := CaseSplit.le_mxr2 (aP29_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP29_3 r0 r4 ≤ MP29_3 := CaseSplit.le_mxr2 (aP29_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP29_4 r0 r5 ≤ MP29_4 := CaseSplit.le_mxr2 (aP29_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP29_5 r0 r6 ≤ MP29_5 := CaseSplit.le_mxr2 (aP29_5) 10 30 r0 r6 (by omega) (by omega)
  have bP6 : aP29_6 r1 r2 ≤ MP29_6 := CaseSplit.le_mxr2 (aP29_6) 12 16 r1 r2 (by omega) (by omega)
  have bP7 : aP29_7 r1 r3 ≤ MP29_7 := CaseSplit.le_mxr2 (aP29_7) 12 18 r1 r3 (by omega) (by omega)
  have bP8 : aP29_8 r1 r4 ≤ MP29_8 := CaseSplit.le_mxr2 (aP29_8) 12 22 r1 r4 (by omega) (by omega)
  have bP9 : aP29_9 r1 r5 ≤ MP29_9 := CaseSplit.le_mxr2 (aP29_9) 12 28 r1 r5 (by omega) (by omega)
  have bP10 : aP29_10 r1 r6 ≤ MP29_10 := CaseSplit.le_mxr2 (aP29_10) 12 30 r1 r6 (by omega) (by omega)
  have bP11 : aP29_11 r2 r3 ≤ MP29_11 := CaseSplit.le_mxr2 (aP29_11) 16 18 r2 r3 (by omega) (by omega)
  have bP12 : aP29_12 r2 r4 ≤ MP29_12 := CaseSplit.le_mxr2 (aP29_12) 16 22 r2 r4 (by omega) (by omega)
  have bP13 : aP29_13 r2 r5 ≤ MP29_13 := CaseSplit.le_mxr2 (aP29_13) 16 28 r2 r5 (by omega) (by omega)
  have bP14 : aP29_14 r2 r6 ≤ MP29_14 := CaseSplit.le_mxr2 (aP29_14) 16 30 r2 r6 (by omega) (by omega)
  have bP15 : aP29_15 r3 r4 ≤ MP29_15 := CaseSplit.le_mxr2 (aP29_15) 18 22 r3 r4 (by omega) (by omega)
  have bP16 : aP29_16 r3 r5 ≤ MP29_16 := CaseSplit.le_mxr2 (aP29_16) 18 28 r3 r5 (by omega) (by omega)
  have bP17 : aP29_17 r3 r6 ≤ MP29_17 := CaseSplit.le_mxr2 (aP29_17) 18 30 r3 r6 (by omega) (by omega)
  have bP18 : aP29_18 r4 r5 ≤ MP29_18 := CaseSplit.le_mxr2 (aP29_18) 22 28 r4 r5 (by omega) (by omega)
  have bP19 : aP29_19 r4 r6 ≤ MP29_19 := CaseSplit.le_mxr2 (aP29_19) 22 30 r4 r6 (by omega) (by omega)
  have bP20 : aP29_20 r5 r6 ≤ MP29_20 := CaseSplit.le_mxr2 (aP29_20) 28 30 r5 r6 (by omega) (by omega)
  have hrhs : rhs29 = (∑ t ∈ Finset.range n29, w29 t) + 7 * (n29 : ℤ) := rfl
  have hc := cert29
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, hn15, hn16, hn17, hn18, hn19, hn20, bS0, bS1, bS2, bS3, bS4, bS5, bS6, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14, bP15, bP16, bP17, bP18, bP19, bP20]

end IncCert31
