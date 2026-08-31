/-
INCREMENT-WIDTH CERTIFICATE, step 29->31, case 26 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_29_31.json, which re-derives every number
from the primes alone).

Machine 31, INCREMENT width 65 = F_2(29) + s_min(31) = 55 + 10,
held gears [5, 7] at phases [3, 5].  Free gears [11, 13, 17, 19, 23, 29, 31].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 13.
-/
import IncCert31B

namespace IncCert31

/-! ### case 26: held gears at phases [3, 5] -/

def p26 : List ℕ := [0, 2, 4, 5, 7, 9, 12, 14, 19, 20, 25, 27, 30, 32, 34, 35, 37, 39, 40, 42, 44, 47, 49, 54, 55, 60, 62]
def q26 (t : ℕ) : ℕ := p26.getD t 0
def n26 : ℕ := 27
def yl26 : List ℤ := [6, 10, 0, 5, 10, 1, 5, 6, 2, 0, 2, 4, 7, 8, 8, 6, 13, 11, 4, 8, 6, 4, 4, 4, 5, 0, 0]
def w26 (t : ℕ) : ℤ := yl26.getD t 0
def ul26 : List ℤ := [(-1), 0, (-1), (-1), (-1), (-10), (-1), (-1), (-1), (-1), 5, (-1), 4, (-5), (-5), 1, (-5), 1, (-5), 1, (-5), 1, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0, (-5), (-5), (-5), (-5), 0, (-5), (-5), (-5), (-5), 0, (-5), 0, (-5), (-5), (-5), (-5), (-5), (-5), (-5), (-1), 0, 0, (-5), (-5), 0, (-1), (-1), 0, (-5), 0, 0, 5, 0, 0, (-4), 5, 0, 1, 0, 0, 0, 0, (-1), (-1), (-1), 0, (-1), 0, 0, (-1), 4, (-1), (-1), 0, 0, 0, 0, (-1), 4, (-1), (-1), (-1), 0, 0, (-4), 0, 0, 1, (-4), 0, (-4), (-4), 0, 0, (-5), (-8), (-8), (-5), (-5), (-8), (-8), (-5), (-8), (-8), (-8), (-8), (-5), (-8), (-5), (-8), (-8), (-8), (-8), 0, (-8), (-8), (-5), (-8), (-8), (-8), (-5), (-8), (-8), 0, 5, 5, 0, 5, 0, 0, 0, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 37, 37, 37, 37, 37, 37, 37, 18, 31, 37, 37, 37, 37, 32, 26, 37, (-37), (-37), (-37), (-37), (-43), (-37), (-37), (-37), (-37), (-37), (-37), (-37), (-37), 36, 36, 36, 36, 32, 35, 24, 36, 24, 29, 27, 36, 23, 24, 36, 36, 36, 36, 33, (-36), (-38), (-36), (-36), (-36), (-36), (-36), (-36), (-36), (-36), (-36), (-36), (-36), 34, 34, 31, 30, 12, 34, 12, 25, 34, 1, 34, 34, 34, 34, 10, 34, 22, 23, 34, 20, 34, 12, 15, (-34), (-34), (-34), (-34), (-34), (-34), (-34), (-34), (-34), (-34), (-34), (-34), (-34), 34, 34, 34, 25, 34, 34, 34, 13, 18, 34, 34, 21, 14, 16, 34, 34, 34, 34, 18, 34, 29, 34, 34, 34, 34, 14, 31, 33, 34, (-34), (-34), (-34), (-34), (-34), (-34), (-34), (-34), (-34), (-34), (-34), (-34), (-34), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-25), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-4), 0, 0, 0, 0, 0, 0, 0, 0, 0, (-3), 0, 0, 0, (-13), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 23, 23, 26, 13, 16, 17, 12, 9, 9, 9, 7, 21, 19, 10, 13, 5, 21, 24, 26, 26, 10, 26, 24, 26, 26, 26, 26, 21, 26, 26, 0, 0, 0, 0, (-4), 0, (-18), 0, 0, 0, 0, 0, 0, 0, 0, (-14), 0, (-11), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def u26 (k : ℕ) : ℤ := ul26.getD k 0

def c26_0 (r t : ℕ) : Bool := gb11 r (q26 t)
def c26_1 (r t : ℕ) : Bool := gb13 r (q26 t)
def c26_2 (r t : ℕ) : Bool := gb17 r (q26 t)
def c26_3 (r t : ℕ) : Bool := gb19 r (q26 t)
def c26_4 (r t : ℕ) : Bool := gb23 r (q26 t)
def c26_5 (r t : ℕ) : Bool := gb29 r (q26 t)
def c26_6 (r t : ℕ) : Bool := gb31 r (q26 t)

def S26_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (w26 t + 5) * (if c26_0 r t then 1 else 0)
def S26_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (w26 t + 5) * (if c26_1 r t then 1 else 0)
def S26_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (w26 t + 5) * (if c26_2 r t then 1 else 0)
def S26_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (w26 t + 5) * (if c26_3 r t then 1 else 0)
def S26_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (w26 t + 5) * (if c26_4 r t then 1 else 0)
def S26_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (w26 t + 5) * (if c26_5 r t then 1 else 0)
def S26_6 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (w26 t + 5) * (if c26_6 r t then 1 else 0)

def L26_0 (r : ℕ) : ℤ := u26 (13 + r) + u26 (41 + r) + u26 (71 + r) + u26 (105 + r) + u26 (145 + r) + u26 (187 + r)
def L26_1 (r : ℕ) : ℤ := u26 (0 + r) + u26 (215 + r) + u26 (247 + r) + u26 (283 + r) + u26 (325 + r) + u26 (369 + r)
def L26_2 (r : ℕ) : ℤ := u26 (24 + r) + u26 (198 + r) + u26 (401 + r) + u26 (441 + r) + u26 (487 + r) + u26 (535 + r)
def L26_3 (r : ℕ) : ℤ := u26 (52 + r) + u26 (228 + r) + u26 (382 + r) + u26 (575 + r) + u26 (623 + r) + u26 (673 + r)
def L26_4 (r : ℕ) : ℤ := u26 (82 + r) + u26 (260 + r) + u26 (418 + r) + u26 (552 + r) + u26 (721 + r) + u26 (775 + r)
def L26_5 (r : ℕ) : ℤ := u26 (116 + r) + u26 (296 + r) + u26 (458 + r) + u26 (594 + r) + u26 (692 + r) + u26 (829 + r)
def L26_6 (r : ℕ) : ℤ := u26 (156 + r) + u26 (338 + r) + u26 (504 + r) + u26 (642 + r) + u26 (744 + r) + u26 (798 + r)

def aS26_0 (r : ℕ) : ℤ := S26_0 r - L26_0 r
def MS26_0 : ℤ := CaseSplit.mxr (aS26_0) 10
def aS26_1 (r : ℕ) : ℤ := S26_1 r - L26_1 r
def MS26_1 : ℤ := CaseSplit.mxr (aS26_1) 12
def aS26_2 (r : ℕ) : ℤ := S26_2 r - L26_2 r
def MS26_2 : ℤ := CaseSplit.mxr (aS26_2) 16
def aS26_3 (r : ℕ) : ℤ := S26_3 r - L26_3 r
def MS26_3 : ℤ := CaseSplit.mxr (aS26_3) 18
def aS26_4 (r : ℕ) : ℤ := S26_4 r - L26_4 r
def MS26_4 : ℤ := CaseSplit.mxr (aS26_4) 22
def aS26_5 (r : ℕ) : ℤ := S26_5 r - L26_5 r
def MS26_5 : ℤ := CaseSplit.mxr (aS26_5) 28
def aS26_6 (r : ℕ) : ℤ := S26_6 r - L26_6 r
def MS26_6 : ℤ := CaseSplit.mxr (aS26_6) 30

def N26_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (if c26_0 ra t && c26_1 rb t then 1 else 0)
def aP26_0 (ra rb : ℕ) : ℤ := -(5) * N26_0 ra rb + u26 (0 + rb) + u26 (13 + ra)
def MP26_0 : ℤ := CaseSplit.mxr2 (aP26_0) 10 12
def N26_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (if c26_0 ra t && c26_2 rb t then 1 else 0)
def aP26_1 (ra rb : ℕ) : ℤ := -(5) * N26_1 ra rb + u26 (24 + rb) + u26 (41 + ra)
def MP26_1 : ℤ := CaseSplit.mxr2 (aP26_1) 10 16
def N26_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (if c26_0 ra t && c26_3 rb t then 1 else 0)
def aP26_2 (ra rb : ℕ) : ℤ := -(5) * N26_2 ra rb + u26 (52 + rb) + u26 (71 + ra)
def MP26_2 : ℤ := CaseSplit.mxr2 (aP26_2) 10 18
def N26_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (if c26_0 ra t && c26_4 rb t then 1 else 0)
def aP26_3 (ra rb : ℕ) : ℤ := -(5) * N26_3 ra rb + u26 (82 + rb) + u26 (105 + ra)
def MP26_3 : ℤ := CaseSplit.mxr2 (aP26_3) 10 22
def N26_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (if c26_0 ra t && c26_5 rb t then 1 else 0)
def aP26_4 (ra rb : ℕ) : ℤ := -(5) * N26_4 ra rb + u26 (116 + rb) + u26 (145 + ra)
def MP26_4 : ℤ := CaseSplit.mxr2 (aP26_4) 10 28
def N26_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (if c26_0 ra t && c26_6 rb t then 1 else 0)
def aP26_5 (ra rb : ℕ) : ℤ := -(5) * N26_5 ra rb + u26 (156 + rb) + u26 (187 + ra)
def MP26_5 : ℤ := CaseSplit.mxr2 (aP26_5) 10 30
def P26_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (if c26_1 ra t && c26_2 rb t then 1 else 0)
def C26_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (if c26_1 ra t && c26_2 rb t && c26_0 s t then 1 else 0)
def M26_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C26_6 ra rb) 10
def E26_6 : List ℕ := [0, 11, 68, 79, 90, 101, 120, 126, 154, 165, 174, 180, 210, 216]
def N26_6 (ra rb : ℕ) : ℤ := if E26_6.contains (ra * 17 + rb) = true then P26_6 ra rb - M26_6 ra rb else 0
def aP26_6 (ra rb : ℕ) : ℤ := -(5) * N26_6 ra rb + u26 (198 + rb) + u26 (215 + ra)
def MP26_6 : ℤ := CaseSplit.mxr2 (aP26_6) 12 16
def P26_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (if c26_1 ra t && c26_3 rb t then 1 else 0)
def C26_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (if c26_1 ra t && c26_3 rb t && c26_0 s t then 1 else 0)
def M26_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C26_7 ra rb) 10
def E26_7 : List ℕ := [27, 30, 33, 64, 67, 70, 91, 98, 106, 140, 146, 167, 170, 174, 198, 204, 238, 246]
def N26_7 (ra rb : ℕ) : ℤ := if E26_7.contains (ra * 19 + rb) = true then P26_7 ra rb - M26_7 ra rb else 0
def aP26_7 (ra rb : ℕ) : ℤ := -(5) * N26_7 ra rb + u26 (228 + rb) + u26 (247 + ra)
def MP26_7 : ℤ := CaseSplit.mxr2 (aP26_7) 12 18
def P26_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (if c26_1 ra t && c26_4 rb t then 1 else 0)
def C26_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (if c26_1 ra t && c26_4 rb t && c26_0 s t then 1 else 0)
def M26_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C26_8 ra rb) 10
def E26_8 : List ℕ := []
def N26_8 (ra rb : ℕ) : ℤ := if E26_8.contains (ra * 23 + rb) = true then P26_8 ra rb - M26_8 ra rb else 0
def aP26_8 (ra rb : ℕ) : ℤ := -(5) * N26_8 ra rb + u26 (260 + rb) + u26 (283 + ra)
def MP26_8 : ℤ := CaseSplit.mxr2 (aP26_8) 12 22
def P26_9 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (if c26_1 ra t && c26_5 rb t then 1 else 0)
def C26_9 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (if c26_1 ra t && c26_5 rb t && c26_0 s t then 1 else 0)
def M26_9 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C26_9 ra rb) 10
def E26_9 : List ℕ := [82, 193, 309, 343]
def N26_9 (ra rb : ℕ) : ℤ := if E26_9.contains (ra * 29 + rb) = true then P26_9 ra rb - M26_9 ra rb else 0
def aP26_9 (ra rb : ℕ) : ℤ := -(5) * N26_9 ra rb + u26 (296 + rb) + u26 (325 + ra)
def MP26_9 : ℤ := CaseSplit.mxr2 (aP26_9) 12 28
def P26_10 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (if c26_1 ra t && c26_6 rb t then 1 else 0)
def C26_10 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (if c26_1 ra t && c26_6 rb t && c26_0 s t then 1 else 0)
def M26_10 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C26_10 ra rb) 10
def E26_10 : List ℕ := [3, 282]
def N26_10 (ra rb : ℕ) : ℤ := if E26_10.contains (ra * 31 + rb) = true then P26_10 ra rb - M26_10 ra rb else 0
def aP26_10 (ra rb : ℕ) : ℤ := -(5) * N26_10 ra rb + u26 (338 + rb) + u26 (369 + ra)
def MP26_10 : ℤ := CaseSplit.mxr2 (aP26_10) 12 30
def N26_11 (_ra _rb : ℕ) : ℤ := 0
def aP26_11 (ra rb : ℕ) : ℤ := -(5) * N26_11 ra rb + u26 (382 + rb) + u26 (401 + ra)
def MP26_11 : ℤ := CaseSplit.mxr2 (aP26_11) 16 18
def N26_12 (_ra _rb : ℕ) : ℤ := 0
def aP26_12 (ra rb : ℕ) : ℤ := -(5) * N26_12 ra rb + u26 (418 + rb) + u26 (441 + ra)
def MP26_12 : ℤ := CaseSplit.mxr2 (aP26_12) 16 22
def N26_13 (_ra _rb : ℕ) : ℤ := 0
def aP26_13 (ra rb : ℕ) : ℤ := -(5) * N26_13 ra rb + u26 (458 + rb) + u26 (487 + ra)
def MP26_13 : ℤ := CaseSplit.mxr2 (aP26_13) 16 28
def N26_14 (_ra _rb : ℕ) : ℤ := 0
def aP26_14 (ra rb : ℕ) : ℤ := -(5) * N26_14 ra rb + u26 (504 + rb) + u26 (535 + ra)
def MP26_14 : ℤ := CaseSplit.mxr2 (aP26_14) 16 30
def N26_15 (_ra _rb : ℕ) : ℤ := 0
def aP26_15 (ra rb : ℕ) : ℤ := -(5) * N26_15 ra rb + u26 (552 + rb) + u26 (575 + ra)
def MP26_15 : ℤ := CaseSplit.mxr2 (aP26_15) 18 22
def N26_16 (_ra _rb : ℕ) : ℤ := 0
def aP26_16 (ra rb : ℕ) : ℤ := -(5) * N26_16 ra rb + u26 (594 + rb) + u26 (623 + ra)
def MP26_16 : ℤ := CaseSplit.mxr2 (aP26_16) 18 28
def N26_17 (_ra _rb : ℕ) : ℤ := 0
def aP26_17 (ra rb : ℕ) : ℤ := -(5) * N26_17 ra rb + u26 (642 + rb) + u26 (673 + ra)
def MP26_17 : ℤ := CaseSplit.mxr2 (aP26_17) 18 30
def N26_18 (_ra _rb : ℕ) : ℤ := 0
def aP26_18 (ra rb : ℕ) : ℤ := -(5) * N26_18 ra rb + u26 (692 + rb) + u26 (721 + ra)
def MP26_18 : ℤ := CaseSplit.mxr2 (aP26_18) 22 28
def N26_19 (_ra _rb : ℕ) : ℤ := 0
def aP26_19 (ra rb : ℕ) : ℤ := -(5) * N26_19 ra rb + u26 (744 + rb) + u26 (775 + ra)
def MP26_19 : ℤ := CaseSplit.mxr2 (aP26_19) 22 30
def N26_20 (_ra _rb : ℕ) : ℤ := 0
def aP26_20 (ra rb : ℕ) : ℤ := -(5) * N26_20 ra rb + u26 (798 + rb) + u26 (829 + ra)
def MP26_20 : ℤ := CaseSplit.mxr2 (aP26_20) 28 30

def rhs26 : ℤ := (∑ t ∈ Finset.range n26, w26 t) + 5 * (n26 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn26 : ∀ t, t < n26 → (0 : ℤ) ≤ w26 t := by decide
theorem plt26 : ∀ t, t < n26 → q26 t < 65 := by decide
theorem pfree26_5 : ∀ t, t < n26 → gb5 3 (q26 t) = false := by decide
theorem pfree26_7 : ∀ t, t < n26 → gb7 5 (q26 t) = false := by decide
theorem MSv26_0 : MS26_0 = 56 := by decide +kernel
theorem MSv26_1 : MS26_1 = 186 := by decide +kernel
theorem MSv26_2 : MS26_2 = 1 := by decide +kernel
theorem MSv26_3 : MS26_3 = 1 := by decide +kernel
theorem MSv26_4 : MS26_4 = 1 := by decide +kernel
theorem MSv26_5 : MS26_5 = 1 := by decide +kernel
theorem MSv26_6 : MS26_6 = 1 := by decide +kernel
theorem MPv26_0 : MP26_0 = 0 := by decide +kernel
theorem MPv26_1 : MP26_1 = 0 := by decide +kernel
theorem MPv26_2 : MP26_2 = 0 := by decide +kernel
theorem MPv26_3 : MP26_3 = 0 := by decide +kernel
theorem MPv26_4 : MP26_4 = 0 := by decide +kernel
theorem MPv26_5 : MP26_5 = 0 := by decide +kernel
theorem MPv26_6 : MP26_6 = 0 := by decide +kernel
theorem MPv26_7 : MP26_7 = 0 := by decide +kernel
theorem MPv26_8 : MP26_8 = 0 := by decide +kernel
theorem MPv26_9 : MP26_9 = 0 := by decide +kernel
theorem MPv26_10 : MP26_10 = 0 := by decide +kernel
theorem MPv26_11 : MP26_11 = 0 := by decide +kernel
theorem MPv26_12 : MP26_12 = 0 := by decide +kernel
theorem MPv26_13 : MP26_13 = 0 := by decide +kernel
theorem MPv26_14 : MP26_14 = 0 := by decide +kernel
theorem MPv26_15 : MP26_15 = 0 := by decide +kernel
theorem MPv26_16 : MP26_16 = 0 := by decide +kernel
theorem MPv26_17 : MP26_17 = 0 := by decide +kernel
theorem MPv26_18 : MP26_18 = 0 := by decide +kernel
theorem MPv26_19 : MP26_19 = 0 := by decide +kernel
theorem MPv26_20 : MP26_20 = 26 := by decide +kernel
theorem rhsv26 : rhs26 = 274 := by decide +kernel

/-- **The case-26 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/13.
    (Scaled by the common denominator 13: 273 < 274.) -/
theorem cert26 : MS26_0 + MS26_1 + MS26_2 + MS26_3 + MS26_4 + MS26_5 + MS26_6 + MP26_0 + MP26_1 + MP26_2 + MP26_3 + MP26_4 + MP26_5 + MP26_6 + MP26_7 + MP26_8 + MP26_9 + MP26_10 + MP26_11 + MP26_12 + MP26_13 + MP26_14 + MP26_15 + MP26_16 + MP26_17 + MP26_18 + MP26_19 + MP26_20 < rhs26 := by
  rw [MSv26_0, MSv26_1, MSv26_2, MSv26_3, MSv26_4, MSv26_5, MSv26_6, MPv26_0, MPv26_1, MPv26_2, MPv26_3, MPv26_4, MPv26_5, MPv26_6, MPv26_7, MPv26_8, MPv26_9, MPv26_10, MPv26_11, MPv26_12, MPv26_13, MPv26_14, MPv26_15, MPv26_16, MPv26_17, MPv26_18, MPv26_19, MPv26_20, rhsv26]
  decide

def Dg26 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := (if c26_0 r0 t then 1 else 0) + (if c26_1 r1 t then 1 else 0) + (if c26_2 r2 t then 1 else 0) + (if c26_3 r3 t then 1 else 0) + (if c26_4 r4 t then 1 else 0) + (if c26_5 r5 t then 1 else 0) + (if c26_6 r6 t then 1 else 0)
def Wl26_0 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c26_0 r0 t && c26_1 r1 t then 1 else 0
def Wl26_1 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c26_0 r0 t && c26_2 r2 t then 1 else 0
def Wl26_2 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c26_0 r0 t && c26_3 r3 t then 1 else 0
def Wl26_3 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c26_0 r0 t && c26_4 r4 t then 1 else 0
def Wl26_4 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c26_0 r0 t && c26_5 r5 t then 1 else 0
def Wl26_5 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c26_0 r0 t && c26_6 r6 t then 1 else 0
def Wl26_6 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c26_0 r0 t && c26_1 r1 t && c26_2 r2 t then 1 else 0
def Wl26_7 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c26_0 r0 t && c26_1 r1 t && c26_3 r3 t then 1 else 0
def Wl26_8 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c26_0 r0 t && c26_1 r1 t && c26_4 r4 t then 1 else 0
def Wl26_9 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c26_0 r0 t && c26_1 r1 t && c26_5 r5 t then 1 else 0
def Wl26_10 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c26_0 r0 t && c26_1 r1 t && c26_6 r6 t then 1 else 0
def Wl26_11 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c26_0 r0 t && !c26_1 r1 t && c26_2 r2 t && c26_3 r3 t then 1 else 0
def Wl26_12 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c26_0 r0 t && !c26_1 r1 t && c26_2 r2 t && c26_4 r4 t then 1 else 0
def Wl26_13 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c26_0 r0 t && !c26_1 r1 t && c26_2 r2 t && c26_5 r5 t then 1 else 0
def Wl26_14 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c26_0 r0 t && !c26_1 r1 t && c26_2 r2 t && c26_6 r6 t then 1 else 0
def Wl26_15 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c26_0 r0 t && !c26_1 r1 t && !c26_2 r2 t && c26_3 r3 t && c26_4 r4 t then 1 else 0
def Wl26_16 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c26_0 r0 t && !c26_1 r1 t && !c26_2 r2 t && c26_3 r3 t && c26_5 r5 t then 1 else 0
def Wl26_17 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c26_0 r0 t && !c26_1 r1 t && !c26_2 r2 t && c26_3 r3 t && c26_6 r6 t then 1 else 0
def Wl26_18 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c26_0 r0 t && !c26_1 r1 t && !c26_2 r2 t && !c26_3 r3 t && c26_4 r4 t && c26_5 r5 t then 1 else 0
def Wl26_19 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c26_0 r0 t && !c26_1 r1 t && !c26_2 r2 t && !c26_3 r3 t && c26_4 r4 t && c26_6 r6 t then 1 else 0
def Wl26_20 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c26_0 r0 t && !c26_1 r1 t && !c26_2 r2 t && !c26_3 r3 t && !c26_4 r4 t && c26_5 r5 t && c26_6 r6 t then 1 else 0

/-- **No configuration blocks the whole window in case 26.** -/
theorem nocov26 {r0 r1 r2 r3 r4 r5 r6 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29) (h6 : r6 < 31)
    (hcov : ∀ t, t < n26 → (c26_0 r0 t || c26_1 r1 t || c26_2 r2 t || c26_3 r3 t || c26_4 r4 t || c26_5 r5 t || c26_6 r6 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n26, (1 : ℤ) + (Wl26_0 r0 r1 r2 r3 r4 r5 r6 t + Wl26_1 r0 r1 r2 r3 r4 r5 r6 t + Wl26_2 r0 r1 r2 r3 r4 r5 r6 t + Wl26_3 r0 r1 r2 r3 r4 r5 r6 t + Wl26_4 r0 r1 r2 r3 r4 r5 r6 t + Wl26_5 r0 r1 r2 r3 r4 r5 r6 t + Wl26_6 r0 r1 r2 r3 r4 r5 r6 t + Wl26_7 r0 r1 r2 r3 r4 r5 r6 t + Wl26_8 r0 r1 r2 r3 r4 r5 r6 t + Wl26_9 r0 r1 r2 r3 r4 r5 r6 t + Wl26_10 r0 r1 r2 r3 r4 r5 r6 t + Wl26_11 r0 r1 r2 r3 r4 r5 r6 t + Wl26_12 r0 r1 r2 r3 r4 r5 r6 t + Wl26_13 r0 r1 r2 r3 r4 r5 r6 t + Wl26_14 r0 r1 r2 r3 r4 r5 r6 t + Wl26_15 r0 r1 r2 r3 r4 r5 r6 t + Wl26_16 r0 r1 r2 r3 r4 r5 r6 t + Wl26_17 r0 r1 r2 r3 r4 r5 r6 t + Wl26_18 r0 r1 r2 r3 r4 r5 r6 t + Wl26_19 r0 r1 r2 r3 r4 r5 r6 t + Wl26_20 r0 r1 r2 r3 r4 r5 r6 t) ≤ Dg26 r0 r1 r2 r3 r4 r5 r6 t := by
    intro t ht
    simp only [Wl26_0, Wl26_1, Wl26_2, Wl26_3, Wl26_4, Wl26_5, Wl26_6, Wl26_7, Wl26_8, Wl26_9, Wl26_10, Wl26_11, Wl26_12, Wl26_13, Wl26_14, Wl26_15, Wl26_16, Wl26_17, Wl26_18, Wl26_19, Wl26_20, Dg26]
    exact CaseSplit.lowest7 _ _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n26, (1 : ℤ) ≤ Dg26 r0 r1 r2 r3 r4 r5 r6 t := by
    intro t ht
    simp only [Dg26]
    exact CaseSplit.degpos7 _ _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n26 : ℤ) + ((∑ t ∈ Finset.range n26, Wl26_0 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n26, Wl26_1 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n26, Wl26_2 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n26, Wl26_3 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n26, Wl26_4 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n26, Wl26_5 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n26, Wl26_6 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n26, Wl26_7 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n26, Wl26_8 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n26, Wl26_9 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n26, Wl26_10 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n26, Wl26_11 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n26, Wl26_12 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n26, Wl26_13 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n26, Wl26_14 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n26, Wl26_15 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n26, Wl26_16 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n26, Wl26_17 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n26, Wl26_18 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n26, Wl26_19 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n26, Wl26_20 r0 r1 r2 r3 r4 r5 r6 t)) ≤ ∑ t ∈ Finset.range n26, Dg26 r0 r1 r2 r3 r4 r5 r6 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N26_0 r0 r1 ≤ ∑ t ∈ Finset.range n26, Wl26_0 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N26_0, Wl26_0, le_refl]
  have hn1 : N26_1 r0 r2 ≤ ∑ t ∈ Finset.range n26, Wl26_1 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N26_1, Wl26_1, le_refl]
  have hn2 : N26_2 r0 r3 ≤ ∑ t ∈ Finset.range n26, Wl26_2 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N26_2, Wl26_2, le_refl]
  have hn3 : N26_3 r0 r4 ≤ ∑ t ∈ Finset.range n26, Wl26_3 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N26_3, Wl26_3, le_refl]
  have hn4 : N26_4 r0 r5 ≤ ∑ t ∈ Finset.range n26, Wl26_4 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N26_4, Wl26_4, le_refl]
  have hn5 : N26_5 r0 r6 ≤ ∑ t ∈ Finset.range n26, Wl26_5 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N26_5, Wl26_5, le_refl]
  have hn6 : N26_6 r1 r2 ≤ ∑ t ∈ Finset.range n26, Wl26_6 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n26, Wl26_6 r0 r1 r2 r3 r4 r5 r6 t
        = (if c26_1 r1 t && c26_2 r2 t then (1:ℤ) else 0)
          - (if c26_1 r1 t && c26_2 r2 t && c26_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl26_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n26, Wl26_6 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl26_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n26, Wl26_6 r0 r1 r2 r3 r4 r5 r6 t
        = P26_6 r1 r2 - C26_6 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P26_6, C26_6]
    have hm : C26_6 r1 r2 r0 ≤ M26_6 r1 r2 :=
      CaseSplit.le_mxr (C26_6 r1 r2) 10 r0 (by omega)
    simp only [N26_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N26_7 r1 r3 ≤ ∑ t ∈ Finset.range n26, Wl26_7 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n26, Wl26_7 r0 r1 r2 r3 r4 r5 r6 t
        = (if c26_1 r1 t && c26_3 r3 t then (1:ℤ) else 0)
          - (if c26_1 r1 t && c26_3 r3 t && c26_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl26_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n26, Wl26_7 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl26_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n26, Wl26_7 r0 r1 r2 r3 r4 r5 r6 t
        = P26_7 r1 r3 - C26_7 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P26_7, C26_7]
    have hm : C26_7 r1 r3 r0 ≤ M26_7 r1 r3 :=
      CaseSplit.le_mxr (C26_7 r1 r3) 10 r0 (by omega)
    simp only [N26_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N26_8 r1 r4 ≤ ∑ t ∈ Finset.range n26, Wl26_8 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n26, Wl26_8 r0 r1 r2 r3 r4 r5 r6 t
        = (if c26_1 r1 t && c26_4 r4 t then (1:ℤ) else 0)
          - (if c26_1 r1 t && c26_4 r4 t && c26_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl26_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n26, Wl26_8 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl26_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n26, Wl26_8 r0 r1 r2 r3 r4 r5 r6 t
        = P26_8 r1 r4 - C26_8 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P26_8, C26_8]
    have hm : C26_8 r1 r4 r0 ≤ M26_8 r1 r4 :=
      CaseSplit.le_mxr (C26_8 r1 r4) 10 r0 (by omega)
    simp only [N26_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N26_9 r1 r5 ≤ ∑ t ∈ Finset.range n26, Wl26_9 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n26, Wl26_9 r0 r1 r2 r3 r4 r5 r6 t
        = (if c26_1 r1 t && c26_5 r5 t then (1:ℤ) else 0)
          - (if c26_1 r1 t && c26_5 r5 t && c26_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl26_9]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n26, Wl26_9 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl26_9]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n26, Wl26_9 r0 r1 r2 r3 r4 r5 r6 t
        = P26_9 r1 r5 - C26_9 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P26_9, C26_9]
    have hm : C26_9 r1 r5 r0 ≤ M26_9 r1 r5 :=
      CaseSplit.le_mxr (C26_9 r1 r5) 10 r0 (by omega)
    simp only [N26_9]
    split
    · rw [hL]; omega
    · exact hnn
  have hn10 : N26_10 r1 r6 ≤ ∑ t ∈ Finset.range n26, Wl26_10 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n26, Wl26_10 r0 r1 r2 r3 r4 r5 r6 t
        = (if c26_1 r1 t && c26_6 r6 t then (1:ℤ) else 0)
          - (if c26_1 r1 t && c26_6 r6 t && c26_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl26_10]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n26, Wl26_10 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl26_10]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n26, Wl26_10 r0 r1 r2 r3 r4 r5 r6 t
        = P26_10 r1 r6 - C26_10 r1 r6 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P26_10, C26_10]
    have hm : C26_10 r1 r6 r0 ≤ M26_10 r1 r6 :=
      CaseSplit.le_mxr (C26_10 r1 r6) 10 r0 (by omega)
    simp only [N26_10]
    split
    · rw [hL]; omega
    · exact hnn
  have hn11 : N26_11 r2 r3 ≤ ∑ t ∈ Finset.range n26, Wl26_11 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N26_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl26_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N26_12 r2 r4 ≤ ∑ t ∈ Finset.range n26, Wl26_12 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N26_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl26_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N26_13 r2 r5 ≤ ∑ t ∈ Finset.range n26, Wl26_13 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N26_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl26_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N26_14 r2 r6 ≤ ∑ t ∈ Finset.range n26, Wl26_14 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N26_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl26_14]
    exact CaseSplit.ind_nonneg _
  have hn15 : N26_15 r3 r4 ≤ ∑ t ∈ Finset.range n26, Wl26_15 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N26_15]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl26_15]
    exact CaseSplit.ind_nonneg _
  have hn16 : N26_16 r3 r5 ≤ ∑ t ∈ Finset.range n26, Wl26_16 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N26_16]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl26_16]
    exact CaseSplit.ind_nonneg _
  have hn17 : N26_17 r3 r6 ≤ ∑ t ∈ Finset.range n26, Wl26_17 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N26_17]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl26_17]
    exact CaseSplit.ind_nonneg _
  have hn18 : N26_18 r4 r5 ≤ ∑ t ∈ Finset.range n26, Wl26_18 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N26_18]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl26_18]
    exact CaseSplit.ind_nonneg _
  have hn19 : N26_19 r4 r6 ≤ ∑ t ∈ Finset.range n26, Wl26_19 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N26_19]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl26_19]
    exact CaseSplit.ind_nonneg _
  have hn20 : N26_20 r5 r6 ≤ ∑ t ∈ Finset.range n26, Wl26_20 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N26_20]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl26_20]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n26, (w26 t + 5) * Dg26 r0 r1 r2 r3 r4 r5 r6 t = S26_0 r0 + S26_1 r1 + S26_2 r2 + S26_3 r3 + S26_4 r4 + S26_5 r5 + S26_6 r6 := by
    simp only [S26_0, S26_1, S26_2, S26_3, S26_4, S26_5, S26_6, Dg26, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n26, (w26 t + 5) * Dg26 r0 r1 r2 r3 r4 r5 r6 t
      = (∑ t ∈ Finset.range n26, w26 t * Dg26 r0 r1 r2 r3 r4 r5 r6 t)
        + 5 * (∑ t ∈ Finset.range n26, Dg26 r0 r1 r2 r3 r4 r5 r6 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n26, w26 t)
      ≤ ∑ t ∈ Finset.range n26, w26 t * Dg26 r0 r1 r2 r3 r4 r5 r6 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg26 r0 r1 r2 r3 r4 r5 r6 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w26 t := wnn26 t (Finset.mem_range.mp ht)
    calc w26 t = w26 t * 1 := (mul_one _).symm
      _ ≤ w26 t * Dg26 r0 r1 r2 r3 r4 r5 r6 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS26_0 r0 + aS26_1 r1 + aS26_2 r2 + aS26_3 r3 + aS26_4 r4 + aS26_5 r5 + aS26_6 r6) + (aP26_0 r0 r1 + aP26_1 r0 r2 + aP26_2 r0 r3 + aP26_3 r0 r4 + aP26_4 r0 r5 + aP26_5 r0 r6 + aP26_6 r1 r2 + aP26_7 r1 r3 + aP26_8 r1 r4 + aP26_9 r1 r5 + aP26_10 r1 r6 + aP26_11 r2 r3 + aP26_12 r2 r4 + aP26_13 r2 r5 + aP26_14 r2 r6 + aP26_15 r3 r4 + aP26_16 r3 r5 + aP26_17 r3 r6 + aP26_18 r4 r5 + aP26_19 r4 r6 + aP26_20 r5 r6) = (S26_0 r0 + S26_1 r1 + S26_2 r2 + S26_3 r3 + S26_4 r4 + S26_5 r5 + S26_6 r6) - 5 * (N26_0 r0 r1 + N26_1 r0 r2 + N26_2 r0 r3 + N26_3 r0 r4 + N26_4 r0 r5 + N26_5 r0 r6 + N26_6 r1 r2 + N26_7 r1 r3 + N26_8 r1 r4 + N26_9 r1 r5 + N26_10 r1 r6 + N26_11 r2 r3 + N26_12 r2 r4 + N26_13 r2 r5 + N26_14 r2 r6 + N26_15 r3 r4 + N26_16 r3 r5 + N26_17 r3 r6 + N26_18 r4 r5 + N26_19 r4 r6 + N26_20 r5 r6) := by
    simp only [aS26_0, aS26_1, aS26_2, aS26_3, aS26_4, aS26_5, aS26_6, aP26_0, aP26_1, aP26_2, aP26_3, aP26_4, aP26_5, aP26_6, aP26_7, aP26_8, aP26_9, aP26_10, aP26_11, aP26_12, aP26_13, aP26_14, aP26_15, aP26_16, aP26_17, aP26_18, aP26_19, aP26_20, L26_0, L26_1, L26_2, L26_3, L26_4, L26_5, L26_6]
    ring
  have bS0 : aS26_0 r0 ≤ MS26_0 := CaseSplit.le_mxr (aS26_0) 10 r0 (by omega)
  have bS1 : aS26_1 r1 ≤ MS26_1 := CaseSplit.le_mxr (aS26_1) 12 r1 (by omega)
  have bS2 : aS26_2 r2 ≤ MS26_2 := CaseSplit.le_mxr (aS26_2) 16 r2 (by omega)
  have bS3 : aS26_3 r3 ≤ MS26_3 := CaseSplit.le_mxr (aS26_3) 18 r3 (by omega)
  have bS4 : aS26_4 r4 ≤ MS26_4 := CaseSplit.le_mxr (aS26_4) 22 r4 (by omega)
  have bS5 : aS26_5 r5 ≤ MS26_5 := CaseSplit.le_mxr (aS26_5) 28 r5 (by omega)
  have bS6 : aS26_6 r6 ≤ MS26_6 := CaseSplit.le_mxr (aS26_6) 30 r6 (by omega)
  have bP0 : aP26_0 r0 r1 ≤ MP26_0 := CaseSplit.le_mxr2 (aP26_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP26_1 r0 r2 ≤ MP26_1 := CaseSplit.le_mxr2 (aP26_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP26_2 r0 r3 ≤ MP26_2 := CaseSplit.le_mxr2 (aP26_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP26_3 r0 r4 ≤ MP26_3 := CaseSplit.le_mxr2 (aP26_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP26_4 r0 r5 ≤ MP26_4 := CaseSplit.le_mxr2 (aP26_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP26_5 r0 r6 ≤ MP26_5 := CaseSplit.le_mxr2 (aP26_5) 10 30 r0 r6 (by omega) (by omega)
  have bP6 : aP26_6 r1 r2 ≤ MP26_6 := CaseSplit.le_mxr2 (aP26_6) 12 16 r1 r2 (by omega) (by omega)
  have bP7 : aP26_7 r1 r3 ≤ MP26_7 := CaseSplit.le_mxr2 (aP26_7) 12 18 r1 r3 (by omega) (by omega)
  have bP8 : aP26_8 r1 r4 ≤ MP26_8 := CaseSplit.le_mxr2 (aP26_8) 12 22 r1 r4 (by omega) (by omega)
  have bP9 : aP26_9 r1 r5 ≤ MP26_9 := CaseSplit.le_mxr2 (aP26_9) 12 28 r1 r5 (by omega) (by omega)
  have bP10 : aP26_10 r1 r6 ≤ MP26_10 := CaseSplit.le_mxr2 (aP26_10) 12 30 r1 r6 (by omega) (by omega)
  have bP11 : aP26_11 r2 r3 ≤ MP26_11 := CaseSplit.le_mxr2 (aP26_11) 16 18 r2 r3 (by omega) (by omega)
  have bP12 : aP26_12 r2 r4 ≤ MP26_12 := CaseSplit.le_mxr2 (aP26_12) 16 22 r2 r4 (by omega) (by omega)
  have bP13 : aP26_13 r2 r5 ≤ MP26_13 := CaseSplit.le_mxr2 (aP26_13) 16 28 r2 r5 (by omega) (by omega)
  have bP14 : aP26_14 r2 r6 ≤ MP26_14 := CaseSplit.le_mxr2 (aP26_14) 16 30 r2 r6 (by omega) (by omega)
  have bP15 : aP26_15 r3 r4 ≤ MP26_15 := CaseSplit.le_mxr2 (aP26_15) 18 22 r3 r4 (by omega) (by omega)
  have bP16 : aP26_16 r3 r5 ≤ MP26_16 := CaseSplit.le_mxr2 (aP26_16) 18 28 r3 r5 (by omega) (by omega)
  have bP17 : aP26_17 r3 r6 ≤ MP26_17 := CaseSplit.le_mxr2 (aP26_17) 18 30 r3 r6 (by omega) (by omega)
  have bP18 : aP26_18 r4 r5 ≤ MP26_18 := CaseSplit.le_mxr2 (aP26_18) 22 28 r4 r5 (by omega) (by omega)
  have bP19 : aP26_19 r4 r6 ≤ MP26_19 := CaseSplit.le_mxr2 (aP26_19) 22 30 r4 r6 (by omega) (by omega)
  have bP20 : aP26_20 r5 r6 ≤ MP26_20 := CaseSplit.le_mxr2 (aP26_20) 28 30 r5 r6 (by omega) (by omega)
  have hrhs : rhs26 = (∑ t ∈ Finset.range n26, w26 t) + 5 * (n26 : ℤ) := rfl
  have hc := cert26
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, hn15, hn16, hn17, hn18, hn19, hn20, bS0, bS1, bS2, bS3, bS4, bS5, bS6, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14, bP15, bP16, bP17, bP18, bP19, bP20]

end IncCert31
