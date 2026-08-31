/-
INCREMENT-WIDTH CERTIFICATE, step 29->31, case 31 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_29_31.json, which re-derives every number
from the primes alone).

Machine 31, INCREMENT width 65 = F_2(29) + s_min(31) = 55 + 10,
held gears [5, 7] at phases [4, 3].  Free gears [11, 13, 17, 19, 23, 29, 31].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 7.
-/
import IncCert31B

namespace IncCert31

/-! ### case 31: held gears at phases [4, 3] -/

def p31 : List ℕ := [1, 4, 6, 8, 9, 11, 13, 14, 16, 18, 21, 23, 28, 29, 34, 36, 39, 41, 43, 44, 46, 48, 49, 51, 53, 56, 58, 63, 64]
def q31 (t : ℕ) : ℕ := p31.getD t 0
def n31 : ℕ := 29
def yl31 : List ℤ := [0, 0, 0, 0, 2, 6, 0, 2, 6, 0, 0, 0, 1, 0, 0, 1, 2, 6, 5, 6, 7, 4, 0, 5, 0, 1, 2, 0, 0]
def w31 (t : ℕ) : ℤ := yl31.getD t 0
def ul31 : List ℤ := [0, (-1), (-1), (-1), 0, (-1), (-1), 2, (-1), (-1), (-1), (-1), (-1), (-2), 1, (-2), 0, (-2), 1, (-2), (-2), 0, 1, 0, (-5), (-5), 0, 0, (-5), (-5), (-2), (-2), 0, (-2), (-5), (-5), (-2), (-2), 0, (-2), (-5), 0, 5, 2, 0, 0, 0, 2, 2, 2, 0, 0, (-1), (-1), (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), (-9), 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, (-5), 0, (-5), (-5), (-5), (-5), 0, (-5), (-5), 0, (-5), (-5), (-5), (-2), 0, (-5), (-5), (-5), (-5), (-5), (-5), (-2), (-2), 2, 0, 5, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-2), (-1), (-1), (-1), (-1), (-2), (-2), (-2), (-2), (-2), (-2), (-1), (-2), 0, (-2), (-2), 0, (-2), 0, (-2), (-2), (-2), (-2), 0, (-2), 0, (-1), (-2), 0, (-2), (-2), 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 20, 26, 31, 31, 27, 31, 23, 30, 31, 31, 25, 21, 18, 30, 31, 26, 31, (-31), (-31), (-31), (-31), (-31), (-31), (-31), (-31), (-31), (-31), (-31), (-31), (-31), 23, 15, 25, 15, 12, 25, 25, 23, 25, 20, 21, 25, 24, 25, 22, 25, 21, 25, 25, (-25), (-25), (-25), (-28), (-25), (-25), (-25), (-25), (-25), (-28), (-25), (-25), (-25), 9, 27, 19, 27, 21, 11, 27, 27, 20, 27, 17, 27, 9, 18, 27, 27, 25, 20, 16, 26, 9, 27, 27, (-27), (-27), (-27), (-27), (-27), (-27), (-27), (-27), (-27), (-27), (-27), (-27), (-28), 16, 16, 16, 16, 16, 16, 16, 13, 11, 16, 16, 16, 16, 16, 16, 11, 16, 13, 16, 16, 16, 0, 11, 16, 16, 16, 16, 1, 16, (-16), (-16), (-16), (-16), (-21), (-20), (-16), (-16), (-16), (-16), (-16), (-16), (-16), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 21, 6, 11, 16, 7, 11, 7, 21, 18, 13, 19, 8, 21, 16, 18, 21, 9, 21, 11, 18, 20, 21, 21, 16, 21, 13, 9, 16, 8, 13, (-1), (-6), (-6), 0, 0, 5, (-5), 5, 0, 0, 5, (-6), 5, 5, (-4), 5, (-6), 5, 5, 5, 5, 5, 0, 0, 2, 5, 0, 5, 0]
def u31 (k : ℕ) : ℤ := ul31.getD k 0

def c31_0 (r t : ℕ) : Bool := gb11 r (q31 t)
def c31_1 (r t : ℕ) : Bool := gb13 r (q31 t)
def c31_2 (r t : ℕ) : Bool := gb17 r (q31 t)
def c31_3 (r t : ℕ) : Bool := gb19 r (q31 t)
def c31_4 (r t : ℕ) : Bool := gb23 r (q31 t)
def c31_5 (r t : ℕ) : Bool := gb29 r (q31 t)
def c31_6 (r t : ℕ) : Bool := gb31 r (q31 t)

def S31_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (w31 t + 5) * (if c31_0 r t then 1 else 0)
def S31_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (w31 t + 5) * (if c31_1 r t then 1 else 0)
def S31_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (w31 t + 5) * (if c31_2 r t then 1 else 0)
def S31_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (w31 t + 5) * (if c31_3 r t then 1 else 0)
def S31_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (w31 t + 5) * (if c31_4 r t then 1 else 0)
def S31_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (w31 t + 5) * (if c31_5 r t then 1 else 0)
def S31_6 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (w31 t + 5) * (if c31_6 r t then 1 else 0)

def L31_0 (r : ℕ) : ℤ := u31 (13 + r) + u31 (41 + r) + u31 (71 + r) + u31 (105 + r) + u31 (145 + r) + u31 (187 + r)
def L31_1 (r : ℕ) : ℤ := u31 (0 + r) + u31 (215 + r) + u31 (247 + r) + u31 (283 + r) + u31 (325 + r) + u31 (369 + r)
def L31_2 (r : ℕ) : ℤ := u31 (24 + r) + u31 (198 + r) + u31 (401 + r) + u31 (441 + r) + u31 (487 + r) + u31 (535 + r)
def L31_3 (r : ℕ) : ℤ := u31 (52 + r) + u31 (228 + r) + u31 (382 + r) + u31 (575 + r) + u31 (623 + r) + u31 (673 + r)
def L31_4 (r : ℕ) : ℤ := u31 (82 + r) + u31 (260 + r) + u31 (418 + r) + u31 (552 + r) + u31 (721 + r) + u31 (775 + r)
def L31_5 (r : ℕ) : ℤ := u31 (116 + r) + u31 (296 + r) + u31 (458 + r) + u31 (594 + r) + u31 (692 + r) + u31 (829 + r)
def L31_6 (r : ℕ) : ℤ := u31 (156 + r) + u31 (338 + r) + u31 (504 + r) + u31 (642 + r) + u31 (744 + r) + u31 (798 + r)

def aS31_0 (r : ℕ) : ℤ := S31_0 r - L31_0 r
def MS31_0 : ℤ := CaseSplit.mxr (aS31_0) 10
def aS31_1 (r : ℕ) : ℤ := S31_1 r - L31_1 r
def MS31_1 : ℤ := CaseSplit.mxr (aS31_1) 12
def aS31_2 (r : ℕ) : ℤ := S31_2 r - L31_2 r
def MS31_2 : ℤ := CaseSplit.mxr (aS31_2) 16
def aS31_3 (r : ℕ) : ℤ := S31_3 r - L31_3 r
def MS31_3 : ℤ := CaseSplit.mxr (aS31_3) 18
def aS31_4 (r : ℕ) : ℤ := S31_4 r - L31_4 r
def MS31_4 : ℤ := CaseSplit.mxr (aS31_4) 22
def aS31_5 (r : ℕ) : ℤ := S31_5 r - L31_5 r
def MS31_5 : ℤ := CaseSplit.mxr (aS31_5) 28
def aS31_6 (r : ℕ) : ℤ := S31_6 r - L31_6 r
def MS31_6 : ℤ := CaseSplit.mxr (aS31_6) 30

def N31_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_0 ra t && c31_1 rb t then 1 else 0)
def aP31_0 (ra rb : ℕ) : ℤ := -(5) * N31_0 ra rb + u31 (0 + rb) + u31 (13 + ra)
def MP31_0 : ℤ := CaseSplit.mxr2 (aP31_0) 10 12
def N31_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_0 ra t && c31_2 rb t then 1 else 0)
def aP31_1 (ra rb : ℕ) : ℤ := -(5) * N31_1 ra rb + u31 (24 + rb) + u31 (41 + ra)
def MP31_1 : ℤ := CaseSplit.mxr2 (aP31_1) 10 16
def N31_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_0 ra t && c31_3 rb t then 1 else 0)
def aP31_2 (ra rb : ℕ) : ℤ := -(5) * N31_2 ra rb + u31 (52 + rb) + u31 (71 + ra)
def MP31_2 : ℤ := CaseSplit.mxr2 (aP31_2) 10 18
def N31_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_0 ra t && c31_4 rb t then 1 else 0)
def aP31_3 (ra rb : ℕ) : ℤ := -(5) * N31_3 ra rb + u31 (82 + rb) + u31 (105 + ra)
def MP31_3 : ℤ := CaseSplit.mxr2 (aP31_3) 10 22
def N31_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_0 ra t && c31_5 rb t then 1 else 0)
def aP31_4 (ra rb : ℕ) : ℤ := -(5) * N31_4 ra rb + u31 (116 + rb) + u31 (145 + ra)
def MP31_4 : ℤ := CaseSplit.mxr2 (aP31_4) 10 28
def N31_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_0 ra t && c31_6 rb t then 1 else 0)
def aP31_5 (ra rb : ℕ) : ℤ := -(5) * N31_5 ra rb + u31 (156 + rb) + u31 (187 + ra)
def MP31_5 : ℤ := CaseSplit.mxr2 (aP31_5) 10 30
def P31_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_1 ra t && c31_2 rb t then 1 else 0)
def C31_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_1 ra t && c31_2 rb t && c31_0 s t then 1 else 0)
def M31_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C31_6 ra rb) 10
def E31_6 : List ℕ := [3, 9, 18, 29, 54, 65, 70, 76, 93, 99, 129, 135, 138, 144, 160, 166, 172, 183, 196, 202]
def N31_6 (ra rb : ℕ) : ℤ := if E31_6.contains (ra * 17 + rb) = true then P31_6 ra rb - M31_6 ra rb else 0
def aP31_6 (ra rb : ℕ) : ℤ := -(5) * N31_6 ra rb + u31 (198 + rb) + u31 (215 + ra)
def MP31_6 : ℤ := CaseSplit.mxr2 (aP31_6) 12 16
def P31_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_1 ra t && c31_3 rb t then 1 else 0)
def C31_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_1 ra t && c31_3 rb t && c31_0 s t then 1 else 0)
def M31_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C31_7 ra rb) 10
def E31_7 : List ℕ := [13, 21, 24, 37, 58, 66, 71, 100, 113, 134, 147, 150, 158, 184, 192, 226, 234, 237]
def N31_7 (ra rb : ℕ) : ℤ := if E31_7.contains (ra * 19 + rb) = true then P31_7 ra rb - M31_7 ra rb else 0
def aP31_7 (ra rb : ℕ) : ℤ := -(5) * N31_7 ra rb + u31 (228 + rb) + u31 (247 + ra)
def MP31_7 : ℤ := CaseSplit.mxr2 (aP31_7) 12 18
def P31_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_1 ra t && c31_4 rb t then 1 else 0)
def C31_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_1 ra t && c31_4 rb t && c31_0 s t then 1 else 0)
def M31_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C31_8 ra rb) 10
def E31_8 : List ℕ := []
def N31_8 (ra rb : ℕ) : ℤ := if E31_8.contains (ra * 23 + rb) = true then P31_8 ra rb - M31_8 ra rb else 0
def aP31_8 (ra rb : ℕ) : ℤ := -(5) * N31_8 ra rb + u31 (260 + rb) + u31 (283 + ra)
def MP31_8 : ℤ := CaseSplit.mxr2 (aP31_8) 12 22
def P31_9 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_1 ra t && c31_5 rb t then 1 else 0)
def C31_9 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_1 ra t && c31_5 rb t && c31_0 s t then 1 else 0)
def M31_9 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C31_9 ra rb) 10
def E31_9 : List ℕ := [39, 73, 189, 223, 300, 339]
def N31_9 (ra rb : ℕ) : ℤ := if E31_9.contains (ra * 29 + rb) = true then P31_9 ra rb - M31_9 ra rb else 0
def aP31_9 (ra rb : ℕ) : ℤ := -(5) * N31_9 ra rb + u31 (296 + rb) + u31 (325 + ra)
def MP31_9 : ℤ := CaseSplit.mxr2 (aP31_9) 12 28
def P31_10 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_1 ra t && c31_6 rb t then 1 else 0)
def C31_10 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n31, (if c31_1 ra t && c31_6 rb t && c31_0 s t then 1 else 0)
def M31_10 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C31_10 ra rb) 10
def E31_10 : List ℕ := [25, 35, 149, 185, 218, 309, 314, 342]
def N31_10 (ra rb : ℕ) : ℤ := if E31_10.contains (ra * 31 + rb) = true then P31_10 ra rb - M31_10 ra rb else 0
def aP31_10 (ra rb : ℕ) : ℤ := -(5) * N31_10 ra rb + u31 (338 + rb) + u31 (369 + ra)
def MP31_10 : ℤ := CaseSplit.mxr2 (aP31_10) 12 30
def N31_11 (_ra _rb : ℕ) : ℤ := 0
def aP31_11 (ra rb : ℕ) : ℤ := -(5) * N31_11 ra rb + u31 (382 + rb) + u31 (401 + ra)
def MP31_11 : ℤ := CaseSplit.mxr2 (aP31_11) 16 18
def N31_12 (_ra _rb : ℕ) : ℤ := 0
def aP31_12 (ra rb : ℕ) : ℤ := -(5) * N31_12 ra rb + u31 (418 + rb) + u31 (441 + ra)
def MP31_12 : ℤ := CaseSplit.mxr2 (aP31_12) 16 22
def N31_13 (_ra _rb : ℕ) : ℤ := 0
def aP31_13 (ra rb : ℕ) : ℤ := -(5) * N31_13 ra rb + u31 (458 + rb) + u31 (487 + ra)
def MP31_13 : ℤ := CaseSplit.mxr2 (aP31_13) 16 28
def N31_14 (_ra _rb : ℕ) : ℤ := 0
def aP31_14 (ra rb : ℕ) : ℤ := -(5) * N31_14 ra rb + u31 (504 + rb) + u31 (535 + ra)
def MP31_14 : ℤ := CaseSplit.mxr2 (aP31_14) 16 30
def N31_15 (_ra _rb : ℕ) : ℤ := 0
def aP31_15 (ra rb : ℕ) : ℤ := -(5) * N31_15 ra rb + u31 (552 + rb) + u31 (575 + ra)
def MP31_15 : ℤ := CaseSplit.mxr2 (aP31_15) 18 22
def N31_16 (_ra _rb : ℕ) : ℤ := 0
def aP31_16 (ra rb : ℕ) : ℤ := -(5) * N31_16 ra rb + u31 (594 + rb) + u31 (623 + ra)
def MP31_16 : ℤ := CaseSplit.mxr2 (aP31_16) 18 28
def N31_17 (_ra _rb : ℕ) : ℤ := 0
def aP31_17 (ra rb : ℕ) : ℤ := -(5) * N31_17 ra rb + u31 (642 + rb) + u31 (673 + ra)
def MP31_17 : ℤ := CaseSplit.mxr2 (aP31_17) 18 30
def N31_18 (_ra _rb : ℕ) : ℤ := 0
def aP31_18 (ra rb : ℕ) : ℤ := -(5) * N31_18 ra rb + u31 (692 + rb) + u31 (721 + ra)
def MP31_18 : ℤ := CaseSplit.mxr2 (aP31_18) 22 28
def N31_19 (_ra _rb : ℕ) : ℤ := 0
def aP31_19 (ra rb : ℕ) : ℤ := -(5) * N31_19 ra rb + u31 (744 + rb) + u31 (775 + ra)
def MP31_19 : ℤ := CaseSplit.mxr2 (aP31_19) 22 30
def N31_20 (_ra _rb : ℕ) : ℤ := 0
def aP31_20 (ra rb : ℕ) : ℤ := -(5) * N31_20 ra rb + u31 (798 + rb) + u31 (829 + ra)
def MP31_20 : ℤ := CaseSplit.mxr2 (aP31_20) 28 30

def rhs31 : ℤ := (∑ t ∈ Finset.range n31, w31 t) + 5 * (n31 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn31 : ∀ t, t < n31 → (0 : ℤ) ≤ w31 t := by decide
theorem plt31 : ∀ t, t < n31 → q31 t < 65 := by decide
theorem pfree31_5 : ∀ t, t < n31 → gb5 4 (q31 t) = false := by decide
theorem pfree31_7 : ∀ t, t < n31 → gb7 3 (q31 t) = false := by decide
theorem MSv31_0 : MS31_0 = 35 := by decide +kernel
theorem MSv31_1 : MS31_1 = 133 := by decide +kernel
theorem MSv31_2 : MS31_2 = 1 := by decide +kernel
theorem MSv31_3 : MS31_3 = 2 := by decide +kernel
theorem MSv31_4 : MS31_4 = 1 := by decide +kernel
theorem MSv31_5 : MS31_5 = 1 := by decide +kernel
theorem MSv31_6 : MS31_6 = 1 := by decide +kernel
theorem MPv31_0 : MP31_0 = 0 := by decide +kernel
theorem MPv31_1 : MP31_1 = 0 := by decide +kernel
theorem MPv31_2 : MP31_2 = 0 := by decide +kernel
theorem MPv31_3 : MP31_3 = 0 := by decide +kernel
theorem MPv31_4 : MP31_4 = 0 := by decide +kernel
theorem MPv31_5 : MP31_5 = 0 := by decide +kernel
theorem MPv31_6 : MP31_6 = 0 := by decide +kernel
theorem MPv31_7 : MP31_7 = 0 := by decide +kernel
theorem MPv31_8 : MP31_8 = 0 := by decide +kernel
theorem MPv31_9 : MP31_9 = 0 := by decide +kernel
theorem MPv31_10 : MP31_10 = 0 := by decide +kernel
theorem MPv31_11 : MP31_11 = 0 := by decide +kernel
theorem MPv31_12 : MP31_12 = 0 := by decide +kernel
theorem MPv31_13 : MP31_13 = 0 := by decide +kernel
theorem MPv31_14 : MP31_14 = 0 := by decide +kernel
theorem MPv31_15 : MP31_15 = 0 := by decide +kernel
theorem MPv31_16 : MP31_16 = 0 := by decide +kernel
theorem MPv31_17 : MP31_17 = 0 := by decide +kernel
theorem MPv31_18 : MP31_18 = 0 := by decide +kernel
theorem MPv31_19 : MP31_19 = 0 := by decide +kernel
theorem MPv31_20 : MP31_20 = 26 := by decide +kernel
theorem rhsv31 : rhs31 = 201 := by decide +kernel

/-- **The case-31 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/7.
    (Scaled by the common denominator 7: 200 < 201.) -/
theorem cert31 : MS31_0 + MS31_1 + MS31_2 + MS31_3 + MS31_4 + MS31_5 + MS31_6 + MP31_0 + MP31_1 + MP31_2 + MP31_3 + MP31_4 + MP31_5 + MP31_6 + MP31_7 + MP31_8 + MP31_9 + MP31_10 + MP31_11 + MP31_12 + MP31_13 + MP31_14 + MP31_15 + MP31_16 + MP31_17 + MP31_18 + MP31_19 + MP31_20 < rhs31 := by
  rw [MSv31_0, MSv31_1, MSv31_2, MSv31_3, MSv31_4, MSv31_5, MSv31_6, MPv31_0, MPv31_1, MPv31_2, MPv31_3, MPv31_4, MPv31_5, MPv31_6, MPv31_7, MPv31_8, MPv31_9, MPv31_10, MPv31_11, MPv31_12, MPv31_13, MPv31_14, MPv31_15, MPv31_16, MPv31_17, MPv31_18, MPv31_19, MPv31_20, rhsv31]
  decide

def Dg31 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := (if c31_0 r0 t then 1 else 0) + (if c31_1 r1 t then 1 else 0) + (if c31_2 r2 t then 1 else 0) + (if c31_3 r3 t then 1 else 0) + (if c31_4 r4 t then 1 else 0) + (if c31_5 r5 t then 1 else 0) + (if c31_6 r6 t then 1 else 0)
def Wl31_0 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c31_0 r0 t && c31_1 r1 t then 1 else 0
def Wl31_1 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c31_0 r0 t && c31_2 r2 t then 1 else 0
def Wl31_2 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c31_0 r0 t && c31_3 r3 t then 1 else 0
def Wl31_3 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c31_0 r0 t && c31_4 r4 t then 1 else 0
def Wl31_4 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c31_0 r0 t && c31_5 r5 t then 1 else 0
def Wl31_5 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c31_0 r0 t && c31_6 r6 t then 1 else 0
def Wl31_6 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c31_0 r0 t && c31_1 r1 t && c31_2 r2 t then 1 else 0
def Wl31_7 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c31_0 r0 t && c31_1 r1 t && c31_3 r3 t then 1 else 0
def Wl31_8 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c31_0 r0 t && c31_1 r1 t && c31_4 r4 t then 1 else 0
def Wl31_9 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c31_0 r0 t && c31_1 r1 t && c31_5 r5 t then 1 else 0
def Wl31_10 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c31_0 r0 t && c31_1 r1 t && c31_6 r6 t then 1 else 0
def Wl31_11 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c31_0 r0 t && !c31_1 r1 t && c31_2 r2 t && c31_3 r3 t then 1 else 0
def Wl31_12 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c31_0 r0 t && !c31_1 r1 t && c31_2 r2 t && c31_4 r4 t then 1 else 0
def Wl31_13 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c31_0 r0 t && !c31_1 r1 t && c31_2 r2 t && c31_5 r5 t then 1 else 0
def Wl31_14 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c31_0 r0 t && !c31_1 r1 t && c31_2 r2 t && c31_6 r6 t then 1 else 0
def Wl31_15 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c31_0 r0 t && !c31_1 r1 t && !c31_2 r2 t && c31_3 r3 t && c31_4 r4 t then 1 else 0
def Wl31_16 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c31_0 r0 t && !c31_1 r1 t && !c31_2 r2 t && c31_3 r3 t && c31_5 r5 t then 1 else 0
def Wl31_17 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c31_0 r0 t && !c31_1 r1 t && !c31_2 r2 t && c31_3 r3 t && c31_6 r6 t then 1 else 0
def Wl31_18 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c31_0 r0 t && !c31_1 r1 t && !c31_2 r2 t && !c31_3 r3 t && c31_4 r4 t && c31_5 r5 t then 1 else 0
def Wl31_19 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c31_0 r0 t && !c31_1 r1 t && !c31_2 r2 t && !c31_3 r3 t && c31_4 r4 t && c31_6 r6 t then 1 else 0
def Wl31_20 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c31_0 r0 t && !c31_1 r1 t && !c31_2 r2 t && !c31_3 r3 t && !c31_4 r4 t && c31_5 r5 t && c31_6 r6 t then 1 else 0

/-- **No configuration blocks the whole window in case 31.** -/
theorem nocov31 {r0 r1 r2 r3 r4 r5 r6 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29) (h6 : r6 < 31)
    (hcov : ∀ t, t < n31 → (c31_0 r0 t || c31_1 r1 t || c31_2 r2 t || c31_3 r3 t || c31_4 r4 t || c31_5 r5 t || c31_6 r6 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n31, (1 : ℤ) + (Wl31_0 r0 r1 r2 r3 r4 r5 r6 t + Wl31_1 r0 r1 r2 r3 r4 r5 r6 t + Wl31_2 r0 r1 r2 r3 r4 r5 r6 t + Wl31_3 r0 r1 r2 r3 r4 r5 r6 t + Wl31_4 r0 r1 r2 r3 r4 r5 r6 t + Wl31_5 r0 r1 r2 r3 r4 r5 r6 t + Wl31_6 r0 r1 r2 r3 r4 r5 r6 t + Wl31_7 r0 r1 r2 r3 r4 r5 r6 t + Wl31_8 r0 r1 r2 r3 r4 r5 r6 t + Wl31_9 r0 r1 r2 r3 r4 r5 r6 t + Wl31_10 r0 r1 r2 r3 r4 r5 r6 t + Wl31_11 r0 r1 r2 r3 r4 r5 r6 t + Wl31_12 r0 r1 r2 r3 r4 r5 r6 t + Wl31_13 r0 r1 r2 r3 r4 r5 r6 t + Wl31_14 r0 r1 r2 r3 r4 r5 r6 t + Wl31_15 r0 r1 r2 r3 r4 r5 r6 t + Wl31_16 r0 r1 r2 r3 r4 r5 r6 t + Wl31_17 r0 r1 r2 r3 r4 r5 r6 t + Wl31_18 r0 r1 r2 r3 r4 r5 r6 t + Wl31_19 r0 r1 r2 r3 r4 r5 r6 t + Wl31_20 r0 r1 r2 r3 r4 r5 r6 t) ≤ Dg31 r0 r1 r2 r3 r4 r5 r6 t := by
    intro t ht
    simp only [Wl31_0, Wl31_1, Wl31_2, Wl31_3, Wl31_4, Wl31_5, Wl31_6, Wl31_7, Wl31_8, Wl31_9, Wl31_10, Wl31_11, Wl31_12, Wl31_13, Wl31_14, Wl31_15, Wl31_16, Wl31_17, Wl31_18, Wl31_19, Wl31_20, Dg31]
    exact CaseSplit.lowest7 _ _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n31, (1 : ℤ) ≤ Dg31 r0 r1 r2 r3 r4 r5 r6 t := by
    intro t ht
    simp only [Dg31]
    exact CaseSplit.degpos7 _ _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n31 : ℤ) + ((∑ t ∈ Finset.range n31, Wl31_0 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n31, Wl31_1 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n31, Wl31_2 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n31, Wl31_3 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n31, Wl31_4 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n31, Wl31_5 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n31, Wl31_6 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n31, Wl31_7 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n31, Wl31_8 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n31, Wl31_9 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n31, Wl31_10 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n31, Wl31_11 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n31, Wl31_12 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n31, Wl31_13 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n31, Wl31_14 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n31, Wl31_15 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n31, Wl31_16 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n31, Wl31_17 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n31, Wl31_18 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n31, Wl31_19 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n31, Wl31_20 r0 r1 r2 r3 r4 r5 r6 t)) ≤ ∑ t ∈ Finset.range n31, Dg31 r0 r1 r2 r3 r4 r5 r6 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N31_0 r0 r1 ≤ ∑ t ∈ Finset.range n31, Wl31_0 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N31_0, Wl31_0, le_refl]
  have hn1 : N31_1 r0 r2 ≤ ∑ t ∈ Finset.range n31, Wl31_1 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N31_1, Wl31_1, le_refl]
  have hn2 : N31_2 r0 r3 ≤ ∑ t ∈ Finset.range n31, Wl31_2 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N31_2, Wl31_2, le_refl]
  have hn3 : N31_3 r0 r4 ≤ ∑ t ∈ Finset.range n31, Wl31_3 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N31_3, Wl31_3, le_refl]
  have hn4 : N31_4 r0 r5 ≤ ∑ t ∈ Finset.range n31, Wl31_4 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N31_4, Wl31_4, le_refl]
  have hn5 : N31_5 r0 r6 ≤ ∑ t ∈ Finset.range n31, Wl31_5 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N31_5, Wl31_5, le_refl]
  have hn6 : N31_6 r1 r2 ≤ ∑ t ∈ Finset.range n31, Wl31_6 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n31, Wl31_6 r0 r1 r2 r3 r4 r5 r6 t
        = (if c31_1 r1 t && c31_2 r2 t then (1:ℤ) else 0)
          - (if c31_1 r1 t && c31_2 r2 t && c31_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl31_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n31, Wl31_6 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl31_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n31, Wl31_6 r0 r1 r2 r3 r4 r5 r6 t
        = P31_6 r1 r2 - C31_6 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P31_6, C31_6]
    have hm : C31_6 r1 r2 r0 ≤ M31_6 r1 r2 :=
      CaseSplit.le_mxr (C31_6 r1 r2) 10 r0 (by omega)
    simp only [N31_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N31_7 r1 r3 ≤ ∑ t ∈ Finset.range n31, Wl31_7 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n31, Wl31_7 r0 r1 r2 r3 r4 r5 r6 t
        = (if c31_1 r1 t && c31_3 r3 t then (1:ℤ) else 0)
          - (if c31_1 r1 t && c31_3 r3 t && c31_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl31_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n31, Wl31_7 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl31_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n31, Wl31_7 r0 r1 r2 r3 r4 r5 r6 t
        = P31_7 r1 r3 - C31_7 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P31_7, C31_7]
    have hm : C31_7 r1 r3 r0 ≤ M31_7 r1 r3 :=
      CaseSplit.le_mxr (C31_7 r1 r3) 10 r0 (by omega)
    simp only [N31_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N31_8 r1 r4 ≤ ∑ t ∈ Finset.range n31, Wl31_8 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n31, Wl31_8 r0 r1 r2 r3 r4 r5 r6 t
        = (if c31_1 r1 t && c31_4 r4 t then (1:ℤ) else 0)
          - (if c31_1 r1 t && c31_4 r4 t && c31_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl31_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n31, Wl31_8 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl31_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n31, Wl31_8 r0 r1 r2 r3 r4 r5 r6 t
        = P31_8 r1 r4 - C31_8 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P31_8, C31_8]
    have hm : C31_8 r1 r4 r0 ≤ M31_8 r1 r4 :=
      CaseSplit.le_mxr (C31_8 r1 r4) 10 r0 (by omega)
    simp only [N31_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N31_9 r1 r5 ≤ ∑ t ∈ Finset.range n31, Wl31_9 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n31, Wl31_9 r0 r1 r2 r3 r4 r5 r6 t
        = (if c31_1 r1 t && c31_5 r5 t then (1:ℤ) else 0)
          - (if c31_1 r1 t && c31_5 r5 t && c31_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl31_9]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n31, Wl31_9 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl31_9]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n31, Wl31_9 r0 r1 r2 r3 r4 r5 r6 t
        = P31_9 r1 r5 - C31_9 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P31_9, C31_9]
    have hm : C31_9 r1 r5 r0 ≤ M31_9 r1 r5 :=
      CaseSplit.le_mxr (C31_9 r1 r5) 10 r0 (by omega)
    simp only [N31_9]
    split
    · rw [hL]; omega
    · exact hnn
  have hn10 : N31_10 r1 r6 ≤ ∑ t ∈ Finset.range n31, Wl31_10 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n31, Wl31_10 r0 r1 r2 r3 r4 r5 r6 t
        = (if c31_1 r1 t && c31_6 r6 t then (1:ℤ) else 0)
          - (if c31_1 r1 t && c31_6 r6 t && c31_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl31_10]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n31, Wl31_10 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl31_10]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n31, Wl31_10 r0 r1 r2 r3 r4 r5 r6 t
        = P31_10 r1 r6 - C31_10 r1 r6 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P31_10, C31_10]
    have hm : C31_10 r1 r6 r0 ≤ M31_10 r1 r6 :=
      CaseSplit.le_mxr (C31_10 r1 r6) 10 r0 (by omega)
    simp only [N31_10]
    split
    · rw [hL]; omega
    · exact hnn
  have hn11 : N31_11 r2 r3 ≤ ∑ t ∈ Finset.range n31, Wl31_11 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N31_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl31_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N31_12 r2 r4 ≤ ∑ t ∈ Finset.range n31, Wl31_12 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N31_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl31_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N31_13 r2 r5 ≤ ∑ t ∈ Finset.range n31, Wl31_13 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N31_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl31_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N31_14 r2 r6 ≤ ∑ t ∈ Finset.range n31, Wl31_14 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N31_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl31_14]
    exact CaseSplit.ind_nonneg _
  have hn15 : N31_15 r3 r4 ≤ ∑ t ∈ Finset.range n31, Wl31_15 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N31_15]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl31_15]
    exact CaseSplit.ind_nonneg _
  have hn16 : N31_16 r3 r5 ≤ ∑ t ∈ Finset.range n31, Wl31_16 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N31_16]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl31_16]
    exact CaseSplit.ind_nonneg _
  have hn17 : N31_17 r3 r6 ≤ ∑ t ∈ Finset.range n31, Wl31_17 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N31_17]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl31_17]
    exact CaseSplit.ind_nonneg _
  have hn18 : N31_18 r4 r5 ≤ ∑ t ∈ Finset.range n31, Wl31_18 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N31_18]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl31_18]
    exact CaseSplit.ind_nonneg _
  have hn19 : N31_19 r4 r6 ≤ ∑ t ∈ Finset.range n31, Wl31_19 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N31_19]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl31_19]
    exact CaseSplit.ind_nonneg _
  have hn20 : N31_20 r5 r6 ≤ ∑ t ∈ Finset.range n31, Wl31_20 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N31_20]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl31_20]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n31, (w31 t + 5) * Dg31 r0 r1 r2 r3 r4 r5 r6 t = S31_0 r0 + S31_1 r1 + S31_2 r2 + S31_3 r3 + S31_4 r4 + S31_5 r5 + S31_6 r6 := by
    simp only [S31_0, S31_1, S31_2, S31_3, S31_4, S31_5, S31_6, Dg31, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n31, (w31 t + 5) * Dg31 r0 r1 r2 r3 r4 r5 r6 t
      = (∑ t ∈ Finset.range n31, w31 t * Dg31 r0 r1 r2 r3 r4 r5 r6 t)
        + 5 * (∑ t ∈ Finset.range n31, Dg31 r0 r1 r2 r3 r4 r5 r6 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n31, w31 t)
      ≤ ∑ t ∈ Finset.range n31, w31 t * Dg31 r0 r1 r2 r3 r4 r5 r6 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg31 r0 r1 r2 r3 r4 r5 r6 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w31 t := wnn31 t (Finset.mem_range.mp ht)
    calc w31 t = w31 t * 1 := (mul_one _).symm
      _ ≤ w31 t * Dg31 r0 r1 r2 r3 r4 r5 r6 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS31_0 r0 + aS31_1 r1 + aS31_2 r2 + aS31_3 r3 + aS31_4 r4 + aS31_5 r5 + aS31_6 r6) + (aP31_0 r0 r1 + aP31_1 r0 r2 + aP31_2 r0 r3 + aP31_3 r0 r4 + aP31_4 r0 r5 + aP31_5 r0 r6 + aP31_6 r1 r2 + aP31_7 r1 r3 + aP31_8 r1 r4 + aP31_9 r1 r5 + aP31_10 r1 r6 + aP31_11 r2 r3 + aP31_12 r2 r4 + aP31_13 r2 r5 + aP31_14 r2 r6 + aP31_15 r3 r4 + aP31_16 r3 r5 + aP31_17 r3 r6 + aP31_18 r4 r5 + aP31_19 r4 r6 + aP31_20 r5 r6) = (S31_0 r0 + S31_1 r1 + S31_2 r2 + S31_3 r3 + S31_4 r4 + S31_5 r5 + S31_6 r6) - 5 * (N31_0 r0 r1 + N31_1 r0 r2 + N31_2 r0 r3 + N31_3 r0 r4 + N31_4 r0 r5 + N31_5 r0 r6 + N31_6 r1 r2 + N31_7 r1 r3 + N31_8 r1 r4 + N31_9 r1 r5 + N31_10 r1 r6 + N31_11 r2 r3 + N31_12 r2 r4 + N31_13 r2 r5 + N31_14 r2 r6 + N31_15 r3 r4 + N31_16 r3 r5 + N31_17 r3 r6 + N31_18 r4 r5 + N31_19 r4 r6 + N31_20 r5 r6) := by
    simp only [aS31_0, aS31_1, aS31_2, aS31_3, aS31_4, aS31_5, aS31_6, aP31_0, aP31_1, aP31_2, aP31_3, aP31_4, aP31_5, aP31_6, aP31_7, aP31_8, aP31_9, aP31_10, aP31_11, aP31_12, aP31_13, aP31_14, aP31_15, aP31_16, aP31_17, aP31_18, aP31_19, aP31_20, L31_0, L31_1, L31_2, L31_3, L31_4, L31_5, L31_6]
    ring
  have bS0 : aS31_0 r0 ≤ MS31_0 := CaseSplit.le_mxr (aS31_0) 10 r0 (by omega)
  have bS1 : aS31_1 r1 ≤ MS31_1 := CaseSplit.le_mxr (aS31_1) 12 r1 (by omega)
  have bS2 : aS31_2 r2 ≤ MS31_2 := CaseSplit.le_mxr (aS31_2) 16 r2 (by omega)
  have bS3 : aS31_3 r3 ≤ MS31_3 := CaseSplit.le_mxr (aS31_3) 18 r3 (by omega)
  have bS4 : aS31_4 r4 ≤ MS31_4 := CaseSplit.le_mxr (aS31_4) 22 r4 (by omega)
  have bS5 : aS31_5 r5 ≤ MS31_5 := CaseSplit.le_mxr (aS31_5) 28 r5 (by omega)
  have bS6 : aS31_6 r6 ≤ MS31_6 := CaseSplit.le_mxr (aS31_6) 30 r6 (by omega)
  have bP0 : aP31_0 r0 r1 ≤ MP31_0 := CaseSplit.le_mxr2 (aP31_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP31_1 r0 r2 ≤ MP31_1 := CaseSplit.le_mxr2 (aP31_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP31_2 r0 r3 ≤ MP31_2 := CaseSplit.le_mxr2 (aP31_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP31_3 r0 r4 ≤ MP31_3 := CaseSplit.le_mxr2 (aP31_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP31_4 r0 r5 ≤ MP31_4 := CaseSplit.le_mxr2 (aP31_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP31_5 r0 r6 ≤ MP31_5 := CaseSplit.le_mxr2 (aP31_5) 10 30 r0 r6 (by omega) (by omega)
  have bP6 : aP31_6 r1 r2 ≤ MP31_6 := CaseSplit.le_mxr2 (aP31_6) 12 16 r1 r2 (by omega) (by omega)
  have bP7 : aP31_7 r1 r3 ≤ MP31_7 := CaseSplit.le_mxr2 (aP31_7) 12 18 r1 r3 (by omega) (by omega)
  have bP8 : aP31_8 r1 r4 ≤ MP31_8 := CaseSplit.le_mxr2 (aP31_8) 12 22 r1 r4 (by omega) (by omega)
  have bP9 : aP31_9 r1 r5 ≤ MP31_9 := CaseSplit.le_mxr2 (aP31_9) 12 28 r1 r5 (by omega) (by omega)
  have bP10 : aP31_10 r1 r6 ≤ MP31_10 := CaseSplit.le_mxr2 (aP31_10) 12 30 r1 r6 (by omega) (by omega)
  have bP11 : aP31_11 r2 r3 ≤ MP31_11 := CaseSplit.le_mxr2 (aP31_11) 16 18 r2 r3 (by omega) (by omega)
  have bP12 : aP31_12 r2 r4 ≤ MP31_12 := CaseSplit.le_mxr2 (aP31_12) 16 22 r2 r4 (by omega) (by omega)
  have bP13 : aP31_13 r2 r5 ≤ MP31_13 := CaseSplit.le_mxr2 (aP31_13) 16 28 r2 r5 (by omega) (by omega)
  have bP14 : aP31_14 r2 r6 ≤ MP31_14 := CaseSplit.le_mxr2 (aP31_14) 16 30 r2 r6 (by omega) (by omega)
  have bP15 : aP31_15 r3 r4 ≤ MP31_15 := CaseSplit.le_mxr2 (aP31_15) 18 22 r3 r4 (by omega) (by omega)
  have bP16 : aP31_16 r3 r5 ≤ MP31_16 := CaseSplit.le_mxr2 (aP31_16) 18 28 r3 r5 (by omega) (by omega)
  have bP17 : aP31_17 r3 r6 ≤ MP31_17 := CaseSplit.le_mxr2 (aP31_17) 18 30 r3 r6 (by omega) (by omega)
  have bP18 : aP31_18 r4 r5 ≤ MP31_18 := CaseSplit.le_mxr2 (aP31_18) 22 28 r4 r5 (by omega) (by omega)
  have bP19 : aP31_19 r4 r6 ≤ MP31_19 := CaseSplit.le_mxr2 (aP31_19) 22 30 r4 r6 (by omega) (by omega)
  have bP20 : aP31_20 r5 r6 ≤ MP31_20 := CaseSplit.le_mxr2 (aP31_20) 28 30 r5 r6 (by omega) (by omega)
  have hrhs : rhs31 = (∑ t ∈ Finset.range n31, w31 t) + 5 * (n31 : ℤ) := rfl
  have hc := cert31
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, hn15, hn16, hn17, hn18, hn19, hn20, bS0, bS1, bS2, bS3, bS4, bS5, bS6, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14, bP15, bP16, bP17, bP18, bP19, bP20]

end IncCert31
