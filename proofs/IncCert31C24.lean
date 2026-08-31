/-
INCREMENT-WIDTH CERTIFICATE, step 29->31, case 24 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_29_31.json, which re-derives every number
from the primes alone).

Machine 31, INCREMENT width 65 = F_2(29) + s_min(31) = 55 + 10,
held gears [5, 7] at phases [3, 3].  Free gears [11, 13, 17, 19, 23, 29, 31].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 14.
-/
import IncCert31B

namespace IncCert31

/-! ### case 24: held gears at phases [3, 3] -/

def p24 : List ℕ := [0, 2, 4, 7, 9, 14, 15, 20, 22, 25, 27, 29, 30, 32, 34, 35, 37, 39, 42, 44, 49, 50, 55, 57, 60, 62, 64]
def q24 (t : ℕ) : ℕ := p24.getD t 0
def n24 : ℕ := 27
def yl24 : List ℤ := [2, 9, 0, 8, 6, 9, 0, 4, 6, 10, 10, 6, 14, 13, 14, 6, 10, 10, 6, 4, 0, 9, 6, 8, 0, 9, 2]
def w24 (t : ℕ) : ℤ := yl24.getD t 0
def ul24 : List ℤ := [(-5), (-5), 0, (-5), (-2), (-5), 0, (-5), 0, (-5), (-3), (-5), 0, 5, 0, 5, 0, (-5), 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-12), 0, 0, 0, 0, (-18), (-8), 0, (-3), (-3), (-3), (-3), (-3), (-3), 0, (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), 0, 0, (-3), (-3), 3, 0, 3, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 33, 18, 33, 31, 33, 33, 32, 19, 33, 33, 33, 33, 19, 33, 33, 33, (-33), (-33), (-33), (-33), (-33), (-33), (-33), (-33), (-33), (-33), (-33), (-33), (-33), 17, 27, 26, 27, 20, 27, 11, 24, 20, 27, 26, 27, 17, 27, 27, 27, 27, 20, 27, (-27), (-27), (-27), (-27), (-27), (-27), (-27), (-27), (-27), (-27), (-27), (-27), (-27), 43, 14, 43, 43, 14, 43, 13, 23, 43, 12, 40, 31, 37, 43, 7, 43, 43, 43, 40, 43, 43, 23, 13, (-43), (-43), (-43), (-43), (-43), (-43), (-43), (-43), (-43), (-43), (-43), (-43), (-43), 17, 14, 25, 26, 26, 26, 26, 13, 9, 26, 12, 9, 9, 12, 22, 9, 13, 10, 14, 26, 26, 25, 26, 17, 26, 26, 25, 26, 26, (-26), (-26), (-26), (-26), (-26), (-26), (-26), (-26), (-26), (-26), (-26), (-26), (-26), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-16), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-7), (-7), (-19), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), 7, 7, 7, 7, 7, 4, 7, 7, 7, 7, 7, 7, 7, (-4), 0, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-12), 0, 0, 0, 0, 0, 10, 24, 25, 25, 25, 16, 25, 25, 3, 13, 10, 15, 20, 7, 9, 9, 7, 25, 15, 10, 13, 3, 24, 25, 25, 25, 16, 25, 24, 10, 25, 0, 0, 0, 0, 0, (-12), 0, 0, 0, (-4), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-12), 0, 0, (-7), 0, 0, 0]
def u24 (k : ℕ) : ℤ := ul24.getD k 0

def c24_0 (r t : ℕ) : Bool := gb11 r (q24 t)
def c24_1 (r t : ℕ) : Bool := gb13 r (q24 t)
def c24_2 (r t : ℕ) : Bool := gb17 r (q24 t)
def c24_3 (r t : ℕ) : Bool := gb19 r (q24 t)
def c24_4 (r t : ℕ) : Bool := gb23 r (q24 t)
def c24_5 (r t : ℕ) : Bool := gb29 r (q24 t)
def c24_6 (r t : ℕ) : Bool := gb31 r (q24 t)

def S24_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (w24 t + 3) * (if c24_0 r t then 1 else 0)
def S24_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (w24 t + 3) * (if c24_1 r t then 1 else 0)
def S24_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (w24 t + 3) * (if c24_2 r t then 1 else 0)
def S24_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (w24 t + 3) * (if c24_3 r t then 1 else 0)
def S24_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (w24 t + 3) * (if c24_4 r t then 1 else 0)
def S24_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (w24 t + 3) * (if c24_5 r t then 1 else 0)
def S24_6 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (w24 t + 3) * (if c24_6 r t then 1 else 0)

def L24_0 (r : ℕ) : ℤ := u24 (13 + r) + u24 (41 + r) + u24 (71 + r) + u24 (105 + r) + u24 (145 + r) + u24 (187 + r)
def L24_1 (r : ℕ) : ℤ := u24 (0 + r) + u24 (215 + r) + u24 (247 + r) + u24 (283 + r) + u24 (325 + r) + u24 (369 + r)
def L24_2 (r : ℕ) : ℤ := u24 (24 + r) + u24 (198 + r) + u24 (401 + r) + u24 (441 + r) + u24 (487 + r) + u24 (535 + r)
def L24_3 (r : ℕ) : ℤ := u24 (52 + r) + u24 (228 + r) + u24 (382 + r) + u24 (575 + r) + u24 (623 + r) + u24 (673 + r)
def L24_4 (r : ℕ) : ℤ := u24 (82 + r) + u24 (260 + r) + u24 (418 + r) + u24 (552 + r) + u24 (721 + r) + u24 (775 + r)
def L24_5 (r : ℕ) : ℤ := u24 (116 + r) + u24 (296 + r) + u24 (458 + r) + u24 (594 + r) + u24 (692 + r) + u24 (829 + r)
def L24_6 (r : ℕ) : ℤ := u24 (156 + r) + u24 (338 + r) + u24 (504 + r) + u24 (642 + r) + u24 (744 + r) + u24 (798 + r)

def aS24_0 (r : ℕ) : ℤ := S24_0 r - L24_0 r
def MS24_0 : ℤ := CaseSplit.mxr (aS24_0) 10
def aS24_1 (r : ℕ) : ℤ := S24_1 r - L24_1 r
def MS24_1 : ℤ := CaseSplit.mxr (aS24_1) 12
def aS24_2 (r : ℕ) : ℤ := S24_2 r - L24_2 r
def MS24_2 : ℤ := CaseSplit.mxr (aS24_2) 16
def aS24_3 (r : ℕ) : ℤ := S24_3 r - L24_3 r
def MS24_3 : ℤ := CaseSplit.mxr (aS24_3) 18
def aS24_4 (r : ℕ) : ℤ := S24_4 r - L24_4 r
def MS24_4 : ℤ := CaseSplit.mxr (aS24_4) 22
def aS24_5 (r : ℕ) : ℤ := S24_5 r - L24_5 r
def MS24_5 : ℤ := CaseSplit.mxr (aS24_5) 28
def aS24_6 (r : ℕ) : ℤ := S24_6 r - L24_6 r
def MS24_6 : ℤ := CaseSplit.mxr (aS24_6) 30

def N24_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (if c24_0 ra t && c24_1 rb t then 1 else 0)
def aP24_0 (ra rb : ℕ) : ℤ := -(3) * N24_0 ra rb + u24 (0 + rb) + u24 (13 + ra)
def MP24_0 : ℤ := CaseSplit.mxr2 (aP24_0) 10 12
def N24_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (if c24_0 ra t && c24_2 rb t then 1 else 0)
def aP24_1 (ra rb : ℕ) : ℤ := -(3) * N24_1 ra rb + u24 (24 + rb) + u24 (41 + ra)
def MP24_1 : ℤ := CaseSplit.mxr2 (aP24_1) 10 16
def N24_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (if c24_0 ra t && c24_3 rb t then 1 else 0)
def aP24_2 (ra rb : ℕ) : ℤ := -(3) * N24_2 ra rb + u24 (52 + rb) + u24 (71 + ra)
def MP24_2 : ℤ := CaseSplit.mxr2 (aP24_2) 10 18
def N24_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (if c24_0 ra t && c24_4 rb t then 1 else 0)
def aP24_3 (ra rb : ℕ) : ℤ := -(3) * N24_3 ra rb + u24 (82 + rb) + u24 (105 + ra)
def MP24_3 : ℤ := CaseSplit.mxr2 (aP24_3) 10 22
def N24_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (if c24_0 ra t && c24_5 rb t then 1 else 0)
def aP24_4 (ra rb : ℕ) : ℤ := -(3) * N24_4 ra rb + u24 (116 + rb) + u24 (145 + ra)
def MP24_4 : ℤ := CaseSplit.mxr2 (aP24_4) 10 28
def N24_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (if c24_0 ra t && c24_6 rb t then 1 else 0)
def aP24_5 (ra rb : ℕ) : ℤ := -(3) * N24_5 ra rb + u24 (156 + rb) + u24 (187 + ra)
def MP24_5 : ℤ := CaseSplit.mxr2 (aP24_5) 10 30
def P24_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (if c24_1 ra t && c24_2 rb t then 1 else 0)
def C24_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (if c24_1 ra t && c24_2 rb t && c24_0 s t then 1 else 0)
def M24_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C24_6 ra rb) 10
def E24_6 : List ℕ := [43, 49, 68, 79, 90, 101, 158, 169, 174, 180, 210, 216]
def N24_6 (ra rb : ℕ) : ℤ := if E24_6.contains (ra * 17 + rb) = true then P24_6 ra rb - M24_6 ra rb else 0
def aP24_6 (ra rb : ℕ) : ℤ := -(3) * N24_6 ra rb + u24 (198 + rb) + u24 (215 + ra)
def MP24_6 : ℤ := CaseSplit.mxr2 (aP24_6) 12 16
def P24_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (if c24_1 ra t && c24_3 rb t then 1 else 0)
def C24_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (if c24_1 ra t && c24_3 rb t && c24_0 s t then 1 else 0)
def M24_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C24_7 ra rb) 10
def E24_7 : List ℕ := [1, 4, 27, 30, 38, 51, 80, 91, 106, 114, 127, 130, 164, 167, 170, 172, 198, 206, 240, 246]
def N24_7 (ra rb : ℕ) : ℤ := if E24_7.contains (ra * 19 + rb) = true then P24_7 ra rb - M24_7 ra rb else 0
def aP24_7 (ra rb : ℕ) : ℤ := -(3) * N24_7 ra rb + u24 (228 + rb) + u24 (247 + ra)
def MP24_7 : ℤ := CaseSplit.mxr2 (aP24_7) 12 18
def P24_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (if c24_1 ra t && c24_4 rb t then 1 else 0)
def C24_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (if c24_1 ra t && c24_4 rb t && c24_0 s t then 1 else 0)
def M24_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C24_8 ra rb) 10
def E24_8 : List ℕ := []
def N24_8 (ra rb : ℕ) : ℤ := if E24_8.contains (ra * 23 + rb) = true then P24_8 ra rb - M24_8 ra rb else 0
def aP24_8 (ra rb : ℕ) : ℤ := -(3) * N24_8 ra rb + u24 (260 + rb) + u24 (283 + ra)
def MP24_8 : ℤ := CaseSplit.mxr2 (aP24_8) 12 22
def P24_9 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (if c24_1 ra t && c24_5 rb t then 1 else 0)
def C24_9 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (if c24_1 ra t && c24_5 rb t && c24_0 s t then 1 else 0)
def M24_9 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C24_9 ra rb) 10
def E24_9 : List ℕ := [82, 115, 343, 376]
def N24_9 (ra rb : ℕ) : ℤ := if E24_9.contains (ra * 29 + rb) = true then P24_9 ra rb - M24_9 ra rb else 0
def aP24_9 (ra rb : ℕ) : ℤ := -(3) * N24_9 ra rb + u24 (296 + rb) + u24 (325 + ra)
def MP24_9 : ℤ := CaseSplit.mxr2 (aP24_9) 12 28
def P24_10 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (if c24_1 ra t && c24_6 rb t then 1 else 0)
def C24_10 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (if c24_1 ra t && c24_6 rb t && c24_0 s t then 1 else 0)
def M24_10 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C24_10 ra rb) 10
def E24_10 : List ℕ := []
def N24_10 (ra rb : ℕ) : ℤ := if E24_10.contains (ra * 31 + rb) = true then P24_10 ra rb - M24_10 ra rb else 0
def aP24_10 (ra rb : ℕ) : ℤ := -(3) * N24_10 ra rb + u24 (338 + rb) + u24 (369 + ra)
def MP24_10 : ℤ := CaseSplit.mxr2 (aP24_10) 12 30
def N24_11 (_ra _rb : ℕ) : ℤ := 0
def aP24_11 (ra rb : ℕ) : ℤ := -(3) * N24_11 ra rb + u24 (382 + rb) + u24 (401 + ra)
def MP24_11 : ℤ := CaseSplit.mxr2 (aP24_11) 16 18
def N24_12 (_ra _rb : ℕ) : ℤ := 0
def aP24_12 (ra rb : ℕ) : ℤ := -(3) * N24_12 ra rb + u24 (418 + rb) + u24 (441 + ra)
def MP24_12 : ℤ := CaseSplit.mxr2 (aP24_12) 16 22
def N24_13 (_ra _rb : ℕ) : ℤ := 0
def aP24_13 (ra rb : ℕ) : ℤ := -(3) * N24_13 ra rb + u24 (458 + rb) + u24 (487 + ra)
def MP24_13 : ℤ := CaseSplit.mxr2 (aP24_13) 16 28
def N24_14 (_ra _rb : ℕ) : ℤ := 0
def aP24_14 (ra rb : ℕ) : ℤ := -(3) * N24_14 ra rb + u24 (504 + rb) + u24 (535 + ra)
def MP24_14 : ℤ := CaseSplit.mxr2 (aP24_14) 16 30
def N24_15 (_ra _rb : ℕ) : ℤ := 0
def aP24_15 (ra rb : ℕ) : ℤ := -(3) * N24_15 ra rb + u24 (552 + rb) + u24 (575 + ra)
def MP24_15 : ℤ := CaseSplit.mxr2 (aP24_15) 18 22
def N24_16 (_ra _rb : ℕ) : ℤ := 0
def aP24_16 (ra rb : ℕ) : ℤ := -(3) * N24_16 ra rb + u24 (594 + rb) + u24 (623 + ra)
def MP24_16 : ℤ := CaseSplit.mxr2 (aP24_16) 18 28
def N24_17 (_ra _rb : ℕ) : ℤ := 0
def aP24_17 (ra rb : ℕ) : ℤ := -(3) * N24_17 ra rb + u24 (642 + rb) + u24 (673 + ra)
def MP24_17 : ℤ := CaseSplit.mxr2 (aP24_17) 18 30
def N24_18 (_ra _rb : ℕ) : ℤ := 0
def aP24_18 (ra rb : ℕ) : ℤ := -(3) * N24_18 ra rb + u24 (692 + rb) + u24 (721 + ra)
def MP24_18 : ℤ := CaseSplit.mxr2 (aP24_18) 22 28
def N24_19 (_ra _rb : ℕ) : ℤ := 0
def aP24_19 (ra rb : ℕ) : ℤ := -(3) * N24_19 ra rb + u24 (744 + rb) + u24 (775 + ra)
def MP24_19 : ℤ := CaseSplit.mxr2 (aP24_19) 22 30
def N24_20 (_ra _rb : ℕ) : ℤ := 0
def aP24_20 (ra rb : ℕ) : ℤ := -(3) * N24_20 ra rb + u24 (798 + rb) + u24 (829 + ra)
def MP24_20 : ℤ := CaseSplit.mxr2 (aP24_20) 28 30

def rhs24 : ℤ := (∑ t ∈ Finset.range n24, w24 t) + 3 * (n24 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn24 : ∀ t, t < n24 → (0 : ℤ) ≤ w24 t := by decide
theorem plt24 : ∀ t, t < n24 → q24 t < 65 := by decide
theorem pfree24_5 : ∀ t, t < n24 → gb5 3 (q24 t) = false := by decide
theorem pfree24_7 : ∀ t, t < n24 → gb7 3 (q24 t) = false := by decide
theorem MSv24_0 : MS24_0 = 54 := by decide +kernel
theorem MSv24_1 : MS24_1 = 175 := by decide +kernel
theorem MSv24_2 : MS24_2 = 1 := by decide +kernel
theorem MSv24_3 : MS24_3 = 2 := by decide +kernel
theorem MSv24_4 : MS24_4 = 1 := by decide +kernel
theorem MSv24_5 : MS24_5 = 1 := by decide +kernel
theorem MSv24_6 : MS24_6 = 1 := by decide +kernel
theorem MPv24_0 : MP24_0 = 0 := by decide +kernel
theorem MPv24_1 : MP24_1 = 0 := by decide +kernel
theorem MPv24_2 : MP24_2 = 0 := by decide +kernel
theorem MPv24_3 : MP24_3 = 0 := by decide +kernel
theorem MPv24_4 : MP24_4 = 0 := by decide +kernel
theorem MPv24_5 : MP24_5 = 0 := by decide +kernel
theorem MPv24_6 : MP24_6 = 0 := by decide +kernel
theorem MPv24_7 : MP24_7 = 0 := by decide +kernel
theorem MPv24_8 : MP24_8 = 0 := by decide +kernel
theorem MPv24_9 : MP24_9 = 0 := by decide +kernel
theorem MPv24_10 : MP24_10 = 0 := by decide +kernel
theorem MPv24_11 : MP24_11 = 0 := by decide +kernel
theorem MPv24_12 : MP24_12 = 0 := by decide +kernel
theorem MPv24_13 : MP24_13 = 0 := by decide +kernel
theorem MPv24_14 : MP24_14 = 0 := by decide +kernel
theorem MPv24_15 : MP24_15 = 0 := by decide +kernel
theorem MPv24_16 : MP24_16 = 0 := by decide +kernel
theorem MPv24_17 : MP24_17 = 0 := by decide +kernel
theorem MPv24_18 : MP24_18 = 0 := by decide +kernel
theorem MPv24_19 : MP24_19 = 0 := by decide +kernel
theorem MPv24_20 : MP24_20 = 25 := by decide +kernel
theorem rhsv24 : rhs24 = 262 := by decide +kernel

/-- **The case-24 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 2/14.
    (Scaled by the common denominator 14: 260 < 262.) -/
theorem cert24 : MS24_0 + MS24_1 + MS24_2 + MS24_3 + MS24_4 + MS24_5 + MS24_6 + MP24_0 + MP24_1 + MP24_2 + MP24_3 + MP24_4 + MP24_5 + MP24_6 + MP24_7 + MP24_8 + MP24_9 + MP24_10 + MP24_11 + MP24_12 + MP24_13 + MP24_14 + MP24_15 + MP24_16 + MP24_17 + MP24_18 + MP24_19 + MP24_20 < rhs24 := by
  rw [MSv24_0, MSv24_1, MSv24_2, MSv24_3, MSv24_4, MSv24_5, MSv24_6, MPv24_0, MPv24_1, MPv24_2, MPv24_3, MPv24_4, MPv24_5, MPv24_6, MPv24_7, MPv24_8, MPv24_9, MPv24_10, MPv24_11, MPv24_12, MPv24_13, MPv24_14, MPv24_15, MPv24_16, MPv24_17, MPv24_18, MPv24_19, MPv24_20, rhsv24]
  decide

def Dg24 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := (if c24_0 r0 t then 1 else 0) + (if c24_1 r1 t then 1 else 0) + (if c24_2 r2 t then 1 else 0) + (if c24_3 r3 t then 1 else 0) + (if c24_4 r4 t then 1 else 0) + (if c24_5 r5 t then 1 else 0) + (if c24_6 r6 t then 1 else 0)
def Wl24_0 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c24_0 r0 t && c24_1 r1 t then 1 else 0
def Wl24_1 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c24_0 r0 t && c24_2 r2 t then 1 else 0
def Wl24_2 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c24_0 r0 t && c24_3 r3 t then 1 else 0
def Wl24_3 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c24_0 r0 t && c24_4 r4 t then 1 else 0
def Wl24_4 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c24_0 r0 t && c24_5 r5 t then 1 else 0
def Wl24_5 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c24_0 r0 t && c24_6 r6 t then 1 else 0
def Wl24_6 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c24_0 r0 t && c24_1 r1 t && c24_2 r2 t then 1 else 0
def Wl24_7 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c24_0 r0 t && c24_1 r1 t && c24_3 r3 t then 1 else 0
def Wl24_8 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c24_0 r0 t && c24_1 r1 t && c24_4 r4 t then 1 else 0
def Wl24_9 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c24_0 r0 t && c24_1 r1 t && c24_5 r5 t then 1 else 0
def Wl24_10 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c24_0 r0 t && c24_1 r1 t && c24_6 r6 t then 1 else 0
def Wl24_11 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c24_0 r0 t && !c24_1 r1 t && c24_2 r2 t && c24_3 r3 t then 1 else 0
def Wl24_12 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c24_0 r0 t && !c24_1 r1 t && c24_2 r2 t && c24_4 r4 t then 1 else 0
def Wl24_13 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c24_0 r0 t && !c24_1 r1 t && c24_2 r2 t && c24_5 r5 t then 1 else 0
def Wl24_14 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c24_0 r0 t && !c24_1 r1 t && c24_2 r2 t && c24_6 r6 t then 1 else 0
def Wl24_15 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c24_0 r0 t && !c24_1 r1 t && !c24_2 r2 t && c24_3 r3 t && c24_4 r4 t then 1 else 0
def Wl24_16 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c24_0 r0 t && !c24_1 r1 t && !c24_2 r2 t && c24_3 r3 t && c24_5 r5 t then 1 else 0
def Wl24_17 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c24_0 r0 t && !c24_1 r1 t && !c24_2 r2 t && c24_3 r3 t && c24_6 r6 t then 1 else 0
def Wl24_18 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c24_0 r0 t && !c24_1 r1 t && !c24_2 r2 t && !c24_3 r3 t && c24_4 r4 t && c24_5 r5 t then 1 else 0
def Wl24_19 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c24_0 r0 t && !c24_1 r1 t && !c24_2 r2 t && !c24_3 r3 t && c24_4 r4 t && c24_6 r6 t then 1 else 0
def Wl24_20 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c24_0 r0 t && !c24_1 r1 t && !c24_2 r2 t && !c24_3 r3 t && !c24_4 r4 t && c24_5 r5 t && c24_6 r6 t then 1 else 0

/-- **No configuration blocks the whole window in case 24.** -/
theorem nocov24 {r0 r1 r2 r3 r4 r5 r6 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29) (h6 : r6 < 31)
    (hcov : ∀ t, t < n24 → (c24_0 r0 t || c24_1 r1 t || c24_2 r2 t || c24_3 r3 t || c24_4 r4 t || c24_5 r5 t || c24_6 r6 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n24, (1 : ℤ) + (Wl24_0 r0 r1 r2 r3 r4 r5 r6 t + Wl24_1 r0 r1 r2 r3 r4 r5 r6 t + Wl24_2 r0 r1 r2 r3 r4 r5 r6 t + Wl24_3 r0 r1 r2 r3 r4 r5 r6 t + Wl24_4 r0 r1 r2 r3 r4 r5 r6 t + Wl24_5 r0 r1 r2 r3 r4 r5 r6 t + Wl24_6 r0 r1 r2 r3 r4 r5 r6 t + Wl24_7 r0 r1 r2 r3 r4 r5 r6 t + Wl24_8 r0 r1 r2 r3 r4 r5 r6 t + Wl24_9 r0 r1 r2 r3 r4 r5 r6 t + Wl24_10 r0 r1 r2 r3 r4 r5 r6 t + Wl24_11 r0 r1 r2 r3 r4 r5 r6 t + Wl24_12 r0 r1 r2 r3 r4 r5 r6 t + Wl24_13 r0 r1 r2 r3 r4 r5 r6 t + Wl24_14 r0 r1 r2 r3 r4 r5 r6 t + Wl24_15 r0 r1 r2 r3 r4 r5 r6 t + Wl24_16 r0 r1 r2 r3 r4 r5 r6 t + Wl24_17 r0 r1 r2 r3 r4 r5 r6 t + Wl24_18 r0 r1 r2 r3 r4 r5 r6 t + Wl24_19 r0 r1 r2 r3 r4 r5 r6 t + Wl24_20 r0 r1 r2 r3 r4 r5 r6 t) ≤ Dg24 r0 r1 r2 r3 r4 r5 r6 t := by
    intro t ht
    simp only [Wl24_0, Wl24_1, Wl24_2, Wl24_3, Wl24_4, Wl24_5, Wl24_6, Wl24_7, Wl24_8, Wl24_9, Wl24_10, Wl24_11, Wl24_12, Wl24_13, Wl24_14, Wl24_15, Wl24_16, Wl24_17, Wl24_18, Wl24_19, Wl24_20, Dg24]
    exact CaseSplit.lowest7 _ _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n24, (1 : ℤ) ≤ Dg24 r0 r1 r2 r3 r4 r5 r6 t := by
    intro t ht
    simp only [Dg24]
    exact CaseSplit.degpos7 _ _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n24 : ℤ) + ((∑ t ∈ Finset.range n24, Wl24_0 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n24, Wl24_1 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n24, Wl24_2 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n24, Wl24_3 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n24, Wl24_4 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n24, Wl24_5 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n24, Wl24_6 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n24, Wl24_7 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n24, Wl24_8 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n24, Wl24_9 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n24, Wl24_10 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n24, Wl24_11 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n24, Wl24_12 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n24, Wl24_13 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n24, Wl24_14 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n24, Wl24_15 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n24, Wl24_16 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n24, Wl24_17 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n24, Wl24_18 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n24, Wl24_19 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n24, Wl24_20 r0 r1 r2 r3 r4 r5 r6 t)) ≤ ∑ t ∈ Finset.range n24, Dg24 r0 r1 r2 r3 r4 r5 r6 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N24_0 r0 r1 ≤ ∑ t ∈ Finset.range n24, Wl24_0 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N24_0, Wl24_0, le_refl]
  have hn1 : N24_1 r0 r2 ≤ ∑ t ∈ Finset.range n24, Wl24_1 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N24_1, Wl24_1, le_refl]
  have hn2 : N24_2 r0 r3 ≤ ∑ t ∈ Finset.range n24, Wl24_2 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N24_2, Wl24_2, le_refl]
  have hn3 : N24_3 r0 r4 ≤ ∑ t ∈ Finset.range n24, Wl24_3 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N24_3, Wl24_3, le_refl]
  have hn4 : N24_4 r0 r5 ≤ ∑ t ∈ Finset.range n24, Wl24_4 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N24_4, Wl24_4, le_refl]
  have hn5 : N24_5 r0 r6 ≤ ∑ t ∈ Finset.range n24, Wl24_5 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N24_5, Wl24_5, le_refl]
  have hn6 : N24_6 r1 r2 ≤ ∑ t ∈ Finset.range n24, Wl24_6 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n24, Wl24_6 r0 r1 r2 r3 r4 r5 r6 t
        = (if c24_1 r1 t && c24_2 r2 t then (1:ℤ) else 0)
          - (if c24_1 r1 t && c24_2 r2 t && c24_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl24_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n24, Wl24_6 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl24_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n24, Wl24_6 r0 r1 r2 r3 r4 r5 r6 t
        = P24_6 r1 r2 - C24_6 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P24_6, C24_6]
    have hm : C24_6 r1 r2 r0 ≤ M24_6 r1 r2 :=
      CaseSplit.le_mxr (C24_6 r1 r2) 10 r0 (by omega)
    simp only [N24_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N24_7 r1 r3 ≤ ∑ t ∈ Finset.range n24, Wl24_7 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n24, Wl24_7 r0 r1 r2 r3 r4 r5 r6 t
        = (if c24_1 r1 t && c24_3 r3 t then (1:ℤ) else 0)
          - (if c24_1 r1 t && c24_3 r3 t && c24_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl24_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n24, Wl24_7 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl24_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n24, Wl24_7 r0 r1 r2 r3 r4 r5 r6 t
        = P24_7 r1 r3 - C24_7 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P24_7, C24_7]
    have hm : C24_7 r1 r3 r0 ≤ M24_7 r1 r3 :=
      CaseSplit.le_mxr (C24_7 r1 r3) 10 r0 (by omega)
    simp only [N24_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N24_8 r1 r4 ≤ ∑ t ∈ Finset.range n24, Wl24_8 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n24, Wl24_8 r0 r1 r2 r3 r4 r5 r6 t
        = (if c24_1 r1 t && c24_4 r4 t then (1:ℤ) else 0)
          - (if c24_1 r1 t && c24_4 r4 t && c24_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl24_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n24, Wl24_8 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl24_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n24, Wl24_8 r0 r1 r2 r3 r4 r5 r6 t
        = P24_8 r1 r4 - C24_8 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P24_8, C24_8]
    have hm : C24_8 r1 r4 r0 ≤ M24_8 r1 r4 :=
      CaseSplit.le_mxr (C24_8 r1 r4) 10 r0 (by omega)
    simp only [N24_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N24_9 r1 r5 ≤ ∑ t ∈ Finset.range n24, Wl24_9 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n24, Wl24_9 r0 r1 r2 r3 r4 r5 r6 t
        = (if c24_1 r1 t && c24_5 r5 t then (1:ℤ) else 0)
          - (if c24_1 r1 t && c24_5 r5 t && c24_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl24_9]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n24, Wl24_9 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl24_9]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n24, Wl24_9 r0 r1 r2 r3 r4 r5 r6 t
        = P24_9 r1 r5 - C24_9 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P24_9, C24_9]
    have hm : C24_9 r1 r5 r0 ≤ M24_9 r1 r5 :=
      CaseSplit.le_mxr (C24_9 r1 r5) 10 r0 (by omega)
    simp only [N24_9]
    split
    · rw [hL]; omega
    · exact hnn
  have hn10 : N24_10 r1 r6 ≤ ∑ t ∈ Finset.range n24, Wl24_10 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n24, Wl24_10 r0 r1 r2 r3 r4 r5 r6 t
        = (if c24_1 r1 t && c24_6 r6 t then (1:ℤ) else 0)
          - (if c24_1 r1 t && c24_6 r6 t && c24_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl24_10]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n24, Wl24_10 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl24_10]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n24, Wl24_10 r0 r1 r2 r3 r4 r5 r6 t
        = P24_10 r1 r6 - C24_10 r1 r6 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P24_10, C24_10]
    have hm : C24_10 r1 r6 r0 ≤ M24_10 r1 r6 :=
      CaseSplit.le_mxr (C24_10 r1 r6) 10 r0 (by omega)
    simp only [N24_10]
    split
    · rw [hL]; omega
    · exact hnn
  have hn11 : N24_11 r2 r3 ≤ ∑ t ∈ Finset.range n24, Wl24_11 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N24_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl24_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N24_12 r2 r4 ≤ ∑ t ∈ Finset.range n24, Wl24_12 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N24_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl24_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N24_13 r2 r5 ≤ ∑ t ∈ Finset.range n24, Wl24_13 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N24_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl24_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N24_14 r2 r6 ≤ ∑ t ∈ Finset.range n24, Wl24_14 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N24_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl24_14]
    exact CaseSplit.ind_nonneg _
  have hn15 : N24_15 r3 r4 ≤ ∑ t ∈ Finset.range n24, Wl24_15 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N24_15]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl24_15]
    exact CaseSplit.ind_nonneg _
  have hn16 : N24_16 r3 r5 ≤ ∑ t ∈ Finset.range n24, Wl24_16 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N24_16]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl24_16]
    exact CaseSplit.ind_nonneg _
  have hn17 : N24_17 r3 r6 ≤ ∑ t ∈ Finset.range n24, Wl24_17 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N24_17]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl24_17]
    exact CaseSplit.ind_nonneg _
  have hn18 : N24_18 r4 r5 ≤ ∑ t ∈ Finset.range n24, Wl24_18 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N24_18]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl24_18]
    exact CaseSplit.ind_nonneg _
  have hn19 : N24_19 r4 r6 ≤ ∑ t ∈ Finset.range n24, Wl24_19 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N24_19]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl24_19]
    exact CaseSplit.ind_nonneg _
  have hn20 : N24_20 r5 r6 ≤ ∑ t ∈ Finset.range n24, Wl24_20 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N24_20]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl24_20]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n24, (w24 t + 3) * Dg24 r0 r1 r2 r3 r4 r5 r6 t = S24_0 r0 + S24_1 r1 + S24_2 r2 + S24_3 r3 + S24_4 r4 + S24_5 r5 + S24_6 r6 := by
    simp only [S24_0, S24_1, S24_2, S24_3, S24_4, S24_5, S24_6, Dg24, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n24, (w24 t + 3) * Dg24 r0 r1 r2 r3 r4 r5 r6 t
      = (∑ t ∈ Finset.range n24, w24 t * Dg24 r0 r1 r2 r3 r4 r5 r6 t)
        + 3 * (∑ t ∈ Finset.range n24, Dg24 r0 r1 r2 r3 r4 r5 r6 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n24, w24 t)
      ≤ ∑ t ∈ Finset.range n24, w24 t * Dg24 r0 r1 r2 r3 r4 r5 r6 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg24 r0 r1 r2 r3 r4 r5 r6 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w24 t := wnn24 t (Finset.mem_range.mp ht)
    calc w24 t = w24 t * 1 := (mul_one _).symm
      _ ≤ w24 t * Dg24 r0 r1 r2 r3 r4 r5 r6 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS24_0 r0 + aS24_1 r1 + aS24_2 r2 + aS24_3 r3 + aS24_4 r4 + aS24_5 r5 + aS24_6 r6) + (aP24_0 r0 r1 + aP24_1 r0 r2 + aP24_2 r0 r3 + aP24_3 r0 r4 + aP24_4 r0 r5 + aP24_5 r0 r6 + aP24_6 r1 r2 + aP24_7 r1 r3 + aP24_8 r1 r4 + aP24_9 r1 r5 + aP24_10 r1 r6 + aP24_11 r2 r3 + aP24_12 r2 r4 + aP24_13 r2 r5 + aP24_14 r2 r6 + aP24_15 r3 r4 + aP24_16 r3 r5 + aP24_17 r3 r6 + aP24_18 r4 r5 + aP24_19 r4 r6 + aP24_20 r5 r6) = (S24_0 r0 + S24_1 r1 + S24_2 r2 + S24_3 r3 + S24_4 r4 + S24_5 r5 + S24_6 r6) - 3 * (N24_0 r0 r1 + N24_1 r0 r2 + N24_2 r0 r3 + N24_3 r0 r4 + N24_4 r0 r5 + N24_5 r0 r6 + N24_6 r1 r2 + N24_7 r1 r3 + N24_8 r1 r4 + N24_9 r1 r5 + N24_10 r1 r6 + N24_11 r2 r3 + N24_12 r2 r4 + N24_13 r2 r5 + N24_14 r2 r6 + N24_15 r3 r4 + N24_16 r3 r5 + N24_17 r3 r6 + N24_18 r4 r5 + N24_19 r4 r6 + N24_20 r5 r6) := by
    simp only [aS24_0, aS24_1, aS24_2, aS24_3, aS24_4, aS24_5, aS24_6, aP24_0, aP24_1, aP24_2, aP24_3, aP24_4, aP24_5, aP24_6, aP24_7, aP24_8, aP24_9, aP24_10, aP24_11, aP24_12, aP24_13, aP24_14, aP24_15, aP24_16, aP24_17, aP24_18, aP24_19, aP24_20, L24_0, L24_1, L24_2, L24_3, L24_4, L24_5, L24_6]
    ring
  have bS0 : aS24_0 r0 ≤ MS24_0 := CaseSplit.le_mxr (aS24_0) 10 r0 (by omega)
  have bS1 : aS24_1 r1 ≤ MS24_1 := CaseSplit.le_mxr (aS24_1) 12 r1 (by omega)
  have bS2 : aS24_2 r2 ≤ MS24_2 := CaseSplit.le_mxr (aS24_2) 16 r2 (by omega)
  have bS3 : aS24_3 r3 ≤ MS24_3 := CaseSplit.le_mxr (aS24_3) 18 r3 (by omega)
  have bS4 : aS24_4 r4 ≤ MS24_4 := CaseSplit.le_mxr (aS24_4) 22 r4 (by omega)
  have bS5 : aS24_5 r5 ≤ MS24_5 := CaseSplit.le_mxr (aS24_5) 28 r5 (by omega)
  have bS6 : aS24_6 r6 ≤ MS24_6 := CaseSplit.le_mxr (aS24_6) 30 r6 (by omega)
  have bP0 : aP24_0 r0 r1 ≤ MP24_0 := CaseSplit.le_mxr2 (aP24_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP24_1 r0 r2 ≤ MP24_1 := CaseSplit.le_mxr2 (aP24_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP24_2 r0 r3 ≤ MP24_2 := CaseSplit.le_mxr2 (aP24_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP24_3 r0 r4 ≤ MP24_3 := CaseSplit.le_mxr2 (aP24_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP24_4 r0 r5 ≤ MP24_4 := CaseSplit.le_mxr2 (aP24_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP24_5 r0 r6 ≤ MP24_5 := CaseSplit.le_mxr2 (aP24_5) 10 30 r0 r6 (by omega) (by omega)
  have bP6 : aP24_6 r1 r2 ≤ MP24_6 := CaseSplit.le_mxr2 (aP24_6) 12 16 r1 r2 (by omega) (by omega)
  have bP7 : aP24_7 r1 r3 ≤ MP24_7 := CaseSplit.le_mxr2 (aP24_7) 12 18 r1 r3 (by omega) (by omega)
  have bP8 : aP24_8 r1 r4 ≤ MP24_8 := CaseSplit.le_mxr2 (aP24_8) 12 22 r1 r4 (by omega) (by omega)
  have bP9 : aP24_9 r1 r5 ≤ MP24_9 := CaseSplit.le_mxr2 (aP24_9) 12 28 r1 r5 (by omega) (by omega)
  have bP10 : aP24_10 r1 r6 ≤ MP24_10 := CaseSplit.le_mxr2 (aP24_10) 12 30 r1 r6 (by omega) (by omega)
  have bP11 : aP24_11 r2 r3 ≤ MP24_11 := CaseSplit.le_mxr2 (aP24_11) 16 18 r2 r3 (by omega) (by omega)
  have bP12 : aP24_12 r2 r4 ≤ MP24_12 := CaseSplit.le_mxr2 (aP24_12) 16 22 r2 r4 (by omega) (by omega)
  have bP13 : aP24_13 r2 r5 ≤ MP24_13 := CaseSplit.le_mxr2 (aP24_13) 16 28 r2 r5 (by omega) (by omega)
  have bP14 : aP24_14 r2 r6 ≤ MP24_14 := CaseSplit.le_mxr2 (aP24_14) 16 30 r2 r6 (by omega) (by omega)
  have bP15 : aP24_15 r3 r4 ≤ MP24_15 := CaseSplit.le_mxr2 (aP24_15) 18 22 r3 r4 (by omega) (by omega)
  have bP16 : aP24_16 r3 r5 ≤ MP24_16 := CaseSplit.le_mxr2 (aP24_16) 18 28 r3 r5 (by omega) (by omega)
  have bP17 : aP24_17 r3 r6 ≤ MP24_17 := CaseSplit.le_mxr2 (aP24_17) 18 30 r3 r6 (by omega) (by omega)
  have bP18 : aP24_18 r4 r5 ≤ MP24_18 := CaseSplit.le_mxr2 (aP24_18) 22 28 r4 r5 (by omega) (by omega)
  have bP19 : aP24_19 r4 r6 ≤ MP24_19 := CaseSplit.le_mxr2 (aP24_19) 22 30 r4 r6 (by omega) (by omega)
  have bP20 : aP24_20 r5 r6 ≤ MP24_20 := CaseSplit.le_mxr2 (aP24_20) 28 30 r5 r6 (by omega) (by omega)
  have hrhs : rhs24 = (∑ t ∈ Finset.range n24, w24 t) + 3 * (n24 : ℤ) := rfl
  have hc := cert24
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, hn15, hn16, hn17, hn18, hn19, hn20, bS0, bS1, bS2, bS3, bS4, bS5, bS6, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14, bP15, bP16, bP17, bP18, bP19, bP20]

end IncCert31
