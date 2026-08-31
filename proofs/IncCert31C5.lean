/-
INCREMENT-WIDTH CERTIFICATE, step 29->31, case 5 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_29_31.json, which re-derives every number
from the primes alone).

Machine 31, INCREMENT width 65 = F_2(29) + s_min(31) = 55 + 10,
held gears [5, 7] at phases [0, 5].  Free gears [11, 13, 17, 19, 23, 29, 31].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 256.
-/
import IncCert31B

namespace IncCert31

/-! ### case 5: held gears at phases [0, 5] -/

def p5 : List ℕ := [0, 2, 5, 7, 12, 13, 18, 20, 23, 25, 27, 28, 30, 32, 33, 35, 37, 40, 42, 47, 48, 53, 55, 58, 60, 62, 63]
def q5 (t : ℕ) : ℕ := p5.getD t 0
def n5 : ℕ := 27
def yl5 : List ℤ := [175, 18, 111, 74, 113, 42, 101, 88, 191, 173, 95, 235, 256, 191, 122, 171, 173, 74, 104, 0, 113, 104, 111, 48, 173, 23, 42]
def w5 (t : ℕ) : ℤ := yl5.getD t 0
def ul5 : List ℤ := [(-12), (-12), (-5), (-12), 0, (-51), (-5), (-12), 0, (-12), (-12), (-5), (-5), 7, 0, 12, 0, 5, 0, 5, 0, 0, 12, 0, (-7), (-7), (-7), (-7), (-231), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), 0, (-7), (-7), 0, 7, 7, 0, 0, 0, 7, 0, 7, 7, 0, (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), 0, (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), 0, 0, 0, 0, 7, 0, 7, 0, 0, 0, 7, 7, 0, 0, 0, 0, (-417), (-74), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-240), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-171), 0, (-180), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 461, 461, 461, 461, 461, 461, 461, 461, 454, 461, 461, 461, 434, 429, 461, 461, 461, (-461), (-461), (-461), (-461), (-461), (-461), (-461), (-461), (-461), (-461), (-461), (-461), (-461), 436, 224, 436, 436, 415, 387, 436, 436, 436, 334, 436, 436, 436, 436, 214, 436, 436, 436, 427, (-436), (-436), (-436), (-436), (-436), (-436), (-436), (-436), (-436), (-436), (-436), (-436), (-436), 454, 454, 454, 454, 454, 454, 454, 454, 454, 454, 454, 454, 454, 360, 454, 454, 0, 454, 454, 454, 454, 0, 454, (-454), (-454), (-454), (-454), (-454), (-454), (-454), (-454), (-454), (-454), (-454), (-454), (-454), 277, 357, 198, 205, 357, 357, 357, 101, 118, 357, 111, 357, 357, 357, 95, 120, 295, 81, 357, 166, 357, 357, 325, 344, 357, 357, 281, 198, 357, (-357), (-357), (-357), (-357), (-357), (-357), (-357), (-357), (-357), (-357), (-357), (-357), (-357), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-235), 0, 0, 0, 0, (-208), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-74), 0, 0, 0, 0, 0, 0, 0, 0, (-76), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-7), 0, 0, (-178), 0, 0, 0, 0, 0, 0, 0, (-157), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-95), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-131), 0, 0, 0, 0, 0, 0, (-240), 0, 0, 0, 0, 0, 0, 0, (-351), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 357, 357, 357, 357, 357, 357, 357, 357, 357, 357, 357, 357, 118, 357, 231, 111, 357, 81, 108, 357, 357, 118, 357, 48, 357, 357, 357, 344, 357, 323, 357, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-238), (-277), 0, 0, 0, 0, (-180), 0, 0, 0, 0, 0, (-120), (-115), 0, 0, 0]
def u5 (k : ℕ) : ℤ := ul5.getD k 0

def c5_0 (r t : ℕ) : Bool := gb11 r (q5 t)
def c5_1 (r t : ℕ) : Bool := gb13 r (q5 t)
def c5_2 (r t : ℕ) : Bool := gb17 r (q5 t)
def c5_3 (r t : ℕ) : Bool := gb19 r (q5 t)
def c5_4 (r t : ℕ) : Bool := gb23 r (q5 t)
def c5_5 (r t : ℕ) : Bool := gb29 r (q5 t)
def c5_6 (r t : ℕ) : Bool := gb31 r (q5 t)

def S5_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (w5 t + 7) * (if c5_0 r t then 1 else 0)
def S5_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (w5 t + 7) * (if c5_1 r t then 1 else 0)
def S5_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (w5 t + 7) * (if c5_2 r t then 1 else 0)
def S5_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (w5 t + 7) * (if c5_3 r t then 1 else 0)
def S5_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (w5 t + 7) * (if c5_4 r t then 1 else 0)
def S5_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (w5 t + 7) * (if c5_5 r t then 1 else 0)
def S5_6 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (w5 t + 7) * (if c5_6 r t then 1 else 0)

def L5_0 (r : ℕ) : ℤ := u5 (13 + r) + u5 (41 + r) + u5 (71 + r) + u5 (105 + r) + u5 (145 + r) + u5 (187 + r)
def L5_1 (r : ℕ) : ℤ := u5 (0 + r) + u5 (215 + r) + u5 (247 + r) + u5 (283 + r) + u5 (325 + r) + u5 (369 + r)
def L5_2 (r : ℕ) : ℤ := u5 (24 + r) + u5 (198 + r) + u5 (401 + r) + u5 (441 + r) + u5 (487 + r) + u5 (535 + r)
def L5_3 (r : ℕ) : ℤ := u5 (52 + r) + u5 (228 + r) + u5 (382 + r) + u5 (575 + r) + u5 (623 + r) + u5 (673 + r)
def L5_4 (r : ℕ) : ℤ := u5 (82 + r) + u5 (260 + r) + u5 (418 + r) + u5 (552 + r) + u5 (721 + r) + u5 (775 + r)
def L5_5 (r : ℕ) : ℤ := u5 (116 + r) + u5 (296 + r) + u5 (458 + r) + u5 (594 + r) + u5 (692 + r) + u5 (829 + r)
def L5_6 (r : ℕ) : ℤ := u5 (156 + r) + u5 (338 + r) + u5 (504 + r) + u5 (642 + r) + u5 (744 + r) + u5 (798 + r)

def aS5_0 (r : ℕ) : ℤ := S5_0 r - L5_0 r
def MS5_0 : ℤ := CaseSplit.mxr (aS5_0) 10
def aS5_1 (r : ℕ) : ℤ := S5_1 r - L5_1 r
def MS5_1 : ℤ := CaseSplit.mxr (aS5_1) 12
def aS5_2 (r : ℕ) : ℤ := S5_2 r - L5_2 r
def MS5_2 : ℤ := CaseSplit.mxr (aS5_2) 16
def aS5_3 (r : ℕ) : ℤ := S5_3 r - L5_3 r
def MS5_3 : ℤ := CaseSplit.mxr (aS5_3) 18
def aS5_4 (r : ℕ) : ℤ := S5_4 r - L5_4 r
def MS5_4 : ℤ := CaseSplit.mxr (aS5_4) 22
def aS5_5 (r : ℕ) : ℤ := S5_5 r - L5_5 r
def MS5_5 : ℤ := CaseSplit.mxr (aS5_5) 28
def aS5_6 (r : ℕ) : ℤ := S5_6 r - L5_6 r
def MS5_6 : ℤ := CaseSplit.mxr (aS5_6) 30

def N5_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (if c5_0 ra t && c5_1 rb t then 1 else 0)
def aP5_0 (ra rb : ℕ) : ℤ := -(7) * N5_0 ra rb + u5 (0 + rb) + u5 (13 + ra)
def MP5_0 : ℤ := CaseSplit.mxr2 (aP5_0) 10 12
def N5_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (if c5_0 ra t && c5_2 rb t then 1 else 0)
def aP5_1 (ra rb : ℕ) : ℤ := -(7) * N5_1 ra rb + u5 (24 + rb) + u5 (41 + ra)
def MP5_1 : ℤ := CaseSplit.mxr2 (aP5_1) 10 16
def N5_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (if c5_0 ra t && c5_3 rb t then 1 else 0)
def aP5_2 (ra rb : ℕ) : ℤ := -(7) * N5_2 ra rb + u5 (52 + rb) + u5 (71 + ra)
def MP5_2 : ℤ := CaseSplit.mxr2 (aP5_2) 10 18
def N5_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (if c5_0 ra t && c5_4 rb t then 1 else 0)
def aP5_3 (ra rb : ℕ) : ℤ := -(7) * N5_3 ra rb + u5 (82 + rb) + u5 (105 + ra)
def MP5_3 : ℤ := CaseSplit.mxr2 (aP5_3) 10 22
def N5_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (if c5_0 ra t && c5_5 rb t then 1 else 0)
def aP5_4 (ra rb : ℕ) : ℤ := -(7) * N5_4 ra rb + u5 (116 + rb) + u5 (145 + ra)
def MP5_4 : ℤ := CaseSplit.mxr2 (aP5_4) 10 28
def N5_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (if c5_0 ra t && c5_6 rb t then 1 else 0)
def aP5_5 (ra rb : ℕ) : ℤ := -(7) * N5_5 ra rb + u5 (156 + rb) + u5 (187 + ra)
def MP5_5 : ℤ := CaseSplit.mxr2 (aP5_5) 10 30
def P5_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (if c5_1 ra t && c5_2 rb t then 1 else 0)
def C5_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (if c5_1 ra t && c5_2 rb t && c5_0 s t then 1 else 0)
def M5_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C5_6 ra rb) 10
def E5_6 : List ℕ := [25, 31, 68, 79, 104, 115, 120, 126, 188, 194, 210, 216]
def N5_6 (ra rb : ℕ) : ℤ := if E5_6.contains (ra * 17 + rb) = true then P5_6 ra rb - M5_6 ra rb else 0
def aP5_6 (ra rb : ℕ) : ℤ := -(7) * N5_6 ra rb + u5 (198 + rb) + u5 (215 + ra)
def MP5_6 : ℤ := CaseSplit.mxr2 (aP5_6) 12 16
def P5_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (if c5_1 ra t && c5_3 rb t then 1 else 0)
def C5_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (if c5_1 ra t && c5_3 rb t && c5_0 s t then 1 else 0)
def M5_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C5_7 ra rb) 10
def E5_7 : List ℕ := [20, 33, 41, 44, 67, 70, 78, 91, 120, 131, 146, 154, 167, 170, 191, 204, 207, 212, 238, 246]
def N5_7 (ra rb : ℕ) : ℤ := if E5_7.contains (ra * 19 + rb) = true then P5_7 ra rb - M5_7 ra rb else 0
def aP5_7 (ra rb : ℕ) : ℤ := -(7) * N5_7 ra rb + u5 (228 + rb) + u5 (247 + ra)
def MP5_7 : ℤ := CaseSplit.mxr2 (aP5_7) 12 18
def P5_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (if c5_1 ra t && c5_4 rb t then 1 else 0)
def C5_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (if c5_1 ra t && c5_4 rb t && c5_0 s t then 1 else 0)
def M5_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C5_8 ra rb) 10
def E5_8 : List ℕ := [2]
def N5_8 (ra rb : ℕ) : ℤ := if E5_8.contains (ra * 23 + rb) = true then P5_8 ra rb - M5_8 ra rb else 0
def aP5_8 (ra rb : ℕ) : ℤ := -(7) * N5_8 ra rb + u5 (260 + rb) + u5 (283 + ra)
def MP5_8 : ℤ := CaseSplit.mxr2 (aP5_8) 12 22
def P5_9 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (if c5_1 ra t && c5_5 rb t then 1 else 0)
def C5_9 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (if c5_1 ra t && c5_5 rb t && c5_0 s t then 1 else 0)
def M5_9 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C5_9 ra rb) 10
def E5_9 : List ℕ := [30, 146]
def N5_9 (ra rb : ℕ) : ℤ := if E5_9.contains (ra * 29 + rb) = true then P5_9 ra rb - M5_9 ra rb else 0
def aP5_9 (ra rb : ℕ) : ℤ := -(7) * N5_9 ra rb + u5 (296 + rb) + u5 (325 + ra)
def MP5_9 : ℤ := CaseSplit.mxr2 (aP5_9) 12 28
def P5_10 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (if c5_1 ra t && c5_6 rb t then 1 else 0)
def C5_10 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (if c5_1 ra t && c5_6 rb t && c5_0 s t then 1 else 0)
def M5_10 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C5_10 ra rb) 10
def E5_10 : List ℕ := []
def N5_10 (ra rb : ℕ) : ℤ := if E5_10.contains (ra * 31 + rb) = true then P5_10 ra rb - M5_10 ra rb else 0
def aP5_10 (ra rb : ℕ) : ℤ := -(7) * N5_10 ra rb + u5 (338 + rb) + u5 (369 + ra)
def MP5_10 : ℤ := CaseSplit.mxr2 (aP5_10) 12 30
def N5_11 (_ra _rb : ℕ) : ℤ := 0
def aP5_11 (ra rb : ℕ) : ℤ := -(7) * N5_11 ra rb + u5 (382 + rb) + u5 (401 + ra)
def MP5_11 : ℤ := CaseSplit.mxr2 (aP5_11) 16 18
def N5_12 (_ra _rb : ℕ) : ℤ := 0
def aP5_12 (ra rb : ℕ) : ℤ := -(7) * N5_12 ra rb + u5 (418 + rb) + u5 (441 + ra)
def MP5_12 : ℤ := CaseSplit.mxr2 (aP5_12) 16 22
def N5_13 (_ra _rb : ℕ) : ℤ := 0
def aP5_13 (ra rb : ℕ) : ℤ := -(7) * N5_13 ra rb + u5 (458 + rb) + u5 (487 + ra)
def MP5_13 : ℤ := CaseSplit.mxr2 (aP5_13) 16 28
def N5_14 (_ra _rb : ℕ) : ℤ := 0
def aP5_14 (ra rb : ℕ) : ℤ := -(7) * N5_14 ra rb + u5 (504 + rb) + u5 (535 + ra)
def MP5_14 : ℤ := CaseSplit.mxr2 (aP5_14) 16 30
def N5_15 (_ra _rb : ℕ) : ℤ := 0
def aP5_15 (ra rb : ℕ) : ℤ := -(7) * N5_15 ra rb + u5 (552 + rb) + u5 (575 + ra)
def MP5_15 : ℤ := CaseSplit.mxr2 (aP5_15) 18 22
def N5_16 (_ra _rb : ℕ) : ℤ := 0
def aP5_16 (ra rb : ℕ) : ℤ := -(7) * N5_16 ra rb + u5 (594 + rb) + u5 (623 + ra)
def MP5_16 : ℤ := CaseSplit.mxr2 (aP5_16) 18 28
def N5_17 (_ra _rb : ℕ) : ℤ := 0
def aP5_17 (ra rb : ℕ) : ℤ := -(7) * N5_17 ra rb + u5 (642 + rb) + u5 (673 + ra)
def MP5_17 : ℤ := CaseSplit.mxr2 (aP5_17) 18 30
def N5_18 (_ra _rb : ℕ) : ℤ := 0
def aP5_18 (ra rb : ℕ) : ℤ := -(7) * N5_18 ra rb + u5 (692 + rb) + u5 (721 + ra)
def MP5_18 : ℤ := CaseSplit.mxr2 (aP5_18) 22 28
def N5_19 (_ra _rb : ℕ) : ℤ := 0
def aP5_19 (ra rb : ℕ) : ℤ := -(7) * N5_19 ra rb + u5 (744 + rb) + u5 (775 + ra)
def MP5_19 : ℤ := CaseSplit.mxr2 (aP5_19) 22 30
def N5_20 (_ra _rb : ℕ) : ℤ := 0
def aP5_20 (ra rb : ℕ) : ℤ := -(7) * N5_20 ra rb + u5 (798 + rb) + u5 (829 + ra)
def MP5_20 : ℤ := CaseSplit.mxr2 (aP5_20) 28 30

def rhs5 : ℤ := (∑ t ∈ Finset.range n5, w5 t) + 7 * (n5 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn5 : ∀ t, t < n5 → (0 : ℤ) ≤ w5 t := by decide
theorem plt5 : ∀ t, t < n5 → q5 t < 65 := by decide
theorem pfree5_5 : ∀ t, t < n5 → gb5 0 (q5 t) = false := by decide
theorem pfree5_7 : ∀ t, t < n5 → gb7 5 (q5 t) = false := by decide
theorem MSv5_0 : MS5_0 = 705 := by decide +kernel
theorem MSv5_1 : MS5_1 = 2242 := by decide +kernel
theorem MSv5_2 : MS5_2 = 1 := by decide +kernel
theorem MSv5_3 : MS5_3 = 1 := by decide +kernel
theorem MSv5_4 : MS5_4 = 1 := by decide +kernel
theorem MSv5_5 : MS5_5 = 1 := by decide +kernel
theorem MSv5_6 : MS5_6 = 1 := by decide +kernel
theorem MPv5_0 : MP5_0 = 0 := by decide +kernel
theorem MPv5_1 : MP5_1 = 0 := by decide +kernel
theorem MPv5_2 : MP5_2 = 0 := by decide +kernel
theorem MPv5_3 : MP5_3 = 0 := by decide +kernel
theorem MPv5_4 : MP5_4 = 0 := by decide +kernel
theorem MPv5_5 : MP5_5 = 0 := by decide +kernel
theorem MPv5_6 : MP5_6 = 0 := by decide +kernel
theorem MPv5_7 : MP5_7 = 0 := by decide +kernel
theorem MPv5_8 : MP5_8 = 0 := by decide +kernel
theorem MPv5_9 : MP5_9 = 0 := by decide +kernel
theorem MPv5_10 : MP5_10 = 0 := by decide +kernel
theorem MPv5_11 : MP5_11 = 0 := by decide +kernel
theorem MPv5_12 : MP5_12 = 0 := by decide +kernel
theorem MPv5_13 : MP5_13 = 0 := by decide +kernel
theorem MPv5_14 : MP5_14 = 0 := by decide +kernel
theorem MPv5_15 : MP5_15 = 0 := by decide +kernel
theorem MPv5_16 : MP5_16 = 0 := by decide +kernel
theorem MPv5_17 : MP5_17 = 0 := by decide +kernel
theorem MPv5_18 : MP5_18 = 0 := by decide +kernel
theorem MPv5_19 : MP5_19 = 0 := by decide +kernel
theorem MPv5_20 : MP5_20 = 357 := by decide +kernel
theorem rhsv5 : rhs5 = 3310 := by decide +kernel

/-- **The case-5 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/256.
    (Scaled by the common denominator 256: 3309 < 3310.) -/
theorem cert5 : MS5_0 + MS5_1 + MS5_2 + MS5_3 + MS5_4 + MS5_5 + MS5_6 + MP5_0 + MP5_1 + MP5_2 + MP5_3 + MP5_4 + MP5_5 + MP5_6 + MP5_7 + MP5_8 + MP5_9 + MP5_10 + MP5_11 + MP5_12 + MP5_13 + MP5_14 + MP5_15 + MP5_16 + MP5_17 + MP5_18 + MP5_19 + MP5_20 < rhs5 := by
  rw [MSv5_0, MSv5_1, MSv5_2, MSv5_3, MSv5_4, MSv5_5, MSv5_6, MPv5_0, MPv5_1, MPv5_2, MPv5_3, MPv5_4, MPv5_5, MPv5_6, MPv5_7, MPv5_8, MPv5_9, MPv5_10, MPv5_11, MPv5_12, MPv5_13, MPv5_14, MPv5_15, MPv5_16, MPv5_17, MPv5_18, MPv5_19, MPv5_20, rhsv5]
  decide

def Dg5 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := (if c5_0 r0 t then 1 else 0) + (if c5_1 r1 t then 1 else 0) + (if c5_2 r2 t then 1 else 0) + (if c5_3 r3 t then 1 else 0) + (if c5_4 r4 t then 1 else 0) + (if c5_5 r5 t then 1 else 0) + (if c5_6 r6 t then 1 else 0)
def Wl5_0 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c5_0 r0 t && c5_1 r1 t then 1 else 0
def Wl5_1 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c5_0 r0 t && c5_2 r2 t then 1 else 0
def Wl5_2 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c5_0 r0 t && c5_3 r3 t then 1 else 0
def Wl5_3 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c5_0 r0 t && c5_4 r4 t then 1 else 0
def Wl5_4 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c5_0 r0 t && c5_5 r5 t then 1 else 0
def Wl5_5 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c5_0 r0 t && c5_6 r6 t then 1 else 0
def Wl5_6 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c5_0 r0 t && c5_1 r1 t && c5_2 r2 t then 1 else 0
def Wl5_7 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c5_0 r0 t && c5_1 r1 t && c5_3 r3 t then 1 else 0
def Wl5_8 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c5_0 r0 t && c5_1 r1 t && c5_4 r4 t then 1 else 0
def Wl5_9 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c5_0 r0 t && c5_1 r1 t && c5_5 r5 t then 1 else 0
def Wl5_10 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c5_0 r0 t && c5_1 r1 t && c5_6 r6 t then 1 else 0
def Wl5_11 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c5_0 r0 t && !c5_1 r1 t && c5_2 r2 t && c5_3 r3 t then 1 else 0
def Wl5_12 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c5_0 r0 t && !c5_1 r1 t && c5_2 r2 t && c5_4 r4 t then 1 else 0
def Wl5_13 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c5_0 r0 t && !c5_1 r1 t && c5_2 r2 t && c5_5 r5 t then 1 else 0
def Wl5_14 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c5_0 r0 t && !c5_1 r1 t && c5_2 r2 t && c5_6 r6 t then 1 else 0
def Wl5_15 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c5_0 r0 t && !c5_1 r1 t && !c5_2 r2 t && c5_3 r3 t && c5_4 r4 t then 1 else 0
def Wl5_16 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c5_0 r0 t && !c5_1 r1 t && !c5_2 r2 t && c5_3 r3 t && c5_5 r5 t then 1 else 0
def Wl5_17 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c5_0 r0 t && !c5_1 r1 t && !c5_2 r2 t && c5_3 r3 t && c5_6 r6 t then 1 else 0
def Wl5_18 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c5_0 r0 t && !c5_1 r1 t && !c5_2 r2 t && !c5_3 r3 t && c5_4 r4 t && c5_5 r5 t then 1 else 0
def Wl5_19 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c5_0 r0 t && !c5_1 r1 t && !c5_2 r2 t && !c5_3 r3 t && c5_4 r4 t && c5_6 r6 t then 1 else 0
def Wl5_20 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c5_0 r0 t && !c5_1 r1 t && !c5_2 r2 t && !c5_3 r3 t && !c5_4 r4 t && c5_5 r5 t && c5_6 r6 t then 1 else 0

/-- **No configuration blocks the whole window in case 5.** -/
theorem nocov5 {r0 r1 r2 r3 r4 r5 r6 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29) (h6 : r6 < 31)
    (hcov : ∀ t, t < n5 → (c5_0 r0 t || c5_1 r1 t || c5_2 r2 t || c5_3 r3 t || c5_4 r4 t || c5_5 r5 t || c5_6 r6 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n5, (1 : ℤ) + (Wl5_0 r0 r1 r2 r3 r4 r5 r6 t + Wl5_1 r0 r1 r2 r3 r4 r5 r6 t + Wl5_2 r0 r1 r2 r3 r4 r5 r6 t + Wl5_3 r0 r1 r2 r3 r4 r5 r6 t + Wl5_4 r0 r1 r2 r3 r4 r5 r6 t + Wl5_5 r0 r1 r2 r3 r4 r5 r6 t + Wl5_6 r0 r1 r2 r3 r4 r5 r6 t + Wl5_7 r0 r1 r2 r3 r4 r5 r6 t + Wl5_8 r0 r1 r2 r3 r4 r5 r6 t + Wl5_9 r0 r1 r2 r3 r4 r5 r6 t + Wl5_10 r0 r1 r2 r3 r4 r5 r6 t + Wl5_11 r0 r1 r2 r3 r4 r5 r6 t + Wl5_12 r0 r1 r2 r3 r4 r5 r6 t + Wl5_13 r0 r1 r2 r3 r4 r5 r6 t + Wl5_14 r0 r1 r2 r3 r4 r5 r6 t + Wl5_15 r0 r1 r2 r3 r4 r5 r6 t + Wl5_16 r0 r1 r2 r3 r4 r5 r6 t + Wl5_17 r0 r1 r2 r3 r4 r5 r6 t + Wl5_18 r0 r1 r2 r3 r4 r5 r6 t + Wl5_19 r0 r1 r2 r3 r4 r5 r6 t + Wl5_20 r0 r1 r2 r3 r4 r5 r6 t) ≤ Dg5 r0 r1 r2 r3 r4 r5 r6 t := by
    intro t ht
    simp only [Wl5_0, Wl5_1, Wl5_2, Wl5_3, Wl5_4, Wl5_5, Wl5_6, Wl5_7, Wl5_8, Wl5_9, Wl5_10, Wl5_11, Wl5_12, Wl5_13, Wl5_14, Wl5_15, Wl5_16, Wl5_17, Wl5_18, Wl5_19, Wl5_20, Dg5]
    exact CaseSplit.lowest7 _ _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n5, (1 : ℤ) ≤ Dg5 r0 r1 r2 r3 r4 r5 r6 t := by
    intro t ht
    simp only [Dg5]
    exact CaseSplit.degpos7 _ _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n5 : ℤ) + ((∑ t ∈ Finset.range n5, Wl5_0 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n5, Wl5_1 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n5, Wl5_2 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n5, Wl5_3 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n5, Wl5_4 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n5, Wl5_5 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n5, Wl5_6 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n5, Wl5_7 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n5, Wl5_8 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n5, Wl5_9 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n5, Wl5_10 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n5, Wl5_11 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n5, Wl5_12 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n5, Wl5_13 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n5, Wl5_14 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n5, Wl5_15 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n5, Wl5_16 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n5, Wl5_17 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n5, Wl5_18 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n5, Wl5_19 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n5, Wl5_20 r0 r1 r2 r3 r4 r5 r6 t)) ≤ ∑ t ∈ Finset.range n5, Dg5 r0 r1 r2 r3 r4 r5 r6 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N5_0 r0 r1 ≤ ∑ t ∈ Finset.range n5, Wl5_0 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N5_0, Wl5_0, le_refl]
  have hn1 : N5_1 r0 r2 ≤ ∑ t ∈ Finset.range n5, Wl5_1 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N5_1, Wl5_1, le_refl]
  have hn2 : N5_2 r0 r3 ≤ ∑ t ∈ Finset.range n5, Wl5_2 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N5_2, Wl5_2, le_refl]
  have hn3 : N5_3 r0 r4 ≤ ∑ t ∈ Finset.range n5, Wl5_3 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N5_3, Wl5_3, le_refl]
  have hn4 : N5_4 r0 r5 ≤ ∑ t ∈ Finset.range n5, Wl5_4 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N5_4, Wl5_4, le_refl]
  have hn5 : N5_5 r0 r6 ≤ ∑ t ∈ Finset.range n5, Wl5_5 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N5_5, Wl5_5, le_refl]
  have hn6 : N5_6 r1 r2 ≤ ∑ t ∈ Finset.range n5, Wl5_6 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n5, Wl5_6 r0 r1 r2 r3 r4 r5 r6 t
        = (if c5_1 r1 t && c5_2 r2 t then (1:ℤ) else 0)
          - (if c5_1 r1 t && c5_2 r2 t && c5_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl5_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n5, Wl5_6 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl5_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n5, Wl5_6 r0 r1 r2 r3 r4 r5 r6 t
        = P5_6 r1 r2 - C5_6 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P5_6, C5_6]
    have hm : C5_6 r1 r2 r0 ≤ M5_6 r1 r2 :=
      CaseSplit.le_mxr (C5_6 r1 r2) 10 r0 (by omega)
    simp only [N5_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N5_7 r1 r3 ≤ ∑ t ∈ Finset.range n5, Wl5_7 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n5, Wl5_7 r0 r1 r2 r3 r4 r5 r6 t
        = (if c5_1 r1 t && c5_3 r3 t then (1:ℤ) else 0)
          - (if c5_1 r1 t && c5_3 r3 t && c5_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl5_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n5, Wl5_7 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl5_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n5, Wl5_7 r0 r1 r2 r3 r4 r5 r6 t
        = P5_7 r1 r3 - C5_7 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P5_7, C5_7]
    have hm : C5_7 r1 r3 r0 ≤ M5_7 r1 r3 :=
      CaseSplit.le_mxr (C5_7 r1 r3) 10 r0 (by omega)
    simp only [N5_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N5_8 r1 r4 ≤ ∑ t ∈ Finset.range n5, Wl5_8 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n5, Wl5_8 r0 r1 r2 r3 r4 r5 r6 t
        = (if c5_1 r1 t && c5_4 r4 t then (1:ℤ) else 0)
          - (if c5_1 r1 t && c5_4 r4 t && c5_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl5_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n5, Wl5_8 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl5_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n5, Wl5_8 r0 r1 r2 r3 r4 r5 r6 t
        = P5_8 r1 r4 - C5_8 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P5_8, C5_8]
    have hm : C5_8 r1 r4 r0 ≤ M5_8 r1 r4 :=
      CaseSplit.le_mxr (C5_8 r1 r4) 10 r0 (by omega)
    simp only [N5_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N5_9 r1 r5 ≤ ∑ t ∈ Finset.range n5, Wl5_9 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n5, Wl5_9 r0 r1 r2 r3 r4 r5 r6 t
        = (if c5_1 r1 t && c5_5 r5 t then (1:ℤ) else 0)
          - (if c5_1 r1 t && c5_5 r5 t && c5_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl5_9]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n5, Wl5_9 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl5_9]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n5, Wl5_9 r0 r1 r2 r3 r4 r5 r6 t
        = P5_9 r1 r5 - C5_9 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P5_9, C5_9]
    have hm : C5_9 r1 r5 r0 ≤ M5_9 r1 r5 :=
      CaseSplit.le_mxr (C5_9 r1 r5) 10 r0 (by omega)
    simp only [N5_9]
    split
    · rw [hL]; omega
    · exact hnn
  have hn10 : N5_10 r1 r6 ≤ ∑ t ∈ Finset.range n5, Wl5_10 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n5, Wl5_10 r0 r1 r2 r3 r4 r5 r6 t
        = (if c5_1 r1 t && c5_6 r6 t then (1:ℤ) else 0)
          - (if c5_1 r1 t && c5_6 r6 t && c5_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl5_10]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n5, Wl5_10 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl5_10]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n5, Wl5_10 r0 r1 r2 r3 r4 r5 r6 t
        = P5_10 r1 r6 - C5_10 r1 r6 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P5_10, C5_10]
    have hm : C5_10 r1 r6 r0 ≤ M5_10 r1 r6 :=
      CaseSplit.le_mxr (C5_10 r1 r6) 10 r0 (by omega)
    simp only [N5_10]
    split
    · rw [hL]; omega
    · exact hnn
  have hn11 : N5_11 r2 r3 ≤ ∑ t ∈ Finset.range n5, Wl5_11 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N5_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl5_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N5_12 r2 r4 ≤ ∑ t ∈ Finset.range n5, Wl5_12 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N5_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl5_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N5_13 r2 r5 ≤ ∑ t ∈ Finset.range n5, Wl5_13 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N5_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl5_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N5_14 r2 r6 ≤ ∑ t ∈ Finset.range n5, Wl5_14 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N5_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl5_14]
    exact CaseSplit.ind_nonneg _
  have hn15 : N5_15 r3 r4 ≤ ∑ t ∈ Finset.range n5, Wl5_15 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N5_15]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl5_15]
    exact CaseSplit.ind_nonneg _
  have hn16 : N5_16 r3 r5 ≤ ∑ t ∈ Finset.range n5, Wl5_16 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N5_16]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl5_16]
    exact CaseSplit.ind_nonneg _
  have hn17 : N5_17 r3 r6 ≤ ∑ t ∈ Finset.range n5, Wl5_17 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N5_17]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl5_17]
    exact CaseSplit.ind_nonneg _
  have hn18 : N5_18 r4 r5 ≤ ∑ t ∈ Finset.range n5, Wl5_18 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N5_18]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl5_18]
    exact CaseSplit.ind_nonneg _
  have hn19 : N5_19 r4 r6 ≤ ∑ t ∈ Finset.range n5, Wl5_19 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N5_19]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl5_19]
    exact CaseSplit.ind_nonneg _
  have hn20 : N5_20 r5 r6 ≤ ∑ t ∈ Finset.range n5, Wl5_20 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N5_20]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl5_20]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n5, (w5 t + 7) * Dg5 r0 r1 r2 r3 r4 r5 r6 t = S5_0 r0 + S5_1 r1 + S5_2 r2 + S5_3 r3 + S5_4 r4 + S5_5 r5 + S5_6 r6 := by
    simp only [S5_0, S5_1, S5_2, S5_3, S5_4, S5_5, S5_6, Dg5, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n5, (w5 t + 7) * Dg5 r0 r1 r2 r3 r4 r5 r6 t
      = (∑ t ∈ Finset.range n5, w5 t * Dg5 r0 r1 r2 r3 r4 r5 r6 t)
        + 7 * (∑ t ∈ Finset.range n5, Dg5 r0 r1 r2 r3 r4 r5 r6 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n5, w5 t)
      ≤ ∑ t ∈ Finset.range n5, w5 t * Dg5 r0 r1 r2 r3 r4 r5 r6 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg5 r0 r1 r2 r3 r4 r5 r6 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w5 t := wnn5 t (Finset.mem_range.mp ht)
    calc w5 t = w5 t * 1 := (mul_one _).symm
      _ ≤ w5 t * Dg5 r0 r1 r2 r3 r4 r5 r6 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS5_0 r0 + aS5_1 r1 + aS5_2 r2 + aS5_3 r3 + aS5_4 r4 + aS5_5 r5 + aS5_6 r6) + (aP5_0 r0 r1 + aP5_1 r0 r2 + aP5_2 r0 r3 + aP5_3 r0 r4 + aP5_4 r0 r5 + aP5_5 r0 r6 + aP5_6 r1 r2 + aP5_7 r1 r3 + aP5_8 r1 r4 + aP5_9 r1 r5 + aP5_10 r1 r6 + aP5_11 r2 r3 + aP5_12 r2 r4 + aP5_13 r2 r5 + aP5_14 r2 r6 + aP5_15 r3 r4 + aP5_16 r3 r5 + aP5_17 r3 r6 + aP5_18 r4 r5 + aP5_19 r4 r6 + aP5_20 r5 r6) = (S5_0 r0 + S5_1 r1 + S5_2 r2 + S5_3 r3 + S5_4 r4 + S5_5 r5 + S5_6 r6) - 7 * (N5_0 r0 r1 + N5_1 r0 r2 + N5_2 r0 r3 + N5_3 r0 r4 + N5_4 r0 r5 + N5_5 r0 r6 + N5_6 r1 r2 + N5_7 r1 r3 + N5_8 r1 r4 + N5_9 r1 r5 + N5_10 r1 r6 + N5_11 r2 r3 + N5_12 r2 r4 + N5_13 r2 r5 + N5_14 r2 r6 + N5_15 r3 r4 + N5_16 r3 r5 + N5_17 r3 r6 + N5_18 r4 r5 + N5_19 r4 r6 + N5_20 r5 r6) := by
    simp only [aS5_0, aS5_1, aS5_2, aS5_3, aS5_4, aS5_5, aS5_6, aP5_0, aP5_1, aP5_2, aP5_3, aP5_4, aP5_5, aP5_6, aP5_7, aP5_8, aP5_9, aP5_10, aP5_11, aP5_12, aP5_13, aP5_14, aP5_15, aP5_16, aP5_17, aP5_18, aP5_19, aP5_20, L5_0, L5_1, L5_2, L5_3, L5_4, L5_5, L5_6]
    ring
  have bS0 : aS5_0 r0 ≤ MS5_0 := CaseSplit.le_mxr (aS5_0) 10 r0 (by omega)
  have bS1 : aS5_1 r1 ≤ MS5_1 := CaseSplit.le_mxr (aS5_1) 12 r1 (by omega)
  have bS2 : aS5_2 r2 ≤ MS5_2 := CaseSplit.le_mxr (aS5_2) 16 r2 (by omega)
  have bS3 : aS5_3 r3 ≤ MS5_3 := CaseSplit.le_mxr (aS5_3) 18 r3 (by omega)
  have bS4 : aS5_4 r4 ≤ MS5_4 := CaseSplit.le_mxr (aS5_4) 22 r4 (by omega)
  have bS5 : aS5_5 r5 ≤ MS5_5 := CaseSplit.le_mxr (aS5_5) 28 r5 (by omega)
  have bS6 : aS5_6 r6 ≤ MS5_6 := CaseSplit.le_mxr (aS5_6) 30 r6 (by omega)
  have bP0 : aP5_0 r0 r1 ≤ MP5_0 := CaseSplit.le_mxr2 (aP5_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP5_1 r0 r2 ≤ MP5_1 := CaseSplit.le_mxr2 (aP5_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP5_2 r0 r3 ≤ MP5_2 := CaseSplit.le_mxr2 (aP5_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP5_3 r0 r4 ≤ MP5_3 := CaseSplit.le_mxr2 (aP5_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP5_4 r0 r5 ≤ MP5_4 := CaseSplit.le_mxr2 (aP5_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP5_5 r0 r6 ≤ MP5_5 := CaseSplit.le_mxr2 (aP5_5) 10 30 r0 r6 (by omega) (by omega)
  have bP6 : aP5_6 r1 r2 ≤ MP5_6 := CaseSplit.le_mxr2 (aP5_6) 12 16 r1 r2 (by omega) (by omega)
  have bP7 : aP5_7 r1 r3 ≤ MP5_7 := CaseSplit.le_mxr2 (aP5_7) 12 18 r1 r3 (by omega) (by omega)
  have bP8 : aP5_8 r1 r4 ≤ MP5_8 := CaseSplit.le_mxr2 (aP5_8) 12 22 r1 r4 (by omega) (by omega)
  have bP9 : aP5_9 r1 r5 ≤ MP5_9 := CaseSplit.le_mxr2 (aP5_9) 12 28 r1 r5 (by omega) (by omega)
  have bP10 : aP5_10 r1 r6 ≤ MP5_10 := CaseSplit.le_mxr2 (aP5_10) 12 30 r1 r6 (by omega) (by omega)
  have bP11 : aP5_11 r2 r3 ≤ MP5_11 := CaseSplit.le_mxr2 (aP5_11) 16 18 r2 r3 (by omega) (by omega)
  have bP12 : aP5_12 r2 r4 ≤ MP5_12 := CaseSplit.le_mxr2 (aP5_12) 16 22 r2 r4 (by omega) (by omega)
  have bP13 : aP5_13 r2 r5 ≤ MP5_13 := CaseSplit.le_mxr2 (aP5_13) 16 28 r2 r5 (by omega) (by omega)
  have bP14 : aP5_14 r2 r6 ≤ MP5_14 := CaseSplit.le_mxr2 (aP5_14) 16 30 r2 r6 (by omega) (by omega)
  have bP15 : aP5_15 r3 r4 ≤ MP5_15 := CaseSplit.le_mxr2 (aP5_15) 18 22 r3 r4 (by omega) (by omega)
  have bP16 : aP5_16 r3 r5 ≤ MP5_16 := CaseSplit.le_mxr2 (aP5_16) 18 28 r3 r5 (by omega) (by omega)
  have bP17 : aP5_17 r3 r6 ≤ MP5_17 := CaseSplit.le_mxr2 (aP5_17) 18 30 r3 r6 (by omega) (by omega)
  have bP18 : aP5_18 r4 r5 ≤ MP5_18 := CaseSplit.le_mxr2 (aP5_18) 22 28 r4 r5 (by omega) (by omega)
  have bP19 : aP5_19 r4 r6 ≤ MP5_19 := CaseSplit.le_mxr2 (aP5_19) 22 30 r4 r6 (by omega) (by omega)
  have bP20 : aP5_20 r5 r6 ≤ MP5_20 := CaseSplit.le_mxr2 (aP5_20) 28 30 r5 r6 (by omega) (by omega)
  have hrhs : rhs5 = (∑ t ∈ Finset.range n5, w5 t) + 7 * (n5 : ℤ) := rfl
  have hc := cert5
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, hn15, hn16, hn17, hn18, hn19, hn20, bS0, bS1, bS2, bS3, bS4, bS5, bS6, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14, bP15, bP16, bP17, bP18, bP19, bP20]

end IncCert31
