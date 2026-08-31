/-
INCREMENT-WIDTH CERTIFICATE, step 29->31, case 16 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_29_31.json, which re-derives every number
from the primes alone).

Machine 31, INCREMENT width 65 = F_2(29) + s_min(31) = 55 + 10,
held gears [5, 7] at phases [2, 2].  Free gears [11, 13, 17, 19, 23, 29, 31].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 384.
-/
import IncCert31B

namespace IncCert31

/-! ### case 16: held gears at phases [2, 2] -/

def p16 : List ℕ := [0, 1, 3, 5, 8, 10, 15, 16, 21, 23, 26, 28, 30, 31, 33, 35, 36, 38, 40, 43, 45, 50, 51, 56, 58, 61, 63]
def q16 (t : ℕ) : ℕ := p16.getD t 0
def n16 : ℕ := 27
def yl16 : List ℤ := [62, 35, 259, 73, 166, 156, 170, 0, 156, 111, 259, 256, 183, 287, 384, 353, 142, 259, 287, 131, 152, 62, 170, 111, 166, 28, 263]
def w16 (t : ℕ) : ℤ := yl16.getD t 0
def ul16 : List ℤ := [0, (-10), (-10), 0, 0, (-10), (-10), 7, (-10), 0, (-10), 7, (-10), (-7), 10, (-7), 3, (-7), 10, (-7), (-7), 0, (-7), 0, (-10), 0, (-10), (-10), (-10), (-10), (-10), 0, 0, (-10), (-10), (-10), 0, (-10), (-10), (-10), (-10), 0, 10, 0, 0, 0, 10, 10, 0, 0, 0, 0, 0, 0, 0, (-121), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, (-10), 0, (-10), (-10), (-10), 0, 0, (-10), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-48), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 578, 346, 692, 692, 692, 602, 484, 692, 692, 643, 692, 692, 329, 595, 682, 692, 626, (-692), (-692), (-692), (-692), (-692), (-692), (-692), (-692), (-692), (-692), (-692), (-692), (-692), 643, 643, 616, 643, 643, 346, 540, 446, 643, 612, 643, 643, 325, 536, 640, 643, 643, 643, 643, (-643), (-643), (-643), (-643), (-643), (-643), (-643), (-643), (-643), (-643), (-643), (-643), (-643), 62, 671, 560, 671, 671, 671, 671, 671, (-10), 671, 671, 671, 671, (-10), 671, 671, 671, 671, 671, 657, 671, 156, 671, (-671), (-671), (-709), (-671), (-671), (-671), (-671), (-671), (-671), (-671), (-671), (-671), (-671), 363, 363, 363, 363, 66, 363, 363, 363, 270, 363, 363, 0, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 242, 363, 363, 363, 363, (-363), (-363), (-363), (-363), (-363), (-363), (-363), (-363), (-363), (-363), (-363), (-363), (-363), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-266), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-519), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-256), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-225), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-142), 0, 0, 0, 0, 0, 0, (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), (-10), 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 547, 495, 547, 526, 329, 547, 547, 83, 277, 187, 547, 547, 173, 131, 547, 547, 356, 547, 187, 280, 21, 426, 536, 547, 547, 329, 547, 547, 187, 547, 547, (-7), 152, 173, 173, 173, 173, (-97), (-242), 173, 173, 173, 121, 173, 173, (-197), (-93), (-187), 173, 173, 173, 173, (-55), (-66), 173, 173, 173, (-66), 173, 0]
def u16 (k : ℕ) : ℤ := ul16.getD k 0

def c16_0 (r t : ℕ) : Bool := gb11 r (q16 t)
def c16_1 (r t : ℕ) : Bool := gb13 r (q16 t)
def c16_2 (r t : ℕ) : Bool := gb17 r (q16 t)
def c16_3 (r t : ℕ) : Bool := gb19 r (q16 t)
def c16_4 (r t : ℕ) : Bool := gb23 r (q16 t)
def c16_5 (r t : ℕ) : Bool := gb29 r (q16 t)
def c16_6 (r t : ℕ) : Bool := gb31 r (q16 t)

def S16_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (w16 t + 10) * (if c16_0 r t then 1 else 0)
def S16_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (w16 t + 10) * (if c16_1 r t then 1 else 0)
def S16_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (w16 t + 10) * (if c16_2 r t then 1 else 0)
def S16_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (w16 t + 10) * (if c16_3 r t then 1 else 0)
def S16_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (w16 t + 10) * (if c16_4 r t then 1 else 0)
def S16_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (w16 t + 10) * (if c16_5 r t then 1 else 0)
def S16_6 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (w16 t + 10) * (if c16_6 r t then 1 else 0)

def L16_0 (r : ℕ) : ℤ := u16 (13 + r) + u16 (41 + r) + u16 (71 + r) + u16 (105 + r) + u16 (145 + r) + u16 (187 + r)
def L16_1 (r : ℕ) : ℤ := u16 (0 + r) + u16 (215 + r) + u16 (247 + r) + u16 (283 + r) + u16 (325 + r) + u16 (369 + r)
def L16_2 (r : ℕ) : ℤ := u16 (24 + r) + u16 (198 + r) + u16 (401 + r) + u16 (441 + r) + u16 (487 + r) + u16 (535 + r)
def L16_3 (r : ℕ) : ℤ := u16 (52 + r) + u16 (228 + r) + u16 (382 + r) + u16 (575 + r) + u16 (623 + r) + u16 (673 + r)
def L16_4 (r : ℕ) : ℤ := u16 (82 + r) + u16 (260 + r) + u16 (418 + r) + u16 (552 + r) + u16 (721 + r) + u16 (775 + r)
def L16_5 (r : ℕ) : ℤ := u16 (116 + r) + u16 (296 + r) + u16 (458 + r) + u16 (594 + r) + u16 (692 + r) + u16 (829 + r)
def L16_6 (r : ℕ) : ℤ := u16 (156 + r) + u16 (338 + r) + u16 (504 + r) + u16 (642 + r) + u16 (744 + r) + u16 (798 + r)

def aS16_0 (r : ℕ) : ℤ := S16_0 r - L16_0 r
def MS16_0 : ℤ := CaseSplit.mxr (aS16_0) 10
def aS16_1 (r : ℕ) : ℤ := S16_1 r - L16_1 r
def MS16_1 : ℤ := CaseSplit.mxr (aS16_1) 12
def aS16_2 (r : ℕ) : ℤ := S16_2 r - L16_2 r
def MS16_2 : ℤ := CaseSplit.mxr (aS16_2) 16
def aS16_3 (r : ℕ) : ℤ := S16_3 r - L16_3 r
def MS16_3 : ℤ := CaseSplit.mxr (aS16_3) 18
def aS16_4 (r : ℕ) : ℤ := S16_4 r - L16_4 r
def MS16_4 : ℤ := CaseSplit.mxr (aS16_4) 22
def aS16_5 (r : ℕ) : ℤ := S16_5 r - L16_5 r
def MS16_5 : ℤ := CaseSplit.mxr (aS16_5) 28
def aS16_6 (r : ℕ) : ℤ := S16_6 r - L16_6 r
def MS16_6 : ℤ := CaseSplit.mxr (aS16_6) 30

def N16_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (if c16_0 ra t && c16_1 rb t then 1 else 0)
def aP16_0 (ra rb : ℕ) : ℤ := -(10) * N16_0 ra rb + u16 (0 + rb) + u16 (13 + ra)
def MP16_0 : ℤ := CaseSplit.mxr2 (aP16_0) 10 12
def N16_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (if c16_0 ra t && c16_2 rb t then 1 else 0)
def aP16_1 (ra rb : ℕ) : ℤ := -(10) * N16_1 ra rb + u16 (24 + rb) + u16 (41 + ra)
def MP16_1 : ℤ := CaseSplit.mxr2 (aP16_1) 10 16
def N16_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (if c16_0 ra t && c16_3 rb t then 1 else 0)
def aP16_2 (ra rb : ℕ) : ℤ := -(10) * N16_2 ra rb + u16 (52 + rb) + u16 (71 + ra)
def MP16_2 : ℤ := CaseSplit.mxr2 (aP16_2) 10 18
def N16_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (if c16_0 ra t && c16_4 rb t then 1 else 0)
def aP16_3 (ra rb : ℕ) : ℤ := -(10) * N16_3 ra rb + u16 (82 + rb) + u16 (105 + ra)
def MP16_3 : ℤ := CaseSplit.mxr2 (aP16_3) 10 22
def N16_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (if c16_0 ra t && c16_5 rb t then 1 else 0)
def aP16_4 (ra rb : ℕ) : ℤ := -(10) * N16_4 ra rb + u16 (116 + rb) + u16 (145 + ra)
def MP16_4 : ℤ := CaseSplit.mxr2 (aP16_4) 10 28
def N16_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (if c16_0 ra t && c16_6 rb t then 1 else 0)
def aP16_5 (ra rb : ℕ) : ℤ := -(10) * N16_5 ra rb + u16 (156 + rb) + u16 (187 + ra)
def MP16_5 : ℤ := CaseSplit.mxr2 (aP16_5) 10 30
def P16_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (if c16_1 ra t && c16_2 rb t then 1 else 0)
def C16_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (if c16_1 ra t && c16_2 rb t && c16_0 s t then 1 else 0)
def M16_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C16_6 ra rb) 10
def E16_6 : List ℕ := [25, 31, 61, 67, 72, 83, 140, 151, 156, 162, 192, 198]
def N16_6 (ra rb : ℕ) : ℤ := if E16_6.contains (ra * 17 + rb) = true then P16_6 ra rb - M16_6 ra rb else 0
def aP16_6 (ra rb : ℕ) : ℤ := -(10) * N16_6 ra rb + u16 (198 + rb) + u16 (215 + ra)
def MP16_6 : ℤ := CaseSplit.mxr2 (aP16_6) 12 16
def P16_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (if c16_1 ra t && c16_3 rb t then 1 else 0)
def C16_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (if c16_1 ra t && c16_3 rb t && c16_0 s t then 1 else 0)
def M16_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C16_7 ra rb) 10
def E16_7 : List ℕ := [7, 10, 31, 37, 60, 71, 86, 107, 110, 113, 144, 147, 150, 152, 178, 186, 220, 226, 228, 231]
def N16_7 (ra rb : ℕ) : ℤ := if E16_7.contains (ra * 19 + rb) = true then P16_7 ra rb - M16_7 ra rb else 0
def aP16_7 (ra rb : ℕ) : ℤ := -(10) * N16_7 ra rb + u16 (228 + rb) + u16 (247 + ra)
def MP16_7 : ℤ := CaseSplit.mxr2 (aP16_7) 12 18
def P16_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (if c16_1 ra t && c16_4 rb t then 1 else 0)
def C16_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (if c16_1 ra t && c16_4 rb t && c16_0 s t then 1 else 0)
def M16_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C16_8 ra rb) 10
def E16_8 : List ℕ := [50]
def N16_8 (ra rb : ℕ) : ℤ := if E16_8.contains (ra * 23 + rb) = true then P16_8 ra rb - M16_8 ra rb else 0
def aP16_8 (ra rb : ℕ) : ℤ := -(10) * N16_8 ra rb + u16 (260 + rb) + u16 (283 + ra)
def MP16_8 : ℤ := CaseSplit.mxr2 (aP16_8) 12 22
def P16_9 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (if c16_1 ra t && c16_5 rb t then 1 else 0)
def C16_9 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (if c16_1 ra t && c16_5 rb t && c16_0 s t then 1 else 0)
def M16_9 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C16_9 ra rb) 10
def E16_9 : List ℕ := [52, 313]
def N16_9 (ra rb : ℕ) : ℤ := if E16_9.contains (ra * 29 + rb) = true then P16_9 ra rb - M16_9 ra rb else 0
def aP16_9 (ra rb : ℕ) : ℤ := -(10) * N16_9 ra rb + u16 (296 + rb) + u16 (325 + ra)
def MP16_9 : ℤ := CaseSplit.mxr2 (aP16_9) 12 28
def P16_10 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (if c16_1 ra t && c16_6 rb t then 1 else 0)
def C16_10 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (if c16_1 ra t && c16_6 rb t && c16_0 s t then 1 else 0)
def M16_10 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C16_10 ra rb) 10
def E16_10 : List ℕ := []
def N16_10 (ra rb : ℕ) : ℤ := if E16_10.contains (ra * 31 + rb) = true then P16_10 ra rb - M16_10 ra rb else 0
def aP16_10 (ra rb : ℕ) : ℤ := -(10) * N16_10 ra rb + u16 (338 + rb) + u16 (369 + ra)
def MP16_10 : ℤ := CaseSplit.mxr2 (aP16_10) 12 30
def N16_11 (_ra _rb : ℕ) : ℤ := 0
def aP16_11 (ra rb : ℕ) : ℤ := -(10) * N16_11 ra rb + u16 (382 + rb) + u16 (401 + ra)
def MP16_11 : ℤ := CaseSplit.mxr2 (aP16_11) 16 18
def N16_12 (_ra _rb : ℕ) : ℤ := 0
def aP16_12 (ra rb : ℕ) : ℤ := -(10) * N16_12 ra rb + u16 (418 + rb) + u16 (441 + ra)
def MP16_12 : ℤ := CaseSplit.mxr2 (aP16_12) 16 22
def N16_13 (_ra _rb : ℕ) : ℤ := 0
def aP16_13 (ra rb : ℕ) : ℤ := -(10) * N16_13 ra rb + u16 (458 + rb) + u16 (487 + ra)
def MP16_13 : ℤ := CaseSplit.mxr2 (aP16_13) 16 28
def N16_14 (_ra _rb : ℕ) : ℤ := 0
def aP16_14 (ra rb : ℕ) : ℤ := -(10) * N16_14 ra rb + u16 (504 + rb) + u16 (535 + ra)
def MP16_14 : ℤ := CaseSplit.mxr2 (aP16_14) 16 30
def N16_15 (_ra _rb : ℕ) : ℤ := 0
def aP16_15 (ra rb : ℕ) : ℤ := -(10) * N16_15 ra rb + u16 (552 + rb) + u16 (575 + ra)
def MP16_15 : ℤ := CaseSplit.mxr2 (aP16_15) 18 22
def N16_16 (_ra _rb : ℕ) : ℤ := 0
def aP16_16 (ra rb : ℕ) : ℤ := -(10) * N16_16 ra rb + u16 (594 + rb) + u16 (623 + ra)
def MP16_16 : ℤ := CaseSplit.mxr2 (aP16_16) 18 28
def N16_17 (_ra _rb : ℕ) : ℤ := 0
def aP16_17 (ra rb : ℕ) : ℤ := -(10) * N16_17 ra rb + u16 (642 + rb) + u16 (673 + ra)
def MP16_17 : ℤ := CaseSplit.mxr2 (aP16_17) 18 30
def N16_18 (_ra _rb : ℕ) : ℤ := 0
def aP16_18 (ra rb : ℕ) : ℤ := -(10) * N16_18 ra rb + u16 (692 + rb) + u16 (721 + ra)
def MP16_18 : ℤ := CaseSplit.mxr2 (aP16_18) 22 28
def N16_19 (_ra _rb : ℕ) : ℤ := 0
def aP16_19 (ra rb : ℕ) : ℤ := -(10) * N16_19 ra rb + u16 (744 + rb) + u16 (775 + ra)
def MP16_19 : ℤ := CaseSplit.mxr2 (aP16_19) 22 30
def N16_20 (_ra _rb : ℕ) : ℤ := 0
def aP16_20 (ra rb : ℕ) : ℤ := -(10) * N16_20 ra rb + u16 (798 + rb) + u16 (829 + ra)
def MP16_20 : ℤ := CaseSplit.mxr2 (aP16_20) 28 30

def rhs16 : ℤ := (∑ t ∈ Finset.range n16, w16 t) + 10 * (n16 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn16 : ∀ t, t < n16 → (0 : ℤ) ≤ w16 t := by decide
theorem plt16 : ∀ t, t < n16 → q16 t < 65 := by decide
theorem pfree16_5 : ∀ t, t < n16 → gb5 2 (q16 t) = false := by decide
theorem pfree16_7 : ∀ t, t < n16 → gb7 2 (q16 t) = false := by decide
theorem MSv16_0 : MS16_0 = 1071 := by decide +kernel
theorem MSv16_1 : MS16_1 = 3160 := by decide +kernel
theorem MSv16_2 : MS16_2 = (-1) := by decide +kernel
theorem MSv16_3 : MS16_3 = 0 := by decide +kernel
theorem MSv16_4 : MS16_4 = 0 := by decide +kernel
theorem MSv16_5 : MS16_5 = 0 := by decide +kernel
theorem MSv16_6 : MS16_6 = 0 := by decide +kernel
theorem MPv16_0 : MP16_0 = 0 := by decide +kernel
theorem MPv16_1 : MP16_1 = 0 := by decide +kernel
theorem MPv16_2 : MP16_2 = 0 := by decide +kernel
theorem MPv16_3 : MP16_3 = 0 := by decide +kernel
theorem MPv16_4 : MP16_4 = 0 := by decide +kernel
theorem MPv16_5 : MP16_5 = 0 := by decide +kernel
theorem MPv16_6 : MP16_6 = 0 := by decide +kernel
theorem MPv16_7 : MP16_7 = 0 := by decide +kernel
theorem MPv16_8 : MP16_8 = 0 := by decide +kernel
theorem MPv16_9 : MP16_9 = 0 := by decide +kernel
theorem MPv16_10 : MP16_10 = 0 := by decide +kernel
theorem MPv16_11 : MP16_11 = 0 := by decide +kernel
theorem MPv16_12 : MP16_12 = 0 := by decide +kernel
theorem MPv16_13 : MP16_13 = 0 := by decide +kernel
theorem MPv16_14 : MP16_14 = 0 := by decide +kernel
theorem MPv16_15 : MP16_15 = 0 := by decide +kernel
theorem MPv16_16 : MP16_16 = 0 := by decide +kernel
theorem MPv16_17 : MP16_17 = 0 := by decide +kernel
theorem MPv16_18 : MP16_18 = 0 := by decide +kernel
theorem MPv16_19 : MP16_19 = 0 := by decide +kernel
theorem MPv16_20 : MP16_20 = 720 := by decide +kernel
theorem rhsv16 : rhs16 = 4951 := by decide +kernel

/-- **The case-16 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/384.
    (Scaled by the common denominator 384: 4950 < 4951.) -/
theorem cert16 : MS16_0 + MS16_1 + MS16_2 + MS16_3 + MS16_4 + MS16_5 + MS16_6 + MP16_0 + MP16_1 + MP16_2 + MP16_3 + MP16_4 + MP16_5 + MP16_6 + MP16_7 + MP16_8 + MP16_9 + MP16_10 + MP16_11 + MP16_12 + MP16_13 + MP16_14 + MP16_15 + MP16_16 + MP16_17 + MP16_18 + MP16_19 + MP16_20 < rhs16 := by
  rw [MSv16_0, MSv16_1, MSv16_2, MSv16_3, MSv16_4, MSv16_5, MSv16_6, MPv16_0, MPv16_1, MPv16_2, MPv16_3, MPv16_4, MPv16_5, MPv16_6, MPv16_7, MPv16_8, MPv16_9, MPv16_10, MPv16_11, MPv16_12, MPv16_13, MPv16_14, MPv16_15, MPv16_16, MPv16_17, MPv16_18, MPv16_19, MPv16_20, rhsv16]
  decide

def Dg16 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := (if c16_0 r0 t then 1 else 0) + (if c16_1 r1 t then 1 else 0) + (if c16_2 r2 t then 1 else 0) + (if c16_3 r3 t then 1 else 0) + (if c16_4 r4 t then 1 else 0) + (if c16_5 r5 t then 1 else 0) + (if c16_6 r6 t then 1 else 0)
def Wl16_0 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c16_0 r0 t && c16_1 r1 t then 1 else 0
def Wl16_1 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c16_0 r0 t && c16_2 r2 t then 1 else 0
def Wl16_2 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c16_0 r0 t && c16_3 r3 t then 1 else 0
def Wl16_3 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c16_0 r0 t && c16_4 r4 t then 1 else 0
def Wl16_4 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c16_0 r0 t && c16_5 r5 t then 1 else 0
def Wl16_5 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c16_0 r0 t && c16_6 r6 t then 1 else 0
def Wl16_6 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c16_0 r0 t && c16_1 r1 t && c16_2 r2 t then 1 else 0
def Wl16_7 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c16_0 r0 t && c16_1 r1 t && c16_3 r3 t then 1 else 0
def Wl16_8 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c16_0 r0 t && c16_1 r1 t && c16_4 r4 t then 1 else 0
def Wl16_9 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c16_0 r0 t && c16_1 r1 t && c16_5 r5 t then 1 else 0
def Wl16_10 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c16_0 r0 t && c16_1 r1 t && c16_6 r6 t then 1 else 0
def Wl16_11 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c16_0 r0 t && !c16_1 r1 t && c16_2 r2 t && c16_3 r3 t then 1 else 0
def Wl16_12 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c16_0 r0 t && !c16_1 r1 t && c16_2 r2 t && c16_4 r4 t then 1 else 0
def Wl16_13 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c16_0 r0 t && !c16_1 r1 t && c16_2 r2 t && c16_5 r5 t then 1 else 0
def Wl16_14 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c16_0 r0 t && !c16_1 r1 t && c16_2 r2 t && c16_6 r6 t then 1 else 0
def Wl16_15 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c16_0 r0 t && !c16_1 r1 t && !c16_2 r2 t && c16_3 r3 t && c16_4 r4 t then 1 else 0
def Wl16_16 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c16_0 r0 t && !c16_1 r1 t && !c16_2 r2 t && c16_3 r3 t && c16_5 r5 t then 1 else 0
def Wl16_17 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c16_0 r0 t && !c16_1 r1 t && !c16_2 r2 t && c16_3 r3 t && c16_6 r6 t then 1 else 0
def Wl16_18 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c16_0 r0 t && !c16_1 r1 t && !c16_2 r2 t && !c16_3 r3 t && c16_4 r4 t && c16_5 r5 t then 1 else 0
def Wl16_19 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c16_0 r0 t && !c16_1 r1 t && !c16_2 r2 t && !c16_3 r3 t && c16_4 r4 t && c16_6 r6 t then 1 else 0
def Wl16_20 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c16_0 r0 t && !c16_1 r1 t && !c16_2 r2 t && !c16_3 r3 t && !c16_4 r4 t && c16_5 r5 t && c16_6 r6 t then 1 else 0

/-- **No configuration blocks the whole window in case 16.** -/
theorem nocov16 {r0 r1 r2 r3 r4 r5 r6 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29) (h6 : r6 < 31)
    (hcov : ∀ t, t < n16 → (c16_0 r0 t || c16_1 r1 t || c16_2 r2 t || c16_3 r3 t || c16_4 r4 t || c16_5 r5 t || c16_6 r6 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n16, (1 : ℤ) + (Wl16_0 r0 r1 r2 r3 r4 r5 r6 t + Wl16_1 r0 r1 r2 r3 r4 r5 r6 t + Wl16_2 r0 r1 r2 r3 r4 r5 r6 t + Wl16_3 r0 r1 r2 r3 r4 r5 r6 t + Wl16_4 r0 r1 r2 r3 r4 r5 r6 t + Wl16_5 r0 r1 r2 r3 r4 r5 r6 t + Wl16_6 r0 r1 r2 r3 r4 r5 r6 t + Wl16_7 r0 r1 r2 r3 r4 r5 r6 t + Wl16_8 r0 r1 r2 r3 r4 r5 r6 t + Wl16_9 r0 r1 r2 r3 r4 r5 r6 t + Wl16_10 r0 r1 r2 r3 r4 r5 r6 t + Wl16_11 r0 r1 r2 r3 r4 r5 r6 t + Wl16_12 r0 r1 r2 r3 r4 r5 r6 t + Wl16_13 r0 r1 r2 r3 r4 r5 r6 t + Wl16_14 r0 r1 r2 r3 r4 r5 r6 t + Wl16_15 r0 r1 r2 r3 r4 r5 r6 t + Wl16_16 r0 r1 r2 r3 r4 r5 r6 t + Wl16_17 r0 r1 r2 r3 r4 r5 r6 t + Wl16_18 r0 r1 r2 r3 r4 r5 r6 t + Wl16_19 r0 r1 r2 r3 r4 r5 r6 t + Wl16_20 r0 r1 r2 r3 r4 r5 r6 t) ≤ Dg16 r0 r1 r2 r3 r4 r5 r6 t := by
    intro t ht
    simp only [Wl16_0, Wl16_1, Wl16_2, Wl16_3, Wl16_4, Wl16_5, Wl16_6, Wl16_7, Wl16_8, Wl16_9, Wl16_10, Wl16_11, Wl16_12, Wl16_13, Wl16_14, Wl16_15, Wl16_16, Wl16_17, Wl16_18, Wl16_19, Wl16_20, Dg16]
    exact CaseSplit.lowest7 _ _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n16, (1 : ℤ) ≤ Dg16 r0 r1 r2 r3 r4 r5 r6 t := by
    intro t ht
    simp only [Dg16]
    exact CaseSplit.degpos7 _ _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n16 : ℤ) + ((∑ t ∈ Finset.range n16, Wl16_0 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n16, Wl16_1 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n16, Wl16_2 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n16, Wl16_3 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n16, Wl16_4 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n16, Wl16_5 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n16, Wl16_6 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n16, Wl16_7 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n16, Wl16_8 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n16, Wl16_9 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n16, Wl16_10 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n16, Wl16_11 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n16, Wl16_12 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n16, Wl16_13 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n16, Wl16_14 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n16, Wl16_15 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n16, Wl16_16 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n16, Wl16_17 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n16, Wl16_18 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n16, Wl16_19 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n16, Wl16_20 r0 r1 r2 r3 r4 r5 r6 t)) ≤ ∑ t ∈ Finset.range n16, Dg16 r0 r1 r2 r3 r4 r5 r6 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N16_0 r0 r1 ≤ ∑ t ∈ Finset.range n16, Wl16_0 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N16_0, Wl16_0, le_refl]
  have hn1 : N16_1 r0 r2 ≤ ∑ t ∈ Finset.range n16, Wl16_1 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N16_1, Wl16_1, le_refl]
  have hn2 : N16_2 r0 r3 ≤ ∑ t ∈ Finset.range n16, Wl16_2 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N16_2, Wl16_2, le_refl]
  have hn3 : N16_3 r0 r4 ≤ ∑ t ∈ Finset.range n16, Wl16_3 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N16_3, Wl16_3, le_refl]
  have hn4 : N16_4 r0 r5 ≤ ∑ t ∈ Finset.range n16, Wl16_4 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N16_4, Wl16_4, le_refl]
  have hn5 : N16_5 r0 r6 ≤ ∑ t ∈ Finset.range n16, Wl16_5 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N16_5, Wl16_5, le_refl]
  have hn6 : N16_6 r1 r2 ≤ ∑ t ∈ Finset.range n16, Wl16_6 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n16, Wl16_6 r0 r1 r2 r3 r4 r5 r6 t
        = (if c16_1 r1 t && c16_2 r2 t then (1:ℤ) else 0)
          - (if c16_1 r1 t && c16_2 r2 t && c16_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl16_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n16, Wl16_6 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl16_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n16, Wl16_6 r0 r1 r2 r3 r4 r5 r6 t
        = P16_6 r1 r2 - C16_6 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P16_6, C16_6]
    have hm : C16_6 r1 r2 r0 ≤ M16_6 r1 r2 :=
      CaseSplit.le_mxr (C16_6 r1 r2) 10 r0 (by omega)
    simp only [N16_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N16_7 r1 r3 ≤ ∑ t ∈ Finset.range n16, Wl16_7 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n16, Wl16_7 r0 r1 r2 r3 r4 r5 r6 t
        = (if c16_1 r1 t && c16_3 r3 t then (1:ℤ) else 0)
          - (if c16_1 r1 t && c16_3 r3 t && c16_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl16_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n16, Wl16_7 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl16_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n16, Wl16_7 r0 r1 r2 r3 r4 r5 r6 t
        = P16_7 r1 r3 - C16_7 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P16_7, C16_7]
    have hm : C16_7 r1 r3 r0 ≤ M16_7 r1 r3 :=
      CaseSplit.le_mxr (C16_7 r1 r3) 10 r0 (by omega)
    simp only [N16_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N16_8 r1 r4 ≤ ∑ t ∈ Finset.range n16, Wl16_8 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n16, Wl16_8 r0 r1 r2 r3 r4 r5 r6 t
        = (if c16_1 r1 t && c16_4 r4 t then (1:ℤ) else 0)
          - (if c16_1 r1 t && c16_4 r4 t && c16_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl16_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n16, Wl16_8 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl16_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n16, Wl16_8 r0 r1 r2 r3 r4 r5 r6 t
        = P16_8 r1 r4 - C16_8 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P16_8, C16_8]
    have hm : C16_8 r1 r4 r0 ≤ M16_8 r1 r4 :=
      CaseSplit.le_mxr (C16_8 r1 r4) 10 r0 (by omega)
    simp only [N16_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N16_9 r1 r5 ≤ ∑ t ∈ Finset.range n16, Wl16_9 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n16, Wl16_9 r0 r1 r2 r3 r4 r5 r6 t
        = (if c16_1 r1 t && c16_5 r5 t then (1:ℤ) else 0)
          - (if c16_1 r1 t && c16_5 r5 t && c16_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl16_9]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n16, Wl16_9 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl16_9]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n16, Wl16_9 r0 r1 r2 r3 r4 r5 r6 t
        = P16_9 r1 r5 - C16_9 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P16_9, C16_9]
    have hm : C16_9 r1 r5 r0 ≤ M16_9 r1 r5 :=
      CaseSplit.le_mxr (C16_9 r1 r5) 10 r0 (by omega)
    simp only [N16_9]
    split
    · rw [hL]; omega
    · exact hnn
  have hn10 : N16_10 r1 r6 ≤ ∑ t ∈ Finset.range n16, Wl16_10 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n16, Wl16_10 r0 r1 r2 r3 r4 r5 r6 t
        = (if c16_1 r1 t && c16_6 r6 t then (1:ℤ) else 0)
          - (if c16_1 r1 t && c16_6 r6 t && c16_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl16_10]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n16, Wl16_10 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl16_10]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n16, Wl16_10 r0 r1 r2 r3 r4 r5 r6 t
        = P16_10 r1 r6 - C16_10 r1 r6 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P16_10, C16_10]
    have hm : C16_10 r1 r6 r0 ≤ M16_10 r1 r6 :=
      CaseSplit.le_mxr (C16_10 r1 r6) 10 r0 (by omega)
    simp only [N16_10]
    split
    · rw [hL]; omega
    · exact hnn
  have hn11 : N16_11 r2 r3 ≤ ∑ t ∈ Finset.range n16, Wl16_11 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N16_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl16_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N16_12 r2 r4 ≤ ∑ t ∈ Finset.range n16, Wl16_12 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N16_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl16_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N16_13 r2 r5 ≤ ∑ t ∈ Finset.range n16, Wl16_13 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N16_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl16_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N16_14 r2 r6 ≤ ∑ t ∈ Finset.range n16, Wl16_14 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N16_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl16_14]
    exact CaseSplit.ind_nonneg _
  have hn15 : N16_15 r3 r4 ≤ ∑ t ∈ Finset.range n16, Wl16_15 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N16_15]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl16_15]
    exact CaseSplit.ind_nonneg _
  have hn16 : N16_16 r3 r5 ≤ ∑ t ∈ Finset.range n16, Wl16_16 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N16_16]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl16_16]
    exact CaseSplit.ind_nonneg _
  have hn17 : N16_17 r3 r6 ≤ ∑ t ∈ Finset.range n16, Wl16_17 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N16_17]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl16_17]
    exact CaseSplit.ind_nonneg _
  have hn18 : N16_18 r4 r5 ≤ ∑ t ∈ Finset.range n16, Wl16_18 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N16_18]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl16_18]
    exact CaseSplit.ind_nonneg _
  have hn19 : N16_19 r4 r6 ≤ ∑ t ∈ Finset.range n16, Wl16_19 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N16_19]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl16_19]
    exact CaseSplit.ind_nonneg _
  have hn20 : N16_20 r5 r6 ≤ ∑ t ∈ Finset.range n16, Wl16_20 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N16_20]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl16_20]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n16, (w16 t + 10) * Dg16 r0 r1 r2 r3 r4 r5 r6 t = S16_0 r0 + S16_1 r1 + S16_2 r2 + S16_3 r3 + S16_4 r4 + S16_5 r5 + S16_6 r6 := by
    simp only [S16_0, S16_1, S16_2, S16_3, S16_4, S16_5, S16_6, Dg16, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n16, (w16 t + 10) * Dg16 r0 r1 r2 r3 r4 r5 r6 t
      = (∑ t ∈ Finset.range n16, w16 t * Dg16 r0 r1 r2 r3 r4 r5 r6 t)
        + 10 * (∑ t ∈ Finset.range n16, Dg16 r0 r1 r2 r3 r4 r5 r6 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n16, w16 t)
      ≤ ∑ t ∈ Finset.range n16, w16 t * Dg16 r0 r1 r2 r3 r4 r5 r6 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg16 r0 r1 r2 r3 r4 r5 r6 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w16 t := wnn16 t (Finset.mem_range.mp ht)
    calc w16 t = w16 t * 1 := (mul_one _).symm
      _ ≤ w16 t * Dg16 r0 r1 r2 r3 r4 r5 r6 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS16_0 r0 + aS16_1 r1 + aS16_2 r2 + aS16_3 r3 + aS16_4 r4 + aS16_5 r5 + aS16_6 r6) + (aP16_0 r0 r1 + aP16_1 r0 r2 + aP16_2 r0 r3 + aP16_3 r0 r4 + aP16_4 r0 r5 + aP16_5 r0 r6 + aP16_6 r1 r2 + aP16_7 r1 r3 + aP16_8 r1 r4 + aP16_9 r1 r5 + aP16_10 r1 r6 + aP16_11 r2 r3 + aP16_12 r2 r4 + aP16_13 r2 r5 + aP16_14 r2 r6 + aP16_15 r3 r4 + aP16_16 r3 r5 + aP16_17 r3 r6 + aP16_18 r4 r5 + aP16_19 r4 r6 + aP16_20 r5 r6) = (S16_0 r0 + S16_1 r1 + S16_2 r2 + S16_3 r3 + S16_4 r4 + S16_5 r5 + S16_6 r6) - 10 * (N16_0 r0 r1 + N16_1 r0 r2 + N16_2 r0 r3 + N16_3 r0 r4 + N16_4 r0 r5 + N16_5 r0 r6 + N16_6 r1 r2 + N16_7 r1 r3 + N16_8 r1 r4 + N16_9 r1 r5 + N16_10 r1 r6 + N16_11 r2 r3 + N16_12 r2 r4 + N16_13 r2 r5 + N16_14 r2 r6 + N16_15 r3 r4 + N16_16 r3 r5 + N16_17 r3 r6 + N16_18 r4 r5 + N16_19 r4 r6 + N16_20 r5 r6) := by
    simp only [aS16_0, aS16_1, aS16_2, aS16_3, aS16_4, aS16_5, aS16_6, aP16_0, aP16_1, aP16_2, aP16_3, aP16_4, aP16_5, aP16_6, aP16_7, aP16_8, aP16_9, aP16_10, aP16_11, aP16_12, aP16_13, aP16_14, aP16_15, aP16_16, aP16_17, aP16_18, aP16_19, aP16_20, L16_0, L16_1, L16_2, L16_3, L16_4, L16_5, L16_6]
    ring
  have bS0 : aS16_0 r0 ≤ MS16_0 := CaseSplit.le_mxr (aS16_0) 10 r0 (by omega)
  have bS1 : aS16_1 r1 ≤ MS16_1 := CaseSplit.le_mxr (aS16_1) 12 r1 (by omega)
  have bS2 : aS16_2 r2 ≤ MS16_2 := CaseSplit.le_mxr (aS16_2) 16 r2 (by omega)
  have bS3 : aS16_3 r3 ≤ MS16_3 := CaseSplit.le_mxr (aS16_3) 18 r3 (by omega)
  have bS4 : aS16_4 r4 ≤ MS16_4 := CaseSplit.le_mxr (aS16_4) 22 r4 (by omega)
  have bS5 : aS16_5 r5 ≤ MS16_5 := CaseSplit.le_mxr (aS16_5) 28 r5 (by omega)
  have bS6 : aS16_6 r6 ≤ MS16_6 := CaseSplit.le_mxr (aS16_6) 30 r6 (by omega)
  have bP0 : aP16_0 r0 r1 ≤ MP16_0 := CaseSplit.le_mxr2 (aP16_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP16_1 r0 r2 ≤ MP16_1 := CaseSplit.le_mxr2 (aP16_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP16_2 r0 r3 ≤ MP16_2 := CaseSplit.le_mxr2 (aP16_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP16_3 r0 r4 ≤ MP16_3 := CaseSplit.le_mxr2 (aP16_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP16_4 r0 r5 ≤ MP16_4 := CaseSplit.le_mxr2 (aP16_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP16_5 r0 r6 ≤ MP16_5 := CaseSplit.le_mxr2 (aP16_5) 10 30 r0 r6 (by omega) (by omega)
  have bP6 : aP16_6 r1 r2 ≤ MP16_6 := CaseSplit.le_mxr2 (aP16_6) 12 16 r1 r2 (by omega) (by omega)
  have bP7 : aP16_7 r1 r3 ≤ MP16_7 := CaseSplit.le_mxr2 (aP16_7) 12 18 r1 r3 (by omega) (by omega)
  have bP8 : aP16_8 r1 r4 ≤ MP16_8 := CaseSplit.le_mxr2 (aP16_8) 12 22 r1 r4 (by omega) (by omega)
  have bP9 : aP16_9 r1 r5 ≤ MP16_9 := CaseSplit.le_mxr2 (aP16_9) 12 28 r1 r5 (by omega) (by omega)
  have bP10 : aP16_10 r1 r6 ≤ MP16_10 := CaseSplit.le_mxr2 (aP16_10) 12 30 r1 r6 (by omega) (by omega)
  have bP11 : aP16_11 r2 r3 ≤ MP16_11 := CaseSplit.le_mxr2 (aP16_11) 16 18 r2 r3 (by omega) (by omega)
  have bP12 : aP16_12 r2 r4 ≤ MP16_12 := CaseSplit.le_mxr2 (aP16_12) 16 22 r2 r4 (by omega) (by omega)
  have bP13 : aP16_13 r2 r5 ≤ MP16_13 := CaseSplit.le_mxr2 (aP16_13) 16 28 r2 r5 (by omega) (by omega)
  have bP14 : aP16_14 r2 r6 ≤ MP16_14 := CaseSplit.le_mxr2 (aP16_14) 16 30 r2 r6 (by omega) (by omega)
  have bP15 : aP16_15 r3 r4 ≤ MP16_15 := CaseSplit.le_mxr2 (aP16_15) 18 22 r3 r4 (by omega) (by omega)
  have bP16 : aP16_16 r3 r5 ≤ MP16_16 := CaseSplit.le_mxr2 (aP16_16) 18 28 r3 r5 (by omega) (by omega)
  have bP17 : aP16_17 r3 r6 ≤ MP16_17 := CaseSplit.le_mxr2 (aP16_17) 18 30 r3 r6 (by omega) (by omega)
  have bP18 : aP16_18 r4 r5 ≤ MP16_18 := CaseSplit.le_mxr2 (aP16_18) 22 28 r4 r5 (by omega) (by omega)
  have bP19 : aP16_19 r4 r6 ≤ MP16_19 := CaseSplit.le_mxr2 (aP16_19) 22 30 r4 r6 (by omega) (by omega)
  have bP20 : aP16_20 r5 r6 ≤ MP16_20 := CaseSplit.le_mxr2 (aP16_20) 28 30 r5 r6 (by omega) (by omega)
  have hrhs : rhs16 = (∑ t ∈ Finset.range n16, w16 t) + 10 * (n16 : ℤ) := rfl
  have hc := cert16
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, hn15, hn16, hn17, hn18, hn19, hn20, bS0, bS1, bS2, bS3, bS4, bS5, bS6, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14, bP15, bP16, bP17, bP18, bP19, bP20]

end IncCert31
