/-
INCREMENT-WIDTH CERTIFICATE, step 29->31, case 3 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_29_31.json, which re-derives every number
from the primes alone).

Machine 31, INCREMENT width 65 = F_2(29) + s_min(31) = 55 + 10,
held gears [5, 7] at phases [0, 3].  Free gears [11, 13, 17, 19, 23, 29, 31].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 12.
-/
import IncCert31B

namespace IncCert31

/-! ### case 3: held gears at phases [0, 3] -/

def p3 : List ℕ := [0, 2, 7, 8, 13, 15, 18, 20, 22, 23, 25, 27, 28, 30, 32, 35, 37, 42, 43, 48, 50, 53, 55, 57, 58, 60, 62, 63]
def q3 (t : ℕ) : ℕ := p3.getD t 0
def n3 : ℕ := 28
def yl3 : List ℤ := [1, 0, 5, 5, 4, 3, 6, 10, 0, 10, 11, 8, 10, 10, 8, 8, 1, 0, 0, 6, 2, 0, 12, 4, 0, 12, 9, 0]
def w3 (t : ℕ) : ℤ := yl3.getD t 0
def ul3 : List ℤ := [(-2), (-2), (-2), (-2), 0, (-2), 0, (-2), (-2), (-2), (-2), (-2), (-2), 2, 0, 2, 0, 0, 2, 2, 2, 0, 2, 0, 0, 0, 0, 0, 0, 6, 6, 2, 0, 0, 0, 6, 6, 2, 0, 0, 2, (-6), (-6), (-6), (-6), (-2), (-2), (-6), (-6), (-6), (-6), 0, 0, (-6), (-6), 0, 0, 0, (-6), (-6), (-6), (-1), (-6), (-1), (-6), 0, (-6), (-1), 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 1, 0, (-4), (-4), 0, (-4), (-4), (-4), (-4), 0, (-4), (-4), (-4), (-4), 0, (-4), (-4), 0, (-4), 0, (-10), (-4), 0, (-4), (-4), 4, (-5), 0, 0, 0, 0, 4, 0, (-7), 0, 0, (-6), (-6), (-6), (-6), (-6), (-6), 0, (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), 6, 0, 6, 6, 0, 0, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 34, 38, 40, 40, 21, 40, 40, 40, 32, 40, 32, 40, 40, 40, 33, 14, 40, (-40), (-40), (-40), (-40), (-40), (-40), (-40), (-40), (-40), (-40), (-40), (-53), (-40), 38, 26, 27, 40, 40, 40, 32, 36, 31, 40, 37, 40, 40, 40, 37, 33, 40, 40, 40, (-40), (-40), (-40), (-40), (-40), (-40), (-40), (-40), (-40), (-40), (-40), (-40), (-40), 31, 16, 41, 19, 35, 34, 13, 41, 19, 16, 41, 30, 41, 28, 29, 41, 13, 35, 41, 38, 39, 4, 41, (-41), (-41), (-41), (-41), (-41), (-41), (-41), (-41), (-41), (-41), (-41), (-41), (-41), 18, 37, 37, 37, 37, 37, 37, 19, 24, 31, 18, 37, 37, 13, 21, 17, 35, 37, 20, 37, 27, 37, 30, 22, 19, 37, 37, 34, 37, (-37), (-37), (-37), (-37), (-37), (-37), (-37), (-37), (-37), (-37), (-37), (-37), (-37), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-25), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 32, 18, 22, 32, 23, 32, 26, 28, 32, 10, 26, 18, 26, 18, 6, 15, 8, 23, 23, 7, 9, 14, 10, 12, 26, 23, 17, 29, 32, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def u3 (k : ℕ) : ℤ := ul3.getD k 0

def c3_0 (r t : ℕ) : Bool := gb11 r (q3 t)
def c3_1 (r t : ℕ) : Bool := gb13 r (q3 t)
def c3_2 (r t : ℕ) : Bool := gb17 r (q3 t)
def c3_3 (r t : ℕ) : Bool := gb19 r (q3 t)
def c3_4 (r t : ℕ) : Bool := gb23 r (q3 t)
def c3_5 (r t : ℕ) : Bool := gb29 r (q3 t)
def c3_6 (r t : ℕ) : Bool := gb31 r (q3 t)

def S3_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (w3 t + 6) * (if c3_0 r t then 1 else 0)
def S3_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (w3 t + 6) * (if c3_1 r t then 1 else 0)
def S3_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (w3 t + 6) * (if c3_2 r t then 1 else 0)
def S3_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (w3 t + 6) * (if c3_3 r t then 1 else 0)
def S3_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (w3 t + 6) * (if c3_4 r t then 1 else 0)
def S3_5 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (w3 t + 6) * (if c3_5 r t then 1 else 0)
def S3_6 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (w3 t + 6) * (if c3_6 r t then 1 else 0)

def L3_0 (r : ℕ) : ℤ := u3 (13 + r) + u3 (41 + r) + u3 (71 + r) + u3 (105 + r) + u3 (145 + r) + u3 (187 + r)
def L3_1 (r : ℕ) : ℤ := u3 (0 + r) + u3 (215 + r) + u3 (247 + r) + u3 (283 + r) + u3 (325 + r) + u3 (369 + r)
def L3_2 (r : ℕ) : ℤ := u3 (24 + r) + u3 (198 + r) + u3 (401 + r) + u3 (441 + r) + u3 (487 + r) + u3 (535 + r)
def L3_3 (r : ℕ) : ℤ := u3 (52 + r) + u3 (228 + r) + u3 (382 + r) + u3 (575 + r) + u3 (623 + r) + u3 (673 + r)
def L3_4 (r : ℕ) : ℤ := u3 (82 + r) + u3 (260 + r) + u3 (418 + r) + u3 (552 + r) + u3 (721 + r) + u3 (775 + r)
def L3_5 (r : ℕ) : ℤ := u3 (116 + r) + u3 (296 + r) + u3 (458 + r) + u3 (594 + r) + u3 (692 + r) + u3 (829 + r)
def L3_6 (r : ℕ) : ℤ := u3 (156 + r) + u3 (338 + r) + u3 (504 + r) + u3 (642 + r) + u3 (744 + r) + u3 (798 + r)

def aS3_0 (r : ℕ) : ℤ := S3_0 r - L3_0 r
def MS3_0 : ℤ := CaseSplit.mxr (aS3_0) 10
def aS3_1 (r : ℕ) : ℤ := S3_1 r - L3_1 r
def MS3_1 : ℤ := CaseSplit.mxr (aS3_1) 12
def aS3_2 (r : ℕ) : ℤ := S3_2 r - L3_2 r
def MS3_2 : ℤ := CaseSplit.mxr (aS3_2) 16
def aS3_3 (r : ℕ) : ℤ := S3_3 r - L3_3 r
def MS3_3 : ℤ := CaseSplit.mxr (aS3_3) 18
def aS3_4 (r : ℕ) : ℤ := S3_4 r - L3_4 r
def MS3_4 : ℤ := CaseSplit.mxr (aS3_4) 22
def aS3_5 (r : ℕ) : ℤ := S3_5 r - L3_5 r
def MS3_5 : ℤ := CaseSplit.mxr (aS3_5) 28
def aS3_6 (r : ℕ) : ℤ := S3_6 r - L3_6 r
def MS3_6 : ℤ := CaseSplit.mxr (aS3_6) 30

def N3_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (if c3_0 ra t && c3_1 rb t then 1 else 0)
def aP3_0 (ra rb : ℕ) : ℤ := -(6) * N3_0 ra rb + u3 (0 + rb) + u3 (13 + ra)
def MP3_0 : ℤ := CaseSplit.mxr2 (aP3_0) 10 12
def N3_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (if c3_0 ra t && c3_2 rb t then 1 else 0)
def aP3_1 (ra rb : ℕ) : ℤ := -(6) * N3_1 ra rb + u3 (24 + rb) + u3 (41 + ra)
def MP3_1 : ℤ := CaseSplit.mxr2 (aP3_1) 10 16
def N3_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (if c3_0 ra t && c3_3 rb t then 1 else 0)
def aP3_2 (ra rb : ℕ) : ℤ := -(6) * N3_2 ra rb + u3 (52 + rb) + u3 (71 + ra)
def MP3_2 : ℤ := CaseSplit.mxr2 (aP3_2) 10 18
def N3_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (if c3_0 ra t && c3_4 rb t then 1 else 0)
def aP3_3 (ra rb : ℕ) : ℤ := -(6) * N3_3 ra rb + u3 (82 + rb) + u3 (105 + ra)
def MP3_3 : ℤ := CaseSplit.mxr2 (aP3_3) 10 22
def N3_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (if c3_0 ra t && c3_5 rb t then 1 else 0)
def aP3_4 (ra rb : ℕ) : ℤ := -(6) * N3_4 ra rb + u3 (116 + rb) + u3 (145 + ra)
def MP3_4 : ℤ := CaseSplit.mxr2 (aP3_4) 10 28
def N3_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (if c3_0 ra t && c3_6 rb t then 1 else 0)
def aP3_5 (ra rb : ℕ) : ℤ := -(6) * N3_5 ra rb + u3 (156 + rb) + u3 (187 + ra)
def MP3_5 : ℤ := CaseSplit.mxr2 (aP3_5) 10 30
def P3_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (if c3_1 ra t && c3_2 rb t then 1 else 0)
def C3_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (if c3_1 ra t && c3_2 rb t && c3_0 s t then 1 else 0)
def M3_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C3_6 ra rb) 10
def E3_6 : List ℕ := [57, 63, 68, 79, 104, 115, 124, 130, 158, 169, 188, 194, 210, 216]
def N3_6 (ra rb : ℕ) : ℤ := if E3_6.contains (ra * 17 + rb) = true then P3_6 ra rb - M3_6 ra rb else 0
def aP3_6 (ra rb : ℕ) : ℤ := -(6) * N3_6 ra rb + u3 (198 + rb) + u3 (215 + ra)
def MP3_6 : ℤ := CaseSplit.mxr2 (aP3_6) 12 16
def P3_7 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (if c3_1 ra t && c3_3 rb t then 1 else 0)
def C3_7 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (if c3_1 ra t && c3_3 rb t && c3_0 s t then 1 else 0)
def M3_7 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C3_7 ra rb) 10
def E3_7 : List ℕ := [1, 4, 7, 10, 38, 41, 44, 80, 86, 91, 114, 120, 144, 167, 170, 172, 178, 212, 220, 246]
def N3_7 (ra rb : ℕ) : ℤ := if E3_7.contains (ra * 19 + rb) = true then P3_7 ra rb - M3_7 ra rb else 0
def aP3_7 (ra rb : ℕ) : ℤ := -(6) * N3_7 ra rb + u3 (228 + rb) + u3 (247 + ra)
def MP3_7 : ℤ := CaseSplit.mxr2 (aP3_7) 12 18
def P3_8 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (if c3_1 ra t && c3_4 rb t then 1 else 0)
def C3_8 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (if c3_1 ra t && c3_4 rb t && c3_0 s t then 1 else 0)
def M3_8 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C3_8 ra rb) 10
def E3_8 : List ℕ := [2]
def N3_8 (ra rb : ℕ) : ℤ := if E3_8.contains (ra * 23 + rb) = true then P3_8 ra rb - M3_8 ra rb else 0
def aP3_8 (ra rb : ℕ) : ℤ := -(6) * N3_8 ra rb + u3 (260 + rb) + u3 (283 + ra)
def MP3_8 : ℤ := CaseSplit.mxr2 (aP3_8) 12 22
def P3_9 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (if c3_1 ra t && c3_5 rb t then 1 else 0)
def C3_9 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (if c3_1 ra t && c3_5 rb t && c3_0 s t then 1 else 0)
def M3_9 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C3_9 ra rb) 10
def E3_9 : List ℕ := [30, 146, 180, 296]
def N3_9 (ra rb : ℕ) : ℤ := if E3_9.contains (ra * 29 + rb) = true then P3_9 ra rb - M3_9 ra rb else 0
def aP3_9 (ra rb : ℕ) : ℤ := -(6) * N3_9 ra rb + u3 (296 + rb) + u3 (325 + ra)
def MP3_9 : ℤ := CaseSplit.mxr2 (aP3_9) 12 28
def P3_10 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (if c3_1 ra t && c3_6 rb t then 1 else 0)
def C3_10 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (if c3_1 ra t && c3_6 rb t && c3_0 s t then 1 else 0)
def M3_10 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C3_10 ra rb) 10
def E3_10 : List ℕ := [121, 245]
def N3_10 (ra rb : ℕ) : ℤ := if E3_10.contains (ra * 31 + rb) = true then P3_10 ra rb - M3_10 ra rb else 0
def aP3_10 (ra rb : ℕ) : ℤ := -(6) * N3_10 ra rb + u3 (338 + rb) + u3 (369 + ra)
def MP3_10 : ℤ := CaseSplit.mxr2 (aP3_10) 12 30
def N3_11 (_ra _rb : ℕ) : ℤ := 0
def aP3_11 (ra rb : ℕ) : ℤ := -(6) * N3_11 ra rb + u3 (382 + rb) + u3 (401 + ra)
def MP3_11 : ℤ := CaseSplit.mxr2 (aP3_11) 16 18
def N3_12 (_ra _rb : ℕ) : ℤ := 0
def aP3_12 (ra rb : ℕ) : ℤ := -(6) * N3_12 ra rb + u3 (418 + rb) + u3 (441 + ra)
def MP3_12 : ℤ := CaseSplit.mxr2 (aP3_12) 16 22
def N3_13 (_ra _rb : ℕ) : ℤ := 0
def aP3_13 (ra rb : ℕ) : ℤ := -(6) * N3_13 ra rb + u3 (458 + rb) + u3 (487 + ra)
def MP3_13 : ℤ := CaseSplit.mxr2 (aP3_13) 16 28
def N3_14 (_ra _rb : ℕ) : ℤ := 0
def aP3_14 (ra rb : ℕ) : ℤ := -(6) * N3_14 ra rb + u3 (504 + rb) + u3 (535 + ra)
def MP3_14 : ℤ := CaseSplit.mxr2 (aP3_14) 16 30
def N3_15 (_ra _rb : ℕ) : ℤ := 0
def aP3_15 (ra rb : ℕ) : ℤ := -(6) * N3_15 ra rb + u3 (552 + rb) + u3 (575 + ra)
def MP3_15 : ℤ := CaseSplit.mxr2 (aP3_15) 18 22
def N3_16 (_ra _rb : ℕ) : ℤ := 0
def aP3_16 (ra rb : ℕ) : ℤ := -(6) * N3_16 ra rb + u3 (594 + rb) + u3 (623 + ra)
def MP3_16 : ℤ := CaseSplit.mxr2 (aP3_16) 18 28
def N3_17 (_ra _rb : ℕ) : ℤ := 0
def aP3_17 (ra rb : ℕ) : ℤ := -(6) * N3_17 ra rb + u3 (642 + rb) + u3 (673 + ra)
def MP3_17 : ℤ := CaseSplit.mxr2 (aP3_17) 18 30
def N3_18 (_ra _rb : ℕ) : ℤ := 0
def aP3_18 (ra rb : ℕ) : ℤ := -(6) * N3_18 ra rb + u3 (692 + rb) + u3 (721 + ra)
def MP3_18 : ℤ := CaseSplit.mxr2 (aP3_18) 22 28
def N3_19 (_ra _rb : ℕ) : ℤ := 0
def aP3_19 (ra rb : ℕ) : ℤ := -(6) * N3_19 ra rb + u3 (744 + rb) + u3 (775 + ra)
def MP3_19 : ℤ := CaseSplit.mxr2 (aP3_19) 22 30
def N3_20 (_ra _rb : ℕ) : ℤ := 0
def aP3_20 (ra rb : ℕ) : ℤ := -(6) * N3_20 ra rb + u3 (798 + rb) + u3 (829 + ra)
def MP3_20 : ℤ := CaseSplit.mxr2 (aP3_20) 28 30

def rhs3 : ℤ := (∑ t ∈ Finset.range n3, w3 t) + 6 * (n3 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn3 : ∀ t, t < n3 → (0 : ℤ) ≤ w3 t := by decide
theorem plt3 : ∀ t, t < n3 → q3 t < 65 := by decide
theorem pfree3_5 : ∀ t, t < n3 → gb5 0 (q3 t) = false := by decide
theorem pfree3_7 : ∀ t, t < n3 → gb7 3 (q3 t) = false := by decide
theorem MSv3_0 : MS3_0 = 62 := by decide +kernel
theorem MSv3_1 : MS3_1 = 212 := by decide +kernel
theorem MSv3_2 : MS3_2 = 1 := by decide +kernel
theorem MSv3_3 : MS3_3 = 1 := by decide +kernel
theorem MSv3_4 : MS3_4 = 1 := by decide +kernel
theorem MSv3_5 : MS3_5 = 1 := by decide +kernel
theorem MSv3_6 : MS3_6 = 1 := by decide +kernel
theorem MPv3_0 : MP3_0 = 0 := by decide +kernel
theorem MPv3_1 : MP3_1 = 0 := by decide +kernel
theorem MPv3_2 : MP3_2 = 0 := by decide +kernel
theorem MPv3_3 : MP3_3 = 0 := by decide +kernel
theorem MPv3_4 : MP3_4 = 0 := by decide +kernel
theorem MPv3_5 : MP3_5 = 0 := by decide +kernel
theorem MPv3_6 : MP3_6 = 0 := by decide +kernel
theorem MPv3_7 : MP3_7 = 0 := by decide +kernel
theorem MPv3_8 : MP3_8 = 0 := by decide +kernel
theorem MPv3_9 : MP3_9 = 0 := by decide +kernel
theorem MPv3_10 : MP3_10 = 0 := by decide +kernel
theorem MPv3_11 : MP3_11 = 0 := by decide +kernel
theorem MPv3_12 : MP3_12 = 0 := by decide +kernel
theorem MPv3_13 : MP3_13 = 0 := by decide +kernel
theorem MPv3_14 : MP3_14 = 0 := by decide +kernel
theorem MPv3_15 : MP3_15 = 0 := by decide +kernel
theorem MPv3_16 : MP3_16 = 0 := by decide +kernel
theorem MPv3_17 : MP3_17 = 0 := by decide +kernel
theorem MPv3_18 : MP3_18 = 0 := by decide +kernel
theorem MPv3_19 : MP3_19 = 0 := by decide +kernel
theorem MPv3_20 : MP3_20 = 32 := by decide +kernel
theorem rhsv3 : rhs3 = 313 := by decide +kernel

/-- **The case-3 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 2/12.
    (Scaled by the common denominator 12: 311 < 313.) -/
theorem cert3 : MS3_0 + MS3_1 + MS3_2 + MS3_3 + MS3_4 + MS3_5 + MS3_6 + MP3_0 + MP3_1 + MP3_2 + MP3_3 + MP3_4 + MP3_5 + MP3_6 + MP3_7 + MP3_8 + MP3_9 + MP3_10 + MP3_11 + MP3_12 + MP3_13 + MP3_14 + MP3_15 + MP3_16 + MP3_17 + MP3_18 + MP3_19 + MP3_20 < rhs3 := by
  rw [MSv3_0, MSv3_1, MSv3_2, MSv3_3, MSv3_4, MSv3_5, MSv3_6, MPv3_0, MPv3_1, MPv3_2, MPv3_3, MPv3_4, MPv3_5, MPv3_6, MPv3_7, MPv3_8, MPv3_9, MPv3_10, MPv3_11, MPv3_12, MPv3_13, MPv3_14, MPv3_15, MPv3_16, MPv3_17, MPv3_18, MPv3_19, MPv3_20, rhsv3]
  decide

def Dg3 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := (if c3_0 r0 t then 1 else 0) + (if c3_1 r1 t then 1 else 0) + (if c3_2 r2 t then 1 else 0) + (if c3_3 r3 t then 1 else 0) + (if c3_4 r4 t then 1 else 0) + (if c3_5 r5 t then 1 else 0) + (if c3_6 r6 t then 1 else 0)
def Wl3_0 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c3_0 r0 t && c3_1 r1 t then 1 else 0
def Wl3_1 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c3_0 r0 t && c3_2 r2 t then 1 else 0
def Wl3_2 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c3_0 r0 t && c3_3 r3 t then 1 else 0
def Wl3_3 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c3_0 r0 t && c3_4 r4 t then 1 else 0
def Wl3_4 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c3_0 r0 t && c3_5 r5 t then 1 else 0
def Wl3_5 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if c3_0 r0 t && c3_6 r6 t then 1 else 0
def Wl3_6 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c3_0 r0 t && c3_1 r1 t && c3_2 r2 t then 1 else 0
def Wl3_7 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c3_0 r0 t && c3_1 r1 t && c3_3 r3 t then 1 else 0
def Wl3_8 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c3_0 r0 t && c3_1 r1 t && c3_4 r4 t then 1 else 0
def Wl3_9 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c3_0 r0 t && c3_1 r1 t && c3_5 r5 t then 1 else 0
def Wl3_10 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c3_0 r0 t && c3_1 r1 t && c3_6 r6 t then 1 else 0
def Wl3_11 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c3_0 r0 t && !c3_1 r1 t && c3_2 r2 t && c3_3 r3 t then 1 else 0
def Wl3_12 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c3_0 r0 t && !c3_1 r1 t && c3_2 r2 t && c3_4 r4 t then 1 else 0
def Wl3_13 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c3_0 r0 t && !c3_1 r1 t && c3_2 r2 t && c3_5 r5 t then 1 else 0
def Wl3_14 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c3_0 r0 t && !c3_1 r1 t && c3_2 r2 t && c3_6 r6 t then 1 else 0
def Wl3_15 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c3_0 r0 t && !c3_1 r1 t && !c3_2 r2 t && c3_3 r3 t && c3_4 r4 t then 1 else 0
def Wl3_16 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c3_0 r0 t && !c3_1 r1 t && !c3_2 r2 t && c3_3 r3 t && c3_5 r5 t then 1 else 0
def Wl3_17 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c3_0 r0 t && !c3_1 r1 t && !c3_2 r2 t && c3_3 r3 t && c3_6 r6 t then 1 else 0
def Wl3_18 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c3_0 r0 t && !c3_1 r1 t && !c3_2 r2 t && !c3_3 r3 t && c3_4 r4 t && c3_5 r5 t then 1 else 0
def Wl3_19 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c3_0 r0 t && !c3_1 r1 t && !c3_2 r2 t && !c3_3 r3 t && c3_4 r4 t && c3_6 r6 t then 1 else 0
def Wl3_20 (r0 r1 r2 r3 r4 r5 r6 t : ℕ) : ℤ := if !c3_0 r0 t && !c3_1 r1 t && !c3_2 r2 t && !c3_3 r3 t && !c3_4 r4 t && c3_5 r5 t && c3_6 r6 t then 1 else 0

/-- **No configuration blocks the whole window in case 3.** -/
theorem nocov3 {r0 r1 r2 r3 r4 r5 r6 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23) (h5 : r5 < 29) (h6 : r6 < 31)
    (hcov : ∀ t, t < n3 → (c3_0 r0 t || c3_1 r1 t || c3_2 r2 t || c3_3 r3 t || c3_4 r4 t || c3_5 r5 t || c3_6 r6 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n3, (1 : ℤ) + (Wl3_0 r0 r1 r2 r3 r4 r5 r6 t + Wl3_1 r0 r1 r2 r3 r4 r5 r6 t + Wl3_2 r0 r1 r2 r3 r4 r5 r6 t + Wl3_3 r0 r1 r2 r3 r4 r5 r6 t + Wl3_4 r0 r1 r2 r3 r4 r5 r6 t + Wl3_5 r0 r1 r2 r3 r4 r5 r6 t + Wl3_6 r0 r1 r2 r3 r4 r5 r6 t + Wl3_7 r0 r1 r2 r3 r4 r5 r6 t + Wl3_8 r0 r1 r2 r3 r4 r5 r6 t + Wl3_9 r0 r1 r2 r3 r4 r5 r6 t + Wl3_10 r0 r1 r2 r3 r4 r5 r6 t + Wl3_11 r0 r1 r2 r3 r4 r5 r6 t + Wl3_12 r0 r1 r2 r3 r4 r5 r6 t + Wl3_13 r0 r1 r2 r3 r4 r5 r6 t + Wl3_14 r0 r1 r2 r3 r4 r5 r6 t + Wl3_15 r0 r1 r2 r3 r4 r5 r6 t + Wl3_16 r0 r1 r2 r3 r4 r5 r6 t + Wl3_17 r0 r1 r2 r3 r4 r5 r6 t + Wl3_18 r0 r1 r2 r3 r4 r5 r6 t + Wl3_19 r0 r1 r2 r3 r4 r5 r6 t + Wl3_20 r0 r1 r2 r3 r4 r5 r6 t) ≤ Dg3 r0 r1 r2 r3 r4 r5 r6 t := by
    intro t ht
    simp only [Wl3_0, Wl3_1, Wl3_2, Wl3_3, Wl3_4, Wl3_5, Wl3_6, Wl3_7, Wl3_8, Wl3_9, Wl3_10, Wl3_11, Wl3_12, Wl3_13, Wl3_14, Wl3_15, Wl3_16, Wl3_17, Wl3_18, Wl3_19, Wl3_20, Dg3]
    exact CaseSplit.lowest7 _ _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n3, (1 : ℤ) ≤ Dg3 r0 r1 r2 r3 r4 r5 r6 t := by
    intro t ht
    simp only [Dg3]
    exact CaseSplit.degpos7 _ _ _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n3 : ℤ) + ((∑ t ∈ Finset.range n3, Wl3_0 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n3, Wl3_1 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n3, Wl3_2 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n3, Wl3_3 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n3, Wl3_4 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n3, Wl3_5 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n3, Wl3_6 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n3, Wl3_7 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n3, Wl3_8 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n3, Wl3_9 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n3, Wl3_10 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n3, Wl3_11 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n3, Wl3_12 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n3, Wl3_13 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n3, Wl3_14 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n3, Wl3_15 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n3, Wl3_16 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n3, Wl3_17 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n3, Wl3_18 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n3, Wl3_19 r0 r1 r2 r3 r4 r5 r6 t) + (∑ t ∈ Finset.range n3, Wl3_20 r0 r1 r2 r3 r4 r5 r6 t)) ≤ ∑ t ∈ Finset.range n3, Dg3 r0 r1 r2 r3 r4 r5 r6 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N3_0 r0 r1 ≤ ∑ t ∈ Finset.range n3, Wl3_0 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N3_0, Wl3_0, le_refl]
  have hn1 : N3_1 r0 r2 ≤ ∑ t ∈ Finset.range n3, Wl3_1 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N3_1, Wl3_1, le_refl]
  have hn2 : N3_2 r0 r3 ≤ ∑ t ∈ Finset.range n3, Wl3_2 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N3_2, Wl3_2, le_refl]
  have hn3 : N3_3 r0 r4 ≤ ∑ t ∈ Finset.range n3, Wl3_3 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N3_3, Wl3_3, le_refl]
  have hn4 : N3_4 r0 r5 ≤ ∑ t ∈ Finset.range n3, Wl3_4 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N3_4, Wl3_4, le_refl]
  have hn5 : N3_5 r0 r6 ≤ ∑ t ∈ Finset.range n3, Wl3_5 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N3_5, Wl3_5, le_refl]
  have hn6 : N3_6 r1 r2 ≤ ∑ t ∈ Finset.range n3, Wl3_6 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n3, Wl3_6 r0 r1 r2 r3 r4 r5 r6 t
        = (if c3_1 r1 t && c3_2 r2 t then (1:ℤ) else 0)
          - (if c3_1 r1 t && c3_2 r2 t && c3_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl3_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n3, Wl3_6 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl3_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n3, Wl3_6 r0 r1 r2 r3 r4 r5 r6 t
        = P3_6 r1 r2 - C3_6 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P3_6, C3_6]
    have hm : C3_6 r1 r2 r0 ≤ M3_6 r1 r2 :=
      CaseSplit.le_mxr (C3_6 r1 r2) 10 r0 (by omega)
    simp only [N3_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N3_7 r1 r3 ≤ ∑ t ∈ Finset.range n3, Wl3_7 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n3, Wl3_7 r0 r1 r2 r3 r4 r5 r6 t
        = (if c3_1 r1 t && c3_3 r3 t then (1:ℤ) else 0)
          - (if c3_1 r1 t && c3_3 r3 t && c3_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl3_7]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n3, Wl3_7 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl3_7]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n3, Wl3_7 r0 r1 r2 r3 r4 r5 r6 t
        = P3_7 r1 r3 - C3_7 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P3_7, C3_7]
    have hm : C3_7 r1 r3 r0 ≤ M3_7 r1 r3 :=
      CaseSplit.le_mxr (C3_7 r1 r3) 10 r0 (by omega)
    simp only [N3_7]
    split
    · rw [hL]; omega
    · exact hnn
  have hn8 : N3_8 r1 r4 ≤ ∑ t ∈ Finset.range n3, Wl3_8 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n3, Wl3_8 r0 r1 r2 r3 r4 r5 r6 t
        = (if c3_1 r1 t && c3_4 r4 t then (1:ℤ) else 0)
          - (if c3_1 r1 t && c3_4 r4 t && c3_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl3_8]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n3, Wl3_8 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl3_8]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n3, Wl3_8 r0 r1 r2 r3 r4 r5 r6 t
        = P3_8 r1 r4 - C3_8 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P3_8, C3_8]
    have hm : C3_8 r1 r4 r0 ≤ M3_8 r1 r4 :=
      CaseSplit.le_mxr (C3_8 r1 r4) 10 r0 (by omega)
    simp only [N3_8]
    split
    · rw [hL]; omega
    · exact hnn
  have hn9 : N3_9 r1 r5 ≤ ∑ t ∈ Finset.range n3, Wl3_9 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n3, Wl3_9 r0 r1 r2 r3 r4 r5 r6 t
        = (if c3_1 r1 t && c3_5 r5 t then (1:ℤ) else 0)
          - (if c3_1 r1 t && c3_5 r5 t && c3_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl3_9]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n3, Wl3_9 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl3_9]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n3, Wl3_9 r0 r1 r2 r3 r4 r5 r6 t
        = P3_9 r1 r5 - C3_9 r1 r5 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P3_9, C3_9]
    have hm : C3_9 r1 r5 r0 ≤ M3_9 r1 r5 :=
      CaseSplit.le_mxr (C3_9 r1 r5) 10 r0 (by omega)
    simp only [N3_9]
    split
    · rw [hL]; omega
    · exact hnn
  have hn10 : N3_10 r1 r6 ≤ ∑ t ∈ Finset.range n3, Wl3_10 r0 r1 r2 r3 r4 r5 r6 t := by
    have hsp : ∀ t ∈ Finset.range n3, Wl3_10 r0 r1 r2 r3 r4 r5 r6 t
        = (if c3_1 r1 t && c3_6 r6 t then (1:ℤ) else 0)
          - (if c3_1 r1 t && c3_6 r6 t && c3_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl3_10]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n3, Wl3_10 r0 r1 r2 r3 r4 r5 r6 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl3_10]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n3, Wl3_10 r0 r1 r2 r3 r4 r5 r6 t
        = P3_10 r1 r6 - C3_10 r1 r6 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P3_10, C3_10]
    have hm : C3_10 r1 r6 r0 ≤ M3_10 r1 r6 :=
      CaseSplit.le_mxr (C3_10 r1 r6) 10 r0 (by omega)
    simp only [N3_10]
    split
    · rw [hL]; omega
    · exact hnn
  have hn11 : N3_11 r2 r3 ≤ ∑ t ∈ Finset.range n3, Wl3_11 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N3_11]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl3_11]
    exact CaseSplit.ind_nonneg _
  have hn12 : N3_12 r2 r4 ≤ ∑ t ∈ Finset.range n3, Wl3_12 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N3_12]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl3_12]
    exact CaseSplit.ind_nonneg _
  have hn13 : N3_13 r2 r5 ≤ ∑ t ∈ Finset.range n3, Wl3_13 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N3_13]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl3_13]
    exact CaseSplit.ind_nonneg _
  have hn14 : N3_14 r2 r6 ≤ ∑ t ∈ Finset.range n3, Wl3_14 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N3_14]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl3_14]
    exact CaseSplit.ind_nonneg _
  have hn15 : N3_15 r3 r4 ≤ ∑ t ∈ Finset.range n3, Wl3_15 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N3_15]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl3_15]
    exact CaseSplit.ind_nonneg _
  have hn16 : N3_16 r3 r5 ≤ ∑ t ∈ Finset.range n3, Wl3_16 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N3_16]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl3_16]
    exact CaseSplit.ind_nonneg _
  have hn17 : N3_17 r3 r6 ≤ ∑ t ∈ Finset.range n3, Wl3_17 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N3_17]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl3_17]
    exact CaseSplit.ind_nonneg _
  have hn18 : N3_18 r4 r5 ≤ ∑ t ∈ Finset.range n3, Wl3_18 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N3_18]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl3_18]
    exact CaseSplit.ind_nonneg _
  have hn19 : N3_19 r4 r6 ≤ ∑ t ∈ Finset.range n3, Wl3_19 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N3_19]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl3_19]
    exact CaseSplit.ind_nonneg _
  have hn20 : N3_20 r5 r6 ≤ ∑ t ∈ Finset.range n3, Wl3_20 r0 r1 r2 r3 r4 r5 r6 t := by
    simp only [N3_20]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl3_20]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n3, (w3 t + 6) * Dg3 r0 r1 r2 r3 r4 r5 r6 t = S3_0 r0 + S3_1 r1 + S3_2 r2 + S3_3 r3 + S3_4 r4 + S3_5 r5 + S3_6 r6 := by
    simp only [S3_0, S3_1, S3_2, S3_3, S3_4, S3_5, S3_6, Dg3, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n3, (w3 t + 6) * Dg3 r0 r1 r2 r3 r4 r5 r6 t
      = (∑ t ∈ Finset.range n3, w3 t * Dg3 r0 r1 r2 r3 r4 r5 r6 t)
        + 6 * (∑ t ∈ Finset.range n3, Dg3 r0 r1 r2 r3 r4 r5 r6 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n3, w3 t)
      ≤ ∑ t ∈ Finset.range n3, w3 t * Dg3 r0 r1 r2 r3 r4 r5 r6 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg3 r0 r1 r2 r3 r4 r5 r6 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w3 t := wnn3 t (Finset.mem_range.mp ht)
    calc w3 t = w3 t * 1 := (mul_one _).symm
      _ ≤ w3 t * Dg3 r0 r1 r2 r3 r4 r5 r6 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS3_0 r0 + aS3_1 r1 + aS3_2 r2 + aS3_3 r3 + aS3_4 r4 + aS3_5 r5 + aS3_6 r6) + (aP3_0 r0 r1 + aP3_1 r0 r2 + aP3_2 r0 r3 + aP3_3 r0 r4 + aP3_4 r0 r5 + aP3_5 r0 r6 + aP3_6 r1 r2 + aP3_7 r1 r3 + aP3_8 r1 r4 + aP3_9 r1 r5 + aP3_10 r1 r6 + aP3_11 r2 r3 + aP3_12 r2 r4 + aP3_13 r2 r5 + aP3_14 r2 r6 + aP3_15 r3 r4 + aP3_16 r3 r5 + aP3_17 r3 r6 + aP3_18 r4 r5 + aP3_19 r4 r6 + aP3_20 r5 r6) = (S3_0 r0 + S3_1 r1 + S3_2 r2 + S3_3 r3 + S3_4 r4 + S3_5 r5 + S3_6 r6) - 6 * (N3_0 r0 r1 + N3_1 r0 r2 + N3_2 r0 r3 + N3_3 r0 r4 + N3_4 r0 r5 + N3_5 r0 r6 + N3_6 r1 r2 + N3_7 r1 r3 + N3_8 r1 r4 + N3_9 r1 r5 + N3_10 r1 r6 + N3_11 r2 r3 + N3_12 r2 r4 + N3_13 r2 r5 + N3_14 r2 r6 + N3_15 r3 r4 + N3_16 r3 r5 + N3_17 r3 r6 + N3_18 r4 r5 + N3_19 r4 r6 + N3_20 r5 r6) := by
    simp only [aS3_0, aS3_1, aS3_2, aS3_3, aS3_4, aS3_5, aS3_6, aP3_0, aP3_1, aP3_2, aP3_3, aP3_4, aP3_5, aP3_6, aP3_7, aP3_8, aP3_9, aP3_10, aP3_11, aP3_12, aP3_13, aP3_14, aP3_15, aP3_16, aP3_17, aP3_18, aP3_19, aP3_20, L3_0, L3_1, L3_2, L3_3, L3_4, L3_5, L3_6]
    ring
  have bS0 : aS3_0 r0 ≤ MS3_0 := CaseSplit.le_mxr (aS3_0) 10 r0 (by omega)
  have bS1 : aS3_1 r1 ≤ MS3_1 := CaseSplit.le_mxr (aS3_1) 12 r1 (by omega)
  have bS2 : aS3_2 r2 ≤ MS3_2 := CaseSplit.le_mxr (aS3_2) 16 r2 (by omega)
  have bS3 : aS3_3 r3 ≤ MS3_3 := CaseSplit.le_mxr (aS3_3) 18 r3 (by omega)
  have bS4 : aS3_4 r4 ≤ MS3_4 := CaseSplit.le_mxr (aS3_4) 22 r4 (by omega)
  have bS5 : aS3_5 r5 ≤ MS3_5 := CaseSplit.le_mxr (aS3_5) 28 r5 (by omega)
  have bS6 : aS3_6 r6 ≤ MS3_6 := CaseSplit.le_mxr (aS3_6) 30 r6 (by omega)
  have bP0 : aP3_0 r0 r1 ≤ MP3_0 := CaseSplit.le_mxr2 (aP3_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP3_1 r0 r2 ≤ MP3_1 := CaseSplit.le_mxr2 (aP3_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP3_2 r0 r3 ≤ MP3_2 := CaseSplit.le_mxr2 (aP3_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP3_3 r0 r4 ≤ MP3_3 := CaseSplit.le_mxr2 (aP3_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP3_4 r0 r5 ≤ MP3_4 := CaseSplit.le_mxr2 (aP3_4) 10 28 r0 r5 (by omega) (by omega)
  have bP5 : aP3_5 r0 r6 ≤ MP3_5 := CaseSplit.le_mxr2 (aP3_5) 10 30 r0 r6 (by omega) (by omega)
  have bP6 : aP3_6 r1 r2 ≤ MP3_6 := CaseSplit.le_mxr2 (aP3_6) 12 16 r1 r2 (by omega) (by omega)
  have bP7 : aP3_7 r1 r3 ≤ MP3_7 := CaseSplit.le_mxr2 (aP3_7) 12 18 r1 r3 (by omega) (by omega)
  have bP8 : aP3_8 r1 r4 ≤ MP3_8 := CaseSplit.le_mxr2 (aP3_8) 12 22 r1 r4 (by omega) (by omega)
  have bP9 : aP3_9 r1 r5 ≤ MP3_9 := CaseSplit.le_mxr2 (aP3_9) 12 28 r1 r5 (by omega) (by omega)
  have bP10 : aP3_10 r1 r6 ≤ MP3_10 := CaseSplit.le_mxr2 (aP3_10) 12 30 r1 r6 (by omega) (by omega)
  have bP11 : aP3_11 r2 r3 ≤ MP3_11 := CaseSplit.le_mxr2 (aP3_11) 16 18 r2 r3 (by omega) (by omega)
  have bP12 : aP3_12 r2 r4 ≤ MP3_12 := CaseSplit.le_mxr2 (aP3_12) 16 22 r2 r4 (by omega) (by omega)
  have bP13 : aP3_13 r2 r5 ≤ MP3_13 := CaseSplit.le_mxr2 (aP3_13) 16 28 r2 r5 (by omega) (by omega)
  have bP14 : aP3_14 r2 r6 ≤ MP3_14 := CaseSplit.le_mxr2 (aP3_14) 16 30 r2 r6 (by omega) (by omega)
  have bP15 : aP3_15 r3 r4 ≤ MP3_15 := CaseSplit.le_mxr2 (aP3_15) 18 22 r3 r4 (by omega) (by omega)
  have bP16 : aP3_16 r3 r5 ≤ MP3_16 := CaseSplit.le_mxr2 (aP3_16) 18 28 r3 r5 (by omega) (by omega)
  have bP17 : aP3_17 r3 r6 ≤ MP3_17 := CaseSplit.le_mxr2 (aP3_17) 18 30 r3 r6 (by omega) (by omega)
  have bP18 : aP3_18 r4 r5 ≤ MP3_18 := CaseSplit.le_mxr2 (aP3_18) 22 28 r4 r5 (by omega) (by omega)
  have bP19 : aP3_19 r4 r6 ≤ MP3_19 := CaseSplit.le_mxr2 (aP3_19) 22 30 r4 r6 (by omega) (by omega)
  have bP20 : aP3_20 r5 r6 ≤ MP3_20 := CaseSplit.le_mxr2 (aP3_20) 28 30 r5 r6 (by omega) (by omega)
  have hrhs : rhs3 = (∑ t ∈ Finset.range n3, w3 t) + 6 * (n3 : ℤ) := rfl
  have hc := cert3
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, hn10, hn11, hn12, hn13, hn14, hn15, hn16, hn17, hn18, hn19, hn20, bS0, bS1, bS2, bS3, bS4, bS5, bS6, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9, bP10, bP11, bP12, bP13, bP14, bP15, bP16, bP17, bP18, bP19, bP20]

end IncCert31
