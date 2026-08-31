/-
INCREMENT-WIDTH CERTIFICATE, step 19->23, case 5 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_19_23.json, which re-derives every number
from the primes alone).

Machine 23, INCREMENT width 39 = F_2(19) + s_min(23) = 31 + 8,
held gears [5, 7] at phases [0, 5].  Free gears [11, 13, 17, 19, 23].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 1.
-/
import IncCert23B

namespace IncCert23

/-! ### case 5: held gears at phases [0, 5] -/

def p5 : List ℕ := [0, 2, 5, 7, 12, 13, 18, 20, 23, 25, 27, 28, 30, 32, 33, 35, 37]
def q5 (t : ℕ) : ℕ := p5.getD t 0
def n5 : ℕ := 17
def yl5 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
def w5 (t : ℕ) : ℤ := yl5.getD t 0
def ul5 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, (-1), (-1), (-1), 0, (-1), 0, (-1), (-1), 0, 0, (-1), (-1), (-1), 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, (-1), (-1), 0, 0, 0, (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), (-1), (-1), 0, 0, 0, (-1), 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, (-1), (-1), (-2), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), (-1), (-1), (-1), 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, (-2), (-2), (-2), (-2), (-2), (-2), (-3), (-2), (-2), (-2), (-2), (-2), (-2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 4, 1, 3, 2, 2, 4, 1, 4, 4, 1, 3, 2, 4, 3, 1, 4, 2, 3, 3, 1, 4, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0]
def u5 (k : ℕ) : ℤ := ul5.getD k 0

def c5_0 (r t : ℕ) : Bool := gb11 r (q5 t)
def c5_1 (r t : ℕ) : Bool := gb13 r (q5 t)
def c5_2 (r t : ℕ) : Bool := gb17 r (q5 t)
def c5_3 (r t : ℕ) : Bool := gb19 r (q5 t)
def c5_4 (r t : ℕ) : Bool := gb23 r (q5 t)

def S5_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (w5 t + 1) * (if c5_0 r t then 1 else 0)
def S5_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (w5 t + 1) * (if c5_1 r t then 1 else 0)
def S5_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (w5 t + 1) * (if c5_2 r t then 1 else 0)
def S5_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (w5 t + 1) * (if c5_3 r t then 1 else 0)
def S5_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (w5 t + 1) * (if c5_4 r t then 1 else 0)

def L5_0 (r : ℕ) : ℤ := u5 (13 + r) + u5 (41 + r) + u5 (71 + r) + u5 (105 + r)
def L5_1 (r : ℕ) : ℤ := u5 (0 + r) + u5 (133 + r) + u5 (165 + r) + u5 (201 + r)
def L5_2 (r : ℕ) : ℤ := u5 (24 + r) + u5 (116 + r) + u5 (233 + r) + u5 (273 + r)
def L5_3 (r : ℕ) : ℤ := u5 (52 + r) + u5 (146 + r) + u5 (214 + r) + u5 (313 + r)
def L5_4 (r : ℕ) : ℤ := u5 (82 + r) + u5 (178 + r) + u5 (250 + r) + u5 (290 + r)

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

def N5_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (if c5_0 ra t && c5_1 rb t then 1 else 0)
def aP5_0 (ra rb : ℕ) : ℤ := -(1) * N5_0 ra rb + u5 (0 + rb) + u5 (13 + ra)
def MP5_0 : ℤ := CaseSplit.mxr2 (aP5_0) 10 12
def N5_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (if c5_0 ra t && c5_2 rb t then 1 else 0)
def aP5_1 (ra rb : ℕ) : ℤ := -(1) * N5_1 ra rb + u5 (24 + rb) + u5 (41 + ra)
def MP5_1 : ℤ := CaseSplit.mxr2 (aP5_1) 10 16
def N5_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (if c5_0 ra t && c5_3 rb t then 1 else 0)
def aP5_2 (ra rb : ℕ) : ℤ := -(1) * N5_2 ra rb + u5 (52 + rb) + u5 (71 + ra)
def MP5_2 : ℤ := CaseSplit.mxr2 (aP5_2) 10 18
def N5_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (if c5_0 ra t && c5_4 rb t then 1 else 0)
def aP5_3 (ra rb : ℕ) : ℤ := -(1) * N5_3 ra rb + u5 (82 + rb) + u5 (105 + ra)
def MP5_3 : ℤ := CaseSplit.mxr2 (aP5_3) 10 22
def P5_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (if c5_1 ra t && c5_2 rb t then 1 else 0)
def C5_4 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (if c5_1 ra t && c5_2 rb t && c5_0 s t then 1 else 0)
def M5_4 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C5_4 ra rb) 10
def E5_4 : List ℕ := [68, 79, 104, 115, 188, 194]
def N5_4 (ra rb : ℕ) : ℤ := if E5_4.contains (ra * 17 + rb) = true then P5_4 ra rb - M5_4 ra rb else 0
def aP5_4 (ra rb : ℕ) : ℤ := -(1) * N5_4 ra rb + u5 (116 + rb) + u5 (133 + ra)
def MP5_4 : ℤ := CaseSplit.mxr2 (aP5_4) 12 16
def P5_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (if c5_1 ra t && c5_3 rb t then 1 else 0)
def C5_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (if c5_1 ra t && c5_3 rb t && c5_0 s t then 1 else 0)
def M5_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C5_5 ra rb) 10
def E5_5 : List ℕ := [41, 67, 78, 91, 131, 154, 167, 207, 212, 238]
def N5_5 (ra rb : ℕ) : ℤ := if E5_5.contains (ra * 19 + rb) = true then P5_5 ra rb - M5_5 ra rb else 0
def aP5_5 (ra rb : ℕ) : ℤ := -(1) * N5_5 ra rb + u5 (146 + rb) + u5 (165 + ra)
def MP5_5 : ℤ := CaseSplit.mxr2 (aP5_5) 12 18
def P5_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (if c5_1 ra t && c5_4 rb t then 1 else 0)
def C5_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n5, (if c5_1 ra t && c5_4 rb t && c5_0 s t then 1 else 0)
def M5_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C5_6 ra rb) 10
def E5_6 : List ℕ := []
def N5_6 (ra rb : ℕ) : ℤ := if E5_6.contains (ra * 23 + rb) = true then P5_6 ra rb - M5_6 ra rb else 0
def aP5_6 (ra rb : ℕ) : ℤ := -(1) * N5_6 ra rb + u5 (178 + rb) + u5 (201 + ra)
def MP5_6 : ℤ := CaseSplit.mxr2 (aP5_6) 12 22
def N5_7 (_ra _rb : ℕ) : ℤ := 0
def aP5_7 (ra rb : ℕ) : ℤ := -(1) * N5_7 ra rb + u5 (214 + rb) + u5 (233 + ra)
def MP5_7 : ℤ := CaseSplit.mxr2 (aP5_7) 16 18
def N5_8 (_ra _rb : ℕ) : ℤ := 0
def aP5_8 (ra rb : ℕ) : ℤ := -(1) * N5_8 ra rb + u5 (250 + rb) + u5 (273 + ra)
def MP5_8 : ℤ := CaseSplit.mxr2 (aP5_8) 16 22
def N5_9 (_ra _rb : ℕ) : ℤ := 0
def aP5_9 (ra rb : ℕ) : ℤ := -(1) * N5_9 ra rb + u5 (290 + rb) + u5 (313 + ra)
def MP5_9 : ℤ := CaseSplit.mxr2 (aP5_9) 18 22

def rhs5 : ℤ := (∑ t ∈ Finset.range n5, w5 t) + 1 * (n5 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn5 : ∀ t, t < n5 → (0 : ℤ) ≤ w5 t := by decide
theorem plt5 : ∀ t, t < n5 → q5 t < 39 := by decide
theorem pfree5_5 : ∀ t, t < n5 → gb5 0 (q5 t) = false := by decide
theorem pfree5_7 : ∀ t, t < n5 → gb7 5 (q5 t) = false := by decide
theorem MSv5_0 : MS5_0 = 3 := by decide +kernel
theorem MSv5_1 : MS5_1 = 9 := by decide +kernel
theorem MSv5_2 : MS5_2 = 0 := by decide +kernel
theorem MSv5_3 : MS5_3 = 0 := by decide +kernel
theorem MSv5_4 : MS5_4 = 0 := by decide +kernel
theorem MPv5_0 : MP5_0 = 0 := by decide +kernel
theorem MPv5_1 : MP5_1 = 0 := by decide +kernel
theorem MPv5_2 : MP5_2 = 0 := by decide +kernel
theorem MPv5_3 : MP5_3 = 0 := by decide +kernel
theorem MPv5_4 : MP5_4 = 0 := by decide +kernel
theorem MPv5_5 : MP5_5 = 0 := by decide +kernel
theorem MPv5_6 : MP5_6 = 0 := by decide +kernel
theorem MPv5_7 : MP5_7 = 0 := by decide +kernel
theorem MPv5_8 : MP5_8 = 0 := by decide +kernel
theorem MPv5_9 : MP5_9 = 5 := by decide +kernel
theorem rhsv5 : rhs5 = 18 := by decide +kernel

/-- **The case-5 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/1.
    (Scaled by the common denominator 1: 17 < 18.) -/
theorem cert5 : MS5_0 + MS5_1 + MS5_2 + MS5_3 + MS5_4 + MP5_0 + MP5_1 + MP5_2 + MP5_3 + MP5_4 + MP5_5 + MP5_6 + MP5_7 + MP5_8 + MP5_9 < rhs5 := by
  rw [MSv5_0, MSv5_1, MSv5_2, MSv5_3, MSv5_4, MPv5_0, MPv5_1, MPv5_2, MPv5_3, MPv5_4, MPv5_5, MPv5_6, MPv5_7, MPv5_8, MPv5_9, rhsv5]
  decide

def Dg5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := (if c5_0 r0 t then 1 else 0) + (if c5_1 r1 t then 1 else 0) + (if c5_2 r2 t then 1 else 0) + (if c5_3 r3 t then 1 else 0) + (if c5_4 r4 t then 1 else 0)
def Wl5_0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c5_0 r0 t && c5_1 r1 t then 1 else 0
def Wl5_1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c5_0 r0 t && c5_2 r2 t then 1 else 0
def Wl5_2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c5_0 r0 t && c5_3 r3 t then 1 else 0
def Wl5_3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c5_0 r0 t && c5_4 r4 t then 1 else 0
def Wl5_4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c5_0 r0 t && c5_1 r1 t && c5_2 r2 t then 1 else 0
def Wl5_5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c5_0 r0 t && c5_1 r1 t && c5_3 r3 t then 1 else 0
def Wl5_6 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c5_0 r0 t && c5_1 r1 t && c5_4 r4 t then 1 else 0
def Wl5_7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c5_0 r0 t && !c5_1 r1 t && c5_2 r2 t && c5_3 r3 t then 1 else 0
def Wl5_8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c5_0 r0 t && !c5_1 r1 t && c5_2 r2 t && c5_4 r4 t then 1 else 0
def Wl5_9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c5_0 r0 t && !c5_1 r1 t && !c5_2 r2 t && c5_3 r3 t && c5_4 r4 t then 1 else 0

/-- **No configuration blocks the whole window in case 5.** -/
theorem nocov5 {r0 r1 r2 r3 r4 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23)
    (hcov : ∀ t, t < n5 → (c5_0 r0 t || c5_1 r1 t || c5_2 r2 t || c5_3 r3 t || c5_4 r4 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n5, (1 : ℤ) + (Wl5_0 r0 r1 r2 r3 r4 t + Wl5_1 r0 r1 r2 r3 r4 t + Wl5_2 r0 r1 r2 r3 r4 t + Wl5_3 r0 r1 r2 r3 r4 t + Wl5_4 r0 r1 r2 r3 r4 t + Wl5_5 r0 r1 r2 r3 r4 t + Wl5_6 r0 r1 r2 r3 r4 t + Wl5_7 r0 r1 r2 r3 r4 t + Wl5_8 r0 r1 r2 r3 r4 t + Wl5_9 r0 r1 r2 r3 r4 t) ≤ Dg5 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Wl5_0, Wl5_1, Wl5_2, Wl5_3, Wl5_4, Wl5_5, Wl5_6, Wl5_7, Wl5_8, Wl5_9, Dg5]
    exact CaseSplit.lowest5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n5, (1 : ℤ) ≤ Dg5 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Dg5]
    exact CaseSplit.degpos5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n5 : ℤ) + ((∑ t ∈ Finset.range n5, Wl5_0 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n5, Wl5_1 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n5, Wl5_2 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n5, Wl5_3 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n5, Wl5_4 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n5, Wl5_5 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n5, Wl5_6 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n5, Wl5_7 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n5, Wl5_8 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n5, Wl5_9 r0 r1 r2 r3 r4 t)) ≤ ∑ t ∈ Finset.range n5, Dg5 r0 r1 r2 r3 r4 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N5_0 r0 r1 ≤ ∑ t ∈ Finset.range n5, Wl5_0 r0 r1 r2 r3 r4 t := by
    simp only [N5_0, Wl5_0, le_refl]
  have hn1 : N5_1 r0 r2 ≤ ∑ t ∈ Finset.range n5, Wl5_1 r0 r1 r2 r3 r4 t := by
    simp only [N5_1, Wl5_1, le_refl]
  have hn2 : N5_2 r0 r3 ≤ ∑ t ∈ Finset.range n5, Wl5_2 r0 r1 r2 r3 r4 t := by
    simp only [N5_2, Wl5_2, le_refl]
  have hn3 : N5_3 r0 r4 ≤ ∑ t ∈ Finset.range n5, Wl5_3 r0 r1 r2 r3 r4 t := by
    simp only [N5_3, Wl5_3, le_refl]
  have hn4 : N5_4 r1 r2 ≤ ∑ t ∈ Finset.range n5, Wl5_4 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n5, Wl5_4 r0 r1 r2 r3 r4 t
        = (if c5_1 r1 t && c5_2 r2 t then (1:ℤ) else 0)
          - (if c5_1 r1 t && c5_2 r2 t && c5_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl5_4]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n5, Wl5_4 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl5_4]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n5, Wl5_4 r0 r1 r2 r3 r4 t
        = P5_4 r1 r2 - C5_4 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P5_4, C5_4]
    have hm : C5_4 r1 r2 r0 ≤ M5_4 r1 r2 :=
      CaseSplit.le_mxr (C5_4 r1 r2) 10 r0 (by omega)
    simp only [N5_4]
    split
    · rw [hL]; omega
    · exact hnn
  have hn5 : N5_5 r1 r3 ≤ ∑ t ∈ Finset.range n5, Wl5_5 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n5, Wl5_5 r0 r1 r2 r3 r4 t
        = (if c5_1 r1 t && c5_3 r3 t then (1:ℤ) else 0)
          - (if c5_1 r1 t && c5_3 r3 t && c5_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl5_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n5, Wl5_5 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl5_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n5, Wl5_5 r0 r1 r2 r3 r4 t
        = P5_5 r1 r3 - C5_5 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P5_5, C5_5]
    have hm : C5_5 r1 r3 r0 ≤ M5_5 r1 r3 :=
      CaseSplit.le_mxr (C5_5 r1 r3) 10 r0 (by omega)
    simp only [N5_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N5_6 r1 r4 ≤ ∑ t ∈ Finset.range n5, Wl5_6 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n5, Wl5_6 r0 r1 r2 r3 r4 t
        = (if c5_1 r1 t && c5_4 r4 t then (1:ℤ) else 0)
          - (if c5_1 r1 t && c5_4 r4 t && c5_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl5_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n5, Wl5_6 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl5_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n5, Wl5_6 r0 r1 r2 r3 r4 t
        = P5_6 r1 r4 - C5_6 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P5_6, C5_6]
    have hm : C5_6 r1 r4 r0 ≤ M5_6 r1 r4 :=
      CaseSplit.le_mxr (C5_6 r1 r4) 10 r0 (by omega)
    simp only [N5_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N5_7 r2 r3 ≤ ∑ t ∈ Finset.range n5, Wl5_7 r0 r1 r2 r3 r4 t := by
    simp only [N5_7]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl5_7]
    exact CaseSplit.ind_nonneg _
  have hn8 : N5_8 r2 r4 ≤ ∑ t ∈ Finset.range n5, Wl5_8 r0 r1 r2 r3 r4 t := by
    simp only [N5_8]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl5_8]
    exact CaseSplit.ind_nonneg _
  have hn9 : N5_9 r3 r4 ≤ ∑ t ∈ Finset.range n5, Wl5_9 r0 r1 r2 r3 r4 t := by
    simp only [N5_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl5_9]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n5, (w5 t + 1) * Dg5 r0 r1 r2 r3 r4 t = S5_0 r0 + S5_1 r1 + S5_2 r2 + S5_3 r3 + S5_4 r4 := by
    simp only [S5_0, S5_1, S5_2, S5_3, S5_4, Dg5, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n5, (w5 t + 1) * Dg5 r0 r1 r2 r3 r4 t
      = (∑ t ∈ Finset.range n5, w5 t * Dg5 r0 r1 r2 r3 r4 t)
        + 1 * (∑ t ∈ Finset.range n5, Dg5 r0 r1 r2 r3 r4 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n5, w5 t)
      ≤ ∑ t ∈ Finset.range n5, w5 t * Dg5 r0 r1 r2 r3 r4 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg5 r0 r1 r2 r3 r4 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w5 t := wnn5 t (Finset.mem_range.mp ht)
    calc w5 t = w5 t * 1 := (mul_one _).symm
      _ ≤ w5 t * Dg5 r0 r1 r2 r3 r4 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS5_0 r0 + aS5_1 r1 + aS5_2 r2 + aS5_3 r3 + aS5_4 r4) + (aP5_0 r0 r1 + aP5_1 r0 r2 + aP5_2 r0 r3 + aP5_3 r0 r4 + aP5_4 r1 r2 + aP5_5 r1 r3 + aP5_6 r1 r4 + aP5_7 r2 r3 + aP5_8 r2 r4 + aP5_9 r3 r4) = (S5_0 r0 + S5_1 r1 + S5_2 r2 + S5_3 r3 + S5_4 r4) - 1 * (N5_0 r0 r1 + N5_1 r0 r2 + N5_2 r0 r3 + N5_3 r0 r4 + N5_4 r1 r2 + N5_5 r1 r3 + N5_6 r1 r4 + N5_7 r2 r3 + N5_8 r2 r4 + N5_9 r3 r4) := by
    simp only [aS5_0, aS5_1, aS5_2, aS5_3, aS5_4, aP5_0, aP5_1, aP5_2, aP5_3, aP5_4, aP5_5, aP5_6, aP5_7, aP5_8, aP5_9, L5_0, L5_1, L5_2, L5_3, L5_4]
    ring
  have bS0 : aS5_0 r0 ≤ MS5_0 := CaseSplit.le_mxr (aS5_0) 10 r0 (by omega)
  have bS1 : aS5_1 r1 ≤ MS5_1 := CaseSplit.le_mxr (aS5_1) 12 r1 (by omega)
  have bS2 : aS5_2 r2 ≤ MS5_2 := CaseSplit.le_mxr (aS5_2) 16 r2 (by omega)
  have bS3 : aS5_3 r3 ≤ MS5_3 := CaseSplit.le_mxr (aS5_3) 18 r3 (by omega)
  have bS4 : aS5_4 r4 ≤ MS5_4 := CaseSplit.le_mxr (aS5_4) 22 r4 (by omega)
  have bP0 : aP5_0 r0 r1 ≤ MP5_0 := CaseSplit.le_mxr2 (aP5_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP5_1 r0 r2 ≤ MP5_1 := CaseSplit.le_mxr2 (aP5_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP5_2 r0 r3 ≤ MP5_2 := CaseSplit.le_mxr2 (aP5_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP5_3 r0 r4 ≤ MP5_3 := CaseSplit.le_mxr2 (aP5_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP5_4 r1 r2 ≤ MP5_4 := CaseSplit.le_mxr2 (aP5_4) 12 16 r1 r2 (by omega) (by omega)
  have bP5 : aP5_5 r1 r3 ≤ MP5_5 := CaseSplit.le_mxr2 (aP5_5) 12 18 r1 r3 (by omega) (by omega)
  have bP6 : aP5_6 r1 r4 ≤ MP5_6 := CaseSplit.le_mxr2 (aP5_6) 12 22 r1 r4 (by omega) (by omega)
  have bP7 : aP5_7 r2 r3 ≤ MP5_7 := CaseSplit.le_mxr2 (aP5_7) 16 18 r2 r3 (by omega) (by omega)
  have bP8 : aP5_8 r2 r4 ≤ MP5_8 := CaseSplit.le_mxr2 (aP5_8) 16 22 r2 r4 (by omega) (by omega)
  have bP9 : aP5_9 r3 r4 ≤ MP5_9 := CaseSplit.le_mxr2 (aP5_9) 18 22 r3 r4 (by omega) (by omega)
  have hrhs : rhs5 = (∑ t ∈ Finset.range n5, w5 t) + 1 * (n5 : ℤ) := rfl
  have hc := cert5
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, bS0, bS1, bS2, bS3, bS4, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9]

end IncCert23
