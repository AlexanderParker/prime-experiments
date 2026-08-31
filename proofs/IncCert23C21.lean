/-
INCREMENT-WIDTH CERTIFICATE, step 19->23, case 21 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_19_23.json, which re-derives every number
from the primes alone).

Machine 23, INCREMENT width 39 = F_2(19) + s_min(23) = 31 + 8,
held gears [5, 7] at phases [3, 0].  Free gears [11, 13, 17, 19, 23].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 1.
-/
import IncCert23B

namespace IncCert23

/-! ### case 21: held gears at phases [3, 0] -/

def p21 : List ℕ := [0, 2, 4, 5, 7, 9, 10, 12, 14, 17, 19, 24, 25, 30, 32, 35, 37]
def q21 (t : ℕ) : ℕ := p21.getD t 0
def n21 : ℕ := 17
def yl21 : List ℤ := [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def w21 (t : ℕ) : ℤ := yl21.getD t 0
def ul21 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, (-1), (-2), 0, (-1), 0, (-1), (-1), (-1), 0, 0, 0, (-1), (-1), (-1), 1, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, (-1), (-1), (-1), 0, 0, 0, (-1), (-1), (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, (-1), (-1), 0, (-1), (-1), (-1), 0, (-1), (-1), 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 2, 3, (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 1, 1, 3, 0, 2, 1, 3, 3, 0, 2, 3, 1, 3, 0, 3, 3, 1, 2, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0]
def u21 (k : ℕ) : ℤ := ul21.getD k 0

def c21_0 (r t : ℕ) : Bool := gb11 r (q21 t)
def c21_1 (r t : ℕ) : Bool := gb13 r (q21 t)
def c21_2 (r t : ℕ) : Bool := gb17 r (q21 t)
def c21_3 (r t : ℕ) : Bool := gb19 r (q21 t)
def c21_4 (r t : ℕ) : Bool := gb23 r (q21 t)

def S21_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (w21 t + 1) * (if c21_0 r t then 1 else 0)
def S21_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (w21 t + 1) * (if c21_1 r t then 1 else 0)
def S21_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (w21 t + 1) * (if c21_2 r t then 1 else 0)
def S21_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (w21 t + 1) * (if c21_3 r t then 1 else 0)
def S21_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (w21 t + 1) * (if c21_4 r t then 1 else 0)

def L21_0 (r : ℕ) : ℤ := u21 (13 + r) + u21 (41 + r) + u21 (71 + r) + u21 (105 + r)
def L21_1 (r : ℕ) : ℤ := u21 (0 + r) + u21 (133 + r) + u21 (165 + r) + u21 (201 + r)
def L21_2 (r : ℕ) : ℤ := u21 (24 + r) + u21 (116 + r) + u21 (233 + r) + u21 (273 + r)
def L21_3 (r : ℕ) : ℤ := u21 (52 + r) + u21 (146 + r) + u21 (214 + r) + u21 (313 + r)
def L21_4 (r : ℕ) : ℤ := u21 (82 + r) + u21 (178 + r) + u21 (250 + r) + u21 (290 + r)

def aS21_0 (r : ℕ) : ℤ := S21_0 r - L21_0 r
def MS21_0 : ℤ := CaseSplit.mxr (aS21_0) 10
def aS21_1 (r : ℕ) : ℤ := S21_1 r - L21_1 r
def MS21_1 : ℤ := CaseSplit.mxr (aS21_1) 12
def aS21_2 (r : ℕ) : ℤ := S21_2 r - L21_2 r
def MS21_2 : ℤ := CaseSplit.mxr (aS21_2) 16
def aS21_3 (r : ℕ) : ℤ := S21_3 r - L21_3 r
def MS21_3 : ℤ := CaseSplit.mxr (aS21_3) 18
def aS21_4 (r : ℕ) : ℤ := S21_4 r - L21_4 r
def MS21_4 : ℤ := CaseSplit.mxr (aS21_4) 22

def N21_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (if c21_0 ra t && c21_1 rb t then 1 else 0)
def aP21_0 (ra rb : ℕ) : ℤ := -(1) * N21_0 ra rb + u21 (0 + rb) + u21 (13 + ra)
def MP21_0 : ℤ := CaseSplit.mxr2 (aP21_0) 10 12
def N21_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (if c21_0 ra t && c21_2 rb t then 1 else 0)
def aP21_1 (ra rb : ℕ) : ℤ := -(1) * N21_1 ra rb + u21 (24 + rb) + u21 (41 + ra)
def MP21_1 : ℤ := CaseSplit.mxr2 (aP21_1) 10 16
def N21_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (if c21_0 ra t && c21_3 rb t then 1 else 0)
def aP21_2 (ra rb : ℕ) : ℤ := -(1) * N21_2 ra rb + u21 (52 + rb) + u21 (71 + ra)
def MP21_2 : ℤ := CaseSplit.mxr2 (aP21_2) 10 18
def N21_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (if c21_0 ra t && c21_4 rb t then 1 else 0)
def aP21_3 (ra rb : ℕ) : ℤ := -(1) * N21_3 ra rb + u21 (82 + rb) + u21 (105 + ra)
def MP21_3 : ℤ := CaseSplit.mxr2 (aP21_3) 10 22
def P21_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (if c21_1 ra t && c21_2 rb t then 1 else 0)
def C21_4 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (if c21_1 ra t && c21_2 rb t && c21_0 s t then 1 else 0)
def M21_4 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C21_4 ra rb) 10
def E21_4 : List ℕ := [75, 81, 154, 165, 190, 201]
def N21_4 (ra rb : ℕ) : ℤ := if E21_4.contains (ra * 17 + rb) = true then P21_4 ra rb - M21_4 ra rb else 0
def aP21_4 (ra rb : ℕ) : ℤ := -(1) * N21_4 ra rb + u21 (116 + rb) + u21 (133 + ra)
def MP21_4 : ℤ := CaseSplit.mxr2 (aP21_4) 12 16
def P21_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (if c21_1 ra t && c21_3 rb t then 1 else 0)
def C21_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (if c21_1 ra t && c21_3 rb t && c21_0 s t then 1 else 0)
def M21_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C21_5 ra rb) 10
def E21_5 : List ℕ := [17, 67, 93, 98, 138, 151, 174, 214, 227, 238]
def N21_5 (ra rb : ℕ) : ℤ := if E21_5.contains (ra * 19 + rb) = true then P21_5 ra rb - M21_5 ra rb else 0
def aP21_5 (ra rb : ℕ) : ℤ := -(1) * N21_5 ra rb + u21 (146 + rb) + u21 (165 + ra)
def MP21_5 : ℤ := CaseSplit.mxr2 (aP21_5) 12 18
def P21_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (if c21_1 ra t && c21_4 rb t then 1 else 0)
def C21_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n21, (if c21_1 ra t && c21_4 rb t && c21_0 s t then 1 else 0)
def M21_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C21_6 ra rb) 10
def E21_6 : List ℕ := []
def N21_6 (ra rb : ℕ) : ℤ := if E21_6.contains (ra * 23 + rb) = true then P21_6 ra rb - M21_6 ra rb else 0
def aP21_6 (ra rb : ℕ) : ℤ := -(1) * N21_6 ra rb + u21 (178 + rb) + u21 (201 + ra)
def MP21_6 : ℤ := CaseSplit.mxr2 (aP21_6) 12 22
def N21_7 (_ra _rb : ℕ) : ℤ := 0
def aP21_7 (ra rb : ℕ) : ℤ := -(1) * N21_7 ra rb + u21 (214 + rb) + u21 (233 + ra)
def MP21_7 : ℤ := CaseSplit.mxr2 (aP21_7) 16 18
def N21_8 (_ra _rb : ℕ) : ℤ := 0
def aP21_8 (ra rb : ℕ) : ℤ := -(1) * N21_8 ra rb + u21 (250 + rb) + u21 (273 + ra)
def MP21_8 : ℤ := CaseSplit.mxr2 (aP21_8) 16 22
def N21_9 (_ra _rb : ℕ) : ℤ := 0
def aP21_9 (ra rb : ℕ) : ℤ := -(1) * N21_9 ra rb + u21 (290 + rb) + u21 (313 + ra)
def MP21_9 : ℤ := CaseSplit.mxr2 (aP21_9) 18 22

def rhs21 : ℤ := (∑ t ∈ Finset.range n21, w21 t) + 1 * (n21 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn21 : ∀ t, t < n21 → (0 : ℤ) ≤ w21 t := by decide
theorem plt21 : ∀ t, t < n21 → q21 t < 39 := by decide
theorem pfree21_5 : ∀ t, t < n21 → gb5 3 (q21 t) = false := by decide
theorem pfree21_7 : ∀ t, t < n21 → gb7 0 (q21 t) = false := by decide
theorem MSv21_0 : MS21_0 = 4 := by decide +kernel
theorem MSv21_1 : MS21_1 = 9 := by decide +kernel
theorem MSv21_2 : MS21_2 = 0 := by decide +kernel
theorem MSv21_3 : MS21_3 = 0 := by decide +kernel
theorem MSv21_4 : MS21_4 = 0 := by decide +kernel
theorem MPv21_0 : MP21_0 = 0 := by decide +kernel
theorem MPv21_1 : MP21_1 = 0 := by decide +kernel
theorem MPv21_2 : MP21_2 = 0 := by decide +kernel
theorem MPv21_3 : MP21_3 = 0 := by decide +kernel
theorem MPv21_4 : MP21_4 = 0 := by decide +kernel
theorem MPv21_5 : MP21_5 = 0 := by decide +kernel
theorem MPv21_6 : MP21_6 = 0 := by decide +kernel
theorem MPv21_7 : MP21_7 = 0 := by decide +kernel
theorem MPv21_8 : MP21_8 = 0 := by decide +kernel
theorem MPv21_9 : MP21_9 = 4 := by decide +kernel
theorem rhsv21 : rhs21 = 18 := by decide +kernel

/-- **The case-21 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/1.
    (Scaled by the common denominator 1: 17 < 18.) -/
theorem cert21 : MS21_0 + MS21_1 + MS21_2 + MS21_3 + MS21_4 + MP21_0 + MP21_1 + MP21_2 + MP21_3 + MP21_4 + MP21_5 + MP21_6 + MP21_7 + MP21_8 + MP21_9 < rhs21 := by
  rw [MSv21_0, MSv21_1, MSv21_2, MSv21_3, MSv21_4, MPv21_0, MPv21_1, MPv21_2, MPv21_3, MPv21_4, MPv21_5, MPv21_6, MPv21_7, MPv21_8, MPv21_9, rhsv21]
  decide

def Dg21 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := (if c21_0 r0 t then 1 else 0) + (if c21_1 r1 t then 1 else 0) + (if c21_2 r2 t then 1 else 0) + (if c21_3 r3 t then 1 else 0) + (if c21_4 r4 t then 1 else 0)
def Wl21_0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c21_0 r0 t && c21_1 r1 t then 1 else 0
def Wl21_1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c21_0 r0 t && c21_2 r2 t then 1 else 0
def Wl21_2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c21_0 r0 t && c21_3 r3 t then 1 else 0
def Wl21_3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c21_0 r0 t && c21_4 r4 t then 1 else 0
def Wl21_4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c21_0 r0 t && c21_1 r1 t && c21_2 r2 t then 1 else 0
def Wl21_5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c21_0 r0 t && c21_1 r1 t && c21_3 r3 t then 1 else 0
def Wl21_6 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c21_0 r0 t && c21_1 r1 t && c21_4 r4 t then 1 else 0
def Wl21_7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c21_0 r0 t && !c21_1 r1 t && c21_2 r2 t && c21_3 r3 t then 1 else 0
def Wl21_8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c21_0 r0 t && !c21_1 r1 t && c21_2 r2 t && c21_4 r4 t then 1 else 0
def Wl21_9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c21_0 r0 t && !c21_1 r1 t && !c21_2 r2 t && c21_3 r3 t && c21_4 r4 t then 1 else 0

/-- **No configuration blocks the whole window in case 21.** -/
theorem nocov21 {r0 r1 r2 r3 r4 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23)
    (hcov : ∀ t, t < n21 → (c21_0 r0 t || c21_1 r1 t || c21_2 r2 t || c21_3 r3 t || c21_4 r4 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n21, (1 : ℤ) + (Wl21_0 r0 r1 r2 r3 r4 t + Wl21_1 r0 r1 r2 r3 r4 t + Wl21_2 r0 r1 r2 r3 r4 t + Wl21_3 r0 r1 r2 r3 r4 t + Wl21_4 r0 r1 r2 r3 r4 t + Wl21_5 r0 r1 r2 r3 r4 t + Wl21_6 r0 r1 r2 r3 r4 t + Wl21_7 r0 r1 r2 r3 r4 t + Wl21_8 r0 r1 r2 r3 r4 t + Wl21_9 r0 r1 r2 r3 r4 t) ≤ Dg21 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Wl21_0, Wl21_1, Wl21_2, Wl21_3, Wl21_4, Wl21_5, Wl21_6, Wl21_7, Wl21_8, Wl21_9, Dg21]
    exact CaseSplit.lowest5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n21, (1 : ℤ) ≤ Dg21 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Dg21]
    exact CaseSplit.degpos5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n21 : ℤ) + ((∑ t ∈ Finset.range n21, Wl21_0 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n21, Wl21_1 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n21, Wl21_2 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n21, Wl21_3 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n21, Wl21_4 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n21, Wl21_5 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n21, Wl21_6 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n21, Wl21_7 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n21, Wl21_8 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n21, Wl21_9 r0 r1 r2 r3 r4 t)) ≤ ∑ t ∈ Finset.range n21, Dg21 r0 r1 r2 r3 r4 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N21_0 r0 r1 ≤ ∑ t ∈ Finset.range n21, Wl21_0 r0 r1 r2 r3 r4 t := by
    simp only [N21_0, Wl21_0, le_refl]
  have hn1 : N21_1 r0 r2 ≤ ∑ t ∈ Finset.range n21, Wl21_1 r0 r1 r2 r3 r4 t := by
    simp only [N21_1, Wl21_1, le_refl]
  have hn2 : N21_2 r0 r3 ≤ ∑ t ∈ Finset.range n21, Wl21_2 r0 r1 r2 r3 r4 t := by
    simp only [N21_2, Wl21_2, le_refl]
  have hn3 : N21_3 r0 r4 ≤ ∑ t ∈ Finset.range n21, Wl21_3 r0 r1 r2 r3 r4 t := by
    simp only [N21_3, Wl21_3, le_refl]
  have hn4 : N21_4 r1 r2 ≤ ∑ t ∈ Finset.range n21, Wl21_4 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n21, Wl21_4 r0 r1 r2 r3 r4 t
        = (if c21_1 r1 t && c21_2 r2 t then (1:ℤ) else 0)
          - (if c21_1 r1 t && c21_2 r2 t && c21_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl21_4]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n21, Wl21_4 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl21_4]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n21, Wl21_4 r0 r1 r2 r3 r4 t
        = P21_4 r1 r2 - C21_4 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P21_4, C21_4]
    have hm : C21_4 r1 r2 r0 ≤ M21_4 r1 r2 :=
      CaseSplit.le_mxr (C21_4 r1 r2) 10 r0 (by omega)
    simp only [N21_4]
    split
    · rw [hL]; omega
    · exact hnn
  have hn5 : N21_5 r1 r3 ≤ ∑ t ∈ Finset.range n21, Wl21_5 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n21, Wl21_5 r0 r1 r2 r3 r4 t
        = (if c21_1 r1 t && c21_3 r3 t then (1:ℤ) else 0)
          - (if c21_1 r1 t && c21_3 r3 t && c21_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl21_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n21, Wl21_5 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl21_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n21, Wl21_5 r0 r1 r2 r3 r4 t
        = P21_5 r1 r3 - C21_5 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P21_5, C21_5]
    have hm : C21_5 r1 r3 r0 ≤ M21_5 r1 r3 :=
      CaseSplit.le_mxr (C21_5 r1 r3) 10 r0 (by omega)
    simp only [N21_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N21_6 r1 r4 ≤ ∑ t ∈ Finset.range n21, Wl21_6 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n21, Wl21_6 r0 r1 r2 r3 r4 t
        = (if c21_1 r1 t && c21_4 r4 t then (1:ℤ) else 0)
          - (if c21_1 r1 t && c21_4 r4 t && c21_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl21_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n21, Wl21_6 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl21_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n21, Wl21_6 r0 r1 r2 r3 r4 t
        = P21_6 r1 r4 - C21_6 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P21_6, C21_6]
    have hm : C21_6 r1 r4 r0 ≤ M21_6 r1 r4 :=
      CaseSplit.le_mxr (C21_6 r1 r4) 10 r0 (by omega)
    simp only [N21_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N21_7 r2 r3 ≤ ∑ t ∈ Finset.range n21, Wl21_7 r0 r1 r2 r3 r4 t := by
    simp only [N21_7]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl21_7]
    exact CaseSplit.ind_nonneg _
  have hn8 : N21_8 r2 r4 ≤ ∑ t ∈ Finset.range n21, Wl21_8 r0 r1 r2 r3 r4 t := by
    simp only [N21_8]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl21_8]
    exact CaseSplit.ind_nonneg _
  have hn9 : N21_9 r3 r4 ≤ ∑ t ∈ Finset.range n21, Wl21_9 r0 r1 r2 r3 r4 t := by
    simp only [N21_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl21_9]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n21, (w21 t + 1) * Dg21 r0 r1 r2 r3 r4 t = S21_0 r0 + S21_1 r1 + S21_2 r2 + S21_3 r3 + S21_4 r4 := by
    simp only [S21_0, S21_1, S21_2, S21_3, S21_4, Dg21, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n21, (w21 t + 1) * Dg21 r0 r1 r2 r3 r4 t
      = (∑ t ∈ Finset.range n21, w21 t * Dg21 r0 r1 r2 r3 r4 t)
        + 1 * (∑ t ∈ Finset.range n21, Dg21 r0 r1 r2 r3 r4 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n21, w21 t)
      ≤ ∑ t ∈ Finset.range n21, w21 t * Dg21 r0 r1 r2 r3 r4 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg21 r0 r1 r2 r3 r4 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w21 t := wnn21 t (Finset.mem_range.mp ht)
    calc w21 t = w21 t * 1 := (mul_one _).symm
      _ ≤ w21 t * Dg21 r0 r1 r2 r3 r4 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS21_0 r0 + aS21_1 r1 + aS21_2 r2 + aS21_3 r3 + aS21_4 r4) + (aP21_0 r0 r1 + aP21_1 r0 r2 + aP21_2 r0 r3 + aP21_3 r0 r4 + aP21_4 r1 r2 + aP21_5 r1 r3 + aP21_6 r1 r4 + aP21_7 r2 r3 + aP21_8 r2 r4 + aP21_9 r3 r4) = (S21_0 r0 + S21_1 r1 + S21_2 r2 + S21_3 r3 + S21_4 r4) - 1 * (N21_0 r0 r1 + N21_1 r0 r2 + N21_2 r0 r3 + N21_3 r0 r4 + N21_4 r1 r2 + N21_5 r1 r3 + N21_6 r1 r4 + N21_7 r2 r3 + N21_8 r2 r4 + N21_9 r3 r4) := by
    simp only [aS21_0, aS21_1, aS21_2, aS21_3, aS21_4, aP21_0, aP21_1, aP21_2, aP21_3, aP21_4, aP21_5, aP21_6, aP21_7, aP21_8, aP21_9, L21_0, L21_1, L21_2, L21_3, L21_4]
    ring
  have bS0 : aS21_0 r0 ≤ MS21_0 := CaseSplit.le_mxr (aS21_0) 10 r0 (by omega)
  have bS1 : aS21_1 r1 ≤ MS21_1 := CaseSplit.le_mxr (aS21_1) 12 r1 (by omega)
  have bS2 : aS21_2 r2 ≤ MS21_2 := CaseSplit.le_mxr (aS21_2) 16 r2 (by omega)
  have bS3 : aS21_3 r3 ≤ MS21_3 := CaseSplit.le_mxr (aS21_3) 18 r3 (by omega)
  have bS4 : aS21_4 r4 ≤ MS21_4 := CaseSplit.le_mxr (aS21_4) 22 r4 (by omega)
  have bP0 : aP21_0 r0 r1 ≤ MP21_0 := CaseSplit.le_mxr2 (aP21_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP21_1 r0 r2 ≤ MP21_1 := CaseSplit.le_mxr2 (aP21_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP21_2 r0 r3 ≤ MP21_2 := CaseSplit.le_mxr2 (aP21_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP21_3 r0 r4 ≤ MP21_3 := CaseSplit.le_mxr2 (aP21_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP21_4 r1 r2 ≤ MP21_4 := CaseSplit.le_mxr2 (aP21_4) 12 16 r1 r2 (by omega) (by omega)
  have bP5 : aP21_5 r1 r3 ≤ MP21_5 := CaseSplit.le_mxr2 (aP21_5) 12 18 r1 r3 (by omega) (by omega)
  have bP6 : aP21_6 r1 r4 ≤ MP21_6 := CaseSplit.le_mxr2 (aP21_6) 12 22 r1 r4 (by omega) (by omega)
  have bP7 : aP21_7 r2 r3 ≤ MP21_7 := CaseSplit.le_mxr2 (aP21_7) 16 18 r2 r3 (by omega) (by omega)
  have bP8 : aP21_8 r2 r4 ≤ MP21_8 := CaseSplit.le_mxr2 (aP21_8) 16 22 r2 r4 (by omega) (by omega)
  have bP9 : aP21_9 r3 r4 ≤ MP21_9 := CaseSplit.le_mxr2 (aP21_9) 18 22 r3 r4 (by omega) (by omega)
  have hrhs : rhs21 = (∑ t ∈ Finset.range n21, w21 t) + 1 * (n21 : ℤ) := rfl
  have hc := cert21
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, bS0, bS1, bS2, bS3, bS4, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9]

end IncCert23
