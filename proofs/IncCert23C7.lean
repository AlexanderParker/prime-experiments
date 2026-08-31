/-
INCREMENT-WIDTH CERTIFICATE, step 19->23, case 7 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_19_23.json, which re-derives every number
from the primes alone).

Machine 23, INCREMENT width 39 = F_2(19) + s_min(23) = 31 + 8,
held gears [5, 7] at phases [1, 0].  Free gears [11, 13, 17, 19, 23].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 1.
-/
import IncCert23B

namespace IncCert23

/-! ### case 7: held gears at phases [1, 0] -/

def p7 : List ℕ := [2, 4, 7, 9, 11, 12, 14, 16, 17, 19, 21, 24, 26, 31, 32, 37]
def q7 (t : ℕ) : ℕ := p7.getD t 0
def n7 : ℕ := 16
def yl7 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def w7 (t : ℕ) : ℤ := yl7.getD t 0
def ul7 : List ℤ := [0, (-1), (-2), (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), (-1), 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, (-1), (-1), 0, 0, (-1), (-1), 0, (-1), (-1), 0, 3, 3, 2, 3, 2, 3, 3, 3, 2, 2, 3, 3, 3, 3, 2, 2, 3, (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), 2, 2, 2, 1, 2, 1, 0, 0, 0, 2, 2, 2, 0, 0, 1, 1, 2, 1, 2, (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0, 2, 2, 1, 2, 1, 1, 2, 0, (-1), 1, (-1), 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, (-1), 1, 0]
def u7 (k : ℕ) : ℤ := ul7.getD k 0

def c7_0 (r t : ℕ) : Bool := gb11 r (q7 t)
def c7_1 (r t : ℕ) : Bool := gb13 r (q7 t)
def c7_2 (r t : ℕ) : Bool := gb17 r (q7 t)
def c7_3 (r t : ℕ) : Bool := gb19 r (q7 t)
def c7_4 (r t : ℕ) : Bool := gb23 r (q7 t)

def S7_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (w7 t + 1) * (if c7_0 r t then 1 else 0)
def S7_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (w7 t + 1) * (if c7_1 r t then 1 else 0)
def S7_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (w7 t + 1) * (if c7_2 r t then 1 else 0)
def S7_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (w7 t + 1) * (if c7_3 r t then 1 else 0)
def S7_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (w7 t + 1) * (if c7_4 r t then 1 else 0)

def L7_0 (r : ℕ) : ℤ := u7 (13 + r) + u7 (41 + r) + u7 (71 + r) + u7 (105 + r)
def L7_1 (r : ℕ) : ℤ := u7 (0 + r) + u7 (133 + r) + u7 (165 + r) + u7 (201 + r)
def L7_2 (r : ℕ) : ℤ := u7 (24 + r) + u7 (116 + r) + u7 (233 + r) + u7 (273 + r)
def L7_3 (r : ℕ) : ℤ := u7 (52 + r) + u7 (146 + r) + u7 (214 + r) + u7 (313 + r)
def L7_4 (r : ℕ) : ℤ := u7 (82 + r) + u7 (178 + r) + u7 (250 + r) + u7 (290 + r)

def aS7_0 (r : ℕ) : ℤ := S7_0 r - L7_0 r
def MS7_0 : ℤ := CaseSplit.mxr (aS7_0) 10
def aS7_1 (r : ℕ) : ℤ := S7_1 r - L7_1 r
def MS7_1 : ℤ := CaseSplit.mxr (aS7_1) 12
def aS7_2 (r : ℕ) : ℤ := S7_2 r - L7_2 r
def MS7_2 : ℤ := CaseSplit.mxr (aS7_2) 16
def aS7_3 (r : ℕ) : ℤ := S7_3 r - L7_3 r
def MS7_3 : ℤ := CaseSplit.mxr (aS7_3) 18
def aS7_4 (r : ℕ) : ℤ := S7_4 r - L7_4 r
def MS7_4 : ℤ := CaseSplit.mxr (aS7_4) 22

def N7_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (if c7_0 ra t && c7_1 rb t then 1 else 0)
def aP7_0 (ra rb : ℕ) : ℤ := -(1) * N7_0 ra rb + u7 (0 + rb) + u7 (13 + ra)
def MP7_0 : ℤ := CaseSplit.mxr2 (aP7_0) 10 12
def N7_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (if c7_0 ra t && c7_2 rb t then 1 else 0)
def aP7_1 (ra rb : ℕ) : ℤ := -(1) * N7_1 ra rb + u7 (24 + rb) + u7 (41 + ra)
def MP7_1 : ℤ := CaseSplit.mxr2 (aP7_1) 10 16
def N7_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (if c7_0 ra t && c7_3 rb t then 1 else 0)
def aP7_2 (ra rb : ℕ) : ℤ := -(1) * N7_2 ra rb + u7 (52 + rb) + u7 (71 + ra)
def MP7_2 : ℤ := CaseSplit.mxr2 (aP7_2) 10 18
def N7_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (if c7_0 ra t && c7_4 rb t then 1 else 0)
def aP7_3 (ra rb : ℕ) : ℤ := -(1) * N7_3 ra rb + u7 (82 + rb) + u7 (105 + ra)
def MP7_3 : ℤ := CaseSplit.mxr2 (aP7_3) 10 22
def P7_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (if c7_1 ra t && c7_2 rb t then 1 else 0)
def C7_4 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (if c7_1 ra t && c7_2 rb t && c7_0 s t then 1 else 0)
def M7_4 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C7_4 ra rb) 10
def E7_4 : List ℕ := [39, 45, 75, 81, 129, 135, 154, 165, 170, 176]
def N7_4 (ra rb : ℕ) : ℤ := if E7_4.contains (ra * 17 + rb) = true then P7_4 ra rb - M7_4 ra rb else 0
def aP7_4 (ra rb : ℕ) : ℤ := -(1) * N7_4 ra rb + u7 (116 + rb) + u7 (133 + ra)
def MP7_4 : ℤ := CaseSplit.mxr2 (aP7_4) 12 16
def P7_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (if c7_1 ra t && c7_3 rb t then 1 else 0)
def C7_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (if c7_1 ra t && c7_3 rb t && c7_0 s t then 1 else 0)
def M7_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C7_5 ra rb) 10
def E7_5 : List ℕ := [11, 17, 87, 93, 98, 151, 174, 227]
def N7_5 (ra rb : ℕ) : ℤ := if E7_5.contains (ra * 19 + rb) = true then P7_5 ra rb - M7_5 ra rb else 0
def aP7_5 (ra rb : ℕ) : ℤ := -(1) * N7_5 ra rb + u7 (146 + rb) + u7 (165 + ra)
def MP7_5 : ℤ := CaseSplit.mxr2 (aP7_5) 12 18
def P7_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (if c7_1 ra t && c7_4 rb t then 1 else 0)
def C7_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n7, (if c7_1 ra t && c7_4 rb t && c7_0 s t then 1 else 0)
def M7_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C7_6 ra rb) 10
def E7_6 : List ℕ := []
def N7_6 (ra rb : ℕ) : ℤ := if E7_6.contains (ra * 23 + rb) = true then P7_6 ra rb - M7_6 ra rb else 0
def aP7_6 (ra rb : ℕ) : ℤ := -(1) * N7_6 ra rb + u7 (178 + rb) + u7 (201 + ra)
def MP7_6 : ℤ := CaseSplit.mxr2 (aP7_6) 12 22
def N7_7 (_ra _rb : ℕ) : ℤ := 0
def aP7_7 (ra rb : ℕ) : ℤ := -(1) * N7_7 ra rb + u7 (214 + rb) + u7 (233 + ra)
def MP7_7 : ℤ := CaseSplit.mxr2 (aP7_7) 16 18
def N7_8 (_ra _rb : ℕ) : ℤ := 0
def aP7_8 (ra rb : ℕ) : ℤ := -(1) * N7_8 ra rb + u7 (250 + rb) + u7 (273 + ra)
def MP7_8 : ℤ := CaseSplit.mxr2 (aP7_8) 16 22
def N7_9 (_ra _rb : ℕ) : ℤ := 0
def aP7_9 (ra rb : ℕ) : ℤ := -(1) * N7_9 ra rb + u7 (290 + rb) + u7 (313 + ra)
def MP7_9 : ℤ := CaseSplit.mxr2 (aP7_9) 18 22

def rhs7 : ℤ := (∑ t ∈ Finset.range n7, w7 t) + 1 * (n7 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn7 : ∀ t, t < n7 → (0 : ℤ) ≤ w7 t := by decide
theorem plt7 : ∀ t, t < n7 → q7 t < 39 := by decide
theorem pfree7_5 : ∀ t, t < n7 → gb5 1 (q7 t) = false := by decide
theorem pfree7_7 : ∀ t, t < n7 → gb7 0 (q7 t) = false := by decide
theorem MSv7_0 : MS7_0 = 3 := by decide +kernel
theorem MSv7_1 : MS7_1 = 9 := by decide +kernel
theorem MSv7_2 : MS7_2 = 0 := by decide +kernel
theorem MSv7_3 : MS7_3 = 0 := by decide +kernel
theorem MSv7_4 : MS7_4 = 0 := by decide +kernel
theorem MPv7_0 : MP7_0 = 0 := by decide +kernel
theorem MPv7_1 : MP7_1 = 0 := by decide +kernel
theorem MPv7_2 : MP7_2 = 0 := by decide +kernel
theorem MPv7_3 : MP7_3 = 0 := by decide +kernel
theorem MPv7_4 : MP7_4 = 0 := by decide +kernel
theorem MPv7_5 : MP7_5 = 0 := by decide +kernel
theorem MPv7_6 : MP7_6 = 0 := by decide +kernel
theorem MPv7_7 : MP7_7 = 0 := by decide +kernel
theorem MPv7_8 : MP7_8 = 0 := by decide +kernel
theorem MPv7_9 : MP7_9 = 3 := by decide +kernel
theorem rhsv7 : rhs7 = 16 := by decide +kernel

/-- **The case-7 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/1.
    (Scaled by the common denominator 1: 15 < 16.) -/
theorem cert7 : MS7_0 + MS7_1 + MS7_2 + MS7_3 + MS7_4 + MP7_0 + MP7_1 + MP7_2 + MP7_3 + MP7_4 + MP7_5 + MP7_6 + MP7_7 + MP7_8 + MP7_9 < rhs7 := by
  rw [MSv7_0, MSv7_1, MSv7_2, MSv7_3, MSv7_4, MPv7_0, MPv7_1, MPv7_2, MPv7_3, MPv7_4, MPv7_5, MPv7_6, MPv7_7, MPv7_8, MPv7_9, rhsv7]
  decide

def Dg7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := (if c7_0 r0 t then 1 else 0) + (if c7_1 r1 t then 1 else 0) + (if c7_2 r2 t then 1 else 0) + (if c7_3 r3 t then 1 else 0) + (if c7_4 r4 t then 1 else 0)
def Wl7_0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c7_0 r0 t && c7_1 r1 t then 1 else 0
def Wl7_1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c7_0 r0 t && c7_2 r2 t then 1 else 0
def Wl7_2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c7_0 r0 t && c7_3 r3 t then 1 else 0
def Wl7_3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c7_0 r0 t && c7_4 r4 t then 1 else 0
def Wl7_4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c7_0 r0 t && c7_1 r1 t && c7_2 r2 t then 1 else 0
def Wl7_5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c7_0 r0 t && c7_1 r1 t && c7_3 r3 t then 1 else 0
def Wl7_6 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c7_0 r0 t && c7_1 r1 t && c7_4 r4 t then 1 else 0
def Wl7_7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c7_0 r0 t && !c7_1 r1 t && c7_2 r2 t && c7_3 r3 t then 1 else 0
def Wl7_8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c7_0 r0 t && !c7_1 r1 t && c7_2 r2 t && c7_4 r4 t then 1 else 0
def Wl7_9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c7_0 r0 t && !c7_1 r1 t && !c7_2 r2 t && c7_3 r3 t && c7_4 r4 t then 1 else 0

/-- **No configuration blocks the whole window in case 7.** -/
theorem nocov7 {r0 r1 r2 r3 r4 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23)
    (hcov : ∀ t, t < n7 → (c7_0 r0 t || c7_1 r1 t || c7_2 r2 t || c7_3 r3 t || c7_4 r4 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n7, (1 : ℤ) + (Wl7_0 r0 r1 r2 r3 r4 t + Wl7_1 r0 r1 r2 r3 r4 t + Wl7_2 r0 r1 r2 r3 r4 t + Wl7_3 r0 r1 r2 r3 r4 t + Wl7_4 r0 r1 r2 r3 r4 t + Wl7_5 r0 r1 r2 r3 r4 t + Wl7_6 r0 r1 r2 r3 r4 t + Wl7_7 r0 r1 r2 r3 r4 t + Wl7_8 r0 r1 r2 r3 r4 t + Wl7_9 r0 r1 r2 r3 r4 t) ≤ Dg7 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Wl7_0, Wl7_1, Wl7_2, Wl7_3, Wl7_4, Wl7_5, Wl7_6, Wl7_7, Wl7_8, Wl7_9, Dg7]
    exact CaseSplit.lowest5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n7, (1 : ℤ) ≤ Dg7 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Dg7]
    exact CaseSplit.degpos5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n7 : ℤ) + ((∑ t ∈ Finset.range n7, Wl7_0 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n7, Wl7_1 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n7, Wl7_2 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n7, Wl7_3 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n7, Wl7_4 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n7, Wl7_5 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n7, Wl7_6 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n7, Wl7_7 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n7, Wl7_8 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n7, Wl7_9 r0 r1 r2 r3 r4 t)) ≤ ∑ t ∈ Finset.range n7, Dg7 r0 r1 r2 r3 r4 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N7_0 r0 r1 ≤ ∑ t ∈ Finset.range n7, Wl7_0 r0 r1 r2 r3 r4 t := by
    simp only [N7_0, Wl7_0, le_refl]
  have hn1 : N7_1 r0 r2 ≤ ∑ t ∈ Finset.range n7, Wl7_1 r0 r1 r2 r3 r4 t := by
    simp only [N7_1, Wl7_1, le_refl]
  have hn2 : N7_2 r0 r3 ≤ ∑ t ∈ Finset.range n7, Wl7_2 r0 r1 r2 r3 r4 t := by
    simp only [N7_2, Wl7_2, le_refl]
  have hn3 : N7_3 r0 r4 ≤ ∑ t ∈ Finset.range n7, Wl7_3 r0 r1 r2 r3 r4 t := by
    simp only [N7_3, Wl7_3, le_refl]
  have hn4 : N7_4 r1 r2 ≤ ∑ t ∈ Finset.range n7, Wl7_4 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n7, Wl7_4 r0 r1 r2 r3 r4 t
        = (if c7_1 r1 t && c7_2 r2 t then (1:ℤ) else 0)
          - (if c7_1 r1 t && c7_2 r2 t && c7_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl7_4]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n7, Wl7_4 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl7_4]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n7, Wl7_4 r0 r1 r2 r3 r4 t
        = P7_4 r1 r2 - C7_4 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P7_4, C7_4]
    have hm : C7_4 r1 r2 r0 ≤ M7_4 r1 r2 :=
      CaseSplit.le_mxr (C7_4 r1 r2) 10 r0 (by omega)
    simp only [N7_4]
    split
    · rw [hL]; omega
    · exact hnn
  have hn5 : N7_5 r1 r3 ≤ ∑ t ∈ Finset.range n7, Wl7_5 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n7, Wl7_5 r0 r1 r2 r3 r4 t
        = (if c7_1 r1 t && c7_3 r3 t then (1:ℤ) else 0)
          - (if c7_1 r1 t && c7_3 r3 t && c7_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl7_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n7, Wl7_5 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl7_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n7, Wl7_5 r0 r1 r2 r3 r4 t
        = P7_5 r1 r3 - C7_5 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P7_5, C7_5]
    have hm : C7_5 r1 r3 r0 ≤ M7_5 r1 r3 :=
      CaseSplit.le_mxr (C7_5 r1 r3) 10 r0 (by omega)
    simp only [N7_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N7_6 r1 r4 ≤ ∑ t ∈ Finset.range n7, Wl7_6 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n7, Wl7_6 r0 r1 r2 r3 r4 t
        = (if c7_1 r1 t && c7_4 r4 t then (1:ℤ) else 0)
          - (if c7_1 r1 t && c7_4 r4 t && c7_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl7_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n7, Wl7_6 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl7_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n7, Wl7_6 r0 r1 r2 r3 r4 t
        = P7_6 r1 r4 - C7_6 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P7_6, C7_6]
    have hm : C7_6 r1 r4 r0 ≤ M7_6 r1 r4 :=
      CaseSplit.le_mxr (C7_6 r1 r4) 10 r0 (by omega)
    simp only [N7_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N7_7 r2 r3 ≤ ∑ t ∈ Finset.range n7, Wl7_7 r0 r1 r2 r3 r4 t := by
    simp only [N7_7]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl7_7]
    exact CaseSplit.ind_nonneg _
  have hn8 : N7_8 r2 r4 ≤ ∑ t ∈ Finset.range n7, Wl7_8 r0 r1 r2 r3 r4 t := by
    simp only [N7_8]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl7_8]
    exact CaseSplit.ind_nonneg _
  have hn9 : N7_9 r3 r4 ≤ ∑ t ∈ Finset.range n7, Wl7_9 r0 r1 r2 r3 r4 t := by
    simp only [N7_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl7_9]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n7, (w7 t + 1) * Dg7 r0 r1 r2 r3 r4 t = S7_0 r0 + S7_1 r1 + S7_2 r2 + S7_3 r3 + S7_4 r4 := by
    simp only [S7_0, S7_1, S7_2, S7_3, S7_4, Dg7, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n7, (w7 t + 1) * Dg7 r0 r1 r2 r3 r4 t
      = (∑ t ∈ Finset.range n7, w7 t * Dg7 r0 r1 r2 r3 r4 t)
        + 1 * (∑ t ∈ Finset.range n7, Dg7 r0 r1 r2 r3 r4 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n7, w7 t)
      ≤ ∑ t ∈ Finset.range n7, w7 t * Dg7 r0 r1 r2 r3 r4 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg7 r0 r1 r2 r3 r4 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w7 t := wnn7 t (Finset.mem_range.mp ht)
    calc w7 t = w7 t * 1 := (mul_one _).symm
      _ ≤ w7 t * Dg7 r0 r1 r2 r3 r4 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS7_0 r0 + aS7_1 r1 + aS7_2 r2 + aS7_3 r3 + aS7_4 r4) + (aP7_0 r0 r1 + aP7_1 r0 r2 + aP7_2 r0 r3 + aP7_3 r0 r4 + aP7_4 r1 r2 + aP7_5 r1 r3 + aP7_6 r1 r4 + aP7_7 r2 r3 + aP7_8 r2 r4 + aP7_9 r3 r4) = (S7_0 r0 + S7_1 r1 + S7_2 r2 + S7_3 r3 + S7_4 r4) - 1 * (N7_0 r0 r1 + N7_1 r0 r2 + N7_2 r0 r3 + N7_3 r0 r4 + N7_4 r1 r2 + N7_5 r1 r3 + N7_6 r1 r4 + N7_7 r2 r3 + N7_8 r2 r4 + N7_9 r3 r4) := by
    simp only [aS7_0, aS7_1, aS7_2, aS7_3, aS7_4, aP7_0, aP7_1, aP7_2, aP7_3, aP7_4, aP7_5, aP7_6, aP7_7, aP7_8, aP7_9, L7_0, L7_1, L7_2, L7_3, L7_4]
    ring
  have bS0 : aS7_0 r0 ≤ MS7_0 := CaseSplit.le_mxr (aS7_0) 10 r0 (by omega)
  have bS1 : aS7_1 r1 ≤ MS7_1 := CaseSplit.le_mxr (aS7_1) 12 r1 (by omega)
  have bS2 : aS7_2 r2 ≤ MS7_2 := CaseSplit.le_mxr (aS7_2) 16 r2 (by omega)
  have bS3 : aS7_3 r3 ≤ MS7_3 := CaseSplit.le_mxr (aS7_3) 18 r3 (by omega)
  have bS4 : aS7_4 r4 ≤ MS7_4 := CaseSplit.le_mxr (aS7_4) 22 r4 (by omega)
  have bP0 : aP7_0 r0 r1 ≤ MP7_0 := CaseSplit.le_mxr2 (aP7_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP7_1 r0 r2 ≤ MP7_1 := CaseSplit.le_mxr2 (aP7_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP7_2 r0 r3 ≤ MP7_2 := CaseSplit.le_mxr2 (aP7_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP7_3 r0 r4 ≤ MP7_3 := CaseSplit.le_mxr2 (aP7_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP7_4 r1 r2 ≤ MP7_4 := CaseSplit.le_mxr2 (aP7_4) 12 16 r1 r2 (by omega) (by omega)
  have bP5 : aP7_5 r1 r3 ≤ MP7_5 := CaseSplit.le_mxr2 (aP7_5) 12 18 r1 r3 (by omega) (by omega)
  have bP6 : aP7_6 r1 r4 ≤ MP7_6 := CaseSplit.le_mxr2 (aP7_6) 12 22 r1 r4 (by omega) (by omega)
  have bP7 : aP7_7 r2 r3 ≤ MP7_7 := CaseSplit.le_mxr2 (aP7_7) 16 18 r2 r3 (by omega) (by omega)
  have bP8 : aP7_8 r2 r4 ≤ MP7_8 := CaseSplit.le_mxr2 (aP7_8) 16 22 r2 r4 (by omega) (by omega)
  have bP9 : aP7_9 r3 r4 ≤ MP7_9 := CaseSplit.le_mxr2 (aP7_9) 18 22 r3 r4 (by omega) (by omega)
  have hrhs : rhs7 = (∑ t ∈ Finset.range n7, w7 t) + 1 * (n7 : ℤ) := rfl
  have hc := cert7
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, bS0, bS1, bS2, bS3, bS4, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9]

end IncCert23
