/-
INCREMENT-WIDTH CERTIFICATE, step 19->23, case 14 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_19_23.json, which re-derives every number
from the primes alone).

Machine 23, INCREMENT width 39 = F_2(19) + s_min(23) = 31 + 8,
held gears [5, 7] at phases [2, 0].  Free gears [11, 13, 17, 19, 23].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 2.
-/
import IncCert23B

namespace IncCert23

/-! ### case 14: held gears at phases [2, 0] -/

def p14 : List ℕ := [0, 3, 5, 10, 11, 16, 18, 21, 23, 25, 26, 28, 30, 31, 33, 35, 38]
def q14 (t : ℕ) : ℕ := p14.getD t 0
def n14 : ℕ := 17
def yl14 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0]
def w14 (t : ℕ) : ℤ := yl14.getD t 0
def ul14 : List ℤ := [(-1), (-1), (-1), 0, (-1), (-1), 0, (-1), 0, (-1), 0, (-1), 0, 0, 0, 0, 0, 1, (-2), 1, 0, 1, 0, 0, (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), 0, 0, (-1), 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, (-1), (-1), 0, (-1), (-1), 0, (-1), (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), 0, 0, 0, (-3), 0, 0, 0, 0, 0, (-3), 0, 1, 0, 0, (-1), 1, (-1), (-1), 0, (-1), (-1), (-5), 0, 1, (-1), (-1), (-1), (-1), (-1), (-1), 1, 1, (-1), 0, (-5), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), 0, (-1), (-1), 1, 0, 3, 1, 3, 5, 5, 5, 5, 1, 2, 5, 5, 5, 1, 3, 3, 5, 5, (-5), (-5), (-5), (-5), (-5), (-5), (-5), (-5), (-5), (-5), (-5), (-5), (-5), 8, 5, 6, 8, 7, 7, 8, 8, 6, 6, 7, 8, 7, 8, 5, 6, 8, 6, 5, (-8), (-8), (-8), (-8), (-8), (-8), (-8), (-8), (-8), (-8), (-10), (-8), (-8), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 7, 3, 3, 7, 1, 3, 7, 2, 7, 1, 5, 5, 1, 7, 7, 7, 7, 1, 7, 7, 3, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def u14 (k : ℕ) : ℤ := ul14.getD k 0

def c14_0 (r t : ℕ) : Bool := gb11 r (q14 t)
def c14_1 (r t : ℕ) : Bool := gb13 r (q14 t)
def c14_2 (r t : ℕ) : Bool := gb17 r (q14 t)
def c14_3 (r t : ℕ) : Bool := gb19 r (q14 t)
def c14_4 (r t : ℕ) : Bool := gb23 r (q14 t)

def S14_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (w14 t + 2) * (if c14_0 r t then 1 else 0)
def S14_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (w14 t + 2) * (if c14_1 r t then 1 else 0)
def S14_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (w14 t + 2) * (if c14_2 r t then 1 else 0)
def S14_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (w14 t + 2) * (if c14_3 r t then 1 else 0)
def S14_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (w14 t + 2) * (if c14_4 r t then 1 else 0)

def L14_0 (r : ℕ) : ℤ := u14 (13 + r) + u14 (41 + r) + u14 (71 + r) + u14 (105 + r)
def L14_1 (r : ℕ) : ℤ := u14 (0 + r) + u14 (133 + r) + u14 (165 + r) + u14 (201 + r)
def L14_2 (r : ℕ) : ℤ := u14 (24 + r) + u14 (116 + r) + u14 (233 + r) + u14 (273 + r)
def L14_3 (r : ℕ) : ℤ := u14 (52 + r) + u14 (146 + r) + u14 (214 + r) + u14 (313 + r)
def L14_4 (r : ℕ) : ℤ := u14 (82 + r) + u14 (178 + r) + u14 (250 + r) + u14 (290 + r)

def aS14_0 (r : ℕ) : ℤ := S14_0 r - L14_0 r
def MS14_0 : ℤ := CaseSplit.mxr (aS14_0) 10
def aS14_1 (r : ℕ) : ℤ := S14_1 r - L14_1 r
def MS14_1 : ℤ := CaseSplit.mxr (aS14_1) 12
def aS14_2 (r : ℕ) : ℤ := S14_2 r - L14_2 r
def MS14_2 : ℤ := CaseSplit.mxr (aS14_2) 16
def aS14_3 (r : ℕ) : ℤ := S14_3 r - L14_3 r
def MS14_3 : ℤ := CaseSplit.mxr (aS14_3) 18
def aS14_4 (r : ℕ) : ℤ := S14_4 r - L14_4 r
def MS14_4 : ℤ := CaseSplit.mxr (aS14_4) 22

def N14_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (if c14_0 ra t && c14_1 rb t then 1 else 0)
def aP14_0 (ra rb : ℕ) : ℤ := -(2) * N14_0 ra rb + u14 (0 + rb) + u14 (13 + ra)
def MP14_0 : ℤ := CaseSplit.mxr2 (aP14_0) 10 12
def N14_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (if c14_0 ra t && c14_2 rb t then 1 else 0)
def aP14_1 (ra rb : ℕ) : ℤ := -(2) * N14_1 ra rb + u14 (24 + rb) + u14 (41 + ra)
def MP14_1 : ℤ := CaseSplit.mxr2 (aP14_1) 10 16
def N14_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (if c14_0 ra t && c14_3 rb t then 1 else 0)
def aP14_2 (ra rb : ℕ) : ℤ := -(2) * N14_2 ra rb + u14 (52 + rb) + u14 (71 + ra)
def MP14_2 : ℤ := CaseSplit.mxr2 (aP14_2) 10 18
def N14_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (if c14_0 ra t && c14_4 rb t then 1 else 0)
def aP14_3 (ra rb : ℕ) : ℤ := -(2) * N14_3 ra rb + u14 (82 + rb) + u14 (105 + ra)
def MP14_3 : ℤ := CaseSplit.mxr2 (aP14_3) 10 22
def P14_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (if c14_1 ra t && c14_2 rb t then 1 else 0)
def C14_4 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (if c14_1 ra t && c14_2 rb t && c14_0 s t then 1 else 0)
def M14_4 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C14_4 ra rb) 10
def E14_4 : List ℕ := [3, 9, 61, 67, 104, 115, 140, 151]
def N14_4 (ra rb : ℕ) : ℤ := if E14_4.contains (ra * 17 + rb) = true then P14_4 ra rb - M14_4 ra rb else 0
def aP14_4 (ra rb : ℕ) : ℤ := -(2) * N14_4 ra rb + u14 (116 + rb) + u14 (133 + ra)
def MP14_4 : ℤ := CaseSplit.mxr2 (aP14_4) 12 16
def P14_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (if c14_1 ra t && c14_3 rb t then 1 else 0)
def C14_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (if c14_1 ra t && c14_3 rb t && c14_0 s t then 1 else 0)
def M14_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C14_5 ra rb) 10
def E14_5 : List ℕ := [31, 73, 107, 118, 131, 152, 194, 207, 228, 244]
def N14_5 (ra rb : ℕ) : ℤ := if E14_5.contains (ra * 19 + rb) = true then P14_5 ra rb - M14_5 ra rb else 0
def aP14_5 (ra rb : ℕ) : ℤ := -(2) * N14_5 ra rb + u14 (146 + rb) + u14 (165 + ra)
def MP14_5 : ℤ := CaseSplit.mxr2 (aP14_5) 12 18
def P14_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (if c14_1 ra t && c14_4 rb t then 1 else 0)
def C14_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n14, (if c14_1 ra t && c14_4 rb t && c14_0 s t then 1 else 0)
def M14_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C14_6 ra rb) 10
def E14_6 : List ℕ := []
def N14_6 (ra rb : ℕ) : ℤ := if E14_6.contains (ra * 23 + rb) = true then P14_6 ra rb - M14_6 ra rb else 0
def aP14_6 (ra rb : ℕ) : ℤ := -(2) * N14_6 ra rb + u14 (178 + rb) + u14 (201 + ra)
def MP14_6 : ℤ := CaseSplit.mxr2 (aP14_6) 12 22
def N14_7 (_ra _rb : ℕ) : ℤ := 0
def aP14_7 (ra rb : ℕ) : ℤ := -(2) * N14_7 ra rb + u14 (214 + rb) + u14 (233 + ra)
def MP14_7 : ℤ := CaseSplit.mxr2 (aP14_7) 16 18
def N14_8 (_ra _rb : ℕ) : ℤ := 0
def aP14_8 (ra rb : ℕ) : ℤ := -(2) * N14_8 ra rb + u14 (250 + rb) + u14 (273 + ra)
def MP14_8 : ℤ := CaseSplit.mxr2 (aP14_8) 16 22
def N14_9 (_ra _rb : ℕ) : ℤ := 0
def aP14_9 (ra rb : ℕ) : ℤ := -(2) * N14_9 ra rb + u14 (290 + rb) + u14 (313 + ra)
def MP14_9 : ℤ := CaseSplit.mxr2 (aP14_9) 18 22

def rhs14 : ℤ := (∑ t ∈ Finset.range n14, w14 t) + 2 * (n14 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn14 : ∀ t, t < n14 → (0 : ℤ) ≤ w14 t := by decide
theorem plt14 : ∀ t, t < n14 → q14 t < 39 := by decide
theorem pfree14_5 : ∀ t, t < n14 → gb5 2 (q14 t) = false := by decide
theorem pfree14_7 : ∀ t, t < n14 → gb7 0 (q14 t) = false := by decide
theorem MSv14_0 : MS14_0 = 8 := by decide +kernel
theorem MSv14_1 : MS14_1 = 21 := by decide +kernel
theorem MSv14_2 : MS14_2 = 0 := by decide +kernel
theorem MSv14_3 : MS14_3 = 0 := by decide +kernel
theorem MSv14_4 : MS14_4 = 0 := by decide +kernel
theorem MPv14_0 : MP14_0 = 0 := by decide +kernel
theorem MPv14_1 : MP14_1 = 0 := by decide +kernel
theorem MPv14_2 : MP14_2 = 0 := by decide +kernel
theorem MPv14_3 : MP14_3 = 0 := by decide +kernel
theorem MPv14_4 : MP14_4 = 0 := by decide +kernel
theorem MPv14_5 : MP14_5 = 0 := by decide +kernel
theorem MPv14_6 : MP14_6 = 0 := by decide +kernel
theorem MPv14_7 : MP14_7 = 0 := by decide +kernel
theorem MPv14_8 : MP14_8 = 0 := by decide +kernel
theorem MPv14_9 : MP14_9 = 7 := by decide +kernel
theorem rhsv14 : rhs14 = 38 := by decide +kernel

/-- **The case-14 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 2/2.
    (Scaled by the common denominator 2: 36 < 38.) -/
theorem cert14 : MS14_0 + MS14_1 + MS14_2 + MS14_3 + MS14_4 + MP14_0 + MP14_1 + MP14_2 + MP14_3 + MP14_4 + MP14_5 + MP14_6 + MP14_7 + MP14_8 + MP14_9 < rhs14 := by
  rw [MSv14_0, MSv14_1, MSv14_2, MSv14_3, MSv14_4, MPv14_0, MPv14_1, MPv14_2, MPv14_3, MPv14_4, MPv14_5, MPv14_6, MPv14_7, MPv14_8, MPv14_9, rhsv14]
  decide

def Dg14 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := (if c14_0 r0 t then 1 else 0) + (if c14_1 r1 t then 1 else 0) + (if c14_2 r2 t then 1 else 0) + (if c14_3 r3 t then 1 else 0) + (if c14_4 r4 t then 1 else 0)
def Wl14_0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c14_0 r0 t && c14_1 r1 t then 1 else 0
def Wl14_1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c14_0 r0 t && c14_2 r2 t then 1 else 0
def Wl14_2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c14_0 r0 t && c14_3 r3 t then 1 else 0
def Wl14_3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c14_0 r0 t && c14_4 r4 t then 1 else 0
def Wl14_4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c14_0 r0 t && c14_1 r1 t && c14_2 r2 t then 1 else 0
def Wl14_5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c14_0 r0 t && c14_1 r1 t && c14_3 r3 t then 1 else 0
def Wl14_6 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c14_0 r0 t && c14_1 r1 t && c14_4 r4 t then 1 else 0
def Wl14_7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c14_0 r0 t && !c14_1 r1 t && c14_2 r2 t && c14_3 r3 t then 1 else 0
def Wl14_8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c14_0 r0 t && !c14_1 r1 t && c14_2 r2 t && c14_4 r4 t then 1 else 0
def Wl14_9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c14_0 r0 t && !c14_1 r1 t && !c14_2 r2 t && c14_3 r3 t && c14_4 r4 t then 1 else 0

/-- **No configuration blocks the whole window in case 14.** -/
theorem nocov14 {r0 r1 r2 r3 r4 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23)
    (hcov : ∀ t, t < n14 → (c14_0 r0 t || c14_1 r1 t || c14_2 r2 t || c14_3 r3 t || c14_4 r4 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n14, (1 : ℤ) + (Wl14_0 r0 r1 r2 r3 r4 t + Wl14_1 r0 r1 r2 r3 r4 t + Wl14_2 r0 r1 r2 r3 r4 t + Wl14_3 r0 r1 r2 r3 r4 t + Wl14_4 r0 r1 r2 r3 r4 t + Wl14_5 r0 r1 r2 r3 r4 t + Wl14_6 r0 r1 r2 r3 r4 t + Wl14_7 r0 r1 r2 r3 r4 t + Wl14_8 r0 r1 r2 r3 r4 t + Wl14_9 r0 r1 r2 r3 r4 t) ≤ Dg14 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Wl14_0, Wl14_1, Wl14_2, Wl14_3, Wl14_4, Wl14_5, Wl14_6, Wl14_7, Wl14_8, Wl14_9, Dg14]
    exact CaseSplit.lowest5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n14, (1 : ℤ) ≤ Dg14 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Dg14]
    exact CaseSplit.degpos5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n14 : ℤ) + ((∑ t ∈ Finset.range n14, Wl14_0 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n14, Wl14_1 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n14, Wl14_2 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n14, Wl14_3 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n14, Wl14_4 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n14, Wl14_5 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n14, Wl14_6 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n14, Wl14_7 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n14, Wl14_8 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n14, Wl14_9 r0 r1 r2 r3 r4 t)) ≤ ∑ t ∈ Finset.range n14, Dg14 r0 r1 r2 r3 r4 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N14_0 r0 r1 ≤ ∑ t ∈ Finset.range n14, Wl14_0 r0 r1 r2 r3 r4 t := by
    simp only [N14_0, Wl14_0, le_refl]
  have hn1 : N14_1 r0 r2 ≤ ∑ t ∈ Finset.range n14, Wl14_1 r0 r1 r2 r3 r4 t := by
    simp only [N14_1, Wl14_1, le_refl]
  have hn2 : N14_2 r0 r3 ≤ ∑ t ∈ Finset.range n14, Wl14_2 r0 r1 r2 r3 r4 t := by
    simp only [N14_2, Wl14_2, le_refl]
  have hn3 : N14_3 r0 r4 ≤ ∑ t ∈ Finset.range n14, Wl14_3 r0 r1 r2 r3 r4 t := by
    simp only [N14_3, Wl14_3, le_refl]
  have hn4 : N14_4 r1 r2 ≤ ∑ t ∈ Finset.range n14, Wl14_4 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n14, Wl14_4 r0 r1 r2 r3 r4 t
        = (if c14_1 r1 t && c14_2 r2 t then (1:ℤ) else 0)
          - (if c14_1 r1 t && c14_2 r2 t && c14_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl14_4]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n14, Wl14_4 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl14_4]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n14, Wl14_4 r0 r1 r2 r3 r4 t
        = P14_4 r1 r2 - C14_4 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P14_4, C14_4]
    have hm : C14_4 r1 r2 r0 ≤ M14_4 r1 r2 :=
      CaseSplit.le_mxr (C14_4 r1 r2) 10 r0 (by omega)
    simp only [N14_4]
    split
    · rw [hL]; omega
    · exact hnn
  have hn5 : N14_5 r1 r3 ≤ ∑ t ∈ Finset.range n14, Wl14_5 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n14, Wl14_5 r0 r1 r2 r3 r4 t
        = (if c14_1 r1 t && c14_3 r3 t then (1:ℤ) else 0)
          - (if c14_1 r1 t && c14_3 r3 t && c14_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl14_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n14, Wl14_5 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl14_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n14, Wl14_5 r0 r1 r2 r3 r4 t
        = P14_5 r1 r3 - C14_5 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P14_5, C14_5]
    have hm : C14_5 r1 r3 r0 ≤ M14_5 r1 r3 :=
      CaseSplit.le_mxr (C14_5 r1 r3) 10 r0 (by omega)
    simp only [N14_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N14_6 r1 r4 ≤ ∑ t ∈ Finset.range n14, Wl14_6 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n14, Wl14_6 r0 r1 r2 r3 r4 t
        = (if c14_1 r1 t && c14_4 r4 t then (1:ℤ) else 0)
          - (if c14_1 r1 t && c14_4 r4 t && c14_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl14_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n14, Wl14_6 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl14_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n14, Wl14_6 r0 r1 r2 r3 r4 t
        = P14_6 r1 r4 - C14_6 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P14_6, C14_6]
    have hm : C14_6 r1 r4 r0 ≤ M14_6 r1 r4 :=
      CaseSplit.le_mxr (C14_6 r1 r4) 10 r0 (by omega)
    simp only [N14_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N14_7 r2 r3 ≤ ∑ t ∈ Finset.range n14, Wl14_7 r0 r1 r2 r3 r4 t := by
    simp only [N14_7]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl14_7]
    exact CaseSplit.ind_nonneg _
  have hn8 : N14_8 r2 r4 ≤ ∑ t ∈ Finset.range n14, Wl14_8 r0 r1 r2 r3 r4 t := by
    simp only [N14_8]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl14_8]
    exact CaseSplit.ind_nonneg _
  have hn9 : N14_9 r3 r4 ≤ ∑ t ∈ Finset.range n14, Wl14_9 r0 r1 r2 r3 r4 t := by
    simp only [N14_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl14_9]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n14, (w14 t + 2) * Dg14 r0 r1 r2 r3 r4 t = S14_0 r0 + S14_1 r1 + S14_2 r2 + S14_3 r3 + S14_4 r4 := by
    simp only [S14_0, S14_1, S14_2, S14_3, S14_4, Dg14, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n14, (w14 t + 2) * Dg14 r0 r1 r2 r3 r4 t
      = (∑ t ∈ Finset.range n14, w14 t * Dg14 r0 r1 r2 r3 r4 t)
        + 2 * (∑ t ∈ Finset.range n14, Dg14 r0 r1 r2 r3 r4 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n14, w14 t)
      ≤ ∑ t ∈ Finset.range n14, w14 t * Dg14 r0 r1 r2 r3 r4 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg14 r0 r1 r2 r3 r4 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w14 t := wnn14 t (Finset.mem_range.mp ht)
    calc w14 t = w14 t * 1 := (mul_one _).symm
      _ ≤ w14 t * Dg14 r0 r1 r2 r3 r4 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS14_0 r0 + aS14_1 r1 + aS14_2 r2 + aS14_3 r3 + aS14_4 r4) + (aP14_0 r0 r1 + aP14_1 r0 r2 + aP14_2 r0 r3 + aP14_3 r0 r4 + aP14_4 r1 r2 + aP14_5 r1 r3 + aP14_6 r1 r4 + aP14_7 r2 r3 + aP14_8 r2 r4 + aP14_9 r3 r4) = (S14_0 r0 + S14_1 r1 + S14_2 r2 + S14_3 r3 + S14_4 r4) - 2 * (N14_0 r0 r1 + N14_1 r0 r2 + N14_2 r0 r3 + N14_3 r0 r4 + N14_4 r1 r2 + N14_5 r1 r3 + N14_6 r1 r4 + N14_7 r2 r3 + N14_8 r2 r4 + N14_9 r3 r4) := by
    simp only [aS14_0, aS14_1, aS14_2, aS14_3, aS14_4, aP14_0, aP14_1, aP14_2, aP14_3, aP14_4, aP14_5, aP14_6, aP14_7, aP14_8, aP14_9, L14_0, L14_1, L14_2, L14_3, L14_4]
    ring
  have bS0 : aS14_0 r0 ≤ MS14_0 := CaseSplit.le_mxr (aS14_0) 10 r0 (by omega)
  have bS1 : aS14_1 r1 ≤ MS14_1 := CaseSplit.le_mxr (aS14_1) 12 r1 (by omega)
  have bS2 : aS14_2 r2 ≤ MS14_2 := CaseSplit.le_mxr (aS14_2) 16 r2 (by omega)
  have bS3 : aS14_3 r3 ≤ MS14_3 := CaseSplit.le_mxr (aS14_3) 18 r3 (by omega)
  have bS4 : aS14_4 r4 ≤ MS14_4 := CaseSplit.le_mxr (aS14_4) 22 r4 (by omega)
  have bP0 : aP14_0 r0 r1 ≤ MP14_0 := CaseSplit.le_mxr2 (aP14_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP14_1 r0 r2 ≤ MP14_1 := CaseSplit.le_mxr2 (aP14_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP14_2 r0 r3 ≤ MP14_2 := CaseSplit.le_mxr2 (aP14_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP14_3 r0 r4 ≤ MP14_3 := CaseSplit.le_mxr2 (aP14_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP14_4 r1 r2 ≤ MP14_4 := CaseSplit.le_mxr2 (aP14_4) 12 16 r1 r2 (by omega) (by omega)
  have bP5 : aP14_5 r1 r3 ≤ MP14_5 := CaseSplit.le_mxr2 (aP14_5) 12 18 r1 r3 (by omega) (by omega)
  have bP6 : aP14_6 r1 r4 ≤ MP14_6 := CaseSplit.le_mxr2 (aP14_6) 12 22 r1 r4 (by omega) (by omega)
  have bP7 : aP14_7 r2 r3 ≤ MP14_7 := CaseSplit.le_mxr2 (aP14_7) 16 18 r2 r3 (by omega) (by omega)
  have bP8 : aP14_8 r2 r4 ≤ MP14_8 := CaseSplit.le_mxr2 (aP14_8) 16 22 r2 r4 (by omega) (by omega)
  have bP9 : aP14_9 r3 r4 ≤ MP14_9 := CaseSplit.le_mxr2 (aP14_9) 18 22 r3 r4 (by omega) (by omega)
  have hrhs : rhs14 = (∑ t ∈ Finset.range n14, w14 t) + 2 * (n14 : ℤ) := rfl
  have hc := cert14
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, bS0, bS1, bS2, bS3, bS4, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9]

end IncCert23
