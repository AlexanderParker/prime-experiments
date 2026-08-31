/-
INCREMENT-WIDTH CERTIFICATE, step 19->23, case 16 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_19_23.json, which re-derives every number
from the primes alone).

Machine 23, INCREMENT width 39 = F_2(19) + s_min(23) = 31 + 8,
held gears [5, 7] at phases [2, 2].  Free gears [11, 13, 17, 19, 23].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 1.
-/
import IncCert23B

namespace IncCert23

/-! ### case 16: held gears at phases [2, 2] -/

def p16 : List ℕ := [0, 1, 3, 5, 8, 10, 15, 16, 21, 23, 26, 28, 30, 31, 33, 35, 36, 38]
def q16 (t : ℕ) : ℕ := p16.getD t 0
def n16 : ℕ := 18
def yl16 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def w16 (t : ℕ) : ℤ := yl16.getD t 0
def ul16 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, (-1), (-1), (-2), (-1), (-1), (-1), (-1), (-2), 0, (-1), 0, (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-2), (-1), 0, (-1), (-1), (-1), 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 2, 3, 3, 2, 3, 1, 1, 3, 3, 2, 2, 3, 2, 3, 3, (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 1, 3, 4, 1, 4, 2, 1, 3, 1, 4, 3, 1, 4, 2, 3, 4, 2, 4, 2, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
def u16 (k : ℕ) : ℤ := ul16.getD k 0

def c16_0 (r t : ℕ) : Bool := gb11 r (q16 t)
def c16_1 (r t : ℕ) : Bool := gb13 r (q16 t)
def c16_2 (r t : ℕ) : Bool := gb17 r (q16 t)
def c16_3 (r t : ℕ) : Bool := gb19 r (q16 t)
def c16_4 (r t : ℕ) : Bool := gb23 r (q16 t)

def S16_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (w16 t + 1) * (if c16_0 r t then 1 else 0)
def S16_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (w16 t + 1) * (if c16_1 r t then 1 else 0)
def S16_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (w16 t + 1) * (if c16_2 r t then 1 else 0)
def S16_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (w16 t + 1) * (if c16_3 r t then 1 else 0)
def S16_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (w16 t + 1) * (if c16_4 r t then 1 else 0)

def L16_0 (r : ℕ) : ℤ := u16 (13 + r) + u16 (41 + r) + u16 (71 + r) + u16 (105 + r)
def L16_1 (r : ℕ) : ℤ := u16 (0 + r) + u16 (133 + r) + u16 (165 + r) + u16 (201 + r)
def L16_2 (r : ℕ) : ℤ := u16 (24 + r) + u16 (116 + r) + u16 (233 + r) + u16 (273 + r)
def L16_3 (r : ℕ) : ℤ := u16 (52 + r) + u16 (146 + r) + u16 (214 + r) + u16 (313 + r)
def L16_4 (r : ℕ) : ℤ := u16 (82 + r) + u16 (178 + r) + u16 (250 + r) + u16 (290 + r)

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

def N16_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (if c16_0 ra t && c16_1 rb t then 1 else 0)
def aP16_0 (ra rb : ℕ) : ℤ := -(1) * N16_0 ra rb + u16 (0 + rb) + u16 (13 + ra)
def MP16_0 : ℤ := CaseSplit.mxr2 (aP16_0) 10 12
def N16_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (if c16_0 ra t && c16_2 rb t then 1 else 0)
def aP16_1 (ra rb : ℕ) : ℤ := -(1) * N16_1 ra rb + u16 (24 + rb) + u16 (41 + ra)
def MP16_1 : ℤ := CaseSplit.mxr2 (aP16_1) 10 16
def N16_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (if c16_0 ra t && c16_3 rb t then 1 else 0)
def aP16_2 (ra rb : ℕ) : ℤ := -(1) * N16_2 ra rb + u16 (52 + rb) + u16 (71 + ra)
def MP16_2 : ℤ := CaseSplit.mxr2 (aP16_2) 10 18
def N16_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (if c16_0 ra t && c16_4 rb t then 1 else 0)
def aP16_3 (ra rb : ℕ) : ℤ := -(1) * N16_3 ra rb + u16 (82 + rb) + u16 (105 + ra)
def MP16_3 : ℤ := CaseSplit.mxr2 (aP16_3) 10 22
def P16_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (if c16_1 ra t && c16_2 rb t then 1 else 0)
def C16_4 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (if c16_1 ra t && c16_2 rb t && c16_0 s t then 1 else 0)
def M16_4 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C16_4 ra rb) 10
def E16_4 : List ℕ := [61, 67, 140, 151]
def N16_4 (ra rb : ℕ) : ℤ := if E16_4.contains (ra * 17 + rb) = true then P16_4 ra rb - M16_4 ra rb else 0
def aP16_4 (ra rb : ℕ) : ℤ := -(1) * N16_4 ra rb + u16 (116 + rb) + u16 (133 + ra)
def MP16_4 : ℤ := CaseSplit.mxr2 (aP16_4) 12 16
def P16_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (if c16_1 ra t && c16_3 rb t then 1 else 0)
def C16_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (if c16_1 ra t && c16_3 rb t && c16_0 s t then 1 else 0)
def M16_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C16_5 ra rb) 10
def E16_5 : List ℕ := [7, 31, 37, 71, 107, 113, 147, 152, 178, 228]
def N16_5 (ra rb : ℕ) : ℤ := if E16_5.contains (ra * 19 + rb) = true then P16_5 ra rb - M16_5 ra rb else 0
def aP16_5 (ra rb : ℕ) : ℤ := -(1) * N16_5 ra rb + u16 (146 + rb) + u16 (165 + ra)
def MP16_5 : ℤ := CaseSplit.mxr2 (aP16_5) 12 18
def P16_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (if c16_1 ra t && c16_4 rb t then 1 else 0)
def C16_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n16, (if c16_1 ra t && c16_4 rb t && c16_0 s t then 1 else 0)
def M16_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C16_6 ra rb) 10
def E16_6 : List ℕ := []
def N16_6 (ra rb : ℕ) : ℤ := if E16_6.contains (ra * 23 + rb) = true then P16_6 ra rb - M16_6 ra rb else 0
def aP16_6 (ra rb : ℕ) : ℤ := -(1) * N16_6 ra rb + u16 (178 + rb) + u16 (201 + ra)
def MP16_6 : ℤ := CaseSplit.mxr2 (aP16_6) 12 22
def N16_7 (_ra _rb : ℕ) : ℤ := 0
def aP16_7 (ra rb : ℕ) : ℤ := -(1) * N16_7 ra rb + u16 (214 + rb) + u16 (233 + ra)
def MP16_7 : ℤ := CaseSplit.mxr2 (aP16_7) 16 18
def N16_8 (_ra _rb : ℕ) : ℤ := 0
def aP16_8 (ra rb : ℕ) : ℤ := -(1) * N16_8 ra rb + u16 (250 + rb) + u16 (273 + ra)
def MP16_8 : ℤ := CaseSplit.mxr2 (aP16_8) 16 22
def N16_9 (_ra _rb : ℕ) : ℤ := 0
def aP16_9 (ra rb : ℕ) : ℤ := -(1) * N16_9 ra rb + u16 (290 + rb) + u16 (313 + ra)
def MP16_9 : ℤ := CaseSplit.mxr2 (aP16_9) 18 22

def rhs16 : ℤ := (∑ t ∈ Finset.range n16, w16 t) + 1 * (n16 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn16 : ∀ t, t < n16 → (0 : ℤ) ≤ w16 t := by decide
theorem plt16 : ∀ t, t < n16 → q16 t < 39 := by decide
theorem pfree16_5 : ∀ t, t < n16 → gb5 2 (q16 t) = false := by decide
theorem pfree16_7 : ∀ t, t < n16 → gb7 2 (q16 t) = false := by decide
theorem MSv16_0 : MS16_0 = 4 := by decide +kernel
theorem MSv16_1 : MS16_1 = 8 := by decide +kernel
theorem MSv16_2 : MS16_2 = 0 := by decide +kernel
theorem MSv16_3 : MS16_3 = 0 := by decide +kernel
theorem MSv16_4 : MS16_4 = 0 := by decide +kernel
theorem MPv16_0 : MP16_0 = 0 := by decide +kernel
theorem MPv16_1 : MP16_1 = 0 := by decide +kernel
theorem MPv16_2 : MP16_2 = 0 := by decide +kernel
theorem MPv16_3 : MP16_3 = 0 := by decide +kernel
theorem MPv16_4 : MP16_4 = 0 := by decide +kernel
theorem MPv16_5 : MP16_5 = 0 := by decide +kernel
theorem MPv16_6 : MP16_6 = 0 := by decide +kernel
theorem MPv16_7 : MP16_7 = 0 := by decide +kernel
theorem MPv16_8 : MP16_8 = 0 := by decide +kernel
theorem MPv16_9 : MP16_9 = 5 := by decide +kernel
theorem rhsv16 : rhs16 = 18 := by decide +kernel

/-- **The case-16 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/1.
    (Scaled by the common denominator 1: 17 < 18.) -/
theorem cert16 : MS16_0 + MS16_1 + MS16_2 + MS16_3 + MS16_4 + MP16_0 + MP16_1 + MP16_2 + MP16_3 + MP16_4 + MP16_5 + MP16_6 + MP16_7 + MP16_8 + MP16_9 < rhs16 := by
  rw [MSv16_0, MSv16_1, MSv16_2, MSv16_3, MSv16_4, MPv16_0, MPv16_1, MPv16_2, MPv16_3, MPv16_4, MPv16_5, MPv16_6, MPv16_7, MPv16_8, MPv16_9, rhsv16]
  decide

def Dg16 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := (if c16_0 r0 t then 1 else 0) + (if c16_1 r1 t then 1 else 0) + (if c16_2 r2 t then 1 else 0) + (if c16_3 r3 t then 1 else 0) + (if c16_4 r4 t then 1 else 0)
def Wl16_0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c16_0 r0 t && c16_1 r1 t then 1 else 0
def Wl16_1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c16_0 r0 t && c16_2 r2 t then 1 else 0
def Wl16_2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c16_0 r0 t && c16_3 r3 t then 1 else 0
def Wl16_3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c16_0 r0 t && c16_4 r4 t then 1 else 0
def Wl16_4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c16_0 r0 t && c16_1 r1 t && c16_2 r2 t then 1 else 0
def Wl16_5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c16_0 r0 t && c16_1 r1 t && c16_3 r3 t then 1 else 0
def Wl16_6 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c16_0 r0 t && c16_1 r1 t && c16_4 r4 t then 1 else 0
def Wl16_7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c16_0 r0 t && !c16_1 r1 t && c16_2 r2 t && c16_3 r3 t then 1 else 0
def Wl16_8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c16_0 r0 t && !c16_1 r1 t && c16_2 r2 t && c16_4 r4 t then 1 else 0
def Wl16_9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c16_0 r0 t && !c16_1 r1 t && !c16_2 r2 t && c16_3 r3 t && c16_4 r4 t then 1 else 0

/-- **No configuration blocks the whole window in case 16.** -/
theorem nocov16 {r0 r1 r2 r3 r4 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23)
    (hcov : ∀ t, t < n16 → (c16_0 r0 t || c16_1 r1 t || c16_2 r2 t || c16_3 r3 t || c16_4 r4 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n16, (1 : ℤ) + (Wl16_0 r0 r1 r2 r3 r4 t + Wl16_1 r0 r1 r2 r3 r4 t + Wl16_2 r0 r1 r2 r3 r4 t + Wl16_3 r0 r1 r2 r3 r4 t + Wl16_4 r0 r1 r2 r3 r4 t + Wl16_5 r0 r1 r2 r3 r4 t + Wl16_6 r0 r1 r2 r3 r4 t + Wl16_7 r0 r1 r2 r3 r4 t + Wl16_8 r0 r1 r2 r3 r4 t + Wl16_9 r0 r1 r2 r3 r4 t) ≤ Dg16 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Wl16_0, Wl16_1, Wl16_2, Wl16_3, Wl16_4, Wl16_5, Wl16_6, Wl16_7, Wl16_8, Wl16_9, Dg16]
    exact CaseSplit.lowest5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n16, (1 : ℤ) ≤ Dg16 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Dg16]
    exact CaseSplit.degpos5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n16 : ℤ) + ((∑ t ∈ Finset.range n16, Wl16_0 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n16, Wl16_1 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n16, Wl16_2 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n16, Wl16_3 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n16, Wl16_4 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n16, Wl16_5 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n16, Wl16_6 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n16, Wl16_7 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n16, Wl16_8 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n16, Wl16_9 r0 r1 r2 r3 r4 t)) ≤ ∑ t ∈ Finset.range n16, Dg16 r0 r1 r2 r3 r4 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N16_0 r0 r1 ≤ ∑ t ∈ Finset.range n16, Wl16_0 r0 r1 r2 r3 r4 t := by
    simp only [N16_0, Wl16_0, le_refl]
  have hn1 : N16_1 r0 r2 ≤ ∑ t ∈ Finset.range n16, Wl16_1 r0 r1 r2 r3 r4 t := by
    simp only [N16_1, Wl16_1, le_refl]
  have hn2 : N16_2 r0 r3 ≤ ∑ t ∈ Finset.range n16, Wl16_2 r0 r1 r2 r3 r4 t := by
    simp only [N16_2, Wl16_2, le_refl]
  have hn3 : N16_3 r0 r4 ≤ ∑ t ∈ Finset.range n16, Wl16_3 r0 r1 r2 r3 r4 t := by
    simp only [N16_3, Wl16_3, le_refl]
  have hn4 : N16_4 r1 r2 ≤ ∑ t ∈ Finset.range n16, Wl16_4 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n16, Wl16_4 r0 r1 r2 r3 r4 t
        = (if c16_1 r1 t && c16_2 r2 t then (1:ℤ) else 0)
          - (if c16_1 r1 t && c16_2 r2 t && c16_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl16_4]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n16, Wl16_4 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl16_4]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n16, Wl16_4 r0 r1 r2 r3 r4 t
        = P16_4 r1 r2 - C16_4 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P16_4, C16_4]
    have hm : C16_4 r1 r2 r0 ≤ M16_4 r1 r2 :=
      CaseSplit.le_mxr (C16_4 r1 r2) 10 r0 (by omega)
    simp only [N16_4]
    split
    · rw [hL]; omega
    · exact hnn
  have hn5 : N16_5 r1 r3 ≤ ∑ t ∈ Finset.range n16, Wl16_5 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n16, Wl16_5 r0 r1 r2 r3 r4 t
        = (if c16_1 r1 t && c16_3 r3 t then (1:ℤ) else 0)
          - (if c16_1 r1 t && c16_3 r3 t && c16_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl16_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n16, Wl16_5 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl16_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n16, Wl16_5 r0 r1 r2 r3 r4 t
        = P16_5 r1 r3 - C16_5 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P16_5, C16_5]
    have hm : C16_5 r1 r3 r0 ≤ M16_5 r1 r3 :=
      CaseSplit.le_mxr (C16_5 r1 r3) 10 r0 (by omega)
    simp only [N16_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N16_6 r1 r4 ≤ ∑ t ∈ Finset.range n16, Wl16_6 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n16, Wl16_6 r0 r1 r2 r3 r4 t
        = (if c16_1 r1 t && c16_4 r4 t then (1:ℤ) else 0)
          - (if c16_1 r1 t && c16_4 r4 t && c16_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl16_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n16, Wl16_6 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl16_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n16, Wl16_6 r0 r1 r2 r3 r4 t
        = P16_6 r1 r4 - C16_6 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P16_6, C16_6]
    have hm : C16_6 r1 r4 r0 ≤ M16_6 r1 r4 :=
      CaseSplit.le_mxr (C16_6 r1 r4) 10 r0 (by omega)
    simp only [N16_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N16_7 r2 r3 ≤ ∑ t ∈ Finset.range n16, Wl16_7 r0 r1 r2 r3 r4 t := by
    simp only [N16_7]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl16_7]
    exact CaseSplit.ind_nonneg _
  have hn8 : N16_8 r2 r4 ≤ ∑ t ∈ Finset.range n16, Wl16_8 r0 r1 r2 r3 r4 t := by
    simp only [N16_8]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl16_8]
    exact CaseSplit.ind_nonneg _
  have hn9 : N16_9 r3 r4 ≤ ∑ t ∈ Finset.range n16, Wl16_9 r0 r1 r2 r3 r4 t := by
    simp only [N16_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl16_9]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n16, (w16 t + 1) * Dg16 r0 r1 r2 r3 r4 t = S16_0 r0 + S16_1 r1 + S16_2 r2 + S16_3 r3 + S16_4 r4 := by
    simp only [S16_0, S16_1, S16_2, S16_3, S16_4, Dg16, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n16, (w16 t + 1) * Dg16 r0 r1 r2 r3 r4 t
      = (∑ t ∈ Finset.range n16, w16 t * Dg16 r0 r1 r2 r3 r4 t)
        + 1 * (∑ t ∈ Finset.range n16, Dg16 r0 r1 r2 r3 r4 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n16, w16 t)
      ≤ ∑ t ∈ Finset.range n16, w16 t * Dg16 r0 r1 r2 r3 r4 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg16 r0 r1 r2 r3 r4 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w16 t := wnn16 t (Finset.mem_range.mp ht)
    calc w16 t = w16 t * 1 := (mul_one _).symm
      _ ≤ w16 t * Dg16 r0 r1 r2 r3 r4 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS16_0 r0 + aS16_1 r1 + aS16_2 r2 + aS16_3 r3 + aS16_4 r4) + (aP16_0 r0 r1 + aP16_1 r0 r2 + aP16_2 r0 r3 + aP16_3 r0 r4 + aP16_4 r1 r2 + aP16_5 r1 r3 + aP16_6 r1 r4 + aP16_7 r2 r3 + aP16_8 r2 r4 + aP16_9 r3 r4) = (S16_0 r0 + S16_1 r1 + S16_2 r2 + S16_3 r3 + S16_4 r4) - 1 * (N16_0 r0 r1 + N16_1 r0 r2 + N16_2 r0 r3 + N16_3 r0 r4 + N16_4 r1 r2 + N16_5 r1 r3 + N16_6 r1 r4 + N16_7 r2 r3 + N16_8 r2 r4 + N16_9 r3 r4) := by
    simp only [aS16_0, aS16_1, aS16_2, aS16_3, aS16_4, aP16_0, aP16_1, aP16_2, aP16_3, aP16_4, aP16_5, aP16_6, aP16_7, aP16_8, aP16_9, L16_0, L16_1, L16_2, L16_3, L16_4]
    ring
  have bS0 : aS16_0 r0 ≤ MS16_0 := CaseSplit.le_mxr (aS16_0) 10 r0 (by omega)
  have bS1 : aS16_1 r1 ≤ MS16_1 := CaseSplit.le_mxr (aS16_1) 12 r1 (by omega)
  have bS2 : aS16_2 r2 ≤ MS16_2 := CaseSplit.le_mxr (aS16_2) 16 r2 (by omega)
  have bS3 : aS16_3 r3 ≤ MS16_3 := CaseSplit.le_mxr (aS16_3) 18 r3 (by omega)
  have bS4 : aS16_4 r4 ≤ MS16_4 := CaseSplit.le_mxr (aS16_4) 22 r4 (by omega)
  have bP0 : aP16_0 r0 r1 ≤ MP16_0 := CaseSplit.le_mxr2 (aP16_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP16_1 r0 r2 ≤ MP16_1 := CaseSplit.le_mxr2 (aP16_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP16_2 r0 r3 ≤ MP16_2 := CaseSplit.le_mxr2 (aP16_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP16_3 r0 r4 ≤ MP16_3 := CaseSplit.le_mxr2 (aP16_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP16_4 r1 r2 ≤ MP16_4 := CaseSplit.le_mxr2 (aP16_4) 12 16 r1 r2 (by omega) (by omega)
  have bP5 : aP16_5 r1 r3 ≤ MP16_5 := CaseSplit.le_mxr2 (aP16_5) 12 18 r1 r3 (by omega) (by omega)
  have bP6 : aP16_6 r1 r4 ≤ MP16_6 := CaseSplit.le_mxr2 (aP16_6) 12 22 r1 r4 (by omega) (by omega)
  have bP7 : aP16_7 r2 r3 ≤ MP16_7 := CaseSplit.le_mxr2 (aP16_7) 16 18 r2 r3 (by omega) (by omega)
  have bP8 : aP16_8 r2 r4 ≤ MP16_8 := CaseSplit.le_mxr2 (aP16_8) 16 22 r2 r4 (by omega) (by omega)
  have bP9 : aP16_9 r3 r4 ≤ MP16_9 := CaseSplit.le_mxr2 (aP16_9) 18 22 r3 r4 (by omega) (by omega)
  have hrhs : rhs16 = (∑ t ∈ Finset.range n16, w16 t) + 1 * (n16 : ℤ) := rfl
  have hc := cert16
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, bS0, bS1, bS2, bS3, bS4, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9]

end IncCert23
