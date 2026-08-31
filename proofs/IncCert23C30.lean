/-
INCREMENT-WIDTH CERTIFICATE, step 19->23, case 30 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_19_23.json, which re-derives every number
from the primes alone).

Machine 23, INCREMENT width 39 = F_2(19) + s_min(23) = 31 + 8,
held gears [5, 7] at phases [4, 2].  Free gears [11, 13, 17, 19, 23].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 1.
-/
import IncCert23B

namespace IncCert23

/-! ### case 30: held gears at phases [4, 2] -/

def p30 : List ℕ := [1, 3, 8, 9, 14, 16, 19, 21, 23, 24, 26, 28, 29, 31, 33, 36, 38]
def q30 (t : ℕ) : ℕ := p30.getD t 0
def n30 : ℕ := 17
def yl30 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def w30 (t : ℕ) : ℤ := yl30.getD t 0
def ul30 : List ℤ := [0, 0, 0, 0, 0, 0, (-2), (-2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), 0, (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 2, 1, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2, (-2), (-2), (-2), (-2), (-4), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, (-1), (-1), (-1), (-1), (-1), (-2), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1, 3, 3, 2, 3, 1, 2, 3, 2, 3, 2, 3, 3, 1, 3, 3, 3, 3, 1, 3, 2, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0]
def u30 (k : ℕ) : ℤ := ul30.getD k 0

def c30_0 (r t : ℕ) : Bool := gb11 r (q30 t)
def c30_1 (r t : ℕ) : Bool := gb13 r (q30 t)
def c30_2 (r t : ℕ) : Bool := gb17 r (q30 t)
def c30_3 (r t : ℕ) : Bool := gb19 r (q30 t)
def c30_4 (r t : ℕ) : Bool := gb23 r (q30 t)

def S30_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (w30 t + 1) * (if c30_0 r t then 1 else 0)
def S30_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (w30 t + 1) * (if c30_1 r t then 1 else 0)
def S30_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (w30 t + 1) * (if c30_2 r t then 1 else 0)
def S30_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (w30 t + 1) * (if c30_3 r t then 1 else 0)
def S30_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (w30 t + 1) * (if c30_4 r t then 1 else 0)

def L30_0 (r : ℕ) : ℤ := u30 (13 + r) + u30 (41 + r) + u30 (71 + r) + u30 (105 + r)
def L30_1 (r : ℕ) : ℤ := u30 (0 + r) + u30 (133 + r) + u30 (165 + r) + u30 (201 + r)
def L30_2 (r : ℕ) : ℤ := u30 (24 + r) + u30 (116 + r) + u30 (233 + r) + u30 (273 + r)
def L30_3 (r : ℕ) : ℤ := u30 (52 + r) + u30 (146 + r) + u30 (214 + r) + u30 (313 + r)
def L30_4 (r : ℕ) : ℤ := u30 (82 + r) + u30 (178 + r) + u30 (250 + r) + u30 (290 + r)

def aS30_0 (r : ℕ) : ℤ := S30_0 r - L30_0 r
def MS30_0 : ℤ := CaseSplit.mxr (aS30_0) 10
def aS30_1 (r : ℕ) : ℤ := S30_1 r - L30_1 r
def MS30_1 : ℤ := CaseSplit.mxr (aS30_1) 12
def aS30_2 (r : ℕ) : ℤ := S30_2 r - L30_2 r
def MS30_2 : ℤ := CaseSplit.mxr (aS30_2) 16
def aS30_3 (r : ℕ) : ℤ := S30_3 r - L30_3 r
def MS30_3 : ℤ := CaseSplit.mxr (aS30_3) 18
def aS30_4 (r : ℕ) : ℤ := S30_4 r - L30_4 r
def MS30_4 : ℤ := CaseSplit.mxr (aS30_4) 22

def N30_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (if c30_0 ra t && c30_1 rb t then 1 else 0)
def aP30_0 (ra rb : ℕ) : ℤ := -(1) * N30_0 ra rb + u30 (0 + rb) + u30 (13 + ra)
def MP30_0 : ℤ := CaseSplit.mxr2 (aP30_0) 10 12
def N30_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (if c30_0 ra t && c30_2 rb t then 1 else 0)
def aP30_1 (ra rb : ℕ) : ℤ := -(1) * N30_1 ra rb + u30 (24 + rb) + u30 (41 + ra)
def MP30_1 : ℤ := CaseSplit.mxr2 (aP30_1) 10 16
def N30_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (if c30_0 ra t && c30_3 rb t then 1 else 0)
def aP30_2 (ra rb : ℕ) : ℤ := -(1) * N30_2 ra rb + u30 (52 + rb) + u30 (71 + ra)
def MP30_2 : ℤ := CaseSplit.mxr2 (aP30_2) 10 18
def N30_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (if c30_0 ra t && c30_4 rb t then 1 else 0)
def aP30_3 (ra rb : ℕ) : ℤ := -(1) * N30_3 ra rb + u30 (82 + rb) + u30 (105 + ra)
def MP30_3 : ℤ := CaseSplit.mxr2 (aP30_3) 10 22
def P30_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (if c30_1 ra t && c30_2 rb t then 1 else 0)
def C30_4 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (if c30_1 ra t && c30_2 rb t && c30_0 s t then 1 else 0)
def M30_4 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C30_4 ra rb) 10
def E30_4 : List ℕ := [39, 45, 61, 67, 86, 97, 140, 151, 170, 176]
def N30_4 (ra rb : ℕ) : ℤ := if E30_4.contains (ra * 17 + rb) = true then P30_4 ra rb - M30_4 ra rb else 0
def aP30_4 (ra rb : ℕ) : ℤ := -(1) * N30_4 ra rb + u30 (116 + rb) + u30 (133 + ra)
def MP30_4 : ℤ := CaseSplit.mxr2 (aP30_4) 12 16
def P30_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (if c30_1 ra t && c30_3 rb t then 1 else 0)
def C30_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (if c30_1 ra t && c30_3 rb t && c30_0 s t then 1 else 0)
def M30_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C30_5 ra rb) 10
def E30_5 : List ℕ := [21, 37, 71, 113, 147, 152, 158, 192, 228, 234]
def N30_5 (ra rb : ℕ) : ℤ := if E30_5.contains (ra * 19 + rb) = true then P30_5 ra rb - M30_5 ra rb else 0
def aP30_5 (ra rb : ℕ) : ℤ := -(1) * N30_5 ra rb + u30 (146 + rb) + u30 (165 + ra)
def MP30_5 : ℤ := CaseSplit.mxr2 (aP30_5) 12 18
def P30_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (if c30_1 ra t && c30_4 rb t then 1 else 0)
def C30_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n30, (if c30_1 ra t && c30_4 rb t && c30_0 s t then 1 else 0)
def M30_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C30_6 ra rb) 10
def E30_6 : List ℕ := []
def N30_6 (ra rb : ℕ) : ℤ := if E30_6.contains (ra * 23 + rb) = true then P30_6 ra rb - M30_6 ra rb else 0
def aP30_6 (ra rb : ℕ) : ℤ := -(1) * N30_6 ra rb + u30 (178 + rb) + u30 (201 + ra)
def MP30_6 : ℤ := CaseSplit.mxr2 (aP30_6) 12 22
def N30_7 (_ra _rb : ℕ) : ℤ := 0
def aP30_7 (ra rb : ℕ) : ℤ := -(1) * N30_7 ra rb + u30 (214 + rb) + u30 (233 + ra)
def MP30_7 : ℤ := CaseSplit.mxr2 (aP30_7) 16 18
def N30_8 (_ra _rb : ℕ) : ℤ := 0
def aP30_8 (ra rb : ℕ) : ℤ := -(1) * N30_8 ra rb + u30 (250 + rb) + u30 (273 + ra)
def MP30_8 : ℤ := CaseSplit.mxr2 (aP30_8) 16 22
def N30_9 (_ra _rb : ℕ) : ℤ := 0
def aP30_9 (ra rb : ℕ) : ℤ := -(1) * N30_9 ra rb + u30 (290 + rb) + u30 (313 + ra)
def MP30_9 : ℤ := CaseSplit.mxr2 (aP30_9) 18 22

def rhs30 : ℤ := (∑ t ∈ Finset.range n30, w30 t) + 1 * (n30 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn30 : ∀ t, t < n30 → (0 : ℤ) ≤ w30 t := by decide
theorem plt30 : ∀ t, t < n30 → q30 t < 39 := by decide
theorem pfree30_5 : ∀ t, t < n30 → gb5 4 (q30 t) = false := by decide
theorem pfree30_7 : ∀ t, t < n30 → gb7 2 (q30 t) = false := by decide
theorem MSv30_0 : MS30_0 = 5 := by decide +kernel
theorem MSv30_1 : MS30_1 = 7 := by decide +kernel
theorem MSv30_2 : MS30_2 = 0 := by decide +kernel
theorem MSv30_3 : MS30_3 = 0 := by decide +kernel
theorem MSv30_4 : MS30_4 = 0 := by decide +kernel
theorem MPv30_0 : MP30_0 = 0 := by decide +kernel
theorem MPv30_1 : MP30_1 = 0 := by decide +kernel
theorem MPv30_2 : MP30_2 = 0 := by decide +kernel
theorem MPv30_3 : MP30_3 = 0 := by decide +kernel
theorem MPv30_4 : MP30_4 = 0 := by decide +kernel
theorem MPv30_5 : MP30_5 = 0 := by decide +kernel
theorem MPv30_6 : MP30_6 = 0 := by decide +kernel
theorem MPv30_7 : MP30_7 = 0 := by decide +kernel
theorem MPv30_8 : MP30_8 = 0 := by decide +kernel
theorem MPv30_9 : MP30_9 = 4 := by decide +kernel
theorem rhsv30 : rhs30 = 17 := by decide +kernel

/-- **The case-30 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/1.
    (Scaled by the common denominator 1: 16 < 17.) -/
theorem cert30 : MS30_0 + MS30_1 + MS30_2 + MS30_3 + MS30_4 + MP30_0 + MP30_1 + MP30_2 + MP30_3 + MP30_4 + MP30_5 + MP30_6 + MP30_7 + MP30_8 + MP30_9 < rhs30 := by
  rw [MSv30_0, MSv30_1, MSv30_2, MSv30_3, MSv30_4, MPv30_0, MPv30_1, MPv30_2, MPv30_3, MPv30_4, MPv30_5, MPv30_6, MPv30_7, MPv30_8, MPv30_9, rhsv30]
  decide

def Dg30 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := (if c30_0 r0 t then 1 else 0) + (if c30_1 r1 t then 1 else 0) + (if c30_2 r2 t then 1 else 0) + (if c30_3 r3 t then 1 else 0) + (if c30_4 r4 t then 1 else 0)
def Wl30_0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c30_0 r0 t && c30_1 r1 t then 1 else 0
def Wl30_1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c30_0 r0 t && c30_2 r2 t then 1 else 0
def Wl30_2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c30_0 r0 t && c30_3 r3 t then 1 else 0
def Wl30_3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c30_0 r0 t && c30_4 r4 t then 1 else 0
def Wl30_4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c30_0 r0 t && c30_1 r1 t && c30_2 r2 t then 1 else 0
def Wl30_5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c30_0 r0 t && c30_1 r1 t && c30_3 r3 t then 1 else 0
def Wl30_6 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c30_0 r0 t && c30_1 r1 t && c30_4 r4 t then 1 else 0
def Wl30_7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c30_0 r0 t && !c30_1 r1 t && c30_2 r2 t && c30_3 r3 t then 1 else 0
def Wl30_8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c30_0 r0 t && !c30_1 r1 t && c30_2 r2 t && c30_4 r4 t then 1 else 0
def Wl30_9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c30_0 r0 t && !c30_1 r1 t && !c30_2 r2 t && c30_3 r3 t && c30_4 r4 t then 1 else 0

/-- **No configuration blocks the whole window in case 30.** -/
theorem nocov30 {r0 r1 r2 r3 r4 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23)
    (hcov : ∀ t, t < n30 → (c30_0 r0 t || c30_1 r1 t || c30_2 r2 t || c30_3 r3 t || c30_4 r4 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n30, (1 : ℤ) + (Wl30_0 r0 r1 r2 r3 r4 t + Wl30_1 r0 r1 r2 r3 r4 t + Wl30_2 r0 r1 r2 r3 r4 t + Wl30_3 r0 r1 r2 r3 r4 t + Wl30_4 r0 r1 r2 r3 r4 t + Wl30_5 r0 r1 r2 r3 r4 t + Wl30_6 r0 r1 r2 r3 r4 t + Wl30_7 r0 r1 r2 r3 r4 t + Wl30_8 r0 r1 r2 r3 r4 t + Wl30_9 r0 r1 r2 r3 r4 t) ≤ Dg30 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Wl30_0, Wl30_1, Wl30_2, Wl30_3, Wl30_4, Wl30_5, Wl30_6, Wl30_7, Wl30_8, Wl30_9, Dg30]
    exact CaseSplit.lowest5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n30, (1 : ℤ) ≤ Dg30 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Dg30]
    exact CaseSplit.degpos5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n30 : ℤ) + ((∑ t ∈ Finset.range n30, Wl30_0 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n30, Wl30_1 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n30, Wl30_2 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n30, Wl30_3 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n30, Wl30_4 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n30, Wl30_5 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n30, Wl30_6 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n30, Wl30_7 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n30, Wl30_8 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n30, Wl30_9 r0 r1 r2 r3 r4 t)) ≤ ∑ t ∈ Finset.range n30, Dg30 r0 r1 r2 r3 r4 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N30_0 r0 r1 ≤ ∑ t ∈ Finset.range n30, Wl30_0 r0 r1 r2 r3 r4 t := by
    simp only [N30_0, Wl30_0, le_refl]
  have hn1 : N30_1 r0 r2 ≤ ∑ t ∈ Finset.range n30, Wl30_1 r0 r1 r2 r3 r4 t := by
    simp only [N30_1, Wl30_1, le_refl]
  have hn2 : N30_2 r0 r3 ≤ ∑ t ∈ Finset.range n30, Wl30_2 r0 r1 r2 r3 r4 t := by
    simp only [N30_2, Wl30_2, le_refl]
  have hn3 : N30_3 r0 r4 ≤ ∑ t ∈ Finset.range n30, Wl30_3 r0 r1 r2 r3 r4 t := by
    simp only [N30_3, Wl30_3, le_refl]
  have hn4 : N30_4 r1 r2 ≤ ∑ t ∈ Finset.range n30, Wl30_4 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n30, Wl30_4 r0 r1 r2 r3 r4 t
        = (if c30_1 r1 t && c30_2 r2 t then (1:ℤ) else 0)
          - (if c30_1 r1 t && c30_2 r2 t && c30_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl30_4]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n30, Wl30_4 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl30_4]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n30, Wl30_4 r0 r1 r2 r3 r4 t
        = P30_4 r1 r2 - C30_4 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P30_4, C30_4]
    have hm : C30_4 r1 r2 r0 ≤ M30_4 r1 r2 :=
      CaseSplit.le_mxr (C30_4 r1 r2) 10 r0 (by omega)
    simp only [N30_4]
    split
    · rw [hL]; omega
    · exact hnn
  have hn5 : N30_5 r1 r3 ≤ ∑ t ∈ Finset.range n30, Wl30_5 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n30, Wl30_5 r0 r1 r2 r3 r4 t
        = (if c30_1 r1 t && c30_3 r3 t then (1:ℤ) else 0)
          - (if c30_1 r1 t && c30_3 r3 t && c30_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl30_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n30, Wl30_5 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl30_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n30, Wl30_5 r0 r1 r2 r3 r4 t
        = P30_5 r1 r3 - C30_5 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P30_5, C30_5]
    have hm : C30_5 r1 r3 r0 ≤ M30_5 r1 r3 :=
      CaseSplit.le_mxr (C30_5 r1 r3) 10 r0 (by omega)
    simp only [N30_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N30_6 r1 r4 ≤ ∑ t ∈ Finset.range n30, Wl30_6 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n30, Wl30_6 r0 r1 r2 r3 r4 t
        = (if c30_1 r1 t && c30_4 r4 t then (1:ℤ) else 0)
          - (if c30_1 r1 t && c30_4 r4 t && c30_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl30_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n30, Wl30_6 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl30_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n30, Wl30_6 r0 r1 r2 r3 r4 t
        = P30_6 r1 r4 - C30_6 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P30_6, C30_6]
    have hm : C30_6 r1 r4 r0 ≤ M30_6 r1 r4 :=
      CaseSplit.le_mxr (C30_6 r1 r4) 10 r0 (by omega)
    simp only [N30_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N30_7 r2 r3 ≤ ∑ t ∈ Finset.range n30, Wl30_7 r0 r1 r2 r3 r4 t := by
    simp only [N30_7]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl30_7]
    exact CaseSplit.ind_nonneg _
  have hn8 : N30_8 r2 r4 ≤ ∑ t ∈ Finset.range n30, Wl30_8 r0 r1 r2 r3 r4 t := by
    simp only [N30_8]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl30_8]
    exact CaseSplit.ind_nonneg _
  have hn9 : N30_9 r3 r4 ≤ ∑ t ∈ Finset.range n30, Wl30_9 r0 r1 r2 r3 r4 t := by
    simp only [N30_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl30_9]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n30, (w30 t + 1) * Dg30 r0 r1 r2 r3 r4 t = S30_0 r0 + S30_1 r1 + S30_2 r2 + S30_3 r3 + S30_4 r4 := by
    simp only [S30_0, S30_1, S30_2, S30_3, S30_4, Dg30, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n30, (w30 t + 1) * Dg30 r0 r1 r2 r3 r4 t
      = (∑ t ∈ Finset.range n30, w30 t * Dg30 r0 r1 r2 r3 r4 t)
        + 1 * (∑ t ∈ Finset.range n30, Dg30 r0 r1 r2 r3 r4 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n30, w30 t)
      ≤ ∑ t ∈ Finset.range n30, w30 t * Dg30 r0 r1 r2 r3 r4 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg30 r0 r1 r2 r3 r4 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w30 t := wnn30 t (Finset.mem_range.mp ht)
    calc w30 t = w30 t * 1 := (mul_one _).symm
      _ ≤ w30 t * Dg30 r0 r1 r2 r3 r4 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS30_0 r0 + aS30_1 r1 + aS30_2 r2 + aS30_3 r3 + aS30_4 r4) + (aP30_0 r0 r1 + aP30_1 r0 r2 + aP30_2 r0 r3 + aP30_3 r0 r4 + aP30_4 r1 r2 + aP30_5 r1 r3 + aP30_6 r1 r4 + aP30_7 r2 r3 + aP30_8 r2 r4 + aP30_9 r3 r4) = (S30_0 r0 + S30_1 r1 + S30_2 r2 + S30_3 r3 + S30_4 r4) - 1 * (N30_0 r0 r1 + N30_1 r0 r2 + N30_2 r0 r3 + N30_3 r0 r4 + N30_4 r1 r2 + N30_5 r1 r3 + N30_6 r1 r4 + N30_7 r2 r3 + N30_8 r2 r4 + N30_9 r3 r4) := by
    simp only [aS30_0, aS30_1, aS30_2, aS30_3, aS30_4, aP30_0, aP30_1, aP30_2, aP30_3, aP30_4, aP30_5, aP30_6, aP30_7, aP30_8, aP30_9, L30_0, L30_1, L30_2, L30_3, L30_4]
    ring
  have bS0 : aS30_0 r0 ≤ MS30_0 := CaseSplit.le_mxr (aS30_0) 10 r0 (by omega)
  have bS1 : aS30_1 r1 ≤ MS30_1 := CaseSplit.le_mxr (aS30_1) 12 r1 (by omega)
  have bS2 : aS30_2 r2 ≤ MS30_2 := CaseSplit.le_mxr (aS30_2) 16 r2 (by omega)
  have bS3 : aS30_3 r3 ≤ MS30_3 := CaseSplit.le_mxr (aS30_3) 18 r3 (by omega)
  have bS4 : aS30_4 r4 ≤ MS30_4 := CaseSplit.le_mxr (aS30_4) 22 r4 (by omega)
  have bP0 : aP30_0 r0 r1 ≤ MP30_0 := CaseSplit.le_mxr2 (aP30_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP30_1 r0 r2 ≤ MP30_1 := CaseSplit.le_mxr2 (aP30_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP30_2 r0 r3 ≤ MP30_2 := CaseSplit.le_mxr2 (aP30_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP30_3 r0 r4 ≤ MP30_3 := CaseSplit.le_mxr2 (aP30_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP30_4 r1 r2 ≤ MP30_4 := CaseSplit.le_mxr2 (aP30_4) 12 16 r1 r2 (by omega) (by omega)
  have bP5 : aP30_5 r1 r3 ≤ MP30_5 := CaseSplit.le_mxr2 (aP30_5) 12 18 r1 r3 (by omega) (by omega)
  have bP6 : aP30_6 r1 r4 ≤ MP30_6 := CaseSplit.le_mxr2 (aP30_6) 12 22 r1 r4 (by omega) (by omega)
  have bP7 : aP30_7 r2 r3 ≤ MP30_7 := CaseSplit.le_mxr2 (aP30_7) 16 18 r2 r3 (by omega) (by omega)
  have bP8 : aP30_8 r2 r4 ≤ MP30_8 := CaseSplit.le_mxr2 (aP30_8) 16 22 r2 r4 (by omega) (by omega)
  have bP9 : aP30_9 r3 r4 ≤ MP30_9 := CaseSplit.le_mxr2 (aP30_9) 18 22 r3 r4 (by omega) (by omega)
  have hrhs : rhs30 = (∑ t ∈ Finset.range n30, w30 t) + 1 * (n30 : ℤ) := rfl
  have hc := cert30
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, bS0, bS1, bS2, bS3, bS4, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9]

end IncCert23
