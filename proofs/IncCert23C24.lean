/-
INCREMENT-WIDTH CERTIFICATE, step 19->23, case 24 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_19_23.json, which re-derives every number
from the primes alone).

Machine 23, INCREMENT width 39 = F_2(19) + s_min(23) = 31 + 8,
held gears [5, 7] at phases [3, 3].  Free gears [11, 13, 17, 19, 23].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 1.
-/
import IncCert23B

namespace IncCert23

/-! ### case 24: held gears at phases [3, 3] -/

def p24 : List ℕ := [0, 2, 4, 7, 9, 14, 15, 20, 22, 25, 27, 29, 30, 32, 34, 35, 37]
def q24 (t : ℕ) : ℕ := p24.getD t 0
def n24 : ℕ := 17
def yl24 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
def w24 (t : ℕ) : ℤ := yl24.getD t 0
def ul24 : List ℤ := [0, (-1), 0, (-1), 0, (-1), 0, 0, (-1), 0, (-1), 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, (-1), (-1), (-1), 0, (-1), 0, (-1), (-1), 0, 0, (-1), (-1), (-1), (-1), 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, (-1), (-1), (-1), 0, 0, (-1), (-1), (-2), (-1), (-1), 0, 0, 0, (-1), (-1), (-1), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), 0, 0, 1, 0, 0, 1, 1, 0, (-1), 1, 0, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 3, 3, 3, 3, 2, 3, (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 4, 1, 4, 3, 1, 4, 2, 1, 3, 1, 4, 3, 1, 4, 2, 3, 3, 4, 4, 2, 2, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0]
def u24 (k : ℕ) : ℤ := ul24.getD k 0

def c24_0 (r t : ℕ) : Bool := gb11 r (q24 t)
def c24_1 (r t : ℕ) : Bool := gb13 r (q24 t)
def c24_2 (r t : ℕ) : Bool := gb17 r (q24 t)
def c24_3 (r t : ℕ) : Bool := gb19 r (q24 t)
def c24_4 (r t : ℕ) : Bool := gb23 r (q24 t)

def S24_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (w24 t + 1) * (if c24_0 r t then 1 else 0)
def S24_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (w24 t + 1) * (if c24_1 r t then 1 else 0)
def S24_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (w24 t + 1) * (if c24_2 r t then 1 else 0)
def S24_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (w24 t + 1) * (if c24_3 r t then 1 else 0)
def S24_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (w24 t + 1) * (if c24_4 r t then 1 else 0)

def L24_0 (r : ℕ) : ℤ := u24 (13 + r) + u24 (41 + r) + u24 (71 + r) + u24 (105 + r)
def L24_1 (r : ℕ) : ℤ := u24 (0 + r) + u24 (133 + r) + u24 (165 + r) + u24 (201 + r)
def L24_2 (r : ℕ) : ℤ := u24 (24 + r) + u24 (116 + r) + u24 (233 + r) + u24 (273 + r)
def L24_3 (r : ℕ) : ℤ := u24 (52 + r) + u24 (146 + r) + u24 (214 + r) + u24 (313 + r)
def L24_4 (r : ℕ) : ℤ := u24 (82 + r) + u24 (178 + r) + u24 (250 + r) + u24 (290 + r)

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

def N24_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (if c24_0 ra t && c24_1 rb t then 1 else 0)
def aP24_0 (ra rb : ℕ) : ℤ := -(1) * N24_0 ra rb + u24 (0 + rb) + u24 (13 + ra)
def MP24_0 : ℤ := CaseSplit.mxr2 (aP24_0) 10 12
def N24_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (if c24_0 ra t && c24_2 rb t then 1 else 0)
def aP24_1 (ra rb : ℕ) : ℤ := -(1) * N24_1 ra rb + u24 (24 + rb) + u24 (41 + ra)
def MP24_1 : ℤ := CaseSplit.mxr2 (aP24_1) 10 16
def N24_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (if c24_0 ra t && c24_3 rb t then 1 else 0)
def aP24_2 (ra rb : ℕ) : ℤ := -(1) * N24_2 ra rb + u24 (52 + rb) + u24 (71 + ra)
def MP24_2 : ℤ := CaseSplit.mxr2 (aP24_2) 10 18
def N24_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (if c24_0 ra t && c24_4 rb t then 1 else 0)
def aP24_3 (ra rb : ℕ) : ℤ := -(1) * N24_3 ra rb + u24 (82 + rb) + u24 (105 + ra)
def MP24_3 : ℤ := CaseSplit.mxr2 (aP24_3) 10 22
def P24_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (if c24_1 ra t && c24_2 rb t then 1 else 0)
def C24_4 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (if c24_1 ra t && c24_2 rb t && c24_0 s t then 1 else 0)
def M24_4 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C24_4 ra rb) 10
def E24_4 : List ℕ := [68, 79, 158, 169]
def N24_4 (ra rb : ℕ) : ℤ := if E24_4.contains (ra * 17 + rb) = true then P24_4 ra rb - M24_4 ra rb else 0
def aP24_4 (ra rb : ℕ) : ℤ := -(1) * N24_4 ra rb + u24 (116 + rb) + u24 (133 + ra)
def MP24_4 : ℤ := CaseSplit.mxr2 (aP24_4) 12 16
def P24_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (if c24_1 ra t && c24_3 rb t then 1 else 0)
def C24_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (if c24_1 ra t && c24_3 rb t && c24_0 s t then 1 else 0)
def M24_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C24_5 ra rb) 10
def E24_5 : List ℕ := [1, 27, 38, 51, 91, 114, 127, 167, 172, 198]
def N24_5 (ra rb : ℕ) : ℤ := if E24_5.contains (ra * 19 + rb) = true then P24_5 ra rb - M24_5 ra rb else 0
def aP24_5 (ra rb : ℕ) : ℤ := -(1) * N24_5 ra rb + u24 (146 + rb) + u24 (165 + ra)
def MP24_5 : ℤ := CaseSplit.mxr2 (aP24_5) 12 18
def P24_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (if c24_1 ra t && c24_4 rb t then 1 else 0)
def C24_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n24, (if c24_1 ra t && c24_4 rb t && c24_0 s t then 1 else 0)
def M24_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C24_6 ra rb) 10
def E24_6 : List ℕ := []
def N24_6 (ra rb : ℕ) : ℤ := if E24_6.contains (ra * 23 + rb) = true then P24_6 ra rb - M24_6 ra rb else 0
def aP24_6 (ra rb : ℕ) : ℤ := -(1) * N24_6 ra rb + u24 (178 + rb) + u24 (201 + ra)
def MP24_6 : ℤ := CaseSplit.mxr2 (aP24_6) 12 22
def N24_7 (_ra _rb : ℕ) : ℤ := 0
def aP24_7 (ra rb : ℕ) : ℤ := -(1) * N24_7 ra rb + u24 (214 + rb) + u24 (233 + ra)
def MP24_7 : ℤ := CaseSplit.mxr2 (aP24_7) 16 18
def N24_8 (_ra _rb : ℕ) : ℤ := 0
def aP24_8 (ra rb : ℕ) : ℤ := -(1) * N24_8 ra rb + u24 (250 + rb) + u24 (273 + ra)
def MP24_8 : ℤ := CaseSplit.mxr2 (aP24_8) 16 22
def N24_9 (_ra _rb : ℕ) : ℤ := 0
def aP24_9 (ra rb : ℕ) : ℤ := -(1) * N24_9 ra rb + u24 (290 + rb) + u24 (313 + ra)
def MP24_9 : ℤ := CaseSplit.mxr2 (aP24_9) 18 22

def rhs24 : ℤ := (∑ t ∈ Finset.range n24, w24 t) + 1 * (n24 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn24 : ∀ t, t < n24 → (0 : ℤ) ≤ w24 t := by decide
theorem plt24 : ∀ t, t < n24 → q24 t < 39 := by decide
theorem pfree24_5 : ∀ t, t < n24 → gb5 3 (q24 t) = false := by decide
theorem pfree24_7 : ∀ t, t < n24 → gb7 3 (q24 t) = false := by decide
theorem MSv24_0 : MS24_0 = 3 := by decide +kernel
theorem MSv24_1 : MS24_1 = 9 := by decide +kernel
theorem MSv24_2 : MS24_2 = 0 := by decide +kernel
theorem MSv24_3 : MS24_3 = 0 := by decide +kernel
theorem MSv24_4 : MS24_4 = 0 := by decide +kernel
theorem MPv24_0 : MP24_0 = 0 := by decide +kernel
theorem MPv24_1 : MP24_1 = 0 := by decide +kernel
theorem MPv24_2 : MP24_2 = 0 := by decide +kernel
theorem MPv24_3 : MP24_3 = 0 := by decide +kernel
theorem MPv24_4 : MP24_4 = 0 := by decide +kernel
theorem MPv24_5 : MP24_5 = 0 := by decide +kernel
theorem MPv24_6 : MP24_6 = 0 := by decide +kernel
theorem MPv24_7 : MP24_7 = 0 := by decide +kernel
theorem MPv24_8 : MP24_8 = 0 := by decide +kernel
theorem MPv24_9 : MP24_9 = 5 := by decide +kernel
theorem rhsv24 : rhs24 = 18 := by decide +kernel

/-- **The case-24 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/1.
    (Scaled by the common denominator 1: 17 < 18.) -/
theorem cert24 : MS24_0 + MS24_1 + MS24_2 + MS24_3 + MS24_4 + MP24_0 + MP24_1 + MP24_2 + MP24_3 + MP24_4 + MP24_5 + MP24_6 + MP24_7 + MP24_8 + MP24_9 < rhs24 := by
  rw [MSv24_0, MSv24_1, MSv24_2, MSv24_3, MSv24_4, MPv24_0, MPv24_1, MPv24_2, MPv24_3, MPv24_4, MPv24_5, MPv24_6, MPv24_7, MPv24_8, MPv24_9, rhsv24]
  decide

def Dg24 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := (if c24_0 r0 t then 1 else 0) + (if c24_1 r1 t then 1 else 0) + (if c24_2 r2 t then 1 else 0) + (if c24_3 r3 t then 1 else 0) + (if c24_4 r4 t then 1 else 0)
def Wl24_0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c24_0 r0 t && c24_1 r1 t then 1 else 0
def Wl24_1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c24_0 r0 t && c24_2 r2 t then 1 else 0
def Wl24_2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c24_0 r0 t && c24_3 r3 t then 1 else 0
def Wl24_3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c24_0 r0 t && c24_4 r4 t then 1 else 0
def Wl24_4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c24_0 r0 t && c24_1 r1 t && c24_2 r2 t then 1 else 0
def Wl24_5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c24_0 r0 t && c24_1 r1 t && c24_3 r3 t then 1 else 0
def Wl24_6 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c24_0 r0 t && c24_1 r1 t && c24_4 r4 t then 1 else 0
def Wl24_7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c24_0 r0 t && !c24_1 r1 t && c24_2 r2 t && c24_3 r3 t then 1 else 0
def Wl24_8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c24_0 r0 t && !c24_1 r1 t && c24_2 r2 t && c24_4 r4 t then 1 else 0
def Wl24_9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c24_0 r0 t && !c24_1 r1 t && !c24_2 r2 t && c24_3 r3 t && c24_4 r4 t then 1 else 0

/-- **No configuration blocks the whole window in case 24.** -/
theorem nocov24 {r0 r1 r2 r3 r4 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23)
    (hcov : ∀ t, t < n24 → (c24_0 r0 t || c24_1 r1 t || c24_2 r2 t || c24_3 r3 t || c24_4 r4 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n24, (1 : ℤ) + (Wl24_0 r0 r1 r2 r3 r4 t + Wl24_1 r0 r1 r2 r3 r4 t + Wl24_2 r0 r1 r2 r3 r4 t + Wl24_3 r0 r1 r2 r3 r4 t + Wl24_4 r0 r1 r2 r3 r4 t + Wl24_5 r0 r1 r2 r3 r4 t + Wl24_6 r0 r1 r2 r3 r4 t + Wl24_7 r0 r1 r2 r3 r4 t + Wl24_8 r0 r1 r2 r3 r4 t + Wl24_9 r0 r1 r2 r3 r4 t) ≤ Dg24 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Wl24_0, Wl24_1, Wl24_2, Wl24_3, Wl24_4, Wl24_5, Wl24_6, Wl24_7, Wl24_8, Wl24_9, Dg24]
    exact CaseSplit.lowest5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n24, (1 : ℤ) ≤ Dg24 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Dg24]
    exact CaseSplit.degpos5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n24 : ℤ) + ((∑ t ∈ Finset.range n24, Wl24_0 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n24, Wl24_1 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n24, Wl24_2 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n24, Wl24_3 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n24, Wl24_4 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n24, Wl24_5 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n24, Wl24_6 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n24, Wl24_7 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n24, Wl24_8 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n24, Wl24_9 r0 r1 r2 r3 r4 t)) ≤ ∑ t ∈ Finset.range n24, Dg24 r0 r1 r2 r3 r4 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N24_0 r0 r1 ≤ ∑ t ∈ Finset.range n24, Wl24_0 r0 r1 r2 r3 r4 t := by
    simp only [N24_0, Wl24_0, le_refl]
  have hn1 : N24_1 r0 r2 ≤ ∑ t ∈ Finset.range n24, Wl24_1 r0 r1 r2 r3 r4 t := by
    simp only [N24_1, Wl24_1, le_refl]
  have hn2 : N24_2 r0 r3 ≤ ∑ t ∈ Finset.range n24, Wl24_2 r0 r1 r2 r3 r4 t := by
    simp only [N24_2, Wl24_2, le_refl]
  have hn3 : N24_3 r0 r4 ≤ ∑ t ∈ Finset.range n24, Wl24_3 r0 r1 r2 r3 r4 t := by
    simp only [N24_3, Wl24_3, le_refl]
  have hn4 : N24_4 r1 r2 ≤ ∑ t ∈ Finset.range n24, Wl24_4 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n24, Wl24_4 r0 r1 r2 r3 r4 t
        = (if c24_1 r1 t && c24_2 r2 t then (1:ℤ) else 0)
          - (if c24_1 r1 t && c24_2 r2 t && c24_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl24_4]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n24, Wl24_4 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl24_4]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n24, Wl24_4 r0 r1 r2 r3 r4 t
        = P24_4 r1 r2 - C24_4 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P24_4, C24_4]
    have hm : C24_4 r1 r2 r0 ≤ M24_4 r1 r2 :=
      CaseSplit.le_mxr (C24_4 r1 r2) 10 r0 (by omega)
    simp only [N24_4]
    split
    · rw [hL]; omega
    · exact hnn
  have hn5 : N24_5 r1 r3 ≤ ∑ t ∈ Finset.range n24, Wl24_5 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n24, Wl24_5 r0 r1 r2 r3 r4 t
        = (if c24_1 r1 t && c24_3 r3 t then (1:ℤ) else 0)
          - (if c24_1 r1 t && c24_3 r3 t && c24_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl24_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n24, Wl24_5 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl24_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n24, Wl24_5 r0 r1 r2 r3 r4 t
        = P24_5 r1 r3 - C24_5 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P24_5, C24_5]
    have hm : C24_5 r1 r3 r0 ≤ M24_5 r1 r3 :=
      CaseSplit.le_mxr (C24_5 r1 r3) 10 r0 (by omega)
    simp only [N24_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N24_6 r1 r4 ≤ ∑ t ∈ Finset.range n24, Wl24_6 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n24, Wl24_6 r0 r1 r2 r3 r4 t
        = (if c24_1 r1 t && c24_4 r4 t then (1:ℤ) else 0)
          - (if c24_1 r1 t && c24_4 r4 t && c24_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl24_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n24, Wl24_6 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl24_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n24, Wl24_6 r0 r1 r2 r3 r4 t
        = P24_6 r1 r4 - C24_6 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P24_6, C24_6]
    have hm : C24_6 r1 r4 r0 ≤ M24_6 r1 r4 :=
      CaseSplit.le_mxr (C24_6 r1 r4) 10 r0 (by omega)
    simp only [N24_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N24_7 r2 r3 ≤ ∑ t ∈ Finset.range n24, Wl24_7 r0 r1 r2 r3 r4 t := by
    simp only [N24_7]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl24_7]
    exact CaseSplit.ind_nonneg _
  have hn8 : N24_8 r2 r4 ≤ ∑ t ∈ Finset.range n24, Wl24_8 r0 r1 r2 r3 r4 t := by
    simp only [N24_8]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl24_8]
    exact CaseSplit.ind_nonneg _
  have hn9 : N24_9 r3 r4 ≤ ∑ t ∈ Finset.range n24, Wl24_9 r0 r1 r2 r3 r4 t := by
    simp only [N24_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl24_9]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n24, (w24 t + 1) * Dg24 r0 r1 r2 r3 r4 t = S24_0 r0 + S24_1 r1 + S24_2 r2 + S24_3 r3 + S24_4 r4 := by
    simp only [S24_0, S24_1, S24_2, S24_3, S24_4, Dg24, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n24, (w24 t + 1) * Dg24 r0 r1 r2 r3 r4 t
      = (∑ t ∈ Finset.range n24, w24 t * Dg24 r0 r1 r2 r3 r4 t)
        + 1 * (∑ t ∈ Finset.range n24, Dg24 r0 r1 r2 r3 r4 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n24, w24 t)
      ≤ ∑ t ∈ Finset.range n24, w24 t * Dg24 r0 r1 r2 r3 r4 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg24 r0 r1 r2 r3 r4 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w24 t := wnn24 t (Finset.mem_range.mp ht)
    calc w24 t = w24 t * 1 := (mul_one _).symm
      _ ≤ w24 t * Dg24 r0 r1 r2 r3 r4 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS24_0 r0 + aS24_1 r1 + aS24_2 r2 + aS24_3 r3 + aS24_4 r4) + (aP24_0 r0 r1 + aP24_1 r0 r2 + aP24_2 r0 r3 + aP24_3 r0 r4 + aP24_4 r1 r2 + aP24_5 r1 r3 + aP24_6 r1 r4 + aP24_7 r2 r3 + aP24_8 r2 r4 + aP24_9 r3 r4) = (S24_0 r0 + S24_1 r1 + S24_2 r2 + S24_3 r3 + S24_4 r4) - 1 * (N24_0 r0 r1 + N24_1 r0 r2 + N24_2 r0 r3 + N24_3 r0 r4 + N24_4 r1 r2 + N24_5 r1 r3 + N24_6 r1 r4 + N24_7 r2 r3 + N24_8 r2 r4 + N24_9 r3 r4) := by
    simp only [aS24_0, aS24_1, aS24_2, aS24_3, aS24_4, aP24_0, aP24_1, aP24_2, aP24_3, aP24_4, aP24_5, aP24_6, aP24_7, aP24_8, aP24_9, L24_0, L24_1, L24_2, L24_3, L24_4]
    ring
  have bS0 : aS24_0 r0 ≤ MS24_0 := CaseSplit.le_mxr (aS24_0) 10 r0 (by omega)
  have bS1 : aS24_1 r1 ≤ MS24_1 := CaseSplit.le_mxr (aS24_1) 12 r1 (by omega)
  have bS2 : aS24_2 r2 ≤ MS24_2 := CaseSplit.le_mxr (aS24_2) 16 r2 (by omega)
  have bS3 : aS24_3 r3 ≤ MS24_3 := CaseSplit.le_mxr (aS24_3) 18 r3 (by omega)
  have bS4 : aS24_4 r4 ≤ MS24_4 := CaseSplit.le_mxr (aS24_4) 22 r4 (by omega)
  have bP0 : aP24_0 r0 r1 ≤ MP24_0 := CaseSplit.le_mxr2 (aP24_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP24_1 r0 r2 ≤ MP24_1 := CaseSplit.le_mxr2 (aP24_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP24_2 r0 r3 ≤ MP24_2 := CaseSplit.le_mxr2 (aP24_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP24_3 r0 r4 ≤ MP24_3 := CaseSplit.le_mxr2 (aP24_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP24_4 r1 r2 ≤ MP24_4 := CaseSplit.le_mxr2 (aP24_4) 12 16 r1 r2 (by omega) (by omega)
  have bP5 : aP24_5 r1 r3 ≤ MP24_5 := CaseSplit.le_mxr2 (aP24_5) 12 18 r1 r3 (by omega) (by omega)
  have bP6 : aP24_6 r1 r4 ≤ MP24_6 := CaseSplit.le_mxr2 (aP24_6) 12 22 r1 r4 (by omega) (by omega)
  have bP7 : aP24_7 r2 r3 ≤ MP24_7 := CaseSplit.le_mxr2 (aP24_7) 16 18 r2 r3 (by omega) (by omega)
  have bP8 : aP24_8 r2 r4 ≤ MP24_8 := CaseSplit.le_mxr2 (aP24_8) 16 22 r2 r4 (by omega) (by omega)
  have bP9 : aP24_9 r3 r4 ≤ MP24_9 := CaseSplit.le_mxr2 (aP24_9) 18 22 r3 r4 (by omega) (by omega)
  have hrhs : rhs24 = (∑ t ∈ Finset.range n24, w24 t) + 1 * (n24 : ℤ) := rfl
  have hc := cert24
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, bS0, bS1, bS2, bS3, bS4, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9]

end IncCert23
