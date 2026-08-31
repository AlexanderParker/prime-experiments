/-
INCREMENT-WIDTH CERTIFICATE, step 19->23, case 9 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_19_23.json, which re-derives every number
from the primes alone).

Machine 23, INCREMENT width 39 = F_2(19) + s_min(23) = 31 + 8,
held gears [5, 7] at phases [1, 2].  Free gears [11, 13, 17, 19, 23].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 1.
-/
import IncCert23B

namespace IncCert23

/-! ### case 9: held gears at phases [1, 2] -/

def p9 : List ℕ := [1, 2, 7, 9, 12, 14, 16, 17, 19, 21, 22, 24, 26, 29, 31, 36, 37]
def q9 (t : ℕ) : ℕ := p9.getD t 0
def n9 : ℕ := 17
def yl9 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
def w9 (t : ℕ) : ℤ := yl9.getD t 0
def ul9 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, (-1), (-1), (-1), 0, 0, 0, (-1), (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 1, 1, 2, 2, 2, 3, 1, 2, 3, 3, 3, 1, 2, 2, (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), 1, 2, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 3, 0, 3, 3, 1, 2, 0, 2, 3, 3, 3, 1, 1, 1, 1, 3, 1, 3, 3, 0, 1, 1, 1, 1, 1, 0, 1, (-1), (-1), 1, 0, (-1), (-1), 1, 1, 1, 1, 0, 0]
def u9 (k : ℕ) : ℤ := ul9.getD k 0

def c9_0 (r t : ℕ) : Bool := gb11 r (q9 t)
def c9_1 (r t : ℕ) : Bool := gb13 r (q9 t)
def c9_2 (r t : ℕ) : Bool := gb17 r (q9 t)
def c9_3 (r t : ℕ) : Bool := gb19 r (q9 t)
def c9_4 (r t : ℕ) : Bool := gb23 r (q9 t)

def S9_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n9, (w9 t + 1) * (if c9_0 r t then 1 else 0)
def S9_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n9, (w9 t + 1) * (if c9_1 r t then 1 else 0)
def S9_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n9, (w9 t + 1) * (if c9_2 r t then 1 else 0)
def S9_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n9, (w9 t + 1) * (if c9_3 r t then 1 else 0)
def S9_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n9, (w9 t + 1) * (if c9_4 r t then 1 else 0)

def L9_0 (r : ℕ) : ℤ := u9 (13 + r) + u9 (41 + r) + u9 (71 + r) + u9 (105 + r)
def L9_1 (r : ℕ) : ℤ := u9 (0 + r) + u9 (133 + r) + u9 (165 + r) + u9 (201 + r)
def L9_2 (r : ℕ) : ℤ := u9 (24 + r) + u9 (116 + r) + u9 (233 + r) + u9 (273 + r)
def L9_3 (r : ℕ) : ℤ := u9 (52 + r) + u9 (146 + r) + u9 (214 + r) + u9 (313 + r)
def L9_4 (r : ℕ) : ℤ := u9 (82 + r) + u9 (178 + r) + u9 (250 + r) + u9 (290 + r)

def aS9_0 (r : ℕ) : ℤ := S9_0 r - L9_0 r
def MS9_0 : ℤ := CaseSplit.mxr (aS9_0) 10
def aS9_1 (r : ℕ) : ℤ := S9_1 r - L9_1 r
def MS9_1 : ℤ := CaseSplit.mxr (aS9_1) 12
def aS9_2 (r : ℕ) : ℤ := S9_2 r - L9_2 r
def MS9_2 : ℤ := CaseSplit.mxr (aS9_2) 16
def aS9_3 (r : ℕ) : ℤ := S9_3 r - L9_3 r
def MS9_3 : ℤ := CaseSplit.mxr (aS9_3) 18
def aS9_4 (r : ℕ) : ℤ := S9_4 r - L9_4 r
def MS9_4 : ℤ := CaseSplit.mxr (aS9_4) 22

def N9_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n9, (if c9_0 ra t && c9_1 rb t then 1 else 0)
def aP9_0 (ra rb : ℕ) : ℤ := -(1) * N9_0 ra rb + u9 (0 + rb) + u9 (13 + ra)
def MP9_0 : ℤ := CaseSplit.mxr2 (aP9_0) 10 12
def N9_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n9, (if c9_0 ra t && c9_2 rb t then 1 else 0)
def aP9_1 (ra rb : ℕ) : ℤ := -(1) * N9_1 ra rb + u9 (24 + rb) + u9 (41 + ra)
def MP9_1 : ℤ := CaseSplit.mxr2 (aP9_1) 10 16
def N9_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n9, (if c9_0 ra t && c9_3 rb t then 1 else 0)
def aP9_2 (ra rb : ℕ) : ℤ := -(1) * N9_2 ra rb + u9 (52 + rb) + u9 (71 + ra)
def MP9_2 : ℤ := CaseSplit.mxr2 (aP9_2) 10 18
def N9_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n9, (if c9_0 ra t && c9_4 rb t then 1 else 0)
def aP9_3 (ra rb : ℕ) : ℤ := -(1) * N9_3 ra rb + u9 (82 + rb) + u9 (105 + ra)
def MP9_3 : ℤ := CaseSplit.mxr2 (aP9_3) 10 22
def P9_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n9, (if c9_1 ra t && c9_2 rb t then 1 else 0)
def C9_4 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n9, (if c9_1 ra t && c9_2 rb t && c9_0 s t then 1 else 0)
def M9_4 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C9_4 ra rb) 10
def E9_4 : List ℕ := [39, 45, 75, 81, 86, 97, 154, 165, 170, 176, 206, 212]
def N9_4 (ra rb : ℕ) : ℤ := if E9_4.contains (ra * 17 + rb) = true then P9_4 ra rb - M9_4 ra rb else 0
def aP9_4 (ra rb : ℕ) : ℤ := -(1) * N9_4 ra rb + u9 (116 + rb) + u9 (133 + ra)
def MP9_4 : ℤ := CaseSplit.mxr2 (aP9_4) 12 16
def P9_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n9, (if c9_1 ra t && c9_3 rb t then 1 else 0)
def C9_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n9, (if c9_1 ra t && c9_3 rb t && c9_0 s t then 1 else 0)
def M9_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C9_5 ra rb) 10
def E9_5 : List ℕ := [17, 21, 51, 93, 127, 158, 192, 234]
def N9_5 (ra rb : ℕ) : ℤ := if E9_5.contains (ra * 19 + rb) = true then P9_5 ra rb - M9_5 ra rb else 0
def aP9_5 (ra rb : ℕ) : ℤ := -(1) * N9_5 ra rb + u9 (146 + rb) + u9 (165 + ra)
def MP9_5 : ℤ := CaseSplit.mxr2 (aP9_5) 12 18
def P9_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n9, (if c9_1 ra t && c9_4 rb t then 1 else 0)
def C9_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n9, (if c9_1 ra t && c9_4 rb t && c9_0 s t then 1 else 0)
def M9_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C9_6 ra rb) 10
def E9_6 : List ℕ := []
def N9_6 (ra rb : ℕ) : ℤ := if E9_6.contains (ra * 23 + rb) = true then P9_6 ra rb - M9_6 ra rb else 0
def aP9_6 (ra rb : ℕ) : ℤ := -(1) * N9_6 ra rb + u9 (178 + rb) + u9 (201 + ra)
def MP9_6 : ℤ := CaseSplit.mxr2 (aP9_6) 12 22
def N9_7 (_ra _rb : ℕ) : ℤ := 0
def aP9_7 (ra rb : ℕ) : ℤ := -(1) * N9_7 ra rb + u9 (214 + rb) + u9 (233 + ra)
def MP9_7 : ℤ := CaseSplit.mxr2 (aP9_7) 16 18
def N9_8 (_ra _rb : ℕ) : ℤ := 0
def aP9_8 (ra rb : ℕ) : ℤ := -(1) * N9_8 ra rb + u9 (250 + rb) + u9 (273 + ra)
def MP9_8 : ℤ := CaseSplit.mxr2 (aP9_8) 16 22
def N9_9 (_ra _rb : ℕ) : ℤ := 0
def aP9_9 (ra rb : ℕ) : ℤ := -(1) * N9_9 ra rb + u9 (290 + rb) + u9 (313 + ra)
def MP9_9 : ℤ := CaseSplit.mxr2 (aP9_9) 18 22

def rhs9 : ℤ := (∑ t ∈ Finset.range n9, w9 t) + 1 * (n9 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn9 : ∀ t, t < n9 → (0 : ℤ) ≤ w9 t := by decide
theorem plt9 : ∀ t, t < n9 → q9 t < 39 := by decide
theorem pfree9_5 : ∀ t, t < n9 → gb5 1 (q9 t) = false := by decide
theorem pfree9_7 : ∀ t, t < n9 → gb7 2 (q9 t) = false := by decide
theorem MSv9_0 : MS9_0 = 4 := by decide +kernel
theorem MSv9_1 : MS9_1 = 8 := by decide +kernel
theorem MSv9_2 : MS9_2 = 0 := by decide +kernel
theorem MSv9_3 : MS9_3 = 0 := by decide +kernel
theorem MSv9_4 : MS9_4 = 0 := by decide +kernel
theorem MPv9_0 : MP9_0 = 0 := by decide +kernel
theorem MPv9_1 : MP9_1 = 0 := by decide +kernel
theorem MPv9_2 : MP9_2 = 0 := by decide +kernel
theorem MPv9_3 : MP9_3 = 0 := by decide +kernel
theorem MPv9_4 : MP9_4 = 0 := by decide +kernel
theorem MPv9_5 : MP9_5 = 0 := by decide +kernel
theorem MPv9_6 : MP9_6 = 0 := by decide +kernel
theorem MPv9_7 : MP9_7 = 0 := by decide +kernel
theorem MPv9_8 : MP9_8 = 0 := by decide +kernel
theorem MPv9_9 : MP9_9 = 4 := by decide +kernel
theorem rhsv9 : rhs9 = 19 := by decide +kernel

/-- **The case-9 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 3/1.
    (Scaled by the common denominator 1: 16 < 19.) -/
theorem cert9 : MS9_0 + MS9_1 + MS9_2 + MS9_3 + MS9_4 + MP9_0 + MP9_1 + MP9_2 + MP9_3 + MP9_4 + MP9_5 + MP9_6 + MP9_7 + MP9_8 + MP9_9 < rhs9 := by
  rw [MSv9_0, MSv9_1, MSv9_2, MSv9_3, MSv9_4, MPv9_0, MPv9_1, MPv9_2, MPv9_3, MPv9_4, MPv9_5, MPv9_6, MPv9_7, MPv9_8, MPv9_9, rhsv9]
  decide

def Dg9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := (if c9_0 r0 t then 1 else 0) + (if c9_1 r1 t then 1 else 0) + (if c9_2 r2 t then 1 else 0) + (if c9_3 r3 t then 1 else 0) + (if c9_4 r4 t then 1 else 0)
def Wl9_0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c9_0 r0 t && c9_1 r1 t then 1 else 0
def Wl9_1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c9_0 r0 t && c9_2 r2 t then 1 else 0
def Wl9_2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c9_0 r0 t && c9_3 r3 t then 1 else 0
def Wl9_3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c9_0 r0 t && c9_4 r4 t then 1 else 0
def Wl9_4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c9_0 r0 t && c9_1 r1 t && c9_2 r2 t then 1 else 0
def Wl9_5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c9_0 r0 t && c9_1 r1 t && c9_3 r3 t then 1 else 0
def Wl9_6 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c9_0 r0 t && c9_1 r1 t && c9_4 r4 t then 1 else 0
def Wl9_7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c9_0 r0 t && !c9_1 r1 t && c9_2 r2 t && c9_3 r3 t then 1 else 0
def Wl9_8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c9_0 r0 t && !c9_1 r1 t && c9_2 r2 t && c9_4 r4 t then 1 else 0
def Wl9_9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c9_0 r0 t && !c9_1 r1 t && !c9_2 r2 t && c9_3 r3 t && c9_4 r4 t then 1 else 0

/-- **No configuration blocks the whole window in case 9.** -/
theorem nocov9 {r0 r1 r2 r3 r4 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23)
    (hcov : ∀ t, t < n9 → (c9_0 r0 t || c9_1 r1 t || c9_2 r2 t || c9_3 r3 t || c9_4 r4 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n9, (1 : ℤ) + (Wl9_0 r0 r1 r2 r3 r4 t + Wl9_1 r0 r1 r2 r3 r4 t + Wl9_2 r0 r1 r2 r3 r4 t + Wl9_3 r0 r1 r2 r3 r4 t + Wl9_4 r0 r1 r2 r3 r4 t + Wl9_5 r0 r1 r2 r3 r4 t + Wl9_6 r0 r1 r2 r3 r4 t + Wl9_7 r0 r1 r2 r3 r4 t + Wl9_8 r0 r1 r2 r3 r4 t + Wl9_9 r0 r1 r2 r3 r4 t) ≤ Dg9 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Wl9_0, Wl9_1, Wl9_2, Wl9_3, Wl9_4, Wl9_5, Wl9_6, Wl9_7, Wl9_8, Wl9_9, Dg9]
    exact CaseSplit.lowest5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n9, (1 : ℤ) ≤ Dg9 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Dg9]
    exact CaseSplit.degpos5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n9 : ℤ) + ((∑ t ∈ Finset.range n9, Wl9_0 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n9, Wl9_1 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n9, Wl9_2 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n9, Wl9_3 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n9, Wl9_4 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n9, Wl9_5 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n9, Wl9_6 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n9, Wl9_7 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n9, Wl9_8 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n9, Wl9_9 r0 r1 r2 r3 r4 t)) ≤ ∑ t ∈ Finset.range n9, Dg9 r0 r1 r2 r3 r4 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N9_0 r0 r1 ≤ ∑ t ∈ Finset.range n9, Wl9_0 r0 r1 r2 r3 r4 t := by
    simp only [N9_0, Wl9_0, le_refl]
  have hn1 : N9_1 r0 r2 ≤ ∑ t ∈ Finset.range n9, Wl9_1 r0 r1 r2 r3 r4 t := by
    simp only [N9_1, Wl9_1, le_refl]
  have hn2 : N9_2 r0 r3 ≤ ∑ t ∈ Finset.range n9, Wl9_2 r0 r1 r2 r3 r4 t := by
    simp only [N9_2, Wl9_2, le_refl]
  have hn3 : N9_3 r0 r4 ≤ ∑ t ∈ Finset.range n9, Wl9_3 r0 r1 r2 r3 r4 t := by
    simp only [N9_3, Wl9_3, le_refl]
  have hn4 : N9_4 r1 r2 ≤ ∑ t ∈ Finset.range n9, Wl9_4 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n9, Wl9_4 r0 r1 r2 r3 r4 t
        = (if c9_1 r1 t && c9_2 r2 t then (1:ℤ) else 0)
          - (if c9_1 r1 t && c9_2 r2 t && c9_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl9_4]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n9, Wl9_4 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl9_4]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n9, Wl9_4 r0 r1 r2 r3 r4 t
        = P9_4 r1 r2 - C9_4 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P9_4, C9_4]
    have hm : C9_4 r1 r2 r0 ≤ M9_4 r1 r2 :=
      CaseSplit.le_mxr (C9_4 r1 r2) 10 r0 (by omega)
    simp only [N9_4]
    split
    · rw [hL]; omega
    · exact hnn
  have hn5 : N9_5 r1 r3 ≤ ∑ t ∈ Finset.range n9, Wl9_5 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n9, Wl9_5 r0 r1 r2 r3 r4 t
        = (if c9_1 r1 t && c9_3 r3 t then (1:ℤ) else 0)
          - (if c9_1 r1 t && c9_3 r3 t && c9_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl9_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n9, Wl9_5 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl9_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n9, Wl9_5 r0 r1 r2 r3 r4 t
        = P9_5 r1 r3 - C9_5 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P9_5, C9_5]
    have hm : C9_5 r1 r3 r0 ≤ M9_5 r1 r3 :=
      CaseSplit.le_mxr (C9_5 r1 r3) 10 r0 (by omega)
    simp only [N9_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N9_6 r1 r4 ≤ ∑ t ∈ Finset.range n9, Wl9_6 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n9, Wl9_6 r0 r1 r2 r3 r4 t
        = (if c9_1 r1 t && c9_4 r4 t then (1:ℤ) else 0)
          - (if c9_1 r1 t && c9_4 r4 t && c9_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl9_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n9, Wl9_6 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl9_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n9, Wl9_6 r0 r1 r2 r3 r4 t
        = P9_6 r1 r4 - C9_6 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P9_6, C9_6]
    have hm : C9_6 r1 r4 r0 ≤ M9_6 r1 r4 :=
      CaseSplit.le_mxr (C9_6 r1 r4) 10 r0 (by omega)
    simp only [N9_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N9_7 r2 r3 ≤ ∑ t ∈ Finset.range n9, Wl9_7 r0 r1 r2 r3 r4 t := by
    simp only [N9_7]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl9_7]
    exact CaseSplit.ind_nonneg _
  have hn8 : N9_8 r2 r4 ≤ ∑ t ∈ Finset.range n9, Wl9_8 r0 r1 r2 r3 r4 t := by
    simp only [N9_8]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl9_8]
    exact CaseSplit.ind_nonneg _
  have hn9 : N9_9 r3 r4 ≤ ∑ t ∈ Finset.range n9, Wl9_9 r0 r1 r2 r3 r4 t := by
    simp only [N9_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl9_9]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n9, (w9 t + 1) * Dg9 r0 r1 r2 r3 r4 t = S9_0 r0 + S9_1 r1 + S9_2 r2 + S9_3 r3 + S9_4 r4 := by
    simp only [S9_0, S9_1, S9_2, S9_3, S9_4, Dg9, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n9, (w9 t + 1) * Dg9 r0 r1 r2 r3 r4 t
      = (∑ t ∈ Finset.range n9, w9 t * Dg9 r0 r1 r2 r3 r4 t)
        + 1 * (∑ t ∈ Finset.range n9, Dg9 r0 r1 r2 r3 r4 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n9, w9 t)
      ≤ ∑ t ∈ Finset.range n9, w9 t * Dg9 r0 r1 r2 r3 r4 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg9 r0 r1 r2 r3 r4 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w9 t := wnn9 t (Finset.mem_range.mp ht)
    calc w9 t = w9 t * 1 := (mul_one _).symm
      _ ≤ w9 t * Dg9 r0 r1 r2 r3 r4 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS9_0 r0 + aS9_1 r1 + aS9_2 r2 + aS9_3 r3 + aS9_4 r4) + (aP9_0 r0 r1 + aP9_1 r0 r2 + aP9_2 r0 r3 + aP9_3 r0 r4 + aP9_4 r1 r2 + aP9_5 r1 r3 + aP9_6 r1 r4 + aP9_7 r2 r3 + aP9_8 r2 r4 + aP9_9 r3 r4) = (S9_0 r0 + S9_1 r1 + S9_2 r2 + S9_3 r3 + S9_4 r4) - 1 * (N9_0 r0 r1 + N9_1 r0 r2 + N9_2 r0 r3 + N9_3 r0 r4 + N9_4 r1 r2 + N9_5 r1 r3 + N9_6 r1 r4 + N9_7 r2 r3 + N9_8 r2 r4 + N9_9 r3 r4) := by
    simp only [aS9_0, aS9_1, aS9_2, aS9_3, aS9_4, aP9_0, aP9_1, aP9_2, aP9_3, aP9_4, aP9_5, aP9_6, aP9_7, aP9_8, aP9_9, L9_0, L9_1, L9_2, L9_3, L9_4]
    ring
  have bS0 : aS9_0 r0 ≤ MS9_0 := CaseSplit.le_mxr (aS9_0) 10 r0 (by omega)
  have bS1 : aS9_1 r1 ≤ MS9_1 := CaseSplit.le_mxr (aS9_1) 12 r1 (by omega)
  have bS2 : aS9_2 r2 ≤ MS9_2 := CaseSplit.le_mxr (aS9_2) 16 r2 (by omega)
  have bS3 : aS9_3 r3 ≤ MS9_3 := CaseSplit.le_mxr (aS9_3) 18 r3 (by omega)
  have bS4 : aS9_4 r4 ≤ MS9_4 := CaseSplit.le_mxr (aS9_4) 22 r4 (by omega)
  have bP0 : aP9_0 r0 r1 ≤ MP9_0 := CaseSplit.le_mxr2 (aP9_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP9_1 r0 r2 ≤ MP9_1 := CaseSplit.le_mxr2 (aP9_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP9_2 r0 r3 ≤ MP9_2 := CaseSplit.le_mxr2 (aP9_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP9_3 r0 r4 ≤ MP9_3 := CaseSplit.le_mxr2 (aP9_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP9_4 r1 r2 ≤ MP9_4 := CaseSplit.le_mxr2 (aP9_4) 12 16 r1 r2 (by omega) (by omega)
  have bP5 : aP9_5 r1 r3 ≤ MP9_5 := CaseSplit.le_mxr2 (aP9_5) 12 18 r1 r3 (by omega) (by omega)
  have bP6 : aP9_6 r1 r4 ≤ MP9_6 := CaseSplit.le_mxr2 (aP9_6) 12 22 r1 r4 (by omega) (by omega)
  have bP7 : aP9_7 r2 r3 ≤ MP9_7 := CaseSplit.le_mxr2 (aP9_7) 16 18 r2 r3 (by omega) (by omega)
  have bP8 : aP9_8 r2 r4 ≤ MP9_8 := CaseSplit.le_mxr2 (aP9_8) 16 22 r2 r4 (by omega) (by omega)
  have bP9 : aP9_9 r3 r4 ≤ MP9_9 := CaseSplit.le_mxr2 (aP9_9) 18 22 r3 r4 (by omega) (by omega)
  have hrhs : rhs9 = (∑ t ∈ Finset.range n9, w9 t) + 1 * (n9 : ℤ) := rfl
  have hc := cert9
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, bS0, bS1, bS2, bS3, bS4, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9]

end IncCert23
