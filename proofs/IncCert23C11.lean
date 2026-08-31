/-
INCREMENT-WIDTH CERTIFICATE, step 19->23, case 11 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_19_23.json, which re-derives every number
from the primes alone).

Machine 23, INCREMENT width 39 = F_2(19) + s_min(23) = 31 + 8,
held gears [5, 7] at phases [1, 4].  Free gears [11, 13, 17, 19, 23].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 1.
-/
import IncCert23B

namespace IncCert23

/-! ### case 11: held gears at phases [1, 4] -/

def p11 : List ℕ := [1, 6, 7, 12, 14, 17, 19, 21, 22, 24, 26, 27, 29, 31, 34, 36]
def q11 (t : ℕ) : ℕ := p11.getD t 0
def n11 : ℕ := 16
def yl11 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def w11 (t : ℕ) : ℤ := yl11.getD t 0
def ul11 : List ℤ := [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, (-1), 0, (-1), (-1), 0, (-1), 0, (-1), 0, (-1), 0, (-1), (-1), 0, 0, (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), 0, 0, (-1), (-1), 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, (-1), (-1), 0, 0, (-1), (-1), 0, 0, (-1), (-1), 0, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 2, 2, (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-4), 0, 2, 2, 2, 2, 1, 0, 0, 2, 2, 2, 0, 2, 2, 1, 2, 2, 2, 2, (-2), (-2), (-2), (-2), (-3), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 0, 2, 2, 1, 2, 0, 1, 1, 2, 2, 1, 2, 2, 0, 2, 1, 2, 2, 0, 1, 1, 0, (-1), 1, 1, 1, 1, 0, 1, 1, 1, (-1), (-1), 1, 1, 0, (-1), 0]
def u11 (k : ℕ) : ℤ := ul11.getD k 0

def c11_0 (r t : ℕ) : Bool := gb11 r (q11 t)
def c11_1 (r t : ℕ) : Bool := gb13 r (q11 t)
def c11_2 (r t : ℕ) : Bool := gb17 r (q11 t)
def c11_3 (r t : ℕ) : Bool := gb19 r (q11 t)
def c11_4 (r t : ℕ) : Bool := gb23 r (q11 t)

def S11_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (w11 t + 1) * (if c11_0 r t then 1 else 0)
def S11_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (w11 t + 1) * (if c11_1 r t then 1 else 0)
def S11_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (w11 t + 1) * (if c11_2 r t then 1 else 0)
def S11_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (w11 t + 1) * (if c11_3 r t then 1 else 0)
def S11_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (w11 t + 1) * (if c11_4 r t then 1 else 0)

def L11_0 (r : ℕ) : ℤ := u11 (13 + r) + u11 (41 + r) + u11 (71 + r) + u11 (105 + r)
def L11_1 (r : ℕ) : ℤ := u11 (0 + r) + u11 (133 + r) + u11 (165 + r) + u11 (201 + r)
def L11_2 (r : ℕ) : ℤ := u11 (24 + r) + u11 (116 + r) + u11 (233 + r) + u11 (273 + r)
def L11_3 (r : ℕ) : ℤ := u11 (52 + r) + u11 (146 + r) + u11 (214 + r) + u11 (313 + r)
def L11_4 (r : ℕ) : ℤ := u11 (82 + r) + u11 (178 + r) + u11 (250 + r) + u11 (290 + r)

def aS11_0 (r : ℕ) : ℤ := S11_0 r - L11_0 r
def MS11_0 : ℤ := CaseSplit.mxr (aS11_0) 10
def aS11_1 (r : ℕ) : ℤ := S11_1 r - L11_1 r
def MS11_1 : ℤ := CaseSplit.mxr (aS11_1) 12
def aS11_2 (r : ℕ) : ℤ := S11_2 r - L11_2 r
def MS11_2 : ℤ := CaseSplit.mxr (aS11_2) 16
def aS11_3 (r : ℕ) : ℤ := S11_3 r - L11_3 r
def MS11_3 : ℤ := CaseSplit.mxr (aS11_3) 18
def aS11_4 (r : ℕ) : ℤ := S11_4 r - L11_4 r
def MS11_4 : ℤ := CaseSplit.mxr (aS11_4) 22

def N11_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (if c11_0 ra t && c11_1 rb t then 1 else 0)
def aP11_0 (ra rb : ℕ) : ℤ := -(1) * N11_0 ra rb + u11 (0 + rb) + u11 (13 + ra)
def MP11_0 : ℤ := CaseSplit.mxr2 (aP11_0) 10 12
def N11_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (if c11_0 ra t && c11_2 rb t then 1 else 0)
def aP11_1 (ra rb : ℕ) : ℤ := -(1) * N11_1 ra rb + u11 (24 + rb) + u11 (41 + ra)
def MP11_1 : ℤ := CaseSplit.mxr2 (aP11_1) 10 16
def N11_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (if c11_0 ra t && c11_3 rb t then 1 else 0)
def aP11_2 (ra rb : ℕ) : ℤ := -(1) * N11_2 ra rb + u11 (52 + rb) + u11 (71 + ra)
def MP11_2 : ℤ := CaseSplit.mxr2 (aP11_2) 10 18
def N11_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (if c11_0 ra t && c11_4 rb t then 1 else 0)
def aP11_3 (ra rb : ℕ) : ℤ := -(1) * N11_3 ra rb + u11 (82 + rb) + u11 (105 + ra)
def MP11_3 : ℤ := CaseSplit.mxr2 (aP11_3) 10 22
def P11_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (if c11_1 ra t && c11_2 rb t then 1 else 0)
def C11_4 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (if c11_1 ra t && c11_2 rb t && c11_0 s t then 1 else 0)
def M11_4 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C11_4 ra rb) 10
def E11_4 : List ℕ := [75, 81, 86, 97, 122, 133, 170, 176, 206, 212]
def N11_4 (ra rb : ℕ) : ℤ := if E11_4.contains (ra * 17 + rb) = true then P11_4 ra rb - M11_4 ra rb else 0
def aP11_4 (ra rb : ℕ) : ℤ := -(1) * N11_4 ra rb + u11 (116 + rb) + u11 (133 + ra)
def MP11_4 : ℤ := CaseSplit.mxr2 (aP11_4) 12 16
def P11_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (if c11_1 ra t && c11_3 rb t then 1 else 0)
def C11_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (if c11_1 ra t && c11_3 rb t && c11_0 s t then 1 else 0)
def M11_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C11_5 ra rb) 10
def E11_5 : List ℕ := [21, 27, 58, 111, 134, 187, 192, 198]
def N11_5 (ra rb : ℕ) : ℤ := if E11_5.contains (ra * 19 + rb) = true then P11_5 ra rb - M11_5 ra rb else 0
def aP11_5 (ra rb : ℕ) : ℤ := -(1) * N11_5 ra rb + u11 (146 + rb) + u11 (165 + ra)
def MP11_5 : ℤ := CaseSplit.mxr2 (aP11_5) 12 18
def P11_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (if c11_1 ra t && c11_4 rb t then 1 else 0)
def C11_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n11, (if c11_1 ra t && c11_4 rb t && c11_0 s t then 1 else 0)
def M11_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C11_6 ra rb) 10
def E11_6 : List ℕ := []
def N11_6 (ra rb : ℕ) : ℤ := if E11_6.contains (ra * 23 + rb) = true then P11_6 ra rb - M11_6 ra rb else 0
def aP11_6 (ra rb : ℕ) : ℤ := -(1) * N11_6 ra rb + u11 (178 + rb) + u11 (201 + ra)
def MP11_6 : ℤ := CaseSplit.mxr2 (aP11_6) 12 22
def N11_7 (_ra _rb : ℕ) : ℤ := 0
def aP11_7 (ra rb : ℕ) : ℤ := -(1) * N11_7 ra rb + u11 (214 + rb) + u11 (233 + ra)
def MP11_7 : ℤ := CaseSplit.mxr2 (aP11_7) 16 18
def N11_8 (_ra _rb : ℕ) : ℤ := 0
def aP11_8 (ra rb : ℕ) : ℤ := -(1) * N11_8 ra rb + u11 (250 + rb) + u11 (273 + ra)
def MP11_8 : ℤ := CaseSplit.mxr2 (aP11_8) 16 22
def N11_9 (_ra _rb : ℕ) : ℤ := 0
def aP11_9 (ra rb : ℕ) : ℤ := -(1) * N11_9 ra rb + u11 (290 + rb) + u11 (313 + ra)
def MP11_9 : ℤ := CaseSplit.mxr2 (aP11_9) 18 22

def rhs11 : ℤ := (∑ t ∈ Finset.range n11, w11 t) + 1 * (n11 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn11 : ∀ t, t < n11 → (0 : ℤ) ≤ w11 t := by decide
theorem plt11 : ∀ t, t < n11 → q11 t < 39 := by decide
theorem pfree11_5 : ∀ t, t < n11 → gb5 1 (q11 t) = false := by decide
theorem pfree11_7 : ∀ t, t < n11 → gb7 4 (q11 t) = false := by decide
theorem MSv11_0 : MS11_0 = 4 := by decide +kernel
theorem MSv11_1 : MS11_1 = 8 := by decide +kernel
theorem MSv11_2 : MS11_2 = 0 := by decide +kernel
theorem MSv11_3 : MS11_3 = 0 := by decide +kernel
theorem MSv11_4 : MS11_4 = 0 := by decide +kernel
theorem MPv11_0 : MP11_0 = 0 := by decide +kernel
theorem MPv11_1 : MP11_1 = 0 := by decide +kernel
theorem MPv11_2 : MP11_2 = 0 := by decide +kernel
theorem MPv11_3 : MP11_3 = 0 := by decide +kernel
theorem MPv11_4 : MP11_4 = 0 := by decide +kernel
theorem MPv11_5 : MP11_5 = 0 := by decide +kernel
theorem MPv11_6 : MP11_6 = 0 := by decide +kernel
theorem MPv11_7 : MP11_7 = 0 := by decide +kernel
theorem MPv11_8 : MP11_8 = 0 := by decide +kernel
theorem MPv11_9 : MP11_9 = 3 := by decide +kernel
theorem rhsv11 : rhs11 = 16 := by decide +kernel

/-- **The case-11 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/1.
    (Scaled by the common denominator 1: 15 < 16.) -/
theorem cert11 : MS11_0 + MS11_1 + MS11_2 + MS11_3 + MS11_4 + MP11_0 + MP11_1 + MP11_2 + MP11_3 + MP11_4 + MP11_5 + MP11_6 + MP11_7 + MP11_8 + MP11_9 < rhs11 := by
  rw [MSv11_0, MSv11_1, MSv11_2, MSv11_3, MSv11_4, MPv11_0, MPv11_1, MPv11_2, MPv11_3, MPv11_4, MPv11_5, MPv11_6, MPv11_7, MPv11_8, MPv11_9, rhsv11]
  decide

def Dg11 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := (if c11_0 r0 t then 1 else 0) + (if c11_1 r1 t then 1 else 0) + (if c11_2 r2 t then 1 else 0) + (if c11_3 r3 t then 1 else 0) + (if c11_4 r4 t then 1 else 0)
def Wl11_0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c11_0 r0 t && c11_1 r1 t then 1 else 0
def Wl11_1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c11_0 r0 t && c11_2 r2 t then 1 else 0
def Wl11_2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c11_0 r0 t && c11_3 r3 t then 1 else 0
def Wl11_3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c11_0 r0 t && c11_4 r4 t then 1 else 0
def Wl11_4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c11_0 r0 t && c11_1 r1 t && c11_2 r2 t then 1 else 0
def Wl11_5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c11_0 r0 t && c11_1 r1 t && c11_3 r3 t then 1 else 0
def Wl11_6 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c11_0 r0 t && c11_1 r1 t && c11_4 r4 t then 1 else 0
def Wl11_7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c11_0 r0 t && !c11_1 r1 t && c11_2 r2 t && c11_3 r3 t then 1 else 0
def Wl11_8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c11_0 r0 t && !c11_1 r1 t && c11_2 r2 t && c11_4 r4 t then 1 else 0
def Wl11_9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c11_0 r0 t && !c11_1 r1 t && !c11_2 r2 t && c11_3 r3 t && c11_4 r4 t then 1 else 0

/-- **No configuration blocks the whole window in case 11.** -/
theorem nocov11 {r0 r1 r2 r3 r4 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23)
    (hcov : ∀ t, t < n11 → (c11_0 r0 t || c11_1 r1 t || c11_2 r2 t || c11_3 r3 t || c11_4 r4 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n11, (1 : ℤ) + (Wl11_0 r0 r1 r2 r3 r4 t + Wl11_1 r0 r1 r2 r3 r4 t + Wl11_2 r0 r1 r2 r3 r4 t + Wl11_3 r0 r1 r2 r3 r4 t + Wl11_4 r0 r1 r2 r3 r4 t + Wl11_5 r0 r1 r2 r3 r4 t + Wl11_6 r0 r1 r2 r3 r4 t + Wl11_7 r0 r1 r2 r3 r4 t + Wl11_8 r0 r1 r2 r3 r4 t + Wl11_9 r0 r1 r2 r3 r4 t) ≤ Dg11 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Wl11_0, Wl11_1, Wl11_2, Wl11_3, Wl11_4, Wl11_5, Wl11_6, Wl11_7, Wl11_8, Wl11_9, Dg11]
    exact CaseSplit.lowest5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n11, (1 : ℤ) ≤ Dg11 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Dg11]
    exact CaseSplit.degpos5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n11 : ℤ) + ((∑ t ∈ Finset.range n11, Wl11_0 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n11, Wl11_1 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n11, Wl11_2 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n11, Wl11_3 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n11, Wl11_4 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n11, Wl11_5 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n11, Wl11_6 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n11, Wl11_7 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n11, Wl11_8 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n11, Wl11_9 r0 r1 r2 r3 r4 t)) ≤ ∑ t ∈ Finset.range n11, Dg11 r0 r1 r2 r3 r4 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N11_0 r0 r1 ≤ ∑ t ∈ Finset.range n11, Wl11_0 r0 r1 r2 r3 r4 t := by
    simp only [N11_0, Wl11_0, le_refl]
  have hn1 : N11_1 r0 r2 ≤ ∑ t ∈ Finset.range n11, Wl11_1 r0 r1 r2 r3 r4 t := by
    simp only [N11_1, Wl11_1, le_refl]
  have hn2 : N11_2 r0 r3 ≤ ∑ t ∈ Finset.range n11, Wl11_2 r0 r1 r2 r3 r4 t := by
    simp only [N11_2, Wl11_2, le_refl]
  have hn3 : N11_3 r0 r4 ≤ ∑ t ∈ Finset.range n11, Wl11_3 r0 r1 r2 r3 r4 t := by
    simp only [N11_3, Wl11_3, le_refl]
  have hn4 : N11_4 r1 r2 ≤ ∑ t ∈ Finset.range n11, Wl11_4 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n11, Wl11_4 r0 r1 r2 r3 r4 t
        = (if c11_1 r1 t && c11_2 r2 t then (1:ℤ) else 0)
          - (if c11_1 r1 t && c11_2 r2 t && c11_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl11_4]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n11, Wl11_4 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl11_4]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n11, Wl11_4 r0 r1 r2 r3 r4 t
        = P11_4 r1 r2 - C11_4 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P11_4, C11_4]
    have hm : C11_4 r1 r2 r0 ≤ M11_4 r1 r2 :=
      CaseSplit.le_mxr (C11_4 r1 r2) 10 r0 (by omega)
    simp only [N11_4]
    split
    · rw [hL]; omega
    · exact hnn
  have hn5 : N11_5 r1 r3 ≤ ∑ t ∈ Finset.range n11, Wl11_5 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n11, Wl11_5 r0 r1 r2 r3 r4 t
        = (if c11_1 r1 t && c11_3 r3 t then (1:ℤ) else 0)
          - (if c11_1 r1 t && c11_3 r3 t && c11_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl11_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n11, Wl11_5 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl11_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n11, Wl11_5 r0 r1 r2 r3 r4 t
        = P11_5 r1 r3 - C11_5 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P11_5, C11_5]
    have hm : C11_5 r1 r3 r0 ≤ M11_5 r1 r3 :=
      CaseSplit.le_mxr (C11_5 r1 r3) 10 r0 (by omega)
    simp only [N11_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N11_6 r1 r4 ≤ ∑ t ∈ Finset.range n11, Wl11_6 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n11, Wl11_6 r0 r1 r2 r3 r4 t
        = (if c11_1 r1 t && c11_4 r4 t then (1:ℤ) else 0)
          - (if c11_1 r1 t && c11_4 r4 t && c11_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl11_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n11, Wl11_6 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl11_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n11, Wl11_6 r0 r1 r2 r3 r4 t
        = P11_6 r1 r4 - C11_6 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P11_6, C11_6]
    have hm : C11_6 r1 r4 r0 ≤ M11_6 r1 r4 :=
      CaseSplit.le_mxr (C11_6 r1 r4) 10 r0 (by omega)
    simp only [N11_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N11_7 r2 r3 ≤ ∑ t ∈ Finset.range n11, Wl11_7 r0 r1 r2 r3 r4 t := by
    simp only [N11_7]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl11_7]
    exact CaseSplit.ind_nonneg _
  have hn8 : N11_8 r2 r4 ≤ ∑ t ∈ Finset.range n11, Wl11_8 r0 r1 r2 r3 r4 t := by
    simp only [N11_8]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl11_8]
    exact CaseSplit.ind_nonneg _
  have hn9 : N11_9 r3 r4 ≤ ∑ t ∈ Finset.range n11, Wl11_9 r0 r1 r2 r3 r4 t := by
    simp only [N11_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl11_9]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n11, (w11 t + 1) * Dg11 r0 r1 r2 r3 r4 t = S11_0 r0 + S11_1 r1 + S11_2 r2 + S11_3 r3 + S11_4 r4 := by
    simp only [S11_0, S11_1, S11_2, S11_3, S11_4, Dg11, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n11, (w11 t + 1) * Dg11 r0 r1 r2 r3 r4 t
      = (∑ t ∈ Finset.range n11, w11 t * Dg11 r0 r1 r2 r3 r4 t)
        + 1 * (∑ t ∈ Finset.range n11, Dg11 r0 r1 r2 r3 r4 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n11, w11 t)
      ≤ ∑ t ∈ Finset.range n11, w11 t * Dg11 r0 r1 r2 r3 r4 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg11 r0 r1 r2 r3 r4 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w11 t := wnn11 t (Finset.mem_range.mp ht)
    calc w11 t = w11 t * 1 := (mul_one _).symm
      _ ≤ w11 t * Dg11 r0 r1 r2 r3 r4 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS11_0 r0 + aS11_1 r1 + aS11_2 r2 + aS11_3 r3 + aS11_4 r4) + (aP11_0 r0 r1 + aP11_1 r0 r2 + aP11_2 r0 r3 + aP11_3 r0 r4 + aP11_4 r1 r2 + aP11_5 r1 r3 + aP11_6 r1 r4 + aP11_7 r2 r3 + aP11_8 r2 r4 + aP11_9 r3 r4) = (S11_0 r0 + S11_1 r1 + S11_2 r2 + S11_3 r3 + S11_4 r4) - 1 * (N11_0 r0 r1 + N11_1 r0 r2 + N11_2 r0 r3 + N11_3 r0 r4 + N11_4 r1 r2 + N11_5 r1 r3 + N11_6 r1 r4 + N11_7 r2 r3 + N11_8 r2 r4 + N11_9 r3 r4) := by
    simp only [aS11_0, aS11_1, aS11_2, aS11_3, aS11_4, aP11_0, aP11_1, aP11_2, aP11_3, aP11_4, aP11_5, aP11_6, aP11_7, aP11_8, aP11_9, L11_0, L11_1, L11_2, L11_3, L11_4]
    ring
  have bS0 : aS11_0 r0 ≤ MS11_0 := CaseSplit.le_mxr (aS11_0) 10 r0 (by omega)
  have bS1 : aS11_1 r1 ≤ MS11_1 := CaseSplit.le_mxr (aS11_1) 12 r1 (by omega)
  have bS2 : aS11_2 r2 ≤ MS11_2 := CaseSplit.le_mxr (aS11_2) 16 r2 (by omega)
  have bS3 : aS11_3 r3 ≤ MS11_3 := CaseSplit.le_mxr (aS11_3) 18 r3 (by omega)
  have bS4 : aS11_4 r4 ≤ MS11_4 := CaseSplit.le_mxr (aS11_4) 22 r4 (by omega)
  have bP0 : aP11_0 r0 r1 ≤ MP11_0 := CaseSplit.le_mxr2 (aP11_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP11_1 r0 r2 ≤ MP11_1 := CaseSplit.le_mxr2 (aP11_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP11_2 r0 r3 ≤ MP11_2 := CaseSplit.le_mxr2 (aP11_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP11_3 r0 r4 ≤ MP11_3 := CaseSplit.le_mxr2 (aP11_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP11_4 r1 r2 ≤ MP11_4 := CaseSplit.le_mxr2 (aP11_4) 12 16 r1 r2 (by omega) (by omega)
  have bP5 : aP11_5 r1 r3 ≤ MP11_5 := CaseSplit.le_mxr2 (aP11_5) 12 18 r1 r3 (by omega) (by omega)
  have bP6 : aP11_6 r1 r4 ≤ MP11_6 := CaseSplit.le_mxr2 (aP11_6) 12 22 r1 r4 (by omega) (by omega)
  have bP7 : aP11_7 r2 r3 ≤ MP11_7 := CaseSplit.le_mxr2 (aP11_7) 16 18 r2 r3 (by omega) (by omega)
  have bP8 : aP11_8 r2 r4 ≤ MP11_8 := CaseSplit.le_mxr2 (aP11_8) 16 22 r2 r4 (by omega) (by omega)
  have bP9 : aP11_9 r3 r4 ≤ MP11_9 := CaseSplit.le_mxr2 (aP11_9) 18 22 r3 r4 (by omega) (by omega)
  have hrhs : rhs11 = (∑ t ∈ Finset.range n11, w11 t) + 1 * (n11 : ℤ) := rfl
  have hc := cert11
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, bS0, bS1, bS2, bS3, bS4, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9]

end IncCert23
