/-
INCREMENT-WIDTH CERTIFICATE, step 19->23, case 12 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_19_23.json, which re-derives every number
from the primes alone).

Machine 23, INCREMENT width 39 = F_2(19) + s_min(23) = 31 + 8,
held gears [5, 7] at phases [1, 5].  Free gears [11, 13, 17, 19, 23].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 3.
-/
import IncCert23B

namespace IncCert23

/-! ### case 12: held gears at phases [1, 5] -/

def p12 : List ℕ := [2, 4, 6, 7, 9, 11, 12, 14, 16, 19, 21, 26, 27, 32, 34, 37]
def q12 (t : ℕ) : ℕ := p12.getD t 0
def n12 : ℕ := 16
def yl12 : List ℤ := [0, 2, 0, 0, 2, 3, 2, 3, 2, 1, 1, 0, 0, 0, 0, 0]
def w12 (t : ℕ) : ℤ := yl12.getD t 0
def ul12 : List ℤ := [0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, (-2), 0, 0, 0, 0, (-2), 0, 0, 0, 0, 0, 0, 0, 0, (-2), (-2), 0, (-2), 0, (-2), (-2), (-2), 0, 0, 0, (-2), (-2), (-2), (-2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, (-1), (-1), 0, (-1), 0, 0, 1, (-1), 0, (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), 0, (-1), 1, (-1), 0, 0, (-1), (-2), (-2), (-2), (-1), (-2), (-2), 0, (-2), (-2), (-2), (-2), (-1), (-2), 0, 0, (-2), (-2), (-2), (-2), (-2), (-2), 0, 0, 0, 0, 0, 2, 0, 0, 1, 2, 0, 6, 6, 5, 7, 7, 7, 7, 3, 7, 4, 7, 7, 6, 3, 7, 7, 7, (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 4, (-4), (-5), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), (-4), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 2, 3, 5, 2, 8, 4, 5, 8, 2, 7, 5, 3, 8, 2, 8, 8, 3, 7, 2, 3, 5, 2, 1, 1, 1, 1, 1, 1, (-1), 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, (-2), 0]
def u12 (k : ℕ) : ℤ := ul12.getD k 0

def c12_0 (r t : ℕ) : Bool := gb11 r (q12 t)
def c12_1 (r t : ℕ) : Bool := gb13 r (q12 t)
def c12_2 (r t : ℕ) : Bool := gb17 r (q12 t)
def c12_3 (r t : ℕ) : Bool := gb19 r (q12 t)
def c12_4 (r t : ℕ) : Bool := gb23 r (q12 t)

def S12_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (w12 t + 2) * (if c12_0 r t then 1 else 0)
def S12_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (w12 t + 2) * (if c12_1 r t then 1 else 0)
def S12_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (w12 t + 2) * (if c12_2 r t then 1 else 0)
def S12_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (w12 t + 2) * (if c12_3 r t then 1 else 0)
def S12_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (w12 t + 2) * (if c12_4 r t then 1 else 0)

def L12_0 (r : ℕ) : ℤ := u12 (13 + r) + u12 (41 + r) + u12 (71 + r) + u12 (105 + r)
def L12_1 (r : ℕ) : ℤ := u12 (0 + r) + u12 (133 + r) + u12 (165 + r) + u12 (201 + r)
def L12_2 (r : ℕ) : ℤ := u12 (24 + r) + u12 (116 + r) + u12 (233 + r) + u12 (273 + r)
def L12_3 (r : ℕ) : ℤ := u12 (52 + r) + u12 (146 + r) + u12 (214 + r) + u12 (313 + r)
def L12_4 (r : ℕ) : ℤ := u12 (82 + r) + u12 (178 + r) + u12 (250 + r) + u12 (290 + r)

def aS12_0 (r : ℕ) : ℤ := S12_0 r - L12_0 r
def MS12_0 : ℤ := CaseSplit.mxr (aS12_0) 10
def aS12_1 (r : ℕ) : ℤ := S12_1 r - L12_1 r
def MS12_1 : ℤ := CaseSplit.mxr (aS12_1) 12
def aS12_2 (r : ℕ) : ℤ := S12_2 r - L12_2 r
def MS12_2 : ℤ := CaseSplit.mxr (aS12_2) 16
def aS12_3 (r : ℕ) : ℤ := S12_3 r - L12_3 r
def MS12_3 : ℤ := CaseSplit.mxr (aS12_3) 18
def aS12_4 (r : ℕ) : ℤ := S12_4 r - L12_4 r
def MS12_4 : ℤ := CaseSplit.mxr (aS12_4) 22

def N12_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (if c12_0 ra t && c12_1 rb t then 1 else 0)
def aP12_0 (ra rb : ℕ) : ℤ := -(2) * N12_0 ra rb + u12 (0 + rb) + u12 (13 + ra)
def MP12_0 : ℤ := CaseSplit.mxr2 (aP12_0) 10 12
def N12_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (if c12_0 ra t && c12_2 rb t then 1 else 0)
def aP12_1 (ra rb : ℕ) : ℤ := -(2) * N12_1 ra rb + u12 (24 + rb) + u12 (41 + ra)
def MP12_1 : ℤ := CaseSplit.mxr2 (aP12_1) 10 16
def N12_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (if c12_0 ra t && c12_3 rb t then 1 else 0)
def aP12_2 (ra rb : ℕ) : ℤ := -(2) * N12_2 ra rb + u12 (52 + rb) + u12 (71 + ra)
def MP12_2 : ℤ := CaseSplit.mxr2 (aP12_2) 10 18
def N12_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (if c12_0 ra t && c12_4 rb t then 1 else 0)
def aP12_3 (ra rb : ℕ) : ℤ := -(2) * N12_3 ra rb + u12 (82 + rb) + u12 (105 + ra)
def MP12_3 : ℤ := CaseSplit.mxr2 (aP12_3) 10 22
def P12_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (if c12_1 ra t && c12_2 rb t then 1 else 0)
def C12_4 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (if c12_1 ra t && c12_2 rb t && c12_0 s t then 1 else 0)
def M12_4 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C12_4 ra rb) 10
def E12_4 : List ℕ := [39, 45, 129, 135, 154, 165]
def N12_4 (ra rb : ℕ) : ℤ := if E12_4.contains (ra * 17 + rb) = true then P12_4 ra rb - M12_4 ra rb else 0
def aP12_4 (ra rb : ℕ) : ℤ := -(2) * N12_4 ra rb + u12 (116 + rb) + u12 (133 + ra)
def MP12_4 : ℤ := CaseSplit.mxr2 (aP12_4) 12 16
def P12_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (if c12_1 ra t && c12_3 rb t then 1 else 0)
def C12_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (if c12_1 ra t && c12_3 rb t && c12_0 s t then 1 else 0)
def M12_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C12_5 ra rb) 10
def E12_5 : List ℕ := [27, 58, 98, 111, 134, 174, 187, 198]
def N12_5 (ra rb : ℕ) : ℤ := if E12_5.contains (ra * 19 + rb) = true then P12_5 ra rb - M12_5 ra rb else 0
def aP12_5 (ra rb : ℕ) : ℤ := -(2) * N12_5 ra rb + u12 (146 + rb) + u12 (165 + ra)
def MP12_5 : ℤ := CaseSplit.mxr2 (aP12_5) 12 18
def P12_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (if c12_1 ra t && c12_4 rb t then 1 else 0)
def C12_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n12, (if c12_1 ra t && c12_4 rb t && c12_0 s t then 1 else 0)
def M12_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C12_6 ra rb) 10
def E12_6 : List ℕ := []
def N12_6 (ra rb : ℕ) : ℤ := if E12_6.contains (ra * 23 + rb) = true then P12_6 ra rb - M12_6 ra rb else 0
def aP12_6 (ra rb : ℕ) : ℤ := -(2) * N12_6 ra rb + u12 (178 + rb) + u12 (201 + ra)
def MP12_6 : ℤ := CaseSplit.mxr2 (aP12_6) 12 22
def N12_7 (_ra _rb : ℕ) : ℤ := 0
def aP12_7 (ra rb : ℕ) : ℤ := -(2) * N12_7 ra rb + u12 (214 + rb) + u12 (233 + ra)
def MP12_7 : ℤ := CaseSplit.mxr2 (aP12_7) 16 18
def N12_8 (_ra _rb : ℕ) : ℤ := 0
def aP12_8 (ra rb : ℕ) : ℤ := -(2) * N12_8 ra rb + u12 (250 + rb) + u12 (273 + ra)
def MP12_8 : ℤ := CaseSplit.mxr2 (aP12_8) 16 22
def N12_9 (_ra _rb : ℕ) : ℤ := 0
def aP12_9 (ra rb : ℕ) : ℤ := -(2) * N12_9 ra rb + u12 (290 + rb) + u12 (313 + ra)
def MP12_9 : ℤ := CaseSplit.mxr2 (aP12_9) 18 22

def rhs12 : ℤ := (∑ t ∈ Finset.range n12, w12 t) + 2 * (n12 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn12 : ∀ t, t < n12 → (0 : ℤ) ≤ w12 t := by decide
theorem plt12 : ∀ t, t < n12 → q12 t < 39 := by decide
theorem pfree12_5 : ∀ t, t < n12 → gb5 1 (q12 t) = false := by decide
theorem pfree12_7 : ∀ t, t < n12 → gb7 5 (q12 t) = false := by decide
theorem MSv12_0 : MS12_0 = 10 := by decide +kernel
theorem MSv12_1 : MS12_1 = 20 := by decide +kernel
theorem MSv12_2 : MS12_2 = 2 := by decide +kernel
theorem MSv12_3 : MS12_3 = 2 := by decide +kernel
theorem MSv12_4 : MS12_4 = 2 := by decide +kernel
theorem MPv12_0 : MP12_0 = 0 := by decide +kernel
theorem MPv12_1 : MP12_1 = 0 := by decide +kernel
theorem MPv12_2 : MP12_2 = 0 := by decide +kernel
theorem MPv12_3 : MP12_3 = 0 := by decide +kernel
theorem MPv12_4 : MP12_4 = 0 := by decide +kernel
theorem MPv12_5 : MP12_5 = 0 := by decide +kernel
theorem MPv12_6 : MP12_6 = 0 := by decide +kernel
theorem MPv12_7 : MP12_7 = 0 := by decide +kernel
theorem MPv12_8 : MP12_8 = 0 := by decide +kernel
theorem MPv12_9 : MP12_9 = 9 := by decide +kernel
theorem rhsv12 : rhs12 = 48 := by decide +kernel

/-- **The case-12 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 3/3.
    (Scaled by the common denominator 3: 45 < 48.) -/
theorem cert12 : MS12_0 + MS12_1 + MS12_2 + MS12_3 + MS12_4 + MP12_0 + MP12_1 + MP12_2 + MP12_3 + MP12_4 + MP12_5 + MP12_6 + MP12_7 + MP12_8 + MP12_9 < rhs12 := by
  rw [MSv12_0, MSv12_1, MSv12_2, MSv12_3, MSv12_4, MPv12_0, MPv12_1, MPv12_2, MPv12_3, MPv12_4, MPv12_5, MPv12_6, MPv12_7, MPv12_8, MPv12_9, rhsv12]
  decide

def Dg12 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := (if c12_0 r0 t then 1 else 0) + (if c12_1 r1 t then 1 else 0) + (if c12_2 r2 t then 1 else 0) + (if c12_3 r3 t then 1 else 0) + (if c12_4 r4 t then 1 else 0)
def Wl12_0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c12_0 r0 t && c12_1 r1 t then 1 else 0
def Wl12_1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c12_0 r0 t && c12_2 r2 t then 1 else 0
def Wl12_2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c12_0 r0 t && c12_3 r3 t then 1 else 0
def Wl12_3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c12_0 r0 t && c12_4 r4 t then 1 else 0
def Wl12_4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c12_0 r0 t && c12_1 r1 t && c12_2 r2 t then 1 else 0
def Wl12_5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c12_0 r0 t && c12_1 r1 t && c12_3 r3 t then 1 else 0
def Wl12_6 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c12_0 r0 t && c12_1 r1 t && c12_4 r4 t then 1 else 0
def Wl12_7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c12_0 r0 t && !c12_1 r1 t && c12_2 r2 t && c12_3 r3 t then 1 else 0
def Wl12_8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c12_0 r0 t && !c12_1 r1 t && c12_2 r2 t && c12_4 r4 t then 1 else 0
def Wl12_9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c12_0 r0 t && !c12_1 r1 t && !c12_2 r2 t && c12_3 r3 t && c12_4 r4 t then 1 else 0

/-- **No configuration blocks the whole window in case 12.** -/
theorem nocov12 {r0 r1 r2 r3 r4 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23)
    (hcov : ∀ t, t < n12 → (c12_0 r0 t || c12_1 r1 t || c12_2 r2 t || c12_3 r3 t || c12_4 r4 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n12, (1 : ℤ) + (Wl12_0 r0 r1 r2 r3 r4 t + Wl12_1 r0 r1 r2 r3 r4 t + Wl12_2 r0 r1 r2 r3 r4 t + Wl12_3 r0 r1 r2 r3 r4 t + Wl12_4 r0 r1 r2 r3 r4 t + Wl12_5 r0 r1 r2 r3 r4 t + Wl12_6 r0 r1 r2 r3 r4 t + Wl12_7 r0 r1 r2 r3 r4 t + Wl12_8 r0 r1 r2 r3 r4 t + Wl12_9 r0 r1 r2 r3 r4 t) ≤ Dg12 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Wl12_0, Wl12_1, Wl12_2, Wl12_3, Wl12_4, Wl12_5, Wl12_6, Wl12_7, Wl12_8, Wl12_9, Dg12]
    exact CaseSplit.lowest5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n12, (1 : ℤ) ≤ Dg12 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Dg12]
    exact CaseSplit.degpos5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n12 : ℤ) + ((∑ t ∈ Finset.range n12, Wl12_0 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n12, Wl12_1 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n12, Wl12_2 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n12, Wl12_3 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n12, Wl12_4 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n12, Wl12_5 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n12, Wl12_6 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n12, Wl12_7 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n12, Wl12_8 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n12, Wl12_9 r0 r1 r2 r3 r4 t)) ≤ ∑ t ∈ Finset.range n12, Dg12 r0 r1 r2 r3 r4 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N12_0 r0 r1 ≤ ∑ t ∈ Finset.range n12, Wl12_0 r0 r1 r2 r3 r4 t := by
    simp only [N12_0, Wl12_0, le_refl]
  have hn1 : N12_1 r0 r2 ≤ ∑ t ∈ Finset.range n12, Wl12_1 r0 r1 r2 r3 r4 t := by
    simp only [N12_1, Wl12_1, le_refl]
  have hn2 : N12_2 r0 r3 ≤ ∑ t ∈ Finset.range n12, Wl12_2 r0 r1 r2 r3 r4 t := by
    simp only [N12_2, Wl12_2, le_refl]
  have hn3 : N12_3 r0 r4 ≤ ∑ t ∈ Finset.range n12, Wl12_3 r0 r1 r2 r3 r4 t := by
    simp only [N12_3, Wl12_3, le_refl]
  have hn4 : N12_4 r1 r2 ≤ ∑ t ∈ Finset.range n12, Wl12_4 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n12, Wl12_4 r0 r1 r2 r3 r4 t
        = (if c12_1 r1 t && c12_2 r2 t then (1:ℤ) else 0)
          - (if c12_1 r1 t && c12_2 r2 t && c12_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl12_4]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n12, Wl12_4 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl12_4]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n12, Wl12_4 r0 r1 r2 r3 r4 t
        = P12_4 r1 r2 - C12_4 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P12_4, C12_4]
    have hm : C12_4 r1 r2 r0 ≤ M12_4 r1 r2 :=
      CaseSplit.le_mxr (C12_4 r1 r2) 10 r0 (by omega)
    simp only [N12_4]
    split
    · rw [hL]; omega
    · exact hnn
  have hn5 : N12_5 r1 r3 ≤ ∑ t ∈ Finset.range n12, Wl12_5 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n12, Wl12_5 r0 r1 r2 r3 r4 t
        = (if c12_1 r1 t && c12_3 r3 t then (1:ℤ) else 0)
          - (if c12_1 r1 t && c12_3 r3 t && c12_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl12_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n12, Wl12_5 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl12_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n12, Wl12_5 r0 r1 r2 r3 r4 t
        = P12_5 r1 r3 - C12_5 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P12_5, C12_5]
    have hm : C12_5 r1 r3 r0 ≤ M12_5 r1 r3 :=
      CaseSplit.le_mxr (C12_5 r1 r3) 10 r0 (by omega)
    simp only [N12_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N12_6 r1 r4 ≤ ∑ t ∈ Finset.range n12, Wl12_6 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n12, Wl12_6 r0 r1 r2 r3 r4 t
        = (if c12_1 r1 t && c12_4 r4 t then (1:ℤ) else 0)
          - (if c12_1 r1 t && c12_4 r4 t && c12_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl12_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n12, Wl12_6 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl12_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n12, Wl12_6 r0 r1 r2 r3 r4 t
        = P12_6 r1 r4 - C12_6 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P12_6, C12_6]
    have hm : C12_6 r1 r4 r0 ≤ M12_6 r1 r4 :=
      CaseSplit.le_mxr (C12_6 r1 r4) 10 r0 (by omega)
    simp only [N12_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N12_7 r2 r3 ≤ ∑ t ∈ Finset.range n12, Wl12_7 r0 r1 r2 r3 r4 t := by
    simp only [N12_7]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl12_7]
    exact CaseSplit.ind_nonneg _
  have hn8 : N12_8 r2 r4 ≤ ∑ t ∈ Finset.range n12, Wl12_8 r0 r1 r2 r3 r4 t := by
    simp only [N12_8]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl12_8]
    exact CaseSplit.ind_nonneg _
  have hn9 : N12_9 r3 r4 ≤ ∑ t ∈ Finset.range n12, Wl12_9 r0 r1 r2 r3 r4 t := by
    simp only [N12_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl12_9]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n12, (w12 t + 2) * Dg12 r0 r1 r2 r3 r4 t = S12_0 r0 + S12_1 r1 + S12_2 r2 + S12_3 r3 + S12_4 r4 := by
    simp only [S12_0, S12_1, S12_2, S12_3, S12_4, Dg12, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n12, (w12 t + 2) * Dg12 r0 r1 r2 r3 r4 t
      = (∑ t ∈ Finset.range n12, w12 t * Dg12 r0 r1 r2 r3 r4 t)
        + 2 * (∑ t ∈ Finset.range n12, Dg12 r0 r1 r2 r3 r4 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n12, w12 t)
      ≤ ∑ t ∈ Finset.range n12, w12 t * Dg12 r0 r1 r2 r3 r4 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg12 r0 r1 r2 r3 r4 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w12 t := wnn12 t (Finset.mem_range.mp ht)
    calc w12 t = w12 t * 1 := (mul_one _).symm
      _ ≤ w12 t * Dg12 r0 r1 r2 r3 r4 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS12_0 r0 + aS12_1 r1 + aS12_2 r2 + aS12_3 r3 + aS12_4 r4) + (aP12_0 r0 r1 + aP12_1 r0 r2 + aP12_2 r0 r3 + aP12_3 r0 r4 + aP12_4 r1 r2 + aP12_5 r1 r3 + aP12_6 r1 r4 + aP12_7 r2 r3 + aP12_8 r2 r4 + aP12_9 r3 r4) = (S12_0 r0 + S12_1 r1 + S12_2 r2 + S12_3 r3 + S12_4 r4) - 2 * (N12_0 r0 r1 + N12_1 r0 r2 + N12_2 r0 r3 + N12_3 r0 r4 + N12_4 r1 r2 + N12_5 r1 r3 + N12_6 r1 r4 + N12_7 r2 r3 + N12_8 r2 r4 + N12_9 r3 r4) := by
    simp only [aS12_0, aS12_1, aS12_2, aS12_3, aS12_4, aP12_0, aP12_1, aP12_2, aP12_3, aP12_4, aP12_5, aP12_6, aP12_7, aP12_8, aP12_9, L12_0, L12_1, L12_2, L12_3, L12_4]
    ring
  have bS0 : aS12_0 r0 ≤ MS12_0 := CaseSplit.le_mxr (aS12_0) 10 r0 (by omega)
  have bS1 : aS12_1 r1 ≤ MS12_1 := CaseSplit.le_mxr (aS12_1) 12 r1 (by omega)
  have bS2 : aS12_2 r2 ≤ MS12_2 := CaseSplit.le_mxr (aS12_2) 16 r2 (by omega)
  have bS3 : aS12_3 r3 ≤ MS12_3 := CaseSplit.le_mxr (aS12_3) 18 r3 (by omega)
  have bS4 : aS12_4 r4 ≤ MS12_4 := CaseSplit.le_mxr (aS12_4) 22 r4 (by omega)
  have bP0 : aP12_0 r0 r1 ≤ MP12_0 := CaseSplit.le_mxr2 (aP12_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP12_1 r0 r2 ≤ MP12_1 := CaseSplit.le_mxr2 (aP12_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP12_2 r0 r3 ≤ MP12_2 := CaseSplit.le_mxr2 (aP12_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP12_3 r0 r4 ≤ MP12_3 := CaseSplit.le_mxr2 (aP12_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP12_4 r1 r2 ≤ MP12_4 := CaseSplit.le_mxr2 (aP12_4) 12 16 r1 r2 (by omega) (by omega)
  have bP5 : aP12_5 r1 r3 ≤ MP12_5 := CaseSplit.le_mxr2 (aP12_5) 12 18 r1 r3 (by omega) (by omega)
  have bP6 : aP12_6 r1 r4 ≤ MP12_6 := CaseSplit.le_mxr2 (aP12_6) 12 22 r1 r4 (by omega) (by omega)
  have bP7 : aP12_7 r2 r3 ≤ MP12_7 := CaseSplit.le_mxr2 (aP12_7) 16 18 r2 r3 (by omega) (by omega)
  have bP8 : aP12_8 r2 r4 ≤ MP12_8 := CaseSplit.le_mxr2 (aP12_8) 16 22 r2 r4 (by omega) (by omega)
  have bP9 : aP12_9 r3 r4 ≤ MP12_9 := CaseSplit.le_mxr2 (aP12_9) 18 22 r3 r4 (by omega) (by omega)
  have hrhs : rhs12 = (∑ t ∈ Finset.range n12, w12 t) + 2 * (n12 : ℤ) := rfl
  have hc := cert12
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, bS0, bS1, bS2, bS3, bS4, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9]

end IncCert23
