/-
INCREMENT-WIDTH CERTIFICATE, step 19->23, case 13 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_19_23.json, which re-derives every number
from the primes alone).

Machine 23, INCREMENT width 39 = F_2(19) + s_min(23) = 31 + 8,
held gears [5, 7] at phases [1, 6].  Free gears [11, 13, 17, 19, 23].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 3.
-/
import IncCert23B

namespace IncCert23

/-! ### case 13: held gears at phases [1, 6] -/

def p13 : List ℕ := [1, 4, 6, 11, 12, 17, 19, 22, 24, 26, 27, 29, 31, 32, 34, 36]
def q13 (t : ℕ) : ℕ := p13.getD t 0
def n13 : ℕ := 16
def yl13 : List ℤ := [0, 0, 0, 0, 0, 1, 1, 2, 3, 2, 3, 2, 0, 0, 2, 0]
def w13 (t : ℕ) : ℤ := yl13.getD t 0
def ul13 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-2), (-2), 0, 0, 0, (-2), (-2), (-2), 0, (-2), 0, (-2), (-2), 0, 0, (-2), (-2), 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, (-1), (-2), (-2), (-2), (-2), (-1), (-2), (-2), (-2), (-1), (-2), 0, (-1), (-1), (-2), (-1), (-2), (-2), (-1), 1, 0, 1, 0, (-2), 0, 0, 1, 0, 2, 0, 0, (-2), (-2), (-1), (-2), (-5), (-7), (-1), 0, (-2), (-2), (-2), (-2), (-2), (-7), 0, 0, (-2), (-1), (-2), (-2), (-2), (-2), 0, 2, 0, 0, 0, 0, 0, 0, 2, 1, 0, 3, 6, 7, 7, 4, 7, 7, 6, 7, 7, 6, 5, 6, 6, 7, 7, 7, (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), (-7), 6, 5, 3, 6, 4, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-5), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 5, 4, 7, 2, 8, 8, 2, 8, 2, 5, 8, 2, 7, 8, 7, 8, 8, 8, 3, 4, 8, 2, (-2), 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, (-2), 0, 0, 0, 0, 0]
def u13 (k : ℕ) : ℤ := ul13.getD k 0

def c13_0 (r t : ℕ) : Bool := gb11 r (q13 t)
def c13_1 (r t : ℕ) : Bool := gb13 r (q13 t)
def c13_2 (r t : ℕ) : Bool := gb17 r (q13 t)
def c13_3 (r t : ℕ) : Bool := gb19 r (q13 t)
def c13_4 (r t : ℕ) : Bool := gb23 r (q13 t)

def S13_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (w13 t + 2) * (if c13_0 r t then 1 else 0)
def S13_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (w13 t + 2) * (if c13_1 r t then 1 else 0)
def S13_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (w13 t + 2) * (if c13_2 r t then 1 else 0)
def S13_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (w13 t + 2) * (if c13_3 r t then 1 else 0)
def S13_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (w13 t + 2) * (if c13_4 r t then 1 else 0)

def L13_0 (r : ℕ) : ℤ := u13 (13 + r) + u13 (41 + r) + u13 (71 + r) + u13 (105 + r)
def L13_1 (r : ℕ) : ℤ := u13 (0 + r) + u13 (133 + r) + u13 (165 + r) + u13 (201 + r)
def L13_2 (r : ℕ) : ℤ := u13 (24 + r) + u13 (116 + r) + u13 (233 + r) + u13 (273 + r)
def L13_3 (r : ℕ) : ℤ := u13 (52 + r) + u13 (146 + r) + u13 (214 + r) + u13 (313 + r)
def L13_4 (r : ℕ) : ℤ := u13 (82 + r) + u13 (178 + r) + u13 (250 + r) + u13 (290 + r)

def aS13_0 (r : ℕ) : ℤ := S13_0 r - L13_0 r
def MS13_0 : ℤ := CaseSplit.mxr (aS13_0) 10
def aS13_1 (r : ℕ) : ℤ := S13_1 r - L13_1 r
def MS13_1 : ℤ := CaseSplit.mxr (aS13_1) 12
def aS13_2 (r : ℕ) : ℤ := S13_2 r - L13_2 r
def MS13_2 : ℤ := CaseSplit.mxr (aS13_2) 16
def aS13_3 (r : ℕ) : ℤ := S13_3 r - L13_3 r
def MS13_3 : ℤ := CaseSplit.mxr (aS13_3) 18
def aS13_4 (r : ℕ) : ℤ := S13_4 r - L13_4 r
def MS13_4 : ℤ := CaseSplit.mxr (aS13_4) 22

def N13_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (if c13_0 ra t && c13_1 rb t then 1 else 0)
def aP13_0 (ra rb : ℕ) : ℤ := -(2) * N13_0 ra rb + u13 (0 + rb) + u13 (13 + ra)
def MP13_0 : ℤ := CaseSplit.mxr2 (aP13_0) 10 12
def N13_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (if c13_0 ra t && c13_2 rb t then 1 else 0)
def aP13_1 (ra rb : ℕ) : ℤ := -(2) * N13_1 ra rb + u13 (24 + rb) + u13 (41 + ra)
def MP13_1 : ℤ := CaseSplit.mxr2 (aP13_1) 10 16
def N13_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (if c13_0 ra t && c13_3 rb t then 1 else 0)
def aP13_2 (ra rb : ℕ) : ℤ := -(2) * N13_2 ra rb + u13 (52 + rb) + u13 (71 + ra)
def MP13_2 : ℤ := CaseSplit.mxr2 (aP13_2) 10 18
def N13_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (if c13_0 ra t && c13_4 rb t then 1 else 0)
def aP13_3 (ra rb : ℕ) : ℤ := -(2) * N13_3 ra rb + u13 (82 + rb) + u13 (105 + ra)
def MP13_3 : ℤ := CaseSplit.mxr2 (aP13_3) 10 22
def P13_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (if c13_1 ra t && c13_2 rb t then 1 else 0)
def C13_4 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (if c13_1 ra t && c13_2 rb t && c13_0 s t then 1 else 0)
def M13_4 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C13_4 ra rb) 10
def E13_4 : List ℕ := [86, 97, 122, 133, 206, 212]
def N13_4 (ra rb : ℕ) : ℤ := if E13_4.contains (ra * 17 + rb) = true then P13_4 ra rb - M13_4 ra rb else 0
def aP13_4 (ra rb : ℕ) : ℤ := -(2) * N13_4 ra rb + u13 (116 + rb) + u13 (133 + ra)
def MP13_4 : ℤ := CaseSplit.mxr2 (aP13_4) 12 16
def P13_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (if c13_1 ra t && c13_3 rb t then 1 else 0)
def C13_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (if c13_1 ra t && c13_3 rb t && c13_0 s t then 1 else 0)
def M13_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C13_5 ra rb) 10
def E13_5 : List ℕ := [11, 87, 98, 111, 151, 174, 187, 227]
def N13_5 (ra rb : ℕ) : ℤ := if E13_5.contains (ra * 19 + rb) = true then P13_5 ra rb - M13_5 ra rb else 0
def aP13_5 (ra rb : ℕ) : ℤ := -(2) * N13_5 ra rb + u13 (146 + rb) + u13 (165 + ra)
def MP13_5 : ℤ := CaseSplit.mxr2 (aP13_5) 12 18
def P13_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (if c13_1 ra t && c13_4 rb t then 1 else 0)
def C13_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n13, (if c13_1 ra t && c13_4 rb t && c13_0 s t then 1 else 0)
def M13_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C13_6 ra rb) 10
def E13_6 : List ℕ := []
def N13_6 (ra rb : ℕ) : ℤ := if E13_6.contains (ra * 23 + rb) = true then P13_6 ra rb - M13_6 ra rb else 0
def aP13_6 (ra rb : ℕ) : ℤ := -(2) * N13_6 ra rb + u13 (178 + rb) + u13 (201 + ra)
def MP13_6 : ℤ := CaseSplit.mxr2 (aP13_6) 12 22
def N13_7 (_ra _rb : ℕ) : ℤ := 0
def aP13_7 (ra rb : ℕ) : ℤ := -(2) * N13_7 ra rb + u13 (214 + rb) + u13 (233 + ra)
def MP13_7 : ℤ := CaseSplit.mxr2 (aP13_7) 16 18
def N13_8 (_ra _rb : ℕ) : ℤ := 0
def aP13_8 (ra rb : ℕ) : ℤ := -(2) * N13_8 ra rb + u13 (250 + rb) + u13 (273 + ra)
def MP13_8 : ℤ := CaseSplit.mxr2 (aP13_8) 16 22
def N13_9 (_ra _rb : ℕ) : ℤ := 0
def aP13_9 (ra rb : ℕ) : ℤ := -(2) * N13_9 ra rb + u13 (290 + rb) + u13 (313 + ra)
def MP13_9 : ℤ := CaseSplit.mxr2 (aP13_9) 18 22

def rhs13 : ℤ := (∑ t ∈ Finset.range n13, w13 t) + 2 * (n13 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn13 : ∀ t, t < n13 → (0 : ℤ) ≤ w13 t := by decide
theorem plt13 : ∀ t, t < n13 → q13 t < 39 := by decide
theorem pfree13_5 : ∀ t, t < n13 → gb5 1 (q13 t) = false := by decide
theorem pfree13_7 : ∀ t, t < n13 → gb7 6 (q13 t) = false := by decide
theorem MSv13_0 : MS13_0 = 9 := by decide +kernel
theorem MSv13_1 : MS13_1 = 22 := by decide +kernel
theorem MSv13_2 : MS13_2 = 2 := by decide +kernel
theorem MSv13_3 : MS13_3 = 2 := by decide +kernel
theorem MSv13_4 : MS13_4 = 2 := by decide +kernel
theorem MPv13_0 : MP13_0 = 0 := by decide +kernel
theorem MPv13_1 : MP13_1 = 0 := by decide +kernel
theorem MPv13_2 : MP13_2 = 0 := by decide +kernel
theorem MPv13_3 : MP13_3 = 0 := by decide +kernel
theorem MPv13_4 : MP13_4 = 0 := by decide +kernel
theorem MPv13_5 : MP13_5 = 0 := by decide +kernel
theorem MPv13_6 : MP13_6 = 0 := by decide +kernel
theorem MPv13_7 : MP13_7 = 0 := by decide +kernel
theorem MPv13_8 : MP13_8 = 0 := by decide +kernel
theorem MPv13_9 : MP13_9 = 8 := by decide +kernel
theorem rhsv13 : rhs13 = 48 := by decide +kernel

/-- **The case-13 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 3/3.
    (Scaled by the common denominator 3: 45 < 48.) -/
theorem cert13 : MS13_0 + MS13_1 + MS13_2 + MS13_3 + MS13_4 + MP13_0 + MP13_1 + MP13_2 + MP13_3 + MP13_4 + MP13_5 + MP13_6 + MP13_7 + MP13_8 + MP13_9 < rhs13 := by
  rw [MSv13_0, MSv13_1, MSv13_2, MSv13_3, MSv13_4, MPv13_0, MPv13_1, MPv13_2, MPv13_3, MPv13_4, MPv13_5, MPv13_6, MPv13_7, MPv13_8, MPv13_9, rhsv13]
  decide

def Dg13 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := (if c13_0 r0 t then 1 else 0) + (if c13_1 r1 t then 1 else 0) + (if c13_2 r2 t then 1 else 0) + (if c13_3 r3 t then 1 else 0) + (if c13_4 r4 t then 1 else 0)
def Wl13_0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c13_0 r0 t && c13_1 r1 t then 1 else 0
def Wl13_1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c13_0 r0 t && c13_2 r2 t then 1 else 0
def Wl13_2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c13_0 r0 t && c13_3 r3 t then 1 else 0
def Wl13_3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c13_0 r0 t && c13_4 r4 t then 1 else 0
def Wl13_4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c13_0 r0 t && c13_1 r1 t && c13_2 r2 t then 1 else 0
def Wl13_5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c13_0 r0 t && c13_1 r1 t && c13_3 r3 t then 1 else 0
def Wl13_6 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c13_0 r0 t && c13_1 r1 t && c13_4 r4 t then 1 else 0
def Wl13_7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c13_0 r0 t && !c13_1 r1 t && c13_2 r2 t && c13_3 r3 t then 1 else 0
def Wl13_8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c13_0 r0 t && !c13_1 r1 t && c13_2 r2 t && c13_4 r4 t then 1 else 0
def Wl13_9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c13_0 r0 t && !c13_1 r1 t && !c13_2 r2 t && c13_3 r3 t && c13_4 r4 t then 1 else 0

/-- **No configuration blocks the whole window in case 13.** -/
theorem nocov13 {r0 r1 r2 r3 r4 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23)
    (hcov : ∀ t, t < n13 → (c13_0 r0 t || c13_1 r1 t || c13_2 r2 t || c13_3 r3 t || c13_4 r4 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n13, (1 : ℤ) + (Wl13_0 r0 r1 r2 r3 r4 t + Wl13_1 r0 r1 r2 r3 r4 t + Wl13_2 r0 r1 r2 r3 r4 t + Wl13_3 r0 r1 r2 r3 r4 t + Wl13_4 r0 r1 r2 r3 r4 t + Wl13_5 r0 r1 r2 r3 r4 t + Wl13_6 r0 r1 r2 r3 r4 t + Wl13_7 r0 r1 r2 r3 r4 t + Wl13_8 r0 r1 r2 r3 r4 t + Wl13_9 r0 r1 r2 r3 r4 t) ≤ Dg13 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Wl13_0, Wl13_1, Wl13_2, Wl13_3, Wl13_4, Wl13_5, Wl13_6, Wl13_7, Wl13_8, Wl13_9, Dg13]
    exact CaseSplit.lowest5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n13, (1 : ℤ) ≤ Dg13 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Dg13]
    exact CaseSplit.degpos5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n13 : ℤ) + ((∑ t ∈ Finset.range n13, Wl13_0 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n13, Wl13_1 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n13, Wl13_2 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n13, Wl13_3 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n13, Wl13_4 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n13, Wl13_5 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n13, Wl13_6 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n13, Wl13_7 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n13, Wl13_8 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n13, Wl13_9 r0 r1 r2 r3 r4 t)) ≤ ∑ t ∈ Finset.range n13, Dg13 r0 r1 r2 r3 r4 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N13_0 r0 r1 ≤ ∑ t ∈ Finset.range n13, Wl13_0 r0 r1 r2 r3 r4 t := by
    simp only [N13_0, Wl13_0, le_refl]
  have hn1 : N13_1 r0 r2 ≤ ∑ t ∈ Finset.range n13, Wl13_1 r0 r1 r2 r3 r4 t := by
    simp only [N13_1, Wl13_1, le_refl]
  have hn2 : N13_2 r0 r3 ≤ ∑ t ∈ Finset.range n13, Wl13_2 r0 r1 r2 r3 r4 t := by
    simp only [N13_2, Wl13_2, le_refl]
  have hn3 : N13_3 r0 r4 ≤ ∑ t ∈ Finset.range n13, Wl13_3 r0 r1 r2 r3 r4 t := by
    simp only [N13_3, Wl13_3, le_refl]
  have hn4 : N13_4 r1 r2 ≤ ∑ t ∈ Finset.range n13, Wl13_4 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n13, Wl13_4 r0 r1 r2 r3 r4 t
        = (if c13_1 r1 t && c13_2 r2 t then (1:ℤ) else 0)
          - (if c13_1 r1 t && c13_2 r2 t && c13_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl13_4]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n13, Wl13_4 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl13_4]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n13, Wl13_4 r0 r1 r2 r3 r4 t
        = P13_4 r1 r2 - C13_4 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P13_4, C13_4]
    have hm : C13_4 r1 r2 r0 ≤ M13_4 r1 r2 :=
      CaseSplit.le_mxr (C13_4 r1 r2) 10 r0 (by omega)
    simp only [N13_4]
    split
    · rw [hL]; omega
    · exact hnn
  have hn5 : N13_5 r1 r3 ≤ ∑ t ∈ Finset.range n13, Wl13_5 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n13, Wl13_5 r0 r1 r2 r3 r4 t
        = (if c13_1 r1 t && c13_3 r3 t then (1:ℤ) else 0)
          - (if c13_1 r1 t && c13_3 r3 t && c13_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl13_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n13, Wl13_5 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl13_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n13, Wl13_5 r0 r1 r2 r3 r4 t
        = P13_5 r1 r3 - C13_5 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P13_5, C13_5]
    have hm : C13_5 r1 r3 r0 ≤ M13_5 r1 r3 :=
      CaseSplit.le_mxr (C13_5 r1 r3) 10 r0 (by omega)
    simp only [N13_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N13_6 r1 r4 ≤ ∑ t ∈ Finset.range n13, Wl13_6 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n13, Wl13_6 r0 r1 r2 r3 r4 t
        = (if c13_1 r1 t && c13_4 r4 t then (1:ℤ) else 0)
          - (if c13_1 r1 t && c13_4 r4 t && c13_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl13_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n13, Wl13_6 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl13_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n13, Wl13_6 r0 r1 r2 r3 r4 t
        = P13_6 r1 r4 - C13_6 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P13_6, C13_6]
    have hm : C13_6 r1 r4 r0 ≤ M13_6 r1 r4 :=
      CaseSplit.le_mxr (C13_6 r1 r4) 10 r0 (by omega)
    simp only [N13_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N13_7 r2 r3 ≤ ∑ t ∈ Finset.range n13, Wl13_7 r0 r1 r2 r3 r4 t := by
    simp only [N13_7]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl13_7]
    exact CaseSplit.ind_nonneg _
  have hn8 : N13_8 r2 r4 ≤ ∑ t ∈ Finset.range n13, Wl13_8 r0 r1 r2 r3 r4 t := by
    simp only [N13_8]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl13_8]
    exact CaseSplit.ind_nonneg _
  have hn9 : N13_9 r3 r4 ≤ ∑ t ∈ Finset.range n13, Wl13_9 r0 r1 r2 r3 r4 t := by
    simp only [N13_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl13_9]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n13, (w13 t + 2) * Dg13 r0 r1 r2 r3 r4 t = S13_0 r0 + S13_1 r1 + S13_2 r2 + S13_3 r3 + S13_4 r4 := by
    simp only [S13_0, S13_1, S13_2, S13_3, S13_4, Dg13, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n13, (w13 t + 2) * Dg13 r0 r1 r2 r3 r4 t
      = (∑ t ∈ Finset.range n13, w13 t * Dg13 r0 r1 r2 r3 r4 t)
        + 2 * (∑ t ∈ Finset.range n13, Dg13 r0 r1 r2 r3 r4 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n13, w13 t)
      ≤ ∑ t ∈ Finset.range n13, w13 t * Dg13 r0 r1 r2 r3 r4 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg13 r0 r1 r2 r3 r4 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w13 t := wnn13 t (Finset.mem_range.mp ht)
    calc w13 t = w13 t * 1 := (mul_one _).symm
      _ ≤ w13 t * Dg13 r0 r1 r2 r3 r4 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS13_0 r0 + aS13_1 r1 + aS13_2 r2 + aS13_3 r3 + aS13_4 r4) + (aP13_0 r0 r1 + aP13_1 r0 r2 + aP13_2 r0 r3 + aP13_3 r0 r4 + aP13_4 r1 r2 + aP13_5 r1 r3 + aP13_6 r1 r4 + aP13_7 r2 r3 + aP13_8 r2 r4 + aP13_9 r3 r4) = (S13_0 r0 + S13_1 r1 + S13_2 r2 + S13_3 r3 + S13_4 r4) - 2 * (N13_0 r0 r1 + N13_1 r0 r2 + N13_2 r0 r3 + N13_3 r0 r4 + N13_4 r1 r2 + N13_5 r1 r3 + N13_6 r1 r4 + N13_7 r2 r3 + N13_8 r2 r4 + N13_9 r3 r4) := by
    simp only [aS13_0, aS13_1, aS13_2, aS13_3, aS13_4, aP13_0, aP13_1, aP13_2, aP13_3, aP13_4, aP13_5, aP13_6, aP13_7, aP13_8, aP13_9, L13_0, L13_1, L13_2, L13_3, L13_4]
    ring
  have bS0 : aS13_0 r0 ≤ MS13_0 := CaseSplit.le_mxr (aS13_0) 10 r0 (by omega)
  have bS1 : aS13_1 r1 ≤ MS13_1 := CaseSplit.le_mxr (aS13_1) 12 r1 (by omega)
  have bS2 : aS13_2 r2 ≤ MS13_2 := CaseSplit.le_mxr (aS13_2) 16 r2 (by omega)
  have bS3 : aS13_3 r3 ≤ MS13_3 := CaseSplit.le_mxr (aS13_3) 18 r3 (by omega)
  have bS4 : aS13_4 r4 ≤ MS13_4 := CaseSplit.le_mxr (aS13_4) 22 r4 (by omega)
  have bP0 : aP13_0 r0 r1 ≤ MP13_0 := CaseSplit.le_mxr2 (aP13_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP13_1 r0 r2 ≤ MP13_1 := CaseSplit.le_mxr2 (aP13_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP13_2 r0 r3 ≤ MP13_2 := CaseSplit.le_mxr2 (aP13_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP13_3 r0 r4 ≤ MP13_3 := CaseSplit.le_mxr2 (aP13_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP13_4 r1 r2 ≤ MP13_4 := CaseSplit.le_mxr2 (aP13_4) 12 16 r1 r2 (by omega) (by omega)
  have bP5 : aP13_5 r1 r3 ≤ MP13_5 := CaseSplit.le_mxr2 (aP13_5) 12 18 r1 r3 (by omega) (by omega)
  have bP6 : aP13_6 r1 r4 ≤ MP13_6 := CaseSplit.le_mxr2 (aP13_6) 12 22 r1 r4 (by omega) (by omega)
  have bP7 : aP13_7 r2 r3 ≤ MP13_7 := CaseSplit.le_mxr2 (aP13_7) 16 18 r2 r3 (by omega) (by omega)
  have bP8 : aP13_8 r2 r4 ≤ MP13_8 := CaseSplit.le_mxr2 (aP13_8) 16 22 r2 r4 (by omega) (by omega)
  have bP9 : aP13_9 r3 r4 ≤ MP13_9 := CaseSplit.le_mxr2 (aP13_9) 18 22 r3 r4 (by omega) (by omega)
  have hrhs : rhs13 = (∑ t ∈ Finset.range n13, w13 t) + 2 * (n13 : ℤ) := rfl
  have hc := cert13
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, bS0, bS1, bS2, bS3, bS4, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9]

end IncCert23
