/-
INCREMENT-WIDTH CERTIFICATE, step 19->23, case 15 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_19_23.json, which re-derives every number
from the primes alone).

Machine 23, INCREMENT width 39 = F_2(19) + s_min(23) = 31 + 8,
held gears [5, 7] at phases [2, 1].  Free gears [11, 13, 17, 19, 23].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 1.
-/
import IncCert23B

namespace IncCert23

/-! ### case 15: held gears at phases [2, 1] -/

def p15 : List ℕ := [1, 3, 6, 8, 10, 11, 13, 15, 16, 18, 20, 23, 25, 30, 31, 36, 38]
def q15 (t : ℕ) : ℕ := p15.getD t 0
def n15 : ℕ := 17
def yl15 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def w15 (t : ℕ) : ℤ := yl15.getD t 0
def ul15 : List ℤ := [0, 0, 0, (-1), (-2), 0, 0, (-2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, (-1), 1, 0, 0, (-1), 0, 1, 1, 0, 0, 0, 0, (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), 0, (-1), (-1), (-2), 0, 0, (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), 0, 1, 0, (-1), 1, 0, 0, 0, 1, 0, 0, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 3, 2, 3, 1, 3, 2, 2, 3, 1, 3, 3, 2, 3, 1, 3, 3, 2, 3, 2, 3, 2, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, (-1), 1, 1, 1, 1, 0]
def u15 (k : ℕ) : ℤ := ul15.getD k 0

def c15_0 (r t : ℕ) : Bool := gb11 r (q15 t)
def c15_1 (r t : ℕ) : Bool := gb13 r (q15 t)
def c15_2 (r t : ℕ) : Bool := gb17 r (q15 t)
def c15_3 (r t : ℕ) : Bool := gb19 r (q15 t)
def c15_4 (r t : ℕ) : Bool := gb23 r (q15 t)

def S15_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (w15 t + 1) * (if c15_0 r t then 1 else 0)
def S15_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (w15 t + 1) * (if c15_1 r t then 1 else 0)
def S15_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (w15 t + 1) * (if c15_2 r t then 1 else 0)
def S15_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (w15 t + 1) * (if c15_3 r t then 1 else 0)
def S15_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (w15 t + 1) * (if c15_4 r t then 1 else 0)

def L15_0 (r : ℕ) : ℤ := u15 (13 + r) + u15 (41 + r) + u15 (71 + r) + u15 (105 + r)
def L15_1 (r : ℕ) : ℤ := u15 (0 + r) + u15 (133 + r) + u15 (165 + r) + u15 (201 + r)
def L15_2 (r : ℕ) : ℤ := u15 (24 + r) + u15 (116 + r) + u15 (233 + r) + u15 (273 + r)
def L15_3 (r : ℕ) : ℤ := u15 (52 + r) + u15 (146 + r) + u15 (214 + r) + u15 (313 + r)
def L15_4 (r : ℕ) : ℤ := u15 (82 + r) + u15 (178 + r) + u15 (250 + r) + u15 (290 + r)

def aS15_0 (r : ℕ) : ℤ := S15_0 r - L15_0 r
def MS15_0 : ℤ := CaseSplit.mxr (aS15_0) 10
def aS15_1 (r : ℕ) : ℤ := S15_1 r - L15_1 r
def MS15_1 : ℤ := CaseSplit.mxr (aS15_1) 12
def aS15_2 (r : ℕ) : ℤ := S15_2 r - L15_2 r
def MS15_2 : ℤ := CaseSplit.mxr (aS15_2) 16
def aS15_3 (r : ℕ) : ℤ := S15_3 r - L15_3 r
def MS15_3 : ℤ := CaseSplit.mxr (aS15_3) 18
def aS15_4 (r : ℕ) : ℤ := S15_4 r - L15_4 r
def MS15_4 : ℤ := CaseSplit.mxr (aS15_4) 22

def N15_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (if c15_0 ra t && c15_1 rb t then 1 else 0)
def aP15_0 (ra rb : ℕ) : ℤ := -(1) * N15_0 ra rb + u15 (0 + rb) + u15 (13 + ra)
def MP15_0 : ℤ := CaseSplit.mxr2 (aP15_0) 10 12
def N15_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (if c15_0 ra t && c15_2 rb t then 1 else 0)
def aP15_1 (ra rb : ℕ) : ℤ := -(1) * N15_1 ra rb + u15 (24 + rb) + u15 (41 + ra)
def MP15_1 : ℤ := CaseSplit.mxr2 (aP15_1) 10 16
def N15_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (if c15_0 ra t && c15_3 rb t then 1 else 0)
def aP15_2 (ra rb : ℕ) : ℤ := -(1) * N15_2 ra rb + u15 (52 + rb) + u15 (71 + ra)
def MP15_2 : ℤ := CaseSplit.mxr2 (aP15_2) 10 18
def N15_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (if c15_0 ra t && c15_4 rb t then 1 else 0)
def aP15_3 (ra rb : ℕ) : ℤ := -(1) * N15_3 ra rb + u15 (82 + rb) + u15 (105 + ra)
def MP15_3 : ℤ := CaseSplit.mxr2 (aP15_3) 10 22
def P15_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (if c15_1 ra t && c15_2 rb t then 1 else 0)
def C15_4 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (if c15_1 ra t && c15_2 rb t && c15_0 s t then 1 else 0)
def M15_4 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C15_4 ra rb) 10
def E15_4 : List ℕ := [57, 63, 93, 99, 136, 147, 172, 183, 188, 194]
def N15_4 (ra rb : ℕ) : ℤ := if E15_4.contains (ra * 17 + rb) = true then P15_4 ra rb - M15_4 ra rb else 0
def aP15_4 (ra rb : ℕ) : ℤ := -(1) * N15_4 ra rb + u15 (116 + rb) + u15 (133 + ra)
def MP15_4 : ℤ := CaseSplit.mxr2 (aP15_4) 12 16
def P15_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (if c15_1 ra t && c15_3 rb t then 1 else 0)
def C15_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (if c15_1 ra t && c15_3 rb t && c15_0 s t then 1 else 0)
def M15_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C15_5 ra rb) 10
def E15_5 : List ℕ := [31, 37, 73, 107, 113, 118, 152, 194, 228, 244]
def N15_5 (ra rb : ℕ) : ℤ := if E15_5.contains (ra * 19 + rb) = true then P15_5 ra rb - M15_5 ra rb else 0
def aP15_5 (ra rb : ℕ) : ℤ := -(1) * N15_5 ra rb + u15 (146 + rb) + u15 (165 + ra)
def MP15_5 : ℤ := CaseSplit.mxr2 (aP15_5) 12 18
def P15_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (if c15_1 ra t && c15_4 rb t then 1 else 0)
def C15_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n15, (if c15_1 ra t && c15_4 rb t && c15_0 s t then 1 else 0)
def M15_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C15_6 ra rb) 10
def E15_6 : List ℕ := []
def N15_6 (ra rb : ℕ) : ℤ := if E15_6.contains (ra * 23 + rb) = true then P15_6 ra rb - M15_6 ra rb else 0
def aP15_6 (ra rb : ℕ) : ℤ := -(1) * N15_6 ra rb + u15 (178 + rb) + u15 (201 + ra)
def MP15_6 : ℤ := CaseSplit.mxr2 (aP15_6) 12 22
def N15_7 (_ra _rb : ℕ) : ℤ := 0
def aP15_7 (ra rb : ℕ) : ℤ := -(1) * N15_7 ra rb + u15 (214 + rb) + u15 (233 + ra)
def MP15_7 : ℤ := CaseSplit.mxr2 (aP15_7) 16 18
def N15_8 (_ra _rb : ℕ) : ℤ := 0
def aP15_8 (ra rb : ℕ) : ℤ := -(1) * N15_8 ra rb + u15 (250 + rb) + u15 (273 + ra)
def MP15_8 : ℤ := CaseSplit.mxr2 (aP15_8) 16 22
def N15_9 (_ra _rb : ℕ) : ℤ := 0
def aP15_9 (ra rb : ℕ) : ℤ := -(1) * N15_9 ra rb + u15 (290 + rb) + u15 (313 + ra)
def MP15_9 : ℤ := CaseSplit.mxr2 (aP15_9) 18 22

def rhs15 : ℤ := (∑ t ∈ Finset.range n15, w15 t) + 1 * (n15 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn15 : ∀ t, t < n15 → (0 : ℤ) ≤ w15 t := by decide
theorem plt15 : ∀ t, t < n15 → q15 t < 39 := by decide
theorem pfree15_5 : ∀ t, t < n15 → gb5 2 (q15 t) = false := by decide
theorem pfree15_7 : ∀ t, t < n15 → gb7 1 (q15 t) = false := by decide
theorem MSv15_0 : MS15_0 = 5 := by decide +kernel
theorem MSv15_1 : MS15_1 = 7 := by decide +kernel
theorem MSv15_2 : MS15_2 = 0 := by decide +kernel
theorem MSv15_3 : MS15_3 = 0 := by decide +kernel
theorem MSv15_4 : MS15_4 = 0 := by decide +kernel
theorem MPv15_0 : MP15_0 = 0 := by decide +kernel
theorem MPv15_1 : MP15_1 = 0 := by decide +kernel
theorem MPv15_2 : MP15_2 = 0 := by decide +kernel
theorem MPv15_3 : MP15_3 = 0 := by decide +kernel
theorem MPv15_4 : MP15_4 = 0 := by decide +kernel
theorem MPv15_5 : MP15_5 = 0 := by decide +kernel
theorem MPv15_6 : MP15_6 = 0 := by decide +kernel
theorem MPv15_7 : MP15_7 = 0 := by decide +kernel
theorem MPv15_8 : MP15_8 = 0 := by decide +kernel
theorem MPv15_9 : MP15_9 = 4 := by decide +kernel
theorem rhsv15 : rhs15 = 17 := by decide +kernel

/-- **The case-15 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/1.
    (Scaled by the common denominator 1: 16 < 17.) -/
theorem cert15 : MS15_0 + MS15_1 + MS15_2 + MS15_3 + MS15_4 + MP15_0 + MP15_1 + MP15_2 + MP15_3 + MP15_4 + MP15_5 + MP15_6 + MP15_7 + MP15_8 + MP15_9 < rhs15 := by
  rw [MSv15_0, MSv15_1, MSv15_2, MSv15_3, MSv15_4, MPv15_0, MPv15_1, MPv15_2, MPv15_3, MPv15_4, MPv15_5, MPv15_6, MPv15_7, MPv15_8, MPv15_9, rhsv15]
  decide

def Dg15 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := (if c15_0 r0 t then 1 else 0) + (if c15_1 r1 t then 1 else 0) + (if c15_2 r2 t then 1 else 0) + (if c15_3 r3 t then 1 else 0) + (if c15_4 r4 t then 1 else 0)
def Wl15_0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c15_0 r0 t && c15_1 r1 t then 1 else 0
def Wl15_1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c15_0 r0 t && c15_2 r2 t then 1 else 0
def Wl15_2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c15_0 r0 t && c15_3 r3 t then 1 else 0
def Wl15_3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c15_0 r0 t && c15_4 r4 t then 1 else 0
def Wl15_4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c15_0 r0 t && c15_1 r1 t && c15_2 r2 t then 1 else 0
def Wl15_5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c15_0 r0 t && c15_1 r1 t && c15_3 r3 t then 1 else 0
def Wl15_6 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c15_0 r0 t && c15_1 r1 t && c15_4 r4 t then 1 else 0
def Wl15_7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c15_0 r0 t && !c15_1 r1 t && c15_2 r2 t && c15_3 r3 t then 1 else 0
def Wl15_8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c15_0 r0 t && !c15_1 r1 t && c15_2 r2 t && c15_4 r4 t then 1 else 0
def Wl15_9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c15_0 r0 t && !c15_1 r1 t && !c15_2 r2 t && c15_3 r3 t && c15_4 r4 t then 1 else 0

/-- **No configuration blocks the whole window in case 15.** -/
theorem nocov15 {r0 r1 r2 r3 r4 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23)
    (hcov : ∀ t, t < n15 → (c15_0 r0 t || c15_1 r1 t || c15_2 r2 t || c15_3 r3 t || c15_4 r4 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n15, (1 : ℤ) + (Wl15_0 r0 r1 r2 r3 r4 t + Wl15_1 r0 r1 r2 r3 r4 t + Wl15_2 r0 r1 r2 r3 r4 t + Wl15_3 r0 r1 r2 r3 r4 t + Wl15_4 r0 r1 r2 r3 r4 t + Wl15_5 r0 r1 r2 r3 r4 t + Wl15_6 r0 r1 r2 r3 r4 t + Wl15_7 r0 r1 r2 r3 r4 t + Wl15_8 r0 r1 r2 r3 r4 t + Wl15_9 r0 r1 r2 r3 r4 t) ≤ Dg15 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Wl15_0, Wl15_1, Wl15_2, Wl15_3, Wl15_4, Wl15_5, Wl15_6, Wl15_7, Wl15_8, Wl15_9, Dg15]
    exact CaseSplit.lowest5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n15, (1 : ℤ) ≤ Dg15 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Dg15]
    exact CaseSplit.degpos5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n15 : ℤ) + ((∑ t ∈ Finset.range n15, Wl15_0 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n15, Wl15_1 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n15, Wl15_2 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n15, Wl15_3 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n15, Wl15_4 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n15, Wl15_5 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n15, Wl15_6 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n15, Wl15_7 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n15, Wl15_8 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n15, Wl15_9 r0 r1 r2 r3 r4 t)) ≤ ∑ t ∈ Finset.range n15, Dg15 r0 r1 r2 r3 r4 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N15_0 r0 r1 ≤ ∑ t ∈ Finset.range n15, Wl15_0 r0 r1 r2 r3 r4 t := by
    simp only [N15_0, Wl15_0, le_refl]
  have hn1 : N15_1 r0 r2 ≤ ∑ t ∈ Finset.range n15, Wl15_1 r0 r1 r2 r3 r4 t := by
    simp only [N15_1, Wl15_1, le_refl]
  have hn2 : N15_2 r0 r3 ≤ ∑ t ∈ Finset.range n15, Wl15_2 r0 r1 r2 r3 r4 t := by
    simp only [N15_2, Wl15_2, le_refl]
  have hn3 : N15_3 r0 r4 ≤ ∑ t ∈ Finset.range n15, Wl15_3 r0 r1 r2 r3 r4 t := by
    simp only [N15_3, Wl15_3, le_refl]
  have hn4 : N15_4 r1 r2 ≤ ∑ t ∈ Finset.range n15, Wl15_4 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n15, Wl15_4 r0 r1 r2 r3 r4 t
        = (if c15_1 r1 t && c15_2 r2 t then (1:ℤ) else 0)
          - (if c15_1 r1 t && c15_2 r2 t && c15_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl15_4]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n15, Wl15_4 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl15_4]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n15, Wl15_4 r0 r1 r2 r3 r4 t
        = P15_4 r1 r2 - C15_4 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P15_4, C15_4]
    have hm : C15_4 r1 r2 r0 ≤ M15_4 r1 r2 :=
      CaseSplit.le_mxr (C15_4 r1 r2) 10 r0 (by omega)
    simp only [N15_4]
    split
    · rw [hL]; omega
    · exact hnn
  have hn5 : N15_5 r1 r3 ≤ ∑ t ∈ Finset.range n15, Wl15_5 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n15, Wl15_5 r0 r1 r2 r3 r4 t
        = (if c15_1 r1 t && c15_3 r3 t then (1:ℤ) else 0)
          - (if c15_1 r1 t && c15_3 r3 t && c15_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl15_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n15, Wl15_5 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl15_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n15, Wl15_5 r0 r1 r2 r3 r4 t
        = P15_5 r1 r3 - C15_5 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P15_5, C15_5]
    have hm : C15_5 r1 r3 r0 ≤ M15_5 r1 r3 :=
      CaseSplit.le_mxr (C15_5 r1 r3) 10 r0 (by omega)
    simp only [N15_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N15_6 r1 r4 ≤ ∑ t ∈ Finset.range n15, Wl15_6 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n15, Wl15_6 r0 r1 r2 r3 r4 t
        = (if c15_1 r1 t && c15_4 r4 t then (1:ℤ) else 0)
          - (if c15_1 r1 t && c15_4 r4 t && c15_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl15_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n15, Wl15_6 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl15_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n15, Wl15_6 r0 r1 r2 r3 r4 t
        = P15_6 r1 r4 - C15_6 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P15_6, C15_6]
    have hm : C15_6 r1 r4 r0 ≤ M15_6 r1 r4 :=
      CaseSplit.le_mxr (C15_6 r1 r4) 10 r0 (by omega)
    simp only [N15_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N15_7 r2 r3 ≤ ∑ t ∈ Finset.range n15, Wl15_7 r0 r1 r2 r3 r4 t := by
    simp only [N15_7]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl15_7]
    exact CaseSplit.ind_nonneg _
  have hn8 : N15_8 r2 r4 ≤ ∑ t ∈ Finset.range n15, Wl15_8 r0 r1 r2 r3 r4 t := by
    simp only [N15_8]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl15_8]
    exact CaseSplit.ind_nonneg _
  have hn9 : N15_9 r3 r4 ≤ ∑ t ∈ Finset.range n15, Wl15_9 r0 r1 r2 r3 r4 t := by
    simp only [N15_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl15_9]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n15, (w15 t + 1) * Dg15 r0 r1 r2 r3 r4 t = S15_0 r0 + S15_1 r1 + S15_2 r2 + S15_3 r3 + S15_4 r4 := by
    simp only [S15_0, S15_1, S15_2, S15_3, S15_4, Dg15, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n15, (w15 t + 1) * Dg15 r0 r1 r2 r3 r4 t
      = (∑ t ∈ Finset.range n15, w15 t * Dg15 r0 r1 r2 r3 r4 t)
        + 1 * (∑ t ∈ Finset.range n15, Dg15 r0 r1 r2 r3 r4 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n15, w15 t)
      ≤ ∑ t ∈ Finset.range n15, w15 t * Dg15 r0 r1 r2 r3 r4 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg15 r0 r1 r2 r3 r4 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w15 t := wnn15 t (Finset.mem_range.mp ht)
    calc w15 t = w15 t * 1 := (mul_one _).symm
      _ ≤ w15 t * Dg15 r0 r1 r2 r3 r4 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS15_0 r0 + aS15_1 r1 + aS15_2 r2 + aS15_3 r3 + aS15_4 r4) + (aP15_0 r0 r1 + aP15_1 r0 r2 + aP15_2 r0 r3 + aP15_3 r0 r4 + aP15_4 r1 r2 + aP15_5 r1 r3 + aP15_6 r1 r4 + aP15_7 r2 r3 + aP15_8 r2 r4 + aP15_9 r3 r4) = (S15_0 r0 + S15_1 r1 + S15_2 r2 + S15_3 r3 + S15_4 r4) - 1 * (N15_0 r0 r1 + N15_1 r0 r2 + N15_2 r0 r3 + N15_3 r0 r4 + N15_4 r1 r2 + N15_5 r1 r3 + N15_6 r1 r4 + N15_7 r2 r3 + N15_8 r2 r4 + N15_9 r3 r4) := by
    simp only [aS15_0, aS15_1, aS15_2, aS15_3, aS15_4, aP15_0, aP15_1, aP15_2, aP15_3, aP15_4, aP15_5, aP15_6, aP15_7, aP15_8, aP15_9, L15_0, L15_1, L15_2, L15_3, L15_4]
    ring
  have bS0 : aS15_0 r0 ≤ MS15_0 := CaseSplit.le_mxr (aS15_0) 10 r0 (by omega)
  have bS1 : aS15_1 r1 ≤ MS15_1 := CaseSplit.le_mxr (aS15_1) 12 r1 (by omega)
  have bS2 : aS15_2 r2 ≤ MS15_2 := CaseSplit.le_mxr (aS15_2) 16 r2 (by omega)
  have bS3 : aS15_3 r3 ≤ MS15_3 := CaseSplit.le_mxr (aS15_3) 18 r3 (by omega)
  have bS4 : aS15_4 r4 ≤ MS15_4 := CaseSplit.le_mxr (aS15_4) 22 r4 (by omega)
  have bP0 : aP15_0 r0 r1 ≤ MP15_0 := CaseSplit.le_mxr2 (aP15_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP15_1 r0 r2 ≤ MP15_1 := CaseSplit.le_mxr2 (aP15_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP15_2 r0 r3 ≤ MP15_2 := CaseSplit.le_mxr2 (aP15_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP15_3 r0 r4 ≤ MP15_3 := CaseSplit.le_mxr2 (aP15_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP15_4 r1 r2 ≤ MP15_4 := CaseSplit.le_mxr2 (aP15_4) 12 16 r1 r2 (by omega) (by omega)
  have bP5 : aP15_5 r1 r3 ≤ MP15_5 := CaseSplit.le_mxr2 (aP15_5) 12 18 r1 r3 (by omega) (by omega)
  have bP6 : aP15_6 r1 r4 ≤ MP15_6 := CaseSplit.le_mxr2 (aP15_6) 12 22 r1 r4 (by omega) (by omega)
  have bP7 : aP15_7 r2 r3 ≤ MP15_7 := CaseSplit.le_mxr2 (aP15_7) 16 18 r2 r3 (by omega) (by omega)
  have bP8 : aP15_8 r2 r4 ≤ MP15_8 := CaseSplit.le_mxr2 (aP15_8) 16 22 r2 r4 (by omega) (by omega)
  have bP9 : aP15_9 r3 r4 ≤ MP15_9 := CaseSplit.le_mxr2 (aP15_9) 18 22 r3 r4 (by omega) (by omega)
  have hrhs : rhs15 = (∑ t ∈ Finset.range n15, w15 t) + 1 * (n15 : ℤ) := rfl
  have hc := cert15
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, bS0, bS1, bS2, bS3, bS4, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9]

end IncCert23
