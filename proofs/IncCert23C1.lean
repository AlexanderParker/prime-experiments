/-
INCREMENT-WIDTH CERTIFICATE, step 19->23, case 1 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_19_23.json, which re-derives every number
from the primes alone).

Machine 23, INCREMENT width 39 = F_2(19) + s_min(23) = 31 + 8,
held gears [5, 7] at phases [0, 1].  Free gears [11, 13, 17, 19, 23].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 1.
-/
import IncCert23B

namespace IncCert23

/-! ### case 1: held gears at phases [0, 1] -/

def p1 : List ℕ := [2, 3, 8, 10, 13, 15, 17, 18, 20, 22, 23, 25, 27, 30, 32, 37, 38]
def q1 (t : ℕ) : ℕ := p1.getD t 0
def n1 : ℕ := 17
def yl1 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
def w1 (t : ℕ) : ℤ := yl1.getD t 0
def ul1 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, (-1), (-1), (-1), 0, 0, 0, (-1), (-1), 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 1, 1, 3, 2, 3, 3, 1, 2, 3, 3, 3, 1, 2, 2, 3, (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), 2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 3, 0, 3, 3, 1, 2, 0, 2, 2, 1, 3, 1, 1, 1, 3, 3, 1, 2, 3, 0, 2, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
def u1 (k : ℕ) : ℤ := ul1.getD k 0

def c1_0 (r t : ℕ) : Bool := gb11 r (q1 t)
def c1_1 (r t : ℕ) : Bool := gb13 r (q1 t)
def c1_2 (r t : ℕ) : Bool := gb17 r (q1 t)
def c1_3 (r t : ℕ) : Bool := gb19 r (q1 t)
def c1_4 (r t : ℕ) : Bool := gb23 r (q1 t)

def S1_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (w1 t + 1) * (if c1_0 r t then 1 else 0)
def S1_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (w1 t + 1) * (if c1_1 r t then 1 else 0)
def S1_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (w1 t + 1) * (if c1_2 r t then 1 else 0)
def S1_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (w1 t + 1) * (if c1_3 r t then 1 else 0)
def S1_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (w1 t + 1) * (if c1_4 r t then 1 else 0)

def L1_0 (r : ℕ) : ℤ := u1 (13 + r) + u1 (41 + r) + u1 (71 + r) + u1 (105 + r)
def L1_1 (r : ℕ) : ℤ := u1 (0 + r) + u1 (133 + r) + u1 (165 + r) + u1 (201 + r)
def L1_2 (r : ℕ) : ℤ := u1 (24 + r) + u1 (116 + r) + u1 (233 + r) + u1 (273 + r)
def L1_3 (r : ℕ) : ℤ := u1 (52 + r) + u1 (146 + r) + u1 (214 + r) + u1 (313 + r)
def L1_4 (r : ℕ) : ℤ := u1 (82 + r) + u1 (178 + r) + u1 (250 + r) + u1 (290 + r)

def aS1_0 (r : ℕ) : ℤ := S1_0 r - L1_0 r
def MS1_0 : ℤ := CaseSplit.mxr (aS1_0) 10
def aS1_1 (r : ℕ) : ℤ := S1_1 r - L1_1 r
def MS1_1 : ℤ := CaseSplit.mxr (aS1_1) 12
def aS1_2 (r : ℕ) : ℤ := S1_2 r - L1_2 r
def MS1_2 : ℤ := CaseSplit.mxr (aS1_2) 16
def aS1_3 (r : ℕ) : ℤ := S1_3 r - L1_3 r
def MS1_3 : ℤ := CaseSplit.mxr (aS1_3) 18
def aS1_4 (r : ℕ) : ℤ := S1_4 r - L1_4 r
def MS1_4 : ℤ := CaseSplit.mxr (aS1_4) 22

def N1_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (if c1_0 ra t && c1_1 rb t then 1 else 0)
def aP1_0 (ra rb : ℕ) : ℤ := -(1) * N1_0 ra rb + u1 (0 + rb) + u1 (13 + ra)
def MP1_0 : ℤ := CaseSplit.mxr2 (aP1_0) 10 12
def N1_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (if c1_0 ra t && c1_2 rb t then 1 else 0)
def aP1_1 (ra rb : ℕ) : ℤ := -(1) * N1_1 ra rb + u1 (24 + rb) + u1 (41 + ra)
def MP1_1 : ℤ := CaseSplit.mxr2 (aP1_1) 10 16
def N1_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (if c1_0 ra t && c1_3 rb t then 1 else 0)
def aP1_2 (ra rb : ℕ) : ℤ := -(1) * N1_2 ra rb + u1 (52 + rb) + u1 (71 + ra)
def MP1_2 : ℤ := CaseSplit.mxr2 (aP1_2) 10 18
def N1_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (if c1_0 ra t && c1_4 rb t then 1 else 0)
def aP1_3 (ra rb : ℕ) : ℤ := -(1) * N1_3 ra rb + u1 (82 + rb) + u1 (105 + ra)
def MP1_3 : ℤ := CaseSplit.mxr2 (aP1_3) 10 22
def P1_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (if c1_1 ra t && c1_2 rb t then 1 else 0)
def C1_4 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (if c1_1 ra t && c1_2 rb t && c1_0 s t then 1 else 0)
def M1_4 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C1_4 ra rb) 10
def E1_4 : List ℕ := [21, 27, 57, 63, 68, 79, 136, 147, 158, 169, 188, 194]
def N1_4 (ra rb : ℕ) : ℤ := if E1_4.contains (ra * 17 + rb) = true then P1_4 ra rb - M1_4 ra rb else 0
def aP1_4 (ra rb : ℕ) : ℤ := -(1) * N1_4 ra rb + u1 (116 + rb) + u1 (133 + ra)
def MP1_4 : ℤ := CaseSplit.mxr2 (aP1_4) 12 16
def P1_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (if c1_1 ra t && c1_3 rb t then 1 else 0)
def C1_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (if c1_1 ra t && c1_3 rb t && c1_0 s t then 1 else 0)
def M1_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C1_5 ra rb) 10
def E1_5 : List ℕ := [1, 31, 73, 107, 138, 172, 214, 244]
def N1_5 (ra rb : ℕ) : ℤ := if E1_5.contains (ra * 19 + rb) = true then P1_5 ra rb - M1_5 ra rb else 0
def aP1_5 (ra rb : ℕ) : ℤ := -(1) * N1_5 ra rb + u1 (146 + rb) + u1 (165 + ra)
def MP1_5 : ℤ := CaseSplit.mxr2 (aP1_5) 12 18
def P1_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (if c1_1 ra t && c1_4 rb t then 1 else 0)
def C1_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n1, (if c1_1 ra t && c1_4 rb t && c1_0 s t then 1 else 0)
def M1_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C1_6 ra rb) 10
def E1_6 : List ℕ := []
def N1_6 (ra rb : ℕ) : ℤ := if E1_6.contains (ra * 23 + rb) = true then P1_6 ra rb - M1_6 ra rb else 0
def aP1_6 (ra rb : ℕ) : ℤ := -(1) * N1_6 ra rb + u1 (178 + rb) + u1 (201 + ra)
def MP1_6 : ℤ := CaseSplit.mxr2 (aP1_6) 12 22
def N1_7 (_ra _rb : ℕ) : ℤ := 0
def aP1_7 (ra rb : ℕ) : ℤ := -(1) * N1_7 ra rb + u1 (214 + rb) + u1 (233 + ra)
def MP1_7 : ℤ := CaseSplit.mxr2 (aP1_7) 16 18
def N1_8 (_ra _rb : ℕ) : ℤ := 0
def aP1_8 (ra rb : ℕ) : ℤ := -(1) * N1_8 ra rb + u1 (250 + rb) + u1 (273 + ra)
def MP1_8 : ℤ := CaseSplit.mxr2 (aP1_8) 16 22
def N1_9 (_ra _rb : ℕ) : ℤ := 0
def aP1_9 (ra rb : ℕ) : ℤ := -(1) * N1_9 ra rb + u1 (290 + rb) + u1 (313 + ra)
def MP1_9 : ℤ := CaseSplit.mxr2 (aP1_9) 18 22

def rhs1 : ℤ := (∑ t ∈ Finset.range n1, w1 t) + 1 * (n1 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn1 : ∀ t, t < n1 → (0 : ℤ) ≤ w1 t := by decide
theorem plt1 : ∀ t, t < n1 → q1 t < 39 := by decide
theorem pfree1_5 : ∀ t, t < n1 → gb5 0 (q1 t) = false := by decide
theorem pfree1_7 : ∀ t, t < n1 → gb7 1 (q1 t) = false := by decide
theorem MSv1_0 : MS1_0 = 4 := by decide +kernel
theorem MSv1_1 : MS1_1 = 8 := by decide +kernel
theorem MSv1_2 : MS1_2 = 0 := by decide +kernel
theorem MSv1_3 : MS1_3 = 0 := by decide +kernel
theorem MSv1_4 : MS1_4 = 0 := by decide +kernel
theorem MPv1_0 : MP1_0 = 0 := by decide +kernel
theorem MPv1_1 : MP1_1 = 0 := by decide +kernel
theorem MPv1_2 : MP1_2 = 0 := by decide +kernel
theorem MPv1_3 : MP1_3 = 0 := by decide +kernel
theorem MPv1_4 : MP1_4 = 0 := by decide +kernel
theorem MPv1_5 : MP1_5 = 0 := by decide +kernel
theorem MPv1_6 : MP1_6 = 0 := by decide +kernel
theorem MPv1_7 : MP1_7 = 0 := by decide +kernel
theorem MPv1_8 : MP1_8 = 0 := by decide +kernel
theorem MPv1_9 : MP1_9 = 4 := by decide +kernel
theorem rhsv1 : rhs1 = 19 := by decide +kernel

/-- **The case-1 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 3/1.
    (Scaled by the common denominator 1: 16 < 19.) -/
theorem cert1 : MS1_0 + MS1_1 + MS1_2 + MS1_3 + MS1_4 + MP1_0 + MP1_1 + MP1_2 + MP1_3 + MP1_4 + MP1_5 + MP1_6 + MP1_7 + MP1_8 + MP1_9 < rhs1 := by
  rw [MSv1_0, MSv1_1, MSv1_2, MSv1_3, MSv1_4, MPv1_0, MPv1_1, MPv1_2, MPv1_3, MPv1_4, MPv1_5, MPv1_6, MPv1_7, MPv1_8, MPv1_9, rhsv1]
  decide

def Dg1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := (if c1_0 r0 t then 1 else 0) + (if c1_1 r1 t then 1 else 0) + (if c1_2 r2 t then 1 else 0) + (if c1_3 r3 t then 1 else 0) + (if c1_4 r4 t then 1 else 0)
def Wl1_0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c1_0 r0 t && c1_1 r1 t then 1 else 0
def Wl1_1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c1_0 r0 t && c1_2 r2 t then 1 else 0
def Wl1_2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c1_0 r0 t && c1_3 r3 t then 1 else 0
def Wl1_3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c1_0 r0 t && c1_4 r4 t then 1 else 0
def Wl1_4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c1_0 r0 t && c1_1 r1 t && c1_2 r2 t then 1 else 0
def Wl1_5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c1_0 r0 t && c1_1 r1 t && c1_3 r3 t then 1 else 0
def Wl1_6 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c1_0 r0 t && c1_1 r1 t && c1_4 r4 t then 1 else 0
def Wl1_7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c1_0 r0 t && !c1_1 r1 t && c1_2 r2 t && c1_3 r3 t then 1 else 0
def Wl1_8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c1_0 r0 t && !c1_1 r1 t && c1_2 r2 t && c1_4 r4 t then 1 else 0
def Wl1_9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c1_0 r0 t && !c1_1 r1 t && !c1_2 r2 t && c1_3 r3 t && c1_4 r4 t then 1 else 0

/-- **No configuration blocks the whole window in case 1.** -/
theorem nocov1 {r0 r1 r2 r3 r4 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23)
    (hcov : ∀ t, t < n1 → (c1_0 r0 t || c1_1 r1 t || c1_2 r2 t || c1_3 r3 t || c1_4 r4 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n1, (1 : ℤ) + (Wl1_0 r0 r1 r2 r3 r4 t + Wl1_1 r0 r1 r2 r3 r4 t + Wl1_2 r0 r1 r2 r3 r4 t + Wl1_3 r0 r1 r2 r3 r4 t + Wl1_4 r0 r1 r2 r3 r4 t + Wl1_5 r0 r1 r2 r3 r4 t + Wl1_6 r0 r1 r2 r3 r4 t + Wl1_7 r0 r1 r2 r3 r4 t + Wl1_8 r0 r1 r2 r3 r4 t + Wl1_9 r0 r1 r2 r3 r4 t) ≤ Dg1 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Wl1_0, Wl1_1, Wl1_2, Wl1_3, Wl1_4, Wl1_5, Wl1_6, Wl1_7, Wl1_8, Wl1_9, Dg1]
    exact CaseSplit.lowest5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n1, (1 : ℤ) ≤ Dg1 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Dg1]
    exact CaseSplit.degpos5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n1 : ℤ) + ((∑ t ∈ Finset.range n1, Wl1_0 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n1, Wl1_1 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n1, Wl1_2 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n1, Wl1_3 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n1, Wl1_4 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n1, Wl1_5 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n1, Wl1_6 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n1, Wl1_7 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n1, Wl1_8 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n1, Wl1_9 r0 r1 r2 r3 r4 t)) ≤ ∑ t ∈ Finset.range n1, Dg1 r0 r1 r2 r3 r4 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N1_0 r0 r1 ≤ ∑ t ∈ Finset.range n1, Wl1_0 r0 r1 r2 r3 r4 t := by
    simp only [N1_0, Wl1_0, le_refl]
  have hn1 : N1_1 r0 r2 ≤ ∑ t ∈ Finset.range n1, Wl1_1 r0 r1 r2 r3 r4 t := by
    simp only [N1_1, Wl1_1, le_refl]
  have hn2 : N1_2 r0 r3 ≤ ∑ t ∈ Finset.range n1, Wl1_2 r0 r1 r2 r3 r4 t := by
    simp only [N1_2, Wl1_2, le_refl]
  have hn3 : N1_3 r0 r4 ≤ ∑ t ∈ Finset.range n1, Wl1_3 r0 r1 r2 r3 r4 t := by
    simp only [N1_3, Wl1_3, le_refl]
  have hn4 : N1_4 r1 r2 ≤ ∑ t ∈ Finset.range n1, Wl1_4 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n1, Wl1_4 r0 r1 r2 r3 r4 t
        = (if c1_1 r1 t && c1_2 r2 t then (1:ℤ) else 0)
          - (if c1_1 r1 t && c1_2 r2 t && c1_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl1_4]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n1, Wl1_4 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl1_4]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n1, Wl1_4 r0 r1 r2 r3 r4 t
        = P1_4 r1 r2 - C1_4 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P1_4, C1_4]
    have hm : C1_4 r1 r2 r0 ≤ M1_4 r1 r2 :=
      CaseSplit.le_mxr (C1_4 r1 r2) 10 r0 (by omega)
    simp only [N1_4]
    split
    · rw [hL]; omega
    · exact hnn
  have hn5 : N1_5 r1 r3 ≤ ∑ t ∈ Finset.range n1, Wl1_5 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n1, Wl1_5 r0 r1 r2 r3 r4 t
        = (if c1_1 r1 t && c1_3 r3 t then (1:ℤ) else 0)
          - (if c1_1 r1 t && c1_3 r3 t && c1_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl1_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n1, Wl1_5 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl1_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n1, Wl1_5 r0 r1 r2 r3 r4 t
        = P1_5 r1 r3 - C1_5 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P1_5, C1_5]
    have hm : C1_5 r1 r3 r0 ≤ M1_5 r1 r3 :=
      CaseSplit.le_mxr (C1_5 r1 r3) 10 r0 (by omega)
    simp only [N1_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N1_6 r1 r4 ≤ ∑ t ∈ Finset.range n1, Wl1_6 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n1, Wl1_6 r0 r1 r2 r3 r4 t
        = (if c1_1 r1 t && c1_4 r4 t then (1:ℤ) else 0)
          - (if c1_1 r1 t && c1_4 r4 t && c1_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl1_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n1, Wl1_6 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl1_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n1, Wl1_6 r0 r1 r2 r3 r4 t
        = P1_6 r1 r4 - C1_6 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P1_6, C1_6]
    have hm : C1_6 r1 r4 r0 ≤ M1_6 r1 r4 :=
      CaseSplit.le_mxr (C1_6 r1 r4) 10 r0 (by omega)
    simp only [N1_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N1_7 r2 r3 ≤ ∑ t ∈ Finset.range n1, Wl1_7 r0 r1 r2 r3 r4 t := by
    simp only [N1_7]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl1_7]
    exact CaseSplit.ind_nonneg _
  have hn8 : N1_8 r2 r4 ≤ ∑ t ∈ Finset.range n1, Wl1_8 r0 r1 r2 r3 r4 t := by
    simp only [N1_8]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl1_8]
    exact CaseSplit.ind_nonneg _
  have hn9 : N1_9 r3 r4 ≤ ∑ t ∈ Finset.range n1, Wl1_9 r0 r1 r2 r3 r4 t := by
    simp only [N1_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl1_9]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n1, (w1 t + 1) * Dg1 r0 r1 r2 r3 r4 t = S1_0 r0 + S1_1 r1 + S1_2 r2 + S1_3 r3 + S1_4 r4 := by
    simp only [S1_0, S1_1, S1_2, S1_3, S1_4, Dg1, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n1, (w1 t + 1) * Dg1 r0 r1 r2 r3 r4 t
      = (∑ t ∈ Finset.range n1, w1 t * Dg1 r0 r1 r2 r3 r4 t)
        + 1 * (∑ t ∈ Finset.range n1, Dg1 r0 r1 r2 r3 r4 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n1, w1 t)
      ≤ ∑ t ∈ Finset.range n1, w1 t * Dg1 r0 r1 r2 r3 r4 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg1 r0 r1 r2 r3 r4 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w1 t := wnn1 t (Finset.mem_range.mp ht)
    calc w1 t = w1 t * 1 := (mul_one _).symm
      _ ≤ w1 t * Dg1 r0 r1 r2 r3 r4 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS1_0 r0 + aS1_1 r1 + aS1_2 r2 + aS1_3 r3 + aS1_4 r4) + (aP1_0 r0 r1 + aP1_1 r0 r2 + aP1_2 r0 r3 + aP1_3 r0 r4 + aP1_4 r1 r2 + aP1_5 r1 r3 + aP1_6 r1 r4 + aP1_7 r2 r3 + aP1_8 r2 r4 + aP1_9 r3 r4) = (S1_0 r0 + S1_1 r1 + S1_2 r2 + S1_3 r3 + S1_4 r4) - 1 * (N1_0 r0 r1 + N1_1 r0 r2 + N1_2 r0 r3 + N1_3 r0 r4 + N1_4 r1 r2 + N1_5 r1 r3 + N1_6 r1 r4 + N1_7 r2 r3 + N1_8 r2 r4 + N1_9 r3 r4) := by
    simp only [aS1_0, aS1_1, aS1_2, aS1_3, aS1_4, aP1_0, aP1_1, aP1_2, aP1_3, aP1_4, aP1_5, aP1_6, aP1_7, aP1_8, aP1_9, L1_0, L1_1, L1_2, L1_3, L1_4]
    ring
  have bS0 : aS1_0 r0 ≤ MS1_0 := CaseSplit.le_mxr (aS1_0) 10 r0 (by omega)
  have bS1 : aS1_1 r1 ≤ MS1_1 := CaseSplit.le_mxr (aS1_1) 12 r1 (by omega)
  have bS2 : aS1_2 r2 ≤ MS1_2 := CaseSplit.le_mxr (aS1_2) 16 r2 (by omega)
  have bS3 : aS1_3 r3 ≤ MS1_3 := CaseSplit.le_mxr (aS1_3) 18 r3 (by omega)
  have bS4 : aS1_4 r4 ≤ MS1_4 := CaseSplit.le_mxr (aS1_4) 22 r4 (by omega)
  have bP0 : aP1_0 r0 r1 ≤ MP1_0 := CaseSplit.le_mxr2 (aP1_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP1_1 r0 r2 ≤ MP1_1 := CaseSplit.le_mxr2 (aP1_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP1_2 r0 r3 ≤ MP1_2 := CaseSplit.le_mxr2 (aP1_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP1_3 r0 r4 ≤ MP1_3 := CaseSplit.le_mxr2 (aP1_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP1_4 r1 r2 ≤ MP1_4 := CaseSplit.le_mxr2 (aP1_4) 12 16 r1 r2 (by omega) (by omega)
  have bP5 : aP1_5 r1 r3 ≤ MP1_5 := CaseSplit.le_mxr2 (aP1_5) 12 18 r1 r3 (by omega) (by omega)
  have bP6 : aP1_6 r1 r4 ≤ MP1_6 := CaseSplit.le_mxr2 (aP1_6) 12 22 r1 r4 (by omega) (by omega)
  have bP7 : aP1_7 r2 r3 ≤ MP1_7 := CaseSplit.le_mxr2 (aP1_7) 16 18 r2 r3 (by omega) (by omega)
  have bP8 : aP1_8 r2 r4 ≤ MP1_8 := CaseSplit.le_mxr2 (aP1_8) 16 22 r2 r4 (by omega) (by omega)
  have bP9 : aP1_9 r3 r4 ≤ MP1_9 := CaseSplit.le_mxr2 (aP1_9) 18 22 r3 r4 (by omega) (by omega)
  have hrhs : rhs1 = (∑ t ∈ Finset.range n1, w1 t) + 1 * (n1 : ℤ) := rfl
  have hc := cert1
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, bS0, bS1, bS2, bS3, bS4, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9]

end IncCert23
