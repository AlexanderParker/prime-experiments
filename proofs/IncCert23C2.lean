/-
INCREMENT-WIDTH CERTIFICATE, step 19->23, case 2 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_19_23.json, which re-derives every number
from the primes alone).

Machine 23, INCREMENT width 39 = F_2(19) + s_min(23) = 31 + 8,
held gears [5, 7] at phases [0, 2].  Free gears [11, 13, 17, 19, 23].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 1.
-/
import IncCert23B

namespace IncCert23

/-! ### case 2: held gears at phases [0, 2] -/

def p2 : List ℕ := [0, 2, 3, 5, 7, 8, 10, 12, 15, 17, 22, 23, 28, 30, 33, 35, 37, 38]
def q2 (t : ℕ) : ℕ := p2.getD t 0
def n2 : ℕ := 18
def yl2 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def w2 (t : ℕ) : ℤ := yl2.getD t 0
def ul2 : List ℤ := [0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 3, 2, 2, 3, 3, 1, 1, 2, 2, 3, 2, 3, 2, 3, 3, 3, 2, (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), 2, 2, 2, 1, 1, 2, 1, 2, 1, 0, 2, 1, 1, 2, 1, 0, 1, 1, 2, (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 4, 1, 4, 3, 1, 3, 1, 3, 2, 2, 4, 2, 3, 3, 2, 4, 1, 3, 4, 1, 3, 1, 0, (-1), 1, 1, 0, 1, 1, 1, 1, (-1), 1, 1, 1, 1, 1, 1, 1, 0]
def u2 (k : ℕ) : ℤ := ul2.getD k 0

def c2_0 (r t : ℕ) : Bool := gb11 r (q2 t)
def c2_1 (r t : ℕ) : Bool := gb13 r (q2 t)
def c2_2 (r t : ℕ) : Bool := gb17 r (q2 t)
def c2_3 (r t : ℕ) : Bool := gb19 r (q2 t)
def c2_4 (r t : ℕ) : Bool := gb23 r (q2 t)

def S2_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n2, (w2 t + 1) * (if c2_0 r t then 1 else 0)
def S2_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n2, (w2 t + 1) * (if c2_1 r t then 1 else 0)
def S2_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n2, (w2 t + 1) * (if c2_2 r t then 1 else 0)
def S2_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n2, (w2 t + 1) * (if c2_3 r t then 1 else 0)
def S2_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n2, (w2 t + 1) * (if c2_4 r t then 1 else 0)

def L2_0 (r : ℕ) : ℤ := u2 (13 + r) + u2 (41 + r) + u2 (71 + r) + u2 (105 + r)
def L2_1 (r : ℕ) : ℤ := u2 (0 + r) + u2 (133 + r) + u2 (165 + r) + u2 (201 + r)
def L2_2 (r : ℕ) : ℤ := u2 (24 + r) + u2 (116 + r) + u2 (233 + r) + u2 (273 + r)
def L2_3 (r : ℕ) : ℤ := u2 (52 + r) + u2 (146 + r) + u2 (214 + r) + u2 (313 + r)
def L2_4 (r : ℕ) : ℤ := u2 (82 + r) + u2 (178 + r) + u2 (250 + r) + u2 (290 + r)

def aS2_0 (r : ℕ) : ℤ := S2_0 r - L2_0 r
def MS2_0 : ℤ := CaseSplit.mxr (aS2_0) 10
def aS2_1 (r : ℕ) : ℤ := S2_1 r - L2_1 r
def MS2_1 : ℤ := CaseSplit.mxr (aS2_1) 12
def aS2_2 (r : ℕ) : ℤ := S2_2 r - L2_2 r
def MS2_2 : ℤ := CaseSplit.mxr (aS2_2) 16
def aS2_3 (r : ℕ) : ℤ := S2_3 r - L2_3 r
def MS2_3 : ℤ := CaseSplit.mxr (aS2_3) 18
def aS2_4 (r : ℕ) : ℤ := S2_4 r - L2_4 r
def MS2_4 : ℤ := CaseSplit.mxr (aS2_4) 22

def N2_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n2, (if c2_0 ra t && c2_1 rb t then 1 else 0)
def aP2_0 (ra rb : ℕ) : ℤ := -(1) * N2_0 ra rb + u2 (0 + rb) + u2 (13 + ra)
def MP2_0 : ℤ := CaseSplit.mxr2 (aP2_0) 10 12
def N2_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n2, (if c2_0 ra t && c2_2 rb t then 1 else 0)
def aP2_1 (ra rb : ℕ) : ℤ := -(1) * N2_1 ra rb + u2 (24 + rb) + u2 (41 + ra)
def MP2_1 : ℤ := CaseSplit.mxr2 (aP2_1) 10 16
def N2_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n2, (if c2_0 ra t && c2_3 rb t then 1 else 0)
def aP2_2 (ra rb : ℕ) : ℤ := -(1) * N2_2 ra rb + u2 (52 + rb) + u2 (71 + ra)
def MP2_2 : ℤ := CaseSplit.mxr2 (aP2_2) 10 18
def N2_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n2, (if c2_0 ra t && c2_4 rb t then 1 else 0)
def aP2_3 (ra rb : ℕ) : ℤ := -(1) * N2_3 ra rb + u2 (82 + rb) + u2 (105 + ra)
def MP2_3 : ℤ := CaseSplit.mxr2 (aP2_3) 10 22
def P2_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n2, (if c2_1 ra t && c2_2 rb t then 1 else 0)
def C2_4 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n2, (if c2_1 ra t && c2_2 rb t && c2_0 s t then 1 else 0)
def M2_4 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C2_4 ra rb) 10
def E2_4 : List ℕ := [111, 117, 190, 201]
def N2_4 (ra rb : ℕ) : ℤ := if E2_4.contains (ra * 17 + rb) = true then P2_4 ra rb - M2_4 ra rb else 0
def aP2_4 (ra rb : ℕ) : ℤ := -(1) * N2_4 ra rb + u2 (116 + rb) + u2 (133 + ra)
def MP2_4 : ℤ := CaseSplit.mxr2 (aP2_4) 12 16
def P2_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n2, (if c2_1 ra t && c2_3 rb t then 1 else 0)
def C2_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n2, (if c2_1 ra t && c2_3 rb t && c2_0 s t then 1 else 0)
def M2_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C2_5 ra rb) 10
def E2_5 : List ℕ := [1, 7, 31, 38, 107, 114, 138, 172, 178, 214]
def N2_5 (ra rb : ℕ) : ℤ := if E2_5.contains (ra * 19 + rb) = true then P2_5 ra rb - M2_5 ra rb else 0
def aP2_5 (ra rb : ℕ) : ℤ := -(1) * N2_5 ra rb + u2 (146 + rb) + u2 (165 + ra)
def MP2_5 : ℤ := CaseSplit.mxr2 (aP2_5) 12 18
def P2_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n2, (if c2_1 ra t && c2_4 rb t then 1 else 0)
def C2_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n2, (if c2_1 ra t && c2_4 rb t && c2_0 s t then 1 else 0)
def M2_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C2_6 ra rb) 10
def E2_6 : List ℕ := []
def N2_6 (ra rb : ℕ) : ℤ := if E2_6.contains (ra * 23 + rb) = true then P2_6 ra rb - M2_6 ra rb else 0
def aP2_6 (ra rb : ℕ) : ℤ := -(1) * N2_6 ra rb + u2 (178 + rb) + u2 (201 + ra)
def MP2_6 : ℤ := CaseSplit.mxr2 (aP2_6) 12 22
def N2_7 (_ra _rb : ℕ) : ℤ := 0
def aP2_7 (ra rb : ℕ) : ℤ := -(1) * N2_7 ra rb + u2 (214 + rb) + u2 (233 + ra)
def MP2_7 : ℤ := CaseSplit.mxr2 (aP2_7) 16 18
def N2_8 (_ra _rb : ℕ) : ℤ := 0
def aP2_8 (ra rb : ℕ) : ℤ := -(1) * N2_8 ra rb + u2 (250 + rb) + u2 (273 + ra)
def MP2_8 : ℤ := CaseSplit.mxr2 (aP2_8) 16 22
def N2_9 (_ra _rb : ℕ) : ℤ := 0
def aP2_9 (ra rb : ℕ) : ℤ := -(1) * N2_9 ra rb + u2 (290 + rb) + u2 (313 + ra)
def MP2_9 : ℤ := CaseSplit.mxr2 (aP2_9) 18 22

def rhs2 : ℤ := (∑ t ∈ Finset.range n2, w2 t) + 1 * (n2 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn2 : ∀ t, t < n2 → (0 : ℤ) ≤ w2 t := by decide
theorem plt2 : ∀ t, t < n2 → q2 t < 39 := by decide
theorem pfree2_5 : ∀ t, t < n2 → gb5 0 (q2 t) = false := by decide
theorem pfree2_7 : ∀ t, t < n2 → gb7 2 (q2 t) = false := by decide
theorem MSv2_0 : MS2_0 = 3 := by decide +kernel
theorem MSv2_1 : MS2_1 = 9 := by decide +kernel
theorem MSv2_2 : MS2_2 = 0 := by decide +kernel
theorem MSv2_3 : MS2_3 = 0 := by decide +kernel
theorem MSv2_4 : MS2_4 = 0 := by decide +kernel
theorem MPv2_0 : MP2_0 = 0 := by decide +kernel
theorem MPv2_1 : MP2_1 = 0 := by decide +kernel
theorem MPv2_2 : MP2_2 = 0 := by decide +kernel
theorem MPv2_3 : MP2_3 = 0 := by decide +kernel
theorem MPv2_4 : MP2_4 = 0 := by decide +kernel
theorem MPv2_5 : MP2_5 = 0 := by decide +kernel
theorem MPv2_6 : MP2_6 = 0 := by decide +kernel
theorem MPv2_7 : MP2_7 = 0 := by decide +kernel
theorem MPv2_8 : MP2_8 = 0 := by decide +kernel
theorem MPv2_9 : MP2_9 = 5 := by decide +kernel
theorem rhsv2 : rhs2 = 18 := by decide +kernel

/-- **The case-2 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/1.
    (Scaled by the common denominator 1: 17 < 18.) -/
theorem cert2 : MS2_0 + MS2_1 + MS2_2 + MS2_3 + MS2_4 + MP2_0 + MP2_1 + MP2_2 + MP2_3 + MP2_4 + MP2_5 + MP2_6 + MP2_7 + MP2_8 + MP2_9 < rhs2 := by
  rw [MSv2_0, MSv2_1, MSv2_2, MSv2_3, MSv2_4, MPv2_0, MPv2_1, MPv2_2, MPv2_3, MPv2_4, MPv2_5, MPv2_6, MPv2_7, MPv2_8, MPv2_9, rhsv2]
  decide

def Dg2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := (if c2_0 r0 t then 1 else 0) + (if c2_1 r1 t then 1 else 0) + (if c2_2 r2 t then 1 else 0) + (if c2_3 r3 t then 1 else 0) + (if c2_4 r4 t then 1 else 0)
def Wl2_0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c2_0 r0 t && c2_1 r1 t then 1 else 0
def Wl2_1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c2_0 r0 t && c2_2 r2 t then 1 else 0
def Wl2_2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c2_0 r0 t && c2_3 r3 t then 1 else 0
def Wl2_3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c2_0 r0 t && c2_4 r4 t then 1 else 0
def Wl2_4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c2_0 r0 t && c2_1 r1 t && c2_2 r2 t then 1 else 0
def Wl2_5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c2_0 r0 t && c2_1 r1 t && c2_3 r3 t then 1 else 0
def Wl2_6 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c2_0 r0 t && c2_1 r1 t && c2_4 r4 t then 1 else 0
def Wl2_7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c2_0 r0 t && !c2_1 r1 t && c2_2 r2 t && c2_3 r3 t then 1 else 0
def Wl2_8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c2_0 r0 t && !c2_1 r1 t && c2_2 r2 t && c2_4 r4 t then 1 else 0
def Wl2_9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c2_0 r0 t && !c2_1 r1 t && !c2_2 r2 t && c2_3 r3 t && c2_4 r4 t then 1 else 0

/-- **No configuration blocks the whole window in case 2.** -/
theorem nocov2 {r0 r1 r2 r3 r4 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23)
    (hcov : ∀ t, t < n2 → (c2_0 r0 t || c2_1 r1 t || c2_2 r2 t || c2_3 r3 t || c2_4 r4 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n2, (1 : ℤ) + (Wl2_0 r0 r1 r2 r3 r4 t + Wl2_1 r0 r1 r2 r3 r4 t + Wl2_2 r0 r1 r2 r3 r4 t + Wl2_3 r0 r1 r2 r3 r4 t + Wl2_4 r0 r1 r2 r3 r4 t + Wl2_5 r0 r1 r2 r3 r4 t + Wl2_6 r0 r1 r2 r3 r4 t + Wl2_7 r0 r1 r2 r3 r4 t + Wl2_8 r0 r1 r2 r3 r4 t + Wl2_9 r0 r1 r2 r3 r4 t) ≤ Dg2 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Wl2_0, Wl2_1, Wl2_2, Wl2_3, Wl2_4, Wl2_5, Wl2_6, Wl2_7, Wl2_8, Wl2_9, Dg2]
    exact CaseSplit.lowest5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n2, (1 : ℤ) ≤ Dg2 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Dg2]
    exact CaseSplit.degpos5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n2 : ℤ) + ((∑ t ∈ Finset.range n2, Wl2_0 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n2, Wl2_1 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n2, Wl2_2 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n2, Wl2_3 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n2, Wl2_4 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n2, Wl2_5 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n2, Wl2_6 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n2, Wl2_7 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n2, Wl2_8 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n2, Wl2_9 r0 r1 r2 r3 r4 t)) ≤ ∑ t ∈ Finset.range n2, Dg2 r0 r1 r2 r3 r4 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N2_0 r0 r1 ≤ ∑ t ∈ Finset.range n2, Wl2_0 r0 r1 r2 r3 r4 t := by
    simp only [N2_0, Wl2_0, le_refl]
  have hn1 : N2_1 r0 r2 ≤ ∑ t ∈ Finset.range n2, Wl2_1 r0 r1 r2 r3 r4 t := by
    simp only [N2_1, Wl2_1, le_refl]
  have hn2 : N2_2 r0 r3 ≤ ∑ t ∈ Finset.range n2, Wl2_2 r0 r1 r2 r3 r4 t := by
    simp only [N2_2, Wl2_2, le_refl]
  have hn3 : N2_3 r0 r4 ≤ ∑ t ∈ Finset.range n2, Wl2_3 r0 r1 r2 r3 r4 t := by
    simp only [N2_3, Wl2_3, le_refl]
  have hn4 : N2_4 r1 r2 ≤ ∑ t ∈ Finset.range n2, Wl2_4 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n2, Wl2_4 r0 r1 r2 r3 r4 t
        = (if c2_1 r1 t && c2_2 r2 t then (1:ℤ) else 0)
          - (if c2_1 r1 t && c2_2 r2 t && c2_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl2_4]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n2, Wl2_4 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl2_4]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n2, Wl2_4 r0 r1 r2 r3 r4 t
        = P2_4 r1 r2 - C2_4 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P2_4, C2_4]
    have hm : C2_4 r1 r2 r0 ≤ M2_4 r1 r2 :=
      CaseSplit.le_mxr (C2_4 r1 r2) 10 r0 (by omega)
    simp only [N2_4]
    split
    · rw [hL]; omega
    · exact hnn
  have hn5 : N2_5 r1 r3 ≤ ∑ t ∈ Finset.range n2, Wl2_5 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n2, Wl2_5 r0 r1 r2 r3 r4 t
        = (if c2_1 r1 t && c2_3 r3 t then (1:ℤ) else 0)
          - (if c2_1 r1 t && c2_3 r3 t && c2_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl2_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n2, Wl2_5 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl2_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n2, Wl2_5 r0 r1 r2 r3 r4 t
        = P2_5 r1 r3 - C2_5 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P2_5, C2_5]
    have hm : C2_5 r1 r3 r0 ≤ M2_5 r1 r3 :=
      CaseSplit.le_mxr (C2_5 r1 r3) 10 r0 (by omega)
    simp only [N2_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N2_6 r1 r4 ≤ ∑ t ∈ Finset.range n2, Wl2_6 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n2, Wl2_6 r0 r1 r2 r3 r4 t
        = (if c2_1 r1 t && c2_4 r4 t then (1:ℤ) else 0)
          - (if c2_1 r1 t && c2_4 r4 t && c2_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl2_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n2, Wl2_6 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl2_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n2, Wl2_6 r0 r1 r2 r3 r4 t
        = P2_6 r1 r4 - C2_6 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P2_6, C2_6]
    have hm : C2_6 r1 r4 r0 ≤ M2_6 r1 r4 :=
      CaseSplit.le_mxr (C2_6 r1 r4) 10 r0 (by omega)
    simp only [N2_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N2_7 r2 r3 ≤ ∑ t ∈ Finset.range n2, Wl2_7 r0 r1 r2 r3 r4 t := by
    simp only [N2_7]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl2_7]
    exact CaseSplit.ind_nonneg _
  have hn8 : N2_8 r2 r4 ≤ ∑ t ∈ Finset.range n2, Wl2_8 r0 r1 r2 r3 r4 t := by
    simp only [N2_8]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl2_8]
    exact CaseSplit.ind_nonneg _
  have hn9 : N2_9 r3 r4 ≤ ∑ t ∈ Finset.range n2, Wl2_9 r0 r1 r2 r3 r4 t := by
    simp only [N2_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl2_9]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n2, (w2 t + 1) * Dg2 r0 r1 r2 r3 r4 t = S2_0 r0 + S2_1 r1 + S2_2 r2 + S2_3 r3 + S2_4 r4 := by
    simp only [S2_0, S2_1, S2_2, S2_3, S2_4, Dg2, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n2, (w2 t + 1) * Dg2 r0 r1 r2 r3 r4 t
      = (∑ t ∈ Finset.range n2, w2 t * Dg2 r0 r1 r2 r3 r4 t)
        + 1 * (∑ t ∈ Finset.range n2, Dg2 r0 r1 r2 r3 r4 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n2, w2 t)
      ≤ ∑ t ∈ Finset.range n2, w2 t * Dg2 r0 r1 r2 r3 r4 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg2 r0 r1 r2 r3 r4 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w2 t := wnn2 t (Finset.mem_range.mp ht)
    calc w2 t = w2 t * 1 := (mul_one _).symm
      _ ≤ w2 t * Dg2 r0 r1 r2 r3 r4 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS2_0 r0 + aS2_1 r1 + aS2_2 r2 + aS2_3 r3 + aS2_4 r4) + (aP2_0 r0 r1 + aP2_1 r0 r2 + aP2_2 r0 r3 + aP2_3 r0 r4 + aP2_4 r1 r2 + aP2_5 r1 r3 + aP2_6 r1 r4 + aP2_7 r2 r3 + aP2_8 r2 r4 + aP2_9 r3 r4) = (S2_0 r0 + S2_1 r1 + S2_2 r2 + S2_3 r3 + S2_4 r4) - 1 * (N2_0 r0 r1 + N2_1 r0 r2 + N2_2 r0 r3 + N2_3 r0 r4 + N2_4 r1 r2 + N2_5 r1 r3 + N2_6 r1 r4 + N2_7 r2 r3 + N2_8 r2 r4 + N2_9 r3 r4) := by
    simp only [aS2_0, aS2_1, aS2_2, aS2_3, aS2_4, aP2_0, aP2_1, aP2_2, aP2_3, aP2_4, aP2_5, aP2_6, aP2_7, aP2_8, aP2_9, L2_0, L2_1, L2_2, L2_3, L2_4]
    ring
  have bS0 : aS2_0 r0 ≤ MS2_0 := CaseSplit.le_mxr (aS2_0) 10 r0 (by omega)
  have bS1 : aS2_1 r1 ≤ MS2_1 := CaseSplit.le_mxr (aS2_1) 12 r1 (by omega)
  have bS2 : aS2_2 r2 ≤ MS2_2 := CaseSplit.le_mxr (aS2_2) 16 r2 (by omega)
  have bS3 : aS2_3 r3 ≤ MS2_3 := CaseSplit.le_mxr (aS2_3) 18 r3 (by omega)
  have bS4 : aS2_4 r4 ≤ MS2_4 := CaseSplit.le_mxr (aS2_4) 22 r4 (by omega)
  have bP0 : aP2_0 r0 r1 ≤ MP2_0 := CaseSplit.le_mxr2 (aP2_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP2_1 r0 r2 ≤ MP2_1 := CaseSplit.le_mxr2 (aP2_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP2_2 r0 r3 ≤ MP2_2 := CaseSplit.le_mxr2 (aP2_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP2_3 r0 r4 ≤ MP2_3 := CaseSplit.le_mxr2 (aP2_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP2_4 r1 r2 ≤ MP2_4 := CaseSplit.le_mxr2 (aP2_4) 12 16 r1 r2 (by omega) (by omega)
  have bP5 : aP2_5 r1 r3 ≤ MP2_5 := CaseSplit.le_mxr2 (aP2_5) 12 18 r1 r3 (by omega) (by omega)
  have bP6 : aP2_6 r1 r4 ≤ MP2_6 := CaseSplit.le_mxr2 (aP2_6) 12 22 r1 r4 (by omega) (by omega)
  have bP7 : aP2_7 r2 r3 ≤ MP2_7 := CaseSplit.le_mxr2 (aP2_7) 16 18 r2 r3 (by omega) (by omega)
  have bP8 : aP2_8 r2 r4 ≤ MP2_8 := CaseSplit.le_mxr2 (aP2_8) 16 22 r2 r4 (by omega) (by omega)
  have bP9 : aP2_9 r3 r4 ≤ MP2_9 := CaseSplit.le_mxr2 (aP2_9) 18 22 r3 r4 (by omega) (by omega)
  have hrhs : rhs2 = (∑ t ∈ Finset.range n2, w2 t) + 1 * (n2 : ℤ) := rfl
  have hc := cert2
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, bS0, bS1, bS2, bS3, bS4, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9]

end IncCert23
