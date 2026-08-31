/-
INCREMENT-WIDTH CERTIFICATE, step 19->23, case 4 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_19_23.json, which re-derives every number
from the primes alone).

Machine 23, INCREMENT width 39 = F_2(19) + s_min(23) = 31 + 8,
held gears [5, 7] at phases [0, 4].  Free gears [11, 13, 17, 19, 23].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 3.
-/
import IncCert23B

namespace IncCert23

/-! ### case 4: held gears at phases [0, 4] -/

def p4 : List ℕ := [0, 3, 5, 7, 8, 10, 12, 13, 15, 17, 20, 22, 27, 28, 33, 35, 38]
def q4 (t : ℕ) : ℕ := p4.getD t 0
def n4 : ℕ := 17
def yl4 : List ℤ := [0, 0, 2, 0, 0, 2, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
def w4 (t : ℕ) : ℤ := yl4.getD t 0
def ul4 : List ℤ := [(-1), (-1), 0, (-1), 0, (-1), 0, (-1), 0, (-5), (-4), 0, (-1), 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, (-2), (-2), 0, (-2), 0, (-2), (-2), (-2), 0, 0, 0, (-2), (-2), (-2), (-2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), (-1), 0, (-1), (-1), 0, (-1), (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, (-1), (-3), (-3), (-3), (-1), (-3), (-3), 0, (-3), (-3), (-3), (-3), (-1), (-3), 0, 0, (-3), (-3), (-3), (-3), (-3), (-3), 0, 1, 0, 0, 0, 3, 0, 0, 1, 3, 0, 0, 8, 5, 11, 11, 11, 6, 11, 11, 5, 11, 11, 8, 11, 8, 8, 11, 11, (-15), (-11), (-11), (-11), (-11), (-11), (-11), (-11), (-11), (-11), (-11), (-11), (-11), 4, 4, 4, 4, 4, (-1), 4, 2, 4, 2, 1, 4, 1, 4, 4, 4, 1, 4, 4, (-4), (-7), (-4), (-4), (-4), (-4), (-5), (-4), (-4), (-4), (-4), (-4), (-4), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 11, 6, 3, 11, 6, 6, 11, 3, 11, 6, 11, 11, 3, 11, 11, 11, 11, 3, 9, 9, 3, 11, 5, 0, 2, 5, 5, 5, 5, 5, 5, 5, 5, 2, 5, 5, 3, 3, 5, 2, 0]
def u4 (k : ℕ) : ℤ := ul4.getD k 0

def c4_0 (r t : ℕ) : Bool := gb11 r (q4 t)
def c4_1 (r t : ℕ) : Bool := gb13 r (q4 t)
def c4_2 (r t : ℕ) : Bool := gb17 r (q4 t)
def c4_3 (r t : ℕ) : Bool := gb19 r (q4 t)
def c4_4 (r t : ℕ) : Bool := gb23 r (q4 t)

def S4_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (w4 t + 3) * (if c4_0 r t then 1 else 0)
def S4_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (w4 t + 3) * (if c4_1 r t then 1 else 0)
def S4_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (w4 t + 3) * (if c4_2 r t then 1 else 0)
def S4_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (w4 t + 3) * (if c4_3 r t then 1 else 0)
def S4_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (w4 t + 3) * (if c4_4 r t then 1 else 0)

def L4_0 (r : ℕ) : ℤ := u4 (13 + r) + u4 (41 + r) + u4 (71 + r) + u4 (105 + r)
def L4_1 (r : ℕ) : ℤ := u4 (0 + r) + u4 (133 + r) + u4 (165 + r) + u4 (201 + r)
def L4_2 (r : ℕ) : ℤ := u4 (24 + r) + u4 (116 + r) + u4 (233 + r) + u4 (273 + r)
def L4_3 (r : ℕ) : ℤ := u4 (52 + r) + u4 (146 + r) + u4 (214 + r) + u4 (313 + r)
def L4_4 (r : ℕ) : ℤ := u4 (82 + r) + u4 (178 + r) + u4 (250 + r) + u4 (290 + r)

def aS4_0 (r : ℕ) : ℤ := S4_0 r - L4_0 r
def MS4_0 : ℤ := CaseSplit.mxr (aS4_0) 10
def aS4_1 (r : ℕ) : ℤ := S4_1 r - L4_1 r
def MS4_1 : ℤ := CaseSplit.mxr (aS4_1) 12
def aS4_2 (r : ℕ) : ℤ := S4_2 r - L4_2 r
def MS4_2 : ℤ := CaseSplit.mxr (aS4_2) 16
def aS4_3 (r : ℕ) : ℤ := S4_3 r - L4_3 r
def MS4_3 : ℤ := CaseSplit.mxr (aS4_3) 18
def aS4_4 (r : ℕ) : ℤ := S4_4 r - L4_4 r
def MS4_4 : ℤ := CaseSplit.mxr (aS4_4) 22

def N4_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_0 ra t && c4_1 rb t then 1 else 0)
def aP4_0 (ra rb : ℕ) : ℤ := -(3) * N4_0 ra rb + u4 (0 + rb) + u4 (13 + ra)
def MP4_0 : ℤ := CaseSplit.mxr2 (aP4_0) 10 12
def N4_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_0 ra t && c4_2 rb t then 1 else 0)
def aP4_1 (ra rb : ℕ) : ℤ := -(3) * N4_1 ra rb + u4 (24 + rb) + u4 (41 + ra)
def MP4_1 : ℤ := CaseSplit.mxr2 (aP4_1) 10 16
def N4_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_0 ra t && c4_3 rb t then 1 else 0)
def aP4_2 (ra rb : ℕ) : ℤ := -(3) * N4_2 ra rb + u4 (52 + rb) + u4 (71 + ra)
def MP4_2 : ℤ := CaseSplit.mxr2 (aP4_2) 10 18
def N4_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_0 ra t && c4_4 rb t then 1 else 0)
def aP4_3 (ra rb : ℕ) : ℤ := -(3) * N4_3 ra rb + u4 (82 + rb) + u4 (105 + ra)
def MP4_3 : ℤ := CaseSplit.mxr2 (aP4_3) 10 22
def P4_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_1 ra t && c4_2 rb t then 1 else 0)
def C4_4 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_1 ra t && c4_2 rb t && c4_0 s t then 1 else 0)
def M4_4 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C4_4 ra rb) 10
def E4_4 : List ℕ := [21, 27, 111, 117, 136, 147, 190, 201]
def N4_4 (ra rb : ℕ) : ℤ := if E4_4.contains (ra * 17 + rb) = true then P4_4 ra rb - M4_4 ra rb else 0
def aP4_4 (ra rb : ℕ) : ℤ := -(3) * N4_4 ra rb + u4 (116 + rb) + u4 (133 + ra)
def MP4_4 : ℤ := CaseSplit.mxr2 (aP4_4) 12 16
def P4_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_1 ra t && c4_3 rb t then 1 else 0)
def C4_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_1 ra t && c4_3 rb t && c4_0 s t then 1 else 0)
def M4_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C4_5 ra rb) 10
def E4_5 : List ℕ := [7, 38, 41, 78, 91, 114, 154, 167, 178, 212]
def N4_5 (ra rb : ℕ) : ℤ := if E4_5.contains (ra * 19 + rb) = true then P4_5 ra rb - M4_5 ra rb else 0
def aP4_5 (ra rb : ℕ) : ℤ := -(3) * N4_5 ra rb + u4 (146 + rb) + u4 (165 + ra)
def MP4_5 : ℤ := CaseSplit.mxr2 (aP4_5) 12 18
def P4_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_1 ra t && c4_4 rb t then 1 else 0)
def C4_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n4, (if c4_1 ra t && c4_4 rb t && c4_0 s t then 1 else 0)
def M4_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C4_6 ra rb) 10
def E4_6 : List ℕ := []
def N4_6 (ra rb : ℕ) : ℤ := if E4_6.contains (ra * 23 + rb) = true then P4_6 ra rb - M4_6 ra rb else 0
def aP4_6 (ra rb : ℕ) : ℤ := -(3) * N4_6 ra rb + u4 (178 + rb) + u4 (201 + ra)
def MP4_6 : ℤ := CaseSplit.mxr2 (aP4_6) 12 22
def N4_7 (_ra _rb : ℕ) : ℤ := 0
def aP4_7 (ra rb : ℕ) : ℤ := -(3) * N4_7 ra rb + u4 (214 + rb) + u4 (233 + ra)
def MP4_7 : ℤ := CaseSplit.mxr2 (aP4_7) 16 18
def N4_8 (_ra _rb : ℕ) : ℤ := 0
def aP4_8 (ra rb : ℕ) : ℤ := -(3) * N4_8 ra rb + u4 (250 + rb) + u4 (273 + ra)
def MP4_8 : ℤ := CaseSplit.mxr2 (aP4_8) 16 22
def N4_9 (_ra _rb : ℕ) : ℤ := 0
def aP4_9 (ra rb : ℕ) : ℤ := -(3) * N4_9 ra rb + u4 (290 + rb) + u4 (313 + ra)
def MP4_9 : ℤ := CaseSplit.mxr2 (aP4_9) 18 22

def rhs4 : ℤ := (∑ t ∈ Finset.range n4, w4 t) + 3 * (n4 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn4 : ∀ t, t < n4 → (0 : ℤ) ≤ w4 t := by decide
theorem plt4 : ∀ t, t < n4 → q4 t < 39 := by decide
theorem pfree4_5 : ∀ t, t < n4 → gb5 0 (q4 t) = false := by decide
theorem pfree4_7 : ∀ t, t < n4 → gb7 4 (q4 t) = false := by decide
theorem MSv4_0 : MS4_0 = 11 := by decide +kernel
theorem MSv4_1 : MS4_1 = 27 := by decide +kernel
theorem MSv4_2 : MS4_2 = 0 := by decide +kernel
theorem MSv4_3 : MS4_3 = 0 := by decide +kernel
theorem MSv4_4 : MS4_4 = 0 := by decide +kernel
theorem MPv4_0 : MP4_0 = 0 := by decide +kernel
theorem MPv4_1 : MP4_1 = 0 := by decide +kernel
theorem MPv4_2 : MP4_2 = 0 := by decide +kernel
theorem MPv4_3 : MP4_3 = 0 := by decide +kernel
theorem MPv4_4 : MP4_4 = 0 := by decide +kernel
theorem MPv4_5 : MP4_5 = 0 := by decide +kernel
theorem MPv4_6 : MP4_6 = 0 := by decide +kernel
theorem MPv4_7 : MP4_7 = 0 := by decide +kernel
theorem MPv4_8 : MP4_8 = 0 := by decide +kernel
theorem MPv4_9 : MP4_9 = 16 := by decide +kernel
theorem rhsv4 : rhs4 = 58 := by decide +kernel

/-- **The case-4 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 4/3.
    (Scaled by the common denominator 3: 54 < 58.) -/
theorem cert4 : MS4_0 + MS4_1 + MS4_2 + MS4_3 + MS4_4 + MP4_0 + MP4_1 + MP4_2 + MP4_3 + MP4_4 + MP4_5 + MP4_6 + MP4_7 + MP4_8 + MP4_9 < rhs4 := by
  rw [MSv4_0, MSv4_1, MSv4_2, MSv4_3, MSv4_4, MPv4_0, MPv4_1, MPv4_2, MPv4_3, MPv4_4, MPv4_5, MPv4_6, MPv4_7, MPv4_8, MPv4_9, rhsv4]
  decide

def Dg4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := (if c4_0 r0 t then 1 else 0) + (if c4_1 r1 t then 1 else 0) + (if c4_2 r2 t then 1 else 0) + (if c4_3 r3 t then 1 else 0) + (if c4_4 r4 t then 1 else 0)
def Wl4_0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c4_0 r0 t && c4_1 r1 t then 1 else 0
def Wl4_1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c4_0 r0 t && c4_2 r2 t then 1 else 0
def Wl4_2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c4_0 r0 t && c4_3 r3 t then 1 else 0
def Wl4_3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c4_0 r0 t && c4_4 r4 t then 1 else 0
def Wl4_4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c4_0 r0 t && c4_1 r1 t && c4_2 r2 t then 1 else 0
def Wl4_5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c4_0 r0 t && c4_1 r1 t && c4_3 r3 t then 1 else 0
def Wl4_6 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c4_0 r0 t && c4_1 r1 t && c4_4 r4 t then 1 else 0
def Wl4_7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c4_0 r0 t && !c4_1 r1 t && c4_2 r2 t && c4_3 r3 t then 1 else 0
def Wl4_8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c4_0 r0 t && !c4_1 r1 t && c4_2 r2 t && c4_4 r4 t then 1 else 0
def Wl4_9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c4_0 r0 t && !c4_1 r1 t && !c4_2 r2 t && c4_3 r3 t && c4_4 r4 t then 1 else 0

/-- **No configuration blocks the whole window in case 4.** -/
theorem nocov4 {r0 r1 r2 r3 r4 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23)
    (hcov : ∀ t, t < n4 → (c4_0 r0 t || c4_1 r1 t || c4_2 r2 t || c4_3 r3 t || c4_4 r4 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n4, (1 : ℤ) + (Wl4_0 r0 r1 r2 r3 r4 t + Wl4_1 r0 r1 r2 r3 r4 t + Wl4_2 r0 r1 r2 r3 r4 t + Wl4_3 r0 r1 r2 r3 r4 t + Wl4_4 r0 r1 r2 r3 r4 t + Wl4_5 r0 r1 r2 r3 r4 t + Wl4_6 r0 r1 r2 r3 r4 t + Wl4_7 r0 r1 r2 r3 r4 t + Wl4_8 r0 r1 r2 r3 r4 t + Wl4_9 r0 r1 r2 r3 r4 t) ≤ Dg4 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Wl4_0, Wl4_1, Wl4_2, Wl4_3, Wl4_4, Wl4_5, Wl4_6, Wl4_7, Wl4_8, Wl4_9, Dg4]
    exact CaseSplit.lowest5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n4, (1 : ℤ) ≤ Dg4 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Dg4]
    exact CaseSplit.degpos5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n4 : ℤ) + ((∑ t ∈ Finset.range n4, Wl4_0 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n4, Wl4_1 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n4, Wl4_2 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n4, Wl4_3 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n4, Wl4_4 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n4, Wl4_5 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n4, Wl4_6 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n4, Wl4_7 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n4, Wl4_8 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n4, Wl4_9 r0 r1 r2 r3 r4 t)) ≤ ∑ t ∈ Finset.range n4, Dg4 r0 r1 r2 r3 r4 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N4_0 r0 r1 ≤ ∑ t ∈ Finset.range n4, Wl4_0 r0 r1 r2 r3 r4 t := by
    simp only [N4_0, Wl4_0, le_refl]
  have hn1 : N4_1 r0 r2 ≤ ∑ t ∈ Finset.range n4, Wl4_1 r0 r1 r2 r3 r4 t := by
    simp only [N4_1, Wl4_1, le_refl]
  have hn2 : N4_2 r0 r3 ≤ ∑ t ∈ Finset.range n4, Wl4_2 r0 r1 r2 r3 r4 t := by
    simp only [N4_2, Wl4_2, le_refl]
  have hn3 : N4_3 r0 r4 ≤ ∑ t ∈ Finset.range n4, Wl4_3 r0 r1 r2 r3 r4 t := by
    simp only [N4_3, Wl4_3, le_refl]
  have hn4 : N4_4 r1 r2 ≤ ∑ t ∈ Finset.range n4, Wl4_4 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n4, Wl4_4 r0 r1 r2 r3 r4 t
        = (if c4_1 r1 t && c4_2 r2 t then (1:ℤ) else 0)
          - (if c4_1 r1 t && c4_2 r2 t && c4_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl4_4]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n4, Wl4_4 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl4_4]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n4, Wl4_4 r0 r1 r2 r3 r4 t
        = P4_4 r1 r2 - C4_4 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P4_4, C4_4]
    have hm : C4_4 r1 r2 r0 ≤ M4_4 r1 r2 :=
      CaseSplit.le_mxr (C4_4 r1 r2) 10 r0 (by omega)
    simp only [N4_4]
    split
    · rw [hL]; omega
    · exact hnn
  have hn5 : N4_5 r1 r3 ≤ ∑ t ∈ Finset.range n4, Wl4_5 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n4, Wl4_5 r0 r1 r2 r3 r4 t
        = (if c4_1 r1 t && c4_3 r3 t then (1:ℤ) else 0)
          - (if c4_1 r1 t && c4_3 r3 t && c4_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl4_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n4, Wl4_5 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl4_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n4, Wl4_5 r0 r1 r2 r3 r4 t
        = P4_5 r1 r3 - C4_5 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P4_5, C4_5]
    have hm : C4_5 r1 r3 r0 ≤ M4_5 r1 r3 :=
      CaseSplit.le_mxr (C4_5 r1 r3) 10 r0 (by omega)
    simp only [N4_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N4_6 r1 r4 ≤ ∑ t ∈ Finset.range n4, Wl4_6 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n4, Wl4_6 r0 r1 r2 r3 r4 t
        = (if c4_1 r1 t && c4_4 r4 t then (1:ℤ) else 0)
          - (if c4_1 r1 t && c4_4 r4 t && c4_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl4_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n4, Wl4_6 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl4_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n4, Wl4_6 r0 r1 r2 r3 r4 t
        = P4_6 r1 r4 - C4_6 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P4_6, C4_6]
    have hm : C4_6 r1 r4 r0 ≤ M4_6 r1 r4 :=
      CaseSplit.le_mxr (C4_6 r1 r4) 10 r0 (by omega)
    simp only [N4_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N4_7 r2 r3 ≤ ∑ t ∈ Finset.range n4, Wl4_7 r0 r1 r2 r3 r4 t := by
    simp only [N4_7]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl4_7]
    exact CaseSplit.ind_nonneg _
  have hn8 : N4_8 r2 r4 ≤ ∑ t ∈ Finset.range n4, Wl4_8 r0 r1 r2 r3 r4 t := by
    simp only [N4_8]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl4_8]
    exact CaseSplit.ind_nonneg _
  have hn9 : N4_9 r3 r4 ≤ ∑ t ∈ Finset.range n4, Wl4_9 r0 r1 r2 r3 r4 t := by
    simp only [N4_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl4_9]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n4, (w4 t + 3) * Dg4 r0 r1 r2 r3 r4 t = S4_0 r0 + S4_1 r1 + S4_2 r2 + S4_3 r3 + S4_4 r4 := by
    simp only [S4_0, S4_1, S4_2, S4_3, S4_4, Dg4, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n4, (w4 t + 3) * Dg4 r0 r1 r2 r3 r4 t
      = (∑ t ∈ Finset.range n4, w4 t * Dg4 r0 r1 r2 r3 r4 t)
        + 3 * (∑ t ∈ Finset.range n4, Dg4 r0 r1 r2 r3 r4 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n4, w4 t)
      ≤ ∑ t ∈ Finset.range n4, w4 t * Dg4 r0 r1 r2 r3 r4 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg4 r0 r1 r2 r3 r4 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w4 t := wnn4 t (Finset.mem_range.mp ht)
    calc w4 t = w4 t * 1 := (mul_one _).symm
      _ ≤ w4 t * Dg4 r0 r1 r2 r3 r4 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS4_0 r0 + aS4_1 r1 + aS4_2 r2 + aS4_3 r3 + aS4_4 r4) + (aP4_0 r0 r1 + aP4_1 r0 r2 + aP4_2 r0 r3 + aP4_3 r0 r4 + aP4_4 r1 r2 + aP4_5 r1 r3 + aP4_6 r1 r4 + aP4_7 r2 r3 + aP4_8 r2 r4 + aP4_9 r3 r4) = (S4_0 r0 + S4_1 r1 + S4_2 r2 + S4_3 r3 + S4_4 r4) - 3 * (N4_0 r0 r1 + N4_1 r0 r2 + N4_2 r0 r3 + N4_3 r0 r4 + N4_4 r1 r2 + N4_5 r1 r3 + N4_6 r1 r4 + N4_7 r2 r3 + N4_8 r2 r4 + N4_9 r3 r4) := by
    simp only [aS4_0, aS4_1, aS4_2, aS4_3, aS4_4, aP4_0, aP4_1, aP4_2, aP4_3, aP4_4, aP4_5, aP4_6, aP4_7, aP4_8, aP4_9, L4_0, L4_1, L4_2, L4_3, L4_4]
    ring
  have bS0 : aS4_0 r0 ≤ MS4_0 := CaseSplit.le_mxr (aS4_0) 10 r0 (by omega)
  have bS1 : aS4_1 r1 ≤ MS4_1 := CaseSplit.le_mxr (aS4_1) 12 r1 (by omega)
  have bS2 : aS4_2 r2 ≤ MS4_2 := CaseSplit.le_mxr (aS4_2) 16 r2 (by omega)
  have bS3 : aS4_3 r3 ≤ MS4_3 := CaseSplit.le_mxr (aS4_3) 18 r3 (by omega)
  have bS4 : aS4_4 r4 ≤ MS4_4 := CaseSplit.le_mxr (aS4_4) 22 r4 (by omega)
  have bP0 : aP4_0 r0 r1 ≤ MP4_0 := CaseSplit.le_mxr2 (aP4_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP4_1 r0 r2 ≤ MP4_1 := CaseSplit.le_mxr2 (aP4_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP4_2 r0 r3 ≤ MP4_2 := CaseSplit.le_mxr2 (aP4_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP4_3 r0 r4 ≤ MP4_3 := CaseSplit.le_mxr2 (aP4_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP4_4 r1 r2 ≤ MP4_4 := CaseSplit.le_mxr2 (aP4_4) 12 16 r1 r2 (by omega) (by omega)
  have bP5 : aP4_5 r1 r3 ≤ MP4_5 := CaseSplit.le_mxr2 (aP4_5) 12 18 r1 r3 (by omega) (by omega)
  have bP6 : aP4_6 r1 r4 ≤ MP4_6 := CaseSplit.le_mxr2 (aP4_6) 12 22 r1 r4 (by omega) (by omega)
  have bP7 : aP4_7 r2 r3 ≤ MP4_7 := CaseSplit.le_mxr2 (aP4_7) 16 18 r2 r3 (by omega) (by omega)
  have bP8 : aP4_8 r2 r4 ≤ MP4_8 := CaseSplit.le_mxr2 (aP4_8) 16 22 r2 r4 (by omega) (by omega)
  have bP9 : aP4_9 r3 r4 ≤ MP4_9 := CaseSplit.le_mxr2 (aP4_9) 18 22 r3 r4 (by omega) (by omega)
  have hrhs : rhs4 = (∑ t ∈ Finset.range n4, w4 t) + 3 * (n4 : ℤ) := rfl
  have hc := cert4
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, bS0, bS1, bS2, bS3, bS4, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9]

end IncCert23
