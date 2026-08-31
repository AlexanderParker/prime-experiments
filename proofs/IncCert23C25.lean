/-
INCREMENT-WIDTH CERTIFICATE, step 19->23, case 25 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_19_23.json, which re-derives every number
from the primes alone).

Machine 23, INCREMENT width 39 = F_2(19) + s_min(23) = 31 + 8,
held gears [5, 7] at phases [3, 4].  Free gears [11, 13, 17, 19, 23].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 1.
-/
import IncCert23B

namespace IncCert23

/-! ### case 25: held gears at phases [3, 4] -/

def p25 : List ℕ := [0, 5, 7, 10, 12, 14, 15, 17, 19, 20, 22, 24, 27, 29, 34, 35]
def q25 (t : ℕ) : ℕ := p25.getD t 0
def n25 : ℕ := 16
def yl25 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
def w25 (t : ℕ) : ℤ := yl25.getD t 0
def ul25 : List ℤ := [(-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), (-1), 0, 0, (-1), (-1), (-1), 0, 0, 0, (-1), (-1), (-1), 0, 0, 0, (-1), 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), (-1), (-1), (-1), 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 2, 3, (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 3, 1, 2, 2, 3, 3, 1, 3, 3, 2, 2, 1, 3, 3, 3, 3, 1, 2, 2, 2, 3, 2, 3, 1, 2, 1, 2, 0, 0, 2, 1, 1, 0, 0, 2, 1, 0, 0, 1, 2, 1, 0]
def u25 (k : ℕ) : ℤ := ul25.getD k 0

def c25_0 (r t : ℕ) : Bool := gb11 r (q25 t)
def c25_1 (r t : ℕ) : Bool := gb13 r (q25 t)
def c25_2 (r t : ℕ) : Bool := gb17 r (q25 t)
def c25_3 (r t : ℕ) : Bool := gb19 r (q25 t)
def c25_4 (r t : ℕ) : Bool := gb23 r (q25 t)

def S25_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (w25 t + 1) * (if c25_0 r t then 1 else 0)
def S25_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (w25 t + 1) * (if c25_1 r t then 1 else 0)
def S25_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (w25 t + 1) * (if c25_2 r t then 1 else 0)
def S25_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (w25 t + 1) * (if c25_3 r t then 1 else 0)
def S25_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (w25 t + 1) * (if c25_4 r t then 1 else 0)

def L25_0 (r : ℕ) : ℤ := u25 (13 + r) + u25 (41 + r) + u25 (71 + r) + u25 (105 + r)
def L25_1 (r : ℕ) : ℤ := u25 (0 + r) + u25 (133 + r) + u25 (165 + r) + u25 (201 + r)
def L25_2 (r : ℕ) : ℤ := u25 (24 + r) + u25 (116 + r) + u25 (233 + r) + u25 (273 + r)
def L25_3 (r : ℕ) : ℤ := u25 (52 + r) + u25 (146 + r) + u25 (214 + r) + u25 (313 + r)
def L25_4 (r : ℕ) : ℤ := u25 (82 + r) + u25 (178 + r) + u25 (250 + r) + u25 (290 + r)

def aS25_0 (r : ℕ) : ℤ := S25_0 r - L25_0 r
def MS25_0 : ℤ := CaseSplit.mxr (aS25_0) 10
def aS25_1 (r : ℕ) : ℤ := S25_1 r - L25_1 r
def MS25_1 : ℤ := CaseSplit.mxr (aS25_1) 12
def aS25_2 (r : ℕ) : ℤ := S25_2 r - L25_2 r
def MS25_2 : ℤ := CaseSplit.mxr (aS25_2) 16
def aS25_3 (r : ℕ) : ℤ := S25_3 r - L25_3 r
def MS25_3 : ℤ := CaseSplit.mxr (aS25_3) 18
def aS25_4 (r : ℕ) : ℤ := S25_4 r - L25_4 r
def MS25_4 : ℤ := CaseSplit.mxr (aS25_4) 22

def N25_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (if c25_0 ra t && c25_1 rb t then 1 else 0)
def aP25_0 (ra rb : ℕ) : ℤ := -(1) * N25_0 ra rb + u25 (0 + rb) + u25 (13 + ra)
def MP25_0 : ℤ := CaseSplit.mxr2 (aP25_0) 10 12
def N25_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (if c25_0 ra t && c25_2 rb t then 1 else 0)
def aP25_1 (ra rb : ℕ) : ℤ := -(1) * N25_1 ra rb + u25 (24 + rb) + u25 (41 + ra)
def MP25_1 : ℤ := CaseSplit.mxr2 (aP25_1) 10 16
def N25_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (if c25_0 ra t && c25_3 rb t then 1 else 0)
def aP25_2 (ra rb : ℕ) : ℤ := -(1) * N25_2 ra rb + u25 (52 + rb) + u25 (71 + ra)
def MP25_2 : ℤ := CaseSplit.mxr2 (aP25_2) 10 18
def N25_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (if c25_0 ra t && c25_4 rb t then 1 else 0)
def aP25_3 (ra rb : ℕ) : ℤ := -(1) * N25_3 ra rb + u25 (82 + rb) + u25 (105 + ra)
def MP25_3 : ℤ := CaseSplit.mxr2 (aP25_3) 10 22
def P25_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (if c25_1 ra t && c25_2 rb t then 1 else 0)
def C25_4 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (if c25_1 ra t && c25_2 rb t && c25_0 s t then 1 else 0)
def M25_4 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C25_4 ra rb) 10
def E25_4 : List ℕ := [21, 27, 75, 81, 111, 117, 122, 133, 190, 201, 206, 212]
def N25_4 (ra rb : ℕ) : ℤ := if E25_4.contains (ra * 17 + rb) = true then P25_4 ra rb - M25_4 ra rb else 0
def aP25_4 (ra rb : ℕ) : ℤ := -(1) * N25_4 ra rb + u25 (116 + rb) + u25 (133 + ra)
def MP25_4 : ℤ := CaseSplit.mxr2 (aP25_4) 12 16
def P25_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (if c25_1 ra t && c25_3 rb t then 1 else 0)
def C25_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (if c25_1 ra t && c25_3 rb t && c25_0 s t then 1 else 0)
def M25_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C25_5 ra rb) 10
def E25_5 : List ℕ := [27, 38, 91, 114, 167, 198]
def N25_5 (ra rb : ℕ) : ℤ := if E25_5.contains (ra * 19 + rb) = true then P25_5 ra rb - M25_5 ra rb else 0
def aP25_5 (ra rb : ℕ) : ℤ := -(1) * N25_5 ra rb + u25 (146 + rb) + u25 (165 + ra)
def MP25_5 : ℤ := CaseSplit.mxr2 (aP25_5) 12 18
def P25_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (if c25_1 ra t && c25_4 rb t then 1 else 0)
def C25_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n25, (if c25_1 ra t && c25_4 rb t && c25_0 s t then 1 else 0)
def M25_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C25_6 ra rb) 10
def E25_6 : List ℕ := []
def N25_6 (ra rb : ℕ) : ℤ := if E25_6.contains (ra * 23 + rb) = true then P25_6 ra rb - M25_6 ra rb else 0
def aP25_6 (ra rb : ℕ) : ℤ := -(1) * N25_6 ra rb + u25 (178 + rb) + u25 (201 + ra)
def MP25_6 : ℤ := CaseSplit.mxr2 (aP25_6) 12 22
def N25_7 (_ra _rb : ℕ) : ℤ := 0
def aP25_7 (ra rb : ℕ) : ℤ := -(1) * N25_7 ra rb + u25 (214 + rb) + u25 (233 + ra)
def MP25_7 : ℤ := CaseSplit.mxr2 (aP25_7) 16 18
def N25_8 (_ra _rb : ℕ) : ℤ := 0
def aP25_8 (ra rb : ℕ) : ℤ := -(1) * N25_8 ra rb + u25 (250 + rb) + u25 (273 + ra)
def MP25_8 : ℤ := CaseSplit.mxr2 (aP25_8) 16 22
def N25_9 (_ra _rb : ℕ) : ℤ := 0
def aP25_9 (ra rb : ℕ) : ℤ := -(1) * N25_9 ra rb + u25 (290 + rb) + u25 (313 + ra)
def MP25_9 : ℤ := CaseSplit.mxr2 (aP25_9) 18 22

def rhs25 : ℤ := (∑ t ∈ Finset.range n25, w25 t) + 1 * (n25 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn25 : ∀ t, t < n25 → (0 : ℤ) ≤ w25 t := by decide
theorem plt25 : ∀ t, t < n25 → q25 t < 39 := by decide
theorem pfree25_5 : ∀ t, t < n25 → gb5 3 (q25 t) = false := by decide
theorem pfree25_7 : ∀ t, t < n25 → gb7 4 (q25 t) = false := by decide
theorem MSv25_0 : MS25_0 = 3 := by decide +kernel
theorem MSv25_1 : MS25_1 = 7 := by decide +kernel
theorem MSv25_2 : MS25_2 = 0 := by decide +kernel
theorem MSv25_3 : MS25_3 = 0 := by decide +kernel
theorem MSv25_4 : MS25_4 = 0 := by decide +kernel
theorem MPv25_0 : MP25_0 = 0 := by decide +kernel
theorem MPv25_1 : MP25_1 = 0 := by decide +kernel
theorem MPv25_2 : MP25_2 = 0 := by decide +kernel
theorem MPv25_3 : MP25_3 = 0 := by decide +kernel
theorem MPv25_4 : MP25_4 = 0 := by decide +kernel
theorem MPv25_5 : MP25_5 = 0 := by decide +kernel
theorem MPv25_6 : MP25_6 = 0 := by decide +kernel
theorem MPv25_7 : MP25_7 = 0 := by decide +kernel
theorem MPv25_8 : MP25_8 = 0 := by decide +kernel
theorem MPv25_9 : MP25_9 = 5 := by decide +kernel
theorem rhsv25 : rhs25 = 17 := by decide +kernel

/-- **The case-25 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 2/1.
    (Scaled by the common denominator 1: 15 < 17.) -/
theorem cert25 : MS25_0 + MS25_1 + MS25_2 + MS25_3 + MS25_4 + MP25_0 + MP25_1 + MP25_2 + MP25_3 + MP25_4 + MP25_5 + MP25_6 + MP25_7 + MP25_8 + MP25_9 < rhs25 := by
  rw [MSv25_0, MSv25_1, MSv25_2, MSv25_3, MSv25_4, MPv25_0, MPv25_1, MPv25_2, MPv25_3, MPv25_4, MPv25_5, MPv25_6, MPv25_7, MPv25_8, MPv25_9, rhsv25]
  decide

def Dg25 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := (if c25_0 r0 t then 1 else 0) + (if c25_1 r1 t then 1 else 0) + (if c25_2 r2 t then 1 else 0) + (if c25_3 r3 t then 1 else 0) + (if c25_4 r4 t then 1 else 0)
def Wl25_0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c25_0 r0 t && c25_1 r1 t then 1 else 0
def Wl25_1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c25_0 r0 t && c25_2 r2 t then 1 else 0
def Wl25_2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c25_0 r0 t && c25_3 r3 t then 1 else 0
def Wl25_3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c25_0 r0 t && c25_4 r4 t then 1 else 0
def Wl25_4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c25_0 r0 t && c25_1 r1 t && c25_2 r2 t then 1 else 0
def Wl25_5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c25_0 r0 t && c25_1 r1 t && c25_3 r3 t then 1 else 0
def Wl25_6 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c25_0 r0 t && c25_1 r1 t && c25_4 r4 t then 1 else 0
def Wl25_7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c25_0 r0 t && !c25_1 r1 t && c25_2 r2 t && c25_3 r3 t then 1 else 0
def Wl25_8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c25_0 r0 t && !c25_1 r1 t && c25_2 r2 t && c25_4 r4 t then 1 else 0
def Wl25_9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c25_0 r0 t && !c25_1 r1 t && !c25_2 r2 t && c25_3 r3 t && c25_4 r4 t then 1 else 0

/-- **No configuration blocks the whole window in case 25.** -/
theorem nocov25 {r0 r1 r2 r3 r4 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23)
    (hcov : ∀ t, t < n25 → (c25_0 r0 t || c25_1 r1 t || c25_2 r2 t || c25_3 r3 t || c25_4 r4 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n25, (1 : ℤ) + (Wl25_0 r0 r1 r2 r3 r4 t + Wl25_1 r0 r1 r2 r3 r4 t + Wl25_2 r0 r1 r2 r3 r4 t + Wl25_3 r0 r1 r2 r3 r4 t + Wl25_4 r0 r1 r2 r3 r4 t + Wl25_5 r0 r1 r2 r3 r4 t + Wl25_6 r0 r1 r2 r3 r4 t + Wl25_7 r0 r1 r2 r3 r4 t + Wl25_8 r0 r1 r2 r3 r4 t + Wl25_9 r0 r1 r2 r3 r4 t) ≤ Dg25 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Wl25_0, Wl25_1, Wl25_2, Wl25_3, Wl25_4, Wl25_5, Wl25_6, Wl25_7, Wl25_8, Wl25_9, Dg25]
    exact CaseSplit.lowest5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n25, (1 : ℤ) ≤ Dg25 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Dg25]
    exact CaseSplit.degpos5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n25 : ℤ) + ((∑ t ∈ Finset.range n25, Wl25_0 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n25, Wl25_1 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n25, Wl25_2 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n25, Wl25_3 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n25, Wl25_4 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n25, Wl25_5 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n25, Wl25_6 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n25, Wl25_7 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n25, Wl25_8 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n25, Wl25_9 r0 r1 r2 r3 r4 t)) ≤ ∑ t ∈ Finset.range n25, Dg25 r0 r1 r2 r3 r4 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N25_0 r0 r1 ≤ ∑ t ∈ Finset.range n25, Wl25_0 r0 r1 r2 r3 r4 t := by
    simp only [N25_0, Wl25_0, le_refl]
  have hn1 : N25_1 r0 r2 ≤ ∑ t ∈ Finset.range n25, Wl25_1 r0 r1 r2 r3 r4 t := by
    simp only [N25_1, Wl25_1, le_refl]
  have hn2 : N25_2 r0 r3 ≤ ∑ t ∈ Finset.range n25, Wl25_2 r0 r1 r2 r3 r4 t := by
    simp only [N25_2, Wl25_2, le_refl]
  have hn3 : N25_3 r0 r4 ≤ ∑ t ∈ Finset.range n25, Wl25_3 r0 r1 r2 r3 r4 t := by
    simp only [N25_3, Wl25_3, le_refl]
  have hn4 : N25_4 r1 r2 ≤ ∑ t ∈ Finset.range n25, Wl25_4 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n25, Wl25_4 r0 r1 r2 r3 r4 t
        = (if c25_1 r1 t && c25_2 r2 t then (1:ℤ) else 0)
          - (if c25_1 r1 t && c25_2 r2 t && c25_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl25_4]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n25, Wl25_4 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl25_4]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n25, Wl25_4 r0 r1 r2 r3 r4 t
        = P25_4 r1 r2 - C25_4 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P25_4, C25_4]
    have hm : C25_4 r1 r2 r0 ≤ M25_4 r1 r2 :=
      CaseSplit.le_mxr (C25_4 r1 r2) 10 r0 (by omega)
    simp only [N25_4]
    split
    · rw [hL]; omega
    · exact hnn
  have hn5 : N25_5 r1 r3 ≤ ∑ t ∈ Finset.range n25, Wl25_5 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n25, Wl25_5 r0 r1 r2 r3 r4 t
        = (if c25_1 r1 t && c25_3 r3 t then (1:ℤ) else 0)
          - (if c25_1 r1 t && c25_3 r3 t && c25_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl25_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n25, Wl25_5 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl25_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n25, Wl25_5 r0 r1 r2 r3 r4 t
        = P25_5 r1 r3 - C25_5 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P25_5, C25_5]
    have hm : C25_5 r1 r3 r0 ≤ M25_5 r1 r3 :=
      CaseSplit.le_mxr (C25_5 r1 r3) 10 r0 (by omega)
    simp only [N25_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N25_6 r1 r4 ≤ ∑ t ∈ Finset.range n25, Wl25_6 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n25, Wl25_6 r0 r1 r2 r3 r4 t
        = (if c25_1 r1 t && c25_4 r4 t then (1:ℤ) else 0)
          - (if c25_1 r1 t && c25_4 r4 t && c25_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl25_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n25, Wl25_6 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl25_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n25, Wl25_6 r0 r1 r2 r3 r4 t
        = P25_6 r1 r4 - C25_6 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P25_6, C25_6]
    have hm : C25_6 r1 r4 r0 ≤ M25_6 r1 r4 :=
      CaseSplit.le_mxr (C25_6 r1 r4) 10 r0 (by omega)
    simp only [N25_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N25_7 r2 r3 ≤ ∑ t ∈ Finset.range n25, Wl25_7 r0 r1 r2 r3 r4 t := by
    simp only [N25_7]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl25_7]
    exact CaseSplit.ind_nonneg _
  have hn8 : N25_8 r2 r4 ≤ ∑ t ∈ Finset.range n25, Wl25_8 r0 r1 r2 r3 r4 t := by
    simp only [N25_8]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl25_8]
    exact CaseSplit.ind_nonneg _
  have hn9 : N25_9 r3 r4 ≤ ∑ t ∈ Finset.range n25, Wl25_9 r0 r1 r2 r3 r4 t := by
    simp only [N25_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl25_9]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n25, (w25 t + 1) * Dg25 r0 r1 r2 r3 r4 t = S25_0 r0 + S25_1 r1 + S25_2 r2 + S25_3 r3 + S25_4 r4 := by
    simp only [S25_0, S25_1, S25_2, S25_3, S25_4, Dg25, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n25, (w25 t + 1) * Dg25 r0 r1 r2 r3 r4 t
      = (∑ t ∈ Finset.range n25, w25 t * Dg25 r0 r1 r2 r3 r4 t)
        + 1 * (∑ t ∈ Finset.range n25, Dg25 r0 r1 r2 r3 r4 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n25, w25 t)
      ≤ ∑ t ∈ Finset.range n25, w25 t * Dg25 r0 r1 r2 r3 r4 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg25 r0 r1 r2 r3 r4 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w25 t := wnn25 t (Finset.mem_range.mp ht)
    calc w25 t = w25 t * 1 := (mul_one _).symm
      _ ≤ w25 t * Dg25 r0 r1 r2 r3 r4 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS25_0 r0 + aS25_1 r1 + aS25_2 r2 + aS25_3 r3 + aS25_4 r4) + (aP25_0 r0 r1 + aP25_1 r0 r2 + aP25_2 r0 r3 + aP25_3 r0 r4 + aP25_4 r1 r2 + aP25_5 r1 r3 + aP25_6 r1 r4 + aP25_7 r2 r3 + aP25_8 r2 r4 + aP25_9 r3 r4) = (S25_0 r0 + S25_1 r1 + S25_2 r2 + S25_3 r3 + S25_4 r4) - 1 * (N25_0 r0 r1 + N25_1 r0 r2 + N25_2 r0 r3 + N25_3 r0 r4 + N25_4 r1 r2 + N25_5 r1 r3 + N25_6 r1 r4 + N25_7 r2 r3 + N25_8 r2 r4 + N25_9 r3 r4) := by
    simp only [aS25_0, aS25_1, aS25_2, aS25_3, aS25_4, aP25_0, aP25_1, aP25_2, aP25_3, aP25_4, aP25_5, aP25_6, aP25_7, aP25_8, aP25_9, L25_0, L25_1, L25_2, L25_3, L25_4]
    ring
  have bS0 : aS25_0 r0 ≤ MS25_0 := CaseSplit.le_mxr (aS25_0) 10 r0 (by omega)
  have bS1 : aS25_1 r1 ≤ MS25_1 := CaseSplit.le_mxr (aS25_1) 12 r1 (by omega)
  have bS2 : aS25_2 r2 ≤ MS25_2 := CaseSplit.le_mxr (aS25_2) 16 r2 (by omega)
  have bS3 : aS25_3 r3 ≤ MS25_3 := CaseSplit.le_mxr (aS25_3) 18 r3 (by omega)
  have bS4 : aS25_4 r4 ≤ MS25_4 := CaseSplit.le_mxr (aS25_4) 22 r4 (by omega)
  have bP0 : aP25_0 r0 r1 ≤ MP25_0 := CaseSplit.le_mxr2 (aP25_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP25_1 r0 r2 ≤ MP25_1 := CaseSplit.le_mxr2 (aP25_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP25_2 r0 r3 ≤ MP25_2 := CaseSplit.le_mxr2 (aP25_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP25_3 r0 r4 ≤ MP25_3 := CaseSplit.le_mxr2 (aP25_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP25_4 r1 r2 ≤ MP25_4 := CaseSplit.le_mxr2 (aP25_4) 12 16 r1 r2 (by omega) (by omega)
  have bP5 : aP25_5 r1 r3 ≤ MP25_5 := CaseSplit.le_mxr2 (aP25_5) 12 18 r1 r3 (by omega) (by omega)
  have bP6 : aP25_6 r1 r4 ≤ MP25_6 := CaseSplit.le_mxr2 (aP25_6) 12 22 r1 r4 (by omega) (by omega)
  have bP7 : aP25_7 r2 r3 ≤ MP25_7 := CaseSplit.le_mxr2 (aP25_7) 16 18 r2 r3 (by omega) (by omega)
  have bP8 : aP25_8 r2 r4 ≤ MP25_8 := CaseSplit.le_mxr2 (aP25_8) 16 22 r2 r4 (by omega) (by omega)
  have bP9 : aP25_9 r3 r4 ≤ MP25_9 := CaseSplit.le_mxr2 (aP25_9) 18 22 r3 r4 (by omega) (by omega)
  have hrhs : rhs25 = (∑ t ∈ Finset.range n25, w25 t) + 1 * (n25 : ℤ) := rfl
  have hc := cert25
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, bS0, bS1, bS2, bS3, bS4, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9]

end IncCert23
