/-
INCREMENT-WIDTH CERTIFICATE, step 19->23, case 23 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_19_23.json, which re-derives every number
from the primes alone).

Machine 23, INCREMENT width 39 = F_2(19) + s_min(23) = 31 + 8,
held gears [5, 7] at phases [3, 2].  Free gears [11, 13, 17, 19, 23].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 1.
-/
import IncCert23B

namespace IncCert23

/-! ### case 23: held gears at phases [3, 2] -/

def p23 : List ℕ := [0, 2, 5, 7, 9, 10, 12, 14, 15, 17, 19, 22, 24, 29, 30, 35, 37]
def q23 (t : ℕ) : ℕ := p23.getD t 0
def n23 : ℕ := 17
def yl23 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def w23 (t : ℕ) : ℤ := yl23.getD t 0
def ul23 : List ℤ := [0, (-1), 0, (-1), 0, (-1), 0, (-1), (-2), 0, (-1), (-1), (-1), 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, (-1), 0, 0, 0, 0, (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), (-1), 0, (-1), (-1), 0, (-1), (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, (-1), (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 2, 3, 3, 2, 1, 2, 1, 3, 2, 3, 1, 2, 3, 3, 2, 2, 1, (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 3, 2, 2, 3, 1, 3, 2, 2, 3, 1, 3, 3, 2, 3, 1, 3, 3, 2, 3, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 0, 2, 2, 2, 0]
def u23 (k : ℕ) : ℤ := ul23.getD k 0

def c23_0 (r t : ℕ) : Bool := gb11 r (q23 t)
def c23_1 (r t : ℕ) : Bool := gb13 r (q23 t)
def c23_2 (r t : ℕ) : Bool := gb17 r (q23 t)
def c23_3 (r t : ℕ) : Bool := gb19 r (q23 t)
def c23_4 (r t : ℕ) : Bool := gb23 r (q23 t)

def S23_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n23, (w23 t + 1) * (if c23_0 r t then 1 else 0)
def S23_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n23, (w23 t + 1) * (if c23_1 r t then 1 else 0)
def S23_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n23, (w23 t + 1) * (if c23_2 r t then 1 else 0)
def S23_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n23, (w23 t + 1) * (if c23_3 r t then 1 else 0)
def S23_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n23, (w23 t + 1) * (if c23_4 r t then 1 else 0)

def L23_0 (r : ℕ) : ℤ := u23 (13 + r) + u23 (41 + r) + u23 (71 + r) + u23 (105 + r)
def L23_1 (r : ℕ) : ℤ := u23 (0 + r) + u23 (133 + r) + u23 (165 + r) + u23 (201 + r)
def L23_2 (r : ℕ) : ℤ := u23 (24 + r) + u23 (116 + r) + u23 (233 + r) + u23 (273 + r)
def L23_3 (r : ℕ) : ℤ := u23 (52 + r) + u23 (146 + r) + u23 (214 + r) + u23 (313 + r)
def L23_4 (r : ℕ) : ℤ := u23 (82 + r) + u23 (178 + r) + u23 (250 + r) + u23 (290 + r)

def aS23_0 (r : ℕ) : ℤ := S23_0 r - L23_0 r
def MS23_0 : ℤ := CaseSplit.mxr (aS23_0) 10
def aS23_1 (r : ℕ) : ℤ := S23_1 r - L23_1 r
def MS23_1 : ℤ := CaseSplit.mxr (aS23_1) 12
def aS23_2 (r : ℕ) : ℤ := S23_2 r - L23_2 r
def MS23_2 : ℤ := CaseSplit.mxr (aS23_2) 16
def aS23_3 (r : ℕ) : ℤ := S23_3 r - L23_3 r
def MS23_3 : ℤ := CaseSplit.mxr (aS23_3) 18
def aS23_4 (r : ℕ) : ℤ := S23_4 r - L23_4 r
def MS23_4 : ℤ := CaseSplit.mxr (aS23_4) 22

def N23_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n23, (if c23_0 ra t && c23_1 rb t then 1 else 0)
def aP23_0 (ra rb : ℕ) : ℤ := -(1) * N23_0 ra rb + u23 (0 + rb) + u23 (13 + ra)
def MP23_0 : ℤ := CaseSplit.mxr2 (aP23_0) 10 12
def N23_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n23, (if c23_0 ra t && c23_2 rb t then 1 else 0)
def aP23_1 (ra rb : ℕ) : ℤ := -(1) * N23_1 ra rb + u23 (24 + rb) + u23 (41 + ra)
def MP23_1 : ℤ := CaseSplit.mxr2 (aP23_1) 10 16
def N23_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n23, (if c23_0 ra t && c23_3 rb t then 1 else 0)
def aP23_2 (ra rb : ℕ) : ℤ := -(1) * N23_2 ra rb + u23 (52 + rb) + u23 (71 + ra)
def MP23_2 : ℤ := CaseSplit.mxr2 (aP23_2) 10 18
def N23_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n23, (if c23_0 ra t && c23_4 rb t then 1 else 0)
def aP23_3 (ra rb : ℕ) : ℤ := -(1) * N23_3 ra rb + u23 (82 + rb) + u23 (105 + ra)
def MP23_3 : ℤ := CaseSplit.mxr2 (aP23_3) 10 22
def P23_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n23, (if c23_1 ra t && c23_2 rb t then 1 else 0)
def C23_4 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n23, (if c23_1 ra t && c23_2 rb t && c23_0 s t then 1 else 0)
def M23_4 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C23_4 ra rb) 10
def E23_4 : List ℕ := [75, 81, 111, 117, 154, 165, 190, 201, 206, 212]
def N23_4 (ra rb : ℕ) : ℤ := if E23_4.contains (ra * 17 + rb) = true then P23_4 ra rb - M23_4 ra rb else 0
def aP23_4 (ra rb : ℕ) : ℤ := -(1) * N23_4 ra rb + u23 (116 + rb) + u23 (133 + ra)
def MP23_4 : ℤ := CaseSplit.mxr2 (aP23_4) 12 16
def P23_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n23, (if c23_1 ra t && c23_3 rb t then 1 else 0)
def C23_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n23, (if c23_1 ra t && c23_3 rb t && c23_0 s t then 1 else 0)
def M23_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C23_5 ra rb) 10
def E23_5 : List ℕ := [1, 17, 38, 51, 93, 114, 127, 138, 172, 214]
def N23_5 (ra rb : ℕ) : ℤ := if E23_5.contains (ra * 19 + rb) = true then P23_5 ra rb - M23_5 ra rb else 0
def aP23_5 (ra rb : ℕ) : ℤ := -(1) * N23_5 ra rb + u23 (146 + rb) + u23 (165 + ra)
def MP23_5 : ℤ := CaseSplit.mxr2 (aP23_5) 12 18
def P23_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n23, (if c23_1 ra t && c23_4 rb t then 1 else 0)
def C23_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n23, (if c23_1 ra t && c23_4 rb t && c23_0 s t then 1 else 0)
def M23_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C23_6 ra rb) 10
def E23_6 : List ℕ := []
def N23_6 (ra rb : ℕ) : ℤ := if E23_6.contains (ra * 23 + rb) = true then P23_6 ra rb - M23_6 ra rb else 0
def aP23_6 (ra rb : ℕ) : ℤ := -(1) * N23_6 ra rb + u23 (178 + rb) + u23 (201 + ra)
def MP23_6 : ℤ := CaseSplit.mxr2 (aP23_6) 12 22
def N23_7 (_ra _rb : ℕ) : ℤ := 0
def aP23_7 (ra rb : ℕ) : ℤ := -(1) * N23_7 ra rb + u23 (214 + rb) + u23 (233 + ra)
def MP23_7 : ℤ := CaseSplit.mxr2 (aP23_7) 16 18
def N23_8 (_ra _rb : ℕ) : ℤ := 0
def aP23_8 (ra rb : ℕ) : ℤ := -(1) * N23_8 ra rb + u23 (250 + rb) + u23 (273 + ra)
def MP23_8 : ℤ := CaseSplit.mxr2 (aP23_8) 16 22
def N23_9 (_ra _rb : ℕ) : ℤ := 0
def aP23_9 (ra rb : ℕ) : ℤ := -(1) * N23_9 ra rb + u23 (290 + rb) + u23 (313 + ra)
def MP23_9 : ℤ := CaseSplit.mxr2 (aP23_9) 18 22

def rhs23 : ℤ := (∑ t ∈ Finset.range n23, w23 t) + 1 * (n23 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn23 : ∀ t, t < n23 → (0 : ℤ) ≤ w23 t := by decide
theorem plt23 : ∀ t, t < n23 → q23 t < 39 := by decide
theorem pfree23_5 : ∀ t, t < n23 → gb5 3 (q23 t) = false := by decide
theorem pfree23_7 : ∀ t, t < n23 → gb7 2 (q23 t) = false := by decide
theorem MSv23_0 : MS23_0 = 3 := by decide +kernel
theorem MSv23_1 : MS23_1 = 8 := by decide +kernel
theorem MSv23_2 : MS23_2 = 0 := by decide +kernel
theorem MSv23_3 : MS23_3 = 0 := by decide +kernel
theorem MSv23_4 : MS23_4 = 0 := by decide +kernel
theorem MPv23_0 : MP23_0 = 0 := by decide +kernel
theorem MPv23_1 : MP23_1 = 0 := by decide +kernel
theorem MPv23_2 : MP23_2 = 0 := by decide +kernel
theorem MPv23_3 : MP23_3 = 0 := by decide +kernel
theorem MPv23_4 : MP23_4 = 0 := by decide +kernel
theorem MPv23_5 : MP23_5 = 0 := by decide +kernel
theorem MPv23_6 : MP23_6 = 0 := by decide +kernel
theorem MPv23_7 : MP23_7 = 0 := by decide +kernel
theorem MPv23_8 : MP23_8 = 0 := by decide +kernel
theorem MPv23_9 : MP23_9 = 5 := by decide +kernel
theorem rhsv23 : rhs23 = 17 := by decide +kernel

/-- **The case-23 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/1.
    (Scaled by the common denominator 1: 16 < 17.) -/
theorem cert23 : MS23_0 + MS23_1 + MS23_2 + MS23_3 + MS23_4 + MP23_0 + MP23_1 + MP23_2 + MP23_3 + MP23_4 + MP23_5 + MP23_6 + MP23_7 + MP23_8 + MP23_9 < rhs23 := by
  rw [MSv23_0, MSv23_1, MSv23_2, MSv23_3, MSv23_4, MPv23_0, MPv23_1, MPv23_2, MPv23_3, MPv23_4, MPv23_5, MPv23_6, MPv23_7, MPv23_8, MPv23_9, rhsv23]
  decide

def Dg23 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := (if c23_0 r0 t then 1 else 0) + (if c23_1 r1 t then 1 else 0) + (if c23_2 r2 t then 1 else 0) + (if c23_3 r3 t then 1 else 0) + (if c23_4 r4 t then 1 else 0)
def Wl23_0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c23_0 r0 t && c23_1 r1 t then 1 else 0
def Wl23_1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c23_0 r0 t && c23_2 r2 t then 1 else 0
def Wl23_2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c23_0 r0 t && c23_3 r3 t then 1 else 0
def Wl23_3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c23_0 r0 t && c23_4 r4 t then 1 else 0
def Wl23_4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c23_0 r0 t && c23_1 r1 t && c23_2 r2 t then 1 else 0
def Wl23_5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c23_0 r0 t && c23_1 r1 t && c23_3 r3 t then 1 else 0
def Wl23_6 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c23_0 r0 t && c23_1 r1 t && c23_4 r4 t then 1 else 0
def Wl23_7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c23_0 r0 t && !c23_1 r1 t && c23_2 r2 t && c23_3 r3 t then 1 else 0
def Wl23_8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c23_0 r0 t && !c23_1 r1 t && c23_2 r2 t && c23_4 r4 t then 1 else 0
def Wl23_9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c23_0 r0 t && !c23_1 r1 t && !c23_2 r2 t && c23_3 r3 t && c23_4 r4 t then 1 else 0

/-- **No configuration blocks the whole window in case 23.** -/
theorem nocov23 {r0 r1 r2 r3 r4 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23)
    (hcov : ∀ t, t < n23 → (c23_0 r0 t || c23_1 r1 t || c23_2 r2 t || c23_3 r3 t || c23_4 r4 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n23, (1 : ℤ) + (Wl23_0 r0 r1 r2 r3 r4 t + Wl23_1 r0 r1 r2 r3 r4 t + Wl23_2 r0 r1 r2 r3 r4 t + Wl23_3 r0 r1 r2 r3 r4 t + Wl23_4 r0 r1 r2 r3 r4 t + Wl23_5 r0 r1 r2 r3 r4 t + Wl23_6 r0 r1 r2 r3 r4 t + Wl23_7 r0 r1 r2 r3 r4 t + Wl23_8 r0 r1 r2 r3 r4 t + Wl23_9 r0 r1 r2 r3 r4 t) ≤ Dg23 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Wl23_0, Wl23_1, Wl23_2, Wl23_3, Wl23_4, Wl23_5, Wl23_6, Wl23_7, Wl23_8, Wl23_9, Dg23]
    exact CaseSplit.lowest5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n23, (1 : ℤ) ≤ Dg23 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Dg23]
    exact CaseSplit.degpos5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n23 : ℤ) + ((∑ t ∈ Finset.range n23, Wl23_0 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n23, Wl23_1 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n23, Wl23_2 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n23, Wl23_3 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n23, Wl23_4 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n23, Wl23_5 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n23, Wl23_6 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n23, Wl23_7 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n23, Wl23_8 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n23, Wl23_9 r0 r1 r2 r3 r4 t)) ≤ ∑ t ∈ Finset.range n23, Dg23 r0 r1 r2 r3 r4 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N23_0 r0 r1 ≤ ∑ t ∈ Finset.range n23, Wl23_0 r0 r1 r2 r3 r4 t := by
    simp only [N23_0, Wl23_0, le_refl]
  have hn1 : N23_1 r0 r2 ≤ ∑ t ∈ Finset.range n23, Wl23_1 r0 r1 r2 r3 r4 t := by
    simp only [N23_1, Wl23_1, le_refl]
  have hn2 : N23_2 r0 r3 ≤ ∑ t ∈ Finset.range n23, Wl23_2 r0 r1 r2 r3 r4 t := by
    simp only [N23_2, Wl23_2, le_refl]
  have hn3 : N23_3 r0 r4 ≤ ∑ t ∈ Finset.range n23, Wl23_3 r0 r1 r2 r3 r4 t := by
    simp only [N23_3, Wl23_3, le_refl]
  have hn4 : N23_4 r1 r2 ≤ ∑ t ∈ Finset.range n23, Wl23_4 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n23, Wl23_4 r0 r1 r2 r3 r4 t
        = (if c23_1 r1 t && c23_2 r2 t then (1:ℤ) else 0)
          - (if c23_1 r1 t && c23_2 r2 t && c23_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl23_4]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n23, Wl23_4 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl23_4]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n23, Wl23_4 r0 r1 r2 r3 r4 t
        = P23_4 r1 r2 - C23_4 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P23_4, C23_4]
    have hm : C23_4 r1 r2 r0 ≤ M23_4 r1 r2 :=
      CaseSplit.le_mxr (C23_4 r1 r2) 10 r0 (by omega)
    simp only [N23_4]
    split
    · rw [hL]; omega
    · exact hnn
  have hn5 : N23_5 r1 r3 ≤ ∑ t ∈ Finset.range n23, Wl23_5 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n23, Wl23_5 r0 r1 r2 r3 r4 t
        = (if c23_1 r1 t && c23_3 r3 t then (1:ℤ) else 0)
          - (if c23_1 r1 t && c23_3 r3 t && c23_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl23_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n23, Wl23_5 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl23_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n23, Wl23_5 r0 r1 r2 r3 r4 t
        = P23_5 r1 r3 - C23_5 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P23_5, C23_5]
    have hm : C23_5 r1 r3 r0 ≤ M23_5 r1 r3 :=
      CaseSplit.le_mxr (C23_5 r1 r3) 10 r0 (by omega)
    simp only [N23_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N23_6 r1 r4 ≤ ∑ t ∈ Finset.range n23, Wl23_6 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n23, Wl23_6 r0 r1 r2 r3 r4 t
        = (if c23_1 r1 t && c23_4 r4 t then (1:ℤ) else 0)
          - (if c23_1 r1 t && c23_4 r4 t && c23_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl23_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n23, Wl23_6 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl23_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n23, Wl23_6 r0 r1 r2 r3 r4 t
        = P23_6 r1 r4 - C23_6 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P23_6, C23_6]
    have hm : C23_6 r1 r4 r0 ≤ M23_6 r1 r4 :=
      CaseSplit.le_mxr (C23_6 r1 r4) 10 r0 (by omega)
    simp only [N23_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N23_7 r2 r3 ≤ ∑ t ∈ Finset.range n23, Wl23_7 r0 r1 r2 r3 r4 t := by
    simp only [N23_7]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl23_7]
    exact CaseSplit.ind_nonneg _
  have hn8 : N23_8 r2 r4 ≤ ∑ t ∈ Finset.range n23, Wl23_8 r0 r1 r2 r3 r4 t := by
    simp only [N23_8]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl23_8]
    exact CaseSplit.ind_nonneg _
  have hn9 : N23_9 r3 r4 ≤ ∑ t ∈ Finset.range n23, Wl23_9 r0 r1 r2 r3 r4 t := by
    simp only [N23_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl23_9]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n23, (w23 t + 1) * Dg23 r0 r1 r2 r3 r4 t = S23_0 r0 + S23_1 r1 + S23_2 r2 + S23_3 r3 + S23_4 r4 := by
    simp only [S23_0, S23_1, S23_2, S23_3, S23_4, Dg23, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n23, (w23 t + 1) * Dg23 r0 r1 r2 r3 r4 t
      = (∑ t ∈ Finset.range n23, w23 t * Dg23 r0 r1 r2 r3 r4 t)
        + 1 * (∑ t ∈ Finset.range n23, Dg23 r0 r1 r2 r3 r4 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n23, w23 t)
      ≤ ∑ t ∈ Finset.range n23, w23 t * Dg23 r0 r1 r2 r3 r4 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg23 r0 r1 r2 r3 r4 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w23 t := wnn23 t (Finset.mem_range.mp ht)
    calc w23 t = w23 t * 1 := (mul_one _).symm
      _ ≤ w23 t * Dg23 r0 r1 r2 r3 r4 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS23_0 r0 + aS23_1 r1 + aS23_2 r2 + aS23_3 r3 + aS23_4 r4) + (aP23_0 r0 r1 + aP23_1 r0 r2 + aP23_2 r0 r3 + aP23_3 r0 r4 + aP23_4 r1 r2 + aP23_5 r1 r3 + aP23_6 r1 r4 + aP23_7 r2 r3 + aP23_8 r2 r4 + aP23_9 r3 r4) = (S23_0 r0 + S23_1 r1 + S23_2 r2 + S23_3 r3 + S23_4 r4) - 1 * (N23_0 r0 r1 + N23_1 r0 r2 + N23_2 r0 r3 + N23_3 r0 r4 + N23_4 r1 r2 + N23_5 r1 r3 + N23_6 r1 r4 + N23_7 r2 r3 + N23_8 r2 r4 + N23_9 r3 r4) := by
    simp only [aS23_0, aS23_1, aS23_2, aS23_3, aS23_4, aP23_0, aP23_1, aP23_2, aP23_3, aP23_4, aP23_5, aP23_6, aP23_7, aP23_8, aP23_9, L23_0, L23_1, L23_2, L23_3, L23_4]
    ring
  have bS0 : aS23_0 r0 ≤ MS23_0 := CaseSplit.le_mxr (aS23_0) 10 r0 (by omega)
  have bS1 : aS23_1 r1 ≤ MS23_1 := CaseSplit.le_mxr (aS23_1) 12 r1 (by omega)
  have bS2 : aS23_2 r2 ≤ MS23_2 := CaseSplit.le_mxr (aS23_2) 16 r2 (by omega)
  have bS3 : aS23_3 r3 ≤ MS23_3 := CaseSplit.le_mxr (aS23_3) 18 r3 (by omega)
  have bS4 : aS23_4 r4 ≤ MS23_4 := CaseSplit.le_mxr (aS23_4) 22 r4 (by omega)
  have bP0 : aP23_0 r0 r1 ≤ MP23_0 := CaseSplit.le_mxr2 (aP23_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP23_1 r0 r2 ≤ MP23_1 := CaseSplit.le_mxr2 (aP23_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP23_2 r0 r3 ≤ MP23_2 := CaseSplit.le_mxr2 (aP23_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP23_3 r0 r4 ≤ MP23_3 := CaseSplit.le_mxr2 (aP23_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP23_4 r1 r2 ≤ MP23_4 := CaseSplit.le_mxr2 (aP23_4) 12 16 r1 r2 (by omega) (by omega)
  have bP5 : aP23_5 r1 r3 ≤ MP23_5 := CaseSplit.le_mxr2 (aP23_5) 12 18 r1 r3 (by omega) (by omega)
  have bP6 : aP23_6 r1 r4 ≤ MP23_6 := CaseSplit.le_mxr2 (aP23_6) 12 22 r1 r4 (by omega) (by omega)
  have bP7 : aP23_7 r2 r3 ≤ MP23_7 := CaseSplit.le_mxr2 (aP23_7) 16 18 r2 r3 (by omega) (by omega)
  have bP8 : aP23_8 r2 r4 ≤ MP23_8 := CaseSplit.le_mxr2 (aP23_8) 16 22 r2 r4 (by omega) (by omega)
  have bP9 : aP23_9 r3 r4 ≤ MP23_9 := CaseSplit.le_mxr2 (aP23_9) 18 22 r3 r4 (by omega) (by omega)
  have hrhs : rhs23 = (∑ t ∈ Finset.range n23, w23 t) + 1 * (n23 : ℤ) := rfl
  have hc := cert23
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, bS0, bS1, bS2, bS3, bS4, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9]

end IncCert23
