/-
INCREMENT-WIDTH CERTIFICATE, step 19->23, case 18 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_19_23.json, which re-derives every number
from the primes alone).

Machine 23, INCREMENT width 39 = F_2(19) + s_min(23) = 31 + 8,
held gears [5, 7] at phases [2, 4].  Free gears [11, 13, 17, 19, 23].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 1.
-/
import IncCert23B

namespace IncCert23

/-! ### case 18: held gears at phases [2, 4] -/

def p18 : List ℕ := [0, 1, 3, 5, 6, 8, 10, 13, 15, 20, 21, 26, 28, 31, 33, 35, 36, 38]
def q18 (t : ℕ) : ℕ := p18.getD t 0
def n18 : ℕ := 18
def yl18 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def w18 (t : ℕ) : ℤ := yl18.getD t 0
def ul18 : List ℤ := [0, (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), 0, (-1), 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), (-2), 0, 0, (-1), 0, 0, 0, (-1), 0, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), 0, 1, 1, 1, (-1), (-1), 0, 1, 1, 1, 0, (-1), (-1), 0, 1, 1, 1, 1, 1, (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-2), (-1), (-1), (-1), (-2), (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 3, 0, 3, 2, 0, 2, 0, 2, 2, 1, 3, 1, 2, 2, 1, 3, 0, 2, 3, 2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 0, 0]
def u18 (k : ℕ) : ℤ := ul18.getD k 0

def c18_0 (r t : ℕ) : Bool := gb11 r (q18 t)
def c18_1 (r t : ℕ) : Bool := gb13 r (q18 t)
def c18_2 (r t : ℕ) : Bool := gb17 r (q18 t)
def c18_3 (r t : ℕ) : Bool := gb19 r (q18 t)
def c18_4 (r t : ℕ) : Bool := gb23 r (q18 t)

def S18_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (w18 t + 1) * (if c18_0 r t then 1 else 0)
def S18_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (w18 t + 1) * (if c18_1 r t then 1 else 0)
def S18_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (w18 t + 1) * (if c18_2 r t then 1 else 0)
def S18_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (w18 t + 1) * (if c18_3 r t then 1 else 0)
def S18_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (w18 t + 1) * (if c18_4 r t then 1 else 0)

def L18_0 (r : ℕ) : ℤ := u18 (13 + r) + u18 (41 + r) + u18 (71 + r) + u18 (105 + r)
def L18_1 (r : ℕ) : ℤ := u18 (0 + r) + u18 (133 + r) + u18 (165 + r) + u18 (201 + r)
def L18_2 (r : ℕ) : ℤ := u18 (24 + r) + u18 (116 + r) + u18 (233 + r) + u18 (273 + r)
def L18_3 (r : ℕ) : ℤ := u18 (52 + r) + u18 (146 + r) + u18 (214 + r) + u18 (313 + r)
def L18_4 (r : ℕ) : ℤ := u18 (82 + r) + u18 (178 + r) + u18 (250 + r) + u18 (290 + r)

def aS18_0 (r : ℕ) : ℤ := S18_0 r - L18_0 r
def MS18_0 : ℤ := CaseSplit.mxr (aS18_0) 10
def aS18_1 (r : ℕ) : ℤ := S18_1 r - L18_1 r
def MS18_1 : ℤ := CaseSplit.mxr (aS18_1) 12
def aS18_2 (r : ℕ) : ℤ := S18_2 r - L18_2 r
def MS18_2 : ℤ := CaseSplit.mxr (aS18_2) 16
def aS18_3 (r : ℕ) : ℤ := S18_3 r - L18_3 r
def MS18_3 : ℤ := CaseSplit.mxr (aS18_3) 18
def aS18_4 (r : ℕ) : ℤ := S18_4 r - L18_4 r
def MS18_4 : ℤ := CaseSplit.mxr (aS18_4) 22

def N18_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (if c18_0 ra t && c18_1 rb t then 1 else 0)
def aP18_0 (ra rb : ℕ) : ℤ := -(1) * N18_0 ra rb + u18 (0 + rb) + u18 (13 + ra)
def MP18_0 : ℤ := CaseSplit.mxr2 (aP18_0) 10 12
def N18_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (if c18_0 ra t && c18_2 rb t then 1 else 0)
def aP18_1 (ra rb : ℕ) : ℤ := -(1) * N18_1 ra rb + u18 (24 + rb) + u18 (41 + ra)
def MP18_1 : ℤ := CaseSplit.mxr2 (aP18_1) 10 16
def N18_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (if c18_0 ra t && c18_3 rb t then 1 else 0)
def aP18_2 (ra rb : ℕ) : ℤ := -(1) * N18_2 ra rb + u18 (52 + rb) + u18 (71 + ra)
def MP18_2 : ℤ := CaseSplit.mxr2 (aP18_2) 10 18
def N18_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (if c18_0 ra t && c18_4 rb t then 1 else 0)
def aP18_3 (ra rb : ℕ) : ℤ := -(1) * N18_3 ra rb + u18 (82 + rb) + u18 (105 + ra)
def MP18_3 : ℤ := CaseSplit.mxr2 (aP18_3) 10 22
def P18_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (if c18_1 ra t && c18_2 rb t then 1 else 0)
def C18_4 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (if c18_1 ra t && c18_2 rb t && c18_0 s t then 1 else 0)
def M18_4 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C18_4 ra rb) 10
def E18_4 : List ℕ := [61, 67, 136, 147]
def N18_4 (ra rb : ℕ) : ℤ := if E18_4.contains (ra * 17 + rb) = true then P18_4 ra rb - M18_4 ra rb else 0
def aP18_4 (ra rb : ℕ) : ℤ := -(1) * N18_4 ra rb + u18 (116 + rb) + u18 (133 + ra)
def MP18_4 : ℤ := CaseSplit.mxr2 (aP18_4) 12 16
def P18_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (if c18_1 ra t && c18_3 rb t then 1 else 0)
def C18_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (if c18_1 ra t && c18_3 rb t && c18_0 s t then 1 else 0)
def M18_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C18_5 ra rb) 10
def E18_5 : List ℕ := [7, 41, 47, 71, 78, 147, 154, 178, 212, 218]
def N18_5 (ra rb : ℕ) : ℤ := if E18_5.contains (ra * 19 + rb) = true then P18_5 ra rb - M18_5 ra rb else 0
def aP18_5 (ra rb : ℕ) : ℤ := -(1) * N18_5 ra rb + u18 (146 + rb) + u18 (165 + ra)
def MP18_5 : ℤ := CaseSplit.mxr2 (aP18_5) 12 18
def P18_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (if c18_1 ra t && c18_4 rb t then 1 else 0)
def C18_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n18, (if c18_1 ra t && c18_4 rb t && c18_0 s t then 1 else 0)
def M18_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C18_6 ra rb) 10
def E18_6 : List ℕ := []
def N18_6 (ra rb : ℕ) : ℤ := if E18_6.contains (ra * 23 + rb) = true then P18_6 ra rb - M18_6 ra rb else 0
def aP18_6 (ra rb : ℕ) : ℤ := -(1) * N18_6 ra rb + u18 (178 + rb) + u18 (201 + ra)
def MP18_6 : ℤ := CaseSplit.mxr2 (aP18_6) 12 22
def N18_7 (_ra _rb : ℕ) : ℤ := 0
def aP18_7 (ra rb : ℕ) : ℤ := -(1) * N18_7 ra rb + u18 (214 + rb) + u18 (233 + ra)
def MP18_7 : ℤ := CaseSplit.mxr2 (aP18_7) 16 18
def N18_8 (_ra _rb : ℕ) : ℤ := 0
def aP18_8 (ra rb : ℕ) : ℤ := -(1) * N18_8 ra rb + u18 (250 + rb) + u18 (273 + ra)
def MP18_8 : ℤ := CaseSplit.mxr2 (aP18_8) 16 22
def N18_9 (_ra _rb : ℕ) : ℤ := 0
def aP18_9 (ra rb : ℕ) : ℤ := -(1) * N18_9 ra rb + u18 (290 + rb) + u18 (313 + ra)
def MP18_9 : ℤ := CaseSplit.mxr2 (aP18_9) 18 22

def rhs18 : ℤ := (∑ t ∈ Finset.range n18, w18 t) + 1 * (n18 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn18 : ∀ t, t < n18 → (0 : ℤ) ≤ w18 t := by decide
theorem plt18 : ∀ t, t < n18 → q18 t < 39 := by decide
theorem pfree18_5 : ∀ t, t < n18 → gb5 2 (q18 t) = false := by decide
theorem pfree18_7 : ∀ t, t < n18 → gb7 4 (q18 t) = false := by decide
theorem MSv18_0 : MS18_0 = 5 := by decide +kernel
theorem MSv18_1 : MS18_1 = 7 := by decide +kernel
theorem MSv18_2 : MS18_2 = 0 := by decide +kernel
theorem MSv18_3 : MS18_3 = 0 := by decide +kernel
theorem MSv18_4 : MS18_4 = 0 := by decide +kernel
theorem MPv18_0 : MP18_0 = 0 := by decide +kernel
theorem MPv18_1 : MP18_1 = 0 := by decide +kernel
theorem MPv18_2 : MP18_2 = 0 := by decide +kernel
theorem MPv18_3 : MP18_3 = 0 := by decide +kernel
theorem MPv18_4 : MP18_4 = 0 := by decide +kernel
theorem MPv18_5 : MP18_5 = 0 := by decide +kernel
theorem MPv18_6 : MP18_6 = 0 := by decide +kernel
theorem MPv18_7 : MP18_7 = 0 := by decide +kernel
theorem MPv18_8 : MP18_8 = 0 := by decide +kernel
theorem MPv18_9 : MP18_9 = 5 := by decide +kernel
theorem rhsv18 : rhs18 = 18 := by decide +kernel

/-- **The case-18 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/1.
    (Scaled by the common denominator 1: 17 < 18.) -/
theorem cert18 : MS18_0 + MS18_1 + MS18_2 + MS18_3 + MS18_4 + MP18_0 + MP18_1 + MP18_2 + MP18_3 + MP18_4 + MP18_5 + MP18_6 + MP18_7 + MP18_8 + MP18_9 < rhs18 := by
  rw [MSv18_0, MSv18_1, MSv18_2, MSv18_3, MSv18_4, MPv18_0, MPv18_1, MPv18_2, MPv18_3, MPv18_4, MPv18_5, MPv18_6, MPv18_7, MPv18_8, MPv18_9, rhsv18]
  decide

def Dg18 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := (if c18_0 r0 t then 1 else 0) + (if c18_1 r1 t then 1 else 0) + (if c18_2 r2 t then 1 else 0) + (if c18_3 r3 t then 1 else 0) + (if c18_4 r4 t then 1 else 0)
def Wl18_0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c18_0 r0 t && c18_1 r1 t then 1 else 0
def Wl18_1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c18_0 r0 t && c18_2 r2 t then 1 else 0
def Wl18_2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c18_0 r0 t && c18_3 r3 t then 1 else 0
def Wl18_3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c18_0 r0 t && c18_4 r4 t then 1 else 0
def Wl18_4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c18_0 r0 t && c18_1 r1 t && c18_2 r2 t then 1 else 0
def Wl18_5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c18_0 r0 t && c18_1 r1 t && c18_3 r3 t then 1 else 0
def Wl18_6 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c18_0 r0 t && c18_1 r1 t && c18_4 r4 t then 1 else 0
def Wl18_7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c18_0 r0 t && !c18_1 r1 t && c18_2 r2 t && c18_3 r3 t then 1 else 0
def Wl18_8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c18_0 r0 t && !c18_1 r1 t && c18_2 r2 t && c18_4 r4 t then 1 else 0
def Wl18_9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c18_0 r0 t && !c18_1 r1 t && !c18_2 r2 t && c18_3 r3 t && c18_4 r4 t then 1 else 0

/-- **No configuration blocks the whole window in case 18.** -/
theorem nocov18 {r0 r1 r2 r3 r4 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23)
    (hcov : ∀ t, t < n18 → (c18_0 r0 t || c18_1 r1 t || c18_2 r2 t || c18_3 r3 t || c18_4 r4 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n18, (1 : ℤ) + (Wl18_0 r0 r1 r2 r3 r4 t + Wl18_1 r0 r1 r2 r3 r4 t + Wl18_2 r0 r1 r2 r3 r4 t + Wl18_3 r0 r1 r2 r3 r4 t + Wl18_4 r0 r1 r2 r3 r4 t + Wl18_5 r0 r1 r2 r3 r4 t + Wl18_6 r0 r1 r2 r3 r4 t + Wl18_7 r0 r1 r2 r3 r4 t + Wl18_8 r0 r1 r2 r3 r4 t + Wl18_9 r0 r1 r2 r3 r4 t) ≤ Dg18 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Wl18_0, Wl18_1, Wl18_2, Wl18_3, Wl18_4, Wl18_5, Wl18_6, Wl18_7, Wl18_8, Wl18_9, Dg18]
    exact CaseSplit.lowest5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n18, (1 : ℤ) ≤ Dg18 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Dg18]
    exact CaseSplit.degpos5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n18 : ℤ) + ((∑ t ∈ Finset.range n18, Wl18_0 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n18, Wl18_1 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n18, Wl18_2 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n18, Wl18_3 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n18, Wl18_4 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n18, Wl18_5 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n18, Wl18_6 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n18, Wl18_7 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n18, Wl18_8 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n18, Wl18_9 r0 r1 r2 r3 r4 t)) ≤ ∑ t ∈ Finset.range n18, Dg18 r0 r1 r2 r3 r4 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N18_0 r0 r1 ≤ ∑ t ∈ Finset.range n18, Wl18_0 r0 r1 r2 r3 r4 t := by
    simp only [N18_0, Wl18_0, le_refl]
  have hn1 : N18_1 r0 r2 ≤ ∑ t ∈ Finset.range n18, Wl18_1 r0 r1 r2 r3 r4 t := by
    simp only [N18_1, Wl18_1, le_refl]
  have hn2 : N18_2 r0 r3 ≤ ∑ t ∈ Finset.range n18, Wl18_2 r0 r1 r2 r3 r4 t := by
    simp only [N18_2, Wl18_2, le_refl]
  have hn3 : N18_3 r0 r4 ≤ ∑ t ∈ Finset.range n18, Wl18_3 r0 r1 r2 r3 r4 t := by
    simp only [N18_3, Wl18_3, le_refl]
  have hn4 : N18_4 r1 r2 ≤ ∑ t ∈ Finset.range n18, Wl18_4 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n18, Wl18_4 r0 r1 r2 r3 r4 t
        = (if c18_1 r1 t && c18_2 r2 t then (1:ℤ) else 0)
          - (if c18_1 r1 t && c18_2 r2 t && c18_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl18_4]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n18, Wl18_4 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl18_4]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n18, Wl18_4 r0 r1 r2 r3 r4 t
        = P18_4 r1 r2 - C18_4 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P18_4, C18_4]
    have hm : C18_4 r1 r2 r0 ≤ M18_4 r1 r2 :=
      CaseSplit.le_mxr (C18_4 r1 r2) 10 r0 (by omega)
    simp only [N18_4]
    split
    · rw [hL]; omega
    · exact hnn
  have hn5 : N18_5 r1 r3 ≤ ∑ t ∈ Finset.range n18, Wl18_5 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n18, Wl18_5 r0 r1 r2 r3 r4 t
        = (if c18_1 r1 t && c18_3 r3 t then (1:ℤ) else 0)
          - (if c18_1 r1 t && c18_3 r3 t && c18_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl18_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n18, Wl18_5 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl18_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n18, Wl18_5 r0 r1 r2 r3 r4 t
        = P18_5 r1 r3 - C18_5 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P18_5, C18_5]
    have hm : C18_5 r1 r3 r0 ≤ M18_5 r1 r3 :=
      CaseSplit.le_mxr (C18_5 r1 r3) 10 r0 (by omega)
    simp only [N18_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N18_6 r1 r4 ≤ ∑ t ∈ Finset.range n18, Wl18_6 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n18, Wl18_6 r0 r1 r2 r3 r4 t
        = (if c18_1 r1 t && c18_4 r4 t then (1:ℤ) else 0)
          - (if c18_1 r1 t && c18_4 r4 t && c18_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl18_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n18, Wl18_6 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl18_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n18, Wl18_6 r0 r1 r2 r3 r4 t
        = P18_6 r1 r4 - C18_6 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P18_6, C18_6]
    have hm : C18_6 r1 r4 r0 ≤ M18_6 r1 r4 :=
      CaseSplit.le_mxr (C18_6 r1 r4) 10 r0 (by omega)
    simp only [N18_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N18_7 r2 r3 ≤ ∑ t ∈ Finset.range n18, Wl18_7 r0 r1 r2 r3 r4 t := by
    simp only [N18_7]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl18_7]
    exact CaseSplit.ind_nonneg _
  have hn8 : N18_8 r2 r4 ≤ ∑ t ∈ Finset.range n18, Wl18_8 r0 r1 r2 r3 r4 t := by
    simp only [N18_8]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl18_8]
    exact CaseSplit.ind_nonneg _
  have hn9 : N18_9 r3 r4 ≤ ∑ t ∈ Finset.range n18, Wl18_9 r0 r1 r2 r3 r4 t := by
    simp only [N18_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl18_9]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n18, (w18 t + 1) * Dg18 r0 r1 r2 r3 r4 t = S18_0 r0 + S18_1 r1 + S18_2 r2 + S18_3 r3 + S18_4 r4 := by
    simp only [S18_0, S18_1, S18_2, S18_3, S18_4, Dg18, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n18, (w18 t + 1) * Dg18 r0 r1 r2 r3 r4 t
      = (∑ t ∈ Finset.range n18, w18 t * Dg18 r0 r1 r2 r3 r4 t)
        + 1 * (∑ t ∈ Finset.range n18, Dg18 r0 r1 r2 r3 r4 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n18, w18 t)
      ≤ ∑ t ∈ Finset.range n18, w18 t * Dg18 r0 r1 r2 r3 r4 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg18 r0 r1 r2 r3 r4 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w18 t := wnn18 t (Finset.mem_range.mp ht)
    calc w18 t = w18 t * 1 := (mul_one _).symm
      _ ≤ w18 t * Dg18 r0 r1 r2 r3 r4 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS18_0 r0 + aS18_1 r1 + aS18_2 r2 + aS18_3 r3 + aS18_4 r4) + (aP18_0 r0 r1 + aP18_1 r0 r2 + aP18_2 r0 r3 + aP18_3 r0 r4 + aP18_4 r1 r2 + aP18_5 r1 r3 + aP18_6 r1 r4 + aP18_7 r2 r3 + aP18_8 r2 r4 + aP18_9 r3 r4) = (S18_0 r0 + S18_1 r1 + S18_2 r2 + S18_3 r3 + S18_4 r4) - 1 * (N18_0 r0 r1 + N18_1 r0 r2 + N18_2 r0 r3 + N18_3 r0 r4 + N18_4 r1 r2 + N18_5 r1 r3 + N18_6 r1 r4 + N18_7 r2 r3 + N18_8 r2 r4 + N18_9 r3 r4) := by
    simp only [aS18_0, aS18_1, aS18_2, aS18_3, aS18_4, aP18_0, aP18_1, aP18_2, aP18_3, aP18_4, aP18_5, aP18_6, aP18_7, aP18_8, aP18_9, L18_0, L18_1, L18_2, L18_3, L18_4]
    ring
  have bS0 : aS18_0 r0 ≤ MS18_0 := CaseSplit.le_mxr (aS18_0) 10 r0 (by omega)
  have bS1 : aS18_1 r1 ≤ MS18_1 := CaseSplit.le_mxr (aS18_1) 12 r1 (by omega)
  have bS2 : aS18_2 r2 ≤ MS18_2 := CaseSplit.le_mxr (aS18_2) 16 r2 (by omega)
  have bS3 : aS18_3 r3 ≤ MS18_3 := CaseSplit.le_mxr (aS18_3) 18 r3 (by omega)
  have bS4 : aS18_4 r4 ≤ MS18_4 := CaseSplit.le_mxr (aS18_4) 22 r4 (by omega)
  have bP0 : aP18_0 r0 r1 ≤ MP18_0 := CaseSplit.le_mxr2 (aP18_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP18_1 r0 r2 ≤ MP18_1 := CaseSplit.le_mxr2 (aP18_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP18_2 r0 r3 ≤ MP18_2 := CaseSplit.le_mxr2 (aP18_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP18_3 r0 r4 ≤ MP18_3 := CaseSplit.le_mxr2 (aP18_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP18_4 r1 r2 ≤ MP18_4 := CaseSplit.le_mxr2 (aP18_4) 12 16 r1 r2 (by omega) (by omega)
  have bP5 : aP18_5 r1 r3 ≤ MP18_5 := CaseSplit.le_mxr2 (aP18_5) 12 18 r1 r3 (by omega) (by omega)
  have bP6 : aP18_6 r1 r4 ≤ MP18_6 := CaseSplit.le_mxr2 (aP18_6) 12 22 r1 r4 (by omega) (by omega)
  have bP7 : aP18_7 r2 r3 ≤ MP18_7 := CaseSplit.le_mxr2 (aP18_7) 16 18 r2 r3 (by omega) (by omega)
  have bP8 : aP18_8 r2 r4 ≤ MP18_8 := CaseSplit.le_mxr2 (aP18_8) 16 22 r2 r4 (by omega) (by omega)
  have bP9 : aP18_9 r3 r4 ≤ MP18_9 := CaseSplit.le_mxr2 (aP18_9) 18 22 r3 r4 (by omega) (by omega)
  have hrhs : rhs18 = (∑ t ∈ Finset.range n18, w18 t) + 1 * (n18 : ℤ) := rfl
  have hc := cert18
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, bS0, bS1, bS2, bS3, bS4, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9]

end IncCert23
