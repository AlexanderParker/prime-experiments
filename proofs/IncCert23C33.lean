/-
INCREMENT-WIDTH CERTIFICATE, step 19->23, case 33 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_19_23.json, which re-derives every number
from the primes alone).

Machine 23, INCREMENT width 39 = F_2(19) + s_min(23) = 31 + 8,
held gears [5, 7] at phases [4, 5].  Free gears [11, 13, 17, 19, 23].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 1.
-/
import IncCert23B

namespace IncCert23

/-! ### case 33: held gears at phases [4, 5] -/

def p33 : List ℕ := [4, 6, 9, 11, 13, 14, 16, 18, 19, 21, 23, 26, 28, 33, 34]
def q33 (t : ℕ) : ℕ := p33.getD t 0
def n33 : ℕ := 15
def yl33 : List ℤ := [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
def w33 (t : ℕ) : ℤ := yl33.getD t 0
def ul33 : List ℤ := [0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), (-1), (-1), 0, 0, (-1), (-2), (-1), 0, (-1), (-1), (-1), (-1), (-1), 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), (-1), (-1), 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 2, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 2, 3, 3, 3, 3, (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), 1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 0, 0, 2, 1, 2, 1, 2, 1, 2, (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 1, 3, 2, 3, 3, 1, 3, 3, 2, 3, 1, 3, 3, 2, 3, 1, 2, 2, 1, 3, 2, 1, 0, 1, 1, 1, (-1), 1, 1, 1, 1, 1, 1, 0, 1, (-1), 0, 0, 1, 0]
def u33 (k : ℕ) : ℤ := ul33.getD k 0

def c33_0 (r t : ℕ) : Bool := gb11 r (q33 t)
def c33_1 (r t : ℕ) : Bool := gb13 r (q33 t)
def c33_2 (r t : ℕ) : Bool := gb17 r (q33 t)
def c33_3 (r t : ℕ) : Bool := gb19 r (q33 t)
def c33_4 (r t : ℕ) : Bool := gb23 r (q33 t)

def S33_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (w33 t + 1) * (if c33_0 r t then 1 else 0)
def S33_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (w33 t + 1) * (if c33_1 r t then 1 else 0)
def S33_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (w33 t + 1) * (if c33_2 r t then 1 else 0)
def S33_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (w33 t + 1) * (if c33_3 r t then 1 else 0)
def S33_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (w33 t + 1) * (if c33_4 r t then 1 else 0)

def L33_0 (r : ℕ) : ℤ := u33 (13 + r) + u33 (41 + r) + u33 (71 + r) + u33 (105 + r)
def L33_1 (r : ℕ) : ℤ := u33 (0 + r) + u33 (133 + r) + u33 (165 + r) + u33 (201 + r)
def L33_2 (r : ℕ) : ℤ := u33 (24 + r) + u33 (116 + r) + u33 (233 + r) + u33 (273 + r)
def L33_3 (r : ℕ) : ℤ := u33 (52 + r) + u33 (146 + r) + u33 (214 + r) + u33 (313 + r)
def L33_4 (r : ℕ) : ℤ := u33 (82 + r) + u33 (178 + r) + u33 (250 + r) + u33 (290 + r)

def aS33_0 (r : ℕ) : ℤ := S33_0 r - L33_0 r
def MS33_0 : ℤ := CaseSplit.mxr (aS33_0) 10
def aS33_1 (r : ℕ) : ℤ := S33_1 r - L33_1 r
def MS33_1 : ℤ := CaseSplit.mxr (aS33_1) 12
def aS33_2 (r : ℕ) : ℤ := S33_2 r - L33_2 r
def MS33_2 : ℤ := CaseSplit.mxr (aS33_2) 16
def aS33_3 (r : ℕ) : ℤ := S33_3 r - L33_3 r
def MS33_3 : ℤ := CaseSplit.mxr (aS33_3) 18
def aS33_4 (r : ℕ) : ℤ := S33_4 r - L33_4 r
def MS33_4 : ℤ := CaseSplit.mxr (aS33_4) 22

def N33_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (if c33_0 ra t && c33_1 rb t then 1 else 0)
def aP33_0 (ra rb : ℕ) : ℤ := -(1) * N33_0 ra rb + u33 (0 + rb) + u33 (13 + ra)
def MP33_0 : ℤ := CaseSplit.mxr2 (aP33_0) 10 12
def N33_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (if c33_0 ra t && c33_2 rb t then 1 else 0)
def aP33_1 (ra rb : ℕ) : ℤ := -(1) * N33_1 ra rb + u33 (24 + rb) + u33 (41 + ra)
def MP33_1 : ℤ := CaseSplit.mxr2 (aP33_1) 10 16
def N33_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (if c33_0 ra t && c33_3 rb t then 1 else 0)
def aP33_2 (ra rb : ℕ) : ℤ := -(1) * N33_2 ra rb + u33 (52 + rb) + u33 (71 + ra)
def MP33_2 : ℤ := CaseSplit.mxr2 (aP33_2) 10 18
def N33_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (if c33_0 ra t && c33_4 rb t then 1 else 0)
def aP33_3 (ra rb : ℕ) : ℤ := -(1) * N33_3 ra rb + u33 (82 + rb) + u33 (105 + ra)
def MP33_3 : ℤ := CaseSplit.mxr2 (aP33_3) 10 22
def P33_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (if c33_1 ra t && c33_2 rb t then 1 else 0)
def C33_4 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (if c33_1 ra t && c33_2 rb t && c33_0 s t then 1 else 0)
def M33_4 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C33_4 ra rb) 10
def E33_4 : List ℕ := [3, 9, 39, 45, 93, 99, 129, 135, 140, 151]
def N33_4 (ra rb : ℕ) : ℤ := if E33_4.contains (ra * 17 + rb) = true then P33_4 ra rb - M33_4 ra rb else 0
def aP33_4 (ra rb : ℕ) : ℤ := -(1) * N33_4 ra rb + u33 (116 + rb) + u33 (133 + ra)
def MP33_4 : ℤ := CaseSplit.mxr2 (aP33_4) 12 16
def P33_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (if c33_1 ra t && c33_3 rb t then 1 else 0)
def C33_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (if c33_1 ra t && c33_3 rb t && c33_0 s t then 1 else 0)
def M33_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C33_5 ra rb) 10
def E33_5 : List ℕ := [47, 58, 111, 134, 187, 218]
def N33_5 (ra rb : ℕ) : ℤ := if E33_5.contains (ra * 19 + rb) = true then P33_5 ra rb - M33_5 ra rb else 0
def aP33_5 (ra rb : ℕ) : ℤ := -(1) * N33_5 ra rb + u33 (146 + rb) + u33 (165 + ra)
def MP33_5 : ℤ := CaseSplit.mxr2 (aP33_5) 12 18
def P33_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (if c33_1 ra t && c33_4 rb t then 1 else 0)
def C33_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n33, (if c33_1 ra t && c33_4 rb t && c33_0 s t then 1 else 0)
def M33_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C33_6 ra rb) 10
def E33_6 : List ℕ := []
def N33_6 (ra rb : ℕ) : ℤ := if E33_6.contains (ra * 23 + rb) = true then P33_6 ra rb - M33_6 ra rb else 0
def aP33_6 (ra rb : ℕ) : ℤ := -(1) * N33_6 ra rb + u33 (178 + rb) + u33 (201 + ra)
def MP33_6 : ℤ := CaseSplit.mxr2 (aP33_6) 12 22
def N33_7 (_ra _rb : ℕ) : ℤ := 0
def aP33_7 (ra rb : ℕ) : ℤ := -(1) * N33_7 ra rb + u33 (214 + rb) + u33 (233 + ra)
def MP33_7 : ℤ := CaseSplit.mxr2 (aP33_7) 16 18
def N33_8 (_ra _rb : ℕ) : ℤ := 0
def aP33_8 (ra rb : ℕ) : ℤ := -(1) * N33_8 ra rb + u33 (250 + rb) + u33 (273 + ra)
def MP33_8 : ℤ := CaseSplit.mxr2 (aP33_8) 16 22
def N33_9 (_ra _rb : ℕ) : ℤ := 0
def aP33_9 (ra rb : ℕ) : ℤ := -(1) * N33_9 ra rb + u33 (290 + rb) + u33 (313 + ra)
def MP33_9 : ℤ := CaseSplit.mxr2 (aP33_9) 18 22

def rhs33 : ℤ := (∑ t ∈ Finset.range n33, w33 t) + 1 * (n33 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn33 : ∀ t, t < n33 → (0 : ℤ) ≤ w33 t := by decide
theorem plt33 : ∀ t, t < n33 → q33 t < 39 := by decide
theorem pfree33_5 : ∀ t, t < n33 → gb5 4 (q33 t) = false := by decide
theorem pfree33_7 : ∀ t, t < n33 → gb7 5 (q33 t) = false := by decide
theorem MSv33_0 : MS33_0 = 3 := by decide +kernel
theorem MSv33_1 : MS33_1 = 8 := by decide +kernel
theorem MSv33_2 : MS33_2 = 0 := by decide +kernel
theorem MSv33_3 : MS33_3 = 0 := by decide +kernel
theorem MSv33_4 : MS33_4 = 0 := by decide +kernel
theorem MPv33_0 : MP33_0 = 0 := by decide +kernel
theorem MPv33_1 : MP33_1 = 0 := by decide +kernel
theorem MPv33_2 : MP33_2 = 0 := by decide +kernel
theorem MPv33_3 : MP33_3 = 0 := by decide +kernel
theorem MPv33_4 : MP33_4 = 0 := by decide +kernel
theorem MPv33_5 : MP33_5 = 0 := by decide +kernel
theorem MPv33_6 : MP33_6 = 0 := by decide +kernel
theorem MPv33_7 : MP33_7 = 0 := by decide +kernel
theorem MPv33_8 : MP33_8 = 0 := by decide +kernel
theorem MPv33_9 : MP33_9 = 4 := by decide +kernel
theorem rhsv33 : rhs33 = 17 := by decide +kernel

/-- **The case-33 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 2/1.
    (Scaled by the common denominator 1: 15 < 17.) -/
theorem cert33 : MS33_0 + MS33_1 + MS33_2 + MS33_3 + MS33_4 + MP33_0 + MP33_1 + MP33_2 + MP33_3 + MP33_4 + MP33_5 + MP33_6 + MP33_7 + MP33_8 + MP33_9 < rhs33 := by
  rw [MSv33_0, MSv33_1, MSv33_2, MSv33_3, MSv33_4, MPv33_0, MPv33_1, MPv33_2, MPv33_3, MPv33_4, MPv33_5, MPv33_6, MPv33_7, MPv33_8, MPv33_9, rhsv33]
  decide

def Dg33 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := (if c33_0 r0 t then 1 else 0) + (if c33_1 r1 t then 1 else 0) + (if c33_2 r2 t then 1 else 0) + (if c33_3 r3 t then 1 else 0) + (if c33_4 r4 t then 1 else 0)
def Wl33_0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c33_0 r0 t && c33_1 r1 t then 1 else 0
def Wl33_1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c33_0 r0 t && c33_2 r2 t then 1 else 0
def Wl33_2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c33_0 r0 t && c33_3 r3 t then 1 else 0
def Wl33_3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c33_0 r0 t && c33_4 r4 t then 1 else 0
def Wl33_4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c33_0 r0 t && c33_1 r1 t && c33_2 r2 t then 1 else 0
def Wl33_5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c33_0 r0 t && c33_1 r1 t && c33_3 r3 t then 1 else 0
def Wl33_6 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c33_0 r0 t && c33_1 r1 t && c33_4 r4 t then 1 else 0
def Wl33_7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c33_0 r0 t && !c33_1 r1 t && c33_2 r2 t && c33_3 r3 t then 1 else 0
def Wl33_8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c33_0 r0 t && !c33_1 r1 t && c33_2 r2 t && c33_4 r4 t then 1 else 0
def Wl33_9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c33_0 r0 t && !c33_1 r1 t && !c33_2 r2 t && c33_3 r3 t && c33_4 r4 t then 1 else 0

/-- **No configuration blocks the whole window in case 33.** -/
theorem nocov33 {r0 r1 r2 r3 r4 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23)
    (hcov : ∀ t, t < n33 → (c33_0 r0 t || c33_1 r1 t || c33_2 r2 t || c33_3 r3 t || c33_4 r4 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n33, (1 : ℤ) + (Wl33_0 r0 r1 r2 r3 r4 t + Wl33_1 r0 r1 r2 r3 r4 t + Wl33_2 r0 r1 r2 r3 r4 t + Wl33_3 r0 r1 r2 r3 r4 t + Wl33_4 r0 r1 r2 r3 r4 t + Wl33_5 r0 r1 r2 r3 r4 t + Wl33_6 r0 r1 r2 r3 r4 t + Wl33_7 r0 r1 r2 r3 r4 t + Wl33_8 r0 r1 r2 r3 r4 t + Wl33_9 r0 r1 r2 r3 r4 t) ≤ Dg33 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Wl33_0, Wl33_1, Wl33_2, Wl33_3, Wl33_4, Wl33_5, Wl33_6, Wl33_7, Wl33_8, Wl33_9, Dg33]
    exact CaseSplit.lowest5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n33, (1 : ℤ) ≤ Dg33 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Dg33]
    exact CaseSplit.degpos5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n33 : ℤ) + ((∑ t ∈ Finset.range n33, Wl33_0 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n33, Wl33_1 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n33, Wl33_2 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n33, Wl33_3 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n33, Wl33_4 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n33, Wl33_5 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n33, Wl33_6 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n33, Wl33_7 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n33, Wl33_8 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n33, Wl33_9 r0 r1 r2 r3 r4 t)) ≤ ∑ t ∈ Finset.range n33, Dg33 r0 r1 r2 r3 r4 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N33_0 r0 r1 ≤ ∑ t ∈ Finset.range n33, Wl33_0 r0 r1 r2 r3 r4 t := by
    simp only [N33_0, Wl33_0, le_refl]
  have hn1 : N33_1 r0 r2 ≤ ∑ t ∈ Finset.range n33, Wl33_1 r0 r1 r2 r3 r4 t := by
    simp only [N33_1, Wl33_1, le_refl]
  have hn2 : N33_2 r0 r3 ≤ ∑ t ∈ Finset.range n33, Wl33_2 r0 r1 r2 r3 r4 t := by
    simp only [N33_2, Wl33_2, le_refl]
  have hn3 : N33_3 r0 r4 ≤ ∑ t ∈ Finset.range n33, Wl33_3 r0 r1 r2 r3 r4 t := by
    simp only [N33_3, Wl33_3, le_refl]
  have hn4 : N33_4 r1 r2 ≤ ∑ t ∈ Finset.range n33, Wl33_4 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n33, Wl33_4 r0 r1 r2 r3 r4 t
        = (if c33_1 r1 t && c33_2 r2 t then (1:ℤ) else 0)
          - (if c33_1 r1 t && c33_2 r2 t && c33_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl33_4]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n33, Wl33_4 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl33_4]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n33, Wl33_4 r0 r1 r2 r3 r4 t
        = P33_4 r1 r2 - C33_4 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P33_4, C33_4]
    have hm : C33_4 r1 r2 r0 ≤ M33_4 r1 r2 :=
      CaseSplit.le_mxr (C33_4 r1 r2) 10 r0 (by omega)
    simp only [N33_4]
    split
    · rw [hL]; omega
    · exact hnn
  have hn5 : N33_5 r1 r3 ≤ ∑ t ∈ Finset.range n33, Wl33_5 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n33, Wl33_5 r0 r1 r2 r3 r4 t
        = (if c33_1 r1 t && c33_3 r3 t then (1:ℤ) else 0)
          - (if c33_1 r1 t && c33_3 r3 t && c33_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl33_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n33, Wl33_5 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl33_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n33, Wl33_5 r0 r1 r2 r3 r4 t
        = P33_5 r1 r3 - C33_5 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P33_5, C33_5]
    have hm : C33_5 r1 r3 r0 ≤ M33_5 r1 r3 :=
      CaseSplit.le_mxr (C33_5 r1 r3) 10 r0 (by omega)
    simp only [N33_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N33_6 r1 r4 ≤ ∑ t ∈ Finset.range n33, Wl33_6 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n33, Wl33_6 r0 r1 r2 r3 r4 t
        = (if c33_1 r1 t && c33_4 r4 t then (1:ℤ) else 0)
          - (if c33_1 r1 t && c33_4 r4 t && c33_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl33_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n33, Wl33_6 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl33_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n33, Wl33_6 r0 r1 r2 r3 r4 t
        = P33_6 r1 r4 - C33_6 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P33_6, C33_6]
    have hm : C33_6 r1 r4 r0 ≤ M33_6 r1 r4 :=
      CaseSplit.le_mxr (C33_6 r1 r4) 10 r0 (by omega)
    simp only [N33_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N33_7 r2 r3 ≤ ∑ t ∈ Finset.range n33, Wl33_7 r0 r1 r2 r3 r4 t := by
    simp only [N33_7]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl33_7]
    exact CaseSplit.ind_nonneg _
  have hn8 : N33_8 r2 r4 ≤ ∑ t ∈ Finset.range n33, Wl33_8 r0 r1 r2 r3 r4 t := by
    simp only [N33_8]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl33_8]
    exact CaseSplit.ind_nonneg _
  have hn9 : N33_9 r3 r4 ≤ ∑ t ∈ Finset.range n33, Wl33_9 r0 r1 r2 r3 r4 t := by
    simp only [N33_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl33_9]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n33, (w33 t + 1) * Dg33 r0 r1 r2 r3 r4 t = S33_0 r0 + S33_1 r1 + S33_2 r2 + S33_3 r3 + S33_4 r4 := by
    simp only [S33_0, S33_1, S33_2, S33_3, S33_4, Dg33, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n33, (w33 t + 1) * Dg33 r0 r1 r2 r3 r4 t
      = (∑ t ∈ Finset.range n33, w33 t * Dg33 r0 r1 r2 r3 r4 t)
        + 1 * (∑ t ∈ Finset.range n33, Dg33 r0 r1 r2 r3 r4 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n33, w33 t)
      ≤ ∑ t ∈ Finset.range n33, w33 t * Dg33 r0 r1 r2 r3 r4 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg33 r0 r1 r2 r3 r4 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w33 t := wnn33 t (Finset.mem_range.mp ht)
    calc w33 t = w33 t * 1 := (mul_one _).symm
      _ ≤ w33 t * Dg33 r0 r1 r2 r3 r4 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS33_0 r0 + aS33_1 r1 + aS33_2 r2 + aS33_3 r3 + aS33_4 r4) + (aP33_0 r0 r1 + aP33_1 r0 r2 + aP33_2 r0 r3 + aP33_3 r0 r4 + aP33_4 r1 r2 + aP33_5 r1 r3 + aP33_6 r1 r4 + aP33_7 r2 r3 + aP33_8 r2 r4 + aP33_9 r3 r4) = (S33_0 r0 + S33_1 r1 + S33_2 r2 + S33_3 r3 + S33_4 r4) - 1 * (N33_0 r0 r1 + N33_1 r0 r2 + N33_2 r0 r3 + N33_3 r0 r4 + N33_4 r1 r2 + N33_5 r1 r3 + N33_6 r1 r4 + N33_7 r2 r3 + N33_8 r2 r4 + N33_9 r3 r4) := by
    simp only [aS33_0, aS33_1, aS33_2, aS33_3, aS33_4, aP33_0, aP33_1, aP33_2, aP33_3, aP33_4, aP33_5, aP33_6, aP33_7, aP33_8, aP33_9, L33_0, L33_1, L33_2, L33_3, L33_4]
    ring
  have bS0 : aS33_0 r0 ≤ MS33_0 := CaseSplit.le_mxr (aS33_0) 10 r0 (by omega)
  have bS1 : aS33_1 r1 ≤ MS33_1 := CaseSplit.le_mxr (aS33_1) 12 r1 (by omega)
  have bS2 : aS33_2 r2 ≤ MS33_2 := CaseSplit.le_mxr (aS33_2) 16 r2 (by omega)
  have bS3 : aS33_3 r3 ≤ MS33_3 := CaseSplit.le_mxr (aS33_3) 18 r3 (by omega)
  have bS4 : aS33_4 r4 ≤ MS33_4 := CaseSplit.le_mxr (aS33_4) 22 r4 (by omega)
  have bP0 : aP33_0 r0 r1 ≤ MP33_0 := CaseSplit.le_mxr2 (aP33_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP33_1 r0 r2 ≤ MP33_1 := CaseSplit.le_mxr2 (aP33_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP33_2 r0 r3 ≤ MP33_2 := CaseSplit.le_mxr2 (aP33_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP33_3 r0 r4 ≤ MP33_3 := CaseSplit.le_mxr2 (aP33_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP33_4 r1 r2 ≤ MP33_4 := CaseSplit.le_mxr2 (aP33_4) 12 16 r1 r2 (by omega) (by omega)
  have bP5 : aP33_5 r1 r3 ≤ MP33_5 := CaseSplit.le_mxr2 (aP33_5) 12 18 r1 r3 (by omega) (by omega)
  have bP6 : aP33_6 r1 r4 ≤ MP33_6 := CaseSplit.le_mxr2 (aP33_6) 12 22 r1 r4 (by omega) (by omega)
  have bP7 : aP33_7 r2 r3 ≤ MP33_7 := CaseSplit.le_mxr2 (aP33_7) 16 18 r2 r3 (by omega) (by omega)
  have bP8 : aP33_8 r2 r4 ≤ MP33_8 := CaseSplit.le_mxr2 (aP33_8) 16 22 r2 r4 (by omega) (by omega)
  have bP9 : aP33_9 r3 r4 ≤ MP33_9 := CaseSplit.le_mxr2 (aP33_9) 18 22 r3 r4 (by omega) (by omega)
  have hrhs : rhs33 = (∑ t ∈ Finset.range n33, w33 t) + 1 * (n33 : ℤ) := rfl
  have hc := cert33
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, bS0, bS1, bS2, bS3, bS4, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9]

end IncCert23
