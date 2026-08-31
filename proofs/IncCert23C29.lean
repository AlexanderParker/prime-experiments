/-
INCREMENT-WIDTH CERTIFICATE, step 19->23, case 29 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_19_23.json, which re-derives every number
from the primes alone).

Machine 23, INCREMENT width 39 = F_2(19) + s_min(23) = 31 + 8,
held gears [5, 7] at phases [4, 1].  Free gears [11, 13, 17, 19, 23].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 1.
-/
import IncCert23B

namespace IncCert23

/-! ### case 29: held gears at phases [4, 1] -/

def p29 : List ℕ := [1, 3, 4, 6, 8, 9, 11, 13, 16, 18, 23, 24, 29, 31, 34, 36, 38]
def q29 (t : ℕ) : ℕ := p29.getD t 0
def n29 : ℕ := 17
def yl29 : List ℤ := [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def w29 (t : ℕ) : ℤ := yl29.getD t 0
def ul29 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, (-2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, (-2), 0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 2, 2, 3, 2, 1, 1, 3, 3, 3, 1, 2, 2, 3, 3, 3, 3, 2, (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-2), 0, 2, 4, 1, 3, 3, 1, 3, 4, 3, 2, 2, 4, 2, 4, 3, 2, 4, 1, 3, 4, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0]
def u29 (k : ℕ) : ℤ := ul29.getD k 0

def c29_0 (r t : ℕ) : Bool := gb11 r (q29 t)
def c29_1 (r t : ℕ) : Bool := gb13 r (q29 t)
def c29_2 (r t : ℕ) : Bool := gb17 r (q29 t)
def c29_3 (r t : ℕ) : Bool := gb19 r (q29 t)
def c29_4 (r t : ℕ) : Bool := gb23 r (q29 t)

def S29_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (w29 t + 1) * (if c29_0 r t then 1 else 0)
def S29_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (w29 t + 1) * (if c29_1 r t then 1 else 0)
def S29_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (w29 t + 1) * (if c29_2 r t then 1 else 0)
def S29_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (w29 t + 1) * (if c29_3 r t then 1 else 0)
def S29_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (w29 t + 1) * (if c29_4 r t then 1 else 0)

def L29_0 (r : ℕ) : ℤ := u29 (13 + r) + u29 (41 + r) + u29 (71 + r) + u29 (105 + r)
def L29_1 (r : ℕ) : ℤ := u29 (0 + r) + u29 (133 + r) + u29 (165 + r) + u29 (201 + r)
def L29_2 (r : ℕ) : ℤ := u29 (24 + r) + u29 (116 + r) + u29 (233 + r) + u29 (273 + r)
def L29_3 (r : ℕ) : ℤ := u29 (52 + r) + u29 (146 + r) + u29 (214 + r) + u29 (313 + r)
def L29_4 (r : ℕ) : ℤ := u29 (82 + r) + u29 (178 + r) + u29 (250 + r) + u29 (290 + r)

def aS29_0 (r : ℕ) : ℤ := S29_0 r - L29_0 r
def MS29_0 : ℤ := CaseSplit.mxr (aS29_0) 10
def aS29_1 (r : ℕ) : ℤ := S29_1 r - L29_1 r
def MS29_1 : ℤ := CaseSplit.mxr (aS29_1) 12
def aS29_2 (r : ℕ) : ℤ := S29_2 r - L29_2 r
def MS29_2 : ℤ := CaseSplit.mxr (aS29_2) 16
def aS29_3 (r : ℕ) : ℤ := S29_3 r - L29_3 r
def MS29_3 : ℤ := CaseSplit.mxr (aS29_3) 18
def aS29_4 (r : ℕ) : ℤ := S29_4 r - L29_4 r
def MS29_4 : ℤ := CaseSplit.mxr (aS29_4) 22

def N29_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_0 ra t && c29_1 rb t then 1 else 0)
def aP29_0 (ra rb : ℕ) : ℤ := -(1) * N29_0 ra rb + u29 (0 + rb) + u29 (13 + ra)
def MP29_0 : ℤ := CaseSplit.mxr2 (aP29_0) 10 12
def N29_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_0 ra t && c29_2 rb t then 1 else 0)
def aP29_1 (ra rb : ℕ) : ℤ := -(1) * N29_1 ra rb + u29 (24 + rb) + u29 (41 + ra)
def MP29_1 : ℤ := CaseSplit.mxr2 (aP29_1) 10 16
def N29_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_0 ra t && c29_3 rb t then 1 else 0)
def aP29_2 (ra rb : ℕ) : ℤ := -(1) * N29_2 ra rb + u29 (52 + rb) + u29 (71 + ra)
def MP29_2 : ℤ := CaseSplit.mxr2 (aP29_2) 10 18
def N29_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_0 ra t && c29_4 rb t then 1 else 0)
def aP29_3 (ra rb : ℕ) : ℤ := -(1) * N29_3 ra rb + u29 (82 + rb) + u29 (105 + ra)
def MP29_3 : ℤ := CaseSplit.mxr2 (aP29_3) 10 22
def P29_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_1 ra t && c29_2 rb t then 1 else 0)
def C29_4 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_1 ra t && c29_2 rb t && c29_0 s t then 1 else 0)
def M29_4 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C29_4 ra rb) 10
def E29_4 : List ℕ := [93, 99, 172, 183]
def N29_4 (ra rb : ℕ) : ℤ := if E29_4.contains (ra * 17 + rb) = true then P29_4 ra rb - M29_4 ra rb else 0
def aP29_4 (ra rb : ℕ) : ℤ := -(1) * N29_4 ra rb + u29 (116 + rb) + u29 (133 + ra)
def MP29_4 : ℤ := CaseSplit.mxr2 (aP29_4) 12 16
def P29_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_1 ra t && c29_3 rb t then 1 else 0)
def C29_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_1 ra t && c29_3 rb t && c29_0 s t then 1 else 0)
def M29_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C29_5 ra rb) 10
def E29_5 : List ℕ := [11, 37, 87, 113, 118, 152, 158, 194, 228, 234]
def N29_5 (ra rb : ℕ) : ℤ := if E29_5.contains (ra * 19 + rb) = true then P29_5 ra rb - M29_5 ra rb else 0
def aP29_5 (ra rb : ℕ) : ℤ := -(1) * N29_5 ra rb + u29 (146 + rb) + u29 (165 + ra)
def MP29_5 : ℤ := CaseSplit.mxr2 (aP29_5) 12 18
def P29_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_1 ra t && c29_4 rb t then 1 else 0)
def C29_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n29, (if c29_1 ra t && c29_4 rb t && c29_0 s t then 1 else 0)
def M29_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C29_6 ra rb) 10
def E29_6 : List ℕ := []
def N29_6 (ra rb : ℕ) : ℤ := if E29_6.contains (ra * 23 + rb) = true then P29_6 ra rb - M29_6 ra rb else 0
def aP29_6 (ra rb : ℕ) : ℤ := -(1) * N29_6 ra rb + u29 (178 + rb) + u29 (201 + ra)
def MP29_6 : ℤ := CaseSplit.mxr2 (aP29_6) 12 22
def N29_7 (_ra _rb : ℕ) : ℤ := 0
def aP29_7 (ra rb : ℕ) : ℤ := -(1) * N29_7 ra rb + u29 (214 + rb) + u29 (233 + ra)
def MP29_7 : ℤ := CaseSplit.mxr2 (aP29_7) 16 18
def N29_8 (_ra _rb : ℕ) : ℤ := 0
def aP29_8 (ra rb : ℕ) : ℤ := -(1) * N29_8 ra rb + u29 (250 + rb) + u29 (273 + ra)
def MP29_8 : ℤ := CaseSplit.mxr2 (aP29_8) 16 22
def N29_9 (_ra _rb : ℕ) : ℤ := 0
def aP29_9 (ra rb : ℕ) : ℤ := -(1) * N29_9 ra rb + u29 (290 + rb) + u29 (313 + ra)
def MP29_9 : ℤ := CaseSplit.mxr2 (aP29_9) 18 22

def rhs29 : ℤ := (∑ t ∈ Finset.range n29, w29 t) + 1 * (n29 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn29 : ∀ t, t < n29 → (0 : ℤ) ≤ w29 t := by decide
theorem plt29 : ∀ t, t < n29 → q29 t < 39 := by decide
theorem pfree29_5 : ∀ t, t < n29 → gb5 4 (q29 t) = false := by decide
theorem pfree29_7 : ∀ t, t < n29 → gb7 1 (q29 t) = false := by decide
theorem MSv29_0 : MS29_0 = 3 := by decide +kernel
theorem MSv29_1 : MS29_1 = 10 := by decide +kernel
theorem MSv29_2 : MS29_2 = 0 := by decide +kernel
theorem MSv29_3 : MS29_3 = 0 := by decide +kernel
theorem MSv29_4 : MS29_4 = 0 := by decide +kernel
theorem MPv29_0 : MP29_0 = 0 := by decide +kernel
theorem MPv29_1 : MP29_1 = 0 := by decide +kernel
theorem MPv29_2 : MP29_2 = 0 := by decide +kernel
theorem MPv29_3 : MP29_3 = 0 := by decide +kernel
theorem MPv29_4 : MP29_4 = 0 := by decide +kernel
theorem MPv29_5 : MP29_5 = 0 := by decide +kernel
theorem MPv29_6 : MP29_6 = 0 := by decide +kernel
theorem MPv29_7 : MP29_7 = 0 := by decide +kernel
theorem MPv29_8 : MP29_8 = 0 := by decide +kernel
theorem MPv29_9 : MP29_9 = 4 := by decide +kernel
theorem rhsv29 : rhs29 = 18 := by decide +kernel

/-- **The case-29 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/1.
    (Scaled by the common denominator 1: 17 < 18.) -/
theorem cert29 : MS29_0 + MS29_1 + MS29_2 + MS29_3 + MS29_4 + MP29_0 + MP29_1 + MP29_2 + MP29_3 + MP29_4 + MP29_5 + MP29_6 + MP29_7 + MP29_8 + MP29_9 < rhs29 := by
  rw [MSv29_0, MSv29_1, MSv29_2, MSv29_3, MSv29_4, MPv29_0, MPv29_1, MPv29_2, MPv29_3, MPv29_4, MPv29_5, MPv29_6, MPv29_7, MPv29_8, MPv29_9, rhsv29]
  decide

def Dg29 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := (if c29_0 r0 t then 1 else 0) + (if c29_1 r1 t then 1 else 0) + (if c29_2 r2 t then 1 else 0) + (if c29_3 r3 t then 1 else 0) + (if c29_4 r4 t then 1 else 0)
def Wl29_0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c29_0 r0 t && c29_1 r1 t then 1 else 0
def Wl29_1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c29_0 r0 t && c29_2 r2 t then 1 else 0
def Wl29_2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c29_0 r0 t && c29_3 r3 t then 1 else 0
def Wl29_3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c29_0 r0 t && c29_4 r4 t then 1 else 0
def Wl29_4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c29_0 r0 t && c29_1 r1 t && c29_2 r2 t then 1 else 0
def Wl29_5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c29_0 r0 t && c29_1 r1 t && c29_3 r3 t then 1 else 0
def Wl29_6 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c29_0 r0 t && c29_1 r1 t && c29_4 r4 t then 1 else 0
def Wl29_7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c29_0 r0 t && !c29_1 r1 t && c29_2 r2 t && c29_3 r3 t then 1 else 0
def Wl29_8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c29_0 r0 t && !c29_1 r1 t && c29_2 r2 t && c29_4 r4 t then 1 else 0
def Wl29_9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c29_0 r0 t && !c29_1 r1 t && !c29_2 r2 t && c29_3 r3 t && c29_4 r4 t then 1 else 0

/-- **No configuration blocks the whole window in case 29.** -/
theorem nocov29 {r0 r1 r2 r3 r4 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23)
    (hcov : ∀ t, t < n29 → (c29_0 r0 t || c29_1 r1 t || c29_2 r2 t || c29_3 r3 t || c29_4 r4 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n29, (1 : ℤ) + (Wl29_0 r0 r1 r2 r3 r4 t + Wl29_1 r0 r1 r2 r3 r4 t + Wl29_2 r0 r1 r2 r3 r4 t + Wl29_3 r0 r1 r2 r3 r4 t + Wl29_4 r0 r1 r2 r3 r4 t + Wl29_5 r0 r1 r2 r3 r4 t + Wl29_6 r0 r1 r2 r3 r4 t + Wl29_7 r0 r1 r2 r3 r4 t + Wl29_8 r0 r1 r2 r3 r4 t + Wl29_9 r0 r1 r2 r3 r4 t) ≤ Dg29 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Wl29_0, Wl29_1, Wl29_2, Wl29_3, Wl29_4, Wl29_5, Wl29_6, Wl29_7, Wl29_8, Wl29_9, Dg29]
    exact CaseSplit.lowest5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n29, (1 : ℤ) ≤ Dg29 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Dg29]
    exact CaseSplit.degpos5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n29 : ℤ) + ((∑ t ∈ Finset.range n29, Wl29_0 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n29, Wl29_1 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n29, Wl29_2 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n29, Wl29_3 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n29, Wl29_4 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n29, Wl29_5 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n29, Wl29_6 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n29, Wl29_7 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n29, Wl29_8 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n29, Wl29_9 r0 r1 r2 r3 r4 t)) ≤ ∑ t ∈ Finset.range n29, Dg29 r0 r1 r2 r3 r4 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N29_0 r0 r1 ≤ ∑ t ∈ Finset.range n29, Wl29_0 r0 r1 r2 r3 r4 t := by
    simp only [N29_0, Wl29_0, le_refl]
  have hn1 : N29_1 r0 r2 ≤ ∑ t ∈ Finset.range n29, Wl29_1 r0 r1 r2 r3 r4 t := by
    simp only [N29_1, Wl29_1, le_refl]
  have hn2 : N29_2 r0 r3 ≤ ∑ t ∈ Finset.range n29, Wl29_2 r0 r1 r2 r3 r4 t := by
    simp only [N29_2, Wl29_2, le_refl]
  have hn3 : N29_3 r0 r4 ≤ ∑ t ∈ Finset.range n29, Wl29_3 r0 r1 r2 r3 r4 t := by
    simp only [N29_3, Wl29_3, le_refl]
  have hn4 : N29_4 r1 r2 ≤ ∑ t ∈ Finset.range n29, Wl29_4 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n29, Wl29_4 r0 r1 r2 r3 r4 t
        = (if c29_1 r1 t && c29_2 r2 t then (1:ℤ) else 0)
          - (if c29_1 r1 t && c29_2 r2 t && c29_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl29_4]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n29, Wl29_4 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl29_4]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n29, Wl29_4 r0 r1 r2 r3 r4 t
        = P29_4 r1 r2 - C29_4 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P29_4, C29_4]
    have hm : C29_4 r1 r2 r0 ≤ M29_4 r1 r2 :=
      CaseSplit.le_mxr (C29_4 r1 r2) 10 r0 (by omega)
    simp only [N29_4]
    split
    · rw [hL]; omega
    · exact hnn
  have hn5 : N29_5 r1 r3 ≤ ∑ t ∈ Finset.range n29, Wl29_5 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n29, Wl29_5 r0 r1 r2 r3 r4 t
        = (if c29_1 r1 t && c29_3 r3 t then (1:ℤ) else 0)
          - (if c29_1 r1 t && c29_3 r3 t && c29_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl29_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n29, Wl29_5 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl29_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n29, Wl29_5 r0 r1 r2 r3 r4 t
        = P29_5 r1 r3 - C29_5 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P29_5, C29_5]
    have hm : C29_5 r1 r3 r0 ≤ M29_5 r1 r3 :=
      CaseSplit.le_mxr (C29_5 r1 r3) 10 r0 (by omega)
    simp only [N29_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N29_6 r1 r4 ≤ ∑ t ∈ Finset.range n29, Wl29_6 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n29, Wl29_6 r0 r1 r2 r3 r4 t
        = (if c29_1 r1 t && c29_4 r4 t then (1:ℤ) else 0)
          - (if c29_1 r1 t && c29_4 r4 t && c29_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl29_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n29, Wl29_6 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl29_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n29, Wl29_6 r0 r1 r2 r3 r4 t
        = P29_6 r1 r4 - C29_6 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P29_6, C29_6]
    have hm : C29_6 r1 r4 r0 ≤ M29_6 r1 r4 :=
      CaseSplit.le_mxr (C29_6 r1 r4) 10 r0 (by omega)
    simp only [N29_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N29_7 r2 r3 ≤ ∑ t ∈ Finset.range n29, Wl29_7 r0 r1 r2 r3 r4 t := by
    simp only [N29_7]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl29_7]
    exact CaseSplit.ind_nonneg _
  have hn8 : N29_8 r2 r4 ≤ ∑ t ∈ Finset.range n29, Wl29_8 r0 r1 r2 r3 r4 t := by
    simp only [N29_8]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl29_8]
    exact CaseSplit.ind_nonneg _
  have hn9 : N29_9 r3 r4 ≤ ∑ t ∈ Finset.range n29, Wl29_9 r0 r1 r2 r3 r4 t := by
    simp only [N29_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl29_9]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n29, (w29 t + 1) * Dg29 r0 r1 r2 r3 r4 t = S29_0 r0 + S29_1 r1 + S29_2 r2 + S29_3 r3 + S29_4 r4 := by
    simp only [S29_0, S29_1, S29_2, S29_3, S29_4, Dg29, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n29, (w29 t + 1) * Dg29 r0 r1 r2 r3 r4 t
      = (∑ t ∈ Finset.range n29, w29 t * Dg29 r0 r1 r2 r3 r4 t)
        + 1 * (∑ t ∈ Finset.range n29, Dg29 r0 r1 r2 r3 r4 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n29, w29 t)
      ≤ ∑ t ∈ Finset.range n29, w29 t * Dg29 r0 r1 r2 r3 r4 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg29 r0 r1 r2 r3 r4 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w29 t := wnn29 t (Finset.mem_range.mp ht)
    calc w29 t = w29 t * 1 := (mul_one _).symm
      _ ≤ w29 t * Dg29 r0 r1 r2 r3 r4 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS29_0 r0 + aS29_1 r1 + aS29_2 r2 + aS29_3 r3 + aS29_4 r4) + (aP29_0 r0 r1 + aP29_1 r0 r2 + aP29_2 r0 r3 + aP29_3 r0 r4 + aP29_4 r1 r2 + aP29_5 r1 r3 + aP29_6 r1 r4 + aP29_7 r2 r3 + aP29_8 r2 r4 + aP29_9 r3 r4) = (S29_0 r0 + S29_1 r1 + S29_2 r2 + S29_3 r3 + S29_4 r4) - 1 * (N29_0 r0 r1 + N29_1 r0 r2 + N29_2 r0 r3 + N29_3 r0 r4 + N29_4 r1 r2 + N29_5 r1 r3 + N29_6 r1 r4 + N29_7 r2 r3 + N29_8 r2 r4 + N29_9 r3 r4) := by
    simp only [aS29_0, aS29_1, aS29_2, aS29_3, aS29_4, aP29_0, aP29_1, aP29_2, aP29_3, aP29_4, aP29_5, aP29_6, aP29_7, aP29_8, aP29_9, L29_0, L29_1, L29_2, L29_3, L29_4]
    ring
  have bS0 : aS29_0 r0 ≤ MS29_0 := CaseSplit.le_mxr (aS29_0) 10 r0 (by omega)
  have bS1 : aS29_1 r1 ≤ MS29_1 := CaseSplit.le_mxr (aS29_1) 12 r1 (by omega)
  have bS2 : aS29_2 r2 ≤ MS29_2 := CaseSplit.le_mxr (aS29_2) 16 r2 (by omega)
  have bS3 : aS29_3 r3 ≤ MS29_3 := CaseSplit.le_mxr (aS29_3) 18 r3 (by omega)
  have bS4 : aS29_4 r4 ≤ MS29_4 := CaseSplit.le_mxr (aS29_4) 22 r4 (by omega)
  have bP0 : aP29_0 r0 r1 ≤ MP29_0 := CaseSplit.le_mxr2 (aP29_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP29_1 r0 r2 ≤ MP29_1 := CaseSplit.le_mxr2 (aP29_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP29_2 r0 r3 ≤ MP29_2 := CaseSplit.le_mxr2 (aP29_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP29_3 r0 r4 ≤ MP29_3 := CaseSplit.le_mxr2 (aP29_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP29_4 r1 r2 ≤ MP29_4 := CaseSplit.le_mxr2 (aP29_4) 12 16 r1 r2 (by omega) (by omega)
  have bP5 : aP29_5 r1 r3 ≤ MP29_5 := CaseSplit.le_mxr2 (aP29_5) 12 18 r1 r3 (by omega) (by omega)
  have bP6 : aP29_6 r1 r4 ≤ MP29_6 := CaseSplit.le_mxr2 (aP29_6) 12 22 r1 r4 (by omega) (by omega)
  have bP7 : aP29_7 r2 r3 ≤ MP29_7 := CaseSplit.le_mxr2 (aP29_7) 16 18 r2 r3 (by omega) (by omega)
  have bP8 : aP29_8 r2 r4 ≤ MP29_8 := CaseSplit.le_mxr2 (aP29_8) 16 22 r2 r4 (by omega) (by omega)
  have bP9 : aP29_9 r3 r4 ≤ MP29_9 := CaseSplit.le_mxr2 (aP29_9) 18 22 r3 r4 (by omega) (by omega)
  have hrhs : rhs29 = (∑ t ∈ Finset.range n29, w29 t) + 1 * (n29 : ℤ) := rfl
  have hc := cert29
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, bS0, bS1, bS2, bS3, bS4, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9]

end IncCert23
