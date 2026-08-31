/-
INCREMENT-WIDTH CERTIFICATE, step 19->23, case 26 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_19_23.json, which re-derives every number
from the primes alone).

Machine 23, INCREMENT width 39 = F_2(19) + s_min(23) = 31 + 8,
held gears [5, 7] at phases [3, 5].  Free gears [11, 13, 17, 19, 23].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 1.
-/
import IncCert23B

namespace IncCert23

/-! ### case 26: held gears at phases [3, 5] -/

def p26 : List ℕ := [0, 2, 4, 5, 7, 9, 12, 14, 19, 20, 25, 27, 30, 32, 34, 35, 37]
def q26 (t : ℕ) : ℕ := p26.getD t 0
def n26 : ℕ := 17
def yl26 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def w26 (t : ℕ) : ℤ := yl26.getD t 0
def ul26 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), 1, 1, (-1), 0, 1, 1, 0, (-1), 1, 1, 0, 3, 3, 2, 2, 1, 2, 3, 3, 1, 1, 2, 3, 3, 2, 2, 1, 2, (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1, 3, 1, 2, 3, 1, 4, 3, 1, 3, 1, 3, 3, 2, 4, 2, 4, 3, 2, 3, 1, 4, 1, 2, 1, 2, 1, 0, 1, 1, 1, 1, 2, 2, 0, 0, 2, 2, 2, 1, 0]
def u26 (k : ℕ) : ℤ := ul26.getD k 0

def c26_0 (r t : ℕ) : Bool := gb11 r (q26 t)
def c26_1 (r t : ℕ) : Bool := gb13 r (q26 t)
def c26_2 (r t : ℕ) : Bool := gb17 r (q26 t)
def c26_3 (r t : ℕ) : Bool := gb19 r (q26 t)
def c26_4 (r t : ℕ) : Bool := gb23 r (q26 t)

def S26_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (w26 t + 1) * (if c26_0 r t then 1 else 0)
def S26_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (w26 t + 1) * (if c26_1 r t then 1 else 0)
def S26_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (w26 t + 1) * (if c26_2 r t then 1 else 0)
def S26_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (w26 t + 1) * (if c26_3 r t then 1 else 0)
def S26_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (w26 t + 1) * (if c26_4 r t then 1 else 0)

def L26_0 (r : ℕ) : ℤ := u26 (13 + r) + u26 (41 + r) + u26 (71 + r) + u26 (105 + r)
def L26_1 (r : ℕ) : ℤ := u26 (0 + r) + u26 (133 + r) + u26 (165 + r) + u26 (201 + r)
def L26_2 (r : ℕ) : ℤ := u26 (24 + r) + u26 (116 + r) + u26 (233 + r) + u26 (273 + r)
def L26_3 (r : ℕ) : ℤ := u26 (52 + r) + u26 (146 + r) + u26 (214 + r) + u26 (313 + r)
def L26_4 (r : ℕ) : ℤ := u26 (82 + r) + u26 (178 + r) + u26 (250 + r) + u26 (290 + r)

def aS26_0 (r : ℕ) : ℤ := S26_0 r - L26_0 r
def MS26_0 : ℤ := CaseSplit.mxr (aS26_0) 10
def aS26_1 (r : ℕ) : ℤ := S26_1 r - L26_1 r
def MS26_1 : ℤ := CaseSplit.mxr (aS26_1) 12
def aS26_2 (r : ℕ) : ℤ := S26_2 r - L26_2 r
def MS26_2 : ℤ := CaseSplit.mxr (aS26_2) 16
def aS26_3 (r : ℕ) : ℤ := S26_3 r - L26_3 r
def MS26_3 : ℤ := CaseSplit.mxr (aS26_3) 18
def aS26_4 (r : ℕ) : ℤ := S26_4 r - L26_4 r
def MS26_4 : ℤ := CaseSplit.mxr (aS26_4) 22

def N26_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (if c26_0 ra t && c26_1 rb t then 1 else 0)
def aP26_0 (ra rb : ℕ) : ℤ := -(1) * N26_0 ra rb + u26 (0 + rb) + u26 (13 + ra)
def MP26_0 : ℤ := CaseSplit.mxr2 (aP26_0) 10 12
def N26_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (if c26_0 ra t && c26_2 rb t then 1 else 0)
def aP26_1 (ra rb : ℕ) : ℤ := -(1) * N26_1 ra rb + u26 (24 + rb) + u26 (41 + ra)
def MP26_1 : ℤ := CaseSplit.mxr2 (aP26_1) 10 16
def N26_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (if c26_0 ra t && c26_3 rb t then 1 else 0)
def aP26_2 (ra rb : ℕ) : ℤ := -(1) * N26_2 ra rb + u26 (52 + rb) + u26 (71 + ra)
def MP26_2 : ℤ := CaseSplit.mxr2 (aP26_2) 10 18
def N26_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (if c26_0 ra t && c26_4 rb t then 1 else 0)
def aP26_3 (ra rb : ℕ) : ℤ := -(1) * N26_3 ra rb + u26 (82 + rb) + u26 (105 + ra)
def MP26_3 : ℤ := CaseSplit.mxr2 (aP26_3) 10 22
def P26_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (if c26_1 ra t && c26_2 rb t then 1 else 0)
def C26_4 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (if c26_1 ra t && c26_2 rb t && c26_0 s t then 1 else 0)
def M26_4 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C26_4 ra rb) 10
def E26_4 : List ℕ := [68, 79, 154, 165]
def N26_4 (ra rb : ℕ) : ℤ := if E26_4.contains (ra * 17 + rb) = true then P26_4 ra rb - M26_4 ra rb else 0
def aP26_4 (ra rb : ℕ) : ℤ := -(1) * N26_4 ra rb + u26 (116 + rb) + u26 (133 + ra)
def MP26_4 : ℤ := CaseSplit.mxr2 (aP26_4) 12 16
def P26_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (if c26_1 ra t && c26_3 rb t then 1 else 0)
def C26_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (if c26_1 ra t && c26_3 rb t && c26_0 s t then 1 else 0)
def M26_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C26_5 ra rb) 10
def E26_5 : List ℕ := [27, 67, 91, 98, 167, 174, 198, 238]
def N26_5 (ra rb : ℕ) : ℤ := if E26_5.contains (ra * 19 + rb) = true then P26_5 ra rb - M26_5 ra rb else 0
def aP26_5 (ra rb : ℕ) : ℤ := -(1) * N26_5 ra rb + u26 (146 + rb) + u26 (165 + ra)
def MP26_5 : ℤ := CaseSplit.mxr2 (aP26_5) 12 18
def P26_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (if c26_1 ra t && c26_4 rb t then 1 else 0)
def C26_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n26, (if c26_1 ra t && c26_4 rb t && c26_0 s t then 1 else 0)
def M26_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C26_6 ra rb) 10
def E26_6 : List ℕ := []
def N26_6 (ra rb : ℕ) : ℤ := if E26_6.contains (ra * 23 + rb) = true then P26_6 ra rb - M26_6 ra rb else 0
def aP26_6 (ra rb : ℕ) : ℤ := -(1) * N26_6 ra rb + u26 (178 + rb) + u26 (201 + ra)
def MP26_6 : ℤ := CaseSplit.mxr2 (aP26_6) 12 22
def N26_7 (_ra _rb : ℕ) : ℤ := 0
def aP26_7 (ra rb : ℕ) : ℤ := -(1) * N26_7 ra rb + u26 (214 + rb) + u26 (233 + ra)
def MP26_7 : ℤ := CaseSplit.mxr2 (aP26_7) 16 18
def N26_8 (_ra _rb : ℕ) : ℤ := 0
def aP26_8 (ra rb : ℕ) : ℤ := -(1) * N26_8 ra rb + u26 (250 + rb) + u26 (273 + ra)
def MP26_8 : ℤ := CaseSplit.mxr2 (aP26_8) 16 22
def N26_9 (_ra _rb : ℕ) : ℤ := 0
def aP26_9 (ra rb : ℕ) : ℤ := -(1) * N26_9 ra rb + u26 (290 + rb) + u26 (313 + ra)
def MP26_9 : ℤ := CaseSplit.mxr2 (aP26_9) 18 22

def rhs26 : ℤ := (∑ t ∈ Finset.range n26, w26 t) + 1 * (n26 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn26 : ∀ t, t < n26 → (0 : ℤ) ≤ w26 t := by decide
theorem plt26 : ∀ t, t < n26 → q26 t < 39 := by decide
theorem pfree26_5 : ∀ t, t < n26 → gb5 3 (q26 t) = false := by decide
theorem pfree26_7 : ∀ t, t < n26 → gb7 5 (q26 t) = false := by decide
theorem MSv26_0 : MS26_0 = 3 := by decide +kernel
theorem MSv26_1 : MS26_1 = 7 := by decide +kernel
theorem MSv26_2 : MS26_2 = 0 := by decide +kernel
theorem MSv26_3 : MS26_3 = 0 := by decide +kernel
theorem MSv26_4 : MS26_4 = 0 := by decide +kernel
theorem MPv26_0 : MP26_0 = 0 := by decide +kernel
theorem MPv26_1 : MP26_1 = 0 := by decide +kernel
theorem MPv26_2 : MP26_2 = 0 := by decide +kernel
theorem MPv26_3 : MP26_3 = 0 := by decide +kernel
theorem MPv26_4 : MP26_4 = 0 := by decide +kernel
theorem MPv26_5 : MP26_5 = 0 := by decide +kernel
theorem MPv26_6 : MP26_6 = 0 := by decide +kernel
theorem MPv26_7 : MP26_7 = 0 := by decide +kernel
theorem MPv26_8 : MP26_8 = 0 := by decide +kernel
theorem MPv26_9 : MP26_9 = 6 := by decide +kernel
theorem rhsv26 : rhs26 = 17 := by decide +kernel

/-- **The case-26 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/1.
    (Scaled by the common denominator 1: 16 < 17.) -/
theorem cert26 : MS26_0 + MS26_1 + MS26_2 + MS26_3 + MS26_4 + MP26_0 + MP26_1 + MP26_2 + MP26_3 + MP26_4 + MP26_5 + MP26_6 + MP26_7 + MP26_8 + MP26_9 < rhs26 := by
  rw [MSv26_0, MSv26_1, MSv26_2, MSv26_3, MSv26_4, MPv26_0, MPv26_1, MPv26_2, MPv26_3, MPv26_4, MPv26_5, MPv26_6, MPv26_7, MPv26_8, MPv26_9, rhsv26]
  decide

def Dg26 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := (if c26_0 r0 t then 1 else 0) + (if c26_1 r1 t then 1 else 0) + (if c26_2 r2 t then 1 else 0) + (if c26_3 r3 t then 1 else 0) + (if c26_4 r4 t then 1 else 0)
def Wl26_0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c26_0 r0 t && c26_1 r1 t then 1 else 0
def Wl26_1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c26_0 r0 t && c26_2 r2 t then 1 else 0
def Wl26_2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c26_0 r0 t && c26_3 r3 t then 1 else 0
def Wl26_3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c26_0 r0 t && c26_4 r4 t then 1 else 0
def Wl26_4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c26_0 r0 t && c26_1 r1 t && c26_2 r2 t then 1 else 0
def Wl26_5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c26_0 r0 t && c26_1 r1 t && c26_3 r3 t then 1 else 0
def Wl26_6 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c26_0 r0 t && c26_1 r1 t && c26_4 r4 t then 1 else 0
def Wl26_7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c26_0 r0 t && !c26_1 r1 t && c26_2 r2 t && c26_3 r3 t then 1 else 0
def Wl26_8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c26_0 r0 t && !c26_1 r1 t && c26_2 r2 t && c26_4 r4 t then 1 else 0
def Wl26_9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c26_0 r0 t && !c26_1 r1 t && !c26_2 r2 t && c26_3 r3 t && c26_4 r4 t then 1 else 0

/-- **No configuration blocks the whole window in case 26.** -/
theorem nocov26 {r0 r1 r2 r3 r4 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23)
    (hcov : ∀ t, t < n26 → (c26_0 r0 t || c26_1 r1 t || c26_2 r2 t || c26_3 r3 t || c26_4 r4 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n26, (1 : ℤ) + (Wl26_0 r0 r1 r2 r3 r4 t + Wl26_1 r0 r1 r2 r3 r4 t + Wl26_2 r0 r1 r2 r3 r4 t + Wl26_3 r0 r1 r2 r3 r4 t + Wl26_4 r0 r1 r2 r3 r4 t + Wl26_5 r0 r1 r2 r3 r4 t + Wl26_6 r0 r1 r2 r3 r4 t + Wl26_7 r0 r1 r2 r3 r4 t + Wl26_8 r0 r1 r2 r3 r4 t + Wl26_9 r0 r1 r2 r3 r4 t) ≤ Dg26 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Wl26_0, Wl26_1, Wl26_2, Wl26_3, Wl26_4, Wl26_5, Wl26_6, Wl26_7, Wl26_8, Wl26_9, Dg26]
    exact CaseSplit.lowest5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n26, (1 : ℤ) ≤ Dg26 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Dg26]
    exact CaseSplit.degpos5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n26 : ℤ) + ((∑ t ∈ Finset.range n26, Wl26_0 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n26, Wl26_1 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n26, Wl26_2 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n26, Wl26_3 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n26, Wl26_4 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n26, Wl26_5 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n26, Wl26_6 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n26, Wl26_7 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n26, Wl26_8 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n26, Wl26_9 r0 r1 r2 r3 r4 t)) ≤ ∑ t ∈ Finset.range n26, Dg26 r0 r1 r2 r3 r4 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N26_0 r0 r1 ≤ ∑ t ∈ Finset.range n26, Wl26_0 r0 r1 r2 r3 r4 t := by
    simp only [N26_0, Wl26_0, le_refl]
  have hn1 : N26_1 r0 r2 ≤ ∑ t ∈ Finset.range n26, Wl26_1 r0 r1 r2 r3 r4 t := by
    simp only [N26_1, Wl26_1, le_refl]
  have hn2 : N26_2 r0 r3 ≤ ∑ t ∈ Finset.range n26, Wl26_2 r0 r1 r2 r3 r4 t := by
    simp only [N26_2, Wl26_2, le_refl]
  have hn3 : N26_3 r0 r4 ≤ ∑ t ∈ Finset.range n26, Wl26_3 r0 r1 r2 r3 r4 t := by
    simp only [N26_3, Wl26_3, le_refl]
  have hn4 : N26_4 r1 r2 ≤ ∑ t ∈ Finset.range n26, Wl26_4 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n26, Wl26_4 r0 r1 r2 r3 r4 t
        = (if c26_1 r1 t && c26_2 r2 t then (1:ℤ) else 0)
          - (if c26_1 r1 t && c26_2 r2 t && c26_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl26_4]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n26, Wl26_4 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl26_4]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n26, Wl26_4 r0 r1 r2 r3 r4 t
        = P26_4 r1 r2 - C26_4 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P26_4, C26_4]
    have hm : C26_4 r1 r2 r0 ≤ M26_4 r1 r2 :=
      CaseSplit.le_mxr (C26_4 r1 r2) 10 r0 (by omega)
    simp only [N26_4]
    split
    · rw [hL]; omega
    · exact hnn
  have hn5 : N26_5 r1 r3 ≤ ∑ t ∈ Finset.range n26, Wl26_5 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n26, Wl26_5 r0 r1 r2 r3 r4 t
        = (if c26_1 r1 t && c26_3 r3 t then (1:ℤ) else 0)
          - (if c26_1 r1 t && c26_3 r3 t && c26_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl26_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n26, Wl26_5 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl26_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n26, Wl26_5 r0 r1 r2 r3 r4 t
        = P26_5 r1 r3 - C26_5 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P26_5, C26_5]
    have hm : C26_5 r1 r3 r0 ≤ M26_5 r1 r3 :=
      CaseSplit.le_mxr (C26_5 r1 r3) 10 r0 (by omega)
    simp only [N26_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N26_6 r1 r4 ≤ ∑ t ∈ Finset.range n26, Wl26_6 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n26, Wl26_6 r0 r1 r2 r3 r4 t
        = (if c26_1 r1 t && c26_4 r4 t then (1:ℤ) else 0)
          - (if c26_1 r1 t && c26_4 r4 t && c26_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl26_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n26, Wl26_6 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl26_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n26, Wl26_6 r0 r1 r2 r3 r4 t
        = P26_6 r1 r4 - C26_6 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P26_6, C26_6]
    have hm : C26_6 r1 r4 r0 ≤ M26_6 r1 r4 :=
      CaseSplit.le_mxr (C26_6 r1 r4) 10 r0 (by omega)
    simp only [N26_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N26_7 r2 r3 ≤ ∑ t ∈ Finset.range n26, Wl26_7 r0 r1 r2 r3 r4 t := by
    simp only [N26_7]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl26_7]
    exact CaseSplit.ind_nonneg _
  have hn8 : N26_8 r2 r4 ≤ ∑ t ∈ Finset.range n26, Wl26_8 r0 r1 r2 r3 r4 t := by
    simp only [N26_8]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl26_8]
    exact CaseSplit.ind_nonneg _
  have hn9 : N26_9 r3 r4 ≤ ∑ t ∈ Finset.range n26, Wl26_9 r0 r1 r2 r3 r4 t := by
    simp only [N26_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl26_9]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n26, (w26 t + 1) * Dg26 r0 r1 r2 r3 r4 t = S26_0 r0 + S26_1 r1 + S26_2 r2 + S26_3 r3 + S26_4 r4 := by
    simp only [S26_0, S26_1, S26_2, S26_3, S26_4, Dg26, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n26, (w26 t + 1) * Dg26 r0 r1 r2 r3 r4 t
      = (∑ t ∈ Finset.range n26, w26 t * Dg26 r0 r1 r2 r3 r4 t)
        + 1 * (∑ t ∈ Finset.range n26, Dg26 r0 r1 r2 r3 r4 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n26, w26 t)
      ≤ ∑ t ∈ Finset.range n26, w26 t * Dg26 r0 r1 r2 r3 r4 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg26 r0 r1 r2 r3 r4 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w26 t := wnn26 t (Finset.mem_range.mp ht)
    calc w26 t = w26 t * 1 := (mul_one _).symm
      _ ≤ w26 t * Dg26 r0 r1 r2 r3 r4 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS26_0 r0 + aS26_1 r1 + aS26_2 r2 + aS26_3 r3 + aS26_4 r4) + (aP26_0 r0 r1 + aP26_1 r0 r2 + aP26_2 r0 r3 + aP26_3 r0 r4 + aP26_4 r1 r2 + aP26_5 r1 r3 + aP26_6 r1 r4 + aP26_7 r2 r3 + aP26_8 r2 r4 + aP26_9 r3 r4) = (S26_0 r0 + S26_1 r1 + S26_2 r2 + S26_3 r3 + S26_4 r4) - 1 * (N26_0 r0 r1 + N26_1 r0 r2 + N26_2 r0 r3 + N26_3 r0 r4 + N26_4 r1 r2 + N26_5 r1 r3 + N26_6 r1 r4 + N26_7 r2 r3 + N26_8 r2 r4 + N26_9 r3 r4) := by
    simp only [aS26_0, aS26_1, aS26_2, aS26_3, aS26_4, aP26_0, aP26_1, aP26_2, aP26_3, aP26_4, aP26_5, aP26_6, aP26_7, aP26_8, aP26_9, L26_0, L26_1, L26_2, L26_3, L26_4]
    ring
  have bS0 : aS26_0 r0 ≤ MS26_0 := CaseSplit.le_mxr (aS26_0) 10 r0 (by omega)
  have bS1 : aS26_1 r1 ≤ MS26_1 := CaseSplit.le_mxr (aS26_1) 12 r1 (by omega)
  have bS2 : aS26_2 r2 ≤ MS26_2 := CaseSplit.le_mxr (aS26_2) 16 r2 (by omega)
  have bS3 : aS26_3 r3 ≤ MS26_3 := CaseSplit.le_mxr (aS26_3) 18 r3 (by omega)
  have bS4 : aS26_4 r4 ≤ MS26_4 := CaseSplit.le_mxr (aS26_4) 22 r4 (by omega)
  have bP0 : aP26_0 r0 r1 ≤ MP26_0 := CaseSplit.le_mxr2 (aP26_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP26_1 r0 r2 ≤ MP26_1 := CaseSplit.le_mxr2 (aP26_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP26_2 r0 r3 ≤ MP26_2 := CaseSplit.le_mxr2 (aP26_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP26_3 r0 r4 ≤ MP26_3 := CaseSplit.le_mxr2 (aP26_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP26_4 r1 r2 ≤ MP26_4 := CaseSplit.le_mxr2 (aP26_4) 12 16 r1 r2 (by omega) (by omega)
  have bP5 : aP26_5 r1 r3 ≤ MP26_5 := CaseSplit.le_mxr2 (aP26_5) 12 18 r1 r3 (by omega) (by omega)
  have bP6 : aP26_6 r1 r4 ≤ MP26_6 := CaseSplit.le_mxr2 (aP26_6) 12 22 r1 r4 (by omega) (by omega)
  have bP7 : aP26_7 r2 r3 ≤ MP26_7 := CaseSplit.le_mxr2 (aP26_7) 16 18 r2 r3 (by omega) (by omega)
  have bP8 : aP26_8 r2 r4 ≤ MP26_8 := CaseSplit.le_mxr2 (aP26_8) 16 22 r2 r4 (by omega) (by omega)
  have bP9 : aP26_9 r3 r4 ≤ MP26_9 := CaseSplit.le_mxr2 (aP26_9) 18 22 r3 r4 (by omega) (by omega)
  have hrhs : rhs26 = (∑ t ∈ Finset.range n26, w26 t) + 1 * (n26 : ℤ) := rfl
  have hc := cert26
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, bS0, bS1, bS2, bS3, bS4, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9]

end IncCert23
