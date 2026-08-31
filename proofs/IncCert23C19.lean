/-
INCREMENT-WIDTH CERTIFICATE, step 19->23, case 19 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_19_23.json, which re-derives every number
from the primes alone).

Machine 23, INCREMENT width 39 = F_2(19) + s_min(23) = 31 + 8,
held gears [5, 7] at phases [2, 5].  Free gears [11, 13, 17, 19, 23].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 1.
-/
import IncCert23B

namespace IncCert23

/-! ### case 19: held gears at phases [2, 5] -/

def p19 : List ℕ := [0, 5, 6, 11, 13, 16, 18, 20, 21, 23, 25, 26, 28, 30, 33, 35]
def q19 (t : ℕ) : ℕ := p19.getD t 0
def n19 : ℕ := 16
def yl19 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def w19 (t : ℕ) : ℤ := yl19.getD t 0
def ul19 : List ℤ := [(-1), (-1), 0, (-1), (-1), (-1), (-1), (-2), (-1), (-1), (-1), 0, (-1), 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), (-1), 0, 0, (-1), 0, (-1), (-1), 0, (-1), (-1), 0, (-1), (-1), (-1), (-1), 0, (-1), (-1), 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-2), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 3, 1, 2, 3, 3, 1, (-4), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 2, 3, 3, 1, 3, 3, 3, 3, 1, 3, 2, 2, 3, 3, 3, 3, 1, 3, 2, 3, 3, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
def u19 (k : ℕ) : ℤ := ul19.getD k 0

def c19_0 (r t : ℕ) : Bool := gb11 r (q19 t)
def c19_1 (r t : ℕ) : Bool := gb13 r (q19 t)
def c19_2 (r t : ℕ) : Bool := gb17 r (q19 t)
def c19_3 (r t : ℕ) : Bool := gb19 r (q19 t)
def c19_4 (r t : ℕ) : Bool := gb23 r (q19 t)

def S19_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n19, (w19 t + 1) * (if c19_0 r t then 1 else 0)
def S19_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n19, (w19 t + 1) * (if c19_1 r t then 1 else 0)
def S19_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n19, (w19 t + 1) * (if c19_2 r t then 1 else 0)
def S19_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n19, (w19 t + 1) * (if c19_3 r t then 1 else 0)
def S19_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n19, (w19 t + 1) * (if c19_4 r t then 1 else 0)

def L19_0 (r : ℕ) : ℤ := u19 (13 + r) + u19 (41 + r) + u19 (71 + r) + u19 (105 + r)
def L19_1 (r : ℕ) : ℤ := u19 (0 + r) + u19 (133 + r) + u19 (165 + r) + u19 (201 + r)
def L19_2 (r : ℕ) : ℤ := u19 (24 + r) + u19 (116 + r) + u19 (233 + r) + u19 (273 + r)
def L19_3 (r : ℕ) : ℤ := u19 (52 + r) + u19 (146 + r) + u19 (214 + r) + u19 (313 + r)
def L19_4 (r : ℕ) : ℤ := u19 (82 + r) + u19 (178 + r) + u19 (250 + r) + u19 (290 + r)

def aS19_0 (r : ℕ) : ℤ := S19_0 r - L19_0 r
def MS19_0 : ℤ := CaseSplit.mxr (aS19_0) 10
def aS19_1 (r : ℕ) : ℤ := S19_1 r - L19_1 r
def MS19_1 : ℤ := CaseSplit.mxr (aS19_1) 12
def aS19_2 (r : ℕ) : ℤ := S19_2 r - L19_2 r
def MS19_2 : ℤ := CaseSplit.mxr (aS19_2) 16
def aS19_3 (r : ℕ) : ℤ := S19_3 r - L19_3 r
def MS19_3 : ℤ := CaseSplit.mxr (aS19_3) 18
def aS19_4 (r : ℕ) : ℤ := S19_4 r - L19_4 r
def MS19_4 : ℤ := CaseSplit.mxr (aS19_4) 22

def N19_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n19, (if c19_0 ra t && c19_1 rb t then 1 else 0)
def aP19_0 (ra rb : ℕ) : ℤ := -(1) * N19_0 ra rb + u19 (0 + rb) + u19 (13 + ra)
def MP19_0 : ℤ := CaseSplit.mxr2 (aP19_0) 10 12
def N19_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n19, (if c19_0 ra t && c19_2 rb t then 1 else 0)
def aP19_1 (ra rb : ℕ) : ℤ := -(1) * N19_1 ra rb + u19 (24 + rb) + u19 (41 + ra)
def MP19_1 : ℤ := CaseSplit.mxr2 (aP19_1) 10 16
def N19_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n19, (if c19_0 ra t && c19_3 rb t then 1 else 0)
def aP19_2 (ra rb : ℕ) : ℤ := -(1) * N19_2 ra rb + u19 (52 + rb) + u19 (71 + ra)
def MP19_2 : ℤ := CaseSplit.mxr2 (aP19_2) 10 18
def N19_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n19, (if c19_0 ra t && c19_4 rb t then 1 else 0)
def aP19_3 (ra rb : ℕ) : ℤ := -(1) * N19_3 ra rb + u19 (82 + rb) + u19 (105 + ra)
def MP19_3 : ℤ := CaseSplit.mxr2 (aP19_3) 10 22
def P19_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n19, (if c19_1 ra t && c19_2 rb t then 1 else 0)
def C19_4 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n19, (if c19_1 ra t && c19_2 rb t && c19_0 s t then 1 else 0)
def M19_4 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C19_4 ra rb) 10
def E19_4 : List ℕ := [3, 9, 93, 99, 104, 115, 140, 151, 188, 194]
def N19_4 (ra rb : ℕ) : ℤ := if E19_4.contains (ra * 17 + rb) = true then P19_4 ra rb - M19_4 ra rb else 0
def aP19_4 (ra rb : ℕ) : ℤ := -(1) * N19_4 ra rb + u19 (116 + rb) + u19 (133 + ra)
def MP19_4 : ℤ := CaseSplit.mxr2 (aP19_4) 12 16
def P19_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n19, (if c19_1 ra t && c19_3 rb t then 1 else 0)
def C19_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n19, (if c19_1 ra t && c19_3 rb t && c19_0 s t then 1 else 0)
def M19_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C19_5 ra rb) 10
def E19_5 : List ℕ := [41, 47, 78, 131, 154, 207, 212, 218]
def N19_5 (ra rb : ℕ) : ℤ := if E19_5.contains (ra * 19 + rb) = true then P19_5 ra rb - M19_5 ra rb else 0
def aP19_5 (ra rb : ℕ) : ℤ := -(1) * N19_5 ra rb + u19 (146 + rb) + u19 (165 + ra)
def MP19_5 : ℤ := CaseSplit.mxr2 (aP19_5) 12 18
def P19_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n19, (if c19_1 ra t && c19_4 rb t then 1 else 0)
def C19_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n19, (if c19_1 ra t && c19_4 rb t && c19_0 s t then 1 else 0)
def M19_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C19_6 ra rb) 10
def E19_6 : List ℕ := []
def N19_6 (ra rb : ℕ) : ℤ := if E19_6.contains (ra * 23 + rb) = true then P19_6 ra rb - M19_6 ra rb else 0
def aP19_6 (ra rb : ℕ) : ℤ := -(1) * N19_6 ra rb + u19 (178 + rb) + u19 (201 + ra)
def MP19_6 : ℤ := CaseSplit.mxr2 (aP19_6) 12 22
def N19_7 (_ra _rb : ℕ) : ℤ := 0
def aP19_7 (ra rb : ℕ) : ℤ := -(1) * N19_7 ra rb + u19 (214 + rb) + u19 (233 + ra)
def MP19_7 : ℤ := CaseSplit.mxr2 (aP19_7) 16 18
def N19_8 (_ra _rb : ℕ) : ℤ := 0
def aP19_8 (ra rb : ℕ) : ℤ := -(1) * N19_8 ra rb + u19 (250 + rb) + u19 (273 + ra)
def MP19_8 : ℤ := CaseSplit.mxr2 (aP19_8) 16 22
def N19_9 (_ra _rb : ℕ) : ℤ := 0
def aP19_9 (ra rb : ℕ) : ℤ := -(1) * N19_9 ra rb + u19 (290 + rb) + u19 (313 + ra)
def MP19_9 : ℤ := CaseSplit.mxr2 (aP19_9) 18 22

def rhs19 : ℤ := (∑ t ∈ Finset.range n19, w19 t) + 1 * (n19 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn19 : ∀ t, t < n19 → (0 : ℤ) ≤ w19 t := by decide
theorem plt19 : ∀ t, t < n19 → q19 t < 39 := by decide
theorem pfree19_5 : ∀ t, t < n19 → gb5 2 (q19 t) = false := by decide
theorem pfree19_7 : ∀ t, t < n19 → gb7 5 (q19 t) = false := by decide
theorem MSv19_0 : MS19_0 = 2 := by decide +kernel
theorem MSv19_1 : MS19_1 = 9 := by decide +kernel
theorem MSv19_2 : MS19_2 = 0 := by decide +kernel
theorem MSv19_3 : MS19_3 = 0 := by decide +kernel
theorem MSv19_4 : MS19_4 = 0 := by decide +kernel
theorem MPv19_0 : MP19_0 = 0 := by decide +kernel
theorem MPv19_1 : MP19_1 = 0 := by decide +kernel
theorem MPv19_2 : MP19_2 = 0 := by decide +kernel
theorem MPv19_3 : MP19_3 = 0 := by decide +kernel
theorem MPv19_4 : MP19_4 = 0 := by decide +kernel
theorem MPv19_5 : MP19_5 = 0 := by decide +kernel
theorem MPv19_6 : MP19_6 = 0 := by decide +kernel
theorem MPv19_7 : MP19_7 = 0 := by decide +kernel
theorem MPv19_8 : MP19_8 = 0 := by decide +kernel
theorem MPv19_9 : MP19_9 = 4 := by decide +kernel
theorem rhsv19 : rhs19 = 16 := by decide +kernel

/-- **The case-19 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/1.
    (Scaled by the common denominator 1: 15 < 16.) -/
theorem cert19 : MS19_0 + MS19_1 + MS19_2 + MS19_3 + MS19_4 + MP19_0 + MP19_1 + MP19_2 + MP19_3 + MP19_4 + MP19_5 + MP19_6 + MP19_7 + MP19_8 + MP19_9 < rhs19 := by
  rw [MSv19_0, MSv19_1, MSv19_2, MSv19_3, MSv19_4, MPv19_0, MPv19_1, MPv19_2, MPv19_3, MPv19_4, MPv19_5, MPv19_6, MPv19_7, MPv19_8, MPv19_9, rhsv19]
  decide

def Dg19 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := (if c19_0 r0 t then 1 else 0) + (if c19_1 r1 t then 1 else 0) + (if c19_2 r2 t then 1 else 0) + (if c19_3 r3 t then 1 else 0) + (if c19_4 r4 t then 1 else 0)
def Wl19_0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c19_0 r0 t && c19_1 r1 t then 1 else 0
def Wl19_1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c19_0 r0 t && c19_2 r2 t then 1 else 0
def Wl19_2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c19_0 r0 t && c19_3 r3 t then 1 else 0
def Wl19_3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c19_0 r0 t && c19_4 r4 t then 1 else 0
def Wl19_4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c19_0 r0 t && c19_1 r1 t && c19_2 r2 t then 1 else 0
def Wl19_5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c19_0 r0 t && c19_1 r1 t && c19_3 r3 t then 1 else 0
def Wl19_6 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c19_0 r0 t && c19_1 r1 t && c19_4 r4 t then 1 else 0
def Wl19_7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c19_0 r0 t && !c19_1 r1 t && c19_2 r2 t && c19_3 r3 t then 1 else 0
def Wl19_8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c19_0 r0 t && !c19_1 r1 t && c19_2 r2 t && c19_4 r4 t then 1 else 0
def Wl19_9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c19_0 r0 t && !c19_1 r1 t && !c19_2 r2 t && c19_3 r3 t && c19_4 r4 t then 1 else 0

/-- **No configuration blocks the whole window in case 19.** -/
theorem nocov19 {r0 r1 r2 r3 r4 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23)
    (hcov : ∀ t, t < n19 → (c19_0 r0 t || c19_1 r1 t || c19_2 r2 t || c19_3 r3 t || c19_4 r4 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n19, (1 : ℤ) + (Wl19_0 r0 r1 r2 r3 r4 t + Wl19_1 r0 r1 r2 r3 r4 t + Wl19_2 r0 r1 r2 r3 r4 t + Wl19_3 r0 r1 r2 r3 r4 t + Wl19_4 r0 r1 r2 r3 r4 t + Wl19_5 r0 r1 r2 r3 r4 t + Wl19_6 r0 r1 r2 r3 r4 t + Wl19_7 r0 r1 r2 r3 r4 t + Wl19_8 r0 r1 r2 r3 r4 t + Wl19_9 r0 r1 r2 r3 r4 t) ≤ Dg19 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Wl19_0, Wl19_1, Wl19_2, Wl19_3, Wl19_4, Wl19_5, Wl19_6, Wl19_7, Wl19_8, Wl19_9, Dg19]
    exact CaseSplit.lowest5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n19, (1 : ℤ) ≤ Dg19 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Dg19]
    exact CaseSplit.degpos5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n19 : ℤ) + ((∑ t ∈ Finset.range n19, Wl19_0 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n19, Wl19_1 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n19, Wl19_2 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n19, Wl19_3 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n19, Wl19_4 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n19, Wl19_5 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n19, Wl19_6 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n19, Wl19_7 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n19, Wl19_8 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n19, Wl19_9 r0 r1 r2 r3 r4 t)) ≤ ∑ t ∈ Finset.range n19, Dg19 r0 r1 r2 r3 r4 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N19_0 r0 r1 ≤ ∑ t ∈ Finset.range n19, Wl19_0 r0 r1 r2 r3 r4 t := by
    simp only [N19_0, Wl19_0, le_refl]
  have hn1 : N19_1 r0 r2 ≤ ∑ t ∈ Finset.range n19, Wl19_1 r0 r1 r2 r3 r4 t := by
    simp only [N19_1, Wl19_1, le_refl]
  have hn2 : N19_2 r0 r3 ≤ ∑ t ∈ Finset.range n19, Wl19_2 r0 r1 r2 r3 r4 t := by
    simp only [N19_2, Wl19_2, le_refl]
  have hn3 : N19_3 r0 r4 ≤ ∑ t ∈ Finset.range n19, Wl19_3 r0 r1 r2 r3 r4 t := by
    simp only [N19_3, Wl19_3, le_refl]
  have hn4 : N19_4 r1 r2 ≤ ∑ t ∈ Finset.range n19, Wl19_4 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n19, Wl19_4 r0 r1 r2 r3 r4 t
        = (if c19_1 r1 t && c19_2 r2 t then (1:ℤ) else 0)
          - (if c19_1 r1 t && c19_2 r2 t && c19_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl19_4]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n19, Wl19_4 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl19_4]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n19, Wl19_4 r0 r1 r2 r3 r4 t
        = P19_4 r1 r2 - C19_4 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P19_4, C19_4]
    have hm : C19_4 r1 r2 r0 ≤ M19_4 r1 r2 :=
      CaseSplit.le_mxr (C19_4 r1 r2) 10 r0 (by omega)
    simp only [N19_4]
    split
    · rw [hL]; omega
    · exact hnn
  have hn5 : N19_5 r1 r3 ≤ ∑ t ∈ Finset.range n19, Wl19_5 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n19, Wl19_5 r0 r1 r2 r3 r4 t
        = (if c19_1 r1 t && c19_3 r3 t then (1:ℤ) else 0)
          - (if c19_1 r1 t && c19_3 r3 t && c19_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl19_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n19, Wl19_5 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl19_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n19, Wl19_5 r0 r1 r2 r3 r4 t
        = P19_5 r1 r3 - C19_5 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P19_5, C19_5]
    have hm : C19_5 r1 r3 r0 ≤ M19_5 r1 r3 :=
      CaseSplit.le_mxr (C19_5 r1 r3) 10 r0 (by omega)
    simp only [N19_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N19_6 r1 r4 ≤ ∑ t ∈ Finset.range n19, Wl19_6 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n19, Wl19_6 r0 r1 r2 r3 r4 t
        = (if c19_1 r1 t && c19_4 r4 t then (1:ℤ) else 0)
          - (if c19_1 r1 t && c19_4 r4 t && c19_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl19_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n19, Wl19_6 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl19_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n19, Wl19_6 r0 r1 r2 r3 r4 t
        = P19_6 r1 r4 - C19_6 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P19_6, C19_6]
    have hm : C19_6 r1 r4 r0 ≤ M19_6 r1 r4 :=
      CaseSplit.le_mxr (C19_6 r1 r4) 10 r0 (by omega)
    simp only [N19_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N19_7 r2 r3 ≤ ∑ t ∈ Finset.range n19, Wl19_7 r0 r1 r2 r3 r4 t := by
    simp only [N19_7]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl19_7]
    exact CaseSplit.ind_nonneg _
  have hn8 : N19_8 r2 r4 ≤ ∑ t ∈ Finset.range n19, Wl19_8 r0 r1 r2 r3 r4 t := by
    simp only [N19_8]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl19_8]
    exact CaseSplit.ind_nonneg _
  have hn9 : N19_9 r3 r4 ≤ ∑ t ∈ Finset.range n19, Wl19_9 r0 r1 r2 r3 r4 t := by
    simp only [N19_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl19_9]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n19, (w19 t + 1) * Dg19 r0 r1 r2 r3 r4 t = S19_0 r0 + S19_1 r1 + S19_2 r2 + S19_3 r3 + S19_4 r4 := by
    simp only [S19_0, S19_1, S19_2, S19_3, S19_4, Dg19, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n19, (w19 t + 1) * Dg19 r0 r1 r2 r3 r4 t
      = (∑ t ∈ Finset.range n19, w19 t * Dg19 r0 r1 r2 r3 r4 t)
        + 1 * (∑ t ∈ Finset.range n19, Dg19 r0 r1 r2 r3 r4 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n19, w19 t)
      ≤ ∑ t ∈ Finset.range n19, w19 t * Dg19 r0 r1 r2 r3 r4 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg19 r0 r1 r2 r3 r4 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w19 t := wnn19 t (Finset.mem_range.mp ht)
    calc w19 t = w19 t * 1 := (mul_one _).symm
      _ ≤ w19 t * Dg19 r0 r1 r2 r3 r4 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS19_0 r0 + aS19_1 r1 + aS19_2 r2 + aS19_3 r3 + aS19_4 r4) + (aP19_0 r0 r1 + aP19_1 r0 r2 + aP19_2 r0 r3 + aP19_3 r0 r4 + aP19_4 r1 r2 + aP19_5 r1 r3 + aP19_6 r1 r4 + aP19_7 r2 r3 + aP19_8 r2 r4 + aP19_9 r3 r4) = (S19_0 r0 + S19_1 r1 + S19_2 r2 + S19_3 r3 + S19_4 r4) - 1 * (N19_0 r0 r1 + N19_1 r0 r2 + N19_2 r0 r3 + N19_3 r0 r4 + N19_4 r1 r2 + N19_5 r1 r3 + N19_6 r1 r4 + N19_7 r2 r3 + N19_8 r2 r4 + N19_9 r3 r4) := by
    simp only [aS19_0, aS19_1, aS19_2, aS19_3, aS19_4, aP19_0, aP19_1, aP19_2, aP19_3, aP19_4, aP19_5, aP19_6, aP19_7, aP19_8, aP19_9, L19_0, L19_1, L19_2, L19_3, L19_4]
    ring
  have bS0 : aS19_0 r0 ≤ MS19_0 := CaseSplit.le_mxr (aS19_0) 10 r0 (by omega)
  have bS1 : aS19_1 r1 ≤ MS19_1 := CaseSplit.le_mxr (aS19_1) 12 r1 (by omega)
  have bS2 : aS19_2 r2 ≤ MS19_2 := CaseSplit.le_mxr (aS19_2) 16 r2 (by omega)
  have bS3 : aS19_3 r3 ≤ MS19_3 := CaseSplit.le_mxr (aS19_3) 18 r3 (by omega)
  have bS4 : aS19_4 r4 ≤ MS19_4 := CaseSplit.le_mxr (aS19_4) 22 r4 (by omega)
  have bP0 : aP19_0 r0 r1 ≤ MP19_0 := CaseSplit.le_mxr2 (aP19_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP19_1 r0 r2 ≤ MP19_1 := CaseSplit.le_mxr2 (aP19_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP19_2 r0 r3 ≤ MP19_2 := CaseSplit.le_mxr2 (aP19_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP19_3 r0 r4 ≤ MP19_3 := CaseSplit.le_mxr2 (aP19_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP19_4 r1 r2 ≤ MP19_4 := CaseSplit.le_mxr2 (aP19_4) 12 16 r1 r2 (by omega) (by omega)
  have bP5 : aP19_5 r1 r3 ≤ MP19_5 := CaseSplit.le_mxr2 (aP19_5) 12 18 r1 r3 (by omega) (by omega)
  have bP6 : aP19_6 r1 r4 ≤ MP19_6 := CaseSplit.le_mxr2 (aP19_6) 12 22 r1 r4 (by omega) (by omega)
  have bP7 : aP19_7 r2 r3 ≤ MP19_7 := CaseSplit.le_mxr2 (aP19_7) 16 18 r2 r3 (by omega) (by omega)
  have bP8 : aP19_8 r2 r4 ≤ MP19_8 := CaseSplit.le_mxr2 (aP19_8) 16 22 r2 r4 (by omega) (by omega)
  have bP9 : aP19_9 r3 r4 ≤ MP19_9 := CaseSplit.le_mxr2 (aP19_9) 18 22 r3 r4 (by omega) (by omega)
  have hrhs : rhs19 = (∑ t ∈ Finset.range n19, w19 t) + 1 * (n19 : ℤ) := rfl
  have hc := cert19
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, bS0, bS1, bS2, bS3, bS4, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9]

end IncCert23
