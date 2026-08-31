/-
INCREMENT-WIDTH CERTIFICATE, step 19->23, case 17 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_19_23.json, which re-derives every number
from the primes alone).

Machine 23, INCREMENT width 39 = F_2(19) + s_min(23) = 31 + 8,
held gears [5, 7] at phases [2, 3].  Free gears [11, 13, 17, 19, 23].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 2.
-/
import IncCert23B

namespace IncCert23

/-! ### case 17: held gears at phases [2, 3] -/

def p17 : List ℕ := [0, 1, 6, 8, 11, 13, 15, 16, 18, 20, 21, 23, 25, 28, 30, 35, 36]
def q17 (t : ℕ) : ℕ := p17.getD t 0
def n17 : ℕ := 17
def yl17 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0]
def w17 (t : ℕ) : ℤ := yl17.getD t 0
def ul17 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-5), (-2), 0, (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-6), (-2), (-2), 0, (-1), (-2), (-2), 1, 2, 2, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, (-1), 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, (-1), (-1), (-1), (-1), (-1), (-1), 0, (-1), (-1), (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 3, 2, 4, 6, 6, 6, 6, 3, 6, 6, 5, 3, 6, (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), (-6), 3, 2, 3, 3, 3, 3, 3, 3, 2, 2, 3, 2, 0, 3, 3, 3, 3, 2, 3, (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 6, 4, 8, 8, 2, 8, 8, 4, 6, 2, 7, 8, 4, 8, 4, 4, 4, 4, 8, 8, 8, 7, 2, 2, 2, 2, 1, 2, 2, 2, 0, 0, 0, 2, 2, (-1), 1, 2, 2, 2, 0]
def u17 (k : ℕ) : ℤ := ul17.getD k 0

def c17_0 (r t : ℕ) : Bool := gb11 r (q17 t)
def c17_1 (r t : ℕ) : Bool := gb13 r (q17 t)
def c17_2 (r t : ℕ) : Bool := gb17 r (q17 t)
def c17_3 (r t : ℕ) : Bool := gb19 r (q17 t)
def c17_4 (r t : ℕ) : Bool := gb23 r (q17 t)

def S17_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n17, (w17 t + 2) * (if c17_0 r t then 1 else 0)
def S17_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n17, (w17 t + 2) * (if c17_1 r t then 1 else 0)
def S17_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n17, (w17 t + 2) * (if c17_2 r t then 1 else 0)
def S17_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n17, (w17 t + 2) * (if c17_3 r t then 1 else 0)
def S17_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n17, (w17 t + 2) * (if c17_4 r t then 1 else 0)

def L17_0 (r : ℕ) : ℤ := u17 (13 + r) + u17 (41 + r) + u17 (71 + r) + u17 (105 + r)
def L17_1 (r : ℕ) : ℤ := u17 (0 + r) + u17 (133 + r) + u17 (165 + r) + u17 (201 + r)
def L17_2 (r : ℕ) : ℤ := u17 (24 + r) + u17 (116 + r) + u17 (233 + r) + u17 (273 + r)
def L17_3 (r : ℕ) : ℤ := u17 (52 + r) + u17 (146 + r) + u17 (214 + r) + u17 (313 + r)
def L17_4 (r : ℕ) : ℤ := u17 (82 + r) + u17 (178 + r) + u17 (250 + r) + u17 (290 + r)

def aS17_0 (r : ℕ) : ℤ := S17_0 r - L17_0 r
def MS17_0 : ℤ := CaseSplit.mxr (aS17_0) 10
def aS17_1 (r : ℕ) : ℤ := S17_1 r - L17_1 r
def MS17_1 : ℤ := CaseSplit.mxr (aS17_1) 12
def aS17_2 (r : ℕ) : ℤ := S17_2 r - L17_2 r
def MS17_2 : ℤ := CaseSplit.mxr (aS17_2) 16
def aS17_3 (r : ℕ) : ℤ := S17_3 r - L17_3 r
def MS17_3 : ℤ := CaseSplit.mxr (aS17_3) 18
def aS17_4 (r : ℕ) : ℤ := S17_4 r - L17_4 r
def MS17_4 : ℤ := CaseSplit.mxr (aS17_4) 22

def N17_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n17, (if c17_0 ra t && c17_1 rb t then 1 else 0)
def aP17_0 (ra rb : ℕ) : ℤ := -(2) * N17_0 ra rb + u17 (0 + rb) + u17 (13 + ra)
def MP17_0 : ℤ := CaseSplit.mxr2 (aP17_0) 10 12
def N17_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n17, (if c17_0 ra t && c17_2 rb t then 1 else 0)
def aP17_1 (ra rb : ℕ) : ℤ := -(2) * N17_1 ra rb + u17 (24 + rb) + u17 (41 + ra)
def MP17_1 : ℤ := CaseSplit.mxr2 (aP17_1) 10 16
def N17_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n17, (if c17_0 ra t && c17_3 rb t then 1 else 0)
def aP17_2 (ra rb : ℕ) : ℤ := -(2) * N17_2 ra rb + u17 (52 + rb) + u17 (71 + ra)
def MP17_2 : ℤ := CaseSplit.mxr2 (aP17_2) 10 18
def N17_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n17, (if c17_0 ra t && c17_4 rb t then 1 else 0)
def aP17_3 (ra rb : ℕ) : ℤ := -(2) * N17_3 ra rb + u17 (82 + rb) + u17 (105 + ra)
def MP17_3 : ℤ := CaseSplit.mxr2 (aP17_3) 10 22
def P17_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n17, (if c17_1 ra t && c17_2 rb t then 1 else 0)
def C17_4 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n17, (if c17_1 ra t && c17_2 rb t && c17_0 s t then 1 else 0)
def M17_4 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C17_4 ra rb) 10
def E17_4 : List ℕ := [3, 9, 57, 63, 93, 99, 104, 115, 172, 183, 188, 194]
def N17_4 (ra rb : ℕ) : ℤ := if E17_4.contains (ra * 17 + rb) = true then P17_4 ra rb - M17_4 ra rb else 0
def aP17_4 (ra rb : ℕ) : ℤ := -(2) * N17_4 ra rb + u17 (116 + rb) + u17 (133 + ra)
def MP17_4 : ℤ := CaseSplit.mxr2 (aP17_4) 12 16
def P17_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n17, (if c17_1 ra t && c17_3 rb t then 1 else 0)
def C17_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n17, (if c17_1 ra t && c17_3 rb t && c17_0 s t then 1 else 0)
def M17_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C17_5 ra rb) 10
def E17_5 : List ℕ := [7, 37, 41, 71, 113, 147, 178, 212]
def N17_5 (ra rb : ℕ) : ℤ := if E17_5.contains (ra * 19 + rb) = true then P17_5 ra rb - M17_5 ra rb else 0
def aP17_5 (ra rb : ℕ) : ℤ := -(2) * N17_5 ra rb + u17 (146 + rb) + u17 (165 + ra)
def MP17_5 : ℤ := CaseSplit.mxr2 (aP17_5) 12 18
def P17_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n17, (if c17_1 ra t && c17_4 rb t then 1 else 0)
def C17_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n17, (if c17_1 ra t && c17_4 rb t && c17_0 s t then 1 else 0)
def M17_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C17_6 ra rb) 10
def E17_6 : List ℕ := []
def N17_6 (ra rb : ℕ) : ℤ := if E17_6.contains (ra * 23 + rb) = true then P17_6 ra rb - M17_6 ra rb else 0
def aP17_6 (ra rb : ℕ) : ℤ := -(2) * N17_6 ra rb + u17 (178 + rb) + u17 (201 + ra)
def MP17_6 : ℤ := CaseSplit.mxr2 (aP17_6) 12 22
def N17_7 (_ra _rb : ℕ) : ℤ := 0
def aP17_7 (ra rb : ℕ) : ℤ := -(2) * N17_7 ra rb + u17 (214 + rb) + u17 (233 + ra)
def MP17_7 : ℤ := CaseSplit.mxr2 (aP17_7) 16 18
def N17_8 (_ra _rb : ℕ) : ℤ := 0
def aP17_8 (ra rb : ℕ) : ℤ := -(2) * N17_8 ra rb + u17 (250 + rb) + u17 (273 + ra)
def MP17_8 : ℤ := CaseSplit.mxr2 (aP17_8) 16 22
def N17_9 (_ra _rb : ℕ) : ℤ := 0
def aP17_9 (ra rb : ℕ) : ℤ := -(2) * N17_9 ra rb + u17 (290 + rb) + u17 (313 + ra)
def MP17_9 : ℤ := CaseSplit.mxr2 (aP17_9) 18 22

def rhs17 : ℤ := (∑ t ∈ Finset.range n17, w17 t) + 2 * (n17 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn17 : ∀ t, t < n17 → (0 : ℤ) ≤ w17 t := by decide
theorem plt17 : ∀ t, t < n17 → q17 t < 39 := by decide
theorem pfree17_5 : ∀ t, t < n17 → gb5 2 (q17 t) = false := by decide
theorem pfree17_7 : ∀ t, t < n17 → gb7 3 (q17 t) = false := by decide
theorem MSv17_0 : MS17_0 = 7 := by decide +kernel
theorem MSv17_1 : MS17_1 = 15 := by decide +kernel
theorem MSv17_2 : MS17_2 = 0 := by decide +kernel
theorem MSv17_3 : MS17_3 = 0 := by decide +kernel
theorem MSv17_4 : MS17_4 = 0 := by decide +kernel
theorem MPv17_0 : MP17_0 = 0 := by decide +kernel
theorem MPv17_1 : MP17_1 = 0 := by decide +kernel
theorem MPv17_2 : MP17_2 = 0 := by decide +kernel
theorem MPv17_3 : MP17_3 = 0 := by decide +kernel
theorem MPv17_4 : MP17_4 = 0 := by decide +kernel
theorem MPv17_5 : MP17_5 = 0 := by decide +kernel
theorem MPv17_6 : MP17_6 = 0 := by decide +kernel
theorem MPv17_7 : MP17_7 = 0 := by decide +kernel
theorem MPv17_8 : MP17_8 = 0 := by decide +kernel
theorem MPv17_9 : MP17_9 = 10 := by decide +kernel
theorem rhsv17 : rhs17 = 38 := by decide +kernel

/-- **The case-17 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 6/2.
    (Scaled by the common denominator 2: 32 < 38.) -/
theorem cert17 : MS17_0 + MS17_1 + MS17_2 + MS17_3 + MS17_4 + MP17_0 + MP17_1 + MP17_2 + MP17_3 + MP17_4 + MP17_5 + MP17_6 + MP17_7 + MP17_8 + MP17_9 < rhs17 := by
  rw [MSv17_0, MSv17_1, MSv17_2, MSv17_3, MSv17_4, MPv17_0, MPv17_1, MPv17_2, MPv17_3, MPv17_4, MPv17_5, MPv17_6, MPv17_7, MPv17_8, MPv17_9, rhsv17]
  decide

def Dg17 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := (if c17_0 r0 t then 1 else 0) + (if c17_1 r1 t then 1 else 0) + (if c17_2 r2 t then 1 else 0) + (if c17_3 r3 t then 1 else 0) + (if c17_4 r4 t then 1 else 0)
def Wl17_0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c17_0 r0 t && c17_1 r1 t then 1 else 0
def Wl17_1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c17_0 r0 t && c17_2 r2 t then 1 else 0
def Wl17_2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c17_0 r0 t && c17_3 r3 t then 1 else 0
def Wl17_3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c17_0 r0 t && c17_4 r4 t then 1 else 0
def Wl17_4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c17_0 r0 t && c17_1 r1 t && c17_2 r2 t then 1 else 0
def Wl17_5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c17_0 r0 t && c17_1 r1 t && c17_3 r3 t then 1 else 0
def Wl17_6 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c17_0 r0 t && c17_1 r1 t && c17_4 r4 t then 1 else 0
def Wl17_7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c17_0 r0 t && !c17_1 r1 t && c17_2 r2 t && c17_3 r3 t then 1 else 0
def Wl17_8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c17_0 r0 t && !c17_1 r1 t && c17_2 r2 t && c17_4 r4 t then 1 else 0
def Wl17_9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c17_0 r0 t && !c17_1 r1 t && !c17_2 r2 t && c17_3 r3 t && c17_4 r4 t then 1 else 0

/-- **No configuration blocks the whole window in case 17.** -/
theorem nocov17 {r0 r1 r2 r3 r4 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23)
    (hcov : ∀ t, t < n17 → (c17_0 r0 t || c17_1 r1 t || c17_2 r2 t || c17_3 r3 t || c17_4 r4 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n17, (1 : ℤ) + (Wl17_0 r0 r1 r2 r3 r4 t + Wl17_1 r0 r1 r2 r3 r4 t + Wl17_2 r0 r1 r2 r3 r4 t + Wl17_3 r0 r1 r2 r3 r4 t + Wl17_4 r0 r1 r2 r3 r4 t + Wl17_5 r0 r1 r2 r3 r4 t + Wl17_6 r0 r1 r2 r3 r4 t + Wl17_7 r0 r1 r2 r3 r4 t + Wl17_8 r0 r1 r2 r3 r4 t + Wl17_9 r0 r1 r2 r3 r4 t) ≤ Dg17 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Wl17_0, Wl17_1, Wl17_2, Wl17_3, Wl17_4, Wl17_5, Wl17_6, Wl17_7, Wl17_8, Wl17_9, Dg17]
    exact CaseSplit.lowest5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n17, (1 : ℤ) ≤ Dg17 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Dg17]
    exact CaseSplit.degpos5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n17 : ℤ) + ((∑ t ∈ Finset.range n17, Wl17_0 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n17, Wl17_1 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n17, Wl17_2 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n17, Wl17_3 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n17, Wl17_4 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n17, Wl17_5 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n17, Wl17_6 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n17, Wl17_7 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n17, Wl17_8 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n17, Wl17_9 r0 r1 r2 r3 r4 t)) ≤ ∑ t ∈ Finset.range n17, Dg17 r0 r1 r2 r3 r4 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N17_0 r0 r1 ≤ ∑ t ∈ Finset.range n17, Wl17_0 r0 r1 r2 r3 r4 t := by
    simp only [N17_0, Wl17_0, le_refl]
  have hn1 : N17_1 r0 r2 ≤ ∑ t ∈ Finset.range n17, Wl17_1 r0 r1 r2 r3 r4 t := by
    simp only [N17_1, Wl17_1, le_refl]
  have hn2 : N17_2 r0 r3 ≤ ∑ t ∈ Finset.range n17, Wl17_2 r0 r1 r2 r3 r4 t := by
    simp only [N17_2, Wl17_2, le_refl]
  have hn3 : N17_3 r0 r4 ≤ ∑ t ∈ Finset.range n17, Wl17_3 r0 r1 r2 r3 r4 t := by
    simp only [N17_3, Wl17_3, le_refl]
  have hn4 : N17_4 r1 r2 ≤ ∑ t ∈ Finset.range n17, Wl17_4 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n17, Wl17_4 r0 r1 r2 r3 r4 t
        = (if c17_1 r1 t && c17_2 r2 t then (1:ℤ) else 0)
          - (if c17_1 r1 t && c17_2 r2 t && c17_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl17_4]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n17, Wl17_4 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl17_4]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n17, Wl17_4 r0 r1 r2 r3 r4 t
        = P17_4 r1 r2 - C17_4 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P17_4, C17_4]
    have hm : C17_4 r1 r2 r0 ≤ M17_4 r1 r2 :=
      CaseSplit.le_mxr (C17_4 r1 r2) 10 r0 (by omega)
    simp only [N17_4]
    split
    · rw [hL]; omega
    · exact hnn
  have hn5 : N17_5 r1 r3 ≤ ∑ t ∈ Finset.range n17, Wl17_5 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n17, Wl17_5 r0 r1 r2 r3 r4 t
        = (if c17_1 r1 t && c17_3 r3 t then (1:ℤ) else 0)
          - (if c17_1 r1 t && c17_3 r3 t && c17_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl17_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n17, Wl17_5 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl17_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n17, Wl17_5 r0 r1 r2 r3 r4 t
        = P17_5 r1 r3 - C17_5 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P17_5, C17_5]
    have hm : C17_5 r1 r3 r0 ≤ M17_5 r1 r3 :=
      CaseSplit.le_mxr (C17_5 r1 r3) 10 r0 (by omega)
    simp only [N17_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N17_6 r1 r4 ≤ ∑ t ∈ Finset.range n17, Wl17_6 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n17, Wl17_6 r0 r1 r2 r3 r4 t
        = (if c17_1 r1 t && c17_4 r4 t then (1:ℤ) else 0)
          - (if c17_1 r1 t && c17_4 r4 t && c17_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl17_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n17, Wl17_6 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl17_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n17, Wl17_6 r0 r1 r2 r3 r4 t
        = P17_6 r1 r4 - C17_6 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P17_6, C17_6]
    have hm : C17_6 r1 r4 r0 ≤ M17_6 r1 r4 :=
      CaseSplit.le_mxr (C17_6 r1 r4) 10 r0 (by omega)
    simp only [N17_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N17_7 r2 r3 ≤ ∑ t ∈ Finset.range n17, Wl17_7 r0 r1 r2 r3 r4 t := by
    simp only [N17_7]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl17_7]
    exact CaseSplit.ind_nonneg _
  have hn8 : N17_8 r2 r4 ≤ ∑ t ∈ Finset.range n17, Wl17_8 r0 r1 r2 r3 r4 t := by
    simp only [N17_8]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl17_8]
    exact CaseSplit.ind_nonneg _
  have hn9 : N17_9 r3 r4 ≤ ∑ t ∈ Finset.range n17, Wl17_9 r0 r1 r2 r3 r4 t := by
    simp only [N17_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl17_9]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n17, (w17 t + 2) * Dg17 r0 r1 r2 r3 r4 t = S17_0 r0 + S17_1 r1 + S17_2 r2 + S17_3 r3 + S17_4 r4 := by
    simp only [S17_0, S17_1, S17_2, S17_3, S17_4, Dg17, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n17, (w17 t + 2) * Dg17 r0 r1 r2 r3 r4 t
      = (∑ t ∈ Finset.range n17, w17 t * Dg17 r0 r1 r2 r3 r4 t)
        + 2 * (∑ t ∈ Finset.range n17, Dg17 r0 r1 r2 r3 r4 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n17, w17 t)
      ≤ ∑ t ∈ Finset.range n17, w17 t * Dg17 r0 r1 r2 r3 r4 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg17 r0 r1 r2 r3 r4 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w17 t := wnn17 t (Finset.mem_range.mp ht)
    calc w17 t = w17 t * 1 := (mul_one _).symm
      _ ≤ w17 t * Dg17 r0 r1 r2 r3 r4 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS17_0 r0 + aS17_1 r1 + aS17_2 r2 + aS17_3 r3 + aS17_4 r4) + (aP17_0 r0 r1 + aP17_1 r0 r2 + aP17_2 r0 r3 + aP17_3 r0 r4 + aP17_4 r1 r2 + aP17_5 r1 r3 + aP17_6 r1 r4 + aP17_7 r2 r3 + aP17_8 r2 r4 + aP17_9 r3 r4) = (S17_0 r0 + S17_1 r1 + S17_2 r2 + S17_3 r3 + S17_4 r4) - 2 * (N17_0 r0 r1 + N17_1 r0 r2 + N17_2 r0 r3 + N17_3 r0 r4 + N17_4 r1 r2 + N17_5 r1 r3 + N17_6 r1 r4 + N17_7 r2 r3 + N17_8 r2 r4 + N17_9 r3 r4) := by
    simp only [aS17_0, aS17_1, aS17_2, aS17_3, aS17_4, aP17_0, aP17_1, aP17_2, aP17_3, aP17_4, aP17_5, aP17_6, aP17_7, aP17_8, aP17_9, L17_0, L17_1, L17_2, L17_3, L17_4]
    ring
  have bS0 : aS17_0 r0 ≤ MS17_0 := CaseSplit.le_mxr (aS17_0) 10 r0 (by omega)
  have bS1 : aS17_1 r1 ≤ MS17_1 := CaseSplit.le_mxr (aS17_1) 12 r1 (by omega)
  have bS2 : aS17_2 r2 ≤ MS17_2 := CaseSplit.le_mxr (aS17_2) 16 r2 (by omega)
  have bS3 : aS17_3 r3 ≤ MS17_3 := CaseSplit.le_mxr (aS17_3) 18 r3 (by omega)
  have bS4 : aS17_4 r4 ≤ MS17_4 := CaseSplit.le_mxr (aS17_4) 22 r4 (by omega)
  have bP0 : aP17_0 r0 r1 ≤ MP17_0 := CaseSplit.le_mxr2 (aP17_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP17_1 r0 r2 ≤ MP17_1 := CaseSplit.le_mxr2 (aP17_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP17_2 r0 r3 ≤ MP17_2 := CaseSplit.le_mxr2 (aP17_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP17_3 r0 r4 ≤ MP17_3 := CaseSplit.le_mxr2 (aP17_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP17_4 r1 r2 ≤ MP17_4 := CaseSplit.le_mxr2 (aP17_4) 12 16 r1 r2 (by omega) (by omega)
  have bP5 : aP17_5 r1 r3 ≤ MP17_5 := CaseSplit.le_mxr2 (aP17_5) 12 18 r1 r3 (by omega) (by omega)
  have bP6 : aP17_6 r1 r4 ≤ MP17_6 := CaseSplit.le_mxr2 (aP17_6) 12 22 r1 r4 (by omega) (by omega)
  have bP7 : aP17_7 r2 r3 ≤ MP17_7 := CaseSplit.le_mxr2 (aP17_7) 16 18 r2 r3 (by omega) (by omega)
  have bP8 : aP17_8 r2 r4 ≤ MP17_8 := CaseSplit.le_mxr2 (aP17_8) 16 22 r2 r4 (by omega) (by omega)
  have bP9 : aP17_9 r3 r4 ≤ MP17_9 := CaseSplit.le_mxr2 (aP17_9) 18 22 r3 r4 (by omega) (by omega)
  have hrhs : rhs17 = (∑ t ∈ Finset.range n17, w17 t) + 2 * (n17 : ℤ) := rfl
  have hc := cert17
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, bS0, bS1, bS2, bS3, bS4, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9]

end IncCert23
