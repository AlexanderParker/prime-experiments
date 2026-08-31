/-
INCREMENT-WIDTH CERTIFICATE, step 19->23, case 8 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_19_23.json, which re-derives every number
from the primes alone).

Machine 23, INCREMENT width 39 = F_2(19) + s_min(23) = 31 + 8,
held gears [5, 7] at phases [1, 1].  Free gears [11, 13, 17, 19, 23].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 1.
-/
import IncCert23B

namespace IncCert23

/-! ### case 8: held gears at phases [1, 1] -/

def p8 : List ℕ := [1, 2, 4, 6, 9, 11, 16, 17, 22, 24, 27, 29, 31, 32, 34, 36, 37]
def q8 (t : ℕ) : ℕ := p8.getD t 0
def n8 : ℕ := 17
def yl8 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
def w8 (t : ℕ) : ℤ := yl8.getD t 0
def ul8 : List ℤ := [0, (-1), 0, (-1), 0, (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), (-1), (-1), 0, 0, 0, (-1), (-1), (-2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, (-1), 0, 0, (-1), (-1), 0, 0, (-1), 0, 2, 2, 3, 3, 2, 3, 1, 1, 3, 3, 3, 3, 2, 2, 3, 3, 2, (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), 3, 3, 3, 3, 2, 3, 3, 3, 2, 2, 3, 3, 2, 3, 3, 2, 3, 2, 3, (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 3, 0, 3, 1, 0, 2, 0, 3, 2, 0, 3, 1, 2, 2, 1, 3, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0]
def u8 (k : ℕ) : ℤ := ul8.getD k 0

def c8_0 (r t : ℕ) : Bool := gb11 r (q8 t)
def c8_1 (r t : ℕ) : Bool := gb13 r (q8 t)
def c8_2 (r t : ℕ) : Bool := gb17 r (q8 t)
def c8_3 (r t : ℕ) : Bool := gb19 r (q8 t)
def c8_4 (r t : ℕ) : Bool := gb23 r (q8 t)

def S8_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n8, (w8 t + 1) * (if c8_0 r t then 1 else 0)
def S8_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n8, (w8 t + 1) * (if c8_1 r t then 1 else 0)
def S8_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n8, (w8 t + 1) * (if c8_2 r t then 1 else 0)
def S8_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n8, (w8 t + 1) * (if c8_3 r t then 1 else 0)
def S8_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n8, (w8 t + 1) * (if c8_4 r t then 1 else 0)

def L8_0 (r : ℕ) : ℤ := u8 (13 + r) + u8 (41 + r) + u8 (71 + r) + u8 (105 + r)
def L8_1 (r : ℕ) : ℤ := u8 (0 + r) + u8 (133 + r) + u8 (165 + r) + u8 (201 + r)
def L8_2 (r : ℕ) : ℤ := u8 (24 + r) + u8 (116 + r) + u8 (233 + r) + u8 (273 + r)
def L8_3 (r : ℕ) : ℤ := u8 (52 + r) + u8 (146 + r) + u8 (214 + r) + u8 (313 + r)
def L8_4 (r : ℕ) : ℤ := u8 (82 + r) + u8 (178 + r) + u8 (250 + r) + u8 (290 + r)

def aS8_0 (r : ℕ) : ℤ := S8_0 r - L8_0 r
def MS8_0 : ℤ := CaseSplit.mxr (aS8_0) 10
def aS8_1 (r : ℕ) : ℤ := S8_1 r - L8_1 r
def MS8_1 : ℤ := CaseSplit.mxr (aS8_1) 12
def aS8_2 (r : ℕ) : ℤ := S8_2 r - L8_2 r
def MS8_2 : ℤ := CaseSplit.mxr (aS8_2) 16
def aS8_3 (r : ℕ) : ℤ := S8_3 r - L8_3 r
def MS8_3 : ℤ := CaseSplit.mxr (aS8_3) 18
def aS8_4 (r : ℕ) : ℤ := S8_4 r - L8_4 r
def MS8_4 : ℤ := CaseSplit.mxr (aS8_4) 22

def N8_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n8, (if c8_0 ra t && c8_1 rb t then 1 else 0)
def aP8_0 (ra rb : ℕ) : ℤ := -(1) * N8_0 ra rb + u8 (0 + rb) + u8 (13 + ra)
def MP8_0 : ℤ := CaseSplit.mxr2 (aP8_0) 10 12
def N8_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n8, (if c8_0 ra t && c8_2 rb t then 1 else 0)
def aP8_1 (ra rb : ℕ) : ℤ := -(1) * N8_1 ra rb + u8 (24 + rb) + u8 (41 + ra)
def MP8_1 : ℤ := CaseSplit.mxr2 (aP8_1) 10 16
def N8_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n8, (if c8_0 ra t && c8_3 rb t then 1 else 0)
def aP8_2 (ra rb : ℕ) : ℤ := -(1) * N8_2 ra rb + u8 (52 + rb) + u8 (71 + ra)
def MP8_2 : ℤ := CaseSplit.mxr2 (aP8_2) 10 18
def N8_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n8, (if c8_0 ra t && c8_4 rb t then 1 else 0)
def aP8_3 (ra rb : ℕ) : ℤ := -(1) * N8_3 ra rb + u8 (82 + rb) + u8 (105 + ra)
def MP8_3 : ℤ := CaseSplit.mxr2 (aP8_3) 10 22
def P8_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n8, (if c8_1 ra t && c8_2 rb t then 1 else 0)
def C8_4 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n8, (if c8_1 ra t && c8_2 rb t && c8_0 s t then 1 else 0)
def M8_4 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C8_4 ra rb) 10
def E8_4 : List ℕ := [122, 133]
def N8_4 (ra rb : ℕ) : ℤ := if E8_4.contains (ra * 17 + rb) = true then P8_4 ra rb - M8_4 ra rb else 0
def aP8_4 (ra rb : ℕ) : ℤ := -(1) * N8_4 ra rb + u8 (116 + rb) + u8 (133 + ra)
def MP8_4 : ℤ := CaseSplit.mxr2 (aP8_4) 12 16
def P8_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n8, (if c8_1 ra t && c8_3 rb t then 1 else 0)
def C8_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n8, (if c8_1 ra t && c8_3 rb t && c8_0 s t then 1 else 0)
def M8_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C8_5 ra rb) 10
def E8_5 : List ℕ := [11, 17, 51, 87, 93, 127, 151, 158, 227, 234]
def N8_5 (ra rb : ℕ) : ℤ := if E8_5.contains (ra * 19 + rb) = true then P8_5 ra rb - M8_5 ra rb else 0
def aP8_5 (ra rb : ℕ) : ℤ := -(1) * N8_5 ra rb + u8 (146 + rb) + u8 (165 + ra)
def MP8_5 : ℤ := CaseSplit.mxr2 (aP8_5) 12 18
def P8_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n8, (if c8_1 ra t && c8_4 rb t then 1 else 0)
def C8_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n8, (if c8_1 ra t && c8_4 rb t && c8_0 s t then 1 else 0)
def M8_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C8_6 ra rb) 10
def E8_6 : List ℕ := []
def N8_6 (ra rb : ℕ) : ℤ := if E8_6.contains (ra * 23 + rb) = true then P8_6 ra rb - M8_6 ra rb else 0
def aP8_6 (ra rb : ℕ) : ℤ := -(1) * N8_6 ra rb + u8 (178 + rb) + u8 (201 + ra)
def MP8_6 : ℤ := CaseSplit.mxr2 (aP8_6) 12 22
def N8_7 (_ra _rb : ℕ) : ℤ := 0
def aP8_7 (ra rb : ℕ) : ℤ := -(1) * N8_7 ra rb + u8 (214 + rb) + u8 (233 + ra)
def MP8_7 : ℤ := CaseSplit.mxr2 (aP8_7) 16 18
def N8_8 (_ra _rb : ℕ) : ℤ := 0
def aP8_8 (ra rb : ℕ) : ℤ := -(1) * N8_8 ra rb + u8 (250 + rb) + u8 (273 + ra)
def MP8_8 : ℤ := CaseSplit.mxr2 (aP8_8) 16 22
def N8_9 (_ra _rb : ℕ) : ℤ := 0
def aP8_9 (ra rb : ℕ) : ℤ := -(1) * N8_9 ra rb + u8 (290 + rb) + u8 (313 + ra)
def MP8_9 : ℤ := CaseSplit.mxr2 (aP8_9) 18 22

def rhs8 : ℤ := (∑ t ∈ Finset.range n8, w8 t) + 1 * (n8 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn8 : ∀ t, t < n8 → (0 : ℤ) ≤ w8 t := by decide
theorem plt8 : ∀ t, t < n8 → q8 t < 39 := by decide
theorem pfree8_5 : ∀ t, t < n8 → gb5 1 (q8 t) = false := by decide
theorem pfree8_7 : ∀ t, t < n8 → gb7 1 (q8 t) = false := by decide
theorem MSv8_0 : MS8_0 = 4 := by decide +kernel
theorem MSv8_1 : MS8_1 = 10 := by decide +kernel
theorem MSv8_2 : MS8_2 = 0 := by decide +kernel
theorem MSv8_3 : MS8_3 = 0 := by decide +kernel
theorem MSv8_4 : MS8_4 = 0 := by decide +kernel
theorem MPv8_0 : MP8_0 = 0 := by decide +kernel
theorem MPv8_1 : MP8_1 = 0 := by decide +kernel
theorem MPv8_2 : MP8_2 = 0 := by decide +kernel
theorem MPv8_3 : MP8_3 = 0 := by decide +kernel
theorem MPv8_4 : MP8_4 = 0 := by decide +kernel
theorem MPv8_5 : MP8_5 = 0 := by decide +kernel
theorem MPv8_6 : MP8_6 = 0 := by decide +kernel
theorem MPv8_7 : MP8_7 = 0 := by decide +kernel
theorem MPv8_8 : MP8_8 = 0 := by decide +kernel
theorem MPv8_9 : MP8_9 = 3 := by decide +kernel
theorem rhsv8 : rhs8 = 18 := by decide +kernel

/-- **The case-8 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/1.
    (Scaled by the common denominator 1: 17 < 18.) -/
theorem cert8 : MS8_0 + MS8_1 + MS8_2 + MS8_3 + MS8_4 + MP8_0 + MP8_1 + MP8_2 + MP8_3 + MP8_4 + MP8_5 + MP8_6 + MP8_7 + MP8_8 + MP8_9 < rhs8 := by
  rw [MSv8_0, MSv8_1, MSv8_2, MSv8_3, MSv8_4, MPv8_0, MPv8_1, MPv8_2, MPv8_3, MPv8_4, MPv8_5, MPv8_6, MPv8_7, MPv8_8, MPv8_9, rhsv8]
  decide

def Dg8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := (if c8_0 r0 t then 1 else 0) + (if c8_1 r1 t then 1 else 0) + (if c8_2 r2 t then 1 else 0) + (if c8_3 r3 t then 1 else 0) + (if c8_4 r4 t then 1 else 0)
def Wl8_0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c8_0 r0 t && c8_1 r1 t then 1 else 0
def Wl8_1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c8_0 r0 t && c8_2 r2 t then 1 else 0
def Wl8_2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c8_0 r0 t && c8_3 r3 t then 1 else 0
def Wl8_3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c8_0 r0 t && c8_4 r4 t then 1 else 0
def Wl8_4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c8_0 r0 t && c8_1 r1 t && c8_2 r2 t then 1 else 0
def Wl8_5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c8_0 r0 t && c8_1 r1 t && c8_3 r3 t then 1 else 0
def Wl8_6 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c8_0 r0 t && c8_1 r1 t && c8_4 r4 t then 1 else 0
def Wl8_7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c8_0 r0 t && !c8_1 r1 t && c8_2 r2 t && c8_3 r3 t then 1 else 0
def Wl8_8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c8_0 r0 t && !c8_1 r1 t && c8_2 r2 t && c8_4 r4 t then 1 else 0
def Wl8_9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c8_0 r0 t && !c8_1 r1 t && !c8_2 r2 t && c8_3 r3 t && c8_4 r4 t then 1 else 0

/-- **No configuration blocks the whole window in case 8.** -/
theorem nocov8 {r0 r1 r2 r3 r4 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23)
    (hcov : ∀ t, t < n8 → (c8_0 r0 t || c8_1 r1 t || c8_2 r2 t || c8_3 r3 t || c8_4 r4 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n8, (1 : ℤ) + (Wl8_0 r0 r1 r2 r3 r4 t + Wl8_1 r0 r1 r2 r3 r4 t + Wl8_2 r0 r1 r2 r3 r4 t + Wl8_3 r0 r1 r2 r3 r4 t + Wl8_4 r0 r1 r2 r3 r4 t + Wl8_5 r0 r1 r2 r3 r4 t + Wl8_6 r0 r1 r2 r3 r4 t + Wl8_7 r0 r1 r2 r3 r4 t + Wl8_8 r0 r1 r2 r3 r4 t + Wl8_9 r0 r1 r2 r3 r4 t) ≤ Dg8 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Wl8_0, Wl8_1, Wl8_2, Wl8_3, Wl8_4, Wl8_5, Wl8_6, Wl8_7, Wl8_8, Wl8_9, Dg8]
    exact CaseSplit.lowest5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n8, (1 : ℤ) ≤ Dg8 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Dg8]
    exact CaseSplit.degpos5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n8 : ℤ) + ((∑ t ∈ Finset.range n8, Wl8_0 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n8, Wl8_1 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n8, Wl8_2 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n8, Wl8_3 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n8, Wl8_4 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n8, Wl8_5 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n8, Wl8_6 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n8, Wl8_7 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n8, Wl8_8 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n8, Wl8_9 r0 r1 r2 r3 r4 t)) ≤ ∑ t ∈ Finset.range n8, Dg8 r0 r1 r2 r3 r4 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N8_0 r0 r1 ≤ ∑ t ∈ Finset.range n8, Wl8_0 r0 r1 r2 r3 r4 t := by
    simp only [N8_0, Wl8_0, le_refl]
  have hn1 : N8_1 r0 r2 ≤ ∑ t ∈ Finset.range n8, Wl8_1 r0 r1 r2 r3 r4 t := by
    simp only [N8_1, Wl8_1, le_refl]
  have hn2 : N8_2 r0 r3 ≤ ∑ t ∈ Finset.range n8, Wl8_2 r0 r1 r2 r3 r4 t := by
    simp only [N8_2, Wl8_2, le_refl]
  have hn3 : N8_3 r0 r4 ≤ ∑ t ∈ Finset.range n8, Wl8_3 r0 r1 r2 r3 r4 t := by
    simp only [N8_3, Wl8_3, le_refl]
  have hn4 : N8_4 r1 r2 ≤ ∑ t ∈ Finset.range n8, Wl8_4 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n8, Wl8_4 r0 r1 r2 r3 r4 t
        = (if c8_1 r1 t && c8_2 r2 t then (1:ℤ) else 0)
          - (if c8_1 r1 t && c8_2 r2 t && c8_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl8_4]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n8, Wl8_4 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl8_4]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n8, Wl8_4 r0 r1 r2 r3 r4 t
        = P8_4 r1 r2 - C8_4 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P8_4, C8_4]
    have hm : C8_4 r1 r2 r0 ≤ M8_4 r1 r2 :=
      CaseSplit.le_mxr (C8_4 r1 r2) 10 r0 (by omega)
    simp only [N8_4]
    split
    · rw [hL]; omega
    · exact hnn
  have hn5 : N8_5 r1 r3 ≤ ∑ t ∈ Finset.range n8, Wl8_5 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n8, Wl8_5 r0 r1 r2 r3 r4 t
        = (if c8_1 r1 t && c8_3 r3 t then (1:ℤ) else 0)
          - (if c8_1 r1 t && c8_3 r3 t && c8_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl8_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n8, Wl8_5 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl8_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n8, Wl8_5 r0 r1 r2 r3 r4 t
        = P8_5 r1 r3 - C8_5 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P8_5, C8_5]
    have hm : C8_5 r1 r3 r0 ≤ M8_5 r1 r3 :=
      CaseSplit.le_mxr (C8_5 r1 r3) 10 r0 (by omega)
    simp only [N8_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N8_6 r1 r4 ≤ ∑ t ∈ Finset.range n8, Wl8_6 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n8, Wl8_6 r0 r1 r2 r3 r4 t
        = (if c8_1 r1 t && c8_4 r4 t then (1:ℤ) else 0)
          - (if c8_1 r1 t && c8_4 r4 t && c8_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl8_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n8, Wl8_6 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl8_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n8, Wl8_6 r0 r1 r2 r3 r4 t
        = P8_6 r1 r4 - C8_6 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P8_6, C8_6]
    have hm : C8_6 r1 r4 r0 ≤ M8_6 r1 r4 :=
      CaseSplit.le_mxr (C8_6 r1 r4) 10 r0 (by omega)
    simp only [N8_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N8_7 r2 r3 ≤ ∑ t ∈ Finset.range n8, Wl8_7 r0 r1 r2 r3 r4 t := by
    simp only [N8_7]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl8_7]
    exact CaseSplit.ind_nonneg _
  have hn8 : N8_8 r2 r4 ≤ ∑ t ∈ Finset.range n8, Wl8_8 r0 r1 r2 r3 r4 t := by
    simp only [N8_8]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl8_8]
    exact CaseSplit.ind_nonneg _
  have hn9 : N8_9 r3 r4 ≤ ∑ t ∈ Finset.range n8, Wl8_9 r0 r1 r2 r3 r4 t := by
    simp only [N8_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl8_9]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n8, (w8 t + 1) * Dg8 r0 r1 r2 r3 r4 t = S8_0 r0 + S8_1 r1 + S8_2 r2 + S8_3 r3 + S8_4 r4 := by
    simp only [S8_0, S8_1, S8_2, S8_3, S8_4, Dg8, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n8, (w8 t + 1) * Dg8 r0 r1 r2 r3 r4 t
      = (∑ t ∈ Finset.range n8, w8 t * Dg8 r0 r1 r2 r3 r4 t)
        + 1 * (∑ t ∈ Finset.range n8, Dg8 r0 r1 r2 r3 r4 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n8, w8 t)
      ≤ ∑ t ∈ Finset.range n8, w8 t * Dg8 r0 r1 r2 r3 r4 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg8 r0 r1 r2 r3 r4 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w8 t := wnn8 t (Finset.mem_range.mp ht)
    calc w8 t = w8 t * 1 := (mul_one _).symm
      _ ≤ w8 t * Dg8 r0 r1 r2 r3 r4 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS8_0 r0 + aS8_1 r1 + aS8_2 r2 + aS8_3 r3 + aS8_4 r4) + (aP8_0 r0 r1 + aP8_1 r0 r2 + aP8_2 r0 r3 + aP8_3 r0 r4 + aP8_4 r1 r2 + aP8_5 r1 r3 + aP8_6 r1 r4 + aP8_7 r2 r3 + aP8_8 r2 r4 + aP8_9 r3 r4) = (S8_0 r0 + S8_1 r1 + S8_2 r2 + S8_3 r3 + S8_4 r4) - 1 * (N8_0 r0 r1 + N8_1 r0 r2 + N8_2 r0 r3 + N8_3 r0 r4 + N8_4 r1 r2 + N8_5 r1 r3 + N8_6 r1 r4 + N8_7 r2 r3 + N8_8 r2 r4 + N8_9 r3 r4) := by
    simp only [aS8_0, aS8_1, aS8_2, aS8_3, aS8_4, aP8_0, aP8_1, aP8_2, aP8_3, aP8_4, aP8_5, aP8_6, aP8_7, aP8_8, aP8_9, L8_0, L8_1, L8_2, L8_3, L8_4]
    ring
  have bS0 : aS8_0 r0 ≤ MS8_0 := CaseSplit.le_mxr (aS8_0) 10 r0 (by omega)
  have bS1 : aS8_1 r1 ≤ MS8_1 := CaseSplit.le_mxr (aS8_1) 12 r1 (by omega)
  have bS2 : aS8_2 r2 ≤ MS8_2 := CaseSplit.le_mxr (aS8_2) 16 r2 (by omega)
  have bS3 : aS8_3 r3 ≤ MS8_3 := CaseSplit.le_mxr (aS8_3) 18 r3 (by omega)
  have bS4 : aS8_4 r4 ≤ MS8_4 := CaseSplit.le_mxr (aS8_4) 22 r4 (by omega)
  have bP0 : aP8_0 r0 r1 ≤ MP8_0 := CaseSplit.le_mxr2 (aP8_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP8_1 r0 r2 ≤ MP8_1 := CaseSplit.le_mxr2 (aP8_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP8_2 r0 r3 ≤ MP8_2 := CaseSplit.le_mxr2 (aP8_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP8_3 r0 r4 ≤ MP8_3 := CaseSplit.le_mxr2 (aP8_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP8_4 r1 r2 ≤ MP8_4 := CaseSplit.le_mxr2 (aP8_4) 12 16 r1 r2 (by omega) (by omega)
  have bP5 : aP8_5 r1 r3 ≤ MP8_5 := CaseSplit.le_mxr2 (aP8_5) 12 18 r1 r3 (by omega) (by omega)
  have bP6 : aP8_6 r1 r4 ≤ MP8_6 := CaseSplit.le_mxr2 (aP8_6) 12 22 r1 r4 (by omega) (by omega)
  have bP7 : aP8_7 r2 r3 ≤ MP8_7 := CaseSplit.le_mxr2 (aP8_7) 16 18 r2 r3 (by omega) (by omega)
  have bP8 : aP8_8 r2 r4 ≤ MP8_8 := CaseSplit.le_mxr2 (aP8_8) 16 22 r2 r4 (by omega) (by omega)
  have bP9 : aP8_9 r3 r4 ≤ MP8_9 := CaseSplit.le_mxr2 (aP8_9) 18 22 r3 r4 (by omega) (by omega)
  have hrhs : rhs8 = (∑ t ∈ Finset.range n8, w8 t) + 1 * (n8 : ℤ) := rfl
  have hc := cert8
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, bS0, bS1, bS2, bS3, bS4, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9]

end IncCert23
