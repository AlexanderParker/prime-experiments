/-
INCREMENT-WIDTH CERTIFICATE, step 19->23, case 32 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_19_23.json, which re-derives every number
from the primes alone).

Machine 23, INCREMENT width 39 = F_2(19) + s_min(23) = 31 + 8,
held gears [5, 7] at phases [4, 4].  Free gears [11, 13, 17, 19, 23].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 1.
-/
import IncCert23B

namespace IncCert23

/-! ### case 32: held gears at phases [4, 4] -/

def p32 : List ℕ := [1, 3, 6, 8, 13, 14, 19, 21, 24, 26, 28, 29, 31, 33, 34, 36, 38]
def q32 (t : ℕ) : ℕ := p32.getD t 0
def n32 : ℕ := 17
def yl32 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
def w32 (t : ℕ) : ℤ := yl32.getD t 0
def ul32 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), (-1), (-1), 0, (-1), 0, (-1), (-1), 0, 0, (-1), (-1), (-1), (-1), 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, (-1), 0, 0, 0, (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), (-1), (-1), 0, 0, 0, (-1), (-1), 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, (-1), 0, 0, 1, 0, (-1), (-1), (-1), 0, (-1), (-1), 0, 0, (-1), (-1), 0, 3, 3, 3, 3, 2, 2, 3, 3, 3, 2, 3, 2, 3, 3, 3, 2, 3, (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 2, 1, 1, 3, 0, 3, 1, 0, 3, 1, 3, 2, 0, 3, 3, 3, 2, 0, 3, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0]
def u32 (k : ℕ) : ℤ := ul32.getD k 0

def c32_0 (r t : ℕ) : Bool := gb11 r (q32 t)
def c32_1 (r t : ℕ) : Bool := gb13 r (q32 t)
def c32_2 (r t : ℕ) : Bool := gb17 r (q32 t)
def c32_3 (r t : ℕ) : Bool := gb19 r (q32 t)
def c32_4 (r t : ℕ) : Bool := gb23 r (q32 t)

def S32_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n32, (w32 t + 1) * (if c32_0 r t then 1 else 0)
def S32_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n32, (w32 t + 1) * (if c32_1 r t then 1 else 0)
def S32_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n32, (w32 t + 1) * (if c32_2 r t then 1 else 0)
def S32_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n32, (w32 t + 1) * (if c32_3 r t then 1 else 0)
def S32_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n32, (w32 t + 1) * (if c32_4 r t then 1 else 0)

def L32_0 (r : ℕ) : ℤ := u32 (13 + r) + u32 (41 + r) + u32 (71 + r) + u32 (105 + r)
def L32_1 (r : ℕ) : ℤ := u32 (0 + r) + u32 (133 + r) + u32 (165 + r) + u32 (201 + r)
def L32_2 (r : ℕ) : ℤ := u32 (24 + r) + u32 (116 + r) + u32 (233 + r) + u32 (273 + r)
def L32_3 (r : ℕ) : ℤ := u32 (52 + r) + u32 (146 + r) + u32 (214 + r) + u32 (313 + r)
def L32_4 (r : ℕ) : ℤ := u32 (82 + r) + u32 (178 + r) + u32 (250 + r) + u32 (290 + r)

def aS32_0 (r : ℕ) : ℤ := S32_0 r - L32_0 r
def MS32_0 : ℤ := CaseSplit.mxr (aS32_0) 10
def aS32_1 (r : ℕ) : ℤ := S32_1 r - L32_1 r
def MS32_1 : ℤ := CaseSplit.mxr (aS32_1) 12
def aS32_2 (r : ℕ) : ℤ := S32_2 r - L32_2 r
def MS32_2 : ℤ := CaseSplit.mxr (aS32_2) 16
def aS32_3 (r : ℕ) : ℤ := S32_3 r - L32_3 r
def MS32_3 : ℤ := CaseSplit.mxr (aS32_3) 18
def aS32_4 (r : ℕ) : ℤ := S32_4 r - L32_4 r
def MS32_4 : ℤ := CaseSplit.mxr (aS32_4) 22

def N32_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n32, (if c32_0 ra t && c32_1 rb t then 1 else 0)
def aP32_0 (ra rb : ℕ) : ℤ := -(1) * N32_0 ra rb + u32 (0 + rb) + u32 (13 + ra)
def MP32_0 : ℤ := CaseSplit.mxr2 (aP32_0) 10 12
def N32_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n32, (if c32_0 ra t && c32_2 rb t then 1 else 0)
def aP32_1 (ra rb : ℕ) : ℤ := -(1) * N32_1 ra rb + u32 (24 + rb) + u32 (41 + ra)
def MP32_1 : ℤ := CaseSplit.mxr2 (aP32_1) 10 16
def N32_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n32, (if c32_0 ra t && c32_3 rb t then 1 else 0)
def aP32_2 (ra rb : ℕ) : ℤ := -(1) * N32_2 ra rb + u32 (52 + rb) + u32 (71 + ra)
def MP32_2 : ℤ := CaseSplit.mxr2 (aP32_2) 10 18
def N32_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n32, (if c32_0 ra t && c32_4 rb t then 1 else 0)
def aP32_3 (ra rb : ℕ) : ℤ := -(1) * N32_3 ra rb + u32 (82 + rb) + u32 (105 + ra)
def MP32_3 : ℤ := CaseSplit.mxr2 (aP32_3) 10 22
def P32_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n32, (if c32_1 ra t && c32_2 rb t then 1 else 0)
def C32_4 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n32, (if c32_1 ra t && c32_2 rb t && c32_0 s t then 1 else 0)
def M32_4 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C32_4 ra rb) 10
def E32_4 : List ℕ := [61, 67, 86, 97, 170, 176]
def N32_4 (ra rb : ℕ) : ℤ := if E32_4.contains (ra * 17 + rb) = true then P32_4 ra rb - M32_4 ra rb else 0
def aP32_4 (ra rb : ℕ) : ℤ := -(1) * N32_4 ra rb + u32 (116 + rb) + u32 (133 + ra)
def MP32_4 : ℤ := CaseSplit.mxr2 (aP32_4) 12 16
def P32_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n32, (if c32_1 ra t && c32_3 rb t then 1 else 0)
def C32_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n32, (if c32_1 ra t && c32_3 rb t && c32_0 s t then 1 else 0)
def M32_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C32_5 ra rb) 10
def E32_5 : List ℕ := [21, 47, 58, 71, 111, 134, 147, 187, 192, 218]
def N32_5 (ra rb : ℕ) : ℤ := if E32_5.contains (ra * 19 + rb) = true then P32_5 ra rb - M32_5 ra rb else 0
def aP32_5 (ra rb : ℕ) : ℤ := -(1) * N32_5 ra rb + u32 (146 + rb) + u32 (165 + ra)
def MP32_5 : ℤ := CaseSplit.mxr2 (aP32_5) 12 18
def P32_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n32, (if c32_1 ra t && c32_4 rb t then 1 else 0)
def C32_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n32, (if c32_1 ra t && c32_4 rb t && c32_0 s t then 1 else 0)
def M32_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C32_6 ra rb) 10
def E32_6 : List ℕ := []
def N32_6 (ra rb : ℕ) : ℤ := if E32_6.contains (ra * 23 + rb) = true then P32_6 ra rb - M32_6 ra rb else 0
def aP32_6 (ra rb : ℕ) : ℤ := -(1) * N32_6 ra rb + u32 (178 + rb) + u32 (201 + ra)
def MP32_6 : ℤ := CaseSplit.mxr2 (aP32_6) 12 22
def N32_7 (_ra _rb : ℕ) : ℤ := 0
def aP32_7 (ra rb : ℕ) : ℤ := -(1) * N32_7 ra rb + u32 (214 + rb) + u32 (233 + ra)
def MP32_7 : ℤ := CaseSplit.mxr2 (aP32_7) 16 18
def N32_8 (_ra _rb : ℕ) : ℤ := 0
def aP32_8 (ra rb : ℕ) : ℤ := -(1) * N32_8 ra rb + u32 (250 + rb) + u32 (273 + ra)
def MP32_8 : ℤ := CaseSplit.mxr2 (aP32_8) 16 22
def N32_9 (_ra _rb : ℕ) : ℤ := 0
def aP32_9 (ra rb : ℕ) : ℤ := -(1) * N32_9 ra rb + u32 (290 + rb) + u32 (313 + ra)
def MP32_9 : ℤ := CaseSplit.mxr2 (aP32_9) 18 22

def rhs32 : ℤ := (∑ t ∈ Finset.range n32, w32 t) + 1 * (n32 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn32 : ∀ t, t < n32 → (0 : ℤ) ≤ w32 t := by decide
theorem plt32 : ∀ t, t < n32 → q32 t < 39 := by decide
theorem pfree32_5 : ∀ t, t < n32 → gb5 4 (q32 t) = false := by decide
theorem pfree32_7 : ∀ t, t < n32 → gb7 4 (q32 t) = false := by decide
theorem MSv32_0 : MS32_0 = 4 := by decide +kernel
theorem MSv32_1 : MS32_1 = 9 := by decide +kernel
theorem MSv32_2 : MS32_2 = 0 := by decide +kernel
theorem MSv32_3 : MS32_3 = 0 := by decide +kernel
theorem MSv32_4 : MS32_4 = 0 := by decide +kernel
theorem MPv32_0 : MP32_0 = 0 := by decide +kernel
theorem MPv32_1 : MP32_1 = 0 := by decide +kernel
theorem MPv32_2 : MP32_2 = 0 := by decide +kernel
theorem MPv32_3 : MP32_3 = 0 := by decide +kernel
theorem MPv32_4 : MP32_4 = 0 := by decide +kernel
theorem MPv32_5 : MP32_5 = 0 := by decide +kernel
theorem MPv32_6 : MP32_6 = 0 := by decide +kernel
theorem MPv32_7 : MP32_7 = 0 := by decide +kernel
theorem MPv32_8 : MP32_8 = 0 := by decide +kernel
theorem MPv32_9 : MP32_9 = 4 := by decide +kernel
theorem rhsv32 : rhs32 = 18 := by decide +kernel

/-- **The case-32 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/1.
    (Scaled by the common denominator 1: 17 < 18.) -/
theorem cert32 : MS32_0 + MS32_1 + MS32_2 + MS32_3 + MS32_4 + MP32_0 + MP32_1 + MP32_2 + MP32_3 + MP32_4 + MP32_5 + MP32_6 + MP32_7 + MP32_8 + MP32_9 < rhs32 := by
  rw [MSv32_0, MSv32_1, MSv32_2, MSv32_3, MSv32_4, MPv32_0, MPv32_1, MPv32_2, MPv32_3, MPv32_4, MPv32_5, MPv32_6, MPv32_7, MPv32_8, MPv32_9, rhsv32]
  decide

def Dg32 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := (if c32_0 r0 t then 1 else 0) + (if c32_1 r1 t then 1 else 0) + (if c32_2 r2 t then 1 else 0) + (if c32_3 r3 t then 1 else 0) + (if c32_4 r4 t then 1 else 0)
def Wl32_0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c32_0 r0 t && c32_1 r1 t then 1 else 0
def Wl32_1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c32_0 r0 t && c32_2 r2 t then 1 else 0
def Wl32_2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c32_0 r0 t && c32_3 r3 t then 1 else 0
def Wl32_3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c32_0 r0 t && c32_4 r4 t then 1 else 0
def Wl32_4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c32_0 r0 t && c32_1 r1 t && c32_2 r2 t then 1 else 0
def Wl32_5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c32_0 r0 t && c32_1 r1 t && c32_3 r3 t then 1 else 0
def Wl32_6 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c32_0 r0 t && c32_1 r1 t && c32_4 r4 t then 1 else 0
def Wl32_7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c32_0 r0 t && !c32_1 r1 t && c32_2 r2 t && c32_3 r3 t then 1 else 0
def Wl32_8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c32_0 r0 t && !c32_1 r1 t && c32_2 r2 t && c32_4 r4 t then 1 else 0
def Wl32_9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c32_0 r0 t && !c32_1 r1 t && !c32_2 r2 t && c32_3 r3 t && c32_4 r4 t then 1 else 0

/-- **No configuration blocks the whole window in case 32.** -/
theorem nocov32 {r0 r1 r2 r3 r4 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23)
    (hcov : ∀ t, t < n32 → (c32_0 r0 t || c32_1 r1 t || c32_2 r2 t || c32_3 r3 t || c32_4 r4 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n32, (1 : ℤ) + (Wl32_0 r0 r1 r2 r3 r4 t + Wl32_1 r0 r1 r2 r3 r4 t + Wl32_2 r0 r1 r2 r3 r4 t + Wl32_3 r0 r1 r2 r3 r4 t + Wl32_4 r0 r1 r2 r3 r4 t + Wl32_5 r0 r1 r2 r3 r4 t + Wl32_6 r0 r1 r2 r3 r4 t + Wl32_7 r0 r1 r2 r3 r4 t + Wl32_8 r0 r1 r2 r3 r4 t + Wl32_9 r0 r1 r2 r3 r4 t) ≤ Dg32 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Wl32_0, Wl32_1, Wl32_2, Wl32_3, Wl32_4, Wl32_5, Wl32_6, Wl32_7, Wl32_8, Wl32_9, Dg32]
    exact CaseSplit.lowest5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n32, (1 : ℤ) ≤ Dg32 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Dg32]
    exact CaseSplit.degpos5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n32 : ℤ) + ((∑ t ∈ Finset.range n32, Wl32_0 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n32, Wl32_1 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n32, Wl32_2 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n32, Wl32_3 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n32, Wl32_4 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n32, Wl32_5 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n32, Wl32_6 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n32, Wl32_7 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n32, Wl32_8 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n32, Wl32_9 r0 r1 r2 r3 r4 t)) ≤ ∑ t ∈ Finset.range n32, Dg32 r0 r1 r2 r3 r4 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N32_0 r0 r1 ≤ ∑ t ∈ Finset.range n32, Wl32_0 r0 r1 r2 r3 r4 t := by
    simp only [N32_0, Wl32_0, le_refl]
  have hn1 : N32_1 r0 r2 ≤ ∑ t ∈ Finset.range n32, Wl32_1 r0 r1 r2 r3 r4 t := by
    simp only [N32_1, Wl32_1, le_refl]
  have hn2 : N32_2 r0 r3 ≤ ∑ t ∈ Finset.range n32, Wl32_2 r0 r1 r2 r3 r4 t := by
    simp only [N32_2, Wl32_2, le_refl]
  have hn3 : N32_3 r0 r4 ≤ ∑ t ∈ Finset.range n32, Wl32_3 r0 r1 r2 r3 r4 t := by
    simp only [N32_3, Wl32_3, le_refl]
  have hn4 : N32_4 r1 r2 ≤ ∑ t ∈ Finset.range n32, Wl32_4 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n32, Wl32_4 r0 r1 r2 r3 r4 t
        = (if c32_1 r1 t && c32_2 r2 t then (1:ℤ) else 0)
          - (if c32_1 r1 t && c32_2 r2 t && c32_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl32_4]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n32, Wl32_4 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl32_4]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n32, Wl32_4 r0 r1 r2 r3 r4 t
        = P32_4 r1 r2 - C32_4 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P32_4, C32_4]
    have hm : C32_4 r1 r2 r0 ≤ M32_4 r1 r2 :=
      CaseSplit.le_mxr (C32_4 r1 r2) 10 r0 (by omega)
    simp only [N32_4]
    split
    · rw [hL]; omega
    · exact hnn
  have hn5 : N32_5 r1 r3 ≤ ∑ t ∈ Finset.range n32, Wl32_5 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n32, Wl32_5 r0 r1 r2 r3 r4 t
        = (if c32_1 r1 t && c32_3 r3 t then (1:ℤ) else 0)
          - (if c32_1 r1 t && c32_3 r3 t && c32_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl32_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n32, Wl32_5 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl32_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n32, Wl32_5 r0 r1 r2 r3 r4 t
        = P32_5 r1 r3 - C32_5 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P32_5, C32_5]
    have hm : C32_5 r1 r3 r0 ≤ M32_5 r1 r3 :=
      CaseSplit.le_mxr (C32_5 r1 r3) 10 r0 (by omega)
    simp only [N32_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N32_6 r1 r4 ≤ ∑ t ∈ Finset.range n32, Wl32_6 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n32, Wl32_6 r0 r1 r2 r3 r4 t
        = (if c32_1 r1 t && c32_4 r4 t then (1:ℤ) else 0)
          - (if c32_1 r1 t && c32_4 r4 t && c32_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl32_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n32, Wl32_6 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl32_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n32, Wl32_6 r0 r1 r2 r3 r4 t
        = P32_6 r1 r4 - C32_6 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P32_6, C32_6]
    have hm : C32_6 r1 r4 r0 ≤ M32_6 r1 r4 :=
      CaseSplit.le_mxr (C32_6 r1 r4) 10 r0 (by omega)
    simp only [N32_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N32_7 r2 r3 ≤ ∑ t ∈ Finset.range n32, Wl32_7 r0 r1 r2 r3 r4 t := by
    simp only [N32_7]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl32_7]
    exact CaseSplit.ind_nonneg _
  have hn8 : N32_8 r2 r4 ≤ ∑ t ∈ Finset.range n32, Wl32_8 r0 r1 r2 r3 r4 t := by
    simp only [N32_8]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl32_8]
    exact CaseSplit.ind_nonneg _
  have hn9 : N32_9 r3 r4 ≤ ∑ t ∈ Finset.range n32, Wl32_9 r0 r1 r2 r3 r4 t := by
    simp only [N32_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl32_9]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n32, (w32 t + 1) * Dg32 r0 r1 r2 r3 r4 t = S32_0 r0 + S32_1 r1 + S32_2 r2 + S32_3 r3 + S32_4 r4 := by
    simp only [S32_0, S32_1, S32_2, S32_3, S32_4, Dg32, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n32, (w32 t + 1) * Dg32 r0 r1 r2 r3 r4 t
      = (∑ t ∈ Finset.range n32, w32 t * Dg32 r0 r1 r2 r3 r4 t)
        + 1 * (∑ t ∈ Finset.range n32, Dg32 r0 r1 r2 r3 r4 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n32, w32 t)
      ≤ ∑ t ∈ Finset.range n32, w32 t * Dg32 r0 r1 r2 r3 r4 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg32 r0 r1 r2 r3 r4 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w32 t := wnn32 t (Finset.mem_range.mp ht)
    calc w32 t = w32 t * 1 := (mul_one _).symm
      _ ≤ w32 t * Dg32 r0 r1 r2 r3 r4 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS32_0 r0 + aS32_1 r1 + aS32_2 r2 + aS32_3 r3 + aS32_4 r4) + (aP32_0 r0 r1 + aP32_1 r0 r2 + aP32_2 r0 r3 + aP32_3 r0 r4 + aP32_4 r1 r2 + aP32_5 r1 r3 + aP32_6 r1 r4 + aP32_7 r2 r3 + aP32_8 r2 r4 + aP32_9 r3 r4) = (S32_0 r0 + S32_1 r1 + S32_2 r2 + S32_3 r3 + S32_4 r4) - 1 * (N32_0 r0 r1 + N32_1 r0 r2 + N32_2 r0 r3 + N32_3 r0 r4 + N32_4 r1 r2 + N32_5 r1 r3 + N32_6 r1 r4 + N32_7 r2 r3 + N32_8 r2 r4 + N32_9 r3 r4) := by
    simp only [aS32_0, aS32_1, aS32_2, aS32_3, aS32_4, aP32_0, aP32_1, aP32_2, aP32_3, aP32_4, aP32_5, aP32_6, aP32_7, aP32_8, aP32_9, L32_0, L32_1, L32_2, L32_3, L32_4]
    ring
  have bS0 : aS32_0 r0 ≤ MS32_0 := CaseSplit.le_mxr (aS32_0) 10 r0 (by omega)
  have bS1 : aS32_1 r1 ≤ MS32_1 := CaseSplit.le_mxr (aS32_1) 12 r1 (by omega)
  have bS2 : aS32_2 r2 ≤ MS32_2 := CaseSplit.le_mxr (aS32_2) 16 r2 (by omega)
  have bS3 : aS32_3 r3 ≤ MS32_3 := CaseSplit.le_mxr (aS32_3) 18 r3 (by omega)
  have bS4 : aS32_4 r4 ≤ MS32_4 := CaseSplit.le_mxr (aS32_4) 22 r4 (by omega)
  have bP0 : aP32_0 r0 r1 ≤ MP32_0 := CaseSplit.le_mxr2 (aP32_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP32_1 r0 r2 ≤ MP32_1 := CaseSplit.le_mxr2 (aP32_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP32_2 r0 r3 ≤ MP32_2 := CaseSplit.le_mxr2 (aP32_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP32_3 r0 r4 ≤ MP32_3 := CaseSplit.le_mxr2 (aP32_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP32_4 r1 r2 ≤ MP32_4 := CaseSplit.le_mxr2 (aP32_4) 12 16 r1 r2 (by omega) (by omega)
  have bP5 : aP32_5 r1 r3 ≤ MP32_5 := CaseSplit.le_mxr2 (aP32_5) 12 18 r1 r3 (by omega) (by omega)
  have bP6 : aP32_6 r1 r4 ≤ MP32_6 := CaseSplit.le_mxr2 (aP32_6) 12 22 r1 r4 (by omega) (by omega)
  have bP7 : aP32_7 r2 r3 ≤ MP32_7 := CaseSplit.le_mxr2 (aP32_7) 16 18 r2 r3 (by omega) (by omega)
  have bP8 : aP32_8 r2 r4 ≤ MP32_8 := CaseSplit.le_mxr2 (aP32_8) 16 22 r2 r4 (by omega) (by omega)
  have bP9 : aP32_9 r3 r4 ≤ MP32_9 := CaseSplit.le_mxr2 (aP32_9) 18 22 r3 r4 (by omega) (by omega)
  have hrhs : rhs32 = (∑ t ∈ Finset.range n32, w32 t) + 1 * (n32 : ℤ) := rfl
  have hc := cert32
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, bS0, bS1, bS2, bS3, bS4, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9]

end IncCert23
