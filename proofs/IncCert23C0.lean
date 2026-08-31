/-
INCREMENT-WIDTH CERTIFICATE, step 19->23, case 0 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_19_23.json, which re-derives every number
from the primes alone).

Machine 23, INCREMENT width 39 = F_2(19) + s_min(23) = 31 + 8,
held gears [5, 7] at phases [0, 0].  Free gears [11, 13, 17, 19, 23].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 1.
-/
import IncCert23B

namespace IncCert23

/-! ### case 0: held gears at phases [0, 0] -/

def p0 : List ℕ := [0, 2, 3, 5, 7, 10, 12, 17, 18, 23, 25, 28, 30, 32, 33, 35, 37, 38]
def q0 (t : ℕ) : ℕ := p0.getD t 0
def n0 : ℕ := 18
def yl0 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def w0 (t : ℕ) : ℤ := yl0.getD t 0
def ul0 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, (-1), 0, (-1), (-1), 0, (-1), 0, (-1), 0, (-1), 0, 0, 0, 0, 0, (-1), 0, (-2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, (-1), 0, 0, (-1), (-1), 0, 0, (-1), 0, 0, 2, 3, 3, 3, 3, 1, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 2, (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 2, 0, 0, 2, 2, 2, (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), (-2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 3, 1, 0, 3, 0, 3, 2, 0, 3, 1, 2, 2, 3, 3, 1, 3, 3, 0, 2, 1, (-1), (-1), 1, 1, 1, 0, (-1), 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
def u0 (k : ℕ) : ℤ := ul0.getD k 0

def c0_0 (r t : ℕ) : Bool := gb11 r (q0 t)
def c0_1 (r t : ℕ) : Bool := gb13 r (q0 t)
def c0_2 (r t : ℕ) : Bool := gb17 r (q0 t)
def c0_3 (r t : ℕ) : Bool := gb19 r (q0 t)
def c0_4 (r t : ℕ) : Bool := gb23 r (q0 t)

def S0_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n0, (w0 t + 1) * (if c0_0 r t then 1 else 0)
def S0_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n0, (w0 t + 1) * (if c0_1 r t then 1 else 0)
def S0_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n0, (w0 t + 1) * (if c0_2 r t then 1 else 0)
def S0_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n0, (w0 t + 1) * (if c0_3 r t then 1 else 0)
def S0_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n0, (w0 t + 1) * (if c0_4 r t then 1 else 0)

def L0_0 (r : ℕ) : ℤ := u0 (13 + r) + u0 (41 + r) + u0 (71 + r) + u0 (105 + r)
def L0_1 (r : ℕ) : ℤ := u0 (0 + r) + u0 (133 + r) + u0 (165 + r) + u0 (201 + r)
def L0_2 (r : ℕ) : ℤ := u0 (24 + r) + u0 (116 + r) + u0 (233 + r) + u0 (273 + r)
def L0_3 (r : ℕ) : ℤ := u0 (52 + r) + u0 (146 + r) + u0 (214 + r) + u0 (313 + r)
def L0_4 (r : ℕ) : ℤ := u0 (82 + r) + u0 (178 + r) + u0 (250 + r) + u0 (290 + r)

def aS0_0 (r : ℕ) : ℤ := S0_0 r - L0_0 r
def MS0_0 : ℤ := CaseSplit.mxr (aS0_0) 10
def aS0_1 (r : ℕ) : ℤ := S0_1 r - L0_1 r
def MS0_1 : ℤ := CaseSplit.mxr (aS0_1) 12
def aS0_2 (r : ℕ) : ℤ := S0_2 r - L0_2 r
def MS0_2 : ℤ := CaseSplit.mxr (aS0_2) 16
def aS0_3 (r : ℕ) : ℤ := S0_3 r - L0_3 r
def MS0_3 : ℤ := CaseSplit.mxr (aS0_3) 18
def aS0_4 (r : ℕ) : ℤ := S0_4 r - L0_4 r
def MS0_4 : ℤ := CaseSplit.mxr (aS0_4) 22

def N0_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n0, (if c0_0 ra t && c0_1 rb t then 1 else 0)
def aP0_0 (ra rb : ℕ) : ℤ := -(1) * N0_0 ra rb + u0 (0 + rb) + u0 (13 + ra)
def MP0_0 : ℤ := CaseSplit.mxr2 (aP0_0) 10 12
def N0_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n0, (if c0_0 ra t && c0_2 rb t then 1 else 0)
def aP0_1 (ra rb : ℕ) : ℤ := -(1) * N0_1 ra rb + u0 (24 + rb) + u0 (41 + ra)
def MP0_1 : ℤ := CaseSplit.mxr2 (aP0_1) 10 16
def N0_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n0, (if c0_0 ra t && c0_3 rb t then 1 else 0)
def aP0_2 (ra rb : ℕ) : ℤ := -(1) * N0_2 ra rb + u0 (52 + rb) + u0 (71 + ra)
def MP0_2 : ℤ := CaseSplit.mxr2 (aP0_2) 10 18
def N0_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n0, (if c0_0 ra t && c0_4 rb t then 1 else 0)
def aP0_3 (ra rb : ℕ) : ℤ := -(1) * N0_3 ra rb + u0 (82 + rb) + u0 (105 + ra)
def MP0_3 : ℤ := CaseSplit.mxr2 (aP0_3) 10 22
def P0_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n0, (if c0_1 ra t && c0_2 rb t then 1 else 0)
def C0_4 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n0, (if c0_1 ra t && c0_2 rb t && c0_0 s t then 1 else 0)
def M0_4 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C0_4 ra rb) 10
def E0_4 : List ℕ := [104, 115, 190, 201]
def N0_4 (ra rb : ℕ) : ℤ := if E0_4.contains (ra * 17 + rb) = true then P0_4 ra rb - M0_4 ra rb else 0
def aP0_4 (ra rb : ℕ) : ℤ := -(1) * N0_4 ra rb + u0 (116 + rb) + u0 (133 + ra)
def MP0_4 : ℤ := CaseSplit.mxr2 (aP0_4) 12 16
def P0_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n0, (if c0_1 ra t && c0_3 rb t then 1 else 0)
def C0_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n0, (if c0_1 ra t && c0_3 rb t && c0_0 s t then 1 else 0)
def M0_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C0_5 ra rb) 10
def E0_5 : List ℕ := [31, 67, 73, 107, 131, 138, 207, 214, 238, 244]
def N0_5 (ra rb : ℕ) : ℤ := if E0_5.contains (ra * 19 + rb) = true then P0_5 ra rb - M0_5 ra rb else 0
def aP0_5 (ra rb : ℕ) : ℤ := -(1) * N0_5 ra rb + u0 (146 + rb) + u0 (165 + ra)
def MP0_5 : ℤ := CaseSplit.mxr2 (aP0_5) 12 18
def P0_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n0, (if c0_1 ra t && c0_4 rb t then 1 else 0)
def C0_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n0, (if c0_1 ra t && c0_4 rb t && c0_0 s t then 1 else 0)
def M0_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C0_6 ra rb) 10
def E0_6 : List ℕ := []
def N0_6 (ra rb : ℕ) : ℤ := if E0_6.contains (ra * 23 + rb) = true then P0_6 ra rb - M0_6 ra rb else 0
def aP0_6 (ra rb : ℕ) : ℤ := -(1) * N0_6 ra rb + u0 (178 + rb) + u0 (201 + ra)
def MP0_6 : ℤ := CaseSplit.mxr2 (aP0_6) 12 22
def N0_7 (_ra _rb : ℕ) : ℤ := 0
def aP0_7 (ra rb : ℕ) : ℤ := -(1) * N0_7 ra rb + u0 (214 + rb) + u0 (233 + ra)
def MP0_7 : ℤ := CaseSplit.mxr2 (aP0_7) 16 18
def N0_8 (_ra _rb : ℕ) : ℤ := 0
def aP0_8 (ra rb : ℕ) : ℤ := -(1) * N0_8 ra rb + u0 (250 + rb) + u0 (273 + ra)
def MP0_8 : ℤ := CaseSplit.mxr2 (aP0_8) 16 22
def N0_9 (_ra _rb : ℕ) : ℤ := 0
def aP0_9 (ra rb : ℕ) : ℤ := -(1) * N0_9 ra rb + u0 (290 + rb) + u0 (313 + ra)
def MP0_9 : ℤ := CaseSplit.mxr2 (aP0_9) 18 22

def rhs0 : ℤ := (∑ t ∈ Finset.range n0, w0 t) + 1 * (n0 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn0 : ∀ t, t < n0 → (0 : ℤ) ≤ w0 t := by decide
theorem plt0 : ∀ t, t < n0 → q0 t < 39 := by decide
theorem pfree0_5 : ∀ t, t < n0 → gb5 0 (q0 t) = false := by decide
theorem pfree0_7 : ∀ t, t < n0 → gb7 0 (q0 t) = false := by decide
theorem MSv0_0 : MS0_0 = 5 := by decide +kernel
theorem MSv0_1 : MS0_1 = 8 := by decide +kernel
theorem MSv0_2 : MS0_2 = 0 := by decide +kernel
theorem MSv0_3 : MS0_3 = 0 := by decide +kernel
theorem MSv0_4 : MS0_4 = 0 := by decide +kernel
theorem MPv0_0 : MP0_0 = 0 := by decide +kernel
theorem MPv0_1 : MP0_1 = 0 := by decide +kernel
theorem MPv0_2 : MP0_2 = 0 := by decide +kernel
theorem MPv0_3 : MP0_3 = 0 := by decide +kernel
theorem MPv0_4 : MP0_4 = 0 := by decide +kernel
theorem MPv0_5 : MP0_5 = 0 := by decide +kernel
theorem MPv0_6 : MP0_6 = 0 := by decide +kernel
theorem MPv0_7 : MP0_7 = 0 := by decide +kernel
theorem MPv0_8 : MP0_8 = 0 := by decide +kernel
theorem MPv0_9 : MP0_9 = 4 := by decide +kernel
theorem rhsv0 : rhs0 = 18 := by decide +kernel

/-- **The case-0 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/1.
    (Scaled by the common denominator 1: 17 < 18.) -/
theorem cert0 : MS0_0 + MS0_1 + MS0_2 + MS0_3 + MS0_4 + MP0_0 + MP0_1 + MP0_2 + MP0_3 + MP0_4 + MP0_5 + MP0_6 + MP0_7 + MP0_8 + MP0_9 < rhs0 := by
  rw [MSv0_0, MSv0_1, MSv0_2, MSv0_3, MSv0_4, MPv0_0, MPv0_1, MPv0_2, MPv0_3, MPv0_4, MPv0_5, MPv0_6, MPv0_7, MPv0_8, MPv0_9, rhsv0]
  decide

def Dg0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := (if c0_0 r0 t then 1 else 0) + (if c0_1 r1 t then 1 else 0) + (if c0_2 r2 t then 1 else 0) + (if c0_3 r3 t then 1 else 0) + (if c0_4 r4 t then 1 else 0)
def Wl0_0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c0_0 r0 t && c0_1 r1 t then 1 else 0
def Wl0_1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c0_0 r0 t && c0_2 r2 t then 1 else 0
def Wl0_2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c0_0 r0 t && c0_3 r3 t then 1 else 0
def Wl0_3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c0_0 r0 t && c0_4 r4 t then 1 else 0
def Wl0_4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c0_0 r0 t && c0_1 r1 t && c0_2 r2 t then 1 else 0
def Wl0_5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c0_0 r0 t && c0_1 r1 t && c0_3 r3 t then 1 else 0
def Wl0_6 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c0_0 r0 t && c0_1 r1 t && c0_4 r4 t then 1 else 0
def Wl0_7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c0_0 r0 t && !c0_1 r1 t && c0_2 r2 t && c0_3 r3 t then 1 else 0
def Wl0_8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c0_0 r0 t && !c0_1 r1 t && c0_2 r2 t && c0_4 r4 t then 1 else 0
def Wl0_9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c0_0 r0 t && !c0_1 r1 t && !c0_2 r2 t && c0_3 r3 t && c0_4 r4 t then 1 else 0

/-- **No configuration blocks the whole window in case 0.** -/
theorem nocov0 {r0 r1 r2 r3 r4 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23)
    (hcov : ∀ t, t < n0 → (c0_0 r0 t || c0_1 r1 t || c0_2 r2 t || c0_3 r3 t || c0_4 r4 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n0, (1 : ℤ) + (Wl0_0 r0 r1 r2 r3 r4 t + Wl0_1 r0 r1 r2 r3 r4 t + Wl0_2 r0 r1 r2 r3 r4 t + Wl0_3 r0 r1 r2 r3 r4 t + Wl0_4 r0 r1 r2 r3 r4 t + Wl0_5 r0 r1 r2 r3 r4 t + Wl0_6 r0 r1 r2 r3 r4 t + Wl0_7 r0 r1 r2 r3 r4 t + Wl0_8 r0 r1 r2 r3 r4 t + Wl0_9 r0 r1 r2 r3 r4 t) ≤ Dg0 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Wl0_0, Wl0_1, Wl0_2, Wl0_3, Wl0_4, Wl0_5, Wl0_6, Wl0_7, Wl0_8, Wl0_9, Dg0]
    exact CaseSplit.lowest5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n0, (1 : ℤ) ≤ Dg0 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Dg0]
    exact CaseSplit.degpos5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n0 : ℤ) + ((∑ t ∈ Finset.range n0, Wl0_0 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n0, Wl0_1 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n0, Wl0_2 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n0, Wl0_3 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n0, Wl0_4 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n0, Wl0_5 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n0, Wl0_6 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n0, Wl0_7 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n0, Wl0_8 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n0, Wl0_9 r0 r1 r2 r3 r4 t)) ≤ ∑ t ∈ Finset.range n0, Dg0 r0 r1 r2 r3 r4 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N0_0 r0 r1 ≤ ∑ t ∈ Finset.range n0, Wl0_0 r0 r1 r2 r3 r4 t := by
    simp only [N0_0, Wl0_0, le_refl]
  have hn1 : N0_1 r0 r2 ≤ ∑ t ∈ Finset.range n0, Wl0_1 r0 r1 r2 r3 r4 t := by
    simp only [N0_1, Wl0_1, le_refl]
  have hn2 : N0_2 r0 r3 ≤ ∑ t ∈ Finset.range n0, Wl0_2 r0 r1 r2 r3 r4 t := by
    simp only [N0_2, Wl0_2, le_refl]
  have hn3 : N0_3 r0 r4 ≤ ∑ t ∈ Finset.range n0, Wl0_3 r0 r1 r2 r3 r4 t := by
    simp only [N0_3, Wl0_3, le_refl]
  have hn4 : N0_4 r1 r2 ≤ ∑ t ∈ Finset.range n0, Wl0_4 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n0, Wl0_4 r0 r1 r2 r3 r4 t
        = (if c0_1 r1 t && c0_2 r2 t then (1:ℤ) else 0)
          - (if c0_1 r1 t && c0_2 r2 t && c0_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl0_4]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n0, Wl0_4 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl0_4]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n0, Wl0_4 r0 r1 r2 r3 r4 t
        = P0_4 r1 r2 - C0_4 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P0_4, C0_4]
    have hm : C0_4 r1 r2 r0 ≤ M0_4 r1 r2 :=
      CaseSplit.le_mxr (C0_4 r1 r2) 10 r0 (by omega)
    simp only [N0_4]
    split
    · rw [hL]; omega
    · exact hnn
  have hn5 : N0_5 r1 r3 ≤ ∑ t ∈ Finset.range n0, Wl0_5 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n0, Wl0_5 r0 r1 r2 r3 r4 t
        = (if c0_1 r1 t && c0_3 r3 t then (1:ℤ) else 0)
          - (if c0_1 r1 t && c0_3 r3 t && c0_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl0_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n0, Wl0_5 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl0_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n0, Wl0_5 r0 r1 r2 r3 r4 t
        = P0_5 r1 r3 - C0_5 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P0_5, C0_5]
    have hm : C0_5 r1 r3 r0 ≤ M0_5 r1 r3 :=
      CaseSplit.le_mxr (C0_5 r1 r3) 10 r0 (by omega)
    simp only [N0_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N0_6 r1 r4 ≤ ∑ t ∈ Finset.range n0, Wl0_6 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n0, Wl0_6 r0 r1 r2 r3 r4 t
        = (if c0_1 r1 t && c0_4 r4 t then (1:ℤ) else 0)
          - (if c0_1 r1 t && c0_4 r4 t && c0_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl0_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n0, Wl0_6 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl0_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n0, Wl0_6 r0 r1 r2 r3 r4 t
        = P0_6 r1 r4 - C0_6 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P0_6, C0_6]
    have hm : C0_6 r1 r4 r0 ≤ M0_6 r1 r4 :=
      CaseSplit.le_mxr (C0_6 r1 r4) 10 r0 (by omega)
    simp only [N0_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N0_7 r2 r3 ≤ ∑ t ∈ Finset.range n0, Wl0_7 r0 r1 r2 r3 r4 t := by
    simp only [N0_7]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl0_7]
    exact CaseSplit.ind_nonneg _
  have hn8 : N0_8 r2 r4 ≤ ∑ t ∈ Finset.range n0, Wl0_8 r0 r1 r2 r3 r4 t := by
    simp only [N0_8]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl0_8]
    exact CaseSplit.ind_nonneg _
  have hn9 : N0_9 r3 r4 ≤ ∑ t ∈ Finset.range n0, Wl0_9 r0 r1 r2 r3 r4 t := by
    simp only [N0_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl0_9]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n0, (w0 t + 1) * Dg0 r0 r1 r2 r3 r4 t = S0_0 r0 + S0_1 r1 + S0_2 r2 + S0_3 r3 + S0_4 r4 := by
    simp only [S0_0, S0_1, S0_2, S0_3, S0_4, Dg0, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n0, (w0 t + 1) * Dg0 r0 r1 r2 r3 r4 t
      = (∑ t ∈ Finset.range n0, w0 t * Dg0 r0 r1 r2 r3 r4 t)
        + 1 * (∑ t ∈ Finset.range n0, Dg0 r0 r1 r2 r3 r4 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n0, w0 t)
      ≤ ∑ t ∈ Finset.range n0, w0 t * Dg0 r0 r1 r2 r3 r4 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg0 r0 r1 r2 r3 r4 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w0 t := wnn0 t (Finset.mem_range.mp ht)
    calc w0 t = w0 t * 1 := (mul_one _).symm
      _ ≤ w0 t * Dg0 r0 r1 r2 r3 r4 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS0_0 r0 + aS0_1 r1 + aS0_2 r2 + aS0_3 r3 + aS0_4 r4) + (aP0_0 r0 r1 + aP0_1 r0 r2 + aP0_2 r0 r3 + aP0_3 r0 r4 + aP0_4 r1 r2 + aP0_5 r1 r3 + aP0_6 r1 r4 + aP0_7 r2 r3 + aP0_8 r2 r4 + aP0_9 r3 r4) = (S0_0 r0 + S0_1 r1 + S0_2 r2 + S0_3 r3 + S0_4 r4) - 1 * (N0_0 r0 r1 + N0_1 r0 r2 + N0_2 r0 r3 + N0_3 r0 r4 + N0_4 r1 r2 + N0_5 r1 r3 + N0_6 r1 r4 + N0_7 r2 r3 + N0_8 r2 r4 + N0_9 r3 r4) := by
    simp only [aS0_0, aS0_1, aS0_2, aS0_3, aS0_4, aP0_0, aP0_1, aP0_2, aP0_3, aP0_4, aP0_5, aP0_6, aP0_7, aP0_8, aP0_9, L0_0, L0_1, L0_2, L0_3, L0_4]
    ring
  have bS0 : aS0_0 r0 ≤ MS0_0 := CaseSplit.le_mxr (aS0_0) 10 r0 (by omega)
  have bS1 : aS0_1 r1 ≤ MS0_1 := CaseSplit.le_mxr (aS0_1) 12 r1 (by omega)
  have bS2 : aS0_2 r2 ≤ MS0_2 := CaseSplit.le_mxr (aS0_2) 16 r2 (by omega)
  have bS3 : aS0_3 r3 ≤ MS0_3 := CaseSplit.le_mxr (aS0_3) 18 r3 (by omega)
  have bS4 : aS0_4 r4 ≤ MS0_4 := CaseSplit.le_mxr (aS0_4) 22 r4 (by omega)
  have bP0 : aP0_0 r0 r1 ≤ MP0_0 := CaseSplit.le_mxr2 (aP0_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP0_1 r0 r2 ≤ MP0_1 := CaseSplit.le_mxr2 (aP0_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP0_2 r0 r3 ≤ MP0_2 := CaseSplit.le_mxr2 (aP0_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP0_3 r0 r4 ≤ MP0_3 := CaseSplit.le_mxr2 (aP0_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP0_4 r1 r2 ≤ MP0_4 := CaseSplit.le_mxr2 (aP0_4) 12 16 r1 r2 (by omega) (by omega)
  have bP5 : aP0_5 r1 r3 ≤ MP0_5 := CaseSplit.le_mxr2 (aP0_5) 12 18 r1 r3 (by omega) (by omega)
  have bP6 : aP0_6 r1 r4 ≤ MP0_6 := CaseSplit.le_mxr2 (aP0_6) 12 22 r1 r4 (by omega) (by omega)
  have bP7 : aP0_7 r2 r3 ≤ MP0_7 := CaseSplit.le_mxr2 (aP0_7) 16 18 r2 r3 (by omega) (by omega)
  have bP8 : aP0_8 r2 r4 ≤ MP0_8 := CaseSplit.le_mxr2 (aP0_8) 16 22 r2 r4 (by omega) (by omega)
  have bP9 : aP0_9 r3 r4 ≤ MP0_9 := CaseSplit.le_mxr2 (aP0_9) 18 22 r3 r4 (by omega) (by omega)
  have hrhs : rhs0 = (∑ t ∈ Finset.range n0, w0 t) + 1 * (n0 : ℤ) := rfl
  have hc := cert0
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, bS0, bS1, bS2, bS3, bS4, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9]

end IncCert23
