/-
INCREMENT-WIDTH CERTIFICATE, step 19->23, case 3 of 35 (GENERATED
by research/gen_inc_lean.py from
research/data/r27/cert_inc_19_23.json, which re-derives every number
from the primes alone).

Machine 23, INCREMENT width 39 = F_2(19) + s_min(23) = 31 + 8,
held gears [5, 7] at phases [0, 3].  Free gears [11, 13, 17, 19, 23].
All numbers are the LP thread's exact rational dual scaled by the
case denominator 1.
-/
import IncCert23B

namespace IncCert23

/-! ### case 3: held gears at phases [0, 3] -/

def p3 : List ℕ := [0, 2, 7, 8, 13, 15, 18, 20, 22, 23, 25, 27, 28, 30, 32, 35, 37]
def q3 (t : ℕ) : ℕ := p3.getD t 0
def n3 : ℕ := 17
def yl3 : List ℤ := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def w3 (t : ℕ) : ℤ := yl3.getD t 0
def ul3 : List ℤ := [0, (-1), 0, (-2), (-1), (-1), 0, (-1), (-1), 0, (-1), 0, (-3), 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), (-1), (-1), (-1), 0, 0, (-1), (-1), (-1), (-2), (-1), (-1), 0, (-1), (-1), (-1), (-1), (-1), (-1), 0, 0, (-1), (-1), 0, 0, 1, (-1), 0, 1, 0, 0, 0, 1, 0, 2, 3, 2, 2, 1, 3, 2, 3, 1, 2, 1, 2, 3, 3, 2, 3, 3, (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), (-3), 1, 1, 1, 1, 1, (-1), (-1), 1, 0, 1, (-1), (-1), (-1), 0, 1, 1, 1, 0, 1, (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-2), (-1), (-1), (-1), (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 3, 1, 3, 2, 2, 3, 1, 3, 2, 2, 3, 2, 3, 3, 1, 3, 2, 3, 3, 1, 3, 1, 1, 0, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 0]
def u3 (k : ℕ) : ℤ := ul3.getD k 0

def c3_0 (r t : ℕ) : Bool := gb11 r (q3 t)
def c3_1 (r t : ℕ) : Bool := gb13 r (q3 t)
def c3_2 (r t : ℕ) : Bool := gb17 r (q3 t)
def c3_3 (r t : ℕ) : Bool := gb19 r (q3 t)
def c3_4 (r t : ℕ) : Bool := gb23 r (q3 t)

def S3_0 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (w3 t + 1) * (if c3_0 r t then 1 else 0)
def S3_1 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (w3 t + 1) * (if c3_1 r t then 1 else 0)
def S3_2 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (w3 t + 1) * (if c3_2 r t then 1 else 0)
def S3_3 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (w3 t + 1) * (if c3_3 r t then 1 else 0)
def S3_4 (r : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (w3 t + 1) * (if c3_4 r t then 1 else 0)

def L3_0 (r : ℕ) : ℤ := u3 (13 + r) + u3 (41 + r) + u3 (71 + r) + u3 (105 + r)
def L3_1 (r : ℕ) : ℤ := u3 (0 + r) + u3 (133 + r) + u3 (165 + r) + u3 (201 + r)
def L3_2 (r : ℕ) : ℤ := u3 (24 + r) + u3 (116 + r) + u3 (233 + r) + u3 (273 + r)
def L3_3 (r : ℕ) : ℤ := u3 (52 + r) + u3 (146 + r) + u3 (214 + r) + u3 (313 + r)
def L3_4 (r : ℕ) : ℤ := u3 (82 + r) + u3 (178 + r) + u3 (250 + r) + u3 (290 + r)

def aS3_0 (r : ℕ) : ℤ := S3_0 r - L3_0 r
def MS3_0 : ℤ := CaseSplit.mxr (aS3_0) 10
def aS3_1 (r : ℕ) : ℤ := S3_1 r - L3_1 r
def MS3_1 : ℤ := CaseSplit.mxr (aS3_1) 12
def aS3_2 (r : ℕ) : ℤ := S3_2 r - L3_2 r
def MS3_2 : ℤ := CaseSplit.mxr (aS3_2) 16
def aS3_3 (r : ℕ) : ℤ := S3_3 r - L3_3 r
def MS3_3 : ℤ := CaseSplit.mxr (aS3_3) 18
def aS3_4 (r : ℕ) : ℤ := S3_4 r - L3_4 r
def MS3_4 : ℤ := CaseSplit.mxr (aS3_4) 22

def N3_0 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (if c3_0 ra t && c3_1 rb t then 1 else 0)
def aP3_0 (ra rb : ℕ) : ℤ := -(1) * N3_0 ra rb + u3 (0 + rb) + u3 (13 + ra)
def MP3_0 : ℤ := CaseSplit.mxr2 (aP3_0) 10 12
def N3_1 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (if c3_0 ra t && c3_2 rb t then 1 else 0)
def aP3_1 (ra rb : ℕ) : ℤ := -(1) * N3_1 ra rb + u3 (24 + rb) + u3 (41 + ra)
def MP3_1 : ℤ := CaseSplit.mxr2 (aP3_1) 10 16
def N3_2 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (if c3_0 ra t && c3_3 rb t then 1 else 0)
def aP3_2 (ra rb : ℕ) : ℤ := -(1) * N3_2 ra rb + u3 (52 + rb) + u3 (71 + ra)
def MP3_2 : ℤ := CaseSplit.mxr2 (aP3_2) 10 18
def N3_3 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (if c3_0 ra t && c3_4 rb t then 1 else 0)
def aP3_3 (ra rb : ℕ) : ℤ := -(1) * N3_3 ra rb + u3 (82 + rb) + u3 (105 + ra)
def MP3_3 : ℤ := CaseSplit.mxr2 (aP3_3) 10 22
def P3_4 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (if c3_1 ra t && c3_2 rb t then 1 else 0)
def C3_4 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (if c3_1 ra t && c3_2 rb t && c3_0 s t then 1 else 0)
def M3_4 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C3_4 ra rb) 10
def E3_4 : List ℕ := [57, 63, 68, 79, 104, 115, 158, 169, 188, 194]
def N3_4 (ra rb : ℕ) : ℤ := if E3_4.contains (ra * 17 + rb) = true then P3_4 ra rb - M3_4 ra rb else 0
def aP3_4 (ra rb : ℕ) : ℤ := -(1) * N3_4 ra rb + u3 (116 + rb) + u3 (133 + ra)
def MP3_4 : ℤ := CaseSplit.mxr2 (aP3_4) 12 16
def P3_5 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (if c3_1 ra t && c3_3 rb t then 1 else 0)
def C3_5 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (if c3_1 ra t && c3_3 rb t && c3_0 s t then 1 else 0)
def M3_5 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C3_5 ra rb) 10
def E3_5 : List ℕ := [1, 7, 38, 41, 91, 114, 167, 172, 178, 212]
def N3_5 (ra rb : ℕ) : ℤ := if E3_5.contains (ra * 19 + rb) = true then P3_5 ra rb - M3_5 ra rb else 0
def aP3_5 (ra rb : ℕ) : ℤ := -(1) * N3_5 ra rb + u3 (146 + rb) + u3 (165 + ra)
def MP3_5 : ℤ := CaseSplit.mxr2 (aP3_5) 12 18
def P3_6 (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (if c3_1 ra t && c3_4 rb t then 1 else 0)
def C3_6 (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n3, (if c3_1 ra t && c3_4 rb t && c3_0 s t then 1 else 0)
def M3_6 (ra rb : ℕ) : ℤ := CaseSplit.mxr (C3_6 ra rb) 10
def E3_6 : List ℕ := []
def N3_6 (ra rb : ℕ) : ℤ := if E3_6.contains (ra * 23 + rb) = true then P3_6 ra rb - M3_6 ra rb else 0
def aP3_6 (ra rb : ℕ) : ℤ := -(1) * N3_6 ra rb + u3 (178 + rb) + u3 (201 + ra)
def MP3_6 : ℤ := CaseSplit.mxr2 (aP3_6) 12 22
def N3_7 (_ra _rb : ℕ) : ℤ := 0
def aP3_7 (ra rb : ℕ) : ℤ := -(1) * N3_7 ra rb + u3 (214 + rb) + u3 (233 + ra)
def MP3_7 : ℤ := CaseSplit.mxr2 (aP3_7) 16 18
def N3_8 (_ra _rb : ℕ) : ℤ := 0
def aP3_8 (ra rb : ℕ) : ℤ := -(1) * N3_8 ra rb + u3 (250 + rb) + u3 (273 + ra)
def MP3_8 : ℤ := CaseSplit.mxr2 (aP3_8) 16 22
def N3_9 (_ra _rb : ℕ) : ℤ := 0
def aP3_9 (ra rb : ℕ) : ℤ := -(1) * N3_9 ra rb + u3 (290 + rb) + u3 (313 + ra)
def MP3_9 : ℤ := CaseSplit.mxr2 (aP3_9) 18 22

def rhs3 : ℤ := (∑ t ∈ Finset.range n3, w3 t) + 1 * (n3 : ℤ)

set_option maxRecDepth 40000
set_option maxHeartbeats 4000000

theorem wnn3 : ∀ t, t < n3 → (0 : ℤ) ≤ w3 t := by decide
theorem plt3 : ∀ t, t < n3 → q3 t < 39 := by decide
theorem pfree3_5 : ∀ t, t < n3 → gb5 0 (q3 t) = false := by decide
theorem pfree3_7 : ∀ t, t < n3 → gb7 3 (q3 t) = false := by decide
theorem MSv3_0 : MS3_0 = 3 := by decide +kernel
theorem MSv3_1 : MS3_1 = 8 := by decide +kernel
theorem MSv3_2 : MS3_2 = 0 := by decide +kernel
theorem MSv3_3 : MS3_3 = 0 := by decide +kernel
theorem MSv3_4 : MS3_4 = 0 := by decide +kernel
theorem MPv3_0 : MP3_0 = 0 := by decide +kernel
theorem MPv3_1 : MP3_1 = 0 := by decide +kernel
theorem MPv3_2 : MP3_2 = 0 := by decide +kernel
theorem MPv3_3 : MP3_3 = 0 := by decide +kernel
theorem MPv3_4 : MP3_4 = 0 := by decide +kernel
theorem MPv3_5 : MP3_5 = 0 := by decide +kernel
theorem MPv3_6 : MP3_6 = 0 := by decide +kernel
theorem MPv3_7 : MP3_7 = 0 := by decide +kernel
theorem MPv3_8 : MP3_8 = 0 := by decide +kernel
theorem MPv3_9 : MP3_9 = 5 := by decide +kernel
theorem rhsv3 : rhs3 = 17 := by decide +kernel

/-- **The case-3 certificate**: the dual objective falls short of the
    recursion row's right-hand side.  Margin 1/1.
    (Scaled by the common denominator 1: 16 < 17.) -/
theorem cert3 : MS3_0 + MS3_1 + MS3_2 + MS3_3 + MS3_4 + MP3_0 + MP3_1 + MP3_2 + MP3_3 + MP3_4 + MP3_5 + MP3_6 + MP3_7 + MP3_8 + MP3_9 < rhs3 := by
  rw [MSv3_0, MSv3_1, MSv3_2, MSv3_3, MSv3_4, MPv3_0, MPv3_1, MPv3_2, MPv3_3, MPv3_4, MPv3_5, MPv3_6, MPv3_7, MPv3_8, MPv3_9, rhsv3]
  decide

def Dg3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := (if c3_0 r0 t then 1 else 0) + (if c3_1 r1 t then 1 else 0) + (if c3_2 r2 t then 1 else 0) + (if c3_3 r3 t then 1 else 0) + (if c3_4 r4 t then 1 else 0)
def Wl3_0 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c3_0 r0 t && c3_1 r1 t then 1 else 0
def Wl3_1 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c3_0 r0 t && c3_2 r2 t then 1 else 0
def Wl3_2 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c3_0 r0 t && c3_3 r3 t then 1 else 0
def Wl3_3 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if c3_0 r0 t && c3_4 r4 t then 1 else 0
def Wl3_4 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c3_0 r0 t && c3_1 r1 t && c3_2 r2 t then 1 else 0
def Wl3_5 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c3_0 r0 t && c3_1 r1 t && c3_3 r3 t then 1 else 0
def Wl3_6 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c3_0 r0 t && c3_1 r1 t && c3_4 r4 t then 1 else 0
def Wl3_7 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c3_0 r0 t && !c3_1 r1 t && c3_2 r2 t && c3_3 r3 t then 1 else 0
def Wl3_8 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c3_0 r0 t && !c3_1 r1 t && c3_2 r2 t && c3_4 r4 t then 1 else 0
def Wl3_9 (r0 r1 r2 r3 r4 t : ℕ) : ℤ := if !c3_0 r0 t && !c3_1 r1 t && !c3_2 r2 t && c3_3 r3 t && c3_4 r4 t then 1 else 0

/-- **No configuration blocks the whole window in case 3.** -/
theorem nocov3 {r0 r1 r2 r3 r4 : ℕ} (h0 : r0 < 11) (h1 : r1 < 13) (h2 : r2 < 17) (h3 : r3 < 19) (h4 : r4 < 23)
    (hcov : ∀ t, t < n3 → (c3_0 r0 t || c3_1 r1 t || c3_2 r2 t || c3_3 r3 t || c3_4 r4 t) = true) : False := by
  have hpt : ∀ t ∈ Finset.range n3, (1 : ℤ) + (Wl3_0 r0 r1 r2 r3 r4 t + Wl3_1 r0 r1 r2 r3 r4 t + Wl3_2 r0 r1 r2 r3 r4 t + Wl3_3 r0 r1 r2 r3 r4 t + Wl3_4 r0 r1 r2 r3 r4 t + Wl3_5 r0 r1 r2 r3 r4 t + Wl3_6 r0 r1 r2 r3 r4 t + Wl3_7 r0 r1 r2 r3 r4 t + Wl3_8 r0 r1 r2 r3 r4 t + Wl3_9 r0 r1 r2 r3 r4 t) ≤ Dg3 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Wl3_0, Wl3_1, Wl3_2, Wl3_3, Wl3_4, Wl3_5, Wl3_6, Wl3_7, Wl3_8, Wl3_9, Dg3]
    exact CaseSplit.lowest5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hd1 : ∀ t ∈ Finset.range n3, (1 : ℤ) ≤ Dg3 r0 r1 r2 r3 r4 t := by
    intro t ht
    simp only [Dg3]
    exact CaseSplit.degpos5 _ _ _ _ _ (hcov t (Finset.mem_range.mp ht))
  have hsum : (n3 : ℤ) + ((∑ t ∈ Finset.range n3, Wl3_0 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n3, Wl3_1 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n3, Wl3_2 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n3, Wl3_3 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n3, Wl3_4 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n3, Wl3_5 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n3, Wl3_6 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n3, Wl3_7 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n3, Wl3_8 r0 r1 r2 r3 r4 t) + (∑ t ∈ Finset.range n3, Wl3_9 r0 r1 r2 r3 r4 t)) ≤ ∑ t ∈ Finset.range n3, Dg3 r0 r1 r2 r3 r4 t := by
    have h := Finset.sum_le_sum hpt
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
    exact h
  have hn0 : N3_0 r0 r1 ≤ ∑ t ∈ Finset.range n3, Wl3_0 r0 r1 r2 r3 r4 t := by
    simp only [N3_0, Wl3_0, le_refl]
  have hn1 : N3_1 r0 r2 ≤ ∑ t ∈ Finset.range n3, Wl3_1 r0 r1 r2 r3 r4 t := by
    simp only [N3_1, Wl3_1, le_refl]
  have hn2 : N3_2 r0 r3 ≤ ∑ t ∈ Finset.range n3, Wl3_2 r0 r1 r2 r3 r4 t := by
    simp only [N3_2, Wl3_2, le_refl]
  have hn3 : N3_3 r0 r4 ≤ ∑ t ∈ Finset.range n3, Wl3_3 r0 r1 r2 r3 r4 t := by
    simp only [N3_3, Wl3_3, le_refl]
  have hn4 : N3_4 r1 r2 ≤ ∑ t ∈ Finset.range n3, Wl3_4 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n3, Wl3_4 r0 r1 r2 r3 r4 t
        = (if c3_1 r1 t && c3_2 r2 t then (1:ℤ) else 0)
          - (if c3_1 r1 t && c3_2 r2 t && c3_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl3_4]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n3, Wl3_4 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl3_4]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n3, Wl3_4 r0 r1 r2 r3 r4 t
        = P3_4 r1 r2 - C3_4 r1 r2 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P3_4, C3_4]
    have hm : C3_4 r1 r2 r0 ≤ M3_4 r1 r2 :=
      CaseSplit.le_mxr (C3_4 r1 r2) 10 r0 (by omega)
    simp only [N3_4]
    split
    · rw [hL]; omega
    · exact hnn
  have hn5 : N3_5 r1 r3 ≤ ∑ t ∈ Finset.range n3, Wl3_5 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n3, Wl3_5 r0 r1 r2 r3 r4 t
        = (if c3_1 r1 t && c3_3 r3 t then (1:ℤ) else 0)
          - (if c3_1 r1 t && c3_3 r3 t && c3_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl3_5]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n3, Wl3_5 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl3_5]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n3, Wl3_5 r0 r1 r2 r3 r4 t
        = P3_5 r1 r3 - C3_5 r1 r3 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P3_5, C3_5]
    have hm : C3_5 r1 r3 r0 ≤ M3_5 r1 r3 :=
      CaseSplit.le_mxr (C3_5 r1 r3) 10 r0 (by omega)
    simp only [N3_5]
    split
    · rw [hL]; omega
    · exact hnn
  have hn6 : N3_6 r1 r4 ≤ ∑ t ∈ Finset.range n3, Wl3_6 r0 r1 r2 r3 r4 t := by
    have hsp : ∀ t ∈ Finset.range n3, Wl3_6 r0 r1 r2 r3 r4 t
        = (if c3_1 r1 t && c3_4 r4 t then (1:ℤ) else 0)
          - (if c3_1 r1 t && c3_4 r4 t && c3_0 r0 t then (1:ℤ) else 0) := by
      intro t _
      simp only [Wl3_6]
      exact CaseSplit.ind_low2 _ _ _
    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n3, Wl3_6 r0 r1 r2 r3 r4 t := by
      apply Finset.sum_nonneg
      intro t _
      simp only [Wl3_6]
      exact CaseSplit.ind_nonneg _
    have hL : ∑ t ∈ Finset.range n3, Wl3_6 r0 r1 r2 r3 r4 t
        = P3_6 r1 r4 - C3_6 r1 r4 r0 := by
      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]
      simp only [P3_6, C3_6]
    have hm : C3_6 r1 r4 r0 ≤ M3_6 r1 r4 :=
      CaseSplit.le_mxr (C3_6 r1 r4) 10 r0 (by omega)
    simp only [N3_6]
    split
    · rw [hL]; omega
    · exact hnn
  have hn7 : N3_7 r2 r3 ≤ ∑ t ∈ Finset.range n3, Wl3_7 r0 r1 r2 r3 r4 t := by
    simp only [N3_7]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl3_7]
    exact CaseSplit.ind_nonneg _
  have hn8 : N3_8 r2 r4 ≤ ∑ t ∈ Finset.range n3, Wl3_8 r0 r1 r2 r3 r4 t := by
    simp only [N3_8]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl3_8]
    exact CaseSplit.ind_nonneg _
  have hn9 : N3_9 r3 r4 ≤ ∑ t ∈ Finset.range n3, Wl3_9 r0 r1 r2 r3 r4 t := by
    simp only [N3_9]
    apply Finset.sum_nonneg
    intro t _
    simp only [Wl3_9]
    exact CaseSplit.ind_nonneg _
  have hS : ∑ t ∈ Finset.range n3, (w3 t + 1) * Dg3 r0 r1 r2 r3 r4 t = S3_0 r0 + S3_1 r1 + S3_2 r2 + S3_3 r3 + S3_4 r4 := by
    simp only [S3_0, S3_1, S3_2, S3_3, S3_4, Dg3, mul_add, Finset.sum_add_distrib]
  have hSD : ∑ t ∈ Finset.range n3, (w3 t + 1) * Dg3 r0 r1 r2 r3 r4 t
      = (∑ t ∈ Finset.range n3, w3 t * Dg3 r0 r1 r2 r3 r4 t)
        + 1 * (∑ t ∈ Finset.range n3, Dg3 r0 r1 r2 r3 r4 t) := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  have hwD : (∑ t ∈ Finset.range n3, w3 t)
      ≤ ∑ t ∈ Finset.range n3, w3 t * Dg3 r0 r1 r2 r3 r4 t := by
    apply Finset.sum_le_sum
    intro t ht
    have h1 : (1:ℤ) ≤ Dg3 r0 r1 r2 r3 r4 t := hd1 t ht
    have h2 : (0:ℤ) ≤ w3 t := wnn3 t (Finset.mem_range.mp ht)
    calc w3 t = w3 t * 1 := (mul_one _).symm
      _ ≤ w3 t * Dg3 r0 r1 r2 r3 r4 t := by exact mul_le_mul_of_nonneg_left h1 h2
  have hid : (aS3_0 r0 + aS3_1 r1 + aS3_2 r2 + aS3_3 r3 + aS3_4 r4) + (aP3_0 r0 r1 + aP3_1 r0 r2 + aP3_2 r0 r3 + aP3_3 r0 r4 + aP3_4 r1 r2 + aP3_5 r1 r3 + aP3_6 r1 r4 + aP3_7 r2 r3 + aP3_8 r2 r4 + aP3_9 r3 r4) = (S3_0 r0 + S3_1 r1 + S3_2 r2 + S3_3 r3 + S3_4 r4) - 1 * (N3_0 r0 r1 + N3_1 r0 r2 + N3_2 r0 r3 + N3_3 r0 r4 + N3_4 r1 r2 + N3_5 r1 r3 + N3_6 r1 r4 + N3_7 r2 r3 + N3_8 r2 r4 + N3_9 r3 r4) := by
    simp only [aS3_0, aS3_1, aS3_2, aS3_3, aS3_4, aP3_0, aP3_1, aP3_2, aP3_3, aP3_4, aP3_5, aP3_6, aP3_7, aP3_8, aP3_9, L3_0, L3_1, L3_2, L3_3, L3_4]
    ring
  have bS0 : aS3_0 r0 ≤ MS3_0 := CaseSplit.le_mxr (aS3_0) 10 r0 (by omega)
  have bS1 : aS3_1 r1 ≤ MS3_1 := CaseSplit.le_mxr (aS3_1) 12 r1 (by omega)
  have bS2 : aS3_2 r2 ≤ MS3_2 := CaseSplit.le_mxr (aS3_2) 16 r2 (by omega)
  have bS3 : aS3_3 r3 ≤ MS3_3 := CaseSplit.le_mxr (aS3_3) 18 r3 (by omega)
  have bS4 : aS3_4 r4 ≤ MS3_4 := CaseSplit.le_mxr (aS3_4) 22 r4 (by omega)
  have bP0 : aP3_0 r0 r1 ≤ MP3_0 := CaseSplit.le_mxr2 (aP3_0) 10 12 r0 r1 (by omega) (by omega)
  have bP1 : aP3_1 r0 r2 ≤ MP3_1 := CaseSplit.le_mxr2 (aP3_1) 10 16 r0 r2 (by omega) (by omega)
  have bP2 : aP3_2 r0 r3 ≤ MP3_2 := CaseSplit.le_mxr2 (aP3_2) 10 18 r0 r3 (by omega) (by omega)
  have bP3 : aP3_3 r0 r4 ≤ MP3_3 := CaseSplit.le_mxr2 (aP3_3) 10 22 r0 r4 (by omega) (by omega)
  have bP4 : aP3_4 r1 r2 ≤ MP3_4 := CaseSplit.le_mxr2 (aP3_4) 12 16 r1 r2 (by omega) (by omega)
  have bP5 : aP3_5 r1 r3 ≤ MP3_5 := CaseSplit.le_mxr2 (aP3_5) 12 18 r1 r3 (by omega) (by omega)
  have bP6 : aP3_6 r1 r4 ≤ MP3_6 := CaseSplit.le_mxr2 (aP3_6) 12 22 r1 r4 (by omega) (by omega)
  have bP7 : aP3_7 r2 r3 ≤ MP3_7 := CaseSplit.le_mxr2 (aP3_7) 16 18 r2 r3 (by omega) (by omega)
  have bP8 : aP3_8 r2 r4 ≤ MP3_8 := CaseSplit.le_mxr2 (aP3_8) 16 22 r2 r4 (by omega) (by omega)
  have bP9 : aP3_9 r3 r4 ≤ MP3_9 := CaseSplit.le_mxr2 (aP3_9) 18 22 r3 r4 (by omega) (by omega)
  have hrhs : rhs3 = (∑ t ∈ Finset.range n3, w3 t) + 1 * (n3 : ℤ) := rfl
  have hc := cert3
  linarith [hsum, hS, hSD, hwD, hid, hrhs, hc, hn0, hn1, hn2, hn3, hn4, hn5, hn6, hn7, hn8, hn9, bS0, bS1, bS2, bS3, bS4, bP0, bP1, bP2, bP3, bP4, bP5, bP6, bP7, bP8, bP9]

end IncCert23
