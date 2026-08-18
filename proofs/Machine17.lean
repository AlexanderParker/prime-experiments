/-
The alpha1 = 4/3 certificate at machine 17.

Machine 17 is the gear set `{5, 7, 11, 13, 17}`, period 85085. The
constructor's per-machine statement (constructor.md sec 21; alpha1 = 4/3,
q' = 19, F_k = 18, F2_k = 25, budget 26.44):

    F2_k(17) <= F_k(17) + alpha1 * q'/3,

which clears denominators to `9 * F2 <= 9 * F + 4 * q'` (225 <= 238).

This is the second machine certified, and it settles the question the round
was set to answer: with the CRT-tuple recipe the PERIOD SCAN STILL WINS at
17. The 85085 cases cost about as much as machine 13's 5005 did, because the
tuple `(k%5, k%7, k%11, k%13, k%17)` keeps every modulus a single digit.
Tiers B and C of the constructor's class machinery are needed for the
human-scale argument, not for the kernel.

Two changes from `Machine13.lean`, both to keep the scan linear:

* the second window fact is stated as a COUNT (`2 <= length of the filtered
  window`) rather than as a nested existential, so the scan costs 25 tests
  per tuple instead of 25 * 25;
* the two witnesses are recovered from the filtered list's first two
  entries, distinct because `List.range` filters to a `Nodup` list.

Verified numerically before formalising: gear residues, F = 18, F2 = 25,
the budget, and both window facts (the 25 is tight - 24 fails).
-/

import Corridor

namespace Machine17

/-! ## The decidable core, on CRT tuples -/

/-- The opening test on a CRT tuple `(k%5, k%7, k%11, k%13, k%17)`. -/
def expT (a b c d e : Nat) : Bool :=
  a != 1 && a != 4 && b != 6 && b != 1 && c != 2 && c != 9 &&
    d != 11 && d != 2 && e != 3 && e != 14

/-- The test `n` slots further on. -/
def atT (a b c d e n : Nat) : Bool :=
  expT ((a+n)%5) ((b+n)%7) ((c+n)%11) ((d+n)%13) ((e+n)%17)

/-- Some slot among the next 18 is an opening. -/
def win18T (a b c d e : Nat) : Bool :=
  (List.range 18).any fun i => atT a b c d e (i+1)

/-- The openings among the next 25 slots, as offsets. -/
def expWin (a b c d e : Nat) : List Nat :=
  (List.range 25).filter fun i => atT a b c d e (i+1)

/-- From an opening, at least two more openings arrive within 25 slots. -/
def pair25T (a b c d e : Nat) : Bool :=
  !(atT a b c d e 0) || decide (2 ≤ (expWin a b c d e).length)

/-- One SLICE of the machine-17 period: all 5005 tuples sharing a fixed
residue `e` mod 17. Each slice is exactly the size of machine 13's whole
period, which is the point - see the note below. -/
def w18Slice (e : Nat) : Bool :=
  (List.range 5).all fun a => (List.range 7).all fun b =>
    (List.range 11).all fun c => (List.range 13).all fun d => win18T a b c d e

def w25Slice (e : Nat) : Bool :=
  (List.range 5).all fun a => (List.range 7).all fun b =>
    (List.range 11).all fun c => (List.range 13).all fun d => pair25T a b c d e

/-! The scan is CHUNKED by the mod-17 coordinate. Two earlier shapes failed
at 85085 cases: a full `decidableBallLT` nest blows the proof TERM (85085
branches, 2 GB and climbing), while folding all five quantifiers into one
`Bool` keeps the term at `rfl` but rebuilds the inner `List.range 17` 5005
times and does not finish. Slicing fixes both: the `Prop`-level quantifier
ranges over 17 values only (tiny term) and each slice is a 5005-tuple
`Bool` - the exact shape already known to evaluate in ~12s at machine 13.
Measured: 16s per slice, both facts. -/

/-! Each slice is checked on its own: 17 independent kernel evaluations of
5005 tuples each, rather than one evaluation of 85085. -/

set_option maxRecDepth 40000 in
theorem s18_0 : w18Slice 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s18_1 : w18Slice 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s18_2 : w18Slice 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s18_3 : w18Slice 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s18_4 : w18Slice 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s18_5 : w18Slice 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s18_6 : w18Slice 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s18_7 : w18Slice 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s18_8 : w18Slice 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s18_9 : w18Slice 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s18_10 : w18Slice 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s18_11 : w18Slice 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s18_12 : w18Slice 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s18_13 : w18Slice 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s18_14 : w18Slice 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s18_15 : w18Slice 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s18_16 : w18Slice 16 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s25_0 : w25Slice 0 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s25_1 : w25Slice 1 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s25_2 : w25Slice 2 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s25_3 : w25Slice 3 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s25_4 : w25Slice 4 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s25_5 : w25Slice 5 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s25_6 : w25Slice 6 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s25_7 : w25Slice 7 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s25_8 : w25Slice 8 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s25_9 : w25Slice 9 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s25_10 : w25Slice 10 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s25_11 : w25Slice 11 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s25_12 : w25Slice 12 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s25_13 : w25Slice 13 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s25_14 : w25Slice 14 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s25_15 : w25Slice 15 = true := by decide +kernel

set_option maxRecDepth 40000 in
theorem s25_16 : w25Slice 16 = true := by decide +kernel


/-- **One period, `F_k(17) <= 18`.** Every 18-slot window holds an opening. -/
theorem w18All : ∀ e < 17, w18Slice e = true := by
  intro e he
  interval_cases e
  exacts [s18_0, s18_1, s18_2, s18_3, s18_4, s18_5, s18_6, s18_7, s18_8, s18_9, s18_10, s18_11, s18_12, s18_13, s18_14, s18_15, s18_16]

/-- **One period, `F2_k(17) <= 25`.** From any opening, two more openings
arrive within 25 slots. -/
theorem w25All : ∀ e < 17, w25Slice e = true := by
  intro e he
  interval_cases e
  exacts [s25_0, s25_1, s25_2, s25_3, s25_4, s25_5, s25_6, s25_7, s25_8, s25_9, s25_10, s25_11, s25_12, s25_13, s25_14, s25_15, s25_16]

theorem w18 {a b c d e : ℕ} (ha : a < 5) (hb : b < 7) (hc : c < 11)
    (hd : d < 13) (he : e < 17) : win18T a b c d e = true := by
  have h := w18All e he
  rw [w18Slice, List.all_eq_true] at h
  have h1 := h a (List.mem_range.mpr ha)
  rw [List.all_eq_true] at h1
  have h2 := h1 b (List.mem_range.mpr hb)
  rw [List.all_eq_true] at h2
  have h3 := h2 c (List.mem_range.mpr hc)
  rw [List.all_eq_true] at h3
  exact h3 d (List.mem_range.mpr hd)

theorem w25 {a b c d e : ℕ} (ha : a < 5) (hb : b < 7) (hc : c < 11)
    (hd : d < 13) (he : e < 17) : pair25T a b c d e = true := by
  have h := w25All e he
  rw [w25Slice, List.all_eq_true] at h
  have h1 := h a (List.mem_range.mpr ha)
  rw [List.all_eq_true] at h1
  have h2 := h1 b (List.mem_range.mpr hb)
  rw [List.all_eq_true] at h2
  have h3 := h2 c (List.mem_range.mpr hc)
  rw [List.all_eq_true] at h3
  exact h3 d (List.mem_range.mpr hd)

/-! ## Openings -/

/-- An opening of machine 17: no gear in `{5,7,11,13,17}` divides either
member of slot `k`. -/
def Exposed17 (k : ℕ) : Prop :=
  ¬ (5 ∣ Census.lo k) ∧ ¬ (5 ∣ Census.hi k) ∧
    ¬ (7 ∣ Census.lo k) ∧ ¬ (7 ∣ Census.hi k) ∧
    ¬ (11 ∣ Census.lo k) ∧ ¬ (11 ∣ Census.hi k) ∧
    ¬ (13 ∣ Census.lo k) ∧ ¬ (13 ∣ Census.hi k) ∧
    ¬ (17 ∣ Census.lo k) ∧ ¬ (17 ∣ Census.hi k)

instance (k : ℕ) : Decidable (Exposed17 k) := by unfold Exposed17; infer_instance

set_option maxHeartbeats 1000000 in
/-- Openings are exactly the CRT-tuple test. -/
theorem exposed17_iff {k : ℕ} (hk : 1 ≤ k) :
    Exposed17 k ↔ expT (k % 5) (k % 7) (k % 11) (k % 13) (k % 17) = true := by
  have h5lo : (5 ∣ Census.lo k) ↔ k % 5 = 1 := by simp only [Census.lo]; omega
  have h5hi : (5 ∣ Census.hi k) ↔ k % 5 = 4 := by simp only [Census.hi]; omega
  have h7lo : (7 ∣ Census.lo k) ↔ k % 7 = 6 := by simp only [Census.lo]; omega
  have h7hi : (7 ∣ Census.hi k) ↔ k % 7 = 1 := by simp only [Census.hi]; omega
  have h11lo : (11 ∣ Census.lo k) ↔ k % 11 = 2 := by simp only [Census.lo]; omega
  have h11hi : (11 ∣ Census.hi k) ↔ k % 11 = 9 := by simp only [Census.hi]; omega
  have h13lo : (13 ∣ Census.lo k) ↔ k % 13 = 11 := by simp only [Census.lo]; omega
  have h13hi : (13 ∣ Census.hi k) ↔ k % 13 = 2 := by simp only [Census.hi]; omega
  have h17lo : (17 ∣ Census.lo k) ↔ k % 17 = 3 := by simp only [Census.lo]; omega
  have h17hi : (17 ∣ Census.hi k) ↔ k % 17 = 14 := by simp only [Census.hi]; omega
  unfold Exposed17
  rw [h5lo, h5hi, h7lo, h7hi, h11lo, h11hi, h13lo, h13hi, h17lo, h17hi]
  simp only [expT, Bool.and_eq_true, bne_iff_ne, ne_eq, and_assoc]

/-- Shifted form. -/
theorem atT_iff {k : ℕ} (hk : 1 ≤ k) (n : ℕ) :
    atT (k % 5) (k % 7) (k % 11) (k % 13) (k % 17) n = true ↔ Exposed17 (k + n) := by
  rw [atT, exposed17_iff (show 1 ≤ k + n by omega)]
  have e5 : (k % 5 + n) % 5 = (k + n) % 5 := by omega
  have e7 : (k % 7 + n) % 7 = (k + n) % 7 := by omega
  have e11 : (k % 11 + n) % 11 = (k + n) % 11 := by omega
  have e13 : (k % 13 + n) % 13 = (k + n) % 13 := by omega
  have e17 : (k % 17 + n) % 17 = (k + n) % 17 := by omega
  rw [e5, e7, e11, e13, e17]

/-- Machine-17 openings are (5,7)-corridor openings. -/
theorem exposed17_exposed {k : ℕ} (h : Exposed17 k) : Corridor.Exposed k :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1⟩

/-! ## The gap bounds -/

/-- **`F_k(17) <= 18`.** -/
theorem gap_le {a b : ℕ} (ha : 1 ≤ a) (hab : a < b)
    (_hEa : Exposed17 a) (_hEb : Exposed17 b)
    (hg : ∀ j, a < j → j < b → ¬ Exposed17 j) : b - a ≤ 18 := by
  by_contra hlt
  have h := w18 (a := a % 5) (b := a % 7) (c := a % 11) (d := a % 13) (e := a % 17)
    (by omega) (by omega) (by omega) (by omega) (by omega)
  rw [win18T, List.any_eq_true] at h
  obtain ⟨i, hi, hv⟩ := h
  have hi18 := List.mem_range.mp hi
  exact hg (a + (i+1)) (by omega) (by omega) ((atT_iff ha _).mp hv)

/-- **`F2_k(17) <= 25`.** Two adjacent machine gaps span at most 25 slots. -/
theorem pair_sum_le {a b c : ℕ} (ha : 1 ≤ a) (_hab : a < b) (_hbc : b < c)
    (hEa : Exposed17 a) (_hEb : Exposed17 b) (_hEc : Exposed17 c)
    (hg1 : ∀ j, a < j → j < b → ¬ Exposed17 j)
    (hg2 : ∀ j, b < j → j < c → ¬ Exposed17 j) : c - a ≤ 25 := by
  have ha0 : atT (a % 5) (a % 7) (a % 11) (a % 13) (a % 17) 0 = true :=
    (atT_iff ha 0).mpr (by simpa using hEa)
  have h := w25 (a := a % 5) (b := a % 7) (c := a % 11) (d := a % 13) (e := a % 17)
    (by omega) (by omega) (by omega) (by omega) (by omega)
  rw [pair25T, ha0] at h
  simp only [Bool.not_true, Bool.false_or, decide_eq_true_iff] at h
  -- two distinct openings in the window, from the filtered list
  have hnd : (expWin (a % 5) (a % 7) (a % 11) (a % 13) (a % 17)).Nodup :=
    List.Nodup.filter _ List.nodup_range
  obtain ⟨x, y, rest, hl⟩ :
      ∃ x y rest, expWin (a % 5) (a % 7) (a % 11) (a % 13) (a % 17) = x :: y :: rest := by
    rcases hlist : expWin (a % 5) (a % 7) (a % 11) (a % 13) (a % 17) with _ | ⟨x, _ | ⟨y, rest⟩⟩
    · rw [hlist] at h; simp at h
    · rw [hlist] at h; simp at h
    · exact ⟨x, y, rest, rfl⟩
  have hxne : x ≠ y := by
    rw [hl] at hnd
    intro hxy
    exact (List.nodup_cons.mp hnd).1 (by rw [hxy]; simp)
  have hx := List.mem_filter.mp (show x ∈ (List.range 25).filter
      (fun i => atT (a % 5) (a % 7) (a % 11) (a % 13) (a % 17) (i+1)) by
    rw [← expWin, hl]; simp)
  have hy := List.mem_filter.mp (show y ∈ (List.range 25).filter
      (fun i => atT (a % 5) (a % 7) (a % 11) (a % 13) (a % 17) (i+1)) by
    rw [← expWin, hl]; simp)
  have hx25 := List.mem_range.mp hx.1
  have hy25 := List.mem_range.mp hy.1
  have hEx : Exposed17 (a + (x+1)) := (atT_iff ha _).mp (by simpa using hx.2)
  have hEy : Exposed17 (a + (y+1)) := (atT_iff ha _).mp (by simpa using hy.2)
  have hbx : b ≤ a + (x+1) := by
    by_contra hc'
    exact hg1 _ (by omega) (by omega) hEx
  have hby : b ≤ a + (y+1) := by
    by_contra hc'
    exact hg1 _ (by omega) (by omega) hEy
  have hcx : a + (x+1) = b ∨ c ≤ a + (x+1) := by
    rcases eq_or_lt_of_le hbx with h' | h'
    · exact Or.inl h'.symm
    · right; by_contra hc'
      exact hg2 _ (by omega) (by omega) hEx
  have hcy : a + (y+1) = b ∨ c ≤ a + (y+1) := by
    rcases eq_or_lt_of_le hby with h' | h'
    · exact Or.inl h'.symm
    · right; by_contra hc'
      exact hg2 _ (by omega) (by omega) hEy
  rcases hcx with hx' | hx' <;> rcases hcy with hy' | hy' <;> omega

/-! ## The headline forms -/

/-- **The alpha1 = 4/3 certificate at machine 17**, denominators cleared:
`9 * F2 <= 9 * F + 4 * q'` with `F = 18`, `q' = 19` (225 <= 238). -/
theorem alpha1_certificate {a b c : ℕ} (ha : 1 ≤ a) (hab : a < b) (hbc : b < c)
    (hEa : Exposed17 a) (hEb : Exposed17 b) (hEc : Exposed17 c)
    (hg1 : ∀ j, a < j → j < b → ¬ Exposed17 j)
    (hg2 : ∀ j, b < j → j < c → ¬ Exposed17 j) :
    9 * (c - a) ≤ 9 * 18 + 4 * 19 := by
  have := pair_sum_le ha hab hbc hEa hEb hEc hg1 hg2
  omega

/-- **Lemma 1 at y = 17**: `F2 - F <= alpha1 * q'` with `alpha1 = 4/3`,
`q' = 19`, in cleared form `3 * (F2 - F) <= 4 * q'`. -/
theorem lemma1_at_17 {a b c : ℕ} (ha : 1 ≤ a) (hab : a < b) (hbc : b < c)
    (hEa : Exposed17 a) (hEb : Exposed17 b) (hEc : Exposed17 c)
    (hg1 : ∀ j, a < j → j < b → ¬ Exposed17 j)
    (hg2 : ∀ j, b < j → j < c → ¬ Exposed17 j) :
    3 * ((c - a) - 18) ≤ 4 * 19 := by
  have := pair_sum_le ha hab hbc hEa hEb hEc hg1 hg2
  omega

end Machine17
