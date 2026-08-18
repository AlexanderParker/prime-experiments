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

/-- The whole machine-17 period as ONE Bool: both window facts, all 85085
CRT tuples. Keeping the quantifiers inside a `Bool` computation rather than
in the `Prop` means the proof term is a single `rfl` - at 85085 cases a
nested `decidableBallLT` term exhausts memory, which is exactly what
happened on the first attempt. -/
def w18Bool : Bool :=
  (List.range 5).all fun a => (List.range 7).all fun b =>
    (List.range 11).all fun c => (List.range 13).all fun d =>
      (List.range 17).all fun e => win18T a b c d e

def w25Bool : Bool :=
  (List.range 5).all fun a => (List.range 7).all fun b =>
    (List.range 11).all fun c => (List.range 13).all fun d =>
      (List.range 17).all fun e => pair25T a b c d e

/-- **One period, `F_k(17) <= 18`.** Every 18-slot window holds an opening. -/
theorem w18Bool_eq : w18Bool = true := by decide +kernel

/-- **One period, `F2_k(17) <= 25`.** From any opening, two more openings
arrive within 25 slots. -/
theorem w25Bool_eq : w25Bool = true := by decide +kernel

theorem w18 {a b c d e : ℕ} (ha : a < 5) (hb : b < 7) (hc : c < 11)
    (hd : d < 13) (he : e < 17) : win18T a b c d e = true := by
  have h := w18Bool_eq
  rw [w18Bool, List.all_eq_true] at h
  have h1 := h a (List.mem_range.mpr ha)
  rw [List.all_eq_true] at h1
  have h2 := h1 b (List.mem_range.mpr hb)
  rw [List.all_eq_true] at h2
  have h3 := h2 c (List.mem_range.mpr hc)
  rw [List.all_eq_true] at h3
  have h4 := h3 d (List.mem_range.mpr hd)
  rw [List.all_eq_true] at h4
  exact h4 e (List.mem_range.mpr he)

theorem w25 {a b c d e : ℕ} (ha : a < 5) (hb : b < 7) (hc : c < 11)
    (hd : d < 13) (he : e < 17) : pair25T a b c d e = true := by
  have h := w25Bool_eq
  rw [w25Bool, List.all_eq_true] at h
  have h1 := h a (List.mem_range.mpr ha)
  rw [List.all_eq_true] at h1
  have h2 := h1 b (List.mem_range.mpr hb)
  rw [List.all_eq_true] at h2
  have h3 := h2 c (List.mem_range.mpr hc)
  rw [List.all_eq_true] at h3
  have h4 := h3 d (List.mem_range.mpr hd)
  rw [List.all_eq_true] at h4
  exact h4 e (List.mem_range.mpr he)

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
theorem pair_sum_le {a b c : ℕ} (ha : 1 ≤ a) (hab : a < b) (_hbc : b < c)
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
    List.Nodup.filter _ (List.nodup_range _)
  obtain ⟨x, y, rest, hl⟩ :
      ∃ x y rest, expWin (a % 5) (a % 7) (a % 11) (a % 13) (a % 17) = x :: y :: rest := by
    rcases hlist : expWin (a % 5) (a % 7) (a % 11) (a % 13) (a % 17) with _ | ⟨x, _ | ⟨y, rest⟩⟩
    · rw [hlist] at h; simp at h
    · rw [hlist] at h; simp at h
    · exact ⟨x, y, rest, hlist⟩
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
