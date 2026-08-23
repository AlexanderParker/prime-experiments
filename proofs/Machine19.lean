/-
The machine-19 certificate: `F_k(19) = 25`, `F2_k(19) = 31`, `F4_k(19) = 38`.

Machine 19 is the gear set `{5, 7, 11, 13, 17, 19}`, period 1,616,615 - the
machine round 15 measured at ~86 minutes and declared the tier-C wall, and
round 18's encoding re-attack (openings-only starts, one counting walk)
brought back into range. The 323 slices of 5005 CRT tuples each live in
`Machine19S0.lean` .. `Machine19S16.lean`; this file assembles them and
draws the consequences:

* `gap_le`       - `F_k(19) <= 25`  (every machine gap spans at most 25 slots)
* `pair_sum_le`  - `F2_k(19) <= 31` (two adjacent gaps span at most 31)
* `quad_sum_le`  - `F4_k(19) <= 38` (four adjacent gaps span at most 38)
* `alpha1_certificate`, `lemma1_at_19` - the alpha1 = 4/3 forms
* `shallow_flatness` - `F_4 <= F + q'` (38 <= 25 + 23): the first
  kernel-checked instance of the shallow-flatness hypothesis that
  `Spectrum.merged_le_of_shallow` consumes, at the machine where the
  route's tier-C wall used to stand.

Verified over the full period numerically before formalising: F ladder
`F_1..F_5 = 25, 31, 35, 38, 47`, openings 378,675, and the fuel census row
(19, 23): N3 = 62, k_max = 3.

Third machine certified after 13 (period 5,005) and 17 (period 85,085);
the scan is 19x machine 17's and lands by the same slice-and-assemble
recipe with the round-18 encoding.
-/

import Spectrum
import Machine19S0
import Machine19S1
import Machine19S2
import Machine19S3
import Machine19S4
import Machine19S5
import Machine19S6
import Machine19S7
import Machine19S8
import Machine19S9
import Machine19S10
import Machine19S11
import Machine19S12
import Machine19S13
import Machine19S14
import Machine19S15
import Machine19S16

namespace Machine19

/-! ## Assembly: the whole period -/

/-- **One period, all 323 slices.** -/
theorem sliceAll : ∀ e < 17, ∀ f < 19, slice e f = true := by
  intro e he
  interval_cases e
  exacts [asm0, asm1, asm2, asm3, asm4, asm5, asm6, asm7, asm8, asm9,
    asm10, asm11, asm12, asm13, asm14, asm15, asm16]

/-- The tuple-level fact, unpacked from the slice. -/
theorem okAll {a b c d e f : ℕ} (ha : a < 5) (hb : b < 7) (hc : c < 11)
    (hd : d < 13) (he : e < 17) (hf : f < 19) : okT a b c d e f = true := by
  have h := sliceAll e he f hf
  rw [slice, List.all_eq_true] at h
  have h1 := h a (List.mem_range.mpr ha)
  rw [List.all_eq_true] at h1
  have h2 := h1 b (List.mem_range.mpr hb)
  rw [List.all_eq_true] at h2
  have h3 := h2 c (List.mem_range.mpr hc)
  rw [List.all_eq_true] at h3
  exact h3 d (List.mem_range.mpr hd)

/-! ## Openings -/

/-- An opening of machine 19: no gear in `{5,7,11,13,17,19}` divides either
member of slot `k`. -/
def Exposed19 (k : ℕ) : Prop :=
  ¬ (5 ∣ Census.lo k) ∧ ¬ (5 ∣ Census.hi k) ∧
    ¬ (7 ∣ Census.lo k) ∧ ¬ (7 ∣ Census.hi k) ∧
    ¬ (11 ∣ Census.lo k) ∧ ¬ (11 ∣ Census.hi k) ∧
    ¬ (13 ∣ Census.lo k) ∧ ¬ (13 ∣ Census.hi k) ∧
    ¬ (17 ∣ Census.lo k) ∧ ¬ (17 ∣ Census.hi k) ∧
    ¬ (19 ∣ Census.lo k) ∧ ¬ (19 ∣ Census.hi k)

instance (k : ℕ) : Decidable (Exposed19 k) := by unfold Exposed19; infer_instance

set_option maxHeartbeats 1000000 in
/-- Openings are exactly the CRT-tuple test. -/
theorem exposed19_iff {k : ℕ} (hk : 1 ≤ k) :
    Exposed19 k ↔
      expT (k % 5) (k % 7) (k % 11) (k % 13) (k % 17) (k % 19) = true := by
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
  have h19lo : (19 ∣ Census.lo k) ↔ k % 19 = 16 := by simp only [Census.lo]; omega
  have h19hi : (19 ∣ Census.hi k) ↔ k % 19 = 3 := by simp only [Census.hi]; omega
  unfold Exposed19
  rw [h5lo, h5hi, h7lo, h7hi, h11lo, h11hi, h13lo, h13hi, h17lo, h17hi,
    h19lo, h19hi]
  simp only [expT, Bool.and_eq_true, bne_iff_ne, ne_eq, and_assoc]

/-- Shifted form. -/
theorem atT_iff {k : ℕ} (hk : 1 ≤ k) (n : ℕ) :
    atT (k % 5) (k % 7) (k % 11) (k % 13) (k % 17) (k % 19) n = true ↔
      Exposed19 (k + n) := by
  rw [atT, exposed19_iff (show 1 ≤ k + n by omega)]
  have e5 : (k % 5 + n) % 5 = (k + n) % 5 := by omega
  have e7 : (k % 7 + n) % 7 = (k + n) % 7 := by omega
  have e11 : (k % 11 + n) % 11 = (k + n) % 11 := by omega
  have e13 : (k % 13 + n) % 13 = (k + n) % 13 := by omega
  have e17 : (k % 17 + n) % 17 = (k + n) % 17 := by omega
  have e19 : (k % 19 + n) % 19 = (k + n) % 19 := by omega
  rw [e5, e7, e11, e13, e17, e19]

/-- Machine-19 openings are (5,7)-corridor openings. -/
theorem exposed19_exposed {k : ℕ} (h : Exposed19 k) : Corridor.Exposed k :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1⟩

/-! ## The window facts, extracted at an opening -/

/-- At an opening, the three window facts of `okT` hold. -/
theorem window_facts {a : ℕ} (ha : 1 ≤ a) (hEa : Exposed19 a) :
    ((List.range 25).any fun i =>
        atT (a % 5) (a % 7) (a % 11) (a % 13) (a % 17) (a % 19) (i+1)) = true ∧
      2 ≤ (List.range 31).countP (fun i =>
        atT (a % 5) (a % 7) (a % 11) (a % 13) (a % 17) (a % 19) (i+1)) ∧
      4 ≤ (List.range 38).countP (fun i =>
        atT (a % 5) (a % 7) (a % 11) (a % 13) (a % 17) (a % 19) (i+1)) := by
  have ha0 : atT (a % 5) (a % 7) (a % 11) (a % 13) (a % 17) (a % 19) 0 = true :=
    (atT_iff ha 0).mpr (by simpa using hEa)
  have h := okAll (a := a % 5) (b := a % 7) (c := a % 11) (d := a % 13)
    (e := a % 17) (f := a % 19)
    (by omega) (by omega) (by omega) (by omega) (by omega) (by omega)
  rw [okT, ha0] at h
  simp only [Bool.not_true, Bool.false_or, Bool.and_eq_true, Nat.ble_eq] at h
  exact ⟨h.1, h.2.1, h.2.2⟩

/-! ## The gap bounds -/

/-- **`F_k(19) <= 25`.** -/
theorem gap_le {a b : ℕ} (ha : 1 ≤ a) (hab : a < b)
    (hEa : Exposed19 a) (_hEb : Exposed19 b)
    (hg : ∀ j, a < j → j < b → ¬ Exposed19 j) : b - a ≤ 25 := by
  by_contra hlt
  have h := (window_facts ha hEa).1
  rw [List.any_eq_true] at h
  obtain ⟨i, hi, hv⟩ := h
  have hi25 := List.mem_range.mp hi
  exact hg (a + (i+1)) (by omega) (by omega) ((atT_iff ha _).mp hv)

/-- **`F2_k(19) <= 31`.** Two adjacent machine gaps span at most 31 slots. -/
theorem pair_sum_le {a b c : ℕ} (ha : 1 ≤ a) (_hab : a < b) (_hbc : b < c)
    (hEa : Exposed19 a) (_hEb : Exposed19 b) (_hEc : Exposed19 c)
    (hg1 : ∀ j, a < j → j < b → ¬ Exposed19 j)
    (hg2 : ∀ j, b < j → j < c → ¬ Exposed19 j) : c - a ≤ 31 := by
  have h := (window_facts ha hEa).2.1
  rw [List.countP_eq_length_filter] at h
  set p : ℕ → Bool := fun i =>
    atT (a % 5) (a % 7) (a % 11) (a % 13) (a % 17) (a % 19) (i+1) with hp
  have hnd : ((List.range 31).filter p).Nodup :=
    List.Nodup.filter _ List.nodup_range
  obtain ⟨x, y, rest, hl⟩ :
      ∃ x y rest, (List.range 31).filter p = x :: y :: rest := by
    rcases hlist : (List.range 31).filter p with _ | ⟨x, _ | ⟨y, rest⟩⟩
    · rw [hlist] at h; simp at h
    · rw [hlist] at h; simp at h
    · exact ⟨x, y, rest, rfl⟩
  have hxne : x ≠ y := by
    rw [hl] at hnd
    intro hxy
    exact (List.nodup_cons.mp hnd).1 (by rw [hxy]; simp)
  have hx := List.mem_filter.mp (show x ∈ (List.range 31).filter p by rw [hl]; simp)
  have hy := List.mem_filter.mp (show y ∈ (List.range 31).filter p by rw [hl]; simp)
  have hx31 := List.mem_range.mp hx.1
  have hy31 := List.mem_range.mp hy.1
  have hEx : Exposed19 (a + (x+1)) := (atT_iff ha _).mp (by simpa using hx.2)
  have hEy : Exposed19 (a + (y+1)) := (atT_iff ha _).mp (by simpa using hy.2)
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

/-- **`F4_k(19) <= 38`.** Four adjacent machine gaps - five consecutive
openings `a < b < c < d < e` - span at most 38 slots. This is the depth-4
spectrum value `F_4(19) = 38`; with `q' = 23` it gives shallow flatness
`F_4 <= F + q'` (38 <= 48), the hypothesis `Spectrum.merged_le_of_shallow`
consumes. -/
theorem quad_sum_le {a b c d e : ℕ} (ha : 1 ≤ a)
    (_hab : a < b) (hbc : b < c) (hcd : c < d) (_hde : d < e)
    (hEa : Exposed19 a) (_hEb : Exposed19 b) (_hEc : Exposed19 c)
    (_hEd : Exposed19 d) (_hEe : Exposed19 e)
    (hg1 : ∀ j, a < j → j < b → ¬ Exposed19 j)
    (hg2 : ∀ j, b < j → j < c → ¬ Exposed19 j)
    (hg3 : ∀ j, c < j → j < d → ¬ Exposed19 j)
    (hg4 : ∀ j, d < j → j < e → ¬ Exposed19 j) : e - a ≤ 38 := by
  have h := (window_facts ha hEa).2.2
  rw [List.countP_eq_length_filter] at h
  set p : ℕ → Bool := fun i =>
    atT (a % 5) (a % 7) (a % 11) (a % 13) (a % 17) (a % 19) (i+1) with hp
  have hnd : ((List.range 38).filter p).Nodup :=
    List.Nodup.filter _ List.nodup_range
  obtain ⟨w, x, y, z, rest, hl⟩ :
      ∃ w x y z rest, (List.range 38).filter p = w :: x :: y :: z :: rest := by
    rcases hlist : (List.range 38).filter p with
      _ | ⟨w, _ | ⟨x, _ | ⟨y, _ | ⟨z, rest⟩⟩⟩⟩
    · rw [hlist] at h; simp at h
    · rw [hlist] at h; simp at h
    · rw [hlist] at h; simp at h
    · rw [hlist] at h; simp at h
    · exact ⟨w, x, y, z, rest, rfl⟩
  -- pairwise distinct offsets, from Nodup of the filtered range
  rw [hl] at hnd
  obtain ⟨hwmem, hnd1⟩ := List.nodup_cons.mp hnd
  obtain ⟨hxmem, hnd2⟩ := List.nodup_cons.mp hnd1
  obtain ⟨hymem, _⟩ := List.nodup_cons.mp hnd2
  have hwx : w ≠ x := by intro h'; exact hwmem (by rw [h']; simp)
  have hwy : w ≠ y := by intro h'; exact hwmem (by rw [h']; simp)
  have hwz : w ≠ z := by intro h'; exact hwmem (by rw [h']; simp)
  have hxy : x ≠ y := by intro h'; exact hxmem (by rw [h']; simp)
  have hxz : x ≠ z := by intro h'; exact hxmem (by rw [h']; simp)
  have hyz : y ≠ z := by intro h'; exact hymem (by rw [h']; simp)
  -- the four witnesses: in-window and exposed
  have hw' := List.mem_filter.mp (show w ∈ (List.range 38).filter p by rw [hl]; simp)
  have hx' := List.mem_filter.mp (show x ∈ (List.range 38).filter p by rw [hl]; simp)
  have hy' := List.mem_filter.mp (show y ∈ (List.range 38).filter p by rw [hl]; simp)
  have hz' := List.mem_filter.mp (show z ∈ (List.range 38).filter p by rw [hl]; simp)
  have hw38 := List.mem_range.mp hw'.1
  have hx38 := List.mem_range.mp hx'.1
  have hy38 := List.mem_range.mp hy'.1
  have hz38 := List.mem_range.mp hz'.1
  have hEw : Exposed19 (a + (w+1)) := (atT_iff ha _).mp (by simpa using hw'.2)
  have hEx : Exposed19 (a + (x+1)) := (atT_iff ha _).mp (by simpa using hx'.2)
  have hEy : Exposed19 (a + (y+1)) := (atT_iff ha _).mp (by simpa using hy'.2)
  have hEz : Exposed19 (a + (z+1)) := (atT_iff ha _).mp (by simpa using hz'.2)
  -- every exposed point after `a` is one of b, c, d, or at least e
  have step : ∀ v, Exposed19 v → a < v → (v = b ∨ v = c ∨ v = d ∨ e ≤ v) := by
    intro v hEv hav
    have h1 : b ≤ v := by
      by_contra h'; exact hg1 v (by omega) (by omega) hEv
    rcases eq_or_lt_of_le h1 with h1' | h1'
    · exact Or.inl h1'.symm
    have h2 : c ≤ v := by
      by_contra h'; exact hg2 v (by omega) (by omega) hEv
    rcases eq_or_lt_of_le h2 with h2' | h2'
    · exact Or.inr (Or.inl h2'.symm)
    have h3 : d ≤ v := by
      by_contra h'; exact hg3 v (by omega) (by omega) hEv
    rcases eq_or_lt_of_le h3 with h3' | h3'
    · exact Or.inr (Or.inr (Or.inl h3'.symm))
    have h4 : e ≤ v := by
      by_contra h'; exact hg4 v (by omega) (by omega) hEv
    exact Or.inr (Or.inr (Or.inr h4))
  have hsw := step _ hEw (by omega)
  have hsx := step _ hEx (by omega)
  have hsy := step _ hEy (by omega)
  have hsz := step _ hEz (by omega)
  -- four distinct positions, only three slots below e: one is >= e, and all
  -- are <= a + 38
  omega

/-! ## The headline forms -/

/-- **The alpha1 = 4/3 certificate at machine 19**, denominators cleared:
`9 * F2 <= 9 * F + 4 * q'` with `F = 25`, `q' = 23` (279 <= 317). -/
theorem alpha1_certificate {a b c : ℕ} (ha : 1 ≤ a) (hab : a < b) (hbc : b < c)
    (hEa : Exposed19 a) (hEb : Exposed19 b) (hEc : Exposed19 c)
    (hg1 : ∀ j, a < j → j < b → ¬ Exposed19 j)
    (hg2 : ∀ j, b < j → j < c → ¬ Exposed19 j) :
    9 * (c - a) ≤ 9 * 25 + 4 * 23 := by
  have := pair_sum_le ha hab hbc hEa hEb hEc hg1 hg2
  omega

/-- **Lemma 1 at y = 19**: `F2 - F <= alpha1 * q'` with `alpha1 = 4/3`,
`q' = 23`, in cleared form `3 * (F2 - F) <= 4 * q'`. -/
theorem lemma1_at_19 {a b c : ℕ} (ha : 1 ≤ a) (hab : a < b) (hbc : b < c)
    (hEa : Exposed19 a) (hEb : Exposed19 b) (hEc : Exposed19 c)
    (hg1 : ∀ j, a < j → j < b → ¬ Exposed19 j)
    (hg2 : ∀ j, b < j → j < c → ¬ Exposed19 j) :
    3 * ((c - a) - 25) ≤ 4 * 23 := by
  have := pair_sum_le ha hab hbc hEa hEb hEc hg1 hg2
  omega

/-- **Shallow flatness at machine 19: `F_4 <= F + q'`** (38 <= 25 + 23).
The depth-4 window bound sits inside the tolerance budget - the first
kernel-checked instance of the flatness hypothesis of
`Spectrum.merged_le_of_shallow`, at the machine where the tier-C wall used
to stand. With `k_win <= 3` (empirical, census-checked elsewhere) this
discharges (D) at `alpha = 3` for machine 19's step. -/
theorem shallow_flatness {a b c d e : ℕ} (ha : 1 ≤ a)
    (hab : a < b) (hbc : b < c) (hcd : c < d) (hde : d < e)
    (hEa : Exposed19 a) (hEb : Exposed19 b) (hEc : Exposed19 c)
    (hEd : Exposed19 d) (hEe : Exposed19 e)
    (hg1 : ∀ j, a < j → j < b → ¬ Exposed19 j)
    (hg2 : ∀ j, b < j → j < c → ¬ Exposed19 j)
    (hg3 : ∀ j, c < j → j < d → ¬ Exposed19 j)
    (hg4 : ∀ j, d < j → j < e → ¬ Exposed19 j) : e - a ≤ 25 + 23 := by
  have := quad_sum_le ha hab hbc hcd hde hEa hEb hEc hEd hEe hg1 hg2 hg3 hg4
  omega

/-! ## The machine's own gap sequence, and the wired flatness instance -/

/-- Multiples of the period are openings, so an opening exists above any
point. -/
theorem exists_exposed_above (k : ℕ) : ∃ m, k < m ∧ Exposed19 m := by
  refine ⟨1616615 * (k + 1), by omega, ?_⟩
  rw [exposed19_iff (by omega)]
  have h5 : (1616615 * (k + 1)) % 5 = 0 := by omega
  have h7 : (1616615 * (k + 1)) % 7 = 0 := by omega
  have h11 : (1616615 * (k + 1)) % 11 = 0 := by omega
  have h13 : (1616615 * (k + 1)) % 13 = 0 := by omega
  have h17 : (1616615 * (k + 1)) % 17 = 0 := by omega
  have h19 : (1616615 * (k + 1)) % 19 = 0 := by omega
  rw [h5, h7, h11, h13, h17, h19]
  decide

/-- The next opening strictly after `k`. -/
def nextOp (k : ℕ) : ℕ := Nat.find (exists_exposed_above k)

theorem nextOp_gt (k : ℕ) : k < nextOp k :=
  (Nat.find_spec (exists_exposed_above k)).1

theorem nextOp_exposed (k : ℕ) : Exposed19 (nextOp k) :=
  (Nat.find_spec (exists_exposed_above k)).2

theorem nextOp_min {k m : ℕ} (h1 : k < m) (h2 : m < nextOp k) :
    ¬ Exposed19 m := fun hE =>
  Nat.find_min (exists_exposed_above k) h2 ⟨h1, hE⟩

/-- The opening sequence of machine 19: the openings in increasing order. -/
def opSeq : ℕ → ℕ
  | 0 => nextOp 0
  | n + 1 => nextOp (opSeq n)

theorem opSeq_exposed (n : ℕ) : Exposed19 (opSeq n) := by
  cases n <;> exact nextOp_exposed _

theorem opSeq_lt_succ (n : ℕ) : opSeq n < opSeq (n + 1) := nextOp_gt _

theorem opSeq_pos (n : ℕ) : 1 ≤ opSeq n := by
  cases n with
  | zero => exact nextOp_gt 0
  | succ m => have h1 := nextOp_gt (opSeq m); have h2 : opSeq (m+1) = nextOp (opSeq m) := rfl; omega

theorem opSeq_le_add (a j : ℕ) : opSeq a ≤ opSeq (a + j) := by
  induction j with
  | zero => rfl
  | succ j ih =>
    have h := opSeq_lt_succ (a + j)
    have he : a + (j + 1) = (a + j) + 1 := by omega
    rw [he]
    omega

/-- No opening sits strictly between consecutive members of `opSeq`. -/
theorem opSeq_gap_empty (n : ℕ) :
    ∀ j, opSeq n < j → j < opSeq (n + 1) → ¬ Exposed19 j :=
  fun _j h1 h2 => nextOp_min h1 h2

/-- **The gap word of machine 19**: `g19 n` is the length of the `n`-th
machine gap. -/
def g19 (n : ℕ) : ℕ := opSeq (n + 1) - opSeq n

/-- Window sums of the gap word telescope to opening differences. -/
theorem windowSum_g19 (a j : ℕ) :
    Spectrum.windowSum g19 a j = opSeq (a + j) - opSeq a := by
  induction j with
  | zero => simp [Spectrum.windowSum]
  | succ j ih =>
    have hs : Spectrum.windowSum g19 a (j + 1)
        = Spectrum.windowSum g19 a j + g19 (a + j) := Finset.sum_range_succ _ _
    have h1 := opSeq_le_add a j
    have h2 := opSeq_lt_succ (a + j)
    have he : a + (j + 1) = (a + j) + 1 := by omega
    rw [hs, ih, g19, he]
    omega

/-- **`SpectrumBound g19 4 38`, kernel-fed**: the depth-4 spectrum bound
over the machine's own gap sequence, from the period scan. -/
theorem spectrum_four : Spectrum.SpectrumBound g19 4 38 := by
  intro a
  rw [windowSum_g19]
  have e1 : a + 1 + 1 = a + 2 := by omega
  have e2 : a + 2 + 1 = a + 3 := by omega
  have e3 : a + 3 + 1 = a + 4 := by omega
  have h := quad_sum_le (opSeq_pos a)
    (opSeq_lt_succ a) (e1 ▸ opSeq_lt_succ (a + 1)) (e2 ▸ opSeq_lt_succ (a + 2))
    (e3 ▸ opSeq_lt_succ (a + 3))
    (opSeq_exposed a) (opSeq_exposed (a + 1)) (opSeq_exposed (a + 2))
    (opSeq_exposed (a + 3)) (opSeq_exposed (a + 4))
    (opSeq_gap_empty a) (e1 ▸ opSeq_gap_empty (a + 1)) (e2 ▸ opSeq_gap_empty (a + 2))
    (e3 ▸ opSeq_gap_empty (a + 3))
  exact h

/-- **Shallow flatness, wired**: `F_4 <= F + q'` over the machine's own gap
sequence (38 <= 48). -/
theorem spectrum_four_flat : Spectrum.SpectrumBound g19 4 (25 + 23) :=
  fun a => le_trans (spectrum_four a) (by norm_num)

/-- **(D) at alpha = 3 at machine 19, end to end.** For every shallow word
(`k_win <= 3`, so the merged window is at most 4 consecutive gaps), the
merged length is at most `F + q' = 48` - over the machine's real gap
sequence, with the flatness half discharged by the kernel scan. The ONLY
remaining hypothesis is the word's shallowness. -/
theorem D_of_shallow_word {a l : ℕ} (hl : l + 2 ≤ 4) :
    g19 a + Spectrum.windowSum g19 (a + 1) l + g19 (a + l + 1) ≤ 25 + 23 :=
  Spectrum.merged_le_of_shallow hl spectrum_four (by norm_num)

end Machine19
