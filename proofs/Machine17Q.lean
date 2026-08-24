/-
Machine 17: the qualifying spectrum closed at every depth - the `17 -> 19`
rung of the (D) ladder (round 22).

Machine 17's certificate (round 15) gave `F_1 = 18` and `F_2 = 25`. This
file adds the machine's own opening enumeration (`opSeq`, `g17`), the three
deeper window bounds `F_3 <= 28`, `F_4 <= 33`, `F_5 <= 35`, the qualifying
depth-6 bound `Q_6 <= 34` (the first depth where the unconditional bound
`F_6 = 40` breaks the budget and the qualifying restriction is doing real
work), and the depth refutation

    no five consecutive gaps are all `>= 6`,

which closes the qualifying spectrum at the floor `2u' = 6` of gear 19:
`Q_j(17; 6) <= 35` for EVERY depth `j >= 3`. With `F_2 = 25` this is
Constructor's R39 criterion at the step, budget `F(17) + 19 = 37`:

    max (F_2, max_j Q_j) = 35 <= 37.

`Ladder.lean` instantiates `MergeLaw.newgap_le_step` on it and gets (D) at
`alpha = 3` for the 17->19 step with no hypotheses.

All facts verified over the full period numerically first (scratchpad
ladder_verify.py / predsim2.py, all 85085 residues, zero failures).
-/

import Machine17QS0
import Machine17QS1
import Machine17QS2
import Spectrum

namespace Machine17

/-! ## Assembly: the whole period -/

/-- **One period, all 17 slices of the chain scan.** -/
theorem qsliceAll : ∀ e < 17, qslice e = true := by
  intro e he
  rcases Nat.lt_or_ge e 6 with h | h
  · exact qasm0 e (by omega) (by omega)
  rcases Nat.lt_or_ge e 12 with h2 | h2
  · exact qasm1 e (by omega) (by omega)
  · exact qasm2 e (by omega) (by omega)

/-- The tuple-level fact, unpacked from the slice. -/
theorem qokAll {a b c d e : ℕ} (ha : a < 5) (hb : b < 7) (hc : c < 11)
    (hd : d < 13) (he : e < 17) : qokT a b c d e = true := by
  have h := qsliceAll e he
  rw [qslice, List.all_eq_true] at h
  have h1 := h a (List.mem_range.mpr ha)
  rw [List.all_eq_true] at h1
  have h2 := h1 b (List.mem_range.mpr hb)
  rw [List.all_eq_true] at h2
  have h3 := h2 c (List.mem_range.mpr hc)
  rw [List.all_eq_true] at h3
  exact h3 d (List.mem_range.mpr hd)

/-! ## The machine's own gap sequence -/

/-- Multiples of the period 85085 are openings, so an opening exists above
any point. -/
theorem exists_exposed_above (k : ℕ) : ∃ m, k < m ∧ Exposed17 m := by
  refine ⟨85085 * (k + 1), by omega, ?_⟩
  rw [exposed17_iff (by omega)]
  have h5 : (85085 * (k + 1)) % 5 = 0 := by omega
  have h7 : (85085 * (k + 1)) % 7 = 0 := by omega
  have h11 : (85085 * (k + 1)) % 11 = 0 := by omega
  have h13 : (85085 * (k + 1)) % 13 = 0 := by omega
  have h17 : (85085 * (k + 1)) % 17 = 0 := by omega
  rw [h5, h7, h11, h13, h17]
  decide

/-- The next opening strictly after `k`. -/
def nextOp (k : ℕ) : ℕ := Nat.find (exists_exposed_above k)

theorem nextOp_gt (k : ℕ) : k < nextOp k :=
  (Nat.find_spec (exists_exposed_above k)).1

theorem nextOp_exposed (k : ℕ) : Exposed17 (nextOp k) :=
  (Nat.find_spec (exists_exposed_above k)).2

theorem nextOp_min {k m : ℕ} (h1 : k < m) (h2 : m < nextOp k) :
    ¬ Exposed17 m := fun hE =>
  Nat.find_min (exists_exposed_above k) h2 ⟨h1, hE⟩

/-- The opening sequence of machine 17, in increasing order. -/
def opSeq : ℕ → ℕ
  | 0 => nextOp 0
  | n + 1 => nextOp (opSeq n)

theorem opSeq_succ (n : ℕ) : opSeq (n + 1) = nextOp (opSeq n) := rfl

theorem opSeq_exposed (n : ℕ) : Exposed17 (opSeq n) := by
  cases n <;> exact nextOp_exposed _

theorem opSeq_lt_succ (n : ℕ) : opSeq n < opSeq (n + 1) := nextOp_gt _

theorem opSeq_pos (n : ℕ) : 1 ≤ opSeq n := by
  cases n with
  | zero => exact nextOp_gt 0
  | succ m =>
    have h1 := nextOp_gt (opSeq m)
    have h2 : opSeq (m + 1) = nextOp (opSeq m) := rfl
    omega

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
    ∀ j, opSeq n < j → j < opSeq (n + 1) → ¬ Exposed17 j :=
  fun _j h1 h2 => nextOp_min h1 h2

/-- **The gap word of machine 17.** -/
def g17 (n : ℕ) : ℕ := opSeq (n + 1) - opSeq n

/-- Window sums of the gap word telescope to opening differences. -/
theorem windowSum_g17 (a j : ℕ) :
    Spectrum.windowSum g17 a j = opSeq (a + j) - opSeq a := by
  induction j with
  | zero => simp [Spectrum.windowSum]
  | succ j ih =>
    have hs : Spectrum.windowSum g17 a (j + 1)
        = Spectrum.windowSum g17 a j + g17 (a + j) := Finset.sum_range_succ _ _
    have h1 := opSeq_le_add a j
    have h2 := opSeq_lt_succ (a + j)
    have he : a + (j + 1) = (a + j) + 1 := by omega
    rw [hs, ih, g17, he]
    omega

/-! ## The seek walk, related to `nextOp` -/

theorem seekT_succ_pos {a b c d e fu s : ℕ} (h : atT a b c d e (s + 1) = true) :
    seekT a b c d e (fu + 1) s = s + 1 := by
  simp only [seekT]
  split
  · rfl
  · rename_i hneg
    exact absurd h hneg

theorem seekT_succ_neg {a b c d e fu s : ℕ} (h : ¬ atT a b c d e (s + 1) = true) :
    seekT a b c d e (fu + 1) s = seekT a b c d e fu (s + 1) := by
  simp only [seekT]
  split
  · rename_i hpos
    exact absurd hpos h
  · rfl

theorem seekT_gt (a b c d e : ℕ) : ∀ fu s, s < seekT a b c d e fu s := by
  intro fu
  induction fu with
  | zero => intro s; simp only [seekT]; omega
  | succ fu ih =>
    intro s
    by_cases h : atT a b c d e (s + 1) = true
    · rw [seekT_succ_pos h]; omega
    · rw [seekT_succ_neg h]
      have := ih (s + 1); omega

theorem seekT_found (a b c d e : ℕ) :
    ∀ fu s t, s < t → t ≤ s + fu → atT a b c d e t = true →
      seekT a b c d e fu s ≤ s + fu := by
  intro fu
  induction fu with
  | zero => intro s t h1 h2 _; omega
  | succ fu ih =>
    intro s t h1 h2 hat
    by_cases h : atT a b c d e (s + 1) = true
    · rw [seekT_succ_pos h]; omega
    · rw [seekT_succ_neg h]
      have hne : t ≠ s + 1 := by intro he; rw [he] at hat; exact h hat
      have := ih (s + 1) t (by omega) (by omega) hat
      omega

theorem seekT_exposed (a b c d e : ℕ) :
    ∀ fu s, seekT a b c d e fu s ≤ s + fu →
      atT a b c d e (seekT a b c d e fu s) = true := by
  intro fu
  induction fu with
  | zero => intro s h; simp only [seekT] at h; omega
  | succ fu ih =>
    intro s h
    by_cases hat : atT a b c d e (s + 1) = true
    · rw [seekT_succ_pos hat]; exact hat
    · rw [seekT_succ_neg hat] at h ⊢
      exact ih (s + 1) (by omega)

theorem seekT_min (a b c d e : ℕ) :
    ∀ fu s t, s < t → t < seekT a b c d e fu s → t ≤ s + fu →
      atT a b c d e t = false := by
  intro fu
  induction fu with
  | zero => intro s t h1 _ h3; omega
  | succ fu ih =>
    intro s t h1 h2 h3
    by_cases hat : atT a b c d e (s + 1) = true
    · rw [seekT_succ_pos hat] at h2; omega
    · rw [seekT_succ_neg hat] at h2
      rcases Nat.lt_or_ge t (s + 2) with hlt | hge
      · have he : t = s + 1 := by omega
        subst he
        simpa using hat
      · exact ih (s + 1) t (by omega) h2 (by omega)

/-! ## `F_1(17) <= 18` from the walk, and the walk is `nextOp` -/

/-- The `o1` check of the scan, at an opening. -/
theorem seek_one_le {x : ℕ} (hx : 1 ≤ x) (hE : Exposed17 x) :
    seekT (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) 18 0 ≤ 18 := by
  have ha0 : atT (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) 0 = true :=
    (atT_iff hx 0).mpr (by simpa using hE)
  have h := qokAll (a := x % 5) (b := x % 7) (c := x % 11) (d := x % 13)
    (e := x % 17) (by omega) (by omega) (by omega) (by omega) (by omega)
  rw [qokT, ha0] at h
  simp only [Bool.not_true, Bool.false_or, chainT, Bool.and_eq_true,
    Nat.ble_eq] at h
  exact h.1

/-- **`F_1(17) <= 18`** over the enumeration. -/
theorem nextOp_le_18 {x : ℕ} (hx : 1 ≤ x) (hE : Exposed17 x) :
    nextOp x ≤ x + 18 := by
  have h1 := seek_one_le hx hE
  have h2 := seekT_exposed (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) 18 0
    (by omega)
  have h3 : Exposed17 (x + seekT (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) 18 0) :=
    (atT_iff hx _).mp h2
  have h4 := seekT_gt (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) 18 0
  have := Nat.find_min' (exists_exposed_above x)
    (show x < x + seekT (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) 18 0 ∧
      Exposed17 (x + seekT (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) 18 0) from
      ⟨by omega, h3⟩)
  simp only [nextOp]
  omega

/-- **The seek walk computes `nextOp`.** -/
theorem seek_next {x s : ℕ} (hx : 1 ≤ x) (hE : Exposed17 (x + s)) :
    x + seekT (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) 18 s = nextOp (x + s) := by
  have hE1 : 1 ≤ x + s := by omega
  have hnle : nextOp (x + s) ≤ x + s + 18 := nextOp_le_18 hE1 hE
  have hngt : x + s < nextOp (x + s) := nextOp_gt _
  have hnE : Exposed17 (nextOp (x + s)) := nextOp_exposed _
  have hat : atT (x % 5) (x % 7) (x % 11) (x % 13) (x % 17)
      (nextOp (x + s) - x) = true := by
    apply (atT_iff hx _).mpr
    rwa [show x + (nextOp (x + s) - x) = nextOp (x + s) by omega]
  have hfound := seekT_found (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) 18 s
    (nextOp (x + s) - x) (by omega) (by omega) hat
  have hσat := seekT_exposed (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) 18 s hfound
  have hσE : Exposed17
      (x + seekT (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) 18 s) :=
    (atT_iff hx _).mp hσat
  have hσgt := seekT_gt (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) 18 s
  have hle1 : nextOp (x + s)
      ≤ x + seekT (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) 18 s :=
    Nat.find_min' (exists_exposed_above (x + s)) ⟨by omega, hσE⟩
  rcases eq_or_lt_of_le hle1 with he | hlt
  · omega
  · exfalso
    have hmin := seekT_min (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) 18 s
      (nextOp (x + s) - x) (by omega) (by omega) (by omega)
    rw [hmin] at hat
    exact Bool.noConfusion hat

/-! ## The chain facts -/

/-- **The six scan facts, over the enumeration**: `F_1 <= 18`, `F_3 <= 28`,
`F_4 <= 33`, `F_5 <= 35`, the qualifying depth-6 bound `Q_6 <= 34`, and no
five consecutive gaps all at or above the floor 6. -/
theorem chain_facts (n : ℕ) :
    opSeq (n + 1) - opSeq n ≤ 18 ∧ opSeq (n + 3) - opSeq n ≤ 28 ∧
      opSeq (n + 4) - opSeq n ≤ 33 ∧ opSeq (n + 5) - opSeq n ≤ 35 ∧
      ((6 ≤ g17 (n + 1) ∧ 6 ≤ g17 (n + 2) ∧ 6 ≤ g17 (n + 3) ∧
          6 ≤ g17 (n + 4)) → opSeq (n + 6) - opSeq n ≤ 34) ∧
      ¬ (6 ≤ g17 n ∧ 6 ≤ g17 (n + 1) ∧ 6 ≤ g17 (n + 2) ∧ 6 ≤ g17 (n + 3) ∧
          6 ≤ g17 (n + 4)) := by
  have hx : 1 ≤ opSeq n := opSeq_pos n
  have hE : Exposed17 (opSeq n) := opSeq_exposed n
  have ha0 : atT (opSeq n % 5) (opSeq n % 7) (opSeq n % 11) (opSeq n % 13)
      (opSeq n % 17) 0 = true := (atT_iff hx 0).mpr (by simpa using hE)
  have h := qokAll (a := opSeq n % 5) (b := opSeq n % 7) (c := opSeq n % 11)
    (d := opSeq n % 13) (e := opSeq n % 17)
    (by omega) (by omega) (by omega) (by omega) (by omega)
  rw [qokT, ha0] at h
  simp only [Bool.not_true, Bool.false_or] at h
  simp only [chainT] at h
  set o1 := seekT (opSeq n % 5) (opSeq n % 7) (opSeq n % 11) (opSeq n % 13)
    (opSeq n % 17) 18 0 with ho1
  set o2 := seekT (opSeq n % 5) (opSeq n % 7) (opSeq n % 11) (opSeq n % 13)
    (opSeq n % 17) 18 o1 with ho2
  set o3 := seekT (opSeq n % 5) (opSeq n % 7) (opSeq n % 11) (opSeq n % 13)
    (opSeq n % 17) 18 o2 with ho3
  set o4 := seekT (opSeq n % 5) (opSeq n % 7) (opSeq n % 11) (opSeq n % 13)
    (opSeq n % 17) 18 o3 with ho4
  set o5 := seekT (opSeq n % 5) (opSeq n % 7) (opSeq n % 11) (opSeq n % 13)
    (opSeq n % 17) 18 o4 with ho5
  set o6 := seekT (opSeq n % 5) (opSeq n % 7) (opSeq n % 11) (opSeq n % 13)
    (opSeq n % 17) 18 o5 with ho6
  simp only [Bool.and_eq_true, Nat.ble_eq, Bool.or_eq_true,
    Bool.not_eq_true'] at h
  obtain ⟨h18, h28, h33, h35, hq6, hrun⟩ := h
  have hE0 : Exposed17 (opSeq n + 0) := by simpa using hE
  have e1 : opSeq n + o1 = opSeq (n + 1) := by
    rw [opSeq_succ]
    have h1 := seek_next hx hE0
    simpa [← ho1] using h1
  have hEo1 : Exposed17 (opSeq n + o1) := by rw [e1]; exact opSeq_exposed _
  have e2 : opSeq n + o2 = opSeq (n + 2) := by
    rw [show n + 2 = (n + 1) + 1 by omega, opSeq_succ]
    have h2 := seek_next hx hEo1
    rw [e1] at h2
    simpa [← ho2] using h2
  have hEo2 : Exposed17 (opSeq n + o2) := by rw [e2]; exact opSeq_exposed _
  have e3 : opSeq n + o3 = opSeq (n + 3) := by
    rw [show n + 3 = (n + 2) + 1 by omega, opSeq_succ]
    have h3 := seek_next hx hEo2
    rw [e2] at h3
    simpa [← ho3] using h3
  have hEo3 : Exposed17 (opSeq n + o3) := by rw [e3]; exact opSeq_exposed _
  have e4 : opSeq n + o4 = opSeq (n + 4) := by
    rw [show n + 4 = (n + 3) + 1 by omega, opSeq_succ]
    have h4 := seek_next hx hEo3
    rw [e3] at h4
    simpa [← ho4] using h4
  have hEo4 : Exposed17 (opSeq n + o4) := by rw [e4]; exact opSeq_exposed _
  have e5 : opSeq n + o5 = opSeq (n + 5) := by
    rw [show n + 5 = (n + 4) + 1 by omega, opSeq_succ]
    have h5 := seek_next hx hEo4
    rw [e4] at h5
    simpa [← ho5] using h5
  have hEo5 : Exposed17 (opSeq n + o5) := by rw [e5]; exact opSeq_exposed _
  have e6 : opSeq n + o6 = opSeq (n + 6) := by
    rw [show n + 6 = (n + 5) + 1 by omega, opSeq_succ]
    have h6 := seek_next hx hEo5
    rw [e5] at h6
    simpa [← ho6] using h6
  have g0 : g17 n = opSeq (n + 1) - opSeq n := rfl
  have g1 : g17 (n + 1) = opSeq (n + 2) - opSeq (n + 1) := by simp only [g17]
  have g2 : g17 (n + 2) = opSeq (n + 3) - opSeq (n + 2) := by simp only [g17]
  have g3 : g17 (n + 3) = opSeq (n + 4) - opSeq (n + 3) := by simp only [g17]
  have g4 : g17 (n + 4) = opSeq (n + 5) - opSeq (n + 4) := by simp only [g17]
  refine ⟨by omega, by omega, by omega, by omega, ?_, ?_⟩
  · rintro ⟨c1, c2, c3, c4⟩
    have hbt : (Nat.ble 6 (o2 - o1) && Nat.ble 6 (o3 - o2) &&
        Nat.ble 6 (o4 - o3) && Nat.ble 6 (o5 - o4)) = true := by
      simp only [Bool.and_eq_true, Nat.ble_eq]
      refine ⟨⟨⟨?_, ?_⟩, ?_⟩, ?_⟩ <;> omega
    rcases hq6 with hfalse | hle
    · rw [hbt] at hfalse; exact Bool.noConfusion hfalse
    · omega
  · rintro ⟨c0, c1, c2, c3, c4⟩
    have hbt : (Nat.ble 6 o1 && Nat.ble 6 (o2 - o1) && Nat.ble 6 (o3 - o2) &&
        Nat.ble 6 (o4 - o3) && Nat.ble 6 (o5 - o4)) = true := by
      simp only [Bool.and_eq_true, Nat.ble_eq]
      refine ⟨⟨⟨⟨?_, ?_⟩, ?_⟩, ?_⟩, ?_⟩ <;> omega
    rw [hbt] at hrun
    exact Bool.noConfusion hrun

/-- **`Q_j(17; 6) = 0` for `j >= 7`**: no five consecutive gaps of `g17` all
meet the qualifying floor 6. -/
theorem no_big_run (n : ℕ) :
    ¬ (6 ≤ g17 n ∧ 6 ≤ g17 (n + 1) ∧ 6 ≤ g17 (n + 2) ∧ 6 ≤ g17 (n + 3) ∧
        6 ≤ g17 (n + 4)) :=
  (chain_facts n).2.2.2.2.2

/-! ## The spectrum ladder over the gap word -/

/-- `F_1(17) <= 18`. -/
theorem spectrum_one : Spectrum.SpectrumBound g17 1 18 := by
  intro a; rw [windowSum_g17]; exact (chain_facts a).1

/-- `F_2(17) <= 25` - the round-15 certificate, wired to `g17`. -/
theorem spectrum_two : Spectrum.SpectrumBound g17 2 25 := by
  intro a
  rw [windowSum_g17]
  have e1 : a + 1 + 1 = a + 2 := by omega
  exact pair_sum_le (opSeq_pos a) (opSeq_lt_succ a) (e1 ▸ opSeq_lt_succ (a + 1))
    (opSeq_exposed a) (opSeq_exposed (a + 1)) (opSeq_exposed (a + 2))
    (opSeq_gap_empty a) (e1 ▸ opSeq_gap_empty (a + 1))

/-- `F_3(17) <= 28`. -/
theorem spectrum_three : Spectrum.SpectrumBound g17 3 28 := by
  intro a; rw [windowSum_g17]; exact (chain_facts a).2.1

/-- `F_4(17) <= 33`. -/
theorem spectrum_four : Spectrum.SpectrumBound g17 4 33 := by
  intro a; rw [windowSum_g17]; exact (chain_facts a).2.2.1

/-- `F_5(17) <= 35`. -/
theorem spectrum_five : Spectrum.SpectrumBound g17 5 35 := by
  intro a; rw [windowSum_g17]; exact (chain_facts a).2.2.2.1

/-- **The kernel-fed spectrum ladder of machine 17**: `F_1..F_5 <= 18, 25,
28, 33, 35`. -/
theorem spectrum_ladder :
    Spectrum.SpectrumBound g17 1 18 ∧ Spectrum.SpectrumBound g17 2 25 ∧
      Spectrum.SpectrumBound g17 3 28 ∧ Spectrum.SpectrumBound g17 4 33 ∧
      Spectrum.SpectrumBound g17 5 35 :=
  ⟨spectrum_one, spectrum_two, spectrum_three, spectrum_four, spectrum_five⟩

/-! ## The qualifying spectrum, closed at every depth -/

/-- **`Q_j(17; 6) <= 35` for every depth `j >= 3`** (floor `2u' = 6`, gear
19): depths 3, 4, 5 from the unconditional ladder, depth 6 from the
qualifying scan fact (`F_6 = 40` breaks the budget, `Q_6 = 34` does not -
this is the depth where the qualifying restriction earns its keep), and NO
qualifying window of depth 7 or more exists at all (`no_big_run`). -/
theorem qual_bound_all : ∀ j, 3 ≤ j → Spectrum.QualBound g17 3 j 35 := by
  intro j hj a hq
  rcases Nat.lt_or_ge j 7 with hj7 | hj7
  · interval_cases j
    · exact le_trans (spectrum_three a) (by omega)
    · exact le_trans (spectrum_four a) (by omega)
    · exact spectrum_five a
    · rw [windowSum_g17]
      refine le_trans ((chain_facts a).2.2.2.2.1 ⟨?_, ?_, ?_, ?_⟩) (by omega)
      · have h1 := hq 1 (by omega) (by omega); omega
      · have h2 := hq 2 (by omega) (by omega)
        rw [show a + 2 = a + 1 + 1 by omega] at h2; omega
      · have h3 := hq 3 (by omega) (by omega)
        rw [show a + 3 = a + 1 + 2 by omega] at h3; omega
      · have h4 := hq 4 (by omega) (by omega)
        rw [show a + 4 = a + 1 + 3 by omega] at h4; omega
  · exfalso
    refine no_big_run (a + 1) ⟨?_, ?_, ?_, ?_, ?_⟩
    · have h1 := hq 1 (by omega) (by omega); omega
    · have h2 := hq 2 (by omega) (by omega)
      rw [show a + 1 + 1 = a + 2 by omega]; omega
    · have h3 := hq 3 (by omega) (by omega)
      rw [show a + 1 + 2 = a + 3 by omega]; omega
    · have h4 := hq 4 (by omega) (by omega)
      rw [show a + 1 + 3 = a + 4 by omega]; omega
    · have h5 := hq 5 (by omega) (by omega)
      rw [show a + 1 + 4 = a + 5 by omega]; omega

/-! ## The opening enumeration is complete -/

theorem opSeq_strict_mono {a b : ℕ} (h : a < b) : opSeq a < opSeq b := by
  have h1 := opSeq_lt_succ a
  have h2 := opSeq_le_add (a + 1) (b - (a + 1))
  rw [show a + 1 + (b - (a + 1)) = b by omega] at h2
  omega

theorem opSeq_reach : ∀ dd A, 1 ≤ A → Exposed17 A → (∃ n, opSeq n = A) →
    ∀ B, Exposed17 B → A < B → B - A ≤ dd → ∃ m, opSeq m = B := by
  intro dd
  induction dd with
  | zero => intro A _ _ _ B _ hAB hd; omega
  | succ dd ih =>
    rintro A hA1 hEA ⟨n, hn⟩ B hEB hAB hd
    have hnext_le : nextOp A ≤ B :=
      Nat.find_min' (exists_exposed_above A) ⟨hAB, hEB⟩
    have hgt := nextOp_gt A
    rcases eq_or_lt_of_le hnext_le with he | hlt
    · exact ⟨n + 1, by rw [opSeq_succ, hn, he]⟩
    · exact ih (nextOp A) (by omega) (nextOp_exposed A)
        ⟨n + 1, by rw [opSeq_succ, hn]⟩ B hEB hlt (by omega)

/-- **Every opening is enumerated**: `opSeq` is onto the openings. -/
theorem opSeq_surj {m : ℕ} (hm : 1 ≤ m) (hE : Exposed17 m) :
    ∃ n, opSeq n = m := by
  have h0 : opSeq 0 = nextOp 0 := rfl
  have hle : nextOp 0 ≤ m := Nat.find_min' (exists_exposed_above 0) ⟨by omega, hE⟩
  rcases eq_or_lt_of_le hle with he | hlt
  · exact ⟨0, by rw [h0, he]⟩
  · exact opSeq_reach (m - nextOp 0) (nextOp 0) (by have := nextOp_gt 0; omega)
      (nextOp_exposed 0) ⟨0, rfl⟩ m hE hlt (by omega)

end Machine17
