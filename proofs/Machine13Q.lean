/-
Machine 13: the qualifying spectrum closed at every depth - the `13 -> 17`
rung of the (D) ladder (round 22).

Machine 13's certificate (round 11) gave `F_1 = 11` and `F_2 = 16`. This
file adds the machine's own opening enumeration (`opSeq`, `g13`), the two
deeper window bounds `F_3 <= 23`, `F_4 <= 26`, and the depth refutation

    no three consecutive gaps are all `>= 6`,

which closes the qualifying spectrum at the floor `2u' = 6` of gear 17:
`Q_j(13; 6) <= 26` for EVERY depth `j >= 3`. With `F_2 = 16` this is
Constructor's R39 criterion at the step, budget `F(13) + 17 = 28`:

    max (F_2, max_j Q_j) = 26 <= 28.

`Ladder.lean` instantiates `MergeLaw.newgap_le_step` on it and gets (D) at
`alpha = 3` for the 13->17 step with no hypotheses.

All facts verified over the full period numerically first (scratchpad
ladder_verify.py / predsim2.py, all 5005 residues, zero failures).
-/

import Machine13QS
import Spectrum

namespace Machine13

/-! ## Unpacking the scan -/

/-- The tuple-level fact, unpacked from the period check. -/
theorem qokAll {a b c d : ℕ} (ha : a < 5) (hb : b < 7) (hc : c < 11)
    (hd : d < 13) : qokT a b c d = true := by
  have h := qasm
  rw [qslice, List.all_eq_true] at h
  have h1 := h a (List.mem_range.mpr ha)
  rw [List.all_eq_true] at h1
  have h2 := h1 b (List.mem_range.mpr hb)
  rw [List.all_eq_true] at h2
  have h3 := h2 c (List.mem_range.mpr hc)
  rw [List.all_eq_true] at h3
  exact h3 d (List.mem_range.mpr hd)

/-! ## The machine's own gap sequence -/

/-- Multiples of the period 5005 are openings, so an opening exists above
any point. -/
theorem exists_exposed_above (k : ℕ) : ∃ m, k < m ∧ Exposed13 m := by
  refine ⟨5005 * (k + 1), by omega, ?_⟩
  rw [exposed13_iff (by omega)]
  have h5 : (5005 * (k + 1)) % 5 = 0 := by omega
  have h7 : (5005 * (k + 1)) % 7 = 0 := by omega
  have h11 : (5005 * (k + 1)) % 11 = 0 := by omega
  have h13 : (5005 * (k + 1)) % 13 = 0 := by omega
  rw [h5, h7, h11, h13]
  decide

/-- The next opening strictly after `k`. -/
def nextOp (k : ℕ) : ℕ := Nat.find (exists_exposed_above k)

theorem nextOp_gt (k : ℕ) : k < nextOp k :=
  (Nat.find_spec (exists_exposed_above k)).1

theorem nextOp_exposed (k : ℕ) : Exposed13 (nextOp k) :=
  (Nat.find_spec (exists_exposed_above k)).2

theorem nextOp_min {k m : ℕ} (h1 : k < m) (h2 : m < nextOp k) :
    ¬ Exposed13 m := fun hE =>
  Nat.find_min (exists_exposed_above k) h2 ⟨h1, hE⟩

/-- The opening sequence of machine 13, in increasing order. -/
def opSeq : ℕ → ℕ
  | 0 => nextOp 0
  | n + 1 => nextOp (opSeq n)

theorem opSeq_succ (n : ℕ) : opSeq (n + 1) = nextOp (opSeq n) := rfl

theorem opSeq_exposed (n : ℕ) : Exposed13 (opSeq n) := by
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
    ∀ j, opSeq n < j → j < opSeq (n + 1) → ¬ Exposed13 j :=
  fun _j h1 h2 => nextOp_min h1 h2

/-- **The gap word of machine 13.** -/
def g13 (n : ℕ) : ℕ := opSeq (n + 1) - opSeq n

/-- Window sums of the gap word telescope to opening differences. -/
theorem windowSum_g13 (a j : ℕ) :
    Spectrum.windowSum g13 a j = opSeq (a + j) - opSeq a := by
  induction j with
  | zero => simp [Spectrum.windowSum]
  | succ j ih =>
    have hs : Spectrum.windowSum g13 a (j + 1)
        = Spectrum.windowSum g13 a j + g13 (a + j) := Finset.sum_range_succ _ _
    have h1 := opSeq_le_add a j
    have h2 := opSeq_lt_succ (a + j)
    have he : a + (j + 1) = (a + j) + 1 := by omega
    rw [hs, ih, g13, he]
    omega

/-! ## The seek walk, related to `nextOp` -/

theorem seekT_succ_pos {a b c d fu s : ℕ} (h : atT a b c d (s + 1) = true) :
    seekT a b c d (fu + 1) s = s + 1 := by
  simp only [seekT]
  split
  · rfl
  · rename_i hneg
    exact absurd h hneg

theorem seekT_succ_neg {a b c d fu s : ℕ} (h : ¬ atT a b c d (s + 1) = true) :
    seekT a b c d (fu + 1) s = seekT a b c d fu (s + 1) := by
  simp only [seekT]
  split
  · rename_i hpos
    exact absurd hpos h
  · rfl

theorem seekT_gt (a b c d : ℕ) : ∀ fu s, s < seekT a b c d fu s := by
  intro fu
  induction fu with
  | zero => intro s; simp only [seekT]; omega
  | succ fu ih =>
    intro s
    by_cases h : atT a b c d (s + 1) = true
    · rw [seekT_succ_pos h]; omega
    · rw [seekT_succ_neg h]
      have := ih (s + 1); omega

theorem seekT_found (a b c d : ℕ) :
    ∀ fu s t, s < t → t ≤ s + fu → atT a b c d t = true →
      seekT a b c d fu s ≤ s + fu := by
  intro fu
  induction fu with
  | zero => intro s t h1 h2 _; omega
  | succ fu ih =>
    intro s t h1 h2 hat
    by_cases h : atT a b c d (s + 1) = true
    · rw [seekT_succ_pos h]; omega
    · rw [seekT_succ_neg h]
      have hne : t ≠ s + 1 := by intro he; rw [he] at hat; exact h hat
      have := ih (s + 1) t (by omega) (by omega) hat
      omega

theorem seekT_exposed (a b c d : ℕ) :
    ∀ fu s, seekT a b c d fu s ≤ s + fu →
      atT a b c d (seekT a b c d fu s) = true := by
  intro fu
  induction fu with
  | zero => intro s h; simp only [seekT] at h; omega
  | succ fu ih =>
    intro s h
    by_cases hat : atT a b c d (s + 1) = true
    · rw [seekT_succ_pos hat]; exact hat
    · rw [seekT_succ_neg hat] at h ⊢
      exact ih (s + 1) (by omega)

theorem seekT_min (a b c d : ℕ) :
    ∀ fu s t, s < t → t < seekT a b c d fu s → t ≤ s + fu →
      atT a b c d t = false := by
  intro fu
  induction fu with
  | zero => intro s t h1 _ h3; omega
  | succ fu ih =>
    intro s t h1 h2 h3
    by_cases hat : atT a b c d (s + 1) = true
    · rw [seekT_succ_pos hat] at h2; omega
    · rw [seekT_succ_neg hat] at h2
      rcases Nat.lt_or_ge t (s + 2) with hlt | hge
      · have he : t = s + 1 := by omega
        subst he
        simpa using hat
      · exact ih (s + 1) t (by omega) h2 (by omega)

/-! ## `F_1(13) <= 11` from the walk, and the walk is `nextOp` -/

/-- The `o1` check of the scan, at an opening. -/
theorem seek_one_le {x : ℕ} (hx : 1 ≤ x) (hE : Exposed13 x) :
    seekT (x % 5) (x % 7) (x % 11) (x % 13) 11 0 ≤ 11 := by
  have ha0 : atT (x % 5) (x % 7) (x % 11) (x % 13) 0 = true :=
    (atT_iff hx 0).mpr (by simpa using hE)
  have h := qokAll (a := x % 5) (b := x % 7) (c := x % 11) (d := x % 13)
    (by omega) (by omega) (by omega) (by omega)
  rw [qokT, ha0] at h
  simp only [Bool.not_true, Bool.false_or, chainT, Bool.and_eq_true,
    Nat.ble_eq] at h
  exact h.1

/-- **`F_1(13) <= 11`** over the enumeration. -/
theorem nextOp_le_11 {x : ℕ} (hx : 1 ≤ x) (hE : Exposed13 x) :
    nextOp x ≤ x + 11 := by
  have h1 := seek_one_le hx hE
  have h2 := seekT_exposed (x % 5) (x % 7) (x % 11) (x % 13) 11 0 (by omega)
  have h3 : Exposed13 (x + seekT (x % 5) (x % 7) (x % 11) (x % 13) 11 0) :=
    (atT_iff hx _).mp h2
  have h4 := seekT_gt (x % 5) (x % 7) (x % 11) (x % 13) 11 0
  have := Nat.find_min' (exists_exposed_above x)
    (show x < x + seekT (x % 5) (x % 7) (x % 11) (x % 13) 11 0 ∧
      Exposed13 (x + seekT (x % 5) (x % 7) (x % 11) (x % 13) 11 0) from
      ⟨by omega, h3⟩)
  simp only [nextOp]
  omega

/-- **The seek walk computes `nextOp`.** -/
theorem seek_next {x s : ℕ} (hx : 1 ≤ x) (hE : Exposed13 (x + s)) :
    x + seekT (x % 5) (x % 7) (x % 11) (x % 13) 11 s = nextOp (x + s) := by
  have hE1 : 1 ≤ x + s := by omega
  have hnle : nextOp (x + s) ≤ x + s + 11 := nextOp_le_11 hE1 hE
  have hngt : x + s < nextOp (x + s) := nextOp_gt _
  have hnE : Exposed13 (nextOp (x + s)) := nextOp_exposed _
  have hat : atT (x % 5) (x % 7) (x % 11) (x % 13) (nextOp (x + s) - x) = true := by
    apply (atT_iff hx _).mpr
    rwa [show x + (nextOp (x + s) - x) = nextOp (x + s) by omega]
  have hfound := seekT_found (x % 5) (x % 7) (x % 11) (x % 13) 11 s
    (nextOp (x + s) - x) (by omega) (by omega) hat
  have hσat := seekT_exposed (x % 5) (x % 7) (x % 11) (x % 13) 11 s hfound
  have hσE : Exposed13 (x + seekT (x % 5) (x % 7) (x % 11) (x % 13) 11 s) :=
    (atT_iff hx _).mp hσat
  have hσgt := seekT_gt (x % 5) (x % 7) (x % 11) (x % 13) 11 s
  have hle1 : nextOp (x + s)
      ≤ x + seekT (x % 5) (x % 7) (x % 11) (x % 13) 11 s :=
    Nat.find_min' (exists_exposed_above (x + s)) ⟨by omega, hσE⟩
  rcases eq_or_lt_of_le hle1 with he | hlt
  · omega
  · exfalso
    have hmin := seekT_min (x % 5) (x % 7) (x % 11) (x % 13) 11 s
      (nextOp (x + s) - x) (by omega) (by omega) (by omega)
    rw [hmin] at hat
    exact Bool.noConfusion hat

/-! ## The chain facts -/

/-- **The four scan facts, over the enumeration**: `F_1 <= 11`, `F_3 <= 23`,
`F_4 <= 26`, and no three consecutive gaps all at or above the floor 6. -/
theorem chain_facts (n : ℕ) :
    opSeq (n + 1) - opSeq n ≤ 11 ∧ opSeq (n + 3) - opSeq n ≤ 23 ∧
      opSeq (n + 4) - opSeq n ≤ 26 ∧
      ¬ (6 ≤ g13 n ∧ 6 ≤ g13 (n + 1) ∧ 6 ≤ g13 (n + 2)) := by
  have hx : 1 ≤ opSeq n := opSeq_pos n
  have hE : Exposed13 (opSeq n) := opSeq_exposed n
  have ha0 : atT (opSeq n % 5) (opSeq n % 7) (opSeq n % 11) (opSeq n % 13) 0
      = true := (atT_iff hx 0).mpr (by simpa using hE)
  have h := qokAll (a := opSeq n % 5) (b := opSeq n % 7) (c := opSeq n % 11)
    (d := opSeq n % 13) (by omega) (by omega) (by omega) (by omega)
  rw [qokT, ha0] at h
  simp only [Bool.not_true, Bool.false_or] at h
  simp only [chainT] at h
  set o1 := seekT (opSeq n % 5) (opSeq n % 7) (opSeq n % 11) (opSeq n % 13) 11 0
    with ho1
  set o2 := seekT (opSeq n % 5) (opSeq n % 7) (opSeq n % 11) (opSeq n % 13) 11 o1
    with ho2
  set o3 := seekT (opSeq n % 5) (opSeq n % 7) (opSeq n % 11) (opSeq n % 13) 11 o2
    with ho3
  set o4 := seekT (opSeq n % 5) (opSeq n % 7) (opSeq n % 11) (opSeq n % 13) 11 o3
    with ho4
  simp only [Bool.and_eq_true, Nat.ble_eq, Bool.not_eq_true'] at h
  obtain ⟨h11, h23, h26, hrun⟩ := h
  have hE0 : Exposed13 (opSeq n + 0) := by simpa using hE
  have e1 : opSeq n + o1 = opSeq (n + 1) := by
    rw [opSeq_succ]
    have h1 := seek_next hx hE0
    simpa [← ho1] using h1
  have hEo1 : Exposed13 (opSeq n + o1) := by rw [e1]; exact opSeq_exposed _
  have e2 : opSeq n + o2 = opSeq (n + 2) := by
    rw [show n + 2 = (n + 1) + 1 by omega, opSeq_succ]
    have h2 := seek_next hx hEo1
    rw [e1] at h2
    simpa [← ho2] using h2
  have hEo2 : Exposed13 (opSeq n + o2) := by rw [e2]; exact opSeq_exposed _
  have e3 : opSeq n + o3 = opSeq (n + 3) := by
    rw [show n + 3 = (n + 2) + 1 by omega, opSeq_succ]
    have h3 := seek_next hx hEo2
    rw [e2] at h3
    simpa [← ho3] using h3
  have hEo3 : Exposed13 (opSeq n + o3) := by rw [e3]; exact opSeq_exposed _
  have e4 : opSeq n + o4 = opSeq (n + 4) := by
    rw [show n + 4 = (n + 3) + 1 by omega, opSeq_succ]
    have h4 := seek_next hx hEo3
    rw [e3] at h4
    simpa [← ho4] using h4
  have g0 : g13 n = opSeq (n + 1) - opSeq n := rfl
  have g1 : g13 (n + 1) = opSeq (n + 2) - opSeq (n + 1) := by simp only [g13]
  have g2 : g13 (n + 2) = opSeq (n + 3) - opSeq (n + 2) := by simp only [g13]
  refine ⟨by omega, by omega, by omega, ?_⟩
  rintro ⟨c0, c1, c2⟩
  have hbt : (Nat.ble 6 o1 && Nat.ble 6 (o2 - o1) && Nat.ble 6 (o3 - o2))
      = true := by
    simp only [Bool.and_eq_true, Nat.ble_eq]
    refine ⟨⟨?_, ?_⟩, ?_⟩ <;> omega
  rw [hbt] at hrun
  exact Bool.noConfusion hrun

/-- **`Q_j(13; 6) = 0` for `j >= 5`**: no three consecutive gaps of `g13`
all meet the qualifying floor 6. -/
theorem no_big_run (n : ℕ) :
    ¬ (6 ≤ g13 n ∧ 6 ≤ g13 (n + 1) ∧ 6 ≤ g13 (n + 2)) :=
  (chain_facts n).2.2.2

/-! ## The spectrum ladder over the gap word -/

/-- `F_1(13) <= 11`. -/
theorem spectrum_one : Spectrum.SpectrumBound g13 1 11 := by
  intro a; rw [windowSum_g13]; exact (chain_facts a).1

/-- `F_2(13) <= 16` - the round-11 certificate, wired to `g13`. -/
theorem spectrum_two : Spectrum.SpectrumBound g13 2 16 := by
  intro a
  rw [windowSum_g13]
  have e1 : a + 1 + 1 = a + 2 := by omega
  exact pair_sum_le (opSeq_pos a) (opSeq_lt_succ a) (e1 ▸ opSeq_lt_succ (a + 1))
    (opSeq_exposed a) (opSeq_exposed (a + 1)) (opSeq_exposed (a + 2))
    (opSeq_gap_empty a) (e1 ▸ opSeq_gap_empty (a + 1))

/-- `F_3(13) <= 23`. -/
theorem spectrum_three : Spectrum.SpectrumBound g13 3 23 := by
  intro a; rw [windowSum_g13]; exact (chain_facts a).2.1

/-- `F_4(13) <= 26`. -/
theorem spectrum_four : Spectrum.SpectrumBound g13 4 26 := by
  intro a; rw [windowSum_g13]; exact (chain_facts a).2.2.1

/-- **The kernel-fed spectrum ladder of machine 13**: `F_1..F_4 <= 11, 16,
23, 26`. -/
theorem spectrum_ladder :
    Spectrum.SpectrumBound g13 1 11 ∧ Spectrum.SpectrumBound g13 2 16 ∧
      Spectrum.SpectrumBound g13 3 23 ∧ Spectrum.SpectrumBound g13 4 26 :=
  ⟨spectrum_one, spectrum_two, spectrum_three, spectrum_four⟩

/-! ## The qualifying spectrum, closed at every depth -/

/-- **`Q_j(13; 6) <= 26` for every depth `j >= 3`** (floor `2u' = 6`, gear
17): depths 3 and 4 from the unconditional ladder, and NO qualifying window
of depth 5 or more exists at all (`no_big_run`). -/
theorem qual_bound_all : ∀ j, 3 ≤ j → Spectrum.QualBound g13 3 j 26 := by
  intro j hj a hq
  rcases Nat.lt_or_ge j 5 with hj5 | hj5
  · interval_cases j
    · exact le_trans (spectrum_three a) (by omega)
    · exact spectrum_four a
  · exfalso
    refine no_big_run (a + 1) ⟨?_, ?_, ?_⟩
    · have h1 := hq 1 (by omega) (by omega); omega
    · have h2 := hq 2 (by omega) (by omega)
      rw [show a + 1 + 1 = a + 2 by omega]; omega
    · have h3 := hq 3 (by omega) (by omega)
      rw [show a + 1 + 2 = a + 3 by omega]; omega

/-! ## The opening enumeration is complete -/

theorem opSeq_strict_mono {a b : ℕ} (h : a < b) : opSeq a < opSeq b := by
  have h1 := opSeq_lt_succ a
  have h2 := opSeq_le_add (a + 1) (b - (a + 1))
  rw [show a + 1 + (b - (a + 1)) = b by omega] at h2
  omega

theorem opSeq_reach : ∀ dd A, 1 ≤ A → Exposed13 A → (∃ n, opSeq n = A) →
    ∀ B, Exposed13 B → A < B → B - A ≤ dd → ∃ m, opSeq m = B := by
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
theorem opSeq_surj {m : ℕ} (hm : 1 ≤ m) (hE : Exposed13 m) :
    ∃ n, opSeq n = m := by
  have h0 : opSeq 0 = nextOp 0 := rfl
  have hle : nextOp 0 ≤ m := Nat.find_min' (exists_exposed_above 0) ⟨by omega, hE⟩
  rcases eq_or_lt_of_le hle with he | hlt
  · exact ⟨0, by rw [h0, he]⟩
  · exact opSeq_reach (m - nextOp 0) (nextOp 0) (by have := nextOp_gt 0; omega)
      (nextOp_exposed 0) ⟨0, rfl⟩ m hE hlt (by omega)

end Machine13
