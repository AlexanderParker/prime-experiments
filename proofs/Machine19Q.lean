/-
Machine 19: the qualifying spectrum closed at every depth - (D) at
`alpha = 3` with NO shallowness hypothesis (round 21).

Round 20's `D_of_shallow_word` proved (D) at machine 19 for words of at
most 2 letters (`l + 2 <= 4`), the flatness half kernel-fed, shallowness
left hypothetical. This file removes the hypothesis. From the round-21
chain scan (`Machine19QS0..16`, 323 slices):

* `chain_facts`     - for every opening: the third next opening is within
  35 (`F_3 <= 35`), the fifth within 47 (`F_5 <= 47`), and the next four
  gaps are never all `>= 8` (`Q_6(19) = 0` - so NO qualifying window of
  any depth `j >= 6` exists at all);
* `spectrum_ladder` - `F_1..F_5 <= 25, 31, 35, 38, 47`, all kernel-fed
  (note `F_5 = 47 <= 48 = F + q'`: depth 5 is flat with no qualifying
  constraint);
* `qual_bound_all`  - `Q_j <= 47` for EVERY depth `j >= 3` (floor
  `2u' = 8`);
* `D_of_word`       - every floor-respecting word of EVERY length merges
  to at most `F + q' = 48`. The only hypothesis left is the floor itself,
  and the merge law discharges it: at the 19->23 step every merge-word
  letter is `0, 8 or 15 mod 23`, hence in `{8, 15, 23}` - all `>= 8`
  (`Machine23.lean` wires this end to end).

The scan reads all three facts off ONE five-step `seekT` walk per opening
(`seek_next` proves the walk computes `nextOp` exactly, using round 20's
kernel fact that gaps cap at 25), so the extraction needs no witness
pigeonhole: the chain IS the consecutive openings. `opSeq_surj` (the
enumeration is onto the openings) completes the toolkit `Machine23.lean`
needs to instantiate `MergeLaw.newgap_le` (R39) and prove that every gap
of machine 23 is at most 47 - (D) at the 19->23 step with no hypotheses.
-/

import Machine19
import Machine19QS0
import Machine19QS1
import Machine19QS2
import Machine19QS3
import Machine19QS4
import Machine19QS5
import Machine19QS6
import Machine19QS7
import Machine19QS8
import Machine19QS9
import Machine19QS10
import Machine19QS11
import Machine19QS12
import Machine19QS13
import Machine19QS14
import Machine19QS15
import Machine19QS16

namespace Machine19

/-! ## Assembly: the whole period -/

/-- **One period, all 323 slices of the chain scan.** -/
theorem qsliceAll : ∀ e < 17, ∀ f < 19, qslice e f = true := by
  intro e he
  interval_cases e
  exacts [qasm0, qasm1, qasm2, qasm3, qasm4, qasm5, qasm6, qasm7, qasm8,
    qasm9, qasm10, qasm11, qasm12, qasm13, qasm14, qasm15, qasm16]

/-- The tuple-level fact, unpacked from the slice. -/
theorem qokAll {a b c d e f : ℕ} (ha : a < 5) (hb : b < 7) (hc : c < 11)
    (hd : d < 13) (he : e < 17) (hf : f < 19) : qokT a b c d e f = true := by
  have h := qsliceAll e he f hf
  rw [qslice, List.all_eq_true] at h
  have h1 := h a (List.mem_range.mpr ha)
  rw [List.all_eq_true] at h1
  have h2 := h1 b (List.mem_range.mpr hb)
  rw [List.all_eq_true] at h2
  have h3 := h2 c (List.mem_range.mpr hc)
  rw [List.all_eq_true] at h3
  exact h3 d (List.mem_range.mpr hd)

/-! ## The seek walk, related to `nextOp` -/

/-- The seek step when the next slot is an opening. -/
theorem seekT_succ_pos {a b c d e f fu s : ℕ}
    (h : atT a b c d e f (s + 1) = true) :
    seekT a b c d e f (fu + 1) s = s + 1 := by
  simp only [seekT]
  split
  · rfl
  · rename_i hneg
    exact absurd h hneg

/-- The seek step when the next slot is not an opening. -/
theorem seekT_succ_neg {a b c d e f fu s : ℕ}
    (h : ¬ atT a b c d e f (s + 1) = true) :
    seekT a b c d e f (fu + 1) s = seekT a b c d e f fu (s + 1) := by
  simp only [seekT]
  split
  · rename_i hpos
    exact absurd hpos h
  · rfl

/-- The seek result strictly advances. -/
theorem seekT_gt (a b c d e f : ℕ) :
    ∀ fu s, s < seekT a b c d e f fu s := by
  intro fu
  induction fu with
  | zero => intro s; simp only [seekT]; omega
  | succ fu ih =>
    intro s
    by_cases h : atT a b c d e f (s + 1) = true
    · rw [seekT_succ_pos h]; omega
    · rw [seekT_succ_neg h]
      have := ih (s + 1); omega

/-- If an opening exists within the fuel, the seek finds one in range. -/
theorem seekT_found (a b c d e f : ℕ) :
    ∀ fu s t, s < t → t ≤ s + fu → atT a b c d e f t = true →
      seekT a b c d e f fu s ≤ s + fu := by
  intro fu
  induction fu with
  | zero => intro s t h1 h2 _; omega
  | succ fu ih =>
    intro s t h1 h2 hat
    by_cases h : atT a b c d e f (s + 1) = true
    · rw [seekT_succ_pos h]
      omega
    · rw [seekT_succ_neg h]
      have hne : t ≠ s + 1 := by intro he; rw [he] at hat; exact h hat
      have := ih (s + 1) t (by omega) (by omega) hat
      omega

/-- A found seek result is an opening. -/
theorem seekT_exposed (a b c d e f : ℕ) :
    ∀ fu s, seekT a b c d e f fu s ≤ s + fu →
      atT a b c d e f (seekT a b c d e f fu s) = true := by
  intro fu
  induction fu with
  | zero => intro s h; simp only [seekT] at h; omega
  | succ fu ih =>
    intro s h
    by_cases hat : atT a b c d e f (s + 1) = true
    · rw [seekT_succ_pos hat]
      exact hat
    · rw [seekT_succ_neg hat] at h ⊢
      exact ih (s + 1) (by omega)

/-- Nothing before the seek result (within fuel) is an opening. -/
theorem seekT_min (a b c d e f : ℕ) :
    ∀ fu s t, s < t → t < seekT a b c d e f fu s → t ≤ s + fu →
      atT a b c d e f t = false := by
  intro fu
  induction fu with
  | zero => intro s t h1 _ h3; omega
  | succ fu ih =>
    intro s t h1 h2 h3
    by_cases hat : atT a b c d e f (s + 1) = true
    · rw [seekT_succ_pos hat] at h2
      omega
    · rw [seekT_succ_neg hat] at h2
      rcases Nat.lt_or_ge t (s + 2) with hlt | hge
      · have he : t = s + 1 := by omega
        subst he
        simpa using hat
      · exact ih (s + 1) t (by omega) h2 (by omega)

/-- The next opening after an opening arrives within 25 slots. -/
theorem nextOp_le_25 {x : ℕ} (hx : 1 ≤ x) (hE : Exposed19 x) :
    nextOp x ≤ x + 25 := by
  have h := (window_facts hx hE).1
  rw [List.any_eq_true] at h
  obtain ⟨i, hi, hv⟩ := h
  have hi25 := List.mem_range.mp hi
  have hEi : Exposed19 (x + (i + 1)) := (atT_iff hx _).mp hv
  have hfind : nextOp x ≤ x + (i + 1) :=
    Nat.find_min' (exists_exposed_above x) ⟨by omega, hEi⟩
  omega

/-- **The seek walk computes `nextOp`.** From base opening `x`, if `x + s`
is an opening then `x + seekT ... 25 s` is the next one. -/
theorem seek_next {x s : ℕ} (hx : 1 ≤ x) (hE : Exposed19 (x + s)) :
    x + seekT (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) (x % 19) 25 s
      = nextOp (x + s) := by
  have hE1 : 1 ≤ x + s := by omega
  have hnle : nextOp (x + s) ≤ x + s + 25 := nextOp_le_25 hE1 hE
  have hngt : x + s < nextOp (x + s) := nextOp_gt _
  have hnE : Exposed19 (nextOp (x + s)) := nextOp_exposed _
  have hat : atT (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) (x % 19)
      (nextOp (x + s) - x) = true := by
    apply (atT_iff hx _).mpr
    rwa [show x + (nextOp (x + s) - x) = nextOp (x + s) by omega]
  have hfound := seekT_found (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) (x % 19)
    25 s (nextOp (x + s) - x) (by omega) (by omega) hat
  have hσat := seekT_exposed (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) (x % 19)
    25 s hfound
  have hσE : Exposed19
      (x + seekT (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) (x % 19) 25 s) :=
    (atT_iff hx _).mp hσat
  have hσgt := seekT_gt (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) (x % 19) 25 s
  have hle1 : nextOp (x + s)
      ≤ x + seekT (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) (x % 19) 25 s :=
    Nat.find_min' (exists_exposed_above (x + s)) ⟨by omega, hσE⟩
  rcases eq_or_lt_of_le hle1 with he | hlt
  · omega
  · exfalso
    have hmin := seekT_min (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) (x % 19)
      25 s (nextOp (x + s) - x) (by omega) (by omega) (by omega)
    rw [hmin] at hat
    exact Bool.noConfusion hat

/-- The enumeration's defining equation, named. -/
theorem opSeq_succ (n : ℕ) : opSeq (n + 1) = nextOp (opSeq n) := rfl

/-! ## The chain facts -/

/-- **The three scan facts, over the enumeration**: from any opening, the
third next opening is within 35 slots, the fifth within 47, and the next
four gaps are never all at or above the qualifying floor 8. -/
theorem chain_facts (n : ℕ) :
    opSeq (n + 3) - opSeq n ≤ 35 ∧ opSeq (n + 5) - opSeq n ≤ 47 ∧
      ¬ (8 ≤ g19 n ∧ 8 ≤ g19 (n + 1) ∧ 8 ≤ g19 (n + 2) ∧ 8 ≤ g19 (n + 3)) := by
  have hx : 1 ≤ opSeq n := opSeq_pos n
  have hE : Exposed19 (opSeq n) := opSeq_exposed n
  have ha0 : atT (opSeq n % 5) (opSeq n % 7) (opSeq n % 11) (opSeq n % 13)
      (opSeq n % 17) (opSeq n % 19) 0 = true :=
    (atT_iff hx 0).mpr (by simpa using hE)
  have h := qokAll (a := opSeq n % 5) (b := opSeq n % 7) (c := opSeq n % 11)
    (d := opSeq n % 13) (e := opSeq n % 17) (f := opSeq n % 19)
    (by omega) (by omega) (by omega) (by omega) (by omega) (by omega)
  rw [qokT, ha0] at h
  simp only [Bool.not_true, Bool.false_or] at h
  simp only [chainT] at h
  set o1 := seekT (opSeq n % 5) (opSeq n % 7) (opSeq n % 11) (opSeq n % 13)
    (opSeq n % 17) (opSeq n % 19) 25 0 with ho1
  set o2 := seekT (opSeq n % 5) (opSeq n % 7) (opSeq n % 11) (opSeq n % 13)
    (opSeq n % 17) (opSeq n % 19) 25 o1 with ho2
  set o3 := seekT (opSeq n % 5) (opSeq n % 7) (opSeq n % 11) (opSeq n % 13)
    (opSeq n % 17) (opSeq n % 19) 25 o2 with ho3
  set o4 := seekT (opSeq n % 5) (opSeq n % 7) (opSeq n % 11) (opSeq n % 13)
    (opSeq n % 17) (opSeq n % 19) 25 o3 with ho4
  set o5 := seekT (opSeq n % 5) (opSeq n % 7) (opSeq n % 11) (opSeq n % 13)
    (opSeq n % 17) (opSeq n % 19) 25 o4 with ho5
  simp only [Bool.and_eq_true, Nat.ble_eq, Bool.not_eq_true'] at h
  obtain ⟨h35, h47, hrun⟩ := h
  -- the chain equations
  have hE0 : Exposed19 (opSeq n + 0) := by simpa using hE
  have e1 : opSeq n + o1 = opSeq (n + 1) := by
    rw [opSeq_succ]
    have h1 := seek_next hx hE0
    simpa [← ho1] using h1
  have hEo1 : Exposed19 (opSeq n + o1) := by rw [e1]; exact opSeq_exposed _
  have e2 : opSeq n + o2 = opSeq (n + 2) := by
    rw [show n + 2 = (n + 1) + 1 by omega, opSeq_succ]
    have h2 := seek_next hx hEo1
    rw [e1] at h2
    simpa [← ho2] using h2
  have hEo2 : Exposed19 (opSeq n + o2) := by rw [e2]; exact opSeq_exposed _
  have e3 : opSeq n + o3 = opSeq (n + 3) := by
    rw [show n + 3 = (n + 2) + 1 by omega, opSeq_succ]
    have h3 := seek_next hx hEo2
    rw [e2] at h3
    simpa [← ho3] using h3
  have hEo3 : Exposed19 (opSeq n + o3) := by rw [e3]; exact opSeq_exposed _
  have e4 : opSeq n + o4 = opSeq (n + 4) := by
    rw [show n + 4 = (n + 3) + 1 by omega, opSeq_succ]
    have h4 := seek_next hx hEo3
    rw [e3] at h4
    simpa [← ho4] using h4
  have hEo4 : Exposed19 (opSeq n + o4) := by rw [e4]; exact opSeq_exposed _
  have e5 : opSeq n + o5 = opSeq (n + 5) := by
    rw [show n + 5 = (n + 4) + 1 by omega, opSeq_succ]
    have h5 := seek_next hx hEo4
    rw [e4] at h5
    simpa [← ho5] using h5
  refine ⟨by omega, by omega, ?_⟩
  rintro ⟨c1, c2, c3, c4⟩
  have g0 : g19 n = opSeq (n + 1) - opSeq n := rfl
  have g1 : g19 (n + 1) = opSeq (n + 2) - opSeq (n + 1) := by simp only [g19]
  have g2 : g19 (n + 2) = opSeq (n + 3) - opSeq (n + 2) := by simp only [g19]
  have g3 : g19 (n + 3) = opSeq (n + 4) - opSeq (n + 3) := by simp only [g19]
  have m1 : opSeq n < opSeq (n + 1) := opSeq_lt_succ n
  have m2 : opSeq (n + 1) < opSeq (n + 2) := by
    rw [show n + 2 = (n + 1) + 1 by omega]; exact opSeq_lt_succ _
  have m3 : opSeq (n + 2) < opSeq (n + 3) := by
    rw [show n + 3 = (n + 2) + 1 by omega]; exact opSeq_lt_succ _
  have m4 : opSeq (n + 3) < opSeq (n + 4) := by
    rw [show n + 4 = (n + 3) + 1 by omega]; exact opSeq_lt_succ _
  have hbt : (Nat.ble 8 o1 && Nat.ble 8 (o2 - o1) && Nat.ble 8 (o3 - o2) &&
      Nat.ble 8 (o4 - o3)) = true := by
    simp only [Bool.and_eq_true, Nat.ble_eq]
    refine ⟨⟨⟨?_, ?_⟩, ?_⟩, ?_⟩ <;> omega
  rw [hbt] at hrun
  exact Bool.noConfusion hrun

/-- **`Q_6(19) = 0` over the gap word**: no four consecutive gaps of `g19`
are all at or above the qualifying floor 8. -/
theorem no_big_run (n : ℕ) :
    ¬ (8 ≤ g19 n ∧ 8 ≤ g19 (n + 1) ∧ 8 ≤ g19 (n + 2) ∧ 8 ≤ g19 (n + 3)) :=
  (chain_facts n).2.2

/-! ## The spectrum ladder over the gap word -/

/-- `F_1 <= 25`, wired to `g19`. -/
theorem spectrum_one : Spectrum.SpectrumBound g19 1 25 := by
  intro a
  rw [windowSum_g19]
  exact gap_le (opSeq_pos a) (opSeq_lt_succ a)
    (opSeq_exposed a) (opSeq_exposed (a + 1)) (opSeq_gap_empty a)

/-- `F_2 <= 31`, wired to `g19`. -/
theorem spectrum_two : Spectrum.SpectrumBound g19 2 31 := by
  intro a
  rw [windowSum_g19]
  have e1 : a + 1 + 1 = a + 2 := by omega
  exact pair_sum_le (opSeq_pos a) (opSeq_lt_succ a) (e1 ▸ opSeq_lt_succ (a + 1))
    (opSeq_exposed a) (opSeq_exposed (a + 1)) (opSeq_exposed (a + 2))
    (opSeq_gap_empty a) (e1 ▸ opSeq_gap_empty (a + 1))

/-- **`F_3(19) <= 35`**, wired to `g19` - kernel-fed by the chain scan. -/
theorem spectrum_three : Spectrum.SpectrumBound g19 3 35 := by
  intro a
  rw [windowSum_g19]
  exact (chain_facts a).1

/-- **`F_5(19) <= 47`**, wired to `g19` - kernel-fed by the chain scan.
Since `47 <= 48 = F + q'`, depth 5 is FLAT with no qualifying constraint. -/
theorem spectrum_five : Spectrum.SpectrumBound g19 5 47 := by
  intro a
  rw [windowSum_g19]
  exact (chain_facts a).2.1

/-- **The kernel-fed spectrum ladder of machine 19**:
`F_1..F_5 <= 25, 31, 35, 38, 47`. -/
theorem spectrum_ladder :
    Spectrum.SpectrumBound g19 1 25 ∧ Spectrum.SpectrumBound g19 2 31 ∧
      Spectrum.SpectrumBound g19 3 35 ∧ Spectrum.SpectrumBound g19 4 38 ∧
      Spectrum.SpectrumBound g19 5 47 :=
  ⟨spectrum_one, spectrum_two, spectrum_three, spectrum_four, spectrum_five⟩

/-! ## The qualifying spectrum, closed at every depth -/

/-- **`Q_j(19) <= 47` for every depth `j >= 3`** (floor `2u' = 8`, i.e.
`u' = 4`): depths 3, 4, 5 from the flat ladder, and NO qualifying window
of depth 6 or more exists at all (`no_big_run`). This is the word-free
criterion `merged_le_of_qual_flat_all` asks for, with `q' = 23` margin 1. -/
theorem qual_bound_all : ∀ j, 3 ≤ j → Spectrum.QualBound g19 4 j 47 := by
  intro j hj a hq
  rcases Nat.lt_or_ge j 6 with hj6 | hj6
  · interval_cases j
    · exact le_trans (spectrum_three a) (by omega)
    · exact le_trans (spectrum_four a) (by omega)
    · exact spectrum_five a
  · exfalso
    refine no_big_run (a + 1) ⟨?_, ?_, ?_, ?_⟩
    · have h1 := hq 1 (by omega) (by omega)
      omega
    · have h2 := hq 2 (by omega) (by omega)
      rw [show a + 1 + 1 = a + 2 by omega]
      omega
    · have h3 := hq 3 (by omega) (by omega)
      rw [show a + 1 + 2 = a + 3 by omega]
      omega
    · have h4 := hq 4 (by omega) (by omega)
      rw [show a + 1 + 3 = a + 4 by omega]
      omega

/-- The brief's literal target, subsumed: `Q_5(19) <= 48`. -/
theorem qual_five_flat : Spectrum.QualBound g19 4 5 (25 + 23) :=
  fun a hq => le_trans (qual_bound_all 5 (by omega) a hq) (by omega)

/-- **(D) at `alpha = 3` at machine 19, for EVERY word length.** A word of
`l` letters all meeting the qualifying floor `2u' = 8` merges to at most
`F + q' = 48` - no shallowness hypothesis, no fuel cap, no word list.
(Every letter of the actual 19->23 merge alphabet is 8, 15 or 23 - see
`Machine23.lean` - so the floor hypothesis is not a restriction.) -/
theorem D_of_word {a l : ℕ} (hw : ∀ i < l, 8 ≤ g19 (a + 1 + i)) :
    g19 a + Spectrum.windowSum g19 (a + 1) l + g19 (a + l + 1) ≤ 25 + 23 := by
  rcases Nat.lt_or_ge l 4 with hl | hl
  · rw [Spectrum.merged_eq]
    interval_cases l
    · exact le_trans (spectrum_two a) (by omega)
    · exact le_trans (spectrum_three a) (by omega)
    · exact le_trans (spectrum_four a) (by omega)
    · exact le_trans (spectrum_five a) (by omega)
  · exfalso
    exact no_big_run (a + 1)
      ⟨by simpa using hw 0 (by omega), hw 1 (by omega), hw 2 (by omega),
        hw 3 (by omega)⟩

/-! ## The opening enumeration is complete -/

/-- `opSeq` is strictly monotone. -/
theorem opSeq_strict_mono {a b : ℕ} (h : a < b) : opSeq a < opSeq b := by
  have h1 := opSeq_lt_succ a
  have h2 := opSeq_le_add (a + 1) (b - (a + 1))
  rw [show a + 1 + (b - (a + 1)) = b by omega] at h2
  omega

/-- Reaching every opening from a reached one. -/
theorem opSeq_reach : ∀ dd A, 1 ≤ A → Exposed19 A → (∃ n, opSeq n = A) →
    ∀ B, Exposed19 B → A < B → B - A ≤ dd → ∃ m, opSeq m = B := by
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
        ⟨n + 1, by rw [opSeq_succ, hn]⟩
        B hEB hlt (by omega)

/-- **Every opening is enumerated**: `opSeq` is onto the openings. -/
theorem opSeq_surj {m : ℕ} (hm : 1 ≤ m) (hE : Exposed19 m) :
    ∃ n, opSeq n = m := by
  have h0 : opSeq 0 = nextOp 0 := rfl
  have hle : nextOp 0 ≤ m := Nat.find_min' (exists_exposed_above 0) ⟨by omega, hE⟩
  rcases eq_or_lt_of_le hle with he | hlt
  · exact ⟨0, by rw [h0, he]⟩
  · exact opSeq_reach (m - nextOp 0) (nextOp 0) (by have := nextOp_gt 0; omega)
      (nextOp_exposed 0) ⟨0, rfl⟩ m hE hlt (by omega)

end Machine19
