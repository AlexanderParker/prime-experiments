/-
R89 AT MACHINE 13, THROUGH THE ROUND-31 LEMMA: `L(13) = 1`, `J_max(13) = 3`,
`A_kill(13 -> 17) = 2` IN THE KERNEL (Formalist, round 31).

R81's second table row.  Machine 11 (round 30, `WordLegal11.lean`) closed for a
degenerate reason - gaps cap at 7 while the letters are `{0, 9, 4}` mod 13, so
only ONE letter was realisable and two of them cannot alternate.  Machine 13 is
the first row where that argument fails: gear 17 has tooth `u' = 3`
(`6 * 3 = 18 = 17 + 1`), the letters are `0`, `2u' = 6` and `-6 = 11` mod 17,
gaps cap at 11 (`Machine13.spectrum_one`), so BOTH literal letters are
realisable as the bare gaps 6 and 11 (they occur 60 and 12 times per period).
What kills the 2-word is the round-31 lemma:

    17 mod 210 = 17 ∈ BareAlt.S, and its bare cap is ONE letter -
    `bareAdmAB 6 11 2 = false`: no translate of `{0,6,17}` or `{0,11,17}`
    avoids the teeth of gears 5 and 7 -

so no two consecutive openings-gaps of machine 13 are `(6,11)` or `(11,6)`.
Gears 5 and 7 alone decide `L(13) <= 1`, exactly as Lateral's round-30 item 79
says they should.  Every gap being at most 11 also rules out the padded letter
(a gap of exactly 17) and every non-bare literal (23, 28, ...), so ALL legal
letters of machine 13 are bare and the lemma's conclusion `L_bare <= 1` is
`L <= 1` here.

    theorem L13     : RealisedWord 3 opSeq 1 ∧ ¬ RealisedWord 3 opSeq 2
    theorem jmax13  : QstarNonempty 3 opSeq J ↔ J ≤ 3          -- J_max(13) = 3
    theorem akill13 : (∃ n, Chain 3 opSeq k n) ↔ k ≤ 2  (1 ≤ k) -- A_kill(13->17) = 2

Census cross-check before formalising (`research/bare_alt_r31.py` and a direct
period scan): machine 13's 1,485 gaps over one period of 5,005 slots contain 60
sixes and 12 elevens and ZERO adjacent (6,11) or (11,6) pairs.
-/

import WordLegal
import BareAlternation
import Machine13Per

namespace WordLegal13

open WordLegal Machine13

/-! ## 1. Gears 5 and 7 block machine 13 -/

theorem blocks13_five : BareAlt.Blocks Exposed13 5 1 := by
  intro k hk
  have h1 : ¬ (5 ∣ Census.lo k) := hk.1
  have h2 : ¬ (5 ∣ Census.hi k) := hk.2.1
  simp only [Census.lo, Census.hi] at h1 h2
  exact ⟨by omega, by omega⟩

theorem blocks13_seven : BareAlt.Blocks Exposed13 7 6 := by
  intro k hk
  have h1 : ¬ (7 ∣ Census.lo k) := hk.2.2.1
  have h2 : ¬ (7 ∣ Census.hi k) := hk.2.2.2.1
  simp only [Census.lo, Census.hi] at h1 h2
  exact ⟨by omega, by omega⟩

/-! ## 2. The letters of gear 17 -/

/-- Gear 17's tooth is `3`: `6 * 3 = 1` in `ZMod 17`. -/
theorem tooth17 : (6 : ZMod 17) * 3 = 1 := by decide

theorem val_up : Cls.val (3 : ZMod 17) .up = ((6 : ℕ) : ZMod 17) := by decide
theorem val_down : Cls.val (3 : ZMod 17) .down = ((11 : ℕ) : ZMod 17) := by decide

/-- Every gap of machine 13 is between 1 and 11. -/
theorem g13_bounds (n : ℕ) : 1 ≤ g13 n ∧ g13 n ≤ 11 := by
  constructor
  · have := opSeq_lt_succ n
    unfold g13
    omega
  · have := spectrum_one n
    simpa [Spectrum.windowSum] using this

theorem gapRes_g13 (n : ℕ) : gapRes (q := 17) opSeq n = ((g13 n : ℕ) : ZMod 17) :=
  gapRes_eq_cast opSeq (opSeq_lt_succ n).le

/-- **Every legal letter of machine 13 is BARE**: the padded letter needs a gap
divisible by 17 and the gaps cap at 11, so the two literal classes are realised
only by the gaps 6 and 11 themselves. -/
theorem letter13 {n : ℕ} {a : Cls}
    (h : gapRes (q := 17) opSeq n = Cls.val (3 : ZMod 17) a) :
    (a = .up ∧ g13 n = 6) ∨ (a = .down ∧ g13 n = 11) := by
  rw [gapRes_g13] at h
  obtain ⟨h1, h11⟩ := g13_bounds n
  cases a with
  | pad =>
      simp only [Cls.val] at h
      have hd := (ZMod.natCast_eq_zero_iff (g13 n) 17).mp h
      have := Nat.le_of_dvd h1 hd
      exfalso
      omega
  | up =>
      rw [val_up, ZMod.natCast_eq_natCast_iff'] at h
      exact Or.inl ⟨rfl, by omega⟩
  | down =>
      rw [val_down, ZMod.natCast_eq_natCast_iff'] at h
      exact Or.inr ⟨rfl, by omega⟩

/-! ## 3. `L(13) >= 1`: the gap 6 at index 16 -/

theorem g13_eq_ow (n : ℕ) : g13 n = ow13 (n + 1) - ow13 n := by
  unfold g13
  rw [opSeq_eq_ow13, opSeq_eq_ow13]
  omega

theorem g13_16 : g13 16 = 6 := by
  rw [g13_eq_ow]
  decide +kernel

theorem realised1 : RealisedWord (3 : ZMod 17) opSeq 1 := by
  refine ⟨16, [.up], rfl, ⟨false, rfl, trivial⟩, ?_, trivial⟩
  rw [gapRes_g13, g13_16, val_up]

/-! ## 4. `L(13) < 2`: the bare pair is inadmissible at gears 5 and 7 -/

/-- The round-31 finite check: neither `{0,6,17}` nor `{0,11,17}` has a
translate inside the exposed sets of gears 5 and 7. -/
theorem bare_pair_inadmissible : BareAlt.bareAdmAB 6 11 2 = false := by decide

/-- A gap value pins the next opening. -/
theorem gapWord_of_g13 {i : ℕ} {x y : ℕ} (hx : g13 i = x) (hy : g13 (i + 1) = y) :
    BareAlt.GapWordAt opSeq i [x, y] := by
  have h1 := opSeq_lt_succ i
  have h2 := opSeq_lt_succ (i + 1)
  unfold g13 at hx hy
  exact ⟨by omega, by omega, trivial⟩

theorem no_word2 : ¬ RealisedWord (3 : ZMod 17) opSeq 2 := by
  rintro ⟨i, w, hl, ⟨t, ht⟩, hw⟩
  rcases w with _ | ⟨a, _ | ⟨b, _ | ⟨c, w⟩⟩⟩
  · simp at hl
  · simp at hl
  · obtain ⟨h1, h2, -⟩ := hw
    have key := BareAlt.no_bare_run (E := Exposed13) opSeq_exposed blocks13_five
      blocks13_seven bare_pair_inadmissible i
    rcases letter13 h1 with ⟨rfl, e1⟩ | ⟨rfl, e1⟩
    · -- first letter `up` (gap 6); alternation forces the second to be `down`
      obtain ⟨-, hb⟩ : t = false ∧ Alt true [b] := ht
      cases b with
      | pad => rcases letter13 h2 with ⟨hc, -⟩ | ⟨hc, -⟩ <;> exact absurd hc (by decide)
      | up => exact absurd hb (by simp [Alt])
      | down =>
          rcases letter13 h2 with ⟨hc, -⟩ | ⟨-, e2⟩
          · exact absurd hc (by decide)
          · exact key.1 (gapWord_of_g13 e1 e2)
    · -- first letter `down` (gap 11); the second must be `up`
      obtain ⟨-, hb⟩ : t = true ∧ Alt false [b] := ht
      cases b with
      | pad => rcases letter13 h2 with ⟨hc, -⟩ | ⟨hc, -⟩ <;> exact absurd hc (by decide)
      | up =>
          rcases letter13 h2 with ⟨-, e2⟩ | ⟨hc, -⟩
          · exact key.2 (gapWord_of_g13 e1 e2)
          · exact absurd hc (by decide)
      | down => exact absurd hb (by simp [Alt])
  · simp at hl

/-- **`L(13) = 1`.** -/
theorem L13 : RealisedWord (3 : ZMod 17) opSeq 1 ∧ ¬ RealisedWord (3 : ZMod 17) opSeq 2 :=
  ⟨realised1, no_word2⟩

/-! ## 5. R89 at machine 13 -/

/-- The gap residues of machine 13 are periodic with period 1,485 (indices),
5,005 slots. -/
theorem hper13 (n : ℕ) : gapRes (q := 17) opSeq (n + 1485) = gapRes opSeq n := by
  rw [gapRes_g13, gapRes_g13, g13_shift]

/-- **`J_max(13) = 3`.** -/
theorem jmax13 (J : ℕ) : QstarNonempty (3 : ZMod 17) opSeq J ↔ J ≤ 3 :=
  jmax (3 : ZMod 17) opSeq hper13 (by norm_num) realised1 no_word2 J

/-- **`A_kill(13 -> 17) = 2`.** -/
theorem akill13 (k : ℕ) (hk : 1 ≤ k) : (∃ n, Chain (3 : ZMod 17) opSeq k n) ↔ k ≤ 2 :=
  akill (3 : ZMod 17) opSeq realised1 no_word2 k hk

end WordLegal13
