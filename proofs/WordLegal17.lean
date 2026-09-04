/-
R89 AT MACHINE 17, THROUGH THE ROUND-31 LEMMA: `L(17) = 1` and
`A_kill(17 -> 19) = 2` IN THE KERNEL (Formalist, round 31).

R81's third table row.  Gear 19 has tooth `u' = 16` (`6 * 16 = 96 = 5*19 + 1`),
so the letters are `0`, `2u' = 32 = 13` and `-13 = 6` mod 19; every gap of
machine 17 is at most 18 (`Machine17.spectrum_one`), so the padded letter (a
gap of exactly 19) and every non-bare literal (25, 32, ...) are out of range
and the two literal classes are realised ONLY by the bare gaps 6 and 13.  The
round-31 lemma then closes the 2-word:

    19 mod 210 = 19 ∈ BareAlt.S, bare cap ONE letter -
    `bareAdmAB 6 13 2 = false`: no translate of `{0,6,19}` or `{0,13,19}`
    avoids the teeth of gears 5 and 7,

so no two consecutive gaps of machine 17 are `(6,13)` or `(13,6)`, and
`L(17) = 1`.  Gears 5 and 7 alone decide it - Lateral's round-30 item 79 at
its second corpus rung.

    theorem L17     : RealisedWord 16 opSeq 1 ∧ ¬ RealisedWord 16 opSeq 2
    theorem akill17 : (∃ n, Chain 16 opSeq k n) ↔ k ≤ 2  (1 ≤ k) -- A_kill(17->19) = 2

NOT PROVED HERE: `J_max(17) = 3`.  `WordLegal.jmax` needs `hper`, the
periodicity of the gap residues, and machine 17 has no period module in the
ledger (`Machine13Per` supplies machine 13's; machine 17's period is 85,085
slots / 19,305 openings, so the `ow` walk's base case is a 19,305-step
`decide +kernel` and was not attempted).  `akill` needs no periodicity at all.

The realiser: `opSeq 15 = 52`, `opSeq 16 = 58`, gap 6.  The opening walk
`ow17` is built here in the `Machine13Per` shape (`Machine17.seek_next` from
the ledger does the work).
-/

import WordLegal
import BareAlternation
import Machine17Q

namespace WordLegal17

open WordLegal Machine17

/-! ## 1. Gears 5 and 7 block machine 17 -/

theorem blocks17_five : BareAlt.Blocks Exposed17 5 1 := by
  intro k hk
  have h1 : ¬ (5 ∣ Census.lo k) := hk.1
  have h2 : ¬ (5 ∣ Census.hi k) := hk.2.1
  simp only [Census.lo, Census.hi] at h1 h2
  exact ⟨by omega, by omega⟩

theorem blocks17_seven : BareAlt.Blocks Exposed17 7 6 := by
  intro k hk
  have h1 : ¬ (7 ∣ Census.lo k) := hk.2.2.1
  have h2 : ¬ (7 ∣ Census.hi k) := hk.2.2.2.1
  simp only [Census.lo, Census.hi] at h1 h2
  exact ⟨by omega, by omega⟩

/-! ## 2. The letters of gear 19 -/

/-- Gear 19's tooth is `16`: `6 * 16 = 1` in `ZMod 19`. -/
theorem tooth19 : (6 : ZMod 19) * 16 = 1 := by decide

theorem val_up : Cls.val (16 : ZMod 19) .up = ((13 : ℕ) : ZMod 19) := by decide
theorem val_down : Cls.val (16 : ZMod 19) .down = ((6 : ℕ) : ZMod 19) := by decide

/-- Every gap of machine 17 is between 1 and 18. -/
theorem g17_bounds (n : ℕ) : 1 ≤ g17 n ∧ g17 n ≤ 18 := by
  constructor
  · have := opSeq_lt_succ n
    unfold g17
    omega
  · have := spectrum_one n
    simpa [Spectrum.windowSum] using this

theorem gapRes_g17 (n : ℕ) : gapRes (q := 19) opSeq n = ((g17 n : ℕ) : ZMod 19) :=
  gapRes_eq_cast opSeq (opSeq_lt_succ n).le

/-- **Every legal letter of machine 17 is BARE.** -/
theorem letter17 {n : ℕ} {a : Cls}
    (h : gapRes (q := 19) opSeq n = Cls.val (16 : ZMod 19) a) :
    (a = .up ∧ g17 n = 13) ∨ (a = .down ∧ g17 n = 6) := by
  rw [gapRes_g17] at h
  obtain ⟨h1, h18⟩ := g17_bounds n
  cases a with
  | pad =>
      simp only [Cls.val] at h
      have hd := (ZMod.natCast_eq_zero_iff (g17 n) 19).mp h
      have := Nat.le_of_dvd h1 hd
      exfalso
      omega
  | up =>
      rw [val_up, ZMod.natCast_eq_natCast_iff'] at h
      exact Or.inl ⟨rfl, by omega⟩
  | down =>
      rw [val_down, ZMod.natCast_eq_natCast_iff'] at h
      exact Or.inr ⟨rfl, by omega⟩

/-! ## 3. The opening walk, and `L(17) >= 1` -/

/-- Machine 17's first opening is slot 5 (`29, 31`). -/
theorem opSeq_zero : opSeq 0 = 5 := by
  have hE := nextOp_exposed 0
  have hgt := nextOp_gt 0
  have hle : nextOp 0 ≤ 5 :=
    Nat.find_min' (exists_exposed_above 0) ⟨by omega, by decide⟩
  have h1 : ¬ Exposed17 1 := by decide
  have h2 : ¬ Exposed17 2 := by decide
  have h3 : ¬ Exposed17 3 := by decide
  have h4 : ¬ Exposed17 4 := by decide
  show nextOp 0 = 5
  rcases Nat.lt_or_ge (nextOp 0) 5 with h | h
  · exfalso
    have hc : nextOp 0 = 1 ∨ nextOp 0 = 2 ∨ nextOp 0 = 3 ∨ nextOp 0 = 4 := by omega
    rcases hc with he | he | he | he <;> rw [he] at hE
    · exact h1 hE
    · exact h2 hE
    · exact h3 hE
    · exact h4 hE
  · omega

/-- **The opening walk of machine 17**, from its first opening at slot 5
(`x % 5, x % 7, x % 11, x % 13, x % 17` = `0, 5, 5, 5, 5`; fuel 18). -/
def ow17 : ℕ → ℕ
  | 0 => 0
  | i + 1 => seekT 0 5 5 5 5 18 (ow17 i)

/-- **The walk is the enumeration.** -/
theorem opSeq_eq_ow17 : ∀ i, opSeq i = 5 + ow17 i := by
  intro i
  induction i with
  | zero => rw [opSeq_zero]; rfl
  | succ i ih =>
    have hE : Exposed17 (5 + ow17 i) := by rw [← ih]; exact opSeq_exposed i
    have h := seek_next (x := 5) (s := ow17 i) (by omega) hE
    rw [opSeq_succ, ih, ← h]
    rfl

theorem g17_eq_ow (n : ℕ) : g17 n = ow17 (n + 1) - ow17 n := by
  unfold g17
  rw [opSeq_eq_ow17, opSeq_eq_ow17]
  omega

set_option maxRecDepth 4000 in
theorem g17_15 : g17 15 = 6 := by
  rw [g17_eq_ow]
  decide +kernel

/-- `L(17) >= 1`: the gap `6` at index 15 (openings 52 and 58). -/
theorem realised1 : RealisedWord (16 : ZMod 19) opSeq 1 := by
  refine ⟨15, [.down], rfl, ⟨true, rfl, trivial⟩, ?_, trivial⟩
  rw [gapRes_g17, g17_15, val_down]

/-! ## 4. `L(17) < 2`: the bare pair is inadmissible at gears 5 and 7 -/

theorem bare_pair_inadmissible : BareAlt.bareAdmAB 6 13 2 = false := by decide

theorem gapWord_of_g17 {i : ℕ} {x y : ℕ} (hx : g17 i = x) (hy : g17 (i + 1) = y) :
    BareAlt.GapWordAt opSeq i [x, y] := by
  have h1 := opSeq_lt_succ i
  have h2 := opSeq_lt_succ (i + 1)
  unfold g17 at hx hy
  exact ⟨by omega, by omega, trivial⟩

theorem no_word2 : ¬ RealisedWord (16 : ZMod 19) opSeq 2 := by
  rintro ⟨i, w, hl, ⟨t, ht⟩, hw⟩
  rcases w with _ | ⟨a, _ | ⟨b, _ | ⟨c, w⟩⟩⟩
  · simp at hl
  · simp at hl
  · obtain ⟨h1, h2, -⟩ := hw
    have key := BareAlt.no_bare_run (E := Exposed17) opSeq_exposed blocks17_five
      blocks17_seven bare_pair_inadmissible i
    rcases letter17 h1 with ⟨rfl, e1⟩ | ⟨rfl, e1⟩
    · obtain ⟨-, hb⟩ : t = false ∧ Alt true [b] := ht
      cases b with
      | pad => rcases letter17 h2 with ⟨hc, -⟩ | ⟨hc, -⟩ <;> exact absurd hc (by decide)
      | up => exact absurd hb (by simp [Alt])
      | down =>
          rcases letter17 h2 with ⟨hc, -⟩ | ⟨-, e2⟩
          · exact absurd hc (by decide)
          · exact key.2 (gapWord_of_g17 e1 e2)
    · obtain ⟨-, hb⟩ : t = true ∧ Alt false [b] := ht
      cases b with
      | pad => rcases letter17 h2 with ⟨hc, -⟩ | ⟨hc, -⟩ <;> exact absurd hc (by decide)
      | up =>
          rcases letter17 h2 with ⟨-, e2⟩ | ⟨hc, -⟩
          · exact key.1 (gapWord_of_g17 e1 e2)
          · exact absurd hc (by decide)
      | down => exact absurd hb (by simp [Alt])
  · simp at hl

/-- **`L(17) = 1`.** -/
theorem L17 : RealisedWord (16 : ZMod 19) opSeq 1 ∧ ¬ RealisedWord (16 : ZMod 19) opSeq 2 :=
  ⟨realised1, no_word2⟩

/-- **`A_kill(17 -> 19) = 2`** (no periodicity needed). -/
theorem akill17 (k : ℕ) (hk : 1 ≤ k) : (∃ n, Chain (16 : ZMod 19) opSeq k n) ↔ k ≤ 2 :=
  akill (16 : ZMod 19) opSeq realised1 no_word2 k hk

end WordLegal17
