/-
R89 AT A MACHINE: `L(11) = 1`, `J_max(11) = 3`, `A_kill(11 -> 13) = 2` IN THE KERNEL
(Formalist, round 30).

`WordLegal.lean` proves R89 for an abstract opening enumeration.  Machine 11
(`{5, 7, 11}`, period 385, 135 gaps) has its enumeration and gap word in the
ledger (`Machine11.opSeq`, `Machine11.g11`, `Machine11.g11_shift`), and gear 13
has tooth `u' = 11` (`6 * 11 = 66 = 5 * 13 + 1`), so the legal letters for
13 are `0`, `2 * 11 = 9` and `-22 = 4` mod 13.  Every gap of machine 11 is at
most 7 (`Machine11.spectrum_one`), so the ONLY legal letter is the gap 4
(class `down`), two of them never alternate, and `L(11) = 1` exactly:

    theorem L11    : RealisedWord 11 opSeq 1 ∧ ¬ RealisedWord 11 opSeq 2
    theorem jmax11 : QstarNonempty 11 opSeq J ↔ J ≤ 3          -- J_max(11) = 3
    theorem akill11: (∃ n, Chain 11 opSeq k n) ↔ k ≤ 2  (1 ≤ k)  -- A_kill(11->13) = 2

which is R81's first table row (m11: J_max 3, A_kill + 1 = 3) - the first time
that row is a kernel fact rather than a census.
-/

import WordLegal
import Machine11Per

namespace WordLegal11

open WordLegal Machine11

/-- Gear 13's tooth is `11`: `6 * 11 = 1` in `ZMod 13`. -/
theorem tooth13 : (6 : ZMod 13) * 11 = 1 := by decide

theorem val_up : Cls.val (11 : ZMod 13) .up = ((9 : ℕ) : ZMod 13) := by decide
theorem val_down : Cls.val (11 : ZMod 13) .down = ((4 : ℕ) : ZMod 13) := by decide

/-- Every gap of machine 11 is between 1 and 7. -/
theorem g11_bounds (n : ℕ) : 1 ≤ g11 n ∧ g11 n ≤ 7 := by
  constructor
  · have := opSeq_lt_succ n
    unfold g11
    omega
  · have := spectrum_one n
    simpa [Spectrum.windowSum] using this

theorem gapRes_g11 (n : ℕ) : gapRes (q := 13) opSeq n = ((g11 n : ℕ) : ZMod 13) :=
  gapRes_eq_cast opSeq (opSeq_lt_succ n).le

/-- A legal letter of machine 11 (for gear 13) is the gap `4`, class `down`. -/
theorem letter11 {n : ℕ} {a : Cls} (h : gapRes (q := 13) opSeq n = Cls.val (11 : ZMod 13) a) :
    a = .down ∧ g11 n = 4 := by
  rw [gapRes_g11] at h
  obtain ⟨h1, h7⟩ := g11_bounds n
  cases a with
  | pad =>
      simp only [Cls.val] at h
      have hd := (ZMod.natCast_eq_zero_iff (g11 n) 13).mp h
      have := Nat.le_of_dvd h1 hd
      exfalso
      omega
  | up =>
      rw [val_up, ZMod.natCast_eq_natCast_iff'] at h
      exfalso
      omega
  | down =>
      rw [val_down, ZMod.natCast_eq_natCast_iff'] at h
      exact ⟨rfl, by omega⟩

theorem g11_12 : g11 12 = 4 := by
  rw [g11_eq_ow]
  decide +kernel

/-- `L(11) >= 1`: the gap `4` at index 12 is a realised legal word of length 1. -/
theorem realised1 : RealisedWord (11 : ZMod 13) opSeq 1 := by
  refine ⟨12, [.down], rfl, ⟨true, rfl, trivial⟩, ?_, trivial⟩
  rw [gapRes_g11, g11_12, val_down]

/-- `L(11) < 2`: two consecutive legal letters would both be `down`. -/
theorem no_word2 : ¬ RealisedWord (11 : ZMod 13) opSeq 2 := by
  rintro ⟨i, w, hl, ⟨t, ht⟩, hw⟩
  rcases w with _ | ⟨a, _ | ⟨b, _ | ⟨c, w⟩⟩⟩
  · simp at hl
  · simp at hl
  · obtain ⟨h1, h2, -⟩ := hw
    obtain ⟨rfl, -⟩ := letter11 h1
    obtain ⟨rfl, -⟩ := letter11 h2
    obtain ⟨rfl, h⟩ : t = true ∧ Alt false [.down] := ht
    obtain ⟨h', -⟩ : false = true ∧ Alt false [] := h
    exact absurd h' (by decide)
  · simp at hl

/-- **`L(11) = 1`.** -/
theorem L11 : RealisedWord (11 : ZMod 13) opSeq 1 ∧ ¬ RealisedWord (11 : ZMod 13) opSeq 2 :=
  ⟨realised1, no_word2⟩

/-- The gap residues of machine 11 are periodic with period 135. -/
theorem hper11 (n : ℕ) : gapRes (q := 13) opSeq (n + 135) = gapRes opSeq n := by
  rw [gapRes_g11, gapRes_g11, g11_shift]

/-- **`J_max(11) = 3`**: a word-legal `J`-window of machine 11 (for gear 13)
exists exactly for `J <= 3`. -/
theorem jmax11 (J : ℕ) : QstarNonempty (11 : ZMod 13) opSeq J ↔ J ≤ 3 :=
  jmax (11 : ZMod 13) opSeq hper11 (by norm_num) realised1 no_word2 J

/-- **`A_kill(11 -> 13) = 2`**: a `k`-chain of machine 11 under gear 13 exists
exactly for `1 <= k <= 2`. -/
theorem akill11 (k : ℕ) (hk : 1 ≤ k) : (∃ n, Chain (11 : ZMod 13) opSeq k n) ↔ k ≤ 2 :=
  akill (11 : ZMod 13) opSeq realised1 no_word2 k hk

end WordLegal11
