/-
THE INCREMENT LAW AS A KERNEL STATEMENT, AT ALL SIX LITERAL STEPS (round 28).

THE LAW (manager, round 26; the base cases of the literal-step induction):

    F(M + q')  -  F_2(M)   <=   s_min(q'),
    s_min(q') = min(2u' mod q', -2u' mod q'),   6u' = q' -+ 1.

It has two halves and they need different kinds of evidence.

  * THE UPPER HALF, `F(M + q') <= F_2(M) + s_min(q')`, is a covering statement.
    The LP thread certified it in round 27 at all six literal steps by running
    the case-split vehicle at the INCREMENT WIDTH `W_inc = F_2(M) + s_min(q')`,
    strictly smaller than the (D) ladder's budget width `F(M) + q'`.  Three of
    those six are kernelised here as `IncCert23`, `IncCert29`, `IncCert31`
    (35 exact dual certificates each).  AT THE OTHER THREE THE CERTIFICATE IS
    NOT NEEDED: the corpus already carries a STRICTLY TIGHTER kernel bound on
    F(q') than W_inc (11 < 15 at machine 13, 18 < 22 at 17, 25 < 31 at 19), so
    those steps are discharged from the existing spectrum ladders.  That is a
    finding, not a shortcut - the increment width is slack at the small
    machines and knife-edge at the large ones.

  * THE LOWER HALF, `F_2(M) >= its claimed value`, is a REALISABILITY statement
    and no dual certificate can carry it.  It needs an exhibited configuration.
    The LP thread emitted six, as phase vectors found by exact-cover backtrack
    with no period scan; CRT turns each phase vector into a single slot of the
    real machine, and that slot is what is checked here.  Machine 13's was
    already in the ledger (`Machine13.pair16_realized`, round 11).

So each theorem below is SELF-CONTAINED: it exhibits three consecutive openings
of the old machine spanning `c - a`, and bounds every gap of the new machine by
`(c - a) + s_min`.  No `F_2` symbol, no census, no period scan.

    step      s_min   F_2(M)   W_inc   upper half from
    11 -> 13     4      11       15    Machine13.spectrum_one  (11 <= 15)
    13 -> 17     6      16       22    Machine17.spectrum_one  (18 <= 22)
    17 -> 19     6      25       31    Machine19.spectrum_one  (25 <= 31)
    19 -> 23     8      31       39    IncCert23.F_le          (35 certificates)
    23 -> 29    10      39       49    IncCert29.F_le          (35 certificates)
    29 -> 31    10      55       65    IncCert31.F_le          (35 certificates)
-/

import IncCert23
import IncCert29
import IncCert31
import Machine13Q
import Machine17Q
import Machine19Q
import Machine13
import Machine11

namespace Increment

set_option maxRecDepth 8192

/-- **An adjacent gap pair, exhibited.**  `a`, `b`, `c` are three CONSECUTIVE
openings of the machine whose opening predicate is `E`, so `c - a` is a
realised value of that machine's two-gap sum and `F_2 >= c - a`. -/
def AdjPair (E : ℕ → Prop) (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ E a ∧ E b ∧ E c ∧
    (∀ j, a < j → j < b → ¬ E j) ∧ (∀ j, b < j → j < c → ¬ E j)

/-! ## 1. The realisers (the lower half) -/

/-- `F_2(11) >= 11`: openings 252, 257, 263 of machine 11, gaps `(5, 6)`. -/
theorem f2_11 : AdjPair Machine11.Exposed11 252 257 263 := by
  refine ⟨by omega, by omega, by omega, by decide, by decide, by decide, ?_, ?_⟩
  · intro j h1 h2; interval_cases j <;> decide
  · intro j h1 h2; interval_cases j <;> decide

/-- `F_2(13) >= 16`: round 11's `Machine13.pair16_realized`, openings 117, 122,
133, gaps `(5, 11)`.  The one realiser this project already had. -/
theorem f2_13 : AdjPair Machine13.Exposed13 117 122 133 := by
  obtain ⟨h1, h2, h3, h4, h5⟩ := Machine13.pair16_realized
  exact ⟨by omega, by omega, by omega, h1, h2, h3, h4, h5⟩

/-- `F_2(17) >= 25`: openings 110, 117, 135 of machine 17, gaps `(7, 18)`. -/
theorem f2_17 : AdjPair Machine17.Exposed17 110 117 135 := by
  refine ⟨by omega, by omega, by omega, by decide, by decide, by decide, ?_, ?_⟩
  · intro j h1 h2; interval_cases j <;> decide
  · intro j h1 h2; interval_cases j <;> decide

/-- `F_2(19) >= 31`: openings 1118917, 1118927, 1118948 of machine 19, gaps
`(10, 21)` - which is exactly the split the LP thread's windowed vehicle
located from the DUAL side in round 26. -/
theorem f2_19 : AdjPair Machine19.Exposed19 1118917 1118927 1118948 := by
  refine ⟨by omega, by omega, by omega,
    (Machine19.exposed19_iff (by omega)).mpr (by decide),
    (Machine19.exposed19_iff (by omega)).mpr (by decide),
    (Machine19.exposed19_iff (by omega)).mpr (by decide), ?_, ?_⟩
  · intro j h1 h2
    rw [Machine19.exposed19_iff (by omega)]
    interval_cases j <;> decide
  · intro j h1 h2
    rw [Machine19.exposed19_iff (by omega)]
    interval_cases j <;> decide

/-- `F_2(23) >= 39`: openings 19016898, 19016903, 19016937 of machine 23, gaps
`(5, 34)`. -/
theorem f2_23 : AdjPair Machine23.Exposed23 19016898 19016903 19016937 := by
  refine ⟨by omega, by omega, by omega, by decide, by decide, by decide, ?_, ?_⟩
  · intro j h1 h2; interval_cases j <;> decide
  · intro j h1 h2; interval_cases j <;> decide

/-- `F_2(29) >= 55`: openings 858386140, 858386160, 858386195 of machine 29,
gaps `(20, 35)`.  This reproduces the project's `F_2(29) = 55` - a full-period
census number - from a single slot. -/
theorem f2_29 : AdjPair Machine29.Exposed29 858386140 858386160 858386195 := by
  refine ⟨by omega, by omega, by omega, by decide, by decide, by decide, ?_, ?_⟩
  · intro j h1 h2; interval_cases j <;> decide
  · intro j h1 h2; interval_cases j <;> decide

/-! ## 2. The six steps -/

theorem g13_le_11 (n : ℕ) : Machine13.g13 n ≤ 11 := by
  have h := Machine13.spectrum_one n
  simpa [Spectrum.windowSum] using h

theorem g17_le_18 (n : ℕ) : Machine17.g17 n ≤ 18 := by
  have h := Machine17.spectrum_one n
  simpa [Spectrum.windowSum] using h

theorem g19_le_25 (n : ℕ) : Machine19.g19 n ≤ 25 := by
  have h := Machine19.spectrum_one n
  simpa [Spectrum.windowSum] using h

/-- **THE INCREMENT LAW AT 11 -> 13**, `s_min(13) = 4`.  Slack: the kernel
bound `F(13) <= 11` is four below the increment width 15. -/
theorem increment_11_13 :
    ∃ a b c, AdjPair Machine11.Exposed11 a b c ∧ c - a = 11 ∧
      ∀ n, Machine13.g13 n ≤ (c - a) + 4 :=
  ⟨252, 257, 263, f2_11, by norm_num, fun n => by have := g13_le_11 n; omega⟩

/-- **THE INCREMENT LAW AT 13 -> 17**, `s_min(17) = 6`. -/
theorem increment_13_17 :
    ∃ a b c, AdjPair Machine13.Exposed13 a b c ∧ c - a = 16 ∧
      ∀ n, Machine17.g17 n ≤ (c - a) + 6 :=
  ⟨117, 122, 133, f2_13, by norm_num, fun n => by have := g17_le_18 n; omega⟩

/-- **THE INCREMENT LAW AT 17 -> 19**, `s_min(19) = 6`. -/
theorem increment_17_19 :
    ∃ a b c, AdjPair Machine17.Exposed17 a b c ∧ c - a = 25 ∧
      ∀ n, Machine19.g19 n ≤ (c - a) + 6 :=
  ⟨110, 117, 135, f2_17, by norm_num, fun n => by have := g19_le_25 n; omega⟩

/-- **THE INCREMENT LAW AT 19 -> 23**, `s_min(23) = 8`.  The upper half is
`IncCert23.F_le`: 35 exact dual certificates at the increment width 39, which
is nine below the (D) rung's budget width 48. -/
theorem increment_19_23 :
    ∃ a b c, AdjPair Machine19.Exposed19 a b c ∧ c - a = 31 ∧
      ∀ n, Machine23.g23 n ≤ (c - a) + 8 :=
  ⟨1118917, 1118927, 1118948, f2_19, by norm_num,
    fun n => by have := IncCert23.F_le n; omega⟩

/-- **THE INCREMENT LAW AT 23 -> 29**, `s_min(29) = 10`. -/
theorem increment_23_29 :
    ∃ a b c, AdjPair Machine23.Exposed23 a b c ∧ c - a = 39 ∧
      ∀ n, Machine29.g29 n ≤ (c - a) + 10 :=
  ⟨19016898, 19016903, 19016937, f2_23, by norm_num,
    fun n => by have := IncCert29.F_le n; omega⟩

/-- **THE INCREMENT LAW AT 29 -> 31**, `s_min(31) = 10`.  The old machine's
realiser is a single slot of machine 29 and the new machine's bound is 35 exact
dual certificates - so a statement about two machines of period 1.08e9 and
3.34e10 is checked without either period being touched. -/
theorem increment_29_31 :
    ∃ a b c, AdjPair Machine29.Exposed29 a b c ∧ c - a = 55 ∧
      ∀ n, Machine31.g31 n ≤ (c - a) + 10 :=
  ⟨858386140, 858386160, 858386195, f2_29, by norm_num,
    fun n => by have := IncCert31.F_le n; omega⟩

/-! ## 3. The realisers as INDICES: the ledger's `F_2` hypotheses are SHARP

The merge-law rungs `Machine29.g29_le` and `Machine31.g31_le_71` each carry a
census hypothesis of the form `SpectrumBound g_M 2 F_2` - an UPPER bound on the
old machine's two-gap record.  The realisers above are LOWER bounds on the same
quantity, so together they pin it: those hypotheses cannot be stated with a
smaller constant.  Turning a realiser into an index costs one abstract lemma
plus each machine's enumeration completeness (`opSeq_surj`).
-/

/-- **A realised adjacent pair is a realised pair of the gap word.**  Abstract
in the machine: the only inputs are `next`'s three defining properties and the
enumeration's completeness. -/
theorem pair_attained {E : ℕ → Prop} {next op : ℕ → ℕ} {a b c : ℕ}
    (hsucc : ∀ n, op (n + 1) = next (op n))
    (hgt : ∀ k, k < next k) (hEn : ∀ k, E (next k))
    (hmin : ∀ k m, k < m → m < next k → ¬ E m)
    (hsurj : ∀ m, 1 ≤ m → E m → ∃ n, op n = m)
    (hw : AdjPair E a b c) :
    ∃ n, op (n + 2) - op n = c - a := by
  obtain ⟨h1a, hab, hbc, hEa, hEb, hEc, hlow, hhigh⟩ := hw
  obtain ⟨n, hn⟩ := hsurj a h1a hEa
  have hnb : next a = b := by
    rcases Nat.lt_trichotomy (next a) b with h | h | h
    · exact absurd (hEn a) (hlow _ (hgt a) h)
    · exact h
    · exact absurd hEb (hmin a b hab h)
  have hnc : next b = c := by
    rcases Nat.lt_trichotomy (next b) c with h | h | h
    · exact absurd (hEn b) (hhigh _ (hgt b) h)
    · exact h
    · exact absurd hEc (hmin b c hbc h)
  refine ⟨n, ?_⟩
  have e1 : op (n + 1) = b := by rw [hsucc, hn, hnb]
  have e2 : op (n + 2) = c := by
    rw [show n + 2 = (n + 1) + 1 by omega, hsucc, e1, hnc]
  rw [e2, hn]

theorem f2_19_index : ∃ n, Machine19.g19 n + Machine19.g19 (n + 1) = 31 := by
  obtain ⟨n, hn⟩ := pair_attained (E := Machine19.Exposed19)
    Machine19.opSeq_succ Machine19.nextOp_gt Machine19.nextOp_exposed
    (fun _ _ h1 h2 => Machine19.nextOp_min h1 h2)
    (fun _ h1 h2 => Machine19.opSeq_surj h1 h2) f2_19
  have m1 := Machine19.opSeq_lt_succ n
  have m2 := Machine19.opSeq_lt_succ (n + 1)
  refine ⟨n, ?_⟩
  have e : Machine19.g19 n + Machine19.g19 (n + 1)
      = (Machine19.opSeq (n + 1) - Machine19.opSeq n)
        + (Machine19.opSeq (n + 1 + 1) - Machine19.opSeq (n + 1)) := rfl
  have h2 : Machine19.opSeq (n + 1 + 1) = Machine19.opSeq (n + 2) := by
    rw [show n + 1 + 1 = n + 2 by omega]
  rw [h2] at e m2
  omega

theorem f2_23_index : ∃ n, Machine23.g23 n + Machine23.g23 (n + 1) = 39 := by
  obtain ⟨n, hn⟩ := pair_attained (E := Machine23.Exposed23)
    Machine23.opSeq23_succ Machine23.nextOp23_gt Machine23.nextOp23_exposed
    (fun _ _ h1 h2 => Machine23.nextOp23_min h1 h2)
    (fun _ h1 h2 => Machine23.opSeq23_surj h1 h2) f2_23
  have m1 := Machine23.opSeq23_lt_succ n
  have m2 := Machine23.opSeq23_lt_succ (n + 1)
  refine ⟨n, ?_⟩
  have e : Machine23.g23 n + Machine23.g23 (n + 1)
      = (Machine23.opSeq23 (n + 1) - Machine23.opSeq23 n)
        + (Machine23.opSeq23 (n + 1 + 1) - Machine23.opSeq23 (n + 1)) := rfl
  have h2 : Machine23.opSeq23 (n + 1 + 1) = Machine23.opSeq23 (n + 2) := by
    rw [show n + 1 + 1 = n + 2 by omega]
  rw [h2] at e m2
  omega

theorem f2_29_index : ∃ n, Machine29.g29 n + Machine29.g29 (n + 1) = 55 := by
  obtain ⟨n, hn⟩ := pair_attained (E := Machine29.Exposed29)
    Machine29.opSeq29_succ Machine29.nextOp29_gt Machine29.nextOp29_exposed
    (fun _ _ h1 h2 => Machine29.nextOp29_min h1 h2)
    (fun _ h1 h2 => Machine29.opSeq29_surj h1 h2) f2_29
  have m1 := Machine29.opSeq29_lt_succ n
  have m2 := Machine29.opSeq29_lt_succ (n + 1)
  refine ⟨n, ?_⟩
  have e : Machine29.g29 n + Machine29.g29 (n + 1)
      = (Machine29.opSeq29 (n + 1) - Machine29.opSeq29 n)
        + (Machine29.opSeq29 (n + 1 + 1) - Machine29.opSeq29 (n + 1)) := rfl
  have h2 : Machine29.opSeq29 (n + 1 + 1) = Machine29.opSeq29 (n + 2) := by
    rw [show n + 1 + 1 = n + 2 by omega]
  rw [h2] at e m2
  omega

/-- **The `F_2(19) <= 31` input is sharp.** -/
theorem f2_19_sharp : ¬ Spectrum.SpectrumBound Machine19.g19 2 30 := by
  intro h
  obtain ⟨n, hn⟩ := f2_19_index
  have hb := h n
  have hw : Spectrum.windowSum Machine19.g19 n 2
      = Machine19.g19 n + Machine19.g19 (n + 1) := by
    simp [Spectrum.windowSum, Finset.sum_range_succ]
  omega

/-- **The census hypothesis of `Machine29.g29_le` is sharp**: `SpectrumBound
g23 2 39` cannot be stated with 38. -/
theorem f2_23_sharp : ¬ Spectrum.SpectrumBound Machine23.g23 2 38 := by
  intro h
  obtain ⟨n, hn⟩ := f2_23_index
  have hb := h n
  have hw : Spectrum.windowSum Machine23.g23 n 2
      = Machine23.g23 n + Machine23.g23 (n + 1) := by
    simp [Spectrum.windowSum, Finset.sum_range_succ]
  omega

/-- **The census hypothesis of `Machine31.g31_le_71` is sharp**: `SpectrumBound
g29 2 55` cannot be stated with 54. -/
theorem f2_29_sharp : ¬ Spectrum.SpectrumBound Machine29.g29 2 54 := by
  intro h
  obtain ⟨n, hn⟩ := f2_29_index
  have hb := h n
  have hw : Spectrum.windowSum Machine29.g29 n 2
      = Machine29.g29 n + Machine29.g29 (n + 1) := by
    simp [Spectrum.windowSum, Finset.sum_range_succ]
  omega

/-- **THE INCREMENT LAW IN ITS SHARPEST KERNEL FORM AT 23 -> 29**: there is an
index of machine 23 whose two-gap sum bounds every gap of machine 29 up to
`s_min(29) = 10`.  No constant on the right that is not itself a realised
quantity of the old machine. -/
theorem increment_23_29_index :
    ∃ i, ∀ n, Machine29.g29 n ≤
      (Machine23.g23 i + Machine23.g23 (i + 1)) + 10 := by
  obtain ⟨i, hi⟩ := f2_23_index
  exact ⟨i, fun n => by have := IncCert29.F_le n; omega⟩

/-- The same at 29 -> 31. -/
theorem increment_29_31_index :
    ∃ i, ∀ n, Machine31.g31 n ≤
      (Machine29.g29 i + Machine29.g29 (i + 1)) + 10 := by
  obtain ⟨i, hi⟩ := f2_29_index
  exact ⟨i, fun n => by have := IncCert31.F_le n; omega⟩

/-- The same at 19 -> 23. -/
theorem increment_19_23_index :
    ∃ i, ∀ n, Machine23.g23 n ≤
      (Machine19.g19 i + Machine19.g19 (i + 1)) + 8 := by
  obtain ⟨i, hi⟩ := f2_19_index
  exact ⟨i, fun n => by have := IncCert23.F_le n; omega⟩

/-- **THE INCREMENT LAW, KERNEL-CHECKED AT EVERY LITERAL STEP OF THE LADDER.**
Six steps, each hypothesis-free: an exhibited adjacent gap pair of the old
machine and a bound on every gap of the new one. -/
theorem increment_law_literal_steps :
    (∃ a b c, AdjPair Machine11.Exposed11 a b c ∧ c - a = 11 ∧
        ∀ n, Machine13.g13 n ≤ (c - a) + 4) ∧
      (∃ a b c, AdjPair Machine13.Exposed13 a b c ∧ c - a = 16 ∧
        ∀ n, Machine17.g17 n ≤ (c - a) + 6) ∧
      (∃ a b c, AdjPair Machine17.Exposed17 a b c ∧ c - a = 25 ∧
        ∀ n, Machine19.g19 n ≤ (c - a) + 6) ∧
      (∃ a b c, AdjPair Machine19.Exposed19 a b c ∧ c - a = 31 ∧
        ∀ n, Machine23.g23 n ≤ (c - a) + 8) ∧
      (∃ a b c, AdjPair Machine23.Exposed23 a b c ∧ c - a = 39 ∧
        ∀ n, Machine29.g29 n ≤ (c - a) + 10) ∧
      (∃ a b c, AdjPair Machine29.Exposed29 a b c ∧ c - a = 55 ∧
        ∀ n, Machine31.g31 n ≤ (c - a) + 10) :=
  ⟨increment_11_13, increment_13_17, increment_17_19, increment_19_23,
    increment_23_29, increment_29_31⟩

end Increment
