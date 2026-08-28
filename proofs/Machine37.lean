/-
THE SEVENTH RUNG: (D) at alpha = 3 at the 31->37 step (round 25).

Same vehicle as `Machine31.lean` one gear up.  Gear 37's teeth are `{6, 31}`
(`6 * 6 = 36 = 37 - 1`, so `37 ∣ 6k + 1` at `k ≡ 6`; `6 * 31 = 186 =
5 * 37 + 1`, so `37 ∣ 6k - 1` at `k ≡ 31`), giving `u = 6` and the qualifying
floor `2u = 12` on machine 31's gap word.

    criterion  max (F_2(31), max_j Q_j(31; 12)) = max (68, 91) = 91
    budget     F(31) + 37 = 58 + 37 = 95              margin 4

Everything is discharged except `Machine31.Census31`: machine 37's own
opening enumeration, the teeth, the containment `Exposed37 -> Exposed31`,
the kill/survive equivalences, machine 31's enumeration completeness
(`Machine31Q.lean`, no scan) and the merge-law wiring.  The two spectrum
inputs come from `Machine31Dict.lean`, kernel-decided over 43,185 explicit
tuples.

This is the second rung climbed by the dictionary vehicle and the first for
which NO period scan of either machine has ever existed in this ledger - the
23->29 rung's machine-23 scan was 37,182,145 residues and took 3 h 36 min;
machine 31's period is 33,426,748,355 slots and will never be scanned by a
kernel.
-/

import Machine31Dict

namespace Machine37

open Machine31

/-! ## Gear 37: teeth and survivors -/

/-- Gear 37 kills slot `k`: the two teeth, at `u = 6` and `37 - 6 = 31`. -/
def Killed37 (k : ℕ) : Prop := k % 37 = 6 ∨ k % 37 = 31

instance (k : ℕ) : Decidable (Killed37 k) := by unfold Killed37; infer_instance

/-- An opening of machine 37 = gears `{5,7,11,13,17,19,23,29,31,37}`. -/
def Exposed37 (k : ℕ) : Prop :=
  Exposed31 k ∧ ¬ (37 ∣ Census.lo k) ∧ ¬ (37 ∣ Census.hi k)

instance (k : ℕ) : Decidable (Exposed37 k) := by unfold Exposed37; infer_instance

/-- The teeth ARE the divisibility conditions. -/
theorem killed37_iff {k : ℕ} (hk : 1 ≤ k) :
    Killed37 k ↔ (37 ∣ Census.lo k ∨ 37 ∣ Census.hi k) := by
  simp only [Killed37, Census.lo, Census.hi]
  omega

theorem not_killed_of_exposed37 {k : ℕ} (hk : 1 ≤ k) (h : Exposed37 k) :
    ¬ Killed37 k :=
  fun hK => ((killed37_iff hk).mp hK).elim h.2.1 h.2.2

theorem exposed37_of {k : ℕ} (hk : 1 ≤ k) (h31 : Exposed31 k)
    (hnk : ¬ Killed37 k) : Exposed37 k :=
  ⟨h31, fun hd => hnk ((killed37_iff hk).mpr (Or.inl hd)),
    fun hd => hnk ((killed37_iff hk).mpr (Or.inr hd))⟩

theorem exposed31_of_37 {k : ℕ} (h : Exposed37 k) : Exposed31 k := h.1

/-! ## Machine 37's own gap sequence -/

/-- Multiples of machine 37's period
`1,236,789,689,135 = 33,426,748,355 * 37` are openings. -/
theorem exists_exposed37_above (k : ℕ) : ∃ m, k < m ∧ Exposed37 m := by
  refine ⟨1236789689135 * (k + 1), by omega,
    ⟨⟨⟨⟨?_, ?_, ?_⟩, ?_, ?_⟩, ?_, ?_⟩, ?_, ?_⟩⟩
  · rw [Machine19.exposed19_iff (by omega)]
    have h5 : (1236789689135 * (k + 1)) % 5 = 0 := by omega
    have h7 : (1236789689135 * (k + 1)) % 7 = 0 := by omega
    have h11 : (1236789689135 * (k + 1)) % 11 = 0 := by omega
    have h13 : (1236789689135 * (k + 1)) % 13 = 0 := by omega
    have h17 : (1236789689135 * (k + 1)) % 17 = 0 := by omega
    have h19 : (1236789689135 * (k + 1)) % 19 = 0 := by omega
    rw [h5, h7, h11, h13, h17, h19]
    decide
  · simp only [Census.lo]; omega
  · simp only [Census.hi]; omega
  · simp only [Census.lo]; omega
  · simp only [Census.hi]; omega
  · simp only [Census.lo]; omega
  · simp only [Census.hi]; omega
  · simp only [Census.lo]; omega
  · simp only [Census.hi]; omega

/-- The next machine-37 opening strictly after `k`. -/
def nextOp37 (k : ℕ) : ℕ := Nat.find (exists_exposed37_above k)

theorem nextOp37_gt (k : ℕ) : k < nextOp37 k :=
  (Nat.find_spec (exists_exposed37_above k)).1

theorem nextOp37_exposed (k : ℕ) : Exposed37 (nextOp37 k) :=
  (Nat.find_spec (exists_exposed37_above k)).2

theorem nextOp37_min {k m : ℕ} (h1 : k < m) (h2 : m < nextOp37 k) :
    ¬ Exposed37 m := fun hE =>
  Nat.find_min (exists_exposed37_above k) h2 ⟨h1, hE⟩

/-- The opening sequence of machine 37, in increasing order. -/
def opSeq37 : ℕ → ℕ
  | 0 => nextOp37 0
  | n + 1 => nextOp37 (opSeq37 n)

theorem opSeq37_exposed (n : ℕ) : Exposed37 (opSeq37 n) := by
  cases n <;> exact nextOp37_exposed _

theorem opSeq37_lt_succ (n : ℕ) : opSeq37 n < opSeq37 (n + 1) := nextOp37_gt _

theorem opSeq37_pos (n : ℕ) : 1 ≤ opSeq37 n := by
  cases n with
  | zero => exact nextOp37_gt 0
  | succ m =>
    have h1 := nextOp37_gt (opSeq37 m)
    have h2 : opSeq37 (m + 1) = nextOp37 (opSeq37 m) := rfl
    omega

/-- No machine-37 opening sits strictly between consecutive members. -/
theorem opSeq37_gap_empty (n : ℕ) :
    ∀ j, opSeq37 n < j → j < opSeq37 (n + 1) → ¬ Exposed37 j :=
  fun _j h1 h2 => nextOp37_min h1 h2

/-- **The gap word of machine 37.** -/
def g37 (n : ℕ) : ℕ := opSeq37 (n + 1) - opSeq37 n

/-! ## The merge alphabet at 31->37 -/

/-- **The 31->37 merge alphabet is `{12, 25, 37, 49}`**: a gap between two
gear-37-killed openings, being at most `F(31) = 58` and `0, 12 or 25` mod 37,
is one of those four - every letter meets the qualifying floor `2u = 12`. -/
theorem merge_alphabet {x y : ℕ} (hk1 : Killed37 x) (hk2 : Killed37 y)
    (hxy : x < y) (hle : y - x ≤ 58) :
    y - x = 12 ∨ y - x = 25 ∨ y - x = 37 ∨ y - x = 49 := by
  rcases hk1 with h1 | h1 <;> rcases hk2 with h2 | h2 <;> omega

/-! ## The rung -/

/-- **The 31->37 rung of the (D) ladder**, with exactly two hypotheses, both
facts about MACHINE 31's own gap word: `F_2(31) <= 68` and
`Q_j(31; 12) <= 91` at every depth `j >= 3`.

`max (68, 91) = 91 <= 95 = F(31) + 37`, margin 4. -/
theorem D_at_31_37 (hF2 : Spectrum.SpectrumBound g31 2 68)
    (hQ : ∀ j, 3 ≤ j → Spectrum.QualBound g31 6 j 91) (n : ℕ) :
    g37 n ≤ 58 + 37 :=
  MergeLaw.newgap_le_step
    (ExO := Exposed31) (ExN := Exposed37) (Kap := Killed37)
    (posO := opSeq31) (posN := opSeq37) (g := g31)
    (q := 37) (u := 6) (B := 58 + 37) (F2 := 68) (Q := fun _ => 91)
    (fun _ => rfl) opSeq31_lt_succ opSeq31_pos opSeq31_exposed
    (fun _ hx hE => Machine31.opSeq31_surj hx hE)
    opSeq37_pos opSeq37_lt_succ opSeq37_exposed
    (fun m x => opSeq37_gap_empty m x)
    (fun _ hx hE => not_killed_of_exposed37 hx hE)
    (fun _ hx hE hnk => exposed37_of hx hE hnk)
    (fun _ h => exposed31_of_37 h)
    (fun _ hk => by rcases hk with h | h <;> omega)
    (by omega) (by omega)
    hF2 (by omega) hQ (fun _ => by omega) n

/-- **R39's own form at the 31->37 step**: every gap of machine 37 is at most
`max (F_2(31), max_j Q_j(31; 12)) = 91`, inside the (D) budget 95. -/
theorem g37_le_91 (hF2 : Spectrum.SpectrumBound g31 2 68)
    (hQ : ∀ j, 3 ≤ j → Spectrum.QualBound g31 6 j 91) (n : ℕ) :
    g37 n ≤ 91 :=
  MergeLaw.newgap_le_step
    (ExO := Exposed31) (ExN := Exposed37) (Kap := Killed37)
    (posO := opSeq31) (posN := opSeq37) (g := g31)
    (q := 37) (u := 6) (B := 91) (F2 := 68) (Q := fun _ => 91)
    (fun _ => rfl) opSeq31_lt_succ opSeq31_pos opSeq31_exposed
    (fun _ hx hE => Machine31.opSeq31_surj hx hE)
    opSeq37_pos opSeq37_lt_succ opSeq37_exposed
    (fun m x => opSeq37_gap_empty m x)
    (fun _ hx hE => not_killed_of_exposed37 hx hE)
    (fun _ hx hE hnk => exposed37_of hx hE hnk)
    (fun _ h => exposed31_of_37 h)
    (fun _ hk => by rcases hk with h | h <;> omega)
    (by omega) (by omega)
    hF2 (by omega) hQ (fun _ => le_refl 91) n

/-- **(D) AT 31->37 FROM THE DICTIONARY** - the seventh rung.  Single
hypothesis `Machine31.Census31`. -/
theorem D_31_37 (h : Census31) (n : ℕ) : g37 n ≤ 58 + 37 :=
  D_at_31_37 (spectrum31_two h) (qual31_all h) n

/-- The same rung in R39's form: `g37 <= 91`. -/
theorem g37_le_of_census (h : Census31) (n : ℕ) : g37 n ≤ 91 :=
  g37_le_91 (spectrum31_two h) (qual31_all h) n

end Machine37
