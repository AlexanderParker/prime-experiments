/-
THE SIXTH RUNG: (D) at alpha = 3 at the 29->31 step, from an EXPLICIT FINITE
DICTIONARY instead of a period scan (round 25).

Round-24 verdict 17 measured the per-rung period-scan vehicle dead here: the
factorisation that gave the 23->29 rung costs ~170 h at 29->31 (7,429 outer
slices with no reusable slice family, a 29-fold inner loop).  Verdict 15 named
the replacement - "the longest-path value over an EXPLICIT edge set, with
`E contains every realised tuple` as a named hypothesis a census discharges" -
and this file is the first rung climbed by it.

Everything here IS discharged: machine 31's own opening enumeration, its teeth
(`u = 5`: `6 * 5 = 30 = 31 - 1` and `6 * 26 = 156 = 5 * 31 + 1`, so gear 31
kills slot residues 5 and 26), the containment `Exposed31 -> Exposed29`,
the kill/survive equivalences, machine 29's enumeration completeness
(`Machine29.opSeq29_surj`, `Machine29Q.lean`, no scan) and the merge-law
wiring.  The two spectrum inputs come from `Machine29Dict.lean`, where they
are kernel-decided over 15,860 explicit tuples.

So `D_29_31` below is (D) at the 29->31 step as a theorem about `g31`, with
EXACTLY ONE named hypothesis: `Machine29.Census29`, the statement that those
15,860 tuples contain every realised qualifying window of machine 29 and that
no six consecutive gaps of machine 29 reach 10.  Nothing else is assumed.

    criterion  max (F_2(29), max_j Q_j(29; 10)) = max (55, 71) = 71
    budget     F(29) + 31 = 43 + 31 = 74            margin 3
-/

import Machine29Dict

namespace Machine31

open Machine29

/-! ## Gear 31: teeth and survivors -/

/-- Gear 31 kills slot `k`: the two teeth, at `u = 5` and `31 - 5 = 26`
(`6 * 5 = 30 = 31 - 1` and `6 * 26 = 156 = 5 * 31 + 1`). -/
def Killed31 (k : ℕ) : Prop := k % 31 = 5 ∨ k % 31 = 26

instance (k : ℕ) : Decidable (Killed31 k) := by unfold Killed31; infer_instance

/-- An opening of machine 31 = gears `{5, 7, 11, 13, 17, 19, 23, 29, 31}`. -/
def Exposed31 (k : ℕ) : Prop :=
  Exposed29 k ∧ ¬ (31 ∣ Census.lo k) ∧ ¬ (31 ∣ Census.hi k)

instance (k : ℕ) : Decidable (Exposed31 k) := by unfold Exposed31; infer_instance

/-- The teeth ARE the divisibility conditions. -/
theorem killed31_iff {k : ℕ} (hk : 1 ≤ k) :
    Killed31 k ↔ (31 ∣ Census.lo k ∨ 31 ∣ Census.hi k) := by
  simp only [Killed31, Census.lo, Census.hi]
  omega

theorem not_killed_of_exposed31 {k : ℕ} (hk : 1 ≤ k) (h : Exposed31 k) :
    ¬ Killed31 k :=
  fun hK => ((killed31_iff hk).mp hK).elim h.2.1 h.2.2

theorem exposed31_of {k : ℕ} (hk : 1 ≤ k) (h29 : Exposed29 k)
    (hnk : ¬ Killed31 k) : Exposed31 k :=
  ⟨h29, fun hd => hnk ((killed31_iff hk).mpr (Or.inl hd)),
    fun hd => hnk ((killed31_iff hk).mpr (Or.inr hd))⟩

theorem exposed29_of_31 {k : ℕ} (h : Exposed31 k) : Exposed29 k := h.1

/-! ## Machine 31's own gap sequence -/

/-- Multiples of machine 31's period
`33,426,748,355 = 1,078,282,205 * 31` are openings, so an opening exists
above any point. -/
theorem exists_exposed31_above (k : ℕ) : ∃ m, k < m ∧ Exposed31 m := by
  refine ⟨33426748355 * (k + 1), by omega, ⟨⟨⟨?_, ?_, ?_⟩, ?_, ?_⟩, ?_, ?_⟩⟩
  · rw [Machine19.exposed19_iff (by omega)]
    have h5 : (33426748355 * (k + 1)) % 5 = 0 := by omega
    have h7 : (33426748355 * (k + 1)) % 7 = 0 := by omega
    have h11 : (33426748355 * (k + 1)) % 11 = 0 := by omega
    have h13 : (33426748355 * (k + 1)) % 13 = 0 := by omega
    have h17 : (33426748355 * (k + 1)) % 17 = 0 := by omega
    have h19 : (33426748355 * (k + 1)) % 19 = 0 := by omega
    rw [h5, h7, h11, h13, h17, h19]
    decide
  · simp only [Census.lo]; omega
  · simp only [Census.hi]; omega
  · simp only [Census.lo]; omega
  · simp only [Census.hi]; omega
  · simp only [Census.lo]; omega
  · simp only [Census.hi]; omega

/-- The next machine-31 opening strictly after `k`. -/
def nextOp31 (k : ℕ) : ℕ := Nat.find (exists_exposed31_above k)

theorem nextOp31_gt (k : ℕ) : k < nextOp31 k :=
  (Nat.find_spec (exists_exposed31_above k)).1

theorem nextOp31_exposed (k : ℕ) : Exposed31 (nextOp31 k) :=
  (Nat.find_spec (exists_exposed31_above k)).2

theorem nextOp31_min {k m : ℕ} (h1 : k < m) (h2 : m < nextOp31 k) :
    ¬ Exposed31 m := fun hE =>
  Nat.find_min (exists_exposed31_above k) h2 ⟨h1, hE⟩

/-- The opening sequence of machine 31, in increasing order. -/
def opSeq31 : ℕ → ℕ
  | 0 => nextOp31 0
  | n + 1 => nextOp31 (opSeq31 n)

theorem opSeq31_exposed (n : ℕ) : Exposed31 (opSeq31 n) := by
  cases n <;> exact nextOp31_exposed _

theorem opSeq31_lt_succ (n : ℕ) : opSeq31 n < opSeq31 (n + 1) := nextOp31_gt _

theorem opSeq31_pos (n : ℕ) : 1 ≤ opSeq31 n := by
  cases n with
  | zero => exact nextOp31_gt 0
  | succ m =>
    have h1 := nextOp31_gt (opSeq31 m)
    have h2 : opSeq31 (m + 1) = nextOp31 (opSeq31 m) := rfl
    omega

/-- No machine-31 opening sits strictly between consecutive members. -/
theorem opSeq31_gap_empty (n : ℕ) :
    ∀ j, opSeq31 n < j → j < opSeq31 (n + 1) → ¬ Exposed31 j :=
  fun _j h1 h2 => nextOp31_min h1 h2

/-- **The gap word of machine 31.** -/
def g31 (n : ℕ) : ℕ := opSeq31 (n + 1) - opSeq31 n

/-! ## The merge alphabet at 29->31 -/

/-- **The 29->31 merge alphabet is `{10, 21, 31, 41}`**: a gap between two
gear-31-killed openings, being at most `F(29) = 43` and `0, 10 or 21` mod 31,
is 10, 21, 31 or 41 - every letter meets the qualifying floor `2u = 10`.
Note the FOURTH letter: at 23->29 the budget `F(23) = 34` left room for only
one representative of each residue, but `F(29) = 43 > 10 + 31`, so the merge
alphabet gains a repeat.  (Not load-bearing: `MergeLaw.newgap_le` derives the
floor itself from residue necessity.  Recorded because it is the concrete
content at this step, and because the growth of the alphabet is exactly what
makes the residue argument alone weaker at each rung.) -/
theorem merge_alphabet {x y : ℕ} (hk1 : Killed31 x) (hk2 : Killed31 y)
    (hxy : x < y) (hle : y - x ≤ 43) :
    y - x = 10 ∨ y - x = 21 ∨ y - x = 31 ∨ y - x = 41 := by
  rcases hk1 with h1 | h1 <;> rcases hk2 with h2 | h2 <;> omega

/-! ## The rung -/

/-- **The 29->31 rung of the (D) ladder**, with exactly two hypotheses, both
facts about MACHINE 29's own gap word: `F_2(29) <= 55` and
`Q_j(29; 10) <= 71` at every depth `j >= 3`.

`max (55, 71) = 71 <= 74 = F(29) + 31`, margin 3. -/
theorem D_at_29_31 (hF2 : Spectrum.SpectrumBound g29 2 55)
    (hQ : ∀ j, 3 ≤ j → Spectrum.QualBound g29 5 j 71) (n : ℕ) :
    g31 n ≤ 43 + 31 :=
  MergeLaw.newgap_le_step
    (ExO := Exposed29) (ExN := Exposed31) (Kap := Killed31)
    (posO := opSeq29) (posN := opSeq31) (g := g29)
    (q := 31) (u := 5) (B := 43 + 31) (F2 := 55) (Q := fun _ => 71)
    (fun _ => rfl) opSeq29_lt_succ opSeq29_pos opSeq29_exposed
    (fun _ hx hE => Machine29.opSeq29_surj hx hE)
    opSeq31_pos opSeq31_lt_succ opSeq31_exposed
    (fun m x => opSeq31_gap_empty m x)
    (fun _ hx hE => not_killed_of_exposed31 hx hE)
    (fun _ hx hE hnk => exposed31_of hx hE hnk)
    (fun _ h => exposed29_of_31 h)
    (fun _ hk => by rcases hk with h | h <;> omega)
    (by omega) (by omega)
    hF2 (by omega) hQ (fun _ => by omega) n

/-- **R39's own form at the 29->31 step**: every gap of machine 31 is at most
`max (F_2(29), max_j Q_j(29; 10)) = 71`, strictly inside the (D) budget 74. -/
theorem g31_le_71 (hF2 : Spectrum.SpectrumBound g29 2 55)
    (hQ : ∀ j, 3 ≤ j → Spectrum.QualBound g29 5 j 71) (n : ℕ) :
    g31 n ≤ 71 :=
  MergeLaw.newgap_le_step
    (ExO := Exposed29) (ExN := Exposed31) (Kap := Killed31)
    (posO := opSeq29) (posN := opSeq31) (g := g29)
    (q := 31) (u := 5) (B := 71) (F2 := 55) (Q := fun _ => 71)
    (fun _ => rfl) opSeq29_lt_succ opSeq29_pos opSeq29_exposed
    (fun _ hx hE => Machine29.opSeq29_surj hx hE)
    opSeq31_pos opSeq31_lt_succ opSeq31_exposed
    (fun m x => opSeq31_gap_empty m x)
    (fun _ hx hE => not_killed_of_exposed31 hx hE)
    (fun _ hx hE hnk => exposed31_of hx hE hnk)
    (fun _ h => exposed29_of_31 h)
    (fun _ hk => by rcases hk with h | h <;> omega)
    (by omega) (by omega)
    hF2 (by omega) hQ (fun _ => le_refl 71) n

/-- **(D) AT 29->31 FROM THE DICTIONARY** - the sixth rung, and the first
climbed without a period scan of either machine.  The single hypothesis
`Machine29.Census29` is the census statement that the 15,860 explicit tuples
of `Machine29D2..D7` contain every realised qualifying window of machine 29,
and that no six consecutive machine-29 gaps reach 10.  Everything above that
line - the two spectrum bounds, the merge law, both enumerations - is
kernel-checked. -/
theorem D_29_31 (h : Census29) (n : ℕ) : g31 n ≤ 43 + 31 :=
  D_at_29_31 (spectrum29_two h) (qual29_all h) n

/-- The same rung in R39's form: `g31 <= 71`. -/
theorem g31_le_of_census (h : Census29) (n : ℕ) : g31 n ≤ 71 :=
  g31_le_71 (spectrum29_two h) (qual29_all h) n

end Machine31
