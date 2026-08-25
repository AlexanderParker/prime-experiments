/-
Machine 29 from machine 23 + gear 29: the 23->29 rung of the (D) ladder,
reduced to TWO DECIDABLE FACTS about machine 23's own gap word (round 23).

Round 22 recorded `Ladder.D_at_23_29` as a hypothesis-explicit instantiation
over an ABSTRACT pair of machines, with the will-not-close verdict that the
merge law cannot supply its own next-rung input. This file removes everything
in that rung except the two spectrum facts:

    (i)  `Spectrum.SpectrumBound Machine23.g23 2 39`      (`F_2(23) = 39`)
    (ii) `∀ j ≥ 3, Spectrum.QualBound Machine23.g23 5 j 60`
                                          (`Q_j(23; 10) <= 60`, floor 2u''=10)

Both are statements about MACHINE 23's OWN gap sequence, both are decidable,
and both were verified over the full 37,182,145-slot period before this file
was written (scratchpad m23_qspec.py: `F_1..F_8 = 34, 39, 50, 58, 65, 77, 83,
88`; `Q_j(23; 10) = 39, 43, 50, 55, 60, 0` for `j = 2..7`; longest run of
gaps `>= 10` is 4). `Machine23QCore.lean` is the kernel encoding that decides
them; formalist.md round 23 records the measured cost of running it.

Everything else IS discharged here: machine 29's own opening enumeration, its
teeth (`u'' = 5`, `6 * 5 = 30 = 29 + 1`, so gear 29 kills slot residues 5 and
24), the containment `Exposed29 -> Exposed23`, the kill/survive equivalences,
machine 23's enumeration completeness (`Machine23Q.opSeq23_surj`) and the
merge-law wiring. So `D_at_23_29` below is (D) at `alpha = 3` at the 23->29
step as a theorem about `g29`, with exactly two named hypotheses:

    criterion  max (F_2(23), max_j Q_j(23; 10)) = max (39, 60) = 60
    budget     F(23) + 29 = 34 + 29 = 63          margin 3
-/

import Machine23Q
import MergeLaw

namespace Machine29

open Machine23

/-! ## Gear 29: teeth and survivors -/

/-- Gear 29 kills slot `k`: the two teeth, at `u'' = 5` and `29 - u'' = 24`. -/
def Killed29 (k : ℕ) : Prop := k % 29 = 5 ∨ k % 29 = 24

instance (k : ℕ) : Decidable (Killed29 k) := by unfold Killed29; infer_instance

/-- An opening of machine 29 = gears `{5, 7, 11, 13, 17, 19, 23, 29}`. -/
def Exposed29 (k : ℕ) : Prop :=
  Exposed23 k ∧ ¬ (29 ∣ Census.lo k) ∧ ¬ (29 ∣ Census.hi k)

instance (k : ℕ) : Decidable (Exposed29 k) := by unfold Exposed29; infer_instance

/-- The teeth ARE the divisibility conditions. -/
theorem killed29_iff {k : ℕ} (hk : 1 ≤ k) :
    Killed29 k ↔ (29 ∣ Census.lo k ∨ 29 ∣ Census.hi k) := by
  simp only [Killed29, Census.lo, Census.hi]
  omega

theorem not_killed_of_exposed29 {k : ℕ} (hk : 1 ≤ k) (h : Exposed29 k) :
    ¬ Killed29 k :=
  fun hK => ((killed29_iff hk).mp hK).elim h.2.1 h.2.2

theorem exposed29_of {k : ℕ} (hk : 1 ≤ k) (h23 : Exposed23 k)
    (hnk : ¬ Killed29 k) : Exposed29 k :=
  ⟨h23, fun hd => hnk ((killed29_iff hk).mpr (Or.inl hd)),
    fun hd => hnk ((killed29_iff hk).mpr (Or.inr hd))⟩

theorem exposed23_of_29 {k : ℕ} (h : Exposed29 k) : Exposed23 k := h.1

/-! ## Machine 29's own gap sequence -/

/-- Multiples of machine 29's period `1,078,282,205 = 37,182,145 * 29` are
openings, so an opening exists above any point. -/
theorem exists_exposed29_above (k : ℕ) : ∃ m, k < m ∧ Exposed29 m := by
  refine ⟨1078282205 * (k + 1), by omega, ⟨⟨?_, ?_, ?_⟩, ?_, ?_⟩⟩
  · rw [Machine19.exposed19_iff (by omega)]
    have h5 : (1078282205 * (k + 1)) % 5 = 0 := by omega
    have h7 : (1078282205 * (k + 1)) % 7 = 0 := by omega
    have h11 : (1078282205 * (k + 1)) % 11 = 0 := by omega
    have h13 : (1078282205 * (k + 1)) % 13 = 0 := by omega
    have h17 : (1078282205 * (k + 1)) % 17 = 0 := by omega
    have h19 : (1078282205 * (k + 1)) % 19 = 0 := by omega
    rw [h5, h7, h11, h13, h17, h19]
    decide
  · simp only [Census.lo]; omega
  · simp only [Census.hi]; omega
  · simp only [Census.lo]; omega
  · simp only [Census.hi]; omega

/-- The next machine-29 opening strictly after `k`. -/
def nextOp29 (k : ℕ) : ℕ := Nat.find (exists_exposed29_above k)

theorem nextOp29_gt (k : ℕ) : k < nextOp29 k :=
  (Nat.find_spec (exists_exposed29_above k)).1

theorem nextOp29_exposed (k : ℕ) : Exposed29 (nextOp29 k) :=
  (Nat.find_spec (exists_exposed29_above k)).2

theorem nextOp29_min {k m : ℕ} (h1 : k < m) (h2 : m < nextOp29 k) :
    ¬ Exposed29 m := fun hE =>
  Nat.find_min (exists_exposed29_above k) h2 ⟨h1, hE⟩

/-- The opening sequence of machine 29, in increasing order. -/
def opSeq29 : ℕ → ℕ
  | 0 => nextOp29 0
  | n + 1 => nextOp29 (opSeq29 n)

theorem opSeq29_exposed (n : ℕ) : Exposed29 (opSeq29 n) := by
  cases n <;> exact nextOp29_exposed _

theorem opSeq29_lt_succ (n : ℕ) : opSeq29 n < opSeq29 (n + 1) := nextOp29_gt _

theorem opSeq29_pos (n : ℕ) : 1 ≤ opSeq29 n := by
  cases n with
  | zero => exact nextOp29_gt 0
  | succ m =>
    have h1 := nextOp29_gt (opSeq29 m)
    have h2 : opSeq29 (m + 1) = nextOp29 (opSeq29 m) := rfl
    omega

/-- No machine-29 opening sits strictly between consecutive members. -/
theorem opSeq29_gap_empty (n : ℕ) :
    ∀ j, opSeq29 n < j → j < opSeq29 (n + 1) → ¬ Exposed29 j :=
  fun _j h1 h2 => nextOp29_min h1 h2

/-- **The gap word of machine 29.** -/
def g29 (n : ℕ) : ℕ := opSeq29 (n + 1) - opSeq29 n

/-! ## The merge alphabet at 23->29 -/

/-- **The 23->29 merge alphabet is `{10, 19, 29}`**: a gap between two
gear-29-killed openings, being at most `F(23) = 34` and `0, 10 or 19` mod 29,
is 10, 19 or 29 - every letter meets the qualifying floor `2u'' = 10`. (Not
load-bearing: `MergeLaw.newgap_le` derives the floor itself from residue
necessity. Recorded because it is the concrete content at this step.) -/
theorem merge_alphabet {x y : ℕ} (hk1 : Killed29 x) (hk2 : Killed29 y)
    (hxy : x < y) (hle : y - x ≤ 34) :
    y - x = 10 ∨ y - x = 19 ∨ y - x = 29 := by
  rcases hk1 with h1 | h1 <;> rcases hk2 with h2 | h2 <;> omega

/-! ## The rung -/

/-- **The 23->29 rung of the (D) ladder**, with exactly two hypotheses, both
decidable facts about MACHINE 23's own gap word: `F_2(23) <= 39` and
`Q_j(23; 10) <= 60` at every depth `j >= 3`. Everything else - machine 29's
enumeration, the teeth, the containment, machine 23's enumeration
completeness and the merge law - is discharged.

`max (39, 60) = 60 <= 63 = F(23) + 29`, margin 3. -/
theorem D_at_23_29 (hF2 : Spectrum.SpectrumBound g23 2 39)
    (hQ : ∀ j, 3 ≤ j → Spectrum.QualBound g23 5 j 60) (n : ℕ) :
    g29 n ≤ 34 + 29 :=
  MergeLaw.newgap_le_step
    (ExO := Exposed23) (ExN := Exposed29) (Kap := Killed29)
    (posO := opSeq23) (posN := opSeq29) (g := g23)
    (q := 29) (u := 5) (B := 34 + 29) (F2 := 39) (Q := fun _ => 60)
    (fun _ => rfl) opSeq23_lt_succ opSeq23_pos opSeq23_exposed
    (fun _ hx hE => Machine23.opSeq23_surj hx hE)
    opSeq29_pos opSeq29_lt_succ opSeq29_exposed
    (fun m x => opSeq29_gap_empty m x)
    (fun _ hx hE => not_killed_of_exposed29 hx hE)
    (fun _ hx hE hnk => exposed29_of hx hE hnk)
    (fun _ h => exposed23_of_29 h)
    (fun _ hk => by rcases hk with h | h <;> omega)
    (by omega) (by omega)
    hF2 (by omega) hQ (fun _ => by omega) n

/-- **R39's own form at the 23->29 step**: every gap of machine 29 is at most
`max (F_2(23), max_j Q_j(23; 10)) = max (39, 60) = 60`, strictly inside the
(D) budget 63. -/
theorem g29_le (hF2 : Spectrum.SpectrumBound g23 2 39)
    (hQ : ∀ j, 3 ≤ j → Spectrum.QualBound g23 5 j 60) (n : ℕ) :
    g29 n ≤ 60 :=
  MergeLaw.newgap_le_step
    (ExO := Exposed23) (ExN := Exposed29) (Kap := Killed29)
    (posO := opSeq23) (posN := opSeq29) (g := g23)
    (q := 29) (u := 5) (B := 60) (F2 := 39) (Q := fun _ => 60)
    (fun _ => rfl) opSeq23_lt_succ opSeq23_pos opSeq23_exposed
    (fun _ hx hE => Machine23.opSeq23_surj hx hE)
    opSeq29_pos opSeq29_lt_succ opSeq29_exposed
    (fun m x => opSeq29_gap_empty m x)
    (fun _ hx hE => not_killed_of_exposed29 hx hE)
    (fun _ hx hE hnk => exposed29_of hx hE hnk)
    (fun _ h => exposed23_of_29 h)
    (fun _ hk => by rcases hk with h | h <;> omega)
    (by omega) (by omega)
    hF2 (by omega) hQ (fun _ => le_refl 60) n

end Machine29
