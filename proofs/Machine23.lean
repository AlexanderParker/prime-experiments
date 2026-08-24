/-
Machine 23 from machine 19 + gear 23: (D) at the 19->23 step, END TO END,
with NO hypotheses (round 21).

Gear 23 has teeth at slot residues 4 and 19 (`u' = 4`, `6u' = 24 = 23 + 1`):
`23 | lo k` iff `k = 4 mod 23`, `23 | hi k` iff `k = 19 mod 23`. Machine
23's openings are machine 19's openings off those teeth, so every machine-23
gap is a MERGED WINDOW of machine 19's gap word whose interior openings are
all on the teeth. `MergeLaw.newgap_le` (R39: merge law + residue necessity)
turns machine 19's kernel-fed bounds

    F_2 <= 31   and   Q_j <= 47 for every j >= 3   (`Machine19Q.lean`)

into

    `g23_le`     : every gap of machine 23 is at most 47,
    `D_at_19_23` : every gap of machine 23 is at most F + q' = 25 + 23,

i.e. (D) at `alpha = 3` at the 19->23 step as an unconditional kernel
theorem - no shallowness, no fuel cap, no word list, no floor hypothesis:
the merge law discharges the floor itself, since every merge-word letter is
`0, 8 or 15 mod 23` and at most 25, hence in `{8, 15, 23}`
(`merge_alphabet`). Census cross-check before formalising: full-period
letters exactly `{8, 15, 23}`, merge depth histogram j = 1..4, and
F(23) = 34 <= 47.
-/

import Machine19Q
import MergeLaw

namespace Machine23

open Machine19

/-! ## Gear 23: teeth and survivors -/

/-- Gear 23 kills slot `k`: the two teeth, at `u' = 4` and `23 - u' = 19`. -/
def Killed23 (k : ℕ) : Prop := k % 23 = 4 ∨ k % 23 = 19

instance (k : ℕ) : Decidable (Killed23 k) := by unfold Killed23; infer_instance

/-- An opening of machine 23 = gears `{5, 7, 11, 13, 17, 19, 23}`. -/
def Exposed23 (k : ℕ) : Prop :=
  Exposed19 k ∧ ¬ (23 ∣ Census.lo k) ∧ ¬ (23 ∣ Census.hi k)

instance (k : ℕ) : Decidable (Exposed23 k) := by unfold Exposed23; infer_instance

/-- The teeth ARE the divisibility conditions. -/
theorem killed23_iff {k : ℕ} (hk : 1 ≤ k) :
    Killed23 k ↔ (23 ∣ Census.lo k ∨ 23 ∣ Census.hi k) := by
  simp only [Killed23, Census.lo, Census.hi]
  omega

theorem not_killed_of_exposed23 {k : ℕ} (hk : 1 ≤ k) (h : Exposed23 k) :
    ¬ Killed23 k :=
  fun hK => ((killed23_iff hk).mp hK).elim h.2.1 h.2.2

theorem exposed23_of {k : ℕ} (hk : 1 ≤ k) (h19 : Exposed19 k)
    (hnk : ¬ Killed23 k) : Exposed23 k :=
  ⟨h19, fun hd => hnk ((killed23_iff hk).mpr (Or.inl hd)),
    fun hd => hnk ((killed23_iff hk).mpr (Or.inr hd))⟩

/-! ## Machine 23's own gap sequence -/

/-- Multiples of machine 23's period `37,182,145 = 1,616,615 * 23` are
openings, so an opening exists above any point. -/
theorem exists_exposed23_above (k : ℕ) : ∃ m, k < m ∧ Exposed23 m := by
  refine ⟨37182145 * (k + 1), by omega, ?_, ?_, ?_⟩
  · rw [exposed19_iff (by omega)]
    have h5 : (37182145 * (k + 1)) % 5 = 0 := by omega
    have h7 : (37182145 * (k + 1)) % 7 = 0 := by omega
    have h11 : (37182145 * (k + 1)) % 11 = 0 := by omega
    have h13 : (37182145 * (k + 1)) % 13 = 0 := by omega
    have h17 : (37182145 * (k + 1)) % 17 = 0 := by omega
    have h19 : (37182145 * (k + 1)) % 19 = 0 := by omega
    rw [h5, h7, h11, h13, h17, h19]
    decide
  · simp only [Census.lo]
    omega
  · simp only [Census.hi]
    omega

/-- The next machine-23 opening strictly after `k`. -/
def nextOp23 (k : ℕ) : ℕ := Nat.find (exists_exposed23_above k)

theorem nextOp23_gt (k : ℕ) : k < nextOp23 k :=
  (Nat.find_spec (exists_exposed23_above k)).1

theorem nextOp23_exposed (k : ℕ) : Exposed23 (nextOp23 k) :=
  (Nat.find_spec (exists_exposed23_above k)).2

theorem nextOp23_min {k m : ℕ} (h1 : k < m) (h2 : m < nextOp23 k) :
    ¬ Exposed23 m := fun hE =>
  Nat.find_min (exists_exposed23_above k) h2 ⟨h1, hE⟩

/-- The opening sequence of machine 23, in increasing order. -/
def opSeq23 : ℕ → ℕ
  | 0 => nextOp23 0
  | n + 1 => nextOp23 (opSeq23 n)

theorem opSeq23_exposed (n : ℕ) : Exposed23 (opSeq23 n) := by
  cases n <;> exact nextOp23_exposed _

theorem opSeq23_lt_succ (n : ℕ) : opSeq23 n < opSeq23 (n + 1) := nextOp23_gt _

theorem opSeq23_pos (n : ℕ) : 1 ≤ opSeq23 n := by
  cases n with
  | zero => exact nextOp23_gt 0
  | succ m =>
    have h1 := nextOp23_gt (opSeq23 m)
    have h2 : opSeq23 (m + 1) = nextOp23 (opSeq23 m) := rfl
    omega

/-- No machine-23 opening sits strictly between consecutive members. -/
theorem opSeq23_gap_empty (n : ℕ) :
    ∀ j, opSeq23 n < j → j < opSeq23 (n + 1) → ¬ Exposed23 j :=
  fun _j h1 h2 => nextOp23_min h1 h2

/-- **The gap word of machine 23.** -/
def g23 (n : ℕ) : ℕ := opSeq23 (n + 1) - opSeq23 n

/-! ## The merge alphabet -/

/-- **The 19->23 merge alphabet is `{8, 15, 23}`**: a gap between two
killed openings, being at most `F(19) = 25` and `0, 8 or 15 mod 23`, is
8, 15 or 23 - every letter meets the qualifying floor `2u' = 8`. -/
theorem merge_alphabet {x y : ℕ} (hk1 : Killed23 x) (hk2 : Killed23 y)
    (hxy : x < y) (hle : y - x ≤ 25) :
    y - x = 8 ∨ y - x = 15 ∨ y - x = 23 := by
  rcases hk1 with h1 | h1 <;> rcases hk2 with h2 | h2 <;> omega

/-! ## The two-machine instance of R39 -/

/-- **Every gap of machine 23 is at most 47** - `F(M + q') <= max(F2,
max_j Q_j) = 47` at the 19->23 step, by `MergeLaw.newgap_le` on machine
19's kernel-fed bounds. (Census: `F(23) = 34`, so 47 is a true bound, not
tight - (D) needs only `<= 48`.) -/
theorem g23_le (n : ℕ) : g23 n ≤ 47 := by
  have hEA : Exposed23 (opSeq23 n) := opSeq23_exposed n
  have hA1 : 1 ≤ opSeq23 n := opSeq23_pos n
  have hEB : Exposed23 (opSeq23 (n + 1)) := opSeq23_exposed (n + 1)
  have hAB : opSeq23 n < opSeq23 (n + 1) := opSeq23_lt_succ n
  -- indices in machine 19's enumeration
  obtain ⟨a, ha⟩ := opSeq_surj hA1 hEA.1
  obtain ⟨b, hb⟩ := opSeq_surj (by omega) hEB.1
  have hab : a < b := by
    by_contra hc
    have hba : b ≤ a := by omega
    have h1 : opSeq b ≤ opSeq a := by
      have h2 := opSeq_le_add b (a - b)
      rwa [show b + (a - b) = a by omega] at h2
    omega
  -- the merged window over machine 19's opening indices
  have hmw : MergeLaw.MergedWindow (fun i => Killed23 (opSeq i)) a (b - a) := by
    refine ⟨by omega, ?_, ?_, ?_⟩
    · show ¬ Killed23 (opSeq a)
      rw [ha]
      exact not_killed_of_exposed23 hA1 hEA
    · show ¬ Killed23 (opSeq (a + (b - a)))
      rw [show a + (b - a) = b by omega, hb]
      exact not_killed_of_exposed23 (by omega) hEB
    · intro i hi0 hij
      show Killed23 (opSeq (a + i))
      have hv1 : opSeq23 n < opSeq (a + i) := by
        rw [← ha]; exact opSeq_strict_mono (by omega)
      have hv2 : opSeq (a + i) < opSeq23 (n + 1) := by
        rw [← hb]; exact opSeq_strict_mono (by omega)
      have hEv : Exposed19 (opSeq (a + i)) := opSeq_exposed _
      by_contra hK
      exact opSeq23_gap_empty n _ hv1 hv2
        (exposed23_of (opSeq_pos _) hEv hK)
  -- the new gap is the merged window sum
  have hgap : g23 n = Spectrum.windowSum g19 a (b - a) := by
    rw [Machine19.windowSum_g19, show a + (b - a) = b by omega, ha, hb]
    rfl
  rw [hgap]
  exact MergeLaw.newgap_le (g := g19) (pos := opSeq)
    (kap := fun i => Killed23 (opSeq i)) (q := 23) (u := 4) (B := 47)
    (F2 := 31) (Q := fun _ => 47)
    (fun _ => rfl) opSeq_lt_succ
    (fun i hk => by
      rcases hk with h | h
      · exact Or.inl h
      · right; omega)
    (by omega) (by omega)
    spectrum_two (by omega)
    qual_bound_all (fun _ => le_refl 47)
    hmw

/-- **(D) at `alpha = 3` at the 19->23 step, end to end and hypothesis-free:
every gap of machine 23 is at most `F(19) + q' = 25 + 23 = 48`.** The first
machine step where (D) is FULLY kernel-checked: flatness, the qualifying
spectrum, the fuel cap (`Q_6 = 0`) and the floor are all discharged by the
period scans and the merge law. -/
theorem D_at_19_23 (n : ℕ) : g23 n ≤ 25 + 23 :=
  le_trans (g23_le n) (by omega)

end Machine23
