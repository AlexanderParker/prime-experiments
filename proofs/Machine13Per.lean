/-
THE PERIODICITY GLUE AT MACHINE 13 (round 26) - round-22 verdict 11's
missing step, the SAME abstract lemma as `Machine11Per` one machine up.

`DepthSum.lean`'s header names the gap in the depth-sum identity: "turning
`depth_partition` into a statement about `pairCount13` requires 'count over
one period of the enumeration = count over residues' - a
periodicity/re-indexing bridge for `Machine13.opSeq` that this file does not
build."  This file builds the bridge:

    opSeq (n + 1485) = opSeq n + 5005

with `1485 = 3 * 5 * 9 * 11 = prod (q - 2)` openings per period of
`5005 = 5 * 7 * 11 * 13` slots.  As at machine 11 the whole content is
`Periodic.op_shift` plus ONE finite kernel computation, `ow13 1485 = 5005`.

The re-indexing CONSUMER (the count identity itself) is still not written -
see the round-26 append.  What is established here is the bridge that
verdict 11 said was missing, at the machine it was missing at.
-/

import Machine13Q
import Periodic

namespace Machine13

/-! ## Machine 13's opening predicate is periodic mod 5005 -/

/-- **Periodicity of the opening predicate**: `5005 = 5 * 7 * 11 * 13`, and
every gear's blocking condition depends only on the slot's residue.  Slot `0`
is excluded for the usual reason (`Census.lo 0 = 0`, not `-1`). -/
theorem exposed13_period {k : ℕ} (hk : 1 ≤ k) :
    Exposed13 (k + 5005) ↔ Exposed13 k := by
  have hlo : Census.lo (k + 5005) = Census.lo k + 30030 := by
    simp only [Census.lo]; omega
  have hhi : Census.hi (k + 5005) = Census.hi k + 30030 := by
    simp only [Census.hi]; omega
  have e5l : (5 ∣ Census.lo k + 30030) ↔ (5 ∣ Census.lo k) := by omega
  have e5h : (5 ∣ Census.hi k + 30030) ↔ (5 ∣ Census.hi k) := by omega
  have e7l : (7 ∣ Census.lo k + 30030) ↔ (7 ∣ Census.lo k) := by omega
  have e7h : (7 ∣ Census.hi k + 30030) ↔ (7 ∣ Census.hi k) := by omega
  have e11l : (11 ∣ Census.lo k + 30030) ↔ (11 ∣ Census.lo k) := by omega
  have e11h : (11 ∣ Census.hi k + 30030) ↔ (11 ∣ Census.hi k) := by omega
  have e13l : (13 ∣ Census.lo k + 30030) ↔ (13 ∣ Census.lo k) := by omega
  have e13h : (13 ∣ Census.hi k + 30030) ↔ (13 ∣ Census.hi k) := by omega
  unfold Exposed13
  rw [hlo, hhi, e5l, e5h, e7l, e7h, e11l, e11h, e13l, e13h]

/-- **The next-opening operator commutes with the period.** -/
theorem nextOp_shift (k : ℕ) : nextOp (k + 5005) = nextOp k + 5005 :=
  Periodic.next_shift (E := Exposed13) nextOp_gt nextOp_exposed
    (fun _k _m h1 h2 => nextOp_min h1 h2)
    (fun _k hk => exposed13_period hk) k

/-! ## The computable opening walk -/

/-- **The opening walk of machine 13**, from its first opening at slot 3. -/
def ow13 : ℕ → ℕ
  | 0 => 0
  | i + 1 => seekT 3 3 3 3 11 (ow13 i)

/-- Machine 13's first opening is slot 3 (`17, 19`). -/
theorem opSeq_zero : opSeq 0 = 3 := by
  have hE := nextOp_exposed 0
  have hgt := nextOp_gt 0
  have hle : nextOp 0 ≤ 3 :=
    Nat.find_min' (exists_exposed_above 0) ⟨by omega, by decide⟩
  have h1 : ¬ Exposed13 1 := by decide
  have h2 : ¬ Exposed13 2 := by decide
  show nextOp 0 = 3
  rcases Nat.lt_or_ge (nextOp 0) 3 with h | h
  · exfalso
    have hc : nextOp 0 = 1 ∨ nextOp 0 = 2 := by omega
    rcases hc with he | he <;> rw [he] at hE
    · exact h1 hE
    · exact h2 hE
  · omega

/-- **The walk is the enumeration.** -/
theorem opSeq_eq_ow13 : ∀ i, opSeq i = 3 + ow13 i := by
  intro i
  induction i with
  | zero => rw [opSeq_zero]; rfl
  | succ i ih =>
    have hE : Exposed13 (3 + ow13 i) := by rw [← ih]; exact opSeq_exposed i
    have h := seek_next (x := 3) (s := ow13 i) (by omega) hE
    rw [opSeq_succ, ih, ← h]
    rfl

set_option maxRecDepth 40000 in
/-- **The base case, in the kernel**: 1,485 openings of machine 13 span
exactly one period of 5,005 slots. -/
theorem ow13_1485 : ow13 1485 = 5005 := by decide +kernel

/-- The period shift, realised once. -/
theorem opSeq_1485 : opSeq 1485 = opSeq 0 + 5005 := by
  rw [opSeq_eq_ow13, opSeq_zero, ow13_1485]

/-! ## The glue -/

/-- **THE PERIODICITY GLUE AT MACHINE 13** (round-22 verdict 11's missing
step): the opening enumeration advances by exactly one period every 1,485
indices. -/
theorem opSeq_shift (n : ℕ) : opSeq (n + 1485) = opSeq n + 5005 :=
  Periodic.op_shift opSeq_succ nextOp_shift opSeq_1485 n

/-- **Machine 13's gap word is periodic**: 1,485 letters. -/
theorem g13_shift (n : ℕ) : g13 (n + 1485) = g13 n :=
  Periodic.gap_shift (fun _ => rfl) opSeq_shift n

/-- 1,485 letters determine the whole sequence. -/
theorem g13_mod (n : ℕ) : g13 n = g13 (n % 1485) :=
  Periodic.gap_mod g13_shift n

/-- **Window sums are periodic too** - the form the depth-sum count needs:
the depth-`j` window at index `a` and the one at `a + 1485` have the same
sum, so the per-depth window counts of `DepthSum.depth_partition` are
counts over ONE period. -/
theorem windowSum_g13_shift (a j : ℕ) :
    Spectrum.windowSum g13 (a + 1485) j = Spectrum.windowSum g13 a j :=
  Periodic.windowSum_shift g13_shift a j

end Machine13
