/-
THE PERIODICITY GLUE AT MACHINE 11 (round 26) - verdict 20's missing step
(ii), and with it machine 11's gap word as a THEOREM rather than a table.

`Periodic.op_shift` reduces "the enumeration repeats" to ONE finite
computation, `opSeq N = opSeq 0 + P`.  At machine 11 that is `opSeq 135 =
3 + 385`, and the way to make it kernel-computable is to replace `Nat.find`
by the `seekT` walk that `Machine11.seek_next` already proves equal to it:

    ow 0 = 0,   ow (i+1) = seekT 3 3 3 7 (ow i)          (`ow`, computable)
    opSeq i = 3 + ow i                                    (`opSeq_eq_ow`)
    ow 135 = 385                                          (`ow_135`, kernel)

so `opSeq (n + 135) = opSeq n + 385` for EVERY `n`, and the 135 gaps of the
walk are the whole gap word of machine 11 at every index.  Nothing here
scans machine 13, and nothing here is a hypothesis: the axiom footprint of
every theorem in this file is `[propext]` or empty.
-/

import Machine11
import Periodic

namespace Machine11

/-! ## Machine 11's opening predicate is periodic mod 385 -/

/-- **Periodicity of the opening predicate.**  `385 = 5 * 7 * 11` and each
gear's blocking condition depends only on the slot's residue, so shifting a
slot by a period changes nothing - above slot `0`, whose pair `(0, 1)` is
the one degenerate slot of the whole ledger (`Exposed11 0` is FALSE while
`Exposed11 385` is TRUE). -/
theorem exposed11_period {k : ℕ} (hk : 1 ≤ k) :
    Exposed11 (k + 385) ↔ Exposed11 k := by
  unfold Exposed11 Census.lo Census.hi
  omega

/-- **The next-opening operator commutes with the period** - `Periodic.next_shift`
at machine 11. -/
theorem nextOp_shift (k : ℕ) : nextOp (k + 385) = nextOp k + 385 :=
  Periodic.next_shift (E := Exposed11) nextOp_gt nextOp_exposed
    (fun _k _m h1 h2 => nextOp_min h1 h2)
    (fun _k hk => exposed11_period hk) k

/-! ## The computable opening walk -/

/-- **The opening walk of machine 11**, from its first opening at slot 3:
`ow i` is the offset of the `i`-th opening after slot 3.  This is the
`Nat.find`-free form of `opSeq`, and it is what makes the base case of the
periodicity glue a kernel computation. -/
def ow : ℕ → ℕ
  | 0 => 0
  | i + 1 => seekT 3 3 3 7 (ow i)

/-- Machine 11's first opening is slot 3 (`17, 19`): slots `1` and `2` carry
`5, 7` and `11, 13`. -/
theorem opSeq_zero : opSeq 0 = 3 := by
  have hE := nextOp_exposed 0
  have hgt := nextOp_gt 0
  have hle : nextOp 0 ≤ 3 :=
    Nat.find_min' (exists_exposed_above 0) ⟨by omega, by decide⟩
  have h1 : ¬ Exposed11 1 := by decide
  have h2 : ¬ Exposed11 2 := by decide
  show nextOp 0 = 3
  rcases Nat.lt_or_ge (nextOp 0) 3 with h | h
  · exfalso
    have hc : nextOp 0 = 1 ∨ nextOp 0 = 2 := by omega
    rcases hc with he | he <;> rw [he] at hE
    · exact h1 hE
    · exact h2 hE
  · omega

/-- **The walk is the enumeration.**  Every opening of machine 11 is `3` plus
a walk offset - `seek_next` one step at a time. -/
theorem opSeq_eq_ow : ∀ i, opSeq i = 3 + ow i := by
  intro i
  induction i with
  | zero => rw [opSeq_zero]; rfl
  | succ i ih =>
    have hE : Exposed11 (3 + ow i) := by rw [← ih]; exact opSeq_exposed i
    have h := seek_next (x := 3) (s := ow i) (by omega) hE
    rw [opSeq_succ, ih, ← h]
    rfl

/-- **The base case, in the kernel**: 135 openings of machine 11 span exactly
one period of 385 slots. -/
theorem ow_135 : ow 135 = 385 := by decide +kernel

/-- The period shift, realised once. -/
theorem opSeq_135 : opSeq 135 = opSeq 0 + 385 := by
  rw [opSeq_eq_ow, opSeq_zero, ow_135]

/-! ## The glue, and what it gives -/

/-- **THE PERIODICITY GLUE AT MACHINE 11** (round-25 verdict 20, item (ii)):
the opening enumeration advances by exactly one period every 135 indices. -/
theorem opSeq_shift (n : ℕ) : opSeq (n + 135) = opSeq n + 385 :=
  Periodic.op_shift opSeq_succ nextOp_shift opSeq_135 n

/-- **The gap word is periodic**: machine 11's gap sequence repeats every 135
letters. -/
theorem g11_shift (n : ℕ) : g11 (n + 135) = g11 n :=
  Periodic.gap_shift (fun _ => rfl) opSeq_shift n

/-- **135 letters determine the whole sequence.** -/
theorem g11_mod (n : ℕ) : g11 n = g11 (n % 135) :=
  Periodic.gap_mod g11_shift n

/-- Every gap of machine 11 is a walk difference. -/
theorem g11_eq_ow (i : ℕ) : g11 i = ow (i + 1) - ow i := by
  have h1 := opSeq_eq_ow i
  have h2 := opSeq_eq_ow (i + 1)
  show opSeq (i + 1) - opSeq i = _
  omega

end Machine11
