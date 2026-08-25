/-
Machine 23's opening enumeration is COMPLETE - the missing ingredient of the
23->29 rung (round 23).

`MergeLaw.newgap_le_step` climbs a rung from the OLD machine's enumeration
plus two spectrum facts. At 19->23 the old machine was 19 and
`Machine19Q.opSeq_surj` supplied the enumeration; at 23->29 the old machine
is 23 and the corresponding fact did not exist. This file supplies it -
`opSeq23_surj`, together with strict monotonicity - by the same
`Nat.find`-minimality induction, using ONLY round-21's `Machine23` material
(no new period scan).

With this, everything in the 23->29 rung except two DECIDABLE facts about
machine 23's own gap word is discharged; see `Machine29.lean`.
-/

import Machine23

namespace Machine23

/-- The enumeration's defining equation, named. -/
theorem opSeq23_succ (n : ℕ) : opSeq23 (n + 1) = nextOp23 (opSeq23 n) := rfl

theorem opSeq23_le_add (a j : ℕ) : opSeq23 a ≤ opSeq23 (a + j) := by
  induction j with
  | zero => rfl
  | succ j ih =>
    have := opSeq23_lt_succ (a + j)
    rw [show a + (j + 1) = (a + j) + 1 by omega]
    omega

theorem opSeq23_strict_mono {a b : ℕ} (h : a < b) : opSeq23 a < opSeq23 b := by
  have h1 := opSeq23_lt_succ a
  have h2 := opSeq23_le_add (a + 1) (b - (a + 1))
  rw [show a + 1 + (b - (a + 1)) = b by omega] at h2
  omega

/-- Window sums of `g23` telescope to position differences. -/
theorem windowSum_g23 (a j : ℕ) :
    Spectrum.windowSum g23 a j = opSeq23 (a + j) - opSeq23 a :=
  MergeLaw.windowSum_telescope (fun _ => rfl)
    (fun m => le_of_lt (opSeq23_lt_succ m)) a j

/-- Reaching every opening from a reached one. -/
theorem opSeq23_reach : ∀ dd A, 1 ≤ A → Exposed23 A → (∃ n, opSeq23 n = A) →
    ∀ B, Exposed23 B → A < B → B - A ≤ dd → ∃ m, opSeq23 m = B := by
  intro dd
  induction dd with
  | zero => intro A _ _ _ B _ hAB hd; omega
  | succ dd ih =>
    rintro A hA1 hEA ⟨n, hn⟩ B hEB hAB hd
    have hnext_le : nextOp23 A ≤ B :=
      Nat.find_min' (exists_exposed23_above A) ⟨hAB, hEB⟩
    have hgt := nextOp23_gt A
    rcases eq_or_lt_of_le hnext_le with he | hlt
    · exact ⟨n + 1, by rw [opSeq23_succ, hn, he]⟩
    · exact ih (nextOp23 A) (by omega) (nextOp23_exposed A)
        ⟨n + 1, by rw [opSeq23_succ, hn]⟩ B hEB hlt (by omega)

/-- **Every machine-23 opening is enumerated**: `opSeq23` is onto the
openings - the enumeration fact `MergeLaw.newgap_le_step` needs at the
23->29 rung. -/
theorem opSeq23_surj {m : ℕ} (hm : 1 ≤ m) (hE : Exposed23 m) :
    ∃ n, opSeq23 n = m := by
  have h0 : opSeq23 0 = nextOp23 0 := rfl
  have hle : nextOp23 0 ≤ m :=
    Nat.find_min' (exists_exposed23_above 0) ⟨by omega, hE⟩
  rcases eq_or_lt_of_le hle with he | hlt
  · exact ⟨0, by rw [h0, he]⟩
  · exact opSeq23_reach (m - nextOp23 0) (nextOp23 0)
      (by have := nextOp23_gt 0; omega) (nextOp23_exposed 0) ⟨0, rfl⟩ m hE hlt
      (by omega)

end Machine23
