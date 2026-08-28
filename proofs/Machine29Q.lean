/-
Machine 29's opening enumeration is COMPLETE - the missing ingredient of the
29->31 rung (round 25).

Exactly `Machine23Q.lean` one machine up: `MergeLaw.newgap_le_step` climbs a
rung from the OLD machine's enumeration plus two spectrum facts, and at
29->31 the old machine is 29.  This file supplies `opSeq29_surj` and strict
monotonicity by the same `Nat.find`-minimality induction, using ONLY
`Machine29.lean`'s material - NO period scan, nothing about machine 29's
1,078,282,205-slot period beyond what `Machine29.exists_exposed29_above`
already proves.
-/

import Machine29

namespace Machine29

/-- The enumeration's defining equation, named. -/
theorem opSeq29_succ (n : ℕ) : opSeq29 (n + 1) = nextOp29 (opSeq29 n) := rfl

theorem opSeq29_le_add (a j : ℕ) : opSeq29 a ≤ opSeq29 (a + j) := by
  induction j with
  | zero => rfl
  | succ j ih =>
    have := opSeq29_lt_succ (a + j)
    rw [show a + (j + 1) = (a + j) + 1 by omega]
    omega

theorem opSeq29_strict_mono {a b : ℕ} (h : a < b) : opSeq29 a < opSeq29 b := by
  have h1 := opSeq29_lt_succ a
  have h2 := opSeq29_le_add (a + 1) (b - (a + 1))
  rw [show a + 1 + (b - (a + 1)) = b by omega] at h2
  omega

/-- Window sums of `g29` telescope to position differences. -/
theorem windowSum_g29 (a j : ℕ) :
    Spectrum.windowSum g29 a j = opSeq29 (a + j) - opSeq29 a :=
  MergeLaw.windowSum_telescope (fun _ => rfl)
    (fun m => le_of_lt (opSeq29_lt_succ m)) a j

/-- Reaching every opening from a reached one. -/
theorem opSeq29_reach : ∀ dd A, 1 ≤ A → Exposed29 A → (∃ n, opSeq29 n = A) →
    ∀ B, Exposed29 B → A < B → B - A ≤ dd → ∃ m, opSeq29 m = B := by
  intro dd
  induction dd with
  | zero => intro A _ _ _ B _ hAB hd; omega
  | succ dd ih =>
    rintro A hA1 hEA ⟨n, hn⟩ B hEB hAB hd
    have hnext_le : nextOp29 A ≤ B :=
      Nat.find_min' (exists_exposed29_above A) ⟨hAB, hEB⟩
    have hgt := nextOp29_gt A
    rcases eq_or_lt_of_le hnext_le with he | hlt
    · exact ⟨n + 1, by rw [opSeq29_succ, hn, he]⟩
    · exact ih (nextOp29 A) (by omega) (nextOp29_exposed A)
        ⟨n + 1, by rw [opSeq29_succ, hn]⟩ B hEB hlt (by omega)

/-- **Every machine-29 opening is enumerated**: `opSeq29` is onto the
openings - the enumeration fact `MergeLaw.newgap_le_step` needs at the
29->31 rung. -/
theorem opSeq29_surj {m : ℕ} (hm : 1 ≤ m) (hE : Exposed29 m) :
    ∃ n, opSeq29 n = m := by
  have h0 : opSeq29 0 = nextOp29 0 := rfl
  have hle : nextOp29 0 ≤ m :=
    Nat.find_min' (exists_exposed29_above 0) ⟨by omega, hE⟩
  rcases eq_or_lt_of_le hle with he | hlt
  · exact ⟨0, by rw [h0, he]⟩
  · exact opSeq29_reach (m - nextOp29 0) (nextOp29 0)
      (by have := nextOp29_gt 0; omega) (nextOp29_exposed 0) ⟨0, rfl⟩ m hE hlt
      (by omega)

end Machine29
