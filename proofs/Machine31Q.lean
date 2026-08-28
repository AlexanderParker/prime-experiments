/-
Machine 31's opening enumeration is COMPLETE - the missing ingredient of the
31->37 rung (round 25).

Exactly `Machine29Q.lean` one machine up, and `Machine23Q.lean` two up: the
same `Nat.find`-minimality induction, using ONLY `Machine31.lean`'s material.
NO period scan; nothing about machine 31's 33,426,748,355-slot period beyond
what `Machine31.exists_exposed31_above` already proves.
-/

import Machine31

namespace Machine31

/-- The enumeration's defining equation, named. -/
theorem opSeq31_succ (n : ℕ) : opSeq31 (n + 1) = nextOp31 (opSeq31 n) := rfl

theorem opSeq31_le_add (a j : ℕ) : opSeq31 a ≤ opSeq31 (a + j) := by
  induction j with
  | zero => rfl
  | succ j ih =>
    have := opSeq31_lt_succ (a + j)
    rw [show a + (j + 1) = (a + j) + 1 by omega]
    omega

theorem opSeq31_strict_mono {a b : ℕ} (h : a < b) : opSeq31 a < opSeq31 b := by
  have h1 := opSeq31_lt_succ a
  have h2 := opSeq31_le_add (a + 1) (b - (a + 1))
  rw [show a + 1 + (b - (a + 1)) = b by omega] at h2
  omega

/-- Window sums of `g31` telescope to position differences. -/
theorem windowSum_g31 (a j : ℕ) :
    Spectrum.windowSum g31 a j = opSeq31 (a + j) - opSeq31 a :=
  MergeLaw.windowSum_telescope (fun _ => rfl)
    (fun m => le_of_lt (opSeq31_lt_succ m)) a j

/-- Reaching every opening from a reached one. -/
theorem opSeq31_reach : ∀ dd A, 1 ≤ A → Exposed31 A → (∃ n, opSeq31 n = A) →
    ∀ B, Exposed31 B → A < B → B - A ≤ dd → ∃ m, opSeq31 m = B := by
  intro dd
  induction dd with
  | zero => intro A _ _ _ B _ hAB hd; omega
  | succ dd ih =>
    rintro A hA1 hEA ⟨n, hn⟩ B hEB hAB hd
    have hnext_le : nextOp31 A ≤ B :=
      Nat.find_min' (exists_exposed31_above A) ⟨hAB, hEB⟩
    have hgt := nextOp31_gt A
    rcases eq_or_lt_of_le hnext_le with he | hlt
    · exact ⟨n + 1, by rw [opSeq31_succ, hn, he]⟩
    · exact ih (nextOp31 A) (by omega) (nextOp31_exposed A)
        ⟨n + 1, by rw [opSeq31_succ, hn]⟩ B hEB hlt (by omega)

/-- **Every machine-31 opening is enumerated** - the enumeration fact
`MergeLaw.newgap_le_step` needs at the 31->37 rung. -/
theorem opSeq31_surj {m : ℕ} (hm : 1 ≤ m) (hE : Exposed31 m) :
    ∃ n, opSeq31 n = m := by
  have h0 : opSeq31 0 = nextOp31 0 := rfl
  have hle : nextOp31 0 ≤ m :=
    Nat.find_min' (exists_exposed31_above 0) ⟨by omega, hE⟩
  rcases eq_or_lt_of_le hle with he | hlt
  · exact ⟨0, by rw [h0, he]⟩
  · exact opSeq31_reach (m - nextOp31 0) (nextOp31 0)
      (by have := nextOp31_gt 0; omega) (nextOp31_exposed 0) ⟨0, rfl⟩ m hE hlt
      (by omega)

end Machine31
