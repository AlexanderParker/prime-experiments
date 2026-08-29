/-
THE CENSUS HYPOTHESIS SHRINKS TO ONE PERIOD (round 26, machine 29).

`Census29` is stated as SIX claims about EVERY index `n` of machine 29's gap
word, plus a run bound - and `research/qual_dict.py` verifies them by
scanning ONE PERIOD.  Between those two there was an unproved step: nothing
in the ledger said that what holds on one period holds at every index.  This
file closes it, using `Periodic.index_reduce` and NOTHING else - in
particular NO walk and NO base case, so it works at a machine whose period a
kernel will never see:

    exposed29_period :  Exposed29 (k + 1078282205) ↔ Exposed29 k    (k >= 1)
    index_reduce29   :  every index's forward gap word is the forward gap
                        word of an index whose opening is in [1, P]
    census29_of_period : Census29P -> Census29

`Census29P` is `Census29` with every `∀ n` replaced by `∀ n, opSeq29 n ≤ P` -
a claim about the 214,708,725 openings of one period, which is EXACTLY the
finite object the gate scans.  The 29->31 rung's hypothesis is therefore
smaller than it was: still not kernel-checked (verdict 21 stands), but no
longer an infinite claim.
-/

import Machine29Dict
import Periodic

namespace Machine29

/-- Machine 29's period: `5 * 7 * 11 * 13 * 17 * 19 * 23 * 29`. -/
theorem exposed29_period {k : ℕ} (hk : 1 ≤ k) :
    Exposed29 (k + 1078282205) ↔ Exposed29 k := by
  have hlo : Census.lo (k + 1078282205) = Census.lo k + 6469693230 := by
    simp only [Census.lo]; omega
  have hhi : Census.hi (k + 1078282205) = Census.hi k + 6469693230 := by
    simp only [Census.hi]; omega
  have e5l : (5 ∣ Census.lo k + 6469693230) ↔ (5 ∣ Census.lo k) := by omega
  have e5h : (5 ∣ Census.hi k + 6469693230) ↔ (5 ∣ Census.hi k) := by omega
  have e7l : (7 ∣ Census.lo k + 6469693230) ↔ (7 ∣ Census.lo k) := by omega
  have e7h : (7 ∣ Census.hi k + 6469693230) ↔ (7 ∣ Census.hi k) := by omega
  have e11l : (11 ∣ Census.lo k + 6469693230) ↔ (11 ∣ Census.lo k) := by omega
  have e11h : (11 ∣ Census.hi k + 6469693230) ↔ (11 ∣ Census.hi k) := by omega
  have e13l : (13 ∣ Census.lo k + 6469693230) ↔ (13 ∣ Census.lo k) := by omega
  have e13h : (13 ∣ Census.hi k + 6469693230) ↔ (13 ∣ Census.hi k) := by omega
  have e17l : (17 ∣ Census.lo k + 6469693230) ↔ (17 ∣ Census.lo k) := by omega
  have e17h : (17 ∣ Census.hi k + 6469693230) ↔ (17 ∣ Census.hi k) := by omega
  have e19l : (19 ∣ Census.lo k + 6469693230) ↔ (19 ∣ Census.lo k) := by omega
  have e19h : (19 ∣ Census.hi k + 6469693230) ↔ (19 ∣ Census.hi k) := by omega
  have e23l : (23 ∣ Census.lo k + 6469693230) ↔ (23 ∣ Census.lo k) := by omega
  have e23h : (23 ∣ Census.hi k + 6469693230) ↔ (23 ∣ Census.hi k) := by omega
  have e29l : (29 ∣ Census.lo k + 6469693230) ↔ (29 ∣ Census.lo k) := by omega
  have e29h : (29 ∣ Census.hi k + 6469693230) ↔ (29 ∣ Census.hi k) := by omega
  unfold Exposed29 Machine23.Exposed23 Machine19.Exposed19
  rw [hlo, hhi, e5l, e5h, e7l, e7h, e11l, e11h, e13l, e13h, e17l, e17h,
    e19l, e19h, e23l, e23h, e29l, e29h]

/-- The next-opening operator commutes with machine 29's period. -/
theorem nextOp29_shift (k : ℕ) :
    nextOp29 (k + 1078282205) = nextOp29 k + 1078282205 :=
  Periodic.next_shift (E := Exposed29) nextOp29_gt nextOp29_exposed
    (fun _k _m h1 h2 => nextOp29_min h1 h2)
    (fun _k hk => exposed29_period hk) k

/-- **THE CENSUS REDUCTION AT MACHINE 29**: every index's forward gap word is
the forward gap word of an index whose opening lies in the first period. -/
theorem index_reduce29 (n : ℕ) :
    ∃ m, opSeq29 m ≤ 1078282205 ∧ ∀ i, g29 (n + i) = g29 (m + i) :=
  Periodic.index_reduce (E := Exposed29) (next := nextOp29) (op := opSeq29)
    (g := g29) (by omega) (fun _ => rfl) nextOp29_shift opSeq29_exposed
    opSeq29_pos (fun _k hk => exposed29_period hk)
    (fun _m hm hE => opSeq29_surj hm hE) (fun _ => rfl) n

/-- **The census, as a ONE-PERIOD claim.**  Identical to `Census29` except
that every clause is restricted to indices whose opening lies in `[1, P]` -
the finite set `research/qual_dict.py` actually scans. -/
structure Census29P : Prop where
  E2 : ∀ n, opSeq29 n ≤ 1078282205 → (g29 n, g29 (n + 1)) ∈ D2
  E3 : ∀ n, opSeq29 n ≤ 1078282205 → 10 ≤ g29 (n + 1) →
    (g29 n, g29 (n + 1), g29 (n + 2)) ∈ D3
  E4 : ∀ n, opSeq29 n ≤ 1078282205 → 10 ≤ g29 (n + 1) → 10 ≤ g29 (n + 2) →
    (g29 n, g29 (n + 1), g29 (n + 2), g29 (n + 3)) ∈ D4
  E5 : ∀ n, opSeq29 n ≤ 1078282205 → 10 ≤ g29 (n + 1) → 10 ≤ g29 (n + 2) →
    10 ≤ g29 (n + 3) →
    (g29 n, g29 (n + 1), g29 (n + 2), g29 (n + 3), g29 (n + 4)) ∈ D5
  E6 : ∀ n, opSeq29 n ≤ 1078282205 → 10 ≤ g29 (n + 1) → 10 ≤ g29 (n + 2) →
    10 ≤ g29 (n + 3) → 10 ≤ g29 (n + 4) →
    (g29 n, g29 (n + 1), g29 (n + 2), g29 (n + 3), g29 (n + 4),
      g29 (n + 5)) ∈ D6
  E7 : ∀ n, opSeq29 n ≤ 1078282205 → 10 ≤ g29 (n + 1) → 10 ≤ g29 (n + 2) →
    10 ≤ g29 (n + 3) → 10 ≤ g29 (n + 4) → 10 ≤ g29 (n + 5) →
    (g29 n, g29 (n + 1), g29 (n + 2), g29 (n + 3), g29 (n + 4),
      g29 (n + 5), g29 (n + 6)) ∈ D7
  run : ∀ n, opSeq29 n ≤ 1078282205 →
    ¬ (10 ≤ g29 (n + 1) ∧ 10 ≤ g29 (n + 2) ∧ 10 ≤ g29 (n + 3) ∧
      10 ≤ g29 (n + 4) ∧ 10 ≤ g29 (n + 5) ∧ 10 ≤ g29 (n + 6))

/-- **THE SHRINKAGE**: the one-period census implies the census.  Every rung
that quotes `Census29` may now quote `Census29P` instead. -/
theorem census29_of_period (h : Census29P) : Census29 where
  E2 := by
    intro n
    obtain ⟨m, hm, hgs⟩ := index_reduce29 n
    have h0 := hgs 0; have h1 := hgs 1
    simp only [Nat.add_zero] at h0
    rw [h0, h1]
    exact h.E2 m hm
  E3 := by
    intro n c1
    obtain ⟨m, hm, hgs⟩ := index_reduce29 n
    have h0 := hgs 0; have h1 := hgs 1; have h2 := hgs 2
    simp only [Nat.add_zero] at h0
    rw [h1] at c1
    rw [h0, h1, h2]
    exact h.E3 m hm c1
  E4 := by
    intro n c1 c2
    obtain ⟨m, hm, hgs⟩ := index_reduce29 n
    have h0 := hgs 0; have h1 := hgs 1; have h2 := hgs 2; have h3 := hgs 3
    simp only [Nat.add_zero] at h0
    rw [h1] at c1; rw [h2] at c2
    rw [h0, h1, h2, h3]
    exact h.E4 m hm c1 c2
  E5 := by
    intro n c1 c2 c3
    obtain ⟨m, hm, hgs⟩ := index_reduce29 n
    have h0 := hgs 0; have h1 := hgs 1; have h2 := hgs 2; have h3 := hgs 3
    have h4 := hgs 4
    simp only [Nat.add_zero] at h0
    rw [h1] at c1; rw [h2] at c2; rw [h3] at c3
    rw [h0, h1, h2, h3, h4]
    exact h.E5 m hm c1 c2 c3
  E6 := by
    intro n c1 c2 c3 c4
    obtain ⟨m, hm, hgs⟩ := index_reduce29 n
    have h0 := hgs 0; have h1 := hgs 1; have h2 := hgs 2; have h3 := hgs 3
    have h4 := hgs 4; have h5 := hgs 5
    simp only [Nat.add_zero] at h0
    rw [h1] at c1; rw [h2] at c2; rw [h3] at c3; rw [h4] at c4
    rw [h0, h1, h2, h3, h4, h5]
    exact h.E6 m hm c1 c2 c3 c4
  E7 := by
    intro n c1 c2 c3 c4 c5
    obtain ⟨m, hm, hgs⟩ := index_reduce29 n
    have h0 := hgs 0; have h1 := hgs 1; have h2 := hgs 2; have h3 := hgs 3
    have h4 := hgs 4; have h5 := hgs 5; have h6 := hgs 6
    simp only [Nat.add_zero] at h0
    rw [h1] at c1; rw [h2] at c2; rw [h3] at c3; rw [h4] at c4; rw [h5] at c5
    rw [h0, h1, h2, h3, h4, h5, h6]
    exact h.E7 m hm c1 c2 c3 c4 c5
  run := by
    intro n
    obtain ⟨m, hm, hgs⟩ := index_reduce29 n
    have h1 := hgs 1; have h2 := hgs 2; have h3 := hgs 3
    have h4 := hgs 4; have h5 := hgs 5; have h6 := hgs 6
    rw [h1, h2, h3, h4, h5, h6]
    exact h.run m hm

/-- The 29->31 rung, on the shrunken hypothesis. -/
theorem spectrum29_two_period (h : Census29P) : Spectrum.SpectrumBound g29 2 55 :=
  spectrum29_two (census29_of_period h)

/-- The qualifying spectrum, on the shrunken hypothesis. -/
theorem qual29_all_period (h : Census29P) :
    ∀ j, 3 ≤ j → Spectrum.QualBound g29 5 j 71 :=
  qual29_all (census29_of_period h)

end Machine29
