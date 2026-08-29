/-
THE CENSUS HYPOTHESIS SHRINKS TO ONE PERIOD (round 26, machine 31) - the
same reduction as `Machine29Cen`, at the machine whose period is
33,426,748,355 slots.

The only machine-specific inputs are the period's value and the fact that
the opening predicate depends on the slot's residues - `exposed31_period`,
one `omega` per gear.  No walk, no base case, no scan.
-/

import Machine31Dict
import Periodic

namespace Machine31

/-- Machine 31's period: `5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31`. -/
theorem exposed31_period {k : ℕ} (hk : 1 ≤ k) :
    Exposed31 (k + 33426748355) ↔ Exposed31 k := by
  have hlo : Census.lo (k + 33426748355) = Census.lo k + 200560490130 := by
    simp only [Census.lo]; omega
  have hhi : Census.hi (k + 33426748355) = Census.hi k + 200560490130 := by
    simp only [Census.hi]; omega
  have e5l : (5 ∣ Census.lo k + 200560490130) ↔ (5 ∣ Census.lo k) := by omega
  have e5h : (5 ∣ Census.hi k + 200560490130) ↔ (5 ∣ Census.hi k) := by omega
  have e7l : (7 ∣ Census.lo k + 200560490130) ↔ (7 ∣ Census.lo k) := by omega
  have e7h : (7 ∣ Census.hi k + 200560490130) ↔ (7 ∣ Census.hi k) := by omega
  have e11l : (11 ∣ Census.lo k + 200560490130) ↔ (11 ∣ Census.lo k) := by omega
  have e11h : (11 ∣ Census.hi k + 200560490130) ↔ (11 ∣ Census.hi k) := by omega
  have e13l : (13 ∣ Census.lo k + 200560490130) ↔ (13 ∣ Census.lo k) := by omega
  have e13h : (13 ∣ Census.hi k + 200560490130) ↔ (13 ∣ Census.hi k) := by omega
  have e17l : (17 ∣ Census.lo k + 200560490130) ↔ (17 ∣ Census.lo k) := by omega
  have e17h : (17 ∣ Census.hi k + 200560490130) ↔ (17 ∣ Census.hi k) := by omega
  have e19l : (19 ∣ Census.lo k + 200560490130) ↔ (19 ∣ Census.lo k) := by omega
  have e19h : (19 ∣ Census.hi k + 200560490130) ↔ (19 ∣ Census.hi k) := by omega
  have e23l : (23 ∣ Census.lo k + 200560490130) ↔ (23 ∣ Census.lo k) := by omega
  have e23h : (23 ∣ Census.hi k + 200560490130) ↔ (23 ∣ Census.hi k) := by omega
  have e29l : (29 ∣ Census.lo k + 200560490130) ↔ (29 ∣ Census.lo k) := by omega
  have e29h : (29 ∣ Census.hi k + 200560490130) ↔ (29 ∣ Census.hi k) := by omega
  have e31l : (31 ∣ Census.lo k + 200560490130) ↔ (31 ∣ Census.lo k) := by omega
  have e31h : (31 ∣ Census.hi k + 200560490130) ↔ (31 ∣ Census.hi k) := by omega
  unfold Exposed31 Machine29.Exposed29 Machine23.Exposed23 Machine19.Exposed19
  rw [hlo, hhi, e5l, e5h, e7l, e7h, e11l, e11h, e13l, e13h, e17l, e17h,
    e19l, e19h, e23l, e23h, e29l, e29h, e31l, e31h]

/-- The next-opening operator commutes with machine 31's period. -/
theorem nextOp31_shift (k : ℕ) :
    nextOp31 (k + 33426748355) = nextOp31 k + 33426748355 :=
  Periodic.next_shift (E := Exposed31) nextOp31_gt nextOp31_exposed
    (fun _k _m h1 h2 => nextOp31_min h1 h2)
    (fun _k hk => exposed31_period hk) k

/-- **THE CENSUS REDUCTION AT MACHINE 31.** -/
theorem index_reduce31 (n : ℕ) :
    ∃ m, opSeq31 m ≤ 33426748355 ∧ ∀ i, g31 (n + i) = g31 (m + i) :=
  Periodic.index_reduce (E := Exposed31) (next := nextOp31) (op := opSeq31)
    (g := g31) (by omega) (fun _ => rfl) nextOp31_shift opSeq31_exposed
    opSeq31_pos (fun _k hk => exposed31_period hk)
    (fun _m hm hE => opSeq31_surj hm hE) (fun _ => rfl) n

/-- **The census, as a ONE-PERIOD claim.** -/
structure Census31P : Prop where
  E2 : ∀ n, opSeq31 n ≤ 33426748355 → (g31 n, g31 (n + 1)) ∈ D2
  E3 : ∀ n, opSeq31 n ≤ 33426748355 → 12 ≤ g31 (n + 1) →
    (g31 n, g31 (n + 1), g31 (n + 2)) ∈ D3
  E4 : ∀ n, opSeq31 n ≤ 33426748355 → 12 ≤ g31 (n + 1) → 12 ≤ g31 (n + 2) →
    (g31 n, g31 (n + 1), g31 (n + 2), g31 (n + 3)) ∈ D4
  E5 : ∀ n, opSeq31 n ≤ 33426748355 → 12 ≤ g31 (n + 1) → 12 ≤ g31 (n + 2) →
    12 ≤ g31 (n + 3) →
    (g31 n, g31 (n + 1), g31 (n + 2), g31 (n + 3), g31 (n + 4)) ∈ D5
  E6 : ∀ n, opSeq31 n ≤ 33426748355 → 12 ≤ g31 (n + 1) → 12 ≤ g31 (n + 2) →
    12 ≤ g31 (n + 3) → 12 ≤ g31 (n + 4) →
    (g31 n, g31 (n + 1), g31 (n + 2), g31 (n + 3), g31 (n + 4),
      g31 (n + 5)) ∈ D6
  E7 : ∀ n, opSeq31 n ≤ 33426748355 → 12 ≤ g31 (n + 1) → 12 ≤ g31 (n + 2) →
    12 ≤ g31 (n + 3) → 12 ≤ g31 (n + 4) → 12 ≤ g31 (n + 5) →
    (g31 n, g31 (n + 1), g31 (n + 2), g31 (n + 3), g31 (n + 4),
      g31 (n + 5), g31 (n + 6)) ∈ D7
  run : ∀ n, opSeq31 n ≤ 33426748355 →
    ¬ (12 ≤ g31 (n + 1) ∧ 12 ≤ g31 (n + 2) ∧ 12 ≤ g31 (n + 3) ∧
      12 ≤ g31 (n + 4) ∧ 12 ≤ g31 (n + 5) ∧ 12 ≤ g31 (n + 6))

/-- **THE SHRINKAGE AT MACHINE 31.** -/
theorem census31_of_period (h : Census31P) : Census31 where
  E2 := by
    intro n
    obtain ⟨m, hm, hgs⟩ := index_reduce31 n
    have h0 := hgs 0; have h1 := hgs 1
    simp only [Nat.add_zero] at h0
    rw [h0, h1]
    exact h.E2 m hm
  E3 := by
    intro n c1
    obtain ⟨m, hm, hgs⟩ := index_reduce31 n
    have h0 := hgs 0; have h1 := hgs 1; have h2 := hgs 2
    simp only [Nat.add_zero] at h0
    rw [h1] at c1
    rw [h0, h1, h2]
    exact h.E3 m hm c1
  E4 := by
    intro n c1 c2
    obtain ⟨m, hm, hgs⟩ := index_reduce31 n
    have h0 := hgs 0; have h1 := hgs 1; have h2 := hgs 2; have h3 := hgs 3
    simp only [Nat.add_zero] at h0
    rw [h1] at c1; rw [h2] at c2
    rw [h0, h1, h2, h3]
    exact h.E4 m hm c1 c2
  E5 := by
    intro n c1 c2 c3
    obtain ⟨m, hm, hgs⟩ := index_reduce31 n
    have h0 := hgs 0; have h1 := hgs 1; have h2 := hgs 2; have h3 := hgs 3
    have h4 := hgs 4
    simp only [Nat.add_zero] at h0
    rw [h1] at c1; rw [h2] at c2; rw [h3] at c3
    rw [h0, h1, h2, h3, h4]
    exact h.E5 m hm c1 c2 c3
  E6 := by
    intro n c1 c2 c3 c4
    obtain ⟨m, hm, hgs⟩ := index_reduce31 n
    have h0 := hgs 0; have h1 := hgs 1; have h2 := hgs 2; have h3 := hgs 3
    have h4 := hgs 4; have h5 := hgs 5
    simp only [Nat.add_zero] at h0
    rw [h1] at c1; rw [h2] at c2; rw [h3] at c3; rw [h4] at c4
    rw [h0, h1, h2, h3, h4, h5]
    exact h.E6 m hm c1 c2 c3 c4
  E7 := by
    intro n c1 c2 c3 c4 c5
    obtain ⟨m, hm, hgs⟩ := index_reduce31 n
    have h0 := hgs 0; have h1 := hgs 1; have h2 := hgs 2; have h3 := hgs 3
    have h4 := hgs 4; have h5 := hgs 5; have h6 := hgs 6
    simp only [Nat.add_zero] at h0
    rw [h1] at c1; rw [h2] at c2; rw [h3] at c3; rw [h4] at c4; rw [h5] at c5
    rw [h0, h1, h2, h3, h4, h5, h6]
    exact h.E7 m hm c1 c2 c3 c4 c5
  run := by
    intro n
    obtain ⟨m, hm, hgs⟩ := index_reduce31 n
    have h1 := hgs 1; have h2 := hgs 2; have h3 := hgs 3
    have h4 := hgs 4; have h5 := hgs 5; have h6 := hgs 6
    rw [h1, h2, h3, h4, h5, h6]
    exact h.run m hm

/-- `F_2(31) <= 68` on the shrunken hypothesis. -/
theorem spectrum31_two_period (h : Census31P) : Spectrum.SpectrumBound g31 2 68 :=
  spectrum31_two (census31_of_period h)

/-- The qualifying spectrum on the shrunken hypothesis. -/
theorem qual31_all_period (h : Census31P) :
    ∀ j, 3 ≤ j → Spectrum.QualBound g31 6 j 91 :=
  qual31_all (census31_of_period h)

end Machine31
