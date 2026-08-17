/-
The per-gear fiber of the supply partition, and the per-gear cap.

`R q S` is one gear's root class: the composites of `S` whose least factor
is exactly `q`. The supply identity (`Supply.card_composites_eq_sum_roots`)
and the bridge (`Bridge.sum_roots_eq_census`) are restated here in `R` form,
so downstream files can speak about a single gear's ledger line.

The caps proved about one line:

* `R_le_card_multiples` - a root class sits inside the multiples of its
  gear (`minFac m = q → q ∣ m`). Immediate, and load-bearing: every
  per-gear counting argument passes through it.
* `R_prefix_le` - over the members of the slot prefix `k < t`, the
  multiples of `q` obey the interval bound, so `R q ≤ 6t/q + 2`. The
  interval bound is `BlockedSlots.card_blocked_by_le` applied to the
  containing interval `[0, 6t)` - members occupy two residues mod 6, and no
  sharpness is fought for beyond what composes.
* `R_eq_zero_of_below_sq` - the shadow law: a gear supplies nothing below
  its own square, because a composite with `minFac m = q` has `q^2 ≤ m`
  (`Nat.minFac_sq_le_self`). A gear's ledger line starts at `q^2`.
-/

import BlockedSlots
import Bridge

namespace Gear

/-- **One gear's ledger line.** `R q S`: the composites of `S` rooted at
`q`, i.e. with least prime factor exactly `q`. -/
def R (q : ℕ) (S : Finset ℕ) : ℕ :=
  (S.filter fun m => ¬ m.Prime ∧ m.minFac = q).card

/-- Supply's identity, per-gear form: composites split as `Σ_p R p`. -/
theorem supply_eq_sum_R {y : ℕ} (S : Finset ℕ)
    (hS : ∀ m ∈ S, y < m ∧ m < y * y) :
    (S.filter fun m => ¬ m.Prime).card
      = ∑ p ∈ (Finset.range y).filter Nat.Prime, R p S :=
  Supply.card_composites_eq_sum_roots S hS

/-- The bridge, per-gear form: `Σ_p R p (members T) = n1 + 2 n2`. -/
theorem sum_R_eq_census {y : ℕ} (T : Finset ℕ)
    (hwin : ∀ k ∈ T, y < Census.lo k ∧ Census.hi k < y * y) :
    (∑ p ∈ (Finset.range y).filter Nat.Prime, R p (Bridge.members T))
      = Census.n1 T + 2 * Census.n2 T :=
  Bridge.sum_roots_eq_census T hwin

/-- **Per-gear cap, set form.** A gear's root class sits inside its
multiples: rooting at `q` in particular means divisibility by `q`. -/
theorem R_le_card_multiples (q : ℕ) (S : Finset ℕ) :
    R q S ≤ (S.filter fun m => q ∣ m).card := by
  apply Finset.card_le_card
  intro m hm
  rw [Finset.mem_filter] at hm ⊢
  exact ⟨hm.1, hm.2.2 ▸ Nat.minFac_dvd m⟩

/-- **Per-gear cap, interval form.** Over the members of the slot prefix
`k < t` (all below `6t`), gear `q` roots at most `6t/q + 2` members: its
root class is inside its multiples, and the multiples of `q` in `[0, 6t)`
obey the interval bound of `BlockedSlots.card_blocked_by_le`. -/
theorem R_prefix_le (q t : ℕ) (hq : 0 < q) :
    R q (Bridge.members (Finset.range t)) ≤ 6 * t / q + 2 := by
  refine le_trans (R_le_card_multiples q _) (le_trans (Finset.card_le_card ?_)
    (BlockedSlots.card_blocked_by_le 0 q (6 * t) hq))
  intro m hm
  rw [Finset.mem_filter] at hm
  obtain ⟨hmem, hdvd⟩ := hm
  rw [Finset.mem_filter, Finset.mem_range]
  refine ⟨?_, by simpa using hdvd⟩
  rw [Bridge.members, Finset.mem_union] at hmem
  rcases hmem with h | h
  · obtain ⟨k, hk, rfl⟩ := Finset.mem_image.mp h
    have hkt := Finset.mem_range.mp hk
    simp only [Census.lo]
    omega
  · obtain ⟨k, hk, rfl⟩ := Finset.mem_image.mp h
    have hkt := Finset.mem_range.mp hk
    simp only [Census.hi]
    omega

/-- A composite rooted at `q` is at least `q^2`. -/
theorem sq_le_of_minFac_eq {q m : ℕ} (h1 : 1 < m) (hnp : ¬ m.Prime)
    (hfac : m.minFac = q) : q * q ≤ m := by
  have h := Nat.minFac_sq_le_self (by omega) hnp
  calc q * q = m.minFac * m.minFac := by rw [hfac]
    _ ≤ m := by simpa [pow_two] using h

/-- **The shadow law.** A gear supplies nothing below its own square: if
every member of `S` is above 1 and below `q * q`, gear `q`'s ledger line is
empty. (The `1 < m` guard matters: `minFac 0 = 2` would otherwise put `0`
in gear 2's class.) -/
theorem R_eq_zero_of_below_sq {q : ℕ} {S : Finset ℕ}
    (hS : ∀ m ∈ S, 1 < m ∧ m < q * q) : R q S = 0 := by
  simp only [R, Finset.card_eq_zero, Finset.filter_eq_empty_iff]
  rintro m hm ⟨hnp, hfac⟩
  obtain ⟨h1, hlt⟩ := hS m hm
  exact absurd (sq_le_of_minFac_eq h1 hnp hfac) (not_le.mpr hlt)

end Gear
