/-
The bridge identity: supply (root partition over members) equals demand
(slot census), kernel-checked end to end.

`Supply.card_composites_eq_sum_roots` counts a window's composite members by
their root gear: `#composites = Σ_{p < y prime} R_p`. `Census.comps_eq`
counts the same composites slot by slot: `C = n1 + 2 n2`. This file supplies
the missing leg - that both count the SAME Finset. The members of a slot set
`T` are `T.image lo ∪ T.image hi`; `lo` and `hi` are injective and their
images are disjoint (values `≡ 5` and `≡ 1 mod 6` respectively), so each
slot contributes its two members distinctly and the member-side filter card
splits into the two slot-side filter cards, which sum to `compsIn T`.

The result is the formal LHS skeleton of the X-consistency equation:

  Σ_{p < y prime} R_p(T) = n1(T) + 2 n2(T)          (`sum_roots_eq_census`)

and under Condition X (`n0 = 0`) the right side is pinned by the census:

  Σ_{p < y prime} R_p(T) = P + 2 (N - P)            (`sum_roots_pinned`)

The only subtlety flagged in the ledger - a slot's two composite members
must not collapse - is handled at the count level by disjointness of the
`lo`/`hi` images, and at the root level by `slot_roots_ne`: the two members
of a slot never share a root (`Supply.roots_ne` at `m = 6k - 1`).
-/

import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise
import Supply
import Census

namespace Bridge

open Census (lo hi)

/-- The members carried by a slot set: `6k - 1` and `6k + 1` for each slot. -/
def members (T : Finset ℕ) : Finset ℕ := T.image lo ∪ T.image hi

theorem lo_injective : Function.Injective lo := fun a b h => by
  simp only [lo] at h; omega

theorem hi_injective : Function.Injective hi := fun a b h => by
  simp only [hi] at h; omega

/-- Lower members sit in residue 5, upper members in residue 1 (mod 6):
the two member families never meet. -/
theorem lo_ne_hi (a b : ℕ) : lo a ≠ hi b := fun h => by
  simp only [lo, hi] at h; omega

theorem disjoint_lo_hi (S T : Finset ℕ) : Disjoint (S.image lo) (T.image hi) := by
  rw [Finset.disjoint_left]
  intro x hx hy
  obtain ⟨a, -, rfl⟩ := Finset.mem_image.mp hx
  obtain ⟨b, -, hb⟩ := Finset.mem_image.mp hy
  exact lo_ne_hi a b hb.symm

/-- Each slot contributes exactly two members. -/
theorem card_members (T : Finset ℕ) : (members T).card = 2 * T.card := by
  rw [members, Finset.card_union_of_disjoint (disjoint_lo_hi T T),
    Finset.card_image_of_injective _ lo_injective,
    Finset.card_image_of_injective _ hi_injective]
  omega

/-- **Count bridge, composite side.** The member-side composite count is the
slot-side `C`: both count the same members, grouped differently. -/
theorem card_comps_members (T : Finset ℕ) :
    ((members T).filter fun m => ¬ m.Prime).card = Census.compsIn T := by
  classical
  rw [members, Finset.filter_union,
    Finset.card_union_of_disjoint
      (Finset.disjoint_filter_filter (disjoint_lo_hi T T)),
    Finset.filter_image, Finset.filter_image,
    Finset.card_image_of_injective _ lo_injective,
    Finset.card_image_of_injective _ hi_injective,
    Finset.card_filter, Finset.card_filter,
    ← Finset.sum_add_distrib]
  simp only [Census.compsIn]
  exact Finset.sum_congr rfl fun k _ => by
    by_cases h1 : (lo k).Prime <;> by_cases h2 : (hi k).Prime <;>
      simp [Census.slotComps, h1, h2]

/-- **Count bridge, prime side.** Likewise the member-side prime count is
the slot-side `P`. -/
theorem card_primes_members (T : Finset ℕ) :
    ((members T).filter fun m => m.Prime).card = Census.primesIn T := by
  classical
  rw [members, Finset.filter_union,
    Finset.card_union_of_disjoint
      (Finset.disjoint_filter_filter (disjoint_lo_hi T T)),
    Finset.filter_image, Finset.filter_image,
    Finset.card_image_of_injective _ lo_injective,
    Finset.card_image_of_injective _ hi_injective,
    Finset.card_filter, Finset.card_filter,
    ← Finset.sum_add_distrib]
  simp only [Census.primesIn]
  exact Finset.sum_congr rfl fun k _ => by
    by_cases h1 : (lo k).Prime <;> by_cases h2 : (hi k).Prime <;>
      simp [Census.slotPrimes, h1, h2]

/-- Window transfer: slot-level window bounds give member-level bounds. -/
theorem members_window {y : ℕ} {T : Finset ℕ}
    (hwin : ∀ k ∈ T, y < lo k ∧ hi k < y * y) :
    ∀ m ∈ members T, y < m ∧ m < y * y := by
  intro m hm
  rw [members, Finset.mem_union] at hm
  rcases hm with hm | hm
  · obtain ⟨k, hk, rfl⟩ := Finset.mem_image.mp hm
    obtain ⟨h1, h2⟩ := hwin k hk
    have hle : lo k ≤ hi k := by simp only [lo, hi]; omega
    exact ⟨h1, lt_of_le_of_lt hle h2⟩
  · obtain ⟨k, hk, rfl⟩ := Finset.mem_image.mp hm
    obtain ⟨h1, h2⟩ := hwin k hk
    have hle : lo k ≤ hi k := by simp only [lo, hi]; omega
    exact ⟨lt_of_lt_of_le h1 hle, h2⟩

/-- **The bridge identity.** Over any slot set inside the window, the supply
side (root partition over members) equals the demand side (slot census):
`Σ_{p < y prime} R_p = n1 + 2 n2`. -/
theorem sum_roots_eq_census {y : ℕ} (T : Finset ℕ)
    (hwin : ∀ k ∈ T, y < lo k ∧ hi k < y * y) :
    (∑ p ∈ (Finset.range y).filter Nat.Prime,
      ((members T).filter fun m => ¬ m.Prime ∧ m.minFac = p).card)
      = Census.n1 T + 2 * Census.n2 T := by
  rw [← Supply.card_composites_eq_sum_roots (members T) (members_window hwin),
    ← Census.comps_eq]
  exact card_comps_members T

/-- **The X-consistency LHS, pinned.** Under Condition X (`n0 = 0`) the
supply must meet the pinned demand exactly: `Σ_p R_p = P + 2 (N - P)`. -/
theorem sum_roots_pinned {y : ℕ} (T : Finset ℕ)
    (hwin : ∀ k ∈ T, y < lo k ∧ hi k < y * y)
    (h0 : Census.n0 T = 0) :
    (∑ p ∈ (Finset.range y).filter Nat.Prime,
      ((members T).filter fun m => ¬ m.Prime ∧ m.minFac = p).card)
      = Census.primesIn T + 2 * (T.card - Census.primesIn T) := by
  have hb := sum_roots_eq_census T hwin
  have hp := Census.census_pinned T h0
  omega

/-- The two members of a slot never share a root (`k ≥ 1`): the pair differs
by 2 and the lower member is odd, so a common least factor would be an odd
gear dividing both, against the slot cap. This is what keeps the per-gear
ledgers overlap-free even at double slots. -/
theorem slot_roots_ne {k : ℕ} (hk : 1 ≤ k) :
    (lo k).minFac ≠ (hi k).minFac := by
  have hlo : 1 < lo k := by simp only [lo]; omega
  have hodd : ¬ 2 ∣ lo k := by simp only [lo]; omega
  have h2 : hi k = lo k + 2 := by simp only [lo, hi]; omega
  rw [h2]
  exact Supply.roots_ne hlo hodd

end Bridge
