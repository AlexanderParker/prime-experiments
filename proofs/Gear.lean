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

/-! ## The semiprime refinement: one gear's line, exactly

In the large-gear regime - every member below `q^3` - a gear's ledger line
is freedom-free: each member rooted at `q` is `q * c` with `c` prime and
`q ≤ c` (equality only at the square `q * q`). The strict case is Layer's
semiprime shape; the square is the shadow-law onset. So the line is in
bijection with its PARTNER PRIMES via `m ↦ m / q`, and `R q S` is exactly a
count of primes: the first exact formula of the supply side.
-/

/-- Fiber members decompose: a composite rooted at `q` below `q^3` is
`q * c` with `c` prime and `q ≤ c` - equality exactly at the square. -/
theorem semiprime_of_fiber {q m : ℕ} (hq : q.Prime) (h1 : 1 < m)
    (hnp : ¬ m.Prime) (hfac : m.minFac = q) (hcube : m < q * q * q) :
    ∃ c, c.Prime ∧ q ≤ c ∧ m = q * c := by
  have hsq : q * q ≤ m := sq_le_of_minFac_eq h1 hnp hfac
  rcases eq_or_lt_of_le hsq with heq | hlt
  · exact ⟨q, hq, le_rfl, heq.symm⟩
  · obtain ⟨c, hc, hqc, hm⟩ := Layer.eq_mul_prime_of_minFac_eq h1 hfac hlt hcube
    exact ⟨c, hc, le_of_lt hqc, hm⟩

/-- A product of two primes is composite. -/
theorem not_prime_mul {q c : ℕ} (hq : q.Prime) (hc : c.Prime) :
    ¬ (q * c).Prime := by
  intro hp
  rcases hp.eq_one_or_self_of_dvd q ⟨c, rfl⟩ with h | h
  · exact hq.ne_one h
  · have h' : q * 1 = q * c := by simpa using h
    exact hc.ne_one (Nat.eq_of_mul_eq_mul_left hq.pos h').symm

/-- The least factor of a product of two primes is the smaller one. -/
theorem minFac_mul {q c : ℕ} (hq : q.Prime) (hc : c.Prime) (hqc : q ≤ c) :
    (q * c).minFac = q := by
  have hne : q * c ≠ 1 := fun h => hq.ne_one (Nat.dvd_one.mp ⟨c, h.symm⟩)
  have hp' : (q * c).minFac.Prime := Nat.minFac_prime hne
  have hple : (q * c).minFac ≤ q := Nat.minFac_le_of_dvd hq.two_le ⟨c, rfl⟩
  rcases (Nat.Prime.dvd_mul hp').mp (Nat.minFac_dvd _) with h | h
  · exact (Nat.prime_dvd_prime_iff_eq hp' hq).mp h
  · have heqc : (q * c).minFac = c := (Nat.prime_dvd_prime_iff_eq hp' hc).mp h
    omega

/-- **The partner primes** of gear `q` in `S`: the cofactors of its ledger
line under `m ↦ m / q`. -/
def partners (q : ℕ) (S : Finset ℕ) : Finset ℕ :=
  (S.filter fun m => ¬ m.Prime ∧ m.minFac = q).image (· / q)

/-- The line and its partner set are in bijection - `m ↦ m / q` is
injective on the fiber, since every fiber member is a multiple of `q`.
No range hypotheses needed. -/
theorem R_eq_card_partners (q : ℕ) (S : Finset ℕ) :
    R q S = (partners q S).card := by
  have hinj : Set.InjOn (· / q)
      (S.filter fun m => ¬ m.Prime ∧ m.minFac = q) := by
    intro a ha b hb hab
    simp only [Finset.coe_filter, Set.mem_ofPred_eq] at ha hb
    have hda : q ∣ a := ha.2.2 ▸ Nat.minFac_dvd a
    have hdb : q ∣ b := hb.2.2 ▸ Nat.minFac_dvd b
    simp only at hab
    calc a = q * (a / q) := (Nat.mul_div_cancel' hda).symm
      _ = q * (b / q) := by rw [hab]
      _ = b := Nat.mul_div_cancel' hdb
  simp only [R, partners, Finset.card_image_of_injOn hinj]

/-- **Exact membership of the partner set** (large-gear regime): the
partners of `q` are precisely the primes `c ≥ q` whose product `q * c`
is a member. This is the freedom-free description of one supply line. -/
theorem mem_partners {q c : ℕ} (hq : q.Prime) {S : Finset ℕ}
    (hS : ∀ m ∈ S, 1 < m ∧ m < q * q * q) :
    c ∈ partners q S ↔ c.Prime ∧ q ≤ c ∧ q * c ∈ S := by
  constructor
  · intro hc
    obtain ⟨m, hm, rfl⟩ := Finset.mem_image.mp hc
    rw [Finset.mem_filter] at hm
    obtain ⟨hmS, hnp, hfac⟩ := hm
    obtain ⟨h1, hcube⟩ := hS m hmS
    obtain ⟨c', hc', hqc', heq⟩ := semiprime_of_fiber hq h1 hnp hfac hcube
    have hdiv : m / q = c' := by
      rw [heq]; exact Nat.mul_div_cancel_left c' hq.pos
    rw [hdiv]
    exact ⟨hc', hqc', heq ▸ hmS⟩
  · rintro ⟨hcp, hqc, hmem⟩
    simp only [partners, Finset.mem_image]
    refine ⟨q * c, ?_, Nat.mul_div_cancel_left c hq.pos⟩
    rw [Finset.mem_filter]
    exact ⟨hmem, not_prime_mul hq hcp, minFac_mul hq hcp hqc⟩

/-- Adapter: window bounds plus the large-gear condition `y^2 ≤ q^3` give
the member bounds that `mem_partners` wants. -/
theorem window_bounds {q y : ℕ} {S : Finset ℕ}
    (hwin : ∀ m ∈ S, y < m ∧ m < y * y) (hy : 1 ≤ y)
    (hthin : y * y ≤ q * q * q) : ∀ m ∈ S, 1 < m ∧ m < q * q * q :=
  fun m hm => ⟨by have := (hwin m hm).1; omega,
    lt_of_lt_of_le (hwin m hm).2 hthin⟩

end Gear
