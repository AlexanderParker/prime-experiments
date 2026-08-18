/-
Placement of one gear's supply line: which slots the semiprimes sit in.

`Gear.lean` settled HOW MANY a line supplies (`R q S = #(partners q S)`,
partner primes exactly, in the large-gear regime). This file settles WHERE.

Everything runs on the mod-6 sign structure: primes `≥ 5` live in the unit
classes `±1 mod 6` (`prime_mod_six`), and on those classes signs multiply
(`sign_law`) - so a semiprime `q * c` of two such primes is itself `±1
mod 6` and is therefore a genuine slot member: the lower member of slot
`(q*c + 1)/6` when `q*c ≡ 5`, the upper when `q*c ≡ 1`. The slot function
`slotOf m = (m + 1) / 6` recovers the slot from either member.

The placement map `c ↦ slotOf (q * c)` is injective on the partner set: two
partners in one slot would put two multiples of `q` at distance ≤ 2, against
the slot cap. So the line occupies exactly `R q S` distinct slots
(`card_slots_of_line`), and over a slot interval the line's size is an
explicit count of primes in an interval (`R_slots_eq`). The ledger now knows
both the count and the position of every supply line in the regime.
-/

import Gear

namespace Placement

/-- Primes at or above 5 live in the unit classes `±1 mod 6`. -/
theorem prime_mod_six {p : ℕ} (hp : p.Prime) (h5 : 5 ≤ p) :
    p % 6 = 1 ∨ p % 6 = 5 := by
  have h2 : ¬ (2 ∣ p) := by
    intro h
    rcases hp.eq_one_or_self_of_dvd 2 h with h' | h' <;> omega
  have h3 : ¬ (3 ∣ p) := by
    intro h
    rcases hp.eq_one_or_self_of_dvd 3 h with h' | h' <;> omega
  omega

/-- **The sign law mod 6.** On the unit classes `±1`, signs multiply: the
product is `+1` exactly when the two signs agree. -/
theorem sign_law {a b : ℕ} (ha : a % 6 = 1 ∨ a % 6 = 5)
    (hb : b % 6 = 1 ∨ b % 6 = 5) :
    ((a * b) % 6 = 1 ↔ a % 6 = b % 6) := by
  rw [Nat.mul_mod]
  rcases ha with h | h <;> rcases hb with h' | h' <;> rw [h, h'] <;> decide

/-- The unit classes are closed under multiplication. -/
theorem unit_mul {a b : ℕ} (ha : a % 6 = 1 ∨ a % 6 = 5)
    (hb : b % 6 = 1 ∨ b % 6 = 5) :
    (a * b) % 6 = 1 ∨ (a * b) % 6 = 5 := by
  rw [Nat.mul_mod]
  rcases ha with h | h <;> rcases hb with h' | h' <;> rw [h, h'] <;> decide

/-- **The slot of a member.** Both members of slot `k` recover `k`:
`(6k - 1 + 1)/6 = (6k + 1 + 1)/6 = k`. -/
def slotOf (m : ℕ) : ℕ := (m + 1) / 6

theorem slotOf_lo {k : ℕ} (hk : 1 ≤ k) : slotOf (Census.lo k) = k := by
  simp only [slotOf, Census.lo]; omega

theorem slotOf_hi (k : ℕ) : slotOf (Census.hi k) = k := by
  simp only [slotOf, Census.hi]; omega

/-- A `≡ 5 (mod 6)` number is the lower member of its slot. -/
theorem lo_slotOf {m : ℕ} (hm : m % 6 = 5) : Census.lo (slotOf m) = m := by
  simp only [slotOf, Census.lo]; omega

/-- A `≡ 1 (mod 6)` number is the upper member of its slot. -/
theorem hi_slotOf {m : ℕ} (hm : m % 6 = 1) : Census.hi (slotOf m) = m := by
  simp only [slotOf, Census.hi]; omega

/-- Membership among a slot set's members is exactly slot membership, for
numbers in the unit classes. -/
theorem mem_members_iff_slot {T : Finset ℕ} {m : ℕ}
    (hm : m % 6 = 1 ∨ m % 6 = 5) :
    m ∈ Bridge.members T ↔ slotOf m ∈ T := by
  constructor
  · intro h
    rw [Bridge.members, Finset.mem_union] at h
    rcases h with h | h
    · obtain ⟨k, hkT, rfl⟩ := Finset.mem_image.mp h
      have hk : slotOf (Census.lo k) = k := by
        simp only [slotOf, Census.lo] at hm ⊢
        omega
      rwa [hk]
    · obtain ⟨k, hkT, rfl⟩ := Finset.mem_image.mp h
      rwa [slotOf_hi k]
  · intro h
    rw [Bridge.members, Finset.mem_union]
    rcases hm with h1 | h5
    · exact Or.inr (Finset.mem_image.mpr ⟨slotOf m, h, hi_slotOf h1⟩)
    · exact Or.inl (Finset.mem_image.mpr ⟨slotOf m, h, lo_slotOf h5⟩)

/-- **The placement map is injective** on a gear's partner set: two partners
in the same slot would put two multiples of `q ≥ 5` at distance at most 2,
against the slot cap. -/
theorem slot_injOn_partners {q : ℕ} (hq : q.Prime) (h5 : 5 ≤ q) {S : Finset ℕ}
    (hS : ∀ m ∈ S, 1 < m ∧ m < q * q * q) :
    Set.InjOn (fun c => slotOf (q * c)) (Gear.partners q S) := by
  intro c1 hc1 c2 hc2 heq
  rw [Finset.mem_coe, Gear.mem_partners hq hS] at hc1 hc2
  obtain ⟨hp1, hq1, hm1⟩ := hc1
  obtain ⟨hp2, hq2, hm2⟩ := hc2
  simp only at heq
  have hcl1 := unit_mul (prime_mod_six hq h5) (prime_mod_six hp1 (le_trans h5 hq1))
  have hcl2 := unit_mul (prime_mod_six hq h5) (prime_mod_six hp2 (le_trans h5 hq2))
  have h25₁ : 5 * 5 ≤ q * c1 := Nat.mul_le_mul h5 (le_trans h5 hq1)
  have h25₂ : 5 * 5 ≤ q * c2 := Nat.mul_le_mul h5 (le_trans h5 hq2)
  rcases hcl1 with h1 | h1 <;> rcases hcl2 with h2 | h2
  · -- both ≡ 1: upper members of the same slot
    have e1 := hi_slotOf h1
    have e2 := hi_slotOf h2
    rw [heq] at e1
    exact Nat.eq_of_mul_eq_mul_left hq.pos (e1.symm.trans e2)
  · -- mixed: q would divide both members of one slot
    exfalso
    have e1 := hi_slotOf h1
    have e2 := lo_slotOf h2
    rw [heq] at e1
    refine Layer.slot_cap (q := q) (m := q * c2) (by omega) ⟨dvd_mul_right q c2, ?_⟩
    have h : q * c2 + 2 = q * c1 := by
      simp only [Census.lo] at e2
      simp only [Census.hi] at e1
      omega
    rw [h]; exact dvd_mul_right q c1
  · -- mixed, the other way
    exfalso
    have e1 := lo_slotOf h1
    have e2 := hi_slotOf h2
    rw [heq] at e1
    refine Layer.slot_cap (q := q) (m := q * c1) (by omega) ⟨dvd_mul_right q c1, ?_⟩
    have h : q * c1 + 2 = q * c2 := by
      simp only [Census.lo] at e1
      simp only [Census.hi] at e2
      omega
    rw [h]; exact dvd_mul_right q c2
  · -- both ≡ 5: lower members of the same slot
    have e1 := lo_slotOf h1
    have e2 := lo_slotOf h2
    rw [heq] at e1
    exact Nat.eq_of_mul_eq_mul_left hq.pos (e1.symm.trans e2)

/-- **One slot per supply member.** The line occupies exactly `R q S`
distinct slots. -/
theorem card_slots_of_line {q : ℕ} (hq : q.Prime) (h5 : 5 ≤ q) {S : Finset ℕ}
    (hS : ∀ m ∈ S, 1 < m ∧ m < q * q * q) :
    ((Gear.partners q S).image fun c => slotOf (q * c)).card = Gear.R q S := by
  rw [Finset.card_image_of_injOn (slot_injOn_partners hq h5 hS),
    Gear.R_eq_card_partners]

/-- **The placed count.** Over the slot interval `[1, t)` (slot 0 is
degenerate), gear `q`'s line is an explicit count of primes: the `c ≥ q`
prime with `q * c` landing in a slot of the interval. The regime hypothesis
`6t ≤ q^3` keeps every member below `q^3`. -/
theorem R_slots_eq {q t : ℕ} (hq : q.Prime) (h5 : 5 ≤ q)
    (hcube : 6 * t ≤ q * q * q) :
    Gear.R q (Bridge.members (Finset.Ico 1 t))
      = ((Finset.range (6 * t)).filter fun c =>
          c.Prime ∧ q ≤ c ∧ slotOf (q * c) ∈ Finset.Ico 1 t).card := by
  have hS : ∀ m ∈ Bridge.members (Finset.Ico 1 t), 1 < m ∧ m < q * q * q := by
    intro m hm
    rw [Bridge.members, Finset.mem_union] at hm
    rcases hm with h | h
    · obtain ⟨k, hk, rfl⟩ := Finset.mem_image.mp h
      have hk' := Finset.mem_Ico.mp hk
      refine ⟨by simp only [Census.lo]; omega, lt_of_lt_of_le ?_ hcube⟩
      simp only [Census.lo]; omega
    · obtain ⟨k, hk, rfl⟩ := Finset.mem_image.mp h
      have hk' := Finset.mem_Ico.mp hk
      refine ⟨by simp only [Census.hi]; omega, lt_of_lt_of_le ?_ hcube⟩
      simp only [Census.hi]; omega
  rw [Gear.R_eq_card_partners]
  congr 1
  ext c
  rw [Gear.mem_partners hq hS, Finset.mem_filter, Finset.mem_range]
  constructor
  · rintro ⟨hcp, hqc, hmem⟩
    have hclass := unit_mul (prime_mod_six hq h5) (prime_mod_six hcp (le_trans h5 hqc))
    have hslot : slotOf (q * c) ∈ Finset.Ico 1 t :=
      (mem_members_iff_slot hclass).mp hmem
    have hlt : q * c < 6 * t := by
      have hb := Finset.mem_Ico.mp hslot
      simp only [slotOf] at hb
      omega
    exact ⟨lt_of_le_of_lt (Nat.le_mul_of_pos_left c hq.pos) hlt, hcp, hqc, hslot⟩
  · rintro ⟨-, hcp, hqc, hslot⟩
    have hclass := unit_mul (prime_mod_six hq h5) (prime_mod_six hcp (le_trans h5 hqc))
    exact ⟨hcp, hqc, (mem_members_iff_slot hclass).mpr hslot⟩

end Placement
