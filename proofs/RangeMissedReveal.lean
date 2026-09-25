import RangeMissed

/-!
# Range missed copy: the reveal rule

Here `g` is a prime gear with `g ≥ 7`. Its missed copy (see `RangeMissed`) has two legs,
`g² + A` and `g² + B`, where the offsets are

* `(A, B) = (28, 30)` in case 1 (`g² % 30 = 1`), and
* `(A, B) = (10, 12)` in case 19 (`g² % 30 = 19`).

The predicate `MissedOffsets g A B` says exactly this; `missedA g`, `missedB g` pick the
offsets from `g` and `missedOffsets_missed` shows they satisfy it for every prime `g ≥ 7`.

This file proves the reveal rule for the missed copy:

* `legs_mod30`: the lower leg is `29` and the upper leg is `1` modulo `30`.
* `legs_coprime_30` (a): each leg is odd, not divisible by `3`, not divisible by `5`,
  and so coprime to `30`. This needs only the offsets, not primality of `g`.
* `composite_leg_small_factor` (b): if a leg `L` is not prime, then some prime `h` with
  `7 ≤ h < g` divides `L`. The witness is the least prime factor `h` of `L`:
  it is not `2`, `3`, `5` by (a), so `h ≥ 7`; it is not `g`, because the gear misses its
  own copy; and it is not above `g`, because a prime `h > g` has `h ≥ g + 2`, so
  `h² > g² + 30 ≥ L`, while the least factor of a composite `L` has `h² ≤ L`.
* `missed_copy_revealed_iff` (c): both legs are prime exactly when no prime `h` with
  `7 ≤ h < g` divides either leg. The copy is revealed as a twin pair precisely when every
  gear from `7` up to, but not including, `g` misses both legs.
* `missed_copy_revealed_iff_case1`, `missed_copy_revealed_iff_case19`: (c) with the
  offsets written out.
-/

namespace RangeLine

/-- The offsets of the missed copy of `g`: `(A, B) = (28, 30)` when `g² % 30 = 1`, and
`(A, B) = (10, 12)` when `g² % 30 = 19`. The legs of the copy are `g² + A` and `g² + B`. -/
def MissedOffsets (g A B : ℕ) : Prop :=
  (g ^ 2 % 30 = 1 ∧ A = 28 ∧ B = 30) ∨ (g ^ 2 % 30 = 19 ∧ A = 10 ∧ B = 12)

/-- The lower offset of the missed copy of `g`: `28` when `g² % 30 = 1`, else `10`. -/
def missedA (g : ℕ) : ℕ := if g ^ 2 % 30 = 1 then 28 else 10

/-- The upper offset of the missed copy of `g`: `30` when `g² % 30 = 1`, else `12`. -/
def missedB (g : ℕ) : ℕ := if g ^ 2 % 30 = 1 then 30 else 12

/-- For a prime `g ≥ 7`, the offsets `missedA g`, `missedB g` are the offsets of its
missed copy. -/
theorem missedOffsets_missed (g : ℕ) (hg : g.Prime) (h7 : 7 ≤ g) :
    MissedOffsets g (missedA g) (missedB g) := by
  unfold MissedOffsets missedA missedB
  rcases sq_mod30_cases g hg h7 with h | h
  · exact Or.inl ⟨h, by simp [h], by simp [h]⟩
  · exact Or.inr ⟨h, by simp [h], by simp [h]⟩

/-- The lower leg `g² + A` is `29` modulo `30` and the upper leg `g² + B` is `1`
modulo `30`. -/
theorem legs_mod30 (g A B : ℕ) (hAB : MissedOffsets g A B) :
    (g ^ 2 + A) % 30 = 29 ∧ (g ^ 2 + B) % 30 = 1 := by
  rcases hAB with ⟨h, rfl, rfl⟩ | ⟨h, rfl, rfl⟩ <;>
  · generalize g ^ 2 = s at h ⊢
    omega

/-- A number that is `29` or `1` modulo `30` is odd, not divisible by `3`, not divisible
by `5`, and coprime to `30`. -/
theorem mod30_unit_facts (L : ℕ) (h : L % 30 = 29 ∨ L % 30 = 1) :
    L % 2 = 1 ∧ L % 3 ≠ 0 ∧ L % 5 ≠ 0 ∧ Nat.Coprime L 30 := by
  refine ⟨by omega, by omega, by omega, ?_⟩
  have hg : Nat.gcd 30 L = 1 := by
    rw [Nat.gcd_rec]
    rcases h with h | h <;> rw [h] <;> norm_num
  exact Nat.Coprime.symm hg

/-- (a) Both legs of the missed copy are coprime to `30`: each of `g² + A`, `g² + B` is
odd, not divisible by `3`, and not divisible by `5`. -/
theorem legs_coprime_30 (g A B : ℕ) (hAB : MissedOffsets g A B) :
    ((g ^ 2 + A) % 2 = 1 ∧ (g ^ 2 + A) % 3 ≠ 0 ∧ (g ^ 2 + A) % 5 ≠ 0 ∧
        Nat.Coprime (g ^ 2 + A) 30) ∧
      ((g ^ 2 + B) % 2 = 1 ∧ (g ^ 2 + B) % 3 ≠ 0 ∧ (g ^ 2 + B) % 5 ≠ 0 ∧
        Nat.Coprime (g ^ 2 + B) 30) := by
  obtain ⟨hA, hB⟩ := legs_mod30 g A B hAB
  exact ⟨mod30_unit_facts _ (Or.inl hA), mod30_unit_facts _ (Or.inr hB)⟩

/-- A prime that divides an odd number not divisible by `3` or `5` is at least `7`. -/
theorem prime_dvd_ge7 (L h : ℕ) (h2 : L % 2 = 1) (h3 : L % 3 ≠ 0) (h5 : L % 5 ≠ 0)
    (hh : h.Prime) (hd : h ∣ L) : 7 ≤ h := by
  by_contra hlt
  have htwo := hh.two_le
  obtain ⟨k, rfl⟩ := hd
  interval_cases h <;> omega

/-- Each leg of the missed copy is at most `g² + 30`. -/
theorem leg_le (g A B L : ℕ) (hAB : MissedOffsets g A B)
    (hLeg : L = g ^ 2 + A ∨ L = g ^ 2 + B) : L ≤ g ^ 2 + 30 := by
  rcases hAB with ⟨_, rfl, rfl⟩ | ⟨_, rfl, rfl⟩ <;> rcases hLeg with rfl | rfl <;> omega

/-- The gear `g` divides neither leg of its own missed copy. -/
theorem gear_not_dvd_leg (g A B L : ℕ) (hg : g.Prime) (h7 : 7 ≤ g)
    (hAB : MissedOffsets g A B) (hLeg : L = g ^ 2 + A ∨ L = g ^ 2 + B) : ¬ g ∣ L := by
  have hm := gear_misses_own_copy g hg h7
  rcases hAB with ⟨hc, rfl, rfl⟩ | ⟨hc, rfl, rfl⟩ <;> rcases hLeg with rfl | rfl
  · exact (hm.1 hc).1
  · exact (hm.1 hc).2
  · exact (hm.2 hc).1
  · exact (hm.2 hc).2

/-- (b) If a leg `L` of the missed copy of a prime gear `g ≥ 7` is not prime, then some
prime `h` with `7 ≤ h < g` divides `L`. (The least prime factor of `L` is such an `h`;
`L > 1` holds automatically since `L ≥ g² ≥ 49`.) -/
theorem composite_leg_small_factor (g A B L : ℕ) (hg : g.Prime) (h7 : 7 ≤ g)
    (hAB : MissedOffsets g A B) (hLeg : L = g ^ 2 + A ∨ L = g ^ 2 + B)
    (hnp : ¬ L.Prime) :
    ∃ h, h.Prime ∧ 7 ≤ h ∧ h < g ∧ h ∣ L := by
  have hsq : 49 ≤ g ^ 2 := by nlinarith
  have hL1 : 1 < L := by rcases hLeg with rfl | rfl <;> omega
  have hmp : L.minFac.Prime := Nat.minFac_prime (by omega)
  have hmd : L.minFac ∣ L := Nat.minFac_dvd L
  have hres : L % 30 = 29 ∨ L % 30 = 1 := by
    obtain ⟨hA, hB⟩ := legs_mod30 g A B hAB
    rcases hLeg with rfl | rfl
    · exact Or.inl hA
    · exact Or.inr hB
  obtain ⟨h2, h3, h5, _⟩ := mod30_unit_facts L hres
  refine ⟨L.minFac, hmp, prime_dvd_ge7 L _ h2 h3 h5 hmp hmd, ?_, hmd⟩
  by_contra hge
  rcases (Nat.le_of_not_lt hge).lt_or_eq with hlt | heq
  · have hbig := larger_gear_sq g L.minFac hg hmp h7 hlt
    have hsmall := Nat.minFac_sq_le_self (by omega) hnp
    have hle := leg_le g A B L hAB hLeg
    omega
  · exact gear_not_dvd_leg g A B L hg h7 hAB hLeg (heq ▸ hmd)

/-- (c) The reveal rule: both legs `g² + A` and `g² + B` of the missed copy of a prime
gear `g ≥ 7` are prime if and only if no prime `h` with `7 ≤ h < g` divides either leg. -/
theorem missed_copy_revealed_iff (g A B : ℕ) (hg : g.Prime) (h7 : 7 ≤ g)
    (hAB : MissedOffsets g A B) :
    ((g ^ 2 + A).Prime ∧ (g ^ 2 + B).Prime) ↔
      ¬ ∃ h, h.Prime ∧ 7 ≤ h ∧ h < g ∧ (h ∣ g ^ 2 + A ∨ h ∣ g ^ 2 + B) := by
  have hgsq : g ≤ g ^ 2 := by nlinarith
  constructor
  · rintro ⟨pA, pB⟩ ⟨h, hh, _, hlt, hd | hd⟩
    · have := (Nat.prime_dvd_prime_iff_eq hh pA).mp hd
      omega
    · have := (Nat.prime_dvd_prime_iff_eq hh pB).mp hd
      omega
  · intro hno
    by_contra hnot
    rcases not_and_or.mp hnot with hA | hB
    · obtain ⟨h, hh, h7', hlt, hd⟩ :=
        composite_leg_small_factor g A B _ hg h7 hAB (Or.inl rfl) hA
      exact hno ⟨h, hh, h7', hlt, Or.inl hd⟩
    · obtain ⟨h, hh, h7', hlt, hd⟩ :=
        composite_leg_small_factor g A B _ hg h7 hAB (Or.inr rfl) hB
      exact hno ⟨h, hh, h7', hlt, Or.inr hd⟩

/-- (c, case 1) If `g² % 30 = 1`, then `g² + 28` and `g² + 30` are both prime if and only
if no prime `h` with `7 ≤ h < g` divides either of them. -/
theorem missed_copy_revealed_iff_case1 (g : ℕ) (hg : g.Prime) (h7 : 7 ≤ g)
    (h1 : g ^ 2 % 30 = 1) :
    ((g ^ 2 + 28).Prime ∧ (g ^ 2 + 30).Prime) ↔
      ¬ ∃ h, h.Prime ∧ 7 ≤ h ∧ h < g ∧ (h ∣ g ^ 2 + 28 ∨ h ∣ g ^ 2 + 30) :=
  missed_copy_revealed_iff g 28 30 hg h7 (Or.inl ⟨h1, rfl, rfl⟩)

/-- (c, case 19) If `g² % 30 = 19`, then `g² + 10` and `g² + 12` are both prime if and
only if no prime `h` with `7 ≤ h < g` divides either of them. -/
theorem missed_copy_revealed_iff_case19 (g : ℕ) (hg : g.Prime) (h7 : 7 ≤ g)
    (h19 : g ^ 2 % 30 = 19) :
    ((g ^ 2 + 10).Prime ∧ (g ^ 2 + 12).Prime) ↔
      ¬ ∃ h, h.Prime ∧ 7 ≤ h ∧ h < g ∧ (h ∣ g ^ 2 + 10 ∨ h ∣ g ^ 2 + 12) :=
  missed_copy_revealed_iff g 10 12 hg h7 (Or.inr ⟨h19, rfl, rfl⟩)

end RangeLine
