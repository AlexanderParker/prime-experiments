/-
LadderWidening (2026-09-20): Lemma B (the widening rule) and the sandwich's lower half.

Columns `n ≥ 1` hold `(6n - 1, 6n + 1)`; gear `g` strikes column `n` when it divides a member.
LEMMA B: below its own square a gear strikes only its home column or columns a smaller gear
already strikes (`widening`).  Hence a twin slot of machine `q` stays unstruck in every machine
whose gears stay below its members (`twin_slot_persists`) - the widening never reaches back.

THE LOWER HALF OF THE RECORD SANDWICH: the pattern of machine `q` is periodic in any common
multiple `P` of its gears (`struck_periodic`); the next prime `q'` is coprime to `P`, so any
column can be translated by a multiple of `P` into the class `n ≡ invSix q' (mod q')` that `q'`
strikes (`tooth_of_class`).  A run of the old machine with a single hole therefore becomes a
fully struck run of the next machine at some translate (`align_single_hole`): `G_1(q) ≤ F(q')`,
i.e. `¬ MaxGapBelow q' L` (`not_maxGapBelow_of_single_hole`).
-/
import LadderFields
import LadderMaxGap

namespace TwinLadder

/-- **Widening, member form**: a member `m` (coprime to 6, `m ≥ 5`) below `g²` that `g` divides
is `g` itself or has a prime factor `h` with `5 ≤ h < g`. -/
theorem widening_member {g m : ℕ} (hm6 : Nat.Coprime m 6)
    (hm5 : 5 ≤ m) (hlt : m < g ^ 2) (hd : g ∣ m) :
    m = g ∨ ∃ h, h.Prime ∧ 5 ≤ h ∧ h < g ∧ h ∣ m := by
  obtain ⟨k, rfl⟩ := hd
  have hk1 : 1 ≤ k := by
    rcases Nat.eq_zero_or_pos k with h0 | h0
    · subst h0; simp at hm5
    · exact h0
  have hkg : k < g := by
    by_contra hge
    push Not at hge
    have h1 : g * g ≤ g * k := Nat.mul_le_mul_left g hge
    rw [sq] at hlt
    exact absurd (lt_of_le_of_lt h1 hlt) (lt_irrefl _)
  rcases Nat.eq_or_lt_of_le hk1 with h1 | h1
  · left; rw [← h1, mul_one]
  · right
    have hk6 : Nat.Coprime k 6 := Nat.Coprime.coprime_dvd_left (dvd_mul_left k g) hm6
    obtain ⟨h, hh, hhk⟩ := Nat.exists_prime_and_dvd (show k ≠ 1 by omega)
    refine ⟨h, hh, ?_, ?_, ?_⟩
    · have hh6 : Nat.Coprime h 6 := Nat.Coprime.coprime_dvd_left hhk hk6
      by_contra hlt5
      push Not at hlt5
      have h2 := hh.two_le
      interval_cases h
      · exact absurd hh6 (by decide)
      · exact absurd hh6 (by decide)
      · exact absurd hh (by decide)
    · exact lt_of_le_of_lt (Nat.le_of_dvd (by omega) hhk) hkg
    · exact Dvd.dvd.mul_left hhk g

/-- **Lemma B, the widening rule**: below its own square a gear strikes only its home column
(a member equals `g`) or columns a smaller gear already strikes. -/
theorem widening {g n : ℕ} (hg : g.Prime) (h5 : 5 ≤ g) (hn : 1 ≤ n) (hlt : 6 * n + 1 < g ^ 2)
    (hs : StrikesBy g n) :
    (6 * n - 1 = g ∨ 6 * n + 1 = g) ∨ ∃ h, h.Prime ∧ 5 ≤ h ∧ h < g ∧ StrikesBy h n := by
  rcases hs with hs | hs
  · rcases widening_member (coprime_sub_six hn) (by omega) (by omega) hs with
      h | ⟨h, hh, h5h, hlt', hd⟩
    · exact Or.inl (Or.inl h)
    · exact Or.inr ⟨h, hh, h5h, hlt', Or.inl hd⟩
  · rcases widening_member (coprime_add_six n) (by omega) hlt hs with
      h | ⟨h, hh, h5h, hlt', hd⟩
    · exact Or.inl (Or.inr h)
    · exact Or.inr ⟨h, hh, h5h, hlt', Or.inr hd⟩

/-- A prime striking a twin column is one of its members. -/
theorem twin_column_strikers {n p : ℕ} (ht : TwinCentre (6 * n)) (hp : p.Prime)
    (hs : StrikesBy p n) : p = 6 * n - 1 ∨ p = 6 * n + 1 := by
  obtain ⟨-, h1, h2⟩ := ht
  rcases hs with hs | hs
  · exact Or.inl ((Nat.prime_dvd_prime_iff_eq hp h1).1 hs)
  · exact Or.inr ((Nat.prime_dvd_prime_iff_eq hp h2).1 hs)

/-- **A twin slot persists**: a twin slot of machine `q` stays unstruck in every machine `q'`
whose gears stay below its members - the widening never reaches back. -/
theorem twin_slot_persists {q q' n : ℕ} (ht : TwinCentre (6 * n)) (hq : q < 6 * n - 1)
    (hq' : q' < 6 * n - 1) : ¬ Struck q' n := by
  rintro ⟨p, hp, hd⟩
  have hle : p ≤ q' := hp.2.2
  rcases twin_column_strikers ht hp.1 hd with h | h <;> omega

/-- **Periodicity**: the pattern of machine `q` repeats in any common multiple `P` of its gears
(columns `n ≥ 1`). -/
theorem struck_periodic {q n P k : ℕ} (hn : 1 ≤ n) (hP : ∀ p, Gear q p → p ∣ P) :
    Struck q (n + k * P) ↔ Struck q n := by
  have key : ∀ p, Gear q p →
      ((p ∣ 6 * (n + k * P) - 1 ∨ p ∣ 6 * (n + k * P) + 1) ↔
        (p ∣ 6 * n - 1 ∨ p ∣ 6 * n + 1)) := by
    intro p hp
    have hd : p ∣ 6 * (k * P) := Dvd.dvd.mul_left (Dvd.dvd.mul_left (hP p hp) k) 6
    have e1 : 6 * (n + k * P) - 1 = (6 * n - 1) + 6 * (k * P) := by omega
    have e2 : 6 * (n + k * P) + 1 = (6 * n + 1) + 6 * (k * P) := by omega
    rw [e1, e2]
    constructor
    · rintro (h | h)
      · exact Or.inl ((Nat.dvd_add_left hd).1 h)
      · exact Or.inr ((Nat.dvd_add_left hd).1 h)
    · rintro (h | h)
      · exact Or.inl ((Nat.dvd_add_left hd).2 h)
      · exact Or.inr ((Nat.dvd_add_left hd).2 h)
  constructor
  · rintro ⟨p, hp, hd⟩; exact ⟨p, hp, (key p hp).1 hd⟩
  · rintro ⟨p, hp, hd⟩; exact ⟨p, hp, (key p hp).2 hd⟩

/-- **The tooth of the next gear**: a column in the class `n ≡ invSix q' (mod q')` is struck by
`q'` (its lower member is `≡ 0`). -/
theorem tooth_of_class {q' n : ℕ} (hq' : q'.Prime) (h5 : 5 ≤ q')
    (hmod : n % q' = invSix q' % q') : StrikesBy q' n := by
  left
  rcases Nat.eq_zero_or_pos n with h0 | h0
  · subst h0; simp
  · have h1 : n ≡ invSix q' [MOD q'] := hmod
    have h2 : 6 * n ≡ 6 * invSix q' [MOD q'] := Nat.ModEq.mul_left 6 h1
    have h3 : 6 * invSix q' ≡ 1 [MOD q'] := by
      unfold Nat.ModEq; rw [invSix_spec hq' h5, Nat.mod_eq_of_lt hq'.one_lt]
    have h4 : 1 ≡ 6 * n [MOD q'] := (h2.trans h3).symm
    exact (Nat.modEq_iff_dvd' (by omega)).1 h4

/-- **A single hole is always alignable** (the sandwich's lower half): a run of machine `q`
with a single hole becomes a fully struck run of the next machine `q'` at some translate by a
multiple of the period `P`, the hole landing on a tooth of `q'`: `G_1(q) ≤ F(q')`. -/
theorem align_single_hole {q q' a L h₀ P : ℕ} (hq' : q'.Prime) (h5 : 5 ≤ q') (hqq : q ≤ q')
    (hP : 0 < P) (hper : ∀ p, Gear q p → p ∣ P) (hcop : Nat.Coprime P q') (ha : 1 ≤ a)
    (hh : a ≤ h₀ ∧ h₀ < a + L) (hhole : ¬ Struck q h₀)
    (hrest : ∀ n, a ≤ n → n < a + L → n ≠ h₀ → Struck q n) :
    ∃ a', 1 ≤ a' ∧ ∀ n, a' ≤ n → n < a' + L → Struck q' n := by
  obtain ⟨u, -, hu⟩ := Nat.exists_mul_mod_eq_one_of_coprime hcop hq'.one_lt
  obtain ⟨d, hd⟩ : ∃ d, d = invSix q' + q' * h₀ - h₀ := ⟨_, rfl⟩
  refine ⟨a + u * d * P, by omega, ?_⟩
  intro n hn1 hn2
  obtain ⟨m, rfl⟩ : ∃ m, n = m + u * d * P := ⟨n - u * d * P, by omega⟩
  have hm1 : a ≤ m := by omega
  have hm2 : m < a + L := by omega
  by_cases hmh : m = h₀
  · rw [hmh]
    refine ⟨q', ⟨hq', h5, le_refl _⟩, ?_⟩
    have hPu : P * u ≡ 1 [MOD q'] := by
      unfold Nat.ModEq; rw [hu, Nat.mod_eq_of_lt hq'.one_lt]
    have h2 : h₀ + d * (P * u) ≡ h₀ + d * 1 [MOD q'] :=
      (Nat.ModEq.refl h₀).add (Nat.ModEq.mul_left d hPu)
    have hle : h₀ ≤ q' * h₀ := Nat.le_mul_of_pos_left h₀ hq'.pos
    have h3 : h₀ + d * 1 = invSix q' + q' * h₀ := by rw [mul_one, hd]; omega
    have h4 : invSix q' + q' * h₀ ≡ invSix q' [MOD q'] := by
      have := (Nat.ModEq.refl (invSix q')).add
        (Nat.modEq_zero_iff_dvd.2 (dvd_mul_right q' h₀))
      rwa [add_zero] at this
    have hmod : h₀ + u * d * P ≡ invSix q' [MOD q'] := by
      calc h₀ + u * d * P = h₀ + d * (P * u) := by ring
        _ ≡ h₀ + d * 1 [MOD q'] := h2
        _ = invSix q' + q' * h₀ := h3
        _ ≡ invSix q' [MOD q'] := h4
    exact tooth_of_class hq' h5 hmod
  · have hs : Struck q m := hrest m hm1 hm2 hmh
    have hs' : Struck q (m + u * d * P) := (struck_periodic (by omega) hper).2 hs
    obtain ⟨p, hp, hd'⟩ := hs'
    exact ⟨p, ⟨hp.1, hp.2.1, le_trans hp.2.2 hqq⟩, hd'⟩

/-- **The lower half in the kernel's max-gap vocabulary**: a single-hole run of machine `q`
refutes `MaxGapBelow q' L` for the next machine. -/
theorem not_maxGapBelow_of_single_hole {q q' a L h₀ P : ℕ} (hq' : q'.Prime) (h5 : 5 ≤ q')
    (hqq : q ≤ q') (hP : 0 < P) (hper : ∀ p, Gear q p → p ∣ P) (hcop : Nat.Coprime P q')
    (ha : 1 ≤ a) (hh : a ≤ h₀ ∧ h₀ < a + L) (hhole : ¬ Struck q h₀)
    (hrest : ∀ n, a ≤ n → n < a + L → n ≠ h₀ → Struck q n) : ¬ MaxGapBelow q' L := by
  intro hmg
  obtain ⟨a', ha', hcov⟩ := align_single_hole hq' h5 hqq hP hper hcop ha hh hhole hrest
  obtain ⟨n, hn1, hn2, hns⟩ := hmg a' ha'
  exact hns (hcov n hn1 hn2)

end TwinLadder
