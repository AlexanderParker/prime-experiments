/-
THE FIELDS, PART C: THE GEAR FIELDS - EACH GEAR'S DILATE OF THE LOWER SURVIVORS FROM ITS SQUARE
ON.  THE SQUARE IS THE FIRST MEMBER (G0), THE LEAST FACTOR IS THE GEAR AND THE GEAR FIELDS
PARTITION THE COMPOSITES OF `S` BY LEAST FACTOR (G1, G2), THE SURVIVORS AND THE GEAR FIELD ARE
PERIODIC (G3), THE SURVIVORS ARE MIRROR-SYMMETRIC IN THE PERIOD (G4), THE CRT COUNT OF THE
SURVIVORS IN ONE PERIOD (G5), AND THE GAPS OF A GEAR FIELD ARE `g` TIMES THE GAPS OF THE
SURVIVORS (G6)  (Formalist).

Builds on `Fields` and `FieldsB` (the objects `InS`, `field j`, and the fold's lemmas).

THE OBJECTS.  `Survivor g m` says `m ∈ S` and no prime `h < g` divides `m` (the primes `2, 3`
are excluded by `InS`, so only the gears `5 ≤ h < g` bite).  `gearField g` is `g` times the
survivors `m ≥ g`: the gear's dilate of the lower survivors, from its square `g · g` on.

WHAT THE PROOFS ACTUALLY NEED (the findings of the kernel).
* G0 needs `g` prime and `g ≥ 5` (for `g ∈ S`).
* G1 forward (`minFac = g`) needs only `g` prime, not `g ≥ 5`; the converse needs `g` prime,
  `n ∈ S`, `n.minFac = g`, `n ≠ g`, and DERIVES `g ≥ 5` from `g ∣ n ∈ S`.  So G2 (disjointness)
  needs only two distinct primes.
* G3 (periodicity) needs only that `L` CONTAINS every prime of `[5, g)`; that the members of `L`
  are such primes is not used.  The shift on the gear field needs `0 < g` and `g^2 ≤ n` only in
  the reverse direction (to bring `m` back to `m ≥ g`).
* G4 (the mirror) likewise needs only that `L` contains the primes of `[5, g)`, and `m < 6P`;
  `1 ≤ m` is not needed (`Survivor g 0` is false).
* G5 (the count) is the totient of `6P`: `Survivor g m ↔ Coprime (6P) m` needs BOTH halves of
  the hypothesis on `L` (contains the primes of `[5, g)`; consists of such primes), then
  `φ(6P) = φ(6) · ∏ (h - 1)` by multiplicativity over the distinct primes.  The count of `[1, 6P]`
  equals the count of `[0, 6P)` because neither `0` nor `6P` is coprime to `6P`.
* G6 needs NOTHING of `g`: `g · m₁ < g · m < g · m₂` forces `m₁ < m < m₂`.

Zero sorries; no `native_decide`, no `decide`, no `Lean.ofReduceBool`.
-/
import FieldsB
import Mathlib.Data.Nat.Totient
import Mathlib.Data.Nat.GCD.BigOperators
import Mathlib.Algebra.BigOperators.Associated
import Mathlib.Tactic.NormNum.Prime

namespace Fields

open OneStepE CoreLeftover SquareColumn

/-! ## The objects -/

/-- `m` survives the gears below `g`: `m ∈ S` and no prime `h < g` divides `m`. -/
def Survivor (g : ℕ) (m : ℕ) : Prop := InS m ∧ ∀ h, h.Prime → h < g → ¬ h ∣ m

/-- The gear field of `g`: `g` times the survivors `m ≥ g`, i.e. the dilate from `g^2` on. -/
def gearField (g : ℕ) : Set ℕ := {n | ∃ m, n = g * m ∧ g ≤ m ∧ Survivor g m}

theorem mem_gearField {g n : ℕ} : n ∈ gearField g ↔ ∃ m, n = g * m ∧ g ≤ m ∧ Survivor g m :=
  Iff.rfl

/-! ## G0: the square is the first member -/

/-- **G0.**  For a prime `g ≥ 5`, `g^2 ∈ gearField g`: `g` itself is a survivor. -/
theorem sq_mem_gearField {g : ℕ} (hg : g.Prime) (h5 : 5 ≤ g) : g ^ 2 ∈ gearField g :=
  ⟨g, by ring, le_rfl, inS_of_prime hg h5, fun h hh hlt hd => by
    have := (Nat.prime_dvd_prime_iff_eq hh hg).mp hd
    omega⟩

/-! ## G1, G2: the least factor is the gear; the gear fields are disjoint -/

/-- **G1, forward.**  Every member of `gearField g` (`g` prime) has least prime factor `g`. -/
theorem gearField_minFac {g n : ℕ} (hg : g.Prime) (hn : n ∈ gearField g) : n.minFac = g := by
  obtain ⟨m, rfl, hgm, -, hsurv⟩ := hn
  have hg2 := hg.two_le
  have hn1 : g * m ≠ 1 := fun h => by
    have := Nat.eq_one_of_mul_eq_one_right h
    omega
  have hle : (g * m).minFac ≤ g := Nat.minFac_le_of_dvd hg2 (dvd_mul_right g m)
  have hmf := Nat.minFac_prime hn1
  rcases (Nat.Prime.dvd_mul hmf).mp (Nat.minFac_dvd (g * m)) with h | h
  · exact (Nat.prime_dvd_prime_iff_eq hmf hg).mp h
  · rcases Nat.lt_or_ge (g * m).minFac g with hlt | hge
    · exact absurd h (hsurv _ hmf hlt)
    · omega

/-- **G1, converse.**  A survivor `n ≠ g` with least prime factor `g` lies in `gearField g`. -/
theorem mem_gearField_of_minFac {g n : ℕ} (hg : g.Prime) (hn : InS n) (hmin : n.minFac = g)
    (hne : n ≠ g) : n ∈ gearField g := by
  have hn0 : n ≠ 0 := ne_zero_of_inS hn
  have hg2 := hg.two_le
  have hgn : g ∣ n := hmin ▸ Nat.minFac_dvd n
  have h5 : 5 ≤ g := five_le_of_dvd hg (inS_iff.mp hn).1 (inS_iff.mp hn).2 hgn
  obtain ⟨m, rfl⟩ := hgn
  have hm0 : m ≠ 0 := by rintro rfl; simp at hn0
  have hm1 : m ≠ 1 := by rintro rfl; simp at hne
  refine ⟨m, rfl, ?_, (inS_mul_iff (inS_of_prime hg h5)).mp hn, fun h hh hlt hd => ?_⟩
  · have hmf := Nat.minFac_prime hm1
    have h1 : (g * m).minFac ≤ m.minFac :=
      Nat.minFac_le_of_dvd hmf.two_le (Dvd.dvd.mul_left (Nat.minFac_dvd m) g)
    rw [hmin] at h1
    exact le_trans h1 (Nat.minFac_le (Nat.pos_of_ne_zero hm0))
  · have := Nat.minFac_le_of_dvd hh.two_le (Dvd.dvd.mul_left hd g)
    omega

/-- **G2.**  The gear fields of distinct primes are disjoint. -/
theorem gearField_disjoint {g g' : ℕ} (hg : g.Prime) (hg' : g'.Prime) (hne : g ≠ g') :
    gearField g ∩ gearField g' = ∅ := by
  ext n
  simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false, not_and]
  intro h1 h2
  exact hne ((gearField_minFac hg h1).symm.trans (gearField_minFac hg' h2))

/-! ## G3: periodicity -/

/-- Transfer of survival: `S`-membership carries over and every gear dividing `m'` divides `m`. -/
theorem survivor_of_survivor {g m m' : ℕ} (hS : InS m → InS m')
    (h : ∀ p, p.Prime → 5 ≤ p → p < g → p ∣ m' → p ∣ m) (hm : Survivor g m) : Survivor g m' := by
  obtain ⟨hmS, hs⟩ := hm
  have hm'S := hS hmS
  refine ⟨hm'S, fun p hp hpg hd => ?_⟩
  by_cases h5 : 5 ≤ p
  · exact hs p hp hpg (h p hp h5 hpg hd)
  · obtain ⟨h2, h3⟩ := inS_iff.mp hm'S
    have h2p := hp.two_le
    have : p = 2 ∨ p = 3 ∨ p = 4 := by omega
    rcases this with rfl | rfl | rfl
    · exact h2 hd
    · exact h3 hd
    · norm_num at hp

/-- Survival is a congruence: same `S`-class and the same gears dividing. -/
theorem survivor_congr {g m m' : ℕ} (hS : InS m ↔ InS m')
    (h : ∀ p, p.Prime → 5 ≤ p → p < g → (p ∣ m ↔ p ∣ m')) : Survivor g m ↔ Survivor g m' :=
  ⟨survivor_of_survivor hS.mp (fun p hp h5 hpg => (h p hp h5 hpg).mpr),
    survivor_of_survivor hS.mpr (fun p hp h5 hpg => (h p hp h5 hpg).mp)⟩

/-- Every prime of `[5, g)` divides `6 P` when `L` contains the primes of `[5, g)`. -/
theorem dvd_six_prod_of_lt {g p : ℕ} {L : Finset ℕ} (hL : ∀ h, h.Prime → 5 ≤ h → h < g → h ∈ L)
    (hp : p.Prime) (h5 : 5 ≤ p) (hpg : p < g) : p ∣ 6 * ∏ h ∈ L, h :=
  Dvd.dvd.mul_left (Finset.dvd_prod_of_mem (fun h => h) (hL p hp h5 hpg)) 6

/-- **G3.**  Survival below `g` is periodic with period `6 P`, `P` the product of the gears. -/
theorem survivor_periodic {g m : ℕ} {L : Finset ℕ} (hL : ∀ h, h.Prime → 5 ≤ h → h < g → h ∈ L) :
    Survivor g m ↔ Survivor g (m + 6 * ∏ h ∈ L, h) := by
  apply survivor_congr
  · unfold InS; omega
  · intro p hp h5 hpg
    exact (Nat.dvd_add_left (dvd_six_prod_of_lt hL hp h5 hpg)).symm

/-- **G3 on the gear field.**  Above `g^2` the gear field is periodic with period `6 g P`. -/
theorem gearField_periodic {g n : ℕ} {L : Finset ℕ} (hg : 0 < g)
    (hL : ∀ h, h.Prime → 5 ≤ h → h < g → h ∈ L) (hn : g ^ 2 ≤ n) :
    n ∈ gearField g ↔ n + 6 * g * ∏ h ∈ L, h ∈ gearField g := by
  constructor
  · rintro ⟨m, rfl, hgm, hs⟩
    exact ⟨m + 6 * ∏ h ∈ L, h, by ring, by omega, (survivor_periodic hL).mp hs⟩
  · rintro ⟨m', he, -, hs⟩
    have e : g * (6 * ∏ h ∈ L, h) = 6 * g * ∏ h ∈ L, h := by ring
    have hle : 6 * ∏ h ∈ L, h ≤ m' := by
      have : g * (6 * ∏ h ∈ L, h) ≤ g * m' := by
        rw [← he, e]; exact Nat.le_add_left _ _
      exact Nat.le_of_mul_le_mul_left this hg
    obtain ⟨d, rfl⟩ : ∃ d, m' = d + 6 * ∏ h ∈ L, h := ⟨m' - 6 * ∏ h ∈ L, h, by omega⟩
    have e' : g * (d + 6 * ∏ h ∈ L, h) = g * d + 6 * g * ∏ h ∈ L, h := by ring
    rw [e'] at he
    have hnd : n = g * d := Nat.add_right_cancel he
    refine ⟨d, hnd, ?_, (survivor_periodic hL).mpr hs⟩
    rw [hnd, sq] at hn
    exact Nat.le_of_mul_le_mul_left hn hg

/-! ## G4: the mirror -/

/-- **G4.**  The survivors of one period are symmetric: `m ↦ 6P - m` preserves survival. -/
theorem survivor_neg {g m : ℕ} {L : Finset ℕ} (hL : ∀ h, h.Prime → 5 ≤ h → h < g → h ∈ L)
    (hlt : m < 6 * ∏ h ∈ L, h) (hs : Survivor g m) : Survivor g (6 * ∏ h ∈ L, h - m) := by
  refine survivor_of_survivor (fun hm => by unfold InS at hm ⊢; omega)
    (fun p hp h5 hpg hd => ?_) hs
  have := Nat.dvd_sub (dvd_six_prod_of_lt hL hp h5 hpg) hd
  rwa [Nat.sub_sub_self hlt.le] at this

/-! ## G5: the CRT count of the survivors in one period -/

/-- A prime divides `6 P` iff it is `2`, `3`, or a member of `L` (`L` a set of primes). -/
theorem prime_dvd_six_prod_iff {L : Finset ℕ} {k : ℕ} (hL' : ∀ h ∈ L, h.Prime) (hk : k.Prime) :
    k ∣ 6 * ∏ h ∈ L, h ↔ k = 2 ∨ k = 3 ∨ k ∈ L := by
  have h6 : k ∣ 6 ↔ k = 2 ∨ k = 3 := by
    rw [show (6 : ℕ) = 2 * 3 by norm_num, hk.dvd_mul,
      Nat.prime_dvd_prime_iff_eq hk Nat.prime_two, Nat.prime_dvd_prime_iff_eq hk Nat.prime_three]
  rw [hk.dvd_mul, h6, Prime.dvd_finsetProd_iff hk.prime]
  constructor
  · rintro ((h | h) | ⟨a, ha, hka⟩)
    · exact Or.inl h
    · exact Or.inr (Or.inl h)
    · rw [Nat.prime_dvd_prime_iff_eq hk (hL' a ha)] at hka
      exact Or.inr (Or.inr (hka ▸ ha))
  · rintro (rfl | rfl | h)
    · exact Or.inl (Or.inl rfl)
    · exact Or.inl (Or.inr rfl)
    · exact Or.inr ⟨k, h, dvd_rfl⟩

/-- Survival below `g` is coprimality to `6 P` when `L` is exactly the primes of `[5, g)`. -/
theorem survivor_iff_coprime {g m : ℕ} {L : Finset ℕ}
    (hL : ∀ h, h.Prime → 5 ≤ h → h < g → h ∈ L) (hL' : ∀ h ∈ L, h.Prime ∧ 5 ≤ h ∧ h < g) :
    Survivor g m ↔ Nat.Coprime (6 * ∏ h ∈ L, h) m := by
  have hLp : ∀ h ∈ L, h.Prime := fun h hh => (hL' h hh).1
  constructor
  · rintro ⟨hm, hs⟩
    apply Nat.coprime_of_dvd
    intro k hk hk6
    rcases (prime_dvd_six_prod_iff hLp hk).mp hk6 with rfl | rfl | hkL
    · exact (inS_iff.mp hm).1
    · exact (inS_iff.mp hm).2
    · exact hs k hk (hL' k hkL).2.2
  · intro hcop
    have key : ∀ k, k.Prime → k ∣ 6 * ∏ h ∈ L, h → ¬ k ∣ m := fun k hk hk6 hkm =>
      hk.one_lt.ne' (Nat.Coprime.eq_one_of_dvd (Nat.Coprime.coprime_dvd_left hk6 hcop) hkm)
    refine ⟨?_, fun h hh hlt hd => ?_⟩
    · rw [inS_iff]
      exact ⟨key 2 Nat.prime_two ((prime_dvd_six_prod_iff hLp Nat.prime_two).mpr (Or.inl rfl)),
        key 3 Nat.prime_three
          ((prime_dvd_six_prod_iff hLp Nat.prime_three).mpr (Or.inr (Or.inl rfl)))⟩
    · refine key h hh ?_ hd
      rw [prime_dvd_six_prod_iff hLp hh]
      by_cases h5 : 5 ≤ h
      · exact Or.inr (Or.inr (hL h hh h5 hlt))
      · have := hh.two_le
        have : h = 2 ∨ h = 3 ∨ h = 4 := by omega
        rcases this with rfl | rfl | rfl
        · exact Or.inl rfl
        · exact Or.inr (Or.inl rfl)
        · norm_num at hh

/-- The totient of a product of distinct primes is the product of `h - 1`. -/
theorem totient_prod_primes {L : Finset ℕ} (hL' : ∀ h ∈ L, h.Prime) :
    Nat.totient (∏ h ∈ L, h) = ∏ h ∈ L, (h - 1) := by
  induction L using Finset.induction_on with
  | empty => simp
  | insert a s ha ih =>
    have hpa : a.Prime := hL' a (Finset.mem_insert_self a s)
    have hs : ∀ h ∈ s, h.Prime := fun h hh => hL' h (Finset.mem_insert_of_mem hh)
    have hcop : Nat.Coprime a (∏ h ∈ s, h) :=
      Nat.coprime_prod_right_iff.mpr fun h hh =>
        (Nat.coprime_primes hpa (hs h hh)).mpr fun e => ha (e ▸ hh)
    rw [Finset.prod_insert ha, Finset.prod_insert ha, Nat.totient_mul hcop,
      Nat.totient_prime hpa, ih hs]

/-- A prime `≥ 5` does not divide `6`. -/
theorem not_dvd_six_of_five_le {h : ℕ} (hp : h.Prime) (h5 : 5 ≤ h) : ¬ h ∣ 6 := by
  intro hd
  rcases hp.dvd_mul.mp (show h ∣ 2 * 3 from hd) with h2 | h3
  · have := Nat.le_of_dvd (by norm_num) h2; omega
  · have := Nat.le_of_dvd (by norm_num) h3; omega

/-- `φ(6 P) = 2 ∏ (h - 1)` for `L` a set of primes `≥ 5`. -/
theorem totient_six_prod {L : Finset ℕ} (hL' : ∀ h ∈ L, h.Prime ∧ 5 ≤ h) :
    Nat.totient (6 * ∏ h ∈ L, h) = 2 * ∏ h ∈ L, (h - 1) := by
  have hcop : Nat.Coprime 6 (∏ h ∈ L, h) :=
    Nat.coprime_prod_right_iff.mpr fun h hh =>
      ((Nat.Prime.coprime_iff_not_dvd (hL' h hh).1).mpr
        (not_dvd_six_of_five_le (hL' h hh).1 (hL' h hh).2)).symm
  have h6 : Nat.totient 6 = 2 := by
    rw [show (6 : ℕ) = 2 * 3 by norm_num,
      Nat.totient_mul ((Nat.coprime_primes Nat.prime_two Nat.prime_three).mpr (by norm_num)),
      Nat.totient_prime Nat.prime_two, Nat.totient_prime Nat.prime_three]
  rw [Nat.totient_mul hcop, totient_prod_primes (fun h hh => (hL' h hh).1), h6]

open Classical in
/-- **G5.**  The survivors of `[1, 6P]` number `2 ∏ (h - 1)`: the CRT count. -/
theorem card_survivors_period {g : ℕ} {L : Finset ℕ}
    (hL : ∀ h, h.Prime → 5 ≤ h → h < g → h ∈ L) (hL' : ∀ h ∈ L, h.Prime ∧ 5 ≤ h ∧ h < g) :
    ((Finset.Icc 1 (6 * ∏ h ∈ L, h)).filter (Survivor g)).card = 2 * ∏ h ∈ L, (h - 1) := by
  rw [← totient_six_prod (fun h hh => ⟨(hL' h hh).1, (hL' h hh).2.1⟩), Nat.totient_eq_card_coprime]
  congr 1
  ext m
  simp only [Finset.mem_filter, Finset.mem_Icc, Finset.mem_range]
  rw [survivor_iff_coprime hL hL']
  constructor
  · rintro ⟨⟨-, h2⟩, hc⟩
    refine ⟨?_, hc⟩
    rcases Nat.lt_or_ge m (6 * ∏ h ∈ L, h) with hlt | hge
    · exact hlt
    · exfalso
      rw [le_antisymm h2 hge, Nat.coprime_self] at hc
      omega
  · rintro ⟨hlt, hc⟩
    refine ⟨⟨?_, hlt.le⟩, hc⟩
    rcases Nat.eq_zero_or_pos m with rfl | hpos
    · rw [Nat.coprime_zero_right] at hc; omega
    · exact hpos

/-! ## G6: the gaps of a gear field are `g` times the gaps of the survivors -/

/-- **G6.**  Consecutive survivors `m₁ < m₂` (with `g ≤ m₁`) dilate to consecutive members. -/
theorem gearField_gap_scaling {g m₁ m₂ : ℕ} (hg : g ≤ m₁) (hlt : m₁ < m₂)
    (h₁ : Survivor g m₁) (h₂ : Survivor g m₂) (hgap : ∀ m, m₁ < m → m < m₂ → ¬ Survivor g m) :
    g * m₁ ∈ gearField g ∧ g * m₂ ∈ gearField g ∧
      (∀ n ∈ gearField g, g * m₁ < n → n < g * m₂ → False) ∧
      g * m₂ - g * m₁ = g * (m₂ - m₁) := by
  refine ⟨⟨m₁, rfl, hg, h₁⟩, ⟨m₂, rfl, by omega, h₂⟩, ?_, (Nat.mul_sub g m₂ m₁).symm⟩
  rintro n ⟨m, rfl, -, hs⟩ hlo hhi
  exact hgap m (Nat.lt_of_mul_lt_mul_left hlo) (Nat.lt_of_mul_lt_mul_left hhi) hs

end Fields
