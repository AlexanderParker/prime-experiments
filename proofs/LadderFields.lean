/-
LadderFields (2026-09-20): the field lemmas of the draft (E1, E2, E3', E4c, E4e).

Columns `n ≥ 1` hold `(6n - 1, 6n + 1)`; gear `p` STRIKES column `n` when `p ∣ 6n - 1` or
`p ∣ 6n + 1` (`StrikesBy`).  E1: one gear never strikes two adjacent columns (`no_adjacent`).
E4c: two columns struck by the same gear are congruent mod `p` or at least `(p - 1)/3` apart
(`strike_distance`, `strike_distance_ge`).  Counting: a residue class meets a run of `L` columns
at most `⌈L/p⌉` times (`class_count`), so a gear strikes at most `2⌈L/p⌉` columns of the run
(`gear_count`).  E3': in a run covered by machine `q'` (the next prime after `q`) the holes of
machine `q` number at most `2⌈L/q'⌉` (`holes_in_covered_run`).  E4e: in a run covered by
machine `q` the holes of the base machine `B` number at most `∑ 2⌈L/p⌉` over the primes
`B < p ≤ q` (`thin_band`).  E2: two gears leave a column free in any five consecutive
(`five_consecutive`), and in any four unless the gears are `{5, 7}` (`four_consecutive`).
-/
import LadderCovering
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Order.Interval.Finset.Nat

namespace TwinLadder

/-- Gear `p` strikes column `n`: `p` divides a member of the column. -/
def StrikesBy (p n : ℕ) : Prop := p ∣ 6 * n - 1 ∨ p ∣ 6 * n + 1

instance (p n : ℕ) : Decidable (StrikesBy p n) := inferInstanceAs (Decidable (_ ∨ _))

/-- **E1, no adjacent strikes**: a gear `p ≥ 5` never strikes columns `n` and `n + 1` (the
member differences are `4, 6, 6, 8`). -/
theorem no_adjacent {p n : ℕ} (hp : p.Prime) (h5 : 5 ≤ p) (hn : 1 ≤ n)
    (h1 : StrikesBy p n) (h2 : StrikesBy p (n + 1)) : False := by
  have hd : p ∣ 4 ∨ p ∣ 6 ∨ p ∣ 8 := by
    rcases h1 with h1 | h1 <;> rcases h2 with h2 | h2
    · have h := Nat.dvd_sub h2 h1
      rw [show 6 * (n + 1) - 1 - (6 * n - 1) = 6 by omega] at h
      exact Or.inr (Or.inl h)
    · have h := Nat.dvd_sub h2 h1
      rw [show 6 * (n + 1) + 1 - (6 * n - 1) = 8 by omega] at h
      exact Or.inr (Or.inr h)
    · have h := Nat.dvd_sub h2 h1
      rw [show 6 * (n + 1) - 1 - (6 * n + 1) = 4 by omega] at h
      exact Or.inl h
    · have h := Nat.dvd_sub h2 h1
      rw [show 6 * (n + 1) + 1 - (6 * n + 1) = 6 by omega] at h
      exact Or.inr (Or.inl h)
  have hle : p ≤ 8 := by
    rcases hd with h | h | h <;> exact le_trans (Nat.le_of_dvd (by norm_num) h) (by norm_num)
  interval_cases p <;> first | omega | exact absurd hp (by decide)

/-- `no_adjacent` with the second column given by an equation (for case splits). -/
theorem no_adjacent' {p n m : ℕ} (hp : p.Prime) (h5 : 5 ≤ p) (hn : 1 ≤ n) (hm : m = n + 1)
    (h1 : StrikesBy p n) (h2 : StrikesBy p m) : False := by
  subst hm; exact no_adjacent hp h5 hn h1 h2

/-- **E4c core, strike distances**: two columns `n < m` struck by the same gear `p` satisfy
`p ∣ m - n` (same sign) or `3(m - n) ≡ ±1 (mod p)` (opposite signs). -/
theorem strike_distance {p n m : ℕ} (hp : p.Prime) (h5 : 5 ≤ p) (hn : 1 ≤ n) (hnm : n < m)
    (h1 : StrikesBy p n) (h2 : StrikesBy p m) :
    p ∣ m - n ∨ 3 * (m - n) ≡ 1 [MOD p] ∨ 3 * (m - n) ≡ p - 1 [MOD p] := by
  have h6 := coprime_six hp h5
  have hp2 : ¬ p ∣ 2 := fun h => by have := Nat.le_of_dvd (by norm_num) h; omega
  rcases h1 with h1 | h1 <;> rcases h2 with h2 | h2
  · have h := Nat.dvd_sub h2 h1
    rw [show 6 * m - 1 - (6 * n - 1) = 6 * (m - n) by omega] at h
    exact Or.inl (h6.dvd_of_dvd_mul_left h)
  · have h := Nat.dvd_sub h2 h1
    rw [show 6 * m + 1 - (6 * n - 1) = 2 * (3 * (m - n) + 1) by omega] at h
    have h' : p ∣ 3 * (m - n) + 1 := by
      rcases (Nat.Prime.dvd_mul hp).1 h with h | h
      · exact absurd h hp2
      · exact h
    refine Or.inr (Or.inr ?_)
    have hz : 3 * (m - n) + 1 ≡ 0 [MOD p] := Nat.modEq_zero_iff_dvd.2 h'
    have hp0 : (p - 1) + 1 ≡ 0 [MOD p] := by
      rw [show p - 1 + 1 = p by omega]; exact Nat.modEq_zero_iff_dvd.2 dvd_rfl
    exact Nat.ModEq.add_right_cancel' 1 (hz.trans hp0.symm)
  · have h := Nat.dvd_sub h2 h1
    rw [show 6 * m - 1 - (6 * n + 1) = 2 * (3 * (m - n) - 1) by omega] at h
    have h' : p ∣ 3 * (m - n) - 1 := by
      rcases (Nat.Prime.dvd_mul hp).1 h with h | h
      · exact absurd h hp2
      · exact h
    refine Or.inr (Or.inl ?_)
    exact ((Nat.modEq_iff_dvd' (by omega)).2 h').symm
  · have h := Nat.dvd_sub h2 h1
    rw [show 6 * m + 1 - (6 * n + 1) = 6 * (m - n) by omega] at h
    exact Or.inl (h6.dvd_of_dvd_mul_left h)

/-- **E4c**: two columns struck by the same gear are congruent mod `p` or at least `(p - 1)/3`
apart: `3(m - n) + 1 ≥ p` when `p ∤ m - n`. -/
theorem strike_distance_ge {p n m : ℕ} (hp : p.Prime) (h5 : 5 ≤ p) (hn : 1 ≤ n) (hnm : n < m)
    (h1 : StrikesBy p n) (h2 : StrikesBy p m) (hnd : ¬ p ∣ m - n) : 3 * (m - n) + 1 ≥ p := by
  rcases strike_distance hp h5 hn hnm h1 h2 with h | h | h
  · exact absurd h hnd
  · have hmod : 3 * (m - n) % p = 1 := by
      unfold Nat.ModEq at h; rw [h, Nat.mod_eq_of_lt (by omega : 1 < p)]
    have hd := Nat.div_add_mod (3 * (m - n)) p
    rw [hmod] at hd
    generalize 3 * (m - n) / p = k at hd
    rcases Nat.eq_zero_or_pos k with h0 | h0
    · subst h0; omega
    · have := Nat.le_mul_of_pos_right p h0
      omega
  · have hmod : 3 * (m - n) % p = p - 1 := by
      unfold Nat.ModEq at h; rw [h, Nat.mod_eq_of_lt (by omega : p - 1 < p)]
    have := Nat.mod_le (3 * (m - n)) p
    omega

/-- Two columns of one residue class mod `p` (from `a ≤ n₁ < n₂`) land in different blocks
of `p` counted from `a`. -/
theorem class_sep {p a r n₁ n₂ : ℕ} (hp : 0 < p) (h1 : n₁ % p = r) (h2 : n₂ % p = r)
    (ha : a ≤ n₁) (hlt : n₁ < n₂) : (n₁ - a) / p < (n₂ - a) / p := by
  have hmod : n₁ ≡ n₂ [MOD p] := h1.trans h2.symm
  obtain ⟨k, hk⟩ := (Nat.modEq_iff_dvd' hlt.le).1 hmod
  have hk0 : 0 < k := by
    rcases Nat.eq_zero_or_pos k with h0 | h0
    · subst h0; omega
    · exact h0
  have hpk : p ≤ p * k := Nat.le_mul_of_pos_right p hk0
  have h3 : n₁ - a + p ≤ n₂ - a := by omega
  have h4 : (n₁ - a + p) / p ≤ (n₂ - a) / p := Nat.div_le_div_right h3
  rw [Nat.add_div_right _ hp] at h4
  omega

/-- **AP count**: a residue class mod `p` meets a run of `L` consecutive integers at most
`⌈L/p⌉ = (L + p - 1)/p` times. -/
theorem class_count (p a L r : ℕ) (hp : 0 < p) :
    ((Finset.Ico a (a + L)).filter (fun n => n % p = r)).card ≤ (L + p - 1) / p := by
  rw [← Finset.card_range ((L + p - 1) / p)]
  refine Finset.card_le_card_of_injOn (fun n => (n - a) / p) ?_ ?_
  · intro n hn
    simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_Ico] at hn
    simp only [Finset.mem_coe, Finset.mem_range]
    have h1 : n - a + p ≤ L + p - 1 := by omega
    have h2 : (n - a + p) / p ≤ (L + p - 1) / p := Nat.div_le_div_right h1
    rw [Nat.add_div_right _ hp] at h2
    omega
  · intro n₁ hn₁ n₂ hn₂ heq
    simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_Ico] at hn₁ hn₂
    have heq' : (n₁ - a) / p = (n₂ - a) / p := heq
    rcases lt_trichotomy n₁ n₂ with h | h | h
    · have := class_sep hp hn₁.2 hn₂.2 hn₁.1.1 h; omega
    · exact h
    · have := class_sep hp hn₂.2 hn₁.2 hn₂.1.1 h; omega

/-- **Gear count**: a gear `p ≥ 5` strikes at most `2⌈L/p⌉` columns of a run of `L` columns
from `a ≥ 1` (its struck columns lie in two residue classes mod `p`). -/
theorem gear_count {p : ℕ} (hp : p.Prime) (h5 : 5 ≤ p) (a L : ℕ) (ha : 1 ≤ a) :
    ((Finset.Ico a (a + L)).filter (fun n => StrikesBy p n)).card ≤ 2 * ((L + p - 1) / p) := by
  have hx := invSix_spec hp h5
  have hsub : (Finset.Ico a (a + L)).filter (fun n => StrikesBy p n) ⊆
      (Finset.Ico a (a + L)).filter (fun n => n % p = invSix p % p) ∪
      (Finset.Ico a (a + L)).filter (fun n => n % p = (p - 1) * invSix p % p) := by
    intro n hn
    rw [Finset.mem_filter] at hn
    rw [Finset.mem_union, Finset.mem_filter, Finset.mem_filter]
    have hn1 : 1 ≤ n := by have := (Finset.mem_Ico.1 hn.1).1; omega
    rcases hn.2 with h | h
    · exact Or.inl ⟨hn.1, mod_eq_of_dvd_sub hp h5 hn1 hx h⟩
    · exact Or.inr ⟨hn.1, mod_eq_of_dvd_add hp h5 hx h⟩
  have hc := Finset.card_le_card hsub
  have hu := Finset.card_union_le
    ((Finset.Ico a (a + L)).filter (fun n => n % p = invSix p % p))
    ((Finset.Ico a (a + L)).filter (fun n => n % p = (p - 1) * invSix p % p))
  have h1 := class_count p a L (invSix p % p) hp.pos
  have h2 := class_count p a L ((p - 1) * invSix p % p) hp.pos
  omega

/-- `gear_count` for every prime: the gears `2, 3` strike no column `n ≥ 1`. -/
theorem gear_count_prime {p : ℕ} (hp : p.Prime) (a L : ℕ) (ha : 1 ≤ a) :
    ((Finset.Ico a (a + L)).filter (fun n => StrikesBy p n)).card ≤ 2 * ((L + p - 1) / p) := by
  by_cases h5 : 5 ≤ p
  · exact gear_count hp h5 a L ha
  · have hempty : (Finset.Ico a (a + L)).filter (fun n => StrikesBy p n) = ∅ := by
      rw [Finset.filter_eq_empty_iff]
      intro n hn hs
      rw [Finset.mem_Ico] at hn
      have h2 := hp.two_le
      have h4 : p = 2 ∨ p = 3 ∨ p = 4 := by omega
      rcases h4 with rfl | rfl | rfl <;> rcases hs with hs | hs <;>
        omega
    rw [hempty, Finset.card_empty]; exact Nat.zero_le _

open Classical in
/-- **E3' upper half**: in a run (from `a ≥ 1`) covered by machine `q'`, the next prime after
`q`, the holes of machine `q` number at most `2⌈L/q'⌉` (each is struck by `q'` itself). -/
theorem holes_in_covered_run {q q' : ℕ} (hq' : q'.Prime) (h5 : 5 ≤ q') (hqq : q < q')
    (hnext : ∀ p, p.Prime → q < p → p ≤ q' → p = q') (a L : ℕ) (ha : 1 ≤ a)
    (hcov : ∀ n, a ≤ n → n < a + L → Struck q' n) :
    ((Finset.Ico a (a + L)).filter (fun n => ¬ Struck q n)).card ≤ 2 * ((L + q' - 1) / q') := by
  classical
  refine le_trans (Finset.card_le_card ?_) (gear_count hq' h5 a L ha)
  intro n hn
  rw [Finset.mem_filter] at hn ⊢
  refine ⟨hn.1, ?_⟩
  have hn' := Finset.mem_Ico.1 hn.1
  obtain ⟨p, hp, hd⟩ := hcov n hn'.1 hn'.2
  by_cases hpq : p ≤ q
  · exact absurd ⟨p, ⟨hp.1, hp.2.1, hpq⟩, hd⟩ hn.2
  · have he : p = q' := hnext p hp.1 (by omega) hp.2.2
    rw [he] at hd
    exact hd

open Classical in
/-- **E4e, the thin-band bound**: in a run (from `a ≥ 1`) covered by machine `q`, the holes of
the base machine `B` number at most `∑ 2⌈L/p⌉` over the primes `B < p ≤ q` (each hole is
struck by a gear of the top band). -/
theorem thin_band {B q : ℕ} (a L : ℕ) (ha : 1 ≤ a)
    (hcov : ∀ n, a ≤ n → n < a + L → Struck q n) :
    ((Finset.Ico a (a + L)).filter (fun n => ¬ Struck B n)).card ≤
      ∑ p ∈ (Finset.Ioc B q).filter Nat.Prime, 2 * ((L + p - 1) / p) := by
  classical
  have hsub : (Finset.Ico a (a + L)).filter (fun n => ¬ Struck B n) ⊆
      ((Finset.Ioc B q).filter Nat.Prime).biUnion
        (fun p => (Finset.Ico a (a + L)).filter (fun n => StrikesBy p n)) := by
    intro n hn
    rw [Finset.mem_filter] at hn
    have hn' := Finset.mem_Ico.1 hn.1
    rw [Finset.mem_biUnion]
    obtain ⟨p, hp, hd⟩ := hcov n hn'.1 hn'.2
    refine ⟨p, ?_, ?_⟩
    · rw [Finset.mem_filter, Finset.mem_Ioc]
      refine ⟨⟨?_, hp.2.2⟩, hp.1⟩
      by_contra hle
      push Not at hle
      exact hn.2 ⟨p, ⟨hp.1, hp.2.1, hle⟩, hd⟩
    · rw [Finset.mem_filter]
      exact ⟨hn.1, hd⟩
  calc ((Finset.Ico a (a + L)).filter (fun n => ¬ Struck B n)).card
      ≤ (((Finset.Ioc B q).filter Nat.Prime).biUnion
          (fun p => (Finset.Ico a (a + L)).filter (fun n => StrikesBy p n))).card :=
        Finset.card_le_card hsub
    _ ≤ ∑ p ∈ (Finset.Ioc B q).filter Nat.Prime,
          ((Finset.Ico a (a + L)).filter (fun n => StrikesBy p n)).card :=
        Finset.card_biUnion_le
    _ ≤ ∑ p ∈ (Finset.Ioc B q).filter Nat.Prime, 2 * ((L + p - 1) / p) := by
        apply Finset.sum_le_sum
        intro p hp
        rw [Finset.mem_filter] at hp
        exact gear_count_prime hp.2 a L ha

/-- `⌈L/g⌉ = 1` for `1 ≤ L ≤ g`. -/
theorem ceil_div_eq_one {L g : ℕ} (hL : 1 ≤ L) (hg : L ≤ g) : (L + g - 1) / g = 1 := by
  have hg0 : 0 < g := by omega
  have h1 : 1 ≤ (L + g - 1) / g := (Nat.le_div_iff_mul_le hg0).2 (by omega)
  have h2 : (L + g - 1) / g < 2 := (Nat.div_lt_iff_lt_mul hg0).2 (by omega)
  omega

/-- **E2a, five columns**: two gears `g, h ≥ 5` leave a column free in any five consecutive
columns from `n ≥ 1` (each strikes at most two of them). -/
theorem five_consecutive {g h n : ℕ} (hg : g.Prime) (h5g : 5 ≤ g) (hh : h.Prime) (h5h : 5 ≤ h)
    (hn : 1 ≤ n) : ∃ m, n ≤ m ∧ m < n + 5 ∧ ¬ StrikesBy g m ∧ ¬ StrikesBy h m := by
  by_contra hcon
  have key : ∀ m, n ≤ m → m < n + 5 → StrikesBy g m ∨ StrikesBy h m := by
    intro m h1 h2
    by_contra hc
    exact hcon ⟨m, h1, h2, fun hs => hc (Or.inl hs), fun hs => hc (Or.inr hs)⟩
  have hsub : Finset.Ico n (n + 5) ⊆
      (Finset.Ico n (n + 5)).filter (fun m => StrikesBy g m) ∪
      (Finset.Ico n (n + 5)).filter (fun m => StrikesBy h m) := by
    intro m hm
    rw [Finset.mem_union, Finset.mem_filter, Finset.mem_filter]
    have hm' := Finset.mem_Ico.1 hm
    rcases key m hm'.1 hm'.2 with hs | hs
    · exact Or.inl ⟨hm, hs⟩
    · exact Or.inr ⟨hm, hs⟩
  have hc := Finset.card_le_card hsub
  have hu := Finset.card_union_le ((Finset.Ico n (n + 5)).filter (fun m => StrikesBy g m))
    ((Finset.Ico n (n + 5)).filter (fun m => StrikesBy h m))
  have h1 := gear_count hg h5g n 5 hn
  have h2 := gear_count hh h5h n 5 hn
  rw [ceil_div_eq_one (show 1 ≤ 5 by norm_num) h5g] at h1
  rw [ceil_div_eq_one (show 1 ≤ 5 by norm_num) h5h] at h2
  rw [Nat.card_Ico] at hc
  omega

/-- A gear striking columns `n` and `n + 2` is `5` or `7` (member differences `10, 12, 14`). -/
theorem strike_two {p n m : ℕ} (hp : p.Prime) (h5 : 5 ≤ p) (hn : 1 ≤ n) (hm : m = n + 2)
    (h1 : StrikesBy p n) (h2 : StrikesBy p m) : p = 5 ∨ p = 7 := by
  subst hm
  have hd : p ∣ 10 ∨ p ∣ 12 ∨ p ∣ 14 := by
    rcases h1 with h1 | h1 <;> rcases h2 with h2 | h2
    · have h := Nat.dvd_sub h2 h1
      rw [show 6 * (n + 2) - 1 - (6 * n - 1) = 12 by omega] at h
      exact Or.inr (Or.inl h)
    · have h := Nat.dvd_sub h2 h1
      rw [show 6 * (n + 2) + 1 - (6 * n - 1) = 14 by omega] at h
      exact Or.inr (Or.inr h)
    · have h := Nat.dvd_sub h2 h1
      rw [show 6 * (n + 2) - 1 - (6 * n + 1) = 10 by omega] at h
      exact Or.inl h
    · have h := Nat.dvd_sub h2 h1
      rw [show 6 * (n + 2) + 1 - (6 * n + 1) = 12 by omega] at h
      exact Or.inr (Or.inl h)
  have hle : p ≤ 14 := by
    rcases hd with h | h | h <;> exact le_trans (Nat.le_of_dvd (by norm_num) h) (by norm_num)
  interval_cases p <;> first | omega | exact absurd hp (by decide)

/-- The alternating pattern `g, h, g, h` on four consecutive columns forces `{g, h} = {5, 7}`. -/
theorem alternating_case {g h n : ℕ} (hg : g.Prime) (h5g : 5 ≤ g) (hh : h.Prime) (h5h : 5 ≤ h)
    (hn : 1 ≤ n) (hne : ¬ ((g = 5 ∧ h = 7) ∨ (g = 7 ∧ h = 5)))
    (h0 : StrikesBy g n) (h1 : StrikesBy h (n + 1)) (h2 : StrikesBy g (n + 2))
    (h3 : StrikesBy h (n + 3)) : False := by
  have hg57 := strike_two hg h5g hn rfl h0 h2
  have hh57 := strike_two hh h5h (by omega : 1 ≤ n + 1) (by omega) h1 h3
  have hgh : g ≠ h := fun e => no_adjacent' hg h5g hn rfl h0 (e ▸ h1)
  omega

/-- **E2b, four columns**: two gears `g, h ≥ 5`, not `{5, 7}`, leave a column free in any four
consecutive columns from `n ≥ 1` (no gear strikes adjacent columns, so a full cover alternates,
and a gear striking at distance two is `5` or `7`). -/
theorem four_consecutive {g h n : ℕ} (hg : g.Prime) (h5g : 5 ≤ g) (hh : h.Prime) (h5h : 5 ≤ h)
    (hn : 1 ≤ n) (hne : ¬ ((g = 5 ∧ h = 7) ∨ (g = 7 ∧ h = 5))) :
    ∃ m, n ≤ m ∧ m < n + 4 ∧ ¬ StrikesBy g m ∧ ¬ StrikesBy h m := by
  by_contra hcon
  have key : ∀ m, n ≤ m → m < n + 4 → StrikesBy g m ∨ StrikesBy h m := by
    intro m h1 h2
    by_contra hc
    exact hcon ⟨m, h1, h2, fun hs => hc (Or.inl hs), fun hs => hc (Or.inr hs)⟩
  have hne' : ¬ ((h = 5 ∧ g = 7) ∨ (h = 7 ∧ g = 5)) := by omega
  have k0 := key n (by omega) (by omega)
  have k1 := key (n + 1) (by omega) (by omega)
  have k2 := key (n + 2) (by omega) (by omega)
  have k3 := key (n + 3) (by omega) (by omega)
  rcases k0 with h0 | h0 <;> rcases k1 with h1 | h1 <;> rcases k2 with h2 | h2 <;>
    rcases k3 with h3 | h3 <;>
    first
    | exact no_adjacent' hg h5g (by omega) (by omega) h0 h1
    | exact no_adjacent' hh h5h (by omega) (by omega) h0 h1
    | exact no_adjacent' hg h5g (by omega) (by omega) h1 h2
    | exact no_adjacent' hh h5h (by omega) (by omega) h1 h2
    | exact no_adjacent' hg h5g (by omega) (by omega) h2 h3
    | exact no_adjacent' hh h5h (by omega) (by omega) h2 h3
    | exact alternating_case hg h5g hh h5h hn hne h0 h1 h2 h3
    | exact alternating_case hh h5h hg h5g hn hne' h0 h1 h2 h3

end TwinLadder
