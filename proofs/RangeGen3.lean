import Mathlib.NumberTheory.Primorial
import Mathlib.Data.ZMod.Basic
import Mathlib.Tactic
import RangeMirror
import RangeActPair
import RangeWindowForm

/-!
# Mirror pairs under acting, for every machine size

Fix a machine size `q ≥ 5` (every statement below holds for all such `q`; the records state
`q ≥ 7`). Write `Q = q#`, `M = Q / 30` (`mirrorM`, odd), `a = Q / 2 = 15M` (`mirrorA`, odd) and
`ρ = a + 1` (`mirrorRho`). For an odd offset `d < M` (equivalently odd `d ∈ [1, M - 2]`) the mirror
pair is `s = (M - d)/2` (`mirrorLo`) and `s' = (M + d)/2` (`mirrorHi`), with legs

  `L1 = 30s - 1 = a - 15d - 1`, `L2 = 30s + 1 = a - 15d + 1`,
  `H1 = 30s' - 1 = a + 15d - 1`, `H2 = 30s' + 1 = a + 15d + 1`.

A gear `g` *acts* at copy `j` when `g² ≤ 30j + 1` (`Acts`, from `RangeActPair`); copy `j`
*survives* `q` when no prime `7 ≤ p ≤ q` strikes it (`Survives`); it is *deleted* when some prime
`g > q` strikes it and acts at it (`Deleted`). `t_g = 15⁻¹` in `ZMod g`.

This file proves, for all `q ≥ 5`, all odd `d < M` and all gears `g` as quantified:

* (i) Class form. `strikes_lo_iff_class` / `strikes_hi_iff_class`: for every prime `g ≥ 7`
  (not only `g > q`), `g` strikes `s` iff `d ≡ M ∓ t_g`, and strikes `s'` iff `d ≡ -M ± t_g`
  (mod `g`). `strikes_lo_iff_of_dvd` / `strikes_hi_iff_of_dvd`: for every divisor `g` of `M`,
  `g` strikes `s` (resp. `s'`) iff `g ∣ 225d² - 1`. `survives_lo_iff`, `survives_hi_iff`,
  `survives_pair`: both members survive or neither does, and they survive iff
  `gcd(225d² - 1, M) = 1`.
* (ii) Acting windows. `acting_windows`: `30s + 1 = ρ - 15d`, `30s' + 1 = ρ + 15d`.
  `acts_lo_iff`: `g` acts on `s` iff `15d ≤ ρ - g²`; `acts_hi_iff`: on `s'` iff `g² - ρ ≤ 15d`.
  `joint_acting_sq`: odd `g` acting on `s` and odd `h` acting on `s'` give `g² + h² < Q + 2`
  (`g² + h² ≡ 2`, `Q + 2 ≡ 0 (mod 4)`), and `joint_acting_mul`: `g h < ρ`.
* (iii) Shared strikers. `same_minus_iff` / `same_plus_iff` (any `g` coprime to `30`):
  `g ∣ L1 ∧ g ∣ H1 ↔ g ∣ Q - 2 ∧ g ∣ d` and `g ∣ L2 ∧ g ∣ H2 ↔ g ∣ Q + 2 ∧ g ∣ d`.
  `strikes_both_iff`: a prime `g > q` strikes both members iff `g ∣ (Q - 2)(Q + 2)` and `g ∣ d`;
  `strikes_both_leg_type` and `no_cross_strike`: the struck legs are `L1, H1` when `g ∣ Q - 2`
  and `L2, H2` when `g ∣ Q + 2`, never a cross pair. `acts_both_iff`: `g` acts on both iff
  `15d ≤ ρ - g²`; `deletes_both_iff`, `deletes_both_mem_uMinus`, `single_gear_deletion`: a prime
  `g > q` deletes both members iff `g ∣ (Q - 2)(Q + 2)`, `g ∣ d`, `15d ≤ ρ - g²`, and then
  `g ∈ U⁻`, `g ∣ Q ∓ 2`, `g² ≤ 30s + 1`.
* (iv) Sandwich. `sandwich`: `G(s) ⊆ U⁻ ⊆ G(s')`, with `G(j) = ActSet q j` the primes `g > q`
  acting at `j` and `U⁻ = uMinusSet q` the primes `g > q` with `g² ≤ ρ`. `core_eq`: when every
  prime of `U⁻` has `g² + 15d ≤ ρ` and every prime `g > q` with `g² > ρ` has `g² > ρ + 15d`,
  `G(s) = G(s') = U⁻`. `core_eq_of_le_coreR`: the same for `d ≤ R = coreR q`, where
  `R = min((ρ - g₋²)/15, largest d with 15d < g₊² - ρ)`, `g₋ = coreLow q` the largest prime
  `≤ √ρ` and `g₊ = coreHigh q` the least prime `> √ρ`.
* (v) Leg classes. `leg_classes`: modulo any `n`, `n ∣ L1, L2, H1, H2` iff `15d ≡ a - 1, a + 1,
  1 - a, -1 - a`. `legF_cast`, `four_leg_roots`: for a prime `g ≥ 7`, `g` divides
  `F(d) = L1·L2·H1·H2` iff `d` is one of `t(a - 1), t(a + 1), t(1 - a), t(-1 - a)`.
  `root_classes_nodup_iff`: these four classes are distinct iff `g ∤ (Q - 2)·Q·(Q + 2)`;
  `root_classes_nodup_iff_above`: for `g > q`, iff `g ∤ (Q - 2)(Q + 2)`.
  `three_classes`: otherwise (`g ∣ (Q - 2)(Q + 2)`) the classes are `{0, 2t, -2t}` (distinct),
  and `legF_of_sq_eq_one` shows `F ≡ 225 d² (15d - 2)(15d + 2)`, so `0` is a double root.
* (vi) Empty band. `band_empty_iff`: `[1, w]` with `15w ≤ ρ - g²` holds no multiple of `g` iff
  `ρ - g² < 15g` (also for odd multiples, `band_empty_odd_iff`; `band_w_iff` reads the band as
  `[1, w_g]`, `w_g = (ρ - g²)/15`). `band_empty_minus` /
  `band_empty_plus`: for odd `g ≥ 3` in `U⁻` with `g ∣ Q ∓ 2`, the band is empty iff
  `a ∓ 1 = g(g + e)` with `e` odd, `1 ≤ e ≤ 13`. `band_square_minus` / `band_square_plus`:
  `a ∓ 1 = g(g + e) ↔ 2Q ∓ 4 + e² = (2g + e)²`, and `band_square_exists_minus` /
  `band_square_exists_plus`: `2Q ∓ 4 + e²` is a square iff `a ∓ 1 = g(g + e)` for some `g`.
* (vii) Range. `deleted_iff_not_twin`: a survivor `j ≥ 1` is deleted iff some leg is composite.
  `mirror_cover` / `mirror_ne` / `mirror_no_fixed`: every copy of `[1, M - 1]` lies in a mirror
  pair, and no pair is a fixed point. `window_iff_revealed`: the range statement on copies holds
  iff some survivor in `[1, M - 1]` has both legs prime. `range_iff_pair`: it holds iff some mirror
  pair of survivors is not doubly deleted. `deleted_lo_iff` / `deleted_hi_iff`: the class form of
  deletion. `rangeCopyWindow_rangeStatement`: the copy form implies `RangeStatement q`.
-/

namespace RangeLine

/-! ### Notation -/

/-- `a = q# / 2`, the centre of the legs of every mirror pair. -/
def mirrorA (q : ℕ) : ℕ := primorial q / 2

/-- `ρ = q# / 2 + 1`; the upper legs of the mirror pair at offset `d` are `ρ ∓ 15d`. -/
def mirrorRho (q : ℕ) : ℕ := primorial q / 2 + 1

/-- Copy `j` survives the machine `q`: no prime `7 ≤ p ≤ q` strikes it. -/
def Survives (q j : ℕ) : Prop := ∀ p, p.Prime → 7 ≤ p → p ≤ q → ¬ StrikesCopy p j

/-- Both legs `30j - 1` and `30j + 1` of copy `j` are prime. -/
def TwinCopy (j : ℕ) : Prop := (30 * j - 1).Prime ∧ (30 * j + 1).Prime

/-- Copy `j` is deleted at machine `q`: some prime `g > q` strikes it and acts at it. -/
def Deleted (q j : ℕ) : Prop := ∃ g, g.Prime ∧ q < g ∧ StrikesCopy g j ∧ Acts g j

/-- The acting set `G(j)`: the primes `g > q` acting at copy `j` (`g² ≤ 30j + 1`). -/
def ActSet (q j : ℕ) : Set ℕ := {g | g.Prime ∧ q < g ∧ Acts g j}

/-- `U⁻`: the primes `g > q` with `g² ≤ ρ`. -/
def uMinusSet (q : ℕ) : Set ℕ := {g | g.Prime ∧ q < g ∧ g ^ 2 ≤ mirrorRho q}

/-- The range statement read on copies: some copy `j ≥ 1` with `q < 30j - 1` and
`30j + 1 ≤ q#` has both legs prime (the left side of `range_copy_iff`). -/
def RangeCopyWindow (q : ℕ) : Prop :=
  ∃ j, 1 ≤ j ∧ q < 30 * j - 1 ∧ 30 * j + 1 ≤ primorial q ∧ (30 * j - 1).Prime ∧
    (30 * j + 1).Prime

/-- `g₋`: the largest prime `≤ √ρ` (`0` if there is none). -/
def coreLow (q : ℕ) : ℕ := Nat.findGreatest Nat.Prime (Nat.sqrt (mirrorRho q))

/-- `g₊`: the least prime `> √ρ`. -/
def coreHigh (q : ℕ) : ℕ := Nat.find (Nat.exists_infinite_primes (Nat.sqrt (mirrorRho q) + 1))

/-- The core bound `R = min((ρ - g₋²)/15, largest d with 15d < g₊² - ρ)`. -/
def coreR (q : ℕ) : ℕ :=
  min ((mirrorRho q - coreLow q ^ 2) / 15) ((coreHigh q ^ 2 - mirrorRho q - 1) / 15)

/-- The four-leg form `F = (a - 15x - 1)(a - 15x + 1)(a + 15x - 1)(a + 15x + 1)` in a ring. -/
def legF {R : Type*} [CommRing R] (a x : R) : R :=
  (a - 15 * x - 1) * (a - 15 * x + 1) * (a + 15 * x - 1) * (a + 15 * x + 1)

/-! ### Basic data -/

/-- The linear data of a mirror pair, for `q ≥ 5` and odd `d < M`: `Q = 30M`, `a = 15M`,
`ρ = 15M + 1`, `2s + d = M`, `s' = s + d`, `s ≥ 1`, `M` odd, `d` odd. -/
theorem mirror_data {q d : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q) :
    primorial q = 30 * mirrorM q ∧ mirrorA q = 15 * mirrorM q ∧
      mirrorRho q = 15 * mirrorM q + 1 ∧ 2 * mirrorLo q d + d = mirrorM q ∧
      mirrorHi q d = mirrorLo q d + d ∧ 1 ≤ mirrorLo q d ∧ mirrorM q % 2 = 1 ∧ d % 2 = 1 := by
  have hQ := (thirty_mul_mirrorM hq).symm
  have ⟨m, hm⟩ := odd_mirrorM hq
  have ⟨e, he⟩ := hd
  refine ⟨hQ, ?_, ?_, ?_, mirrorHi_eq hq hd hdM, one_le_mirrorLo hq hd hdM, by omega, by omega⟩
  · unfold mirrorA
    rw [hQ]
    omega
  · unfold mirrorRho
    rw [hQ]
    omega
  · unfold mirrorLo
    omega

/-- A prime `g ≥ 7` does not divide `15`. -/
theorem not_dvd_fifteen {g : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g) : ¬ g ∣ 15 := by
  intro h
  have h35 : g ∣ 3 * 5 := h
  rcases (Nat.Prime.dvd_mul hg).1 h35 with h3 | h5
  · have := Nat.le_of_dvd (by norm_num) h3
    omega
  · have := Nat.le_of_dvd (by norm_num) h5
    omega

/-- For a prime `g ≥ 7`, `15 ≠ 0` in `ZMod g`. -/
theorem fifteen_ne_zero_zmod {g : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g) : (15 : ZMod g) ≠ 0 := by
  intro h
  have h' : ((15 : ℕ) : ZMod g) = 0 := by exact_mod_cast h
  rw [ZMod.natCast_eq_zero_iff] at h'
  exact not_dvd_fifteen hg hg7 h'

/-- For a prime `g ≥ 7`, `2 ≠ 0` in `ZMod g`. -/
theorem two_ne_zero_zmod {g : ℕ} (hg7 : 7 ≤ g) : (2 : ZMod g) ≠ 0 := by
  intro h
  have h' : ((2 : ℕ) : ZMod g) = 0 := by exact_mod_cast h
  rw [ZMod.natCast_eq_zero_iff] at h'
  have := Nat.le_of_dvd (by norm_num) h'
  omega

/-- A prime above `q ≥ 5` is at least `7`. -/
theorem seven_le_of_prime_gt {q g : ℕ} (hq : 5 ≤ q) (hg : g.Prime) (hgq : q < g) : 7 ≤ g := by
  by_contra h
  have h6 : g = 6 := by omega
  subst h6
  norm_num at hg

/-- A prime above `q` does not divide `q#`. -/
theorem not_dvd_primorial_of_gt {q g : ℕ} (hg : g.Prime) (hgq : q < g) : ¬ g ∣ primorial q := by
  rw [hg.dvd_primorial_iff]
  omega

/-- A prime `p` divides `M = q#/30` iff `7 ≤ p ≤ q` (for `q ≥ 5`). -/
theorem prime_dvd_mirrorM_iff {q p : ℕ} (hq : 5 ≤ q) (hp : p.Prime) :
    p ∣ mirrorM q ↔ 7 ≤ p ∧ p ≤ q := by
  have hQ := (thirty_mul_mirrorM hq).symm
  constructor
  · intro hpM
    have hpQ : p ∣ primorial q := by
      rw [hQ]
      exact Dvd.dvd.mul_left hpM 30
    refine ⟨?_, hp.dvd_primorial_iff.1 hpQ⟩
    by_contra h7
    have h7' : p < 7 := by omega
    have h30 : p ∣ 30 := by
      interval_cases p <;> first | (norm_num; done) | exact absurd hp (by norm_num)
    have hpp : p * p ∣ primorial q := by
      rw [hQ]
      exact mul_dvd_mul h30 hpM
    have hu := Nat.isUnit_iff.1 (squarefree_primorial q p hpp)
    exact hp.one_lt.ne' hu
  · rintro ⟨h7, hpq⟩
    have hpQ : p ∣ 30 * mirrorM q := by
      rw [← hQ]
      exact hp.dvd_primorial_iff.2 hpq
    have hcop : Nat.Coprime p 30 := (Nat.Prime.coprime_iff_not_dvd hp).2 (not_dvd_thirty hp h7)
    exact Nat.Coprime.dvd_of_dvd_mul_left hcop hpQ

/-! ### (ii) Acting windows -/

/-- **(ii) Acting windows.** `30s + 1 = ρ - 15d` and `30s' + 1 = ρ + 15d`
(with `30s + 1 + 15d = ρ`). -/
theorem acting_windows {q d : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q) :
    30 * mirrorLo q d + 1 = mirrorRho q - 15 * d ∧ 30 * mirrorLo q d + 1 + 15 * d = mirrorRho q ∧
      30 * mirrorHi q d + 1 = mirrorRho q + 15 * d := by
  obtain ⟨_, _, hR, h2, hH, _, _, _⟩ := mirror_data hq hd hdM
  omega

/-- **(ii) Acting on the low member.** `g` acts on `s` iff `15d ≤ ρ - g²`. -/
theorem acts_lo_iff {q d g : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q) :
    Acts g (mirrorLo q d) ↔ 15 * d ≤ mirrorRho q - g ^ 2 := by
  obtain ⟨_, _, hR, h2, hH, _, _, hdo⟩ := mirror_data hq hd hdM
  unfold Acts
  omega

/-- **(ii) Acting on the high member.** `g` acts on `s'` iff `g² - ρ ≤ 15d`. -/
theorem acts_hi_iff {q d g : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q) :
    Acts g (mirrorHi q d) ↔ g ^ 2 - mirrorRho q ≤ 15 * d := by
  obtain ⟨_, _, hR, h2, hH, _, _, _⟩ := mirror_data hq hd hdM
  unfold Acts
  omega

/-- The square of an odd number is `1 mod 4`. -/
theorem odd_sq_mod_four {g : ℕ} (hg : Odd g) : g ^ 2 % 4 = 1 := by
  obtain ⟨k, rfl⟩ := hg
  have e : (2 * k + 1) ^ 2 = 4 * (k ^ 2 + k) + 1 := by ring
  rw [e]
  omega

/-- **(ii) Joint acting.** If an odd `g` acts on the low member and an odd `h` acts on the high
member, then `g² + h² < Q + 2`: the height split gives `≤`, and equality is impossible because
`g² + h² ≡ 2` while `Q + 2 ≡ 0 (mod 4)`. -/
theorem joint_acting_sq {q d g h : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q)
    (hg : Odd g) (hh : Odd h) (h1 : Acts g (mirrorLo q d)) (h2 : Acts h (mirrorHi q d)) :
    g ^ 2 + h ^ 2 < primorial q + 2 := by
  obtain ⟨hQ, _, _, _, _, _, hMo, _⟩ := mirror_data hq hd hdM
  have hs := height_split hq hd hdM
  have hg4 := odd_sq_mod_four hg
  have hh4 := odd_sq_mod_four hh
  unfold Acts at h1 h2
  omega

/-- **(ii) Joint acting, product form.** If an odd `g` acts on the low member and an odd `h` acts
on the high member, then `g h < ρ`. -/
theorem joint_acting_mul {q d g h : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q)
    (hg : Odd g) (hh : Odd h) (h1 : Acts g (mirrorLo q d)) (h2 : Acts h (mirrorHi q d)) :
    g * h < mirrorRho q := by
  obtain ⟨hQ, _, hR, _, _, _, _, _⟩ := mirror_data hq hd hdM
  have hj := joint_acting_sq hq hd hdM hg hh h1 h2
  have h2gh : 2 * (g * h) ≤ g ^ 2 + h ^ 2 := by
    have := two_mul_le_add_sq g h
    rw [mul_assoc] at this
    exact this
  omega

/-! ### (i) Class form and survivors -/

/-- Class form of the low strike in a field: if `15t = 1` and `2x = M - d`, then
`(30x - 1)(30x + 1) = 0` iff `d = M - t` or `d = M + t`. -/
theorem class_lo_field {F : Type*} [Field F] {x M d t : F} (ht : 15 * t = 1)
    (hx : 2 * x = M - d) :
    (30 * x - 1) * (30 * x + 1) = 0 ↔ (d = M - t ∨ d = M + t) := by
  constructor
  · intro h
    rcases mul_eq_zero.1 h with h1 | h1
    · left
      have e : 15 * (M - d) = 1 := by linear_combination h1 - 15 * hx
      linear_combination (-t) * e + (M - d) * ht
    · right
      have e : 15 * (M - d) = -1 := by linear_combination h1 - 15 * hx
      linear_combination (-t) * e + (M - d) * ht
  · rintro (h | h)
    · linear_combination (30 * x + 1) * (15 * hx - 15 * h + ht)
    · linear_combination (30 * x - 1) * (15 * hx - 15 * h - ht)

/-- Class form of the high strike in a field: if `15t = 1` and `2x = M + d`, then
`(30x - 1)(30x + 1) = 0` iff `d = -M + t` or `d = -M - t`. -/
theorem class_hi_field {F : Type*} [Field F] {x M d t : F} (ht : 15 * t = 1)
    (hx : 2 * x = M + d) :
    (30 * x - 1) * (30 * x + 1) = 0 ↔ (d = -M + t ∨ d = -M - t) := by
  constructor
  · intro h
    rcases mul_eq_zero.1 h with h1 | h1
    · left
      have e : 15 * (M + d) = 1 := by linear_combination h1 - 15 * hx
      linear_combination t * e - (M + d) * ht
    · right
      have e : 15 * (M + d) = -1 := by linear_combination h1 - 15 * hx
      linear_combination t * e - (M + d) * ht
  · rintro (h | h)
    · linear_combination (30 * x + 1) * (15 * hx + 15 * h + ht)
    · linear_combination (30 * x - 1) * (15 * hx + 15 * h - ht)

/-- **(i) Class form, low member.** For every prime `g ≥ 7` (in particular every prime `g > q`),
`g` strikes `s` iff `d ≡ M - t_g` or `d ≡ M + t_g (mod g)`, where `t_g = 15⁻¹` in `ZMod g`. -/
theorem strikes_lo_iff_class {q d g : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q)
    (hg : g.Prime) (hg7 : 7 ≤ g) :
    StrikesCopy g (mirrorLo q d) ↔
      ((d : ZMod g) = (mirrorM q : ZMod g) - (15 : ZMod g)⁻¹ ∨
        (d : ZMod g) = (mirrorM q : ZMod g) + (15 : ZMod g)⁻¹) := by
  have := Fact.mk hg
  obtain ⟨_, _, _, h2, _, hs1, _, _⟩ := mirror_data hq hd hdM
  rw [strikesCopy_iff_zmod hs1]
  apply class_lo_field (mul_inv_cancel₀ (fifteen_ne_zero_zmod hg hg7))
  have := congrArg (Nat.cast : ℕ → ZMod g) h2
  push_cast at this
  linear_combination this

/-- **(i) Class form, high member.** For every prime `g ≥ 7`, `g` strikes `s'` iff
`d ≡ -M + t_g` or `d ≡ -M - t_g (mod g)`. -/
theorem strikes_hi_iff_class {q d g : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q)
    (hg : g.Prime) (hg7 : 7 ≤ g) :
    StrikesCopy g (mirrorHi q d) ↔
      ((d : ZMod g) = -(mirrorM q : ZMod g) + (15 : ZMod g)⁻¹ ∨
        (d : ZMod g) = -(mirrorM q : ZMod g) - (15 : ZMod g)⁻¹) := by
  have := Fact.mk hg
  obtain ⟨_, _, _, h2, hH, hs1, _, _⟩ := mirror_data hq hd hdM
  have h2' : 2 * mirrorHi q d = mirrorM q + d := by omega
  rw [strikesCopy_iff_zmod (show 1 ≤ mirrorHi q d by omega)]
  apply class_hi_field (mul_inv_cancel₀ (fifteen_ne_zero_zmod hg hg7))
  have := congrArg (Nat.cast : ℕ → ZMod g) h2'
  push_cast at this
  linear_combination this

/-- **(i) Machine gears on the low member.** For every divisor `g` of `M`, `g` strikes `s` iff
`g ∣ 225d² - 1` (the class form with `M ≡ 0`: `15d ≡ ±1`). -/
theorem strikes_lo_iff_of_dvd {q d g : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q)
    (hgM : g ∣ mirrorM q) :
    StrikesCopy g (mirrorLo q d) ↔ g ∣ 225 * d ^ 2 - 1 := by
  obtain ⟨_, _, _, h2, _, hs1, _, hdo⟩ := mirror_data hq hd hdM
  rw [strikesCopy_iff_sq]
  have hs2 : 1 ≤ mirrorLo q d ^ 2 := Nat.one_le_pow _ _ (by omega)
  have hd2 : 1 ≤ d ^ 2 := Nat.one_le_pow _ _ (by omega)
  have h900 : 1 ≤ 900 * mirrorLo q d ^ 2 := by omega
  have h225 : 1 ≤ 225 * d ^ 2 := by omega
  rw [← Int.natCast_dvd_natCast, ← Int.natCast_dvd_natCast (m := g) (n := 225 * d ^ 2 - 1)]
  push_cast [Nat.cast_sub h900, Nat.cast_sub h225]
  have hz : 2 * (mirrorLo q d : ℤ) + d = mirrorM q := by exact_mod_cast h2
  have key : 900 * (mirrorLo q d : ℤ) ^ 2 - 1 =
      (225 * (d : ℤ) ^ 2 - 1) + (mirrorM q : ℤ) * (225 * ((mirrorM q : ℤ) - 2 * d)) := by
    linear_combination (225 * ((mirrorM q : ℤ) - d + 2 * mirrorLo q d)) * hz
  rw [key]
  exact dvd_add_left (Dvd.dvd.mul_right (Int.natCast_dvd_natCast.2 hgM) _)

/-- **(i) Machine gears on the high member.** For every divisor `g` of `M`, `g` strikes `s'` iff
`g ∣ 225d² - 1`. -/
theorem strikes_hi_iff_of_dvd {q d g : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q)
    (hgM : g ∣ mirrorM q) :
    StrikesCopy g (mirrorHi q d) ↔ g ∣ 225 * d ^ 2 - 1 := by
  obtain ⟨_, _, _, h2, hH, hs1, _, hdo⟩ := mirror_data hq hd hdM
  rw [strikesCopy_iff_sq]
  have h2' : 2 * mirrorHi q d = mirrorM q + d := by omega
  have hs2 : 1 ≤ mirrorHi q d ^ 2 := Nat.one_le_pow _ _ (by omega)
  have hd2 : 1 ≤ d ^ 2 := Nat.one_le_pow _ _ (by omega)
  have h900 : 1 ≤ 900 * mirrorHi q d ^ 2 := by omega
  have h225 : 1 ≤ 225 * d ^ 2 := by omega
  rw [← Int.natCast_dvd_natCast, ← Int.natCast_dvd_natCast (m := g) (n := 225 * d ^ 2 - 1)]
  push_cast [Nat.cast_sub h900, Nat.cast_sub h225]
  have hz : 2 * (mirrorHi q d : ℤ) = mirrorM q + d := by exact_mod_cast h2'
  have key : 900 * (mirrorHi q d : ℤ) ^ 2 - 1 =
      (225 * (d : ℤ) ^ 2 - 1) + (mirrorM q : ℤ) * (225 * ((mirrorM q : ℤ) + 2 * d)) := by
    linear_combination (225 * (2 * (mirrorHi q d : ℤ) + mirrorM q + d)) * hz
  rw [key]
  exact dvd_add_left (Dvd.dvd.mul_right (Int.natCast_dvd_natCast.2 hgM) _)

/-- **(i) Survivors, low member.** `s` survives `q` iff `gcd(225d² - 1, M) = 1`. -/
theorem survives_lo_iff {q d : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q) :
    Survives q (mirrorLo q d) ↔ Nat.Coprime (225 * d ^ 2 - 1) (mirrorM q) := by
  constructor
  · intro hS
    apply Nat.coprime_of_dvd
    intro p hp h1 h2
    obtain ⟨h7, hpq⟩ := (prime_dvd_mirrorM_iff hq hp).1 h2
    exact hS p hp h7 hpq ((strikes_lo_iff_of_dvd hq hd hdM h2).2 h1)
  · intro hC p hp h7 hpq hstr
    have hpM := (prime_dvd_mirrorM_iff hq hp).2 ⟨h7, hpq⟩
    have h1 := (strikes_lo_iff_of_dvd hq hd hdM hpM).1 hstr
    have h := Nat.dvd_gcd h1 hpM
    rw [Nat.Coprime.gcd_eq_one hC] at h
    exact hp.one_lt.ne' (Nat.dvd_one.1 h)

/-- **(i) Survivors, high member.** `s'` survives `q` iff `gcd(225d² - 1, M) = 1`. -/
theorem survives_hi_iff {q d : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q) :
    Survives q (mirrorHi q d) ↔ Nat.Coprime (225 * d ^ 2 - 1) (mirrorM q) := by
  constructor
  · intro hS
    apply Nat.coprime_of_dvd
    intro p hp h1 h2
    obtain ⟨h7, hpq⟩ := (prime_dvd_mirrorM_iff hq hp).1 h2
    exact hS p hp h7 hpq ((strikes_hi_iff_of_dvd hq hd hdM h2).2 h1)
  · intro hC p hp h7 hpq hstr
    have hpM := (prime_dvd_mirrorM_iff hq hp).2 ⟨h7, hpq⟩
    have h1 := (strikes_hi_iff_of_dvd hq hd hdM hpM).1 hstr
    have h := Nat.dvd_gcd h1 hpM
    rw [Nat.Coprime.gcd_eq_one hC] at h
    exact hp.one_lt.ne' (Nat.dvd_one.1 h)

/-- **(i) Both members survive or neither does.** -/
theorem survives_pair {q d : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q) :
    Survives q (mirrorLo q d) ↔ Survives q (mirrorHi q d) := by
  rw [survives_lo_iff hq hd hdM, survives_hi_iff hq hd hdM]

/-! ### (iii) Shared strikers and joint acting -/

/-- **(iii) Minus legs together.** For every `g` coprime to `30`: `g ∣ L1 ∧ g ∣ H1` iff
`g ∣ Q - 2 ∧ g ∣ d` (`L1 + H1 = Q - 2`, `H1 - L1 = 30d`, `L1 = (a - 1) - 15d`). -/
theorem same_minus_iff {q d g : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q)
    (hg : Nat.Coprime g 30) :
    (g ∣ 30 * mirrorLo q d - 1 ∧ g ∣ 30 * mirrorHi q d - 1) ↔
      (g ∣ primorial q - 2 ∧ g ∣ d) := by
  obtain ⟨hQ, hA, _, h2, hH, hs1, _, _⟩ := mirror_data hq hd hdM
  have hcop2 : Nat.Coprime g 2 := Nat.Coprime.coprime_dvd_right (by norm_num) hg
  have eH : 30 * mirrorHi q d - 1 = (30 * mirrorLo q d - 1) + 30 * d := by omega
  constructor
  · rintro ⟨h1, h2'⟩
    refine ⟨?_, ?_⟩
    · have e : primorial q - 2 = (30 * mirrorLo q d - 1) + (30 * mirrorHi q d - 1) := by omega
      rw [e]
      exact dvd_add h1 h2'
    · rw [eH] at h2'
      exact Nat.Coprime.dvd_of_dvd_mul_left hg ((Nat.dvd_add_right h1).1 h2')
  · rintro ⟨hQ2, hdd⟩
    have ha : g ∣ mirrorA q - 1 := by
      have e : primorial q - 2 = 2 * (mirrorA q - 1) := by omega
      rw [e] at hQ2
      exact Nat.Coprime.dvd_of_dvd_mul_left hcop2 hQ2
    have hL : g ∣ 30 * mirrorLo q d - 1 := by
      have e : mirrorA q - 1 = (30 * mirrorLo q d - 1) + 15 * d := by omega
      rw [e] at ha
      exact (Nat.dvd_add_left (Dvd.dvd.mul_left hdd 15)).1 ha
    refine ⟨hL, ?_⟩
    rw [eH]
    exact dvd_add hL (Dvd.dvd.mul_left hdd 30)

/-- **(iii) Plus legs together.** For every `g` coprime to `30`: `g ∣ L2 ∧ g ∣ H2` iff
`g ∣ Q + 2 ∧ g ∣ d` (`L2 + H2 = Q + 2`, `H2 - L2 = 30d`, `L2 = (a + 1) - 15d`). -/
theorem same_plus_iff {q d g : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q)
    (hg : Nat.Coprime g 30) :
    (g ∣ 30 * mirrorLo q d + 1 ∧ g ∣ 30 * mirrorHi q d + 1) ↔
      (g ∣ primorial q + 2 ∧ g ∣ d) := by
  obtain ⟨hQ, hA, _, h2, hH, hs1, _, _⟩ := mirror_data hq hd hdM
  have hcop2 : Nat.Coprime g 2 := Nat.Coprime.coprime_dvd_right (by norm_num) hg
  have eH : 30 * mirrorHi q d + 1 = (30 * mirrorLo q d + 1) + 30 * d := by omega
  constructor
  · rintro ⟨h1, h2'⟩
    refine ⟨?_, ?_⟩
    · have e : primorial q + 2 = (30 * mirrorLo q d + 1) + (30 * mirrorHi q d + 1) := by omega
      rw [e]
      exact dvd_add h1 h2'
    · rw [eH] at h2'
      exact Nat.Coprime.dvd_of_dvd_mul_left hg ((Nat.dvd_add_right h1).1 h2')
  · rintro ⟨hQ2, hdd⟩
    have ha : g ∣ mirrorA q + 1 := by
      have e : primorial q + 2 = 2 * (mirrorA q + 1) := by omega
      rw [e] at hQ2
      exact Nat.Coprime.dvd_of_dvd_mul_left hcop2 hQ2
    have hL : g ∣ 30 * mirrorLo q d + 1 := by
      have e : mirrorA q + 1 = (30 * mirrorLo q d + 1) + 15 * d := by omega
      rw [e] at ha
      exact (Nat.dvd_add_left (Dvd.dvd.mul_left hdd 15)).1 ha
    refine ⟨hL, ?_⟩
    rw [eH]
    exact dvd_add hL (Dvd.dvd.mul_left hdd 30)

/-- **(iii) No cross strike above `q`.** A prime `g > q` never divides both `L1` and `H2`, nor
both `L2` and `H1` (either pair sums to `Q`, and `g ∤ Q`). -/
theorem no_cross_strike {q d g : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q)
    (hg : g.Prime) (hgq : q < g) :
    ¬ (g ∣ 30 * mirrorLo q d - 1 ∧ g ∣ 30 * mirrorHi q d + 1) ∧
      ¬ (g ∣ 30 * mirrorLo q d + 1 ∧ g ∣ 30 * mirrorHi q d - 1) := by
  obtain ⟨_, c2, c3, _⟩ := shared_striker_cases (g := g) hq hd hdM
  have hgQ := not_dvd_primorial_of_gt hg hgq
  exact ⟨fun ⟨h1, h2⟩ => hgQ (c2 h1 h2), fun ⟨h1, h2⟩ => hgQ (c3 h1 h2)⟩

/-- **(iii) Strikes on both members.** A prime `g > q` strikes both `s` and `s'` iff
`g ∣ (Q - 2)(Q + 2)` and `g ∣ d`. -/
theorem strikes_both_iff {q d g : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q)
    (hg : g.Prime) (hgq : q < g) :
    (StrikesCopy g (mirrorLo q d) ∧ StrikesCopy g (mirrorHi q d)) ↔
      (g ∣ (primorial q - 2) * (primorial q + 2) ∧ g ∣ d) := by
  have hg7 := seven_le_of_prime_gt hq hg hgq
  have hcop : Nat.Coprime g 30 := (Nat.Prime.coprime_iff_not_dvd hg).2 (not_dvd_thirty hg hg7)
  obtain ⟨nc1, nc2⟩ := no_cross_strike hq hd hdM hg hgq
  rw [strikesCopy_iff_leg hg, strikesCopy_iff_leg hg]
  constructor
  · rintro ⟨h1 | h1, h2 | h2⟩
    · obtain ⟨hQ2, hdd⟩ := (same_minus_iff hq hd hdM hcop).1 ⟨h1, h2⟩
      exact ⟨Dvd.dvd.mul_right hQ2 _, hdd⟩
    · exact absurd ⟨h1, h2⟩ nc1
    · exact absurd ⟨h1, h2⟩ nc2
    · obtain ⟨hQ2, hdd⟩ := (same_plus_iff hq hd hdM hcop).1 ⟨h1, h2⟩
      exact ⟨Dvd.dvd.mul_left hQ2 _, hdd⟩
  · rintro ⟨hprod, hdd⟩
    rcases (Nat.Prime.dvd_mul hg).1 hprod with h | h
    · obtain ⟨h1, h2⟩ := (same_minus_iff hq hd hdM hcop).2 ⟨h, hdd⟩
      exact ⟨Or.inl h1, Or.inl h2⟩
    · obtain ⟨h1, h2⟩ := (same_plus_iff hq hd hdM hcop).2 ⟨h, hdd⟩
      exact ⟨Or.inr h1, Or.inr h2⟩

/-- **(iii) Legs of the same type.** If a prime `g > q` strikes both members, then when
`g ∣ Q - 2` it divides exactly the minus legs `L1, H1` (not `L2, H2`), and when `g ∣ Q + 2` it
divides exactly the plus legs `L2, H2` (not `L1, H1`); and one of the two cases holds. -/
theorem strikes_both_leg_type {q d g : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q)
    (hg : g.Prime) (hgq : q < g)
    (hboth : StrikesCopy g (mirrorLo q d) ∧ StrikesCopy g (mirrorHi q d)) :
    (g ∣ primorial q - 2 ∨ g ∣ primorial q + 2) ∧
    (g ∣ primorial q - 2 → g ∣ 30 * mirrorLo q d - 1 ∧ g ∣ 30 * mirrorHi q d - 1 ∧
        ¬ g ∣ 30 * mirrorLo q d + 1 ∧ ¬ g ∣ 30 * mirrorHi q d + 1) ∧
      (g ∣ primorial q + 2 → g ∣ 30 * mirrorLo q d + 1 ∧ g ∣ 30 * mirrorHi q d + 1 ∧
        ¬ g ∣ 30 * mirrorLo q d - 1 ∧ ¬ g ∣ 30 * mirrorHi q d - 1) := by
  have hg7 := seven_le_of_prime_gt hq hg hgq
  have hcop : Nat.Coprime g 30 := (Nat.Prime.coprime_iff_not_dvd hg).2 (not_dvd_thirty hg hg7)
  obtain ⟨hprod, hdd⟩ := (strikes_both_iff hq hd hdM hg hgq).1 hboth
  obtain ⟨_, _, _, h2, hH, hs1, _, _⟩ := mirror_data hq hd hdM
  have hn2 : ¬ g ∣ 2 := fun h => absurd (Nat.le_of_dvd (by norm_num) h) (by omega)
  have eL : 30 * mirrorLo q d + 1 = (30 * mirrorLo q d - 1) + 2 := by omega
  have eH : 30 * mirrorHi q d + 1 = (30 * mirrorHi q d - 1) + 2 := by omega
  refine ⟨(Nat.Prime.dvd_mul hg).1 hprod, ?_, ?_⟩
  · intro h
    obtain ⟨h1, h2'⟩ := (same_minus_iff hq hd hdM hcop).2 ⟨h, hdd⟩
    refine ⟨h1, h2', ?_, ?_⟩
    · rw [eL]
      exact fun h' => hn2 ((Nat.dvd_add_right h1).1 h')
    · rw [eH]
      exact fun h' => hn2 ((Nat.dvd_add_right h2').1 h')
  · intro h
    obtain ⟨h1, h2'⟩ := (same_plus_iff hq hd hdM hcop).2 ⟨h, hdd⟩
    refine ⟨h1, h2', ?_, ?_⟩
    · intro h'
      rw [eL] at h1
      exact hn2 ((Nat.dvd_add_right h').1 h1)
    · intro h'
      rw [eH] at h2'
      exact hn2 ((Nat.dvd_add_right h').1 h2')

/-- **(iii) Acting on both members.** `g` acts on both `s` and `s'` iff `15d ≤ ρ - g²`
(acting on `s'` follows from acting on `s`). -/
theorem acts_both_iff {q d g : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q) :
    (Acts g (mirrorLo q d) ∧ Acts g (mirrorHi q d)) ↔ 15 * d ≤ mirrorRho q - g ^ 2 := by
  obtain ⟨_, _, hR, h2, hH, _, _, hdo⟩ := mirror_data hq hd hdM
  unfold Acts
  omega

/-- **(iii) Acting on the low member forces `U⁻`.** If `g` acts on `s` then `g² + 15 ≤ ρ`; in
particular `g² < ρ`. -/
theorem acts_lo_sq_lt {q d g : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q)
    (h : Acts g (mirrorLo q d)) : g ^ 2 + 15 ≤ mirrorRho q := by
  obtain ⟨_, _, hR, h2, hH, _, _, hdo⟩ := mirror_data hq hd hdM
  unfold Acts at h
  omega

/-- **(iii) Deleting both members with one gear.** A prime `g > q` strikes and acts on both
members iff `g ∣ (Q - 2)(Q + 2)`, `g ∣ d` and `15d ≤ ρ - g²`. -/
theorem deletes_both_iff {q d g : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q)
    (hg : g.Prime) (hgq : q < g) :
    (StrikesCopy g (mirrorLo q d) ∧ Acts g (mirrorLo q d) ∧
        StrikesCopy g (mirrorHi q d) ∧ Acts g (mirrorHi q d)) ↔
      (g ∣ (primorial q - 2) * (primorial q + 2) ∧ g ∣ d ∧ 15 * d ≤ mirrorRho q - g ^ 2) := by
  constructor
  · rintro ⟨h1, h2, h3, h4⟩
    obtain ⟨hp, hdd⟩ := (strikes_both_iff hq hd hdM hg hgq).1 ⟨h1, h3⟩
    exact ⟨hp, hdd, (acts_both_iff hq hd hdM).1 ⟨h2, h4⟩⟩
  · rintro ⟨hp, hdd, hle⟩
    obtain ⟨h1, h3⟩ := (strikes_both_iff hq hd hdM hg hgq).2 ⟨hp, hdd⟩
    obtain ⟨h2, h4⟩ := (acts_both_iff hq hd hdM).2 hle
    exact ⟨h1, h2, h3, h4⟩

/-- **(iii) A single gear deleting both members lies in `U⁻`.** -/
theorem deletes_both_mem_uMinus {q d g : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q)
    (hg : g.Prime) (hgq : q < g) (ha : Acts g (mirrorLo q d)) : g ∈ uMinusSet q := by
  have := acts_lo_sq_lt hq hd hdM ha
  exact ⟨hg, hgq, by omega⟩

/-- **(vii) Single-gear deletion of both members** needs `g ∣ Q - 2` or `g ∣ Q + 2`, with
`g ∣ d` and `g² ≤ 30s + 1`. -/
theorem single_gear_deletion {q d g : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q)
    (hg : g.Prime) (hgq : q < g)
    (h : StrikesCopy g (mirrorLo q d) ∧ Acts g (mirrorLo q d) ∧
        StrikesCopy g (mirrorHi q d) ∧ Acts g (mirrorHi q d)) :
    (g ∣ primorial q - 2 ∨ g ∣ primorial q + 2) ∧ g ∣ d ∧ g ^ 2 ≤ 30 * mirrorLo q d + 1 := by
  obtain ⟨hprod, hdd, _⟩ := (deletes_both_iff hq hd hdM hg hgq).1 h
  exact ⟨(Nat.Prime.dvd_mul hg).1 hprod, hdd, h.2.1⟩

/-! ### (iv) Sandwich and core -/

/-- **(iv) Sandwich.** `G(s) ⊆ U⁻ ⊆ G(s')`: a gear acting on the low member has `g² ≤ ρ`, and a
gear with `g² ≤ ρ` acts on the high member. -/
theorem sandwich {q d : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q) :
    ActSet q (mirrorLo q d) ⊆ uMinusSet q ∧ uMinusSet q ⊆ ActSet q (mirrorHi q d) := by
  obtain ⟨_, _, hR, h2, hH, _, _, _⟩ := mirror_data hq hd hdM
  refine ⟨?_, ?_⟩
  · rintro g ⟨hg, hgq, ha⟩
    unfold Acts at ha
    exact ⟨hg, hgq, by omega⟩
  · rintro g ⟨hg, hgq, ha⟩
    refine ⟨hg, hgq, ?_⟩
    unfold Acts
    omega

/-- **(iv) Core.** If every prime `g > q` with `g² ≤ ρ` has `g² + 15d ≤ ρ`, and every prime
`g > q` with `g² > ρ` has `g² > ρ + 15d`, then `G(s) = G(s') = U⁻`. -/
theorem core_eq {q d : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q)
    (hlow : ∀ g, g.Prime → q < g → g ^ 2 ≤ mirrorRho q → g ^ 2 + 15 * d ≤ mirrorRho q)
    (hhigh : ∀ g, g.Prime → q < g → mirrorRho q < g ^ 2 → mirrorRho q + 15 * d < g ^ 2) :
    ActSet q (mirrorLo q d) = uMinusSet q ∧ ActSet q (mirrorHi q d) = uMinusSet q := by
  obtain ⟨hs1, hs2⟩ := sandwich hq hd hdM
  obtain ⟨_, _, hR, h2, hH, _, _, _⟩ := mirror_data hq hd hdM
  refine ⟨Set.Subset.antisymm hs1 ?_, Set.Subset.antisymm ?_ hs2⟩
  · rintro g ⟨hg, hgq, hsq⟩
    have := hlow g hg hgq hsq
    refine ⟨hg, hgq, ?_⟩
    unfold Acts
    omega
  · rintro g ⟨hg, hgq, ha⟩
    refine ⟨hg, hgq, ?_⟩
    unfold Acts at ha
    by_contra hc
    have := hhigh g hg hgq (by omega)
    omega

/-- `g₋² ≤ ρ`. -/
theorem coreLow_sq_le (q : ℕ) : coreLow q ^ 2 ≤ mirrorRho q :=
  le_trans (Nat.pow_le_pow_left (Nat.findGreatest_le _) 2) (Nat.sqrt_le' _)

/-- Every prime `g` with `g² ≤ ρ` is at most `g₋`. -/
theorem le_coreLow {q g : ℕ} (hg : g.Prime) (hsq : g ^ 2 ≤ mirrorRho q) : g ≤ coreLow q :=
  Nat.le_findGreatest (Nat.le_sqrt'.2 hsq) hg

/-- Every prime `g` with `g² > ρ` is at least `g₊`. -/
theorem coreHigh_le {q g : ℕ} (hg : g.Prime) (hsq : mirrorRho q < g ^ 2) : coreHigh q ≤ g :=
  Nat.find_min' _ ⟨Nat.sqrt_lt'.2 hsq, hg⟩

/-- **(iv) Core, with the bound `R`.** If `15d ≤ ρ - g₋²` and `15d < g₊² - ρ` (i.e. `d ≤ R`),
then `G(s) = G(s') = U⁻`. -/
theorem core_eq_of_bounds {q d : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q)
    (h1 : 15 * d ≤ mirrorRho q - coreLow q ^ 2) (h2 : 15 * d < coreHigh q ^ 2 - mirrorRho q) :
    ActSet q (mirrorLo q d) = uMinusSet q ∧ ActSet q (mirrorHi q d) = uMinusSet q := by
  have hL := coreLow_sq_le q
  apply core_eq hq hd hdM
  · intro g hg _ hsq
    have := Nat.pow_le_pow_left (le_coreLow hg hsq) 2
    omega
  · intro g hg _ hsq
    have := Nat.pow_le_pow_left (coreHigh_le hg hsq) 2
    omega

/-- **(iv) Core, `d ≤ R`.** For odd `d < M` with `d ≤ R = coreR q`,
`G(s) = G(s') = U⁻`. -/
theorem core_eq_of_le_coreR {q d : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q)
    (hR : d ≤ coreR q) :
    ActSet q (mirrorLo q d) = uMinusSet q ∧ ActSet q (mirrorHi q d) = uMinusSet q := by
  have hd1 : 1 ≤ d := by
    have ⟨e, he⟩ := hd
    omega
  unfold coreR at hR
  apply core_eq_of_bounds hq hd hdM
  · have := min_le_left ((mirrorRho q - coreLow q ^ 2) / 15)
      ((coreHigh q ^ 2 - mirrorRho q - 1) / 15)
    omega
  · have := min_le_right ((mirrorRho q - coreLow q ^ 2) / 15)
      ((coreHigh q ^ 2 - mirrorRho q - 1) / 15)
    omega

/-! ### (v) Leg classes and root classes -/

/-- **(v) Leg classes in `15d`-space.** Modulo any `n`: `n ∣ L1` iff `15d ≡ a - 1`, `n ∣ L2` iff
`15d ≡ a + 1`, `n ∣ H1` iff `15d ≡ 1 - a`, and `n ∣ H2` iff `15d ≡ -1 - a`. -/
theorem leg_classes {q d : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q) (n : ℕ) :
    (n ∣ 30 * mirrorLo q d - 1 ↔ 15 * (d : ZMod n) = (mirrorA q : ZMod n) - 1) ∧
    (n ∣ 30 * mirrorLo q d + 1 ↔ 15 * (d : ZMod n) = (mirrorA q : ZMod n) + 1) ∧
    (n ∣ 30 * mirrorHi q d - 1 ↔ 15 * (d : ZMod n) = 1 - (mirrorA q : ZMod n)) ∧
    (n ∣ 30 * mirrorHi q d + 1 ↔ 15 * (d : ZMod n) = -1 - (mirrorA q : ZMod n)) := by
  obtain ⟨_, hA, _, h2, hH, hs1, _, _⟩ := mirror_data hq hd hdM
  have eL : 30 * mirrorLo q d + 15 * d = mirrorA q := by omega
  have eH : 30 * mirrorHi q d = mirrorA q + 15 * d := by omega
  have EL : 30 * (mirrorLo q d : ZMod n) + 15 * d = mirrorA q := by
    have := congrArg (Nat.cast : ℕ → ZMod n) eL
    push_cast at this
    exact this
  have EH : 30 * (mirrorHi q d : ZMod n) = mirrorA q + 15 * d := by
    have := congrArg (Nat.cast : ℕ → ZMod n) eH
    push_cast at this
    exact this
  have h1L : 1 ≤ 30 * mirrorLo q d := by omega
  have h1H : 1 ≤ 30 * mirrorHi q d := by omega
  simp only [← ZMod.natCast_eq_zero_iff]
  push_cast [Nat.cast_sub h1L, Nat.cast_sub h1H]
  refine ⟨⟨fun h => ?_, fun h => ?_⟩, ⟨fun h => ?_, fun h => ?_⟩, ⟨fun h => ?_, fun h => ?_⟩,
    ⟨fun h => ?_, fun h => ?_⟩⟩
  · linear_combination EL - h
  · linear_combination EL - h
  · linear_combination EL - h
  · linear_combination EL - h
  · linear_combination h - EH
  · linear_combination EH + h
  · linear_combination h - EH
  · linear_combination EH + h

/-- The four-leg product read modulo any `n` is `legF a d`:
`L1·L2·H1·H2 ≡ (a - 15d - 1)(a - 15d + 1)(a + 15d - 1)(a + 15d + 1)`. -/
theorem legF_cast {q d : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q) (n : ℕ) :
    (((30 * mirrorLo q d - 1) * (30 * mirrorLo q d + 1) *
        (30 * mirrorHi q d - 1) * (30 * mirrorHi q d + 1) : ℕ) : ZMod n) =
      legF (mirrorA q : ZMod n) (d : ZMod n) := by
  obtain ⟨_, hA, _, h2, hH, hs1, _, _⟩ := mirror_data hq hd hdM
  have eL : 30 * mirrorLo q d + 15 * d = mirrorA q := by omega
  have eH : 30 * mirrorHi q d = mirrorA q + 15 * d := by omega
  have EL : 30 * (mirrorLo q d : ZMod n) = mirrorA q - 15 * d := by
    have := congrArg (Nat.cast : ℕ → ZMod n) eL
    push_cast at this
    linear_combination this
  have EH : 30 * (mirrorHi q d : ZMod n) = mirrorA q + 15 * d := by
    have := congrArg (Nat.cast : ℕ → ZMod n) eH
    push_cast at this
    exact this
  have h1L : 1 ≤ 30 * mirrorLo q d := by omega
  have h1H : 1 ≤ 30 * mirrorHi q d := by omega
  push_cast [Nat.cast_sub h1L, Nat.cast_sub h1H]
  rw [EL, EH]
  unfold legF
  ring

/-- In a field with `15t = 1`: `c - 15x = 0` iff `x = t c`. -/
theorem lin_root {F : Type*} [Field F] {t : F} (ht : 15 * t = 1) (c x : F) :
    c - 15 * x = 0 ↔ x = t * c := by
  constructor
  · intro h
    linear_combination (-t) * h - x * ht
  · intro h
    linear_combination (-15) * h - c * ht

/-- **(v) Root classes of `F`.** In a field with `15t = 1`, `legF a x = 0` iff `x` is one of the
four classes `t(a - 1)`, `t(a + 1)`, `t(1 - a)`, `t(-1 - a)`. -/
theorem legF_eq_zero_iff {F : Type*} [Field F] {t : F} (ht : 15 * t = 1) (a x : F) :
    legF a x = 0 ↔
      (x = t * (a - 1) ∨ x = t * (a + 1) ∨ x = t * (1 - a) ∨ x = t * (-1 - a)) := by
  unfold legF
  have e1 : a - 15 * x - 1 = (a - 1) - 15 * x := by ring
  have e2 : a - 15 * x + 1 = (a + 1) - 15 * x := by ring
  have e3 : a + 15 * x - 1 = -((1 - a) - 15 * x) := by ring
  have e4 : a + 15 * x + 1 = -((-1 - a) - 15 * x) := by ring
  rw [e1, e2, e3, e4]
  simp only [mul_eq_zero, neg_eq_zero, lin_root ht, or_assoc]

/-- **(v) Root classes of the four-leg product.** For a prime `g ≥ 7`, `g` divides
`F(d) = L1·L2·H1·H2` iff `d ≡ t(a - 1), t(a + 1), t(1 - a)` or `t(-1 - a) (mod g)`, with
`t = 15⁻¹`. -/
theorem four_leg_roots {q d g : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q)
    (hg : g.Prime) (hg7 : 7 ≤ g) :
    g ∣ (30 * mirrorLo q d - 1) * (30 * mirrorLo q d + 1) *
        (30 * mirrorHi q d - 1) * (30 * mirrorHi q d + 1) ↔
      ((d : ZMod g) = (15 : ZMod g)⁻¹ * ((mirrorA q : ZMod g) - 1) ∨
        (d : ZMod g) = (15 : ZMod g)⁻¹ * ((mirrorA q : ZMod g) + 1) ∨
        (d : ZMod g) = (15 : ZMod g)⁻¹ * (1 - (mirrorA q : ZMod g)) ∨
        (d : ZMod g) = (15 : ZMod g)⁻¹ * (-1 - (mirrorA q : ZMod g))) := by
  have := Fact.mk hg
  rw [← ZMod.natCast_eq_zero_iff, legF_cast hq hd hdM g]
  exact legF_eq_zero_iff (mul_inv_cancel₀ (fifteen_ne_zero_zmod hg hg7)) _ _

/-- A four-element list is duplicate-free iff its six pairs are distinct. -/
theorem nodup_four {α : Type*} (x1 x2 x3 x4 : α) :
    [x1, x2, x3, x4].Nodup ↔
      (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) := by
  simp only [List.nodup_cons, List.mem_cons, List.not_mem_nil, List.nodup_nil, or_false, not_or]
  tauto

/-- A three-element list is duplicate-free iff its three pairs are distinct. -/
theorem nodup_three {α : Type*} (x1 x2 x3 : α) :
    [x1, x2, x3].Nodup ↔ (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) := by
  simp only [List.nodup_cons, List.mem_cons, List.not_mem_nil, List.nodup_nil, or_false, not_or]
  tauto

/-- **(v) Distinct classes in a field.** With `15t = 1` and `2 ≠ 0`, the four classes
`t(a - 1), t(a + 1), t(1 - a), t(-1 - a)` are distinct iff `a ≠ 0`, `a ≠ 1` and `a ≠ -1`. -/
theorem classes_nodup_field {F : Type*} [Field F] {t : F} (ht : 15 * t = 1)
    (h2 : (2 : F) ≠ 0) (a : F) :
    [t * (a - 1), t * (a + 1), t * (1 - a), t * (-1 - a)].Nodup ↔
      (a ≠ 0 ∧ a ≠ 1 ∧ a ≠ -1) := by
  have ht0 : t ≠ 0 := by
    rintro rfl
    simp at ht
  rw [nodup_four]
  simp only [ne_eq, mul_right_inj' ht0]
  constructor
  · rintro ⟨_, h13, h14, _, h24, _⟩
    refine ⟨fun h0 => h14 (by linear_combination 2 * h0),
      fun h1 => h13 (by linear_combination 2 * h1), fun hm => h24 (by linear_combination 2 * hm)⟩
  · rintro ⟨h0, h1, hm⟩
    refine ⟨fun h => ?_, fun h => ?_, fun h => ?_, fun h => ?_, fun h => ?_, fun h => ?_⟩
    · exact h2 (by linear_combination -h)
    · have e : 2 * (a - 1) = 0 := by linear_combination h
      exact h1 (sub_eq_zero.1 ((mul_eq_zero.1 e).resolve_left h2))
    · have e : 2 * a = 0 := by linear_combination h
      exact h0 ((mul_eq_zero.1 e).resolve_left h2)
    · have e : 2 * a = 0 := by linear_combination h
      exact h0 ((mul_eq_zero.1 e).resolve_left h2)
    · have e : 2 * (a + 1) = 0 := by linear_combination h
      exact hm (eq_neg_of_add_eq_zero_left ((mul_eq_zero.1 e).resolve_left h2))
    · exact h2 (by linear_combination h)

/-- `a ≡ 0 (mod g)` iff `g ∣ Q`, for a prime `g ≥ 7`. -/
theorem mirrorA_zmod_zero_iff {q g : ℕ} (hq : 5 ≤ q) (hg : g.Prime) (hg7 : 7 ≤ g) :
    (mirrorA q : ZMod g) = 0 ↔ g ∣ primorial q := by
  have hQ := (thirty_mul_mirrorM hq).symm
  have hQa : primorial q = 2 * mirrorA q := by
    unfold mirrorA
    omega
  have hcop2 : Nat.Coprime g 2 :=
    (Nat.Prime.coprime_iff_not_dvd hg).2 (fun h => absurd (Nat.le_of_dvd (by norm_num) h)
      (by omega))
  rw [ZMod.natCast_eq_zero_iff, hQa]
  exact ⟨fun h => Dvd.dvd.mul_left h 2, fun h => hcop2.dvd_of_dvd_mul_left h⟩

/-- `a ≡ 1 (mod g)` iff `g ∣ Q - 2`, for a prime `g ≥ 7`. -/
theorem mirrorA_zmod_one_iff {q g : ℕ} (hq : 5 ≤ q) (hg : g.Prime) (hg7 : 7 ≤ g) :
    (mirrorA q : ZMod g) = 1 ↔ g ∣ primorial q - 2 := by
  have hQ := (thirty_mul_mirrorM hq).symm
  have ⟨m, hm⟩ := odd_mirrorM hq
  have hQa : primorial q = 2 * mirrorA q := by
    unfold mirrorA
    omega
  have h1a : 1 ≤ mirrorA q := by
    unfold mirrorA
    omega
  have hcop2 : Nat.Coprime g 2 :=
    (Nat.Prime.coprime_iff_not_dvd hg).2 (fun h => absurd (Nat.le_of_dvd (by norm_num) h)
      (by omega))
  have e : ((mirrorA q - 1 : ℕ) : ZMod g) = (mirrorA q : ZMod g) - 1 := by
    push_cast [Nat.cast_sub h1a]
    ring
  rw [← sub_eq_zero, ← e, ZMod.natCast_eq_zero_iff,
    show primorial q - 2 = 2 * (mirrorA q - 1) by omega]
  exact ⟨fun h => Dvd.dvd.mul_left h 2, fun h => hcop2.dvd_of_dvd_mul_left h⟩

/-- `a ≡ -1 (mod g)` iff `g ∣ Q + 2`, for a prime `g ≥ 7`. -/
theorem mirrorA_zmod_neg_one_iff {q g : ℕ} (hq : 5 ≤ q) (hg : g.Prime) (hg7 : 7 ≤ g) :
    (mirrorA q : ZMod g) = -1 ↔ g ∣ primorial q + 2 := by
  have hQ := (thirty_mul_mirrorM hq).symm
  have hQa : primorial q = 2 * mirrorA q := by
    unfold mirrorA
    omega
  have hcop2 : Nat.Coprime g 2 :=
    (Nat.Prime.coprime_iff_not_dvd hg).2 (fun h => absurd (Nat.le_of_dvd (by norm_num) h)
      (by omega))
  have e : ((mirrorA q + 1 : ℕ) : ZMod g) = (mirrorA q : ZMod g) + 1 := by
    push_cast
    ring
  rw [eq_neg_iff_add_eq_zero, ← e, ZMod.natCast_eq_zero_iff,
    show primorial q + 2 = 2 * (mirrorA q + 1) by omega]
  exact ⟨fun h => Dvd.dvd.mul_left h 2, fun h => hcop2.dvd_of_dvd_mul_left h⟩

/-- **(v) Four distinct root classes, every prime `g ≥ 7`.** The classes
`t(a - 1), t(a + 1), t(1 - a), t(-1 - a)` in `ZMod g` (`t = 15⁻¹`) are distinct iff
`g ∤ (Q - 2)·Q·(Q + 2)`. -/
theorem root_classes_nodup_iff {q g : ℕ} (hq : 5 ≤ q) (hg : g.Prime) (hg7 : 7 ≤ g) :
    [(15 : ZMod g)⁻¹ * ((mirrorA q : ZMod g) - 1), (15 : ZMod g)⁻¹ * ((mirrorA q : ZMod g) + 1),
        (15 : ZMod g)⁻¹ * (1 - (mirrorA q : ZMod g)),
        (15 : ZMod g)⁻¹ * (-1 - (mirrorA q : ZMod g))].Nodup ↔
      ¬ g ∣ (primorial q - 2) * primorial q * (primorial q + 2) := by
  have := Fact.mk hg
  rw [classes_nodup_field (mul_inv_cancel₀ (fifteen_ne_zero_zmod hg hg7)) (two_ne_zero_zmod hg7)]
  simp only [ne_eq, mirrorA_zmod_zero_iff hq hg hg7, mirrorA_zmod_one_iff hq hg hg7,
    mirrorA_zmod_neg_one_iff hq hg hg7, Nat.Prime.dvd_mul hg]
  tauto

/-- **(v) Four distinct root classes above `q`.** For a prime `g > q`, the four root classes of
`F` are distinct iff `g ∤ (Q - 2)(Q + 2)`. -/
theorem root_classes_nodup_iff_above {q g : ℕ} (hq : 5 ≤ q) (hg : g.Prime) (hgq : q < g) :
    [(15 : ZMod g)⁻¹ * ((mirrorA q : ZMod g) - 1), (15 : ZMod g)⁻¹ * ((mirrorA q : ZMod g) + 1),
        (15 : ZMod g)⁻¹ * (1 - (mirrorA q : ZMod g)),
        (15 : ZMod g)⁻¹ * (-1 - (mirrorA q : ZMod g))].Nodup ↔
      ¬ g ∣ (primorial q - 2) * (primorial q + 2) := by
  have hg7 := seven_le_of_prime_gt hq hg hgq
  have hgQ := not_dvd_primorial_of_gt hg hgq
  rw [root_classes_nodup_iff hq hg hg7]
  simp only [Nat.Prime.dvd_mul hg]
  tauto

/-- `F` when `a² = 1`: `legF a x = 225 x² (15x - 2)(15x + 2)` (so `0` is a double root). -/
theorem legF_of_sq_eq_one {R : Type*} [CommRing R] (a x : R) (ha : a ^ 2 = 1) :
    legF a x = 225 * x ^ 2 * (15 * x - 2) * (15 * x + 2) := by
  unfold legF
  linear_combination (a ^ 2 - 1 - 2 * (15 * x) ^ 2) * ha

/-- The degenerate roots in a field with `15t = 1`: `225 x² (15x - 2)(15x + 2) = 0` iff
`x = 0`, `x = 2t` or `x = -2t`. -/
theorem degenerate_roots_field {F : Type*} [Field F] {t : F} (ht : 15 * t = 1) (x : F) :
    225 * x ^ 2 * (15 * x - 2) * (15 * x + 2) = 0 ↔ (x = 0 ∨ x = 2 * t ∨ x = -(2 * t)) := by
  have h15 : (15 : F) ≠ 0 := by
    intro h
    rw [h, zero_mul] at ht
    exact zero_ne_one ht
  have h225 : (225 : F) ≠ 0 := by
    have : (225 : F) = 15 * 15 := by norm_num
    rw [this]
    exact mul_ne_zero h15 h15
  constructor
  · intro h
    rcases mul_eq_zero.1 h with h | h
    · rcases mul_eq_zero.1 h with h | h
      · rcases mul_eq_zero.1 h with h | h
        · exact absurd h h225
        · exact Or.inl (pow_eq_zero_iff (by norm_num) |>.1 h)
      · exact Or.inr (Or.inl (by linear_combination t * h - x * ht))
    · exact Or.inr (Or.inr (by linear_combination t * h - x * ht))
  · rintro (h | h | h)
    · rw [h]
      ring
    · have e : 15 * x - 2 = 0 := by linear_combination 15 * h + 2 * ht
      rw [e]
      ring
    · have e : 15 * x + 2 = 0 := by linear_combination 15 * h - 2 * ht
      rw [e]
      ring

/-- **(v) Three classes when `g ∣ (Q - 2)(Q + 2)`.** For a prime `g ≥ 7` dividing
`(Q - 2)(Q + 2)`, `F(d) ≡ 225 d² (15d - 2)(15d + 2) (mod g)`, so `g ∣ F(d)` iff
`d ≡ 0, 2t` or `-2t`, and these three classes are distinct. -/
theorem three_classes {q d g : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q)
    (hg : g.Prime) (hg7 : 7 ≤ g) (hdiv : g ∣ (primorial q - 2) * (primorial q + 2)) :
    (((30 * mirrorLo q d - 1) * (30 * mirrorLo q d + 1) *
        (30 * mirrorHi q d - 1) * (30 * mirrorHi q d + 1) : ℕ) : ZMod g) =
        225 * (d : ZMod g) ^ 2 * (15 * (d : ZMod g) - 2) * (15 * (d : ZMod g) + 2) ∧
    (g ∣ (30 * mirrorLo q d - 1) * (30 * mirrorLo q d + 1) *
        (30 * mirrorHi q d - 1) * (30 * mirrorHi q d + 1) ↔
      ((d : ZMod g) = 0 ∨ (d : ZMod g) = 2 * (15 : ZMod g)⁻¹ ∨
        (d : ZMod g) = -(2 * (15 : ZMod g)⁻¹))) ∧
    [0, 2 * (15 : ZMod g)⁻¹, -(2 * (15 : ZMod g)⁻¹)].Nodup := by
  have := Fact.mk hg
  have ht : 15 * (15 : ZMod g)⁻¹ = 1 := mul_inv_cancel₀ (fifteen_ne_zero_zmod hg hg7)
  have h2 := two_ne_zero_zmod (g := g) hg7
  have ha : (mirrorA q : ZMod g) ^ 2 = 1 := by
    rcases (Nat.Prime.dvd_mul hg).1 hdiv with h | h
    · rw [(mirrorA_zmod_one_iff hq hg hg7).2 h]
      ring
    · rw [(mirrorA_zmod_neg_one_iff hq hg hg7).2 h]
      ring
  have hF := legF_cast hq hd hdM g
  rw [legF_of_sq_eq_one _ _ ha] at hF
  refine ⟨hF, ?_, ?_⟩
  · rw [← ZMod.natCast_eq_zero_iff, hF]
    exact degenerate_roots_field ht _
  · have ht0 : (15 : ZMod g)⁻¹ ≠ 0 := inv_ne_zero (fifteen_ne_zero_zmod hg hg7)
    have h2t : 2 * (15 : ZMod g)⁻¹ ≠ 0 := mul_ne_zero h2 ht0
    rw [nodup_three]
    refine ⟨fun h => h2t h.symm, fun h => h2t (neg_eq_zero.1 h.symm), fun h => ?_⟩
    have e : 2 * (2 * (15 : ZMod g)⁻¹) = 0 := by linear_combination h
    exact h2t ((mul_eq_zero.1 e).resolve_left h2)

/-! ### (vi) Empty band -/

/-- **(vi) Band emptiness.** For `g ≥ 1`, no multiple `d ≥ 1` of `g` has `15d ≤ ρ - g²` iff
`ρ - g² < 15g` (the first multiple, `g` itself, falls outside). -/
theorem band_empty_iff (ρ g : ℕ) (hg : 1 ≤ g) :
    (¬ ∃ d, 1 ≤ d ∧ 15 * d ≤ ρ - g ^ 2 ∧ g ∣ d) ↔ ρ - g ^ 2 < 15 * g := by
  constructor
  · intro h
    by_contra hc
    exact h ⟨g, hg, by omega, dvd_rfl⟩
  · rintro hlt ⟨d, hd1, hdle, hdd⟩
    have := Nat.le_of_dvd hd1 hdd
    omega

/-- **(vi) Band emptiness, odd multiples.** For odd `g`, no odd multiple `d ≥ 1` of `g` has
`15d ≤ ρ - g²` iff `ρ - g² < 15g`. -/
theorem band_empty_odd_iff (ρ g : ℕ) (hg : Odd g) :
    (¬ ∃ d, Odd d ∧ 1 ≤ d ∧ 15 * d ≤ ρ - g ^ 2 ∧ g ∣ d) ↔ ρ - g ^ 2 < 15 * g := by
  have hg1 : 1 ≤ g := by
    have ⟨k, hk⟩ := hg
    omega
  constructor
  · intro h
    by_contra hc
    exact h ⟨g, hg, hg1, by omega, dvd_rfl⟩
  · rintro hlt ⟨d, _, hd1, hdle, hdd⟩
    have := Nat.le_of_dvd hd1 hdd
    omega

/-- **(vi) The band as `[1, w_g]`.** With `w_g = (ρ - g²)/15` (floor), a multiple `d ≥ 1` of `g`
lies in `[1, w_g]` iff `15d ≤ ρ - g²`. -/
theorem band_w_iff (ρ g : ℕ) :
    (∃ d, 1 ≤ d ∧ d ≤ (ρ - g ^ 2) / 15 ∧ g ∣ d) ↔ (∃ d, 1 ≤ d ∧ 15 * d ≤ ρ - g ^ 2 ∧ g ∣ d) := by
  constructor <;> rintro ⟨d, h1, h2, h3⟩ <;> exact ⟨d, h1, by omega, h3⟩

/-- The facts `Q = 2a`, `ρ = a + 1`, `a` odd, `a ≥ 15` (for `q ≥ 5`). -/
theorem mirrorA_facts {q : ℕ} (hq : 5 ≤ q) :
    primorial q = 2 * mirrorA q ∧ mirrorRho q = mirrorA q + 1 ∧ mirrorA q % 2 = 1 ∧
      15 ≤ mirrorA q := by
  have hQ := (thirty_mul_mirrorM hq).symm
  have ⟨m, hm⟩ := odd_mirrorM hq
  unfold mirrorA mirrorRho
  omega

/-- **(vi) Empty band, `g ∣ Q - 2`.** For odd `g ≥ 3` with `g ∣ Q - 2` and `g² ≤ ρ`, the band
`15d ≤ ρ - g²` holds no multiple of `g` iff `a - 1 = g(g + e)` for an odd `e` with
`1 ≤ e ≤ 13`. -/
theorem band_empty_minus {q g : ℕ} (hq : 5 ≤ q) (hg : Odd g) (hg3 : 3 ≤ g)
    (hgQ : g ∣ primorial q - 2) (hU : g ^ 2 ≤ mirrorRho q) :
    (¬ ∃ d, 1 ≤ d ∧ 15 * d ≤ mirrorRho q - g ^ 2 ∧ g ∣ d) ↔
      ∃ e, Odd e ∧ 1 ≤ e ∧ e ≤ 13 ∧ mirrorA q - 1 = g * (g + e) := by
  rw [band_empty_iff _ _ (by omega)]
  obtain ⟨hQa, hR, hao, ha15⟩ := mirrorA_facts hq
  have hcop2 : Nat.Coprime g 2 := hg.coprime_two_right
  have ha : g ∣ mirrorA q - 1 := by
    rw [show primorial q - 2 = 2 * (mirrorA q - 1) by omega] at hgQ
    exact hcop2.dvd_of_dvd_mul_left hgQ
  obtain ⟨m, hm⟩ := ha
  constructor
  · intro hlt
    have hU' : g ^ 2 ≤ g * m + 2 := by omega
    have hlt' : g * m + 2 < g ^ 2 + 15 * g := by omega
    have hgm : g ≤ m := by
      by_contra hc
      have hc' : m + 1 ≤ g := by omega
      have := Nat.mul_le_mul_left g hc'
      nlinarith
    have hm14 : m ≤ g + 14 := by
      by_contra hc
      have hc' : g + 15 ≤ m := by omega
      have := Nat.mul_le_mul_left g hc'
      nlinarith
    have hmev : Even m := by
      have hev : Even (g * m) := by
        rw [← hm]
        exact Nat.even_iff.2 (by omega)
      rcases Nat.even_mul.1 hev with h | h
      · exact absurd h (Nat.not_even_iff_odd.2 hg)
      · exact h
    obtain ⟨k, hk⟩ := hmev
    obtain ⟨j, hj⟩ := hg
    refine ⟨m - g, ⟨k - j - 1, by omega⟩, by omega, by omega, ?_⟩
    rw [Nat.add_sub_cancel' hgm]
    exact hm
  · rintro ⟨e, _, _, he13, hae⟩
    have h1 : g * (g + e) = g ^ 2 + g * e := by ring
    have h2 : g * e ≤ g * 13 := Nat.mul_le_mul_left g he13
    omega

/-- **(vi) Empty band, `g ∣ Q + 2`.** For odd `g ≥ 3` with `g ∣ Q + 2` and `g² ≤ ρ`, the band
`15d ≤ ρ - g²` holds no multiple of `g` iff `a + 1 = g(g + e)` for an odd `e` with
`1 ≤ e ≤ 13`. -/
theorem band_empty_plus {q g : ℕ} (hq : 5 ≤ q) (hg : Odd g) (hg3 : 3 ≤ g)
    (hgQ : g ∣ primorial q + 2) (hU : g ^ 2 ≤ mirrorRho q) :
    (¬ ∃ d, 1 ≤ d ∧ 15 * d ≤ mirrorRho q - g ^ 2 ∧ g ∣ d) ↔
      ∃ e, Odd e ∧ 1 ≤ e ∧ e ≤ 13 ∧ mirrorA q + 1 = g * (g + e) := by
  rw [band_empty_iff _ _ (by omega)]
  obtain ⟨hQa, hR, hao, ha15⟩ := mirrorA_facts hq
  have hcop2 : Nat.Coprime g 2 := hg.coprime_two_right
  have ha : g ∣ mirrorA q + 1 := by
    rw [show primorial q + 2 = 2 * (mirrorA q + 1) by omega] at hgQ
    exact hcop2.dvd_of_dvd_mul_left hgQ
  obtain ⟨m, hm⟩ := ha
  constructor
  · intro hlt
    have hU' : g ^ 2 ≤ g * m := by omega
    have hlt' : g * m < g ^ 2 + 15 * g := by omega
    have hgm : g ≤ m := by
      by_contra hc
      have hc' : m + 1 ≤ g := by omega
      have := Nat.mul_le_mul_left g hc'
      nlinarith
    have hm14 : m ≤ g + 14 := by
      by_contra hc
      have hc' : g + 15 ≤ m := by omega
      have := Nat.mul_le_mul_left g hc'
      nlinarith
    have hmev : Even m := by
      have hev : Even (g * m) := by
        rw [← hm]
        exact Nat.even_iff.2 (by omega)
      rcases Nat.even_mul.1 hev with h | h
      · exact absurd h (Nat.not_even_iff_odd.2 hg)
      · exact h
    obtain ⟨k, hk⟩ := hmev
    obtain ⟨j, hj⟩ := hg
    refine ⟨m - g, ⟨k - j - 1, by omega⟩, by omega, by omega, ?_⟩
    rw [Nat.add_sub_cancel' hgm]
    exact hm
  · rintro ⟨e, _, _, he13, hae⟩
    have h1 : g * (g + e) = g ^ 2 + g * e := by ring
    have h2 : g * e ≤ g * 13 := Nat.mul_le_mul_left g he13
    omega

/-- **(vi) Square form, `Q - 2`.** `a - 1 = g(g + e)` iff `2Q - 4 + e² = (2g + e)²`. -/
theorem band_square_minus {q : ℕ} (hq : 5 ≤ q) (g e : ℕ) :
    mirrorA q - 1 = g * (g + e) ↔ 2 * primorial q - 4 + e ^ 2 = (2 * g + e) ^ 2 := by
  obtain ⟨hQa, _, _, ha15⟩ := mirrorA_facts hq
  have hsq : (2 * g + e) ^ 2 = 4 * (g * (g + e)) + e ^ 2 := by ring
  rw [hsq]
  omega

/-- **(vi) Square form, `Q + 2`.** `a + 1 = g(g + e)` iff `2Q + 4 + e² = (2g + e)²`. -/
theorem band_square_plus {q : ℕ} (hq : 5 ≤ q) (g e : ℕ) :
    mirrorA q + 1 = g * (g + e) ↔ 2 * primorial q + 4 + e ^ 2 = (2 * g + e) ^ 2 := by
  obtain ⟨hQa, _, _, ha15⟩ := mirrorA_facts hq
  have hsq : (2 * g + e) ^ 2 = 4 * (g * (g + e)) + e ^ 2 := by ring
  rw [hsq]
  omega

/-- `n² ≡ n (mod 2)`. -/
theorem sq_mod_two (n : ℕ) : n ^ 2 % 2 = n % 2 := by
  rw [Nat.pow_mod]
  rcases Nat.mod_two_eq_zero_or_one n with h | h <;> rw [h]

/-- **(vi) Square criterion, `Q - 2`.** `2Q - 4 + e²` is a square iff `a - 1 = g(g + e)` for
some `g` (then the root is `2g + e`). -/
theorem band_square_exists_minus {q : ℕ} (hq : 5 ≤ q) (e : ℕ) :
    (∃ n, 2 * primorial q - 4 + e ^ 2 = n ^ 2) ↔ ∃ g, mirrorA q - 1 = g * (g + e) := by
  obtain ⟨hQa, _, _, ha15⟩ := mirrorA_facts hq
  constructor
  · rintro ⟨n, hn⟩
    have hen : e ≤ n := by
      by_contra hc
      have := Nat.pow_lt_pow_left (show n < e by omega) (show 2 ≠ 0 by norm_num)
      omega
    have hn2 := sq_mod_two n
    have he2 := sq_mod_two e
    refine ⟨(n - e) / 2, (band_square_minus hq _ e).2 ?_⟩
    have : 2 * ((n - e) / 2) + e = n := by omega
    rw [this]
    exact hn
  · rintro ⟨g, hg⟩
    exact ⟨2 * g + e, (band_square_minus hq g e).1 hg⟩

/-- **(vi) Square criterion, `Q + 2`.** `2Q + 4 + e²` is a square iff `a + 1 = g(g + e)` for
some `g`. -/
theorem band_square_exists_plus {q : ℕ} (hq : 5 ≤ q) (e : ℕ) :
    (∃ n, 2 * primorial q + 4 + e ^ 2 = n ^ 2) ↔ ∃ g, mirrorA q + 1 = g * (g + e) := by
  obtain ⟨hQa, _, _, ha15⟩ := mirrorA_facts hq
  constructor
  · rintro ⟨n, hn⟩
    have hen : e ≤ n := by
      by_contra hc
      have := Nat.pow_lt_pow_left (show n < e by omega) (show 2 ≠ 0 by norm_num)
      omega
    have hn2 := sq_mod_two n
    have he2 := sq_mod_two e
    refine ⟨(n - e) / 2, (band_square_plus hq _ e).2 ?_⟩
    have : 2 * ((n - e) / 2) + e = n := by omega
    rw [this]
    exact hn
  · rintro ⟨g, hg⟩
    exact ⟨2 * g + e, (band_square_plus hq g e).1 hg⟩

/-! ### (vii) Range and doubly deleted pairs -/

/-- A prime dividing a leg `30j ± 1` (with `j ≥ 1`) is at least `7`. -/
theorem prime_dvd_leg_ge_seven {m j L : ℕ} (hm : m.Prime) (hj : 1 ≤ j)
    (hL : L = 30 * j - 1 ∨ L = 30 * j + 1) (hmL : m ∣ L) : 7 ≤ m := by
  by_contra h7
  have h7' : m < 7 := by omega
  have h30 : m ∣ 30 := by
    interval_cases m <;> first | (norm_num; done) | exact absurd hm (by norm_num)
  have h30j : m ∣ 30 * j := Dvd.dvd.mul_right h30 j
  have h1 : m ∣ 1 := by
    rcases hL with rfl | rfl
    · have := Nat.dvd_sub h30j hmL
      rwa [show 30 * j - (30 * j - 1) = 1 by omega] at this
    · have := Nat.dvd_sub hmL h30j
      rwa [show 30 * j + 1 - 30 * j = 1 by omega] at this
  exact hm.one_lt.ne' (Nat.dvd_one.1 h1)

/-- A copy `j ≥ 1` with both legs prime is not deleted: a prime gear dividing a prime leg is the
leg itself, and the leg's square exceeds `30j + 1`. -/
theorem not_deleted_of_twin {q j : ℕ} (hj : 1 ≤ j) (ht : TwinCopy j) : ¬ Deleted q j := by
  rintro ⟨g, hg, _, hs, ha⟩
  unfold Acts at ha
  have hsq : g ^ 2 = g * g := sq g
  rcases (strikesCopy_iff_leg hg).1 hs with h | h
  · have hgL : g = 30 * j - 1 := (Nat.prime_dvd_prime_iff_eq hg ht.1).1 h
    have h29 : 29 ≤ g := by omega
    have := Nat.mul_le_mul_right g h29
    omega
  · have hgL : g = 30 * j + 1 := (Nat.prime_dvd_prime_iff_eq hg ht.2).1 h
    have h29 : 29 ≤ g := by omega
    have := Nat.mul_le_mul_right g h29
    omega

/-- A surviving copy `j ≥ 1` with a composite leg is deleted: the least prime factor `m` of that
leg has `7 ≤ m`, `m > q` (survival) and `m² ≤ leg ≤ 30j + 1`. -/
theorem deleted_of_not_twin {q j : ℕ} (hj : 1 ≤ j) (hS : Survives q j) (ht : ¬ TwinCopy j) :
    Deleted q j := by
  unfold TwinCopy at ht
  have key : ∀ L, (L = 30 * j - 1 ∨ L = 30 * j + 1) → ¬ L.Prime → Deleted q j := by
    intro L hL hLp
    have hL1 : L ≠ 1 := by omega
    have hL0 : 0 < L := by omega
    have hm : L.minFac.Prime := Nat.minFac_prime hL1
    have hmL : L.minFac ∣ L := Nat.minFac_dvd L
    have hsq : L.minFac ^ 2 ≤ L := Nat.minFac_sq_le_self hL0 hLp
    have h7 := prime_dvd_leg_ge_seven hm hj hL hmL
    have hstr : StrikesCopy L.minFac j := by
      have hLdvd : L ∣ (30 * j - 1) * (30 * j + 1) := by
        rcases hL with h | h
        · rw [h]
          exact dvd_mul_right _ _
        · rw [h]
          exact dvd_mul_left _ _
      exact dvd_trans hmL hLdvd
    have hmq : q < L.minFac := by
      by_contra hc
      exact hS _ hm h7 (by omega) hstr
    refine ⟨L.minFac, hm, hmq, hstr, ?_⟩
    unfold Acts
    omega
  rcases not_and_or.1 ht with h | h
  · exact key _ (Or.inl rfl) h
  · exact key _ (Or.inr rfl) h

/-- **(vii) Deleted survivors.** A surviving copy `j ≥ 1` is deleted (some prime `g > q` strikes
and acts at it) iff some leg is not prime. -/
theorem deleted_iff_not_twin {q j : ℕ} (hj : 1 ≤ j) (hS : Survives q j) :
    Deleted q j ↔ ¬ TwinCopy j :=
  ⟨fun hD ht => not_deleted_of_twin hj ht hD, deleted_of_not_twin hj hS⟩

/-- **(vii) Every copy of `[1, M - 1]` lies in a mirror pair.** For `1 ≤ j < M` there is an odd
`d < M` with `j = s` or `j = s'` (`d = |M - 2j|`). -/
theorem mirror_cover {q j : ℕ} (hq : 5 ≤ q) (hj1 : 1 ≤ j) (hjM : j < mirrorM q) :
    ∃ d, Odd d ∧ d < mirrorM q ∧ (j = mirrorLo q d ∨ j = mirrorHi q d) := by
  have ⟨m, hm⟩ := odd_mirrorM hq
  by_cases h : 2 * j < mirrorM q
  · refine ⟨mirrorM q - 2 * j, ⟨m - j, by omega⟩, by omega, Or.inl ?_⟩
    unfold mirrorLo
    omega
  · refine ⟨2 * j - mirrorM q, ⟨j - m - 1, by omega⟩, by omega, Or.inr ?_⟩
    unfold mirrorHi
    omega

/-- **(vii) The two members of a pair differ.** `s ≠ s'`. -/
theorem mirror_ne {q d : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q) :
    mirrorLo q d ≠ mirrorHi q d := by
  obtain ⟨_, _, _, _, hH, _, _, hdo⟩ := mirror_data hq hd hdM
  omega

/-- **(vii) No fixed point.** Since `M` is odd, `j ↦ M - j` has no fixed point: `2j ≠ M`. -/
theorem mirror_no_fixed {q : ℕ} (hq : 5 ≤ q) (j : ℕ) : 2 * j ≠ mirrorM q := by
  have ⟨m, hm⟩ := odd_mirrorM hq
  omega

/-- **(vii) Range on copies ⇔ a revealed survivor.** The copy form of the range statement holds
iff some survivor `j ∈ [1, M - 1]` has both legs prime. -/
theorem window_iff_revealed {q : ℕ} (hq : 5 ≤ q) :
    RangeCopyWindow q ↔ ∃ j, 1 ≤ j ∧ j < mirrorM q ∧ Survives q j ∧ TwinCopy j := by
  have hQ := (thirty_mul_mirrorM hq).symm
  constructor
  · rintro ⟨j, hj, hqj, hjQ, hp1, hp2⟩
    refine ⟨j, hj, by omega, ?_, hp1, hp2⟩
    intro p hp _ hpq hstr
    rcases (strikesCopy_iff_leg hp).1 hstr with h | h
    · have := (Nat.prime_dvd_prime_iff_eq hp hp1).1 h
      omega
    · have := (Nat.prime_dvd_prime_iff_eq hp hp2).1 h
      omega
  · rintro ⟨j, hj, hjM, hS, hp1, hp2⟩
    refine ⟨j, hj, ?_, by omega, hp1, hp2⟩
    by_contra hc
    have h7 : 7 ≤ 30 * j - 1 := by omega
    exact hS (30 * j - 1) hp1 h7 (by omega) ((strikesCopy_iff_leg hp1).2 (Or.inl dvd_rfl))

/-- **(vii) Range ⇔ some survivor mirror pair is not doubly deleted.** For every `q ≥ 5`, the
copy form of the range statement holds iff some odd `d < M` gives a mirror pair of survivors
whose two members are not both deleted. -/
theorem range_iff_pair {q : ℕ} (hq : 5 ≤ q) :
    RangeCopyWindow q ↔
      ∃ d, Odd d ∧ d < mirrorM q ∧ Survives q (mirrorLo q d) ∧ Survives q (mirrorHi q d) ∧
        ¬ (Deleted q (mirrorLo q d) ∧ Deleted q (mirrorHi q d)) := by
  rw [window_iff_revealed hq]
  constructor
  · rintro ⟨j, hj, hjM, hS, ht⟩
    obtain ⟨d, hd, hdM, hjd⟩ := mirror_cover hq hj hjM
    have hpair := survives_pair hq hd hdM
    refine ⟨d, hd, hdM, ?_, ?_, ?_⟩
    · rcases hjd with rfl | rfl
      · exact hS
      · exact hpair.2 hS
    · rcases hjd with rfl | rfl
      · exact hpair.1 hS
      · exact hS
    · rintro ⟨h1, h2⟩
      rcases hjd with rfl | rfl
      · exact not_deleted_of_twin hj ht h1
      · exact not_deleted_of_twin hj ht h2
  · rintro ⟨d, hd, hdM, hS1, hS2, hnd⟩
    obtain ⟨_, _, _, h2, hH, hs1, _, _⟩ := mirror_data hq hd hdM
    rcases not_and_or.1 hnd with h | h
    · refine ⟨mirrorLo q d, hs1, by omega, hS1, ?_⟩
      by_contra ht
      exact h (deleted_of_not_twin hs1 hS1 ht)
    · refine ⟨mirrorHi q d, by omega, by omega, hS2, ?_⟩
      by_contra ht
      exact h (deleted_of_not_twin (by omega) hS2 ht)

/-- **(vii) Deletion of the low member, class form.** `s` is deleted iff some prime `g > q` has
`d ≡ M ∓ t_g (mod g)` and `15d ≤ ρ - g²`. -/
theorem deleted_lo_iff {q d : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q) :
    Deleted q (mirrorLo q d) ↔
      ∃ g, g.Prime ∧ q < g ∧
        ((d : ZMod g) = (mirrorM q : ZMod g) - (15 : ZMod g)⁻¹ ∨
          (d : ZMod g) = (mirrorM q : ZMod g) + (15 : ZMod g)⁻¹) ∧
        15 * d ≤ mirrorRho q - g ^ 2 := by
  constructor
  · rintro ⟨g, hg, hgq, hs, ha⟩
    exact ⟨g, hg, hgq,
      (strikes_lo_iff_class hq hd hdM hg (seven_le_of_prime_gt hq hg hgq)).1 hs,
      (acts_lo_iff hq hd hdM).1 ha⟩
  · rintro ⟨g, hg, hgq, hs, ha⟩
    exact ⟨g, hg, hgq,
      (strikes_lo_iff_class hq hd hdM hg (seven_le_of_prime_gt hq hg hgq)).2 hs,
      (acts_lo_iff hq hd hdM).2 ha⟩

/-- **(vii) Deletion of the high member, class form.** `s'` is deleted iff some prime `g > q` has
`d ≡ -M ± t_g (mod g)` and `g² - ρ ≤ 15d`. -/
theorem deleted_hi_iff {q d : ℕ} (hq : 5 ≤ q) (hd : Odd d) (hdM : d < mirrorM q) :
    Deleted q (mirrorHi q d) ↔
      ∃ g, g.Prime ∧ q < g ∧
        ((d : ZMod g) = -(mirrorM q : ZMod g) + (15 : ZMod g)⁻¹ ∨
          (d : ZMod g) = -(mirrorM q : ZMod g) - (15 : ZMod g)⁻¹) ∧
        g ^ 2 - mirrorRho q ≤ 15 * d := by
  constructor
  · rintro ⟨g, hg, hgq, hs, ha⟩
    exact ⟨g, hg, hgq,
      (strikes_hi_iff_class hq hd hdM hg (seven_le_of_prime_gt hq hg hgq)).1 hs,
      (acts_hi_iff hq hd hdM).1 ha⟩
  · rintro ⟨g, hg, hgq, hs, ha⟩
    exact ⟨g, hg, hgq,
      (strikes_hi_iff_class hq hd hdM hg (seven_le_of_prime_gt hq hg hgq)).2 hs,
      (acts_hi_iff hq hd hdM).2 ha⟩

/-- **(vii) The copy form gives the range statement.** A revealed copy in the window is a twin
pair `(30j - 1, 30j + 1)` with `q < 30j - 1` and `30j + 1 ≤ q#`. -/
theorem rangeCopyWindow_rangeStatement {q : ℕ} (h : RangeCopyWindow q) : RangeStatement q :=
  copy_range_implies_rangeStatement h

end RangeLine
