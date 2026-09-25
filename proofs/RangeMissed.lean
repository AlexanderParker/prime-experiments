import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Tactic

/-!
# Range missed copy: the copy just above a gear's square, and why no gear strikes it early

The copy `J` (for `J ≥ 1`) is the pair of numbers `30 J - 1` and `30 J + 1`, its two legs.
Here `g` is a prime gear with `g ≥ 7`.

This file proves:

* `sq_mod30_cases`: for a prime `g ≥ 7`, `g² mod 30` is `1` or `19`
  ("case 1" and "case 19"); `sq_mod30_exactly_one`: exactly one of the two holds.
* `missed_copy_legs` (with `missed_copy_legs_case1`, `missed_copy_legs_case19`):
  in case 1, `30` divides `g² + 29`, and the copy `J = (g² + 29) / 30` has legs
  `30 J - 1 = g² + 28` and `30 J + 1 = g² + 30`;
  in case 19, `30` divides `g² + 11`, and the copy `J = (g² + 11) / 30` has legs
  `30 J - 1 = g² + 10` and `30 J + 1 = g² + 12`.
  (These two facts need only the value of `g² mod 30`, not primality.)
* `gear_misses_own_copy`: the gear `g` divides neither leg of that copy:
  not `g² + 28`, not `g² + 30` in case 1; not `g² + 10`, not `g² + 12` in case 19.
  (Otherwise `g` would divide `28`, `30`, `10` or `12`; the only prime `≥ 7` among their
  divisors is `7`, and `7² = 49 ≡ 19 (mod 30)` puts `7` in case 19, where `7` divides
  neither `10` nor `12`.)
* `larger_gear_sq`, `no_larger_gear_acts`: for primes `g ≥ 7` and `h > g`,
  `h² > g² + 30` (since `h ≥ g + 2` and `4 g + 4 > 30`); hence if `h` divides a leg `L`
  with `L ≤ g² + 30` and `h < L`, then the cofactor `L / h` is below `h`, so `h` is never
  the smaller factor of such a leg.
-/

namespace RangeLine

/-- A prime `g ≥ 7` is odd and is divisible by neither `3` nor `5`. -/
theorem prime_ge7_mod (g : ℕ) (hg : g.Prime) (h7 : 7 ≤ g) :
    g % 2 = 1 ∧ g % 3 ≠ 0 ∧ g % 5 ≠ 0 := by
  have key : ∀ p, p ∣ g → p ≠ 1 → p = g := fun p hp h1 =>
    (hg.eq_one_or_self_of_dvd p hp).resolve_left h1
  refine ⟨?_, ?_, ?_⟩
  · by_contra h
    have h2 : 2 ∣ g := by omega
    have := key 2 h2 (by norm_num)
    omega
  · intro h
    have := key 3 (Nat.dvd_of_mod_eq_zero h) (by norm_num)
    omega
  · intro h
    have := key 5 (Nat.dvd_of_mod_eq_zero h) (by norm_num)
    omega

/-- (a) For a prime `g ≥ 7`, the square `g²` is `1` or `19` modulo `30`. -/
theorem sq_mod30_cases (g : ℕ) (hg : g.Prime) (h7 : 7 ≤ g) :
    g ^ 2 % 30 = 1 ∨ g ^ 2 % 30 = 19 := by
  obtain ⟨h2, h3, h5⟩ := prime_ge7_mod g hg h7
  have hsq : g ^ 2 % 30 = (g % 30) ^ 2 % 30 := Nat.pow_mod g 2 30
  have hr : g % 30 < 30 := Nat.mod_lt _ (by norm_num)
  have h2' : (g % 30) % 2 = 1 := by omega
  have h3' : (g % 30) % 3 ≠ 0 := by omega
  have h5' : (g % 30) % 5 ≠ 0 := by omega
  rw [hsq]
  generalize g % 30 = r at hr h2' h3' h5' ⊢
  interval_cases r <;> simp_all

/-- (a') For a prime `g ≥ 7`, exactly one of case 1 (`g² % 30 = 1`) and
case 19 (`g² % 30 = 19`) holds. -/
theorem sq_mod30_exactly_one (g : ℕ) (hg : g.Prime) (h7 : 7 ≤ g) :
    (g ^ 2 % 30 = 1 ∧ ¬ g ^ 2 % 30 = 19) ∨ (g ^ 2 % 30 = 19 ∧ ¬ g ^ 2 % 30 = 1) := by
  rcases sq_mod30_cases g hg h7 with h | h
  · exact Or.inl ⟨h, by omega⟩
  · exact Or.inr ⟨h, by omega⟩

/-- (b, case 1) If `g² % 30 = 1`, then `30 ∣ g² + 29`, and the copy `J = (g² + 29) / 30`
has legs `30 J - 1 = g² + 28` and `30 J + 1 = g² + 30`. -/
theorem missed_copy_legs_case1 (g : ℕ) (h1 : g ^ 2 % 30 = 1) :
    30 ∣ g ^ 2 + 29 ∧ 30 * ((g ^ 2 + 29) / 30) - 1 = g ^ 2 + 28 ∧
      30 * ((g ^ 2 + 29) / 30) + 1 = g ^ 2 + 30 := by
  generalize g ^ 2 = s at h1 ⊢
  omega

/-- (b, case 19) If `g² % 30 = 19`, then `30 ∣ g² + 11`, and the copy `J = (g² + 11) / 30`
has legs `30 J - 1 = g² + 10` and `30 J + 1 = g² + 12`. -/
theorem missed_copy_legs_case19 (g : ℕ) (h19 : g ^ 2 % 30 = 19) :
    30 ∣ g ^ 2 + 11 ∧ 30 * ((g ^ 2 + 11) / 30) - 1 = g ^ 2 + 10 ∧
      30 * ((g ^ 2 + 11) / 30) + 1 = g ^ 2 + 12 := by
  generalize g ^ 2 = s at h19 ⊢
  omega

/-- (b) The missed copy and its legs, both cases together: in case 1 the copy
`J = (g² + 29) / 30` has legs `g² + 28`, `g² + 30`; in case 19 the copy
`J = (g² + 11) / 30` has legs `g² + 10`, `g² + 12`. -/
theorem missed_copy_legs (g : ℕ) :
    (g ^ 2 % 30 = 1 →
      30 ∣ g ^ 2 + 29 ∧ 30 * ((g ^ 2 + 29) / 30) - 1 = g ^ 2 + 28 ∧
        30 * ((g ^ 2 + 29) / 30) + 1 = g ^ 2 + 30) ∧
    (g ^ 2 % 30 = 19 →
      30 ∣ g ^ 2 + 11 ∧ 30 * ((g ^ 2 + 11) / 30) - 1 = g ^ 2 + 10 ∧
        30 * ((g ^ 2 + 11) / 30) + 1 = g ^ 2 + 12) :=
  ⟨missed_copy_legs_case1 g, missed_copy_legs_case19 g⟩

/-- If `g` divides `g² + c`, then `g` divides `c`. -/
theorem dvd_of_dvd_sq_add (g c : ℕ) (h : g ∣ g ^ 2 + c) : g ∣ c :=
  (Nat.dvd_add_right (dvd_pow_self g two_ne_zero)).mp h

/-- (c) A prime gear `g ≥ 7` misses its own copy: in case 1 it divides neither
`g² + 28` nor `g² + 30`; in case 19 it divides neither `g² + 10` nor `g² + 12`. -/
theorem gear_misses_own_copy (g : ℕ) (hg : g.Prime) (h7 : 7 ≤ g) :
    (g ^ 2 % 30 = 1 → ¬ g ∣ g ^ 2 + 28 ∧ ¬ g ∣ g ^ 2 + 30) ∧
    (g ^ 2 % 30 = 19 → ¬ g ∣ g ^ 2 + 10 ∧ ¬ g ∣ g ^ 2 + 12) := by
  refine ⟨fun h1 => ⟨fun h => ?_, fun h => ?_⟩, fun h19 => ⟨fun h => ?_, fun h => ?_⟩⟩
  · have hd := dvd_of_dvd_sq_add g 28 h
    have hle := Nat.le_of_dvd (by norm_num) hd
    interval_cases g <;> omega
  · have hd := dvd_of_dvd_sq_add g 30 h
    have hle := Nat.le_of_dvd (by norm_num) hd
    interval_cases g <;> omega
  · have hd := dvd_of_dvd_sq_add g 10 h
    have hle := Nat.le_of_dvd (by norm_num) hd
    interval_cases g <;> omega
  · have hd := dvd_of_dvd_sq_add g 12 h
    have hle := Nat.le_of_dvd (by norm_num) hd
    interval_cases g <;> omega

/-- (d, square step) For primes `g ≥ 7` and `h > g`: `h ≥ g + 2` (both odd), so
`h² ≥ g² + 4 g + 4 > g² + 30`. -/
theorem larger_gear_sq (g h : ℕ) (hg : g.Prime) (hh : h.Prime) (h7 : 7 ≤ g) (hgh : g < h) :
    g ^ 2 + 30 < h ^ 2 := by
  have ho := (prime_ge7_mod g hg h7).1
  have ho' := (prime_ge7_mod h hh (by omega)).1
  have h2 : g + 2 ≤ h := by omega
  nlinarith

/-- (d) No larger gear acts on a leg at or below `g² + 30`: for primes `g ≥ 7` and `h > g`,
`h² > g² + 30`, and if `h` divides a leg `L` with `L ≤ g² + 30` and `h < L`, then
`L / h < h`, so `h` is not the smaller factor of `L`. -/
theorem no_larger_gear_acts (g h : ℕ) (hg : g.Prime) (hh : h.Prime) (h7 : 7 ≤ g)
    (hgh : g < h) :
    g ^ 2 + 30 < h ^ 2 ∧ ∀ L, h ∣ L → L ≤ g ^ 2 + 30 → h < L → L / h < h := by
  have hsq := larger_gear_sq g h hg hh h7 hgh
  refine ⟨hsq, ?_⟩
  intro L hd hL _
  by_contra hc
  have hc : h ≤ L / h := Nat.le_of_not_lt hc
  obtain ⟨k, rfl⟩ := hd
  have hpos : 0 < h := hh.pos
  rw [Nat.mul_div_cancel_left k hpos] at hc
  nlinarith

end RangeLine
