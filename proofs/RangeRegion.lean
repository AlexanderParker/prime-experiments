import Mathlib.Data.Nat.Prime.Basic
import Mathlib.NumberTheory.Primorial
import Mathlib.Tactic
import RangeCopies

/-!
# The region law, stated with the least prime factor

For `j ≥ 1` the copy `j` is the pair of legs `30 j - 1` and `30 j + 1`, and their product is
`900 j² - 1` (`RangeCopies.copy_product`). Write `P j` for the least prime factor of that product,
`P j = Nat.minFac (900 j² - 1)`.

This file proves:

* `P_eq_minFac_legs`: `P j` is the least prime factor of the product of the two legs.
* `revealed_iff_minFac` (a): for `j ≥ 1`, both legs of copy `j` are prime exactly when
  `30 j + 1 < (P j)²`.
  - If both legs are prime, `P j` is one of the two legs, so `P j ≥ 30 j - 1`, and
    `(30 j - 1)² > 30 j + 1` for `j ≥ 1`.
  - If a leg `L` is not prime, its least prime factor `m` has `m² ≤ L ≤ 30 j + 1`, and `m`
    divides the product, so `P j ≤ m`; hence `(P j)² ≤ 30 j + 1`.
* `total_blame_iff` (b): for any bound `Nmax`, no copy `j` with `1 ≤ j < Nmax` has both legs
  prime exactly when every copy `j` with `1 ≤ j < Nmax` has `(P j)² ≤ 30 j + 1`.
* `total_blame_iff_primorial`: the same with the bound `Nmax = primorial q / 30`.
-/

namespace RangeLine

/-- The least prime factor of the product `900 j² - 1 = (30 j - 1)(30 j + 1)` of the two legs
of copy `j`. -/
def P (j : ℕ) : ℕ := Nat.minFac (900 * j ^ 2 - 1)

/-- `P j` is the least prime factor of the product of the two legs `(30 j - 1)(30 j + 1)`. -/
theorem P_eq_minFac_legs (j : ℕ) : P j = Nat.minFac ((30 * j - 1) * (30 * j + 1)) := by
  unfold P
  rw [copy_product]

/-- If `L` divides `n`, `L ≠ 1`, and `L` is not prime (with `L > 0`), then the least prime
factor of `n` is at most the least prime factor `m` of `L`, and `m² ≤ L`; so
`(minFac n)² ≤ L`. -/
lemma minFac_sq_le_of_composite_dvd {n L : ℕ} (hL : 0 < L) (hL1 : L ≠ 1) (hLp : ¬ L.Prime)
    (hdvd : L ∣ n) : (Nat.minFac n) ^ 2 ≤ L := by
  have hm2 : 2 ≤ Nat.minFac L := (Nat.minFac_prime hL1).two_le
  have hmn : Nat.minFac L ∣ n := dvd_trans (Nat.minFac_dvd L) hdvd
  have hle : Nat.minFac n ≤ Nat.minFac L := Nat.minFac_le_of_dvd hm2 hmn
  have hsq : (Nat.minFac L) ^ 2 ≤ L := Nat.minFac_sq_le_self hL hLp
  calc (Nat.minFac n) ^ 2 ≤ (Nat.minFac L) ^ 2 := Nat.pow_le_pow_left hle 2
    _ ≤ L := hsq

/-- (a) The region law. For `j ≥ 1`, both legs `30 j - 1` and `30 j + 1` of copy `j` are prime
exactly when `30 j + 1 < (P j)²`, where `P j` is the least prime factor of `900 j² - 1`. -/
theorem revealed_iff_minFac {j : ℕ} (hj : 1 ≤ j) :
    ((30 * j - 1).Prime ∧ (30 * j + 1).Prime) ↔ 30 * j + 1 < (P j) ^ 2 := by
  rw [P_eq_minFac_legs]
  set a := 30 * j - 1 with ha
  set b := 30 * j + 1 with hb
  have ha29 : 29 ≤ a := by omega
  have hab : b = a + 2 := by omega
  constructor
  · rintro ⟨hpa, hpb⟩
    -- the least prime factor of `a * b` is a prime dividing `a` or `b`, hence equal to one of them
    have hne : a * b ≠ 1 := by
      intro h
      have := Nat.eq_one_of_mul_eq_one_right h
      omega
    have hmp : (Nat.minFac (a * b)).Prime := Nat.minFac_prime hne
    have hge : a ≤ Nat.minFac (a * b) := by
      rcases (Nat.Prime.dvd_mul hmp).mp (Nat.minFac_dvd (a * b)) with h | h
      · rw [(Nat.prime_dvd_prime_iff_eq hmp hpa).mp h]
      · rw [(Nat.prime_dvd_prime_iff_eq hmp hpb).mp h]
        omega
    have hsq : a ^ 2 ≤ (Nat.minFac (a * b)) ^ 2 := Nat.pow_le_pow_left hge 2
    have ha2 : b < a ^ 2 := by
      rw [hab]
      nlinarith
    omega
  · intro hlt
    by_contra hnot
    rcases not_and_or.mp hnot with hna | hnb
    · have h := minFac_sq_le_of_composite_dvd (by omega) (by omega) hna (dvd_mul_right a b)
      omega
    · have h := minFac_sq_le_of_composite_dvd (by omega) (by omega) hnb (dvd_mul_left b a)
      omega

/-- The failure form of (a): for `j ≥ 1`, some leg of copy `j` is not prime exactly when
`(P j)² ≤ 30 j + 1`. -/
theorem blamed_iff_minFac {j : ℕ} (hj : 1 ≤ j) :
    ¬ ((30 * j - 1).Prime ∧ (30 * j + 1).Prime) ↔ (P j) ^ 2 ≤ 30 * j + 1 := by
  rw [revealed_iff_minFac hj, not_lt]

/-- (b) Total blame, with an arbitrary bound `Nmax`: no copy `j` with `1 ≤ j < Nmax` has both
legs prime exactly when every copy `j` with `1 ≤ j < Nmax` has `(P j)² ≤ 30 j + 1`. -/
theorem total_blame_iff (Nmax : ℕ) :
    (¬ ∃ j, 1 ≤ j ∧ j < Nmax ∧ (30 * j - 1).Prime ∧ (30 * j + 1).Prime) ↔
      ∀ j, 1 ≤ j → j < Nmax → (P j) ^ 2 ≤ 30 * j + 1 := by
  constructor
  · intro h j hj hjN
    exact (blamed_iff_minFac hj).mp (fun hp => h ⟨j, hj, hjN, hp⟩)
  · rintro h ⟨j, hj, hjN, hp⟩
    exact (blamed_iff_minFac hj).mpr (h j hj hjN) hp

/-- (b) at the bound `Nmax = primorial q / 30`: no copy `j` with `1 ≤ j < primorial q / 30` has
both legs prime exactly when every such copy has `(P j)² ≤ 30 j + 1`. -/
theorem total_blame_iff_primorial (q : ℕ) :
    (¬ ∃ j, 1 ≤ j ∧ j < primorial q / 30 ∧ (30 * j - 1).Prime ∧ (30 * j + 1).Prime) ↔
      ∀ j, 1 ≤ j → j < primorial q / 30 → (P j) ^ 2 ≤ 30 * j + 1 :=
  total_blame_iff (primorial q / 30)

end RangeLine
