/-
LadderCovering (2026-09-20): the free covering route to the window statement.

Columns `n` hold `(6n - 1, 6n + 1)`.  The GEARS of the machine `q` are the primes `p` with
`5 ≤ p ≤ q`; gear `p` STRIKES column `n` when `p ∣ 6n - 1` or `p ∣ 6n + 1`.  Since 6 is a unit
mod `p`, the struck columns of gear `p` are exactly two residue classes mod `p`
(`struck_classes`).  FREE covering lets every gear pick ANY two classes; if no choice covers a
run of columns then the actual classes do not either, so the run holds an unstruck column
(`exists_unstruck`).  An unstruck column of the window `q < 6n - 1`, `6n + 1 ≤ q²` has both
members prime by the least-prime-factor argument (`prime_of_unstruck_member`), so it is a twin
centre (`window_twin_of_free_uncoverable`).  THE COVERING HYPOTHESIS is free uncoverability of
the whole window of every machine; it gives the window statement and twins unbounded.

Column 0 is a degenerate case of the ℕ-truncated `6 * 0 - 1 = 0`, which every gear divides; the
class lemmas therefore carry `1 ≤ n` / `1 ≤ a`, which every window run satisfies.
-/
import TwinLadderTheorem
import Mathlib.Data.Nat.ModEq
import Mathlib.Data.Int.GCD

namespace TwinLadder

/-- A gear of the machine `q`: a prime `p` with `5 ≤ p ≤ q`. -/
def Gear (q p : ℕ) : Prop := p.Prime ∧ 5 ≤ p ∧ p ≤ q

/-- Column `n` is struck by some gear of `q`. -/
def Struck (q n : ℕ) : Prop := ∃ p, Gear q p ∧ (p ∣ 6 * n - 1 ∨ p ∣ 6 * n + 1)

/-- The run of `L` columns starting at `a`. -/
def Run (a L : ℕ) : Set ℕ := {n | a ≤ n ∧ n < a + L}

/-- The classes `r p`, `s p` (one pair per gear) cover every column of the run. -/
def FreeCovers (q a L : ℕ) (r s : ℕ → ℕ) : Prop :=
  ∀ n, a ≤ n → n < a + L → ∃ p, Gear q p ∧ (n % p = r p % p ∨ n % p = s p % p)

/-- No choice of two classes per gear covers the run. -/
def FreeUncoverable (q a L : ℕ) : Prop := ∀ r s : ℕ → ℕ, ¬ FreeCovers q a L r s

/-- A prime `p ≥ 5` is coprime to 6. -/
theorem coprime_six {p : ℕ} (hp : p.Prime) (h5 : 5 ≤ p) : Nat.Coprime p 6 := by
  have h2 : Nat.Coprime p 2 := (Nat.coprime_primes hp Nat.prime_two).2 (by omega)
  have h3 : Nat.Coprime p 3 := (Nat.coprime_primes hp Nat.prime_three).2 (by omega)
  have h := Nat.Coprime.mul_right h2 h3
  have e : (2 : ℕ) * 3 = 6 := by norm_num
  rw [e] at h
  exact h

/-- An inverse of 6 mod `p` (any value when `p` is not coprime to 6 or `p ≤ 1`). -/
noncomputable def invSix (p : ℕ) : ℕ :=
  if h : Nat.Coprime 6 p ∧ 1 < p then
    Classical.choose (Nat.exists_mul_mod_eq_one_of_coprime h.1 h.2)
  else 0

/-- For a prime `p ≥ 5`, `6 · invSix p ≡ 1 (mod p)`. -/
theorem invSix_spec {p : ℕ} (hp : p.Prime) (h5 : 5 ≤ p) : 6 * invSix p % p = 1 := by
  have h : Nat.Coprime 6 p ∧ 1 < p := ⟨(coprime_six hp h5).symm, hp.one_lt⟩
  unfold invSix
  split_ifs
  exact (Classical.choose_spec (Nat.exists_mul_mod_eq_one_of_coprime h.1 h.2)).2

/-- `p ∣ 6n - 1` (with `n ≥ 1`) forces `n ≡ x (mod p)` when `6x ≡ 1 (mod p)`. -/
theorem mod_eq_of_dvd_sub {p n x : ℕ} (hp : p.Prime) (h5 : 5 ≤ p) (hn : 1 ≤ n)
    (hx : 6 * x % p = 1) (h : p ∣ 6 * n - 1) : n % p = x % p := by
  have h1 : 1 ≡ 6 * n [MOD p] := (Nat.modEq_iff_dvd' (by omega)).2 h
  have h2 : 6 * x ≡ 1 [MOD p] := by
    unfold Nat.ModEq; rw [hx, Nat.mod_eq_of_lt hp.one_lt]
  have h3 : 6 * x ≡ 6 * n [MOD p] := h2.trans h1
  exact (Nat.ModEq.cancel_left_of_coprime (coprime_six hp h5) h3).symm

/-- `p ∣ 6n + 1` forces `n ≡ (p - 1) x (mod p)` when `6x ≡ 1 (mod p)`. -/
theorem mod_eq_of_dvd_add {p n x : ℕ} (hp : p.Prime) (h5 : 5 ≤ p)
    (hx : 6 * x % p = 1) (h : p ∣ 6 * n + 1) : n % p = (p - 1) * x % p := by
  have h1 : 6 * n + 1 ≡ 0 [MOD p] := Nat.modEq_zero_iff_dvd.2 h
  have hp0 : (p - 1) + 1 ≡ 0 [MOD p] := by
    rw [show p - 1 + 1 = p by omega]; exact Nat.modEq_zero_iff_dvd.2 dvd_rfl
  have h3 : 6 * n ≡ p - 1 [MOD p] := Nat.ModEq.add_right_cancel' 1 (h1.trans hp0.symm)
  have hx' : 6 * x ≡ 1 [MOD p] := by
    unfold Nat.ModEq; rw [hx, Nat.mod_eq_of_lt hp.one_lt]
  have h4 : (p - 1) * (6 * x) ≡ (p - 1) * 1 [MOD p] := Nat.ModEq.mul_left _ hx'
  rw [mul_one] at h4
  have h5' : 6 * ((p - 1) * x) ≡ 6 * n [MOD p] := by
    calc 6 * ((p - 1) * x) = (p - 1) * (6 * x) := by ring
      _ ≡ p - 1 [MOD p] := h4
      _ ≡ 6 * n [MOD p] := h3.symm
  exact (Nat.ModEq.cancel_left_of_coprime (coprime_six hp h5) h5').symm

/-- **The struck columns of each gear lie in two residue classes**: there are classes `r p`,
`s p` such that every struck column `n ≥ 1` matches one of them for some gear. -/
theorem struck_classes (q : ℕ) : ∃ r s : ℕ → ℕ, ∀ n, 1 ≤ n → Struck q n →
    ∃ p, Gear q p ∧ (n % p = r p % p ∨ n % p = s p % p) := by
  refine ⟨invSix, fun p => (p - 1) * invSix p, ?_⟩
  rintro n hn ⟨p, hp, hdvd⟩
  refine ⟨p, hp, ?_⟩
  have hx := invSix_spec hp.1 hp.2.1
  rcases hdvd with h | h
  · exact Or.inl (mod_eq_of_dvd_sub hp.1 hp.2.1 hn hx h)
  · exact Or.inr (mod_eq_of_dvd_add hp.1 hp.2.1 hx h)

/-- **A free-uncoverable run (from column `a ≥ 1`) holds an unstruck column.** -/
theorem exists_unstruck (q a L : ℕ) (ha : 1 ≤ a) (h : FreeUncoverable q a L) :
    ∃ n, a ≤ n ∧ n < a + L ∧ ¬ Struck q n := by
  obtain ⟨r, s, hrs⟩ := struck_classes q
  have hnc := h r s
  unfold FreeCovers at hnc
  push Not at hnc
  obtain ⟨n, hn1, hn2, hn⟩ := hnc
  refine ⟨n, hn1, hn2, fun hs => ?_⟩
  obtain ⟨p, hp, hc⟩ := hrs n (by omega) hs
  rcases hc with hc | hc
  · exact (hn p hp).1 hc
  · exact (hn p hp).2 hc

/-- **An unstruck member of the window is prime**: `m` coprime to 6 with `q < m ≤ q²` and no
gear dividing it; otherwise its least prime factor `f` has `f² ≤ m ≤ q²`, so `f ≤ q`, and
`f ≥ 5` by coprimality with 6, making `f` a gear dividing `m`. -/
theorem prime_of_unstruck_member {q m : ℕ} (hq : q.Prime) (h5 : 5 ≤ q) (hm6 : Nat.Coprime m 6)
    (hq_lt : q < m) (hle : m ≤ q ^ 2) (hno : ∀ p, Gear q p → ¬ p ∣ m) : m.Prime := by
  by_contra hnp
  have hm1 : m ≠ 1 := by omega
  have hf : (Nat.minFac m).Prime := Nat.minFac_prime hm1
  have hfd : Nat.minFac m ∣ m := Nat.minFac_dvd m
  have hsq : Nat.minFac m ^ 2 ≤ m := Nat.minFac_sq_le_self (by omega) hnp
  have hfq : Nat.minFac m ≤ q := by
    by_contra hlt
    push Not at hlt
    have : q ^ 2 < Nat.minFac m ^ 2 := Nat.pow_lt_pow_left hlt (by norm_num)
    omega
  have hf6 : Nat.Coprime (Nat.minFac m) 6 := Nat.Coprime.coprime_dvd_left hfd hm6
  have hf5 : 5 ≤ Nat.minFac m := by
    by_contra hlt
    push Not at hlt
    have h2 := hf.two_le
    interval_cases (Nat.minFac m)
    · exact absurd hf6 (by decide)
    · exact absurd hf6 (by decide)
    · exact absurd hf (by decide)
  exact hno _ ⟨hf, hf5, hfq⟩ hfd

/-- `6n - 1` is coprime to 6 for `n ≥ 1`. -/
theorem coprime_sub_six {n : ℕ} (hn : 1 ≤ n) : Nat.Coprime (6 * n - 1) 6 := by
  have h : (6 * n - 1) % 6 = 5 := by omega
  rw [Nat.Coprime, Nat.gcd_comm, Nat.gcd_rec, h]
  decide

/-- `6n + 1` is coprime to 6. -/
theorem coprime_add_six (n : ℕ) : Nat.Coprime (6 * n + 1) 6 := by
  have h : (6 * n + 1) % 6 = 1 := by omega
  rw [Nat.Coprime, Nat.gcd_comm, Nat.gcd_rec, h]
  decide

/-- **A free-uncoverable run inside the window of `q` holds a twin centre.** -/
theorem window_twin_of_free_uncoverable (q : ℕ) (hq : q.Prime) (h5 : 5 ≤ q) (a L : ℕ)
    (ha : q < 6 * a - 1) (hL : 6 * (a + L - 1) + 1 ≤ q ^ 2) (hL0 : 1 ≤ L)
    (h : FreeUncoverable q a L) :
    ∃ n, q < 6 * n - 1 ∧ 6 * n + 1 ≤ q ^ 2 ∧ TwinCentre (6 * n) := by
  obtain ⟨n, hn1, hn2, hns⟩ := exists_unstruck q a L (by omega) h
  have hn : 1 ≤ n := by omega
  have hlt : q < 6 * n - 1 := by omega
  have hle : 6 * n + 1 ≤ q ^ 2 := by omega
  have hno1 : ∀ p, Gear q p → ¬ p ∣ 6 * n - 1 := fun p hp hd => hns ⟨p, hp, Or.inl hd⟩
  have hno2 : ∀ p, Gear q p → ¬ p ∣ 6 * n + 1 := fun p hp hd => hns ⟨p, hp, Or.inr hd⟩
  have hp1 : (6 * n - 1).Prime :=
    prime_of_unstruck_member hq h5 (coprime_sub_six hn) hlt (by omega) hno1
  have hp2 : (6 * n + 1).Prime :=
    prime_of_unstruck_member hq h5 (coprime_add_six n) (by omega) hle hno2
  exact ⟨n, hlt, hle, ⟨dvd_mul_right 6 n, hp1, hp2⟩⟩

/-- **The covering hypothesis**: for every machine `q` the whole window, as one run from column
`(q + 7) / 6` to column `(q² - 1) / 6`, is free-uncoverable. -/
def CoveringHyp : Prop :=
  ∀ q, q.Prime → 5 ≤ q → FreeUncoverable q ((q + 7) / 6) ((q ^ 2 - 1) / 6 - (q + 7) / 6 + 1)

/-- **The window statement from the covering hypothesis.** -/
theorem windowStatement_of_coveringHyp (h : CoveringHyp) :
    ∀ q, q.Prime → 5 ≤ q → ∃ n, q < 6 * n - 1 ∧ 6 * n + 1 ≤ q ^ 2 ∧ TwinCentre (6 * n) := by
  intro q hq h5
  have hqq : q + 8 ≤ q ^ 2 := by nlinarith
  have ht : 25 ≤ q ^ 2 := by nlinarith
  refine window_twin_of_free_uncoverable q hq h5 _ _ ?_ ?_ ?_ (h q hq h5)
  · omega
  · generalize q ^ 2 = t at *; omega
  · omega

/-- **Twins unbounded from the covering hypothesis.** -/
theorem twins_unbounded_of_coveringHyp (h : CoveringHyp) :
    ∀ N : ℕ, ∃ m : ℕ, N < 6 * m - 1 ∧ (6 * m - 1).Prime ∧ (6 * m + 1).Prime := by
  intro N
  obtain ⟨q, hNq, hq⟩ := Nat.exists_infinite_primes (N + 6)
  obtain ⟨n, hlt, -, -, hp1, hp2⟩ := windowStatement_of_coveringHyp h q hq (by omega)
  exact ⟨n, by omega, hp1, hp2⟩

end TwinLadder
