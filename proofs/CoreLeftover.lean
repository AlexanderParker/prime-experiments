/-
THE CORE'S LEFTOVER: RIGIDITY OF THE CORE (S11), THE CROSSING (S12), THE TWO-PRIME LEMMA,
AND THE DEPTH LEMMA  (Formalist, round 40).

Sources.  `research/proof/core_leftover.md` section 5: S11 (a pairwise coprime set of
integers `>= 5`, coprime to 6, all `<= X`, has at most as many members as there are
primes in `[5, X]`, and with that many members every member is a power of a distinct
prime in `[5, X]`) and S12 (`min over x of K_L(x) = 0` iff the machine's longest fully
covered run is at least `L`).  `research/proof/step_evidence.md` section 7: a member of
a leftover slot below `(6L + 1)^3` with no prime factor `<= 6L + 1` is a prime or a
product of exactly two primes above `6L + 1`.  The depth lemma (candidate (d) of the
next unstick pass; theorem 3 of the proof skeleton in slot form): a slot `(n, n + 2)`
with both members free of prime factors below `B` and `n + 2 < B^2` is a twin.

THE OBJECTS.  `Rough B n` says every prime factor of `n` is at least `B` - the
"no prime factor below `B`" of the documents, in the form the arithmetic uses.
`primesIn N` is the set of primes in `[5, N]`.  The machine's struck slots are taken
from `OneStepE`: `SmallFactor q n` (a gear `p` of the engine `{5..q}` divides `n`)
and `Blocked q k` (column `k = (6k - 1, 6k + 1)` is struck), so the core of the
document (the primes `<= 6L + 1`) is the engine `{5..q}` at `q = 6L + 1`, and the
core-open slots of `[c, c')` are the columns `k` with `¬ Blocked (6L + 1) k`.
`TopMachine.Strikes` (over `ℤ`, arbitrary `Finset` of gears) was not used: the
documents' core is a threshold on primes, which is exactly `OneStepE`'s engine, and
`OneStepE.smallFactor_iff_not_prime` is reused verbatim for the twin identification
above the engine.

The crossing S12 is stated for an ABSTRACT struck predicate `Struck : ℕ → Prop` (the
machine enters only through `Struck := Blocked q`), with `leftover Struck L x` the
number of `i < L` with `x + i` unstruck, `Covered Struck x r` the fully covered stretch
`[x, x + r)`, `run Struck x` the longest covered stretch beginning at `x` (a `sSup`,
meaningful when some slot at or after `x` is unstruck), and `record Struck a b` the
longest run beginning in `[a, b)`.  The identification is exact: for every `L`,
`(∃ x, L ≤ run x) ↔ (∃ x, leftover L x = 0)`, and on a section, `L ≤ record a b` iff
some start in `[a, b)` has leftover `0` iff the minimum of `leftover L` over the
section is `0`.  For the machine itself the boundedness hypothesis is DISCHARGED:
`blocked_unbounded_open` shows every engine leaves an open column beyond every `x`
(the column `6k = 4P^2 + 2 + 6Px`, `P` the product of the engine's gears, has members
`≡ 1` and `≡ 3` modulo every gear).

Zero sorries; no `native_decide`, no `decide`, no `Lean.ofReduceBool`.
-/
import OneStepE
import Mathlib.Data.Nat.Factorization.Basic
import Mathlib.Data.Nat.GCD.BigOperators
import Mathlib.Order.Lattice.Nat
import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Algebra.Order.BigOperators.Group.Finset

namespace CoreLeftover

open OneStepE

/-! ## Rough numbers: the arithmetic behind the two-prime lemma and the depth lemma -/

/-- `Rough B n`: every prime factor of `n` is at least `B` ("no prime factor below `B`"). -/
def Rough (B n : ℕ) : Prop := ∀ p, p.Prime → p ∣ n → B ≤ p

theorem rough_of_dvd {B n m : ℕ} (h : Rough B n) (hm : m ∣ n) : Rough B m :=
  fun p hp hpm => h p hp (dvd_trans hpm hm)

/-- A number above `1` all of whose prime factors are `≥ B` is itself `≥ B`. -/
theorem le_of_rough {B n : ℕ} (h1 : 1 < n) (h : Rough B n) : B ≤ n :=
  le_trans (h _ (Nat.minFac_prime (by omega)) (Nat.minFac_dvd n)) (Nat.minFac_le (by omega))

/-- **The depth lemma, one member.**  A `B`-rough number below `B^2` is prime: a
composite `n` has least prime factor `p` with `p * p ≤ n < B^2`, so `p < B`. -/
theorem prime_of_rough_lt_sq {B n : ℕ} (h1 : 1 < n) (hlt : n < B ^ 2) (h : Rough B n) :
    n.Prime := by
  by_contra hn
  have hp : (n.minFac).Prime := Nat.minFac_prime (by omega)
  have hpp : n.minFac * n.minFac ≤ n := by
    have := Nat.minFac_sq_le_self (by omega) hn
    rwa [sq] at this
  have hB := h _ hp (Nat.minFac_dvd n)
  have := lt_of_mul_self_le_of_lt_sq hpp hlt
  omega

/-- `n` is a prime, or a product of two primes (possibly equal) both `≥ B`. -/
def PrimeOrSemiprime (B n : ℕ) : Prop :=
  n.Prime ∨ ∃ p q, p.Prime ∧ q.Prime ∧ B ≤ p ∧ B ≤ q ∧ n = p * q

/-- **The two-prime lemma** (step_evidence.md section 7).  A `B`-rough number `n` with
`1 < n < B^3` is a prime or a product of exactly two primes `≥ B`: three prime factors
counted with multiplicity would give `n ≥ B * B * B`.  No hypothesis on `B` is needed
(for `B ≤ 1` the bound `n < B^3` already contradicts `1 < n`). -/
theorem primeOrSemiprime_of_rough_lt_cube {B n : ℕ} (h1 : 1 < n) (hlt : n < B ^ 3)
    (h : Rough B n) : PrimeOrSemiprime B n := by
  have hp : (n.minFac).Prime := Nat.minFac_prime (by omega)
  have hBp : B ≤ n.minFac := h _ hp (Nat.minFac_dvd n)
  obtain ⟨m, hm⟩ := Nat.minFac_dvd n
  by_cases hm1 : m = 1
  · left
    rw [hm1, mul_one] at hm
    exact Nat.prime_def_minFac.mpr ⟨by omega, hm.symm⟩
  · have hm0 : m ≠ 0 := by rintro rfl; rw [mul_zero] at hm; omega
    have hroughm : Rough B m := rough_of_dvd h (Dvd.intro_left _ hm.symm)
    have hq : (m.minFac).Prime := Nat.minFac_prime hm1
    have hBq : B ≤ m.minFac := hroughm _ hq (Nat.minFac_dvd m)
    obtain ⟨r, hr⟩ := Nat.minFac_dvd m
    by_cases hr1 : r = 1
    · right
      have hm' : m = m.minFac := by simpa [hr1] using hr
      refine ⟨n.minFac, m.minFac, hp, hq, hBp, hBq, ?_⟩
      rw [← hm']
      exact hm
    · have hr0 : r ≠ 0 := by rintro rfl; rw [mul_zero] at hr; omega
      have hBr : B ≤ r :=
        le_of_rough (by omega) (rough_of_dvd hroughm (Dvd.intro_left _ hr.symm))
      exfalso
      have key : B * B * B ≤ n.minFac * m.minFac * r :=
        Nat.mul_le_mul (Nat.mul_le_mul hBp hBq) hBr
      have hn' : n.minFac * m.minFac * r = n := by rw [mul_assoc, ← hr, ← hm]
      have h3 : B ^ 3 = B * B * B := by ring
      rw [h3] at hlt
      exact lt_irrefl _ (lt_of_le_of_lt (key.trans hn'.le) hlt)

/-- **The depth lemma, slot form.**  Both members of `(n, n + 2)` `B`-rough and
`n + 2 < B^2`: the slot is a twin. -/
theorem twin_of_rough {B n : ℕ} (h1 : 1 < n) (hlt : n + 2 < B ^ 2) (hn : Rough B n)
    (hn2 : Rough B (n + 2)) : n.Prime ∧ (n + 2).Prime :=
  ⟨prime_of_rough_lt_sq h1 (lt_of_le_of_lt (Nat.le_add_right n 2) hlt) hn,
    prime_of_rough_lt_sq (by omega) hlt hn2⟩

/-- Section 7 of step_evidence.md, slot form: both members of a `B`-rough slot below
`B^3` are primes or products of two primes `≥ B`. -/
theorem slot_types_of_rough {B n : ℕ} (h1 : 1 < n) (hlt : n + 2 < B ^ 3) (hn : Rough B n)
    (hn2 : Rough B (n + 2)) : PrimeOrSemiprime B n ∧ PrimeOrSemiprime B (n + 2) :=
  ⟨primeOrSemiprime_of_rough_lt_cube h1 (lt_of_le_of_lt (Nat.le_add_right n 2) hlt) hn,
    primeOrSemiprime_of_rough_lt_cube (by omega) hlt hn2⟩

/-! ## The bridge to the engine `{5..q}` of `OneStepE` -/

/-- For `n` coprime to 6, "no gear of `{5..q}` divides `n`" is "`n` is `(q + 1)`-rough". -/
theorem not_smallFactor_iff_rough {q n : ℕ} (h2 : ¬ 2 ∣ n) (h3 : ¬ 3 ∣ n) :
    ¬ SmallFactor q n ↔ Rough (q + 1) n := by
  constructor
  · intro h p hp hd
    by_contra hlt
    exact h ⟨p, hp, five_le_of_dvd hp h2 h3 hd, by omega, hd⟩
  · rintro h ⟨p, hp, _, hpq, hd⟩
    have := h p hp hd
    omega

/-- **The depth lemma, column form.**  A column `k ≥ 1` open under `{5..q}` with
`6k + 1 < (q + 1)^2` is a twin prime pair.  No "next prime" is involved: `q + 1` is the
first integer above the engine. -/
theorem twin_of_not_blocked {q k : ℕ} (hk : 1 ≤ k) (hlt : 6 * k + 1 < (q + 1) ^ 2)
    (h : ¬ Blocked q k) : (6 * k - 1).Prime ∧ (6 * k + 1).Prime := by
  have h1 : ¬ SmallFactor q (6 * k - 1) := fun h' => h (Or.inl h')
  have h2 : ¬ SmallFactor q (6 * k + 1) := fun h' => h (Or.inr h')
  have hr1 := (not_smallFactor_iff_rough (by omega) (by omega)).mp h1
  have hr2 := (not_smallFactor_iff_rough (by omega) (by omega)).mp h2
  have e : 6 * k + 1 = 6 * k - 1 + 2 := by omega
  rw [e] at hr2 hlt ⊢
  exact twin_of_rough (by omega) hlt hr1 hr2

/-- Section 7 in column form: a column open under `{5..q}` below `(q + 1)^3` has both
members prime or a product of two primes above `q`. -/
theorem slot_types_of_not_blocked {q k : ℕ} (hk : 1 ≤ k) (hlt : 6 * k + 1 < (q + 1) ^ 3)
    (h : ¬ Blocked q k) :
    PrimeOrSemiprime (q + 1) (6 * k - 1) ∧ PrimeOrSemiprime (q + 1) (6 * k + 1) := by
  have h1 : ¬ SmallFactor q (6 * k - 1) := fun h' => h (Or.inl h')
  have h2 : ¬ SmallFactor q (6 * k + 1) := fun h' => h (Or.inr h')
  have hr1 := (not_smallFactor_iff_rough (by omega) (by omega)).mp h1
  have hr2 := (not_smallFactor_iff_rough (by omega) (by omega)).mp h2
  have e : 6 * k + 1 = 6 * k - 1 + 2 := by omega
  rw [e] at hr2 hlt ⊢
  exact slot_types_of_rough (by omega) hlt hr1 hr2

/-- Above the engine (`q < 6k - 1`) and below `(q + 1)^2`, open IS twin - both
directions, by `OneStepE.smallFactor_iff_not_prime` with `q' = q + 1`. -/
theorem not_blocked_iff_twin {q k : ℕ} (hq : q < 6 * k - 1) (hlt : 6 * k + 1 < (q + 1) ^ 2) :
    ¬ Blocked q k ↔ (6 * k - 1).Prime ∧ (6 * k + 1).Prime := by
  have hgap : ∀ p, p.Prime → p < q + 1 → p ≤ q := fun p _ h => by omega
  have e1 := smallFactor_iff_not_prime hgap (n := 6 * k - 1) (by omega) (by omega) hq
    (lt_of_le_of_lt (by omega) hlt) (by omega)
  have e2 := smallFactor_iff_not_prime hgap (n := 6 * k + 1) (by omega) (by omega)
    (by omega) hlt (by omega)
  unfold Blocked
  rw [e1, e2]
  tauto

/-! ## S11: rigidity of the core -/

/-- The primes in `[5, N]`: the core of the document at `N = 6L + 1`. -/
def primesIn (N : ℕ) : Finset ℕ := (Finset.Icc 5 N).filter Nat.Prime

theorem mem_primesIn {N p : ℕ} : p ∈ primesIn N ↔ p.Prime ∧ 5 ≤ p ∧ p ≤ N := by
  rw [primesIn, Finset.mem_filter, Finset.mem_Icc]
  tauto

theorem not_two_dvd_of_coprime_six {n : ℕ} (h : Nat.Coprime n 6) : ¬ 2 ∣ n := by
  intro h2
  have := Nat.dvd_gcd h2 (by norm_num : 2 ∣ 6)
  rw [h.gcd_eq_one] at this
  omega

theorem not_three_dvd_of_coprime_six {n : ℕ} (h : Nat.Coprime n 6) : ¬ 3 ∣ n := by
  intro h3
  have := Nat.dvd_gcd h3 (by norm_num : 3 ∣ 6)
  rw [h.gcd_eq_one] at this
  omega

/-- The least prime factor of a member (`5 ≤ n ≤ N`, coprime to 6) is a prime of `[5, N]`. -/
theorem minFac_mem_primesIn {N n : ℕ} (h5 : 5 ≤ n) (hN : n ≤ N) (h6 : Nat.Coprime n 6) :
    n.minFac ∈ primesIn N := by
  have hp : (n.minFac).Prime := Nat.minFac_prime (by omega)
  exact mem_primesIn.mpr ⟨hp,
    five_le_of_dvd hp (not_two_dvd_of_coprime_six h6) (not_three_dvd_of_coprime_six h6)
      (Nat.minFac_dvd n),
    le_trans (Nat.minFac_le (by omega)) hN⟩

/-- Pairwise coprimality makes the least prime factor injective on the set. -/
theorem minFac_injOn {S : Finset ℕ} (h5 : ∀ n ∈ S, 5 ≤ n)
    (hcop : ∀ a ∈ S, ∀ b ∈ S, a ≠ b → Nat.Coprime a b) :
    Set.InjOn Nat.minFac (S : Set ℕ) := by
  intro a ha b hb hab
  by_contra hne
  have hc := hcop a ha b hb hne
  have hd : a.minFac ∣ Nat.gcd a b :=
    Nat.dvd_gcd (Nat.minFac_dvd a) (by rw [hab]; exact Nat.minFac_dvd b)
  rw [hc.gcd_eq_one] at hd
  have := (Nat.minFac_prime (n := a) (by have := h5 a ha; omega)).two_le
  have := Nat.le_of_dvd one_pos hd
  omega

/-- **S11, the count.**  A pairwise coprime set of integers in `[5, N]`, each coprime
to 6, has at most as many members as there are primes in `[5, N]`. -/
theorem card_le_card_primesIn {N : ℕ} {S : Finset ℕ} (hS : ∀ n ∈ S, 5 ≤ n ∧ n ≤ N)
    (h6 : ∀ n ∈ S, Nat.Coprime n 6) (hcop : ∀ a ∈ S, ∀ b ∈ S, a ≠ b → Nat.Coprime a b) :
    S.card ≤ (primesIn N).card :=
  Finset.card_le_card_of_injOn Nat.minFac
    (fun n hn => minFac_mem_primesIn (hS n hn).1 (hS n hn).2 (h6 n hn))
    (minFac_injOn (fun n hn => (hS n hn).1) hcop)

/-- With as many members as primes, the least prime factors are exactly the primes of `[5, N]`. -/
theorem image_minFac_eq {N : ℕ} {S : Finset ℕ} (hS : ∀ n ∈ S, 5 ≤ n ∧ n ≤ N)
    (h6 : ∀ n ∈ S, Nat.Coprime n 6) (hcop : ∀ a ∈ S, ∀ b ∈ S, a ≠ b → Nat.Coprime a b)
    (hcard : (primesIn N).card ≤ S.card) : S.image Nat.minFac = primesIn N := by
  apply Finset.eq_of_subset_of_card_le
  · intro p hp
    obtain ⟨n, hn, rfl⟩ := Finset.mem_image.mp hp
    exact minFac_mem_primesIn (hS n hn).1 (hS n hn).2 (h6 n hn)
  · rw [Finset.card_image_of_injOn (minFac_injOn (fun n hn => (hS n hn).1) hcop)]
    exact hcard

/-- **S11, the prime-power clause.**  With as many members as primes, every member is
a positive power of its least prime factor: a second prime factor `q'` of `n` would be
the least prime factor of some other member, which then shares `q'` with `n`. -/
theorem eq_minFac_pow {N : ℕ} {S : Finset ℕ} (hS : ∀ n ∈ S, 5 ≤ n ∧ n ≤ N)
    (h6 : ∀ n ∈ S, Nat.Coprime n 6) (hcop : ∀ a ∈ S, ∀ b ∈ S, a ≠ b → Nat.Coprime a b)
    (hcard : (primesIn N).card ≤ S.card) {n : ℕ} (hn : n ∈ S) :
    ∃ k, 0 < k ∧ n = n.minFac ^ k := by
  have h5 := (hS n hn).1
  have hp : (n.minFac).Prime := Nat.minFac_prime (by omega)
  obtain ⟨e, m, hpm, hnm⟩ :=
    Nat.exists_eq_pow_mul_and_not_dvd (by omega : n ≠ 0) n.minFac hp.ne_one
  by_cases hm1 : m = 1
  · refine ⟨e, ?_, by rw [hm1, mul_one] at hnm; exact hnm⟩
    by_contra h0
    have he : e = 0 := by omega
    rw [he, pow_zero, hm1, mul_one] at hnm
    omega
  · exfalso
    have hm0 : m ≠ 0 := by rintro rfl; rw [mul_zero] at hnm; omega
    have hq : (m.minFac).Prime := Nat.minFac_prime hm1
    have hqm : m.minFac ∣ m := Nat.minFac_dvd m
    have hqn : m.minFac ∣ n := by rw [hnm]; exact Dvd.dvd.mul_left hqm _
    have hqne : m.minFac ≠ n.minFac := by
      intro h; apply hpm; rw [← h]; exact hqm
    have hq5 : 5 ≤ m.minFac :=
      five_le_of_dvd hq (not_two_dvd_of_coprime_six (h6 n hn))
        (not_three_dvd_of_coprime_six (h6 n hn)) hqn
    have hqN : m.minFac ≤ N := le_trans (Nat.le_of_dvd (by omega) hqn) (hS n hn).2
    have hqmem : m.minFac ∈ primesIn N := mem_primesIn.mpr ⟨hq, hq5, hqN⟩
    rw [← image_minFac_eq hS h6 hcop hcard] at hqmem
    obtain ⟨b, hb, hbq⟩ := Finset.mem_image.mp hqmem
    have hbn : b ≠ n := by
      intro h; rw [h] at hbq; exact hqne hbq.symm
    have hc := hcop b hb n hn hbn
    have hd : m.minFac ∣ Nat.gcd b n :=
      Nat.dvd_gcd (by rw [← hbq]; exact Nat.minFac_dvd b) hqn
    rw [hc.gcd_eq_one] at hd
    have := Nat.le_of_dvd one_pos hd
    have := hq.two_le
    omega

/-- **S11 (rigidity of the core), the package.**  A pairwise coprime set `S` of
integers in `[5, N]`, each coprime to 6, with at least as many members as there are
primes in `[5, N]`: every member is a positive power of its least prime factor, and
`minFac` is a bijection from `S` onto the primes of `[5, N]` - one prime power per
prime.  (With `card_le_card_primesIn`, the count is then exactly the number of primes.) -/
theorem rigidity {N : ℕ} {S : Finset ℕ} (hS : ∀ n ∈ S, 5 ≤ n ∧ n ≤ N)
    (h6 : ∀ n ∈ S, Nat.Coprime n 6) (hcop : ∀ a ∈ S, ∀ b ∈ S, a ≠ b → Nat.Coprime a b)
    (hcard : (primesIn N).card ≤ S.card) :
    (∀ n ∈ S, ∃ k, 0 < k ∧ n = n.minFac ^ k) ∧
      Set.BijOn Nat.minFac (S : Set ℕ) (primesIn N : Set ℕ) := by
  refine ⟨fun n hn => eq_minFac_pow hS h6 hcop hcard hn, ?_, ?_, ?_⟩
  · exact fun n hn => minFac_mem_primesIn (hS n hn).1 (hS n hn).2 (h6 n hn)
  · exact minFac_injOn (fun n hn => (hS n hn).1) hcop
  · intro p hp
    rw [← Finset.coe_image, image_minFac_eq hS h6 hcop hcard]
    exact hp

theorem card_eq_card_primesIn {N : ℕ} {S : Finset ℕ} (hS : ∀ n ∈ S, 5 ≤ n ∧ n ≤ N)
    (h6 : ∀ n ∈ S, Nat.Coprime n 6) (hcop : ∀ a ∈ S, ∀ b ∈ S, a ≠ b → Nat.Coprime a b)
    (hcard : (primesIn N).card ≤ S.card) : S.card = (primesIn N).card :=
  le_antisymm (card_le_card_primesIn hS h6 hcop) hcard

/-! ## S12: the crossing, for an abstract struck predicate -/

section Crossing

variable (Struck : ℕ → Prop)

/-- The stretch `[x, x + r)` is fully covered: every slot struck. -/
def Covered (x r : ℕ) : Prop := ∀ i, i < r → Struck (x + i)

open Classical in
/-- The leftover of the stretch of `L` slots beginning at `x`: the number of unstruck slots. -/
noncomputable def leftover (L x : ℕ) : ℕ :=
  ((Finset.range L).filter (fun i => ¬ Struck (x + i))).card

/-- The run at `x`: the longest fully covered stretch beginning at `x` (as a `sSup`;
it is the true maximum whenever some slot at or after `x` is unstruck, `run_isGreatest`). -/
noncomputable def run (x : ℕ) : ℕ := sSup {r | Covered Struck x r}

/-- The record of the section `[a, b)`: the longest run beginning in it. -/
noncomputable def record (a b : ℕ) : ℕ := (Finset.Ico a b).sup (run Struck)

open Classical in
theorem leftover_eq_zero_iff {L x : ℕ} : leftover Struck L x = 0 ↔ Covered Struck x L := by
  unfold leftover Covered
  rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
  constructor
  · intro h i hi
    have := h (Finset.mem_range.mpr hi)
    tauto
  · intro h i hi
    exact not_not.mpr (h i (Finset.mem_range.mp hi))

/-- The trivial half of S12: some stretch of length `L` has leftover `0` iff some
stretch of length `L` is fully covered. -/
theorem exists_leftover_zero_iff {L : ℕ} :
    (∃ x, leftover Struck L x = 0) ↔ ∃ x, Covered Struck x L :=
  exists_congr (fun _ => leftover_eq_zero_iff Struck)

theorem covered_mono {x r r' : ℕ} (h : r' ≤ r) (hc : Covered Struck x r) :
    Covered Struck x r' :=
  fun i hi => hc i (lt_of_lt_of_le hi h)

theorem covered_zero (x : ℕ) : Covered Struck x 0 :=
  fun _ hi => absurd hi (Nat.not_lt_zero _)

theorem bddAbove_covered {x r0 : ℕ} (h0 : ¬ Struck (x + r0)) :
    BddAbove {r | Covered Struck x r} := by
  refine ⟨r0, fun r hr => ?_⟩
  have hr' : Covered Struck x r := hr
  by_contra hlt
  exact h0 (hr' r0 (by omega))

/-- When some slot at or after `x` is unstruck, `run x` is the greatest covered length. -/
theorem run_isGreatest {x : ℕ} (h : ∃ r, ¬ Struck (x + r)) :
    IsGreatest {r | Covered Struck x r} (run Struck x) := by
  obtain ⟨r0, h0⟩ := h
  have hbdd := bddAbove_covered Struck h0
  have hne : {r | Covered Struck x r}.Nonempty := ⟨0, covered_zero Struck x⟩
  exact ⟨Nat.sSup_mem hne hbdd, fun r hr => le_csSup hbdd hr⟩

theorem le_run_iff {x L : ℕ} (h : ∃ r, ¬ Struck (x + r)) :
    L ≤ run Struck x ↔ Covered Struck x L := by
  obtain ⟨hmem, hub⟩ := run_isGreatest Struck h
  exact ⟨fun hL => covered_mono Struck hL hmem, fun hc => hub hc⟩

/-- **S12, the crossing.**  Some stretch of length `L` has leftover `0` iff some run
is at least `L` (the machine's longest fully covered run is `≥ L`).  The hypothesis:
beyond every position some slot is unstruck (so every run is finite). -/
theorem exists_run_ge_iff {L : ℕ} (h : ∀ x, ∃ r, ¬ Struck (x + r)) :
    (∃ x, L ≤ run Struck x) ↔ ∃ x, leftover Struck L x = 0 := by
  constructor
  · rintro ⟨x, hx⟩
    exact ⟨x, (leftover_eq_zero_iff Struck).mpr ((le_run_iff Struck (h x)).mp hx)⟩
  · rintro ⟨x, hx⟩
    exact ⟨x, (le_run_iff Struck (h x)).mpr ((leftover_eq_zero_iff Struck).mp hx)⟩

/-- S12 on a section: the record of `[a, b)` is at least `L` iff some start in `[a, b)`
has leftover `0`.  (`0 < L` because the record of an empty section is `0`.) -/
theorem le_record_iff {a b L : ℕ} (hL : 0 < L) (h : ∀ x, ∃ r, ¬ Struck (x + r)) :
    L ≤ record Struck a b ↔ ∃ x, a ≤ x ∧ x < b ∧ leftover Struck L x = 0 := by
  unfold record
  rw [Finset.le_sup_iff (show (⊥ : ℕ) < L from hL)]
  constructor
  · rintro ⟨x, hx, hLx⟩
    exact ⟨x, (Finset.mem_Ico.mp hx).1, (Finset.mem_Ico.mp hx).2,
      (leftover_eq_zero_iff Struck).mpr ((le_run_iff Struck (h x)).mp hLx)⟩
  · rintro ⟨x, ha, hb, hx⟩
    exact ⟨x, Finset.mem_Ico.mpr ⟨ha, hb⟩,
      (le_run_iff Struck (h x)).mpr ((leftover_eq_zero_iff Struck).mp hx)⟩

/-- The minimum of the leftover over a nonempty section is `0` iff some start in the
section has leftover `0`. -/
theorem sInf_leftover_eq_zero_iff {a b L : ℕ} (hab : a < b) :
    sInf ((leftover Struck L) '' Set.Ico a b) = 0 ↔
      ∃ x, a ≤ x ∧ x < b ∧ leftover Struck L x = 0 := by
  rw [Nat.sInf_eq_zero]
  constructor
  · rintro (⟨x, hx, hx0⟩ | hempty)
    · exact ⟨x, hx.1, hx.2, hx0⟩
    · exfalso
      have hmem : leftover Struck L a ∈ (leftover Struck L) '' Set.Ico a b :=
        Set.mem_image_of_mem _ ⟨le_rfl, hab⟩
      rw [hempty] at hmem
      exact hmem
  · rintro ⟨x, ha, hb, hx⟩
    exact Or.inl ⟨x, ⟨ha, hb⟩, hx⟩

/-- **S12 as the document states it**: on a section `[a, b)`, `min over x of K_L(x) = 0`
iff the section's longest fully covered run is at least `L`. -/
theorem min_leftover_eq_zero_iff_record {a b L : ℕ} (hab : a < b) (hL : 0 < L)
    (h : ∀ x, ∃ r, ¬ Struck (x + r)) :
    sInf ((leftover Struck L) '' Set.Ico a b) = 0 ↔ L ≤ record Struck a b := by
  rw [sInf_leftover_eq_zero_iff Struck hab, le_record_iff Struck hL h]

end Crossing

/-! ## S12 for the machine: `Struck := Blocked q`, the boundedness discharged -/

/-- Every engine `{5..q}` leaves an open column at or beyond every `x`: with `P` the
product of the gears, the column `k` with `6k = 4P^2 + 2 + 6Px` has members
`4P^2 + 6Px + 1 ≡ 1` and `4P^2 + 6Px + 3 ≡ 3` modulo every gear. -/
theorem blocked_unbounded_open (q x : ℕ) : ∃ r, ¬ Blocked q (x + r) := by
  set P : ℕ := ∏ p ∈ primesIn q, p with hP
  have hPpos : 0 < P :=
    Nat.pos_of_ne_zero (Finset.prod_ne_zero_iff.mpr (fun p hp => (mem_primesIn.mp hp).1.ne_zero))
  have hP3 : Nat.Coprime P 3 := Nat.Coprime.prod_left (fun p hp =>
    (Nat.coprime_primes (mem_primesIn.mp hp).1 Nat.prime_three).mpr
      (by have := (mem_primesIn.mp hp).2.1; omega))
  have h3P : ¬ 3 ∣ P := fun h => by have := hP3.symm.eq_one_of_dvd h; omega
  have hu : (P * P) % 3 = 1 := by
    have : P % 3 = 1 ∨ P % 3 = 2 := by omega
    rcases this with h | h <;> simp [Nat.mul_mod, h]
  obtain ⟨c, hc⟩ : 6 ∣ 4 * (P * P) + 2 := by omega
  have hdvd : ∀ p ∈ primesIn q, p ∣ 4 * (P * P) + 6 * (P * x) := by
    intro p hp
    have : 4 * (P * P) + 6 * (P * x) = P * (4 * P + 6 * x) := by ring
    rw [this]
    exact Dvd.dvd.mul_right (Finset.dvd_prod_of_mem _ hp) _
  have hxk : x ≤ c + P * x := le_trans (Nat.le_mul_of_pos_left x hPpos) (Nat.le_add_left _ _)
  refine ⟨c + P * x - x, ?_⟩
  have hk : x + (c + P * x - x) = c + P * x := by omega
  rw [hk]
  have e1 : 6 * (c + P * x) - 1 = 4 * (P * P) + 6 * (P * x) + 1 := by omega
  have e2 : 6 * (c + P * x) + 1 = 4 * (P * P) + 6 * (P * x) + 3 := by omega
  rintro (⟨p, hp, h5, hpq, hd⟩ | ⟨p, hp, h5, hpq, hd⟩)
  · rw [e1] at hd
    have := (Nat.dvd_add_right (hdvd p (mem_primesIn.mpr ⟨hp, h5, hpq⟩))).mp hd
    have := Nat.le_of_dvd one_pos this
    omega
  · rw [e2] at hd
    have := (Nat.dvd_add_right (hdvd p (mem_primesIn.mpr ⟨hp, h5, hpq⟩))).mp hd
    have := Nat.le_of_dvd (by norm_num) this
    omega

/-- **S12 for the machine `{5..q}`**, unconditionally: some stretch of `L` columns has
leftover `0` iff some run of `{5..q}` is at least `L`. -/
theorem crossing (q L : ℕ) :
    (∃ x, L ≤ run (Blocked q) x) ↔ ∃ x, leftover (Blocked q) L x = 0 :=
  exists_run_ge_iff _ (blocked_unbounded_open q)

/-- **S12 for the core of the document**: the core for length `L` is the engine
`{5..6L + 1}`, and `min over x in [a, b) of K_L(x) = 0` iff `R(6L + 1) ≥ L` on the section. -/
theorem crossing_core {a b L : ℕ} (hab : a < b) (hL : 0 < L) :
    sInf ((leftover (Blocked (6 * L + 1)) L) '' Set.Ico a b) = 0 ↔
      L ≤ record (Blocked (6 * L + 1)) a b :=
  min_leftover_eq_zero_iff_record _ hab hL (blocked_unbounded_open _)

/-! ## The depth lemma on a stretch: the leftover count IS the twin count -/

open Classical in
/-- On a stretch of `L` columns starting above the engine (`q < 6x - 1`) and ending
below `(q + 1)^2`, the leftover of `{5..q}` is exactly the number of twin prime columns:
every leftover slot is a twin, and every twin is a leftover slot. -/
theorem leftover_eq_card_twins {q x L : ℕ} (hq : q < 6 * x - 1)
    (hlt : 6 * (x + L) + 1 ≤ (q + 1) ^ 2) :
    leftover (Blocked q) L x =
      ((Finset.range L).filter
        (fun i => (6 * (x + i) - 1).Prime ∧ (6 * (x + i) + 1).Prime)).card := by
  unfold leftover
  congr 1
  apply Finset.filter_congr
  intro i hi
  rw [Finset.mem_range] at hi
  exact not_blocked_iff_twin (by omega) (lt_of_lt_of_le (by omega) hlt)

/-- The stretch form of the depth lemma with only the one-sided hypothesis: every
core-open column of a stretch below `(q + 1)^2` is a twin. -/
theorem stretch_twins {q x L : ℕ} (hx : 1 ≤ x) (hlt : 6 * (x + L) + 1 ≤ (q + 1) ^ 2) :
    ∀ i, i < L → ¬ Blocked q (x + i) →
      (6 * (x + i) - 1).Prime ∧ (6 * (x + i) + 1).Prime :=
  fun i hi h => twin_of_not_blocked (by omega) (lt_of_lt_of_le (by omega) hlt) h

end CoreLeftover
