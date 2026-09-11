/-
THE FIELDS, PART B: THE FLOOR OF EACH FIELD (D3), THE LEAST FACTOR BOUND AND THE CONFINEMENT
BELOW `P^2` (D4), THE DEEPEST FIELD PRESENT BELOW `P^2` (D5), THE SQUARE FIELD IS EMPTY BETWEEN
CONSECUTIVE PRIME SQUARES (B5a) AND AT A CUT (B5b), THE OVERLAY'S STRIKE PREDICATE IS PERIODIC
(E6) AND NO FIELD IS PERIODIC (E6b)  (Formalist).

Builds on `Fields` (the objects `InS`, `field j`, `squareField`, `overlay`, `Hits`).

WHAT THE PROOFS ACTUALLY NEED.
* D3/D4 are one fact: a list of `j` factors each `≥ b` has product `≥ b^j`
  (`List.pow_card_le_prod`); the field form is `n = prod (primeFactorsList n)`.  For D3 the
  bound `b = 5` is `five_le_of_mem_primeFactorsList`; for D4 the bound `b = n.minFac` is
  `Nat.minFac_le_of_dvd`.  Neither needs `j ≥ 1` or `j ≥ 2`; the statements are given without
  those hypotheses (strictly stronger than asked).
* B5a needs only the gap hypothesis and `q` prime: `p^2 < q^2 < p'^2` forces `p < q < p'`.
* E6 needs NOTHING of the gears (not prime, not `≥ 5`): only `g ∣ M` for `g ∈ G`, which is
  `Finset.dvd_prod_of_mem`.  It needs `1 ≤ k` because `6·0 - 1 = 0` in `ℕ` is divisible by
  everything.
* E6b: if `field j` were `M`-periodic then `5^j ∈ field j` would force
  `5^j (1 + M) = 5^j + 5^j · M ∈ field j`, but that number has at least `j + 1` factors.

Zero sorries; no `native_decide`, no `decide`, no `Lean.ofReduceBool`.
-/
import Fields
import Mathlib.Algebra.Order.BigOperators.Group.List

namespace Fields

open OneStepE CoreLeftover SquareColumn

/-! ## D3, D4, D5: the floor of a field, the least factor, the deepest field below `P^2` -/

/-- A survivor with `j` prime factors, each `≥ b`, is at least `b^j`. -/
theorem pow_length_le_of_forall_le {n b : ℕ} (hn : n ≠ 0)
    (hb : ∀ p ∈ n.primeFactorsList, b ≤ p) : b ^ n.primeFactorsList.length ≤ n := by
  have := List.pow_card_le_prod n.primeFactorsList b hb
  rwa [Nat.prod_primeFactorsList hn] at this

/-- **D3.**  Field `j` is empty below `5^j`: every member of field `j` is at least `5^j`. -/
theorem field_empty_below {n j : ℕ} (hn : n ∈ field j) : 5 ^ j ≤ n := by
  obtain ⟨hS, hlen⟩ := mem_field.mp hn
  rw [← hlen]
  exact pow_length_le_of_forall_le (ne_zero_of_inS hS)
    (fun p hp => five_le_of_mem_primeFactorsList hS hp)

/-- **D4.**  The least prime factor of a member of field `j` is at most its `j`-th root:
`n.minFac ^ j ≤ n`.  (No hypothesis `j ≥ 1` is needed.) -/
theorem field_least_factor_le {n j : ℕ} (hn : n ∈ field j) : n.minFac ^ j ≤ n := by
  obtain ⟨hS, hlen⟩ := mem_field.mp hn
  rw [← hlen]
  refine pow_length_le_of_forall_le (ne_zero_of_inS hS) (fun p hp => ?_)
  exact Nat.minFac_le_of_dvd (Nat.prime_of_mem_primeFactorsList hp).two_le
    (Nat.dvd_of_mem_primeFactorsList hp)

/-- **D4, confined.**  A member of field `j` below `P^2` is struck by a gear `g = n.minFac`
with `g ^ j < P^2`.  (No hypothesis `j ≥ 2` is needed.) -/
theorem field_confined {n j P : ℕ} (hn : n ∈ field j) (hP : n < P ^ 2) :
    n.minFac ^ j < P ^ 2 :=
  lt_of_le_of_lt (field_least_factor_le hn) hP

/-- **D5.**  The deepest field present below `P^2` has index `j` with `5^j < P^2`. -/
theorem field_index_le {n j P : ℕ} (hn : n ∈ field j) (hP : n < P ^ 2) : 5 ^ j < P ^ 2 :=
  lt_of_le_of_lt (field_empty_below hn) hP

/-! ## B5a, B5b: the square field between consecutive prime squares, and at a cut -/

/-- **B5a.**  For consecutive primes `p < p'` (no prime strictly between) no prime square lies
strictly between `p^2` and `p'^2`. -/
theorem squareField_empty_between {p p' q : ℕ} (hgap : ∀ r, r.Prime → p < r → p' ≤ r)
    (hq : q.Prime) (h1 : p ^ 2 < q ^ 2) (h2 : q ^ 2 < p' ^ 2) : False := by
  have hpq : p < q := (Nat.pow_lt_pow_iff_left (by norm_num)).mp h1
  have hqp' : q < p' := (Nat.pow_lt_pow_iff_left (by norm_num)).mp h2
  have := hgap q hq hpq
  omega

/-- **B5a on the square field.**  Between the squares of consecutive primes `p < p'` the square
field has no member: `p^2 < m < p'^2` and `m ∈ squareField` is impossible. -/
theorem squareField_empty_between' {p p' m : ℕ} (hgap : ∀ r, r.Prime → p < r → p' ≤ r)
    (hm : m ∈ squareField) (h1 : p ^ 2 < m) (h2 : m < p' ^ 2) : False := by
  obtain ⟨q, hq, -, rfl⟩ := hm
  exact squareField_empty_between hgap hq h1 h2

/-- **B5b.**  If `p ≥ 5` is the least prime at or above the cut `c`, then `p^2` is in the square
field and every prime square at or above `c^2` is at least `p^2`: `p^2` is the first square of
the square field at or after the cut. -/
theorem squareField_cuts {c p : ℕ} (hp : p.Prime) (h5 : 5 ≤ p)
    (hleast : ∀ q, q.Prime → c ≤ q → p ≤ q) :
    p ^ 2 ∈ squareField ∧ ∀ q, q.Prime → c ^ 2 ≤ q ^ 2 → p ^ 2 ≤ q ^ 2 := by
  refine ⟨⟨p, hp, h5, rfl⟩, fun q hq hcq => ?_⟩
  have hcq' : c ≤ q := (Nat.pow_le_pow_iff_left (by norm_num)).mp hcq
  exact (Nat.pow_le_pow_iff_left (by norm_num)).mpr (hleast q hq hcq')

/-! ## E6: the overlay's strike predicate is periodic; E6b: no field is periodic -/

/-- **E6.**  For a finite set of gears `G` with product `M`, "some `g ∈ G` divides `6k - 1` or
`6k + 1`" is periodic in `k ≥ 1` with period `M`.  Nothing is assumed of the gears. -/
theorem overlay_periodic {G : Finset ℕ} {k : ℕ} (hk : 1 ≤ k) :
    (∃ g ∈ G, g ∣ 6 * (k + ∏ h ∈ G, h) - 1 ∨ g ∣ 6 * (k + ∏ h ∈ G, h) + 1) ↔
      (∃ g ∈ G, g ∣ 6 * k - 1 ∨ g ∣ 6 * k + 1) := by
  apply exists_congr
  intro g
  apply and_congr_right
  intro hg
  have hdvd : g ∣ 6 * ∏ h ∈ G, h := Dvd.dvd.mul_left (Finset.dvd_prod_of_mem (fun h => h) hg) 6
  have e1 : 6 * (k + ∏ h ∈ G, h) - 1 = (6 * k - 1) + 6 * ∏ h ∈ G, h := by
    have : 1 ≤ 6 * k := by omega
    rw [Nat.mul_add]; omega
  have e2 : 6 * (k + ∏ h ∈ G, h) + 1 = (6 * k + 1) + 6 * ∏ h ∈ G, h := by
    rw [Nat.mul_add]; omega
  rw [e1, e2, Nat.dvd_add_left hdvd, Nat.dvd_add_left hdvd]

/-- `5^j` lies in field `j`: the leftmost member of every field. -/
theorem five_pow_mem_field (j : ℕ) : 5 ^ j ∈ field j := by
  induction j with
  | zero =>
    refine ⟨Or.inl rfl, ?_⟩
    rw [pow_zero, Nat.primeFactorsList_one]; rfl
  | succ j ih =>
    rw [pow_succ, mul_comm]
    exact (field_dilate (by norm_num) le_rfl).mp ih

/-- A member of field `j` times a number `≥ 2` has at least `j + 1` factors, so leaves field `j`. -/
theorem mul_not_mem_field {n j m : ℕ} (hn : n ∈ field j) (hm : 2 ≤ m) : n * m ∉ field j := by
  intro h
  obtain ⟨hS, hlen⟩ := mem_field.mp hn
  obtain ⟨-, hlen'⟩ := mem_field.mp h
  have hn0 : n ≠ 0 := ne_zero_of_inS hS
  have hm0 : m ≠ 0 := by omega
  rw [(Nat.perm_primeFactorsList_mul hn0 hm0).length_eq, List.length_append, hlen] at hlen'
  have hne : m.primeFactorsList ≠ [] := by
    rw [Ne, Nat.primeFactorsList_eq_nil]; omega
  have : 0 < m.primeFactorsList.length := List.length_pos_iff.mpr hne
  omega

/-- If field `j` were invariant under `+ M` it would be invariant under `+ t·M`. -/
theorem field_add_mul_of_add {j M : ℕ} (hstep : ∀ n, n ∈ field j → n + M ∈ field j) (n : ℕ)
    (hn : n ∈ field j) (t : ℕ) : n + t * M ∈ field j := by
  induction t with
  | zero => simpa using hn
  | succ t ih =>
    have := hstep _ ih
    rwa [Nat.succ_mul, ← Nat.add_assoc]

/-- **E6b, one direction.**  For every `M ≥ 1` and every `j`, field `j` is not closed under
`+ M`: some `n ∈ field j` has `n + M ∉ field j`. -/
theorem field_not_closed_add {j M : ℕ} (hM : 1 ≤ M) :
    ∃ n, n ∈ field j ∧ n + M ∉ field j := by
  by_contra h
  push Not at h
  have hall := field_add_mul_of_add h (5 ^ j) (five_pow_mem_field j) (5 ^ j)
  have e : 5 ^ j + 5 ^ j * M = 5 ^ j * (1 + M) := by ring
  rw [e] at hall
  exact mul_not_mem_field (five_pow_mem_field j) (by omega) hall

/-- **E6b.**  No field is periodic: for every `M ≥ 1` and every `j` there is `n` with
`n ∈ field j` and `n + M ∉ field j`, or the converse.  (The witness is always of the first
kind, `field_not_closed_add`; `j ≥ 1` is not needed.) -/
theorem field_not_periodic {j M : ℕ} (hM : 1 ≤ M) :
    ∃ n, (n ∈ field j ∧ n + M ∉ field j) ∨ (n ∉ field j ∧ n + M ∈ field j) := by
  obtain ⟨n, hn, hnM⟩ := field_not_closed_add (j := j) hM
  exact ⟨n, Or.inl ⟨hn, hnM⟩⟩

end Fields
