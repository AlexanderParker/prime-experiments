/-
THEOREM (E) AT ONE STEP (E6), THE EFFECTIVE-MACHINE THEOREM (E) IN ITS TRUE FORM, AND
PREFIX INHERITANCE (E7)  (Formalist, round 39).

Source: `research/proof/lengthen_never_precede.md` (Setup; 4.1 E6 with its proof; 4.2 E7;
"What is new" 1 and 2) and `research/proof/position_frontier.md` (theorem (E): "for every
column `k` with `6k - 1 > q`, `k` is blocked under `{5..q}` iff it is blocked under
`{5..floor(sqrt(6k+1))}`").

THE OBJECT.  The engine `{5..q}` in the anchored column coordinate: column `k` is the
pair `(6k - 1, 6k + 1)`, a gear `p` (a prime with `5 <= p <= q`) strikes `k` iff it
divides a member, and `k` is BLOCKED under `{5..q}` iff some gear strikes it
(`Blocked q k`).  `W q' = (q'^2 - 1)/6` is the top column of the window of `{5..q}` when
`q'` is the next prime (`6 W + 1 = q'^2`, `six_mul_W_add_one`).  Columns are `ℕ` and
`6k - 1` is truncated subtraction, so column `0` is an artefact (`6·0 - 1 = 0`, divisible
by everything); every statement carries `1 <= k`.

The corpus's column predicates (`TopMachineWheel.ColOpen` over `ℤ` with a `Finset` of
gears; `Census.lo`/`Census.hi` as bare members) are tied to the top machine's wheels and
carry no "next prime" structure, so the predicate is defined afresh here in `ℕ`, in its
own namespace, with the gear set `{5..q}` written as the bound `5 <= p <= q` on primes.
Nothing is imported from the corpus; only mathlib.

WHAT THE PROOFS ACTUALLY NEED.  "`q'` is the next prime after `q`" enters only as the
gap condition `hgap : ∀ p, p.Prime → p < q' → p ≤ q` (no prime strictly between), plus
`q ≤ q'` (for E6's easy direction) or `q < q'` (for the exception set: `q'^2` and `q'`
must not be struck by `{5..q}`).  `q` itself is never assumed prime.  Theorem (E) in its
general form (`blocked_iff_of_sqrt_le`) needs no `q'` at all - only `q < 6k - 1`; the
document's form with `floor(sqrt(6k+1))` on the right is TRUE ONLY INSIDE THE PREFIX
`6k + 1 < q'^2` (`blocked_iff_sqrt`), and `E_needs_prefix` is the refuting instance
`q = 5, k = 8` (members `47, 49`, `sqrt 49 = 7` blocks, `{5}` does not) for the
statement without that hypothesis.

Zero sorries; no `native_decide`, no `decide`, no `Lean.ofReduceBool`.
-/
import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Data.Nat.Sqrt
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum.Prime

namespace OneStepE

/-! ## The objects -/

/-- `SmallFactor q n`: some gear of the engine `{5..q}` (a prime `p` with `5 ≤ p ≤ q`)
divides `n`. -/
def SmallFactor (q n : ℕ) : Prop := ∃ p, p.Prime ∧ 5 ≤ p ∧ p ≤ q ∧ p ∣ n

/-- Column `k = (6k - 1, 6k + 1)` is blocked under the engine `{5..q}`: a gear strikes
one of its members. -/
def Blocked (q k : ℕ) : Prop := SmallFactor q (6 * k - 1) ∨ SmallFactor q (6 * k + 1)

/-- The square column `W(q) = (q'^2 - 1)/6`, indexed by the next prime `q'`: the top
column of the window of `{5..q}`, whose upper member is `q'^2`. -/
def W (q' : ℕ) : ℕ := (q' ^ 2 - 1) / 6

/-- `MaxRun q a b`: the columns `a..b` are blocked under `{5..q}` and the two neighbours
`a - 1`, `b + 1` are open (a maximal blocked run, read as first column / last column). -/
def MaxRun (q a b : ℕ) : Prop :=
  a ≤ b ∧ (∀ j, a ≤ j → j ≤ b → Blocked q j) ∧ ¬ Blocked q (a - 1) ∧ ¬ Blocked q (b + 1)

/-! ## Monotonicity: blocking only grows up the ladder -/

theorem smallFactor_mono {q r n : ℕ} (h : q ≤ r) : SmallFactor q n → SmallFactor r n
  | ⟨p, hp, h5, hq, hd⟩ => ⟨p, hp, h5, le_trans hq h, hd⟩

theorem blocked_mono {q r k : ℕ} (h : q ≤ r) : Blocked q k → Blocked r k
  | Or.inl h1 => Or.inl (smallFactor_mono h h1)
  | Or.inr h1 => Or.inr (smallFactor_mono h h1)

/-! ## Arithmetic of members coprime to 6 -/

/-- A prime dividing a number coprime to 6 is at least 5. -/
theorem five_le_of_dvd {p n : ℕ} (hp : p.Prime) (h2 : ¬ 2 ∣ n) (h3 : ¬ 3 ∣ n)
    (hd : p ∣ n) : 5 ≤ p := by
  have h2p := hp.two_le
  by_contra h
  have hc : p = 2 ∨ p = 3 ∨ p = 4 := by omega
  rcases hc with rfl | rfl | rfl
  · exact h2 hd
  · exact h3 hd
  · rcases hp.eq_one_or_self_of_dvd 2 (by omega) with h | h <;> omega

/-- A composite number coprime to 6 has a prime factor `p ≥ 5` with `p * p ≤ n`
(its least prime factor). -/
theorem exists_small_prime_factor {n : ℕ} (h1 : 1 < n) (hn : ¬ n.Prime) (h2 : ¬ 2 ∣ n)
    (h3 : ¬ 3 ∣ n) : ∃ p, p.Prime ∧ 5 ≤ p ∧ p * p ≤ n ∧ p ∣ n := by
  have hp : (Nat.minFac n).Prime := Nat.minFac_prime (by omega)
  have hd : Nat.minFac n ∣ n := Nat.minFac_dvd n
  refine ⟨n.minFac, hp, five_le_of_dvd hp h2 h3 hd, ?_, hd⟩
  have := Nat.minFac_sq_le_self (by omega) hn
  rwa [sq] at this

/-- `p * p ≤ n < m^2` forces `p < m`. -/
theorem lt_of_mul_self_le_of_lt_sq {p n m : ℕ} (hpp : p * p ≤ n) (hlt : n < m ^ 2) :
    p < m := by
  rcases Nat.lt_or_ge p m with h | h
  · exact h
  · have := Nat.mul_self_le_mul_self h
    rw [sq] at hlt
    exact absurd (lt_of_le_of_lt (le_trans this hpp) hlt) (lt_irrefl _)

/-- Reduction (R) on one member: for `n` coprime to 6 with `q < n < q'^2` (no prime
strictly between `q` and `q'`), the engine `{5..q}` strikes `n` iff `n` is not prime. -/
theorem smallFactor_iff_not_prime {q q' n : ℕ} (hgap : ∀ p, p.Prime → p < q' → p ≤ q)
    (h2 : ¬ 2 ∣ n) (h3 : ¬ 3 ∣ n) (hqn : q < n) (hlt : n < q' ^ 2) (h1 : 1 < n) :
    SmallFactor q n ↔ ¬ n.Prime := by
  constructor
  · rintro ⟨p, hp, -, hpq, hd⟩ hn
    have := hp.two_le
    rcases hn.eq_one_or_self_of_dvd p hd with h | h <;> omega
  · intro hn
    obtain ⟨p, hp, h5, hpp, hd⟩ := exists_small_prime_factor h1 hn h2 h3
    exact ⟨p, hp, h5, hgap p hp (lt_of_mul_self_le_of_lt_sq hpp hlt), hd⟩

/-- A member below `q'`, coprime to 6 and at least 5, is struck by `{5..q}`: it is a
prime `≤ q` (a gear) or composite with a prime factor `≤ q`. -/
theorem smallFactor_of_lt {q q' n : ℕ} (hgap : ∀ p, p.Prime → p < q' → p ≤ q)
    (h2 : ¬ 2 ∣ n) (h3 : ¬ 3 ∣ n) (h5 : 5 ≤ n) (hlt : n < q') : SmallFactor q n := by
  by_cases hn : n.Prime
  · exact ⟨n, hn, h5, hgap n hn hlt, dvd_rfl⟩
  · obtain ⟨p, hp, hp5, hpp, hd⟩ := exists_small_prime_factor (by omega) hn h2 h3
    refine ⟨p, hp, hp5, hgap p hp ?_, hd⟩
    have : p ≤ p * p := Nat.le_mul_self p
    omega

/-- Every column whose lower member is below `q'` is blocked under `{5..q}`: the initial
run of `{5..q}` reaches at least to the home column of `q'`. -/
theorem blocked_of_lo_lt {q q' j : ℕ} (hgap : ∀ p, p.Prime → p < q' → p ≤ q)
    (hj : 1 ≤ j) (hlt : 6 * j - 1 < q') : Blocked q j :=
  Or.inl (smallFactor_of_lt hgap (by omega) (by omega) (by omega) hlt)

/-- The identity `h(q') ≤ d_0(M)`: an open column `j ≥ 1` of `{5..q}` has `6j - 1 ≥ q'`. -/
theorem le_lo_of_open {q q' j : ℕ} (hgap : ∀ p, p.Prime → p < q' → p ≤ q)
    (hj : 1 ≤ j) (h : ¬ Blocked q j) : q' ≤ 6 * j - 1 := by
  by_contra hc
  exact h (blocked_of_lo_lt hgap hj (by omega))

/-! ## Theorem (E), the effective-machine theorem, in its true general form -/

/-- **Theorem (E), general form.**  For a column above the engine (`6k - 1 > q`), the
gears up to `q` block `k` iff the gears up to any `r` with `sqrt(6k+1) ≤ r ≤ q` do: a
gear `p ≤ q` dividing a member `n > q` shows `n` composite, and the least prime factor of
`n` is `≥ 5` and `≤ sqrt n ≤ sqrt(6k+1)`.  No next prime is involved. -/
theorem blocked_iff_of_sqrt_le {q r k : ℕ} (hr : Nat.sqrt (6 * k + 1) ≤ r) (hrq : r ≤ q)
    (hqk : q < 6 * k - 1) : Blocked q k ↔ Blocked r k := by
  refine ⟨fun h => ?_, blocked_mono hrq⟩
  have key : ∀ n, (n = 6 * k - 1 ∨ n = 6 * k + 1) → SmallFactor q n → SmallFactor r n := by
    rintro n hn ⟨p, hp, h5, hpq, hd⟩
    have h2 : ¬ 2 ∣ n := by rcases hn with rfl | rfl <;> omega
    have h3 : ¬ 3 ∣ n := by rcases hn with rfl | rfl <;> omega
    have h1 : 1 < n := by rcases hn with rfl | rfl <;> omega
    have hnle : n ≤ 6 * k + 1 := by rcases hn with rfl | rfl <;> omega
    have hqn : q < n := by rcases hn with rfl | rfl <;> omega
    have hpn : ¬ n.Prime := by
      intro hprime
      have := hp.two_le
      rcases hprime.eq_one_or_self_of_dvd p hd with h | h <;> omega
    obtain ⟨p', hp', h5', hpp, hd'⟩ := exists_small_prime_factor h1 hpn h2 h3
    exact ⟨p', hp', h5', le_trans (Nat.le_sqrt.mpr (le_trans hpp hnle)) hr, hd'⟩
  rcases h with h | h
  · exact Or.inl (key _ (Or.inl rfl) h)
  · exact Or.inr (key _ (Or.inr rfl) h)

/-- Theorem (E) with the effective machine `{5..min q (sqrt(6k+1))}`: the version that
holds for every `q` and every column above it. -/
theorem blocked_iff_min_sqrt {q k : ℕ} (hqk : q < 6 * k - 1) :
    Blocked q k ↔ Blocked (min q (Nat.sqrt (6 * k + 1))) k := by
  rcases le_total (Nat.sqrt (6 * k + 1)) q with h | h
  · rw [min_eq_right h]; exact blocked_iff_of_sqrt_le le_rfl h hqk
  · rw [min_eq_left h]

/-- **Theorem (E) as position_frontier.md states it**, with its true hypothesis made
explicit: inside the prefix (`6k + 1 < q'^2`, `q'` the next prime after `q`) and above the
engine (`6k - 1 > q`), `k` is blocked under `{5..q}` iff under `{5..floor(sqrt(6k+1))}`. -/
theorem blocked_iff_sqrt {q q' k : ℕ} (hgap : ∀ p, p.Prime → p < q' → p ≤ q)
    (hqk : q < 6 * k - 1) (hlt : 6 * k + 1 < q' ^ 2) :
    Blocked q k ↔ Blocked (Nat.sqrt (6 * k + 1)) k := by
  rw [blocked_iff_min_sqrt hqk]
  constructor
  · exact blocked_mono (min_le_right _ _)
  · intro h
    have hs : Nat.sqrt (6 * k + 1) < q' := Nat.sqrt_lt'.mpr hlt
    rcases h with ⟨p, hp, h5, hpr, hd⟩ | ⟨p, hp, h5, hpr, hd⟩
    · exact Or.inl ⟨p, hp, h5, le_min (hgap p hp (by omega)) hpr, hd⟩
    · exact Or.inr ⟨p, hp, h5, le_min (hgap p hp (by omega)) hpr, hd⟩

/-- **The prefix hypothesis cannot be dropped.**  At `q = 5`, `k = 8` (members `47, 49`,
`6k - 1 = 47 > q`): `sqrt(6k+1) = 7`, and `{5..7}` blocks the column (`7 | 49`) while
`{5}` does not.  So (E) as written in position_frontier.md is a statement about the
prefix `6k + 1 < q'^2`, not about every column above the engine. -/
theorem E_needs_prefix :
    Blocked 7 8 ∧ ¬ Blocked 5 8 ∧ Nat.sqrt (6 * 8 + 1) = 7 := by
  refine ⟨Or.inr ⟨7, by norm_num, by omega, le_rfl, by omega⟩, ?_, ?_⟩
  · rintro (⟨p, hp, h5, hpq, hd⟩ | ⟨p, hp, h5, hpq, hd⟩)
    · have : p = 5 := by omega
      subst this; omega
    · have : p = 5 := by omega
      subst this; omega
  · have : (6 * 8 + 1 : ℕ) = 7 ^ 2 := by norm_num
    rw [this]; exact Nat.sqrt_eq' 7

/-! ## E6: theorem (E) at one step -/

/-- **E6, the one-step form.**  For `q'` a prime with no prime strictly between `q` and
`q'` (`q ≤ q'`), and a column `1 ≤ k` below the square column (`6k + 1 < q'^2`): `k` is
blocked under `{5..q'}` iff it is blocked under `{5..q}` or `q'` is itself a member of
`k`.  The new gear's only new strikes below its square are at its home column. -/
theorem blocked_succ_iff {q q' k : ℕ} (hq' : q'.Prime) (hle : q ≤ q')
    (hgap : ∀ p, p.Prime → p < q' → p ≤ q) (hk : 1 ≤ k) (hlt : 6 * k + 1 < q' ^ 2) :
    Blocked q' k ↔ Blocked q k ∨ 6 * k - 1 = q' ∨ 6 * k + 1 = q' := by
  constructor
  · intro h
    have key : ∀ n, (n = 6 * k - 1 ∨ n = 6 * k + 1) → SmallFactor q' n →
        SmallFactor q n ∨ n = q' := by
      rintro n hn ⟨p, hp, h5, hpq, hd⟩
      by_cases hpq' : p ≤ q
      · exact Or.inl ⟨p, hp, h5, hpq', hd⟩
      have hpe : p = q' := by
        by_contra hne
        exact hpq' (hgap p hp (lt_of_le_of_ne hpq hne))
      by_cases hnq : n = q'
      · exact Or.inr hnq
      left
      have h2 : ¬ 2 ∣ n := by rcases hn with rfl | rfl <;> omega
      have h3 : ¬ 3 ∣ n := by rcases hn with rfl | rfl <;> omega
      have h1 : 1 < n := by rcases hn with rfl | rfl <;> omega
      have hnle : n ≤ 6 * k + 1 := by rcases hn with rfl | rfl <;> omega
      have hnlt : n < q' ^ 2 := lt_of_le_of_lt hnle hlt
      have hnprime : ¬ n.Prime := fun hnpr => by
        have := hp.two_le
        rcases hnpr.eq_one_or_self_of_dvd p hd with h | h <;> omega
      obtain ⟨r, hr, hr5, hrr, hrd⟩ := exists_small_prime_factor h1 hnprime h2 h3
      exact ⟨r, hr, hr5, hgap r hr (lt_of_mul_self_le_of_lt_sq hrr hnlt), hrd⟩
    rcases h with h | h
    · rcases key _ (Or.inl rfl) h with h' | h'
      · exact Or.inl (Or.inl h')
      · exact Or.inr (Or.inl h')
    · rcases key _ (Or.inr rfl) h with h' | h'
      · exact Or.inl (Or.inr h')
      · exact Or.inr (Or.inr h')
  · rintro (h | h | h)
    · exact blocked_mono hle h
    · exact Or.inl ⟨q', hq', by omega, le_rfl, by rw [h]⟩
    · exact Or.inr ⟨q', hq', by omega, le_rfl, by rw [h]⟩

/-! ## The two boundary columns: the home column and the square column -/

theorem coprime_six_of_prime {q' : ℕ} (hq' : q'.Prime) (h5 : 5 ≤ q') :
    ¬ 2 ∣ q' ∧ ¬ 3 ∣ q' := by
  constructor
  · intro h; rcases hq'.eq_one_or_self_of_dvd 2 h with h | h <;> omega
  · intro h; rcases hq'.eq_one_or_self_of_dvd 3 h with h | h <;> omega

/-- `q'^2 ≡ 1 (mod 6)` for `q'` coprime to 6. -/
theorem exists_sq_eq_six {q' : ℕ} (h2 : ¬ 2 ∣ q') (h3 : ¬ 3 ∣ q') :
    ∃ t, q' ^ 2 = 6 * t + 1 := by
  obtain ⟨m, hm⟩ : ∃ m, q' = 6 * m + 1 ∨ q' = 6 * m + 5 := ⟨q' / 6, by omega⟩
  rcases hm with rfl | rfl
  · exact ⟨6 * m ^ 2 + 2 * m, by ring⟩
  · exact ⟨6 * m ^ 2 + 10 * m + 4, by ring⟩

/-- The square column carries `q'^2` as its upper member. -/
theorem six_mul_W_add_one {q' : ℕ} (h2 : ¬ 2 ∣ q') (h3 : ¬ 3 ∣ q') :
    6 * W q' + 1 = q' ^ 2 := by
  obtain ⟨t, ht⟩ := exists_sq_eq_six h2 h3
  unfold W; rw [ht]; omega

/-- The new gear always strikes its square column. -/
theorem blocked_succ_W {q' : ℕ} (hq' : q'.Prime) (h5 : 5 ≤ q') : Blocked q' (W q') := by
  obtain ⟨h2, h3⟩ := coprime_six_of_prime hq' h5
  exact Or.inr ⟨q', hq', h5, le_rfl, by
    rw [six_mul_W_add_one h2 h3]; exact dvd_pow_self q' two_ne_zero⟩

/-- The square column under the old engine: blocked iff `q'^2 - 2` is composite
(so it is NEW for `q'` iff `q'^2 - 2` is prime, the set `N_2` of E6). -/
theorem blocked_W_iff {q q' : ℕ} (hq' : q'.Prime) (h5 : 5 ≤ q') (hlt : q < q')
    (hgap : ∀ p, p.Prime → p < q' → p ≤ q) :
    Blocked q (W q') ↔ ¬ (q' ^ 2 - 2).Prime := by
  obtain ⟨h2, h3⟩ := coprime_six_of_prime hq' h5
  obtain ⟨t, ht⟩ := exists_sq_eq_six h2 h3
  have hW : W q' = t := by unfold W; rw [ht]; omega
  have h5q : 5 * q' ≤ q' ^ 2 := by rw [sq]; exact Nat.mul_le_mul_right q' h5
  have hsq : ¬ SmallFactor q (6 * t + 1) := by
    rintro ⟨p, hp, -, hpq, hd⟩
    rw [← ht] at hd
    have := (Nat.prime_dvd_prime_iff_eq hp hq').mp (hp.dvd_of_dvd_pow hd)
    omega
  have hn : q' ^ 2 - 2 = 6 * t - 1 := by omega
  unfold Blocked
  rw [hW, hn]
  constructor
  · rintro (h | h)
    · exact (smallFactor_iff_not_prime hgap (by omega) (by omega) (by omega) (by omega)
        (by omega)).mp h
    · exact absurd h hsq
  · intro h
    exact Or.inl ((smallFactor_iff_not_prime hgap (by omega) (by omega) (by omega) (by omega)
      (by omega)).mpr h)

/-- The home column of a prime `q' = 6k - 1` under the old engine: open iff `q' + 2` is
prime (so it is NEW for `q'` iff `(q', q' + 2)` is a twin pair, the set `N_1` of E6).
The member `q'` itself is never struck by `{5..q}`; the other member `q' + 2` is struck
iff it is composite (reduction (R)). -/
theorem home_minus_open_iff {q q' k : ℕ} (hq' : q'.Prime) (h5 : 5 ≤ q') (hlt : q < q')
    (hgap : ∀ p, p.Prime → p < q' → p ≤ q) (hk : 6 * k - 1 = q') :
    ¬ Blocked q k ↔ (q' + 2).Prime := by
  have hsf : ¬ SmallFactor q (6 * k - 1) := by
    rintro ⟨p, hp, -, hpq, hd⟩
    rw [hk] at hd
    have := (Nat.prime_dvd_prime_iff_eq hp hq').mp hd
    omega
  have h5q : 5 * q' ≤ q' ^ 2 := by rw [sq]; exact Nat.mul_le_mul_right q' h5
  have hiff := smallFactor_iff_not_prime (q := q) (q' := q') (n := 6 * k + 1) hgap
    (by omega) (by omega) (by omega) (by omega) (by omega)
  have he : 6 * k + 1 = q' + 2 := by omega
  unfold Blocked
  rw [not_or, hiff, not_not, he]
  exact ⟨fun h => h.2, fun h => ⟨hsf, h⟩⟩

/-- The home column of a prime `q' = 6k + 1` is already blocked under `{5..q}`: its
other member `q' - 2 < q'` is a gear or composite.  So a home column is new only for
`q' ≡ 5 (mod 6)`. -/
theorem home_plus_blocked {q q' k : ℕ} (hgap : ∀ p, p.Prime → p < q' → p ≤ q)
    (hk : 6 * k + 1 = q') (h7 : 7 ≤ q') : Blocked q k :=
  blocked_of_lo_lt hgap (by omega) (by omega)

/-- When the home column of `q' = 6k - 1` is new (`q' + 2` prime), it is `d_0(M)`: the
least open column of `{5..q}`.  (The second clause needs neither primality of `q' + 2`
nor of `q'`.) -/
theorem home_isLeast_open {q q' k : ℕ} (hq' : q'.Prime) (h5 : 5 ≤ q') (hlt : q < q')
    (hgap : ∀ p, p.Prime → p < q' → p ≤ q) (hk : 6 * k - 1 = q') (htw : (q' + 2).Prime) :
    ¬ Blocked q k ∧ ∀ j, 1 ≤ j → j < k → Blocked q j :=
  ⟨(home_minus_open_iff hq' h5 hlt hgap hk).mpr htw,
   fun j hj hjk => blocked_of_lo_lt hgap hj (by omega)⟩

/-! ## E6 with its exact exception set `N_1 ∪ N_2` -/

/-- **E6, the exception set.**  On the prefix `[1, W]` the columns blocked by `{5..q'}`
and not by `{5..q}` are exactly: the home column `(q' + 1)/6` when `(q', q' + 2)` is a
twin pair (`N_1`), and the square column `W` when `q'^2 - 2` is prime (`N_2`).  Nothing
else in `q'^2/6` columns. -/
theorem new_iff {q q' k : ℕ} (hq' : q'.Prime) (h5 : 5 ≤ q') (hlt : q < q')
    (hgap : ∀ p, p.Prime → p < q' → p ≤ q) (hk : 1 ≤ k) (hkW : k ≤ W q') :
    (Blocked q' k ∧ ¬ Blocked q k) ↔
      (6 * k - 1 = q' ∧ (q' + 2).Prime) ∨ (k = W q' ∧ (q' ^ 2 - 2).Prime) := by
  obtain ⟨h2, h3⟩ := coprime_six_of_prime hq' h5
  have hW := six_mul_W_add_one h2 h3
  rcases Nat.lt_or_eq_of_le hkW with hklt | hkeq
  · have hlt' : 6 * k + 1 < q' ^ 2 := by omega
    have hkW' : k ≠ W q' := ne_of_lt hklt
    rw [blocked_succ_iff hq' hlt.le hgap hk hlt']
    constructor
    · rintro ⟨h | h | h, hnb⟩
      · exact absurd h hnb
      · exact Or.inl ⟨h, (home_minus_open_iff hq' h5 hlt hgap h).mp hnb⟩
      · exact absurd (home_plus_blocked hgap h (by omega)) hnb
    · rintro (⟨h, hp⟩ | ⟨h, -⟩)
      · exact ⟨Or.inr (Or.inl h), (home_minus_open_iff hq' h5 hlt hgap h).mpr hp⟩
      · exact absurd h hkW'
  · subst hkeq
    rw [blocked_W_iff hq' h5 hlt hgap]
    constructor
    · rintro ⟨-, h⟩
      exact Or.inr ⟨rfl, not_not.mp h⟩
    · rintro (⟨h, -⟩ | ⟨-, h⟩)
      · exfalso
        have h5q : 5 * q' ≤ q' ^ 2 := by rw [sq]; exact Nat.mul_le_mul_right q' h5
        omega
      · exact ⟨blocked_succ_W hq' h5, not_not.mpr h⟩

/-- E6 counted: at most the two named columns are new; in particular a column of the
prefix that is neither the home column nor the square column is new for nobody. -/
theorem not_new_of_ne {q q' k : ℕ} (hq' : q'.Prime) (h5 : 5 ≤ q') (hlt : q < q')
    (hgap : ∀ p, p.Prime → p < q' → p ≤ q) (hk : 1 ≤ k) (hkW : k ≤ W q')
    (h1 : 6 * k - 1 ≠ q') (h2 : k ≠ W q') : ¬ (Blocked q' k ∧ ¬ Blocked q k) := by
  intro h
  rcases (new_iff hq' h5 hlt hgap hk hkW).mp h with ⟨h', -⟩ | ⟨h', -⟩
  · exact h1 h'
  · exact h2 h'

/-! ## E7: prefix inheritance -/

/-- **E7, pointwise.**  On `[1, W]` minus the two columns `(q' + 1)/6` and `W`, the
blocked predicates of `{5..q}` and `{5..q'}` agree. -/
theorem blocked_succ_iff_of_ne {q q' k : ℕ} (hq' : q'.Prime) (h5 : 5 ≤ q') (hlt : q < q')
    (hgap : ∀ p, p.Prime → p < q' → p ≤ q) (hk : 1 ≤ k) (hkW : k ≤ W q')
    (h1 : 6 * k - 1 ≠ q') (h2 : k ≠ W q') : Blocked q' k ↔ Blocked q k :=
  ⟨fun h => by
    by_contra hn
    exact not_new_of_ne hq' h5 hlt hgap hk hkW h1 h2 ⟨h, hn⟩,
   blocked_mono hlt.le⟩

/-- E7, pointwise, in the sharper form: the predicates agree at every column of `[1, W]`
that is not in `N_1 ∪ N_2` (the home column is exempt when `q' + 2` is composite, the
square column when `q'^2 - 2` is composite). -/
theorem blocked_succ_iff_of_not_new {q q' k : ℕ} (hq' : q'.Prime) (h5 : 5 ≤ q')
    (hlt : q < q') (hgap : ∀ p, p.Prime → p < q' → p ≤ q) (hk : 1 ≤ k) (hkW : k ≤ W q')
    (h1 : ¬ (6 * k - 1 = q' ∧ (q' + 2).Prime)) (h2 : ¬ (k = W q' ∧ (q' ^ 2 - 2).Prime)) :
    Blocked q' k ↔ Blocked q k :=
  ⟨fun h => by
    by_contra hn
    rcases (new_iff hq' h5 hlt hgap hk hkW).mp ⟨h, hn⟩ with h' | h'
    · exact h1 h'
    · exact h2 h',
   blocked_mono hlt.le⟩

/-- **E7, maximal runs.**  Every maximal blocked run of `{5..q'}` lying inside
`[2, W - 1]` (it does not contain column 1 and does not reach the square column) is a
maximal blocked run of `{5..q}`, same first and last column.  The home column cannot lie
in such a run: everything below it is blocked under `{5..q}`, so a run through it starts
at column 1. -/
theorem maxRun_succ {q q' a b : ℕ} (hq' : q'.Prime) (h5 : 5 ≤ q') (hlt : q < q')
    (hgap : ∀ p, p.Prime → p < q' → p ≤ q) (ha : 2 ≤ a) (hb : b + 1 ≤ W q')
    (h : MaxRun q' a b) : MaxRun q a b := by
  obtain ⟨hab, hblk, hopen1, hopen2⟩ := h
  refine ⟨hab, fun j hj1 hj2 => ?_, fun hc => hopen1 (blocked_mono hlt.le hc),
    fun hc => hopen2 (blocked_mono hlt.le hc)⟩
  by_cases hhome : 6 * j - 1 = q'
  · exfalso
    exact hopen1 (blocked_mono hlt.le (blocked_of_lo_lt hgap (by omega) (by omega)))
  · exact (blocked_succ_iff_of_ne hq' h5 hlt hgap (by omega) (by omega) hhome
      (by omega)).mp (hblk j hj1 hj2)

/-- E7, the converse: a maximal run of `{5..q}` inside `[2, W - 1]` whose two neighbours
are not the home column stays a maximal run of `{5..q'}` (the neighbour `b + 1 ≤ W` may
be the square column: then it must not be new, which is what `hb'` excludes). -/
theorem maxRun_succ_of {q q' a b : ℕ} (hq' : q'.Prime) (h5 : 5 ≤ q') (hlt : q < q')
    (hgap : ∀ p, p.Prime → p < q' → p ≤ q) (ha : 2 ≤ a) (hb : b + 1 ≤ W q')
    (ha' : 6 * (a - 1) - 1 ≠ q') (hb' : 6 * (b + 1) - 1 ≠ q') (hbW : b + 1 ≠ W q')
    (h : MaxRun q a b) : MaxRun q' a b := by
  obtain ⟨hab, hblk, hopen1, hopen2⟩ := h
  refine ⟨hab, fun j hj1 hj2 => blocked_mono hlt.le (hblk j hj1 hj2), ?_, ?_⟩
  · intro hc
    exact hopen1 ((blocked_succ_iff_of_ne hq' h5 hlt hgap (by omega) (by omega) ha'
      (by omega)).mp hc)
  · intro hc
    exact hopen2 ((blocked_succ_iff_of_ne hq' h5 hlt hgap (by omega) (by omega) hb'
      hbW).mp hc)

end OneStepE
