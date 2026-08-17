/-
Formalisation of the zero-slack census: the demand side of the ledger.

A slot `k` carries the pair `(6k - 1, 6k + 1)`. Over any finite set `T` of
slots the ledger counts, in the session's names:

  N  = #T            (slots)                     = `T.card`
  P  = prime members                             = `primesIn T`
  C  = composite members                         = `compsIn T`
  n0 = twin slots (0 composite members)          = `n0 T`
  n1 = fragile slots (exactly 1 composite)       = `n1 T`
  n2 = double slots (both members composite)     = `n2 T`

The census identities proved here:

  n0 + n1 + n2 = N        (slots partition by composite count)
  n1 + 2 n2 = C           (counting composite members slot by slot)
  P + C = 2 N             (each slot carries two members)
  P = n1 + 2 n0           (consequence)

and the ZERO-SLACK PINNING: under Condition X on `T` (no twin slot, n0 = 0)
the census has no freedom left -

  n1 = P   and   n2 = N - P.

Every root kill is load-bearing and the doubles count is pinned to the prime
census of the window. The `n0 = 0` hypothesis is exactly "no twin pair among
the slot pairs" (`n0_eq_zero_iff`), and the prefix form instantiates `T` at
`Finset.range t`. Everything is stated for an arbitrary Finset of slots, so
prefixes, windows, and layer bands all specialise it.
-/

import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

namespace Census

/-- Lower member of slot `k`. -/
def lo (k : ℕ) : ℕ := 6 * k - 1

/-- Upper member of slot `k`. -/
def hi (k : ℕ) : ℕ := 6 * k + 1

/-- Number of prime members of slot `k` (0, 1, or 2). -/
def slotPrimes (k : ℕ) : ℕ :=
  (if (lo k).Prime then 1 else 0) + (if (hi k).Prime then 1 else 0)

/-- Number of composite (non-prime) members of slot `k` (0, 1, or 2). -/
def slotComps (k : ℕ) : ℕ :=
  (if (lo k).Prime then 0 else 1) + (if (hi k).Prime then 0 else 1)

/-- `P`: prime members over the slot set. -/
def primesIn (T : Finset ℕ) : ℕ := ∑ k ∈ T, slotPrimes k

/-- `C`: composite members over the slot set. -/
def compsIn (T : Finset ℕ) : ℕ := ∑ k ∈ T, slotComps k

/-- `n0`: slots with no composite member - the twin slots. -/
def n0 (T : Finset ℕ) : ℕ := (T.filter fun k => slotComps k = 0).card

/-- `n1`: slots with exactly one composite member - the fragile slots. -/
def n1 (T : Finset ℕ) : ℕ := (T.filter fun k => slotComps k = 1).card

/-- `n2`: slots with both members composite - the double slots. -/
def n2 (T : Finset ℕ) : ℕ := (T.filter fun k => slotComps k = 2).card

/-- Each slot's two members split between primes and composites. -/
theorem slotPrimes_add_slotComps (k : ℕ) : slotPrimes k + slotComps k = 2 := by
  by_cases h1 : (lo k).Prime <;> by_cases h2 : (hi k).Prime <;>
    simp [slotPrimes, slotComps, h1, h2]

theorem slotComps_le_two (k : ℕ) : slotComps k ≤ 2 := by
  by_cases h1 : (lo k).Prime <;> by_cases h2 : (hi k).Prime <;>
    simp [slotComps, h1, h2]

/-- A slot has zero composite members exactly when it is a twin slot. -/
theorem slotComps_eq_zero_iff (k : ℕ) :
    slotComps k = 0 ↔ (lo k).Prime ∧ (hi k).Prime := by
  by_cases h1 : (lo k).Prime <;> by_cases h2 : (hi k).Prime <;>
    simp [slotComps, h1, h2]

/-- `n0 = 0` is Condition X on the slot set: no slot is a twin pair. -/
theorem n0_eq_zero_iff (T : Finset ℕ) :
    n0 T = 0 ↔ ∀ k ∈ T, ¬ ((lo k).Prime ∧ (hi k).Prime) := by
  rw [n0, Finset.card_eq_zero, Finset.filter_eq_empty_iff]
  constructor
  · intro h k hk hkp
    exact h hk ((slotComps_eq_zero_iff k).mpr hkp)
  · intro h k hk hc
    exact h k hk ((slotComps_eq_zero_iff k).mp hc)

/-- **Census partition:** `n0 + n1 + n2 = N`. Slots split by composite
count. -/
theorem census_partition (T : Finset ℕ) : n0 T + n1 T + n2 T = T.card := by
  classical
  have H : ∀ k ∈ T, slotComps k ∈ Finset.range 3 := fun k _ =>
    Finset.mem_range.mpr (by have := slotComps_le_two k; omega)
  rw [Finset.card_eq_sum_card_fiberwise H,
    Finset.sum_range_succ, Finset.sum_range_succ, Finset.sum_range_one]
  rfl

/-- **Composite supply by slot class:** `n1 + 2 n2 = C`. -/
theorem comps_eq (T : Finset ℕ) : compsIn T = n1 T + 2 * n2 T := by
  classical
  have H : ∀ k ∈ T, slotComps k ∈ Finset.range 3 := fun k _ =>
    Finset.mem_range.mpr (by have := slotComps_le_two k; omega)
  have inner : ∀ b, (∑ k ∈ T.filter fun k => slotComps k = b, slotComps k)
      = (T.filter fun k => slotComps k = b).card * b := fun b =>
    Finset.sum_const_nat fun k hk => (Finset.mem_filter.mp hk).2
  show (∑ k ∈ T, slotComps k) = n1 T + 2 * n2 T
  rw [← Finset.sum_fiberwise_of_maps_to H slotComps,
    Finset.sum_range_succ, Finset.sum_range_succ, Finset.sum_range_one,
    inner 0, inner 1, inner 2]
  simp [n1, n2, mul_comm]

/-- **Two members per slot:** `P + C = 2 N`. -/
theorem primes_add_comps (T : Finset ℕ) :
    primesIn T + compsIn T = 2 * T.card := by
  show (∑ k ∈ T, slotPrimes k) + (∑ k ∈ T, slotComps k) = 2 * T.card
  rw [← Finset.sum_add_distrib,
    Finset.sum_const_nat fun k _ => slotPrimes_add_slotComps k,
    Nat.mul_comm]

/-- **Prime census by slot class:** `P = n1 + 2 n0`. -/
theorem primes_eq (T : Finset ℕ) : primesIn T = n1 T + 2 * n0 T := by
  have h1 := census_partition T
  have h2 := comps_eq T
  have h3 := primes_add_comps T
  omega

/-- **Zero-slack pinning.** Under Condition X (`n0 = 0`) the census has no
freedom: `n1 = P` and `n2 = N - P`. Every fragile slot's kill is load-bearing
and the doubles count is pinned to the prime census. -/
theorem census_pinned (T : Finset ℕ) (h0 : n0 T = 0) :
    n1 T = primesIn T ∧ n2 T = T.card - primesIn T := by
  have h1 := census_partition T
  have h2 := comps_eq T
  have h3 := primes_add_comps T
  omega

/-- The additive form of the pinning, subtraction-free for composition:
`n2 + P = N`. -/
theorem census_pinned_add (T : Finset ℕ) (h0 : n0 T = 0) :
    n2 T + primesIn T = T.card := by
  have h1 := census_partition T
  have h2 := comps_eq T
  have h3 := primes_add_comps T
  omega

/-- **Prefix form.** Condition X on the prefix `[0, t)` pins the census of
every such prefix: `n1(t) = P(t)` and `n2(t) = t - P(t)`. -/
theorem census_pinned_prefix (t : ℕ)
    (hX : ∀ k < t, ¬ ((lo k).Prime ∧ (hi k).Prime)) :
    n1 (Finset.range t) = primesIn (Finset.range t) ∧
      n2 (Finset.range t) = t - primesIn (Finset.range t) := by
  have h0 : n0 (Finset.range t) = 0 :=
    (n0_eq_zero_iff _).mpr fun k hk => hX k (Finset.mem_range.mp hk)
  have h := census_pinned _ h0
  rwa [Finset.card_range] at h

end Census
