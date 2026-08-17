/-
Formalisation of the supply identity: root attribution partitions the
composites of a window.

The constructor ledger attributes every composite window member to its ROOT:
the gear `lpf m = minFac m`. Inside the window `(y, y * y)` the horizon
theorem guarantees the root is a prime strictly below `y`, so attribution is
a map from composites to gears `p < y`, and being a function it partitions
them. The supply identity follows: the number of composites equals the sum
over gears `p < y` of the size of `p`'s root class `R(p)`. In ledger terms
`Σ_q R(q) = C = 2N - P`: this file proves the partition equality
`C = Σ_q R(q)` and the bookkeeping form `members = primes + Σ_q R(q)`;
`C = 2N - P` is then arithmetic at the call site, where the members come in
`N` pairs.

The slot-level corollary comes from the slot cap: an odd member and its
partner `m + 2` never share a root, so the two kills of a double slot are
always supplied by distinct gears - the ledger is overlap-free at every
slot, not only in total.
-/

import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Horizon
import Layer

namespace Supply

/-- **Root attribution lands below the horizon.** Inside the window, a
composite's least factor is a prime strictly below `y` - the horizon theorem
hands a prime factor `p < y`, and the least factor is at most `p`. -/
theorem minFac_mem_gears {y m : ℕ} (hym : y < m) (hmyy : m < y * y)
    (hnp : ¬ m.Prime) : m.minFac ∈ (Finset.range y).filter Nat.Prime := by
  obtain ⟨p, hp, hpy, hpd⟩ := Horizon.exists_prime_factor_lt hym hmyy hnp
  have h1 : m ≠ 1 := by
    rintro rfl
    exact hp.ne_one (Nat.dvd_one.mp hpd)
  have hple : m.minFac ≤ p := Nat.minFac_le_of_dvd hp.two_le hpd
  rw [Finset.mem_filter, Finset.mem_range]
  exact ⟨lt_of_le_of_lt hple hpy, Nat.minFac_prime h1⟩

/-- **The supply identity, partition form.** For any finite set `S` of window
members, the composites of `S` are partitioned by their root: the composite
count equals the sum over gears `p < y` of the root class size
`R(p) = #{m ∈ S : m composite, minFac m = p}`. -/
theorem card_composites_eq_sum_roots {y : ℕ} (S : Finset ℕ)
    (hS : ∀ m ∈ S, y < m ∧ m < y * y) :
    (S.filter fun m => ¬ m.Prime).card =
      ∑ p ∈ (Finset.range y).filter Nat.Prime,
        (S.filter fun m => ¬ m.Prime ∧ m.minFac = p).card := by
  classical
  have H : ∀ m ∈ S.filter (fun m => ¬ m.Prime),
      m.minFac ∈ (Finset.range y).filter Nat.Prime := by
    intro m hm
    rw [Finset.mem_filter] at hm
    exact minFac_mem_gears (hS m hm.1).1 (hS m hm.1).2 hm.2
  rw [Finset.card_eq_sum_card_fiberwise H]
  refine Finset.sum_congr rfl fun p _ => ?_
  rw [Finset.filter_filter]

/-- **Ledger form: members = primes + supply.** `#S = P + Σ_p R(p)` - every
window member is either a prime or attributed to exactly one gear below the
horizon. -/
theorem card_eq_primes_add_sum_roots {y : ℕ} (S : Finset ℕ)
    (hS : ∀ m ∈ S, y < m ∧ m < y * y) :
    S.card = (S.filter Nat.Prime).card
      + ∑ p ∈ (Finset.range y).filter Nat.Prime,
          (S.filter fun m => ¬ m.Prime ∧ m.minFac = p).card := by
  classical
  rw [← card_composites_eq_sum_roots S hS,
    Finset.card_filter_add_card_filter_not Nat.Prime]

/-- **Distinct roots at a slot.** An odd `m` and its partner `m + 2` never
share a root: a common least factor would be an odd gear dividing both
members, against the slot cap. So a double slot's two kills are supplied by
distinct gears. -/
theorem roots_ne {m : ℕ} (h1 : 1 < m) (hodd : ¬ 2 ∣ m) :
    m.minFac ≠ (m + 2).minFac := by
  intro heq
  have hp : m.minFac.Prime := Nat.minFac_prime (by omega)
  have h2 : m.minFac ≠ 2 := fun h => hodd (h ▸ Nat.minFac_dvd m)
  have hq3 : 3 ≤ m.minFac := lt_of_le_of_ne hp.two_le (Ne.symm h2)
  have hd2 : m.minFac ∣ m + 2 := by rw [heq]; exact Nat.minFac_dvd (m + 2)
  exact Layer.slot_cap hq3 ⟨Nat.minFac_dvd m, hd2⟩

end Supply
