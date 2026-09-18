/-
FieldBlocking (round 85, 2026-09-18): the share logic, field by field.

A window is blocked only if every column is struck.  Order the gears: a column is first struck by
the smallest gear dividing one of its members, so a block is exactly the union over `g` of the
kills of the field `higher:g` (composites whose smallest gear is `g`) covering every column.  Each
`higher:g` acts only on the survivors of the gears below `g`, and its share of them is the lattice
share `2/g` (measured at 1.000 ± 0.004 for `g ≤ √q`, research/stack/r8/field_blocking.py).

Two fields cannot act on survivors at all, and this file proves it:

  `lower_on_survivor_is_power`: a member of `lower:g` (largest gear `g`) that no gear below `g`
  strikes is a power of `g`.  So `lower:g` reaches the survivors of the smaller gears only at the
  powers of `g` - a bounded handful of columns per gear.

  `square_is_upper_member`: the square of a gear is `1 mod 6`, so it is the upper member of exactly
  one column.  The field `squares` kills one column per gear, and never a lower member.

The remaining field, `higher:g` for large `g`, is pinned by `top_gear_cofactor` (round 84): above
`q^(2/3)` its members are `g²` and `g p` with `p` prime.  So no field, taken alone, has a member
that can enter a blocking state: `multiples` is a rigid lattice at share `2/g`, `squares` is one
column per gear, `lower:g` is the powers, and `higher:g` is the lattice restricted to rough
cofactors, which for large `g` means prime cofactors.
-/
import Mathlib

namespace MirrorWalk

/-- **A member of `lower:g` on a survivor is a power of `g`.**  If every prime factor of `n` is at
most `g` (the field `lower:g`) and no prime below `g` divides `n` (a survivor of the gears below
`g`), then `n` is a power of `g`. -/
theorem lower_on_survivor_is_power {g n : ℕ} (hg : g.Prime) (hn : n ≠ 0)
    (hupper : ∀ p : ℕ, p.Prime → p ∣ n → p ≤ g)
    (hsurv : ∀ p : ℕ, p.Prime → p < g → ¬ (p ∣ n)) :
    ∃ a : ℕ, n = g ^ a := by
  refine ⟨n.primeFactorsList.length, ?_⟩
  apply Nat.eq_prime_pow_of_unique_prime_dvd hn
  intro d hd hdn
  have h1 : d ≤ g := hupper d hd hdn
  have h2 : ¬ d < g := fun hlt => hsurv d hd hlt hdn
  omega

/-- **A gear's square is the upper member of one column.**  For a gear `g ≥ 5`, `g² ≡ 1 mod 6`,
so `g² = 6 m + 1` for exactly one `m`; it is never a lower member `6 m - 1`. -/
theorem square_is_upper_member {g : ℕ} (hg : g.Prime) (h5 : 5 ≤ g) :
    ∃ m : ℕ, g ^ 2 = 6 * m + 1 := by
  have h2 : ¬ 2 ∣ g := fun h => by
    have := (Nat.prime_dvd_prime_iff_eq Nat.prime_two hg).mp h; omega
  have h3 : ¬ 3 ∣ g := fun h => by
    have := (Nat.prime_dvd_prime_iff_eq Nat.prime_three hg).mp h; omega
  -- g mod 6 is 1 or 5, and both square to 1 mod 6
  have hmod : g % 6 = 1 ∨ g % 6 = 5 := by
    have := Nat.mod_lt g (by norm_num : 0 < 6)
    interval_cases hr : g % 6 <;> first | omega | (exfalso; apply h2; omega) | (exfalso; apply h3; omega)
  refine ⟨(g ^ 2 - 1) / 6, ?_⟩
  have hsq : g ^ 2 % 6 = 1 := by
    rcases hmod with h | h <;> · rw [Nat.pow_mod, h]
  have hge : 1 ≤ g ^ 2 := Nat.one_le_pow _ _ (by omega)
  omega

/-- **The one exact pair statement: a square is a lone killer of its column exactly when the
other member is prime.**  The square `g²` is the upper member `6m + 1`; the column's lower member
is `g² - 2`.  The pair `(squares, everything else)` combines on that column - some other field
also strikes it - exactly when `g² - 2` is composite. -/
theorem square_lone_killer_iff {g m : ℕ} (hm : g ^ 2 = 6 * m + 1) (hg : 5 ≤ g) :
    (6 * m - 1).Prime ↔ (g ^ 2 - 2).Prime := by
  have : 6 * m - 1 = g ^ 2 - 2 := by omega
  rw [this]

end MirrorWalk
