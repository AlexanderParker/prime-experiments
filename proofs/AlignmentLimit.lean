/-
AlignmentLimit (round 77, 2026-09-18): the alignment at infinity, and what it does and does not
give.

The observation: home is open because at 0 every gear sits at residue 0, and the two neighbours
are the one place no gear can reach - a gear striking `±1` would divide 1.  Carry that to the
limit: the primorial of ALL gears would put every gear at 0 at once, so its neighbours would be
open to everything, hence a twin.

Two of the three steps are true, and this file proves both, together with the step that fails.

  `open_columns_for_any_gears`: for EVERY finite set of gears and every bound, there are columns
  beyond that bound open to all of them.  This is the alignment, at every finite level, and it is
  exactly the primorial construction - take the column to be a multiple of the product.

  `no_column_open_to_all_gears`: no column is open to ALL gears.  Every member above 1 has a prime
  factor, and that factor is a gear striking it.

So the alignment exists at every finite level and nowhere in the limit: the quantifiers do not
commute.  "For every gear set there is an open column" is true; "there is a column open to every
gear" is false; and the window statement lives between them, asking for a column whose gear set is
determined by the column's own size:

    for every machine q, some column in (q, q²] is open to the gears up to q.

The gear set grows with the column, and that is precisely what no single alignment can answer.
The measured form of the same thing: the pair either side of the primorial is open to every gear
up to P, but its members are of size e^P, whose primality is decided by the gears up to e^(P/2) -
a set that is not aligned.  It is a twin only at P = 3, 5 and 11 up to P = 53; at P = 7 the lower
member is 209 = 11 x 19.
-/
import Mathlib

namespace MirrorWalk

/-- **The alignment exists at every finite level.**  For any finite set of gears and any bound,
some column beyond the bound is open to all of them: take the column to be a multiple of their
product, so every gear divides `6m` and therefore misses `6m ± 1`. -/
theorem open_columns_for_any_gears (G : Finset ℕ) (hG : ∀ g ∈ G, 1 < g) (N : ℕ) :
    ∃ m : ℕ, N ≤ m ∧ ∀ g ∈ G, ¬ (g ∣ (6 * m - 1)) ∧ ¬ (g ∣ (6 * m + 1)) := by
  classical
  set P : ℕ := ∏ g ∈ G, g with hP
  have hPpos : 0 < P := by
    rw [hP]
    exact Finset.prod_pos (fun g hg => by have := hG g hg; omega)
  refine ⟨P * (N + 1), ?_, ?_⟩
  · calc N ≤ N + 1 := by omega
      _ ≤ P * (N + 1) := Nat.le_mul_of_pos_left _ hPpos
  · intro g hg
    have hgP : g ∣ P := hP ▸ Finset.dvd_prod_of_mem _ hg
    have hg6m : g ∣ 6 * (P * (N + 1)) := Dvd.dvd.mul_left (hgP.mul_right (N + 1)) 6
    have hg1 : 1 < g := hG g hg
    have hpos : 1 ≤ 6 * (P * (N + 1)) := by
      have : 1 ≤ P * (N + 1) := Nat.one_le_iff_ne_zero.mpr (by positivity)
      omega
    constructor
    · intro hd
      have : g ∣ 6 * (P * (N + 1)) - (6 * (P * (N + 1)) - 1) := Nat.dvd_sub hg6m hd
      rw [Nat.sub_sub_self hpos] at this
      have := Nat.le_of_dvd one_pos this
      omega
    · intro hd
      have h1 : g ∣ 6 * (P * (N + 1)) + 1 - 6 * (P * (N + 1)) := Nat.dvd_sub hd hg6m
      have e : 6 * (P * (N + 1)) + 1 - 6 * (P * (N + 1)) = 1 := by omega
      rw [e] at h1
      have := Nat.le_of_dvd one_pos h1
      omega

/-- **The alignment does not exist in the limit.**  No column is open to every gear: the lower
member is above 1, so it has a prime factor, and that factor strikes it. -/
theorem no_column_open_to_all_gears {m : ℕ} (hm : 1 ≤ m) :
    ∃ g : ℕ, g.Prime ∧ (g ∣ (6 * m - 1) ∨ g ∣ (6 * m + 1)) := by
  have h1 : 2 ≤ 6 * m - 1 := by omega
  obtain ⟨g, hg, hgd⟩ := Nat.exists_prime_and_dvd (by omega : 6 * m - 1 ≠ 1)
  exact ⟨g, hg, Or.inl hgd⟩

/-- **The window statement sits between them.**  It asks for a column whose gear set is fixed by
the column's own size - the quantifier pattern neither of the two above supplies. -/
def WindowStatement : Prop :=
  ∀ q : ℕ, q.Prime → 5 ≤ q →
    ∃ m : ℕ, q < 6 * m - 1 ∧ 6 * m + 1 ≤ q ^ 2 ∧
      ∀ g : ℕ, g.Prime → g ≤ q → ¬ (g ∣ (6 * m - 1)) ∧ ¬ (g ∣ (6 * m + 1))

end MirrorWalk
