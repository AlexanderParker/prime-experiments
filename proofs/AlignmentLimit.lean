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

/-- **An aligned neighbour always acquires a factor above the alignment.**  If no gear up to `B`
divides `n` and `n > 1`, then `n` has a prime factor larger than `B`.  At the primorial this is
the exact reason the construction stops: the neighbours are open to everything aligned and are
struck by something above it.  The square-root rule is what converts "the factor is above `B`"
into "there is no factor" - and it applies only below `B²`. -/
theorem aligned_neighbour_factor {n B : ℕ} (hn : 1 < n)
    (hopen : ∀ g : ℕ, g.Prime → g ≤ B → ¬ (g ∣ n)) :
    ∃ p : ℕ, p.Prime ∧ p ∣ n ∧ B < p := by
  obtain ⟨p, hp, hpd⟩ := Nat.exists_prime_and_dvd (by omega : n ≠ 1)
  refine ⟨p, hp, hpd, ?_⟩
  by_contra hle
  push_neg at hle
  exact hopen p hp hle hpd

/-- **There is no top gear.**  Above every bound there is another gear.  This is Euclid, and it is
why "no new gears above the alignment" fails in any system that keeps enough arithmetic for
primality to mean anything: the same successor that builds the gears forbids a last one. -/
theorem gears_above_every_bound (B : ℕ) : ∃ p : ℕ, B < p ∧ p.Prime := by
  obtain ⟨p, hle, hp⟩ := Nat.exists_infinite_primes (B + 1)
  exact ⟨p, by omega, hp⟩

/-- **The allowance around zero is forced, not chosen.**  In any ring where 1 ≠ 0, zero has no
inverse: from `0 * x = 1` one gets `0 = 1`.  So "division by zero is undefined" was never a
convention that could have gone the other way, and defining `n / 0` to be a new element does not
remove the obstruction - it moves it, since the element so defined cannot satisfy the ring laws.
-/
theorem zero_not_invertible {R : Type*} [Ring R] [Nontrivial R] (x : R) : (0 : R) * x ≠ 1 := by
  intro h
  rw [zero_mul] at h
  exact zero_ne_one h

/-- **The window always holds a prime.**  The one-member version of the window statement is a
theorem, and a wasteful one: Bertrand puts a prime already inside `(q, 2q]`, far below `q²`.  The
contrast with the pair version is the whole difficulty - a gear strikes one class of a single
number and two classes of a pair, and everything downstream follows from that 1 against 2. -/
theorem window_has_prime {q : ℕ} (hq : 2 ≤ q) : ∃ p : ℕ, p.Prime ∧ q < p ∧ p ≤ q ^ 2 := by
  obtain ⟨p, hp, hlo, hhi⟩ := Nat.exists_prime_lt_and_le_two_mul (n := q) (by omega)
  refine ⟨p, hp, hlo, ?_⟩
  have h2q : 2 * q ≤ q ^ 2 := by nlinarith
  omega

end MirrorWalk
