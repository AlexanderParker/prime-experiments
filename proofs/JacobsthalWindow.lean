/-
JacobsthalWindow (round 75, 2026-09-18): the window statement from a gap bound, and the exponent
that is missing.

Requirement 3 of the anatomy asks for a mechanism that names a candidate open against the gears
between the primorial's reach and `√N`.  There is one classical object that does exactly that, and
this project has already worked on it: the paired Jacobsthal function `j₂`, the longest run of
columns in which no column escapes every gear of a primorial.

A bound on `j₂` names an open column in EVERY run of that length - no counting, no sieve
cancellation, no choice of mirror.  So it converts directly into the window statement:

  `window_of_column_gap`: if every run of `J` consecutive columns contains one that no gear of the
  machine strikes, and a run of `J` columns starting just above the machine fits under `q²`, then
  the window holds a twin prime pair.

That is the whole implication, proved here.  What it needs is `J` of the order of the window's
length, which in the machine's own units means a bound

    j₂(q#)  <  q² - q.

The project's ladder (docs/novel, j2-upper-bound) proves `j₂(p_n#) ≪ p_n^{4.266+ε}` by the
fundamental lemma, after an elementary `3^{n+1} log² p_n` and a quasi-polynomial rung.  The window
needs exponent 2.  The ladder also carries the ceiling: exponent 2 sits below Selberg's
conjectural floor `2κ = 4` for dimension-2 sifting, so the missing factor is parity rather than
technique - the same wall that stops the analytic route (loop entry 77).

So this file records the route in its exact form: the implication is cheap and proved; the input
is an exponent-2 bound on `j₂`, and no sieve can give it.
-/
import MirrorWalkConditional

namespace MirrorWalk

open SquareColumn

/-- **The window statement from a run bound.**  `J` bounds the length of a run of columns with no
open column in it; if such a run starting at the machine's own top gear still ends below `q²`,
the window holds a twin prime pair. -/
theorem window_of_column_gap {G : Finset ℕ} {P J : ℕ} (hP : 5 ≤ P)
    (hfull : ∀ r, r.Prime → 5 ≤ r → r < P → r ∈ G)
    (hgap : ∀ x : ℕ, ∃ m, x ≤ m ∧ m < x + J ∧ ∀ g ∈ G, ¬ (g ∣ 6 * m - 1) ∧ ¬ (g ∣ 6 * m + 1))
    (hfit : 6 * (P + J) + 1 < P ^ 2) :
    ∃ m : ℕ, 1 ≤ m ∧ 6 * m + 1 < P ^ 2 ∧ (6 * m - 1).Prime ∧ (6 * m + 1).Prime := by
  obtain ⟨m, hlo, hhi, hopen⟩ := hgap P
  have hm1 : 1 ≤ m := by omega
  have hmlt : 6 * m + 1 < P ^ 2 := by omega
  have hns : ¬ StruckBy G m := by
    rintro ⟨g, hg, hd | hd⟩
    · exact (hopen g hg).1 hd
    · exact (hopen g hg).2 hd
  obtain ⟨hp1, hp2⟩ := section_twin_of_unstruck hfull hm1 hmlt hns
  exact ⟨m, hm1, hmlt, hp1, hp2⟩

/-- **The window statement for every machine, from a uniform run bound.**  If the run length grows
slower than the window - `6 (P + J P) + 1 < P²` at every machine - every window holds a twin. -/
theorem window_statement_of_gap_law {J : ℕ → ℕ}
    (hgap : ∀ (P : ℕ), P.Prime → 5 ≤ P → ∃ G : Finset ℕ,
      (∀ r, r.Prime → 5 ≤ r → r < P → r ∈ G) ∧
      ∀ x : ℕ, ∃ m, x ≤ m ∧ m < x + J P ∧ ∀ g ∈ G, ¬ (g ∣ 6 * m - 1) ∧ ¬ (g ∣ 6 * m + 1))
    (hfit : ∀ (P : ℕ), P.Prime → 5 ≤ P → 6 * (P + J P) + 1 < P ^ 2) :
    ∀ (P : ℕ), P.Prime → 5 ≤ P →
      ∃ m : ℕ, 1 ≤ m ∧ 6 * m + 1 < P ^ 2 ∧ (6 * m - 1).Prime ∧ (6 * m + 1).Prime := by
  intro P hP h5
  obtain ⟨G, hfull, hg⟩ := hgap P hP h5
  exact window_of_column_gap h5 hfull hg (hfit P hP h5)

end MirrorWalk
