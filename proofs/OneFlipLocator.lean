/-
OneFlipLocator (round 60, 2026-09-17): the construction in its final form.

Sixty-four rounds of search over walks ended where they began: the best construction is one
flip from home.  From the open pair `(-1, 1)`, flipping about the mirror `{2, 3, g}` with `k`
periods in either direction lands on the column `-1 + 12 g k d`; the machine's own gears decide
whether that column is open, and the square-root rule turns an open column below `q²` into a
twin prime pair.

  `OneFlipOpen G g K P`: some column `-1 + 12 g k d`, `k = 1 … K`, `d = ±1`, lies in the window
  `(P, P²]` and is struck by no gear of `G`.

  `oneflip_twin`: with `G` holding every prime from 5 below `P`, `OneFlipOpen` gives a twin
  prime pair in the window.

Measured (research/stack/r8/oneflip_margin.py): with `g` the first gear above `√q` and
`K = (ln q)³` the family holds thirty to fifty open columns at every machine to 20000, the
first at a period between 10 and 19; the margin follows the candidate count and not the shape
of the family (one stride with `(ln q)³` periods does as well as eight with `(ln q)²`).
Everything else the search produced - spirals, descents, settle walks with proved prefixes -
performs worse for the same room in the window (the trade lemma, `mirror_times_candidates`).
-/
import MirrorWalkConditional

namespace MirrorWalk

open SquareColumn

/-- The columns a single flip about `{2, 3, g}` can reach from home. -/
def oneFlip (g : ℤ) (k : ℕ) (d : ℤ) : ℤ := -1 + 12 * g * k * d

/-- **The one-flip hypothesis.**  Some column of the family is a column of the window that no
gear of `G` strikes. -/
def OneFlipOpen (G : Finset ℕ) (g : ℤ) (K P : ℕ) : Prop :=
  ∃ (k : ℕ) (d : ℤ) (m : ℕ), 1 ≤ k ∧ k ≤ K ∧ (d = 1 ∨ d = -1) ∧
    (m : ℤ) = oneFlip g k d ∧ 1 ≤ m ∧ 6 * m + 1 < P ^ 2 ∧
    ∀ h ∈ G, ¬ (h ∣ 6 * m - 1) ∧ ¬ (h ∣ 6 * m + 1)

/-- **The locator's theorem.**  With `G` holding every prime from 5 below `P`, the one-flip
hypothesis gives a twin prime pair inside the window `(P, P²]`. -/
theorem oneflip_twin {G : Finset ℕ} {g : ℤ} {K P : ℕ}
    (hfull : ∀ r, r.Prime → 5 ≤ r → r < P → r ∈ G)
    (hopen : OneFlipOpen G g K P) :
    ∃ m : ℕ, 1 ≤ m ∧ 6 * m + 1 < P ^ 2 ∧ (6 * m - 1).Prime ∧ (6 * m + 1).Prime := by
  obtain ⟨k, d, m, hk1, hkK, hd, hcast, hm1, hmlt, hstruck⟩ := hopen
  have hns : ¬ StruckBy G m := by
    rintro ⟨h, hh, hdvd | hdvd⟩
    · exact (hstruck h hh).1 hdvd
    · exact (hstruck h hh).2 hdvd
  obtain ⟨hp1, hp2⟩ := section_twin_of_unstruck hfull hm1 hmlt hns
  exact ⟨m, hm1, hmlt, hp1, hp2⟩

/-- **The window statement, from the locator.**  If every machine's one-flip family holds an
open column, every machine's window holds a twin prime pair. -/
theorem window_statement_of_oneflip
    (H : ∀ (P : ℕ), P.Prime → 5 ≤ P → ∃ (G : Finset ℕ) (g : ℤ) (K : ℕ),
        (∀ r, r.Prime → 5 ≤ r → r < P → r ∈ G) ∧ OneFlipOpen G g K P) :
    ∀ (P : ℕ), P.Prime → 5 ≤ P →
      ∃ m : ℕ, 1 ≤ m ∧ 6 * m + 1 < P ^ 2 ∧ (6 * m - 1).Prime ∧ (6 * m + 1).Prime := by
  intro P hP h5
  obtain ⟨G, g, K, hfull, hopen⟩ := H P hP h5
  exact oneflip_twin hfull hopen

end MirrorWalk
