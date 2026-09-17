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

The mirror that works is the smallest one that fits the window: `g = 5`, stride `360`
(research/stack/r8/oneflip_classes.py, oneflip_small_mirror.py, round 61).  Over every
admissible mirror at `q = 5000` the open columns run from 55 (at `g = 5`) down to 0 (at
`g = 2843`), mean 8.3; the mirror at the first gear above `√q` leaves none at `q = 101` and
half as many as `g = 5` at every larger machine tested.  With `g = 5` and `K = (ln q)³` the
family holds 25 to 58 open columns at every machine from 19 to 20011, and the only machines
with no candidate at all are `q = 11, 13, 17`, where the stride 360 does not fit below `q²`
(the mirror `{2, 3}`, stride 72, covers those down to `q = 13`).

The teeth law below says why the family is rigid: on it every gear's two striking classes are
the fixed pair `(7, 5)` scaled by that gear's own inverse of the stride.  Everything else the
search produced - spirals, descents, settle walks with proved prefixes - performs worse for the
same room in the window (the trade lemma, `mirror_times_candidates`).
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

/-- **The teeth of a gear on the one-flip family.**  With `u` an inverse of the stride `s`
modulo `h`, the gear `h` divides `s * k - c` exactly when `k ≡ c * u`. -/
theorem strike_iff_scaled {h : ℕ} {s u c k : ℤ} (hu : s * u ≡ 1 [ZMOD h]) :
    (h : ℤ) ∣ s * k - c ↔ k ≡ c * u [ZMOD h] := by
  constructor
  · intro hd
    have h0 : s * k ≡ c [ZMOD h] := Int.ModEq.symm (Int.modEq_iff_dvd.mpr (by simpa using hd))
    have h1 : k * (s * u) ≡ k * 1 [ZMOD h] := hu.mul_left k
    have h2 : k * (s * u) = (s * k) * u := by ring
    have h3 : (s * k) * u ≡ c * u [ZMOD h] := h0.mul_right u
    have h4 : k * 1 ≡ c * u [ZMOD h] := by
      have h5 : k * 1 ≡ (s * k) * u [ZMOD h] := by rw [← h2]; exact h1.symm
      exact h5.trans h3
    rwa [mul_one] at h4
  · intro hk
    have h1 : s * k ≡ s * (c * u) [ZMOD h] := hk.mul_left s
    have h3 : s * (c * u) ≡ c [ZMOD h] := by
      have h3' : c * (s * u) ≡ c * 1 [ZMOD h] := hu.mul_left c
      calc s * (c * u) = c * (s * u) := by ring
        _ ≡ c * 1 [ZMOD h] := h3'
        _ = c := by ring
    exact Int.ModEq.dvd (Int.ModEq.symm (h1.trans h3))

/-- **Both teeth at once.**  On the family `-1 + 12 g k` the two members are `72 g k - 7` and
`72 g k - 5`, so a gear `h` inverting the stride at `u` strikes exactly at `k ≡ 7 u` and
`k ≡ 5 u`: every gear's teeth are the fixed pair `(7, 5)` scaled by its own unit. -/
theorem oneflip_teeth {h : ℕ} {g u k : ℤ} (hu : (72 * g) * u ≡ 1 [ZMOD h]) :
    ((h : ℤ) ∣ 72 * g * k - 7 ↔ k ≡ 7 * u [ZMOD h]) ∧
    ((h : ℤ) ∣ 72 * g * k - 5 ↔ k ≡ 5 * u [ZMOD h]) :=
  ⟨strike_iff_scaled hu, strike_iff_scaled hu⟩

/-- The members of the family's column, in the form the teeth law uses. -/
theorem oneflip_members (g : ℤ) (k : ℕ) :
    6 * oneFlip g k 1 - 1 = 72 * g * k - 7 ∧ 6 * oneFlip g k 1 + 1 = 72 * g * k - 5 := by
  unfold oneFlip; constructor <;> ring

end MirrorWalk
