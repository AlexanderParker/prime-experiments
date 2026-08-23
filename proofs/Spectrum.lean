/-
The bridge identity: a merged window is a window of consecutive gaps.

Constructor's decomposition of part (D) rests on one arithmetic fact. A
qualifying word `w` occupies `l` consecutive gaps of `M`; its two FLANKS are
the gaps immediately outside it; and the merge it produces has length

    merged  =  span(w) + FS(w)

where `span` is the sum of the `l` gaps the word covers and `FS` the sum of
the two flanking gaps. Those `l + 2` gaps are CONSECUTIVE, so the merged
length is a window sum - and is therefore bounded by the spectrum value
`F_{l+2}(M)`. Writing `k = l + 1` for the number of killed openings (a
`k`-chain merges `k + 1` gaps) this is `merged <= F_{k+1}(M)`.

That is the load-bearing formal step: it is what lets the two-part
decomposition

    [ k_win <= 3 ]   and   [ F_4(M) - F(M) <= q' ]

imply (D) at `alpha = 3`, with no reference to fuel, words, residues or
padding - see `merged_le_of_shallow` at the end.

Nothing here is empirical: both halves above are being tested elsewhere.
This file proves only the bridge, which holds for any gap sequence
whatsoever.
-/

import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset

namespace Spectrum

/-- The sum of `j` consecutive gaps of `g` starting at index `a`. -/
def windowSum (g : ℕ → ℕ) (a j : ℕ) : ℕ := ∑ i ∈ Finset.range j, g (a + i)

/-- `Fj` bounds every window of `j` consecutive gaps: the spectrum value. -/
def SpectrumBound (g : ℕ → ℕ) (j Fj : ℕ) : Prop := ∀ a, windowSum g a j ≤ Fj

@[simp] theorem windowSum_zero (g : ℕ → ℕ) (a : ℕ) : windowSum g a 0 = 0 := by
  simp [windowSum]

@[simp] theorem windowSum_one (g : ℕ → ℕ) (a : ℕ) : windowSum g a 1 = g a := by
  simp [windowSum]

/-- Window sums grow with the window: gaps are non-negative. -/
theorem windowSum_mono (g : ℕ → ℕ) (a : ℕ) {j j' : ℕ} (h : j ≤ j') :
    windowSum g a j ≤ windowSum g a j' :=
  Finset.sum_le_sum_of_subset (fun _x hx => Finset.mem_range.mpr
    (lt_of_lt_of_le (Finset.mem_range.mp hx) h))

/-- **The bridge identity.** A word occupying the `l` gaps
`g (a+1), ..., g (a+l)`, together with its two flanks `g a` and
`g (a+l+1)`, spans exactly the `l + 2` consecutive gaps starting at `a`. -/
theorem merged_eq (g : ℕ → ℕ) (a l : ℕ) :
    g a + windowSum g (a + 1) l + g (a + l + 1) = windowSum g a (l + 2) := by
  have h : windowSum g a (l + 2)
      = (∑ i ∈ Finset.range (l + 1), g (a + i)) + g (a + (l + 1)) :=
    Finset.sum_range_succ _ _
  have h2 : (∑ i ∈ Finset.range (l + 1), g (a + i))
      = (∑ i ∈ Finset.range l, g (a + (i + 1))) + g (a + 0) :=
    Finset.sum_range_succ' _ _
  have h3 : (∑ i ∈ Finset.range l, g (a + (i + 1))) = windowSum g (a + 1) l :=
    Finset.sum_congr rfl (fun i _ => by
      have hi : a + (i + 1) = a + 1 + i := by omega
      rw [hi])
  rw [h, h2, h3]
  have e1 : g (a + 0) = g a := rfl
  have e2 : g (a + (l + 1)) = g (a + l + 1) := by
    have hl : a + (l + 1) = a + l + 1 := by omega
    rw [hl]
  rw [e1, e2]
  omega

/-- **Merged windows are bounded by the spectrum.** Immediately from the
identity: the merge is a window of `l + 2 = k + 1` consecutive gaps. -/
theorem merged_le_spectrum {g : ℕ → ℕ} {a l Fj : ℕ}
    (h : SpectrumBound g (l + 2) Fj) :
    g a + windowSum g (a + 1) l + g (a + l + 1) ≤ Fj := by
  rw [merged_eq]
  exact h a

/-- The `k`-indexed form: a `k`-chain merges `k + 1` gaps, so its merged
length is at most `F_{k+1}(M)`. -/
theorem merged_le_spectrum_succ {g : ℕ → ℕ} {a l Fk : ℕ}
    (h : SpectrumBound g ((l + 1) + 1) Fk) :
    g a + windowSum g (a + 1) l + g (a + l + 1) ≤ Fk :=
  merged_le_spectrum (by simpa [Nat.add_assoc] using h)

/-- **(D) at `alpha = 3` from the two halves.** If the winning word is
shallow - `k_win <= 3`, so it occupies at most `l <= 2` gaps and its merged
window is at most 4 consecutive gaps - and shallow flatness
`F_4 <= F + q'` holds, then every such merged window is at most `F + q'`.

No fuel, no `k_max`, no word list, no residues, no padding enter the
statement: only the gap sequence and the two hypotheses. -/
theorem merged_le_of_shallow {g : ℕ → ℕ} {a l F4 F q : ℕ}
    (hl : l + 2 ≤ 4) (h4 : SpectrumBound g 4 F4) (hflat : F4 ≤ F + q) :
    g a + windowSum g (a + 1) l + g (a + l + 1) ≤ F + q := by
  rw [merged_eq]
  exact le_trans (le_trans (windowSum_mono g a hl) (h4 a)) hflat

/-! ## Suppression-corrected flatness

Round 17 refuted RAW flatness (`F_j - F <= q'` fails at 5 of 15
machine-depth pairs). Constructor's repaired requirement adds the measured
suppression term:

    F_j - F  <=  q' + lambda * (j-2) * L        (corrected flatness)

and pairs it with the suppression law itself,

    qualmax_j + lambda * (j-2) * L  <=  F_j     (suppression)

whose composition is exactly (D). Both are census-checkable and both are
HYPOTHESES here, so nothing in this file is at risk from a census revision -
only the composition is proved. Writing `d` for the depth excess `j - 2`
keeps the statement free of truncated subtraction.
-/

/-- **The composition.** Corrected flatness plus suppression give (D):
the qualifying maximum at depth `j = d + 2` is at most `F + q'`. -/
theorem qual_le_of_suppressed {Fj Qj F q lam d L : ℕ}
    (hflat : Fj ≤ F + q + lam * d * L)
    (hsupp : Qj + lam * d * L ≤ Fj) : Qj ≤ F + q := by
  omega

/-- The same in window form: a merged window of a word occupying `l` gaps
sits at depth `j = l + 2`, so `d = l`. -/
theorem merged_le_of_suppressed {g : ℕ → ℕ} {a l Fj Qj F q lam L : ℕ}
    (hQ : g a + windowSum g (a + 1) l + g (a + l + 1) ≤ Qj)
    (hflat : Fj ≤ F + q + lam * l * L)
    (hsupp : Qj + lam * l * L ≤ Fj) :
    g a + windowSum g (a + 1) l + g (a + l + 1) ≤ F + q :=
  le_trans hQ (qual_le_of_suppressed hflat hsupp)

/-! ## The qualifying spectrum

Mechanic's word-free criterion is built on `Q_j(M; a)`: the largest sum of
`j` consecutive gaps whose `j - 2` MIDDLE gaps are all at least the
qualifying floor `a = 2u'` (research/qspec_table.py). The definitions below
fix that object over an abstract gap sequence, and the theorems compose it:
a word all of whose letters meet the floor makes its merged window a
QUALIFYING window, so (D) follows from `Q_j <= F + q'` alone - at every
depth at once, with no `k_win`, no fuel and no word list in the statement.
This is round 19's suppression-corrected flatness in the form the censuses
can discharge machine by machine.
-/

/-- The `j - 2` interior gaps of the `j`-window at `a` all meet the
qualifying floor `2u`. (Windows of depth `j <= 2` have no interior and
qualify vacuously.) -/
def Qualifying (g : ℕ → ℕ) (u a j : ℕ) : Prop :=
  ∀ i, 1 ≤ i → i + 1 < j → 2 * u ≤ g (a + i)

/-- `Qj` bounds every QUALIFYING window of depth `j`: the qualifying
spectrum value `Q_j` of research/qspec_table.py, over an abstract gap
sequence. -/
def QualBound (g : ℕ → ℕ) (u j Qj : ℕ) : Prop :=
  ∀ a, Qualifying g u a j → windowSum g a j ≤ Qj

/-- A word all of whose `l` letters are at least `2u` makes its merged
window qualifying: the interiors of the `(l+2)`-window at `a` are exactly
the word's letters. -/
theorem qualifying_of_word {g : ℕ → ℕ} {u a l : ℕ}
    (hw : ∀ i < l, 2 * u ≤ g (a + 1 + i)) :
    Qualifying g u a (l + 2) := by
  intro i h1 hi
  have he : a + i = a + 1 + (i - 1) := by omega
  rw [he]
  exact hw (i - 1) (by omega)

/-- The merged window of a qualifying word is bounded by `Q_{l+2}`. -/
theorem merged_le_qual {g : ℕ → ℕ} {u a l Qj : ℕ}
    (hQ : QualBound g u (l + 2) Qj)
    (hw : ∀ i < l, 2 * u ≤ g (a + 1 + i)) :
    g a + windowSum g (a + 1) l + g (a + l + 1) ≤ Qj := by
  rw [merged_eq]
  exact hQ a (qualifying_of_word hw)

/-- **Suppression-corrected flatness implies (D), depth by depth.** If the
qualifying spectrum at depth `l + 2` is `(F + q)`-flat, every merged window
of a floor-respecting word of `l` letters is within tolerance. -/
theorem merged_le_of_qual_flat {g : ℕ → ℕ} {u a l Qj F q : ℕ}
    (hQ : QualBound g u (l + 2) Qj) (hflat : Qj ≤ F + q)
    (hw : ∀ i < l, 2 * u ≤ g (a + 1 + i)) :
    g a + windowSum g (a + 1) l + g (a + l + 1) ≤ F + q :=
  le_trans (merged_le_qual hQ hw) hflat

/-- **The word-free criterion, all depths at once.** If `Q_j <= F + q` for
EVERY depth `j`, then (D) holds for every floor-respecting word of every
length - no `k_win` bound, no fuel cap, no word list. (`Q_j = 0` for depths
carrying no qualifying window discharges deep `j` for free.) -/
theorem merged_le_of_qual_flat_all {g : ℕ → ℕ} {u F q : ℕ} (Q : ℕ → ℕ)
    (hQ : ∀ j, QualBound g u j (Q j)) (hflat : ∀ j, Q j ≤ F + q) :
    ∀ a l, (∀ i < l, 2 * u ≤ g (a + 1 + i)) →
      g a + windowSum g (a + 1) l + g (a + l + 1) ≤ F + q :=
  fun _a l hw => merged_le_of_qual_flat (hQ (l + 2)) (hflat (l + 2)) hw

/-- **R31's two-part form, hypothesis-explicit.** Raw flatness corrected by
`lam * l * L` (corrected flatness) plus the suppression law at depth
`l + 2` (every qualifying window sits `lam * l * L` below `F_j`) give (D)
for every floor-respecting word - the composition of round 19's two
census-checked statements, with the qualifying hypothesis now explicit. -/
theorem merged_le_of_corrected {g : ℕ → ℕ} {u a l Fj F q lam L : ℕ}
    (hflat : Fj ≤ F + q + lam * l * L)
    (hsupp : ∀ b, Qualifying g u b (l + 2) → windowSum g b (l + 2) + lam * l * L ≤ Fj)
    (hw : ∀ i < l, 2 * u ≤ g (a + 1 + i)) :
    g a + windowSum g (a + 1) l + g (a + l + 1) ≤ F + q := by
  rw [merged_eq]
  have h := hsupp a (qualifying_of_word hw)
  omega

/-- Both literal letters meet the qualifying floor: `2u' <= q' - 2u'`
whenever `6u' = q' -+ 1` and `u' >= 1`, so literal words are
floor-respecting and `merged_le_qual` applies to them. -/
theorem alphabet_ge_floor {q u : ℕ} (hu : 6 * u + 1 = q ∨ 6 * u = q + 1) :
    2 * u ≤ q - 2 * u := by omega

/-- Padded letters meet the floor too: `2u' <= q'`. -/
theorem padded_ge_floor {q u : ℕ} (hu : 6 * u + 1 = q ∨ 6 * u = q + 1) :
    2 * u ≤ q := by omega

/-! ## The correlation object

The search's open requirement is now a single object: the joint rate of
QUALIFYING gaps at consecutive separations, and its deficit against
independence. Constructor, mechanic and lateral are each computing it, so
the definitions below fix what is being computed - division-free, so the
deficit factor `D` is an integer comparison rather than a ratio.

`qualCount` counts qualifying gaps among the first `N`; `jointCount ... j`
counts positions where `j` CONSECUTIVE gaps all qualify. Independence would
predict `jointCount * N^(j-1) = qualCount^j`; `NegCorrelated ... D` asserts
the measured joint count falls short by a factor `D` - the measured
deficits being x26, x6.7 and x1400.
-/

/-- Qualifying gaps among the first `N`. -/
def qualCount (Q : ℕ → Bool) (g : ℕ → ℕ) (N : ℕ) : ℕ :=
  ((Finset.range N).filter fun i => Q (g i) = true).card

/-- Positions where `j` consecutive gaps all qualify. -/
def jointCount (Q : ℕ → Bool) (g : ℕ → ℕ) (N j : ℕ) : ℕ :=
  ((Finset.range N).filter fun i => ((List.range j).all fun s => Q (g (i + s))) = true).card

/-- **Negative correlation at depth `j`, by a factor `D`.** Independence
would give `jointCount * N^(j-1) = qualCount^j`; this says the joint count
is short by at least `D`. -/
def NegCorrelated (Q : ℕ → Bool) (g : ℕ → ℕ) (N j D : ℕ) : Prop :=
  D * jointCount Q g N j * N ^ (j - 1) ≤ (qualCount Q g N) ^ j

@[simp] theorem jointCount_one (Q : ℕ → Bool) (g : ℕ → ℕ) (N : ℕ) :
    jointCount Q g N 1 = qualCount Q g N := by
  simp [jointCount, qualCount]

/-- Deeper joint events are rarer: `j+1` consecutive qualifying gaps in
particular give `j` consecutive ones. This is what lets deficits compound. -/
theorem jointCount_antitone (Q : ℕ → Bool) (g : ℕ → ℕ) (N j : ℕ) :
    jointCount Q g N (j + 1) ≤ jointCount Q g N j := by
  apply Finset.card_le_card
  intro i hi
  rw [Finset.mem_filter] at hi ⊢
  refine ⟨hi.1, ?_⟩
  have h := hi.2
  rw [List.all_eq_true] at h ⊢
  intro s hs
  exact h s (List.mem_range.mpr (by have := List.mem_range.mp hs; omega))

end Spectrum
