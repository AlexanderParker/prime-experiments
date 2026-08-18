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

end Spectrum
