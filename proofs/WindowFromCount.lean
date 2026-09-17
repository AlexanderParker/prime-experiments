/-
WindowFromCount (round 72, 2026-09-18): the bridge from a counting bound to a window statement.

The construction's own route is closed at the step (round 71).  The analytic routes - Chen's
theorem for almost-prime twins, bounded gaps - do not prove the twin window statement either,
since the parity barrier blocks 2 even under the strongest distribution hypotheses.  What they
can give is a weaker statement of the same SHAPE, and this file is the piece that turns any such
counting bound into a statement about the window, with no analysis in it at all:

    more good numbers below `q²` than below `q`  ⟹  a good number inside `(q, q²]`.

That is `exists_in_window_of_count`.  It is the general form of what the chain does by hand: the
analytic input is a lower bound at `q²` against an upper bound at `q`, and the conclusion is the
window statement for whatever property `P` the bound counts.

`chen_window_of_count` is the instance for Chen pairs - a prime `p` whose partner `p + 2` has at
most two prime factors - the statement the analytic literature can actually support.  The
literature position, recorded in research/proof/loop_algorithms.md entry 77: the best explicit
constant for the twin half of Chen's theorem is `1.205 C₂ x / (log x)²` (Bordignon-Starichkova
2024) but it is stated for sufficiently large `x` with no computable threshold, and the fully
explicit versions on the Goldbach side start at `exp(exp(32.7))`.  So the hypothesis of
`chen_window_of_count` is not yet available for every machine, and this file says exactly what
would have to be supplied.
-/
import Mathlib

namespace MirrorWalk

open Finset

/-- **The bridge.**  If the machine's window edge `q` has fewer good numbers below it than `q²`
does, then a good number lies inside the window `(q, q²]`. -/
theorem exists_in_window_of_count {P : ℕ → Prop} [DecidablePred P] {q : ℕ}
    (h : ((range (q + 1)).filter P).card < ((range (q ^ 2 + 1)).filter P).card) :
    ∃ n, q < n ∧ n ≤ q ^ 2 ∧ P n := by
  by_contra hcon
  push_neg at hcon
  have hsub : (range (q ^ 2 + 1)).filter P ⊆ (range (q + 1)).filter P := by
    intro n hn
    rw [mem_filter, mem_range] at hn
    obtain ⟨hlt, hP⟩ := hn
    have hle : n ≤ q := by
      by_contra hgt
      push_neg at hgt
      exact absurd hP (hcon n hgt (by omega))
    exact mem_filter.mpr ⟨mem_range.mpr (by omega), hP⟩
  have := card_le_card hsub
  omega

/-- A Chen pair: `p` is prime and `p + 2` has at most two prime factors with multiplicity. -/
def ChenPair (p : ℕ) : Prop := p.Prime ∧ (p + 2).primeFactorsList.length ≤ 2

instance : DecidablePred ChenPair := fun _ => inferInstanceAs (Decidable (_ ∧ _))

/-- **Chen pairs in the window, from a counting bound.**  Whatever lower bound the sieve gives at
`q²`, once it exceeds the count at `q` the window holds a prime whose partner two above has at
most two prime factors. -/
theorem chen_window_of_count {q : ℕ}
    (h : ((range (q + 1)).filter ChenPair).card < ((range (q ^ 2 + 1)).filter ChenPair).card) :
    ∃ p, q < p ∧ p ≤ q ^ 2 ∧ p.Prime ∧ (p + 2).primeFactorsList.length ≤ 2 := by
  obtain ⟨p, hlo, hhi, hp⟩ := exists_in_window_of_count (P := ChenPair) h
  exact ⟨p, hlo, hhi, hp.1, hp.2⟩

/-- A twin pair is a Chen pair, so the window statement implies its Chen form - the analytic
target is genuinely weaker. -/
theorem chenPair_of_twin {p : ℕ} (hp : p.Prime) (hq : (p + 2).Prime) : ChenPair p := by
  refine ⟨hp, ?_⟩
  have : (p + 2).primeFactorsList = [p + 2] := Nat.primeFactorsList_prime hq
  rw [this]
  simp

end MirrorWalk
