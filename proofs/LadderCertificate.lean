/-
LadderCertificate (round 117, 2026-09-19): the canonical twin ladder, certified for six rungs.

The canonical ladder takes, from each twin centre `s`, the twin centre of its stretch nearest the
centre (negative offset first on ties): s₀ = 6 (the twin (5, 7)), then
  30 (29, 31), 882 (881, 883), 777978 (777977, 777979), 605249768610 (12 digits),
  366327282402458541331692 (24 digits), 134195677832370611289419659133121237872763586710 (48).
Each rung is two primality certificates (norm_num below 10⁸, Lucas/Pratt certificates from
LadderPratt.lean above) and two containment inequalities checked by norm_num.  With
`twins_unbounded_of_ladder` (TwinLadderTheorem.lean), the certified prefix composes with the
ladder hypothesis beyond it: every twin centre above 10⁴⁷ having a rung gives twins unbounded.
(Tree nodes R5.f.v, R5.f.vii.)
-/
import TwinLadderTheorem
import LadderPratt

namespace TwinLadder

open Pratt

theorem tc_6 : TwinCentre 6 := ⟨by norm_num, by norm_num, by norm_num⟩
theorem tc_30 : TwinCentre 30 := ⟨by norm_num, by norm_num, by norm_num⟩
theorem tc_882 : TwinCentre 882 := ⟨by norm_num, by norm_num, by norm_num⟩
theorem tc_777978 : TwinCentre 777978 := ⟨by norm_num, by norm_num, by norm_num⟩
theorem tc_605249768610 : TwinCentre 605249768610 :=
  ⟨by norm_num, prime_605249768609, prime_605249768611⟩
theorem tc_366327282402458541331692 : TwinCentre 366327282402458541331692 :=
  ⟨by norm_num, prime_366327282402458541331691, prime_366327282402458541331693⟩
theorem tc_134195677832370611289419659133121237872763586710 :
    TwinCentre 134195677832370611289419659133121237872763586710 :=
  ⟨by norm_num, prime_134195677832370611289419659133121237872763586709,
    prime_134195677832370611289419659133121237872763586711⟩

theorem rung_0 : Rung 6 30 := ⟨tc_30, by norm_num, by norm_num⟩
theorem rung_1 : Rung 30 882 := ⟨tc_882, by norm_num, by norm_num⟩
theorem rung_2 : Rung 882 777978 := ⟨tc_777978, by norm_num, by norm_num⟩
theorem rung_3 : Rung 777978 605249768610 := ⟨tc_605249768610, by norm_num, by norm_num⟩
theorem rung_4 : Rung 605249768610 366327282402458541331692 :=
  ⟨tc_366327282402458541331692, by norm_num, by norm_num⟩
theorem rung_5 : Rung 366327282402458541331692 134195677832370611289419659133121237872763586710 :=
  ⟨tc_134195677832370611289419659133121237872763586710, by norm_num, by norm_num⟩

/-- **The certified ladder**: six rungs from the twin (5, 7) to a 48-digit twin centre. -/
theorem canonical_ladder_six_rungs :
    Rung 6 30 ∧ Rung 30 882 ∧ Rung 882 777978 ∧ Rung 777978 605249768610 ∧
    Rung 605249768610 366327282402458541331692 ∧
    Rung 366327282402458541331692 134195677832370611289419659133121237872763586710 :=
  ⟨rung_0, rung_1, rung_2, rung_3, rung_4, rung_5⟩

/-- **Composition with the ladder hypothesis above the certified prefix**: if every twin centre
at least as large as the sixth rung has a rung, twin primes are unbounded. -/
theorem twins_unbounded_of_ladder_above
    (hL : ∀ s : ℕ, TwinCentre s → 134195677832370611289419659133121237872763586710 ≤ s →
      ∃ s' : ℕ, Rung s s') :
    ∀ N : ℕ, ∃ m : ℕ, N < 6 * m - 1 ∧ (6 * m - 1).Prime ∧ (6 * m + 1).Prime := by
  intro N
  -- climb from the certified top by the hypothesis
  have climb : ∀ n : ℕ, ∃ s : ℕ, TwinCentre s ∧
      134195677832370611289419659133121237872763586710 ≤ s ∧ n ≤ s := by
    intro n
    induction n with
    | zero => exact ⟨_, tc_134195677832370611289419659133121237872763586710, le_refl _, Nat.zero_le _⟩
    | succ k ih =>
      obtain ⟨s, hs, hbig, hk⟩ := ih
      obtain ⟨s', hr⟩ := hL s hs hbig
      have := rung_gt hs hr
      exact ⟨s', hr.1, by omega, by omega⟩
  obtain ⟨s, ⟨⟨m, rfl⟩, hp1, hp2⟩, _, hN⟩ := climb (N + 2)
  exact ⟨m, by omega, hp1, hp2⟩

end TwinLadder
