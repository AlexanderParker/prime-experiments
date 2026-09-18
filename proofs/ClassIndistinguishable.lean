/-
ClassIndistinguishable (round 76, 2026-09-18): what residues can never decide.

Every mechanism this machine has names a column by its residue class.  Carrying names the class of
the mirror's multiples; the period rule names a class modulo the product of the gears it dodges
(`openness_periodic`, round 74); a walk of flips names the class of the gcd's multiples
(`gcdL_dvd_combo`, round 55).  That is the whole toolkit.

This file proves the toolkit cannot finish the job, and the proof needs no measurement:

  `class_has_closed_column`: every residue class contains columns that are not twin columns -
  given any modulus `M` and any gear `p` outside it, some column of the class, as far out as you
  like, has its lower member divisible by `p` and larger than `p`.

So no statement of the form "the columns of this class are twin columns" is ever true, for any
class of any modulus.  A mirror-walk argument can narrow the search to a class and no further, and
every class holds closed columns; whatever picks the twin columns out of a class is not visible in
the residues.  That is requirement 3 of research/proof/failure_anatomy.md stated as an
impossibility rather than a gap, and it is the machine's own form of the parity obstruction.
-/
import Mathlib

namespace MirrorWalk

/-- **Every class contains a closed column.**  For any modulus `M`, any residue `r`, any gear `p`
that divides neither 6 nor `M`, and any bound `N`, there is a column `m ≥ N` in the class of `r`
whose lower member `6m - 1` is divisible by `p` and larger than `p`. -/
theorem class_has_closed_column {M p : ℕ} (hM : 0 < M) (hp : p.Prime)
    (hp6 : ¬ (p ∣ 6)) (hpM : ¬ (p ∣ M)) (r N : ℕ) :
    ∃ m : ℕ, N ≤ m ∧ m ≡ r [MOD M] ∧ p ∣ (6 * m - 1) ∧ p < 6 * m - 1 := by
  classical
  haveI : Fact p.Prime := ⟨hp⟩
  have hp2 : 2 ≤ p := hp.two_le
  -- 6 M is invertible modulo p
  have hne : ((6 * M : ℕ) : ZMod p) ≠ 0 := by
    intro h
    have : p ∣ 6 * M := (ZMod.natCast_eq_zero_iff _ _).mp h
    rcases (Nat.Prime.dvd_mul hp).mp this with h6 | hMd
    · exact hp6 h6
    · exact hpM hMd
  set u : ZMod p := ((6 * M : ℕ) : ZMod p) with hu
  set s0 : ZMod p := (1 - 6 * (r : ZMod p)) * u⁻¹ with hs0
  set s : ℕ := s0.val with hs
  refine ⟨r + M * s + M * p * (N + 1), ?_, ?_, ?_, ?_⟩
  · have hbig : N + 1 ≤ M * p * (N + 1) := Nat.le_mul_of_pos_left _ (by positivity)
    omega
  · -- the class modulo M is unchanged
    have hdvd : M ∣ (M * s + M * p * (N + 1)) := ⟨s + p * (N + 1), by ring⟩
    have : r + M * s + M * p * (N + 1) = r + (M * s + M * p * (N + 1)) := by ring
    rw [this]
    calc r + (M * s + M * p * (N + 1)) ≡ r + 0 [MOD M] :=
          Nat.ModEq.add_left r ((Nat.modEq_zero_iff_dvd).mpr hdvd)
      _ = r := by ring
  · -- the lower member is struck by p
    have hcast : ((6 * (r + M * s + M * p * (N + 1)) : ℕ) : ZMod p) = 1 := by
      push_cast
      have hps : ((p : ZMod p)) = 0 := ZMod.natCast_self p
      have hsval : ((s : ℕ) : ZMod p) = s0 := by
        rw [hs]; simp [ZMod.natCast_val, ZMod.cast_id]
      rw [hsval, hps]
      have h6M : (6 : ZMod p) * (M : ZMod p) = u := by rw [hu]; push_cast; ring
      have hinv : u * u⁻¹ = 1 := ZMod.mul_inv_of_unit u (Ne.isUnit hne)
      calc 6 * ((r : ZMod p) + (M : ZMod p) * s0 + (M : ZMod p) * 0 * ((N : ZMod p) + 1))
          = 6 * (r : ZMod p) + (6 * (M : ZMod p)) * s0 := by ring
        _ = 6 * (r : ZMod p) + u * ((1 - 6 * (r : ZMod p)) * u⁻¹) := by rw [h6M, hs0]
        _ = 6 * (r : ZMod p) + (u * u⁻¹) * (1 - 6 * (r : ZMod p)) := by ring
        _ = 6 * (r : ZMod p) + (1 - 6 * (r : ZMod p)) := by rw [hinv]; ring
        _ = 1 := by ring
    have hmod : 6 * (r + M * s + M * p * (N + 1)) ≡ 1 [MOD p] := by
      have h1 : ((1 : ℕ) : ZMod p) = 1 := by push_cast; ring
      exact (ZMod.natCast_eq_natCast_iff _ _ _).mp (by rw [hcast, h1])
    have hge : 1 ≤ 6 * (r + M * s + M * p * (N + 1)) := by
      have hbig : 1 ≤ M * p * (N + 1) := Nat.one_le_iff_ne_zero.mpr (by positivity)
      omega
    exact (Nat.modEq_iff_dvd' hge).mp hmod.symm
  · have hbig : p * (N + 1) ≤ M * p * (N + 1) := Nat.mul_le_mul_right _ (Nat.le_mul_of_pos_left _ hM)
    have hpp : p ≤ p * (N + 1) := Nat.le_mul_of_pos_right _ (by omega)
    omega

/-- The column produced above is not a twin column: its lower member has a proper divisor. -/
theorem not_prime_of_dvd_lt {p n : ℕ} (hp : 1 < p) (hdvd : p ∣ n) (hlt : p < n) : ¬ n.Prime := by
  intro h
  rcases (Nat.Prime.eq_one_or_self_of_dvd h p hdvd) with h1 | h2
  · omega
  · omega

/-- **No class is a class of twin columns.**  Combining the two: for every modulus and residue,
some column of that class, as far out as one likes, has a composite lower member. -/
theorem no_class_of_twins {M p : ℕ} (hM : 0 < M) (hp : p.Prime)
    (hp6 : ¬ (p ∣ 6)) (hpM : ¬ (p ∣ M)) (r N : ℕ) :
    ∃ m : ℕ, N ≤ m ∧ m ≡ r [MOD M] ∧ ¬ (6 * m - 1).Prime := by
  obtain ⟨m, hN, hclass, hdvd, hlt⟩ := class_has_closed_column hM hp hp6 hpM r N
  exact ⟨m, hN, hclass, not_prime_of_dvd_lt hp.one_lt hdvd hlt⟩

end MirrorWalk
