/-
Polignac/Goldbach transfer of the blocked-slot reduction.

`BlockedSlots.lean` proves the twin case: infinitude of twin primes is
equivalent to the windowed survivor statement for the pattern {0, 2}. Nothing
in that argument uses the gap 2 beyond bookkeeping, and this file makes that
precise: for EVERY even gap `2*d` the same equivalence holds verbatim, and the
Goldbach frame (`n`, `N - n`) admits the same windowed reduction, with the
converse holding for the representations with both parts above `sqrt N`.

Placement: Ziller-Morack (arXiv:1706.00317, Theorem 4.1) prove their paired-
Jacobsthal bound (Conjecture 6, a single max over all even differences)
SUFFICIENT for Goldbach and for prime pairs at every even difference. The
statements here are per-difference and are equivalences, machine-checked -
the difference-2 case is `BlockedSlots.twins_infinite_iff_survivor_in_window`,
and `survivorGap_one_iff` shows the specialisation is definitional.

Transfer law (slot cap): an odd prime divides both members of a gap-2d slot
iff it divides d. For d = 1 that is never (`Layer.slot_cap`), and in general
exactly the gears dividing d collapse to a single blocked residue - the
mechanical origin of the Hardy-Littlewood factor prod (q-1)/(q-2) over odd
q | d, derived here from divisibility alone.
-/

import BlockedSlots

namespace Polignac

/-! ## The per-member horizon lemma -/

/-- A number above 1 with no prime factor at or below its square root is
prime. The `sqrt`-graded form of `Horizon.prime_of_no_prime_factor_lt`. -/
theorem prime_of_no_factor_le_sqrt {y m : ℕ} (h1 : 1 < m) (hy : Nat.sqrt m ≤ y)
    (h : ∀ q, q.Prime → q ≤ y → ¬ q ∣ m) : m.Prime := by
  by_contra hnp
  have hsq : m.minFac ^ 2 ≤ m := Nat.minFac_sq_le_self (by omega) hnp
  have hle : m.minFac ≤ Nat.sqrt m := by
    have hsq' : m.minFac * m.minFac ≤ m := by simpa [pow_two] using hsq
    exact Nat.le_sqrt.mpr hsq'
  exact h m.minFac (Nat.minFac_prime (by omega)) (le_trans hle hy) (Nat.minFac_dvd m)

/-! ## The gap-2d pattern -/

/-- `SurvivorGap d y m`: no prime `q ≤ y` divides `m` or `m + 2*d` - the
gap-`2d` pattern's survivor description. `d = 1` is `BlockedSlots.Survivor`. -/
def SurvivorGap (d y m : ℕ) : Prop :=
  ∀ q, q.Prime → q ≤ y → ¬ (q ∣ m) ∧ ¬ (q ∣ m + 2 * d)

/-- The specialisation to `d = 1` is definitional. -/
theorem survivorGap_one_iff (y m : ℕ) :
    SurvivorGap 1 y m ↔ BlockedSlots.Survivor y m := by
  constructor <;> intro h q hq hqy <;>
    · have := h q hq hqy
      constructor
      · exact this.1
      · intro hd
        exact this.2 (by simpa [two_mul] using hd)

/-- **Slot-cap transfer law.** An odd prime blocks both members of a gap-`2d`
slot only if it divides `d`: exactly the gears dividing `d` collapse their two
blocked residues to one, which is the mechanical origin of the Hardy-Littlewood
factor for the gap `2d`. -/
theorem slot_cap_gap {d q m : ℕ} (hq : q.Prime) (hq2 : q ≠ 2)
    (h1 : q ∣ m) (h2 : q ∣ m + 2 * d) : q ∣ d := by
  have hsub : q ∣ (m + 2 * d) - m := Nat.dvd_sub h2 h1
  have h2d : q ∣ 2 * d := by simpa using hsub
  rcases (Nat.Prime.dvd_mul hq).mp h2d with hd2 | hdd
  · exact absurd ((Nat.prime_dvd_prime_iff_eq hq Nat.prime_two).mp hd2) hq2
  · exact hdd

/-- The twin case recovered: no odd prime blocks both members of a slot. -/
theorem slot_cap_twin {q m : ℕ} (hq : q.Prime) (hq2 : q ≠ 2)
    (h1 : q ∣ m) (h2 : q ∣ m + 2) : False := by
  have h1' : q ∣ 1 := slot_cap_gap hq hq2 h1 (by simpa [mul_one] using h2)
  exact hq.one_lt.ne' (Nat.dvd_one.mp h1')

/-! ## The windowed equivalence, per gap -/

/-- Inside the certified window a gap-`2d` survivor is exactly a prime pair at
gap `2d` - the gap-`2d` case of `BlockedSlots.survivor_iff_twin`. -/
theorem survivorGap_iff_pair {d y m : ℕ} (hym : y < m) (hwin : m + 2 * d ≤ y * y)
    (h2 : 1 < m) : SurvivorGap d y m ↔ (m.Prime ∧ (m + 2 * d).Prime) := by
  have hy2 : Nat.sqrt (m + 2 * d) ≤ y := by
    have h := Nat.sqrt_le_sqrt hwin
    rwa [Nat.sqrt_eq] at h
  have hy1 : Nat.sqrt m ≤ y := le_trans (Nat.sqrt_le_sqrt (by omega)) hy2
  constructor
  · intro hs
    constructor
    · exact prime_of_no_factor_le_sqrt h2 hy1 fun q hq hqy => (hs q hq hqy).1
    · exact prime_of_no_factor_le_sqrt (by omega) hy2 fun q hq hqy => (hs q hq hqy).2
  · rintro ⟨hp, hp2⟩ q hq hqy
    constructor
    · intro hd
      have : q = m := (hp.eq_one_or_self_of_dvd q hd).resolve_left hq.ne_one
      omega
    · intro hd
      have : q = m + 2 * d := (hp2.eq_one_or_self_of_dvd q hd).resolve_left hq.ne_one
      omega

/-- **The per-gap reduction, both directions.** For every `d`, infinitude of
prime pairs at gap `2*d` is EQUIVALENT to the windowed survivor statement for
the gap-`2d` pattern. Polignac's conjecture for `2d` is the left side; the
right side is the difference-`2d` slice of the paired-Jacobsthal window bound.
`d = 1` is `BlockedSlots.twins_infinite_iff_survivor_in_window`. -/
theorem gapPairs_infinite_iff_survivor_in_window (d : ℕ) :
    {p : ℕ | p.Prime ∧ (p + 2 * d).Prime}.Infinite ↔
      ∀ N, ∃ y, N ≤ y ∧ ∃ m, y < m ∧ m + 2 * d ≤ y * y ∧ SurvivorGap d y m := by
  constructor
  · -- infinitude hands back a survivor in the certified window, at every scale
    intro H N
    have hun : ∀ a : ℕ, ∃ b ∈ {p : ℕ | p.Prime ∧ (p + 2 * d).Prime}, a < b := by
      intro a
      by_contra hc
      push_neg at hc
      exact H (Set.Finite.subset (Set.finite_le_nat a) fun x hx => hc x hx)
    obtain ⟨p, hp, hgt⟩ := hun (N * N + 2 * d + 8)
    obtain ⟨hp1, hp2⟩ := hp
    set y := Nat.sqrt (p + 2 * d) + 1 with hy
    have hwin : p + 2 * d ≤ y * y := le_of_lt (Nat.lt_succ_sqrt (p + 2 * d))
    have hlow : y < p := by
      have hq7 : 7 ≤ p - 1 := by omega
      have h2p : 2 * p < (p - 1) * (p - 1) :=
        lt_of_lt_of_le (by omega : 2 * p < 7 * (p - 1)) (Nat.mul_le_mul hq7 le_rfl)
      have hlt : p + 2 * d < (p - 1) * (p - 1) := by omega
      have hs : Nat.sqrt (p + 2 * d) < p - 1 := Nat.sqrt_lt.mpr hlt
      omega
    have hN : N ≤ y := by
      have hle : N * N ≤ p + 2 * d := by omega
      have h3 := Nat.sqrt_le_sqrt hle
      rw [Nat.sqrt_eq] at h3
      omega
    exact ⟨y, hN, p, hlow, hwin,
      (survivorGap_iff_pair hlow hwin (by omega)).mpr ⟨hp1, hp2⟩⟩
  · -- a survivor at every scale forces infinitude
    intro H
    apply Set.infinite_of_forall_exists_gt
    intro a
    obtain ⟨y, hay, m, hym, hwin, hs⟩ := H (a + 2)
    refine ⟨m, ?_, by omega⟩
    exact (survivorGap_iff_pair hym hwin (by omega)).mp hs

/-! ## The Goldbach frame

For even `N` the pattern is (`n`, `N - n`): the blocked residues of gear `q`
are `0` and `N mod q`, collapsing to one exactly when `q ∣ N` - the same
two-residue sieve with the collapse condition `q ∣ N` in place of `q ∣ d`.
The horizon works unchanged: both parts above `sqrt N` and free of factors at
or below `sqrt N` are prime. -/

/-- **Goldbach window reduction.** A survivor of the paired sieve with both
parts above `sqrt N` is a Goldbach representation of `N`. -/
theorem goldbach_of_survivor {N n : ℕ} (hlo : Nat.sqrt N < n)
    (hhi : Nat.sqrt N < N - n)
    (hs : ∀ q, q.Prime → q ≤ Nat.sqrt N → ¬ q ∣ n ∧ ¬ q ∣ (N - n)) :
    n.Prime ∧ (N - n).Prime ∧ n + (N - n) = N := by
  have hnN : n < N := by omega
  have hup := Nat.lt_succ_sqrt N
  have hpos : 1 ≤ Nat.sqrt N := by
    rcases Nat.eq_zero_or_pos (Nat.sqrt N) with h0 | h1
    · exfalso
      rw [h0] at hup
      simp at hup
      omega
    · exact h1
  have hpn : n.Prime :=
    prime_of_no_factor_le_sqrt (by omega) (Nat.sqrt_le_sqrt (le_of_lt hnN))
      fun q hq hqy => (hs q hq hqy).1
  have hpn' : (N - n).Prime :=
    prime_of_no_factor_le_sqrt (by omega) (Nat.sqrt_le_sqrt (by omega))
      fun q hq hqy => (hs q hq hqy).2
  exact ⟨hpn, hpn', by omega⟩

/-- Existence form: such a survivor makes `N` a sum of two primes. -/
theorem goldbach_rep_of_survivor {N n : ℕ} (hlo : Nat.sqrt N < n)
    (hhi : Nat.sqrt N < N - n)
    (hs : ∀ q, q.Prime → q ≤ Nat.sqrt N → ¬ q ∣ n ∧ ¬ q ∣ (N - n)) :
    ∃ p p', p.Prime ∧ p'.Prime ∧ p + p' = N := by
  obtain ⟨h1, h2, h3⟩ := goldbach_of_survivor hlo hhi hs
  exact ⟨n, N - n, h1, h2, h3⟩

/-- The converse on the central representations: a Goldbach representation with
both parts above `sqrt N` IS a survivor of the paired sieve, so the reduction
loses nothing there - the window statement is equivalent to "N has a
representation with both parts above its square root". -/
theorem survivor_of_goldbach_rep {N p p' : ℕ} (hp : p.Prime) (hp' : p'.Prime)
    (hsum : p + p' = N) (hlo : Nat.sqrt N < p) (hlo' : Nat.sqrt N < p') :
    Nat.sqrt N < p ∧ Nat.sqrt N < N - p ∧
      ∀ q, q.Prime → q ≤ Nat.sqrt N → ¬ q ∣ p ∧ ¬ q ∣ (N - p) := by
  have hNp : N - p = p' := by omega
  refine ⟨hlo, by omega, ?_⟩
  intro q hq hqy
  rw [hNp]
  constructor
  · intro hd
    have : q = p := (hp.eq_one_or_self_of_dvd q hd).resolve_left hq.ne_one
    omega
  · intro hd
    have : q = p' := (hp'.eq_one_or_self_of_dvd q hd).resolve_left hq.ne_one
    omega

end Polignac
