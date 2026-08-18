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
import Census
import Mathlib.Data.Nat.ModEq

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

/-! ## The g = 2 pinning: the split class of a twin pair

The g = 2 slice of the gap-graded split law (research/split_gap_law.py): for a
gear pair (q, q + g) the split double-kill class "q kills left, q + g kills
right" has least representative at depth ~P/(6g) conditionally on mod-6
alignment - EXCEPT at g = 2, where m0 = 0 and the class pins unconditionally
at the pair's own slot u = (p+1)/6 <= (y+1)/6, the bottom band of every window
at every scale. This is the first exact fact that distinguishes twins from
every other gap inside the general frame: twins below y are the unique gap
class with unconditionally guaranteed contribution to the level-y^2 doubles
ledger. Formalised here: the pin (existence, closed form, location), the full
CRT class as an iff, the mirror class, the product slot, and the uniqueness of
the own-slot pin (only g = 2 pins there).

Slot vocabulary: slot k holds the pair (6k - 1, 6k + 1). -/

/-- A twin pair above 3 sits at 5 mod 6 (so its slot coordinate is exact). -/
theorem twin_mod_six {p : ℕ} (hp : p.Prime) (hp2 : (p + 2).Prime) (h3 : 3 < p) :
    p % 6 = 5 := by
  have h2 : ¬ 2 ∣ p := fun hd =>
    by have := (hp.eq_one_or_self_of_dvd 2 hd).resolve_left (by norm_num); omega
  have hd3 : ¬ 3 ∣ p := fun hd =>
    by have := (hp.eq_one_or_self_of_dvd 3 hd).resolve_left (by norm_num); omega
  have hd3' : ¬ 3 ∣ (p + 2) := fun hd =>
    by have := (hp2.eq_one_or_self_of_dvd 3 hd).resolve_left (by norm_num); omega
  -- single-modulus decomposition p = 6k + r, then explicit divisibility
  -- witnesses per residue (omega does not combine congruences across moduli)
  obtain ⟨k, r, hkr, hr⟩ : ∃ k r, p = 6 * k + r ∧ r < 6 :=
    ⟨p / 6, p % 6, by omega, by omega⟩
  interval_cases r
  · exact absurd ⟨3 * k, by omega⟩ h2
  · exact absurd ⟨2 * k + 1, by omega⟩ hd3'
  · exact absurd ⟨3 * k + 1, by omega⟩ h2
  · exact absurd ⟨2 * k + 1, by omega⟩ hd3
  · exact absurd ⟨3 * k + 2, by omega⟩ h2
  · omega

/-- **The pin.** A twin pair (p, p+2), p > 3, IS the slot u = (p+1)/6: closed
form for the split representative, with p killing the left member and p + 2
the right - both trivially, because the members ARE the pair. -/
theorem twin_pin {p : ℕ} (hp : p.Prime) (hp2 : (p + 2).Prime) (h3 : 3 < p) :
    ∃ u, 0 < u ∧ 6 * u = p + 1 ∧ 6 * u - 1 = p ∧ 6 * u + 1 = p + 2 ∧
      p ∣ 6 * u - 1 ∧ (p + 2) ∣ 6 * u + 1 := by
  have h6 := twin_mod_six hp hp2 h3
  refine ⟨(p + 1) / 6, by omega, by omega, by omega, by omega, ⟨1, by omega⟩,
    ⟨1, by omega⟩⟩

/-- **Bottom-band location.** The pin depth is `u ≤ (y+1)/6` for EVERY machine
scale `y ≥ p`: the guaranteed double sits in the bottom band of every window,
at every scale - unconditionally. -/
theorem twin_pin_le {p u y : ℕ} (hu : 6 * u = p + 1) (hpy : p ≤ y) :
    u ≤ (y + 1) / 6 := by omega

/-- **The class.** Slot k is a split double-kill of the twin pair {p, p+2}
(p kills the left member, p + 2 the right) exactly on the CRT class of the pin
u modulo P = p(p+2). The g = 2 case of the roots-of-unity law, as an iff. -/
theorem twin_split_class_iff {p u k : ℕ} (hp : p.Prime) (hp2 : (p + 2).Prime)
    (h3 : 3 < p) (hu : 6 * u = p + 1) (hk : 1 ≤ k) :
    (p ∣ 6 * k - 1 ∧ (p + 2) ∣ 6 * k + 1) ↔ k ≡ u [MOD p * (p + 2)] := by
  have hu1 : 1 ≤ u := by omega
  -- neither member of the pair divides 6
  have hnd6 : ¬ p ∣ 6 := by
    intro hd
    have hle := Nat.le_of_dvd (by norm_num) hd
    interval_cases p <;> revert hd <;> revert hp <;> decide
  have hnd6' : ¬ (p + 2) ∣ 6 := by
    intro hd
    have hle := Nat.le_of_dvd (by norm_num) hd
    have hp4 : p = 4 := by omega
    subst hp4
    exact absurd hp (by decide)
  have hcop6 : Nat.Coprime p 6 := (Nat.Prime.coprime_iff_not_dvd hp).mpr hnd6
  have hcop6' : Nat.Coprime (p + 2) 6 := (Nat.Prime.coprime_iff_not_dvd hp2).mpr hnd6'
  have hcopP : Nat.Coprime p (p + 2) := (Nat.coprime_primes hp hp2).mpr (by omega)
  have hpu : p ∣ 6 * u - 1 := ⟨1, by omega⟩
  have hp2u : (p + 2) ∣ 6 * u + 1 := ⟨1, by omega⟩
  constructor
  · rintro ⟨hL, hR⟩
    rcases Nat.lt_or_ge k u with hlt | hle
    · -- k below the pin is impossible: p + 2 would divide 0 < 6(u-k) < 6u = p+1
      exfalso
      have hd6 : (p + 2) ∣ 6 * (u - k) := by
        have h := Nat.dvd_sub hp2u hR
        rwa [show (6 * u + 1) - (6 * k + 1) = 6 * (u - k) by omega] at h
      have hd' : (p + 2) ∣ u - k := by
        have h' : (p + 2) ∣ (u - k) * 6 := by rwa [Nat.mul_comm] at hd6
        exact Nat.Coprime.dvd_of_dvd_mul_right hcop6' h'
      have := Nat.le_of_dvd (by omega) hd'
      omega
    · -- both gears divide 6(k - u), hence k - u; coprimality lifts to P
      have hdL6 : p ∣ 6 * (k - u) := by
        have h := Nat.dvd_sub hL hpu
        rwa [show (6 * k - 1) - (6 * u - 1) = 6 * (k - u) by omega] at h
      have hdR6 : (p + 2) ∣ 6 * (k - u) := by
        have h := Nat.dvd_sub hR hp2u
        rwa [show (6 * k + 1) - (6 * u + 1) = 6 * (k - u) by omega] at h
      have hdL : p ∣ k - u := by
        have h' : p ∣ (k - u) * 6 := by rwa [Nat.mul_comm] at hdL6
        exact Nat.Coprime.dvd_of_dvd_mul_right hcop6 h'
      have hdR : (p + 2) ∣ k - u := by
        have h' : (p + 2) ∣ (k - u) * 6 := by rwa [Nat.mul_comm] at hdR6
        exact Nat.Coprime.dvd_of_dvd_mul_right hcop6' h'
      have hP : p * (p + 2) ∣ k - u :=
        Nat.Coprime.mul_dvd_of_dvd_of_dvd hcopP hdL hdR
      exact ((Nat.modEq_iff_dvd' hle).mpr hP).symm
  · intro hmod
    -- k = u + P t, and both divisibilities are explicit
    have hA : p + 1 ≤ p * (p + 2) := by
      have h := Nat.mul_le_mul (le_refl p) (show 6 ≤ p + 2 by omega)
      omega
    have hku : u ≤ k := by
      have h1 : k % (p * (p + 2)) = u % (p * (p + 2)) := hmod
      have h2 : u % (p * (p + 2)) = u := Nat.mod_eq_of_lt (by omega)
      have h3' : k % (p * (p + 2)) ≤ k := Nat.mod_le k _
      omega
    obtain ⟨t, ht⟩ := (Nat.modEq_iff_dvd' hku).mp hmod.symm
    constructor
    · have e : 6 * k - 1 = p + 6 * (p * (p + 2) * t) := by omega
      rw [e]
      exact Nat.dvd_add (dvd_refl p) (((dvd_mul_right p (p + 2)).mul_right t).mul_left 6)
    · have e : 6 * k + 1 = (p + 2) + 6 * (p * (p + 2) * t) := by omega
      rw [e]
      exact Nat.dvd_add (dvd_refl (p + 2))
        (((dvd_mul_left (p + 2) p).mul_right t).mul_left 6)

/-- **Mirror class.** The other split class (p kills right, p + 2 kills left)
sits at P - u: with 6(P - u) - 1 = 6P - (p + 2) and 6(P - u) + 1 = 6P - p,
both divisibilities are explicit. -/
theorem twin_mirror_slot {p u : ℕ} (hu : 6 * u = p + 1) :
    p ∣ 6 * (p * (p + 2) - u) + 1 ∧ (p + 2) ∣ 6 * (p * (p + 2) - u) - 1 := by
  have hA : p + 1 ≤ p * (p + 2) := by
    have h := Nat.mul_le_mul (le_refl p) (show 6 ≤ p + 2 by omega)
    omega
  constructor
  · have e : 6 * (p * (p + 2) - u) + 1 = p * (6 * (p + 2)) - p := by
      have e1 : p * (6 * (p + 2)) = 6 * (p * (p + 2)) := by ring
      omega
    rw [e]
    exact Nat.dvd_sub (dvd_mul_right p (6 * (p + 2))) (dvd_refl p)
  · have e : 6 * (p * (p + 2) - u) - 1 = (p + 2) * (6 * p) - (p + 2) := by
      have e1 : (p + 2) * (6 * p) = 6 * (p * (p + 2)) := by ring
      omega
    rw [e]
    exact Nat.dvd_sub (dvd_mul_right (p + 2) (6 * p)) (dvd_refl (p + 2))

/-- **Product slot.** The pair's second guaranteed slot: at kp = u(p+1) BOTH
gears strike the same left member, which is the semiprime p(p+2) itself -
the machine re-ingesting its own output. -/
theorem twin_product_slot {p u : ℕ} (hu : 6 * u = p + 1) :
    6 * (u * (p + 1)) - 1 = p * (p + 2) ∧
      p ∣ 6 * (u * (p + 1)) - 1 ∧ (p + 2) ∣ 6 * (u * (p + 1)) - 1 := by
  have h1 : 6 * (u * (p + 1)) = (6 * u) * (p + 1) := by ring
  have h2 : (p + 1) * (p + 1) = p * (p + 2) + 1 := by ring
  have e : 6 * (u * (p + 1)) - 1 = p * (p + 2) := by
    rw [h1, hu]
    omega
  refine ⟨e, ?_, ?_⟩
  · rw [e]; exact dvd_mul_right p (p + 2)
  · rw [e]; exact dvd_mul_left (p + 2) p

/-- **Uniqueness of the own-slot pin.** If a prime pair (q, q + g), both odd,
split-kills the slot holding q itself (6k - 1 = q and (q + g) ∣ 6k + 1), then
g = 2: only twin pairs pin at their own slot. Every other gap's split class
sits strictly deeper - alignment-conditional, never guaranteed. -/
theorem own_slot_pin_gap_two {q g k : ℕ} (hq : q.Prime) (hgp : (q + g).Prime)
    (h3 : 3 < q) (hg : 0 < g) (hslot : 6 * k - 1 = q) (hk : 1 ≤ k)
    (hd : (q + g) ∣ 6 * k + 1) : g = 2 := by
  have h1 : 6 * k + 1 = q + 2 := by omega
  rw [h1] at hd
  have hle : q + g ≤ q + 2 := Nat.le_of_dvd (by omega) hd
  -- both primes are odd, so the gap is even; with 0 < g ≤ 2 that forces g = 2
  have hodd : ¬ 2 ∣ q := fun hdd =>
    by have := (hq.eq_one_or_self_of_dvd 2 hdd).resolve_left (by norm_num); omega
  have hodd' : ¬ 2 ∣ (q + g) := fun hdd =>
    by have := (hgp.eq_one_or_self_of_dvd 2 hdd).resolve_left (by norm_num); omega
  obtain ⟨m, hm⟩ : ∃ m, q = 2 * m + 1 := ⟨q / 2, by omega⟩
  obtain ⟨n, hn⟩ : ∃ n, q + g = 2 * n + 1 := ⟨(q + g) / 2, by omega⟩
  omega

/-! ## The SAME-side census: one CRT class and its floor count

First layer of the master supply formula (Lateral round 4): for any modulus m
coprime to 6 - in particular any squarefree gear product - the slots whose
left member 6k-1 is divisible by m form ONE residue class mod m (the slot map
is invertible), and the count of that class among the first t slots is
closed-form floor arithmetic. Instantiated at m = q*r this is the SAME-side
pair census; the window corollary is the composite root law's "exactly once
if it fits", with the window hypotheses explicit. -/

/-- Class representatives below the modulus are unique. -/
theorem class_rep_unique {m a b : ℕ} (ha : a < m) (hb : b < m)
    (h : a ≡ b [MOD m]) : a = b := by
  have h1 : a % m = a := Nat.mod_eq_of_lt ha
  have h2 : b % m = b := Nat.mod_eq_of_lt hb
  have h3 : a % m = b % m := h
  omega

/-- Inverting the slot map: for m coprime to 6 and any target residue c, the
k with 6k ≡ c (mod m) form exactly one class mod m. -/
theorem six_mul_class {m : ℕ} (c : ℕ) (hco : Nat.Coprime 6 m) (hm : 1 < m) :
    ∃ a, a < m ∧ 6 * a ≡ c [MOD m] ∧
      ∀ k, (6 * k ≡ c [MOD m] ↔ k ≡ a [MOD m]) := by
  obtain ⟨b, hbm, hb⟩ := Nat.exists_mul_mod_eq_of_coprime c hco (by omega)
  have hb' : 6 * b ≡ c [MOD m] := hb
  refine ⟨b, hbm, hb', fun k => ⟨?_, ?_⟩⟩
  · intro hk
    exact Nat.ModEq.cancel_left_of_coprime (Nat.Coprime.symm hco) (hk.trans hb'.symm)
  · intro hk
    exact (Nat.ModEq.mul_left 6 hk).trans hb'

/-- Left-member divisibility is the residue condition 6k ≡ 1, for k ≥ 1. -/
theorem left_dvd_iff {m k : ℕ} (hk : 1 ≤ k) :
    m ∣ 6 * k - 1 ↔ 6 * k ≡ 1 [MOD m] := by
  rw [← Nat.modEq_iff_dvd' (by omega : 1 ≤ 6 * k)]
  exact ⟨Nat.ModEq.symm, Nat.ModEq.symm⟩

/-- Right-member divisibility is the residue condition 6k ≡ m - 1. -/
theorem right_dvd_iff {m k : ℕ} (hm : 1 ≤ m) :
    m ∣ 6 * k + 1 ↔ 6 * k ≡ m - 1 [MOD m] := by
  constructor
  · intro hd
    have h0 : 6 * k + 1 ≡ 0 [MOD m] := Nat.modEq_zero_iff_dvd.mpr hd
    have h1 : (0 : ℕ) ≡ (m - 1) + 1 [MOD m] := by
      rw [show (m - 1) + 1 = m by omega]
      exact (Nat.modEq_zero_iff_dvd.mpr (dvd_refl m)).symm
    exact Nat.ModEq.add_right_cancel' 1 (h0.trans h1)
  · intro hmod
    have h2 : 6 * k + 1 ≡ (m - 1) + 1 [MOD m] := hmod.add_right 1
    have h4 : ((m - 1) + 1 : ℕ) ≡ 0 [MOD m] := by
      rw [show (m - 1) + 1 = m by omega]
      exact Nat.modEq_zero_iff_dvd.mpr (dvd_refl m)
    exact Nat.modEq_zero_iff_dvd.mp (h2.trans h4)

/-- **The floor count.** Among the slots 1..t, the residue class of a
(mod m), 1 ≤ a ≤ m, has exactly (t + m - a) / m members. -/
theorem card_class_Ico {m a : ℕ} (hm : 0 < m) (ha1 : 1 ≤ a) (ham : a ≤ m)
    (t : ℕ) :
    ((Finset.Ico 1 (t + 1)).filter (fun k => k % m = a % m)).card
      = (t + m - a) / m := by
  induction t with
  | zero =>
    simp only [Nat.zero_add]
    rw [Finset.Ico_self, Finset.filter_empty, Finset.card_empty]
    exact (Nat.div_eq_of_lt (by omega)).symm
  | succ n ih =>
    have hins : Finset.Ico 1 (n + 1 + 1) = insert (n + 1) (Finset.Ico 1 (n + 1)) :=
      Nat.Ico_succ_right_eq_insert_Ico (by omega)
    rw [hins, Finset.filter_insert]
    by_cases hc : (n + 1) % m = a % m
    · rw [if_pos hc]
      have hnot : (n + 1) ∉ (Finset.Ico 1 (n + 1)).filter
          (fun k => k % m = a % m) := by
        intro hmem
        have h1 := (Finset.mem_filter.mp hmem).1
        have h2 := (Finset.mem_Ico.mp h1).2
        omega
      rw [Finset.card_insert_of_notMem hnot, ih]
      have h1 : a ≡ n + 1 + m [MOD m] := by
        have hx : (n + 1) ≡ a [MOD m] := hc
        have hy : (n + 1 + m) ≡ (n + 1) [MOD m] := Nat.add_mod_right (n + 1) m
        exact (hy.trans hx).symm
      have hdvd : m ∣ (n + m - a) + 1 := by
        have h0 : m ∣ (n + 1 + m) - a := (Nat.modEq_iff_dvd' (by omega)).mp h1
        rwa [show (n + 1 + m) - a = (n + m - a) + 1 by omega] at h0
      rw [show n + 1 + m - a = (n + m - a) + 1 by omega,
        Nat.succ_div_of_dvd hdvd]
    · rw [if_neg hc, ih]
      have hnd : ¬ m ∣ (n + m - a) + 1 := by
        intro h0
        have h1 : m ∣ (n + 1 + m) - a := by
          rwa [show (n + m - a) + 1 = (n + 1 + m) - a by omega] at h0
        have h2 : a ≡ n + 1 + m [MOD m] := (Nat.modEq_iff_dvd' (by omega)).mpr h1
        have hy : (n + 1 + m) ≡ (n + 1) [MOD m] := Nat.add_mod_right (n + 1) m
        exact hc ((h2.trans hy).symm)
      rw [show n + 1 + m - a = (n + m - a) + 1 by omega,
        Nat.succ_div_of_not_dvd hnd]

/-- A prime at or above 5 does not divide 6. -/
theorem not_dvd_six {q : ℕ} (hq : q.Prime) (hq5 : 5 ≤ q) : ¬ q ∣ 6 := by
  intro hd
  have hle := Nat.le_of_dvd (by norm_num) hd
  interval_cases q <;> revert hd <;> revert hq <;> decide

/-- **SAME-side pair census, left member.** For distinct primes q, r ≥ 5, the
slots whose LEFT member 6k-1 is divisible by both are exactly one CRT class
mod q*r, and the class count over the first t slots is (t + qr - a) / qr -
pure floor arithmetic. -/
theorem same_left_census {q r : ℕ} (hq : q.Prime) (hr : r.Prime)
    (hq5 : 5 ≤ q) (hr5 : 5 ≤ r) (hne : q ≠ r) :
    ∃ a, 1 ≤ a ∧ a < q * r ∧
      (∀ k, 1 ≤ k → ((q ∣ 6 * k - 1 ∧ r ∣ 6 * k - 1) ↔ k ≡ a [MOD q * r])) ∧
      (∀ t, ((Finset.Ico 1 (t + 1)).filter
          (fun k => k % (q * r) = a % (q * r))).card
        = (t + q * r - a) / (q * r)) := by
  have h25 : 25 ≤ q * r := by
    calc 25 = 5 * 5 := by norm_num
    _ ≤ q * r := Nat.mul_le_mul hq5 hr5
  have hco6 : Nat.Coprime 6 (q * r) :=
    Nat.Coprime.mul_right
      (Nat.Coprime.symm ((Nat.Prime.coprime_iff_not_dvd hq).mpr (not_dvd_six hq hq5)))
      (Nat.Coprime.symm ((Nat.Prime.coprime_iff_not_dvd hr).mpr (not_dvd_six hr hr5)))
  have hcoqr : Nat.Coprime q r := (Nat.coprime_primes hq hr).mpr hne
  obtain ⟨a, ham, ha6, hiff⟩ := six_mul_class 1 hco6 (by omega)
  have ha1 : 1 ≤ a := by
    rcases Nat.eq_zero_or_pos a with h0 | h1
    · exfalso
      subst h0
      have h6 : (0 : ℕ) ≡ 1 [MOD q * r] := by simpa using ha6
      have hd : q * r ∣ 1 := by
        have h7 := (Nat.modEq_iff_dvd' (by omega)).mp h6
        simpa using h7
      have := Nat.le_of_dvd (by norm_num) hd
      omega
    · exact h1
  refine ⟨a, ha1, ham, ?_, ?_⟩
  · intro k hk
    have hboth : (q ∣ 6 * k - 1 ∧ r ∣ 6 * k - 1) ↔ q * r ∣ 6 * k - 1 :=
      ⟨fun h => hcoqr.mul_dvd_of_dvd_of_dvd h.1 h.2,
       fun h => ⟨dvd_trans (dvd_mul_right q r) h, dvd_trans (dvd_mul_left r q) h⟩⟩
    rw [hboth, left_dvd_iff hk]
    exact hiff k
  · intro t
    exact card_class_Ico (by omega) ha1 (by omega) t

/-- **SAME-side pair census, right member.** The mirror statement for 6k+1,
with target residue -1: one CRT class, same floor count. No k ≥ 1 hypothesis
is needed (slot 0's right member is 1, divisible by nothing above 1). -/
theorem same_right_census {q r : ℕ} (hq : q.Prime) (hr : r.Prime)
    (hq5 : 5 ≤ q) (hr5 : 5 ≤ r) (hne : q ≠ r) :
    ∃ a, 1 ≤ a ∧ a < q * r ∧
      (∀ k, ((q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1) ↔ k ≡ a [MOD q * r])) ∧
      (∀ t, ((Finset.Ico 1 (t + 1)).filter
          (fun k => k % (q * r) = a % (q * r))).card
        = (t + q * r - a) / (q * r)) := by
  have h25 : 25 ≤ q * r := by
    calc 25 = 5 * 5 := by norm_num
    _ ≤ q * r := Nat.mul_le_mul hq5 hr5
  have hco6 : Nat.Coprime 6 (q * r) :=
    Nat.Coprime.mul_right
      (Nat.Coprime.symm ((Nat.Prime.coprime_iff_not_dvd hq).mpr (not_dvd_six hq hq5)))
      (Nat.Coprime.symm ((Nat.Prime.coprime_iff_not_dvd hr).mpr (not_dvd_six hr hr5)))
  have hcoqr : Nat.Coprime q r := (Nat.coprime_primes hq hr).mpr hne
  obtain ⟨a, ham, ha6, hiff⟩ := six_mul_class (q * r - 1) hco6 (by omega)
  have ha1 : 1 ≤ a := by
    rcases Nat.eq_zero_or_pos a with h0 | h1
    · exfalso
      subst h0
      have h6 : (0 : ℕ) ≡ q * r - 1 [MOD q * r] := by simpa using ha6
      have hd : q * r ∣ q * r - 1 := by
        have h7 := (Nat.modEq_iff_dvd' (by omega)).mp h6
        simpa using h7
      have := Nat.le_of_dvd (by omega) hd
      omega
    · exact h1
  refine ⟨a, ha1, ham, ?_, ?_⟩
  · intro k
    have hboth : (q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1) ↔ q * r ∣ 6 * k + 1 :=
      ⟨fun h => hcoqr.mul_dvd_of_dvd_of_dvd h.1 h.2,
       fun h => ⟨dvd_trans (dvd_mul_right q r) h, dvd_trans (dvd_mul_left r q) h⟩⟩
    rw [hboth, right_dvd_iff (by omega : 1 ≤ q * r)]
    exact hiff k
  · intro t
    exact card_class_Ico (by omega) ha1 (by omega) t

/-- **Composite root law, windowed ("exactly once if it fits").** If the class
representative a lands within the first t slots and the window stops before
the next period return a + P, the coincidence occurs EXACTLY once. -/
theorem same_census_once {P a t : ℕ} (hP : 0 < P) (ha1 : 1 ≤ a) (haP : a ≤ P)
    (hat : a ≤ t) (htw : t < a + P) :
    ((Finset.Ico 1 (t + 1)).filter (fun k => k % P = a % P)).card = 1 := by
  rw [card_class_Ico hP ha1 haP t]
  exact Nat.div_eq_of_lt_le (by omega) (by omega)

/-- **The product acts at its own value** (left member): when q*r = 5 mod 6,
the slot (q*r + 1)/6 holds q*r itself as left member, and both gears strike
it there - the class representative made explicit. -/
theorem same_left_own_value {q r : ℕ} (h25 : 25 ≤ q * r) (h56 : q * r % 6 = 5) :
    1 ≤ (q * r + 1) / 6 ∧ (q * r + 1) / 6 < q * r ∧
      6 * ((q * r + 1) / 6) - 1 = q * r ∧
      q ∣ 6 * ((q * r + 1) / 6) - 1 ∧ r ∣ 6 * ((q * r + 1) / 6) - 1 := by
  refine ⟨by omega, by omega, by omega, ?_, ?_⟩
  · rw [show 6 * ((q * r + 1) / 6) - 1 = q * r by omega]
    exact dvd_mul_right q r
  · rw [show 6 * ((q * r + 1) / 6) - 1 = q * r by omega]
    exact dvd_mul_left r q

/-! ## The PAIRSPLIT census: the split (cross-member) class

The other layer of the master supply formula: for distinct gears q, r the
slots where q strikes the LEFT member and r the RIGHT (q | 6k-1, r | 6k+1)
form one CRT class mod qr - a nontrivial root of 36k^2 ≡ 1, in the summary's
language. The mirror class (r left, q right) is the same theorem with the
roles swapped. The count is card_class_Ico again. The g = 2 specialisation
closes the loop with the pinning section: for a twin pair the split
representative IS the pin u = (p+1)/6 (`split_rep_twin_eq_pin`). -/

/-- **The split class.** For distinct primes q, r ≥ 5, the slots whose left
member q divides and whose right member r divides are exactly one CRT class
mod q*r, with the floor count over the first t slots. Swapping q and r gives
the mirror class. -/
theorem split_class {q r : ℕ} (hq : q.Prime) (hr : r.Prime)
    (hq5 : 5 ≤ q) (hr5 : 5 ≤ r) (hne : q ≠ r) :
    ∃ a, 1 ≤ a ∧ a < q * r ∧
      (∀ k, 1 ≤ k → ((q ∣ 6 * k - 1 ∧ r ∣ 6 * k + 1) ↔ k ≡ a [MOD q * r])) ∧
      (∀ t, ((Finset.Ico 1 (t + 1)).filter
          (fun k => k % (q * r) = a % (q * r))).card
        = (t + q * r - a) / (q * r)) := by
  have h25 : 25 ≤ q * r := by
    calc 25 = 5 * 5 := by norm_num
    _ ≤ q * r := Nat.mul_le_mul hq5 hr5
  have hco6 : Nat.Coprime 6 (q * r) :=
    Nat.Coprime.mul_right
      (Nat.Coprime.symm ((Nat.Prime.coprime_iff_not_dvd hq).mpr (not_dvd_six hq hq5)))
      (Nat.Coprime.symm ((Nat.Prime.coprime_iff_not_dvd hr).mpr (not_dvd_six hr hr5)))
  have hcoqr : Nat.Coprime q r := (Nat.coprime_primes hq hr).mpr hne
  -- the joint target residue: c ≡ 1 (mod q), c ≡ r - 1 (mod r)
  obtain ⟨c, hc1, hc2⟩ := Nat.chineseRemainder hcoqr 1 (r - 1)
  obtain ⟨a, ham, ha6, hiff6⟩ := six_mul_class c hco6 (by omega)
  have ha1 : 1 ≤ a := by
    rcases Nat.eq_zero_or_pos a with h0 | h1
    · exfalso
      subst h0
      have h6 : (0 : ℕ) ≡ c [MOD q * r] := by simpa using ha6
      have hq0 : (0 : ℕ) ≡ 1 [MOD q] :=
        (Nat.ModEq.of_dvd (dvd_mul_right q r) h6).trans hc1
      have hd : q ∣ 1 := by
        have h7 := (Nat.modEq_iff_dvd' (by omega)).mp hq0
        simpa using h7
      have := Nat.le_of_dvd (by norm_num) hd
      omega
    · exact h1
  refine ⟨a, ha1, ham, ?_, ?_⟩
  · intro k hk
    rw [left_dvd_iff hk, right_dvd_iff (by omega : 1 ≤ r)]
    constructor
    · rintro ⟨hL, hR⟩
      have hqr : 6 * k ≡ c [MOD q * r] :=
        (Nat.modEq_and_modEq_iff_modEq_mul hcoqr).mp
          ⟨hL.trans hc1.symm, hR.trans hc2.symm⟩
      exact (hiff6 k).mp hqr
    · intro hka
      have hqr : 6 * k ≡ c [MOD q * r] := (hiff6 k).mpr hka
      exact ⟨(Nat.ModEq.of_dvd (dvd_mul_right q r) hqr).trans hc1,
        (Nat.ModEq.of_dvd (dvd_mul_left r q) hqr).trans hc2⟩
  · intro t
    exact card_class_Ico (by omega) ha1 (by omega) t

/-- **g = 2 closes the loop.** For a twin pair (p, p+2), any representative
below the modulus of the split class is the pin u = (p+1)/6: the PAIRSPLIT
representative of a twin pair IS its own slot. With `twin_pin_le` this is the
formal "twins below y are the unique unconditionally guaranteed line item of
the doubles ledger": every other pair's representative is not anchored. -/
theorem split_rep_twin_eq_pin {p u a : ℕ} (hp : p.Prime) (hp2 : (p + 2).Prime)
    (h3 : 3 < p) (hu : 6 * u = p + 1) (ha : a < p * (p + 2))
    (hiff : ∀ k, 1 ≤ k →
      ((p ∣ 6 * k - 1 ∧ (p + 2) ∣ 6 * k + 1) ↔ k ≡ a [MOD p * (p + 2)])) :
    a = u := by
  have h6 := twin_mod_six hp hp2 h3
  have hu1 : 1 ≤ u := by omega
  have hsplit : p ∣ 6 * u - 1 ∧ (p + 2) ∣ 6 * u + 1 :=
    ⟨⟨1, by omega⟩, ⟨1, by omega⟩⟩
  have hmod := (hiff u hu1).mp hsplit
  have huP : u < p * (p + 2) := by
    have h := Nat.mul_le_mul (le_refl p) (show 6 ≤ p + 2 by omega)
    omega
  exact class_rep_unique ha huP hmod.symm

/-- The twin pair's split count over the first t slots, anchored at the pin:
(t + p(p+2) - u) / (p(p+2)) - equal to 1 as soon as u ≤ t < u + p(p+2), the
guaranteed bottom-band double, counted. -/
theorem twin_split_count {p u : ℕ} (hp : p.Prime) (hp2 : (p + 2).Prime)
    (h3 : 3 < p) (hu : 6 * u = p + 1) (t : ℕ) :
    ((Finset.Ico 1 (t + 1)).filter
        (fun k => k % (p * (p + 2)) = u % (p * (p + 2)))).card
      = (t + p * (p + 2) - u) / (p * (p + 2)) := by
  have hu1 : 1 ≤ u := by omega
  have h := Nat.mul_le_mul (le_refl p) (show 6 ≤ p + 2 by omega)
  exact card_class_Ico (by omega) hu1 (by omega) t

/-! ## The CORR layer: two-sided product classes and the signed triple

The last structural layer of the master supply formula. A both-sided term
(s_L | 6k-1, s_R | 6k+1) with s_L, s_R coprime squarefree gear products is
again ONE CRT class mod s_L * s_R - `twoSided_class` proves it for arbitrary
coprime moduli, subsuming `split_class` (both prime) and giving every
CORR term. The first genuinely new case is the triple (3 gears, both-sided):
`corr_triple_class` instantiates s_L = q*r, s_R = s, and `corr_triple_signed`
is the inclusion-exclusion identity with the sign realised subtraction-free:
distinct doubles |A ∪ B| plus the triple overlap equal the two PAIRSPLIT
incidence counts - the triple class is exactly what the signed sum removes. -/

/-- A prime at or above 5 is coprime to 6. -/
theorem six_coprime_prime {q : ℕ} (hq : q.Prime) (hq5 : 5 ≤ q) :
    Nat.Coprime 6 q :=
  Nat.Coprime.symm ((Nat.Prime.coprime_iff_not_dvd hq).mpr (not_dvd_six hq hq5))

/-- **The two-sided class, general moduli.** For coprime mL, mR > 1, both
coprime to 6, the slots with mL dividing the left member and mR the right are
exactly one CRT class mod mL * mR, with the floor count. Every both-sided
term of the master formula (SAME excluded: mR = 1) is an instance. -/
theorem twoSided_class {mL mR : ℕ} (h6L : Nat.Coprime 6 mL)
    (h6R : Nat.Coprime 6 mR) (hLR : Nat.Coprime mL mR)
    (hL1 : 1 < mL) (hR1 : 1 < mR) :
    ∃ a, 1 ≤ a ∧ a < mL * mR ∧
      (∀ k, 1 ≤ k → ((mL ∣ 6 * k - 1 ∧ mR ∣ 6 * k + 1) ↔ k ≡ a [MOD mL * mR])) ∧
      (∀ t, ((Finset.Ico 1 (t + 1)).filter
          (fun k => k % (mL * mR) = a % (mL * mR))).card
        = (t + mL * mR - a) / (mL * mR)) := by
  have hM1 : 1 < mL * mR := by
    have h := Nat.mul_le_mul (show 2 ≤ mL by omega) (show 2 ≤ mR by omega)
    omega
  have hco6 : Nat.Coprime 6 (mL * mR) := Nat.Coprime.mul_right h6L h6R
  obtain ⟨c, hc1, hc2⟩ := Nat.chineseRemainder hLR 1 (mR - 1)
  obtain ⟨a, ham, ha6, hiff6⟩ := six_mul_class c hco6 hM1
  have ha1 : 1 ≤ a := by
    rcases Nat.eq_zero_or_pos a with h0 | h1
    · exfalso
      subst h0
      have h6 : (0 : ℕ) ≡ c [MOD mL * mR] := by simpa using ha6
      have hq0 : (0 : ℕ) ≡ 1 [MOD mL] :=
        (Nat.ModEq.of_dvd (dvd_mul_right mL mR) h6).trans hc1
      have hd : mL ∣ 1 := by
        have h7 := (Nat.modEq_iff_dvd' (by omega)).mp hq0
        simpa using h7
      have := Nat.le_of_dvd (by norm_num) hd
      omega
    · exact h1
  refine ⟨a, ha1, ham, ?_, ?_⟩
  · intro k hk
    rw [left_dvd_iff hk, right_dvd_iff (by omega : 1 ≤ mR)]
    constructor
    · rintro ⟨hL, hR⟩
      have hqr : 6 * k ≡ c [MOD mL * mR] :=
        (Nat.modEq_and_modEq_iff_modEq_mul hLR).mp
          ⟨hL.trans hc1.symm, hR.trans hc2.symm⟩
      exact (hiff6 k).mp hqr
    · intro hka
      have hqr : 6 * k ≡ c [MOD mL * mR] := (hiff6 k).mpr hka
      exact ⟨(Nat.ModEq.of_dvd (dvd_mul_right mL mR) hqr).trans hc1,
        (Nat.ModEq.of_dvd (dvd_mul_left mR mL) hqr).trans hc2⟩
  · intro t
    exact card_class_Ico (by omega) ha1 (by omega) t

/-- **The CORR triple class.** For distinct primes q, r, s ≥ 5 the both-sided
triple (qr on the left, s on the right) is one CRT class mod qrs with the
floor count - the first genuinely new term of the signed correction. The
other role splits (q left, rs right; etc.) are further instantiations of
`twoSided_class`. -/
theorem corr_triple_class {q r s : ℕ} (hq : q.Prime) (hr : r.Prime)
    (hs : s.Prime) (hq5 : 5 ≤ q) (hr5 : 5 ≤ r) (hs5 : 5 ≤ s)
    (hqr : q ≠ r) (hqs : q ≠ s) (hrs : r ≠ s) :
    ∃ a, 1 ≤ a ∧ a < q * r * s ∧
      (∀ k, 1 ≤ k → ((q * r ∣ 6 * k - 1 ∧ s ∣ 6 * k + 1) ↔ k ≡ a [MOD q * r * s])) ∧
      (∀ t, ((Finset.Ico 1 (t + 1)).filter
          (fun k => k % (q * r * s) = a % (q * r * s))).card
        = (t + q * r * s - a) / (q * r * s)) := by
  have h6L : Nat.Coprime 6 (q * r) :=
    Nat.Coprime.mul_right (six_coprime_prime hq hq5) (six_coprime_prime hr hr5)
  have h6R : Nat.Coprime 6 s := six_coprime_prime hs hs5
  have hLR : Nat.Coprime (q * r) s :=
    (Nat.Coprime.mul_right ((Nat.coprime_primes hs hq).mpr (Ne.symm hqs))
      ((Nat.coprime_primes hs hr).mpr (Ne.symm hrs))).symm
  have hL1 : 1 < q * r := by
    have h := Nat.mul_le_mul hq5 hr5
    omega
  exact twoSided_class h6L h6R hLR hL1 (by omega)

/-- **The signed triple, subtraction-free.** Over the first t slots, the
DISTINCT slots hit by either of the two split classes sharing right gear s,
PLUS the triple class, equal the two split incidence counts. This is the
inclusion-exclusion step of the master formula for one triple: the triple
class (the overlap, computed by `corr_triple_class`) is exactly what the
signed sum removes when incidences are converted to distinct slots. -/
theorem corr_triple_signed {q r s : ℕ} (hco : Nat.Coprime q r) (t : ℕ) :
    ((Finset.Ico 1 (t + 1)).filter
        (fun k => (q ∣ 6 * k - 1 ∧ s ∣ 6 * k + 1) ∨
          (r ∣ 6 * k - 1 ∧ s ∣ 6 * k + 1))).card
      + ((Finset.Ico 1 (t + 1)).filter
        (fun k => q * r ∣ 6 * k - 1 ∧ s ∣ 6 * k + 1)).card
    = ((Finset.Ico 1 (t + 1)).filter
        (fun k => q ∣ 6 * k - 1 ∧ s ∣ 6 * k + 1)).card
      + ((Finset.Ico 1 (t + 1)).filter
        (fun k => r ∣ 6 * k - 1 ∧ s ∣ 6 * k + 1)).card := by
  have hinter :
      ((Finset.Ico 1 (t + 1)).filter (fun k => q ∣ 6 * k - 1 ∧ s ∣ 6 * k + 1))
        ∩ ((Finset.Ico 1 (t + 1)).filter (fun k => r ∣ 6 * k - 1 ∧ s ∣ 6 * k + 1))
      = (Finset.Ico 1 (t + 1)).filter
          (fun k => q * r ∣ 6 * k - 1 ∧ s ∣ 6 * k + 1) := by
    ext k
    simp only [Finset.mem_inter, Finset.mem_filter]
    constructor
    · rintro ⟨⟨hk1, hq1, hs1⟩, ⟨-, hr1, -⟩⟩
      exact ⟨hk1, hco.mul_dvd_of_dvd_of_dvd hq1 hr1, hs1⟩
    · rintro ⟨hk1, hqr1, hs1⟩
      exact ⟨⟨hk1, dvd_trans (dvd_mul_right q r) hqr1, hs1⟩,
        ⟨hk1, dvd_trans (dvd_mul_left r q) hqr1, hs1⟩⟩
  rw [Finset.filter_or, ← hinter]
  exact Finset.card_union_add_card_inter _ _

/-! ## The assembly: inclusion-exclusion over the incidence classes

From the per-term core to the assembled formula. `three_sets_ie` is n = 3
inclusion-exclusion, subtraction-free, for arbitrary finsets; instantiated at
the mark sets M_q = {k : q | 6k-1 or q | 6k+1} it is `three_gear_assembly` -
"assembled sum = sieve overcount", since overcount = marks - distinct is a
rearrangement of the identity. The bridges `card_marks_eq` and
`card_pair_inter_eq` decompose the per-gear and pairwise terms disjointly into
side classes - each ONE CRT class with a floor count via `six_mul_class` /
`twoSided_class`. The triple decomposition (8 side classes) is the same
mechanics; verified numerically in research/assembly_check.py, left paper-side. -/

/-- Filter of a disjunction splits the count when the disjuncts exclude each
other on the set. -/
theorem card_filter_or_of_excl {α : Type*} [DecidableEq α] {s : Finset α}
    {p q : α → Prop} [DecidablePred p] [DecidablePred q]
    (h : ∀ a ∈ s, ¬(p a ∧ q a)) :
    (s.filter (fun a => p a ∨ q a)).card
      = (s.filter p).card + (s.filter q).card := by
  rw [Finset.filter_or]
  apply Finset.card_union_of_disjoint
  rw [Finset.disjoint_left]
  intro a haP haQ
  have h1 := Finset.mem_filter.mp haP
  have h2 := Finset.mem_filter.mp haQ
  exact h a h1.1 ⟨h1.2, h2.2⟩

/-- **n = 3 inclusion-exclusion, subtraction-free.** For any three finsets:
distinct elements of the union plus the three pairwise intersections equal
the three cardinalities plus the triple intersection. -/
theorem three_sets_ie {α : Type*} [DecidableEq α] (A B C : Finset α) :
    (A ∪ B ∪ C).card + (A ∩ B).card + (A ∩ C).card + (B ∩ C).card
      = A.card + B.card + C.card + (A ∩ B ∩ C).card := by
  have h1 := Finset.card_union_add_card_inter A B
  have h2 := Finset.card_union_add_card_inter (A ∪ B) C
  have h3 := Finset.card_union_add_card_inter (A ∩ C) (B ∩ C)
  rw [Finset.union_inter_distrib_right] at h2
  have he : (A ∩ C) ∩ (B ∩ C) = A ∩ B ∩ C := by
    ext x
    simp only [Finset.mem_inter]
    constructor
    · rintro ⟨⟨hA, hC⟩, ⟨hB, -⟩⟩
      exact ⟨⟨hA, hB⟩, hC⟩
    · rintro ⟨⟨hA, hB⟩, hC⟩
      exact ⟨⟨hA, hC⟩, ⟨hB, hC⟩⟩
  rw [he] at h3
  omega

/-- The predicate form of `three_sets_ie` over one filtered set. -/
theorem three_preds_ie {α : Type*} [DecidableEq α] (S : Finset α)
    (P Q R : α → Prop) [DecidablePred P] [DecidablePred Q] [DecidablePred R] :
    (S.filter fun a => P a ∨ Q a ∨ R a).card
      + (S.filter fun a => P a ∧ Q a).card
      + (S.filter fun a => P a ∧ R a).card
      + (S.filter fun a => Q a ∧ R a).card
    = (S.filter P).card + (S.filter Q).card + (S.filter R).card
      + (S.filter fun a => P a ∧ Q a ∧ R a).card := by
  have h1 : (S.filter fun a => P a ∨ Q a ∨ R a)
      = S.filter P ∪ S.filter Q ∪ S.filter R := by
    rw [Finset.filter_or, Finset.filter_or]
    exact (Finset.union_assoc _ _ _).symm
  have h2 : (S.filter fun a => P a ∧ Q a) = S.filter P ∩ S.filter Q := by
    ext a
    simp only [Finset.mem_filter, Finset.mem_inter]
    constructor
    · rintro ⟨ha, hP, hQ⟩
      exact ⟨⟨ha, hP⟩, ⟨ha, hQ⟩⟩
    · rintro ⟨⟨ha, hP⟩, ⟨-, hQ⟩⟩
      exact ⟨ha, hP, hQ⟩
  have h3 : (S.filter fun a => P a ∧ R a) = S.filter P ∩ S.filter R := by
    ext a
    simp only [Finset.mem_filter, Finset.mem_inter]
    constructor
    · rintro ⟨ha, hP, hR⟩
      exact ⟨⟨ha, hP⟩, ⟨ha, hR⟩⟩
    · rintro ⟨⟨ha, hP⟩, ⟨-, hR⟩⟩
      exact ⟨ha, hP, hR⟩
  have h4 : (S.filter fun a => Q a ∧ R a) = S.filter Q ∩ S.filter R := by
    ext a
    simp only [Finset.mem_filter, Finset.mem_inter]
    constructor
    · rintro ⟨ha, hQ, hR⟩
      exact ⟨⟨ha, hQ⟩, ⟨ha, hR⟩⟩
    · rintro ⟨⟨ha, hQ⟩, ⟨-, hR⟩⟩
      exact ⟨ha, hQ, hR⟩
  have h5 : (S.filter fun a => P a ∧ Q a ∧ R a)
      = S.filter P ∩ S.filter Q ∩ S.filter R := by
    ext a
    simp only [Finset.mem_filter, Finset.mem_inter]
    constructor
    · rintro ⟨ha, hP, hQ, hR⟩
      exact ⟨⟨⟨ha, hP⟩, ⟨ha, hQ⟩⟩, ⟨ha, hR⟩⟩
    · rintro ⟨⟨⟨ha, hP⟩, ⟨-, hQ⟩⟩, ⟨-, hR⟩⟩
      exact ⟨ha, hP, hQ, hR⟩
  rw [h1, h2, h3, h4, h5]
  exact three_sets_ie _ _ _

/-- **The 3-gear assembly** ("assembled sum = sieve overcount",
subtraction-free): distinct marked slots plus the three pairwise-intersection
counts equal the per-gear mark counts plus the triple. Purely set-level - no
primality hypotheses; overcount = marks - distinct rearranges this. -/
theorem three_gear_assembly (q r s t : ℕ) :
    ((Finset.Ico 1 (t + 1)).filter fun k =>
        (q ∣ 6 * k - 1 ∨ q ∣ 6 * k + 1) ∨ (r ∣ 6 * k - 1 ∨ r ∣ 6 * k + 1) ∨
        (s ∣ 6 * k - 1 ∨ s ∣ 6 * k + 1)).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        (q ∣ 6 * k - 1 ∨ q ∣ 6 * k + 1) ∧ (r ∣ 6 * k - 1 ∨ r ∣ 6 * k + 1)).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        (q ∣ 6 * k - 1 ∨ q ∣ 6 * k + 1) ∧ (s ∣ 6 * k - 1 ∨ s ∣ 6 * k + 1)).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        (r ∣ 6 * k - 1 ∨ r ∣ 6 * k + 1) ∧ (s ∣ 6 * k - 1 ∨ s ∣ 6 * k + 1)).card
    = ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k - 1 ∨ q ∣ 6 * k + 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        r ∣ 6 * k - 1 ∨ r ∣ 6 * k + 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        s ∣ 6 * k - 1 ∨ s ∣ 6 * k + 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        (q ∣ 6 * k - 1 ∨ q ∣ 6 * k + 1) ∧ (r ∣ 6 * k - 1 ∨ r ∣ 6 * k + 1) ∧
        (s ∣ 6 * k - 1 ∨ s ∣ 6 * k + 1)).card :=
  three_preds_ie (Finset.Ico 1 (t + 1)) _ _ _

/-- A gear marks a slot on at most one side (slot cap in mark form). -/
theorem mark_side_unique {q k : ℕ} (hq : q.Prime) (hq2 : q ≠ 2) (hk : 1 ≤ k)
    (hL : q ∣ 6 * k - 1) (hR : q ∣ 6 * k + 1) : False := by
  refine slot_cap_twin hq hq2 hL ?_
  rwa [show 6 * k - 1 + 2 = 6 * k + 1 by omega]

/-- **Per-gear bridge.** A gear's mark count splits disjointly by side; each
side is one CRT class (`six_mul_class` + `left_dvd_iff`/`right_dvd_iff`). -/
theorem card_marks_eq {q t : ℕ} (hq : q.Prime) (hq2 : q ≠ 2) :
    ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k - 1 ∨ q ∣ 6 * k + 1).card
      = ((Finset.Ico 1 (t + 1)).filter fun k => q ∣ 6 * k - 1).card
        + ((Finset.Ico 1 (t + 1)).filter fun k => q ∣ 6 * k + 1).card := by
  apply card_filter_or_of_excl
  intro k hk
  rintro ⟨hL, hR⟩
  exact mark_side_unique hq hq2 (Finset.mem_Ico.mp hk).1 hL hR

/-- **Pair bridge.** A pairwise mark intersection decomposes disjointly into
the four side classes - each ONE CRT class with a floor count (SAME-left,
split, mirror split, SAME-right). The set-level assembly meets the
class-and-count layer here. -/
theorem card_pair_inter_eq {q r t : ℕ} (hq : q.Prime) (hr : r.Prime)
    (hq2 : q ≠ 2) (hr2 : r ≠ 2) :
    ((Finset.Ico 1 (t + 1)).filter fun k =>
        (q ∣ 6 * k - 1 ∨ q ∣ 6 * k + 1) ∧ (r ∣ 6 * k - 1 ∨ r ∣ 6 * k + 1)).card
    = ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k - 1 ∧ r ∣ 6 * k - 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k - 1 ∧ r ∣ 6 * k + 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k + 1 ∧ r ∣ 6 * k - 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1).card := by
  have hpred : ((Finset.Ico 1 (t + 1)).filter fun k =>
      (q ∣ 6 * k - 1 ∨ q ∣ 6 * k + 1) ∧ (r ∣ 6 * k - 1 ∨ r ∣ 6 * k + 1))
    = (Finset.Ico 1 (t + 1)).filter fun k =>
      (q ∣ 6 * k - 1 ∧ r ∣ 6 * k - 1) ∨ ((q ∣ 6 * k - 1 ∧ r ∣ 6 * k + 1) ∨
      ((q ∣ 6 * k + 1 ∧ r ∣ 6 * k - 1) ∨ (q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1))) := by
    ext k
    simp only [Finset.mem_filter]
    constructor
    · rintro ⟨hk, ⟨hL | hR, hL' | hR'⟩⟩
      · exact ⟨hk, Or.inl ⟨hL, hL'⟩⟩
      · exact ⟨hk, Or.inr (Or.inl ⟨hL, hR'⟩)⟩
      · exact ⟨hk, Or.inr (Or.inr (Or.inl ⟨hR, hL'⟩))⟩
      · exact ⟨hk, Or.inr (Or.inr (Or.inr ⟨hR, hR'⟩))⟩
    · rintro ⟨hk, h⟩
      rcases h with ⟨h1, h2⟩ | ⟨h1, h2⟩ | ⟨h1, h2⟩ | ⟨h1, h2⟩
      · exact ⟨hk, Or.inl h1, Or.inl h2⟩
      · exact ⟨hk, Or.inl h1, Or.inr h2⟩
      · exact ⟨hk, Or.inr h1, Or.inl h2⟩
      · exact ⟨hk, Or.inr h1, Or.inr h2⟩
  rw [hpred]
  have excl1 : ∀ k ∈ Finset.Ico 1 (t + 1),
      ¬((q ∣ 6 * k - 1 ∧ r ∣ 6 * k - 1) ∧
        ((q ∣ 6 * k - 1 ∧ r ∣ 6 * k + 1) ∨ ((q ∣ 6 * k + 1 ∧ r ∣ 6 * k - 1) ∨
         (q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1)))) := by
    intro k hk
    have hk1 : 1 ≤ k := (Finset.mem_Ico.mp hk).1
    rintro ⟨⟨hqL, hrL⟩, h⟩
    rcases h with ⟨-, hrR⟩ | ⟨hqR, -⟩ | ⟨hqR, -⟩
    · exact mark_side_unique hr hr2 hk1 hrL hrR
    · exact mark_side_unique hq hq2 hk1 hqL hqR
    · exact mark_side_unique hq hq2 hk1 hqL hqR
  have excl2 : ∀ k ∈ Finset.Ico 1 (t + 1),
      ¬((q ∣ 6 * k - 1 ∧ r ∣ 6 * k + 1) ∧
        ((q ∣ 6 * k + 1 ∧ r ∣ 6 * k - 1) ∨ (q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1))) := by
    intro k hk
    have hk1 : 1 ≤ k := (Finset.mem_Ico.mp hk).1
    rintro ⟨⟨hqL, -⟩, h⟩
    rcases h with ⟨hqR, -⟩ | ⟨hqR, -⟩ <;>
      exact mark_side_unique hq hq2 hk1 hqL hqR
  have excl3 : ∀ k ∈ Finset.Ico 1 (t + 1),
      ¬((q ∣ 6 * k + 1 ∧ r ∣ 6 * k - 1) ∧ (q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1)) := by
    intro k hk
    have hk1 : 1 ≤ k := (Finset.mem_Ico.mp hk).1
    rintro ⟨⟨-, hrL⟩, ⟨-, hrR⟩⟩
    exact mark_side_unique hr hr2 hk1 hrL hrR
  rw [card_filter_or_of_excl excl1, card_filter_or_of_excl excl2,
    card_filter_or_of_excl excl3]
  omega

/-- **Triple bridge.** The triple mark intersection decomposes disjointly into
the eight side classes LLL..RRR - each ONE CRT class with a floor count
(`six_mul_class` / `twoSided_class` instances). Identical mechanics to
`card_pair_inter_eq`, 2^3 cases. -/
theorem card_triple_inter_eq {q r s t : ℕ} (hq : q.Prime) (hr : r.Prime)
    (hs : s.Prime) (hq2 : q ≠ 2) (hr2 : r ≠ 2) (hs2 : s ≠ 2) :
    ((Finset.Ico 1 (t + 1)).filter fun k =>
        (q ∣ 6 * k - 1 ∨ q ∣ 6 * k + 1) ∧ (r ∣ 6 * k - 1 ∨ r ∣ 6 * k + 1) ∧
        (s ∣ 6 * k - 1 ∨ s ∣ 6 * k + 1)).card
    = ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k - 1 ∧ r ∣ 6 * k - 1 ∧ s ∣ 6 * k - 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k - 1 ∧ r ∣ 6 * k - 1 ∧ s ∣ 6 * k + 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k - 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k - 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k - 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k + 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k + 1 ∧ r ∣ 6 * k - 1 ∧ s ∣ 6 * k - 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k + 1 ∧ r ∣ 6 * k - 1 ∧ s ∣ 6 * k + 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k - 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k + 1).card := by
  have hpred : ((Finset.Ico 1 (t + 1)).filter fun k =>
      (q ∣ 6 * k - 1 ∨ q ∣ 6 * k + 1) ∧ (r ∣ 6 * k - 1 ∨ r ∣ 6 * k + 1) ∧
      (s ∣ 6 * k - 1 ∨ s ∣ 6 * k + 1))
    = (Finset.Ico 1 (t + 1)).filter fun k =>
      (q ∣ 6 * k - 1 ∧ r ∣ 6 * k - 1 ∧ s ∣ 6 * k - 1) ∨
      ((q ∣ 6 * k - 1 ∧ r ∣ 6 * k - 1 ∧ s ∣ 6 * k + 1) ∨
      ((q ∣ 6 * k - 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k - 1) ∨
      ((q ∣ 6 * k - 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k + 1) ∨
      ((q ∣ 6 * k + 1 ∧ r ∣ 6 * k - 1 ∧ s ∣ 6 * k - 1) ∨
      ((q ∣ 6 * k + 1 ∧ r ∣ 6 * k - 1 ∧ s ∣ 6 * k + 1) ∨
      ((q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k - 1) ∨
       (q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k + 1))))))) := by
    ext k
    simp only [Finset.mem_filter]
    constructor
    · rintro ⟨hk, hq' | hq', hr' | hr', hs' | hs'⟩
      · exact ⟨hk, Or.inl ⟨hq', hr', hs'⟩⟩
      · exact ⟨hk, Or.inr (Or.inl ⟨hq', hr', hs'⟩)⟩
      · exact ⟨hk, Or.inr (Or.inr (Or.inl ⟨hq', hr', hs'⟩))⟩
      · exact ⟨hk, Or.inr (Or.inr (Or.inr (Or.inl ⟨hq', hr', hs'⟩)))⟩
      · exact ⟨hk, Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨hq', hr', hs'⟩))))⟩
      · exact ⟨hk, Or.inr (Or.inr (Or.inr (Or.inr (Or.inr
          (Or.inl ⟨hq', hr', hs'⟩)))))⟩
      · exact ⟨hk, Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr
          (Or.inl ⟨hq', hr', hs'⟩))))))⟩
      · exact ⟨hk, Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr
          (Or.inr ⟨hq', hr', hs'⟩))))))⟩
    · rintro ⟨hk, ⟨ha, hb, hc⟩ | ⟨ha, hb, hc⟩ | ⟨ha, hb, hc⟩ | ⟨ha, hb, hc⟩ |
        ⟨ha, hb, hc⟩ | ⟨ha, hb, hc⟩ | ⟨ha, hb, hc⟩ | ⟨ha, hb, hc⟩⟩
      · exact ⟨hk, Or.inl ha, Or.inl hb, Or.inl hc⟩
      · exact ⟨hk, Or.inl ha, Or.inl hb, Or.inr hc⟩
      · exact ⟨hk, Or.inl ha, Or.inr hb, Or.inl hc⟩
      · exact ⟨hk, Or.inl ha, Or.inr hb, Or.inr hc⟩
      · exact ⟨hk, Or.inr ha, Or.inl hb, Or.inl hc⟩
      · exact ⟨hk, Or.inr ha, Or.inl hb, Or.inr hc⟩
      · exact ⟨hk, Or.inr ha, Or.inr hb, Or.inl hc⟩
      · exact ⟨hk, Or.inr ha, Or.inr hb, Or.inr hc⟩
  have excl1 : ∀ k ∈ Finset.Ico 1 (t + 1),
      ¬((q ∣ 6 * k - 1 ∧ r ∣ 6 * k - 1 ∧ s ∣ 6 * k - 1) ∧
        ((q ∣ 6 * k - 1 ∧ r ∣ 6 * k - 1 ∧ s ∣ 6 * k + 1) ∨
        ((q ∣ 6 * k - 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k - 1) ∨
        ((q ∣ 6 * k - 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k + 1) ∨
        ((q ∣ 6 * k + 1 ∧ r ∣ 6 * k - 1 ∧ s ∣ 6 * k - 1) ∨
        ((q ∣ 6 * k + 1 ∧ r ∣ 6 * k - 1 ∧ s ∣ 6 * k + 1) ∨
        ((q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k - 1) ∨
         (q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k + 1)))))))) := by
    intro k hk
    have hk1 : 1 ≤ k := (Finset.mem_Ico.mp hk).1
    rintro ⟨⟨hqL, hrL, hsL⟩, h⟩
    rcases h with ⟨-, -, hsR⟩ | ⟨-, hrR, -⟩ | ⟨-, hrR, -⟩ | ⟨hqR, -, -⟩ |
      ⟨hqR, -, -⟩ | ⟨hqR, -, -⟩ | ⟨hqR, -, -⟩
    · exact mark_side_unique hs hs2 hk1 hsL hsR
    · exact mark_side_unique hr hr2 hk1 hrL hrR
    · exact mark_side_unique hr hr2 hk1 hrL hrR
    · exact mark_side_unique hq hq2 hk1 hqL hqR
    · exact mark_side_unique hq hq2 hk1 hqL hqR
    · exact mark_side_unique hq hq2 hk1 hqL hqR
    · exact mark_side_unique hq hq2 hk1 hqL hqR
  have excl2 : ∀ k ∈ Finset.Ico 1 (t + 1),
      ¬((q ∣ 6 * k - 1 ∧ r ∣ 6 * k - 1 ∧ s ∣ 6 * k + 1) ∧
        ((q ∣ 6 * k - 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k - 1) ∨
        ((q ∣ 6 * k - 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k + 1) ∨
        ((q ∣ 6 * k + 1 ∧ r ∣ 6 * k - 1 ∧ s ∣ 6 * k - 1) ∨
        ((q ∣ 6 * k + 1 ∧ r ∣ 6 * k - 1 ∧ s ∣ 6 * k + 1) ∨
        ((q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k - 1) ∨
         (q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k + 1))))))) := by
    intro k hk
    have hk1 : 1 ≤ k := (Finset.mem_Ico.mp hk).1
    rintro ⟨⟨hqL, hrL, -⟩, h⟩
    rcases h with ⟨-, hrR, -⟩ | ⟨-, hrR, -⟩ | ⟨hqR, -, -⟩ | ⟨hqR, -, -⟩ |
      ⟨hqR, -, -⟩ | ⟨hqR, -, -⟩
    · exact mark_side_unique hr hr2 hk1 hrL hrR
    · exact mark_side_unique hr hr2 hk1 hrL hrR
    · exact mark_side_unique hq hq2 hk1 hqL hqR
    · exact mark_side_unique hq hq2 hk1 hqL hqR
    · exact mark_side_unique hq hq2 hk1 hqL hqR
    · exact mark_side_unique hq hq2 hk1 hqL hqR
  have excl3 : ∀ k ∈ Finset.Ico 1 (t + 1),
      ¬((q ∣ 6 * k - 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k - 1) ∧
        ((q ∣ 6 * k - 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k + 1) ∨
        ((q ∣ 6 * k + 1 ∧ r ∣ 6 * k - 1 ∧ s ∣ 6 * k - 1) ∨
        ((q ∣ 6 * k + 1 ∧ r ∣ 6 * k - 1 ∧ s ∣ 6 * k + 1) ∨
        ((q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k - 1) ∨
         (q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k + 1)))))) := by
    intro k hk
    have hk1 : 1 ≤ k := (Finset.mem_Ico.mp hk).1
    rintro ⟨⟨hqL, -, hsL⟩, h⟩
    rcases h with ⟨-, -, hsR⟩ | ⟨hqR, -, -⟩ | ⟨hqR, -, -⟩ | ⟨hqR, -, -⟩ |
      ⟨hqR, -, -⟩
    · exact mark_side_unique hs hs2 hk1 hsL hsR
    · exact mark_side_unique hq hq2 hk1 hqL hqR
    · exact mark_side_unique hq hq2 hk1 hqL hqR
    · exact mark_side_unique hq hq2 hk1 hqL hqR
    · exact mark_side_unique hq hq2 hk1 hqL hqR
  have excl4 : ∀ k ∈ Finset.Ico 1 (t + 1),
      ¬((q ∣ 6 * k - 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k + 1) ∧
        ((q ∣ 6 * k + 1 ∧ r ∣ 6 * k - 1 ∧ s ∣ 6 * k - 1) ∨
        ((q ∣ 6 * k + 1 ∧ r ∣ 6 * k - 1 ∧ s ∣ 6 * k + 1) ∨
        ((q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k - 1) ∨
         (q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k + 1))))) := by
    intro k hk
    have hk1 : 1 ≤ k := (Finset.mem_Ico.mp hk).1
    rintro ⟨⟨hqL, -, -⟩, h⟩
    rcases h with ⟨hqR, -, -⟩ | ⟨hqR, -, -⟩ | ⟨hqR, -, -⟩ | ⟨hqR, -, -⟩ <;>
      exact mark_side_unique hq hq2 hk1 hqL hqR
  have excl5 : ∀ k ∈ Finset.Ico 1 (t + 1),
      ¬((q ∣ 6 * k + 1 ∧ r ∣ 6 * k - 1 ∧ s ∣ 6 * k - 1) ∧
        ((q ∣ 6 * k + 1 ∧ r ∣ 6 * k - 1 ∧ s ∣ 6 * k + 1) ∨
        ((q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k - 1) ∨
         (q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k + 1)))) := by
    intro k hk
    have hk1 : 1 ≤ k := (Finset.mem_Ico.mp hk).1
    rintro ⟨⟨-, hrL, hsL⟩, h⟩
    rcases h with ⟨-, -, hsR⟩ | ⟨-, hrR, -⟩ | ⟨-, hrR, -⟩
    · exact mark_side_unique hs hs2 hk1 hsL hsR
    · exact mark_side_unique hr hr2 hk1 hrL hrR
    · exact mark_side_unique hr hr2 hk1 hrL hrR
  have excl6 : ∀ k ∈ Finset.Ico 1 (t + 1),
      ¬((q ∣ 6 * k + 1 ∧ r ∣ 6 * k - 1 ∧ s ∣ 6 * k + 1) ∧
        ((q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k - 1) ∨
         (q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k + 1))) := by
    intro k hk
    have hk1 : 1 ≤ k := (Finset.mem_Ico.mp hk).1
    rintro ⟨⟨-, hrL, -⟩, h⟩
    rcases h with ⟨-, hrR, -⟩ | ⟨-, hrR, -⟩ <;>
      exact mark_side_unique hr hr2 hk1 hrL hrR
  have excl7 : ∀ k ∈ Finset.Ico 1 (t + 1),
      ¬((q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k - 1) ∧
        (q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k + 1)) := by
    intro k hk
    have hk1 : 1 ≤ k := (Finset.mem_Ico.mp hk).1
    rintro ⟨⟨-, -, hsL⟩, ⟨-, -, hsR⟩⟩
    exact mark_side_unique hs hs2 hk1 hsL hsR
  rw [hpred, card_filter_or_of_excl excl1, card_filter_or_of_excl excl2,
    card_filter_or_of_excl excl3, card_filter_or_of_excl excl4,
    card_filter_or_of_excl excl5, card_filter_or_of_excl excl6,
    card_filter_or_of_excl excl7]
  omega

/-- **The master formula for three gears, end-to-end.** Every mark-set term of
the assembly identity decomposed into its side classes: distinct marked slots
plus the twelve pair side classes equal the six single side classes plus the
eight triple side classes. Subtraction-free; overcount = marks - distinct
rearranges it to overcount = pairs - triples. EVERY term on both sides beyond
the first is one CRT class whose count is closed-form floor arithmetic
(`six_mul_class` / `twoSided_class` + `card_class_Ico`) - the formal statement
of what research/assembly_check.py verified with zero fails. -/
theorem three_gear_master {q r s t : ℕ} (hq : q.Prime) (hr : r.Prime)
    (hs : s.Prime) (hq2 : q ≠ 2) (hr2 : r ≠ 2) (hs2 : s ≠ 2) :
    ((Finset.Ico 1 (t + 1)).filter fun k =>
        (q ∣ 6 * k - 1 ∨ q ∣ 6 * k + 1) ∨ (r ∣ 6 * k - 1 ∨ r ∣ 6 * k + 1) ∨
        (s ∣ 6 * k - 1 ∨ s ∣ 6 * k + 1)).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k - 1 ∧ r ∣ 6 * k - 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k - 1 ∧ r ∣ 6 * k + 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k + 1 ∧ r ∣ 6 * k - 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k - 1 ∧ s ∣ 6 * k - 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k - 1 ∧ s ∣ 6 * k + 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k + 1 ∧ s ∣ 6 * k - 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k + 1 ∧ s ∣ 6 * k + 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        r ∣ 6 * k - 1 ∧ s ∣ 6 * k - 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        r ∣ 6 * k - 1 ∧ s ∣ 6 * k + 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        r ∣ 6 * k + 1 ∧ s ∣ 6 * k - 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        r ∣ 6 * k + 1 ∧ s ∣ 6 * k + 1).card
    = ((Finset.Ico 1 (t + 1)).filter fun k => q ∣ 6 * k - 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k => q ∣ 6 * k + 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k => r ∣ 6 * k - 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k => r ∣ 6 * k + 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k => s ∣ 6 * k - 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k => s ∣ 6 * k + 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k - 1 ∧ r ∣ 6 * k - 1 ∧ s ∣ 6 * k - 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k - 1 ∧ r ∣ 6 * k - 1 ∧ s ∣ 6 * k + 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k - 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k - 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k - 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k + 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k + 1 ∧ r ∣ 6 * k - 1 ∧ s ∣ 6 * k - 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k + 1 ∧ r ∣ 6 * k - 1 ∧ s ∣ 6 * k + 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k - 1).card
      + ((Finset.Ico 1 (t + 1)).filter fun k =>
        q ∣ 6 * k + 1 ∧ r ∣ 6 * k + 1 ∧ s ∣ 6 * k + 1).card := by
  have h0 := three_gear_assembly q r s t
  rw [card_marks_eq hq hq2, card_marks_eq hr hr2, card_marks_eq hs hs2,
    card_pair_inter_eq hq hr hq2 hr2, card_pair_inter_eq hq hs hq2 hs2,
    card_pair_inter_eq hr hs hr2 hs2,
    card_triple_inter_eq hq hr hs hq2 hr2 hs2] at h0
  omega

/-! ## The self-block, composed with the census -/

/-- **The self-block, formal.** The twin pair's pin slot u is an actual twin
slot of the census (both members prime) - yet it is NEVER a survivor of any
machine whose divisor bound reaches p: the machine is blind to its own pair.
This is why the u'-pin list U of the master supply formula is invisible to
n2: its slots are prime-membered, blocked only by their own gears. -/
theorem twin_pin_self_block {p y u : ℕ} (hp : p.Prime) (hp2 : (p + 2).Prime)
    (h3 : 3 < p) (hpy : p ≤ y) (hu : 6 * u = p + 1) :
    Census.slotComps u = 0 ∧ ¬ BlockedSlots.Survivor y (Census.lo u) := by
  have hlo : Census.lo u = p := by
    unfold Census.lo
    omega
  have hhi : Census.hi u = p + 2 := by
    unfold Census.hi
    omega
  constructor
  · rw [Census.slotComps_eq_zero_iff u, hlo, hhi]
    exact ⟨hp, hp2⟩
  · intro hs
    have h1 := (hs p hp hpy).1
    rw [hlo] at h1
    exact h1 (dvd_refl p)

/-! ## The covering-search endpoint law: F(2,y) ≡ 0 mod 3

The number-theoretic core of the mod-3 pruning used by
`rust2/src/bin/maxgap_pruned.rs`. In the adjacent (covering-search) frame each
odd prime `q` blocks one pair of ADJACENT residues `{o, o+1}` mod `q`. At
`q = 3` that pair leaves exactly ONE free class, so gear 3 cannot leave two
incongruent positions uncovered: both flanks of a maximal covered run are
congruent mod 3, and the run's gap length is a multiple of 3.

In the search's terms `F(2,y) = M + 1` where `M` is the largest coverable run,
so `endpoint_run_mod_three` says `F(2,y) ≡ 0 (mod 3)` - matching all thirteen
known exact values (33, 48, 54, 75, 102, 117, 129, 174, 264, 273, 309, ...).
Verified exhaustively over ALL offset tuples in research/lefttaut_check.py
(y = 11, 13, 17) and numerically in research/literal_cap_gap_d.py (T3). -/

/-- `AdjBlocked q o i`: gear `q`, sitting at offset `o`, blocks position `i`
in the adjacent frame - the covering search's blocking relation, one pair of
adjacent residues per gear. -/
def AdjBlocked (q o i : ℕ) : Prop := i % q = o % q ∨ i % q = (o + 1) % q

instance (q o i : ℕ) : Decidable (AdjBlocked q o i) := by
  unfold AdjBlocked; infer_instance

/-- **One free class at gear 3.** The adjacent pair `{o, o+1}` covers two of
the three classes mod 3, so an unblocked position sits in the single
remaining class `o + 2`. -/
theorem free_class_three {o x : ℕ} (h : ¬ AdjBlocked 3 o x) :
    x % 3 = (o + 2) % 3 := by
  unfold AdjBlocked at h
  push_neg at h
  omega

/-- **Uniqueness.** Gear 3 cannot leave two incongruent positions uncovered. -/
theorem free_class_unique_three {o x y : ℕ} (hx : ¬ AdjBlocked 3 o x)
    (hy : ¬ AdjBlocked 3 o y) : x % 3 = y % 3 := by
  rw [free_class_three hx, free_class_three hy]

/-- **The endpoint law.** If a run of `M` positions starting at `s` has both
flanks `s - 1` and `s + M` unblocked by gear 3, the gap length `M + 1` is a
multiple of 3. With `M` maximal this is `F(2,y) ≡ 0 (mod 3)`: the mod-3 skip
of the pruned covering search, and the reason every known exact value of
`F(2,y)` is divisible by 3. -/
theorem endpoint_run_mod_three {o s M : ℕ} (hs : 1 ≤ s)
    (hL : ¬ AdjBlocked 3 o (s - 1)) (hR : ¬ AdjBlocked 3 o (s + M)) :
    (M + 1) % 3 = 0 ∧ 3 ∣ (M + 1) := by
  have h := free_class_unique_three hL hR
  refine ⟨by omega, ?_⟩
  exact Nat.dvd_of_mod_eq_zero (by omega)

end Polignac
