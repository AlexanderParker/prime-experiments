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

end Polignac
