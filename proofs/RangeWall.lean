import Mathlib.NumberTheory.Primorial
import Mathlib.NumberTheory.Bertrand
import Mathlib.Tactic
import RangeLocator

/-!
# The wall: `nextprime(X)² < X#` for every natural `X ≥ 7`, and its consequences on the range

Ported into the kernel from the scratch files `RangeWall.lean` / `adj_c6/C6AdjWall.lean` and
`C6Defence.lean` / `adj_c6/C6AdjDef.lean`. Names that clash with existing kernel names are renamed
(`wall_primorial_five`, `wall_primorial_seven`, `wall_thirty_dvd_primorial`; the two scratch
`four_sq_lt_primorial` statements are `four_sq_lt_primorial_of_prime` and
`four_sq_lt_primorial_of_ge_eleven`).

**The wall.**
* `wall_inequality_nat`: for every natural `X ≥ 7` and every `p` at most every prime above `X`
  (in particular `p = nextprime X`): `p² < X#`.
* `wall_fails_below_seven`: for every `X < 7` and every `p > X`: `X# ≤ p²`.
* `wall_iff`: for `p` the least prime above `X` (any natural `X`): `p² < X# ↔ 7 ≤ X`.
* `nextprime_sq_lt_primorial`: the nextprime form, for every natural `X ≥ 7`.

**The doubled wall.**
* `four_sq_lt_primorial_iff`: for every natural `X`, `(2X)² < X# ↔ X = 0 ∨ X = 7 ∨ 11 ≤ X`.

**Acting from the period.**
* `acts_from_period`: for every `q ≥ 7`, every `g` at most every prime above `q` (every gear
  `g ≤ q'`) acts on every copy `j ≥ q#/30`: `g² ≤ 30 j + 1`.

**Window and range top.**
* `window_below_primorial`, `window_mem_range_iff`: a window-cut copy (`30 j + 1 < q'²`) sits below
  `q#`, so it is a copy of the range iff its lower leg is above `q`.
* `below_range_iff`, `below_range_exists_iff`: the window-cut copies below the range are exactly
  `1 ≤ j ≤ (q + 1)/30`, and there is one iff `29 ≤ q`.

**Lap bounds.**
* `locator_closure_plus`: the upper-leg locator.
* `lap_bound`, `lap_bound_sharp`: within `q'` (resp. `q' - 1`) laps of `M = q#/30` from any `r`, a
  copy struck by `q'` on which `q'` acts.

**Silent classes.**
* `silent_class_period`, `silent_class_beyond_wall`, `silent_class_one_per_period`,
  `silent_class_one_in_band`: a class `j ≡ r (mod N)` never struck by a gear of `[7, X]` has
  `X# ≤ 30 N`, hence `p² < 30 N`, and at most one member per period and per certified band.
-/

namespace RangeLine

/-- `5# = 2 · 3 · 5 = 30`. -/
theorem wall_primorial_five : primorial 5 = 30 := by decide

/-- `7# = 2 · 3 · 5 · 7 = 210`. -/
theorem wall_primorial_seven : primorial 7 = 210 := by decide

/-- `11# = 2310`. -/
theorem wall_primorial_eleven : primorial 11 = 2310 := by decide

/-- `23# = 223092870`. -/
theorem wall_primorial_twentythree : primorial 23 = 223092870 := by decide

/-- For every `X ≥ 5`, `30 ∣ X#`. -/
theorem wall_thirty_dvd_primorial {X : ℕ} (hX : 5 ≤ X) : 30 ∣ primorial X := by
  have h := primorial_dvd_primorial hX
  rwa [wall_primorial_five] at h

/-- Every prime `p ≥ 7` is coprime to `30`. -/
theorem coprime_thirty_of_prime {p : ℕ} (hp : p.Prime) (h7 : 7 ≤ p) : Nat.Coprime 30 p := by
  have h30 : (30 : ℕ) = 2 * 3 * 5 := by norm_num
  rw [h30]
  refine Nat.coprime_mul_iff_left.mpr ⟨Nat.coprime_mul_iff_left.mpr ⟨?_, ?_⟩, ?_⟩ <;>
    exact (Nat.coprime_primes (by norm_num) hp).mpr (by omega)

/-- For every prime `X ≥ 11`: `15 X (X + 1) ≤ X#` (Bertrand gives a prime `p` with
`X/2 < p < X`, and `2 · 3 · 5 · p · X ∣ X#`). -/
theorem fifteen_mul_le_primorial {X : ℕ} (hX : X.Prime) (h11 : 11 ≤ X) :
    15 * X * (X + 1) ≤ primorial X := by
  have hodd : X % 2 = 1 := by
    rcases hX.eq_two_or_odd with h | h
    · omega
    · exact h
  obtain ⟨p, hp, hlo, hhi⟩ := Nat.exists_prime_lt_and_le_two_mul (X / 2) (by omega)
  have hpX : p < X := by omega
  have hp7 : 7 ≤ p := by
    have h6 : 6 ≤ p := by omega
    rcases Nat.lt_or_ge p 7 with h | h
    · have : p = 6 := by omega
      subst this
      exact absurd hp (by norm_num)
    · exact h
  have h2p : X + 1 ≤ 2 * p := by omega
  have hdvd30 : 30 ∣ primorial X := wall_thirty_dvd_primorial (by omega)
  have hdvdp : p ∣ primorial X := (hp.dvd_primorial_iff).mpr hpX.le
  have hdvdX : X ∣ primorial X := (hX.dvd_primorial_iff).mpr le_rfl
  have hcop_X : Nat.Coprime (30 * p) X :=
    Nat.coprime_mul_iff_left.mpr ⟨coprime_thirty_of_prime hX (by omega),
      (Nat.coprime_primes hp hX).mpr (by omega)⟩
  have h30p : 30 * p ∣ primorial X :=
    Nat.Coprime.mul_dvd_of_dvd_of_dvd (coprime_thirty_of_prime hp hp7) hdvd30 hdvdp
  have hall : 30 * p * X ∣ primorial X := Nat.Coprime.mul_dvd_of_dvd_of_dvd hcop_X h30p hdvdX
  have hle : 30 * p * X ≤ primorial X := Nat.le_of_dvd (primorial_pos X) hall
  nlinarith

/-- The quantitative wall at primes: `(2X)² < X#` for every prime `X ≥ 7`. -/
theorem four_sq_lt_primorial_of_prime {X : ℕ} (hX : X.Prime) (h7 : 7 ≤ X) :
    (2 * X) ^ 2 < primorial X := by
  rcases Nat.lt_or_ge X 11 with h | h
  · interval_cases X
    · rw [wall_primorial_seven]; norm_num
    all_goals exact absurd hX (by norm_num)
  · have := fifteen_mul_le_primorial hX h
    nlinarith

/-- Bertrand: for every `X ≥ 1`, a number at most every prime above `X` is at most `2X`. -/
theorem le_two_mul_of_least {X p : ℕ} (hX : 1 ≤ X) (hmin : ∀ r, r.Prime → X < r → p ≤ r) :
    p ≤ 2 * X := by
  obtain ⟨r, hr, h1, h2⟩ := Nat.exists_prime_lt_and_le_two_mul X (by omega)
  exact le_trans (hmin r hr h1) h2

/-- **Wall inequality, prime form.** For every prime `X ≥ 7` and every `p` at most every prime
above `X` (in particular `p = nextprime X`): `p² < X#`. -/
theorem wall_inequality {X p : ℕ} (hX : X.Prime) (h7 : 7 ≤ X)
    (hmin : ∀ r, r.Prime → X < r → p ≤ r) : p ^ 2 < primorial X := by
  have h1 := le_two_mul_of_least (by omega) hmin
  have h2 := four_sq_lt_primorial_of_prime hX h7
  calc p ^ 2 ≤ (2 * X) ^ 2 := Nat.pow_le_pow_left h1 2
    _ < primorial X := h2

/-- **Wall inequality, every natural `X ≥ 7`.** For every natural `X ≥ 7` and every `p` at most
every prime above `X`: `p² < X#` (reduce to the largest prime `≤ X`). -/
theorem wall_inequality_nat {X p : ℕ} (h7 : 7 ≤ X)
    (hmin : ∀ r, r.Prime → X < r → p ≤ r) : p ^ 2 < primorial X := by
  have hP7 : 7 ≤ Nat.findGreatest Nat.Prime X := Nat.le_findGreatest h7 (by norm_num)
  have hPX : Nat.findGreatest Nat.Prime X ≤ X := Nat.findGreatest_le X
  have hPprime : (Nat.findGreatest Nat.Prime X).Prime :=
    Nat.findGreatest_spec (P := Nat.Prime) h7 (by norm_num)
  have hminP : ∀ r, r.Prime → Nat.findGreatest Nat.Prime X < r → p ≤ r := by
    intro r hr hPr
    rcases Nat.lt_or_ge X r with h | h
    · exact hmin r hr h
    · exact absurd hr (Nat.findGreatest_is_greatest hPr h)
  calc p ^ 2 < primorial (Nat.findGreatest Nat.Prime X) := wall_inequality hPprime hP7 hminP
    _ ≤ primorial X := primorial_mono hPX

/-- **Below `7` the wall fails.** For every `X < 7` and every `p > X`: `X# ≤ p²`. -/
theorem wall_fails_below_seven {X p : ℕ} (hX : X < 7) (hp : X < p) : primorial X ≤ p ^ 2 := by
  have h0 : primorial 0 = 1 := by decide
  have h1 : primorial 1 = 1 := by decide
  have h2 : primorial 2 = 2 := by decide
  have h3 : primorial 3 = 6 := by decide
  have h4 : primorial 4 = 6 := by decide
  have h6 : primorial 6 = 30 := by decide
  interval_cases X
  · rw [h0]; nlinarith
  · rw [h1]; nlinarith
  · rw [h2]; nlinarith
  · rw [h3]; nlinarith
  · rw [h4]; nlinarith
  · rw [wall_primorial_five]; nlinarith
  · rw [h6]; nlinarith

/-- **The exact threshold.** For every natural `X` and `p` the least prime above `X`:
`p² < X# ↔ 7 ≤ X`. -/
theorem wall_iff {X p : ℕ} (hXp : X < p) (hmin : ∀ r, r.Prime → X < r → p ≤ r) :
    p ^ 2 < primorial X ↔ 7 ≤ X := by
  constructor
  · intro h
    by_contra h7
    have := wall_fails_below_seven (by omega) hXp
    omega
  · intro h7
    exact wall_inequality_nat h7 hmin

/-- Doubled wall, large case: for every `X ≥ 44`, `(2X)² < X#` (Bertrand twice gives primes
`R ∈ (X/2, X]`, `p ∈ (R/2, R)` with `p ≥ 12`, so `210 · p · R ∣ X#`). -/
theorem four_sq_lt_primorial_large {X : ℕ} (hX : 44 ≤ X) : (2 * X) ^ 2 < primorial X := by
  obtain ⟨R, hR, hRlo, hRhi⟩ := Nat.exists_prime_lt_and_le_two_mul (X / 2) (by omega)
  obtain ⟨p, hp, hplo, hphi⟩ := Nat.exists_prime_lt_and_le_two_mul (R / 2) (by omega)
  have hRodd : R % 2 = 1 := Nat.odd_iff.mp (hR.odd_of_ne_two (by omega))
  have hpR : p < R := by omega
  have hp8 : 8 ≤ p := by omega
  have hRX : R ≤ X := by omega
  have h7d : primorial 7 ∣ primorial X := primorial_dvd_primorial (by omega)
  have hpd : p ∣ primorial X := (hp.dvd_primorial_iff).mpr (by omega)
  have hRd : R ∣ primorial X := (hR.dvd_primorial_iff).mpr hRX
  have c7p : Nat.Coprime (primorial 7) p :=
    ((Nat.Prime.coprime_iff_not_dvd hp).mpr
      (fun h => by have := (hp.dvd_primorial_iff).mp h; omega)).symm
  have c7R : Nat.Coprime (primorial 7) R :=
    ((Nat.Prime.coprime_iff_not_dvd hR).mpr
      (fun h => by have := (hR.dvd_primorial_iff).mp h; omega)).symm
  have cpR : Nat.Coprime p R := (Nat.coprime_primes hp hR).mpr (by omega)
  have h7p : primorial 7 * p ∣ primorial X := Nat.Coprime.mul_dvd_of_dvd_of_dvd c7p h7d hpd
  have c7pR : Nat.Coprime (primorial 7 * p) R := Nat.Coprime.mul_left c7R cpR
  have h7pR : primorial 7 * p * R ∣ primorial X :=
    Nat.Coprime.mul_dvd_of_dvd_of_dvd c7pR h7p hRd
  have hle := Nat.le_of_dvd (primorial_pos X) h7pR
  rw [wall_primorial_seven] at hle
  have hXR : X + 1 ≤ 2 * R := by omega
  have hRp : R + 1 ≤ 2 * p := by omega
  have h1 : (X + 1) * (X + 1) ≤ (2 * R) * (2 * R) := Nat.mul_le_mul hXR hXR
  have h2 : (R + 1) * R ≤ (2 * p) * R := Nat.mul_le_mul_right R hRp
  nlinarith

/-- Doubled wall, middle case: for every `11 ≤ X ≤ 43`, `(2X)² < X#` (monotonicity from
`11# = 2310` and `23# = 223092870`). -/
theorem four_sq_lt_primorial_mid {X : ℕ} (h11 : 11 ≤ X) (h43 : X ≤ 43) :
    (2 * X) ^ 2 < primorial X := by
  by_cases h22 : X ≤ 22
  · have hm : primorial 11 ≤ primorial X := primorial_mono h11
    rw [wall_primorial_eleven] at hm
    have : (2 * X) ^ 2 ≤ 44 ^ 2 := Nat.pow_le_pow_left (by omega) 2
    omega
  · have hm : primorial 23 ≤ primorial X := primorial_mono (by omega)
    rw [wall_primorial_twentythree] at hm
    have : (2 * X) ^ 2 ≤ 86 ^ 2 := Nat.pow_le_pow_left (by omega) 2
    omega

/-- Doubled wall: `(2X)² < X#` for every natural `X ≥ 11`. -/
theorem four_sq_lt_primorial_of_ge_eleven {X : ℕ} (h11 : 11 ≤ X) :
    (2 * X) ^ 2 < primorial X := by
  by_cases h : 44 ≤ X
  · exact four_sq_lt_primorial_large h
  · exact four_sq_lt_primorial_mid h11 (by omega)

/-- **Doubled wall, exact domain.** For every natural `X`:
`(2X)² < X# ↔ X = 0 ∨ X = 7 ∨ 11 ≤ X` (so over `X ≥ 7` the failures are exactly `8, 9, 10`). -/
theorem four_sq_lt_primorial_iff (X : ℕ) :
    (2 * X) ^ 2 < primorial X ↔ X = 0 ∨ X = 7 ∨ 11 ≤ X := by
  constructor
  · intro h
    by_contra hc
    push Not at hc
    obtain ⟨h0, h7, h11⟩ := hc
    interval_cases X <;> first | omega | (revert h; decide)
  · rintro (rfl | rfl | h)
    · decide
    · decide
    · exact four_sq_lt_primorial_of_ge_eleven h

/-- **Wall, nextprime form.** For every natural `X ≥ 7` and `r` the least prime above `X`:
`r² < X#`. -/
theorem nextprime_sq_lt_primorial {X r : ℕ} (hX : 7 ≤ X) (_hr : r.Prime) (_hXr : X < r)
    (hmin : ∀ s, s.Prime → X < s → r ≤ s) : r ^ 2 < primorial X := by
  by_cases h10 : X ≤ 10
  · have hr11 : r ≤ 11 := hmin 11 (by norm_num) (by omega)
    have hm : primorial 7 ≤ primorial X := primorial_mono hX
    rw [wall_primorial_seven] at hm
    have : r ^ 2 ≤ 11 ^ 2 := Nat.pow_le_pow_left hr11 2
    omega
  · obtain ⟨s, hs, hslo, hshi⟩ := Nat.exists_prime_lt_and_le_two_mul X (by omega)
    have hs2 : s ≠ 2 * X := by
      rintro rfl
      have := (Nat.prime_mul_iff.mp hs)
      omega
    have hrs : r ≤ s := hmin s hs hslo
    have hlt : r < 2 * X := by omega
    have := Nat.pow_lt_pow_left hlt (by norm_num : (2 : ℕ) ≠ 0)
    have := four_sq_lt_primorial_of_ge_eleven (X := X) (by omega)
    omega

/-- For every `X ≥ 5`: `30 · (X#/30) = X#`. -/
theorem thirty_mul_period {X : ℕ} (hX : 5 ≤ X) : 30 * (primorial X / 30) = primorial X :=
  Nat.mul_div_cancel' (wall_thirty_dvd_primorial hX)

/-- **Acting from the period on.** For every `q ≥ 7` and every `g` at most every prime above `q`
(so `g = q'`, or any gear `g ≤ q'`): every copy `j ≥ q#/30` has `g² ≤ 30 j + 1`. -/
theorem acts_from_period {q g j : ℕ} (hq : 7 ≤ q) (hmin : ∀ r, r.Prime → q < r → g ≤ r)
    (hj : primorial q / 30 ≤ j) : g ^ 2 ≤ 30 * j + 1 := by
  have h1 := wall_inequality_nat hq hmin
  have h2 := thirty_mul_period (X := q) (by omega)
  have h3 : 30 * (primorial q / 30) ≤ 30 * j := Nat.mul_le_mul_left 30 hj
  omega

/-- **Window cut is below `q#`.** For every natural `q ≥ 7`, `q'` the least prime above `q`, and
every `j` with `30 j + 1 < q'²`: `30 j + 1 < q#` and `j < q# / 30`. -/
theorem window_below_primorial {q q' j : ℕ} (hq : 7 ≤ q) (hq' : q'.Prime) (hqq' : q < q')
    (hmin : ∀ s, s.Prime → q < s → q' ≤ s) (hj : 30 * j + 1 < q' ^ 2) :
    30 * j + 1 < primorial q ∧ j < primorial q / 30 := by
  have h := nextprime_sq_lt_primorial hq hq' hqq' hmin
  obtain ⟨M, hM⟩ := wall_thirty_dvd_primorial (X := q) (by omega)
  rw [hM, Nat.mul_div_cancel_left M (by norm_num)]
  constructor <;> omega

/-- **Window copies in the range.** For every natural `q ≥ 7`, `q'` the least prime above `q`, and
every `j` with `30 j + 1 < q'²`: `j` is a copy of the range (`q < 30 j - 1` and `30 j + 1 ≤ q#`)
iff its lower leg is above `q`. -/
theorem window_mem_range_iff {q q' j : ℕ} (hq : 7 ≤ q) (hq' : q'.Prime) (hqq' : q < q')
    (hmin : ∀ s, s.Prime → q < s → q' ≤ s) (hj : 30 * j + 1 < q' ^ 2) :
    (q < 30 * j - 1 ∧ 30 * j + 1 ≤ primorial q) ↔ q < 30 * j - 1 := by
  have h := (window_below_primorial hq hq' hqq' hmin hj).1
  constructor
  · exact fun h' => h'.1
  · exact fun h' => ⟨h', h.le⟩

/-- **Window copies below the range.** For every natural `q` and every `q' > q`: the copies
`j ≥ 1` with `30 j + 1 < q'²` and lower leg `≤ q` are exactly `1 ≤ j ≤ (q + 1) / 30`. -/
theorem below_range_iff {q q' j : ℕ} (hqq' : q < q') :
    (1 ≤ j ∧ 30 * j + 1 < q' ^ 2 ∧ 30 * j - 1 ≤ q) ↔ (1 ≤ j ∧ j ≤ (q + 1) / 30) := by
  constructor
  · rintro ⟨h1, _, h3⟩
    exact ⟨h1, by omega⟩
  · rintro ⟨h1, h2⟩
    refine ⟨h1, ?_, by omega⟩
    have hq29 : 29 ≤ q := by omega
    have hq30 : 30 * q' ≤ q' ^ 2 := by nlinarith
    omega

/-- **Some window copy below the range iff `29 ≤ q`.** For every natural `q` and every `q' > q`:
some `j ≥ 1` with `30 j + 1 < q'²` has `30 j - 1 ≤ q` iff `29 ≤ q`. -/
theorem below_range_exists_iff {q q' : ℕ} (hqq' : q < q') :
    (∃ j, 1 ≤ j ∧ 30 * j + 1 < q' ^ 2 ∧ 30 * j - 1 ≤ q) ↔ 29 ≤ q := by
  constructor
  · rintro ⟨j, h1, _, h3⟩
    omega
  · intro h
    exact ⟨1, le_refl 1, by nlinarith, by omega⟩

/-- **Upper-leg locator.** For every prime `g ≥ 7` with `g ∤ N` and every `r`: some `k < g` has
`g ∣ 30 (r + kN) + 1`. -/
theorem locator_closure_plus {g N : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g) (hgN : ¬ g ∣ N) (r : ℕ) :
    ∃ k < g, g ∣ 30 * (r + k * N) + 1 := by
  have := Fact.mk hg
  have h30 := thirty_ne_zero hg hg7
  obtain ⟨k, hk, hmod⟩ :=
    class_meets_every_residue hg hgN r ((-((30 : ℕ) : ZMod g)⁻¹).val)
  rw [← ZMod.natCast_eq_natCast_iff', ZMod.natCast_zmod_val] at hmod
  refine ⟨k, hk, ?_⟩
  rw [← ZMod.natCast_eq_zero_iff, Nat.cast_add, Nat.cast_mul, hmod, Nat.cast_one, mul_neg,
    mul_inv_cancel₀ h30]
  ring

/-- **Lap bound.** For every `q ≥ 7`, `q'` the least prime above `q`, `M = q#/30` and every `r`:
some `k` with `1 ≤ k ≤ q'` gives a copy `j = r + kM` with `q' ∣ 30 j - 1` on which `q'` acts. -/
theorem lap_bound {q q' r : ℕ} (h7 : 7 ≤ q) (hq' : q'.Prime) (hqq' : q < q')
    (hmin : ∀ s, s.Prime → q < s → q' ≤ s) :
    ∃ k, 1 ≤ k ∧ k ≤ q' ∧ q' ∣ 30 * (r + k * (primorial q / 30)) - 1 ∧
      q' ^ 2 ≤ 30 * (r + k * (primorial q / 30)) + 1 := by
  have hM := thirty_mul_period (X := q) (by omega)
  have hndvd : ¬ q' ∣ primorial q / 30 := by
    intro h
    have h' : q' ∣ primorial q := h.trans ⟨30, by rw [Nat.mul_comm]; exact hM.symm⟩
    have := (hq'.dvd_primorial_iff).mp h'
    omega
  obtain ⟨k, hk, _, hdvd⟩ :=
    locator_closure_minus hq' (by omega) hndvd (r + primorial q / 30)
  refine ⟨k + 1, by omega, by omega, ?_, ?_⟩
  · have : r + (k + 1) * (primorial q / 30) = r + primorial q / 30 + k * (primorial q / 30) := by
      ring
    rw [this]; exact hdvd
  · exact acts_from_period h7 hmin (by nlinarith)

/-- **Lap bound, sharp.** For every `q ≥ 7`, `q'` the least prime above `q`, `M = q#/30` and every
`r`: some `k` with `1 ≤ k ≤ q' - 1` gives a copy `j = r + kM` struck by `q'`
(`q' ∣ (30j - 1)(30j + 1)`) and acted on by `q'`. -/
theorem lap_bound_sharp {q q' r : ℕ} (h7 : 7 ≤ q) (hq' : q'.Prime) (hqq' : q < q')
    (hmin : ∀ s, s.Prime → q < s → q' ≤ s) :
    ∃ k, 1 ≤ k ∧ k + 1 ≤ q' ∧
      q' ∣ (30 * (r + k * (primorial q / 30)) - 1) * (30 * (r + k * (primorial q / 30)) + 1) ∧
      q' ^ 2 ≤ 30 * (r + k * (primorial q / 30)) + 1 := by
  set M := primorial q / 30 with hMdef
  have hM : 30 * M = primorial q := thirty_mul_period (X := q) (by omega)
  have hndvd : ¬ q' ∣ M := by
    intro h
    have h' : q' ∣ primorial q := h.trans ⟨30, by rw [Nat.mul_comm]; exact hM.symm⟩
    have := (hq'.dvd_primorial_iff).mp h'
    omega
  obtain ⟨k1, hk1, hpos1, hdvd1⟩ := locator_closure_minus hq' (by omega) hndvd (r + M)
  obtain ⟨k2, hk2, hdvd2⟩ := locator_closure_plus hq' (by omega) hndvd (r + M)
  have hne : k1 ≠ k2 := by
    intro h
    subst h
    have h2 : q' ∣ (30 * (r + M + k1 * M) + 1) - (30 * (r + M + k1 * M) - 1) :=
      Nat.dvd_sub hdvd2 hdvd1
    have h3 : (30 * (r + M + k1 * M) + 1) - (30 * (r + M + k1 * M) - 1) = 2 := by omega
    rw [h3] at h2
    have := Nat.le_of_dvd (by norm_num) h2
    omega
  have hshift : ∀ k, r + (k + 1) * M = r + M + k * M := fun k => by ring
  have hact : ∀ k, q' ^ 2 ≤ 30 * (r + (k + 1) * M) + 1 := fun k =>
    acts_from_period h7 hmin (by nlinarith)
  rcases Nat.lt_or_ge k1 k2 with h | h
  · refine ⟨k1 + 1, by omega, by omega, ?_, hact k1⟩
    rw [hshift]; exact Dvd.dvd.mul_right hdvd1 _
  · refine ⟨k2 + 1, by omega, by omega, ?_, hact k2⟩
    rw [hshift]; exact Dvd.dvd.mul_left hdvd2 _

/-- For every `X ≥ 5`: the gears `[7, X]` times the lower set `{2, 3, 5}` are the primorial,
`30 · ∏_{p ∈ [7, X] prime} p = X#`. -/
theorem thirty_mul_prod_Icc {X : ℕ} (hX : 5 ≤ X) :
    30 * ∏ p ∈ (Finset.Icc 7 X).filter Nat.Prime, p = primorial X := by
  have hsplit : (Finset.range (X + 1)).filter Nat.Prime =
      ({2, 3, 5} : Finset ℕ) ∪ (Finset.Icc 7 X).filter Nat.Prime := by
    ext x
    simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_union, Finset.mem_insert,
      Finset.mem_singleton, Finset.mem_Icc]
    constructor
    · rintro ⟨hx, hp⟩
      by_cases h7 : 7 ≤ x
      · exact Or.inr ⟨⟨h7, by omega⟩, hp⟩
      · left
        interval_cases x <;> first | (exfalso; norm_num at hp; done) | simp
    · rintro ((h | h | h) | ⟨⟨h1, h2⟩, hp⟩)
      · subst h; exact ⟨by omega, Nat.prime_two⟩
      · subst h; exact ⟨by omega, Nat.prime_three⟩
      · subst h; exact ⟨by omega, Nat.prime_five⟩
      · exact ⟨by omega, hp⟩
  have hdisj : Disjoint ({2, 3, 5} : Finset ℕ) ((Finset.Icc 7 X).filter Nat.Prime) := by
    rw [Finset.disjoint_left]
    intro a ha hb
    simp only [Finset.mem_insert, Finset.mem_singleton] at ha
    simp only [Finset.mem_filter, Finset.mem_Icc] at hb
    omega
  unfold primorial
  rw [hsplit, Finset.prod_union hdisj]
  simp

/-- **Silent class, modulus form.** For every `N ≥ 1`, `r` and `X ≥ 5`: a class `j ≡ r (mod N)`
no copy `j ≥ 1` of which is struck by a prime of `[7, X]` has `X# ≤ 30 N`. -/
theorem silent_class_period {N r X : ℕ} (hN : 1 ≤ N) (h5 : 5 ≤ X)
    (hclass : ∀ j, 1 ≤ j → j ≡ r [MOD N] →
      ∀ g, g.Prime → 7 ≤ g → g ≤ X → ¬ g ∣ (30 * j - 1) * (30 * j + 1)) :
    primorial X ≤ 30 * N := by
  have h := locator_modulus_le hN hclass
  rw [← thirty_mul_prod_Icc h5]
  exact Nat.mul_le_mul_left 30 h

/-- **Silent class, wall form.** Same class, for every `X ≥ 7` and `p` at most every prime above
`X`: `p² < 30 N`. -/
theorem silent_class_beyond_wall {N r X p : ℕ} (hN : 1 ≤ N) (h7 : 7 ≤ X)
    (hmin : ∀ s, s.Prime → X < s → p ≤ s)
    (hclass : ∀ j, 1 ≤ j → j ≡ r [MOD N] →
      ∀ g, g.Prime → 7 ≤ g → g ≤ X → ¬ g ∣ (30 * j - 1) * (30 * j + 1)) :
    p ^ 2 < 30 * N :=
  lt_of_lt_of_le (wall_inequality_nat h7 hmin) (silent_class_period hN (by omega) hclass)

/-- **Silent class, one member per period.** For every `X ≥ 5`, such a class has at most one copy
`j ≥ 1` with `30 j ≤ X#`. -/
theorem silent_class_one_per_period {N r X j₁ j₂ : ℕ} (hN : 1 ≤ N) (h5 : 5 ≤ X)
    (hclass : ∀ j, 1 ≤ j → j ≡ r [MOD N] →
      ∀ g, g.Prime → 7 ≤ g → g ≤ X → ¬ g ∣ (30 * j - 1) * (30 * j + 1))
    (h1 : 1 ≤ j₁) (h2 : 1 ≤ j₂) (hc1 : j₁ ≡ r [MOD N]) (hc2 : j₂ ≡ r [MOD N])
    (hb1 : 30 * j₁ ≤ primorial X) (hb2 : 30 * j₂ ≤ primorial X) : j₁ = j₂ := by
  have hP := silent_class_period hN h5 hclass
  have h12 : j₁ ≡ j₂ [MOD N] := hc1.trans hc2.symm
  by_contra hne
  rcases Nat.lt_or_ge j₁ j₂ with h | h
  · have hd : N ∣ j₂ - j₁ := (Nat.modEq_iff_dvd' h.le).mp h12
    have := Nat.le_of_dvd (by omega) hd
    omega
  · have h' : j₂ < j₁ := by omega
    have hd : N ∣ j₁ - j₂ := (Nat.modEq_iff_dvd' h'.le).mp h12.symm
    have := Nat.le_of_dvd (by omega) hd
    omega

/-- **Silent class, one member in the certified band.** For every `X ≥ 7` and `p` at most every
prime above `X`, such a class has at most one copy `j ≥ 1` with `30 j + 1 < p²`. -/
theorem silent_class_one_in_band {N r X p j₁ j₂ : ℕ} (hN : 1 ≤ N) (h7 : 7 ≤ X)
    (hmin : ∀ s, s.Prime → X < s → p ≤ s)
    (hclass : ∀ j, 1 ≤ j → j ≡ r [MOD N] →
      ∀ g, g.Prime → 7 ≤ g → g ≤ X → ¬ g ∣ (30 * j - 1) * (30 * j + 1))
    (h1 : 1 ≤ j₁) (h2 : 1 ≤ j₂) (hc1 : j₁ ≡ r [MOD N]) (hc2 : j₂ ≡ r [MOD N])
    (hb1 : 30 * j₁ + 1 < p ^ 2) (hb2 : 30 * j₂ + 1 < p ^ 2) : j₁ = j₂ := by
  have hw := wall_inequality_nat h7 hmin
  exact silent_class_one_per_period hN (by omega) hclass h1 h2 hc1 hc2 (by omega) (by omega)

end RangeLine
