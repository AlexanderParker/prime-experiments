import Mathlib.NumberTheory.Bertrand
import Mathlib.NumberTheory.Primorial
import Mathlib.Data.Set.Card
import Mathlib.Tactic
import RangeMirror
import RangeActPair
import RangeHandoff
import RangeLocator

/-!
# The gain structure of the range, for every `q ≥ 7` (P2)

Copy `j` is the pair of legs `30 j - 1`, `30 j + 1`; a prime `p` strikes copy `j` when it divides
a leg (`StrikesCopy`, from `RangeCentre`); a gear `g` acts at copy `j` when `g² ≤ 30 j + 1`
(`Acts`, from `RangeActPair`). Fix any natural number `q ≥ 7` (no primality, no size bound) and
write `q#` for `primorial q`, `M = q#/30` (`mirrorM`, from `RangeMirror`).

The notation below is the notation of the height split: the same definitions, with the same
bodies, as in the module `RangeGen1`. They are declared here, in the sub-namespace
`RangeLine.Gen2`, so that this module does not depend on `RangeGen1` and the two can be imported
together; each definition is equal by `rfl`/`Iff.rfl` to its namesake there.

* `ρ = q#/2 + 1 = 15 M + 1` (`rho`); `Q` = the largest prime `≤ √ρ` (`cutLo`);
  `Q*` = the largest prime `≤ √(q#)` (`cutHi`).
* `S_X` (`Clear X j`): no prime `p` with `7 ≤ p ≤ X` strikes copy `j`.
* low half `[1, (M-1)/2]` (`LowHalf`), high half `[(M+1)/2, M-1]` (`HighHalf`).
* `U⁻` = primes in `(q, Q]` (`Uminus`), `B` = primes in `(Q, Q*]` (`Band`), `a_g = 30⁻¹ (mod g)`
  (`gearRes`).
* `Revealed q j`: `1 ≤ j ≤ M - 1`, `j ∈ S_q`, both legs prime. `Loss q j`: a twin copy whose lower
  leg lies in `(q, Q]`.
* `G_q` (`GainCopy q j`): `(S_Q ∩ high) \ S_{Q*}`.

## What is proved (every statement is for every `q ≥ 7`, every copy, every gear)

**(i) Gain set.**
* `gain_iff_band_strike`: `G_q = S_Q ∩ high ∩ {j : some h ∈ B strikes j}`;
  `gain_iff_residue`: the same with the strike written `j ≡ ±a_h (mod h)`.
* `revealed_iff_gain`: `Revealed_q = Loss_q ∪ ((S_Q ∩ [1, M-1]) \ G_q)`, and `loss_not_clear_cutLo`
  (`Loss_q ∩ S_Q = ∅`) makes the union disjoint.

**(ii) Composite legs.** `high_leg_factor`, `high_leg_two_primes`: every composite leg `L` of a
copy in `S_Q ∩ high` is `L = h · c` with `h = minFac L ∈ B`, `c` prime, `h ≤ c < 2h`; `h` acts on
the copy, and `c` does not act when `c > h`. `high_leg_band_iff`: on such a copy, a leg is
composite iff some `h ∈ B` divides it.

**(iii) Pair form.** `GainPair q j h ε c` is the pair condition: `h ∈ B`, `ε = ±1`, `c` prime,
`h ≤ c ≤ (q# - 29)/h`, `c ≡ ε h⁻¹ (mod 30)`, `c ≢ 0` and `c ≢ 2 ε h⁻¹ (mod p)` for every prime
`7 ≤ p ≤ Q`, and `j = (h c - ε)/30`.
* `gain_iff_pair`: `j ∈ G_q ⇔ ∃ h ε c, GainPair q j h ε c`.
* `gainPair_iff_leg`: on `S_Q ∩ high`, the pairs of copy `j` are exactly
  `(minFac L, ε, L / minFac L)` for the composite legs `L = 30 j + ε`.
* `pair_count`: for `j ∈ G_q`, the set of pairs of `j` has exactly two elements iff both legs are
  composite, and exactly one element otherwise.
* `strike_iff_pair_class`, `mirror_strike_iff_pair_class`: with `30 j + ε = h c`, a prime `p ∤ h`
  strikes copy `j` iff `c ≡ 0` or `c ≡ 2 ε h⁻¹ (mod p)`, and strikes the mirror copy `M - j` iff
  `c ≡ h⁻¹ q#` or `c ≡ h⁻¹ (q# + 2 ε) (mod p)`.

**(iv) Mirror trichotomy.** `low_trichotomy`: for every low `x ∈ S_q`, with the three cases
(a) `NoUStrike` (no `U⁻` strike), (b) `QuietUStrike` (some `U⁻` strike, none acting),
(c) `ActingUStrike` (some acting `U⁻` strike):
(a) `⇔` revealed and not in `Loss`; (b) `⇔` revealed and in `Loss`; (c) `⇔` not revealed.
`trichotomy_exactly_one`: exactly one of (a), (b), (c) holds. `quiet_striker_is_leg`: in case (b)
every `U⁻` striker is a leg (shadowing). `mirror_trichotomy`: for `j ∈ G_q`, `x = M - j` is a low
survivor and all of this applies to it.

**(v) Shared class.** `shared_class`: if a prime `h > Q` divides `q# - 2` and `d` (odd `d < M`),
then the low member `s = (M - d)/2` has minus leg `30 s - 1 = h · c*` with `c* < h` and
`c*² < ρ`, `h` also divides the high member's minus leg, and if `s ∈ S_q` then every prime factor
of `c*` is a `U⁻` gear striking and acting on `s`. `shared_class_band`: the case `h ∈ B`.

**(vi) Three-class rule.** `FourClassesDistinct p h ε N`: the classes `0`, `2 ε h⁻¹`, `h⁻¹ N`,
`h⁻¹ (N + 2 ε)` in `ZMod p` are pairwise distinct.
* `four_classes_distinct_iff`: for every odd prime `p ∤ h` and every `N ≥ 2`, they are distinct
  iff `p ∤ N`, `p ∤ N - 2` and `p ∤ N + 2`.
* `three_class_rule`: for every prime `p > q` (in particular every `U⁻` gear) and `p ∤ h`:
  the four classes at `N = q#` are distinct iff `p ∤ q# - 2` and `p ∤ q# + 2`.
  `three_class_rule_band`: the case `p ∈ U⁻`, `h ∈ B`.

## Corrections to the requested wording

* (vi) As written ("distinct iff `p ∤ q# ± 2`") it holds for primes `p > q` only: for `p ≤ q`,
  `h⁻¹ q# ≡ 0`, so two classes always coincide. The general rule (`four_classes_distinct_iff`) is
  "distinct iff `p ∤ q# (q# - 2)(q# + 2)`", for every odd prime `p ∤ h`. For `p ∈ U⁻`, `h ∈ B`
  both side conditions hold automatically.
* (v) The factor `c*` satisfies `1 ≤ c* < √ρ`; `c* = 1` is not excluded by the argument (it would
  need the minus leg itself to be `h`), so `1 < c*` is not claimed. The hypothesis `h ∈ B` is
  weakened to "prime `h > Q`" (the bound `h ≤ Q*` is not used).
* (iv) is proved for every low survivor `x`, as three equivalences, not only for mirrors of `G_q`
  copies; `Loss_q` is taken, as in P1, as the twin copies whose lower leg is in `(q, Q]`.
-/

namespace RangeLine

namespace Gen2

/-! ## Notation (as in the height split) -/

/-- `ρ = q#/2 + 1`. -/
def rho (q : ℕ) : ℕ := primorial q / 2 + 1

/-- `Q`: the largest prime `≤ √ρ` (`0` if there is none; for `q ≥ 7` it is prime). -/
def cutLo (q : ℕ) : ℕ := Nat.findGreatest Nat.Prime (Nat.sqrt (rho q))

/-- `Q*`: the largest prime `≤ √(q#)`. -/
def cutHi (q : ℕ) : ℕ := Nat.findGreatest Nat.Prime (Nat.sqrt (primorial q))

/-- `S_X`: no prime `p` with `7 ≤ p ≤ X` strikes copy `j`. -/
def Clear (X j : ℕ) : Prop := ∀ p, p.Prime → 7 ≤ p → p ≤ X → ¬ StrikesCopy p j

/-- Copy `j` is a twin copy: both legs `30 j - 1` and `30 j + 1` are prime. -/
def Twin (j : ℕ) : Prop := (30 * j - 1).Prime ∧ (30 * j + 1).Prime

/-- The low half of the range: `1 ≤ j ≤ (M - 1)/2`. -/
def LowHalf (q j : ℕ) : Prop := 1 ≤ j ∧ j ≤ (mirrorM q - 1) / 2

/-- The high half of the range: `(M + 1)/2 ≤ j ≤ M - 1`. -/
def HighHalf (q j : ℕ) : Prop := (mirrorM q + 1) / 2 ≤ j ∧ j ≤ mirrorM q - 1

/-- A revealed range copy: `1 ≤ j ≤ M - 1`, `j ∈ S_q`, and both legs prime. -/
def Revealed (q j : ℕ) : Prop := 1 ≤ j ∧ j ≤ mirrorM q - 1 ∧ Clear q j ∧ Twin j

/-- `Loss_q`: a twin copy whose lower leg lies in `(q, Q]`. -/
def Loss (q j : ℕ) : Prop := 1 ≤ j ∧ q < 30 * j - 1 ∧ 30 * j - 1 ≤ cutLo q ∧ Twin j

/-- `U⁻`: the primes in `(q, Q]`. -/
def Uminus (q g : ℕ) : Prop := g.Prime ∧ q < g ∧ g ≤ cutLo q

/-- `B`: the primes in `(Q, Q*]`. -/
def Band (q g : ℕ) : Prop := g.Prime ∧ cutLo q < g ∧ g ≤ cutHi q

/-- `a_g = 30⁻¹` in `ZMod g`. -/
def gearRes (g : ℕ) : ZMod g := (30 : ZMod g)⁻¹

/-- `G_q = (S_Q ∩ high) \ S_{Q*}`: high copies clear up to `Q` but struck below `Q*`. -/
def GainCopy (q j : ℕ) : Prop := HighHalf q j ∧ Clear (cutLo q) j ∧ ¬ Clear (cutHi q) j

/-- The pair condition of the pair form: `h ∈ B`, `ε = ±1`, `c` prime,
`h ≤ c ≤ (q# - 29)/h`, `c ≡ ε h⁻¹ (mod 30)`, `c ≢ 0, 2 ε h⁻¹ (mod p)` for every prime
`7 ≤ p ≤ Q`, and `j = (h c - ε)/30`. -/
def GainPair (q j h : ℕ) (ε : ℤ) (c : ℕ) : Prop :=
  Band q h ∧ (ε = 1 ∨ ε = -1) ∧ c.Prime ∧ h ≤ c ∧ c ≤ (primorial q - 29) / h ∧
    (c : ZMod 30) = (ε : ZMod 30) * (h : ZMod 30)⁻¹ ∧
    (∀ p, p.Prime → 7 ≤ p → p ≤ cutLo q →
      (c : ZMod p) ≠ 0 ∧ (c : ZMod p) ≠ 2 * (ε : ZMod p) * (h : ZMod p)⁻¹) ∧
    (j : ℤ) = ((h : ℤ) * c - ε) / 30

/-- The set of pairs `(h, ε, c)` from which copy `j` arises. -/
def pairSet (q j : ℕ) : Set (ℕ × ℤ × ℕ) := {t | GainPair q j t.1 t.2.1 t.2.2}

/-- Case (a) of the trichotomy: no `U⁻` gear strikes copy `x`. -/
def NoUStrike (q x : ℕ) : Prop := ∀ p, Uminus q p → ¬ StrikesCopy p x

/-- Case (b) of the trichotomy: some `U⁻` gear strikes copy `x`, and none of the `U⁻` strikers
acts on `x`. -/
def QuietUStrike (q x : ℕ) : Prop :=
  (∃ p, Uminus q p ∧ StrikesCopy p x) ∧ ∀ p, Uminus q p → StrikesCopy p x → ¬ Acts p x

/-- Case (c) of the trichotomy: some `U⁻` gear strikes copy `x` and acts on it. -/
def ActingUStrike (q x : ℕ) : Prop := ∃ p, Uminus q p ∧ StrikesCopy p x ∧ Acts p x

/-- The four classes `0`, `2 ε h⁻¹`, `h⁻¹ N`, `h⁻¹ (N + 2 ε)` in `ZMod p` are pairwise
distinct. -/
def FourClassesDistinct (p h : ℕ) (ε : ℤ) (N : ℕ) : Prop :=
  (0 : ZMod p) ≠ 2 * (ε : ZMod p) * (h : ZMod p)⁻¹ ∧
  (0 : ZMod p) ≠ (h : ZMod p)⁻¹ * (N : ZMod p) ∧
  (0 : ZMod p) ≠ (h : ZMod p)⁻¹ * ((N : ZMod p) + 2 * (ε : ZMod p)) ∧
  2 * (ε : ZMod p) * (h : ZMod p)⁻¹ ≠ (h : ZMod p)⁻¹ * (N : ZMod p) ∧
  2 * (ε : ZMod p) * (h : ZMod p)⁻¹ ≠ (h : ZMod p)⁻¹ * ((N : ZMod p) + 2 * (ε : ZMod p)) ∧
  (h : ZMod p)⁻¹ * (N : ZMod p) ≠ (h : ZMod p)⁻¹ * ((N : ZMod p) + 2 * (ε : ZMod p))

/-! ## Sizes -/

/-- For `q ≥ 7`, `q# ≥ 210`. -/
theorem primorial_ge_210 {q : ℕ} (hq : 7 ≤ q) : 210 ≤ primorial q :=
  primorial_seven ▸ primorial_mono hq

/-- For `q ≥ 5`, `ρ = 15 M + 1`. -/
theorem rho_eq {q : ℕ} (hq : 5 ≤ q) : rho q = 15 * mirrorM q + 1 := by
  have h := thirty_mul_mirrorM hq
  unfold rho
  omega

/-- For `q ≥ 7`, `M ≥ 7`. -/
theorem seven_le_mirrorM {q : ℕ} (hq : 7 ≤ q) : 7 ≤ mirrorM q := by
  have h := thirty_mul_mirrorM (by omega : 5 ≤ q)
  have h2 := primorial_ge_210 hq
  omega

/-- **Growth lemma.** For every prime `p` with `7 ≤ p ≤ q`, `2 p² < q#` (Bertrand). -/
theorem two_mul_sq_lt_primorial {p q : ℕ} (hp : p.Prime) (hp7 : 7 ≤ p) (hpq : p ≤ q) :
    2 * p ^ 2 < primorial q := by
  obtain ⟨r, hr, hnr, hr2⟩ := Nat.exists_prime_lt_and_le_two_mul (p / 2) (by omega)
  have hodd : p % 2 = 1 := Nat.odd_iff.mp (hp.odd_of_ne_two (by omega))
  have hrp : r < p := by omega
  have hr4 : 4 ≤ r := by omega
  have h2 : 2 ∣ primorial q := (Nat.prime_two.dvd_primorial_iff).mpr (by omega)
  have h3 : 3 ∣ primorial q := (Nat.prime_three.dvd_primorial_iff).mpr (by omega)
  have hrd : r ∣ primorial q := (hr.dvd_primorial_iff).mpr (by omega)
  have hpd : p ∣ primorial q := (hp.dvd_primorial_iff).mpr hpq
  have c2r : Nat.Coprime 2 r := (Nat.coprime_primes Nat.prime_two hr).mpr (by omega)
  have c3r : Nat.Coprime 3 r := (Nat.coprime_primes Nat.prime_three hr).mpr (by omega)
  have c2p : Nat.Coprime 2 p := (Nat.coprime_primes Nat.prime_two hp).mpr (by omega)
  have c3p : Nat.Coprime 3 p := (Nat.coprime_primes Nat.prime_three hp).mpr (by omega)
  have crp : Nat.Coprime r p := (Nat.coprime_primes hr hp).mpr (by omega)
  have h6 : 2 * 3 ∣ primorial q := Nat.Coprime.mul_dvd_of_dvd_of_dvd (by norm_num) h2 h3
  have h6r : 2 * 3 * r ∣ primorial q :=
    Nat.Coprime.mul_dvd_of_dvd_of_dvd (Nat.Coprime.mul_left c2r c3r) h6 hrd
  have c6rp : Nat.Coprime (2 * 3 * r) p :=
    Nat.Coprime.mul_left (Nat.Coprime.mul_left c2p c3p) crp
  have h6rp : 2 * 3 * r * p ∣ primorial q := Nat.Coprime.mul_dvd_of_dvd_of_dvd c6rp h6r hpd
  have hle := Nat.le_of_dvd (primorial_pos q) h6rp
  have hr' : p + 1 ≤ 2 * r := by omega
  have hmul : (p + 1) * p ≤ 2 * r * p := Nat.mul_le_mul_right p hr'
  nlinarith

/-! ## The cut-offs `Q` and `Q*` -/

/-- A prime `p` is at most `Q` exactly when `p² ≤ ρ`. -/
theorem le_cutLo_iff {q p : ℕ} (hp : p.Prime) : p ≤ cutLo q ↔ p ^ 2 ≤ rho q := by
  unfold cutLo
  constructor
  · intro h
    exact Nat.le_sqrt'.mp (le_trans h (Nat.findGreatest_le _))
  · intro h
    exact Nat.le_findGreatest (Nat.le_sqrt'.mpr h) hp

/-- A prime `p` is at most `Q*` exactly when `p² ≤ q#`. -/
theorem le_cutHi_iff {q p : ℕ} (hp : p.Prime) : p ≤ cutHi q ↔ p ^ 2 ≤ primorial q := by
  unfold cutHi
  constructor
  · intro h
    exact Nat.le_sqrt'.mp (le_trans h (Nat.findGreatest_le _))
  · intro h
    exact Nat.le_findGreatest (Nat.le_sqrt'.mpr h) hp

/-- For `q ≥ 7`, `7 ≤ Q`. -/
theorem seven_le_cutLo {q : ℕ} (hq : 7 ≤ q) : 7 ≤ cutLo q := by
  have h := primorial_ge_210 hq
  exact (le_cutLo_iff (by norm_num)).mpr (by unfold rho; omega)

/-- For `q ≥ 7`, `Q` is prime. -/
theorem cutLo_prime {q : ℕ} (hq : 7 ≤ q) : (cutLo q).Prime := by
  have h := primorial_ge_210 hq
  exact Nat.findGreatest_spec (m := 7) (Nat.le_sqrt'.mpr (by unfold rho; omega)) (by norm_num)

/-- `Q² ≤ ρ`. -/
theorem cutLo_sq_le (q : ℕ) : cutLo q ^ 2 ≤ rho q :=
  Nat.le_sqrt'.mp (Nat.findGreatest_le _)

/-- `Q*² ≤ q#`. -/
theorem cutHi_sq_le (q : ℕ) : cutHi q ^ 2 ≤ primorial q :=
  Nat.le_sqrt'.mp (Nat.findGreatest_le _)

/-- For `q ≥ 7`, `Q ≤ Q*`. -/
theorem cutLo_le_cutHi {q : ℕ} (hq : 7 ≤ q) : cutLo q ≤ cutHi q := by
  have h := primorial_ge_210 hq
  have h1 := cutLo_sq_le q
  refine (le_cutHi_iff (cutLo_prime hq)).mpr ?_
  unfold rho at h1
  omega

/-- Every prime `p` with `7 ≤ p ≤ q` is at most `Q`. -/
theorem le_cutLo_of_le {p q : ℕ} (hp : p.Prime) (hp7 : 7 ≤ p) (hpq : p ≤ q) : p ≤ cutLo q := by
  have h := two_mul_sq_lt_primorial hp hp7 hpq
  refine (le_cutLo_iff hp).mpr ?_
  unfold rho
  generalize p ^ 2 = s at h ⊢
  omega

/-- Every prime `p` with `7 ≤ p ≤ q` is at most `Q*`. -/
theorem le_cutHi_of_le {p q : ℕ} (hp : p.Prime) (hp7 : 7 ≤ p) (hpq : p ≤ q) : p ≤ cutHi q := by
  have h := two_mul_sq_lt_primorial hp hp7 hpq
  refine (le_cutHi_iff hp).mpr ?_
  generalize p ^ 2 = s at h ⊢
  omega

/-- A band gear is at least `7` (`q ≥ 7`). -/
theorem band_seven_le {q h : ℕ} (hq : 7 ≤ q) (hB : Band q h) : 7 ≤ h := by
  have := seven_le_cutLo hq
  have := hB.2.1
  omega

/-- A band gear `h` has `ρ < h²` (it is above `Q`). -/
theorem band_sq_gt {q h : ℕ} (hh : h.Prime) (hQh : cutLo q < h) : rho q < h ^ 2 := by
  by_contra hle
  push Not at hle
  have := (le_cutLo_iff hh).mpr hle
  omega

/-- A prime `p ≤ Q` does not divide a prime `h > Q`. -/
theorem not_dvd_of_le_cutLo_lt {q p h : ℕ} (hp : p.Prime) (hh : h.Prime) (hpQ : p ≤ cutLo q)
    (hQh : cutLo q < h) : ¬ p ∣ h := by
  intro hd
  have := (Nat.prime_dvd_prime_iff_eq hp hh).mp hd
  omega

/-- A prime `h ≥ 7` is coprime to `30`. -/
theorem coprime_thirty {h : ℕ} (hh : h.Prime) (hh7 : 7 ≤ h) : Nat.Coprime h 30 :=
  (Nat.Prime.coprime_iff_not_dvd hh).mpr fun hd =>
    thirty_ne_zero hh hh7 ((ZMod.natCast_eq_zero_iff 30 h).mpr hd)

/-! ## Clear sets -/

/-- `S_Y ⊆ S_X` when `X ≤ Y`. -/
theorem clear_mono {X Y j : ℕ} (hXY : X ≤ Y) (h : Clear Y j) : Clear X j :=
  fun p hp hp7 hpX => h p hp hp7 (le_trans hpX hXY)

/-- `S_Q ⊆ S_q`. -/
theorem clear_q_of_clear_cutLo {q j : ℕ} (h : Clear (cutLo q) j) : Clear q j :=
  fun p hp hp7 hpq => h p hp hp7 (le_cutLo_of_le hp hp7 hpq)

/-- `S_{Q*} ⊆ S_q`. -/
theorem clear_q_of_clear_cutHi {q j : ℕ} (h : Clear (cutHi q) j) : Clear q j :=
  fun p hp hp7 hpq => h p hp hp7 (le_cutHi_of_le hp hp7 hpq)

/-- `S_{Q*} ⊆ S_Q` (`q ≥ 7`). -/
theorem clear_cutLo_of_clear_cutHi {q j : ℕ} (hq : 7 ≤ q) (h : Clear (cutHi q) j) :
    Clear (cutLo q) j :=
  clear_mono (cutLo_le_cutHi hq) h

/-- A divisor of a leg strikes the copy. -/
theorem strikes_of_dvd_leg {m j L : ℕ} (hL : L = 30 * j - 1 ∨ L = 30 * j + 1) (hd : m ∣ L) :
    StrikesCopy m j := by
  unfold StrikesCopy
  rcases hL with rfl | rfl
  · exact Dvd.dvd.mul_right hd _
  · exact Dvd.dvd.mul_left hd _

/-- A prime dividing a leg of copy `j ≥ 1` is at least `7` (legs are `±1 (mod 30)`). -/
theorem prime_dvd_leg_ge_seven {m j L : ℕ} (hm : m.Prime) (hj : 1 ≤ j)
    (hL : L = 30 * j - 1 ∨ L = 30 * j + 1) (hd : m ∣ L) : 7 ≤ m := by
  by_contra h7
  push Not at h7
  have h2 := hm.two_le
  interval_cases m <;> omega

/-! ## The T-rule on one copy -/

/-- A twin copy whose lower leg is above `X` is in `S_X`. -/
theorem clear_of_twin {X j : ℕ} (ht : Twin j) (h : X < 30 * j - 1) : Clear X j := by
  intro p hp _ hpX hs
  rcases (strikesCopy_iff_leg hp).mp hs with h1 | h1
  · have := (Nat.prime_dvd_prime_iff_eq hp ht.1).mp h1
    omega
  · have := (Nat.prime_dvd_prime_iff_eq hp ht.2).mp h1
    omega

/-- A twin copy (`j ≥ 1`) in `S_X` has its lower leg above `X`. -/
theorem lt_of_twin_clear {X j : ℕ} (hj : 1 ≤ j) (ht : Twin j) (hc : Clear X j) :
    X < 30 * j - 1 := by
  by_contra h
  push Not at h
  exact hc (30 * j - 1) ht.1 (by omega) h ((strikesCopy_iff_leg ht.1).mpr (Or.inl dvd_rfl))

/-- A leg `L` of copy `j ≥ 1` is prime when copy `j` is in `S_X` and every prime `m` with
`m² ≤ L` is at most `X`. -/
theorem leg_prime_of_clear {X j L : ℕ} (hj : 1 ≤ j) (hL : L = 30 * j - 1 ∨ L = 30 * j + 1)
    (hX : ∀ m, m.Prime → m ^ 2 ≤ L → m ≤ X) (hc : Clear X j) : L.Prime := by
  by_contra hnp
  have hL1 : L ≠ 1 := by omega
  have hm : L.minFac.Prime := Nat.minFac_prime hL1
  have hmL : L.minFac ∣ L := Nat.minFac_dvd L
  have hsq : L.minFac ^ 2 ≤ L := Nat.minFac_sq_le_self (by omega) hnp
  exact hc _ hm (prime_dvd_leg_ge_seven hm hj hL hmL) (hX _ hm hsq) (strikes_of_dvd_leg hL hmL)

/-- Copy `j ≥ 1` in `S_X` is a twin copy when every prime `m` with `m² ≤ 30 j + 1` is at most
`X`. -/
theorem twin_of_clear {X j : ℕ} (hj : 1 ≤ j) (hX : ∀ m, m.Prime → m ^ 2 ≤ 30 * j + 1 → m ≤ X)
    (hc : Clear X j) : Twin j :=
  ⟨leg_prime_of_clear hj (Or.inl rfl) (fun m hm h => hX m hm (by omega)) hc,
    leg_prime_of_clear hj (Or.inr rfl) hX hc⟩

/-! ## The two halves -/

/-- The range `[1, M - 1]` is the union of the two halves (`q ≥ 5`). -/
theorem range_iff_halves {q j : ℕ} (hq : 5 ≤ q) :
    (1 ≤ j ∧ j ≤ mirrorM q - 1) ↔ LowHalf q j ∨ HighHalf q j := by
  obtain ⟨k, hk⟩ := odd_mirrorM hq
  unfold LowHalf HighHalf
  omega

/-- The two halves are disjoint. -/
theorem lowHalf_not_highHalf {q j : ℕ} (hq : 5 ≤ q) (hl : LowHalf q j) : ¬ HighHalf q j := by
  obtain ⟨k, hk⟩ := odd_mirrorM hq
  unfold LowHalf at hl
  unfold HighHalf
  omega

/-- A high copy has `j ≥ 1` (`q ≥ 7`). -/
theorem highHalf_one_le {q j : ℕ} (hq : 7 ≤ q) (h : HighHalf q j) : 1 ≤ j := by
  have := seven_le_mirrorM hq
  unfold HighHalf at h
  omega

/-- Low legs are below `ρ`. -/
theorem lowHalf_leg_lt {q j : ℕ} (hq : 5 ≤ q) (h : LowHalf q j) : 30 * j + 1 < rho q := by
  obtain ⟨k, hk⟩ := odd_mirrorM hq
  have hr := rho_eq hq
  unfold LowHalf at h
  omega

/-- High legs lie in `[15 M + 14, q# - 29]`. -/
theorem highHalf_legs {q j : ℕ} (hq : 5 ≤ q) (h : HighHalf q j) :
    15 * mirrorM q + 14 ≤ 30 * j - 1 ∧ 30 * j + 1 + 29 ≤ primorial q := by
  obtain ⟨k, hk⟩ := odd_mirrorM hq
  have h30 := thirty_mul_mirrorM hq
  unfold HighHalf at h
  omega

/-- The mirror sends the high half into the low half. -/
theorem highHalf_mirror {q y : ℕ} (hq : 5 ≤ q) (h : HighHalf q y) :
    LowHalf q (mirrorM q - y) := by
  obtain ⟨k, hk⟩ := odd_mirrorM hq
  unfold HighHalf at h
  unfold LowHalf
  omega

/-- `Loss_q` lies in the low half. -/
theorem loss_lowHalf {q j : ℕ} (hq : 7 ≤ q) (h : Loss q j) : LowHalf q j := by
  obtain ⟨hj, _, hQ, _⟩ := h
  have hsq : (30 * j - 1) ^ 2 ≤ rho q :=
    le_trans (Nat.pow_le_pow_left hQ 2) (cutLo_sq_le q)
  have h29 : 29 * (30 * j - 1) ≤ (30 * j - 1) ^ 2 := by
    rw [pow_two]
    exact Nat.mul_le_mul_right _ (by omega)
  have hr := rho_eq (by omega : 5 ≤ q)
  obtain ⟨k, hk⟩ := odd_mirrorM (by omega : 5 ≤ q)
  unfold LowHalf
  generalize (30 * j - 1) ^ 2 = s at hsq h29
  omega

/-- `Loss_q ∩ S_Q = ∅`: the lower leg of a `Loss_q` copy is a prime in `[7, Q]`. -/
theorem loss_not_clear_cutLo {q j : ℕ} (h : Loss q j) : ¬ Clear (cutLo q) j := by
  obtain ⟨hj, _, hQ, ht⟩ := h
  intro hc
  have := lt_of_twin_clear hj ht hc
  omega

/-! ## The low rule, the high rule and the height split -/

/-- **Low rule.** For a low copy: revealed `⇔ Loss_q ∨ S_Q`. -/
theorem revealed_low_iff {q j : ℕ} (hq : 7 ≤ q) (hlow : LowHalf q j) :
    Revealed q j ↔ Loss q j ∨ Clear (cutLo q) j := by
  have hjM : 1 ≤ j ∧ j ≤ mirrorM q - 1 := (range_iff_halves (by omega)).mpr (Or.inl hlow)
  have hleg := lowHalf_leg_lt (by omega) hlow
  constructor
  · rintro ⟨hj, _, hc, ht⟩
    have hqj := lt_of_twin_clear hj ht hc
    by_cases hQ : 30 * j - 1 ≤ cutLo q
    · exact Or.inl ⟨hj, hqj, hQ, ht⟩
    · exact Or.inr (clear_of_twin ht (by omega))
  · rintro (⟨hj, hqj, _, ht⟩ | hc)
    · exact ⟨hj, hjM.2, clear_of_twin ht hqj, ht⟩
    · have ht : Twin j := twin_of_clear hjM.1
        (fun m hm hm2 => (le_cutLo_iff hm).mpr (by omega)) hc
      exact ⟨hjM.1, hjM.2, clear_q_of_clear_cutLo hc, ht⟩

/-- **High rule.** For a high copy: revealed `⇔ S_{Q*}`. -/
theorem revealed_high_iff {q j : ℕ} (hq : 7 ≤ q) (hhigh : HighHalf q j) :
    Revealed q j ↔ Clear (cutHi q) j := by
  have hjM : 1 ≤ j ∧ j ≤ mirrorM q - 1 := (range_iff_halves (by omega)).mpr (Or.inr hhigh)
  obtain ⟨hlo, hhi⟩ := highHalf_legs (by omega) hhigh
  have h30 := thirty_mul_mirrorM (by omega : 5 ≤ q)
  constructor
  · rintro ⟨_, _, _, ht⟩
    apply clear_of_twin ht
    by_contra h
    push Not at h
    have h1 : (30 * j - 1) ^ 2 ≤ primorial q :=
      le_trans (Nat.pow_le_pow_left h 2) (cutHi_sq_le q)
    have h2 : 2 * (30 * j - 1) ≤ (30 * j - 1) ^ 2 := by
      rw [pow_two]
      exact Nat.mul_le_mul_right _ (by omega)
    generalize (30 * j - 1) ^ 2 = s at h1 h2
    omega
  · intro hc
    have ht : Twin j := twin_of_clear hjM.1
      (fun m hm hm2 => (le_cutHi_iff hm).mpr (by omega)) hc
    exact ⟨hjM.1, hjM.2, clear_q_of_clear_cutHi hc, ht⟩

/-- **Height split.** Revealed `⇔ Loss_q ∨ (S_Q ∩ low) ∨ (S_{Q*} ∩ high)`. -/
theorem revealed_iff_split {q : ℕ} (hq : 7 ≤ q) (j : ℕ) :
    Revealed q j ↔
      Loss q j ∨ (Clear (cutLo q) j ∧ LowHalf q j) ∨ (Clear (cutHi q) j ∧ HighHalf q j) := by
  constructor
  · intro h
    rcases (range_iff_halves (by omega)).mp ⟨h.1, h.2.1⟩ with hl | hh
    · rcases (revealed_low_iff hq hl).mp h with h' | h'
      · exact Or.inl h'
      · exact Or.inr (Or.inl ⟨h', hl⟩)
    · exact Or.inr (Or.inr ⟨(revealed_high_iff hq hh).mp h, hh⟩)
  · rintro (h | ⟨hc, hl⟩ | ⟨hc, hh⟩)
    · exact (revealed_low_iff hq (loss_lowHalf hq h)).mpr (Or.inl h)
    · exact (revealed_low_iff hq hl).mpr (Or.inr hc)
    · exact (revealed_high_iff hq hh).mpr hc

/-! ## The mirror and the `U⁻` strikes -/

/-- If `p ∣ N` and `a + b = N`, then `p ∣ a ↔ p ∣ b`. -/
theorem split_dvd_iff_of_add_eq {p a b N : ℕ} (hN : p ∣ N) (h : a + b = N) :
    p ∣ a ↔ p ∣ b := by
  subst h
  exact ⟨fun ha => (Nat.dvd_add_right ha).mp hN, fun hb => (Nat.dvd_add_left hb).mp hN⟩

/-- For `q ≥ 5`, a prime `p ≤ q` and `1 ≤ x < M`: `p` strikes copy `M - x` iff it strikes
copy `x`. -/
theorem strikes_mirror {q p x : ℕ} (hq : 5 ≤ q) (hp : p.Prime) (hpq : p ≤ q)
    (hx1 : 1 ≤ x) (hx : x < mirrorM q) :
    StrikesCopy p (mirrorM q - x) ↔ StrikesCopy p x := by
  have hM := thirty_mul_mirrorM hq
  have hpd : p ∣ primorial q := (hp.dvd_primorial_iff).mpr hpq
  have e1 : (30 * (mirrorM q - x) - 1) + (30 * x + 1) = primorial q := by omega
  have e2 : (30 * (mirrorM q - x) + 1) + (30 * x - 1) = primorial q := by omega
  rw [strikesCopy_iff_leg hp, strikesCopy_iff_leg hp, split_dvd_iff_of_add_eq hpd e1,
    split_dvd_iff_of_add_eq hpd e2]
  exact Or.comm

/-- `S_q` is mirror-invariant: for `q ≥ 5` and `1 ≤ x < M`, `M - x ∈ S_q ⇔ x ∈ S_q`. -/
theorem clear_mirror {q x : ℕ} (hq : 5 ≤ q) (hx1 : 1 ≤ x) (hx : x < mirrorM q) :
    Clear q (mirrorM q - x) ↔ Clear q x := by
  constructor
  · intro h p hp hp7 hpq hs
    exact h p hp hp7 hpq ((strikes_mirror hq hp hpq hx1 hx).mpr hs)
  · intro h p hp hp7 hpq hs
    exact h p hp hp7 hpq ((strikes_mirror hq hp hpq hx1 hx).mp hs)

/-- `S_Q = S_q ∖ (U⁻-struck copies)` for `q ≥ 7`. -/
theorem clear_cutLo_iff {q j : ℕ} (hq : 7 ≤ q) :
    Clear (cutLo q) j ↔ Clear q j ∧ ∀ p, Uminus q p → ¬ StrikesCopy p j := by
  constructor
  · intro hc
    exact ⟨clear_q_of_clear_cutLo hc, fun p ⟨hp, hqp, hpQ⟩ => hc p hp (by omega) hpQ⟩
  · rintro ⟨hc, hu⟩ p hp hp7 hpQ
    by_cases hpq : p ≤ q
    · exact hc p hp hp7 hpq
    · exact hu p ⟨hp, by omega, hpQ⟩

/-- **Strike in residue form.** For a prime `p ≥ 7` and `k ≥ 1`: `p` strikes copy `k` iff
`k ≡ ±a_p (mod p)`. -/
theorem strikes_iff_gearRes {p k : ℕ} (hp : p.Prime) (hp7 : 7 ≤ p) (hk : 1 ≤ k) :
    StrikesCopy p k ↔ ((k : ZMod p) = gearRes p ∨ (k : ZMod p) = -gearRes p) := by
  have := Fact.mk hp
  have h30 : (30 : ZMod p) ≠ 0 := by exact_mod_cast thirty_ne_zero hp hp7
  rw [strikesCopy_iff_zmod hk, mul_eq_zero, sub_eq_zero, add_eq_zero_iff_eq_neg]
  unfold gearRes
  rw [← eq_inv_mul_iff_mul_eq₀ h30, ← eq_inv_mul_iff_mul_eq₀ h30, mul_one, mul_neg_one]

/-! ## (i) The gain set -/

/-- **(i), strike form.** `G_q = S_Q ∩ high ∩ {j : some h ∈ B strikes j}` (`q ≥ 7`). -/
theorem gain_iff_band_strike {q j : ℕ} (hq : 7 ≤ q) :
    GainCopy q j ↔ HighHalf q j ∧ Clear (cutLo q) j ∧ ∃ h, Band q h ∧ StrikesCopy h j := by
  unfold GainCopy
  refine and_congr_right fun _ => and_congr_right fun hc => ?_
  constructor
  · intro hn
    unfold Clear at hn
    push Not at hn
    obtain ⟨p, hp, hp7, hpQ, hs⟩ := hn
    refine ⟨p, ⟨hp, ?_, hpQ⟩, hs⟩
    by_contra hle
    push Not at hle
    exact hc p hp hp7 hle hs
  · rintro ⟨h, hB, hs⟩ hc'
    exact hc' h hB.1 (band_seven_le hq hB) hB.2.2 hs

/-- **(i), residue form.** `G_q = S_Q ∩ high ∩ ⋃_{h ∈ B} {j ≡ ±a_h (mod h)}` (`q ≥ 7`). -/
theorem gain_iff_residue {q j : ℕ} (hq : 7 ≤ q) :
    GainCopy q j ↔ HighHalf q j ∧ Clear (cutLo q) j ∧
      ∃ h, Band q h ∧ ((j : ZMod h) = gearRes h ∨ (j : ZMod h) = -gearRes h) := by
  rw [gain_iff_band_strike hq]
  refine and_congr_right fun hh => and_congr_right fun _ =>
    exists_congr fun h => and_congr_right fun hB => ?_
  exact strikes_iff_gearRes hB.1 (band_seven_le hq hB) (highHalf_one_le hq hh)

/-- **(i), revealed set.** `Revealed_q = Loss_q ∪ ((S_Q ∩ [1, M - 1]) \ G_q)` (`q ≥ 7`); the
union is disjoint by `loss_not_clear_cutLo`. -/
theorem revealed_iff_gain {q : ℕ} (hq : 7 ≤ q) (j : ℕ) :
    Revealed q j ↔ Loss q j ∨
      ((Clear (cutLo q) j ∧ 1 ≤ j ∧ j ≤ mirrorM q - 1) ∧ ¬ GainCopy q j) := by
  have hq5 : 5 ≤ q := by omega
  rw [revealed_iff_split hq j]
  constructor
  · rintro (hL | ⟨hc, hl⟩ | ⟨hc, hh⟩)
    · exact Or.inl hL
    · exact Or.inr ⟨⟨hc, (range_iff_halves hq5).mpr (Or.inl hl)⟩,
        fun hg => lowHalf_not_highHalf hq5 hl hg.1⟩
    · exact Or.inr ⟨⟨clear_cutLo_of_clear_cutHi hq hc,
        (range_iff_halves hq5).mpr (Or.inr hh)⟩, fun hg => hg.2.2 hc⟩
  · rintro (hL | ⟨⟨hc, hr⟩, hng⟩)
    · exact Or.inl hL
    · rcases (range_iff_halves hq5).mp hr with hl | hh
      · exact Or.inr (Or.inl ⟨hc, hl⟩)
      · refine Or.inr (Or.inr ⟨?_, hh⟩)
        by_contra hn
        exact hng ⟨hh, hc, hn⟩

/-! ## (ii) Composite legs of high `S_Q` copies -/

/-- **(ii), factor form.** For `q ≥ 7`, a high copy `j ∈ S_Q` and a composite leg `L` of it:
`h = minFac L` is in `B`, `c = L / h` is prime, `h ≤ c < 2h` and `L = h c`. -/
theorem high_leg_factor {q j L : ℕ} (hq : 7 ≤ q) (hh : HighHalf q j) (hc : Clear (cutLo q) j)
    (hL : L = 30 * j - 1 ∨ L = 30 * j + 1) (hnp : ¬ L.Prime) :
    Band q L.minFac ∧ (L / L.minFac).Prime ∧ L.minFac ≤ L / L.minFac ∧
      L / L.minFac < 2 * L.minFac ∧ L = L.minFac * (L / L.minFac) := by
  have hq5 : 5 ≤ q := by omega
  obtain ⟨hlo, hhi⟩ := highHalf_legs hq5 hh
  have hr := rho_eq hq5
  have h30 := thirty_mul_mirrorM hq5
  have hj1 : 1 ≤ j := highHalf_one_le hq hh
  have hLhi : L < primorial q := by omega
  have hLlo : rho q < L := by omega
  have hL1 : L ≠ 1 := by omega
  set m := L.minFac with hm
  have hmp : m.Prime := Nat.minFac_prime hL1
  have hmL : m ∣ L := Nat.minFac_dvd L
  have hm7 : 7 ≤ m := prime_dvd_leg_ge_seven hmp hj1 hL hmL
  have hQm : cutLo q < m := by
    by_contra hle
    push Not at hle
    exact hc m hmp hm7 hle (strikes_of_dvd_leg hL hmL)
  have hsq : rho q < m ^ 2 := band_sq_gt hmp hQm
  obtain ⟨c, hcL⟩ := hmL
  have hdiv : L / m = c := by
    rw [hcL]
    exact Nat.mul_div_cancel_left c hmp.pos
  rw [hdiv]
  have hc1 : c ≠ 1 := by
    rintro rfl
    rw [mul_one] at hcL
    exact hnp (hcL ▸ hmp)
  have hc0 : 0 < c := by
    rcases Nat.eq_zero_or_pos c with h0 | h0
    · rw [h0, mul_zero] at hcL
      omega
    · exact h0
  have hcm : c.minFac.Prime := Nat.minFac_prime hc1
  have hmcm : m ≤ c.minFac :=
    Nat.minFac_le_of_dvd hcm.two_le (dvd_trans (Nat.minFac_dvd c) (Dvd.intro_left m hcL.symm))
  have hmc : m ≤ c := le_trans hmcm (Nat.minFac_le hc0)
  have hcp : c.Prime := by
    by_contra hcnp
    have h1 : c.minFac ^ 2 ≤ c := Nat.minFac_sq_le_self hc0 hcnp
    have h2 : m ^ 2 ≤ c := le_trans (Nat.pow_le_pow_left hmcm 2) h1
    have h3 : m * m ^ 2 ≤ L := by
      rw [hcL]
      exact Nat.mul_le_mul_left m h2
    have h4 : 7 * m ^ 2 ≤ m * m ^ 2 := Nat.mul_le_mul_right _ hm7
    generalize m * m ^ 2 = t at h3 h4
    generalize m ^ 2 = s at h3 h4 hsq
    omega
  have hmcL : m * m ≤ L := by
    rw [hcL]
    exact Nat.mul_le_mul_left m hmc
  have hlt : c < 2 * m := by
    have h1 : m * c < m * (2 * m) := by
      have e : m * (2 * m) = 2 * (m * m) := by ring
      rw [e, ← hcL]
      rw [pow_two] at hsq
      generalize m * m = s at hsq ⊢
      omega
    exact Nat.lt_of_mul_lt_mul_left h1
  have hmQs : m ≤ cutHi q := by
    refine (le_cutHi_iff hmp).mpr ?_
    rw [pow_two]
    omega
  exact ⟨⟨hmp, hQm, hmQs⟩, hcp, hmc, hlt, hcL⟩

/-- **(ii).** For `q ≥ 7`, every composite leg `L` of a copy `j ∈ S_Q ∩ high` is `L = h c` with
`h ∈ B`, `c` prime, `h ≤ c < 2h`; `h` acts on `j`, and `c` does not act on `j` when `c > h`. -/
theorem high_leg_two_primes {q j L : ℕ} (hq : 7 ≤ q) (hh : HighHalf q j)
    (hc : Clear (cutLo q) j) (hL : L = 30 * j - 1 ∨ L = 30 * j + 1) (hnp : ¬ L.Prime) :
    ∃ h c, Band q h ∧ c.Prime ∧ h ≤ c ∧ c < 2 * h ∧ L = h * c ∧ Acts h j ∧
      (h < c → ¬ Acts c j) := by
  obtain ⟨hB, hcp, hle, hlt, hL'⟩ := high_leg_factor hq hh hc hL hnp
  generalize L / L.minFac = c at hcp hle hlt hL'
  generalize L.minFac = h at hB hcp hle hlt hL'
  have hh7 := band_seven_le hq hB
  refine ⟨h, c, hB, hcp, hle, hlt, hL', ?_, ?_⟩
  · unfold Acts
    have h1 : h ^ 2 ≤ h * c := by
      rw [pow_two]
      exact Nat.mul_le_mul_left h hle
    rw [← hL'] at h1
    omega
  · intro hhc hact
    unfold Acts at hact
    have h1 : c * (h + 1) ≤ c ^ 2 := by
      rw [pow_two]
      exact Nat.mul_le_mul_left c hhc
    have e : c * (h + 1) = h * c + c := by ring
    rw [e, ← hL'] at h1
    generalize c ^ 2 = s at h1 hact
    omega

/-- A prime of `B` dividing a leg of a high copy makes the leg composite (`q ≥ 7`). -/
theorem band_dvd_high_leg_not_prime {q j L h : ℕ} (hq : 7 ≤ q) (hh : HighHalf q j)
    (hL : L = 30 * j - 1 ∨ L = 30 * j + 1) (hB : Band q h) (hd : h ∣ L) : ¬ L.Prime := by
  intro hLp
  have heq := (Nat.prime_dvd_prime_iff_eq hB.1 hLp).mp hd
  have h2 := (le_cutHi_iff hB.1).mp hB.2.2
  obtain ⟨hlo, hhi⟩ := highHalf_legs (by omega) hh
  have h30 := thirty_mul_mirrorM (by omega : 5 ≤ q)
  subst heq
  have h3 : 2 * h ≤ h ^ 2 := by
    rw [pow_two]
    exact Nat.mul_le_mul_right h (by omega)
  generalize h ^ 2 = s at h2 h3
  omega

/-- **(ii), leg law.** On a copy `j ∈ S_Q ∩ high` (`q ≥ 7`), a leg is composite iff some
`h ∈ B` divides it. -/
theorem high_leg_band_iff {q j L : ℕ} (hq : 7 ≤ q) (hh : HighHalf q j)
    (hc : Clear (cutLo q) j) (hL : L = 30 * j - 1 ∨ L = 30 * j + 1) :
    (∃ h, Band q h ∧ h ∣ L) ↔ ¬ L.Prime := by
  constructor
  · rintro ⟨h, hB, hd⟩
    exact band_dvd_high_leg_not_prime hq hh hL hB hd
  · intro hnp
    exact ⟨_, (high_leg_factor hq hh hc hL hnp).1, Nat.minFac_dvd L⟩

/-! ## (iii) The pair form -/

/-- **Direct classes.** If `30 j + ε = h c` (`ε = ±1`, `j ≥ 1`) and `p` is a prime not dividing
`h`, then `p` strikes copy `j` iff `c ≡ 0` or `c ≡ 2 ε h⁻¹ (mod p)`. -/
theorem strike_iff_pair_class {p j h c : ℕ} {ε : ℤ} (hp : p.Prime) (hj : 1 ≤ j)
    (hε : ε = 1 ∨ ε = -1) (hph : ¬ p ∣ h) (hid : (30 * j + ε : ℤ) = h * c) :
    StrikesCopy p j ↔
      ((c : ZMod p) = 0 ∨ (c : ZMod p) = 2 * (ε : ZMod p) * (h : ZMod p)⁻¹) := by
  have := Fact.mk hp
  have hh0 : (h : ZMod p) ≠ 0 := by
    rw [Ne, ZMod.natCast_eq_zero_iff]
    exact hph
  have hε2 : (ε : ZMod p) ^ 2 = 1 := by
    rcases hε with rfl | rfl <;> simp
  have hid' : 30 * (j : ZMod p) = (h : ZMod p) * c - ε := by
    have := congrArg (Int.cast : ℤ → ZMod p) hid
    push_cast at this
    linear_combination this
  have hinv : (h : ZMod p) * (h : ZMod p)⁻¹ = 1 := mul_inv_cancel₀ hh0
  rw [strikesCopy_iff_zmod hj]
  have key : (30 * (j : ZMod p) - 1) * (30 * (j : ZMod p) + 1) =
      ((h : ZMod p) * c) * ((h : ZMod p) * c - 2 * ε) := by
    rw [hid']
    linear_combination hε2
  rw [key, mul_eq_zero]
  constructor
  · rintro (h1 | h1)
    · exact Or.inl ((mul_eq_zero.mp h1).resolve_left hh0)
    · right
      rw [eq_mul_inv_iff_mul_eq₀ hh0]
      linear_combination h1
  · rintro (h1 | h1)
    · left
      rw [h1, mul_zero]
    · right
      rw [h1]
      linear_combination (2 * (ε : ZMod p)) * hinv

/-- **Mirror classes.** If `30 j + ε = h c` (`ε = ±1`, `j < M`, `q ≥ 5`) and `p` is a prime not
dividing `h`, then `p` strikes the mirror copy `M - j` iff `c ≡ h⁻¹ q#` or
`c ≡ h⁻¹ (q# + 2 ε) (mod p)`. -/
theorem mirror_strike_iff_pair_class {q p j h c : ℕ} {ε : ℤ} (hq : 5 ≤ q) (hp : p.Prime)
    (hjM : j < mirrorM q) (hε : ε = 1 ∨ ε = -1) (hph : ¬ p ∣ h)
    (hid : (30 * j + ε : ℤ) = h * c) :
    StrikesCopy p (mirrorM q - j) ↔
      ((c : ZMod p) = (h : ZMod p)⁻¹ * (primorial q : ZMod p) ∨
        (c : ZMod p) = (h : ZMod p)⁻¹ * ((primorial q : ZMod p) + 2 * (ε : ZMod p))) := by
  have := Fact.mk hp
  have hh0 : (h : ZMod p) ≠ 0 := by
    rw [Ne, ZMod.natCast_eq_zero_iff]
    exact hph
  have hε2 : (ε : ZMod p) ^ 2 = 1 := by
    rcases hε with rfl | rfl <;> simp
  have hid' : 30 * (j : ZMod p) = (h : ZMod p) * c - ε := by
    have := congrArg (Int.cast : ℤ → ZMod p) hid
    push_cast at this
    linear_combination this
  have hM : 30 * (mirrorM q : ZMod p) = (primorial q : ZMod p) := by
    exact_mod_cast congrArg (Nat.cast : ℕ → ZMod p) (thirty_mul_mirrorM hq)
  rw [strikesCopy_iff_zmod (by omega), Nat.cast_sub hjM.le]
  have key : (30 * ((mirrorM q : ZMod p) - (j : ZMod p)) - 1) *
      (30 * ((mirrorM q : ZMod p) - (j : ZMod p)) + 1) =
      ((primorial q : ZMod p) - (h : ZMod p) * c) *
        ((primorial q : ZMod p) + 2 * ε - (h : ZMod p) * c) := by
    have e : 30 * ((mirrorM q : ZMod p) - (j : ZMod p)) =
        (primorial q : ZMod p) - (h : ZMod p) * c + ε := by
      rw [mul_sub, hM, hid']
      ring
    rw [e]
    linear_combination hε2
  rw [key, mul_eq_zero, sub_eq_zero, sub_eq_zero, eq_inv_mul_iff_mul_eq₀ hh0,
    eq_inv_mul_iff_mul_eq₀ hh0]
  constructor
  · rintro (h1 | h1)
    · exact Or.inl h1.symm
    · exact Or.inr h1.symm
  · rintro (h1 | h1)
    · exact Or.inl h1.symm
    · exact Or.inr h1.symm

/-- From a pair, the leg identity `30 j + ε = h c` (`q ≥ 7`). -/
theorem pair_leg {q j h c : ℕ} {ε : ℤ} (hq : 7 ≤ q) (hp : GainPair q j h ε c) :
    (30 * j + ε : ℤ) = h * c := by
  obtain ⟨hB, -, -, -, -, h30, -, hj⟩ := hp
  have hinv : (h : ZMod 30) * (h : ZMod 30)⁻¹ = 1 :=
    ZMod.coe_mul_inv_eq_one h (coprime_thirty hB.1 (band_seven_le hq hB))
  have hz : (((h : ℤ) * c - ε : ℤ) : ZMod 30) = 0 := by
    push_cast
    rw [h30]
    linear_combination (ε : ZMod 30) * hinv
  have hd : (30 : ℤ) ∣ (h : ℤ) * c - ε := by
    have := (ZMod.intCast_zmod_eq_zero_iff_dvd _ 30).mp hz
    exact_mod_cast this
  generalize (h : ℤ) * c = X at hd hj ⊢
  omega

/-- A factorisation `30 j + ε = h c` of a leg of a high `S_Q` copy, with `h ∈ B`, `c` prime and
`h ≤ c`, is a pair of the pair form (`q ≥ 7`). -/
theorem pair_of_factor {q j h c : ℕ} {ε : ℤ} (hq : 7 ≤ q) (hh : HighHalf q j)
    (hc : Clear (cutLo q) j) (hB : Band q h) (hcp : c.Prime) (hhc : h ≤ c)
    (hε : ε = 1 ∨ ε = -1) (hid : (30 * j + ε : ℤ) = h * c) : GainPair q j h ε c := by
  have hq5 : 5 ≤ q := by omega
  have hh7 := band_seven_le hq hB
  have hj1 := highHalf_one_le hq hh
  obtain ⟨-, hhi⟩ := highHalf_legs hq5 hh
  refine ⟨hB, hε, hcp, hhc, ?_, ?_, ?_, ?_⟩
  · rw [Nat.le_div_iff_mul_le hB.1.pos]
    have hN : ((c * h : ℕ) : ℤ) = 30 * j + ε := by
      push_cast
      rw [hid]
      ring
    generalize c * h = N at hN ⊢
    rcases hε with rfl | rfl <;> omega
  · have hinv : (h : ZMod 30) * (h : ZMod 30)⁻¹ = 1 :=
      ZMod.coe_mul_inv_eq_one h (coprime_thirty hB.1 hh7)
    have h300 : (30 : ZMod 30) = 0 := by decide
    have hz : (h : ZMod 30) * c = ε := by
      have := congrArg (Int.cast : ℤ → ZMod 30) hid
      push_cast at this
      linear_combination -this + (j : ZMod 30) * h300
    calc (c : ZMod 30) = (h : ZMod 30) * c * (h : ZMod 30)⁻¹ := by
          linear_combination (-(c : ZMod 30)) * hinv
      _ = (ε : ZMod 30) * (h : ZMod 30)⁻¹ := by rw [hz]
  · intro p hp hp7 hpQ
    have hph : ¬ p ∣ h := not_dvd_of_le_cutLo_lt hp hB.1 hpQ hB.2.1
    have hns := hc p hp hp7 hpQ
    rw [strike_iff_pair_class hp hj1 hε hph hid] at hns
    push Not at hns
    exact hns
  · have e : (h : ℤ) * c - ε = 30 * j := by linarith
    rw [e]
    omega

/-- **(iii), pair form.** For `q ≥ 7`: `j ∈ G_q` iff `j = (h c - ε)/30` for some pair
`(h, ε, c)`: `h ∈ B`, `ε = ±1`, `c` prime, `h ≤ c ≤ (q# - 29)/h`, `c ≡ ε h⁻¹ (mod 30)`, and
`c ≢ 0`, `c ≢ 2 ε h⁻¹ (mod p)` for every prime `7 ≤ p ≤ Q`. -/
theorem gain_iff_pair {q j : ℕ} (hq : 7 ≤ q) :
    GainCopy q j ↔ ∃ h ε c, GainPair q j h ε c := by
  have hq5 : 5 ≤ q := by omega
  constructor
  · intro hg
    obtain ⟨hh, hc, g, hB, hs⟩ := (gain_iff_band_strike hq).mp hg
    have hj1 := highHalf_one_le hq hh
    rcases (strikesCopy_iff_leg hB.1).mp hs with hd | hd
    · have hnp := band_dvd_high_leg_not_prime hq hh (Or.inl rfl) hB hd
      obtain ⟨hB', hcp, hle, -, hL⟩ := high_leg_factor hq hh hc (Or.inl rfl) hnp
      refine ⟨_, -1, _, pair_of_factor hq hh hc hB' hcp hle (Or.inr rfl) ?_⟩
      rw [← Nat.cast_mul, ← hL]
      omega
    · have hnp := band_dvd_high_leg_not_prime hq hh (Or.inr rfl) hB hd
      obtain ⟨hB', hcp, hle, -, hL⟩ := high_leg_factor hq hh hc (Or.inr rfl) hnp
      refine ⟨_, 1, _, pair_of_factor hq hh hc hB' hcp hle (Or.inl rfl) ?_⟩
      rw [← Nat.cast_mul, ← hL]
      push_cast
      ring
  · rintro ⟨h, ε, c, hp⟩
    have hid := pair_leg hq hp
    obtain ⟨hB, hε, hcp, hhc, hcle, -, hpc, -⟩ := hp
    have hh7 := band_seven_le hq hB
    have hsq := band_sq_gt hB.1 hB.2.1
    have hr := rho_eq hq5
    have h30 := thirty_mul_mirrorM hq5
    obtain ⟨k, hk⟩ := odd_mirrorM hq5
    have hhc2 : h * h ≤ h * c := Nat.mul_le_mul_left h hhc
    rw [Nat.le_div_iff_mul_le hB.1.pos] at hcle
    have hN : (((h * c : ℕ) : ℤ)) = 30 * j + ε := by
      push_cast
      rw [hid]
    rw [pow_two] at hsq
    have hcomm : c * h = h * c := Nat.mul_comm c h
    -- the leg identity in `ℕ`
    have hleg : (ε = 1 ∧ 30 * j + 1 = h * c) ∨ (ε = -1 ∧ 1 ≤ j ∧ 30 * j - 1 = h * c) := by
      generalize h * c = N at hN hhc2 ⊢
      generalize h * h = s at hhc2 hsq
      rcases hε with rfl | rfl
      · left
        exact ⟨rfl, by omega⟩
      · right
        exact ⟨rfl, by omega⟩
    have hhigh : HighHalf q j := by
      unfold HighHalf
      generalize h * c = N at hleg hhc2 hcle hcomm
      generalize h * h = s at hhc2 hsq
      rw [hcomm] at hcle
      rcases hleg with ⟨_, hl⟩ | ⟨_, _, hl⟩ <;> omega
    have hj1 := highHalf_one_le hq hhigh
    have hcl : Clear (cutLo q) j := by
      intro p hp hp7 hpQ hs
      have hph : ¬ p ∣ h := not_dvd_of_le_cutLo_lt hp hB.1 hpQ hB.2.1
      rcases (strike_iff_pair_class hp hj1 hε hph hid).mp hs with h1 | h1
      · exact (hpc p hp hp7 hpQ).1 h1
      · exact (hpc p hp hp7 hpQ).2 h1
    refine ⟨hhigh, hcl, fun hcH => ?_⟩
    have hhs : StrikesCopy h j := by
      rcases hleg with ⟨_, hl⟩ | ⟨_, _, hl⟩
      · exact strikes_of_dvd_leg (Or.inr rfl) ⟨c, hl⟩
      · exact strikes_of_dvd_leg (Or.inl rfl) ⟨c, hl⟩
    exact hcH h hB.1 hh7 hB.2.2 hhs

/-- A product of two primes `h ≤ c` is composite, with least prime factor `h`. -/
theorem two_prime_product {h c : ℕ} (hh : h.Prime) (hc : c.Prime) (hhc : h ≤ c) :
    ¬ (h * c).Prime ∧ (h * c).minFac = h ∧ h * c / h = c := by
  refine ⟨Nat.not_prime_mul hh.ne_one hc.ne_one, ?_, Nat.mul_div_cancel_left c hh.pos⟩
  have h4 : 2 * 2 ≤ h * c := Nat.mul_le_mul hh.two_le hc.two_le
  have hmp : (h * c).minFac.Prime := Nat.minFac_prime (by omega)
  have hle : (h * c).minFac ≤ h := Nat.minFac_le_of_dvd hh.two_le (dvd_mul_right h c)
  rcases (Nat.Prime.dvd_mul hmp).mp (Nat.minFac_dvd _) with hd | hd
  · exact (Nat.prime_dvd_prime_iff_eq hmp hh).mp hd
  · have := (Nat.prime_dvd_prime_iff_eq hmp hc).mp hd
    omega

/-- **(iii), the pairs of one copy.** For `q ≥ 7` and a copy `j ∈ S_Q ∩ high`: `(h, ε, c)` is a
pair of `j` iff the leg `L = 30 j + ε` is composite, `h = minFac L` and `c = L / h`. -/
theorem gainPair_iff_leg {q j h c : ℕ} {ε : ℤ} (hq : 7 ≤ q) (hh : HighHalf q j)
    (hc : Clear (cutLo q) j) :
    GainPair q j h ε c ↔
      (ε = -1 ∧ ¬ (30 * j - 1).Prime ∧ h = (30 * j - 1).minFac ∧
          c = (30 * j - 1) / (30 * j - 1).minFac) ∨
        (ε = 1 ∧ ¬ (30 * j + 1).Prime ∧ h = (30 * j + 1).minFac ∧
          c = (30 * j + 1) / (30 * j + 1).minFac) := by
  have hj1 := highHalf_one_le hq hh
  constructor
  · intro hp
    have hid := pair_leg hq hp
    obtain ⟨hB, hε, hcp, hhc, -⟩ := hp
    obtain ⟨hnp, hmin, hdiv⟩ := two_prime_product hB.1 hcp hhc
    have hN : (((h * c : ℕ) : ℤ)) = 30 * j + ε := by
      push_cast
      rw [hid]
    generalize hNdef : h * c = N at hN hnp hmin hdiv
    rcases hε with rfl | rfl
    · right
      have hL : 30 * j + 1 = N := by omega
      rw [hL]
      exact ⟨rfl, hnp, hmin.symm, by rw [hmin, hdiv]⟩
    · left
      have hL : 30 * j - 1 = N := by omega
      rw [hL]
      exact ⟨rfl, hnp, hmin.symm, by rw [hmin, hdiv]⟩
  · rintro (⟨rfl, hnp, rfl, rfl⟩ | ⟨rfl, hnp, rfl, rfl⟩)
    · obtain ⟨hB, hcp, hle, -, hL⟩ := high_leg_factor hq hh hc (Or.inl rfl) hnp
      refine pair_of_factor hq hh hc hB hcp hle (Or.inr rfl) ?_
      rw [← Nat.cast_mul, ← hL]
      omega
    · obtain ⟨hB, hcp, hle, -, hL⟩ := high_leg_factor hq hh hc (Or.inr rfl) hnp
      refine pair_of_factor hq hh hc hB hcp hle (Or.inl rfl) ?_
      rw [← Nat.cast_mul, ← hL]
      push_cast
      ring

/-- **(iii), multiplicity.** For `q ≥ 7` and `j ∈ G_q`: copy `j` arises from exactly two pairs
iff both legs are composite, and from exactly one pair otherwise. -/
theorem pair_count {q j : ℕ} (hq : 7 ≤ q) (hg : GainCopy q j) :
    ((pairSet q j).ncard = 2 ↔ (¬ (30 * j - 1).Prime ∧ ¬ (30 * j + 1).Prime)) ∧
      ((pairSet q j).ncard = 1 ↔ ((30 * j - 1).Prime ∨ (30 * j + 1).Prime)) := by
  obtain ⟨hh, hc, g, hB, hs⟩ := (gain_iff_band_strike hq).mp hg
  have key : ∀ t : ℕ × ℤ × ℕ, t ∈ pairSet q j ↔
      (t = ((30 * j - 1).minFac, -1, (30 * j - 1) / (30 * j - 1).minFac) ∧
          ¬ (30 * j - 1).Prime) ∨
        (t = ((30 * j + 1).minFac, 1, (30 * j + 1) / (30 * j + 1).minFac) ∧
          ¬ (30 * j + 1).Prime) := by
    rintro ⟨h, ε, c⟩
    simp only [pairSet, Set.mem_ofPred_eq, Prod.mk.injEq]
    rw [gainPair_iff_leg hq hh hc]
    tauto
  have hsome : ¬ (30 * j - 1).Prime ∨ ¬ (30 * j + 1).Prime := by
    rcases (strikesCopy_iff_leg hB.1).mp hs with hd | hd
    · exact Or.inl (band_dvd_high_leg_not_prime hq hh (Or.inl rfl) hB hd)
    · exact Or.inr (band_dvd_high_leg_not_prime hq hh (Or.inr rfl) hB hd)
  by_cases hm : (30 * j - 1).Prime <;> by_cases hp : (30 * j + 1).Prime
  · exfalso
    tauto
  · have e : pairSet q j = {((30 * j + 1).minFac, (1 : ℤ), (30 * j + 1) / (30 * j + 1).minFac)} := by
      ext t
      rw [key, Set.mem_singleton_iff]
      tauto
    rw [e, Set.ncard_singleton]
    simp [hm, hp]
  · have e : pairSet q j = {((30 * j - 1).minFac, (-1 : ℤ), (30 * j - 1) / (30 * j - 1).minFac)} := by
      ext t
      rw [key, Set.mem_singleton_iff]
      tauto
    rw [e, Set.ncard_singleton]
    simp [hm, hp]
  · have e : pairSet q j = {((30 * j - 1).minFac, (-1 : ℤ), (30 * j - 1) / (30 * j - 1).minFac),
        ((30 * j + 1).minFac, (1 : ℤ), (30 * j + 1) / (30 * j + 1).minFac)} := by
      ext t
      rw [key, Set.mem_insert_iff, Set.mem_singleton_iff]
      tauto
    have hne : ((30 * j - 1).minFac, (-1 : ℤ), (30 * j - 1) / (30 * j - 1).minFac) ≠
        ((30 * j + 1).minFac, (1 : ℤ), (30 * j + 1) / (30 * j + 1).minFac) := by
      simp
    rw [e, Set.ncard_pair hne]
    simp [hm, hp]

/-! ## (iv) The mirror trichotomy -/

/-- On a revealed copy no prime striker acts (a prime striker is a leg, and a leg squared
exceeds `30 x + 1`). -/
theorem revealed_not_acts {q x p : ℕ} (hr : Revealed q x) (hp : p.Prime)
    (hs : StrikesCopy p x) : ¬ Acts p x := by
  obtain ⟨hx1, -, -, ht⟩ := hr
  intro ha
  unfold Acts at ha
  rcases (strikesCopy_iff_leg hp).mp hs with hd | hd
  · have heq := (Nat.prime_dvd_prime_iff_eq hp ht.1).mp hd
    subst heq
    have h1 : 29 * (30 * x - 1) ≤ (30 * x - 1) ^ 2 := by
      rw [pow_two]
      exact Nat.mul_le_mul_right _ (by omega)
    generalize (30 * x - 1) ^ 2 = s at h1 ha
    omega
  · have heq := (Nat.prime_dvd_prime_iff_eq hp ht.2).mp hd
    subst heq
    have h1 : 31 * (30 * x + 1) ≤ (30 * x + 1) ^ 2 := by
      rw [pow_two]
      exact Nat.mul_le_mul_right _ (by omega)
    generalize (30 * x + 1) ^ 2 = s at h1 ha
    omega

/-- On a low survivor `x` all of whose `U⁻` strikers are non-acting, every leg is prime: the
least prime factor of a composite leg would be an acting `U⁻` striker. -/
theorem low_leg_prime_of_quiet {q x L : ℕ} (hq : 7 ≤ q) (hx : LowHalf q x) (hc : Clear q x)
    (hna : ∀ p, Uminus q p → StrikesCopy p x → ¬ Acts p x)
    (hL : L = 30 * x - 1 ∨ L = 30 * x + 1) : L.Prime := by
  have hq5 : 5 ≤ q := by omega
  have hlt := lowHalf_leg_lt hq5 hx
  have hx1 := hx.1
  by_contra hnp
  have hL1 : L ≠ 1 := by omega
  have hmp : L.minFac.Prime := Nat.minFac_prime hL1
  have hmL : L.minFac ∣ L := Nat.minFac_dvd L
  have hsq : L.minFac ^ 2 ≤ L := Nat.minFac_sq_le_self (by omega) hnp
  have hm7 : 7 ≤ L.minFac := prime_dvd_leg_ge_seven hmp hx1 hL hmL
  have hstr : StrikesCopy L.minFac x := strikes_of_dvd_leg hL hmL
  have hqm : q < L.minFac := by
    by_contra hle
    push Not at hle
    exact hc _ hmp hm7 hle hstr
  have hmQ : L.minFac ≤ cutLo q := (le_cutLo_iff hmp).mpr (by omega)
  have hna' := hna _ ⟨hmp, hqm, hmQ⟩ hstr
  unfold Acts at hna'
  generalize L.minFac ^ 2 = s at hsq hna'
  omega

/-- **Case (b) is `Loss_q`.** A low survivor struck by `U⁻`, with no acting `U⁻` striker, is a
twin copy with lower leg in `(q, Q]`. -/
theorem loss_of_quiet {q x : ℕ} (hq : 7 ≤ q) (hx : LowHalf q x) (hc : Clear q x)
    (hb : QuietUStrike q x) : Loss q x := by
  obtain ⟨hs, hna⟩ := hb
  have ht : Twin x := ⟨low_leg_prime_of_quiet hq hx hc hna (Or.inl rfl),
    low_leg_prime_of_quiet hq hx hc hna (Or.inr rfl)⟩
  have hx1 := hx.1
  have hqx := lt_of_twin_clear hx1 ht hc
  obtain ⟨p, ⟨hp, hqp, hpQ⟩, hps⟩ := hs
  refine ⟨hx1, hqx, ?_, ht⟩
  rcases (strikesCopy_iff_leg hp).mp hps with hd | hd
  · have := (Nat.prime_dvd_prime_iff_eq hp ht.1).mp hd
    omega
  · have := (Nat.prime_dvd_prime_iff_eq hp ht.2).mp hd
    omega

/-- **Shadowing.** In case (b), every `U⁻` striker of the low survivor `x` is one of its legs. -/
theorem quiet_striker_is_leg {q x p : ℕ} (hq : 7 ≤ q) (hx : LowHalf q x) (hc : Clear q x)
    (hb : QuietUStrike q x) (hu : Uminus q p) (hps : StrikesCopy p x) :
    p = 30 * x - 1 ∨ p = 30 * x + 1 := by
  have ht := (loss_of_quiet hq hx hc hb).2.2.2
  rcases (strikesCopy_iff_leg hu.1).mp hps with hd | hd
  · exact Or.inl ((Nat.prime_dvd_prime_iff_eq hu.1 ht.1).mp hd)
  · exact Or.inr ((Nat.prime_dvd_prime_iff_eq hu.1 ht.2).mp hd)

/-- **Exactly one case.** For every `q` and every copy `x`, exactly one of (a) `NoUStrike`,
(b) `QuietUStrike`, (c) `ActingUStrike` holds. -/
theorem trichotomy_exactly_one (q x : ℕ) :
    (NoUStrike q x ∨ QuietUStrike q x ∨ ActingUStrike q x) ∧
      ¬ (NoUStrike q x ∧ QuietUStrike q x) ∧ ¬ (NoUStrike q x ∧ ActingUStrike q x) ∧
      ¬ (QuietUStrike q x ∧ ActingUStrike q x) := by
  unfold NoUStrike QuietUStrike ActingUStrike
  refine ⟨?_, ?_, ?_, ?_⟩
  · by_cases hs : ∃ p, Uminus q p ∧ StrikesCopy p x
    · by_cases ha : ∃ p, Uminus q p ∧ StrikesCopy p x ∧ Acts p x
      · exact Or.inr (Or.inr ha)
      · push Not at ha
        exact Or.inr (Or.inl ⟨hs, ha⟩)
    · push Not at hs
      exact Or.inl hs
  · rintro ⟨ha, ⟨p, hu, hs⟩, -⟩
    exact ha p hu hs
  · rintro ⟨ha, ⟨p, hu, hs, -⟩⟩
    exact ha p hu hs
  · rintro ⟨⟨-, hna⟩, ⟨p, hu, hs, hact⟩⟩
    exact hna p hu hs hact

/-- **(iv), trichotomy for every low survivor.** For `q ≥ 7` and a low copy `x ∈ S_q`:
(a) no `U⁻` strike `⇔` revealed and not in `Loss_q`;
(b) some `U⁻` strike, none acting `⇔` revealed and in `Loss_q`;
(c) some acting `U⁻` strike `⇔` not revealed. -/
theorem low_trichotomy {q x : ℕ} (hq : 7 ≤ q) (hx : LowHalf q x) (hc : Clear q x) :
    (NoUStrike q x ↔ Revealed q x ∧ ¬ Loss q x) ∧
      (QuietUStrike q x ↔ Revealed q x ∧ Loss q x) ∧
      (ActingUStrike q x ↔ ¬ Revealed q x) := by
  have ha : NoUStrike q x → Revealed q x ∧ ¬ Loss q x := by
    intro h
    have hQ : Clear (cutLo q) x := (clear_cutLo_iff hq).mpr ⟨hc, h⟩
    exact ⟨(revealed_low_iff hq hx).mpr (Or.inr hQ), fun hL => loss_not_clear_cutLo hL hQ⟩
  have hb : QuietUStrike q x → Revealed q x ∧ Loss q x := by
    intro h
    have hL := loss_of_quiet hq hx hc h
    exact ⟨(revealed_low_iff hq hx).mpr (Or.inl hL), hL⟩
  have hcc : ActingUStrike q x → ¬ Revealed q x := by
    rintro ⟨p, hu, hs, hact⟩ hr
    exact revealed_not_acts hr hu.1 hs hact
  refine ⟨⟨ha, ?_⟩, ⟨hb, ?_⟩, ⟨hcc, ?_⟩⟩
  · rintro ⟨hr, hnL⟩ p hu hs
    rcases (revealed_low_iff hq hx).mp hr with hL | hQ
    · exact hnL hL
    · exact hQ p hu.1 (by have := hu.2.1; omega) hu.2.2 hs
  · rintro ⟨hr, hL⟩
    obtain ⟨hx1, hqx, hxQ, ht⟩ := hL
    exact ⟨⟨30 * x - 1, ⟨ht.1, hqx, hxQ⟩, strikes_of_dvd_leg (Or.inl rfl) dvd_rfl⟩,
      fun p hu hs hact => revealed_not_acts hr hu.1 hs hact⟩
  · intro hnr
    rcases (trichotomy_exactly_one q x).1 with h | h | h
    · exact absurd (ha h).1 hnr
    · exact absurd (hb h).1 hnr
    · exact h

/-- **(iv), mirror trichotomy.** For `q ≥ 7` and `j ∈ G_q`, the mirror `x = M - j` is a low
survivor, exactly one of (a), (b), (c) holds for it, and (a) `⇔` revealed and not in `Loss_q`,
(b) `⇔` revealed and in `Loss_q`, (c) `⇔` not revealed. -/
theorem mirror_trichotomy {q j : ℕ} (hq : 7 ≤ q) (hg : GainCopy q j) :
    LowHalf q (mirrorM q - j) ∧ Clear q (mirrorM q - j) ∧
      ((NoUStrike q (mirrorM q - j) ∨ QuietUStrike q (mirrorM q - j) ∨
          ActingUStrike q (mirrorM q - j)) ∧
        ¬ (NoUStrike q (mirrorM q - j) ∧ QuietUStrike q (mirrorM q - j)) ∧
        ¬ (NoUStrike q (mirrorM q - j) ∧ ActingUStrike q (mirrorM q - j)) ∧
        ¬ (QuietUStrike q (mirrorM q - j) ∧ ActingUStrike q (mirrorM q - j))) ∧
      (NoUStrike q (mirrorM q - j) ↔
          Revealed q (mirrorM q - j) ∧ ¬ Loss q (mirrorM q - j)) ∧
      (QuietUStrike q (mirrorM q - j) ↔
          Revealed q (mirrorM q - j) ∧ Loss q (mirrorM q - j)) ∧
      (ActingUStrike q (mirrorM q - j) ↔ ¬ Revealed q (mirrorM q - j)) := by
  have hq5 : 5 ≤ q := by omega
  obtain ⟨hh, hcQ, -⟩ := hg
  have hj1 := highHalf_one_le hq hh
  have hjM : j < mirrorM q := by
    have := seven_le_mirrorM hq
    unfold HighHalf at hh
    omega
  have hx : LowHalf q (mirrorM q - j) := highHalf_mirror hq5 hh
  have hc : Clear q (mirrorM q - j) :=
    (clear_mirror hq5 hj1 hjM).mpr (clear_q_of_clear_cutLo hcQ)
  exact ⟨hx, hc, trichotomy_exactly_one q _, low_trichotomy hq hx hc⟩

/-! ## (v) The shared class -/

/-- **(v), shared class.** For `q ≥ 7`, a prime `h > Q` dividing `q# - 2`, and an odd `d < M`
with `h ∣ d`: the low member `s = (M - d)/2` of the mirror pair has minus leg `30 s - 1 = h c*`
with `c* < h` and `c*² < ρ`; `h` also divides the high member's minus leg; and if `s ∈ S_q`, then
every prime factor of `c*` is a `U⁻` gear that strikes `s` and acts on it. -/
theorem shared_class {q d h : ℕ} (hq : 7 ≤ q) (hh : h.Prime) (hQh : cutLo q < h)
    (hd : Odd d) (hdM : d < mirrorM q) (hdiv : h ∣ primorial q - 2) (hhd : h ∣ d) :
    ∃ c, 30 * mirrorLo q d - 1 = h * c ∧ h ∣ 30 * mirrorHi q d - 1 ∧ c < h ∧
      c ^ 2 < rho q ∧
      (Clear q (mirrorLo q d) → ∀ f, f.Prime → f ∣ c →
        Uminus q f ∧ StrikesCopy f (mirrorLo q d) ∧ Acts f (mirrorLo q d)) := by
  have hq5 : 5 ≤ q := by omega
  have hs1 : 1 ≤ mirrorLo q d := one_le_mirrorLo hq5 hd hdM
  have hs' : mirrorHi q d = mirrorLo q d + d := mirrorHi_eq hq5 hd hdM
  have hsum := (legs_diff hq5 hd hdM).1
  rw [hs'] at hsum ⊢
  have hslow : LowHalf q (mirrorLo q d) := by
    refine ⟨hs1, ?_⟩
    obtain ⟨e, he⟩ := hd
    unfold mirrorLo
    omega
  have hlt := lowHalf_leg_lt hq5 hslow
  generalize mirrorLo q d = s at hs1 hsum hslow hlt ⊢
  have h7 : 7 ≤ h := by
    have := seven_le_cutLo hq
    omega
  have hsq := band_sq_gt hh hQh
  have h30d : h ∣ 30 * d := Dvd.dvd.mul_left hhd 30
  have e1 : primorial q - 2 = 2 * (30 * s - 1) + 30 * d := by omega
  have h2L : h ∣ 2 * (30 * s - 1) := by
    rw [e1] at hdiv
    exact (Nat.dvd_add_left h30d).mp hdiv
  have hL : h ∣ 30 * s - 1 := by
    rcases (Nat.Prime.dvd_mul hh).mp h2L with h2 | h2
    · have := Nat.le_of_dvd (by norm_num) h2
      omega
    · exact h2
  have hH : h ∣ 30 * (s + d) - 1 := by
    have e2 : 30 * (s + d) - 1 = (30 * s - 1) + 30 * d := by omega
    rw [e2]
    exact dvd_add hL h30d
  obtain ⟨c, hcL⟩ := hL
  have hc0 : 0 < c := by
    rcases Nat.eq_zero_or_pos c with h0 | h0
    · rw [h0, mul_zero] at hcL
      omega
    · exact h0
  have hch : c < h := by
    have h1 : h * c < h * h := by
      rw [← hcL, ← pow_two]
      omega
    exact Nat.lt_of_mul_lt_mul_left h1
  have hcc : c * c ≤ h * c := Nat.mul_le_mul_right c hch.le
  have hc2 : c ^ 2 < rho q := by
    rw [pow_two]
    omega
  refine ⟨c, hcL, hH, hch, hc2, ?_⟩
  intro hcs f hf hfc
  have hfle : f ≤ c := Nat.le_of_dvd hc0 hfc
  have hfL : f ∣ 30 * s - 1 := dvd_trans hfc (Dvd.intro_left h hcL.symm)
  have hf7 : 7 ≤ f := prime_dvd_leg_ge_seven hf hs1 (Or.inl rfl) hfL
  have hstr : StrikesCopy f s := strikes_of_dvd_leg (Or.inl rfl) hfL
  have hff : f * f ≤ c * c := Nat.mul_le_mul hfle hfle
  have hqf : q < f := by
    by_contra hle
    push Not at hle
    exact hcs f hf hf7 hle hstr
  have hfQ : f ≤ cutLo q := by
    refine (le_cutLo_iff hf).mpr ?_
    rw [pow_two]
    rw [pow_two] at hc2
    omega
  refine ⟨⟨hf, hqf, hfQ⟩, hstr, ?_⟩
  unfold Acts
  rw [pow_two]
  omega

/-- **(v) for a band gear.** The shared-class statement for `h ∈ B` dividing `q# - 2`. -/
theorem shared_class_band {q d h : ℕ} (hq : 7 ≤ q) (hB : Band q h)
    (hd : Odd d) (hdM : d < mirrorM q) (hdiv : h ∣ primorial q - 2) (hhd : h ∣ d) :
    ∃ c, 30 * mirrorLo q d - 1 = h * c ∧ h ∣ 30 * mirrorHi q d - 1 ∧ c < h ∧
      c ^ 2 < rho q ∧
      (Clear q (mirrorLo q d) → ∀ f, f.Prime → f ∣ c →
        Uminus q f ∧ StrikesCopy f (mirrorLo q d) ∧ Acts f (mirrorLo q d)) :=
  shared_class hq hB.1 hB.2.1 hd hdM hdiv hhd

/-! ## (vi) The three-class rule -/

/-- **Four classes in a field of odd characteristic.** For an odd prime `p`, `a ≠ 0` and
`ε = ±1` in `ZMod p`: the classes `0`, `2 ε a`, `a N`, `a (N + 2 ε)` are pairwise distinct iff
`N ≠ 0`, `N ≠ 2` and `N ≠ -2`. -/
theorem four_classes_distinct_zmod {p : ℕ} [Fact p.Prime] (hp2 : p ≠ 2) {a N ε : ZMod p}
    (ha : a ≠ 0) (hε : ε = 1 ∨ ε = -1) :
    ((0 : ZMod p) ≠ 2 * ε * a ∧ (0 : ZMod p) ≠ a * N ∧ (0 : ZMod p) ≠ a * (N + 2 * ε) ∧
      2 * ε * a ≠ a * N ∧ 2 * ε * a ≠ a * (N + 2 * ε) ∧ a * N ≠ a * (N + 2 * ε)) ↔
      (N ≠ 0 ∧ N ≠ 2 ∧ N ≠ -2) := by
  have h2 : (2 : ZMod p) ≠ 0 := by
    intro h
    have h' : ((2 : ℕ) : ZMod p) = 0 := by exact_mod_cast h
    rw [ZMod.natCast_eq_zero_iff] at h'
    exact hp2 ((Nat.prime_dvd_prime_iff_eq Fact.out Nat.prime_two).mp h')
  have hε0 : ε ≠ 0 := by
    rcases hε with rfl | rfl
    · exact one_ne_zero
    · exact neg_ne_zero.mpr one_ne_zero
  have h2ε : 2 * ε ≠ 0 := mul_ne_zero h2 hε0
  have hinj : ∀ x y : ZMod p, a * x = a * y → x = y := fun x y h => mul_left_cancel₀ ha h
  constructor
  · rintro ⟨-, c2, c3, c4, -, -⟩
    refine ⟨fun hN => c2 (by rw [hN, mul_zero]), ?_, ?_⟩
    · intro hN
      rcases hε with rfl | rfl
      · exact c4 (by rw [hN]; ring)
      · exact c3 (by rw [hN]; ring)
    · intro hN
      rcases hε with rfl | rfl
      · exact c3 (by rw [hN]; ring)
      · exact c4 (by rw [hN]; ring)
  · rintro ⟨hN0, hN2, hNm2⟩
    have hNε : N + 2 * ε ≠ 0 := by
      intro h
      rcases hε with rfl | rfl
      · exact hNm2 (by linear_combination h)
      · exact hN2 (by linear_combination h)
    have hNε' : N ≠ 2 * ε := by
      intro h
      rcases hε with rfl | rfl
      · exact hN2 (by linear_combination h)
      · exact hNm2 (by linear_combination h)
    refine ⟨fun h => mul_ne_zero h2ε ha h.symm, fun h => mul_ne_zero ha hN0 h.symm,
      fun h => mul_ne_zero ha hNε h.symm, ?_, ?_, ?_⟩
    · intro h
      exact hNε' (hinj _ _ (by linear_combination h)).symm
    · intro h
      exact hN0 (by
        have := hinj _ _ (show a * (2 * ε) = a * (N + 2 * ε) by linear_combination h)
        linear_combination -this)
    · intro h
      exact h2ε (by
        have := hinj _ _ (show a * N = a * (N + 2 * ε) by exact h)
        linear_combination -this)

/-- **(vi), general form.** For every odd prime `p` not dividing `h`, every `ε = ±1` and every
`N ≥ 2`: the classes `0`, `2 ε h⁻¹`, `h⁻¹ N`, `h⁻¹ (N + 2 ε)` mod `p` are pairwise distinct iff
`p ∤ N`, `p ∤ N - 2` and `p ∤ N + 2`. -/
theorem four_classes_distinct_iff {p h N : ℕ} {ε : ℤ} (hp : p.Prime) (hp2 : p ≠ 2)
    (hph : ¬ p ∣ h) (hε : ε = 1 ∨ ε = -1) (hN : 2 ≤ N) :
    FourClassesDistinct p h ε N ↔ (¬ p ∣ N ∧ ¬ p ∣ N - 2 ∧ ¬ p ∣ N + 2) := by
  have := Fact.mk hp
  have hh0 : (h : ZMod p) ≠ 0 := by
    rw [Ne, ZMod.natCast_eq_zero_iff]
    exact hph
  have hε' : (ε : ZMod p) = 1 ∨ (ε : ZMod p) = -1 := by
    rcases hε with rfl | rfl <;> simp
  unfold FourClassesDistinct
  rw [four_classes_distinct_zmod hp2 (inv_ne_zero hh0) hε']
  have e1 : (N : ZMod p) ≠ 0 ↔ ¬ p ∣ N := by rw [Ne, ZMod.natCast_eq_zero_iff]
  have e2 : (N : ZMod p) ≠ 2 ↔ ¬ p ∣ N - 2 := by
    rw [← ZMod.natCast_eq_zero_iff, Nat.cast_sub hN, Ne, sub_eq_zero]
    push_cast
    rfl
  have e3 : (N : ZMod p) ≠ -2 ↔ ¬ p ∣ N + 2 := by
    rw [← ZMod.natCast_eq_zero_iff, Nat.cast_add, Nat.cast_ofNat, ← eq_neg_iff_add_eq_zero]
  rw [e1, e2, e3]

/-- **(vi), three-class rule.** For `q ≥ 7`, every prime `p > q` (in particular every `U⁻` gear),
every `h` with `p ∤ h` and every `ε = ±1`: the four classes `0`, `2 ε h⁻¹`, `h⁻¹ q#`,
`h⁻¹ (q# + 2 ε)` mod `p` are pairwise distinct iff `p ∤ q# - 2` and `p ∤ q# + 2`. -/
theorem three_class_rule {q p h : ℕ} {ε : ℤ} (hq : 7 ≤ q) (hp : p.Prime) (hqp : q < p)
    (hph : ¬ p ∣ h) (hε : ε = 1 ∨ ε = -1) :
    FourClassesDistinct p h ε (primorial q) ↔
      (¬ p ∣ primorial q - 2 ∧ ¬ p ∣ primorial q + 2) := by
  have h210 := primorial_ge_210 hq
  rw [four_classes_distinct_iff hp (by omega) hph hε (by omega)]
  have hnd : ¬ p ∣ primorial q := fun hd => by
    have := (hp.dvd_primorial_iff).mp hd
    omega
  exact ⟨fun h => h.2, fun h => ⟨hnd, h⟩⟩

/-- **(vi) for `U⁻` and `B`.** For `q ≥ 7`, `p ∈ U⁻`, `h ∈ B` and `ε = ±1`: the four classes
are pairwise distinct iff `p ∤ q# - 2` and `p ∤ q# + 2`. -/
theorem three_class_rule_band {q p h : ℕ} {ε : ℤ} (hq : 7 ≤ q) (hp : Uminus q p)
    (hB : Band q h) (hε : ε = 1 ∨ ε = -1) :
    FourClassesDistinct p h ε (primorial q) ↔
      (¬ p ∣ primorial q - 2 ∧ ¬ p ∣ primorial q + 2) :=
  three_class_rule hq hp.1 hp.2.1 (not_dvd_of_le_cutLo_lt hp.1 hB.1 hp.2.2 hB.2.1) hε

end Gen2

end RangeLine
