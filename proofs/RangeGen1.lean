import Mathlib.NumberTheory.Bertrand
import Mathlib.NumberTheory.Primorial
import Mathlib.Tactic
import RangeMirror
import RangeHandoff
import RangeLocator

/-!
# The height split of the range, for every `q ≥ 7`

Copy `j` is the pair of legs `30 j - 1`, `30 j + 1`; a prime `p` strikes copy `j` when it divides
a leg (`RangeCentre.StrikesCopy`). Fix any natural number `q ≥ 7` (no primality, no size bound)
and write `q#` for `primorial q`. Notation, all as definitions below:

* `M = q# / 30` (`RangeMirror.mirrorM`; odd, and `30 M = q#`), `ρ = q#/2 + 1 = 15 M + 1` (`rho`).
* `Q` = the largest prime `≤ √ρ` (`cutLo`), `Q*` = the largest prime `≤ √(q#)` (`cutHi`).
* low half `[1, (M-1)/2]` (`LowHalf`), high half `[(M+1)/2, M-1]` (`HighHalf`).
* `S_X` (`Clear X j`): no prime `p` with `7 ≤ p ≤ X` strikes copy `j`.
* `U⁻` = primes in `(q, Q]` (`Uminus`), `B` = primes in `(Q, Q*]` (`Band`).
* a revealed range copy (`Revealed q j`): `1 ≤ j ≤ M - 1`, `j ∈ S_q` (a survivor) and both legs
  prime (`Twin j`); `Range(q)` (`RangeCopy q`) says some revealed range copy exists.
* `Loss_q` (`Loss q j`): a twin copy whose lower leg lies in `(q, Q]`.
* `a_g = 30⁻¹ (mod g)` (`gearRes g`).

This file proves, for every `q ≥ 7`:

* The cut-offs: `cutLo_prime`, `cutHi_prime`, `cutLo_sq_le`, `cutHi_sq_le`, and the
  characterisations `le_cutLo_iff` (a prime `p` is `≤ Q` iff `p² ≤ ρ`) and `le_cutHi_iff`
  (`p ≤ Q*` iff `p² ≤ q#`), so `Q`, `Q*` are the largest primes below `√ρ`, `√(q#)`.
* The growth lemma `two_mul_sq_lt_primorial` (from Bertrand's postulate): for every prime `p`
  with `7 ≤ p ≤ q`, `2 p² < q#`. Hence every such `p` is `≤ Q ≤ Q*` (`le_cutLo_of_le`,
  `le_cutHi_of_le`), so `S_Q ⊆ S_q` and `S_{Q*} ⊆ S_q`.
* The T-rule in the form used: `clear_of_twin`, `lt_of_twin_clear`, and `twin_of_clear` (a leg
  whose least prime factor would be at most the cut-off must be prime).
* `revealed_low_iff` (low rule): on the low half, revealed `⇔ Loss_q ∨ S_Q`.
* `revealed_high_iff` (high rule): on the high half, revealed `⇔ S_{Q*}`.
* `revealed_iff_split`: revealed `⇔ Loss_q ∨ (S_Q ∩ low) ∨ (S_{Q*} ∩ high)`, and
  `split_disjoint`: the three parts are pairwise disjoint.
* `range_iff_split`: `Range(q) ⇔ Loss_q ≠ ∅ ∨ S_Q ∩ low ≠ ∅ ∨ S_{Q*} ∩ high ≠ ∅`, and
  `not_range_iff_empty`, its negation.
* The mirror: `strikes_mirror` / `clear_mirror` (`S_q` is invariant under `j ↦ M - j`),
  `lowHalf_mirror`, `highHalf_mirror`.
* `clear_cutLo_iff`, `clear_cutHi_iff`: `S_Q = S_q minus the U⁻-struck copies`,
  `S_{Q*} = S_q minus the (U⁻ ∪ B)-struck copies`; `uminus_or_band_iff`: `U⁻ ∪ B` is the set of
  primes in `(q, Q*]`.
* `not_range_iff`: `¬ Range(q) ⇔ Loss_q = ∅ ∧` every low `x ∈ S_q` is struck by some `p ∈ U⁻`
  and its mirror `M - x` by some `g ∈ U⁻ ∪ B`.
* `strikes_iff_gearRes`, `strikes_mirror_iff_gearRes`: the residue forms `x ≡ ±a_p (mod p)` and
  `x ≡ M ± a_g (mod g)`; `not_range_iff_residue`: the not-Range statement in residue form.
* `rangeCopy_implies_rangeStatement`: a revealed range copy is a witness of
  `RangeHandoff.RangeStatement q`.

Correction to the requested wording: `Loss_q` is taken as the twin copies whose *lower* leg lies in
`(q, Q]`. The upper leg is then in `(q, Q + 2]`; it exceeds `Q` exactly when `Q = 30 j - 1` and
`Q + 2` is a prime above `√ρ`, and with "both legs in `(q, Q]`" such a copy (revealed, and struck
by `Q`, so not in `S_Q`) would lie in none of the three parts.
-/

namespace RangeLine

/-! ## Notation -/

/-- `ρ = q#/2 + 1`. -/
def rho (q : ℕ) : ℕ := primorial q / 2 + 1

/-- `Q`: the largest prime `≤ √ρ`, i.e. the largest prime `≤ Nat.sqrt ρ` (`0` if there is none;
for `q ≥ 7` there is one, `cutLo_prime`). -/
def cutLo (q : ℕ) : ℕ := Nat.findGreatest Nat.Prime (Nat.sqrt (rho q))

/-- `Q*`: the largest prime `≤ √(q#)`, i.e. the largest prime `≤ Nat.sqrt (q#)`. -/
def cutHi (q : ℕ) : ℕ := Nat.findGreatest Nat.Prime (Nat.sqrt (primorial q))

/-- `S_X`: no prime `p` with `7 ≤ p ≤ X` strikes copy `j`. -/
def Clear (X j : ℕ) : Prop := ∀ p, p.Prime → 7 ≤ p → p ≤ X → ¬ StrikesCopy p j

/-- Copy `j` is a twin copy: both legs `30 j - 1` and `30 j + 1` are prime. -/
def Twin (j : ℕ) : Prop := (30 * j - 1).Prime ∧ (30 * j + 1).Prime

/-- The low half of the range: `1 ≤ j ≤ (M - 1)/2`. -/
def LowHalf (q j : ℕ) : Prop := 1 ≤ j ∧ j ≤ (mirrorM q - 1) / 2

/-- The high half of the range: `(M + 1)/2 ≤ j ≤ M - 1`. -/
def HighHalf (q j : ℕ) : Prop := (mirrorM q + 1) / 2 ≤ j ∧ j ≤ mirrorM q - 1

/-- A revealed range copy: `1 ≤ j ≤ M - 1`, `j ∈ S_q` (a survivor), and both legs prime. -/
def Revealed (q j : ℕ) : Prop := 1 ≤ j ∧ j ≤ mirrorM q - 1 ∧ Clear q j ∧ Twin j

/-- `Loss_q`: a twin copy whose lower leg lies in `(q, Q]`. -/
def Loss (q j : ℕ) : Prop := 1 ≤ j ∧ q < 30 * j - 1 ∧ 30 * j - 1 ≤ cutLo q ∧ Twin j

/-- `Range(q)` in the copy frame: some revealed range copy exists. -/
def RangeCopy (q : ℕ) : Prop := ∃ j, Revealed q j

/-- `U⁻`: the primes in `(q, Q]`. -/
def Uminus (q g : ℕ) : Prop := g.Prime ∧ q < g ∧ g ≤ cutLo q

/-- `B`: the primes in `(Q, Q*]`. -/
def Band (q g : ℕ) : Prop := g.Prime ∧ cutLo q < g ∧ g ≤ cutHi q

/-- `a_g = 30⁻¹` in `ZMod g`: gear `g ≥ 7` strikes copy `j` iff `j ≡ ±a_g (mod g)`. -/
def gearRes (g : ℕ) : ZMod g := (30 : ZMod g)⁻¹

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

/-- **Growth lemma.** For every prime `p` with `7 ≤ p ≤ q`, `2 p² < q#`. (Bertrand gives a prime
`r` with `p/2 < r < p`; then `2 · 3 · r · p ∣ q#` and `6 r ≥ 3 p + 3`.) -/
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

/-- `Q² ≤ ρ`. -/
theorem cutLo_sq_le (q : ℕ) : cutLo q ^ 2 ≤ rho q :=
  Nat.le_sqrt'.mp (Nat.findGreatest_le _)

/-- `Q*² ≤ q#`. -/
theorem cutHi_sq_le (q : ℕ) : cutHi q ^ 2 ≤ primorial q :=
  Nat.le_sqrt'.mp (Nat.findGreatest_le _)

/-- For `q ≥ 7`, `7 ≤ Q` (as `7² ≤ ρ`). -/
theorem seven_le_cutLo {q : ℕ} (hq : 7 ≤ q) : 7 ≤ cutLo q := by
  have h := primorial_ge_210 hq
  exact (le_cutLo_iff (by norm_num)).mpr (by unfold rho; omega)

/-- For `q ≥ 7`, `Q` is prime (so it is the largest prime `≤ √ρ`). -/
theorem cutLo_prime {q : ℕ} (hq : 7 ≤ q) : (cutLo q).Prime := by
  have h := primorial_ge_210 hq
  exact Nat.findGreatest_spec (m := 7) (Nat.le_sqrt'.mpr (by unfold rho; omega)) (by norm_num)

/-- For `q ≥ 7`, `Q*` is prime (so it is the largest prime `≤ √(q#)`). -/
theorem cutHi_prime {q : ℕ} (hq : 7 ≤ q) : (cutHi q).Prime := by
  have h := primorial_ge_210 hq
  exact Nat.findGreatest_spec (m := 7) (Nat.le_sqrt'.mpr (by omega)) (by norm_num)

/-- For `q ≥ 7`, `Q ≤ Q*`. -/
theorem cutLo_le_cutHi {q : ℕ} (hq : 7 ≤ q) : cutLo q ≤ cutHi q := by
  have h := primorial_ge_210 hq
  have h1 := cutLo_sq_le q
  refine (le_cutHi_iff (cutLo_prime hq)).mpr ?_
  unfold rho at h1
  omega

/-- Every prime `p` with `7 ≤ p ≤ q` is at most `Q` (growth lemma: `p² < q#/2 < ρ`). -/
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

/-- `S_Q ⊆ S_q`, for every `q`. -/
theorem clear_q_of_clear_cutLo {q j : ℕ} (h : Clear (cutLo q) j) : Clear q j :=
  fun p hp hp7 hpq => h p hp hp7 (le_cutLo_of_le hp hp7 hpq)

/-- `S_{Q*} ⊆ S_q`, for every `q`. -/
theorem clear_q_of_clear_cutHi {q j : ℕ} (h : Clear (cutHi q) j) : Clear q j :=
  fun p hp hp7 hpq => h p hp hp7 (le_cutHi_of_le hp hp7 hpq)

/-! ## The T-rule on one copy -/

/-- A twin copy whose lower leg is above `X` is in `S_X` (a prime striking a prime leg is the
leg itself). -/
theorem clear_of_twin {X j : ℕ} (ht : Twin j) (h : X < 30 * j - 1) : Clear X j := by
  intro p hp _ hpX hs
  rcases (strikesCopy_iff_leg hp).mp hs with h1 | h1
  · have := (Nat.prime_dvd_prime_iff_eq hp ht.1).mp h1
    omega
  · have := (Nat.prime_dvd_prime_iff_eq hp ht.2).mp h1
    omega

/-- A twin copy (`j ≥ 1`) in `S_X` has its lower leg above `X` (else the lower leg, a prime
`≥ 29`, strikes it). -/
theorem lt_of_twin_clear {X j : ℕ} (hj : 1 ≤ j) (ht : Twin j) (hc : Clear X j) :
    X < 30 * j - 1 := by
  by_contra h
  push Not at h
  exact hc (30 * j - 1) ht.1 (by omega) h ((strikesCopy_iff_leg ht.1).mpr (Or.inl dvd_rfl))

/-- A leg `L` of copy `j ≥ 1` is prime when copy `j` is in `S_X` and every prime `m` with
`m² ≤ L` is at most `X`: otherwise the least prime factor `m` of `L` has `m² ≤ L`, and `m ≥ 7`
since `L ≡ ±1 (mod 30)`, so `m` is a prime in `[7, X]` striking `j`. -/
theorem leg_prime_of_clear {X j L : ℕ} (hj : 1 ≤ j) (hL : L = 30 * j - 1 ∨ L = 30 * j + 1)
    (hX : ∀ m, m.Prime → m ^ 2 ≤ L → m ≤ X) (hc : Clear X j) : L.Prime := by
  by_contra hnp
  have hL1 : L ≠ 1 := by omega
  set m := L.minFac with hmdef
  have hm : m.Prime := Nat.minFac_prime hL1
  have hmL : m ∣ L := Nat.minFac_dvd L
  have hsq : m ^ 2 ≤ L := Nat.minFac_sq_le_self (by omega) hnp
  have hm7 : 7 ≤ m := by
    by_contra h7
    push Not at h7
    have h2 := hm.two_le
    interval_cases m <;> omega
  refine hc m hm hm7 (hX m hm hsq) ?_
  rw [strikesCopy_iff_leg hm]
  rcases hL with rfl | rfl
  · exact Or.inl hmL
  · exact Or.inr hmL

/-- Copy `j ≥ 1` in `S_X` is a twin copy when every prime `m` with `m² ≤ 30 j + 1` is at most
`X`. -/
theorem twin_of_clear {X j : ℕ} (hj : 1 ≤ j) (hX : ∀ m, m.Prime → m ^ 2 ≤ 30 * j + 1 → m ≤ X)
    (hc : Clear X j) : Twin j :=
  ⟨leg_prime_of_clear hj (Or.inl rfl) (fun m hm h => hX m hm (by omega)) hc,
    leg_prime_of_clear hj (Or.inr rfl) hX hc⟩

/-- For every `q`: a revealed range copy is exactly a twin copy `1 ≤ j ≤ M - 1` with lower leg
above `q`. -/
theorem revealed_iff_twin_above {q j : ℕ} :
    Revealed q j ↔ 1 ≤ j ∧ j ≤ mirrorM q - 1 ∧ q < 30 * j - 1 ∧ Twin j := by
  constructor
  · rintro ⟨hj, hjM, hc, ht⟩
    exact ⟨hj, hjM, lt_of_twin_clear hj ht hc, ht⟩
  · rintro ⟨hj, hjM, hq, ht⟩
    exact ⟨hj, hjM, clear_of_twin ht hq, ht⟩

/-! ## The two halves -/

/-- The range `[1, M - 1]` is the union of the two halves (`q ≥ 5`, `M` odd). -/
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

/-- Low legs are below `ρ`: for a low copy, `30 j + 1 ≤ 15 M - 14 < ρ`. -/
theorem lowHalf_leg_lt {q j : ℕ} (hq : 5 ≤ q) (h : LowHalf q j) : 30 * j + 1 < rho q := by
  obtain ⟨k, hk⟩ := odd_mirrorM hq
  have hr := rho_eq hq
  unfold LowHalf at h
  omega

/-- High legs lie in `[q#/2 + 14, q# - 29]`: `15 M + 14 ≤ 30 j - 1` and `30 j + 1 < q#`. -/
theorem highHalf_legs {q j : ℕ} (hq : 5 ≤ q) (h : HighHalf q j) :
    15 * mirrorM q + 14 ≤ 30 * j - 1 ∧ 30 * j + 1 < primorial q := by
  obtain ⟨k, hk⟩ := odd_mirrorM hq
  have h30 := thirty_mul_mirrorM hq
  unfold HighHalf at h
  omega

/-- `Loss_q` lies in the low half: `(30 j - 1)² ≤ Q² ≤ ρ = 15 M + 1` forces `j ≤ (M - 1)/2`. -/
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

/-- A `Loss_q` copy is not in `S_Q`: its lower leg is a prime in `[7, Q]`. -/
theorem loss_not_clear_cutLo {q j : ℕ} (h : Loss q j) : ¬ Clear (cutLo q) j := by
  obtain ⟨hj, _, hQ, ht⟩ := h
  intro hc
  have := lt_of_twin_clear hj ht hc
  omega

/-! ## The low rule and the high rule -/

/-- **Low rule.** For `q ≥ 7` and a low copy `j`: `j` is a revealed range copy exactly when
`j ∈ Loss_q` or `j ∈ S_Q`. Low legs are `< ρ`, so a leg with no prime factor `≤ Q` is prime. -/
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

/-- **High rule.** For `q ≥ 7` and a high copy `j`: `j` is a revealed range copy exactly when
`j ∈ S_{Q*}`. High legs lie above `Q*` and below `q#`, so a composite leg has a prime factor
in `[7, Q*]`. -/
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

/-! ## The height split -/

/-- **Height split.** For every `q ≥ 7` and every copy `j`: `j` is a revealed range copy exactly
when `j ∈ Loss_q`, or `j ∈ S_Q` and `j` is low, or `j ∈ S_{Q*}` and `j` is high. -/
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

/-- **The three parts are pairwise disjoint** (`q ≥ 7`): `Loss_q ∩ S_Q = ∅`, `Loss_q` is low, and
the two halves are disjoint. -/
theorem split_disjoint {q : ℕ} (hq : 7 ≤ q) (j : ℕ) :
    ¬ (Loss q j ∧ Clear (cutLo q) j ∧ LowHalf q j) ∧
      ¬ (Loss q j ∧ Clear (cutHi q) j ∧ HighHalf q j) ∧
      ¬ ((Clear (cutLo q) j ∧ LowHalf q j) ∧ (Clear (cutHi q) j ∧ HighHalf q j)) := by
  refine ⟨fun ⟨hL, hc, _⟩ => loss_not_clear_cutLo hL hc, fun ⟨hL, _, hh⟩ => ?_,
    fun ⟨⟨_, hl⟩, ⟨_, hh⟩⟩ => lowHalf_not_highHalf (by omega) hl hh⟩
  exact lowHalf_not_highHalf (by omega) (loss_lowHalf hq hL) hh

/-- **Range(q) by the split.** For every `q ≥ 7`:
`Range(q) ⇔ Loss_q ≠ ∅ ∨ S_Q ∩ low ≠ ∅ ∨ S_{Q*} ∩ high ≠ ∅`. -/
theorem range_iff_split {q : ℕ} (hq : 7 ≤ q) :
    RangeCopy q ↔ (∃ j, Loss q j) ∨ (∃ j, Clear (cutLo q) j ∧ LowHalf q j) ∨
      (∃ j, Clear (cutHi q) j ∧ HighHalf q j) := by
  constructor
  · rintro ⟨j, h⟩
    rcases (revealed_iff_split hq j).mp h with h' | h' | h'
    · exact Or.inl ⟨j, h'⟩
    · exact Or.inr (Or.inl ⟨j, h'⟩)
    · exact Or.inr (Or.inr ⟨j, h'⟩)
  · rintro (⟨j, h⟩ | ⟨j, h⟩ | ⟨j, h⟩)
    · exact ⟨j, (revealed_iff_split hq j).mpr (Or.inl h)⟩
    · exact ⟨j, (revealed_iff_split hq j).mpr (Or.inr (Or.inl h))⟩
    · exact ⟨j, (revealed_iff_split hq j).mpr (Or.inr (Or.inr h))⟩

/-- **not-Range(q), set form.** For every `q ≥ 7`: no revealed range copy exists exactly when
`Loss_q = ∅`, no low copy is in `S_Q`, and no high copy is in `S_{Q*}`. -/
theorem not_range_iff_empty {q : ℕ} (hq : 7 ≤ q) :
    ¬ RangeCopy q ↔ (∀ j, ¬ Loss q j) ∧ (∀ j, LowHalf q j → ¬ Clear (cutLo q) j) ∧
      (∀ j, HighHalf q j → ¬ Clear (cutHi q) j) := by
  rw [range_iff_split hq]
  constructor
  · intro h
    refine ⟨fun j hj => h (Or.inl ⟨j, hj⟩), fun j hl hc => h (Or.inr (Or.inl ⟨j, hc, hl⟩)),
      fun j hh hc => h (Or.inr (Or.inr ⟨j, hc, hh⟩))⟩
  · rintro ⟨h1, h2, h3⟩ (⟨j, hj⟩ | ⟨j, hc, hl⟩ | ⟨j, hc, hh⟩)
    · exact h1 j hj
    · exact h2 j hl hc
    · exact h3 j hh hc

/-! ## The mirror `j ↦ M - j` -/

/-- If `p ∣ N` and `a + b = N`, then `p ∣ a ↔ p ∣ b`. -/
theorem split_dvd_iff_of_add_eq {p a b N : ℕ} (hN : p ∣ N) (h : a + b = N) :
    p ∣ a ↔ p ∣ b := by
  subst h
  exact ⟨fun ha => (Nat.dvd_add_right ha).mp hN, fun hb => (Nat.dvd_add_left hb).mp hN⟩

/-- **Mirror invariance of a strike by `p ≤ q`.** For `q ≥ 5`, a prime `p ≤ q` and
`1 ≤ x < M`: `p` strikes copy `M - x` iff it strikes copy `x` (the cross legs sum to `q#`). -/
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

/-- **`S_q` is mirror-invariant.** For `q ≥ 5` and `1 ≤ x < M`: `M - x ∈ S_q ⇔ x ∈ S_q`. -/
theorem clear_mirror {q x : ℕ} (hq : 5 ≤ q) (hx1 : 1 ≤ x) (hx : x < mirrorM q) :
    Clear q (mirrorM q - x) ↔ Clear q x := by
  constructor
  · intro h p hp hp7 hpq hs
    exact h p hp hp7 hpq ((strikes_mirror hq hp hpq hx1 hx).mpr hs)
  · intro h p hp hp7 hpq hs
    exact h p hp hp7 hpq ((strikes_mirror hq hp hpq hx1 hx).mp hs)

/-- The mirror sends the low half into the high half. -/
theorem lowHalf_mirror {q x : ℕ} (hq : 5 ≤ q) (h : LowHalf q x) :
    HighHalf q (mirrorM q - x) := by
  obtain ⟨k, hk⟩ := odd_mirrorM hq
  unfold LowHalf at h
  unfold HighHalf
  omega

/-- The mirror sends the high half into the low half. -/
theorem highHalf_mirror {q y : ℕ} (hq : 5 ≤ q) (h : HighHalf q y) :
    LowHalf q (mirrorM q - y) := by
  obtain ⟨k, hk⟩ := odd_mirrorM hq
  unfold HighHalf at h
  unfold LowHalf
  omega

/-! ## `S_Q` and `S_{Q*}` as `S_q` minus the upper strikes -/

/-- `U⁻ ∪ B` is the set of primes in `(q, Q*]` (`q ≥ 7`). -/
theorem uminus_or_band_iff {q g : ℕ} (hq : 7 ≤ q) :
    (Uminus q g ∨ Band q g) ↔ g.Prime ∧ q < g ∧ g ≤ cutHi q := by
  have hQ7 := seven_le_cutLo hq
  have hQQ := cutLo_le_cutHi hq
  constructor
  · rintro (⟨hg, hqg, hgQ⟩ | ⟨hg, hQg, hgQ⟩)
    · exact ⟨hg, hqg, le_trans hgQ hQQ⟩
    · refine ⟨hg, ?_, hgQ⟩
      by_contra h
      push Not at h
      have := le_cutLo_of_le hg (by omega) h
      omega
  · rintro ⟨hg, hqg, hgQ⟩
    by_cases h : g ≤ cutLo q
    · exact Or.inl ⟨hg, hqg, h⟩
    · exact Or.inr ⟨hg, by omega, hgQ⟩

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

/-- `S_{Q*} = S_q ∖ ((U⁻ ∪ B)-struck copies)` for `q ≥ 7`. -/
theorem clear_cutHi_iff {q j : ℕ} (hq : 7 ≤ q) :
    Clear (cutHi q) j ↔ Clear q j ∧ ∀ g, (Uminus q g ∨ Band q g) → ¬ StrikesCopy g j := by
  constructor
  · intro hc
    refine ⟨clear_q_of_clear_cutHi hc, fun g hg => ?_⟩
    obtain ⟨hgp, hqg, hgQ⟩ := (uminus_or_band_iff hq).mp hg
    exact hc g hgp (by omega) hgQ
  · rintro ⟨hc, hu⟩ p hp hp7 hpQ
    by_cases hpq : p ≤ q
    · exact hc p hp hp7 hpq
    · exact hu p ((uminus_or_band_iff hq).mpr ⟨hp, by omega, hpQ⟩)

/-- **not-Range(q), strike form.** For every `q ≥ 7`: no revealed range copy exists exactly when
`Loss_q = ∅` and every low `x ∈ S_q` is struck by some `p ∈ U⁻` while its mirror `M - x` is
struck by some `g ∈ U⁻ ∪ B`. -/
theorem not_range_iff {q : ℕ} (hq : 7 ≤ q) :
    ¬ RangeCopy q ↔ (∀ j, ¬ Loss q j) ∧
      ∀ x, LowHalf q x → Clear q x →
        (∃ p, Uminus q p ∧ StrikesCopy p x) ∧
        (∃ g, (Uminus q g ∨ Band q g) ∧ StrikesCopy g (mirrorM q - x)) := by
  have hq5 : 5 ≤ q := by omega
  rw [not_range_iff_empty hq]
  constructor
  · rintro ⟨hL, hlow, hhigh⟩
    refine ⟨hL, fun x hx hc => ⟨?_, ?_⟩⟩
    · by_contra hn
      push Not at hn
      exact hlow x hx ((clear_cutLo_iff hq).mpr ⟨hc, hn⟩)
    · by_contra hn
      push Not at hn
      have hxM : x < mirrorM q := by
        have := lowHalf_mirror hq5 hx
        unfold HighHalf at this
        unfold LowHalf at hx
        omega
      exact hhigh _ (lowHalf_mirror hq5 hx)
        ((clear_cutHi_iff hq).mpr ⟨(clear_mirror hq5 hx.1 hxM).mpr hc, hn⟩)
  · rintro ⟨hL, h⟩
    refine ⟨hL, fun x hx hc => ?_, fun y hy hc => ?_⟩
    · have hc' := (clear_cutLo_iff hq).mp hc
      obtain ⟨⟨p, hu, hs⟩, _⟩ := h x hx hc'.1
      exact hc'.2 p hu hs
    · have hc' := (clear_cutHi_iff hq).mp hc
      have hy1 : 1 ≤ y ∧ y < mirrorM q := by
        obtain ⟨k, hk⟩ := odd_mirrorM hq5
        unfold HighHalf at hy
        omega
      obtain ⟨_, ⟨g, hg, hs⟩⟩ :=
        h _ (highHalf_mirror hq5 hy) ((clear_mirror hq5 hy1.1 hy1.2).mpr hc'.1)
      rw [show mirrorM q - (mirrorM q - y) = y by omega] at hs
      exact hc'.2 g hg hs

/-! ## Residue forms -/

/-- **Strike in residue form.** For a prime `p ≥ 7` and `k ≥ 1`: `p` strikes copy `k` iff
`k ≡ ±a_p (mod p)`, with `a_p = 30⁻¹ (mod p)`. -/
theorem strikes_iff_gearRes {p k : ℕ} (hp : p.Prime) (hp7 : 7 ≤ p) (hk : 1 ≤ k) :
    StrikesCopy p k ↔ ((k : ZMod p) = gearRes p ∨ (k : ZMod p) = -gearRes p) := by
  have := Fact.mk hp
  have h30 : (30 : ZMod p) ≠ 0 := by exact_mod_cast thirty_ne_zero hp hp7
  rw [strikesCopy_iff_zmod hk, mul_eq_zero, sub_eq_zero, add_eq_zero_iff_eq_neg]
  unfold gearRes
  rw [← eq_inv_mul_iff_mul_eq₀ h30, ← eq_inv_mul_iff_mul_eq₀ h30, mul_one, mul_neg_one]

/-- **Mirror strike in residue form.** For a prime `g ≥ 7` and `x < M` (any `M`): `g` strikes
copy `M - x` iff `x ≡ M ± a_g (mod g)`. -/
theorem strikes_mirror_iff_gearRes {g M x : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g) (hx : x < M) :
    StrikesCopy g (M - x) ↔
      ((x : ZMod g) = (M : ZMod g) + gearRes g ∨ (x : ZMod g) = (M : ZMod g) - gearRes g) := by
  rw [strikes_iff_gearRes hg hg7 (by omega), Nat.cast_sub hx.le]
  constructor
  · rintro (h | h)
    · right
      linear_combination -h
    · left
      linear_combination -h
  · rintro (h | h)
    · right
      linear_combination -h
    · left
      linear_combination -h

/-- **not-Range(q), residue form.** For every `q ≥ 7`: no revealed range copy exists exactly when
`Loss_q = ∅` and every low `x ∈ S_q` satisfies both `x ≡ ±a_p (mod p)` for some `p ∈ U⁻` and
`x ≡ M ± a_g (mod g)` for some `g ∈ U⁻ ∪ B`. -/
theorem not_range_iff_residue {q : ℕ} (hq : 7 ≤ q) :
    ¬ RangeCopy q ↔ (∀ j, ¬ Loss q j) ∧
      ∀ x, LowHalf q x → Clear q x →
        (∃ p, Uminus q p ∧ ((x : ZMod p) = gearRes p ∨ (x : ZMod p) = -gearRes p)) ∧
        (∃ g, (Uminus q g ∨ Band q g) ∧
          ((x : ZMod g) = (mirrorM q : ZMod g) + gearRes g ∨
            (x : ZMod g) = (mirrorM q : ZMod g) - gearRes g)) := by
  have hq5 : 5 ≤ q := by omega
  have hQ7 := seven_le_cutLo hq
  rw [not_range_iff hq]
  refine and_congr_right fun _ => forall_congr' fun x => imp_congr_right fun hx =>
    imp_congr_right fun _ => and_congr ?_ ?_
  · refine exists_congr fun p => and_congr_right fun hu => ?_
    exact strikes_iff_gearRes hu.1 (by have := hu.2.1; omega) hx.1
  · refine exists_congr fun g => and_congr_right fun hu => ?_
    obtain ⟨hgp, hqg, _⟩ := (uminus_or_band_iff hq).mp hu
    have hxM : x < mirrorM q := by
      obtain ⟨k, hk⟩ := odd_mirrorM hq5
      unfold LowHalf at hx
      omega
    exact strikes_mirror_iff_gearRes hgp (by omega) hxM

/-! ## Link to the range statement -/

/-- A revealed range copy `j` is a witness of the range statement at `q` (`q ≥ 5`): the twin
pair `(30 j - 1, 30 j + 1)` has `q < 30 j - 1` and `30 j + 1 ≤ q# - 29`. -/
theorem rangeCopy_implies_rangeStatement {q : ℕ} (hq : 5 ≤ q) (h : RangeCopy q) :
    RangeStatement q := by
  obtain ⟨j, hj, hjM, hc, ht⟩ := h
  have hqj := lt_of_twin_clear hj ht hc
  have hM := thirty_mul_mirrorM hq
  have hsum : 30 * j - 1 + 2 = 30 * j + 1 := by omega
  refine ⟨30 * j - 1, hqj, ?_, ht.1, ?_⟩
  · omega
  · rw [hsum]
    exact ht.2

end RangeLine
