import Mathlib.NumberTheory.Primorial
import Mathlib.Data.Nat.Nth
import Mathlib.Tactic
import RangeRegion
import RangeActPair
import RangeMirror

/-!
# Anatomy of total blame, for every machine size (P4)

Copy `j ≥ 1` is the pair of legs `30 j - 1`, `30 j + 1`; a number `g` strikes copy `j` when it
divides the product of the legs (`StrikesCopy`, from `RangeCentre`), and `g` acts at copy `j`
when `g² ≤ 30 j + 1` (`Acts`, from `RangeActPair`). `P j` is the least prime factor of
`900 j² - 1 = (30 j - 1)(30 j + 1)` (`RangeRegion.P`), and `M = M_q = q# / 30` (`mirrorM`, from
`RangeMirror`). The notation of this file (sub-namespace `RangeLine.Gen4`):

* `Surv q j` (`j ∈ S_q`): no prime `p` with `7 ≤ p ≤ q` strikes copy `j`. `surv_iff_coprime`:
  for `q ≥ 5` and `j ≥ 1` this is `gcd(900 j² - 1, M) = 1`, the record's definition of `S_q`.
* `InU q g` (`g ∈ U_q`): `g` is a prime with `q < g ≤ isqrt(q#)`.
* `Twin j` (revealed): both legs of copy `j` are prime.
* `Home g j`: copy `j` is a home copy of `g`, i.e. `g` is one of its legs.
* `BlameF q g j` (`j ∈ F_g`): `1 ≤ j < M`, `P j = g`, and `j` is not a home copy of `g`.
* `cutH q` (`H_q`): `⌊(isqrt(q#) + 1) / 30⌋`.
* `FirstGears q k g`: `g` is one of the first `k` gears `g_1 < g_2 < … < g_k` of `U_q`, where
  the `(i+1)`-th gear is `Nat.nth (InU q) i` (Mathlib's enumeration of a set of naturals in
  increasing order).
* `nextGear q` (`q'`): the least prime above `q`.

Every statement below is quantified over all `q` (no size bound unless one is stated, and none is
specialised to particular machine sizes), all copies `j`, all gears `g`, all legs and all `k`.
The only arithmetic used is the T-rule (a composite `n` has `(minFac n)² ≤ n`) and the region law
`RangeRegion.blamed_iff_minFac` (`¬ Twin j ↔ (P j)² ≤ 30 j + 1` for `j ≥ 1`), itself the T-rule.

## Basic facts

* `seven_le_of_dvd_leg`: a prime dividing a leg of a copy `j ≥ 1` is at least `7`.
* `lt_of_dvd_leg`: every prime factor of a leg of a survivor `j ≥ 1` of machine `q` exceeds `q`.
* `surv_iff_lt_P`: for `j ≥ 1`, `j ∈ S_q ⇔ q < P j`.
* `P_eq_of_twin`: a revealed copy has `P j = 30 j - 1` (its own lower leg).
* `home_not_acts`: a gear never acts on its home copy.
* `shadowing` (row 1, W1): for a survivor `j ≥ 1`, a leg `L` and any `g ∣ L` with `L < g²`,
  either `L = g`, or `h = minFac L` has `q < h < g` and acts at `j`.

## (i) Blame partition
* `nonrevealed_iff_blameF`: for `1 ≤ j < M`, a survivor `j` is not revealed iff `j ∈ F_g` for
  some `g ∈ U_q`.
* `nonrevealed_eq_iUnion`: `{j ∈ [1, M) : j ∈ S_q, ¬ revealed} = ⋃_{g ∈ U_q} F_g`, and
  `blameF_pairwiseDisjoint`: the `F_g` are pairwise disjoint (so the union is `⊔`).
* `blameF_surv`, `blameF_not_twin`: every member of `F_g` (`g > q`) is a non-revealed survivor;
  `blameF_acts`: `g` acts on every member of `F_g`.

## (ii) Revealed survivors
* `revealed_iff_strikers_home`: for `1 ≤ j < M`, a survivor `j` is revealed iff every `U_q`
  striker of `j` is one of its own legs.
* `legs_gt_sqrt_of_gt_cutH`: for `j > H_q` both legs exceed `isqrt(q#)`.
* `revealed_iff_no_strike`: for `H_q < j < M`, a survivor is revealed iff no `U_q` gear strikes it.

## (iii) Joint redundancy
* `joint_redundancy`: for every survivor `j ≥ 1` and every set `D` of `U_q` gears closed downward
  inside `U_q`, some gear of `D` strikes `j` iff some gear of `D` strikes and acts at `j` or `j` is
  a home copy of a gear of `D`.
* `joint_redundancy_le`: the case `D = U_q ∩ [0, X]`, for every `X`.
* `firstGears_down`, `joint_redundancy_first`, `joint_redundancy_set`: the case `D = {g_1, …, g_k}`
  for every `k`, pointwise and as the set identity
  `S_q ∩ ⋃_{i≤k} Str(g_i) = S_q ∩ (⋃_{i≤k} ActStr(g_i) ∪ Home(g_1..g_k))` on the copies `j ≥ 1`.

## (iv) Legs
* `acts_iff_sq_le_leg`: for `j ≥ 1`, a leg `L` and any `h ≥ 2` dividing `L`:
  `h² ≤ 30 j + 1 ⇔ h² ≤ L`. `acts_iff_le_cofactor`: with `L = h c`, acting iff `h ≤ c`.
* `nonacting_factor_unique`: a leg has at most one prime factor that does not act.

## (v) Bottom strata
* `revealed_of_sq_le`: a survivor `j ≥ 1` with `30 j + 1 ≤ q²` is revealed.
* `stratum_nonrevealed`: if no prime lies in `(q, q')`, a non-revealed survivor `j ≥ 1` with
  `30 j + 1 ≤ q'²` has `30 j + 1 = q'²` (so `j = (q'² - 1)/30`), lower leg `q'² - 2` prime, and
  `P j = q'`.
* `stratum_exception`: conversely, for a prime `q' > q` with `q'² ≡ 1 (mod 30)` and `q'² - 2`
  prime, `j = (q'² - 1)/30` is such a survivor.
* `stratum_iff`, `stratum_iff_nextGear`: some non-revealed survivor lies in the stratum
  `30 j + 1 ≤ q'²` iff `q'² ≡ 1 (mod 30)` and `q'² - 2` is prime.

## (vi) The next gear
* `strike_acts_or_home`: if `q'` is a prime and no prime lies in `(q, q')`, then `q'` strikes a
  survivor `j ≥ 1` of `q` only by acting, or at its home copy.
  `nextGear_strike_acts_or_home`: the case `q' = nextGear q`.

## Corrections and sharpenings to the requested wording

* No statement needs `q ≥ 7`: every theorem holds for every `q` (only `surv_iff_coprime` needs
  `q ≥ 5`, where `M = q#/30` is the product of the primes `7..q`). Copies are `j ≥ 1`: in `ℕ`,
  `900 · 0² - 1 = 0`, so copy `0` is struck by every number.
* (iii) holds for every survivor `j ≥ 1` of the whole copy line (no bound `j < M`), and for every
  downward-closed set of `U_q` gears, not only the prefixes `{g_1, …, g_k}`.
* (iv) holds for every copy `j ≥ 1` (not only survivors); the acting equivalence holds for every
  divisor `h ≥ 2` of the leg, prime or not.
* (v) The stratum statement needs only that no prime lies strictly between `q` and `q'`; its
  conclusion also gives `P j = q'` and that `j` is unique. The converse needs `q'` prime, `q' > q`.
* (vi) holds for every survivor `j ≥ 1` of the whole copy line, in particular inside `[1, M_q)`.
-/

namespace RangeLine

namespace Gen4

/-! ## Notation -/

/-- `j ∈ S_q`: copy `j` survives machine `q`, i.e. no prime `p` with `7 ≤ p ≤ q` strikes it. -/
def Surv (q j : ℕ) : Prop := ∀ p, p.Prime → 7 ≤ p → p ≤ q → ¬ StrikesCopy p j

/-- `g ∈ U_q`: `g` is a prime with `q < g ≤ isqrt(q#)`. -/
def InU (q g : ℕ) : Prop := g.Prime ∧ q < g ∧ g ≤ Nat.sqrt (primorial q)

/-- Copy `j` is revealed (a twin copy): both legs `30 j - 1` and `30 j + 1` are prime. -/
def Twin (j : ℕ) : Prop := (30 * j - 1).Prime ∧ (30 * j + 1).Prime

/-- Copy `j` is a home copy of `g`: `g` is one of the two legs of copy `j`. -/
def Home (g j : ℕ) : Prop := 30 * j - 1 = g ∨ 30 * j + 1 = g

/-- `j ∈ F_g`: `1 ≤ j < M = q#/30`, the least prime factor `P j` of `900 j² - 1` is `g`, and `j`
is not a home copy of `g`. -/
def BlameF (q g j : ℕ) : Prop := 1 ≤ j ∧ j < mirrorM q ∧ P j = g ∧ ¬ Home g j

/-- `H_q = ⌊(isqrt(q#) + 1) / 30⌋`. -/
def cutH (q : ℕ) : ℕ := (Nat.sqrt (primorial q) + 1) / 30

/-- `g` is one of the first `k` gears `g_1 < … < g_k` of `U_q`: a gear of `U_q` equal to
`Nat.nth (InU q) i` for some `i < k` (`Nat.nth (InU q) i` is the `(i+1)`-th smallest element). -/
def FirstGears (q k g : ℕ) : Prop := InU q g ∧ ∃ i, i < k ∧ Nat.nth (InU q) i = g

/-- `q'`: the next gear after `q`, the least prime `p` with `q < p`. -/
noncomputable def nextGear (q : ℕ) : ℕ :=
  @Nat.find (fun p => q + 1 ≤ p ∧ p.Prime) (Classical.decPred _) (Nat.exists_infinite_primes (q + 1))

/-! ## Basic facts -/

/-- A prime dividing a leg of a copy `j ≥ 1` is at least `7` (the legs are `≡ ±1 (mod 30)`). -/
theorem seven_le_of_dvd_leg {p j : ℕ} (hp : p.Prime) (hj : 1 ≤ j)
    (hd : p ∣ 30 * j - 1 ∨ p ∣ 30 * j + 1) : 7 ≤ p := by
  by_contra hlt0
  have hlt : p < 7 := Nat.lt_of_not_le hlt0
  have h2 := hp.two_le
  interval_cases p
  · rcases hd with hd | hd <;> omega
  · rcases hd with hd | hd <;> omega
  · norm_num at hp
  · rcases hd with hd | hd <;> omega
  · norm_num at hp

/-- A divisor of a leg strikes the copy. -/
theorem strikes_of_dvd_leg {p j : ℕ} (hd : p ∣ 30 * j - 1 ∨ p ∣ 30 * j + 1) :
    StrikesCopy p j := by
  unfold StrikesCopy
  rcases hd with h | h
  · exact Dvd.dvd.mul_right h _
  · exact Dvd.dvd.mul_left h _

/-- A prime striking copy `j` divides one of its legs. -/
theorem dvd_leg_of_strikes {p j : ℕ} (hp : p.Prime) (h : StrikesCopy p j) :
    p ∣ 30 * j - 1 ∨ p ∣ 30 * j + 1 :=
  (Nat.Prime.dvd_mul hp).mp h

/-- Every prime factor of a leg of a survivor `j ≥ 1` of machine `q` exceeds `q`. -/
theorem lt_of_dvd_leg {q j p : ℕ} (hj : 1 ≤ j) (hs : Surv q j) (hp : p.Prime)
    (hd : p ∣ 30 * j - 1 ∨ p ∣ 30 * j + 1) : q < p := by
  by_contra hle0
  have hle : p ≤ q := Nat.le_of_not_lt hle0
  exact hs p hp (seven_le_of_dvd_leg hp hj hd) hle (strikes_of_dvd_leg hd)

/-- For `j ≥ 1`, `P j` is prime. -/
theorem P_prime {j : ℕ} (hj : 1 ≤ j) : (P j).Prime := by
  unfold P
  apply Nat.minFac_prime
  have : 1 ≤ j ^ 2 := Nat.one_le_pow _ _ hj
  omega

/-- `P j` strikes copy `j`. -/
theorem P_strikes (j : ℕ) : StrikesCopy (P j) j :=
  (strikesCopy_iff_sq _ _).mpr (Nat.minFac_dvd _)

/-- For `j ≥ 1`, `P j` divides one of the legs of copy `j`. -/
theorem P_dvd_leg {j : ℕ} (hj : 1 ≤ j) : P j ∣ 30 * j - 1 ∨ P j ∣ 30 * j + 1 :=
  dvd_leg_of_strikes (P_prime hj) (P_strikes j)

/-- `P j` is at most every prime striking copy `j`. -/
theorem P_le_of_strikes {p j : ℕ} (hp : p.Prime) (h : StrikesCopy p j) : P j ≤ p :=
  Nat.minFac_le_of_dvd hp.two_le ((strikesCopy_iff_sq p j).mp h)

/-- For `j ≥ 1`: copy `j` survives machine `q` iff `q < P j`. -/
theorem surv_iff_lt_P {q j : ℕ} (hj : 1 ≤ j) : Surv q j ↔ q < P j := by
  constructor
  · intro hs
    exact lt_of_dvd_leg hj hs (P_prime hj) (P_dvd_leg hj)
  · intro hlt p hp _ hpq hst
    have := P_le_of_strikes hp hst
    omega

/-- For `q ≥ 5` and `j ≥ 1`: `j ∈ S_q` (no prime `7 ≤ p ≤ q` strikes copy `j`) iff
`gcd(900 j² - 1, M) = 1` with `M = q#/30`. -/
theorem surv_iff_coprime {q j : ℕ} (hq : 5 ≤ q) (hj : 1 ≤ j) :
    Surv q j ↔ Nat.Coprime (900 * j ^ 2 - 1) (mirrorM q) := by
  have hM : 30 * mirrorM q = primorial q := thirty_mul_mirrorM hq
  constructor
  · intro hs
    apply Nat.coprime_of_dvd
    intro p hp hpa hpM
    have hpQ : p ∣ primorial q := by
      rw [← hM]
      exact Dvd.dvd.mul_left hpM 30
    have hpq : p ≤ q := hp.dvd_primorial_iff.mp hpQ
    have hst : StrikesCopy p j := (strikesCopy_iff_sq p j).mpr hpa
    exact hs p hp (seven_le_of_dvd_leg hp hj (dvd_leg_of_strikes hp hst)) hpq hst
  · intro hc p hp hp7 hpq hst
    have hpa : p ∣ 900 * j ^ 2 - 1 := (strikesCopy_iff_sq p j).mp hst
    have hpQ : p ∣ 30 * mirrorM q := by
      rw [hM]
      exact hp.dvd_primorial_iff.mpr hpq
    have hcop : Nat.Coprime p 30 := (Nat.Prime.coprime_iff_not_dvd hp).mpr (not_dvd_thirty hp hp7)
    have hpM : p ∣ mirrorM q := (Nat.Coprime.dvd_mul_left hcop).mp hpQ
    have h1 : p ∣ Nat.gcd (900 * j ^ 2 - 1) (mirrorM q) := Nat.dvd_gcd hpa hpM
    rw [Nat.Coprime.gcd_eq_one hc] at h1
    exact hp.one_lt.ne' (Nat.dvd_one.mp h1)

/-- A revealed copy `j ≥ 1` has `P j = 30 j - 1`: its least prime factor is its own lower leg. -/
theorem P_eq_of_twin {j : ℕ} (hj : 1 ≤ j) (ht : Twin j) : P j = 30 * j - 1 := by
  obtain ⟨ha, hb⟩ := ht
  have hle : P j ≤ 30 * j - 1 := P_le_of_strikes ha (strikes_of_dvd_leg (Or.inl dvd_rfl))
  rcases P_dvd_leg hj with h | h
  · exact (Nat.prime_dvd_prime_iff_eq (P_prime hj) ha).mp h
  · have := (Nat.prime_dvd_prime_iff_eq (P_prime hj) hb).mp h
    omega

/-- A gear never acts on its home copy: if `g` is a leg of copy `j ≥ 1`, then `g² > 30 j + 1`. -/
theorem home_not_acts {g j : ℕ} (hj : 1 ≤ j) (hh : Home g j) : ¬ Acts g j := by
  unfold Acts
  have hg : 30 * j - 1 ≤ g := by
    rcases hh with h | h <;> omega
  have h29 : 29 ≤ g := by omega
  have : 29 * g ≤ g ^ 2 := by nlinarith
  omega

/-- The prime-factor bound below `q#`: for `j ≥ 1` not revealed with `30 j + 1 ≤ q#`,
`P j ≤ isqrt(q#)`. -/
theorem P_le_sqrt {q j : ℕ} (hj : 1 ≤ j) (hjQ : 30 * j + 1 ≤ primorial q) (hnt : ¬ Twin j) :
    P j ≤ Nat.sqrt (primorial q) :=
  Nat.le_sqrt'.mpr (le_trans ((blamed_iff_minFac hj).mp hnt) hjQ)

/-- `j < M = q#/30` gives `30 j + 1 ≤ q#`. -/
theorem le_primorial_of_lt_mirrorM {q j : ℕ} (hjM : j < mirrorM q) : 30 * j + 1 ≤ primorial q := by
  unfold mirrorM at hjM
  omega

/-- **Shadowing (row 1, W1).** Let `j ≥ 1` be a survivor of machine `q`, `L` a leg of copy `j`,
and `g` any number with `g ∣ L` and `L < g²`. Then `L = g`, or `h = minFac L` satisfies
`q < h < g` and `h` acts at `j`. -/
theorem shadowing {q j L g : ℕ} (hj : 1 ≤ j) (hs : Surv q j)
    (hL : L = 30 * j - 1 ∨ L = 30 * j + 1) (hgL : g ∣ L) (hLg : L < g ^ 2) :
    L = g ∨ (q < Nat.minFac L ∧ Nat.minFac L < g ∧ Acts (Nat.minFac L) j) := by
  obtain ⟨c, hc⟩ := hgL
  have hL29 : 29 ≤ L := by rcases hL with rfl | rfl <;> omega
  have hLj : L ≤ 30 * j + 1 := by rcases hL with rfl | rfl <;> omega
  have hc0 : 1 ≤ c := by
    rcases Nat.eq_zero_or_pos c with h0 | h0
    · rw [h0, mul_zero] at hc
      omega
    · exact h0
  have hcg : c < g := by
    have : g * c < g * g := by rw [← hc, ← pow_two]; exact hLg
    exact Nat.lt_of_mul_lt_mul_left this
  rcases Nat.lt_or_ge c 2 with hc1 | hc2
  · left
    have : c = 1 := by omega
    rw [hc, this, mul_one]
  · right
    have hg2 : 2 ≤ g := by omega
    have hLnp : ¬ L.Prime := by
      rw [hc]
      exact Nat.not_prime_mul (by omega) (by omega)
    have hmp : (Nat.minFac L).Prime := Nat.minFac_prime (by omega)
    have hmdvd : Nat.minFac L ∣ L := Nat.minFac_dvd L
    have hmc : Nat.minFac L ≤ c := Nat.minFac_le_of_dvd hc2 ⟨g, by rw [hc, mul_comm]⟩
    have hmq : q < Nat.minFac L := by
      apply lt_of_dvd_leg hj hs hmp
      rcases hL with rfl | rfl
      · exact Or.inl hmdvd
      · exact Or.inr hmdvd
    have hsq : (Nat.minFac L) ^ 2 ≤ L := Nat.minFac_sq_le_self (by omega) hLnp
    refine ⟨hmq, by omega, ?_⟩
    unfold Acts
    omega

/-! ## (i) The blame partition of the non-revealed survivors -/

/-- **(i), pointwise.** For every `q` and every survivor `j` of machine `q` with `1 ≤ j < M`:
`j` is not revealed iff `j ∈ F_g` for some `g ∈ U_q` (namely `g = P j`). -/
theorem nonrevealed_iff_blameF {q j : ℕ} (hj : 1 ≤ j) (hjM : j < mirrorM q) (hs : Surv q j) :
    ¬ Twin j ↔ ∃ g, InU q g ∧ BlameF q g j := by
  constructor
  · intro hnt
    refine ⟨P j, ⟨P_prime hj, (surv_iff_lt_P hj).mp hs,
      P_le_sqrt hj (le_primorial_of_lt_mirrorM hjM) hnt⟩, hj, hjM, rfl, ?_⟩
    intro hh
    exact home_not_acts hj hh ((blamed_iff_minFac hj).mp hnt)
  · rintro ⟨g, _, _, _, hPg, hnh⟩ ht
    exact hnh (Or.inl ((P_eq_of_twin hj ht).symm.trans hPg))

/-- Every member of `F_g` is not revealed: a revealed copy has `P j` equal to its own lower leg,
a home copy of `P j`. -/
theorem blameF_not_twin {q g j : ℕ} (hF : BlameF q g j) : ¬ Twin j := by
  obtain ⟨hj, _, hPg, hnh⟩ := hF
  intro ht
  exact hnh (Or.inl ((P_eq_of_twin hj ht).symm.trans hPg))

/-- For `g > q` (in particular `g ∈ U_q`), every member of `F_g` is a survivor of machine `q`. -/
theorem blameF_surv {q g j : ℕ} (hg : q < g) (hF : BlameF q g j) : Surv q j := by
  obtain ⟨hj, _, hPg, _⟩ := hF
  rw [surv_iff_lt_P hj, hPg]
  exact hg

/-- **(i), acting.** The gear `g` acts on every member of `F_g`. -/
theorem blameF_acts {q g j : ℕ} (hF : BlameF q g j) : Acts g j := by
  have h := (blamed_iff_minFac hF.1).mp (blameF_not_twin hF)
  rw [hF.2.2.1] at h
  exact h

/-- **(i), set form.** For every `q`, the non-revealed survivors of `[1, M)` are the union of the
sets `F_g` over `g ∈ U_q`. -/
theorem nonrevealed_eq_iUnion (q : ℕ) :
    {j | 1 ≤ j ∧ j < mirrorM q ∧ Surv q j ∧ ¬ Twin j} =
      ⋃ g ∈ {g | InU q g}, {j | BlameF q g j} := by
  ext j
  simp only [Set.mem_ofPred_eq, Set.mem_iUnion, exists_prop]
  constructor
  · rintro ⟨hj, hjM, hs, hnt⟩
    exact (nonrevealed_iff_blameF hj hjM hs).mp hnt
  · rintro ⟨g, hU, hF⟩
    exact ⟨hF.1, hF.2.1, blameF_surv hU.2.1 hF, blameF_not_twin hF⟩

/-- **(i), disjointness.** For every `q`, the sets `F_g` (`g ∈ U_q`) are pairwise disjoint, so
the union in `nonrevealed_eq_iUnion` is a disjoint union. -/
theorem blameF_pairwiseDisjoint (q : ℕ) :
    Set.PairwiseDisjoint {g | InU q g} (fun g => {j | BlameF q g j}) := by
  intro g _ g' _ hne
  refine Set.disjoint_left.mpr ?_
  intro j hj hj'
  have h1 : BlameF q g j := hj
  have h2 : BlameF q g' j := hj'
  exact hne (h1.2.2.1.symm.trans h2.2.2.1)

/-! ## (ii) Revealed survivors -/

/-- **(ii).** For every `q` and every survivor `j` of machine `q` with `1 ≤ j < M`: `j` is
revealed iff every `U_q` gear striking `j` is one of its own legs. -/
theorem revealed_iff_strikers_home {q j : ℕ} (hj : 1 ≤ j) (hjM : j < mirrorM q) (hs : Surv q j) :
    Twin j ↔ ∀ g, InU q g → StrikesCopy g j → Home g j := by
  constructor
  · rintro ⟨ha, hb⟩ g hg hst
    rcases dvd_leg_of_strikes hg.1 hst with h | h
    · exact Or.inl ((Nat.prime_dvd_prime_iff_eq hg.1 ha).mp h).symm
    · exact Or.inr ((Nat.prime_dvd_prime_iff_eq hg.1 hb).mp h).symm
  · intro hall
    by_contra hnt
    obtain ⟨g, hU, hF⟩ := (nonrevealed_iff_blameF hj hjM hs).mp hnt
    have hst : StrikesCopy g j := by
      rw [← hF.2.2.1]
      exact P_strikes j
    exact hF.2.2.2 (hall g hU hst)

/-- For every `q` and every copy `j > H_q`, both legs exceed `isqrt(q#)`:
`isqrt(q#) < 30 j - 1`. -/
theorem legs_gt_sqrt_of_gt_cutH {q j : ℕ} (hH : cutH q < j) :
    Nat.sqrt (primorial q) < 30 * j - 1 := by
  unfold cutH at hH
  omega

/-- **(ii), strict form.** For every `q` and every survivor `j` of machine `q` with
`H_q < j < M`: `j` is revealed iff no `U_q` gear strikes `j`. -/
theorem revealed_iff_no_strike {q j : ℕ} (hH : cutH q < j) (hjM : j < mirrorM q)
    (hs : Surv q j) : Twin j ↔ ∀ g, InU q g → ¬ StrikesCopy g j := by
  have hj : 1 ≤ j := by omega
  have hlt := legs_gt_sqrt_of_gt_cutH hH
  rw [revealed_iff_strikers_home hj hjM hs]
  constructor
  · intro h g hg hst
    have hgs := hg.2.2
    rcases h g hg hst with e | e <;> omega
  · intro h g hg hst
    exact absurd hst (h g hg)

/-! ## (iii) Joint redundancy -/

/-- **(iii), general form.** For every `q`, every survivor `j ≥ 1` of machine `q`, and every set
`D` of gears of `U_q` that is closed downward inside `U_q` (`g ∈ D`, `h ∈ U_q`, `h < g` give
`h ∈ D`): some gear of `D` strikes `j` iff some gear of `D` strikes `j` and acts at `j`, or `j` is
a home copy of a gear of `D`. -/
theorem joint_redundancy {q j : ℕ} (D : ℕ → Prop) (hDU : ∀ g, D g → InU q g)
    (hDown : ∀ g h, D g → InU q h → h < g → D h) (hj : 1 ≤ j) (hs : Surv q j) :
    (∃ g, D g ∧ StrikesCopy g j) ↔
      (∃ g, D g ∧ StrikesCopy g j ∧ Acts g j) ∨ (∃ g, D g ∧ Home g j) := by
  constructor
  · rintro ⟨g, hD, hst⟩
    by_cases hact : Acts g j
    · exact Or.inl ⟨g, hD, hst, hact⟩
    · have hU := hDU g hD
      obtain ⟨L, hL, hgL⟩ : ∃ L, (L = 30 * j - 1 ∨ L = 30 * j + 1) ∧ g ∣ L := by
        rcases dvd_leg_of_strikes hU.1 hst with h | h
        · exact ⟨_, Or.inl rfl, h⟩
        · exact ⟨_, Or.inr rfl, h⟩
      have hLg : L < g ^ 2 := by
        unfold Acts at hact
        rcases hL with rfl | rfl <;> omega
      rcases shadowing hj hs hL hgL hLg with hLe | ⟨hq, hlt, hacts⟩
      · right
        refine ⟨g, hD, ?_⟩
        rcases hL with rfl | rfl
        · exact Or.inl hLe
        · exact Or.inr hLe
      · left
        have hmp : (Nat.minFac L).Prime :=
          Nat.minFac_prime (by rcases hL with rfl | rfl <;> omega)
        have hgs := hU.2.2
        refine ⟨Nat.minFac L, hDown g _ hD ⟨hmp, hq, by omega⟩ hlt, ?_, hacts⟩
        apply strikes_of_dvd_leg
        rcases hL with rfl | rfl
        · exact Or.inl (Nat.minFac_dvd _)
        · exact Or.inr (Nat.minFac_dvd _)
  · rintro (⟨g, hD, hst, _⟩ | ⟨g, hD, hh⟩)
    · exact ⟨g, hD, hst⟩
    · refine ⟨g, hD, strikes_of_dvd_leg ?_⟩
      rcases hh with h | h
      · exact Or.inl ⟨1, by omega⟩
      · exact Or.inr ⟨1, by omega⟩

/-- **(iii), threshold form.** For every `q`, every bound `X` and every survivor `j ≥ 1` of
machine `q`, with `D = {g ∈ U_q : g ≤ X}`: some gear of `D` strikes `j` iff some gear of `D`
strikes and acts at `j` or `j` is a home copy of a gear of `D`. -/
theorem joint_redundancy_le {q j X : ℕ} (hj : 1 ≤ j) (hs : Surv q j) :
    (∃ g, (InU q g ∧ g ≤ X) ∧ StrikesCopy g j) ↔
      (∃ g, (InU q g ∧ g ≤ X) ∧ StrikesCopy g j ∧ Acts g j) ∨
        (∃ g, (InU q g ∧ g ≤ X) ∧ Home g j) :=
  joint_redundancy (fun g => InU q g ∧ g ≤ X) (fun _ hg => hg.1)
    (fun _ _ hg hh hlt => ⟨hh, by have := hg.2; omega⟩) hj hs

/-- `U_q` is a finite set (it lies in `[0, isqrt(q#)]`). -/
theorem finite_U (q : ℕ) : (Set.ofPred (InU q)).Finite :=
  (Set.finite_Iic (Nat.sqrt (primorial q))).subset (fun _ hg => hg.2.2)

/-- The first `k` gears of `U_q` are closed downward inside `U_q`: if `g` is among
`g_1, …, g_k` and `h ∈ U_q` with `h < g`, then `h` is among `g_1, …, g_k`. -/
theorem firstGears_down {q k g h : ℕ} (hg : FirstGears q k g) (hh : InU q h) (hlt : h < g) :
    FirstGears q k h := by
  obtain ⟨_, i, hik, hi⟩ := hg
  obtain ⟨i', hi'c, hi'⟩ := Nat.exists_lt_card_finite_nth_eq (finite_U q) hh
  have hlt' : Nat.nth (InU q) i' < Nat.nth (InU q) i := by rw [hi, hi']; exact hlt
  have hii : i' < i := Nat.lt_of_nth_lt_nth_of_lt_card (finite_U q) hlt' hi'c
  exact ⟨hh, i', by omega, hi'⟩

/-- **(iii), for the gears `g_1 < g_2 < …` of `U_q`.** For every `q`, every `k` and every
survivor `j ≥ 1` of machine `q`: some gear among `g_1, …, g_k` strikes `j` iff some gear among
them strikes and acts at `j`, or `j` is a home copy of one of them. -/
theorem joint_redundancy_first {q j k : ℕ} (hj : 1 ≤ j) (hs : Surv q j) :
    (∃ g, FirstGears q k g ∧ StrikesCopy g j) ↔
      (∃ g, FirstGears q k g ∧ StrikesCopy g j ∧ Acts g j) ∨
        (∃ g, FirstGears q k g ∧ Home g j) :=
  joint_redundancy (FirstGears q k) (fun _ hg => hg.1)
    (fun _ _ hg hh hlt => firstGears_down hg hh hlt) hj hs

/-- **(iii), set identity.** For every `q` and every `k`, on the copies `j ≥ 1`:
`S_q ∩ ⋃_{i≤k} Str(g_i) = S_q ∩ (⋃_{i≤k} ActStr(g_i) ∪ Home(g_1..g_k))`. -/
theorem joint_redundancy_set (q k : ℕ) :
    {j | 1 ≤ j ∧ Surv q j} ∩ (⋃ g ∈ {g | FirstGears q k g}, {j | StrikesCopy g j}) =
      {j | 1 ≤ j ∧ Surv q j} ∩
        ((⋃ g ∈ {g | FirstGears q k g}, {j | StrikesCopy g j ∧ Acts g j}) ∪
          {j | ∃ g, FirstGears q k g ∧ Home g j}) := by
  ext j
  simp only [Set.mem_inter_iff, Set.mem_union, Set.mem_ofPred_eq, Set.mem_iUnion, exists_prop]
  constructor
  · rintro ⟨⟨hj, hs⟩, h⟩
    exact ⟨⟨hj, hs⟩, (joint_redundancy_first hj hs).mp h⟩
  · rintro ⟨⟨hj, hs⟩, h⟩
    exact ⟨⟨hj, hs⟩, (joint_redundancy_first hj hs).mpr h⟩

/-! ## (iv) Legs: acting per leg, and at most one non-acting prime factor -/

/-- **(iv), acting per leg.** For every copy `j ≥ 1`, every leg `L` of copy `j` and every
`h ≥ 2` dividing `L`: `h² ≤ 30 j + 1` iff `h² ≤ L`. -/
theorem acts_iff_sq_le_leg {j L h : ℕ} (hj : 1 ≤ j) (hL : L = 30 * j - 1 ∨ L = 30 * j + 1)
    (hh : 2 ≤ h) (hd : h ∣ L) : Acts h j ↔ h ^ 2 ≤ L := by
  unfold Acts
  rcases hL with rfl | rfl
  · constructor
    · intro hle
      have hne0 : h ^ 2 ≠ 30 * j := by
        intro he
        have h1 : h ∣ 30 * j := by
          rw [← he]
          exact dvd_pow_self h two_ne_zero
        have h3 : h ∣ 30 * j - (30 * j - 1) := Nat.dvd_sub h1 hd
        rw [show 30 * j - (30 * j - 1) = 1 by omega] at h3
        have := Nat.le_of_dvd one_pos h3
        omega
      have hne1 : h ^ 2 ≠ 30 * j + 1 := by
        intro he
        have h1 : h ∣ 30 * j + 1 := by
          rw [← he]
          exact dvd_pow_self h two_ne_zero
        have h3 : h ∣ 30 * j + 1 - (30 * j - 1) := Nat.dvd_sub h1 hd
        rw [show 30 * j + 1 - (30 * j - 1) = 2 by omega] at h3
        have h4 : h ≤ 2 := Nat.le_of_dvd two_pos h3
        have h5 : h = 2 := by omega
        subst h5
        omega
      omega
    · intro hle
      omega
  · exact Iff.rfl

/-- **(iv), cofactor form.** For every copy `j ≥ 1`, every leg `L = h · c` of copy `j` with
`h ≥ 2`: `h` acts at `j` iff `h ≤ c`. -/
theorem acts_iff_le_cofactor {j L h c : ℕ} (hj : 1 ≤ j) (hL : L = 30 * j - 1 ∨ L = 30 * j + 1)
    (hh : 2 ≤ h) (hc : L = h * c) : Acts h j ↔ h ≤ c := by
  rw [acts_iff_sq_le_leg hj hL hh ⟨c, hc⟩, hc, pow_two]
  exact Nat.mul_le_mul_left_iff (by omega)

/-- **(iv), at most one non-acting prime factor.** For every copy `j ≥ 1` and every leg `L` of
copy `j`: two primes dividing `L` that both fail to act at `j` are equal. -/
theorem nonacting_factor_unique {j L h₁ h₂ : ℕ} (hj : 1 ≤ j)
    (hL : L = 30 * j - 1 ∨ L = 30 * j + 1) (hp₁ : h₁.Prime) (hp₂ : h₂.Prime)
    (hd₁ : h₁ ∣ L) (hd₂ : h₂ ∣ L) (hn₁ : ¬ Acts h₁ j) (hn₂ : ¬ Acts h₂ j) : h₁ = h₂ := by
  by_contra hne
  have hcop : Nat.Coprime h₁ h₂ := (Nat.coprime_primes hp₁ hp₂).mpr hne
  have hmul : h₁ * h₂ ∣ L := Nat.Coprime.mul_dvd_of_dvd_of_dvd hcop hd₁ hd₂
  have hLpos : 0 < L := by rcases hL with rfl | rfl <;> omega
  have hle : h₁ * h₂ ≤ L := Nat.le_of_dvd hLpos hmul
  have hL1 : L ≤ 30 * j + 1 := by rcases hL with rfl | rfl <;> omega
  unfold Acts at hn₁ hn₂
  have hn₁' : 30 * j + 1 < h₁ ^ 2 := Nat.lt_of_not_le hn₁
  have hn₂' : 30 * j + 1 < h₂ ^ 2 := Nat.lt_of_not_le hn₂
  rcases le_total h₁ h₂ with h | h
  · nlinarith [Nat.mul_le_mul_left h₁ h]
  · nlinarith [Nat.mul_le_mul_left h₂ h]

/-! ## (v) The bottom strata -/

/-- **(v), first stratum.** For every `q` and every survivor `j ≥ 1` of machine `q` with
`30 j + 1 ≤ q²`: `j` is revealed. -/
theorem revealed_of_sq_le {q j : ℕ} (hj : 1 ≤ j) (hs : Surv q j) (hjq : 30 * j + 1 ≤ q ^ 2) :
    Twin j := by
  by_contra hnt
  have h1 := (blamed_iff_minFac hj).mp hnt
  have h2 := (surv_iff_lt_P hj).mp hs
  have h3 : (q + 1) ^ 2 ≤ (P j) ^ 2 := Nat.pow_le_pow_left h2 2
  nlinarith

/-- **(v), the stratum `30 j + 1 ≤ q'²`.** For every `q` and every `q'` such that no prime lies
in `(q, q')` (every prime `p > q` has `p ≥ q'`), a survivor `j ≥ 1` of machine `q` with
`30 j + 1 ≤ q'²` that is not revealed has `30 j + 1 = q'²` (so `j = (q'² - 1)/30`), its lower leg
`30 j - 1 = q'² - 2` is prime, and `P j = q'`. -/
theorem stratum_nonrevealed {q q' j : ℕ} (hnext : ∀ p, p.Prime → q < p → q' ≤ p) (hj : 1 ≤ j)
    (hs : Surv q j) (hjq : 30 * j + 1 ≤ q' ^ 2) (hnt : ¬ Twin j) :
    30 * j + 1 = q' ^ 2 ∧ j = (q' ^ 2 - 1) / 30 ∧ (30 * j - 1).Prime ∧ P j = q' := by
  have h1 := (blamed_iff_minFac hj).mp hnt
  have h2 : q' ≤ P j := hnext _ (P_prime hj) ((surv_iff_lt_P hj).mp hs)
  have h3 : q' ^ 2 ≤ (P j) ^ 2 := Nat.pow_le_pow_left h2 2
  have heq : 30 * j + 1 = q' ^ 2 := by omega
  have hPq : P j = q' := by
    by_contra hne
    have h4 : q' + 1 ≤ P j := by omega
    have h5 : (q' + 1) ^ 2 ≤ (P j) ^ 2 := Nat.pow_le_pow_left h4 2
    nlinarith
  refine ⟨heq, by omega, ?_, hPq⟩
  by_contra hnp
  have hm := Nat.minFac_prime (show 30 * j - 1 ≠ 1 by omega)
  have hmq : q < Nat.minFac (30 * j - 1) :=
    lt_of_dvd_leg hj hs hm (Or.inl (Nat.minFac_dvd _))
  have hmq' := hnext _ hm hmq
  have hsq := Nat.minFac_sq_le_self (show 0 < 30 * j - 1 by omega) hnp
  have : q' ^ 2 ≤ (Nat.minFac (30 * j - 1)) ^ 2 := Nat.pow_le_pow_left hmq' 2
  omega

/-- **(v), the exception occurs.** For every `q` and every prime `q' > q` with
`q'² ≡ 1 (mod 30)` and `q'² - 2` prime, the copy `j = (q'² - 1)/30` has `j ≥ 1`, survives machine
`q`, has `30 j + 1 = q'²`, and is not revealed. -/
theorem stratum_exception {q q' : ℕ} (hq' : q'.Prime) (hqq' : q < q') (hmod : q' ^ 2 % 30 = 1)
    (hpr : (q' ^ 2 - 2).Prime) :
    1 ≤ (q' ^ 2 - 1) / 30 ∧ Surv q ((q' ^ 2 - 1) / 30) ∧
      30 * ((q' ^ 2 - 1) / 30) + 1 = q' ^ 2 ∧ ¬ Twin ((q' ^ 2 - 1) / 30) := by
  have hq2 := hq'.two_le
  have hsq4 : 4 ≤ q' ^ 2 := by nlinarith
  have hsq : q' ≤ q' ^ 2 - 2 := by
    have : q' + 2 ≤ q' ^ 2 := by nlinarith
    omega
  set j := (q' ^ 2 - 1) / 30 with hjdef
  have hj1 : 30 * j + 1 = q' ^ 2 := by omega
  have hj0 : 1 ≤ j := by omega
  have hjm : 30 * j - 1 = q' ^ 2 - 2 := by omega
  refine ⟨hj0, ?_, hj1, ?_⟩
  · intro p hp _ hpq hst
    rcases dvd_leg_of_strikes hp hst with h | h
    · rw [hjm] at h
      have := (Nat.prime_dvd_prime_iff_eq hp hpr).mp h
      omega
    · rw [hj1] at h
      have := (Nat.prime_dvd_prime_iff_eq hp hq').mp (hp.dvd_of_dvd_pow h)
      omega
  · rintro ⟨_, hb⟩
    rw [hj1, pow_two] at hb
    exact Nat.not_prime_mul (by omega) (by omega) hb

/-- **(v), the stratum law as an equivalence.** For every `q` and every prime `q' > q` such that
no prime lies in `(q, q')`: some survivor `j ≥ 1` of machine `q` with `30 j + 1 ≤ q'²` is not
revealed iff `q'² ≡ 1 (mod 30)` and `q'² - 2` is prime (and then it is `j = (q'² - 1)/30`,
`stratum_nonrevealed`). -/
theorem stratum_iff {q q' : ℕ} (hq' : q'.Prime) (hqq' : q < q')
    (hnext : ∀ p, p.Prime → q < p → q' ≤ p) :
    (∃ j, 1 ≤ j ∧ Surv q j ∧ 30 * j + 1 ≤ q' ^ 2 ∧ ¬ Twin j) ↔
      (q' ^ 2 % 30 = 1 ∧ (q' ^ 2 - 2).Prime) := by
  constructor
  · rintro ⟨j, hj, hs, hjq, hnt⟩
    obtain ⟨heq, _, hpr, _⟩ := stratum_nonrevealed hnext hj hs hjq hnt
    have hm : 30 * j - 1 = q' ^ 2 - 2 := by omega
    refine ⟨by omega, ?_⟩
    rw [← hm]
    exact hpr
  · rintro ⟨hmod, hpr⟩
    obtain ⟨hj, hs, heq, hnt⟩ := stratum_exception hq' hqq' hmod hpr
    exact ⟨_, hj, hs, heq.le, hnt⟩

/-! ## The next gear -/

/-- `nextGear q` is prime and exceeds `q`. -/
theorem nextGear_spec (q : ℕ) : q < nextGear q ∧ (nextGear q).Prime := by
  have := @Nat.find_spec (fun p => q + 1 ≤ p ∧ p.Prime) (Classical.decPred _)
    (Nat.exists_infinite_primes (q + 1))
  exact ⟨by unfold nextGear; omega, this.2⟩

/-- No prime lies strictly between `q` and `nextGear q`: every prime `p > q` has
`nextGear q ≤ p`. -/
theorem nextGear_le {q p : ℕ} (hp : p.Prime) (hqp : q < p) : nextGear q ≤ p :=
  @Nat.find_min' (fun p => q + 1 ≤ p ∧ p.Prime) (Classical.decPred _)
    (Nat.exists_infinite_primes (q + 1)) p ⟨hqp, hp⟩

/-- **(v) with `q' = nextGear q`.** For every `q`: some survivor `j ≥ 1` of machine `q` with
`30 j + 1 ≤ q'²` is not revealed iff `q'² ≡ 1 (mod 30)` and `q'² - 2` is prime. -/
theorem stratum_iff_nextGear (q : ℕ) :
    (∃ j, 1 ≤ j ∧ Surv q j ∧ 30 * j + 1 ≤ nextGear q ^ 2 ∧ ¬ Twin j) ↔
      (nextGear q ^ 2 % 30 = 1 ∧ (nextGear q ^ 2 - 2).Prime) :=
  stratum_iff (nextGear_spec q).2 (nextGear_spec q).1 (fun _ hp hqp => nextGear_le hp hqp)

/-! ## (vi) The next gear strikes survivors only by acting, except at home -/

/-- **(vi).** For every `q` and every prime `q'` such that no prime lies in `(q, q')`: if `q'`
strikes a survivor `j ≥ 1` of machine `q` (anywhere on the copy line, in particular inside
`[1, M_q)`), then `q'` acts at `j`, or `j` is the home copy of `q'`. -/
theorem strike_acts_or_home {q q' j : ℕ} (hnext : ∀ p, p.Prime → q < p → q' ≤ p)
    (hq' : q'.Prime) (hj : 1 ≤ j) (hs : Surv q j) (hst : StrikesCopy q' j) :
    Acts q' j ∨ Home q' j := by
  by_cases hact : Acts q' j
  · exact Or.inl hact
  · right
    obtain ⟨L, hL, hgL⟩ : ∃ L, (L = 30 * j - 1 ∨ L = 30 * j + 1) ∧ q' ∣ L := by
      rcases dvd_leg_of_strikes hq' hst with h | h
      · exact ⟨_, Or.inl rfl, h⟩
      · exact ⟨_, Or.inr rfl, h⟩
    have hLg : L < q' ^ 2 := by
      unfold Acts at hact
      rcases hL with rfl | rfl <;> omega
    rcases shadowing hj hs hL hgL hLg with hLe | ⟨hq, hlt, _⟩
    · rcases hL with rfl | rfl
      · exact Or.inl hLe
      · exact Or.inr hLe
    · have hmp : (Nat.minFac L).Prime :=
        Nat.minFac_prime (by rcases hL with rfl | rfl <;> omega)
      have := hnext _ hmp hq
      omega

/-- **(vi) with `q' = nextGear q`.** For every `q`: the next gear strikes a survivor `j ≥ 1` of
machine `q` only by acting, or at its home copy. -/
theorem nextGear_strike_acts_or_home {q j : ℕ} (hj : 1 ≤ j) (hs : Surv q j)
    (hst : StrikesCopy (nextGear q) j) : Acts (nextGear q) j ∨ Home (nextGear q) j :=
  strike_acts_or_home (fun _ hp hqp => nextGear_le hp hqp) (nextGear_spec q).2 hj hs hst

end Gen4

end RangeLine
