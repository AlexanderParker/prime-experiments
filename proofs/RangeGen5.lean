import Mathlib.NumberTheory.Bertrand
import Mathlib.NumberTheory.Primorial
import Mathlib.Tactic
import RangeActPair
import RangeMirror

/-!
# Acting against acting-free on the whole line, for every cut and every machine

Copy `k` is the pair of legs `30 k - 1`, `30 k + 1`; a number `g` strikes copy `k` when it divides
`(30 k - 1)(30 k + 1)` (`RangeCentre.StrikesCopy`), and `g` acts at copy `k` when
`g² ≤ 30 k + 1` (`RangeActPair.Acts`).

Everything is stated first for an arbitrary cut `X` (any natural number), then read at the cut
`X = isqrt(q#)` of machine `q` (any natural number `q`; no primality and no size bound is needed).

* `G_X` (`InGears X g`): the primes `g` with `7 ≤ g ≤ X`; `P_X = ∏ G_X` (`gearProd X`).
* `E_X(k)` (`ActClear X k`): no acting gear of `G_X` strikes copy `k`.
* `F_X(k)` (`FreeClear X k`): no gear of `G_X` strikes copy `k` (acting dropped).
* `LegsPrime k`: both legs of copy `k` are prime.
* `cutSqrt q = Nat.sqrt (q#)`; `M = q#/30` is `RangeMirror.mirrorM q`.

This file proves, for every `X` (and every `k`, `N` as stated):

* Periodicity: `strikesCopy_add_dvd` (a strike on `k ≥ 1` is a strike on `k + N` whenever
  `g ∣ N`), `freeClear_add_iff` (`F_X` has period `P_X` on `k ≥ 1`).
* Size: `thirty_mul_gearProd` (`30 P_X = X#` for `X ≥ 5`) and, from Bertrand's postulate,
  `two_sq_lt_thirty_gearProd`: `2 g² < 30 P_X` for every `g ∈ G_X`. Hence `acts_add_gearProd`:
  every gear of `G_X` acts at every copy `k + P_X`.
* `actClear_add_gearProd_iff`: for `k ≥ 1`, `E_X(k + P_X) ⇔ F_X(k)`: one period up, the acting
  pattern is the acting-free pattern.
* (a) `actClear_shift_iff_of_acts`: if every gear of `G_X` acts at `k`, then
  `E_X(k + N) ⇔ E_X(k)` for every multiple `N` of `P_X`; `actClear_period_of_sq_le` is the case
  `X² ≤ 30 k + 1`.
* (b) `actClear_of_actClear_add`: `E_X(k + N) ⇒ E_X(k)` for every `k` and every multiple `N` of
  `P_X`.
* (c) `actClear_jump_iff`: for every `k`,
  `E_X(k) ∧ ¬ E_X(k + P_X) ⇔` both legs of `k` prime `∧ 30 k - 1 ≤ X`.
* `acting_free_differ_iff`, `patterns_differ_iff`: on copies `k ≥ 1` the acting and acting-free
  deletion patterns of `G_X` differ exactly on the copies with both legs prime and a leg `≤ X`;
  `actClear_of_freeClear`: no copy is acting-struck but acting-free unstruck.
* Revealed status: `actClear_iff_legsPrime` (with acting, revealed `⇔` both legs prime) and
  `freeClear_iff` (acting-free, revealed `⇔` both legs prime and above `X`), for
  `1 ≤ k` with `30 k + 1 < (X + 1)²`.
* No period: `lower_leg_shift` (copy `j + (30 j - 1) N` has lower leg `(30 j - 1)(1 + 30 N)`),
  `not_legsPrime_shift`, `legsPrime_no_period` (for every `N ≥ 1`, "both legs prime" is not
  `N`-periodic on the copies `j ≥ 1`).

At the cut `X = isqrt(q#)`, for every `q`:

* `acts_of_mirrorM_le`: every gear of `G` acts at every copy `k ≥ M`.
* `actClear_period_range` (a), `actClear_of_actClear_add_range` (b), `actClear_jump_iff_range` (c),
  `patterns_differ_iff_range`.
* `actClear_iff_legsPrime_range`, `freeClear_iff_range`: on the copies `1 ≤ j` with
  `30 j + 1 ≤ q#`, revealed with acting `⇔` both legs prime, and acting-free revealed `⇔` both legs
  prime and both legs above `√(q#)`.
* `lower_leg_shift_range`, `legsPrime_no_period_range`: the copy `j + p P` has lower leg
  `p (1 + 30 P)` when `p = 30 j - 1`, and "both legs prime" has no period `P`.
* `fixed_cut_copy`, `fixed_cut`, `fixed_cut_sq`: "some range copy (`q < 30 j - 1`,
  `30 j + 1 ≤ q#`) is struck by no gear of `[7, √(q#)]`" is exactly "some copy with
  `30 j + 1 ≤ q#` has both legs prime and its lower leg above `isqrt(q#)`" (the window statement at
  the cut `X = isqrt(q#)`).

Corrections to the requested wording (each is a sharpening; nothing requested is dropped):

* No `q ≥ 7` hypothesis is needed anywhere: (a), (b), (c), the difference law and the fixed cut
  hold for every `q`, and in the `X` form for every cut `X`.
* (b) holds for every `k` (also `k = 0`) and for every multiple `N` of `P`; (c) holds for every
  `k` (also `k = 0`, where both sides are false).
* (a) holds at every `k` at which all gears of `G` act, i.e. whenever `X² ≤ 30 k + 1`; `k ≥ M` is
  one such range.
* "Acting-free revealed `⇔` both legs prime and both above `√(q#)`" needs the copy to lie in the
  range, `30 j + 1 ≤ q#` (in general: `30 k + 1 < (X + 1)²`). Above that it fails: at `q = 7`
  (`X = 14`, `G = {7, 11, 13}`) copy `13` has legs `389` and `391 = 17 · 23`, struck by no gear of
  `G`, and `391` is not prime.
-/

namespace RangeLine

/-! ## Definitions -/

/-- `g ∈ G_X`: `g` is a prime gear with `7 ≤ g ≤ X`. -/
def InGears (X g : ℕ) : Prop := g.Prime ∧ 7 ≤ g ∧ g ≤ X

/-- The finite gear set `G_X = {g prime : 7 ≤ g ≤ X}`. -/
def gearSet (X : ℕ) : Finset ℕ := (Finset.range (X + 1)).filter (fun g => g.Prime ∧ 7 ≤ g)

/-- `P_X = ∏ G_X`, the product of the prime gears `7 ≤ g ≤ X` (`1` when there are none). -/
def gearProd (X : ℕ) : ℕ := ∏ g ∈ gearSet X, g

/-- `E_X(k)`: no acting gear of `G_X` strikes copy `k`. -/
def ActClear (X k : ℕ) : Prop := ∀ g, InGears X g → StrikesCopy g k → ¬ Acts g k

/-- `F_X(k)`: no gear of `G_X` strikes copy `k` (acting dropped). -/
def FreeClear (X k : ℕ) : Prop := ∀ g, InGears X g → ¬ StrikesCopy g k

/-- Copy `k` has both legs prime. -/
def LegsPrime (k : ℕ) : Prop := (30 * k - 1).Prime ∧ (30 * k + 1).Prime

/-- The cut of machine `q`: `isqrt(q#) = Nat.sqrt (primorial q)`. -/
def cutSqrt (q : ℕ) : ℕ := Nat.sqrt (primorial q)

/-! ## The gear set and its product -/

/-- Membership in the finite gear set is membership in `G_X`. -/
theorem mem_gearSet {X g : ℕ} : g ∈ gearSet X ↔ InGears X g := by
  unfold gearSet InGears
  rw [Finset.mem_filter, Finset.mem_range]
  constructor
  · rintro ⟨h1, h2, h3⟩
    exact ⟨h2, h3, by omega⟩
  · rintro ⟨h1, h2, h3⟩
    exact ⟨by omega, h1, h2⟩

/-- Every gear of `G_X` divides `P_X`. -/
theorem dvd_gearProd {X g : ℕ} (h : InGears X g) : g ∣ gearProd X :=
  Finset.dvd_prod_of_mem _ (mem_gearSet.mpr h)

/-- `P_X` is positive. -/
theorem gearProd_pos (X : ℕ) : 0 < gearProd X :=
  Finset.prod_pos fun _ hg => (mem_gearSet.mp hg).1.pos

/-- For `X ≥ 5`, `30 · P_X = X#`: the primes up to `X` are `2, 3, 5` and the gears of `G_X`. -/
theorem thirty_mul_gearProd {X : ℕ} (hX : 5 ≤ X) : 30 * gearProd X = primorial X := by
  unfold gearProd gearSet primorial
  rw [← Finset.prod_filter_mul_prod_filter_not
    ((Finset.range (X + 1)).filter Nat.Prime) (fun g => 7 ≤ g)]
  have h1 : ((Finset.range (X + 1)).filter Nat.Prime).filter (fun g => 7 ≤ g) =
      (Finset.range (X + 1)).filter (fun g => g.Prime ∧ 7 ≤ g) := by
    rw [Finset.filter_filter]
  have h2 : ((Finset.range (X + 1)).filter Nat.Prime).filter (fun g => ¬ 7 ≤ g) = {2, 3, 5} := by
    ext x
    simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_insert, Finset.mem_singleton]
    constructor
    · rintro ⟨⟨_, hp⟩, h7⟩
      have hx : x < 7 := by omega
      interval_cases x <;> first | omega | (norm_num at hp)
    · rintro (rfl | rfl | rfl) <;> exact ⟨⟨by omega, by norm_num⟩, by omega⟩
  rw [h1, h2]
  rw [mul_comm]
  rfl

/-- **Growth lemma** (Bertrand's postulate). For every prime `p` with `7 ≤ p ≤ n`, `2 p² < n#`.
(Bertrand gives a prime `r` with `p/2 < r < p`; then `2 · 3 · r · p ∣ n#` and `6 r ≥ 3 p + 3`.) -/
theorem prime_two_sq_lt_primorial {p n : ℕ} (hp : p.Prime) (hp7 : 7 ≤ p) (hpn : p ≤ n) :
    2 * p ^ 2 < primorial n := by
  obtain ⟨r, hr, hnr, hr2⟩ := Nat.exists_prime_lt_and_le_two_mul (p / 2) (by omega)
  have hodd : p % 2 = 1 := Nat.odd_iff.mp (hp.odd_of_ne_two (by omega))
  have hrp : r < p := by omega
  have hr4 : 4 ≤ r := by omega
  have h2 : 2 ∣ primorial n := (Nat.prime_two.dvd_primorial_iff).mpr (by omega)
  have h3 : 3 ∣ primorial n := (Nat.prime_three.dvd_primorial_iff).mpr (by omega)
  have hrd : r ∣ primorial n := (hr.dvd_primorial_iff).mpr (by omega)
  have hpd : p ∣ primorial n := (hp.dvd_primorial_iff).mpr hpn
  have c2r : Nat.Coprime 2 r := (Nat.coprime_primes Nat.prime_two hr).mpr (by omega)
  have c3r : Nat.Coprime 3 r := (Nat.coprime_primes Nat.prime_three hr).mpr (by omega)
  have c2p : Nat.Coprime 2 p := (Nat.coprime_primes Nat.prime_two hp).mpr (by omega)
  have c3p : Nat.Coprime 3 p := (Nat.coprime_primes Nat.prime_three hp).mpr (by omega)
  have crp : Nat.Coprime r p := (Nat.coprime_primes hr hp).mpr (by omega)
  have h6 : 2 * 3 ∣ primorial n := Nat.Coprime.mul_dvd_of_dvd_of_dvd (by norm_num) h2 h3
  have h6r : 2 * 3 * r ∣ primorial n :=
    Nat.Coprime.mul_dvd_of_dvd_of_dvd (Nat.Coprime.mul_left c2r c3r) h6 hrd
  have c6rp : Nat.Coprime (2 * 3 * r) p :=
    Nat.Coprime.mul_left (Nat.Coprime.mul_left c2p c3p) crp
  have h6rp : 2 * 3 * r * p ∣ primorial n := Nat.Coprime.mul_dvd_of_dvd_of_dvd c6rp h6r hpd
  have hle := Nat.le_of_dvd (primorial_pos n) h6rp
  have hr' : p + 1 ≤ 2 * r := by omega
  have hmul : (p + 1) * p ≤ 2 * r * p := Nat.mul_le_mul_right p hr'
  nlinarith

/-- For every gear `g ∈ G_X`, `2 g² < 30 · P_X`. -/
theorem two_sq_lt_thirty_gearProd {X g : ℕ} (h : InGears X g) : 2 * g ^ 2 < 30 * gearProd X := by
  obtain ⟨hp, h7, hX⟩ := h
  rw [thirty_mul_gearProd (by omega)]
  exact prime_two_sq_lt_primorial hp h7 hX

/-- Every gear of `G_X` acts at every copy `k + P_X`. -/
theorem acts_add_gearProd {X g : ℕ} (h : InGears X g) (k : ℕ) : Acts g (k + gearProd X) := by
  have := two_sq_lt_thirty_gearProd h
  unfold Acts
  nlinarith

/-! ## Strikes on legs -/

/-- A number dividing a leg of copy `k` strikes copy `k`. -/
theorem strikesCopy_of_leg {h k : ℕ} (hd : h ∣ 30 * k - 1 ∨ h ∣ 30 * k + 1) : StrikesCopy h k := by
  unfold StrikesCopy
  rcases hd with hd | hd
  · exact Dvd.dvd.mul_right hd _
  · exact Dvd.dvd.mul_left hd _

/-- A prime dividing a leg of a copy `k ≥ 1` is at least `7` (the legs are prime to `30`). -/
theorem leg_prime_factor_ge_seven {h k : ℕ} (hh : h.Prime) (hk : 1 ≤ k)
    (hd : h ∣ 30 * k - 1 ∨ h ∣ 30 * k + 1) : 7 ≤ h := by
  by_contra hlt
  push Not at hlt
  have h2 := hh.two_le
  interval_cases h <;> omega

/-- If a leg `L` of a copy `k ≥ 1` is not prime, its least prime factor is a prime `≥ 7` that
strikes copy `k` and has square at most `L`. -/
theorem minFac_composite_leg {k L : ℕ} (hk : 1 ≤ k) (hL : L = 30 * k - 1 ∨ L = 30 * k + 1)
    (hnp : ¬ L.Prime) :
    (Nat.minFac L).Prime ∧ 7 ≤ Nat.minFac L ∧ StrikesCopy (Nat.minFac L) k ∧
      (Nat.minFac L) ^ 2 ≤ L := by
  have hL1 : L ≠ 1 := by omega
  have hL0 : 0 < L := by omega
  have hp := Nat.minFac_prime hL1
  have hd : Nat.minFac L ∣ 30 * k - 1 ∨ Nat.minFac L ∣ 30 * k + 1 := by
    rcases hL with rfl | rfl
    · exact Or.inl (Nat.minFac_dvd _)
    · exact Or.inr (Nat.minFac_dvd _)
  exact ⟨hp, leg_prime_factor_ge_seven hp hk hd, strikesCopy_of_leg hd,
    Nat.minFac_sq_le_self hL0 hnp⟩

/-- A prime striking a copy whose legs are both prime is one of the two legs. -/
theorem eq_leg_of_strikes_legsPrime {g k : ℕ} (h : LegsPrime k) (hg : g.Prime)
    (hs : StrikesCopy g k) : g = 30 * k - 1 ∨ g = 30 * k + 1 := by
  rcases (strikesCopy_iff_leg hg).mp hs with hd | hd
  · exact Or.inl ((Nat.prime_dvd_prime_iff_eq hg h.1).mp hd)
  · exact Or.inr ((Nat.prime_dvd_prime_iff_eq hg h.2).mp hd)

/-- **Strikes are periodic.** For `k ≥ 1` and any `g ∣ N`, `g` strikes copy `k + N` exactly when
it strikes copy `k`. -/
theorem strikesCopy_add_dvd {g k N : ℕ} (hk : 1 ≤ k) (hgN : g ∣ N) :
    StrikesCopy g (k + N) ↔ StrikesCopy g k := by
  rw [strikesCopy_iff_sq, strikesCopy_iff_sq]
  have h1 : 1 ≤ 900 * k ^ 2 := by nlinarith
  have h2 : 1 ≤ 900 * (k + N) ^ 2 := by
    have : 1 ≤ (k + N) ^ 2 := Nat.one_le_pow _ _ (by omega)
    omega
  have e : 900 * (k + N) ^ 2 - 1 = (900 * k ^ 2 - 1) + 900 * N * (2 * k + N) := by
    zify [h1, h2]
    ring
  rw [e]
  exact (Nat.dvd_add_left (Dvd.dvd.mul_right (Dvd.dvd.mul_left hgN 900) _))

/-! ## The acting-free pattern and the acting pattern -/

/-- No copy is acting-struck but acting-free unstruck: `F_X(k) ⇒ E_X(k)`. -/
theorem actClear_of_freeClear {X k : ℕ} (h : FreeClear X k) : ActClear X k :=
  fun g hg hs _ => h g hg hs

/-- **The acting-free pattern is periodic.** For `k ≥ 1` and every multiple `N` of `P_X`,
`F_X(k + N) ⇔ F_X(k)`. -/
theorem freeClear_add_iff {X k N : ℕ} (hk : 1 ≤ k) (hN : gearProd X ∣ N) :
    FreeClear X (k + N) ↔ FreeClear X k := by
  constructor
  · intro h g hg hs
    exact h g hg ((strikesCopy_add_dvd hk (dvd_trans (dvd_gearProd hg) hN)).mpr hs)
  · intro h g hg hs
    exact h g hg ((strikesCopy_add_dvd hk (dvd_trans (dvd_gearProd hg) hN)).mp hs)

/-- **One period up, acting is acting-free.** For `k ≥ 1`, `E_X(k + P_X) ⇔ F_X(k)`. -/
theorem actClear_add_gearProd_iff {X k : ℕ} (hk : 1 ≤ k) :
    ActClear X (k + gearProd X) ↔ FreeClear X k := by
  constructor
  · intro hE g hg hs
    exact hE g hg ((strikesCopy_add_dvd hk (dvd_gearProd hg)).mpr hs) (acts_add_gearProd hg k)
  · intro hF g hg hs _
    exact hF g hg ((strikesCopy_add_dvd hk (dvd_gearProd hg)).mp hs)

/-- **(b)** For every `k` and every multiple `N` of `P_X`, `E_X(k + N) ⇒ E_X(k)`: an acting strike
on `k` is an acting strike on `k + N`. -/
theorem actClear_of_actClear_add {X k N : ℕ} (hN : gearProd X ∣ N) (hE : ActClear X (k + N)) :
    ActClear X k := by
  intro g hg hs ha
  rcases Nat.eq_zero_or_pos k with rfl | hk
  · unfold Acts at ha
    have := hg.2.1
    nlinarith
  · exact hE g hg ((strikesCopy_add_dvd hk (dvd_trans (dvd_gearProd hg) hN)).mpr hs)
      (acts_mono ha (by omega))

/-- **(a)** If every gear of `G_X` acts at copy `k`, then `E_X(k + N) ⇔ E_X(k)` for every multiple
`N` of `P_X`. -/
theorem actClear_shift_iff_of_acts {X k N : ℕ} (hN : gearProd X ∣ N)
    (hact : ∀ g, InGears X g → Acts g k) : ActClear X (k + N) ↔ ActClear X k := by
  constructor
  · exact actClear_of_actClear_add hN
  · intro hE g hg hs _
    rcases Nat.eq_zero_or_pos k with rfl | hk
    · have ha := hact g hg
      unfold Acts at ha
      have := hg.2.1
      nlinarith
    · exact hE g hg ((strikesCopy_add_dvd hk (dvd_trans (dvd_gearProd hg) hN)).mp hs) (hact g hg)

/-- **(a)**, square form: if `X² ≤ 30 k + 1` then `E_X(k + P_X) ⇔ E_X(k)`. -/
theorem actClear_period_of_sq_le {X k : ℕ} (hX : X ^ 2 ≤ 30 * k + 1) :
    ActClear X (k + gearProd X) ↔ ActClear X k :=
  actClear_shift_iff_of_acts dvd_rfl fun _ hg =>
    le_trans (Nat.pow_le_pow_left hg.2.2 2) hX

/-- **Legs prime from acting-clear.** For `k ≥ 1` with `30 k + 1 < (X + 1)²`, if no acting gear
of `G_X` strikes copy `k`, both legs of copy `k` are prime. -/
theorem legsPrime_of_actClear {X k : ℕ} (hk : 1 ≤ k) (hb : 30 * k + 1 < (X + 1) ^ 2)
    (hE : ActClear X k) : LegsPrime k := by
  have key : ∀ L, (L = 30 * k - 1 ∨ L = 30 * k + 1) → L.Prime := by
    intro L hL
    by_contra hnp
    obtain ⟨hp, h7, hs, hsq⟩ := minFac_composite_leg hk hL hnp
    have hLle : L ≤ 30 * k + 1 := by omega
    have hX : Nat.minFac L ≤ X := by
      by_contra hc
      push Not at hc
      have : (X + 1) ^ 2 ≤ (Nat.minFac L) ^ 2 := Nat.pow_le_pow_left hc 2
      omega
    exact hE _ ⟨hp, h7, hX⟩ hs (by unfold Acts; omega)
  exact ⟨key _ (Or.inl rfl), key _ (Or.inr rfl)⟩

/-- A copy `k ≥ 1` with both legs prime is struck by no acting gear, for every `X`. -/
theorem actClear_of_legsPrime {X k : ℕ} (hk : 1 ≤ k) (h : LegsPrime k) : ActClear X k := by
  intro g hg hs ha
  have hge : 30 * k - 1 ≤ g := by
    rcases eq_leg_of_strikes_legsPrime h hg.1 hs with e | e <;> omega
  unfold Acts at ha
  have hsq : (30 * k - 1) ^ 2 ≤ g ^ 2 := Nat.pow_le_pow_left hge 2
  have h29 : 29 ≤ 30 * k - 1 := by omega
  have e : 30 * k + 1 = (30 * k - 1) + 2 := by omega
  rw [e] at ha
  nlinarith

/-- A copy with both legs prime and lower leg above `X` is struck by no gear of `G_X`. -/
theorem freeClear_of_legsPrime_above {X k : ℕ} (hL : LegsPrime k) (hX : X < 30 * k - 1) :
    FreeClear X k := by
  intro g hg hs
  have := hg.2.2
  rcases eq_leg_of_strikes_legsPrime hL hg.1 hs with e | e <;> omega

/-- The copy `P_X` is struck by no gear of `G_X`: its legs `30 P_X ∓ 1` are prime to `P_X`. -/
theorem freeClear_gearProd (X : ℕ) : FreeClear X (gearProd X) := by
  intro g hg hs
  rw [strikesCopy_iff_sq] at hs
  have hP := gearProd_pos X
  have h1 : 1 ≤ 900 * gearProd X ^ 2 := by nlinarith
  have hd : g ∣ 900 * gearProd X ^ 2 :=
    Dvd.dvd.mul_left (dvd_pow (dvd_gearProd hg) two_ne_zero) 900
  have h1d : g ∣ 1 := by
    have := Nat.dvd_sub hd hs
    rwa [Nat.sub_sub_self h1] at this
  exact hg.1.one_lt.ne' (Nat.dvd_one.mp h1d)

/-- If `k ≥ 1` is acting-clear but some gear `g ∈ G_X` strikes it, then both legs are prime and
`30 k - 1 ≤ X` (the struck leg is the gear itself). -/
theorem legsPrime_of_actClear_strikes {X k g : ℕ} (hk : 1 ≤ k) (hE : ActClear X k)
    (hg : InGears X g) (hs : StrikesCopy g k) : LegsPrime k ∧ 30 * k - 1 ≤ X := by
  have hna : ¬ Acts g k := hE g hg hs
  unfold Acts at hna
  push Not at hna
  have hgX : g ^ 2 ≤ X ^ 2 := Nat.pow_le_pow_left hg.2.2 2
  have hX1 : X ^ 2 < (X + 1) ^ 2 := Nat.pow_lt_pow_left (Nat.lt_succ_self X) two_ne_zero
  have hb : 30 * k + 1 < (X + 1) ^ 2 := by omega
  have hL := legsPrime_of_actClear hk hb hE
  refine ⟨hL, ?_⟩
  have := hg.2.2
  rcases eq_leg_of_strikes_legsPrime hL hg.1 hs with e | e <;> omega

/-- **(c)** For every `X` and every `k`: `E_X(k) ∧ ¬ E_X(k + P_X)` exactly when both legs of copy
`k` are prime and `30 k - 1 ≤ X`. -/
theorem actClear_jump_iff {X k : ℕ} :
    (ActClear X k ∧ ¬ ActClear X (k + gearProd X)) ↔ LegsPrime k ∧ 30 * k - 1 ≤ X := by
  rcases Nat.eq_zero_or_pos k with rfl | hk
  · constructor
    · rintro ⟨_, hn⟩
      exact absurd (by rw [zero_add]; exact actClear_of_freeClear (freeClear_gearProd X)) hn
    · rintro ⟨⟨h0, _⟩, _⟩
      norm_num at h0
  · constructor
    · rintro ⟨hE, hn⟩
      unfold ActClear at hn
      push Not at hn
      obtain ⟨g, hg, hs, _⟩ := hn
      exact legsPrime_of_actClear_strikes hk hE hg ((strikesCopy_add_dvd hk (dvd_gearProd hg)).mp hs)
    · rintro ⟨hL, hX⟩
      refine ⟨actClear_of_legsPrime hk hL, fun hE => ?_⟩
      have hg : InGears X (30 * k - 1) := ⟨hL.1, by omega, hX⟩
      exact hE _ hg ((strikesCopy_add_dvd hk (dvd_gearProd hg)).mpr
        (strikesCopy_of_leg (Or.inl dvd_rfl))) (acts_add_gearProd hg k)

/-- **Where acting and acting-free differ.** For `k ≥ 1`: `E_X(k) ∧ ¬ F_X(k)` exactly when both
legs of copy `k` are prime and `30 k - 1 ≤ X`. -/
theorem acting_free_differ_iff {X k : ℕ} (hk : 1 ≤ k) :
    (ActClear X k ∧ ¬ FreeClear X k) ↔ LegsPrime k ∧ 30 * k - 1 ≤ X := by
  rw [← actClear_add_gearProd_iff hk]
  exact actClear_jump_iff

/-- **The two deletion patterns differ exactly on the low twin copies.** For `k ≥ 1`, the acting
and acting-free patterns of `G_X` disagree at copy `k` exactly when both legs are prime and some
leg is at most `X`. -/
theorem patterns_differ_iff {X k : ℕ} (hk : 1 ≤ k) :
    ¬ (ActClear X k ↔ FreeClear X k) ↔
      LegsPrime k ∧ (30 * k - 1 ≤ X ∨ 30 * k + 1 ≤ X) := by
  have key := acting_free_differ_iff (X := X) hk
  constructor
  · intro hne
    have h : ActClear X k ∧ ¬ FreeClear X k := by
      by_contra hc
      apply hne
      constructor
      · intro hE
        by_contra hF
        exact hc ⟨hE, hF⟩
      · exact actClear_of_freeClear
    obtain ⟨hL, hX⟩ := key.mp h
    exact ⟨hL, Or.inl hX⟩
  · rintro ⟨hL, hX⟩ hiff
    have hX' : 30 * k - 1 ≤ X := by omega
    obtain ⟨hE, hF⟩ := key.mpr ⟨hL, hX'⟩
    exact hF (hiff.mp hE)

/-- **Revealed with acting.** For `k ≥ 1` with `30 k + 1 < (X + 1)²`: `E_X(k)` exactly when both
legs of copy `k` are prime. -/
theorem actClear_iff_legsPrime {X k : ℕ} (hk : 1 ≤ k) (hb : 30 * k + 1 < (X + 1) ^ 2) :
    ActClear X k ↔ LegsPrime k :=
  ⟨legsPrime_of_actClear hk hb, actClear_of_legsPrime hk⟩

/-- **Revealed acting-free.** For `k ≥ 1` with `30 k + 1 < (X + 1)²`: `F_X(k)` exactly when both
legs of copy `k` are prime and the lower leg is above `X`. -/
theorem freeClear_iff {X k : ℕ} (hk : 1 ≤ k) (hb : 30 * k + 1 < (X + 1) ^ 2) :
    FreeClear X k ↔ LegsPrime k ∧ X < 30 * k - 1 := by
  constructor
  · intro hF
    have hL := legsPrime_of_actClear hk hb (actClear_of_freeClear hF)
    refine ⟨hL, ?_⟩
    by_contra hle
    push Not at hle
    exact hF _ ⟨hL.1, by omega, hle⟩ (strikesCopy_of_leg (Or.inl dvd_rfl))
  · rintro ⟨hL, hX⟩
    exact freeClear_of_legsPrime_above hL hX

/-! ## No period for "both legs prime" -/

/-- **Lower leg of a shifted copy.** For `j ≥ 1` and every `N`, the copy `j + (30 j - 1) N` has
lower leg `(30 j - 1)(1 + 30 N)`. -/
theorem lower_leg_shift (j N : ℕ) (hj : 1 ≤ j) :
    30 * (j + (30 * j - 1) * N) - 1 = (30 * j - 1) * (1 + 30 * N) := by
  obtain ⟨m, rfl⟩ : ∃ m, j = m + 1 := ⟨j - 1, by omega⟩
  have e : 30 * (m + 1) - 1 = 30 * m + 29 := by omega
  rw [e]
  have e2 : 30 * (m + 1 + (30 * m + 29) * N) = (30 * m + 29) * (1 + 30 * N) + 1 := by ring
  rw [e2, Nat.add_sub_cancel]

/-- For `j ≥ 1` and `N ≥ 1`, the copy `j + (30 j - 1) N` does not have both legs prime. -/
theorem not_legsPrime_shift {j N : ℕ} (hj : 1 ≤ j) (hN : 1 ≤ N) :
    ¬ LegsPrime (j + (30 * j - 1) * N) := by
  rintro ⟨h1, _⟩
  rw [lower_leg_shift j N hj] at h1
  exact Nat.not_prime_mul (by omega) (by omega) h1

/-- Copy `1` (legs `29`, `31`) has both legs prime. -/
theorem legsPrime_one : LegsPrime 1 := ⟨by norm_num, by norm_num⟩

/-- **No period.** For every `N ≥ 1`, "both legs prime" is not `N`-periodic on the copies
`j ≥ 1`. -/
theorem legsPrime_no_period {N : ℕ} (hN : 1 ≤ N) :
    ¬ ∀ j, 1 ≤ j → (LegsPrime (j + N) ↔ LegsPrime j) := by
  intro hper
  have hstep : ∀ m, LegsPrime (1 + m * N) := by
    intro m
    induction m with
    | zero => simpa using legsPrime_one
    | succ m ih =>
      have e : 1 + (m + 1) * N = (1 + m * N) + N := by ring
      rw [e]
      exact (hper _ (by omega)).mpr ih
  have h := not_legsPrime_shift (j := 1) le_rfl hN
  have e : 1 + (30 * 1 - 1) * N = 1 + 29 * N := by norm_num
  rw [e] at h
  exact h (hstep 29)

/-! ## At the cut `X = isqrt(q#)` -/

/-- A number is at most `isqrt(q#)` exactly when its square is at most `q#`. -/
theorem le_cutSqrt_iff {q g : ℕ} : g ≤ cutSqrt q ↔ g ^ 2 ≤ primorial q := Nat.le_sqrt'

/-- `isqrt(q#) < n` exactly when `q# < n²`. -/
theorem cutSqrt_lt_iff {q n : ℕ} : cutSqrt q < n ↔ primorial q < n ^ 2 := Nat.sqrt_lt'

/-- `q# < (isqrt(q#) + 1)²`. -/
theorem primorial_lt_cutSqrt_succ_sq (q : ℕ) : primorial q < (cutSqrt q + 1) ^ 2 :=
  Nat.lt_succ_sqrt' _

/-- **Every gear acts above `M`.** For every `q`, every gear of `G = [7, isqrt(q#)]` acts at every
copy `k ≥ M = q#/30` (`g² ≤ q# = 30 M ≤ 30 k`). -/
theorem acts_of_mirrorM_le {q k g : ℕ} (hk : mirrorM q ≤ k) (hg : InGears (cutSqrt q) g) :
    Acts g k := by
  have hsq : g ^ 2 ≤ primorial q := le_cutSqrt_iff.mp hg.2.2
  have h49 : 49 ≤ g ^ 2 := by
    have := hg.2.1
    nlinarith
  have hq : 5 ≤ q := by
    by_contra hq
    push Not at hq
    have hmono : primorial q ≤ primorial 4 := primorial_mono (by omega)
    have h4 : primorial 4 = 6 := by decide
    omega
  have h30 := thirty_mul_mirrorM hq
  unfold Acts
  omega

/-- **(a) at the cut `isqrt(q#)`.** For every `q` and every copy `k ≥ M`,
`E(k + P) ⇔ E(k)`, with `G = [7, isqrt(q#)]` and `P = ∏ G`. -/
theorem actClear_period_range (q k : ℕ) (hk : mirrorM q ≤ k) :
    ActClear (cutSqrt q) (k + gearProd (cutSqrt q)) ↔ ActClear (cutSqrt q) k :=
  actClear_shift_iff_of_acts dvd_rfl fun _ hg => acts_of_mirrorM_le hk hg

/-- **(b) at the cut `isqrt(q#)`.** For every `q` and every `k`, `E(k + P) ⇒ E(k)`. -/
theorem actClear_of_actClear_add_range (q k : ℕ)
    (h : ActClear (cutSqrt q) (k + gearProd (cutSqrt q))) : ActClear (cutSqrt q) k :=
  actClear_of_actClear_add dvd_rfl h

/-- **(c) at the cut `isqrt(q#)`.** For every `q` and every `k`, `E(k) ∧ ¬ E(k + P)` exactly when
both legs of copy `k` are prime and `30 k - 1 ≤ isqrt(q#)`. -/
theorem actClear_jump_iff_range (q k : ℕ) :
    (ActClear (cutSqrt q) k ∧ ¬ ActClear (cutSqrt q) (k + gearProd (cutSqrt q))) ↔
      LegsPrime k ∧ 30 * k - 1 ≤ cutSqrt q :=
  actClear_jump_iff

/-- **The difference law at the cut `isqrt(q#)`.** For every `q` and every copy `j ≥ 1`, the
acting and acting-free deletion patterns of `G = [7, isqrt(q#)]` differ at `j` exactly when both
legs are prime and some leg is at most `isqrt(q#)`. -/
theorem patterns_differ_iff_range (q j : ℕ) (hj : 1 ≤ j) :
    ¬ (ActClear (cutSqrt q) j ↔ FreeClear (cutSqrt q) j) ↔
      LegsPrime j ∧ (30 * j - 1 ≤ cutSqrt q ∨ 30 * j + 1 ≤ cutSqrt q) :=
  patterns_differ_iff hj

/-- **Revealed with acting, on the range.** For every `q` and every copy `j ≥ 1` with
`30 j + 1 ≤ q#`: no acting gear of `[7, isqrt(q#)]` strikes `j` exactly when both legs are
prime. -/
theorem actClear_iff_legsPrime_range {q j : ℕ} (hj : 1 ≤ j) (hjq : 30 * j + 1 ≤ primorial q) :
    ActClear (cutSqrt q) j ↔ LegsPrime j :=
  actClear_iff_legsPrime hj (lt_of_le_of_lt hjq (primorial_lt_cutSqrt_succ_sq q))

/-- **Revealed acting-free, on the range.** For every `q` and every copy `j ≥ 1` with
`30 j + 1 ≤ q#`: no gear of `[7, isqrt(q#)]` strikes `j` exactly when both legs are prime and both
legs are above `√(q#)`. -/
theorem freeClear_iff_range {q j : ℕ} (hj : 1 ≤ j) (hjq : 30 * j + 1 ≤ primorial q) :
    FreeClear (cutSqrt q) j ↔
      LegsPrime j ∧ primorial q < (30 * j - 1) ^ 2 ∧ primorial q < (30 * j + 1) ^ 2 := by
  rw [freeClear_iff hj (lt_of_le_of_lt hjq (primorial_lt_cutSqrt_succ_sq q)), cutSqrt_lt_iff]
  constructor
  · rintro ⟨hL, h⟩
    exact ⟨hL, h, lt_of_lt_of_le h (Nat.pow_le_pow_left (by omega) 2)⟩
  · rintro ⟨hL, h, _⟩
    exact ⟨hL, h⟩

/-- **No period, at the cut `isqrt(q#)`.** For every `q` and every copy `j ≥ 1`, with `p = 30 j - 1`
and `P = ∏ [7, isqrt(q#)]`, the copy `j + p P` has lower leg `p (1 + 30 P)`, so it does not have
both legs prime. -/
theorem lower_leg_shift_range (q j : ℕ) (hj : 1 ≤ j) :
    30 * (j + (30 * j - 1) * gearProd (cutSqrt q)) - 1 =
        (30 * j - 1) * (1 + 30 * gearProd (cutSqrt q)) ∧
      ¬ LegsPrime (j + (30 * j - 1) * gearProd (cutSqrt q)) :=
  ⟨lower_leg_shift j _ hj, not_legsPrime_shift hj (gearProd_pos _)⟩

/-- **"Both legs prime" has no period `P`.** For every `q`, with `P = ∏ [7, isqrt(q#)]`, it is not
the case that `LegsPrime (j + P) ⇔ LegsPrime j` for all copies `j ≥ 1`. -/
theorem legsPrime_no_period_range (q : ℕ) :
    ¬ ∀ j, 1 ≤ j → (LegsPrime (j + gearProd (cutSqrt q)) ↔ LegsPrime j) :=
  legsPrime_no_period (gearProd_pos _)

/-- **Fixed cut, copy by copy.** For every `q` and every copy `j ≥ 1` with `30 j + 1 ≤ q#`:
`j` is a range copy (`q < 30 j - 1`) struck by no gear of `[7, isqrt(q#)]` exactly when both legs
are prime and the lower leg is above `isqrt(q#)`. -/
theorem fixed_cut_copy {q j : ℕ} (hj : 1 ≤ j) (hjq : 30 * j + 1 ≤ primorial q) :
    (q < 30 * j - 1 ∧ FreeClear (cutSqrt q) j) ↔ (cutSqrt q < 30 * j - 1 ∧ LegsPrime j) := by
  have hb : 30 * j + 1 < (cutSqrt q + 1) ^ 2 :=
    lt_of_le_of_lt hjq (primorial_lt_cutSqrt_succ_sq q)
  constructor
  · rintro ⟨_, hF⟩
    obtain ⟨hL, hX⟩ := (freeClear_iff hj hb).mp hF
    exact ⟨hX, hL⟩
  · rintro ⟨hX, hL⟩
    refine ⟨?_, freeClear_of_legsPrime_above hL hX⟩
    by_contra hle
    push Not at hle
    have h2 := prime_two_sq_lt_primorial hL.1 (by omega) hle
    have h3 := cutSqrt_lt_iff.mp hX
    omega

/-- **Fixed cut.** For every `q`: some range copy (`1 ≤ j`, `q < 30 j - 1`, `30 j + 1 ≤ q#`) is
struck by no gear of `[7, isqrt(q#)]` exactly when some copy `j ≥ 1` with `30 j + 1 ≤ q#` has both
legs prime and its lower leg above `isqrt(q#)` (the window statement at the cut `isqrt(q#)`). -/
theorem fixed_cut (q : ℕ) :
    (∃ j, 1 ≤ j ∧ q < 30 * j - 1 ∧ 30 * j + 1 ≤ primorial q ∧ FreeClear (cutSqrt q) j) ↔
      (∃ j, 1 ≤ j ∧ cutSqrt q < 30 * j - 1 ∧ 30 * j + 1 ≤ primorial q ∧ LegsPrime j) := by
  constructor
  · rintro ⟨j, hj, hq, hjq, hF⟩
    obtain ⟨hX, hL⟩ := (fixed_cut_copy hj hjq).mp ⟨hq, hF⟩
    exact ⟨j, hj, hX, hjq, hL⟩
  · rintro ⟨j, hj, hX, hjq, hL⟩
    obtain ⟨hq, hF⟩ := (fixed_cut_copy hj hjq).mpr ⟨hX, hL⟩
    exact ⟨j, hj, hq, hjq, hF⟩

/-- **Fixed cut, square form.** For every `q`: some range copy is struck by no gear of
`[7, isqrt(q#)]` exactly when some copy `j ≥ 1` with `30 j + 1 ≤ q#` has both legs prime and
`q# < (30 j - 1)²`, i.e. is a twin copy above `√(q#)`. -/
theorem fixed_cut_sq (q : ℕ) :
    (∃ j, 1 ≤ j ∧ q < 30 * j - 1 ∧ 30 * j + 1 ≤ primorial q ∧ FreeClear (cutSqrt q) j) ↔
      (∃ j, 1 ≤ j ∧ primorial q < (30 * j - 1) ^ 2 ∧ 30 * j + 1 ≤ primorial q ∧ LegsPrime j) := by
  rw [fixed_cut]
  simp only [cutSqrt_lt_iff]

end RangeLine
