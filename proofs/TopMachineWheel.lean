/-
THE TOP MACHINE: THE WHEEL COUNT AND THE CONJUGACY (Formalist, round 32).

Two laws of `research/proof/top_machine_1.md` section 4 that need more than
the raw line:

  * **L5, the wheel count** - a wheel of gears `G` has exactly `prod (g - 2)`
    open pairs per turn.  Proved by CRT, by induction on `G`: the general
    splitting lemma `card_filter_crt` says a residue-invariant conjunction
    counts multiplicatively across coprime moduli, and the single-gear factor
    is `TopMachine.card_open_residues` (L1).

  * **L19, the conjugacy** - `n |-> k = 6^{-1}(n + 1)` carries the top
    machine's open-pair set onto the SAME gears' opening set in the bottom
    machine's COLUMN coordinate, where slot `k` carries `(6k - 1, 6k + 1)`
    (`Census.lo`, `Census.hi`).  The map exists whenever the wheel is coprime
    to 6, i.e. whenever every gear is at least 5.  This is the branch's
    organising fact: counting and symmetry laws are common property of the two
    coordinates, metric laws are not.
-/

import TopMachine
import Census
import Mathlib.Data.Nat.GCD.BigOperators
import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise
import Mathlib.Data.Int.GCD

namespace TopMachine

/-! ## Residue invariance -/

/-- `StrikesR g` depends only on the residue mod `g`. -/
theorem strikesR_congr {g x y : ℕ} (h : x % g = y % g) : StrikesR g x ↔ StrikesR g y := by
  unfold StrikesR
  rw [Nat.add_mod x 2 g, Nat.add_mod y 2 g, h]

/-- `OpenN G` depends only on the residue mod any common multiple of the
gears - in particular mod the wheel. -/
theorem openN_congr {G : Finset ℕ} {W x y : ℕ} (hdvd : ∀ g ∈ G, g ∣ W)
    (h : x % W = y % W) : OpenN G x ↔ OpenN G y := by
  unfold OpenN
  refine forall_congr' fun g => forall_congr' fun hg => not_congr (strikesR_congr ?_)
  rw [← Nat.mod_mod_of_dvd x (hdvd g hg), ← Nat.mod_mod_of_dvd y (hdvd g hg), h]

/-- Splitting off one gear. -/
theorem openN_insert {a : ℕ} {s : Finset ℕ} {n : ℕ} :
    OpenN (insert a s) n ↔ (¬ StrikesR a n ∧ OpenN s n) := by
  constructor
  · intro h
    exact ⟨h a (Finset.mem_insert_self a s), fun g hg => h g (Finset.mem_insert_of_mem hg)⟩
  · rintro ⟨h1, h2⟩ g hg
    rcases Finset.mem_insert.mp hg with rfl | hg'
    · exact h1
    · exact h2 g hg'

/-! ## The CRT splitting lemma -/

/-- **CRT, in counting form.**  A conjunction of one predicate depending only
on the residue mod `a` and one depending only on the residue mod `b`, with
`a`, `b` coprime, counts multiplicatively over a period `a * b`. -/
theorem card_filter_crt {a b : ℕ} (ha : 0 < a) (hb : 0 < b) (hab : Nat.Coprime a b)
    (p q : ℕ → Prop) [DecidablePred p] [DecidablePred q]
    (hp : ∀ x y : ℕ, x % a = y % a → (p x ↔ p y))
    (hq : ∀ x y : ℕ, x % b = y % b → (q x ↔ q y)) :
    ((Finset.range (a * b)).filter (fun n => p n ∧ q n)).card
      = ((Finset.range a).filter p).card * ((Finset.range b).filter q).card := by
  classical
  have hab0 : 0 < a * b := Nat.mul_pos ha hb
  have hda : a ∣ a * b := ⟨b, rfl⟩
  have hdb : b ∣ a * b := ⟨a, Nat.mul_comm a b⟩
  rw [← Finset.card_product]
  refine Finset.card_bij' (fun n _ => (n % a, n % b))
    (fun z _ => (Nat.chineseRemainder hab z.1 z.2 : ℕ) % (a * b)) ?_ ?_ ?_ ?_
  · -- forward map lands in the product
    intro n hn
    obtain ⟨hnr, hnp, hnq⟩ : n < a * b ∧ p n ∧ q n := by
      obtain ⟨h1, h2⟩ := Finset.mem_filter.mp hn
      exact ⟨Finset.mem_range.mp h1, h2.1, h2.2⟩
    refine Finset.mem_product.mpr
      ⟨Finset.mem_filter.mpr ⟨Finset.mem_range.mpr (Nat.mod_lt _ ha), ?_⟩,
       Finset.mem_filter.mpr ⟨Finset.mem_range.mpr (Nat.mod_lt _ hb), ?_⟩⟩
    · exact (hp n (n % a) (Nat.mod_mod_of_dvd n (dvd_refl a)).symm).mp hnp
    · exact (hq n (n % b) (Nat.mod_mod_of_dvd n (dvd_refl b)).symm).mp hnq
  · -- backward map lands in the filtered range
    intro z hz
    obtain ⟨hz1, hz2⟩ := Finset.mem_product.mp hz
    obtain ⟨hz1r, hz1p⟩ := Finset.mem_filter.mp hz1
    obtain ⟨hz2r, hz2q⟩ := Finset.mem_filter.mp hz2
    have hz1lt : z.1 < a := Finset.mem_range.mp hz1r
    have hz2lt : z.2 < b := Finset.mem_range.mp hz2r
    have hka : (Nat.chineseRemainder hab z.1 z.2 : ℕ) % a = z.1 := by
      have h := (Nat.chineseRemainder hab z.1 z.2).2.1
      unfold Nat.ModEq at h
      rw [h, Nat.mod_eq_of_lt hz1lt]
    have hkb : (Nat.chineseRemainder hab z.1 z.2 : ℕ) % b = z.2 := by
      have h := (Nat.chineseRemainder hab z.1 z.2).2.2
      unfold Nat.ModEq at h
      rw [h, Nat.mod_eq_of_lt hz2lt]
    refine Finset.mem_filter.mpr ⟨Finset.mem_range.mpr (Nat.mod_lt _ hab0), ?_, ?_⟩
    · refine (hp z.1 _ ?_).mp hz1p
      rw [Nat.mod_mod_of_dvd _ hda, hka, Nat.mod_eq_of_lt hz1lt]
    · refine (hq z.2 _ ?_).mp hz2q
      rw [Nat.mod_mod_of_dvd _ hdb, hkb, Nat.mod_eq_of_lt hz2lt]
  · -- left inverse
    intro n hn
    have hnr : n < a * b := Finset.mem_range.mp (Finset.mem_filter.mp hn).1
    have hka : (Nat.chineseRemainder hab (n % a) (n % b) : ℕ) % a = n % a := by
      have h := (Nat.chineseRemainder hab (n % a) (n % b)).2.1
      unfold Nat.ModEq at h
      rw [h, Nat.mod_mod_of_dvd n (dvd_refl a)]
    have hkb : (Nat.chineseRemainder hab (n % a) (n % b) : ℕ) % b = n % b := by
      have h := (Nat.chineseRemainder hab (n % a) (n % b)).2.2
      unfold Nat.ModEq at h
      rw [h, Nat.mod_mod_of_dvd n (dvd_refl b)]
    have h1 : ((Nat.chineseRemainder hab (n % a) (n % b) : ℕ) % (a * b)) ≡ n [MOD a] := by
      unfold Nat.ModEq
      rw [Nat.mod_mod_of_dvd _ hda, hka]
    have h2 : ((Nat.chineseRemainder hab (n % a) (n % b) : ℕ) % (a * b)) ≡ n [MOD b] := by
      unfold Nat.ModEq
      rw [Nat.mod_mod_of_dvd _ hdb, hkb]
    have h3 := (Nat.modEq_and_modEq_iff_modEq_mul hab).mp ⟨h1, h2⟩
    unfold Nat.ModEq at h3
    rwa [Nat.mod_mod_of_dvd _ (dvd_refl (a * b)), Nat.mod_eq_of_lt hnr] at h3
  · -- right inverse
    intro z hz
    obtain ⟨hz1, hz2⟩ := Finset.mem_product.mp hz
    have hz1lt : z.1 < a := Finset.mem_range.mp (Finset.mem_filter.mp hz1).1
    have hz2lt : z.2 < b := Finset.mem_range.mp (Finset.mem_filter.mp hz2).1
    have hka : (Nat.chineseRemainder hab z.1 z.2 : ℕ) % a = z.1 := by
      have h := (Nat.chineseRemainder hab z.1 z.2).2.1
      unfold Nat.ModEq at h
      rw [h, Nat.mod_eq_of_lt hz1lt]
    have hkb : (Nat.chineseRemainder hab z.1 z.2 : ℕ) % b = z.2 := by
      have h := (Nat.chineseRemainder hab z.1 z.2).2.2
      unfold Nat.ModEq at h
      rw [h, Nat.mod_eq_of_lt hz2lt]
    refine Prod.ext ?_ ?_
    · show (Nat.chineseRemainder hab z.1 z.2 : ℕ) % (a * b) % a = z.1
      rw [Nat.mod_mod_of_dvd _ hda, hka]
    · show (Nat.chineseRemainder hab z.1 z.2 : ℕ) % (a * b) % b = z.2
      rw [Nat.mod_mod_of_dvd _ hdb, hkb]

/-! ## L5 (the wheel count) -/

/-- **L5, the wheel count.**  A wheel of pairwise coprime gears, each at least
3, has exactly `prod (g - 2)` open pairs per turn.  Verified exactly on the
branch's wheels (495, 1485, 2805, 5355, 7425, 25245, 126225, ...). -/
theorem wheel_count : ∀ (G : Finset ℕ), (∀ g ∈ G, 3 ≤ g) →
    (∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) →
    ((Finset.range (∏ g ∈ G, g)).filter (fun n => OpenN G n)).card
      = ∏ g ∈ G, (g - 2) := by
  classical
  intro G
  induction G using Finset.induction_on with
  | empty => intro _ _; simp [OpenN]
  | insert a s hasnot ih =>
      intro hG3 hcop
      have hmemA : a ∈ insert a s := Finset.mem_insert_self a s
      have ha3 : 3 ≤ a := hG3 a hmemA
      have ha0 : 0 < a := by omega
      have hs3 : ∀ g ∈ s, 3 ≤ g := fun g hg => hG3 g (Finset.mem_insert_of_mem hg)
      have hscop : ∀ g ∈ s, ∀ h ∈ s, g ≠ h → Nat.Coprime g h := fun g hg h hh hne =>
        hcop g (Finset.mem_insert_of_mem hg) h (Finset.mem_insert_of_mem hh) hne
      have hP0 : 0 < ∏ g ∈ s, g := by
        have h1 : 1 ≤ ∏ g ∈ s, g :=
          Finset.one_le_prod' fun i hi => by have := hs3 i hi; omega
        omega
      have hcopr : Nat.Coprime a (∏ g ∈ s, g) :=
        Nat.Coprime.prod_right fun i hi =>
          hcop a hmemA i (Finset.mem_insert_of_mem hi) (by rintro rfl; exact hasnot hi)
      have hdvd : ∀ g ∈ s, g ∣ (∏ g ∈ s, g) := fun g hg => Finset.dvd_prod_of_mem _ hg
      rw [Finset.prod_insert hasnot, Finset.prod_insert hasnot]
      have hfil : (Finset.range (a * ∏ g ∈ s, g)).filter (fun n => OpenN (insert a s) n)
          = (Finset.range (a * ∏ g ∈ s, g)).filter
              (fun n => (¬ StrikesR a n) ∧ OpenN s n) :=
        Finset.filter_congr fun n _ => openN_insert
      rw [hfil,
        card_filter_crt ha0 hP0 hcopr (fun n => ¬ StrikesR a n) (OpenN s)
          (fun x y h => not_congr (strikesR_congr h))
          (fun x y h => openN_congr hdvd h),
        card_open_residues ha3, ih hs3 hscop]

/-! ## L19 (the conjugacy with the column coordinate) -/

/-- The column-coordinate opening predicate: slot `k` carries the pair
`(6k - 1, 6k + 1)`, and a gear blocks the slot iff it divides either member.
This is `Census.lo` / `Census.hi` written over `ℤ`. -/
def ColOpen (G : Finset ℕ) (k : ℤ) : Prop :=
  ∀ g ∈ G, ¬ ((g : ℤ) ∣ 6 * k - 1 ∨ (g : ℤ) ∣ 6 * k + 1)

/-- **L19, one gear.**  If `6k = n + 1 (mod g)` then gear `g` strikes the pair
`n` exactly when it blocks the column `k`: `g | n` iff `g | 6k - 1`, and
`g | n + 2` iff `g | 6k + 1`. -/
theorem strikes_iff_col {g : ℕ} {n k : ℤ} (h : (g : ℤ) ∣ 6 * k - (n + 1)) :
    Strikes g n ↔ ((g : ℤ) ∣ 6 * k - 1 ∨ (g : ℤ) ∣ 6 * k + 1) := by
  have d1 : (g : ℤ) ∣ (6 * k - 1) - n := by
    obtain ⟨t, ht⟩ := h; exact ⟨t, by linear_combination ht⟩
  have d2 : (g : ℤ) ∣ (6 * k + 1) - (n + 2) := by
    obtain ⟨t, ht⟩ := h; exact ⟨t, by linear_combination ht⟩
  unfold Strikes
  rw [← dvd_iff_of_dvd_sub d1, ← dvd_iff_of_dvd_sub d2]

/-- **L19, the conjugacy.**  `n |-> k` with `6k = n + 1` modulo every gear
carries the top machine's open-pair set exactly onto the same gears' opening
set in the column coordinate.  Verified with 0 mismatches on the branch's
wheels. -/
theorem conjugacy {G : Finset ℕ} {n k : ℤ} (h : ∀ g ∈ G, (g : ℤ) ∣ 6 * k - (n + 1)) :
    IsOpen G n ↔ ColOpen G k :=
  forall_congr' fun g =>
    ⟨fun hh hg => (not_congr (strikes_iff_col (h g hg))).mp (hh hg),
     fun hh hg => (not_congr (strikes_iff_col (h g hg))).mpr (hh hg)⟩

/-- **The conjugating column exists.**  If the wheel `W` is coprime to 6 -
which it is as soon as every gear is at least 5 - then `6` is invertible mod
`W` and every pair `n` has a column `k = 6^{-1}(n + 1)`. -/
theorem exists_column {G : Finset ℕ} {W : ℕ} (hW : Nat.gcd 6 W = 1)
    (hdvd : ∀ g ∈ G, (g : ℤ) ∣ (W : ℤ)) (n : ℤ) :
    ∃ k : ℤ, ∀ g ∈ G, (g : ℤ) ∣ 6 * k - (n + 1) := by
  have hbez : (1 : ℤ) = 6 * Nat.gcdA 6 W + (W : ℤ) * Nat.gcdB 6 W := by
    have h := Nat.gcd_eq_gcd_ab 6 W
    rw [hW] at h
    exact_mod_cast h
  refine ⟨Nat.gcdA 6 W * (n + 1), fun g hg => ?_⟩
  have hWdvd : (W : ℤ) ∣ 6 * (Nat.gcdA 6 W * (n + 1)) - (n + 1) :=
    ⟨-(Nat.gcdB 6 W * (n + 1)), by linear_combination (-(n + 1)) * hbez⟩
  exact dvd_trans (hdvd g hg) hWdvd

/-- **L19 against the project's column definitions.**  The same statement with
the bottom machine's own slot members `Census.lo k = 6k - 1` and
`Census.hi k = 6k + 1`, over `ℕ` slots. -/
theorem conjugacy_census {G : Finset ℕ} {n : ℤ} {k : ℕ} (hk : 1 ≤ k)
    (h : ∀ g ∈ G, (g : ℤ) ∣ 6 * (k : ℤ) - (n + 1)) :
    IsOpen G n ↔ ∀ g ∈ G, ¬ (g ∣ Census.lo k ∨ g ∣ Census.hi k) := by
  have hlo : ((Census.lo k : ℕ) : ℤ) = 6 * (k : ℤ) - 1 := by
    unfold Census.lo
    have : 1 ≤ 6 * k := by omega
    push_cast [Nat.cast_sub this]
    ring
  have hhi : ((Census.hi k : ℕ) : ℤ) = 6 * (k : ℤ) + 1 := by
    unfold Census.hi; push_cast; ring
  rw [conjugacy h]
  unfold ColOpen
  refine forall_congr' fun g => forall_congr' fun _ => not_congr (or_congr ?_ ?_)
  · rw [← hlo]; exact Int.natCast_dvd_natCast
  · rw [← hhi]; exact Int.natCast_dvd_natCast

end TopMachine
