/-
THE SQUARE COLUMN, THE OFFSET-STRIKE LAW, THE BLIND CLASSES, THE SECTION'S TWINS, THE RUN
THROUGH THE SQUARE COLUMN, THE TWO-TOOTH FAMILY, AND THE RECORD ROUTE TO STEP 8
(Formalist, round 41).

Sources.  `research/proof/proof_skeleton.md` section 16 (the record route: a section is
`[p_k^2, p_{k+1}^2)`, the machine is every prime below `p_{k+1}`, and step 8 follows from
"the record `F` of the machine is shorter than the section"), sections 13-15 (the finer
statement between CONSECUTIVE prime squares, the square column, the offset-strike law and
the blind classes), and `research/proof/first_realisation.md` C3 and C4 (below the top of
the section the engine's runs are twin gaps; step 8 at the cut is `L_a(p) < l_p`, the run
through the square column against the section's length).  `research/proof/base_and_step.md`
for the blind-class statement of R4.d.i.a.

THE COORDINATE.  Column `k` is the slot `(6k - 1, 6k + 1)`; a gear `g` strikes column `k`
iff `g` divides a member.  The engine `{5..q}` and its `Blocked q k` are `OneStepE`'s; the
run `run Struck a` (the longest fully struck stretch beginning at `a`) is `CoreLeftover`'s,
with its boundedness `blocked_unbounded_open` already discharged for every engine.  For the
offset-strike law the column is an INTEGER (offsets from the square column run both ways),
`StrikesZ g k` over `ℤ`, bridged to the `ℕ` predicates at `1 ≤ k` (`strikesZ_natCast`,
`blockedZ_natCast`).

WHAT THE PROOFS ACTUALLY NEED (the findings of the kernel).
* S0 (`section_twin_of_unstruck`): the gear set `G` is an arbitrary `Finset ℕ`; the only
  hypothesis is that `G` CONTAINS every prime of `[5, P)` (`hfull`).  Primality of the gears
  is not used, extra gears are harmless, and the section's top enters only as
  `6(a + l) + 1 ≤ P^2`, i.e. the stretch stops at (does not include) the square column of
  `P`.  That bound is exact: the square column of `P` is open under every engine below `P`
  and is a twin iff `P^2 - 2` is prime (`OneStepE.blocked_W_iff`).
* S2 (`offset_strike`, `offset_strike_modEq`): the law `g` strikes `a + i` iff
  `p^2 ≡ -6i` or `p^2 ≡ 2 - 6i (mod g)` needs NOTHING of `p` or `g` beyond `p^2 = 6a + 1`;
  in particular the blind-class corollary (`blind_class`) holds for every integer `p` with
  `p^2 ≡ 1 (mod 6)`, not only for primes `p > g`.
* S4 (`L_lt_iff`): the equivalence "an unstruck column in `[a, a + l)` iff `L_a < l`" is
  the crossing S12 of `CoreLeftover` read at one start, and needs only the unboundedness of
  the open columns, which `blocked_unbounded_open` gives for free.
* S5 (`blockedZ_eq_family`): the real machine is the two-tooth family at the teeth
  `u_g = 6^{-1} (mod g)` and `-u_g`; the inverse exists because `g ≥ 5` is prime.

Zero sorries; no `native_decide`, no `decide`, no `Lean.ofReduceBool`.
-/
import CoreLeftover
import Mathlib.Data.Int.ModEq
import Mathlib.RingTheory.Coprime.Lemmas
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.Ring

namespace SquareColumn

open OneStepE CoreLeftover

/-! ## S0: the record route to step 8 (proof_skeleton.md section 16) -/

/-- Column `k = (6k - 1, 6k + 1)` is struck by the gear set `G`: some `g ∈ G` divides a member. -/
def StruckBy (G : Finset ℕ) (k : ℕ) : Prop := ∃ g ∈ G, g ∣ 6 * k - 1 ∨ g ∣ 6 * k + 1

/-- The engine `{5..q}` as a gear set: `StruckBy (primesIn q)` is `OneStepE.Blocked q`. -/
theorem struckBy_primesIn_iff {q k : ℕ} : StruckBy (primesIn q) k ↔ Blocked q k := by
  unfold StruckBy Blocked SmallFactor
  constructor
  · rintro ⟨g, hg, h | h⟩
    · obtain ⟨hp, h5, hq⟩ := mem_primesIn.mp hg
      exact Or.inl ⟨g, hp, h5, hq, h⟩
    · obtain ⟨hp, h5, hq⟩ := mem_primesIn.mp hg
      exact Or.inr ⟨g, hp, h5, hq, h⟩
  · rintro (⟨g, hp, h5, hq, h⟩ | ⟨g, hp, h5, hq, h⟩)
    · exact ⟨g, mem_primesIn.mpr ⟨hp, h5, hq⟩, Or.inl h⟩
    · exact ⟨g, mem_primesIn.mpr ⟨hp, h5, hq⟩, Or.inr h⟩

/-- A member coprime to 6 that no gear of `G` divides is `P`-rough whenever `G` contains
every prime of `[5, P)`. -/
theorem rough_of_not_dvd {G : Finset ℕ} {P n : ℕ} (hfull : ∀ r, r.Prime → 5 ≤ r → r < P → r ∈ G)
    (h2 : ¬ 2 ∣ n) (h3 : ¬ 3 ∣ n) (h : ∀ g ∈ G, ¬ g ∣ n) : Rough P n := by
  intro r hr hrn
  by_contra hlt
  exact h r (hfull r hr (five_le_of_dvd hr h2 h3 hrn) (not_le.mp hlt)) hrn

/-- **S0, one column** (proof_skeleton.md section 16, the record route).  A column `k ≥ 1`
with `6k + 1 < P^2`, struck by no gear of a set `G` that contains every prime of `[5, P)`,
is a twin prime pair.  Nothing else is assumed of `G`. -/
theorem section_twin_of_unstruck {G : Finset ℕ} {P k : ℕ}
    (hfull : ∀ r, r.Prime → 5 ≤ r → r < P → r ∈ G) (hk : 1 ≤ k) (hlt : 6 * k + 1 < P ^ 2)
    (h : ¬ StruckBy G k) : (6 * k - 1).Prime ∧ (6 * k + 1).Prime := by
  have h1 : ∀ g ∈ G, ¬ g ∣ 6 * k - 1 := fun g hg hd => h ⟨g, hg, Or.inl hd⟩
  have h2 : ∀ g ∈ G, ¬ g ∣ 6 * k + 1 := fun g hg hd => h ⟨g, hg, Or.inr hd⟩
  have hr1 := rough_of_not_dvd hfull (by omega) (by omega) h1
  have hr2 := rough_of_not_dvd hfull (by omega) (by omega) h2
  have e : 6 * k + 1 = 6 * k - 1 + 2 := by omega
  rw [e] at hr2 hlt ⊢
  exact twin_of_rough (by omega) hlt hr1 hr2

/-- **S0 on a stretch.**  On the stretch of columns `[a, a + l)` below the square column of
`P` (`6(a + l) + 1 ≤ P^2`), every column unstruck by `G` is a twin prime pair. -/
theorem stretch_twin_of_unstruck {G : Finset ℕ} {P a l i : ℕ}
    (hfull : ∀ r, r.Prime → 5 ≤ r → r < P → r ∈ G) (ha : 1 ≤ a)
    (hlt : 6 * (a + l) + 1 ≤ P ^ 2) (hi : i < l) (h : ¬ StruckBy G (a + i)) :
    (6 * (a + i) - 1).Prime ∧ (6 * (a + i) + 1).Prime :=
  section_twin_of_unstruck hfull (by omega) (by omega) h

/-- The converse on a stretch above the gears: a column above every gear of `G` (each gear
`2 ≤ g < 6k - 1`) is unstruck iff it is a twin prime pair. -/
theorem not_struckBy_iff_twin {G : Finset ℕ} {P k : ℕ}
    (hfull : ∀ r, r.Prime → 5 ≤ r → r < P → r ∈ G) (hG : ∀ g ∈ G, 2 ≤ g ∧ g < 6 * k - 1)
    (hk : 1 ≤ k) (hlt : 6 * k + 1 < P ^ 2) :
    ¬ StruckBy G k ↔ (6 * k - 1).Prime ∧ (6 * k + 1).Prime := by
  refine ⟨section_twin_of_unstruck hfull hk hlt, ?_⟩
  rintro ⟨hp1, hp2⟩ ⟨g, hg, hd | hd⟩
  · obtain ⟨h2, hlt'⟩ := hG g hg
    rcases hp1.eq_one_or_self_of_dvd g hd with h | h <;> omega
  · obtain ⟨h2, hlt'⟩ := hG g hg
    rcases hp2.eq_one_or_self_of_dvd g hd with h | h <;> omega

/-- **S0, the record route** (`section_twin_of_record`).  If every window of `F + 1`
consecutive columns contains a column unstruck by `G` (the record of `G` is at most `F`),
and the stretch `[a, a + l)` below the square column of `P` is longer than the record
(`F + 1 ≤ l`), then the stretch contains a twin prime pair. -/
theorem section_twin_of_record {G : Finset ℕ} {P a l F : ℕ}
    (hfull : ∀ r, r.Prime → 5 ≤ r → r < P → r ∈ G) (ha : 1 ≤ a)
    (hlt : 6 * (a + l) + 1 ≤ P ^ 2) (hF : ∀ x, ∃ j, j ≤ F ∧ ¬ StruckBy G (x + j))
    (hFl : F + 1 ≤ l) :
    ∃ i, i < l ∧ (6 * (a + i) - 1).Prime ∧ (6 * (a + i) + 1).Prime := by
  obtain ⟨j, hj, hopen⟩ := hF a
  exact ⟨j, by omega, stretch_twin_of_unstruck hfull ha hlt (by omega) hopen⟩

/-- **The record route for the engine `{5..q}`**, with the record as `CoreLeftover.run`:
if every run of `{5..q}` is at most `F` and the stretch `[a, a + l)` below `P^2` (no prime
in `(q, P)`) has `F + 1 ≤ l` columns, it contains a twin prime pair. -/
theorem section_twin_of_run_le {q P a l F : ℕ} (hgap : ∀ r, r.Prime → r < P → r ≤ q)
    (ha : 1 ≤ a) (hlt : 6 * (a + l) + 1 ≤ P ^ 2) (hF : ∀ x, run (Blocked q) x ≤ F)
    (hFl : F + 1 ≤ l) :
    ∃ i, i < l ∧ (6 * (a + i) - 1).Prime ∧ (6 * (a + i) + 1).Prime := by
  have hfull : ∀ r, r.Prime → 5 ≤ r → r < P → r ∈ primesIn q :=
    fun r hr h5 hlt => mem_primesIn.mpr ⟨hr, h5, hgap r hr hlt⟩
  refine section_twin_of_record hfull ha hlt (fun x => ?_) hFl
  by_contra hc
  push Not at hc
  have hcov : Covered (Blocked q) x (F + 1) :=
    fun i hi => struckBy_primesIn_iff.mp (hc i (by omega))
  have h1 := (le_run_iff (Blocked q) (blocked_unbounded_open q x)).mpr hcov
  have h2 := hF x
  omega

/-- The record route in the skeleton's own coordinates: the section `[W p, W P)` of the
engine `{5..q}` (`p ≤ P` coprime to 6, no prime in `(q, P)`) holds a twin prime pair as
soon as the engine's record `F` satisfies `F + 1 ≤ W P - W p`. -/
theorem section_twin_of_record_W {q p P F : ℕ} (hp2 : ¬ 2 ∣ p) (hp3 : ¬ 3 ∣ p)
    (hP2 : ¬ 2 ∣ P) (hP3 : ¬ 3 ∣ P) (h5 : 5 ≤ p) (hpP : p ≤ P)
    (hgap : ∀ r, r.Prime → r < P → r ≤ q) (hF : ∀ x, run (Blocked q) x ≤ F)
    (hFl : F + 1 ≤ W P - W p) :
    ∃ i, i < W P - W p ∧ (6 * (W p + i) - 1).Prime ∧ (6 * (W p + i) + 1).Prime := by
  have hWp := six_mul_W_add_one hp2 hp3
  have hWP := six_mul_W_add_one hP2 hP3
  have hsq : p ^ 2 ≤ P ^ 2 := Nat.pow_le_pow_left hpP 2
  have h25 : 25 ≤ p ^ 2 := by
    have := Nat.mul_le_mul h5 h5
    rw [sq]; omega
  exact section_twin_of_run_le hgap (by omega) (by omega) hF hFl

/-! ## S1: the square column -/

/-- **S1.**  For a prime `p ≥ 5`, `p^2 = 6a + 1` for exactly one `a`: the square is the
right member of the column `a`. -/
theorem square_column {p : ℕ} (hp : p.Prime) (h5 : 5 ≤ p) : ∃! a, p ^ 2 = 6 * a + 1 := by
  obtain ⟨h2, h3⟩ := coprime_six_of_prime hp h5
  obtain ⟨t, ht⟩ := exists_sq_eq_six h2 h3
  exact ⟨t, ht, fun a ha => by omega⟩

/-- The square column is `W p = (p^2 - 1)/6` of `OneStepE`: its right member is `p^2` and its
left member is `p^2 - 2`. -/
theorem square_column_W {p : ℕ} (hp : p.Prime) (h5 : 5 ≤ p) :
    p ^ 2 = 6 * W p + 1 ∧ 6 * W p - 1 = p ^ 2 - 2 := by
  obtain ⟨h2, h3⟩ := coprime_six_of_prime hp h5
  have := six_mul_W_add_one h2 h3
  constructor <;> omega

/-- The column of `p^2` is `W p`: the unique `a` of S1. -/
theorem eq_W_of_sq {p a : ℕ} (hp : p.Prime) (h5 : 5 ≤ p) (ha : p ^ 2 = 6 * a + 1) : a = W p := by
  have := (square_column_W hp h5).1
  omega

/-! ## S2: the offset-strike law and the blind classes (proof_skeleton.md section 15) -/

/-- `StrikesZ g k`: gear `g` strikes the integer column `k = (6k - 1, 6k + 1)`. -/
def StrikesZ (g : ℕ) (k : ℤ) : Prop := (g : ℤ) ∣ 6 * k - 1 ∨ (g : ℤ) ∣ 6 * k + 1

/-- The integer column predicate agrees with the `ℕ` one at every column `k ≥ 1`. -/
theorem strikesZ_natCast {g k : ℕ} (hk : 1 ≤ k) :
    StrikesZ g k ↔ (g ∣ 6 * k - 1 ∨ g ∣ 6 * k + 1) := by
  unfold StrikesZ
  have e1 : (6 * (k : ℤ) - 1) = ((6 * k - 1 : ℕ) : ℤ) := by omega
  have e2 : (6 * (k : ℤ) + 1) = ((6 * k + 1 : ℕ) : ℤ) := by omega
  rw [e1, e2, Int.natCast_dvd_natCast, Int.natCast_dvd_natCast]

/-- `Blocked q k` in the integer coordinate. -/
def BlockedZ (q : ℕ) (k : ℤ) : Prop := ∃ g, g.Prime ∧ 5 ≤ g ∧ g ≤ q ∧ StrikesZ g k

theorem blockedZ_natCast {q k : ℕ} (hk : 1 ≤ k) : BlockedZ q k ↔ Blocked q k := by
  unfold BlockedZ Blocked SmallFactor
  constructor
  · rintro ⟨g, hp, h5, hq, h⟩
    rcases (strikesZ_natCast hk).mp h with h | h
    · exact Or.inl ⟨g, hp, h5, hq, h⟩
    · exact Or.inr ⟨g, hp, h5, hq, h⟩
  · rintro (⟨g, hp, h5, hq, h⟩ | ⟨g, hp, h5, hq, h⟩)
    · exact ⟨g, hp, h5, hq, (strikesZ_natCast hk).mpr (Or.inl h)⟩
    · exact ⟨g, hp, h5, hq, (strikesZ_natCast hk).mpr (Or.inr h)⟩

/-- **S2, the offset-strike law** (proof_skeleton.md section 15).  With `p^2 = 6a + 1`, a
gear `g` strikes the column `a + i` iff `g ∣ p^2 + 6i - 2` (the left member) or
`g ∣ p^2 + 6i` (the right member).  No hypothesis on `p` or `g`. -/
theorem offset_strike {p a : ℕ} (ha : p ^ 2 = 6 * a + 1) (g : ℕ) (i : ℤ) :
    StrikesZ g (a + i) ↔
      ((g : ℤ) ∣ (p : ℤ) ^ 2 + 6 * i - 2 ∨ (g : ℤ) ∣ (p : ℤ) ^ 2 + 6 * i) := by
  have h : (p : ℤ) ^ 2 = 6 * (a : ℤ) + 1 := by exact_mod_cast ha
  have e1 : 6 * ((a : ℤ) + i) - 1 = (p : ℤ) ^ 2 + 6 * i - 2 := by rw [h]; ring
  have e2 : 6 * ((a : ℤ) + i) + 1 = (p : ℤ) ^ 2 + 6 * i := by rw [h]; ring
  unfold StrikesZ
  rw [e1, e2]

/-- **S2 in congruence form.**  `g` strikes `a + i` iff `p^2 ≡ 2 - 6i` or `p^2 ≡ -6i (mod g)`. -/
theorem offset_strike_modEq {p a : ℕ} (ha : p ^ 2 = 6 * a + 1) (g : ℕ) (i : ℤ) :
    StrikesZ g (a + i) ↔
      ((p : ℤ) ^ 2 ≡ 2 - 6 * i [ZMOD g] ∨ (p : ℤ) ^ 2 ≡ -(6 * i) [ZMOD g]) := by
  rw [offset_strike ha, Int.modEq_iff_dvd, Int.modEq_iff_dvd]
  have e1 : 2 - 6 * i - (p : ℤ) ^ 2 = -((p : ℤ) ^ 2 + 6 * i - 2) := by ring
  have e2 : -(6 * i) - (p : ℤ) ^ 2 = -((p : ℤ) ^ 2 + 6 * i) := by ring
  rw [e1, e2, dvd_neg, dvd_neg]

/-- `r` is a square modulo `g`. -/
def SquareMod (g : ℕ) (r : ℤ) : Prop := ∃ s : ℤ, s ^ 2 ≡ r [ZMOD g]

/-- **S2, the blind-class corollary** (base_and_step.md, R4.d.i.a; proof_skeleton.md
section 15).  If neither `2 - 6i` nor `-6i` is a square modulo `g`, then `g` strikes the
offset `i` from the square column of NO `p` with `p^2 = 6a + 1` - primality of `p` and
`p > g` are not needed. -/
theorem blind_class {g : ℕ} {i : ℤ} (h1 : ¬ SquareMod g (2 - 6 * i))
    (h2 : ¬ SquareMod g (-(6 * i))) {p a : ℕ} (ha : p ^ 2 = 6 * a + 1) :
    ¬ StrikesZ g (a + i) := by
  rw [offset_strike_modEq ha]
  rintro (h | h)
  · exact h1 ⟨(p : ℤ), h⟩
  · exact h2 ⟨(p : ℤ), h⟩

/-- The blind-class corollary in the form the brief states it: for every prime `p > g`. -/
theorem blind_class_prime {g : ℕ} {i : ℤ} (h1 : ¬ SquareMod g (2 - 6 * i))
    (h2 : ¬ SquareMod g (-(6 * i))) {p : ℕ} (hp : p.Prime) (h5 : 5 ≤ p) (_hgp : g < p) :
    ¬ StrikesZ g (W p + i) :=
  blind_class h1 h2 (square_column_W hp h5).1

/-- **S2, the four classes** (proof_skeleton.md section 15): `g` strikes `a + i` iff `p` is
congruent to `±s` for some root `s` of `2 - 6i` or of `-6i` modulo `g`. -/
theorem strikesZ_iff_root {p a : ℕ} (ha : p ^ 2 = 6 * a + 1) (g : ℕ) (i : ℤ) :
    StrikesZ g (a + i) ↔
      ∃ s : ℤ, (s ^ 2 ≡ 2 - 6 * i [ZMOD g] ∨ s ^ 2 ≡ -(6 * i) [ZMOD g]) ∧
        ((p : ℤ) ≡ s [ZMOD g] ∨ (p : ℤ) ≡ -s [ZMOD g]) := by
  rw [offset_strike_modEq ha]
  constructor
  · intro h
    exact ⟨(p : ℤ), h, Or.inl Int.ModEq.rfl⟩
  · rintro ⟨s, hs, hps⟩
    have hsq : (p : ℤ) ^ 2 ≡ s ^ 2 [ZMOD g] := by
      rcases hps with h | h
      · exact h.pow 2
      · have := h.pow 2
        rwa [neg_sq] at this
    rcases hs with h | h
    · exact Or.inl (hsq.trans h)
    · exact Or.inr (hsq.trans h)

/-! ## S3: below the top of the section, open is twin (first_realisation.md C3) -/

/-- **S3, one column, both directions.**  For a column above the engine (`q < 6k - 1`) and
below `P^2` (no prime in `(q, P)`): open under `{5..q}` iff a twin prime pair.
(`CoreLeftover.not_blocked_iff_twin` with `P` in place of `q + 1`.) -/
theorem not_blocked_iff_twin_of_lt_sq {q P k : ℕ} (hgap : ∀ r, r.Prime → r < P → r ≤ q)
    (hqk : q < 6 * k - 1) (hlt : 6 * k + 1 < P ^ 2) :
    ¬ Blocked q k ↔ (6 * k - 1).Prime ∧ (6 * k + 1).Prime := by
  have e1 := smallFactor_iff_not_prime hgap (n := 6 * k - 1) (by omega) (by omega) hqk
    (lt_of_le_of_lt (by omega) hlt) (by omega)
  have e2 := smallFactor_iff_not_prime hgap (n := 6 * k + 1) (by omega) (by omega)
    (by omega) hlt (by omega)
  unfold Blocked
  rw [e1, e2]
  tauto

/-- The section from the square column of `p` to the square column of `P`, in columns:
`l = (P^2 - p^2)/6 = W P - W p`. -/
theorem W_sub_W {p P : ℕ} (hp2 : ¬ 2 ∣ p) (hp3 : ¬ 3 ∣ p) (hP2 : ¬ 2 ∣ P) (hP3 : ¬ 3 ∣ P) :
    (P ^ 2 - p ^ 2) / 6 = W P - W p := by
  have h1 := six_mul_W_add_one hp2 hp3
  have h2 := six_mul_W_add_one hP2 hP3
  omega

/-- **S3 on the section** (first_realisation.md C3).  The section `[W p, W P)` of the engine
`{5..q}` (`p` coprime to 6, `P` coprime to 6, no prime in `(q, P)`): a column `W p + i` with
`i < W P - W p`, above the engine, is open iff it is a twin prime pair.  The bound `i < l`
is exact: at `i = l` the column is `W P`, open under `{5..q}` on its right member `P^2` and
a twin only if `P^2 - 2` is prime (`OneStepE.blocked_W_iff`). -/
theorem section_open_iff_twin {q p P i : ℕ} (hp2 : ¬ 2 ∣ p) (hp3 : ¬ 3 ∣ p)
    (hP2 : ¬ 2 ∣ P) (hP3 : ¬ 3 ∣ P) (hgap : ∀ r, r.Prime → r < P → r ≤ q)
    (hq : q < 6 * (W p + i) - 1) (hi : i < W P - W p) :
    ¬ Blocked q (W p + i) ↔ (6 * (W p + i) - 1).Prime ∧ (6 * (W p + i) + 1).Prime := by
  have h1 := six_mul_W_add_one hp2 hp3
  have h2 := six_mul_W_add_one hP2 hP3
  exact not_blocked_iff_twin_of_lt_sq hgap hq (by omega)

/-- S3 in the brief's form: a column of the section struck by no gear of `{5..q}` is a twin
prime pair (the one-sided half needs only `1 ≤ W p`, not `q < 6k - 1`). -/
theorem section_twin_of_open {q p P i : ℕ} (hp2 : ¬ 2 ∣ p) (hp3 : ¬ 3 ∣ p) (h5 : 5 ≤ p)
    (hP2 : ¬ 2 ∣ P) (hP3 : ¬ 3 ∣ P) (hgap : ∀ r, r.Prime → r < P → r ≤ q)
    (hi : i < W P - W p) (h : ¬ Blocked q (W p + i)) :
    (6 * (W p + i) - 1).Prime ∧ (6 * (W p + i) + 1).Prime := by
  have h1 := six_mul_W_add_one hp2 hp3
  have h2 := six_mul_W_add_one hP2 hP3
  have h25 : 25 ≤ p ^ 2 := by
    have := Nat.mul_le_mul h5 h5
    rw [sq]; omega
  have hfull : ∀ r, r.Prime → 5 ≤ r → r < P → r ∈ primesIn q :=
    fun r hr h5 hlt => mem_primesIn.mpr ⟨hr, h5, hgap r hr hlt⟩
  exact section_twin_of_unstruck hfull (by omega) (by omega)
    (fun hs => h (struckBy_primesIn_iff.mp hs))

/-! ## S4: step 8 at the cut is `L_a < l` (first_realisation.md C4) -/

/-- `L q a`: the length of the fully struck run of `{5..q}` beginning at column `a` (the
number of consecutive columns from `a` on, each struck by a gear of `{5..q}`); the
`CoreLeftover.run` of the machine at `a`. -/
noncomputable def L (q a : ℕ) : ℕ := run (Blocked q) a

/-- The run at `a` is the offset of the first open column: if `a, …, a + r - 1` are struck
and `a + r` is open then `run a = r`. -/
theorem run_eq_of_first_open (Struck : ℕ → Prop) {a r : ℕ}
    (hcov : ∀ i, i < r → Struck (a + i)) (hopen : ¬ Struck (a + r)) : run Struck a = r := by
  obtain ⟨hmem, hub⟩ := run_isGreatest Struck ⟨r, hopen⟩
  have hmem' : Covered Struck a (run Struck a) := hmem
  apply le_antisymm
  · by_contra hc
    exact hopen (hmem' r (not_le.mp hc))
  · exact hub (show Covered Struck a r from hcov)

/-- `L q a` is the offset of the first open column above `a`. -/
theorem L_eq_of_first_open {q a r : ℕ} (hcov : ∀ i, i < r → Blocked q (a + i))
    (hopen : ¬ Blocked q (a + r)) : L q a = r :=
  run_eq_of_first_open (Blocked q) hcov hopen

/-- **S4** (first_realisation.md C4).  There is an open column of `{5..q}` in `[a, a + l)`
iff the run through `a` is shorter than `l`.  Unconditional: every engine leaves an open
column beyond every position (`blocked_unbounded_open`). -/
theorem L_lt_iff {q a l : ℕ} : L q a < l ↔ ∃ i, i < l ∧ ¬ Blocked q (a + i) := by
  unfold L
  rw [← not_le, le_run_iff (Blocked q) (blocked_unbounded_open q a)]
  show ¬ (∀ i, i < l → Blocked q (a + i)) ↔ _
  constructor
  · intro h
    by_contra hc
    push Not at hc
    exact h hc
  · rintro ⟨i, hi, hopen⟩ h
    exact hopen (h i hi)

/-- S4 read the other way: the stretch `[a, a + l)` is fully struck iff `l ≤ L q a`. -/
theorem le_L_iff {q a l : ℕ} : l ≤ L q a ↔ ∀ i, i < l → Blocked q (a + i) := by
  rw [← not_lt, L_lt_iff]
  push Not
  rfl

/-- **C4 in full: step 8 at the cut `p` is `L_a(p) < l_p`.**  For the section `[W p, W P)`
of the engine `{5..q}` (`q < 6 W p - 1`, no prime in `(q, P)`): the section holds a twin
prime pair iff the run through the square column is shorter than the section. -/
theorem twin_in_section_iff_L_lt {q p P : ℕ} (hp2 : ¬ 2 ∣ p) (hp3 : ¬ 3 ∣ p)
    (hP2 : ¬ 2 ∣ P) (hP3 : ¬ 3 ∣ P) (hgap : ∀ r, r.Prime → r < P → r ≤ q)
    (hq : q < 6 * W p - 1) :
    (∃ i, i < W P - W p ∧ (6 * (W p + i) - 1).Prime ∧ (6 * (W p + i) + 1).Prime) ↔
      L q (W p) < W P - W p := by
  rw [L_lt_iff]
  constructor
  · rintro ⟨i, hi, htw⟩
    exact ⟨i, hi, (section_open_iff_twin hp2 hp3 hP2 hP3 hgap (by omega) hi).mpr htw⟩
  · rintro ⟨i, hi, hopen⟩
    exact ⟨i, hi, (section_open_iff_twin hp2 hp3 hP2 hP3 hgap (by omega) hi).mp hopen⟩

/-- The run through the square column is the first-twin offset above `p^2` whenever it is
shorter than the section: `L q (W p) = r` with `W p + r` the first twin column above `W p`. -/
theorem L_eq_first_twin_offset {q p P r : ℕ} (hp2 : ¬ 2 ∣ p) (hp3 : ¬ 3 ∣ p)
    (hP2 : ¬ 2 ∣ P) (hP3 : ¬ 3 ∣ P) (hgap : ∀ r, r.Prime → r < P → r ≤ q)
    (hq : q < 6 * W p - 1) (hr : r < W P - W p)
    (hcov : ∀ i, i < r → ¬ ((6 * (W p + i) - 1).Prime ∧ (6 * (W p + i) + 1).Prime))
    (htw : (6 * (W p + r) - 1).Prime ∧ (6 * (W p + r) + 1).Prime) : L q (W p) = r := by
  apply L_eq_of_first_open
  · intro i hi
    by_contra hopen
    exact hcov i hi
      ((section_open_iff_twin hp2 hp3 hP2 hP3 hgap (by omega) (by omega)).mp hopen)
  · exact (section_open_iff_twin hp2 hp3 hP2 hP3 hgap (by omega) hr).mpr htw

/-! ## S5: the two-tooth family, and the real machine as its special case -/

/-- `Teeth v w g k`: the gear `g` with the two teeth `v, w` strikes the integer column `k`
iff `k ≡ v` or `k ≡ w (mod g)`. -/
def Teeth (v w : ℤ) (g : ℕ) (k : ℤ) : Prop := k ≡ v [ZMOD g] ∨ k ≡ w [ZMOD g]

/-- The two-tooth family on the engine `{5..q}`: an arbitrary assignment of two residues
`v g, w g` per gear. -/
def FamilyBlocked (v w : ℕ → ℤ) (q : ℕ) (k : ℤ) : Prop :=
  ∃ g, g.Prime ∧ 5 ≤ g ∧ g ≤ q ∧ Teeth (v g) (w g) g k

/-- The teeth read from the square column: `a + i` is struck iff the offset `i` is
`≡ v - a` or `≡ w - a (mod g)`. -/
theorem teeth_offset (v w : ℤ) (g : ℕ) (a i : ℤ) :
    Teeth v w g (a + i) ↔ Teeth (v - a) (w - a) g i := by
  have key : ∀ c : ℤ, (a + i ≡ c [ZMOD g] ↔ i ≡ c - a [ZMOD g]) := by
    intro c
    constructor
    · intro h
      have := h.sub_right a
      rwa [add_sub_cancel_left] at this
    · intro h
      have := h.add_left a
      rwa [add_sub_cancel] at this
  unfold Teeth
  rw [key v, key w]

/-- `6` is invertible modulo a prime `g ≥ 5`. -/
theorem exists_six_inv {g : ℕ} (hg : g.Prime) (h5 : 5 ≤ g) : ∃ u : ℤ, 6 * u ≡ 1 [ZMOD g] := by
  have h2 : Nat.Coprime 2 g := (Nat.coprime_primes Nat.prime_two hg).mpr (by omega)
  have h3 : Nat.Coprime 3 g := (Nat.coprime_primes Nat.prime_three hg).mpr (by omega)
  have h6 : Nat.Coprime 6 g := Nat.Coprime.mul_left h2 h3
  obtain ⟨x, y, hxy⟩ := Nat.isCoprime_iff_coprime.mpr h6
  push_cast at hxy
  refine ⟨x, ?_⟩
  rw [Int.modEq_iff_dvd]
  exact ⟨y, by linear_combination -hxy⟩

/-- **The real teeth.**  With `6u ≡ 1 (mod g)`, the gear `g` strikes the column `k` iff
`k ≡ u` or `k ≡ -u (mod g)`: the real machine's teeth are `±6^{-1}`. -/
theorem strikesZ_iff_teeth {g : ℕ} {u : ℤ} (hu : 6 * u ≡ 1 [ZMOD g]) (k : ℤ) :
    StrikesZ g k ↔ Teeth u (-u) g k := by
  have key : ∀ c : ℤ, (k ≡ c [ZMOD g] ↔ 6 * k ≡ 6 * c [ZMOD g]) := by
    intro c
    constructor
    · exact fun h => h.mul_left 6
    · intro h
      calc k = 1 * k := (one_mul k).symm
        _ ≡ (6 * u) * k [ZMOD g] := hu.symm.mul_right k
        _ = u * (6 * k) := by ring
        _ ≡ u * (6 * c) [ZMOD g] := h.mul_left u
        _ = (6 * u) * c := by ring
        _ ≡ 1 * c [ZMOD g] := hu.mul_right c
        _ = c := one_mul c
  have h1 : k ≡ u [ZMOD g] ↔ (g : ℤ) ∣ 6 * k - 1 := by
    rw [key u]
    constructor
    · intro h
      have := h.trans hu
      rw [Int.modEq_iff_dvd] at this
      rwa [← neg_sub, dvd_neg] at this
    · intro h
      have h' : 6 * k ≡ 1 [ZMOD g] := by
        rw [Int.modEq_iff_dvd, ← neg_sub, dvd_neg]
        exact h
      exact h'.trans hu.symm
  have hu' : 6 * (-u) ≡ -1 [ZMOD g] := by
    rw [mul_neg]
    exact hu.neg
  have h2 : k ≡ -u [ZMOD g] ↔ (g : ℤ) ∣ 6 * k + 1 := by
    rw [key (-u)]
    have e : (-1 : ℤ) - 6 * k = -(6 * k + 1) := by ring
    constructor
    · intro h
      have := h.trans hu'
      rw [Int.modEq_iff_dvd, e, dvd_neg] at this
      exact this
    · intro h
      have h' : 6 * k ≡ -1 [ZMOD g] := by
        rw [Int.modEq_iff_dvd, e, dvd_neg]
        exact h
      exact h'.trans hu'.symm
  unfold StrikesZ Teeth
  rw [h1, h2]

/-- Every gear `g ≥ 5` of the real machine is a two-tooth gear with teeth `u, -u`,
`6u ≡ 1 (mod g)`. -/
theorem real_teeth {g : ℕ} (hg : g.Prime) (h5 : 5 ≤ g) :
    ∃ u : ℤ, 6 * u ≡ 1 [ZMOD g] ∧ ∀ k : ℤ, StrikesZ g k ↔ Teeth u (-u) g k := by
  obtain ⟨u, hu⟩ := exists_six_inv hg h5
  exact ⟨u, hu, strikesZ_iff_teeth hu⟩

/-- **S5.**  The real machine is the special case of the two-tooth family at the teeth
`v g = 6^{-1}`, `w g = -6^{-1} (mod g)`: one assignment `v, w` serves every engine `{5..q}`. -/
theorem blockedZ_eq_family :
    ∃ v w : ℕ → ℤ, ∀ q : ℕ, ∀ k : ℤ, BlockedZ q k ↔ FamilyBlocked v w q k := by
  have h : ∀ g : ℕ, ∃ u : ℤ, g.Prime → 5 ≤ g → ∀ k : ℤ, StrikesZ g k ↔ Teeth u (-u) g k := by
    intro g
    by_cases hg : g.Prime ∧ 5 ≤ g
    · obtain ⟨u, -, hu⟩ := real_teeth hg.1 hg.2
      exact ⟨u, fun _ _ => hu⟩
    · exact ⟨0, fun hp h5 => absurd ⟨hp, h5⟩ hg⟩
  choose u hu using h
  refine ⟨u, fun g => -(u g), fun q k => ?_⟩
  unfold BlockedZ FamilyBlocked
  exact exists_congr fun g => and_congr_right fun hp => and_congr_right fun h5 =>
    and_congr_right fun _ => hu g hp h5 k

end SquareColumn
