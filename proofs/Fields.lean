/-
THE FIELDS: THE FOLD'S SURVIVORS SPLIT BY THE COUNT OF PRIME FACTORS - THE PRIME FIELD IS THE
TWIN'S DEFINITION, THE SQUARE FIELD IS RIGHT-ONLY AND ONE COLUMN PER PRIME, THE SQUARE FIELD
CANNOT COVER A SECTION, THE CLASS RULE, THE MIRROR LAW OF A DILATE, THE NESTING PER FIELD, AND
THE OVERLAY AS THE COMPOSITES  (Formalist).

Sources.  `research/proof/fields.md` section 0.1 (the objects, by construction) and section 1,
the exact facts E1-E8; `research/proof/proof_skeleton.md` IV.3a (the fields as a construction).

THE COORDINATE.  The fold's survivors are `S = {n : n % 6 = 1 ∨ n % 6 = 5}` (`InS`).  Field `j`
(`field j`) is the set of survivors with exactly `j` prime factors counted with multiplicity,
`n.primeFactorsList.length = j`; every prime factor of a survivor is `≥ 5`
(`five_le_of_mem_primeFactorsList`), so this is the document's `Omega`.  Column `k` is the slot
`(6k - 1, 6k + 1)` of `OneStepE`/`SquareColumn`; a set `A` HITS column `k` (`Hits A k`) if a
member of the column lies in `A`.  The square field is `{p^2 : p prime, p ≥ 5}`; the overlay is
the union of the fields `j ≥ 2`.  The square column `W p = (p^2 - 1)/6` is `OneStepE.W`.

WHAT THE PROOFS ACTUALLY NEED (the findings of the kernel).
* F1: `field 1` is the primes `≥ 5` with no hypothesis (`5 ≤` comes from membership in `S`: a
  prime in `S` is not 2 or 3); the twin statement needs only `1 ≤ k`.
* F2: `p^2 ≡ 1 (mod 6)` needs only `p` coprime to 6 (`OneStepE.exists_sq_eq_six`); primality
  enters only through `coprime_six_of_prime`.
* F3: the injection column `↦ sqrt(6k + 1)` needs no hypothesis at all; the bound
  `#(primes of [p, P)) ≤ P - p` and the consecutive-prime bound `≤ 1` need only `p ≤ P`,
  both prime `≥ 5`; and the section `[W p, W P)` of two consecutive primes ALWAYS has at least
  4 columns (`four_le_W_sub_W`), so the square field never covers it (`squares_cannot_cover_W`).
* F4: the class rule needs only that every factor is `≡ ±1 (mod 6)`; primality is not used
  (`prod_mod_six`), and the field form is the identity `n = prod (primeFactorsList n)`.
* F5: the mirror law `g(6m ∓ 1) = 6gm ∓ g` needs NOTHING of `g` (not `g ≥ 5`, not `g ∈ S`);
  `m ≥ 1` only to keep `6m - 1` honest in `ℕ`.  The column form `k₁ + k₂ = 2gm` likewise.
* F6: `n ∈ field j ↔ p n ∈ field (j + 1)` needs `p` prime and `p ∈ S` (i.e. `p ≥ 5`); the
  length step is `perm_primeFactorsList_mul`, the `S`-step is `(±1)·S = S` (`inS_mul_iff`).
* F7: the overlay on `S \ {1}` is exactly the composites; no engine and no size hypothesis.
  With `OneStepE.smallFactor_iff_not_prime` this is E1: below `q'^2` the struck set is the overlay.

Zero sorries; no `native_decide`, no `decide`, no `Lean.ofReduceBool`.
-/
import SquareColumn
import Mathlib.Data.Nat.Factors
import Mathlib.Data.Nat.Sqrt
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

namespace Fields

open OneStepE CoreLeftover SquareColumn

/-! ## The objects (fields.md 0.1) -/

/-- The fold's survivors: `n ≡ ±1 (mod 6)`. -/
def InS (n : ℕ) : Prop := n % 6 = 1 ∨ n % 6 = 5

/-- Field `j`: the survivors with exactly `j` prime factors counted with multiplicity. -/
def field (j : ℕ) : Set ℕ := {n | InS n ∧ n.primeFactorsList.length = j}

/-- A set `A` hits column `k = (6k - 1, 6k + 1)`: a member of the column lies in `A`. -/
def Hits (A : Set ℕ) (k : ℕ) : Prop := 6 * k - 1 ∈ A ∨ 6 * k + 1 ∈ A

/-- The square field: the squares of the primes `≥ 5` (the diagonal of field 2). -/
def squareField : Set ℕ := {n | ∃ p, p.Prime ∧ 5 ≤ p ∧ n = p ^ 2}

/-- The overlay: the union of the fields `j ≥ 2`. -/
def overlay : Set ℕ := {n | ∃ j, 2 ≤ j ∧ n ∈ field j}

theorem mem_field {n j : ℕ} : n ∈ field j ↔ InS n ∧ n.primeFactorsList.length = j := Iff.rfl

theorem inS_iff {n : ℕ} : InS n ↔ ¬ 2 ∣ n ∧ ¬ 3 ∣ n := by
  unfold InS; omega

theorem ne_zero_of_inS {n : ℕ} (h : InS n) : n ≠ 0 := by
  unfold InS at h; omega

theorem inS_of_prime {p : ℕ} (hp : p.Prime) (h5 : 5 ≤ p) : InS p :=
  inS_iff.mpr (coprime_six_of_prime hp h5)

/-- Every prime factor of a survivor is `≥ 5`: the fields count only gears. -/
theorem five_le_of_mem_primeFactorsList {n p : ℕ} (hn : InS n) (hp : p ∈ n.primeFactorsList) :
    5 ≤ p :=
  five_le_of_dvd (Nat.prime_of_mem_primeFactorsList hp) (inS_iff.mp hn).1 (inS_iff.mp hn).2
    (Nat.dvd_of_mem_primeFactorsList hp)

theorem length_eq_one_exists {l : List ℕ} (h : l.length = 1) : ∃ a, l = [a] := by
  match l with
  | [a] => exact ⟨a, rfl⟩
  | [] => simp at h
  | _ :: _ :: _ => simp at h

/-! ## F1: the prime field (fields.md 0.1, E1) -/

/-- **F1.**  Field 1 is exactly the primes `≥ 5`. -/
theorem field_one_iff_prime_ge_five {n : ℕ} : n ∈ field 1 ↔ n.Prime ∧ 5 ≤ n := by
  rw [mem_field]
  constructor
  · rintro ⟨hS, hlen⟩
    have hn0 : n ≠ 0 := ne_zero_of_inS hS
    obtain ⟨p, hp⟩ := length_eq_one_exists hlen
    have hprod := Nat.prod_primeFactorsList hn0
    rw [hp, List.prod_singleton] at hprod
    have hpp : p.Prime := Nat.prime_of_mem_primeFactorsList (by rw [hp]; exact List.mem_singleton_self p)
    have h2 := hpp.two_le
    unfold InS at hS
    rw [← hprod] at hS ⊢
    exact ⟨hpp, by omega⟩
  · rintro ⟨hp, h5⟩
    refine ⟨inS_of_prime hp h5, ?_⟩
    rw [Nat.primeFactorsList_prime hp]; rfl

/-- **F1, the twin.**  Column `k ≥ 1` is a twin prime pair iff both members lie in field 1:
the prime field is the twin's DEFINITION, not a striker. -/
theorem prime_field_never_kills {k : ℕ} (hk : 1 ≤ k) :
    ((6 * k - 1).Prime ∧ (6 * k + 1).Prime) ↔ (6 * k - 1 ∈ field 1 ∧ 6 * k + 1 ∈ field 1) := by
  rw [field_one_iff_prime_ge_five, field_one_iff_prime_ge_five]
  constructor
  · rintro ⟨h1, h2⟩; exact ⟨⟨h1, by omega⟩, ⟨h2, by omega⟩⟩
  · rintro ⟨⟨h1, -⟩, ⟨h2, -⟩⟩; exact ⟨h1, h2⟩

/-! ## F2: the square field is right-only and one column per prime (E5, E7) -/

theorem sq_mod_six {p : ℕ} (hp : p.Prime) (h5 : 5 ≤ p) : p ^ 2 % 6 = 1 := by
  obtain ⟨h2, h3⟩ := coprime_six_of_prime hp h5
  obtain ⟨t, ht⟩ := exists_sq_eq_six h2 h3
  omega

/-- **F2, right-only.**  The square of a prime `p ≥ 5` is `≡ 1 (mod 6)`: it is a right member
`6k + 1` and never a left member `6k - 1`. -/
theorem square_right_only {p : ℕ} (hp : p.Prime) (h5 : 5 ≤ p) :
    (∃ k, p ^ 2 = 6 * k + 1) ∧ ∀ k, p ^ 2 ≠ 6 * k - 1 := by
  have h := sq_mod_six hp h5
  exact ⟨⟨p ^ 2 / 6, by omega⟩, fun k hk => by omega⟩

/-- The square field is blind to every left member. -/
theorem squareField_not_left {k : ℕ} : 6 * k - 1 ∉ squareField := by
  rintro ⟨p, hp, h5, h⟩
  exact (square_right_only hp h5).2 k h.symm

/-- The square field hits column `k` iff its right member is a prime square. -/
theorem hits_squareField_iff {k : ℕ} :
    Hits squareField k ↔ ∃ p, p.Prime ∧ 5 ≤ p ∧ 6 * k + 1 = p ^ 2 := by
  unfold Hits
  constructor
  · rintro (h | h)
    · exact absurd h squareField_not_left
    · exact h
  · intro h
    exact Or.inr h

/-- **F2, one column.**  Each prime `p ≥ 5` hits exactly one column, `k = (p^2 - 1)/6`. -/
theorem square_column_unique {p : ℕ} (hp : p.Prime) (h5 : 5 ≤ p) :
    ∃! k, p ^ 2 = 6 * k - 1 ∨ p ^ 2 = 6 * k + 1 := by
  have h := sq_mod_six hp h5
  exact ⟨(p ^ 2 - 1) / 6, Or.inr (by omega), fun k hk => by omega⟩

/-- The unique column is the square column `W p` of `OneStepE`. -/
theorem square_column_eq_W {p k : ℕ} (hp : p.Prime) (h5 : 5 ≤ p)
    (hk : p ^ 2 = 6 * k - 1 ∨ p ^ 2 = 6 * k + 1) : k = W p := by
  have h := sq_mod_six hp h5
  have hW := (square_column_W hp h5).1
  omega

/-- The square field hits `W p`, and only there, for each prime `p ≥ 5`. -/
theorem hits_squareField_W {p : ℕ} (hp : p.Prime) (h5 : 5 ≤ p) : Hits squareField (W p) :=
  hits_squareField_iff.mpr ⟨p, hp, h5, (square_column_W hp h5).1.symm⟩

/-! ## F3: the square field cannot cover a stretch (E7) -/

open Classical in
/-- The columns of the stretch `[a, a + l)` hit by the square field. -/
noncomputable def squareHits (a l : ℕ) : Finset ℕ :=
  (Finset.Ico a (a + l)).filter (Hits squareField)

/-- The primes `p ≥ 5` whose squares land in `[6a + 1, 6(a + l) + 1)`. -/
def sectionPrimes (a l : ℕ) : Finset ℕ :=
  (Finset.range (6 * (a + l) + 1)).filter
    (fun p => p.Prime ∧ 5 ≤ p ∧ 6 * a + 1 ≤ p ^ 2 ∧ p ^ 2 < 6 * (a + l) + 1)

theorem mem_sectionPrimes {a l p : ℕ} :
    p ∈ sectionPrimes a l ↔ p.Prime ∧ 5 ≤ p ∧ 6 * a + 1 ≤ p ^ 2 ∧ p ^ 2 < 6 * (a + l) + 1 := by
  unfold sectionPrimes
  rw [Finset.mem_filter, Finset.mem_range]
  constructor
  · rintro ⟨-, h⟩; exact h
  · rintro ⟨hp, h5, hlo, hhi⟩
    refine ⟨?_, hp, h5, hlo, hhi⟩
    have : p ≤ p ^ 2 := by rw [sq]; exact Nat.le_mul_self p
    omega

open Classical in
/-- **F3, the count.**  The columns of `[a, a + l)` hit by the square field are at most the
primes whose squares land in `[6a + 1, 6(a + l) + 1)`: column `k ↦ sqrt(6k + 1)` is an
injection.  No hypothesis. -/
theorem squares_in_section_le (a l : ℕ) : (squareHits a l).card ≤ (sectionPrimes a l).card := by
  apply Finset.card_le_card_of_injOn (fun k => Nat.sqrt (6 * k + 1))
  · intro k hk
    simp only [Finset.mem_coe, squareHits, Finset.mem_filter, Finset.mem_Ico] at hk
    obtain ⟨⟨hak, hkl⟩, hhit⟩ := hk
    obtain ⟨p, hp, h5, hsq⟩ := hits_squareField_iff.mp hhit
    simp only [Finset.mem_coe]
    rw [mem_sectionPrimes, hsq, Nat.sqrt_eq']
    exact ⟨hp, h5, by omega, by omega⟩
  · intro k hk k' hk' heq
    simp only [Finset.mem_coe, squareHits, Finset.mem_filter] at hk hk'
    obtain ⟨p, -, -, hsq⟩ := hits_squareField_iff.mp hk.2
    obtain ⟨p', -, -, hsq'⟩ := hits_squareField_iff.mp hk'.2
    dsimp only at heq
    rw [hsq, hsq', Nat.sqrt_eq', Nat.sqrt_eq'] at heq
    subst heq
    omega

open Classical in
/-- **F3, the cover.**  If fewer primes have squares in the stretch than the stretch has
columns, some column of `[a, a + l)` is not hit by the square field. -/
theorem squares_cannot_cover {a l : ℕ} (h : (sectionPrimes a l).card < l) :
    ∃ k, a ≤ k ∧ k < a + l ∧ ¬ Hits squareField k := by
  by_contra hc
  push Not at hc
  have hfull : squareHits a l = Finset.Ico a (a + l) := by
    unfold squareHits
    apply Finset.filter_true_of_mem
    intro k hk
    rw [Finset.mem_Ico] at hk
    exact hc k hk.1 hk.2
  have := squares_in_section_le a l
  rw [hfull, Nat.card_Ico] at this
  omega

/-- **The construction's section.**  With `a = W p` and `l = W P - W p` (`p ≤ P` primes `≥ 5`),
the primes whose squares land in the section are exactly the primes of `[p, P)`. -/
theorem mem_sectionPrimes_W {p P r : ℕ} (hp : p.Prime) (h5 : 5 ≤ p) (hP : P.Prime) (hP5 : 5 ≤ P)
    (hpP : p ≤ P) : r ∈ sectionPrimes (W p) (W P - W p) ↔ r.Prime ∧ p ≤ r ∧ r < P := by
  have h1 := (square_column_W hp h5).1
  have h2 := (square_column_W hP hP5).1
  have hW : W p ≤ W P := by
    have := Nat.mul_self_le_mul_self hpP
    rw [← sq, ← sq] at this
    omega
  rw [mem_sectionPrimes]
  have e : 6 * (W p + (W P - W p)) + 1 = P ^ 2 := by omega
  rw [e, ← h1]
  constructor
  · rintro ⟨hr, -, hlo, hhi⟩
    refine ⟨hr, ?_, ?_⟩
    · by_contra hlt
      push Not at hlt
      have := Nat.mul_self_lt_mul_self hlt
      rw [← sq, ← sq] at this
      omega
    · by_contra hle
      push Not at hle
      have := Nat.mul_self_le_mul_self hle
      rw [← sq, ← sq] at this
      omega
  · rintro ⟨hr, hlo, hhi⟩
    have hlo' := Nat.mul_self_le_mul_self hlo
    have hhi' := Nat.mul_self_lt_mul_self hhi
    rw [← sq, ← sq] at hlo' hhi'
    exact ⟨hr, by omega, hlo', hhi'⟩

/-- The primes with squares in `[W p, W P)` number at most `P - p`. -/
theorem card_sectionPrimes_W_le {p P : ℕ} (hp : p.Prime) (h5 : 5 ≤ p) (hP : P.Prime) (hP5 : 5 ≤ P)
    (hpP : p ≤ P) : (sectionPrimes (W p) (W P - W p)).card ≤ P - p := by
  calc (sectionPrimes (W p) (W P - W p)).card
      ≤ (Finset.Ico p P).card := by
        apply Finset.card_le_card
        intro r hr
        exact Finset.mem_Ico.mpr ((mem_sectionPrimes_W hp h5 hP hP5 hpP).mp hr).2
    _ = P - p := Nat.card_Ico p P

/-- For CONSECUTIVE primes (no prime strictly between `p` and `P`) the only square in the
section is `p^2` itself: at most one prime. -/
theorem card_sectionPrimes_W_le_one {p P : ℕ} (hp : p.Prime) (h5 : 5 ≤ p) (hP : P.Prime)
    (hP5 : 5 ≤ P) (hpP : p ≤ P) (hgap : ∀ r, r.Prime → p < r → P ≤ r) :
    (sectionPrimes (W p) (W P - W p)).card ≤ 1 := by
  rw [Finset.card_le_one]
  intro r hr s hs
  obtain ⟨hr, hpr, hrP⟩ := (mem_sectionPrimes_W hp h5 hP hP5 hpP).mp hr
  obtain ⟨hs, hps, hsP⟩ := (mem_sectionPrimes_W hp h5 hP hP5 hpP).mp hs
  have hr' : r = p := by
    by_contra h
    have := hgap r hr (by omega)
    omega
  have hs' : s = p := by
    by_contra h
    have := hgap s hs (by omega)
    omega
  omega

/-- Two primes `≥ 5` with `p < P` are both odd, so `P ≥ p + 2`, and the section
`[W p, W P)` has at least `4` columns. -/
theorem four_le_W_sub_W {p P : ℕ} (hp : p.Prime) (h5 : 5 ≤ p) (hP : P.Prime) (hP5 : 5 ≤ P)
    (hpP : p < P) : 4 ≤ W P - W p := by
  have h1 := (square_column_W hp h5).1
  have h2 := (square_column_W hP hP5).1
  have hp2 := (coprime_six_of_prime hp h5).1
  have hP2 := (coprime_six_of_prime hP hP5).1
  have hle : p + 2 ≤ P := by omega
  have := Nat.mul_self_le_mul_self hle
  have e : (p + 2) * (p + 2) = p ^ 2 + 4 * p + 4 := by ring
  rw [e, ← sq] at this
  omega

/-- **F3 on the construction's section, the general bound.**  If `P - p < W P - W p` then some
column of `[W p, W P)` is not hit by the square field. -/
theorem squares_cannot_cover_W_of_lt {p P : ℕ} (hp : p.Prime) (h5 : 5 ≤ p) (hP : P.Prime)
    (hP5 : 5 ≤ P) (hpP : p ≤ P) (hl : P - p < W P - W p) :
    ∃ k, W p ≤ k ∧ k < W p + (W P - W p) ∧ ¬ Hits squareField k :=
  squares_cannot_cover (lt_of_le_of_lt (card_sectionPrimes_W_le hp h5 hP hP5 hpP) hl)

/-- **F3 on the construction's section, consecutive primes.**  For consecutive primes
`p < P` (both `≥ 5`) the section `[W p, W P)` ALWAYS has a column not hit by the square
field: only `p^2` lands in it, and the section has at least 4 columns. -/
theorem squares_cannot_cover_W {p P : ℕ} (hp : p.Prime) (h5 : 5 ≤ p) (hP : P.Prime)
    (hP5 : 5 ≤ P) (hpP : p < P) (hgap : ∀ r, r.Prime → p < r → P ≤ r) :
    ∃ k, W p ≤ k ∧ k < W p + (W P - W p) ∧ ¬ Hits squareField k := by
  have h1 := card_sectionPrimes_W_le_one hp h5 hP hP5 hpP.le hgap
  have h4 := four_le_W_sub_W hp h5 hP hP5 hpP
  exact squares_cannot_cover (by omega)

/-! ## F4: the class rule (E5) -/

/-- **F4, two factors.**  For primes `p, q ≥ 5`: `pq ≡ 1 (mod 6)` iff `p` and `q` are in the
same class, `pq ≡ 5 (mod 6)` iff in different classes. -/
theorem class_rule_two {p q : ℕ} (hp : p.Prime) (hp5 : 5 ≤ p) (hq : q.Prime) (hq5 : 5 ≤ q) :
    ((p * q) % 6 = 1 ↔ p % 6 = q % 6) ∧ ((p * q) % 6 = 5 ↔ p % 6 ≠ q % 6) := by
  have h1 := inS_of_prime hp hp5
  have h2 := inS_of_prime hq hq5
  unfold InS at h1 h2
  rw [Nat.mul_mod]
  rcases h1 with h1 | h1 <;> rcases h2 with h2 | h2 <;> rw [h1, h2] <;> norm_num

/-- The number of class `-1` factors (`≡ 5 (mod 6)`) in a list. -/
def countMinus (L : List ℕ) : ℕ := L.countP (fun r => r % 6 = 5)

/-- The class of a product of survivors is `(-1)^{count of class -1 factors}`.  Primality is
not used: only that each factor is `≡ ±1 (mod 6)`. -/
theorem prod_mod_six {L : List ℕ} (hL : ∀ r ∈ L, r % 6 = 1 ∨ r % 6 = 5) :
    (L.prod % 6 = 1 ∧ countMinus L % 2 = 0) ∨ (L.prod % 6 = 5 ∧ countMinus L % 2 = 1) := by
  induction L with
  | nil => left; simp [countMinus]
  | cons r L ih =>
    have hr := hL r (List.mem_cons_self ..)
    have ih := ih (fun s hs => hL s (List.mem_cons_of_mem r hs))
    rw [List.prod_cons, Nat.mul_mod]
    unfold countMinus at ih ⊢
    rw [List.countP_cons]
    rcases hr with hr | hr <;> rcases ih with ⟨h1, h2⟩ | ⟨h1, h2⟩ <;> simp [hr, h1] <;> omega

/-- **F4, the general class rule.**  A product of primes `≥ 5` is `≡ 1 (mod 6)` iff the number
of factors `≡ 5 (mod 6)` is even, and `≡ 5 (mod 6)` iff odd. -/
theorem class_rule {L : List ℕ} (hL : ∀ r ∈ L, r.Prime ∧ 5 ≤ r) :
    (L.prod % 6 = 1 ↔ countMinus L % 2 = 0) ∧ (L.prod % 6 = 5 ↔ countMinus L % 2 = 1) := by
  have := prod_mod_six (fun r hr => inS_of_prime (hL r hr).1 (hL r hr).2)
  omega

/-- **E5, per member.**  A survivor is a right member iff its count of class `-1` prime
factors is even, a left member iff odd (with multiplicity). -/
theorem class_rule_field {n : ℕ} (hn : InS n) :
    (n % 6 = 1 ↔ countMinus n.primeFactorsList % 2 = 0) ∧
      (n % 6 = 5 ↔ countMinus n.primeFactorsList % 2 = 1) := by
  have h := class_rule (L := n.primeFactorsList) (fun r hr =>
    ⟨Nat.prime_of_mem_primeFactorsList hr, five_le_of_mem_primeFactorsList hn hr⟩)
  rwa [Nat.prod_primeFactorsList (ne_zero_of_inS hn)] at h

/-! ## F5: the mirror law of a dilate (E6) -/

/-- **F5.**  The members of column `m`, dilated by `g`, sit at distance `g` on either side of
`6gm`: `g(6m - 1) + g = 6gm` and `g(6m + 1) = 6gm + g`.  NOTHING is assumed of `g`. -/
theorem mirror_law {g m : ℕ} (hm : 1 ≤ m) :
    g * (6 * m - 1) + g = 6 * g * m ∧ g * (6 * m + 1) = 6 * g * m + g := by
  obtain ⟨m', rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
  have e : 6 * (m' + 1) - 1 = 6 * m' + 5 := by omega
  rw [e]
  constructor <;> ring

/-- F5 in subtraction form: `6gm - g(6m - 1) = g = g(6m + 1) - 6gm`. -/
theorem mirror_law_sub {g m : ℕ} (hm : 1 ≤ m) :
    6 * g * m - g * (6 * m - 1) = g ∧ g * (6 * m + 1) - 6 * g * m = g := by
  obtain ⟨e1, e2⟩ := mirror_law (g := g) hm
  omega

/-- **F5 in columns.**  If `g(6m - 1)` is a member of column `k₁` and `g(6m + 1)` a member of
column `k₂` (either side each), then `k₁ + k₂ = 2gm`: the two dilates are mirror images about
column `gm`. -/
theorem mirror_law_columns {g m k₁ k₂ : ℕ} (hm : 1 ≤ m)
    (h₁ : g * (6 * m - 1) = 6 * k₁ - 1 ∨ g * (6 * m - 1) = 6 * k₁ + 1)
    (h₂ : g * (6 * m + 1) = 6 * k₂ - 1 ∨ g * (6 * m + 1) = 6 * k₂ + 1) :
    k₁ + k₂ = 2 * g * m := by
  obtain ⟨e1, e2⟩ := mirror_law (g := g) hm
  have e3 : 6 * g * m = 3 * (2 * g * m) := by ring
  generalize g * (6 * m - 1) = A at *
  generalize g * (6 * m + 1) = B at *
  generalize 6 * g * m = C at *
  generalize 2 * g * m = D at *
  omega

/-- **F5 with the sign, class `+1`.**  For `g = 6c + 1`: `g(6m - 1)` is the LEFT member of
column `gm - c` and `g(6m + 1)` the RIGHT member of column `gm + c`. -/
theorem mirror_law_class_plus {c m : ℕ} (hm : 1 ≤ m) :
    (6 * c + 1) * (6 * m - 1) = 6 * ((6 * c + 1) * m - c) - 1 ∧
      (6 * c + 1) * (6 * m + 1) = 6 * ((6 * c + 1) * m + c) + 1 := by
  obtain ⟨m', rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
  have e : 6 * (m' + 1) - 1 = 6 * m' + 5 := by omega
  rw [e]
  have k1 : (6 * c + 1) * (6 * m' + 5) + 6 * c + 1 = 6 * ((6 * c + 1) * (m' + 1)) := by ring
  have k2 : (6 * c + 1) * (6 * (m' + 1) + 1) = 6 * ((6 * c + 1) * (m' + 1)) + 6 * c + 1 := by ring
  have k3 : c ≤ (6 * c + 1) * (m' + 1) := by nlinarith
  generalize (6 * c + 1) * (m' + 1) = X at *
  generalize (6 * c + 1) * (6 * m' + 5) = A at *
  generalize (6 * c + 1) * (6 * (m' + 1) + 1) = B at *
  omega

/-- **F5 with the sign, class `-1`.**  For `g = 6c + 5 = 6(c + 1) - 1`: `g(6m - 1)` is the
RIGHT member of column `gm - (c + 1)` and `g(6m + 1)` the LEFT member of column `gm + (c + 1)`. -/
theorem mirror_law_class_minus {c m : ℕ} (hm : 1 ≤ m) :
    (6 * c + 5) * (6 * m - 1) = 6 * ((6 * c + 5) * m - (c + 1)) + 1 ∧
      (6 * c + 5) * (6 * m + 1) = 6 * ((6 * c + 5) * m + (c + 1)) - 1 := by
  obtain ⟨m', rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
  have e : 6 * (m' + 1) - 1 = 6 * m' + 5 := by omega
  rw [e]
  have k1 : (6 * c + 5) * (6 * m' + 5) + 6 * c + 5 = 6 * ((6 * c + 5) * (m' + 1)) := by ring
  have k2 : (6 * c + 5) * (6 * (m' + 1) + 1) = 6 * ((6 * c + 5) * (m' + 1)) + 6 * c + 5 := by ring
  have k3 : c + 1 ≤ (6 * c + 5) * (m' + 1) := by nlinarith
  generalize (6 * c + 5) * (m' + 1) = X at *
  generalize (6 * c + 5) * (6 * m' + 5) = A at *
  generalize (6 * c + 5) * (6 * (m' + 1) + 1) = B at *
  omega

/-- The mirror `n ↦ 12gm - n` maps `g(6m + i)` to `g(6m - i)` for every radius `i ≤ 6m`. -/
theorem mirror_dilate {g m i : ℕ} (hi : i ≤ 6 * m) :
    12 * g * m - g * (6 * m + i) = g * (6 * m - i) := by
  obtain ⟨d, hd⟩ : ∃ d, 6 * m = i + d := ⟨6 * m - i, by omega⟩
  have e : 6 * m - i = d := by omega
  have e12 : 12 * g * m = 2 * g * (6 * m) := by ring
  rw [e, e12, hd]
  have : 2 * g * (i + d) = g * (i + d + i) + g * d := by ring
  omega

/-! ## F6: the nesting per field (E2, E3) -/

/-- Multiplying by a survivor preserves and reflects membership in `S`: `(±1) · S = S`. -/
theorem inS_mul_iff {p n : ℕ} (hp : InS p) : InS (p * n) ↔ InS n := by
  unfold InS at *
  rw [Nat.mul_mod]
  rcases hp with hp | hp <;> rw [hp] <;> omega

theorem length_primeFactorsList_mul {p n : ℕ} (hp : p.Prime) (hn : n ≠ 0) :
    (p * n).primeFactorsList.length = n.primeFactorsList.length + 1 := by
  rw [(Nat.perm_primeFactorsList_mul hp.ne_zero hn).length_eq, List.length_append,
    Nat.primeFactorsList_prime hp, List.length_singleton]
  omega

/-- **F6, the dilation.**  For a prime `p ≥ 5`: `n ∈ field j ↔ p n ∈ field (j + 1)`. -/
theorem field_dilate {p n j : ℕ} (hp : p.Prime) (h5 : 5 ≤ p) :
    n ∈ field j ↔ p * n ∈ field (j + 1) := by
  have hpS := inS_of_prime hp h5
  rw [mem_field, mem_field]
  constructor
  · rintro ⟨hS, hlen⟩
    refine ⟨(inS_mul_iff hpS).mpr hS, ?_⟩
    rw [length_primeFactorsList_mul hp (ne_zero_of_inS hS), hlen]
  · rintro ⟨hS, hlen⟩
    have hnS := (inS_mul_iff hpS).mp hS
    refine ⟨hnS, ?_⟩
    rw [length_primeFactorsList_mul hp (ne_zero_of_inS hnS)] at hlen
    omega

/-- **F6, the union.**  Field `j + 1` is the union over the primes `p ≥ 5` of `p · field j`. -/
theorem field_succ_eq_union (j : ℕ) :
    field (j + 1) = ⋃ p ∈ {p : ℕ | p.Prime ∧ 5 ≤ p}, (fun m => p * m) '' field j := by
  ext n
  simp only [Set.mem_iUnion, Set.mem_image, Set.mem_setOf_eq, exists_prop]
  constructor
  · intro hn
    obtain ⟨hS, hlen⟩ := mem_field.mp hn
    have hne : n.primeFactorsList ≠ [] := by
      intro h; rw [h] at hlen; simp at hlen
    obtain ⟨p, hp⟩ := List.exists_mem_of_ne_nil _ hne
    have hpp := Nat.prime_of_mem_primeFactorsList hp
    have hp5 := five_le_of_mem_primeFactorsList hS hp
    obtain ⟨m, rfl⟩ := Nat.dvd_of_mem_primeFactorsList hp
    exact ⟨p, ⟨hpp, hp5⟩, m, (field_dilate hpp hp5).mpr hn, rfl⟩
  · rintro ⟨p, ⟨hpp, hp5⟩, m, hm, rfl⟩
    exact (field_dilate hpp hp5).mp hm

/-- **E3, every field is the primes dilated by 5.**  The multiples of 5 in field `j + 1` are
exactly `5 · field j`. -/
theorem field_inter_five {n j : ℕ} :
    (n ∈ field (j + 1) ∧ 5 ∣ n) ↔ ∃ m, n = 5 * m ∧ m ∈ field j := by
  have h5 : Nat.Prime 5 := by norm_num
  constructor
  · rintro ⟨hn, ⟨m, rfl⟩⟩
    exact ⟨m, rfl, (field_dilate h5 le_rfl).mpr hn⟩
  · rintro ⟨m, rfl, hm⟩
    exact ⟨(field_dilate h5 le_rfl).mp hm, dvd_mul_right 5 m⟩

/-! ## F7: the overlay is the composites (E1) -/

/-- Every survivor lies in the field of its factor count. -/
theorem mem_field_length {n : ℕ} (hn : InS n) : n ∈ field n.primeFactorsList.length := ⟨hn, rfl⟩

/-- The fields are disjoint: a number is in at most one field. -/
theorem field_unique {n i j : ℕ} (hi : n ∈ field i) (hj : n ∈ field j) : i = j :=
  hi.2.symm.trans hj.2

/-- **F7.**  A survivor `n > 1` lies in some field `j ≥ 2` iff it is composite. -/
theorem overlay_iff_composite {n : ℕ} (hn : InS n) (h1 : 1 < n) :
    (∃ j, 2 ≤ j ∧ n ∈ field j) ↔ ¬ n.Prime := by
  constructor
  · rintro ⟨j, hj, hnj⟩ hp
    have h5 : 5 ≤ n := by unfold InS at hn; omega
    have := field_unique hnj (field_one_iff_prime_ge_five.mpr ⟨hp, h5⟩)
    omega
  · intro hp
    refine ⟨n.primeFactorsList.length, ?_, mem_field_length hn⟩
    have hne : n.primeFactorsList ≠ [] := by
      rw [Ne, Nat.primeFactorsList_eq_nil]; omega
    have h1' : n.primeFactorsList.length ≠ 1 := by
      intro h
      exact hp (field_one_iff_prime_ge_five.mp ⟨hn, h⟩).1
    have hpos : 0 < n.primeFactorsList.length := by
      rcases hl : n.primeFactorsList with _ | ⟨a, l⟩
      · exact absurd hl hne
      · simp
    omega

theorem mem_overlay_iff {n : ℕ} (hn : InS n) (h1 : 1 < n) : n ∈ overlay ↔ ¬ n.Prime :=
  overlay_iff_composite hn h1

/-- **E1, the twin as blindness to every field `≥ 2`.**  Column `k ≥ 1` is a twin prime pair
iff the overlay hits neither member.  No engine, no size hypothesis. -/
theorem twin_iff_not_hits_overlay {k : ℕ} (hk : 1 ≤ k) :
    ((6 * k - 1).Prime ∧ (6 * k + 1).Prime) ↔ ¬ Hits overlay k := by
  have hS1 : InS (6 * k - 1) := by unfold InS; omega
  have hS2 : InS (6 * k + 1) := by unfold InS; omega
  unfold Hits
  rw [mem_overlay_iff hS1 (by omega), mem_overlay_iff hS2 (by omega)]
  tauto

/-- **E1 on the sight.**  For a member `n ∈ S` with `q < n < q'^2` (no prime in `(q, q')`), the
engine `{5..q}` strikes `n` iff `n` lies in the overlay: below `q'^2` the struck set IS the
union of the fields `≥ 2`. -/
theorem smallFactor_iff_mem_overlay {q q' n : ℕ} (hgap : ∀ p, p.Prime → p < q' → p ≤ q)
    (hn : InS n) (hqn : q < n) (hlt : n < q' ^ 2) (h1 : 1 < n) :
    SmallFactor q n ↔ n ∈ overlay := by
  rw [smallFactor_iff_not_prime hgap (inS_iff.mp hn).1 (inS_iff.mp hn).2 hqn hlt h1,
    mem_overlay_iff hn h1]

/-- **E1 in columns.**  Above the engine and below `q'^2`, `Blocked q k` iff the overlay hits `k`. -/
theorem blocked_iff_hits_overlay {q q' k : ℕ} (hgap : ∀ p, p.Prime → p < q' → p ≤ q)
    (hq : q < 6 * k - 1) (hlt : 6 * k + 1 < q' ^ 2) : Blocked q k ↔ Hits overlay k := by
  have hS1 : InS (6 * k - 1) := by unfold InS; omega
  have hS2 : InS (6 * k + 1) := by unfold InS; omega
  unfold Blocked Hits
  rw [smallFactor_iff_mem_overlay hgap hS1 hq (by omega) (by omega),
    smallFactor_iff_mem_overlay hgap hS2 (by omega) hlt (by omega)]

end Fields
