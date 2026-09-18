/-
OwnerArgument (round 102, 2026-09-19): the owner's argument as the spine of the proof, with its
one remaining lemma named, and the first law of the attack on that lemma.

The argument:
  1. the machine always generates twin gaps - open columns recur at every level;
  2. sometimes a gap is blocked, by ordinary gears striking in the ordinary way;
  3. no mechanic of the machine blocks the gaps permanently;
  4. therefore the machine generates twins without end.

Steps 1 and 2 are theorems already in the kernel.  Step 4 follows from step 3 by the square-root
rule, and this file makes that implication a theorem: `twins_unbounded_of_survival`.  Step 3, in
the exact form the implication needs, is the SURVIVAL LEMMA:

    for every gear p, some column of the stretch (p², q²) - q the next gear - escapes every gear
    up to p.

Everything the search has produced bears on that lemma and nothing else.  The attack on it works
in the machine's own terms: the base pattern's open runs, the top gears' strikes as products of
primes straddling p, and the location law.  `product_kill_square_law` is the first law of that
attack: a product of two primes straddling `p`, at distances `a` below and `b` above, lands on the
offset `o` above `p²` only if `(b - a)² - 4 o` is a perfect square modulo `p` - because
`(p - a)(p + b) = p² + (b - a) p - a b` and `(a + b)² = (b - a)² + 4 a b`.
-/
import MirrorWalkConditional
import Mathlib.NumberTheory.Bertrand

namespace MirrorWalk

open SquareColumn

/-- **The survival lemma** - step 3 of the argument in the form step 4 needs: at every gear `p`
with next gear `q`, some column strictly inside the stretch `(p², q²)` escapes every gear up to
`p`.  (The column of `q²` itself is excluded: its upper member is the square.) -/
def Survival : Prop :=
  ∀ p q : ℕ, p.Prime → 5 ≤ p → q.Prime → p < q → (∀ r : ℕ, r.Prime → p < r → q ≤ r) →
    ∃ m : ℕ, p ^ 2 < 6 * m - 1 ∧ 6 * m + 1 < q ^ 2 ∧
      ∀ r : ℕ, r.Prime → r ≤ p → ¬ (r ∣ 6 * m - 1) ∧ ¬ (r ∣ 6 * m + 1)

/-- **Step 4 from step 3.**  A surviving column of the stretch is a twin prime pair: its members
lie below `q²` and escape every prime below `q`, so by the square-root rule both are prime.  Hence
above every bound there is a twin prime pair. -/
theorem twins_unbounded_of_survival (hS : Survival) :
    ∀ N : ℕ, ∃ m : ℕ, N < 6 * m - 1 ∧ (6 * m - 1).Prime ∧ (6 * m + 1).Prime := by
  intro N
  classical
  obtain ⟨p, hpN, hp⟩ := Nat.exists_infinite_primes (N + 6)
  have hp5 : 5 ≤ p := by omega
  obtain ⟨q, hqp, hq⟩ := Nat.exists_infinite_primes (p + 1)
  have hex : ∃ r : ℕ, r.Prime ∧ p < r := ⟨q, hq, by omega⟩
  set q₀ := Nat.find hex with hq₀def
  have hq₀ : q₀.Prime ∧ p < q₀ := Nat.find_spec hex
  have hleast : ∀ r : ℕ, r.Prime → p < r → q₀ ≤ r := fun r hr hpr =>
    Nat.find_min' hex ⟨hr, hpr⟩
  obtain ⟨m, hlo, hhi, hopen⟩ := hS p q₀ hp hp5 hq₀.1 hq₀.2 hleast
  have hfull : ∀ r, r.Prime → 5 ≤ r → r < q₀ → r ∈ (Finset.range q₀).filter Nat.Prime := by
    intro r hr _ hrq
    exact Finset.mem_filter.mpr ⟨Finset.mem_range.mpr hrq, hr⟩
  have hrp : ∀ r, r ∈ (Finset.range q₀).filter Nat.Prime → r ≤ p := by
    intro r hr
    have hr' := Finset.mem_filter.mp hr
    have hrq : r < q₀ := Finset.mem_range.mp hr'.1
    by_contra h; push_neg at h
    have := hleast r hr'.2 h; omega
  have hns : ¬ StruckBy ((Finset.range q₀).filter Nat.Prime) m := by
    rintro ⟨r, hr, hd | hd⟩
    · exact (hopen r (Finset.mem_filter.mp hr).2 (hrp r hr)).1 hd
    · exact (hopen r (Finset.mem_filter.mp hr).2 (hrp r hr)).2 hd
  have hm1 : 1 ≤ m := by
    by_contra h; push_neg at h
    have : m = 0 := by omega
    rw [this] at hlo; simp at hlo
  obtain ⟨hp1, hp2⟩ := section_twin_of_unstruck hfull hm1 hhi hns
  have hp2sq : p ^ 2 ≥ p := by nlinarith
  exact ⟨m, by omega, hp1, hp2⟩

/-- **The first law of the attack on the survival lemma.**  A product of two primes straddling
`p` - at distance `a` below and `b` above - lands on the offset `o = (b - a) p - a b` above `p²`,
and `(a + b)² = (b - a)² + 4 a b`.  So modulo `p`, `(b - a)² - 4 o` is the square `(a + b)²`: a
straddling product can only reach offsets whose `(b - a)² - 4 o` is a quadratic residue. -/
theorem product_kill_square_law (p a b : ℤ) :
    (p - a) * (p + b) = p ^ 2 + ((b - a) * p - a * b) ∧
    (a + b) ^ 2 = (b - a) ^ 2 + 4 * (a * b) ∧
    (b - a) ^ 2 - 4 * ((b - a) * p - a * b) = (a + b) ^ 2 - 4 * (b - a) * p := by
  refine ⟨by ring, by ring, by ring⟩

/-- **Survival on the primorial family.**  The family `30 t ± 1` is the mirror `{2, 3, 5}` from
home; a gear `h ≥ 7` strikes it at `t ≡ ±30⁻¹` modulo `h`, positions fixed by `h` alone - ONE
fixed pattern on the `t`-line for all machines, of which the machine `p` sees the range
`[p²/30, q²/30]`.  If at every gear that fixed pattern leaves some `t` of the range open to the
gears up to `p`, the survival lemma holds (the column is `m = 5 t`). -/
theorem survival_of_family
    (hfam : ∀ p q : ℕ, p.Prime → 5 ≤ p → q.Prime → p < q → (∀ r : ℕ, r.Prime → p < r → q ≤ r) →
      ∃ t : ℕ, p ^ 2 < 30 * t - 1 ∧ 30 * t + 1 < q ^ 2 ∧
        ∀ r : ℕ, r.Prime → r ≤ p → ¬ (r ∣ 30 * t - 1) ∧ ¬ (r ∣ 30 * t + 1)) :
    Survival := by
  intro p q hp hp5 hq hpq hnext
  obtain ⟨t, hlo, hhi, hopen⟩ := hfam p q hp hp5 hq hpq hnext
  refine ⟨5 * t, ?_, ?_, ?_⟩
  · have : 6 * (5 * t) - 1 = 30 * t - 1 := by ring_nf
    omega
  · omega
  · intro r hr hrp
    have e1 : 6 * (5 * t) - 1 = 30 * t - 1 := by ring_nf
    have e2 : 6 * (5 * t) + 1 = 30 * t + 1 := by ring
    rw [e1, e2]
    exact hopen r hr hrp

/-! ### Round 104: the survival lemma in its exact weight

`Survival` asks every stretch to survive.  Step 4 uses only that stretches above every bound
survive, and that weaker form is EXACTLY the statement "twins unbounded" - both directions are
proved below.  So the owner's line 3, in the form line 4 needs, is the conjecture itself, not a
reduction of it; and the strong form `Survival` (a twin pair between every two consecutive prime
squares) is a stronger statement, of the same kind as Legendre's conjecture on primes between
consecutive squares. -/

/-- **The survival lemma, weak form**: above every bound some stretch survives. -/
def SurvivalInf : Prop :=
  ∀ N : ℕ, ∃ p q : ℕ, N < p ∧ p.Prime ∧ 5 ≤ p ∧ q.Prime ∧ p < q ∧
    (∀ r : ℕ, r.Prime → p < r → q ≤ r) ∧
    ∃ m : ℕ, p ^ 2 < 6 * m - 1 ∧ 6 * m + 1 < q ^ 2 ∧
      ∀ r : ℕ, r.Prime → r ≤ p → ¬ (r ∣ 6 * m - 1) ∧ ¬ (r ∣ 6 * m + 1)

/-- A surviving column of a stretch is a twin prime pair (the square-root rule). -/
theorem twin_of_surviving_stretch {p q m : ℕ} (hp : p.Prime) (hp5 : 5 ≤ p) (hq : q.Prime)
    (hpq : p < q) (hnext : ∀ r : ℕ, r.Prime → p < r → q ≤ r)
    (hlo : p ^ 2 < 6 * m - 1) (hhi : 6 * m + 1 < q ^ 2)
    (hopen : ∀ r : ℕ, r.Prime → r ≤ p → ¬ (r ∣ 6 * m - 1) ∧ ¬ (r ∣ 6 * m + 1)) :
    (6 * m - 1).Prime ∧ (6 * m + 1).Prime := by
  classical
  have hfull : ∀ r, r.Prime → 5 ≤ r → r < q → r ∈ (Finset.range q).filter Nat.Prime := by
    intro r hr _ hrq
    exact Finset.mem_filter.mpr ⟨Finset.mem_range.mpr hrq, hr⟩
  have hrp : ∀ r, r ∈ (Finset.range q).filter Nat.Prime → r ≤ p := by
    intro r hr
    have hr' := Finset.mem_filter.mp hr
    have hrq : r < q := Finset.mem_range.mp hr'.1
    by_contra h; push_neg at h
    have := hnext r hr'.2 h; omega
  have hns : ¬ StruckBy ((Finset.range q).filter Nat.Prime) m := by
    rintro ⟨r, hr, hd | hd⟩
    · exact (hopen r (Finset.mem_filter.mp hr).2 (hrp r hr)).1 hd
    · exact (hopen r (Finset.mem_filter.mp hr).2 (hrp r hr)).2 hd
  have hm1 : 1 ≤ m := by
    by_contra h; push_neg at h
    have : m = 0 := by omega
    rw [this] at hlo; simp at hlo
  exact section_twin_of_unstruck hfull hm1 hhi hns

theorem survivalInf_of_survival (hS : Survival) : SurvivalInf := by
  intro N
  obtain ⟨p, hpN, hp⟩ := Nat.exists_infinite_primes (N + 6)
  obtain ⟨q, hqp, hq⟩ := Nat.exists_infinite_primes (p + 1)
  have hex : ∃ r : ℕ, r.Prime ∧ p < r := ⟨q, hq, by omega⟩
  classical
  have hq₀ : (Nat.find hex).Prime ∧ p < Nat.find hex := Nat.find_spec hex
  have hleast : ∀ r : ℕ, r.Prime → p < r → Nat.find hex ≤ r := fun r hr hpr =>
    Nat.find_min' hex ⟨hr, hpr⟩
  exact ⟨p, Nat.find hex, by omega, hp, by omega, hq₀.1, hq₀.2, hleast,
    hS p _ hp (by omega) hq₀.1 hq₀.2 hleast⟩

/-- **Weak survival gives twins unbounded** - the direction step 4 uses. -/
theorem twins_unbounded_of_survivalInf (hS : SurvivalInf) :
    ∀ N : ℕ, ∃ m : ℕ, N < 6 * m - 1 ∧ (6 * m - 1).Prime ∧ (6 * m + 1).Prime := by
  intro N
  obtain ⟨p, q, hpN, hp, hp5, hq, hpq, hnext, m, hlo, hhi, hopen⟩ := hS N
  obtain ⟨h1, h2⟩ := twin_of_surviving_stretch hp hp5 hq hpq hnext hlo hhi hopen
  have hp2sq : p ^ 2 ≥ p := by nlinarith
  exact ⟨m, by omega, h1, h2⟩

/-- **Twins unbounded gives weak survival** - the converse.  A twin pair `(6m-1, 6m+1)` far
above `N` sits in the stretch of the largest prime `p` with `p² < 6m-1` (the next prime `q` has
`q² > 6m+1`, since `q²` is odd, composite, and at least `6m-1`); the pair escapes every gear up to
`p` because its members are primes above `p²`; and `p > N` by Bertrand's postulate. -/
theorem survivalInf_of_twins_unbounded
    (hT : ∀ N : ℕ, ∃ m : ℕ, N < 6 * m - 1 ∧ (6 * m - 1).Prime ∧ (6 * m + 1).Prime) :
    SurvivalInf := by
  intro N
  classical
  obtain ⟨m, hNm, hp1, hp2⟩ := hT (4 * (N + 6) ^ 2)
  have hm1 : 1 ≤ m := by
    by_contra h; push_neg at h
    have : m = 0 := by omega
    rw [this] at hNm; simp at hNm
  have hex : ∃ q : ℕ, q.Prime ∧ 6 * m + 1 < q ^ 2 := by
    obtain ⟨q, hq1, hq2⟩ := Nat.exists_infinite_primes (6 * m + 2)
    exact ⟨q, hq2, by nlinarith⟩
  have hq : (Nat.find hex).Prime ∧ 6 * m + 1 < (Nat.find hex) ^ 2 := Nat.find_spec hex
  have hqmin : ∀ r, r.Prime → 6 * m + 1 < r ^ 2 → Nat.find hex ≤ r :=
    fun r hr h => Nat.find_min' hex ⟨hr, h⟩
  set q := Nat.find hex with hqdef
  have hN36 : 36 ≤ (N + 6) ^ 2 := by
    have : 6 ≤ N + 6 := by omega
    calc 36 = 6 ^ 2 := by norm_num
      _ ≤ (N + 6) ^ 2 := Nat.pow_le_pow_left this 2
  have h5q : 5 < q := by
    by_contra h; push_neg at h
    have : q ^ 2 ≤ 25 := by nlinarith
    omega
  set p := Nat.findGreatest Nat.Prime (q - 1) with hpdef
  have hp : p.Prime :=
    Nat.findGreatest_spec (P := Nat.Prime) (show 5 ≤ q - 1 by omega) Nat.prime_five
  have hp5 : 5 ≤ p := Nat.le_findGreatest (show 5 ≤ q - 1 by omega) Nat.prime_five
  have hpq : p < q := by
    have := Nat.findGreatest_le (P := Nat.Prime) (n := q - 1); omega
  have hnext : ∀ r, r.Prime → p < r → q ≤ r := by
    intro r hr hpr
    by_contra h; push_neg at h
    exact Nat.findGreatest_is_greatest hpr (by omega) hr
  have hpsq : p ^ 2 ≤ 6 * m + 1 := by
    by_contra h; push_neg at h
    have := hqmin p hp h; omega
  have hodd : p % 2 = 1 := by
    rcases hp.eq_two_or_odd with h | h
    · omega
    · exact h
  have hsqodd : p ^ 2 % 2 = 1 := by rw [Nat.pow_mod, hodd]
  have hne1 : p ^ 2 ≠ 6 * m + 1 := by
    intro h
    have hd : p ∣ 6 * m + 1 := ⟨p, by rw [← h]; ring⟩
    rcases (Nat.dvd_prime hp2).mp hd with h1 | h1
    · omega
    · nlinarith
  have hne2 : p ^ 2 ≠ 6 * m - 1 := by
    intro h
    have hd : p ∣ 6 * m - 1 := ⟨p, by rw [← h]; ring⟩
    rcases (Nat.dvd_prime hp1).mp hd with h1 | h1
    · omega
    · nlinarith
  have hlo : p ^ 2 < 6 * m - 1 := by omega
  have hpN : N < p := by
    by_contra h; push_neg at h
    obtain ⟨r, hr, hpr, hr2⟩ := Nat.exists_prime_lt_and_le_two_mul p (by omega)
    have hqr := hnext r hr hpr
    have hq2N : q ≤ 2 * N := by omega
    have : q ^ 2 ≤ (2 * N) ^ 2 := Nat.pow_le_pow_left hq2N 2
    have e1 : (2 * N) ^ 2 = 4 * N ^ 2 := by ring
    have e2 : N ^ 2 ≤ (N + 6) ^ 2 := Nat.pow_le_pow_left (by omega) 2
    omega
  refine ⟨p, q, hpN, hp, hp5, hq.1, hpq, hnext, m, hlo, hq.2, ?_⟩
  intro r hr hrp
  have hp2sq : p ≤ p ^ 2 := by nlinarith
  constructor
  · intro hd
    rcases (Nat.dvd_prime hp1).mp hd with h1 | h1
    · exact hr.one_lt.ne' h1
    · omega
  · intro hd
    rcases (Nat.dvd_prime hp2).mp hd with h1 | h1
    · exact hr.one_lt.ne' h1
    · omega

/-- **Line 3, weighed exactly.**  The weak survival lemma IS the statement that twin primes are
unbounded: neither side is a reduction of the other. -/
theorem survivalInf_iff_twins_unbounded :
    SurvivalInf ↔ ∀ N : ℕ, ∃ m : ℕ, N < 6 * m - 1 ∧ (6 * m - 1).Prime ∧ (6 * m + 1).Prime :=
  ⟨twins_unbounded_of_survivalInf, survivalInf_of_twins_unbounded⟩

end MirrorWalk
