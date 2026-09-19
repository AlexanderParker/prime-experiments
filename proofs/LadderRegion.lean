/-
LadderRegion (2026-09-20): the ladder hypothesis region by region, and the window statement.

The REGION HYPOTHESIS is the machine's window statement read between consecutive prime squares:
for consecutive primes `p < q` with `p ≥ 5` there is a twin centre `u` strictly inside
`(p², q²)`, i.e. `p² < u - 1` and `u + 1 < q²`.  A twin centre `s` makes `s - 1, s + 1` consecutive
primes, so the region hypothesis implies the ladder hypothesis and hence twins unbounded.

The WINDOW HYPOTHESIS is the owner's statement with the window `[q, (q+1)²)`: for every `q ≥ 6`
there is a twin centre `u` with `q ≤ u - 1` and `u + 1 < (q+1)²`.  It follows from the ladder
hypothesis by taking the largest twin centre `s ≤ q` and its rung.
-/
import TwinLadderTheorem

namespace TwinLadder

/-- `p < q` are consecutive primes: both prime, nothing prime strictly between. -/
def Consecutive (p q : ℕ) : Prop :=
  p.Prime ∧ q.Prime ∧ p < q ∧ ∀ r, p < r → r < q → ¬ r.Prime

/-- **The region hypothesis**: between the squares of any two consecutive primes `p < q` with
`p ≥ 5` lies a twin centre `u`, strictly: `p² < u - 1` and `u + 1 < q²`. -/
def RegionHyp : Prop :=
  ∀ p q, 5 ≤ p → Consecutive p q → ∃ u, TwinCentre u ∧ p ^ 2 < u - 1 ∧ u + 1 < q ^ 2

/-- A twin centre's two primes are consecutive: the only integer strictly between them is `s`
itself, a multiple of 6 at least 6, hence not prime. -/
theorem consecutive_of_twinCentre {s : ℕ} (h : TwinCentre s) : Consecutive (s - 1) (s + 1) := by
  have h6 := twinCentre_ge_six h
  obtain ⟨hd, hp1, hp2⟩ := h
  refine ⟨hp1, hp2, by omega, ?_⟩
  intro r h1 h2 hr
  have hrs : r = s := by omega
  subst hrs
  have h2r : 2 ∣ r := dvd_trans (by norm_num) hd
  have := hr.eq_one_or_self_of_dvd 2 h2r
  omega

/-- **Region implies ladder**: apply the region hypothesis to the consecutive pair
`(s - 1, s + 1)` of a twin centre `s` (here `s - 1 ≥ 5`). -/
theorem ladderHyp_of_regionHyp (h : RegionHyp) : LadderHyp := by
  intro s hs
  have h6 := twinCentre_ge_six hs
  obtain ⟨u, hu, h1, h2⟩ := h (s - 1) (s + 1) (by omega) (consecutive_of_twinCentre hs)
  exact ⟨u, hu, h1, h2⟩

/-- **Twins unbounded from the region hypothesis.** -/
theorem twins_unbounded_of_region (h : RegionHyp) :
    ∀ N : ℕ, ∃ m : ℕ, N < 6 * m - 1 ∧ (6 * m - 1).Prime ∧ (6 * m + 1).Prime :=
  twins_unbounded_of_ladder (ladderHyp_of_regionHyp h)

/-- **The window hypothesis** (owner's window statement): for every `q ≥ 6` there is a twin centre
`u` in the window `[q, (q+1)²)`, i.e. `q ≤ u - 1` and `u + 1 < (q+1)²`.  (The lower bound is
`q ≤ u - 1`, not `q < u - 1`: with `q ≡ 5 (mod 6)` the next twin centre can sit at `u = q + 1`.) -/
def WindowHyp : Prop :=
  ∀ q, 6 ≤ q → ∃ u, TwinCentre u ∧ q ≤ u - 1 ∧ u + 1 < (q + 1) ^ 2

/-- **Ladder implies window**: let `s` be the largest twin centre `≤ q` (there is one, `6 ≤ q`);
its rung `s'` is a twin centre above `q` (else `s` was not the largest) with
`s' + 1 < (s+1)² ≤ (q+1)²`. -/
theorem windowHyp_of_ladderHyp (hL : LadderHyp) : WindowHyp := by
  classical
  intro q hq
  set s := Nat.findGreatest TwinCentre q with hs_def
  have hs : TwinCentre s := Nat.findGreatest_spec hq twinCentre_six
  have hsq : s ≤ q := Nat.findGreatest_le q
  obtain ⟨s', hr⟩ := hL s hs
  have hgt := rung_gt hs hr
  have hs'q : q < s' := by
    by_contra hle
    exact Nat.findGreatest_is_greatest (by omega : s < s') (Nat.le_of_not_lt hle) hr.1
  refine ⟨s', hr.1, by omega, ?_⟩
  have h1 : s' + 1 < (s + 1) ^ 2 := hr.2.2
  have h2 : (s + 1) ^ 2 ≤ (q + 1) ^ 2 := Nat.pow_le_pow_left (by omega) 2
  omega

/-- **Region implies window.** -/
theorem windowHyp_of_regionHyp (h : RegionHyp) : WindowHyp :=
  windowHyp_of_ladderHyp (ladderHyp_of_regionHyp h)

end TwinLadder
