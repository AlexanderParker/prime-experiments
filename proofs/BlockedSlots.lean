/-
Formalisation of the blocked-slot algorithm.

The algorithm being formalised is `get_next_prime_gap` in `rust2/src/main.rs`:
for a starting number `n`, each trial divisor `q` holds a cursor at the next gap
`g` with `q ∣ n + g`, the cursors are advanced lazily, and the first gap no
cursor lands on is returned.

This file defines that operation and proves it correct, then does the same for
the twin version, where each divisor carries two cursors instead of one. It ends
with the reduction of the twin prime conjecture to a statement about how early a
survivor of the twin pattern appears (see `docs/twin-prime-program.md`).

Nothing here assumes any bound on prime gaps. The only outside input is Euclid's
theorem, used once, to know the cursor search terminates.
-/

import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Data.Nat.Prime.Infinite
import Mathlib.Data.Nat.Sqrt
import Mathlib.Data.Nat.Factorial.Basic
import Mathlib.Data.Set.Finite.Basic
import Mathlib.Data.Set.Finite.Lattice
import Mathlib.Order.Preorder.Finite
import Mathlib.Order.Interval.Finset.Basic
import Mathlib.Order.Interval.Finset.Nat
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.Ring

namespace BlockedSlots

/-! ## The blocking relation -/

/-- `Blocked n y g` says the gap `g` is ruled out from `n` by the divisors up to
`y`: some prime `q ≤ y` satisfies `g ≡ -n (mod q)`, equivalently `q ∣ n + g`.
This is exactly what one cursor of the algorithm records. -/
def Blocked (n y g : ℕ) : Prop :=
  ∃ q, q.Prime ∧ q ≤ y ∧ q ∣ (n + g)

instance (n y g : ℕ) : Decidable (Blocked n y g) := by
  refine decidable_of_iff (∃ q ∈ Finset.range (y + 1), q.Prime ∧ q ∣ (n + g)) ?_
  simp only [Finset.mem_range, Nat.lt_succ_iff]
  constructor
  · rintro ⟨q, hq, hp, hd⟩; exact ⟨q, hp, hq, hd⟩
  · rintro ⟨q, hp, hq, hd⟩; exact ⟨q, hq, hp, hd⟩

/-- The cursor form: `q` blocks the gaps `r, r + q, r + 2q, ...` where `r` is the
first blocked gap, the unique residue below `q` with `q ∣ n + r`. This is the
bridge between the relation above and the running buckets in the Rust code, where
each divisor stores one such `r` and advances it by `q`. -/
theorem blocked_iff_cursor (n y g : ℕ) :
    Blocked n y g ↔
      ∃ q, q.Prime ∧ q ≤ y ∧ ∃ r k, r < q ∧ q ∣ (n + r) ∧ g = r + k * q := by
  constructor
  · rintro ⟨q, hp, hqy, hdvd⟩
    have hq0 : 0 < q := hp.pos
    have hmd : q * (g / q) + g % q = g := Nat.div_add_mod g q
    refine ⟨q, hp, hqy, g % q, g / q, Nat.mod_lt _ hq0, ?_, by linarith⟩
    have key : n + g = q * (g / q) + (n + g % q) := by linarith
    rw [key] at hdvd
    exact (Nat.dvd_add_right (Dvd.intro (g / q) rfl)).mp hdvd
  · rintro ⟨q, hp, hqy, r, k, hrq, hnr, hg⟩
    refine ⟨q, hp, hqy, ?_⟩
    have key : n + g = (n + r) + k * q := by rw [hg]; ring
    rw [key]
    exact Nat.dvd_add hnr (Nat.dvd_mul_left q k)

/-! ## Soundness and completeness of one unblocked slot -/

/-- An unblocked slot yields a prime, provided the divisor bound reaches the
candidate's square root. This is the only thing the algorithm needs to be right. -/
theorem prime_of_not_blocked {n y g : ℕ} (h1 : 1 < n + g)
    (hy : Nat.sqrt (n + g) ≤ y) (hfree : ¬ Blocked n y g) : (n + g).Prime := by
  set m := n + g with hm
  by_contra hnp
  have hpos : 0 < m := by omega
  have hmin : m.minFac ^ 2 ≤ m := Nat.minFac_sq_le_self hpos hnp
  have hle : m.minFac ≤ Nat.sqrt m := by
    have : m.minFac * m.minFac ≤ m := by simpa [pow_two] using hmin
    exact Nat.le_sqrt.mpr this
  exact hfree ⟨m.minFac, Nat.minFac_prime (by omega), le_trans hle hy, Nat.minFac_dvd m⟩

/-- A prime above the divisor bound is never blocked: no false negatives. -/
theorem not_blocked_of_prime {n y g : ℕ} (hp : (n + g).Prime) (hy : y < n + g) :
    ¬ Blocked n y g := by
  rintro ⟨q, hq, hqy, hdvd⟩
  have : q = n + g := ((Nat.Prime.eq_one_or_self_of_dvd hp q hdvd).resolve_left hq.ne_one)
  omega

/-! ## The next-gap operation -/

/-- The gaps the algorithm accepts: at least 1, and unblocked by every divisor up
to the square root of the candidate. The divisor bound tracks the candidate, which
is what the cursor loop does when the divisor list is extended as the tested gap
grows. -/
def GapOK (n g : ℕ) : Prop := 1 ≤ g ∧ ¬ Blocked n (Nat.sqrt (n + g)) g

instance (n g : ℕ) : Decidable (GapOK n g) := by
  unfold GapOK; infer_instance

/-- The search terminates. The only input is Euclid's theorem. -/
theorem exists_gapOK (n : ℕ) (hn : 1 ≤ n) : ∃ g, GapOK n g := by
  obtain ⟨p, hple, hp⟩ := Nat.exists_infinite_primes (n + 1)
  refine ⟨p - n, by omega, ?_⟩
  have hpn : n + (p - n) = p := by omega
  rw [hpn]
  have h2 : 2 ≤ p := hp.two_le
  have : Nat.sqrt p < p := Nat.sqrt_lt_self (by omega)
  exact not_blocked_of_prime (by rw [hpn]; exact hp) (by rw [hpn]; exact this)

/-- The operation itself: the gap from `n` to the next prime, computed from `n`'s
residues alone. -/
def nextGap (n : ℕ) (hn : 1 ≤ n) : ℕ := Nat.find (exists_gapOK n hn)

theorem nextGap_pos (n : ℕ) (hn : 1 ≤ n) : 1 ≤ nextGap n hn :=
  (Nat.find_spec (exists_gapOK n hn)).1

/-- The returned slot is prime. -/
theorem prime_add_nextGap (n : ℕ) (hn : 1 ≤ n) : (n + nextGap n hn).Prime := by
  have hspec := Nat.find_spec (exists_gapOK n hn)
  have h1 : 1 < n + nextGap n hn := by
    have := nextGap_pos n hn; omega
  exact prime_of_not_blocked h1 le_rfl hspec.2

/-- Nothing between `n` and the returned slot is prime, so the operation really
returns the next prime gap. -/
theorem no_prime_between (n : ℕ) (hn : 1 ≤ n) :
    ∀ m, n < m → m < n + nextGap n hn → ¬ m.Prime := by
  intro m hlt hub hprime
  have hg : m - n < nextGap n hn := by omega
  have hnot : ¬ GapOK n (m - n) := Nat.find_min (exists_gapOK n hn) hg
  have hmn : n + (m - n) = m := by omega
  have h1 : 1 ≤ m - n := by omega
  have : Blocked n (Nat.sqrt m) (m - n) := by
    by_contra hb
    exact hnot ⟨h1, by rw [hmn] at *; exact hb⟩
  obtain ⟨q, hq, hqy, hdvd⟩ := this
  rw [hmn] at hdvd
  have hqm : q = m := (hprime.eq_one_or_self_of_dvd q hdvd).resolve_left hq.ne_one
  have : Nat.sqrt m < m := Nat.sqrt_lt_self (by omega)
  omega

/-- Summary: `nextGap` is the next prime gap. -/
theorem nextGap_spec (n : ℕ) (hn : 1 ≤ n) :
    (n + nextGap n hn).Prime ∧ ∀ m, n < m → m < n + nextGap n hn → ¬ m.Prime :=
  ⟨prime_add_nextGap n hn, no_prime_between n hn⟩

/-! ## The twin version: two cursors per divisor -/

/-- `BlockedTwin n y g` rules out `g` as the start of a twin pair: some prime
`q ≤ y` divides `n + g` or `n + g + 2`. In the algorithm this is the same loop
with two buckets per divisor, started at `-n mod q` and `-(n+2) mod q`. -/
def BlockedTwin (n y g : ℕ) : Prop :=
  ∃ q, q.Prime ∧ q ≤ y ∧ (q ∣ (n + g) ∨ q ∣ (n + g + 2))

instance (n y g : ℕ) : Decidable (BlockedTwin n y g) := by
  refine decidable_of_iff
    (∃ q ∈ Finset.range (y + 1), q.Prime ∧ (q ∣ (n + g) ∨ q ∣ (n + g + 2))) ?_
  simp only [Finset.mem_range, Nat.lt_succ_iff]
  constructor
  · rintro ⟨q, hq, hp, hd⟩; exact ⟨q, hp, hq, hd⟩
  · rintro ⟨q, hp, hq, hd⟩; exact ⟨q, hq, hp, hd⟩

/-- Soundness of the twin search: an unblocked slot gives two primes. -/
theorem twin_of_not_blockedTwin {n y g : ℕ} (h1 : 1 < n + g)
    (hy : Nat.sqrt (n + g + 2) ≤ y) (hfree : ¬ BlockedTwin n y g) :
    (n + g).Prime ∧ (n + g + 2).Prime := by
  have hsqrt_le : Nat.sqrt (n + g) ≤ y :=
    le_trans (Nat.sqrt_le_sqrt (by omega)) hy
  constructor
  · refine prime_of_not_blocked h1 hsqrt_le ?_
    rintro ⟨q, hq, hqy, hdvd⟩
    exact hfree ⟨q, hq, hqy, Or.inl hdvd⟩
  · have h1' : 1 < n + g + 2 := by omega
    have : ¬ Blocked (n + 2) y g := by
      rintro ⟨q, hq, hqy, hdvd⟩
      refine hfree ⟨q, hq, hqy, Or.inr ?_⟩
      have : n + 2 + g = n + g + 2 := by omega
      rwa [this] at hdvd
    have heq : n + 2 + g = n + g + 2 := by omega
    have := prime_of_not_blocked (n := n + 2) (y := y) (g := g)
      (by omega) (by rw [heq]; exact hy) this
    rwa [heq] at this

/-- Completeness of the twin search: a twin pair above the divisor bound is never
blocked. -/
theorem not_blockedTwin_of_twin {n y g : ℕ} (hp : (n + g).Prime)
    (hp2 : (n + g + 2).Prime) (hy : y < n + g) : ¬ BlockedTwin n y g := by
  rintro ⟨q, hq, hqy, hdvd | hdvd⟩
  · have : q = n + g := (hp.eq_one_or_self_of_dvd q hdvd).resolve_left hq.ne_one
    omega
  · have : q = n + g + 2 := (hp2.eq_one_or_self_of_dvd q hdvd).resolve_left hq.ne_one
    omega

/-- Slots the twin search accepts. -/
def TwinGapOK (n g : ℕ) : Prop := 1 ≤ g ∧ ¬ BlockedTwin n (Nat.sqrt (n + g + 2)) g

instance (n g : ℕ) : Decidable (TwinGapOK n g) := by
  unfold TwinGapOK; infer_instance

/-- The twin gap operation, defined only where the search terminates. Unlike
`nextGap`, termination here is exactly the twin prime conjecture, so it is taken
as a hypothesis rather than proved. -/
def twinGap (n : ℕ) (h : ∃ g, TwinGapOK n g) : ℕ := Nat.find h

theorem twinGap_spec (n : ℕ) (h : ∃ g, TwinGapOK n g) (hn : 1 ≤ n) :
    (n + twinGap n h).Prime ∧ (n + twinGap n h + 2).Prime := by
  have hspec := Nat.find_spec h
  have hpos : 1 ≤ twinGap n h := hspec.1
  have h1 : 1 < n + twinGap n h := by
    simp only [twinGap] at hpos ⊢; omega
  exact twin_of_not_blockedTwin h1 le_rfl hspec.2

/-- For `m ≥ 3` the divisor bound of the twin search stays below the candidate. -/
theorem sqrt_add_two_lt {m : ℕ} (hm : 3 ≤ m) : Nat.sqrt (m + 2) < m := by
  have hlt : m + 2 < m * m := by nlinarith
  exact Nat.sqrt_lt.mpr hlt

/-- And nothing between `n` and the returned slot starts a twin pair. -/
theorem no_twin_between (n : ℕ) (h : ∃ g, TwinGapOK n g) :
    ∀ m, n < m → m < n + twinGap n h → ¬ (m.Prime ∧ (m + 2).Prime) := by
  rintro m hlt hub ⟨hp, hp2⟩
  have hm3 : 3 ≤ m := by
    have h2 : 2 ≤ m := hp.two_le
    rcases eq_or_lt_of_le h2 with heq | hlt2
    · exfalso; subst heq; exact absurd hp2 (by decide)
    · omega
  have hg : m - n < twinGap n h := by omega
  have hnot : ¬ TwinGapOK n (m - n) := Nat.find_min h hg
  have hmn : n + (m - n) = m := by omega
  refine hnot ⟨by omega, ?_⟩
  rw [hmn]
  exact not_blockedTwin_of_twin (n := n) (y := Nat.sqrt (m + 2)) (g := m - n)
    (by rw [hmn]; exact hp) (by rw [hmn]; exact hp2)
    (by rw [hmn]; exact sqrt_add_two_lt hm3)

/-! ## The reduction

The survivor description of the twin pattern, and the statement that a survivor
appearing inside the certified window forces infinitely many twin pairs.
-/

/-- `Survivor y m` says `m` starts a twin pair as far as the divisors up to `y`
can tell: no prime `q ≤ y` divides `m` or `m + 2`. -/
def Survivor (y m : ℕ) : Prop :=
  ∀ q, q.Prime → q ≤ y → ¬ (q ∣ m) ∧ ¬ (q ∣ m + 2)

/-- A survivor is a twin pair as soon as the divisor bound reaches the square root
of the upper member. No window hypothesis is needed in this direction. -/
theorem twin_of_survivor {y m : ℕ} (h2 : 1 < m) (hy : Nat.sqrt (m + 2) ≤ y)
    (hs : Survivor y m) : m.Prime ∧ (m + 2).Prime := by
  have hfree : ¬ BlockedTwin 0 y m := by
    rintro ⟨q, hq, hqy, hdvd | hdvd⟩
    · exact (hs q hq hqy).1 (by simpa using hdvd)
    · exact (hs q hq hqy).2 (by simpa using hdvd)
  have := twin_of_not_blockedTwin (n := 0) (y := y) (g := m) (by simpa using h2)
    (by simpa using hy) hfree
  simpa using this

/-- Inside the certified window a survivor is exactly a twin pair. -/
theorem survivor_iff_twin {y m : ℕ} (hym : y < m) (hwin : m + 2 ≤ y * y) (h2 : 1 < m) :
    Survivor y m ↔ (m.Prime ∧ (m + 2).Prime) := by
  have hy : Nat.sqrt (m + 2) ≤ y := by
    have h := Nat.sqrt_le_sqrt hwin
    rwa [Nat.sqrt_eq] at h
  constructor
  · exact twin_of_survivor h2 hy
  · rintro ⟨hp, hp2⟩
    intro q hq hqy
    constructor
    · intro hdvd
      have : q = m := (hp.eq_one_or_self_of_dvd q hdvd).resolve_left hq.ne_one
      omega
    · intro hdvd
      have : q = m + 2 := (hp2.eq_one_or_self_of_dvd q hdvd).resolve_left hq.ne_one
      omega

/-- **The reduction.** If for every bound there is a divisor set whose certified
window `(y, y * y]` contains a survivor, there are infinitely many twin primes. -/
theorem twins_infinite_of_survivor_in_window
    (H : ∀ N, ∃ y, N ≤ y ∧ ∃ m, y < m ∧ m + 2 ≤ y * y ∧ Survivor y m) :
    {p : ℕ | p.Prime ∧ (p + 2).Prime}.Infinite := by
  apply Set.infinite_of_forall_exists_gt
  intro a
  obtain ⟨y, hay, m, hym, hwin, hs⟩ := H (a + 2)
  have h2 : 1 < m := by omega
  refine ⟨m, ?_, by omega⟩
  exact (survivor_iff_twin hym hwin h2).mp hs

/-- The converse: an infinitude of twin primes hands back a survivor in the
certified window for arbitrarily large divisor bounds. Given a large twin start `p`,
take the divisor bound to be `sqrt (p + 2) + 1`, which is just past the square root,
so the window reaches `p + 2`. -/
theorem survivor_in_window_of_twins_infinite
    (H : {p : ℕ | p.Prime ∧ (p + 2).Prime}.Infinite) :
    ∀ N, ∃ y, N ≤ y ∧ ∃ m, y < m ∧ m + 2 ≤ y * y ∧ Survivor y m := by
  intro N
  have hun : ∀ a : ℕ, ∃ b ∈ {p : ℕ | p.Prime ∧ (p + 2).Prime}, a < b := by
    intro a
    by_contra hc
    push_neg at hc
    exact H (Set.Finite.subset (Set.finite_le_nat a) fun x hx => hc x hx)
  obtain ⟨p, hp, hgt⟩ := hun (N * N + 8)
  obtain ⟨hp1, hp2⟩ := hp
  set y := Nat.sqrt (p + 2) + 1 with hy
  -- the divisor bound is past the square root, so the window reaches the pair
  have hwin : p + 2 ≤ y * y := le_of_lt (Nat.lt_succ_sqrt (p + 2))
  -- and it stays below the candidate
  have hlow : y < p := by
    have h8 : 8 ≤ p := by omega
    have hlt : p + 2 < (p - 1) * (p - 1) := by
      have hp1' : p - 1 + 1 = p := by omega
      nlinarith [hp1']
    have hs : Nat.sqrt (p + 2) < p - 1 := Nat.sqrt_lt.mpr hlt
    omega
  have hN : N ≤ y := by
    have h2 : N * N ≤ p + 2 := by omega
    have h3 := Nat.sqrt_le_sqrt h2
    rw [Nat.sqrt_eq] at h3
    omega
  exact ⟨y, hN, p, hlow, hwin, (survivor_iff_twin hlow hwin (by omega)).mpr ⟨hp1, hp2⟩⟩

/-- So the reduction loses nothing: the window hypothesis is *equivalent* to the
twin prime conjecture, not merely sufficient for it. -/
theorem twins_infinite_iff_survivor_in_window :
    {p : ℕ | p.Prime ∧ (p + 2).Prime}.Infinite ↔
      ∀ N, ∃ y, N ≤ y ∧ ∃ m, y < m ∧ m + 2 ≤ y * y ∧ Survivor y m :=
  ⟨survivor_in_window_of_twins_infinite, twins_infinite_of_survivor_in_window⟩

/-- The gap form of the same statement: if every window of length `G` holds a
survivor and `G` fits inside the certified range, the window statement follows. -/
theorem survivor_in_window_of_gap_bound {y G : ℕ}
    (hG : ∀ a, ∃ m, a < m ∧ m ≤ a + G ∧ Survivor y m)
    (hfit : y + G + 2 ≤ y * y) :
    ∃ m, y < m ∧ m + 2 ≤ y * y ∧ Survivor y m := by
  obtain ⟨m, hlt, hle, hs⟩ := hG y
  exact ⟨m, hlt, by omega, hs⟩

/-! ## Centred form: the blocked residues are `±1` for every divisor

Running the same rule at the midpoint `c` of a candidate pair removes the base
from the description entirely: `c` is blocked by `q` exactly when `c ≡ ±1 (mod q)`,
equivalently when `q ∣ c^2 - 1`. So the twin pattern is one fixed nested family of
sets rather than a family of translates.
-/

/-- `CentreSurvivor y c` says no divisor up to `y` divides `c^2 - 1`, i.e. `c`
avoids the residues `+1` and `-1` modulo every prime `q ≤ y`. -/
def CentreSurvivor (y c : ℕ) : Prop :=
  ∀ q, q.Prime → q ≤ y → ¬ q ∣ (c - 1) * (c + 1)

/-- The centred description is the same condition as the survivor description at
the pair's lower member. -/
theorem centreSurvivor_iff_survivor {y c : ℕ} (hc : 1 ≤ c) :
    CentreSurvivor y c ↔ Survivor y (c - 1) := by
  have hcc : c - 1 + 2 = c + 1 := by omega
  constructor
  · intro h q hq hqy
    have hnd := h q hq hqy
    constructor
    · intro hd; exact hnd (hd.mul_right _)
    · intro hd; rw [hcc] at hd; exact hnd (hd.mul_left _)
  · intro h q hq hqy hd
    rcases (Nat.Prime.dvd_mul hq).mp hd with hd1 | hd2
    · exact (h q hq hqy).1 hd1
    · exact (h q hq hqy).2 (by rw [hcc]; exact hd2)

/-- Inside the certified window, the centred condition is twinhood: `(c-1, c+1)`
is a twin pair exactly when `c^2 - 1` has no prime factor at or below `y`. -/
theorem centreSurvivor_iff_twin {y c : ℕ} (hy : y < c - 1) (hwin : c + 1 ≤ y * y)
    (hc : 3 ≤ c) : CentreSurvivor y c ↔ ((c - 1).Prime ∧ (c + 1).Prime) := by
  have h1 : 1 ≤ c := by omega
  have hcc : c - 1 + 2 = c + 1 := by omega
  rw [centreSurvivor_iff_survivor h1]
  have := survivor_iff_twin (y := y) (m := c - 1) hy (by omega) (by omega)
  rw [this, hcc]

/-- Centred soundness with the divisor bound at its natural value: if `c^2 - 1`
has no prime factor at or below `sqrt (c+1)` then `(c-1, c+1)` is a twin pair. -/
theorem twin_of_centreSurvivor {c : ℕ} (hc : 5 ≤ c)
    (hs : CentreSurvivor (Nat.sqrt (c + 1)) c) : (c - 1).Prime ∧ (c + 1).Prime := by
  have h1 : 1 ≤ c := by omega
  have hsurv : Survivor (Nat.sqrt (c + 1)) (c - 1) :=
    (centreSurvivor_iff_survivor h1).mp hs
  have hcc : c - 1 + 2 = c + 1 := by omega
  have := twin_of_survivor (y := Nat.sqrt (c + 1)) (m := c - 1) (by omega)
    (by rw [hcc]) hsurv
  rw [hcc] at this
  exact this

/-- The covering form of "only finitely many twin primes": beyond some point every
midpoint would have to be caught by a divisor at or below its own square root.
This is the statement any proof has to contradict. -/
theorem covering_of_not_infinite
    (H : ¬ {p : ℕ | p.Prime ∧ (p + 2).Prime}.Infinite) :
    ∃ N, ∀ c, N ≤ c → ∃ q, q.Prime ∧ q ≤ Nat.sqrt (c + 1) ∧ q ∣ (c - 1) * (c + 1) := by
  rw [Set.not_infinite] at H
  obtain ⟨B, hB⟩ := H.bddAbove
  refine ⟨B + 6, fun c hc => ?_⟩
  by_contra hcon
  push_neg at hcon
  have hs : CentreSurvivor (Nat.sqrt (c + 1)) c := by
    intro q hq hqy hd
    exact absurd hd (hcon q hq hqy)
  obtain ⟨hp, hp2⟩ := twin_of_centreSurvivor (by omega) hs
  have hmem : (c - 1) ∈ {p : ℕ | p.Prime ∧ (p + 2).Prime} := by
    refine ⟨hp, ?_⟩
    have hcc : c - 1 + 2 = c + 1 := by omega
    rw [hcc]; exact hp2
  have := hB hmem
  omega

/-! ## Survivors always exist - just not where they are needed

The blocked pattern can never cover everything: any `c` divisible by every divisor
up to `y` is automatically a survivor, because a prime dividing both `c` and
`c^2 - 1` would divide 1. The smallest such `c` is the primorial of `y`, of size
`exp(y)`, whereas the certified window reaches only `y^2`. So an explicit survivor
is always available and always astronomically too far out - which is the whole
difficulty, stated as a proof rather than as a heuristic.
-/

theorem centreSurvivor_of_forall_dvd {y c : ℕ} (hc : 1 ≤ c)
    (h : ∀ q, q.Prime → q ≤ y → q ∣ c) : CentreSurvivor y c := by
  intro q hq hqy hd
  have hqc : q ∣ c := h q hq hqy
  have hone : q ∣ 1 := by
    rcases (Nat.Prime.dvd_mul hq).mp hd with hd1 | hd2
    · have : q ∣ c - (c - 1) := Nat.dvd_sub hqc hd1
      rwa [show c - (c - 1) = 1 by omega] at this
    · have : q ∣ (c + 1) - c := Nat.dvd_sub hd2 hqc
      rwa [show (c + 1) - c = 1 by omega] at this
  exact hq.one_lt.ne' (Nat.dvd_one.mp hone)

/-- An explicit survivor for every divisor bound: `y !` is divisible by every
divisor up to `y`, so it survives all of them. The point is its size - the witness
sits at `y !`, while the window the divisors can certify ends at `y * y`. -/
theorem centreSurvivor_factorial (y : ℕ) : CentreSurvivor y (Nat.factorial y) :=
  centreSurvivor_of_forall_dvd (Nat.one_le_iff_ne_zero.mpr (Nat.factorial_ne_zero y))
    fun q hq hqy => Nat.dvd_factorial hq.pos hqy

/-- So the twin pattern never runs out of survivors, for any divisor bound. -/
theorem exists_centreSurvivor (y : ℕ) : ∃ c, 1 ≤ c ∧ CentreSurvivor y c :=
  ⟨Nat.factorial y, Nat.one_le_iff_ne_zero.mpr (Nat.factorial_ne_zero y),
    centreSurvivor_factorial y⟩

/-! ## The divisor 3 forces the midpoint

At `q = 3` the survivor condition has only one residue left, because the two
blocked residues `+1` and `-1` leave exactly `q - 2 = 1` class. So the pattern is
not merely thin at `q = 3`, it is pinned: every survivor's midpoint is divisible
by 6. This is the extreme case of the `prod (q - 2)` count - the factor is 1.
-/

theorem six_dvd_succ_of_survivor {y m : ℕ} (hy : 3 ≤ y) (hm : 3 < m)
    (hs : Survivor y m) : 6 ∣ (m + 1) := by
  obtain ⟨h2, -⟩ := hs 2 Nat.prime_two (by omega)
  obtain ⟨h3, h3'⟩ := hs 3 Nat.prime_three (by omega)
  have e2 : m % 2 ≠ 0 := fun h => h2 (Nat.dvd_of_mod_eq_zero h)
  have e3 : m % 3 ≠ 0 := fun h => h3 (Nat.dvd_of_mod_eq_zero h)
  have e3' : (m + 2) % 3 ≠ 0 := fun h => h3' (Nat.dvd_of_mod_eq_zero h)
  have d2 : 2 ∣ (m + 1) := by refine Nat.dvd_of_mod_eq_zero ?_; omega
  have d3 : 3 ∣ (m + 1) := by
    have hcase : m % 3 = 0 ∨ m % 3 = 1 ∨ m % 3 = 2 := by omega
    rcases hcase with h | h | h
    · exact absurd h e3
    · exact absurd (by omega : (m + 2) % 3 = 0) e3'
    · refine Nat.dvd_of_mod_eq_zero ?_; omega
  show 2 * 3 ∣ (m + 1)
  exact Nat.Coprime.mul_dvd_of_dvd_of_dvd (by decide) d2 d3

/-! ## What counting can and cannot do

The reduction needs an upper bound on the largest gap. Counting gives the inequality
below, and section 14a of the program document records why it is vacuous: each odd prime
`q` blocks `O(L/q)` of `L` consecutive slots, and `sum 2/q` over the odd primes up to `y`
already exceeds 1 by `y = 11`, so the bound never constrains `L`.

Only the inequality itself is a theorem; "no counting argument can succeed" is a claim
about proof strategies, not something Lean can hold. What is formalised here is the exact
per-divisor count that any such argument would have to use.
-/

/-- One divisor blocks at most `L/q + 2` of the slots `0, ..., L-1`. The slack of 2
rather than 1 comes from the two boundary quotients and is harmless: what matters is
the `L/q` term. -/
theorem card_blocked_by_le (n q L : ℕ) (hq : 0 < q) :
    ((Finset.range L).filter (fun g => q ∣ (n + g))).card ≤ L / q + 2 := by
  classical
  -- `g` is determined by the quotient `(n + g) / q`, and the quotients live in a short
  -- interval, so the count is at most that interval's length
  have hmap : Set.InjOn (fun g => (n + g) / q)
      ((Finset.range L).filter (fun g => q ∣ (n + g))) := by
    intro a ha b hb hab
    simp only [Finset.coe_filter, Set.mem_setOf_eq, Finset.mem_range] at ha hb
    have hda : q * ((n + a) / q) = n + a := Nat.mul_div_cancel' ha.2
    have hdb : q * ((n + b) / q) = n + b := Nat.mul_div_cancel' hb.2
    simp only at hab
    have : n + a = n + b := by rw [← hda, ← hdb, hab]
    omega
  have hsub : ((Finset.range L).filter (fun g => q ∣ (n + g))).image
      (fun g => (n + g) / q) ⊆ Finset.Icc (n / q) ((n + L) / q) := by
    intro x hx
    simp only [Finset.mem_image, Finset.mem_filter, Finset.mem_range] at hx
    obtain ⟨g, ⟨hgL, -⟩, rfl⟩ := hx
    refine Finset.mem_Icc.mpr ⟨Nat.div_le_div_right (by omega), Nat.div_le_div_right (by omega)⟩
  have hcard := Finset.card_le_card hsub
  rw [Finset.card_image_of_injOn hmap] at hcard
  refine le_trans hcard ?_
  rw [Nat.card_Icc]
  -- (n + L) / q is at most n/q + L/q + 1: both remainders are below q, so their sum
  -- contributes at most one extra quotient step
  have hlt : (n + L) / q < n / q + L / q + 2 := by
    refine (Nat.div_lt_iff_lt_mul hq).mpr ?_
    have h1 : n < q * (n / q) + q := by
      have := Nat.div_add_mod n q
      have := Nat.mod_lt n hq
      omega
    have h2 : L < q * (L / q) + q := by
      have := Nat.div_add_mod L q
      have := Nat.mod_lt L hq
      omega
    calc n + L < (q * (n / q) + q) + (q * (L / q) + q) := by omega
      _ = (n / q + L / q + 2) * q := by ring
  -- card (Icc (n/q) ((n+L)/q)) = (n+L)/q + 1 - n/q, and n/q <= (n+L)/q
  have hmono : n / q ≤ (n + L) / q := Nat.div_le_div_right (by omega)
  omega

/-! ## Why constructed witnesses are always exponentially large

`centreSurvivor_factorial` produces a survivor at `y !`, and other constructions (for
instance `m = 1` modulo every prime up to `z`) land at the primorial. That is not a
weakness of those particular constructions: any congruence condition that *guarantees*
`q` misses a number must have modulus divisible by `q`, so guaranteeing survivorship
against every prime up to `z` forces a modulus divisible by their product. Witnesses built
this way are exponential in `z` by necessity, never of size `y^2`, so no construction of
this shape can settle the window question of section 4.
-/

-- NOT FORMALISED HERE. The statement is:
--
--   if `q` is prime and no member of `a, a + M, a + 2M, ...` is divisible by `q`,
--   then `q ∣ M`
--
-- Proof: if `q` does not divide `M` then `M` is invertible mod `q`, so `k` can be chosen
-- with `a + k M = 0 (mod q)`. Formalising it wants `ZMod q` as a field; an attempt using
-- raw `Nat.mod` arithmetic did not go through, and it is left out rather than left broken.
-- See section 14b of docs/twin-prime-program.md.

/-! ## A filter every candidate proof has to pass

The same rule with two cursors per divisor, run on `c * (c - 1)` instead of
`c^2 - 1`, has blocked residues `0` and `1`, which cover everything modulo 2. The
survivor count per period is `prod (q - r_q)` with `r_2 = 2`, hence zero, and the
conclusion is finite rather than infinite. Any argument that proves the twin case
must therefore break here; if it would go through verbatim for consecutive primes,
it is wrong.
-/

theorem consecutive_primes_subset_two :
    {c : ℕ | c.Prime ∧ (c + 1).Prime} ⊆ {2} := by
  intro c ⟨hc, hc1⟩
  rcases Nat.even_or_odd c with he | ho
  · have : c = 2 := (Nat.Prime.even_iff hc).mp he
    simpa using this
  · exfalso
    have he1 : Even (c + 1) := Odd.add_one ho
    have : c + 1 = 2 := (Nat.Prime.even_iff hc1).mp he1
    have hc0 : c = 1 := by omega
    exact hc.one_lt.ne' hc0

/-! ## Lockstep: one new divisor destroys at most one survivor -/

/-- Moving the divisor bound from `y` to the next prime `y'` can only destroy the
survivor whose own member is `y'`. Everything else in the certified window is
untouched, so the removal side of the accounting leaks nothing. -/
theorem survivor_step {y y' m : ℕ} (hyy : y < y') (hy' : y'.Prime)
    (hnext : ∀ q, q.Prime → y < q → q < y' → False)
    (hs : Survivor y m) (hwin : m + 2 ≤ y * y) (hm : 0 < m)
    (hfail : ¬ Survivor y' m) : m = y' ∨ m + 2 = y' := by
  obtain ⟨q, hq, hqy', hbad⟩ : ∃ q, q.Prime ∧ q ≤ y' ∧ (q ∣ m ∨ q ∣ m + 2) := by
    by_contra hc
    refine hfail ?_
    intro q hq hqy'
    constructor
    · intro hd; exact hc ⟨q, hq, hqy', Or.inl hd⟩
    · intro hd; exact hc ⟨q, hq, hqy', Or.inr hd⟩
  have hqy : y < q := by
    by_contra hle
    push_neg at hle
    rcases hbad with hd | hd
    · exact (hs q hq hle).1 hd
    · exact (hs q hq hle).2 hd
  have hqeq : q = y' := by
    rcases lt_or_eq_of_le hqy' with hlt | heq
    · exact absurd (hnext q hq hqy hlt) (by simp)
    · exact heq
  subst hqeq
  -- the other factor is below y, so it would already have been blocked
  have hy2 : 2 ≤ y := by
    rcases Nat.lt_or_ge y 2 with hsmall | hbig
    · interval_cases y <;> omega
    · exact hbig
  rcases hbad with hd | hd
  · obtain ⟨r, hr⟩ := hd
    rcases Nat.eq_zero_or_pos r with h0 | hpos
    · exfalso; subst h0; simp only [Nat.mul_zero] at hr; omega
    rcases Nat.lt_or_ge r 2 with h1 | h2
    · left
      have hr1 : r = 1 := by omega
      subst hr1; simpa using hr
    · exfalso
      have hrlt : r < y := by
        by_contra hcon
        push_neg at hcon
        have hstep1 : y * y ≤ y * r := by nlinarith
        have hstep2 : y * r < q * r := by nlinarith
        linarith
      have hrp : r.minFac.Prime := Nat.minFac_prime (by omega)
      have hrd : r.minFac ∣ m := hr ▸ Dvd.dvd.mul_left (Nat.minFac_dvd r) q
      have hrle : r.minFac ≤ y := le_trans (Nat.minFac_le (by omega)) (by omega)
      exact (hs r.minFac hrp hrle).1 hrd
  · obtain ⟨r, hr⟩ := hd
    rcases Nat.eq_zero_or_pos r with h0 | hpos
    · exfalso; subst h0; simp only [Nat.mul_zero] at hr; omega
    rcases Nat.lt_or_ge r 2 with h1 | h2
    · right
      have hr1 : r = 1 := by omega
      subst hr1; simpa using hr
    · exfalso
      have hrlt : r < y := by
        by_contra hcon
        push_neg at hcon
        have hstep1 : y * y ≤ y * r := by nlinarith
        have hstep2 : y * r < q * r := by nlinarith
        linarith
      have hrp : r.minFac.Prime := Nat.minFac_prime (by omega)
      have hrd : r.minFac ∣ m + 2 := hr ▸ Dvd.dvd.mul_left (Nat.minFac_dvd r) q
      have hrle : r.minFac ≤ y := le_trans (Nat.minFac_le (by omega)) (by omega)
      exact (hs r.minFac hrp hrle).2 hrd

end BlockedSlots
