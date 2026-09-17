/-
MirrorWalkChain (round 63, 2026-09-17): one landing serves a whole band of machines, and a chain
of landings serves them all.

A twin pair `(t - 1, t + 1)` sits inside the window `(q, q²]` of EVERY machine `q` with

    √t ≤ q < t - 1,

a band running from the landing's square root up to the landing itself.  So the construction
does not need one flip per machine: it needs a sequence of landings whose bands overlap, and the
overlap condition is just `t (n+1) + 1 < (t n - 1)²` - each landing below the square of the one
before.

`chain_covers` is that statement, proved.  `window_statement_of_chain` is the window statement
for every machine from the first landing onward, given such a chain.

Why this is the right shape for the one-flip locator.  The mirror `S` (the gears `2, 3, …B`)
has product `M = B#`, and the flip from home at period `k` lands on the pair `2 M k ± 1`, whose
column is `oneFlip (M / 6) k 1`; the gears of `S` never strike it
(`mirror_gear_never_strikes`).  MEASURED (research/stack/r8/oneflip_primorial_mirror.py and the
primorial chain, round 63): for every primorial mirror from `B = 5` to `B = 109`, the first
period whose landing is a twin is small - `k = 1, 1, 2, 3, 4, 12, 2, 8, 11, 2, 37, 12, 72, 14,
7, 130, 121, 32, 103, 10, 56, 62, 36, 40, 24, 63, 113` - and the twenty-six consecutive bands
all overlap, so those twenty-seven landings alone cover every machine from 8 to 6.3 × 10⁴⁶.

What the chain needs from each primorial is therefore weak next to the window statement itself:
a twin somewhere on the progression `2 B# k ± 1` with `k` up to about `B# / B`, instead of a
twin inside `(q, q²]` for every `q` separately.
-/
import Mathlib

namespace MirrorWalk

/-- A landing: the pair either side of `t`. -/
def TwinCenter (t : ℕ) : Prop := (t - 1).Prime ∧ (t + 1).Prime

/-- A sequence grows at least as fast as its index. -/
theorem le_of_strictMono {t : ℕ → ℕ} (hmono : ∀ n, t n < t (n + 1)) (n : ℕ) : n ≤ t n := by
  induction n with
  | zero => exact Nat.zero_le _
  | succ m ih => exact Nat.succ_le_of_lt (lt_of_le_of_lt ih (hmono m))

/-- **The chain covers every machine.**  If each landing is below the square of the one before,
then every `q` from the first landing onward has a landing inside its window `(q, q²]`. -/
theorem chain_covers {t : ℕ → ℕ} (hmono : ∀ n, t n < t (n + 1))
    (hchain : ∀ n, t (n + 1) + 1 < (t n - 1) ^ 2) {q : ℕ} (hq : t 0 - 1 ≤ q) :
    ∃ n, q < t n - 1 ∧ t n + 1 ≤ q ^ 2 := by
  classical
  -- the set of indices whose landing is above q is nonempty
  have hex : ∃ n, q < t n - 1 := by
    refine ⟨q + 2, ?_⟩
    have h1 : q + 2 ≤ t (q + 2) := le_of_strictMono hmono (q + 2)
    omega
  -- use the least such index
  set N := Nat.find hex with hN
  have hNspec : q < t N - 1 := Nat.find_spec hex
  have hNpos : N ≠ 0 := by
    intro h
    rw [h] at hNspec
    omega
  obtain ⟨m, hm⟩ : ∃ m, N = m + 1 := ⟨N - 1, by omega⟩
  have hprev : ¬ (q < t m - 1) := Nat.find_min hex (by omega)
  have hle : t m - 1 ≤ q := by omega
  have hsq : (t m - 1) ^ 2 ≤ q ^ 2 := Nat.pow_le_pow_left hle 2
  have hstep : t N + 1 < (t m - 1) ^ 2 := by rw [hm]; exact hchain m
  exact ⟨N, hNspec, by omega⟩

/-- **The window statement from a chain of landings.**  If every landing is a twin pair on a
column (a multiple of 6) and the chain condition holds, every machine from the first landing
onward has a twin pair inside its window. -/
theorem window_statement_of_chain {t : ℕ → ℕ}
    (hmono : ∀ n, t n < t (n + 1)) (hchain : ∀ n, t (n + 1) + 1 < (t n - 1) ^ 2)
    (hsix : ∀ n, 6 ∣ t n) (h6 : ∀ n, 6 ≤ t n) (htwin : ∀ n, TwinCenter (t n))
    {q : ℕ} (hq : t 0 - 1 ≤ q) :
    ∃ m : ℕ, 1 ≤ m ∧ q < 6 * m - 1 ∧ 6 * m + 1 ≤ q ^ 2 ∧
      (6 * m - 1).Prime ∧ (6 * m + 1).Prime := by
  obtain ⟨n, hlo, hhi⟩ := chain_covers hmono hchain hq
  obtain ⟨m, hm⟩ := hsix n
  refine ⟨m, ?_, ?_, ?_, ?_, ?_⟩
  · have := h6 n; omega
  · omega
  · omega
  · have := (htwin n).1; rwa [hm] at this
  · have := (htwin n).2; rwa [hm] at this

/-- **A finite chain covers a bounded range of machines.**  With the chain condition holding for
the first `N` steps, every machine from the first landing up to the last has a landing inside its
window.  This is what makes a certificate possible: a handful of landings settles every machine
below the last one. -/
theorem chain_covers_upto {t : ℕ → ℕ} {N : ℕ}
    (hchain : ∀ n < N, t (n + 1) + 1 < (t n - 1) ^ 2)
    {q : ℕ} (hlo : t 0 - 1 ≤ q) (hhi : q < t N - 1) :
    ∃ n, n ≤ N ∧ q < t n - 1 ∧ t n + 1 ≤ q ^ 2 := by
  classical
  have hex : ∃ n, q < t n - 1 := ⟨N, hhi⟩
  set L := Nat.find hex with hL
  have hLspec : q < t L - 1 := Nat.find_spec hex
  have hLN : L ≤ N := Nat.find_min' hex hhi
  have hLpos : L ≠ 0 := by
    intro h
    rw [h] at hLspec
    omega
  obtain ⟨m, hm⟩ : ∃ m, L = m + 1 := ⟨L - 1, by omega⟩
  have hprev : ¬ (q < t m - 1) := Nat.find_min hex (by omega)
  have hle : t m - 1 ≤ q := by omega
  have hsq : (t m - 1) ^ 2 ≤ q ^ 2 := Nat.pow_le_pow_left hle 2
  have hstep : t L + 1 < (t m - 1) ^ 2 := by
    rw [hm]; exact hchain m (by omega)
  exact ⟨L, hLN, hLspec, by omega⟩

/-- **The window statement over a bounded range, from a finite chain.**  Given `N + 1` landings,
each a twin pair on a column and each below the square of the one before, every machine from the
first landing to the last has a twin pair inside its window. -/
theorem window_statement_upto {t : ℕ → ℕ} {N : ℕ}
    (hchain : ∀ n < N, t (n + 1) + 1 < (t n - 1) ^ 2)
    (hsix : ∀ n, n ≤ N → 6 ∣ t n) (h6 : ∀ n, n ≤ N → 6 ≤ t n)
    (htwin : ∀ n, n ≤ N → TwinCenter (t n))
    {q : ℕ} (hlo : t 0 - 1 ≤ q) (hhi : q < t N - 1) :
    ∃ m : ℕ, 1 ≤ m ∧ q < 6 * m - 1 ∧ 6 * m + 1 ≤ q ^ 2 ∧
      (6 * m - 1).Prime ∧ (6 * m + 1).Prime := by
  obtain ⟨n, hnN, hlo', hhi'⟩ := chain_covers_upto hchain hlo hhi
  obtain ⟨m, hm⟩ := hsix n hnN
  refine ⟨m, ?_, ?_, ?_, ?_, ?_⟩
  · have := h6 n hnN; omega
  · omega
  · omega
  · have := (htwin n hnN).1; rwa [hm] at this
  · have := (htwin n hnN).2; rwa [hm] at this

end MirrorWalk
