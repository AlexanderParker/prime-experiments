/-
THE PERIODIC-ENUMERATION LEMMA (round 26) - one abstract lemma for two
standing gaps.

Round-25 verdict 20 and round-22 verdict 11 named the SAME missing step at
two machines:

  * verdict 11 (depth-sum identity, `DepthSum.lean`): the re-indexing bridge
    `Machine13.opSeq (n + 1485) = Machine13.opSeq n + 5005`;
  * verdict 20 (the survivor generator, `Gen11.lean`): the periodicity glue
    `Machine11.opSeq (n + 135) = Machine11.opSeq n + 385`.

Both are instances of one fact about ANY enumeration of ANY periodic
predicate, and this file proves it once, abstractly, with no machine, no
gears and no arithmetic beyond `omega`:

    THE NEXT-POINT OPERATOR COMMUTES WITH THE PERIOD  (`next_shift`)
    AN ENUMERATION THAT REALISES THE PERIOD SHIFT ONCE
    REALISES IT FOREVER                              (`op_shift`)

`next_shift` is where the mathematics is: `E` periodic makes `next k + P` an
`E`-point above `k + P`, and pulling `next (k + P)` back by `P` makes an
`E`-point above `k`, so the two minimality facts pin the two values to each
other.  `op_shift` is then a one-line induction, and the base case
`op N = op 0 + P` is a FINITE computation at each machine (`S_135 = 385` at
machine 11, `S_1485 = 5005` at machine 13) - which is exactly what makes the
glue kernel-checkable at all.

The `1 <= k` side condition on periodicity is not cosmetic: slot `0` carries
the pair `(0, 1)` rather than `(-1, 1)`, so `Exposed 0` is FALSE while
`Exposed P` is TRUE at every machine of this project.  Every use of `hper`
below is at a point that is provably positive.
-/

import Spectrum

namespace Periodic

/-! ## The abstract statements

`E` is the predicate being enumerated (an opening), `next` the "least
`E`-point strictly above" operator, `op` the enumeration, `P` the period and
`N` the number of `E`-points per period.  The three `next` hypotheses are
exactly what `Nat.find` gives at every machine in this ledger
(`nextOp_gt`, `nextOp_exposed`, `nextOp_min`).
-/

/-- **The next-point operator commutes with the period.**  If `E` is
periodic mod `P` (above `0`) and `next k` is the least `E`-point strictly
above `k`, then `next (k + P) = next k + P`. -/
theorem next_shift {E : ℕ → Prop} {next : ℕ → ℕ} {P : ℕ}
    (hgt : ∀ k, k < next k) (hE : ∀ k, E (next k))
    (hmin : ∀ k m, k < m → m < next k → ¬ E m)
    (hper : ∀ k, 1 ≤ k → (E (k + P) ↔ E k)) (k : ℕ) :
    next (k + P) = next k + P := by
  have hgtk := hgt k
  have hgtkP := hgt (k + P)
  have hEshift : E (next k + P) := (hper (next k) (by omega)).mpr (hE k)
  rcases Nat.lt_trichotomy (next (k + P)) (next k + P) with hlt | heq | hgt'
  · -- pull the shifted next-point back by `P`: an `E`-point between `k` and `next k`
    exfalso
    have hb : next (k + P) - P + P = next (k + P) := by omega
    have hEm : E (next (k + P) - P) :=
      (hper (next (k + P) - P) (by omega)).mp (by rw [hb]; exact hE (k + P))
    exact hmin k (next (k + P) - P) (by omega) (by omega) hEm
  · exact heq
  · exact absurd hEshift (hmin (k + P) (next k + P) (by omega) hgt')

/-- **The periodic-enumeration lemma.**  An enumeration built from `next`
that realises the period shift ONCE - `op N = op 0 + P`, a finite
computation - realises it at every index. -/
theorem op_shift {next op : ℕ → ℕ} {P N : ℕ}
    (hsucc : ∀ n, op (n + 1) = next (op n))
    (hnext : ∀ k, next (k + P) = next k + P)
    (h0 : op N = op 0 + P) (n : ℕ) : op (n + N) = op n + P := by
  induction n with
  | zero => rw [Nat.zero_add]; exact h0
  | succ n ih =>
    rw [show n + 1 + N = (n + N) + 1 by omega, hsucc, ih, hnext, ← hsucc]

/-- The shift iterated: `t` periods move the enumeration by `t * N` indices
and `t * P` slots. -/
theorem op_shift_mul {op : ℕ → ℕ} {P N : ℕ}
    (h : ∀ n, op (n + N) = op n + P) (t n : ℕ) :
    op (n + t * N) = op n + t * P := by
  induction t with
  | zero => simp
  | succ t ih =>
    rw [Nat.succ_mul, ← Nat.add_assoc, h, ih, Nat.succ_mul, Nat.add_assoc]

/-- Periodicity of the predicate, iterated. -/
theorem pred_shift_mul {E : ℕ → Prop} {P : ℕ}
    (hper : ∀ k, 1 ≤ k → (E (k + P) ↔ E k)) (t : ℕ) :
    ∀ k, 1 ≤ k → (E (k + t * P) ↔ E k) := by
  induction t with
  | zero => intro k _; simp
  | succ t ih =>
    intro k hk
    rw [Nat.succ_mul, ← Nat.add_assoc, hper (k + t * P) (by omega), ih k hk]

/-- The next-point operator's shift, iterated. -/
theorem next_shift_mul {next : ℕ → ℕ} {P : ℕ}
    (hnext : ∀ k, next (k + P) = next k + P) (t : ℕ) :
    ∀ k, next (k + t * P) = next k + t * P := by
  induction t with
  | zero => intro k; simp
  | succ t ih =>
    intro k
    rw [Nat.succ_mul, ← Nat.add_assoc, hnext (k + t * P), ih k]
    omega

/-! ## The census reduction: an infinite claim is a one-period claim

Every hypothesis of the dictionary vehicle has the shape `∀ n, φ (g n, ...,
g (n + j))` - a claim about EVERY index.  What the full-period scan actually
verifies is the same claim for the indices of ONE period.  `index_reduce`
closes that gap for any periodic machine, with no walk and no base case: the
forward gap word at any index is the forward gap word at an index whose
opening lies in `[1, P]`.
-/

/-- **THE CENSUS REDUCTION.**  For every index `n` there is an index `m` whose
opening lies in the FIRST PERIOD and whose entire forward gap word agrees with
`n`'s.  Hence any `∀ n`-claim about the gap word follows from the same claim
restricted to `op m ≤ P`. -/
theorem index_reduce {E : ℕ → Prop} {next op g : ℕ → ℕ} {P : ℕ} (hP : 0 < P)
    (hsucc : ∀ n, op (n + 1) = next (op n))
    (hnext : ∀ k, next (k + P) = next k + P)
    (hEop : ∀ n, E (op n)) (hposop : ∀ n, 1 ≤ op n)
    (hper : ∀ k, 1 ≤ k → (E (k + P) ↔ E k))
    (hsurj : ∀ m, 1 ≤ m → E m → ∃ n, op n = m)
    (hg : ∀ n, g n = op (n + 1) - op n)
    (n : ℕ) : ∃ m, op m ≤ P ∧ ∀ i, g (n + i) = g (m + i) := by
  have hx : 1 ≤ op n := hposop n
  have hdm : (op n - 1) % P + (op n - 1) / P * P = op n - 1 :=
    Nat.mod_add_div' _ _
  have hmod : (op n - 1) % P < P := Nat.mod_lt _ hP
  have hxr : op n = ((op n - 1) % P + 1) + (op n - 1) / P * P := by omega
  have hr1 : 1 ≤ (op n - 1) % P + 1 := by omega
  have hEr : E ((op n - 1) % P + 1) := by
    have h := pred_shift_mul (E := E) hper ((op n - 1) / P) _ hr1
    rw [← hxr] at h
    exact h.mp (hEop n)
  obtain ⟨m, hm⟩ := hsurj _ hr1 hEr
  refine ⟨m, by omega, ?_⟩
  -- the two enumerations run in lockstep, one period apart
  have hlock : ∀ i, op (n + i) = op (m + i) + (op n - 1) / P * P := by
    intro i
    induction i with
    | zero => rw [Nat.add_zero, Nat.add_zero, hm]; exact hxr
    | succ i ih =>
      rw [show n + (i + 1) = (n + i) + 1 by omega, hsucc, ih,
        next_shift_mul hnext, ← hsucc, show m + i + 1 = m + (i + 1) by omega]
  intro i
  have h1 := hlock i
  have h2 := hlock (i + 1)
  rw [show n + (i + 1) = (n + i) + 1 by omega] at h2
  rw [show m + (i + 1) = (m + i) + 1 by omega] at h2
  rw [hg (n + i), hg (m + i)]
  omega

/-! ## Consequences for the gap word -/

/-- **The gap word is periodic with period `N`.** -/
theorem gap_shift {g op : ℕ → ℕ} {P N : ℕ}
    (hg : ∀ n, g n = op (n + 1) - op n)
    (h : ∀ n, op (n + N) = op n + P) (n : ℕ) : g (n + N) = g n := by
  have h1 := h n
  have h2 := h (n + 1)
  have e1 : g (n + N) = op (n + N + 1) - op (n + N) := hg (n + N)
  have e2 : g n = op (n + 1) - op n := hg n
  rw [show n + N + 1 = n + 1 + N by omega] at e1
  rw [h2, h1] at e1
  omega

/-- The gap word iterated: `t` periods of indices change nothing. -/
theorem gap_shift_mul {g : ℕ → ℕ} {N : ℕ}
    (h : ∀ n, g (n + N) = g n) (t n : ℕ) : g (n + t * N) = g n := by
  induction t with
  | zero => simp
  | succ t ih =>
    rw [Nat.succ_mul, ← Nat.add_assoc, h, ih]

/-- **The gap word is its own residue table**: `g n = g (n % N)`, so a word
of `N` letters determines the whole sequence. -/
theorem gap_mod {g : ℕ → ℕ} {N : ℕ}
    (h : ∀ n, g (n + N) = g n) (n : ℕ) : g n = g (n % N) := by
  have hd := gap_shift_mul h (n / N) (n % N)
  rw [show n % N + n / N * N = n from Nat.mod_add_div' n N] at hd
  exact hd

/-- Window sums inherit the period. -/
theorem windowSum_shift {g : ℕ → ℕ} {N : ℕ}
    (h : ∀ n, g (n + N) = g n) (a j : ℕ) :
    Spectrum.windowSum g (a + N) j = Spectrum.windowSum g a j := by
  simp only [Spectrum.windowSum]
  exact Finset.sum_congr rfl fun i _ => by
    rw [show a + N + i = (a + i) + N by omega, h]

end Periodic
