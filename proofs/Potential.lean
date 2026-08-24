/-
(D) WITHOUT A DEPTH QUANTIFIER - the potential / super-solution certificate
(round 22, from Constructor's R46, docs/novel/kleene-generator.md).

Constructor's Kleene generator writes `F(M + q')` as a max-plus product
`L^T (x) K* (x) R`, whose `m`-th layer is exactly `qualmax_{m+2}`: one algebra
for every layer of R39's ladder. Its corollary is the form that matters to a
proof: by max-plus LP duality, (D) at `alpha = 3` holds IFF a POTENTIAL `h`
exists satisfying three ONE-STEP, ONE-OPENING inequalities

    (C1)  d x <= h x                          for every state
    (C2)  d x + h y <= h x                    for every legal step x -> y
    (C3)  e x + h x <= B                      for every state

- the first form of (D) that is not an infinite family of statements indexed
by depth. THIS FILE KERNEL-CHECKS THE DIRECTION THAT DOES PROOF WORK: a
potential CERTIFIES the bound, for chains of EVERY length, by one induction.
Whoever exhibits an `h` at a machine gets (D) there with no depth analysis,
no fuel cap and no word list - the certificate is three inequalities.

Two forms are given: `D_of_potential`, abstract over any state type and step
relation (Constructor's states are `(opening, tooth)` pairs, so the state type
must stay general), and `merged_le_of_potential`, the same statement in the
gap-word vocabulary of `Spectrum.lean`, whose conclusion is literally (D)'s
merged window `g a + windowSum g (a+1) l + g (a+l+1) <= F + q'`.

NOT FORMALISED, deliberately (see formalist.md round 22): (i) the Kleene
identity itself, `F(M + q') = L^T (x) K* (x) R` - an equality, needing max-plus
matrix machinery and the machine's own `K`; (ii) the CONVERSE direction (a
potential always exists), which is where nilpotency of `K` is used - `h` is the
least super-solution, i.e. the max over tails, and building it needs the finite
path bound. Only the certificate direction is claimed here, and it is the one a
proof consumes.

HONEST CAVEAT carried from Constructor: the generator is arity-free but NOT YET
machine-free. Bounded-state certificates certify (D) at 19->23 (45 <= 48) and
FAIL at 29->31 (mod 35 / 385 / 5005 give 99 / 99 / 91 against a budget of 74).
So this file makes the target statement precise; it does not prove (D).
-/

import Spectrum

namespace Potential

/-! ## The abstract certificate -/

/-- **A potential** for a weighted step relation: `d` is the weight collected
at a state, `e` the entry weight, `B` the budget. All three clauses are
one-step and one-state - there is NO quantifier over path length anywhere in
this definition. -/
def IsPotential {St : Type*} (Step : St → St → Prop) (d e h : St → ℕ) (B : ℕ) :
    Prop :=
  (∀ x, d x ≤ h x) ∧ (∀ x y, Step x y → d x + h y ≤ h x) ∧ (∀ x, e x + h x ≤ B)

/-- **The certificate lemma.** (C1) and (C2) alone bound the total weight of
every legal chain by the potential at its head - one induction, any length. -/
theorem chain_le_potential {St : Type*} {Step : St → St → Prop} {d h : St → ℕ}
    (hC1 : ∀ x, d x ≤ h x) (hC2 : ∀ x y, Step x y → d x + h y ≤ h x) :
    ∀ (l : ℕ) (p : ℕ → St), (∀ k, k < l → Step (p k) (p (k + 1))) →
      ∑ k ∈ Finset.range (l + 1), d (p k) ≤ h (p 0) := by
  intro l
  induction l with
  | zero =>
    intro p _
    simpa using hC1 (p 0)
  | succ l ih =>
    intro p hstep
    have hIH : ∑ k ∈ Finset.range (l + 1), d (p (k + 1)) ≤ h (p 1) :=
      ih (fun k => p (k + 1)) (fun k hk => hstep (k + 1) (by omega))
    have hsplit : ∑ k ∈ Finset.range (l + 1 + 1), d (p k)
        = (∑ k ∈ Finset.range (l + 1), d (p (k + 1))) + d (p 0) :=
      Finset.sum_range_succ' (fun k => d (p k)) (l + 1)
    have h0 := hC2 (p 0) (p 1) (hstep 0 (by omega))
    omega

/-- **(D) from a potential, with no depth quantifier.** A potential bounds the
entry weight plus the whole chain by the budget - for chains of EVERY length.
The hypothesis `IsPotential` is three one-step inequalities; the conclusion is
an infinite family. -/
theorem D_of_potential {St : Type*} {Step : St → St → Prop} {d e h : St → ℕ}
    {B : ℕ} (hP : IsPotential Step d e h B) (l : ℕ) (p : ℕ → St)
    (hstep : ∀ k, k < l → Step (p k) (p (k + 1))) :
    e (p 0) + ∑ k ∈ Finset.range (l + 1), d (p k) ≤ B := by
  obtain ⟨hC1, hC2, hC3⟩ := hP
  have h1 := chain_le_potential hC1 hC2 l p hstep
  have h2 := hC3 (p 0)
  omega

/-! ## The same certificate in the gap-word vocabulary -/

/-- Peeling the FIRST gap off a window sum. -/
theorem windowSum_succ_left (g : ℕ → ℕ) (b l : ℕ) :
    Spectrum.windowSum g b (l + 1) = g b + Spectrum.windowSum g (b + 1) l := by
  simp only [Spectrum.windowSum]
  rw [Finset.sum_range_succ' (fun i => g (b + i)) l]
  have hsh : ∀ i, g (b + (i + 1)) = g (b + 1 + i) := by
    intro i; congr 1; omega
  simp only [hsh, Nat.add_zero]
  omega

/-- The chain bound over a gap word: from a state `b`, a run of `l`
floor-respecting letters plus the trailing flank is bounded by `h b`. -/
theorem tail_le_potential {g h : ℕ → ℕ} {u : ℕ}
    (hC1 : ∀ i, g i ≤ h i)
    (hC2 : ∀ i, 2 * u ≤ g i → g i + h (i + 1) ≤ h i) :
    ∀ (l b : ℕ), (∀ i < l, 2 * u ≤ g (b + i)) →
      Spectrum.windowSum g b l + g (b + l) ≤ h b := by
  intro l
  induction l with
  | zero =>
    intro b _
    simpa [Spectrum.windowSum] using hC1 b
  | succ l ih =>
    intro b hw
    have hq : 2 * u ≤ g b := by simpa using hw 0 (by omega)
    have hIH : Spectrum.windowSum g (b + 1) l + g (b + 1 + l) ≤ h (b + 1) := by
      refine ih (b + 1) ?_
      intro i hi
      have := hw (i + 1) (by omega)
      rwa [show b + (i + 1) = b + 1 + i by omega] at this
    have hstep := hC2 b hq
    have hpeel := windowSum_succ_left g b l
    have he : b + (l + 1) = b + 1 + l := by omega
    rw [hpeel, he]
    omega

/-- **(D) at `alpha = 3` from a potential, in the merged-window form.** `h` is
Constructor's potential on the gap word: (C1) it dominates each gap, (C2) it
decreases by the gap along every QUALIFYING step, (C3) a flank plus the
potential fits the budget. Then EVERY floor-respecting word of EVERY length
merges inside `F + q'` - no `k_win`, no fuel cap, no word list, and no
quantifier over depth in any hypothesis. -/
theorem merged_le_of_potential {g h : ℕ → ℕ} {u F q : ℕ}
    (hC1 : ∀ i, g i ≤ h i)
    (hC2 : ∀ i, 2 * u ≤ g i → g i + h (i + 1) ≤ h i)
    (hC3 : ∀ i, g i + h (i + 1) ≤ F + q)
    {a l : ℕ} (hw : ∀ i < l, 2 * u ≤ g (a + 1 + i)) :
    g a + Spectrum.windowSum g (a + 1) l + g (a + l + 1) ≤ F + q := by
  have htail : Spectrum.windowSum g (a + 1) l + g (a + 1 + l) ≤ h (a + 1) :=
    tail_le_potential hC1 hC2 l (a + 1) hw
  have he : a + 1 + l = a + l + 1 := by omega
  rw [he] at htail
  have := hC3 a
  omega

end Potential
