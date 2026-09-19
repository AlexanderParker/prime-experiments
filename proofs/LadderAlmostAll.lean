/-
LadderAlmostAll (round 120, 2026-09-19): chains of every finite length follow from an almost-all
power-saving lower bound on rung counts.

THE STATEMENT IN WORDS.  Suppose (a) the twin centres up to `X` number at least `X^(1-η)` for all
large `X`, and (b) all but at most `X^(1-δ)` of them have at least `c·s/(log s)²` rungs.  Then the
rung forest has chains of every finite length (`ChainHyp`), hence twin primes are unbounded
(`twins_unbounded_of_chains` in LadderDepth).

THE MECHANISM.  `Bad_k(X)` is the set of twin centres `s ≤ X` rooting no chain of length `k`.
A centre is in `Bad_{k+1}(X)` only if every one of its rungs lies in `Bad_k((X+1)²)`.  Because the
rung graph is a forest (`parent_unique`: a twin centre is a rung of at most one centre), the rung
sets of distinct centres are DISJOINT, so summing rung counts over `Bad_{k+1}(X)` gives at most
`|Bad_k((X+1)²)|`.  Splitting `Bad_{k+1}(X)` into the exceptional centres (at most `X^(1-δ)`), the
centres below `X^(1-γ)` (at most `X^(1-γ) + 1`) and the rest - each with at least
`c·ε²·X^(1-δ)` rungs where `ε = (δ-γ)/2`, by `log x ≤ x^ε/ε` - gives
`|Bad_{k+1}(X)| ≤ C'·X^(1-γ)` from `|Bad_k(Y)| ≤ C·Y^(1-γ')` with `γ' = (γ+δ)/2`.  The saving
`γ` may be taken anywhere in `(0, δ)`, uniformly in `k`; the exceptional-set bound is the
bottleneck, not the depth.  With `γ` between `η` and `δ`, `|Bad_n(X)| < X^(1-η) ≤ |twins ≤ X|` for
large `X`, so some twin centre roots a chain of length `n`.

HYPOTHESES.  The theorem is stated with the requested hypotheses `0 < c`, `0 < δ < 1`, `0 < η`,
`2η < δ`; the proof uses only `η < δ` from the last one (no adjustment was needed).  The
exceptional set is a power saving `X^(1-δ)` below the twin count `X^(1-η)`.
-/
import LadderDepth
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Pow.Asymptotics

open Filter

namespace TwinLadder

open scoped Classical

/-- The rungs of `s`, as a finite set: every rung `s'` satisfies `s' < (s+1)^2`. -/
noncomputable def rungs (s : ℕ) : Finset ℕ :=
  (Finset.range ((s + 1) ^ 2)).filter (fun s' => Rung s s')

/-- The twin centres up to `X`. -/
noncomputable def twinsUpTo (X : ℕ) : Finset ℕ := (Finset.range (X + 1)).filter TwinCentre

/-- `Roots k s`: the twin centre `s` roots a chain of `k` rungs. -/
def Roots : ℕ → ℕ → Prop
  | 0, s => TwinCentre s
  | k + 1, s => TwinCentre s ∧ ∃ s', Rung s s' ∧ Roots k s'

/-- `Bad_k(X)`: the twin centres `s ≤ X` that root no chain of length `k`. -/
noncomputable def bad (k X : ℕ) : Finset ℕ := (twinsUpTo X).filter (fun s => ¬ Roots k s)

/-- The exceptional predicate: fewer than `c·s/(log s)²` rungs. -/
def isExc (c : ℝ) (s : ℕ) : Prop := ((rungs s).card : ℝ) < c * s / (Real.log s) ^ 2

/-- The exceptional twin centres up to `X`: those with fewer than `c·s/(log s)²` rungs.  This is
`(twinsUpTo X).filter (fun s => ((rungs s).card : ℝ) < c * s / (Real.log s) ^ 2)`. -/
noncomputable def exceptional (c : ℝ) (X : ℕ) : Finset ℕ := (twinsUpTo X).filter (isExc c)

/-- **The almost-all hypothesis** with constants `c, δ, η`: (a) at least `X^(1-η)` twin centres
up to `X`, and (b) at most `X^(1-δ)` exceptional ones, for all large `X`. -/
def AlmostAll (c δ η : ℝ) : Prop :=
  (∀ᶠ X : ℕ in atTop, (X : ℝ) ^ (1 - η) ≤ (twinsUpTo X).card) ∧
  (∀ᶠ X : ℕ in atTop, ((exceptional c X).card : ℝ) ≤ (X : ℝ) ^ (1 - δ))

/-! ### The combinatorial kernel: roots, bad sets, disjoint rung sets -/

/-- Membership in `rungs s`. -/
theorem mem_rungs {s s' : ℕ} : s' ∈ rungs s ↔ s' < (s + 1) ^ 2 ∧ Rung s s' := by
  simp only [rungs, Finset.mem_filter, Finset.mem_range]

/-- Membership in `twinsUpTo X`. -/
theorem mem_twinsUpTo {X s : ℕ} : s ∈ twinsUpTo X ↔ s ≤ X ∧ TwinCentre s := by
  simp only [twinsUpTo, Finset.mem_filter, Finset.mem_range, Nat.lt_succ_iff]

/-- Membership in `bad k X`. -/
theorem mem_bad {k X s : ℕ} : s ∈ bad k X ↔ (s ≤ X ∧ TwinCentre s) ∧ ¬ Roots k s := by
  simp only [bad, Finset.mem_filter, mem_twinsUpTo]

/-- Every twin centre roots the empty chain: `Bad_0(X) = ∅`. -/
theorem bad_zero (X : ℕ) : bad 0 X = ∅ := by
  ext s
  simp only [mem_bad, Finset.notMem_empty, iff_false, not_and, not_not]
  intro h; exact h.2

/-- A rung of a centre in `Bad_{k+1}(X)` lies in `Bad_k((X+1)²)`. -/
theorem rungs_subset_bad {k X s : ℕ} (hs : s ∈ bad (k + 1) X) :
    rungs s ⊆ bad k ((X + 1) ^ 2) := by
  intro s' hs'
  rw [mem_rungs] at hs'
  rw [mem_bad] at hs ⊢
  refine ⟨⟨?_, hs'.2.1⟩, ?_⟩
  · have : (s + 1) ^ 2 ≤ (X + 1) ^ 2 := Nat.pow_le_pow_left (by omega) 2
    omega
  · intro hr
    exact hs.2 ⟨hs.1.2, s', hs'.2, hr⟩

/-- The rung sets of distinct twin centres are disjoint (`parent_unique`). -/
theorem rungs_disjoint {s t : ℕ} (hs : TwinCentre s) (ht : TwinCentre t) (hne : s ≠ t) :
    Disjoint (rungs s) (rungs t) := by
  rw [Finset.disjoint_left]
  intro s' h1 h2
  rw [mem_rungs] at h1 h2
  exact hne (parent_unique hs ht h1.2 h2.2)

/-- **The counting step**: the rung counts over `Bad_{k+1}(X)` sum to at most `|Bad_k((X+1)²)|`. -/
theorem sum_rungs_le (k X : ℕ) :
    ∑ s ∈ bad (k + 1) X, (rungs s).card ≤ (bad k ((X + 1) ^ 2)).card := by
  have hdisj : ((bad (k + 1) X : Finset ℕ) : Set ℕ).PairwiseDisjoint rungs := by
    intro s hs t ht hne
    exact rungs_disjoint (mem_bad.1 (Finset.mem_coe.1 hs)).1.2
      (mem_bad.1 (Finset.mem_coe.1 ht)).1.2 hne
  rw [← Finset.card_biUnion hdisj]
  exact Finset.card_le_card (Finset.biUnion_subset.2 fun s hs => rungs_subset_bad hs)

/-- A root of a chain of length `n` gives a chain of length `n` starting there. -/
theorem chain_of_roots : ∀ n s, Roots n s → ∃ f : ℕ → ℕ, Chain n f ∧ f 0 = s := by
  intro n
  induction n with
  | zero =>
    intro s hs
    exact ⟨fun _ => s, ⟨hs, fun k hk => by omega⟩, rfl⟩
  | succ k ih =>
    intro s hs
    obtain ⟨hs0, s', hr, hs'⟩ := hs
    obtain ⟨f, hf, hf0⟩ := ih s' hs'
    refine ⟨fun i => match i with | 0 => s | j + 1 => f j, ⟨hs0, ?_⟩, rfl⟩
    intro i hi
    cases i with
    | zero => show Rung s (f 0); rw [hf0]; exact hr
    | succ j => show Rung (f j) (f (j + 1)); exact hf.2 j (by omega)

/-! ### The analytic step -/

/-- The per-centre lower bound: a non-exceptional twin centre `s` with `X^(1-γ) ≤ s ≤ X` has at
least `c·ε²·X^(1-δ)` rungs, `ε = (δ-γ)/2`, from `(log s)² ≤ (log X)² ≤ X^(2ε)/ε²`. -/
theorem lower_bound_of_ge {c δ γ : ℝ} {s X : ℕ} (hc : 0 < c) (hγδ : γ < δ)
    (hs1 : 1 < s) (hsX : s ≤ X) (hY : (X : ℝ) ^ (1 - γ) ≤ s) :
    c * ((δ - γ) / 2) ^ 2 * (X : ℝ) ^ (1 - δ) ≤ c * s / (Real.log s) ^ 2 := by
  have hεpos : 0 < (δ - γ) / 2 := by linarith
  have hs1' : (1 : ℝ) < s := by exact_mod_cast hs1
  have hsX' : (s : ℝ) ≤ X := by exact_mod_cast hsX
  have hX0 : (0 : ℝ) < X := by linarith
  have hlogs : 0 < Real.log s := Real.log_pos hs1'
  rw [le_div_iff₀ (pow_pos hlogs 2)]
  have h1 : Real.log s ≤ Real.log X := Real.log_le_log (by linarith) hsX'
  have h2 : Real.log X ≤ (X : ℝ) ^ ((δ - γ) / 2) / ((δ - γ) / 2) :=
    Real.log_le_rpow_div hX0.le hεpos
  have h3 : (Real.log s) ^ 2 ≤ ((X : ℝ) ^ ((δ - γ) / 2) / ((δ - γ) / 2)) ^ 2 :=
    pow_le_pow_left₀ hlogs.le (h1.trans h2) 2
  have h4 : ((X : ℝ) ^ ((δ - γ) / 2)) ^ 2 = (X : ℝ) ^ (δ - γ) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hX0.le]
    congr 1; push_cast; ring
  have h5 : (X : ℝ) ^ (1 - δ) * (X : ℝ) ^ (δ - γ) = (X : ℝ) ^ (1 - γ) := by
    rw [← Real.rpow_add hX0]; congr 1; ring
  have hε2 : 0 < ((δ - γ) / 2) ^ 2 := pow_pos hεpos 2
  calc c * ((δ - γ) / 2) ^ 2 * (X : ℝ) ^ (1 - δ) * (Real.log s) ^ 2
      ≤ c * ((δ - γ) / 2) ^ 2 * (X : ℝ) ^ (1 - δ) *
          ((X : ℝ) ^ (δ - γ) / ((δ - γ) / 2) ^ 2) := by
        rw [← h4, ← div_pow]
        exact mul_le_mul_of_nonneg_left h3
          (mul_nonneg (mul_nonneg hc.le hε2.le) (Real.rpow_nonneg hX0.le _))
    _ = c * (X : ℝ) ^ (1 - γ) := by
        have hne : δ - γ ≠ 0 := by linarith
        rw [← h5]; field_simp
    _ ≤ c * s := mul_le_mul_of_nonneg_left hY hc.le

/-- `Bad_{k+1}(X)` splits into the exceptional centres, the small ones (below `Y`) and the large
non-exceptional ones. -/
theorem card_bad_split (k X : ℕ) (c Y : ℝ) :
    (bad (k + 1) X).card ≤
      ((bad (k + 1) X).filter (isExc c)).card +
      ((bad (k + 1) X).filter (fun s => ¬ isExc c s ∧ (s : ℝ) < Y)).card +
      ((bad (k + 1) X).filter (fun s => ¬ isExc c s ∧ Y ≤ s)).card := by
  have hsub : bad (k + 1) X ⊆
      (bad (k + 1) X).filter (isExc c) ∪
      (bad (k + 1) X).filter (fun s => ¬ isExc c s ∧ (s : ℝ) < Y) ∪
      (bad (k + 1) X).filter (fun s => ¬ isExc c s ∧ Y ≤ s) := by
    intro s hs
    simp only [Finset.mem_union, Finset.mem_filter]
    by_cases h1 : isExc c s
    · exact Or.inl (Or.inl ⟨hs, h1⟩)
    by_cases h2 : (s : ℝ) < Y
    · exact Or.inl (Or.inr ⟨hs, h1, h2⟩)
    · exact Or.inr ⟨hs, h1, not_lt.1 h2⟩
  calc (bad (k + 1) X).card ≤ _ := Finset.card_le_card hsub
    _ ≤ _ := Finset.card_union_le _ _
    _ ≤ _ := Nat.add_le_add_right (Finset.card_union_le _ _) _

/-- The exceptional part of `Bad_{k+1}(X)` is inside the exceptional set. -/
theorem card_filter_exc_le (k X : ℕ) (c : ℝ) :
    ((bad (k + 1) X).filter (isExc c)).card ≤ (exceptional c X).card :=
  Finset.card_le_card (Finset.filter_subset_filter _ (Finset.filter_subset _ _))

/-- The small part: naturals below `Y` number at most `Y + 1`. -/
theorem card_filter_small_le (k X : ℕ) (c Y : ℝ) (hY : 0 ≤ Y) :
    (((bad (k + 1) X).filter (fun s => ¬ isExc c s ∧ (s : ℝ) < Y)).card : ℝ) ≤ Y + 1 := by
  have hsub : (bad (k + 1) X).filter (fun s => ¬ isExc c s ∧ (s : ℝ) < Y) ⊆
      Finset.range (⌊Y⌋₊ + 1) := by
    intro s hs
    rw [Finset.mem_filter] at hs
    rw [Finset.mem_range, Nat.lt_succ_iff]
    exact Nat.le_floor hs.2.2.le
  calc (((bad (k + 1) X).filter (fun s => ¬ isExc c s ∧ (s : ℝ) < Y)).card : ℝ)
      ≤ ((Finset.range (⌊Y⌋₊ + 1)).card : ℝ) := by exact_mod_cast Finset.card_le_card hsub
    _ = (⌊Y⌋₊ : ℝ) + 1 := by simp
    _ ≤ Y + 1 := by linarith [Nat.floor_le hY]

/-- The large part: with `|Bad_k((X+1)²)| ≤ C·((X+1)²)^(1-γ')`, `γ' = (γ+δ)/2`, the large
non-exceptional centres of `Bad_{k+1}(X)` number at most `(4C/(c ε²))·X^(1-γ)`, `ε = (δ-γ)/2`. -/
theorem card_filter_large_le {c δ γ C : ℝ} {k X : ℕ} (hc : 0 < c) (hγ : 0 < γ) (hγδ : γ < δ)
    (hδ1 : δ < 1) (hC : 0 ≤ C) (hX1 : 1 ≤ X)
    (hB : ((bad k ((X + 1) ^ 2)).card : ℝ) ≤ C * (((X + 1) ^ 2 : ℕ) : ℝ) ^ (1 - (γ + δ) / 2)) :
    (((bad (k + 1) X).filter (fun s => ¬ isExc c s ∧ (X : ℝ) ^ (1 - γ) ≤ s)).card : ℝ) ≤
      (4 * C / (c * ((δ - γ) / 2) ^ 2)) * (X : ℝ) ^ (1 - γ) := by
  have hX0 : (0 : ℝ) < X := Nat.cast_pos.2 (by omega)
  have hX1' : (1 : ℝ) ≤ X := by exact_mod_cast hX1
  have hε2 : 0 < ((δ - γ) / 2) ^ 2 := pow_pos (by linarith) 2
  have hcε : 0 < c * ((δ - γ) / 2) ^ 2 := mul_pos hc hε2
  have hXδ : 0 < (X : ℝ) ^ (1 - δ) := Real.rpow_pos_of_pos hX0 _
  -- the per-element lower bound on the large non-exceptional centres
  have hlow : ∀ s ∈ (bad (k + 1) X).filter (fun s => ¬ isExc c s ∧ (X : ℝ) ^ (1 - γ) ≤ s),
      c * ((δ - γ) / 2) ^ 2 * (X : ℝ) ^ (1 - δ) ≤ ((rungs s).card : ℝ) := by
    intro s hs
    obtain ⟨hsB, hnexc, hsY⟩ := Finset.mem_filter.1 hs
    have hmem := mem_bad.1 hsB
    have h6 := twinCentre_ge_six hmem.1.2
    have hlb := lower_bound_of_ge (s := s) (X := X) hc hγδ (by omega) hmem.1.1 hsY
    exact hlb.trans (not_lt.1 hnexc)
  -- sum over the large part, then over all of Bad_{k+1}(X), then the counting step
  have hsum : (((bad (k + 1) X).filter
        (fun s => ¬ isExc c s ∧ (X : ℝ) ^ (1 - γ) ≤ s)).card : ℝ) *
        (c * ((δ - γ) / 2) ^ 2 * (X : ℝ) ^ (1 - δ)) ≤
      ∑ s ∈ (bad (k + 1) X).filter (fun s => ¬ isExc c s ∧ (X : ℝ) ^ (1 - γ) ≤ s),
        ((rungs s).card : ℝ) := by
    have := Finset.card_nsmul_le_sum _ (fun s => ((rungs s).card : ℝ)) _ hlow
    simpa [nsmul_eq_mul] using this
  have hsum2 : ∑ s ∈ (bad (k + 1) X).filter (fun s => ¬ isExc c s ∧ (X : ℝ) ^ (1 - γ) ≤ s),
        ((rungs s).card : ℝ) ≤ ∑ s ∈ bad (k + 1) X, ((rungs s).card : ℝ) :=
    Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _)
      (fun _ _ _ => Nat.cast_nonneg _)
  have hsum3 : ∑ s ∈ bad (k + 1) X, ((rungs s).card : ℝ) ≤ ((bad k ((X + 1) ^ 2)).card : ℝ) := by
    exact_mod_cast sum_rungs_le k X
  -- ((X+1)²)^(1-γ') ≤ 4·X^(1-γ)·X^(1-δ)
  have hpow : (((X + 1) ^ 2 : ℕ) : ℝ) ^ (1 - (γ + δ) / 2) ≤
      4 * ((X : ℝ) ^ (1 - γ) * (X : ℝ) ^ (1 - δ)) := by
    have hγ'0 : 0 ≤ 1 - (γ + δ) / 2 := by linarith
    have hγ'1 : 1 - (γ + δ) / 2 ≤ 1 := by linarith
    have hcast : (((X + 1) ^ 2 : ℕ) : ℝ) = ((X : ℝ) + 1) ^ 2 := by push_cast; ring
    have hle : ((X : ℝ) + 1) ^ 2 ≤ 4 * (X : ℝ) ^ 2 := by nlinarith
    have h4 : (4 : ℝ) ^ (1 - (γ + δ) / 2) ≤ 4 := by
      calc (4 : ℝ) ^ (1 - (γ + δ) / 2) ≤ (4 : ℝ) ^ (1 : ℝ) :=
            Real.rpow_le_rpow_of_exponent_le (by norm_num) hγ'1
        _ = 4 := Real.rpow_one 4
    calc (((X + 1) ^ 2 : ℕ) : ℝ) ^ (1 - (γ + δ) / 2)
        = (((X : ℝ) + 1) ^ 2) ^ (1 - (γ + δ) / 2) := by rw [hcast]
      _ ≤ (4 * (X : ℝ) ^ 2) ^ (1 - (γ + δ) / 2) :=
          Real.rpow_le_rpow (by positivity) hle hγ'0
      _ = (4 : ℝ) ^ (1 - (γ + δ) / 2) * ((X : ℝ) ^ 2) ^ (1 - (γ + δ) / 2) :=
          Real.mul_rpow (by norm_num) (by positivity)
      _ ≤ 4 * ((X : ℝ) ^ 2) ^ (1 - (γ + δ) / 2) :=
          mul_le_mul_of_nonneg_right h4 (Real.rpow_nonneg (by positivity) _)
      _ = 4 * ((X : ℝ) ^ (1 - γ) * (X : ℝ) ^ (1 - δ)) := by
          rw [← Real.rpow_add hX0, ← Real.rpow_natCast, ← Real.rpow_mul hX0.le]
          congr 2; push_cast; ring
  -- assemble and cancel X^(1-δ) and c ε²
  have key : ((((bad (k + 1) X).filter
        (fun s => ¬ isExc c s ∧ (X : ℝ) ^ (1 - γ) ≤ s)).card : ℝ) * (c * ((δ - γ) / 2) ^ 2)) *
        (X : ℝ) ^ (1 - δ) ≤
      (4 * C * (X : ℝ) ^ (1 - γ)) * (X : ℝ) ^ (1 - δ) := by
    calc _ = (((bad (k + 1) X).filter
          (fun s => ¬ isExc c s ∧ (X : ℝ) ^ (1 - γ) ≤ s)).card : ℝ) *
          (c * ((δ - γ) / 2) ^ 2 * (X : ℝ) ^ (1 - δ)) := by ring
      _ ≤ _ := hsum
      _ ≤ _ := hsum2
      _ ≤ _ := hsum3
      _ ≤ C * (((X + 1) ^ 2 : ℕ) : ℝ) ^ (1 - (γ + δ) / 2) := hB
      _ ≤ C * (4 * ((X : ℝ) ^ (1 - γ) * (X : ℝ) ^ (1 - δ))) := mul_le_mul_of_nonneg_left hpow hC
      _ = _ := by ring
  have h1 := le_of_mul_le_mul_right key hXδ
  rw [div_mul_eq_mul_div, le_div_iff₀ hcε]
  exact h1

/-- **The induction**: for every depth `k` and every saving `γ ∈ (0, δ)`, `|Bad_k(X)| ≤ C·X^(1-γ)`
for all large `X`. -/
theorem bad_card_le {c δ η : ℝ} (hc : 0 < c) (hδ1 : δ < 1) (h : AlmostAll c δ η) :
    ∀ k : ℕ, ∀ γ : ℝ, 0 < γ → γ < δ →
      ∃ C : ℝ, 0 ≤ C ∧ ∀ᶠ X : ℕ in atTop, ((bad k X).card : ℝ) ≤ C * (X : ℝ) ^ (1 - γ) := by
  intro k
  induction k with
  | zero =>
    intro γ _ _
    refine ⟨0, le_rfl, Eventually.of_forall fun X => ?_⟩
    simp [bad_zero]
  | succ k ih =>
    intro γ hγ hγδ
    obtain ⟨C, hC, hev⟩ := ih ((γ + δ) / 2) (by linarith) (by linarith)
    have hε2 : 0 < ((δ - γ) / 2) ^ 2 := pow_pos (by linarith) 2
    have hcε : 0 < c * ((δ - γ) / 2) ^ 2 := mul_pos hc hε2
    refine ⟨3 + 4 * C / (c * ((δ - γ) / 2) ^ 2),
      add_nonneg (by norm_num) (div_nonneg (by linarith) hcε.le), ?_⟩
    have hsq : Tendsto (fun X : ℕ => (X + 1) ^ 2) atTop atTop :=
      tendsto_atTop_mono (fun X => show X ≤ (X + 1) ^ 2 by nlinarith) tendsto_id
    filter_upwards [h.2, hsq.eventually hev, eventually_ge_atTop 1] with X hE hB hX1
    have hX0 : (0 : ℝ) < X := Nat.cast_pos.2 (by omega)
    have hX1' : (1 : ℝ) ≤ X := by exact_mod_cast hX1
    have hY1 : 1 ≤ (X : ℝ) ^ (1 - γ) := Real.one_le_rpow hX1' (by linarith)
    have hXδγ : (X : ℝ) ^ (1 - δ) ≤ (X : ℝ) ^ (1 - γ) :=
      Real.rpow_le_rpow_of_exponent_le hX1' (by linarith)
    have hsplit := card_bad_split k X c ((X : ℝ) ^ (1 - γ))
    have hE' : (((bad (k + 1) X).filter (isExc c)).card : ℝ) ≤ (X : ℝ) ^ (1 - δ) :=
      le_trans (by exact_mod_cast card_filter_exc_le k X c) hE
    have hSm := card_filter_small_le k X c ((X : ℝ) ^ (1 - γ)) (Real.rpow_nonneg hX0.le _)
    have hS := card_filter_large_le hc hγ hγδ hδ1 hC hX1 hB
    calc ((bad (k + 1) X).card : ℝ)
        ≤ (((bad (k + 1) X).filter (isExc c)).card : ℝ) +
          (((bad (k + 1) X).filter (fun s => ¬ isExc c s ∧ (s : ℝ) < (X : ℝ) ^ (1 - γ))).card : ℝ) +
          (((bad (k + 1) X).filter (fun s => ¬ isExc c s ∧ (X : ℝ) ^ (1 - γ) ≤ s)).card : ℝ) := by
          exact_mod_cast hsplit
      _ ≤ (X : ℝ) ^ (1 - δ) + ((X : ℝ) ^ (1 - γ) + 1) +
          (4 * C / (c * ((δ - γ) / 2) ^ 2)) * (X : ℝ) ^ (1 - γ) := by
          gcongr
      _ ≤ (3 + 4 * C / (c * ((δ - γ) / 2) ^ 2)) * (X : ℝ) ^ (1 - γ) := by
          nlinarith

/-- **Chains of every finite length from the almost-all hypothesis.**  With `η < γ < δ`,
`|Bad_n(X)| ≤ C·X^(1-γ) < X^(1-η) ≤ |twins ≤ X|` for large `X`, so some twin centre up to `X`
roots a chain of length `n`. -/
theorem chainHyp_of_almostAll (c δ η : ℝ) (hc : 0 < c) (hδ : 0 < δ) (hδ1 : δ < 1) (hη : 0 < η)
    (hηδ : 2 * η < δ) (h : AlmostAll c δ η) : ChainHyp := by
  intro n
  obtain ⟨C, hC, hev⟩ := bad_card_le hc hδ1 h n ((η + δ) / 2) (by linarith) (by linarith)
  have hpow : Tendsto (fun X : ℕ => (X : ℝ) ^ ((η + δ) / 2 - η)) atTop atTop :=
    (tendsto_rpow_atTop (by linarith)).comp tendsto_natCast_atTop_atTop
  have hgt : ∀ᶠ X : ℕ in atTop, C < (X : ℝ) ^ ((η + δ) / 2 - η) := hpow.eventually_gt_atTop C
  obtain ⟨X, hX1, hbad, htw, hgtX⟩ := ((eventually_ge_atTop 1).and (hev.and (h.1.and hgt))).exists
  have hX0 : (0 : ℝ) < X := Nat.cast_pos.2 (by omega)
  have hlt : ((bad n X).card : ℝ) < ((twinsUpTo X).card : ℝ) := by
    calc ((bad n X).card : ℝ) ≤ C * (X : ℝ) ^ (1 - (η + δ) / 2) := hbad
      _ < (X : ℝ) ^ ((η + δ) / 2 - η) * (X : ℝ) ^ (1 - (η + δ) / 2) :=
          mul_lt_mul_of_pos_right hgtX (Real.rpow_pos_of_pos hX0 _)
      _ = (X : ℝ) ^ (1 - η) := by rw [← Real.rpow_add hX0]; congr 1; ring
      _ ≤ _ := htw
  have hlt' : (bad n X).card < (twinsUpTo X).card := by exact_mod_cast hlt
  obtain ⟨s, hs, hsn⟩ := Finset.exists_mem_notMem_of_card_lt_card hlt'
  have hroots : Roots n s := by
    by_contra hcon
    exact hsn (Finset.mem_filter.2 ⟨hs, hcon⟩)
  obtain ⟨f, hf, _⟩ := chain_of_roots n s hroots
  exact ⟨f, hf⟩

/-- **Twin primes unbounded from the almost-all hypothesis**, composing with LadderDepth. -/
theorem twins_unbounded_of_almostAll (c δ η : ℝ) (hc : 0 < c) (hδ : 0 < δ) (hδ1 : δ < 1)
    (hη : 0 < η) (hηδ : 2 * η < δ) (h : AlmostAll c δ η) :
    ∀ N : ℕ, ∃ m : ℕ, N < 6 * m - 1 ∧ (6 * m - 1).Prime ∧ (6 * m + 1).Prime :=
  twins_unbounded_of_chains (chainHyp_of_almostAll c δ η hc hδ hδ1 hη hηδ h)

end TwinLadder
