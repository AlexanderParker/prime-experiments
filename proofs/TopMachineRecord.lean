/-
# Round 36: the loaded record rule (branch R7, `research/proof/top_machine_7.md`)

The wheel record as a covering problem, in the kernel:

* **L67, the piece law** - what a gear can show inside a window `[0, L)`.  For a
  gear above the boundary (`L + 1 < g`) the trace is a subset of ONE distance-2
  domino `{c, c + 2}`, hence lives in one parity class; a gear at least as big as
  the window can join the two ends of the window (cross parity) exactly when
  `g <= L + 1`, and it can strike BOTH endpoints `0` and `L - 1` exactly when
  `g = L + 1`.
* **L68, the matching lemma** - the domino cost of a set: dominoes never cross
  parity, a set splits over the two parity classes, and a step-2 run of `j` cells
  costs exactly `ceil(j / 2)`.
* **L69, the loaded record rule**, both directions: a window of length `L` is
  covered by SOME phasing of `G` iff some phasing of the core gears
  (`g <= L + 1`) leaves an uncovered set of domino cost at most the number of
  tail gears (`g > L + 1`).
* **the boundary corollary** - with an empty core the cost of `[0, L)` is
  `2 floor(L/4) + min(L mod 4, 2)`, the largest affordable length is
  `2m - (m mod 2)`, and this re-proves the parity law L17 independently of
  `parity_upper` / `parity_attained`.

Everything is stated for an arbitrary `Finset` of gears; the hypotheses are the
ones the proofs actually needed and they are WEAKER than the branch's:

* the piece law and the necessity half of L69 need NOTHING (not primality, not
  oddness, not coprimality) - only the definition of the tail;
* the sufficiency half needs pairwise coprimality alone (no size hypothesis on
  the tail gears at all: a gear always strikes both cells of the domino its
  residue names);
* oddness and `2m + 1 < g` are used only in the boundary corollary, exactly as
  in round 32/33.

Zero sorries, no `native_decide`, no `decide`, no `Lean.ofReduceBool`.
-/
import TopMachineWalk

namespace TopMachine

/-! ## 1. The piece law (L67)

The trace of gear `g` at phase `n` inside the window `[0, L)`. -/

instance decStrikesRec (g : ℕ) (n : ℤ) : Decidable (Strikes g n) := by
  unfold Strikes; infer_instance

/-- The cells of `[0, L)` that gear `g` strikes when the window starts at `n`. -/
def trace (g : ℕ) (n : ℤ) (L : ℕ) : Finset ℕ :=
  (Finset.range L).filter (fun c => Strikes g (n + (c : ℤ)))

theorem mem_trace {g : ℕ} {n : ℤ} {L c : ℕ} :
    c ∈ trace g n L ↔ c < L ∧ Strikes g (n + (c : ℤ)) := by
  simp [trace]

/-- **L67, the tail piece.**  A gear above the boundary shows inside `[0, L)`
a subset of a single distance-2 domino `{c, c + 2}`: two cells, or one cell when
the partner falls outside, or none. -/
theorem trace_subset_domino {g : ℕ} {n : ℤ} {L : ℕ} (hg : L + 1 < g) :
    ∃ c : ℕ, trace g n L ⊆ ({c, c + 2} : Finset ℕ) := by
  classical
  rcases Finset.eq_empty_or_nonempty (trace g n L) with h | h
  · exact ⟨0, by rw [h]; exact Finset.empty_subset _⟩
  refine ⟨(trace g n L).min' h, ?_⟩
  intro x hx
  have ha := Finset.min'_mem _ h
  have hx' := mem_trace.mp hx
  have ha' := mem_trace.mp ha
  have hxa : (trace g n L).min' h ≤ x := Finset.min'_le _ x hx
  have hp := window_pair (g := g) (n := n) (L := L) (by omega) hx'.1 ha'.1 hx'.2 ha'.2
  have hcast : (((trace g n L).min' h : ℕ) : ℤ) ≤ (x : ℤ) := by exact_mod_cast hxa
  simp only [Finset.mem_insert, Finset.mem_singleton]
  omega

/-- A gear above the boundary strikes at most two cells of the window. -/
theorem trace_card_le_two {g : ℕ} {n : ℤ} {L : ℕ} (hg : L + 1 < g) :
    (trace g n L).card ≤ 2 := by
  obtain ⟨c, hc⟩ := trace_subset_domino (n := n) hg
  refine le_trans (Finset.card_le_card hc) ?_
  exact le_trans (Finset.card_insert_le _ _) (by simp)

/-- A gear above the boundary keeps to ONE parity class inside the window. -/
theorem trace_one_parity {g : ℕ} {n : ℤ} {L : ℕ} (hg : L + 1 < g)
    {x y : ℕ} (hx : x ∈ trace g n L) (hy : y ∈ trace g n L) : x % 2 = y % 2 := by
  obtain ⟨c, hc⟩ := trace_subset_domino (n := n) hg
  have hx' := hc hx
  have hy' := hc hy
  simp only [Finset.mem_insert, Finset.mem_singleton] at hx' hy'
  omega

/-- For a gear at least as big as the window, the trace is contained in the two
listed residues `off g n`, `off g (n + 2)` - the gear's two teeth. -/
theorem trace_subset_off {g : ℕ} (hg : 0 < g) {n : ℤ} {L : ℕ} (hL : L ≤ g) :
    trace g n L ⊆ ({off g n, off g (n + 2)} : Finset ℕ) := by
  intro c hc
  obtain ⟨hcL, hs⟩ := mem_trace.mp hc
  have hcg : c < g := lt_of_lt_of_le hcL hL
  simp only [Finset.mem_insert, Finset.mem_singleton]
  rcases hs with h | h
  · exact Or.inl (off_eq_of_dvd hg hcg h).symm
  · refine Or.inr (off_eq_of_dvd hg hcg ?_).symm
    rw [show (n + 2) + (c : ℤ) = n + (c : ℤ) + 2 by ring]
    exact h

/-- The two teeth of a gear are at distance `2`, or at distance `g - 2` the
other way round: the ONLY two separations a gear can show. -/
theorem off_pair_diff {g : ℕ} (hg : 2 ≤ g) (n : ℤ) :
    ((off g n : ℕ) : ℤ) - ((off g (n + 2) : ℕ) : ℤ) = 2 ∨
      ((off g n : ℕ) : ℤ) - ((off g (n + 2) : ℕ) : ℤ) = 2 - (g : ℤ) := by
  have hg0 : 0 < g := by omega
  have hgz : (0 : ℤ) < (g : ℤ) := by exact_mod_cast hg0
  have hg2 : (2 : ℤ) ≤ (g : ℤ) := by exact_mod_cast hg
  have ha : ((off g n : ℕ) : ℤ) < (g : ℤ) := by exact_mod_cast off_lt hg0 n
  have hb : ((off g (n + 2) : ℕ) : ℤ) < (g : ℤ) := by exact_mod_cast off_lt hg0 (n + 2)
  have ha0 : (0 : ℤ) ≤ ((off g n : ℕ) : ℤ) := Int.natCast_nonneg _
  have hb0 : (0 : ℤ) ≤ ((off g (n + 2) : ℕ) : ℤ) := Int.natCast_nonneg _
  have h1 := dvd_add_off hg0 n
  have h2 := dvd_add_off hg0 (n + 2)
  have hd : (g : ℤ) ∣ (((off g n : ℕ) : ℤ) - ((off g (n + 2) : ℕ) : ℤ) - 2) := by
    have h3 := dvd_sub h1 h2
    rwa [show n + ((off g n : ℕ) : ℤ) - (n + 2 + ((off g (n + 2) : ℕ) : ℤ))
        = ((off g n : ℕ) : ℤ) - ((off g (n + 2) : ℕ) : ℤ) - 2 by ring] at h3
  obtain ⟨k, hk⟩ := hd
  have hk1 : k ≤ 0 := by
    by_contra hc
    have hle : (g : ℤ) * 1 ≤ (g : ℤ) * k := by
      exact mul_le_mul_of_nonneg_left (by omega) (le_of_lt hgz)
    rw [mul_one] at hle
    linarith
  have hk2 : -1 ≤ k := by
    by_contra hc
    have hle : (g : ℤ) * k ≤ (g : ℤ) * (-2) := by
      exact mul_le_mul_of_nonneg_left (by omega) (le_of_lt hgz)
    rw [show (g : ℤ) * (-2) = -(2 * (g : ℤ)) by ring] at hle
    linarith
  have hk3 : k = -1 ∨ k = 0 := by omega
  rcases hk3 with rfl | rfl
  · right; omega
  · left; omega

/-- **L67, the ends-joining piece - the branch's form.**  For a gear at least as
big as the window, the gear's trace can contain two cells of DIFFERENT parity -
"join the two ends" - if and only if `g <= L + 1`.  Every other piece in the
machine lives inside one parity class (`trace_one_parity`). -/
theorem trace_crosses_parity_iff {g L : ℕ} (hodd : g % 2 = 1) (h5 : 5 ≤ g)
    (h4 : 4 ≤ L) (hgL : L ≤ g) :
    (∃ (n : ℤ) (x y : ℕ), x ∈ trace g n L ∧ y ∈ trace g n L ∧ x % 2 ≠ y % 2)
      ↔ g ≤ L + 1 := by
  constructor
  · rintro ⟨n, x, y, hx, hy, hxy⟩
    have hg0 : 0 < g := by omega
    have hsub := trace_subset_off (g := g) hg0 (n := n) (L := L) hgL
    have hx' := hsub hx
    have hy' := hsub hy
    simp only [Finset.mem_insert, Finset.mem_singleton] at hx' hy'
    have hxL := (mem_trace.mp hx).1
    have hyL := (mem_trace.mp hy).1
    have hdiff := off_pair_diff (g := g) (by omega) n
    -- the two cells must be the two distinct teeth
    have hxne : x ≠ y := by omega
    have hcase : (x = off g n ∧ y = off g (n + 2)) ∨ (x = off g (n + 2) ∧ y = off g n) := by
      rcases hx' with hx1 | hx1 <;> rcases hy' with hy1 | hy1
      · exact absurd (hx1.trans hy1.symm) hxne
      · exact Or.inl ⟨hx1, hy1⟩
      · exact Or.inr ⟨hx1, hy1⟩
      · exact absurd (hx1.trans hy1.symm) hxne
    have hxz : (x : ℤ) < (L : ℤ) := by exact_mod_cast hxL
    have hyz : (y : ℤ) < (L : ℤ) := by exact_mod_cast hyL
    rcases hcase with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> rcases hdiff with hd | hd <;> omega
  · intro hle
    refine ⟨0, 0, g - 2, ?_, ?_, by omega⟩
    · refine mem_trace.mpr ⟨by omega, Or.inl ?_⟩
      simp
    · refine mem_trace.mpr ⟨by omega, Or.inr ?_⟩
      have : (0 : ℤ) + ((g - 2 : ℕ) : ℤ) + 2 = (g : ℤ) := by
        have h2 : (2 : ℕ) ≤ g := by omega
        push_cast [Nat.cast_sub h2]
        ring
      rw [this]

/-- **L67, the end pair, sharp.**  For a gear at least as big as the window,
the gear strikes BOTH endpoints `0` and `L - 1` at some phase if and only if
`g = L + 1`.  (At `g = L` the gear's ends-joining piece is a WRAP pair
`{0, L - 2}` or `{1, L - 1}`, which is a parity crossing - `trace_crosses_parity_iff` -
but not the end pair.) -/
theorem ends_join_iff {g L : ℕ} (h4 : 4 ≤ L) (hgL : L ≤ g) :
    (∃ n : ℤ, (0 : ℕ) ∈ trace g n L ∧ (L - 1) ∈ trace g n L) ↔ g = L + 1 := by
  constructor
  · rintro ⟨n, h0, h1⟩
    have hg0 : 0 < g := by omega
    have hsub := trace_subset_off (g := g) hg0 (n := n) (L := L) hgL
    have h0' := hsub h0
    have h1' := hsub h1
    simp only [Finset.mem_insert, Finset.mem_singleton] at h0' h1'
    have hdiff := off_pair_diff (g := g) (by omega) n
    have hne : (0 : ℕ) ≠ L - 1 := by omega
    have hcase : ((0 : ℕ) = off g n ∧ L - 1 = off g (n + 2)) ∨
        ((0 : ℕ) = off g (n + 2) ∧ L - 1 = off g n) := by
      rcases h0' with hx1 | hx1 <;> rcases h1' with hy1 | hy1
      · exact absurd (hx1.trans hy1.symm) hne
      · exact Or.inl ⟨hx1, hy1⟩
      · exact Or.inr ⟨hx1, hy1⟩
      · exact absurd (hx1.trans hy1.symm) hne
    have hL1 : ((L - 1 : ℕ) : ℤ) = (L : ℤ) - 1 := by
      have : (1 : ℕ) ≤ L := by omega
      push_cast [Nat.cast_sub this]; ring
    rcases hcase with ⟨h2, h3⟩ | ⟨h2, h3⟩ <;>
      rw [← h2, ← h3] at hdiff <;> rw [hL1] at hdiff <;> omega
  · rintro rfl
    refine ⟨0, mem_trace.mpr ⟨by omega, Or.inl (by simp)⟩,
      mem_trace.mpr ⟨by omega, Or.inr ?_⟩⟩
    have h2 : (1 : ℕ) ≤ L := by omega
    have : (0 : ℤ) + ((L - 1 : ℕ) : ℤ) + 2 = ((L + 1 : ℕ) : ℤ) := by
      push_cast [Nat.cast_sub h2]; ring
    rw [this]

/-! ## 2. The matching lemma (L68)

A **piece** is a distance-2 domino `{x, x + 2}` (a single cell is a piece too,
being a subset of one).  The **domino cost** of a finite set is the least number
of pieces whose union contains it. -/

/-- The domino at `x`. -/
def Piece (x : ℕ) : Finset ℕ := {x, x + 2}

theorem mem_piece {x y : ℕ} : y ∈ Piece x ↔ y = x ∨ y = x + 2 := by
  simp [Piece]

/-- **Dominoes never cross parity.** -/
theorem piece_one_parity {x y : ℕ} (h : y ∈ Piece x) : y % 2 = x % 2 := by
  rcases mem_piece.mp h with rfl | rfl <;> omega

theorem card_piece_le (x : ℕ) : (Piece x).card ≤ 2 := by
  unfold Piece
  exact le_trans (Finset.card_insert_le _ _) (by simp)

/-- `S` is covered by `k` pieces. -/
def CoveredBy (S : Finset ℕ) (k : ℕ) : Prop :=
  ∃ P : Finset ℕ, P.card ≤ k ∧ S ⊆ P.biUnion Piece

theorem coveredBy_self (S : Finset ℕ) : CoveredBy S S.card :=
  ⟨S, le_rfl, fun x hx => Finset.mem_biUnion.mpr ⟨x, hx, by simp [Piece]⟩⟩

theorem coveredBy_mono {S : Finset ℕ} {k l : ℕ} (h : CoveredBy S k) (hkl : k ≤ l) :
    CoveredBy S l := by
  obtain ⟨P, h1, h2⟩ := h
  exact ⟨P, le_trans h1 hkl, h2⟩

/-- **The domino cost `D(S)`**: the least number of pieces covering `S`. -/
noncomputable def domCost (S : Finset ℕ) : ℕ := sInf {k | CoveredBy S k}

theorem domCost_spec (S : Finset ℕ) : CoveredBy S (domCost S) := by
  unfold domCost
  exact Nat.sInf_mem (s := {k | CoveredBy S k}) ⟨S.card, coveredBy_self S⟩

theorem domCost_le {S : Finset ℕ} {k : ℕ} (h : CoveredBy S k) : domCost S ≤ k := by
  unfold domCost
  exact Nat.sInf_le h

theorem domCost_mono {S T : Finset ℕ} (h : S ⊆ T) : domCost S ≤ domCost T := by
  obtain ⟨P, hc, hcov⟩ := domCost_spec T
  exact domCost_le ⟨P, hc, subset_trans h hcov⟩

theorem domCost_empty : domCost (∅ : Finset ℕ) = 0 :=
  Nat.le_zero.mp (domCost_le ⟨∅, by simp, by simp⟩)

/-- **L68, the lower bound.**  A piece covers at most two cells. -/
theorem card_le_two_mul_of_coveredBy {S : Finset ℕ} {k : ℕ} (h : CoveredBy S k) :
    S.card ≤ 2 * k := by
  obtain ⟨P, hc, hcov⟩ := h
  calc S.card ≤ (P.biUnion Piece).card := Finset.card_le_card hcov
    _ ≤ ∑ p ∈ P, (Piece p).card := Finset.card_biUnion_le
    _ ≤ ∑ _p ∈ P, 2 := Finset.sum_le_sum (fun p _ => card_piece_le p)
    _ = 2 * P.card := by rw [Finset.sum_const, smul_eq_mul, mul_comm]
    _ ≤ 2 * k := by omega

theorem card_le_two_mul_domCost (S : Finset ℕ) : S.card ≤ 2 * domCost S :=
  card_le_two_mul_of_coveredBy (domCost_spec S)

/-- **L68, the parity split.**  A set that splits over the two parity classes
costs the sum of the two costs - pieces never cross parity, so the two classes
are paid for by disjoint pools of pieces. -/
theorem domCost_union_parity {A B : Finset ℕ}
    (hA : ∀ x ∈ A, x % 2 = 0) (hB : ∀ x ∈ B, x % 2 = 1) :
    domCost (A ∪ B) = domCost A + domCost B := by
  classical
  refine le_antisymm ?_ ?_
  · obtain ⟨PA, hA1, hA2⟩ := domCost_spec A
    obtain ⟨PB, hB1, hB2⟩ := domCost_spec B
    refine domCost_le ⟨PA ∪ PB, le_trans (Finset.card_union_le _ _) (Nat.add_le_add hA1 hB1), ?_⟩
    intro x hx
    rcases Finset.mem_union.mp hx with h | h
    · obtain ⟨p, hp, hxp⟩ := Finset.mem_biUnion.mp (hA2 h)
      exact Finset.mem_biUnion.mpr ⟨p, Finset.mem_union_left _ hp, hxp⟩
    · obtain ⟨p, hp, hxp⟩ := Finset.mem_biUnion.mp (hB2 h)
      exact Finset.mem_biUnion.mpr ⟨p, Finset.mem_union_right _ hp, hxp⟩
  · obtain ⟨P, hc, hcov⟩ := domCost_spec (A ∪ B)
    have hAe : CoveredBy A (P.filter (fun p => p % 2 = 0)).card := by
      refine ⟨P.filter (fun p => p % 2 = 0), le_rfl, ?_⟩
      intro x hx
      obtain ⟨p, hp, hxp⟩ := Finset.mem_biUnion.mp (hcov (Finset.mem_union_left _ hx))
      refine Finset.mem_biUnion.mpr ⟨p, Finset.mem_filter.mpr ⟨hp, ?_⟩, hxp⟩
      have h1 := piece_one_parity hxp
      have h2 := hA x hx
      omega
    have hBo : CoveredBy B (P.filter (fun p => p % 2 = 1)).card := by
      refine ⟨P.filter (fun p => p % 2 = 1), le_rfl, ?_⟩
      intro x hx
      obtain ⟨p, hp, hxp⟩ := Finset.mem_biUnion.mp (hcov (Finset.mem_union_right _ hx))
      refine Finset.mem_biUnion.mpr ⟨p, Finset.mem_filter.mpr ⟨hp, ?_⟩, hxp⟩
      have h1 := piece_one_parity hxp
      have h2 := hB x hx
      omega
    have hdisj : Disjoint (P.filter (fun p => p % 2 = 0)) (P.filter (fun p => p % 2 = 1)) := by
      rw [Finset.disjoint_left]
      intro a h1 h2
      have := (Finset.mem_filter.mp h1).2
      have := (Finset.mem_filter.mp h2).2
      omega
    have hsum : (P.filter (fun p => p % 2 = 0)).card + (P.filter (fun p => p % 2 = 1)).card
        ≤ P.card := by
      rw [← Finset.card_union_of_disjoint hdisj]
      exact Finset.card_le_card (Finset.union_subset (Finset.filter_subset _ _)
        (Finset.filter_subset _ _))
    calc domCost A + domCost B
        ≤ (P.filter (fun p => p % 2 = 0)).card + (P.filter (fun p => p % 2 = 1)).card :=
          Nat.add_le_add (domCost_le hAe) (domCost_le hBo)
      _ ≤ P.card := hsum
      _ ≤ domCost (A ∪ B) := hc

/-- A step-2 run of `j` cells starting at `a`. -/
def run (a j : ℕ) : Finset ℕ := (Finset.range j).image (fun i => a + 2 * i)

theorem mem_run {a j x : ℕ} : x ∈ run a j ↔ ∃ i, i < j ∧ x = a + 2 * i := by
  simp [run, eq_comm]

theorem card_run (a j : ℕ) : (run a j).card = j := by
  unfold run
  rw [Finset.card_image_of_injective _ (fun x y h => by omega), Finset.card_range]

/-- **L68 on a run.**  A step-2 run of `j` cells costs exactly `ceil(j / 2)`:
the lower bound is the cardinality bound, the upper bound is the explicit
tiling by the pieces at `a, a + 4, a + 8, ...`. -/
theorem domCost_run (a j : ℕ) : domCost (run a j) = (j + 1) / 2 := by
  classical
  refine le_antisymm ?_ ?_
  · refine domCost_le ⟨(Finset.range ((j + 1) / 2)).image (fun i => a + 4 * i), ?_, ?_⟩
    · exact le_trans (Finset.card_image_le) (by rw [Finset.card_range])
    · intro x hx
      obtain ⟨t, ht, rfl⟩ := mem_run.mp hx
      refine Finset.mem_biUnion.mpr ⟨a + 4 * (t / 2), ?_, ?_⟩
      · exact Finset.mem_image.mpr ⟨t / 2, Finset.mem_range.mpr (by omega), rfl⟩
      · rw [mem_piece]; omega
  · have h := card_le_two_mul_domCost (run a j)
    rw [card_run] at h
    omega

/-- `[0, L)` is the even run and the odd run. -/
theorem range_eq_runs (L : ℕ) :
    Finset.range L = run 0 ((L + 1) / 2) ∪ run 1 (L / 2) := by
  ext x
  simp only [Finset.mem_range, Finset.mem_union, mem_run]
  constructor
  · intro hx
    by_cases h : x % 2 = 0
    · exact Or.inl ⟨x / 2, by omega, by omega⟩
    · exact Or.inr ⟨x / 2, by omega, by omega⟩
  · rintro (⟨i, hi, rfl⟩ | ⟨i, hi, rfl⟩) <;> omega

/-- **The empty-core cost in closed form** (branch P5):
`D([0, L)) = 2 floor(L/4) + min(L mod 4, 2)`. -/
theorem boundary_cost (L : ℕ) :
    domCost (Finset.range L) = 2 * (L / 4) + min (L % 4) 2 := by
  rw [range_eq_runs L, domCost_union_parity, domCost_run, domCost_run]
  · omega
  · intro x hx
    obtain ⟨i, _, rfl⟩ := mem_run.mp hx
    omega
  · intro x hx
    obtain ⟨i, _, rfl⟩ := mem_run.mp hx
    omega

/-- **The largest affordable length** (branch P6): `max { L : D(L) <= m } = 2m - (m mod 2)`. -/
theorem boundary_greatest (m : ℕ) :
    IsGreatest {L : ℕ | domCost (Finset.range L) ≤ m} (2 * m - m % 2) := by
  constructor
  · show domCost (Finset.range (2 * m - m % 2)) ≤ m
    rw [boundary_cost]
    omega
  · intro L hL
    have h : domCost (Finset.range L) ≤ m := hL
    rw [boundary_cost] at h
    omega

/-! ## 3. The loaded record rule (L69) -/

/-- The core gears at length `L`: those that can join the two ends. -/
def core (G : Finset ℕ) (L : ℕ) : Finset ℕ := G.filter (fun g => g ≤ L + 1)

/-- The tail gears at length `L`. -/
def tailG (G : Finset ℕ) (L : ℕ) : Finset ℕ := G.filter (fun g => L + 1 < g)

/-- The cells of `[0, L)` that the core gears leave uncovered at phase `n`. -/
def uncovered (C : Finset ℕ) (n : ℤ) (L : ℕ) : Finset ℕ :=
  (Finset.range L).filter (fun i => ∀ g ∈ C, ¬ Strikes g (n + (i : ℤ)))

/-- A window of length `L` is coverable: some phase makes every one of its `L`
consecutive pairs struck. -/
def Coverable (G : Finset ℕ) (L : ℕ) : Prop :=
  ∃ n : ℤ, ∀ i : ℕ, i < L → ¬ IsOpen G (n + (i : ℤ))

theorem exists_strikes_of_not_isOpen {G : Finset ℕ} {x : ℤ} (h : ¬ IsOpen G x) :
    ∃ g ∈ G, Strikes g x := by
  by_contra hc
  exact h fun g hg hs => hc ⟨g, hg, hs⟩

theorem coverable_mono {G : Finset ℕ} {L L' : ℕ} (hle : L' ≤ L) (h : Coverable G L) :
    Coverable G L' := by
  obtain ⟨n, hn⟩ := h
  exact ⟨n, fun i hi => hn i (lt_of_lt_of_le hi hle)⟩

/-- **L69, necessity.**  If some phase covers the window, then the core gears at
that same phase leave a set of domino cost at most the number of tail gears -
because each tail gear shows at most one clipped domino (`trace_subset_domino`).

**No hypothesis at all**: not coprimality, not oddness, not primality. -/
theorem cost_le_tail_of_coverable {G : Finset ℕ} {L : ℕ} {n : ℤ}
    (hstruck : ∀ i : ℕ, i < L → ¬ IsOpen G (n + (i : ℤ))) :
    domCost (uncovered (core G L) n L) ≤ (tailG G L).card := by
  classical
  have hpick : ∀ h : ℕ, ∃ c : ℕ, h ∈ tailG G L → trace h n L ⊆ ({c, c + 2} : Finset ℕ) := by
    intro h
    by_cases hh : h ∈ tailG G L
    · have hsize : L + 1 < h := (Finset.mem_filter.mp hh).2
      obtain ⟨c, hc⟩ := trace_subset_domino (g := h) (n := n) (L := L) hsize
      exact ⟨c, fun _ => hc⟩
    · exact ⟨0, fun h' => absurd h' hh⟩
  choose c hc using hpick
  refine domCost_le ⟨(tailG G L).image c, Finset.card_image_le, ?_⟩
  intro i hi
  have hi' := Finset.mem_filter.mp hi
  have hiL : i < L := Finset.mem_range.mp hi'.1
  obtain ⟨g, hgG, hgs⟩ := exists_strikes_of_not_isOpen (hstruck i hiL)
  have hgtail : g ∈ tailG G L := by
    refine Finset.mem_filter.mpr ⟨hgG, ?_⟩
    by_contra hle
    exact hi'.2 g (Finset.mem_filter.mpr ⟨hgG, by omega⟩) hgs
  have hmem : i ∈ trace g n L := mem_trace.mpr ⟨hiL, hgs⟩
  have := hc g hgtail hmem
  simp only [Finset.mem_insert, Finset.mem_singleton] at this
  refine Finset.mem_biUnion.mpr ⟨c g, Finset.mem_image.mpr ⟨g, hgtail, rfl⟩, ?_⟩
  rw [mem_piece]
  exact this

/-- **L69, sufficiency.**  If some phase of the core gears leaves an uncovered
set of domino cost at most the number of tail gears, the window is coverable:
place the core by CRT at that phase, and give one tail gear per needed piece the
residue that puts its own domino on that piece.

**The only hypothesis is pairwise coprimality** - no size hypothesis on the tail
gears is used, since a gear always strikes both cells of the domino its residue
names (as in `parity_attained`). -/
theorem coverable_of_cost_le_tail {G : Finset ℕ}
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) {L : ℕ} {n : ℤ}
    (hU : domCost (uncovered (core G L) n L) ≤ (tailG G L).card) :
    Coverable G L := by
  classical
  obtain ⟨P, hPcard, hPcov⟩ := domCost_spec (uncovered (core G L) n L)
  have hPt : P.card ≤ (tailG G L).card := le_trans hPcard hU
  obtain ⟨pAt, hpAt⟩ : ∃ f : ℕ → ℕ, ∀ (j : ℕ) (hj : j < P.card),
      f j = ((P.equivFin.symm ⟨j, hj⟩ : {x // x ∈ P}) : ℕ) := by
    have hchoice : ∀ j : ℕ, ∃ y : ℕ, ∀ hj : j < P.card,
        y = ((P.equivFin.symm ⟨j, hj⟩ : {x // x ∈ P}) : ℕ) := by
      intro j
      by_cases hj : j < P.card
      · exact ⟨((P.equivFin.symm ⟨j, hj⟩ : {x // x ∈ P}) : ℕ), fun _ => rfl⟩
      · exact ⟨0, fun h' => absurd h' hj⟩
    choose f hf using hchoice
    exact ⟨f, hf⟩
  have hpick : ∀ x : ℕ, ∃ z : ℤ,
      (x ∈ core G L → z = n) ∧
      (∀ hx : x ∈ tailG G L,
        z = -((pAt ((tailG G L).equivFin ⟨x, hx⟩ : Fin (tailG G L).card).val : ℕ) : ℤ) - 2) := by
    intro x
    by_cases hcore : x ∈ core G L
    · refine ⟨n, fun _ => rfl, fun ht => ?_⟩
      exfalso
      have h1 := (Finset.mem_filter.mp hcore).2
      have h2 := (Finset.mem_filter.mp ht).2
      omega
    · by_cases ht : x ∈ tailG G L
      · exact ⟨-((pAt ((tailG G L).equivFin ⟨x, ht⟩ : Fin (tailG G L).card).val : ℕ) : ℤ) - 2,
          fun h => absurd h hcore, fun _ => rfl⟩
      · exact ⟨0, fun h => absurd h hcore, fun h => absurd h ht⟩
  choose r hrcore hrtail using hpick
  obtain ⟨N, hN⟩ := exists_crt G hcop r
  have hAssign : ∀ p ∈ P, ∃ h ∈ tailG G L, (h : ℤ) ∣ N + (p : ℤ) + 2 := by
    intro p hp
    have hjP : ((P.equivFin ⟨p, hp⟩ : Fin P.card)).val < P.card := (P.equivFin ⟨p, hp⟩).isLt
    have hjT : ((P.equivFin ⟨p, hp⟩ : Fin P.card)).val < (tailG G L).card :=
      lt_of_lt_of_le hjP hPt
    refine ⟨(((tailG G L).equivFin.symm
        ⟨((P.equivFin ⟨p, hp⟩ : Fin P.card)).val, hjT⟩ : {x // x ∈ tailG G L}) : ℕ),
      ((tailG G L).equivFin.symm ⟨_, hjT⟩).2, ?_⟩
    set h := (((tailG G L).equivFin.symm
        ⟨((P.equivFin ⟨p, hp⟩ : Fin P.card)).val, hjT⟩ : {x // x ∈ tailG G L}) : ℕ) with hhdef
    have hmem : h ∈ tailG G L := ((tailG G L).equivFin.symm ⟨_, hjT⟩).2
    have hidx : (((tailG G L).equivFin ⟨h, hmem⟩ : Fin (tailG G L).card)).val
        = ((P.equivFin ⟨p, hp⟩ : Fin P.card)).val := by
      have h1 : (⟨h, hmem⟩ : {y // y ∈ tailG G L})
          = (tailG G L).equivFin.symm ⟨((P.equivFin ⟨p, hp⟩ : Fin P.card)).val, hjT⟩ := rfl
      rw [h1, Equiv.apply_symm_apply]
    have hpatv : pAt ((P.equivFin ⟨p, hp⟩ : Fin P.card)).val = p := by
      rw [hpAt _ hjP]
      have h2 : (⟨((P.equivFin ⟨p, hp⟩ : Fin P.card)).val, hjP⟩ : Fin P.card)
          = P.equivFin ⟨p, hp⟩ := rfl
      rw [h2, Equiv.symm_apply_apply]
    have hr := hrtail h hmem
    rw [hidx, hpatv] at hr
    have hdvd := hN h (Finset.mem_filter.mp hmem).1
    rw [hr] at hdvd
    rwa [show N - (-(p : ℤ) - 2) = N + (p : ℤ) + 2 by ring] at hdvd
  refine ⟨N, fun i hi hopen => ?_⟩
  by_cases hiU : i ∈ uncovered (core G L) n L
  · obtain ⟨p, hp, hip⟩ := Finset.mem_biUnion.mp (hPcov hiU)
    obtain ⟨h, hmemT, hdvd⟩ := hAssign p hp
    refine hopen h (Finset.mem_filter.mp hmemT).1 ?_
    rcases mem_piece.mp hip with rfl | rfl
    · exact Or.inr hdvd
    · refine Or.inl ?_
      rwa [show N + ((p + 2 : ℕ) : ℤ) = N + (p : ℤ) + 2 by push_cast; ring]
  · have hex : ∃ g ∈ core G L, Strikes g (n + (i : ℤ)) := by
      by_contra hcon
      exact hiU (Finset.mem_filter.mpr ⟨Finset.mem_range.mpr hi,
        fun g hg hs => hcon ⟨g, hg, hs⟩⟩)
    obtain ⟨g, hg, hs⟩ := hex
    have hgG : g ∈ G := (Finset.mem_filter.mp hg).1
    refine hopen g hgG ?_
    have hdvd := hN g hgG
    rw [hrcore g hg] at hdvd
    exact (strikes_congr (by rwa [show N + (i : ℤ) - (n + (i : ℤ)) = N - n by ring])).mpr hs

/-- **L69, THE LOADED RECORD RULE.**  A window of length `L` is coverable iff
some phasing of the core gears leaves an uncovered set of domino cost at most
the number of tail gears.  The only hypothesis is pairwise coprimality (used by
the sufficiency half alone). -/
theorem loaded_record_rule {G : Finset ℕ}
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) (L : ℕ) :
    Coverable G L ↔
      ∃ n : ℤ, domCost (uncovered (core G L) n L) ≤ (tailG G L).card := by
  constructor
  · rintro ⟨n, hn⟩
    exact ⟨n, cost_le_tail_of_coverable hn⟩
  · rintro ⟨n, hn⟩
    exact coverable_of_cost_le_tail hcop hn

/-- The record set and the cost set are the SAME set of lengths. -/
theorem record_set_eq {G : Finset ℕ}
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) :
    {L : ℕ | ∃ n : ℤ, ∀ i : ℕ, i < L → ¬ IsOpen G (n + (i : ℤ))}
      = {L : ℕ | ∃ n : ℤ, domCost (uncovered (core G L) n L) ≤ (tailG G L).card} :=
  Set.ext fun L => loaded_record_rule hcop L

/-- **`F_top` in the rule's form.**  A length is the record iff it is the
greatest length whose minimal core cost is affordable. -/
theorem record_isGreatest_iff {G : Finset ℕ}
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) (F : ℕ) :
    IsGreatest {L : ℕ | ∃ n : ℤ, ∀ i : ℕ, i < L → ¬ IsOpen G (n + (i : ℤ))} F ↔
      IsGreatest {L : ℕ | ∃ n : ℤ,
        domCost (uncovered (core G L) n L) ≤ (tailG G L).card} F := by
  rw [record_set_eq hcop]

/-! ## 4. The boundary corollary: the parity law from the rule -/

theorem core_eq_empty {G : Finset ℕ} {L : ℕ} (h : ∀ g ∈ G, L + 1 < g) :
    core G L = ∅ := by
  refine Finset.filter_eq_empty_iff.mpr ?_
  intro g hg
  have := h g hg
  omega

theorem tailG_eq_self {G : Finset ℕ} {L : ℕ} (h : ∀ g ∈ G, L + 1 < g) :
    tailG G L = G := by
  refine Finset.filter_true_of_mem ?_
  exact fun g hg => h g hg

theorem uncovered_empty (n : ℤ) (L : ℕ) : uncovered ∅ n L = Finset.range L := by
  simp [uncovered]

/-- **The parity law, PROVED FROM THE RULE.**  For gears odd, pairwise coprime
and bigger than `2m + 1`, the longest run of consecutive struck pairs is exactly
`2m - (m mod 2)`.  This is an independent proof of `parity_law` (round 33): the
lower bound is the rule's sufficiency with an empty core and the explicit run
tiling, the upper bound is the rule's necessity plus the closed-form cost - no
use of `parity_upper` or `parity_attained`. -/
theorem parity_law_of_rule {G : Finset ℕ} (hodd : ∀ g ∈ G, g % 2 = 1)
    (hbig : ∀ g ∈ G, 2 * G.card + 1 < g)
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) :
    IsGreatest {L : ℕ | ∃ n : ℤ, ∀ i : ℕ, i < L → ¬ IsOpen G (n + (i : ℤ))}
      (2 * G.card - G.card % 2) := by
  classical
  set m := G.card with hm
  set L₀ := 2 * m - m % 2 with hL₀
  have hcore₀ : ∀ g ∈ G, L₀ + 1 < g := by
    intro g hg
    have := hbig g hg
    omega
  constructor
  · -- membership: an empty core, and the whole window costs exactly `m`
    have hcost : domCost (uncovered (core G L₀) 0 L₀) ≤ (tailG G L₀).card := by
      rw [core_eq_empty hcore₀, uncovered_empty, tailG_eq_self hcore₀, boundary_cost]
      omega
    exact coverable_of_cost_le_tail hcop hcost
  · rintro L ⟨N, hN⟩
    by_contra hlt
    -- the record would reach `L₀ + 1`
    obtain ⟨N', hN'⟩ : Coverable G (L₀ + 1) :=
      coverable_mono (by omega) (⟨N, hN⟩ : Coverable G L)
    have hcore₁ : ∀ g ∈ G, L₀ + 1 + 1 < g := by
      intro g hg
      have h1 := hbig g hg
      have h2 := hodd g hg
      omega
    have hcost := cost_le_tail_of_coverable (G := G) (L := L₀ + 1) (n := N') hN'
    rw [core_eq_empty hcore₁, uncovered_empty, tailG_eq_self hcore₁, boundary_cost] at hcost
    omega

/-- The two proofs agree: the record from the rule is the parity law's value. -/
theorem parity_law_agrees {G : Finset ℕ} (hodd : ∀ g ∈ G, g % 2 = 1)
    (hbig : ∀ g ∈ G, 2 * G.card + 1 < g)
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) {F : ℕ}
    (hF : IsGreatest {L : ℕ | ∃ n : ℤ, ∀ i : ℕ, i < L → ¬ IsOpen G (n + (i : ℤ))} F) :
    F = 2 * G.card - G.card % 2 :=
  hF.unique (parity_law_of_rule hodd hbig hcop)

/-- **The rule's own maximisation problem has the parity law's answer.**  For a
free wheel, `max { L : min over core phasings of D(U) <= t(L) } = 2m - (m mod 2)`. -/
theorem rule_gives_parity_law {G : Finset ℕ} (hodd : ∀ g ∈ G, g % 2 = 1)
    (hbig : ∀ g ∈ G, 2 * G.card + 1 < g)
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) :
    IsGreatest {L : ℕ | ∃ n : ℤ,
      domCost (uncovered (core G L) n L) ≤ (tailG G L).card}
      (2 * G.card - G.card % 2) :=
  (record_isGreatest_iff hcop _).mp (parity_law_of_rule hodd hbig hcop)

end TopMachine
