/-
The anchor-2,3,5 layer laws, in general form.

`docs/proof-search/anchor-235.md` sections 9d/9f describe how one gear `g`
sits on top of the machine `M = {5..g-}`:

* HIT LAW.  `g` hops at the lower landing `x` iff `x = +-u` (mod g),
  `u = 6^{-1} mod g` - the two teeth.
* CHAIN LAW.  two lower openings `x < y` are both hopped iff
  `y - x = 0` or `+-d` (mod g), `d = 2u`.
* PHASE FORM.  the full period is `g` copies of the lower period, copy `j`
  shifted by `j * P_M`; since `P_M` is invertible mod `g` the copies realise
  every deletion phase `r` in `Z_g` exactly once, copy `j` deleting the lower
  openings whose class lies in the TWO-CLASS SET `{r, r + d}`.
* NESTED FORM (9f).  `W_g(s) = W_M(s) + h1 (1 + W_M(x+1)) + h1 h2 (...) + ...`
  - after a hit the walk re-enters the lower machine at `x + 1`.

This file states all four machine-free: the residue half over `ZMod g` (no
`decide` anywhere - the proofs are algebra, uniform in `g`), and the walk half
over an abstract pair of predicates, so the `hop` identity that the nested
formula is built from is proved once for every layer of every machine.

WHAT IS NOT HERE, and it is the honest boundary.  The chain DEPTH `D_g` (the
run length that caps the nesting) is NOT an algebraic consequence: a run in a
two-class set alternates freely between the two classes, so nothing in the
residue arithmetic bounds its length.  `D_g` is a fact about the lower gap
SIZES (`anchor-235.md` 9d: the admissible gaps are the elements of
`{d, g-d, g, ...}` below `F_M + 1`), and that is a per-machine measurement.
-/

import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Nat.Find
import Mathlib.Tactic.Ring
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.NormNum

namespace AnchorChain

/-! ## 1. The residue half, over `ZMod g` -/

section Residues

variable {g : ℕ}

/-- A slot is a HIT of the gear with tooth `u` iff its class is `u` or `-u`. -/
def OnTeeth (u x : ZMod g) : Prop := x = u ∨ x = -u

/-- A slot is DELETED AT PHASE `r` (spacing `d`) iff its class lies in the
two-class set `{r, r + d}`. -/
def DeletedAt (d r x : ZMod g) : Prop := x = r ∨ x = r + d

instance (u x : ZMod g) [DecidableEq (ZMod g)] : Decidable (OnTeeth u x) := by
  unfold OnTeeth; infer_instance

/-- **The teeth ARE a two-class set**: with `d = 2u` the hits of the gear are
exactly the slots deleted at phase `-u`. -/
theorem teeth_eq_phase (u x : ZMod g) : OnTeeth u x ↔ DeletedAt (2 * u) (-u) x := by
  constructor
  · rintro (h | h)
    · exact Or.inr (by rw [h]; ring)
    · exact Or.inl h
  · rintro (h | h)
    · exact Or.inr h
    · exact Or.inl (by rw [h]; ring)

/-- **THE CHAIN LAW.**  Two slots lie in a COMMON two-class set `{r, r + d}`
exactly when their difference is `0`, `d` or `-d`.  Machine-free, every `g`. -/
theorem chain_law (d x y : ZMod g) :
    (∃ r, DeletedAt d r x ∧ DeletedAt d r y) ↔ (y - x = 0 ∨ y - x = d ∨ y - x = -d) := by
  constructor
  · rintro ⟨r, hx | hx, hy | hy⟩ <;> subst hx <;> subst hy
    · exact Or.inl (by ring)
    · exact Or.inr (Or.inl (by ring))
    · exact Or.inr (Or.inr (by ring))
    · exact Or.inl (by ring)
  · rintro (h | h | h)
    · exact ⟨x, Or.inl rfl, Or.inl (by linear_combination h)⟩
    · exact ⟨x, Or.inl rfl, Or.inr (by linear_combination h)⟩
    · exact ⟨y, Or.inr (by linear_combination -h), Or.inl rfl⟩

/-- **The copy-`j` phase.**  In the copy shifted by `j * P`, the hits of the
gear are exactly the lower slots deleted at phase `-u - j * P`. -/
theorem copy_phase (u P j x : ZMod g) :
    OnTeeth u (x + j * P) ↔ DeletedAt (2 * u) (-u - j * P) x := by
  constructor
  · rintro (h | h)
    · exact Or.inr (by linear_combination h)
    · exact Or.inl (by linear_combination h)
  · rintro (h | h)
    · exact Or.inr (by linear_combination h)
    · exact Or.inl (by linear_combination h)

/-- **Every phase exactly once.**  When the lower period `P` is invertible mod
`g`, `j ↦ -u - j * P` is a bijection of `ZMod g`: the `g` copies realise every
deletion phase exactly once. -/
theorem phase_bijective (u P : ZMod g) (hP : IsUnit P) :
    Function.Bijective (fun j : ZMod g => -u - j * P) := by
  obtain ⟨v, hv⟩ := hP.exists_right_inv
  refine ⟨fun a b hab => ?_, fun r => ⟨(-u - r) * v, ?_⟩⟩
  · simp only at hab
    have h : a * P = b * P := by linear_combination -hab
    calc a = a * P * v := by rw [mul_assoc, hv, mul_one]
      _ = b * P * v := by rw [h]
      _ = b := by rw [mul_assoc, hv, mul_one]
  · simp only
    calc -u - (-u - r) * v * P = -u - (-u - r) * (P * v) := by ring
      _ = -u - (-u - r) := by rw [hv, mul_one]
      _ = r := by ring

/-- **T3 ALTERNATION, the algebraic half.**  Three slots in one two-class set
cannot be reached by two `+d` steps unless `2d = 0`. -/
theorem no_two_up {d r x y z : ZMod g}
    (_hx : DeletedAt d r x) (_hy : DeletedAt d r y) (hz : DeletedAt d r z)
    (h1 : y - x = d) (h2 : z - y = d) (hx0 : x = r) : 2 * d = 0 := by
  have hy' : y = r + d := by linear_combination h1 + hx0
  have hz' : z = r + 2 * d := by linear_combination h2 + hy'
  rcases hz with h | h
  · linear_combination h - hz'
  · linear_combination 2 * h - 2 * hz'

/-- The same with two `-d` steps (the mirror statement). -/
theorem no_two_down {d r x y z : ZMod g}
    (hx : DeletedAt d r x) (_hy : DeletedAt d r y) (_hz : DeletedAt d r z)
    (h1 : y - x = -d) (h2 : z - y = -d) (hz0 : z = r) : 2 * d = 0 := by
  have hy' : y = r + d := by linear_combination -h2 + hz0
  have hx' : x = r + 2 * d := by linear_combination -h1 + hy'
  rcases hx with h | h
  · linear_combination h - hx'
  · linear_combination 2 * h - 2 * hx'

/-- **THE NEIGHBOUR-OF-HIT IDENTITY** (`anchor-235.md` 9e/9f): the neighbour of
a hit is never a hit, for EVERY gear `g >= 5`, because the two teeth are
`d = 2u = 3^{-1}` apart and `3^{-1} = +-1` would force `g | 2` or `g | 4`.
This is what makes the nested formula's `x + 1` restart legitimate. -/
theorem neighbour_of_hit {u : ZMod g} (h6 : (6 : ZMod g) * u = 1)
    (hg : 5 ≤ g) {x : ZMod g} (hx : OnTeeth u x) : ¬ OnTeeth u (x + 1) := by
  intro hx1
  have hcast : ∀ n : ℕ, (n : ZMod g) = 0 → g ∣ n := fun n hn =>
    (ZMod.natCast_eq_zero_iff n g).mp hn
  rw [teeth_eq_phase] at hx hx1
  have h : (x + 1) - x = 0 ∨ (x + 1) - x = 2 * u ∨ (x + 1) - x = -(2 * u) :=
    (chain_law (2 * u) x (x + 1)).mp ⟨-u, hx, hx1⟩
  have e : (x + 1) - x = (1 : ZMod g) := by ring
  rw [e] at h
  rcases h with h | h | h
  · have hd : g ∣ 1 := hcast 1 (by simpa using h)
    have := Nat.le_of_dvd (by norm_num) hd
    omega
  · -- 1 = 2u, so 3 = 6u = 1, so 2 = 0
    have h3 : (3 : ZMod g) = 1 := by linear_combination (3 : ZMod g) * h + h6
    have h2 : ((2 : ℕ) : ZMod g) = 0 := by push_cast; linear_combination h3
    have := Nat.le_of_dvd (by norm_num) (hcast 2 h2)
    omega
  · -- 1 = -2u, so -3 = 6u = 1, so 4 = 0
    have h3 : (-3 : ZMod g) = 1 := by linear_combination (-3 : ZMod g) * h + h6
    have h4 : ((4 : ℕ) : ZMod g) = 0 := by push_cast; linear_combination -h3
    have := Nat.le_of_dvd (by norm_num) (hcast 4 h4)
    omega

end Residues

/-! ## 2. The walk half: the nested formula's hop step, abstractly

`M` is the lower machine's opening predicate and `H` the new gear's hit
predicate.  `nextM x` is the next lower opening strictly above `x`, `nextG x`
the next opening of the enlarged machine.  The nested formula of 9f is the
statement that `nextG` is `nextM` iterated until a non-hit is reached. -/

section Walk

variable (M H : ℕ → Prop) [DecidablePred M] [DecidablePred H]

/-- Cofinality of the lower openings. -/
def Unbounded : Prop := ∀ x, ∃ y, x < y ∧ M y

/-- Cofinality of the enlarged machine's openings. -/
def UnboundedG : Prop := ∀ x, ∃ y, x < y ∧ M y ∧ ¬ H y

variable {M H}

/-- The next lower opening strictly above `x`. -/
def nextM (hM : Unbounded M) (x : ℕ) : ℕ :=
  Nat.find (hM x)

/-- The next opening of the enlarged machine strictly above `x`. -/
def nextG (hG : UnboundedG M H) (x : ℕ) : ℕ :=
  Nat.find (hG x)

theorem nextM_spec (hM : Unbounded M) (x : ℕ) : x < nextM hM x ∧ M (nextM hM x) :=
  Nat.find_spec (hM x)

theorem nextM_min (hM : Unbounded M) {x y : ℕ} (h1 : x < y) (h2 : M y) :
    nextM hM x ≤ y :=
  Nat.find_le ⟨h1, h2⟩

theorem nextG_spec (hG : UnboundedG M H) (x : ℕ) :
    x < nextG hG x ∧ M (nextG hG x) ∧ ¬ H (nextG hG x) :=
  Nat.find_spec (hG x)

theorem nextG_min (hG : UnboundedG M H) {x y : ℕ} (h1 : x < y) (h2 : M y)
    (h3 : ¬ H y) : nextG hG x ≤ y :=
  Nat.find_le ⟨h1, h2, h3⟩

/-- Every enlarged opening is a lower opening above `x`, so `nextM <= nextG`. -/
theorem nextM_le_nextG (hM : Unbounded M) (hG : UnboundedG M H) (x : ℕ) :
    nextM hM x ≤ nextG hG x :=
  nextM_min hM (nextG_spec hG x).1 (nextG_spec hG x).2.1

/-- **HOP 0** - no hit at the landing: the layer does not move the walk. -/
theorem hop_zero (hM : Unbounded M) (hG : UnboundedG M H) {x : ℕ}
    (h : ¬ H (nextM hM x)) : nextG hG x = nextM hM x := by
  refine le_antisymm (nextG_min hG (nextM_spec hM x).1 (nextM_spec hM x).2 h) ?_
  exact nextM_le_nextG hM hG x

/-- The iterate is strictly above the start. -/
theorem lt_iterate (hM : Unbounded M) (x : ℕ) : ∀ k, x < (nextM hM)^[k + 1] x := by
  intro k
  induction k with
  | zero => simpa using (nextM_spec hM x).1
  | succ n ih =>
      rw [Function.iterate_succ_apply']
      exact lt_trans ih (nextM_spec hM _).1

theorem M_iterate (hM : Unbounded M) (x : ℕ) : ∀ k, M ((nextM hM)^[k + 1] x) := by
  intro k
  rw [Function.iterate_succ_apply']
  exact (nextM_spec hM _).2

omit [DecidablePred H] in
/-- Any enlarged opening above `x` is at or past the `(k+1)`-st lower opening,
when the first `k` lower openings above `x` are all hits. -/
theorem iterate_le_of_hits (hM : Unbounded M) {x y : ℕ} (hy : M y) (hxy : x < y)
    (hny : ¬ H y) :
    ∀ k, (∀ i, i < k → H ((nextM hM)^[i + 1] x)) → (nextM hM)^[k + 1] x ≤ y := by
  intro k
  induction k with
  | zero => intro _; simpa using nextM_min hM hxy hy
  | succ n ih =>
      intro hall
      have hprev : (nextM hM)^[n + 1] x ≤ y := ih (fun i hi => hall i (by omega))
      have hhit : H ((nextM hM)^[n + 1] x) := hall n (by omega)
      have hne : (nextM hM)^[n + 1] x ≠ y := by
        intro h; exact hny (h ▸ hhit)
      rw [Function.iterate_succ_apply']
      exact nextM_min hM (lt_of_le_of_ne hprev hne) hy

/-- **THE NESTED-FORMULA HOP STEP** (`anchor-235.md` 9f).  If the first `k`
lower openings above `x` are hits of the new gear and the `(k+1)`-st is not,
the enlarged machine's next opening is exactly that `(k+1)`-st lower opening:
`nextG = nextM` iterated past the hit run.  `k = 0` is `hop_zero`, `k = 1` and
`k = 2` are the two- and three-term nested formulas, and `k <= D_g` is the
chain-depth cap. -/
theorem hop_iter (hM : Unbounded M) (hG : UnboundedG M H) {x k : ℕ}
    (hrun : ∀ i, i < k → H ((nextM hM)^[i + 1] x))
    (hstop : ¬ H ((nextM hM)^[k + 1] x)) :
    nextG hG x = (nextM hM)^[k + 1] x := by
  refine le_antisymm (nextG_min hG (lt_iterate hM x k) (M_iterate hM x k) hstop) ?_
  exact iterate_le_of_hits hM (nextG_spec hG x).2.1 (nextG_spec hG x).1
    (nextG_spec hG x).2.2 k hrun

/-- The two-term form used in `nested_form.py`: one hit, then a clear landing. -/
theorem hop_one (hM : Unbounded M) (hG : UnboundedG M H) {x : ℕ}
    (h1 : H (nextM hM x)) (h2 : ¬ H (nextM hM (nextM hM x))) :
    nextG hG x = nextM hM (nextM hM x) := by
  have := hop_iter hM hG (k := 1) (x := x)
    (fun i hi => by interval_cases i; simpa using h1) (by simpa using h2)
  simpa using this
