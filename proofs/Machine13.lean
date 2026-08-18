/-
The alpha1 = 1 certificate at machine 13 - the first machine-checked
instance of tolerance lemma 1.

Machine 13 is the gear set {5, 7, 11, 13}; its openings are the slots no
gear marks, a residue condition of period 5005. The constructor's
per-machine statement (constructor.md sec 21, tool
research/strata_adjacency.py; alpha1 = 1, q' = 17):

    F2_k(13) <= F_k(13) + alpha1 * q'/3,   F_k = 11, F2_k = 16, budget 16.67.

Formalised here, per-instance and kernel-checked:

* `gap_le` / `gap11_realized`   - every machine gap is <= 11, and 11 occurs:
  `F_k(13) = 11`.
* `pair_sum_le` / `pair16_realized` - adjacent gaps span <= 16, and 16
  occurs: `F2_k(13) = 16`.
* `alpha1_certificate` - budget form `3*(c-a) <= 3*11 + 1*17`;
  `lemma1_at_13` - lemma-1 form `(c-a) - 11 <= 1*17`.

The two period facts are decided over the CRT tuple `(k%5, k%7, k%11, k%13)`
rather than over residues mod 5005: the same 5005 cases, but every modulus
is a single digit and the decision tree has depth <= 13 instead of 5005.
That is what makes the scan kernel-feasible at all - a direct mod-5005 scan
does not terminate in practice (see docs/proof-search/formalist.md).

The scan subsumes certificate tiers B and C: at fixed y the strata census is
itself a one-period fact. Tier A - the machine-free A3 law - is formalised
separately (`tierA_forbidden`, `tierA_kills`) because it is the piece that
survives at machines whose period is beyond kernel reach.

Verified against research/strata_adjacency.py before formalising: the
residue predicate matches the tool's exposed array on all 5005 residues,
F = 11, F2 = 16, and the witnesses 122 (gap 11) and 117 (pair 5+11) come
from its period scan.
-/

import Corridor

namespace Machine13

/-! ## The decidable core, on CRT tuples -/

/-- The opening test on a CRT tuple `(a,b,c,d) = (k%5, k%7, k%11, k%13)`:
gear `q` marks the slot when `k` hits the two blocked classes mod `q`. -/
def expT (a b c d : Nat) : Bool :=
  a != 1 && a != 4 && b != 6 && b != 1 && c != 2 && c != 9 && d != 11 && d != 2

/-- The test `n` slots further on. -/
def atT (a b c d n : Nat) : Bool :=
  expT ((a+n)%5) ((b+n)%7) ((c+n)%11) ((d+n)%13)

/-- Some slot among the next 11 is an opening. -/
def win11T (a b c d : Nat) : Bool :=
  (List.range 11).any fun i => atT a b c d (i+1)

/-- Slot `i+1` is the FIRST opening after the current one. -/
def firstAt (a b c d i : Nat) : Bool :=
  atT a b c d (i+1) && ((List.range i).all fun j => !(atT a b c d (j+1)))

/-- From an opening, once the first following opening is at `i+1`, a second
one arrives no later than slot 16. -/
def pairT (a b c d : Nat) : Bool :=
  !(atT a b c d 0) ||
    ((List.range 11).all fun i =>
      !(firstAt a b c d i) || ((List.range (15-i)).any fun j => atT a b c d (i+2+j)))

set_option maxRecDepth 10000 in
/-- **One period, `F_k(13) <= 11`.** Every 11-slot window holds an opening. -/
theorem w11 : ∀ a < 5, ∀ b < 7, ∀ c < 11, ∀ d < 13, win11T a b c d = true := by
  decide

set_option maxRecDepth 10000 in
/-- **One period, `F2_k(13) <= 16`.** From any opening, two more openings
arrive within 16 slots. -/
theorem w16 : ∀ a < 5, ∀ b < 7, ∀ c < 11, ∀ d < 13, pairT a b c d = true := by
  decide

/-! ## Openings -/

/-- An opening of machine 13: no gear in `{5,7,11,13}` divides either member
of slot `k`. -/
def Exposed13 (k : ℕ) : Prop :=
  ¬ (5 ∣ Census.lo k) ∧ ¬ (5 ∣ Census.hi k) ∧
    ¬ (7 ∣ Census.lo k) ∧ ¬ (7 ∣ Census.hi k) ∧
    ¬ (11 ∣ Census.lo k) ∧ ¬ (11 ∣ Census.hi k) ∧
    ¬ (13 ∣ Census.lo k) ∧ ¬ (13 ∣ Census.hi k)

instance (k : ℕ) : Decidable (Exposed13 k) := by unfold Exposed13; infer_instance

set_option maxHeartbeats 1000000 in
/-- Openings are exactly the CRT-tuple test. -/
theorem exposed13_iff {k : ℕ} (hk : 1 ≤ k) :
    Exposed13 k ↔ expT (k % 5) (k % 7) (k % 11) (k % 13) = true := by
  have h5lo : (5 ∣ Census.lo k) ↔ k % 5 = 1 := by simp only [Census.lo]; omega
  have h5hi : (5 ∣ Census.hi k) ↔ k % 5 = 4 := by simp only [Census.hi]; omega
  have h7lo : (7 ∣ Census.lo k) ↔ k % 7 = 6 := by simp only [Census.lo]; omega
  have h7hi : (7 ∣ Census.hi k) ↔ k % 7 = 1 := by simp only [Census.hi]; omega
  have h11lo : (11 ∣ Census.lo k) ↔ k % 11 = 2 := by simp only [Census.lo]; omega
  have h11hi : (11 ∣ Census.hi k) ↔ k % 11 = 9 := by simp only [Census.hi]; omega
  have h13lo : (13 ∣ Census.lo k) ↔ k % 13 = 11 := by simp only [Census.lo]; omega
  have h13hi : (13 ∣ Census.hi k) ↔ k % 13 = 2 := by simp only [Census.hi]; omega
  unfold Exposed13
  rw [h5lo, h5hi, h7lo, h7hi, h11lo, h11hi, h13lo, h13hi]
  simp only [expT, Bool.and_eq_true, bne_iff_ne, ne_eq, and_assoc]

/-- Shifted form: the tuple test `n` slots on is the opening test at `k+n`. -/
theorem atT_iff {k : ℕ} (hk : 1 ≤ k) (n : ℕ) :
    atT (k % 5) (k % 7) (k % 11) (k % 13) n = true ↔ Exposed13 (k + n) := by
  rw [atT, exposed13_iff (show 1 ≤ k + n by omega)]
  have e5 : (k % 5 + n) % 5 = (k + n) % 5 := by omega
  have e7 : (k % 7 + n) % 7 = (k + n) % 7 := by omega
  have e11 : (k % 11 + n) % 11 = (k + n) % 11 := by omega
  have e13 : (k % 13 + n) % 13 = (k + n) % 13 := by omega
  rw [e5, e7, e11, e13]

/-- Machine-13 openings are (5,7)-corridor openings - the bridge to every
machine-free corridor law. -/
theorem exposed13_exposed {k : ℕ} (h : Exposed13 k) : Corridor.Exposed k :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1⟩

/-! ## The gap bounds -/

/-- **`F_k(13) <= 11`.** A machine gap - consecutive openings with nothing
exposed between - has length at most 11. -/
theorem gap_le {a b : ℕ} (ha : 1 ≤ a) (hab : a < b)
    (_hEa : Exposed13 a) (_hEb : Exposed13 b)
    (hg : ∀ j, a < j → j < b → ¬ Exposed13 j) : b - a ≤ 11 := by
  by_contra hlt
  have h := w11 (a % 5) (by omega) (a % 7) (by omega) (a % 11) (by omega)
    (a % 13) (by omega)
  rw [win11T, List.any_eq_true] at h
  obtain ⟨i, hi, hv⟩ := h
  have hi11 := List.mem_range.mp hi
  exact hg (a + (i+1)) (by omega) (by omega) ((atT_iff ha _).mp hv)

/-- **`F2_k(13) <= 16`.** Two adjacent machine gaps span at most 16 slots. -/
theorem pair_sum_le {a b c : ℕ} (ha : 1 ≤ a) (hab : a < b) (_hbc : b < c)
    (hEa : Exposed13 a) (hEb : Exposed13 b) (_hEc : Exposed13 c)
    (hg1 : ∀ j, a < j → j < b → ¬ Exposed13 j)
    (hg2 : ∀ j, b < j → j < c → ¬ Exposed13 j) : c - a ≤ 16 := by
  have hs := gap_le ha hab hEa hEb hg1
  have ha0 : atT (a % 5) (a % 7) (a % 11) (a % 13) 0 = true :=
    (atT_iff ha 0).mpr (by simpa using hEa)
  have h := w16 (a % 5) (by omega) (a % 7) (by omega) (a % 11) (by omega)
    (a % 13) (by omega)
  rw [pairT, ha0] at h
  simp only [Bool.not_true, Bool.false_or] at h
  rw [List.all_eq_true] at h
  have hi11 : b - a - 1 < 11 := by omega
  have hstep := h (b - a - 1) (List.mem_range.mpr hi11)
  have hfirst : firstAt (a % 5) (a % 7) (a % 11) (a % 13) (b - a - 1) = true := by
    rw [firstAt, Bool.and_eq_true]
    refine ⟨(atT_iff ha _).mpr ?_, ?_⟩
    · rw [show a + (b - a - 1 + 1) = b by omega]; exact hEb
    · rw [List.all_eq_true]
      intro j hj
      have hjlt := List.mem_range.mp hj
      have hnot : ¬ Exposed13 (a + (j+1)) := hg1 _ (by omega) (by omega)
      cases hb : atT (a % 5) (a % 7) (a % 11) (a % 13) (j+1) with
      | false => rfl
      | true => exact absurd ((atT_iff ha _).mp hb) hnot
  rw [hfirst] at hstep
  simp only [Bool.not_true, Bool.false_or] at hstep
  rw [List.any_eq_true] at hstep
  obtain ⟨j, hj, hv⟩ := hstep
  have hjlt := List.mem_range.mp hj
  have hE := (atT_iff ha _).mp hv
  by_contra hcon
  exact hg2 _ (by omega) (by omega) hE

/-! ## Sharpness: the bounds are attained -/

/-- The gap 11 is realized at openings 122, 133: `F_k(13) = 11`. -/
theorem gap11_realized :
    Exposed13 122 ∧ Exposed13 133 ∧ ∀ j, 122 < j → j < 133 → ¬ Exposed13 j := by
  refine ⟨(exposed13_iff (by omega)).mpr (by decide),
    (exposed13_iff (by omega)).mpr (by decide), ?_⟩
  intro j h1 h2
  rw [exposed13_iff (by omega)]
  interval_cases j <;> decide

/-- The pair sum 16 is realized at openings 117, 122, 133 - adjacent gaps
`(5, 11)`. So `F2_k(13) = 16` and the alpha1 = 1 budget 16.67 is tight. -/
theorem pair16_realized :
    Exposed13 117 ∧ Exposed13 122 ∧ Exposed13 133 ∧
      (∀ j, 117 < j → j < 122 → ¬ Exposed13 j) ∧
      (∀ j, 122 < j → j < 133 → ¬ Exposed13 j) := by
  refine ⟨(exposed13_iff (by omega)).mpr (by decide),
    (exposed13_iff (by omega)).mpr (by decide),
    (exposed13_iff (by omega)).mpr (by decide), ?_, gap11_realized.2.2⟩
  intro j h1 h2
  rw [exposed13_iff (by omega)]
  interval_cases j <;> decide

/-! ## The headline forms -/

/-- **The alpha1 = 1 certificate at machine 13**, in the constructor's
budget form `3*F2_k <= 3*F_k + alpha1*q'` with `F_k = 11`, `q' = 17`. -/
theorem alpha1_certificate {a b c : ℕ} (ha : 1 ≤ a) (hab : a < b) (hbc : b < c)
    (hEa : Exposed13 a) (hEb : Exposed13 b) (hEc : Exposed13 c)
    (hg1 : ∀ j, a < j → j < b → ¬ Exposed13 j)
    (hg2 : ∀ j, b < j → j < c → ¬ Exposed13 j) :
    3 * (c - a) ≤ 3 * 11 + 1 * 17 := by
  have := pair_sum_le ha hab hbc hEa hEb hEc hg1 hg2
  omega

/-- **Lemma 1 at y = 13**: `F2 - F <= alpha1 * q'` with `F = 11`,
`alpha1 = 1`, `q' = 17`. -/
theorem lemma1_at_13 {a b c : ℕ} (ha : 1 ≤ a) (hab : a < b) (hbc : b < c)
    (hEa : Exposed13 a) (hEb : Exposed13 b) (hEc : Exposed13 c)
    (hg1 : ∀ j, a < j → j < b → ¬ Exposed13 j)
    (hg2 : ∀ j, b < j → j < c → ¬ Exposed13 j) :
    (c - a) - 11 ≤ 1 * 17 := by
  have := pair_sum_le ha hab hbc hEa hEb hEc hg1 hg2
  omega

/-! ## Tier A: the machine-free half of the certificate

The five dangerous pairs killed by the (5,7)-corridor A3 law alone -
`(6,11), (8,11), (11,6), (11,8), (11,11)`. This tier needs no period scan,
so it is the part that scales to machines whose period is beyond kernel
reach.
-/

/-- The five tier-A dangerous pairs have empty A3: forbidden machine-free. -/
theorem tierA_forbidden :
    Corridor.allowed3 6 11 = ∅ ∧ Corridor.allowed3 8 11 = ∅ ∧
      Corridor.allowed3 11 6 = ∅ ∧ Corridor.allowed3 11 8 = ∅ ∧
      Corridor.allowed3 11 11 = ∅ := by decide

/-- Tier A kills a dangerous pair at any machine refining gears `{5,7}`. -/
theorem tierA_kills {a s1 s2 : ℕ}
    (hf : Corridor.allowed3 (s1 % 35) (s2 % 35) = ∅) (ha : 1 ≤ a)
    (h1 : Exposed13 a) (h2 : Exposed13 (a + s1))
    (h3 : Exposed13 (a + s1 + s2)) : False :=
  Corridor.no_chain_of_forbidden hf ha (exposed13_exposed h1)
    (exposed13_exposed h2) (exposed13_exposed h3)

/-- Two maximal gaps can never be adjacent at machine 13 - by the
machine-free law alone, no period scan needed. -/
theorem no_11_11_chain {a : ℕ} (ha : 1 ≤ a) (h1 : Exposed13 a)
    (h2 : Exposed13 (a + 11)) (h3 : Exposed13 (a + 11 + 11)) : False :=
  tierA_kills (by decide) ha h1 h2 h3

end Machine13
