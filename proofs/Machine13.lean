/-
The alpha1 = 1 certificate at machine 13 - the first machine-checked
instance of tolerance lemma 1.

Machine 13 is the gear set {5, 7, 11, 13}, period 5005. Its openings are
the slots no gear marks (`Exposed13`), a residue condition mod 5005
(`exposed13_iff`). The constructor's per-machine statement (constructor.md
sec 21, tool research/strata_adjacency.py, alpha1 = 1, q' = 17):

    F2_k(13) ≤ F_k(13) + alpha1 * q'/3,  i.e.  pair sums ≤ 16.67, F_k = 11.

Formalised here in per-instance form, everything kernel-checked:

* `gap_le` / `gap11_realized`  - every machine gap is ≤ 11, and 11 occurs:
  F_k(13) = 11.
* `pair_sum_le` / `pair16_realized` - adjacent gap sums are ≤ 16, and 16
  occurs: F2_k(13) = 16.
* `alpha1_certificate` - the budget form `3*(c-a) ≤ 3*11 + 1*17`;
  `lemma1_at_13` - the lemma-1 form `(c-a) - F ≤ alpha1 * q'`.

Proof route: the two window facts are evaluated over one full period by the
proof kernel (`decide +kernel`, no native code) - every 11-window holds an
opening, every 16-window holds two. A pair sum ≥ 17 would trap its middle
opening alone in a 16-window, contradiction. The scan replaces certificate
tiers B and C outright: at fixed y the strata census (tier B) is itself a
one-period fact, so the scan is the minimal kernel content. Tier A - the
machine-free A3 law - is formalised separately (`tierA_forbidden`,
`tierA_kills`) because it is the piece that survives at machines whose
period is beyond kernel reach.

Everything was verified against research/strata_adjacency.py before
formalising: residue predicate matches the tool's exposed array on all 5005
residues; both window claims hold cyclically; witnesses 122 (gap 11) and
117 (pair 5+11 = 16) taken from the period scan.
-/

import Corridor

namespace Machine13

/-- An opening of machine 13: no gear in {5, 7, 11, 13} divides either
member of slot `k`. -/
def Exposed13 (k : ℕ) : Prop :=
  ¬ (5 ∣ Census.lo k) ∧ ¬ (5 ∣ Census.hi k) ∧
    ¬ (7 ∣ Census.lo k) ∧ ¬ (7 ∣ Census.hi k) ∧
    ¬ (11 ∣ Census.lo k) ∧ ¬ (11 ∣ Census.hi k) ∧
    ¬ (13 ∣ Census.lo k) ∧ ¬ (13 ∣ Census.hi k)

instance (k : ℕ) : Decidable (Exposed13 k) := by
  unfold Exposed13; infer_instance

/-- The residue form: the marked classes are `±6⁻¹` mod each gear. -/
def ExposedRes (r : ℕ) : Prop :=
  r % 5 ≠ 1 ∧ r % 5 ≠ 4 ∧ r % 7 ≠ 6 ∧ r % 7 ≠ 1 ∧
    r % 11 ≠ 2 ∧ r % 11 ≠ 9 ∧ r % 13 ≠ 11 ∧ r % 13 ≠ 2

instance (r : ℕ) : Decidable (ExposedRes r) := by
  unfold ExposedRes; infer_instance

/-- Openings are a residue condition mod 5005. -/
theorem exposed13_iff {k : ℕ} (hk : 1 ≤ k) :
    Exposed13 k ↔ ExposedRes (k % 5005) := by
  have h5lo : (5 ∣ Census.lo k) ↔ k % 5 = 1 := by simp only [Census.lo]; omega
  have h5hi : (5 ∣ Census.hi k) ↔ k % 5 = 4 := by simp only [Census.hi]; omega
  have h7lo : (7 ∣ Census.lo k) ↔ k % 7 = 6 := by simp only [Census.lo]; omega
  have h7hi : (7 ∣ Census.hi k) ↔ k % 7 = 1 := by simp only [Census.hi]; omega
  have h11lo : (11 ∣ Census.lo k) ↔ k % 11 = 2 := by simp only [Census.lo]; omega
  have h11hi : (11 ∣ Census.hi k) ↔ k % 11 = 9 := by simp only [Census.hi]; omega
  have h13lo : (13 ∣ Census.lo k) ↔ k % 13 = 11 := by simp only [Census.lo]; omega
  have h13hi : (13 ∣ Census.hi k) ↔ k % 13 = 2 := by simp only [Census.hi]; omega
  have e5 : k % 5005 % 5 = k % 5 := by omega
  have e7 : k % 5005 % 7 = k % 7 := by omega
  have e11 : k % 5005 % 11 = k % 11 := by omega
  have e13 : k % 5005 % 13 = k % 13 := by omega
  simp only [Exposed13, ExposedRes, h5lo, h5hi, h7lo, h7hi, h11lo, h11hi,
    h13lo, h13hi, e5, e7, e11, e13, ne_eq]

/-- Machine-13 openings are (5,7)-corridor openings - the bridge to every
machine-free corridor law. -/
theorem exposed13_exposed {k : ℕ} (h : Exposed13 k) : Corridor.Exposed k :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1⟩

/-! ## The two period facts, evaluated by the kernel -/

set_option maxRecDepth 20000 in
/-- Every 11-slot window holds an opening (this is `F_k(13) ≤ 11`). One full
period, checked by the proof kernel. -/
theorem window11_core :
    ∀ r < 5005, ∃ d < 12, 1 ≤ d ∧ ExposedRes ((r + d) % 5005) := by
  decide +kernel

set_option maxRecDepth 20000 in
/-- Every 16-slot window holds two openings (this is `F2_k(13) ≤ 16`). One
full period, checked by the proof kernel. -/
theorem window16_core :
    ∀ r < 5005, ∃ d1 < 17, ∃ d2 < 17, 1 ≤ d1 ∧ d1 < d2 ∧
      ExposedRes ((r + d1) % 5005) ∧ ExposedRes ((r + d2) % 5005) := by
  decide +kernel

/-! ## The gap bounds -/

/-- **`F_k(13) ≤ 11`.** A machine gap - consecutive openings `a < b` with
nothing exposed between - has length at most 11. -/
theorem gap_le {a b : ℕ} (ha : 1 ≤ a) (hab : a < b)
    (hEa : Exposed13 a) (hEb : Exposed13 b)
    (hg : ∀ j, a < j → j < b → ¬ Exposed13 j) : b - a ≤ 11 := by
  by_contra hlt
  obtain ⟨d, hd, h1d, hr⟩ := window11_core (a % 5005) (Nat.mod_lt _ (by omega))
  have e : (a % 5005 + d) % 5005 = (a + d) % 5005 := by omega
  rw [e] at hr
  have hE : Exposed13 (a + d) := (exposed13_iff (by omega)).mpr hr
  exact hg (a + d) (by omega) (by omega) hE

/-- **`F2_k(13) ≤ 16`.** Adjacent machine gaps `(a,b)` and `(b,c)` have
total span at most 16: a span of 17 would trap the middle opening `b` alone
inside a 16-window that must hold two openings. -/
theorem pair_sum_le {a b c : ℕ} (ha : 1 ≤ a) (hab : a < b) (hbc : b < c)
    (hEa : Exposed13 a) (hEb : Exposed13 b) (hEc : Exposed13 c)
    (hg1 : ∀ j, a < j → j < b → ¬ Exposed13 j)
    (hg2 : ∀ j, b < j → j < c → ¬ Exposed13 j) : c - a ≤ 16 := by
  by_contra hlt
  obtain ⟨d1, hd1, d2, hd2, h1le, h12, hr1, hr2⟩ :=
    window16_core (a % 5005) (Nat.mod_lt _ (by omega))
  have e1 : (a % 5005 + d1) % 5005 = (a + d1) % 5005 := by omega
  have e2 : (a % 5005 + d2) % 5005 = (a + d2) % 5005 := by omega
  rw [e1] at hr1
  rw [e2] at hr2
  have hE1 : Exposed13 (a + d1) := (exposed13_iff (by omega)).mpr hr1
  have hE2 : Exposed13 (a + d2) := (exposed13_iff (by omega)).mpr hr2
  have hb1 : a + d1 = b := by
    by_contra hne
    rcases Nat.lt_or_ge (a + d1) b with h | h
    · exact hg1 _ (by omega) h hE1
    · exact hg2 _ (by omega) (by omega) hE1
  have hb2 : a + d2 = b := by
    by_contra hne
    rcases Nat.lt_or_ge (a + d2) b with h | h
    · exact hg1 _ (by omega) h hE2
    · exact hg2 _ (by omega) (by omega) hE2
  omega

/-! ## The witnesses (sharpness) -/

/-- The gap 11 is realized: openings 122 and 133, nothing between.
So `F_k(13) = 11`. -/
theorem gap11_realized :
    Exposed13 122 ∧ Exposed13 133 ∧
      ∀ j, 122 < j → j < 133 → ¬ Exposed13 j := by
  refine ⟨(exposed13_iff (by omega)).mpr (by decide),
    (exposed13_iff (by omega)).mpr (by decide), ?_⟩
  intro j h1 h2
  rw [exposed13_iff (by omega)]
  interval_cases j <;> decide

/-- The pair sum 16 is realized: openings 117, 122, 133 - adjacent gaps
(5, 11). So `F2_k(13) = 16`, and the alpha1 = 1 budget `16.67` is tight. -/
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
budget form: for adjacent machine gaps, `3*(c - a) ≤ 3*F_k + alpha1*q'`
with `F_k = 11`, `q' = 17`. -/
theorem alpha1_certificate {a b c : ℕ} (ha : 1 ≤ a) (hab : a < b)
    (hbc : b < c) (hEa : Exposed13 a) (hEb : Exposed13 b) (hEc : Exposed13 c)
    (hg1 : ∀ j, a < j → j < b → ¬ Exposed13 j)
    (hg2 : ∀ j, b < j → j < c → ¬ Exposed13 j) :
    3 * (c - a) ≤ 3 * 11 + 1 * 17 := by
  have := pair_sum_le ha hab hbc hEa hEb hEc hg1 hg2
  omega

/-- Lemma 1 at y = 13, per-instance form: `F2 - F ≤ alpha1 * q'` with
`F = 11`, `alpha1 = 1`, `q' = 17`. -/
theorem lemma1_at_13 {a b c : ℕ} (ha : 1 ≤ a) (hab : a < b) (hbc : b < c)
    (hEa : Exposed13 a) (hEb : Exposed13 b) (hEc : Exposed13 c)
    (hg1 : ∀ j, a < j → j < b → ¬ Exposed13 j)
    (hg2 : ∀ j, b < j → j < c → ¬ Exposed13 j) :
    (c - a) - 11 ≤ 1 * 17 := by
  have := pair_sum_le ha hab hbc hEa hEb hEc hg1 hg2
  omega

/-! ## Tier A of the certificate, machine-free

The five dangerous pairs that die by the (5,7)-corridor A3 law alone:
`(6,11), (8,11), (11,6), (11,8), (11,11)`. This tier needs no period scan
and survives at machines whose period is beyond kernel reach - it is the
scaling template. (Tiers B and C of the constructor's certificate are
subsumed here by the period scan `window16_core`; their separate class
machinery becomes necessary only from machine 17 up.)
-/

/-- The five tier-A dangerous pairs have empty A3: forbidden machine-free. -/
theorem tierA_forbidden :
    Corridor.allowed3 6 11 = ∅ ∧ Corridor.allowed3 8 11 = ∅ ∧
      Corridor.allowed3 11 6 = ∅ ∧ Corridor.allowed3 11 8 = ∅ ∧
      Corridor.allowed3 11 11 = ∅ := by decide

/-- Tier A kills a dangerous pair for ANY machine refining gears {5,7}:
an empty A3 class set means the three chained openings cannot exist. -/
theorem tierA_kills {a s1 s2 : ℕ}
    (hf : Corridor.allowed3 (s1 % 35) (s2 % 35) = ∅) (ha : 1 ≤ a)
    (h1 : Exposed13 a) (h2 : Exposed13 (a + s1))
    (h3 : Exposed13 (a + s1 + s2)) : False :=
  Corridor.no_chain_of_forbidden hf ha (exposed13_exposed h1)
    (exposed13_exposed h2) (exposed13_exposed h3)

/-- Instance: the pair (11,11) - two maximal gaps - can never be adjacent
at machine 13, by the machine-free law alone. -/
theorem no_11_11_chain {a : ℕ} (ha : 1 ≤ a) (h1 : Exposed13 a)
    (h2 : Exposed13 (a + 11)) (h3 : Exposed13 (a + 11 + 11)) : False :=
  tierA_kills (by decide) ha h1 h2 h3

end Machine13
