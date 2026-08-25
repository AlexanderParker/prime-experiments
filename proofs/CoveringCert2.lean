/-
CONSISTENT COVERING CERTIFICATES: the 11->13 and 13->17 rungs of (D), scan-free
(round 23, after the LP-duality thread's round-23 filing).

`CoveringCert.lean` proves `F(19) <= 37` by bounding each block of the level-2
covering LP SEPARATELY:

    sum y  <=  max_r S_5(r) + sum_j (max_r S_j(r) - min_(r5,rj) P_j(r5,rj)).

That lets gear 5 use one phase in `S_5` and a DIFFERENT one in the pair minima,
which is exactly the marginal consistency the LP-duality thread identified as
missing from round 22's relaxation (and from the whole classical
Bonferroni/Kounias family). It is why that shape fails at 11->13 and 13->17 -
and the failure is not a lack of DEGREE: at machine 13 the inconsistent
relaxation is feasible at degree 2, 3 AND 4.

TYING THE PHASES restores consistency with no dual multiplier. The quantity
being bounded is literally `sum_i y_i * Kounias_i`, so keeping the phases under
ONE quantifier gives the stronger, still finite

    sum y  <=  max over PHASE TUPLES of [ S_5(r5) + sum_j (S_j(rj) - P_j(r5,rj)) ]

and the certificate becomes one bounded-quantified inequality over the machine's
phase tuples, checked by `decide +kernel`. No new lemma is needed - the
rearrangement of `cover_bound` is pure `omega`.

THE CERTIFICATES ARE TINY. The thread's fully-consistent dual at 11->13 is 106
integers over a common denominator and 2,868 rational operations; the phase-tied
form needs

    rung      width  weights                      sum   max over tuples  margin
    11 -> 13    20    20 integers, eighteen 1s      22         21          1
    13 -> 17    28    28 integers in [2,5]          94         92          2

both palindromes (the machine's mirror symmetry `k -> -k`), verified exactly over
every one of the 5,005 and 85,085 phase tuples before formalising (scratchpad
consistent_cert.py, gen_consistent.py; 3,850 and 59,767 DISTINCT coefficient
vectors respectively).

With `CoveringCert.D_17_19_lp` this vehicle now proves THREE consecutive (D)
rungs in the kernel - 11->13, 13->17, 17->19 - sharing nothing with the merge
law. The thread's fourth, 7->11, is not stated here.
-/

import CoveringCert
import Machine13Q
import Machine17Q

namespace CoveringCert2

open CoveringCert

set_option maxRecDepth 40000

/-! ## Rung 11 -> 13: `F(13) <= 20 = F(11) + 13`, window width 20 -/

def yl13 : List ℕ := [1,1,1,1,1,1,1,1,1,2,2,1,1,1,1,1,1,1,1,1]
def y13 (i : ℕ) : ℕ := yl13.getD i 0

def T13 : ℕ := ∑ i ∈ Finset.range 20, y13 i
def S13 (bq : ℕ → ℕ → Bool) (r : ℕ) : ℕ :=
  ∑ i ∈ Finset.range 20, (if bq r i then y13 i else 0)
def P13 (bq : ℕ → ℕ → Bool) (r5 rq : ℕ) : ℕ :=
  ∑ i ∈ Finset.range 20, (if b5 r5 i && bq rq i then y13 i else 0)

theorem T13_eq : T13 = 22 := by decide +kernel

/-- **The consistent certificate at machine 13, width 20**: over EVERY phase
tuple the Kounias-weighted total falls short of the total weight. Margin 1 out
of 22. -/
theorem cert13 : ∀ a < 5, ∀ b < 7, ∀ c < 11, ∀ d < 13,
    S13 b5 a + (S13 b7 b - P13 b7 a b) + (S13 b11 c - P13 b11 a c)
      + (S13 b13 d - P13 b13 a d) < 22 := by decide +kernel

/-- Kounias with a distinguished event, four events. -/
theorem kounias4 (a b c d : Bool) (h : (a || b || c || d) = true) (w : ℕ) :
    w + ((if a && b then w else 0) + (if a && c then w else 0) +
         (if a && d then w else 0))
      ≤ (if a then w else 0) + (if b then w else 0) + (if c then w else 0) +
        (if d then w else 0) := by
  revert h
  cases a <;> cases b <;> cases c <;> cases d <;> simp <;> omega

theorem cover13 {r5 r7 r11 r13 : ℕ}
    (hcov : ∀ i < 20, (b5 r5 i || b7 r7 i || b11 r11 i || b13 r13 i) = true) :
    T13 + (P13 b7 r5 r7 + P13 b11 r5 r11 + P13 b13 r5 r13)
      ≤ S13 b5 r5 + S13 b7 r7 + S13 b11 r11 + S13 b13 r13 := by
  have key : ∀ i ∈ Finset.range 20,
      y13 i + ((if b5 r5 i && b7 r7 i then y13 i else 0) +
        (if b5 r5 i && b11 r11 i then y13 i else 0) +
        (if b5 r5 i && b13 r13 i then y13 i else 0))
      ≤ (if b5 r5 i then y13 i else 0) + (if b7 r7 i then y13 i else 0) +
        (if b11 r11 i then y13 i else 0) + (if b13 r13 i then y13 i else 0) :=
    fun i hi => kounias4 _ _ _ _ (hcov i (Finset.mem_range.mp hi)) (y13 i)
  have hL : T13 + (P13 b7 r5 r7 + P13 b11 r5 r11 + P13 b13 r5 r13)
      = ∑ i ∈ Finset.range 20,
        (y13 i + ((if b5 r5 i && b7 r7 i then y13 i else 0) +
          (if b5 r5 i && b11 r11 i then y13 i else 0) +
          (if b5 r5 i && b13 r13 i then y13 i else 0))) := by
    simp only [T13, P13, Finset.sum_add_distrib]
  have hR : S13 b5 r5 + S13 b7 r7 + S13 b11 r11 + S13 b13 r13
      = ∑ i ∈ Finset.range 20,
        ((if b5 r5 i then y13 i else 0) + (if b7 r7 i then y13 i else 0) +
          (if b11 r11 i then y13 i else 0) + (if b13 r13 i then y13 i else 0)) := by
    simp only [S13, Finset.sum_add_distrib]
  rw [hL, hR]
  exact Finset.sum_le_sum key

theorem no_cover13 {r5 r7 r11 r13 : ℕ} (h5 : r5 < 5) (h7 : r7 < 7)
    (h11 : r11 < 11) (h13 : r13 < 13)
    (hcov : ∀ i < 20, (b5 r5 i || b7 r7 i || b11 r11 i || b13 r13 i) = true) :
    False := by
  have hb := cover13 hcov
  have hc := cert13 r5 h5 r7 h7 r11 h11 r13 h13
  rw [T13_eq] at hb
  omega

set_option maxHeartbeats 1000000 in
theorem blocked13 {p i : ℕ} (hp : 1 ≤ p) (h : ¬ Machine13.Exposed13 (p + i)) :
    (b5 (p % 5) i || b7 (p % 7) i || b11 (p % 11) i || b13 (p % 13) i) = true := by
  have q5 : (p % 5 + i) % 5 = (p + i) % 5 := by omega
  have q7 : (p % 7 + i) % 7 = (p + i) % 7 := by omega
  have q11 : (p % 11 + i) % 11 = (p + i) % 11 := by omega
  have q13 : (p % 13 + i) % 13 = (p + i) % 13 := by omega
  simp only [b5, b7, b11, b13, q5, q7, q11, q13, Bool.or_eq_true, beq_iff_eq]
  by_contra hc
  apply h
  rw [Machine13.exposed13_iff (show 1 ≤ p + i by omega)]
  simp only [Machine13.expT, Bool.and_eq_true, bne_iff_ne, ne_eq, and_assoc]
  push Not at hc
  tauto

/-- **`F(13) <= 20`, by LP duality with marginal consistency**: every window of
20 consecutive slots contains a machine-13 opening. -/
theorem no_20_run {p : ℕ} (hp : 1 ≤ p) : ∃ i < 20, Machine13.Exposed13 (p + i) := by
  by_contra hc
  push Not at hc
  exact no_cover13 (r5 := p % 5) (r7 := p % 7) (r11 := p % 11) (r13 := p % 13)
    (by omega) (by omega) (by omega) (by omega)
    (fun i hi => blocked13 hp (hc i hi))

theorem F13_le_20 (n : ℕ) : Machine13.g13 n ≤ 20 := by
  by_contra hcon
  obtain ⟨i, hi, hE⟩ := no_20_run (p := Machine13.opSeq n + 1)
    (by have := Machine13.opSeq_pos n; omega)
  have hgap : Machine13.g13 n = Machine13.opSeq (n + 1) - Machine13.opSeq n := rfl
  have hlt := Machine13.opSeq_lt_succ n
  exact Machine13.opSeq_gap_empty n (Machine13.opSeq n + 1 + i)
    (by omega) (by omega) hE

/-- **(D) at the 11->13 step, PROVED BY A COVERING CERTIFICATE.** The rung the
inconsistent relaxation misses by exactly one, at degrees 2, 3 and 4 alike. -/
theorem D_11_13_lp (n : ℕ) : Machine13.g13 n ≤ 7 + 13 := F13_le_20 n

/-! ## Rung 13 -> 17: `F(17) <= 28 = F(13) + 17`, window width 28 -/

def yl17 : List ℕ :=
  [2,2,3,2,2,4,4,4,3,3,4,4,5,5,5,5,4,4,3,3,4,4,4,2,2,3,2,2]
def y17 (i : ℕ) : ℕ := yl17.getD i 0

def T17 : ℕ := ∑ i ∈ Finset.range 28, y17 i
def S17 (bq : ℕ → ℕ → Bool) (r : ℕ) : ℕ :=
  ∑ i ∈ Finset.range 28, (if bq r i then y17 i else 0)
def P17 (bq : ℕ → ℕ → Bool) (r5 rq : ℕ) : ℕ :=
  ∑ i ∈ Finset.range 28, (if b5 r5 i && bq rq i then y17 i else 0)

theorem T17_eq : T17 = 94 := by decide +kernel

/-- **The consistent certificate at machine 17, width 28**: margin 2 out of 94,
over all 85,085 phase tuples. -/
theorem cert17 : ∀ a < 5, ∀ b < 7, ∀ c < 11, ∀ d < 13, ∀ e < 17,
    S17 b5 a + (S17 b7 b - P17 b7 a b) + (S17 b11 c - P17 b11 a c)
      + (S17 b13 d - P17 b13 a d) + (S17 b17 e - P17 b17 a e) < 94 := by
  decide +kernel

theorem kounias5 (a b c d e : Bool) (h : (a || b || c || d || e) = true) (w : ℕ) :
    w + ((if a && b then w else 0) + (if a && c then w else 0) +
         (if a && d then w else 0) + (if a && e then w else 0))
      ≤ (if a then w else 0) + (if b then w else 0) + (if c then w else 0) +
        (if d then w else 0) + (if e then w else 0) := by
  revert h
  cases a <;> cases b <;> cases c <;> cases d <;> cases e <;> simp <;> omega

theorem cover17 {r5 r7 r11 r13 r17 : ℕ}
    (hcov : ∀ i < 28,
      (b5 r5 i || b7 r7 i || b11 r11 i || b13 r13 i || b17 r17 i) = true) :
    T17 + (P17 b7 r5 r7 + P17 b11 r5 r11 + P17 b13 r5 r13 + P17 b17 r5 r17)
      ≤ S17 b5 r5 + S17 b7 r7 + S17 b11 r11 + S17 b13 r13 + S17 b17 r17 := by
  have key : ∀ i ∈ Finset.range 28,
      y17 i + ((if b5 r5 i && b7 r7 i then y17 i else 0) +
        (if b5 r5 i && b11 r11 i then y17 i else 0) +
        (if b5 r5 i && b13 r13 i then y17 i else 0) +
        (if b5 r5 i && b17 r17 i then y17 i else 0))
      ≤ (if b5 r5 i then y17 i else 0) + (if b7 r7 i then y17 i else 0) +
        (if b11 r11 i then y17 i else 0) + (if b13 r13 i then y17 i else 0) +
        (if b17 r17 i then y17 i else 0) :=
    fun i hi => kounias5 _ _ _ _ _ (hcov i (Finset.mem_range.mp hi)) (y17 i)
  have hL : T17 + (P17 b7 r5 r7 + P17 b11 r5 r11 + P17 b13 r5 r13 + P17 b17 r5 r17)
      = ∑ i ∈ Finset.range 28,
        (y17 i + ((if b5 r5 i && b7 r7 i then y17 i else 0) +
          (if b5 r5 i && b11 r11 i then y17 i else 0) +
          (if b5 r5 i && b13 r13 i then y17 i else 0) +
          (if b5 r5 i && b17 r17 i then y17 i else 0))) := by
    simp only [T17, P17, Finset.sum_add_distrib]
  have hR : S17 b5 r5 + S17 b7 r7 + S17 b11 r11 + S17 b13 r13 + S17 b17 r17
      = ∑ i ∈ Finset.range 28,
        ((if b5 r5 i then y17 i else 0) + (if b7 r7 i then y17 i else 0) +
          (if b11 r11 i then y17 i else 0) + (if b13 r13 i then y17 i else 0) +
          (if b17 r17 i then y17 i else 0)) := by
    simp only [S17, Finset.sum_add_distrib]
  rw [hL, hR]
  exact Finset.sum_le_sum key

theorem no_cover17 {r5 r7 r11 r13 r17 : ℕ} (h5 : r5 < 5) (h7 : r7 < 7)
    (h11 : r11 < 11) (h13 : r13 < 13) (h17 : r17 < 17)
    (hcov : ∀ i < 28,
      (b5 r5 i || b7 r7 i || b11 r11 i || b13 r13 i || b17 r17 i) = true) :
    False := by
  have hb := cover17 hcov
  have hc := cert17 r5 h5 r7 h7 r11 h11 r13 h13 r17 h17
  rw [T17_eq] at hb
  omega

set_option maxHeartbeats 1000000 in
theorem blocked17 {p i : ℕ} (hp : 1 ≤ p) (h : ¬ Machine17.Exposed17 (p + i)) :
    (b5 (p % 5) i || b7 (p % 7) i || b11 (p % 11) i || b13 (p % 13) i ||
      b17 (p % 17) i) = true := by
  have q5 : (p % 5 + i) % 5 = (p + i) % 5 := by omega
  have q7 : (p % 7 + i) % 7 = (p + i) % 7 := by omega
  have q11 : (p % 11 + i) % 11 = (p + i) % 11 := by omega
  have q13 : (p % 13 + i) % 13 = (p + i) % 13 := by omega
  have q17 : (p % 17 + i) % 17 = (p + i) % 17 := by omega
  simp only [b5, b7, b11, b13, b17, q5, q7, q11, q13, q17, Bool.or_eq_true,
    beq_iff_eq]
  by_contra hc
  apply h
  rw [Machine17.exposed17_iff (show 1 ≤ p + i by omega)]
  simp only [Machine17.expT, Bool.and_eq_true, bne_iff_ne, ne_eq, and_assoc]
  push Not at hc
  tauto

/-- **`F(17) <= 28`, by LP duality with marginal consistency.** -/
theorem no_28_run {p : ℕ} (hp : 1 ≤ p) : ∃ i < 28, Machine17.Exposed17 (p + i) := by
  by_contra hc
  push Not at hc
  exact no_cover17 (r5 := p % 5) (r7 := p % 7) (r11 := p % 11) (r13 := p % 13)
    (r17 := p % 17) (by omega) (by omega) (by omega) (by omega) (by omega)
    (fun i hi => blocked17 hp (hc i hi))

theorem F17_le_28 (n : ℕ) : Machine17.g17 n ≤ 28 := by
  by_contra hcon
  obtain ⟨i, hi, hE⟩ := no_28_run (p := Machine17.opSeq n + 1)
    (by have := Machine17.opSeq_pos n; omega)
  have hgap : Machine17.g17 n = Machine17.opSeq (n + 1) - Machine17.opSeq n := rfl
  have hlt := Machine17.opSeq_lt_succ n
  exact Machine17.opSeq_gap_empty n (Machine17.opSeq n + 1 + i)
    (by omega) (by omega) hE

/-- **(D) at the 13->17 step, PROVED BY A COVERING CERTIFICATE.** -/
theorem D_13_17_lp (n : ℕ) : Machine17.g17 n ≤ 11 + 17 := F17_le_28 n

/-- **THREE CONSECUTIVE (D) RUNGS BY COVERING CERTIFICATES**, sharing nothing
with the merge law: 11->13 and 13->17 need marginal consistency, 17->19 does
not. -/
theorem lp_ladder :
    (∀ n, Machine13.g13 n ≤ 7 + 13) ∧ (∀ n, Machine17.g17 n ≤ 11 + 17) ∧
      (∀ n, Machine19.g19 n ≤ 18 + 19) :=
  ⟨D_11_13_lp, D_13_17_lp, CoveringCert.D_17_19_lp⟩

end CoveringCert2
