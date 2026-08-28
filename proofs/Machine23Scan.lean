/-
Machine 23's spectrum and qualifying spectrum EXTRACTED from the
position-indexed period scan, and THE FIFTH RUNG OF THE (D) LADDER,
HYPOTHESIS-FREE (round 24).

`Machine23Idx.lean` holds the position-indexed encoding (the round-23
will-not-close verdict's named fix: index the machine-19 opening chain by
POSITION, not offset, so the 23 gear-23 phases share one machine-19 walk);
`Machine23IdxS0..S16` decide its 323 slices in the kernel.  This file turns
those Bools into the two statements `Machine29.D_at_23_29` consumes:

    spectrum23_two : Spectrum.SpectrumBound g23 2 39      -- F_2(23) <= 39
    qual23_all     : ∀ j, 3 ≤ j → Spectrum.QualBound g23 5 j 60

and instantiates the rung:

    D_23_29 (n) : Machine29.g29 n ≤ 34 + 29     -- (D) at 23->29, NO hypotheses
    g29_le_60 (n) : Machine29.g29 n ≤ 60        -- R39's form, margin 3

THE BRIDGE, in one line: machine 23's openings are machine 19's openings off
gear 23's two teeth, so the `k`-th machine-19 opening after a base (`W`,
position-indexed, phase-free) is a machine-23 opening exactly when its offset
avoids the teeth mod 23 - and `NS` steps to the next position where it does.
`next23_step` is the whole content: the position the survivor step lands on
carries `nextOp23`.  The scan certifies its own fuel (each clause carries the
position check `p_i ≤ p_(i-1) + 5` = "`NS` did not reach its sentinel"), so
nothing here imports a bound from outside the kernel.

Validated over the full 37,182,145-slot period in numpy BEFORE building
(scratchpad idx23b/idx23d.py): the chain Bool is true at all 7,952,175
machine-23 openings, and the values it reads are exactly
`F_1..F_6(23) = 34, 39, 50, 58, 65, 77` and `Q_j(23;10) ≤ 60`.
-/

import Machine29
import Machine23IdxS0
import Machine23IdxS1
import Machine23IdxS2
import Machine23IdxS3
import Machine23IdxS4
import Machine23IdxS5
import Machine23IdxS6
import Machine23IdxS7
import Machine23IdxS8
import Machine23IdxS9
import Machine23IdxS10
import Machine23IdxS11
import Machine23IdxS12
import Machine23IdxS13
import Machine23IdxS14
import Machine23IdxS15
import Machine23IdxS16

namespace Machine23

/-- **One period, all 323 slices of the position-indexed chain scan.** -/
theorem qsliceIdxAll : ∀ e < 17, ∀ f < 19, qsliceIdx e f = true := by
  intro e he
  interval_cases e
  exacts [iasm0, iasm1, iasm2, iasm3, iasm4, iasm5, iasm6, iasm7, iasm8,
    iasm9, iasm10, iasm11, iasm12, iasm13, iasm14, iasm15, iasm16]

open Machine19

/-- The tuple-and-phase fact, unpacked from the slice. -/
theorem qokIdxAll {a b c d e f g : ℕ} (ha : a < 5) (hb : b < 7) (hc : c < 11)
    (hd : d < 13) (he : e < 17) (hf : f < 19) (hg : g < 23)
    (hop : Machine19.atT a b c d e f 0 = true) :
    qokIdx a b c d e f g = true := by
  have h := qsliceIdxAll e he f hf
  rw [qsliceIdx, List.all_eq_true] at h
  have h1 := h a (List.mem_range.mpr ha)
  rw [List.all_eq_true] at h1
  have h2 := h1 b (List.mem_range.mpr hb)
  rw [List.all_eq_true] at h2
  have h3 := h2 c (List.mem_range.mpr hc)
  rw [List.all_eq_true] at h3
  have h4 := h3 d (List.mem_range.mpr hd)
  rw [hop] at h4
  simp only [Bool.not_true, Bool.false_or, List.all_eq_true] at h4
  exact h4 g (List.mem_range.mpr hg)

/-! ## The position chain at a base value -/

/-- The machine-19 position chain at base `x`: the offset of the `k`-th
machine-19 opening after `x`.  Phase-free. -/
def W (x k : ℕ) : ℕ := w19 (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) (x % 19) k

/-- The gear-23 survivor step at base `x`. -/
def NS (x fu k : ℕ) : ℕ :=
  nsurv (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) (x % 19) (x % 23) fu k

theorem W_zero (x : ℕ) : W x 0 = 0 := rfl

theorem W_succ (x k : ℕ) :
    W x (k + 1) =
      Machine19.seekT (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) (x % 19) 25 (W x k) :=
  rfl

/-- The position chain is strictly increasing. -/
theorem W_lt_succ (x k : ℕ) : W x k < W x (k + 1) := by
  rw [W_succ]
  exact Machine19.seekT_gt _ _ _ _ _ _ 25 (W x k)

theorem W_strict_mono {x k k' : ℕ} (h : k < k') : W x k < W x k' := by
  induction k' with
  | zero => omega
  | succ k' ih =>
    rcases Nat.lt_or_ge k k' with hlt | hge
    · exact lt_trans (ih hlt) (W_lt_succ x k')
    · have : k = k' := by omega
      subst this
      exact W_lt_succ x k

/-! ## The survivor step -/

theorem nsurv_succ_kill {a b c d e f g fu k : ℕ}
    (h : kil23 g (w19 a b c d e f (k + 1)) = true) :
    nsurv a b c d e f g (fu + 1) k = nsurv a b c d e f g fu (k + 1) := by
  simp only [nsurv]
  split
  · rfl
  · rename_i hneg
    exact absurd h hneg

theorem nsurv_succ_alive {a b c d e f g fu k : ℕ}
    (h : ¬ kil23 g (w19 a b c d e f (k + 1)) = true) :
    nsurv a b c d e f g (fu + 1) k = k + 1 := by
  simp only [nsurv]
  split
  · rename_i hpos
    exact absurd hpos h
  · rfl

theorem nsurv_gt (a b c d e f g : ℕ) : ∀ fu k, k < nsurv a b c d e f g fu k := by
  intro fu
  induction fu with
  | zero => intro k; simp only [nsurv]; omega
  | succ fu ih =>
    intro k
    by_cases h : kil23 g (w19 a b c d e f (k + 1)) = true
    · rw [nsurv_succ_kill h]
      have := ih (k + 1); omega
    · rw [nsurv_succ_alive h]; omega

/-- Within fuel, the position the survivor step lands on survives gear 23. -/
theorem nsurv_alive (a b c d e f g : ℕ) : ∀ fu k,
    nsurv a b c d e f g fu k ≤ k + fu →
      kil23 g (w19 a b c d e f (nsurv a b c d e f g fu k)) = false := by
  intro fu
  induction fu with
  | zero => intro k h; simp only [nsurv] at h; omega
  | succ fu ih =>
    intro k h
    by_cases hk : kil23 g (w19 a b c d e f (k + 1)) = true
    · rw [nsurv_succ_kill hk] at h ⊢
      exact ih (k + 1) (by omega)
    · rw [nsurv_succ_alive hk]
      exact Bool.eq_false_iff.mpr hk

/-- Every position strictly between is killed by gear 23. -/
theorem nsurv_min (a b c d e f g : ℕ) : ∀ fu k i, k < i →
    i < nsurv a b c d e f g fu k → i ≤ k + fu →
      kil23 g (w19 a b c d e f i) = true := by
  intro fu
  induction fu with
  | zero => intro k i h1 _ h3; omega
  | succ fu ih =>
    intro k i h1 h2 h3
    by_cases hk : kil23 g (w19 a b c d e f (k + 1)) = true
    · rw [nsurv_succ_kill hk] at h2
      rcases Nat.lt_or_ge (k + 1) i with hlt | hge
      · exact ih (k + 1) i hlt h2 (by omega)
      · have : i = k + 1 := by omega
        subst this; exact hk
    · rw [nsurv_succ_alive hk] at h2; omega

theorem NS_gt (x fu k : ℕ) : k < NS x fu k := nsurv_gt _ _ _ _ _ _ _ fu k

theorem NS_alive (x fu k : ℕ) (h : NS x fu k ≤ k + fu) :
    kil23 (x % 23) (W x (NS x fu k)) = false := nsurv_alive _ _ _ _ _ _ _ fu k h

theorem NS_min (x fu k i : ℕ) (h1 : k < i) (h2 : i < NS x fu k) (h3 : i ≤ k + fu) :
    kil23 (x % 23) (W x i) = true := nsurv_min _ _ _ _ _ _ _ fu k i h1 h2 h3

/-! ## The bridge to machine 23's openings -/

/-- Gear 23's kill test on the phase is machine 23's own `Killed23`. -/
theorem kil23_iff (x t : ℕ) : kil23 (x % 23) t = true ↔ Killed23 (x + t) := by
  simp only [kil23, Killed23, Bool.or_eq_true, beq_iff_eq]
  omega

/-- **The position chain reads machine 19's enumeration.** -/
theorem W_eq {x m : ℕ} (hx : 1 ≤ x) (hm : Machine19.opSeq m = x) :
    ∀ k, x + W x k = Machine19.opSeq (m + k) := by
  intro k
  induction k with
  | zero => rw [W_zero]; simpa using hm.symm
  | succ k ih =>
    rw [W_succ]
    have hE : Machine19.Exposed19 (x + W x k) := by
      rw [ih]; exact Machine19.opSeq_exposed _
    have hs := Machine19.seek_next (x := x) (s := W x k) hx hE
    rw [hs, ih, show m + (k + 1) = (m + k) + 1 by omega, Machine19.opSeq_succ]

/-- **The survivor step carries `nextOp23`.**  If `x` is a machine-19 opening
and the survivor step from position `k` did not reach its sentinel, then the
position it lands on carries the next machine-23 opening. -/
theorem next23_step {x m k : ℕ} (hx : 1 ≤ x) (hm : Machine19.opSeq m = x)
    (hp : NS x 5 k ≤ k + 5) :
    nextOp23 (x + W x k) = x + W x (NS x 5 k) := by
  have hkp : k < NS x 5 k := NS_gt x 5 k
  have halive : kil23 (x % 23) (W x (NS x 5 k)) = false := NS_alive x 5 k hp
  have hWp : x + W x (NS x 5 k) = Machine19.opSeq (m + NS x 5 k) := W_eq hx hm _
  have hWk : x + W x k = Machine19.opSeq (m + k) := W_eq hx hm k
  have hE19 : Machine19.Exposed19 (x + W x (NS x 5 k)) := by
    rw [hWp]; exact Machine19.opSeq_exposed _
  have hpos : 1 ≤ x + W x (NS x 5 k) := by omega
  have hnk : ¬ Killed23 (x + W x (NS x 5 k)) := by
    intro hK
    rw [← kil23_iff] at hK
    rw [halive] at hK
    exact Bool.noConfusion hK
  have hE23 : Exposed23 (x + W x (NS x 5 k)) := exposed23_of hpos hE19 hnk
  have hlt : x + W x k < x + W x (NS x 5 k) := by
    have := W_strict_mono (x := x) hkp; omega
  have hle : nextOp23 (x + W x k) ≤ x + W x (NS x 5 k) :=
    Nat.find_min' (exists_exposed23_above (x + W x k)) ⟨hlt, hE23⟩
  rcases eq_or_lt_of_le hle with hEq | hLt
  · exact hEq
  · exfalso
    have hgt := nextOp23_gt (x + W x k)
    have hEz : Exposed23 (nextOp23 (x + W x k)) := nextOp23_exposed _
    obtain ⟨j, hj⟩ :=
      Machine19.opSeq_surj (m := nextOp23 (x + W x k)) (by omega) hEz.1
    have hjk : m + k < j := by
      by_contra hc
      have h1 : Machine19.opSeq j ≤ Machine19.opSeq (m + k) := by
        rcases eq_or_lt_of_le (show j ≤ m + k by omega) with he | hl
        · rw [he]
        · exact le_of_lt (Machine19.opSeq_strict_mono hl)
      omega
    have hjp : j < m + NS x 5 k := by
      by_contra hc
      have h1 : Machine19.opSeq (m + NS x 5 k) ≤ Machine19.opSeq j := by
        rcases eq_or_lt_of_le (show m + NS x 5 k ≤ j by omega) with he | hl
        · rw [he]
        · exact le_of_lt (Machine19.opSeq_strict_mono hl)
      omega
    have hkill := NS_min x 5 k (j - m) (by omega) (by omega) (by omega)
    have hWj : x + W x (j - m) = Machine19.opSeq j := by
      have := W_eq hx hm (j - m)
      rwa [show m + (j - m) = j by omega] at this
    have hKil : Killed23 (x + W x (j - m)) := (kil23_iff x (W x (j - m))).mp hkill
    rw [hWj, hj] at hKil
    exact not_killed_of_exposed23 (by omega) hEz hKil

/-! ## The chain check, restated at a base value -/

/-- `chainIdx` at the CRT tuple of a base value `x`, in the `NS`/`W`
vocabulary.  Definitionally equal to
`chainIdx (x%5) (x%7) (x%11) (x%13) (x%17) (x%19) (x%23)`. -/
def chainNS (x : ℕ) : Bool :=
  let p1 := NS x 5 0
  let p2 := NS x 5 p1
  let p3 := NS x 5 p2
  let p4 := NS x 5 p3
  let p5 := NS x 5 p4
  let p6 := NS x 5 p5
  let o1 := W x p1
  let o2 := W x p2
  let o3 := W x p3
  let o4 := W x p4
  let o5 := W x p5
  let o6 := W x p6
  let q2 := Nat.ble 10 (o2 - o1)
  let q3 := Nat.ble 10 (o3 - o2)
  let q4 := Nat.ble 10 (o4 - o3)
  let q5 := Nat.ble 10 (o5 - o4)
  Nat.ble p1 5 && Nat.ble o1 34 &&
  Nat.ble p2 (p1 + 5) && Nat.ble o2 39 &&
  (!q2 || (Nat.ble p3 (p2 + 5) && Nat.ble o3 60)) &&
  (!(q2 && q3) || (Nat.ble p4 (p3 + 5) && Nat.ble o4 60)) &&
  (!(q2 && q3 && q4) || (Nat.ble p5 (p4 + 5) && Nat.ble o5 60)) &&
  (!(q2 && q3 && q4 && q5) || (Nat.ble p6 (p5 + 5) && Nat.ble o6 60)) &&
  !(Nat.ble 10 o1 && q2 && q3 && q4 && q5)

theorem chainNS_eq (x : ℕ) :
    chainNS x = chainIdx (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) (x % 19) (x % 23) :=
  rfl

/-! ## The chain facts over machine 23's enumeration -/

/-- **The seven scan facts at every machine-23 opening**, over the
enumeration: `F_1 <= 34`, `F_2 <= 39`, the four guarded qualifying rungs
`<= 60`, and the five-run refutation. -/
theorem chain_facts23 (n : ℕ) :
    (opSeq23 (n + 1) - opSeq23 n ≤ 34) ∧
    (opSeq23 (n + 2) - opSeq23 n ≤ 39) ∧
    (10 ≤ g23 (n + 1) → opSeq23 (n + 3) - opSeq23 n ≤ 60) ∧
    (10 ≤ g23 (n + 1) → 10 ≤ g23 (n + 2) → opSeq23 (n + 4) - opSeq23 n ≤ 60) ∧
    (10 ≤ g23 (n + 1) → 10 ≤ g23 (n + 2) → 10 ≤ g23 (n + 3) →
      opSeq23 (n + 5) - opSeq23 n ≤ 60) ∧
    (10 ≤ g23 (n + 1) → 10 ≤ g23 (n + 2) → 10 ≤ g23 (n + 3) → 10 ≤ g23 (n + 4) →
      opSeq23 (n + 6) - opSeq23 n ≤ 60) ∧
    ¬ (10 ≤ g23 n ∧ 10 ≤ g23 (n + 1) ∧ 10 ≤ g23 (n + 2) ∧ 10 ≤ g23 (n + 3) ∧
        10 ≤ g23 (n + 4)) := by
  set x := opSeq23 n with hxdef
  have hx : 1 ≤ x := opSeq23_pos n
  have hE : Exposed23 x := opSeq23_exposed n
  obtain ⟨m, hm⟩ := Machine19.opSeq_surj hx hE.1
  have hat : Machine19.atT (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) (x % 19) 0 = true :=
    (Machine19.atT_iff hx 0).mpr (by simpa using hE.1)
  have hkil0 : kil23 (x % 23) 0 = false := by
    rw [Bool.eq_false_iff]
    intro hK
    exact not_killed_of_exposed23 hx hE (by simpa using (kil23_iff x 0).mp hK)
  have h : qokIdx (x % 5) (x % 7) (x % 11) (x % 13) (x % 17) (x % 19) (x % 23) = true :=
    qokIdxAll (by omega) (by omega) (by omega) (by omega) (by omega) (by omega)
      (by omega) hat
  rw [qokIdx, hkil0, Bool.false_or, ← chainNS_eq] at h
  rw [chainNS] at h
  set k1 := NS x 5 0 with hk1
  set k2 := NS x 5 k1 with hk2
  set k3 := NS x 5 k2 with hk3
  set k4 := NS x 5 k3 with hk4
  set k5 := NS x 5 k4 with hk5
  set k6 := NS x 5 k5 with hk6
  set o1 := W x k1 with ho1
  set o2 := W x k2 with ho2
  set o3 := W x k3 with ho3
  set o4 := W x k4 with ho4
  set o5 := W x k5 with ho5
  set o6 := W x k6 with ho6
  simp only [Bool.and_eq_true, Nat.ble_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨hp1, hv1⟩, hp2⟩, hv2⟩, hC3⟩, hC4⟩, hC5⟩, hC6⟩, hC7⟩ := h
  have hm12 : o1 < o2 := W_strict_mono (NS_gt x 5 k1)
  have hm23 : o2 < o3 := W_strict_mono (NS_gt x 5 k2)
  have hm34 : o3 < o4 := W_strict_mono (NS_gt x 5 k3)
  have hm45 : o4 < o5 := W_strict_mono (NS_gt x 5 k4)
  have hm56 : o5 < o6 := W_strict_mono (NS_gt x 5 k5)
  have e1 : opSeq23 (n + 1) = x + o1 := by
    have hstep := next23_step (m := m) (k := 0) hx hm (by omega)
    rw [W_zero, Nat.add_zero] at hstep
    rw [opSeq23_succ, ← hxdef, ho1, hk1]
    exact hstep
  have e2 : opSeq23 (n + 2) = x + o2 := by
    have hstep := next23_step (m := m) (k := k1) hx hm (by omega)
    rw [show n + 2 = (n + 1) + 1 by omega, opSeq23_succ, e1, ho1, ho2, hk2]
    exact hstep
  have hg1 : g23 n = o1 := by
    simp only [g23]
    rw [e1, ← hxdef]
    omega
  have hg2v : g23 (n + 1) = o2 - o1 := by
    simp only [g23]
    rw [show n + 1 + 1 = n + 2 by omega, e2, e1]
    omega
  have step3 : 10 ≤ g23 (n + 1) →
      k3 ≤ k2 + 5 ∧ o3 ≤ 60 ∧ opSeq23 (n + 3) = x + o3 := by
    intro hq
    have hb : Nat.ble 10 (o2 - o1) = true := Nat.ble_eq.mpr (by omega)
    rw [hb] at hC3
    simp only [Bool.not_true, Bool.false_or, Bool.and_eq_true, Nat.ble_eq] at hC3
    obtain ⟨hk, hv⟩ := hC3
    have hstep := next23_step (m := m) (k := k2) hx hm (by omega)
    refine ⟨hk, hv, ?_⟩
    rw [show n + 3 = (n + 2) + 1 by omega, opSeq23_succ, e2, ho2, ho3, hk3]
    exact hstep
  have step4 : 10 ≤ g23 (n + 1) → 10 ≤ g23 (n + 2) →
      k4 ≤ k3 + 5 ∧ o4 ≤ 60 ∧ opSeq23 (n + 4) = x + o4 := by
    intro hq1 hq2
    obtain ⟨hk3le, hv3, e3⟩ := step3 hq1
    have hg3v : g23 (n + 2) = o3 - o2 := by
      simp only [g23]
      rw [show n + 2 + 1 = n + 3 by omega, e3, e2]
      omega
    have hb1 : Nat.ble 10 (o2 - o1) = true := Nat.ble_eq.mpr (by omega)
    have hb2 : Nat.ble 10 (o3 - o2) = true := Nat.ble_eq.mpr (by omega)
    rw [hb1, hb2] at hC4
    simp only [Bool.and_true, Bool.not_true, Bool.false_or, Bool.and_eq_true,
      Nat.ble_eq] at hC4
    obtain ⟨hk, hv⟩ := hC4
    have hstep := next23_step (m := m) (k := k3) hx hm (by omega)
    refine ⟨hk, hv, ?_⟩
    rw [show n + 4 = (n + 3) + 1 by omega, opSeq23_succ, e3, ho3, ho4, hk4]
    exact hstep
  have step5 : 10 ≤ g23 (n + 1) → 10 ≤ g23 (n + 2) → 10 ≤ g23 (n + 3) →
      k5 ≤ k4 + 5 ∧ o5 ≤ 60 ∧ opSeq23 (n + 5) = x + o5 := by
    intro hq1 hq2 hq3
    obtain ⟨hk3le, hv3, e3⟩ := step3 hq1
    obtain ⟨hk4le, hv4, e4⟩ := step4 hq1 hq2
    have hg3v : g23 (n + 2) = o3 - o2 := by
      simp only [g23]
      rw [show n + 2 + 1 = n + 3 by omega, e3, e2]
      omega
    have hg4v : g23 (n + 3) = o4 - o3 := by
      simp only [g23]
      rw [show n + 3 + 1 = n + 4 by omega, e4, e3]
      omega
    have hb1 : Nat.ble 10 (o2 - o1) = true := Nat.ble_eq.mpr (by omega)
    have hb2 : Nat.ble 10 (o3 - o2) = true := Nat.ble_eq.mpr (by omega)
    have hb3 : Nat.ble 10 (o4 - o3) = true := Nat.ble_eq.mpr (by omega)
    rw [hb1, hb2, hb3] at hC5
    simp only [Bool.and_true, Bool.not_true, Bool.false_or, Bool.and_eq_true,
      Nat.ble_eq] at hC5
    obtain ⟨hk, hv⟩ := hC5
    have hstep := next23_step (m := m) (k := k4) hx hm (by omega)
    refine ⟨hk, hv, ?_⟩
    rw [show n + 5 = (n + 4) + 1 by omega, opSeq23_succ, e4, ho4, ho5, hk5]
    exact hstep
  have step6 : 10 ≤ g23 (n + 1) → 10 ≤ g23 (n + 2) → 10 ≤ g23 (n + 3) →
      10 ≤ g23 (n + 4) → o6 ≤ 60 ∧ opSeq23 (n + 6) = x + o6 := by
    intro hq1 hq2 hq3 hq4
    obtain ⟨hk3le, hv3, e3⟩ := step3 hq1
    obtain ⟨hk4le, hv4, e4⟩ := step4 hq1 hq2
    obtain ⟨hk5le, hv5, e5⟩ := step5 hq1 hq2 hq3
    have hg3v : g23 (n + 2) = o3 - o2 := by
      simp only [g23]
      rw [show n + 2 + 1 = n + 3 by omega, e3, e2]
      omega
    have hg5v : g23 (n + 4) = o5 - o4 := by
      simp only [g23]
      rw [show n + 4 + 1 = n + 5 by omega, e5, e4]
      omega
    have hb1 : Nat.ble 10 (o2 - o1) = true := Nat.ble_eq.mpr (by omega)
    have hb2 : Nat.ble 10 (o3 - o2) = true := Nat.ble_eq.mpr (by omega)
    have hg4v : g23 (n + 3) = o4 - o3 := by
      simp only [g23]
      rw [show n + 3 + 1 = n + 4 by omega, e4, e3]
      omega
    have hb3 : Nat.ble 10 (o4 - o3) = true := Nat.ble_eq.mpr (by omega)
    have hb4 : Nat.ble 10 (o5 - o4) = true := Nat.ble_eq.mpr (by omega)
    rw [hb1, hb2, hb3, hb4] at hC6
    simp only [Bool.and_true, Bool.not_true, Bool.false_or, Bool.and_eq_true,
      Nat.ble_eq] at hC6
    obtain ⟨hk, hv⟩ := hC6
    have hstep := next23_step (m := m) (k := k5) hx hm (by omega)
    refine ⟨hv, ?_⟩
    rw [show n + 6 = (n + 5) + 1 by omega, opSeq23_succ, e5, ho5, ho6, hk6]
    exact hstep
  refine ⟨by rw [e1]; omega, by rw [e2]; omega, ?_, ?_, ?_, ?_, ?_⟩
  · intro hq1
    obtain ⟨_, hv, e3⟩ := step3 hq1
    rw [e3]
    omega
  · intro hq1 hq2
    obtain ⟨_, hv, e4⟩ := step4 hq1 hq2
    rw [e4]
    omega
  · intro hq1 hq2 hq3
    obtain ⟨_, hv, e5⟩ := step5 hq1 hq2 hq3
    rw [e5]
    omega
  · intro hq1 hq2 hq3 hq4
    obtain ⟨hv, e6⟩ := step6 hq1 hq2 hq3 hq4
    rw [e6]
    omega
  · rintro ⟨hr0, hr1, hr2, hr3, hr4⟩
    obtain ⟨hk3le, hv3, e3⟩ := step3 hr1
    obtain ⟨hk4le, hv4, e4⟩ := step4 hr1 hr2
    obtain ⟨hk5le, hv5, e5⟩ := step5 hr1 hr2 hr3
    have hg3v : g23 (n + 2) = o3 - o2 := by
      simp only [g23]
      rw [show n + 2 + 1 = n + 3 by omega, e3, e2]
      omega
    have hg4v : g23 (n + 3) = o4 - o3 := by
      simp only [g23]
      rw [show n + 3 + 1 = n + 4 by omega, e4, e3]
      omega
    have hg5v : g23 (n + 4) = o5 - o4 := by
      simp only [g23]
      rw [show n + 4 + 1 = n + 5 by omega, e5, e4]
      omega
    have hb0 : Nat.ble 10 o1 = true := Nat.ble_eq.mpr (by omega)
    have hb1 : Nat.ble 10 (o2 - o1) = true := Nat.ble_eq.mpr (by omega)
    have hb2 : Nat.ble 10 (o3 - o2) = true := Nat.ble_eq.mpr (by omega)
    have hb3 : Nat.ble 10 (o4 - o3) = true := Nat.ble_eq.mpr (by omega)
    have hb4 : Nat.ble 10 (o5 - o4) = true := Nat.ble_eq.mpr (by omega)
    rw [hb0, hb1, hb2, hb3, hb4] at hC7
    simp at hC7

/-! ## The spectrum facts, and the rung -/

/-- **`F(23) <= 34`**, from the position-indexed scan. -/
theorem spectrum23_one : Spectrum.SpectrumBound g23 1 34 := by
  intro a
  rw [windowSum_g23]
  exact (chain_facts23 a).1

/-- **`F_2(23) <= 39`** - the first hypothesis of the 23->29 rung,
discharged. -/
theorem spectrum23_two : Spectrum.SpectrumBound g23 2 39 := by
  intro a
  rw [windowSum_g23]
  exact (chain_facts23 a).2.1

/-- **`Q_j(23; 10) <= 60` for every depth `j >= 3`** (floor `2u'' = 10`,
`u'' = 5`, gear 29) - the second hypothesis of the 23->29 rung, discharged:
depths 3-6 from the guarded rungs, and NO qualifying window of depth
`j >= 7` exists at all (the five-run refutation). -/
theorem qual23_all : ∀ j, 3 ≤ j → Spectrum.QualBound g23 5 j 60 := by
  intro j hj a hq
  rcases Nat.lt_or_ge j 7 with hj7 | hj7
  · rw [windowSum_g23]
    interval_cases j
    · exact (chain_facts23 a).2.2.1 (by simpa using hq 1 (by omega) (by omega))
    · exact (chain_facts23 a).2.2.2.1
        (by simpa using hq 1 (by omega) (by omega))
        (by simpa using hq 2 (by omega) (by omega))
    · exact (chain_facts23 a).2.2.2.2.1
        (by simpa using hq 1 (by omega) (by omega))
        (by simpa using hq 2 (by omega) (by omega))
        (by simpa using hq 3 (by omega) (by omega))
    · exact (chain_facts23 a).2.2.2.2.2.1
        (by simpa using hq 1 (by omega) (by omega))
        (by simpa using hq 2 (by omega) (by omega))
        (by simpa using hq 3 (by omega) (by omega))
        (by simpa using hq 4 (by omega) (by omega))
  · exfalso
    refine (chain_facts23 (a + 1)).2.2.2.2.2.2 ⟨?_, ?_, ?_, ?_, ?_⟩
    · simpa using hq 1 (by omega) (by omega)
    · have := hq 2 (by omega) (by omega)
      rw [show a + 2 = a + 1 + 1 by omega] at this
      simpa using this
    · have := hq 3 (by omega) (by omega)
      rw [show a + 3 = a + 1 + 2 by omega] at this
      simpa using this
    · have := hq 4 (by omega) (by omega)
      rw [show a + 4 = a + 1 + 3 by omega] at this
      simpa using this
    · have := hq 5 (by omega) (by omega)
      rw [show a + 5 = a + 1 + 4 by omega] at this
      simpa using this

/-- **THE FIFTH RUNG, HYPOTHESIS-FREE: (D) at `alpha = 3` at the 23->29
step.**  Every gap of machine 29 is at most `F(23) + 29 = 34 + 29 = 63`.
Both hypotheses of `Machine29.D_at_23_29` are now kernel facts. -/
theorem D_23_29 (n : ℕ) : Machine29.g29 n ≤ 34 + 29 :=
  Machine29.D_at_23_29 spectrum23_two qual23_all n

/-- **R39's own form at the 23->29 step, hypothesis-free**: every gap of
machine 29 is at most `max (F_2, max_j Q_j) = 60 < 63` (margin 3). -/
theorem g29_le_60 (n : ℕ) : Machine29.g29 n ≤ 60 :=
  Machine29.g29_le spectrum23_two qual23_all n

end Machine23
