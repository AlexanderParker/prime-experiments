/-
Machine 11 (gears `{5, 7, 11}`, period 385) - the bottom rung of the (D)
ladder (round 22).

The rung `11 -> 13`. Gear 13 has teeth at slot residues `u' = 2` and
`13 - u' = 11`, so the qualifying floor is `2u' = 4` and the tolerance
budget is `F(11) + q' = 7 + 13 = 20`. `MergeLaw.newgap_le` needs, of
machine 11 alone,

    F_2 <= 20   and   Q_j(11; 4) <= 20 for every depth j >= 3.

Full-period Python first (scratchpad ladder_verify.py, all 385 residues):
F_1..F_8(11) = 7, 11, 16, 18, 23, 26, 28, 30 and
Q_j(11; 4) = 16, 18, 20, 0, 0, ... - depths 3 and 4 are covered by the
UNCONDITIONAL bounds `F_3 <= 16`, `F_4 <= 18`; depth 5 is the first that
needs the qualifying restriction (`F_5 = 23 > 20` but `Q_5 = 20`, EXACTLY
the budget - this rung is TIGHT); and every depth `j >= 6` is discharged by
one refutation: no four consecutive gaps are all `>= 4` (longest run 3).

All six facts are read off ONE five-step `seekT` walk per opening (round
21's encoding); the first check `o1 <= 7` re-derives `F_1(11) <= 7` from the
same walk, which is what makes fuel 7 provably sufficient. The whole period
is a single 385-tuple kernel check.
-/

import Corridor
import Spectrum

namespace Machine11

/-! ## The period scan -/

/-- Opening test on the CRT tuple `(k%5, k%7, k%11)`. Teeth: gear 5 at
`{1, 4}`, gear 7 at `{6, 1}`, gear 11 at `{2, 9}`. -/
def expT (a b c : Nat) : Bool :=
  a != 1 && a != 4 && b != 6 && b != 1 && c != 2 && c != 9

/-- The test `n` slots further on. -/
def atT (a b c n : Nat) : Bool := expT ((a + n) % 5) ((b + n) % 7) ((c + n) % 11)

/-- First offset `t > s` with `atT ... t = true`, searched with `fu` slots of
fuel; `s + 999` if the fuel runs out. At an opening the sentinel is
unreachable: machine-11 gaps cap at 7. -/
def seekT (a b c : Nat) : Nat → Nat → Nat
  | 0, s => s + 999
  | fu + 1, s => if atT a b c (s + 1) then s + 1 else seekT a b c fu (s + 1)

/-- The five-opening chain check from an opening: `F_1 <= 7`, `F_2 <= 11`,
`F_3 <= 16`, `F_4 <= 18`, the qualifying depth-5 bound `Q_5 <= 20` (only
when the three interior gaps all meet the floor 4), and the refutation of
any four-in-a-row run of gaps `>= 4` (`Q_j(11; 4) = 0` for every `j >= 6`). -/
def chainT (a b c : Nat) : Bool :=
  let o1 := seekT a b c 7 0
  let o2 := seekT a b c 7 o1
  let o3 := seekT a b c 7 o2
  let o4 := seekT a b c 7 o3
  let o5 := seekT a b c 7 o4
  Nat.ble o1 7 &&
    (Nat.ble o2 11 &&
      (Nat.ble o3 16 &&
        (Nat.ble o4 18 &&
          ((!(Nat.ble 4 (o2 - o1) && Nat.ble 4 (o3 - o2) && Nat.ble 4 (o4 - o3))
              || Nat.ble o5 20) &&
            !(Nat.ble 4 o1 && Nat.ble 4 (o2 - o1) && Nat.ble 4 (o3 - o2) &&
              Nat.ble 4 (o4 - o3))))))

/-- From an opening, the chain facts hold; non-openings are skipped. -/
def qokT (a b c : Nat) : Bool := !(atT a b c 0) || chainT a b c

/-- The whole period: all 385 CRT tuples. -/
def qslice : Bool :=
  (List.range 5).all fun a => (List.range 7).all fun b =>
    (List.range 11).all fun c => qokT a b c

set_option maxRecDepth 40000 in
/-- **One period, kernel-checked.** -/
theorem qasm : qslice = true := by decide +kernel

/-- The tuple-level fact, unpacked from the slice. -/
theorem qokAll {a b c : ℕ} (ha : a < 5) (hb : b < 7) (hc : c < 11) :
    qokT a b c = true := by
  have h := qasm
  rw [qslice, List.all_eq_true] at h
  have h1 := h a (List.mem_range.mpr ha)
  rw [List.all_eq_true] at h1
  have h2 := h1 b (List.mem_range.mpr hb)
  rw [List.all_eq_true] at h2
  exact h2 c (List.mem_range.mpr hc)

/-! ## Openings -/

/-- An opening of machine 11: no gear in `{5, 7, 11}` divides either member
of slot `k`. -/
def Exposed11 (k : ℕ) : Prop :=
  ¬ (5 ∣ Census.lo k) ∧ ¬ (5 ∣ Census.hi k) ∧
    ¬ (7 ∣ Census.lo k) ∧ ¬ (7 ∣ Census.hi k) ∧
    ¬ (11 ∣ Census.lo k) ∧ ¬ (11 ∣ Census.hi k)

instance (k : ℕ) : Decidable (Exposed11 k) := by unfold Exposed11; infer_instance

/-- Openings are exactly the CRT-tuple test. -/
theorem exposed11_iff {k : ℕ} (hk : 1 ≤ k) :
    Exposed11 k ↔ expT (k % 5) (k % 7) (k % 11) = true := by
  have h5lo : (5 ∣ Census.lo k) ↔ k % 5 = 1 := by simp only [Census.lo]; omega
  have h5hi : (5 ∣ Census.hi k) ↔ k % 5 = 4 := by simp only [Census.hi]; omega
  have h7lo : (7 ∣ Census.lo k) ↔ k % 7 = 6 := by simp only [Census.lo]; omega
  have h7hi : (7 ∣ Census.hi k) ↔ k % 7 = 1 := by simp only [Census.hi]; omega
  have h11lo : (11 ∣ Census.lo k) ↔ k % 11 = 2 := by simp only [Census.lo]; omega
  have h11hi : (11 ∣ Census.hi k) ↔ k % 11 = 9 := by simp only [Census.hi]; omega
  unfold Exposed11
  rw [h5lo, h5hi, h7lo, h7hi, h11lo, h11hi]
  simp only [expT, Bool.and_eq_true, bne_iff_ne, ne_eq, and_assoc]

/-- Shifted form. -/
theorem atT_iff {k : ℕ} (hk : 1 ≤ k) (n : ℕ) :
    atT (k % 5) (k % 7) (k % 11) n = true ↔ Exposed11 (k + n) := by
  rw [atT, exposed11_iff (show 1 ≤ k + n by omega)]
  have e5 : (k % 5 + n) % 5 = (k + n) % 5 := by omega
  have e7 : (k % 7 + n) % 7 = (k + n) % 7 := by omega
  have e11 : (k % 11 + n) % 11 = (k + n) % 11 := by omega
  rw [e5, e7, e11]

/-- Machine-11 openings are (5,7)-corridor openings. -/
theorem exposed11_exposed {k : ℕ} (h : Exposed11 k) : Corridor.Exposed k :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1⟩

/-! ## The machine's own gap sequence -/

/-- Multiples of the period 385 are openings, so an opening exists above any
point. -/
theorem exists_exposed_above (k : ℕ) : ∃ m, k < m ∧ Exposed11 m := by
  refine ⟨385 * (k + 1), by omega, ?_⟩
  rw [exposed11_iff (by omega)]
  have h5 : (385 * (k + 1)) % 5 = 0 := by omega
  have h7 : (385 * (k + 1)) % 7 = 0 := by omega
  have h11 : (385 * (k + 1)) % 11 = 0 := by omega
  rw [h5, h7, h11]
  decide

/-- The next opening strictly after `k`. -/
def nextOp (k : ℕ) : ℕ := Nat.find (exists_exposed_above k)

theorem nextOp_gt (k : ℕ) : k < nextOp k :=
  (Nat.find_spec (exists_exposed_above k)).1

theorem nextOp_exposed (k : ℕ) : Exposed11 (nextOp k) :=
  (Nat.find_spec (exists_exposed_above k)).2

theorem nextOp_min {k m : ℕ} (h1 : k < m) (h2 : m < nextOp k) :
    ¬ Exposed11 m := fun hE =>
  Nat.find_min (exists_exposed_above k) h2 ⟨h1, hE⟩

/-- The opening sequence of machine 11, in increasing order. -/
def opSeq : ℕ → ℕ
  | 0 => nextOp 0
  | n + 1 => nextOp (opSeq n)

theorem opSeq_succ (n : ℕ) : opSeq (n + 1) = nextOp (opSeq n) := rfl

theorem opSeq_exposed (n : ℕ) : Exposed11 (opSeq n) := by
  cases n <;> exact nextOp_exposed _

theorem opSeq_lt_succ (n : ℕ) : opSeq n < opSeq (n + 1) := nextOp_gt _

theorem opSeq_pos (n : ℕ) : 1 ≤ opSeq n := by
  cases n with
  | zero => exact nextOp_gt 0
  | succ m =>
    have h1 := nextOp_gt (opSeq m)
    have h2 : opSeq (m + 1) = nextOp (opSeq m) := rfl
    omega

theorem opSeq_le_add (a j : ℕ) : opSeq a ≤ opSeq (a + j) := by
  induction j with
  | zero => rfl
  | succ j ih =>
    have h := opSeq_lt_succ (a + j)
    have he : a + (j + 1) = (a + j) + 1 := by omega
    rw [he]
    omega

/-- No opening sits strictly between consecutive members of `opSeq`. -/
theorem opSeq_gap_empty (n : ℕ) :
    ∀ j, opSeq n < j → j < opSeq (n + 1) → ¬ Exposed11 j :=
  fun _j h1 h2 => nextOp_min h1 h2

/-- **The gap word of machine 11.** -/
def g11 (n : ℕ) : ℕ := opSeq (n + 1) - opSeq n

/-- Window sums of the gap word telescope to opening differences. -/
theorem windowSum_g11 (a j : ℕ) :
    Spectrum.windowSum g11 a j = opSeq (a + j) - opSeq a := by
  induction j with
  | zero => simp [Spectrum.windowSum]
  | succ j ih =>
    have hs : Spectrum.windowSum g11 a (j + 1)
        = Spectrum.windowSum g11 a j + g11 (a + j) := Finset.sum_range_succ _ _
    have h1 := opSeq_le_add a j
    have h2 := opSeq_lt_succ (a + j)
    have he : a + (j + 1) = (a + j) + 1 := by omega
    rw [hs, ih, g11, he]
    omega

/-! ## The seek walk, related to `nextOp` -/

theorem seekT_succ_pos {a b c fu s : ℕ} (h : atT a b c (s + 1) = true) :
    seekT a b c (fu + 1) s = s + 1 := by
  simp only [seekT]
  split
  · rfl
  · rename_i hneg
    exact absurd h hneg

theorem seekT_succ_neg {a b c fu s : ℕ} (h : ¬ atT a b c (s + 1) = true) :
    seekT a b c (fu + 1) s = seekT a b c fu (s + 1) := by
  simp only [seekT]
  split
  · rename_i hpos
    exact absurd hpos h
  · rfl

theorem seekT_gt (a b c : ℕ) : ∀ fu s, s < seekT a b c fu s := by
  intro fu
  induction fu with
  | zero => intro s; simp only [seekT]; omega
  | succ fu ih =>
    intro s
    by_cases h : atT a b c (s + 1) = true
    · rw [seekT_succ_pos h]; omega
    · rw [seekT_succ_neg h]
      have := ih (s + 1); omega

theorem seekT_found (a b c : ℕ) :
    ∀ fu s t, s < t → t ≤ s + fu → atT a b c t = true →
      seekT a b c fu s ≤ s + fu := by
  intro fu
  induction fu with
  | zero => intro s t h1 h2 _; omega
  | succ fu ih =>
    intro s t h1 h2 hat
    by_cases h : atT a b c (s + 1) = true
    · rw [seekT_succ_pos h]; omega
    · rw [seekT_succ_neg h]
      have hne : t ≠ s + 1 := by intro he; rw [he] at hat; exact h hat
      have := ih (s + 1) t (by omega) (by omega) hat
      omega

theorem seekT_exposed (a b c : ℕ) :
    ∀ fu s, seekT a b c fu s ≤ s + fu → atT a b c (seekT a b c fu s) = true := by
  intro fu
  induction fu with
  | zero => intro s h; simp only [seekT] at h; omega
  | succ fu ih =>
    intro s h
    by_cases hat : atT a b c (s + 1) = true
    · rw [seekT_succ_pos hat]; exact hat
    · rw [seekT_succ_neg hat] at h ⊢
      exact ih (s + 1) (by omega)

theorem seekT_min (a b c : ℕ) :
    ∀ fu s t, s < t → t < seekT a b c fu s → t ≤ s + fu → atT a b c t = false := by
  intro fu
  induction fu with
  | zero => intro s t h1 _ h3; omega
  | succ fu ih =>
    intro s t h1 h2 h3
    by_cases hat : atT a b c (s + 1) = true
    · rw [seekT_succ_pos hat] at h2; omega
    · rw [seekT_succ_neg hat] at h2
      rcases Nat.lt_or_ge t (s + 2) with hlt | hge
      · have he : t = s + 1 := by omega
        subst he
        simpa using hat
      · exact ih (s + 1) t (by omega) h2 (by omega)

/-! ## The first scan fact: `F_1(11) <= 7`, and the walk is `nextOp` -/

/-- The `o1` check of the scan, at an opening: the walk's first step lands
within 7 slots. -/
theorem seek_one_le {x : ℕ} (hx : 1 ≤ x) (hE : Exposed11 x) :
    seekT (x % 5) (x % 7) (x % 11) 7 0 ≤ 7 := by
  have ha0 : atT (x % 5) (x % 7) (x % 11) 0 = true :=
    (atT_iff hx 0).mpr (by simpa using hE)
  have h := qokAll (a := x % 5) (b := x % 7) (c := x % 11)
    (by omega) (by omega) (by omega)
  rw [qokT, ha0] at h
  simp only [Bool.not_true, Bool.false_or, chainT, Bool.and_eq_true,
    Nat.ble_eq] at h
  exact h.1

/-- **`F_1(11) <= 7`**: the next opening after an opening arrives within 7
slots - read off the same walk that carries the rest of the scan. -/
theorem nextOp_le_7 {x : ℕ} (hx : 1 ≤ x) (hE : Exposed11 x) : nextOp x ≤ x + 7 := by
  have h1 := seek_one_le hx hE
  have h2 := seekT_exposed (x % 5) (x % 7) (x % 11) 7 0 (by omega)
  have h3 : Exposed11 (x + seekT (x % 5) (x % 7) (x % 11) 7 0) := (atT_iff hx _).mp h2
  have h4 := seekT_gt (x % 5) (x % 7) (x % 11) 7 0
  have := Nat.find_min' (exists_exposed_above x)
    (show x < x + seekT (x % 5) (x % 7) (x % 11) 7 0 ∧
      Exposed11 (x + seekT (x % 5) (x % 7) (x % 11) 7 0) from ⟨by omega, h3⟩)
  simp only [nextOp]
  omega

/-- **The seek walk computes `nextOp`.** -/
theorem seek_next {x s : ℕ} (hx : 1 ≤ x) (hE : Exposed11 (x + s)) :
    x + seekT (x % 5) (x % 7) (x % 11) 7 s = nextOp (x + s) := by
  have hE1 : 1 ≤ x + s := by omega
  have hnle : nextOp (x + s) ≤ x + s + 7 := nextOp_le_7 hE1 hE
  have hngt : x + s < nextOp (x + s) := nextOp_gt _
  have hnE : Exposed11 (nextOp (x + s)) := nextOp_exposed _
  have hat : atT (x % 5) (x % 7) (x % 11) (nextOp (x + s) - x) = true := by
    apply (atT_iff hx _).mpr
    rwa [show x + (nextOp (x + s) - x) = nextOp (x + s) by omega]
  have hfound := seekT_found (x % 5) (x % 7) (x % 11) 7 s
    (nextOp (x + s) - x) (by omega) (by omega) hat
  have hσat := seekT_exposed (x % 5) (x % 7) (x % 11) 7 s hfound
  have hσE : Exposed11 (x + seekT (x % 5) (x % 7) (x % 11) 7 s) :=
    (atT_iff hx _).mp hσat
  have hσgt := seekT_gt (x % 5) (x % 7) (x % 11) 7 s
  have hle1 : nextOp (x + s) ≤ x + seekT (x % 5) (x % 7) (x % 11) 7 s :=
    Nat.find_min' (exists_exposed_above (x + s)) ⟨by omega, hσE⟩
  rcases eq_or_lt_of_le hle1 with he | hlt
  · omega
  · exfalso
    have hmin := seekT_min (x % 5) (x % 7) (x % 11) 7 s
      (nextOp (x + s) - x) (by omega) (by omega) (by omega)
    rw [hmin] at hat
    exact Bool.noConfusion hat

/-! ## The chain facts -/

/-- **The six scan facts, over the enumeration.** -/
theorem chain_facts (n : ℕ) :
    opSeq (n + 1) - opSeq n ≤ 7 ∧ opSeq (n + 2) - opSeq n ≤ 11 ∧
      opSeq (n + 3) - opSeq n ≤ 16 ∧ opSeq (n + 4) - opSeq n ≤ 18 ∧
      ((4 ≤ g11 (n + 1) ∧ 4 ≤ g11 (n + 2) ∧ 4 ≤ g11 (n + 3)) →
        opSeq (n + 5) - opSeq n ≤ 20) ∧
      ¬ (4 ≤ g11 n ∧ 4 ≤ g11 (n + 1) ∧ 4 ≤ g11 (n + 2) ∧ 4 ≤ g11 (n + 3)) := by
  have hx : 1 ≤ opSeq n := opSeq_pos n
  have hE : Exposed11 (opSeq n) := opSeq_exposed n
  have ha0 : atT (opSeq n % 5) (opSeq n % 7) (opSeq n % 11) 0 = true :=
    (atT_iff hx 0).mpr (by simpa using hE)
  have h := qokAll (a := opSeq n % 5) (b := opSeq n % 7) (c := opSeq n % 11)
    (by omega) (by omega) (by omega)
  rw [qokT, ha0] at h
  simp only [Bool.not_true, Bool.false_or] at h
  simp only [chainT] at h
  set o1 := seekT (opSeq n % 5) (opSeq n % 7) (opSeq n % 11) 7 0 with ho1
  set o2 := seekT (opSeq n % 5) (opSeq n % 7) (opSeq n % 11) 7 o1 with ho2
  set o3 := seekT (opSeq n % 5) (opSeq n % 7) (opSeq n % 11) 7 o2 with ho3
  set o4 := seekT (opSeq n % 5) (opSeq n % 7) (opSeq n % 11) 7 o3 with ho4
  set o5 := seekT (opSeq n % 5) (opSeq n % 7) (opSeq n % 11) 7 o4 with ho5
  simp only [Bool.and_eq_true, Nat.ble_eq, Bool.or_eq_true,
    Bool.not_eq_true'] at h
  obtain ⟨h7, h11, h16, h18, hq5, hrun⟩ := h
  -- the chain equations
  have hE0 : Exposed11 (opSeq n + 0) := by simpa using hE
  have e1 : opSeq n + o1 = opSeq (n + 1) := by
    rw [opSeq_succ]
    have h1 := seek_next hx hE0
    simpa [← ho1] using h1
  have hEo1 : Exposed11 (opSeq n + o1) := by rw [e1]; exact opSeq_exposed _
  have e2 : opSeq n + o2 = opSeq (n + 2) := by
    rw [show n + 2 = (n + 1) + 1 by omega, opSeq_succ]
    have h2 := seek_next hx hEo1
    rw [e1] at h2
    simpa [← ho2] using h2
  have hEo2 : Exposed11 (opSeq n + o2) := by rw [e2]; exact opSeq_exposed _
  have e3 : opSeq n + o3 = opSeq (n + 3) := by
    rw [show n + 3 = (n + 2) + 1 by omega, opSeq_succ]
    have h3 := seek_next hx hEo2
    rw [e2] at h3
    simpa [← ho3] using h3
  have hEo3 : Exposed11 (opSeq n + o3) := by rw [e3]; exact opSeq_exposed _
  have e4 : opSeq n + o4 = opSeq (n + 4) := by
    rw [show n + 4 = (n + 3) + 1 by omega, opSeq_succ]
    have h4 := seek_next hx hEo3
    rw [e3] at h4
    simpa [← ho4] using h4
  have hEo4 : Exposed11 (opSeq n + o4) := by rw [e4]; exact opSeq_exposed _
  have e5 : opSeq n + o5 = opSeq (n + 5) := by
    rw [show n + 5 = (n + 4) + 1 by omega, opSeq_succ]
    have h5 := seek_next hx hEo4
    rw [e4] at h5
    simpa [← ho5] using h5
  -- the gaps, in terms of the walk
  have g0 : g11 n = opSeq (n + 1) - opSeq n := rfl
  have g1 : g11 (n + 1) = opSeq (n + 2) - opSeq (n + 1) := by simp only [g11]
  have g2 : g11 (n + 2) = opSeq (n + 3) - opSeq (n + 2) := by simp only [g11]
  have g3 : g11 (n + 3) = opSeq (n + 4) - opSeq (n + 3) := by simp only [g11]
  refine ⟨by omega, by omega, by omega, by omega, ?_, ?_⟩
  · rintro ⟨c1, c2, c3⟩
    have hbt : (Nat.ble 4 (o2 - o1) && Nat.ble 4 (o3 - o2) &&
        Nat.ble 4 (o4 - o3)) = true := by
      simp only [Bool.and_eq_true, Nat.ble_eq]
      refine ⟨⟨?_, ?_⟩, ?_⟩ <;> omega
    rcases hq5 with hfalse | hle
    · rw [hbt] at hfalse; exact Bool.noConfusion hfalse
    · omega
  · rintro ⟨c0, c1, c2, c3⟩
    have hbt : (Nat.ble 4 o1 && Nat.ble 4 (o2 - o1) && Nat.ble 4 (o3 - o2) &&
        Nat.ble 4 (o4 - o3)) = true := by
      simp only [Bool.and_eq_true, Nat.ble_eq]
      refine ⟨⟨⟨?_, ?_⟩, ?_⟩, ?_⟩ <;> omega
    rw [hbt] at hrun
    exact Bool.noConfusion hrun

/-- **`Q_j(11; 4) = 0` for `j >= 6`**: no four consecutive gaps of `g11` all
meet the qualifying floor 4. -/
theorem no_big_run (n : ℕ) :
    ¬ (4 ≤ g11 n ∧ 4 ≤ g11 (n + 1) ∧ 4 ≤ g11 (n + 2) ∧ 4 ≤ g11 (n + 3)) :=
  (chain_facts n).2.2.2.2.2

/-! ## The spectrum ladder over the gap word -/

/-- `F_1(11) <= 7`. -/
theorem spectrum_one : Spectrum.SpectrumBound g11 1 7 := by
  intro a; rw [windowSum_g11]; exact (chain_facts a).1

/-- `F_2(11) <= 11`. -/
theorem spectrum_two : Spectrum.SpectrumBound g11 2 11 := by
  intro a; rw [windowSum_g11]; exact (chain_facts a).2.1

/-- `F_3(11) <= 16`. -/
theorem spectrum_three : Spectrum.SpectrumBound g11 3 16 := by
  intro a; rw [windowSum_g11]; exact (chain_facts a).2.2.1

/-- `F_4(11) <= 18`. -/
theorem spectrum_four : Spectrum.SpectrumBound g11 4 18 := by
  intro a; rw [windowSum_g11]; exact (chain_facts a).2.2.2.1

/-- **The kernel-fed spectrum ladder of machine 11**: `F_1..F_4 <= 7, 11,
16, 18`. -/
theorem spectrum_ladder :
    Spectrum.SpectrumBound g11 1 7 ∧ Spectrum.SpectrumBound g11 2 11 ∧
      Spectrum.SpectrumBound g11 3 16 ∧ Spectrum.SpectrumBound g11 4 18 :=
  ⟨spectrum_one, spectrum_two, spectrum_three, spectrum_four⟩

/-! ## The qualifying spectrum, closed at every depth -/

/-- **`Q_j(11; 4) <= 20` for every depth `j >= 3`** (floor `2u' = 4`, i.e.
`u' = 2`, gear 13): depths 3 and 4 from the unconditional ladder, depth 5
from the qualifying scan fact (`Q_5 = 20` EXACTLY - this rung is tight), and
NO qualifying window of depth 6 or more exists at all (`no_big_run`). -/
theorem qual_bound_all : ∀ j, 3 ≤ j → Spectrum.QualBound g11 2 j 20 := by
  intro j hj a hq
  rcases Nat.lt_or_ge j 6 with hj6 | hj6
  · interval_cases j
    · exact le_trans (spectrum_three a) (by omega)
    · exact le_trans (spectrum_four a) (by omega)
    · rw [windowSum_g11]
      refine (chain_facts a).2.2.2.2.1 ⟨?_, ?_, ?_⟩
      · have h1 := hq 1 (by omega) (by omega); omega
      · have h2 := hq 2 (by omega) (by omega)
        rw [show a + 2 = a + 1 + 1 by omega] at h2; omega
      · have h3 := hq 3 (by omega) (by omega)
        rw [show a + 3 = a + 1 + 2 by omega] at h3; omega
  · exfalso
    refine no_big_run (a + 1) ⟨?_, ?_, ?_, ?_⟩
    · have h1 := hq 1 (by omega) (by omega); omega
    · have h2 := hq 2 (by omega) (by omega)
      rw [show a + 1 + 1 = a + 2 by omega]; omega
    · have h3 := hq 3 (by omega) (by omega)
      rw [show a + 1 + 2 = a + 3 by omega]; omega
    · have h4 := hq 4 (by omega) (by omega)
      rw [show a + 1 + 3 = a + 4 by omega]; omega

/-! ## The opening enumeration is complete -/

theorem opSeq_strict_mono {a b : ℕ} (h : a < b) : opSeq a < opSeq b := by
  have h1 := opSeq_lt_succ a
  have h2 := opSeq_le_add (a + 1) (b - (a + 1))
  rw [show a + 1 + (b - (a + 1)) = b by omega] at h2
  omega

theorem opSeq_reach : ∀ dd A, 1 ≤ A → Exposed11 A → (∃ n, opSeq n = A) →
    ∀ B, Exposed11 B → A < B → B - A ≤ dd → ∃ m, opSeq m = B := by
  intro dd
  induction dd with
  | zero => intro A _ _ _ B _ hAB hd; omega
  | succ dd ih =>
    rintro A hA1 hEA ⟨n, hn⟩ B hEB hAB hd
    have hnext_le : nextOp A ≤ B :=
      Nat.find_min' (exists_exposed_above A) ⟨hAB, hEB⟩
    have hgt := nextOp_gt A
    rcases eq_or_lt_of_le hnext_le with he | hlt
    · exact ⟨n + 1, by rw [opSeq_succ, hn, he]⟩
    · exact ih (nextOp A) (by omega) (nextOp_exposed A)
        ⟨n + 1, by rw [opSeq_succ, hn]⟩ B hEB hlt (by omega)

/-- **Every opening is enumerated**: `opSeq` is onto the openings. -/
theorem opSeq_surj {m : ℕ} (hm : 1 ≤ m) (hE : Exposed11 m) : ∃ n, opSeq n = m := by
  have h0 : opSeq 0 = nextOp 0 := rfl
  have hle : nextOp 0 ≤ m := Nat.find_min' (exists_exposed_above 0) ⟨by omega, hE⟩
  rcases eq_or_lt_of_le hle with he | hlt
  · exact ⟨0, by rw [h0, he]⟩
  · exact opSeq_reach (m - nextOp 0) (nextOp 0) (by have := nextOp_gt 0; omega)
      (nextOp_exposed 0) ⟨0, rfl⟩ m hE hlt (by omega)

end Machine11
