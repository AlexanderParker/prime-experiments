/-
THE CRT SLOTS: F_2 RECORDS AS EXPLICIT SLOTS OF THEIR OWN MACHINES
(Formalist, round 30; Mechanic's verdict-36 delivery, research/r29_results.txt (a)).

Each theorem exhibits FIVE CONSECUTIVE OPENINGS of the real machine - the
record's adjacent pair (open at both ends, exactly one interior opening,
every other slot of the span blocked) together with the previous and the
next opening outside it, so the pair is pinned as a maximal adjacent triple
and its two flank gaps are on record too.  The `AdjPair` corollaries are the
round-28 shape: `F_2(M) >= span` as a self-contained realisability statement.

    F_2(37) >= 90   m37  y = 90816580900          word [5, 2, 88, 2]
                    (the LP thread's round-29 (2,88) phase vector, CRT'd to a slot)
    F_2(41) >= 103  m41  y = 21157523372970       word [7, 28, 75, 4]
    F_2(53) >= 159  m53  y = 327666424664536738   word [6, 77, 82, 3]
    F_2(59) >= 173  m59  y = 307199471342884027665   word [13, 100, 73, 4]  (A)
    F_2(59) >= 173  m59  y = 13260587016151412007    word [4, 73, 100, 13]  (B)

SCOPE, exactly as Mechanic separated it: F_2(41) = 103 and F_2(53) = 159 are
exact and unconditional (the upper halves rest on the deletion-ladder caps
F(43) = 103 and F(59) = 161); the kernel carries the LOWER halves here.  For
machine 59 only `>= 173` is unconditional; `F_2(59) <= 173` carries the span
condition "no 2-window of machine 59 has span in (173, 220]" (the round-28
scan's cap, which cannot be retired until F(61) is a number) and is
deliberately NOT stated here.  The two m59 slots are an exact mirror pair,
`y_A + y_B + 173 = P(59)`, with reversed words and reversed flanks
(`mirror_59`, `period_59`).

Every slot fact is `decide +kernel` on the residue test of `MachineUp.lean`
(bignum `%` in the kernel, no `native_decide`), transported to the real
opening predicate by `exposed_q_iff`.
-/

import MachineUp
import Increment

namespace CrtSlots

open MachineUp

/-- `xs` are CONSECUTIVE openings of `E`: each is open and nothing strictly
between successive entries is open. -/
def Consec (E : ℕ → Prop) : List ℕ → Prop
  | a :: b :: l => E a ∧ a < b ∧ (∀ j, a < j → j < b → ¬ E j) ∧ Consec E (b :: l)
  | [a] => E a
  | [] => True

/-- A blocked stretch, from a residue-test `decide` over the offsets. -/
theorem blocked_of_ball {E O : ℕ → Prop} (hEO : ∀ k, 1 ≤ k → (E k ↔ O k)) {a b : ℕ}
    (ha : 1 ≤ a) (h : ∀ t, t < b - a - 1 → ¬ O (a + 1 + t)) :
    ∀ j, a < j → j < b → ¬ E j := by
  intro j h1 h2 hE
  have := h (j - a - 1) (by omega)
  rw [show a + 1 + (j - a - 1) = j by omega] at this
  exact this ((hEO j (by omega)).mp hE)

/-! ## Machine 37: F_2(37) >= 90, the LP thread's (2, 88) witness as a slot -/

theorem open37_w : Open37 90816580895 ∧ Open37 90816580900 ∧ Open37 90816580902 ∧
    Open37 90816580990 ∧ Open37 90816580992 := by decide +kernel

theorem blocked37_w :
    (∀ t, t < 90816580900 - 90816580895 - 1 → ¬ Open37 (90816580895 + 1 + t)) ∧
    (∀ t, t < 90816580902 - 90816580900 - 1 → ¬ Open37 (90816580900 + 1 + t)) ∧
    (∀ t, t < 90816580990 - 90816580902 - 1 → ¬ Open37 (90816580902 + 1 + t)) ∧
    (∀ t, t < 90816580992 - 90816580990 - 1 → ¬ Open37 (90816580990 + 1 + t)) := by
  decide +kernel

/-- Five consecutive openings of machine 37, gap word `[5, 2, 88, 2]`. -/
theorem five_37 : Consec Machine37.Exposed37
    [90816580895, 90816580900, 90816580902, 90816580990, 90816580992] := by
  obtain ⟨o1, o2, o3, o4, o5⟩ := open37_w
  obtain ⟨b1, b2, b3, b4⟩ := blocked37_w
  have e : ∀ k, 1 ≤ k → (Machine37.Exposed37 k ↔ Open37 k) := fun k hk => exposed37_iff hk
  simp only [Consec]
  exact ⟨(e _ (by norm_num)).mpr o1, by norm_num, blocked_of_ball e (by norm_num) b1,
    (e _ (by norm_num)).mpr o2, by norm_num, blocked_of_ball e (by norm_num) b2,
    (e _ (by norm_num)).mpr o3, by norm_num, blocked_of_ball e (by norm_num) b3,
    (e _ (by norm_num)).mpr o4, by norm_num, blocked_of_ball e (by norm_num) b4,
    (e _ (by norm_num)).mpr o5⟩

/-- `F_2(37) >= 90`: openings 90816580900, 90816580902, 90816580990 of machine
37, gaps `(2, 88)` - the project's recorded m37 maximiser, from the LP
thread's scan-free phase vector, now a kernel slot. -/
theorem f2_37 : Increment.AdjPair Machine37.Exposed37 90816580900 90816580902 90816580990 := by
  obtain ⟨_, o2, o3, o4, _⟩ := open37_w
  obtain ⟨_, b2, b3, _⟩ := blocked37_w
  have e : ∀ k, 1 ≤ k → (Machine37.Exposed37 k ↔ Open37 k) := fun k hk => exposed37_iff hk
  exact ⟨by norm_num, by norm_num, by norm_num, (e _ (by norm_num)).mpr o2,
    (e _ (by norm_num)).mpr o3, (e _ (by norm_num)).mpr o4,
    blocked_of_ball e (by norm_num) b2, blocked_of_ball e (by norm_num) b3⟩

/-! ## Machine 41: F_2(41) = 103, the lower half -/

theorem open41_w : Open41 21157523372963 ∧ Open41 21157523372970 ∧ Open41 21157523372998 ∧
    Open41 21157523373073 ∧ Open41 21157523373077 := by decide +kernel

theorem blocked41_w :
    (∀ t, t < 21157523372970 - 21157523372963 - 1 → ¬ Open41 (21157523372963 + 1 + t)) ∧
    (∀ t, t < 21157523372998 - 21157523372970 - 1 → ¬ Open41 (21157523372970 + 1 + t)) ∧
    (∀ t, t < 21157523373073 - 21157523372998 - 1 → ¬ Open41 (21157523372998 + 1 + t)) ∧
    (∀ t, t < 21157523373077 - 21157523373073 - 1 → ¬ Open41 (21157523373073 + 1 + t)) := by
  decide +kernel

/-- Five consecutive openings of machine 41, gap word `[7, 28, 75, 4]`. -/
theorem five_41 : Consec Exposed41
    [21157523372963, 21157523372970, 21157523372998, 21157523373073, 21157523373077] := by
  obtain ⟨o1, o2, o3, o4, o5⟩ := open41_w
  obtain ⟨b1, b2, b3, b4⟩ := blocked41_w
  have e : ∀ k, 1 ≤ k → (Exposed41 k ↔ Open41 k) := fun k hk => exposed41_iff hk
  simp only [Consec]
  exact ⟨(e _ (by norm_num)).mpr o1, by norm_num, blocked_of_ball e (by norm_num) b1,
    (e _ (by norm_num)).mpr o2, by norm_num, blocked_of_ball e (by norm_num) b2,
    (e _ (by norm_num)).mpr o3, by norm_num, blocked_of_ball e (by norm_num) b3,
    (e _ (by norm_num)).mpr o4, by norm_num, blocked_of_ball e (by norm_num) b4,
    (e _ (by norm_num)).mpr o5⟩

/-- **`F_2(41) >= 103`**: openings 21157523372970, 21157523372998, 21157523373073
of machine 41, gaps `(28, 75)`.  Mechanic's round-29 slot; exact and
unconditional both ways (the upper half is `F_2(41) <= F(43) = 103`). -/
theorem f2_41 : Increment.AdjPair Exposed41 21157523372970 21157523372998 21157523373073 := by
  obtain ⟨_, o2, o3, o4, _⟩ := open41_w
  obtain ⟨_, b2, b3, _⟩ := blocked41_w
  have e : ∀ k, 1 ≤ k → (Exposed41 k ↔ Open41 k) := fun k hk => exposed41_iff hk
  exact ⟨by norm_num, by norm_num, by norm_num, (e _ (by norm_num)).mpr o2,
    (e _ (by norm_num)).mpr o3, (e _ (by norm_num)).mpr o4,
    blocked_of_ball e (by norm_num) b2, blocked_of_ball e (by norm_num) b3⟩

theorem span_41 : 21157523373073 - 21157523372970 = 103 := by norm_num

/-! ## Machine 53: F_2(53) = 159, the lower half -/

theorem open53_w : Open53 327666424664536732 ∧ Open53 327666424664536738 ∧
    Open53 327666424664536815 ∧ Open53 327666424664536897 ∧ Open53 327666424664536900 := by
  decide +kernel

theorem blocked53_w :
    (∀ t, t < 327666424664536738 - 327666424664536732 - 1 → ¬ Open53 (327666424664536732 + 1 + t)) ∧
    (∀ t, t < 327666424664536815 - 327666424664536738 - 1 → ¬ Open53 (327666424664536738 + 1 + t)) ∧
    (∀ t, t < 327666424664536897 - 327666424664536815 - 1 → ¬ Open53 (327666424664536815 + 1 + t)) ∧
    (∀ t, t < 327666424664536900 - 327666424664536897 - 1 → ¬ Open53 (327666424664536897 + 1 + t)) := by
  decide +kernel

/-- Five consecutive openings of machine 53, gap word `[6, 77, 82, 3]`. -/
theorem five_53 : Consec Exposed53
    [327666424664536732, 327666424664536738, 327666424664536815, 327666424664536897,
      327666424664536900] := by
  obtain ⟨o1, o2, o3, o4, o5⟩ := open53_w
  obtain ⟨b1, b2, b3, b4⟩ := blocked53_w
  have e : ∀ k, 1 ≤ k → (Exposed53 k ↔ Open53 k) := fun k hk => exposed53_iff hk
  simp only [Consec]
  exact ⟨(e _ (by norm_num)).mpr o1, by norm_num, blocked_of_ball e (by norm_num) b1,
    (e _ (by norm_num)).mpr o2, by norm_num, blocked_of_ball e (by norm_num) b2,
    (e _ (by norm_num)).mpr o3, by norm_num, blocked_of_ball e (by norm_num) b3,
    (e _ (by norm_num)).mpr o4, by norm_num, blocked_of_ball e (by norm_num) b4,
    (e _ (by norm_num)).mpr o5⟩

/-- **`F_2(53) >= 159`**: openings 327666424664536738, 327666424664536815,
327666424664536897 of machine 53, gaps `(77, 82)`.  Exact and unconditional
both ways (the upper half is `F_2(53) <= F(59) = 161 < 200`, round 28). -/
theorem f2_53 : Increment.AdjPair Exposed53 327666424664536738 327666424664536815 327666424664536897 := by
  obtain ⟨_, o2, o3, o4, _⟩ := open53_w
  obtain ⟨_, b2, b3, _⟩ := blocked53_w
  have e : ∀ k, 1 ≤ k → (Exposed53 k ↔ Open53 k) := fun k hk => exposed53_iff hk
  exact ⟨by norm_num, by norm_num, by norm_num, (e _ (by norm_num)).mpr o2,
    (e _ (by norm_num)).mpr o3, (e _ (by norm_num)).mpr o4,
    blocked_of_ball e (by norm_num) b2, blocked_of_ball e (by norm_num) b3⟩

theorem span_53 : 327666424664536897 - 327666424664536738 = 159 := by norm_num

/-! ## Machine 59: F_2(59) >= 173 (the unconditional half), two mirror slots -/

theorem open59_A : Open59 307199471342884027652 ∧ Open59 307199471342884027665 ∧
    Open59 307199471342884027765 ∧ Open59 307199471342884027838 ∧
    Open59 307199471342884027842 := by decide +kernel

theorem blocked59_A :
    (∀ t, t < 307199471342884027665 - 307199471342884027652 - 1 →
      ¬ Open59 (307199471342884027652 + 1 + t)) ∧
    (∀ t, t < 307199471342884027765 - 307199471342884027665 - 1 →
      ¬ Open59 (307199471342884027665 + 1 + t)) ∧
    (∀ t, t < 307199471342884027838 - 307199471342884027765 - 1 →
      ¬ Open59 (307199471342884027765 + 1 + t)) ∧
    (∀ t, t < 307199471342884027842 - 307199471342884027838 - 1 →
      ¬ Open59 (307199471342884027838 + 1 + t)) := by
  decide +kernel

/-- Five consecutive openings of machine 59, gap word `[13, 100, 73, 4]`. -/
theorem five_59_A : Consec Exposed59
    [307199471342884027652, 307199471342884027665, 307199471342884027765,
      307199471342884027838, 307199471342884027842] := by
  obtain ⟨o1, o2, o3, o4, o5⟩ := open59_A
  obtain ⟨b1, b2, b3, b4⟩ := blocked59_A
  have e : ∀ k, 1 ≤ k → (Exposed59 k ↔ Open59 k) := fun k hk => exposed59_iff hk
  simp only [Consec]
  exact ⟨(e _ (by norm_num)).mpr o1, by norm_num, blocked_of_ball e (by norm_num) b1,
    (e _ (by norm_num)).mpr o2, by norm_num, blocked_of_ball e (by norm_num) b2,
    (e _ (by norm_num)).mpr o3, by norm_num, blocked_of_ball e (by norm_num) b3,
    (e _ (by norm_num)).mpr o4, by norm_num, blocked_of_ball e (by norm_num) b4,
    (e _ (by norm_num)).mpr o5⟩

/-- **`F_2(59) >= 173`** (witness A): openings 307199471342884027665,
307199471342884027765, 307199471342884027838 of machine 59, gaps `(100, 73)`.
UNCONDITIONAL.  The matching `<= 173` is NOT stated: it carries the span
condition "no 2-window of machine 59 has span in (173, 220]". -/
theorem f2_59_A : Increment.AdjPair Exposed59 307199471342884027665 307199471342884027765
    307199471342884027838 := by
  obtain ⟨_, o2, o3, o4, _⟩ := open59_A
  obtain ⟨_, b2, b3, _⟩ := blocked59_A
  have e : ∀ k, 1 ≤ k → (Exposed59 k ↔ Open59 k) := fun k hk => exposed59_iff hk
  exact ⟨by norm_num, by norm_num, by norm_num, (e _ (by norm_num)).mpr o2,
    (e _ (by norm_num)).mpr o3, (e _ (by norm_num)).mpr o4,
    blocked_of_ball e (by norm_num) b2, blocked_of_ball e (by norm_num) b3⟩

theorem open59_B : Open59 13260587016151412003 ∧ Open59 13260587016151412007 ∧
    Open59 13260587016151412080 ∧ Open59 13260587016151412180 ∧
    Open59 13260587016151412193 := by decide +kernel

theorem blocked59_B :
    (∀ t, t < 13260587016151412007 - 13260587016151412003 - 1 →
      ¬ Open59 (13260587016151412003 + 1 + t)) ∧
    (∀ t, t < 13260587016151412080 - 13260587016151412007 - 1 →
      ¬ Open59 (13260587016151412007 + 1 + t)) ∧
    (∀ t, t < 13260587016151412180 - 13260587016151412080 - 1 →
      ¬ Open59 (13260587016151412080 + 1 + t)) ∧
    (∀ t, t < 13260587016151412193 - 13260587016151412180 - 1 →
      ¬ Open59 (13260587016151412180 + 1 + t)) := by
  decide +kernel

/-- Five consecutive openings of machine 59, gap word `[4, 73, 100, 13]` - the
mirror image of witness A, flanks included. -/
theorem five_59_B : Consec Exposed59
    [13260587016151412003, 13260587016151412007, 13260587016151412080,
      13260587016151412180, 13260587016151412193] := by
  obtain ⟨o1, o2, o3, o4, o5⟩ := open59_B
  obtain ⟨b1, b2, b3, b4⟩ := blocked59_B
  have e : ∀ k, 1 ≤ k → (Exposed59 k ↔ Open59 k) := fun k hk => exposed59_iff hk
  simp only [Consec]
  exact ⟨(e _ (by norm_num)).mpr o1, by norm_num, blocked_of_ball e (by norm_num) b1,
    (e _ (by norm_num)).mpr o2, by norm_num, blocked_of_ball e (by norm_num) b2,
    (e _ (by norm_num)).mpr o3, by norm_num, blocked_of_ball e (by norm_num) b3,
    (e _ (by norm_num)).mpr o4, by norm_num, blocked_of_ball e (by norm_num) b4,
    (e _ (by norm_num)).mpr o5⟩

/-- **`F_2(59) >= 173`** (witness B), gaps `(73, 100)`.  UNCONDITIONAL. -/
theorem f2_59_B : Increment.AdjPair Exposed59 13260587016151412007 13260587016151412080
    13260587016151412180 := by
  obtain ⟨_, o2, o3, o4, _⟩ := open59_B
  obtain ⟨_, b2, b3, _⟩ := blocked59_B
  have e : ∀ k, 1 ≤ k → (Exposed59 k ↔ Open59 k) := fun k hk => exposed59_iff hk
  exact ⟨by norm_num, by norm_num, by norm_num, (e _ (by norm_num)).mpr o2,
    (e _ (by norm_num)).mpr o3, (e _ (by norm_num)).mpr o4,
    blocked_of_ball e (by norm_num) b2, blocked_of_ball e (by norm_num) b3⟩

theorem span_59_A : 307199471342884027838 - 307199471342884027665 = 173 := by norm_num
theorem span_59_B : 13260587016151412180 - 13260587016151412007 = 173 := by norm_num

/-- The period of machine 59. -/
theorem period_59 :
    [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59].prod = 320460058359035439845 := by
  norm_num

/-- The two m59 slots are an exact MIRROR PAIR: `y_A + y_B + 173 = P(59)`. -/
theorem mirror_59 : 307199471342884027665 + 13260587016151412007 + 173 =
    [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59].prod := by
  rw [period_59]

end CrtSlots
