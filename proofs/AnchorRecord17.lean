/-
THE PHASE-REDUCTION RECORD LAW AT MACHINE 17 - the bridge to the corpus.

`AnchorRecord17Core` computes, from the 1485 openings of `{5, 7, 11, 13}` on
the period 5005 alone, that the largest merged gap over all 17 deletion phases
is 18.  This file connects that number to the machine the corpus already knows:

* `surv_shift` - **the phase reduction itself**: for every phase `r < 17` the
  phase-`r` survivor set is machine 17's OWN opening set, shifted by
  `tOf r * 5005` (one whole lower period per copy).  `t = (14 - r) * 5 mod 17`
  because `5005 = 7 mod 17` and `7⁻¹ = 5 mod 17`.  This is the Lean form of
  `anchor-235.md` 9f's "the `g` copies realise every deletion phase in `Z_g`
  exactly once"; the abstract statement, for every gear and every lower period,
  is `AnchorChain.copy_phase` + `AnchorChain.phase_bijective`.
* `phase_is_machine` - the same statement against `Machine17.Exposed17`.
* `gap18_realized` / `F17_eq_18` - and since the phase table's maximum is
  attained, machine 17's record is EXACTLY 18: the corpus had `<= 18`
  (`Machine17.gap_le`) and no attainment theorem, so this pins it.

HONEST SCOPE.  What is kernel-proved here is (i) the phase table, (ii) the
shift bijection between a phase and the machine, and (iii) `F(17) = 18` exactly.
What is NOT kernel-proved is that `mg` is a correct max-gap oracle - i.e. the
identity "max over phases = F(17) + 1" is verified in the kernel at BOTH ends
(the table by `AnchorRecord17Core`, the machine record by `gap18_realized` plus
`Machine17.gap_le`) rather than derived from one to the other, because that
would need a correctness proof of the walk against `Machine17.nextOp`.  Both
ends give 18; `research/anchor235/r29_record17_gate.py` gates the same identity
outside the kernel.
-/

import Machine17
import AnchorRecord17Core

namespace AnchorRecord17

/-- Machine 17's opening test, written in `surv`'s shape.  Gear 17's teeth are
`{3, 14}` and `{14, (14 + 6) % 17} = {14, 3}`, so machine 17 IS phase 14. -/
def openT17 (y : Nat) : Bool :=
  lowOpen (y % 5005) && y % 17 != 14 && y % 17 != 3

theorem surv_fourteen (y : Nat) : surv 14 y = openT17 y := rfl

/-- The copy shift carrying phase `r` onto machine 17. -/
def tOf (r : Nat) : Nat := ((31 - r) * 5) % 17

/-- The residue half of the shift: all 17 phases, one kernel check. -/
theorem shift_res : ∀ r, r < 17 → ∀ m, m < 17 →
    (((m + (tOf r * 5005) % 17) % 17 != 14) = (m != r)) ∧
    (((m + (tOf r * 5005) % 17) % 17 != 3) = (m != (r + 6) % 17)) := by decide

/-- **THE PHASE REDUCTION.**  The survivors at phase `r` are exactly machine
17's openings in the copy shifted by `tOf r` lower periods. -/
theorem surv_shift {r : Nat} (hr : r < 17) (y : Nat) :
    surv r y = openT17 (y + tOf r * 5005) := by
  have hlow : (y + tOf r * 5005) % 5005 = y % 5005 := Nat.add_mul_mod_self_right ..
  have hres : (y + tOf r * 5005) % 17 = (y % 17 + (tOf r * 5005) % 17) % 17 :=
    Nat.add_mod ..
  have hm : y % 17 < 17 := Nat.mod_lt _ (by norm_num)
  obtain ⟨h1, h2⟩ := shift_res r hr (y % 17) hm
  unfold surv openT17
  rw [hlow, hres, h1, h2]

/-- The Bool test and the corpus predicate agree. -/
theorem openT17_iff {k : ℕ} (hk : 1 ≤ k) :
    openT17 k = true ↔ Machine17.Exposed17 k := by
  have m5 : k % 5005 % 5 = k % 5 := Nat.mod_mod_of_dvd k (by norm_num)
  have m7 : k % 5005 % 7 = k % 7 := Nat.mod_mod_of_dvd k (by norm_num)
  have m11 : k % 5005 % 11 = k % 11 := Nat.mod_mod_of_dvd k (by norm_num)
  have m13 : k % 5005 % 13 = k % 13 := Nat.mod_mod_of_dvd k (by norm_num)
  rw [Machine17.exposed17_iff hk]
  simp only [openT17, lowOpen, Machine17.expT, Bool.and_eq_true, bne_iff_ne,
    ne_eq, m5, m7, m11, m13]
  tauto

/-- The phase reduction against the corpus predicate. -/
theorem phase_is_machine {r : Nat} (hr : r < 17) {y : Nat}
    (hy : 1 ≤ y + tOf r * 5005) :
    surv r y = true ↔ Machine17.Exposed17 (y + tOf r * 5005) := by
  rw [surv_shift hr y, openT17_iff hy]

/-! ## The record is attained -/

/-- Machine 17 realises a gap of 18: openings 117 and 135, nothing between. -/
theorem gap18_realized :
    Machine17.Exposed17 117 ∧ Machine17.Exposed17 135 ∧
      ∀ j, 117 < j → j < 135 → ¬ Machine17.Exposed17 j := by
  refine ⟨by decide, by decide, fun j h1 h2 => ?_⟩
  interval_cases j <;> decide

/-- **`F(17) = 18` EXACTLY.**  `Machine17.gap_le` gave the upper half; the
phase table's maximum is attained, and this is the witness. -/
theorem F17_eq_18 :
    (∀ a b : ℕ, 1 ≤ a → a < b → Machine17.Exposed17 a → Machine17.Exposed17 b →
        (∀ j, a < j → j < b → ¬ Machine17.Exposed17 j) → b - a ≤ 18) ∧
    (∃ a b : ℕ, 1 ≤ a ∧ a < b ∧ Machine17.Exposed17 a ∧ Machine17.Exposed17 b ∧
        (∀ j, a < j → j < b → ¬ Machine17.Exposed17 j) ∧ b - a = 18) :=
  ⟨fun a b ha hab hEa hEb hg => Machine17.gap_le ha hab hEa hEb hg,
   ⟨117, 135, by norm_num, by norm_num, gap18_realized.1, gap18_realized.2.1,
     gap18_realized.2.2, by norm_num⟩⟩

end AnchorRecord17
