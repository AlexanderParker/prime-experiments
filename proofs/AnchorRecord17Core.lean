/-
THE PHASE-REDUCTION RECORD LAW AT MACHINE 17 - the computational half.

`docs/proof-search/anchor-235.md` 9f, and `research/anchor235/chain_depth.py`:

    F_g + 1 = max over phases r in Z_g of the merged gap,

where the merged gaps at phase `r` are the gaps of the LOWER machine's opening
sequence after deleting the openings whose class mod `g` lies in the two-class
set `{r, r + d}`, `d = 2 * u_g`.  The point of the law is that the layer needs
the LOWER opening residues only: at `g = 17` the input is the 1485 openings of
`{5, 7, 11, 13}` on the period 5005, not the 85085 slots of `{5..17}`.

CONVENTIONS, and they differ by one between the two halves of the project.
`chain_depth.py` reports `F_17 = 17` in the BLOCKED-COUNT convention (the
number of blocked slots strictly between the two surviving endpoints); the Lean
corpus reports `Machine17.gap_le : b - a <= 18` in the MAX-GAP convention.  THIS
FILE FORMALISES THE MAX-GAP CONVENTION: `mg r` is the merged GAP (endpoint minus
endpoint), and the law reads

    max over r < 17 of mg r  =  18  =  F(17) + 1 in the blocked-count convention.

(The round-29 brief named 25 / 24 for this machine; those are the layer-19 line
of `chain_depth.py` - `{5..19}` on the 22275 openings of `{5..17}` - not the
layer-17 line.  Gated against `research/anchor235/r29_record17_gate.py`, which
recomputes `mg` in exactly this encoding and asserts it against an independent
full-period scan of `{5,7,11,13,17}`: both 18.)

THE ENCODING is `chain_depth.py`'s, slot for slot.  The lower openings are
walked over `[0, 5005 + 64)` - one lower period plus a look-ahead well past any
gap - and the residue mod 17 is the ABSOLUTE one, so a run that crosses the
period boundary sees the shifted phase of the next copy exactly as the real
machine does.  Only gaps whose LEFT endpoint lies in `[0, 5005)` are counted,
which is `chain_depth.py`'s `starts < len(X)`.
-/

import Mathlib.Tactic.IntervalCases

namespace AnchorRecord17

/-- The lower machine `{5, 7, 11, 13}`, teeth `u_g = 6⁻¹ mod g`:
`(1,4)`, `(6,1)`, `(2,9)`, `(11,2)`.  Period 5005, 1485 openings. -/
def lowOpen (k : Nat) : Bool :=
  k % 5 != 1 && k % 5 != 4 &&
  k % 7 != 6 && k % 7 != 1 &&
  k % 11 != 2 && k % 11 != 9 &&
  k % 13 != 11 && k % 13 != 2

/-- A lower opening SURVIVES the layer at deletion phase `r` iff its class mod
17 avoids the two-class set `{r, r + 6}` (`d = 2 * u_17 = 6`). -/
def surv (r y : Nat) : Bool :=
  lowOpen (y % 5005) && y % 17 != r && y % 17 != (r + 6) % 17

/-- The merged-gap walk: `fuel`, current slot `y`, last survivor plus one
(`0` = none yet), running maximum. -/
def walk (r : Nat) : Nat → Nat → Nat → Nat → Nat
  | 0, _, _, best => best
  | fuel + 1, y, last, best =>
      bif surv r y then
        (bif last == 0 then walk r fuel (y + 1) (y + 1) best
         else bif Nat.blt (last - 1) 5005 && Nat.blt best (y - (last - 1)) then
                walk r fuel (y + 1) (y + 1) (y - (last - 1))
              else walk r fuel (y + 1) (y + 1) best)
      else walk r fuel (y + 1) last best

/-- The largest merged gap at phase `r`. -/
def mg (r : Nat) : Nat := walk r 5069 0 0 0

/-! ## The seventeen phases, one kernel computation each -/

set_option maxRecDepth 20000

theorem mg0 : mg 0 = 16 := by decide +kernel
theorem mg1 : mg 1 = 16 := by decide +kernel
theorem mg2 : mg 2 = 18 := by decide +kernel
theorem mg3 : mg 3 = 18 := by decide +kernel
theorem mg4 : mg 4 = 18 := by decide +kernel
theorem mg5 : mg 5 = 16 := by decide +kernel
theorem mg6 : mg 6 = 18 := by decide +kernel
theorem mg7 : mg 7 = 18 := by decide +kernel
theorem mg8 : mg 8 = 16 := by decide +kernel
theorem mg9 : mg 9 = 15 := by decide +kernel
theorem mg10 : mg 10 = 16 := by decide +kernel
theorem mg11 : mg 11 = 18 := by decide +kernel
theorem mg12 : mg 12 = 18 := by decide +kernel
theorem mg13 : mg 13 = 16 := by decide +kernel
theorem mg14 : mg 14 = 18 := by decide +kernel
theorem mg15 : mg 15 = 18 := by decide +kernel
theorem mg16 : mg 16 = 18 := by decide +kernel

/-- **THE RECORD LAW AT MACHINE 17, computational half.**  No phase's merged
gap exceeds 18, and phase 2 attains it. -/
theorem record_max : (∀ r, r < 17 → mg r ≤ 18) ∧ mg 2 = 18 := by
  refine ⟨fun r hr => ?_, mg2⟩
  interval_cases r <;>
    simp only [mg0, mg1, mg2, mg3, mg4, mg5, mg6, mg7, mg8, mg9, mg10, mg11,
      mg12, mg13, mg14, mg15, mg16] <;>
    omega

end AnchorRecord17
