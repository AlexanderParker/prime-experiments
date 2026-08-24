/-
Machine 13 qualifying-spectrum scan - shared definitions (round 22).

The rung `13 -> 17` of the (D) ladder. Gear 17 has teeth at slot residues
`u' = 3` and `17 - u' = 14`, so the qualifying floor is `2u' = 6` and the
tolerance budget is `F(13) + q' = 11 + 17 = 28`. `MergeLaw.newgap_le` needs,
of machine 13 alone,

    F_2 <= 28   and   Q_j(13; 6) <= 28 for every depth j >= 3.

Full-period Python first (scratchpad ladder_verify.py, all 5005 residues):
F_1..F_8(13) = 11, 16, 23, 26, 28, 31, 34, 38 and Q_j(13; 6) = 18, 23, 0, 0,
... - the qualifying spectrum is EMPTY from depth 5 on, because the longest
run of gaps `>= 6` is 2. So depths 3 and 4 are covered by the UNCONDITIONAL
window bounds `F_3 <= 23` and `F_4 <= 26` (both already inside the budget),
and every depth `j >= 5` is discharged by one refutation: no three
consecutive gaps are all `>= 6`.

All three facts are read off ONE four-step `seekT` walk per opening, the
round-21 encoding: the walk visits each slot once, and `seek_next` turns
extraction into equations rather than a witness pigeonhole. The first check
`o1 <= 11` re-derives `F_1(13) <= 11` from the same walk, which is what
makes fuel 11 provably sufficient.
-/

import Machine13

namespace Machine13

/-- First offset `t > s` (walking the CRT tuple) with `atT ... t = true`,
searched with `fu` slots of fuel; `s + 999` if the fuel runs out. At an
opening the sentinel is unreachable: machine-13 gaps cap at 11. -/
def seekT (a b c d : Nat) : Nat → Nat → Nat
  | 0, s => s + 999
  | fu + 1, s => if atT a b c d (s + 1) then s + 1 else seekT a b c d fu (s + 1)

/-- The four-opening chain check from an opening: the next opening is within
11 (`F_1 <= 11`), the third within 23 (`F_3 <= 23`), the fourth within 26
(`F_4 <= 26`), and the first three gaps are never all at or above the
qualifying floor `2u' = 6` (`Q_j(13; 6) = 0` for every `j >= 5`). -/
def chainT (a b c d : Nat) : Bool :=
  let o1 := seekT a b c d 11 0
  let o2 := seekT a b c d 11 o1
  let o3 := seekT a b c d 11 o2
  let o4 := seekT a b c d 11 o3
  Nat.ble o1 11 &&
    (Nat.ble o3 23 &&
      (Nat.ble o4 26 &&
        !(Nat.ble 6 o1 && Nat.ble 6 (o2 - o1) && Nat.ble 6 (o3 - o2))))

/-- From an opening, the chain facts hold; non-openings are skipped. -/
def qokT (a b c d : Nat) : Bool := !(atT a b c d 0) || chainT a b c d

/-- The whole period: all 5005 CRT tuples. -/
def qslice : Bool :=
  (List.range 5).all fun a => (List.range 7).all fun b =>
    (List.range 11).all fun c => (List.range 13).all fun d => qokT a b c d

end Machine13
