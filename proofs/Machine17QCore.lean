/-
Machine 17 qualifying-spectrum scan - shared definitions (round 22).

The rung `17 -> 19` of the (D) ladder. Gear 19 has teeth at slot residues
`u' = 3` and `19 - u' = 16`, so the qualifying floor is `2u' = 6` and the
tolerance budget is `F(17) + q' = 18 + 19 = 37`. `MergeLaw.newgap_le` needs,
of machine 17 alone,

    F_2 <= 37   and   Q_j(17; 6) <= 37 for every depth j >= 3.

Full-period Python first (scratchpad ladder_verify.py, all 85085 residues):
F_1..F_8(17) = 18, 25, 28, 33, 35, 40, 43, 48 and
Q_j(17; 6) = 28, 31, 32, 34, 0, 0, ... - so depths 3, 4, 5 are covered by the
UNCONDITIONAL window bounds `F_3 <= 28`, `F_4 <= 33`, `F_5 <= 35` (all inside
the budget); depth 6 is the first that needs the qualifying restriction
(`F_6 = 40 > 37` but `Q_6 = 34`); and every depth `j >= 7` is discharged by
one refutation: no five consecutive gaps are all `>= 6` (the longest run is
4).

All five facts are read off ONE six-step `seekT` walk per opening (round
21's encoding). The first check `o1 <= 18` re-derives `F_1(17) <= 18` from
the same walk, which is what makes fuel 18 provably sufficient. The scan is
chunked by `e = k % 17` into 17 slices of 5005 tuples each - exactly machine
13's size - living in `Machine17QS0/1/2.lean`.
-/

import Machine17

namespace Machine17

/-- First offset `t > s` (walking the CRT tuple) with `atT ... t = true`,
searched with `fu` slots of fuel; `s + 999` if the fuel runs out. At an
opening the sentinel is unreachable: machine-17 gaps cap at 18. -/
def seekT (a b c d e : Nat) : Nat → Nat → Nat
  | 0, s => s + 999
  | fu + 1, s =>
      if atT a b c d e (s + 1) then s + 1 else seekT a b c d e fu (s + 1)

/-- The six-opening chain check from an opening: `F_1 <= 18`, `F_3 <= 28`,
`F_4 <= 33`, `F_5 <= 35`, the qualifying depth-6 bound `Q_6 <= 34` (only
when the four interior gaps all meet the floor 6), and the refutation of any
five-in-a-row run of gaps `>= 6` (`Q_j(17; 6) = 0` for every `j >= 7`). -/
def chainT (a b c d e : Nat) : Bool :=
  let o1 := seekT a b c d e 18 0
  let o2 := seekT a b c d e 18 o1
  let o3 := seekT a b c d e 18 o2
  let o4 := seekT a b c d e 18 o3
  let o5 := seekT a b c d e 18 o4
  let o6 := seekT a b c d e 18 o5
  Nat.ble o1 18 &&
    (Nat.ble o3 28 &&
      (Nat.ble o4 33 &&
        (Nat.ble o5 35 &&
          ((!(Nat.ble 6 (o2 - o1) && Nat.ble 6 (o3 - o2) &&
                Nat.ble 6 (o4 - o3) && Nat.ble 6 (o5 - o4))
              || Nat.ble o6 34) &&
            !(Nat.ble 6 o1 && Nat.ble 6 (o2 - o1) && Nat.ble 6 (o3 - o2) &&
              Nat.ble 6 (o4 - o3) && Nat.ble 6 (o5 - o4))))))

/-- From an opening, the chain facts hold; non-openings are skipped. -/
def qokT (a b c d e : Nat) : Bool := !(atT a b c d e 0) || chainT a b c d e

/-- One slice: all 5005 tuples sharing a fixed `k % 17`. -/
def qslice (e : Nat) : Bool :=
  (List.range 5).all fun a => (List.range 7).all fun b =>
    (List.range 11).all fun c => (List.range 13).all fun d => qokT a b c d e

end Machine17
