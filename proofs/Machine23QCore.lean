/-
Machine 23's qualifying-spectrum scan - shared definitions (round 23).

THE FACTORISED SCAN.  Machine 23's period is 37,182,145 = 1,616,615 * 23, so
a direct slice family would need 7,434 slices of 5,005 CRT tuples.  Instead
the scan is factored as

    (machine-19 residue tuple)  x  (PHASE  g = k % 23),

i.e. the 323 slices of machine 19 that `Machine19QS0..16` already use, each
carrying an inner 23-fold loop over the gear-23 phase.  Every machine-23
residue occurs exactly once, so this is the full period; but the outer skip
`!(Machine19.atT a b c d e f 0)` is tested ONCE per machine-19 tuple instead
of 23 times, and the walk is the ordinary opening walk with one extra
residue counter.

WHAT ONE WALK READS OFF (all verified over the full 37,182,145-slot period
first - scratchpad m23_qspec.py: F_1..F_8 = 34, 39, 50, 58, 65, 77, 83, 88;
Q_j(23; 10) = 39, 43, 50, 55, 60, 0 for j = 2..7; longest run of gaps >= 10
is 4):

* `o1 <= 34`  - `F_1(23) = 34`.  This is the SCAN-FIRST-CHECK: it re-derives
  the fuel-sufficiency fact that `seek23_next` needs, so the scan imports no
  bound from anywhere;
* `o2 <= 39`  - `F_2(23) = 39`, the `SpectrumBound g23 2 39` the 23->29 rung
  consumes;
* four guarded rungs `o3, o4, o5, o6 <= 60` under the qualifying floor
  `2u'' = 10` (`u'' = 5`, gear 29), i.e. `Q_3..Q_6(23; 10) <= 60`;
* `NO five consecutive gaps all >= 10`, which is `Q_j(23; 10) = 0` for every
  `j >= 7` and so discharges all remaining depths.

Together: `max (F_2, max_j Q_j) <= 60 <= 63 = F(23) + 29` - the 23->29 rung.

COST.  The guards are Bool `&&`/`||`, which the kernel evaluates lazily, so
a tuple whose second gap is below the floor never walks past `o2`: the
average walk is ~10 slots rather than the ~28 a flat six-step chain would
cost, while the extraction stays flat (no `if` to case-split).
-/

import Machine23

namespace Machine23

/-- Opening test on the machine-23 CRT tuple
`(k%5, k%7, k%11, k%13, k%17, k%19, k%23)`: a machine-19 opening off gear
23's two teeth (slot residues 4 and 19). -/
def expT23 (a b c d e f g : Nat) : Bool :=
  Machine19.expT a b c d e f && g != 4 && g != 19

/-- The test `n` slots further on. -/
def atT23 (a b c d e f g n : Nat) : Bool :=
  expT23 ((a + n) % 5) ((b + n) % 7) ((c + n) % 11) ((d + n) % 13)
    ((e + n) % 17) ((f + n) % 19) ((g + n) % 23)

/-- First offset `t > s` (walking the CRT tuple) with `atT23 ... t = true`,
searched with `fu` slots of fuel; `s + 999` if the fuel runs out.  At an
opening the sentinel is unreachable: `F_1(23) = 34`, kernel-checked by the
first clause of `chain23` itself, so fuel 34 always finds the next opening. -/
def seek23 (a b c d e f g : Nat) : Nat → Nat → Nat
  | 0, s => s + 999
  | fu + 1, s =>
      if atT23 a b c d e f g (s + 1) then s + 1 else seek23 a b c d e f g fu (s + 1)

/-- The six-opening chain check from a machine-23 opening.  Clauses, in
order: `F_1 <= 34`, `F_2 <= 39`, the four qualifying rungs `Q_3..Q_6 <= 60`
(each guarded by its floor condition, so the walk stops early), and the
five-run refutation that empties every depth `j >= 7`. -/
def chain23 (a b c d e f g : Nat) : Bool :=
  let o1 := seek23 a b c d e f g 34 0
  let o2 := seek23 a b c d e f g 34 o1
  let o3 := seek23 a b c d e f g 34 o2
  let o4 := seek23 a b c d e f g 34 o3
  let o5 := seek23 a b c d e f g 34 o4
  let o6 := seek23 a b c d e f g 34 o5
  Nat.ble o1 34 &&
  Nat.ble o2 39 &&
  (!(Nat.ble 10 (o2 - o1)) || Nat.ble o3 60) &&
  (!(Nat.ble 10 (o2 - o1) && Nat.ble 10 (o3 - o2)) || Nat.ble o4 60) &&
  (!(Nat.ble 10 (o2 - o1) && Nat.ble 10 (o3 - o2) && Nat.ble 10 (o4 - o3)) ||
      Nat.ble o5 60) &&
  (!(Nat.ble 10 (o2 - o1) && Nat.ble 10 (o3 - o2) && Nat.ble 10 (o4 - o3) &&
      Nat.ble 10 (o5 - o4)) || Nat.ble o6 60) &&
  !(Nat.ble 10 o1 && Nat.ble 10 (o2 - o1) && Nat.ble 10 (o3 - o2) &&
      Nat.ble 10 (o4 - o3) && Nat.ble 10 (o5 - o4))

/-- From a machine-23 opening the chain facts hold; killed or non-opening
phases are skipped. -/
def qok23 (a b c d e f g : Nat) : Bool :=
  !(atT23 a b c d e f g 0) || chain23 a b c d e f g

/-- One slice: all 5005 machine-19 tuples sharing a fixed `(k%17, k%19)`,
each with all 23 gear-23 phases.  The machine-19 opening test is hoisted
out of the phase loop. -/
def qslice23 (e f : Nat) : Bool :=
  (List.range 5).all fun a => (List.range 7).all fun b =>
    (List.range 11).all fun c => (List.range 13).all fun d =>
      !(Machine19.atT a b c d e f 0) ||
        ((List.range 23).all fun g => qok23 a b c d e f g)

end Machine23
