/-
Machine 19 qualifying-spectrum scan - shared definitions (round 21).

Round 20 kernel-checked `F_1 = 25`, `F_2 = 31` and `F_4 = 38` and wired the
first end-to-end (D) instance `D_of_shallow_word`, whose ONLY remaining
hypothesis was the word's shallowness (`l + 2 <= 4`). This scan removes that
hypothesis. Three new facts over the same 323-slice CRT chunking (all
verified over the full period numerically first - see formalist.md round 21):

* `F_3(19) <= 35` - the third opening after an opening arrives within 35;
* `F_5(19) <= 47` - the fifth arrives within 47 (note `47 <= 48 = F + q'`:
  depth 5 is flat with no qualifying constraint at all);
* NO opening starts four consecutive gaps all `>= 8` (the `Q_6(19) = 0`
  carrier). Every deeper qualifying window contains such a run, so this one
  refutation discharges EVERY depth `j >= 6`.

All three are read off ONE walk: `seekT` locates the next opening exactly
(machine gaps cap at 25 - kernel-checked in round 20 - so fuel 25 always
suffices), iterated five times. The walk visits each slot at most once and
stops at `o5 <= 47`, so the per-opening cost is BELOW the round-20 `okT`
(which walked 25 + 31 + 38 slots in three passes); and the extraction needs
no witness pigeonhole - the chain IS the consecutive openings.

Together with round 20's facts the whole depth ladder `j = 2..5` is under
`F + q' = 48` and `j >= 6` is empty - (D) at `alpha = 3` at machine 19 with
no shallowness hypothesis (`Machine19Q.lean`), and, through the merge law,
the full two-machine instance at the 19->23 step (`Machine23.lean`).
-/

import Machine19Core

namespace Machine19

/-- First offset `t > s` (walking the CRT tuple) with `atT ... t = true`,
searched with `fu` slots of fuel; `s + 999` if the fuel runs out. At an
opening the sentinel is unreachable: machine-19 gaps cap at 25
(`Machine19.gap_le`), so fuel 25 always finds the next opening. -/
def seekT (a b c d e f : Nat) : Nat → Nat → Nat
  | 0, s => s + 999
  | fu + 1, s => if atT a b c d e f (s + 1) then s + 1 else seekT a b c d e f fu (s + 1)

/-- The five-opening chain check from an opening: the third next opening is
within 35 (`F3 <= 35`), the fifth within 47 (`F5 <= 47`), and the four
gaps of the chain are never all at or above the qualifying floor `2u' = 8`
(`Q_6(19) = 0`). -/
def chainT (a b c d e f : Nat) : Bool :=
  let o1 := seekT a b c d e f 25 0
  let o2 := seekT a b c d e f 25 o1
  let o3 := seekT a b c d e f 25 o2
  let o4 := seekT a b c d e f 25 o3
  let o5 := seekT a b c d e f 25 o4
  Nat.ble o3 35 &&
    (Nat.ble o5 47 &&
      !(Nat.ble 8 o1 && Nat.ble 8 (o2 - o1) && Nat.ble 8 (o3 - o2) &&
        Nat.ble 8 (o4 - o3)))

/-- From an opening, the chain facts hold; non-openings are skipped. -/
def qokT (a b c d e f : Nat) : Bool :=
  !(atT a b c d e f 0) || chainT a b c d e f

/-- One slice: all 5005 tuples sharing a fixed `(k%17, k%19)`. -/
def qslice (e f : Nat) : Bool :=
  (List.range 5).all fun a => (List.range 7).all fun b =>
    (List.range 11).all fun c => (List.range 13).all fun d => qokT a b c d e f

end Machine19
