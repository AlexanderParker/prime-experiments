/-
Machine 23's qualifying-spectrum scan, POSITION-INDEXED (round 24).

Round 23 factored the machine-23 period `37,182,145 = 1,616,615 * 23` as

    (machine-19 CRT tuple)  x  (gear-23 phase  g = k % 23)

and walked the SLOTS once per phase (`Machine23QCore.lean`).  That paid a
measured 21x: gear 23 kills only 2 of 23 phases at any one opening, so the
machine-19 walking is identical for 21 of the 23 phases and the kernel could
not share it - the walks are indexed by `g` even where they compute the same
number.  A control settled the mechanism: making the loop body `g`-free
collapsed 12 s to 1.35 s, so the kernel DOES share structurally identical
subterms; what it cannot do is "evaluate once, then branch".

THIS FILE IS THE NAMED FIX: INDEX THE MACHINE-19 CHAIN BY POSITION, NOT BY
OFFSET.  `w19 a b c d e f k` is the offset of the `k`-th machine-19 opening
after the base.  For a LITERAL `k` this term does not mention `g` at all, and
after one unfolding step at a literal index its body is the canonical term
`Machine19.seekT a b c d e f 25 (w19 a b c d e f k)` whatever route reached
it - so the whole machine-19 walk is evaluated ONCE per CRT tuple and every
phase reads cached values.  The phase loop only SELECTS positions: `nsurv`
steps forward over positions until it finds one whose opening survives gear
23, and gear-23 survival at offset `t` is the two-comparison test
`(g + t) % 23 not in {4, 19}`.

SIZED OVER THE FULL PERIOD FIRST (scratchpad idx23.py / idx23b.py / idx23c.py,
numpy over machine 19's 1,616,615 slots x 23 phases = all 7,952,175 machine-23
openings):

* at most **4** machine-19 positions separate consecutive machine-23 openings,
  so `nsurv` fuel 5 never reaches its sentinel;
* at most **11** machine-19 positions are spanned by 6 machine-23 openings;
* the chain Bool below is TRUE at every one of the 7,952,175 openings, and the
  values it reads off are exactly `F_1..F_6(23) = 34, 39, 50, 58, 65, 77` -
  the round-23 numbers, re-derived from the position-indexed form.

WHAT ONE SLICE READS OFF (identical clauses to `Machine23QCore.chain23`):

* `o1 <= 34`  - `F_1(23) = 34`, the true `F(23)` the 23->29 budget uses;
* `o2 <= 39`  - `F_2(23) = 39`, the `SpectrumBound g23 2 39` the rung consumes;
* four guarded rungs `o3, o4, o5, o6 <= 60` under the qualifying floor
  `2u'' = 10` (`u'' = 5`, gear 29), i.e. `Q_3..Q_6(23; 10) <= 60`;
* NO five consecutive gaps all `>= 10`, which is `Q_j(23; 10) = 0` for every
  `j >= 7` and so discharges all remaining depths.

Together: `max (F_2, max_j Q_j) <= 60 <= 63 = F(23) + 29` - the 23->29 rung.
-/

import Machine19QCore

namespace Machine23

/-- **The machine-19 opening chain, indexed by POSITION.**
`w19 a b c d e f k` is the offset of the `k`-th machine-19 opening at or
after the base (`k = 0` is the base itself).  The term mentions no gear-23
phase, so all 23 phases share its reduction. -/
def w19 (a b c d e f : Nat) : Nat → Nat
  | 0 => 0
  | k + 1 => Machine19.seekT a b c d e f 25 (w19 a b c d e f k)

/-- Gear 23's kill test at phase `g` and offset `t`: the two teeth sit at
slot residues 4 and 19. -/
def kil23 (g t : Nat) : Bool := (g + t) % 23 == 4 || (g + t) % 23 == 19

/-- The next POSITION after `k` whose machine-19 opening survives gear 23 at
phase `g`, searched with `fu` positions of fuel; `k + 99` if the fuel runs
out.  At most 4 consecutive machine-19 openings are killed (full-period
census), so fuel 5 never reaches the sentinel. -/
def nsurv (a b c d e f g : Nat) : Nat → Nat → Nat
  | 0, k => k + 99
  | fu + 1, k =>
      if kil23 g (w19 a b c d e f (k + 1)) then nsurv a b c d e f g fu (k + 1)
      else k + 1

/-- The six-opening machine-23 chain check from a machine-23 opening, read
off the shared machine-19 position chain.  Clauses, in order: `F_1 <= 34`,
`F_2 <= 39`, the four qualifying rungs `Q_3..Q_6 <= 60` (each guarded by its
floor condition, so the walk stops early), and the five-run refutation that
empties every depth `j >= 7`.

Each opening's clause is preceded by its POSITION CHECK `p_i <= p_{i-1} + 5`,
which is exactly "`nsurv` did not reach its sentinel".  So the scan certifies
its own fuel and imports no bound from anywhere: it is the position-indexed
counterpart of the round-23 encoding's `o1 <= 34` scan-first-check. -/
def chainIdx (a b c d e f g : Nat) : Bool :=
  let p1 := nsurv a b c d e f g 5 0
  let p2 := nsurv a b c d e f g 5 p1
  let p3 := nsurv a b c d e f g 5 p2
  let p4 := nsurv a b c d e f g 5 p3
  let p5 := nsurv a b c d e f g 5 p4
  let p6 := nsurv a b c d e f g 5 p5
  let o1 := w19 a b c d e f p1
  let o2 := w19 a b c d e f p2
  let o3 := w19 a b c d e f p3
  let o4 := w19 a b c d e f p4
  let o5 := w19 a b c d e f p5
  let o6 := w19 a b c d e f p6
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

/-- From a machine-23 opening the chain facts hold; a base killed by gear 23
is skipped (the machine-19 opening test is hoisted out of the phase loop). -/
def qokIdx (a b c d e f g : Nat) : Bool :=
  kil23 g 0 || chainIdx a b c d e f g

/-- One slice: all 5005 machine-19 tuples sharing a fixed `(k%17, k%19)`,
each with all 23 gear-23 phases. -/
def qsliceIdx (e f : Nat) : Bool :=
  (List.range 5).all fun a => (List.range 7).all fun b =>
    (List.range 11).all fun c => (List.range 13).all fun d =>
      !(Machine19.atT a b c d e f 0) ||
        ((List.range 23).all fun g => qokIdx a b c d e f g)

end Machine23
