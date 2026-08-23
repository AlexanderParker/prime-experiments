/-
Machine 19 (gears {5,7,11,13,17,19}, period 1,616,615) - shared definitions.

The certificate is `F_k(19) = 25`, `F2_k(19) = 31`. The period is 323 times
machine 13's, so the scan is chunked by the pair `(k%17, k%19)` into 323
slices of 5005 tuples each - exactly machine 13's size, the shape known to
evaluate quickly.

Two encoding improvements over `Machine17` make this affordable (round 18's
re-attack on the round-15 wall):

* only OPENINGS are scanned - a gap runs between openings, so a tuple that
  is not itself an opening starts no gap. Opening density here is
  `prod (1 - 2/q) = 0.234`, a 4.3x cut;
* both facts are checked in ONE walk of 31 steps rather than two separate
  walks of 25 and 31.
-/

import Corridor

namespace Machine19

/-- Opening test on the CRT tuple `(k%5, k%7, k%11, k%13, k%17, k%19)`. -/
def expT (a b c d e f : Nat) : Bool :=
  a != 1 && a != 4 && b != 6 && b != 1 && c != 2 && c != 9 &&
    d != 11 && d != 2 && e != 3 && e != 14 && f != 16 && f != 3

/-- The test `n` slots further on. -/
def atT (a b c d e f n : Nat) : Bool :=
  expT ((a+n)%5) ((b+n)%7) ((c+n)%11) ((d+n)%13) ((e+n)%17) ((f+n)%19)

/-- From an opening: the next opening arrives within 25 slots (`F <= 25`) and
a second one within 31 (`F2 <= 31`). Non-openings are skipped. -/
def okT (a b c d e f : Nat) : Bool :=
  !(atT a b c d e f 0) ||
    (((List.range 25).any fun i => atT a b c d e f (i+1)) &&
      Nat.ble 2 ((List.range 31).countP fun i => atT a b c d e f (i+1)))

/-- One slice: all 5005 tuples sharing a fixed `(k%17, k%19)`. -/
def slice (e f : Nat) : Bool :=
  (List.range 5).all fun a => (List.range 7).all fun b =>
    (List.range 11).all fun c => (List.range 13).all fun d => okT a b c d e f

end Machine19
