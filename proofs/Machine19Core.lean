/-
Machine 19 (gears {5,7,11,13,17,19}, period 1,616,615) - shared definitions.

The certificate is `F_k(19) = 25`, `F2_k(19) = 31`, and - new this round -
the depth-4 spectrum value `F4_k(19) = 38`, which feeds the
suppression-corrected flatness instance `F_4 <= F + q'` (38 <= 48) that
`Spectrum.merged_le_of_shallow` consumes. All three verified over the full
period numerically before formalising (F_j ladder 25, 31, 35, 38, 47).

The period is 323 times machine 13's, so the scan is chunked by the pair
`(k%17, k%19)` into 323 slices of 5005 tuples each - exactly machine 13's
size, the shape known to evaluate quickly. The slices live in
`Machine19S0.lean` .. `Machine19S16.lean` (one file per `k%17` residue, so
lake checks them in parallel processes); `Machine19.lean` assembles them.

Two encoding improvements over `Machine17` make this affordable (round 18's
re-attack on the round-15 wall):

* only OPENINGS are scanned - a gap runs between openings, so a tuple that
  is not itself an opening starts no gap. Opening density here is
  `prod (1 - 2/q) = 0.234`, a 4.3x cut;
* all three facts are checked by counting openings along ONE window walk
  (`countP`, allocation-light) rather than separate nested scans.
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

/-- From an opening: the next opening arrives within 25 slots (`F <= 25`),
a second within 31 (`F2 <= 31`), and a fourth within 38 (`F4 <= 38`).
Non-openings are skipped. -/
def okT (a b c d e f : Nat) : Bool :=
  !(atT a b c d e f 0) ||
    (((List.range 25).any fun i => atT a b c d e f (i+1)) &&
      (Nat.ble 2 ((List.range 31).countP fun i => atT a b c d e f (i+1)) &&
        Nat.ble 4 ((List.range 38).countP fun i => atT a b c d e f (i+1))))

/-- One slice: all 5005 tuples sharing a fixed `(k%17, k%19)`. -/
def slice (e f : Nat) : Bool :=
  (List.range 5).all fun a => (List.range 7).all fun b =>
    (List.range 11).all fun c => (List.range 13).all fun d => okT a b c d e f

end Machine19
