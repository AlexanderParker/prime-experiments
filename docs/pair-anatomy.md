# Pair anatomy: two gears jointly, starting with 5 x 7

Session vocabulary continues from `docs/umbrellas-and-shields.md`. Slot k = pair `(6k-1, 6k+1)`.
For a pair of gears (q, r) the joint machine has period qr and every slot carries a role-pair.

## The 5 x 7 map (period 35)

`O` = open to both, `5`/`7` = blocked by that gear only, `X` = blocked by both:

    slot:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
           O  X  O  O  5  O  X  O  7  5  O  5  O  7  5  7  5  O

    slot: 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34
           O  5  7  5  7  O  5  O  5  7  O  X  O  5  O  O  X

15 openings = (5-2)(7-2). Gap word (cyclic): 2,1,2,2,3,2,5,1,5,2,3,2,2,1,2 - max gap 5 = F_k(7).

**Mirror:** fold at 17.5 and the map matches itself: openings pair 2<->33, 3<->32, 5<->30, 7<->28,
10<->25, 12<->23, 17<->18; X slots pair 1<->34, 6<->29. The half-period mirror at pair level.

## Role-pair census (exact, product rule by CRT)

    counts per 35        7 kills (2)   7 shields (1)   7 misses (4)
    5 kills (2)               4             2               8
    5 shields (1)             2             1               4
    5 misses (2)              4             2               8

Open-to-both = (1+2)(1+4) = 15. Double-kills = 2x2 = 4 - the "pairwise coincidence is always
exactly 4" law (twin-prime-program.md section 31b).

## Taxonomy of the four double-kills

* slot 1 = (5,7): both teeth on their own shared pair - the tooth-sharing law; twin gears kill the
  pair they are.
* slot 6 = (35,37): both kill the left member 35 = 5*7 - the joint block at the product.
* slot 29 = (173,175): both kill the right member 175 = 5^2*7.
* slot 34 = (203,205): crossed - 7 kills the left (203 = 7*29), 5 kills the right (205 = 5*41);
  the pair annihilated from both sides at once.

The four fall into mirror pairs (1,34) and (6,29). In general the four coincidences of any gear
pair are: shared/own structure at the CRT lift of (+u_q, +u_r), the product-type slots at mixed
signs, and one crossed kill - the four sign choices (+-u_q, +-u_r), which is the `2^n` law of the
threat lattice (section 32b) at n = 2.

## 5 x 11 (period 55) and 7 x 11 (period 77)

5 x 11, teeth {1,4} and {2,9}: 27 openings = 3*9, max gap 4, mirror-symmetric.

    OQROQOQOOXOQORQOQOOQRQOOXOQOOQOXOOQRQOOQOQROQOXOOQOQORQ

X slots, mirror-paired (9,46), (24,31):
* slot 9 = (53,55): 55 = 5*11 - product block, both kill right
* slot 46 = (275,277): 275 = 5^2*11 - product block, both kill left
* slot 24 = (143,145): 11 left (11*13), 5 right (5*29) - crossed
* slot 31 = (185,187): 5 left (5*37), 11 right (11*17) - crossed

7 x 11, teeth {1,6} and {2,9}: 45 openings = 5*9, max gap 4, mirror-symmetric.

    OQROOOQOQROOOXOQOOOOQOQOROOQOQOROOQRQOOOOQRQOOROQOQOOROQOXOOOOQOXOOORQOQOOORQ

X slots, mirror-paired (13,64), (20,57):
* slot 13 = (77,79): 77 = 7*11 - product block, both left
* slot 64 = (383,385): 385 = 5*7*11 - product block, both right
* slot 20 = (119,121): 7 left (7*17), 11 right (11^2) - crossed
* slot 57 = (341,343): 11 left (11*31), 7 right (7^3) - crossed

## 11 x 13 (period 143): the twin-gear collapse confirmed

Teeth {2,9} and {2,11} - shared tooth residue 2, the collapse itself. 99 openings = 9*11, max gap 3,
mirror-symmetric. X slots (2,141), (24,119):

* **slot 2 = (11,13): the own-pair share** - twin gears killing themselves at birth
* slot 24 = (143,145): 143 = 11*13 - product block, both left
* slot 119 = (713,715): 715 = 5*11*13 - product block, both right
* slot 141 = (845,847): 13 left (5*13^2), 11 right (7*11^2) - crossed

Taxonomy law (corrected below in "The pair table"): every pair has exactly 1 both-left + 1
both-right + 2 crossed coincidences, and one *crossed* slot lands on the pair (q,r) itself precisely
when q and r are twin gears (at their shared tooth residue, each gear killing its own member).

Max gap falls as the pair grows: 5 (5x7) -> 4 (5x11, 7x11) -> 3 (11x13). A lone pair of bigger
gears covers less; gaps grow only through accumulation of gears, never through the size of one pair.

## What varies across pairs and what does not

Invariant for every pair: openings (q-2)(r-2); exactly 4 double-kills = the CRT lifts of the sign
choices (+-u_q, +-u_r); exact mirror symmetry. Taxonomy of the four: same-sign lifts are product
blocks (both gears hit the same member), mixed-sign lifts are crossed kills - and for twin gears the
(+,+) lift collapses onto their own shared pair (the 5x7 slot-1 case). What the slip controls is the
interleaving only: 5x7 has max gap 5 while 5x11 and 7x11 both max at 4 - same tooth inventory,
different bunching. Counts are pair-independent theorems; placement is slip arithmetic - the
pair-level preview of the programme's single open question.

## The pair table: single-gear laws lifted to composites

| pair  | qr  | family | both-left | both-right | low tooth | low pair holds | crossed  | shield | maxgap |
|-------|-----|--------|-----------|------------|-----------|----------------|----------|--------|--------|
| 5x7   | 35  | 5      | *6*       | 29         | 6         | 35             | 1!, 34   | 0      | 5 |
| 5x11  | 55  | 1      | 46        | *9*        | 9         | 55             | 24, 31   | 0      | 4 |
| 5x13  | 65  | 5      | *11*      | 54         | 11        | 65             | 24, 41   | 0      | 4 |
| 7x11  | 77  | 5      | *13*      | 64         | 13        | 77             | 20, 57   | 0      | 4 |
| 7x13  | 91  | 1      | 76        | *15*       | 15        | 91             | 41, 50   | 0      | 4 |
| 11x13 | 143 | 5      | *24*      | 119        | 24        | 143            | 2!, 141  | 0      | 3 |

(italic = low tooth; ! = crossed slot on the gears' own pair, twin gears only.)

The pair's same-side kills ARE the composite gear qr, so every single-gear law lifts verbatim:
both-left tooth = 6^{-1} mod qr and both-right = its mirror, summing to qr; the family law extends
by sign multiplication (qr mod 6 = product of the factors' +-1; family 5 -> both-left low, family
1 -> both-right low); self-blocking extends (low tooth = round(qr/6) = the slot whose pair contains
qr); joint shield at 0. The genuinely new object is the crossed pair - one member killed by each
gear, mirror-summing to qr, positions governed by the slip inverse (X_crossed = 1 + q((-2q^{-1})
mod r), twin-prime-program.md section 28b) - and for twin gears one crossed slot is their own pair.
Max gap falls as the pair grows: 5, 4, 4, 4, 4, 3.

## Triples: the 8-lift prediction confirmed

All four triples from {5,7,11,13} have exactly 2^3 = 8 coincidence slots: 2 all-same-side teeth
(the composite gear qrs - all-left at 6^{-1} mod P, all-right at its mirror, low tooth =
round(P/6) whose pair contains the product, family by sign multiplication) plus 6 mixed lifts
(three minority-gear flavours x two orientations), mirror-paired with complementary L/R signatures:

    (5,7,11)  P=385  fam 1: all-R low at 64 = (383,385);  mixed 134/251, 141/244, 174/211
    (5,7,13)  P=455  fam 5: all-L low at 76 = (455,457);  mixed 41/414, 106/349, 141/314
    (5,11,13) P=715  fam 1: all-R low at 119 = (713,715); mixed 24/691, 141/574, 284/431
    (7,11,13) P=1001 fam 5: all-L low at 167 = (1001,1003); mixed 141/860, 288/713, 405/596

Deep slots recur across machines: slot 141 = (845,847) = (5*13^2, 7*11^2) has teeth of 5, 7, 11
and 13 all present, so every triple drawn from them coincides there; likewise slots 24 and 596.
Many-factor slots are the coincidence hubs.

Max gap resumes growing under accumulation: pairs 5,4,4,4,4,3 but triples 7,7,6,6 - adding gears
grows gaps even as each gear individually weakens.

**The accumulation model, verified to n = 3:** a machine of n gears = one composite gear (2 teeth,
all classical laws lifted verbatim) + a crossed cloud of 2^n - 2 mixed lifts, mirror-paired with
sign complement. The cloud doubles per gear; its placement is the programme's single open question.
(The square-roots-of-unity lattice of twin-prime-program.md section 28c, read member-by-member.)

## Certified twins from two gears

The {5,7} window is slots <= 8 (6k+1 <= 49). Openings there: 2, 3, 5, 7 ->
(11,13), (17,19), (29,31), (41,43) - every twin in the window, generated by two gears' umbrellas.
