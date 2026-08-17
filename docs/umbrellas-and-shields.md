# Umbrellas and shields: the per-gear anatomy of slot space

Session vocabulary, 2026-08-17. Slot k is the candidate pair `(6k-1, 6k+1)`. Per gear q >= 5, every
residue of k mod q has exactly one role:

* **teeth** (2 residues) - the kills. The *left-kill tooth* at `k = u` (where `u = 6^{-1} mod q`)
  kills the left member: `q | 6k-1`. The *right-kill tooth* at `k = q-u` kills the right member:
  `q | 6k+1`.
* **shield** (1 residue) - `k = 0 mod q`: the gear divides the midpoint `6k`, so it provably cannot
  divide either member. Structural safety for that pair, from this gear, forever.
* **umbrellas** (q-3 residues in two runs) - the open runs between the teeth. The *short umbrella*
  (length `2u'-1`, with `u' = min(u, q-u)`) has the shield at its exact centre; the *long umbrella*
  (length `q-2u'-1`) is pure misses. Lengths tend to q/3 and 2q/3, since the tooth separation is
  `3^{-1} mod q`.

## The table

| gear | left-kill tooth (k=u) | right-kill tooth (k=q-u) | shield | short umbrella | long umbrella | short/long |
|------|------|------|---|------------------|---------|------|
| 5    | 1    | 4    | 0 | {0}              | {2,3}   | 1/2  |
| 7    | 6    | 1    | 0 | {0}              | {2..5}  | 1/4  |
| 11   | 2    | 9    | 0 | {10,0,1}         | {3..8}  | 3/6  |
| 13   | 11   | 2    | 0 | {12,0,1}         | {3..10} | 3/8  |
| 17   | 3    | 14   | 0 | {15,16,0,1,2}    | {4..13} | 5/10 |
| 19   | 16   | 3    | 0 | {17,18,0,1,2}    | {4..15} | 5/12 |

Checks: gear 11 tooth 2 -> slot 2 = (11,13), 11 | 11 (left); tooth 9 -> slot 9 = (53,55), 11 | 55
(right). Gear 5 shield residue 0 -> slot 5 = (29,31), midpoint 30.

## Why the teeth sum to q

The teeth are `+-u`: the left tooth solves `6k = +1 mod q`, the right solves `6k = -1`, and negating
one solution gives the other. So right tooth `= q - u` and the sum is q (gear 19: 16 + 3 = 19). The
mirror is the pair structure itself: `6(-k) + 1 = -(6k - 1)` - killing lefts and killing rights are
one event seen in a mirror held at the shield. The same single symmetry yields: teeth summing to q,
the shield centred in the short umbrella, the whole pattern symmetric about slot 0, and the mirror
axis at half of every machine's period.

## The rotation-level view (one lap of a gear)

In n-space a gear blocks once per rotation; of every six rotations exactly two land on candidate
members (one left, one right), one lands on a midpoint (shield), three land on ground gears 2 and 3
own (misses). Gear 5's laps: 5 kills left of (5,7); 10, 15, 20 miss; 25 kills right of (23,25); 30
shields (29,31) - then the same score forever, advanced by q slots per lap. Gears = 1 mod 6 (7, 13,
19) walk the six slots in the opposite direction, so they kill right-then-left; gears = 5 mod 6 kill
left-then-right. The kills are spaced 4-then-2 rotations (or 2-then-4), producing the long and short
umbrellas.

## The machine statement in this vocabulary

A twin slot is a slot standing under every relevant gear's umbrella at once (shield-centre or plain
miss). Example, slot 12 = (71,73): mod 5 -> 2 (long umbrella), mod 7 -> 5 (long), mod 11 -> 1
(short), mod 13 -> 12 (short) - covered everywhere. The umbrellas provably overlap somewhere in every
period (the alignment law: the smallest gear's long umbrella survives intact at some phase, whatever
the other gears do). The single open question of the programme is *where*: whether an all-umbrella
slot always occurs inside the certification window `6k+1 <= y^2` - Reduction A in umbrella language.
