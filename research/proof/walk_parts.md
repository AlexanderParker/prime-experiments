# The walk in parts (owner, 2026-09-13)

Owner's direction: no counting. Stick to the mechanism of the walk: the origin pair, the rules
for taking steps, the actual landing of each step, how each landing relates to its last step,
and the target zone. Each part gets its own proof; the walk is not proved as one object.
Kernel: proofs/MirrorWalk.lean (round 45) and proofs/MirrorWalkParts.lean (round 46), both built,
0 sorries, axioms propext / Classical.choice / Quot.sound only.

## Part 1. The origin pair

- Home (-1, 1): open to every gear. PROVED [MirrorWalk.home_open]
- A gear pair (g, g+2), both prime: open to every prime gear other than its two members.
  PROVED [MirrorWalk.pair_open]
- A twin (t, t+2) in a window: open to every gear up to q (it is a twin because no gear strikes
  it). By definition.

## Part 2. The step rule

- An axis is a multiple k M of the product M of a gear set S containing 2 and 3; the pattern of
  S is symmetric about every such multiple. A step from the column n about the axis a lands on
  flip a n = 2a - n - 2, the left member of the mirror image. PROVED (definition)
  [MirrorWalk.flip]
- Two steps compose to a slide: flip a (flip b n) = n + 2(a - b). PROVED [MirrorWalk.flip_flip]

## Part 3. How a landing relates to its last step, gear by gear

- Carried gears: a gear dividing the axis product 2a finds the landing open iff it found the
  origin open. PROVED [MirrorWalk.flip_carries, openTo_flip_iff]
- The landing law for every gear: h strikes the landing of the step about a from n iff
  2a = n + 2 (mod h) (h then divides the landing's left member) or 2a = n (mod h) (its right
  member). The landing's fate for each gear is one congruence between the axis product and a
  member of the origin. PROVED [MirrorWalk.struck_flip_iff]
- One class per side: for a prime h not dividing 2M, two axes k M, k' M whose landings are both
  struck on the same side by h have k = k' (mod h). PROVED [MirrorWalk.same_class_of_struck]
- One gear never blocks a step: for a prime gear h >= 5 not dividing 2M, among any three
  consecutive multiples k0 M, (k0+1) M, (k0+2) M some axis lands the origin on a column open to
  h. PROVED [MirrorWalk.exists_axis_open]

## Part 4. Steps in sequence

- A walk about a_1, a_2, ... ends at 2A - n - 2 (odd length) or n + 2A (even), A the alternating
  sum; the end is open to exactly the gears dividing 2A that the origin was open to. PROVED
  [MirrorWalk.walk_eq, openTo_walk_iff]
- From home every walk lands on (2A - 1, 2A + 1), open to every gear dividing A. PROVED
  [MirrorWalk.walk_home_odd, walk_home_even, landing_open_of_dvd]

## Part 5. The target zone

- A landing (12m - 1, 12m + 1) with 12m + 1 < P^2, the gear set holding every prime of [5, P),
  and every gear not dividing 12m missing both members, is a twin prime pair. PROVED
  [MirrorWalk.landing_twin]
- The carry cap: the distinct primes a walk carries all divide A, so their product is at most
  A, and a landing inside the window (2A + 1 <= q^2) carries at most the primes whose product
  is at most q^2 / 2. PROVED [MirrorWalk.carried_product_le, carried_product_le_window]

## Part 6. What is left, as a property of a step

Every gear the landing must be open to is handled in one of two ways, both proved above:
carried (it divides the axis product) or settled by the landing law (the axis product avoids
the two residues n and n + 2 modulo the gear). The carry cap says the carried set is small; the
one-gear theorem says each remaining gear is settled by the choice of k within three
consecutive multiples. What is not yet a theorem: one axis k M inside the window that settles
every remaining gear at once. In the walk's terms this is a property of the step, not of the
window: the residues of 2kM modulo the remaining gears must avoid n and n + 2 for each of them
simultaneously, with 2kM + 1 <= q^2. For two gears this follows from the two one-gear
statements and the Chinese remainder theorem within h h' consecutive multiples; the joint
statement over all remaining gears within the window is the open part.

## Standing of the parts

| part | statement | status |
|---|---|---|
| origin: home | open to every gear | PROVED |
| origin: gear pair | open to every gear but its members | PROVED |
| step | flip about a real axis; two steps slide | PROVED |
| landing vs last step, carried gears | openness preserved | PROVED |
| landing vs last step, any gear | one congruence of the axis product with a member of the origin | PROVED |
| one gear, one step | settled within three consecutive multiples | PROVED |
| sequence of steps | flip about the alternating sum; carried set = its divisors | PROVED |
| target zone | unstruck landing below P^2 is a twin | PROVED |
| carry cap | carried primes' product at most q^2/2 | PROVED |
| the joint step | one axis in the window settling every remaining gear | OPEN |
