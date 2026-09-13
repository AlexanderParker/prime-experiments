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

## Both levers (owner, 2026-09-13): the origin lever and settling gears one after another

Kernel proofs/MirrorWalkSettle.lean (round 47, built, 0 sorries, standard axioms).

### The origin lever

- PROVED [anchor_certifies]: a landing L is open to h if some column v open to h has h dividing
  L + v + 2; L is then the flip of v about the axis (L + v + 2)/2, which h divides. Every known
  twin, every gear pair and home is such a v for the gears it is open to, so a walk that hops
  through known twins certifies the final landing for the union of what each visited origin
  can carry: the certified set is no longer the divisors of one axis sum but the union over
  origins, and the carry cap applies per origin, not to the union.
- What the lever does at the walk's landing, with the origins available without search (home
  and the gear pairs below q):

| q | landing | origins | gears certified by some origin | largest certified | left to the landing law |
|---|---|---|---|---|---|
| 31 | 59 | 5 | 3 of 9 | 13 | 6 |
| 101 | 107 | 8 | 4 of 24 | 23 | 20 |
| 401 | 419 | 21 | 14 of 77 | 103 | 63 |
| 1009 | 1019 | 35 | 22 of 167 | 317 | 145 |
| 3001 | 3119 | 82 | 40 of 429 | 877 | 389 |

The origins certify the small gears (a gear h is certified iff some origin sits in the class
-L - 2 mod h, and the origins number about q / ln^2 q against h - 2 classes); the large gears
are left to the landing law. Same shape as the anchor coverage (anchors_walk.md), now as a
theorem about origins.

### Settling gears one after another

- PROVED [openTo_add_of_dvd]: sliding a column by a multiple of h keeps its openness to h.
- PROVED [flip_stride]: moving the axis by j D units moves the landing by 2 j D M.
- PROVED [exists_axis_open_stride]: for a prime h >= 5 not dividing 2 M D, among the axes
  (k_1 + j D) M, j = 0, 1, 2, some landing is open to h (the three landings sit in three distinct
  classes mod h, and at most two classes are struck).
- PROVED [settle_two]: distinct primes h, h' >= 5 not dividing 2M: some axis k M with
  k_0 <= k <= k_0 + 2 + 2h lands the origin on a column open to both. The first gear is settled
  within three multiples; the second by strides of the first, which keep the first settled.
- The same step adds any further gear at a stride equal to the product of the gears already
  settled (the stride must be a multiple of each settled gear so they stay settled, and
  coprime to the new one so its three landings are distinct classes). So r gears are settled
  one after another within 2 (1 + h_1 + h_1 h_2 + ... + h_1 ... h_(r-1)) multiples of M. The
  cost against the zone:

| q | gears | zone length in k | gears settleable one after another inside the zone | the next one needs |
|---|---|---|---|---|
| 11 | 3 | 9 | 1 | 12 |
| 31 | 9 | 77 | 2 | 82 |
| 101 | 24 | 841 | 3 | 852 |
| 401 | 77 | 13366 | 5 | 181032 |
| 1009 | 167 | 84756 | 5 | 181032 |

So the joint step, built by settling gears in sequence, stays inside the zone for the first
two to five gears only: the stride is the product of the settled gears and the zone grows as
q^2. The one-at-a-time construction is a proof for small sets and the exact reason it stops
is the product. The joint step for all gears at once remains the open part, now with its
per-gear cost proved.

## The unrestricted sequential walk (owner's question, 2026-09-13)

Owner: if the steps are not restricted to the window, does the sequential walk land on a pair
inside the window? Built as the greedy settle walk: start at the zone start k_0; for each gear
h = 5, 7, 11, ... in turn, if the current landing is struck by h move by the stride D (the
product of the gears already settled), trying j = 0, 1, 2; then D := D h. Every step is one
per-gear operation, blind to any destination; the final landing is open to every gear of the
machine by construction.

Machines 11 to 3000 (426): the final landing lies inside the window at 236, outside at 190.
Moves per walk: mean 2.5, at most 9. The gear that forced the last move: none (59 walks: the
zone start was already open to everything), 7 (47), 5 (45), 11 (45), 13 (27), 263 (14), 23
(14), 29 (14). First outside cases: q = 293, 347, 349, 353 all end at k = 50,708,377,339,650
(the last move forced by 43, after the stride had passed the window); q = 431, 433 end at a
k of 100 digits (last move forced by 263).

What this says:

- The walk stays inside exactly when every move happens while the stride is still small, i.e.
  when the first few gears' moves already reach a column no later gear strikes. In those cases
  the walk is the seeing rule in disguise: the column it stops on is the first aligned twin
  above k_0 reachable by those small moves.
- The walk leaves when a later gear strikes the current column after the stride has grown past
  the window: the move by the stride jumps to the scale of the product. The landing there is
  unstruck by every gear up to q but far above q^2, so nothing says it is prime. It cannot come
  back: the column's class modulo the product is fixed, and the class's representative in the
  window is the one it did not land on.
- The same final k for whole runs of machines (293 to 353) shows the walk's path depends only
  on which gears strike along the way, not on q, until a new gear enters.

So the answer is no: unrestricted, the sequential walk lands inside the window at about half
the machines and, once out, stays out. The blind per-gear walk cannot be made to land in the
window by removing the restriction; the joint step stays the open part.

## The repair walk: the consistent sequential walk (owner, 2026-09-13)

Owner: do not rule the sequential walk out; look for a consistent origin type with a consistent
per-step rule (step count may vary). Variants of the product-stride walk (origins: zone start,
primorial-adjacent, square column, window middle; orders ascending and descending; step
choices 0,1,2 and 0,-1,1; research/stack/r8/sequential_variants.py, machines to 3000): none is
consistent; the best is the baseline (236 of 426); every variant leaves when a gear strikes
after the stride has passed the window, and no origin controls the late gears' residues.

The consistent one: THE REPAIR WALK. Origin: the zone start. Rule, the same at every step: take
the smallest gear striking the current column and move forward by the smallest amount that
clears it. Stop when no gear strikes.

- Machines 11 to 20000: lands on a twin inside the window at 2258 of 2258. Steps: mean 6.4,
  maximum 41 (q = 13007). Largest move: 2 columns. Open columns passed over by a move: 0.
- Gears repaired over all walks: 5: 4478, 7: 2526, 11: 1626, 13: 1333, 17: 740, 19: 623, 23:
  463, 29: 364, 31: 287, 43: 234.

Its parts, proved (kernel proofs/MirrorWalkRepair.lean, round 48):

- clear_step: a prime gear h >= 5 striking column k misses k + 1 or misses k + 2 (its two teeth
  cover at most two of any three consecutive columns). So a step is always 1 or 2.
- A step of 2 is taken only when k + 1 is struck by the clearing gear, so a step never passes a
  column open to the machine.
- reach_first_open: a walk with steps of 1 or 2 that never passes an open column, started at or
  below an open column t, reaches an open column at or below t. So the repair walk from the
  zone start stops at the first column above q that no gear strikes.
- landing_twin: that column, if below q^2, is a twin prime pair.

What is left, as a property of this walk: that it stops before the window's end, i.e. that the
first column above q open to every gear lies below q^2. The step count is the painted run at
the zone start, L(q) (mean 8.7, max 55 to 20000, against zones of millions). The walk is blind
to any destination, every step is one gear's operation, the rule never changes, and its
termination inside the window is exactly the one open statement, now attached to the first
step sequence above q rather than to a record over the window.
