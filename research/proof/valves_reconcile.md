# The valves: reconciling the scratch lane and the review lane (2026-09-07)

The owner's hybrid first step: a scratch lane built the valves from the definitions alone
(`valves_scratch.md`; no access to the wall, the tree or the refiled facts) while a review lane
sorted every recorded interaction fact into interface objects and put predictions on record
(`valves_review.md`). Rule: found by both = solid; scratch-only = new; review-only = suspect
until reproduced in the charge picture; disagreements = predictions on record and a separating
test.

## Found by both (solid)

| object | review's prediction | scratch's finding | status |
|---|---|---|---|
| **imprint** (valve timing) | each family (s, s') occupies exactly the classes of P mod 30 and 210 solving s'P' - sP = 2; timing on the raw line is one forbidden class per engine gear per sign (predictions 1, 4) | IMPRINT: a family occupies exactly the residues mod q# with n = 0 (p divides s), n = -2 (p divides s'), n not in {0, -2} (other p <= q); size prod over odd p <= q not dividing ss' of (p - 2); in columns every burning gear pins the family to one tooth; 0 exceptions in 45,358 families and 3,486,234 column charges; CRT proof | PROOF, both coordinates |
| **the pure charge's independence of q** | every family's count at fixed Q is independent of q; raising q only adds families (prediction 2; spot-checked 8,134 twins at q = 5 and 7, 831 families unchanged) | the count identity charges = sum of N(s, s'), engine-and-manifold-open = N(1, 1) = the twins as sets, 9 of 9 runs; 440,107 at Q = 10^4 | EXACT |
| **the record is a twin gap** | the valves' record is a twin gap at every split, shortened only by air pairs with a prime neighbour (prediction 3, W103) | EMBER: every burnt charge in turns 1 and 2 has an ember member (a q-smooth number above Q); so the manifold's record in the first two turns is a twin gap unless an ember with a prime neighbour splits it; 0 exceptions, proof by the air cap plus parity | PROOF (turns 1-2), ROOT beyond |
| **the red flag held** | no manifold metric law survives into the valves' open set | the scratch lane claimed none; its "nearest burnt charge never at distance 4" is the manifold's forbidden gap read on charges, a manifold statement, not a valves one | consistent |

## Scratch only (new, with the definitions as the only input)

- **PORT** (raw line): a family's class mod 6 is fixed by its air: column port iff gcd(ss', 6) = 1,
  port 3 iff 3 divides s, port 1 iff 3 divides s', even iff 2 divides s. 0 exceptions in
  23,969,812 pairs; one-line proof. (This is the review's "the column coordinate is the first
  valve", found from the other side: the fold is the port.)
- **TURN and ONSET** (raw line; turn m = (mQ, (m + 1)Q]): a family (s, s') has no member before
  turn max(s, s'); the air of any fuelled member in turn m is at most m; turns 1 and 2 carry no
  fuelled family but (1, 1). 0 exceptions; proof n = sP > sQ. Measured: every family with
  max(s, s') <= 25 fires exactly at its turn (Q = 10^4: 25 of 25 at q = 5, 24 at q = 7, 21 at
  q = 11), and the families present in turn m are exactly the admissible pairs with
  max <= m for m = 1..12 at all three q. THE VALVES OPEN IN ORDER OF THEIR AIR.
- **INVENTORY**: (s, s') exists iff q-smooth, gcd dividing 2, same parity, and 4 divides
  exactly one of an even pair. Realised is a subset of admissible with 0 exceptions (9 runs);
  every admissible pair with max(s, s') <= 1,500 realised at q = 5, Q = 10^4. Proof: local
  solvability of s'P' - sP = 2 in units. The pre-registration missed the mod-4 clause; the data
  refused (2, 6) and the proof followed.
- **YIELD**: N(s, s') equals the imprint's local density integrated,
  (1/phi(b)) C(s, s') integral dP / (log P log((sP + 2)/s')), within 2% over 310 families at
  q = 11, Q = 10^4 (mean ratio 0.9986). Measured only. Prior art, named by the lane at the end:
  the local factors of Hardy-Littlewood / Bateman-Horn for the pair (P, (sP + 2)/s'); family
  (4, 2) is the Sophie Germain primes.
- **SPOKE**: the columns of mQ are always engine-open when q# divides Q (mQ +- 1 coprime to q#);
  twin at 11 / 37 / 248 / 2,097 / 10,303 spokes at 5# .. 17#.
- **Neighbour facts**: the nearest burnt charge to a twin is at distance 2 for 37% of twins and
  those are always the families (3a, 1) / (1, 3a'); the burnt count between consecutive twins is
  not a function of the gap (its mean is linear in the gap with a density rising by turn).
- **The first pure charge**: t_1 >= p_1 is the only exact relation; P(t_1 = p_1) falls 0.138 to
  0.093 from Q = 10^4 to 10^7, tracking 2 C_2 / log Q; correlation with p_1 - Q is 0.04 to 0.08.

## Review only (suspect until reproduced in the charge picture)

- The corridor mod 35 (= the opening set of {5, 7}); the island witness and K(d) (K(d) quantifies
  over phase vectors no integer realises and does not translate into families); the four cells
  and the level of distribution (translated into zones by the review); the zero-interaction
  region (a property of every split); theorem (E)'s exception set as the inner boundary (the
  engine's doubly occupied placements); the twisted copies as back pressure in the cofactor
  coordinate. Each is an engine-coordinate object; the scratch lane, working on the raw line,
  did not meet them. They stay PARTIAL with the review's translation attached.

## Verdict

The valves have a mechanism, found from the definitions and matching the recorded facts where
they overlap: every family is a valve with an imprint (which residues it can occupy, proved),
an onset (it opens at the turn equal to its air, proved as a bound and exact as measured), a
port (its class mod 6, proved), and a yield (its count, a local density, measured). The pure
charge is the one valve with no air: it has no onset, it is in every turn, and its count is
the same for every engine. Nothing exact yet says the pure charge has a member in a given turn;
that is the root, and it now has a shape: in turns 1 and 2 the only fuelled valve is (1, 1)
and the only other charges are embers, so the question in the first two turns is whether the
twin primes in (Q, 3Q] can be absent while the embers are present, which is the conjecture
with the window cut to two turns.

Next: the turn ledger. Per turn m, the total charges T_m, the burnt charges B_m as the sum of
the open valves' yields, and the pure charge P_m = T_m - B_m; whether an exact relation between
T_m and B_m (both counts over the same primes above Q) forces P_m > 0, and where the counting
face of the wall reappears in it.
