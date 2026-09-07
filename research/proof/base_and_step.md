# The base case and the step (branch R4.d.i)

Prover, round 2 of the stack line, 2026-09-07. Parent: R4.d (the stacked machines by squares,
research/proof/stacked_squares.md, laws S1-S9), spawned by the owner's direction of 2026-09-07:
"definitely worth pursuing the base case; the smallest machine that builds a twin pair is 2, 3."
With S7 (the twins of band k are the twin-gear pairs of machine k + 1) the conjecture along the
stack is a chain of one statement, and a chain needs a base and a step. Scripts in
research/stack/r2/ (chain.py, zone.py, bandstep.py, excess.py), outputs in research/stack/r2/results/
(untracked; every number the text uses is in the text). Laws are numbered S10 onward: S8 (the
cycle at q#) and S9 (the machine count) are already issued in stacked_squares.md section 6, so
the register continues from there. Nothing here is committed by the prover.

Prior results checked before opening (docs/novel/README.md, the tree R4.d, top_machine_1.md,
valves_scratch.md): the band structure and S7 (stacked_squares.md, exact); the first pure charge
t_1(Q) above Q = q# and 10^k is on record as MEASURED ONLY, not bounded by p_1 = nextprime(Q)
(valves_scratch.md section 4, P6); the neighbour-of-a-hit law P(open | next to a g-hit) = P(open)
g/(g - 2) (anchor-235 line, commit 88117ad) and the kernel's neighbour_of_hit; the +4 collision
law and the head collision (docs/proofs/21-collision-laws.md, ArcFloor.collision_add); the origin
clump L6 and the gear zone L21 of the top machine. What this branch can find that is not known:
the chain from the base as a table (no link count is on record), the exact object the step acts
on (how twin gears strike a band, and whether twin-gear-ness matters), the law of the start-of-band
excess in the band's own coordinate with its mechanism, and the exact relation between the first
twin of a band and the lower machines' pattern at the square.

## 1. Pre-registered (written before any script ran; verdicts filled in afterwards)

### The chain, in the owner's words

Link k is the square interval [g_k, g_k^2) with g_{k+1} = nextprime(g_k^2). From the base
g_1 = 3: [3, 9) holds (3, 5) and (5, 7); g_2 = 11, g_3 = 127, g_4 = nextprime(127^2 = 16129),
g_5 = nextprime(g_4^2), ... The base machine is {2, 3, 5, 7} (2, 3 the fold, (3, 5) and (5, 7) its
twin gears). "Link k holds a twin" is, by S7, "machine k + 1 has a twin-gear pair". The step:
machines 1..k, each with twin gears, leave a slot open on band k.

### Theory

T1 (the base and the links). Every link ever computed holds a twin pair, by a margin that grows
like g_k / ln g_k; the base is exact and the links are counts. The chains from different bases
merge (3 -> 11 is the chain from 11; 5 -> 29; 7 -> 53), so there are few distinct chains.

T2 (the step's object). A twin gear pair (p, p + 2) of machine k acts on band k exactly like two
gears of its size: it strikes the slots n = 0, -2 (mod p) and n = 0, -2 (mod p + 2) (n the lower
member), and its two gears collide on exactly four residues mod p(p + 2), namely n = 0, -2, p,
-(p + 2), which is the +4 law read on the raw line. Twin-gear-ness carries no further action: the
share of the band's blocked slots struck only by twin gears is the share their sizes predict, and
the twins of the band neither avoid nor prefer the collision cycles beyond the one-number-per-cycle
exclusion of S3, which is the neighbour-of-a-hit law (a known result; noted, not pursued).

T3 (the excess). At height x on band k, a gear g of machine k has struck a g_k-rough number iff
g g_k <= x, and has made a NEW strike (least prime factor g) iff g^2 <= x. So in the square zone
Z_k = [g_k^2, g_k g_k') (g_k' = nextprime(g_k)) machine k strikes the rough set exactly once, at
g_k^2, and the twins of Z_k are exactly the pairs the lower machines leave open there (the square's
own slot excluded). The excess ratio joint / product of S6 is therefore P(both prime | both rough)
/ P(slot open under k) with P(both prime | both rough) = 1 in Z_k, and its decay in u = ln x /
ln g_k is Buchstab's: P(prime | rough) = 1 / (u omega(u)) = 1 / (1 + ln(u - 1)) on 2 <= u <= 3.
The excess is a density, not a forcing (ROOT if used as a count).

T4 (the base-case quantity). The first twin above g_k^2 (the first twin-gear pair of machine
k + 1) is the first pair open under machines 1..k-1 above g_k^2, whenever that pair lies in Z_k; its
distance from g_k^2 is a property of the lower machines' residue pattern at the square, not of
machine k. No structural quantity of the machines below bounds it: the lower machines' pattern
contains rough-free stretches longer than Z_k (Jacobsthal), so the pattern alone cannot force a
rough number, let alone a pair, into Z_k; only the position of g_k^2 could.

### Predictions with numbers

P1. Link counts: [3, 9) 2 twins; [11, 121) 8 twins ((11,13), (17,19), (29,31), (41,43), (59,61),
(71,73), (101,103), (107,109)); [127, 16129) about 340 (twins below 16129 about 370 less those
below 127); [g_4, g_4^2) with g_4 in {16139, 16141} and g_4^2 = 2.6 x 10^8: about 700,000 (twins
below 2.6 x 10^8: 440,312 at 10^8 scaled by the Hardy-Littlewood integral, about 1.02 x 10^6 at
2.6 x 10^8). g_5^2 = 6.8 x 10^16 is out of sieve reach; the link count is stated from the twin
tables (pi_2(10^16) = 10,304,195,697,298; pi_2(10^17) = 90,948,839,353,159) and the
Hardy-Littlewood integral as an estimate, about 6 x 10^13. Every link of every chain from every
base g_1 <= 31,622 (all g with g^2 <= 10^9, one sieve): 0 empty links. Refuted by one empty link.

P2. The first twin above g_k along the base chain: above 3 it is (3, 5) (distance 0); above 11,
(11, 13) (0); above 127, (137, 139) (10); above g_4, unknown, predicted within 3 ln^2 g_4 = 280 of
g_4. Along the chain to g_10 (about 10^270, probable primes) the distance (t_1 - g_k) / ln^2 g_k
stays within [0, 3] (valves P6: mean 0.63-0.93 at 10^4..10^7). Refuted by a distance above
5 ln^2 g_k.

P3. The step's object: on every band k of q = 7..23 the fraction of blocked slots struck only by
twin gears of machines 1..k is within 0.05 of the value obtained by replacing each twin gear by
its nearest non-twin prime of the same machine (a size-matched control); the twin rate in the
other two slots of a collision cycle of a twin gear pair (p, p + 2) is (1 - 2/p)^-1 (1 - 2/(p +
2))^-1 times the band's twin rate, within the Poisson error, at every pair with >= 50 collision
cycles on the band. Refuted by a twin-only share differing from the control by more than 0.05, or
a collision-cycle twin rate off the neighbour-of-a-hit value by more than three standard errors
in a consistent direction across pairs.

P4. The gear zone: at every band and every height x, the set of gears of machine k with a
strike on a g_k-rough number below x is exactly {g : g g_k <= x}, and the set with a new strike
below x is exactly {g : g^2 <= x}: 0 exceptions. In Z_k the twins equal the rough pairs (lower
member in (g_k^2, g_k g_k')) exactly: 0 exceptions at all 16 bands of q = 7..23 and at every prime
g_k <= 10^5 (zone.py). Refuted by one exception.

P5. The excess in the band's first cycles: joint / product = 1 / P_k(open slot, measured on the
same cycles) within the Poisson error over the cycles of Z_k, at every band; along the band the
per-number ratio follows 1 / (u omega(u) P_k(open number)) to within 10 % in every bin of width
0.25 in u from u = 2 to the band's end, at q = 17, 19, 23 for k = 2 and 3. The law is universal in
u (same curve for every k and q up to the value of P_k(open), which is 1/4 per slot by S5 for every
square-built machine). Refuted by a bin off by more than 20 %.

P6. The base-case quantity over all primes g <= 10^5: the first twin above g^2 coincides with the
first pair open under the primes below g above g^2 in every case but at most 3 (a rough pair lying
above g g' with a rough composite member); the first rough pair above g^2 lies inside Z_g in every
case but at most 3; the minimum over g of the twin count in Z_g is attained at a small g (g <= 13)
and the count grows like g / ln g (least-squares slope of twins(Z_g) ln g / g within [0.5, 1.5] of
its Hardy-Littlewood value 2 C_2 (g' - g) / ln g ... stated as: the ratio twins(Z_g) / (2 C_2 g
(g' - g) / ln^2 g x 30/8 correction-free) has mean within 0.7-1.3 over g in [10^4, 10^5]).
Refuted by 4 or more coincidence failures, or a zone with 0 twins at any g >= 7.

P7 (the structural test of T4). The Jacobsthal function of the primes below g, j(g#) (the longest
run of integers with a factor below g), exceeds the zone length g (g' - g) for the g at which
g' - g is small: at g = 7 the zone is 28 long and j(7#) = 22 (no), at g = 11 zone 22, j(11#) = 34
(yes, exceeds), and for every prime g >= 11 with g' - g = 2 the zone 2g is shorter than j(g#).
So the lower machines' pattern alone contains rough-free stretches longer than the zone at
every twin prime g: the pattern alone cannot force a rough number into Z_g; ROOT. Refuted if
j(g#) < 2g for some twin prime g >= 11 (known values j(p#): 4, 6, 10, 14, 22, 26, 34, 40, 46, 58,
66, 74, 90, 100 for p = 2..43).

### The owner's predictions (recorded so a refutation in the owner's favour is visible)

O1. The base case is worth pursuing: the smallest machine that builds a twin pair is 2, 3 (the fold),
and the chain from it has a base and a step. Prediction as read: the base is exact (link 1 holds
(3, 5) and (5, 7)) and every link holds a twin.
O2 (implicit in the brief). The start-of-band excess is a mechanism: at the band's start only
machine k's smallest gears have acted, so the open slots near g_k^2 are the lower machines' rough
pairs thinned only by gears up to x / g_k; the brief asks whether this forces a twin near every
band's start.

### Scorecard

| # | owner | claim | verdict |
|---|---|---|---|
| P1 | prover | link counts; 0 empty links over every base to g^2 <= 10^9 | (filled below) |
| P2 | prover | first twin above g_k along the base chain within 3 ln^2 g_k | (filled below) |
| P3 | prover | twin gears act like size-matched gears; collision cycles = neighbour-of-a-hit | (filled below) |
| P4 | prover | gear zone exact; twins of Z_k = rough pairs of Z_k, 0 exceptions | (filled below) |
| P5 | prover | excess = 1 / P_k(open) in Z_k; Buchstab decay within 10 % per bin | (filled below) |
| P6 | prover | first twin above g^2 = first rough pair above g^2 in all but <= 3 of g <= 10^5 | (filled below) |
| P7 | prover | j(g#) > 2g at every twin prime g >= 11: the pattern alone cannot force a rough number into Z_g | (filled below) |
| O1 | owner | the base is exact and every link holds a twin | (filled below) |
| O2 | owner (as read) | the excess is a mechanism that forces a twin near every band's start | (filled below) |
