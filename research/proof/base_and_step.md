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

### The previous lane's scorecard, filled from its result files (2026-09-09, before the new runs)

The lane of 2026-09-07 ran chain.py and zone.py (results/chain_a.json, chain_b.json, zone.json
on disk) and did not fill its scorecard. Filled from those files by the present lane, without
re-running: P1 CONFIRMED (3,401 bases g with g^2 <= 10^9; 1 empty link, the trivial [2, 4); 0 empty
links from g = 3 on; the sieve reproduces pi_2(10^8) = 440,312; link counts 276 at [127, 16129),
1,028,184 at [16141, 16141^2) as filed, against 289 and 1,029,119 from the Hardy-Littlewood
integral). P2 REFUTED in its number: along the base chain the first twin above g_k lies at
distance 0, 0, 10, 0, 0, 1170, 478, 30004, 156214, 143958 for k = 1..10, and 30004 / ln^2 g_8
with g_8 of 68 digits (ln g_8 = 155) is 1.25, 156214 / ln^2 g_9 (ln = 310) is 1.63 and 143958 /
ln^2 g_10 (ln = 621) is 0.37: within 3 ln^2 g; but over ALL bases g <= 31622 the maximum of
(first twin above g^2 - g^2) / ln^2(g^2) is 7.32 at g = 8699 (distance 2410), above the refutation
line 5. P4 and P6 CONFIRMED (0 coincidence failures, 0 in-zone failures, 0 rough composites in
any zone, minimum zone twin count 1 at g = 11 and 17, over 9,589 primes g <= 10^5 and 3,242 zones
g <= 30,000). P3 and P5 were not run (bandstep.py and excess.py have no result files); left
unfilled. P7 CONFIRMED from the known values (j(11#) = 34 > 22). O1 CONFIRMED as far as computed
(every link of every base holds a twin). O2 not tested by that lane.

## Part II. The recursion: the generated survivor set and the step (branch R4.d.i.b, 2026-09-09)

Prover, third round of the stack line. Parent: the review of research/proof/step_evidence.md
(2026-09-08), whose conclusion was that the one input no counter-machine reproduces is the
recursion of proof_skeleton.md section 6: machine k + 1's gears are the survivors of machines
1..k in section k + 1. The three counter-machines of skeleton section 10 (shifted gears, the
saturated machine with the section's twins as gears, the parity set) all kill the count and all
violate that recursion. So the step, if it has a proof, uses a property of GENERATED survivor
sets that arbitrary prime sets lack. This part names that property exactly, tests it against the
counter-machines, and writes the step as a statement about the composite machine's record on
one section. Scripts: research/stack/r2/generated.py (the chains: the explicit construction,
the properties table, the composite record per section), research/stack/r2/record_scan.py (the
composite record on the section [q^2, q'^2) for every prime q with q'^2 <= 10^9, and the
chain-link pairs), research/stack/r2/period_record.py (the full-period record of the composite
machines small enough to scan). Outputs in research/stack/r2/results/ (untracked); every
number used is in this text. Laws numbered S10 onward (S8 and S9 are issued in
stacked_squares.md). Vocabulary: the raw line with the anchor 2, 3, 5 as the clock (cycle j =
the numbers 30j .. 30j + 29, three slots per cycle at 30j + 11/13, 17/19, 29/31); cut c_{k+1} =
p_k^2 with p_k the first prime at or above c_k; section k + 1 = [c_{k+1}, c_{k+2}); machine 1 = the
primes from 7 up to c_2, machine k + 1 = the primes of section k + 1. The gears of machines 1..k
are therefore exactly the primes 7 <= p < p_{k+1}, and P_k = their product times 30 is "the lower
product".

Prior results checked before opening: docs/novel/README.md entries paired-jacobsthal-values
(the free-phase two-class covering record h_2(p_n#) of Ziller-Morack, OEIS A288815: 2, 6, 18,
30, 66, 150, 192, 258, 366, 450, 570, 708, 894, 1044, 1284, 1422, 1656, 1902, 2190, 2460, 2622 for
p_n = 2..73, in numbers on the raw line, with the record "Conjecture 6 h_2(n) < p_n^2 - p_n
implies the project's target statement" already on file, 2026-09-06), j2-upper-bound (the best
proved upper bound on the paired function is polynomial, exponent 4.266 + eps, and the exponent
2 that the target needs sits below the parity floor), j2-lower-ladder (h_2(P(z)) >= 1.349 z log z
proved; the empirical growth c z log^2 z with c about 2.5), research/harvest/r1/jacobsthal_check.md
(the real-phase full-period record of {5..p_n} is 6 F(M) = 12, 30, 42, 66, 108, 150, 204 numbers
at p_n = 5..23, below h_2 from p_n = 11 on), phase_zero.md V14 and V17 (saturation, and the
saturated counter-machine), turn_ledger.md V2 and valve_existence.md V12 (the structural laws
are blind to the fuel's primality; the only separating invariant of the charge set is phase
zero, whose existence consequence is the root), stacked_squares.md S1-S9, and step_evidence.md.
What this part can find that is not on record: the exact separating property of a GENERATED
survivor set stated as a set property and tested on the same section against the three
counter-machines at once (V12 tested the fuel's properties, not the section's survivor set as
the next machine); the composite record on each section along the chains against the section's
length and the previous section's record (no such table exists: the project's records are
full-period records of small machines, and the section is a slice); which links of the chains
are already proved by the free-phase record table; and the growth law of the section record
over every cut q^2 <= 10^9 with counts. What it will NOT do: re-derive Conjecture 6's implication
(on record), or the Jacobsthal bounds.

### Pre-registered (written before any script of this part ran; verdicts filled in afterwards)

#### Theory

T5 (the generated set is the coprime set, and that is all it is). Machine k + 1 = {n in
[p_k^2, p_{k+1}^2) : gcd(n, P_k) = 1}, P_k = 30 x prod of the primes 7 <= p < p_{k+1} (skeleton
section 5; there is no prime in [p_k^2, p_{k+1}) by the definition of p_{k+1}, so the primes below
p_{k+1} are exactly the anchor and the gears of machines 1..k). Every property of the generated
set on its section is a property of "the numbers coprime to P_k on one interval". The properties
an arbitrary prime set of the same size on the same section lacks are therefore exactly two,
and both are set-theoretic, not counts: (M) MAXIMALITY, every number of the section coprime to
P_k is a member (the saturated counter-machine V17 and a random subset both violate it: they
drop members); and (I) the gear set is an INITIAL SEGMENT of the primes, every prime below the
cut is a gear and no prime above it (V17 violates it by adding the section's twin members, which
lie above the cut, as gears). (M) is what the recursion gives and (I) is what the recursion
forbids. Every forbidden-configuration law of the manifold (no three members at n, n + 2, n + 4;
no member in the class 0 of a lower gear; the residue laws of consecutive gaps) is inherited by
every SUBSET of the coprime set, so the counter-machines (b) and (c) satisfy all of them; the
parity set (d) violates the class-0 law because its members carry lower-gear factors. No
census, mirror or character identity is exact on a slice: the section is a slice of length
about p_k^4 in a period of length exp(p_k^2), it is not self-mirrored, and every class count on
it deviates from the period average by a square-root fluctuation.

T6 (the step through the recursion is a record statement). By skeleton section 5, on section
k + 1 a slot is open under machines 1..k iff it is a twin prime pair, and by T5 the open slots are
the pairs (n, n + 2) with both members coprime to P_k. So the step at link k is exactly: THE
LONGEST RUN OF CONSECUTIVE STRUCK SLOTS OF MACHINES 1..k INSIDE SECTION k + 1 IS SHORTER THAN THE
SECTION. Two records bound it from above: the real-phase full-period record R(P_k) of the
composite machine (the longest run of struck slots anywhere in its period), and the free-phase
two-class covering record h_2 of the same gear sizes (any phases; Ziller-Morack's function),
with section record <= R(P_k) <= h_2(gears) + O(1). If h_2 of the primes below the cut is shorter
than the section, the step at that link is PROVED by the covering record alone, with no
reference to the section's position. The exact table of h_2 reaches p_n = 73, so this proves
the links whose lower gears are the primes below 73 + something: link 1 of base 3 (gears above
the fold 2, 3: {5, 7}, h_2 = 30 against a section of 112 numbers), link 1 of base 5 ({5..23}, 366
against 816) and link 1 of base 7 ({5..47}, 1284 against 2760). No other link of any chain is
within the table, and the proved upper bounds on h_2 (exponent 4.266) are a full power above the
section (exponent 2 in the largest gear), so from link 2 on the reduction is the target
statement at the cut (ROOT) and is the README's recorded "Conjecture 6 implies the target" read
in the stack's coordinate. The recursion adds nothing to the record: the record of a machine is
a function of its gear sizes and phases, the recursion fixes the sizes to be the initial
segment of primes and the phases to be zero, and that is the definition of the real machine.

T7 (the growth of the section record). The composite record on section k + 1 is the largest
twin-free run of slots in [p_k^2, p_{k+1}^2). Its growth is the growth of maximal twin gaps,
about (ln x)^2 numbers at height x up to a constant near 1-2 (Cramer-type for pairs), i.e. about
(ln x)^2 / 10 slots, while the section has about p_k^4 / 10 slots. Along a chain the cut is
squared at each link, ln x doubles, and the record per section grows by a factor about 4 while
the section grows by the square: the ratio record / section falls like ln^2 p_k / p_k^2. The
previous section's record does not determine the next (the record is a property of the primes
near height x, not of the gears' gaps): the ratio record(k + 1) / record(k) scatters around 4.

#### Predictions with numbers, and what would refute each

Q1 (the explicit construction). Machines built recursively from the base by striking multiples
of the members of the machines below (no primality test in the construction) equal the primes
of every section: 0 mismatches at the 8 full sections (base 3: [9, 121), [121, 16129), [16129,
260,467,321); base 5: [25, 841), [841, 727,609); base 7: [49, 2809), [2809, 7,946,761); base 13:
[169, 29,929), [29,929, 896,822,809)) and on the prefixes to 10^9 of the three sections that run
beyond it (base 5 from 727,609, base 7 from 7,946,761). Base 11's chain merges with base 3's at
[121, 16129). Refuted by one mismatch.

Q2 (the properties table). On every full section, against (a) the generated set G, (b) the
saturated counter-machine's survivors B = G minus the twin members, (c) a random subset C of G
with |C| = |B|, (d) the parity set D = {s P in the section : P prime >= p_{k+1}, s coprime to 30
with an even number of prime factors} (the Liouville-negative charges, which contain G): (M) the
count of numbers coprime to P_k missing from the set is 0 for G and D, 2 x twins for B, |G| - |B|
for C; (I) the count of members divisible by a lower gear is 0 for G, B, C and positive for D;
the forbidden patterns (a member in class 0 of a gear 7 <= g < p_{k+1}; three members at n, n + 2,
n + 4; consecutive gaps (2, 4) or (4, 2)) have count 0 for G, B and C at every section; the
reflection of the section's members through the section's midpoint lands on members at the
chance rate (within 3 standard deviations of |set|^2 / (numbers coprime to 30 in the section))
for all four sets; the class census mod 7, 11, 13 and mod 30 has every nonzero class within 4
standard deviations of the mean for G, B, C, and no two classes exactly equal beyond chance
(fewer than 2 exact ties per section among the classes mod 7, 11, 13); the character sums are
never zero. Refuted by an exact identity (a census tie or a zero character sum) that holds at
every section, or by a forbidden pattern in G.

Q3 (the ends). The smallest member of every generated set is p_{k+1} (the first prime at or
above the cut), exactly, 8 of 8; the first twin above the cut lies within 160 numbers (5.3 cycles)
of it at every full section and every prefix, the maximum being the 160 of base 7's section 2
(cut 2809, first twin 2969), and the last twin lies within 1000 numbers of the section's end at
every full section. Refuted by a first twin above 160 numbers.

Q4 (the composite record per section, exact). At every full section the longest run of struck
slots of machines 1..k equals the longest twin-free run of slots (0 mismatches); its length in
slots is at most 3 % of the section's slots at [121, 16129) and at most 0.01 % at the sections
above 10^8; on the section [9, 121) it is 2 slots of 11 (77/79, 89/91 struck by 7), on [25, 841)
at most 10 of 82. Refuted by a ratio above 10 % at any section from [121, 16129) on.

Q5 (the scan). Over every prime q with 7 <= q and q'^2 <= 10^9 (about 3,400 sections), the
section record (longest twin-free slot run in [q^2, q'^2), counting the runs from the section's
start to the first twin and from the last twin to its end) is shorter than the section at every
q (this is the count: 0 empty sections); the largest ratio record / section over q >= 7 is at a
q below 100 and is below 0.5; for every q >= 1000 the ratio is below 0.02; the record in numbers
is within [0.5, 4] x (ln q^2)^2 for at least 95 % of the sections. Refuted by a ratio above 0.5,
or by fewer than 90 % of the sections in the stated band.

Q6 (the chain pairs). For every prime r with r'' = nextprime(r^2) and nextprime(r''^2)^2 <= 10^9
(about 40 pairs), the ratio record[r''^2, nextprime(r''^2)^2) / record[r^2, r''^2) has median in
[2, 8] and no value above 30; the previous section's record does not determine the next (the
Spearman correlation of the two records, after removing the trend in ln r, is below 0.5 in
absolute value). Refuted by a median outside [2, 8].

Q7 (the record proves three links). h_2({5, 7}) = 30 < 112, h_2({5..23}) = 366 < 816,
h_2({5..47}) = 1284 < 2760: the first link of the chains from 3, 5, 7 holds by the free-phase
covering record; for the first link of base 11 (gears {5..113}, section 16008) and base 13
({5..167}, section 29760) no exact h_2 is on record, the proved bound is far above the section,
and the measured growth (about 2.5 z ln^2 z: 6,300 and 11,000) sits below the section. The
real-phase full-period record of the composite machine of base 5's link 1 (gears 7..23 above the
anchor, period 7,436,429 cycles) in slots is about 20 (6 F(23) = 204 numbers), below the section's
82 slots; the section's own record is smaller than the full-period record at every scannable
case. Refuted by a full-period record above the section.

The owner's prediction, as read from the brief (O3): "IF the real record of the composite
machine is below the section length, the step holds at that k by the record alone", and the
question whether the recursion (each machine's gears are the previous section's survivors,
whose gaps are bounded by the previous record) gives the record a bound. Recorded prediction:
the record is below the section at every computed cut, by a margin growing like p_k^2 / ln^2 p_k;
the recursion gives the record no bound beyond "the gears are all the primes below the cut".

#### Scorecard

| # | owner | claim | verdict |
|---|---|---|---|
| Q1 | prover | recursive construction = primes, 0 mismatches at 8 sections + 3 prefixes | **CONFIRMED** (11 full runs + 5 prefixes, 1,333,333,231 numbers coprime to 30, 0 mismatches; the pre-registered counts 8 and 3 are miscounts of the 9 and 5 the chains actually have) |
| Q2 | prover | (M) and (I) are the only separating properties; forbidden patterns inherited by subsets; no exact slice identity | **HELD in the main, THREE clauses REFUTED**: no exact identity and no forbidden pattern in G at any section; but (2, 4) is not a forbidden pattern, the reflection is the Goldbach count and not chance, and B is separable from G by its class census (161 sd), which T5 did not allow for |
| Q3 | prover | smallest member = p_{k+1}; first twin within 160 numbers at every section | **HELD at the 11 full sections** (max offset exactly the predicted 160, base 7 section 2), **REFUTED at the prefixes** (268 at base 5 from 727,609); last twin within 1,000 of the end at every full section (max 362) |
| Q4 | prover | struck-slot record = twin-free record; <= 3 % of the section at [121, 16129), <= 0.01 % above 10^8 | **HELD on three clauses** (0 mismatches at 16 of 16; 1.686 %; 0.00133 % and 0.00053 %), **REFUTED on the fourth**: [25, 841) has 14 of 82 slots, not "at most 10" |
| Q5 | prover | scan: record < section at every q; max ratio < 0.5 at q < 100; < 0.02 for q >= 1000; 95 % within [0.5, 4] (ln q^2)^2 | **HELD on the count** (3,397 sections, 0 empty), **REFUTED on all three numbers**: max ratio 0.6667 at q = 29; 653 sections with q >= 1000 above 0.02 (max 0.1744 at q = 1,289); 23.31 % in the band, 0 below it and 2,605 above |
| Q6 | prover | chain pairs: record ratio median in [2, 8]; no determination by the previous record | **REFUTED**: there are 5 pairs, not "about 40" (the cap forces r <= 13); ratios 13.500, 7.143, 9.500, 12.852, 9.714, median 9.714 outside [2, 8]; residual Spearman 0.90, above the 0.5 line, on 5 points |
| Q7 | prover | h_2 proves link 1 of bases 3, 5, 7 and no other link; full-period record of {7..23} about 20 slots < 82 | **CONFIRMED** (30 < 112, 366 < 816, 1,284 < 2,760; full-period record 19 slots, largest cyclic gap 204 numbers, reproducing jacobsthal_check exactly), **one clause REFUTED**: at base 3 link 1 the section record equals the full-period record (2 = 2), it is not smaller |
| O3 | owner (as read) | the record is below the section at every cut, and the recursion may bound the record | **CONFIRMED as far as computed**: record < section at all 3,397 cuts and all 16 chain runs, margin 1.500 (q = 29) to 1,493.8 (q = 31,397); the recursion gives no bound, and the record's law is an extreme value (S18) |

### Part II filled from the result files (prover, 2026-09-10, lead U1 of research/proof/tree_review.md)

The lane that pre-registered Q1-Q7 and O3 died at the weekly limit before writing a verdict. This
section fills the scorecard above from `research/stack/r2/results/`. What was on disk and used as
it stood: `record_scan.json` and `record_scan_rows.npz` (the complete scan, 3,397 sections with
`q'^2 <= 10^9`, and the chain pairs) and `twins_1e9.npy` (3,424,506 twins to 10^9, reproducing
`pi_2(10^8) = 440,312` and `pi_2(10^9) = 3,424,506`). Two things had to be run because the result
was missing or was not the pre-registered one:

- `generated.json` on disk was a reduced run (`N = 10^5`, bases 3 and 5 only), not the
  pre-registered `N = 10^9` over the five bases. Re-run as pre-registered: **386.5 s on one core**,
  well inside the half-hour allowance.
- `period_record.json` did not exist at all (Q7's full-period clause). Run: **1 s**.

Nothing else was recomputed; the mirror recount of Q2 and the extreme-value fit of the closing
section are new reads of files already on disk. Scripts of this fill:
`research/stack/r6/part2_fill.py`, `gen_report.py`, `mirror_check.py`. Laws are numbered **S18
onward**: S15-S16 are issued in dead_branches_reopened_4.md and S17 is reserved by
leftover_depth.md, so tree_review.md's instruction "it must start at S15" is stale.

#### Q1. The explicit construction is the primes. CONFIRMED.

The five chains give **11 full section runs and 5 prefix runs** (9 and 3 of them distinct: base
11's chain merges into base 3's at `[121, 16129)`, so its two rows repeat base 3's). Over all 16
runs the recursive construction - striking multiples of the members of the machines below, with no
primality test anywhere in it - was compared against an independent Eratosthenes sieve on
**1,333,333,231 numbers coprime to 30**: **0 mismatches**. The gmpy2 spot check on samples drawn
through the run: **0 of 104,251 members not prime, 0 of 104,426 non-members prime**. The
pre-registration's "8 full sections" and "3 prefixes" are miscounts of its own list (which names 9
and 2); the true figures are 9 distinct full sections and 3 distinct prefixes.

#### Q2. The properties table. Held in the main; three clauses refuted, one of them usefully.

Read set by set on the 11 full runs (`G` = the generated set, `B` = `G` minus the twin members
(V17's survivors), `C` = `G` minus a random subset of the same size, `D` = the Liouville-negative
charges).

**(M) maximality.** Numbers of the section coprime to the lower product that are missing: **0 for
`G` at 11 of 11, 0 for `D` at 11 of 11**. For `B` the count is `2 x (twins of G)` exactly at 8 of
the 11 and `2 x twins + 1` at three - `[121, 16129)`, `[25, 841)`, `[169, 29929)`. The excess is
exact and explicable: the section's top is `p_k^2` and the pair `(p_k^2 - 2, p_k^2)` survives the
LOWER gears (the only divisor of `p_k^2` is `p_k`, which is not one of them), so when `p_k^2 - 2`
is prime the member `p_k^2 - 2` is flagged as a twin member although its partner lies outside the
section. That is the case at exactly those three sections (16127, 839, 29927 prime; 119, 2807,
727607, 7946759, 260467319, 896822807 composite). It is the same phenomenon that
research/proof/frontier_floor_1e7.md measures at 11.27 % of all cuts to 10^7.

**(I) initial segment.** Members divisible by a lower gear: **0 for `G`, `B`, `C` at 11 of 11**;
positive for `D` at 9 of 11 (66, 14,094, 228,906, 185, 10,523,643, 39,294,738, ...), and **0 at the
two smallest sections** `[9, 121)` and `[25, 841)`, where `D = G` exactly because no
Liouville-negative charge with a lower-gear factor fits. So (I) separates `D` from `G` at 9 of 11
sections, not at all 11.

**Forbidden patterns.** Triples `n, n+2, n+4`: **0 for all four sets at 11 of 11**. A member in
class 0 of a lower gear: 0 for `G`, `B`, `C` at 11 of 11 (the `0/1/1` at `[9, 121)` is 11 and 13
themselves, which are members there and not gears there). Consecutive gaps `(2, 4)` or `(4, 2)`:
**REFUTED as a forbidden pattern** - it is legal and common (10 at `[9, 121)`, 135 at
`[121, 16129)`, 245,016 at `[16129, 260467321)`, 691,971 at `[29929, 896822809)`). `B` has 0 of
them only because removing every twin member removes every gap 2. The pre-registration misfiled a
legal word as forbidden.

**The reflection.** REFUTED as pre-registered ("lands on members at the chance rate, within 3
standard deviations"), and the deviation is fully accounted for by two things, neither of them a
slice identity:

1. `M` is a multiple of 30, so the image `M - 2 - n` is coprime to 30 for only **3 of the 8 classes
   mod 30** (`n = 11, 17, 29`): the ceiling on ordered hits is `3/8` of the naive chance. Measured
   share of members in those three classes: 0.3751 and 0.3750 at the two sections recounted.
2. Conditioned on that, the count is the Hardy-Littlewood Goldbach count. Recounted independently
   (`mirror_check.py`): at `[2809, 7946761)`, `M - 2 = 2 x 7 x 13 x 31 x 1409`, ordered hits 64,574
   `= 0.4761` of chance against the prediction `2 C_2 (8/30) prod_{p | M-2, p odd} (p-1)/(p-2)
   = 0.4771`; at `[16129, 260467321)`, `M - 2 = 2 x 8069 x 16141`, hits 1,021,230 `= 0.3509`
   against 0.3522. Agreement to 0.2 % and 0.4 %.

That is a known count and it is filed as known, not pursued. One measurement fault was found on the
way: `generated.py` counts a same-segment reflection pair twice and a cross-segment pair once, so
its reported ratio halves on every section longer than one 2^24 segment (0.181 reported against
0.3509 true at `[16129, 260467321)`; single-segment sections such as `[2809, 7946761)` agree with
the recount exactly, 64,574 = 64,574).

**The class census.** Largest deviation of a nonzero class mod 7, 11, 13 from the mean, in standard
deviations: `G` at most **0.46** over all 11 sections, `C` at most **1.82** - both inside the
predicted 4. `B` **REFUTED**: 3.09 at `[121, 16129)`, 9.57 at `[841, 727609)`, 24.01 at
`[2809, 7946761)`, 96.53 at `[16129, 260467321)`, **161.15 at `[29929, 896822809)`**, growing with
the section. The mechanism is exact and it matters: `B` removes the twin members, and a twin's
lower member is barred from two residues mod every gear `g` (class 0 and class `-2`), so the
removed set is class-biased and its removal prints that bias onto `B`. `C` removes the same NUMBER
of members at random and shows nothing. So the saturated counter-machine is separable from the
generated set by a census of its own members, which T5's "the properties an arbitrary prime set
lacks are exactly (M) and (I)" did not allow for. It is still a consequence of maximality - it is
WHICH members are missing - but it is the first separating property on this branch that is
measurable on the set alone, without reference to the coprime set it came from.

**Ties and characters.** Exact ties among the classes mod 7, 11, 13 for `G`: 4/7/9 at `[9, 121)`
down to 0/0/0 at the three largest sections - the "fewer than 2 per section" clause fails at the
small sections by pigeonhole (few members spread over `g - 1` classes) and holds at the large ones.
Legendre character sums for `G`: **none of the 33 values is zero** (the refutation line is not
reached); `B` has `chi_7 = 0` at two small sections. A pattern worth naming and not pursuing: **all
33 of `G`'s character sums are negative**, at every section and every modulus. That is the
Chebyshev bias, a known result; recorded, not opened.

**Verdict on T5.** The refutation line ("an exact identity holding at every section, or a forbidden
pattern in `G`") is not reached: no census tie and no zero character sum survives to the large
sections, and `G` has no forbidden pattern anywhere. T5 stands, sharpened by the census finding
above.

#### Q3. The ends. Split.

The smallest member of every generated set is `p_{k+1}` at **16 of 16 runs** (true by construction
and confirmed). The first twin above the cut, in numbers: `[9, 121)` +2, `[121, 16129)` +16,
`[16129, 260467321)` +10, `[25, 841)` +4, `[841, 727609)` +16, `[49, 2809)` +10,
`[2809, 7946761)` **+160**, `[169, 29929)` +10, `[29929, 896822809)` +82. So at the 11 full
sections the maximum is **exactly the predicted 160**, at base 7 section 2 (cut 2809, first twin
2969) - the pre-registration named the maximiser correctly. At the prefixes it is **REFUTED**: base
5's prefix from 727,609 has its first twin at 727,877, **+268**. The last twin lies within 1,000
numbers of the end at every full section: the largest gap is **362** (`[16129, 260467321)`, last
twin 260,466,959); the others are 14, 62, 14, 110, 8, 80, 50, 128.

#### Q4. The record per section. Three clauses held, one refuted.

The longest run of struck slots of machines 1..k equals the longest twin-free run of slots at **16
of 16 runs, 0 mismatches** (the two are computed by different routes in `generated.py`: one from
the survivor mask, one from the pair list). The ratios: `[121, 16129)` **27 of 1,601 = 1.686 %**
(predicted at most 3 %); `[16129, 260467321)` **347 of 26,045,119 = 0.00133 %** and
`[29929, 896822809)` **476 of 89,679,288 = 0.00053 %** (predicted at most 0.01 % above 10^8).
`[9, 121)`: **2 slots**, the pair of slots 77/79 and 89/91 both struck by 7 - the mechanism is as
predicted, but the section has **12** slots, not the 11 the pre-registration wrote. `[25, 841)`:
**14 of 82 slots**, against the predicted "at most 10" - **REFUTED**.

#### Q5. The scan. The count held; every number refuted.

3,397 sections, `q` from 7 to 31,601, every prime `q >= 7` with `q'^2 <= 10^9`.

- **0 empty sections.** The record is shorter than the section at every one of the 3,397 cuts.
  This is the clause that matters and it holds.
- Largest ratio record/section: **0.6667 at `q = 29`** - the section `[841, 961)`, 8 struck slots
  of 12, holding just two twins (857 and 881). The argmax is below 100 as predicted, but 0.6667 is
  **above the 0.5 refutation line**.
- "below 0.02 for every `q >= 1000`": **REFUTED**, 653 of the 3,232 sections with `q >= 1000` are
  above it; the maximum is **0.17442 at `q = 1,289`** (section `[1661521, 1666681)`, 90 slots of
  516, 30 twins), and 0.06143 at `q = 10,499`, 0.0308 at `q = 23,537`.
- "record in numbers inside `[0.5, 4] (ln q^2)^2` for at least 95 %": **REFUTED**, the share is
  **23.31 %**, with **0 sections below the band and 2,605 above it**. The band is mis-centred, not
  too narrow: the ratio runs from 0.760 (`q = 13`) to 11.478 (`q = 26,423`) with median 4.867, and
  **98.91 % lie in `[0.5, 8]`**. The correct shape is S18 below.

#### Q6. The chain pairs. Refuted, including its own count of pairs.

There are **5 pairs**, not "about 40": `r`, `r'' = nextprime(r^2)`, `nextprime(r''^2)` with
`nextprime(r''^2)^2 <= 10^9` forces `r <= 13`, so `r = 3, 5, 7, 11, 13`. Their record ratios are
**13.500, 7.143, 9.500, 12.852, 9.714**; median **9.714, outside `[2, 8]` - REFUTED**; none above
30, as predicted. The residual Spearman correlation after removing the trend in `ln r` is **0.90**,
above the 0.5 line, so the "no determination by the previous record" clause also fails on the
number - but on five points it carries no weight and should not be read as a finding. The fits
themselves: `log(previous record) = -1.040 + 1.923 log r`, `log(next record) = 1.347 + 1.893 log r`.

#### Q7. The record proves three links. Confirmed; one clause refuted.

`h_2({5, 7}) = 30 < 112`, `h_2({5..23}) = 366 < 816`, `h_2({5..47}) = 1,284 < 2,760`: the first
link of the chains from 3, 5 and 7 holds by the free-phase covering record alone, and the sections
in numbers are exactly 112, 816 and 2,760. No other link is inside the exact table, which stops at
`p_n = 73`. The full-period real-phase records of the composite machines, run for this fill
(`period_record.py`, 1 s):

| gears | period in cycles | record in slots | largest cyclic gap in numbers |
|---|---|---|---|
| {7} | 7 | 2 | 30 |
| {7, 11} | 77 | 3 | 42 |
| {7, 11, 13} | 1,001 | 6 | 66 |
| {7..17} | 17,017 | 10 | 108 |
| {7..19} | 323,323 | 14 | 150 |
| {7..23} | 7,436,429 | **19** | **204** |

The number column reproduces research/harvest/r1/jacobsthal_check.md exactly (30, 42, 66, 108,
150, 204). So base 5's link 1 has a full-period record of **19 slots against the section's 82** -
the predicted "about 20", and the refutation line (a full-period record above the section) is not
reached. The last clause, "the section's own record is smaller than the full-period record at every
scannable case", is **REFUTED at the smallest case**: base 3 link 1 has section record 2 and
full-period record 2, equal. At base 5 link 1 it is 14 against 19, smaller as predicted.

#### O3. The owner's reading. Confirmed as far as computed; the recursion bounds nothing.

The record is below the section at all 3,397 cuts of the scan and at all 16 chain runs. The margin
`section/record` runs from **1.500** (`q = 29`) through a median of **103.1** to **1,493.8**
(`q = 31,397`). "A margin growing like `p_k^2 / ln^2 p_k`" is the right shape but not the right
constant: `margin / (q^2 / ln^2 q)` has median `1.0 x 10^-4` and maximum 0.2705, because the
section's slot count is `(q'^2 - q^2)/10` and is therefore set by the prime gap `q' - q`, which
fluctuates - the margin's minimum sits where the gap is 2 (`q = 29`, section 12 slots) and its
maximum where the gap is 72 (`q = 31,397`, section 452,635 slots). On the second half: **the
recursion gives the record no bound**. Q2 is the evidence - what the recursion supplies is (M) and
(I), two set-theoretic properties, and neither is an inequality on a run length; nothing in the
scan or the properties table produced one.

### What Part II says about the step

**The records against the section lengths, along the scan.** By decade of `q`, the ratio
record/section:

| `q` | sections | median ratio | max ratio (at `q`) | median record (slots) | max record | median section (slots) |
|---|---|---|---|---|---|---|
| 7 - 100 | 22 | 0.2453 | 0.6667 (29) | 8.5 | 20 | 33.5 |
| 100 - 1,000 | 143 | 0.0784 | 0.3256 (107) | 44 | 144 | 576 |
| 1,000 - 10^4 | 1,061 | 0.0191 | 0.1744 (1,289) | 124 | 286 | 6,619 |
| 10^4 - 31,601 | 2,171 | 0.0066 | 0.0614 (10,499) | 205 | 476 | 31,128 |

**The largest ratio record/length seen anywhere in the scan is 0.6667**, at `q = 29`: the section
`[841, 961)` has 12 slots, holds exactly two twins (857 and 881), and its record is the 8-slot
run from 881 to the section's end. 97 of the 3,397 sections exceed 0.1, 261 exceed 0.05, 818
exceed 0.02.

**No section had no twin: 0 of 3,397.** The sparsest are `q = 11` (`[121, 169)`, 5 slots, 2 twins),
`q = 17` (`[289, 361)`, 7 slots, 2 twins) and `q = 29` (12 slots, 2 twins); only those three have
two or fewer, and the median section holds 673.

**S18 (THE SECTION RECORD IS AN EXTREME VALUE; measured over 3,397 sections, mechanism known).**
Write `mu(x) = ln^2 x / (2 C_2)` for the Hardy-Littlewood mean twin gap at height `x` and `T` for
the number of twins in the section. Then the composite record of the section `[q^2, q'^2)`, in
numbers, is

    record = c x mu(q^2) x ln T ,     c with median 0.9831 over the 3,397 sections,

and `c` is flat across the scan: median **1.2411 / 0.9789 / 0.9791 / 0.9853** on the four decades
of `q` above, i.e. constant to 1 % over three decades; quartiles `[0.8835, 1.1102]`, 95th
percentile 1.3848, 7 sections above 2 and the maximum 3.3599 at `q = 29` (the 12-slot section,
where `ln T = ln 2`). That is the Gumbel form for the maximum of about `T` roughly exponential
gaps, with coefficient 1. Refuted by a decade whose median `c` leaves `[0.9, 1.1]`, or by a section
above `c = 4`.

The extreme-value form is the known heuristic for maximal gaps and is filed as known. What is new
here is that it is exact on the STACK's slices - a section is not an interval chosen for
convenience, it is `[q^2, q'^2)`, and the record on it is nonetheless the plain extreme value of
its own twin count, with no visible contribution from the cut being a square.

**Two consequences for the step, stated exactly.**

1. The step at link `k` is "the record on section `k+1` is shorter than the section". By S18 the
   record is an extreme value of a count, and an extreme value has no unconditional upper bound;
   the only proved upper bounds on a record available anywhere on the tree are the composite
   machine's full-period record `F` and the free-phase covering record `h_2` at the same gear
   sizes, and both exceed the section from link 2 of every chain on. Q7 is exactly the list of
   links they do settle: three. So Part II's answer to "does the recursion give the step" is no,
   and the reason is not that the recursion is weak but that the object to be bounded is a maximum,
   while everything the recursion supplies is a set property.
2. The margin is enormous and it is not the point. `record/section` falls like `ln^2 q^2 x ln T`
   over `(q'^2 - q^2)/10`, i.e. like `ln^2 q / q^2` up to the prime gap, and the measured fall is
   0.245 -> 0.0784 -> 0.0191 -> 0.0066 by decade. A margin that grows is what a count gives; it is
   not a proof, and it is the same margin the review's face-E reading names.

**One correction to a claim on the tree.** tree_review.md section 2.9 says the composite record of
a section "sits near the TOP of the section", from two examples (base 3 at 0.98 of its section,
base 23 at 0.64). Over all 3,397 sections the record's start is **uniform in the section**: median
position 0.480, mean 0.485, **48.1 %** in the top half (and 48.5 % restricted to `q >= 1000`). The
record is the section's head run at 17 sections, its tail run at 22, and strictly interior at
3,358. The two examples were a coincidence; nothing about the cut pulls the record to either end.

**S19 (THE FIRST TWIN ABOVE THE CUT AND THE LAST BELOW THE END; measured on the whole scan).**
Over the 3,397 sections the first twin above the cut `q^2` lies at most **2,410** numbers above it
(at `q = 8,699`, cut 75,672,601, first twin 75,675,011), with median 172; the last twin lies at
most **2,534** numbers below the section's end (at `q = 15,107`). Both are of the order of the
maximal twin gap at that height and neither is exceptional against it, which is why the head and
tail runs are the record at only 39 of the 3,397 sections. Refuted by a first-twin offset above
the section's own record.
