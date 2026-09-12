# The locator: a fixed column after every square (owner, 2026-09-13: "we just need one location")

Script research/stack/r8/locator.py. Came out of the walk network (walk_network.md): the only
columns the anchor families knew open to every gear were the columns two after a square,
(g^2 + 10, g^2 + 12), at g = 13, 97, 379.

## The construction

Column offset i after the square g^2 is the column (g^2 - 2 + 6i, g^2 + 6i). By class alone:

- open to 5 for every g iff i = 0 or 2 (mod 5), since g^2 is 1 or 4 mod 5;
- open to 7 for every g iff i = 3 or 5 (mod 7), since g^2 is 1, 2 or 4 mod 7;
- both: i = 5, 10, 12, 17 (mod 35), the blind classes; offset 2 is open to 5 always and to 7
  unless g = +-2 mod 7;
- open to g itself for g > 6i.

For any other gear h the column is struck iff g^2 = -(6i - 2) or g^2 = -6i (mod h). So each
gear strikes the offset-i column of at most four residue classes of g mod h (the square roots
of those two values), and a gear for which neither value is a quadratic residue never strikes
that offset from any square: a blind gear for the offset. The condition "the offset-i column
after g^2 is a twin" is therefore a sieve on the gear line g, at most four classes removed per
gear h below sqrt(g^2 + 6i), none for the blind gears, with the classes in closed form.

Blind gears below 200 and mean classes removed per gear (the twin sieve on a column removes 2):

| offset i | blind gears | mean classes per gear |
|---|---|---|
| 2 | 17, 29, 71, 83, 101, 107, 113, 137, 149, 191 | 1.89 |
| 5 | 19, 41, 61, 73, 83, 89, 97, 103, 139, 173, 181 | 2.05 |
| 10 | 7, 11, 13, 41, 43, 71, 73, 89, 97, 103, 131, 149, 163, 193 | 1.64 |
| 12 | 13, 23, 29, 31, 109, 127, 149, 157, 173, 199 | 2.00 |
| 17 | 7, 11, 19, 31, 43, 47, 67, 79, 107, 131, 191, 199 | 1.86 |

## The location, per machine

For machine q the located twin is the offset-i column after g^2 for a gear g with sqrt(q) < g
<= q (so that g^2 lies in the window). Primes q from 11 to 20000 (2259 machines):

| offset i | machines with a located twin in the window | hits g below 20000 | largest ratio of consecutive hits | fewest hits in a window |
|---|---|---|---|---|
| 2 | 2259 of 2259 | 47 | 7.46 (13 to 97) | 1 (q = 11) |
| 5 | 2258 (none at q = 7) | 98 | 3.31 | 1 |
| 10 | 2259 of 2259 | 230 | 1.63 | 2 |
| 12 | 2258 (none at q = 7) | 123 | 2.68 | 1 |
| 17 | 2259 of 2259 | 176 | 1.86 | 1 |

Over the five offsets together no machine from 11 to 20000 lacks a located twin. Hits for
offset 2: g = 7, 13, 97, 127, 211, 223, 379, 601, 1427, 1709, 2213, ...; for offset 10 (the
densest): every window from q = 11 holds at least two.

The location statement in the machine's own shape: the hit gears form a chain under squaring
(each next hit is below the square of the previous one), the same shape as step 8 with the
primes replaced by the gears whose offset-i column is a twin.

## Honest placing

What is closed form: the candidate columns (a fixed offset after every square), the exclusion
of 5, 7 and g by class, each other gear's strike set as the square roots of two fixed numbers,
and the blind gears. What is not: that some gear between sqrt(q) and q has the column open to
every remaining gear. That is a sieve on the gear line with about two classes removed per
gear, the same density as the twin sieve on a column, so the location buys structure (a
formula for where to look, gears that cannot interfere) and not density. Prior-art line: twins
of the form (g^2 + a, g^2 + a + 2) with g prime are primes in quadratic polynomials
(Bunyakovsky, Hardy-Littlewood conjecture F); no case of a quadratic polynomial taking prime
values infinitely often is proved. The locator turns "a twin in the window" into "a hit gear
below q above sqrt(q)", and the hit gears' own chain under squaring is the same statement one
level down.

## What is new

The offset construction and the blind-gear law are exact and were not on the tree: the column
at a fixed offset after any square is struck by gear h only from the square-root classes of
-(6i - 2) and -6i, and gears with neither a residue never strike it. This is the square origin
made into a locator: one formula, one candidate per square, and a list of gears that cannot
touch it.

## Steps, and the independence test (owner, 2026-09-13)

Steps of the locator = gears tried from just above sqrt(q) upward until the candidate column
after the square is a twin (2258 machines, q = 11 .. 20000): offset 2: min 1, median 5, mean
6.9, max 19 (q = 173); offset 10: median 2, max 6; offset 17: median 2, max 8; any of the five
offsets: median 1, max 2. Each step certifies the candidate against the gears below its square
root that can act (at q = 1009, offset 10: 5 of the 10 gears; offset 2: 16 of 23).

Independence test. Predict the hit count in (sqrt q, q] from the closed-form root classes
alone: sum over gears g in the window of the product over gears h < g of (1 - roots_h /
(h - 1)), roots_h the nonzero square roots of -(6i - 2) and -6i mod h (class 0 is already
excluded because g is prime). Against the actual count:

| q | offset 2: expected / actual / ratio | offset 10: expected / actual / ratio |
|---|---|---|
| 101 | 2.6 / 2 / 0.76 | 13.4 / 12 / 0.90 |
| 307 | 4.0 / 4 / 1.01 | 22.1 / 18 / 0.81 |
| 1009 | 7.5 / 6 / 0.80 | 41.7 / 37 / 0.89 |
| 3001 | 14.4 / 10 / 0.69 | 79.3 / 63 / 0.79 |
| 10007 | 30.7 / 21 / 0.68 | 168.0 / 130 / 0.77 |
| 19997 | 48.1 / 43 / 0.89 | 263.0 / 214 / 0.81 |

The ratio holds near 0.8 with no drift: the gears act on the gear line independently, at their
closed-form shares, and nothing in the counts points at a structure beyond the classes. The
constant below 1 is the usual truncation effect of a product over all gears below g (the
Mertens constant), not a conspiracy.

Reading of the locator in the machine's shape: for g > 3i the candidate column after g^2 lies
below the next square, so it is a twin iff it is open to the gears up to g, i.e. it is an
opening of machine g at a fixed offset in machine g's own top layer. The hit gears are the
survivors of a sieve on the prime line (avoid the root classes mod every smaller gear), as the
primes are the survivors of a sieve on the number line (avoid class 0): the same construction
one level up, with up to four classes per gear instead of one. "A hit gear in (sqrt q, q]" is
the Bertrand-shaped statement for that second sieve.

## In the fields (owner: look in the fields; go ahead with the work; 2026-09-13)

The locator as a field of the owner's construction: docs/locator_field.html (rows gears h,
columns the natural numbers m, row h painted where h strikes the candidate after m^2; the
purple box (sqrt q, q]) and, inside the explorer and the twin, the field kind `locator:i`
(rows gears, columns the natural numbers n, painted at the members of the candidate columns
the gear divides; the unpainted candidates are the located twins). Probes in
research/stack/r8/locator_field.py and blind_offsets.py.

Read off the field:

- Every row is periodic in m with period h and mirror-symmetric about every multiple of h
  (teeth in pairs r, h - r): this machine has a mirror at every gear's multiples.
- Teeth per gear (gears to 3000, offset 10): none 24.3%, two 52.6%, four 22.7%, mean 1.96;
  the same at every offset tried. The twin machine has exactly two per gear. Same load,
  redistributed: a quarter of the gears absent, a quarter doubled.
- The white columns are the fold one level up: only m coprime to 6 carries a twin slot after
  its square.
- Blind gears across offsets, exact: as i runs mod h, gear h is blind for
  (h - 1)/4 offset classes when h = 1 mod 8, (h - 3)/4 when h = 3 or 7 mod 8, (h - 5)/4 when
  h = 5 mod 8 (the count of non-residue pairs at distance 2; checked to 2100, e.g. 13: 2,
  17: 4, 101: 24, 401: 100, 1009: 252). A quarter of the offsets per gear, a quarter of the
  gears per offset. No gear is blind on a whole class i = c mod 35 (checked for c = 5, 10,
  12, 17 to 2100): the blind set of an offset is a fixed quarter of the gears, but which
  quarter changes with the offset.

The two sieves side by side (offset 10): survivors among their own candidates

| q | primes in (sqrt q, q] | locator survivors | columns in (q, q^2] | twin survivors |
|---|---|---|---|---|
| 101 | 22 | 12 (0.545) | 1683 | 201 (0.119) |
| 307 | 56 | 18 (0.321) | 15657 | 1144 (0.073) |
| 1009 | 158 | 37 (0.234) | 169512 | 8278 (0.049) |
| 3001 | 415 | 63 (0.152) | 1500500 | 53804 (0.036) |

So the locator machine is the twin machine's own shape on the gear line: same fold, same
mean teeth, mirrors at all multiples instead of one, a quarter of its gears missing at every
offset, with the missing quarter given exactly by h mod 8 and the offset.

## The start of the chain (owner: what do we know about the start pair; 2026-09-13)

Twins the locator rules prove with no checking: the candidate after g^2 is a twin by
construction when every gear below its square root is excluded by class (5, 7), blind for the
offset, or has g outside its teeth, and g itself does not divide 6i - 2 or 6i.

| offset i | self-striking g (g divides 6i - 2 or 6i) | twins proved by the rules alone, g <= 60 | first candidate a gear actually strikes |
|---|---|---|---|
| 2 | 5 | (59, 61) g = 7; (179, 181) g = 13 | g = 11: 7 strikes 133 |
| 5 | 7 | (149, 151) g = 11; (197, 199) g = 13; (1877, 1879) g = 43; (2237, 2239) g = 47 | g = 17: 11 strikes 319 |
| 10 | 5, 29 | (107, 109), (179, 181), (227, 229), (347, 349), (419, 421) for g = 7, 11, 13, 17, 19; (1019, 1021) g = 31; (1427, 1429) g = 37; (2267, 2269) g = 47; (3539, 3541) g = 59 | g = 23: 19 strikes 589 |
| 12 | 5 | (191, 193) g = 11; (239, 241) g = 13; (431, 433) g = 19; (599, 601) g = 23; (1031, 1033) g = 31 | g = 7: 11 strikes 121 |
| 17 | 5 | (149, 151) g = 7; (269, 271) g = 13; (389, 391) g = 17; (461, 463) g = 19; (1061, 1063) g = 31; (1949, 1951) g = 43; (2309, 2311) g = 47; (3581, 3583) g = 59 | g = 11: 13 strikes 221 |

Offset 10 is the richest start because 7, 11 and 13 are blind for it: for g up to 19 no gear
below the square root can act at all, so the first five candidates are twins by the rules,
and the first strike anywhere is 19 on the candidate after 23^2. These are the machines
1..19 handled by construction; from 23 on the surviving g are the survivors of the second
sieve.

The start pair in the walk sense, (5, 7): the first column of the machine's own line, struck
by 5 and 7 themselves, open to every other gear; the anchor every killed column carries its
killer 5 or 7 onto; residue classes 5 and 7 mod every larger gear, covering two classes per
gear; at offset 2 its right member 7 is the first hit gear (59, 61), at offset 10 the pair
(5, 7) has 5 self-striking (83, 85) and 7 a rule-proved twin (107, 109).

## Rule walks: a deterministic walk with the sub-machine as the rule (2026-09-13)

Owner: a walk proven needs stepwise rules that pick the next anchor, no search, no
pre-checking of landings, certification afterwards. Framework research/stack/r8/rule_walk.py
(rules are small functions returning the next axis; the landing is certified after the walk).
Machines 11 to 3000 (426), then to 20000 (2258).

Rules tried, machines succeeded of 426 to 3000: ladder to the first gear above sqrt q at
offset 2 / 10 / 17: 15 / 203 / 129; top gear's square: 71; square axes g^2 - 1 in order: 8;
Tower-of-Hanoi over the gear periods: 78; blind hops at offset 5 / 10: 94 / 186; ladder
choosing the first gear g above sqrt q whose classes avoid the teeth of the gears up to B, at
offset 10: B = 13: 203, B = 31: 382, B = 101: 426 of 426.

Failures of the residue-blind rules are exactly the teeth of named gears, in blocks (every
machine sharing its first gear above sqrt q shares its fate; all machines between 23^2 and 29^2
fail at offset 10 because 29 divides 58). Walks that carry knowledge (square axes, Hanoi) do
worse than one flip: 5 and 7 strike their landings at full rate.

THE RULE THAT WORKS, machines 11 to 20000: consult only the gears up to sqrt q (the
sub-machine). Take the first gear g above sqrt q whose classes mod the sub-machine's gears
avoid their teeth for offset 10; flip from (5, 7) onto the column (g^2 + 58, g^2 + 60). One
flip. Succeeds on 2253 of 2258 machines; the 5 failures are q = 11 .. 23, where g = 5 and 5
divides its own candidate (5 divides 60). Cutoffs 2 sqrt q and q give the same result (2253,
2256). Between sqrt q and the chosen g there were 2546 gears the rule never consulted, over
1211 machines; none of them struck the landing.

Why the unconsulted gears cannot strike: the strike law in residue form. For any gear h below
g, with r = g mod h, gear h strikes the offset-i candidate after g^2 iff h divides r^2 + 6i - 2
or r^2 + 6i (g^2 = r^2 mod h); checked exhaustively, 45,150 cases on g to 2000, 0 violations.
This is the caustic law (e = s h - r^2) read at a fixed offset. For a gear h just below g the
residue is the gap d = g - h, so h can strike only if h divides d^2 + 58 or d^2 + 60, which
needs h <= d^2 + 60; the gaps in range are at most 20, so the numbers to divide are at most
460, and no unconsulted gear divided one (1208 of the 2546 were small enough to be allowed by
the bound; 0 divided).

What this changes. The locator for machine q is decided by the machine of size sqrt q: its
teeth on the prime line just above sqrt q pick g, and the gears between sqrt q and g are
harmless by the gap law. Termination of the walk, for all q, is now two statements about the
sub-machine and the gap: (i) a prime g above sqrt q avoiding the sub-machine's teeth exists
within a gap d of sqrt q with d^2 + 60 below the smallest unconsulted gear (in range d <= 20);
(ii) no unconsulted gear divides d^2 + 58 or d^2 + 60 (in range, never). Both are statements
on the prime line near sqrt q, not on the window.

## The direct construction: squares and roots, no search (owner's guess, 2026-09-13)

Owner: anchor decisions tied to squares and roots, plus a start rule, should navigate straight
to an open twin. Built as: g = the first prime above sqrt q (the first square in the window);
i = the smallest offset the strike law allows against the gears below g, i.e. for every h < g
with r = g mod h neither r^2 + 6i - 2 nor r^2 + 6i is divisible by h, and g does not divide
6i; land on (g^2 + 6i - 2, g^2 + 6i). No search over g, no check of the landing; the offset is
read off the residues (the roots) of g.

Result, machines 11 to 20000: 2258 of 2258, offset at most 27 columns, mean 8.3. The start
pair does not enter the location (a one-flip walk from any start lands on the same column and
differs only in what it carries).

Why it works, exact. The gears below g are exactly the gears up to sqrt q (g is the first
prime above sqrt q), so "the sub-machine" and "every gear below g" are the same set. A number
in (g^2, g g'), g' the next prime, with no prime factor below g is prime (a composite there
would need two factors at least g, hence be at least g g'). So every column in (g^2, g g')
that the gears below g miss, and that g itself misses, is a twin: checked exhaustively, 8194
such columns for g to 1500, 0 exceptions. The rule's offset stayed inside that zone at every
machine (largest ratio to the zone 0.82, at q = 53; mean 0.20), so no landing relied on luck.

Termination for all q, in one line: in the zone (g^2, g g') after the first square above q,
the wheel of the gears below g leaves a column open. The zone has g (g' - g) / 6 columns, at
least g / 3. The teeth of gear h in the zone, in offset coordinates, are the two classes of i
with r_h^2 + 6i = 0 or 2 (mod h), r_h = g mod h: a tooth family fixed by the roots of g. Gears
with r_h^2 + 6i < h for every i in the zone cannot strike at all (the gears just below g).

## The walk on real mirror axes (owner's correction, 2026-09-13)

Owner: an axis must be an actual mirror of a gear combination, a multiple of M = the product of
the chosen gears (2, 3 mirror at 6; 2, 3, 5 at 30; 2, 3, 11 at 66); no offsets; the mirror is a
property of the combined gears, not a word for flipping. Built as such:
research/stack/r8/true_mirror_walk.py.

- Axis k M, flip n -> 2 k M - n - 2 (columns to columns since 6 divides M). The pattern of the
  gears in S repeats with period M and is symmetric about 0, hence about every multiple of M,
  so the flip carries the openness of every gear in S.
- Start at home (-1, 1), open to every gear. One flip about k M lands on (2 k M - 1, 2 k M + 1),
  open to every gear of S for every k.
- The remaining gears h (h <= q, not in S) strike that landing iff k = -(2M)^-1 or +(2M)^-1
  (mod h): two classes of k per gear, read off the roots. Rule: the smallest k whose landing
  lies in the window (q, q^2] and whose class avoids those teeth for every remaining gear.
- Exact: if such a k exists the landing is a twin (both members below q^2, no gear up to q
  divides them; the S gears by the mirror, the rest by the choice of k).

| S | M | machines with a landing (11 .. 5000) | k, mean / largest | landing position 2kM/q^2, mean / max | landings not twins |
|---|---|---|---|---|---|
| 2, 3 | 6 | 665 of 665 | 200 / 425 | 0.0037 / 0.50 | 0 |
| 2, 3, 5 | 30 | 665 of 665 | 42 / 85 | 0.0040 / 0.50 | 0 |
| 2, 3, 5, 7 | 210 | 661 of 665 (none at 11, 13, 17, 19) | 9 / 14 | 0.0080 / 0.79 | 0 |
| 2, 3, 5, 7, 11 | 2310 | 645 of 665 (none below 41) | 2 / 2 | 0.027 / 0.98 | 0 |

To 20000 with S = {2, 3} and {2, 3, 5}: every machine has a landing (see the run line in the
log). The landing sits just above q (a fraction of a percent into the window on average); the
bigger S, the fewer multiples fit below q^2 and the small machines lose their landing.

What this is: the mirror walk as the owner described it, one flip from home about a true
mirror axis of the small gears, the multiple chosen by the roots of the remaining gears; the
landing is a twin whenever the multiple exists, by the mirror for S and by the choice of k for
the rest. Termination = the existence of k: a class of k mod every remaining gear that avoids
two teeth, with 2 k M inside the window. That is the twin sieve on the multiples of M, the
S gears removed from it by the mirror.

## Multi-step walks on real mirror axes (owner: do it; 2026-09-13)

Script research/stack/r8/multi_mirror_walk.py. Flips about real axes a_1, a_2, ... (each a
multiple of the product of a gear set containing 2, 3).

- Composition law, exact (3000 random walks of 1 to 5 flips, 0 violations): the walk from n
  ends at 2A - n - 2 after an odd number of flips and at n + 2A after an even number, where A
  is the alternating sum of the axes (last axis positive). A slide by 2A and a reflection
  about A both carry exactly the gears dividing A.
- So the certification of the end of a walk is the set of gears dividing A, whatever the
  intermediate landings were. Stepwise tracking (intersecting the carried gears flip by flip)
  undercounts it: in 2062 of 3000 walks the end was certified for gears no single flip
  carried (two flips about 30 k_1 and 42 k_2 with 42 k_2 - 30 k_1 = 66 carry 11, though
  neither axis does).
- From home (-1, 1) every walk therefore ends on the column (2A - 1, 2A + 1) with A a
  multiple of 6, certified for the gears dividing A. Multi-step walks reach exactly the
  landings one flip reaches, with A now any multiple of 6 (a difference of two real axes),
  so the gears the mirrors carry are the divisors of A, not a fixed set.
- Two-step rule: A = the smallest multiple of 6 with the landing in the window whose class
  avoids the teeth of every gear not dividing A (teeth: 2A = -+1 mod h), realised as home ->
  flip about 30 k_1 -> flip about 42 k_2 with 42 k_2 - 30 k_1 = A. Machines 11 to 5000: 665
  of 665 landings are twins (exact by the same lemma: below q^2, no gear up to q divides
  them). Gears carried by the mirrors per machine: mean 1.25; handled by the roots: mean
  334. The landing sits just above q.

Reading. The mirrors decide the family of landings (columns straddling a multiple of 12) and
carry the divisors of A; the roots decide which member of the family; the lemma makes the
landing a twin whenever the roots find a member inside the window. Termination for all q is
one statement: among the multiples of 6 with 2A in (q, q^2], some A avoids two classes mod
every gear up to q not dividing A. The multi-step form changes the route and the carried set,
not the landing family.

## Carrying many gears, and the termination statement on the m-line (owner: both; 2026-09-13)

Scripts research/stack/r8/carry_many.py and mline_records.py.

Carrying many gears. Effective axis A = k P_m, P_m the product of the first m gears; landing
(2 k P_m - 1, 2 k P_m + 1) certified for those m gears by the mirror; the roots must clear the
rest. The largest m with a landing in the window, machines 11 to 5000:

| q | gears in the machine | largest m carried | multiples of 2 P_m below q^2 | k | gears left to the roots | landing / q^2 |
|---|---|---|---|---|---|---|
| 31 | 9 | 4 | 2 | 1 | 7 | 0.44 |
| 101 | 24 | 5 | 2 | 2 | 21 | 0.91 |
| 401 | 77 | 5 | 34 | 2 | 74 | 0.06 |
| 1009 | 167 | 6 | 16 | 3 | 163 | 0.18 |
| 3001 | 429 | 7 | 8 | 4 | 424 | 0.45 |
| 4999 | 667 | 7 | 24 | 4 | 662 | 0.16 |

Every largest-m landing is a twin (0 failures). The mirrors can carry only the gears whose
primorial stays below q^2 / 2: m grows like the number of primes up to about 2 ln q, seven
gears by q = 3000, against hundreds in the machine. The roots do nearly all the work at every
size; carrying more gears buys nothing but a landing higher in the window and fewer multiples
to choose from.

The termination statement on the m-line. From home the landing family is (12 m - 1, 12 m + 1);
gear h strikes m iff m = -+12^-1 (mod h): two teeth per gear, symmetric about the multiples of
h. The m-line field: rows the gears, columns m. Termination for q: an m with q < 12 m - 1,
12 m + 1 <= q^2 unpainted in every row h <= q. Exact on the m-line to q^2 / 12:

| q | open m in the window | first landing (12 m - 1) | record R(q): longest struck run of m below q^2/12 | window length in m | 12 R / q^2 |
|---|---|---|---|---|---|
| 11 | 3 | 59 | 4 | 9 | 0.40 |
| 31 | 14 | 59 | 13 | 77 | 0.16 |
| 101 | 100 | 107 | 43 | 841 | 0.051 |
| 401 | 906 | 419 | 80 | 13366 | 0.0060 |
| 1009 | 4179 | 1019 | 191 | 84756 | 0.0023 |
| 3001 | 26960 | 3119 | 278 | 750250 | 0.0004 |

The walk terminates for q as long as R(q) is below the window length; the record's share of
the window falls from 0.40 at q = 11 to 0.0004 at q = 3001, and the count of open m in the
window grows like q^2 / ln^2 q. This is step 8 in run form on the m-line (R(q) below the
window), the twin machine with teeth -+12^-1 instead of -+6^-1: the same object, half the
columns, the same margin.
