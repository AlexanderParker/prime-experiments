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
