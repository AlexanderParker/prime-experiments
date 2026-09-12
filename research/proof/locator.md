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
