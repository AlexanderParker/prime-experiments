# The fields in a machine's window over many cycles (2026-09-12)

Owner's request: run the layers again but look at what fields kill in the window over many
cycles, note how the fields appear, then which rows are the killers in those fields; summarise
which fields never kill, which are periodic, which become periodic, which never; list the gears
that killed under each field per machine layer; check the window's twins at the window's mirror;
and find which fields mirror.

Scripts: research/stack/r8/window_fields.py (machines 5 to 31, 12 cycles each; full output in
research/stack/r8/results_window_fields.txt), mirror_fields.py (results_mirror_fields.txt),
cycle_survival.py. Definitions: machine q, gears the primes up to q, window (q, q^2], cycle q#
(the product of the primes up to q), cycle c = the window shifted by (c - 1) q#, layer =
window(q) minus window(p) = (p^2, q^2], mirror of n = q# - n. A kill = a composite twin-slot
member (n = 1 or 5 mod 6). Fields as in the explorer: multiples, squares, products:j, higher:g
(kills whose smallest gear is g), higher1:g (g once), lower:g (largest gear g), lower1:g.

Two bugs corrected before these results: sympy's primorial(q) is the product of the first q
primes, not of the primes up to q (the cycle lengths of a first run were wrong); and a field
that simply stops killing was being read as "becomes periodic".

## 1. Which fields kill in the window, cycle by cycle

Cycle 1 (the window itself, q < n <= q^2): every kill has a gear factor at most q, so the
killing fields are exactly higher:g for g = 5 .. q (each one kills, since g^2 is in the window),
higher1:g for g < q, squares of the gears 5 .. q, products:j for j up to log_5 (q^2), and lower:g
for the largest factors that occur (g up to q^2 / 5). No higher:g with g > q, no square of a
gear above q, and no product count beyond log_5 (q^2) can kill in cycle 1, at any size.

Cycles 2 and on: the machine's rows repeat exactly (the offsets of every kill by a gear up to q
are the same in every cycle), so the only change from cycle to cycle is on the machine's open
columns, which in cycle 1 are exactly the window's twins. Each of them is, in cycle c, a twin
again or a kill of one field higher:g with g > q, and no other field can touch it. So the fields
that APPEAR after cycle 1 are exactly the higher:g fields of the gears above q, one per opening
they eat, and they appear sporadically (a given higher:g with g > q kills in some cycles, not in
others; none of them is periodic and most kill in one or two of the twelve cycles).

The machine's openings across cycles (cycle_survival.py; the eaters are the smallest gear of
the member that fell):

| machine | openings | twins again in cycles 1 .. 12 | eaters (gear: count) |
|---|---|---|---|
| 5 | 2 | 2, 1, 1, 2, 1, 0, 2, 1, 0, 1, 1, 1 | 7: 7, 11: 3, 13: 2, 17: 1 |
| 7 | 4 | 4, 2, 2, 2, 2, 2, 3, 2, 2, 1, 3, 1 | 11: 8, 13: 6, 17: 3, 19: 5, 23: 4 |
| 11 | 7 | 7, 2, 3, 2, 2, 0, 3, 2, 2, 2, 1, 1 | 13: 13, 17: 10, 19: 3, 23: 5, 29: 5, 31: 7 |
| 13 | 9 | 9, 2, 3, 1, 0, 3, 2, 2, 2, 0, 1, 1 | 17: 12, 19: 12, 23: 9, 29: 3, 31: 6 |
| 17 | 15 | 15, 5, 6, 2, 0, 4, 2, 1, 2, 4, 1, 2 | 19: 19, 23: 16, 29: 9, 31: 10, 37: 9 |
| 19 | 17 | 17, 3, 3, 2, 4, 0, 2, 4, 2, 2, 4, 2 | 23: 15, 29: 11, 31: 13, 37: 11, 41: 7 |
| 23 | 21 | 21, 0, 1, 2, 3, 0, 2, 4, 2, 3, 1, 1 | 29: 15, 31: 13, 37: 13, 41: 8, 43: 11 |
| 29 | 28 | 28, 5, 4, 1, 3, 1, 2, 1, 4, 5, 2, 2 | 31: 25, 37: 22, 41: 15, 43: 14, 47: 14 |
| 31 | 30 | 30, 2, 0, 0, 1, 3, 3, 2, 1, 2, 3, 0 | 37: 20, 41: 17, 43: 19, 47: 9, 53: 15 |

The pattern in how the fields appear: cycle 1 is the only cycle where the window is closed by
the machine alone; from cycle 2 the next gears (q', q'', ...) take the openings at their own
rates (the first eater is always the next prime, with the largest count), and the count of
openings that are twins again in cycle c is the count of the window's twins that the gears
above q miss at the offset (c - 1) q#: zero in some cycles (machine 23 cycle 2, machine 31
cycles 3, 4, 12).

## 2. Which rows kill, in which fields

Read from the layer (p^2, q^2] in cycle 1 (the full lists for every field and machine are in
results_window_fields.txt under "layer ... gears that killed per field"):

- higher:g: the rows are g, the primes in (p^2 / g, q^2 / g] (the cofactors of g's strikes in
  the layer), and the factors of the composite cofactors. Examples: layer (169, 289]: higher:5
  rows 5, 7, 11, 37, 41, 43, 47, 53; higher:7 rows 7, 29, 31, 37, 41; higher:11 rows 11, 17, 19,
  23; higher:13 rows 13, 17, 19; higher:17 row 17 only (its square). Layer (841, 961]: higher:5
  rows 5, 7, 11, 13, 17, 37, 173, 179, 181, 191; higher:29 rows 29, 31; higher:31 row 31.
- The newest gear q kills in its own layer only at q^2 in the layer (p^2, q^2] (its next strike
  q q' lies above q^2), so its row set there is {q}: every layer to 31.
- squares: row g at g^2 for the gears in the layer, one per layer (the newest gear).
- multiples and products:j: every prime factor of every kill; the same row sets.
- lower:g and lower1:g: the rows are the factors below g of the numbers whose largest factor is
  g; one field per largest factor, most with a single kill in the layer.

Over the cycles, the killer rows of a periodic field (higher:g, g <= q) grow with the cycles
(new cofactors at each shift), the row g itself constant.

## 3. Summary: never kill, periodic, becomes periodic, never periodic

Over machines 5 to 31 and 12 cycles (the verdicts by field kind, results_window_fields.txt):

| field kind | verdict | reason |
|---|---|---|
| higher:g, g <= q | PERIODIC in every machine | "smallest gear is g" depends on n mod g#, and g# divides q# |
| higher:g, g > q | never kills in cycle 1 (exact, all sizes); after that sporadic, never periodic, most kill in one or two cycles of twelve | its kills are g m with g the smallest factor, so above g^2 > q^2; where it lands is decided by primes above q |
| higher1:g, g <= q | not periodic (12 of the verdicts "becomes periodic" are the last three cycles agreeing by chance: g^2 not dividing) | "g exactly once" depends on n mod g^2, and g^2 does not divide q# |
| lower:g, lower1:g, any g | never periodic; each stops killing after a few cycles | "largest gear is g" is not a residue property of any period |
| multiples (all rows) | never periodic; restricted to the machine's rows it is the wheel, periodic by construction | rows above q |
| products:j | never periodic (j = 2 kills in every cycle; j = 4, 5 kill in a few cycles and stop) | the factor count is not periodic |
| squares | kill only in cycle 1 for every machine from 7 on (machine 5's cycle is 30, so later squares land in its window: 49, 169, 289) | a square lands in a shifted window only by accident |

No field kind never kills at any scale. What never kills at any scale is: in cycle 1, higher:g
and higher1:g for g > q, the squares of gears above q, and products:j for j > log_5 (q^2). No
field genuinely becomes periodic: every "becomes periodic" verdict in the raw output is the
last three of twelve cycles agreeing while the field has stopped or paused.

## 4. The window's twins at the window's mirror

Mirror about q#/2: the column (n, n+2) goes to (q# - n - 2, q# - n). The machine's rows are
symmetric under it (g divides n iff g divides q# - n, for every gear up to q), so every
window twin's mirror is open to the machine; it is a twin again iff no gear above q strikes it.
Reading the window's twins from the top down and checking the mirror:

| machine | window twins | twins again at the mirror | the pairs (window left member, mirror left member) |
|---|---|---|---|
| 5 | 2 | 2 | (17, 11), (11, 17): the two window twins are each other's mirror in the cycle 30 |
| 7 | 4 | 3 | (29, 179), (17, 191), (11, 197); 41 mirrors to 167, 169 = 13^2 |
| 11 | 7 | 2 | (71, 2237), (41, 2267) |
| 13 | 9 | 2 | (149, 29879), (17, 30011) |
| 17 | 15 | 2 | (107, 510401), (59, 510449) |
| 19 | 17 | 3 | (239, 9699449), (179, 9699509), (41, 9699647) |
| 23 | 21 | 1 | (197, 223092671) |
| 29 | 28 | 4 | (827, ...401), (809, ...419), (419, ...809), (149, ...079) |
| 31 | 30 | 1 | (269, 200560489859) |

The share falls with q (from all of them at 5 and 7 to one in thirty at 31) as the mirror sits
near q# where the gears above q strike at their full rate; the mirror is exact for the machine
and carries nothing about the primes above it.

## 5. Which fields mirror

Kills in the window (cycle 1) whose mirror q# - n is a kill of the same field (mirror_fields.py,
machines 5 to 31):

| field kind | mirrors | note |
|---|---|---|
| higher:g, g <= q | 494 of 495 (exact) | the one miss is machine 5, 25 mirrors to 5, the gear itself |
| multiples, machine rows | 494 of 495 (exact) | the same miss |
| higher1:g, g <= q | 331 of 405 = 0.82 | breaks when g^2 divides the mirror image, about 1/g of the time |
| products:j | 127 of 495 = 0.26 | the factor count of the mirror is unrelated; chance |
| squares | 0 of 43 | q# - g^2 is never a square in range |
| lower:g, lower1:g | 1 of 232, 0 of 263 | the largest factor of the mirror is unrelated |

Every window kill's mirror is a kill (495 of 495 composite): the mirror keeps the small gear.
The fields that mirror are exactly the fields decided by the machine's own rows (higher:g for
g up to q and the machine's multiples); no field built on a factor count, a largest factor or a
square mirrors.

## What is new and what is not

New as a reading: the window across cycles is the machine's fixed openings being eaten only by
higher:g fields of the next gears, one field per opening eaten, and no other field ever acts on
an opening; the first eater is always q' with the largest count. The mirror separates the
fields cleanly into the machine's (exact mirror) and the rest (none). Not new: the periodicity
verdicts are the residue facts (g# divides q#; g^2 does not); the mirror survival rate falls as
the primes above q thin the mirror, an ordinary density.

## 6. The order ceiling: which composites can kill in the window (owner's question, 2026-09-12)

Order j = the number of prime factors with multiplicity (the explorer's "products of j gears").
Every gear in a twin slot is at least 5 and 5^j is itself a twin-slot composite (5^j alternates
between 5 and 1 mod 6), so products:j kills in the window (q, q^2] iff 5^j <= q^2:

    j_max(window of q) = floor(log_5 q^2) = floor(2 log_5 q)   (exact; equal to the measured
    largest killing order at every machine checked: q = 5 .. 59, 101, 211, 401)

The layer (p^2, q^2] has the same ceiling except when 5^j already sits below p^2 (q = 401:
window 7, layer 6). The whole cycle (up to q#) has ceiling floor(log_5 q#), about 0.62 q: the
order available in the cycle grows linearly in q, the order available in the window only like
1.24 ln q. The window's order steps up by one each time q crosses 5^(k/2): 5, 13, 29, 59, 128,
280, 626, ...

Where the kills live inside the ceiling (window kills by order; a cofactor "above q" is a prime
of the window itself):

| q | window kills | order 2 | of which gear x gear | gear x prime above q | order 3 | orders >= 4 |
|---|---|---|---|---|---|---|
| 31 | 159 | 131 (0.82) | 44 | 87 | 26 (0.16) | 2 (0.01) |
| 101 | 2141 | 1509 (0.70) | 291 | 1218 | 540 (0.25) | 92 (0.04) |
| 211 | 10190 | 6590 (0.65) | 1012 | 5578 | 2912 (0.29) | 688 (0.07) |

What the ceiling gives and what it does not. It fixes the exact list of fields that exist in
the window (products:2 .. products:floor(2 log_5 q); higher:g for g <= q; squares of the gears
up to q), an upper edge of the area the proof has to cover, and that edge is narrow and grows
only logarithmically. Inside it, orders 4 and up are the smooth field (closed form, a few
percent of the kills), and orders 2 and 3 carry 90 to 99 percent of them. The order-2 kills are
mostly a gear times a prime ABOVE q, i.e. a gear times a prime of the window itself (87 of 131 at
q = 31, 5578 of 6590 at q = 211): the window's own primes, handed up, are what closes most of
its columns. So the narrowness is real in order and in field list, and the difficulty sits
entirely in the two lowest orders, where the cofactors are the primes the statement is about.
