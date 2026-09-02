# The lower sieve and the new gear (the human's construction)

Manager, round 29 (2026-09-02). Pre-registration with scorecard:
research/data/r29/lower_sieve_prereg.md; script research/lower_sieve_r29.py; log
research/data/r29/lower_sieve.log (666 rungs, q_next <= 5000, 5/5 gates after one correction).

## 1. The construction in the repository's coordinates

A machine is two parts: the LOWER SIEVE - the gears found so far, a periodic word of period
P = product of the gears - and the NEW GEAR q. The lower sieve repeats up to q^2 and finds the
primes there; then gear q joins it, and the next machine's window runs to q_next^2. Two
questions:

  Q1  when gear q joins the lower sieve, what is its contribution inside the new section
      (q^2, q_next^2), and can that contribution be a permanent blocking condition;
  Q2  what does the new gear change in the lower sieve itself - can a new gear block the
      lower sieve's twin spots.

Section of the rung q -> q_next: slots q^2 < 6k+1 < q_next^2, numbers 6k_lo-1 .. 6k_hi+1.
The lower sieve's twin spots there are the slots with neither number divisible by a gear
below q.

## 2. Q1 - gear q's contribution to the new section

Forced and checked at every rung (gates L1, L2):

  - The lower sieve's twin spots in the section are exactly the new twins plus the slots
    whose death rung is q. So gear q's whole contribution is the set of lower-sieve twin
    spots it bites, and it is the ONLY gear that can bite one there: every other gear is
    already in the lower sieve, and gear q_next kills nothing below q_next^2.
  - Gear q's class +-u_q meets the section at the numbers q x m, m = +-1 mod 6 in
    (q, q_next^2/q), i.e. m = q_next, q_next + 2, ... - one to six slots, at least (q-1)/3
    slots apart. Most are masked: by a smaller gear on the same side when m is composite
    (47 x 49, 997 x 1001), or on the other side (2491 = 47 x 53 sits beside 2489 = 19 x 131).
    A bite - a lower-sieve twin spot actually removed by gear q - is q x prime beside a prime.

Listings from the log. 5 -> 7: spots k = 5 TWIN, k = 6 BITTEN (35 = 5 x 7 | 37), k = 7 TWIN.
7 -> 11: six spots; gear 7 bites k = 13 (77 = 7 x 11 | 79) and k = 15 (89 | 91 = 7 x 13);
twins at k = 10, 12, 17, 18. 11 -> 13: two spots, both twins; gear 11's only product
143 = 11 x 13 is masked (145 = 5 x 29). 13 -> 17: eight spots, one bitten (221 = 13 x 17 | 223),
245 | 247 = 13 x 19 masked. 47 -> 53: thirteen spots, all twins; gear 47's four products
47 x 49, 47 x 53, 47 x 55, 47 x 59 all masked. 997 -> 1009: 4011 slots, 193 spots, two bitten
(1005973 = 997 x 1009 beside the prime 1005971; 1009961 = 997 x 1013 beside the prime
1009963), 191 new twins; 997 x 1001, 997 x 1003, 997 x 1007 masked.

Over the 666 rungs (L4): gear q bites 0 spots at 502 rungs, 1 at 146, 2 at 15, 3 at 3, never
more; the bitten fraction of the section's spots is below 0.047 at every q >= 300. And (L5)
at every rung q >= 300 there is a new twin between any two consecutive kills of gear q.

Answer to Q1. Gear q's contribution to the new section is at most three isolated near-twins,
spaced at least (q-1)/3 slots apart, with new twins between them: it can only shorten the
lower sieve's list by a few isolated entries, it cannot join two of the lower sieve's gaps
into a block, because its own kills never even close the gap between two of themselves. A
permanent blocking condition inside a window cannot come from the new gear. It could only
come from the lower sieve's longest run of blocked slots - the record gap F, a period-scale
object - reaching the window length W. That is the F/W question of the record law
(word-tree.md section 3, killer-spec.md 3.1: F/W measured near 1/4 and not rising). In this
construction the danger is never the new part; it is whether the old part's worst stretch
ever catches the window, and every measurement says the window outruns it (W ~ q^2 while
the gaps inside the window are ~ ln^2 q).

## 3. Q2 - what the new gear changes in the lower sieve

Forced (L3, printed for q = 7, 11, 13): adding gear q lifts the period-P word to q copies and
deletes the class +-u_q. Every opening of the old word is deleted in exactly 2 of its q lifts
- never in all of them, never in none - so no single new gear can block a twin spot of the
lower sieve; it thins every spot by the same factor (q-2)/q. Inside a window, the lifts of
one old opening that land there number at most one once P > W (q >= 11), and the new gear
blocks that lift iff the lift's number is q x m - the near-twins of section 2. Which old
openings lose which lifts is CRT: the listing shows, for the lower sieve {5, 7}, each of its
15 openings a mod 35 losing the two lifts k = a + 35j with k = +-2 mod 11.

Answer to Q2. A new gear never blocks a twin spot of the lower sieve; it removes two of its
q copies. Over all later gears the copies of one spot thin like prod (1 - 2/g) - to density
zero, never to zero. Which copies survive in a given window is the residue-vector question
of the provenance pass (uniform over tooth-avoiding vectors, word-tree.md section 8).

## 4. What the construction adds

It separates cleanly what earlier passes mixed: the new gear's own action in the window is
tiny and fully listed (at most three near-twins), and everything that could threaten twins
lives in the lower sieve's period-scale behaviour, i.e. the record law and (D). The
section-view rule - every check on the new part only - is exactly this construction's
reading of the machine.

Correction on record: the pre-registration wrote gear q's kills as q x primes; they are
q x every m = +-1 mod 6 in the band, the composite m masked on the same side. Scorecard in
the pre-registration file.
