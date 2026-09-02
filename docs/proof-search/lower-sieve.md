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

## 5. The new gear's stride over the lower sieve (human's follow-up)

Pre-registration and scorecard research/data/r29/stride_prereg.md; script
research/stride_r29.py; log research/data/r29/stride.log (lower sieves {5} to {5..23}, new
gear 7 to 29, P up to 37182145; 4/5 gates, S5 refuted in the flat direction).

The new period qP is q copies of the lower sieve's word. Gear q's teeth hit 2 of every 7
slots; in copy j (slots jP .. jP+P-1) a hit lands on an already-blocked slot (nothing) or on
an old opening (a new block). Slot coordinates fold 2 and 3: the human's "2, 3, 5, length 30"
is gear 5 alone, five slots; 7 x 30 = 210 numbers is 35 slots; gear 7 hits 10 of them.

Forced (S1, S2, checked by direct enumeration to q = 17). Let n(r) be the number of old
openings a in [0, P) with a = r mod q. Gear q hits jP + a iff a = +-u - jP (mod q), so

    new blocks in copy j = n(u - jP mod q) + n(-u - jP mod q),

summing to 2N over the q copies (every old opening loses exactly 2 lifts). The whole stride
pattern is the residue histogram n(r) of the old openings mod q read through the pairing
j -> +-u - jP. The old word is a mirror, so n(r) = n(P - r) up to the slot a = 0, and the copy
profile is palindromic: copy j and copy q-1-j get the same number of new blocks up to 1.
Per period, 2P hits, 2N new blocks, the rest on the sieve: the fraction of gear q's strides
wasted on already-blocked slots is 1 - prod (1 - 2/g) = 0.40, 0.57, 0.65, 0.70, 0.74, 0.77,
0.79 at q = 7 .. 29, rising to 1.

The human's example, {5} + 7 (35 slots, 7 copies of 5): copy 0 one hit (k = 1, the slot
5 | 7, already gear 5's) UNTOUCHED; copy 1 two hits, one new block; copies 2, 3 one hit each,
new; copy 4 two hits, both new; copy 5 two hits, one new; copy 6 one hit (k = 34,
203 = 7 x 29 | 205 = 5 x 41, already gear 5's) UNTOUCHED. The mirror pair 0, 6.

Untouched copies (S3). Only at that first step. {5, 7} + 11: the old openings' residues mod 11
miss {4, 9}, but no copy has both its hit residues +-2 - jP empty; 1 to 4 new blocks per copy.
From {5, 7, 11} + 13 on every residue mod q holds an old opening and no copy is ever untouched
again: every copy loses new blocks.

The pattern as the machine grows (S4, S5). The profile flattens far faster than random: new
blocks per copy run 20-23 (q = 13), 169-178 (17), 2340-2351 (19), 32923-32935 (23),
548402-548442 (29). I pre-registered square-root fluctuation (spread/sqrt(mean) in [2, 6]);
measured 0.66, 0.68, 0.23, 0.07, 0.05 - refuted, the spread is bounded, not square-root. The
reason is forced once seen: n(r) counts the t < P/q with r + qt open, a two-forbidden-
residues-per-gear condition on t, and inclusion-exclusion over the m gears has 3^m terms each
with error below 1, so |n(r) - N/q| < 3^m whatever q is (measured at most 2, 3.4, 4, 6, 17 for
q = 13 .. 29). The lower sieve's openings are equidistributed modulo the new gear to a
bounded error; a new gear never finds a rich copy or a poor one, it takes 2N/q from each
copy up to a constant.

Reading in the window: once P > W (q >= 11) the window sits inside copy 0, and gear q's
in-window hits on old openings are the near-twins of section 2 (at most three per rung);
everything else it strides over below q_next^2 is already the lower sieve's.

Two corrections to the human's reading of the copy tables, for the record. Copy 0 of the
{5, 7} lower sieve under gear 11 is not untouched: gear 11 makes one new block there, at the
slot (11 | 13) - the gear's own prime. Untouched copies exist only at {5} + 7.

## 6. Twin-twin spacings on the section (human's follow-up: 210, 2310, 2520)

Script research/pair_spacing_r29.py, log research/data/r29/pair_spacing.log; sections
1000 <= q <= 5000 (501 sections), pairs of new twins of the same section at slot spacing D
(numbers 6D), D <= 450.

Forced mechanism. In the lower sieve's word the number of residues a mod g with a and a + D
both open is c_g(D) = g-2 if g | D, g-3 if D = +-2u mod g, g-4 otherwise; pairs of openings
at spacing D carry the weight prod_g c_g(D). Relative to a generic spacing (no small gear
divides D or D +- 2u) this is x 3 for 5 | D, x 5/3 for 7 | D, x 9/7 for 11 | D: 30 -> 3, 210 -> 5,
2310 -> 6.43 from the first three gears alone, and 2520 = 2310 + 210 gets only 210's factor
(11 does not divide it).

Measured on the sections (count / generic mean, predicted in brackets): 6: 1.05 (1.00);
12: 2.68 (2.67); 30: 3.95 (4.00); 42: 3.78 (3.81); 210: 5.70 (5.56); 420: 6.09 (6.10);
1260: 5.47 (5.55); 2310: 6.55 (6.92); 2520: 5.32 (5.56). Over all 450 spacings
observed / predicted averages 0.991 (min 0.924, max 1.105). The most frequent spacing below
2700 is 2310 (9759 pairs), then 420, 2550, 1890, 210. So yes: twins favour 210 and 2310, and
2520 only as a multiple of 210. The trend is the lower sieve's pair weight, inherited by the
twins because twins are the word's openings that turned out prime - CRT, not a new law.
