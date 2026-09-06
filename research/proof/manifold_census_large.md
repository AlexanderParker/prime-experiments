# The manifold census at large Q (overnight run, 2026-09-07)

Manager's precomputation for the valves work, run while the owner slept (script
`research/topmachine/r9/manifold_census.py`; outputs `research/topmachine/r9/results/`,
gitignored; every number used is here). The manifold = the primes in (q, Q] acting on [1, Q^2]
by a segmented sieve on those primes only, so an integer is open iff it has no prime factor in
(q, Q] (the quiet-zone rule: open = q-smooth times at most one prime above Q). Validated at
(q, Q) = (5, 10^4) against `top_machine_6.md`: family (1, 1) = 440,107 = pi_2(10^8) - pi_2(10^4)
exactly; quiet-zone record 420 after 26,261 (the branch's 419 at 26,262 in the blocked-count
convention); no gap of 4.

## Results

| q | Q | range | open pairs | quiet-zone record (after position) | position / Q | families | family (1, 1) | gap 4 | gap 3 / gap 5 |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 10^5 | 10^10 | 323,712,289 | 924 (187,907) | 1.879 | 3,824 | 27,411,455 | 0 | 5,790,776 / 7,463,698 |
| 7 | 10^5 | 10^10 | 468,247,493 | 924 (187,907) | 1.879 | 18,009 | 27,411,455 | 0 | 12,419,495 / 12,436,555 |
| 11 | 10^5 | 10^10 | 585,656,884 | 924 (187,907) | 1.879 | 57,053 | 27,411,455 | 0 | 18,788,361 / 18,810,295 |
| 13 | 10^5 | 10^10 | 705,579,231 | 924 (187,907) | 1.879 | 147,100 | 27,411,455 | 0 | 26,106,214 / 26,121,314 |
| 5 | 3 x 10^5 | 9 x 10^10 | 2,369,458,575 | 1452 (850,349) | 2.834 | 5,072 | 203,707,420 | 0 | 35,414,112 / 45,821,140 |
| 7 | 3 x 10^5 | 9 x 10^10 | 3,420,912,134 | 1301 (850,500) | 2.835 | 26,224 | 203,707,420 | 0 | 77,016,729 / 77,106,172 |

Wall time 8 to 60 minutes per run on one core.

## What the table says, mechanism first

1. **At Q = 10^5 the manifold's quiet-zone record is the same gap at the same place for all
   four engines.** 924, starting after 187,907, for q = 5, 7, 11, 13. Checked by factoring:
   187,907 and 187,909 are twin primes, and the next twin pair is 188,831 and 188,833; the
   position lies in the bottom stratum (Q, 2Q], where the only open numbers are primes and
   q-smooth numbers, so the gap is a twin-prime gap and the engine cannot touch it. This is
   the ROOT verdict of `top_machine_6.md` (L65) made concrete at scale: the manifold's own
   record is a twin-prime gap.
2. **Family (1, 1) is identical across engines**: 27,411,455 at Q = 10^5 and 203,707,420 at
   Q = 3 x 10^5, for every q. It is the twin primes in (Q, Q^2], and raising q adds families
   (3,824 to 147,100) without changing that one. The obstruction lives in the one family no
   valve can touch.
3. **At Q = 3 x 10^5 the record differs between q = 5 and q = 7 at the same place, and the
   difference is one pure-byproduct pair.** The twin gap 850,349 -> 851,801 is 1,452 long.
   For q = 7 the number 850,500 = 2^2 3^5 5^3 7 is 7-smooth (pure air, open under the manifold)
   and 850,502 = 2 x 425,251 with 425,251 prime above Q, so (850,500, 850,502) is an open pair
   of the family (850500, 2) that splits the gap into 151 + 1,301. For q = 5 the factor 7 is
   a manifold gear and strikes it, so the gap stays 1,452. A smooth number reaching into the
   second stratum is exactly the air of the valves picture: it shortens the manifold's gap and
   is burnt by the engine.
4. **The forbidden gap 4 holds at 10^10 and 9 x 10^10**: 0 gaps of 4 among 2.4 to 3.4 billion
   open pairs per run.
5. **The gap-3 = gap-5 identity survives on ranges as an approximate equality whenever 7 is
   not a manifold gear** (q >= 7: 12,419,495 against 12,436,555; 18,788,361 against 18,810,295;
   26,106,214 against 26,121,314; 77,016,729 against 77,106,172) and fails by 29% when 7 is a
   gear (q = 5), as W24 says.
6. **The largest gaps overall are the smooth-zone gap** (Q - 160 - 2 at q = 5: 99,991 and
   299,989 after 160; Q - 8,748 - ... at q = 7: 91,403 and 291,401 after 8,748; at q = 11 the
   list below Q ends at 19,600 and the gap is 80,551; at q = 13 at 21,294, 78,857), the
   Stormer-Lehmer constants of W64, and then the quiet-zone record two to three Q above the cut.
   Above 10^9 the largest gaps are 600 to 651 at q = 5, i.e. the mixture of families keeps the
   high strata far below the bottom stratum's twin gap.

## Use

For the valves: the manifold's open set in the quiet zone is a union of families; the engine
vents every family but (1, 1); the record is made by (1, 1) alone in the bottom stratum and is
shortened above it by air pairs. Any valve law must explain why (1, 1) is never empty in
(Q, Q^2], and the table says that count is the twin-prime count, unchanged by the split. Not a
new statement; a measured location for the old one.

## Addendum (2026-09-07): the identical record is a mechanism, and it is the usual case

Owner's question: four engines with the identical gap, more than a coincidence?  Yes, a
mechanism.  In the bottom stratum (Q, 2Q] every open number is a prime or a q-smooth number
(a charge s x P with s >= 2 already exceeds 2Q).  So an open pair there is a twin prime pair, or
a smooth number with a prime neighbour at distance 2.  The smooth numbers are so sparse that the
record twin gap contains none with a prime neighbour: inside 187,907 .. 188,831 there are 0, 1, 2,
3 q-smooth numbers for q = 5, 7, 11, 13, and none of them has a prime at distance 2.  Hence the
same gap for all four engines.

How often (exact scan of (Q, 4Q] for Q = 20,000 .. 200,000 step 10,000, q = 5, 7, 11, 13):
the quiet-zone record is identical across the four engines at **14 of 19** values of Q.  It is
also sticky: the twin gap 187,907 -> 188,831 is the record for every Q from 100,000 to 180,000
(and already for q = 5 from 70,000), then 251,969 -> 252,827 (858) takes over at 190,000.  The
five non-identical cases are all splits by a smooth number with a prime neighbour: at
Q = 20,000 the gap 498 after 24,419 is split to 422 for q = 13 only; at 70,000 .. 90,000 the
q = 5 record 924 is split for q >= 7 by 7-smooth numbers landing inside, giving 673 / 780 / 827,
and for q = 13 again by 13-smooth ones (600, 687).

Law W103 (measured, mechanism stated): the manifold's quiet-zone record at Q is the largest
gap between consecutive twin primes in its low strata, shortened only where a q-smooth number
with a prime neighbour lands inside the gap; the engine's size enters through the smooth numbers
alone, and the twin primes themselves are the same for every engine.  (ROOT: bounding it is
bounding twin gaps.)
