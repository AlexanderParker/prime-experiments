# twin-percentile - the twin difference sits at the 13th percentile of difficulty inside its own family

Status: COMPUTED (exhaustive finite computation, script-verified; not a theorem).
Prior-art verdict: NOVEL AS FAR AS SEARCHED. See section 6, checked 2026-08-23.

## 1. What it is

Plain language. Polignac's conjecture is a family of statements, one per even gap
2d, and the twin prime conjecture is the d = 1 member. It is folklore that all
members are "equally hard" (the parity barrier blocks them all uniformly). This
finding measures, inside the finite combinatorial family that controls the
conjecture's windowed form, where the twin case actually sits - and it sits near
the EASY end: about 77% of the differences in its own class are strictly harder,
and the hardest is more than twice as hard, in the exact bounded quantity.

Precise form. For the machine of primes <= y, and each even difference 2e, let
F_e(y) be the maximal window length avoiding a slot with both members coprime to
the primorial (the per-difference paired-Jacobsthal quantity; see
paired-jacobsthal-values.md). At gears <= 13, restrict to the 2,880 differences e
coprime to P = 15015 - the hardest gcd class, the one containing the twin
difference. Exhaustively over that class:

- F_e ranges 30..75, mean 38.83, median 39;
- the twin difference gives F = 33 - rank 385 of 2,880, the 13.3rd percentile;
- 77.2% of coprime differences have a strictly LARGER maximal gap;
- the extremal differences reach 75 = 2.27x the twin value.

At gears <= 17 the picture persists: twin 54 vs family max 96 (1.78x), the 21st
percentile (coprime class, same convention as the y = 13 number; exact tie-aware
counts in the round-20 addendum below). Structural reading: the twin difference has delta_q(e) =
min(e mod q, q - e mod q) = 1 for EVERY gear q - it is the maximally clustered
member of the family, while every extremal profile spreads its deltas
((1,1,1,3,6) at <= 13, (1,1,2,4,6,8) / (1,1,2,3,4,3) at <= 17).

Consequence, stated carefully: the twin case of Ziller-Morack's Conjecture 6
(which is the project's Reduction A) is strictly the easy end of "prove it for
every even difference" - by a measured factor > 2 in the bounded quantity, not by
a constant. Any method strong enough to handle the EXTREMAL differences of the
family would give all of Polignac at once; conversely, "the method handles the
twin case; the general case is similar" is measurably false inside this family.

Companion measurement: density does not determine the extreme. Across the 31 gcd
classes at gears <= 13, F_max/lambda (extreme over mean gap) ranges 2.88
(gcd = 5005) to 7.52 (gcd = 3) - two classes with the same mean gap can differ by
more than 2x in maximal gap. The d-dependence is genuine second-order structure
that the density heuristic (Hardy-Littlewood singular series) misses.

## 2. Why it might be novel

The literature treats the even-difference family uniformly. Ziller-Morack bound
all differences at once (one function h_2 = max over the family; their reduction
to Goldbach/Polignac uses the same bound for every d), and their condensed
computational form eliminates the difference variable entirely - so their
framework cannot even ASK where a given difference sits in the family. The
sieve-theoretic literature ranks differences by DENSITY (the singular series
factor, classical since Hardy-Littlewood), which is a different and - per the
companion measurement above - non-determining statistic for the extremal
quantity. No published work found ranks the even differences by the exact
covering maximum, places the twin difference within that ranking, or observes
that the twin case of the Conjecture 6 family is quantitatively easy.

What it is NOT a restatement of: (a) the singular-series density ordering (twins
are the DENSEST pair class among coprime differences by that measure; the finding
is about the maximal gap, where density provably does not decide - measured
above); (b) parity-barrier folklore (that is about methods for the infinite
conjecture, this is a finite measured asymmetry in the reduction target);
(c) Kourbatov-Wolf-style maximal-gap statistics for actual prime pairs (that is
the prime realm at large x; this is the covering/sieve realm at fixed y, the
quantity Conjecture 6 bounds).

## 3. Proof / verification

Status: COMPUTED (exhaustive finite computation at y = 13 and y = 17; percentile
is exact rank arithmetic, not sampling). Not a theorem - the numbers are facts,
the reading "twins are the easy end" is an interpretation of two data points of
y plus the delta-profile structure.

- research/jacobsthal_family.py, research/jacobsthal_h2_17.py - the exhaustive
  per-difference computation and percentile/rank extraction (2,880-strong coprime
  class at y = 13; full 127,627-difference family at y = 17).
- research/maximiser_shape.py, research/why13.py - the delta-profile structure
  behind "maximally clustered vs spread".
- The gcd-class F_max/lambda spread (2.88-7.52) is from the same family
  computation, recorded in docs/proof-search/harvester.md section 3.

## 4. Implications

Inside the project: Reduction A (the twin case) is the right FIRST target - it is
genuinely the cheapest member of the family, so closing it does not close
Polignac, and route arguments must not silently assume twin-case constants
transfer to general d (the budget audit already prices each d separately). The
percentile also explains why the corridor/cap machinery generalises easily (it
was built at the easy end) while per-d constants degrade toward the extremal
differences.

Outside: two exportable statements. (1) A sharpened, per-difference reading of
Ziller-Morack's Conjecture 6: the uniform bound p_n^2 - p_n has, at y = 13, a
factor-4.7 slack at the twin difference (156/33) but only 2.08 at the family
extreme (156/75) - the conjecture's tightness lives at unstructured differences,
not at twins. (2) A concrete caution for the "twin case first" instinct: in the
one finite family where difficulty is exactly measurable, the twin case is at
the 13th-21st percentile, and the ratio to the extreme grows the quantity that
any uniform method must control.

## 4a. Round-20 addendum: external validation through y = 43, and exact percentiles

The "two data points of y" caveat is now materially weaker. Using Ziller-Morack's
published h_2 table (arXiv:1706.03668) as an EXTERNAL denominator (family max =
h_2/2) against the project's own fixed-twin ladder F(2,y), the twin-to-extreme
ratio is now known at TWELVE machines (research/zm_margin_mechanism.py section F):

    y        5    7    11    13    17    19    23    29    31    37    41    43
    twin F   6   15    21    33    54    75   102   129   174   264   273   309
    max F    9   15    33    75    96   129   183   225   285   354   447   522
    extreme/twin  1.50 1.00 1.57 2.27 1.78 1.72 1.79 1.74 1.64 1.34 1.64 1.69

The twin difference attains the family maximum only at y = 7; at every other
machine it does NOT, and from y = 11 on the extreme runs 1.34x-2.27x the twin
value (median 1.70x over y >= 11). The one high twin share (0.746 at y = 37) is
the twins' own near-budget jump 2.432 q' at 31->37 - the corpus's known outlier -
and it relaxes back to ~0.6 immediately after. The y = 13 point (2.27x) is the
family's extremal-jump step (see paired-jacobsthal-values.md section 4a), not the
trend.

Exact tie-aware percentile bookkeeping (research/family17_percentile.py; per-class
arrays saved to research/data/f13_family.npy, f17_family.npy):

    y=13 coprime class (n=2,880):  below 384 (13.3%)  ties 272 (9.4%)   above 2,224 (77.2%)
    y=13 full family  (n=7,507):   below 4,519 (60.2%) ties 396 (5.3%)  above 2,592 (34.5%)
    y=17 coprime class (n=46,080): below 9,824 (21.3%) ties 4,640 (10.1%) above 31,616 (68.6%)
    y=17 full family  (n=127,627): below 84,859 (66.5%) ties 6,920 (5.4%) above 35,848 (28.1%)

So the headline "13.3rd/21st percentile" numbers are the coprime-class
strictly-below shares; within the twin difference's own (hardest) gcd class,
68.6-77.2% of differences are strictly harder at both machines. Note the full
family is easier on average than the coprime class (most non-coprime classes are
denser machines with smaller F), which is why the full-family percentile is
higher; the like-for-like comparison is the coprime class.

## 5. Unsolved questions or conjectures it touches

- Polignac's conjecture (per fixed even gap) and the twin prime conjecture: gives
  measured structure to the difficulty ORDERING of the family, which the parity
  barrier treats as uniform.
- Ziller-Morack Conjecture 6: locates where the bound is tight (extremal
  unstructured differences), suggesting a per-difference refinement of the
  conjecture as the natural sharper statement.
- The project's route: whether the lemma-(D) constants degrade monotonically with
  percentile rank is open; the extremal delta-profiles are the stress test.
- Stability question: does the twin percentile converge as y grows (13.3% at 13,
  21% at 17 - two exhaustive points; the RATIO to the extreme now has twelve
  points through y = 43 via ZM's table, median 1.70x, sec. 4a - the percentile
  itself still needs per-difference scans beyond 17)?

## 6. Prior-art check (2026-08-23)

Searches run:

- Fetched and read in full: Ziller-Morack arXiv:1706.00317 (theory paper) and
  arXiv:1706.03668 with its 32-page ancillary full_details.pdf (computation).
  Their treatment of differences: one uniform bound h_2 for the whole family;
  the condensed function omega_2 (two free residues per prime) FORGETS the
  difference; the ancillary files list all extremal sequences (in modulus,
  remainder, and permutation representations) but tabulate no per-difference
  values and contain no comparison of the twin difference to any other - no
  percentile, no ranking, no easy/hard statement anywhere in either paper.
- WebSearch: `maximal gaps between prime pairs "d = 2" generalized twins Polignac
  Kourbatov Wolf` -> nearest hits: Kourbatov & Wolf, "Predicting maximal gaps in
  sets of primes" (Mathematics 7 (2019), arXiv:1901.03785) and Wolf (J. Integer
  Seq. 23 (2020)); also the javascripter.net "Maximal gaps between prime
  k-tuples" tables (Kourbatov). These measure and model maximal gaps between
  ACTUAL prime pairs/k-tuples at various differences up to 10^14 - a
  difficulty-across-the-family measurement, but of a different object (prime
  occurrences at large x, Gumbel-fitted), in which per-difference differences
  are normalised away by the tuple's own density; none ranks the covering/
  Jacobsthal-type quantity, and none states a twin-is-easy (or twin-is-hard)
  conclusion for a Polignac-family reduction.
- WebSearch: `"even difference" OR "even gap" prime pairs which is hardest
  "Jacobsthal" fixed difference maximal gap coprime primorial` -> jumping-
  champions literature (Odlyzko-Rubinstein-Wolf class: MOST COMMON gap between
  consecutive primes - a density ranking, different object and different
  ordering), Ziller arXiv:2007.01808 (which even numbers occur as gaps between
  consecutive coprimes-to-primorial - single-number realm, no pair family), and
  the papers already covered. Nothing ranks Polignac cases by difficulty.
- WebSearch: `"paired Jacobsthal" OR "generalised Jacobsthal function for paired
  progressions" -Ziller` -> no third-party follow-up literature on the paired
  function at all (2017-2026).
- OEIS text-interface lookups (per-difference family values, any of which a
  prior per-difference study would have deposited): 21,33,54,75,102 (twin
  ladder); 33,54,75,102,129; 16,28,39,57,65 (F for d = 0 mod 6); 42,66,108,150,204;
  264,273,309 - ALL no results. Keyword "paired Jacobsthal" -> A288815 only
  (the max-over-family values).
- Classical shadow checked from knowledge, not search: the Hardy-Littlewood
  singular series orders pair classes by density (twins densest among coprime
  differences, factor prod (p-1)/(p-2) for p | d); the finding's companion
  measurement (F_max/lambda spread 2.88-7.52 at equal density) is precisely the
  statement that this classical ordering does not determine the covering
  extreme, so the percentile is not derivable from it.

Nearest prior art: Ziller-Morack 2017 (same family, no per-difference data, no
ranking); Kourbatov-Wolf 2019-2020 (difficulty-like comparisons across prime
pairs at different gaps, but for prime-realm maximal gaps, density-normalised,
with no covering-function analogue and no twin-percentile-type conclusion).

VERDICT: NOVEL AS FAR AS SEARCHED. Both the measurement (exhaustive
per-difference ranking of the paired-Jacobsthal family, twin at the 13.3rd
percentile of 2,880 at y = 13, 21st at y = 17, extreme 2.27x/1.78x) and the
conclusion (the twin case of the Conjecture 6 / Polignac reduction family is
measurably the easy end, by a factor > 2, and density does not determine the
extreme) appear in no located publication or OEIS entry. Caveat honestly priced:
the percentile rests on two values of y; the interpretation is only as strong as
that trend.

## ROUND-22 NOTE (harvester): still novel, but not a paper on its own

Two things changed this round and both are recorded honestly rather than argued away.

1. THE VERDICT SURVIVES. Ziller-Morack's ancillary files (full_details.pdf Table 1's
   nseq column, plus remainders_2.txt / permutations_2.txt / moduli_2.txt) do publish
   the exhaustive MAXIMISER sets, which the round-20/21 checks had missed - but they
   publish maximisers, not a per-difference ranking. Nothing in them orders the family
   or locates the twin difference inside it, so the percentile result stands as
   NOVEL AS FAR AS SEARCHED.

2. THE PUBLICATION PRICING IS DOWNGRADED. This is data with no theorem attached, and
   it rests on two exhaustive machines (y = 13, 17) plus twelve external max-only
   points. It belongs as a section inside the paired-Jacobsthal upper-bounds note
   (docs/novel/j2-upper-bound.md + paired-jacobsthal-values.md), where it plays the
   role it actually plays: the reason the per-difference sieve dimension kappa_d
   matters (the hardest class is the coprime one, kappa_d = 2, and the family extreme
   is 1.3x-2.3x the twin case at every machine). It is not a unit on its own.
