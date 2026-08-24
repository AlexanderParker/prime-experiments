# Harvester workstream - side theorems and adjacent conjectures (COMPACTED)

Compacted 2026-08-23; full verbatim rounds 1-19 log at
archive/harvester-full-r1-19.md. Nothing below is new; every claim has its
reproduction pointer (section 7).

MANDATE: statements weaker or adjacent to twin primes where the session's
machinery yields actual results, priced honestly ("not reachable" = not
reachable with currently published methods - an imported corpus limit, distinct
from any event in the machine itself). Rounds 3-14 were twin-route support by
coordinator steering, not this lane's initiative; back on its own mandate from
round 15. The round-1 ranking (C1-C10) is SUPERSEDED by the round-15 re-ranking
below; the original table survives only in the archive.

## 1. Final ranking (round-15 re-pricing, updated through round 17)

| # | candidate | statement | what moved it | status / honest pricing |
|---|-----------|-----------|---------------|-------------------------|
| C10+N3 | paired-Jacobsthal family F_d (data + structure) | exact values and structure of F_d(y) / h_2 = max_e F_2e (Ziller-Morack arXiv:1706.00317) | pruned F(2,53) search (>= 426), mod-3 endpoint law, slot_cap_gap collapse; rounds 16-17 delivered first h_2 values, percentile location, delta-profile law | TOP. A named function with a literature where the corpus holds both new data and new structure. Bites executed rounds 15-17. |
| N1 | universal Polignac cap | literal chains have <= 12 members for EVERY even gap and every gear; cap decided by a 48-class check on q' mod 105, by gcd(e,105) | capOf_le_twelve (kernel-checked, formalist) + this lane's round-10 gcd table | DONE - standalone finite theorem. |
| N2 | gear-3 non-adjacency | for d = 0 mod 6, two padded links can never be adjacent in any legal word, unconditionally, by gear 3 alone | corridor laws + round-13 computation | DONE - one-line proof: padded step q' not divisible by 3, so openings r, r+q', r+2q' hit all three classes mod 3 and gear 3 blocks one. |
| C2 | Polignac for fixed even gap 2d (the conjecture) | infinitude of prime pairs at any chosen gap | capOf_le_twelve, corridor laws, rounds 13-14 transfer + budget audit | CONJECTURE UNMOVED (parity barrier, imported corpus limit), REDUCTION TRANSFORMED: same single open lemma (D) as twins, uniform in d, budget verified 7 gaps x 5 steps. |
| C4 | Goldbach via the paired-Jacobsthal frame | conjecture (ZM Conjecture 6 class, unreachable); windowed reduction (reachable) | horizon argument; slot_cap_gap has analogue q \| N; exposed-set = HL-factor identification | window reduction DONE round 1 (kernel-checked, with exact converse on central representations). Small live bite: the singular-series arithmetic factor over q \| N IS an exposed-set size - kernel-reachable. |
| C8 | constant-2 fragile law | fragile * pi_win / (twins * W1) -> 2 (measured to 0.43% at y=50021) | horizon theorem + master formula: window primality IS non-divisibility below y, so the fragile census is EXACT gear arithmetic | form improved (explicit inclusion-exclusion, not an empirical fit); ASYMPTOTIC still Hardy-Littlewood-class, unreachable unconditionally. Would need an unconditional handle on the exposed set's pair-correlation, not more measurement. |
| C5 | quadruplets / k-tuples | counting laws p(x) = prod(q-2+2x) etc. | completeness theorem | STRUCK as publishable: re-derives the classical admissibility criterion. Would come back via the F_d family generalised to k-tuples - a computation this machinery can do and nobody has. |
| C1 | Legendre / Oppermann / Brocard bands | a prime in every band | NOTHING, demonstrably | STRUCK. Horizon theorem explains why in one line: gears < y decide the window exactly, so the machinery is about divisibility inside a window and contains no prime-side localisation; bands need exactly that (exponent 0.5 vs published 0.525, Alweiss-Luo). No tool in the toolkit is of that type. |
| C3, C6, C7, C9 | per-gap reduction iff, overcount census, g=2 pinning, onset bound L0(y) <= 27129 | - | DELIVERED rounds 1-7 (see section 2). C9 was already in-corpus (Brun-Titchmarsh class, diagnostic only). |

Remaining live list, in order: C10+N3 (Jacobsthal family), C4 small bite
(singular-series factor as exposed-set size), N1/N2 as packaging candidates.

## 2. Kernel-checked results (all in proofs/Polignac.lean unless noted; all on
## standard axioms [propext, Classical.choice, Quot.sound] or fewer; ledger
## green at 1252 jobs, zero sorry)

ZM-frame reductions (round 1):
- prime_of_no_factor_le_sqrt: sqrt-graded horizon lemma, per-member, gap-blind.
- SurvivorGap d y m; survivorGap_one_iff (d=1 is BlockedSlots.Survivor).
- slot_cap_gap: odd prime blocks both members of a gap-2d slot => q | d;
  corollary slot_cap_twin. The exact transfer condition for the whole corpus:
  every slot-cap-based law holds verbatim for gap 2d at gears coprime to d, and
  q | d gears collapse to one residue (the Hardy-Littlewood factor,
  mechanically).
- survivorGap_iff_pair: windowed survivor <=> prime pair at gap 2d.
- gapPairs_infinite_iff_survivor_in_window (d): THE per-gap iff - Polignac for
  2d is equivalent to "every scale has a window (y, y^2] containing a gap-2d
  survivor of the gears <= y", both directions, every d (d=0 degenerates to
  infinitude of primes). Sharper than Ziller-Morack Thm 4.1 (their bound is
  sufficient-only, all differences at once); per-difference machine-checked
  equivalences are new.
- goldbach_of_survivor / goldbach_rep_of_survivor / survivor_of_goldbach_rep:
  if some n in (sqrt N, N - sqrt N) has n and N-n free of prime factors
  <= sqrt N then N is a sum of two primes, with the exact converse on
  representations with both parts above sqrt N.

g=2 pinning (round 2) - "twins below y are the unique unconditionally
guaranteed line item of the level-y^2 doubles ledger":
- twin_mod_six: p, p+2 prime, p > 3 => p = 5 mod 6.
- twin_pin: the pair IS slot u = (p+1)/6 (6u-1 = p, 6u+1 = p+2, split
  representative in closed form - the pin is the pair).
- twin_pin_le: u <= (y+1)/6 for every y >= p - bottom band of every window,
  every scale, unconditionally.
- twin_split_class_iff: slot k split-killed by {p, p+2} IFF k = u mod p(p+2).
- twin_mirror_slot (second class at P - u), twin_product_slot (same-member
  double at u(p+1), with 6u(p+1) - 1 = p(p+2) exactly).
- own_slot_pin_gap_two (UNIQUENESS): an odd prime pair (q, q+g) split-killing
  the slot holding q itself forces g = 2. Only twins pin at their own slot;
  other gaps' representatives sit at depth ~P/(6g), alignment-conditional
  (quantitative half paper-side: research/split_gap_law.py).

SAME-side census (round 3):
- six_mul_class (slot-map inversion: one class mod m), left_dvd_iff /
  right_dvd_iff, card_class_Ico (THE FLOOR COUNT: #{k in [1,t] : k = a mod m}
  = (t+m-a)/m for 1 <= a <= m), same_left_census / same_right_census (SAME-side
  pair term: ONE CRT class mod qr, count (t+qr-a)/qr), same_census_once
  (composite root law windowed: a <= t < a+P => count exactly 1),
  same_left_own_value (qr = 5 mod 6 => class rep is qr's own slot),
  class_rep_unique, not_dvd_six.
- twin_pin_self_block: the pin slot u has Census.slotComps u = 0 (real twin
  slot) AND is never a Survivor of any machine with divisor bound >= p - the
  machine is blind to its own pair (why the U-pin list is invisible to n2).

PAIRSPLIT (round 4):
- split_class: q left / r right kills are ONE CRT class mod qr with the floor
  count; mirror is the role swap.
- split_rep_twin_eq_pin (g=2 LOOP-CLOSER): the PAIRSPLIT representative of a
  twin pair IS its own pin slot.
- twin_split_count: split count over first t slots = 1 exactly on
  u <= t < u + p(p+2).

CORR (round 5):
- twoSided_class (THE GENERAL BOTH-SIDED TERM): coprime moduli mL, mR > 1
  coprime to 6 => mL | left, mR | right is ONE CRT class mod mL*mR, floor
  count (t+M-a)/M. Yields EVERY both-sided term of the master formula.
- corr_triple_class (qr | left, s | right: one class mod qrs), and
  corr_triple_signed (the inclusion-exclusion sign, subtraction-free: distinct
  slots + triple = the two split incidence counts), six_coprime_prime.

Assembly (rounds 6-7):
- three_sets_ie (n=3 inclusion-exclusion, subtraction-free, any finsets),
  three_preds_ie, three_gear_assembly (assembled sum = sieve overcount,
  set-level, no primality hypotheses), card_marks_eq (per-gear bridge),
  card_pair_inter_eq (pair bridge: 4 disjoint side classes),
  card_triple_inter_eq (triple bridge: 8 disjoint side classes LLL..RRR),
  card_filter_or_of_excl.
- three_gear_master (END-TO-END, 26 filter-card terms, subtraction-free):
  distinct + 12 pair side classes = 6 single side classes + 8 triple side
  classes, over the first t slots, any distinct odd primes q, r, s; every term
  beyond "distinct" is one CRT class with closed-form floor count. Rearranges
  to overcount = pairs - triples. THE ASSEMBLY LINE FOR 3 GEARS IS CLOSED.
  n > 3 assessed and deferred: mechanical (iterated three_sets_ie or mathlib
  signed inclusion-exclusion), nothing conceptually new.

Pruning cores (round 9):
- AdjBlocked q o i (the covering search's blocking relation), free_class_three,
  free_class_unique_three, endpoint_run_mod_three (THE ENDPOINT LAW: both
  flanks unblocked by gear 3 => 3 | (M+1), i.e. F(2,y) = 0 mod 3 -
  justification of the pruned search's mod-3 skip; all thirteen known exact
  values 33..309 comply). Axioms [propext, Quot.sound] only.
- LEFT-TAUT EQUIVALENCE (paper proof, handed to Formalist via agents-shared):
  Cov(L) <=> Cov(L) with position -1 unblocked by every gear; lets every gear
  drop offsets q-2, q-1. Verified exhaustively y <= 17, every L
  (research/lefttaut_check.py, independent of the pruned code).

Mod-3 dichotomy for the F_d family (round 15):
- GearSurvivor q e n, three_survivors_congr (3 does not divide e => any two
  survivors congruent mod 3), three_dvd_gap (hence every gap, F_d(y) included,
  is a multiple of 3), three_survivors_adjacent (CONVERSE: 3 | e => survivors
  one apart exist, no mod-3 law), no_mod_law_above_three (q >= 5 leaves >= 3
  classes: gear 3 is the only gear that can force such a law). The complete
  sharp dichotomy: 3 | F_d(y) for every gear set <=> d != 0 mod 6. Verified
  15/15 gap classes first (research/jacobsthal_mod3.py); e.g. F_2 = 21, 33,
  54, 75, 102 all divisible by 3 vs F_6 = 16, 28, 39, 57, 65 none forced.

Delivered code: rust2/src/bin/maxgap_pruned.rs (endpoint law + left-taut in
covering-search form; verified identical to original on F(2,y) = 21, 33, 54,
75, 102, 129, 264 for y = 11..37; the mirror-canonical o5 halving is UNSOUND
combined with left-tautness and was removed).

## 3. Computed values

Paired-Jacobsthal h_2(n) = j_2(p_n#), first exact values in the literature
(ZM compute none). Exhaustive over every even difference; h_2 = 2 x max_e F_e
(halved coordinates):

    y    P (odd)   #diffs    h_2   p^2-p   Conj.6   margin
    3        3         1       0      6    holds      -
    5       15         7      18     20    HOLDS    10.0%
    7      105        52      30     42    HOLDS    28.6%
    11    1155       577      66    110    HOLDS    40.0%
    13   15015      7507     150    156    HOLDS     3.8%   <- the dip
    17  255255    127627     192    272    HOLDS    29.4%

- Maximisers: y=13: e = 344, 734, 839, 916, 2164 (all coprime to P, none
  small, none structured); y=17: F = 96 at e = 2791, 3176, 5584, 5794, 6361,
  6571.
- THE 13 DIP EXPLAINED (round 17): the dip belongs to the STEP 11->13, not the
  difference class. It needs both a twin prime step (bound p^2-p grows only
  x1.42) and a clean extension of the extremal delta-profile (h_2 gains fully,
  x2.27). At 17 the profile must compromise (x1.28) and the bound grows x1.74.
  Champions of gears <= 13 are only 99.3-99.8th percentile at 17, not extremal
  - so no difference class owns the dip.
- Delta-profile law (delta_q(e) = min(e mod q, q - e mod q)): maximisers are
  exactly the carriers of specific profiles - (1,1,1,3) at gears <= 11
  (8 winners, F=33), (1,1,1,3,6) at <= 13 (16 of 7507, all F=75; recall and
  precision 100%), (1,1,2,4,6,8) and (1,1,2,3,4,3) at <= 17 (64 winners,
  precision 100%, recall 50/50). Every winning profile begins
  delta_3 = delta_5 = 1. "Maximally spread at the top" fits some maximisers,
  not all - description, not law.
- y=19 lift: exhaustive scan out of reach (2,424,922 differences); lifting the
  gears<=17 elite (1140 candidates) gives h_2(19) >= 222 (F = 111 at
  e = 1,335,364) vs bound 342 (64.9%) - does NOT refute Conjecture 6;
  status at 19 UNDECIDED (17->19 is the next twin step). Expected ~288
  (margin ~16%) on clean extension, ~250 (~27%) on compromise.

Twin location inside its family (percentile-of-family): at gears <= 13,
coprime-to-P differences (2880, the hardest class): F_e range 30..75, mean
38.83, median 39; twin F = 33 = 13.3rd percentile (rank 385 of 2880), 77.2%
of coprime differences have a LARGER maximal gap, extremal is 2.27x twin.
At gears <= 17: twin 54 vs max 96 (1.78x), 21st percentile. Twins have
delta_q = 1 for every q - the maximally clustered member. Consequence: the twin
case of ZM Conjecture 6 (= Reduction A) is strictly the EASY end of the family,
by a factor > 2 in the bounded quantity; and F_max/lambda ranges 2.88
(gcd = 5005) to 7.52 (gcd = 3) over the 31 gcd classes at <= 13 - density does
NOT determine the extreme.

F(2,53): pruned search running from resume point; reproduced "420 coverable",
skipped 421/422 by the mod-3 law; bound >= 426 as of round 14 (coordinator).
Needs <= 486 for the tolerance constant; quadratic-law prediction ~441.
Log research/data/maxgap53_pruned.log. Pruning: mod-3 skip cuts 2/3 of
coverable increments; timing check 1.12s vs 1.74s at y=37 from L=250.

Universal cap table (round 10, COMPLETE over all even d since cap spectrum
depends only on gcd(e,105), e = d/2, and all 8 divisors computed; 48-class
mod-105 invariance, zero mismatches, q' <= 1200):

    gcd(e,105)   |E_d| mod 105   cap spectrum              max cap
        1             15         {2:24, 3:4, 4:14, 6:6}       6
        5             20         {4:24, 6:24}                 6
        7             18         {2:24, 4:12, 6:12}           6
        3             30         {4:36, 5:4, 6:8}             6    <- d = 0 mod 6
       21             36         {4:36, 6:12}                 6
       35             24         {6:48}                       6
       15             40         {6:8, 7:8, 8:24, 10:8}      10    <- ceiling breaks
      105             48         {12:48}                     12    <- absolute ceiling

|E_d| = prod over q in {3,5,7} of (q - r_q), r_q = 1 iff q | e (the collapse is
kernel-checked slot_cap_gap; the HL factor and the exposed-set size are the
same object). 12 IS THE ABSOLUTE CEILING OVER ALL POLIGNAC GAPS.

Route-transfer audit (rounds 13-14) - the tolerance route is a THEOREM SCHEMA
over all even d with one open lemma (D):
- (A) finite word list from q' mod 105: transfers verbatim (48 classes, zero
  mismatches); list sizes {1,2,3,5,8} for 3 not dividing e, {11,12,20,21,23}
  for gcd = 3, {43..56} for gcd = 15.
- (B) literal span <= ceil((cap_d - 1)/2) x q' frame units: <= 5 letters /
  3q' generic, <= 9 / 5q' at gcd 15, <= 11 / 6q' at 105.
- (C) padded count p <= F/c_d, onset gated by F >= c_d; 8/8 configurations,
  zero violations.
- (E) both-flanks-maximal exclusion: forbidden for d=2: 662/980 (68%), d=4:
  696/980 (71%), d=6: 2412/2940 (82%), d=12: 2308/2940 (79%).
- (D) flank bound FS_max(w) <= F + (alpha/3)q' - span(w): contains no
  d-specific structure - THE SAME OPEN LEMMA for every even d.
- Budget arithmetic (round 14, exact full periods, steps 11->13 .. 23->29):
  max incr/q' by d: 2: 1.235 (13->17); 4: 1.846 (11->13); 6: 0.947; 10: 1.421;
  12: 1.538 (11->13); 30: 0.632; 210: 0.483. All 35 (d, step) pairs pass at
  alpha = 2.5 AND 3; worst 1.846 is 26% under 2.5. The only measured value
  near budget is twins' own 2.432 at 31->37 (corpus, adjacent units).

Padding economics (rounds 11-12; frame conflict was UNITS - padded link cost is
q' slot = 3q' halved = 6q' member for twins, vs q' halved = 2q' member for
3 | e; explicit example: machine 31, slots k = 8,288,068 / 8,288,105, gap 37
slot = 111 halved = 222 member, both endpoints on tooth k = 31 mod 37):
factor 3 cheaper absolute for d = 0 mod 6, 1.5x scale-relative (the 3|e machine
is twice as dense, mean gap 16.11 vs 32.21), ~10x availability at machine-31
scale (exp(2q'/lambda) = exp(2.30)); moves padding onset from the sixth step
(twins, 31->37) to the FIRST (d=12 padded winner at 11->13). Supply cross-check
26,184 extrapolated vs 26,366 census (0.7%); links (endpoint on tooth) are
2/37 of supply, ~1,400. Firing law for general d: gap g = 0 mod q' => padded
link, g = +-e mod q' => literal, else illegal; alternation of nonzero letters
forced; F(M+q') = max legal-run span from the OLD machine alone - verified 14/14
configurations exact, the ONLY d-dependence being 2u -> e.

Corridor law d-analogue (round 13): adjacent padded links need openings r,
r+c, r+2c all exposed. Impossible for d=2: 34/74 probes (independently
reproducing lateral's proved 37->41 case), d=4: 40/74, d=6 and 12: 74/74,
d=30: 72/72. For 3 | e it is a THEOREM (N2 above), unconditional. Structural
compensation: padding is cheaper for d = 0 mod 6 but can never repeat
consecutively there.

Word identity transfer (round 10): identity shape F(M+q') = max(F2(M), tiers)
and firing (rests only on gcd(P_M, q') = 1, contains no d) transfer verbatim,
13/13 configurations; tier_1 = F2(M) exactly in every row. Degenerate q' | e:
frame letters collapse to the single value 3q', F(M+q') = F2(M) exactly.

## 4. Refuted / failed predictions (kept as refuted)

- PREDICTED Conjecture 6 breach at y=17 (from ratio extrapolation of the first
  four h_2 points): REFUTED same round by exact computation - h_2(17) = 192,
  holds with 29.4% margin. The margin is non-monotone with a one-off dip at 13.
- FLAGGED gcd(e,105) = 15 and 105 (doubled literal-span constants) as "exactly
  where a budget could fail": REFUTED by round-14 computation - those classes
  have the SMALLEST increments (0.632, 0.483 vs 1.235 twins). Reason
  structural: larger cap comes from a denser exposed set, denser machines have
  much smaller F (63, 49 vs 129 at y=29); density wins.
- "Tooth alternation FAILS for 3 | e" (round-13-era label): WRONG LAW TESTED -
  under the corrected merge law a same-tooth adjacency is a legal PADDED link
  (letter 0 mod q'), not a violation. The observation was real; it carried the
  padding-cost finding.
- "No padded gap at all for d=2" vs mechanic's census of thousands: BOTH TRUE -
  measured below the padding onset (F < 3q') vs machine 31 above it. And the
  round-14 "exponential chasm" phrasing for padding availability: corrected to
  factor 3 absolute / 1.5 scale-relative / ~10x at machine 31. The census
  number 26,366 is padding SUPPLY, not links (~1,400).
- Word-list check first pass: 73/73 "mismatches" from comparing letter VALUES
  where the claim is about RESIDUES - own bug, corrected to zero mismatches.
- Wrap-around artifacts (twice): round-10 letter extractor (differences mod P
  across the period end) and round-11 np.roll kill-status corruption; both
  fixed by computing at absolute positions over two periods, counts to zero.
- C1 band statements: struck after twelve rounds moved nothing (see ranking).
- Mirror-canonical o5 pruning: UNSOUND combined with left-tautness (maps
  left-taut to right-taut coverings); removed, nothing lost.

## 5. Open questions

- (D), the flank bound - the single open lemma of the tolerance route, same for
  every even d ("closing D closes every d", NOT "every d is closed"). The twin
  route itself is not closed.
- Conjecture 6 at y=19: undecided (h_2(19) >= 222 vs 342). Cheap attack in
  reach: enumerate candidate delta-profiles by CRT (2^7 differences per
  profile, few thousand evaluations) instead of the ~1.2e13-op full scan.
- F(2,53) termination: >= 426, needs <= 486 for the tolerance constant;
  prediction ~441.
- Budget arithmetic beyond step 23->29 unchecked for every d (twins' 2.432 at
  31->37 is the one near-budget value known); gcd classes 7, 21, 35 (d = 14,
  42, 70) untested (their exposed sets sit between tested extremes).
- d = 0 mod 6 word grammar: richer alphabet (3 letters, one short) needs its
  own word list before the tolerance route can be quoted there.
- n > 3 assembly (full CORR beyond three gears): mechanical, deferred.
- k-tuple F_d family (max-gap function for k-tuples): computable, nobody has.
- C4 small bite: singular-series factor over q | N as an exposed-set size,
  kernel-reachable.

## 6. Publication-worthy

- "Machine-checked reductions of Polignac-type and Goldbach-type statements to
  paired-Jacobsthal window bounds" (Polignac.lean + BlockedSlots.lean + the
  F(2,y) data). Contains no progress on any conjecture - it is the frame made
  formal; per-difference equivalences are new vs Ziller-Morack.
- First computed h_2 values (18, 30, 66, 150, 192) + F(2,y) table: OEIS + note.
- The percentile result (twin difference at the 13th percentile of its own
  family; "the method handles the twin case; the general case is similar" is
  measurably false, and a method reaching extremal differences would give all
  of Polignac at once).
- N1 (universal cap <= 12, complete gcd(e,105) table) and N2 (gear-3
  non-adjacency) as standalone finite theorems.
- The mod-3 dichotomy for F_d (complete iff, sharp at gear 3, operationally
  paid for via the pruned search).

## 7. Pointers (reproduction)

Proofs (kernel): proofs/Polignac.lean (all sections above; registered in
proofs/lakefile.toml). Composes with BlockedSlots, Horizon, Layer, Supply,
Census, Bridge, Gear.

Scripts (research/): polignac_transfer_check.py (round-1 iffs), twin_pin_check.py
(g=2 pinning, 81 twin pairs + uniqueness scan to 400), same_census_check.py
(105 pairs), pairsplit_check.py (210 ordered pairs), corr_triple_check.py
(60 two-sided cases), assembly_check.py + master3_check.py (assembly + 26-term
identity), lefttaut_check.py (pruning cores, independent), literal_cap_gap_d.py
+ literal_cap_mod105.py (cap tables), word_identity_gap_d.py (W1-W5),
firing_padding_gap_d.py (firing law + padding), frame_reconcile.py +
pad_count_bound.py (units + count bound), route_transfer_audit.py (A/B/C/E +
corridor), budget_per_d.py (per-d budget), jacobsthal_mod3.py (mod-3
dichotomy), jacobsthal_family.py + jacobsthal_h2_17.py (h_2 values,
percentiles), why13.py + maximiser_shape.py + h2_19_lift.py (round 17),
split_gap_law.py (depth ~P/(6g) quantitative half), general_gap.py (class
count), topgap_endpoint_law.py (fixed-offset endpoint law).

Search: rust2/src/bin/maxgap_pruned.rs; log research/data/maxgap53_pruned.log.

Lean environment notes preserved for the team: omega does not combine
congruences across moduli - decompose to one modulus; import
Mathlib.Data.Nat.ModEq for [MOD n]; Nat.dvd_sub here is the old Nat.dvd_sub';
Finset.card_insert_of_notMem rename; Nat.Ico_succ_right_eq_insert_Ico lives in
namespace Nat; beware rwa rewriting the ModEq modulus occurrence; count
primitive pattern: induction + Nat.succ_div_of_dvd/not_dvd avoids
division-by-variable omega limits.

## CORRECTION (manager, 2026-08-23, from the prior-art sweep)

The premise "Ziller-Morack compute no h_2 values" is FALSE. Their companion note
arXiv:1706.03668 (11 days after the theory paper 1706.00317, which we read) computes h_2
exactly for all p_n <= 73; Table 1 contains our 18, 30, 66, 150, 192 verbatim. Consequences:
our five values are an exact independent REPLICATION (cross-validation, not first computation);
their h_2(19) = 258 < 342 SETTLES our open y=19 question (Conjecture 6 holds, margin 24.6%,
round-17 "~250" prediction right); the 3.8% dip at 13 remains the UNIQUE extreme through
p_n = 73 in their full table - the "why is 13 extremal?" question stands, now with 12 more
data points. What remains ours: the per-difference family F_d(y), the fixed-twin ladder
F(2,37)=264 / F(2,41)=273 / F(2,43)=309 / F(2,53)>=426, maximiser/delta-profile structure.
Full analysis: docs/novel/paired-jacobsthal-values.md.

## ROUND 20 - why 13, sharpened percentile, and the literature adjacency of the two frames

All computations single-run, assertions green; scripts research/zm_margin_mechanism.py,
family17_percentile.py, paired_holt_recursion.py; outputs in research/data/ (same stems,
.out) plus per-class arrays f13_family.npy / f17_family.npy (stop recomputing these).

### (a) WHY IS 13 EXTREMAL - answered as four exact events (details: docs/novel/
### paired-jacobsthal-values.md sec. 4a)

1. QUANTISATION: slack B - h_2 is quantised mod 6 (min admissible 6 for p = 1 mod 6,
   2 for p = 5 mod 6). The minimum is attained at p = 5 and p = 13 ONLY, through 73.
   The 13 dip is "one quantum above equality": omega_2(6) = 24 = cap 25 - 1.
2. STEP LAW over ZM's 18 steps: margin falls at ALL 6 twin steps (>= 13), rises at
   ALL 5 gap-6 steps, gap-4 mixed (3 up 2 down); absolute slack falls ONLY at twin
   steps (->13, ->31, ->61). Crossover mechanism: d(B)/B ~ 2g/p vs d(h_2)/h_2 ~ 2r/p,
   so the sign flips at gap ~ r (mean ~2). A dip needs r >> g.
3. UNIQUE JUMP: r = Delta(maxF)/q' = 3.231 at 11->13 is the unique value > 2.6 in
   all 18 steps (runner-up 2.553 at ->47). The dip = that outlier on a twin step.
4. WHY THE JUMP - THE LAST CLEAN-EXTENSION STEP: winners extend winners at 7->11 and
   11->13 (16/16 winners at 13 have F_11 = 33 -> F_13 = 75, same fixed e), and NEVER
   again: best 17-extension of a 13-winner is 87 vs true max 96; the 19-argmax
   restricts to the twin's own value (54) at 17 with 35,848 classes above it
   (merge-law round's rank claim verified from my full 17-scan). So 13 is where the
   family maximum last grows by full profile extension - landing on a twin bound-step.

ROUTE-RELEVANT BUDGET EVENTS (for Constructor's pricing, measured not argued): fixed
differences exist with single-step increments 3.231 q' (e = 344, 11->13), 3.947 q'
(e = 1,532,627, 17->19, verified 54 -> 129 by direct construction), 4.435 q'
(e = 107,207,699, 19->23, verified 81 -> 183). Round-14 audit's structured-d worst was
1.846, twins' own 2.432. NO uniform alpha <= 3 increment budget holds over the full
family; "closing (D) closes every d" needs per-d constants (or a family-max exclusion),
and the known family-argmax jumps are non-decreasing (3.23, 3.95, 4.43).

### (b) TWIN PERCENTILE - externally validated (docs/novel/twin-percentile.md sec. 4a)

Using ZM's h_2 as external family-max denominator: twin/extreme known at 12 machines
y = 5..43; twin attains the max only at y = 7; extreme runs 1.34x-2.27x twin, median
1.70x (y >= 11). The 0.746 share at 37 is twins' own 2.432 q' outlier jump, relaxing
immediately. Exact tie-aware percentiles: coprime class below-share 13.3% (y=13) /
21.3% (y=17); strictly-above 77.2% / 68.6%. Publication statement now: "in the one
family where difficulty is exactly measurable, the twin case sits at the 13th-21st
percentile of its own hardest class, the extreme is 1.3x-2.3x harder at every one of
twelve machines (externally cross-checked against Ziller-Morack's independent table),
and density does not determine the extreme."

### (c) LITERATURE ADJACENCY OF THE TWO FRAMES (my mandate under the directive)

HOLT-RUDD, READ PROPERLY (arXiv:1510.00743 in detail via ar5iv; 1408.6002; earlier
1402.1970). What they have: the cycle-of-gaps recursion (concatenate p' copies, close
at elementwise-product positions - our merge transform, one residue class); an EXACT
population dynamics n_{s,j}(p'#) = (p'-j-1) n_{s,j}(p#) + driving terms; transfer
matrix with diagonal (p-j-1)/(p-2), superdiagonal j/(p-2), and p-INDEPENDENT
eigenvectors (binomial/Pascal) - hence closed-form asymptotic gap-population ratios,
Polignac-in-the-sieve (their Thm 5.5) and HL Conjecture B ratios in cycles. What they
lack: any maximal-gap tracking (explicitly out of scope for them - our merge law owns
that readout), any two-residue/paired object, any per-difference family.
THE IMPORT, DELIVERED THIS ROUND: the PAIRED HOLT RECURSION - exact linear population
dynamics for two-residue sieves, n_g(M+q') = sum_w coef(w) n_w(M) with the position-
free coefficient coef(w) = #{r in Z_q' : flanks alive, interiors in T}; verified
EXACT for every gap value at 4 rungs (slot 5005->85085->1616615; family e=344 +17;
gcd collapse e=102 +17 = Holt's own case). Diagonal = Lateral's c_q(g) law exactly
(two lanes' constructs are one object); word-survival eigenvalue scale
(q'-2j-2)/(q'-2) vs Holt's (q'-j-1)/(q'-2) - the paired system contracts twice as
fast per word length. docs/novel/paired-holt-recursion.md. Remaining import not yet
used: Holt's eigenvector p-independence -> closed-form asymptotic paired-gap ratios
(would turn Mechanic's histograms into theorems); flagged for next round.

EXPONENTIAL SUMS OVER SIEVE RESIDUES: the flagship prior art is Iwaniec's bound on
the ordinary Jacobsthal function, j(n) << (omega log omega)^2 (via the linear sieve;
the only known route to quadratic-log control of exactly our F-type quantity; ladder
below it: Kanold 2^k, Stevens polynomial). THE PAIRED LADDER IS EMPTY: Ziller-Morack
(both papers re-read in full this round) prove NO upper bound on j_2 of any strength
- their Remark 2.2 lists only elementary monotonicity (product, gcd, prime-power
collapse) - cite no Iwaniec, and give NO heuristic for the p^2 - p bound; no
follow-up literature supplies one. So "any nontrivial upper bound on j_2(p#)" is an
open problem with zero published attempts, sitting directly against our exact table.
Honest pricing: an Iwaniec-method transfer is parity-adjacent and hard; but the
gap between nothing and anything is where a harvest can live. Related toolkit for
covering questions of our type: Filaseta-Ford-Konyagin-Pomerance-Yu (JAMS 2007,
sieving by one class per large modulus), Hough (minimum modulus), Balister-Bollobas-
Morris-Sahasrabudhe-Tiba distortion method - all one-class-per-modulus; none paired.

TRANSFER-MATRIX SIEVES: searched with Holt excluded - there is NO other
transfer-matrix sieve literature; the frame is Holt's alone (plus unrelated physics
usage). One unreviewed Zenodo preprint (Ojaroudi 2026, claimed unconditional twin
prime theorem, "replication-deletion primorial sieve") located and assessed: no
population recursion, no explicit coefficients, claim class far beyond method.

ALSO SETTLED THIS ROUND: ZM's computation note has NO growth-rate commentary and no
remark on the 13 case, so the h_2 ~ (p^2-p)/2 empirical share and the step law above
have no counterpart in print.

### Ranking changes (honest pricing)

- C10+N3 stays TOP and the round-17 "why 13" question is now CLOSED as four exact
  events (quantisation, step law, unique jump, last-clean-extension). What remains
  open there: WHY clean extension dies at 17 (a profile-collision mechanism - the
  best extension loses by exactly 9; unexplained), and the per-difference Conjecture
  6 refinement as a publishable statement.
- NEW CANDIDATE (N4): the paired upper-bound problem - any proved bound
  j_2(p#) < f(p) at all. Literature ladder empty (established above). Bites in
  reach: (i) an elementary Kanold-type bound via the paired machinery; (ii) the
  paired Holt recursion as a route to population lower bounds that force gaps to
  close. Priced honestly: full Iwaniec transfer is beyond current published methods.
- NEW CANDIDATE (N5): the paired Holt eigen-analysis - closed-form asymptotic
  paired-gap population ratios (HL Conjecture B in paired cycles). The recursion is
  verified; the eigenvector analysis is a bounded computation, next-round sized.
- The h_2 ~ (p^2-p)/2 share: recorded as observation + candidate conjecture
  (paired-jacobsthal-values.md 4a); prior-art clean.

### Needs from other lanes

- CONSTRUCTOR: the paired Holt matrix is the transfer-matrix formulation the
  directive assigned you for p_j - the coefficient formula is in
  paired_holt_recursion.py (20 lines); your anti-correlation deficit should be a
  spectral statement over it. Also: the budget events above mean (D)'s constant
  cannot be family-uniform - price per-d or exclude family argmaxes explicitly.
- MECHANIC: your joint gap-pair census n_(g1,g2) is the length-2 input row of the
  recursion; the recursion PREDICTS your next-machine histograms exactly - a
  falsification target at 29/31 scale if you want one.
- LATERAL: your c_q(g1,g2) is coef of two-letter words in the same matrix; the
  interior-disjunction obstruction you named is exactly the coef formula's interior
  clause - it does factorise per-copy (r-wise), which may be the way around it.
- FORMALIST: coef position-freeness + one fixed rung of the paired recursion is
  finite and kernel-checkable (words and residues only); also endpoint c-law cases.

### Deliverables ledger (round 20)

Scripts: research/zm_margin_mechanism.py, research/family17_percentile.py,
research/paired_holt_recursion.py (all assert-gated, all green, single-threaded).
Data: research/data/{zm_margin_mechanism,family17_percentile,paired_holt_recursion}.out,
f13_family.npy, f17_family.npy. Docs: docs/novel/paired-holt-recursion.md (new),
paired-jacobsthal-values.md sec. 4a + pointers, twin-percentile.md sec. 4a + caveat
update, README index entry. No detached jobs launched; nothing pending.

## ROUND 21 - N4 executed (first j_2 bounds), the exact 9 explained, N5 executed
## (paired HL-B in cycles + full diagonalisation)

All three briefed items landed. Scripts assert-gated and green: research/j2_bound.py,
ext_death.py, ext_death2.py, paired_hlb.py; outputs research/data/{j2_bound,
ext_death,ext_death2,paired_hlb}.out. All jobs launched early and finished before
write-up; nothing pending. Prior-art checks run by me, dated 2026-08-24.

### (a) N4 - THE FIRST UPPER BOUNDS ON j_2 (docs/novel/j2-upper-bound.md, new)

The empty ladder now has two rungs, both proved:

1. ELEMENTARY (Legendre IE, complete paper proof in the doc, constants
   script-verified): j_2(p_n#) <= 2*3^(n-1)/V_n + 1 with
   V_n = (1/2) prod_{3<=p<=p_n}(1-2/p) exact; explicitly
   j_2(p_n#) < 3^(n+1) (log p_n)^2 for all n >= 3 (n = 2: exact bound 37).
   Worst case over differences is omega = 2 at every odd prime (per-prime
   E/V factor 3p/(p-2) > 2p/(p-1) always) - the bound is uniform in d, and
   p | d differences get strictly better constants (the F_d refinement).
   Explicit-constant chain: (1-2/p) = (1-1/p)^2 (1-1/(p-1)^2), partial twin
   products decrease to C_2 = 0.66016..., Rosser-Schoenfeld (3.27); verified
   with exact V_n through n = 4203 (worst ratio 0.858 at n = 3). Sub-primorial:
   exp(O(p/log p)) vs the trivial exp((1+o(1))p).
2. POLYNOMIAL (fundamental lemma, dimension kappa = 2, by citation):
   j_2(p_n#) <<_eps p_n^(beta_2+eps), beta_2 <= 4.85 (beta sieve, Friedlander-
   Iwaniec) and < 4.45 (DHR-type, Blight). Proved exponent < 4.5 vs conjectured 2.
3. LOWER TRANSFER: b - a = p_n# collapses paired to ordinary (survivor sets
   equal - verified exactly n = 3,4,5), so j_2(p#) >= j(p#) and FGKMT lower
   bounds transfer verbatim.
4. WHY THE LADDER WAS EMPTY (recorded honestly): Iwaniec's one-residue
   (k log k)^2 is order p^2 at primorials - the SAME order as ZM Conjecture 6;
   a paired Iwaniec-strength bound would land within a constant of a statement
   implying Goldbach + Polignac (ZM Thm 4.1), i.e. it is parity-critical. The
   sub-conjecture rungs are parity-safe and were simply never written down.
   Named wall: the Iwaniec-analogue; named next rung: Brun pure sieve
   (quasi-polynomial, elementary) and any beta_2 improvement (transfers free).
   The bound's price vs the exact table: x6 at p=3 up to x1.3e8 at p=73 -
   Legendre is exponentially lossy, rung 2 is where the honesty lives.

### (b) THE CLEAN-EXTENSION DEATH AT 17 - the exact 9 is now an accounting
### identity (paired-jacobsthal-values.md sec. 4b, new)

THE SHALLOW-EXTENSION CAP: a family maximiser's record window is a maximal gap -
NO interior openings - so lifting to gear q' it can only grow by fusing adjacent
gaps; interiors must sit in the 2-element tooth set mod q', and 3 interiors would
need 3 distinct residues in a 2-set (asserted distinct in every observed case),
so AT MOST TWO adjacent gaps ever fuse and the lift choice grants any single
separation congruence: best extension = F_old + best adjacent 2-gap sum.
EVENTS: all 16 13-winners have the SAME local context (..6,3,6,[75],6,3,6..);
75 = 7 mod 17 so e = +-7 mod 17 fuses both flanks: cap = 6+75+6 = 87, and the
exhaustive extension value set over 272 lifts is exactly {81, 84, 87} =
{75+6, 75+6+3, 6+75+6}. The winner 96 is a 4-5-gap DEEP fusion (interiors
filling both tooth classes) on mediocre bases (F_13 in {42, 51}).
THE 9 = 96 - (6+75+6). THE LADDER: best 19-extension of all 64 17-winners
(1216 lifts, direct sieves) = 111 = 96+(6+9), deficit 18; best 23-extension of
the 19-argmax = 147 = 129+(6+12), deficit 36 (lineage-only caveat: the full set
of 19-winners is unknown). Deficits 9, 18, 36 - DOUBLING; the records' adjacent
2-sums grow by 3 (12, 15, 18). Anatomies: 111-window = [96,6,9], 147-window =
[129,6,12] - one-sided chains exactly at the cap. My round-20 cap guess
(g_L + F + g_R only) was WRONG and the failed assertion found the truth: one-sided
two-gap chains beat both-flank fusion from 19 on. Caveat for any general theorem:
the 3-interior impossibility needs the non-collision conditions (q' not dividing
F_old or the adjacent separations) - the collision case is exactly PADDING.

### (c) N5 - PAIRED HL-B IN CYCLES + THE FULL EIGEN-ANALYSIS
### (docs/novel/paired-hlb-cycles.md, new; the Holt import now fully spent)

1. LOCAL FACTOR IDENTITY (proved, 2 lines; asserted q < 2000): c_q(g) =
   q - nu_q({0, 2, 6g, 6g+2}) - the round-19 autocorrelation law IS the
   Hardy-Littlewood PRIME-QUADRUPLET local factor of (p, p+2, p+6g, p+6g+2).
2. PINCH THEOREM (proved: depth-sum identity + union bound; verified by full
   sieve at machines 13/17/19, every g <= 26): N2(g) - sum_t N3(0,t,g) <=
   n_g(M) <= N2(g), all closed-form CRT products, any scale, no scan. Hence
   fixed-gap population ratios converge AT RATE 1/log^2 y to HL quadruplet
   singular-series ratios (finite products - factors cancel beyond q = 6g+2):
   paired Hardy-Littlewood Conjecture B holds PROVABLY inside the sieve.
   Numerics: n_5/n_4 -> 3.150, pinched to [3.06, 3.22] at y = 10^6.
3. EIGEN-ANALYSIS: aggregated by (sum, length), the paired transfer is
   generically diag(q-2j-2) + superdiag(2j) (sporadic share 6.9% at +17,
   carried exactly by the word-level transfer) and is diagonalised by
   v^(k)_j = (-1)^(k-j) C(k-1,j-1) - q-INDEPENDENT Pascal eigenvectors,
   IDENTICAL to Holt's one-residue system (the q-dependence cancels in the
   eigenvector recursion): the paired system is Holt's with DOUBLED level
   spacing. Verified in exact rationals, q in {17,19,101,997}, k <= 12.
4. WORD-LEVEL TRANSFER: the round-20 recursion upgraded from gap totals to the
   FULL word census - n_w(M+q') exact for all 6714 words (sum <= 24) at
   5005 -> 85085 and 10489 at 85085 -> 1616615 via deterministic per-copy
   image enumeration (no composition explosion). The census-to-census linear
   map is now a verified exact object; paired-holt-recursion.md status upgraded.

### Ranking changes (honest pricing)

- N4: EXECUTED - moves from "candidate" to "delivered, ladder started". Remaining
  life in the lane: Brun rung (elementary, one round if wanted), beta_2 watching
  (free), Iwaniec-analogue priced as parity-critical wall (not reachable with
  published methods - imported corpus limit, stated in the doc).
- N5: EXECUTED AND CLOSED as sized - the Holt import is fully spent (recursion
  r20, diagonalisation + HL-B payoff r21). Follow-on if ever needed: driving-term
  asymptotics with doubled spacing (Holt's own open end, inherited).
- C10+N3 stays TOP; the round-20 open item (the exact 9) is CLOSED; what remains
  there: the per-difference Conjecture 6 refinement as a publishable statement,
  and the new micro-question - does the deficit keep doubling (needs the full
  19-winner set; the 23 row is lineage-only)?
- New publication-worthy additions to section 6's list: the j_2 bounds note
  (first rungs of an empty ladder + why it was empty), and the paired-HL-B note
  (machine local factor = HL quadruplet factor; Pascal diagonalisation).

### Needs from other lanes

- CONSTRUCTOR: the paired transfer matrix is now DIAGONALISED (exact q-independent
  eigenbasis, doubled spacing) - if the anti-correlation deficit is to become a
  spectral statement, this is the basis to write it in; the pinch theorem also
  gives closed-form N2/N3 baselines for every lag with no scan.
- MECHANIC: the pinch bounds predict every fixed-gap population within
  [N2 - sum N3, N2] at 37/41/53 with zero scanning - a free cross-check row for
  COV-SAT outputs.
- FORMALIST: three finite kernel candidates if ever wanted: the local-factor
  identity at fixed q (two-line affine bijection), one rung of the word-level
  transfer, the Pascal eigenvector identity at fixed size. None urgent.
- LATERAL: your c_q(g) is now identified as the HL quadruplet local factor -
  the golden-gap/corridor objects all sit inside a 4-point HL frame.

### Deliverables ledger (round 21)

Scripts: research/j2_bound.py, ext_death.py, ext_death2.py, paired_hlb.py (all
assert-gated, green). Data: research/data/{j2_bound,ext_death,ext_death2,
paired_hlb}.out. Docs: docs/novel/j2-upper-bound.md (new), paired-hlb-cycles.md
(new), paired-jacobsthal-values.md sec. 4b, paired-holt-recursion.md status
upgrade, README index (2 entries). Prior-art checks dated 2026-08-24 in both new
docs. No detached jobs pending at write-up.
