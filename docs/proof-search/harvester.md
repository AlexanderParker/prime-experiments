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

## ROUND 22 - the j_2 ladder gets its middle rung, its ceiling and a per-difference
## refinement; the deficit doubling dies by arithmetic; the pinch is generalised and
## then PARTLY GIVEN BACK to Holt; publication shape re-priced

Scripts (all assertion-gated, all green): research/delta_frame.py, family_scan.py,
family_scan_fast.py, family_scan23.py, ext_deficit19.py, ext_deficit23.py,
zm_seq_reconcile.py, j2_brun.py, j2_perdiff.py, hlb_effective.py,
pinch_bonferroni.py, holt_correspondence.py, ext23_witness.py. Data:
research/data/{ext_deficit19,ext_deficit23,ext23_witness,family_scan23,
zm_seq_reconcile,j2_brun,j2_perdiff,hlb_effective,pinch_bonferroni,
holt_correspondence}.out plus
family_w19_delta.npy, family_w23_delta.npy, ext19_to23.npy, ext23_all.npy. Prior-art
checks run by me and dated 2026-08-24. All jobs launched early and finished before
write-up.

THE ROUND'S MOST IMPORTANT ITEM IS A CORRECTION, so it goes first.

### (0) PRIOR-ART CORRECTION - Holt arXiv:2502.20470 (Feb 2025) contains two of the
### project's claimed-novel identities and explains a third

Round 20/21 searched Holt arXiv:1510.00743 and Holt-Rudd arXiv:1408.6002 and found no
paired counterparts. Round 22's re-search surfaced a paper that DID NOT EXIST at that
time: Fred B. Holt, "Eratosthenes sieve supports the k-tuple conjecture",
arXiv:2502.20470 (v1 Feb 2025, v3 Jul 2025). Text extracted and read. His Corollary 1:

    for an admissible constellation s of length J,
        sum_{j >= J} n_{s,j}(p#)  =  prod_{q <= p} (q - nu_q(s)),
    nu_q(s) = number of distinct residues mod q among the J+1 boundary points,

i.e. the aggregate population of s AND ITS DRIVING TERMS. Since a twin-slot survivor
is EXACTLY a gap of 2 in Holt's cycle of gaps, a pair of twin-slot survivors at lag g
is an instance of his constellation (2, 6g-2, 2) with boundary points
{0, 2, 6g, 6g+2} = H_g. Therefore:

- MY LOCAL-FACTOR IDENTITY c_q(g) = q - nu_q(H_g) (round 21, "the machine's transfer
  diagonal IS the HL prime-quadruplet local factor") is Holt's q - nu_q(s),
  specialised. The affine-bijection proof is still the right proof of the closed form;
  the identification is his framework.
- LATERAL'S DEPTH-SUM IDENTITY sum_j W_j(g) = N2(g) (round 20) IS Holt's Corollary 1
  at that constellation. The identity and the proof are correct; the novelty claim is
  not. FLAGGED TO LATERAL - I have not edited their doc, only recorded the verdict in
  docs/novel/README.md's index entry (which honestly said "not yet checked").
- "THE PAIRED SYSTEM IS HOLT'S WITH DOUBLED LEVEL SPACING" (round 21) is now DERIVED,
  not observed: a paired gap word of length j is a constellation with 2j+2 boundary
  points, and his population dynamics carries diagonal q - (number of points), so
  q - 2j - 2 against his q - (j+1). Better understanding, weaker novelty.
- FORMALIST: your kernel check of the local-factor identity is unaffected as
  verification; only its novelty label changes.

CHECKED, NOT ASSERTED (research/holt_correspondence.py, green): (A) twin-slot
survivors ARE exactly the left endpoints of the gaps of 2 in the rough cycle (sets
equal, 1,485 at P = 30,030 and 22,275 at P = 510,510); (B) N2(g) = prod_q c_q(g)
equals to the unit the count of positions with all four boundary points of
s = (2, 6g-2, 2) rough - Holt's right-hand side - at every g <= 6 and both machines;
(C) the objects separate at once, e.g. machine 17, g = 5: n_g = 4,230 vs Holt's
n_{s,J} = 0.

WHAT SURVIVES, and why it is a different object: Holt's n_{s,J} counts constellation
instances with NO ROUGH NUMBER between the boundary points. The paired sieve's gap
population n_g counts CONSECUTIVE TWIN-SLOT SURVIVORS - no twin candidate between,
ordinary rough numbers allowed. The twin-slot subsequence of Holt's cycle is not
studied in his papers and n_g is none of his n_{s,J}. Everything proved about n_g -
the pinch, its Bonferroni series, the moment identity, the effective threshold - has
no counterpart found. Also checked and clear: Holt arXiv:2603.25915 (Mar 2026,
"Surviving Eratosthenes sieve I") is one-residue, Legendre-conjecture-directed, with
no HL-B-in-cycles and nothing paired. Full correction: docs/novel/paired-hlb-cycles.md
section 0 + section 6; docs/novel/paired-holt-recursion.md CORRECTION.

### (a) THE NEXT j_2 RUNG - delivered, plus a relabelled wall and a per-difference
### refinement (docs/novel/j2-upper-bound.md: THEOREM 3, THE CEILING, COROLLARY, sec 7)

1. THEOREM 3 (Brun pure sieve; elementary, exact constants, no implied constant):
   for every ODD K with R_K < V_n,
       j_2(p_n#) <= E_K/(V_n - R_K) + 1,
   E_K = sum_{j<=K} e_j(omega(p)), R_K = sum_{j>K} e_j(omega(p)/p), e_j the elementary
   symmetric polynomials. It CONTAINS round 21's Theorem 1 as the case K >= n (there
   R_K = 0 and E_K = prod(1+omega(p)) = 2*3^(n-1), asserted at every n), and at the
   optimal K (measured K* = 3,5,7,9,11,13 over p_n = 5..27449, i.e. K* ~ lambda T_n
   with T_n = sum omega(p)/p ~ 2 log log p_n) it is QUASI-POLYNOMIAL:
   exp(C log p_n log log p_n) = p_n^{C log log p_n}, with the measured
   C = log(bound)/(log p log log p) in [3.47, 4.16] for p_n = 173..27449 while
   Theorem 1's own ratio diverges 5.6 -> 139. Crossover: Theorem 3 is strictly better
   from p_n = 13 (not merely asymptotically), by >300x at p_n = 73 (1.082e9 vs 3.316e11).
   The inequality itself is checked directly against brute-force survivor counts on
   1800 real paired windows (n = 3..6, odd K = 1..n+1), tightest ratio 0.54.
   Ladder alignment: Theorem 1 = Kanold slot (2^k), Theorem 3 = Stevens slot
   (2 k^{2+2e log k}), Theorem 2 = Iwaniec slot.

2. BETA_2 MOVED, FOR FREE: the best proved dimension-2 sifting limit is 4.266
   (Diamond-Halberstam-Richert; Franze arXiv:1012.3809 Table 1 via ar5iv, which also
   gives 4.516 for Lambda^2 Lambda^- at kappa = 2 and shows Lambda^2 Lambda^- winning
   only from kappa >= 3). Round 21 cited 4.85 / 4.45. Rung 2 is now
   j_2(p_n#) <<_eps p_n^{4.266+eps} with no new work - the "watch beta_2" the brief
   asked for, and it paid.

3. THE PARITY ASSESSMENT STANDS BUT WAS MISLABELLED - self-caught. Round 21 filed
   "a paired Iwaniec bound is PARITY-CRITICAL, the Iwaniec-analogue is open". Wrong
   slot: Iwaniec's ordinary j(n) << (k log k)^2 is, at primorials, exactly p^{beta_1}
   with beta_1 = 2 the dimension-ONE sifting limit, and round 21's own Theorem 2
   already delivers the dimension-TWO counterpart p^{beta_2}. The corrected, sharper
   wall:
     * our sieve loses nothing on the LEVEL of distribution (|r_d| <= 3^{nu(d)}, so
       sum_{d<D}|r_d| << D log^2 D and D = m^{1-o(1)}), so the exponent is EXACTLY the
       sifting limit and no bilinear / well-factorable refinement of the level helps;
     * Selberg's conjectural optimum is beta_kappa = 2 kappa, i.e. 4 at kappa = 2, and
       no sieve attains 2 kappa for any kappa > 1 (Selberg: beta_kappa <~
       2 kappa + 19/36 for large kappa);
     * ZM Conjecture 6 asks for exponent 2 = beta_1 on a kappa = 2 problem - BELOW
       EVEN THE CONJECTURAL FLOOR by a factor of two in the exponent;
     * and in the project's own horizon frame exponent 2 is precisely the level at
       which a sieve survivor in (y, y^2] IS a prime pair (Reduction A), which is why
       ZM Thm 4.1 extracts Goldbach and Polignac from it.
   So: parity-blocked, not unproved-but-approachable. Remaining moves: any beta_2
   improvement (free) and an explicit constant in rung 2. Lowering the exponent toward
   2 is not a move.

4. PER-DIFFERENCE REFINEMENT - the first upper bound attached to the project's own
   F_d family (research/j2_perdiff.py). The sieve removes omega_p(d) = 2 classes for
   p not dividing d and 1 for p | d, so the sifting DIMENSION is d-dependent:
       kappa_d = 2 - (1/log y) sum_{p | d, p <= y} log p / p     (Mertens),
   running over all of [1,2], and F_d(y) <<_eps y^{beta(kappa_d)+eps}. Both endpoints
   are attained inside the family: kappa = 2 exactly for d coprime to the primorial
   (the class the percentile work identifies as hardest), and kappa = 1 + O(1/log y)
   for d = 0 mod the primorial - exactly the round-21 verified collapse j_2 = j, so
   the interpolation is anchored at both ends. d divisible by exactly the primes in
   (y^theta, y] gives kappa = 1 + theta, verified at three thetas and three scales.
   Honest caveat in the doc: for FIXED d and y -> infinity, kappa_d -> 2, so this is a
   statement about differences that grow with the machine - i.e. the family setting.

### (b) THE DEFICIT-DOUBLING MICRO-QUESTION - ANSWERED, NEGATIVE
### (docs/novel/paired-jacobsthal-values.md section 4c)

THE ENABLING REDUCTION (proved, verified). For 3 not dividing e, F_e(y) depends on e
ONLY through delta = e*3^{-1} mod Q, Q = prod_{5<=q<=y} q, and equals 3*G(delta) with
G the maximal cyclic gap of {k : k != 0, -delta mod q}. (Gear 3 pins survivors to one
class mod 3; n = 3k + c turns each tooth pair into a translate of {0, -delta} by the
single integer -c*3^{-1} mod Q.) Combined with a HELD-OUT-TOP-GEAR PREFILTER that is
exact rather than heuristic - a run of L killed positions forces every survivor of the
smaller gears in the window into {0, -delta} mod qt, hence at most TWO residues among
the offsets, which pins delta mod qt - the y=19 scan that round 17 called "out of
reach" (2,424,922 differences) costs minutes and keeps 64 of 1,616,615 deltas
(0.0040%). Validated against brute force in delta space at y = 13 and y = 17.

1. h_2(19) = 258 REPLICATED by exhaustive family scan (max G = 43 over the whole
   family, nothing above), by a method entirely different from ZM's.
2. The COMPLETE 19-winner set: exactly 64 deltas; ladder 8, 16, 64, 64 at
   y = 11, 13, 17, 19. The 19-winners are not lifts of the 17-winners.
3. The 3 | e branch settled EXHAUSTIVELY (not by observation): for 3 | e a gap of 3G
   needs killed runs in BOTH sub-lattices, each a translate of the same S_delta, so
   its delta must already be a G >= 43 winner; checking those 64 gives best F = 44
   against 129.
4. The deficit ladder recomputed over COMPLETE winner sets by independent code:
   9 (13->17, 16 winners), 18 (17->19, 64 winners), 36 (19->23, 64 winners) -
   round 21's three numbers confirmed and the 36 is no longer lineage-only.
5. THE DOUBLING IS REFUTED - by arithmetic, before any computation. A deficit can
   never exceed the increment F(new) - F(old) because the best extension is at least
   F(old). OEIS A288815 (pulled in full 2026-08-24) gives F = h_2/2 = 75, 96, 129,
   183, 225, 285, 354 at y = 13..37, so the increments are 21, 33, 54, 42, 60, 69: the
   23->29 increment COLLAPSES to 42 < 72. The 9, 18, 36 doubling was a coincidence of
   three consecutive increments. What survives is the accounting identity
   deficit = increment - (best adjacent 2-gap sum of a record), with 2-gap sums
   12, 15, 18 (+3 per rung), which PREDICTED deficit 21 at 23->29.
6. RECONCILED WITH ZM'S OWN EXHAUSTIVE DATA - the round's best cross-check, and a
   novelty downgrade I found myself. ZM's full_details.pdf Table 1 carries a column
   nseq = "number of sequences of maximum length" (1, 6, 1, 1, 4, 2, 2, 14, ... at
   p_n = 5, 7, 11, 13, 17, 19, 23, 29) with exhaustive ancillary lists
   (remainders_2.txt / permutations_2.txt / moduli_2.txt) - a representation the
   project had never looked at. Converting each winning delta's record windows into
   ZM's covering pattern (which gear kills each position):
       y = 11: 8 deltas, 8 windows -> 1 pattern = nseq 1 (self-symmetric)
       y = 13: 16 deltas, 16 windows -> 1 pattern = nseq 1 (self-symmetric)
       y = 17: 64 deltas, 128 windows -> 4 patterns = nseq 4
       y = 19: 64 deltas, 128 windows -> 2 patterns = nseq 2
   EXACT at all four, reverses counted separately exactly as ZM state, and ZM's own
   remark that the single sequences at n = 5, 6 are self-symmetric is reproduced.
   CONSEQUENCE, and it goes further than I first wrote: the winner sets are recoverable
   from ZM's published files, AND the delta reduction is essentially their Proposition
   1.5(2) ("for every prime there exist two non-zero residue classes covering
   {1,...,m}", which already drops the pair (a,b)), AND their algorithm suite reaches
   p_n = 73 where this scan reaches 23. So neither the data nor the reduction nor the
   search method is a contribution. What IS new: the independent replication, the
   exhaustive settlement of the 3 | e branch, and the CROSS-GEAR EXTENSION LADDER -
   a question ZM never ask.
7. THE y=23 RUNG (exhaustive, 1,616,615 prefilter classes, 4 shards): 7. THE y=23 RUNG (exhaustive, 1,616,615 prefilter classes, four shards, ~3 h wall).
   The prefilter keeps 128 of 37,182,145 deltas (0.00034%); all reach G = 61 and none
   exceeds it, so h_2(23) = 366 is REPLICATED EXHAUSTIVELY - the second independent
   replication this round. Complete 23-winner set = 128 deltas; ladder 8, 16, 64, 64,
   128 at y = 11, 13, 17, 19, 23. Pre-registered check 1 PASSED: exactly 2 distinct
   covering patterns = ZM's nseq(23) = 2 (five machines now agree in both
   representations).

8. THE 23 -> 29 DEFICIT IS ZERO, AND ROUND 21'S SECOND CONCLUSION IS ALSO REFUTED.
   My own pre-registered prediction was 21 (increment 42 minus a 2-gap sum continuing
   12, 15, 18 to 21). MEASURED: ZERO. 23-winners lift to the FULL y = 29 family
   maximum G = 75, F = 225, h_2 = 450. CERTIFIED, not merely computed
   (research/ext23_witness.py, independent code path, no sieve array): delta_29 =
   743,911,918 (from delta_23 = 269,018, lift r = 3 mod 29) has k = 134,406,257 ..
   134,406,330 - 74 consecutive positions - each killed by an explicitly listed gear,
   with both flanking positions open on every gear, so G = 75 exactly. Three further
   witnesses, all at r = 3 mod 29. AND IT IS NOT ONE LUCKY WINNER: over the complete
   128 winners x 29 lifts, EVERY one of the 128 reaches G = 75, each at exactly the
   same four lift residues r in {3, 12, 17, 26} mod 29 = {+-3, +-12} - 512
   (winner, lift) pairs, and no other r works for any winner. Those two residue pairs
   are precisely the two interior separations available in the fused word: the openings
   around a record sit at 0, 2, 14, 75, 77, 79, so the 75-gap is either 0 -> 75
   (killing the openings at 2 and 14, separation 12, forcing delta = -+12 mod 29) or
   its mirror 2 -> 77 (killing 4 and 65, separation 61 = 3 mod 29, forcing
   delta = -+3). At this rung the cap law does not merely BOUND the extension - it
   PREDICTS the admissible lifts exactly. Side effect: h_2(29) = 450 now has an independent
   explicit lower-bound witness, so three consecutive ZM values are confirmed here by
   three different routes.
   WHY, AND THE CAP LAW IS CONFIRMED RATHER THAN BROKEN: the 23-machine's gap word
   around a record is [2, 12, 61, 2, 2] in slot units; the lift fuses 61 + 12 + 2 = 75,
   i.e. TWO interior openings killed - exactly the shallow-extension cap law's maximum
   (paired-jacobsthal-values.md 4b), attained. In F units 183 + (36+6) = 225. The
   accounting identity deficit = increment - (record's best adjacent 2-gap sum)
   survives intact; what was wrong was the guess that the 2-sums continue 12, 15, 18,
   21 - they run 12, 15, 18, 42, and at 23->29 the 2-sum EQUALS the increment.
   CORRECTIONS TO ROUND 21, both mine: (i) "the deficit doubles" - the ladder is
   9, 18, 36, 0; (ii) "from 17 on the argmax trajectory is forced to abandon its
   ancestors, a record window is self-limiting, each new gear's winner is a fresh deep
   resonance" - REFUTED at 23->29 by explicit certificate; maximiser persistence is not
   monotone in y, it fails at 17, 19, 23 and returns at 29. The mechanism (cap law) was
   right; the extrapolation from three points was not.
   HONEST LIMIT: one more rung, not a law. The 2-gap sum beside a record is an
   arithmetic accident of that neighbourhood, and the 29-winner set would be a
   1.08e9-delta scan - out of reach for this prefilter (y=23 already cost ~3 CPU-hours
   x 4 shards).

### (c) HL-B CONSEQUENCES - the pinch generalised, made effective, and bounded
### (docs/novel/paired-hlb-cycles.md sections 3a, 3b)

THE PINCH IS BONFERRONI ORDER 1 - my own round-21 theorem generalised
(research/pinch_bonferroni.py). With S_k = sum over 0 < t_1 < ... < t_k < g of
N_{k+2}(0,t_1,...,t_k,g) (closed-form CRT products), inclusion-exclusion over which
interior offsets are open gives EXACTLY
    n_g = sum_{k>=0} (-1)^k S_k,
with Bonferroni truncations alternating (even K upper, odd K lower). K = 0 and K = 1
ARE the two sides of the round-21 pinch. Moment form: since a depth-j window has j-1
interior openings, S_k = sum_j C(j-1, k) W_j(g), so S_0 = N2 is the depth-sum identity
and S_1 overcounts sum_{j>=2} W_j by exactly sum_{j>=3} (j-2) W_j - THE PINCH'S SLACK
IS AN EXPLICIT QUANTITY. Identities and alternation verified by full sieve at machines
13 and 17, g = 4,5,6,8,10, k <= 3.

POSITIVE: EFFECTIVE POLIGNAC IN THE PAIRED SIEVE. Let y_0(g) be the least y at which
the lower bound is positive; then gap g occurs in M_y for EVERY y >= y_0(g),
unconditionally, no scan. Holt proves constellations "arise and persist" but gives no
stage index; this is a number. Splitting at q = 6g+2 (beyond which every local ratio
is generic) gives one monotone table:
    g            2   3   4   5   6   8  10   12   15   20    25    30   40     50
    y_0 order 1 14  20  26  32  38  50  62  103  199  467  1009  2609  12157  42257
    y_0 order 3  -   -   -   -  41  53  67   79   97  127   167   367
with log y_0(g)/sqrt(g) confined to [1.305, 1.531] at order 1 and about 1.08 at
order 3: THE THRESHOLD IS y_0(g) = exp(Theta(sqrt g)), NOT polynomial in g, and the
higher Bonferroni orders improve the CONSTANT but not the SHAPE - so the square root
is not a union-bound artifact and a polynomial threshold would need a different
argument. That closes, negatively, the open item I was about to name.

NEGATIVE, PRICED SO NOBODY RE-DERIVES IT: every gap <= G(y) occurring gives
F(2,y) >= 3 G(y) ~ c (log y)^2 - 60, 90, 180, 240 at y = 10^3..10^6 against a truth of
order y^2. The pinch contributes NOTHING to the j_2 lower ladder; that stays with the
FGKMT transfer.

THE BOUNDARY, QUANTIFIED. The pinch is a FULL-PERIOD statement; primality of survivors
lives in the window (y, y^2] (horizon theorem), a share y^2/P_y = exp(-(1+o(1)) y) of
the period: 2.2e-4 at y = 19, 1.1e-9 at y = 37, 2.6e-34 at y = 101. No full-period
population statement, however exact, localises into a share that thin. That is the
entire distance between "paired HL-B in cycles, proved with rate" and "paired HL-B for
primes, open". This document proves NOTHING about prime quadruplets and no
unconditional prime-side consequence was found. Outside the sieve the eigen-analysis
buys exactly one structural fact, and after item (0) it is a consequence of Holt's
point count rather than a discovery: the paired system relaxes at rate (log y)^{-2}
because its constellations carry twice as many points.

### (d) PUBLICATION SHAPE - honest pricing of my own holdings, AFTER item (0)

Item (0) re-orders this. Two units are real; one is a separate-venue unit; the rest are
sections. Nothing was written - this is pricing.

UNIT 1 (now the strongest): "The paired Jacobsthal function: first upper bounds, and
the structure of its maximisers" = j2-upper-bound.md (Theorems 1-3, the per-difference
corollary, THE CEILING) + twin-percentile.md + paired-jacobsthal-values.md 4a/4b/4c.
First bounds of any strength on a function named and conjectured about since 2017,
aligned rung-for-rung with the ordinary ladder, plus the structural remark that ZM
Conjecture 6 asks for a dimension-1-quality exponent on a dimension-2 problem - which
is the paper's most interesting paragraph and is not in ZM. The computational half
supplies the percentile result and the shallow-extension cap law. TO ADD: an EXPLICIT
constant in rung 2 (turn <<_eps into a stated inequality with a stated n_0 - the one
piece of real work and the obvious referee ask); a careful statement of the sieve
dimension and remainder bound; and honest positioning of the computational half as
replication-plus-structure given ZM's ancillary files (item b6).

UNIT 2 (DOWNGRADED THIS ROUND from "strongest" to "a short note"): the twin-slot gap
population, = paired-hlb-cycles.md after the section-0 correction. Its two headline
identities are Holt's; what is left is one object (n_g), one theorem about it (the
exact Bonferroni series with the moment identity, of which the pinch is orders 0-1),
and one effective corollary (y_0(g) = exp(Theta(sqrt g))). That is a legitimate short
note extending Holt's program to the twin-candidate subsequence, and it would cite him
on nearly every page. TO ADD: uniform error terms; a decision on whether the effective
threshold can be improved beyond the constant. Note also that Holt's own programme
appears to live on arXiv and primegaps.info rather than in journals (1510.00743 was a
conference presentation), which affects where such a note would go.

UNIT 3 (separate venue, self-contained): the Lean development - machine-checked
per-difference equivalences for Polignac, the Goldbach window reduction with its exact
converse, the mod-3 dichotomy, the universal cap <= 12. Formalization venues take work
containing no new mathematics; needs packaging, not research.

NOT PUBLISHABLE ALONE, and now priced that way: (i) twin-percentile - data, no
theorem, belongs inside unit 1; (ii) the h_2 replication AND the delta-reduction /
prefilter method - struck entirely: the reduction is ZM Proposition 1.5(2), their
algorithms reach p_n = 73 against this scan's 23, and the winner data is in their
ancillary files (a three-part downgrade of round 21's framing, all of it self-found
this round); (iii) the cap law and the deficit ladder - a good section of unit 1. The
cap law came out of this round STRENGTHENED (at 23->29 it predicts the admissible
lifts exactly, not just an upper bound), but it still holds only under observed
non-collision conditions and the deficit ladder is four points, one of which (0)
falsified the extrapolation drawn from the other three.

WHAT IS NOT MINE TO PUBLISH: everything on the twin route (other lanes), and the
kernel work on my identities.

### Ranking changes (honest pricing)

- N4 (the j_2 ladder) UPGRADED from "two rungs" to "three rungs + a ceiling + a
  per-difference refinement", and its named wall RE-LABELLED (a3). Remaining moves are
  cheap and bounded: beta_2 watching (free) and an explicit rung-2 constant. This is
  now the lane's strongest holding.
- N5 (paired HL-B) DOWNGRADED by my own prior-art find (item 0), then partly restored
  by the Bonferroni generalisation and the effective threshold. Net: a short note, not
  a paper.
- C10+N3 (the paired-Jacobsthal family) DOWNGRADED on the data side (b6) and held on
  the structure side (cap law, extension ladder, percentile). The round-21 open
  micro-question is CLOSED, negative.
- STANDING LESSON, recorded because it cost novelty twice in two rounds: prior-art
  checks EXPIRE. Both this round's downgrades came from documents that existed but had
  not been looked at (ZM's ancillary files, 2017) or did not exist at the last sweep
  (Holt, Feb 2025). Any claim of novelty older than a round should be re-searched
  before it is repeated in a summary.

### Needs from other lanes

- LATERAL: docs/novel/depth-sum-identity.md - your identity is Holt arXiv:2502.20470
  Corollary 1 at the constellation (2, 6g-2, 2). Proof and value unaffected; the
  novelty label is. I recorded the verdict in the README index and did not edit your
  doc.
- FORMALIST: thanks for the kernel check of the local-factor identity - unaffected as
  verification. Two further finite candidates if ever wanted: the delta reduction at a
  fixed machine, and the Bonferroni step of Theorem 3 at fixed n and K.
- MECHANIC / CONSTRUCTOR: nothing blocking. The pinch's closed-form population windows
  remain a free cross-check row at any machine, now at any Bonferroni order.

## ROUND 23 - UNIT 1 TAKEN TO PUBLICATION READINESS: RUNG 2 MADE FULLY EXPLICIT
## (exponent 19, no ineffective threshold), rung 1.5 given its proved constant, the
## nested-truncation validity obstruction SOLVED, a referee pass that found five
## defects (all mine), and the ceiling reframed after my own citations failed audit

Scripts (all assertion-gated, all green, all run from the repo root):
research/j2_explicit.py (THEOREM 3E + the rung-2 level/error costing),
research/j2_fi77.py (THEOREM 2E via Friedlander-Iwaniec Opera de Cribro Thm 7.7,
plus an audit of my own arithmetic), research/j2_nested.py (the correct nested
truncation, tested), research/j2_referee.py (the referee pass; caches
research/data/ref_fam_<y>.npy), research/j2_lower.py (the lower ladder). Re-ran
round 21/22's j2_bound.py, j2_brun.py, j2_perdiff.py - all still green. Data:
research/data/{j2_explicit,j2_fi77,j2_nested,j2_referee,j2_lower}.out plus
ref_fam_{3,5,7,11,13,17}.npy. Prior-art and citation checks run BY ME and dated
2026-08-25. All jobs finished before write-up; nothing pending.

TWO OF MY OWN CONCLUSIONS WERE OVERTURNED INSIDE THIS ROUND - "rung 2 cannot be
made explicit" and "a valid nested truncation is the missing construct" - both
after leads were verified against actual text. Both corrections are in item (a).

### (a) THE BRIEF'S NAMED TARGET - the explicit constant in rung 2

SPLIT VERDICT, and the honest half is the more useful one.

WHAT WAS DELIVERED (THEOREM 3E, docs/novel/j2-upper-bound.md). The bound that CAN
be made fully explicit is rung 1.5, and it now is:

    j_2(p_n#)  <  p_n^{9.30 log log p_n}   for every n >= 3,

with the ASYMPTOTIC constant of Theorem 3 identified exactly as

    C_infinity = 2 lambda_* = 7.182242...,  lambda_* = 3.591121... the root of
                                            lambda (log lambda - 1) = 1.

The proof is four explicit ingredients - Rosser-Schoenfeld (3.20) for T_n, RS
(3.27) plus the twin-constant factorisation for V_n, e_j(x) <= (sum x)^j/j! for the
Bonferroni tail, and RS (3.6) with C(n-1,K) <= (e(n-1)/K)^K for the remainder cost -
plus exact rational verification for 5 <= p_n <= 139 and an analytic tail for
p_n >= 142. TWO THINGS FELL OUT THAT MATTER MORE THAN THE THEOREM:
  * ROUND 22'S "MEASURED CONSTANT IN [3.47, 4.16]" DOES NOT CONTAIN THE LIMIT. The
    ratio log(bound)/(log p_n log log p_n) rises to 7.1822; the shortfall at
    accessible sizes is the factor (1 - (log log p_n + log K)/log p_n), still only
    0.70 at p_n = 27449. Quoting a measured band where a constant belongs would
    have been the referee's first question and the answer would have been wrong.
  * MAKING K EXPLICIT IS FREE. The explicit rule (least odd K with R_K <= V_n/2)
    picks the SAME K as round 22's numerical optimisation at every n tested, ratio
    1.000x. So nothing was given away to get a stated constant.

RUNG 2 MADE EXPLICIT - THEOREM 2E. The first pass of this round concluded rung 2
could not be made explicit; that conclusion was WRONG and is corrected here, in
the same round, after a collaborator's lead was verified against actual text.

    j_2(p_n#)  <=  1.0963 x 10^10 * p_n^19 * (log p_n)^10  +  1   (p_n >= 285)

with every constant stated and NO ineffective threshold; more generally
j_2(p_n#) << p_n^s for every real s > 18.308. Verification: research/j2_fi77.py.

THE INGREDIENT is not a fundamental lemma at all - that was the first pass's
mistake. It is the EXPLICIT, CONSTANT-FREE Selberg Lambda^- Lambda^2 sieve,
FRIEDLANDER-IWANIEC OPERA DE CRIBRO THEOREM 7.7:
    S(A,z) >= X V(z){1 - ((s+3)/(2 e^k))(2 e k/(s-3))^{(s-3)/2}} - 2 R_4(A,D),
    s = log D/log z,  k = kappa + log K,  s >= 2k+3,
    R_4(A,D) = sum_{d | P(z), d < D} tau_4(d)|r_d|,
under prod_{w<=p<z}(1-g(p))^{-1} <= K (log z/log w)^kappa.
WHY IT APPLIES TO US WITH NO WORK: Dudek & Dunn (arXiv:2602.22720, Feb 2026,
"An Explicit Result for the Sum of Two Almost Primes") prove as their Lemma 2.1
that this hypothesis holds with kappa = 2, K = 3 for the multiplicative g with
g(2) = 1/2 and g(p) = 2/p - WHICH IS LITERALLY OUR omega(p)/p. Not a coincidence:
they sift n and N - n simultaneously, i.e. the Goldbach side of ZM Theorem 4.1,
the same two-classes-per-prime problem. METHOD NOTE WORTH KEEPING FOR THE LANE:
the explicit-Goldbach literature is the natural source of explicit tools for the
paired Jacobsthal ladder.
INDEPENDENTLY RE-DERIVED HERE rather than taken on trust: K = 3 is exact and BEST
POSSIBLE (grid search over all (w,z) with w,z < 2e5 returns exactly 3.000000; the
supremum sits at w = 3, z -> 3+, where the product is (1-2/3)^{-1} = 3 and
(log z/log w)^2 -> 1). Also re-derived: k = 3.098612; FI's s >= 2k+3 = 9.1972 is
NECESSARY BUT NOT SUFFICIENT - the bracket only turns positive at s* = 18.30802,
and equals 0.2507/0.5199/0.8202 at s = 19/20/22. And the pre-sieved K values,
which reproduce the collaborator's numbers to three decimals from my own code:
K = 5/3 for p >= 5 (s* = 16.136), 1.4 for p >= 7 (15.474), 1.2624 for p >= 11
(15.077), 1.0479 for p >= 101 (14.353).
THE ARITHMETIC: |r_d| <= omega(d) <= 2^nu(d) and tau_4(d) = 4^nu(d) on squarefree
d, so R_4 <= sum_{d<D} 8^nu(d) <= D prod_{p<D}(1+8/p) <= C_8 D (log D)^8 with
C_8 = e^{8 gamma} prod_{p<10^6}(1+8/p)(1-1/p)^8 = 0.0316 (that product is
DECREASING, so evaluating at 10^6 is a valid UPPER bound for every D >= 10^6 -
using the limit would have been unsafe, and I nearly did). Positivity then needs
m > (2/bracket) C_8 z^s (s log z)^8 / V(z) with V(z) >= 0.3905/(log z)^2.

SOURCE STATUS, stated because round 22 was burned by exactly this: Opera de Cribro
itself was NOT consulted directly. Theorem 7.7 is taken from two independent
verbatim transcriptions that agree exactly - Dudek-Dunn Theorem 1.3 and Campbell
Theorem 2.1 (arXiv:2608.09488, Aug 2026) - both read in full text 2026-08-25. The
book must be checked before publication. Also recorded: DO NOT use Yamada
arXiv:1511.03409 Theorem 3.1 as an alternative explicit sieve (unproved as stated).

WHAT REMAINS AT 4.266, AND WHY IT IS NOT BOOKKEEPING: beta_2 = 4.266 is the
numerically-solved output of the DHR differential-delay system, and the sieve
inequality at that dimension carries an uncomputed O((loglog y)^2 (log y)^{-1/6})
error; even computed, the 1/6 means s = beta_2 + 0.01 would need log y ~ 10^12.
There is no explicit-constant sieve AT its sifting limit for any kappa > 1. So the
note now carries TWO polynomial rungs and says which is which: exponent 19 fully
explicit, exponent 4.266 not explicit and not makeable so.

AND THE VALIDITY OBSTRUCTION I NAMED IS ALSO SOLVED (research/j2_nested.py). The
first pass showed the PER-BAND product truncation {d : nu(d_j) <= K_j for all j}
is not a valid lower-bound sieve (36 explicit counterexamples). The correct object
counts the WHOLE UPPER TAIL - nu(d restricted to primes above z^{alpha_j}) <= H_j,
nested constraints, H_j = 2h_j+1 for the lower sieve and 2h_j+2 for the upper -
which is the refinement Tenenbaum describes in the paragraph before his
fundamental lemma (GSM 163 p.70, set as Exercise 86, proved nowhere there).
TESTED, NOT ASSUMED: 168,400 (depth pattern, bad-count) configurations over 1, 2
and 3 partition points, ZERO violations of Lambda^- <= [survives] <= Lambda^+,
against 36 failures for the per-band form. A PRE-REGISTERED GUESS OF MINE WAS
REFUTED IN THE SAME SCRIPT: I expected monotone depths h_j to be needed for
validity and wrote it into the script before running the control - 0 violations
over all 271 non-monotone patterns. Monotonicity is a LEVEL-COST convenience, not
a validity requirement.
SO THE REMAINING GAP IS EXACTLY ONE THING: an explicit MAIN-TERM estimate for the
nested truncation (an explicit lower bound on sum_{d in D^-} mu(d) g(d) against
V(z)). My own level/error accounting for that design says exponent ~9 is
reachable - theta = 1/2, geometric depths ceil(4 x 1.05^{j-1}), s = 9.07 at
truncation cost 0.36 - and a lead (Halberstam-Richert's own Memoire, Mem. S.M.F.
25 (1971) 97-106, level exponent -> 7.972) is recorded as UNVERIFIED because I
could not obtain the text. That is the named construct for the next round.

A CITATION-NUMBERING CHIMERA CAUGHT AND KILLED, and it had reached two of my
documents. "IWANIEC-KOWALSKI THEOREM 6.9" DOES NOT EXIST. IK Chapter 6
("Elementary Sieve Methods") stops at Theorem 6.7; in IK, 6.9 and 6.10 are
EQUATION labels, and the 6.9/6.10 numbering belongs to OPERA DE CRIBRO - the two
were conflated. IK's "s >= 9 kappa + 1 with K^10" result is IK THEOREM 6.1 /
COROLLARY 6.2 (p.158), and IK's FUNDAMENTAL LEMMA 6.3 has no lower bound on s but
hides its K-dependence inside an O to the tenth power, so it is not explicit
either. Fixed in j2-upper-bound.md sections 6a(8) and 8, and in my agents-shared
block. TWO MORE NUMBERING TRAPS RECORDED so nobody walks into them: Tenenbaum's
fundamental lemma is THEOREM 4.4 (Theorem 3 in the 1995 CUP edition), not 4.3,
and "Theorem I.4.2" does not exist (I.4.2 is a COROLLARY, the Bonferroni
inequality); and Nathanson Chapter 6 is a dead end (no general-dimension sieve at
all). CLEAN AS WE HAD IT: "Friedlander-Iwaniec Opera de Cribro Thm 6.9" is a real
fundamental lemma and our two uses of that phrase stand. THIS IS THE SECOND
NUMBERING ERROR IN TWO EXCHANGES, both in results about to be leaned on, so the
referee pass now carries a CITATION-NUMBERING SWEEP as a standing step.

AND A SECOND EXPLICIT THEOREM I HAD MISSED, priced rather than adopted. Opera de
Cribro carries THREE constant-free results, and the choice costs ten in the
exponent. All three thresholds re-derived by me from the stated inequalities
(research/j2_fi77.py section F5) and asserted:
    ODC Thm 6.9 (D >= z^{9kappa+1}):  positive iff s > 9 kappa + 10 log K
    ODC Cor 6.10 (only D >= z >= 2, NO hypothesis on s):
                                      s > 9 kappa + log(4(9kappa+1)^kappa K^11)
    ODC Thm 7.7:                      the bracket used for Theorem 2E
                        K = 3        K = 1.097 (pre-sieved at 3)
    ODC Thm 6.9      s > 28.986      s > 18.926
    ODC Cor 6.10     s > 37.360      s > 26.294
    ODC Thm 7.7      s > 18.308      s > 14.532
THEOREM 7.7 STANDS - K^10 is brutal at K = 3 (10 log 3 = 10.99 on its own).
Thm 6.9 is a cleaner-looking fallback; Cor 6.10's value is assuming nothing about
s. Every figure above reproduces the collaborator's to three decimals from my own
code, which is why I am willing to carry them.

A NEW ANALYTIC FACT FOR THE CEILING, and it is the best thing in this exchange.
In ODC's beta-sieve (Thms 11.12/11.13, whose F, f, beta, A, B are ALL pinned
exactly by (11.55)-(11.63) - the only unevaluated object is an O((log D)^{-1/6})),
THE LOWER-BOUND CONSTANT B IS ZERO WHENEVER kappa >= 1/2. Our kappa is 2. So the
beta-sieve's lower bound is not merely weak at the sifting limit for us, it is
IDENTICALLY ZERO. Set beside my own arithmetic finding (ZM Conjecture 6 asks
exponent 2 on a kappa = 2 problem, below even Selberg's conjectural floor
2 kappa = 4), the two say from the analytic and the arithmetic side at once why
the natural tool cannot reach the natural target. Added to THE CEILING.

MY OWN ARITHMETIC AUDITED after a warning that the per-range factor might be
inverted: it is NOT. j2_explicit.py divides the Bonferroni tail by V_j, an
AMPLIFICATION by (1/theta)^kappa = 4, which is the correct orientation; closed
forms T_j = 2 log(1/theta) and V_j = theta^kappa confirmed against an empirical
Mertens product over (10^3, 10^6] (j2_fi77.py section F4). Also checked and clean:
no document of mine ever cited "Tenenbaum I.4.3" for the fundamental lemma (that
number is a different theorem, about Phi(x,y)); the miscitation existed only in a
working note and never entered Unit 1.

### (b) THE REFEREE PASS - five defects, all in my own documents

research/j2_referee.py recomputes every recomputable numerical claim of Unit 1 by
independent code (the per-difference family arrays are rebuilt from scratch at
y = 3..17 and then compared ELEMENTWISE against round 20's f13/f17 arrays -
identical). Everything that was meant to reproduce, reproduced: the h_2 table and
#diffs, the margin column, all four tie-aware percentile rows, the 31-class
F_max/lambda spread 2.88..7.52, the delta-profile law at 100% precision AND recall,
the 13->17 cap law (272 lifts, extension multiset {81:208, 84:32, 87:32}, best 87,
THE EXACT 9), the b-a = p# collapse, Theorem 1's explicit chain, and the y=19
winner set reaching G = 43. Five defects found:

1. THE y = 3 ROW WAS WRONG, AND THE CORRECTION IS SHARPER THAN THE ERROR.
   paired-jacobsthal-values.md tabulated "y = 3, h_2 = 0, Conj.6 holds". That 0 is a
   code artefact: research/jacobsthal_family.py returns 0 whenever a period carries
   fewer than two survivors, and at gears {3}, e = 1 the survivor set mod 3 is the
   single class {1}, whose CYCLIC gap is 3. The truth is h_2 = 6 = p^2 - p exactly,
   confirmed by A288815. So Conjecture 6 fails BY EQUALITY at n = 2 (and at n = 1:
   h_2(2) = 2 = 2^2 - 2), which means ZM's "n >= 3" hypothesis is SHARP rather than
   conservative - a fact worth a sentence in the paper and one the project had
   inverted.
2. THE MAXIMISER LISTS WERE TRUNCATED ARGMAX SLICES presented as complete. The true
   counts are 8, 16, 64 at y = 11, 13, 17 (matching round 22's delta-space ladder);
   the doc printed the first 5 and first 6. Complete lists now in the doc.
3. "WORST RATIO 0.858 AT n = 3" (round 21, Theorem 1's explicit chain) omits the
   "+1" that is part of the bound. With it the ratio is 0.8627.
4. "V_n >= 0.3908/(log p_n)^2 for p_n >= 285" DOES NOT FOLLOW from the stated
   ingredients: 2 e^{-2 gamma} C_2 (1 - 1/log^2 285)^2 = 0.390569 < 0.3908. The safe
   constant is 0.3905; Theorem 1's conclusion is unaffected. (The INEQUALITY is
   nonetheless true where checked - exact V_n log^2 p_n >= 0.4048 over
   285 <= p_n <= 2731 - only its derivation was one digit short. Recorded both ways.)
5. The quasi-polynomial constant, item (a).

### (c) THE CITATION AUDIT - five second-hand errors, and a reframed ceiling

Every source behind rung 2 and THE CEILING was read in full text (arXiv PDFs,
ar5iv, published theses, archive.org OCR) rather than at second hand. Round 22's
sieve-theory paragraph, which I called "the paper's most interesting", contained
the errors:

* "NO SIEVE ATTAINS 2 kappa FOR ANY kappa > 1" was written as an impossibility
  theorem. IT IS AN OPEN PROBLEM (Brady, Stanford thesis 2017: "it is currently not
  known whether there is any kappa > 1 with beta_kappa < 2 kappa"), and it is FALSE
  as a blanket statement - Rosser-Iwaniec beats 2 kappa for 1/2 < kappa < 1.
* THE PROVED FLOOR IS beta_kappa >= (1 + o(1)) 2 kappa/e (Brady, improving Selberg's
  own by a factor 2), i.e. about 1.47 at kappa = 2. So ZM's exponent 2 is NOT proved
  to sit below the sifting limit - only below the CONJECTURED one (Selberg's
  2 kappa = 4), and Brady even conjectures 2 kappa is itself beatable.
* WHAT SURVIVES, and it is the honest form: the block at exponent 2 is PARITY, not
  an arithmetic fact about beta_2. Exponent 2 is exactly the level at which a
  survivor in (y, y^2] IS a prime pair, so a dimension-2 lower-bound sieve at that
  level would manufacture two simultaneous primes - what Selberg's parity example
  forbids. The sifting-limit numbers (4.266 proved, 4 conjectured, ~1.47 the proved
  floor) now do the job they can actually do: they calibrate the distance, and they
  leave "is beta_2 < 4?" open as a genuinely separate question.
* Four more, all fixed in the doc's new section 6a: the author is C. S. (CRAIG)
  FRANZE, not "M. Franze" (JNT 131 (2011) 1962-1982); Selberg's conjecture is NOT in
  Franze (the word "conjecture" does not occur there - the source is Selberg's
  Lectures on Sieves sec. 14, restated in Blight's Rutgers thesis sec. 2.1); Franze
  says 2 kappa + 19/36 where Ford (2023) and Brady (2017) both give 2 kappa + 0.4454
  from the same Selberg equation (14.40) - a genuine conflict, now flagged rather
  than picked; Iwaniec's theorem is h(k) << (k log k)^2 with k = omega(n),
  equivalently J(P(z)) << z^2, and the "(log n)^2" phrasing is a weaker corollary;
  and Costello-Watts' 2 e^gamma k^{5+5 log log k} rung is arXiv:1306.1064, not
  1208.5342 (which is a range-restricted computational bound, 50 <= k <= 10000).
* CONFIRMED AS QUOTED: Franze's Table 1 at kappa = 2 is DHR 4.266 / Lambda^2Lambda^-
  4.516, verbatim in three independent renderings, with Blight's 4.266450 at full
  precision and Ford's 4.2665; "Lambda^2Lambda^- wins only from kappa >= 3" is
  Franze's own sentence. beta_2 = 4.266 stands as the best value at kappa = 2
  (Blight's kappa = 2 figure, 4.45, is worse; her improvement bites at kappa = 3).

### (d) NOVELTY RE-CHECKED TODAY, BY CITATION GRAPH RATHER THAN KEYWORDS

The standing lesson is mine and it was applied to my own strongest holding. Method
upgraded because keyword sweeps are what missed Holt in round 22:

* Semantic Scholar: arXiv:1706.00317 has EXACTLY ONE citation in nine years - ZM's
  own companion note. arXiv:1706.03668 has ZERO.
* zbMATH Open: "paired Jacobsthal" returns NO RECORDS AT ALL.
* OpenAlex full text: only the two ZM records.
* OEIS A288815, pulled again (record stamp #19 Apr 12 2026): 21 terms, two links
  (both ZM), comment states only the conjecture. No proved bound deposited.
* arXiv API metadata sweep over the COMPLETE math.NT Jacobsthal set (54 records) and
  all-category listings: every 2025-2026 Jacobsthal item concerns Jacobsthal
  NUMBERS, SUMS, POLYNOMIALS or CONGRUENCES - a different Jacobsthal. None touches
  the Jacobsthal FUNCTION.
* HOLT arXiv:2502.20470, re-examined for THIS document specifically: full text
  downloaded, "Jacobsthal" occurs ZERO times. The round-22 downgrade was real but it
  touches Unit 2 (paired-hlb-cycles.md), NOT Unit 1. Recorded so the next round does
  not over-correct.
* The ORDINARY ladder's frontier re-verified against the live Erdos-problems
  database (problems 970 and 687, fetched 2026-08-25): IWANIEC 1978 IS STILL THE
  RECORD IN AUGUST 2026 - FGKMT 2018 improved only the lower bound, Costello-Watts
  only explicit constants.
VERDICT: NOVEL, re-confirmed 2026-08-25 with stronger evidence than any previous
sweep.

### (e) THE OTHER END OF THE LADDER - a new section and a named open problem

A referee asked to price the two-sided sandwich would find the note silent, so
research/j2_lower.py prices it (A048670 recomputed from scratch for p_n <= 19):

    proved lower   j(p_n#) = p_n^{1+o(1)}     measured exponent 1.10-1.22, p_n<=73
    TRUTH          h_2 ~ (p_n^2 - p_n)/2      measured exponent 1.75-1.95
    proved upper   p_n^{4.266+eps}

THE LOWER LADDER IS EMPTIER THAN THE UPPER ONE WAS: short by a factor p^{1-o(1)}.
NAMED OPEN PROBLEM added to the doc: prove h_2(p_n#) >> p_n^{1+delta} for some
delta > 0. It is a CONSTRUCTION problem, not a sieve-bound problem - exhibit two
residue classes per odd prime p <= p_n covering an interval of length >> p_n^2 - and
nothing in the parity barrier obstructs a construction. Nobody has stated it.
AND THE ONE-LINE REASON THE PAIRED PROBLEM IS QUADRATIC: by CRT the killed residues
mod p are {-a, -a-2e} with a and e independently free, so j_2(p_n#) - 1 is exactly
the longest interval coverable by TWO ARBITRARY classes per odd prime. The covering
CAPACITY sum_{p<=z} omega(p)/p is 1.34 / 1.46 / 1.76 (ordinary) against
2.19 / 2.41 / 3.01 (paired) at z = 13 / 19 / 73: the ordinary covering is
COUNTING-CONSTRAINED at every size where exact values exist and its answer is
near-linear; the paired one is not counting-constrained at all and nothing
elementary caps it below z^2. That is why h_2/j runs 3.0 -> 13.8 over p_n = 5..73,
and why exponent 2 is plausible as the TRUTH while being far out of reach as a
theorem.

### (f) WHAT THE PAPER DOES NOT CLAIM - now a numbered section of the doc

New section 4a of j2-upper-bound.md, six items, written for the referee: no progress
on Conjecture 6; no new sieve theory (the contribution is that the ladder was empty,
not that the rungs are hard); rung 2 is NOT fully explicit and the best bound with
all constants is quasi-polynomial; the computational half is replication plus
structure given ZM's ancillary files; no lower bound beyond the collapse; and
nothing about primes - every statement is about coverings of an interval.

### Ranking changes (honest pricing)

- UNIT 1 IS PUBLICATION-READY. The brief's named blocker is GONE: there is now a
  fully explicit polynomial bound, j_2(p_n#) <= 1.0963e10 p_n^19 (log p_n)^10 + 1
  for p_n >= 285, with no ineffective threshold. The headline is precise rather
  than weaker: "first proved upper bounds on j_2 - an explicit quasi-polynomial
  rung, an explicit polynomial rung at exponent 19, and the best-exponent rung
  4.266 by citation, with an honest statement of which constants exist."
- N4 (the j_2 ladder) holds as the lane's strongest holding, with TWO rungs
  upgraded this round (1.5 -> 1.5E and 2 -> 2E, both explicit) and the 4.266 rung
  honestly capped as not-explicit-and-not-makeable-so.
- THE REMAINING MATHEMATICS IS NOW ONE NAMED OBJECT, not two: an explicit MAIN-TERM
  estimate for the nested Brun truncation (validity is settled). Target exponent ~8;
  the Halberstam-Richert Memoire lead (7.972) is UNVERIFIED and should be obtained
  first - it may make the whole item a citation rather than a derivation.
- NEW, RANKED SECOND IN THIS LANE: the LOWER-bound problem (item e). It is the only
  open question I have found in this area that is not parity-blocked - it asks for a
  construction - and it is completely unattacked.
- The referee pass is now a standing artefact: research/j2_referee.py should be
  re-run before any future claim about Unit 1, and it caches its family arrays so
  the cost is seconds after the first run.
- STANDING LESSON EXTENDED, TWICE. Round 22's lesson was "prior-art checks expire".
  Round 23 adds: (i) SECOND-HAND CITATIONS EXPIRE FASTER - five of the sieve-theory
  facts in my own strongest paragraph were wrong or misattributed, and every one
  came from a summary rather than a source; (ii) "NOT AVAILABLE IN THE LITERATURE"
  EXPIRES TOO, AND FASTEST OF ALL. I concluded mid-round that no explicit
  dimension-2 lower-bound sieve existed. That was true of every FUNDAMENTAL LEMMA I
  checked and false of the problem: the tool was an explicit Selberg sieve, and it
  became citable in 2026 because two papers on the almost-prime GOLDBACH problem
  needed it for exactly our density function. When a search for a tool fails, search
  the neighbouring PROBLEM, not more variants of the tool's name.

### Needs from other lanes

- None blocking. FORMALIST, if ever wanted, three finite decidable candidates:
  Theorem 3E's finite half (at a fixed n, E_K/(V_n - R_K) + 1 against
  p_n^{9.3 log log p_n} is a statement about explicit rationals); the invalidity of
  the PER-BAND product truncation (36 explicit small-integer witnesses); and the
  VALIDITY of the nested upper-tail truncation at a fixed depth pattern
  (Lambda^-(m) <= [all m_i = 0] <= Lambda^+(m), a finite alternating binomial sum -
  research/j2_nested.py enumerates the instances).
- MECHANIC / CONSTRUCTOR / LATERAL: nothing this round.
- TO WHOEVER RELAYED THE FI 7.7 LEAD: verified and adopted, and it turned the
  round's named blocker into a theorem. Every number in it was independently
  re-derived here and matched (K = 3 and the pre-sieved 1.6667 / 1.4000 / 1.2624 /
  1.0479, s* = 18.308 / 16.136 / 15.474 / 15.077 / 14.353). Two of the flagged
  items came back clean rather than confirmed: our per-range factor is NOT inverted,
  and no document of ours ever carried the Tenenbaum 4.3 miscitation.
