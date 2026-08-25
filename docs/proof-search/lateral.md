# lateral workstream log (compacted)

Compacted 2026-08-23; full verbatim rounds 1-19 log at
archive/lateral-full-r1-19.md. Note on numbering: the verbatim log's headings
run Round 1..18 then "Round 20" (no heading labeled 19); commit history calls
the last entry round 19. Round tags below (r1..r18, r20) match the verbatim
headings.

MANDATE: the lateral lane - unorthodox angles on the twin-prime machine that
the straight-ahead workstreams (constructor, mechanic, harvester, formalist)
do not take. Standing directives: build NEW objects out of RELATIONSHIPS
between the machine's parts; treat "arithmetically selected, no smooth law" as
a target, not a verdict; untested is not dead (abandoned angles stay listed).

## Definitions (once each)

- Slot k = the pair (6k-1, 6k+1). Opening = surviving slot. A gap is a
  difference of slot indices. FRAME: all lateral gap numbers are SLOT units;
  member-space gap = 6 x slot gap; corpus halved-coordinate gap = 3 x slot
  gap (so lateral "padded link costs exactly q'" == harvester "3q'" halved).
- Machine M_y: gears = primes 5..y; gear q kills via teeth at +-u'(q),
  u' = round(q/6), in centred coordinates; window K = (y^2-1)/6, V = 6K+1;
  open interior = slots with both members > y.
- Exposed set A_q = Z_q minus the two teeth, |A_q| = q-2. Gear 5 exposes
  {0,2,3} mod 5 (every opening has k mod 5 in {0,2,3}).
- B-slot: both members gearful (hit on both sides). U(t) = #{self-block slots
  u'(q) <= t with partner member gearful} - finite, confined to bottom y/6
  slots. n2(t) = distinct both-composite slots <= t. T(y) = twin count.
- F(M) = max gap of M; F_j(M) = max sum of j consecutive gaps; F2 = F_2;
  lambda = mean gap of M.
- Merge step M -> M + q': a killed run is k consecutive old openings killed
  by gear q'. Literal link: spacing s = 2u or q'-s mod q' (opposite teeth),
  letters must alternate. Padded link: spacing = 0 mod q' (same tooth), size
  a multiple of q', costs a gap >= q' in M. Legal run: spacings in
  {0, +-2u} mod q', nonzero letters alternating, zeros insertable freely.
- Saturated run: twin-free run of slots each carrying a prime member
  (load = primes/length = 1). Side word: letter per slot = which member is
  prime (L/R); the machine mirror k -> -k reverses order and swaps L/R
  (revcomp).
- X: the counterfactual twin-free window (the flagship identity's binding
  case).

## Established results

1. TWIN PIN (r1). A twin pair (p, p+2) shares tooth value u', so its 4
   within-pair double-kill CRT classes mod P = p(p+2) are pinned in closed
   form: {+u', -u', +u'(p+1), -u'(p+1)}. +-u' are split kills (own slot and
   mirror); the mixed class is the twin-product slot: 6u'(p+1) - 1 =
   (p+1)^2 - 1 = p(p+2) exactly. Verified 60/60 twin pairs to 2000. Every
   twin gear pair donates >= 2 deterministic in-window wasted kills at every
   level. Generalises: every gear pair's cross coincidences pin at semiprime
   slots qq'.

2. SHARING LAW (r1). Conservation: survivors per full period = prod(q-2)
   regardless of phases - sharing moves WHERE, never HOW MANY. Sub-period
   law: E[waste_shared - waste_indep] = 1 - 2R/P, R = K mod P, sign flipping
   at R = P/2. Confirmed pair-by-pair (15 twin pairs, e.g. (239,241)
   measured +0.907 vs predicted +0.917), totals +2.03 vs +1.91. Net survivor
   effect ~ +1.3 per 10 twin pairs = exactly the law, nothing more.

3. OVERCOUNT CENSUS IDENTITIES (r2). overcount = SAME + B where SAME =
   sum over members of (omega_G - 1) (semiprime census) and B = # both-
   gearful slots (split census); exact against window arrays. Random-phase
   side has closed forms (P(q kills k) = 2/(q-1) independent); the round-1
   z = +6.1 anomaly closes as a difference of formulas: real - E[random] =
   +47.87 overcount, -95.57 lone, one cause (deterministic semiprime +
   Bezout coincidences), two faces. Phase randomisation preserves supply
   exactly; the anomaly was pure position.

4. GAP-GRADED SPLIT LAW (r3). For gears q < q' = q + g the split class rep
   is pure arithmetic: m0 = (-2 q^{-1}) mod g; b0 = (2 + m0 q)/g;
   i = (q' - b0) q^{-1} mod 6; x = (q'(b0 + iq) - 1)/6; mirror class at
   P - x. x is the nontrivial square root of 1 mod qq' in closed form.
   Verified against brute CRT for ALL 2850 prime pairs 5 <= q < q' <= 400,
   zero failures; re-verified at scale (13,861 pairs at y=1009; 753,378
   pairs / 43.9M incidences at y=10007). g = 2 is the UNIQUE gap with
   b0 = 1: its split rep is x = u' <= K, in-window at every scale
   unconditionally. Largest-pair hit rates: twins 100% at every scale
   (y=101/211/503), non-twin decaying to ~50.8%. So twin pairs at gear scale
   are the unique gap class whose split-double contribution to the level-y^2
   ledger is unconditionally guaranteed; all others are alignment-rated.

5. MASTER SUPPLY FORMULA (r4). overcount(t) = sum over coprime pairs of
   squarefree gear products (s_L | 6k-1, s_R | 6k+1), |s_L|+|s_R| >= 2, of
   (-1)^{#gears} N(s_L, s_R; t) - pure floor arithmetic (SAME = one-sided
   terms, PAIRSPLIT = single-single, CORR = both-sided >= 3 gears).
   n2(t) = B(t) - U(t); bridge overcount = SAME + U + n2. Verified with max
   |formula - census| = 0 over EVERY prefix t at y=101 and y=211.
   Availability schedule: u'-pins arrive first (t = 1,2,3,5,7,...), first
   SAME at t=6 (35), FIRST n2 slot at t=20 = (119,121) - so under X, demand
   n2(t) = N(t) - P(t) has no supply before t=20 at any y >= 211 (explains
   the Constructor's onset finding).

6. FLAGSHIP REALITY IDENTITY (r5). P(t) = t + T_win(t) - B(t) + U(t) at
   every t (per-slot residual 0 at y=1009 and y=10007, 1.67e7 slots). X <=>
   the identity binds with T = 0: THE BINDING DEFECT OF THE FLAGSHIP
   IDENTITY IS THE TWIN COUNT - reality deviates by exactly one unit per
   twin slot. Kernel-checkable bookkeeping (all terms census objects).

7. DERIVATIVE-SCAN GEOMETRY (r5). Top-1% strides carry 87-90% of ambient
   prime load (0.869 at y=1009 -> 0.901 at y=10007, discount SHRINKING with
   scale); hub-rate/ambient = 0.999/1.006 (X-likeness is not extra pile-up);
   the bottom band is stride-hostile (max stride in first 1% of window is
   half the global max: 242 vs 478 at y=10007). Compression frontier in one
   line: reality does X-like behaviour at length ~478 shedding 10% of prime
   load; X needs length 1.7e7 shedding none.

8. LOAD-LENGTH FRONTIER (r6). maxload(L) over twin-free runs is ABSOLUTE -
   identical across scales because record-holders are the same integer
   landmarks. Touches the X-ceiling (load 1) exactly up to L* = 13, achieved
   at slots 2452-2464 (members 14713..14783, side word RLLRRLLLLRLRL);
   then a staircase of fixed rationals (13/14, 20/25, 23/32, 52/100, 0.32 at
   478). Saturated runs of L <= 10 renewable at all depths tested (to member
   5e7). Bound target region: L ~ 14-32 (gap < 0.29); no leverage at
   L >= 63. Bottom band is load-OPTIMAL (contains the record runs) but
   length-starved. Record-run anatomy: pure n1 (no n2 inside), composites
   57-70% killed by gears <= 13, every slot a pseudo-twin slot
   (fragile-dense). Load-extremal and length-extremal (chain/fuel) runs are
   DIFFERENT families, merging only at the top of the length range.

9. WORD LAWS (r7). (a) Parity theorem (proved): odd-length saturated runs
   are never self-mirror under reverse-complement. (b) Mirror statistics:
   word distributions closest to revcomp-symmetric (TV 0.328 vs 0.564
   reverse-only vs 0.600 complement-only at L=8) - exactly the k -> -k
   machine symmetry. (c) Duplicate words are CRT alignment: position
   differences divisible by 5 in 86% of duplicate pairs (baseline 20%);
   forced-letter fraction 0.729 (gears <= 13). (d) STRICT-ALTERNATION CAP =
   6, PROVED: gear 5 alone caps strict LRLR... at 6 slots (L-first) / 5
   (R-first); realised witness LRLRLR at slot 19125 at both scales. Perfect
   strict alternation is impossible beyond 6 slots anywhere, ever.

10. HORIZON THEOREM (r8, unconditional, 2 lines). Gear pair (5,7) has
    B-classes {1, 34} mod 35; max cyclic gap 33; so ANY 33 consecutive slots
    contain a both-composite slot. Every saturated run - at every scale,
    forever - has length <= 32; same cap for any run of slots each carrying
    >= 1 prime. L0 = 32 survives adding gears through 23 (period 37.2M,
    8.6M B-slots); the widest corridor starts at k = 2 mod 35 and the
    L* = 13 landmark (slot 2452 = 2 mod 35) sits at the corridor mouth.
    Whether lim L0 = 32 over all gears is a Jacobsthal-type finite-check
    question (monotone non-increasing, >= 13). Language census (gears <=
    13): all 2^L words admissible through L = 4; first exclusions at L = 5
    (LLLLL/RRRRR - same-letter blocks cap at 4, gear 5); growth plateaus at
    ~1100-2600 words for L = 18..32; EMPTY at L = 33. Finite language with a
    wall - the opposite object to the infinite gap-word antidictionary. All
    757 observed run words (members to 7.2e10) are in-language, 0 failures;
    the six L=13 runs use six distinct corridor phases mod 35. Corollary:
    unconditional load ceiling past the horizon, maxload <= 1 - minB(L)/L,
    asymptote 1 - 730/5005 = 0.854 (first unconditional maxload < 1 for
    L > 32).

11. PERSISTENCE LADDER (r7, scoping the HL caveat). persistence(L) = every
    level-y open interior contains an L-saturated run; equivalent to a
    Bertrand-type postulate 6r_{n+1} - 1 < (6r_n + 1)^2 on run positions.
    persistence(1) is a THEOREM (via Brun); persistence(2) = disjunctive
    Polignac over gaps {4,6,8} with side structure, OPEN, strictly weaker
    than twin; persistence(L >= 3) = disjunctive HL at tuple size L.
    Decidable per fixed y; verified bands stay verified. The frontier is a
    DESCRIPTIVE envelope: if persistence fails anywhere, the gap to X widens
    and every bound gets EASIER - the caveat cannot hurt the programme.

12. TOP-GAP STRUCTURE (r9). (a) Mirror pairing exact at every machine
    (slot 0 is a universal opening; maximal gaps come in proper mirror
    pairs). (b) Address pinning: maximal gaps concentrate into 1-2 endpoint
    classes mod 35 and 2-6 classes mod 385 (~30x over baseline); the pinned
    address DRIFTS with the machine (gaps are machine-relative; saturated
    runs absolute). (c) Chain skeleton at new maxima: kill sides strictly
    alternate, interior spacings EXACTLY {2u', q - 2u'} of the new gear.
    (d) New maxima grow from MEDIUM old gaps (0.16-0.68 F_old, chains
    k = 2-3); max-extends-max is the exception. (e) First-flank alphabet of
    near-top gaps = {1,2,3,4,5} slots (first flank only - deeper parts grow
    with y, r11). alpha1 evidence: (F2 - F) * 3 / q_next = 0.88, 1.11,
    0.78, 0.52, 1.16 for y = 13..29 - all below 1.24, no trend; F2 anatomy
    has a flank regime and a medium+medium regime (the asymptotic one).

13. WORD-PINNING LAW (r10, "LAW A" - the uniformity engine). The
    neighbourhood word of a near-top gap pins its address mod 385 to <= 4
    phases, unique for 87% of words; gear 5 pinned to exactly one offset by
    every near-top word (206/206 across five machines); containment exact
    (0 fails), tightness 71-85%. Top-stratum class counts stay FLAT (6-14)
    from y=13 to 29 while gap counts swing 20-106 (word overlap on shared
    skeletons). Survives its first k=4 test (address pinned to 3 phases,
    r11). Machine-independent alpha1 needs the OPEN half: uniformity of the
    near-top word grammar itself.

14. INTERIOR GRAMMAR / GRADED FRAME (r11). A literal merge word with k
    interior kills is side-alternating with spacing word alternating
    sigma = 2u' and q' - sigma: exactly 2 candidates per k, so
    |shapes(k)| <= 2c^{k+1} - finite per k, machine-independent; interior
    grammar finite iff k_max bounded (k_max grows: 2,2,3,2,4 by step).
    Phase-free k=4 census at machine 29: exactly 4 sites with word
    (10,21,10) (two mirror pairs, confirming N4 = 4), ZERO sites for the
    other permitted word (21,10,21) - grammar allows two, arithmetic
    selection realises one. Graded increments (F_{k+1} - F)/q_next at
    machines 23/29: k=2: 0.55/0.71, k=3: 0.83/0.87, k=4: 1.07/1.35, k=5:
    1.48/1.52 - under the 2.5 budget; the grading prices Wall V, it does
    not evade it.

15. FIRING LAW (r12, exact). Inside a chain of gear q', the spacing word's
    first entry fixes the orientation and hence a SINGLE firing residue:
    word starting with s fires iff p = -u (mod q'); starting with q' - s
    fires iff p = +u. Density 1/q' per window (k=1 kills fire at both
    teeth, 2/q'). Zero violations over 13,062 sites. Across the new
    machine's full period every fuel site fires EXACTLY ONCE, at
    j = (fire - p) * P_old^{-1} (mod q'). Hence realized k-chains per new
    period = N_k exactly: alignment is a DENSITY factor, never a COUNT
    factor - no suppression multiplier exists for the graded constant.

16. EXACT MERGE ALGORITHM + EXCESS LAW (r13, incl. two same-day
    corrections). F(M + q') = max over maximal LEGAL killed runs of
    o[i+k] - o[i-1], computed from the OLD machine alone (legal = spacings
    in {0, +-2u} mod q', nonzero letters alternating, zeros free).
    Verified EXACTLY at six steps: F = 18, 25, 34, 43, 58, 88 for
    13->17 .. 31->37. (Literal-only matching undershoots - 71 vs 88 at
    31->37; allowing unordered {0,+-2u} overshoots - 45 vs 43 at 23->29 on
    illegal word (10,10); the alternation-with-free-zeros condition is the
    right one.) Excess law: excess = F_new - F2 = max over words w of
    [span(w) - deficit(w)]; literal-word deficit fit ~ 2.52 ln(openings/
    occurrences) - 1.17 (sd 3.4). First five steps have LITERAL winners; at
    31->37 the winner is the FIRST PADDED RUN: kills at spacings (37, 12),
    span 49 = q' + B, one padded link of exactly 37, excess 20 = 0.541 q'.
    The crossover is a PADDING ONSET (as lambda grows, a padded link's span
    q' at price e^{-q'/lambda} becomes affordable). Measured increments/q':
    0.412, 0.368, 0.391, 0.310, 0.484, 0.811 - max 3.1x under the 2.5
    budget; excess overtakes lemma 1 exactly at the largest fuel population
    (31->37: 0.541 vs 0.270, N3 = 70,964).

17. PADDING LEMMA (r14, spectrum form - exact where it applies; scope later
    superseded by the residue laws of r15-r17). If F_{j+2}(M) < 2q' + jL for
    every j >= 0 (L = min(s, q'-s)), every legal run carries at most ONE
    padded link; if 2q' > F(M), every padded link has size EXACTLY q'.
    Verified at all steps through 31->37; census agrees (19->23: 86 gaps of
    exactly 23; 29->31: 2090; 31->37: 26,367 gaps of exactly 37, max 1
    padded/run, 0 adjacent pairs - matches the mechanic's 26,366 within one
    period-wrap unit). Restores a span ceiling 5q' + 2s <= 6.35 q' in the
    computed range; run form [literal chain] --q'-- [literal chain]; general
    branch form span <= (4+p) q' + 2s. The lemma's threshold DIES at 37->41
    (F(37) >= 88 > 82 = 2q' and F_2(37) >= 90 > 82): a small-machine
    phenomenon, exactly dated.

18. CORRIDOR LAW FOR ADJACENT PADDING (r15). Feasibility of two ADJACENT
    equal padded links depends only on q' mod 35: IMPOSSIBLE for exactly 12
    of the 24 invertible classes (q' = 29, 31, 41, 59, 61, 71, 79, 89 ...),
    possible for the rest (23, 37, 43, 47, 53, 67, 73, 83, 97 ...). Perfect
    dichotomy: where (1,1) is feasible the unequal shapes (1,2)/(2,1) are
    infeasible and vice versa. At 37->41 (g = 6): r, r+6, r+12 all in the
    15-residue exposed set mod 35 has ZERO solutions - adjacent padding
    impossible by the (5,7) corridor alone, no spectrum input.

19. AP LEMMA + SHAPE LAW (r16, scale-free). Openings have k mod 5 in
    {0,2,3}; four terms of an AP with difference coprime to 5 occupy four
    distinct residues; three residues cannot hold four. So NO four openings
    in arithmetic progression with common difference q', for every prime
    q' > 5. Corollaries: j = 2 and j = 4 literal links between two padded
    links are IMPOSSIBLE for every q'; p = 3 all-adjacent impossible.
    SHAPE LAW: consecutive padded links are separated by j in {0, 1} only -
    verified for every prime to 4000; feasibility is a function of q' mod
    210 (42 residues). This replaces r14's expiring spectrum threshold with
    a residue criterion that never expires.

20. COMPLETENESS LEMMA (r17). A shape with n openings can be blocked by
    gear q only if q <= 2n (two teeth forbid <= 2n of q phases; CRT makes
    gears independent). So for n <= 5 the mod-35 test IS the entire
    corridor; gear 11 first enters at n = 6. Consequences: the 37->41 j=1
    shape (offsets 0, 41, 55, 96) is GENUINELY FEASIBLE at every modulus;
    all r15-r16 mod-35 verdicts were already complete. GENERALISED AP
    LEMMA: four openings at pure multiples i*q' with the four i distinct
    mod 5 are impossible - kills p=3 patterns (0,0) and (1,1); survivors
    (0,1)/(1,0) first corridor-feasible at q' = 43. With F monotone in the
    machine, F_2(41) >= F(37) = 88 > 86 = 2*43: 41->43 IS THE FIRST STEP
    WITH NO OBSTRUCTION OF ANY KIND (feasible, not thereby occurring).
    Near-miss observation, no mechanism claimed: the j=1 shape misses by
    exactly ONE at two consecutive steps (needs 86 vs F_3(31) = 85; needs
    96 vs F_3(37) prefix 95).

21. EXPOSED-SET AUTOCORRELATION (r18). c_q(g) = #{r in A_q : r + g in A_q}
    has the closed form: q-2 if q | g; q-3 if g = +-2u_q mod q (the
    literal-link lag); q-4 otherwise. Verified gears 5..31, all lags, 0
    mismatches. Endpoint phase count mod 35 for a lag-g pair = c_5(g) c_7(g)
    in {3..15} - a five-fold swing from the two smallest gears. Explains the
    notorious absences: gap 24 (absent at machines 19 and 23) and gap 29
    (count 6 between neighbours 322 and 112) both carry the MINIMUM value
    3; three of the four below-F absent gap values are minimum-3. Regression
    of log(count) on g with/without log(c_5 c_7): R^2 0.856 -> 0.896
    (machine 23), 0.913 -> 0.934 (machine 29) - the law is multiplicative
    and arithmetic, and accounts for ~1/4 of what was called noise.

22. OPENINGS AP THEOREM (r18). An AP of L openings has common difference
    divisible by every gear q < L + 2: 3 consecutive equal gaps require
    5 | g, 5 require 35 | g, 9 require 385 | g, and L >= y + 2 needs the
    full primorial P(y). Verified on machines 13..29, zero violations;
    longest equal-gap run is 3-4 at every machine, always with g = 5 (the
    minimal witness realised).

23. N-POINT CLOSED FORM (r20). c_q(d_1..d_n) = q - 2n + O, where
    O = #{pairs with d_i - d_j = 0 or +-2u mod q}, exact whenever
    q >= 2n. Verified by brute force, 16,500 checks over gears 5..43 and
    n = 1..5, 0 mismatches. Subsumes the exposed set (n=1), the r18
    three-case form (n=2), and the completeness lemma (c_q > 0 forced when
    q > 2n).

24. ADJACENT-GAP EXCLUSION LAW (r20, a proof, not a statistic). Three
    consecutive openings with gaps (g1, g2) are IMPOSSIBLE whenever
    (g1 mod 5, g2 mod 5) is in {(1,1), (1,3), (2,4), (3,1), (4,2), (4,4)} -
    6 of 25 classes. Forced at every scale in every machine containing gear
    5, and COMPLETE: by the completeness lemma only gear 5 can block a
    3-point shape. Scope: ADJACENT gaps only (at separation j >= 2 no
    exclusion follows). Cross-checked against the Mechanic's census
    (gap_pair_joint.csv, six machines y = 11..29): 1,589 populated lag-1
    cells, ZERO in a forbidden class; at lag >= 2 the same classes carry up
    to 35.8M counts - the law forbids where it claims and is silent where
    it claims. First forbidden-configuration kernel target from this lane.

25. ENHANCED-LAG LAW (r20). Since 2u_q = 3^{-1} mod q: gear q is enhanced
    at lag g <=> q | 3g - 1 or q | 3g + 1. Padded-link endpoint arithmetic
    is governed by the factorisation of 3q' +- 1 (e.g. q' = 37: enhanced
    gears 5, 7, 11). Honest measure: accounts for roughly a tenth of the
    330x padding-supply erraticity - the rest is the interior condition.

26. INTERIOR EXPANSION (r20). density(gap exactly g) = alternating sum
    over interior subsets T of (-1)^|T| D({0,g} u T), with
    D(S) = prod_q c_q(S)/q - every term closed-form from the n-point
    construct. Bonferroni truncation is rigorous (even depth upper bound,
    odd depth lower bound); depth needed grows ~ g/4 (Brun's problem in the
    machine's own language). The construct prunes its own expansion:
    c_5(S) = 0 whenever S occupies >= 4 residues mod 5 (g=20: 97% of terms
    pruned at depth 5), strengthening exactly where terms are most numerous.

## Refuted angles (kept as refuted - do not retry without new input)

1. Umbrella nesting (M2) as a twin-specific mechanism (r1): ANY two gears'
   short umbrellas are concentric at joint shields; the only twin-specific
   part is the edge coincidence, which IS M1's pinned classes. One
   mechanism, not two.
2. Closing the recursion by tooth-sharing COUNT (r1): exact conservation +
   the sharing law give net gain O(T(y)) per window vs needed ~K/log^2, and
   the two guaranteed wasted kills land on already-decided slots (self-block
   and semiprime). Also: sharing does not move max stride (B-C = +0.02 +-
   0.05; real machine z = -0.6).
3. Phase-vector extremality (r2): full phase-space enumeration - the real
   vector is top 10-25% on waste metrics, never extremal beyond the
   degenerate 2-gear mirror space. No variational handle exists.
4. Matched-real-primes RICH/POOR design (r1): confound-dominated by the
   kill-density mismatch (sum 2/q); shows nothing the synthetic design
   doesn't show cleanly.
5. Drift recursion "new max address = f(old top-stratum address)" (r10):
   striking early matches, then 0/4 and 1/2 reachability at steps 19->23,
   23->29 - new maxima come from deep-medium gaps no stratum tracks.
   The honest law is LOCAL: address = pin(word), not inherited.
6. A-priori stabilisation of the near-top word-SHAPE family (r11): the
   CRT-admissible superset (3798 half-shapes) is finite, but cross-machine
   full-shape recurrence is ZERO at every machine, max flank part grows
   7 -> 13 with y, observed halves = 3.2% of admissible and disjoint per
   machine. Extreme-value selection roams without repeating.
7. "1 of 4 fuel sites fire / fuel and records decoupled" (r11, withdrawn
   r12): one-window artifact. Every site fires exactly once per new-machine
   period; realized = N_k per period, exactly.
8. Fuel x alignment "double rarity" multiplier for the graded constant
   (r12): alignment is a density factor, never a count factor. One rarity,
   counted twice. No free multiplier exists.
9. Literal-only asymptotic safety of lemma 2 (r13, withdrawn same day):
   "excess <= 2.67 q' by the cap-6 theorem" - the cap-6 theorem covers
   LITERAL chains only; padded runs escape it. The 31->37 winner is padded;
   literal-only matching stalls at 71 vs the true 88. The "longer literal
   words become profitable" mechanism was half the story - the crossover is
   padding onset.
10. "The ceiling stands on structure" (r16 phrasing, corrected r17): the
    SHAPE law is permanent but the COUNT p is capped only by
    p <= F/q' + alpha/3, which grows; so span <= F + O(q'), not O(q').
    p <= 2 is NOT provable from the AP lemma (survivor patterns (0,1)/(1,0)
    are corridor-feasible from q' = 43).
11. Covering/capacity explanation of absent gaps (r18): residual interior
    demand has positive slack (8-16 spare kills) at every g at both machines
    tested. Gap 24's absence is arithmetic selection plus rarity, NOT
    impossibility. Do not look for a covering obstruction.
12. Smooth supply^2/gaps prediction of padding events (r15): padding
    switches on/off with q' mod 35; it predicted ~5 double-padded runs at
    37->41 where the corridor forbids the adjacent shape outright. Arithmetic
    selection beats the smooth law (same lesson as the k=4 fuel census).

## Abandoned but NOT refuted (untested is not dead)

1. Joint-necessity census (r1): twin pairs jointly own a pseudo-twin at the
   product slot when p(p+2) + 2 is prime (p = 5, 149, 179, 239, ...); never
   censused against generic pairs.
2. Jacobsthal push (r8): does L0 = 32 survive gears <= 100; is lim L0 = 32?
   Finitely checkable per gear set, monotone non-increasing, >= 13.
3. Medium-medium adjacency at word level (r9-r10): can two near-top words
   sit adjacent - a finite CRT check per word pair on the pinned phase sets
   (word lists in address_drift.py's groups). Converted from a period scan
   to grammar-level arithmetic but never run. If pinned classes can never be
   adjacent, alpha1 follows per machine.
4. Extreme-value grammar (r10): characterise a priori which words CAN be
   near-top (flank alphabet + skeleton + pinning) - would make the alpha1
   adjacency check machine-independent.
5. Excess-share pricing vs fuel census (r12): is excess/q' ~ c log(N3)/q'
   or spectrum-driven? Needs machine 37/41 spectra.
6. Deficit >= 0, i.e. FS_max(w) <= F2 for every word (r13): all 13
   observations positive; FS is a sum of two NON-adjacent gaps vs F2 the
   best ADJACENT pair - empirical, unproved. If proved, lemma 2 reduces to
   lemma 1 (for the literal part).
7. The 37->41 knife-edge (r15-r17): the only surviving double-padding shape
   needs F_3(37) >= 96 against a 9.7%-prefix value of 95. One unit decides
   the census. Pre-registered prediction: NO double-padded run at 37->41;
   BANKED PREDICTION: first double-padded run at 41->43 (adjacent shape
   corridor-allowed and spectrum-guaranteed there). Also pre-registered
   (lower-biased, literal-derived): F(41) discriminator <= 100 (saturation)
   vs >= 103 (climbing); expectation CLIMB.
8. Gear-7 AP extension (r16): does gear 7 (exposes 5 of 7) forbid SIX
   openings in q'-AP, capping padded structure further?
9. Cheapest surviving p-shape cost (r17): AP lemma forces p=3 shapes to
   spend literals; if the cheapest surviving p-shape cost (finite
   computation per q' mod 210) grows faster than F_j(M), p is capped
   structurally after all.
10. Persistence(L) empirics (r7): L = 13 witnessed only at the absolute
    landmark; its next Bertrand band unexplored; L = 14 language has 579
    words (Mechanic's hunt sanctioned, 14 <= 32).
11. Kernel/Lean handoffs proposed from this lane, status not tracked here:
    horizon theorem (3 lines: any 33 consecutive slots contain k = 1 or 34
    mod 35); per-slot identity P = t + T - B + U; AP lemma; adjacent-gap
    exclusion law (would be the first forbidden-configuration theorem in
    the Lean ledger).

## Open questions and the current named next construct

- NEXT CONSTRUCT: c_q(g1, g2) - gear x lag-pair autocorrelation. The n-point
  closed form (result 23) applies at n = 3, and part (D)'s flank sum FS is
  also an n = 3 object (left flank, word span, right flank): first time this
  lane's construct and the open requirement have the same arity. Apply the
  closed form to FS directly with the word's span as the middle offset.
- The full gap correlation function (interior disjunction): compute via the
  alternating-sum expansion (result 26) with Bonferroni truncation and the
  c_5 = 0 pruning; the complexity lives there, not in a wall.
- Autocorrelation of the exposed set against the padded lag q' itself -
  where the padding thread and the autocorrelation thread meet.
- Machine-independent alpha1: the word-grammar-uniformity half (result 13)
  is still open; word counts are non-growing (20-106, no trend) but words
  are machine-relative.
- The knife-edge unit of F_3(37), and the 41->43 double-padding prediction.

## Reproduction pointers (script -> claim; all under research/)

- r1  tooth_sharing.py - twin pins, sharing law, part-3/4 tables.
- r2  overcount_census.py - census identities, random closed forms,
      extremality enumeration.
- r3  split_gap_law.py - gap-graded split law (2850-pair verification),
      overcount = SAME + PAIRSPLIT - CORR at y = 53/101/211.
- r4  supply_formula.py - master formula, n2 = B - U, availability
      schedule (first n2 at t = 20).
- r5  derivative_scan.py - P = t + T - B + U per-slot at y = 1009/10007,
      stride geometry, load figures.
- r6  load_frontier.py - frontier curve, landmarks, record anatomy; also
      the decision procedure for persistence at fixed y.
- r7  alternation_words.py - parity theorem data, mirror TVs, duplicates,
      strict-alternation cap.
- r8  word_grammar.py - horizon theorem, language census, 757-run check
      (input research/data/satruns_ge10.csv).
- r9  topgap_corridor.py, topgap_nesting.py - mirror pairing, address
      pinning, chain skeletons, alpha1 table.
- r10 address_drift.py - word-pinning law, drift-recursion refutation
      (near-top word groups live here).
- r11 word_shapes.py, k4_pinning.py - two grammars, k=4 census, graded
      increments.
- r12 firing_ratio.py, firing_law_check.py, firing3137.py,
      graded_constant.py - firing law (0 violations), once-per-period
      correction, graded table.
- r13 merge_decompose.py (literal-only, undershoots), merge_general.py
      (overshoots), merge_correct.py (the right condition; six-step
      verification), excess_law.py, excess_predict.py, merge3137.py.
- r14 padding_bound.py, padding_horizon.py, padding31.py - padding lemma,
      thresholds, census.
- r15 padding_37_41.py, padding_corridor_law.py - corridor law, mod-35
      dichotomy, 37->41 branches.
- r16 corridor_shapes.py, corridor_ap_lemma.py - AP lemma, shape law,
      mod-210 feasibility, banked predictions.
- r17 corridor_complete.py, padding_onset.py - completeness lemma,
      unobstructed-step table, generalised AP lemma.
- r18 exposed_autocorr.py, residual_demand.py, autocorr_fit.py,
      openings_ap.py, openings_ap2.py - c_q(g), regressions, slack
      negative, openings AP theorem. (Tooling: .venv/Scripts/python.exe.)
- r20 npoint_autocorr.py, lagpair_predict.py, lambda_law.py, padded_lag.py,
      bonferroni_gap.py, ie_pruning.py - n-point form, adjacent-gap
      exclusion law (cross-check data research/data/gap_pair_joint.csv),
      enhanced-lag law, Bonferroni + pruning tables.
- Cross-lane data cited: research/data/satruns_ge10.csv,
  research/data/gap_pair_joint.csv, research/data/gap_histograms.csv,
  docs/forbidden-configurations.md (antidictionary contrast).

## Round 20 (numbering note: the FIRST section actually headed "Round 20";
## the earlier r20 tags above are round 19's work, see header note)

Brief: c_q(g1,g2) applied to (D)/flank sums + the padded lag; the human's
COMPLEX-NUMBER frame (DFT of corridor, gap spectra, joint gap-pair
distribution) re-entered with the round-19 objects. All jobs finished before
this write-up (machine-29 background run included); machine cap <= 4 threads
respected (all scripts single-threaded numpy).

27. DEPTH-SUM IDENTITY (proved, one line; verified integer-exact machines
    11-29, g = 1..64, zero mismatches; 214.7M openings at machine 29).
    sum_{j>=1} W_j(g) = prod_q c_q(g) = N2(g), where W_j(g) = # cyclic
    j-windows of consecutive gaps summing to g. Every ordered opening pair at
    lag g is the endpoint pair of exactly one window (j <= g); CRT counts the
    pairs. Corollary: W_j(g) <= prod_q c_q(g) for EVERY depth j - a
    depth-uniform closed-form upper bound on every window-sum count, no
    period scan, arbitrarily large machines. The whole spectrum family F_j
    lives inside one closed-form sum rule. docs/novel/depth-sum-identity.md;
    research/depth_identity.py (persists exact W_j tables to
    research/data/depth_identity_<y>.csv).

28. RENEWAL DECOMPOSITION OF THE GAP HISTOGRAM (the interior disjunction
    measured in isolation). W1(g) = N2(g) * prod_{t=1..g-1}(1 - N3(0,t,g)/
    N2(g)) * kappa(g): closed-form endpoint arithmetic x closed-form
    interior-independent product x measured remainder kappa. Facts:
    (a) kappa(1) = kappa(2) = 1 trivially; kappa(4) = 1 EXACTLY at machines
    13-29 (integer-exact vs full inclusion-exclusion) while kappa(3) < 1 -
    and the per-gear multiplicativity behind it is FALSE for q >= 7, so the
    exactness is a cross-gear cancellation (mirror symmetry suspected; open
    micro-question, kernel-checkable per machine).
    (b) kappa decays smoothly and log-CONVEXLY (accelerating tail); fitted
    slope stabilises ~ -0.16/slot at machines 23/29/31 (-0.164, -0.169,
    -0.156). With the 2-parameter kappa law the model explains 94.9-98.7% of
    the log-variance of the full histogram at machines 19-31.
    (c) WIGGLE TEST (r18 upgraded): dividing out N2 alone removes only
    11-30% of the post-trend residual at machines 19-31 - barely more than
    r18's c5*c7 (24-28%): the ENDPOINT-arithmetic dividend saturates; the
    remaining wiggle is INTERIOR arithmetic, and the full closed form (with
    3-point interior products) captures it: see 29(d).
    (d) Density rescaling g*rho collapses kappa curves only to first order -
    jagged residual arithmetic persists; NOT a clean universal curve
    (tested, refuted as stated). research/renewal_law.py.

29. THE MACHINE IN FREQUENCY SPACE (human directive frame 2; all exact).
    (a) The DFT factorises over gears and is REAL and closed-form:
    hat_q(0) = q-2, hat_q(j) = -2cos(2 pi j u/q); global spectrum verified
    against FFT at machine 17 over all 85085 frequencies (dev 4e-11).
    (b) T3 LAW: 3u = (q+1)/2 mod q for every prime q >= 5 (one line from
    6u = 1; asserted to q = 100000): the tripled teeth are ADJACENT residues
    at the antipode, hat_q(3) = -2cos(pi/q) -> -2 - at local frequency 3
    every gear is nearly a single point in phase. Fourier avatar of the
    tooth law (teeth at +-60 deg) and of 2u = 3^{-1}.
    (c) GOLDEN SPECTRAL GAP: hat_5(2) = phi exactly; for every machine
    containing gear 5, max non-DC |hat|/DC = phi/3 = 0.539345, attained by
    gear 5's +-2 mode alone (full character enumeration, machines 13/17).
    The spectral-gap form of gear-5 corridor dominance (AP lemma, exclusion
    law, pinning). The gap histogram's dominant oscillatory line at machines
    29/31 is this golden line (freq 2/5, power 0.169/0.112, largest by 2-4x)
    - and subtracting the FULL closed form removes 99.6%+ of its power
    (0.1687 -> 0.0006; 0.1116 -> 0.0004); other gear lines drop 36-94% at
    machine 31. Machine 23 is noisier (33 points) and only partially
    collapses. "No smooth law, only the histogram" is now: histogram =
    closed-form arithmetic x smooth renewal, with named spectral lines.
    docs/novel/golden-spectral-gap.md; research/machine_dft.py.

30. PAIR RENEWAL AND WHAT (D)'s ANTI-CORRELATION IS MADE OF
    (research/pair_renewal.py, on the Mechanic's full-period joint census
    including machine 31 = 6.2e9 adjacent pairs).
    (a) Exclusion law extended to machine 31: ZERO counts in the 6 forbidden
    mod-5 classes (was verified to 29).
    (b) Bulk pairs: kappa2/(kappa1*kappa1) has count-weighted mean 0.99 at
    machines 29 and 31 - ON AVERAGE the irreducible correction factorises
    (pair interaction = closed-form arithmetic x independent singles) - but
    per-cell log-sd ~ 1.0-1.4 and machine-drifting median (2.1 -> 0.57):
    the naive factorisation is NOT a per-cell law.
    (c) QUALIFYING pairs (size >= 2u', residue 0/+-2u' mod q'): measured
    R(1) sits x2.4-x6 BELOW the closed-form + factorising-renewal
    prediction (meas/pred 0.167, 0.302-0.309, 0.377-0.418 at machines 23,
    29, 31) - and upgrading to the FULL 4-point interior predictor does not
    close it. THE ANTI-CORRELATION (D) NEEDS IS GENUINELY BEYOND 3- AND
    4-POINT CRT ARITHMETIC. The residual factor shrinks monotonically toward
    1 with machine size (3 points - watch, don't extrapolate). Definition
    flag: my measured machine-19 R(1) = 1.28 (positive!) vs Constructor's
    reported deficit range - their exact qualifying set needs stating
    (size-only gives 2.03 at 19; size+residue gives 1.28; neither matches).
    (d) PADDED LAG: the full closed-form predictor brings the padding-supply
    erraticity down to kappa(q') in [0.007, 0.107] across steps 19->23,
    23->29, 29->31, 31->37 - a 15x residual spread where round 19's
    endpoint-only sigma explained ~1/10 of 330x. The 23->29 supply (6) sits
    BELOW its own machine's kappa trend; padding is renewal-suppressed
    beyond even the interior closed form.

31. FLANK-SUM CORRIDOR LAW - (D) IS NEVER CORRIDOR-FORCED (research/
    fs_corridor.py). A word occurrence with flanks is a 4-point shape
    {0, gL, gL+s, s+T} (s = span, T = FS); by the completeness lemma only
    gears 5, 7 can block it, so feasibility is decided mod 35 with TWO free
    phases (machine phase r, flank split gL). Result: 0 of the 1225
    (s, T) mod-35 classes are blocked - EVERY flank-sum value above the (D)
    requirement is corridor-feasible for every span; checked against all 47
    census word-steps: no (D)-critical interval contains a blocked T.
    BUT 51.6% of individual (gL, s, gR) mod-35 triples ARE blocked - more
    than half the splits are forbidden and the SUM survives through the
    split disjunction. Consequence for Constructor: do not hunt a corridor
    proof of (D); the only route is counting/occurrence (R33 form). This is
    the r18 "selection plus rarity, not obstruction" lesson, now at the
    exact object (D) bounds.

Round-20 refuted angles (added to the standing list):
- "Pair interaction reduces per-cell to endpoint arithmetic x singles":
  killed by log-sd ~1.2 and machine-drifting bias (30b); only the
  count-weighted average factorises.
- "The full closed-form predictor explains the padding supply": 15x spread
  remains (30d).
- "kappa(g*density) is a universal curve": first-order only (28d).
- "(D) might be corridor-forced at n = 4": decisively no - 0/1225 (31).

Round-20 untested angles left open:
- Mechanism of exact kappa(4) = 1 (cross-gear cancellation; mirror symmetry
  suspected). Finite per machine; kernel-checkable.
- kappa's log-convex TAIL as the residence of extreme-value structure: fit
  kappa on the top decile of g only, relate curvature to F - the Wall V
  content of this decomposition, not measured this round.
- Lag >= 2 R prediction needs middle-gap-summed arithmetic - exactly a
  transfer-matrix product over the closed-form 3-point kernels
  (Constructor's frame-1 target; the kernels are ready here).
- The extra qualifying suppression (x2.4-x6, shrinking): constant, or -> 1?
  Needs machines 37/41 joint census (Mechanic).
- A genuine large-sieve inequality on W_j from the exact spectrum (29a) -
  named, not built: the power spectrum is closed-form, so a Beurling-
  Selberg-style bound on window counts is a finite construction.

Reproduction pointers (round 20): depth_identity.py (identity + exact W_j
tables -> data/depth_identity_<y>.csv), renewal_law.py (decomposition,
wiggle test, kappa, padded lag), machine_dft.py (spectrum, T3, golden gap,
line collapse), pair_renewal.py (exclusion at 31, factorisation, qualifying
R), fs_corridor.py (0/1225 law, 51.6% split blocking). All with assertions;
.venv via uv run.

## Round 21

Brief: (a) the C14 +126 deg machine-independent phase (Mechanic's handoff);
(b) the eigenphase statistics test vs GUE/Poisson (the human's Riemann-bridge
hunch); (c) the PSD / large-sieve constraint - does positive-definiteness
bite on (D)-violating windows. All three served; every launched job finished
before this write-up (asym sweep, depth ladder 13-23, machine-31 eigenvalue
sort, machine-23 deep DFS - all DONE, logs in research/data/).

32. THE POLE-PHASE LAW - C14's +126 RESOLVED (docs/novel/pole-phase-law.md;
    research/c14_phase.py). 126 deg = 90 + 36 = arg(omega/(1-omega)) =
    arg(omega - 1), omega = e(1/5): the POLE PHASE of the one-sided integer
    lattice at frequency 1/5. Abel summation (exact identity): H_p(k) =
    [omega/(1-omega)] * B, B = the differenced histogram's transform; pole
    phase = 90 + 180k/p deg for every gear p, frequency k. The measured
    constancy IS "B is real": arg B(5,1) = +3.63, +3.65, +1.82, +0.33,
    +0.35, +0.06, -0.23, -0.34 at machines 11..37 - crossing 0 near m29-31;
    from m19 on, 100.00% of the freq-1 deviation energy lies in the
    126-direction. TWO CONFIRMATIONS: (i) freq 2's pole phase is -18 deg
    (mod 180) and the measured arg H_5(2) converges to it monotonically
    (-31.7 -> -5.7 over 8 machines) - a new measured regularity predicted
    by the frame; (ii) gear 7's bracket is NOT real (drifts -3 -> +17 deg):
    no pin, exactly Mechanic's observed mod-7 drift. Equivalent exact
    forms: golden constraint phi^2(N0+N1) = (N2+N4) + 2 phi N3 on the
    residue-class counts; antisymmetry of the freq-1 deviation under
    v -> 1-v (mod 5) (swaps 0<->1, 2<->4, fixes 3). Anchor (proved +
    integer-asserted 13-23): sum over ALL depths of What_j(omega) =
    (2-phi) prod_{q!=5}(q-2)^2 - N, REAL - openings are exactly uniform on
    A_5, so the depth family's phases close a polygon; the j<=25 spiral is
    measured (irregular; W_2's arm climbs toward the pole phase with
    machine size: 66.5 -> 87.7 -> 113.2 at 17/19/23).
    MECHANISM + HONEST LIMIT: the closed-form predictor N2 * prod(1-N3/N2)
    reproduces the measured phase to +-1.5 deg at 11-31 (gear 7 to
    +-2.5 incl. its drift) - the phase is CRT arithmetic. Pushed beyond all
    data (machines 37..499, pure closed form), the model phase does NOT pin:
    it crosses 126 at ~m31-47 and drifts (124.6 at y=97, 117.6 at y=499).
    So "+126 machine-independent" = pole phase + a plateau; pin-vs-drift is
    DECIDABLE AT m41/43: model predicts 125.5-125.9 there; a return to
    126.0 +- 0.1 falsifies the drift. Amplitude near-law recorded:
    |H_5(1)|/H0 * mean_gap = 1.010..1.037 (+-1%, no trend) - unexplained.

33. JACOBSTHAL OPERATOR SPECTRA ARE POISSON, NOT GUE (docs/novel/
    eigenvalue-statistics.md; research/eig_stats.py). The Riemann-bridge
    test, run exactly. (i) The machine's unitaries (slot shift, renewal
    operator) are exact CLOCKS: single-cycle permutations, eigenphases =
    all roots of unity, spacing = delta(s-1), r = 1 - the rigid extreme,
    proved. (ii) The Hermitian circulant's spectrum (closed form, product
    multiset): desymmetrized consecutive-spacing-ratio <r~> = 0.3964,
    0.3867, 0.3963, 0.3945, 0.3871, 0.3865, 0.3862 at machines 11..31
    (130,636,800 exact levels at 31) vs Poisson 0.38629 / GOE 0.5359 /
    GUE 0.6027: POISSON TO FOUR FIGURES, trend toward Poisson, away from
    GUE. KS to Poisson 0.43 -> 0.0022 (m29); repulsion probe P(s<0.1) =
    0.094 ~ Poisson's 0.0952 vs GOE's 0.0078 - no repulsion. EXACT
    degeneracy law: full-spectrum tie count = P - prod (q+1)/2 at machines
    11/13/17 EXACTLY (313/4501/80549) - the mirror symmetry accounts for
    every degeneracy, zero accidental collisions (desym near-collisions at
    1e-12: 6 at m29, 613 at m31, fraction <= 5e-6, unresolved). Bonus
    exact: det C_M = prod(q-2) = open count. VERDICT: clock and Poisson
    bracket GUE from both sides; neither approaches it. Structural reason:
    any CRT-product spectrum is Berry-Tabor/integrable -> Poisson by
    construction; a GUE-bearing operator would need gear coupling - which
    is exactly the non-tensor obstruction B = I - (x)E_q (Wall V in
    operator form). The bridge fails at tensor operators; only the
    non-tensor sector (nilpotent BS, H's non-triangular part) could carry
    it. Pre-registered expectation (Poisson) confirmed - recorded as a
    test result, not a surprise.

34. PSD DOES NOT BITE - AND WHERE THE SIZE LAW ACTUALLY LIVES
    (research/psd_bite.py). Two exact formalisations of "(do position laws
    force size bounds?":
    (a) MOMENT LP: f(x) = # openings in [x, x+W); moments m1..m4 exact
    closed forms (Stirling coefficients x N1..N4 tables; no scan; machines
    13..41 including beyond-scan 37/41). LP: max # empty windows consistent
    with moments of order <= K. Verdict: NEVER BITES - max p0 at K=4 is
    67.6 (m13, W=F) growing to 4.3e10 (m41); at the (D) thresholds
    W = F_old + q' + 1: 112 (13->17) growing to 1.6e10 (37->41). Pair
    level (K=2, the true PSD content of Wiener-Khinchin) is 5-30x weaker
    still. Correlations of bounded order leave ASTRONOMIC slack: (D) is
    invisible to K <= 4 occupancy moments, margin GROWING with machine -
    consistent with Constructor's r20 "anti-correlation beyond any fixed
    order" and the corridor-resonance non-Markov findings, now with the
    slack quantified.
    (b) EXACT RUN CERTIFICATE (the positive result): E(L) = # runs of L
    consecutive blocked slots = IE over window subsets with hereditary-zero
    pruning (masks only shrink -> zero subtrees skipped exactly). E(F) = 0
    and E(F-1) > 0 recovered EXACTLY (integers) at machines 13/17/19/23:
    F = 11/18/25/34 derived from position laws alone, NO period scan, with
    only 397 / 5,345 / 46,349 / 578,890 nonzero subsets (vs 2^F up to
    1.7e10) - the feasible-pattern set is tiny. Bonferroni truncations
    certify NOTHING short of full depth: first certifying depth k* = 8, 10,
    12, 14 = max nonzero depth + 1 at all four machines (partial sums still
    4 / 16 / 48 / 1154 at the deepest even truncation). So: bounded-order
    correlation data never bites; bounded-level covering LP bites weakly
    and dies at 29 (the parallel round-21 covering-lp-certificates entry,
    matrix_shapes.py - complementary, theirs uses generative phase
    structure); FULL-depth pruned IE is exact and cheap. CROSS-LANE: the
    pruned DFS is a working zero-certificate pattern counter - Constructor's
    named blocker (qualmax_j = 0 without scan) is servable by seeding the
    DFS masks with required-open points; node counts above say the cost is
    1e3-1e6, not 2^|Y|.

Round-21 refuted angles (added to the standing list):
- M2 corridor-hardness beta-model as the sole phase mechanism: phases
  -163..-169 deg at every beta for gear 5 - dead (c14_phase.py part 4).
- "126 deg is an asymptotic arithmetic invariant": unsupported - within the
  model that reproduces all measured phases, the phase drifts through 126
  (plateau, not pin); decidable at m41/43.
- GUE drift of machine operator spectra: refuted with exact spectra
  (toward Poisson at every step, both KS and <r~>).
- PSD / bounded-moment bite on (D)-violating windows: refuted with margins
  (67.6 .. 4.3e10, growing).

Round-21 untested angles left open:
- WHY the gear-5 bracket is real (+-0.4 deg) while gear 7's drifts:
  reproduced by closed form, not conceptually derived. Finite per machine.
- The 1/mean_gap amplitude near-law (constant 1.015 +- 1%).
- The 613 near-collisions in the m31 desymmetrized spectrum (algebraic
  cosine-product coincidences - finite, checkable).
- Lipschitz/transfer strengthening of the moment LP (joint (f(x), f(x+g))
  distributions, all closed-form): would sharpen (a) but was not built.
- Spectral statistics of the NON-tensor operators (nilpotent BS has no
  spectrum; the word-level H's non-triangular sector does) - named as the
  only place a GUE-bearing operator could live.
- The machine-29 depth spiral (ladder run only to 23).

## Round 22

Brief: (a) characterise the NON-TENSOR SECTOR as linear algebra - its rank /
codimension against (x)E_q, and does it GROW with the machine (the round's
spine, Constructor attacking the same question from the counting side);
(b) the spectral statistics of that sector alone (the honest continuation of
the round-21 Riemann-bridge refutation); (c) owed to Constructor, the closed
form of their corridor-phase chain's complex lambda_2; (d) my own open item,
why gear 5's pole bracket is real while gear 7's drifts. All four served; the
one detached job (machine-23 rank profile, 527 s) finished before write-up.
Three novel-register docs; four scripts, all assertion-gated.

35. THE NON-TENSOR SECTOR, MEASURED - AND IT GROWS
    (docs/novel/nontensor-sector.md; research/nontensor.py; machine-23 log
    research/data/nontensor_big.log). The right dimension is the SCHMIDT RANK
    across a gear bipartition G1|G2: CRT makes any function on Z_P a d1 x d2
    matrix, rank 1 = a product, rank r = a sum of r products and no fewer.
    Max over cuts is a certified LOWER bound on tensor rank, and rank over
    GF(p) is a certified lower bound on rank over Q, so measured growth
    cannot be an artifact. Three results:
    (a) DEPTH 1 IS BOUNDED - A THEOREM. b = 1 - (x)e_q reshapes to J - x y^T
    with x, y non-constant, so SCHMIDT RANK OF B IS EXACTLY 2 AT EVERY CUT,
    EVERY MACHINE (exposure is rank 1); same for BS = (x)S_q - (x)(E_qS_q).
    Asserted over ALL bipartitions at machines 11/13/17. The non-tensor
    sector at depth 1 is ONE rank-one correction - the difficulty of F is NOT
    dimensional there.
    (b) THE MERGE CUT IS LINEAR - A SECOND THEOREM. Cutting off the top gear,
    V[r,k] = prod over i with (k+i) OPEN in the old machine of [r+i in T_q'],
    so the column depends ONLY on the old machine's opening pattern O, and
    since |T_q'| = 2 it VANISHES unless |O| <= 2 (for n <= q'). Hence
    rank_n = [n < F_old] + #singleton classes + #literal pairs <= 2n+1;
    measured == predicted at EVERY row, machines 11-23, n <= min(14,q').
    This is the merge law's old-machine-only character derived as a rank
    computation, and it prices it.
    (c) WINDOW DEPTH IS UNBOUNDED - THE SPINE ANSWER. v_n(k) = prod_{i<n}
    b(k+i), F = min{n : v_n = 0}, (BS)^n = diag(v_n) S^n; IE gives
    rank_n <= min(2^n, d1, d2). Measured exactly (mod-p ranks at two primes,
    agreeing): at the FIXED corridor cut {5,7} (d1 = 35 always) the peak is
    15, 26, 33, 35 at m13/17/19/23 - it SATURATES. At EVERY fixed cut the
    peak rises m19 -> m23, and five cuts go FULL (peak = d1): {5,7} 33->35,
    {5,11} 48->55, {7,11} 69->77, {11,13} 126->143, {11,17} 140->187; others
    {13,17} 138->220, {13,19} 119->244, {17,19} 109->286, {5,7,11} 119->201;
    every SINGLE-GEAR cut is already FULL from m17 on (5/5,7/7,11/11,13/13,17/17).
    Certified tensor-rank lower bound TR_low = 6, 17, 54, 161, 326 at
    m11/13/17/19/23. THE SECTOR FILLS WHATEVER DIMENSION THE CUT PROVIDES,
    so the tensor rank grows like the largest available cut ~ sqrt(P).
    VERDICT: NO FIXED-ARITY RULE EXISTS for the window/realizability content;
    only an arity-free generator survives. AND THE GROWTH IS IN THE NILPOTENT
    DIRECTION: (BS)^n has spectrum {0} at every depth, so the direction that
    grows is exactly the one with no eigenvalues and no bounded-order
    correlation signature - the same wall as R37's tropical boundary, R41's
    counting boundary, and my own r21 moment-LP non-bite, now with a
    dimension attached.

36. THE NON-TENSOR SECTOR CANNOT CARRY GUE EITHER - AND THE REASON IS A
    THEOREM (docs/novel/farey-chebyshev-spectrum.md; research/
    nontensor_spec.py). PATH-DECOMPOSITION THEOREM: A = BS + (BS)^T is the
    adjacency matrix of the graph on Z_P with edge {k,k+1} iff k+1 is
    blocked, so A is the disjoint union over the machine's GAPS of PATH
    graphs - a gap of g slots contributes P_g. Hence exactly
    spec(A) = union over g, with multiplicity W_1(g), of {2cos(pi j/(g+1))}.
    Verified: dense eigvalsh at m11 agrees to 1.3e-15; path bookkeeping
    (#paths = #openings, sum of lengths = P, longest = F) asserted at
    m13/17/19/23. COROLLARIES: (i) #distinct eigenvalues = |Farey(F+1)| - 2 =
    sum_{b<=F+1} phi(b) = O(F^2) - 21/45/119/211/383/603/1085/2455 at
    m11..37, against periods up to 1.2e12, i.e. P/F^2-fold ties on every
    level; (ii) the distinct levels are a smooth image of a FAREY set, whose
    spacings obey HALL's law with a HARD GAP: measured s_min/s_mean =
    0.476, 0.386, 0.340, 0.333, 0.328, 0.321 descending to 3/pi^2 = 0.30396,
    P(s < 0.1 mean) = 0 exactly, and <r~> = 0.703 - ABOVE GUE's 0.6027;
    (iii) any diag(w) S^t + h.c. has max degree 2, so every such operator is
    a union of paths and cycles; (iv) the growing-rank operators are
    nilpotent; (v) the word-level H is triangular with an INTEGER diagonal.
    THE DICHOTOMY: where the spectrum is rich the operator factorises
    (Poisson); where the operator does not factorise the spectrum is
    degenerate or empty. GUE is now bracketed THREE times and hit zero:
    clock 1.000 > Farey-Chebyshev 0.703 > GUE 0.603 > GOE 0.536 >
    Poisson 0.386 (r21's tensor sector). The Riemann bridge is closed at
    finite machines, with a reason rather than a statistic.

37. THE CORRIDOR RESONANCE IN CLOSED FORM - OWED TO CONSTRUCTOR, DELIVERED
    (docs/novel/corridor-eigenvalue-closed-form.md; research/
    corridor_lambda.py). Exact input (CRT, no fit): openings are exactly
    equidistributed over the exposed phase set E mod m, so the per-slot
    hazard is EXACTLY h(r) = rho [r in E], rho = prod_{q not | m}(1 - 2/q).
    One modelling step (slot independence) makes the phase chain
    M = (I - B)^{-1}O with B = S D_{1-h}, O = S D_h, and then
    M x = lambda x <=> S D_{lambda(1-h)+h} x = lambda x, whose characteristic
    polynomial is that of a weighted single m-cycle:
        lambda^m = prod_s [lambda(1-h(s)) + h(s)] = lambda^{m-e} [(1-rho)
        lambda + rho]^e,   e = |E| = prod_{q|m}(q-2),
    so lambda = 0 with multiplicity m-e and otherwise
        LAMBDA_j = rho w_j / (1 - (1-rho) w_j),  w_j = e(j/e).
    THE SPECTRUM IS A MOEBIUS IMAGE OF THE e-TH ROOTS OF UNITY, hence lies on
    ONE CIRCLE |z - (1-rho)/(2-rho)| = 1/(2-rho) through 1. The resonance is
    mod 15, not mod 35: e = |A_5||A_7| = 15, because the walk never visits a
    blocked phase - which is why the measured period is near 8 and not near
    17. MEASURED vs CLOSED FORM (exact full-period chains, m11-23):
    |l2| 0.9849/0.9634/0.9396/0.9125/0.8859 vs 0.9773/0.9487/0.9205/0.8900/
    0.8614, arg +29.27/+34.39/+38.67/+42.77/+46.31 vs +29.07/+33.88/+37.80/
    +41.48/+44.59 - within 0.008-0.025 in modulus and 0.20-1.71 deg. (The
    m13/19/23 rows reproduce Constructor's 0.96/0.91/0.89 and their 34-46 deg
    range.) mod 385 (e = 135) matches arg to 0.001 deg. Circle residual over
    ALL eigenvalues <= 0.15 R (mod 35), <= 0.10 R (mod 385). THE RESIDUAL IS
    THE ANTI-CORRELATION, and it is positive at every machine (the real chain
    keeps MORE memory than independence) with decelerating increments
    70,46,33,20 e-4 -> PRE-REGISTERED: machine 29 mod 35 measures
    |lambda_2| = 0.862 +- 0.004, arg = +49.2 +- 0.4 deg (closed form 0.8366 /
    +47.09). Closed-form predictions with no scan: m29/31/37/41 mod 35 give
    |l2| 0.8366/0.8118/0.7900/0.7696, arg +47.09/+49.44/+51.40/+53.17,
    periods 7.65/7.28/7.00/6.77 lags - the resonance period SHORTENS with the
    machine, it is not a fixed "period 8".

38. WHY GEAR 5's BRACKET IS REAL - HYPOTHESIS PRE-REGISTERED AND REFUTED
    (research/bracket_why.py). The pole-phase law makes "+126 deg" equivalent
    to "B(5,1) is real"; measured arg B(5,1) = +4.70, +3.78, +1.81, +0.33,
    +0.35 at m11-23 while arg B(7,1) climbs -2.41 -> +14.31. I pre-registered
    (in the script docstring, before running) that item 37's corridor-renewal
    model would make arg B(5,1) IDENTICALLY ZERO in its single parameter
    a = 1 - rho, which would have derived the round-21 open question as an
    identity. FALSE: in the model arg B(5,1) spans 90 deg over a, and at the
    machines' own a values it sits at +11.0 -> +14.2 (moving AWAY from 0
    while the machine moves TOWARD 0), and arg B(7,1) sits at a nearly flat
    -19.5 -> -15.0 while the machine climbs through it. THE MODEL IS WRONG IN
    THE SIGN OF DRIFT FOR BOTH GEARS. So the gear-5 reality is NOT an
    endpoint/independence effect - it is produced by the slot-to-slot
    CORRELATION the model discards (the interior/kappa term). Useful negative:
    the same one-parameter model settles the mean-hazard quantity
    (lambda_2, to 1-2%) and is refuted by the fine phase quantity (arg B) -
    the two observables separate cleanly, and the round-21 question narrows
    from "why gear 5" to "why does the interior correlation cancel the
    endpoint phase at p = 5 and not at p = 7".

Round-22 refuted angles (added to the standing list):
- "the non-tensor sector is small": at depth 1 yes (rank exactly 2) but at
  window depth it saturates whole gear cuts - the bounded reading is wrong.
- "a GUE-bearing operator lives in the non-tensor sector" (my own r21
  localisation): refuted - the sector's Hermitian operators are path unions
  (Farey/Chebyshev, MORE rigid than GUE) and its high-rank operators are
  nilpotent (no spectrum).
- "the corridor-renewal model explains the pole-bracket phase": refuted,
  wrong in the sign of drift at both gears 5 and 7 (item 38).
- "arg B(5,1) is identically zero in the renewal model": pre-registered and
  falsified in the same script.

Round-22 untested angles left open:
- Is rank_n = min(2^n, d1, d2) EXACTLY in a range of n (is the sector
  generically full)? Peaks reach 35/35 at {5,7} but 326/391 at {17,23}; the
  deficit's law is unknown and is a finite computation per machine.
- The rank profile's PEAK DEPTH (6, 8, 10, 11 at {5,7} for m13/17/19/23)
  against F (11, 18, 25, 34) - the peak sits near 0.4F and drifting; is
  peak depth a function of the mean gap?
- Machine 29 rank profile (P = 1.08e9; needs a streaming/bitset build, not
  the dense reshape used here) - deliberately scoped out this round.
- Whether the corridor lambda_2 residual saturates (+0.027) or grows: needs
  the m29 corridor chain, which is a full-period pass.
- The interior/kappa mechanism behind gear 5's bracket reality (item 38's
  narrowed question).
- Carried from r21 and still open: the 1/mean_gap amplitude near-law; the
  613 m31 near-collisions; the Lipschitz-strengthened moment LP; the m29
  depth spiral.

Reproduction pointers (round 22): nontensor.py (parts 1-3 + verdict;
--big adds machine 23, log research/data/nontensor_big.log),
nontensor_spec.py (--big adds the m29/31/37 distinct-level and Farey rows),
corridor_lambda.py (--big adds machine 23), bracket_why.py. All
assertion-gated; .venv/Scripts/python.exe.

Reproduction pointers (round 21): c14_phase.py (parts 1-6: phases, golden
constraint, depth spiral + closure, models M0/M1/M2, asymptotic sweep,
pole-phase decomposition; data/c14_asym2.log, c14_ladder.log),
eig_stats.py (--big for m31; data/eig_big.log), psd_bite.py (--deep-only
for m23 DFS; data/psd_deep.log). All assertion-gated; inputs
data/gap_pair_hist.csv (Mechanic), data/depth_identity_*.csv (r20).

## Round 23

Brief: (a) WHAT REPLACES SPECTRUM IN A NILPOTENT SECTOR - build the object,
measure it, say what it captures that the spectrum cannot; (b) push the
round-22 path-decomposition theorem; (c) close the 0.029 modulus deficit in
lambda_2 (Constructor's corridor pinning from my side). All three served.
Three scripts, all assertion-gated: nilpotent_invariants.py (33 s),
potential_arity.py (LP ladder), lambda2_pair.py (6 s). Two new
novel-register docs plus a round-23 update section on my own
corridor-eigenvalue doc. Compute stayed within the thread cap.

39. THE ANSWER TO (a), AND IT IS A THEOREM PLUS A NEGATIVE
    (docs/novel/nilpotent-invariants.md; research/nilpotent_invariants.py).
    JORDAN = GAP HISTOGRAM. N = BS acts by N e_k = b(k+1) e_{k+1}, so its
    directed graph is the disjoint union of the chains of consecutive blocked
    slots and

        N is PERMUTATION-similar (hence unitarily equivalent) to
        (+)_g J_g^{(+) W_1(g)} - one nilpotent Jordan block per GAP.

    Equivalently rank(N^n) = sum_g W_1(g)(g-n)_+ (the histogram TAIL SUM) and
    #blocks of size exactly L = W_1(L), largest block = F. Verified as EXACT
    INTEGERS at m11/13/17/19, and the permutation is built explicitly at
    m11/13 with the permuted matrix asserted EQUAL to the block sum entry by
    entry.
    THE NEGATIVE (the real content): EVERY UNITARY INVARIANT OF N IS A
    FUNCTION OF THE GAP HISTOGRAM ALONE - singular values, all Schatten
    norms, Jordan type, kernel-filtration dimensions, numerical range,
    resolvent norms, pseudospectra. So the brief's candidate list is
    exhausted in one line: none of them can bound F except circularly. This
    is Wall V in invariant-theoretic form, and it UPGRADES round 22's path
    theorem - A = N + N^T being a union of paths P_g is the symmetrised
    shadow of this Jordan decomposition, same index set.
    WHAT THE INVARIANTS STILL BUY (three, each turning F into a different
    kind of quantity):
    (i) THE NORM CLIFF. N^n = diag(v_n)S^n is a PARTIAL ISOMETRY, singular
    values 0/1, so ||N^n||_op = 1 for n < F and 0 for n >= F - a step
    function with no decay rate at all, and any envelope ||N^n|| <= C lam^n
    (lam < 1) forces C >= lam^(1-F): F SITS ENTIRELY IN THE CONSTANT. That is
    why every analytic decay frame has stalled, stated exactly.
    (ii) THE NUMERICAL RADIUS. w(N) = cos(pi/(F+1)) EXACTLY and the numerical
    RANGE is that disk (direction-independence verified to 6.7e-16 at m11), so
    F = pi/arccos(w) - 1: THE MAXIMAL GAP IS A VARIATIONAL, SDP-REPRESENTABLE
    QUANTITY, and every upper bound on it has a dual certificate. Checked
    two-sidedly at m11-19 with no eigensolver (path Perron weight as an exact
    Schur test, theta = 2cos(pi/(F+1)) to 1e-9).
    (iii) THE PSEUDOSPECTRUM. Spectrum {0}, but ||(zI-N)^-1|| = |z|^(-F)
    (1+O(|z|)) and r_eps = eps^(1/F)(1+o(1)); recovered exponent 25.782 ->
    25.107 -> 25.005 at m19 for eps = 1e-6/1e-12/1e-24, monotone from above.
    With z = e^(-1/t) this is MASLOV DEQUANTISATION: t log||(zI-N)^-1|| ->
    F, i.e. THE (+,x) RESOLVENT COMPUTES THE (max,+) LONGEST PATH.
    Constructor's Kleene star, the Boolean window filtration and the analytic
    resolvent are ONE computation in three semirings.
    WHERE THE NON-INVARIANT CONTENT LIVES: ker N^n is a COORDINATE subspace
    (verified m11), so the kernel FLAG is a nested family of SUBSETS of Z_P;
    its dimensions are histogram tail sums (circular) while its POSITION
    against the CRT gear basis is not a unitary invariant at all - and that
    position is exactly round 22's Schmidt-rank profile, the part that GROWS.
    Round 22 and round 23 therefore fit exactly: invariants = histogram,
    growth = alignment of the kernel flag with the gear tensor basis.

40. THE CERTIFICATE ARITY LADDER - (a)'s only escape route, measured
    (docs/novel/potential-arity-ladder.md; research/potential_arity.py).
    A certificate is not an invariant, so it escapes item 39. For h: Z_P -> R
    with (*) h(k) - h(k-1) >= 1 at every BLOCKED slot, F <= 1 + osc(h), and it
    is TIGHT (h = distance back to the previous opening gives osc = F-1
    exactly, asserted m11-19). Multiplicative form w = exp(h/t) is a SCHUR
    TEST on A; the tropical limit is Constructor's max-plus potential. The
    frame loses nothing - only ARITY can fail, and arity is the round's spine
    question asked as a proof obligation (an infeasibility verdict rules out
    EVERY certificate of that arity, not one attempt).
    T1 (one line, proved): a potential depending only on k mod m for a PROPER
    divisor m of P is infeasible - every class mod m contains a blocked slot
    (asserted m = 35, 385 at m11-19), so (*) forces h(r) > h(r-1) all round
    the m-cycle and 0 >= m. A state that has forgotten a gear cannot see that
    a slot is blocked - this is why bounded-state certificates mod 35/385/5005
    cannot bound F (Constructor's 23->29 failures explained structurally).
    T2 (MERTENS NO-GO, proved, exact rationals): a LEVEL-1 (per-gear)
    potential exists only if sigma(y) = sum_(5<=q<=y) 1/q < 1/2. Proof by two
    CRT averages (over all slots, and over "gear q at a tooth with all others
    exposed"), giving Sigma(1 - 2 sigma) >= 2 sigma with Sigma > 0.
    sigma(11) = 167/385 = 0.4338 but sigma(13) = 2556/5005 = 0.5107, and
    sigma DIVERGES: ARITY-1 CERTIFICATES DIE AT MACHINE 13 AND NEVER RETURN.
    MEASURED LADDER (LP; every FEASIBLE verdict re-checked by rebuilding h and
    testing (*) at every blocked slot over the full period, so no bound trusts
    the solver):
        y=11 F=7 : arity1 23.902 (3.41x), arity2 7.753 (1.11x), arity3 7.000
        y=13 F=11: arity1 INFEASIBLE, arity2 17.980 (1.63x), arity3 11.000,
                   arity4 (full) 11.000
        y=17 F=18: arity1 INFEASIBLE, arity2 37.102 (2.06x)
        y=19 F=25: arity1 INFEASIBLE (1,237,940 rows, 110 s), arity2 FEASIBLE
                   (certificate found on a 4,836-row subsample then VERIFIED
                   against all 1,237,940 blocked slots, min step 1.0000; bound
                   <= 195.5, not the optimum - see the scoped-out note below)
    MY OWN PRE-REGISTERED P2 REFUTED: I wrote "r*(19) >= 3" into the script
    docstring before running; arity 2 IS feasible at m19, so r*(19) = 2. The
    correction is the threshold law itself - r* grows only when sigma crosses
    the next half-integer, i.e. DOUBLY LOGARITHMICALLY (level 2 survives to
    y ~ 109). Right in direction, badly wrong in rate, and the rate is the
    point: a fixed arity stays FEASIBLE long after its BOUND is worthless, so
    feasibility alone is the wrong statistic to watch.
    Two facts: arity 1 dies exactly where T2 says; and WHERE A FIXED ARITY
    SURVIVES ITS QUALITY DECAYS - the arity-2 bound is 1.11x, 1.63x, 2.06x the
    truth at m11/13/17. A fixed-arity certificate becomes asymptotically
    vacuous while remaining feasible.
    THE THRESHOLD LAW (conjectured, derivation and its gap stated): the same
    averaging at level r gives sum_U |U| a_U <= (2A-2) sigma, which closes to
    a contradiction when sigma >= r/2 PROVIDED the a_U are non-positive - the
    sign condition is the gap, named not hidden. Taken at face value LEVEL r
    DIES AT sigma(y) >= r/2, fitting every measured cell, with doubly
    exponential thresholds: level 1 at y=13, level 2 at y=109, level 3 at
    y=2741, level 4 at y=483281. So required arity r*(y) ~ 2 sigma(y) ~
    2 log log y - unbounded, doubly logarithmically slow. The law was written
    down BEFORE the m19 arity-2 cell resolved and predicted it correctly
    (2 sigma(19) = 1.244 < 2, so level 2 survives), and it fits 8 of 8
    measured cells.
    SCOPED OUT, with the cost measured: the OPTIMAL arity-2 bound at m19. The
    osc-minimising LP at full row count (1.24M rows x 30 nonzeros) exceeds
    memory, and the row-generation version did not converge within the round's
    budget while ~20 jobs from other lanes were running. Feasibility itself is
    settled and proved; only the sharpest number is missing.
    THE CONVERGENCE THAT MATTERS: the project's LP-DUALITY thread, on a
    completely different certificate family (covering/Farkas duals for (D)
    rungs, not potentials for F), independently found required degree ~
    2*S1(y) with the same reciprocal-prime sum, and used it to identify
    Constructor's truncation arity 3 -> 4. TWO UNRELATED CERTIFICATE FRAMES,
    THE SAME ARITY LAW r* proportional to sum_(q<=y) 1/q. "No fixed-arity
    rule" now has an arithmetic source: the divergence of sum 1/q.

41. TWO CHECKED NON-GAINS ON THE PATH-DECOMPOSITION THEOREM (item (b)),
    recorded so they are not rebuilt.
    (a) MOMENTS REDUCE TO THE RUN LADDER. tr(A^2t) = sum_L m_t(L) r_L where
    m_t(L) counts closed 2t-walks on Z of RANGE L and r_L = rank(N^L) - a
    closed walk's support is an interval, so it demands exactly an L-run of
    blocked slots. Verified t = 1..6 at m11 (500, 1044, 2600, 7140, 20660,
    61584, exact). So EVERY trace/moment - equivalently every exponential-sum
    - attack on lambda_max(A), hence on F, is a POSITIVE COMBINATION of the
    r_L ladder round 21 already computes exactly and scan-free. No new
    information in the moment frame.
    (b) WEYL ON THE MERGE STEP IS VACUOUS. A_new = A_old + Delta with Delta
    the edges whose right endpoint is newly blocked by q'. MEASURED: the
    longest run of consecutive newly-blocked slots is 1 at every step 11->13,
    13->17, 17->19, 19->23, so lambda_max(Delta) = 1 exactly and the Weyl
    bound is 2.848, 2.932, 2.973, 2.985 - above 2 at every step, hence
    vacuous. The merge step's content is WHICH edges are added, never how
    many.

42. ITEM (c) CLOSED - AND THE 0.029 WAS A COORDINATE, NOT A CORRELATION
    (research/lambda2_pair.py; round-23 update section in
    docs/novel/corridor-eigenvalue-closed-form.md).
    THE IDENTITY: let q(n) be the EXPOSED-STEP LAW - the distribution of how
    many exposed phases mod m one gap crosses. Then

        lambda_2 = q-hat(1/e) = sum_n q(n) e(n/e)   TO 1e-5,

    measured against the exact full-period chain: modulus residual 0.000000,
    0.000011, 0.000032, 0.000065 and argument residual 0.000, 0.005, 0.009,
    0.012 deg at m11/13/17/19 (mod 35); 0 to six decimals at mod 385. In the
    exposed-step coordinate a phase-blind chain is an exact CIRCULANT on Z_e,
    so its eigenvalues are exactly q-hat(j/e), and the 1e-5 residual is the
    whole of the phase-dependence.
    WHAT ROUND 22's FORMULA WAS: under independence the step law is
    GEOMETRIC(rho) and sum rho(1-rho)^(n-1) w^n = mu(w) - my Moebius form is
    exactly q-hat with q geometric. SO THE DEFICIT IS ENTIRELY THE
    NON-GEOMETRICITY OF THE STEP LAW, and Constructor's p-hat(1) is the same
    object in the same coordinate.
    TWO PRE-REGISTERED MODELS REFUTED (my own, both written into the script
    docstring before running): MODEL 1 (exact 2-point conditional hazard
    prod_q c_q(t)/(q-2) replacing the constant rho) and MODEL 2 (the
    both-endpoint 3-point interior, i.e. round 20's renewal law with kappa=1)
    BOTH MOVED |lambda_2| THE WRONG WAY, worsening the deficit by 52-67% and
    83-117% respectively. They failed because they corrected the SLOT-LAG
    hazard, which double-counts the phase structure the corridor already
    carries. Deficits (mod 35, m11..19): M0 +0.0076/+0.0146/+0.0192/+0.0225,
    M1 +0.0116/+0.0228/+0.0307/+0.0375, M2 +0.0140/+0.0287/+0.0394/+0.0487,
    M3 (exact gap law, phase-blind) +0.0176/+0.0169/+0.0175/+0.0185,
    M4 (exact STEP law, phase-blind) +0.0000/+0.0000/+0.0000/+0.0001.
    THE STEP LAW, PARTLY IN CLOSED FORM: its mean is EXACTLY 1/rho at every
    machine (CRT identity - the geometric model already has the right mean, so
    the deficit is pure SHAPE); and its first term is exact closed form,
        q(1) = avg over r in E of prod_(q not | m) c_q(d(r))/(q-2)
    with d(r) the slot distance to the next exposed phase - verified to
    2.2e-16 at m11/13/17/19 (7/9, 0.6363636364, 0.5515151515, 0.4866310160).
    Shape: q(n)/geometric(n) is SUPPRESSED at n=1 (0.951, 0.919, 0.903, 0.890)
    and ENHANCED at n=2 (1.494, 1.378, 1.299, 1.247), then decays. The
    corridor pinning is therefore a one-dimensional measurable object.

Round-23 refuted angles (added to the standing list):
- "the lambda_2 deficit is a 2-point (or 3-point) CRT anti-correlation
  effect" - my own pre-registered hypothesis, refuted in the SIGN by my own
  script; both corrections make the deficit worse. The deficit is the
  non-geometricity of the exposed-step law, in the step coordinate.
- "singular values / Jordan structure / pseudospectra could carry what the
  spectrum cannot" - the brief's candidate list, refuted as a class: they are
  all unitary invariants and every unitary invariant of BS equals the gap
  histogram (item 39).
- moment / exponential-sum bounds on lambda_max(A): reduce exactly to the
  r_L run ladder (41a) - no new information.
- Weyl/eigenvalue-perturbation across the merge step: vacuous at every
  measured step (41b).

Round-23 untested angles left open:
- Prove or refute the sigma >= r/2 arity threshold for r >= 2 (the sign
  condition on the a_U is the whole gap; it is a finite statement per r). The
  sharpest available test is the m19 arity-2 OPTIMUM (scoped out this round on
  cost) and the level-2 death predicted at y ~ 109 - out of scan reach, but
  the LP is over pairs only, so a smarter formulation might reach it.
- TRANSPORT THE CERTIFICATE ACROSS A MERGE STEP: h_new from h_old plus a
  gear-q' part would be the merge law in certificate form, and the arity
  ladder says exactly how much room there is. Named, not built.
- The numerical-radius SDP: does it admit ANY tensor-structured dual
  certificate (the arity ladder is the LP shadow of exactly this question)?
- Whether a bound proved in the max-plus semiring dequantises to a usable
  resolvent bound (item 39(iii) makes the dictionary exact; nothing has been
  pushed through it).
- The phase-dependence residual of the step law (1e-5, growing 0 -> 6.5e-5
  over m11..19): is it O(1/e^2) or does it grow?
- Carried and still open: rank_n = min(2^n,d1,d2) exactly?; the peak-depth
  law; the m29 rank profile; the interior/kappa mechanism behind gear 5's
  bracket reality; the 1/mean_gap amplitude near-law; the m31 near-collisions.

Reproduction pointers (round 23): nilpotent_invariants.py (parts 1-8; log
data/nilpotent_invariants.log), potential_arity.py (T1/T2 + tightness +
ladder; a "y:arity[:row_stride]" argument runs one cell; solve_cutting does
row generation for large cells; log data/potential_arity.log),
lambda2_pair.py (models 0-4 with pre-registered verdicts; log
data/lambda2_pair.log). All under .venv/Scripts/python.exe.
