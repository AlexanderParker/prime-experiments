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
