# lateral workstream log (compacted, rounds 1-24)

Compacted 2026-08-29 into ONE cumulative summary. Full verbatim logs:
archive/lateral-full-r1-19.md and archive/lateral-full-r20-24.md - go there for verification
tables, per-machine number dumps and round narrative. Where a later round self-corrected an
earlier claim the LATER verdict stands here and the earlier one is under Refuted. Result
numbers 1-45 are cited by other lanes - do not renumber. Tags r1..r18, r20 match the r1-19
log's headings (it heads Round 1..18 then "Round 20", none labeled 19); r21-r24 are real.

MANDATE: unorthodox angles on the twin-prime machine that the straight-ahead lanes
(constructor, mechanic, harvester, formalist) do not take. Build NEW objects out of
RELATIONSHIPS between the machine's parts; treat "arithmetically selected, no smooth law" as
a target, not a verdict; untested is not dead.

MANDATE AUDIT (round 24, the manager's own finding): rounds 22, 23, 24 all ran on
certificates and relaxations of the LIVE ROUTE's system. Each brief had a native-frame
justification (spectral, Jordan, SDP are this lane's tools) but cumulatively it made Lateral
a second analyst on the live route, which the mandate forbids. THE BRIEFS CAME FROM THE
MANAGER; the drift is the manager's. ROUND 25 RESTORES THE LANE'S OWN MANDATE and its own
backlog - see Untested backlog, ordered accordingly. Route-serving ideas now route through
the manager rather than becoming the round.

## Definitions (once each)

- Slot k = (6k-1, 6k+1). Opening = surviving slot. Gap = difference of slot indices. FRAME:
  all lateral gap numbers are SLOT units; member-space gap = 6 x slot gap; corpus
  halved-coordinate gap = 3 x slot gap (lateral "padded link costs q'" == harvester "3q'").
- Machine M_y: gears = primes 5..y; gear q kills at teeth +-u'(q), u' = round(q/6), centred
  coords; window K = (y^2-1)/6; open interior = slots with both members > y. Period P = prod q.
- Exposed set A_q = Z_q minus the two teeth, |A_q| = q-2. Gear 5 exposes {0,2,3} mod 5.
- B-slot: both members gearful. U(t) = #{self-block slots u'(q) <= t with partner gearful}.
  n2(t) = distinct both-composite slots <= t. T(y) = twin count.
- F(M) = max gap; F_j(M) = max sum of j consecutive gaps; F2 = F_2; lambda = mean gap;
  rho = prod (1 - 2/q).
- Merge M -> M + q': a killed run is k consecutive old openings killed by q'. Literal link:
  spacing s = 2u or q'-s mod q' (opposite teeth), letters alternate. Padded link: spacing 0
  mod q' (same tooth), size a multiple of q', costs a gap >= q' in M. Legal run: spacings in
  {0, +-2u} mod q', nonzero letters alternating, zeros free.
- Saturated run: twin-free run of slots each carrying a prime member (load = 1). Side word:
  letter per slot = which member is prime (L/R); the mirror k -> -k reverses order and swaps
  L/R (revcomp).
- X: the counterfactual twin-free window (the flagship identity's binding case).
- c_q(S) = #{r in Z_q : r+s in A_q for all s in S}. W_j(g) = # cyclic j-windows of consecutive
  gaps summing to g; W_1 = gap histogram. N2(g) = prod_q c_q(g); N3, N4 likewise.
- Operators on Z_P: S = slot shift; b(k) = [k blocked]; B = diag(b); N = BS (the nilpotent
  window operator); A = N + N^T; E_q = per-gear exposure projector.
- sigma(y) = sum_{5<=q<=y} 1/q - the Mertens sum that prices every certificate family.

## Established results

1. TWIN PIN (r1). A twin pair (p,p+2) shares tooth u', so its 4 within-pair double-kill CRT
   classes mod p(p+2) are pinned: {+u', -u', +u'(p+1), -u'(p+1)}; the mixed class is the
   twin-product slot, 6u'(p+1)-1 = (p+1)^2-1 = p(p+2) exactly (60/60 pairs to 2000). Every
   twin gear pair donates >= 2 deterministic wasted kills per window; every gear pair's cross
   coincidences pin at semiprime slots qq'.

2. SHARING LAW (r1). Survivors per full period = prod(q-2) regardless of phases - sharing
   moves WHERE, never HOW MANY. Sub-period law E[waste_shared - waste_indep] = 1 - 2R/P,
   R = K mod P, sign flipping at R = P/2. Net effect ~ +1.3 per 10 twin pairs = the law exactly.

3. OVERCOUNT CENSUS IDENTITIES (r2). overcount = SAME + B (SAME = sum over members of
   omega_G - 1; B = # both-gearful slots), exact against window arrays; random-phase closed
   form P(q kills k) = 2/(q-1), independent. The r1 z = +6.1 anomaly closes as a difference of
   formulas (+47.87 overcount, -95.57 lone): one cause, two faces. Phase randomisation
   preserves supply exactly - the anomaly was pure position.

4. GAP-GRADED SPLIT LAW (r3). For gears q < q' = q+g: m0 = (-2 q^{-1}) mod g,
   b0 = (2 + m0 q)/g, i = (q'-b0) q^{-1} mod 6, x = (q'(b0+iq)-1)/6, mirror at P - x; x is the
   nontrivial square root of 1 mod qq' in closed form. Verified for ALL 2850 pairs
   5 <= q < q' <= 400 and 753,378 pairs at y=10007, zero failures. g = 2 is the UNIQUE gap
   with b0 = 1: its split rep x = u' <= K is in-window at every scale unconditionally (twin
   hit rate 100% everywhere, non-twin decaying to ~50.8%) - twin pairs at gear scale are the
   unique gap class whose split-double contribution is guaranteed, all others alignment-rated.

5. MASTER SUPPLY FORMULA (r4). overcount(t) = sum over coprime pairs of squarefree gear
   products (s_L | 6k-1, s_R | 6k+1), |s_L|+|s_R| >= 2, of (-1)^{#gears} N(s_L,s_R;t) - pure
   floor arithmetic; n2(t) = B(t) - U(t); bridge overcount = SAME + U + n2. Max |formula -
   census| = 0 over EVERY prefix t at y = 101, 211. Availability: first SAME at t = 6 (35),
   FIRST n2 slot at t = 20 = (119,121) - under X, demand n2(t) has no supply before t = 20.

6. FLAGSHIP REALITY IDENTITY (r5). P(t) = t + T_win(t) - B(t) + U(t) at every t (per-slot
   residual 0 at y = 1009 and 10007, 1.67e7 slots). X <=> the identity binds with T = 0: THE
   BINDING DEFECT OF THE FLAGSHIP IDENTITY IS THE TWIN COUNT - reality deviates by exactly one
   unit per twin slot. Kernel-checkable (all terms census objects).

7. DERIVATIVE-SCAN GEOMETRY (r5). Top-1% strides carry 87-90% of ambient prime load (0.869 at
   y=1009 -> 0.901 at y=10007, discount SHRINKING with scale); hub-rate/ambient 0.999/1.006
   (X-likeness is not extra pile-up). In one line: reality does X-like behaviour at length
   ~478 shedding 10% of prime load; X needs length 1.7e7 shedding none.

8. LOAD-LENGTH FRONTIER (r6). maxload(L) over twin-free runs is ABSOLUTE - identical across
   scales, the record-holders being the same integer landmarks. Touches load 1 exactly up to
   L* = 13 (slots 2452-2464, members 14713..14783, word RLLRRLLLLRLRL); then a staircase of
   fixed rationals (13/14, 20/25, 23/32, 52/100, 0.32 at 478). Target region L ~ 14-32; no
   leverage at L >= 63. Load-extremal and length-extremal (chain/fuel) runs are DIFFERENT
   families, merging only at the top.

9. WORD LAWS (r7). (a) Parity theorem (proved): odd-length saturated runs are never
   self-mirror under revcomp. (b) Word distributions closest to revcomp-symmetric (TV 0.328 vs
   0.564 / 0.600 at L=8) - exactly the k -> -k symmetry. (c) Duplicate words are CRT
   alignment: position differences divisible by 5 in 86% of pairs (baseline 20%).
   (d) STRICT-ALTERNATION CAP = 6, PROVED: gear 5 alone caps strict LRLR... at 6 slots
   (L-first) / 5 (R-first). Impossible beyond 6 anywhere, ever.

10. HORIZON THEOREM (r8, unconditional, 2 lines). Gear pair (5,7) has B-classes {1, 34} mod 35,
    max cyclic gap 33, so ANY 33 consecutive slots contain a both-composite slot. EVERY
    saturated run - every scale, forever - has length <= 32; same cap for any run of slots each
    carrying >= 1 prime. L0 = 32 survives gears through 23; the widest corridor starts at
    k = 2 mod 35 and the L* = 13 landmark sits at its mouth. lim L0 = 32? is a Jacobsthal-type
    finite check. Language census (gears <= 13): all 2^L words through L = 4, first exclusions
    at L = 5, ~1100-2600 words for L = 18..32, EMPTY at L = 33 - a finite language with a wall,
    the opposite object to the infinite gap-word antidictionary; all 757 observed run words
    in-language. Corollary: unconditional maxload <= 1 - minB(L)/L past the horizon, asymptote
    1 - 730/5005 = 0.854.

11. PERSISTENCE LADDER (r7, scoping the HL caveat). persistence(L) = every level-y open
    interior contains an L-saturated run; equivalent to a Bertrand-type postulate
    6r_{n+1} - 1 < (6r_n + 1)^2. persistence(1) is a THEOREM (via Brun); persistence(2) =
    disjunctive Polignac over gaps {4,6,8}, OPEN, strictly weaker than twin; persistence(L>=3)
    = disjunctive HL at tuple size L. The frontier is a DESCRIPTIVE envelope: if persistence
    fails anywhere the gap to X widens and every bound gets EASIER - the caveat cannot hurt.

12. TOP-GAP STRUCTURE (r9). Mirror pairing exact at every machine. Maximal gaps concentrate
    into 1-2 endpoint classes mod 35 and 2-6 mod 385 (~30x baseline), but the pinned address
    DRIFTS with the machine (gaps machine-relative, saturated runs absolute). At new maxima
    kill sides strictly alternate with interior spacings EXACTLY {2u', q-2u'} of the new gear;
    new maxima grow from MEDIUM old gaps (0.16-0.68 F_old, chains k = 2-3), max-extends-max
    being the exception. First-flank alphabet of near-top gaps = {1,2,3,4,5} slots. alpha1
    evidence: (F2-F)*3/q_next = 0.88, 1.11, 0.78, 0.52, 1.16 for y = 13..29 - below 1.24, no
    trend.

13. WORD-PINNING LAW (r10, "LAW A"). The neighbourhood word of a near-top gap pins its address
    mod 385 to <= 4 phases, unique for 87% of words; gear 5 pinned to exactly one offset by
    every near-top word (206/206 across five machines); tightness 71-85%. Top-stratum class
    counts stay FLAT (6-14) from y=13 to 29 while gap counts swing 20-106. Machine-independent
    alpha1 needs the OPEN half: uniformity of the near-top word grammar itself.

14. INTERIOR GRAMMAR / GRADED FRAME (r11). A literal merge word with k interior kills is
    side-alternating with spacing word alternating sigma = 2u' and q' - sigma: exactly 2
    candidates per k, so |shapes(k)| <= 2c^{k+1}, finite per k and machine-independent (k_max
    grows 2,2,3,2,4 by step). Phase-free k=4 census at m29: exactly 4 sites with word
    (10,21,10), ZERO for the other permitted word (21,10,21) - grammar allows two, arithmetic
    selection realises one. Graded increments (F_{k+1}-F)/q_next at m23/29: 0.55/0.71,
    0.83/0.87, 1.07/1.35, 1.48/1.52 for k = 2..5 - under the 2.5 budget; the grading prices
    Wall V, it does not evade it.

15. FIRING LAW (r12, exact). Inside a chain of gear q' the spacing word's first entry fixes the
    orientation and hence a SINGLE firing residue: word starting with s fires iff p = -u
    (mod q'), starting with q'-s iff p = +u; density 1/q' per window. Zero violations over
    13,062 sites. Across the new machine's full period every fuel site fires EXACTLY ONCE, at
    j = (fire - p) P_old^{-1} (mod q'), so realized k-chains per new period = N_k exactly:
    alignment is a DENSITY factor, never a COUNT factor - no suppression multiplier exists.

16. EXACT MERGE ALGORITHM + EXCESS LAW (r13, incl. two same-day corrections).
    F(M+q') = max over maximal LEGAL killed runs of o[i+k] - o[i-1], from the OLD machine
    alone. Verified EXACTLY at six steps: F = 18, 25, 34, 43, 58, 88 for 13->17 .. 31->37
    (literal-only undershoots 71 vs 88; unordered {0,+-2u} overshoots 45 vs 43 at 23->29).
    Excess law: excess = F_new - F2 = max over words w of [span(w) - deficit(w)]. First five
    steps have LITERAL winners; at 31->37 the winner is the FIRST PADDED RUN (spacings (37,12),
    span 49, one padded link of exactly 37, excess 20 = 0.541 q'). The crossover is a PADDING
    ONSET: as lambda grows a padded link's span q' at price e^{-q'/lambda} becomes affordable.
    Increments/q' 0.412, 0.368, 0.391, 0.310, 0.484, 0.811 - max 3.1x under the 2.5 budget.

17. PADDING LEMMA (r14; scope later superseded by the r15-r17 residue laws). If
    F_{j+2}(M) < 2q' + jL for every j >= 0 (L = min(s, q'-s)), every legal run carries at most
    ONE padded link; if 2q' > F(M) every padded link has size EXACTLY q'. Verified through
    31->37 (26,367 gaps of exactly 37, max 1 padded/run, 0 adjacent pairs). Span ceiling
    5q' + 2s <= 6.35 q'; run form [literal chain] --q'-- [literal chain]. The threshold DIES at
    37->41 (F(37) >= 88 > 82 = 2q'): a small-machine phenomenon, exactly dated.

18. CORRIDOR LAW FOR ADJACENT PADDING (r15). Feasibility of two ADJACENT equal padded links
    depends only on q' mod 35: IMPOSSIBLE for exactly 12 of the 24 invertible classes (29, 31,
    41, 59, 61, 71, 79, 89 ...), possible for the rest (23, 37, 43, 47, 53, 67, 73, 83, 97
    ...). Perfect dichotomy: where (1,1) is feasible the unequal shapes (1,2)/(2,1) are
    infeasible and vice versa. At 37->41 (g = 6), r, r+6, r+12 all in the 15-residue exposed
    set mod 35 has ZERO solutions - adjacent padding impossible by the (5,7) corridor alone.

19. AP LEMMA + SHAPE LAW (r16, scale-free). Openings have k mod 5 in {0,2,3}; four AP terms
    with difference coprime to 5 occupy four distinct residues; three residues cannot hold
    four. So NO four openings in arithmetic progression with common difference q', for every
    prime q' > 5. Corollaries: j = 2 and j = 4 literal links between two padded links are
    IMPOSSIBLE for every q'; p = 3 all-adjacent impossible. SHAPE LAW: consecutive padded links
    are separated by j in {0,1} only (verified for every prime to 4000); feasibility is a
    function of q' mod 210. Replaces r14's expiring spectrum threshold with a residue criterion
    that never expires.

20. COMPLETENESS LEMMA (r17). A shape with n openings can be blocked by gear q only if q <= 2n
    (two teeth forbid <= 2n of q phases; CRT makes gears independent). So for n <= 5 the mod-35
    test IS the entire corridor; gear 11 first enters at n = 6. Hence the 37->41 j=1 shape
    (offsets 0, 41, 55, 96) is GENUINELY FEASIBLE at every modulus and all r15-r16 verdicts
    were already complete. GENERALISED AP LEMMA: four openings at pure multiples i*q' with the
    four i distinct mod 5 are impossible - kills p=3 patterns (0,0) and (1,1); survivors
    (0,1)/(1,0) first corridor-feasible at q' = 43. With F monotone, F_2(41) >= F(37) = 88 >
    86 = 2*43: 41->43 IS THE FIRST STEP WITH NO OBSTRUCTION OF ANY KIND (feasible, not thereby
    occurring). Near-miss, no mechanism claimed: the j=1 shape misses by exactly ONE at two
    consecutive steps (86 vs F_3(31) = 85; 96 vs F_3(37) prefix 95).

21. EXPOSED-SET AUTOCORRELATION (r18). c_q(g) = q-2 if q | g; q-3 if g = +-2u_q mod q (the
    literal-link lag); q-4 otherwise. Verified gears 5..31, all lags, 0 mismatches. Endpoint
    phase count mod 35 for a lag-g pair = c_5(g) c_7(g) in {3..15} - a five-fold swing from the
    two smallest gears. Explains the notorious absences: gap 24 (absent at m19, m23) and gap 29
    both carry the MINIMUM value 3. Regression of log(count) on g with/without log(c_5 c_7):
    R^2 0.856 -> 0.896 (m23), 0.913 -> 0.934 (m29) - ~1/4 of what was called noise.

22. OPENINGS AP THEOREM (r18). An AP of L openings has common difference divisible by every
    gear q < L+2: 3 consecutive equal gaps require 5 | g, 5 require 35 | g, 9 require 385 | g,
    and L >= y+2 needs the full primorial P(y). Verified m13..29, zero violations; longest
    equal-gap run is 3-4 at every machine, always with g = 5.

23. N-POINT CLOSED FORM (r20-tag). c_q(d_1..d_n) = q - 2n + O, O = #{pairs with d_i - d_j = 0
    or +-2u mod q}, exact whenever q >= 2n (16,500 brute-force checks, gears 5..43, n = 1..5,
    0 mismatches). Subsumes the exposed set, the r18 three-case form, and the completeness
    lemma.

24. ADJACENT-GAP EXCLUSION LAW (r20-tag; a proof, not a statistic). Three consecutive openings
    with gaps (g1,g2) are IMPOSSIBLE whenever (g1 mod 5, g2 mod 5) is in
    {(1,1),(1,3),(2,4),(3,1),(4,2),(4,4)} - 6 of 25 classes. Forced at every scale in every
    machine containing gear 5, and COMPLETE: by the completeness lemma only gear 5 can block a
    3-point shape. Scope: ADJACENT gaps only (at separation j >= 2 no exclusion follows).
    Cross-checked against the Mechanic's census (y = 11..29): 1,589 populated lag-1 cells, ZERO
    forbidden, while at lag >= 2 the same classes carry up to 35.8M counts; extended in round
    20 to m31 (6.2e9 adjacent pairs, still zero). First forbidden-configuration kernel target.

25. ENHANCED-LAG LAW (r20-tag). Since 2u_q = 3^{-1} mod q: gear q is enhanced at lag g <=>
    q | 3g-1 or q | 3g+1. Padded-link endpoint arithmetic is governed by the factorisation of
    3q' +- 1 (q' = 37: gears 5, 7, 11). Honest measure: ~a tenth of the 330x padding-supply
    erraticity; the rest is interior.

26. INTERIOR EXPANSION (r20-tag). density(gap exactly g) = alternating sum over interior subsets
    T of (-1)^|T| D({0,g} u T), D(S) = prod_q c_q(S)/q - every term closed-form. Bonferroni
    truncation is rigorous (even depth upper, odd lower); depth needed grows ~ g/4 (Brun's
    problem in the machine's own language). The construct prunes its own expansion: c_5(S) = 0
    whenever S occupies >= 4 residues mod 5 (g=20: 97% of terms pruned at depth 5).

27. DEPTH-SUM IDENTITY (r20; proved, one line; integer-exact m11-29, g = 1..64):

        sum_{j>=1} W_j(g) = prod_q c_q(g) = N2(g).

    Every ordered opening pair at lag g is the endpoint pair of exactly one window (j <= g);
    CRT counts the pairs. COROLLARY: W_j(g) <= N2(g) for EVERY depth j - a depth-uniform
    closed-form upper bound on every window-sum count, no period scan, arbitrarily large
    machines. The whole spectrum family F_j lives inside one closed-form sum rule.
    PRIOR-ART VERDICT (Harvester, 2026-08-24): KNOWN - this is HOLT, arXiv:2502.20470 (Feb
    2025), Corollary 1, specialised to one constellation (his sum_{j>=J} n_{s,j}(p#) =
    prod (q - nu_q(s)); a twin-slot pair at lag g is his constellation (2, 6g-2, 2)).
    holt_correspondence.py checks it exactly. The identity is TRUE and was derived
    independently; only the NOVELTY label was wrong, and the kernel check (DepthSum.lean) is
    unaffected as verification. The frameworks still separate in general (m17, g=5: our
    n_g = 4,230 vs Holt's n_{s,J} = 0). STANDING LESSON: PRIOR-ART CHECKS EXPIRE - Holt
    postdates the r20/r21 sweeps, which were correct when run; re-check before PUBLICATION.

28. RENEWAL DECOMPOSITION OF THE GAP HISTOGRAM (r20).
    W1(g) = N2(g) * prod_{t=1..g-1}(1 - N3(0,t,g)/N2(g)) * kappa(g). (a) kappa(4) = 1 EXACTLY at
    m13-29 while kappa(3) < 1, and per-gear multiplicativity is FALSE for q >= 7, so the
    exactness is a cross-gear cancellation (mirror symmetry suspected; kernel-checkable).
    (b) kappa decays log-CONVEXLY, slope ~ -0.16/slot at m23/29/31; the 2-parameter kappa law
    explains 94.9-98.7% of the full histogram's log-variance at m19-31. (c) WIGGLE TEST:
    dividing out N2 alone removes only 11-30% of the post-trend residual, barely more than
    r18's c5*c7 (24-28%) - the ENDPOINT dividend saturates, the rest is INTERIOR arithmetic.
    (d) Density rescaling g*rho collapses kappa curves only to first order - NOT universal.

29. THE MACHINE IN FREQUENCY SPACE (r20; all exact).
    (a) MACHINE DFT CLOSED FORM - factorises over gears and is REAL:

        hat_q(0) = q - 2,    hat_q(j) = -2 cos(2 pi j u / q),

    global spectrum = the product over gears (checked against FFT at m17 over all 85,085
    frequencies, deviation 4e-11).
    (b) T3 LAW: 3u = (q+1)/2 mod q for every prime q >= 5 (one line from 6u = 1; asserted to
    q = 100000): the tripled teeth are ADJACENT residues at the antipode, hat_q(3) =
    -2 cos(pi/q) -> -2 - at local frequency 3 every gear is nearly a single point in phase.
    Fourier avatar of the tooth law (teeth at +-60 deg) and of 2u = 3^{-1}.
    (c) GOLDEN SPECTRAL GAP: hat_5(2) = phi EXACTLY; for every machine containing gear 5,

        max non-DC |hat| / DC = phi/3 = 0.539345...,

    attained by gear 5's +-2 mode ALONE (full character enumeration, m13/17) - the spectral-gap
    form of gear-5 corridor dominance (AP lemma, exclusion law, pinning). The gap histogram's
    dominant oscillatory line at m29/31 IS this golden line (freq 2/5, power 0.169/0.112,
    largest by 2-4x), and subtracting the FULL closed form removes 99.6%+ of its power
    (0.1687 -> 0.0006; 0.1116 -> 0.0004); other gear lines drop 36-94% at m31. "No smooth law,
    only the histogram" becomes: histogram = closed-form arithmetic x smooth renewal, with
    named spectral lines.

30. PAIR RENEWAL AND WHAT (D)'s ANTI-CORRELATION IS MADE OF (r20, on the Mechanic's full-period
    joint census incl. m31 = 6.2e9 adjacent pairs). (a) Bulk pairs: kappa2/(kappa1 kappa1) has
    count-weighted mean 0.99 at m29/31 - ON AVERAGE the correction factorises - but per-cell
    log-sd ~1.0-1.4 and drifting median (2.1 -> 0.57): NOT a per-cell law. (b) QUALIFYING pairs
    (size >= 2u', residue 0/+-2u' mod q'): measured R(1) sits x2.4-x6 BELOW the closed-form +
    factorising-renewal prediction and the FULL 4-point interior predictor does not close it -
    THE ANTI-CORRELATION (D) NEEDS IS GENUINELY BEYOND 3- AND 4-POINT CRT ARITHMETIC; the
    residual shrinks monotonically toward 1 with machine size (3 points - don't extrapolate).
    Definition flag: measured m19 R(1) = 1.28 (positive) vs Constructor's deficit range - their
    qualifying set needs stating. (c) PADDED LAG: the closed-form predictor brings
    padding-supply erraticity down to kappa(q') in [0.007, 0.107] across 19->23 .. 31->37, a 15x
    spread where round 19's endpoint-only sigma explained ~1/10 of 330x - padding is
    renewal-suppressed beyond even the interior closed form.

31. FLANK-SUM CORRIDOR LAW - (D) IS NEVER CORRIDOR-FORCED (r20). A word occurrence with flanks
    is a 4-point shape {0, gL, gL+s, s+T} (s = span, T = flank sum); by the completeness lemma
    only gears 5, 7 can block it, so feasibility is decided mod 35 with TWO free phases. Result:
    0 of the 1225 (s,T) mod-35 classes are blocked - EVERY flank-sum value above the (D)
    requirement is corridor-feasible for every span (checked against all 47 census word-steps) -
    yet 51.6% of individual (gL,s,gR) triples ARE blocked: the SUM survives through the split
    disjunction. Do not hunt a corridor proof of (D); the only route is counting/occurrence.

32. THE POLE-PHASE LAW - C14's +126 deg RESOLVED (r21). 126 deg = 90 + 36 =
    arg(omega/(1-omega)) = arg(omega - 1), omega = e(1/5): the POLE PHASE of the one-sided
    integer lattice at frequency 1/5. Abel summation gives the exact identity

        H_p(k) = [omega/(1-omega)] * B,   pole phase = 90 + 180k/p deg
        for every gear p and frequency k,

    B = the differenced histogram's transform. The measured constancy IS "B is real":
    arg B(5,1) = +3.63, +3.65, +1.82, +0.33, +0.35, +0.06, -0.23, -0.34 at m11..37, crossing 0
    near m29-31; from m19 on, 100.00% of the freq-1 deviation energy lies in the 126-direction.
    TWO CONFIRMATIONS: (i) freq 2's pole phase is -18 deg (mod 180) and measured arg H_5(2)
    converges to it monotonically (-31.7 -> -5.7 over 8 machines) - a new regularity predicted
    by the frame; (ii) gear 7's bracket is NOT real (drifts -3 -> +17 deg), exactly the
    Mechanic's mod-7 drift. Equivalent exact forms: the golden constraint phi^2(N0+N1) =
    (N2+N4) + 2 phi N3 on residue-class counts; antisymmetry of the freq-1 deviation under
    v -> 1-v (mod 5). ANCHOR (proved + integer-asserted m13-23): sum over ALL depths of
    What_j(omega) = (2-phi) prod_{q!=5}(q-2)^2 - N, REAL - openings are exactly uniform on A_5,
    so the depth family's phases close a polygon; the j <= 25 spiral is measured (irregular;
    W_2's arm climbs toward the pole phase 66.5 -> 87.7 -> 113.2 at m17/19/23). MECHANISM +
    HONEST LIMIT (where the PIN dies, Refuted 17): the closed-form predictor
    N2 * prod(1-N3/N2) reproduces the measured phase to +-1.5 deg at m11-31 - the phase is CRT
    arithmetic - but pushed beyond all data (m37..499) the model phase does NOT pin: it crosses
    126 at ~m31-47 then drifts (124.6 at y=97, 117.6 at y=499). "+126 machine-independent" =
    pole phase + a PLATEAU; decidable at m41/43 where the model predicts 125.5-125.9. AMPLITUDE
    NEAR-LAW, unexplained: |H_5(1)|/H0 * mean_gap = 1.010..1.037.

33. JACOBSTHAL OPERATOR SPECTRA ARE POISSON, NOT GUE (r21, the Riemann-bridge test run exactly).
    The machine's unitaries are exact CLOCKS (single-cycle permutations, eigenphases = all roots
    of unity, <r> = 1 - the rigid extreme, proved). The Hermitian circulant's spectrum (closed
    form, product multiset) has desymmetrized spacing-ratio <r~> = 0.3964 .. 0.3862 at m11..31
    (130,636,800 exact levels at m31) vs Poisson 0.38629 / GOE 0.5359 / GUE 0.6027: POISSON TO
    FOUR FIGURES, trending toward Poisson; KS to Poisson 0.43 -> 0.0022 (m29); no repulsion
    (P(s<0.1) = 0.094 ~ Poisson's 0.0952 vs GOE's 0.0078). EXACT DEGENERACY LAW: full-spectrum
    tie count = P - prod (q+1)/2 at m11/13/17 EXACTLY (313/4501/80549) - mirror symmetry
    accounts for every degeneracy, zero accidental collisions (desym near-collisions at 1e-12:
    6 at m29, 613 at m31, UNRESOLVED - backlog U5). Bonus exact: det C_M = prod(q-2) = open
    count. Structural reason: any CRT-product spectrum is Berry-Tabor/integrable -> Poisson by
    construction. The round's own hedge - "only the non-tensor sector could carry GUE" - was
    REFUTED by items 35-36 (Refuted 18).

34. PSD DOES NOT BITE - AND WHERE THE SIZE LAW LIVES (r21). (a) MOMENT LP: moments m1..m4 of the
    window occupancy are exact closed forms (no scan; m13..41 including beyond-scan 37/41); LP =
    max # empty windows consistent with moments of order <= K. NEVER BITES: max p0 at K=4 is
    67.6 (m13, W=F) growing to 4.3e10 (m41); at the (D) thresholds W = F_old + q' + 1, 112 to
    1.6e10. Pair level (K=2, the true PSD content of Wiener-Khinchin) is 5-30x weaker still -
    (D) is invisible to K <= 4 occupancy moments, margin GROWING with the machine. (b) EXACT RUN
    CERTIFICATE (the positive result): E(L) = # runs of L consecutive blocked slots by
    inclusion-exclusion with hereditary-zero pruning; E(F) = 0 and E(F-1) > 0 recovered EXACTLY
    as integers at m13/17/19/23 - F = 11/18/25/34 from POSITION LAWS ALONE, NO period scan, with
    only 397 / 5,345 / 46,349 / 578,890 nonzero subsets (vs 2^F up to 1.7e10). Bonferroni
    truncations certify NOTHING short of full depth (first certifying depth k* = max nonzero
    depth + 1). The pruned DFS is a reusable zero-certificate pattern counter.

35. THE NON-TENSOR SECTOR, MEASURED - AND IT GROWS (r22). The right dimension is the SCHMIDT
    RANK across a gear bipartition: CRT makes any function on Z_P a d1 x d2 matrix, rank 1 = a
    product. Max over cuts is a certified LOWER bound on tensor rank; GF(p) rank lower-bounds
    rank over Q, so the growth cannot be an artifact. (a) DEPTH 1 IS BOUNDED - A THEOREM:
    b = 1 - (x)e_q reshapes to J - x y^T with x, y non-constant, so the SCHMIDT RANK OF B IS
    EXACTLY 2 AT EVERY CUT, EVERY MACHINE (same for BS) - the difficulty of F is NOT dimensional
    there. (b) THE MERGE CUT IS LINEAR - A SECOND THEOREM: V[r,k] = prod over i with (k+i) OPEN
    in the old machine of [r+i in T_q'], so the column depends ONLY on the old machine's opening
    pattern O and VANISHES unless |O| <= 2; hence rank_n = [n < F_old] + #singleton classes +
    #literal pairs <= 2n+1, measured == predicted at EVERY row, m11-23. (c) WINDOW DEPTH IS
    UNBOUNDED - THE SPINE ANSWER: with v_n(k) = prod_{i<n} b(k+i), F = min{n : v_n = 0} and
    (BS)^n = diag(v_n) S^n, at the FIXED corridor cut {5,7} (d1 = 35) the peak is 15, 26, 33, 35
    at m13/17/19/23 - it SATURATES; at EVERY fixed cut the peak rises m19 -> m23, five cuts go
    FULL ({5,7} 33->35, {5,11} 48->55, {7,11} 69->77, {11,13} 126->143, {11,17} 140->187), and
    every SINGLE-GEAR cut is FULL from m17 on; certified tensor-rank lower bound TR_low = 6, 17,
    54, 161, 326 at m11/13/17/19/23. THE SECTOR FILLS WHATEVER DIMENSION THE CUT PROVIDES, so
    tensor rank grows like the largest available cut ~ sqrt(P). VERDICT: NO FIXED-ARITY RULE
    EXISTS for the window/realizability content; only an arity-free generator survives. AND THE
    GROWTH IS IN THE NILPOTENT DIRECTION - (BS)^n has spectrum {0} at every depth, so the
    direction that grows has no eigenvalues and no bounded-order correlation signature.

36. PATH-DECOMPOSITION THEOREM AND THE FAREY-CHEBYSHEV SPECTRUM (r22). A = BS + (BS)^T is the
    adjacency matrix of the graph on Z_P with edge {k, k+1} iff k+1 is blocked, so A is the
    DISJOINT UNION OVER THE MACHINE'S GAPS OF PATH GRAPHS - a gap of g slots contributes P_g:

        spec(A) = union over g, with multiplicity W_1(g),
                  of {2 cos(pi j/(g+1)) : j = 1..g}.

    Dense eigvalsh at m11 agrees to 1.3e-15; path bookkeeping (#paths = #openings, sum of
    lengths = P, longest = F) asserted m13-23. COROLLARIES: (i) #distinct eigenvalues =
    |Farey(F+1)| - 2 = sum_{b<=F+1} phi(b) = O(F^2) - measured 21/45/119/211/383/603/1085/2455
    at m11..37 against periods up to 1.2e12, i.e. P/F^2-fold ties on every level; (ii) the
    distinct levels are a smooth image of a FAREY set whose spacings obey HALL's LAW WITH A HARD
    GAP: s_min/s_mean = 0.476, 0.386, 0.340, 0.333, 0.328, 0.321 descending to
    3/pi^2 = 0.30396, P(s < 0.1 mean) = 0 EXACTLY, and <r~> = 0.703 - ABOVE GUE's 0.6027;
    (iii) any diag(w) S^t + h.c. has max degree 2, so every such operator is a union of paths
    and cycles; (iv) the growing-rank operators are nilpotent; (v) the word-level H is triangular
    with an INTEGER diagonal. THE DICHOTOMY: where the spectrum is rich the operator factorises
    (Poisson); where the operator does not factorise the spectrum is degenerate or empty. GUE is
    bracketed THREE times and hit zero: clock 1.000 > Farey-Chebyshev 0.703 > GUE 0.603 > GOE
    0.536 > Poisson 0.386. THE RIEMANN BRIDGE IS CLOSED at finite machines, with a reason rather
    than a statistic.

37. THE CORRIDOR RESONANCE IN CLOSED FORM (r22). Exact input (CRT, no fit): openings are exactly
    equidistributed over the exposed phase set E mod m, so the per-slot hazard is EXACTLY
    h(r) = rho [r in E]. One modelling step (slot independence) makes the phase chain
    M = (I-B)^{-1}O with B = S D_{1-h}, O = S D_h, whose CHARACTERISTIC POLYNOMIAL is that of a
    weighted single m-cycle:

        lambda^m = prod_s [lambda(1-h(s)) + h(s)] = lambda^{m-e} [(1-rho) lambda + rho]^e,
        e = |E| = prod_{q|m}(q-2),

    so lambda = 0 with multiplicity m-e and otherwise LAMBDA_j = rho w_j / (1 - (1-rho) w_j),
    w_j = e(j/e). THE SPECTRUM IS A MOEBIUS IMAGE OF THE e-TH ROOTS OF UNITY, hence lies on ONE
    CIRCLE |z - (1-rho)/(2-rho)| = 1/(2-rho) through 1. The resonance is mod 15, not mod 35
    (e = |A_5||A_7| = 15, the walk never visits a blocked phase) - which is why the measured
    period is near 8 and not near 17. Measured vs closed form (exact full-period chains, m11-23)
    agree within 0.008-0.025 in modulus and 0.20-1.71 deg; mod 385 (e = 135) matches arg to
    0.001 deg. Scan-free predictions m29/31/37/41 mod 35: |l2| 0.8366/0.8118/0.7900/0.7696, arg
    +47.09/+49.44/+51.40/+53.17, periods 7.65/7.28/7.00/6.77 lags - THE RESONANCE PERIOD
    SHORTENS with the machine; it is not a fixed "period 8".

38. THE lambda_2 DEFICIT IS THE NON-GEOMETRICITY OF THE STEP LAW (r23, closing item 37's 0.029
    deficit). With q(n) the EXPOSED-STEP LAW (how many exposed phases mod m one gap crosses),
    lambda_2 = q-hat(1/e) = sum_n q(n) e(n/e) TO 1e-5 against the exact full-period chain: in
    the exposed-step coordinate a phase-blind chain is an EXACT CIRCULANT on Z_e, so its
    eigenvalues are exactly q-hat(j/e) and the 1e-5 residual is the whole phase-dependence.
    Round 22's Moebius form is exactly q-hat with q GEOMETRIC: THE DEFICIT WAS A COORDINATE, NOT
    A CORRELATION. Deficits (mod 35, m11..19): M0 (geometric) +0.0076 -> +0.0225; M3 (exact gap
    law, phase-blind) ~+0.0175 flat; M4 (exact STEP law) 0.0000 .. 0.0001. The step law's mean
    is EXACTLY 1/rho at every machine (CRT identity - so the deficit is pure SHAPE) and its
    first term is closed form, q(1) = avg over r in E of prod_{q not | m} c_q(d(r))/(q-2) with
    d(r) the slot distance to the next exposed phase (verified to 2.2e-16 at m11-19). Shape:
    q(n)/geometric(n) is SUPPRESSED at n=1 (0.951 -> 0.890), ENHANCED at n=2 (1.494 -> 1.247),
    then decays. The corridor pinning is a one-dimensional measurable object.

39. JORDAN = GAP HISTOGRAM, AND ITS CONSEQUENCES (r23; a theorem plus a negative). N = BS acts
    by N e_k = b(k+1) e_{k+1}, so its directed graph is the disjoint union of the chains of
    consecutive blocked slots and

        N is PERMUTATION-similar (hence unitarily equivalent) to
        (+)_g J_g^{(+) W_1(g)} - ONE NILPOTENT JORDAN BLOCK PER GAP.

    Equivalently rank(N^n) = sum_g W_1(g)(g-n)_+ (the histogram TAIL SUM); #blocks of size
    exactly L = W_1(L); largest block = F. EXACT INTEGERS m11-19, with the permutation built
    explicitly at m11/13 and the permuted matrix asserted EQUAL to the block sum entry by entry.
    THE NEGATIVE (the real content): EVERY UNITARY INVARIANT OF N IS A FUNCTION OF THE GAP
    HISTOGRAM ALONE - singular values, all Schatten norms, Jordan type, kernel-filtration
    dimensions, numerical range, resolvent norms, pseudospectra. None can bound F except
    circularly. Wall V in invariant-theoretic form; it UPGRADES item 36 (A = N + N^T as a union
    of paths P_g is the symmetrised shadow of the same decomposition, same index set).
    WHAT THE INVARIANTS STILL BUY, three, each turning F into a different kind of quantity:
    (i) NORM CLIFF. N^n = diag(v_n) S^n is a PARTIAL ISOMETRY (singular values 0/1), so
    ||N^n||_op = 1 for n < F and 0 for n >= F - a step function with no decay rate at all; any
    envelope ||N^n|| <= C lam^n (lam < 1) forces C >= lam^{1-F}: F SITS ENTIRELY IN THE
    CONSTANT. That is why every analytic decay frame has stalled.
    (ii) NUMERICAL RADIUS. w(N) = cos(pi/(F+1)) EXACTLY and the numerical RANGE is that disk
    (direction-independence to 6.7e-16 at m11), so F = pi/arccos(w) - 1: THE MAXIMAL GAP IS A
    VARIATIONAL, SDP-REPRESENTABLE QUANTITY and every upper bound on it has a dual certificate
    (checked two-sidedly m11-19 with no eigensolver, path Perron weight as an exact Schur test).
    (iii) PSEUDOSPECTRUM / MASLOV BRIDGE. Spectrum {0}, but ||(zI-N)^{-1}|| = |z|^{-F}(1+O(|z|))
    and r_eps = eps^{1/F}(1+o(1)); recovered exponent 25.782 -> 25.107 -> 25.005 at m19 for
    eps = 1e-6/1e-12/1e-24, monotone from above. With z = e^{-1/t} this is MASLOV
    DEQUANTISATION: t log||(zI-N)^{-1}|| -> F, i.e. THE (+,x) RESOLVENT COMPUTES THE (max,+)
    LONGEST PATH - Constructor's Kleene star, the Boolean window filtration and the analytic
    resolvent are ONE computation in three semirings.
    WHERE THE NON-INVARIANT CONTENT LIVES: ker N^n is a COORDINATE subspace, so the kernel FLAG
    is a nested family of SUBSETS of Z_P; its dimensions are histogram tail sums (circular)
    while its POSITION against the CRT gear basis is not a unitary invariant at all - and that
    position is exactly item 35's Schmidt-rank profile, the part that GROWS. Rounds 22 and 23
    fit exactly: invariants = histogram, growth = alignment of the kernel flag with the gear
    tensor basis.

40. THE POTENTIAL / CERTIFICATE ARITY LADDER (r23). A certificate is not an invariant, so it
    escapes item 39. For h: Z_P -> R with

        (*)  h(k) - h(k-1) >= 1 at every BLOCKED slot,   then F <= 1 + osc(h),

    and it is TIGHT (h = distance back to the previous opening gives osc = F-1 exactly, m11-19).
    Multiplicative form w = exp(h/t) is a SCHUR TEST on A; the tropical limit is the max-plus
    potential. The frame loses nothing - only ARITY can fail, and an infeasibility verdict rules
    out EVERY certificate of that arity.
    T1 (one line, PROVED): a potential depending only on k mod m for a PROPER divisor m of P is
    INFEASIBLE - every class mod m contains a blocked slot (asserted m = 35, 385 at m11-19), so
    (*) forces h(r) > h(r-1) all round the m-cycle and 0 >= m. A state that has forgotten a gear
    cannot see that a slot is blocked - this is why bounded-state certificates mod 35/385/5005
    cannot bound F.
    T2 (MERTENS NO-GO, PROVED, exact rationals): a LEVEL-1 (per-gear) potential exists only if
    sigma(y) < 1/2. Proof by two CRT averages (over all slots, and over "gear q at a tooth with
    all others exposed") giving Sigma(1 - 2 sigma) >= 2 sigma with Sigma > 0.
    sigma(11) = 167/385 = 0.4338 but sigma(13) = 2556/5005 = 0.5107, and sigma DIVERGES:
    ARITY-1 CERTIFICATES DIE AT MACHINE 13 AND NEVER RETURN.
    MEASURED LADDER (LP; every FEASIBLE verdict re-checked by rebuilding h and testing (*) at
    every blocked slot over the full period):
        y=11 F=7 : arity1 23.902 (3.41x), arity2 7.753 (1.11x), arity3 7.000
        y=13 F=11: arity1 INFEASIBLE, arity2 17.980 (1.63x), arity3 11.000, arity4 11.000
        y=17 F=18: arity1 INFEASIBLE, arity2 37.102 (2.06x)
        y=19 F=25: arity1 INFEASIBLE (1,237,940 rows), arity2 FEASIBLE (certificate found on a
                   4,836-row subsample then VERIFIED against all 1,237,940 blocked slots, min
                   step 1.0000; bound <= 195.5, not the optimum)
    TWO FACTS: arity 1 dies exactly where T2 says; and WHERE A FIXED ARITY SURVIVES ITS QUALITY
    DECAYS (1.11x, 1.63x, 2.06x at m11/13/17) - a fixed-arity certificate becomes asymptotically
    VACUOUS while remaining FEASIBLE, so feasibility alone is the wrong statistic to watch.
    THE THRESHOLD LAW (conjectured; derivation and its gap both stated): the same averaging at
    level r gives sum_U |U| a_U <= (2A-2) sigma, closing to a contradiction when sigma >= r/2
    PROVIDED the a_U are non-positive - THE SIGN CONDITION IS THE GAP, named not hidden. Taken
    at face value LEVEL r DIES AT sigma(y) >= r/2, fitting every measured cell, with doubly
    exponential thresholds: level 1 at y=13, level 2 at y=109, level 3 at y=2741, level 4 at
    y=483281. So r*(y) ~ 2 sigma(y) ~ 2 log log y - UNBOUNDED, doubly logarithmically slow.
    Written down BEFORE the m19 arity-2 cell resolved and predicted it correctly; fits 8 of 8.
    THE CONVERGENCE THAT MATTERS: the project's LP-DUALITY thread, on a completely different
    certificate family (covering/Farkas duals for (D) rungs, not potentials for F),
    independently found required degree ~ 2*S1(y) with the same reciprocal-prime sum. TWO
    UNRELATED CERTIFICATE FRAMES, THE SAME ARITY LAW r* proportional to sum_{q<=y} 1/q. "No
    fixed-arity rule" now has an arithmetic source: sum 1/q diverges.

41. TWO CHECKED NON-GAINS ON THE PATH DECOMPOSITION (r23; recorded so they are not rebuilt).
    (a) MOMENTS REDUCE TO THE RUN LADDER: tr(A^{2t}) = sum_L m_t(L) r_L where m_t(L) counts
    closed 2t-walks on Z of RANGE L and r_L = rank(N^L) - a closed walk's support is an
    interval, so it demands exactly an L-run of blocked slots (verified t = 1..6 at m11). EVERY
    trace/moment - equivalently every exponential-sum - attack on lambda_max(A), hence on F, is
    a POSITIVE COMBINATION of the r_L ladder item 34(b) already computes scan-free. (b) WEYL ON
    THE MERGE STEP IS VACUOUS: the longest run of consecutive newly-blocked slots is 1 at every
    step 11->13 .. 19->23, so lambda_max(Delta) = 1 and the Weyl bound is 2.848, 2.932, 2.973,
    2.985 - above 2 at every step. The merge step's content is WHICH edges are added, never how
    many.

42. LP(MF) = CLOSURE EXACTNESS THEOREM (r24). Constructor's machine-free system MF_m is a
    max-plus closure = longest-path problem, and its natural LP (potentials p with p_u >= R_u,
    p_u >= w_e + p_dst) has optimum EXACTLY the closure value: the closure is the least fixed
    point and every feasible p dominates it by induction. Verified 12/12 to 0.00e0. COROLLARY:
    every relaxation sandwiched between the LP and the truth - every Lasserre level, every SDP -
    returns exactly the closure value. NO CONVEX RELAXATION OF THE MACHINE-FREE SYSTEM CAN
    IMPROVE IT BY ONE UNIT. Its 125-vs-74 gap at 29->31 is 100% EDGE SET (support), 0%
    relaxation gap - exactly why CEGAR (deleting unrealised tuples) was the only lever that ever
    moved it.

43. THE COVERING HIERARCHY: EXACT TO m17, BREAKS AT m19 ON ARITY (r24). F(M) = 1 + max coverable
    L is a covering CSP over one free offset per gear (no period, no scan); the convex hierarchy
    over THAT is machine-free by construction.
    (a) LEVEL 1 (fractional cover) dies exactly at sigma(y) >= 1/2 - T2's Mertens threshold
    reappearing on the covering side. Machine 11 only.
    (b) LEVEL 2 (pairwise moment LP over offset literals, matrix 1 + sum_q q, conditional
    covering with slack objective V(L)) IS EXACT AT MACHINES 11, 13, 17: L* = min{L : V(L) > 0}
    = F exactly (7, 11, 18). A POLYNOMIAL-SIZE MACHINE-FREE LP COMPUTES THE JACOBSTHAL-TYPE
    MAXIMUM, three machines running. EXACT RATIONAL DUAL CERTIFICATES (weak duality,
    integer-checked): 479/1152, 1041/2081, 1673/19767.
    (c) THE BREAK IS AT MACHINE 19, AND THE MECHANISM IS ARITY: L*(19) = 27 vs F = 25, exact
    dual 2927/270613 at L = 27; V(25) = V(26) = 0. The PSD constraint does NOT repair it: at the
    truly-impossible L = 26 the cutting-plane loop CONVERGES to a PSD moment matrix with V = 0
    (min eig -1.2e-14, 39 cuts) - the full level-2 SDP is FEASIBLE at an impossible length
    (numerical, flagged) - and at L = 25 it stalls (187 cuts). SO EVERY CERTIFICATE OF
    F(19) <= 26 REQUIRES THREE-GEAR INFORMATION; pairwise reasoning, linear or semidefinite,
    provably (numerically at 26, measured at 25) cannot see the obstruction.
    (d) VACUITY GROWTH, the same law from a THIRD family: L*/F = 1.000, 1.000, 1.000, 1.080,
    1.647, >= 1.721 (m23: L* = 56 vs F = 34, exact dual 3427/746861 at L = 56; m29: V = 0 at
    every tested L <= 73 under a 5400 s cap, so L* >= 74). DRIFT, not infeasibility - the same
    failure axis as item 40's ladder and the LP thread's degree law. The exactness margin V(F)
    collapses first: 5/6, 1, 0.169, 0.
    VERDICT: THE SDP DOES NOT BITE, and the reason has two halves, both proved or measured - on
    the machine-free system there is NO relaxation gap to close (42), and on the covering CSP
    the gap that opens at m19 is an ARITY gap PSD cannot cross. There is no small
    arity-independent convex statement bounding osc(h); the smallest arity-independent
    statements are the exact ones (level-2 at m11-17) and they stop existing at m19.

44. THE NORM CLIFF IS A REDUCTION, NOT A TECHNIQUE (r24; exact integers m11-17). For weighted
    sup-norms ||x||_w = max |x_k|/w_k, one line gives ||N||_w = max over blocked k of
    w_{k-1}/w_k, and with w = 2^h the envelope constant is C = 2^{osc(h)}, rate
    lam = 2^{-min step}, so 1 + log C / log(1/lam) = 1 + osc(h)/(min step) - IDENTICALLY the
    item-40 potential bound (verified exact, = F at every machine, tight h). ANY envelope bound
    in a weighted norm IS a potential certificate of the same oscillation; any envelope in a
    unitarily invariant norm is histogram-circular by item 39. "F sits in the constant" converts
    into NOTHING NEW: extracting the constant is re-deriving a potential, and the arity ladder
    already prices those.

45. THE MASLOV BRIDGE IS AN ISOMORPHISM OF BOUNDS, NOT AN AMPLIFIER (r24, follows from 44). It
    transports bounds between the three semirings at the SAME oscillation - item 44 is the
    dequantised identity, computed. Nothing to push through it until someone proves a
    max-plus-side bound that does not come from a potential.

## Refuted angles (do not retry without new input)

1. Umbrella nesting (M2) as a twin-specific mechanism (r1): any two gears' short umbrellas are
   concentric at joint shields; the only twin-specific part IS result 1's pinned classes.
2. Closing the recursion by tooth-sharing COUNT (r1): net gain O(T(y)) per window vs the needed
   ~K/log^2, and the guaranteed wasted kills land on already-decided slots.
3. Phase-vector extremality (r2): full enumeration - the real vector is top 10-25% on waste
   metrics, never extremal beyond the degenerate 2-gear mirror space. No variational handle.
4. Matched-real-primes RICH/POOR design (r1): confound-dominated by kill-density mismatch.
5. Drift recursion "new max address = f(old top-stratum address)" (r10): 0/4 and 1/2
   reachability at 19->23, 23->29. The honest law is LOCAL: address = pin(word).
6. A-priori stabilisation of the near-top word-SHAPE family (r11): the CRT-admissible superset
   (3798 half-shapes) is finite, but cross-machine full-shape recurrence is ZERO, max flank part
   grows 7 -> 13 with y, observed halves 3.2% of admissible and disjoint per machine.
7. "1 of 4 fuel sites fire" (r11, withdrawn r12): one-window artifact; every site fires exactly
   once per new-machine period.
8. Fuel x alignment "double rarity" multiplier (r12): alignment is a density factor, never a
   count factor.
9. Literal-only asymptotic safety of lemma 2 (r13, withdrawn same day): the cap-6 theorem covers
   LITERAL chains only; padded runs escape it. The crossover is padding onset.
10. "The ceiling stands on structure" (r16, corrected r17): the SHAPE law is permanent but the
    COUNT p is capped only by p <= F/q' + alpha/3, which grows; span <= F + O(q'), not O(q').
    p <= 2 is NOT provable from the AP lemma.
11. Covering/capacity explanation of absent gaps (r18): residual interior demand has positive
    slack (8-16 spare kills) at every g at both machines tested. Gap 24's absence is arithmetic
    selection plus rarity, NOT impossibility.
12. Smooth supply^2/gaps prediction of padding events (r15): padding switches on/off with
    q' mod 35; it predicted ~5 double-padded runs at 37->41 where the corridor forbids the
    adjacent shape outright.
13. "Pair interaction reduces per-cell to endpoint arithmetic x singles" (r20): log-sd ~1.2 and
    drifting bias; only the count-weighted average factorises (30a).
14. "The full closed-form predictor explains the padding supply" (r20): 15x spread remains.
15. "kappa(g*density) is a universal curve" (r20): first-order only (28d).
16. "(D) might be corridor-forced at n = 4" (r20): decisively no - 0 of 1225 (31).
17. THE POLE-PHASE PIN. "+126 deg is an asymptotic arithmetic invariant" (r21): REFUTED BY ITS
    OWN DRIFT MODEL - the closed-form predictor that reproduces every measured phase to
    +-1.5 deg crosses 126 at ~m31-47 and then drifts (124.6 at y=97, 117.6 at y=499). The LAW
    (item 32) stands; the PIN does not - pole phase plus a plateau, decidable at m41/43 (model
    125.5-125.9). Also refuted that round: the M2 corridor-hardness beta-model as the sole phase
    mechanism (phases -163..-169 deg at every beta for gear 5).
18. GUE ANYWHERE, INCLUDING THE NON-TENSOR SECTOR. r21 refuted GUE for the tensor operators but
    hedged that "only the non-tensor sector - nilpotent BS, H's non-triangular part - could
    carry it". THAT LOCALISATION IS REFUTED BY THIS LANE'S OWN LATER MEASUREMENT: the sector's
    Hermitian operators are unions of paths (item 36 - Farey/Chebyshev, <r~> = 0.703, MORE rigid
    than GUE, Hall hard gap, P(s<0.1) = 0 exactly) and its high-rank operators are nilpotent,
    spectrum {0}, no statistics at all (item 39: every unitary invariant of the nilpotent part
    is the gap histogram). GUE bracketed three ways, hit zero. The Riemann bridge is CLOSED at
    finite machines; do not re-open it with another operator from this family.
19. PSD / bounded-moment bite on (D)-violating windows (r21): margins 67.6 .. 4.3e10, GROWING.
20. "The lambda_2 deficit is a 2- or 3-point CRT anti-correlation effect" (r23, my own
    pre-registered hypothesis): refuted IN THE SIGN by my own script - the 2-point
    conditional-hazard model worsens the deficit by 52-67%, the 3-point interior model by
    83-117%. They corrected the SLOT-LAG hazard, double-counting phase structure the corridor
    already carries (38).
21. "The corridor-renewal model explains the pole-bracket phase" (r22): refuted, WRONG IN THE
    SIGN OF DRIFT at both gears. Pre-registered claim "arg B(5,1) is identically zero in the
    renewal model" falsified in the same script: the model puts arg B(5,1) at +11.0 -> +14.2 at
    the machines' own a = 1-rho values (moving AWAY from 0 while the machine moves TOWARD 0) and
    arg B(7,1) nearly flat at -19.5 -> -15.0 while the machine climbs through it. USEFUL
    NEGATIVE: the same one-parameter model settles the MEAN-hazard quantity (lambda_2, to 1-2%)
    and fails on the FINE PHASE quantity - the two observables separate cleanly.
22. "The non-tensor sector is small" (r22): at depth 1 yes (rank exactly 2, a theorem) but at
    window depth it SATURATES whole gear cuts (35).
23. "Singular values / Jordan structure / pseudospectra could carry what the spectrum cannot"
    (r23 brief's candidate list): refuted AS A CLASS - all unitary invariants, and every unitary
    invariant of BS equals the gap histogram (39).
24. Moment / exponential-sum bounds on lambda_max(A) (r23): reduce exactly to the r_L run ladder
    - no new information (41a).
25. Weyl / eigenvalue-perturbation across the merge step (r23): vacuous, 2.85-2.99 vs 2 (41b).
26. "An SDP relaxation of the machine-free system is the natural next object" (r24 brief's
    premise): the object exists and provably equals the LP equals the closure (42). The gap
    there is pure support.
27. "PSD strictly improves the pairwise covering LP" (r24, my own P4): refuted at every tested
    cell; at m19 L = 26 the SDP is FEASIBLE at an impossible length (43c).
28. "The norm cliff converts into a lower-bound technique" (r24 brief item c): refuted by exact
    identity - it reduces to the potential/arity ladder (44).
29. My own pre-registered "r*(19) >= 3" (r23): REFUTED - arity 2 IS feasible at m19, so
    r*(19) = 2. The correction is the threshold law itself: r* grows only when sigma crosses the
    next half-integer, i.e. DOUBLY LOGARITHMICALLY. Right in direction, badly wrong in rate, and
    the rate is the point.

## Untested backlog (untested is NOT dead)

ROUND-25 PRIORITY - the lane's own mandate, restored (see the mandate audit). These five are
named in the round-25 brief as this lane's territory and have been carried, unworked, through
the three route-support rounds:

U1. THE m29 DEPTH SPIRAL. The depth-family phase spiral sum_j What_j(omega) was measured only to
    m23; its ANCHOR is a proved identity ((2-phi) prod_{q!=5}(q-2)^2 - N, REAL) and W_2's arm
    climbs toward the pole phase with machine size (66.5 -> 87.7 -> 113.2 at m17/19/23). m29 is
    the next rung and nobody has run it.
U2. THE AMPLITUDE NEAR-LAW. |H_5(1)|/H0 * mean_gap = 1.010..1.037 (1.015 +- 1%, no trend across
    eight machines) - a clean measured near-invariant with NO explanation. Unlike the phase it
    has not even been modelled.
U3. GEAR 7's DRIFTING BRACKET vs GEAR 5's REAL ONE. Why is arg B(5,1) real to +-0.4 deg while
    arg B(7,1) climbs -2.41 -> +14.31 over m11-23? The closed form REPRODUCES both but derives
    neither, and the renewal model is refuted in the sign (Refuted 21), so the answer lives in
    the interior/kappa correlation. Narrowed: why does the interior correlation cancel the
    endpoint phase at p = 5 and not at p = 7? Finite per machine.
U4. FAREY-SPECTRUM CONSEQUENCES. Item 36 gives an exact, closed-form, Farey-indexed spectrum
    with Hall's law and a HARD gap (s_min/s_mean -> 3/pi^2, P(s<0.1 mean) = 0 exactly, distinct
    level count = sum_{b<=F+1} phi(b)). NOTHING has been drawn from it: the level count is
    O(F^2) against periods of 1.2e12, the multiplicities ARE the gap histogram, and the Farey
    structure is a genuinely number-theoretic object this lane produced and then left. Named,
    not built.
U5. THE 613 COSINE NEAR-COLLISIONS. In the desymmetrized spectrum, 6 near-collisions at m29 and
    613 at m31 at tolerance 1e-12 survive after the EXACT degeneracy law
    (ties = P - prod (q+1)/2, exact at m11/13/17) has accounted for every mirror-symmetry tie.
    Algebraic cosine-product coincidences - finite, checkable, unexplained.

Also standing (the one route-adjacent item the round-25 brief permits, because it is this lane's
own): THE MIRROR LAW as a structural constraint - openings are closed under k -> -k; what else
does exact mirror symmetry force about ADJACENT gaps? A reframing question.

CARRIED FROM EARLIER ROUNDS (all still open):

B1. Joint-necessity census (r1): twin pairs jointly own a pseudo-twin at the product slot when
    p(p+2)+2 is prime (p = 5, 149, 179, 239, ...); never censused vs generic pairs.
B2. Jacobsthal push (r8): does L0 = 32 survive gears <= 100; is lim L0 = 32?
B3. Medium-medium adjacency at word level (r9-r10): can two near-top words sit adjacent - a
    finite CRT check per word pair on the pinned phase sets (word lists in address_drift.py).
    Converted from a period scan to grammar-level arithmetic but never run. If pinned classes
    can never be adjacent, alpha1 follows per machine.
B4. Extreme-value grammar (r10): characterise a priori which words CAN be near-top - would make
    the alpha1 adjacency check machine-independent. The open half of law A.
B5. Excess-share pricing vs fuel census (r12): is excess/q' ~ c log(N3)/q' or spectrum-driven?
    Needs machine 37/41 spectra.
B6. Deficit >= 0, i.e. FS_max(w) <= F2 for every word (r13): 13/13 observations positive but
    unproved. If proved, lemma 2 reduces to lemma 1 for the literal part.
B7. The 37->41 knife-edge (r15-r17): the only surviving double-padding shape needs
    F_3(37) >= 96 against a 9.7%-prefix value of 95. One unit decides the census.
B8. Gear-7 AP extension (r16): does gear 7 (exposes 5 of 7) forbid SIX openings in q'-AP?
B9. Cheapest surviving p-shape cost (r17): if it (finite per q' mod 210) grows faster than
    F_j(M), p is capped structurally after all.
B10. Persistence(L) empirics (r7): L = 13 witnessed only at the absolute landmark; its next
    Bertrand band unexplored; the L = 14 language has 579 words.
B11. Mechanism of exact kappa(4) = 1 (r20): a cross-gear cancellation (mirror symmetry
    suspected). Kernel-checkable per machine.
B12. kappa's log-convex TAIL as the residence of extreme-value structure (r20): fit kappa on the
    top decile of g, relate curvature to F - the Wall V content of that decomposition.
B13. Lag >= 2 pair prediction (r20): a transfer-matrix product over the closed-form 3-point
    kernels. The kernels are ready.
B14. The extra qualifying suppression (r20; x2.4-x6 and shrinking): constant or -> 1? Needs the
    m37/41 joint census.
B15. A genuine large-sieve inequality on W_j from the exact spectrum (r20): the power spectrum
    is closed-form, so a Beurling-Selberg-style bound on window counts is a finite construction.
    Named, not built.
B16. Lipschitz / transfer strengthening of the moment LP (r21): joint (f(x), f(x+g))
    distributions, all closed-form.
B17. Is rank_n = min(2^n, d1, d2) EXACTLY in a range of n - is the sector generically full?
    Peaks reach 35/35 at {5,7} but 326/391 at {17,23}; the deficit's law is unknown (r22).
B18. The rank profile's PEAK DEPTH (6, 8, 10, 11 at cut {5,7} for m13/17/19/23) against F (11,
    18, 25, 34) - the peak sits near 0.4F and drifts; a function of the mean gap? (r22)
B19. Machine-29 rank profile (P = 1.08e9; needs a streaming/bitset build) (r22).
B20. Whether the corridor lambda_2 residual saturates (+0.027) or grows - needs the m29 corridor
    chain, a full-period pass (r22). Pre-registered value in the scorecard.
B21. The phase-dependence residual of the step law (1e-5, growing 0 -> 6.5e-5 over m11..19):
    O(1/e^2) or growing? (r23)
B22. Prove or refute the sigma >= r/2 arity threshold for r >= 2 - THE SIGN CONDITION ON THE a_U
    IS THE WHOLE GAP, a finite statement per r. Sharpest tests: the m19 arity-2 OPTIMUM (scoped
    out on cost - 1.24M rows x 30 nonzeros exceeds memory, row generation did not converge in
    budget) and the level-2 death predicted at y ~ 109. (r23)
B23. TRANSPORT THE CERTIFICATE ACROSS A MERGE STEP: h_new from h_old plus a gear-q' part would
    be the merge law in certificate form, and the arity ladder says how much room there is.
B24. The numerical-radius SDP: does it admit ANY tensor-structured dual certificate? (The arity
    ladder is the LP shadow of exactly this question.) (r23)
B25. Whether a bound proved in the max-plus semiring dequantises to a usable resolvent bound -
    item 39(iii) makes the dictionary exact; nothing has been pushed through it.
B26. Partial LEVEL 3 (triple moments on chosen gear triples) at m19, L = 25/26: does arity 3
    suffice, and WHICH TRIPLE is the obstruction? One targeted lift, not a full level-3 - the
    sharpest next probe of the arity law (r24, not built; round interrupted).
B27. Exact rational PSD completion of the m19 L = 26 converged point (facial reduction) - would
    upgrade the SDP-feasibility verdict from numerical to exact (r24).
B28. The exactness margin V(F) = 5/6, 1, 54752/323401(num), 0: closed form? Its collapse between
    m17 and m19 is the break's shadow and nothing explains its VALUES (r24).
B29. m29 L* (>= 74): each V(L) evaluation is 400-2000 s at the current implementation; a
    sparse/dual-simplex reformulation would reach it (r24).
B30. Level-2 exactness at m19 for SHORTER impossible lengths: V first turns positive at 27;
    whether another formulation of the same level (e.g. the part-C g(i) witness variables at
    scale) moves 27 to 26 is untested - part-C runs at m17+ were stopped for cost (r24).

KERNEL / LEAN HANDOFFS proposed from this lane (status tracked elsewhere): the horizon theorem
(3 lines: any 33 consecutive slots contain k = 1 or 34 mod 35); the per-slot identity
P = t + T - B + U; the AP lemma; the adjacent-gap exclusion law (would be the first
forbidden-configuration theorem in the Lean ledger); the depth-sum identity (DepthSum.lean).

## Prediction scorecards (pre-registered, then scored)

BANKED / STILL OPEN (r15-r17 padding thread; status tracked in the constructor/mechanic logs):
- NO double-padded run at 37->41 (the corridor forbids the adjacent shape).
- FIRST double-padded run at 41->43 (adjacent shape corridor-allowed and spectrum-guaranteed).
- F(41) discriminator: <= 100 (saturation) vs >= 103 (climbing); expectation CLIMB.
  Lower-biased, literal-derived.

ROUND 21:
- Eigenphase statistics will be POISSON, not GUE: CONFIRMED (0.3862-0.3964 vs Poisson 0.38629;
  KS 0.0022 at m29). A test result, not a surprise.
- Pole-phase PIN vs DRIFT: decidable at m41/43, model predicts 125.5-125.9. OPEN. (The pin as
  an asymptotic invariant is already refuted - Refuted 17.)

ROUND 22:
- "arg B(5,1) is identically zero in the corridor-renewal model": FALSIFIED in the same script;
  the model is wrong in the SIGN OF DRIFT for both gears 5 and 7.
- Machine-29 corridor chain mod 35 will measure |lambda_2| = 0.862 +- 0.004,
  arg = +49.2 +- 0.4 deg (closed form 0.8366 / +47.09; residual positive at every machine with
  decelerating increments 70, 46, 33, 20 e-4). OPEN - needs a full-period m29 pass.

ROUND 23:
- P2 "r*(19) >= 3": REFUTED by my own LP (arity 2 feasible; r*(19) = 2). Right in direction,
  wrong in rate; the correction became the threshold law.
- The sigma >= r/2 threshold law was written down BEFORE the m19 arity-2 cell resolved and
  PREDICTED IT CORRECTLY (2 sigma(19) = 1.244 < 2); fits 8 of 8 cells.
- Two pre-registered lambda_2 correction models: BOTH REFUTED, and refuted in the SIGN.

ROUND 24 - eight predictions written into sdp_cover.py docstrings before the relevant runs
(P6-P8 after m17 was known exact and V(24) = 0 known at m19, before V(25) resolved). SCORED
5 CONFIRMED - 2 REFUTED - 1 UNRESOLVED:
  P1 LP(MF) = closure: CONFIRMED 12/12 (it became item 42's theorem).
  P2 LP1 dies at sigma >= 1/2 from m13: CONFIRMED.
  P3 SA2 finite at every machine 11..23: CONFIRMED (56 at m23); UNRESOLVED at m29 (cap hit at
     L = 73 with V still 0).
  P4 SDP2 strictly improves SA2 somewhere: REFUTED everywhere tested.
  P5 SA2 at m13 within 2x of truth: CONFIRMED (exact, 1.00x).
  P6 SA2 exactness breaks at or below m37: CONFIRMED, at m19.
  P7 PSD recovers >= 1 unit at the break: REFUTED (the break survives PSD).
  P8 soundness L* >= F always: CONFIRMED (V(F-1) = 0 anchor asserted at every machine).
Both refutations are this lane's own predictions, killed by its own scripts - house standard.

## Reproduction pointers

All scripts under research/, assertion-gated; interpreter
C:\dev\primes\.venv\Scripts\python.exe (or `uv run`). Logs and CSVs under research/data/.

Rounds 1-19 (script -> round): tooth_sharing.py r1; overcount_census.py r2; split_gap_law.py r3;
supply_formula.py r4; derivative_scan.py r5; load_frontier.py r6 (also the persistence decision
procedure); alternation_words.py r7; word_grammar.py r8 (horizon theorem, language census,
757-run check; input data/satruns_ge10.csv); topgap_corridor.py + topgap_nesting.py r9;
address_drift.py r10 (word-pinning law, drift refutation - the B3/B4 word groups live here);
word_shapes.py + k4_pinning.py r11; firing_ratio.py, firing_law_check.py, firing3137.py,
graded_constant.py r12; merge_correct.py (the right condition, six-step verification),
merge_decompose.py (literal-only, undershoots), merge_general.py (overshoots), excess_law.py,
excess_predict.py, merge3137.py r13; padding_bound.py, padding_horizon.py, padding31.py r14;
padding_37_41.py, padding_corridor_law.py r15; corridor_shapes.py, corridor_ap_lemma.py r16;
corridor_complete.py, padding_onset.py r17; exposed_autocorr.py, residual_demand.py,
autocorr_fit.py, openings_ap.py, openings_ap2.py r18; npoint_autocorr.py, lagpair_predict.py,
lambda_law.py, padded_lag.py, bonferroni_gap.py, ie_pruning.py r20-tag.

Round 20: depth_identity.py (identity + exact W_j tables -> data/depth_identity_<y>.csv),
renewal_law.py (decomposition, wiggle test, kappa, padded lag), machine_dft.py (spectrum, T3,
golden gap, line collapse), pair_renewal.py (exclusion at m31, factorisation, qualifying R),
fs_corridor.py (0/1225 law, 51.6% split blocking), holt_correspondence.py.

Round 21: c14_phase.py (parts 1-6: phases, golden constraint, depth spiral + closure, models
M0/M1/M2, asymptotic sweep, pole-phase decomposition; logs data/c14_asym2.log,
data/c14_ladder.log), eig_stats.py (--big for m31; log data/eig_big.log), psd_bite.py
(--deep-only for the m23 DFS; log data/psd_deep.log). Inputs data/gap_pair_hist.csv (Mechanic),
data/depth_identity_*.csv.

Round 22: nontensor.py (--big adds m23, log data/nontensor_big.log), nontensor_spec.py (--big
adds the m29/31/37 distinct-level and Farey rows), corridor_lambda.py (--big adds m23),
bracket_why.py.

Round 23: nilpotent_invariants.py (parts 1-8; log data/nilpotent_invariants.log),
potential_arity.py (T1/T2 + tightness + ladder; a "y:arity[:row_stride]" argument runs one cell;
solve_cutting does row generation; log data/potential_arity.log), lambda2_pair.py (models 0-4
with pre-registered verdicts; log data/lambda2_pair.log).

Round 24: sdp_cover.py - partA (MF-LP theorem check, 12/12), partB (CSP sanity + LP1 + sigma
table), partC (big lift with g(i) witnesses, m11 only at practical cost), partD (small-lift
first-certificate scan), partE (L* ladder by doubling + bisection, exact duals), partF
(norm-cliff reduction, exact integers), partG (PSD cutting-plane loop at chosen (y, L));
exact_dual_bound2 = the weak-duality rationaliser (strictly-feasible dual, floor at denominator
D, integer-exact check). Logs data/sdp_ladder_a.log, sdp_ladder_b.log, sdp_psd_19.log,
sdp_cover_17_19.log, sdp_exact_23.log. NOTE: data/survivor_31.log is NOT this lane's file
(survivor_generator.py, another lane).

Novel-register docs owned by this lane: docs/novel/depth-sum-identity.md, golden-spectral-gap.md,
pole-phase-law.md, eigenvalue-statistics.md, nontensor-sector.md, farey-chebyshev-spectrum.md,
corridor-eigenvalue-closed-form.md (incl. the round-23 update section), nilpotent-invariants.md,
potential-arity-ladder.md, covering-hierarchy-exactness.md.

Cross-lane data cited: data/satruns_ge10.csv, data/gap_pair_joint.csv, data/gap_pair_hist.csv,
data/gap_histograms.csv, docs/forbidden-configurations.md.
