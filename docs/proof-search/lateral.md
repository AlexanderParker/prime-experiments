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

## Round 25 (2026-08-29) - mandate restored: own backlog, own choice

CHOSE U1 + U2 + U3 and the permitted MIRROR-LAW item, because they turned out to be
ONE object: the involution k -> -k, which pins the parity of every count in the machine
AND supplies the coordinates in which the round-21 phase/amplitude laws become exact
integer statements. U4 (Farey consequences) and U5 (the 613 cosine near-collisions) were
NOT worked this round - depth on a connected cluster beat breadth; they stay in the
backlog untouched and unclaimed.

### Established results (continuing the numbering; do not renumber 1-45)

46. THE MIRROR PARITY LAWS (r25; proved, elementary, and script-verified exact).
    Slot k is blocked iff a gear divides 6k-+1, invariant under k -> -k, so the OPENING
    SET IS EXACTLY CLOSED UNDER NEGATION; 0 is always an opening and P is odd, so 0 is
    the ONLY fixed slot and on indices the map is o_t -> o_{N-t}.
    (a) WINDOW PARITY. Mirror sends the depth-j window at index t to the one at N-t-j;
        N = prod(q-2) is odd so 2t = -j (mod N) has EXACTLY ONE solution. Hence for
        every depth j, W_j(g) is EVEN for every g except the single length g_j* of the
        window at t_j = -j/2 (mod N), where it is ODD. Asserted j = 1..12 at m11-19,
        with g_j* predicted in advance and matched every time.
        Corollary: W_1(F) is EVEN unless F is the antipodal gap (it is not, at any
        machine tested) - THE MAXIMAL GAP ALWAYS OCCURS AN EVEN NUMBER OF TIMES.
    (b) WORD REVERSAL. Mirror reverses gap words, so the depth-j gap-word census is
        EXACTLY reverse-symmetric and every PALINDROMIC word count is EVEN except one
        word per depth. At j = 2 that word is FORCED to be (k_1,k_1), k_1 = the first
        opening. Asserted j = 2,3,4 at m11-19 (25/50/73 .. 221/1216/4489 words).
        This upgrades r7 item 9(b) ("closest to revcomp-symmetric") from approximate to
        EXACT for gap words - the residual asymmetry there is in the L/R labelling only.
    (c) THE ADJACENT-GAP COROLLARY (what the brief asked for). Since k_1 < F always, the
        unique self-mirror adjacent pair is never (F,F). SO ANY ADJACENT CONFIGURATION
        WITH g_1 = g_2 - IN PARTICULAR AN (F,F) PAIR REALISING F_2 = 2F - OCCURS AN EVEN
        NUMBER OF TIMES. "Big next to big" of equal size can never happen exactly once:
        a counting argument that caps such configurations at ONE proves there are NONE.
        Measured #(F,F) = 0 at m11/13/17/19. ROUTED TO THE MANAGER, not developed here.
    (d) FREE CONSISTENCY CHECK, and it fired immediately: EVERY full-period ghist row in
        data/gap_pair_hist.csv carries N-1 gaps, not N - the census closed the period
        LINEARLY and dropped the WRAP-AROUND gap (size k_1 = 3,3,5,5,5,7,7 at m11..31).
        Relative error 1e-9, harmless for densities, fatal to every exact integer
        identity. mirror_cells.load_ghist repairs and asserts it. TEAM-WIDE FINDING.

47. THE GEAR-p CELL DECOMPOSITION (r25; proved + exact). Openings lie on the exposed set
    A_p (|A_p| = p-2) and p-2 consecutive exposed phases span exactly p slots, so for
    zeta = e(1/p), zeta^{gap} depends ONLY on (start phase i, n mod (p-2)) with n the
    EXPOSED-STEP COUNT of item 38. The (p-2)x(p-2) integer CELL MATRIX M[i][s] therefore
    carries the whole frequency-1/p transform of the gap histogram. CRT fixes row sums
    (N/(p-2) each); mirror pairs cell (i,s) with (-(phase_i+Delta), s). Count:
    (p-2)^2 cells, p-2 mirror-fixed, (p-2)(p-1)/2 orbits, (p-2)(p-3)/2 FREE INTEGERS.
    AT p = 5 THAT IS THREE - for every machine, for ever.
    Explicitly, with (e,b,c) = (T[2][0], T[0][2], T[0][3]) and a = N/3 - b - c:
        N_0 = a+2e, N_1 = N/3-e-c, N_2 = 2b, N_3 = 2c, N_4 = N/3-e-b,
    so N_2 and N_3 are ALWAYS EVEN and
        THE MIRROR RELATION:  2 (N_1 - N_4) = N_2 - N_3,  exactly, every machine,
    and (using 1 + omega + omega^4 = phi)
        Re H_5(1) = phi*N/3 + (3-phi) e - ((3phi+1)/2)(b+c)
        Im H_5(1) = (2 sin36 + sin72)(b-c) = (2 sin36 + sin72)(N_2-N_3)/2.
    Asserted from full-period scans at m11-19 and from the census at m11-31; the
    partial-coverage m37 row is carried as a CONTROL and fails, as a period-wide law
    must. The IMAGINARY part of the whole transform is ONE integer.

48. THE PARITY THEOREM - THE POLE PHASE IS UNATTAINABLE - AND WHY GEAR 5 IS SPECIAL
    (r25; closes backlog U3). arg H_5(1) = 126 deg (equivalently: the bracket
    B = H(1-omega)/omega is real) iff (a) N_0+N_1 = 2N_3 AND (b) N_0+N_1 = N_2+N_4. In
    cell variables (b) reads 2(b+c-e) = N/3, and N/3 is ODD. Therefore
        THEOREM. D := (N_0+N_1) - (N_2+N_4) is ODD at every machine, equivalently
        (N_2+N_3) - 2N_0 = 2 (mod 4). THE POLE PHASE 126 IS NEVER ATTAINED EXACTLY.
    Asserted at m11..m31 (defects 38, 282, 2998, 37306, 634182, 13462586, and the m31
    row - all = 2 mod 4). A SECOND, INDEPENDENT refutation of the r21 pin (Refuted 17
    killed it by drift; this kills it by arithmetic).
    BRACKET FORM: Im B_5 = alpha_1 sin72 + alpha_2 sin36 with alpha_r = beta_r-beta_{-r},
    beta_r = N_{r+1}-N_r; the sines are Q-independent so integer realness forces
    alpha_1 = alpha_2 = 0, and alpha_1 = -D is odd. What the machine does instead is
    drive the RATIO to the golden direction: alpha_1/alpha_2 -> -sin36/sin72 = -1/phi,
    measured -0.8636, -0.8393, -0.7305, -0.6403, -0.6448, -0.6231, -0.5943 at m11..31,
    CROSSING -1/phi between m29 and m31 - exactly where arg B(5,1) crosses 0
    (+0.06 at m29, -0.23 at m31, -0.34 on partial m37).
    U3 ANSWERED, two ways. (i) STRUCTURAL: a GF(2) test over mirror orbits shows GEAR 5
    IS THE ONLY PARITY-OBSTRUCTED GEAR for p <= 37; and realness costs (p-1)/2 integer
    equations, so at p = 5 it is ONE ratio chasing one irrational while at p = 7 THREE
    independent asymmetries must vanish at once. (ii) MEASURED: gear 7's asymmetries are
    an order of magnitude larger and decay far more slowly (max|alpha|/N 0.259 -> 0.164
    over m11..37 vs gear 5's 0.141 -> 0.019) - which is the whole of "arg B(7,1) climbs
    -2.4 -> +17.0 while arg B(5,1) converges to 0".
    Honest scope: the parity floor forces |dev| > 0 by only ~1e-6 deg. It kills the pin
    as an EXACT statement; the measured +-4 deg is a different quantity.

49. THE AMPLITUDE NEAR-LAW IS A CROSSING SCALE, NOT AN INVARIANT (r25; closes U2).
    (a) EXACT ANCHOR. The r21 closure gives sum_{j=1..N-1} What_j(omega) =
        (2-phi) n_side^2 - N, so THE MEAN ARM over all proper depths is
        ((2-phi)n_side^2 - N)/(N-1) -> (2-phi)N/9 = 0.042440 N - REAL POSITIVE, and
        exactly the value |What_1| would take if consecutive openings decorrelated.
        Verified over ALL N-1 depths, exactly and real, at m11 and m13. Hence the
        near-law is precisely |What_1|/mean arm = 23.92/lam, and lam = 23.92 IS THE
        MACHINE SIZE AT WHICH DEPTH 1 BECOMES A TYPICAL ARM. 1.015 is that crossing
        scale. (Measured |H|/N*lam: 1.1260, 1.0362, 1.0150, 1.0139, 1.0193, 1.0161 at
        m11..29 - the m11 value is already 11% off, so "no trend over eight machines"
        was m17-onward flatness, 1.0158 +- 0.25%.)
    (b) PHASE GRADING. The phase-blind step model (M[i][s] independent of i) forces
        N_2 = 2N_1 and N_3 = 2N_4; measured N_2/2N_1 = 1.200 -> 1.065 and N_3/2N_4 =
        1.833 -> 1.126 at m11..29, and the blind model recovers |H| to 91.5% -> 95.7%.
        So the amplitude is ~95% a statement about the exposed-step count mod 3 (item
        38's object) and ~5% about the starting phase - AND THE GRADED PART IS SHRINKING.
    (c) CORRIDOR-RENEWAL LADDER (new construct). Model openings as an independent
        thinning, at the rate fixed by the true mean gap, of the slots exposed mod m;
        compute E[omega^gap] exactly by first passage on the m-cycle. GATE: at m = P the
        model reproduces the machine to 1e-9 (asserted). Result, |H|/N*lam:
            y      lam    meas     m=5    m=35   m=385  m=5005 m=85085
            11   2.852  1.1260  1.0916  1.2194  1.1260      -       -
            13   3.370  1.0362  1.0199  1.1874  1.1050  1.0362      -
            17   3.820  1.0150  0.9707  1.1709  1.1032  1.0380  1.0150
            19   4.269  1.0139  0.9292  1.1594  1.1066  1.0441  1.0259
            23   4.676  1.0193  0.8965  1.1512  1.1111  1.0503  1.0354
            29   5.022  1.0161  0.8713  1.1449  1.1149  1.0553  1.0428
        NO FIXED CORRIDOR DEPTH REPRODUCES THE FLAT 1.015: m=5 decays 1.09 -> 0.87 and
        every deeper column RISES. The flatness is the cancellation of a decaying
        shallow drift against a rising deep drift as the machine's own corridor depth
        grows with it. Pushed past the data each fixed-m column has a MINIMUM near
        lam ~ 16-24 and then grows toward (2-phi)lam/9.
    HONEST OPEN PART: the plateau's WIDTH and the DIRECTION of its break are not
    settled - the corridor model turns UP past lam ~ 24 while the r21 closed-form M1
    predictor DECLINES (1.060 -> 0.906 over y = 41..449, data/r25_asym.log). Both agree
    it is not constant. Decidable by one full-period m37 or m41 gap histogram.

50. THE m29 DEPTH SPIRAL (r25; closes U1 - the rung nobody had run). Streaming rewrite
    (research/spiral29.py: only opening RESIDUES mod 5 in a rolling J-buffer, since
    omega^5 = 1; ~200 MB peak instead of the r21 code's ~1.7 GB) over the full m29
    period P = 1,078,282,205, N = 214,708,725 openings, all gates green.
    W_1 arm: |W_1|/N = 0.2023 at arg 126.06 deg. W_2 ARM LADDER, reproducing r21 and
    extending it: -9.20, +33.90, +66.47, +87.71, +113.15, +118.78 deg at m11..29 -
    STILL CLIMBING TOWARD THE POLE PHASE, but the increment COLLAPSED from +25.4 to
    +5.6. |W_2|/N decays 0.0985 -> 0.0173. The large-j arms settle near the mean-arm
    value with small argument (|W_j|/N ~ 0.042-0.061, arg -3 .. +13 deg for j = 19..25),
    which is item 49(a)'s floor showing up directly in the spiral.

### Refuted angles (continuing)

30. My own pre-registered P4 (r25): "the corridor-renewal ladder converges UPWARD with
    m, and m=385 is within 5% of measured at m19/23". REFUTED by my own script: m=5 is
    below, every m >= 35 is ABOVE, so convergence is from above, and m=385 is 9.1% high
    at m19. The useful residue is the correct statement in 49(c).
31. My own pre-registered P5 (r25): "the phase-blind step model departs from the truth
    by >= 5% at every machine". REFUTED: 4.8/4.5/4.3% at m19/23/29 and SHRINKING. Right
    in direction, wrong in size, and the shrinkage is the finding.
32. My own pre-registered P6 (r25): "the closed-form M1 model exceeds 1.10 by y <= 200",
    i.e. the amplitude law breaks UPWARD within reach. REFUTED: M1 gives 1.060, 1.055,
    1.053, 1.049, 1.044, 1.037, 1.027, 1.019, 1.002, 0.986, 0.967, 0.947, 0.927, 0.906
    at y = 41..449 - a slow DECLINE. Two models now disagree on the break direction
    (Refuted-30's ladder turns up, M1 turns down); only "not constant" is established.
33. "The 1.015 amplitude near-law is a machine-independent invariant" (r21 item 32's own
    label, carried as backlog U2): REFUTED as an invariant - it is the crossing scale
    lam = 23.92 of item 49(a), it is already 11% off at m11, and no fixed corridor depth
    reproduces its flatness. The LAW that survives is the exact mean-arm identity.

### Prediction scorecard, round 25 (pre-registered before the runs)

  P1 unique odd depth-2 palindrome is (k_1,k_1); #(F,F) even everywhere: CONFIRMED
     (m11-19; #(F,F) = 0 at all four).
  P2 gear 7's pole condition is NOT parity-obstructed: CONFIRMED (GF(2) test).
  P3 gear 5 is the ONLY parity-obstructed gear among p <= 37: CONFIRMED - and this was
     the round's most useful prediction, since it is the structural half of U3.
  P4 corridor ladder converges upward, m=385 within 5% at m19/23: REFUTED (Refuted 30).
  P5 |H_pb|/|H| departs from 1 by >= 5% everywhere: REFUTED (Refuted 31).
  P6 the closed-form model exceeds 1.10 by y <= 200: REFUTED (Refuted 32).
  3 confirmed, 3 refuted, all three refutations this lane's own.

### Backlog changes

CLOSED this round: U1 (m29 spiral, item 50), U2 (amplitude near-law, item 49),
U3 (gear 7 vs gear 5, item 48). The permitted mirror-law item delivered items 46-48.
STILL UNTOUCHED, carried verbatim: U4 (Farey-spectrum consequences) and U5 (the 613
cosine near-collisions at m31) - not worked, not weakened, not claimed.

NEW, from this round:
U6. WHY does alpha_1/alpha_2 converge on -1/phi? Measured, not derived; and it has now
    CROSSED (-0.6231 at m29, -0.5943 at m31, -0.5778 on partial m37). Does it overshoot
    permanently, or oscillate about the golden direction? Needs one more full-period
    histogram (m37) - the same datum item 49 needs.
U7. The gear-7 cell matrix measured DIRECTLY (10 free integers): which orbit carries the
    drift? Item 48 measured the aggregate asymmetries only. Cheap: one full-period pass
    already written (spiral29.py generalises from p = 5 to any p).
U8. Kernel handoff: the parity theorem (N_2+N_3) - 2N_0 = 2 (mod 4) and the mirror
    relation 2(N_1-N_4) = N_2-N_3 are finite, integer, per-machine statements - the
    cheapest Lean targets this lane has produced.
U9. The plateau break: two models disagree in DIRECTION (Refuted 32). One full-period
    m37 or m41 gap histogram decides it.

### Reproduction pointers

Round 25: research/spiral29.py (streaming depth spiral + cell table + parity columns for
any machine; "python spiral29.py <y> --J 25"; logs data/spiral_{11,13,17,19,23}.log,
data/spiral29.log, JSON data/spiral_<y>.json); research/mirror_cells.py (parts A-F;
"--parts ABCDEF --maxy 19"; log data/mirror_cells.log; 9 assertion gates, exit 0);
data/r25_asym.log (c14_phase.py --asym sweep to y = 449, the P6 test).
NOTE: mirror_cells.load_ghist REPAIRS data/gap_pair_hist.csv's missing wrap gap; any
future consumer of that file should use it or repeat the repair.
Novel-register docs added: docs/novel/mirror-parity-laws.md,
docs/novel/gear-cell-decomposition.md.

## Round 26 (2026-08-29) - own backlog, own choice: the parity lever's second half

CHOSE the brief's optional item ("what OTHER counting arguments does the lever unlock?")
plus U4 (Farey-spectrum consequences), because they are ONE object: the lever's reach is
decided by where the mirror's FIXED POINTS are, and the Farey spectrum's multiplicities
are exactly the gap histogram's residue-class counts, whose parities are those same
fixed points. Depth on the connected pair beat breadth again.
NOT WORKED, and honestly: U5 (the 613 cosine near-collisions) - untouched, unclaimed.
U7 (gear-7 cells) - its FRAMING changed under item 56 and it should be re-posed before
it is run (see backlog). U6 and U9 are blocked on one full-period m37/m41 gap histogram,
which Mechanic is computing this round; nothing this lane can do without it.
GATE: `research/mirror_lever2.py --parts ABCDEFG` -> 52 assertion gates, exit 0
(log `research/data/mirror_lever2.log`). Predictions pre-registered in
`research/data/r26_lateral_predictions.txt` before any of it was coded.

### Established results (continuing the numbering; do not renumber 1-50)

51. THE SYMMETRY GROUP IS EXACTLY Z/2 - AN EXACT CEILING ON THE LEVER (r26; proved,
    brute-force gated).
    (a) AFFINE FORM. The affine maps k -> ck + b of Z_P carrying the opening set onto
        itself are exactly {k -> ck : c = +-1 (mod q) for every gear q} = (Z/2)^m, m =
        #gears. Proof: preserving O means permuting each tooth pair {+-u_q}; adding the
        two requirements cu + b = -+u gives 2b = 0 (mod q) with q odd, so b = 0, and then
        cu = +-u with u invertible gives c = +-1. The element flipping the gears in S has
        exactly P/prod_S q fixed slots - ONE when S is everything (the mirror).
        Gated at m11 over ALL 240 units x 385 shifts = 92,400 affine maps, at m13 over
        all 2,880 units; group, fixed-point counts and adjacency verdict all exact.
    (b) ONLY c = +-1 ACTS ON WINDOWS. Of the 2^m symmetries only the identity and the
        mirror send consecutive openings to consecutive openings (gated m11, m13).
    (c) AND WITHOUT THE AFFINE ASSUMPTION. Anything acting on windows preserves the
        CIRCULAR order of Z_P, so it is a rotation k -> k+b or a reflection k -> b-k.
        Both force b = 0 (mod q) at every gear by the same two-equation argument. So
            THE FULL SYMMETRY GROUP OF THE OPENING SET INSIDE THE CIRCLE Z_P IS
            {identity, mirror} = Z/2, EXACTLY.
        Brute-forced over all 2P rotations and reflections at m11 and m13.
    CONSEQUENCE, and it is the honest half: the lever "cap at one, parity gives zero" is
    worth EXACTLY ONE UNIT - a factor of two in a counting argument - and there is no
    mod-4 version to hope for from any symmetry of the machine. A finer parity must come
    from something that is not a symmetry of the opening set.

52. THE EXCEPTIONAL WINDOW, RELOCATED FROM AN INDEX TO AN ADDRESS (r26; proved + gated).
    Round 25 (item 46a) located the unique self-mirror depth-j window by its INDEX
    t_j = -j/2 (mod N), which is useless without an enumerated period. In slot space: a
    depth-j window with endpoints x < y = x+g in [0,P) is self-mirror iff x + y = 0
    (mod P), i.e. 2x + g in {P, 2P}, so it is CENTRED ON THE ANTIPODE (g odd) or ON SLOT 0
    (g even). Counting openings on each arc,
        j even:  g_j* = 2 o_{j/2}             (through slot 0, itself an opening)
        j odd :  g_j* = 2 b_{(j+1)/2} - P     (through the antipode)
    with o_i the openings just above 0 and b_i those just above (P-1)/2. Both lists come
    from sieving a few dozen slots, so g_j* IS SCAN-FREE AT EVERY MACHINE.
    COROLLARY (the free half): g_j* = j (mod 2), so W_j(g) is EVEN for every g of the
    wrong parity with NO computation at all.
    VERIFIED against the exact full-period W_j census at m11..m29 for every depth j <= 12:
    the set of g with W_j(g) odd is exactly {g_j*}, no exceptions. Table:

        y \ j    1   2   3   4   5   6   7   8   9  10  11  12
        11       1   6  11  10  21  14  25  20  31  24  35  34
        13       1   6  11  10  21  14  25  20  31  24  39  34
        17       1  10  21  14  25  20  31  24  39  34  41  36
        19       1  10  21  14  31  20  39  24  41  34  49  36
        23       1  10  21  14  39  20  41  24  49  34  55  36
        29       1  14  21  20  41  24  49  34  55  36  71  46
        31       1  14  49  20  55  24  71  34  85  36  99  46
        37       1  14  55  20  71  24  85  34 105  36 109  46
        41       1  20  71  24  85  34 105  36 111  46 115  50
        43       1  20  71  24  85  34 105  36 111  46 119  50
        47       1  20  71  24  85  34 105  36 111  46 119  50
        53       1  20  85  24 105  34 111  36 119  46 129  50

    (m31 and above are computed at machines no scan reaches.)

53. g_1* = 1 ALWAYS - THE ANTIPODAL GAP IS A THEOREM (r26). Round 25 recorded g_1* = 1
    "at the machines checked". It is universal, and it is this lane's own T3 law wearing
    a different hat. P = 0 (mod q), so the antipodal slot s = (P+1)/2 reduces mod every
    gear to inverse(2) = (q+1)/2. Multiply by 6: 6s = 3(q+1) = 3 (mod q), while
    6(+-u_q) = +-1 by the tooth law. So s is a tooth iff 3 = +-1 (mod q), i.e. q | 2 or
    q | 4 - impossible for q >= 5. THE ANTIPODAL SLOTS (P+-1)/2 ARE OPENINGS AT EVERY
    MACHINE, the antipodal gap has length 1, and therefore

        W_1(g) IS EVEN FOR EVERY g >= 2, at every machine, unconditionally.

    Only the count of gaps of size 1 is odd. In particular the number of MAXIMAL gaps is
    even with no side condition - round 25's caveat ("unless F is the antipodal gap") is
    discharged for ever - so the maximal gap never occurs exactly once.

54. THE FIXED-POINT CRITERION, THE REVERSAL THEOREM, AND WHAT NOT KNOWING IT COST (r26).
    (a) CRITERION. For a PALINDROMIC tuple w of span s the occurrence set is
        mirror-invariant and an occurrence at k is self-mirror iff 2k = -s (mod P); P is
        odd, so there is exactly one candidate address k_w = -s * inverse(2) (mod P).
        THEOREM: #occ(w) is ODD iff w occurs at k_w - an O(#gears) test. Specialising to
        w = (g,g) gives k_w = -g and forces openings at -g, 0, g with nothing between,
        i.e. g = k_1: round 25's "the unique odd depth-2 palindrome is (k_1,k_1)" in one
        line, now valid at every arity. Gated at m11..m23 against the exact period census:
        the criterion predicts the parity of EVERY palindromic 2- and 3-tuple.
    (b) REVERSAL. The mirror sends an occurrence of w at k to one of reverse(w) at
        -(k + span w), bijectively, so #occ(w) = #occ(reverse w) EXACTLY, and
        realisability is reverse-invariant - including for MERGE KILL WORDS, since the old
        machine's openings and the new gear's teeth are both negation-symmetric. So every
        realisability census need only decide ONE WORD PER REVERSE CLASS.
        Gated: the realised 4-tuple dictionaries at m23/m29/m31/m37 are exactly
        reverse-closed (15,696 / 45,854 / 115,193 / 291,675 tuples; at m37, 145,768
        reverse pairs and 139 palindromes = 145,907 classes, a 50.0% decision saving).
        AUDIT of this project's own arity censuses (research/data/r24/akillp_*.log):
        82 word decisions, EVERY reverse pair agreeing - the theorem's falsifiable gate -
        and 12,877 s of 27,946 s (46%) spent deciding the second member of a reverse pair,
        including two of the four span-141 words at 47->53 that cost 20,005 s between them.
    (c) THE LEVER HAS NO SIDE CONDITION ON THE (D) FAMILY. The merge law quantifies only
        over QUALIFYING windows (middle gaps >= the next gear's tooth floor a = 2u'). The
        exceptional window sits against slot 0 or the antipode, where the gaps are the
        machine's shortest, so it is NEVER qualifying: checked at every rung 11->13 ..
        47->53 and every depth j <= 7, all 66 cells negative. Hence an exact bound "at
        most ONE qualifying depth-j window exceeds the budget" proves there are NONE.
        (Reported for the route, not developed - mandate.)

55. THE MULTIPLICITY THEOREM, AND THE FAREY LEVEL COUNT CORRECTED (r26; closes U4).
    In the path decomposition (item 36) write an eigenvalue 2cos(pi j/(g+1)) as
    2cos(pi a/b) in lowest terms. Then b | g+1, and for FIXED b every a coprime to b
    arises from exactly the gaps g = -1 (mod b). So

        THEOREM. mult(2 cos(pi a/b)) = Sigma(b) := sum_{g = -1 mod b} W_1(g),
        INDEPENDENT of a.

    The eigenvalue multiplicities of A ARE the gap histogram's residue-class counts, one
    class per modulus, and the map inverts: W_1(b-1) = sum_{t>=1} mu(t) Sigma(tb), with
    F + 1 = max{b : Sigma(b) > 0}. This is the CONSTRUCTIVE form of item 39's negative
    ("every unitary invariant of BS is the gap histogram") - here is the inversion.
    (a) PARITY (with item 53). Sigma(b) is odd iff b | 2, so EVERY eigenvalue multiplicity
        of A is EVEN except that of the eigenvalue 0. Asserted exactly at m11..m29 for
        every b <= F+1. Corollary: A never has a simple Perron eigenvalue.
    (b) THE LEVEL COUNT IS A DIVISOR-CLOSURE STATISTIC, and item 36's corollary 1 was the
        NAIVE Farey count: #distinct = sum{phi(b) : b >= 2 divides g+1 for some REALISED
        gap g}. Recomputed on the true supports (m11-29 from the full-period census,
        m31/m37 from the exact 4-tuple dictionaries - validated at m23/m29 where both
        sources exist - m41 from the COV-SAT support {1..91}\{84,87,89}):

            y          11   13   17   19   23   29     31      37      41
            TRUE       21   41  113  183  363  549    981   1,813   2,467
            published  21   45  119  211  383  603  1,085   2,455       -

        Only machine 11 (holeless) was right. LOSS RULE, exact at all nine machines: b is
        absent iff every multiple of b in [2,F+1] is (hole+1); since every observed hole
        is in the top half of the range, loss = sum over holes h of phi(h+1) - e.g.
        phi(85)+phi(88)+phi(90) = 128 at m41, pre-registered and matched.
        SO THE HOLE LIST IS A SPECTRAL OBSERVABLE: the arithmetic-selection object nobody
        can predict is exactly the defect between the true level count and the Farey count.
    (c) RIGIDITY, recomputed on the true level set: <r~> = 0.7206, 0.6306, 0.6785, 0.6507,
        0.6876, 0.6982, 0.6897, 0.6788, 0.6938 at m11..m41 (floats), still ABOVE GUE's
        0.6027 everywhere, and P(s < 0.1 mean) = 0 exactly - now for a REASON, since
        deleting levels only lengthens spacings, so a subset of a set with a hard gap keeps
        it. Round 22's conclusion is unchanged; two of its numbers are not.

56. EVERY GEAR IS PARITY-OBSTRUCTED - THE POLE PHASE IS UNATTAINABLE EVERYWHERE (r26).
    Since W_1(1) is the only odd histogram entry (item 53) and 1 = 1 (mod p) for every p,

        N_1^(p) := #{gaps = 1 mod p} is ODD and N_r^(p) is EVEN for every other r,

    at EVERY machine and EVERY modulus p. The bracket B_p = sum_s beta_s omega^s
    (beta_r = N_{r+1}-N_r) is real iff alpha_s := beta_s - beta_{-s} vanishes for
    s = 1..(p-1)/2, the omega^s - omega^{-s} being Q-linearly independent (disjoint pairs
    of the power basis). But

        alpha_1 = beta_1 - beta_{p-1} = N_2 - N_1 - N_0 + N_{p-1}
                = even - ODD - even + even = ODD != 0.

    THEOREM: for every gear p >= 5 and every machine, alpha_1(p) is odd, so B_p is never
    exactly real and THE POLE PHASE IS NEVER ATTAINED - at gear 7, gear 11, gear 37,
    everywhere, not only at gear 5. Asserted at m11..m29 for every p in the gear set plus
    41 and 43 (alpha_1 values in the log). This supersedes item 48's uniqueness claim;
    item 48's gear-5 conclusion stands, and its MEASURED half (three equations instead of
    one, asymmetries an order of magnitude larger and slower to decay) is still the real
    explanation of why gear 5's bracket looks real and gear 7's drifts.

### Refuted angles (continuing)

34. "GEAR 5 IS THE ONLY PARITY-OBSTRUCTED GEAR FOR p <= 37" - my own round-25 prediction
    P3, scored CONFIRMED and called the structural half of U3. REFUTED by item 56: every
    gear is parity-obstructed. The round-25 GF(2) test was sound but answered a NARROWER
    question than its label - whether the CELL-MATRIX constraints alone (row sums odd plus
    the pole equations) force a parity contradiction. Those constraints know nothing about
    W_1(1), which is where the real obstruction lives. LESSON, and it is general: a
    satisfiability verdict over a chosen constraint set is a statement about THAT SET, not
    about the machine; label it with the set.
35. "#distinct eigenvalues = |Farey(F+1)| - 2" (item 36 corollary 1, r22): REFUTED for
    every machine with holes - true only at m11. The published table 21/45/119/211/383/
    603/1085/2455 is the naive count; the true one is item 55(b). My own error, found by
    my own script, and the cause is exact: the code enumerated `for g in range(1, F+1)`
    instead of the realised support.
36. "s_min/s_mean descends to Hall's 3/pi^2 = 0.30396" (item 36 corollary 2, r22):
    REFUTED on the true level set - the ratio is not monotone and dips to 0.2422 at m37,
    because deleting levels raises the mean spacing while the surviving minimal spacing is
    unchanged. The ABSOLUTE hard gap survives (P(s < 0.1 mean) = 0 exactly, now proved by
    the subset argument); the NORMALISED statement is not a law.

### Prediction scorecard, round 26 (pre-registered in data/r26_lateral_predictions.txt)

  P1  only the 2^m multiplications, only c=+-1 acts on windows, no mod-4 lever: CONFIRMED
  P2  scan-free g_j* matches the only odd W_j column at m11..29, j <= 12:      CONFIRMED
  P3  g_j* = j (mod 2):                                                       CONFIRMED
  P4  every reverse pair in the A_kill logs agrees; > 40% redundant time:      CONFIRMED
      (82 decisions, zero disagreements, 46%)
  P5  the m23/29/31/37 4-tuple dictionaries are exactly reverse-closed:        CONFIRMED
  P6  the m41 level count is short by exactly phi(85)+phi(88)+phi(90) = 128:   CONFIRMED
  P7  the loss rule (only holes above (F+1)/2 cost anything) is exact:         CONFIRMED
  P8  Sigma(b) odd exactly when b | g_1*+1:                                    CONFIRMED
  P9  N_1^(p) odd, all other N_r^(p) even, every machine and modulus:          CONFIRMED
  P10 alpha_1(p) odd for EVERY gear - contradicting my round-25 P3:            CONFIRMED

  10 of 10, and I do not read that as a good scorecard. P1-P3, P5, P8 and P9 are
  corollaries of theorems I had already proved when I wrote them down, so they were cheap.
  The risk this round was taken elsewhere and it paid: P6, P7 and P10 were real bets, and
  P10 overturned a prediction this lane had scored CONFIRMED last round. The three
  refutations above are all of MY OWN published record, two of them numbers other lanes
  could have cited.

### Backlog changes

CLOSED: U4 (Farey-spectrum consequences) - item 55, plus the two corrections it forced.
STRENGTHENED: U8 (kernel handoff) - g_1* = 1 is now the cheapest Lean target this lane has
  ever produced: "6s = 3 and 6u = +-1, and 3 != +-1 mod q for q >= 5" is five lines and it
  implies the whole even-count law for gaps.
RE-POSED: U7 (gear-7 cells). Item 56 answers the parity half for every gear at once, so
  the remaining question is not "is gear 7 obstructed" but "WHICH cell orbit carries the
  measured drift, and why does its magnitude decay so much more slowly than gear 5's".
STILL UNTOUCHED: U5 (the 613 cosine near-collisions at m31) - not worked, not weakened.
STILL BLOCKED: U6 (-1/phi overshoot) and U9 (the plateau break direction) - both need one
  full-period m37 or m41 gap histogram. Mechanic's h37 workers were running at round
  close; the moment that array exists, both are a five-minute computation for this lane.
NEW:
U10. Where could a mod-4 lever come from? Item 51 proves NO symmetry of the opening set
     supplies one. Candidates that are not symmetries: a free Z/4 action on a SUBSET of
     configurations (e.g. the qualifying family), or a pairing not induced by a map of Z_P
     at all. Named, not built.
U11. Is "every hole lies in the top half of the gap range" a theorem? It is what makes
     item 55(b)'s loss rule exact, it holds at all nine machines with hole data, and the
     project already observes that the spectrum fills monotonically from below.

### Needs / handoffs

(1) MECHANIC: item 54(b) halves every arity census and every dictionary build - decide one
    word per reverse class and copy the verdict. Measured, on your own logs: 46% of
    27,946 s. The check is one line (`w[::-1] in decided`).
(2) FORMALIST: three cheap kernel targets, in increasing size - g_1* = 1 (item 53, five
    lines, implies "every gap length >= 2 occurs an even number of times"); the fixed-point
    criterion (item 54a, a decidable membership test at one address); and the round-25
    parity theorem, which item 56 now derives more simply.
(3) MANAGER / CONSTRUCTOR: item 54(c) - on the qualifying family the lever has NO side
    condition, so a first-moment argument only has to reach "fewer than two" rather than
    "fewer than one". Item 51 prices that honestly: it is one factor of two and no more.
(4) LP THREAD (UNTESTED, offered not claimed): the covering CSP is invariant under
    negating every gear offset together with reflecting the window, so its feasible set is
    symmetric and any LP/SDP relaxation may be restricted to the symmetric subspace without
    loss - roughly halving the variable count. Not built or measured here.
(5) ANY LANE citing item 36's distinct-level table: use item 55(b)'s numbers.

### Reproduction pointers

Round 26: research/mirror_lever2.py (parts A-G; "--parts ABCDEFG"; log
data/mirror_lever2.log, 52 assertion gates, exit 0); predictions in
data/r26_lateral_predictions.txt. Inputs: data/depth_identity_{11,13,17,19,23,29}.csv
(exact W_j census), data/gap_tuples_{23,29,31,37}_4.csv and
data/gap_tuples_41_4_transfer.csv (supports), data/r24/akillp_{43_47,47_53}.log (the
arity-census audit). Novel-register docs updated: docs/novel/mirror-parity-laws.md
section 7, docs/novel/farey-chebyshev-spectrum.md section 7 (which CORRECTS its own
corollaries 1 and 2), and the two README index entries.

## Round 27 (2026-08-29) - the routed 2n law, plus three backlog items closed

CHOSE: the routed 2n-gap law in full (prove / prior-art / the shuffle question),
then U6 + U9 (Mechanic's exact full-period m37 histogram finally exists, so both
really were the five-minute computations round 26 said they were), then U5 - the
item this lane has left untouched for three consecutive rounds - which turned out
to fall to a two-line field-theory argument once it was actually attacked. The 2n
work then threw off a NEW object that had nothing to do with the brief and is the
round's most interesting number: the twin machine's position in the distribution
of F over its own counterfactual sievings.
NOT WORKED, honestly: U7 (which cell orbit carries the gear-7 drift) and U10/U11
- untouched, unclaimed, and U7's re-posed form is still the right one.
GATES, all from clean processes at round close, all exit 0:
  research/lex_odometer.py --parts ABCDEFGH   -> 145 gates (log data/lex_odometer.log)
  research/ghist37_u69.py                     ->  45 gates (log data/ghist37_u69.log)
  research/u5_collisions.py --y 29            ->  10 gates (log data/u5_collisions_29.log)
  research/tooth_counterfactual.py --upto 19  ->  10 gates (log data/tooth_counterfactual.log)
  research/tooth_msweep.py                    ->  (report, log data/tooth_msweep.log)
Predictions P1-P13 pre-registered in research/data/r27_lateral_predictions.txt,
each block written before the code it scores. Every job this round launched has
finished; nothing is left running.

### Established results (continuing the numbering; do not renumber 1-56)

57. THE 2n-GAP REORDERING LAW - PROVED, WITH ITS VALUES AND MULTIPLICITIES
    (r27; the human's sort-step idea, routed by the manager).
    Under CRT the opening set is the product A_1 x ... x A_n, A_i = Z_{q_i}
    minus the two teeth, and CRT-LEX ORDER IS EXACTLY THE MIXED-RADIX ODOMETER
    on the digit vector. The lex successor increments the last non-maximal digit
    i and wraps everything below it, so the value difference is

        D(i, delta) = CRT( 0 for i'<i ; delta for i ; w_{i'} for i'>i ),
        w_i = -max(A_i) mod q_i,  delta a consecutive difference of sorted A_i.

    Coordinates below i are 0, so the carry position is RECOVERABLE from the
    difference and distinct (i, delta) give distinct differences. Hence

        #distinct differences = sum_i d_i,   d_i = #distinct consecutive
                                                   differences of sorted A_i,

    and for the machine d_i = 2 at every gear because the teeth {u, -u} are
    NEVER adjacent (adjacency needs 3 = +-1 mod q, i.e. q | 4) and 0 is never a
    tooth. So the count is 2n. Multiplicities are closed form:
    mult(D(i,delta)) = s_i(delta) * prod_{i'<i}(q_{i'}-2) with s_i(2) = 1 at
    gears 5 and 7 and 2 elsewhere, s_i(1) = q_i - 3 - s_i(2).
    (a) THE CYCLIC CLOSURE IS FREE, and that IS a fact about the machine: the
        last-to-first difference has first coordinate w_1 = -max(A_1), which is 1
        when q_1-1 is exposed and 2 when q_1-1 is a tooth (q_1 = 5 or 7) - either
        way w_1 in {1,2}, so the wrap IS D(1,w_1). Linear and cyclic counts are
        both 2n. For a general two-point sieve the wrap CAN be a 2n+1-st value.
    (b) ORDER-INDEPENDENT: d_i depends only on A_i, so all n! gear orderings give
        2n (all orderings at n <= 4, twelve sampled at n = 5,6). The 2n VALUES do
        depend on the ordering; only the count is invariant.
    (c) THE STEP-TYPE LAW, the general form: for an arbitrary removed set T the
        sorted survivors' differences are {L+1 : L an INTERIOR maximal run of T}
        plus 1 if two survivors are adjacent, "interior" = the run touches
        neither 0 nor q-1. Gated on 400 random removals.
    (d) THE DIGITAL-SEQUENCE FORM: Phi(t) = sum_i A_i[j_i(t)] E_i mod P (E_i the
        CRT idempotent, j_i(t) the mixed-radix digits) is an explicit bijection
        [0,N) -> O - a generalised van der Corput / Halton point set, in which F
        is exactly P times the sequence's DISPERSION. Gated exactly at m7..m17.
    PRIOR ART (checked in-round, web): KNOWN IN MECHANISM. Langevin's theorem -
    lex order on a planar lattice has successor in {w+u, w+v, w+u+v}, and it
    RECOVERS the three-distance and three-gap theorems - is the same carry
    argument; Fried-Sos generalise it to ordered abelian groups (both reported in
    Chevallier, "Cyclic groups and the three distance theorem"). The finite CRT
    version is folklore-grade. The delta that survives is technical: the exact
    multiplicity table, the free wrap, order-independence.

58. THE DEFLATION - THE 2n COUNT IS BLIND TO F, AND THAT IS A PROOF (r27).
    By 57 the count depends on each gear ONLY through "how many distinct interior
    run lengths does the removed set have", which is 1 for EVERY two-point
    removal except the degenerate terminal pair {q-2,q-1}. Consequences, gated:
    (a) Over 60 admissible re-choices of the teeth at mods [5,7,11,13] the
        distinct-difference count is 2n = 8 EVERY TIME while F ranges over
        [10,18] - a factor of 1.8.
    (b) It does not even need primes: coprime NON-prime moduli [8,9,25] with
        two-point removals also give 2n.
    (c) The only way to leave 2n is to remove {q-2,q-1} - a fact about where you
        cut the cycle.
    So the reordering is an exact change of coordinates that DISCARDS precisely
    the arithmetic F depends on. And F is not a statistic of the phase-order ->
    natural-order PERMUTATION at all: a permutation records order, F needs the
    metric, and the metric lives in the VALUES Phi(t). The dual measurement says
    the same thing from the other side - the number of distinct LEX-INDEX
    displacements between NATURAL-order neighbours is 5, 25, 95, 368, 1362 at
    n = 2..6, i.e. growing. THE DOC'S OWN FRAMING ("every hard question is a
    property of the shuffle alone") IS TRUE AND EMPTY: the trivial side is
    trivial for reasons independent of the arithmetic. Closed line, not a route.
    What the frame DOES buy, and it is real if small: an exact sieve-free
    O(n)-memory streaming enumeration of the opening set at any machine, in phase
    order, from 3n integers (n radices + 2n strides).

59. U5 CLOSED - THERE ARE NO ACCIDENTAL COSINE COLLISIONS, AND THE DEGENERACY
    LAW IS A THEOREM AT EVERY MACHINE (r27; three rounds untouched, then two
    lines). The circulant's eigenvalue at frequency vector (j_q) is
    prod_q f_q(j_q) with per-gear factor set S_q = {q-2} u {-2cos(2 pi r/q)},
    and no element vanishes. If prod f_q = prod f'_q then prod (f_q/f'_q) = 1
    with f_q/f'_q in K_q = Q(zeta_q)^+; the K_q have pairwise COPRIME CONDUCTORS,
    hence each is linearly disjoint from the compositum of the others and meets
    it in Q, so every ratio is RATIONAL. A rational ratio inside S_q forces
    equality: both rational means both are q-2; one rational and one not is
    impossible; both irrational makes them Galois conjugates with equal norms, so
    the ratio is +-1, and -1 needs 2(r+r') = q or 2(r'-r) = q, impossible for odd
    q. THEREFORE lambda(j) = lambda(j') iff j'_q = +-j_q at every gear:
        the degeneracy group is exactly (Z/2)^{#gears},
        #distinct = prod (q+1)/2 and ties = P - prod (q+1)/2, EVERY machine.
    Round 21 had this as a MEASUREMENT at m11/13/17; it is now a theorem
    everywhere, and it implies every "near-collision" is a near-miss.
    TESTED DECISIVELY at m29, where round 21 reported 6 at tolerance 1e-12: all
    8,164,800 desymmetrized levels rebuilt, exactly those 6 pairs found, each
    recomputed at 60 decimal digits - ALL SIX SEPARATE, smallest separation
    8.635e-14, none zero. Crowding measured: median adjacent spacing 1.30e-05,
    bottom 1% below 4.20e-08, 19.0% of levels inside |lambda| < 1. m31's 613 are
    covered by the theorem and were NOT re-measured (1.3e8 labelled levels is not
    memory-safe here, and the theorem makes it unnecessary) - said plainly rather
    than reported around. Free double-source: the same script re-derives round
    21's tie counts 313 / 4501 / 80549 at m11/13/17 by brute force.

60. U6 AND U9 CLOSED ON THE EXACT m37 HISTOGRAM - AND -1/phi IS A CROSSING, NOT
    A LIMIT (r27, on research/data/r26/ghist_37.csv).
    (a) U6. The gear-5 asymmetry ratio, now exact at m37: alpha_1 = 4,107,707,379,
        alpha_2 = -7,109,650,222, ratio -0.577765. Full exact ladder m11..m37:
        -0.863636, -0.839286, -0.730507, -0.640249, -0.644811, -0.623140,
        -0.594340, -0.577765. It CROSSES -1/phi = -0.618034 between m29 and m31
        and keeps rising: at m37 it is +0.0403 past the golden direction, seven
        times its distance at m29, with post-crossing increments +0.0288 and
        +0.0166 - decaying, but not turning. SO ITEM 48's "the machine drives the
        ratio TO the golden direction" IS REFUTED as an asymptotic claim (see
        Refuted 37). The exact identities under it are untouched: Im B_5 =
        alpha_1 sin72 + alpha_2 sin36, alpha_1 odd, pole phase unattainable.
        The crossing is the same event as arg H_5(1) crossing 126 deg.
    (b) U9. The amplitude plateau |H_5(1)|/N * lam, exact at m11..m37:
        1.125953, 1.036230, 1.015003, 1.013946, 1.019315, 1.016081, 1.009970,
        1.014085. IT DOES NOT BREAK - it OSCILLATES inside [1.0100, 1.0193] from
        m17 on with no monotone trend. The m31 -> m37 move is UP, which is the
        corridor-renewal ladder's direction and against the M1 model's - and
        against this lane's own pre-registered P8, which called DOWN (Refuted 38).
        The honest verdict is that the "which model" question was mis-posed:
        neither direction is in evidence at reachable machines.
    CROSS-GATES, all passing: total gap count = prod(q-2) and gap sum = P at all
    eight machines; gap 1 the ONLY odd entry at all eight (item 53); alpha_1 odd
    at all eight (item 56); arg H_5(1) reproduces Mechanic's exact ladder
    (129.776 ... 125.659) to 5e-3 deg; the amplitude column reproduces this lane's
    round-25 table to 6e-4 at m11..m29.

61. THE TOOTH COUNTERFACTUAL - THE TWIN MACHINE IS A LOW-F OUTLIER IN ITS OWN
    FAMILY (r27; new object, exhaustive, exact; docs/novel/tooth-counterfactual-
    percentile.md). The machine has two inputs: WHICH gears, and WHERE the teeth
    are. The gears are the problem; the teeth are forced (v_q = 6^{-1} mod q).
    Move them. Keep the mirror symmetry (teeth +-v_q) and let v_q range over
    {1..(q-1)/2}: every member has the SAME period, the SAME survivor count
    prod(q-2) (the sharing law) and the same per-gear density - only positions
    move. F is invariant under k -> +-k+b but NOT under k -> ck, so F genuinely
    varies, and |V(y)| = 30 / 180 / 1440 / 12960 at m11/13/17/19 is small enough
    to ENUMERATE EXHAUSTIVELY.

        y    |V|     F(twin)  min  median  max   twin's percentile
        11   30      7        6    8       11    20.0%
        13   180     11       10   13      25    18.1%
        17   1440    18       14   19      32    26.4%
        19   12960   25       20   28      43    17.1%

    THE TWIN MACHINE'S RECORD GAP IS IN THE BOTTOM FIFTH TO QUARTER OF ITS OWN
    COUNTERFACTUAL DISTRIBUTION at every machine tested, ~10-15% below the median,
    never the minimum, in a family whose maximum is 1.6-1.9x the twin value.
    This is the FIRST quantity on which the real phase vector is distinguished -
    round 2's enumeration (Refuted 3) scored it on WASTE metrics and found no
    handle; F itself separates, and in the favourable direction.
    TWO MECHANISMS PROPOSED AND BOTH REFUTED, both mine:
      - P11 ANGULAR COHERENCE (the twin has v_q/q ~ 1/6 at every gear, the
        smallest angular dispersion in the family). REFUTED IN THE SIGN:
        spearman(F, dispersion) = -0.14 / -0.20 / -0.11 at m13/17/19, and the
        twin sits in the LOWEST-dispersion quartile, which has the HIGHEST mean F
        (28.56 vs 27.69 at m19). Inside that quartile alone the twin is at the
        15.6% / 20.8% / 10.5% percentile - a low-F outlier INSIDE the high-F class.
      - P13 "m IS SMALL" (every vector is v_q = m^{-1} mod q for some m; the twin
        is m = 6). REFUTED: over m = 1..60 at m19 the median F is 28.0, EXACTLY
        the full family's median, with m = 1,2,4 giving 33,34,32 and the sweep's
        minimum F = 20 at m = 12, not at m = 6.
    HONEST CAVEAT, stated because it matters: the four rows are NESTED (the m19
    twin vector extends the m17 one), so they are not four independent draws and
    no p-value is claimed - only "consistently below median, deficit neither
    growing nor shrinking".

### Refuted angles (continuing)

37. "alpha_1/alpha_2 -> -sin36/sin72 = -1/phi - the machine drives the integer
    ratio to the golden direction" - MY OWN item 48 (r25), carried as backlog U6
    and cited in docs/novel/gear-cell-decomposition.md. REFUTED on the exact m37
    histogram (item 60a): the ratio crosses -1/phi between m29 and m31 and is
    +0.0403 past it at m37, still rising. The golden value is a point the ladder
    passes through, not a limit. Everything exact under item 48 survives; only
    the asymptotic reading dies.
38. My own pre-registered P8 (r27): "the amplitude plateau breaks DOWNWARD, siding
    with M1 against my own corridor ladder". REFUTED by my own script - the
    m31 -> m37 move is UP. And the correction is bigger than the sign: there is
    no break at all, only a 0.9%-wide oscillation (item 60b).
39. My own pre-registered P11 (r27): angular coherence explains the twin machine's
    low F. REFUTED IN THE SIGN (item 61).
40. My own pre-registered P13 (r27): the feature is "the teeth are the reciprocal
    of a SMALL integer". REFUTED - the m-sweep's median is exactly the family
    median (item 61).
41. THE 2n REORDERING AS A ROUTE - the routed item's own stated promise ("the
    machine = a trivial product order x an arithmetic shuffle, and every hard
    question is a property of the shuffle alone"). REFUTED by item 58: the count
    is invariant under every tooth re-choice while F moves by 1.8x, so the
    coordinates discard exactly what F depends on; and F is not a function of the
    order permutation at all. The LAW is true and now proved; the ROUTE is dead.

### Prediction scorecard, round 27 (pre-registered in data/r27_lateral_predictions.txt)

  P1  the 2n law holds for EVERY gear ordering:                       CONFIRMED
  P2  the closed-form value set AND multiplicity table:               CONFIRMED
  P3  the cyclic wrap is free (equals D(1,w_1), w_1 in {1,2}):        CONFIRMED
  P4  2n is not a product-set fact; the law is sum_i d_i:             CONFIRMED
  P5  the digital-sequence closed form Phi is a bijection:            CONFIRMED
  P6  the inverse shuffle is NOT small:                               CONFIRMED
  P7  U6: exact m37 ratio in (-0.60,-0.55) and still rising:          CONFIRMED (-0.5778)
  P8  U9: the plateau breaks DOWNWARD:                                REFUTED (it is UP,
      and there is no break)
  P9  m37: gap 1 the only odd entry, total prod(q-2):                 CONFIRMED
  P10 the twin's F is below the counterfactual median everywhere:     CONFIRMED (17-26%)
  P11 angular coherence is the mechanism:                             REFUTED IN THE SIGN
  P12 U5: every near-collision separates at 60 digits:                CONFIRMED
  P13 the mechanism is "m small":                                     REFUTED

  9 confirmed, 4 refuted, all four refutations my own. Self-assessment, since
  last round's 10/10 was rightly called suspicious: P1, P2, P3 and P5 are
  corollaries of a proof I had already sketched before writing them down and were
  cheap. The real bets were P7, P8, P10, P11, P12, P13 - and I lost three of the
  six. That is the right shape for a round; last round's shape was not.

### Backlog changes

CLOSED: U5 (the 613 cosine near-collisions - item 59, by theorem plus a decisive
  60-digit test at m29), U6 (item 60a - and it refuted the claim it was asking
  about), U9 (item 60b - answered as far as the data go, with the question
  re-posed as mis-framed).
STILL UNTOUCHED, carried verbatim and unclaimed: U7 (which gear-7 cell orbit
  carries the drift - its round-26 re-posing stands), U10 (where could a mod-4
  lever come from), U11 (is "every hole lies in the top half of the gap range" a
  theorem). U8 (kernel handoffs) is with Formalist.
NEW:
U12. THE MECHANISM OF ITEM 61. Two candidates dead (angular coherence, small m).
     The next probes, in order of cost: (i) the m23 rung of the percentile table
     (142,560 sievings over P = 37,182,145, about an hour single-core) - does the
     ~20% percentile drift with the machine or hold; (ii) what DOES separate the
     low-F vectors - the obvious untried statistic is the joint distribution of
     tooth DIFFERENCES v_q - v_q' against the corridor classes mod 35, since
     gears 5 and 7 decide every <= 5-point shape (completeness lemma) and the
     twin vector's (v_5, v_7) = (1,1) is one of only six (v_5,v_7) classes;
     (iii) the same percentile for F_2 and for the hole list, which would say
     whether the effect is about the record or about the whole spectrum.
U13. Does item 61 transfer to the LIVE ROUTE's quantity? The counterfactual family
     leaves prod(q-2) fixed, so it is a clean null model for (D) as well: what is
     the twin machine's percentile in the counterfactual distribution of
     F(M+q') - F(M) - q' (the budget slack)? If the twin is favourably placed
     there too, the first-moment transfer has a measured amount of room it is
     currently not using. Named, not built; routed as a QUESTION to the manager
     rather than worked here (mandate).
U14. The 2n frame's one surviving asset (item 58): a 3n-integer, sieve-free,
     O(n)-memory streaming enumerator of the opening set in phase order at ANY
     machine. Nothing in the project currently needs phase order - but a census
     that is invariant under order (any histogram of a phase-local statistic)
     could use it at machines no sieve reaches. Named, not built.

### Needs / handoffs

(1) MECHANIC: nothing owed to me - your exact cyclic m37 histogram closed two of
    this lane's backlog items in one pass, and my script re-derives your arg
    ladder to 5e-3 deg as an independent check of it. If an exact m41 histogram
    ever lands, item 60's two ladders each gain one more rung and the U6 question
    "does the overshoot decelerate to a limit" becomes decidable.
(2) MANAGER: the routed 2n item is PROVED and PRIOR-ART-CHECKED, and it is a
    closed line, not a route (items 57-58, Refuted 41) - docs/novel/
    two-n-gap-reordering.md is rewritten accordingly and its verdict is KNOWN IN
    MECHANISM / PARTIAL OVERLAP. The item worth your attention from this round is
    instead item 61 and U13.
(3) FORMALIST (offered, not claimed): item 59's counting half is a clean finite
    statement per machine - #distinct eigenvalues = prod (q+1)/2 - and item 57's
    2n law is a finite combinatorial statement whose proof is a carry argument.
    Both are smaller than the mirror even-count lemma you already have queued.
(4) ANY LANE citing "alpha_1/alpha_2 -> -1/phi" (item 48, and the
    gear-cell-decomposition index entry): it is a crossing, not a limit - use
    item 60a's exact ladder.

### Reproduction pointers

Round 27: research/lex_odometer.py (parts A-H; "--parts ABCDEFGH"; log
data/lex_odometer.log, 145 gates); research/ghist37_u69.py (U6/U9 on the exact
histograms; log data/ghist37_u69.log, 45 gates); research/u5_collisions.py
("--y 29"; needs mpmath; log data/u5_collisions_29.log, 10 gates);
research/tooth_counterfactual.py ("--upto 19"; log data/tooth_counterfactual.log,
10 gates); research/tooth_msweep.py (log data/tooth_msweep.log). Predictions in
data/r27_lateral_predictions.txt. Inputs: research/data/r26/ghist_{13,17,19,23,
29,31,37}.csv (Mechanic's exact cyclic full-period histograms - m11 is sieved
directly in-script). Novel-register docs: docs/novel/two-n-gap-reordering.md
(rewritten - proof, deflation, prior-art verdict), docs/novel/tooth-counterfactual-
percentile.md (new), docs/novel/eigenvalue-statistics.md section 7 (U5),
docs/novel/gear-cell-decomposition.md section 7 (U6/U9, and it corrects that
doc's own section 5), plus three README index entries.
