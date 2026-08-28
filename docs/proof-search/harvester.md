# Harvester workstream - side theorems and adjacent conjectures (COMPACTED, r1-24)

Compacted 2026-08-29 into ONE cumulative summary over all 24 rounds. Verbatim logs:
`archive/harvester-full-r1-19.md`; `archive/harvester-full-r20-24.md` (the
pre-compaction file: r1-19 core + verbose r20-24 + the manager correction). Nothing
below is new. Where a later round corrected an earlier one the LATER statement is
given and the earlier appears in section 6 - never silently deleted. Reproduction: 8.

MANDATE: statements weaker or adjacent to twin primes where the session's machinery
yields actual results, priced honestly ("not reachable" = not reachable with currently
published methods - an imported corpus limit, distinct from any event in the machine
itself). Rounds 3-14 were twin-route support by coordinator steering; back on this
lane's own mandate from round 15.

## 0. Definitions (once)

`P(z)`/`p_n#` primorial over odd primes <= z = p_n; `m` the modulus sifted. ORDINARY
Jacobsthal `j(n)`: largest gap between consecutive integers coprime to n - one killed
class per prime. PAIRED Jacobsthal `j_2(n)` (Ziller-Morack arXiv:1706.00317): largest
gap between consecutive n with BOTH n and n+E coprime to the modulus, maximised over
even E - TWO killed classes per odd prime; `h_2(p_n#) = j_2(p_n#)`. **ZM CONJECTURE
6:** `h_2(p_n#) < p_n^2 - p_n`, n >= 3. `F_d(y)`: our per-difference family - maximal
gap of the two-residue sieve by gears q <= y at fixed even d = 2e, halved (slot)
coordinates; `h_2 = 2 max_e F_e`; `F(2,y)` the fixed-twin member; OEIS A288815 uses
F = h_2/2. DELTA PROFILE `delta_q(e) = min(e mod q, q - e mod q)`; the r22 delta
reduction collapses e to `delta = e*3^{-1} mod Q`, `Q = prod_{5<=q<=y} q`. MACHINE M_y
= sieve by all odd primes <= y; SLOT k <-> pair (6k-1, 6k+1); SURVIVOR, TOOTH, OPENING,
WORD, LETTER, PADDED/LITERAL LINK = covering-word vocabulary of the twin route
(docs/proof-search/agents-shared.md). `omega(p)` = classes removed mod p (2 for p not
dividing d, 1 for p | d, 1 at p = 2); `V(z) = prod(1 - omega(p)/p)`; sifting dimension
`kappa` (= 2 here); `beta_kappa` the sifting limit. `n_g(M)` = population of gap g
between consecutive twin-slot survivors; `c_q(g)` its per-gear transfer diagonal;
`N_k(...)` closed-form CRT products; `W_j(g)` depth-j window count.

## 1. State of the lane (final ranking)

| # | candidate | status |
|---|-----------|--------|
| N4 | j_2 upper ladder | **TOP.** Executed r21-24. First proved upper bounds of any strength on a function named and conjectured about since 2017. Sec. 2. |
| P1-P3 | j_2 lower ladder | RANK 2. (P1) proved r24 - first bound using paired structure at all. (P2)/(P3) open. Sec. 3. |
| C10+N3 | family F_d | HELD on structure (cap law, extension ladder, percentile, why-13); DOWNGRADED on data (6.11: replication, not first computation). 5b-5e. |
| N5 | paired HL-B / paired Holt recursion | SHORT NOTE, not a paper (r22 prior-art downgrade, partly restored by Bonferroni + effective threshold). 5f. |
| N1 | universal Polignac cap <= 12 | DONE - standalone finite theorem, kernel-checked. |
| N2 | gear-3 non-adjacency, d = 0 mod 6 | DONE - one-line unconditional proof (5g). |
| C2 | Polignac at fixed even gap 2d | CONJECTURE UNMOVED (parity barrier). REDUCTION TRANSFORMED: same single open lemma (D) as twins, uniform in d. |
| C4 | Goldbach via the paired frame | window reduction DONE r1 (kernel-checked, exact converse). Live bite: the singular-series factor over q \| N IS an exposed-set size - kernel-reachable. |
| C8 | constant-2 fragile law | form improved (exact inclusion-exclusion, not a fit); ASYMPTOTIC remains Hardy-Littlewood-class, unreachable. |
| C5 | quadruplets / k-tuples | STRUCK as publishable (re-derives classical admissibility); returns only via F_d for k-tuples - computable, nobody has. |
| C1 | Legendre / Oppermann / Brocard bands | STRUCK. Horizon theorem, one line: gears < y decide the window exactly, so the machinery is about divisibility inside a window and has no prime-side localisation; bands need exactly that (exponent 0.5 vs published 0.525, Alweiss-Luo). |
| C3, C6, C7, C9 | per-gap iff, overcount census, g=2 pinning, onset bound | DELIVERED r1-7 (5a). C9 already in-corpus (Brun-Titchmarsh class). |

The round-1 ranking (C1-C10) was superseded by the round-15 re-ranking and survives
only in the r1-19 archive.

## 2. THE j_2 UPPER LADDER (docs/novel/j2-upper-bound.md)

Before r21 the ladder was EMPTY: ZM prove no upper bound of any strength (Remark 2.2
lists only elementary monotonicity - product, gcd, prime-power collapse), cite no
Iwaniec, give no heuristic for the p^2 - p bound; no follow-up supplies one.

**2a. THEOREM 1** (elementary Legendre inclusion-exclusion, r21; Kanold slot):
`j_2(p_n#) <= 2*3^(n-1)/V_n + 1`, `V_n = (1/2) prod_{3<=p<=p_n}(1-2/p)`; explicitly
`j_2(p_n#) < 3^(n+1)(log p_n)^2` for n >= 3 (n = 2: exact 37). Worst case over
differences is omega = 2 at every odd prime (per-prime E/V factor 3p/(p-2) >
2p/(p-1) always), so the bound is uniform in d and p | d differences get better
constants. Explicit chain: (1-2/p) = (1-1/p)^2(1-1/(p-1)^2), partial twin products
decrease to C_2 = 0.66016..., Rosser-Schoenfeld (3.27); verified with exact V_n
through n = 4203, worst ratio 0.8627 at n = 3 (6.17). Sub-primorial exp(O(p/log p))
vs trivial exp((1+o(1))p).

**2b. THEOREM 3 / 3E** (Brun pure sieve; quasi-polynomial, FULLY EXPLICIT; Stevens
slot): for every ODD K with R_K < V_n, `j_2(p_n#) <= E_K/(V_n - R_K) + 1`, where
`E_K = sum_{j<=K} e_j(omega(p))`, `R_K = sum_{j>K} e_j(omega(p)/p)`, e_j the
elementary symmetric polynomials. Contains Theorem 1 as K >= n (R_K = 0,
E_K = prod(1+omega(p)) = 2*3^(n-1), asserted at every n). At the optimal K (measured
K* = 3,5,7,9,11,13 over p_n = 5..27449; K* ~ lambda T_n, T_n = sum omega(p)/p ~
2 log log p_n) it is quasi-polynomial, strictly better than Theorem 1 from p_n = 13 -
not merely asymptotically - by >300x at p_n = 73 (1.082e9 vs 3.316e11). Checked
against brute-force survivor counts on 1800 real paired windows (n = 3..6, odd
K = 1..n+1), tightest ratio 0.54. THE EXPLICIT FORM (r23):

    j_2(p_n#) < p_n^{9.30 log log p_n}  for every n >= 3, with the ASYMPTOTIC
    constant identified EXACTLY as C_infinity = 2 lambda_* = 7.182242... ,
    lambda_* = 3.591121... the root of lambda(log lambda - 1) = 1.

Ingredients: RS (3.20) for T_n; RS (3.27) + twin-constant factorisation for V_n;
e_j(x) <= (sum x)^j/j! for the Bonferroni tail; RS (3.6) with
C(n-1,K) <= (e(n-1)/K)^K for the remainder; exact rationals 5 <= p_n <= 139, analytic
tail p_n >= 142. Two by-products: r22's "measured constant in [3.47, 4.16]" DOES NOT
CONTAIN THE LIMIT (6.14 - the ratio rises to 7.1822; the shortfall is the factor
(1 - (log log p_n + log K)/log p_n), still only 0.70 at p_n = 27449); and MAKING K
EXPLICIT IS FREE - the rule "least odd K with R_K <= V_n/2" picks the SAME K as r22's
numerical optimisation at every n, ratio 1.000x.

**2c. THEOREM 2E / 2E' / 2E''** (polynomial, FULLY EXPLICIT; exponent floor 15).
Round 23's first pass concluded rung 2 could not be made explicit; WRONG, corrected
inside the same round (6.24).

    2E (r23): j_2(p_n#) <= 1.0963e10 * p_n^19 * (log p_n)^10 + 1, p_n >= 285,
    every constant stated, NO ineffective threshold; more generally
    j_2(p_n#) << p_n^s for every real s > 18.308.

THE INGREDIENT is not a fundamental lemma - that was the first pass's mistake. It is
the EXPLICIT, CONSTANT-FREE Selberg Lambda^- Lambda^2 sieve, **Friedlander-Iwaniec,
Opera de Cribro, Theorem 7.7**:
`S(A,z) >= X V(z){1 - ((s+3)/(2 e^k))(2 e k/(s-3))^{(s-3)/2}} - 2 R_4(A,D)`,
s = log D/log z, k = kappa + log K, s >= 2k+3,
R_4 = sum_{d | P(z), d<D} tau_4(d)|r_d|, under
prod_{w<=p<z}(1-g(p))^{-1} <= K (log z/log w)^kappa. WHY IT APPLIES WITH NO WORK:
Dudek & Dunn (arXiv:2602.22720, Feb 2026, "An Explicit Result for the Sum of Two
Almost Primes") prove as Lemma 2.1 that the hypothesis holds with kappa = 2, K = 3 for
the multiplicative g with g(2) = 1/2, g(p) = 2/p - LITERALLY OUR omega(p)/p. Not a
coincidence: they sift n and N - n simultaneously, i.e. the Goldbach side of ZM Theorem
4.1, the same two-classes-per-prime problem. **METHOD NOTE: the explicit-Goldbach
literature is the natural source of explicit tools for the paired ladder.**

Re-derived here rather than trusted: K = 3 is exact and BEST POSSIBLE (grid search over
all (w,z), w,z < 2e5, returns exactly 3.000000; supremum at w = 3, z -> 3+);
k = 3.098612; FI's s >= 2k+3 = 9.1972 is NECESSARY BUT NOT SUFFICIENT - the bracket
only turns positive at s* = 18.30802, and equals 0.2507/0.5199/0.8202 at s = 19/20/22.
Pre-sieved K: 5/3 for p >= 5 (s* = 16.136), 1.4 for p >= 7 (15.474), 1.2624 for p >= 11
(15.077), 1.0479 for p >= 101 (14.353). ARITHMETIC: |r_d| <= 2^nu(d),
tau_4(d) = 4^nu(d) on squarefree d, so R_4 <= sum_{d<D} 8^nu(d) <= C_8 D (log D)^8 with
C_8 = e^{8 gamma} prod_{p<10^6}(1+8/p)(1-1/p)^8 = 0.0316 - that product is DECREASING,
so evaluating at 10^6 is a valid UPPER bound for every D >= 10^6; using the limit would
have been unsafe. Positivity needs m > (2/bracket) C_8 z^s (s log z)^8/V(z) with
V(z) >= 0.3905/(log z)^2 (6.18).

PRE-SIEVING (r24) changes exactly ONE factor in 2E's constant,
N_pre = prod_{p<p_0}(p - omega(p)) (|r_d| <= 2^{nu(d)} N_pre; X V'(z) = m V(z)
exactly). N_pre(5) = 1 - gear 3 keeps a single class - so pre-sieving 2 and 3 is FREE:

    2E'  (p_0 = 5):  j_2(p_n#) <= 3.5301e9  p_n^17 (log p_n)^10 + 1, p_n >= 285
         - smaller exponent AND constant, dominates 2E everywhere.
    2E'' (p_0 = 13, N_pre = 135):
                     j_2(p_n#) <= 7.2671e11 p_n^15 (log p_n)^10 + 1, p_n >= 285
         - dominates 2E' from the threshold on (395x at p_n = 285).

FLOOR: as p_0 grows, K -> 1, k -> 2, s* -> 14.169 > 14, so **EXPONENT 15 IS THE
SMALLEST INTEGER FI 7.7 CAN EVER GIVE at kappa = 2**, and p_0 = 13 attains it; the
ladder p_0 = 2..101 is priced, p_0 = 13 optimal at every p_n tested. General form:
j_2 << p_n^s for every s > 16.136 free, s > 14.822 at cost 135. The route below 15 is
the explicitness question of 7c, not more pre-sieving.

**2d. Rung 2 by citation: `j_2(p_n#) <<_eps p_n^{beta_2+eps}`, beta_2 = 4.266** - the
best proved dimension-2 sifting limit (Diamond-Halberstam-Richert; Franze
arXiv:1012.3809 Table 1, which also gives 4.516 for Lambda^2 Lambda^- at kappa = 2 and
shows Lambda^2 Lambda^- winning only from kappa >= 3). R21's cited 4.85/4.45 superseded
for free. Confirmed in three independent renderings, with Blight's 4.266450 at full
precision and Ford's 4.2665; Blight's OWN kappa = 2 value is 4.450 (worse) - her
full-precision figure quotes DHR, her improvement bites at kappa = 3. The 4.266 book is
Diamond-Halberstam-**Galway** (Cambridge Tracts 177, Zbl 1207.11099); the METHOD is
DHR. **THE EXPLICITNESS BOUNDARY**, stated so nobody re-attempts it: 4.266 is the
numerically-solved output of the DHR differential-delay system, and the sieve
inequality at that dimension carries an uncomputed O((loglog y)^2 (log y)^{-1/6})
error; even computed, the 1/6 means s = beta_2 + 0.01 needs log y ~ 10^12. **There is no
explicit-constant sieve AT its sifting limit for any kappa > 1.** So the note carries
TWO polynomial rungs and says which is which: exponent 15 fully explicit, exponent
4.266 not explicit and not makeable so.

**2e. THE CEILING (why exponent 2 is unreachable) - final form.**
- LEVEL IS NOT THE OBSTRUCTION: |r_d| <= 3^{nu(d)}, so sum_{d<D}|r_d| << D log^2 D and
  D = m^{1-o(1)} - the exponent is EXACTLY the sifting limit; no bilinear or
  well-factorable refinement of the level helps.
- ZM Conjecture 6 asks exponent 2 on a kappa = 2 problem - below Selberg's CONJECTURED
  optimum beta_kappa = 2 kappa (= 4 here); Selberg's own upper estimate is
  beta_kappa <= 2 kappa + 19/36 for large kappa (settled, 4c). The PROVED floor is
  beta_kappa >= (1+o(1)) 2 kappa/e (Brady 2017, improving Selberg's own by a factor 2),
  ~1.47 at kappa = 2 - so exponent 2 is NOT proved to sit below the sifting limit, only
  below the conjectured one.
- **WHAT ACTUALLY BLOCKS IT IS PARITY**, not an arithmetic fact about beta_2: exponent
  2 is exactly the level at which a survivor in (y, y^2] IS a prime pair (horizon
  theorem / Reduction A), so a dimension-2 lower-bound sieve at that level would
  manufacture two simultaneous primes - what Selberg's parity example forbids; hence ZM
  Thm 4.1 extracts Goldbach and Polignac from Conjecture 6. The sifting-limit numbers
  (4.266 proved, 4 conjectured, ~1.47 proved floor) CALIBRATE THE DISTANCE and leave
  "is beta_2 < 4?" genuinely separate.
- ANALYTIC SIDE (r23, from ODC): in ODC's beta-sieve (Thms 11.12/11.13, whose F, f,
  beta, A, B are all pinned exactly by (11.55)-(11.63); the only unevaluated object is
  an O((log D)^{-1/6})), **THE LOWER-BOUND CONSTANT B IS ZERO WHENEVER kappa >= 1/2**.
  Our kappa is 2, so the beta-sieve's lower bound is IDENTICALLY ZERO, not merely weak.
  Analytic and arithmetic side say the same thing: the natural tool cannot reach the
  natural target.
- Iwaniec's ordinary j(n) << (k log k)^2 is at primorials exactly p^{beta_1},
  beta_1 = 2 the dimension-ONE limit; Theorem 2 already gives the dimension-TWO
  counterpart (r21's "Iwaniec-analogue is the open wall" was the wrong slot - 6.13).

**2f. The nested truncation** (validity SETTLED; main term is the open piece). The
PER-BAND product truncation {d : nu(d_j) <= K_j for all j} is NOT a valid lower-bound
sieve - 36 explicit counterexamples. The correct object counts the WHOLE UPPER TAIL:
nu(d restricted to primes above z^{alpha_j}) <= H_j, nested, H_j = 2h_j+1 lower /
2h_j+2 upper - the refinement Tenenbaum describes before his fundamental lemma (GSM 163
p.70, Exercise 86, proved nowhere there). TESTED, NOT ASSUMED: 168,400 (depth pattern,
bad-count) configurations over 1, 2, 3 partition points, ZERO violations of
Lambda^- <= [survives] <= Lambda^+, against 36 failures for the per-band form. Monotone
depths are a LEVEL-COST convenience, NOT a validity requirement (6.26). REMAINING GAP,
exactly one object: an explicit MAIN-TERM estimate for the nested truncation (explicit
lower bound on sum_{d in D^-} mu(d) g(d) against V(z)); own level/error accounting says
exponent ~9 is reachable (theta = 1/2, geometric depths ceil(4 x 1.05^{j-1}), s = 9.07
at truncation cost 0.36). Its HR-Memoire / ODC-Ch.6 form: 7c.

## 3. THE j_2 LOWER LADDER (docs/novel/j2-lower-ladder.md)

**3a. RESTATEMENT** (proved; brute-forced exact against ZM at z = 3..13):
`j_2(P(z)) - 1 = the longest [1, L] such that for some even E, every z-ROUGH n has n+E
divisible by some p <= z.` Smooth numbers are covered FREE by the 0-classes, so the
paired covering only has to reach a set ONE LOG THINNER than the ordinary problem's
"every integer" - that factor, one logarithm once, is the whole structural separation
between j and j_2. By CRT the killed residues mod p are {-a, -a-2e} with a and e
independently free, so j_2(p_n#) - 1 is exactly the longest interval coverable by TWO
ARBITRARY classes per odd prime.

**3b. THEOREM (P1)** - the first lower bound using the paired structure:

    h_2(P(z)) >= (1/(2 e^{-gamma} C_2) + o(1)) z log z = (1.349 + o(1)) z log z.

GREEDY PHASE (each odd p <= z^{1-eps} kills the largest nonzero class of the surviving
rough numbers, shrink factor (p-2)/(p-1), product = the twin constant's 0.7413/log w) +
MATCHING PHASE (one unused prime per survivor) + CRT for E. Beats the r21 collapse
transfer ASYMPTOTICALLY (FGKMT is o(z log z)) and at every finite z, with an explicit
constant. Certificates built and re-verified BY INDEPENDENT SIEVE (disjoint code path)
at z = 13..10^5; at every z with known h_2, L <= h_2. As RUN the greedy beats its
worst-case analysis: L/(z log^2 z) settles near 0.7 over z = 10^3..10^5 (spread <
1.25x) while L/(z log z) climbs 4.7 -> 8.5 - the CONSTRUCTION already tracks the
one-extra-log law; only the PROOF loses it. Earlier rung retained - LOWER TRANSFER
(r21): b - a = p_n# collapses paired to ordinary (survivor sets equal, verified exactly
n = 3,4,5), so j_2(p#) >= j(p#) and FGKMT lower bounds transfer verbatim.

**3c. THE GROWTH LAW REREAD (r24) - the sandwich in final form:**

    proved lower  h_2 >= (1.349+o(1)) z log z   [P1]
                  (j(p_n#) = p_n^{1+o(1)}; measured exponent 1.10-1.22)
    TRUTH         z^{1+o(1)}, best model ~2.56 z (log z)^2
                  (measured local exponent 1.75-1.95)
    proved upper  p_n^{15} explicit / p_n^{4.266+eps} by citation

The r23 reading "TRUTH ~ (p^2-p)/2" is RETRACTED (6.28). On ZM's 21 exact values: c z^2
and c z (log z)^2 fit EQUALLY (implied-constant spread 1.87x EACH; the laws differ by
z/(log z)^2, which only moves 2.1x across the table - 21 points cannot separate them);
residuals drift in OPPOSITE directions (h_2/(z^2-z) falls 0.962 -> 0.499,
h_2/(z log^2 z) rises 1.754 -> 1.951); the parameter-free extreme-value model puts h_2
at ~2.56 z (log z)^2 with measured/model 0.78-0.92 (ordinary j: 0.34-0.47 of ITS model
- the two problems behave alike under it); and the DISCRIMINATING measurement, the
paired-minus-ordinary local exponent gap, is 0.33-0.75 on every range tested - the
signature of a LOGARITHMIC separation (predicted 0.25-0.50), nowhere near the +1.0 a
quadratic-vs-linear law needs. FALSIFICATION TARGET, decidable and named: **one exact
h_2 beyond p_n = 73** (at z = 151-251 the log-law models sit 2.6-3.6x below the
quadratic; ZM's algorithm reached 73 on 2017 hardware) - the single most decisive number
purchasable here.

**3d. Named open problems, corrected.** (P2) `h_2 >> z (log z)^2/(loglog z)^{O(1)}` via
Rankin/FGKMT LAYERING - still a construction, still parity-free; replaces r23's
"h_2 >> p^{1+delta}", which asked for something probably false (6.29). (P3) the
paired-Iwaniec UPPER question `h_2 = O(z (log z)^A)`. (P4) on this model ZM Conjecture 6
is TRUE WITH ROOM - it asks far less than the truth; coexists unchanged with r22's
sieve-side point (exponent 2 on a kappa = 2 problem is below the conjectural floor):
easy as a statement about the truth, hard as a statement provable by sieve. PRIOR ART
(checked 2026-08-28): Kalmynin-Konyagin arXiv:2302.00459 ("A polynomial analogue of
Jacobsthal function", full text on disk) is nearest - Rankin machinery on shifted
polynomial VALUES; for quadratic f the killed classes are the <= 2 square roots of ONE
global shift, so neither family contains the other, their covered object is a polynomial
sequence not an interval, and paired/two-residue Jacobsthal appears nowhere. NOT prior
art for (P1); STRONG evidence for (P2) (their M(f) = 2 buys two Rankin-type logs on a
2-class sieve). FKMPT "Long gaps in sieved sets": one class per prime per the search
relay - flagged RELAY-SOURCED, re-verify before citing.

## 4. THE VERIFICATION RECORD

**4a. Opera de Cribro checked directly - THEOREM 2E's foundation STANDS.** R23 used ODC
Thm 7.7 from two independent verbatim transcriptions agreeing exactly (Dudek-Dunn Thm
1.3; Campbell arXiv:2608.09488 Thm 2.1, both read in full 2026-08-25) and recorded that
the book had NOT been consulted. R24 consulted it: the book's own text (Google Books OCR
of the AMS printing, harvested by a sub-search with OCR and reconstruction reported
separately) shows **Theorem 7.7 on p. 111, Chapter 7 (Selberg Lambda^2 Lambda^-)**,
matching both transcriptions in every particular - statement, hypothesis s >= 2k+3,
bracket, remainder 2 R_4. Three renderings now agree, one of them the book. **HONEST
RESIDUE: seen through OCR, not held - the check is of mathematical content, not
typography.** SAME-PAGE DEAD ENDS, closed with numbers: **(7.122)** is a LOOSE
sufficient condition - 2k + 2 sqrt(2k log k) + log k + 9 = **21.6** at our k = 3.0986,
weaker than the exact s* = 18.308; **Corollary 7.8** is asymptotic-in-k and buys nothing
at k ~ 3.1. THE THREE ODC CONSTANT-FREE RESULTS PRICED (thresholds re-derived here from
the stated inequalities and asserted): Thm 6.9 (D >= z^{9kappa+1}) is positive iff
s > 9 kappa + 10 log K; Cor 6.10 (only D >= z >= 2, NO hypothesis on s) needs
s > 9 kappa + log(4(9kappa+1)^kappa K^11); Thm 7.7 is 2E's bracket.

                    K = 3        K = 1.097 (pre-sieved at 3)
    ODC Thm 6.9    s > 28.986    s > 18.926
    ODC Cor 6.10   s > 37.360    s > 26.294
    ODC Thm 7.7    s > 18.308    s > 14.532

THEOREM 7.7 STANDS - K^10 is brutal at K = 3 (10 log 3 = 10.99 on its own). Thm 6.9 is a
cleaner-looking fallback; Cor 6.10's value is assuming nothing about s. Every figure
reproduces an external collaborator's to three decimals from own code.

**4b. The Halberstam-Richert Memoire OBTAINED; 7.972 re-derived.** "A new look at Brun's
sieve", Mem. Soc. Math. France 25 (1971) 97-106 - free numdam scan, located and read in
r24. VERDICT: **the 7.972 lead is REAL, DERIVED NOT PRINTED, and the exponent-8 route is
an EXPLICITNESS problem.** It treats EXACTLY our density (worked example A = {n(n+2)}:
omega(2) = 1, omega(p) = 2), and its two printed conditions - (1.2)
lambda e^{1+lambda} < 1, and positivity lambda^2 e^{2 lambda}(2 + e^2) < 1 - admit every
level exponent u > 1 + 2.01/(e^{lambda*} - 1). The figure 7.972 is NOT in the paper (it
says only "u < 8"); RE-DERIVED here from the printed conditions: **lambda* = 0.2533219,
u = 7.971954833**, asserted in research/j2_presieve.py P4 - the lead is confirmed
independently of whoever first reported it. Every remainder in the Memoire is an
unspecified O(.), so what r23 called "one missing piece: an explicit main-term estimate
for the nested truncation" is now precisely located: **make THIS 1971 theorem explicit.**

**4c. 19/36 vs 0.4454 SETTLED FOR 19/36, first-hand.** Honest form: "19/36
three-sourced, 0.4454 unverified", not "0.4454 wrong". FIRST-HAND, fetched 2026-08-28
via the zbMATH Open API: Greaves' review of Selberg's OWN announcement (Oslo 1987, Zbl
0675.10030) - "alpha_k > 1/(2k+19/36) for all sufficiently large k" (reciprocal
convention); Heath-Brown's review of Franze (Zbl 1235.11089) - Selberg "showed that the
sieving limit satisfies beta_kappa <= 2 kappa + 19/36 + o(1)". FIRST-HAND from the
on-disk Franze full text: the pp. 174-176 computation RE-DERIVED IN EXACT RATIONALS -
optimal a = 1/4, threshold d = -7/72, constant EXACTLY 19/36. NUMERICAL: at
2 kappa + 0.4454 the Selberg functional's main term is strictly negative (-0.0369) - no
lower bound lives there; Franze's own table (kappa = 2..10) rises to 0.525, every entry
already above 0.4454. LABELLED SPECULATION on the origin: Greaves' review carries "a
certain constant c close to 1/2.445" ONE SENTENCE before the 19/36 - the digit string
sits beside the true constant in the primary source's own review. RESIDUAL: the printed
Selberg (14.40) remains unread (in-copyright, no scan found); one page scan closes it
forever. Selberg's conjecture beta_kappa = 2 kappa is NOT in Franze (the word
"conjecture" does not occur there) - the source is Selberg's *Lectures on Sieves* sec.
14, restated in Blight's Rutgers thesis sec. 2.1.

**4d. The citation-numbering sweep (now a standing referee step).**
- **"IWANIEC-KOWALSKI THEOREM 6.9" DOES NOT EXIST** - a chimera, and it had reached two
  of our documents. IK Ch. 6 ("Elementary Sieve Methods") stops at Theorem 6.7; in IK,
  6.9 and 6.10 are EQUATION labels, and the 6.9/6.10 THEOREM numbering belongs to Opera
  de Cribro - the two were conflated. IK's "s >= 9 kappa + 1 with K^10" result is IK
  **Theorem 6.1 / Corollary 6.2** (p.158); IK's **Fundamental Lemma 6.3** has no lower
  bound on s but hides its K-dependence inside an O to the tenth power, so it is not
  explicit either. CLEAN AS WE HAD IT: "Friedlander-Iwaniec Opera de Cribro Thm 6.9" IS
  a real fundamental lemma; both our uses stand.
- Tenenbaum's fundamental lemma is **Theorem 4.4** (Theorem 3 in the 1995 CUP edition),
  not 4.3; "Theorem I.4.2" does not exist (I.4.2 is a COROLLARY, the Bonferroni
  inequality). Checked clean: no document of ours ever cited "Tenenbaum I.4.3" for the
  fundamental lemma - it lived only in a working note. Nathanson Ch. 6 is a DEAD END (no
  general-dimension sieve at all).
- arXiv:1012.3809 is by **C. S. (Craig) Franze**, not "M. Franze" (JNT 131 (2011)
  1962-1982). Costello-Watts' 2 e^gamma k^{5+5 log log k} rung is **arXiv:1306.1064**,
  not 1208.5342 (a range-restricted computational bound, 50 <= k <= 10000). Iwaniec's
  theorem is h(k) << (k log k)^2 with k = omega(n), equivalently J(P(z)) << z^2; the
  "(log n)^2" phrasing is a weaker corollary.
- Franze says 2 kappa + 19/36 where Ford (2023) and Brady (2017) both give
  2 kappa + 0.4454 from the same Selberg equation (14.40) - a genuine conflict, FLAGGED
  in r23 rather than picked, then settled in 4c.
- DO NOT use Yamada arXiv:1511.03409 Theorem 3.1 as an alternative explicit sieve
  (unproved as stated).

**4e. The referee pass - research/j2_referee.py, STANDING ARTEFACT.** Recomputes every
recomputable numerical claim of Unit 1 by INDEPENDENT code; the per-difference family
arrays are rebuilt from scratch at y = 3..17 and compared ELEMENTWISE against r20's
f13/f17 arrays - identical. Caches research/data/ref_fam_<y>.npy (seconds after the
first run). **Re-run before any future claim about Unit 1.** REPRODUCED CLEAN: the h_2
table and #diffs, the margin column, all four tie-aware percentile rows, the 31-class
F_max/lambda spread 2.88..7.52, the delta-profile law at 100% precision AND recall, the
13->17 cap law (272 lifts, extension multiset {81:208, 84:32, 87:32}, best 87, THE EXACT
9), the b-a = p# collapse, Theorem 1's explicit chain, and the y=19 winner set reaching
G = 43. FIVE DEFECTS FOUND, all in our own documents: 6.15-6.19; the sharpest is the
y = 3 row.

**4f. Novelty re-checked by citation graph (2026-08-25), not keywords** - method upgraded
because keyword sweeps are what missed Holt in r22. Semantic Scholar: arXiv:1706.00317
has EXACTLY ONE citation in nine years (ZM's own companion note); 1706.03668 has ZERO.
zbMATH Open: "paired Jacobsthal" returns NO RECORDS AT ALL; OpenAlex full text: only the
two ZM records. OEIS A288815, pulled again (record stamp #19 Apr 12 2026): 21 terms, two
links (both ZM), comment states only the conjecture, no proved bound deposited. arXiv API
metadata sweep over the COMPLETE math.NT Jacobsthal set (54 records) and all-category
listings: every 2025-2026 Jacobsthal item concerns Jacobsthal NUMBERS, SUMS, POLYNOMIALS
or CONGRUENCES - a different Jacobsthal; none touches the Jacobsthal FUNCTION. Holt
arXiv:2502.20470 re-examined for Unit 1 specifically: full text downloaded, "Jacobsthal"
occurs ZERO times - the r22 downgrade was real but touches Unit 2, NOT Unit 1 (recorded
so no future round over-corrects). The ORDINARY ladder's frontier re-verified against the
live Erdos-problems database (problems 970 and 687, fetched 2026-08-25): **IWANIEC 1978
IS STILL THE RECORD IN AUGUST 2026** - FGKMT 2018 improved only the lower bound,
Costello-Watts only explicit constants. VERDICT: NOVEL, re-confirmed 2026-08-25 with
stronger evidence than any previous sweep.

## 5. OTHER RESULTS

**5a. Kernel-checked** (proofs/Polignac.lean unless noted; standard axioms [propext,
Classical.choice, Quot.sound] or fewer; ledger green, zero sorry). Headline theorems:
**slot_cap_gap** (an odd prime blocking both members of a gap-2d slot => q | d) - THE
EXACT TRANSFER CONDITION: every slot-cap law holds verbatim for gap 2d at gears coprime
to d, and q | d gears collapse to one residue (the HL factor, mechanically);
**gapPairs_infinite_iff_survivor_in_window (d)** - THE PER-GAP IFF: Polignac for 2d <=>
"every scale has a window (y, y^2] containing a gap-2d survivor of the gears <= y", both
directions, every d, sharper than ZM Thm 4.1 (sufficient-only, all differences at once);
**goldbach_of_survivor** + exact converse survivor_of_goldbach_rep (the C4 window
reduction); **own_slot_pin_gap_two** (UNIQUENESS: an odd prime pair (q, q+g) split-killing
the slot holding q itself forces g = 2; other gaps sit at depth ~P/(6g)); **twin_pin**
(the twin pair IS slot u = (p+1)/6) and twin_pin_self_block (the machine is blind to its
own pair); **card_class_Ico** (THE FLOOR COUNT (t+m-a)/m); **twoSided_class** (THE GENERAL
BOTH-SIDED TERM: coprime mL, mR give ONE CRT class mod mL*mR); **three_gear_master**
(END-TO-END 26-term subtraction-free identity, rearranging to overcount = pairs - triples;
the 3-gear assembly line is CLOSED, n > 3 mechanical and deferred);
**endpoint_run_mod_three** (THE ENDPOINT LAW: both flanks unblocked by gear 3 =>
F(2,y) = 0 mod 3, the pruned search's mod-3 skip; all thirteen known exact values 33..309
comply) plus the LEFT-TAUT EQUIVALENCE (paper proof; every gear drops offsets q-2, q-1;
exhaustive y <= 17); and the **MOD-3 DICHOTOMY FOR F_d** - the complete sharp iff
`3 | F_d(y) for every gear set <=> d != 0 mod 6`, verified 15/15 gap classes first.
Supporting: prime_of_no_factor_le_sqrt, SurvivorGap/survivorGap_iff_pair, the same-side
census (r3), PAIRSPLIT (r4, incl. split_rep_twin_eq_pin), CORR triples (r5), the assembly
lemmas (r6-7). Code: rust2/src/bin/maxgap_pruned.rs (endpoint law + left-taut, identical
to the original on F(2,y) = 21,33,54,75,102,129,264 for y = 11..37).

**5b. Computed values, replications, winner sets.**

    y    P (odd)   #diffs    h_2   p^2-p   Conj.6   margin
    2         1        -        2      2    FAILS BY EQUALITY
    3         3         1       6      6    FAILS BY EQUALITY   <- corrected, 6.15
    5        15         7      18     20    HOLDS    10.0%
    7       105        52      30     42    HOLDS    28.6%
    11     1155       577      66    110    HOLDS    40.0%
    13    15015      7507     150    156    HOLDS     3.8%   <- the dip
    17   255255    127627     192    272    HOLDS    29.4%
    19        -         -     258    342    HOLDS    24.6%
    23        -         -     366    506    HOLDS

REPLICATION STATUS: an exact independent REPLICATION, not a first computation (6.1).
Additionally replicated here by ENTIRELY DIFFERENT METHODS: **h_2(19) = 258** by
exhaustive family scan in delta space (r22; max G = 43 over the whole family);
**h_2(23) = 366** EXHAUSTIVELY (r22; the prefilter keeps 128 of 37,182,145 deltas =
0.00034%, all reach G = 61, none exceeds it); **h_2(29) = 450** given an independent
explicit LOWER-BOUND WITNESS (5d) - three consecutive ZM values confirmed here by three
different routes. COMPLETE WINNER SETS (delta space): ladder **8, 16, 64, 64, 128** at
y = 11,13,17,19,23; the 19-winners are NOT lifts of the 17-winners; the 3 | e branch
settled EXHAUSTIVELY (for 3 | e a gap of 3G needs killed runs in BOTH sub-lattices, each
a translate of the same S_delta, so its delta must already be a G >= 43 winner; checking
those 64 gives best F = 44 against 129).

THE DELTA REDUCTION (r22): for 3 not dividing e, F_e(y) depends on e ONLY through
delta = e*3^{-1} mod Q and equals 3*G(delta), G the maximal cyclic gap of
{k : k != 0, -delta mod q} (gear 3 pins survivors to one class mod 3; n = 3k+c turns each
tooth pair into a translate of {0, -delta}). Plus a HELD-OUT-TOP-GEAR PREFILTER that is
EXACT not heuristic: a run of L killed positions forces every survivor of the smaller
gears in the window into {0, -delta} mod qt, pinning delta mod qt. Validated against
brute force in delta space at y = 13 and 17. NOVELTY: essentially ZM Prop 1.5(2) (6.11).

CROSS-CHECK AGAINST ZM'S ANCILLARY DATA (r22 - the project's best cross-check and a
self-found novelty downgrade). ZM's full_details.pdf Table 1 carries nseq = "number of
sequences of maximum length" (1,6,1,1,4,2,2,14,... at p_n = 5..29) with exhaustive
ancillary lists (remainders_2.txt / permutations_2.txt / moduli_2.txt). Converting each
winning delta's record windows into ZM's covering pattern:

    y = 11:   8 deltas,   8 windows -> 1 pattern = nseq 1 (self-symmetric)
    y = 13:  16 deltas,  16 windows -> 1 pattern = nseq 1 (self-symmetric)
    y = 17:  64 deltas, 128 windows -> 4 patterns = nseq 4
    y = 19:  64 deltas, 128 windows -> 2 patterns = nseq 2
    y = 23: 128 deltas              -> 2 patterns = nseq 2 (pre-registered check 1,
                                       PASSED)

EXACT at all five, reverses counted separately exactly as ZM state, and ZM's own remark
that the single sequences at n = 5, 6 are self-symmetric is reproduced.

DELTA-PROFILE LAW: maximisers are exactly the carriers of specific profiles - (1,1,1,3)
at gears <= 11 (8 winners, F = 33), (1,1,1,3,6) at <= 13 (16 of 7507, all F = 75; recall
and precision 100%), (1,1,2,4,6,8) and (1,1,2,3,4,3) at <= 17 (64 winners, precision
100%, recall 50/50). Every winning profile begins delta_3 = delta_5 = 1. "Maximally
spread at the top" fits some maximisers, not all - description, not law. (The published
maximiser LISTS were truncated slices - 6.16.) FIXED-TWIN LADDER (ours, not ZM's):
F(2,37) = 264, F(2,41) = 273, F(2,43) = 309, F(2,53) >= 426 (needs <= 486 for the
tolerance constant; quadratic-law prediction ~441). OEIS A288815: F = h_2/2 = 75, 96,
129, 183, 225, 285, 354 at y = 13..37.

**5c. WHY 13 IS EXTREMAL - CLOSED as four exact events (r20).**
1. **QUANTISATION.** The slack B - h_2 is quantised mod 6 (min admissible 6 for
   p = 1 mod 6, 2 for p = 5 mod 6). The minimum is attained at p = 5 and p = 13 ONLY,
   through 73. The 13 dip is "one quantum above equality": omega_2(6) = 24 = cap 25 - 1.
2. **STEP LAW** over ZM's 18 steps: margin falls at ALL 6 twin steps (>= 13), rises at
   ALL 5 gap-6 steps, gap-4 mixed (3 up, 2 down); absolute slack falls ONLY at twin steps
   (->13, ->31, ->61). Crossover: d(B)/B ~ 2g/p vs d(h_2)/h_2 ~ 2r/p, so the sign flips
   at gap ~ r (mean ~2). A dip needs r >> g.
3. **UNIQUE JUMP.** r = Delta(maxF)/q' = 3.231 at 11->13 is the unique value > 2.6 in all
   18 steps (runner-up 2.553 at ->47). The dip = that outlier landing on a twin step.
4. **THE LAST CLEAN-EXTENSION STEP.** Winners extend winners at 7->11 and 11->13 (16/16
   winners at 13 have F_11 = 33 -> F_13 = 75, same fixed e), and NEVER again: the best
   17-extension of a 13-winner is 87 vs true max 96; the 19-argmax restricts to the twin's
   own value (54) at 17 with 35,848 classes above it. So 13 is where the family maximum
   last grows by full profile extension - on a twin bound-step.

ZM's computation note has NO growth-rate commentary and no remark on the 13 case, so the
step law has no counterpart in print.

**5d. The shallow-extension CAP LAW and the extension ladder (r21-22).** CAP LAW (proved
under stated non-collision conditions): a maximiser's record window is a maximal gap - NO
interior openings - so lifting to gear q' it can only grow by FUSING adjacent gaps;
interiors must sit in the 2-element tooth set mod q', and 3 interiors would need 3
distinct residues in a 2-set, so AT MOST TWO adjacent gaps ever fuse and the lift choice
grants any single separation congruence: `best extension = F_old + best adjacent 2-gap
sum`. CAVEAT: the 3-interior impossibility needs the non-collision conditions (q' not
dividing F_old or the adjacent separations) - the collision case is exactly PADDING.
THE EXACT 9 (13->17): all 16 13-winners have the SAME local context (..6,3,6,[75],6,3,6..);
75 = 7 mod 17 so e = +-7 mod 17 fuses both flanks, cap = 6+75+6 = 87, and the exhaustive
extension value set over 272 lifts is exactly {81, 84, 87}. The winner 96 is a 4-5-gap
DEEP fusion on mediocre bases (F_13 in {42, 51}); **THE 9 = 96 - (6+75+6)**. Anatomies:
111-window = [96,6,9], 147-window = [129,6,12] - one-sided two-gap chains at the cap.
DEFICIT LADDER over COMPLETE winner sets, independent code: **9, 18, 36, 0** at 13->17,
17->19, 19->23, 23->29; the identity `deficit = increment - (record's best adjacent 2-gap
sum)` survives intact, 2-gap sums 12, 15, 18, 42.
THE 23->29 ZERO, CERTIFIED not merely computed: 23-winners lift to the FULL y = 29 family
maximum G = 75, F = 225, h_2 = 450; witness delta_29 = 743,911,918 (from
delta_23 = 269,018, lift r = 3 mod 29) has k = 134,406,257..134,406,330 - 74 consecutive
positions - each killed by an explicitly listed gear, both flanks open on every gear
(three further witnesses, all r = 3 mod 29). AND NOT ONE LUCKY WINNER: over the complete
128 winners x 29 lifts EVERY one reaches G = 75, each at exactly the same four lift
residues r in {3, 12, 17, 26} mod 29 = {+-3, +-12} - 512 pairs, no other r works. Those
residues are precisely the two interior separations available in the fused word (openings
at 0, 2, 14, 75, 77, 79: the 75-gap is 0 -> 75, killing 2 and 14, separation 12, forcing
delta = -+12; or its mirror 2 -> 77, killing 4 and 65, separation 61 = 3 mod 29, forcing
delta = -+3). **At this rung the cap law does not merely BOUND the extension - it PREDICTS
the admissible lifts exactly.** The gap word [2,12,61,2,2] fuses 61+12+2 = 75, exactly the
cap law's maximum attained; in F units 183 + (36+6) = 225. HONEST LIMIT: one more rung,
not a law - the 2-gap sum beside a record is an arithmetic accident of that neighbourhood,
and the 29-winner set would be a 1.08e9-delta scan, out of reach for this prefilter
(y = 23 cost ~3 CPU-hours x 4 shards).

**5e. TWIN PERCENTILE - the twin case is the EASY end of its own family.** At gears <= 13,
coprime-to-P differences (2880, the hardest class): F_e range 30..75, mean 38.83, median
39; twin F = 33 = **13.3rd percentile** (rank 385 of 2880), 77.2% of coprime differences
have a LARGER maximal gap, extremal 2.27x twin. At gears <= 17: twin 54 vs max 96 (1.78x),
**21.3rd percentile**, strictly-above 68.6%. Twins have delta_q = 1 for every q - the
maximally clustered member. F_max/lambda ranges 2.88 (gcd = 5005) to 7.52 (gcd = 3) over
the 31 gcd classes at <= 13: **density does NOT determine the extreme.** EXTERNALLY
VALIDATED using ZM's h_2 as an independent family-max denominator: twin/extreme known at
12 machines y = 5..43; twin attains the max only at y = 7; extreme runs 1.34x-2.27x twin,
median 1.70x (y >= 11); the 0.746 share at 37 is twins' own 2.432 q' outlier jump,
relaxing immediately. PUBLICATION STATEMENT: "in the one family where difficulty is
exactly measurable, the twin case sits at the 13th-21st percentile of its own hardest
class, the extreme is 1.3x-2.3x harder at every one of twelve machines (externally
cross-checked against Ziller-Morack's independent table), and density does not determine
the extreme." CONSEQUENCE: "the method handles the twin case; the general case is similar"
is MEASURABLY FALSE, and a method reaching extremal differences would give all of Polignac
at once.

**5f. Paired Holt recursion and paired HL-B in cycles (r20-22).**

PRIOR-ART VERDICT FIRST (r22, self-found). R20/21 searched Holt arXiv:1510.00743 and
Holt-Rudd arXiv:1408.6002 and found no paired counterparts. R22's re-search surfaced a
paper that DID NOT EXIST at that time: Fred B. Holt, "Eratosthenes sieve supports the
k-tuple conjecture", **arXiv:2502.20470** (v1 Feb 2025, v3 Jul 2025). His Corollary 1: for
an admissible constellation s of length J, `sum_{j>=J} n_{s,j}(p#) =
prod_{q<=p}(q - nu_q(s))`, nu_q(s) = distinct residues mod q among the J+1 boundary
points. A twin-slot survivor is EXACTLY a gap of 2 in Holt's cycle of gaps, so a pair of
twin-slot survivors at lag g is an instance of his constellation (2, 6g-2, 2) with boundary
points {0, 2, 6g, 6g+2} = H_g. THEREFORE: our LOCAL-FACTOR IDENTITY c_q(g) = q - nu_q(H_g)
(r21) is Holt's q - nu_q(s) specialised - the affine-bijection proof is still the right
proof of the closed form, the identification is his framework; LATERAL'S DEPTH-SUM
IDENTITY sum_j W_j(g) = N2(g) (r20) IS Holt's Corollary 1 at that constellation - identity
and proof correct, the novelty claim not (flagged to Lateral; their doc not edited, verdict
recorded in docs/novel/README.md's index entry); "THE PAIRED SYSTEM IS HOLT'S WITH DOUBLED
LEVEL SPACING" is now DERIVED not observed (a paired gap word of length j is a
constellation with 2j+2 boundary points and his dynamics carries diagonal q - (number of
points), so q - 2j - 2 against his q - (j+1)) - better understanding, weaker novelty;
Formalist's kernel check of the local-factor identity is unaffected AS VERIFICATION.
CHECKED, NOT ASSERTED (research/holt_correspondence.py): (A) twin-slot survivors ARE
exactly the left endpoints of the gaps of 2 in the rough cycle (sets equal, 1,485 at
P = 30,030 and 22,275 at P = 510,510); (B) N2(g) = prod_q c_q(g) equals to the unit Holt's
right-hand side at every g <= 6 and both machines; (C) the objects separate at once -
machine 17, g = 5: n_g = 4,230 vs Holt's n_{s,J} = 0. WHAT SURVIVES, and why it is a
different object: Holt's n_{s,J} counts constellation instances with NO ROUGH NUMBER
between the boundary points; n_g counts CONSECUTIVE TWIN-SLOT SURVIVORS - no twin candidate
between, ordinary rough numbers allowed. The twin-slot subsequence of Holt's cycle is not
studied in his papers and n_g is none of his n_{s,J}. Also clear: Holt arXiv:2603.25915
(Mar 2026) is one-residue, Legendre-directed, nothing paired.

What we have on n_g. (1) **PAIRED HOLT RECURSION** (r20): exact linear population dynamics
for two-residue sieves, n_g(M+q') = sum_w coef(w) n_w(M) with POSITION-FREE
coef(w) = #{r in Z_q' : flanks alive, interiors in T}; EXACT for every gap value at 4 rungs
(5005 -> 85085 -> 1616615; family e = 344 +17; gcd collapse e = 102 +17 = Holt's own case);
diagonal = Lateral's c_q(g) law exactly (two lanes' constructs are one object); eigenvalue
scale (q'-2j-2)/(q'-2) vs Holt's (q'-j-1)/(q'-2). (2) **WORD-LEVEL TRANSFER** (r21): the
full word census - n_w(M+q') exact for all 6714 words (sum <= 24) at 5005 -> 85085 and
10489 at 85085 -> 1616615, by deterministic per-copy image enumeration; the
census-to-census linear map is a verified exact object. (3) **EIGEN-ANALYSIS**: aggregated
by (sum, length) the paired transfer is generically diag(q-2j-2) + superdiag(2j) (sporadic
share 6.9% at +17, carried exactly by the word-level transfer), diagonalised by
v^(k)_j = (-1)^(k-j) C(k-1, j-1) - q-INDEPENDENT Pascal eigenvectors, IDENTICAL to Holt's
one-residue system; exact rationals, q in {17,19,101,997}, k <= 12. (4) **PINCH THEOREM
generalised to the exact BONFERRONI SERIES**: with S_k = sum over 0 < t_1 < ... < t_k < g
of N_{k+2}(0,t_1,...,t_k,g), inclusion-exclusion over which interior offsets are open gives
EXACTLY `n_g = sum_{k>=0} (-1)^k S_k`, truncations alternating (even K upper, odd K lower);
K = 0 and K = 1 ARE the two sides of the original pinch
`N2(g) - sum_t N3(0,t,g) <= n_g <= N2(g)`. MOMENT FORM: S_k = sum_j C(j-1,k) W_j(g), so
S_0 = N2 is the depth-sum identity and S_1 overcounts sum_{j>=2} W_j by exactly
sum_{j>=3} (j-2) W_j - THE PINCH'S SLACK IS AN EXPLICIT QUANTITY. Verified by full sieve at
machines 13/17/19. (5) **PAIRED HARDY-LITTLEWOOD CONJECTURE B HOLDS PROVABLY INSIDE THE
SIEVE**: fixed-gap population ratios converge AT RATE 1/log^2 y to HL quadruplet
singular-series ratios (finite products, factors cancel beyond q = 6g+2); n_5/n_4 -> 3.150,
pinched to [3.06, 3.22] at y = 10^6. (6) **EFFECTIVE POLIGNAC IN THE PAIRED SIEVE**: with
y_0(g) the least y at which the lower bound is positive, gap g occurs in M_y for EVERY
y >= y_0(g), unconditionally, no scan (Holt proves constellations "arise and persist" but
gives no stage index; this is a number):

    g            2   3   4   5   6   8  10   12   15   20    25    30   40     50
    y_0 order 1 14  20  26  32  38  50  62  103  199  467  1009  2609  12157  42257
    y_0 order 3  -   -   -   -  41  53  67   79   97  127   167   367

with log y_0(g)/sqrt(g) in [1.305, 1.531] at order 1 and about 1.08 at order 3:
**y_0(g) = exp(Theta(sqrt g)), NOT polynomial in g**; higher Bonferroni orders improve the
CONSTANT but not the SHAPE, so the square root is not a union-bound artefact and a
polynomial threshold needs a different argument. (7) **NEGATIVE, PRICED SO NOBODY
RE-DERIVES IT**: every gap <= G(y) occurring gives F(2,y) >= 3 G(y) ~ c (log y)^2 - 60, 90,
180, 240 at y = 10^3..10^6 against a truth of order y^2; the pinch contributes NOTHING to
the j_2 lower ladder, which stays with (P1) and the FGKMT transfer. (8) **THE BOUNDARY,
QUANTIFIED**: the pinch is a FULL-PERIOD statement while primality lives in the window
(y, y^2], a share y^2/P_y = exp(-(1+o(1)) y) of the period - 2.2e-4 at y = 19, 1.1e-9 at
y = 37, 2.6e-34 at y = 101. No full-period population statement, however exact, localises
into a share that thin. That is the entire distance between "paired HL-B in cycles, proved
with rate" and "paired HL-B for primes, open"; nothing here proves anything about prime
quadruplets and no prime-side consequence was found.

**5g. Universal cap, route transfer, padding (r10-15).** UNIVERSAL CAP TABLE (COMPLETE over
all even d: the spectrum depends only on gcd(e,105), e = d/2, all 8 divisors computed;
48-class mod-105 invariance, zero mismatches, q' <= 1200):

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
kernel-checked slot_cap_gap; the HL factor and the exposed-set size are the same object).
**12 IS THE ABSOLUTE CEILING OVER ALL POLIGNAC GAPS** (N1).

ROUTE-TRANSFER AUDIT (r13-14): the tolerance route is a THEOREM SCHEMA over all even d with
ONE open lemma. (A) the finite word list from q' mod 105 transfers verbatim (48 classes,
zero mismatches; sizes {1,2,3,5,8} for 3 not dividing e, {11,12,20,21,23} for gcd = 3,
{43..56} for gcd = 15); (B) literal span <= ceil((cap_d-1)/2) x q' frame units; (C) padded
count p <= F/c_d, onset gated by F >= c_d, 8/8 zero violations; (E) both-flanks-maximal
exclusion forbidden for 68-82% of probes across d = 2,4,6,12; **(D) flank bound
FS_max(w) <= F + (alpha/3)q' - span(w) contains no d-specific structure - THE SAME OPEN
LEMMA for every even d.** BUDGET (exact full periods, steps 11->13..23->29): max incr/q' by
d - 2: 1.235; 4: 1.846; 6: 0.947; 10: 1.421; 12: 1.538; 30: 0.632; 210: 0.483; all 35
(d, step) pairs pass at alpha = 2.5 and 3, twins' own measured 2.432 at 31->37 the one
near-budget corpus value. **BUT (r20, measured not argued) fixed differences exist with
single-step increments 3.231 q' (e = 344, 11->13), 3.947 q' (e = 1,532,627, 17->19,
verified 54 -> 129 by direct construction), 4.435 q' (e = 107,207,699, 19->23, verified
81 -> 183): NO uniform alpha <= 3 budget holds over the full family** - "closing (D) closes
every d" needs per-d constants or an explicit family-argmax exclusion, and the known argmax
jumps are non-decreasing (3.23, 3.95, 4.43).

WORD IDENTITY / FIRING / PADDING (r10-13). F(M+q') = max(F2(M), tiers) and the firing law
rest only on gcd(P_M, q') = 1 and contain no d - transfer verbatim, 13/13 configurations,
tier_1 = F2(M) exactly; degenerate q' | e collapses frame letters to 3q'. Firing for general
d: g = 0 mod q' => padded link, g = +-e mod q' => literal, else illegal; alternation forced;
F(M+q') = max legal-run span from the OLD machine alone (14/14 exact, the ONLY d-dependence
2u -> e). PADDING ECONOMICS: the frame conflict was UNITS - a padded link costs
q' slot = 6q' member for twins vs 2q' member for 3 | e; factor 3 cheaper absolute for
d = 0 mod 6, 1.5x scale-relative (that machine is twice as dense, mean gap 16.11 vs 32.21),
~10x availability at machine-31 scale; moves padding onset from the sixth step (twins,
31->37) to the FIRST (d = 12 at 11->13); supply cross-check 26,184 extrapolated vs 26,366
census (0.7%), links 2/37 of supply (~1,400). CORRIDOR LAW d-ANALOGUE: adjacent padded links
need openings r, r+c, r+2c all exposed - impossible for d = 2 in 34/74 probes (independently
reproducing lateral's proved 37->41 case), d = 4: 40/74, d = 6 and 12: 74/74, d = 30: 72/72;
for 3 | e it is a THEOREM (N2), unconditional - the padded step q' is not divisible by 3, so
r, r+q', r+2q' hit all three classes mod 3 and gear 3 blocks one. STRUCTURAL COMPENSATION:
padding is cheaper for d = 0 mod 6 but can never repeat consecutively there.

**5h. Literature adjacency (r20-21, still standing).** HOLT-RUDD (arXiv:1510.00743 via
ar5iv; 1408.6002; 1402.1970) HAVE the cycle-of-gaps recursion (our merge transform, one
residue class), an EXACT population dynamics with driving terms, and a transfer matrix with
p-INDEPENDENT Pascal eigenvectors giving closed-form asymptotic gap-population ratios,
Polignac-in-the-sieve (their Thm 5.5) and HL Conjecture B in cycles; they LACK any
maximal-gap tracking (explicitly out of scope - our merge law owns that readout), any
two-residue/paired object, any per-difference family. TRANSFER-MATRIX SIEVES: searched with
Holt excluded - NO other such literature, the frame is Holt's alone; one unreviewed Zenodo
preprint (Ojaroudi 2026, claimed unconditional twin prime theorem) assessed as claim class
far beyond method. COVERING TOOLKIT of our type: Filaseta-Ford-Konyagin-Pomerance-Yu (JAMS
2007), Hough (minimum modulus), Balister-Bollobas-Morris-Sahasrabudhe-Tiba distortion - all
one-class-per-modulus, NONE PAIRED. Holt's programme lives on arXiv and primegaps.info
rather than in journals, which affects where a note would go.

**5i. Per-difference sieve refinement (r22).** The sieve removes omega_p(d) = 2 classes for
p not dividing d and 1 for p | d, so the sifting DIMENSION is d-dependent:
`kappa_d = 2 - (1/log y) sum_{p | d, p <= y} log p / p` (Mertens), over all of [1,2], and
F_d(y) <<_eps y^{beta(kappa_d)+eps}. Both endpoints are attained inside the family:
kappa = 2 exactly for d coprime to the primorial (the hardest class per 5e), and
kappa = 1 + O(1/log y) for d = 0 mod the primorial - exactly the verified collapse j_2 = j -
so the interpolation is anchored at both ends; d divisible by exactly the primes in
(y^theta, y] gives kappa = 1 + theta (three thetas, three scales). HONEST CAVEAT: for FIXED
d and y -> infinity, kappa_d -> 2, so this is about differences that GROW with the machine -
the family setting.

## 6. RETRACTED / REFUTED (kept AS retracted; numbered)

*Premises and framing*

1. **"Ziller-Morack compute no h_2 values" - FALSE** (manager correction, 2026-08-23).
   Their companion note arXiv:1706.03668 (11 days after the theory paper 1706.00317, which
   we had read) computes h_2 exactly for all p_n <= 73; Table 1 contains our 18, 30, 66,
   150, 192 verbatim. CONSEQUENCES: our five values are an exact independent REPLICATION,
   not a first computation; their h_2(19) = 258 < 342 SETTLED our open y = 19 question
   (margin 24.6%, the r17 "~250" prediction right); the 3.8% dip at 13 remains the UNIQUE
   extreme through p_n = 73 in their full table, so "why is 13 extremal?" stood, with 12
   more data points. What remained ours: F_d(y), the fixed-twin ladder,
   maximiser/delta-profile structure.
2. PREDICTED Conjecture 6 breach at y = 17 (extrapolating the first four h_2 points):
   REFUTED same round - h_2(17) = 192, 29.4% margin; the margin is non-monotone with a
   one-off dip at 13.
3. FLAGGED gcd(e,105) = 15 and 105 as "exactly where a budget could fail": REFUTED by the
   r14 computation - those classes have the SMALLEST increments (0.632, 0.483 vs 1.235
   twins). Larger cap comes from a denser exposed set and denser machines have much smaller
   F (63, 49 vs 129 at y = 29); density wins.
4. "Tooth alternation FAILS for 3 | e" (r13-era): WRONG LAW TESTED - under the corrected
   merge law a same-tooth adjacency is a legal PADDED link, not a violation. The
   observation was real and carried the padding-cost finding.
5. "No padded gap at all for d = 2" vs mechanic's census of thousands: BOTH TRUE - measured
   below the padding onset (F < 3q') vs machine 31 above it. The r14 "exponential chasm"
   phrasing corrected to factor 3 absolute / 1.5 scale-relative / ~10x at machine 31; the
   census number 26,366 is padding SUPPLY, not links (~1,400).
6. Word-list check first pass: 73/73 "mismatches" from comparing letter VALUES where the
   claim is about RESIDUES - own bug, corrected to zero.
7. Wrap-around artefacts (twice): the r10 letter extractor and the r11 np.roll kill-status
   corruption; both fixed by absolute positions over two periods, counts to zero.
8. C1 band statements: STRUCK after twelve rounds moved nothing.
9. Mirror-canonical o5 pruning: UNSOUND with left-tautness (maps left-taut to right-taut
   coverings); removed, nothing lost.
10. "The y = 19 exhaustive scan is out of reach (2,424,922 differences)" (r17): superseded
    by the r22 delta reduction + prefilter - minutes, keeping 64 of 1,616,615 deltas.

*Novelty downgrades (all self-found)*

11. The DELTA REDUCTION and PREFILTER METHOD are NOT a contribution: the reduction is
    essentially ZM Proposition 1.5(2), their algorithms reach p_n = 73 where our scan
    reaches 23, and the winner data is in their ancillary files. What IS new: the
    independent replication, the exhaustive settlement of the 3 | e branch, and the
    CROSS-GEAR EXTENSION LADDER - a question ZM never ask.
12. The LOCAL-FACTOR IDENTITY, the DEPTH-SUM IDENTITY and "doubled level spacing" are Holt
    arXiv:2502.20470 (5f). Proofs and values unaffected; novelty labels are not.

*Sieve-theory corrections*

13. R21's "a paired Iwaniec bound is PARITY-CRITICAL; the Iwaniec-analogue is the open
    wall": WRONG SLOT, self-caught in r22 - Iwaniec's ordinary bound is at dimension ONE
    and r21's own Theorem 2 already delivers the dimension-TWO counterpart. Replaced by the
    ceiling of 2e.
14. **THE ROUND-22 "2 kappa IMPOSSIBILITY" PARAGRAPH IS RETRACTED (r23).** "NO SIEVE
    ATTAINS 2 kappa FOR ANY kappa > 1" was written as an impossibility theorem. IT IS AN
    OPEN PROBLEM (Brady, Stanford thesis 2017: "it is currently not known whether there is
    any kappa > 1 with beta_kappa < 2 kappa"), and FALSE as a blanket statement -
    Rosser-Iwaniec beats 2 kappa for 1/2 < kappa < 1. The PROVED floor is
    beta_kappa >= (1+o(1)) 2 kappa/e ~ 1.47 at kappa = 2, so ZM's exponent 2 is NOT proved
    below the sifting limit, only below the CONJECTURED one; Brady even conjectures 2 kappa
    is beatable. What survives is the PARITY form (2e). Same paragraph: r22's "measured
    constant in [3.47, 4.16]" does not contain the limit (2 lambda_* = 7.182242); "M.
    Franze" is C. S. Franze; Selberg's conjecture is not in Franze; the 19/36 vs 0.4454
    conflict (settled 4c); Iwaniec's statement is h(k) << (k log k)^2, the "(log n)^2"
    phrasing a weaker corollary; Costello-Watts is arXiv:1306.1064.

*The five referee defects (r23, all in our own documents)*

15. **THE y = 3 ROW WAS WRONG, AND THE CORRECTION IS SHARPER THAN THE ERROR.**
    paired-jacobsthal-values.md tabulated "y = 3, h_2 = 0, Conj. 6 holds". That 0 is a code
    artefact: research/jacobsthal_family.py returns 0 whenever a period carries fewer than
    two survivors, and at gears {3}, e = 1 the survivor set mod 3 is the single class {1},
    whose CYCLIC gap is 3. The truth is **h_2 = 6 = p^2 - p exactly**, confirmed by
    A288815. So Conjecture 6 FAILS BY EQUALITY at n = 2 (and at n = 1: h_2(2) = 2 = 2^2-2),
    which means **ZM's "n >= 3" hypothesis is SHARP rather than conservative** - worth a
    sentence in the paper, and a fact the project had inverted.
16. THE MAXIMISER LISTS WERE TRUNCATED ARGMAX SLICES presented as complete. True counts 8,
    16, 64 at y = 11, 13, 17; the doc printed the first 5 and first 6. Complete lists now
    in the doc.
17. "Worst ratio 0.858 at n = 3" (Theorem 1's chain) omits the "+1" that is part of the
    bound; with it, 0.8627.
18. "V_n >= 0.3908/(log p_n)^2 for p_n >= 285" DOES NOT FOLLOW from the stated ingredients:
    2 e^{-2 gamma} C_2 (1 - 1/log^2 285)^2 = 0.390569 < 0.3908. The safe constant is
    **0.3905**; Theorem 1's conclusion is unaffected. (The INEQUALITY is true where checked
    - exact V_n log^2 p_n >= 0.4048 over 285 <= p_n <= 2731 - only its derivation was one
    digit short. Recorded both ways.)
19. The quasi-polynomial constant - item 14 / Theorem 3E.

*Structural extrapolations refuted by later computation*

20. THE ROUND-20 CAP GUESS (best extension = g_L + F + g_R only) was WRONG; the failed
    assertion found the truth - one-sided two-gap chains beat both-flank fusion from 19 on.
21. **THE DEFICIT DOUBLING (9, 18, 36) IS REFUTED** - by arithmetic, before any computation:
    a deficit can never exceed the increment F(new) - F(old) because the best extension is
    at least F(old), and A288815's increments 21, 33, 54, 42, 60, 69 show the 23->29
    increment COLLAPSING to 42 < 72. The doubling was a coincidence of three consecutive
    increments; what survives is the accounting identity (5d).
22. THE PRE-REGISTERED DEFICIT PREDICTION 21 AT 23->29 WAS WRONG: measured ZERO (certified,
    5d). The 2-gap sums do not continue 12, 15, 18, 21 - they run 12, 15, 18, 42.
23. R21's "from 17 on the argmax trajectory is forced to abandon its ancestors; a record
    window is self-limiting; each new gear's winner is a fresh deep resonance": REFUTED at
    23->29 by explicit certificate. Maximiser persistence is NOT monotone in y - it fails at
    17, 19, 23 and returns at 29. The mechanism (cap law) was right; the extrapolation from
    three points was not.

*Round-23 conclusions overturned inside round 23*

24. "RUNG 2 CANNOT BE MADE EXPLICIT" (first pass) - WRONG; corrected in the same round after
    a collaborator's lead was verified against actual text. The mistake was looking for a
    FUNDAMENTAL LEMMA; the tool is an explicit Selberg sieve (2c).
25. "A VALID NESTED TRUNCATION IS THE MISSING CONSTRUCT" - WRONG; validity is settled (2f).
    What is really invalid is the PER-BAND product truncation (36 explicit counterexamples).
26. PRE-REGISTERED GUESS that monotone depths h_j are needed for validity - REFUTED in the
    same script written to assume it: 0 violations over all 271 non-monotone patterns.
    Monotonicity is a LEVEL-COST convenience.
27. "IWANIEC-KOWALSKI THEOREM 6.9" - a citation chimera that had reached two documents (4d).

*Round-23 model claims retracted in round 24*

28. **"THE TRUTH IS h_2 ~ (p^2 - p)/2" IS RETRACTED.** Unsupported by its own data: c z^2
    and c z (log z)^2 fit equally on ZM's 21 values, the residuals drift in OPPOSITE
    directions, and the discriminating paired-minus-ordinary exponent gap (0.33-0.75) is the
    signature of a LOGARITHMIC separation, not the +1.0 a quadratic-vs-linear law needs.
    Supported reading: h_2 = z^{1+o(1)}, best model ~2.56 z (log z)^2 (3c).
29. **THE COVERING-CAPACITY ARGUMENT IS RETRACTED.** R23 argued: "the covering CAPACITY
    sum_{p<=z} omega(p)/p is 1.34/1.46/1.76 (ordinary) against 2.19/2.41/3.01 (paired) at
    z = 13/19/73, so the ordinary covering is COUNTING-CONSTRAINED where exact values exist
    and the paired one is not, hence quadratic". WRONG - **capacity is not scale-free**: the
    ordinary covering reaches the same capacities at z ~ 4e3..6e6 with its answer still
    z^{1+o(1)}. Consequently the r23 NAMED OPEN PROBLEM "h_2 >> p^{1+delta}" asked for
    something PROBABLY FALSE; replaced by (P2)/(P3) in 3d. (The one-line CRT restatement in
    3a survives; only the capacity inference does not.)
30. TWO ROUND-24 OVERSTATEMENTS CAUGHT BY THEIR OWN ASSERTION GATES IN-ROUND, recorded as
    the quality signal they are: (i) a draft claimed h_2/(z^2-z) decreases MONOTONICALLY
    from z = 13 - false (8 of 15 steps down; it DRIFTS); (ii) a draft claimed h_2/j "tracks
    W/V within 1.3x" - false (1.33x..2.51x, drifting up; in exponent terms
    h_2/j = (W/V)^t, t = 1.22..1.51). Both scripts now state the honest numbers and assert
    them.

## 7. PUBLICATION STATE AND NAMED OPENINGS

**7a. The units. UNIT 1 - PUBLICATION-READY.** "The paired Jacobsthal function: first upper
bounds, and the structure of its maximisers" = j2-upper-bound.md (Theorems 1, 3/3E,
2E/2E'/2E'', the beta_2 rung, the per-difference corollary, THE CEILING) +
j2-lower-ladder.md (P1) + twin-percentile.md + paired-jacobsthal-values.md 4a/4b/4c. First
bounds of any strength on a function named and conjectured about since 2017, aligned
rung-for-rung with the ordinary ladder. HEADLINE: "first proved upper bounds on j_2 - an
explicit quasi-polynomial rung, an explicit polynomial rung at exponent 15, and the
best-exponent rung 4.266 by citation, with an honest statement of which constants exist -
together with the first lower bound using the paired structure, (1.349+o(1)) z log z." All
three r23 submission blockers DISCHARGED (ODC confirmed with an OCR caveat; HR Memoire
obtained and 7.972 re-derived; 19/36 settled with a stated residual). STILL TO DO: the
sandwich paragraph must be REWRITTEN - no longer "around a truth of p^2/2" but "around a
truth the data cannot separate from ~2.6 z (log z)^2, with the quadratic reading measurably
losing ground" - which also gains the paper a falsifiable prediction (h_2 at z = 151+), and
referees like that; plus a careful statement of the sieve dimension and remainder bound,
and honest positioning of the computational half as replication-plus-structure given ZM's
ancillary files.

**UNIT 2 - A SHORT NOTE, not a paper** (downgraded r22 by 6.12, partly restored by the
Bonferroni generalisation) = paired-hlb-cycles.md after its section-0 correction: one object
(n_g), one theorem about it (the exact Bonferroni series with the moment identity, of which
the pinch is orders 0-1), one effective corollary (y_0(g) = exp(Theta(sqrt g))). A
legitimate short note extending Holt's programme to the twin-candidate subsequence; it would
cite him on nearly every page. TO ADD: uniform error terms; a decision on whether the
effective threshold's CONSTANT can be improved.

**UNIT 3 - separate venue, self-contained:** the Lean development - machine-checked
per-difference equivalences for Polignac, the Goldbach window reduction with its exact
converse, the mod-3 dichotomy, the universal cap <= 12. Formalization venues take work
containing no new mathematics; needs packaging, not research.

**NOT PUBLISHABLE ALONE, and priced that way:** (i) twin-percentile - data, no theorem,
belongs inside Unit 1; (ii) the h_2 replication AND the delta-reduction / prefilter method -
struck entirely (6.11); (iii) the cap law and the deficit ladder - a good SECTION of Unit 1
(strengthened at 23->29 where it predicts the admissible lifts exactly, but holding only
under observed non-collision conditions, and the deficit ladder is four points one of which
falsified the extrapolation drawn from the other three). **NOT OURS TO PUBLISH:** everything
on the twin route (other lanes), and the kernel work on our identities.

**7b. WHAT THE PAPER DOES NOT CLAIM** (numbered section 4a of j2-upper-bound.md, written for
the referee), six items: (1) no progress on Conjecture 6; (2) no new sieve theory - the
contribution is that the ladder was EMPTY, not that the rungs are hard; (3) the beta_2 rung
is NOT fully explicit and the best bound with all constants stated is the exponent-15
polynomial (and quasi-polynomial 3E); (4) the computational half is replication plus
structure given ZM's ancillary files; (5) no lower bound beyond (P1) and the collapse
transfer; (6) nothing about primes - every statement is about coverings of an interval.

**7c. Named openings (in order).**
1. **BLIGHT'S THESIS** ("Refinements of Selberg's sieve", Rutgers 2010) may hold an EXPLICIT
   Lambda^2 Lambda^- variant - NOT OBTAINED. Cheapest route below exponent 15.
2. **ODC CHAPTER 6 beta_2 EXPLICITNESS.** ODC Ch. 6 prints beta_1 = 3.8629,
   beta_2 = 7.5941 around pp. 71-73. IF that apparatus is constant-free at kappa = 2 the
   explicit exponent drops from 15 to ~8.6; its explicitness is exactly the open question
   (the Ch. 11 beta-sieve is known non-explicit). SAME OBJECT as the HR-Memoire item: the
   remaining mathematics of the upper ladder is ONE thing seen from two sides - an explicit
   dimension-2 lower-bound sieve near exponent 8. Named, priced, next-round sized.
3. **RANKIN LAYERING** for (P2): h_2 >> z (log z)^2/(loglog z)^{O(1)}. Genuinely new work;
   Kalmynin-Konyagin shows the machinery lands on a 2-class sieve (M(f) = 2 buys two
   Rankin-type logs). Still a construction, still parity-free.
4. **h_2 at p_n = 151..251** - the decisive computation for the growth law, against ZM's
   stopping point of 73. A compute decision, not this lane's alone: the delta/prefilter
   method reached z = 23 exhaustively; ZM's published algorithm reached 73 in 2017.
5. The residual page scan of Selberg's printed (14.40) - closes 19/36 forever.
6. **(D), the flank bound** - the single open lemma of the tolerance route, same for every
   even d ("closing D closes every d", NOT "every d is closed"), now known to need per-d
   constants or an explicit family-argmax exclusion (5g). The twin route is not closed.
7. Smaller, still live: WHY clean extension dies at 17 as a profile-collision mechanism;
   F(2,53) termination (>= 426, needs <= 486, prediction ~441); budget arithmetic beyond
   step 23->29 for every d; gcd classes 7, 21, 35 (d = 14, 42, 70) untested; the d = 0 mod 6
   word grammar (3 letters, one short) needs its own word list before the tolerance route
   can be quoted there; n > 3 assembly (mechanical, deferred); the k-tuple F_d family
   (computable, nobody has); the C4 bite (singular-series factor over q | N as an exposed-set
   size, kernel-reachable).

**7d. THE STANDING CITATION-HYGIENE LESSON** (three clauses; cost novelty or correctness in
three consecutive rounds).
1. **PRIOR-ART CHECKS EXPIRE.** Both r22 downgrades came from documents that existed but had
   not been looked at (ZM's ancillary files, 2017) or did not exist at the last sweep (Holt,
   Feb 2025). Any claim of novelty older than a round should be RE-SEARCHED before it is
   repeated in a summary - and by CITATION GRAPH, not keywords (4f).
2. **SECOND-HAND CITATIONS EXPIRE FASTER.** Five of the sieve-theory facts in our own
   strongest paragraph were wrong or misattributed, and every one came from a summary rather
   than a source. Read the full text. The referee pass now carries a CITATION-NUMBERING
   SWEEP as a standing step (4d).
3. **"NOT AVAILABLE IN THE LITERATURE" EXPIRES TOO, AND FASTEST OF ALL.** We concluded
   mid-r23 that no explicit dimension-2 lower-bound sieve existed. True of every FUNDAMENTAL
   LEMMA checked and false of the problem: the tool was an explicit Selberg sieve, citable
   in 2026 because two papers on the almost-prime GOLDBACH problem needed it for exactly our
   density function. **When a search for a TOOL fails, search the neighbouring PROBLEM, not
   more variants of the tool's name.**

COROLLARY ADDED R24: **MODEL CLAIMS EXPIRE LIKE CITATIONS.** "The truth is p^2/2" was
repeated round to round as if measured; it was one of two fits and the project had inverted
which was supported. Any "truth ~ f(z)" statement in a summary should carry the competing
fit and the discriminating measurement, or not appear.

## 8. POINTERS (reproduction)

**Kernel.** proofs/Polignac.lean (all of 5a; registered in proofs/lakefile.toml). Composes
with BlockedSlots, Horizon, Layer, Supply, Census, Bridge, Gear.

**Standing artefact.** `research/j2_referee.py` - RE-RUN BEFORE ANY FUTURE CLAIM ABOUT UNIT
1; recomputes every recomputable numerical claim by independent code, caches
research/data/ref_fam_{3,5,7,11,13,17}.npy (seconds after the first run).

**j_2 ladder (research/).** j2_bound.py (Thm 1), j2_brun.py (Thm 3), j2_perdiff.py
(kappa_d), j2_explicit.py (Thm 3E + rung-2 level/error costing), j2_fi77.py (Thm 2E via ODC
7.7, the self-audit, the F5 threshold table), j2_nested.py, j2_lower.py (r23 sandwich),
j2_lower2.py (P1 + growth-law reread), j2_presieve.py (2E'/2E'', the exponent-15 floor, the
HR-Memoire re-derivation at P4), j2_selberg.py (19/36 in exact rationals). Data:
research/data/{j2_bound,j2_brun,j2_perdiff,j2_explicit,j2_fi77,j2_nested,j2_referee,
j2_lower,j2_lower2,j2_presieve,j2_selberg}.out.

**Family / structure (research/).** jacobsthal_family.py + jacobsthal_h2_17.py (h_2 values,
percentiles - NOTE the empty-period 0 artefact of 6.15), jacobsthal_mod3.py, why13.py,
maximiser_shape.py, h2_19_lift.py, zm_margin_mechanism.py, family17_percentile.py,
delta_frame.py, family_scan{,_fast,23}.py (delta reduction + prefilter), ext_death.py,
ext_death2.py, ext_deficit19.py, ext_deficit23.py, ext23_witness.py (the 23->29 certificate,
independent code path, no sieve array), zm_seq_reconcile.py (ZM nseq cross-check). Arrays:
f13_family.npy, f17_family.npy, family_w19_delta.npy, family_w23_delta.npy, ext19_to23.npy,
ext23_all.npy.

**Paired-Holt / HL-B (research/).** paired_holt_recursion.py, paired_hlb.py,
hlb_effective.py, pinch_bonferroni.py, holt_correspondence.py.

**Twin-route support (research/).** polignac_transfer_check.py, twin_pin_check.py,
same_census_check.py, pairsplit_check.py, corr_triple_check.py, assembly_check.py +
master3_check.py, lefttaut_check.py, literal_cap_gap_d.py + literal_cap_mod105.py,
word_identity_gap_d.py, firing_padding_gap_d.py, frame_reconcile.py + pad_count_bound.py,
route_transfer_audit.py, budget_per_d.py, split_gap_law.py, general_gap.py,
topgap_endpoint_law.py.

**Search.** rust2/src/bin/maxgap_pruned.rs; log research/data/maxgap53_pruned.log.

**Docs (docs/novel/).** j2-upper-bound.md (sections 1, 4, 6a, 8, 9 = the verification
record), j2-lower-ladder.md, paired-jacobsthal-values.md (4a why-13, 4b cap law, 4c deficit
ladder), twin-percentile.md, paired-hlb-cycles.md (sec. 0 + 6 = the Holt correction; 3a/3b
Bonferroni), paired-holt-recursion.md (CORRECTION block), README.md index.

**Lean environment notes.** omega does not combine congruences across moduli - decompose to
one modulus; import Mathlib.Data.Nat.ModEq for [MOD n]; Nat.dvd_sub here is the old
Nat.dvd_sub'; Finset.card_insert_of_notMem rename; Nat.Ico_succ_right_eq_insert_Ico lives in
namespace Nat; beware rwa rewriting the ModEq modulus occurrence; count primitive pattern:
induction + Nat.succ_div_of_dvd/not_dvd avoids division-by-variable omega limits.

**Finite kernel candidates offered to Formalist, never withdrawn.** Theorem 3E's finite half
at a fixed n; the INVALIDITY of the per-band product truncation (36 explicit witnesses); the
VALIDITY of the nested upper-tail truncation at a fixed depth pattern (a finite alternating
binomial sum; j2_nested.py enumerates the instances); Theorem (P1)'s certificate at one fixed
z; the exact-rational 19/36 derivation (S1 of j2_selberg.py); the delta reduction at a fixed
machine; the Bonferroni step of Theorem 3 at fixed n and K; one rung of the word-level paired
transfer; the Pascal eigenvector identity at fixed size; the local-factor identity at fixed
q; endpoint c-law cases.
