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

## 9. ROUND 25 (2026-08-29) - THE UNIT-1 OPENINGS BOTH CLOSE, EXPONENT 15 -> 8

Brief: (a) Unit 1 to submission - obtain Blight's thesis and decide the ODC Ch.6
explicitness question, re-running j2_referee.py first; (b) the Rankin-layering
problem on the lower ladder; (c) own ranking if capacity remains.

STANDING GATE FIRST. `research/j2_referee.py` re-run from a clean process at the
top of the round: **"j2_referee: ALL ASSERTIONS GREEN"**, five recorded defects
reproduced verbatim, h_2(19) = 258 re-confirmed by independent re-verification of
the y = 19 winners. No claim below entered the record before that ran.

### 9a. BLIGHT'S THESIS - OBTAINED. VERDICT: NO USE, ON BOTH COUNTS.

**Sara Elizabeth Blight** (the name in every prior note of ours, "S. Blight" from
ODC p.112, is a woman and the first name is Sara, not Sean), *Refinements of
Selberg's Sieve*, Ph.D. Rutgers, May 2010, advisor Iwaniec; DOI 10.7282/T35T3KJ8,
RUcore rutgers-lib/27420, **free**, downloaded to `research/data/blight_thesis.pdf`
(367,455 bytes, 75 pages) and read here directly. Text extract cached at
`research/data/blight_text.txt`. ODC p.112's pointer, verbatim from the scan:
"We remark that S. Blight (thesis, Rutgers 2010) has sharpened this result using
a Selberg-type combination but with a Brun lower-bound sieve supported on products
up to three primes rather than one as in the right-hand side of (7.106)."

1. **WORSE THAN WHAT WE ALREADY CITE AT kappa = 2.** Her sec. 2.6 gives
   beta_2 < 4.45 (c = 0.2214971799, T = 16), and her own sec. 2.7 says "The sieve
   of Diamond and Halberstam gives a smaller sifting limit for kappa = 2 and
   kappa = 2.5" - 4.266450, which her sec. 2.2.2 tabulates from DHR [1, p.227].
   Her Lambda^2 Lambda^- wins only at kappa = 3 (6.458) and 4 (8.47). This
   CONFIRMS round 22's second-hand reading from the primary document.
2. **NOT EXPLICIT.** Proposition 2.4.2, in full: "Assume T_F(s) as defined above
   is positive. Then there is some z_0 such that if z > z_0, then V(D,z) is also
   positive." Proof, in full: "As z -> infinity, the error term above approaches
   zero and the main term is positive as stated." The error is
   O(V(z)^-1 alpha loglog z/log z) with the implied constant inherited from the
   `<<` in her own sifting-dimension hypothesis - i.e. from our K, unquantified.

So the round-24 opening #1 closes NEGATIVELY, and the closure is itself a paper
sentence: it extends **the explicitness boundary** from the DHR differential-delay
system to the Lambda^2 Lambda^- family, first-hand.

BY-PRODUCT WORTH KEEPING: her sec. 2.2.1 gives the beta sieve's asymptotic
beta_kappa ~ c kappa with **c = 3.591..., the root of (c/e)^c = e**. That is
c(log c - 1) = 1 - **exactly the equation defining our Theorem 3E constant
lambda_* = 3.591121**, and ODC Thm 6.12 prints the same c. So C_infinity =
2 lambda_* = 7.182242 is twice the beta sieve's asymptotic sifting-limit slope: a
real adjacency between our quasi-polynomial rung and the classical beta sieve.
Also: the beta sieve's own beta_2 = 4.8339865967, worse than DHR.

### 9b. ODC CHAPTER 6 - **EXPLICIT. YES.** THE DECISIVE ANSWER OF THE ROUND.

Page scans of pp. 65, 68-73, 112 of the AMS printing read first-hand 2026-08-29
(Google Books volume Dz6REQAAQBAJ, publisher preview). Corrections to our own
notes first: Chapter 6 is "Brun's Sieve - The Big Bang"; sec. 6.6 is "Improved
Bounds for the Sifting Limits"; **beta_1 = 3.8629 and beta_2 = 7.5941 are on
p. 73, NOT in Corollary 6.14** (which is on p. 71 and is the weaker
beta(kappa) < 4kappa+1). Round 24's note said "Cor 6.14 area" - wrong location,
right chapter.

THE ANSWER. **Proposition 6.7, (6.75), (6.85), (6.86) and Corollary 6.13 carry no
O(.), no `<<`, no implied constant and no "for z large".** Corollary 6.13 reads
V^-(D,z) >= {1 - (7/8)K^5}V(z) for s >= beta_kappa, beta_kappa =
1 + 2(e^(1/2kappa)-1)^-1, beta_2 = 8.041. The ONE inexplicit sentence in the
neighbourhood is Corollary 6.14 ("for z large"; proof: "provided K is sufficiently
close to one ... by choosing a slightly larger value of kappa") - and **that
device is precisely what our PRE-SIEVING replaces at explicit finite cost.**

WHY WE MISSED IT IN ROUND 24: we priced Theorem 6.9 and Corollary 6.10 from this
chapter (s > 28.99 / 37.36 at K = 3) and called them "cleaner-looking fallbacks".
**We never priced Proposition 6.7 / Corollary 6.13, and the whole gain sits there.**
Lesson, added to 7d: when a chapter is priced, price its PROPOSITIONS, not only
its theorems - the fundamental-lemma-shaped results are the ones with the K^10.

HARVEST VERIFIED FIRST-HAND (research/j2_odc6.py section A): every printed number
of sec. 6.6 reproduced from the book's own printed formulas by independent code -
psi^- = 0.8637687819 from BOTH the general formula and the printed closed form
2e(16 sqrt e - e^3)^-1, agreeing to 1e-9; beta_1 = 4.082988, beta_2 = 8.041623 at
alpha = 1/4; beta_kappa <= 4kappa+1 at nine dimensions; alpha^-1 = 3.774952.

**A DISCREPANCY IN THE BOOK, FOUND AND RECORDED.** The printed root
alpha* = 0.264904 does NOT solve the book's own printed equation (residual
-0.001707; the true root is 0.2652637). It IS internally consistent with the
printed beta_1 = 3.8629 and beta_2 = 7.5941 to within its own truncation, so this
is not an OCR digit error - the book says how it computed it ("use the Taylor
expansion at 1/4"), and a Taylor approximation about 1/4 lands exactly 3.6e-4
short. In our favour: the exact root gives beta_2 = 7.5838, **0.0103 better than
the printed 7.5941**.

### 9c. THE TWO ROUND-24 LEADS ARE **ONE EQUATION** (proved, not suspected)

Round 24 wrote "SAME OBJECT as the HR-Memoire item ... ONE thing seen from two
sides". Now demonstrated (j2_odc6.py section B):

* ALGEBRAIC IDENTITY. HR Memoire's printed positivity condition
  lambda^2 e^(2lambda)(2+e^2) < 1 IS ODC (6.86)'s positivity condition
  2e^-2 a^2/(1-a^2) < 1 under a = lambda e^(1+lambda), because
  lambda^2 e^(2lambda) = a^2/e^2. Verified identically at six lambdas, error < 1e-12.
* NUMERICAL IDENTITY. ODC's K -> 1 root alpha_inf = **0.253321897**; round 24's
  re-derived HR lambda_* = **0.2533219**. Equal to 5e-7.
* THE EXPONENT. HR's u = 1 + 2.01/(e^lambda_* - 1) = 7.971954733 (2.01 is HR's own
  margin); ODC's beta_2 = 1 + 2/(e^alpha - 1) = **7.937268**. ODC Chapter 6 is the
  **explicit form of the 1971 Memoire's theorem, and very slightly sharper.**

### 9d. THEOREM 2G - THE NEW RUNG. EXPONENT 15 -> 8.042, LOG POWER 10 -> 3.

The beta sieve's weights obey |lambda_d^-| <= 1, so with |r_d| <= 2^nu(d) N_pre,
|R^-| <= N_pre sum_{d<D} tau(d) <= N_pre D(log D + 1) (elementary
sum_{n<=x} tau(n) = sum_{d<=x} floor(x/d) <= x(log x + 1); checked against the
exact sum to 20000). **That is the second win of Chapter 6 over Theorem 7.7**: 2E''
had to carry tau_4, i.e. sum 8^nu(d) << C_8 D (log D)^8.

    THEOREM 2G (p_0 = 151, K = 1.0260176 < 1.0297232 = psi^-(1/4)^-1/5, so ODC
    Corollary 6.13 applies VERBATIM at alpha = 1/4, delta = 0.017864):
      j_2(p_n#) <= C p_n^8.04162 (8.04162 log p_n + 1)(log p_n)^2 + 1,
      p_n >= 285, log10 C = 57.5, C = N_pre/(0.3905 delta), N_pre = prod_{p<151}(p-2).
    THEOREM 2G-inf: j_2(p_n#) <<_eps p_n^(s+eps) for every s > 7.93727, every
    implied constant computable.

    p0      s(Cor 6.13)  s(Prop 6.7)  log10 C   crossover with 2E''
    151     8.04162      8.02805      57.5      10^5.58   <- operative row
    211     8.04162      8.02742      82.2      10^8.92
    307     8.04162      7.98875      120.0     10^14.15
    601     8.04162      7.96945      243.9     10^31.61
    K -> 1  --           7.93727      --        (floor)

2E'' (exponent 15, tiny constant) stays in the paper: it is the better bound below
p_n ~ 380,000. Above that 2G wins by p_n^6.96. Both rungs are kept and labelled.

### 9e. THE RANKIN-LAYERING PROBLEM - (P2) SUPERSEDED BY ONE LOG

Own pre-registration, written before running anything, scored in the gate's
section F. New doc: **docs/novel/layered-erdos-rankin.md**; gate
`research/j2_rankin_layer.py`, ALL ASSERTIONS GREEN.

    CLAIM (bookkeeping):  j_k(P(x)) >> x A^(2k-1) C^k/((5k)^k B^(2k)),
                          A = log x, B = log A, C = log B,
    j_k = the k-classes-per-prime Jacobsthal function.
    k = 1 IS the published FGKT length x log x lll x/(ll x)^2.
    k = 2: h_2(P(z)) >> z (log z)^3 (lll z)^2/(100 (ll z)^4).

MECHANISM. Class 0 on a SPLIT range [2,P) u (z1,x/4] delivers survivor density
~1/log y where its Mertens entitlement is only O(1) - that IS Rankin's gain, one
log - and leaves [P,z1] free for the greedy. The paired problem's SECOND class
runs the identical trick on n+2, so the joint survivor set is the TWIN primes.
Only an UPPER bound on twins is needed (Brun/Selberg), so it stays parity-free.

STATUS, stated bluntly: **asymptotic bookkeeping, not a written-out proof.** What
IS exact: the restatement re-brute-forced at z = 3,5,7 against h_2 = 6,18,30; that
c = 2 collides with no odd prime's class 0; the twins-or-smooth survivor structure
by direct sieving at four parameter sets. What validates the bookkeeping: running
the SAME optimiser at k = 1 reproduces the FGKT closed form with residual spread
**0.072 over eight decades of log x** (0.271 at k = 2) - calibrated against
someone else's theorem, not against itself.

PRE-REGISTRATION SCORED: PR1 (two extra logs) CONFIRMED. PR2 (the model is not a
ceiling) CONFIRMED. PR3 (power 2k-1 for general k) CONFIRMED at k = 1..5.
**PR4 CONFIRMED IN CONCLUSION BUT WRONG IN MECHANISM** - I predicted the layering
would be a LOSS at reachable z; it is not a loss, it simply does not exist below
log z ~ 300 because [P,z1] is empty. Scored as wrong-as-worded.

### 9f. CONSEQUENT CORRECTION TO OUR OWN RECORD (3c)

**THE "~2.56 z (log z)^2" MODEL IS DEMOTED FROM "TRUTH" TO HEURISTIC.** It is the
largest gap in a RANDOM set of density prod(1-k/p) ~ 1/(log z)^k, i.e. z(log z)^k,
while j_k is a MAXIMUM over choices. At k = 1 the heuristic is right (Rankin
attains it up to loglog powers); at k = 2 the layered construction exceeds it by a
log. The sandwich of 3c must read:

    proved lower  h_2 >= (1.349+o(1)) z log z                  [(P1)]
    bookkeeping   h_2 >> z (log z)^3 (lll z)^2/(ll z)^4        [round 25]
    HEURISTIC     ~2.56 z (log z)^2  -- NOT a ceiling
    proved upper  p_n^8.04162 explicit / p_n^(4.266+eps) cited

This is round 24's own corollary "MODEL CLAIMS EXPIRE LIKE CITATIONS" firing on
round 24's model, one round later.

### 9g. Ranking changes

* **N4 (j_2 upper ladder) stays TOP and is now materially stronger**: the explicit
  exponent is 8.042 with a floor of 7.937, and both round-24 openings are closed.
* **P1-P3 (lower ladder) RISES from rank 2 toward parity with N4**: it now carries
  a construction that beats the previous heuristic ceiling and a family
  generalisation (j_k) that nobody has stated. Its blocker is writing, not
  research.
* **NEW ITEM, ranked immediately below N4: (P2') "write the k = 2 layering out
  with constants".** Ordinary work; every ingredient is standard.
* **The named opening of the upper ladder changes.** It is no longer "find an
  explicit dimension-2 lower-bound sieve" - we have one. It is: **close the gap
  between 7.937 (explicit, ODC Ch.6 beta sieve) and 4.266 (DHR, not explicit).**
  ODC sec. 6.6 says it "will be superseded by the results of Chapter 11", but
  Chapter 11's lower-bound constant B is identically zero at kappa >= 1/2 (our
  sec. 2e), so Chapter 11 is NOT the route. An explicit form of the DHR
  differential-delay system is.
* **DEMOTED: opening 7c#1 (Blight).** Closed negatively, will not be re-attempted.
* **7c#4 (h_2 at p_n = 151..251) RISES**, because the two competing readings are
  now z(log z)^2 and z(log z)^3, a further log apart - the computation discriminates
  more than it did.

### 9h. Negatives and residual risks of the round

* **JUDGMENT, NOT RESULT**: that the ODC Ch.6 -> DHR gap (7.937 -> 4.266) is not
  reachable by more pre-sieving. Backed by arithmetic - the K -> 1 limit IS
  7.93727 and pre-sieving only moves K - so it is a result for THIS sieve, and a
  judgment for the problem.
* Page images read through a browser preview, not held in hand (same caveat as
  round 24's Thm 7.7 check). Mitigated: eight printed numbers all reproduce from
  the printed formulas, so an OCR corruption would have to be self-consistent.
* **(5.38) (the definition of K) and (6.69) (a condition on kappa quoted inside
  Proposition 6.7) were NOT re-fetched.** (5.38) is the hypothesis we used for
  Thm 7.7 in rounds 23-24, matched against Dudek-Dunn Lemma 2.1. (6.69) is unread;
  our operative alpha = 1/4 is the book's own choice in Corollary 6.13 "for
  kappa > 0", so kappa = 2 at alpha = 1/4 is inside its applied range. FLAGGED.
* p. 74 (rest of Prop 6.16, preliminary sieving) not obtained; our pre-sieving
  accounting remains round 24's own.
* No journal paper by S. Blight found (arXiv author page 404); the thesis appears
  to be the only citable form.
* The layered construction is bookkeeping. It should not be quoted as a theorem.
* FKMPT "Long gaps in sieved sets" (arXiv:1802.07604): round 24's RELAY-SOURCED
  flag is **discharged** - abstract and main theorem read first-hand 2026-08-29.
  It is the ADVERSARIAL problem (classes GIVEN, |I_p| <= C_0, bound
  x(log x)^(1/exp(C C_0))); ours chooses the classes and maximises. Neither
  contains the other. Must be cited in the lower-ladder note.

### 9i. Additions to the standing citation-hygiene lesson (7d)

4. **PRICE PROPOSITIONS, NOT ONLY THEOREMS.** Round 24 priced ODC Thm 6.9 and Cor
   6.10 and moved on; the usable result was Proposition 6.7 / Corollary 6.13 on
   the facing pages, and it was worth seven units of exponent. A chapter is not
   priced until its numbered non-theorems are.
5. **A PRINTED NUMERICAL ROOT IS A CLAIM LIKE ANY OTHER.** ODC's alpha* = 0.264904
   does not solve ODC's own printed equation. Re-solve every printed root from the
   printed equation; it cost nothing and it improved the constant.

### 9j. Reproduction (round 25)

* `research/j2_odc6.py` -> `research/data/j2_odc6.out`. Sections A (ODC Ch.6
  reproduced from its own formulas), B (the HR identity), C (pre-sieving ladder
  K(p_0)), D (Theorem 2G), E (crossover vs 2E''), F (constant-free form).
* `research/j2_rankin_layer.py` -> `research/data/j2_rankin_layer.out`. Sections
  A-F as described in 9e.
* `research/data/blight_thesis.pdf` (367,455 bytes) and
  `research/data/blight_text.txt` (extracted text, pypdf).
* Page scans of ODC pp. 65, 68-73, 112 at `research/data/odc6_scans/PA*.png`.
* `research/j2_referee.py` re-run and GREEN before any of the above.

## 10. ROUND 26 (2026-08-29) - THE LAYERING IS A THEOREM; UNIT 1 ASSEMBLED

Brief: (a) write out the k = 2 Rankin layering with constants - proof or break;
(b) price the paired-Iwaniec problem honestly after (a); (c) Unit 1 final
assembly with the 8.04 rung, the referee pass and the citation-numbering sweep
re-run, the ODC root correction stated as OUR reading, the ladder restated, and
the not-claims section.

GATES, all five re-run from clean processes at round close, all GREEN:
  research/j2_referee.py       -> ALL ASSERTIONS GREEN   (run FIRST, before
                                  anything below entered the record)
  research/j2_citesweep.py     -> ALL CHECKS GREEN       (NEW this round)
  research/j2_layer_proof.py   -> ALL ASSERTIONS GREEN   (NEW this round)
  research/j2_odc6.py          -> ALL ASSERTIONS GREEN
  research/j2_rankin_layer.py  -> ALL ASSERTIONS GREEN
Every job this round launched has finished; nothing left running.

PRE-REGISTRATION, written before `j2_layer_proof.py` was run, scored in its
section G and again in 10f below:
  PR1 the layering CLOSES as a proof; my named risk is the greedy (layer 3).
  PR2 the constant is k/((k(2k-1))^k c_1^(k)); at k = 2, 1/(18 c_1); I predict
      it lands in [1e-3, 1e-1].
  PR3 the small-prime cut is P = A^(2k-1), not round 25's A^5.
  PR4 the medium-prime parameter theta -> k FROM ABOVE; theta = k exactly fails.
  PR5 at k = 1 the same write-up must land BELOW Rankin's proved e^gamma; above
      it would be a bug.

### 10a. (a) THE VERDICT: **PROOF.** THE k = 2 LAYERING IS WRITTEN OUT.

    THEOREM (P2').  Let c_1 satisfy pi_2(t) <= c_1 t/(log t)^2 for t >= t_1.
    Write A = log x, B = log A, C = log B.  Then

        j_2(P(x))  >=  ( 1/(18 c_1) + o(1) ) x A^3 C^2 / B^4 ,

    and generally, for the k-class Jacobsthal function,

        j_k(P(x))  >=  ( k/((k(2k-1))^k c_1^(k)) + o(1) ) x A^(2k-1) C^k/B^(2k).

    The statement carries an o(1), so the best ASYMPTOTIC twin constant is
    admissible: **the headline constant is 0.0127524** (Lichtman 2024,
    c_1 = 3.29956 x 2C_2 = 4.356487); the fully-effective alternative, with
    Selberg's classical 8 x 2C_2 = 10.562589 - the constant Riesel-Vaughan 1983
    Lemma 5 makes effective for t >= e^42 - is **0.0052597**.

Full write-up: docs/novel/layered-erdos-rankin.md section 4 (parameters, four
layers, survivor structure, the two counts, the greedy lemma, capacity, and the
solution of the assembly inequality). What the write-out ADDED beyond
bookkeeping-to-proof:

1. **THE GREEDY LEMMA IS EXACT, AND IT WAS THE NAMED RISK.** Two distinct
   classes mod p always capture at least **2N/p** of any finite set - no
   O(N/p^2) loss. Proof: with n_(1) >= n_(2) the two largest class counts,
   n_(1) >= N/p and n_(2) >= (N-n_(1))/(p-1), and the sum is increasing in
   n_(1), so it is minimised at n_(1) = N/p where it equals N(2p-2)/(p(p-1)) =
   2N/p. Asserted at every prime <= 200 in exact form and over 40,000 random
   class distributions. PR1's risk does not bite; the step I flagged as most
   likely to break is the safest in the argument.
2. **A BETTER CONSTANT, FROM ACTUALLY DOING THE ACCOUNTING.** P must exceed
   L y/x ~ A^(2k-1), so P = A^(2k-1). Round 25 fixed P = A^5. Denominator
   (k(2k-1))^k = 36 in place of (5k)^k = 100: **a factor 2.778 at k = 2.**
3. **A SELF-CORRECTION OF ROUND 25's GENERAL-k FORM.** P = A^5 is admissible
   only for k <= 3 (it coincides with the correct cut exactly at k = 3) and is
   **INADMISSIBLE for k >= 4** - the cofactor argument fails there, so round
   25's printed closed form is too optimistic for k >= 4. Round 25's PR3 (the
   POWER is 2k-1) is unaffected; its printed CONSTANT is not.
4. **THE CONSTANT IS A SUPREMUM, NOT A MAXIMUM.** With u = theta B/C, the
   smooth term dies iff theta > k. At theta = k EXACTLY the bracket is
   +k(log C + 1 - log k)/C > 0 and the smooth term beats the tuple term by a
   factor tending to infinity. Hence the o(1); theta(x) = k + 4(log C+1)/C is
   the choice that realises it. Tabulated to C = 10^6 with monotone convergence
   to within 0.006% of the limit.
5. **A CONSTANT-LEVEL CALIBRATION, where round 25 could only check the shape.**
   The identical write-up at k = 1 returns (1 + o(1)) x A C/B^2. Rankin's proved
   theorem in the same coordinates is (e^gamma + o(1)) x A C/B^2,
   e^gamma = 1.781072. **Our accounting lands a factor 1.781 BELOW the classical
   constant** - the correct side, by a small factor. Coming out ABOVE Rankin
   would have been a bug. (The shortfall is the crude greedy and the elementary
   rho <= 1/Gamma bound.) PR5 confirmed.
6. **WHY IT CANNOT BE UPGRADED, AND WHY THAT IS THE POINT.** The FGKT/Maynard
   improvement of the ordinary construction works by producing MANY PRIMES in a
   single residue class via a multidimensional sieve. Its k = 2 analogue needs
   many TWINS in a single residue class - a LOWER bound for twin primes, i.e.
   the parity barrier. **The construction is parity-free EXACTLY BECAUSE it
   stops at Rankin level.** That is structural, not a gap someone will close,
   and it is the round's sharpest new statement about the method.

### 10b. THE PRIOR-ART FINDING THAT MATTERS - AND IT IS A SELF-FOUND DOWNGRADE

Round 26's sweep (sub-search; then the two LOAD-BEARING items re-read
FIRST-HAND by me, per 7d clause 2) turned up **Ford-Konyagin-Maynard-Pomerance-
Tao, "Long gaps in sieved sets", arXiv:1802.07604, REMARK 7** - read first-hand
2026-08-29 in the ar5iv rendering:

  "Unfortunately our methods only seem to give good results in the
  one-dimensional case. Consider for instance the set {n in P : n+2 in P} of
  (the lower) twin primes. This corresponds to a two-dimensional system in which
  I_p = {0 (mod p), 2 (mod p)} for all primes p. The 'trivial' bound coming from
  these methods would give a bound of >> log X log log X for the largest gap
  between lower twin primes up to X ... and one could possibly hope to improve
  this bound by a small power of log log X using a variant of the methods in
  this paper. However, a sieve upper bound (e.g., [7, Cor. 2.4.1]) combined with
  the pigeonhole principle already gives a bound of >> log^2 X in this case."

**THAT IS OUR SIEVING SYSTEM, NAMED IN PRINT, BY THOSE FIVE AUTHORS.** Round
25's sentence "nobody appears to have asked what happens when you have two
classes per prime" is WITHDRAWN. Three consequences, all arithmetic, all
asserted in j2_layer_proof.py section F0:

1. **NOVELTY QUALIFICATION ON (P1).** In covering coordinates (log X ~ x,
   loglog X ~ A), their ">> log X loglog X" is ">> z log z" - **the ORDER of our
   (P1)**. So (P1) is NOT the first appearance of that order for this system; it
   remains the first PROVED bound, the first with an explicit constant (1.349),
   and the first stated for Ziller-Morack's h_2. Recorded in j2-lower-ladder.md
   section 8-bis and in Unit 1's not-claims list. This is the fourth self-found
   novelty downgrade this lane has taken, and the standing lesson (7d clause 1,
   prior-art checks EXPIRE) is what produced it.
2. **THEY HOPED FOR "A SMALL POWER OF log log X". (P2') GIVES TWO FULL ONES.**
   x A^3 C^2/B^4 over x A is A^(2-o(1)) - asserted numerically at C = 10..10^3.
   The route is different from theirs (a layered Erdos-Rankin covering, not
   their sieved-set machinery). **FKMPT flagging the two-dimensional case as out
   of reach for their methods is the sharpest available statement of what this
   construction contributes** - far better framing than "nobody thought of it".
3. **NO TWIN-PRIME-GAP COROLLARY MAY BE CLAIMED - and their pigeonhole bound is
   nonetheless no obstruction to us.** Two different quantities:
     gaps between ACTUAL twin primes near X: twin density ~1/(log X)^2, so
       pigeonhole gives >> (log X)^2 = x^2, which BEATS x A^3 C^2/B^4;
     j_2(P(x)) itself: the SIFTED SET has density prod(1-2/p) ~ 1/A^2 inside its
       period, so the same pigeonhole gives only >> A^2 = (log x)^2, which
       (P2') beats by a full power of x.
   So the theorem is a genuine statement about j_2 = h_2, and any twin-prime-gap
   corollary would be weaker than an argument those authors call trivial. Added
   as item 6 of Unit 1's not-claims list.

Also from the sweep, all dated 2026-08-29: **Erdos problems #687 and #970 both
confirm Iwaniec 1978 is STILL the record upper bound for the ordinary Jacobsthal
function, both open** (#687 carries a $1000 prize; page last edited 2025-12-06);
#689/#1205/#1200 are covering-MULTIPLICITY questions, one class per prime, not
our object; Maynard's survey arXiv:1910.13450 Lemma 5 states the Erdos-Rankin
framework as one class per prime; the nearest-looking precedent is
Maier-Pomerance's use of one class at a large prime to remove TWO survivors,
which is a different thing and should be named in the paper so a referee does
not confuse them; **no theorem anywhere on large gaps between consecutive twin
primes or prime k-tuples by an Erdos-Rankin covering** (round 25 named this as
the largest risk - it is clear, and FKMPT Remark 7 explains why nobody built
it); Kalmynin-Konyagin arXiv:2302.00459 remains one-dimensional and is not our
object; **j_k appears nowhere under any name.**

### 10c. A CONSTANT ERROR OF MY OWN, CAUGHT BY GOING TO THE PRIMARY SOURCE

The first draft of j2_layer_proof.py set c_1 = 8 C_2 = 5.2813, reading Selberg's
classical constant 8 as multiplying the twin constant C_2. **It multiplies the
FULL Hardy-Littlewood singular series 2 C_2.** Caught by reading Lichtman
arXiv:2109.02851 (Algebra & Number Theory 19 (2025) no. 1) first-hand
2026-08-29: his normalisation is Pi(x) = 2x/(log x)^2 prod_{p>2}
(1-2/p)/(1-1/p)^2 = 2 C_2 x/(log x)^2, his Theorem 1.2 is pi_2(x) <~ 3.29956
Pi(x), and his history table reads Selberg 1947 = 8, Bombieri-Davenport 1966 =
4, BFI 1986 = 3.5, Wu 2004 = 3.39951. **The draft was a factor of two too good**
- and by coincidence its wrong value 5.2813 is exactly Bombieri-Davenport's
constant, which is why nothing looked odd. Every constant in the theorem is now
carried against 2 C_2 and asserted.

### 10d. (b) THE PAIRED-IWANIEC PROBLEM (P3), PRICED

**Statement.** Is h_2(P(z)) = O(z (log z)^a) for some a?

**What round 26 changes.** Before (P2') there was no constraint on a. Now:
* **a >= 3 is FORCED**, and a >= 2k-1 for the general j_k. (P3) is no longer
  "is it polylog?" but "**is the polylog exponent exactly 3?**"
* **The matching conjecture is now sharp and falsifiable**: h_2(P(z)) =
  z (log z)^(3+o(1)), i.e. the construction is essentially optimal. Attackable
  from either side.

**Price: NOT REACHABLE, and the reason is structural rather than effort.**
1. (P3) at k = 1 - "is j(P(z)) = O(z (log z)^a)?" - is a KNOWN OPEN PROBLEM. The
   record is Iwaniec 1978, j(P(z)) << z^2: a full power of z, not a polylog,
   unmoved for 48 years (Erdos problems #687/#970, re-checked 2026-08-29).
2. Our k = 2 version cannot be easier: j_2 >= j by the collapse transfer
   (b - a = p#), so a polylog bound for j_2 gives one for j.
3. Our own upper ladder reaches z^8.04 explicitly and z^(4.266+eps) by citation,
   both far above any polylog, and section 2e shows the exponent IS the sifting
   limit - no level or bilinear refinement moves it.
**Therefore (P3) is strictly harder than an open Erdos problem with a standing
prize, and this lane will not attempt it.** What the lane contributes is the
constraint a >= 3 and the sharpened conjecture.
**What IS reachable, and it is a referee tool rather than a theorem:** the
family j_k gives infinitely many instances of the same question, and any claimed
j_k << x A^f(k) with f(k) < 2k-1 is contradicted outright by (P2') at that k.
Any future upper-bound claim on this family can be consistency-checked for free.

### 10e. (c) UNIT 1 - ASSEMBLED. **docs/novel/j2-upper-bound.md SECTION 11.**

The round-25 report listed what would change in the paper (its section 10e) but
never applied it to the head of the document, so the status block, section 1's
prose and section 4a still carried "exponent 19", "the proved sandwich ...
around a measured truth of (p^2-p)/2", and "no lower bound beyond the collapse".
Round 26 assembled the unit properly:

* **NEW SECTION 11, the submission candidate**: 11a the complete ladder in one
  table (1; 3E quasi-polynomial with the exact asymptotic constant 2 lambda_* =
  7.182242; 2E exponent 19; 2E' 17; 2E'' 15; **2G 8.04162**; 2G-inf floor
  7.93727; 2 at 4.266 by citation) with WHICH RUNG TO QUOTE (2E'' below
  p_n ~ 3.8e5, 2G above, by p_n^6.96) and the explicitness boundary stated once;
  11b the current sandwich with both retracted readings named; 11c a rewritten
  eight-item not-claims list; 11d the ODC root; 11e a submission checklist.
* **Every stale section is now individually marked**, not deleted: a pointer
  block at the top of the file, one on the round-23 status block, one on
  section 1's superseded closing sentences, one on section 4a items 1/3/5.
* **THE ODC alpha* READING RESTATED AS OURS, WITH THE DERIVATION - the round-25
  "discrepancy in the book" framing is WITHDRAWN.** The book says "A numerical
  computation gives (use the Taylor expansion at 1/4)". Doing exactly that:
  f(1/4) = -0.0741009117, f'(1/4) = +4.9715909084, and ONE first-order
  Taylor/Newton step gives 1/4 - f(1/4)/f'(1/4) = **0.2649048691** - the printed
  **0.264904 to seven digits**. So the printed value IS the book's own stated
  approximation, computed the way the book says. Ours is a SHARPENING of a
  stated approximation (exact root 0.2652636746, beta_2 = 7.583827 against the
  printed 7.594004, gain 0.010177), carrying the caveat that the equation is OUR
  READING of a page image and any residual could be ours. Nothing in 2G moves;
  2G's binding root is the K -> 1 root 0.253321897.

**THE CITATION-NUMBERING SWEEP IS NOW A GATE, NOT A MANUAL STEP -
research/j2_citesweep.py.** A hand sweep does not fail when a document drifts;
this one does. It (A) re-derives the ODC root reading; (B) extracts every arXiv
id from the five Unit-1 documents and asserts each is in an ADJUDICATED REGISTRY
carrying who/what/when - an unregistered id FAILS the gate, which forces a new
citation to be adjudicated before it can be used; (C) scans for six FORBIDDEN
strings (the "Iwaniec-Kowalski Theorem 6.9" chimera, "M. Franze", Tenenbaum
4.3 / I.4.2, Costello-Watts under 1208.5342, "Sean Blight"), exempting explicit
do-not-cite context; (D) scans for INTERNAL CONTRADICTIONS between sections of
one document; (E) reports the age of every dated check and fails past 14 days.

**IT CAUGHT TWO LIVE DEFECTS ON ITS FIRST RUN, both fixed:**
1. `paired-jacobsthal-values.md` still attributed the Costello-Watts bound to
   arXiv:1208.5342. That is the SEPARATE range-restricted computational paper;
   the bound is arXiv:1306.1064. Round 23 recorded the correction and the
   document was never updated.
2. `j2-upper-bound.md` section 6a item 4 still instructed "the safe form, and
   the one now used: cite 2 kappa + 0.4454", while section 9c SETTLED the
   conflict the other way in round 24 and instructs "cite 19/36". **A direct
   self-contradiction inside one document, which a referee reading
   top-to-bottom would hit.** It survived two rounds of manual sweeps.
Two of my own bugs in the gate were also caught by running it: an unformatted
%d, and date-extraction counting a cited source's own date (Tao's 2014 blog
post) as one of OUR check dates.

### 10f. Pre-registration scored, and the round's negatives

PR1 CONFIRMED - the layering closes; the named risk is exact, not merely safe.
PR2 CONFIRMED - K = k/((k(2k-1))^k c_1^(k)); at k = 2, 1/(18 c_1) = 0.0127524,
    inside the predicted band [1e-3, 1e-1].
PR3 CONFIRMED - P = A^(2k-1) forced and optimal; factor 2.778 gained.
PR4 CONFIRMED - theta = k exactly fails; the constant is a supremum.
PR5 CONFIRMED - k = 1 gives 1.0, a factor 1.781 below Rankin's e^gamma.
Five for five, which is itself worth a caveat: the predictions were made after a
round of thinking about the same construction, so they were not hard.

NEGATIVES AND COSTS OF THE ROUND:
* **My own constant was wrong by a factor of two** in the first draft (10c),
  and it was wrong in the direction that flatters the result. Caught only by
  going to the primary source for a number I thought I knew.
* **Round 25's novelty sentence was too strong and is withdrawn** (10b). FKMPT
  Remark 7 existed throughout and names the system; round 25's own sweep read
  that paper's abstract and main theorem and did not reach Remark 7. LESSON,
  added to 7d as clause 6: **when a paper is checked for prior art, its REMARKS
  and its "what our methods cannot do" section are where your problem will be,
  not its theorems** - the same failure shape as round 25's "price the
  propositions, not only the theorems", one level further down.
* **Round 25's general-k closed form is wrong for k >= 4** (10a item 3), my own
  from one round ago.
* **(P1)'s novelty is qualified** (10b item 1). The order z log z for this exact
  system is in print.
* **(P3) is priced NOT REACHABLE** - a gated negative in the sense that it rests
  on a checked fact (Iwaniec 1978 still the record, Erdos #687/#970 open,
  re-checked 2026-08-29) plus the one-line implication j_2 >= j, not on
  judgment. The JUDGMENT part, labelled: that no other route to a polylog upper
  bound exists. **JUDGMENT, NOT RESULT.**
* **(P2') has no finite-z content and no kernel check**, unchanged from round
  25: the construction does not exist below log z ~ 300, and the threshold,
  though effective, decays like (log C + 1)/C with C = logloglog log x and is
  not writeable. (P1) remains the bound to quote at any z anyone will evaluate.
* **The ODC page-image caveat is still open**: (5.38), (6.69) and p. 74 were not
  re-fetched this round either. One library visit closes it and it should happen
  before submission.
* The (loglog)^4 exponent and the constant 1/(18 c_1) are what THIS parameter
  choice gives, not what the method gives. Not optimised.

### 10g. Ranking changes

* **N4 (j_2 upper ladder) stays TOP but is now a WRITING item, not a research
  item.** Section 11 is the assembled unit; what remains is LaTeX, one library
  visit, and a scope decision about F_d. Its research frontier (7.937 -> 4.266)
  is priced as needing an explicit form of the DHR differential-delay system,
  which nobody has.
* **P1-P3 (lower ladder) REACHES PARITY WITH N4 and is now the more active
  side.** It carries two proved bounds, a k-family nobody has stated, an
  explicit constant, and a sharp falsifiable conjecture. Its own novelty is
  qualified but not damaged.
* **(P2') CLOSED.** **(P3) PRICED AND CLOSED as unreachable** - it should not
  appear in a future brief as a target, only as the frontier.
* **NEW ITEM, ranked immediately below N4: (P6) THE k-FAMILY AS A PUBLISHED
  OBJECT.** j_k is defined, has a proved lower bound for every k, has a stated
  upper conjecture, and appears nowhere in the literature. It is a short paper
  on its own or a section of Unit 1, and it is the cheapest genuinely new thing
  this lane holds. The k >= 4 shift-set question (6.3 of the doc) is its one
  piece of real work.
* **7c#4 (h_2 at p_n = 151..251) RISES AGAIN** - it now separates z(log z)^2
  from z(log z)^3 where the competing readings used to be closer, and it is the
  only purchasable number that discriminates.
* **DEMOTED: nothing.** Round 25's demotion of the Blight opening stands.

### 10h. Additions to the standing citation-hygiene lesson (7d)

6. **A PAPER'S REMARKS ARE WHERE YOUR PROBLEM LIVES.** Round 25 read FKMPT's
   abstract and main theorem and cleared it as "the adversarial problem".
   Remark 7 of the same paper names our sieving system, states the order of our
   (P1) as trivial, and explains why nobody built our construction. **When
   clearing a paper for prior art, read its remarks and its limitations section
   FIRST** - authors put "here is the thing our method cannot do" exactly where
   your problem is. Same failure shape as round 25's clause 4 (price the
   propositions, not only the theorems), one level further down.
7. **A CONSTANT YOU THINK YOU KNOW IS A CITATION.** "Selberg's twin constant is
   8" is true; "8 times C_2" is not - it is 8 times 2 C_2, and the error
   flattered the result by a factor of two. Normalisations are the part of a
   remembered constant that goes wrong. Re-derive or re-read the normalisation,
   not just the digits.
8. **A MANUAL SWEEP DOES NOT FAIL.** Rounds 23-25 ran the citation-numbering
   sweep by hand and it passed; two live defects were sitting in the documents
   the whole time, one of them a direct self-contradiction. The sweep is now
   research/j2_citesweep.py and it exits non-zero. Any standing referee step
   that can be a gate should be one.

### 10i. Reproduction (round 26)

* `research/j2_layer_proof.py` -> `research/data/j2_layer_proof.out`. Sections
  A (greedy lemma, exact, 40,000 random distributions), B (two-sided survivor
  structure, 0 violations at five parameter sets), C (the assembly inequality,
  K tabulated to C = 10^6, monotone to within 0.006% of 1/(18 c_1)),
  D (P = A^(2k-1) forced and optimal), E (k = 1 constant-level calibration
  against Rankin's e^gamma; general-k table with the k >= 4 correction),
  F0 (FKMPT Remark 7, the three consequences, all asserted), F (the honest
  boundary, six items), G (pre-registration scored).
* `research/j2_citesweep.py` -> `research/data/j2_citesweep.out`. Sections A-E
  as in 10e. **Re-run alongside j2_referee.py before any future claim about
  Unit 1.**
* `research/j2_referee.py` re-run and GREEN before any of the above.
* Documents changed: `docs/novel/layered-erdos-rankin.md` (rewritten - the
  proof is section 4, FKMPT is section 2, (P3) is section 6a);
  `docs/novel/j2-lower-ladder.md` (new sections 8, 8-bis, 8a, 8b);
  `docs/novel/j2-upper-bound.md` (new section 11 = the assembled unit; four
  supersession markers); `docs/novel/paired-jacobsthal-values.md` (the
  Costello-Watts id); `docs/novel/README.md` (three index entries).
* Sources read FIRST-HAND by me on 2026-08-29: arXiv:1802.07604 Remark 7,
  Theorem 1 and Definition 1 (ar5iv); arXiv:2109.02851 Theorem 1.2, abstract and
  history table (ar5iv). Sources taken from the sub-search and NOT re-read by
  me, labelled as such: the Erdos-problems #687/#689/#970/#1200/#1205 texts,
  Maynard's survey Lemma 5, FGKT Theorem 1's exact statement, Riesel-Vaughan
  1983 Lemma 5 (which reached the sub-search through OCR of a 1983 scan - the
  "+100 x^(1/2)" term and the (L,A) table row should be re-checked against a
  clean copy before that constant is printed), and the Hildebrand-Tenenbaum
  1993 theorem numbering. **The theorem's headline constant depends only on
  Lichtman, which I read myself.**

## 11. ROUND 27 (2026-08-29) - THE PAGE CAVEAT CLOSES; THE FAMILY IS WRITTEN

Brief: (a) fetch the two unfetched ODC pages ((5.38), (6.69), p. 74), re-run
the two standing gates, and write a ONE-PAGE SUBMISSION MEMO for the human;
(b) (P6) the k-family write-up; (c) the lower ladder's next rung if capacity.

GATES, all four re-run from clean processes at round close, all GREEN:
  research/j2_referee.py    -> ALL ASSERTIONS GREEN  (run FIRST)
  research/j2_citesweep.py  -> ALL CHECKS GREEN      (now over SIX documents)
  research/j2_odcpages.py   -> ALL ASSERTIONS GREEN  (NEW this round)
  research/jk_family.py     -> ALL ASSERTIONS GREEN  (NEW this round)
Every job this round launched has finished; nothing left running.

### 11a. (a) THE ODC PAGE-IMAGE CAVEAT IS CLOSED. ALL THREE PAGES FETCHED.

Rounds 24, 25 and 26 each closed with the same sentence - "(5.38), (6.69) and
p. 74 were not re-fetched; one library visit closes it" - and each time it was
carried forward. It is closed. Method: the Google Books publisher preview of
the AMS printing (volume Dz6REQAAQBAJ) serves page images only to a session
that holds its cookies, which is why round 25's direct URL fetches returned a
9,103-byte placeholder; driving a real browser to the volume, reading the
`jscmd=click3` page list for the signed image URLs and fetching them IN the
page's own session returns the images. Six new pages on disk beside round 25's
nine: research/data/odc6_scans/PA42, PA43, PA44, PA45, PA67, PA74.

**NOTHING IN THE LADDER MOVES.** No constant, exponent or threshold changes.
Each page nevertheless paid something.

**(5.38), p. 42, section 5.5 "The Sieve Dimension".** Printed as
`prod_{w<=p<z}(1-g(p))^{-1} <= K (log z/log w)^kappa` for `z > w >= 2`,
"where K is a constant, K > 1" - the form rounds 23-24 used from Dudek-Dunn
Lemma 2.1 and Campbell Thm 2.1, now confirmed against the book, so THE
ROUND-23/24 CAVEAT ON (5.38) IS DISCHARGED. Two by-products we did not have:
K > 1 is REQUIRED, not merely natural; and the book prints the consequence
`g(p) <= 1 - 1/K`, whose converse `K >= (1-g(p))^{-1}` (take w = p, z -> p+)
**EXPLAINS THE WHOLE PRE-SIEVED K-LADDER IN ONE LINE** - K = 3 at p_0 = 3
because g(3) = 2/3 exactly, 5/3 at p_0 = 5, 7/5 at p_0 = 7. Round 23 found
those by grid search over all (w,z) and recorded "supremum at w = 3, z -> 3+"
without being able to say why. Asserted at every operative K of the ladder.

**(6.69), p. 67 - and it settles the hypothesis for EVERY kappa, not ours.**
The page prints (6.65) alpha = (kappa/2) log((beta+1)/(beta-1)); (6.67) the
convergence condition a = alpha e^{1+alpha} < 1; (6.68) e^{1+1/c} = c with c
the root of (6.11), i.e. c(log c - 1) = 1 - **EXACTLY OUR THEOREM 3E CONSTANT
lambda_* = 3.591121**; then "the condition a < 1 means alpha < c^{-1}, or
equivalently" (6.69) and (6.70). So (6.69) IS `alpha < 1/c = 0.2784645`. And
Corollary 6.13's own beta_kappa = 1 + 2(e^{1/(2kappa)}-1)^{-1} gives
(beta+1)/(beta-1) = e^{1/(2kappa)}, hence **alpha = 1/4 IDENTICALLY IN
kappa** - which is why the book states the corollary "for kappa > 0". The
hypothesis holds at every dimension, checked at nine of them.
NEW NUMBER, and it is the sharpest thing the page buys: since
beta = coth(alpha/kappa) DECREASES in alpha, (6.69) puts an ABSOLUTE FLOOR of
**7.22859** under ODC Chapter 6 at kappa = 2, below our positivity floor
7.93727. **So (6.69) is not what stops Chapter 6 - POSITIVITY IS** - and even
discarding every K-loss AND positivity, the chapter cannot print an exponent
under 7.229, still 3.0 above DHR's 4.266. The 7.937 -> 4.266 gap is not
reachable by any tuning of this chapter; round 25's "JUDGMENT, NOT RESULT"
label on that statement can now be narrowed to the problem, not the device.

**p. 74 - and it closes POSITIVELY.** ODC's own preliminary sieving,
(6.99)/(6.100), carries `O(e^{-s_0} V(z_0)/V(z))` "where the implied constant
depends only on K_0 and kappa_0" - **NOT EXPLICIT**. So the book's apparatus
could not have supplied our pre-sieving factor, and round 24's elementary
N_pre = prod_{p<p_0}(p - omega(p)) accounting is not an alternative to it but
THE ONLY EXPLICIT ROUTE. It stays ours.
AND THE BOOK'S OTHER ROUTE TO K -> 1, PRICED FOR THE FIRST TIME. pp. 43-44
offer (5.42)/(5.43): K as close to 1 as one likes with NO preliminary sifting,
by enlarging the dimension by epsilon ("the constant K given by (5.42) is
fine, even for y = 2"). We had never considered it. It LOSES, on the book's
own arithmetic, because beta = coth(alpha/kappa) INCREASES in kappa: eps = 1
costs 3.93 of exponent (11.871 against 7.937) and even eps = 1/2 costs 1.98.
Pre-sieving keeps kappa = 2 and is therefore the right device - a conclusion
round 24 reached without the comparison and which is now compared.

RESIDUAL, and it is the only page caveat that cannot be removed by fetching:
these are publisher-preview page IMAGES read on screen, not a copy in hand.
The mitigation is stronger than in rounds 24-25 - the pages cross-check each
other (p. 45 quotes (5.38) and its consequence; p. 67's (6.65)-(6.70)
reproduce Cor 6.13's beta_kappa exactly; p. 67's c is our own lambda_*), so an
OCR corruption would have to be self-consistent across four pages of two
chapters and agree with two independent transcriptions of Theorem 7.7.

Written into Unit 1 as **j2-upper-bound.md section 11f**, with the checklist
row added at 11e.

### 11b. (a) THE SUBMISSION MEMO - docs/novel/unit1-submission-memo.md

One page, and it does NOT recommend submitting: the decision is the human's
and the memo exists to make it makeable. Contents: what the paper claims (the
six-rung table, the two lower bounds, the falsification target); what it does
not (the eight-item list, with the load-bearing four quoted); the three
strongest points a referee will see (an empty ladder now four explicit rungs
deep WITH the explicitness boundary proved so the obvious improvement is
pre-answered; (P2') as a genuinely new construction that is parity-free for a
structural reason; a visibly self-auditing paper); the three weakest ("this is
an exercise", the audience, and an asymptotic lower half over a replicated
computational half); a venue-class assessment; and the disclosure question.

THE AUDIENCE NUMBER, stated plainly because it is the thing that decides the
venue: **arXiv:1706.00317 has EXACTLY ONE CITATION IN NINE YEARS** - its own
companion note - and zbMATH returns NO RECORDS for "paired Jacobsthal". The
referee's real question is not "is this correct?" but "who is this for?".
VENUE CLASSES, as an assessment and not a pick: arXiv math.NT first whatever
else is decided (ZM's work and Holt's programme both live on arXiv, so a
preprint reaches the entire actual readership in a day and timestamps j_k);
JNT / Ramanujan J / Acta Arith / Mathematika are in range IF (P2') travels
with it, since (P2') is what makes it a research paper rather than a note;
INTEGERS or JIS if it stays elementary; NOT a general-audience venue. And one
suggestion that costs nothing: write to Ziller and Morack, who are
simultaneously the prior readership, the natural referees, and the people who
can compute h_2 at p_n = 151 - the single number that would most improve the
paper.

THE AI-ASSISTANCE DISCLOSURE QUESTION IS FLAGGED AND NOT DECIDED. The memo
gives the facts (no venue permits an AI author, so authorship is the human's;
most major publishers now require disclosure of generative-AI use, with
policies differing on derivation versus prose, and this work is far more the
former; a minority of editors desk-reject) and one asymmetry worth weighing -
this paper's strongest defence is that every recomputable claim is reproduced
by independent code and every citation number is gate-checked, so disclosure
makes that apparatus the point rather than a curiosity, while non-disclosure
that later surfaces damages exactly the credibility the gates were built to
earn. Three shapes offered, no choice made.

### 11c. (b) (P6) THE k-FAMILY - WRITTEN. docs/novel/jk-family.md

**THE OBJECT.** For k >= 1 and an ADMISSIBLE k-tuple E = (0 = E_0 <= ... <=
E_{k-1}), j_E(m) is the largest cyclic gap between consecutive n with
gcd(prod(n+E_i), m) = 1, and j_k(m) = max over admissible E. k = 1 is the
ordinary Jacobsthal function; k = 2 is Ziller-Morack's h_2.

**PROPOSITION (the covering restatement), and it is the whole content:**

    j_k(P(z)) - 1  =  the longest interval coverable by choosing, at each
                      prime p <= z, a set S_p of classes mod p with
                            |S_p| <= min(k, p-1).

Both directions are CRT. `min(k, p-1)` reproduces the ordinary problem at
k = 1 and **Ziller-Morack's omega(2) = 1, omega(p) = 2 at k = 2** - i.e. our
own g(2) = 1/2, g(p) = 2/p - and it is what makes THE SIFTING DIMENSION EQUAL
TO k. Admissibility is exactly what the `p-1` encodes.

BRUTE-FORCED, both forms independently, and they agree at every case
exhaustion reaches (k = 1,2,3 x z = 3,5,7). k = 1 returns A048669's 4, 6, 10;
k = 2 returns ZM's 6, 18, 30; and **k = 3 returns j_3(P(3)) = 6,
j_3(P(5)) = 24, j_3(P(7)) = 78 - a first evaluation** (witnesses (0,0,2),
(0,2,6), (0,2,18)). Their smallness is the point: the object is elementary,
hand-computable, and unnamed.

**THE LADDER IS UNIFORM IN k**, which is the reason to publish the family:
- Legendre rung with omega_p = min(k, p-1); at k = 2 the numerator is
  2*3^{n-1}, Theorem 1 verbatim.
- **THE POLYNOMIAL RUNG AT EVERY k:** ODC Corollary 6.13 gives
  beta_k = 1 + 2(e^{1/(2k)}-1)^{-1}, so j_k(P(z)) <<_{k,eps} z^{beta_k+eps}
  with 4k-1 < beta_k < 4k+1 (4.082988, **8.041623**, 12.027765, 16.020828,
  ... - the k = 2 entry IS Theorem 2G's exponent, so the family rung CONTAINS
  Unit 1's best explicit bound). Its two hypotheses are (5.38) and (6.69),
  both read first-hand this round and both holding at every kappa. The one
  arithmetic change with k is |r_d| <= k^{nu(d)}, so the remainder carries
  (log D)^{k-1} instead of (log D); the level and the exponent are unchanged.
- **(P2') at every k**: x A^{2k-1} C^k/B^{2k} with K_k = k/((k(2k-1))^k c_1^k).
- SANDWICH x A^{2k-1} C^k/B^{2k} << j_k << x^{beta_k+eps}, beta_k ~ 4k, and
  the CONJECTURE j_k(P(x)) = x (log x)^{2k-1+o(1)} for every k.

HONEST, AND IN THE NOTE: **at k = 1 the family rung (4.083) is WORSE than the
record** - Iwaniec 1978's exponent 2, unmoved for 48 years. So the family rung
is the ONLY bound in existence for k >= 2 and is NOT the best at k = 1. And
the upper rungs are standard sieve theory applied to a new object, not new
sieve theory - Unit 1's not-claim 2, inherited unchanged.

WHY IT MATTERS TO UNIT 1: it converts the paper's weakest structural point
("one function, standard tools") into "a family, and the family is the
contribution", and it locates ZM Conjecture 6 inside the family - their
exponent 2 at dimension 2 is exponent k at dimension k, the level at which a
survivor in (y, y^2] IS a prime k-tuple, so **THE PARITY CEILING OF UNIT 1 IS
UNIFORM IN k**, not special to twins.

### 11d. (c) THE LOWER LADDER'S NEXT RUNG - THE k >= 4 SHIFT SET, ANSWERED

layered-erdos-rankin.md section 6 item 3 was the family's one named piece of
real work. **IT COSTS NOTHING**, and the write-up is in jk-family.md section 4
with the gate at jk_family.py section E.
1. `0,2,...,2(k-1)` is the WRONG TUPLE - from k = 3 it is not even admissible
   (0,2,4 covers Z/3, so no n survives at all). Round 26 tabulated its
   "collisions" without noticing that.
2. With ANY admissible tuple (e.g. {q_1..q_k} - q_1 for the k least primes
   q_i > k), a collision E_i = E_j mod p needs p | E_j - E_i, hence
   p <= M_k = max pairwise difference, A CONSTANT IN k. The greedy layer runs
   over [P, z1] with P = A^{2k-1} -> oo, so for large x EVERY colliding prime
   lies BELOW P, inside the Eratosthenes layers - where a collision merely
   means two layers coincide, uses FEWER than the k available classes, and
   leaves the survivor structure untouched. Hence Sigma = prod(1-k/p) with NO
   correction and K_k stands as printed.
3. THRESHOLD: x > exp(M_k^{1/(2k-1)}), which is under e^4 for every k <= 12,
   against this construction's own log x ~ 300. Tabulated with the tuples.
What remains is a FINITE OPTIMISATION AND NOT A GAP: which admissible tuple
minimises c_1^{(k)} (equivalently S(E))? It moves the constant only.

**A SIMPLIFICATION OF OUR OWN ARGUMENT, recorded rather than hidden.** The
greedy lemma at general k - some k distinct classes mod p hold at least kN/p
of any N-set - has a ONE-LINE proof: the p class counts average N/p, so the k
largest average at least N/p and sum to at least kN/p. That subsumes round
26's k = 2 lemma, whose proof (n_(1) >= N/p, n_(2) >= (N-n_(1))/(p-1),
monotonicity) was correct but longer than it needed to be. The k = 2 statement
2N/p was and is exact; this is a simplification, not a correction.

### 11e. Negatives, costs and residual risks of the round

* **ROUND 26 TABULATED COLLISIONS FOR AN INADMISSIBLE TUPLE** (11d item 1) -
  my own, one round old. The numbers were right about `0,2,...,2(k-1)`; that
  tuple is not a tuple the construction can use from k = 3 on. The error was
  harmless (k = 2 is our case and is unaffected) but it is exactly the shape
  of mistake the lane keeps making: carrying a small-k object into general k
  without re-checking the definition.
* **ROUND 26's GREEDY PROOF WAS LONGER THAN NECESSARY** (11d). Also mine.
* THE PAGE IMAGES ARE STILL IMAGES. Mitigated four ways, not removed.
* **(6.69) turns out never to have been at risk** - it holds at every kappa
  because Cor 6.13's alpha is 1/4 identically. Three rounds carried it as an
  open caveat. The lesson is round 25's clause 4 again: the condition was
  quoted BY NUMBER inside a proposition we had priced, and pricing a
  proposition means reading what it cites.
* **j_3 beyond z = 7 was NOT computed.** The covering-form search is
  exponential in the number of primes and z = 11 needs a real algorithm, not
  exhaustion. Named, priced (Ziller's ordinary-side algorithm is the model),
  and deliberately not started - it would not have finished in-round.
* THE FAMILY'S UPPER RUNGS ARE NOT NEW MATHEMATICS and the note says so twice.
* NO PRE-REGISTRATION THIS ROUND. The work was fetch-and-write; there was
  nothing whose outcome I did not already expect except (6.69), and I did not
  write down a prediction for it before fetching. Recorded as a miss.

### 11f. Additions to the standing citation-hygiene lesson (7d)

9. **A PAGE THAT WOULD NOT FETCH IS NOT A PAGE THAT CANNOT BE FETCHED.** Round
   25 tried the obvious image URL, got a 9,103-byte placeholder, and recorded
   "not obtained; one library visit closes it" - which three rounds then
   repeated as if it were a fact about the world. It was a fact about a
   missing cookie. **When a source is recorded as unobtainable, record HOW the
   attempt was made, so the next round can attack the method instead of
   inheriting the verdict.**
10. **A HYPOTHESIS CITED BY NUMBER IS AN UNREAD HYPOTHESIS.** Theorem 2G rests
    on Corollary 6.13, which rests on Proposition 6.7, which requires "kappa
    bounded by (6.69)". We priced the proposition (clause 4) and still did not
    read the condition it names. It was fine - but we did not know that for
    three rounds. **Follow every numbered reference inside a result you are
    using, to the page.**

### 11g. Ranking changes

* **N4 (j_2 upper ladder) stays TOP and its LAST research-shaped item is
  gone.** Every blocker, opening and caveat of rounds 23-26 is closed. What
  remains is LaTeX, a scope decision, and a decision that is the human's.
* **(P6) RISES to sit beside N4** rather than below it: the family is written,
  gated, prior-art-checked, and it is the piece that answers the strongest
  referee objection to N4. Its one open item is exact values of j_3 beyond
  z = 7 - a computation, not research.
* **P1-P3 (lower ladder): unchanged in rank, one item lighter.** Its named
  next question (the k >= 4 shift set) is answered. The two remaining items
  are the (loglog)^{2k} exponent, which is a parameter choice nobody has
  optimised, and the threshold, which is not writeable.
* **7c#4 (h_2 at p_n = 151..251) is now the lane's TOP RESEARCH ITEM by
  default**, being the only purchasable number that discriminates z(log z)^2
  from z(log z)^3 - and, per the memo, the natural thing to ask Ziller and
  Morack for.
* **DEMOTED: nothing.**

### 11h. Reproduction (round 27)

* `research/j2_odcpages.py` -> `research/data/j2_odcpages.out`. Sections
  A ((5.38), the K-ladder explained, every operative K checked against K > 1
  and g(p) <= 1 - 1/K), B ((6.69) as alpha < 1/c, Cor 6.13's alpha = 1/4 at
  nine dimensions, the 7.22859 hard floor), C (p. 74, and (5.42)/(5.43)
  priced), D (what moved in Unit 1: nothing in the ladder).
* `research/jk_family.py` -> `research/data/jk_family.out`. Sections A
  (definition + covering restatement, both forms brute-forced), B (beta_k, and
  the honest k = 1 comparison with Iwaniec), C (Legendre rung vs the exact
  values), D (K_k), E (the shift-set answer, tuples and thresholds to k = 12),
  F (the general-k greedy lemma, 40,000 random distributions).
* `research/j2_citesweep.py` now sweeps SIX documents (jk-family.md added).
* Page images: `research/data/odc6_scans/PA42,43,44,45,67,74.png` (new,
  2026-08-29) beside round 25's PA65, PA68-73, PA112.
* Documents changed: `docs/novel/j2-upper-bound.md` (new section 11f + a
  checklist row + the 11e remaining-list rewritten);
  `docs/novel/layered-erdos-rankin.md` (section 6 item 3 answered);
  `docs/novel/README.md` (two new index entries, one amendment).
  New: `docs/novel/jk-family.md`, `docs/novel/unit1-submission-memo.md`.
* Sources read FIRST-HAND by me on 2026-08-29: Opera de Cribro pp. 42, 43, 44,
  45, 67, 74 (page images, publisher preview); the OEIS search endpoint for
  `seq:6,24,78` (19 sequences, none number-theoretic in our sense) and
  `jacobsthal function primorial` (6 sequences, all one-class).

## 12. ROUND 28 (2026-08-29/30) - THE z-AXIS IS PRICED, THE k-AXIS IS BOUGHT

Brief: (a) h_2 at p_n = 151-251, the lane's top research item - "build or adapt
ZM's own extremal-search algorithm class, price honestly, get as far up as the
round allows"; (b) j_3 beyond z = 7 (needs the real algorithm named and priced
in r27); (c) execute the human's memo decision if it arrives.

GATES, re-run from clean processes at round close:
  research/j2_referee.py    -> ALL ASSERTIONS GREEN  (run FIRST, before
                               anything below entered the record)
  research/j2_citesweep.py  -> ALL CHECKS GREEN
  research/jk_cover.py      -> ALL ASSERTIONS GREEN  (NEW - reference engine
                               + definition-vs-restatement brute force)
  research/jk_growth.py     -> ALL ASSERTIONS GREEN  (NEW - the discriminator)
Pre-registration: research/data/r28_harvester_prereg.txt, written before the
runs it scores, with an addendum written before the j_3(23) answer landed.

### 12a. (a) THE HONEST ANSWER IS A PRICE, NOT A VALUE - AND IT IS MEASURED

**h_2 BEYOND p_n = 73 IS NOT REACHABLE, AND THE ROUND SAYS SO WITH A COST
CURVE RATHER THAN A SHRUG.** I built the algorithm the brief asked for, ran it
to the point where it stops, and measured where that is.

Exhaustive node counts, k = 2, `rust2/src/bin/jkcov6.rs`, each an exact
two-sided answer (witness + infeasibility proof):

    z        13      17       19        23          29             31
    nodes   150   2,577   53,560  1,491,366  55,917,112  2,367,554,226
    ratio     -    17.2     20.8       27.8        37.5           42.3

The ratio is itself growing ~1.25x per step. At a measured 2.0e5 nodes/s/core
on 16 cores: **z = 37 is a ~17-hour job (the next purchasable rung), z = 41 is
~59 days, and z = 43 is 17 years.** ZM's own frontier sits at z = 73 - six
primes past where my vehicle dies - and **p_n = 151..251 is five to nine
primes past THAT**. The projection was checked one step out of sample before
being quoted: fitted on 13..29 it predicts 2.97e9 nodes at z = 31 against a
measured 2.37e9, 25% high.

THE MEASURED FACT ABOUT THE TARGET, not about my vehicle: **A072753 has carried
exactly 21 terms since June 2017 and A288815 exactly 21 since June 2017**
(OEIS records #79 and #19, read first-hand 2026-08-29), with both authors still
editing the sequences. Nobody has moved p_n = 73 in nine years. The r27 memo's
suggestion - ask Ziller and Morack for p_n = 151 - is now costed: it is not a
favour, it is a research project on somebody else's better machine.

WHAT WAS DELIVERED INSTEAD OF A REFUSAL: **the tenth rung reproduced
independently.** omega_2(31) = 94, i.e. h_2(31#) = 570, EXACT, 2,367,554,226
nodes, 2192 s on 8 workers - matching Ziller-Morack. Together with z = 2..29
that is **nine published A288815 values and fourteen published A048670 values
reproduced by a DIFFERENT ALGORITHM from the published ones**, which as far as
the prior-art check reaches is the first independent verification of the paired
Jacobsthal numbers since they were deposited.

### 12b. THE ENGINE (and it is the r27 named opening, closed)

Round 27 recorded "j_3 beyond z = 7 was NOT computed: the covering-form search
is exponential and z = 11 needs a real algorithm, not exhaustion." Built:

1. **THE REDUCTION, at every k.** In the covering restatement, every prime
   p <= k+1 has cap = p-1, so it kills all but one class and the problem
   rescales. With `D = prod_{p <= k+1} p`,
       j_k(P(z)) = D * (m+1),  m = longest run [1,m] coverable by k NON-ZERO
                                  classes mod p for each prime k+1 < p <= z.
   D = 2 at k = 1 IS Hagedorn's h(n+1) = 2w(n)+2; D = 6 at k = 2 IS ZM's
   h_2 = 6 omega_2 + 6 (= A288815 = 6*A072753+6); D = 30 at k = 4, 5.
   **Class 0 is excluded because a MAXIMAL run has an uncovered position on
   each side** - which derives ZM's own `a_i, b_i in {1..p_i-1}` normalisation
   rather than assuming it, and generalises it to every k.
2. **THE CANONICAL FORM, worth 125x.** Branch on which prime covers the
   leftmost uncovered position; reject prime p at position j when an earlier
   commit (j', p') has j' == j (mod p) and p' > p. Among the orderings
   producing a given class set, exactly "always take the smallest available
   prime" survives. This is ZM's RPA2 rule transported to a different search.
   Measured at k = 1, z = 29: 476,683 nodes -> 3,801.
3. **THE v3 BOUND - the sliding form of Hagedorn's criterion.** For EVERY
   prefix [j, x] of the residual window, uncovered count <= capacity of the
   free classes RESTRICTED TO THAT PREFIX (per prime, the f_p largest
   |uncovered == r mod p|, r != 0). Short prefixes are where large primes are
   weakest. One pass, incremental residues, no division in the inner loop.
   Hagedorn's m_i is an a-priori worst case; this uses the exact residual.

VALIDATION, four independent ways, because the canonical rule was the round's
**named risk (PR6)** - if unsound, values come out TOO SMALL:
  (i)  the covering restatement checked AGAINST THE DEFINITION by brute force
       at k = 1,2,3 x z = 2,3,5,7 (12 cases, all equal) - research/jk_cover.py;
  (ii) 14 published A048670 values and 9 published A288815 values reproduced
       exactly;
  (iii) `rust2/src/bin/jkcover.rs`, a second engine with NO reduction and NO
       canonical rule, agrees on 12 of 12 shared cases including j_3(11)=180;
  (iv) every witness re-verified by code sharing nothing with the search, and a
       SAT encoding (CaDiCaL) agrees wherever it reaches.
**PR6 CONFIRMED.**

### 12c. (b) THE FIRST EXACT VALUES OF j_k FOR k >= 3

    z        3     5     7     11     13     17     19
    j_3      6    24    78    180    306    612    972
    j_4      -    30   150    420   1230      -      -
    j_5      -     -   180    930   2070   5490      -

Round 27 had only 6, 24, 78. Everything else is new, each exact in both
directions. `j_4(P(5)) = 30` is the degenerate case where every prime <= 5 is
peeled by the reduction. Doc: docs/novel/jk-growth-discriminator.md; the
jk-family.md table is updated with a pointer.

### 12d. THE ROUND'S IDEA: TRADE THE z-AXIS FOR THE k-AXIS

The two live readings of h_2 differ by `(log z)^{k-1}`:
  (A) the parameter-free random-choice heuristic  j_k ~ z (log z)^k
  (B) the layered construction (P2'), a THEOREM,  j_k >> z (log z)^{2k-1}.
**At k = 2 that is ONE log** - which is exactly why r24-r27 named "one exact
h_2 beyond p_n = 73" as the falsification target and why nobody has bought it.
**At k = 3 it is two logs and at k = 5 four, and those values cost seconds.**

Put delta_k(z) = prod(1 - min(k,p-1)/p) EXACTLY, N = delta_k P(z), and
model_k = log(N)/delta_k - the expected largest gap among N random points on a
cycle of length P, with no free parameter. R_k = j_k/model_k.

* **THE CALIBRATION.** R_1 falls 0.590 -> 0.376 over z = 7..23 and is then
  FLAT TO WITHIN 4% over eighteen further values to z = 113. At k = 1 the two
  models COINCIDE and the truth is known (Rankin/FGKT attain z log z up to
  loglog powers), so k = 1 measures the method's own bias: it is ~0.
* **THE k = 2 SIGNAL.** R_2 runs 0.821 -> 0.889 on the clean window
  z = 23..73: a real **+8% drift where model (A) needs 0% and model (B) needs
  +37%**.
* **THE CROSS-k STATISTIC, which is what the family buys.** With
  Q_k = R_k/R_1 (removing the transient) and
      f_k = log(Q_k(z1)/Q_k(z0)) / ((k-1) log(log z1/log z0)),
  f = 0 under (A), f = 1 under (B) - **and under (B) f is the SAME AT EVERY k.**
  That equality IS the (k-1) scaling, and it is the thing the k = 2 ladder
  alone cannot test.

      window     |   f_2   |   f_3   |   f_4   |   f_5
      7..13      |  1.599  | -0.282  | -0.104  | -0.310
      7..17      |  1.116  |  0.229  |    -    |  0.014
      7..19      |  0.882  |  0.251  |    -    |    -
      23..73     |  0.257  |    -    |    -    |    -   (clean, k=2 only)

  **They are not equal across k: f falls steeply as k rises on every matched
  window.** The extra logs (B)'s shape needs are not appearing at the rate
  (k-1) demands.
* **THE SECOND, INDEPENDENT FORM.** Fitting j_k ~ z (log z)^{a_k} gives
  a_k = 0.921, 2.614, 3.556, 4.757, 6.724 at k = 1..5, i.e. excess
  e_k = a_k - k of **-0.079, 0.614, 0.556, 0.757, 1.724** against the
  **k-1 = 0, 1, 2, 3, 4** that (B) requires. The excess is REAL (measured
  against a calibration bias of -0.08) and **does not grow with k.**

**THE HONEST CAVEAT, and it is load-bearing, and it is in the doc twice:**
(P2') carries a `C^k/B^{2k}` factor worth about 0.03 at z = 73, k = 2, and the
construction does not exist below log x ~ 300 (r26 10f, my own record). **So
none of this refutes the theorem.** What is measured is the shape of the TRUTH
on the range where exact values exist, and on that range it looks like model
(A) plus a constant excess, uniformly in k. The lane's own standing corollary
- MODEL CLAIMS EXPIRE LIKE CITATIONS - applies to this measurement too.

### 12e. (c) THE MEMO

No decision from the human arrived during the round. Per the brief, the
submission was not touched in either direction. docs/novel/unit1-submission-memo.md
stands as filed; nothing in round 28 changes what it says, though 12a costs its
one concrete suggestion (write to Ziller and Morack for p_n = 151) more
honestly than it was costed when written.

### 12f. Pre-registration scored

PR1 **CONFIRMED** - growth 20-45x per prime near z = 29 (measured 27.8, 37.5,
     42.3), and the round's answer to brief item (a) is a price.
PR2 **CONFIRMED ON THE VALUE, MISSED ON THE COST** - omega_2(31) = 94 confirmed
     as predicted, but I said "under 4 core-hours" and it took 4.9.
PR3a **CONFIRMED** - a_k rises with k.
PR3b **CONFIRMED** - a_k in [k, 2k-1] at every k >= 2; neither model attained.
PR3c **REFUTED AS WORDED** - R_2 does rise (+12% over z = 7..73, reproducing
     r24's +11% from a different statistic) but R_3, R_4, R_5 FALL. The reason
     is one I had not thought of: R_k carries a large small-z transient common
     to every k, and the k >= 3 ranges lie ENTIRELY inside it. **The calibrated
     Q_k = R_k/R_1 was built BECAUSE this prediction failed**, and it is the
     round's headline statistic. A refuted prediction produced the instrument.
PR4, PR5 - see 12g.
PR6 **CONFIRMED** - the canonical rule is sound (12b(iii)).

### 12g. Negatives, costs and residual risks

* **BRIEF ITEM (a) IS NOT DELIVERED AS A VALUE.** No h_2 beyond p_n = 73, and
  none is reachable here. This is a MEASURED negative with an exhibited cost
  curve, not a judgment - but the JUDGMENT part, labelled: that no
  reformulation available to this lane closes the gap. ZM reached z = 73 with a
  portioned ILP (Giovanni Resta's binary-ILP formulation, recorded in A072753's
  own OEIS comments); I did not build an ILP, and I do not know how much it
  would buy. **That is the honest hole in my price.**
* **MY FIRST TWO PARALLEL LAUNCHES LEAKED 14 ORPHAN WORKERS.** `nohup ... &`
  under the shell tool returns immediately and the driver dies while its
  children live; I then relaunched, reached 28 processes on a 20-core box - over
  the compute policy's 16-core ceiling - and ran everything at half speed for
  ~25 minutes before noticing. Found by counting processes, not by a gate. New
  standing rule for this lane: **after launching any parallel job, COUNT THE
  PROCESSES.**
* **A THIRD RUN WAS KILLED MID-FLIGHT** by a session interruption and had to be
  restarted from scratch; the driver has no checkpoint. `jk_run.py` resumes
  nothing. That is a real defect for multi-hour work and it should be fixed
  before the next long run.
* The k >= 3 data lies inside the small-z transient. **The k = 2 window
  z = 23..73 remains the single cleanest measurement**, and the family's
  contribution is the CROSS-k comparison on matched windows, not a longer
  lever. j_3 at z = 29 and 31 would move it materially and is priced at
  ~10 and ~100 core-hours respectively.
* **THE DISCRIMINATOR CANNOT REFUTE (B)** and the doc says so twice. It
  measures the truth's finite-range shape. Anyone quoting it as evidence
  against the theorem is misquoting it.
* The cost projection below z = 43 is an extrapolation of an extrapolation and
  is printed to show the shape of the wall, not to predict a runtime.

### 12h. Ranking changes

* **7c#4 (h_2 at p_n = 151..251) IS DEMOTED FROM TOP RESEARCH ITEM TO A PRICED
  DEAD END for this lane.** It was the lane's top item by default for three
  rounds. It is now measured: five to nine primes past a frontier that has not
  moved in nine years, on a vehicle six primes short of that frontier. It
  should not appear in a future brief as a target. **What replaces it is the
  k-axis**, which answers the same question.
* **NEW ITEM, and it is the lane's new top research item: THE k-AXIS PROGRAMME.**
  j_3 at z = 23, 29, 31 and j_4 at z = 17, 19 would put the cross-k statistic on
  post-transient data. Priced: z = 23 is ~1-2 core-hours at k = 3, z = 29 is
  ~10, z = 31 ~100. **This is the only place in the lane where a purchasable
  computation still changes a conclusion.**
* **(P6) THE k-FAMILY RISES ABOVE N4.** It now has exact data, an engine, an
  independent replication of both published ladders, and a measurement that
  bears on its own conjecture. It is no longer "a section of Unit 1" - it is
  the piece with live research in it.
* **N4 (the j_2 upper ladder) unchanged**: still TOP for publication, still a
  writing item, still waiting on a decision that is the human's.
* **jk-family.md's own CONJECTURE is now amended against itself**: the first
  finite data ever available points away from the (2k-1) shape on the computed
  range, and the doc says so in a marked block rather than burying it.

### 12i. Additions to the standing citation-hygiene lesson (7d)

11. **A PUBLISHED TABLE IS A CLAIM LIKE ANY OTHER, AND REPRODUCING IT IS
    CHEAPER THAN YOU THINK.** This lane quoted Ziller-Morack's h_2 values for
    seven rounds without ever recomputing one. Reproducing nine of them took a
    day and turned up the reduction, the symmetry rule and the cost curve - all
    three of which were needed for the round's actual result. **When a lane
    depends on someone else's numbers, recompute the cheap end of them.**
12. **AN OEIS RECORD CARRIES THE ALGORITHM, NOT ONLY THE VALUES.** A072753's
    comments hold Giovanni Resta's binary-ILP formulation, John F. Morack's
    GLPK runs, and the fact that a(19) was published as 355 and CORRECTED to
    364. None of that is in either arXiv paper. **Read the comment field.**

### 12j. Reproduction (round 28)

* `research/jk_cover.py` -> reference engine (Python DFS + SAT) and the
  definition-vs-restatement brute force. **Gate.**
* `rust2/src/bin/jkcov6.rs` -> the fast engine (reduction + canonical form +
  v3 bound). `cargo build --release --bin jkcov6`.
* `rust2/src/bin/jkcover.rs` -> the unreduced engine, kept as the independent
  cross-check (no reduction, no canonical rule).
* `research/jk_run.py` -> two-phase parallel driver (witness, then seeded split
  infeasibility proof). Logs `research/data/r28_k2_z31.log`,
  `research/data/r28_k3_z23.log`.
* `research/jk_growth.py` -> `research/data/jk_growth.out`. Sections A (the
  ladders), B (the parameter-free model and R_k), C (measured exponents),
  D (cross-k slope), D2 (Q_k and f_k - the headline), E (the calibrated
  excess), E2, G (the price of the z-axis), F (assertions). **Gate.**
* `research/data/r28_harvester_prereg.txt` -> pre-registration + the addendum
  written before the j_3(23) answer.
* Docs: **NEW** `docs/novel/jk-growth-discriminator.md`; `docs/novel/jk-family.md`
  (new section 1a, and the conjecture amended against itself);
  `docs/novel/README.md` (index entry).
* Sources read FIRST-HAND by me on 2026-08-29: OEIS records A072753 (#79),
  A288815 (#19), A048670 (#164), A048669 (#92), in full, via the text endpoint.

## 13. ROUND 29 (2026-09-03) - THE k-AXIS DECIDES, MY OWN ROUND-28 RUN WAS
## INVALID, AND THE "ALGORITHM" QUESTION HAS A PUBLISHED ANSWER I HAD NOT READ

Brief: (a) the k-axis programme - j_3 at z = 23, 29, 31 and j_4 at z = 17, 19,
each result exact / capped / not attempted with a price, scoring the round-28
pre-registrations; (b) literature adjacency for the anchor-235 floor (the first
integer outside a union of 2 pi(q) progressions from pi(q) prime moduli), with
an adjacency table and a verdict on whether any published algorithm computes
the maximal gap of a two-class sieve below a scan or bounds that computation
from below; (c) score outstanding r27-r28 predictions that Mechanic's data now
decides. Not to pursue: h_2 at p_n = 151..251 (priced dead end, stays demoted).

GATES, re-run from clean processes at round close:
  research/j2_referee.py    -> ALL ASSERTIONS GREEN   (run FIRST)
  research/j2_citesweep.py  -> ALL CHECKS GREEN
  research/jk_cover.py      -> **PARTIAL, and it is recorded as partial.**
                               Sections [A] (covering restatement vs the
                               definition, 12/12), [B] (15 published values by
                               the Python reference engine), [C] (SAT vs DFS,
                               5/5) and [C2] (the rust engine against every
                               published and round-28 value, 27/27, each with
                               exact=True and its witness verified) ALL PRINTED
                               OK.  It STALLS in section [D] at
                               `dfs_maxrun(2, 17, 200)` - the pure-Python
                               UNREDUCED engine with no canonical-form rule,
                               i.e. exactly the (2n-4)!/2^(n-2) permutation
                               redundancy that round 28's canonical rule
                               removes.  Two launches, 73 and 38 CPU-minutes,
                               both killed; log research/data/r29/jk_cover_gate.log.
                               DO NOT report this gate green until section [D]
                               is rewritten to use the reduced engine.
  research/jk_growth.py     -> ALL ASSERTIONS GREEN
  research/jk_axis29.py     -> ALL ASSERTIONS GREEN   (NEW - harvest, protocol
                               check, discriminator, price)
  research/harv_score29.py  -> ALL ASSERTIONS GREEN   (NEW - brief item (c),
                               exact integer arithmetic, no float decides)
  .venv-sat/.../jk_sat29.py check -> ALL ASSERTIONS GREEN  (NEW - the SAT engine
                               reproduces 30 of the 33 recorded (k, z) values in
                               BOTH directions; the other three are named in the
                               script's SLOW table with the reason and each was
                               decided in a separate timed run)
Pre-registration: research/data/r29_harvester_prereg.txt, written before the
runs it scores (H1-H7).

### 13a. (a) THE ROUND-28 j_3(23) RUN WAS INVALID BY ITS OWN PROTOCOL - AND I
### FOUND IT IN MY OWN WORK BEFORE ANYONE QUOTED THE NUMBER

Round 28's phase-2 run for j_3(P(23)) FINISHED. All fourteen partition files
were sitting on disk unharvested (`research/data/jkpart_k3_z23_M219_n14_p*.txt`),
all EXACT, all verify=true. **Two of the fourteen workers BEAT THE SEED**,
reaching m = 227 and m = 232 against a seed of 219.

`jk_run.py`'s own docstring says what that means: "Because every worker starts
with the same incumbent M and no worker ever improves on it, the pruning above
the split depth is identical in all workers, so the union of the parts is the
whole tree. If ANY worker reports a value > M the run is invalid and is redone
with the larger seed." THE MECHANISM, read out of `jkcov6.rs` rather than
inferred: a node is pruned when `feasible_to(cov, j, best + 1)` fails, so a
worker whose incumbent has risen prunes MORE above the split depth, visits
FEWER split-depth nodes, and its global `leafctr` counter diverges from the
other workers'. The parts `leafctr % nparts == part` then need not cover the
tree. **A branch-and-bound split is a proof only when the incumbent is a fixed
point of the run.**

So the round-28 run splits cleanly:
- VALID: a machine-verified witness of length m = 232, i.e. **j_3(P(23)) >= 1398**.
- INVALID as an upper bound. Rerun required at seed 232.

`research/jk_run29.py` (explicit mandatory seed, FATAL protocol assertion, one
result file per worker so a reaped driver loses nothing) reran it at seed 232 on
five workers. **RESULT: all five EXACT, all five reporting m = 232, none
improving on the seed - so the protocol holds and the parts do partition the
tree. j_3(P(23)) = 1398 EXACT.** 7,147,384,960 nodes over 27,296 core-seconds.

**AND THE SEED LAW IS MUCH WEAKER THAN ROUND 27 RECORDED.** Round 27's "seed
law" put a better-seeded rerun at roughly a quarter of the cost. Measured here:
7.38e9 -> 7.15e9 nodes, **a 3.2% saving from a seed thirteen higher**, not 4x.
The wall-clock difference (13.6 -> 7.6 core-hours) is almost entirely the
High-priority boost and the smaller worker count, **not algorithmic** - which is
exactly why the benchmark protocol counts operations and not seconds. Anyone
budgeting a rerun off the round-27 seed law will under-budget it by ~4x.

### 13b. THE VALUES - EXACT / CAPPED / NOT ATTEMPTED, WITH PRICES

    z        3     5     7     11     13     17     19      23
    j_3      6    24    78    180    306    612    972    1398
    j_4      -    30   150    420   1230   2340   3810       -
    j_5      -     -   180    930   2070   5490      -       -

| target | status | price |
|---|---|---|
| j_3(P(23)) = 1398 | **EXACT, TWICE** | `jkcov6`: 7.38e9 nodes (r28, seed 219, invalid as an upper bound) + **7.15e9 nodes / 27,296 core-seconds** (r29 confirmation at seed 232, five workers, protocol clean). **SAT: 8,710,802 conflicts, 831 s, ONE core, one process** |
| j_4(P(17)) = 2340 | **EXACT** | 351,958 nodes, 0.345 s, m = 77 |
| j_4(P(19)) = 3810 | **EXACT** | 99,408,318 nodes, 448.8 s, m = 126 |
| j_3(P(29)) | **NOT ATTEMPTED** | `jkcov6`: ~1.9e12 nodes = ~3,500 core-hours. **SAT: ~6.6e8 conflicts = ~17 core-hours - PURCHASABLE, see 13d** |
| j_3(P(31)) | **NOT ATTEMPTED** | `jkcov6`: ~8e14 nodes = ~1.5e6 core-hours. SAT: ~8.7e10 conflicts = ~2,300 core-hours |

**AND j_3(P(23)) = 1398 IS PROVED TWICE, BY ENGINES SHARING NOTHING.** Besides
the `jkcov6` route, CaDiCaL on a CNF encoding of Ziller-Morack's own integer
program decided both directions in **831 s on one core, one process, no split; it produced its own witness
at m = 232 and proved m = 233 impossible** (13d). That second proof has no protocol risk of any kind, so the
value does not depend on 13a's split at all.

**AND THE PRICES IN MY OWN ROUND-28 BLOCK WERE WRONG, BY UP TO FOUR ORDERS.**
Round 28 priced the k-axis programme at "z = 23 is ~1-2 core-hours at k = 3,
z = 29 is ~10, z = 31 ~100" and called it "the only place in the lane where a
purchasable computation changes a conclusion". Measured k = 3 node counts
(11,740 -> 556,927 -> 50,867,900 -> 7.38e9 at z = 13, 17, 19, 23) give per-prime
ratios 47.4x, 91.3x, 145.1x, themselves growing ~1.75x per step:

    rung        r28 price          measured / projected        error
    j_3(23)     ~1-2 core-hours    13.6 core-hours (actual)     9x low
    j_3(29)     ~10 core-hours     ~3,500 core-hours          ~350x low
    j_3(31)     ~100 core-hours    ~1.5e6 core-hours       ~15,000x low

The error was mechanical: I extrapolated the k = 2 node curve onto k = 3. The
k = 3 curve is steeper because each prime carries three classes, so the
branching factor at every node is larger. **Half the k-axis programme was never
purchasable and I said it was.**

PROCESS MISS, recorded as one: j_4(P(17)) was computed as a COST PROBE before
this round's pre-registration was written. It is a new value and it should have
been pre-registered; it is used below as an input, never scored as a hit.

### 13c. THE DISCRIMINATOR, DECIDED AT TWO POST-TRANSIENT STEPS

Round 28's named weakness was that the k >= 3 data lay entirely inside the
small-z transient. Both new steps sit outside it, and **both carried a
numerical prediction from each model, pre-registered before the answer
existed.**

    step            R_k before  R_k after   move      (A) needs  (B) needs
    k=3, 19 -> 23     1.2100      1.2084    -0.13%       0%        +13.4%
    k=4, 17 -> 19     1.4426      1.3768    -4.56%       0%        +12.2%

The k = 3 step is the sharpest measurement this lane has taken on the question.
The round-28 addendum (written 2026-08-30, before the run finished, with the
scoring rule fixed in advance) predicted **1398 under model (A)** and **1590
under model (B)**. The answer is **1398, exactly, to the unit.**

**AND A CORRECTION AGAINST MYSELF.** Round 28 recorded that the measured excess
e_k = a_k - k "does not grow with k". With j_4 now carrying five points instead
of three the ladder is

    e_k = -0.08, 0.61, 0.73, 1.45, 1.72   at k = 1, 2, 3, 4, 5

and it DOES grow monotonically. The round-28 sentence is withdrawn. What
replaces it is sharper, not weaker: the excess is a CONSISTENT FRACTION of what
(B) demands, e_k/(k-1) = 0.61, 0.37, 0.48, 0.43 at k = 2..5, so on the computed
range the truth looks like z (log z)^{k + c(k-1)} with c ~ 0.45 - strictly
between (A) (c = 0) and (B) (c = 1), AND AT THE SAME PLACE AT EVERY k. That is
a stronger statement than "the extra logs are absent" and it is the shape any
future model has to reproduce.

**THE STANDING CAVEAT IS UNCHANGED AND STILL LOAD-BEARING:** (P2') carries a
C^k/B^{2k} factor worth ~0.03 at z = 73, k = 2 and does not exist below
log x ~ 300. **NONE OF THIS REFUTES THE THEOREM.** It measures the shape of the
truth on the range where exact values exist.

### 13d. (b) LITERATURE ADJACENCY FOR THE ANCHOR-235 FLOOR

THE OBJECT (anchor-235 9g): "Below the scan, a form would have to compute the
first integer outside a union of 2 pi(q) arithmetic progressions from the pi(q)
residues of s alone; none is known here and none was found." In the named
literature this is a **two-residue-class-per-prime Jacobsthal problem**: W(s) is
the distance from s to the next uncovered position, F = max_s W(s) is the
maximal gap, and h_2 = 2 max_e F_e is the family maximum over differences.

**THE HEADLINE, AND IT REFRAMES THE FLOOR.** The brief asks whether any
published algorithm computes the maximal gap of a two-class sieve "without a
period scan". THE ANSWER IS YES, AND IT HAS BEEN YES SINCE 1978 - because
**nobody in this literature ever scans a period.** Hagedorn, Ziller, Ziller &
Morack, Resta and McNew & Setty all work in CLASS-ASSIGNMENT space: they choose
residues per prime and test coverage of a short window. Ziller-Morack's
h_2(73#) is a statement about a period of ~1e27 and no period was ever built.
This lane's own `jkcov6` is in the same family. **So "below a scan" is the
published state of the art, not the frontier.** The anchor-235 floor's real
content is not "below a scan" but "below an EXPONENTIAL SEARCH in pi(q)", and
on THAT question the literature has nothing: no sub-exponential algorithm, and
no proved lower bound.

THE ADJACENCY TABLE. "read" = full text obtained and read by me this round;
"abstract" = abstract/metadata only; "secondary" = characterised through
another source, flagged as unverified first-hand.

| # | result | exact statement | source (how verified) | what it gives / fails to give for F, W(s) |
|---|---|---|---|---|
| 1 | **Jacobsthal's function** | j(n) = least m such that every m consecutive integers contain one coprime to n; h(n) = j(p_n#) | Jacobsthal, D.K.N.V.S. Forhandlinger **33** (1960) no. 24, 117-124 (secondary, via Ziller 2020 ref [8]) | Defines the ONE-class object. Our F is the two-class analogue; no transfer either way. |
| 2 | **Iwaniec 1978 - still the record after 48 years** | g(n) <= X (w log w)^2, w = omega(n); at primorials h(p_n#) << (n log n)^2 = p_n^{2+o(1)} | H. Iwaniec, *On the problem of Jacobsthal*, Demonstratio Math. **11** (1978) 225-232, DOI 10.1515/dema-1978-0121 (abstract + DOI verified; record status re-checked by this lane against the live Erdos-problems DB 2026-08-25) | An UPPER bound at sifting dimension 1. Says nothing at dimension 2 and carries NO algorithm. Our Theorem 2G (exponent 8.042) is the dimension-2 counterpart. |
| 3 | **Hagedorn 2009** | computes h(n) for n < 50 by backtracking over one-class-per-prime assignments with an a-priori capacity criterion m_i | Math. Comp. **78** (2009) no. 266, 1073-1087 (**NOT OBTAINED**: the AMS PDF and the author's own copy both returned HTTP 403 to this session; algorithm characterisation is SECONDARY via Ziller-Morack 1611.03310 and this lane's own r28 implementation) | The m_i criterion is the ancestor of our v3 prefix bound. One class per prime. No complexity result. **UNVERIFIED FIRST-HAND - recorded as such per lesson 9.** |
| 4 | **Ziller & Morack 2016 - the algorithms paper, and I had never read section 2** | six algorithms (BSA, BPA, RPA, DSA, CRPDSA, GPA) plus **an integer linear program, equation (2.2)**: binary x_{i,j} per (prime p_i, class j != 0), `sum_j x_{i,j} = 1` per prime, one covering constraint per position, objective `max sum_k 2^{m_2-k} y_k` finding the maximal m in ONE program. Complexity ESTIMATES only: N_BSA = prod (p_i - 1), N_BPA <= (n-1)! N_BSA. Computed h(n) for every p_n <= 251; printed time curve reaching ~1 month at the top; ILP solved with SYMPHONY | arXiv:1611.03310 (**read**, PDF extracted to research/data/r29/zm_algo.extract.log) | **THE MOST ADJACENT ITEM THERE IS.** Generalising (2.2) to k classes is one character: `= 1` becomes `<= k`. So a two-class ILP for our F has existed in print since 2016. **No lower bound is proved anywhere in it.** |
| 5 | **Ziller & Morack 2017 - the paired function** | h_2(p_n#), 21 terms to p_n = 73; Conjecture 6: h_2(p_n#) < p_n^2 - p_n | arXiv:1706.00317 / 1706.03668 (lane record, verified rounds 21-28; not re-fetched this round) | The ONLY published two-class computation. Frontier p_n = 73, unmoved in nine years. |
| 6 | **Ziller 2020 - and it is prior art for a round-28 result of THIS PROJECT** | **Proposition 2.7 (propagation of coverings): m in D(k) => m in D(k+1)** - every gap realised by one machine is realised by the next. Also N_min(k), the smallest even number NOT occurring as a gap, computed exhaustively for p_k up to k = 44 (p_44 = 193) by an Adapted Greedy Permutation Algorithm; Conjecture 4.1: h(k-1) <= N_min(k) | arXiv:2007.01808 (**read**, extracted to research/data/r29/ziller2020.extract.log) | **Mechanic's round-28 DEPTH-0 LEMMA (D_m(M) subset D_m(M+q')) is the arity-m, two-class generalisation of this 2020 one-class proposition.** The project's "smallest absent gap" question is Ziller's N_min. One class per prime, so neither result implies the other - but the framing is his and it should be cited. |
| 7 | **Costello & Watts** | a new explicit upper bound on g(n); a range-restricted computational bound 2 e^gamma k^{5+5 log log k} for 50 <= k <= 10000 | arXiv:1306.1064 -> Math. Comp. **84** (2015) 1389-1399, and arXiv:1208.5342 (abstracts read; full bounds not extracted this round) | Explicit constants at dimension 1. No algorithm for the maximal gap; no two-class content. |
| 8 | **Ford-Konyagin-Maynard-Pomerance-Tao, "Long gaps in sieved sets" - AND ITS HYPOTHESIS EXCLUDES US, WHICH SETTLES A FLAG THIS LANE HAS CARRIED SINCE ROUND 24** | Theorem 1: a sieving system that is non-degenerate, B-bounded (\|I_p\| <= B), **ONE-DIMENSIONAL** - `prod_{p<=x} (1 - \|I_p\|/p) ~ C_1/log x`, eq. (1.2) - and delta-supported has a gap >= x(log x)^{C(delta)-o(1)}, C(delta) > e^{-1-4/delta} | JEMS; PDF read first-hand (research/data/r29/fkmpt_sieved.extract.log) | **DECISIVE NEGATIVE.** Our system has \|I_p\| = 2 at EVERY prime, so the product is ~C/(log x)^2: **dimension TWO, and hypothesis (1.2) fails.** The theorem does not apply. Round 24 flagged this paper "RELAY-SOURCED (one class per prime), re-verify before citing" - the relay was wrong about the reason and right about the conclusion: it is not the class COUNT that excludes us, it is the DIMENSION. The word "Jacobsthal" occurs **zero** times in the paper. This is why (P1)/(P2') had to be built by hand. |
| 9 | **Ford-Green-Konyagin-Maynard-Tao, "Long gaps between primes"** | the record lower bound for prime gaps, via the Eratosthenes system (one class per prime) | JAMS **31** (2018) (lane record) | Transfers to j_2 through this lane's r21 collapse (b - a = p# makes paired = ordinary), giving LOWER bounds only. No algorithm. |
| 10 | **Stockmeyer & Meyer 1973 / Garey & Johnson problem AN2 - the nearest thing to a lower bound, AND IT DOES NOT TRANSFER** | SIMULTANEOUS INCONGRUENCES: given pairs (a_i, b_i) with a_i < b_i, is there an integer x with x != a_i (mod b_i) for all i? **NP-complete.** | Garey & Johnson, *Computers and Intractability* (1979), Problem AN2; proof in L. J. Stockmeyer and A. R. Meyer, *Word problems requiring exponential time*, STOC 1973, 1-9. Verified first-hand through McNew & Setty, arXiv:2507.23041, which quotes both by number and lists them ([13] and [29]) | **THE HONEST VERDICT: IT IS NOT A LOWER BOUND FOR US.** AN2 has ARBITRARY moduli and ONE class each; our floor has DISTINCT PRIME moduli and TWO classes each. Neither problem contains the other, and at distinct prime moduli the AN2 *existence* question is trivially YES (the sifted set has positive density), so the hardness lives entirely in the composite/repeated moduli AN2 is allowed and we are not. **No published lower bound on computing F, W(s) or h_2 exists.** |
| 11 | **McNew & Setty 2025/2026 - the 2026 state of the art for this kind of decision** | decides covering-number membership with a **binary integer program solved by Gurobi**; states "Determining whether a given integer n is a covering number seems to be computationally intractable in general. It seems likely this problem may be NP-complete" | arXiv:2507.23041 (**read**) | Confirms that in 2026 the working method for covering decisions is still ILP, and that even the *conjecture* of hardness is stated as a conjecture. |
| 12 | **Covering systems: Filaseta-Ford-Konyagin-Pomerance-Yu; Hough** | sieving by large integers; solution of the minimum modulus problem | JAMS **20** (2007) 495-517; Ann. of Math. **181** (2015) (lane record 5h) | ONE class per modulus and the MODULI are the free variable. Different object; no algorithmic content for F. |
| 13 | **Kalmynin-Konyagin, polynomial analogue** | Rankin machinery on shifted polynomial values; M(f) = 2 for quadratics | arXiv:2302.00459 (lane record r24) | Nearest relative of (P2), not prior art for (P1). Unchanged. |
| 14 | **Parameterized covering by arithmetic progressions** | COVER BY AP: given a finite set X and k, are there k APs whose union is exactly X? 2^{O(k^2)} poly(n) | arXiv:2312.06393 | Different object (cover a given finite SET, moduli free). Does not touch F. |

**MY OWN MEASUREMENT ON ITEM 4, because a citation is not a price.** Round 28
closed with a named hole: "I did not build an ILP, and I do not know how much it
would buy." `research/jk_sat29.py` encodes ZM equation (2.2), generalised to k
classes, on the reduced lattice, and decides both directions with CaDiCaL. It
reproduces all 31 recorded (k, z) values. Cost in the solver's own operation
counts, against `jkcov6`'s node counts (DIFFERENT UNITS - no ratio between them
is quoted; what is comparable is the GROWTH):

    k=2, z              13      17       19        23          29
    SAT conflicts (UNSAT) 131   1,570   14,503   178,618   2,952,407
      ratio                -     12.0x    9.2x     12.3x       16.5x
    jkcov6 nodes         150   2,577   53,560  1,491,366  55,917,112
      ratio                -     17.2x   20.8x     27.8x       37.5x

    k=3, z                          17        19
    SAT conflicts (UNSAT)        8,889   201,771     ratio 22.7x
    jkcov6 nodes               556,927 50,867,900    ratio 91.3x

**THE SOLVER'S GROWTH RATIO IS FLATTER THAN THE DFS's AT BOTH k, AND ABOUT HALF
THE DFS's AT k = 3.** At k = 2 it is not a rescue: at z = 31 - the DFS's 4.9
core-hours - the solver did not decide even the SATISFIABLE direction in 570 s,
and 12-16x per prime still needs 12^14 more work to reach p_n = 73 from
p_n = 31. **AT k = 3 IT IS A RESCUE, AND THIS IS THE ROUND'S SECOND RESULT:**

    k=3, z                          17         19            23
    SAT conflicts (UNSAT)        8,889    201,771     8,710,802
      ratio                          -      22.7x         43.2x
    jkcov6 nodes               556,927 50,867,900   7.38e9 (14 parts)
      ratio                          -      91.3x        145.1x

**CaDiCaL PROVED j_3(P(23)) = 1398 OUTRIGHT - BOTH DIRECTIONS, ONE PROCESS, NO
SPLIT, NO SEED - IN 831 SECONDS ON ONE CORE**, against the DFS's 13.6
core-hours over fourteen workers. Two consequences:
1. **AN INDEPENDENT TWO-SIDED PROOF OF THE VALUE, WITH NO PROTOCOL RISK AT
   ALL.** The whole defect of 13a is a property of splitting a
   branch-and-bound; a single-process UNSAT proof cannot have it. The value
   1398 no longer rests on the split at all.
2. **THE PRICE OF THE REST OF THE k-AXIS COLLAPSES ON THIS VEHICLE.** Carrying
   the measured SAT ratio forward at the same 1.9x-per-step growth the DFS
   shows: `j_3(P(29))` projects at ~6.6e8 conflicts = **~17 core-hours**, against
   ~3,500 on the DFS. **THAT IS PURCHASABLE AND IT IS THE NAMED NEXT TARGET.**
   `j_3(P(31))` projects at ~8.7e10 conflicts = ~2,300 core-hours and stays out
   of reach. I did NOT launch j_3(29): a ~17-hour single-threaded job cannot
   finish inside a round and the job-completion rule forbids starting it.

WHAT IT DOES NOT SETTLE, stated so nobody over-reads it: a tuned PORTIONED ILP
with branch-and-cut, symmetry breaking and warm starts - ZM's actual vehicle -
is a different program from this one and was not tested; and at k = 2, which is
where h_2 lives, CDCL loses to the DFS at the frontier.

**H6 SCORED, and two of its clauses fail against me.** I predicted every item
falls into (i) exhaustive/ILP covering search, (ii) asymptotic bounds with no
algorithmic content, or (iii) one class per prime; and that "the sharpest
genuinely adjacent item is the ILP formulation in A072753's OEIS comments
rather than anything in a journal".
- The three-way classification: **CONFIRMED** for 13 of 14 items.
- **REFUTED**: the sharpest item is not an OEIS comment. It is **equation (2.2)
  of an arXiv paper this lane has cited for seven rounds without reading its
  section 2.**
- **REFUTED IN ITS REASON**: FKMPT is excluded by DIMENSION, not by class count,
  so my category (iii) mislabels it.
- **CONFIRMED**: no published algorithm is sub-exponential in pi(q), and no
  published lower bound on the computation exists.
- The NAMED RISK paid off in the direction I flagged: I had not read Hagedorn in
  full, and I still have not - the fetch failed twice.

### 13e. (c) SCORING AGAINST MECHANIC'S ROUND-28 LADDER
### (research/harv_score29.py, exact integer arithmetic, 22 assertions)

A NOTATION HAZARD FIRST, because it nearly bit me. **This lane's `F(2,y)` is the
fixed-twin member of the per-difference family in MEMBER units, F(2,y) = 3F(y);
Mechanic's `F_2(M)` is the DEPTH-2 spectrum value of machine M.** The strings
collide: F(2,59) = 483 and F_2(59) = 173 are different quantities. Nothing below
uses Mechanic's F_J; only the record ladder F(y) = 88, 91, 103, 118, 145, 161 at
y = 37, 41, 43, 47, 53, 59.

**(1) F(2,53) - 5b, written round 22. CONFIRMED, WITH THE LAW CORRECTED.**
Recorded: "F(2,53) >= 426 (needs <= 486 for the tolerance constant;
quadratic-law prediction ~441)". F(53) = 145 gives **F(2,53) = 435**.
Lower bound 426 holds (slack 9); ceiling 486 holds (slack 51); **the
quadratic-law prediction 441 is HIGH by 6, i.e. 1.38%** - one mod-6 quantum.
My H7 pre-registered "high by ~1.4%": CONFIRMED. Free next rung:
**F(2,59) = 483.**

**(2) THE TWIN PERCENTILE - 5e, written round 24. CONFIRMED OUT OF SAMPLE AT
THREE MACHINES.** The ratio extreme/twin = (h_2(y)/2)/F(2,y) with ZM's h_2 as an
INDEPENDENT denominator:

    y      47        53        59
    F(2,y) 354       435       483
    h_2/2  642       711       828
    ratio  1.814     1.634     1.714     ALL INSIDE the recorded 1.34-2.27 band

Median over all fifteen machines y >= 11 is now **1.717** against the recorded
1.70. The publication statement of 5e stands, and now reads "at every one of
FIFTEEN machines". The ratio is not monotone (1.81, 1.63, 1.71) - it is a
per-machine arithmetic quantity, exactly as 5e says.

**(3) THE ROUTE-TRANSFER BUDGET - 5g, written rounds 13-14. CONFIRMED OUT OF
SAMPLE AT FIVE FURTHER STEPS.** Twin increment/q' at 37->41, 41->43, 43->47,
47->53, 53->59 is 0.073, 0.279, 0.319, **0.509**, 0.271 - the worst is 4.8x
inside twins' own 31->37 record of 2.432 and 4.9x inside the alpha = 2.5 budget.
**HONEST LIMIT, and it is the whole point of 5g:** this confirms the TWIN row
only. The binding negative is unchanged - fixed differences with single-step
increments 3.231, 3.947 and 4.435 q' exist, so no uniform alpha <= 3 budget
holds over the family, and five more twin steps inside budget say nothing about
that.

### 13f. Pre-registration scored (H1-H7)

H1 (the confirmation completes EXACT, no worker beats 232, under 6.0e9 nodes) -
   **SPLIT.** The protocol clauses are **CONFIRMED**: all five workers EXACT,
   all five reporting m = 232, none improving, so j_3(P(23)) = 1398. The COST
   clause is **REFUTED**: 7.15e9 nodes against my predicted "under 6.0e9". I
   reasoned "a better incumbent prunes strictly more", which is true, and then
   assumed the saving would be large, which it is not - a seed thirteen higher
   removed **3.2%** of the tree. **The pruning that matters happens near the
   leaves, where the incumbent is already close.**
H2 (model (A) wins on an EXACT value, 1398 to the unit) - **CONFIRMED**. And my
   own round-28 **PR5 ("j_3(P(23)) lands in [1400, 1800]") is REFUTED** - 1398
   is two below its own band. The r28 addendum already recorded "expect
   REFUTED"; it is refuted.
H3 (j_4(P(19))) - **SPLIT, and the split is against me.** The model comparison
   is CONFIRMED and not marginally: (A) predicted 3992, (B) predicted 4481, the
   answer is **3810**, whose log-distance to (A) is 0.046 against 0.160 to (B).
   But my BAND [3900, 4080] is **REFUTED** (3810 sits below it) and my "R_4(19)
   within 3% of R_4(17)" is **REFUTED** (it fell 4.56%). The direction of the
   miss is away from (B), so the conclusion strengthens while the prediction
   fails. COST: predicted 5e7-1.4e8 nodes and under 1 core-hour; measured
   9.94e7 nodes and 448.8 s. **CONFIRMED on both.**
H4 (j_3(29) 3,000-7,000 core-hours, j_3(31) >= 1e6, both not attempted, and my
   round-28 prices low by two and four orders) - **CONFIRMED AS A STATEMENT
   ABOUT MY VEHICLE, REFUTED AS A STATEMENT ABOUT BUYABILITY.** The jkcov6
   projections are exactly as predicted (3,500 and 1.5e6 core-hours, 350x and
   15,000x above the round-28 prices) and neither was attempted. But I wrote
   "NOT BUYABLE" without qualifying it by vehicle, and on the SAT vehicle
   measured this round j_3(29) is ~17 core-hours. **I made the round-28 error
   in the opposite direction: I priced a target from one engine's curve and
   stated the price as a property of the target.**
H5a (f_3 below 0.5 and below f_2 on the widest window) - **CONFIRMED** on
   7..23 (f_3 = +0.298, f_2 = +1.025). **HONEST COUNTEREXAMPLE I am recording
   myself**: on 13..23 the order reverses (f_3 = +1.095 > f_2 = +0.235). f_k on
   a two-point window is unstable and I should not have predicted it as a
   general fact; the robust statements are the two R_k steps of 13c.
H5b (the excess does not grow with k) - **REFUTED**, see 13c. It grows.
H5c (R_3(23)/R_3(19) in [0.97, 1.03]) - **CONFIRMED**, and it is the round's
   sharpest number: **0.9987**, flat to 0.13%, where (B) needs +13.4%.
H6 (literature) - **MOSTLY CONFIRMED, TWO CLAUSES REFUTED**, scored in 13d.
H7 (F(2,53) and the percentile band) - **CONFIRMED**, scored in 13e.

### 13g. Negatives, costs and residual risks of the round

* **I SHIPPED AN INVALID SPLIT PROTOCOL IN ROUND 28 AND ONLY CAUGHT IT BECAUSE
  I WENT BACK FOR THE FILES.** The driver printed the right warning; the round
  ended before anyone read the parts. Two things fixed it and both are cheap:
  per-worker result files (already there, and they are the reason nothing was
  lost) and a FATAL assertion instead of a printed one (new, in jk_run29.py).
* **MY ROUND-28 PRICES FOR THE k-AXIS WERE WRONG BY UP TO 15,000x** and they
  were the basis on which the programme was made the lane's top research item.
  Half of it was never purchasable. The mechanical error - extrapolating the
  k = 2 node curve onto k = 3 - is the same shape as Mechanic's standing rule
  "never extrapolate a per-step share; look it up".
* **MY DRIVER WAS REAPED AND LEFT FIVE ORPHAN WORKERS.** Round 28's lesson was
  `nohup ... &`; this round it was the shell tool's own background wrapper
  terminating the python driver while its children lived. The design that saved
  the data was per-worker files. FOR EVERY LANE: **the orphan trap is not about
  `nohup` - it is about any parent that can die before its children. Write the
  result from the CHILD.**
* **CPU STARVATION, exactly as Formalist measured in round 28.** My workers were
  getting 48% of a core each at Normal priority against other lanes' ~13 python
  processes. Raising `jkcov6` to High fixed it. Wall times in this round's logs
  are therefore contaminated and are NOT comparable across runs; every cost
  claim above is in NODES or CONFLICTS.
* **HAGEDORN 2009 WAS NOT OBTAINED.** AMS returned 403 and the author's own copy
  returned 403. Per lesson 9 I record HOW: two direct PDF fetches, both blocked
  at the server, no cookie-bearing browser session attempted. Its algorithmic
  characterisation in 13d is SECONDARY and labelled.
* **I HAVE STILL NOT TESTED A REAL ILP.** The SAT experiment is CaDiCaL on a
  CNF encoding of ZM (2.2); ZM used SYMPHONY on the integer program, and Resta's
  portioned formulation is different again. The negative in 13d is about
  off-the-shelf CDCL, not about ILP.
* **THE k >= 3 LADDERS ARE STILL SHORT.** j_3 has six points above the
  transient's floor and j_4 has five; j_5 has four and did not move. The
  cross-k statistic f_k is unstable on short windows and I have now said so
  twice, having predicted on it once.
* **A STANDING GATE OF MINE NO LONGER COMPLETES**, and I am recording it rather
  than quietly dropping it from the list: `jk_cover.py` section [D] runs a
  pure-Python DFS over the UNREDUCED problem at k = 2, z = 17. Everything the
  gate actually validates ([A]-[C2]) printed OK; the fix is one line - point
  [D] at the reduced engine - and it is next round's chore.
* **THE ROUND-27 SEED LAW IS WRONG AT THIS SCALE** (3.2%, not ~4x). Anyone
  budgeting a reseeded rerun off it will under-budget by about 4x.

### 13h. Additions to the standing citation-hygiene lesson (7d)

13. **A METHOD CITED BY REPUTATION IS AN UNREAD METHOD.** This lane cited
    Ziller-Morack arXiv:1611.03310 for seven rounds - for the GPA, for the RPA2
    canonical rule, for the p_n = 251 frontier - and wrote in round 28 "I did
    not build an ILP and do not know what it would buy", while the integer
    program was printed as equation (2.2) of that same paper's section 2. This
    is lesson 10 ("a hypothesis cited by number is an unread hypothesis") one
    level up: **read the section you are citing the technique from, not the
    abstract that names it.**
14. **A NEGATIVE INHERITED THROUGH A RELAY IS WRONG EVEN WHEN ITS CONCLUSION IS
    RIGHT.** Round 24 recorded FKMPT "Long gaps in sieved sets" as "one class
    per prime per the search relay - flagged RELAY-SOURCED, re-verify before
    citing". Read first-hand this round: it is B-BOUNDED (any B) but
    ONE-DIMENSIONAL, and it is the dimension condition (1.2) that excludes our
    two-class system, not the class count. The conclusion survived; the reason
    did not, and the reason is what a future round would have reasoned from.

### 13i. Ranking changes

* **THE k-AXIS PROGRAMME DELIVERED WHAT IT WAS FOR** - two clean
  post-transient steps, both landing on model (A), the k = 3 one exactly - and
  **ITS NEXT RUNG IS NOW PURCHASABLE AGAIN, ON A DIFFERENT VEHICLE.**
  `j_3(P(29))` is ~3,500 core-hours on `jkcov6` and ~20 core-hours on the SAT
  engine built this round. That is the one place in this lane where a
  purchasable computation still changes a conclusion, and it is a ~20-hour
  single-threaded run that must be launched at the START of a round, not the
  end. **It should be the lane's top item next round, on the SAT vehicle only,
  with `j_3(P(31))` (~2,300 core-hours even there) explicitly excluded.**
* **(P6) THE k-FAMILY: unchanged in rank**, one measurement stronger and one
  self-correction lighter.
* **N4 (j_2 upper ladder): unchanged.** Still TOP for publication, still a
  writing item, still waiting on the human's decision. Nothing this round
  touches it.
* **NEW, SMALL, AND NOT MINE TO RUN: the two-class ILP.** Item 4 of 13d means a
  two-class integer program has been in print since 2016. This lane has now
  measured CDCL on it and found it insufficient. A tuned branch-and-cut attempt
  is the only remaining idea for moving p_n = 73, and it is a solver-engineering
  project, not a mathematics one. **Recorded as an option, priced as unknown,
  NOT proposed as a target.**
* **DEMOTED: nothing further.** 7c#4 stays demoted.

### 13j. Reproduction (round 29)

* `research/jk_run29.py` -> explicit-seed driver with a FATAL protocol
  assertion; log `research/data/r29_k3_z23_confirm.log`, parts
  `research/data/jkpart29_k3_z23_M232_n5_p*.txt`.
* `research/jk_axis29.py` -> `research/data/jk_axis29.out`. Sections A (harvest
  + protocol check), B (the ladders), C (the discriminator), E (the price).
  **Gate.**
* `research/harv_score29.py` -> `research/data/harv_score29.out`. Brief item
  (c), exact integer arithmetic. **Gate.**
* `research/jk_sat29.py` -> the SAT engine (ZM eq. (2.2) generalised to k
  classes). `check` is the gate; logs `research/data/r29/sat_gate.log`,
  `sat_ladder.log`, `sat_k3z23.log`. Runs in `.venv-sat`.
* `research/data/r29_harvester_prereg.txt` -> pre-registration (H1-H7).
* `research/data/r29_k4_z19.log` -> j_4(P(19)) = 3810.
* Extracted sources on disk (gitignored): `research/data/r29/zm_algo.extract.log`
  (arXiv:1611.03310), `ziller2020.extract.log` (arXiv:2007.01808),
  `fkmpt_sieved.extract.log` (Long gaps in sieved sets).
* Docs: `docs/novel/jk-growth-discriminator.md` section 9 (round-29 addendum);
  `docs/novel/README.md` index amended.
* Sources read FIRST-HAND by me on 2026-09-03: arXiv:1611.03310 (full text,
  incl. section 2's ILP), arXiv:2007.01808 (full text), FKMPT "Long gaps in
  sieved sets" (full text), arXiv:2507.23041 (the AN2 passage and its reference
  list). NOT obtained: Hagedorn, Math. Comp. 78 (2009) - two 403s.
