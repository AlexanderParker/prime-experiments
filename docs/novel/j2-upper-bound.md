# j2-upper-bound - the first upper bounds on the paired Jacobsthal function j_2

Status: PROVED (paper proofs below, elementary; explicit constants script-verified,
research/j2_bound.py and research/j2_brun.py, all assertions green) for Theorems 1
and 3; PROVED-BY-STANDARD-CITATION (fundamental lemma of sieve theory, dimension 2)
for Theorem 2, the polynomial rung. Prior-art verdict: NOVEL AS FAR AS SEARCHED - the published upper-bound
ladder for j_2 is empty (established round 20 by full-text reads of both
Ziller-Morack papers; re-checked 2026-08-24, no 2018-2026 follow-up). See section 6.

## 1. What it is

Plain language. The paired Jacobsthal function j_2(n) asks: how long can a run of
consecutive positions be, in a pair of integer sequences offset by a fixed even
difference, before some position must carry a pair with BOTH entries coprime to n?
Ziller and Morack defined it in 2017, conjectured j_2(p_n#) < p_n^2 - p_n (their
Conjecture 6), and proved that conjecture implies Goldbach's conjecture AND the
infinitude of prime pairs at every fixed even difference. They proved no upper
bound of any strength, and neither has anyone since: the analogue of the
Kanold -> Stevens -> Iwaniec ladder for the ordinary Jacobsthal function simply
does not exist for the paired one. The only bound implicit anywhere is the trivial
period bound j_2(p_n#) <= p_n# (exponential in p_n). This document supplies the
first three rungs - one per slot of the ordinary ladder - and the honest ceiling
above them.

Precise form. Following Ziller-Morack (arXiv:1706.00317, Def. 2.1-2.2): j_2(n) is
the smallest m such that every paired progression <a,b>_m = {(a+i, b+i) : i=1..m}
with 2 | b-a contains a pair (x,y) with gcd(x,n) = gcd(y,n) = 1; h_2(n) = j_2(p_n#).

THEOREM 1 (elementary; the first sub-primorial bound). For every n >= 2,

    j_2(p_n#)  <=  2*3^(n-1) / V_n  +  1,      V_n = (1/2) * prod_{3<=p<=p_n} (1 - 2/p),

and explicitly

    j_2(p_n#)  <  3^(n+1) * (log p_n)^2        for all n >= 3     (n = 2: bound = 37).

Since 3^n = exp(n log 3) with n ~ p_n / log p_n, this is exp(O(p_n / log p_n)) -
genuinely below the trivial p_n# = exp((1+o(1)) p_n).

THEOREM 3 (round 22; elementary, Brun's pure sieve, no implied constant, and it
CONTAINS Theorem 1). With omega(2) = 1, omega(p) = 2 for odd p <= p_n, and
e_j(.) the elementary symmetric polynomials of the n listed weights, put

    E_K = sum_{j<=K} e_j( omega(p_1), ..., omega(p_n) )
    R_K = sum_{j>K}  e_j( omega(p_1)/p_1, ..., omega(p_n)/p_n )
    V_n = prod_p (1 - omega(p)/p) = (1/2) prod_{3<=p<=p_n} (1 - 2/p).

Then for EVERY ODD K with R_K < V_n,

    j_2(p_n#)  <=  E_K / (V_n - R_K)  +  1.

K >= n gives R_K = 0 and E_K = prod_p (1 + omega(p)) = 2*3^(n-1) - that is exactly
Theorem 1. The optimal K is far smaller (K* = 3, 5, 7, 9, 11, 13 over p_n = 5 ..
27449; K* ~ lambda*T_n with T_n = sum omega(p)/p ~ 2 log log p_n), and then
E_{K*} ~ (2 e n / K*)^{K*}, so the bound is QUASI-POLYNOMIAL:

    j_2(p_n#)  <=  exp( C * log p_n * log log p_n )  =  p_n^{C log log p_n},

with the measured constant log(bound)/(log p_n log log p_n) sitting in
[3.47, 4.16] for p_n = 173 .. 27449 (Theorem 1's own ratio diverges: 5.6 -> 139).
Theorem 3 is strictly better than Theorem 1 from p_n = 13 onwards, and by more than
a factor 300 already at p_n = 73 (1.082e9 vs 3.316e11).

THEOREM 2 (polynomial, by the fundamental lemma of sieve theory). There is an
absolute constant beta_2 (the dimension-2 sifting limit) such that

    j_2(p_n#)  <<_eps  p_n^(beta_2 + eps).

Round-22 update to the constant: the best proved dimension-2 sifting limit is
beta_2 = 4.266, the Diamond-Halberstam-Richert value (Franze, arXiv:1012.3809,
Table 1, which also gives 4.516 for Selberg's Lambda^2 Lambda^- sieve at kappa = 2
and shows Lambda^2 Lambda^- winning only from kappa >= 3). This supersedes round
21's cited 4.85 / 4.45. The conjectured truth (ZM Conjecture 6 + the project's
measured ~(p^2-p)/2 share) is exponent 2.

THE CEILING (round 22; this is the sharp form of round 21's "parity-critical").
The exponent beta_2 is not an artifact of the chosen sieve - it is the SIFTING
LIMIT of dimension 2, the threshold below which no lower-bound sieve of the
classical type produces a positive lower bound at all. Two consequences:

  (i) Theorem 2 IS the Iwaniec-analogue, already delivered. Iwaniec's ordinary
      bound j(n) << (k log k)^2 is, at primorials, exactly p_n^2 = p_n^{beta_1}
      with beta_1 = 2 the dimension-ONE sifting limit (attained by the linear /
      Rosser-Iwaniec sieve, and known optimal by Selberg's parity example). The
      paired problem has dimension 2 because each prime removes two classes, so
      the same argument delivers p_n^{beta_2}. Round 21 filed the
      Iwaniec-analogue as "open"; that was the wrong slot - see section 7.
  (ii) Ziller-Morack Conjecture 6 asks for exponent 2 on a dimension-2 problem.
      Selberg's conjectural optimal sifting limit is beta_kappa = 2*kappa, i.e.
      4 for kappa = 2 (and no sieve attaining 2*kappa is known for any kappa > 1;
      Selberg proved beta_kappa <~ 2*kappa + 19/36 for large kappa). So exponent
      2 is below even the CONJECTURAL floor of the method by a factor of two in
      the exponent. It is not an unproved-but-approachable target: it is
      parity-blocked. Consistently, in the project's own horizon frame exponent 2
      is exactly the level at which a sieve survivor in the window (y, y^2] IS a
      prime pair (Reduction A) - which is why ZM Theorem 4.1 can deduce Goldbach
      and Polignac from it.

COROLLARY (round 22; the PER-DIFFERENCE refinement, which is what the project's
F_d family actually measures). The sieve removes omega_p(d) = 2 classes for
p not dividing d and 1 class for p | d, so the sifting dimension is d-DEPENDENT:

    sum_{p<=y} omega_p(d) log p / p  =  kappa_d log y + O(1),
    kappa_d  =  2  -  (1/log y) sum_{p | d, p <= y} log p / p     (Mertens),

which runs over the whole interval [1, 2] as d ranges over residues mod the
primorial, and the same fundamental-lemma argument gives

    F_d(y)  <<_eps  y^(beta(kappa_d) + eps).

Both endpoints are attained INSIDE the family: kappa_d = 2 exactly for d coprime to
the primorial (the generic class, and by the project's percentile measurements the
hardest one), and kappa_d = 1 + O(1/log y) for d = 0 mod the primorial - which is
precisely the exact collapse j_2 = j verified in round 21, i.e. the kappa = 1
endpoint has a verified anchor. In between, d divisible by exactly the primes in
(y^theta, y] gives kappa_d = 1 + theta + O(1/log y). Verified numerically at
theta = 0.25, 0.5, 0.75 and y = 10^4, 10^5, 10^6 (research/j2_perdiff.py).
HONEST CAVEAT: for a FIXED d and y -> infinity, kappa_d -> 2, so the refinement is a
statement about differences that grow with the machine - which is exactly the family
setting F_d(y), and exactly where the project's measurements live. It is, as far as
searched, the first per-difference upper bound in the family.

COMPLEMENT (lower bound transfer). Choosing b - a = p_n# collapses the paired
problem onto the ordinary one (gcd(x + p_n#, p_n#) = gcd(x, p_n#)), so
j_2(p_n#) >= j(p_n#) and every ordinary-Jacobsthal lower bound
(Ford-Green-Konyagin-Maynard-Tao class) transfers verbatim. Script-verified
exactly at n = 3, 4, 5 (the survivor sets coincide).

## 2. Why it might be novel

Not because it is deep - Theorem 1 is Legendre inclusion-exclusion and Theorem 2
is a standard sieve citation - but because the ladder it starts did not exist:

- Ziller-Morack (both papers, full-text reads, round 20): no upper bound of any
  strength on j_2; their Remark 2.2 lists only elementary monotonicity; no
  Iwaniec citation; no heuristic for p^2 - p.
- No follow-up literature 2017-2026 computes further values or proves any bound
  (searches in section 6).
- The one-residue ladder (Kanold 2^k, Stevens polynomial-in-k... Iwaniec
  (k log k)^2) is explicitly about ONE residue class per prime; none of those
  papers treats the paired case.

Why the ladder was empty is itself worth recording, in the corrected round-22
form. For the ORDINARY function, Iwaniec's j(n) << (k log k)^2 is order p_n^2 at
primorials, which is exactly the dimension-ONE sifting limit p^(beta_1), beta_1 = 2.
ZM's conjectured paired bound is ALSO order p^2 - but the paired problem has
dimension TWO, where the sifting limit is beta_2 = 4.266 proved and 4 conjectured.
So ZM Conjecture 6 asks for a dimension-one-quality exponent on a dimension-two
problem; it is not the analogue of Iwaniec's theorem but strictly stronger than
anything a dimension-2 sieve can give, which is why ZM Thm 4.1 can extract Goldbach
and Polignac from it. The sub-conjecture rungs (3^n, quasi-polynomial, p^4.266)
are parity-safe, and nobody had bothered to write them down.

## 3. Proof

THEOREM 1. Fix n >= 2, P = p_n#, and a paired progression <a,b>_m, 2 | b-a. For
each p <= p_n let Omega_p = {-a mod p, -b mod p} and omega(p) = |Omega_p|; then
omega(2) = 1 (a, b share parity), omega(p) = 1 iff p | b-a, else 2. Position i is
"bad" (some member shares a factor with P) iff i mod p in Omega_p for some p.
Legendre inclusion-exclusion over squarefree d | rad(P): the count N_d of i <= m
hit in the prescribed classes for every p | d is, by CRT, a union of
omega(d) = prod_{p|d} omega(p) residue classes mod d, so
|N_d - omega(d) m / d| <= omega(d). Hence the survivor count satisfies

    S  >=  m * prod_p (1 - omega(p)/p)  -  prod_p (1 + omega(p))  =  m*V - E.

(Script section A verifies this inequality against direct counts on 8000 real
windows, exhaustive gear sets n = 3, 4.) The per-prime contribution to E/V is
(1 + omega)p/(p - omega); since 3p/(p-2) > 2p/(p-1) for every p, the worst case
over differences is omega = 2 at every odd prime: E <= 2*3^(n-1),
V >= (1/2) prod_{3<=p<=p_n}(1-2/p) = V_n. A fully-bad run of length m forces
S = 0, hence m <= E/V; so j_2(p_n#) <= 2*3^(n-1)/V_n + 1. (Differences with
p | b-a for small p get strictly better constants - the per-difference refinement
the project's F_d family measures.)

Explicit form: the identity (1-2/p) = (1-1/p)^2 (1 - 1/(p-1)^2) gives

    prod_{3<=p<=z} (1-2/p)  =  [2 prod_{p<=z}(1-1/p)]^2 * prod_{3<=p<=z}(1-1/(p-1)^2),

the last factor decreasing to the twin-prime constant C_2 = 0.66016... (so every
partial product exceeds C_2 - script-verified at z = 40000: 0.6601632 > C_2), and
Rosser-Schoenfeld (3.27) gives prod_{p<=z}(1-1/p) > e^(-gamma)/log z * (1 - 1/log^2 z)
for z >= 285. Chaining: V_n >= 0.3908/(log p_n)^2 for p_n >= 285, so
2*3^(n-1)/V_n + 1 < 1.71 * 3^n (log p_n)^2 + 1 < 3^(n+1) (log p_n)^2. For
p_n < 285 the inequality is verified with EXACT rational V_n (script section C:
holds for all 3 <= n <= 4203 with worst ratio 0.858 at n = 3, so the constant is
not tight anywhere). QED.

THEOREM 3. Bonferroni with an ODD truncation depth K is a LOWER bound for the
survivor indicator: a position lying in exactly r of the bad classes contributes
sum_{j<=K} (-1)^j C(r,j) = (-1)^K C(r-1,K), which is -C(r-1,K) <= 0 for r >= 1 and
equals 1 for r = 0. Summing over the m positions with the same
N_d = m*omega(d)/d + theta_d*omega(d), |theta_d| <= 1 as in Theorem 1,

    S  >=  sum_{d | rad(P), omega(d) <= K} mu(d) N_d
       >=  m * sum_{j<=K} (-1)^j e_j(omega(p)/p)  -  sum_{j<=K} e_j(omega(p))
       >=  m (V_n - R_K)  -  E_K,

since sum_{j<=K} (-1)^j e_j = V_n - sum_{j>K} (-1)^j e_j >= V_n - R_K. A fully bad
run of length m forces S = 0, hence m <= E_K/(V_n - R_K). QED.
Asymptotic shape: with K ~ lambda T_n, T_n = sum_p omega(p)/p = 2 log log p_n +
O(1), the truncation cost obeys R_K <= T^(K+1)/(K+1)! ~ (e/lambda)^(lambda T) =
(log p_n)^(-2 lambda log(lambda/e)), so any lambda with lambda log(lambda/e) > 1
(e.g. lambda = 4) makes R_K < V_n ~ c (log p_n)^(-2); and then
E_K <= (2 e n / K)^K = exp(K log(2en/K)) = exp(O(log p_n log log p_n)). The exact
optimal K is found numerically in research/j2_brun.py (exact rationals).

THEOREM 2 (sketch, standard). Sift the interval [1, m] by the classes Omega_p,
p <= z = p_n. The sieve problem has dimension kappa = 2 (omega(p) <= 2), remainders
|r_d| <= omega(d) <= 3^(nu(d)), so sum_{d < D} |r_d| << D (log D)^2 and the level of
distribution is D = m^(1-o(1)) - NOTHING is lost on the level, so the exponent is
exactly the sifting limit and no bilinear / well-factorable refinement can help. The
fundamental lemma (Halberstam-Richert Thm 2.5; Friedlander-Iwaniec Opera de Cribro
Thm 6.9; beta-sieve / DHR sifting limit beta_2 = 4.266, Franze arXiv:1012.3809
Table 1) gives S >= (1/2) m V(z) > 0 once D = z^(beta_2 + eps) and m >= D log^4 z.
Hence any fully-bad run has length << z^(beta_2 + eps). QED (by citation).

Verification of Theorem 3: research/j2_brun.py (exact rational arithmetic, all
assertions green; output research/data/j2_brun.out): the full table of
(K*, bound) against Theorem 1 and against p^4.266 for n = 3..400; the assertion
that K >= n reproduces E = 2*3^(n-1) and R = 0 (i.e. Theorem 1) at every n; that
every bound exceeds the true ZM h_2 at all comparable points; and the
quasi-polynomial ratio table to p_n = 27449.

Verification of Theorems 1-2: research/j2_bound.py (all assertions green; output
research/data/j2_bound.out): (A) the counting inequality on 8000 real windows;
(B) bound values dominate the exact ZM h_2 table at all 20 known points - the
honest price is x6 at p = 3 growing to x1.3e8 at p = 73 (a Legendre-type bound is
exponentially lossy; that is what rung 2 fixes); (C) the explicit-form inequality
with exact V_n through n = 4203 plus the monotone twin-constant check; (D) the
b-a = p# collapse, exact at n = 3, 4, 5.

## 4. Implications

Inside the project: none directly on the twin route - this is Harvester lane
(N4 executed). It prices what the machinery's exact table sits under: exact values
h_2 grow like ~p^2/2 while the proved ceiling is p^4.266; the gap between
exponent 4.266 and exponent 2 is the paired parity wall made quantitative, and
round 22 shows it is a SIFTING-LIMIT gap (a factor of two in the exponent even at
Selberg's conjectural optimum 2*kappa), not a technology gap.

Outside: the ladder now reads

    rung 0    p_n#                                  trivial (periodicity)
    rung 1    2*3^(n-1)/V_n + 1 < 3^(n+1) log^2 p   elementary        (round 21)
    rung 1.5  E_K/(V_n-R_K)+1 = p_n^(O(log log p_n))  elementary, explicit,
              quasi-polynomial                      (round 22, Theorem 3)
    rung 2    <<_eps p_n^(4.266+eps)                fundamental lemma / DHR
    CEILING   p_n^(beta_2), conjecturally p_n^4     sifting limit of dimension 2
    TARGET    p_n^2 - p_n (ZM Conjecture 6)         parity-blocked

and it aligns rung-for-rung with the ordinary ladder: Theorem 1 is the
Kanold-analogue (2^k), Theorem 3 is the Stevens-analogue (quasi-polynomial;
Stevens' g(n) <= 2 k^(2 + 2e log k)), Theorem 2 is the Iwaniec-analogue (the
sifting-limit bound at its own dimension). Named remaining moves, both cheap and
both free: (i) any improvement of beta_2 transfers verbatim; (ii) an explicit
constant in rung 2 (the fundamental lemma with explicit constants would give
j_2(p_n#) <= p_n^C for a stated C and n_0). The one move that is NOT available is
lowering the exponent towards 2 - see THE CEILING above.

## 5. Unsolved questions or conjectures it touches

- Ziller-Morack Conjecture 6 (j_2(p_n#) < p_n^2 - p_n): Theorem 2 is the first
  proved statement of the same shape (polynomial in p_n); the conjecture's
  exponent 2 vs proved 4.266 vs the dimension-2 sifting limit's conjectural
  floor 4.
- The sifting limit beta_kappa at kappa = 2 (Selberg's conjecture beta = 2 kappa,
  open for every kappa > 1): any progress there moves rung 2 directly.
- Via ZM Theorem 4.1: Goldbach and fixed-difference Polignac sit exactly at the
  top of this ladder.
- The ordinary-Jacobsthal ladder (Iwaniec's (k log k)^2, open improvement) - the
  paired case now formally joins it.
- OEIS A288815 (h_2 values): the first proved bounding sequence.

## 6. Prior-art check (2026-08-24)

Searches run this round (WebSearch):

- `Jacobsthal function generalization "several residue classes" OR "two residue
  classes" per prime upper bound sieve` - hits: Costello-Watts 1208.5342
  (computational, ordinary function), Iwaniec-ladder references, ZM's own
  computation notes, FGKMT large-gaps papers. NO published upper bound for any
  multi-class/paired Jacobsthal variant.
- `"j_2" OR "paired progressions" Jacobsthal upper bound "p_n^2" Ziller Morack
  follow-up 2018..2026` - only the two 2017 ZM papers and OEIS A288815/A072753;
  no follow-up bounds or computations found 2018-2026.
- `"sifting limit" dimension 2 Diamond Halberstam Richert value beta` -
  beta sieve beta(2) <= 4.85 (Friedlander-Iwaniec); Blight (Rutgers thesis,
  "Refinements of Selberg's sieve") beta_2 < 4.45; Franze (arXiv:1012.3809,
  Lambda^2 Lambda^- sieve) for further refinement context. These calibrate the
  Theorem 2 exponent.
- Round-20 basis (recorded in harvester.md): both ZM papers read in full - no
  bound, no bound attempt, no heuristic for p^2 - p; transfer-matrix and paired
  literature searched with and without Holt.

ROUND-22 searches (WebSearch, 2026-08-24):

- "sifting limit beta(kappa) dimension 2 value Diamond Halberstam Richert sieve"
  and "sifting limit kappa = 2 sieve dimension two lower bound positive" ->
  Franze arXiv:1012.3809 (fetched via ar5iv; Table 1 gives beta_kappa = 4.516 for
  the Lambda^2 Lambda^- sieve at kappa = 2 and 4.266 for the DHR sieves, with
  Lambda^2 Lambda^- winning only from kappa >= 3); Blight, "Refinements of
  Selberg's Sieve" (Rutgers). Also recorded: Selberg's conjecture
  beta_kappa = 2 kappa, and beta_kappa <~ 2 kappa + 19/36 for large kappa. This
  both improves and CAPS the Theorem-2 exponent.
- "Iwaniec 1978 On the problem of Jacobsthal bound omega(n)^2 log^2 sieve linear"
  -> Demonstratio Math. 11 (1978) 225-231; the bound is h(k) <= C (k log k)^2,
  i.e. p^(beta_1) at primorials. Confirms the rung-alignment above.
- "Brun pure sieve Jacobsthal function upper bound quasi-polynomial Kanold Stevens
  ladder" -> Kanold g(n) <= 2^k; Stevens g(n) <= 2 k^(2+2e log k);
  Costello-Watts improvement 2 e^gamma k^(5+5 log log k); Erdos's remark that
  Brun's method yields g(n) = O(k^c) with no accessible explicit version. All
  ONE-residue; no paired analogue of any of them. Theorem 3 occupies the Stevens
  slot of the paired ladder.
- OEIS A288815 pulled in full (text interface, 2026-08-24): 2, 6, 18, 30, 66, 150,
  192, 258, 366, 450, 570, 708, 894, 1044, 1284, 1422, 1656, 1902, 2190, 2460,
  2622 for p_n = 2..73 - the reference values every rung is measured against.

Nearest prior art: (i) the one-residue ladder (Kanold 1967 2^k; Stevens 1977
quasi-polynomial; Iwaniec 1978 (k log k)^2) - different function, methods
one-class; the paired ladder now has all three rungs, and this document's
Theorem 2 is the Iwaniec-slot bound, not an open problem (round 21 filed it as
open - see section 7); (ii) the trivial period bound j_2 <= p_n# implicit in
periodicity. VERDICT: NOVEL AS FAR AS SEARCHED (the statements are new; the
methods are deliberately classical - the contribution is the first occupied rungs
of an empty ladder, with the honest observation of why it was empty).

## 7. Corrections to round 21 (self-caught, round 22)

- "The Iwaniec-analogue is open and parity-critical" was the WRONG SLOT. Iwaniec's
  ordinary bound is the dimension-1 sifting-limit bound; Theorem 2 is its exact
  dimension-2 counterpart and was already proved in round 21. What is
  parity-blocked is not "the Iwaniec analogue" but ZM Conjecture 6 itself, which
  asks for exponent 2 = beta_1 on a kappa = 2 problem. The wall is real; it was
  mislabelled, and the corrected form is sharper: the gap is a factor of two in
  the exponent and survives even Selberg's conjectural optimum.
- "beta_2 <= 4.85 / < 4.45": superseded by the DHR value 4.266 (source above).
- "Next rung: Brun pure sieve (quasi-polynomial)": delivered as Theorem 3, and it
  turned out to CONTAIN Theorem 1 (the case K >= n) rather than sit beside it, and
  to beat it from p_n = 13 rather than only asymptotically.
