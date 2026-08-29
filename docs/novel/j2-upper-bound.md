# j2-upper-bound - the first upper bounds on the paired Jacobsthal function j_2

> **READ SECTION 11 FIRST.** Round 26 assembled Unit 1 into a submission
> candidate: section 11 carries the CURRENT ladder, the CURRENT sandwich, and
> the CURRENT not-claims list. Sections 1-10 are the working record, written
> round by round; where they disagree with section 11, **section 11 wins**, and
> the disagreements are individually marked. In particular the status block
> immediately below and section 4a items 1/3/5 were written before rounds
> 24-26 and are superseded there.
> Standing gates, all re-run green at round-26 close: `research/j2_referee.py`,
> `research/j2_citesweep.py`, `research/j2_odc6.py`, `research/j2_layer_proof.py`.

## 0. STATUS BLOCK AS OF ROUND 23 (superseded by section 11 - kept as record)

Status: PROVED WITH ALL CONSTANTS EXPLICIT (paper proofs below, elementary;
script-verified in exact rationals - research/j2_bound.py, j2_brun.py,
j2_explicit.py, all assertions green) for Theorems 1, 3 and 3E; PROVED WITH ALL
CONSTANTS EXPLICIT BY CITATION TO A CONSTANT-FREE SIEVE (Friedlander-Iwaniec
Opera de Cribro Theorem 7.7, plus the sieve-hypothesis constant K = 3 re-derived
here; research/j2_fi77.py) for THEOREM 2E, the explicit polynomial rung at
exponent 19; PROVED-BY-STANDARD-CITATION, CONSTANT NOT EXPLICIT AND NOT MAKEABLE SO
(dimension-2 sifting limit) for Theorem 2, the best-exponent polynomial rung at
4.266. Section 8 settles the explicit-constant question in both directions.
ROUND-24 UPDATES (2026-08-28, section 9): exponent 19 -> 17 FREE and -> 15 at
constant cost 135 (THEOREMS 2E', 2E''; research/j2_presieve.py); Opera de Cribro
Theorem 7.7 CHECKED AGAINST THE BOOK'S OWN TEXT (OCR of the AMS printing;
section 9a); the Halberstam-Richert Memoire OBTAINED AND READ (numdam scan;
section 9b); the 19/36-vs-0.4454 conflict SETTLED for 19/36 (section 9c;
research/j2_selberg.py); and the round-23 lower-ladder subsection of section 1
is superseded in three marked places by docs/novel/j2-lower-ladder.md.
Prior-art verdict: NOVEL AS FAR AS SEARCHED - the
published upper-bound ladder for j_2 is empty (established round 20 by full-text
reads of both Ziller-Morack papers; RE-CHECKED 2026-08-25 by citation graph rather
than keywords: Ziller-Morack arXiv:1706.00317 has exactly ONE citation in nine
years - their own companion note - which itself has ZERO, and zbMATH Open has no
"paired Jacobsthal" record at all). See section 6 for the sweeps and section 6a for
a citation audit that corrected five second-hand facts in the round-22 text.

## 1. What it is

> **NOTE (round 26).** The closing sentences of the next paragraph - "the proved
> sandwich is `p_n^{1+o(1)} .. p_n^{4.266}` around a measured truth of
> `(p_n^2 - p_n)/2`" and "an entirely self-contained polynomial bound (exponent
> 19)" - are BOTH superseded. The measured-truth reading was retracted in round
> 24; the explicit exponent is 8.04162 since round 25. Section 11a/11b are
> current. The theorem statements in this section are unaffected.

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
first three rungs - one per slot of the ordinary ladder - the honest ceiling above
them, and (round 23) the honest floor below them: the proved sandwich is
p_n^{1+o(1)} .. p_n^{4.266} around a measured truth of (p_n^2 - p_n)/2, so the
LOWER ladder is now the emptier of the two. On constants: rungs 1, 1.5, 1.5E and
2E carry all of theirs, so an entirely self-contained polynomial bound
(exponent 19) exists; the BEST-exponent rung (4.266) does not and cannot with
published tools. Section 8 settles that in both directions.

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

with the ratio log(bound)/(log p_n log log p_n) MEASURED in [3.47, 4.16] for
p_n = 173 .. 27449 (Theorem 1's own ratio diverges: 5.6 -> 139). Theorem 3 is
strictly better than Theorem 1 from p_n = 13 onwards, and by more than a factor 300
already at p_n = 73 (1.082e9 vs 3.316e11). WARNING, round 23: that measured band is
PRE-ASYMPTOTIC and does not contain the limit - see Theorem 3E immediately below,
which supplies the proved constant.

THEOREM 3E (round 23; the EXPLICIT form of Theorem 3 - a stated constant, not a
measured band). Take K = K(n) := the least ODD integer with R_K <= V_n/2, so that
Theorem 3 reads j_2(p_n#) <= 2 E_K/V_n + 1. Then

    j_2(p_n#)  <  p_n^{9.30 log log p_n}         for every n >= 3,

and the ASYMPTOTIC constant of Theorem 3 is exactly

    C_infinity  =  2 lambda_*  =  7.182242...,   lambda_* the root of
                                                 lambda (log lambda - 1) = 1
                                                 (lambda_* = 3.591121...).

Round 22 quoted the quasi-polynomial constant as "measured in [3.47, 4.16] for
p_n = 173 .. 27449". THAT BAND IS PRE-ASYMPTOTIC AND THE LIMIT IS NOT IN IT: the
ratio log(bound)/(log p_n log log p_n) climbs to 2 lambda_* = 7.1822 (the shortfall
at accessible n is the factor (1 - (log log p_n + log K)/log p_n) in
log(2 e n/K), which is 0.70 even at p_n = 27449). A measured band is not a
theorem and this one would have been the referee's first question; both numbers
are now proved. Proof and verification: research/j2_explicit.py sections A-C
(exact rationals for 5 <= p_n <= 139 and a spot ladder to p_n = 27449; the
analytic tail argument covers p_n >= 142 uniformly). n = 2 is genuinely excluded
(log log 3 = 0.094), exactly as ZM Conjecture 6 excludes n = 2.
NOTE, and it is what makes the explicit form free: the explicit choice of K
(least odd K with R_K <= V_n/2) coincides with round 22's NUMERICALLY OPTIMAL K
at every n tested, so making the constant explicit costs a factor 1.000.

THEOREM 2 (polynomial, by the fundamental lemma of sieve theory). There is an
absolute constant beta_2 (the dimension-2 sifting limit) such that

    j_2(p_n#)  <<_eps  p_n^(beta_2 + eps).

Round-22 update to the constant: the best proved dimension-2 sifting limit is
beta_2 = 4.266, the Diamond-Halberstam-Richert value (Franze, arXiv:1012.3809,
Table 1, which also gives 4.516 for Selberg's Lambda^2 Lambda^- sieve at kappa = 2
and shows Lambda^2 Lambda^- winning only from kappa >= 3). This supersedes round
21's cited 4.85 / 4.45. The conjectured truth (ZM Conjecture 6 + the project's
measured ~(p^2-p)/2 share) is exponent 2.

THEOREM 2E (round 23; RUNG 2 WITH EVERY CONSTANT EXPLICIT - the round's named
target, delivered). For every p_n >= 285,

    j_2(p_n#)  <=  1.0963 * 10^10 * p_n^19 * (log p_n)^10  +  1,

and more generally j_2(p_n#) << p_n^s for every real s > 18.308, with the constant
computable from s. Nothing here is an implied constant and there is no ineffective
threshold.

THE INGREDIENT that makes this possible, and it was not available when round 22
filed rung 2 as citation-dependent: an EXPLICIT, CONSTANT-FREE lower-bound sieve.

  THEOREM (Friedlander & Iwaniec, Opera de Cribro, Theorem 7.7). Let g be a
  density function with prod_{w<=p<z}(1-g(p))^{-1} <= K (log z/log w)^kappa for all
  z > w >= 2. Let D >= z, s := log D/log z, k := kappa + log K, and let
  Lambda = Lambda^- Lambda^2 be the Selberg lower-bound sieve of level D. If
  s >= 2k + 3, then
      S(A,z)  >=  X V(z) { 1 - ((s+3)/(2 e^k)) (2 e k/(s-3))^{(s-3)/2} }
                  -  2 R_4(A,D),      R_4(A,D) = sum_{d | P(z), d < D} tau_4(d)|r_d|.

  LEMMA (Dudek & Dunn, arXiv:2602.22720, Lemma 2.1). For the multiplicative g with
  g(2) = 1/2 and g(p) = 2/p for p >= 3,
      prod_{w<=p<z}(1-g(p))^{-1}  <=  3 (log z/log w)^2      (2 <= w < z).

THAT DENSITY IS LITERALLY OURS: omega(2) = 1 gives g(2) = 1/2 and omega(p) = 2 for
odd p gives g(p) = 2/p. It is not a coincidence - Dudek & Dunn sift n and N - n
simultaneously (the Goldbach side of ZM Theorem 4.1), which is the same
two-classes-per-prime structure as the paired Jacobsthal problem. So kappa = 2 and
K = 3 with no further work, and K = 3 IS BEST POSSIBLE (independently re-derived
here: the supremum is attained in the limit w = 3, z -> 3+, where the product is
(1-2/3)^{-1} = 3 while (log z/log w)^2 -> 1; grid search over all (w,z) with
w, z < 2*10^5 returns exactly 3.000000).

THE DERIVATION (research/j2_fi77.py, all assertions green):
  * k = 2 + log 3 = 3.098612, so FI's hypothesis is s >= 2k+3 = 9.1972. That is
    NECESSARY BUT NOT SUFFICIENT for a positive main term: the bracket vanishes at
    s* = 18.30802 and is 0.2507 / 0.5199 / 0.8202 at s = 19 / 20 / 22.
  * A = {1,...,m}, X = m, |r_d| <= omega(d) <= 2^{nu(d)}; for squarefree d,
    tau_4(d) = 4^{nu(d)}, so R_4(A,D) <= sum_{d<D squarefree} 8^{nu(d)}
    <= D prod_{p<D}(1+8/p) <= C_8 D (log D)^8 with
    C_8 = e^{8 gamma} prod_{p<10^6}(1+8/p)(1-1/p)^8 = 0.0316 (the product is
    DECREASING, so evaluating it at 10^6 is a valid upper bound for every
    D >= 10^6, which D = z^19 >= 285^19 satisfies with enormous room).
  * Positivity of m V(z) x bracket - 2 R_4 needs
    m > (2/bracket) C_8 z^s (s log z)^8 / V(z), and V(z) >= 0.3905/(log z)^2.
    At s = 19 this is m > 1.0963 * 10^10 * z^19 (log z)^10.
HONEST NOTES. (i) The bound is stated in log-form because a single clean exponent
costs an absurd threshold - p_n^{20} only holds from p_n >= 1.44 * 10^28 - and the
log-form's threshold is 285. (ii) PRE-SIEVING the small primes lowers K and hence
the exponent: computed here, K = 5/3 for p >= 5 (s* = 16.136), 1.4 for p >= 7
(15.474), 1.2624 for p >= 11 (15.077), 1.0479 for p >= 101 (14.353) - so a
pre-sieved version reaches exponent ~15, at the cost of carrying the pre-sieve
construction. Campbell (arXiv:2608.09488) does exactly this for the almost-prime
Goldbach problem, reaching K = 1.097 and s = 14.66. NOT done here; named.
(iii) SOURCE STATUS, stated because round 22 was burned by second-hand citations:
Opera de Cribro itself was not consulted directly. Theorem 7.7 is taken from TWO
INDEPENDENT VERBATIM TRANSCRIPTIONS that agree exactly - Dudek & Dunn Theorem 1.3
(arXiv:2602.22720, "An Explicit Result for the Sum of Two Almost Primes", Feb 2026)
and Campbell Theorem 2.1 (arXiv:2608.09488, Aug 2026) - both read in full text on
2026-08-25. Before publication the book must be checked directly.
(iv) DO NOT use Yamada arXiv:1511.03409 Theorem 3.1 as an alternative explicit
sieve: it is recorded as unproved as stated and inconsistent with the standard
references.
(v) WHY THEOREM 7.7 AND NOT THE OTHER TWO. Opera de Cribro carries three
constant-free results, and the choice is worth ten in the exponent. All three
thresholds are re-derived from the stated inequalities in research/j2_fi77.py
section F5:
    ODC Thm 6.9 (p.69), S >= X V(z){1 - e^{9 kappa - s} K^10} + R^-(A,D), valid
      for D >= z^{9 kappa + 1}: positive iff s > 9 kappa + 10 log K;
    ODC Cor 6.10 (p.69), S = X V(z){1 + 4 theta (9 kappa + 1)^kappa
      e^{9 kappa - s} K^11} + theta R(A,D), |theta| <= 1, needing only D >= z >= 2
      - NO hypothesis on s at all: positive iff
      s > 9 kappa + log(4 (9 kappa + 1)^kappa K^11);
    ODC Thm 7.7, the bracket used above.
                        K = 3        K = 1.097 (pre-sieved at 3)
    ODC Thm 6.9      s > 28.986      s > 18.926
    ODC Cor 6.10     s > 37.360      s > 26.294
    ODC Thm 7.7      s > 18.308      s > 14.532
Theorem 7.7 wins because K^10 is brutal at K = 3 (10 log 3 = 10.99 by itself).
Theorem 6.9 is a cleaner-looking fallback; Corollary 6.10's value is that it
assumes nothing whatever about s.

THE CEILING (round 22, corrected round 23; the sharp form of round 21's
"parity-critical"). TERMINOLOGY, fixed in round 23 because round 22 conflated two
things: the SIFTING LIMIT beta_kappa is by definition the infimum, over all
lower-bound sieves of level exponent s, of the s below which no positive lower
bound is produced. What is proved at kappa = 2 is 1.47 <~ beta_2 <= 4.266 - the
upper end is the DHR sieve's own limit (a CONSTRUCTION), the lower end is Brady's
(1+o(1)) 2 kappa/e. So 4.266 is what the best CONSTRUCTED dimension-2 sieve
achieves, not a proved floor. With that fixed, two consequences:

  (i) Theorem 2 IS the Iwaniec-analogue, already delivered. Iwaniec's ordinary
      bound j(n) << (k log k)^2 is, at primorials, exactly p_n^2 = p_n^{beta_1}
      with beta_1 = 2 the dimension-ONE sifting limit (attained by the linear /
      Rosser-Iwaniec sieve, and known optimal by Selberg's parity example). The
      paired problem has dimension 2 because each prime removes two classes, so
      the same argument delivers p_n^{beta_2}. Round 21 filed the
      Iwaniec-analogue as "open"; that was the wrong slot - see section 7.
  (ii) Ziller-Morack Conjecture 6 asks for exponent 2 on a dimension-2 problem.
      ROUND-23 CORRECTION - round 22 overstated this in two places and the
      corrected version is both weaker and sharper (sources verified against
      primary text 2026-08-25; see section 6 and section 7):
        * Selberg CONJECTURED beta_kappa = 2*kappa, so the conjectural optimum at
          kappa = 2 is 4 and exponent 2 is half of it. That part stands. But the
          conjecture is stated in Selberg's Lectures on Sieves (Collected Papers
          II, sec. 14) and is reported in Blight's thesis - NOT in Franze, which
          contains no conjecture at all.
        * "No sieve ATTAINS 2*kappa for any kappa > 1" was wrong as written. What
          is true: no lower-bound sieve attaining 2*kappa is KNOWN for kappa > 1,
          and whether one exists is OPEN (Brady, Stanford thesis 2017: "it is
          currently not known whether there is any kappa > 1 with
          beta_kappa < 2*kappa"). For 1/2 < kappa < 1 the Rosser-Iwaniec sieve
          DOES achieve beta_kappa < 2*kappa, so no blanket statement holds.
        * The best PROVED LOWER bound on the sifting limit is
          beta_kappa >= (1 + o(1)) * 2*kappa/e (Brady, improving Selberg's own by
          a factor 2), i.e. about 1.47 at kappa = 2. So exponent 2 is NOT proved
          to be below the sifting limit; it is below the CONJECTURED one, and it
          is far below what any constructed sieve reaches (4.266). Brady even
          conjectures 2*kappa is itself beatable for large kappa.
      What survives, and is the honest statement: the barrier at exponent 2 is
      the PARITY barrier, not an arithmetic fact about beta_2. In the project's
      own horizon frame exponent 2 is exactly the level at which a sieve survivor
      in the window (y, y^2] IS a prime pair (Reduction A) - which is why ZM
      Theorem 4.1 deduces Goldbach AND fixed-difference Polignac from Conjecture
      6. A dimension-2 lower-bound sieve working at level exponent 2 would
      therefore produce two simultaneous primes from a sieve lower bound, which
      is exactly what Selberg's parity example forbids for sieves of this type.
      The gap from 4.266 to 2 is thus a PARITY gap that happens to be measured in
      sifting-limit units, and the sifting-limit numbers calibrate how far the
      method is from it (4.266 proved, 4 conjectured optimal, ~1.47 the best
      proved floor - so the sieve-side question "is beta_2 < 4?" is genuinely
      open and independent of the parity obstruction).
  (iii) AND THERE IS AN ANALYTIC REASON, not just an arithmetic one, why the
      natural tool cannot reach the natural target. In Opera de Cribro's
      beta-sieve (Theorems 11.12/11.13, with F, f, beta, A, B all pinned exactly
      by (11.55)-(11.63)), the lower-bound constant satisfies B = 0 WHENEVER
      kappa >= 1/2. Our problem has kappa = 2. So the beta-sieve's lower bound is
      identically worthless at the sifting limit for us - it is not that the
      constant is small or the error term unevaluated, it is that the constant is
      ZERO. Put beside the arithmetic statement (ZM Conjecture 6 asks exponent 2
      on a kappa = 2 problem, below even Selberg's conjectural floor 2 kappa = 4),
      the two together say precisely why the natural tool cannot reach the natural
      target, from the analytic and the arithmetic side at once.

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

THE LOWER LADDER - round-23 subsection, SUPERSEDED IN THREE PLACES by
docs/novel/j2-lower-ladder.md (round 24); kept with corrections marked.

    proved lower   h_2(P(z)) >= (1.349+o(1)) z log z
                   [ROUND 24, Theorem (P1) of j2-lower-ladder.md - the first
                    bound using the paired structure; strictly beats the
                    round-23 row j(p_n#) = p_n^{1+o(1)}, whose FGKMT form is
                    z log z logloglog z / loglog z = o(z log z). (Round 23
                    wrote Rankin's (loglog)^2 denominator against the FGKMT
                    attribution; the mixed citation is corrected here.)]
    "TRUTH"        round 23 wrote "h_2 ~ (p_n^2 - p_n)/2 (measured)".
                   SUPERSEDED: on ZM's own 21 exact values, c z^2 and
                   c z (log z)^2 fit equally well (spread 1.87x each) with
                   residuals drifting in opposite directions, and the local-
                   exponent gap against the ordinary function is 0.33-0.75,
                   nowhere near the +1.0 a quadratic law needs. The supported
                   reading is h_2 = z^{1+o(1)}, model ~ 2.56 z (log z)^2.
                   Full analysis: j2-lower-ladder.md 1c.
    proved upper   p_n^{4.266+eps}; explicit p_n^15 (round 24, Theorem 2E'')

The round-23 NAMED OPEN PROBLEM "prove h_2(p_n#) >> p_n^{1+delta}" is
SUPERSEDED: on the corrected reading that target is asking for something false.
The right problems are (P2) h_2 >> z (log z)^2 / (loglog z)^{O(1)} by carrying
Rankin/FGKMT machinery through the paired construction (still a construction,
still parity-free), and (P3) the paired-Iwaniec upper question
h_2 = O(z (log z)^A). See j2-lower-ladder.md 1e.

WHY THE PAIRED PROBLEM SITS ABOVE THE ORDINARY ONE - round 23's capacity
argument here ("the ordinary covering is counting-constrained, the paired one
is not, so exponent 2 is plausible") is RETRACTED (round 24): capacity is not
scale-free, and the ordinary problem reaches the same capacities at larger z
with its answer still z^{1+o(1)}. What survives, and is proved: the CRT freedom
(killed residues {-a, -a-2e} mod p with a, e independently free) makes
j_2(p_n#) - 1 the longest interval coverable by classes {0, -E} per odd prime;
the class 0 covers every z-smooth number for free, so the paired covering only
has to reach the z-ROUGH numbers - a set thinner by one factor log z than what
the ordinary covering must reach. THAT is the structural separation between the
two problems, it is worth one logarithm, and it is the mechanism behind
Theorem (P1). Full statement: j2-lower-ladder.md 1a, 1d.

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
dimension TWO, where the best CONSTRUCTED sieve reaches beta_2 = 4.266 and Selberg
conjectures the optimum is 2 kappa = 4 (the best proved floor being only ~1.47).
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
for z >= 285. Chaining: V_n >= 0.3905/(log p_n)^2 for p_n >= 285 (ROUND-23
CORRECTION: round 21/22 wrote 0.3908, but the chain gives
2 e^{-2 gamma} C_2 (1 - 1/log^2 285)^2 = 0.390569 < 0.3908 - the stated constant
did not follow from the stated ingredients at the stated threshold; 0.3905 does,
and the exact V_n log^2 p_n is >= 0.4048 over 285 <= p_n <= 2731 anyway), so
2*3^(n-1)/V_n + 1 < 1.708 * 3^n (log p_n)^2 + 1 < 3^(n+1) (log p_n)^2. For
p_n < 285 the inequality is verified with EXACT rational V_n (script section C:
holds for all 3 <= n <= 4203 with worst ratio 0.863 at n = 3 - ROUND-23
CORRECTION: round 21's 0.858 omitted the '+1' that is part of the bound - so the
constant is not tight anywhere). QED.

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
|r_d| <= omega(d) <= 2^(nu(d)), so sum_{d < D} |r_d| << D log D and the level of
distribution is D = m^(1-o(1)) - NOTHING is lost on the level, so the exponent is
exactly the sifting limit and no bilinear / well-factorable refinement can help.
(ROUND-23 NOTE: if the sieve used is Friedlander-Iwaniec Theorem 7.7 - as in
Theorem 2E - its remainder is TAU_4-WEIGHTED, and the corresponding bound is
sum_{d<D} tau_4(d)|r_d| <= sum_{d<D squarefree} 8^{nu(d)} << D (log D)^7. Log
powers only; the exponent is unaffected either way.) The
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

    rung -1   (1.349+o(1)) p_n log p_n  LOWER      round 24, j2-lower-ladder.md
              (supersedes the FGKMT transfer row, which is o(p log p))
    TRUTH     z^{1+o(1)}, model ~2.56 p (log p)^2   round 24 reread; the round-23
              "~(p^2-p)/2" is one of two readings the data cannot separate, and
              the less supported one (j2-lower-ladder.md 1c)
    rung 0    p_n#                                  trivial (periodicity)
    rung 1    2*3^(n-1)/V_n + 1 < 3^(n+1) log^2 p   elementary        (round 21)
    rung 1.5  E_K/(V_n-R_K)+1                       elementary, exact rationals
                                                    (round 22, Theorem 3)
    rung 1.5E < p_n^{9.30 log log p_n} for n >= 3,  EXPLICIT constant; asymptotic
              constant exactly 2 lambda_* = 7.1822  (round 23, Theorem 3E)
    rung 2    <<_eps p_n^(4.266+eps)                fundamental lemma / DHR,
                                                    constant NOT explicit
    rung 2E   <= 1.0963e10 p_n^19 (log p_n)^10 + 1  EXPLICIT, p_n >= 285
              for p_n >= 285                        (round 23, FI Opera de
                                                    Cribro Thm 7.7 + K = 3)
    rung 2E'  <= 3.5301e9 p_n^17 (log p_n)^10 + 1   EXPLICIT, p_n >= 285
                                                    (round 24: pre-sieve 2,3 -
                                                    FREE, N_pre = 1, K = 5/3;
                                                    dominates 2E everywhere)
    rung 2E'' <= 7.2671e11 p_n^15 (log p_n)^10 + 1  EXPLICIT, p_n >= 285
                                                    (round 24: pre-sieve
                                                    2..11, N_pre = 135;
                                                    dominates 2E' from the
                                                    threshold on; 15 is the
                                                    SMALLEST integer exponent
                                                    FI 7.7 can ever give at
                                                    kappa = 2 - floor s* =
                                                    14.169 as K -> 1)
    CEILING   p_n^(beta_2): 4.266 proved, 4 conjectured optimal (Selberg),
              ~1.47 the best proved floor (Brady 2*kappa/e) - so the sieve-side
              question "is beta_2 < 4?" is OPEN
    TARGET    p_n^2 - p_n (ZM Conjecture 6)         parity-blocked (ZM Thm 4.1)

and it aligns rung-for-rung with the ordinary ladder: Theorem 1 is the
Kanold-analogue (2^k), Theorem 3 is the Stevens-analogue (quasi-polynomial;
Stevens' g(n) <= 2 k^(2 + 2e log k)), Theorem 2 is the Iwaniec-analogue (the
sifting-limit bound at its own dimension). Note also (round 23, verified against
erdosproblems.com problems 970 and 687 on 2026-08-25) that the ORDINARY ladder's
top rung has not moved since 1978: Iwaniec is still the record, FGKMT (JAMS 31,
2018) improved only the lower bound, and Costello-Watts only the explicit
constants. So the paired ladder is not chasing a moving target.
NAMED REMAINING MOVES, re-priced in round 23:
  (i)  any improvement of beta_2 transfers verbatim - still free;
  (ii) an EXPLICIT rung 2: DONE (Theorem 2E, exponent 19, round 23), and the
       pre-sieving move is now ALSO DONE (round 24, Theorems 2E'/2E'': exponent
       17 free, 15 at constant cost 135, and 15 is the floor of the method -
       research/j2_presieve.py, section 9d). What remains on this axis is ONLY
       exponent ~15 -> ~8 by the nested Brun / Brun-Hooley route: validity
       settled (round 23), and the target 7.972 is now READ FROM THE SOURCE
       (the HR Memoire treats exactly our density with level exponent u > 7.972
       admissible, but all its remainders are O(.) - section 9b), so the item
       is an EXPLICITNESS problem in a known theorem, not a new sieve. What is
       NOT available at any price is an explicit constant AT exponent 4.266.
       Full costing in section 8.
  (iii) the LOWER bound: round 23's h_2 >> p^{1+delta} target is SUPERSEDED
       (it asks for something probably false); the delivered rung is
       h_2 >= (1.349+o(1)) z log z (round 24, Theorem (P1)), and the open
       companions are (P2)/(P3) of j2-lower-ladder.md.
The one move that is NOT available is lowering the upper exponent towards 2 - see
THE CEILING above.

## 4a. WHAT THIS NOTE DOES NOT CLAIM (round 23, written for the referee)

> **SUPERSEDED BY SECTION 11c.** Items 1, 3 and 5 below were written in round 23
> and are out of date: item 1's "the distance is a factor of more than two"
> predates the round-23 retraction of the "2 kappa impossibility"; item 3 names
> exponent 19 where the current explicit exponent is 8.04162; item 5 says there
> is no lower bound beyond the collapse transfer, and there are now two. Section
> 11c is the current list and adds two items 4a never had (no twin-prime-gap
> corollary; the `z log z` order is not new). Kept here as the record.

Stated positively so nobody has to infer it from what is missing.

1. NO PROGRESS ON ZILLER-MORACK CONJECTURE 6. The proved exponent is 4.266; the
   conjecture asks for 2. The distance is a factor of more than two in the
   exponent, and section THE CEILING argues it is a parity gap. Nothing here
   moves the conjecture, and nothing here should be read as evidence for or
   against it beyond the observation that the empirical share h_2/(p^2-p) sits
   near 1/2 through p_n = 73.
2. NO NEW SIEVE THEORY. Theorems 1, 3 and 3E are Legendre and Brun with the
   arithmetic done carefully; Theorem 2 is the fundamental lemma applied to a
   dimension-2 problem. The contribution is that the ladder was EMPTY, not that
   the rungs are hard. Anyone who knows the ordinary ladder could have written
   these in an afternoon - the point is that in nine years nobody did (the two
   Ziller-Morack papers have one citation between them, their own).
3. THE EXPLICIT BOUND AND THE BEST BOUND ARE NOT THE SAME BOUND. Rung 2E is
   fully explicit but carries exponent 19; rung 2 carries exponent 4.266 but its
   constant is not explicit, and CANNOT be made so with published tools, because
   4.266 is the numerically-solved output of the DHR differential-delay system and
   the sieve inequality at that dimension carries an uncomputed
   O((log log y)^2 (log y)^{-1/6}) error. Anyone wanting "the best proved
   exponent" must take the inexplicit statement; anyone wanting a self-contained
   inequality must take exponent 19. Both are stated. See section 8.
4. THE COMPUTATIONAL HALF IS REPLICATION PLUS STRUCTURE, NOT NEW DATA. Ziller-
   Morack's companion note arXiv:1706.03668 computes h_2 to p_n = 73 and its
   ancillary files list the extremal configurations; the delta reduction the
   project used is essentially their Proposition 1.5(2). What is new on that side
   is the PER-DIFFERENCE family F_d(y), the twin percentile inside it, the
   shallow-extension cap law, and the cross-gear extension ladder - questions
   they do not ask.
5. NO LOWER BOUND BEYOND THE COLLAPSE. j_2(p#) >= j(p#) is p_n^{1+o(1)}; there is
   no proved lower bound of order p_n^{1+delta}. The sandwich this note proves is
   p_n^{1+o(1)} .. p_n^{4.266} around a truth of p_n^2/2.
6. NOTHING ABOUT PRIMES. Every statement here is about coverings of an interval
   by residue classes. The bridge to Goldbach and Polignac is Ziller-Morack's
   Theorem 4.1 and it needs the conjecture, not these bounds.

## 5. Unsolved questions or conjectures it touches

- Ziller-Morack Conjecture 6 (j_2(p_n#) < p_n^2 - p_n): Theorem 2 is the first
  proved statement of the same shape (polynomial in p_n); the conjecture's
  exponent 2 against the 4.266 the best constructed dimension-2 sieve reaches and
  Selberg's conjectural optimum 2 kappa = 4.
- The sifting limit beta_kappa at kappa = 2 (Selberg's conjecture beta = 2 kappa,
  open for every kappa > 1): any progress there moves rung 2 directly.
- Via ZM Theorem 4.1: Goldbach and fixed-difference Polignac sit exactly at the
  top of this ladder.
- The ordinary-Jacobsthal ladder (Iwaniec's (k log k)^2, open improvement) - the
  paired case now formally joins it.
- OEIS A288815 (h_2 values): the first proved bounding sequence.
- ROUND 23 ADDITIONS:
  * A NEW named problem of this note's own: prove h_2(p_n#) >> p_n^{1+delta} for
    some delta > 0 (ideally >> p_n^2). A construction problem, not parity-blocked,
    with no proved rung above the collapse transfer j_2 >= j - see THE LOWER
    LADDER in section 1.
  * Sharpness of ZM Conjecture 6's hypothesis: h_2 = p_n^2 - p_n by EQUALITY at
    n = 1 and n = 2, so "n >= 3" is exactly sharp (round-23 referee finding; the
    project's own table previously had the n = 2 row wrong).
  * The 19/36 versus 0.4454 discrepancy in the large-kappa asymptotic of
    beta_kappa (Franze versus Ford/Brady, apparently from the same Selberg
    equation) is unresolved here and is flagged for whoever needs that constant.

## 6. Prior-art check (rounds 21-22 dated 2026-08-24; ROUND-23 re-run dated 2026-08-25 at the end of this section)

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

ROUND-23 SWEEP, RUN AND DATED 2026-08-25 (the standing lesson from round 22 is
that prior-art checks EXPIRE, so this is a full re-run from scratch before any
publication claim, not a re-quote of the 2026-08-24 verdict). Method changed from
keyword search to CITATION GRAPH, which is stronger evidence:

- Semantic Scholar citations of arXiv:1706.00317 in nine years: EXACTLY ONE -
  Ziller-Morack's own companion note arXiv:1706.03668. Citations of 1706.03668:
  ZERO.
- zbMATH Open API, "paired Jacobsthal": NO RECORDS AT ALL.
- OpenAlex full-text "paired Jacobsthal": only the two Ziller-Morack records.
  (One 2026 false hit, "The Jacobsthal Window", Zenodo 10.5281/zenodo.19164834,
  is about exoplanet period ratios.)
- OEIS A288815 pulled again (record stamp #19 Apr 12 2026): still 21 terms, still
  only two links (both ZM 2017), comment states only the CONJECTURE. No proved
  bound has been deposited.
- arXiv API metadata sweep, all:Jacobsthal AND cat:math.NT (the complete set of
  math.NT records, 54) plus all:Jacobsthal across categories. EVERY 2025-2026
  Jacobsthal item concerns Jacobsthal NUMBERS, SUMS, POLYNOMIALS or CONGRUENCES -
  a different Jacobsthal. None concerns the Jacobsthal FUNCTION, and none proves
  a bound. (Listed: 2608.01748, 2608.00347, 2607.20763, 2607.11068, 2606.24936,
  2605.31114v2, 2605.16956, 2601.12579, 2601.02664, 2510.16571, 2508.11650,
  2508.13165, 2506.12612, 2505.20547, 2504.15505, 2503.12561, 2502.11045,
  2504.03646.)
- Holt arXiv:2502.20470 - the paper that cost round 22 two novelty labels
  elsewhere - was re-examined for THIS document specifically: full text
  downloaded, "Jacobsthal" occurs ZERO times. It is not prior art for the
  upper-bound ladder. Holt arXiv:2603.25915 (Mar 2026) mentions Jacobsthal twice
  in passing (citing Hagedorn for a table of values) and proves no bound.
- ZM's own primitives re-verified against full text: Def. 2.1 (j_2), Def. 2.2
  (h_2), Conjecture 6 wording, and the fact that their papers contain only
  Propositions 3.2/3.5, Corollaries 3.3/3.4 and Theorem 4.1 - all CONDITIONAL
  implications - plus computation to p_21 = 73. They prove no upper bound of any
  strength. The "first proved upper bounds" framing is supported.
- Ordinary-ladder frontier re-verified against the live Erdos-problems database
  (erdosproblems.com, fetched 2026-08-25): Problem 970 is OPEN and states
  "Iwaniec [Iw78] proved h(k) << (k log k)^2"; Problem 687 states "the best known
  upper bound is due to Iwaniec, Y(x) << x^2". So IWANIEC 1978 IS STILL THE
  RECORD IN AUGUST 2026 - FGKMT (JAMS 31, 2018) improved only the LOWER bound,
  and Costello-Watts improved only EXPLICIT constants.

Nearest prior art: (i) the one-residue ladder (Kanold 1967 2^k; Stevens 1977
quasi-polynomial; Iwaniec 1978 (k log k)^2) - different function, methods
one-class; the paired ladder now has all three rungs, and this document's
Theorem 2 is the Iwaniec-slot bound, not an open problem (round 21 filed it as
open - see section 7); (ii) the trivial period bound j_2 <= p_n# implicit in
periodicity. VERDICT: NOVEL AS FAR AS SEARCHED (the statements are new; the
methods are deliberately classical - the contribution is the first occupied rungs
of an empty ladder, with the honest observation of why it was empty).

## 6a. CITATION AUDIT (round 23, every source read in full text, 2026-08-25)

Round 22 cited several facts at second hand. Checked against primary text, five
were wrong or imprecise. All are now fixed in the body; they are listed here
because a referee would find them and because the pattern (second-hand sieve
folklore) is worth recording.

1. FRANZE'S INITIAL. The author of arXiv:1012.3809, "Sifting limits for the
   Lambda^2 Lambda^- sieve", is C. S. (CRAIG) FRANZE, not "M. Franze". Published:
   Journal of Number Theory 131 (2011) 1962-1982, doi 10.1016/j.jnt.2011.04.008.
   (Cross-confirmed by the DHR book preface, which thanks "Craig Franze".)
2. TABLE 1 IS CORRECT AS QUOTED. Verbatim, three independent renderings (arXiv
   PDF, ar5iv, archive.org OCR) agree:
       kappa        2      3      4      5       6     ...
       DHR beta   4.266  6.640  9.072  11.534  14.014
       L^2L^-     4.516  6.520  8.522  10.523  12.524
   So beta_2 = 4.266 (DHR) and 4.516 (Lambda^2 Lambda^-) at kappa = 2, and
   "Lambda^2 Lambda^- wins only from kappa >= 3" is Franze's own sentence. Blight
   reproduces DHR at full precision: 4.266450 at kappa = 2. Ford's 2023 sieve
   notes give beta(2) <= 4.2665. CAVEATS NOW RECORDED: Franze's table values are
   computed, and his abstract says "the evidence strongly suggests"; and his
   table is NOT the state of the art at kappa = 3, where Blight's 6.458 beats
   both. At kappa = 2, 4.266 stands (Blight's kappa = 2 value is 4.45, worse).
3. THE SELBERG CONJECTURE IS NOT IN FRANZE. The word "conjecture" does not occur
   in arXiv:1012.3809. beta_kappa = 2*kappa is Selberg's, in Lectures on Sieves
   (Collected Papers II, sec. 14); the accessible restatement is Blight,
   "Refinements of Selberg's Sieve" (Rutgers thesis), sec. 2.1.
4. 19/36 VERSUS 0.4454. **SUPERSEDED BY SECTION 9c - READ THAT INSTEAD.** As
   written in round 23 this item said "the safe form, and the one now used: cite
   2*kappa + 0.4454 (Ford/Brady), and 19/36 only as 'as stated in Franze'".
   Round 24 SETTLED the conflict the other way, first-hand and in exact
   rationals: **cite 2*kappa + 19/36**; 0.4454 is recorded as UNVERIFIED, not as
   wrong. The round-23 text is kept here only as the record of what was believed
   before the primary sources were read. (The stale instruction survived two
   rounds inside this document and was caught in round 26 by
   research/j2_citesweep.py section D - which is why that sweep is now a gate
   and not a manual step.)
   The round-23 observation itself stands: Franze writes "Selberg proved that
   for sufficiently large kappa, this sieve yields beta_kappa <~ 2*kappa +
   19/36", while Ford (Sieve Methods notes, 2023) and Brady (Stanford thesis,
   2017) attribute 0.4454 to the same Selberg source. That IS a genuine conflict
   in the literature; section 9c resolves it.
5. IWANIEC'S THEOREM, EXACT FORM. It is h(k) << (k log k)^2 with k = omega(n),
   equivalently J(P(z)) << z^2 (Granville arXiv:2010.01211 states the deduction
   explicitly). "g(n) << (log n)^2" is a strictly weaker corollary and is NOT the
   theorem; this document uses the (k log k)^2 / z^2 forms only.
6. COSTELLO-WATTS ARXIV ID. The bound g(n) <= 2 e^gamma k^{5+5 log log k}
   (k > 120) is in arXiv:1306.1064, "A short note on Jacobsthal's function".
   arXiv:1208.5342 is a different, range-restricted COMPUTATIONAL result
   (h(k) <= 0.27749612254 k^2 log k for 50 <= k <= 10000); a third paper,
   arXiv:1209.3464, became Math. Comp. 84 (2015) no. 293.
7. KANOLD AND STEVENS, exact sources: H.-J. Kanold, Math. Ann. 170 (1967)
   314-326 (g(n) <= 2^k); H. Stevens, Math. Ann. 226 (1977) 95-97
   (g(n) <= 2 k^{2+2e log k}), stronger than Kanold for k >= 260.

EXPLICIT-SIEVE SOURCES, adjudicated from primary text on 2026-08-25 (this is what
made Theorem 2E possible and what rules out the alternatives):

8. WHICH FUNDAMENTAL LEMMAS ARE EXPLICIT, adjudicated by page-by-page reading -
   and CORRECTED, because a numbering chimera passed through two drafts of this
   document before it was caught.
   * IWANIEC-KOWALSKI HAS NO THEOREM 6.9 AND NO COROLLARY 6.10. Chapter 6
     ("Elementary Sieve Methods") stops at Theorem 6.7; in IK, 6.9 and 6.10 are
     EQUATION labels. That numbering belongs to OPERA DE CRIBRO, and the two were
     conflated. IK's "s >= 9 kappa + 1 with K^10" result is IK THEOREM 6.1 /
     COROLLARY 6.2 (p.158). Earlier drafts of this section said
     "Iwaniec-Kowalski Theorem 6.9"; that object does not exist.
   * IK's FUNDAMENTAL LEMMA 6.3 has no lower bound on s but its error is
     1 + O(e^{-s}(1 + K/log z)^{10}) - the K-dependence sits INSIDE the O and to
     the tenth power - so it is NOT explicit.
   * HALBERSTAM-RICHERT Theorem 2.5 and FRIEDLANDER-IWANIEC 11.12/11.13 carry
     unspecified O's. (For 11.12/11.13 the precise position is worth recording:
     F, f, beta, A and B are ALL PINNED DOWN EXACTLY by (11.55)-(11.63); the only
     unevaluated object is an O((log D)^{-1/6}) whose implied constant depends
     continuously on kappa and L.)
   * TENENBAUM's fundamental lemma is his THEOREM 4.4 (Theorem 3 in the 1995 CUP
     edition), NOT 4.3, and "Theorem I.4.2" does not exist either - I.4.2 is a
     COROLLARY, the Bonferroni inequality. It reads
     S(A,P;y) = X prod(1 - w(p)/p){1 + O(u^{-u/2})} + O(sum_{d <= y^u}|R_d|),
     two bare O's with constants depending on kappa and A. Confirmed against both
     editions. So "the fundamental lemma gives exponent ~6.5" is the SHAPE of a
     theorem, not one.
   * NATHANSON Chapter 6 is a dead end for this purpose: it is "Elementary
     estimates for primes" and contains no general-dimension sieve at all.
9. WHAT DOES WORK is not a fundamental lemma at all: the explicit Selberg
   Lambda^- Lambda^2 sieve, Friedlander-Iwaniec Opera de Cribro THEOREM 7.7,
   which is constant-free. It became citable because two 2026 papers on the
   almost-prime Goldbach problem needed exactly it for exactly our density
   function: Dudek & Dunn, arXiv:2602.22720 (Theorem 1.3, Lemma 2.1) and
   Campbell, arXiv:2608.09488 (Theorem 2.1). Both were read in full text and
   their transcriptions of FI 7.7 AGREE VERBATIM. The book itself was not
   consulted directly - stated as a caveat, and to be closed before publication.
   METHOD NOTE WORTH KEEPING: sifting n and N - n is the same
   two-classes-per-prime problem as the paired Jacobsthal function, so the
   explicit-Goldbach literature is the natural source of explicit tools for this
   ladder. That is why Dudek-Dunn's Lemma 2.1 is literally our K = 3.
10. DO NOT CITE Yamada arXiv:1511.03409 Theorem 3.1 as an explicit sieve: it is
   recorded as unproved as stated and inconsistent with the standard references.

## 7. Corrections to round 21 (self-caught, round 22) and to round 22 (round 23)

ROUND-23 CORRECTIONS TO ROUND 22, all self-caught in the referee pass:

- THE CEILING's clause "no sieve attaining 2*kappa is known for any kappa > 1"
  was written as if it were an impossibility theorem. It is an OPEN PROBLEM, and
  it is false as a blanket statement (Rosser-Iwaniec beats 2*kappa for
  1/2 < kappa < 1). Corrected in the body, with the proved floor
  beta_kappa >= (1+o(1)) 2*kappa/e (Brady) recorded - so exponent 2 is NOT proved
  to sit below the sifting limit, only below the conjectured one. The barrier
  that actually blocks exponent 2 is PARITY, via ZM Theorem 4.1.
- "Quasi-polynomial with measured constant in [3.47, 4.16]" was a measured band
  presented where a theorem belongs. THEOREM 3E supplies the theorem, and the
  measured band turns out not to contain the limit: the asymptotic constant is
  2 lambda_* = 7.1822.
- The explicit chain "V_n >= 0.3908/(log p_n)^2 for p_n >= 285" does not follow
  from the stated ingredients: 2 e^{-2 gamma} C_2 (1 - 1/log^2 285)^2 = 0.390569,
  which is below 0.3908. The safe constant is 0.3905, and the conclusion of
  Theorem 1 is unaffected either way. (The INEQUALITY as stated is nonetheless
  true where checked - exact V_n log^2 p_n >= 0.4048 over 285 <= p_n <= 2731 -
  only its derivation was one digit short. research/j2_explicit.py section A.)
- Five citation-level errors, listed in section 6a.

ORIGINAL ROUND-22 CORRECTIONS TO ROUND 21 (unchanged):

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

## 8. THE EXPLICIT-CONSTANT QUESTION, SETTLED IN BOTH DIRECTIONS (round 23)

The referee ask is "what is the implied constant in Theorem 2?". The answer has
two layers and they come out differently, which is why both rungs are stated.

LAYER 1 - AT EXPONENT 4.266 THE CONSTANT IS NOT AVAILABLE, AND THAT IS STRUCTURAL.
beta_2 = 4.266 is not a closed form. It is the output of the
Diamond-Halberstam-Richert differential-delay system; the value is certified (a
20-decimal interval-arithmetic certification exists) but the SIEVE INEQUALITY at
that dimension carries an uncomputed error of shape
O((log log y)^2 (log y)^{-1/6}), and no explicit-constant version of any sieve AT
its sifting limit exists for any kappa > 1. Worse, even if that constant were
computed, the exponent 1/6 means s = beta_2 + 0.01 would need log y of order
10^12. So "j_2(p_n#) <= C p_n^{4.266+eps} for n >= n_0, C and n_0 stated" is not
available and would not be usable if it were.

LAYER 2 - AT A LARGER EXPONENT IT IS AVAILABLE, AND ROUND 23 TOOK IT.
Theorem 2E in section 1 gives, with every constant stated and threshold 285,

    j_2(p_n#)  <=  1.0963 * 10^10 * p_n^19 * (log p_n)^10  +  1,

from Friedlander-Iwaniec Opera de Cribro Theorem 7.7 (an explicit, constant-free
Lambda^- Lambda^2 lower-bound sieve) together with the sieve-hypothesis constant
K = 3, which is exact and best possible for our density and which Dudek-Dunn
supply as their Lemma 2.1 for LITERALLY OUR g. Full derivation and verification:
research/j2_fi77.py.

WHY THE FIRST ATTEMPT MISSED IT, recorded as a method note. The round's first pass
looked for an explicit FUNDAMENTAL LEMMA at dimension 2 and found none - correctly:
Halberstam-Richert Theorem 2.5, Iwaniec-Kowalski Theorem 6.1/Corollary 6.2 and
Fundamental Lemma 6.3, and Friedlander-Iwaniec 11.12/11.13 all carry unspecified
O's, and that conclusion stands. (Earlier drafts wrote "Iwaniec-Kowalski Theorem
6.9"; no such result exists - see section 6a item 8.) The tool that works is an
explicit
SELBERG Lambda^- Lambda^2 sieve, and it became citable only because two 2026
papers on the almost-prime Goldbach problem (Dudek-Dunn, Campbell) needed exactly
it, for exactly our density function. The adjacency is not luck: sifting n and
N - n is the same two-classes-per-prime problem, so the Goldbach literature is the
natural place to find explicit tools for the paired Jacobsthal function. That is a
reusable observation for anyone continuing this ladder.

WHAT REMAINS, priced.
  (a) EXPONENT 19 -> ~15 by PRE-SIEVING. K falls from 3 to 5/3 (p >= 5, s* =
      16.136), 1.4 (p >= 7, 15.474), 1.2624 (p >= 11, 15.077), 1.0479 (p >= 101,
      14.353) - all computed here. Campbell reaches K = 1.097 and s = 14.66 for
      the Goldbach analogue. The cost is carrying the pre-sieve construction and
      its own threshold. Bounded, mechanical, not done.
  (b) EXPONENT ~15 -> ~8 by the BRUN-HOOLEY route. The dyadic-range design
      (primes split at z^{alpha_j}, Bonferroni depth K_j in range j, with
      T_j = 2 log(alpha_{j-1}/alpha_j) and V_j = (alpha_j/alpha_{j-1})^2 per
      range) has level exponent s = sum_j alpha_{j-1} K_j and a truncation cost
      that converges when the depths grow. Our own optimisation over GEOMETRIC
      alpha_j = theta^{j-1} reaches s = 9.07 at cost 0.36 (theta = 1/2, depths
      ceil(4 * 1.05^{j-1})). Two caveats, both real:
        * VALIDITY: SOLVED THIS ROUND, and the fix is a one-word change.
          The naive PRODUCT truncation {d : nu(d_j) <= K_j for all j}, counting
          prime factors inside EACH BAND separately, is NOT a valid lower-bound
          sieve - each band's odd-depth Bonferroni factor is <= 0, so a product of
          two of them is >= 0; 36 explicit small-integer counterexamples are
          enumerated in research/j2_explicit.py section D, the smallest being
          K_1 = K_2 = 1, r_1 = r_2 = 2, Lambda = +1 against an indicator of 0.
          THE CORRECT TRUNCATION COUNTS THE WHOLE UPPER TAIL: with a partition
          1 = alpha_0 > alpha_1 > ... and depths H_j, require
              nu( d restricted to primes above z^{alpha_j} )  <=  H_j
          for every j - nested constraints, not independent ones - and take
          H_j = 2 h_j + 1 for the lower sieve, 2 h_j + 2 for the upper. This is
          the refinement Tenenbaum describes in the paragraph before his
          fundamental lemma (GSM 163, p. 70; he sets it as Exercise 86 and proves
          nothing about it there). VERIFIED, not assumed:
          research/j2_nested.py tests 168,400 (depth pattern, bad-count)
          configurations over 1, 2 and 3 partition points and finds ZERO
          violations of Lambda^- <= [survives] <= Lambda^+, while the per-band
          form fails on 36 of them. A PRE-REGISTERED GUESS WAS REFUTED IN THE
          SAME SCRIPT: monotone depths h_j are NOT needed for validity (0
          violations over all 271 non-monotone patterns tested) - monotonicity is
          what makes the LEVEL cost sum_j alpha_{j-1} H_j converge, nothing more.
          WHAT IS STILL MISSING is therefore not validity but the explicit
          MAIN-TERM estimate for that truncation - an explicit lower bound on
          sum_{d in D^-} mu(d) g(d) against V(z). Halberstam-Richert's own
          Mémoire (Mém. S.M.F. 25 (1971) 97-106) is reported to carry exactly
          that, with level exponent tending to 7.972; that lead could NOT be
          verified against the actual text this round and is recorded as
          UNVERIFIED.
        * geometric alpha_j is suboptimal: the reported optimum spends the budget
          on MANY SHALLOW ranges (a polynomially decaying alpha_j) rather than
          geometrically thinning deep ones, reaching about 7.78. Also unverified
          here.
      One thing WAS checked, because it was a possible error in our own arithmetic
      rather than a lead: the per-range factor in our cost functional is an
      AMPLIFICATION by (1/theta)^kappa = 4, i.e. the tail is DIVIDED by V_j. It is
      not carried upside down (research/j2_fi77.py section F4, closed forms
      confirmed against an empirical Mertens product over (10^3, 10^6]).
  (c) The gap from ~8 to 4.266 is the sifting-limit gap and is not bookkeeping.

REMAINDER-TERM CORRECTION, recorded because round 22's Theorem-2 sketch has it in
the weaker form: FI Theorem 7.7's remainder is TAU_4-WEIGHTED. The correct
statement for our problem is sum_{d<D} tau_4(d)|r_d| <= sum_{d<D squarefree}
8^{nu(d)} << D (log D)^8, not the unweighted << D (log D)^2. Log powers only - the
EXPONENT is unaffected - but the sketch must say the weighted form.


## 9. ROUND-24 VERIFICATION RECORD (2026-08-28) - the three named items of the
## submission checklist, plus the pre-sieved rungs

### 9a. Opera de Cribro Theorem 7.7 - CHECKED AGAINST THE BOOK'S OWN TEXT

Round 23 carried Theorem 7.7 on two agreeing verbatim transcriptions (Dudek-Dunn
arXiv:2602.22720 Thm 1.3; Campbell arXiv:2608.09488 Thm 2.1) and said the book
must be seen before submission. THIS ROUND IT WAS: the AMS printing's own text
(Google Books OCR of the two library-record volumes, harvested in-round via a
literature sub-search and relayed with OCR/reconstruction separated) shows
Theorem 7.7 on p. 111, in Chapter 7 (the Selberg Lambda^2 Lambda^- chapter),
with statement, hypothesis (s >= 2k+3, k = kappa + log K), bracket and remainder
2 R_4(A, D) matching BOTH transcriptions in every particular. Three independent
renderings of the theorem now agree, one of them the book itself. CAVEAT,
recorded honestly: the book was seen through OCR snippets, not held; the check
is of the mathematical content, not of the typography. RESIDUE FROM THE SAME
PAGES: (7.122) is only a loose sufficient condition for bracket positivity
(s >= 2k + 2 sqrt(2k log k) + log k + 9 = 21.6 at our k = 3.0986, WEAKER than
the exact threshold s* = 18.308 we compute from the bracket itself), and
Corollary 7.8 (s >= 2k + (2+c) sqrt(2k log k)) is asymptotic-in-k and does not
improve our exponent at k ~ 3.1 - both examined so nobody re-opens them.
TWO NAMED OPENINGS from the same harvest, NOT yet resolved (next-round items,
priced): ODC Chapter 6's beta-sieve pages print sifting-limit values
beta_1 = 3.8629, beta_2 = 7.5941 around pp. 71-73 - if THAT apparatus is
explicit (constant-free) at kappa = 2, the explicit exponent drops from 15 to
~8.6, converging with the HR Memoire's 7.972 (9b); its explicitness is NOT
established (the Chapter 11 beta-sieve carries an uncomputed O((log D)^{-1/6}),
round 23 - whether the Chapter 6 version does too is exactly the open
question). And S. Blight, "Refinements of Selberg's sieve" (PhD thesis, Rutgers
2010; title and her own values beta_2 <= 4.450, beta_3 <= 6.458, beta_4 <=
8.470 now first-hand from Heath-Brown's zbMATH review, 9c) may contain an
explicit Lambda^2 Lambda^- variant; not obtained.

### 9b. The Halberstam-Richert Memoire - OBTAINED AND READ, and the 7.972 lead
### is CONFIRMED AS DERIVED, NOT PRINTED

Halberstam & Richert, "A new look at Brun's sieve" (Mem. Soc. Math. France 25
(1971) 97-106; free numdam scan, located and read in-round). It treats EXACTLY
our density - the paper's example is A = {n(n+2) : n <= x} with omega(2) = 1,
omega(p) = 2 - and its two printed conditions ((1.2): lambda e^{1+lambda} < 1;
positivity: lambda^2 e^{2 lambda}(2 + e^2) < 1) admit every level exponent
u > 1 + 2.01/(e^{lambda*} - 1) where lambda* is the root of the second
condition. The figure 7.972 does NOT appear in the paper (it says only
"u < 8"); it FOLLOWS from the printed conditions, and this lane re-derived it
independently in research/j2_presieve.py P4: lambda* = 0.2533219,
u = 7.971954833 (asserted). STATUS OF THE LEAD, now precise: the exponent-8
target is real, sits on OUR density in the SOURCE, and is an EXPLICITNESS
problem (every remainder in the Memoire is an unspecified O(.)) - not a new
sieve. The nested-truncation route of round 23 is the modern form of exactly
this construction.

### 9c. The 19/36 vs 0.4454 conflict - SETTLED FOR 19/36
### (research/j2_selberg.py, all assertions green)

VERDICT: beta_kappa <= 2 kappa + 19/36 + o(1) (kappa -> infinity) is Selberg's
own announced constant; 0.4454 (Ford 2023 lecture notes; Brady 2017 thesis,
both citing Lectures on Sieves (14.40)) could not be reproduced and every
computable consequence contradicts it. Evidence, tiered:
 1. FIRST-HAND (fetched by this lane 2026-08-28, zbMATH Open API): Greaves'
    review of Selberg's own announcement (Oslo 1987 symposium; Zbl 0675.10030):
    "alpha_k > 1/(2k+19/36) for all sufficiently large k" (Selberg's alpha is
    the reciprocal convention). Heath-Brown's review of Franze (Zbl
    1235.11089): Selberg "showed that the sieving limit satisfies beta_kappa <=
    2 kappa + 19/36 + o(1)".
 2. FIRST-HAND (arXiv:1012.3809 full text on disk): Franze's reproduction of
    Selberg's pp. 174-176 computation, re-derived here IN EXACT RATIONALS:
    optimal a = 1/4, threshold d = -7/72, constant exactly 19/36.
 3. NUMERICAL: at 2 kappa + 0.4454 the Selberg functional's main term is
    strictly NEGATIVE (coefficient -0.0369) - no lower bound exists there.
 4. Franze's own computed Lambda^2 Lambda^- table (kappa = 2..10) approaches
    0.5278 from below, every entry already ABOVE 0.4454.
LABELLED SPECULATION on the error's origin: Greaves' review carries, one
sentence before the 19/36, "beta_k ~ c/k ... for a certain constant c close to
1/2.445" (the Buchstab-iterated family) - the digit string 2.445 sits directly
beside 19/36 in the primary source's own review. RESIDUAL CAVEAT: the printed
(14.40) itself remains unread (Lectures on Sieves is in copyright and no scan
was found); 0.4454 is recorded as UNVERIFIED rather than WRONG. FOR THE PAPER:
cite 19/36; nothing numerical changes anywhere (it is a large-kappa statement).
CITATION CORRECTIONS BANKED (extending section 6a, both first-hand from
Heath-Brown's review): the 4.266 book is Diamond-Halberstam-GALWAY, "A
higher-dimensional sieve method", Cambridge Tracts 177 (2008), Zbl 1207.11099
(the METHOD is DHR); and Blight's OWN kappa = 2 value is 4.450 - the 4.266450
she prints at full precision is her quotation of DHR, not her result.

### 9d. THEOREMS 2E' AND 2E'' - the pre-sieved explicit rungs
### (research/j2_presieve.py, all assertions green; round-23 constants
### reproduced as in-script controls)

Pre-sieving changes exactly one factor in the Theorem 2E constant: with the
primes below p_0 sieved out first, A' is a union of N_pre = prod_{p<p_0}
(p - omega(p)) residue classes mod prod_{p<p_0} p, so |r_d| <= 2^{nu(d)} N_pre
uniformly, while X V'(z) = m V(z) exactly; positivity needs
m > (2/bracket(s, k(p_0))) N_pre C_8 z^s (s log z)^8 / V(z). And N_pre(5) = 1
because gear 3 keeps a single class - PRE-SIEVING 2 AND 3 IS FREE.

  THEOREM 2E'  (p_0 = 5, K = 5/3, s* = 16.136):
      j_2(p_n#) <= 3.5301e9 * p_n^17 * (log p_n)^10 + 1     (p_n >= 285)
    - smaller exponent AND smaller constant than 2E: dominates it everywhere.
  THEOREM 2E'' (p_0 = 13, N_pre = 135, K = 1.18182, s* = 14.822):
      j_2(p_n#) <= 7.2671e11 * p_n^15 * (log p_n)^10 + 1    (p_n >= 285)
    - dominates 2E' from the threshold on (factor 395 at p_n = 285).
  FLOOR OF THE METHOD: as p_0 grows, K -> 1, k -> 2, and the FI 7.7 bracket's
  threshold tends to s*(k=2) = 14.169 > 14. So EXPONENT 15 IS THE SMALLEST
  INTEGER FI 7.7 CAN EVER DELIVER at kappa = 2, and p_0 = 13 already attains
  it; p_0 = 13 is also the optimum of the full ladder at every p_n tested
  (deeper pre-sieving buys s* only in the third decimal while N_pre grows
  double-exponentially). More generally j_2(p_n#) << p_n^s for every real
  s > 16.136 free, and for every s > 14.822 at constant cost 135.

## 10. ROUND-25: THE EXPONENT FALLS 15 -> 8, AND BOTH NAMED OPENINGS CLOSE
## (research/j2_odc6.py, all assertions green; sources read first-hand 2026-08-29)

Round 24 left the upper ladder with exactly two named openings, both of them
acquisition problems rather than mathematics. Both are now closed, and together
they drop the fully explicit exponent from **15 to 8.042**, with the log power
falling from 10 to 3.

### 10a. OPENING 1 - BLIGHT'S THESIS: OBTAINED, READ, AND IT DOES NOT HELP

**Sara Elizabeth Blight** (not "Sean"), *Refinements of Selberg's Sieve*, Ph.D.,
Rutgers, May 2010, advisor Henryk Iwaniec. DOI 10.7282/T35T3KJ8, RUcore handle
rutgers-lib/27420. **FREELY available**; downloaded to
research/data/blight_thesis.pdf (367,455 bytes, 75 pages) and read here directly,
2026-08-29. Opera de Cribro p.112 points at it in a free-standing remark, quoted
verbatim from the page scan:

> "We remark that S. Blight (thesis, Rutgers 2010) has sharpened this result
> using a Selberg-type combination but with a Brun lower-bound sieve supported on
> products up to three primes rather than one as in the right-hand side of
> (7.106)."

WHAT IT ACTUALLY CONTAINS (thesis sec. 2.5.2 and 2.6, read first-hand). The
sharpening is Lambda_1 = Lambda^2 Lambda^-, with Lambda^- the three-prime sieve
lambda_1 = 1, lambda_p = -1, lambda_{p1p2} = (4T-2)/(T(T+1)),
lambda_{p1p2p3} = -6/(T(T+1)) for p_i <= z^(1/3), T a positive INTEGER; the
non-negativity rests on the identity sum_{d | n*} lambda_d =
-(m-1)(T-m)(T-(m-1))/(T(T+1)) <= 0, with the explicit remark "if T is not an
integer, this condition does not hold." Her new sifting limits, with
F(t) = 1 - t + c and Maple integration to error < 1e-10:

    kappa = 2:    c = 0.2214971799, T = 16  =>  beta_2   < 4.45
    kappa = 2.5:  c = 0.17,         T = 19  =>  beta_2.5 < 5.455
    kappa = 3:    c = 0.13,         T = 24  =>  beta_3   < 6.458
    kappa = 4:    c = 0.11,         T = 31  =>  beta_4   < 8.47

**VERDICT: NO USE TO US, ON BOTH COUNTS.**

1. AT kappa = 2 IT IS WORSE THAN WHAT WE ALREADY CITE. Her own sec. 2.7 says so:
   "The sieve of Diamond and Halberstam gives a smaller sifting limit for
   kappa = 2 and kappa = 2.5." 4.45 against DHR's 4.266450 - which her own
   sec. 2.2.2 tabulates, quoting Diamond-Halberstam-Richert [1, p.227]. Her
   improvement bites at kappa = 3, 4. This CONFIRMS round 22's second-hand
   reading (harvester 2d) from the primary document.
2. IT IS NOT EXPLICIT. Her Proposition 2.4.2 reads in full: "Let F be a
   continuous piecewise smooth function ... Assume T_F(s) as defined above is
   positive. **Then there is some z_0 such that if z > z_0, then V(D,z) is also
   positive.**" Proof, in its entirety: "As z -> infinity, the error term above
   approaches zero and the main term is positive as stated." The error term is
   O(V(z)^-1 alpha loglog z/log z) with an implied constant inherited from the
   "<<" in her own sifting-dimension hypothesis (sec. 2.8, condition 2) - i.e.
   from our K, unquantified. So the thesis is a DHR-class, non-explicit result.

So opening 1 is closed NEGATIVELY, and the closure is worth stating in the paper:
it extends **the explicitness boundary** (sec. 8) from the DHR differential-delay
system to the Lambda^2 Lambda^- family, first-hand.

BY-PRODUCT, and it is not a coincidence. Blight sec. 2.2.1 records that the beta
sieve has beta_kappa ~ c kappa asymptotically with **c = 3.591... the root of
(c/e)^c = e**. That equation is c(log c - 1) = 1 - exactly the equation defining
our Theorem 3E constant lambda_* = 3.591121, and ODC Theorem 6.12 prints the same
"c = 3.591...". Our quasi-polynomial constant C_infinity = 2 lambda_* = 7.182242
is therefore twice the beta sieve's asymptotic sifting-limit slope. Also from her
table: the beta sieve's own beta_2 = 4.8339865967 (worse than DHR).

### 10b. OPENING 2 - ODC CHAPTER 6 IS EXPLICIT. VERDICT: **YES.**

Chapter 6 is "Brun's Sieve - The Big Bang"; sec. 6.6 is "Improved Bounds for the
Sifting Limits"; **beta_1 = 3.8629..., beta_2 = 7.5941... are on p. 73**, not in
Corollary 6.14 (which is on p. 71 and carries the weaker beta(kappa) < 4 kappa+1).
Page scans of pp. 65, 68-73 and 112 of the AMS printing were read first-hand on
2026-08-29 (Google Books volume Dz6REQAAQBAJ, publisher preview). The objects:

    PROPOSITION 6.7 (p.68)   V^-(D,z) >= {1 - psi^-(a, s-beta) K^(1+1/alpha)} V(z),
                             s >= beta, with a = alpha e^(1+alpha)  (6.67) and
                             beta = 1 + 2(e^(2 alpha/kappa) - 1)^-1  (6.94)
    (6.86)                   psi^-(a, s-beta) < 2 e^-2 (1-a^2)^-1 a^2,  s >= beta
    COROLLARY 6.13 (p.71)    alpha = 1/4:  V^-(D,z) >= {1 - (7/8) K^5} V(z)
                             if s >= beta_kappa;  beta_1 = 4.082, beta_2 = 8.041

**THE DECISIVE POINT.** Proposition 6.7, (6.75), (6.85), (6.86) and Corollary
6.13 carry **no O(.), no "<<", no implied constant and no "for z large"**. The
single inexplicit sentence in the neighbourhood is COROLLARY 6.14, whose statement
says "which means that, for z large" and whose proof says "This follows by (6.89)
**provided K is sufficiently close to one** ... we can depress its size close to
one by choosing a slightly larger value of kappa". That device - and ONLY that
device - is asymptotic. **We do not need it: we buy small K by PRE-SIEVING at
explicit finite cost, which is exactly round 24's 2E'/2E'' machinery.** So the
Chapter 6 apparatus is usable with every constant stated.

Round 24 had priced Theorem 6.9 and Corollary 6.10 from this chapter and recorded
them as "cleaner-looking fallbacks" at s > 28.99 / 37.36 (K = 3). **It had not
priced Proposition 6.7 / Corollary 6.13, and that is where the whole gain sits.**

FIRST-HAND VERIFICATION OF THE HARVEST. Every printed number of sec. 6.6 is
reproduced from the book's own printed formulas by independent code
(research/j2_odc6.py section A): psi^- = 0.8637687819 from BOTH of the book's
renderings (the general formula, and the closed form 2e(16 sqrt e - e^3)^-1)
agreeing to 1e-9; beta_1 = 4.082988 and beta_2 = 8.041623 at alpha = 1/4;
beta_kappa <= 4 kappa + 1 at nine dimensions; alpha^-1 = 3.774952 for (6.97).

**ONE DISCREPANCY IN THE BOOK, RECORDED.** The printed root alpha* = 0.264904
does NOT solve the book's own printed equation
"alpha + (2+3a)/(3+4a) + log alpha + log((3+4a)/(2+3a)) = 0"; the residual there
is -0.001707 and the true root is 0.2652637. The printed alpha* IS internally
consistent with the printed beta_1 = 3.8629 and beta_2 = 7.5941 to within its own
truncation, so this is not an OCR digit error - it is the book's own root-finding,
and the book says how: "A numerical computation gives (**use the Taylor expansion
at 1/4**)". A Taylor approximation about 1/4 is exactly what lands 3.6e-4 short.
CONSEQUENCE, IN OUR FAVOUR: the exact root of the book's own equation gives
**beta_2 = 7.5838, 0.0103 better than the printed 7.5941**. It does not move our
theorem either way, because our binding root is the K -> 1 root below.

### 10c. THE TWO ROUND-24 LEADS ARE ONE EQUATION

Round 24 wrote: "SAME OBJECT as the HR-Memoire item: the remaining mathematics of
the upper ladder is ONE thing seen from two sides." That is now **proved, not
suspected** (j2_odc6.py section B):

* ALGEBRAIC. The Halberstam-Richert Memoire's printed positivity condition
  lambda^2 e^(2 lambda)(2 + e^2) < 1 and ODC (6.86)'s positivity condition
  2 e^-2 a^2/(1-a^2) < 1 with a = lambda e^(1+lambda) are **the same
  inequality**, since lambda^2 e^(2 lambda) = a^2/e^2. Verified identically at
  six values of lambda, max error < 1e-12.
* NUMERICAL. ODC's K -> 1 root is alpha_infinity = **0.253321897**; round 24's
  re-derived HR lambda_* = **0.2533219**. Equal to 5e-7.
* THE LEVEL EXPONENT. HR's printed form is u = 1 + 2.01/(e^lambda_* - 1) =
  7.971954733 (round 24's re-derivation, reproduced here); ODC's is
  beta_2 = 1 + 2/(e^alpha - 1) = **7.937268**. The 2.01 is HR's own safety
  margin. **ODC Chapter 6 is the EXPLICIT form of the 1971 Memoire's theorem,
  and it is very slightly sharper.**

So round 24's "explicitness problem in a known 1971 theorem" is solved by the
observation that the 2010 book already contains the explicit version.

### 10d. THEOREM 2G - the new rung

The beta-sieve's weights satisfy |lambda_d^-| <= 1 (they are the combinatorial
Rosser-Iwaniec weights), so with |r_d| <= 2^nu(d) N_pre after pre-sieving at p_0,

    |R^-(A,D)| <= N_pre sum_{d<D} 2^nu(d) <= N_pre sum_{d<D} tau(d)
                <= N_pre D (log D + 1),

using the elementary sum_{n<=x} tau(n) = sum_{d<=x} floor(x/d) <= x(log x + 1)
(checked against the exact sum to x = 20000). **This is where Chapter 6 beats
Theorem 7.7 twice over**: 2E/2E'' had to carry tau_4, i.e. sum_{d<D} 8^nu(d)
<< C_8 D (log D)^8. Here the log power is 1, not 8.

    THEOREM 2G. Let p_0 = 151, so that K <= 1.0260176 < psi^-(1/4)^(-1/5)
    = 1.0297232 and ODC Corollary 6.13 applies verbatim at alpha = 1/4 with
    delta = 1 - psi^- K^5 = 0.017864. Then for every n with p_n >= 285,

        j_2(p_n#)  <=  C p_n^8.04162 (8.04162 log p_n + 1)(log p_n)^2  +  1,

    with C = N_pre/(0.3905 delta), log_10 C = 57.5, N_pre = prod_{p<151}(p-2).
    Every constant is stated; nothing is asymptotic.

    THEOREM 2G-inf (the constant-free form). j_2(p_n#) <<_eps p_n^(s+eps) for
    every real s > 7.93727, every implied constant computable.

LADDER OF RUNGS, showing the trade of exponent against N_pre and the crossover
against 2E'' (the p_n beyond which 2G is the better bound):

    p0      s (Cor 6.13)  s (Prop 6.7)  log10 C   crossover with 2E''
    151     8.04162       8.02805       57.5      10^5.58
    211     8.04162       8.02742       82.2      10^8.92
    307     8.04162       7.98875       120.0     10^14.15
    601     8.04162       7.96945       243.9     10^31.61
    10007   8.04162       7.94387       4296.8    none in range
    K -> 1  --            7.93727       --        (the floor)

**THE OPERATIVE ROW IS p_0 = 151: exponent 8.04162, crossover at p_n ~ 380,000.**
Below that 2E'' (exponent 15, tiny constant) remains the better bound; above it
2G wins and the margin grows like p_n^6.96. Both rungs stay in the paper.

    p_n        log10 (2E'' bound)   log10 (2G bound)
    285        56.21                80.45
    10^4       81.50                93.51
    10^6       113.27               110.12   <- 2G ahead
    10^10      175.48               142.95
    10^30      480.25               305.21

### 10e. WHAT CHANGES IN THE PAPER

* The headline becomes: "an explicit quasi-polynomial rung, an explicit
  polynomial rung at **exponent 8.042** (and 15 with a far smaller constant, which
  is the better bound below p_n ~ 4e5), and the best-exponent rung 4.266 by
  citation".
* Sec. 4a item (3) is rewritten: the best bound with all constants stated is now
  the exponent-8.042 polynomial, not the exponent-15 one.
* Sec. 8's explicitness boundary GAINS a first-hand datum (Blight) and LOSES its
  pessimism about Chapter 6: the boundary is at the DHR / optimised-Lambda^2Lambda^-
  sifting limits, not at every dimension-2 lower-bound sieve.
* The floor statement changes: FI 7.7's floor was exponent 15 at kappa = 2; ODC
  Ch. 6's floor is **7.93727**, and it is the same number the 1971 Memoire was
  pointing at. Below that needs a genuinely better sieve, not more pre-sieving.
* NAMED OPENING, NEW: the gap between 7.937 (explicit, Ch. 6 beta sieve) and
  4.266 (DHR, not explicit) is now the whole remaining question of the upper
  ladder. ODC's own sec. 6.6 says it "will be superseded by the results of
  Chapter 11" - and Chapter 11's lower-bound constant B is identically zero at
  kappa >= 1/2 (sec. 8), so Chapter 11 is NOT the route. The route, if any, is an
  explicit form of the DHR system.

### 10f. Residual risks, stated

* The page images were read through a browser preview, not held in hand - the
  same caveat as round 24's Theorem 7.7 check. Mitigation: every printed number
  of sec. 6.6 is reproduced from the printed formulas by independent code, so an
  OCR corruption would have to be self-consistent across eight numbers.
* Equation (5.38) (the definition of K) and (6.69) (a condition on kappa quoted
  in Proposition 6.7) were NOT re-fetched this round. (5.38) is the same
  hypothesis we used for Theorem 7.7 in rounds 23-24 and matched against
  Dudek-Dunn Lemma 2.1. (6.69) is unread; our operative alpha = 1/4 is the value
  the book itself uses in Corollary 6.13 "for kappa > 0", so kappa = 2 at
  alpha = 1/4 is inside the book's own applied range. FLAGGED, not assumed.
* p. 74 (the rest of Proposition 6.16, on preliminary sieving) was not obtained;
  our pre-sieving accounting is round 24's own, not the book's.

## 11. UNIT 1 - FINAL ASSEMBLY (round 26, 2026-08-29)

This section is the submission candidate. Everything above is the working
record; where they disagree, this section is current.

**TITLE.** The paired Jacobsthal function: first upper bounds, a first lower
bound from the paired structure, and the structure of its maximisers.

**HEADLINE.** The first proved upper bounds on a function named and conjectured
about since 2017 - an explicit quasi-polynomial rung, an explicit polynomial
rung at exponent 8.04162, and the best-exponent rung 4.266 by citation, with an
honest statement of which constants exist - together with the first proved
lower bounds using the paired structure: `(1.349+o(1)) z log z` at every finite
scale, and `(0.01275+o(1)) z (log z)^3 (lll z)^2/(ll z)^4` asymptotically.

### 11a. THE LADDER, COMPLETE AND IN ORDER

| rung | statement | constants | where |
|---|---|---|---|
| **1** | `j_2(p_n#) <= 2*3^(n-1)/V_n + 1`; explicitly `< 3^(n+1)(log p_n)^2`, n >= 3 | ALL EXPLICIT | sec. 1, Thm 1 |
| **3 / 3E** | `j_2 <= E_K/(V_n - R_K) + 1` for every odd K with `R_K < V_n`; at the optimal K, quasi-polynomial `j_2 < p_n^{9.30 loglog p_n}`, asymptotic constant EXACTLY `2 lambda_* = 7.182242` | ALL EXPLICIT | sec. 1, Thm 3/3E |
| **2E** | `j_2 <= 1.0963e10 p_n^19 (log p_n)^10 + 1`, `p_n >= 285` | ALL EXPLICIT | sec. 1, Thm 2E |
| **2E'** | `j_2 <= 3.5301e9 p_n^17 (log p_n)^10 + 1`, `p_n >= 285` (pre-sieve at 5; free) | ALL EXPLICIT | sec. 9d |
| **2E''** | `j_2 <= 7.2671e11 p_n^15 (log p_n)^10 + 1`, `p_n >= 285` (pre-sieve at 13, `N_pre = 135`) | ALL EXPLICIT | sec. 9d |
| **2G** | `j_2 <= C p_n^8.04162 (8.04162 log p_n + 1)(log p_n)^2 + 1`, `p_n >= 285`, `log10 C = 57.5` | ALL EXPLICIT | sec. 10d |
| **2G-inf** | `j_2 <<_eps p_n^{s+eps}` for every `s > 7.93727`, every implied constant computable | EXPLICIT IN PRINCIPLE | sec. 10d |
| **2** | `j_2 <<_eps p_n^{beta_2+eps}`, `beta_2 = 4.266` | **NOT EXPLICIT, and not makeable so** | sec. 1, sec. 8 |

**WHICH RUNG TO QUOTE.** 2E'' (exponent 15, tiny constant) is the better bound
below `p_n ~ 3.8e5`; 2G (exponent 8.04162, `log10 C = 57.5`) wins above that, by
`p_n^6.96`. **Both are kept and both are labelled.** 2E and 2E' are dominated
everywhere by 2E'' and are retained only because they are the intermediate steps
of the pre-sieving ladder and a referee will want the cost of each.

**THE EXPLICITNESS BOUNDARY, stated once so nobody re-attempts it.** 4.266 is
the numerically-solved output of the DHR differential-delay system; the sieve
inequality at that dimension carries an uncomputed
`O((loglog y)^2 (log y)^{-1/6})`, and even computed, the 1/6 puts
`s = beta_2 + 0.01` at `log y ~ 10^12`. There is no explicit-constant sieve AT
its sifting limit for any `kappa > 1`. Round 25 extended this boundary
first-hand from DHR to the `Lambda^2 Lambda^-` family via Blight's thesis (her
Proposition 2.4.2 is "there is some z_0", the constant inherited from an
unquantified `<<`). **The remaining question of the upper ladder is the gap
between 7.937 (explicit) and 4.266 (not explicit)**, and ODC Chapter 11 is NOT
the route to it - its lower-bound constant B is identically zero for
`kappa >= 1/2`.

### 11b. THE SANDWICH, CURRENT

    PROVED LOWER   h_2 >= (1.349 + o(1)) z log z                     [(P1)]
    PROVED LOWER   h_2 >= (0.01275 + o(1)) z (log z)^3 (lll z)^2/(ll z)^4
                                                                     [(P2')]
    HEURISTIC      ~2.56 z (log z)^2  -- a RANDOM-CHOICE model, and PROVED
                   NOT to be a ceiling (it is exceeded by a full log)
    PROVED UPPER   p_n^{8.04162} explicit  /  p_n^{4.266+eps} by citation

Neither lower bound dominates the other at any `z` a human will evaluate: (P1)
is the one to quote at finite scale, (P2') is the asymptotic statement.
**Two earlier readings are RETRACTED and must not reappear**: "the truth is
`(p^2-p)/2`" (retracted r24 - `c z^2` and `c z log^2 z` fit ZM's 21 values
equally, the residuals drift in OPPOSITE directions, and the discriminating
paired-minus-ordinary exponent gap 0.33-0.75 is the signature of a LOGARITHMIC
separation, not the +1.0 a quadratic-vs-linear law needs) and "the
extreme-value model is the truth" (retracted r25/r26 - it is a competitor to a
lower bound, not a ceiling).

**FALSIFICATION TARGET, and the paper should print it**: one exact `h_2(p_n#)`
beyond `p_n = 73`. The competing readings are now `z(log z)^2` and `z(log z)^3`,
a full `log z` apart, so the computation discriminates more than it did.

### 11c. WHAT THIS PAPER DOES NOT CLAIM (round-26 replacement for section 4a)

Section 4a was written in round 23; its items 1, 3 and 5 are out of date. This
is the current list.

1. **NO PROGRESS ON ZILLER-MORACK CONJECTURE 6.** The proved exponent is 4.266
   (8.04162 with constants); the conjecture asks for 2. Section "THE CEILING"
   argues the gap is parity, not arithmetic: exponent 2 is exactly the level at
   which a survivor in `(y, y^2]` IS a prime pair, so a dimension-2 lower-bound
   sieve there would manufacture two simultaneous primes. Nothing here moves the
   conjecture. **The sifting-limit numbers calibrate the distance and no more:
   4.266 proved, 4 conjectured (Selberg), proved floor
   `beta_kappa >= (1+o(1)) 2 kappa/e ~ 1.47` (Brady 2017). Exponent 2 is NOT
   proved to sit below the sifting limit, only below the conjectured one** -
   round 22's "2 kappa impossibility" paragraph is RETRACTED.
2. **NO NEW SIEVE THEORY.** Theorems 1, 3, 3E are Legendre and Brun with the
   arithmetic done carefully; 2E/2E'/2E''/2G are explicit sieves of
   Friedlander-Iwaniec applied to a dimension-2 problem. **The contribution is
   that the ladder was EMPTY, not that the rungs are hard.**
3. **THE EXPLICIT BOUND AND THE BEST BOUND ARE NOT THE SAME BOUND.** The best
   bound with all constants stated is **2G, exponent 8.04162** (below
   `p_n ~ 3.8e5`, 2E'' at exponent 15 with a far smaller constant); the best
   exponent, 4.266, is not explicit and cannot be made so with published tools.
   Both are stated, and 11a says which is which. *(Replaces 4a item 3, which
   named exponent 19 and then 15.)*
4. **THE COMPUTATIONAL HALF IS REPLICATION PLUS STRUCTURE.** Ziller-Morack's
   companion note arXiv:1706.03668 computes `h_2` to `p_n = 73` and its
   ancillary files list the extremal configurations; our delta reduction is
   essentially their Proposition 1.5(2). What is new is the PER-DIFFERENCE
   family `F_d(y)`, the twin percentile inside it, the shallow-extension cap
   law, and the cross-gear extension ladder - questions they do not ask.
5. **THE LOWER BOUNDS ARE THE PAIRED-STRUCTURE ONES OF 11b, AND THE ORDER
   `z log z` IS NOT NEW.** *(Replaces 4a item 5, which said there was no lower
   bound beyond the collapse transfer.)* Ford-Konyagin-Maynard-Pomerance-Tao,
   arXiv:1802.07604 **Remark 7** names this exact sieving system
   (`I_p = {0 (mod p), 2 (mod p)}`) and records `>> log X log log X` - the order
   of (P1) - as "the 'trivial' bound coming from these methods", without proof
   or constant. **(P1) is the first PROVED bound with an explicit constant and
   the first stated for `h_2`; it is not the first appearance of the order.**
   (P2') is two full powers of `log log X` above that, where the same remark
   hoped only for "a small power of log log X".
6. **NO TWIN-PRIME-GAP COROLLARY.** The same Remark 7 notes that a sieve upper
   bound plus pigeonhole already gives `>> log^2 X` for gaps between actual twin
   primes - which beats anything (P2') implies about them. That argument is
   **no obstruction to a statement about `j_2`** (there the sifted set has
   density `~1/(log x)^2`, so pigeonhole gives only `>> (log x)^2`, beaten by a
   full power of `x`), but any twin-prime-gap corollary would be weaker than an
   argument those authors call trivial. **We claim none.**
7. **NOTHING ABOUT PRIMES.** Every statement here is about coverings of an
   interval by residue classes. The bridge to Goldbach and Polignac is
   Ziller-Morack's Theorem 4.1 and it needs their conjecture, not these bounds.
8. **THE ODC alpha* READING IS A SHARPENING, NOT AN ERRATUM.** See 11d.

### 11d. THE ODC alpha* READING - stated as ours, with the derivation

Round 25 recorded "a discrepancy in the book": ODC sec. 6.6's printed root
`alpha* = 0.264904` does not solve the equation printed beside it. **That
framing is withdrawn.** Round 26 did what the book instructs and the number
comes out exactly:

* Our transcription, from the page image, of the printed equation is
  `f(a) = a + (2+3a)/(3+4a) + log a + log((3+4a)/(2+3a)) = 0`.
* The book says "A numerical computation gives (**use the Taylor expansion at
  1/4**)".
* `f(1/4) = -0.0741009117`, `f'(1/4) = +4.9715909084`, so ONE first-order
  Taylor/Newton step from `a = 1/4` gives
  `1/4 - f(1/4)/f'(1/4) = 0.2649048691` - **the printed `0.264904`, to seven
  digits.**

So the printed value IS the book's own stated approximation, computed the way
the book says it was. What we have is a sharpening of a stated approximation:
the exact root of the same equation is `0.2652636746`, giving
`beta_2 = 7.583827` against the printed `7.594004`, an improvement of
`0.010177`. **And the caveat that belongs with it: `f` is OUR READING of a page
image, so any residual could equally be ours.** Nothing in Theorem 2G moves
either way - 2G's binding root is the `K -> 1` root `alpha_infinity =
0.253321897`. Gate: `research/j2_citesweep.py` section A.

### 11e. SUBMISSION CHECKLIST - state at round-26 close

| item | state |
|---|---|
| Round-23 blockers (ODC 7.7 confirmed; HR Memoire obtained; 19/36 settled) | DISCHARGED (r24) |
| Round-24 openings (Blight; ODC Ch.6 explicitness) | CLOSED (r25) - one negatively, one decisively |
| Sandwich paragraph rewritten off "truth = p^2/2" | DONE (11b) |
| Lower ladder written out with constants | DONE (r26; (P2')) |
| Not-claims section current | DONE (11c) |
| Citation-numbering sweep | GATE, GREEN (`j2_citesweep.py`; caught 2 live defects in r26) |
| Referee pass over every recomputable number | GATE, GREEN (`j2_referee.py`) |
| FKMPT Remark 7 quoted and addressed | DONE (11c items 5-6) |
| Prior art re-checked this round | DONE (2026-08-29; layered-erdos-rankin.md sec. 7) |

**REMAINING, and it is writing rather than research:** (i) the ODC page-image
caveat is unresolved and should be closed by one library visit before
submission - (5.38), (6.69) and p. 74 were never re-fetched; (ii) the paper must
decide whether the per-difference family `F_d` travels with Unit 1 or splits
off; (iii) LaTeX. **Unit 1 is handable to the human as a submission candidate.**
