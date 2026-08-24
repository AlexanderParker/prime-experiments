# j2-upper-bound - the first upper bounds on the paired Jacobsthal function j_2

Status: PROVED (paper proof below, elementary; explicit constants script-verified,
research/j2_bound.py all assertions green) for the main theorem; PROVED-BY-STANDARD-
CITATION (fundamental lemma of sieve theory, dimension 2) for the polynomial
corollary. Prior-art verdict: NOVEL AS FAR AS SEARCHED - the published upper-bound
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
first two rungs.

Precise form. Following Ziller-Morack (arXiv:1706.00317, Def. 2.1-2.2): j_2(n) is
the smallest m such that every paired progression <a,b>_m = {(a+i, b+i) : i=1..m}
with 2 | b-a contains a pair (x,y) with gcd(x,n) = gcd(y,n) = 1; h_2(n) = j_2(p_n#).

THEOREM 1 (elementary; the first sub-primorial bound). For every n >= 2,

    j_2(p_n#)  <=  2*3^(n-1) / V_n  +  1,      V_n = (1/2) * prod_{3<=p<=p_n} (1 - 2/p),

and explicitly

    j_2(p_n#)  <  3^(n+1) * (log p_n)^2        for all n >= 3     (n = 2: bound = 37).

Since 3^n = exp(n log 3) with n ~ p_n / log p_n, this is exp(O(p_n / log p_n)) -
genuinely below the trivial p_n# = exp((1+o(1)) p_n).

THEOREM 2 (polynomial, by the fundamental lemma of sieve theory). There is an
absolute constant beta_2 (the dimension-2 sifting limit; beta_2 <= 4.85 via the
beta sieve, < 4.45 via Diamond-Halberstam-Richert-type refinements) such that

    j_2(p_n#)  <<_eps  p_n^(beta_2 + eps).

The conjectured truth (ZM Conjecture 6 + the project's measured ~(p^2-p)/2 share)
is exponent 2; the first proved exponent lands under 4.5.

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

Why the ladder is empty is itself worth recording: for the ORDINARY function,
Iwaniec's j(n) << (k log k)^2 is order p_n^2 at primorials - the SAME order as ZM's
conjectured paired bound. A paired theorem of Iwaniec's literal strength would
land within a constant of a statement implying Goldbach and Polignac (ZM Thm 4.1),
i.e. it is parity-critical. The sub-conjecture rungs (3^n, p^4.5) are parity-safe,
and nobody had bothered to write them down.

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

THEOREM 2 (sketch, standard). Sift the interval [1, m] by the classes Omega_p,
p <= z = p_n. The sieve problem has dimension kappa = 2 (omega(p) <= 2), remainders
|r_d| <= omega(d) <= 3^(nu(d)), so sum_{d < D} |r_d| << D (log D)^2. The
fundamental lemma (Halberstam-Richert Thm 2.5; Friedlander-Iwaniec Opera de Cribro
Thm 6.9; beta sieve with beta(2) <= 4.85, DHR refinement < 4.45 per Blight) gives
S >= (1/2) m V(z) > 0 once D = z^(beta_2 + eps) and m >= D log^4 z. Hence any
fully-bad run has length << z^(beta_2 + eps). QED (by citation).

Verification: research/j2_bound.py (all assertions green; output
research/data/j2_bound.out): (A) the counting inequality on 8000 real windows;
(B) bound values dominate the exact ZM h_2 table at all 20 known points - the
honest price is x6 at p = 3 growing to x1.3e8 at p = 73 (a Legendre-type bound is
exponentially lossy; that is what rung 2 fixes); (C) the explicit-form inequality
with exact V_n through n = 4203 plus the monotone twin-constant check; (D) the
b-a = p# collapse, exact at n = 3, 4, 5.

## 4. Implications

Inside the project: none directly on the twin route - this is Harvester lane
(N4 executed). It prices what the machinery's exact table sits under: exact values
h_2 grow like ~p^2/2 while the first proved ceiling is p^4.5; the gap between
exponent 4.5 and exponent 2 is the paired parity wall made quantitative.

Outside: the paired upper-bound problem now has a nonempty ladder with named next
rungs: (i) improve 3^n by Brun's pure sieve (quasi-polynomial, elementary);
(ii) the beta_2 constant (any dimension-2 sifting-limit improvement transfers
verbatim); (iii) an Iwaniec-method paired bound - order p^2 (log ratio)^2 - which
would meet ZM Conjecture 6 up to constants and is therefore parity-critical, i.e.
almost certainly not reachable by published methods; recorded as the honest wall.

## 5. Unsolved questions or conjectures it touches

- Ziller-Morack Conjecture 6 (j_2(p_n#) < p_n^2 - p_n): Theorem 2 is the first
  proved statement of the same shape (polynomial in p_n); the conjecture's
  exponent 2 vs proved 4.5.
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

Nearest prior art: (i) the one-residue ladder (Kanold 1967 2^k; Stevens 1977;
Iwaniec 1978 (k log k)^2) - different function, methods one-class; Theorem 1 is
the Kanold-analogue, Theorem 2 the Stevens-analogue, the Iwaniec-analogue is open
and parity-critical; (ii) the trivial period bound j_2 <= p_n# implicit in
periodicity. VERDICT: NOVEL AS FAR AS SEARCHED (the statements are new; the
methods are deliberately classical - the contribution is the first occupied rungs
of an empty ladder, with the honest observation of why it was empty).
