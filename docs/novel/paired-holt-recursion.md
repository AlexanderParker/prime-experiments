# paired-holt-recursion - exact linear population dynamics for two-residue sieves, with the exposed-set autocorrelation as its transfer-matrix diagonal

Status: SCRIPT-VERIFIED (exact identity, every gap value, four rungs including the
degenerate one-residue case); the proof is a short counting argument (below), not yet
kernel-checked. ROUND-21 UPGRADE: the recursion is now verified at the FULL WORD-CENSUS
level (n_w(M+q') for every word w, not just gap totals): 6714 words exact at
5005 -> 85085 and 10489 at 85085 -> 1616615 (research/paired_hlb.py 4a), and its
eigen-analysis is complete - q-independent Pascal eigenvectors shared with Holt's
matrix, doubled spacing, and the HL-B payoff: see docs/novel/paired-hlb-cycles.md. Prior-art verdict: PARTIAL OVERLAP - the one-residue case is Holt's
Theorem 3.2 (arXiv:1510.00743); the two-residue (paired) recursion, its coefficient
formula, and the identification of its diagonal with the project's autocorrelation law
are NOVEL AS FAR AS SEARCHED. Checked 2026-08-23.

## 1. What it is

Plain language. The project's machines delete TWO residue classes per prime (the two
teeth), where Eratosthenes sieve deletes one. Holt and Holt-Rudd showed that for the
one-residue sieve, the POPULATIONS of gaps and gap-constellations obey an exact linear
dynamical system from one primorial stage to the next - a transfer matrix with known
eigenvalues. This finding is the two-residue analogue: an exact linear recursion that
takes the complete gap-word census of a machine M to the complete gap histogram of
M + q', with coefficients given by a one-line counting formula - and the diagonal of
that recursion is exactly the exposed-set autocorrelation c_q(g) that Lateral derived
in round 19 from an unrelated direction.

Precise form. Let M be a machine with period P and two teeth per gear, and q' a new
gear with tooth set T (|T| <= 2, gcd(P, q') = 1). Write n_w(M) for the number of
occurrences of the word w = (g_1, ..., g_j) as consecutive gaps in M's cyclic gap
sequence, and sigma_i = g_1 + ... + g_i for its interior partial sums. Then for every
gap value g:

    n_g(M + q') = sum over words w with sum(w) = g of  coef(w) * n_w(M),

    coef(w) = #{ r in Z_q' : r not in T,  r + sum(w) not in T,
                 r + sigma_i in T for every i = 1..j-1 }.

The coefficient depends only on the word mod q' - it is position-free - so the map
from the old census vector to the new one is linear with universal coefficients.
Structure of the coefficients:

- j = 1 (a gap surviving as itself): coef(g) = #{r : r not in T, r+g not in T} =
  q'-2 if q' | g, q'-3 if g = +-(t_2 - t_1) mod q', q'-4 otherwise. For twin teeth
  T = {u, -u} the middle case is g = +-2u: THIS IS EXACTLY the round-19 exposed-set
  autocorrelation law c_q'(g). The autocorrelation is the transfer-matrix diagonal.
- Word survival (a length-j word persisting unfused): generically q' - 2(j+1), so
  after normalising by the openings growth A(M+q') = (q'-2) A(M), the eigenvalue
  scale is (q' - 2j - 2)/(q' - 2). Holt's one-residue system has (q' - j - 1)/(q' - 2):
  the paired system contracts TWICE as fast per unit of word length.
- |T| = 1 (the gcd collapse q' | e in the general-difference frame) degenerates the
  same formula to Holt's own one-residue recursion - his theorem is the collapsed
  case of this one.

## 2. Why it might be novel

Holt (arXiv:1510.00743) and Holt-Rudd (arXiv:1408.6002) develop exactly this dynamics
for the ONE-residue sieve: cycle-of-gaps recursion, driving terms, transfer matrix
M(p) with eigenvalues (p-j-1)/(p-2) and p-independent eigenvectors. Their framework
never treats a two-residue deletion, and the project's prior-art sweeps found no
two-residue/paired analogue anywhere (the paired literature - Ziller-Morack - computes
only max-gap values, no population dynamics). What is new here beyond Holt:

- the paired coefficient formula and its three-case diagonal, which turns out to BE
  the project's independently derived autocorrelation law (two constructs from two
  workstreams identified as one object);
- the factor-2 spectral contraction (q'-2j-2 vs q'-j-1): a structural statement about
  WHY paired (twin-type) censuses decorrelate faster than single-residue ones;
- the degenerate-case unification: one formula containing both Holt's sieve dynamics
  and the paired dynamics, switching on |T|.

What it is NOT: not the merge law restated. The merge law (docs/novel/merge-law.md)
is the MAX-GAP readout of the same cycle recursion; this is the full population-level
linear system. Together they give the project both the extreme and the distribution
of the new machine from the old word alone.

## 3. Proof / verification

Status: SCRIPT-VERIFIED (exact, finite) at four rungs; paper proof is elementary.

Proof sketch (the counting argument). In the merge construction, M + q' is q'
concatenated copies of M's cycle with the new gear's kills applied. Because
gcd(P, q') = 1, an old opening at position o is killed in exactly |T| copies - one
for each tooth t: the unique lap k with o + kP = t mod q'. Parameterise copies by
r = (o + kP) mod q': a maximal run g_1..g_j fuses into a single new gap in a given
copy iff every interior opening is killed there (r + sigma_i in T) while both
flanking openings survive (r not in T, r + sum not in T). Counting admissible r in
Z_q' gives coef(w); summing over occurrences gives the identity, and no position
dependence enters because r ranges over all of Z_q' for every occurrence.

Verification (research/paired_holt_recursion.py, all assertions pass; output at
research/data/paired_holt_recursion.out): the full histogram identity - every gap
value g, exact equality of predicted vs directly-constructed populations - at

- slot frame [5,7,11,13] -> +17 (period 5,005 -> 85,085), 17 gap values;
- slot frame [5,7,11,13,17] -> +19 (85,085 -> 1,616,615), 23 gap values;
- general-difference frame e = 344 (a 13-winner class), [3,5,7,11,13] -> +17,
  T = {0, -e}, 22 gap values;
- the collapse e = 102 with 17 | e, T = {0}: the one-residue Holt case, 26 gap
  values - same formula, still exact.

Plus: the j = 1 diagonal equals the c-law in all cases, and the word-survival
diagonals match q' - 2(j+1) generically (observed value sets printed per length).

## 4. Implications

Inside the project:

- The HUMAN DIRECTIVE's transfer-matrix frame now has its exact object: the round-19
  anti-correlation deficit (x26, x6.7, x1400) and Constructor's p_j target can be
  restated on this recursion - the deficit is a spectral property of a matrix whose
  entries are KNOWN (the coef formula), not fitted. The factor-2 contraction
  (q'-2j-2 vs Holt's q'-j-1) is the first structural quantity of that kind.
- Lateral's c_q(g) is the diagonal; Lateral's round-20 target c_q(g1,g2) is the
  length-2 block of the same matrix (coef of two-letter words). Three lanes'
  constructs are one object.
- The round-19 histogram residue law (richest classes at +-2u of gears 5 and 7, the
  log(c_5 c_7) variance reduction) stops being a fit: the recursion makes the
  histogram's arithmetic structure a product of known diagonals plus fusion inflow.
- Formalist: coef position-freeness and any fixed rung of the identity are finite
  and kernel-checkable; the statement mentions only words and residues.
- Import path from Holt not yet used: his eigenvector analysis (p-independent,
  binomial/Pascal structure) and driving-term asymptotics - if the paired matrix has
  the analogous p-independent eigenvectors, the asymptotic census ratios of every
  gap value follow in closed form, which is Mechanic's histogram data as a theorem.

Outside: Holt's program gets its two-residue extension; the paired case is the one
that talks to twin primes, Polignac, and the Ziller-Morack function (the max-gap of
the object whose distribution this recursion controls).

## 5. Unsolved questions or conjectures it touches

- The project's lemma (D) / suppression law: the anti-correlation deficit as a
  spectral gap of the truncated paired transfer matrix - open, now well-posed.
- Hardy-Littlewood Conjecture B analogue in cycles (Holt proves it one-residue,
  Theorem 5.5): the paired version - do normalised paired-gap populations converge
  to the HL pair-correlation ratios? The recursion is the tool; not yet run.
- Ziller-Morack h_2: this recursion plus the merge law is the incremental machinery
  behind the project's replication of their table (docs/novel/merge-law-h2-test.md).
- Eigenvector p-independence for the paired matrix: conjectured by analogy with
  Holt, untested.

## 6. Prior-art check (2026-08-23)

Searches run this round (WebSearch + WebFetch, full text where listed):

- arXiv:1510.00743 (Holt, "Combinatorics of the gaps between primes") READ IN
  DETAIL via ar5iv: cycle recursion (Lemma 2.1: concatenate p copies, close at
  elementwise product positions), population recursion (Theorem 3.2:
  n_{s,j}(p'#) = (p'-j-1) n_{s,j}(p#) + n_{s,j+1}(p#)), transfer matrix with
  diagonal (p-j-1)/(p-2), superdiagonal j/(p-2), p-independent binomial
  eigenvectors, Polignac-in-the-sieve (Thm 5.5), HL Conjecture B ratios. ALL
  one-residue. The paper explicitly does not track maximal gaps.
- arXiv:1408.6002 (Holt-Rudd) abstract + structure: same framework, earlier.
- WebSearch `"transfer matrix" OR "dynamical system" sieve gaps primorial coprime
  residues eigenvalues recursion -Holt`: no non-Holt transfer-matrix sieve
  literature (hits are Holt's own papers, unrelated physics transfer matrices,
  and control-theory coprime factorisation).
- WebSearch `"cycle of gaps" OR "gaps among generators" recursion sieve "two
  residues" OR "residue pair" OR twin constellation population dynamics
  primorial`: only Holt's papers and one unreviewed Zenodo preprint (Ojaroudi
  2026, claimed unconditional twin-prime theorem via a "replication-deletion
  primorial sieve") - fetched and assessed: no exact linear population recursion,
  no explicit transfer coefficients, claim class far beyond its method; not
  substantive prior art for this object.
- Ziller-Morack arXiv:1706.00317 + 1706.03668 (read in full this round): paired
  max-gap VALUES only; no population dynamics, no recursion.

Nearest prior art: Holt's Theorem 3.2 - the |T| = 1 degenerate case of this
recursion. VERDICT: PARTIAL OVERLAP (one-residue case known and due to Holt; the
two-residue recursion, coefficient formula, autocorrelation-as-diagonal
identification, and factor-2 spectral contraction NOVEL AS FAR AS SEARCHED).

## CORRECTION (round 22, harvester, self-caught)

Holt arXiv:2502.20470 "Eratosthenes sieve supports the k-tuple conjecture" (Feb 2025,
v3 Jul 2025) postdates the round-20 prior-art sweep behind this document and narrows
its claim. A twin-slot survivor is exactly a gap of 2 in Holt's cycle of gaps, so a
paired gap word of length j is one of his constellations with 2j+2 boundary points;
his Corollary 1 gives sum_{j>=J} n_{s,j}(p#) = prod_q (q - nu_q(s)) for every
constellation, and his population recursion carries the diagonal p - (number of
points). Consequently the coefficient structure reported here - diagonal
q' - 2j - 2, i.e. "Holt's system with doubled level spacing" - is a consequence of the
point count in his general dynamics, not a separate phenomenon. What remains specific
to this document is the explicit position-free coefficient formula
coef(w) = #{r in Z_q' : flanks alive, interiors in T} and its exact verification at
the full word-census level (6714 + 10489 words at two rungs), which is a computational
statement about the two-residue readout rather than a new dynamical law.
See docs/novel/paired-hlb-cycles.md section 0 for the full correction.
