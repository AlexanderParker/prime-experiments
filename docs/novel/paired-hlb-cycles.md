# paired-hlb-cycles - Hardy-Littlewood Conjecture B in paired cycles: the twin-slot gap population n_g, its exact Bonferroni series, and effective Polignac in the paired sieve (round-22 title: the earlier one claimed the local-factor and doubled-spacing statements, which section 0 gives back to Holt)

Status: PROVED (paper proofs below, elementary) + SCRIPT-VERIFIED (exact, at scale:
research/paired_hlb.py, all assertions green; output research/data/paired_hlb.out).
The LOCAL-FACTOR IDENTITY c_q(g) = q - nu_q({0,2,6g,6g+2}) is KERNEL-CHECKED since
round 22 at all four gears of machine 13 for every g < 40 -
`DepthSum.local_factor_5/7/11/13` (proofs/DepthSum.lean), NO AXIOMS - together with
its product form over the whole 5005-slot period (`DepthSum.depth_sum_hl_form`):
the machine's lag-g pair population IS prod_q (q - nu_q), exactly.
Prior-art verdict, CORRECTED IN ROUND 22 AND NARROWED: Holt arXiv:2502.20470
(Feb 2025, v3 Jul 2025 - a paper that did not exist when this line was first searched)
CONTAINS the local-factor identity and the depth-sum identity as special cases of his
Corollary 1. See section 0 (CORRECTION) and section 6. What survives as novel as far
as searched: the paired sieve's OWN gap sequence n_g and the pinch bounding it, its
Bonferroni series and moment identity, and the effective threshold y_0(g).
Checked 2026-08-24.

## 0. CORRECTION (round 22, self-caught) - Holt arXiv:2502.20470 contains two of
## these results, and explains a third

Round 21 checked Holt arXiv:1510.00743 and Holt-Rudd arXiv:1408.6002 and found the
paired statements absent.  Round 22's re-search surfaced a paper that did not exist at
that time: Fred B. Holt, "Eratosthenes sieve supports the k-tuple conjecture",
arXiv:2502.20470 (v1 Feb 2025, v3 Jul 2025).  Its Corollary 1 reads

    for an admissible constellation s of length J,
        sum_{j >= J} n_{s,j}(p#)  =  prod_{q <= p} (q - nu_q(s)),
    nu_q(s) = the number of distinct residues mod q among the J+1 boundary points,

i.e. the aggregate population of s AND ITS DRIVING TERMS.  Consequences for this
document, stated plainly:

- THE LOCAL-FACTOR IDENTITY IS HOLT'S, SPECIALISED.  A twin-slot survivor is exactly a
  gap of 2 in Holt's cycle of gaps, so a pair of twin-slot survivors at lag g is an
  instance of his constellation s = (2, 6g-2, 2), whose boundary points are
  {0, 2, 6g, 6g+2} = H_g.  Then c_q(g) = q - nu_q(H_g) is precisely his q - nu_q(s).
  The two-line affine-bijection proof below is still the right proof of the closed
  form (q-2 / q-3 / q-4 by divisibility), but the identification of the machine's
  autocorrelation with an HL local factor is his framework, not new.
- THE DEPTH-SUM IDENTITY IS HOLT'S COROLLARY 1.  sum_j W_j(g) = N2(g) is exactly the
  displayed formula at s = (2, 6g-2, 2): "s and its driving terms" is precisely "the
  boundary points are open, interiors arbitrary".  This is a correction to
  docs/novel/depth-sum-identity.md (Lateral, round 20) as well, flagged to that lane -
  the identity and its proof are correct, the novelty claim is not.
- THE DOUBLED SPACING IS NOW DERIVED RATHER THAN OBSERVED.  A paired gap word of
  length j is a constellation with 2j+2 boundary points, and Holt's population
  dynamics carries the diagonal q - (number of points); q - (2j+2) = q - 2j - 2 is the
  diagonal measured here, against q - (j+1) for one-residue words of length j.  So
  "the paired system is Holt's with doubled level spacing" is not a coincidence to be
  reported but a one-line consequence of the point count - a better statement than the
  round-21 one, and a weaker novelty claim.

ALL OF THE ABOVE IS ASSERTION-CHECKED (research/holt_correspondence.py, green):
(A) the twin-slot survivors of the project's machine ARE exactly the left endpoints of
the gaps of 2 in the cycle of rough numbers (1,485 of them at P = 30,030; 22,275 at
P = 510,510, sets equal); (B) N2(g) = prod_q c_q(g) equals, to the unit, the count of
positions where the four boundary points of s = (2, 6g-2, 2) are rough - i.e. Holt's
right-hand side - at every g <= 6 and both machines; (C) the two population objects
separate immediately, e.g. at machine 17, g = 5: n_g = 4,230 while Holt's n_{s,J} = 0
(his constellation forbids ANY rough number in the 6g-2 span, ours forbids only twin
candidates).

WHAT SURVIVES, and why it is a different object.  Holt's n_{s,J} counts instances of a
constellation with NO ROUGH NUMBER between the boundary points.  The paired sieve's
gap population n_g counts consecutive twin-slot survivors - no TWIN-CANDIDATE between
the boundary pairs, while ordinary rough numbers in between are allowed.  The twin-slot
gap sequence is a derived sequence of Holt's cycle that he does not study, and n_g is
not any of his n_{s,J}.  Everything this document proves ABOUT n_g - the pinch, its
identification as Bonferroni order 1 of an exact alternating series, the moment form
S_k = sum_j C(j-1,k) W_j, and the effective threshold y_0(g) - is about that object and
has no counterpart found in his papers (which give no effective threshold at all).

## 1. What it is

Plain language. The project's twin-slot machines delete two residue classes per
gear; their surviving slots are twin-candidate pairs. Three exact results about the
POPULATIONS of gaps between consecutive survivors:

(i) each gear's effect on a fixed gap value is numerically identical to the
Hardy-Littlewood local factor of a prime QUADRUPLET; (ii) as gears accumulate, the
relative populations of any two fixed gap values converge - provably, with an
explicit rate - to the ratio of Hardy-Littlewood quadruplet constants: Conjecture B
holds INSIDE the sieve, in its paired form; (iii) the exact linear dynamics driving
this (round-20's paired Holt recursion) diagonalises with the SAME q-independent
Pascal eigenvector matrix as Holt's one-residue system, with eigenvalue spacing
doubled. This delivers the eigen-analysis flagged as N5, and settles what the
paired recursion "derives" in the sense Holt's one-residue version derives HL-B.

Precise forms (slot frame; gears q >= 5 prime, teeth T_q = {u, -u}, u = 6^{-1} mod q;
machine M_y = all gears 5..y, period P = prod q, openings = surviving slots).

LOCAL FACTOR IDENTITY. For every prime q >= 5 and every g >= 1,

    c_q(g) := #{r in Z_q : r, r+g not in T_q}  =  q - nu_q(H_g),

where H_g = {0, 2, 6g, 6g+2} and nu_q counts its distinct residues mod q. So
c_q(g) = q-2 iff q | 6g, q-3 iff q | 6g -+ 2, q-4 otherwise: the exposed-set
autocorrelation law (round 19) is exactly the Hardy-Littlewood local factor of the
prime quadruple (p, p+2, p+6g, p+6g+2). Proof: the forbidden r are
{u, -u, u-g, -u-g}; multiplying by the unit -6 and shifting by +1 maps this set to
{0, 2, 6g, 6g+2}; affine bijections preserve distinctness. QED. (Asserted for all
q < 2000, g in 1..59 plus boundary cases.)

PAIRED HL-B IN CYCLES (the pinch theorem). Let n_g(M) be the population of gap
value g in M's cyclic gap sequence, N2(g) = prod_{q<=y} c_q(g) (the CRT count of
opening pairs at lag g), N3(0,t,g) = prod_q c_q({0,t,g}) (triples; c_q(X) =
q - |union_x (T_q - x)|). Then, exactly,

    N2(g) - sum_{t=1}^{g-1} N3(0,t,g)   <=   n_g(M)   <=   N2(g).

Upper: the depth-sum identity sum_j W_j(g) = N2(g) (Lateral, round 20, proved)
with n_g = W_1. Lower: every j>=2 window summing to g has an interior opening at
some offset t in (0, g); the union bound over t majorises sum_{j>=2} W_j. QED.
Both sides are closed-form CRT products - no scan at any scale. Since
N3/N2 ~ prod (q-6)/(q-4) -> 0 like 1/(log y)^2, and since c_q(g)/c_q(g') = 1 for
every q > 6 max(g,g') + 2, the population ratio of two fixed gaps converges with
explicit rate to a FINITE product:

    n_g(M_y) / n_g'(M_y)  ->  S(g)/S(g'),   S(g) = prod_{5<=q<=6g+2} (q - nu_q(H_g))/(q-4)

= the ratio of Hardy-Littlewood quadruplet singular series for
(p, p+2, p+6g, p+6g+2) vs (p, p+2, p+6g', p+6g'+2) (the p = 2, 3 factors are
g-independent and cancel). Verified: the pinch exact at machines 13/17/19 by full
sieve for every g <= 26; the interval closing onto the HL target through y = 10^6
by pure CRT products, e.g. n_5/n_4: target 3.150, interval [3.06, 3.22] at
y = 10^6; correction sums falling 0.20 -> 0.02 (g=4) across y = 10^2..10^6.

EIGEN-ANALYSIS (the dynamical route). Aggregating word counts by (sum s, length j),
m_{s,j}(M), the paired Holt recursion is generically

    m_{s,j}(M + q')  =  (q' - 2j - 2) m_{s,j}(M)  +  2j m_{s,j+1}(M)  +  sporadic,

the exact two-residue analogue of Holt's (p-j-1) diagonal + j superdiagonal
(sporadic = the finitely many residue coincidences mod q'; measured share 6.9% at
the +17 rung, carried exactly by the word-level transfer below). The matrix
A_q = diag(q-2j-2) + superdiag(2j) has eigenvector matrix

    v^(k)_j = (-1)^(k-j) C(k-1, j-1)     (inverse Pascal),

INDEPENDENT of q and IDENTICAL to the eigenvectors of Holt's one-residue matrix
diag(q-j-1) + superdiag(j): solving (A - lambda_k)v = 0 gives
2(k-j) v_j = -2j v_{j+1} in the paired case and (k-j) v_j = -j v_{j+1} in Holt's -
the factor cancels, only the binomial recursion survives. THE PAIRED SYSTEM IS
HOLT'S SYSTEM WITH DOUBLED LEVEL SPACING: same diagonalising frame, eigenvalues
(q-2j-2)/(q-2) vs (q-j-1)/(q-2). Verified in exact rational arithmetic for
q in {17, 19, 101, 997}, k <= 12, both systems.

WORD-LEVEL TRANSFER (the recursion upgraded from gap counts to the full word
census). For a word W occurring in M and a copy residue r in Z_q', the fate of W
is deterministic: opening at offset sigma killed iff r + sigma in T; if both ends
survive, W fuses to the image word (splits at surviving interiors). Summing
n_W(M) * #{r : image(W,r) = w, ends alive} over W with sum(W) = sum(w) gives
n_w(M + q') EXACTLY - verified for every one of 6714 words (sum <= 24) at
5005 -> 85085 and 10489 words at 85085 -> 1616615. Round 20 verified the gap-level
readout; the full census-to-census linear map is now verified.

## 2. Why it might be novel

Holt proves the one-residue statements: HL-B ratios in cycles (1510.00743 Thm 5.5),
the Pascal eigenvector structure, and - in the paper found only in round 22 -
constellation populations with the local factor q - nu_q(s) and the aggregate identity
(2502.20470 Cor. 1).  After the section-0 correction the remaining deltas are:

- THE OBJECT.  n_g is the gap population of the TWIN-SLOT SUBSEQUENCE of Holt's cycle
  (consecutive twin candidates, ordinary rough numbers allowed in between).  It is not
  any n_{s,J} of his, and the derived sequence is not studied in his papers.
- THE PINCH, and its round-22 completion: n_g = sum_k (-1)^k S_k exactly, with
  Bonferroni alternation and the moment form S_k = sum_j C(j-1,k) W_j, so the
  round-21 two-sided bound is orders 0 and 1 of an exact series whose slack is the
  explicit quantity sum_{j>=3} (j-2) W_j.
- THE EFFECTIVE THRESHOLD y_0(g) = exp(Theta(sqrt g)): Holt proves constellations
  "arise and persist" but gives no stage index; this is a number, for the paired
  object, at every g.
- NOT a delta any more: the local-factor identity, the depth-sum identity, and the
  doubled level spacing (see section 0).

What it is NOT: not a statement about primes (all populations live in the sieve;
crossing to primes is the usual HL wall, untouched); not the depth-sum identity
restated (that identity is one ingredient of the lower pinch).

## 3. Proof / verification

Proofs: section 1 carries the two short proofs (affine bijection; depth-sum +
union bound) and the eigenvector recursion. Script: research/paired_hlb.py, four
assertion families, all green (2026-08-24): (1) identity at all q < 2000;
(2) pinch vs exact sieves, machines 13/17/19, every g <= 26; (3) convergence
tables to y = 10^6 with the HL target inside every interval; (4a) word-level
transfer exact at two rungs; (4b) aggregated-law sporadic share; (4c) eigenvectors
in exact rationals. Kernel-checkable pieces (finite, named for Formalist if ever
wanted): the identity for fixed q; one rung of the word-level transfer; the
eigenvector identity for fixed size.

## 3a. Round-22: THE PINCH IS BONFERRONI ORDER 1 - the full alternating series

(research/pinch_bonferroni.py, all assertions green; output
research/data/pinch_bonferroni.out.)

The round-21 pinch turns out to be the first two truncations of an exact series.  For
0 < t_1 < ... < t_k < g write N_{k+2}(0,t_1,...,t_k,g) = prod_q c_q({0,t_1,...,t_k,g}),
the closed-form CRT count of positions at which all k+2 listed offsets are open, and

    S_k  =  sum_{0 < t_1 < ... < t_k < g}  N_{k+2}(0, t_1, ..., t_k, g),   S_0 = N2(g).

THEOREM (exact).   n_g  =  sum_{k >= 0} (-1)^k S_k,   and the Bonferroni truncations
alternate: sum_{k<=K} (-1)^k S_k is an UPPER bound on n_g for even K and a LOWER bound
for odd K.  Proof: inclusion-exclusion over which interior offsets are open, applied to
"the pair at lag g has NO interior opening".  K = 0 and K = 1 are exactly the two sides
of the round-21 pinch.

MOMENT FORM (the depth-window reading, and the reason the pinch was lossy).  Since a
depth-j window at lag g has exactly j-1 interior openings,

    S_k  =  sum_j C(j-1, k) W_j(g),

so S_0 = sum_j W_j = N2 is the depth-sum identity, and S_1 = sum_j (j-1) W_j
OVERCOUNTS sum_{j>=2} W_j by exactly sum_{j>=3} (j-2) W_j.  The pinch's slack is
therefore an explicit quantity, not an unknown, and the higher orders remove it.
Both identities verified exactly by full sieve at machines 13 and 17 for
g = 4, 5, 6, 8, 10 and k <= 3, together with the alternation.

HOW MUCH IT BUYS.  Order 3 tightens the effective Polignac threshold of section 3b:

    g            6    8   10    12    15    20     25     30
    y_0 order 1  41   53   67   103   199   467   1009   2609
    y_0 order 3  41   53   67    79    97   127    167    367

- never worse, and log y_0 falls by up to a factor 1.35.  The exp(Theta(sqrt g))
SHAPE survives (log y_0/sqrt g moves from ~1.44 to ~1.08 but stays flat), so the
square root is NOT a union-bound artifact - it is the real behaviour of this method.

## 3b. Round-22: what the pinch buys outside the sieve, and exactly where it stops

(research/hlb_effective.py, all assertions green; output research/data/hlb_effective.out.)

EFFECTIVE POLIGNAC IN THE PAIRED SIEVE (the positive consequence).  Define y_0(g) as
the least y for which the pinch lower bound is positive.  Then for EVERY y >= y_0(g)
the gap value g occurs in M_y's cyclic gap sequence - unconditionally, with no scan,
and with y_0(g) an explicit number rather than an asymptotic.  Holt's one-residue
Theorem 5.5 has no effective form; this one does, because both sides of the pinch are
closed-form CRT products.  The computation splits at q = 6g+2, beyond which every
local ratio is generic:

    rho(y,g) = sum_t prod_q c_q({0,t,g})/c_q({0,g})  =  A(g) * B(y)/B(6g+2),
    A(g) = sum_{t=1}^{g-1} prod_{q <= 6g+2} (ratio),   B(y) = prod_{5<=q<=y} (q-6)/(q-4),

and y_0(g) is read off one monotone table.  Measured:

    g        2    3    4    5    6    8   10    12    15    20    25     30
    y_0(g)  14   20   26   32   38   50   62   103   199   467  1009   2609
    g        40      50      60       80        100
    y_0(g) 12157   42257   96401   882061   4424687

with log y_0(g)/sqrt(g) confined to [1.305, 1.531] over g in [10, 100]:

    THE EFFECTIVE THRESHOLD IS y_0(g) = exp(Theta(sqrt(g))) - not polynomial in g.

(The sqrt is structural: the correction sum has g-1 terms each of size ~ C/(log y)^2,
so positivity needs log y >~ sqrt(g); section 3a confirms it is not an artifact of the
union bound by recomputing at Bonferroni order 3.)  Every gap up to
g = 2, 3, 4, 5, 6, 8, 10 is already guaranteed at y = 6g+2 itself.

THE MAX-GAP CONSEQUENCE, PRICED (a negative, recorded so it is not re-derived).
Since every gap <= G(y) := max{g : y_0(g) <= y} occurs, the sieve's maximal gap - which
IS the paired Jacobsthal value F(2,y) at the twin difference - satisfies
F(2,y) >= 3 G(y) ~ c (log y)^2: 60, 90, 180, 240 at y = 10^3..10^6.  The truth is of
order y^2 (F(2,37) = 264 already).  So the pinch contributes NOTHING to the j_2 lower
ladder, which stays with the Ford-Green-Konyagin-Maynard-Tao transfer
(docs/novel/j2-upper-bound.md).

THE BOUNDARY (the honest statement of what does NOT transfer).  The pinch is a
FULL-PERIOD statement.  Primality of survivors lives in the window (y, y^2] - the
project's own horizon theorem - and that window is a share

    y^2 / P_y  =  exp( -(1+o(1)) y )

of the period: 2.2e-4 at y = 19, 1.1e-9 at y = 37, 2.6e-34 at y = 101.  No
full-period population statement, however exact, localises into a share that thin.
That is the entire distance between "paired HL-B in cycles, proved with an explicit
rate" and "paired HL-B for primes, open"; it is the same short-interval wall the
whole subject has, stated in the machine's own coordinates.  In particular this
document proves NOTHING about prime quadruplets, and no unconditional prime-side
consequence has been found from it.

## 4. Implications

Inside the project: the machine's histograms now have a THEOREM layer - every
fixed-gap population is pinched by two closed-form CRT products, at every scale,
no scan (complements Mechanic's COV-SAT from the exact side and turns the "no
smooth law, only the histogram" verdict into: histogram = HL quadruplet
arithmetic x vanishing interior correction). The paired transfer matrix is now
DIAGONALISED - Constructor's spectral restatement of the anti-correlation deficit
can use the exact eigenbasis, not just the eigenvalues. The doubled spacing is
the structural reason paired objects decorrelate twice per rung.

Outside: Holt's program acquires its two-residue extension in full (recursion:
round 20; diagonalisation + HL-B payoff: this round). The quadruplet
identification connects the Ziller-Morack function's underlying sieve directly to
the Hardy-Littlewood 4-tuple constants.

## 5. Unsolved questions or conjectures it touches

- Hardy-Littlewood Conjecture B (and the 4-tuple conjecture): proved here inside
  the paired sieve at every finite level with rate; the prime-side statement is
  untouched (the transfer from sieve populations to prime populations is the
  whole HL problem).
- The project's lemma (D): the deficit measured there is the deviation of
  QUALIFYING-gap correlations from this document's baseline; the exact eigenbasis
  is the right frame to pose it in.
- Holt's open asymptotics (driving-term analysis): the paired version inherits
  them with doubled spacing.
- Polignac in the sieve: the round-22 threshold y_0(g) = exp(Theta(sqrt(g)))
  makes the paired in-sieve statement effective.  The union bound behind it is
  Bonferroni order 1 and IS lossy, but order 3 only improves the constant, not the
  shape (section 3a) - so whether the TRUE threshold is polynomial in g remains open,
  and would need a genuinely different argument, not a higher order.

## 6. Prior-art check (2026-08-24)

- WebSearch `gaps between "twin" coprime residues primorial cycle Hardy-Littlewood
  ratios population "cycle of gaps" two residues`: hits are Holt 1510.00743 and
  Holt-Rudd 1408.6002 (one-residue; the known nearest art) plus unreviewed
  "Prime Generator Theory" preprints (no exact recursion, no paired dynamics -
  not substantive prior art).
- Round-20 sweeps (recorded in harvester.md / paired-holt-recursion.md):
  transfer-matrix sieves excluding Holt - nothing; Ziller-Morack full-text - no
  population dynamics; Ziller 2007.01808 - single-residue gap spectra only.
- Holt Thm 5.5 read in detail (round 20, ar5iv): his HL-B-in-cycles is
  one-residue 2-point; his eigenvectors are the same binomial family (which is
  exactly the overlap claimed).

ROUND-22 searches (WebSearch/WebFetch, 2026-08-24) - these changed the verdict:

- `Fred Holt "cycle of gaps" primorial sieve arXiv 1510.00743 published journal Holt
  Rudd` -> surfaced two Holt papers POSTDATING the round-20/21 sweeps:
  arXiv:2502.20470 "Eratosthenes sieve supports the k-tuple conjecture" (Feb 2025)
  and arXiv:2603.25915 "Surviving Eratosthenes sieve I" (Mar 2026).
- 2502.20470 fetched and its text extracted: Theorem 1 (every admissible instance of
  every admissible constellation has a unique occurrence in G(p_k#)) and Corollary 1
  (the aggregate identity quoted in section 0), plus the population recursion with
  diagonal p - (number of points).  This CONTAINS the local-factor and depth-sum
  identities and explains the doubled spacing.  No effective stage index, no
  twin-candidate subsequence, no analogue of n_g.
- 2603.25915 fetched: one-residue only (quadratic density, Legendre's conjecture); no
  HL-B-in-cycles statement, nothing paired.
- Holt/Holt-Rudd publication status: 1510.00743 was presented at Connections in
  Discrete Mathematics (SFU); the programme appears to live on arXiv and
  primegaps.info rather than in journals - relevant to how a paired extension would
  be positioned.

VERDICT (corrected): PARTIAL OVERLAP, NARROWER THAN CLAIMED IN ROUND 21. Holt owns
the local-factor identity, the aggregate/depth-sum identity, and (as a consequence of
his point count) the doubled level spacing. What is NOVEL AS FAR AS SEARCHED: the
twin-slot gap population n_g as an object, the pinch and its exact Bonferroni series
with the moment identity, and the effective threshold y_0(g).
