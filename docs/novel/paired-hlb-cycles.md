# paired-hlb-cycles - Hardy-Littlewood Conjecture B in paired cycles: the machine's local factor IS the HL quadruplet factor, and the paired transfer matrix is Holt's with doubled spacing

Status: PROVED (paper proofs below, elementary) + SCRIPT-VERIFIED (exact, at scale:
research/paired_hlb.py, all assertions green; output research/data/paired_hlb.out).
Prior-art verdict: PARTIAL OVERLAP - the one-residue statements are Holt's
(arXiv:1510.00743 Thm 5.5 and his eigenvector analysis); every paired statement here
is NOVEL AS FAR AS SEARCHED. Checked 2026-08-24.

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

Holt proves precisely these statements for the ONE-residue sieve: HL-B ratios in
cycles (his Thm 5.5) and the Pascal eigenvector structure. The paired analogues
did not exist (round-20 search: transfer-matrix sieve literature is Holt alone;
Ziller-Morack have no population dynamics; re-searched 2026-08-24). The deltas:

- c_q(g) = q - nu_q(HL quadruplet): the identification of a two-residue sieve
  correlation with a 4-point HL local factor is a genuinely paired statement (the
  one-residue analogue identifies with 2-point factors). It says the machine's
  pair-correlation structure at every finite level IS the HL quadruplet
  prediction, exactly, not asymptotically.
- The pinch proof of paired HL-B is elementary and quantitative (explicit rate,
  both bounds closed-form CRT products usable at any y with no scan) - Holt's
  argument is dynamical and one-residue.
- Same-eigenvectors/doubled-spacing: a structural statement about WHY paired
  censuses decorrelate exactly twice as fast, sharpening round-20's eigenvalue
  observation into a full diagonalisation.

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

VERDICT: PARTIAL OVERLAP (Holt owns every one-residue counterpart; the paired
identity, paired pinch theorem with rate, word-level transfer verification, and
shared-eigenvector/doubled-spacing diagonalisation are NOVEL AS FAR AS SEARCHED).
