# matrix-formulation - the machine's laws as one operating linear algebra

Status: SCRIPT-VERIFIED (every claim below is asserted in
`research/matrix_machine.py`, exact integer/rational arithmetic except where a
line is explicitly labeled FLOAT). Established round 20 (implementation task,
human matrix directive). Prior-art check: done 2026-08-24, per-piece
verdicts in section 6.

## 1. WHAT IT IS

Plain language. The project's discovered laws - the open count, the corridor
law, the autocorrelation c-law, the depth-sum identity, F(M), the merge law,
the paired-Holt population recursion, and the closed-form Fourier spectrum -
were established one at a time by different workstreams with different
vocabularies (censuses, CRT counting, word combinatorics, DFTs). This entry
shows they are all statements about ONE small set of matrices, and that the
matrix forms genuinely OPERATE: the known quantities are recomputed, exactly,
by matrix products, traces, Kronecker lifts, and nilpotency indices - not by
re-running the censuses and decorating them with matrix notation.

The algebra. For each gear (prime q >= 5), on V_q = Z^q:

    D_q = diag(teeth {u, -u}),  u = 6^{-1} mod q     (blocking projector)
    E_q = I - D_q                                    (exposure projector)
    S_q = cyclic shift, S_q e_r = e_{r+1}
    C_q = sum_r a_q(r) S_q^r                         (circulant of the
                                                      exposure indicator a_q)

CRT identifies Z_P (P = prod q) with the tensor product of the V_q: there is
an explicit permutation matrix Perm with

    S_P   = Perm^T ( (x)_q S_q )  Perm               (slot shift = tensor of
                                                      per-gear cyclic shifts)
    E_(M) = Perm^T ( (x)_q E_q )  Perm               (machine exposure
                                                      projector)
    C_(M) = Perm^T ( (x)_q C_q )  Perm               (machine circulant)

all verified as exact integer matrix identities at machine 11 (dim 385) and
mod 35. In this algebra the project's laws become:

1. OPEN COUNT = trace: |E(M)| = trace((x)_q E_q) = prod_q trace(E_q)
   = prod (q-2). Verified == census at machines 11, 13, 17.
2. AUTOCORRELATION / CORRIDOR = a matrix product entry:
   c_q(g) = trace(E_q S_q^g E_q S_q^-g) = (C_q C_q^T)[g, 0], and the
   Wiener-Khinchin identity C_q C_q^T = sum_g c_q(g) S_q^g holds as an exact
   integer matrix identity - Lateral's three-case law {q-2, q-3, q-4} is the
   autocorrelation row of the squared circulant. The mod-35 corridor:
   trace(E_35 S^g E_35 S^-g) = c_5(g) c_7(g) for all g (70/70 exact), the
   round-18 admissible-endpoint-phase law as a 35x35 trace.
3. DEPTH-SUM IDENTITY = the same trace at machine level:
   sum_j W_j(g) = trace(E_M S_P^g E_M S_P^-g) = prod_q c_q(g), because the
   trace of a Kronecker product is the product of the per-gear traces.
   Verified exactly against direct window censuses at machines 11 and 13,
   g = 1..40.
4. F(M) = NILPOTENCY INDEX of B S (B = I - E_M), with the exact Kronecker
   splitting

       B S = (x)_q S_q  -  (x)_q (E_q S_q)

   (a DIFFERENCE of two Kronecker products; B itself is NOT a tensor
   product - blocking is the complement of a product). Verified by exact
   matrix powering: dense 385x385 at machine 11 (index 7 = F), sparse exact
   at machines 13 (index 11) and 17 (index 18); sparse cost 185,782 nnz-ops
   at machine 17 where dense powering would cost ~1e16.
5. MERGE LAW = LIFT-TENSOR-DELETE: adding gear q' is the operator move
   E_new = E_P (x) E_q', S_new = S_P (x) S_q' on the P x q' tensor grid -
   the CRT-recombined period of length P q' is never materialised. Then
   F(M+q') = nilpotency index of (I - E_P (x) E_q')(S_P (x) S_q').
   Verified at three merge steps: {5,7,11}+13 -> 11, {5,7,11,13}+17 -> 18,
   {5,...,19}+23 -> 34 (the gear-recursion 4a values).
6. PAIRED-HOLT RECURSION = an explicit transfer matrix: T[g, w] =
   coef(w) [sum(w) = g] maps the old machine's word-population vector to the
   new machine's gap histogram by one exact integer mat-vec. Verified EXACT
   for every gap value at all four validated rungs ([5,7,11,13]+17,
   [..17]+19, e=344 +17, and the e=102 one-residue Holt collapse); the
   diagonal coef((g,)) equals the c-law of item 2 at every value.
   NEW BEYOND THE ROUND-20 ENTRY: the square word-level matrix H (fusion
   coefficients between words) exactly predicts the new machine's WORD
   populations, pairs included - verified for all 150 words of length <= 2
   at rung 1, with a closure certificate (zero contributions at source
   length 9). H is block-triangular by word length (same-length entries are
   only the diagonal), so its eigenvalues are its diagonal EXACTLY - exact
   rationals after normalising by the openings growth q'-2:

       eigenvalue floor law:  coef_diag(w) = q' - #distinct{(t - p) mod q'}
                              >= q' - 2(j+1)   (j = word length),

   with equality iff the j+1 window points are in general position mod q'.
   The floor q'-2j-2, normalised (q'-2j-2)/(q'-2), is Harvester's eigen-scale
   - CONFIRMED attained and modal at lengths 1-3; at length 4 the floor is
   attained (115 words) but the MODAL value is 8 = floor+1 (118 words):
   residue collisions mod q' are generic once window sums pass q'. That
   correction to "generic = modal" is a measured event, not a fit.
7. DFT DIAGONALISATION: Lateral's closed-form spectrum
   hat_q(j) = -2 cos(2 pi j u / q) is precisely the eigenvalue list of the
   circulant C_q of item 2 (FLOAT check < 1e-9 for q = 5, 7, 11, 13, on top
   of the classical circulant-diagonalisation identity). The golden bound is
   EXACT at the eigenvalue level: sympy factors the characteristic
   polynomial of the 5x5 integer matrix C_5 as

       charpoly(C_5) = (x - 3)(x^2 - x - 1)^2,

   so the non-Perron eigenvalues are phi and 1-phi (each twice) and the
   largest non-Perron |eigenvalue| is phi EXACTLY; phi/3 is an exact
   eigenvalue ratio, and full character enumeration at machines 35, 385,
   5005 finds max non-DC |eig|/DC = phi/3 with multiplicity exactly 2
   (gear 5's +-2 mode only; FLOAT, labeled).

## 2. WHY IT MIGHT BE NOVEL

Individually, each law already had its own entry (merge-law,
depth-sum-identity, golden-spectral-gap, paired-holt-recursion); circulant
diagonalisation, Kronecker traces, and CRT are classical. What this entry
adds:

- THE UNIFICATION IS EXACT AND OPERATIONAL: one algebra (E_q, S_q, C_q,
  Kronecker, one CRT permutation) computes all of them, with assertions, in
  one script. Notably: the c-law, the corridor law, the depth-sum identity,
  and the paired-Holt diagonal are revealed as literally ONE object
  ((C C^T)[g,0]) seen four ways.
- The Kronecker-difference identity B S = (x)S_q - (x)(E_q S_q) for the
  nilpotent operator whose index is F(M) - a compact structural statement of
  why F is a joint-space quantity (the subtrahend factorises, the
  difference does not).
- The word-level (constellation) verification of the paired-Holt matrix:
  the round-20 entry verified the gap HISTOGRAM (length-1 targets); here
  the same matrix's two-block sector exactly reproduces the new machine's
  PAIR populations - the paired analogue of Holt's constellation dynamics,
  previously unverified.
- The eigenvalue floor law with the general-position criterion, and the
  measured modal deviation at depth 4 (floor+1 overtakes the floor).
- charpoly(C_5) = (x-3)(x^2-x-1)^2 as a finite, kernel-checkable form of
  the golden spectral gap (an integer matrix identity a Lean kernel can
  decide).

## 3. PROOF

Status: SCRIPT-VERIFIED. `research/matrix_machine.py` (run:
`uv run python research/matrix_machine.py`) - every numbered claim above is
an assertion; integer arithmetic throughout (numpy int64 / Python int /
Fraction; sympy for the exact characteristic polynomial); floats appear only
in the two lines labeled FLOAT (DFT diagonalisation check, character
enumeration of phi/3). Known values used as cross-checks: the F ladder
7/11/18/25/34 (k-frame), open counts prod(q-2), the four paired-Holt rungs,
the c-law closed form (itself re-derived by brute force inside coef).

Kernel-checkable candidates (finite integer statements): the Kronecker
splitting of B S at a fixed machine; charpoly(C_5) = (x-3)(x^2-x-1)^2;
C_q C_q^T = sum_g c_q(g) S_q^g per gear; a fixed rung of the word-level H
identity.

## 4. IMPLICATIONS

What the matrix form UNIFIES (previously separate laws, now one algebra):

- open count, corridor mod 35, c-law, depth-sum identity: all traces /
  entries of products of {E_q, S_q} and their Kronecker lifts;
- F(M) and the fuel cap: nilpotency indices (Constructor R35/R36);
- the merge law and the paired-Holt recursion: the SAME lift-tensor-delete
  move read at two resolutions (max-gap = nilpotency of the lifted operator;
  histogram = the transfer matrix), with the c-law as the diagonal of the
  latter and the (q'-2j-2)/(q'-2) eigen-scale exact;
- the Fourier frame: the spectrum is the eigenvalue side of the same
  circulant whose matrix side carries the autocorrelation - c-law and
  golden gap are one matrix's row space and eigenvalue list.

What it computes MORE CHEAPLY (op counts, benchmark protocol; counters
printed by the script):

- open count: 53 diagonal-read ops vs ~170,170 census sieve+scan ops at
  machine 17 (x3200);
- new-machine gap histogram via the transfer matrix vs sieving the new
  period: x6 ([5,7,11,13]+17: 27,761 vs 170,170), x11 ([..17]+19: 285,783
  vs 3,233,230), x17 (e=344 +17), x12 (e=102 +17) - the ratio grows with
  q' x mean-gap, and the matrix route additionally yields the coefficients
  (reusable for any old machine with the same q');
- F by sparse nilpotent powering: 185,782 nnz-ops at machine 17 vs ~1e16
  dense-equivalent (the operator has <= one nonzero per column, so exact
  powering costs O(P) per step);
- spectrum: closed form per gear, O(sum q) vs O(P log P) FFT.

What it does NOT do (honest costs): the merge-law-as-operator route (part 3)
costs 3 P q' element ops per iteration x F(M+q') iterations - the SAME order
as sieving the new period; its value is unification and exactness, not
speed (the cheap route to F(M+q') remains the word-level merge law /
rust3 f_next). The transfer-matrix histogram route needs the old machine's
gap word in hand (its cost ledger includes the word pass).

What it CANNOT do (Constructor's round-20 refutations, recorded as the
boundary, unsoftened):

- NO SPECTRAL GAP in the exact operator frame: the renewal operator
  R = sum_v G_v is a permutation (a single |E|-cycle), eigenvalues roots of
  unity - decorrelation is an aggregation phenomenon, and no
  Perron-Frobenius contraction argument runs on the exact frame.
- THE AGGREGATED CHAIN IS NOT MARKOV: the exact one-step transfer matrix on
  gap values over-predicts deep qualifying runs by GROWING factors (x49 at
  machine 29 depth 3; size floors x4.4/x12.6/x40 with depth). No
  fixed-order transfer matrix on gap values carries the anti-correlation
  law; the machine is MORE anti-correlated than any pair-based spectral
  bound. The H matrix here evolves populations ACROSS machines exactly; it
  does not certify within-machine joint suppression.
- B = I - (x)E_q is not a Kronecker product (blocking is a complement of a
  product), so the nilpotency index F has no per-gear factor shortcut - the
  joint space is irreducible there, which is Wall V in operator form.
- The interior condition of window events is a disjunction and does not
  CRT-factorise (Lateral round 19); the renewal ladder's 2^|Y|
  inclusion-exclusion cost stands.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

- Requirement (D) / the suppression law: the well-posed spectral question
  now lives on the word-level H (whose entries are known exact rationals),
  not on any within-machine chain - Holt's p-independent eigenvector import
  (flagged unused by Harvester) is the named next step.
- Jacobsthal / Ziller-Morack h_2: nilpotency-of-BS is an exact operator
  characterisation of the paired Jacobsthal value; the golden charpoly is a
  quotable exact constant of the twin sieve.
- Twin prime route: nothing here closes (D); the unification is the frame
  in which (D)'s remaining content is an eigen/structure statement about H's
  non-triangular sector at fixed machine - open.

## 6. PRIOR-ART CHECK (2026-08-24)

Checked piece by piece - the entry bundles standard machinery with genuinely
new statements, and a blanket verdict would launder the former. Engine:
WebSearch, plus WebFetch full text where noted. The overlapping checks in
paired-holt-recursion.md (2026-08-23; Holt and Holt-Rudd read in detail
there) and golden-spectral-gap.md (which had NO recorded check - the piece-5
searches below are the first for that material too) are extended, not
repeated.

PIECE 1 - CRT state space as Kronecker product of per-gear spaces, blocking
as diagonal projectors. VERDICT: KNOWN (standard/folklore).
- Searches (WebSearch): "Chinese remainder theorem Kronecker product
  circulant matrices decomposition composite order"; "wheel factorization
  sieve residue classes product structure primorial 'tensor product' OR
  'direct product'"; "Good-Thomas prime factor algorithm CRT index mapping
  DFT tensor product permutation cyclic shift".
- The permutation Perm with S_P = Perm^T ((x)_q S_q) Perm is precisely the
  Good-Thomas prime-factor index mapping (I. J. Good 1958, L. H. Thomas
  1963): the CRT re-indexing that turns a length-P DFT/cyclic algebra into
  a tensor product over the coprime factors, no twiddle factors. Circulant
  and shift algebra of coprime composite order as Kronecker products is
  textbook (P. J. Davis, Circulant Matrices, 1979). The sifted set as a
  per-gear product is wheel factorization (reduced residues mod a
  primorial, Pritchard's wheel sieve) and the local-densities-multiply fact
  behind every Hardy-Littlewood singular series. Diagonal projectors
  respecting the tensor factorisation add nothing non-standard. The entry
  does not claim this piece as novel; recorded so nothing upstream ever
  cites the frame itself as new.

PIECE 2 - F(M) = nilpotency index of the blocked-walk operator B S.
VERDICT: KNOWN technique / NOVEL* application.
- Searches (WebSearch): "nilpotent adjacency matrix longest path
  'nilpotency index' directed acyclic graph"; "Jacobsthal function matrix
  formulation nilpotent operator 'transfer matrix' maximal gap coprime
  residues".
- Stripped bare this is the standard graph-theory fact: a digraph is
  acyclic iff its adjacency matrix is nilpotent, and the nilpotency index
  reads off the longest path (any text; recent algebraic treatment
  arXiv:2312.11469, "An Algebraic Approach to the Longest Path Problem").
  KNOWN as a technique. The computational Jacobsthal literature
  (Hajdu-Saradha arXiv:1209.3464; Ziller-Morack arXiv:1611.03310,
  arXiv:1208.5342) enumerates residue combinations combinatorially and
  contains no operator or nilpotency formulation of h(n) or h_2; no hit
  anywhere for Jacobsthal-type maximal gaps as a nilpotency index. The
  application is NOVEL AS FAR AS SEARCHED - with the honest reading that
  its value is the operational frame (sparse exact powering, merge as
  lift-tensor-delete on the same operator), not mathematical depth.

PIECE 3 - the Wiener-Khinchin identity unifying c-law, corridor, and
depth-sum. VERDICT: KNOWN in substance (identity and values); the
three-way identification NOVEL* but elementary.
- Searches (WebSearch): "Montgomery Vaughan distribution reduced residues
  autocorrelation pair correlation sifted set modulo q"; "'Wiener-Khinchin'
  OR 'autocorrelation' circulant identity sieve 'residue classes'
  exponential sums singular series two residues"; "number of pairs
  consecutive integers both coprime to n formula product '1-2/p' Lehmer
  twin totient"; "Schemmel totient function consecutive coprime residues
  1869 generalization"; "Hardy-Littlewood k-tuple singular series local
  factor 'number of residue classes' occupied admissible pattern nu_p".
- C_q C_q^T = sum_g c_q(g) S_q^g is the classical circulant form of
  Wiener-Khinchin (autocorrelation = squared spectrum; Davis 1979, any DSP
  text) - KNOWN, as the entry already says. The three-case values
  c_q(g) in {q-2, q-3, q-4} are a standard local-density count: q minus
  the residues occupied by T union (T - g), the Hardy-Littlewood local
  factor nu_p computation for a two-point pattern against its shift (any
  k-tuples exposition, e.g. Kedlaya's 18.785 notes). The one-residue
  analogue is Schemmel's totient S_2(n) = n prod_p (1 - 2/p) (Schemmel
  1869), and Montgomery-Vaughan's reduced-residue moment machinery is
  built on the same counts. A December 2025 preprint (arXiv:2512.03288,
  Caicedo & Ramos-Fernandez; unrefereed, claim class far beyond its
  method - flagged, not endorsed) computes the per-prime twin survival
  correlation tau_p(d) with its CRT product over primes and Fourier
  coefficients, so even the two-residue correlation-product packaging
  exists in preprint form. Not found stated anywhere: that the corridor
  law, the depth-sum identity and the paired-Holt diagonal are ONE trace
  object - true, but an immediate consequence of trace multiplicativity
  over Kronecker products once the pieces exist. NOVEL* at the level of
  identification, not of mathematical content.

PIECE 4 - B S = (x)_q S_q - (x)_q (E_q S_q): the walk operator as a
difference of two Kronecker products (Kronecker rank 2), stating the
non-factoring obstruction. VERDICT: KNOWN objects / NOVEL* formulation,
elementary.
- Searches (WebSearch): "'difference of Kronecker products' OR 'sum of two
  Kronecker products' nilpotent matrix structure rank"; "nilpotency index
  structured matrix 'Kronecker rank 2' tensor decomposition operator
  sieve".
- Sums and differences of Kronecker products with their Kronecker rank
  (Sylvester index) are a standard numerical-linear-algebra frame (Van
  Loan, "The ubiquitous Kronecker product", 2000; recent bounds e.g.
  arXiv:2605.30908), and nilpotency of A (x) B is elementary. Nothing
  found treating nilpotency indices of DIFFERENCES of Kronecker products
  as a problem, and nothing connecting such differences to sieves or
  Jacobsthal-type functions. The identity itself is one line of algebra
  once piece 1 is set up; the content is the structural reading (the
  subtrahend factorises, the difference does not - Wall V in operator
  form). NOVEL AS FAR AS SEARCHED.

PIECE 5 - charpoly(C_5) = (x-3)(x^2-x-1)^2, phi as exact eigenvalue,
phi/3 machine-independent spectral ratio (also the first recorded check
for golden-spectral-gap.md's material). VERDICT: PARTIAL OVERLAP - the
raw material is classical, the sieve statement is not found.
- Searches: WebSearch "golden ratio eigenvalue circulant matrix order 5
  '2 cos(pi/5)' spectrum pentagon adjacency"; WebSearch "golden ratio
  Fourier coefficient sieve twin primes exponential sum mod 5 spectral gap
  'phi/3' OR 'golden'"; WebFetch arXiv:2512.03288 (abstract + PDF + HTML
  full text).
- phi = 2 cos(pi/5) among graph/circulant eigenvalues 2 cos(k pi/n) is
  classical (path graph P_4 has spectrum {+-phi, +-1/phi}; Coxeter/ADE
  spectra), and the eigenvalues of C_5 are three-term sums of 5th roots of
  unity (Gauss periods) - the VALUE is classical trivia, as
  golden-spectral-gap.md itself anticipates. Nearest sieve-side art is
  again arXiv:2512.03288: per-prime Fourier coefficients of the twin
  pattern with CRT factorisation (search snippet quotes
  4 cos^2(pi k/p)/p^2, the squared modulus of this entry's hat_q(j) -
  elementary for any two-point set); full-text check confirms NO golden
  ratio, NO special role of p = 5, NO spectral-gap ratio anywhere in it
  (its distinguished prime is 3). No hit for phi as the dominant non-DC
  mode of the twin-eligibility sieve, for the machine-independent phi/3
  ratio, or for the integer charpoly form. Those statements NOVEL AS FAR
  AS SEARCHED; the closed-form two-point DFT itself KNOWN/elementary.

PIECE 6 - the word-level block-triangular transfer matrix H predicting
pair populations across primorial levels, eigen-scale (q'-2j-2)/(q'-2),
floor law with the general-position criterion. VERDICT: PARTIAL OVERLAP
(Holt, one-residue); the paired delta NOVEL*.
- Searches (WebSearch): "Holt 'cycle of gaps' constellations eigenvalues
  'p-j-1' triangular transfer matrix populations primorial two residue
  classes extension"; "'general position' residues modulo prime
  coefficient floor eigenvalue 'block triangular' word length sieve gap
  constellation recursion". Extends paired-holt-recursion.md's 2026-08-23
  check (Holt arXiv:1510.00743 read in full there); not repeated.
- Holt's one-residue framework ALREADY contains the word-level analogue:
  constellation populations under a triangular transfer matrix, diagonal
  growth p - J - 1 for length-J constellations, eigenstructure independent
  of the specific constellation, populations Theta(prod_p (p - J - 1)) -
  and his continuations extend it (arXiv:2603.25915 and arXiv:2603.25896,
  "Surviving Eratosthenes sieve", 2026; "Eratosthenes sieve supports the
  k-tuple conjecture"). Block-triangularity and reading eigenvalues off
  the diagonal are Holt's moves, one-residue. Not found anywhere: the
  two-residue H, its exact pair-population (two-block-sector)
  verification, the floor law coef_diag(w) = q' - #distinct{(t - p) mod
  q'} >= q' - 2(j+1) with the general-position equality criterion, or the
  measured modal deviation at length 4 (floor+1 overtaking the floor).
  Delta NOVEL AS FAR AS SEARCHED - the same overlap axis as
  paired-holt-recursion.md, extended one level (words, not just gaps).

OVERALL: the unification survives as a presentation-level contribution
assembled from KNOWN machinery - pieces 1 and the identity/values of piece
3 are standard and now cited as such; the residue of genuine novelty is
piece 2's application, piece 4's formulation, and the deltas of pieces 5
and 6. No source found that states any of section 2's bullets.
