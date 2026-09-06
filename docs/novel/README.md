# Novel findings register

Anything the search produces that MIGHT be new to mathematics gets its own document here.
"Potentially novel" is the bar - over-inclusion is fine, the prior-art check sorts it out.
One file per finding, kebab-case slug. Index at the bottom of this file.

## Required sections per document

1. WHAT IT IS - the statement, exact, with definitions. Plain language paragraph first,
   then the precise form.
2. WHY IT MIGHT BE NOVEL - what makes it non-obvious; what standard result it is NOT
   a restatement of (be honest: most sieve-flavoured statements have a classical shadow).
3. PROOF - the proof itself, or a pointer to the kernel-checked Lean theorem
   (file + theorem name + axiom footprint) and/or the script whose assertions verify it.
   State clearly which status it has: KERNEL-CHECKED / SCRIPT-VERIFIED (finite) /
   MEASURED (not proved) / CONJECTURED.
4. IMPLICATIONS - what it changes inside the project, and what it would mean outside it.
5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES - named problems it solves, weakens,
   reframes, or gives new data for (e.g. Ziller-Morack h_2, Polignac, Conjecture 6).
6. PRIOR-ART CHECK - searches actually run (terms + where), nearest published results
   found, and the verdict: NOVEL AS FAR AS SEARCHED / KNOWN (cite) / PARTIAL OVERLAP
   (cite, state the delta). Date the check. "Not yet checked" is an allowed interim
   status but the finding stays UNCONFIRMED until a check is recorded.

## Rules

- Any agent that establishes something potentially novel writes the document in the same
  round (summary + proof pointer + honest status), and marks section 6 "not yet checked"
  if it has no web access. The manager runs or delegates the prior-art check and updates
  the verdict.
- Nothing here is announced as new until section 6 has a dated check with a verdict.
- Refuted or found-known entries are NOT deleted - the verdict is updated and the file
  stays (a recorded near-miss prevents a re-derivation).
- Index below: one line per finding - slug, one-phrase statement, status, prior-art verdict.

## Index

Verdicts dated 2026-08-23. NOVEL* = novel as far as searched.

- merge-law - F(M+q') from the old machine's gap word alone - PROVED(paper)+SCRIPT-VERIFIED -
  PARTIAL OVERLAP (Holt-Rudd cycle recursion is the one-class analogue; the no-reconstruction
  maximal-gap formula itself NOVEL*)
- deletion-spacing - merge deletions >= q-1 apart, tight - PROVED - PARTIAL OVERLAP
  (Holt-Rudd Lemma 3.1 one-class; two-teeth q-1 bound NOVEL*)
- saturation-theorem - q-1 > F(M) implies F(M+q) = F2(M) - PROVED - NOVEL*
- literal-cap - literal chains <= 6 members forever, function of q' mod 210 - KERNEL-CHECKED
  in full since r20 (per-class cap table + census, LiteralCapTable.lean) - NOVEL*
- corridor-law - 12 of 24 gcd classes forbidden, dichotomy - KERNEL-CHECKED - NOVEL*
  (classification; method standard CRT)
- polignac-cap - capOf_le_twelve, all 8 gcd classes, empty axiom footprint - KERNEL-CHECKED -
  NOVEL* (cap; |E_e| product is the known HL local factor)
- suppression-law - joint qualifying-gap deficits x26/x6.7/x1400 + rate law - MEASURED -
  PARTIAL OVERLAP on phenomenon (Maier, Ford-Maynard-Tao chains); law + shape NOVEL*
- tooth-sharing-pinning - twin gear pair pins 4 CRT kill classes closed-form - PROVED -
  PARTIAL OVERLAP (CRT core classical, Clement 1949 modulus); slot-frame identity NOVEL*
  but elementary
- paired-jacobsthal-values - exact h_2 values + (round 22) the delta reduction, the
  complete 19-winner set, h_2(19) = 258 replicated by exhaustive family scan, and the
  extension-deficit ladder over complete winner sets: 9, 18, 36, and then ZERO at
  23->29 (all 128 complete 23-winners lift to the full y=29 maximum h_2(29)=450, at
  exactly the four residues r = +-3, +-12 mod 29 the cap law predicts; certified by an
  independent 74-position witness). Both round-21 conclusions - the deficit doubling
  AND the permanence of clean-extension death - REFUTED; h_2(23)=366 also replicated
  exhaustively - COMPUTED - PARTIAL OVERLAP: Ziller-Morack
  arXiv:1706.03668 Table 1 already has 18,30,66,150,192 (project premise "ZM compute none"
  FALSE - exact independent replication). Per-difference family F_d, fixed-twin ladder
  F(2,37..53), dip analysis NOVEL*. ZM h_2(19)=258 settles the project's open y=19 question.
  ROUND 23 referee pass (research/j2_referee.py): the h_2 table, #diffs, margins,
  delta-profile law (precision AND recall 100%), the 13->17 cap law (272 lifts,
  extension multiset {81:208, 84:32, 87:32}, the exact 9) and the y=19 winner set
  all reproduce from scratch; TWO documentation defects fixed - the y=3 row was a
  single-survivor code artefact (h_2 = 6 = p^2-p, so Conjecture 6 fails by EQUALITY
  at n = 2 and its 'n >= 3' is sharp), and the maximiser lists were truncated
  argmax slices presented as complete (true counts 8, 16, 64 at y = 11, 13, 17)
- twin-percentile - twins at 13.3rd percentile of own family - COMPUTED - NOVEL*
  (round 23: every number re-derived by independent code, research/j2_referee.py
  sections R3/R4 - all four tie-aware percentile rows, the 30..75 range, mean 38.83,
  median 39, rank 385/2880, and the 31-class F_max/lambda spread 2.88..7.52 -
  reproduce exactly; no defect found in this document)
- depth-sum-identity - sum_j W_j(g) = prod_q c_q(g), closed-form sum rule over
  all window depths + depth-uniform bound - PROVED+SCRIPT-VERIFIED (machines
  11-29 exact) - PRIOR ART FOUND (harvester, 2026-08-24): this is Holt
  arXiv:2502.20470 Corollary 1 (Feb 2025) at the constellation s = (2, 6g-2, 2),
  sum_{j>=J} n_{s,j}(p#) = prod_q (q - nu_q(s)) - the identity and proof are correct,
  the novelty is not; see docs/novel/paired-hlb-cycles.md section 0.  ROUND 26: the
  KERNEL GLUE round 22 lacked is built - Machine13.opSeq_shift (opSeq (n+1485) =
  opSeq n + 5005) from the abstract Periodic.op_shift, so the window counts are counts
  over one period; only the Finset re-indexing to pairCount13 remains
- golden-spectral-gap - machine DFT closed-form and real; spectral gap phi/3
  machine-independent (gear 5's golden mode); T3 law 3u = (q+1)/2 -
  PROVED+SCRIPT-VERIFIED - not yet checked
- paired-holt-recursion - exact linear population dynamics for two-residue sieves
  (n_g(M+q') = sum coef(w) n_w(M), coef position-free); diagonal = the round-19
  autocorrelation c_q(g); eigen-scale (q'-2j-2)/(q'-2) vs Holt's (q'-j-1)/(q'-2) -
  SCRIPT-VERIFIED (4 rungs exact) - PARTIAL OVERLAP (Holt Thm 3.2 is the one-residue
  case; paired recursion + c-law-as-diagonal NOVEL*), checked 2026-08-23
- renewal-ladder - nested closed-form CRT upper bounds on joint qualifying-gap counts
  mod primorials (exposure bound -> exact), clears (D)'s anti-correlation requirement
  at every constrained case incl. both R32 failures; first joint-gap bounds at
  unscannable machines (37+) - PROVED(validity)+SCRIPT-VERIFIED(values) - not yet
  checked (round 20)
- matrix-formulation - the laws as ONE operating linear algebra (traces -> open
  count/corridor/depth-sum; F = nilpotency of BS = (x)S - (x)(ES); merge =
  lift-tensor-delete; paired-Holt as explicit matrix incl. NEW exact word-level/pair
  verification + eigenvalue floor law; charpoly(C_5) = (x-3)(x^2-x-1)^2 exact golden
  gap) - SCRIPT-VERIFIED (research/matrix_machine.py) - checked 2026-08-24 per piece:
  CRT/Kronecker frame KNOWN (Good-Thomas 1958, Davis 1979); WK identity + c-law values
  KNOWN (classical WK; Schemmel 1869 / HL local factor); nilpotency-as-longest-run KNOWN
  technique, Jacobsthal application NOVEL*; Kronecker-difference form NOVEL* (elementary);
  golden charpoly + phi/3 NOVEL* (value classical; nearest arXiv:2512.03288 has no phi);
  word-level H delta NOVEL* over Holt's one-residue constellation dynamics
- j2-upper-bound - first upper bounds on the paired Jacobsthal function j_2, now
  THREE rungs, one per slot of the ordinary Kanold/Stevens/Iwaniec ladder
  (elementary: j_2(p_n#) < 3^(n+1) log^2 p_n; ROUND 22 - Brun pure sieve with a free
  odd truncation depth K, containing the first as K >= n and quasi-polynomial
  p_n^(O(log log p_n)) at the optimal K, better from p_n = 13 on; polynomial
  << p_n^(4.266+eps) by the fundamental lemma, beta_2 improved from 4.45 to the DHR
  value; lower-bound transfer j_2 >= j) plus THE CEILING: exponent beta_2 IS the
  dimension-2 sifting limit and ZM Conjecture 6's exponent 2 sits below even
  Selberg's conjectural floor 2*kappa = 4, so the gap is parity, not technology -
  PROVED(paper)+SCRIPT-VERIFIED (exact rationals) - NOVEL* (the published ladder is
  empty; ZM prove no bound, no 2018-2026 follow-up), checked 2026-08-24.
  ROUND 26: **SECTION 11 IS THE ASSEMBLED UNIT-1 SUBMISSION CANDIDATE** - the
  complete ladder (1, 3E quasi-poly, 2E exp-19, 2E' 17, 2E'' 15, 2G 8.04162,
  2G-inf floor 7.93727, 2 at 4.266 by citation with the explicitness boundary),
  the current sandwich, a rewritten not-claims list (8 items, two of them new),
  and a submission checklist. Sections 1-10 are the working record and are
  individually marked where superseded. The round-25 "discrepancy in the book"
  framing of ODC's printed root is WITHDRAWN: one Newton step from 1/4 - the
  book's own stated method - reproduces 0.264904 to seven digits, so it is the
  book's stated approximation and ours is a sharpening of it (sec. 11d).
  New standing gate research/j2_citesweep.py (citation-numbering sweep promoted
  from manual referee step; caught two live defects on its first run).
  ROUND 23 - publication readiness pass: (0) THEOREM 2E - RUNG 2 IS NOW FULLY
  EXPLICIT: j_2(p_n#) <= 1.0963e10 p_n^19 (log p_n)^10 + 1 for every p_n >= 285,
  no ineffective threshold, via the constant-free Friedlander-Iwaniec Opera de
  Cribro Thm 7.7 plus kappa = 2, K = 3 (K = 3 independently re-derived and shown
  best possible; the hypothesis is Dudek-Dunn arXiv:2602.22720 Lemma 2.1 for
  LITERALLY our density, because sifting n and N-n is the same
  two-classes-per-prime problem). Exponent 4.266 remains non-explicit and cannot
  be made so (DHR delay system + an uncomputed (log y)^{-1/6} error). Also settled:
  the PER-BAND truncation is invalid (36 witnesses) but the UPPER-TAIL NESTED one
  is valid - 168,400 configurations, zero violations - so the only piece left is an
  explicit main-term estimate for it; (a) THEOREM 3E makes the quasi-polynomial
  rung EXPLICIT - j_2(p_n#) < p_n^(9.30 log log p_n) for all n >= 3, asymptotic
  constant exactly 2 lambda_* = 7.1822 (so round 22's measured band [3.47,4.16] did
  NOT contain the limit); (b) the LOWER ladder priced and a named open problem added
  (proved sandwich p^(1+o(1)) .. p^4.266 around a truth of p^2/2; the covering-
  capacity count explains why the paired problem is quadratic and the ordinary one
  near-linear); (c) THE CEILING corrected - "no sieve attains 2 kappa" is an OPEN
  problem not a theorem, the best proved floor is 2 kappa/e ~ 1.47 (Brady), so
  exponent 2 is below the CONJECTURED optimum only, and the actual blocker is
  PARITY via ZM Thm 4.1; (d) a citation audit fixing five second-hand errors
  (C.S. not M. Franze; Selberg's conjecture is not in Franze; 19/36 vs Ford/Brady's
  0.4454; Iwaniec's theorem is h(k) << (k log k)^2 not (log n)^2; Costello-Watts'
  2e^gamma rung is arXiv:1306.1064 not 1208.5342); (e) novelty RE-CHECKED by
  citation graph 2026-08-25 - ZM 1706.00317 has exactly ONE citation in nine years
  (their own note), 1706.03668 has zero, zbMATH has no "paired Jacobsthal" record,
  and Iwaniec 1978 is still the ordinary-ladder record per erdosproblems.com 970/687.
  ROUND 24 - submission checklist discharged + pre-sieved rungs (section 9):
  ODC Thm 7.7 CHECKED AGAINST THE BOOK'S OWN TEXT (p. 111 OCR; three renderings
  now agree, one the book; (7.122)/Cor 7.8 examined and dead for our k); the HR
  Memoire OBTAINED (numdam: "A new look at Brun's sieve", treats exactly our
  density; 7.972 DERIVED from its printed conditions, re-derived and asserted in
  research/j2_presieve.py - the exponent-8 route is an explicitness problem, not
  a new sieve); 19/36 vs 0.4454 SETTLED FOR 19/36 (Selberg's own announcement
  via Greaves' review + Heath-Brown's review, both fetched first-hand, + exact-
  rational re-derivation; 0.4454 recorded unverified, research/j2_selberg.py);
  THEOREMS 2E'/2E'': exponent 19 -> 17 FREE (N_pre = 1) and -> 15 at constant
  cost 135, with 15 PROVED THE FLOOR of FI 7.7 at kappa = 2 (s* -> 14.169 as
  K -> 1); named openings: ODC Ch.6 beta_2 = 7.5941 explicitness, Blight thesis
  ROUND 25 - BOTH NAMED OPENINGS CLOSED AND THE EXPONENT FALLS 15 -> 8:
  Blight's thesis (Sara E. Blight, Rutgers 2010, DOI 10.7282/T35T3KJ8) OBTAINED
  and read - its kappa=2 value 4.45 is WORSE than the DHR 4.266 we already cite
  (she says so herself) and its Prop 2.4.2 gives only "there is some z_0", so it
  is NOT explicit: opening closed NEGATIVELY; ODC Chapter 6 IS EXPLICIT (Prop 6.7
  / Cor 6.13 carry no O(.), no implied constant, no "z large" - only Cor 6.14's
  "K sufficiently close to one" is asymptotic, and pre-sieving replaces it),
  giving THEOREM 2G: j_2(p_n#) <= C p_n^8.04162 (8.04 log p_n + 1)(log p_n)^2 + 1
  with log10 C = 57.5 at p_0 = 151, floor 7.93727, log power 10 -> 3 because the
  beta-sieve's weights are bounded by 1 so the remainder carries tau not tau_4;
  and the HR-Memoire and ODC Ch.6 leads PROVED TO BE ONE EQUATION (HR's
  lambda^2 e^2lambda (2+e^2) < 1 IS ODC's 2e^-2 a^2/(1-a^2) < 1, and HR's
  lambda_* = 0.2533219 = ODC's K->1 root 0.253321897 to 5e-7) - SCRIPT-VERIFIED
  (research/j2_odc6.py) - sources first-hand 2026-08-29. Also caught: ODC's
  printed alpha* = 0.264904 does not solve ODC's own printed equation (true root
  0.2652637, so the book's beta_2 = 7.5941 is 0.0103 conservative).
- j2-lower-ladder - ROUND 24, NEW: the paired covering restated as "cover only
  the z-ROUGH numbers" (one log thinner than ordinary - the structural
  separation), THEOREM (P1): h_2(P(z)) >= (1.349+o(1)) z log z - FIRST lower
  bound using the paired structure, beats the FGKMT transfer asymptotically,
  greedy+matching proof, certificates independently sieve-verified at
  z = 13..10^5 (as run they track ~0.7 z log^2 z); the round-23 "truth ~ p^2/2"
  DOWNGRADED (c z^2 and c z log^2 z fit ZM's table equally, spread 1.87x each;
  local-exponent gap vs ordinary is 0.33-0.75, nowhere near quadratic's +1.0;
  model says ~2.56 z (log z)^2) and the round-23 capacity argument RETRACTED
  (capacity is not scale-free); open problems restated (P2: Rankin layering,
  P3: paired-Iwaniec upper, P4: Conj. 6 true-with-room); falsification target:
  one exact h_2 beyond p_n = 73 (models differ 2.6-3.6x by z = 151-251) -
  PROVED(paper)+SCRIPT-VERIFIED (research/j2_lower2.py) - NOVEL* (KK
  arXiv:2302.00459 is the nearest work: shifted polynomial VALUES, square-root
  classes - neither family contains the other; checked 2026-08-28)
  ROUND 25: (P2) SUPERSEDED by the layered construction (see
  layered-erdos-rankin below) and the "~2.56 z (log z)^2" model DEMOTED from
  "truth" to random-choice heuristic - it is not a ceiling and the construction
  exceeds it.
  ROUND 26: (P2') DISCHARGED (layered-erdos-rankin is now a theorem); (P3), the
  paired-Iwaniec problem, PRICED - a >= 3 is now FORCED, so the question becomes
  "is a = 3?", and the answer is NOT REACHABLE (j_2 >= j, and the record for j is
  still Iwaniec 1978's z^2, open, Erdos problems #687/#970, re-checked
  2026-08-29). NOVELTY QUALIFICATION on (P1), self-found: FKMPT Remark 7 records
  the ORDER z log z for this exact system as trivially available; (P1) is the
  first PROVED bound with a constant, not the first appearance of the order.
- layered-erdos-rankin - ROUND 25, NEW: the Erdos-Rankin construction run k
  times, one layer per available residue class, giving the k-class Jacobsthal
  function j_k(P(x)) >> x A^(2k-1) C^k/((5k)^k B^(2k)) whose k=1 case IS the
  published FGKT length and whose k=2 case is h_2(P(z)) >> z (log z)^3
  (lll z)^2/(ll z)^4 - TWO logs above round 24's (P1) and ONE log above what
  round 24's open problem (P2) asked for. Mechanism: class 0 on a SPLIT range
  buys a full log where its Mertens entitlement is O(1), and the paired
  problem's second class buys it again on n+2, so the joint survivor set is the
  TWIN primes; only an UPPER bound on twins is needed, so it is parity-free -
  research/j2_rankin_layer.py.
  **ROUND 26: WRITTEN OUT - IT IS A THEOREM WITH AN EXPLICIT CONSTANT.**
  j_2(P(x)) >= (1/(18 c_1) + o(1)) x A^3 C^2/B^4, c_1 any admissible constant in
  pi_2(t) <= c_1 t/(log t)^2; with Lichtman's record 3.29956 x 2C_2 = 4.356487
  the constant is 0.0127524 (0.0052597 with Selberg's explicit 8 x 2C_2).
  General k: k/((k(2k-1))^k c_1^(k)) - the small-prime cut is P = A^(2k-1), NOT
  round 25's A^5, which improves the denominator 100 -> 36 at k=2 and is
  INADMISSIBLE for k >= 4 (round-25 self-correction). The greedy lemma is EXACT
  (two classes always capture 2N/p). theta = k exactly FAILS, so the constant is
  a supremum. k=1 calibration is now at CONSTANT level: the same write-up gives
  1.0 against Rankin's proved e^gamma = 1.781 - below it, which is the right
  side. No finite-z content (does not exist below log z ~ 300) -
  research/j2_layer_proof.py - NOVELTY QUALIFIED, checked first-hand 2026-08-29:
  **FKMPT arXiv:1802.07604 REMARK 7 NAMES THIS SIEVING SYSTEM** (I_p = {0,2}),
  records >> log X loglog X - the ORDER of (P1) - as "the 'trivial' bound coming
  from these methods" (no proof, no constant), and hopes for "a small power of
  log log X", which this theorem beats by TWO FULL POWERS. Their >> log^2 X
  pigeonhole bound is no obstruction to a statement about j_2 (density 1/A^2
  there) but FORBIDS ANY TWIN-PRIME-GAP COROLLARY - none is claimed. No theorem
  anywhere on large gaps between twin primes/k-tuples by an Erdos-Rankin
  covering; j_k appears nowhere under any name.
  **ROUND 27: its one named open item, the k >= 4 shift set (sec. 6 item 3),
  is ANSWERED and costs nothing** - see jk-family below.
- jk-family - ROUND 27, NEW (item (P6)): **the k-class Jacobsthal family j_k as
  a published object.** Definition (max over admissible k-tuples E of the
  largest gap between consecutive n with all n+E_i coprime to m) plus the
  COVERING RESTATEMENT - j_k(P(z)) - 1 is the longest interval coverable by
  choosing at each prime p <= z a set S_p of classes with |S_p| <= min(k, p-1) -
  which reproduces the ordinary function at k = 1 and Ziller-Morack's
  omega(2)=1, omega(p)=2 at k = 2, and makes the sifting dimension equal to k.
  Both forms BRUTE-FORCED and equal at k = 1,2,3 and z = 3,5,7; the k = 1 row
  is A048669 (4, 6, 10) and the k = 2 row is ZM's h_2 (6, 18, 30); the k = 3
  row - **j_3(P(3)) = 6, j_3(P(5)) = 24, j_3(P(7)) = 78** - is a first
  evaluation. THE LADDER IS UNIFORM IN k: the Legendre rung with
  omega_p = min(k,p-1); the explicit polynomial rung
  j_k << z^(beta_k + eps), beta_k = 1 + 2(e^(1/2k) - 1)^-1 in (4k-1, 4k+1),
  whose k=2 case IS Theorem 2G's 8.041623; and (P2')'s lower bound
  x A^(2k-1) C^k/B^(2k) at every k. CONJECTURE j_k(P(x)) = x (log x)^(2k-1+o(1)).
  The ODC hypotheses (5.38) and (6.69) were read FIRST-HAND this round and hold
  at EVERY kappa (Cor 6.13 sits at alpha = 1/4 identically in kappa). The
  k >= 4 shift-set question is ANSWERED: any admissible k-tuple works, and
  every prime that could collide lies below the small-prime cut P = A^(2k-1),
  so it costs nothing (threshold x > exp(M_k^(1/(2k-1))) < e^4 for k <= 12).
  Also a one-line proof of the general-k greedy lemma that subsumes round 26's.
  HONEST: the upper rungs are standard sieve theory applied to a new object,
  and at k = 1 the family rung (4.083) is WORSE than Iwaniec's record 2 -
  PROVED(paper) + SCRIPT-VERIFIED (research/jk_family.py,
  research/j2_odcpages.py, both ALL GREEN) - NOVEL* (j_k appears nowhere; OEIS
  seq:6,24,78 returns 19 unrelated sequences and `jacobsthal function
  primorial` returns 6, all one-class), checked 2026-08-29.
- unit1-submission-memo - ROUND 27, NEW: not a finding, a HANDOVER. One page for
  the human: what Unit 1 claims and does not, the three strongest and three
  weakest points a referee will see, venue-class assessment with the honest
  audience number (the anchor paper arXiv:1706.00317 has ONE citation in nine
  years), and the AI-assistance disclosure question flagged as the human's
  decision. THE SUBMISSION DECISION IS THE HUMAN'S; the memo does not recommend.
- paired-hlb-cycles - c_q(g) = q - nu_q({0,2,6g,6g+2}) (machine diagonal = HL
  quadruplet local factor); pinch theorem N2 - sum N3 <= n_g <= N2 (paired HL-B in
  cycles with explicit 1/log^2 rate, both bounds closed-form CRT); paired transfer
  matrix diagonalised by the SAME q-independent Pascal eigenvectors as Holt's with
  doubled spacing; word-level census transfer verified exact (6714 + 10489 words,
  two rungs); ROUND 22 - the pinch identified as BONFERRONI ORDER 1 of an exact
  alternating series n_g = sum_k (-1)^k S_k with moment form S_k = sum_j C(j-1,k) W_j
  (so its slack is the explicit quantity sum over j>=3 of (j-2) W_j; orders 2-3
  verified, they improve the constant not the shape), plus EFFECTIVE Polignac in the
  paired sieve, y_0(g) =
  exp(Theta(sqrt g)) explicit (gap g provably occurs in M_y for every y >= y_0(g),
  no scan), plus the honest boundary (the pinch is full-period; primality lives in a
  share exp(-(1+o(1))y) of it, so nothing transfers to primes) - PROVED(paper)
  +SCRIPT-VERIFIED, local-factor identity KERNEL-CHECKED - PARTIAL OVERLAP (Holt
  Thm 5.5 + eigenvectors are the one-residue case) and NARROWED IN ROUND 22: Holt
  arXiv:2502.20470 (Feb 2025, postdating the earlier sweeps) Cor. 1 CONTAINS the
  local-factor identity and the depth-sum identity, and explains the doubled spacing
  by point count; still NOVEL* are the twin-slot gap population n_g as an object, the
  pinch + its Bonferroni series, and the effective y_0(g) - checked 2026-08-24

- nontensor-sector - how big is the part of the machine that does NOT factor over
  gears, measured as Schmidt rank across gear cuts: EXACTLY 2 at depth 1 (theorem,
  every cut every machine); <= 2n+1 across the merge cut (theorem, and the structural
  reason the merge law is old-machine-only); but at window depth it SATURATES the cut
  (peak rank = d1 at five cuts at machine 23; TR_low = 6, 17, 54, 161, 326 at machines
  11-23), so the tensor rank grows ~ sqrt(P). The growth lives entirely in the
  NILPOTENT direction, which has no spectrum - so no fixed-arity rule can exist and
  nilpotency is where the content is - PROVED(depth 1, merge cut, 2^n bound) +
  SCRIPT-VERIFIED (research/nontensor.py, exact mod-p ranks at two primes) - not yet
  checked (round 22)
- farey-chebyshev-spectrum - the non-tensor sector's Hermitian operators are disjoint
  unions of PATH graphs, one per gap: spec(BS + (BS)^T) = union over gaps g of
  {2cos(pi j/(g+1))}, so there are only |Farey(F+1)| - 2 = sum_{b<=F+1} phi(b) = O(F^2)
  distinct levels with P/F^2-fold ties, and their spacings obey Hall's law with a HARD
  GAP at 3/pi^2 of the mean - <r~> = 0.703, ABOVE GUE. With round 21 (tensor sector ->
  Poisson) this closes the Riemann/GUE bridge at finite machines from both sides:
  spectral richness and non-factorisation are mutually exclusive here - PROVED +
  SCRIPT-VERIFIED (research/nontensor_spec.py) - not yet checked (round 22).
  ROUND-26 EXTENSION AND CORRECTION (section 7): the multiplicity of 2cos(pi a/b) is
  Sigma(b) = #{gaps = -1 mod b}, INDEPENDENT of a - so the eigenvalue multiplicities
  ARE the gap histogram's residue-class counts, invertible by Mobius over multiples
  (W_1(b-1) = sum_t mu(t) Sigma(tb)) and even for every level except 0.  AND THE
  PUBLISHED LEVEL COUNTS WERE THE NAIVE FAREY ONES: the true count is a
  DIVISOR-CLOSURE statistic of the REALISED gap set, so HOLES cost exactly
  sum phi(hole+1) - 21/41/113/183/363/549/981/1813/2467 at m11..m41 against the
  published 21/45/119/211/383/603/1085/2455.  The hard gap and <r~> > GUE survive; the
  NORMALISED gap 3/pi^2 does not (0.2422 at m37) - research/mirror_lever2.py part F
- corridor-eigenvalue-closed-form - Constructor's measured corridor resonance derived:
  the corridor-phase chain's whole spectrum is the image of the e-th roots of unity
  (e = |E| = prod (q-2) over the small gears, 15 for mod 35) under the Moebius map
  mu(w) = rho w/(1 - (1-rho)w), rho = prod_{q not | m}(1 - 2/q); all eigenvalues lie on
  the circle |z - (1-rho)/(2-rho)| = 1/(2-rho) through 1. Matches every measured
  lambda_2 (m11-23, mod 35 and 385) to 1-2% in modulus and < 1.8 deg in argument;
  the residual IS the anti-correlation and is pre-registered for m29 -
  PROVED(model) + SCRIPT-VERIFIED(machine) - not yet checked (round 22)

- two-n-gap-reordering - openings sorted by phase vector have exactly 2n distinct
  gaps (n gears) - PROVED + SCRIPT-VERIFIED (r27: odometer/carry proof with exact
  values and multiplicities, free cyclic closure, order-independence, and the
  general step-type law) - KNOWN IN MECHANISM / PARTIAL OVERLAP, checked
  2026-08-29 (Langevin's lex-successor theorem and Fried-Sos are the classical
  shadow; the delta is the multiplicity table and the free wrap). AND DEFLATED:
  the 2n count is provably BLIND to F - re-choosing every gear's teeth leaves the
  count at 2n while F moves by a factor 1.8, so the reordering is not a route

- tooth-counterfactual-percentile - keeping the gears and the survivor count and
  moving only the teeth (+-v_q over all (q-1)/2 half-widths), the TWIN machine's
  F sits at the 17-26th percentile of the exhaustive counterfactual distribution
  at m11/13/17/19 - the first quantity on which the real phase vector is
  distinguished, and in the favourable direction - SCRIPT-VERIFIED (exhaustive,
  exact) - not yet checked (round 27; angular-coherence mechanism pre-registered
  and REFUTED in the sign)
  ROUND-28 EXTENSION (section 5A): the comparison extended to the statistics the
  LIVE ROUTE consumes. THE FAVOURABLE ONE IS THE INCREMENT LAW'S OWN MARGIN -
  the twin uses less of s_min - (F(M+q') - F_2(M)) than 67-82% of its own
  counterfactuals at every step; F_2 is below the median everywhere and at the
  TWO LARGEST machines the effect STRENGTHENS WITH DEPTH (m19: 17.1 / 12.3 / 6.3
  percentile for F/F_2/F_3; m23: F 11.9%, F_2 3.1%) - the direction that matters,
  since the route consumes F_2 and not F. The m23 rung holds the ~20% plateau
  over five machines but is delivered only for the PINNED family (v_q' fixed at
  the twin's value); the full 142,560-member family did not complete.
  THE INCREMENT LAW IS NOT GENERIC (violated by 13-22% of the family, growing
  with the machine) and most of what it needs is the NEW GEAR'S TOOTH (pinning
  v_q' = round(q'/6) drops violations to 0-6.5%). HONEST NEGATIVE: the budget
  slack F(M+q')-F(M)-q' places the twin at 59.0%/37.2% - undistinguished. And a
  THIRD mechanism is dead: the effect is not a main effect of any gear's tooth
  (max eta^2 = 0.09, never gear 5), the twin's (v_5,v_7) class is the HIGHEST-F
  class, and inside it the twin sits at the 1.7-6.9 percentile - an interaction,
  not a main effect
  ROUND-29 EXTENSION (section 5B): what the RESIDUAL violators are after the
  tooth is pinned. First, a structural result the family gives free - THE RECORD
  LAW IS FAMILY-WIDE: max(F_2, max_J Q*_J) = F(M+q') at ALL 27,570 counterfactual
  machines across five steps, so Constructor's attainment theorem survives moving
  the teeth exactly where (D) and the increment law do not, which localises the
  arithmetic to the SIZE of Q*_J and not to the identity that computes F(M+q').
  Then the brief's question, answered NEGATIVELY: the residual set is NOT one
  congruence on F(M) - "F(M) = a tooth difference mod q'" has sensitivity 34.0%
  at 17->19 and 5.6% at 19->23, the depth-3 attaining middle IS the old record in
  0.0% of 19->23 violators, and the best predictor of the form "F mod q' in S"
  reaches 57.9% balanced accuracy. What describes it is a DEPTH-4 WORD-LEGAL
  WINDOW (depth 3 explains 100% / 63% / 30% of violators at 13->17 / 17->19 /
  19->23; depth 4 is the MODE at 19->23 and depth 6 is populated), and the
  elementary necessary condition is the PEEL BOUND on the FLANKS (min flank >
  s_min), not any condition on the middle. Also priced: the purely spectral
  certificate max(F_2..F_5) <= F_2 + s_min is sound but certifies only 0.3-1.2%
  of the family against word-legality's 96-100% - the arithmetic is worth ~100x
  in coverage
  ROUND-30 EXTENSION (section 5C): L(M) (the depth cap, = A_kill - 1) on the family - max L is 1,3,3,3,5 at 7->11..19->23 against the real 0,1,1,1,2 (the FULL 142,560-row 19->23 family for the first time), so 'L bounded' is NOT structural; the teeth enter through the mod-{5,7} ADMISSIBILITY of the bare alternation (a,b,a) - P(L>=3 | not admissible) = 0.0000 at every step, and the real machine's alternation is inadmissible at 13->17, 17->19, 23->29; the depth-2 slack F + q' - F_2 is ORDINARY for the real machine (35-61 percentile) and fails at exactly ONE of 14,616 old machines, which is the self-mirror 2-window (d_0, d_0) with d_0 = 25 (F_2 >= 2 d_0 is a theorem) - SCRIPT-VERIFIED (exhaustive to 19->23; a 601-member sample at 23->29) - not yet checked

- walk-transform-pole-identity - the distance-to-next-opening function W has
  What(m)(1 - e(m/P)) = -e(m/P) Ghat(m) at every nonzero frequency, where Ghat is
  the GAP-WEIGHTED opening transform: the walk has no Fourier content of its own,
  only a Dirichlet pole times one hard object, and lateral's round-21 pole-phase
  law is exactly this transform. Ghat splits as lambda*Shat + Dhat with Shat =
  prod_q hat_q(m c_q) CLOSED FORM (and the CRT frequency c_q = (P/q)^{-1} mod q,
  which the round-20 statement of the machine DFT omits); Parseval puts
  0.77/0.74/0.71/0.69 of the walk's energy in the closed-form part at m11..m19,
  DECREASING. THE USABLE HALF IS A PROVED NEGATIVE: sum_m |Shat|/P = prod_q S_q/q
  with S_q = (q-2) + sum_k |2cos(2 pi k/q)| is INDEPENDENT OF THE TEETH, so the
  L1 character bound is identical at all 30/180/1440 counterfactual tooth vectors
  while F spreads 1.83x/2.50x/2.29x - no bound built from |Shat| can determine F.
  The L2/Chebyshev bound does vary with the teeth but its rank correlation with F
  is -0.038/+0.023/-0.186, and its certifying length is 7.7x/29.4x/161.3x above F.
  Verdict on the anchor-235 floor: the character form is THE SCAN IN DISGUISE
  (scan 2 pi(q)(F+1) tests beats P coefficients and 2^(F+1) inclusion-exclusion
  terms at every machine) but it NAMES the irreducible object - PROVED (identities)
  + SCRIPT-VERIFIED to 5e-16 at m11..m19 - CHECKED 2026-09-03 (harvester r30):
  mechanisms KNOWN (discrete partial summation / sawtooth transform; Ramanujan-sum
  multiplicativity by CRT; dilation invariance of the L1 norm), the first-passage
  identity for a two-residue sieve and the L1-blindness obstruction against the null
  family NOVEL AS FAR AS SEARCHED; Identity 1 to be presented as elementary

- legal-word-length-mechanism - ROUND 30 (mechanic): the length L_g(M) of the longest
  realised legal word (= A_kill - 1) is predicted to within one unit by an independent-
  letter model with the REAL class densities of the legal alphabet in M's gap histogram
  (3.7/3, 4.0/3, 4.0/2 at m29/31/37 against the naive 8/9/10), while the COUNT of legal
  windows tracks the model at short lengths and collapses at the top (4 vs 279 at m29,
  216 vs 1,610 then 0 vs 2.5 at m31, 27 vs 10,500 at length 2 on m37); every one-letter
  extension of a longest word at m19..m37 is refuted by the cover half alone (no slot of
  M blocks the punctured interior) except the pure alternation, killed by gears 5 and 7
  jointly - MEASURED - not yet checked

## Seeding backlog - COMPLETE 2026-08-23 (all 10 written up and checked; kept for provenance)

- merge-law - F(M+q') computable from the old machine alone (proved, script-verified at 4 steps)
- literal-cap - literal chains have at most 6 members, every gear, forever (exact over 48 classes)
- corridor-law - 12 of 24 gcd classes forbidden, kernel-checked
- polignac-cap - capOf_le_twelve, all 8 gcd classes, kernel-checked
- suppression-law - F_j - qualmax_j ~ lambda*(j-2)*ln(1/p_1) with anti-correlation deficits
  x26/x6.7/x1400 (measured, round 19)
- paired-jacobsthal-values - first exact h_2 values 18,30,66,150,192 at y=5..17 vs
  Ziller-Morack (computed; literature computes none)
- twin-percentile - twins are the 13.3rd percentile of difficulty in their own even-gap
  family (computed)
- tooth-sharing-pinning - twin gear pairs pin 4 CRT double-kill classes in closed form
  incl. the twin-product slot (proved)
- deletion-spacing - merge deletions are >= q-1 apart, tight (proved)
- saturation-theorem - q-1 > F(M) implies F(M+q) = F2(M) (proved)
- cov-sat-exact-spectra - exact gap/hole/window spectra of unscannable machines by CRT+SAT;
  F(41)=91, adjacency refuted at 31/37/41 - SCRIPT-VERIFIED (witnesses machine-checked) -
  not yet checked
- corridor-resonance - extreme gaps phase-locked mod 35: barely damped wave, peaks at
  slot separations 35/70/105, stable pinned classes {7,12,17,18} - MEASURED - not yet checked
- pole-phase-law - the C14 gap-histogram residue phase resolved: 126 deg = the pole
  phase 90 + 180k/p of the one-sided lattice (arg(omega/(1-omega))); measured law =
  the DIFFERENCED histogram's transform is real (+-0.4 deg, m19-37); freq-2 line
  converges to -18 deg confirming; gear 7 not pinned; golden constraint
  phi^2(N0+N1) = N2+N4+2 phi N3; pin-vs-drift open, decidable at m41/43 -
  MEASURED+PROVED(identities) - not yet checked (round 21)
- eigenvalue-statistics - Jacobsthal-machine operator spectra vs GUE: unitaries are
  exact clocks, the circulant's desymmetrized spectrum is Poisson (<r~> = 0.3862 at
  1.3e8 exact levels, KS->Poisson 0.002; trend TOWARD Poisson, away from GUE);
  mirror-degeneracy count P - prod(q+1)/2 EXACT; Riemann/GUE bridge fails at tensor
  operators, only the non-tensor sector could carry it - SCRIPT-VERIFIED on
  closed-form spectra - not yet checked (round 21). ROUND-27 (section 7): the
  degeneracy count is now a THEOREM at every machine (real cyclotomic fields of
  coprime conductor are linearly disjoint, so a product identity forces every
  per-gear ratio into Q and hence equality) - so NO accidental exact collision
  exists anywhere, and the m29 near-collisions all separate at 60 digits
  (smallest 8.6e-14). Backlog U5 closed
- two-teeth-kill-spacing - an added gear's kill spacings lie in the two letter values
  {2u', q'-2u'} (+ exact q' padding), strictly alternating, min 2u' -> fuel <= 1 +
  3*span/(q'-1) closed form; operator form: the spacing law IS the support of the right
  factor of B_new S_new = (B S) (x) S' + (E S) (x) (B'S'); index growth of the sum is a
  >= 3-point statement (2-point relaxation unbounded) - KERNEL-CHECKED(T1-T5,
  proofs/TwoTeeth.lean + MergeLaw.lean, r21 formalist)+MEASURED(M1) - not yet
  checked (round 21)
- covering-lp-certificates - scan-free F(M) upper bounds by LP duality over the exact
  phase-covering IP.  Round 22: thresholds now EXACT ON BOTH ENDPOINTS (W* = 8/21/31/37
  at machines 11/13/17/19, gaps 1.14/1.91/1.72/1.48), the PAIR-VISIBILITY reduction
  q_a q_b > 4W kills a pair outright, and F(19) <= 37 = F(17) + 19 PROVES THE (D) STEP
  17->19 exactly from 37 rationals with no period scan (1,092x fewer operations than the
  1.6M-slot scan); (D) also proved at 7->11, missed by 1 at 11->13.  ROUND 23: THAT
  CERTIFICATE IS NOW KERNEL-CHECKED (proofs/CoveringCert.lean, `F19_le_37` /
  `D_17_19_lp`, standard three axioms, `cert_signs` on NONE) - and formalising it showed
  the optimum is supported on a SINGLE distinguished gear (all 37 weights on rows (i,5)),
  is a PALINDROME, and signs by 17 in 12489.  Otherwise SCRIPT-VERIFIED
  (research/lp_dual_certs.py, research/exact_lp.py; origin research/matrix_shapes.py) -
  PARTIAL OVERLAP (closed-form counting corollary is a weaker case of Costello-Watts
  arXiv:1208.5342; the dual-certificate form, the visibility reduction and the (D)
  application NOVEL*).  ROUND 23: Costello-Watts read from its LaTeX source and
  TRANSFERRED to the two-teeth machine (research/cw_transfer.py) - their dilation lemma
  makes the twin machine SELF-SIMILAR under "restrict to a pair modulus"; measured, the
  transfer gives F(13)<=35, F(17)<=65, F(19)<=110, F(23)<=230, F(29)<=322, i.e. 3.2x-7.5x
  above the true F, so it cannot prove a (D) rung while the dual certificate proves four.
  See consistency-over-degree for what closed the miss-by-one.
  ROUND 27 (section 7): THE CASE-SPLIT SPECIES IS NOW KERNEL-CHECKED - `CaseCert23.D_19_23_case` and `CaseCert31.D_29_31_case`, hypothesis-free, and the second REPLACES a census-hypothesis rung. Three of the four soundness lemmas turned out vacuous at the artefacts (every cut row is the base cut at all 75 certified cases); the real content is the LOWEST-BLOCKER IDENTITY behind the recursion row, and the sizing fact that `n_ab = 0` for 96.4% of the gear-index-1 columns.

- consistency-over-degree - in the covering LP for F(M), one level of MARGINAL
  CONSISTENCY at degree 2 strictly beats two extra degrees without it, and it is what
  proves the (D) rungs.  At machine 13, width 20 (the 11->13 budget) the round-22
  block-independent relaxation is FEASIBLE at degree 2, 3 AND 4 - degree 4 being all the
  gears, i.e. the full per-position joint information, each verdict an exact point
  completable at every position - while the CONSISTENT degree-2 LP is infeasible with an
  exact certificate (660/37 < 664/37, 106 integers over one denominator).  So the
  round-22 MISS BY ONE at 11->13 is CLOSED, and the vehicle proves four consecutive
  rungs 7->11, 11->13, 13->17, 17->19.  Mechanism: a degree-l cut constrains one
  position and per-position completability already contains all of them (Frechet
  included); consistency is a statement ACROSS BLOCKS that no moment inequality can see.
  Corollary: round 22's PAIR VISIBILITY (q_a q_b > 4W kills a pair) is an artefact of
  the missing consistency, not a property of the machine - SCRIPT-VERIFIED exact both
  ways (research/lp_degree_range.py) - PARTIAL OVERLAP (Sherali-Adams/Lasserre
  consistency and the Bonferroni families are classical; the measured direction on this
  problem and the (D) application NOVEL*)
- moment-degree-ceiling - every fixed-degree covering certificate for F(M) goes VACUOUS at
  a computable machine, family-free: the uniform product measure's degree-<=l moments
  extend to a distribution with no empty atom, so every degree-l cut is satisfied at every
  width.  Sharp ceilings degree 1 -> machine 13, degree 2 -> 29 (ROUND-24 CORRECTION:
  "Kounias already degree-2-optimal" is REFUTED - the sharp block-independent degree-2
  threshold at m17 is W* = 30 < Kounias' 31, exact both sides; round 22's 8/21/31/37 are
  Kounias-FAMILY thresholds), degree 3 -> >= 151; required degree ~ 2*S1(y) ~ 4 log log y,
  UNBOUNDED, so no fixed-arity covering certificate exists - the LP-side answer to the
  round-22 arity question.  Chain-cut slope has the telescoping closed form
  S1*prod(1-2/q) + beta and the chain family is exponentially weaker than the sharp
  bound - SCRIPT-VERIFIED exact (research/lp_dual_certs.py C,D) - PARTIAL OVERLAP
  (Boole-Bonferroni LP of Prekopa/Boros is the machinery, Brun truncation growth is the
  classical shadow; the per-degree exact ceiling machine and the degree law NOVEL*).
  ROUND-23 AMENDMENT: the ceiling is NOT the operative limit for (D) - a rung needs an
  integrality gap B(y)/F(y), measured 2.29, 1.82, 1.56, 1.48, 1.41, 1.47, 1.28, 1.08,
  1.42 at 7->11 .. 37->41 (never above 1.48 after the first step, dipping to 1.08 at
  31->37, and -> 1 asymptotically), so the rung-proving range ends far below the
  vacuity ceiling.  The ceilings are UNCHANGED by marginal consistency (the uniform
  product measure is a global distribution, hence feasible for the consistent hierarchy
  too): consistency buys WIDTH, not MACHINES.  And the ceiling does not bind
  Costello-Watts, whose recursive pair term has unbounded effective degree.
- recursion-consistency-composition - composing the consistent degree-2 covering LP with
  ONE valid row built from Costello-Watts' recursion (sum_q S_q - sum_ij n_ij >= W, with
  n_ij the exact pair minimum over the lower gears' phases; f <= open asserted at every
  phase tuple of m11-m19, full period): the row CUTS THE UNIFORM PRODUCT MEASURE at
  budget widths through machine 37 - past the degree-2 vacuity ceiling at 29 - losing it
  only at m41; the composition proves the same four (D) rungs with certificates 2-3x
  SMALLER (562/1,456/3,303/8,179 ops), keeps the FLAT gap (1.000/1.273/1.278/1.320,
  W* = 7/14/23/33) while the row alone wanders (1.57 -> 3.26), and certifies width 33 at
  m19 where NO degree-2 cut certificate exists (block-independent feasible through 34) -
  but NO NEW RUNG: 19->23 stays undecided, per the pre-registered expectation (6 of 6
  pre-registered judgments recorded, one split).  SCRIPT-VERIFIED exact
  (research/cw_consistent.py) - prior-art for the composition NOT YET CHECKED.
  ROUND-25: BOTH OPEN RUNGS NOW EXACTLY REFUTED - 19->23 at width 48 and 23->29 at width
  63 each carry an EXHIBITED exact rational feasible point of the full composition (every
  consistency link exact, every position completable, the recursive row satisfied with
  slack +0.5309 and +0.8384), so the vehicle proves no certificate at either step; with the
  uniform-point refutation at 37->41 the rung ladder is CLOSED at the four rungs it had,
  and round 24's pre-registered E5 is confirmed by proof rather than by empty search.
  ROUND-25 CORRECTIONS (section 0 of the entry): the "uniform frontier is machine 41"
  reading is WRONG-FRAMED (see product-measure-frontier below); "width 33 at m19 where no
  degree-2 certificate of any kind exists" is REFUTED - consistency alone certifies 33
  (20,919 ops, exhibited), so the m19 width belongs to consistency, not the recursion;
  and "2-3x smaller certificates" holds at the budget widths only (1.06x at m19 W=33).
- product-measure-frontier - the composed row's margin against the uniform product measure
  has the closed form E_u[f] = W*Pi(y) - Delta(y,W), where Pi(y) = prod_{5<=q<=y}(1-2/q) is
  the machine's OWN survival density and Delta >= 0 is the summed excess of a phase MAXIMUM
  over its MEAN inside the Costello-Watts pair minima.  Proved identity: the second-order
  lowest-blocker expansion A(y) = 1 - 2*S1 + 4*sum_{i<j} pi_i/(q_i q_j) equals Pi(y)
  EXACTLY (every blocker but the lowest is a blocker above the lowest), and A(y) is both an
  exact upper bound on E_u[f]/W at every width and its exact limit.  CONSEQUENCE: the row is
  never uniformly vacuous at any machine - only ever TOO NARROW - so round 24's "frontier =
  machine 41" is really budget(41) = 129 < 135 = W_u(41), missing by six.  Exact thresholds
  W_u = 10/48/83/135/211/362/558 at y = 29/31/37/41/43/47/53; budget/W_u falls monotonically
  through 1 between m37 and m41.  37 -> 41 is REFUTED exactly (the uniform product measure is
  an exhibited feasible point of the full composition at width 129).  STAR-k restores it:
  holding gear 5's phase explicit gives +8.89 at m41 and stays positive through m53 -
  SCRIPT-VERIFIED exact, no float anywhere (research/row_decay.py) - PRIOR ART NOT YET
  CHECKED.
- kleene-generator - F(M+q') = L^T (x) K* (x) R exactly, where K is the max-plus matrix of
  qualifying-and-T3-alternating successor steps and K* its Kleene star: the merge law's
  increment IS a longest path, so (D) becomes ONE arity-free dual certificate
  (h >= R, h >= K (x) h, L + h <= F + q') with no depth quantifier; the m-th layer of
  the star is qualmax_{m+2}.  Measured: the value-only abstraction is CYCLIC (bound
  vacuous) exactly where the infinite alternating word survives the 2-point relaxation,
  and adding the corridor phase mod 35 restores nilpotency and certifies (D).
  ROUND 23 (section 4b): the HISTORY LADDER A_m - state = the last m-1 gap VALUES, edge
  = a REALISED m-tuple - makes weight, base and left flank all exact and CERTIFIES (D) at
  29->31 where round 22's corridor states failed (A_3 + phase 385 gives 72 <= 74), with
  A_4 (three gap values, phase-free, 14,368 states) EXACT at all six scannable steps; and
  A_m is nilpotent exactly when m > A_relax(M).  The machine-free version (corridor-
  admissible tuples, values 1..F) is SATURATED: mod 35, mod 385 and three gaps of history
  give identical, failing bounds at all seven steps; but counterexample-guided refinement
  from the machine-free system, given only F_2(M), CERTIFIES 29->31 after 6,395 yes/no
  "is this gap 4-tuple realised" queries against a 1.08e9-slot period -
  SCRIPT-VERIFIED (exact, full period, steps 11->13 .. 29->31;
  research/kleene_generator.py, kleene_stream.py, kleene_history.py,
  machinefree_cert.py) - not yet checked (rounds 22-23).
  ROUND 26 (section 4c): the identity is a TWO-SIDED LAW and it resolves
  Mechanic's Q* conjecture (old-machine-spectrum s8/s9) - Q*_J is
  definitionally layer J-2 of K*, so max_J Q*_J = F(M+q') is exactly this
  document's identity and is a THEOREM, the open direction being the CRT
  choice of the killing copy (an ATTAINMENT THEOREM stated without max-plus in
  s4c; J = 2 is the deletion ladder).  VERIFIED EXACTLY AT EIGHT STEPS
  11->13 .. 37->41 with no depth cap and no span cap (research/qstar.py),
  attaining depth J* = k_win + 1 every time.  Consequence, and it is a
  negative: the word-legal criterion is NOT a relaxation of (D) but the same
  statement in another representation, so there is no slack in it to exploit
- nilpotent-invariants - the blocked walk N = BS is PERMUTATION-similar to the direct sum
  over the machine's GAPS of nilpotent Jordan blocks (one J_g per gap), so rank(N^n) is the
  gap histogram's tail sum and EVERY UNITARY INVARIANT of N - singular values, Schatten
  norms, Jordan type, kernel-filtration dimensions, numerical range, resolvent norms,
  pseudospectra - is a function of the gap histogram alone: no operator invariant can bound
  F non-circularly (Wall V in invariant-theoretic form, and the round-22 path decomposition
  is this theorem symmetrised).  Three still buy something: ||N^n||_op = 1 for n<F then 0
  (a cliff - F sits entirely in the constant of any decay envelope); w(N) = cos(pi/(F+1))
  EXACTLY, so the maximal gap is an SDP-representable VARIATIONAL quantity; and
  r_eps = eps^(1/F), a Maslov dequantisation making the (+,x) resolvent, the (max,+) Kleene
  star and the Boolean filtration one computation in three semirings.  Two checked
  NON-GAINS: moments/exponential sums reduce to the r_L run ladder, and Weyl across a merge
  step is vacuous (2.85-2.99 > 2) - PROVED + SCRIPT-VERIFIED exact integers
  (research/nilpotent_invariants.py, machines 11-19) - not yet checked (round 23)
- potential-arity-ladder - F(M) <= 1 + osc(h) for any potential with h(k) >= h(k-1)+1 at
  every blocked slot, TIGHT (distance-to-previous-opening attains it), so F is exactly an LP
  optimum and the only thing that can fail is the certificate's ARITY.  T1: a potential that
  depends only on k mod m for a proper divisor m certifies NOTHING (one line) - why
  bounded-state certificates mod 35/385/5005 cannot bound F.  T2 (MERTENS NO-GO, proved,
  exact rationals): a per-gear (arity-1) potential exists only if sum_(5<=q<=y) 1/q < 1/2,
  so arity 1 dies at machine 13 and never returns.  Measured ladder with every feasible
  certificate re-verified against the full period: arity 2 gives 1.11x, 1.63x, 2.06x the
  true F at m11/13/17 - a fixed arity goes asymptotically vacuous while staying feasible.
  Conjectured threshold sigma >= r/2 (sign condition named as the gap) puts level 2 dead at
  y=109, level 3 at y=2741, level 4 at y=483281, i.e. required arity ~ 2 sum 1/q ~
  2 log log y - the SAME arity law the LP-duality thread found independently on a different
  certificate family - PROVED (T1,T2) + SCRIPT-VERIFIED LP ladder
  (research/potential_arity.py) - not yet checked (round 23)
- old-machine-spectrum - the lap-phase transfer: a window of the machine r gears ahead is a
  window of THIS machine plus r free CRT phases, so (A) the whole qualifying ladder
  Q_J(M + q_1..q_r; a) is computable EXACTLY on M's period, at 1/(q_1...q_r) of the cost,
  and (B) F_{r+1}(M) <= F(M + q_1 + ... + q_r) (r new gears buy r rungs of the F_j ladder,
  one designated kill each).  (A) validated at r=1 on five steps and at r=2 (machine 31's
  full-period ladder 68/85/90/91/90/88 recovered from machine 23's period in 338 s, period
  ratio 899); (B) asserted at all 32 (M,j) pairs with both sides known, and it PINS
  F_2(41) = 103 with no descent (cap F(43) = 103 free, witness at 103).  Also RETRACTS my
  own round-22 "the marked spectrum loses the 29->31 rung (85 vs 74)": tool bug, the true
  value is 71.  APPLIED, and it decided the project's named open computation: the word-free
  criterion max_J Q_J <= F + q' holds at every step to 41->43 and FAILS at 43->47 (152 vs
  150) and 47->53 (177 vs 171), both witnesses asserted at the target machine, both failures
  confined to depths 6-7.  ROUND 25 (section 8) REPAIRS BOTH FAILURES with the WORD-LEGAL
  CRITERION Q*_J: the plain criterion only asks that the J-2 middle gaps clear a = 2u', but
  the merge law needs the J-1 interiors deleted by ONE phase of q', i.e. the middle gaps must
  form a legal KILL WORD (each in V = {0,+s,-s} mod q', induced letter word of prefix-sum
  range <= 1); ">= a" is merely its shadow, since the smallest positive legal value IS a.  The
  failing 47->53 window has middles [22,28,30,67], not one of them legal mod 53 - the criterion
  was failing on a relaxation the merge law never needed.  Q*_J is pointwise <= Q_J, costs the
  same transfer, and CERTIFIES BOTH BROKEN STEPS at EVERY depth J = 2..7, so neither
  certification consumes a fuel-arity bound (which matters: the same round proved
  A_kill(47->53) = 5, killing the arity route there).  Two two-sided anchors are EXACT -
  max_J Q*_J = 88 = F(37) at 31->37 and 58 = F(31) at 29->31, against the plain 91 and 71,
  each attained at depth k_win+1 reproducing an independently measured k_win - motivating the
  CONJECTURE (2 exact points) that Q*_max IS the merge-law value F(M+q'), not just an upper
  bound - PROVED (elementary) + SCRIPT-VERIFIED (anchors exact; the two repairs seeded at
  budget-1 and span-capped at 200, stated) - not yet checked (rounds 23, 25).
  ROUND 27 (section 11): the record law used as a COMPUTATIONAL INSTRUMENT where the answer
  was NOT already known.  53 -> 59 was the first step of the ladder with no upper bound on
  the new machine's F anywhere - the corpus F(2,y) ladder stops at y = 53.  Computed on
  MACHINE 23's period (37,182,145 slots) for a property of MACHINE 59 (period 1.96e19,
  ratio 5.3e11): max_J Q*_J(53; legal for 59) <= 203 < 204 = F(53) + 59, so (D) HOLDS AT
  53 -> 59, and 161 <= F(59) <= 178 with the lower bound an exhibited machine-53 window
  (gaps [10,118,33], middle 118 = 2q' a padded kill letter).  The enabling technique is a
  DESCENDING LADDER OF BANDS - a run seeded at lo with cap hi decides exactly "the maximum
  in (lo, hi]", so bands compose with no gap, each finishes, each tightens the bound, and
  an EMPTY band also refutes every OTHER object whose span lands in it (here it closed a
  whole A_kill level at every depth with no solver call)
- covering-hierarchy-exactness - the Jacobsthal covering CSP's pairwise (Sherali-Adams
  level-2) LP computes F EXACTLY at machines 11/13/17 (exact rational dual certificates:
  479/1152, 1041/2081, 1673/19767) and BREAKS at machine 19 (L* = 27 vs F = 25), where the
  impossibility of runs 25 and 26 is invisible to ALL pairwise reasoning - the level-2 SDP
  is feasible at the impossible L = 26 (converged PSD moment matrix, numerical) - so every
  certificate of F(19) <= 26 needs arity >= 3.  Vacuity ratios 1.00, 1.00, 1.00, 1.08,
  1.65, >= 1.72 at m11..29: a THIRD independent certificate family obeying the project's
  arity law.  Companion theorem: the machine-free max-plus system equals its own LP
  (12/12 steps), so NO convex relaxation of it can improve one unit - its gap is 100%
  edge-set.  Level 1 dies exactly at sigma >= 1/2 (T2's threshold, covering side) -
  PROVED (soundness, MF-LP) + EXACT RATIONAL DUALS + MEASURED (SDP verdicts numerical,
  flagged) (research/sdp_cover.py) - not yet checked (round 24; web budget exhausted,
  manager to run)
- survivor-generator - F_2(M+q') (and, proved though not yet script-checked, every
  F_j(M+q')) is the SAME max-plus Kleene algebra over machine M as F(M+q'), with ONE extra
  transition: a skip of weight d_i + d_{i+1} through the unique SURVIVING opening, guarded
  by "cls(d_i) illegal from the current tooth" - so the two-gap statement at a step is
  layer 0 of the previous step's generator, and the "one extra integer" R53's CEGAR needed
  is a PROJECTION of the dictionary the certificate already queries (the realised-pair
  sub-dictionary).  Verified exact, full period, at all six steps 11->13 .. 29->31
  (F_2 = 16, 25, 31, 39, 55, 68 against the independent pair census); A_4(M) bounds
  F_2(M+q') by 16, 25, 31, 42, 57, 93 - clearing the next step's two-gap budget at every
  step - PROVED + SCRIPT-VERIFIED (j = 2) (research/survivor_generator.py) - not yet
  checked (round 24).  ROUND 26: KERNEL-PROVED SOUND AT 11 -> 13 - the generator does
  not merely COMPUTE F_1..F_4(13) = 11, 16, 23, 26 from machine 11's 135-letter word,
  it PROVES them (proofs/Gen11Sound.lean, generator_sound), with machine 13's own
  5,005-slot period absent from the derivation - gated by a transitive-dependency audit
  (proofs/DepAudit.lean), not asserted.
- mirror-parity-laws - the opening set's exact symmetry k -> -k pins the PARITY of
  every window and gap-word count: for each depth j, W_j(g) is even for all g except
  the single length of the window at index t = -j/2 (mod N), which is odd; the
  depth-j gap-word census is EXACTLY reverse-symmetric with exactly one odd
  palindrome per depth, forced to be (k_1,k_1) at j = 2.  COROLLARY FOR THE TWO-GAP
  LAW: any adjacent pair with g_1 = g_2 - in particular an (F,F) pair realising
  F_2 = 2F - occurs an EVEN number of times, so a counting argument capping such
  configurations at ONE proves there are NONE.  Also caught a real defect in the
  shared census file (every full-period ghist row drops the wrap-around gap) -
  PROVED (elementary) + SCRIPT-VERIFIED m11..m29 (research/mirror_cells.py parts A,B)
  - not yet checked (round 25).  ROUND-26 EXTENSION (section 7): the FULL affine
  symmetry group of the opening set is (Z/2)^m (multiplication by c = +-1 mod every
  gear) and only c = +-1 mod P preserves adjacency, so the lever is worth EXACTLY one
  unit and there is no mod-4 version; the exceptional window is relocated from an
  INDEX to an ADDRESS - it is the window centred on slot 0 (j even) or on the antipode
  (j odd), giving g_j* = j (mod 2) and a scan-free formula at ANY machine; the
  antipodal slots (P+-1)/2 are openings at every machine by the tooth law, so g_1* = 1
  ALWAYS and W_1(g) is EVEN for EVERY g >= 2 unconditionally (ROUND-26 FORMALIST: the
  ARITHMETIC halves are now KERNEL-CHECKED - proofs/Mirror.lean, mirror_gear,
  antipode_open, self_mirror_unique, footprint [propext, Quot.sound]; the EVEN-COUNT
  half needs an involution-counting lemma and is NOT yet formalised); a palindromic tuple's
  count is odd iff it occurs at the single address -span/2 (mod P); the exceptional
  window is never QUALIFYING, so the lever applies to the (D) family with no side
  condition; and reverse-invariance of every realisability census is audited against
  the project's own arity runs (82 decisions, every reverse pair agreeing, 46% of
  27,946 s spent on the redundant half) - research/mirror_lever2.py, 52 gates
  ROUND 27 (Formalist): THE COUNTING HALF IS KERNEL-CHECKED - `Mirror.even_card_involution`, `window_count_even`, `adjacent_equal_even`, `none_of_at_most_one`. Round 26's named gap is closed for the ABSTRACT lever; instantiating it at a machine (mirror-equivariance of the window family through the opening enumeration) is NOT built and is named as the next step.
  ROUND-28 EXTENSION (section 8, Lateral): the kernel lemma's ONE non-free
  hypothesis `hexc : L t0 <> 2F` gets a closed form. The self-mirror window at
  depth j has SPAN 2*o_{j/2} (j even) or P - 2*o_{M-j/2} (j odd), M = (N-1)/2 -
  so at the route's own depth-2 target `hexc <=> d_0 <> F`, a one-line inequality
  at every machine. Its span is a TYPICAL j-window span against F_j's maximum,
  so the LEVER'S EXCEPTION LIST (span_self = F_j exactly) is m7 {3,7,9,11,14} and
  m11 {11} and EMPTY at m13/m17/m19/m23 for all j <= 30. Reversal-closure extends
  to the two CRT TRANSFER supersets (2.4M and 4.2M tuples, 546 and 874
  palindromes), which had no a-priori reason to inherit it. Self-corrections:
  section 8.4 is a REPLICATION of item 51 in different coordinates, not a new
  result; a claim to close half of backlog U10 was WITHDRAWN (8.7); and round
  25's "exactly one odd palindrome" is correct only for PALINDROMES and only as
  "at most one" (8.5)
  ROUND-29 EXTENSION (section 9, Lateral): THE LEVER'S HYPOTHESIS IS NOW A
  THEOREM AT EVERY DEPTH >= 3, on the WORD-LEGAL family the live route actually
  uses. The self-mirror depth-J window is NEVER word-legal for J >= 3: at ODD J
  its central middle is the antipodal gap, of length 1, and 1 is a legal letter
  only if 3 = +-1 mod q' (impossible, since 2u' = 3^{-1}); at EVEN J >= 4 its
  two CENTRAL middles are both d_0, so T3 forbids two equal nonzero classes and
  0 < d_0 < q' forbids both being padded. Hence the mirror is FIXED-POINT-FREE
  on that family and every span count is EVEN with no exceptional class, no
  exception list and no census - which replaces round 26's 66-cell check and
  round 28's span table by arithmetic. J = 2 is the ONLY depth needing a
  hypothesis, and there it is exactly d_0 != F. Gated at m11..m23, J = 2..7,
  185 assertions, and every Q*_J reproduces Constructor's R68 table by a
  different vehicle. HONEST NEGATIVE, and it answers the round-29 brief: the
  even-J reversal symmetry yields NO inequality on F_J or Q*_J - being
  span-preserving, all it adds is the quotient by an involution, i.e. the SAME
  one unit ("fewer than two proves none") that section 7.1 already proved is
  the ceiling. What changes is the PRICE of that unit, not its size
  ROUND-30 EXTENSION (section 10): the mirror k' = P - k - s gated on all 24 exact record windows on file (150 assertions; F_2(59) B is A's mirror, the F_5(41) pair is a mirror pair in machine-41 coordinates), its TRANSFER-COORDINATE form (k, c_q) -> (P0 - k - s, P0 - c_q mod q) gated on the F_6(47), F_5(41) and F_2(59) witnesses (a factor 2 on every transfer sweep), and the one new consequence: the J = 2 self-mirror window (d_0, d_0) is the counterfactual family's only depth-2 failure - PROVED + SCRIPT-VERIFIED - not yet checked

- gear-cell-decomposition - the frequency-1/p Fourier coefficient of the gap
  histogram is a function of only (p-2)(p-3)/2 integers, for EVERY machine (three at
  p = 5), via the cell matrix M[i][s] indexed by (start exposed phase, exposed-step
  count mod p-2); mirror + CRT give the exact relation 2(N_1-N_4) = N_2-N_3 on gap
  residue classes mod 5 and the three-integer closed forms for Re/Im H_5(1).
  THEOREM: (N_2+N_3) - 2N_0 = 2 (mod 4) at every machine, so round 21's pole phase
  126 deg is NEVER attained exactly - the machine instead drives an integer ratio
  toward -1/phi (crossing it between m29 and m31).  ROUND-27 CORRECTION (section 7,
  on Mechanic's exact full-period m37 histogram): -1/phi IS NOT A LIMIT, it is a
  CROSSING - the exact ladder runs ... -0.6231 (m29), -0.5943 (m31), -0.5778 (m37),
  i.e. 0.040 PAST the golden direction and still rising, the same event as
  arg H_5(1) crossing 126 deg; and the 1.015 amplitude plateau OSCILLATES in
  [1.0100, 1.0193] to m37 with no break in either model's direction (backlog U6 and
  U9 closed).  ROUND-26 CORRECTION: "gear 5 is
  the ONLY parity-obstructed gear" is TRUE OF THAT GF(2) TEST AND FALSE OF THE MACHINE -
  since the antipodal gap has length 1 always, W_1(1) is the only odd histogram entry,
  so N_1^(p) is odd and alpha_1(p) is odd for EVERY gear p, and the pole phase is
  unattainable everywhere; the gear-5 vs gear-7 asymmetry is then entirely the MEASURED
  one (three equations not one, asymmetries an order of magnitude larger).
  Backlog U2 closed: the 1.015 amplitude near-law is the crossing scale lam = 23.92
  at which the depth-1 arm meets the exactly-computable MEAN arm (2-phi)N/9, and its
  flatness is a cancellation between a decaying shallow-corridor drift and a rising
  deep-corridor drift - no fixed corridor depth reproduces it - PROVED (elementary)
  + SCRIPT-VERIFIED m11..m31 (research/mirror_cells.py parts C-F, research/spiral29.py)
  - not yet checked (round 25)
- scanfree-certificate - requirement (D) at one step as a FINITE CRT COMPUTATION: a gap
  tuple is realised iff a set-cover CSP over the gears is feasible ("the prefix-sum
  points open, every interior point covered"), so the realised-tuple dictionary, the low
  spectrum F, F_2, F_3, F_4, the A_4 abstraction and the whole counterexample-guided
  certificate are computable from the LIST OF PRIMES ALONE, with no period anywhere.
  Gated: decision == R43's independent pruned-IE count on 2,013 tuples; the corpus
  ladder F = 7..88 and F_2 = 11..90 recovered scan-free; the scan-free D_4(23) is
  SET-EQUAL to Mechanic's full-period census (15,696 tuples); the certificate reproduces
  round 24's 181/90/955 queries with 100% oracle agreement and then certifies the NEW
  rung 31->37 (95 <= 95, 3,399 queries) where no scan and no dump exists.  Includes the
  COVERING FORM of the two-gap law and its three machine-free instruments - capacity
  (kills only both-gaps-near-F pairs), the first moment (gets the law RIGHT, unlike the
  histogram and the corridor), and the closed-form asymptotic incr ~ log^3 y against a
  budget q' ~ y (measured decay of incr/q' from 0.385 to 0.0145) - SCRIPT-VERIFIED
  (finite, exact) (research/crt_dict.py, scanfree_dict.py, chain_cegar.py, chain_a4.py,
  twogap_threshold.py) - not yet checked (round 25)
- qualifying-dictionary-rung - a (D) rung whose certificate is the size of a DICTIONARY,
  not of a period: the merge law consumes only F_2 and the QUALIFYING spectrum Q_j, whose
  windows all have interiors above the next gear's tooth floor, so the whole input is the
  stratified family D_j of realised qualifying j-windows - and it TERMINATES at j = K+2
  where K is the longest qualifying run (3, 4, 5 at m19, m23, m29).  At 29->31 that is
  15,860 tuples against a period of 1,078,282,205 slots, and the SIXTH (D) rung
  (proofs/Machine31.lean, D_29_31, one named census hypothesis; the six dictionary checks
  have an EMPTY axiom footprint) builds in FIVE MINUTES where round-24 verdict 17 priced
  the period-scan vehicle at ~170 h.  Applied again the same round to give the SEVENTH
  rung 31->37 (43,185 tuples against 33,426,748,355 slots, margin 4) - where it found
  the FIRST NON-MONOTONE qualifying spectrum, Q_j(31;12) = 68,85,90,91,90,88, so the
  binding constraint is a FIVE-gap window and not the two-gap statement at all.  The
  dictionary grows ~3-5x per gear while the period grows ~30x, and K (which sets the
  family's depth) did not grow from 29 to 31.  Confirms F_2(29) = 55, F_2(31) = 68 and
  the corrected marked spectrum Q_J(29) = 55,65,68,71,71,71 by independent routes -
  KERNEL-CHECKED (given the census) + SCRIPT-VERIFIED full period, four-gated at both
  machines - not yet checked (round 25).  ROUND 26: THE CENSUS HYPOTHESIS IS NOW A
  ONE-PERIOD CLAIM - Census29P/Census31P restrict every clause to the openings of one
  period (exactly what the gates scan) and IMPLY Census29/Census31 via
  Periodic.index_reduce, which needs no walk and no base case (proofs/Machine29Cen.lean,
  Machine31Cen.lean, LadderPeriod.lean).  The rung is still hypothesis-explicit; the
  hypothesis is now finite.
- phase-saturation-arity - a CLOSED-FORM, SOLVER-FREE ceiling on kill-chain arity: a
  gap word with exposed offsets X cannot occur at all if some gear q has no admissible
  phase, i.e. if (X mod q) u ((X - s_q) mod q) = Z_q with s_q = -2*6^{-1} mod q.  Since
  that union has at most 2|X| elements the whole content sits at gears 5, 7, 11, and
  applied to the pure ALTERNATION (s, q'-s, s, ...) - the shape that carries every long
  chain in this project - it gives a per-step ceiling in closed form: 6, 2, 2, 2, 5, 3,
  3, 4 at 31->37 .. 61->67, ATTAINED at 47->53 where A_kill = 5.  It refutes round 25's
  alternation-pair predictor at 53->59 (the pair (20,39) IS realised at machine 53, with
  a definition-level verified witness, and the 4- and 5-letter alternations are
  nevertheless ZERO BY THEOREM - by gear 7 and gear 5 respectively), so the FIRST
  5-CHAIN WAS NOT THE START OF A TREND and 53 was special for an arithmetic reason.
  Gated: 37 realised words at five steps, none wrongly zeroed; reproduces the three
  structural zeros already on record ((18,35,18,35,18) at 47->53 and (16,31),(31,16) at
  43->47) - SCRIPT-VERIFIED (finite, exact) (research/akill_verify_r26.py,
  research/alt_obstruct_r26.py) - not yet checked (round 26).
  ROUND 27: A_kill(53->59) MEASURED = 4 EXACTLY, every level complete - the pure
  alternation is zero from k = 4 as the ceiling 3 requires, and what reaches 4 is the
  PADDED alternation (s, q'+(q'-s), s) = (20,98,20).  A_kill - ceiling is now
  -2,+1,+1,+1,0,+1 at 31->37..53->59: at every step whose ceiling is 2 or 3 the answer
  is ceiling+1 and the lifting word is padded.  Named next construct: the same
  pigeonhole applied to the PADDED alternation family (research/akill_verify_r27.py)
- uniform-order-bound - the truncation ORDER of the qualifying-run algebra, bounded
  uniformly for the first time.  A_relax(M) (the order at which the pure alternation
  stops being realised, the project's proxy for how much gap history a sound abstraction
  needs) is <= 5 AT EVERY MACHINE, by phase saturation at gears 5 and 7 alone, hence by a
  function of q' mod 210 with no machine in it; and <= 4 except on the six classes
  q' = 37, 53, 83, 127, 157, 173 (mod 210) - which are EXACTLY R20's litcap-6 classes,
  identifying two independently found invariants of the machine as one object (both are
  "the longest alternation whose prefix-sum walk stays in the corridor E mod 35"; litcap
  maximises over the two starting letters where the order minimises).  Gears 11 and 13
  buy nothing.  CORRECTS the published A_relax ladder at m37 (2, not 3 - R45's script
  assumed the 2-letter alternation realised instead of looking it up; gear 5 refutes it).
  THE COMPANION NEGATIVE IS THE SHARPER HALF: for the order the chain actually needs -
  N(M), the smallest m at which A_m is acyclic, which must break PADDED cycles too - the
  corridor cap on all legal words is finite only while F(M)/q' is small and goes INFINITE
  from the step 53 -> 59 on, so no fixed set of small gears can ever cap the order again;
  and the folk identity N = max(2, A_relax) (7 of 7 in R49) is REFUTED at the eighth
  machine, A_relax(37) = 2 but N(37) = 3, the extra order bought by a padded cycle -
  SCRIPT-VERIFIED (finite, exact) (research/uniform_order.py) - not yet checked (round 27)
- restricted-covering-certificates - the composed covering LP applied to a RESTRICTED
  POSITION SET with a RESTRICTED PHASE DOMAIN, one construct with two payoffs.  (i) THE
  CASE SPLIT: holding the k smallest gears' phases makes case w the same vehicle on
  [0,W) minus what they block, and a certificate in every case certifies the rung - a
  species strictly stronger than the STAR-3 LP (a family of case points always mixes into
  a STAR-3 point; a STAR-3 point does not condition into case points).  IT CERTIFIES
  EVERY (D) RUNG THE PROJECT HAS, at the ladder's own budget widths, HYPOTHESIS-FREE from
  the list of primes alone: 19->23 (hold 5; five cases at iteration zero, 38,677 exact
  ops), 23->29 and 29->31 (hold 5,7; 35 cases; 362,049 and 576,472 ops), 31->37 and
  37->41 (hold 5,7,11; 385 cases; 8,388,426 and 12,778,058 ops) - and round 25 had
  REFUTED 19->23, 23->29 and 37->41 for the level-2 member with exhibited exact witnesses,
  so the refutations bound the member and not the family.  The number of held gears is a
  real ladder parameter (each one roughly halves the residual and multiplies the case
  count by that gear), and the cost is a PRIMORIAL in k - a new kind of limit for this
  project, neither a degree ceiling nor a width frontier.  The same vehicle becomes TIGHT
  ON F: F(19) <= 25, F(23) <= 34, F(29) <= 43 all certified at k = 2.  (ii) THE WINDOWED
  STATEMENT: prescribing open positions decides ADJACENT-GAP-PAIR realisability, i.e. the
  level-2 gap dictionary the chain and the merge law consume, by LP duality instead of
  search - giving complete scan-free proofs of the EXACT F_2(19) <= 31 and F_2(23) <= 39,
  locating the F_2 maximiser (at span 31 exactly the two realised splits (10,21),(21,10)
  fail, each by an exact in-polytope witness), and certifying SPECTRUM HOLES (24, 27, 29
  at m19, two of them below F = 25).  Sound everywhere against a full-period scan (zero
  unsound cells in 277) but NOT exact: nine unrealised cells at m19 spans 28 and 30 are
  not certified, four with exact feasible witnesses.  The two restrictions COMPOSE and
  the composition is strictly stronger than either - SCRIPT-VERIFIED (finite, exact
  rationals; every certificate and witness saved and re-verified from a clean rebuild)
  (research/star_case.py GATE - eight items, research/window_dict.py) - not yet checked
  (round 26).  ROUND 27 adds three: the NINTH RUNG 41->43 finished (all 385 cases at
  k = 3, 18,649,193 exact ops, every certificate re-verified from disk - the step the
  project's other scan-free route reported as not certified); the vehicle REACHES THE
  INCREMENT WIDTH W_inc = F_2(M) + s_min(q') at all six literal steps 11->13 .. 29->31,
  so the manager's increment law has its base cases by CERTIFICATE (upper half) plus
  scan-free CRT WITNESS (lower half), tight at every one; and the two kernel-bound rungs'
  certificates use NO degree-2 cut at all (every row is the base cut).  Not reached:
  41->43 at W_inc = 117, where the pre-test passes but the cut loop does not converge
  (research/gate_rung_41_43_r27.py, research/increment_cert_r27.py GATE,
  research/emit_certs_r27.py GATE)
  ROUND 27 (Formalist): KERNEL-CHECKED AT TWO RUNGS - `CaseCert23.D_19_23_case` (19->23) and `CaseCert31.D_29_31_case` (29->31, 35 case modules), no census hypothesis anywhere. Two facts fell out: EVERY cut row of all 75 certified cases of 19->23/23->29/29->31 is the base cut (the loop never separated at these widths), and `n_ab = 0` for 96.4% of the gear-index-1 columns, so the recursion row is numerically almost entirely a Kounias row at the smallest free gear.
- per-j-window-analogues - the per-J family Q*_J of the manager's triple inequality: the
  MIDDLE-SUM LEMMA (a literal J-window's middles sum to >= floor((J-2)/2)q', so the flank
  envelope must collapse at rate q' per two levels of J), the J-PARITY OF PALINDROMES (for J
  even a literal legal window is NEVER a palindrome; for J odd the class word is forced
  palindromic), the PEEL BOUND, and the exact table: Delta_J = Q*_J - F_2 lies in [-3,+4]
  uniformly in BOTH M and J at m11..m41 (against s_min growing linearly), Delta_5 = 0 exactly
  at both machines where J = 5 is non-empty, J_max(M) = A_kill(M) + 1 at all eight censused
  machines, and A_kill(41) = 3 exactly (project item O7 closed).  THE SHARP FORM is one
  inequality per legal middle word - 36 rows over the whole corpus, of which exactly THREE
  fail, all at machine 31, all containing the padded letter 37.  The manager's
  "extremal implies palindromic" step is TRUE at J = 5 (2 of 2, unique self-reverse
  maximisers) and FALSE at J = 3, 4 (15 cells, none) - Theorem B explains the even-J half -
  PROVED (Theorems A-D) + SCRIPT-VERIFIED (exact, gated against R68's independent Q* table
  and reproduced by a second scan-free CRT vehicle) - CHECKED 2026-09-03 (harvester r30):
  NOVEL AS FAR AS SEARCHED for Q*_J, Theorems A, B, D and the Delta_J table; PARTIAL
  OVERLAP for the merge mechanism (Holt-Rudd arXiv:1408.6002 Lemma 2.1 / Lemma 3.1, one
  class, no alternation); the one-class mirror is Holt-Rudd remark (v)
  ROUND 29 (Constructor): the family's DEPTH quantifier is closed - J_max(M) = L(M) + 2 with
  L the longest realised legal letter word (see even-j-mechanism), so "J_max = A_kill + 1"
  is proved rather than measured and every EMPTY cell of the Delta table is free.  The
  literal Delta_J table is reproduced by a third independent vehicle
  (research/evenj_r29.py) and gains the m31 literal J = 4 maximiser (6,25,12,28), span 71,
  Phi = 34, not previously exhibited
- spectrum-depth-certificate - F(M+q') <= max_{2<=J<=J_max} F_J(M), from R68's attainment
  theorem plus "Q*_J <= F_J" and "emptiness is upward closed": (D) at a step follows from the
  OLD machine's spectrum over a FINITE depth range, with no word list, no flank envelope, no
  CEGAR loop and no realisability oracle.  It CERTIFIES (D) AT 41 -> 43 - THE NINTH RUNG,
  margin +16, left open by two previous rounds - from F_2(41)=103, F_3(41)<=117, F_4(41)=118
  and the new Q*_5(41) = -inf; corollary F(43) <= 118 from machine 41 alone.  The criterion
  is genuine, not a restatement: it certifies 8 of the 9 steps whose spectrum is complete and
  FAILS at 29 -> 31, where the new exact value F_5(29) = 85 is 11 over budget while
  Q*_5(29) = 55 - PROVED + SCRIPT-VERIFIED (every input asserted against the corpus; the one
  new input re-derived from scratch with 0 undecided) - CHECKED 2026-09-03 (harvester r30):
  PARTIAL OVERLAP in the one-class shadow (Holt-Rudd arXiv:1408.6002 Lemma 2.1 = the
  one-class merge, Theorem 2.3 = the CRT step of the deletion-ladder cap, Lemma 3.1's
  g < 2p threshold = the one-class A_kill = 1 regime); the certificate with a finite depth
  cap and its A_kill scope NOVEL AS FAR AS SEARCHED
  ROUND 29 (Constructor): THE TENTH RUNG, 43 -> 47, RATIFIED - max_J F_J(43) = 132 <= 150 =
  F(43) + 47, MARGIN +18, from Mechanic's exact unconditional F_2(43)=116, F_3(43)=125,
  F_4(43)=132 and J_max(43) = 4; every input re-asserted against its own deletion-ladder cap
  from a clean process (research/rung10_r29.py).  AND THE DEPENDENCY IS NAMED: those three
  values' EXHAUSTIVENESS rests on span caps F(47)/F(53)/F(59), corpus values ABOVE machine 43,
  so rung ten - like every rung below m59 - is a METHOD rung, not a logically independent
  bound on F(47).  The independent version is priced exactly: 38,072 CRT decisions at machine
  43 (J=2/3/4 bands above 150 up to j.F(43)), at a measured 30-46 s and only ~25% decided per
  instance at a 300,000-node budget
- even-j-mechanism - the even-J half of the increment law, which the round-28 palindrome
  route provably cannot reach (Theorem B forbids literal even-J palindromes).  THE WORD
  REDUCTION (proved; the forward half is Mechanic's r28 index observation, the converse is
  new): Q*_J > -inf iff L(M) >= J-2 with L the longest REALISED legal letter word, so
  J_max = L+2 and A_kill = L+1 - R81's MEASURED 8/8 identity is a theorem, and J = 6 is
  certified empty at every machine on record from one dictionary fact.  THE SAME-TOOTH LEMMA
  (proved, 38/38): the middle span is 0 mod q' exactly when the number of non-padded middles
  is even, so a literal even-J chain starts and ends on the SAME tooth.  THE IDENTITY IS A
  TWO-WAY TOOL: L(43) = 2 and L(47) = 4 were computed this round directly from the word
  dictionary (8 and 4 CRT calls, 0 undecided, no census and no period), re-deriving
  A_kill(43 -> 47) = 3 and INDEPENDENTLY CONFIRMING the round-25 value A_kill(47 -> 53) = 5;
  the realised word (18,35,18,35) at m47 is the first legal 4-word recorded in the project.
  ROUND-30 ADDENDUM (section 7): eps(v) = d - g_out is a LEMMA (kept-flank deficit minus
  new outer flank, asserted 30/30, both terms > s_min at four literal cells - a
  cancellation, not a smallness); the pre-registered order-statistic mechanism is REFUTED
  at every padded failure cell by the COUNTED census; the padded failure is the F_3 WALL -
  at m31 the F_3 maximisers are (18,37,30)/(30,37,18), the old machine's depth-3 record with
  the PADDED LETTER as its middle, Phi(37) + 37 = F_3(31) exactly, and at no other machine
  m11..m29 does the F_3 maximiser have a legal middle (gated).
  THE NEW OBJECT is
  the PAR-TRADING RESIDUAL eps(v) = Phi(u) - Phi(v) - x, the failure of the flank envelope to
  pay for one added letter: Delta_J = Delta_{J-1} - eps, so "Delta_J = O(1) uniformly in J"
  splits into "eps = O(1) per letter" and "L(M) bounded".  MEASURED: |eps| <= s_min at 14 of
  14 LITERAL cells and 10 of 16 PADDED cells (all six failures carry the letter q'), and
  max |eps| = 4 along the maximising chains, against s_min 4..14 - PROVED (two theorems) +
  SCRIPT-VERIFIED (21 of 22 recorded Q*_J cells reproduced, 0 mismatches) - CHECKED
  2026-09-03 (harvester r30): NOVEL AS FAR AS SEARCHED for the word reduction, the
  same-tooth lemma and eps; the "runs of consecutive gaps in residue classes" theme is a
  prime-side EXISTENCE literature (Shiu 2000; Banks-Freiberg-Turnage-Butterbaugh
  arXiv:1311.7003 via Maynard-Tao; Maynard 2016) or one-class at run length ONE (Ziller
  2020 D(k)); Holt-Rudd remark (vi) is the one-class run constraint
- dictionary-monotonicity-onset - TWO findings.  (a) THE DEPTH-0 LEMMA: the realised gap
  m-tuple dictionary is MONOTONE along the machine ladder, D_m(M) subset D_m(M+q') for
  every q' > 2(m+1) - a new gear destroys openings and merges gaps, yet every old
  configuration survives, because the pattern forbids at most 2(m+1) < q' phases and CRT
  supplies a lap with an admissible one.  Checked at all six exact pairs 13->17 ... 31->37
  and against the round-27 exact m41 shard, and at output arities 2,3,4,5,6,7; it decides
  16.7% of the m41 arity-4 superset (145,907 reverse classes, 291,675 tuples) as YES with no
  solver, at every span.  THE HYPOTHESIS IS SHARP: at q' = 17 and 19 the first m at which
  monotonicity fails is EXACTLY the first m the proof does not cover (slack 1 at q' = 11, 13),
  and every failure witness is a dense small-gap pattern.
  (b) THE INFLATION-ONSET LAW: the dictionary transfer M -> q' is exact below a sharp span
  and over-generates above it, and that span is  min span of D_4(q'') \ D_4(q')  with q''
  the NEXT prime - "the transfer first over-generates exactly where the next machine's new
  repertoire begins" - onset ladder 13, 15, 17, 25, 31, 41, 53, 68 at 11->13 ... 37->41; the
  causal version (the tuples refuted at the onset span are realised at the NEXT machine) is
  8 of 8 and the refined law with it (the right-hand side must be intersected with what the
  transfer can EMIT), while the SIMPLE form is 7 of 8 at arity 4 and only 2 of 6 at ARITY 3 -
  so the intersection is the law, not a patch.  THE LAW IS ARITY-INDEPENDENT: over output
  arities 2, 3, 4 and 5 the refined form is 20 of 20 and the simple form 12 of 20, and the
  depth-0 lemma holds at every one of those arities.  AND one step OUT OF SAMPLE
  (nu(41->43) = 68 from the m41 shard alone reproduces round 27's measured onset(37->41), by
  a route that never saw it).  Three pre-registered closed forms in F, F_2 and ratios were
  all REFUTED: the onset is a recursion in the ladder, not a letter combination.  Mechanism half-explained and quantified: X_5(M) = 9 with the
  SAME witness (1,2,3,2,1) at every machine, phase-saturated at gear 5, which explains the
  unscreened onset (9 at every step) exactly; and Y_5 (the same minimum with the saturated
  walks removed) = 10, 17, 18, 22, 30, 38 at m13..m31 against onsets 15, 17, 25, 31, 41, 53,
  so the remaining multiplicity residue is a near-constant FACTOR (onset/Y_5 = 1.389, 1.409,
  1.367, 1.395 at the four largest machines, a band of width 0.042) rather than a growing gap.
  Machines 29 and 31's exact 5-TUPLE dictionaries (208,668 and 636,575 tuples, by streamed
  full-period passes over 1.08e9 and 3.34e10 slots) are new objects, each gated by its induced
  4-tuple dictionary matching the round-25 full-period census cell for cell.  Also proves the WALK SCREEN
  (screen the transfer's walk, not its emission - sound, strictly stronger, subsumes the
  round-26 screen, 2.4-11.7% tighter supersets) - (a) PROVED + SCRIPT-VERIFIED,
  (b) MEASURED 6/6 - CHECKED 2026-09-03 (harvester r30): (a) PARTIAL OVERLAP - the
  arity-1 one-class case IS Ziller, arXiv:2007.01808, Proposition 2.7 "Propagation of
  coverings" (m in D(k) => m in D(k+1)), framing credited by Ziller to de Polignac 1849;
  the arity-m two-class lemma with its sharp hypothesis q' > 2(m+1) NOVEL AS FAR AS
  SEARCHED (Holt-Rudd Lemma 3.1 is the one-class survival COUNT under g < 2p); (b) NOVEL
  AS FAR AS SEARCHED; citation paragraph written into the doc's section 6
- jk-growth-discriminator - THE k-AXIS REPLACES THE z-AXIS.  (i) THE FIRST EXACT VALUES OF
  j_k FOR k >= 3: j_3(P(z)) = 180, 306, 612, 972 at z = 11, 13, 17, 19 (round 27 had only
  6, 24, 78 at z = 3, 5, 7 and recorded that z = 11 "needs a real algorithm"), plus
  j_4 = 30, 150, 420, 1230 and j_5 = 180, 930, 2070, 5490 - each with a machine-verified
  witness AND an exhaustive proof that one more position cannot be covered.  The engine
  (reduced lattice j_k = D(m+1) with D = prod_{p<=k+1} p, generalising Hagedorn's 2w+2 and
  Ziller-Morack's 6*omega_2+6; the RPA2 canonical form, worth 125x; and a sliding
  prefix-window capacity bound) reproduces FOURTEEN published A048670 values and NINE
  published A288815 values by a DIFFERENT algorithm - the first independent verification of
  the paired Jacobsthal numbers since they were deposited.  (ii) THE METHOD: the two live
  growth readings of h_2 differ by (log z)^(k-1), so at k = 2 they need z = 151..251 to
  separate - a number nobody has computed in nine years (A072753 and A288815 both still
  carry exactly 21 terms; read first-hand 2026-08-29) - while at k = 3 they are two logs
  apart and at k = 5 four, and those values cost seconds.  MEASURED: against the
  parameter-free random-choice model, R_1 is FLAT to 4% over eighteen values (the
  calibration, where the two models coincide and the truth is known), R_2 drifts +8% on the
  clean window where model (A) needs 0% and model (B) needs +37%, and the calibrated
  A-to-B fraction f_k FALLS steeply with k on every matched window where model (B) requires
  it to be CONSTANT.  The excess exponent a_k - k is 0.61, 0.56, 0.76 at k = 2, 3, 4 against
  a calibration bias of -0.08 and against the k-1 = 1, 2, 3 that the (2k-1) reading needs.
  HONEST SCOPE, stated in the doc: (P2') has no finite-z content below log x ~ 300, so this
  is evidence about the TRUTH's shape on the computed range and NOT a refutation of the
  theorem.  ROUND-29 ADDENDUM (section 9): THREE NEW EXACT VALUES - j_3(P(23)) = 1398,
  j_4(P(17)) = 2340, j_4(P(19)) = 3810 - and the model question DECIDED at two clean
  POST-TRANSIENT steps, both pre-registered with a number from each model before the
  answer existed: at k = 3, 19 -> 23 the prediction was 1398 under (A) and 1590 under (B)
  and the answer is 1398 EXACTLY TO THE UNIT (R_3 flat to 0.13% where (B) needs +13.4%);
  at k = 4, 17 -> 19 R_4 FELL 4.56% where (B) needs +12.2%.  Round 28's "the excess does
  not grow with k" is WITHDRAWN against itself and replaced by something sharper - the
  excess is a CONSISTENT FRACTION of what (B) demands, e_k/(k-1) = 0.61, 0.37, 0.48, 0.43,
  so the truth on the computed range looks like z(log z)^(k + 0.45(k-1)), strictly between
  the two models AND AT THE SAME PLACE AT EVERY k.  Also: round 28's own j_3(23) split run
  was INVALID BY ITS OWN PROTOCOL (two of fourteen workers beat the seed, so the parts need
  not partition the tree - a branch-and-bound split is a proof only when the incumbent is a
  fixed point of the run), caught here and reproved TWICE; and the round-28 ILP hole is
  closed by measurement - CaDiCaL on Ziller-Morack's own eq. (2.2) generalised to k classes
  proved j_3(P(23)) = 1398 in 831 s on ONE core against the DFS's 13.6 core-hours, which
  makes j_3(P(29)) purchasable (~17 core-hours) after the DFS priced it at ~3,500.  A PRICE
  IS A PROPERTY OF A VEHICLE, NOT OF A TARGET - this doc learned it twice.
  Also prices brief item (a) with a measured cost curve: z = 37 is the next
  purchasable rung (~15 h at 16 cores), z = 41 is 51 days - SCRIPT-VERIFIED (values) +
  MEASURED (statistics) - prior art checked first-hand 2026-08-29, NOVEL AS FAR AS SEARCHED
- anchor-235-layer-laws - the anchor-2,3,5 line's per-layer laws as one register entry
  (written by the harvester, round 30, from anchor-235.md 9d-9g, Formalist r29 and
  Mechanic r29): (L1) THE CHAIN LAW - two slots lie in a common two-class set {r, r+d}
  iff their difference is 0 or +-d mod g, so two consecutive lower openings are both
  deleted by gear g iff their gap is 0 or +-d_g mod g, with the T3 half (no two steps the
  same way) - KERNEL-CHECKED for every g (AnchorChain.chain_law, no_two_up/down);
  (L2) NEIGHBOUR-OF-HIT - the neighbour of a hit is never a hit for every g >= 5, from
  6u = 1 alone (d = 3^{-1} is never +-1) - KERNEL-CHECKED (neighbour_of_hit);
  (L3) THE PHASE-REDUCTION RECORD LAW - the g copies of the lower period realise every
  deletion phase exactly once (copy_phase + phase_bijective, KERNEL-CHECKED), so the
  next record is max over phases of gap-before + run-span + gap-after over two-class runs
  on ONE lower period: F_bc = 4,6,10,17,24,33,42 (corpus 5..43) at {5..7}..{5..29} and
  58, 88, 91 at 31/37/41 SCRIPT-VERIFIED exact, and at machine 17 KERNEL-CHECKED at both
  ends (AnchorRecord17.record_max, phase_is_machine, F17_eq_18 - the attainment at 17 is
  new) with the nested formula's recursion a theorem (hop_iter); (L4) D_g = A_kill(M -> g),
  7 for 7 (Mechanic C49), so D_g = L(M) + 1 - SCRIPT-VERIFIED; D_g bounded OPEN - CHECKED
  2026-09-03: PARTIAL OVERLAP (the copies-and-phases picture is Holt-Rudd arXiv:1408.6002
  Lemma 2.1 / Theorem 2.3, one class); the two-class chain law, neighbour-of-hit as a
  theorem for every gear, the record law as a computation of the maximal gap, and
  D_g = A_kill NOVEL AS FAR AS SEARCHED
- cover-half-counter-ladder - (constructor, round 30) THE EXPOSURE CAP AND THE COUNTER
  LADDER: for a word length m, A_m (abstract T3-legal words, closed form) >= S_m (the
  exposure half: words whose prefix-sum set has a slot with all points open, = R43's
  depth-0 term >= 1, = phase saturation at every gear) >= S_m^(2) >= S_m^(4) >= D_m (the
  cover half: realised words; D_m > 0 iff L(M) >= m).  THEOREM: exposure at length m is
  decided by the gears <= 2m+2 alone.  MEASURED, exact, m11..m53: EXPCAP(M) = max{m :
  S_m > 0} = 1,1,1,4,2,3,5,18,13,10,5,21 against L = 1,1,1,2,1,3,3,2,2,2,4,3, so EXPCAP - L
  is unbounded along the ladder (16, 11, 8, 18 at m37, m41, m43, m53); fixed-depth
  Bonferroni (depth 2 and 4) kills NOTHING among the exposure survivors at 21 cells while
  the exact count sits 1e2..4e6 below the depth-0 term.  VERDICT (gated for fixed depth):
  no fixed-depth truncation of the inclusion-exclusion counter and no exposure-only
  argument bounds L uniformly; the term that grows is the P-scale depth-0 count, and a
  uniform bound on L needs the cover half at full depth on a candidate set that is itself
  unbounded.  R75's CORRCAP row is reproduced exactly as the gate - SCRIPT-VERIFIED (exact
  integers) - prior art not yet checked
- bare-word-uniform-cap - (constructor, round 31) THE FIRST UNIFORM CAP ON HALF OF L:
  with a, b the two BARE letters (the smallest positive values in the two nonzero legal
  classes, a + b = q', 3a = q' -+ 1), a bare legal word is forced by T3 to be one of the
  two alternations abab.. / baba.., and a realised word's prefix-sum offsets must fit
  inside the exposed sets of EVERY gear of M - in particular gears 5 and 7.  Hence
  L_bare(M) <= PSORD(q' mod 210) <= 5 for every machine, where PSORD(c) is the longest
  bare alternation (max over the two phases, counted in letters) admissible at {5,7}.
  PSORD takes the values 1 (24 classes), 2 (4), 3 (14), 5 (6, exactly R74's
  {37,53,83,127,157,173}); PSORD = 4 is EMPTY.  S = {PSORD <= 2} has 28 classes (density
  7/12 of the primes), and on S the LEMMA reads L_bare(M) <= 2.  Corpus gate m11..m47:
  L_bare = 1,1,1,2,1,3,3,1,1,1,4 against L = 1,1,1,2,1,3,3,2,2,2,4, so L_bare <= PSORD
  everywhere (tight at m29 and at m37/m41/m43) and <= 2 at all seven S-machines.  This is
  NOT R74's A_relax: R74 minimises over the two phases and counts points (it asks for a
  cycle), this maximises and counts letters (it asks for existence); R74's own 24/16/2/6
  distribution is reproduced in R74's convention as the gate.  CONSEQUENCE: with
  L = max(L_bare, L_pad), requirement (B) is now exactly "L_pad bounded" - the words that
  use a letter of size >= q'.  Immediate new value: at m53, PSORD(59) = 2 gives
  L_bare(53) <= 2 and the recorded L(53) = 3 forces L_pad(53) = 3 exactly; and L_pad(47) =
  3 measured (three non-bare 3-words realised by CRT, every non-bare 4-word refuted by
  R98), so L_pad = 0,0,0,1,1,1,2,2,2,2,3,3 at m11..m53 and GROWS while L_bare is capped
  forever.  KERNEL-CONFIRMED the same round: Formalist's proofs/BareAlternation.lean
  defines the identical 28-class S and proves S_card, psord_le_five, psord_ne_four by
  decide - PROVED (the cap) + KERNEL-CHECKED (S, PSORD <= 5) + SCRIPT-VERIFIED (the
  tables) - prior art not yet checked
- spectrum-bound-on-L - (lateral, round 31) THE OTHER HALF, AND IT COVERS PADDED WORDS:
  every legal letter is >= a or >= b in its nonzero class and >= q' if padded, T3 makes
  the nonzero classes strictly alternate, and a + b = q' exactly - so a realised legal
  word of m letters, p of them padded, has span >= (p + floor((m-p)/2)) q'.  R68's
  attainment theorem (proved) caps that span by F(M+q') - 2.  Hence, with
  T = floor((F(M+q') - 2)/q'),  L(M) <= 2T + 1 - p, and unconditionally
  L(M) <= 2 F(M+q')/q' + 1: L is O(F/q'), NOT O(1), and hypothesis (B) as posed is
  probably false in the limit and is not needed.  Corpus bound row 1,1,3,3,3,3,5,5,5,5,5,5
  (parity-refined 1,1,2,3,3,3,5,4,5,5,5,5) against L = 1,1,1,2,1,3,3,2,2,2,4,3 - TIGHT at
  m11, m13, m29; beats Constructor's EXPCAP at five of twelve steps (5 vs 18 at m37, 5 vs
  21 at m53).  Substituted back into R99 without circularity it gives
  F(M+q') <= (q'(F_2 + c_A) - 4 c_A)/(q' - 2 c_A) and hence (D) whenever
  8 F <= q'^2 - (F_2 - F + 12) q' + 16 - true at 8 of 13 corpus steps, margin growing
  (F/RHS 0.87 at m41 -> 0.57 at m59), F/q'^2 = 0.038..0.052 throughout.  Gated at all
  165,584 rows of the tooth-counterfactual family, 0 violations - PROVED (paper) +
  SCRIPT-VERIFIED - prior art not yet checked
- walk-tooth-frame - (R2.a, round 37) the walk from q^2 under {5..q} STARTS ON A TOOTH of the
  top gear (6 k_0 = q^2 - 1 = -1 mod q), so the top gear's next strike is d = 2c mod q columns
  on (2 u_q if q = 5 mod 6, q - 2 u_q if q = 1 mod 6) and it strikes the whole walk exactly
  once, at the first column - one exception in 667 walks (q = 53); the deepest layer of the
  walk is the top gear itself IFF q^2 - 2 is prime (the square gate), ZERO exceptions in 667
  walks (153 open, 514 shut); the pair born at the landing strikes the next level's walk once
  each at distance 2k = (g+1)/3, its own twin-product column, 0 failures at 667 landings; and
  a LEVEL-FREE TRANSFER RULE - a gear striking the column j away from a birth column strikes
  the column i steps into that pair's own walk iff it divides (6j)^2 + 6i - 2 or (6j)^2 + 6i
  (lower-member flank) or (6j+2)^2 + 6i - 2 or (6j+2)^2 + 6i (upper), with neither k nor g in
  the condition: 832,915 checks, 0 mismatches, and the j = +-1, i = 0 set is exactly
  {7, 17, 31}, confirmed by census (3,093 carry-overs of 50,906, no other gear) - PROVED
  (elementary) + SCRIPT-VERIFIED (research/anchor235/r37/sf_walks.py, sf_birth.py) - prior art
  NOT YET CHECKED

- walk-path-transforms - (W.t, round 38) WHAT BUILDS THE PATH OF THE WALK FROM q^2, by order of
  interaction, every prime gear q = 5..19,997 (2,260 walks). ORDER 0 (the anchor): q^2 = 1 or 19
  mod 30, so the walk STARTS ON THE 29|31 SLOT when q = +-1, +-11 mod 30 and on 17|19 when
  q = +-7, +-13 mod 30, NEVER on 11|13 (0 of 2,260); hence gear 5 never strikes the first column,
  ALWAYS strikes offset 1 (so L >= 2 at every q > 5), and its whole contribution is fixed in
  advance at the offsets 1, 4 or 1, 3 mod 5 - exactly 2/5 of the path (0.4025 of 88,677 columns),
  with P(next letter = 5 | this letter not 5) = 2/3 by residues. ORDER 1 (one gear): gear g
  strikes OFFSET i iff i = (2-q^2)6^-1 or -q^2 6^-1 mod g - two progressions of difference g
  separated by d_g, phase a function of q^2 mod g (0 disagreements in 493,101,490 checks), whence
  (i) THE QUADRATIC-RESIDUE BAR - g can reach offset i only if 2-6i or -6i is a nonzero QR mod g,
  so 3/4 of the machine at a generic offset (median 0.7473), ALL of it at i = -6t^2, and EXACTLY
  the gears g = +-1 mod 8 at i = 0 (0 of 3,212 first-column strikers outside that class), and
  (ii) THE SQUARE PHASE VECTOR - the phase vector of the walk is a square in every coordinate, one
  part in 2^pi(q) of the phase space; (iii) the q^2 tooth is the UNIQUE SOLE-STRIKER TOOTH of the
  top gear in its own window (0 of 337,011), so the walk starts at the shallowest tooth (depth
  2.4212 vs 3.2692) and the square gate is a half-machine event; (iv) ONE TOOTH PER RUN, two-sided
  (L < d and L^- < q-d): 2 exceptions in 2,260, q = 53 forward and q = 31 backward, each in the
  SHORT arc of its own class (max path/arc 1.50 short vs 0.77 long). ORDER 2, needed exactly once:
  the depth profile's dip-plateau-spike (2.42 / 3.27 / 3.77) against the machine's own values
  sum 2/g = 3.1805 and sum 2/(g-2) = 3.7007, the spike being neighbour-of-hit gear by gear. The
  per-offset mean depth is a function of the offset alone (root counts of -6i and 2-6i,
  correlation 0.9694, range 2.05 to 5.82) and the columns k_0 - 6t^2 are blocked at every q while
  no such column lies within the forward arc. NULLS: no spectral or local-density distinction of
  the position of q^2 (percentiles 0.37-0.50), L typical (percentile 0.5270 among random blocked
  columns; 47.3% of the other teeth's walks longer), nothing crosses chain levels except the frame
  (49 levels). LOWEST-ORDER UNEXPLAINED FEATURE: the length L itself - PROVED (elementary) +
  SCRIPT-VERIFIED (research/anchor235/r38/pt_path.py, pt_qr.py, pt_spectrum.py, pt_levels.py) -
  prior art NOT YET CHECKED
- walk-path-parts - the path from q^2 taken apart: anchor pinned by q mod 30, gear-5 offsets {1,4}/{1,3} mod 5, L never 1 mod 5, 15-class law mod 35, top gear inert beyond offset 0, offset character law (mean depth a fixed function of the offset), re-phasing shortens only for sole strikers - EXACT to q = 19,997 - prior art NOT YET CHECKED (research/proof/walk_path.md)
- reachability-landscape - the q-free set of gears that can reach each offset past q^2 (bar size closed form, islands = CRT classes, doubling law, large gears strike islands at exactly 2/g); the island witness: for every prime 1489..19997 an offset in {5,10,12,17} mod 35 below the top gear's next tooth is struck by no gear - EXACT - prior art NOT YET CHECKED (research/proof/reachability.md)
- island-witness-integers - the island witness holds for every integer coprime to 30 above 2849 (multiples of 5 fail by a proved relocation law), 0 exceptions in 17,748 primes to 200,000, one class 12 mod 35 suffices from 5477, free island within 0.152 d and absolute offset <= 2,392; cover number K(d) grows (3..20 at d = 35..1120) while counting stalls - EXACT - prior art NOT YET CHECKED (research/proof/island_witness.md)
- flank-walk - the junction theorem (junctions of M+q' are ordinary openings of M, the max flank sum at junctions is F_2(M)); L6 exact with its two forced consequences; three gear bands with the middle band striking at a constant 0.796; flanks coupled by the anchor mod 5 and 35; the window has at most two junctions (columns of q' and q'^2) with flanks d_0 and the walk from q'^2 - EXACT - prior art NOT YET CHECKED (research/proof/flank_walk.md)

## The wheels and the exhaust - prior art checked 2026-09-06 (harvester)

The wheels are the top machine (the primes in `(q, q#]` on the raw line, teeth at 0 and -2,
dominoes {x, x+2}); the exhaust is every tier above them.  Their 85 + 8 laws now carry ONE
project-wide numbering, W1-W85 and X1-X8, in **research/proof/law_register.md**, which holds the
statement, status and prior-art verdict of every one.  Counts: 14 KNOWN, 18 KNOWN VARIANT,
38 NEW, 22 STANDARD TOOL, 1 REFUTED.  This discharges objects_ledger O-W7 and O-X5 and the
"prior art not checked for the machine as an object" line in docs/proofs/22.  Entries below name
every NEW and KNOWN VARIANT law; STANDARD TOOL and KNOWN rows are in the register only.

  **CORRECTION 2026-09-06 (harvester r1, by computation - research/harvest/r1/jacobsthal_check.md).**
  This entry first read "ONE NAME TO ADOPT: `F_top + 1` IS Ziller-Morack's paired Jacobsthal
  function `j_2`, and their Conjecture 6 IS this project's window target".  **Both halves are
  false.**  `j_2` (arXiv:1706.00317 Def 2.1-2.2) quantifies OVER the even difference `D = b - a`
  of the paired progression, which by CRT is two ARBITRARY residue classes per prime; so `h_2` at
  primorials is the FREE-RESIDUE two-class covering record, `A288815 = 6 A072753 + 6` (A072753 =
  Resta's cover of `1..m` by arbitrary pairs mod each prime) - the adversary of the_wall.md face
  5a, not the real teeth.  The project's `F(M)` is the single instance `D = 2`.  Verified
  exhaustively: `F(M) - 1` = 1, 4, 6, 10, 17 against A072753 = 2, 4, 10, 24, 31 at
  `{5}, {5,7}, ..., {5..17}` - **equal at `{5,7}` alone, strictly below everywhere else**, and
  `F(M) - 1 <= A072753` is a theorem, `D = 2` being one competitor in the maximum.  Hence
  **Ziller-Morack's Conjecture 6 is the ADVERSARIAL window statement, strictly stronger than the
  project's**: `F(M) <= A072753 + 1 < (p_n^2 - p_n)/6 < (p_{n+1}^2 - 1)/6` means Conjecture 6
  IMPLIES the project's window statement (and, by their Theorem 4.1, Goldbach and prime pairs at
  every even difference), while the converse fails.  The project's window statement is Conjecture
  6's real-teeth specialisation.  The correct name for `F_top` is the **real-teeth record**, not
  the paired Jacobsthal function; jk-growth-discriminator.md below already had this right.

- wheels-domino-form - W3 the partner law (every strike has a partner exactly 2 away, so a gear's
  struck set is a disjoint union of dominoes {x, x+2}), W4 the forbidden gap 4 (n, n+4 open
  implies n+2 open), W54 the holes (d = 4 in the pair view, d = 2 and 3 in the triple view) -
  KERNEL-CHECKED, no hypothesis (`partner`, `strikes_iff_domino`, `open_of_open_add_four`,
  `no_start_gap`) - NOVEL* (checked 2026-09-06: the ingredient "{0, g-2} are 2 apart" is in every
  twin-sieve paper; the domino inventory and the forbidden gap are nowhere.  The one-class
  analogue question - which differences occur between consecutive numbers coprime to a primorial
  - is Ziller arXiv:2007.01808; the two-class question was never asked.  HONEST: two-line proofs,
  "immediate but unrecorded")
- wheels-record-parity - W17 the parity law F_top = 2m - (m mod 2) for gears above 2m+1 (the
  record is decided by the PARITY OF THE GEAR COUNT and by nothing else - not by the gears'
  sizes), W71 the sharp threshold (iff q' >= 2m+1 at even m, 2m+3 at odd m, iff the record cover
  is a free domino tiling), W29 the collision law (no three traces pairwise intersect, so the
  cover is a perfect tiling at even m and wastes exactly one unit at odd m - the defect in
  geometric form), W44 the triple record F_3 = 3m EXACTLY with no defect (a solid triomino tiles,
  a gapped domino cannot: the defect is a property of the tooth SEPARATION, not the tooth count),
  W38 removal independence, W30 the record of a triple/quadruple, W40 the bound re-proved from
  the mex - KERNEL-CHECKED as equalities (`parity_law`, `triple_law`, both `IsGreatest`) +
  MEASURED (W29 11 of 11, W71 7 boundary pairs, W30 exhaustive over 8,855 sets, 0 exceptions) -
  NOVEL* (checked 2026-09-06.  Nearest: the classical ONE-class proposition "q_1 > omega(n)
  implies j(n) = omega(n) + 1", background in Erdos, Math. Scand. 10 (1962) 163-170 and
  Ziller-Morack arXiv:1611.03310 - the size-independence half is its direct analogue, but THE
  PARITY DEFECT HAS NO ANALOGUE IN PRINT, because the one-class case has no defect.  W29 is the
  strongest claim: the covering-systems literature (Mirsky-Newman, Znam, Hough Ann. of Math. 181
  (2015), Balister-Bollobas-Morris-Sahasrabudhe-Tiba) is about infinite covers of Z with distinct
  moduli and reciprocal sums and would not have found the waste parity of a finite two-class
  interval cover.  CAUTION: W52 is NOT independent evidence - its proof needs W29)
- wheels-gap-census - W22 the gap census law N_d as inclusion-exclusion over CRT products, W23
  the universal signature and the collapse threshold, W24 W1 closed (the counts of gap 3 and gap
  5 are EQUAL in every wheel whose gears exceed 7 and unequal exactly when 7 is a gear, because
  both share the signature prod(g-4) - 2prod(g-5) + prod(g-6) and only gear 7 collapses a class;
  (3,5) the only coincident pair, 4 the only identically zero length, d <= 16), W25 the degree
  law in elementary symmetric polynomials (degree m - r(d), gear-independent exactly when
  r(d) = m, which DERIVES W18's universal record multiplicity), W26 r(d) = the parity covering
  number with d = 4 the unique boundary exception, W85 both surviving onto the range - PROOF
  (CRT + inclusion-exclusion; not in the kernel) + MEASURED (15 wheels every gap length and the
  full 6.7e9 period of an eight-gear wheel to d = 16, 0 mismatches; W25 rests on the UNPROVED
  vanishing M_k(d) = 0 for k < r(d), open item O-W1) - PARTIAL OVERLAP / KNOWN VARIANT, checked
  2026-09-06: **Steven Brown, arXiv:2311.06873 / Notes on Number Theory and Discrete Mathematics
  30(1) (2024) 81-99** is the ONE-class version with the same shape - the CRT product
  nu(T) = prod(p - card(T mod p)) (Thm 3.1), the inclusion-exclusion over the uncovered interior
  (Thm 2.1), the threshold universality (Lemma 3.1) and even a coincidence of the same type
  (K(2,P) = K(4,P), eqs 16-17, though for a DIFFERENT reason: his is parity annihilation, ours is
  equal class counts).  CITE HIM.  Delta: two classes per gear, arbitrary pairwise-coprime odd
  gears.  Holt is NOT the blocker - his cycle-of-gaps machinery (arXiv:1408.6002, 1510.00743,
  2502.20470, 2603.25915) is RECURSIVE with only N_2 = prod(q-2) in closed form.  W25, W26 NOVEL*
- wheels-run-spectrum-duality - W11 the run spectrum is the SECOND DIFFERENCE of
  A(L) = prod(g-2-L), hence a polynomial of degree m-2 (an arithmetic progression of common
  difference exactly 6 at three gears, second difference 24 at four, top length occurring
  prod(g-q'+1) times); W45 the dual C-identities (the gap census is the second difference of the
  all-struck count C, and F_top = max{j : C(j) > 0}); W46 the closed form of C(j) as a signed sum
  of shifted wheel products with T(j,k,e) a path convolution - and the structural moral that
  "all open" is a per-gear condition giving ONE PRODUCT while "all struck" is not and gives a
  SIGNED SUM; W50 the consequence (the spectrum decides the run record and provably CANNOT decide
  the blocked record) - PROOF + MEASURED (12 wheels every L; all five C-identities 10 wheels;
  T brute-forced to j = 12; spectral minima 3.3e-07 to 3.1e-05, never 0) - NOVEL* (checked
  2026-09-06: A(L) itself is the Schemmel totient (Schemmel 1869) / admissible-tuple CRT count
  and is STANDARD; the second-difference law, the degree structure, the AP-of-6, the C-identities
  and the T(j,k,e) form were located nowhere.  Ingredients standard - binom(k-1,e)binom(n-k+1,k-e)
  is the Kaplansky/Riordan subsets-with-adjacencies count.  THE SHARPEST RE-EXPRESSION FOUND:
  F_top as the support edge of an exactly computable function, where the whole Jacobsthal
  literature only bounds it)
- wheels-mex-walk - W39 the mex form (the next open pair after x is x + mex of the 2m residues
  {(-x) mod g, (-x-2) mod g}, exact whenever every gear exceeds 2m, O(m) operations, no scan and
  no period), W66 the SHARP criterion M(x) < q' implies exactness (checkable at each x, contains
  W39), W43 the three-teeth twin-candidate form, W73 the certified column mex for the motor,
  W47/W74 the hop collapse (chain length at most 2 with an iff, in either coordinate) -
  KERNEL-CHECKED as `IsLeast` (`mex_form`, `triple_mex_form`) + MEASURED (1.4M positions nine
  wheels; 94,774 positions 18 wheels for W66; 890,501 certified column walks; 391,048 hop
  positions, 0 exceptions throughout) - NOVEL* (checked 2026-09-06: "mex"/minimal excludant
  appears in combinatorial game theory and in partitions (Andrews-Newman 2019) and in NO
  number-theoretic sieve; the nearest functional prior art is Pritchard's wheel sieve, which gets
  the next coprime by table lookup over a period, not by an O(m) residue formula.  W47 is a
  KNOWN VARIANT of Holt-Rudd arXiv:1408.6002 Lemma 2.2 - "each closure of adjacent gaps occurs
  exactly once per copy" - in two classes.  HONEST: W39 is the constructive content of W17's
  threshold, and F_top < q' fails badly at the real anchor, so it is a primitive and a base case,
  not a route to the window)
- wheels-symmetry-and-census-parity - W8/W70 the affine symmetry group (exactly n -> c(n+1) - 1
  with c = +-1 mod every gear, order 2^{#odd gears}, adjacency-preserving subgroup Z/2), W7 the
  mirror n -> -n - 2 with the shield n = -1 its unique fixed point, W35 the cyclic gap word read
  from the shield is a PALINDROME, W67 exactly one gap length has an odd count and it is the
  mirror-self-paired gap, W6 the origin clump of 2(q'-3)+1 forced slots - KERNEL (W7, W8) +
  PROOF + MEASURED (23 wheels; brute force over all W^2 affine maps at seven wheels) -
  PARTIAL OVERLAP, checked 2026-09-06: the one-class cycle palindrome is Holt-Rudd arXiv:1408.6002
  Remark 2.2(v) ("except for the final 2, the cycle of differences is symmetric") and the shield
  is Ziller arXiv:2007.01808's p# +- 1 argument.  Delta: two classes, mirror n -> -n-2 with fixed
  point -1 rather than the r/-r pairing, arbitrary gear set rather than an initial segment.
  PUBLISH AS A LEMMA, NEVER AS A DISCOVERY.  NOVEL* are the full 2^{#odd} stabiliser, the
  adjacency subgroup Z/2, and W67.  **W9 of document 1 ("every gap length has an even count
  except length 1") is REFUTED** - false as soon as N_1 = 0; W67 is the true statement and needs
  no hypothesis
- wheels-core-tail - W16 the record as an exact cover by the gears' letters with NO FLANKS, W31
  the sub-threshold reduction (F_top is a function of m and of the gears below F_top + 1 alone),
  W62 the core/tail rule as a formula (F_top = max{L : min over core phases of the domino cost
  D(U) <= t}), W61 in use the tail is EMPTY, W63 the additive form holds exactly when the core is
  empty, W18 universal record multiplicity 18/24/480/720, W33 the record blocks pinned mod the
  small gears - PROOF (W16 exact by CRT, validated against the full-period scan 15 of 15; W62
  proof sketch being closed in top_machine_7.md, 0 mismatches on 13 known records and 89 sets
  decided) + MEASURED - PARTIAL OVERLAP / KNOWN VARIANT, checked 2026-09-06: the covering
  equivalence is Ziller arXiv:2007.01808 Definition 2.4 ("restricted covering") and Proposition
  1.8 for ONE class per prime with a flank condition, and the same reduction is the engine of
  Erdos-Rankin and FGKMT arXiv:1412.5029; the core/tail principle ("primes above the window enter
  only as a count") is the standard engine of every Jacobsthal computation - Hagedorn, Math. Comp.
  78 (2009), Costello-Watts arXiv:1208.5342, Ziller-Morack arXiv:1611.03310.  DO NOT CLAIM THE
  PRINCIPLE.  Delta, and it is real: the two-class version where a large gear removes a DOMINO
  and not a point, the explicit domino cost D(U), the max-min certificate, and the exactness with
  no flank condition.  W18, W61, W63, W33 NOVEL*.  **THE HIGHEST-LEVERAGE ITEM IN THE WHEELS**:
  W31/W62 are the only laws that apply with small gears present, i.e. in the actual window regime
- wheels-xor-parity-bit - W52: the longest run of ones of the striker-parity bit (the XOR of the
  gears' masks, a Liouville-type bit restricted to G) equals F_top EXACTLY, because a record block
  struck exactly once at every cell always exists - so the parity bit, which HAS a product
  formula, bounds the record, which does not, and the bound is tight - MEASURED, 10 of 10 wheels,
  both parities of m - NOVEL* (checked 2026-09-06: nothing in the Liouville / parity-problem
  literature is a finite-period run statement, and no twin-sieve paper defines such a bit.  The
  COUNT half - #even - #odd = prod(g-4), W51 - is KNOWN, being Schemmel S_4 plus a one-line
  character sum.  TWO CAUTIONS: W52 is not independent of W29 and they stand or fall together;
  and this pushes on the parity problem from the inside (Selberg 1949; Tao, "Open question: the
  parity problem in sieve theory", 2007), so expect it to bound the SIEVE record, not the primes)
- wheels-anchor-rescaling - W72: adding a gear that leaves exactly one slot rescales the record
  exactly - F(G+{2}) = 2F_2(G)+1, F(G+{3}) = 3F_3(G)+2, and F(G+{2,3}) = 6 F_col(G) + 5, where
  F_col is the record of the BOTTOM machine in the column coordinate; with W19 the conjugacy
  (n -> 6^{-1}(n+1) carries one machine's open set exactly onto the other's) and W69 the corrected
  ceilings - PROOF (mechanism) + MEASURED, 30 of 30 exact, records from 5 to 107 - PARTIAL
  OVERLAP, checked 2026-09-06: the one-class analogue j(2n) = 2j(n) (and j(p^k n) = j(pn)) is the
  basic reduction of every Jacobsthal computation (Ziller-Morack arXiv:1611.03310, Hagedorn 2009)
  and the composed 6F + 5 is the record-level form of the 6k +- 1 folklore; Ziller-Morack could
  not have reached p = 73 without collapsing 2 and 3 somehow, so read it as USED BUT UNSTATED.
  Delta: the exact identity with the sub-lattice coordinate, an equality not an inequality.
  **ACTION DISCHARGED AND CORRECTED 2026-09-06** (harvester r1): the earlier action - "rescale F
  by 6F + 5 and compare against arXiv:1706.03668 Table 1 / OEIS A288815" - is the WRONG
  comparison, because A288815 is the free-class record and F is the D = 2 record (see the header
  correction above); run as written it would have reported a machine-wide failure (A288815 = 18,
  66, 150, 192, 258, 366 at n = 3, 5..9 against 6F = 12, 42, 66, 108, 150, 204).  The validation
  that DOES exist, and now run: 6 F(M) against the REAL twin-candidate max cyclic gap at p_n#
  (direct sieve of {k : gcd(k(k+2), p_n#) = 1}), **7 of 7 exact** at n = 3..9 - 12, 30, 42, 66,
  108, 150, 204 - up to p_9# = 223,092,870
- wheels-smooth-zone - W55 the gear-zone identity (for n <= Q - 2 the pair n is open iff n and
  n+2 are both q-smooth), W56 the in-use record is the largest gap of that FINITE list (value AND
  position), W57 the proved lower bound F_range >= sqrt(N) - s(q) - 2 with no sieve estimate, W58
  no bound of the parity kind, W64 no saturation - KERNEL (W55 = `smooth_zone`) + PROOF +
  MEASURED (36 machines, 0 exceptions; ratio truth/bound exactly 1.000 at 17 of 36) - **PRIOR ART
  FOUND, and one entry DEMOTED**, checked 2026-09-06: W55 is KNOWN - the object is the
  Stormer-Lehmer difference-2 smooth-pair problem (Stormer 1897; **D. H. Lehmer, "On a problem of
  Stormer", Illinois J. Math. 8 (1964) 57-69**, which reduces to ~2^k - 1 Pell equations and
  treats difference 2 explicitly; OEIS A002071/A002072; still recomputed for isogeny cryptography,
  arXiv:2211.04315, arXiv:2509.17699).  **W64 IS DEMOTED FROM A PHENOMENON TO A COROLLARY: the
  measured constant in F_range = Q - 160 - u at q = 5 IS s(5) = 160** (160 = 2^5*5, 162 = 2*3^4,
  the last {2,3,5}-smooth pair), so "no saturation" is W57's bound running at equality with a
  DIOPHANTINE constant, predictable in advance for every q from Lehmer 1964 and Luca-Najman,
  Math. Comp. 80 (2011).  NOVEL* is what remains: no study of THE GAP STRUCTURE OF THE STORMER
  LIST itself was located, and W56/W57 are that.  Delta against FGKMT-style results (FKMPT
  arXiv:1802.07604, whose Remark 7 already names the I_p = {0,2} system): theirs is a full-period
  theorem driven by density, ours is confined to [1, Q] where the answer is Diophantine and not
  probabilistic - so ANY THEORY THAT PREDICTS THE IN-USE RECORD FROM DENSITIES IS MEASURING THE
  WRONG THING
- wheels-quiet-zone-families - W79 the family decomposition (the zone's open pairs are the
  DISJOINT UNION, over q-smooth pairs (s, s') with gcd(s,s') | 2, of the linear twin-prime
  problems s'P' - sP = 2 in primes above Q - at most Psi(Q,q)^2 families, a polylog in Q at fixed
  q), W80 the exact count and the q-INDEPENDENCE of each family (raising q adds families and
  changes none; family (1,1) contributes pi_2(Q^2) - pi_2(Q) at every q), W81 the walk in the zone
  as a minimum over smooth scalings of nextprime, W82 alignments always occur (a proved lower
  bound on the zone record from ordinary prime gaps alone), W83 the U-shaped record profile, W84
  the exact reason no upper bound is available - PROOF + MEASURED (1,510 families at q = 5,
  Q = 10^4, 0 forbidden gcds among 5.4 million pairs; 11.5 million walk positions, 0 mismatches;
  48 machines for W82, 0 exceptions) - NOVEL* for W79-W83, KNOWN for the ambient rule W76/W77
  (the Legendre-Meissel-Lehmer partial sieve phi(x,a) = pi(x) - a + 1 for x < p_{a+1}^2), checked
  2026-09-06.  **NAMING**: the ambient set is "integers free of prime divisors from an interval"
  (Weingartner, Acta Arith. 98 (2001) 117-131 and 104 (2002) 309-343; Tenenbaum, Ann. Sci. ENS
  (4) 19 (1986) 1-30) - do NOT call it "semismooth", which Bach-Peralta, Math. Comp. 65 (1996),
  use for the OPPOSITE inequality.  **CORRECTION TO W84 CARRIED OUT OF THE CHECK**: "the zone
  record is exactly as hard to bound above as the gaps of the family (1,1)" OVER-STATES it - an
  upper bound needs an open pair from ANY ONE of W79's families, i.e. a FINITE DISJUNCTION of
  binary prime problems, strictly weaker than the twin problem itself.  It is still parity-blocked
  (Selberg 1949; Tao 2007), so the reading survives and the sharp phrasing does not - and the
  disjunction IS THE PLACE TO PUSH.  Live neighbour: van Doorn and Tang, "Consecutive integers
  free of certain prime factors", arXiv:2606.19863 (June 2026), settling Erdos Problem #451 on the
  same ambient object with the complementary extremal question
- wheels-range-record-first-hit - W32: for a fixed gear set the range record is a first hit on the
  wheel's own census, F_range(N) = max{d : W/c(d) <= N} - 1, so **the wheel record IS reached**, at
  0.005% to 10.9% of the period; with W65, the two regimes and the crossover - MEASURED, within
  one unit at 19 of 21 checkpoints over three eight-gear machines, N = 10^4..10^10 - PARTIAL
  OVERLAP, checked 2026-09-06: the "set the expected count to 1 and solve" first-occurrence
  heuristic with a Gumbel law is Kourbatov, J. Integer Seq. 16 (2013) arXiv:1301.2242,
  Kourbatov-Wolf arXiv:1901.03785, Kourbatov arXiv:2002.02115 - FOR THE PRIMES.  Delta: asserted
  EXACTLY for a fixed periodic sifted set with a known deterministic census c(d).  HONEST: it
  cannot be a theorem in general (W/c(d) is a mean spacing and long gaps are not evenly spread),
  so THE EXACTNESS IS THE FINDING and it is the thing to try to break.  **This REFUTES document
  1's W2 and docs/proofs/22's "the wheel record is a hard ceiling approached slowly from below"**,
  which was an artefact of stopping at N = 10^7
- exhaust-stack - X5 non-containment (a gear never spans its own tier, and spans the tier below
  only by equalling its period - impossible - so ROOM APPEARS EXACTLY TWO FLOORS UP AND NOT
  BEFORE), X6 the exhaust cap (on (C, C^2] every strike of a gear above the cut is a HOME STRIKE
  or an ECHO, no primality used, so a pair the lower tiers leave open IS a twin prime), X1-X4
  stride containment, plus the cut recursion cut_k = tier k's period = tier k+2's lower edge -
  KERNEL-CHECKED, 48 declarations, zero sorries, no native_decide (proofs/MachineStack.lean) -
  **KNOWN**, checked 2026-09-06: X6 is the sieve of Eratosthenes to the square root (Legendre
  ~1808; Pomerance, "The Sieve of Eratosthenes and Rough Numbers"), and its stack framing is a
  KNOWN VARIANT of **Fred B. Holt, "Surviving Eratosthenes sieve I: quadratic density and
  Legendre's conjecture", arXiv:2603.25915 (March 2026)**, whose INTERVAL OF SURVIVAL
  [p_k^2, p_{k+1}^2] is this dichotomy as a stack-of-stages object - read it before writing this
  up.  Delta: tier indexing rather than consecutive primes, plus the explicit "nothing higher can
  help OR hurt" clause, located nowhere.  X1-X4 are STANDARD TOOL (division, periodicity;
  Holt-Rudd Lemma 2.2 is one-strike-per-copy; segmented-sieve folklore).  X5 and the cut recursion
  are NOVEL* AS STATED BUT MATHEMATICALLY TRIVIAL - do not present them as theorems.  **HANDLE X6
  WITH CARE**: the cap is exactly the shape of the parity obstruction, so "the search is capped to
  the motor plus one machine" restates WHY the classical route stalls; any claim that the cap
  HELPS must say what it does that a sieve weight does not.  One correction: q <= q# is trivial
  (q divides q#) - Bertrand is what is needed for TIER NON-EMPTINESS, and docs/proofs/23 should
  say so

## The record rule, the cover polynomial and the exhaust's first pass - prior art checked 2026-09-07 (harvester)

Register rows **W86-W102** (`top_machine_7.md` L67-L75 = W86-W94; `top_machine_8.md` = W95-W102)
and **X9-X24** (`exhaust_1.md`), all in research/proof/law_register.md with statement, status and
verdict.  Counts for the 33 new rows: 6 KNOWN, 7 KNOWN VARIANT, 14 NEW, 6 STANDARD TOOL; the
register's totals are now 20 / 25 / 52 / 28 / 1 REFUTED over 126 rows.  Vocabulary from here on
is the owner's: ENGINE (the primes up to q), MANIFOLD (the primes in (q, q#] on the raw line,
teeth 0 and -2 - what the entries above call the wheels / the top machine), VALVES (the engine
acting inside the manifold's open set), EXHAUST (every tier above).  Entries below name every
NEW and KNOWN VARIANT law; STANDARD TOOL and KNOWN rows are in the register only.  The
web-search budget was exhausted before this pass; it ran on the arXiv API, ar5iv full texts and
zbMATH reviews, and the register's section 3 lists exactly what was and was not read.

- manifold-loaded-record-rule - W88 the loaded record rule as an iff: [0, L) is coverable iff
  min over core phase vectors of D(U) <= t(L), D the domino cost (sum of ceil(run/2) over the
  step-2 runs of the uncovered set per parity class), core = gears <= L + 1, t = the tail count;
  hence F_top = max{L : min D_L(U) <= t(L)}; with W86 the piece law (a gear joins the two ends of
  the window iff g <= L + 1, and those are the only parity-crossing pieces) and W87 the matching
  lemma - KERNEL (`loaded_record_rule`, both directions; `ends_join_iff`; `domCost_run`) +
  MEASURED 6,659 gear sets, 5,006 loaded, 0 mismatches - **KNOWN VARIANT**, checked 2026-09-07:
  the necessity half is the rejection criterion of every one-class Jacobsthal computation -
  **Ziller-Morack, arXiv:1611.03310 (2016) Corollary 2.3** (positions still uncovered after the
  small primes against what the remaining primes can cover, one each above the interval length,
  Lemma 2.2), **Hagedorn, Math. Comp. 78 (2009) 1073-1087** (killing sieves, h(n+1) = 2w(n)+2),
  **Costello-Watts arXiv:1208.5342**.  Delta, and it is the whole two-class content: the tail
  piece is a domino, the exact price is Gallai's edge-cover identity on the parity runs (W87 is
  **KNOWN**: Gallai 1959, min edge cover = vertices - max matching), the rule is an iff by CRT
  assignment, and the boundary is g > L + 1 not g > L - moving one gear across it breaks the rule
  at 605 of 6,659 sets, a fact with no one-class analogue because a point has no parity.  W89
  the capacity bound and **W102 the parity-refined capacity bound** (per-parity-class counting;
  exact on every set with core density rho = sum_{g <= F+2} 2/g below 0.376, never above 0.7,
  vacuous at rho >= 1 - and the in-use manifold sits above 1) are KNOWN VARIANT of the same
  counting (Costello-Watts Thm 4.4's phi_min bound; Ziller-Morack Cor 2.3); Iwaniec 1978,
  Stevens 1977, Kanold 1967 bound j(n) uniformly and do not see a fixed gear set.  W102's regime
  statement is the useful part: capacity counting stops working BEFORE the window regime
- manifold-parity-law-derived - W90 the parity law F_top = 2m - (m mod 2) as a corollary of W88
  (empty core: D([0,L)) = 2 floor(L/4) + min(L mod 4, 2)), a second kernel proof of W17 using
  neither W16 nor W29 (`parity_law_of_rule`); W91 the sharp threshold q' >= 2m+1 (even m) /
  2m+3 (odd m) derived, the +2 at odd m being one parity bit (the single core gear's ends-joining
  piece leaves two runs of m-1, even, cost m-1 = t) - KERNEL + PROOF - **NOVEL*** as W17/W71
  (checked 2026-09-07: the one-class threshold q_1 > k is sufficient only, Erdos 1962; no parity
  defect exists in one class because the tail piece is a point)
- manifold-moment-vanishing - W92 M_k(d) = 0 for k < r(d) PROVED (the alternating sum over the
  Boolean cube is the top multilinear coefficient of f^k, which needs k pieces covering
  [1, d-1]); W93 r(d) = D(d-1), the census's covering number IS the record's cost function, with
  d = 4 the degenerate case r = infinity (N_4 = 0 identically, so W4 is no exception to W26);
  W95 the non-cancellation with the sign, M_{r(d)}(d) = (-1)^r r! C_r(d) (a minimum cover has one
  covering subfamily, so every minimum cover carries the sign (-1)^r) - PROOF, d = 2..26 exact
  - **NOVEL*** for the statements, STANDARD TOOL for the mechanism (Mobius inversion on the
  Boolean lattice, Rota 1964; the "sign uniformity of minimum covers" is that one line and is not
  a theorem), checked 2026-09-07: Brown arXiv:2311.06873 Lemma 3.1 has the one-class threshold
  universality but no moments and no vanishing order; Holt-Rudd have recursions and no
  polynomial.  Closes O-W1 and W25's hypothesis: the degree law is a theorem
- manifold-cover-polynomial - W97 the universal signature is a cover polynomial,
  sum_e c_e(d) z^e = sum_t C_t(d) (1-z)^t z^{d+3-t}, C_t(d) the number of covers of [1, d-1] by
  t pieces (dominoes {p-2, p} plus the end singletons); W98 the universal census as a cover sum
  N_d = sum_t C_t(d) (-1)^t Delta^t P_G(d+3-t), m - r + 1 terms, C_t by a two-state transfer
  matrix in O(d^2) - the universal census is polynomial in d, not 2^{d-1}; W96 the minimum-cover
  count in closed form C_r(d) = d/4 - 1, 1, (d+6)/4, ((d+1)/4)^2 by d mod 4, so the universal
  record multiplicity is r! C_r(d) (18, 96, 24, 24, 480, 6480, 1440, 720 at d = 6..13, 8 of 8);
  W99 the bijection (minimum cover, bijection gears -> pieces) -> n mod W onto the gaps of length
  d, so the multiplicity is a count of labelled tilings; W100 the second moment - PROOF, every
  coefficient d = 2..26 and 27 universal (d, G) cases, 0 mismatches - **NOVEL*** for the identity,
  the closed form, the bijection; **KNOWN VARIANT** for W98 and **KNOWN** for the polynomial
  itself, checked 2026-09-07: sum_t C_t x^t is the **edge cover polynomial** of a path with two
  pendant singletons (**Akbari-Oboudi, European J. Combin. 34 (2013) 297-321**), a Fibonacci-type
  transfer-matrix object; **Brown arXiv:2311.06873 Theorem 3.2 / eq. (15)** is the one-class
  census as a signed sum over interior subsets with prod(q - k), exponential in d, with no
  generating polynomial, no finite-difference form and no covering interpretation.  Do NOT cite
  the Chung-Graham cover polynomial (JCTB 65 (1995)) - it counts path-cycle covers of digraphs
  and is a different object.  Nothing located identifies a sieve gap-count signature with an
  edge cover polynomial
- manifold-half-turn - W101 the two parity classes of [0, L) are two windows of ONE
  adjacent-teeth core manifold: multiplying by 2^{-1} mod W_core conjugates teeth {0,-2} to
  {0,-1}, the even cells become [0, ceil(L/2)) and the odd cells [H, H + floor(L/2)) with
  H = (W_core + 1)/2, half a turn; F_top = max{L : min_x c(x) + c(x + H) <= t(L)}, a scan of
  W_core (median gain 29,939x on the family; nothing when the tail is empty, the in-use regime);
  measured with it: the windows do NOT minimise independently (3,611 of 5,006 loaded sets, all
  with four or more core gears), counts alone do not decide (1,875 sets), anchoring at the ends
  fails on exactly the 210 sets {7} u {four gears > 13} at F = 12 by one mechanism - PROOF (CRT)
  + MEASURED 19,896 phase vectors, 0 mismatches - **NOVEL*** for the coupled two-window form and
  the non-decoupling census; STANDARD TOOL for the map (the affine conjugacy of W19, and the
  normalisation of a paired progression <a,b>_m to <0,D> in Ziller-Morack arXiv:1706.00317; their
  paired algorithm sits in an ancillary file not read, so whether they halve is unknown).  What
  remains of the minimisation is a complexity question, class (b), not a structural one
- exhaust-first-pass - X12 `CutMono` is a theorem from q = 5 (cut_{k+1} > 4 cut_k; KERNEL
  `cutMono_of_five_le`, so `stack_open_iff_twin` is unconditional at q >= 5) - STANDARD TOOL
  (Bertrand iterated dyadically); X13 the exact cause of the degeneracy at q = 2, 3 (the primes
  below q must multiply to at least 4) - NOVEL* as a stated threshold and TRIVIAL, as X5; X15
  the redundancy lemma in range form (g^2 > N: only the home strike and echoes, no primality of g
  used) - KNOWN (Eratosthenes to the square root, least-prime-factor form); X19 the exhaust's
  share of the open pairs above Q^2 (36.9% .. 86.6% over five decades at q = 5) - MEASURED,
  KNOWN VARIANT (a two-class Buchstab count; the numbers are new to the object, the phenomenon
  is not); X20 a tier's record migrates into its smooth zone above a crossover that a raised
  split pushes up, F = max(F_smooth, F_quiet) at 9 of 9 - KNOWN VARIANT of W56/W64/W65
  (Stormer 1897 / Lehmer 1964: the 30-smooth pair list saturates at 423 pairs, last member
  354,365,440, so the smooth record is Q - 2 - s with slope 1); X22 the U-profile is NOT
  universal - it belongs to the split, not the zone (holds at q = 5 tier 3, fails at q = 7 tier
  3, mechanism family starvation while the smooth list is thin) - MEASURED, NOVEL*, and it
  CORRECTS W83; X23 W24 transfers with its exception exactly gear 7 - KNOWN VARIANT as W24
  (Brown eqs 16-17); X24 a tier's family (1,1) does not depend on the split (twin counts of
  (Q, 2Q] identical at q = 5 and 7) - NOVEL* as stated, but it is W80's q-independence read from
  above and NOT independent evidence; X21 the exhaust's own record is ROOT - KNOWN as an
  obstruction (the parity problem, as W84, with W84's correction: an upper bound needs one open
  pair from any one of W79's families, a finite disjunction).  Checked 2026-09-07.  Verdict of
  the pass: the exhaust's first pass adds no new mathematics and says so; O-X1, O-X2, O-X4 are
  closed and nothing in it is a lever
