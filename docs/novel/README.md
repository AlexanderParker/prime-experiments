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
  the novelty is not; see docs/novel/paired-hlb-cycles.md section 0
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
- layered-erdos-rankin - ROUND 25, NEW: the Erdos-Rankin construction run k
  times, one layer per available residue class, giving the k-class Jacobsthal
  function j_k(P(x)) >> x A^(2k-1) C^k/((5k)^k B^(2k)) whose k=1 case IS the
  published FGKT length and whose k=2 case is h_2(P(z)) >> z (log z)^3
  (lll z)^2/(ll z)^4 - TWO logs above round 24's (P1) and ONE log above what
  round 24's open problem (P2) asked for. Mechanism: class 0 on a SPLIT range
  buys a full log where its Mertens entitlement is O(1), and the paired
  problem's second class buys it again on n+2, so the joint survivor set is the
  TWIN primes; only an UPPER bound on twins is needed, so it is parity-free -
  STATUS: ASYMPTOTIC BOOKKEEPING, script-verified and calibrated at k=1 against
  FGKT (residual spread 0.072 over eight decades of log x), NOT a written-out
  proof; no finite-z content (does not exist below log z ~ 300) -
  research/j2_rankin_layer.py - NOVEL* (FKMPT "Long gaps in sieved sets"
  arXiv:1802.07604 is the nearest: GIVEN classes, adversarial, x(log x)^tiny -
  neither contains the other; j_k appears nowhere; checked first-hand 2026-08-29)
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
  SCRIPT-VERIFIED (research/nontensor_spec.py) - not yet checked (round 22)
- corridor-eigenvalue-closed-form - Constructor's measured corridor resonance derived:
  the corridor-phase chain's whole spectrum is the image of the e-th roots of unity
  (e = |E| = prod (q-2) over the small gears, 15 for mod 35) under the Moebius map
  mu(w) = rho w/(1 - (1-rho)w), rho = prod_{q not | m}(1 - 2/q); all eigenvalues lie on
  the circle |z - (1-rho)/(2-rho)| = 1/(2-rho) through 1. Matches every measured
  lambda_2 (m11-23, mod 35 and 385) to 1-2% in modulus and < 1.8 deg in argument;
  the residual IS the anti-correlation and is pre-registered for m29 -
  PROVED(model) + SCRIPT-VERIFIED(machine) - not yet checked (round 22)

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
  closed-form spectra - not yet checked (round 21)
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
  machinefree_cert.py) - not yet checked (rounds 22-23)
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
  budget-1 and span-capped at 200, stated) - not yet checked (rounds 23, 25)
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
  checked (round 24)
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
  - not yet checked (round 25)
- gear-cell-decomposition - the frequency-1/p Fourier coefficient of the gap
  histogram is a function of only (p-2)(p-3)/2 integers, for EVERY machine (three at
  p = 5), via the cell matrix M[i][s] indexed by (start exposed phase, exposed-step
  count mod p-2); mirror + CRT give the exact relation 2(N_1-N_4) = N_2-N_3 on gap
  residue classes mod 5 and the three-integer closed forms for Re/Im H_5(1).
  THEOREM: (N_2+N_3) - 2N_0 = 2 (mod 4) at every machine, so round 21's pole phase
  126 deg is NEVER attained exactly - the machine instead drives an integer ratio
  toward -1/phi (crossing it between m29 and m31).  Gear 5 is the ONLY
  parity-obstructed gear for p <= 37 (GF(2) test) - that is backlog U3's asymmetry.
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
  machines - not yet checked (round 25)
