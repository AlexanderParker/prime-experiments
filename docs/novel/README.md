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
- twin-percentile - twins at 13.3rd percentile of own family - COMPUTED - NOVEL*
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
  empty; ZM prove no bound, no 2018-2026 follow-up), checked 2026-08-24
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
  1.6M-slot scan); (D) also proved at 7->11, missed by 1 at 11->13 - SCRIPT-VERIFIED
  (research/lp_dual_certs.py, research/exact_lp.py; origin research/matrix_shapes.py) -
  PARTIAL OVERLAP (closed-form counting corollary is a weaker case of Costello-Watts
  arXiv:1208.5342; the dual-certificate form, the visibility reduction and the (D)
  application NOVEL*)
- moment-degree-ceiling - every fixed-degree covering certificate for F(M) goes VACUOUS at
  a computable machine, family-free: the uniform product measure's degree-<=l moments
  extend to a distribution with no empty atom, so every degree-l cut is satisfied at every
  width.  Sharp ceilings degree 1 -> machine 13, degree 2 -> 29 (so Kounias was already
  degree-2-optimal), degree 3 -> >= 151; required degree ~ 2*S1(y) ~ 4 log log y,
  UNBOUNDED, so no fixed-arity covering certificate exists - the LP-side answer to the
  round-22 arity question.  Chain-cut slope has the telescoping closed form
  S1*prod(1-2/q) + beta and the chain family is exponentially weaker than the sharp
  bound - SCRIPT-VERIFIED exact (research/lp_dual_certs.py C,D) - PARTIAL OVERLAP
  (Boole-Bonferroni LP of Prekopa/Boros is the machinery, Brun truncation growth is the
  classical shadow; the per-degree exact ceiling machine and the degree law NOVEL*)
- kleene-generator - F(M+q') = L^T (x) K* (x) R exactly, where K is the max-plus matrix of
  qualifying-and-T3-alternating successor steps and K* its Kleene star: the merge law's
  increment IS a longest path, so (D) becomes ONE arity-free dual certificate
  (h >= R, h >= K (x) h, L + h <= F + q') with no depth quantifier; the m-th layer of
  the star is qualmax_{m+2}.  Measured: the value-only abstraction is CYCLIC (bound
  vacuous) exactly where the infinite alternating word survives the 2-point relaxation,
  and adding the corridor phase mod 35 restores nilpotency and certifies (D) -
  SCRIPT-VERIFIED (exact, full period, steps 11->13 .. 23->29;
  research/kleene_generator.py, kleene_stream.py) - not yet checked (round 22)
