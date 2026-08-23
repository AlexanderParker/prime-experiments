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
- literal-cap - literal chains <= 6 members forever, function of q' mod 210 - KERNEL-CHECKED core
  + SCRIPT-VERIFIED spectrum - NOVEL*
- corridor-law - 12 of 24 gcd classes forbidden, dichotomy - KERNEL-CHECKED - NOVEL*
  (classification; method standard CRT)
- polignac-cap - capOf_le_twelve, all 8 gcd classes, empty axiom footprint - KERNEL-CHECKED -
  NOVEL* (cap; |E_e| product is the known HL local factor)
- suppression-law - joint qualifying-gap deficits x26/x6.7/x1400 + rate law - MEASURED -
  PARTIAL OVERLAP on phenomenon (Maier, Ford-Maynard-Tao chains); law + shape NOVEL*
- tooth-sharing-pinning - twin gear pair pins 4 CRT kill classes closed-form - PROVED -
  PARTIAL OVERLAP (CRT core classical, Clement 1949 modulus); slot-frame identity NOVEL*
  but elementary
- paired-jacobsthal-values - exact h_2 values - COMPUTED - PARTIAL OVERLAP: Ziller-Morack
  arXiv:1706.03668 Table 1 already has 18,30,66,150,192 (project premise "ZM compute none"
  FALSE - exact independent replication). Per-difference family F_d, fixed-twin ladder
  F(2,37..53), dip analysis NOVEL*. ZM h_2(19)=258 settles the project's open y=19 question.
- twin-percentile - twins at 13.3rd percentile of own family - COMPUTED - NOVEL*


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
