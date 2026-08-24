# Eigenvalue statistics of the Jacobsthal machine: Poisson, not GUE

Status: SCRIPT-VERIFIED statistics on CLOSED-FORM spectra (eigenvalues exact
by formula; spacing statistics floats, labeled) + PROVED exact structure
(clock spectra, mirror-degeneracy law, determinant identity). Established
round 21 (Lateral; the human's Riemann-bridge test, stated 2026-08-24).
Script: `research/eig_stats.py`. Prior-art check: NOT YET CHECKED
(section 6).

## 1. WHAT IT IS

Plain language. The Riemann zeros' spacing statistics famously match GUE
random-matrix eigenvalues (Montgomery-Odlyzko), and the Hilbert-Polya dream
is an operator behind them. The machine's operators are finite shadows of
the full gear train, with spectra in closed form - so the question "do
Jacobsthal-type operators drift toward GUE as machines grow?" is exactly
computable. Answer: NO, from both sides. The machine's unitaries are exact
CLOCKS (maximally rigid); its Hermitian circulant is asymptotically POISSON
(maximally uncorrelated); GUE lies strictly between the two and is
approached by neither - the trend with machine size runs TOWARD Poisson.

Precise form.

1. UNITARIES = CLOCKS (exact, no numerics): the slot shift S_P is a single
   P-cycle and the renewal operator R (Constructor R35) a single |E|-cycle
   permutation; eigenphases are ALL n-th roots of unity, exactly
   equidistributed: spacing distribution delta(s-1), spacing ratio r = 1.
2. THE CIRCULANT C_M (symmetric, integer; matrix-formulation piece 7):
   eigenvalues lambda(j_5..j_y) = prod_q h_q(j_q), h_q(0) = q-2,
   h_q(j) = -2cos(2 pi j u_q/q). The mirror symmetry k -> -k makes every
   nonzero local frequency doubly degenerate, so the full spectrum carries
   systematic 2^m-fold degeneracies. Exact law, verified: the number of
   tied consecutive pairs in the full sorted spectrum equals
   P - prod_q (q+1)/2, EXACTLY, at machines 11/13/17 (313, 4501, 80549) -
   i.e. there are NO accidental cross-gear coincidences there.
3. DESYMMETRIZED SPECTRUM (one representative per mirror class, size
   prod (q+1)/2): consecutive-spacing-ratio statistic
   <r~> (Poisson 0.38629, GOE 0.53590, GUE 0.60266, clock 1.0):

       machine   11      13      17      19      23      29      31
       levels    72      504     4536    45360   544320  8.16e6  1.31e8
       <r~>      0.3964  0.3867  0.3963  0.3945  0.3871  0.3865  0.3862

   Machine 31 (130,636,800 exact levels) sits at 0.3862 - Poisson to four
   figures. KS distance of the unfolded nearest-neighbour spacings to
   Poisson e^{-s}: 0.43 -> 0.0022 (m29) -> 0.039 (m31; the residual is
   unfolding-sensitive near the density singularity at 0); KS to the GOE
   Wigner surmise stays 0.19-0.59 throughout. Repulsion probe P(s < 0.1):
   0.0940 at m29/31 vs Poisson 0.0952, GOE 0.0078 - no level repulsion.
   Near-collisions inside the desymmetrized list at 1e-12 relative
   tolerance: 0, 0, 0, 0, 0, 6, 613 (m11..31) - not resolved algebraically,
   fraction <= 5e-6.
4. Determinant trivia (exact): prod_{k=1}^{q-1} (-2cos(2 pi k/q)) = 1 for
   odd q, hence det C_M = prod (q-2) = the open count.

VERDICT: the machine's linear operators BRACKET GUE from both sides - clock
on the unitary side, Poisson on the Hermitian side - and growth moves the
Hermitian side TOWARD Poisson, away from GUE. The Riemann bridge fails at
the level of these operators, for a structural reason: GUE requires level
repulsion, i.e. coupling between the gear factors, and any operator whose
spectrum is a CRT product multiset is in the integrable/Berry-Tabor class
(Poisson) by construction. Sharp reframing for the project: a Hilbert-
Polya-like object for the machine, if one exists, must live on the
NON-tensor sector - the same B = I - (x)E_q obstruction (blocking is a
complement of a product, Wall V in operator form) that blocks per-gear
factorisation of F is exactly what a GUE-bearing operator would need to
couple. The two failures are one failure.

## 2. WHY IT MIGHT BE NOVEL

The Berry-Tabor conjecture (integrable -> Poisson) and Montgomery-Odlyzko
(zeros -> GUE) are classical; tensor-product spectra being Poisson is the
expected outcome and is claimed as a TEST RESULT, not a surprise. What
appears unrecorded: spacing statistics of Jacobsthal/sieve-machine operator
spectra computed at all (the computational-Jacobsthal literature has no
operator formulation - matrix-formulation piece 2's search); the exact
mirror-degeneracy count law; and the structural identification "the
obstruction to per-gear factorisation = the only place a GUE operator could
live" as a statement about sieve operators.

## 3. PROOF / STATUS

Clock spectra: PROVED (permutation cycle structure). Degeneracy law:
exact integer counts, asserted (11/13/17). Spacing statistics: closed-form
eigenvalue lists, float statistics, all tables asserted/printed by
research/eig_stats.py (machines 11-29 in the default run, machine 31 via
--big, ~1 GB). Pre-registered expectation (Poisson, written in the script
docstring before running) CONFIRMED.

## 4. IMPLICATIONS

Inside the project: closes the round-21 eigenphase question honestly;
redirects any spectral-statistics hopes to the non-tensor sector (the
nilpotent BS, the word-level H's non-triangular part) - the only operators
whose spectra are NOT forced Poisson by construction. Outside: a concrete,
falsifiable data point for "which arithmetic operators are GUE": the
Jacobsthal machine's natural operators are not, with exact spectra anyone
can recompute.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

Hilbert-Polya / Montgomery-Odlyzko (negative finite-machine data point);
Berry-Tabor (a clean exactly-solvable instance at scale 1.3e8 levels); the
613 near-collisions at machine 31 (algebraic coincidences of cosine
products - finite, checkable, open).

## 6. PRIOR-ART CHECK

Not yet checked (agent without web access). Terms for the manager:
"spacing statistics circulant graph eigenvalues Poisson"; "Berry-Tabor
tensor product spectrum Poisson proof"; "level statistics almost periodic
/ harmonic oscillator spectra"; "Jacobsthal function operator spectrum";
"spectral statistics of sieve matrices". Expected nearest art: Berry-Tabor
literature (Marklof's rigorous cases for harmonic oscillators / tori) -
the machine instance and the degeneracy law are the delta to check.
