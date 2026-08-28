# The gear-p cell decomposition of the gap histogram's Fourier transform

Lateral, round 25 (2026-08-29). Scripts: `research/mirror_cells.py` parts C-F
(log `research/data/mirror_cells.log`), `research/spiral29.py`
(logs `research/data/spiral_{11,13,17,19,23}.log`, `spiral29.log`, JSON
`data/spiral_<y>.json`). Companion: `docs/novel/mirror-parity-laws.md`.
Continues `docs/novel/pole-phase-law.md` (round 21, item 32).

## 1. WHAT IT IS

Round 21 found that the DFT of the machine's gap histogram at gear 5,
`H_5(1) = sum_g W_1(g) e(g/5)`, has argument `126 deg` to within a few degrees
at every machine, and modulus obeying an unexplained near-law
`|H_5(1)|/H_0 * meangap = 1.015 +- 1%`. This document reduces both to exact
integer statements, proves that the `126 deg` phase is **never attained
exactly**, and shows the `1.015` is a crossing scale rather than an invariant.

### (a) The cell decomposition

Fix a gear `p`. Openings avoid `p`'s two teeth, so they lie on the exposed set
`A_p` (`|A_p| = p-2`), and `p-2` consecutive exposed phases span exactly `p`
slots. Hence, with `zeta = e(1/p)`,

>   `zeta^{gap}` depends ONLY on the pair (start phase `i`, `n mod (p-2)`),
>   where `n` = the number of exposed `p`-phases the gap crosses.

So the `(p-2) x (p-2)` integer **cell matrix** `M[i][s]` carries the entire
frequency-`1/p` transform. CRT fixes the row sums (`N/(p-2)` each, exactly), and
the mirror involution `k -> -k` pairs cell `(i,s)` with `(-(phase_i + Delta), s)`.
Counting: `(p-2)^2` cells, `p-2` mirror-fixed, hence `(p-2)(p-1)/2` orbits and
**`(p-2)(p-3)/2` free integers** after the row sums. At `p = 5` that is THREE.

### (b) Gear 5, exactly

`A_5 = {0,2,3}`, `Delta[i][s] = [[0,2,3],[0,1,3],[0,2,4]]`. With
`T[r][v]` = #gaps starting at slot residue `r` with length `v (mod 5)`, mirror
forces exactly

>   `T[0][2] = T[3][2]`, `T[0][3] = T[2][3]`, `T[2][0] = T[3][0]`,

so writing `(e,b,c) = (T[2][0], T[0][2], T[0][3])` and `a = N/3 - b - c`,
the gap residue classes `N_r` (`r = 0..4`) are

>   `N_0 = a + 2e`, `N_1 = N/3 - e - c`, `N_2 = 2b`, `N_3 = 2c`,
>   `N_4 = N/3 - e - b`,

whence **`N_2` and `N_3` are always even** and

>   **THE MIRROR RELATION.  `2 (N_1 - N_4) = N_2 - N_3`, exactly, at every
>   machine.**

Substituting into `H = sum_r N_r omega^r` and using `1 + omega + omega^4 = phi`
gives the three-integer closed form

>   `Re H_5(1) = phi*N/3 + (3-phi)*e - ((3phi+1)/2)*(b+c)`
>   `Im H_5(1) = (2 sin36 + sin72) * (b - c) = (2 sin36 + sin72)*(N_2-N_3)/2`

- the whole transform is carried by three integers, and its imaginary part by
ONE.

### (c) The parity theorem: 126 deg is unattainable

`arg H_5(1) = 126 deg` (equivalently: the pole bracket `B = H(1-omega)/omega` is
real) holds iff two integer conditions hold,

>   (a) `N_0 + N_1 = 2 N_3`   and   (b) `N_0 + N_1 = N_2 + N_4`.

In cell variables (b) reads `2 (b + c - e) = N/3`. But `N = prod (q-2)` is a
product of odd numbers, so `N/3` is ODD and (b) is unsatisfiable:

>   **THEOREM.  `D := (N_0+N_1) - (N_2+N_4)` is ODD at every machine;
>   equivalently `(N_2+N_3) - 2 N_0 = 2 (mod 4)`.  The pole phase `126 deg`
>   is NEVER attained exactly, at any machine.**

Equivalently in bracket form: `Im B_5 = alpha_1 sin72 + alpha_2 sin36` with
`alpha_r = beta_r - beta_{-r}`, `beta_r = N_{r+1} - N_r`; the two sines are
`Q`-independent, so realness with integer `alpha` forces `alpha_1 = alpha_2 = 0`,
while `alpha_1 = -D` is odd. What the machine does instead is drive the RATIO to
the irrational golden direction:

>   `alpha_1/alpha_2 -> -sin36/sin72 = -1/phi = -0.618034`,

measured `-0.8636, -0.8393, -0.7305, -0.6403, -0.6448, -0.6231, -0.5943` at
m11..m31 - crossing `-1/phi` between m29 and m31, exactly where `arg B(5,1)`
crosses 0 (`+0.06` at m29, `-0.23` at m31). Scope: the parity floor forces the
deviation to be nonzero but only by `~1e-6` degrees; it kills the *pin* as an
exact statement, not the empirical closeness.

### (d) Gear 7 vs gear 5 (backlog U3, answered structurally)

The GF(2) satisfiability of the same system was tested for `p = 5..37`:

| p | cells | orbits | free | pole eqs | parity-obstructed |
|---|-------|--------|------|----------|-------------------|
| 5 | 9 | 6 | 3 | 2 | **YES** |
| 7 | 25 | 15 | 10 | 3 | no |
| 11..37 | ... | ... | ... | ... | no |

**Gear 5 is the only parity-obstructed gear.** And the structural asymmetry is
the codimension: at `p = 5` realness is ONE ratio of two integers approaching
one irrational direction; at `p = 7` THREE independent asymmetries must vanish
at once. Measured, gear 7's asymmetries are an order of magnitude larger and
decay far more slowly (`max|alpha|/N` = 0.259 -> 0.168 over m11..m37, versus
gear 5's 0.141 -> 0.019), which is precisely the observed picture: `arg B(5,1)`
converges to 0 while `arg B(7,1)` climbs `-2.4 -> +17.0`.

### (e) The amplitude near-law (backlog U2), reduced

Two exact reductions and one model ladder:

1. **Exact anchor.** The round-21 closure (depth-sum identity) gives
   `sum_{j=1..N-1} What_j(omega) = (2-phi) n_side^2 - N`, real positive, so the
   MEAN ARM over all proper depths is `((2-phi)n_side^2 - N)/(N-1) ->
   (2-phi)N/9 = 0.042440 N` - which is exactly the value `|What_1|` would take
   if consecutive openings decorrelated. Verified over ALL `N-1` depths, exactly
   and real, at m11 and m13. Therefore the near-law is the statement
   `|What_1| / mean arm = 23.92 / lam`, and `lam = 23.92` is the machine size at
   which depth 1 becomes a TYPICAL arm. **1.015 is a crossing scale, not an
   invariant.**
2. **Phase grading.** The phase-blind step model (`M[i][s]` independent of `i`)
   would force `N_2 = 2 N_1` and `N_3 = 2 N_4`; measured ratios are
   `N_2/2N_1` = 1.200, 1.126, 1.100, 1.081, 1.072, 1.065 and `N_3/2N_4` =
   1.833, 1.344, 1.239, 1.177, 1.146, 1.126 at m11..m29, and the blind model
   recovers `|H|` to 91.5%, 93.5%, 94.5%, 95.2%, 95.5%, 95.7%. So the amplitude
   is ~95% a statement about the exposed-step count `n mod 3` and ~5% about
   which phase the gap starts from - and the phase-graded part is SHRINKING.
3. **Corridor-renewal ladder** (new): model the openings as an independent
   thinning, at the rate fixed by the true mean gap, of the slots exposed mod
   `m`, and compute `E[omega^gap]` exactly by first passage on the `m`-cycle.
   Gate: at `m = P` the model reproduces the machine to `1e-9` (asserted).
   Result:

   | y | lam | measured | m=5 | m=35 | m=385 | m=5005 | m=85085 |
   |---|-----|----------|-----|------|-------|--------|---------|
   | 11 | 2.852 | 1.1260 | 1.0916 | 1.2194 | 1.1260 | - | - |
   | 13 | 3.370 | 1.0362 | 1.0199 | 1.1874 | 1.1050 | 1.0362 | - |
   | 17 | 3.820 | 1.0150 | 0.9707 | 1.1709 | 1.1032 | 1.0380 | 1.0150 |
   | 19 | 4.269 | 1.0139 | 0.9292 | 1.1594 | 1.1066 | 1.0441 | 1.0259 |
   | 23 | 4.676 | 1.0193 | 0.8965 | 1.1512 | 1.1111 | 1.0503 | 1.0354 |
   | 29 | 5.022 | 1.0161 | 0.8713 | 1.1449 | 1.1149 | 1.0553 | 1.0428 |

   **NO fixed corridor depth reproduces the flat 1.015.** The `m=5` column
   decays (1.09 -> 0.87); every deeper column rises. The measured flatness is
   the cancellation of those two drifts as the machine's own corridor depth
   grows with it. Pushed past the data, every fixed-`m` column has a MINIMUM
   (near `lam ~ 16-24`) and then grows without bound toward `(2-phi)lam/9`.

## 2. WHY IT MIGHT BE NOVEL

- The cell decomposition is a genuinely new coordinate on a classical object:
  it says the frequency-`1/p` Fourier coefficient of a primorial sieve's gap
  histogram is a function of `(p-2)(p-3)/2` integers, independent of the machine
  size - three integers at `p = 5`, for every machine, for ever. That is a
  dimension-reduction statement about `G(P#)` that this project has not seen in
  the literature it has read.
- The exact mirror relation `2(N_1-N_4) = N_2 - N_3` on gap residue classes is a
  linear identity among census numbers with no obvious classical shadow.
- The parity theorem is a *negative* of an unusual kind: it rules out an exact
  Fourier phase by a `mod 4` argument on residue-class counts, and it singles
  out gear 5 among all gears.
- The reframing of the round-21 `126 deg` "invariant" as an integer ratio
  converging on `-1/phi` connects it to the lane's golden spectral gap
  (`hat_5(2) = phi` exactly, `docs/novel/golden-spectral-gap.md`) - the same
  irrational appears as the target direction of a discrete phase.

Honest shadow: reduced-residue Fourier coefficients are classical (Ramanujan
sums), and the per-gear factorisation `hat_q(j) = -2 cos(2 pi j u/q)` was
already this lane's round-20 item 29. The new content is the *gap-histogram*
transform (a second-order object, not the sieve's own transform) and its
integer parametrisation.

## 3. PROOF

PROVED (elementary, complete as written) + SCRIPT-VERIFIED (exact integers).

`research/mirror_cells.py`, 9 assertion gates, exit 0:
- part C: mirror equalities, CRT row sums, the derived forms of `N_0..N_4`, and
  `2(N_1-N_4) = N_2-N_3`, asserted at m11/13/17/19 from full-period scans AND at
  m11/13/17/19/23/29/31 from the census histograms. The partial-coverage m37 row
  is carried as a CONTROL and fails, as a period-wide law must.
- part D: the three-integer closed forms asserted against direct evaluation; `D`
  odd and `(N_2+N_3)-2N_0 = 2 (mod 4)` asserted at every full-period machine.
- part E: an independent GF(2) linear-algebra test over mirror orbits reproduces
  the `p = 5` obstruction and finds no other for `p <= 37`.
- part F: closure verified over all `N-1` depths at m11/m13; the
  corridor-renewal ladder gated by `m = P` exactness.
- `research/spiral29.py` independently confirms the cell equalities and
  `(N_2+N_3)-2N_0 = 2 (mod 4)` at m11..m29 (defects 38, 282, 2998, 37306,
  634182, 13462586 - all `= 2 mod 4`).

The `(2-phi)/9` decorrelation floor is a HEURISTIC limit (asymptotic uniformity
of the next opening's residue), not a theorem; the *mean-arm* value equal to it
IS exact. Model columns are floats and labelled as models.

## 4. IMPLICATIONS

- Round 21's item 32 is upgraded and bounded: the pole-phase LAW stands, the PIN
  was already refuted by drift (Refuted 17), and it is now refuted a second,
  independent way - by parity, as an exact impossibility.
- Backlog U2 is closed as posed: the `1.015` amplitude is not an invariant, it is
  the scale `lam = 23.92` at which the depth-1 arm meets the mean arm, and the
  flatness over m17..m29 is a two-drift cancellation. No further model is needed.
- Backlog U3 is answered structurally and measured: gear 5's bracket is one
  integer ratio chasing `-1/phi`; gear 7's is a three-dimensional alignment.
- The census-file defect found by the parity law (missing wrap gap in every
  full-period `ghist` row) is a process finding for the whole team.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

- Jacobsthal-type gap statistics for primorial moduli: the residue-class counts
  `N_r` of the gap histogram mod a gear are constrained by an exact linear
  relation and a parity congruence, which any model of `G(p#)` must respect.
- The lane's own arity/threshold programme: the free-integer count
  `(p-2)(p-3)/2` is another "arity" that grows with the gear.
- Open: WHY the ratio `alpha_1/alpha_2` converges on `-1/phi` at all (the
  convergence itself is measured, not derived); and whether it overshoots
  permanently (it has crossed at m31 and continues to `-0.578` on the partial
  m37 data).

## 6. PRIOR-ART CHECK

**Not yet checked** (no web access in this lane). Suggested terms: "Fourier
coefficients of the gap distribution of reduced residues mod primorial";
"Ramanujan sum gap histogram primorial sieve"; "distribution of prime gaps
modulo small primes residue classes parity"; Holt arXiv:2502.20470 (cycle of
gaps `G(p#)`) - his framework is the closest and would be the place a relation
like `2(N_1-N_4) = N_2-N_3` could already sit; Hagedorn; Ziller & Morack;
Sun's work on gaps mod small primes. Standing lesson: prior-art checks expire.
