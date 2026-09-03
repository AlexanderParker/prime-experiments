# The walk's Fourier transform: a pole factor times the gap-weighted opening sum

Lateral, round 29 (2026-09-03).  Status per statement in section 3.  Script:
`research/walk_fourier_r29.py` (55 assertion gates, exit 0, log
`research/data/r29/walk_fourier.log`); results archived in
`research/lateral_r29_results.txt` block C.

## 1. WHAT IT IS

**Plain language.**  The machine's basic question is "how far is it from here to
the next surviving slot?".  Call that distance `W(s)`.  Writing `W` in
frequencies (a Fourier transform) turns out to be almost free: the transform of
`W` is a simple pole factor times the transform of ONE other object - the
opening set weighted by the gap that starts at each opening.  Everything that is
hard about the machine sits in that one object, and nothing else.  The pole
factor is exactly the one this project already met in its round-21 "pole phase"
law, which is therefore not a curiosity about gap histograms: it is the Fourier
transform of the walk itself.

**Precise form.**  Machine `M_y`: gears the primes `5..y`, gear `q` blocks slot
`k` iff `6k = -+1 (mod q)`, i.e. `k = +-v_q` with `v_q = 6^{-1} mod q`.
`P = prod q`, `N = prod (q-2)` openings per period, `chi` the open indicator,
`B = 1 - chi`, `W(s) = min{d >= 1 : chi(s+d) = 1}`, `g(o)` the gap starting at
the opening `o`, `lambda = P/N` the mean gap.  Write `e(x) = exp(2 pi i x)` and
`Xhat(m) = sum_s X(s) e(-ms/P)`.

> **IDENTITY 1 (the pole identity).**  For every `m != 0`,
>
>     What(m) * (1 - e(m/P))  =  - e(m/P) * Ghat(m),
>     Ghat(m) := sum over openings o of g(o) e(-mo/P).
>
> At `m = 0`: `Ghat(0) = P` and `What(0) = sum_g W_1(g) g(g+1)/2`.

*Proof.*  `W` satisfies the exact recursion `W(s) = 1 + B(s+1) W(s+1)` (if
`s+1` is open then `W(s) = 1`; otherwise `W(s) = 1 + W(s+1)`).  Transform, use
`sum_s e(-ms/P) = 0` for `m != 0`, and substitute `t = s+1`:
`What(m) = e(m/P) [ What(m) - (chi W)hat(m) ]`, and `chi(t) W(t) = g(t)` on
openings and `0` elsewhere, so `(chi W)hat = Ghat`.  []

> **IDENTITY 2 (the split, and where the closed form stops).**
>
>     Ghat  =  lambda * Shat  +  Dhat,
>     Shat(m) = sum over openings o of e(-mo/P)  =  prod_q hat_q(m c_q),
>       hat_q(0) = q - 2,   hat_q(j) = -2 cos(2 pi j v_q / q),
>       c_q = (P/q)^{-1} mod q      (the CRT frequency each gear sees),
>
> so the first term is CLOSED FORM - `pi(y)` multiplications per frequency, no
> scan, no period - and `Dhat`, the gap-FLUCTUATION transform, is not.  By
> Parseval the energy shares are exactly `lambda^2 : Var(g)`, i.e.
>
>     closed-form share  =  lambda^2 / (lambda^2 + Var(g))
>                        =  0.7683, 0.7385, 0.7117, 0.6902  at m11, m13, m17, m19,
>
> decreasing in the machine.

The `c_q` is not cosmetic: the additive character mod `P` factors over the gears
as `m s / P = sum_q m c_q s_q / q`, so gear `q` sees frequency `m c_q`, not `m`.
The project's round-20 statement of the machine DFT (`golden-spectral-gap.md`,
lateral item 29a) omits it; the corrected form is gated here on the first 4000
frequencies at four machines (agreement to `3e-12`).

> **IDENTITY 3 (L1 blindness) - a proved NEGATIVE.**
>
>     sum_m |Shat(m)| / P  =  prod_q S_q / q,
>     S_q = (q-2) + sum_{k=1}^{q-1} |2 cos(2 pi k / q)|,
>
> which is INDEPENDENT OF THE TOOTH VECTOR (`k -> k v_q` permutes the summands
> because `v_q` is invertible).  Hence the trivial (large-sieve shaped)
> character bound on the number of openings in a window is IDENTICAL at every
> member of the tooth-counterfactual family - while `F` varies over that family
> by a factor 1.83 / 2.50 / 2.29 at m11 / m13 / m17.  **No bound built from
> `|Shat|` alone can determine `F`.**  Per-gear factor `S_q/(q-2) -> 1 + 4/pi
> = 2.2732`; the measured mass ratios grow by x2.32, x2.31, x2.31 per gear.

## 2. WHY IT MIGHT BE NOVEL

* Identity 1 is elementary but it is a *reduction*: it says the distance-to-next-
  survivor function of a two-teeth-per-prime sieve has NO Fourier content of its
  own - the whole of it is the gap-weighted opening sum divided by a Dirichlet
  pole.  The project had been treating "the walk" and "the gap histogram" as two
  objects; they are one, related by an explicit divisor.
* Identity 3 is the useful half and is a *proved obstruction of the same species*
  as this lane's round-27 deflation of the 2n-gap reordering: an instrument that
  is constant on the counterfactual family cannot see `F`.  The literature's
  standard first move on a covering/sieve gap problem is exactly the L1 character
  bound; this says, with an exact constant, that it is blind by construction and
  not merely weak.
* The classical shadow: `Shat` is the singular-series-style exponential sum of a
  two-residue-per-prime sieve, and bounding gaps by exponential sums is
  Erdos-Rankin-adjacent standard practice.  What is not standard is (i) writing
  the FIRST-PASSAGE function itself in that language with an exact pole factor,
  and (ii) measuring an instrument's power against a null family that fixes the
  density and moves only the residue classes.

## 3. PROOF / STATUS

| statement | status | pointer |
|---|---|---|
| Identity 1 (pole identity) | **PROVED** (three lines from the recursion) | section 1; verified at ALL nonzero frequencies at m11/13/17/19, max relative error 6.95e-16 / 3.49e-16 / 2.93e-16 / 5.07e-16 (`walk_fourier_r29.py`) |
| `Shat = prod_q hat_q(m c_q)` | **PROVED** (CRT) + SCRIPT-VERIFIED to 3e-12 | same |
| Identity 2 (Parseval shares) | **PROVED** + SCRIPT-VERIFIED (exact agreement) | same |
| Identity 3 (L1 mass is tooth-independent) | **PROVED** (`k -> k v_q` is a permutation) + SCRIPT-VERIFIED at all 30 / 180 / 1440 tooth vectors | same |
| the term-count table | **EXACT COUNTS** | section 4 |
| the L1 and L2 vacuity ladders | **MEASURED, exact** | section 4 |
| "no character-sum form beats the scan" | **NOT PROVED** - none was found here, and none is claimed absent | section 5 |

## 4. IMPLICATIONS

**The floor named in `anchor-235.md` 9g is not lifted, and now it has an
address.**  Three exact forms of the same function, priced by term count:

    y    P           N         F   scan tests  flat/DFT terms  IE subsets
    11   385         135       7   48          385             2^8  = 2.56e2
    13   5005        1485      11  96          5005            2^12 = 4.10e3
    17   85085       22275     18  190         85085           2^19 = 5.24e5
    19   1616615     378675    25  312         1616615         2^26 = 6.71e7
    23   37182145    7952175   34  490         37182145        2^35 = 3.44e10

The scan (`2 pi(q) (F+1)` residue tests) wins at every machine.  The character
route gives either the flat form (`P` coefficients) or the inclusion-exclusion
form (`2^{F+1}` gear-factorised terms).  **It is the scan in disguise** - but the
disguise is informative, because Identity 1 says exactly which object carries the
irreducible content: `Ghat`, the gap-weighted opening transform, i.e. depth-1
adjacency selection.  That is the same place lateral item 27's depth-SUM identity
`sum_{j>=1} W_j(g) = prod_q c_q(g)` is closed form while its depth-1 term is not.

**Both bounds the frame supplies are vacuous, by exactly measured factors.**

    L1 (large sieve):  bound / main term at L = F
                       2.740, 5.254, 8.501, 15.37  at m11, m13, m17, m19
    L2 (Chebyshev on the number of empty windows), Var closed form from
    c(d) = prod_q c_q(d):
      L = F-1: bound 63.2 / 669.2 / 6948.9 / 89921.7 vs TRUE 4 / 12 / 20 / 20
               - vacuity 15.8x / 55.8x / 347.4x / 4496.1x, growing
      L = F:   bound 49.0 / 612.6 / 6373.2 / 81565.3 vs TRUE 0 - NO CERTIFICATE

**And the second moment sees the teeth without seeing `F`.**  Over the
counterfactual family, `L2cert` = the smallest `L` whose Chebyshev bound drops
below 1:

    m11  F in [6,11]   L2cert in [35,131]     median L2cert/F   7.7x
    m13  F in [10,25]  L2cert in [280,637]    median          29.4x
    m17  F in [14,32]  L2cert in [2119,4989]  median         161.3x
    spearman(F, L2cert) = -0.038, +0.023, -0.186

So the L2 instrument is not blind (its spread is 2.3-3.7x) but its variation is
essentially uncorrelated with the quantity it is supposed to bound, and at the
largest machine the correlation is NEGATIVE.  This is the sharpest available
statement of why moment methods have never bitten in this project: it is not
that they are weak, it is that what they vary with is not `F`.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

* The `anchor-235.md` 9g floor: "compute the first integer outside a union of
  `2 pi(q)` arithmetic progressions from the `pi(q)` residues of `s` alone".
  Identity 1 reduces any such form to a closed form for `Ghat` - so the floor is
  now the single question "is the gap-weighted opening transform computable
  below a scan?".
* Jacobsthal-type bounds by exponential sums (Erdos-Rankin and descendants):
  Identity 3 is a concrete statement that the first-moment character bound
  cannot distinguish two-residue sieves with the same density, which is the same
  obstruction FKMPT record for the adversarial problem from the other side.
* Lateral's round-21 pole-phase law: now identified as the `m`-th Fourier
  coefficient of `W`, which explains why the phase `90 + 180k/p` degrees appeared
  at every gear and frequency - it is the argument of `-e(m/P)/(1-e(m/P))`.

## 6. PRIOR-ART CHECK

**Not yet checked** (this lane has no web access).  Terms for the manager:
"first passage function of a periodic sieve Fourier transform";
"distance to next survivor exponential sum covering system";
"Jacobsthal function large sieve exponential sum lower bound obstruction";
"gap-weighted exponential sum reduced residue system";
"discrete renewal generating function 1/(1 - e(theta)) sieve".
The nearest relatives inside the project are `golden-spectral-gap.md` (the
machine DFT, whose statement this file corrects with the CRT frequency `c_q`),
`pole-phase-law.md` (now a corollary), `depth-sum-identity.md` (the same
depth-1-versus-depth-sum split), and `tooth-counterfactual-percentile.md` (the
null family that makes Identity 3 a usable negative).
