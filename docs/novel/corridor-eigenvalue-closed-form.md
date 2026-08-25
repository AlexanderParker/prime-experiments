# The corridor resonance in closed form: a Moebius image of a root of unity

Status: PROVED (exact spectrum of the model, one-line determinant) +
SCRIPT-VERIFIED against the real machine's exact full-period chains
(`research/corridor_lambda.py`, machines 11/13/17/19/23, moduli 35 and 385).
Established round 22 (Lateral, on Constructor's R42 request). Prior-art
check: NOT YET CHECKED (section 6).

## 1. WHAT IT IS

Plain language. Constructor R42 found that the machine's extreme gaps are
carried by a chain whose state is the CORRIDOR PHASE (the slot index mod 35,
or mod 385), and measured that this chain's second eigenvalue is COMPLEX -
|lambda_2| between 0.89 and 0.96, argument between 34 and 46 degrees - so the
corridor "resonates" with a period of 360/arg = 7.8 to 8.4 lags. That
eigenvalue was the corridor resonance itself, but it was a measured number
with no formula. This entry gives the formula. The whole spectrum of the
corridor-phase chain is the image of the ROOTS OF UNITY under a MOEBIUS MAP
whose single parameter is the density of the large gears.

Precise form. Let m be a product of small gears (m = 35 = 5*7 or
m = 385 = 5*7*11), let A_q = Z_q minus gear q's two teeth, and let

    E = the exposed phase set mod m  (E = prod_{q | m} A_q under CRT),
    e = |E| = prod_{q | m} (q - 2)   (e = 15 for m = 35; e = 135 for 385),
    rho = prod_{q | M, q not | m} (1 - 2/q).

EXACT INPUT (CRT, no measurement): every opening has phase in E, and the
openings are exactly equidistributed over E - each r in E carries
prod_{q not | m} (q-2) of them. So the per-slot opening hazard is EXACTLY
h(r) = rho * [r in E]: constant on E, zero off it.

MODELLING STEP (the only one): slots independent given their phases. Then
the phase-of-the-next-opening chain is M = (I - B)^{-1} O with
B = S D_{1-h}, O = S D_h, (S D)[r,s] = [s = r+1] d_s.

THEOREM. M x = lambda x  <=>  S D_{lambda(1-h)+h} x = lambda x, and
S D_d is a weighted single m-cycle with characteristic polynomial
lambda^m - prod_s d_s. Hence

    lambda^m = prod_{s mod m} [ lambda(1 - h(s)) + h(s) ]
             = lambda^{m-e} [ (1-rho) lambda + rho ]^e,

so lambda = 0 with multiplicity m - e (the phases off E) and otherwise

    lambda_j = mu(w_j),  mu(w) = rho w / (1 - (1-rho) w),  w_j = e(j/e),
    j = 0, ..., e-1.

COROLLARIES, all exact:
* lambda_1 = mu(1) = 1 (Perron);
* THE WHOLE SPECTRUM LIES ON ONE CIRCLE, because mu has real coefficients
  and maps the unit circle to a circle whose real diameter is
  [mu(-1), mu(1)] = [-rho/(2-rho), 1]:

      |z - c| = R,   c = (1 - rho)/(2 - rho),   R = 1/(2 - rho),  c + R = 1;

* lambda_2 = mu(e(1/e)) (and its conjugate), so

      |lambda_2| = rho / |1 - (1-rho) e(1/e)|,
      arg lambda_2 = 360/e + |arg(1 - (1-rho) e(1/e))|  degrees,

  and the resonance period is 360 / arg lambda_2 lags.

THE RESONANCE IS NOT MOD 35 - IT IS MOD 15. The root of unity is an e-th
root, e = |A_5||A_7| = 3*5 = 15, not a 35th root: the walk never visits a
blocked phase, so the resonance counts EXPOSED phases only. That is why the
measured period sits near 8 rather than near 35/2.

## 1b. ROUND-23 UPDATE - THE DEFICIT IS CLOSED, AND IT WAS A COORDINATE

The formula above understated the measured modulus by a stable, converging
amount (+0.0076, +0.0146, +0.0192, +0.0225, +0.0245 at machines 11-23;
Constructor independently reported ~0.029 for their p-hat(1) form). Round 23
closed it EXACTLY, and the answer was not a correlation correction - it was
the choice of coordinate.

THE IDENTITY. Let q(n) be the machine's EXPOSED-STEP LAW: the distribution of
the number of exposed phases mod m that one gap crosses (equivalently, how
many steps the walk on E advances). Then

    lambda_2  =  q-hat(1/e)  =  sum_n q(n) e(n/e)      TO 1e-5,

measured against the exact full-period chain: |lambda_2| residual 0.000000,
0.000011, 0.000032, 0.000065 and argument residual 0.000, 0.005, 0.009, 0.012
degrees at machines 11/13/17/19 (mod 35), and 0 to 6 decimals at mod 385.
A phase-blind chain in the EXPOSED-STEP coordinate is an exact circulant on
Z_e, so its eigenvalues are exactly q-hat(j/e); the 1e-5 residual is the
entire phase-dependence of the step law.

WHAT THE OLD FORMULA WAS. Under the independence hypothesis the step law is
GEOMETRIC(rho), and sum_n rho(1-rho)^{n-1} w^n = rho w/(1-(1-rho)w) = mu(w).
So the round-22 closed form is exactly q-hat with q geometric, and

    THE 0.029 DEFICIT IS ENTIRELY THE NON-GEOMETRICITY OF THE STEP LAW.

It is not a 2-point, 3-point or higher CRT correlation effect: two refined
models built and PRE-REGISTERED in round 23 (a 2-point conditional hazard,
and the both-endpoint 3-point interior of the round-20 renewal law) both moved
|lambda_2| in the WRONG DIRECTION and made the deficit 50-120% worse. They
failed because they corrected the SLOT-LAG hazard, which double-counts the
phase structure the corridor already carries.

THE STEP LAW, PARTLY IN CLOSED FORM. Its mean is EXACTLY 1/rho at every
machine (a CRT identity - the geometric model has the right mean, so the
deficit is pure shape). Its first term is exactly closed-form: if d(r) is the
slot distance from an exposed phase r to the next exposed phase, then

    q(1) = avg over r in E of prod_{q not | m} c_q(d(r))/(q-2),

verified against the exact law to 2.2e-16 at machines 11/13/17/19
(0.777777777778 = 7/9, 0.636363636364, 0.551515151515, 0.486631016043).
Measured q(n)/geometric(n) for n = 1..6 shows the shape: n=1 SUPPRESSED
(0.951, 0.919, 0.903, 0.890 at m11/13/17/19 - two-point repulsion at the
corridor's own step) and n=2 ENHANCED (1.494, 1.378, 1.299, 1.247), then a
decaying tail. So the corridor pinning is a one-dimensional, fully
measurable object: the departure of one step law from geometric.

Script: `research/lambda2_pair.py` (models 0-4, all pre-registered verdicts
printed including the two refuted ones); log
`research/data/lambda2_pair.log`. STATUS of this update: the identity
lambda_2 = q-hat(1/e) is SCRIPT-VERIFIED to 1e-5 (it is exact for a
phase-independent step law, which the machine's is not quite); the q(1)
closed form is PROVED (CRT) and verified to machine precision; the mean
identity is exact.

## 2. WHY IT MIGHT BE NOVEL

The pieces are standard (a Markov-renewal chain observed at renewal epochs;
the characteristic polynomial of a weighted cyclic permutation). What looks
new is the combination and the conclusion:

- the observed-at-openings chain of a periodic-hazard renewal process on Z_m
  has its ENTIRE spectrum given by one Moebius map of the e-th roots of
  unity, with e = the number of positive-hazard phases (not m), and hence
  lies on an explicit circle through 1;
- applied to a sieve, this identifies a measured "corridor resonance" of a
  prime-gap machine as a specific algebraic number, with the small gears
  fixing the root of unity and the large gears fixing the Moebius parameter;
- the DEVIATION from the formula is then a clean, one-number measurement of
  the sieve's anti-correlation, because the formula is exactly the
  independence hypothesis.

## 3. PROOF / STATUS

The eigenvalue equation is proved above (two lines). What is verified
numerically is that the real machine's chain - built from full periods, exact
integer transition counts - matches it:

  mod  e   rho        |l2| meas  |l2| pred  arg meas  arg pred   d|l2|   darg
   35  15  9/11       0.984944   0.977314   +29.265   +29.068  +0.0076  +0.20
   35  15  9/13       0.963366   0.948729   +34.393   +33.875  +0.0146  +0.52
   35  15  135/221    0.939602   0.920450   +38.667   +37.798  +0.0192  +0.87
   35  15  135/247    0.912492   0.890002   +42.768   +41.477  +0.0225  +1.29
   35  15  2835/5681  0.885867   0.861354   +46.305   +44.592  +0.0245  +1.71
  385 135  11/13      0.999830   0.999767    +3.151    +3.151  +0.0001  +0.00
  385 135  165/221    0.999630   0.999508    +3.571    +3.571  +0.0001  +0.00
  385 135  165/247    0.999385   0.999195    +3.991    +3.990  +0.0002  +0.00
  385 135  3465/5681  0.999125   0.998866    +4.371    +4.370  +0.0003  +0.00

(machines 11, 13, 17, 19, 23; the mod-35 rows reproduce Constructor's
measured 0.96 / 0.91 / 0.89 at machines 13 / 19 / 23 and their 34-46 degree
argument range exactly.) The circle statement is verified too: the residual
| |z - c| - R | over ALL e eigenvalues is at most 0.15 R at mod 35 and
0.10 R at mod 385.

Status: PROVED for the model; SCRIPT-VERIFIED as an approximation to the
machine, with the residual measured and reported rather than hidden.

## 4. IMPLICATIONS

Inside the project:
- Constructor's corridor resonance now has a formula, so it can be predicted
  at machines nobody can scan. Closed-form predictions (mod 35):
  y = 29: |l2| = 0.8366, arg +47.09; y = 31: 0.8118, +49.44;
  y = 37: 0.7900, +51.40; y = 41: 0.7696, +53.17.
- PRE-REGISTERED, FALSIFIABLE: the measured residual has been POSITIVE in
  modulus at every machine (the real chain keeps MORE memory than
  independence predicts) and grows with a decelerating increment
  (+0.0076, +0.0146, +0.0192, +0.0225, +0.0245; increments 70, 46, 33, 20
  e-4). So machine 29 mod 35 should measure |lambda_2| = 0.862 +- 0.004 and
  arg = +49.2 +- 0.4 deg. A measured |lambda_2| BELOW the closed form at any
  machine refutes the direction of the residual, not just its size.
- The resonance PERIOD shortens monotonically with the machine
  (12.3 -> 7.8 lags measured, 6.8 predicted at y = 41) - it is not a fixed
  "period 8" phenomenon.
- The residual IS the anti-correlation, measured as a single number per
  machine. This is a much cheaper handle on the (D) suppression than a joint
  census: one eigenvalue against one formula.

Outside: an exactly solvable spectrum for observed-at-renewal chains with
periodic hazard - the spectrum is a Moebius image of roots of unity on an
explicit circle - which is a statement about renewal processes, not about
primes.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

Requirement (D) / the suppression law: the residual above is a new,
one-number proxy for the machine's excess memory; whether it converges
(saturating near +0.027) or keeps growing decides whether the corridor chain
is asymptotically renewal. Twin prime route: nothing here closes anything -
it prices the anti-correlation, it does not bound it.

## 6. PRIOR-ART CHECK

Not yet checked (agent without web access). Terms for the manager:
"spectrum of a Markov renewal process observed at renewal epochs periodic
hazard"; "eigenvalues weighted cyclic permutation matrix characteristic
polynomial lambda^n - product"; "Moebius transform roots of unity Markov
chain spectrum circle"; "phase-type renewal chain modulo m spectral
decomposition"; "spectral gap of residue walk sieve corridor". Expected
nearest art: semi-Markov / Markov-renewal spectral theory (the resolvent
identity M = (I-B)^{-1} O is standard); the delta to check is the closed
circle |z - (1-rho)/(2-rho)| = 1/(2-rho) and the e-th (not m-th) roots.
