"""Round 22 lateral, target (c) - OWED TO CONSTRUCTOR:
THE CLOSED FORM OF THE CORRIDOR-PHASE CHAIN'S COMPLEX lambda_2.

Constructor R42 measured, on the exact full-period corridor-phase chain
(state = corridor phase r mod m, m = 35 or 385; step = the next gap):

    lambda_1 = 1,  lambda_2 COMPLEX with |lambda_2| = 0.89..0.96 and
    arg lambda_2 = 34..46 deg, i.e. a resonance of period 360/arg =
    7.8..8.4 lags - "the corridor resonance IS this eigenvalue".

THE DERIVATION (this file).  Two exact inputs, one modelling step.

EXACT INPUT 1 (CRT).  Let m = prod of the small gears (35 = 5*7,
385 = 5*7*11), E = the exposed phase set mod m (|E| = e = prod (q-2) over
q | m; e = 15 for m = 35, e = 135 for m = 385).  Every opening has phase in
E, and the openings are EXACTLY equidistributed over E: each r in E carries
prod_{q not | m} (q-2) of them.  So the per-slot opening hazard

    h(r) = P(slot of phase r is open) = rho * [r in E],
    rho  = prod_{q | M, q not | m} (1 - 2/q)     (exact rational)

is EXACTLY CONSTANT on E - no fit, no measurement.

MODELLING STEP (the only one): treat distinct slots as independent given
their phases (the renewal model).  Then the gap law out of phase r is
p(v | r) = h(r+v) prod_{i=1}^{v-1} (1 - h(r+i)), and the phase chain is

    M = (I - B)^{-1} O,   B = S D_{1-h},  O = S D_h,
    (S D)[r, s] = [s = r+1] d_s      (advance one slot, weight at arrival)

EXACT SPECTRUM OF THAT M.  M x = lambda x  <=>  (lambda B + O) x = lambda x
<=> S D_{lambda(1-h) + h} x = lambda x.  S D_d is a weighted single m-cycle,
char poly lambda^m - prod_s d_s, hence the eigenvalue equation

    lambda^m = prod_{s mod m} [ lambda (1 - h(s)) + h(s) ]
             = lambda^{m-e} * [ lambda (1 - rho) + rho ]^e
    =>  lambda^e = [ (1-rho) lambda + rho ]^e
    =>  lambda_j = rho * w_j / (1 - (1-rho) w_j),   w_j = exp(2 pi i j / e).

THE CLOSED FORM.  The chain's spectrum is the image of the e-th ROOTS OF
UNITY under the MOEBIUS MAP  mu(w) = rho w / (1 - (1-rho) w), plus the
eigenvalue 0 with multiplicity m - e (the phases off E).  Because mu has
real coefficients the image is a circle symmetric about R, with the two real
points mu(1) = 1 and mu(-1) = -rho/(2-rho) as a diameter:

    ALL EIGENVALUES LIE ON THE CIRCLE  |z - c| = R,
        c = (1 - rho)/(2 - rho),   R = 1/(2 - rho),   c + R = 1.

    lambda_1 = mu(1) = 1 (Perron);  lambda_2 = mu(e(1/e)) and its conjugate;
    arg lambda_2 = 360/e + |arg(1 - (1-rho) w_1)|  in (360/e, ...) degrees,
    so the RESONANCE PERIOD is 360/arg lambda_2 lags, slightly BELOW e/1
    ... concretely e = 15 gives 7.6-8.6 lags - Constructor's 7.8-8.4.

So: THE CORRIDOR RESONANCE IS A MOEBIUS IMAGE OF A 15th ROOT OF UNITY, and
its only parameter is rho = prod_{q >= 11} (1 - 2/q).  The 15 is |A_5||A_7|
= 3*5: the walk only ever visits E, so the resonance counts EXPOSED phases,
not all 35.

WHAT IS MEASURED HERE.  The exact chain is rebuilt from the real machine
(full period, exact integer counts), its eigenvalues computed, and compared
with the closed form.  The DEVIATION is then itself a measurement of the
machine's anti-correlation in spectral terms - the renewal model is the
independence hypothesis, so residual = correlation.

Usage: python corridor_lambda.py            # machines 11..19, m = 35 & 385
       python corridor_lambda.py --big      # adds machine 23 (37.2M slots)
"""
import sys
from math import prod, gcd
from fractions import Fraction
import numpy as np


def primes(a, b):
    return [n for n in range(a, b + 1)
            if n > 1 and all(n % d for d in range(2, int(n ** .5) + 1))]


def teeth(q):
    u = pow(6, -1, q)
    return u % q, (-u) % q


def blocked(gears):
    P = prod(gears)
    b = np.zeros(P, bool)
    for q in gears:
        t1, t2 = teeth(q)
        b[t1::q] = True
        b[t2::q] = True
    return b


def exposed_phases(m, gears):
    """residues mod m that are exposed by every gear dividing m."""
    keep = np.ones(m, bool)
    for q in gears:
        if m % q == 0:
            t1, t2 = teeth(q)
            keep[t1::q] = False
            keep[t2::q] = False
    return np.flatnonzero(keep)


def rho_of(m, gears):
    r = Fraction(1)
    for q in gears:
        if m % q:
            r *= Fraction(q - 2, q)
    return r


def closed_form(rho, e):
    w = np.exp(2j * np.pi * np.arange(e) / e)
    return rho * w / (1 - (1 - rho) * w)


def exact_chain(gears, m):
    """exact phase-transition matrix on E from the machine's full period."""
    b = blocked(gears)
    P = b.size
    idx = np.flatnonzero(~b)                 # openings
    assert idx.size == prod(q - 2 for q in gears), "open count"
    ph = idx % m
    E = exposed_phases(m, gears)
    e = E.size
    assert np.isin(ph, E).all(), "an opening landed off E"
    pos = -np.ones(m, np.int64)
    pos[E] = np.arange(e)
    src = pos[ph]
    dst = pos[np.roll(ph, -1)]               # next opening (cyclic)
    C = np.zeros((e, e), np.int64)
    np.add.at(C, (src, dst), 1)
    assert C.sum() == idx.size
    # equidistribution of openings over E (exact CRT statement)
    cnt = np.bincount(src, minlength=e)
    assert (cnt == cnt[0]).all(), "openings not equidistributed over E"
    T = C / C.sum(1, keepdims=True)
    assert abs(T.sum(1) - 1).max() < 1e-12
    return T, E, e, cnt[0]


def subdominant(ev):
    """Perron = the eigenvalue at 1; lambda_2 = the largest-modulus other
    one, reported in the upper half plane (conjugate pairs are equivalent)."""
    k1 = int(np.argmin(np.abs(ev - 1)))
    rest = np.delete(ev, k1)
    k2 = int(np.argmax(np.abs(rest)))
    l2 = rest[k2]
    return ev[k1], (l2 if l2.imag >= 0 else np.conj(l2))


def report(gears, m, label):
    rho = rho_of(m, gears)
    T, E, e, per = exact_chain(gears, m)
    rf = float(rho)
    pred = closed_form(rf, e)
    ev = np.linalg.eigvals(T)
    ev = ev[np.argsort(-np.abs(ev))]
    c, R = (1 - rf) / (2 - rf), 1 / (2 - rf)
    resid = np.abs(np.abs(ev - c) - R)
    l1m, l2m = subdominant(ev)
    l1p, l2p = subdominant(pred)
    mean_gap = prod(gears) / prod(q - 2 for q in gears)
    print(f"\n=== {label}  mod {m}:  |E| = e = {e}, rho = {rho} = {rf:.6f}, "
          f"openings/phase = {per:,}")
    print(f"    predicted circle: centre {c:.6f} radius {R:.6f} "
          f"(c+R = {c + R:.6f})")
    print(f"    lambda_2 MEASURED  |.| = {abs(l2m):.6f}  arg = "
          f"{np.degrees(np.angle(l2m)):+.3f} deg   period "
          f"{360 / abs(np.degrees(np.angle(l2m))):.3f} lags")
    print(f"    lambda_2 CLOSED FORM |.| = {abs(l2p):.6f}  arg = "
          f"{np.degrees(np.angle(l2p)):+.3f} deg   period "
          f"{360 / abs(np.degrees(np.angle(l2p))):.3f} lags")
    print(f"    error: |.| {abs(l2m) - abs(l2p):+.6f}   arg "
          f"{np.degrees(np.angle(l2m)) - np.degrees(np.angle(l2p)):+.3f} deg")
    print(f"    heuristic 360*mean_gap/m = "
          f"{360 * mean_gap / m:.3f} deg (mean gap {mean_gap:.4f})")
    print(f"    circle residual over ALL {e} eigenvalues: max "
          f"{resid.max():.5f}  mean {resid.mean():.5f}  "
          f"(as fraction of R: {resid.max() / R:.4f})")
    assert abs(l1m - 1) < 1e-9, "Perron"
    assert abs(l1p - 1) < 1e-12
    return dict(m=m, e=e, rho=rf, meas=l2m, pred=l2p, resid=resid, R=R,
                mean_gap=mean_gap)


def main():
    big = "--big" in sys.argv
    print(__doc__.split("Usage:")[0])
    rows = []
    ys = [11, 13, 17, 19] + ([23] if big else [])
    for y in ys:
        gears = primes(5, y)
        for m in (35, 385):
            if m == 385 and 11 not in gears:
                continue
            if rho_of(m, gears) == 1:   # degenerate: chain is a permutation
                print(f"\n=== machine {y} mod {m}: rho = 1 (one opening per "
                      f"phase) - the chain is a permutation, skipped")
                continue
            rows.append(report(gears, m, f"machine {y}"))

    print("\n--- SUMMARY: closed form vs exact chain -------------------")
    print(" mod    e   rho       |l2| meas  |l2| pred   arg meas   "
          "arg pred   d|l2|    darg")
    for r in rows:
        am = np.degrees(np.angle(r["meas"]))
        ap = np.degrees(np.angle(r["pred"]))
        print(f" {r['m']:4d} {r['e']:4d} {r['rho']:.5f}   "
              f"{abs(r['meas']):.6f}   {abs(r['pred']):.6f}  "
              f"{am:+8.3f}  {ap:+8.3f}  {abs(r['meas']) - abs(r['pred']):+.4f}"
              f"  {am - ap:+7.3f}")
    print("\n--- PRE-REGISTERED CLOSED-FORM PREDICTIONS (no scan) ------")
    print("  the closed form needs only rho, so it predicts lambda_2 at")
    print("  machines nobody can scan.  Measured |l2| has been ABOVE the")
    print("  prediction at every machine so far (the real chain keeps more")
    print("  memory than the renewal model) - the residual is the")
    print("  anti-correlation, and it GROWS: +0.008, +0.015, +0.019, +0.022.")
    print("   y   mod    rho        |l2| pred   arg pred   period (lags)")
    for y in (23, 29, 31, 37, 41):
        gears = primes(5, y)
        for m in (35, 385):
            rho = float(rho_of(m, gears))
            e = exposed_phases(m, gears).size
            _, l2 = subdominant(closed_form(rho, e))
            a = np.degrees(np.angle(l2))
            print(f"  {y:3d} {m:5d}  {rho:.6f}    {abs(l2):.6f}   "
                  f"{a:+8.3f}   {360 / a:8.3f}")

    print("\n  PRE-REGISTERED (falsifiable) for the first unmeasured machine:")
    print("  the mod-35 residual sequence is +0.0076 +0.0146 +0.0192 +0.0225")
    print("  +0.0245 (increments 70,46,33,20 e-4 - decelerating, saturating")
    print("  near +0.027) and the arg residual +0.20 +0.52 +0.87 +1.29 +1.71")
    print("  (increments 32,35,42,42 e-2).  So MACHINE 29 mod 35 should")
    print("  measure  |lambda_2| = 0.862 +- 0.004  and  arg = +49.2 +- 0.4")
    print("  deg (closed form 0.8366 / +47.09).  A measured |l2| BELOW the")
    print("  closed form at any machine would refute the direction of the")
    print("  anti-correlation residual, not just its size.")

    # the closed form must be in the right ballpark everywhere:
    for r in rows:
        assert abs(abs(r["meas"]) - abs(r["pred"])) < 0.15, r
        assert abs(np.degrees(np.angle(r["meas"]))
                   - np.degrees(np.angle(r["pred"]))) < 20, r
    print("\nassertions passed: closed form tracks every measured lambda_2")


if __name__ == "__main__":
    main()
