"""Round 22 (constructor): CLOSED FORM OF THE CORRIDOR-PHASE lambda_2.

R42 built the corridor-phase transfer chain (state = left-endpoint phase mod
35 / 385) and found its second eigenvalue is COMPLEX: |l2| = 0.963 / 0.912 /
0.886 at machines 13 / 19 / 23, arg 34 / 43 / 46 degrees, so the corridor
resonance has period 360/arg = 7.8-8.4 lags.  The value was measured from the
empirical transition matrix; this script derives it.

THE CLOSED FORM.  The phase chain's transition is "add the next gap, mod M".
If the gap value is taken independently of the phase, the transition matrix on
Z_M is the CIRCULANT of the gap distribution, whose eigenvalues are exactly
its discrete Fourier coefficients:

    lambda_k  =  phat(k)  =  sum_g  P(gap = g) e(g k / M),   k = 0..M-1

so lambda_1 = 1 (k = 0) and

    lambda_2 = the largest non-DC phat(k) = the GAP DISTRIBUTION'S
               CHARACTERISTIC FUNCTION AT THE CORRIDOR FREQUENCY.

That is a closed form in one exactly-known object (the full-period gap
histogram), and it explains both measured features at once:
    arg lambda_2  ~  2 pi k mean_gap / M      (leading order; the R42
                                               observation arg ~ 2pi gbar/35)
    |lambda_2|    ~  exp(-2 pi^2 k^2 var / M^2)  (Gaussian/cumulant form)

Checked here against the exact circulant (which needs no independence
assumption to be an eigenvalue statement about ITSELF) and against the
measured R42 values.  The EXACT phase chain is also diagonalised directly, so
the gap between "iid-circulant closed form" and "true chain" is quantified
rather than assumed away.

Usage: uv run python research/lambda2_closed.py [y ...]
"""
import cmath
import math
import os
import sys
from math import prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MEASURED = {13: (0.963, 34.0), 19: (0.912, 43.0), 23: (0.886, 46.0)}


def primes(a, b):
    return [n for n in range(a, b + 1)
            if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))]


def lateral_lambda(y, M, j=1):
    """Lateral's round-22 closed form: lambda_j = rho w_j / (1 - (1-rho) w_j)
    with w_j = e(j/e), e = |E| = prod_{q | M} (q-2), rho = prod_{q not | M}
    (1 - 2/q).  Note this is exactly the characteristic function of a
    GEOMETRIC gap distribution of density rho evaluated at an e-th root of
    unity - i.e. the renewal instance of phat, which is why the two closed
    forms agree in shape."""
    gears = primes(5, y)
    e = prod(q - 2 for q in gears if M % q == 0)
    rho = prod(1 - 2 / q for q in gears if M % q != 0)
    w = cmath.exp(2j * math.pi * j / e)
    return rho * w / (1 - (1 - rho) * w), e, rho


def stream_chain(y, M, seg=48_000_000):
    """Exact 35x35 (or MxM) phase-transition counts and gap histogram for a
    machine too big to hold: one segmented pass, cyclic seam stitched."""
    gears = primes(5, y)
    P = prod(gears)
    uvals = [pow(6, -1, g) for g in gears]
    T = np.zeros((M, M), np.float64)
    hist = np.zeros(256, np.float64)
    tail = None
    head = None
    for lo in range(0, P, seg):
        hi = min(P, lo + seg)
        ex = np.zeros(hi - lo, bool)
        for g, u in zip(gears, uvals):
            ex[(u - lo) % g::g] = True
            ex[(-u - lo) % g::g] = True
        op = (np.flatnonzero(~ex) + lo).astype(np.int64)
        del ex
        if head is None:
            head = op[:1].copy()
        ops = op if tail is None else np.concatenate([tail, op])
        d = np.diff(ops)
        ph = ops[:-1] % M
        np.add.at(T, (ph, (ph + d) % M), 1.0)
        hist += np.bincount(d, minlength=256)[:256]
        tail = ops[-1:].copy()
        del op, ops
    ops = np.concatenate([tail, head + P])
    d = np.diff(ops)
    ph = ops[:-1] % M
    np.add.at(T, (ph, (ph + d) % M), 1.0)
    hist += np.bincount(d, minlength=256)[:256]
    return T, hist


def gaps_and_phases(y, M):
    gears = primes(5, y)
    P = prod(gears)
    ex = np.zeros(P, bool)
    for g in gears:
        u = pow(6, -1, g)
        ex[u % g::g] = True
        ex[(-u) % g::g] = True
    op = np.flatnonzero(~ex).astype(np.int64)
    d = np.diff(np.concatenate([op, [op[0] + P]]))
    return op, d, P


def run(y, M=35, stream=False):
    if stream:
        T0, hist = stream_chain(y, M)
        n = int(hist.sum())
        gs0 = np.arange(len(hist))
        gbar = float((gs0 * hist).sum() / n)
        var = float((gs0 ** 2 * hist).sum() / n - gbar ** 2)
    else:
        op, d, P = gaps_and_phases(y, M)
        n = len(d)
        gbar = d.mean()
        var = d.var()
        hist = np.bincount(d, minlength=int(d.max()) + 1).astype(np.float64)
    p = hist / hist.sum()
    gs = np.arange(len(p))
    lam = np.array([np.sum(p * np.exp(2j * math.pi * gs * k / M))
                    for k in range(M)])
    k2 = 1 + int(np.argmax(np.abs(lam[1:])))
    l2 = lam[k2]
    # gaussian/cumulant prediction for that k
    th = 2 * math.pi * k2 / M
    approx = cmath.exp(1j * th * gbar - 0.5 * th * th * var)
    # the EXACT phase chain (no independence assumption): empirical
    # transition matrix on Z_M, phase -> phase + gap
    if stream:
        T = T0
    else:
        T = np.zeros((M, M))
        ph = (op % M).astype(np.int64)
        np.add.at(T, (ph, (ph + d) % M), 1.0)
    sup = np.flatnonzero(T.sum(1) > 0)      # only the exposed corridor
    T = T[np.ix_(sup, sup)]
    T /= T.sum(1, keepdims=True)
    ev = np.linalg.eigvals(T)
    ev = ev[np.argsort(-np.abs(ev))]
    assert abs(abs(ev[0]) - 1) < 1e-9
    ex2 = ev[1]
    print("machine %2d  (M = %d, %d gaps, mean %0.4f, var %0.4f)"
          % (y, M, n, gbar, var))
    print("   closed form  lambda_2 = phat(%d) = %.4f  arg %+.2f deg"
          % (k2, abs(l2), math.degrees(cmath.phase(l2))))
    print("   cumulant     approx    = %.4f  arg %+.2f deg   "
          "(exp(i.th.gbar - th^2 var/2), th = 2pi.%d/%d)"
          % (abs(approx), math.degrees(cmath.phase(approx)), k2, M))
    print("   exact chain  lambda_2 = %.4f  arg %+.2f deg   "
          "[iid-circulant error %.4f in modulus, %.2f deg in phase]"
          % (abs(ex2), math.degrees(cmath.phase(ex2)),
             abs(abs(ex2) - abs(l2)),
             abs(math.degrees(cmath.phase(ex2)) -
                 math.degrees(cmath.phase(l2)))))
    if y in MEASURED:
        mm, ma = MEASURED[y]
        print("   R42 measured          = %.3f  arg %+.1f deg" % (mm, ma))
        assert abs(abs(ex2) - mm) < 0.01, (y, abs(ex2), mm)
        assert abs(abs(math.degrees(cmath.phase(ex2))) - ma) < 1.5, \
            (y, math.degrees(cmath.phase(ex2)), ma)
        print("   -> R42's measured lambda_2 REPRODUCED from the "
              "phase chain")
    lat, e, rho = lateral_lambda(y, M)
    print("   Lateral      lambda_2 = %.4f  arg %+.2f deg   "
          "[rho*w/(1-(1-rho)w), e = %d, rho = %.4f; err %.4f, %.2f deg]"
          % (abs(lat), math.degrees(cmath.phase(lat)), e, rho,
             abs(abs(ex2) - abs(lat)),
             abs(math.degrees(cmath.phase(ex2)) -
                 math.degrees(cmath.phase(lat)))))
    # resonance period
    per = 360.0 / abs(math.degrees(cmath.phase(l2)))
    print("   resonance period 360/arg = %.2f lags (closed form)   "
          "35/mean_gap = %.2f" % (per, M / gbar))
    return dict(y=y, k2=k2, l2=l2, exact=ex2, gbar=gbar, var=var)


def main():
    ys = [int(x) for x in sys.argv[1:]] or [11, 13, 17, 19, 23]
    rs = [run(y, stream=(y >= 29)) for y in ys]
    print("\n=== summary: lambda_2 = phat(k*) at the corridor frequency")
    print("    y   k*   |phat|    arg      |exact|   arg      mean gap")
    for r in rs:
        print("  %3d %4d   %.4f  %+7.2f   %.4f  %+7.2f   %.4f"
              % (r["y"], r["k2"], abs(r["l2"]),
                 math.degrees(cmath.phase(r["l2"])), abs(r["exact"]),
                 math.degrees(cmath.phase(r["exact"])), r["gbar"]))
    print("\nall assertions passed")


if __name__ == "__main__":
    main()
