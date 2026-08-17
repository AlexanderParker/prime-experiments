"""What `kappa(L)` is made of: a linear term minus a sum of pair weights.

`n(L) = N(L)/P` is the chance that `L` consecutive positions are all blocked. Expanding over which
positions are exposed,

    n(L) = sum_{T subset of [0,L)} (-1)^{|T|} v(T),    v(T) = prod_q ( 1 - |W_q(T)|/q ),

so `v(empty) = 1`, every singleton has `v = prod (1 - 2/q) = d`, and pairs are the first term of order
`d^2`. Writing `A_L` for the pair sum and keeping terms to second order,

    n(L)   = 1 - L d + A_L + O(d^3),        A_L = sum_delta (L - delta) v(delta)
    h(L)   = ( n(L) - n(L+1) ) / n(L) = ( d - B_L ) / n(L),   B_L = sum_{delta <= L} v(delta)
           = d - B_L + L d^2 + O(d^3)

and therefore, with `psi(delta) = v(delta)/d^2`,

    kappa(L) = ( h(L)/d - 1 ) / d = L - sum_{delta <= L} psi(delta) + O(d).

**The pair weight in closed form.** For a pair at distance `delta` the four values `t-1, t, t+delta-1,
t+delta` collide mod `q` exactly when `q | delta` (two collisions, `|W_q| = 2`), or `q | delta-1` or
`q | delta+1` (one collision, `|W_q| = 3`); otherwise `|W_q| = 4`. Gear 3 always divides one of
`delta-1, delta, delta+1`, and the case `3 | delta +- 1` gives the factor `1 - 3/3 = 0`, so **only
`delta = 0 mod 3` contributes** - the gear-3 law reappearing as a term-by-term vanishing. Dividing
through by `d^2`:

    psi(delta) = 3 C * prod_{q | delta, q >= 5} (q-2)/(q-4) * prod_{q | delta^2 - 1, q >= 5} (q-3)/(q-4)

with `C = prod_{q >= 5} ( 1 - 4/(q-2)^2 )`, the `q >= 5` factor of `(1-4/q)/(1-2/q)^2`.

So the remaining gap - `kappa(L) >= 1/(1-d)` for `L >= 2` - is, to leading order, the statement that the
pair weights average to less than 1 per unit of `L`:

    sum_{delta <= L, 3 | delta} psi(delta)  <=  L - 1

which is an inequality about a divisor-sum function rather than about the pattern. This module checks the
expansion against the measured `kappa` and then measures the average of `psi`.
"""

import sys
from math import prod
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from closed_hazard import kappa


def primes_upto(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n**0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return [int(p) for p in np.flatnonzero(s)]


def odd_prime_factors(n, primes):
    """Primes `>= 5` dividing `n`.

    The 2s and 3s must be divided out first even though they are not wanted in the output. Skipping
    them in the trial-division loop instead leaves them in the cofactor, and the final
    `if n >= 5: append(n)` then treats a composite remainder as prime - `odd_prime_factors(6)` returns
    `[6]`, which fed a spurious factor `(6-2)/(6-4) = 2` into `psi(6)`.
    """
    for small in (2, 3):
        while n % small == 0:
            n //= small
    out = []
    for q in primes:
        if q < 5:
            continue
        if q * q > n:
            break
        if n % q == 0:
            out.append(q)
            while n % q == 0:
                n //= q
    if n >= 5:
        out.append(n)
    return out


def C_constant(primes, y=None):
    """`prod_{q >= 5, q <= y} ( 1 - 4/(q-2)^2 )`."""
    c = 1.0
    for q in primes:
        if q < 5:
            continue
        if y is not None and q > y:
            break
        c *= 1.0 - 4.0 / (q - 2) ** 2
    return c


_C_CACHE = {}


def C_cached(primes, y):
    """`C` is a product over every gear, so it is computed once per `y`, not once per call."""
    if y not in _C_CACHE:
        _C_CACHE[y] = C_constant(primes, y)
    return _C_CACHE[y]


def psi(delta, primes, y=None):
    """The normalised pair weight `v(delta)/d^2`. Zero unless `3 | delta`."""
    if delta % 3:
        return 0.0
    val = 3.0 * C_cached(primes, y)
    for q in odd_prime_factors(delta, primes):
        if y is None or q <= y:
            val *= (q - 2) / (q - 4)
    for m in (delta - 1, delta + 1):
        for q in odd_prime_factors(m, primes):
            if y is None or q <= y:
                val *= (q - 3) / (q - 4)
    return val


def psi_series(Lmax, primes, y=None):
    """`psi(delta)` for every multiple of 3 up to `Lmax`, and its running sum."""
    vals = {}
    running = {}
    s = 0.0
    for delta in range(3, Lmax + 1, 3):
        vals[delta] = psi(delta, primes, y)
        s += vals[delta]
        running[delta] = s
    return vals, running


def kappa_predicted(L, primes, y=None, running=None):
    """`L - sum_{delta <= L} psi(delta)`, the leading-order value."""
    if running is not None:
        return L - running[L - (L % 3) if L % 3 else L]
    return L - sum(psi(delta, primes, y) for delta in range(3, L + 1, 3))


def psi_sieved(Lmax):
    """`psi(delta)` for every multiple of 3 up to `Lmax`, via a smallest-prime-factor sieve.

    Trial division costs about 450 divisions per value at `Lmax = 10^7`, which is far too slow for a
    few million values. Sieving the smallest prime factor once makes each factorisation a handful of
    lookups. Returns `(sum_psi_running, kappa_min, argmin)` where the running sum is indexed by
    `delta/3`.
    """
    n = Lmax + 2
    spf = np.zeros(n, dtype=np.int32)
    for i in range(2, int(n**0.5) + 1):
        if spf[i] == 0:
            spf[i * i::i] = np.where(spf[i * i::i] == 0, i, spf[i * i::i])
    # any position still 0 and >= 2 is prime
    C = C_constant(primes_upto(2_000_000))

    def factors_ge5(m):
        out = []
        while m > 1:
            p = int(spf[m]) if spf[m] else m
            if p >= 5:
                out.append(p)
            while m % p == 0:
                m //= p
        return out

    running = 0.0
    kmin, argmin = float("inf"), None
    for delta in range(3, Lmax + 1, 3):
        val = 3.0 * C
        for q in factors_ge5(delta):
            val *= (q - 2) / (q - 4)
        for q in factors_ge5(delta - 1):
            val *= (q - 3) / (q - 4)
        for q in factors_ge5(delta + 1):
            val *= (q - 3) / (q - 4)
        running += val
        k = delta - running
        if k < kmin:
            kmin, argmin = k, delta
    return running, kmin, argmin


if __name__ == "__main__":
    primes = primes_upto(2_000_000)
    y = 100003
    gears = [q for q in primes if 2 < q <= y]

    print(f"C = prod_(q>=5) (1 - 4/(q-2)^2) = {C_constant(primes):.9f},  3C = {3 * C_constant(primes):.9f}")
    print(f"psi(3) should be 3C exactly, since 3^2 - 1 = 8 has no prime factor >= 5: "
          f"{psi(3, primes):.9f}")

    print(f"\nexpansion against measured kappa, at y = {y} (d is small, so O(d) is small)")
    print(f"  {'L':>4} {'measured':>10} {'predicted':>10} {'diff':>9} {'sum psi':>10}")
    for L in [3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 45, 54, 63]:
        meas, d = kappa(L, gears)
        pred = kappa_predicted(L, primes, y)
        s = sum(psi(delta, primes, y) for delta in range(3, L + 1, 3))
        print(f"  {L:>4} {meas:>10.4f} {pred:>10.4f} {meas - pred:>9.4f} {s:>10.4f}")

    print("\nthe average of psi over multiples of 3, and the implied slope of kappa")
    LMAX = 300_000
    vals, running = psi_series(LMAX, primes)
    print(f"  {'up to L':>9} {'terms':>7} {'sum psi':>12} {'mean psi':>9} "
          f"{'kappa = L - sum':>16} {'kappa/L':>8}")
    for L in (63, 300, 3000, 30000, 300000):
        s = running[L]
        n = L // 3
        print(f"  {L:>9} {n:>7} {s:>12.3f} {s / n:>9.5f} {L - s:>16.3f} {(L - s) / L:>8.5f}")

    print("\nminimum of the predicted kappa over L, which is what a uniform bound must clear")
    cand = [(L - running[L], L) for L in range(3, LMAX + 1, 3)]
    lo = sorted(cand)[:8]
    print(f"  lowest predicted kappa over L <= {LMAX}: {[(round(v, 4), L) for v, L in lo]}")
    neg = [(L, round(v, 4)) for v, L in cand if v < 1.0]
    print(f"  block starts with predicted kappa < 1: {len(neg)}"
          + (f", first few {neg[:6]}" if neg else ""))
