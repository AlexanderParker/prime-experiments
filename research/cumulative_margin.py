"""Constructor round 3: the cumulative C2 margin over FULL windows.

For each y: sieve members to y^2, walk the window slots, and record
  M(t)   = N(t) - P(t)  (prefix margin; X requires M(t) >= 0 everywhere)
  E(y)   = max over runs I of P(I) - N(I)  (Kadane; E >= 1 iff window has a twin)
plus where the extremes live. Also: layer-band length vs the proven
short-interval localisation exponent 0.525 (Baker-Harman-Pintz / Alweiss-Luo).

Cross-check against research/data/prefix_census.csv (Mechanic round 2):
overlapping y=101, 1009 first-200-slot minima must agree.
"""
import math


def sieve(n):
    bs = bytearray([1]) * (n + 1)
    bs[0:2] = b"\x00\x00"
    for i in range(2, int(n**0.5) + 1):
        if bs[i]:
            bs[i * i:: i] = bytearray(len(bs[i * i:: i]))
    return bs


def analyse(y):
    top = y * y
    isp = sieve(top + 2)
    k = y // 6 + 1
    while 6 * k - 1 <= y:
        k += 1
    ks = k
    ke = (top - 2) // 6
    M = 0
    minM, argmin, lastneg = 0, None, None
    run, best, blo, bhi, lo = 0, 0, None, None, ks
    n0 = 0
    for k in range(ks, ke + 1):
        p = isp[6 * k - 1] + isp[6 * k + 1]
        if p == 2:
            n0 += 1
        M += 1 - p
        if M < minM:
            minM, argmin = M, k
        if M < 0:
            lastneg = k
        run += p - 1
        if run < 0:
            run, lo = 0, k + 1
        elif run > best:
            best, blo, bhi = run, lo, k
    N = ke - ks + 1
    P = N - M  # final margin M = N - P
    frac_lo = (6 * blo - 1 - y) / (top - y)
    frac_hi = (6 * bhi + 1 - y) / (top - y)
    print(f"y={y:5d} N={N:8d} P={P:8d} twins={n0:6d} M(end)={M:7d} | "
          f"minM={minM:3d} at member ~{6*argmin+1 if argmin else '-'} "
          f"lastneg@{6*lastneg+1 if lastneg else '-'} | "
          f"E={best} on members {6*blo-1}..{6*bhi+1} "
          f"(window frac {frac_lo:.4f}..{frac_hi:.4f})")


def layer_bands():
    print("\nlayer bands at height x = y'^2: length vs proven localisation x^0.525")
    print("  (band_typ = 2*sqrt(x)*ln(sqrt(x)) ~ x^(1/2)*ln; band_min = 4*sqrt(x), twin y')")
    for e in (6, 9, 12, 18, 24, 30):
        x = 10.0 ** e
        typ = 2 * math.sqrt(x) * math.log(math.sqrt(x))
        thin = 4 * math.sqrt(x)
        loc = x ** 0.525
        print(f"  x=1e{e:<3d} band_typ=1e{math.log10(typ):5.2f}  "
              f"band_min=1e{math.log10(thin):5.2f}  x^0.525=1e{math.log10(loc):5.2f}  "
              f"loc/thin={loc/thin:9.3g}")


if __name__ == "__main__":
    for y in (47, 101, 199, 503, 1009, 2003, 5003):
        analyse(y)
    layer_bands()
