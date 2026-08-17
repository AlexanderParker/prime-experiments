"""Constructor round 2: the bottom-band double-onset law.

Objects (window of y = slots with both members in (y, y^2)):
  k_start(y) = first window slot
  k1(y)      = first DOUBLE slot above y (both members composite)
  L0(y)      = k1 - k_start  (the onset lag: n2 = 0 on the first L0 slots,
               unconditionally - no gear pair can place a double there)

Laws checked here:
  A. Roots-of-unity law: slot k is double-hit by the (unordered) gear pair
     {q, q'} iff 6k is a square root of 1 mod qq'. Trivial roots +-1 =
     same-member hits (qq' divides one member: the semiprime-multiple slots);
     nontrivial roots +-r (r = CRT(+1 mod q, -1 mod q')) = cross-member hits.
     So: k is a double slot iff 6k = +-r_{qq'} (mod qq') for some active pair
     - Lateral's semiprime pinning, generalised and made an iff.
  B. Brun-Titchmarsh cap on the onset lag: a run of L slots each containing a
     prime holds >= L primes in an interval of length 6L+2, and Montgomery-
     Vaughan gives pi(x+H) - pi(x) < 2H/ln H for every x, H >= 2. So
     L <= 2(6L+2)/ln(6L+2), i.e. ln(6L+2) <= 12 + 4/L: L0(y) <= L* ~ 27125
     for EVERY y, unconditionally. (Computed exactly below.)
  C. Ladder census: L0(y) and twins-before-first-double down a prime ladder.
     Under X the onset prefix must be perfectly fragile (one prime per slot);
     reality's violation there = twin count in the prefix (no doubles exist
     to offset them - C2's margin can only fall on the prefix).
  D. Descent mini-table: layer band (y'^2, y''^2) length in slots vs the
     measured max stride 0.47*ln^3(member)/6 at that height.
"""
import math
import sympy


def k_start(y):
    k = y // 6 + 1
    while 6 * k - 1 <= y:
        k += 1
    return k


def onset(y):
    """Return (k_start, k1, L0, twins in the onset prefix)."""
    ks = k_start(y)
    k, twins = ks, 0
    while True:
        pa, pb = sympy.isprime(6 * k - 1), sympy.isprime(6 * k + 1)
        if not pa and not pb:
            return ks, k, k - ks, twins
        if pa and pb:
            twins += 1
        k += 1


def check_roots_of_unity(y):
    """Law A, both directions, on the full window of y."""
    ks, ke = k_start(y), (y * y - 2) // 6
    for k in range(ks, ke + 1):
        a, b = 6 * k - 1, 6 * k + 1
        is_double = (not sympy.isprime(a)) and (not sympy.isprime(b))
        if is_double:
            q, qp = min(sympy.factorint(a)), min(sympy.factorint(b))
            assert q != qp, "slot-cap lemma"
            r = sympy.ntheory.modular.crt([q, qp], [1, -1])[0]
            m = q * qp
            assert pow(r, 2, m) == 1 and r % m not in (1, m - 1), "r nontrivial"
            assert (6 * k) % m in (r % m, (-r) % m), "double must sit on +-r"
        # converse: if 6k = +-r mod qq' for an ACTIVE pair (q,q' <= sqrt member,
        # q|a and q'|b) then both members are composite - immediate since the
        # members exceed y > q, q'. Spot-verify via any gear divisors:
        for q in sympy.primerange(5, int(math.isqrt(b)) + 1):
            if a % q == 0:
                for qp in sympy.primerange(5, int(math.isqrt(b)) + 1):
                    if qp != q and b % qp == 0:
                        assert is_double
    print(f"roots-of-unity law verified on full window y={y} "
          f"({ke - ks + 1} slots)")


def bt_cap():
    """Largest L compatible with L <= 2(6L+2)/ln(6L+2)."""
    L = 2
    while L * math.log(6 * L + 2) <= 2 * (6 * L + 2):
        L += 1
    return L - 1


def ladder(ymax):
    print("\nladder: y  k_start  k1  L0  twins-in-prefix (prefix = slots "
          "before the first double)")
    recs = []
    worstL0, worst_y = -1, None
    n_zero = n_twinlead = 0
    for y in sympy.primerange(13, ymax + 1):
        ks, k1, L0, tw = onset(y)
        recs.append((y, ks, k1, L0, tw))
        if L0 > worstL0:
            worstL0, worst_y = L0, y
        n_zero += (L0 == 0)
        n_twinlead += (tw >= 1)
    for row in recs[:8]:
        print("   %6d %7d %5d %4d %6d" % row)
    n = len(recs)
    print(f"  ... {n} windows tested (y <= {ymax})")
    print(f"  max L0 = {worstL0} at y = {worst_y}   "
          f"(vs unconditional BT cap L* = {bt_cap()})")
    print(f"  L0 = 0 (first window slot already double): {n_zero}/{n}")
    print(f"  >=1 twin strictly before the first double: {n_twinlead}/{n}")
    # distribution tail
    top = sorted(recs, key=lambda r: -r[3])[:5]
    print("  largest onset lags:", [(y, L0, f"tw={tw}") for y, _, _, L0, tw in top])


def descent_table():
    print("\ndescent: layer band (y'^2, y''^2) slots vs measured stride "
          "0.47*ln^3/6 at that height")
    for yp in (97, 997, 9973):
        ypp = sympy.nextprime(yp)
        band = (ypp * ypp - yp * yp) / 6
        stride = 0.47 * math.log(yp * yp) ** 3 / 6
        print(f"  y'={yp:6d} y''={ypp:6d}  band={band:9.0f} slots  "
              f"stride~{stride:7.0f}  ratio={band/stride:6.1f}")


if __name__ == "__main__":
    check_roots_of_unity(47)
    print(f"\nBrun-Titchmarsh onset cap: L* = {bt_cap()} slots "
          f"(6L*+2 = {6*bt_cap()+2}, e^12 = {math.e**12:.0f})")
    ladder(3163)
    descent_table()
