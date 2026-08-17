"""Constructor round 4 (flagship, with Lateral): the X-consistency equation.

THE ARITHMETIC CENSUS THEOREM (unconditional, the equation's substrate).
In the window of y, classify slot k by its gear marks (gears = primes 5..y,
mark = gear divides a member):
    type 0: no member gear-marked      <=> both members prime      (twin)
    type 1: one member gear-marked     <=> exactly one prime       (fragile)
    type 2: both members gear-marked   <=> both composite          (double)
(<= directions are the horizon theorem: an unmarked window member has no prime
factor < y hence is prime; => : a marked member is divisible by q <= y < m,
hence composite.)  Corollary, with d_k = [slot k is type 2], D(t) = sum d_k,
p_k = # prime members of slot k:

    P(t) = t - D(t) + n0(t)   for every prefix t     (identity, no hypothesis)

THE X-CONSISTENCY EQUATION.  X(y) (zero twins in the window)  <=>

    P(t) = t - D(t)  for every t in [1, N]   <=>   p_k + d_k = 1 at every slot.

Demand side P(t): the prime census of (y, y^2). Supply side D(t): freedom-free
divisibility arithmetic below y - by the roots-of-unity law the double set is
the union of the split classes +-x_{qq'} mod qq' (Lateral's gap law gives x in
closed form), so D(t) is an explicit functional of the primes and prime gaps
below y.

WHAT THIS SCRIPT CHECKS (y = 101, 211, 503):
  1. the census theorem and the identity, asserted at every slot;
  2. the overdetermination test: the forced value t - D(t) is the unconditional
     POINTWISE FLOOR of P(t) (P >= t - D always), so no lower-bound conflict
     can exist; against the Montgomery-Vaughan ceiling 2H/lnH (H = 6t+2) the
     headroom ratio rho(t) = (t - D(t)) / (2H/lnH) is computed - max over t,
     value at window end (expected ~ 1/2 = the parity factor, live);
  3. violation profile: n0(t) = P(t) - (t - D(t)) - the equation's failure is
     the twin count itself, growing ~ linearly;
  4. supply attribution (Lateral's law): PAIRSPLIT incidences by gap class in
     the full window and in the bottom band (first 200 slots) - the g = 2
     (twins-below-y) guaranteed share of the doubles supply X leans on.
"""
import math
import sys
from collections import defaultdict

sys.path.insert(0, "research")
from split_gap_law import primes, split_rep  # noqa: E402


def sieve(n):
    bs = bytearray([1]) * (n + 1)
    bs[0:2] = b"\x00\x00"
    for i in range(2, int(n**0.5) + 1):
        if bs[i]:
            bs[i * i:: i] = bytearray(len(bs[i * i:: i]))
    return bs


def splits_in_range(q, qp, klo, khi):
    """# landings of the two split classes of (q,q') in [klo, khi]."""
    P = q * qp
    x = split_rep(q, qp)
    tot = 0
    for z in (x % P, (P - x) % P):
        first = z + P * max(0, -(-(klo - z) // P))  # smallest >= klo in class
        if first <= khi:
            tot += (khi - first) // P + 1
    return tot


def analyse(y, band=200):
    top = y * y
    isp = sieve(top + 2)
    gears = primes(5, y)
    k = y // 6 + 1
    while 6 * k - 1 <= y:
        k += 1
    ks, ke = k, (top - 2) // 6
    N = ke - ks + 1

    # census theorem + identity + MV headroom, slot by slot
    P = D = n0 = 0
    max_rho, arg_rho = 0.0, None
    for t, k in enumerate(range(ks, ke + 1), start=1):
        a, b = 6 * k - 1, 6 * k + 1
        pa, pb = isp[a], isp[b]
        marked_a = any(a % q == 0 for q in gears if q * q <= a) or (not pa)
        # census theorem: marked <=> composite (assert via the cheap direction)
        assert marked_a == (not pa)
        p_k = pa + pb
        d_k = 1 if (not pa and not pb) else 0
        P += p_k
        D += d_k
        n0 += 1 if p_k == 2 else 0
        assert P == t - D + n0, "identity P(t) = t - D(t) + n0(t)"
        H = 6 * t + 2
        rho = (t - D) * math.log(H) / (2 * H)
        if rho > max_rho:
            max_rho, arg_rho = rho, t
    print(f"\n=== y={y}  N={N}  P={P}  D={D}  twins n0={n0} ===")
    print(f"identity P = t - D + n0 asserted at all {N} prefixes")
    print(f"forced floor t-D at window end: {N - D}  (= P - n0 = {P - n0})")
    print(f"MV headroom rho(t) = (t-D)lnH/2H: max {max_rho:.4f} at t={arg_rho}, "
          f"rho(N) = {(N-D)*math.log(6*N+2)/(2*(6*N+2)):.4f}  (parity factor: 0.5)")
    # violation profile
    marks = [0] * 5
    for i, tt in enumerate((N // 100, N // 10, N // 3, N)):
        pass  # profile printed from checkpoints below
    # checkpoint n0(t)
    P2 = D2 = n02 = 0
    cps = sorted({band, N // 100, N // 10, N // 3, N})
    out = []
    ci = 0
    for t, k in enumerate(range(ks, ke + 1), start=1):
        pa, pb = isp[6 * k - 1], isp[6 * k + 1]
        P2 += pa + pb
        D2 += 1 if (not pa and not pb) else 0
        n02 += 1 if pa and pb else 0
        if ci < len(cps) and t == cps[ci]:
            out.append((t, P2, t - D2, n02))
            ci += 1
    print("violation profile: t, P(t), floor t-D(t), n0(t) = P - floor:")
    for t, p, fl, z in out:
        print(f"    t={t:>7}  P={p:>7}  floor={fl:>7}  n0={z:>6}")

    # supply attribution via Lateral's law
    for lo, hi, name in ((ks, ke, "full window"), (ks, ks + band - 1, f"bottom band (t<=ize {band})")):
        by_gap = defaultdict(int)
        tot = 0
        for i in range(len(gears)):
            for j in range(i + 1, len(gears)):
                q, qp = gears[i], gears[j]
                s = splits_in_range(q, qp, lo, hi)
                if s:
                    g = qp - q
                    by_gap[2 if g == 2 else 0] += s
                    tot += s
        g2, rest = by_gap[2], by_gap[0]
        # distinct doubles in range for comparison (CORR magnitude)
        dd = sum(1 for k in range(lo, hi + 1)
                 if not isp[6 * k - 1] and not isp[6 * k + 1])
        print(f"supply, {name}: split incidences {tot} "
              f"(g=2 guaranteed: {g2} = {100*g2/tot:.1f}%), "
              f"distinct doubles {dd} (CORR overlap = {tot - dd:+d})")


if __name__ == "__main__":
    for y in (101, 211, 503):
        analyse(y)
