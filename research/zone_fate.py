"""Constructor round 6: (1) the inversion zone's fate at large y;
(2) the third-moment (LP/moment-problem) ceiling, mirror question included.

Part 1 - zone fate. Bottom-band scan (first T slots of the interior window;
marking all gears 5..y over the band: unmarked member <=> prime, so no
primality tests). R(t) = (S1^2/M2)/(t-P); decomposition R = eff * boost with
eff = (S1^2/M2)/n2 (Cauchy-Schwarz efficiency of the m-vector) and
boost = n2/(t-P) = 1 + n0/(t-P) (the twin surplus). The zone {R>1} dies iff
eff*boost < 1 everywhere: this scan reports which factor moves.

Part 2 - third-moment ceiling. Given moments S1, M2, M3 of the m-vector, the
sharp moment-problem lower bound on n2 = #{m>=1} is the LP
    V3 = max a*S1 + b*M2 + c*M3   s.t.  a*m + b*m^2 + c*m^3 <= 1  (all m >= 1)
(solved by active-set enumeration over integer m; feasibility checked on a
grid far beyond the arithmetic cap m <= (log_5 y^2)^2). Ceiling C3 = S1/V3;
compare C3 vs C_CS = M2/S1-based and vs the need M_X = S1/(t-P).
V2_int (integer 2-moment LP) is also reported - the honest version of CS.

MIRROR (both chunks, answered by proof, in the doc): the involution
k -> -k (mod any period) swaps left/right marks, so m(-k) = m(k), P and all
moments double, and every ratio (R, eff, boost, C_CS, C3, M_X) is invariant.
Mirror-restriction changes no arithmetic at moment level.
"""
import math
import sys
from itertools import combinations


def gear_sieve(y):
    gs = bytearray([1]) * (y + 1)
    gs[:2] = b"\x00\x00"
    for i in range(2, int(y**0.5) + 1):
        if gs[i]:
            gs[i * i:: i] = bytearray(len(gs[i * i:: i]))
    return [q for q in range(5, y + 1) if gs[q]]


def band_marks(y, T):
    """(wl, wr, ks) for the first T interior slots."""
    k = y // 6 + 1
    while 6 * k - 1 <= y:
        k += 1
    ks = k
    wl, wr = bytearray(T), bytearray(T)
    hi_member = 6 * (ks + T - 1) + 1
    for q in gear_sieve(y):
        m = (y // q + 1) * q
        while m <= hi_member:
            r = m % 6
            if r == 5:
                i = (m + 1) // 6 - ks
                if 0 <= i < T:
                    wl[i] += 1
            elif r == 1:
                i = (m - 1) // 6 - ks
                if 0 <= i < T:
                    wr[i] += 1
            m += q
    return wl, wr, ks


def zone_scan(y, T=50000):
    wl, wr, ks = band_marks(y, T)
    S1 = M2 = P = n2 = n0 = 0
    best = (0.0, None, 0, 0)                     # R, t, eff, boost
    first = last = None
    for i in range(T):
        a, b = wl[i], wr[i]
        P += (a == 0) + (b == 0)
        n0 += 1 if (a == 0 and b == 0) else 0
        m = a * b
        if m:
            n2 += 1
            S1 += m
            M2 += m * m
        t = i + 1
        dem = t - P
        if dem > 0 and S1:
            cs = S1 * S1 / M2
            R = cs / dem
            if R > best[0]:
                best = (R, t, cs / n2, n2 / dem)
            if R > 1:
                last = t
                if first is None:
                    first = t
    R, t, eff, boost = best
    zone = f"[{first},{last}]" if first else "EMPTY"
    cens = " (censored at T)" if last == T else ""
    print(f"y={y:>8}: sup R = {R:6.3f} at t={t:>6}  eff={eff:.3f} "
          f"boost={boost:.3f}  zone {zone}{cens}")
    return R


def lp_lower_bound(S1, M2, M3, grid=2000, mmax_active=60):
    """Sharp 3-moment lower bound on #{m>=1}; also the 2-moment integer LP."""
    ms = range(1, grid + 1)

    def feasible(a, b, c):
        return all(a * m + b * m * m + c * m ** 3 <= 1 + 1e-9 for m in ms)

    best2 = 0.0
    for m1, m2 in combinations(range(1, mmax_active + 1), 2):
        # solve a*m + b*m^2 = 1 at m1, m2
        det = m1 * m2 * m2 - m2 * m1 * m1
        a = (m2 * m2 - m1 * m1) / det
        b = (m1 - m2) / det
        if feasible(a, b, 0):
            best2 = max(best2, a * S1 + b * M2)
    best3 = best2
    for m1, m2, m3 in combinations(range(1, mmax_active + 1), 3):
        A = [[m1, m1**2, m1**3], [m2, m2**2, m2**3], [m3, m3**2, m3**3]]
        # solve 3x3 by Cramer
        d = (A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
             - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
             + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]))
        if d == 0:
            continue
        dx = ((A[1][1]*A[2][2]-A[1][2]*A[2][1])
              - A[0][1]*(A[2][2]-A[1][2]) + A[0][2]*(A[2][1]-A[1][1]))
        # simpler: numeric solve
        import numpy as _np  # local, tiny systems
        try:
            sol = _np.linalg.solve(_np.array(A, float), _np.ones(3))
        except Exception:
            continue
        a, b, c = sol
        if c > 1e-12:                            # cubic must not blow up
            continue
        if feasible(a, b, c):
            best3 = max(best3, a * S1 + b * M2 + c * M3)
    return best2, best3


def moment_ceilings(y, T=None, label=""):
    if T is None:                                # full interior window
        T = (y * y - 2) // 6 - (band_marks(y, 1)[2]) + 1
    wl, wr, _ = band_marks(y, T)
    S1 = M2 = M3 = P = n2 = 0
    for i in range(T):
        a, b = wl[i], wr[i]
        P += (a == 0) + (b == 0)
        m = a * b
        if m:
            n2 += 1
            S1 += m
            M2 += m * m
            M3 += m ** 3
    dem = T - P
    V2, V3 = lp_lower_bound(S1, M2, M3)
    mx = S1 / dem
    print(f"{label} (t={T}): n2={n2}  t-P={dem}  M_X={mx:.3f}  "
          f"M_real={S1/n2:.3f}")
    print(f"    ceilings: C_CS={M2/S1:.3f}  C2_int={S1/V2:.3f}  "
          f"C3={S1/V3:.3f}   (need < M_X={mx:.3f})")
    print(f"    n2 bounds: CS {S1*S1/M2:,.0f}  LP2 {V2:,.0f}  LP3 {V3:,.0f}  "
          f"true {n2:,}  X-demand {dem:,}")


if __name__ == "__main__":
    if "moments" in sys.argv:
        moment_ceilings(2003, None, "y=2003 full window")
        moment_ceilings(5003, None, "y=5003 full window")
        moment_ceilings(10007, 17204, "y=10007 zone prefix")
        moment_ceilings(50021, 50000, "y=50021 bottom band")
    else:
        for y in (10007, 20011, 50021, 100003, 200003, 500009,
                  1000003, 2000003, 5000011, 10000019):
            zone_scan(y)
