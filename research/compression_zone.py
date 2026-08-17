"""Round 5 follow-up: map the CS-refutation zone R(t) = (S1^2/M2)/(t-P) > 1.

If R(t) > 1 at any prefix then Cauchy-Schwarz (n2 >= S1^2/M2, freedom-free
arithmetic) exceeds X's demanded n2 = t - P: any proof that P(t) >= t - S1^2/M2
refutes X at y. Report sup_t R, its location, and the prime density that a
closing lower bound would need there, vs the actual density.
"""
import math
import sys


def scan(y):
    top = y * y
    k = y // 6 + 1
    while 6 * k - 1 <= y:
        k += 1
    ks, ke = k, (top - 2) // 6
    N = ke - ks + 1
    wl, wr = bytearray(N), bytearray(N)
    gs = bytearray([1]) * (y + 1)
    gs[:2] = b"\x00\x00"
    for i in range(2, int(y**0.5) + 1):
        if gs[i]:
            gs[i * i:: i] = bytearray(len(gs[i * i:: i]))
    for q in (q for q in range(5, y + 1) if gs[q]):
        m = (y // q + 1) * q
        while m < top:
            r = m % 6
            if r == 5:
                i = (m + 1) // 6 - ks
                if 0 <= i < N:
                    wl[i] += 1
            elif r == 1:
                i = (m - 1) // 6 - ks
                if 0 <= i < N:
                    wr[i] += 1
            m += q
    S1 = M2 = P = 0
    bestR, argt, first_gt1, last_gt1 = 0.0, None, None, None
    for i in range(N):
        a, b = wl[i], wr[i]
        P += (a == 0) + (b == 0)
        m = a * b
        if m:
            S1 += m
            M2 += m * m
        t = i + 1
        dem = t - P
        if dem > 0 and S1:
            R = S1 * S1 / M2 / dem
            if R > bestR:
                bestR, argt = R, t
            if R > 1:
                last_gt1 = t
                if first_gt1 is None:
                    first_gt1 = t
    t = argt
    # recompute quantities at argt for the density report
    S1 = M2 = P = 0
    for i in range(t):
        a, b = wl[i], wr[i]
        P += (a == 0) + (b == 0)
        m = a * b
        if m:
            S1 += m
            M2 += m * m
    need = t - S1 * S1 / M2          # closing bound must give P(t) > need
    mem = 6 * (ks + t - 1) + 1
    print(f"y={y:>5}: sup R = {bestR:.3f} at t={t} (member ~{mem}); "
          f"R>1 on t in [{first_gt1},{last_gt1}]" if first_gt1 else
          f"y={y:>5}: sup R = {bestR:.3f} at t={argt} (member ~{mem}); zone empty")
    print(f"        closing lower bound needs P({t}) > {need:.0f} "
          f"({need/t:.3f}/slot, {need/(6*t):.4f}/integer); actual P = {P} "
          f"({P/t:.3f}/slot); actual density 6/ln(m) = {6/math.log(mem):.3f}/slot")


if __name__ == "__main__":
    for y in [int(a) for a in sys.argv[1:]] or (101, 211, 503, 1009, 2003):
        scan(y)
