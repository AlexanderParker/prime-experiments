"""Fragile census at scale (mechanic workstream).

Slot k = (6k-1, 6k+1). Gear q (prime, 5 <= q <= y) blocks slot k iff
k = +-u mod q, u = 6^{-1} mod q  (equivalently q divides a member).
Window of y = slots with a member in [y, y^2]:
    k_lo = ceil((y-1)/6),  k_hi = floor((y^2+1)/6).

Degree of a slot = total number of distinct gear divisors over both members.
A member with degree contribution 0 is prime and > y (any composite < y^2 has
lpf <= y, and a prime <= y is its own gear).

Classes counted per window:
  twin         both members degree-0 primes (matches the class-tree census:
               (11,13) at y=13 is NOT a twin here - its members are gears).
  frag_loose   one member a degree-0 prime, the other composite with EXACTLY
               ONE distinct gear divisor q (the owning gear). Any composite
               shape: q*p, q^2, q^2*p, q^3, ... (125 = 5^3 counts at y=13,
               giving the documented 10 fragile vs 9 twins).
  frag_semi    frag_loose AND the composite is a semiprime, i.e. q*p with p
               prime (> y) or p = q. Test: not divisible by q^2, or equal q^2.

Boundary note: a slot whose "composite" side is actually the gear y itself
(member == owning gear, e.g. (29,31) at y=29) is excluded from fragile - the
member is prime, not composite. Such slots are also not twins here (degree 1).

Also tallied: owning-gear decile (by rank in the gear list) for loose fragile;
lone-composite member counts S1 (composite member with exactly one gear
divisor, loose/semi, regardless of partner) and their Hardy-Littlewood weight
sum W1 = sum over those members of (q-1)/(q-2), q = owning gear. The sharp
candidate law tested: fragile ~ twins * W1 / pi_window.

Usage:  uv run python research/fragile_census.py [ymax_dense] [extra ys...]
Defaults: dense sweep over all primes 13..503, extras 1009 2003 3001 5003 10007.
"""
import sys
import time
import numpy as np
from math import log


def primes_upto(n):
    s = bytearray([1]) * (n + 1)
    s[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = bytearray(len(s[i * i::i]))
    return [i for i in range(n + 1) if s[i]]


def census(y, seg=4_000_000):
    gears = [q for q in primes_upto(y) if q >= 5]
    G = len(gears)
    garr = np.array(gears, dtype=np.int64)
    k_lo = -((-(y - 1)) // 6)
    k_hi = (y * y + 1) // 6
    y2 = y * y
    twins = 0
    frag_loose = 0
    frag_semi = 0
    primes_in = 0  # degree-0 prime members with value in (y, y^2]
    s1_loose = 0   # lone-composite members (one gear divisor), any shape
    s1_semi = 0    # ... semiprime shape
    w1_loose = 0.0  # sum of (q-1)/(q-2) over lone composites, loose
    w1_semi = 0.0
    dec_counts = np.zeros(10, dtype=np.int64)  # owning-gear decile, loose
    uvals = [pow(6, -1, q) for q in gears]
    u2vals = [pow(6, -1, q * q) for q in gears]
    for a in range(k_lo, k_hi + 1, seg):
        b = min(k_hi + 1, a + seg)
        n = b - a
        cntL = np.zeros(n, np.int16)
        cntR = np.zeros(n, np.int16)
        ownL = np.zeros(n, np.int32)
        ownR = np.zeros(n, np.int32)
        sqL = np.zeros(n, bool)
        sqR = np.zeros(n, bool)
        top_member = 6 * (b - 1) + 1
        for q, u, u2 in zip(gears, uvals, u2vals):
            s = (u - a) % q          # 6k-1 = 0 mod q
            cntL[s::q] += 1
            ownL[s::q] = q
            s = (-u - a) % q         # 6k+1 = 0 mod q
            cntR[s::q] += 1
            ownR[s::q] = q
            q2 = q * q
            if q2 <= top_member:
                sqL[(u2 - a) % q2::q2] = True
                sqR[(-u2 - a) % q2::q2] = True
        kk = np.arange(a, b, dtype=np.int64)
        mL = 6 * kk - 1
        mR = 6 * kk + 1
        pL = cntL == 0
        pR = cntR == 0
        twins += int((pL & pR).sum())
        primes_in += int((pL & (mL <= y2)).sum() + (pR & (mR <= y2)).sum())
        oL = ownL.astype(np.int64)
        oR = ownR.astype(np.int64)
        fragL = pR & (cntL == 1) & (mL != oL)   # composite on the L side
        fragR = pL & (cntR == 1) & (mR != oR)
        frag_loose += int(fragL.sum() + fragR.sum())
        semL = fragL & (~sqL | (mL == oL * oL))
        semR = fragR & (~sqR | (mR == oR * oR))
        frag_semi += int(semL.sum() + semR.sum())
        # lone-composite members (any partner), restricted to (y, y^2]
        for cnt, m, o, sq in ((cntL, mL, oL, sqL), (cntR, mR, oR, sqR)):
            lone = (cnt == 1) & (m != o) & (m > y) & (m <= y2)
            lsem = lone & (~sq | (m == o * o))
            s1_loose += int(lone.sum())
            s1_semi += int(lsem.sum())
            w = (o[lone] - 1.0) / (o[lone] - 2.0)
            w1_loose += float(w.sum())
            w = (o[lsem] - 1.0) / (o[lsem] - 2.0)
            w1_semi += float(w.sum())
        for frag, own in ((fragL, ownL), (fragR, ownR)):
            qs = own[frag].astype(np.int64)
            if len(qs):
                idx = np.searchsorted(garr, qs)
                dec = (10 * idx) // G
                dec_counts += np.bincount(dec, minlength=10)[:10]
    W = k_hi - k_lo + 1
    return dict(y=y, W=W, twins=twins, frag_semi=frag_semi,
                frag_loose=frag_loose, primes_in=primes_in,
                s1_loose=s1_loose, s1_semi=s1_semi,
                w1_loose=w1_loose, w1_semi=w1_semi,
                dec=dec_counts, G=G)


def main():
    ymax_dense = int(sys.argv[1]) if len(sys.argv) > 1 else 503
    extras = [int(v) for v in sys.argv[2:]] or [1009, 2003, 3001, 5003, 10007]
    ys = [p for p in primes_upto(ymax_dense) if p >= 13] + \
         [v for v in extras if v > ymax_dense]
    rows = []
    print(f"{'y':>6} {'W':>10} {'twins':>9} {'fragS':>9} {'fragL':>9} "
          f"{'S/tw':>6} {'L/tw':>6} {'pi_win':>10} {'S1semi':>9} "
          f"{'S1loose':>9} {'sec':>6}")
    for y in ys:
        t0 = time.time()
        r = census(y)
        dt = time.time() - t0
        rows.append(r)
        tw = r["twins"]
        print(f"{y:>6} {r['W']:>10} {tw:>9} {r['frag_semi']:>9} "
              f"{r['frag_loose']:>9} {r['frag_semi']/tw:>6.3f} "
              f"{r['frag_loose']/tw:>6.3f} {r['primes_in']:>10} "
              f"{r['s1_semi']:>9} {r['s1_loose']:>9} {dt:>6.2f}")
        sys.stdout.flush()

    # ---- law-candidate comparison (fits, not laws) ----
    print("\nCandidate normalisations (constant column = good law):")
    print(f"{'y':>6} {'L/tw':>7} {'L/(W/ln^3 y2)':>14} {'L/(pi/ln y2)':>13} "
          f"{'L/(tw*lnln y2)':>15} {'cS=S*pi/twW1s':>14} {'cL=L*pi/twW1l':>14}")
    for r in rows:
        y, W, tw, fl = r["y"], r["W"], r["twins"], r["frag_loose"]
        fs, pi_w = r["frag_semi"], r["primes_in"]
        L2 = log(y * y)
        pi_y2 = pi_w + len(primes_upto(y))  # ~pi(y^2)
        cS = fs * pi_w / (tw * r["w1_semi"])
        cL = fl * pi_w / (tw * r["w1_loose"])
        print(f"{y:>6} {fl/tw:>7.3f} {fl/(W/L2**3):>14.4f} "
              f"{fl/(pi_y2/L2):>13.4f} {fl/(tw*log(L2)):>15.4f} "
              f"{cS:>14.4f} {cL:>14.4f}")

    print("\nOwning-gear decile of loose-fragile slots (decile by gear rank):")
    print(f"{'y':>6} " + " ".join(f"d{i}" for i in range(10)))
    for r in rows:
        if r["y"] in (13, 53, 101, 251, 503, 1009, 2003, 5003, 10007,
                      20011, 50021):
            tot = r["dec"].sum()
            pct = ["%4.1f" % (100 * c / tot) for c in r["dec"]]
            print(f"{r['y']:>6} " + " ".join(pct))


if __name__ == "__main__":
    main()
