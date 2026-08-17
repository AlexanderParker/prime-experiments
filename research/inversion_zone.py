"""Inversion-zone tracker at scale (mechanic round 6).

Constructor's round-5 criterion: with m_k = omega_G(mL)*omega_G(mR),
S1(t) = sum m_k, M2(t) = sum m_k^2 over the first t slots, and P(t) the
prime-member count, Cauchy-Schwarz gives n2(t) >= S1^2/M2, and since
n2 = t - P + n0 identically,
    R(t) = (S1^2/M2) / (t - P) > 1   forces n0(t) >= 1 (a twin in the
prefix) by moment arithmetic alone. This script tracks sup_t R(t) per y.

Scan policy: full-window scans for y <= 100003 (proves no late zone);
prefix scans t <= T = min(W, 8y) above (zone extent measured <= 1.8y at
every full-scan y - margin noted in output). Slots with t - P <= 0 and
S1 > 0 also force a twin (denominator collapse); counted separately as
'inf' points, not in sup R.

Convention (rounds 1-5): a first-slot member <= y that is prime (the gear
y itself) counts omega = 0 and prime. O(1) effect at t <= 2.

Outputs (append): research/data/zone_summary.csv (per y: sup R, argmax,
zone extent, anatomy shares), zone_curves.csv (S1, M2, P, R at dense
checkpoints - the Constructor's requested curves), zone_anatomy.csv
(m-histogram + top-5 M2 slots at the argmax prefix).
Usage: uv run python research/inversion_zone.py [ytargets...]
Default targets: 20011 50021 100003 200003 500000 1000000 2000000
5000000 10000000 (snapped to next prime).
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fragile_census import primes_upto
from prefix_census import is_prime

FULL_LIMIT = 100003  # full-window scan at or below this y


def next_prime(n):
    n = n if n % 2 else n + 1
    while not is_prime(n):
        n += 2
    return n


def checkpoints_for(T):
    cps = {int(round(10 ** (i / 16))) for i in range(0, 400)}
    cps = {t for t in cps if 1 <= t <= T} | set(range(1, min(65, T + 1)))
    cps.add(T)
    return sorted(cps)


def scan(y, seg=8_000_000):
    gears = [q for q in primes_upto(y) if q >= 5]
    uvals = [pow(6, -1, q) for q in gears]
    k_lo = -((-(y - 1)) // 6)
    k_hi = (y * y + 1) // 6
    W = k_hi - k_lo + 1
    T = W if y <= FULL_LIMIT else min(W, 8 * y)
    cps = checkpoints_for(T)
    cS1 = cM2 = cP = 0
    supR, tstar = -1.0, None
    supB, tstarB = -1.0, None  # bulk sup, t >= 64 (boundary-convention-proof)
    star = None  # (S1, M2, P) at argmax
    zone_lo = zone_hi = None
    zone_n = 0
    n_inf = 0
    curves = []
    for a in range(k_lo, k_lo + T, seg):
        b = min(k_lo + T, a + seg)
        n = b - a
        cntL = np.zeros(n, np.int16)
        cntR = np.zeros(n, np.int16)
        for q, u in zip(gears, uvals):
            cntL[(u - a) % q::q] += 1
            cntR[(-u - a) % q::q] += 1
        if a == k_lo:
            for arr, m in ((cntL, 6 * k_lo - 1), (cntR, 6 * k_lo + 1)):
                if m <= y and is_prime(m):
                    arr[0] = 0
        m = cntL.astype(np.int64) * cntR
        pc = (cntL == 0).astype(np.int64) + (cntR == 0)
        s1 = np.cumsum(m) + cS1
        m2 = np.cumsum(m * m) + cM2
        pp = np.cumsum(pc) + cP
        tt = np.arange(a - k_lo + 1, b - k_lo + 1, dtype=np.int64)
        den = tt - pp
        valid = den > 0
        R = np.zeros(n)
        R[valid] = (s1[valid].astype(float) ** 2
                    / (m2[valid].astype(float) * den[valid]))
        n_inf += int(((~valid) & (s1 > 0)).sum())
        i = int(np.argmax(R)) if n else 0
        if n and R[i] > supR:
            supR, tstar = float(R[i]), int(tt[i])
            star = (int(s1[i]), int(m2[i]), int(pp[i]))
        bulk = tt >= 64
        if bulk.any():
            Rb = np.where(bulk, R, 0.0)
            ib = int(np.argmax(Rb))
            if Rb[ib] > supB:
                supB, tstarB = float(Rb[ib]), int(tt[ib])
        zi = np.flatnonzero(R > 1.0)
        if len(zi):
            if zone_lo is None:
                zone_lo = int(tt[zi[0]])
            zone_hi = int(tt[zi[-1]])
            zone_n += len(zi)
        for t in cps:
            if tt[0] <= t <= tt[-1]:
                j = t - int(tt[0])
                curves.append((t, int(s1[j]), int(m2[j]), int(pp[j]),
                               float(R[j]) if den[j] > 0 else float("inf")))
        cS1, cM2, cP = int(s1[-1]), int(m2[-1]), int(pp[-1])
    return dict(y=y, W=W, T=T, k_lo=k_lo, supR=supR, tstar=tstar, star=star,
                supB=supB, tstarB=tstarB,
                zone=(zone_lo, zone_hi, zone_n), n_inf=n_inf, curves=curves)


def anatomy(y, tstar):
    """second pass over the argmax prefix: m-histogram + top M2 slots."""
    gears = [q for q in primes_upto(y) if q >= 5]
    uvals = [pow(6, -1, q) for q in gears]
    k_lo = -((-(y - 1)) // 6)
    n = tstar
    cntL = np.zeros(n, np.int16)
    cntR = np.zeros(n, np.int16)
    for q, u in zip(gears, uvals):
        cntL[(u - k_lo) % q::q] += 1
        cntR[(-u - k_lo) % q::q] += 1
    for arr, m in ((cntL, 6 * k_lo - 1), (cntR, 6 * k_lo + 1)):
        if m <= y and is_prime(m):
            arr[0] = 0
    m = cntL.astype(np.int64) * cntR
    hist = np.bincount(m, minlength=1)
    top = np.argsort(m * m, kind="stable")[::-1][:5]
    tops = [(int(t0) + 1, 6 * (k_lo + int(t0)) - 1, int(cntL[t0]),
             int(cntR[t0]), int(m[t0])) for t0 in top if m[t0] > 0]
    return hist, tops


def main():
    targets = [int(a) for a in sys.argv[1:]] or [
        20011, 50021, 100003, 200003, 500000, 1000000, 2000000,
        5000000, 10000000]
    ys = sorted({next_prime(t) for t in targets})
    ddir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(ddir, exist_ok=True)

    def opencsv(name, header):
        path = os.path.join(ddir, name)
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        f = open(path, "a")
        if new:
            f.write(header + "\n")
        return f
    fsum = opencsv("zone_summary.csv",
                   "y,W,T_scanned,supR,t_star,supR_bulk64,t_star_bulk,"
                   "S1_star,M2_star,P_star,"
                   "zone_lo,zone_hi,zone_count,n_inf")
    fcur = opencsv("zone_curves.csv", "y,t,S1,M2,P,R")
    fana = opencsv("zone_anatomy.csv", "y,t_star,kind,detail")
    print(f"{'y':>9} {'T/W':>7} {'supR':>7} {'t*':>7} {'supB64':>7} "
          f"{'t*B':>8} {'zone':>17} {'#zone':>6} {'#inf':>5} {'sec':>6}")
    for y in ys:
        t0 = time.time()
        r = scan(y)
        zl, zh, zn = r["zone"]
        S1s, M2s, Ps = r["star"] if r["star"] else (0, 0, 0)
        fsum.write(f"{y},{r['W']},{r['T']},{r['supR']:.4f},{r['tstar']},"
                   f"{r['supB']:.4f},{r['tstarB']},"
                   f"{S1s},{M2s},{Ps},{zl},{zh},{zn},{r['n_inf']}\n")
        for t, s1, m2, pp, R in r["curves"]:
            fcur.write(f"{y},{t},{s1},{m2},{pp},{R:.5f}\n")
        if r["tstar"] and r["tstar"] <= 5_000_000:
            hist, tops = anatomy(y, r["tstar"])
            hs = ";".join(f"{mv}:{int(c)}" for mv, c in enumerate(hist) if c)
            fana.write(f"{y},{r['tstar']},m_hist,{hs}\n")
            for tt, mem, wl, wr, mv in tops:
                fana.write(f"{y},{r['tstar']},top_slot,"
                           f"t={tt} member={mem} wL={wl} wR={wr} m={mv}\n")
        print(f"{y:>9} {r['T']/r['W']:>7.3f} {r['supR']:>7.3f} "
              f"{r['tstar']:>7} {r['supB']:>7.3f} {r['tstarB']:>8} "
              f"{str((zl, zh)):>17} {zn:>6} {r['n_inf']:>5} "
              f"{time.time()-t0:>6.0f}")
        sys.stdout.flush()
    for f in (fsum, fcur, fana):
        f.close()
    print("wrote zone_summary.csv, zone_curves.csv, zone_anatomy.csv")


if __name__ == "__main__":
    main()
