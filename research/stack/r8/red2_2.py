"""Red lane, script 2: task C (margins vs law), task E (worst t), and the K / residue fixes."""
import math
import os
import sys

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
C2 = 0.6601618158468695739278121100145
LAW = 2 * C2  # 1.3203236...

SQ_LIMIT = 10**5  # square stretch computed exactly for twin centres t <= SQ_LIMIT


def primes_upto(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for q in range(2, int(n**0.5) + 1):
        if s[q]:
            s[q * q :: q] = False
    return np.nonzero(s)[0].astype(np.int64)


def main():
    twins = np.load(os.path.join(OUT, "twins.npy"))
    n6 = np.load(os.path.join(OUT, "n6.npy")).astype(np.float64)
    tarr = np.load(os.path.join(OUT, "t6.npy")).astype(np.int64)

    print("=" * 74)
    print("FIX: K = prod_{p>=5} r_p,  r_p = p(p^2-6p+10)/(p-2)^3")
    ps = primes_upto(5_000_000)
    ps = ps[ps >= 5].astype(np.float64)
    rp = ps * (ps * ps - 6 * ps + 10) / (ps - 2) ** 3
    logs = np.log(rp)
    for P in (7, 13, 47, 100, 1000, 10**4, 10**5, 10**6, 5 * 10**6):
        val = math.exp(float(logs[ps <= P].sum()))
        print(f"  prod r_p, p <= {P:>9} : {val:.6f}")
    X = 5_000_000
    tail = math.exp(-2.0 / (X * math.log(X)))  # sum_{p>X} 2/p^2 ~ 2/(X ln X)
    K = math.exp(float(logs.sum())) * tail
    print(f"  K (converged)                : {K:.6f}     (lane reported 0.8243)")

    print("=" * 74)
    print("FIX: is the uniform-residue model right? distribution of twin centres mod p")
    for p in (5, 7, 11, 13):
        res = twins % p
        cnt = np.bincount(res, minlength=p)
        adm = [a for a in range(p) if a not in (1, p - 1)]
        tot = int(cnt[adm].sum())
        print(f"  p={p:<3} N={len(twins)}  counts on admissible residues "
              f"{{{', '.join(f'{a}:{int(cnt[a])}' for a in adm)}}}  "
              f"expected {tot/len(adm):.0f} each; hits on +-1: {int(cnt[1])+int(cnt[p-1])}")
        # r_p recomputed from the empirical residue distribution of twin centres
        w = cnt[adm].astype(np.float64) / tot
        good = 0.0
        for i, a in enumerate(adm):
            for j, b in enumerate(adm):
                if (a * b) % p not in (1, p - 1):
                    good += w[i] * w[j]
        print(f"        r_{p} model={p*((p-2)**2-2*(p-3))/(p-2)**3:.6f}  "
              f"r_{p} from empirical residues={good/(1-2/p):.6f}")

    # ---------------- Task C, part 1: N_6(t) against its law
    print("=" * 74)
    print("TASK C.1: N_6(t) = #twin centres in (5(t-1), 7(t+1)) vs law 2(t+6)*1.3203/ln^2(6t)")
    law6 = 2 * (tarr + 6) * LAW / np.log(6.0 * tarr) ** 2
    rat6 = n6 / law6
    for lo in (6, 30, 100, 1000, 10**4):
        m = tarr >= lo
        r = rat6[m]
        t = tarr[m]
        o = np.argsort(r)[:5]
        print(f"  t >= {lo:<6} n={m.sum():>5}  min={r.min():.4f} at t={int(t[int(np.argmin(r))])}  "
              f"mean={r.mean():.4f}  median={np.median(r):.4f}  max={r.max():.4f}")
        print(f"        5 smallest: " + "  ".join(
            f"t={int(t[int(i)])}({r[int(i)]:.4f},N={int(n6[m][int(i)])})" for i in o))

    # ---------------- Task C, part 2: square stretch
    print("=" * 74)
    print(f"TASK C.2: T(t) = #twin centres in ((t-1)^2,(t+1)^2) vs law 1.3203*t/ln^2(t)")
    print(f"  computed exactly for twin centres t <= {SQ_LIMIT} (segmented sieve; "
          f"(t+1)^2 up to {(SQ_LIMIT+1)**2})")
    ts = tarr[tarr <= SQ_LIMIT]
    pr = primes_upto(SQ_LIMIT + 2)
    pr = pr[pr >= 5]
    T = np.zeros(len(ts), dtype=np.int64)
    inv6_cache = {}
    for idx, t in enumerate(ts):
        t = int(t)
        lo = (t - 1) ** 2
        hi = (t + 1) ** 2
        m0 = lo // 6 + 1
        m1 = (hi - 1) // 6
        L = m1 - m0 + 1
        ok = np.ones(L, dtype=bool)
        sub = pr[pr <= t + 1]
        for p in sub.tolist():
            i6 = inv6_cache.get(p)
            if i6 is None:
                i6 = pow(6, -1, p)
                inv6_cache[p] = i6
            for r in (i6, (-i6) % p):
                st = (r - m0) % p
                if st < L:
                    ok[st::p] = False
        T[idx] = int(ok.sum())
        if idx % 300 == 0:
            print(f"    ... t={t} T={T[idx]}", file=sys.stderr)
    np.save(os.path.join(OUT, "sqT.npy"), T)
    np.save(os.path.join(OUT, "sqt.npy"), ts)
    lawS = LAW * ts.astype(np.float64) / np.log(ts.astype(np.float64)) ** 2
    ratS = T / lawS
    print(f"  empty square stretches: {int((T == 0).sum())}   min T={int(T.min())} at t={int(ts[int(np.argmin(T))])}")
    for lo in (6, 30, 100, 1000, 10**4):
        m = ts >= lo
        r = ratS[m]
        t = ts[m]
        o = np.argsort(r)[:5]
        print(f"  t >= {lo:<6} n={int(m.sum()):>5}  min={r.min():.4f} at t={int(t[int(np.argmin(r))])}  "
              f"mean={r.mean():.4f}  median={np.median(r):.4f}  max={r.max():.4f}")
        print(f"        5 smallest: " + "  ".join(
            f"t={int(t[int(i)])}({r[int(i)]:.4f},T={int(T[m][int(i)])})" for i in o))

    # head of the square table
    print("  first 12 square stretches:")
    for i in range(12):
        print(f"    t={int(ts[i]):>6}  T={int(T[i]):>5}  law={lawS[i]:8.2f}  ratio={ratS[i]:.4f}")

    # ---------------- Task E
    print("=" * 74)
    print("TASK E: worst t for W(6,t) vs worst t for the square stretch")
    i6 = int(np.argmin(rat6))
    print(f"  furthest below law, W(6,t), all t <= 10^6 : t={int(tarr[i6])}  "
          f"N_6={int(n6[i6])}  law={law6[i6]:.2f}  ratio={rat6[i6]:.4f}")
    m = tarr >= 1000
    sub_t = tarr[m]
    sub_r = rat6[m]
    j = int(np.argmin(sub_r))
    print(f"  furthest below law, W(6,t), t >= 1000     : t={int(sub_t[j])}  "
          f"N_6={int(n6[m][j])}  ratio={sub_r[j]:.4f}")
    iS = int(np.argmin(ratS))
    print(f"  furthest below law, square, t <= {SQ_LIMIT}    : t={int(ts[iS])}  "
          f"T={int(T[iS])}  law={lawS[iS]:.2f}  ratio={ratS[iS]:.4f}")
    # head-to-head on the common range
    mc = tarr <= SQ_LIMIT
    assert np.array_equal(tarr[mc], ts)
    r6c = rat6[mc]
    ic = int(np.argmin(r6c))
    print(f"  common range t <= {SQ_LIMIT}: W(6,.) min ratio at t={int(ts[ic])} ({r6c[ic]:.4f}); "
          f"square min ratio at t={int(ts[iS])} ({ratS[iS]:.4f}) -> same t: {ic == iS}")
    corr = float(np.corrcoef(r6c, ratS)[0, 1])
    print(f"  Pearson corr(ratio_W6, ratio_square) on t <= {SQ_LIMIT}: {corr:.4f}")
    mm = ts >= 1000
    print(f"  same, t in [1000,{SQ_LIMIT}]: {float(np.corrcoef(r6c[mm], ratS[mm])[0,1]):.4f}")
    # rank agreement of the 10 worst
    w6 = set(int(x) for x in ts[np.argsort(r6c)[:10]])
    wS = set(int(x) for x in ts[np.argsort(ratS)[:10]])
    print(f"  10 worst W(6,.) t: {sorted(w6)}")
    print(f"  10 worst square t: {sorted(wS)}")
    print(f"  overlap: {sorted(w6 & wS)}")
    print("=" * 74)
    print("done")


if __name__ == "__main__":
    main()
