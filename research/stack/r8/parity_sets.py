"""Task 1: Omega(6k-1), Omega(6k+1) for 6k+1 <= 10^7; sets P, Q, M' (and Q61, M'61); count laws.
Writes lane_parity/sieve.npz for task 2."""
import math
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N = 10**7


def spf_sieve(n):
    spf = np.zeros(n + 1, dtype=np.int32)
    spf[2::2] = 2
    lim = int(math.isqrt(n))
    for p in range(3, lim + 1, 2):
        if spf[p] == 0:
            block = spf[p * p:: 2 * p]
            mask = block == 0
            block[mask] = p
    rest = spf == 0
    rest[:2] = False
    spf[rest] = np.arange(n + 1, dtype=np.int32)[rest]
    return spf


def omega_from_spf(spf):
    n = spf.shape[0] - 1
    m = np.arange(n + 1, dtype=np.int64)
    m[:2] = 1
    om = np.zeros(n + 1, dtype=np.int8)
    passes = 0
    while True:
        alive = m > 1
        if not alive.any():
            break
        om[alive] += 1
        m[alive] //= spf[m[alive]]
        passes += 1
    return om, passes


def fit_table(name, flags, k):
    """count of set members with 6k+1 <= x, and C = count ln^2 x / x."""
    rows = []
    for e in (4, 4.5, 5, 5.5, 6, 6.5, 7):
        x = int(round(10**e))
        kmax = (x - 1) // 6
        cnt = int(flags[: kmax].sum())
        C = cnt * math.log(x) ** 2 / x
        rows.append((x, cnt, C, cnt / kmax))
    print(f"\n{name}: count law against C x/ln^2 x (and density among columns)")
    print(f"{'x':>10} {'count':>9} {'C = cnt ln^2x/x':>16} {'cnt/columns':>12}")
    for x, cnt, C, d in rows:
        print(f"{x:>10} {cnt:>9} {C:>16.4f} {d:>12.5f}")


def main():
    t0 = time.time()
    spf = spf_sieve(N)
    print(f"spf sieve to {N}: {time.time()-t0:.1f}s")
    om, passes = omega_from_spf(spf)
    print(f"Omega by repeated division: {passes} passes, {time.time()-t0:.1f}s")

    K = (N - 1) // 6  # 6K+1 <= N
    k = np.arange(1, K + 1, dtype=np.int64)
    a = 6 * k - 1
    b = 6 * k + 1
    oa = om[a].astype(np.int16)
    ob = om[b].astype(np.int16)
    pa = spf[a] == a
    pb = spf[b] == b
    P = pa & pb
    Q = (oa % 2 == 0) & (ob % 2 == 0)  # even Omega => composite automatically
    Mp = (oa % 2 == 1) & (ob % 2 == 1) & ~pa & ~pb
    rough61 = (spf[a] > 61) & (spf[b] > 61)
    Q61 = Q & rough61
    Mp61 = Mp & rough61
    M = (oa % 2 == 1) & (ob % 2 == 1)

    print(f"\ncolumns K = {K}  (6k+1 <= {N})")
    for name, f in (("P (twins)", P), ("Q (both Omega even)", Q), ("M' (both Omega odd, neither prime)", Mp),
                    ("M (both Omega odd, incl. twins)", M), ("61-rough columns", rough61),
                    ("Q61 = Q & 61-rough", Q61), ("M'61 = M' & 61-rough", Mp61)):
        print(f"  |{name}| = {int(f.sum())}")
    # check Q members composite
    assert not (Q & (pa | pb)).any()
    # Omega parity distribution of the two members, independence check
    ea, eb = (oa % 2 == 0), (ob % 2 == 0)
    print(f"\nP(Omega(6k-1) even) = {ea.mean():.5f}, P(Omega(6k+1) even) = {eb.mean():.5f}, "
          f"P(both even) = {(ea & eb).mean():.5f}, product = {ea.mean()*eb.mean():.5f}")

    fit_table("P", P, k)
    fit_table("Q", Q, k)
    fit_table("M'", Mp, k)
    fit_table("Q61", Q61, k)
    fit_table("M'61", Mp61, k)

    np.savez(os.path.join(HERE, "sieve.npz"), spf=spf, om=om, P=P, Q=Q, Mp=Mp, Q61=Q61, Mp61=Mp61)
    print(f"\nsaved sieve.npz, total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
