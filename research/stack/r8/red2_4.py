"""Red lane, script 4: recompute square stretch counts (distinct filenames) + small-t bias control."""
import os

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
C2 = 0.6601618158468695739278121100145
LAW = 2 * C2
SQ_LIMIT = 10**5


def primes_upto(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for q in range(2, int(n**0.5) + 1):
        if s[q]:
            s[q * q :: q] = False
    return np.nonzero(s)[0].astype(np.int64)


n6 = np.load(os.path.join(OUT, "n6.npy")).astype(np.float64)
tarr = np.load(os.path.join(OUT, "t6.npy")).astype(np.int64)
ts = tarr[tarr <= SQ_LIMIT]
pr = primes_upto(SQ_LIMIT + 2)
pr = pr[pr >= 5]
T = np.zeros(len(ts), dtype=np.int64)
inv = {}
for idx, t in enumerate(ts.tolist()):
    lo, hi = (t - 1) ** 2, (t + 1) ** 2
    m0, m1 = lo // 6 + 1, (hi - 1) // 6
    L = m1 - m0 + 1
    ok = np.ones(L, dtype=bool)
    for p in pr[pr <= t + 1].tolist():
        i6 = inv.get(p)
        if i6 is None:
            i6 = pow(6, -1, p)
            inv[p] = i6
        for r in (i6, (-i6) % p):
            st = (r - m0) % p
            if st < L:
                ok[st::p] = False
    T[idx] = int(ok.sum())
np.save(os.path.join(OUT, "square_counts.npy"), T)
np.save(os.path.join(OUT, "square_tvals.npy"), ts)

r6 = n6[tarr <= SQ_LIMIT] / (2 * (ts + 6) * LAW / np.log(6.0 * ts) ** 2)
rS = T / (LAW * ts.astype(np.float64) / np.log(ts.astype(np.float64)) ** 2)
print("band-by-band mean ratio (law over-prediction control for task E)")
print(f"{'t band':>18} {'n':>5} {'mean W6':>9} {'mean sq':>9} {'min W6':>8} {'min sq':>8}")
for lo, hi in ((6, 200), (200, 1000), (1000, 10**4), (10**4, 10**5 + 1)):
    m = (ts >= lo) & (ts < hi)
    print(f"  [{lo:>6},{hi:>7}) {int(m.sum()):>5} {r6[m].mean():>9.4f} {rS[m].mean():>9.4f}"
          f" {r6[m].min():>8.4f} {rS[m].min():>8.4f}")
print()
print(f"argmin W6 on t<={SQ_LIMIT}: t={int(ts[int(np.argmin(r6))])} ratio={r6.min():.4f}")
print(f"argmin sq on t<={SQ_LIMIT}: t={int(ts[int(np.argmin(rS))])} ratio={rS.min():.4f}")
print(f"corr all: {float(np.corrcoef(r6, rS)[0,1]):.4f}   "
      f"corr t>=1000: {float(np.corrcoef(r6[ts>=1000], rS[ts>=1000])[0,1]):.4f}")
# probability that the two independent series share an argmin, given the small-t bias
print(f"rank of t=72 in W6: {int(np.argsort(np.argsort(r6))[list(ts).index(72)])+1} of {len(ts)}")
print(f"rank of t=72 in sq: {int(np.argsort(np.argsort(rS))[list(ts).index(72)])+1} of {len(ts)}")
print(f"T(72)={int(T[list(ts).index(72)])}, N6(72)={int(n6[tarr<=SQ_LIMIT][list(ts).index(72)])}")
print("10 worst W6:", [int(x) for x in ts[np.argsort(r6)[:10]]])
print("10 worst sq:", [int(x) for x in ts[np.argsort(rS)[:10]]])
