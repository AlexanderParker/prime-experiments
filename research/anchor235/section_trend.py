import sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), ".."))
import numpy as np
from word_tree_r29 import spf_sieve
QMAX = 5000
spf = spf_sieve(QMAX * QMAX + 10).astype(np.int64)
primes = [int(x) for x in np.flatnonzero(spf_sieve(QMAX + 100) == np.arange(QMAX + 101)) if x >= 5]
rows = []
for i in range(len(primes) - 1):
    q, qn = primes[i], primes[i + 1]
    if qn > QMAX: break
    k_lo, k_hi = q * q // 6 + 1, (qn * qn - 2) // 6
    ks = np.arange(k_lo, k_hi + 1)
    lo, hi = 6 * ks - 1, 6 * ks + 1
    ao = np.isin(ks % 5, (0, 2, 3))
    al = ((spf[lo] == lo) & (spf[hi] == hi))[ao]
    kk = ks[ao]
    # longest blocked run in numbers: gap between consecutive aligned slots (and section edges) minus one slot
    pos = np.flatnonzero(al)
    if len(pos) == 0:
        run = 6 * (kk[-1] - kk[0]) + 2
    else:
        edges = np.concatenate([[-1], pos, [len(kk)]])
        gaps = np.diff(edges) - 1  # blocked anchor-open slots between aligned ones
        j = gaps.argmax()
        a, b = edges[j] + 1, edges[j + 1] - 1
        run = int(6 * (kk[b] - kk[a]) + 2) if gaps[j] > 0 else 0
    rows.append((q, qn, 6 * len(ks), int(al.sum()), run))
rows = np.array(rows, dtype=float)
q, W, A, R = rows[:, 0], rows[:, 2], rows[:, 3], rows[:, 4]
print("per-section trend (each section = numbers strictly between q^2 and q'^2), binned by q:")
print("  q range        sections  aligned/section min..max   blocked run (numbers) median / max   run/window median / max")
for a, b in [(5, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000)]:
    m = (q >= a) & (q < b)
    print(f"  {a:>5}-{b:<5}    {int(m.sum()):>6}   {int(A[m].min()):>5}..{int(A[m].max()):<5}         {np.median(R[m]):>7.0f} / {R[m].max():<7.0f}       {np.median(R[m]/W[m]):.3f} / {(R[m]/W[m]).max():.3f}")
m = q >= 100
c = np.polyfit(np.log(q[m]), np.log(R[m]), 1)
print(f"fit of longest blocked run vs q on sections q >= 100: run ~ q^{c[0]:.2f}; window >= 4q+4 always (twin-prime rungs), typically ~ 2q ln q")
# run against ln^2(q^2) scale
print(f"blocked run / ln^2(6k) at the section, median by bin: " + ", ".join(f"{a}-{b}: {np.median(R[(q>=a)&(q<b)] / np.log(q[(q>=a)&(q<b)]**2)**2):.2f}" for a, b in [(100,200),(500,1000),(1000,2000),(2000,3000),(4000,5000)]))
z = A == 0
print(f"sections with zero aligned slots: {int(z.sum())}")
