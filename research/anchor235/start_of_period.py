# how the run at the window start compares with runs elsewhere in the period, and where q^2/6 sits: 
# for gears <= q (q = 13..23), the blocked-run length distribution over all period positions vs the run at k = q^2/6, and at the section for every q up to 5000 using the sieve (run at q^2/6 in the <=q word = the actual twin gap covering q^2)
import sys; sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), ".."))
import numpy as np
from word_tree_r29 import spf_sieve
Q = 5000
spf = spf_sieve(Q * Q + 10).astype(np.int64)
primes = [int(x) for x in np.flatnonzero(spf_sieve(Q + 100) == np.arange(Q + 101)) if x >= 5]
K = (Q * Q) // 6 - 2
k = np.arange(1, K); lo, hi = 6 * k - 1, 6 * k + 1
tw = np.flatnonzero((spf[lo] == lo) & (spf[hi] == hi)) + 1   # twin slots k
print("run at the window start vs the section, q up to 5000: the blocked run of the <=q pattern covering slot q^2/6 = the actual twin-free stretch around q^2")
rows = []
for q, qn in zip(primes, primes[1:]):
    if qn > Q: break
    k0 = q * q // 6 + 1; W = (qn * qn - q * q) // 6
    i = np.searchsorted(tw, k0)
    prev, nxt_ = tw[i - 1], tw[i]
    run = nxt_ - prev - 1          # blocked slots between the last twin below q^2 and the first twin above
    rows.append((q, W, run, nxt_ - k0))
rows = np.array(rows)
q, W, run, first = rows.T
print(f"  sections: {len(rows)}; run covering q^2 exceeds W (no twin in the section from the start-run alone) at {(run >= W).sum()} rungs; max run/W {(run / W).max():.3f} at q={int(q[(run/W).argmax()])}")
print(f"  slots from q^2 to the first twin: median {np.median(first):.0f}, max {first.max()} at q={int(q[first.argmax()])}; as fraction of W: median {np.median(first/W):.3f}, max {(first/W).max():.3f}")
for a, b in [(5, 100), (100, 1000), (1000, 5000)]:
    m = (q >= a) & (q < b)
    print(f"  q in [{a},{b}): run covering q^2 median {np.median(run[m]):.0f} max {run[m].max()}, W median {np.median(W[m]):.0f}; first twin within W: {(first[m] < W[m]).mean():.3f}")
