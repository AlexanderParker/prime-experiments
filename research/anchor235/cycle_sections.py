import sys; sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), ".."))
import numpy as np, bisect
from word_tree_r29 import spf_sieve
X = 100_000_000
spf = spf_sieve(X + 40); isp = spf == np.arange(len(spf)); isp[:2] = False
J = X // 30; j = np.arange(J); base = 30 * j
full = np.flatnonzero(np.all(np.stack([isp[base + r] for r in (11, 13, 17, 19, 29, 31)], 1), 1))
P = [int(p) for p in np.flatnonzero(isp[:10100]) if p >= 5]
n0 = 30 * full + 11
sec = [bisect.bisect_right(P, int(np.sqrt(n))) - 1 for n in n0]   # index of q with q^2 <= n < q'^2
qs = np.array([P[s] for s in sec]); qn = np.array([P[s + 1] for s in sec])
frac = (n0 - qs * qs) / (qn * qn - qs * qs)
print("open cycles by section (q^2, q'^2), to 1e8:")
cnt = {}
for s in sec: cnt[s] = cnt.get(s, 0) + 1
nsec = bisect.bisect_right(P, 10000) - 1
per = np.zeros(nsec, int)
for s, c in cnt.items(): per[s] = c
print(f"  sections: {nsec}; with 0 open cycles: {(per==0).sum()}, 1: {(per==1).sum()}, 2: {(per==2).sum()}, 3+: {(per>=3).sum()}")
for a, b in [(5, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 10000)]:
    idx = [i for i in range(nsec) if a <= P[i] < b]
    print(f"  q in [{a},{b}): {len(idx)} sections, {sum(per[i] > 0 for i in idx)} hold an open cycle ({100*sum(per[i]>0 for i in idx)/len(idx):.0f}%), open cycles {sum(per[i] for i in idx)}")
print(f"  position inside section (0 = at q^2, 1 = at q'^2): mean {frac.mean():.2f}; quartiles {np.percentile(frac,[25,50,75]).round(2).tolist()}; nearest to q^2: {frac.min():.3f}, to q'^2: {frac.max():.3f}")
# longest stretch of consecutive sections with none
runs, r = [], 0
for i in range(nsec):
    r = r + 1 if per[i] == 0 else 0
    runs.append(r)
i = int(np.argmax(runs)); L = runs[i]
print(f"  longest run of sections with no open cycle: {L} sections, q from {P[i-L+1]} to {P[i]} (numbers {P[i-L+1]**2}..{P[i+1]**2})")
print("  first 20 open cycles: (q, cycle j, number 30j+11):", [(int(a), int(b), int(c)) for a, b, c in zip(qs[:20], full[:20], n0[:20])])
