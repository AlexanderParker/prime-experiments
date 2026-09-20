"""The twin stretch in the products field. For twin centre s (stretch = columns with members in
((s-1)^2, (s+1)^2)), base level B = floor(sqrt(2s)): a column is base-open if both members are
B-rough. Exact split: base-open = twins + columns with a member that is a product of primes > B
(P_2 or P_3 with all factors > B). Report per s: base-open, twins, plugged by products, the member
types (P_2 count, P_3 count), and the share plugged. Also the top band's painting: products h p with
h in (s/2, s] - how many columns they paint and how many of those are base-open.
Pre-registered: plugged/base-open in [0.6, 0.9] (the Chen-type share), rising slowly with s; every
plug member is P_2 or P_3 (no P_4 below (s+1)^2 with factors > sqrt(2s)); top-band products paint
about 1.8 s / ln s columns of which a fraction about 3.6 / ln s are base-open.
"""
import numpy as np, sys
S_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
N = (S_MAX + 1) ** 2 + 2
spf = np.zeros(N, dtype=np.int32)
for p in range(2, int(N ** 0.5) + 1):
    if spf[p] == 0:
        blk = spf[p*p::p]; blk[blk == 0] = p; spf[p*p::p] = blk
idx = np.nonzero(spf == 0)[0]; spf[idx] = idx; spf[0] = spf[1] = 1
def is_prime(n): return n >= 2 and spf[n] == n
def factors(n):
    out = []
    while n > 1:
        p = int(spf[n]); out.append(p); n //= p
    return out
tw = [s for s in range(102, S_MAX + 1, 6) if is_prime(s - 1) and is_prime(s + 1)]
print(" s     B  cols  base-open  twins  plugged  share  P2-members P3-members  topband-cols topband-baseopen")
for s in tw[::max(1, len(tw)//12)]:
    c = s // 6; B = int((2 * s) ** 0.5)
    js = np.arange(-(2 * c - 1), 2 * c)
    lo = s * s + 6 * js - 1; hi = lo + 2
    rough = (spf[lo] > B) & (spf[hi] > B)
    twin = (spf[lo] == lo) & (spf[hi] == hi)
    plug = rough & ~twin
    p2 = p3 = 0
    for a, b in zip(lo[plug], hi[plug]):
        for m in (int(a), int(b)):
            f = factors(m)
            if len(f) == 2: p2 += 1
            elif len(f) == 3: p3 += 1
            elif len(f) > 3: raise SystemExit(f"P4 member {m} at s={s}")
    # top band: members with a prime factor h in (s/2, s]
    top = np.zeros(len(js), dtype=bool)
    for k, (a, b) in enumerate(zip(lo, hi)):
        for m in (int(a), int(b)):
            f = factors(m)
            if any(s / 2 < h <= s for h in f): top[k] = True; break
    print(f"{s:5d} {B:4d} {len(js):5d} {int(rough.sum()):9d} {int(twin.sum()):6d} {int(plug.sum()):8d} "
          f"{plug.sum()/rough.sum():.3f} {p2:10d} {p3:10d} {int(top.sum()):12d} {int((top & rough).sum()):16d}")
