"""Second batch of exact measurements for ATTEMPTS.md (under 2 minutes).
(a) Ladder window, all twin centres s <= 4000: |R_61|, T_61, X_61, |R_mid| (rough to s/3),
    T_big(R_mid), X_big(R_mid), r(s); min margins of the candidate GAP inequalities.
(b) Dichotomy data: min r(s) over s >= 1000, s <= 1e4.
(c) Six window (5(t-1), 7(t+1)): largest twin centre t <= 1e7 with no twin centre of the form
    6mP +- 1 (P = 1, 5, 35, 385) inside.
"""
import numpy as np, math, time
t0 = time.time()
LIM = 7 * 10**7 + 100
sieve = np.ones(LIM + 1, dtype=bool); sieve[:2] = False
for i in range(2, int(LIM**0.5) + 1):
    if sieve[i]:
        sieve[i*i::i] = False
ks = np.arange(1, LIM // 6)
twin_centres = 6 * ks[sieve[6*ks - 1] & sieve[6*ks + 1]]
tcset = set(twin_centres.tolist())
primes_all = np.nonzero(sieve[:20000])[0]

def window_stats(s):
    c = s // 6; N = 4*c - 1
    js = np.arange(-2*c + 1, 2*c)
    lower = s*s + 6*js - 1; upper = s*s + 6*js + 1
    gears = primes_all[(primes_all >= 5) & (primes_all <= s - 5)]
    nz = js != 0
    struck = {}
    for p in gears:
        struck[int(p)] = ((lower % p == 0) | (upper % p == 0)) & nz
    R61 = nz.copy()
    for p in gears[gears <= 61]: R61 &= ~struck[int(p)]
    Rmid = R61.copy()
    for p in gears[(gears > 61) & (gears <= s // 3)]: Rmid &= ~struck[int(p)]
    om61 = np.zeros(N, dtype=np.int64); ombig = np.zeros(N, dtype=np.int64)
    for p in gears[gears > 61]:
        om61 += struck[int(p)] & R61
        if p > s // 3: ombig += struck[int(p)] & Rmid
    r = int(((om61 == 0) & R61).sum())
    T61 = int(om61[R61].sum()); X61 = int(np.maximum(om61[R61] - 1, 0).sum())
    Tb = int(ombig[Rmid].sum()); Xb = int(np.maximum(ombig[Rmid] - 1, 0).sum())
    return dict(s=s, N=N, R61=int(R61.sum()), T61=T61, X61=X61, Rmid=int(Rmid.sum()), Tbig=Tb, Xbig=Xb, r=r)

rows = [window_stats(int(s)) for s in twin_centres[(twin_centres >= 60) & (twin_centres <= 10000)]]
print("windows measured:", len(rows), " time", round(time.time()-t0,1))
# GAP A-1 identity check and margins
bad_id = [r for r in rows if r['r'] != r['R61'] - r['T61'] + r['X61']]
print("identity r = |R61| - T61 + X61 violated at:", [r['s'] for r in bad_id])
m1 = min((r['X61'] - (r['T61'] - r['R61'] + 1), r['s']) for r in rows)
print("GAP A-1 margin X61 - (T61 - |R61| + 1): min", m1)
m2 = [(r['Rmid'] - r['Tbig'], r['s']) for r in rows]
print("GAP A-3' margin |Rmid| - Tbig(Rmid): min", min(m2), " failures (<=0):", [s for v, s in m2 if v <= 0])
print("sample rows:")
for r in rows[:6] + rows[-4:]:
    print("  ", r)
print("s=10008 stats:", window_stats(10008))

# (b) dichotomy data
def r_of(s):
    lo, hi = (s-1)**2, (s+1)**2
    return int(np.searchsorted(twin_centres, hi) - np.searchsorted(twin_centres, lo, side='right'))
rr = [(r_of(int(s)), int(s)) for s in twin_centres[(twin_centres >= 1000) & (twin_centres <= 8000)]]
print("min r(s) for 1000 <= s <= 8000 (stretch <= 6.4e7):", min(rr), " count", len(rr))

# (c) Six window with primorial-multiple centres
tw_small = twin_centres[twin_centres <= 10**7]
for P in (1, 5, 35, 385, 5005):
    M = 6 * P
    cand = twin_centres[twin_centres % M == 0]
    lo = 5 * (tw_small - 1); hi = 7 * (tw_small + 1)
    cnt = np.searchsorted(cand, hi) - np.searchsorted(cand, lo, side='right')
    fails = tw_small[cnt == 0]
    print(f"Six window, centres = multiples of {M}: failing t count {len(fails)}, largest failing t = {int(fails.max()) if len(fails) else None}, min count for t>largest fail = {int(cnt[tw_small > (fails.max() if len(fails) else 0)].min()) if (cnt[tw_small > (fails.max() if len(fails) else 0)]).size else None}")
print("total", round(time.time()-t0,1), "s")
