"""Small exact measurements used in ATTEMPTS.md (under 2 minutes).
1. Sieve to 1e8; twin centres s <= 1e4; r(s) = twin centres in the stretch; min r(s).
2. Orphans: twin centres t <= 1e8 with / without a twin-centre parent; longest orphan run.
3. For s = 10008 (twin centre): exact |R_B| for B = 13, 19, 61; strike counts of gears in bands;
   omega(j) = number of gears in (61, s-5] striking a 61-rough column; T = sum; twins in window.
"""
import numpy as np, math, sys, time
t0 = time.time()
LIM = 10**8
sieve = np.ones(LIM + 1, dtype=bool); sieve[:2] = False
for i in range(2, int(LIM**0.5) + 1):
    if sieve[i]:
        sieve[i*i::i] = False
print("sieve done", round(time.time()-t0,1), "s")

# twin centres t = 6k with t-1, t+1 prime
ks = np.arange(1, LIM // 6)
tc_mask = sieve[6*ks - 1] & sieve[6*ks + 1]
twin_centres = 6 * ks[tc_mask]
print("twin centres <= 1e8:", len(twin_centres))

# 1. rungs for s <= 1e4
small = twin_centres[twin_centres <= 10**4]
rs = []
for s in small:
    lo, hi = (s-1)**2, (s+1)**2
    n = np.searchsorted(twin_centres, hi) - np.searchsorted(twin_centres, lo, side='right')
    rs.append(int(n))
rs = np.array(rs)
print("twin centres s <= 1e4:", len(small), " min r(s) =", rs.min(), " at s =", small[rs.argmin()],
      " r(s)=1 count:", int((rs == 1).sum()), " r(s)<=2 count:", int((rs <= 2).sum()))
print("first 12 (s, r):", list(zip(small[:12].tolist(), rs[:12].tolist())))
print("last 5 (s, r):", list(zip(small[-5:].tolist(), rs[-5:].tolist())))
print("s, r(s) for s in 1000..1200:", [(int(s), int(r)) for s, r in zip(small, rs) if 1000 <= s <= 1200])

# 2. orphans: parent candidate = multiple of 6 within distance 1 of sqrt(t)
sq = np.sqrt(twin_centres.astype(np.float64))
cand = 6 * np.round(sq / 6).astype(np.int64)
has_cand = np.abs(sq - cand) < 1
tcset = set(twin_centres.tolist())
has_parent = np.array([bool(hc) and (int(cd) in tcset) for hc, cd in zip(has_cand, cand)])
print("with a candidate parent:", int(has_cand.sum()), " with a twin parent:", int(has_parent.sum()))
# longest run of consecutive orphans (no twin parent)
best = cur = 0
for hp in has_parent:
    cur = 0 if hp else cur + 1
    best = max(best, cur)
print("longest run of consecutive orphans:", best)
# longest run of consecutive twin centres each having r(s) >= 1 is all of them (no leaves) -- checked above

# 3. one window in detail
s = 10008; c = s // 6
assert s in tcset
N = 4*c - 1
js = np.arange(-2*c + 1, 2*c)            # offsets
lower = s*s + 6*js - 1; upper = s*s + 6*js + 1
primes = np.nonzero(sieve[:s])[0]; gears = primes[primes >= 5]
gears = gears[gears <= s - 5]
print("s =", s, "c =", c, "N =", N, "gears 5..s-5:", len(gears))
def struck_by(p):
    return (lower % p == 0) | (upper % p == 0)
S = {int(p): struck_by(p) for p in gears}
for B in (13, 19, 61):
    m = np.zeros(N, dtype=bool)
    for p in gears[gears <= B]:
        m |= S[int(p)]
    R = ~m
    print(f"B={B}: |R_B| = {int(R.sum())}  (N*prod(1-2/p) = {N*math.prod(1-2/p for p in gears[gears<=B]):.1f})")
    if B == 61:
        R61 = R
# twins in window
tw = ~np.zeros(N, dtype=bool)
for p in gears:
    tw &= ~S[int(p)]
tw[js == 0] = False   # centre column is (s-1)(s+1), struck by twin gears only
print("twin columns in window (r(s)):", int(tw.sum()), " offsets:", js[tw].tolist())
# strikes on R61 by gears > 61, band ledger
omega = np.zeros(N, dtype=np.int64)
bands = [(61, int(math.isqrt(2*s))), (int(math.isqrt(2*s)), s//3), (s//3, s//2), (s//2, 2*s//3), (2*s//3, s-5)]
for lo, hi in bands:
    tot = 0; on_R = 0; cnt = 0
    for p in gears[(gears > lo) & (gears <= hi)]:
        sp = S[int(p)]; tot += int(sp.sum()); on_R += int((sp & R61).sum()); cnt += 1
        omega += (sp & R61)
    print(f"gears in ({lo},{hi}]: count {cnt}, total strikes {tot}, strikes on R_61 {on_R}")
R61n = int(R61.sum())
T = int(omega[R61].sum())
print("|R_61| =", R61n, " T = sum omega =", T, " T/|R_61| =", round(T/R61n, 3))
vals, cts = np.unique(omega[R61], return_counts=True)
print("omega distribution on R_61:", dict(zip(vals.tolist(), cts.tolist())))
print("free columns k=0 (e coprime to 6, e^2<2s-1):", len([e for e in range(5, math.isqrt(2*s-1)+1) if e % 6 in (1,5)]),
      " k=1 (u>=1, 6u^2<=4c-1):", math.isqrt((4*c-1)//6))
# top band: each gear strikes 1 or 2
tb = [int(S[int(p)].sum()) for p in gears[gears > 2*s//3]]
print("top band strike counts: min", min(tb), "max", max(tb), "n gears", len(tb))
# F7-general check: j = 0 mod p never struck at lower member, for all gears
bad = 0
for p in gears:
    jj = js[js % p == 0]
    if np.any((s*s + 6*jj - 1) % p == 0): bad += 1
print("gears striking a lower member at an offset j = 0 mod p:", bad)
# primorial-offset columns: j = k*5005 (B=13), both members 13-rough?
P = 5*7*11*13
Q = [p for p in (5,7,11,13) if (s*s+1) % p == 0]
print("Q = primes <=13 dividing s^2+1:", Q)
kk = [k for k in range(-(2*c-1)//P, (2*c-1)//P + 1) if k != 0]
rough13 = [k for k in kk if all((s*s+6*k*P-1) % p and (s*s+6*k*P+1) % p for p in (5,7,11,13))]
print("primorial offsets k*5005, k in", kk, " 13-rough:", rough13, " twin among them:", [k for k in rough13 if tw[np.searchsorted(js, k*P)]])
print("total", round(time.time()-t0,1), "s")
