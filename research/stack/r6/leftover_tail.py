"""Third measurement of the run-count deviation (branch R4.d.i.c).

leftover_depth_u.py and leftover_runs.py tie the core t' to the stretch length L', so the predicted
counts of twin gaps of at least L' slots come from a DIFFERENT model at each length and cannot be
differenced into band counts.  Here one core t' is fixed and the stretch length G is swept, so
N(>= G) is predicted by one model throughout and the bands [G, G') are consistent.  Also: the
overdispersion check -- the observed variance of the twin count PP over stretches of G slots against
the variance the independent-slot model itself has (E of the within-stretch variance plus the
variance of the stretch's own mean).

usage: uv run python research/stack/r6/leftover_tail.py BASE SECTION TPRIME G1,G2,... [NSIM] [TAG]
"""
import sys, os, math
import numpy as np
from sympy import nextprime, isprime

CHUNK = 4_000_000
base = int(sys.argv[1]); ksec = int(sys.argv[2]); tp = int(sys.argv[3])
Gs = [int(x) for x in sys.argv[4].split(',')]
NSIM = int(sys.argv[5]) if len(sys.argv) > 5 else 20
TAG = sys.argv[6] if len(sys.argv) > 6 else f"tail_b{base}s{ksec}_t{tp}"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, f"{TAG}.txt"), "w", encoding="utf-8")
def say(s=""):
    LOG.write(str(s) + "\n"); LOG.flush()


def sieve(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i * i::i] = False
    return s


cuts = [base]; firsts = [base]
for _ in range(ksec):
    f = firsts[-1]; c = f * f; cuts.append(c); firsts.append(c if isprime(c) else nextprime(c))
lo, hi = cuts[ksec - 1], cuts[ksec]
isp = sieve(hi + 40)
n0 = lo + ((5 - lo) % 6)
slots = np.arange(n0, hi, 6, dtype=np.int64)
if slots[-1] + 2 >= hi: slots = slots[:-1]
S = slots.size
twin = isp[slots] & isp[slots + 2]
sq = int(math.isqrt(hi)) + 1
smallp = np.nonzero(sieve(sq))[0]; smallp = smallp[smallp >= 5]
DT = np.int16 if sq < 32000 else np.int32
spf_low = np.zeros(S, dtype=DT); spf_up = np.zeros(S, dtype=DT)
for p in smallp[::-1]:
    p = int(p); inv = pow(6, -1, p)
    spf_low[((0 - n0) * inv) % p::p] = p
    spf_up[(((-2) % p - n0) * inv) % p::p] = p
del slots, isp
op = ((spf_low == 0) | (spf_low > tp)) & ((spf_up == 0) | (spf_up > tp))
if tp >= n0:
    op[:(tp - n0) // 6 + 1] = False
del spf_low, spf_up
cst = np.concatenate([[0], np.cumsum(twin, dtype=np.int32)])
cso = np.concatenate([[0], np.cumsum(op, dtype=np.int32)])
tidx = np.nonzero(twin)[0]
gaps = (np.diff(np.concatenate([[-1], tidx, [S]])) - 1).astype(np.int32)
del tidx

# local calibration of p, stretch W slots
W = 1000001; h = W // 2
j = np.arange(S)
a = np.clip(j - h, 0, S); b = np.clip(j + h + 1, 0, S)
pslot = np.where(op, (cst[b] - cst[a]) / np.maximum((cso[b] - cso[a]), 1), 0.0)
del j, a, b
lp = np.where(pslot >= 1.0, -50.0, np.log1p(-np.minimum(pslot, 0.999999999)))
lq = np.concatenate([[0.0], np.cumsum(lp, dtype=np.float64)])
csp = np.concatenate([[0.0], np.cumsum(pslot, dtype=np.float64)])
csq = np.concatenate([[0.0], np.cumsum(pslot * (1 - pslot), dtype=np.float64)])
del lp
ps32 = pslot.astype(np.float32)

say(f"# base {base} section {ksec} = [{lo}, {hi}); slots {S}; twins {int(twin.sum())}; "
    f"record {int(gaps.max())}; fixed core t' = {tp}; core-open slots {int(op.sum())}; "
    f"p calibrated locally over W = {W} slots; NSIM = {NSIM}")
say("G | mean K | observed N(gaps >= G) | predicted N(>= G) | sim mean | sim sd | z | "
    "observed mean PP | model mean PP | observed var PP | model var PP | var ratio")
res = {}
for G in Gs:
    nstart = S - G + 1
    obs = int((gaps >= G).sum())
    prd = 0.0
    sPP = 0.0; sPP2 = 0.0; mM = 0.0; vW = 0.0; sM = 0.0; sM2 = 0.0
    sumK = 0.0
    for s0 in range(0, nstart, CHUNK):
        s1 = min(s0 + CHUNK, nstart)
        w = np.exp(lq[s0 + G:s1 + G] - lq[s0:s1])
        prev = pslot[s0 - 1:s1 - 1] if s0 > 0 else np.concatenate([[0.0], pslot[0:s1 - 1]])
        prd += float((w * prev).sum())
        PP = (cst[s0 + G:s1 + G] - cst[s0:s1]).astype(np.float64)
        sPP += float(PP.sum()); sPP2 += float((PP * PP).sum())
        M = csp[s0 + G:s1 + G] - csp[s0:s1]
        sM += float(M.sum()); sM2 += float((M * M).sum())
        vW += float((csq[s0 + G:s1 + G] - csq[s0:s1]).sum())
        sumK += float((cso[s0 + G:s1 + G] - cso[s0:s1]).sum())
        del w, prev, PP, M
    n = nstart
    obs_mean = sPP / n; obs_var = sPP2 / n - obs_mean ** 2
    mod_mean = sM / n; mod_var = vW / n + (sM2 / n - mod_mean ** 2)
    sr = np.zeros(NSIM, dtype=np.int64)
    rng = np.random.default_rng(77 + G)
    for s in range(NSIM):
        st = rng.random(S, dtype=np.float32) < ps32
        cs = np.concatenate([[0], np.cumsum(st, dtype=np.int32)])
        for s0 in range(0, nstart, CHUNK):
            s1 = min(s0 + CHUNK, nstart)
            tf = (cs[s0 + G:s1 + G] - cs[s0:s1]) == 0
            prevtw = st[s0 - 1:s1 - 1] if s0 > 0 else np.concatenate([[True], st[0:s1 - 1]])
            sr[s] += int((tf & prevtw).sum())
            del tf, prevtw
        del st, cs
    sm = float(sr.mean()); sd = float(sr.std(ddof=1))
    z = (obs - prd) / sd if sd > 0 else float('nan')
    res[G] = (obs, prd)
    say(f"{G} | {sumK/n:.3f} | {obs} | {prd:.2f} | {sm:.2f} | {sd:.2f} | "
        + (f"{z:+.2f}" if z == z else "-")
        + f" | {obs_mean:.4f} | {mod_mean:.4f} | {obs_var:.4f} | {mod_var:.4f} | "
          f"{obs_var/mod_var:.4f}")

say()
say("bands (one model throughout): [G, G') | observed | predicted | Poisson z")
gs = sorted(res)
for i in range(len(gs) - 1):
    o = res[gs[i]][0] - res[gs[i + 1]][0]
    p = res[gs[i]][1] - res[gs[i + 1]][1]
    say(f"[{gs[i]}, {gs[i+1]}) | {o} | {p:.2f} | " + (f"{(o-p)/math.sqrt(max(p,1e-9)):+.2f}" if p > 0 else "-"))
say(); say("done")
LOG.close()
print("ok")
