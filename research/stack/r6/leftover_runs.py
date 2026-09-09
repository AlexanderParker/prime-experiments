"""Second, independent measurement of the independent-slot prediction of twin-free runs
(branch R4.d.i.c).  The main script calibrates the model's per-slot twin probability p on depth
bins of width 0.1; a bin at the top of a section can span a large range of n, and p varies inside
it, which by convexity makes the bin-calibrated prediction of "no twin in the stretch" too small.
This script recomputes the same prediction under four calibrations of p and prints them side by
side, plus a by-K table (does the PP share among core-open slots depend on the stretch's own K?).

Calibrations: depth bins of width 0.1; depth bins of width 0.02; a local sliding estimate over W
slots (two values of W); and one global constant.  No factorisation: only the leftover, the twins
and the run counts.

usage: uv run python research/stack/r6/leftover_runs.py BASE SECTION L1,L2,... [NSIM] [TAG]
"""
import sys, os, math, json
import numpy as np
from sympy import nextprime, isprime

CHUNK = 4_000_000
base = int(sys.argv[1]); ksec = int(sys.argv[2])
Ls = [int(x) for x in sys.argv[3].split(',')]
NSIM = int(sys.argv[4]) if len(sys.argv) > 4 else 20
TAG = sys.argv[5] if len(sys.argv) > 5 else f"runs_b{base}s{ksec}"

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
lnn = np.log(slots.astype(np.float64)).astype(np.float32)
del slots
del isp
cst = np.concatenate([[0], np.cumsum(twin, dtype=np.int32)])
tidx = np.nonzero(twin)[0]
gaps = (np.diff(np.concatenate([[-1], tidx, [S]])) - 1).astype(np.int32)
del tidx
say(f"# base {base} section {ksec} = [{lo}, {hi}); slots {S}; twins {int(twin.sum())}; "
    f"record {int(gaps.max())} slots; NSIM {NSIM}")


def pred(pslot, Lp, nstart, logq=None):
    """analytic predicted twin-free starts and runs under independent slots with per-slot p."""
    pc = pslot.astype(np.float64)
    lp = np.where(pc >= 1.0, -50.0, np.log1p(-np.minimum(pc, 0.999999999)))
    lq = np.concatenate([[0.0], np.cumsum(lp, dtype=np.float64)])
    ps = 0.0; pr = 0.0
    for s0 in range(0, nstart, CHUNK):
        s1 = min(s0 + CHUNK, nstart)
        w = np.exp(lq[s0 + Lp:s1 + Lp] - lq[s0:s1])
        ps += float(w.sum())
        prev = pc[s0 - 1:s1 - 1] if s0 > 0 else np.concatenate([[0.0], pc[0:s1 - 1]])
        pr += float((w * prev).sum())
    return ps, pr


for Lp in Ls:
    tp = 6 * Lp + 1
    rl = (spf_low == 0) | (spf_low > tp)
    ru = (spf_up == 0) | (spf_up > tp)
    if tp >= n0:
        rl[:(tp - n0) // 6 + 1] = False
        ru[:max(0, (tp - n0 - 2) // 6 + 1)] = False
    op = rl & ru
    del rl, ru
    nstart = S - Lp + 1
    cso = np.concatenate([[0], np.cumsum(op, dtype=np.int32)])
    obs_tf = 0; obs_runs = 0
    for s0 in range(0, nstart, CHUNK):
        s1 = min(s0 + CHUNK, nstart)
        tf = (cst[s0 + Lp:s1 + Lp] - cst[s0:s1]) == 0
        obs_tf += int(tf.sum())
        prevtw = twin[s0 - 1:s1 - 1] if s0 > 0 else np.concatenate([[True], twin[0:s1 - 1]])
        obs_runs += int((tf & prevtw).sum())
        del tf, prevtw

    cal = {}
    for bw in (0.1, 0.02):
        u = lnn / np.float32(math.log(tp))
        b0 = int(math.floor(float(u.min()) / bw)); b1 = int(math.floor(float(u.max()) / bw))
        NB = b1 - b0 + 1
        bi = np.clip((u / np.float32(bw)).astype(np.int32) - b0, 0, NB - 1)
        del u
        co = np.bincount(bi, weights=op.astype(np.float64), minlength=NB)
        ct = np.bincount(bi, weights=(twin & op).astype(np.float64), minlength=NB)
        pb = np.where(co > 0, ct / np.maximum(co, 1), 0.0)
        cal[f"bin {bw}"] = np.where(op, pb[bi], 0.0)
        del bi, co, ct, pb
    for W in (200001, 1000001):
        h = W // 2
        j = np.arange(S)
        a = np.clip(j - h, 0, S); b = np.clip(j + h + 1, 0, S)
        tw = (cst[b] - cst[a]).astype(np.float64)
        ow = (cso[b] - cso[a]).astype(np.float64)
        cal[f"local W={W}"] = np.where(op, tw / np.maximum(ow, 1), 0.0)
        del j, a, b, tw, ow
    pg = float((twin & op).sum()) / float(op.sum())
    cal["global"] = np.where(op, pg, 0.0)

    say()
    say(f"## L' = {Lp}: t' = {tp}, core-open slots {int(op.sum())}, mean K "
        f"{float(cso[-1]) * 0 + (cso[Lp:nstart+Lp] - cso[0:nstart]).mean():.4f}, "
        f"observed twin-free starts {obs_tf}, runs {obs_runs}")
    say("calibration of p | predicted twin-free starts | obs/pred | predicted runs | obs/pred")
    for name, ps in cal.items():
        a, b = pred(ps, Lp, nstart)
        say(f"{name} | {a:.2f} | {obs_tf/max(a,1e-12):.3f} | {b:.2f} | {obs_runs/max(b,1e-12):.3f}")

    # the model's own spread under the local W=1000001 calibration
    ps = cal["local W=1000001"].astype(np.float32)
    sr = np.zeros(NSIM, dtype=np.int64); ss = np.zeros(NSIM, dtype=np.int64)
    rng = np.random.default_rng(4242 + Lp)
    for s in range(NSIM):
        st = rng.random(S, dtype=np.float32) < ps
        cs = np.concatenate([[0], np.cumsum(st, dtype=np.int32)])
        for s0 in range(0, nstart, CHUNK):
            s1 = min(s0 + CHUNK, nstart)
            tf = (cs[s0 + Lp:s1 + Lp] - cs[s0:s1]) == 0
            ss[s] += int(tf.sum())
            prevtw = st[s0 - 1:s1 - 1] if s0 > 0 else np.concatenate([[True], st[0:s1 - 1]])
            sr[s] += int((tf & prevtw).sum())
            del tf, prevtw
        del st, cs
    say(f"simulated (local W=1000001): starts {ss.mean():.1f} +- {ss.std(ddof=1):.1f}; "
        f"runs {sr.mean():.2f} +- {sr.std(ddof=1):.2f}; observed runs {obs_runs} -> z "
        f"{(obs_runs - sr.mean())/max(sr.std(ddof=1),1e-9):+.2f}")

    # by-K table: does the PP share among a stretch's core-open slots depend on K?
    pb01 = cal["bin 0.1"]; ploc = cal["local W=1000001"]
    lp01 = np.where(pb01 >= 1.0, -50.0, np.log1p(-np.minimum(pb01.astype(np.float64), 0.999999999)))
    lq01 = np.concatenate([[0.0], np.cumsum(lp01, dtype=np.float64)])
    lpl = np.where(ploc >= 1.0, -50.0, np.log1p(-np.minimum(ploc.astype(np.float64), 0.999999999)))
    lql = np.concatenate([[0.0], np.cumsum(lpl, dtype=np.float64)])
    KMAX = 60
    kc = np.zeros(KMAX + 1, dtype=np.int64); kpp = np.zeros(KMAX + 1, dtype=np.int64)
    ktf = np.zeros(KMAX + 1, dtype=np.int64)
    kp01 = np.zeros(KMAX + 1); kploc = np.zeros(KMAX + 1)
    for s0 in range(0, nstart, CHUNK):
        s1 = min(s0 + CHUNK, nstart)
        K = np.minimum((cso[s0 + Lp:s1 + Lp] - cso[s0:s1]), KMAX).astype(np.int64)
        PP = (cst[s0 + Lp:s1 + Lp] - cst[s0:s1]).astype(np.int64)
        kc += np.bincount(K, minlength=KMAX + 1)
        kpp += np.bincount(K, weights=PP, minlength=KMAX + 1).astype(np.int64)
        ktf += np.bincount(K[PP == 0], minlength=KMAX + 1)
        kp01 += np.bincount(K, weights=np.exp(lq01[s0 + Lp:s1 + Lp] - lq01[s0:s1]), minlength=KMAX + 1)
        kploc += np.bincount(K, weights=np.exp(lql[s0 + Lp:s1 + Lp] - lql[s0:s1]), minlength=KMAX + 1)
        del K, PP
    say("K | starts | mean PP | PP share of the class (mean PP / K) | twin-free observed | "
        "predicted (bin 0.1) | predicted (local) | predicted (class's own share)")
    for k in range(0, min(KMAX, 41) + 1):
        if kc[k] == 0: continue
        sh = (kpp[k] / kc[k] / k) if k else None
        cls = (kc[k] * (1 - sh) ** k) if k else float(kc[k])
        say(f"{k} | {int(kc[k])} | {kpp[k]/kc[k]:.4f} | "
            + (f"{sh:.6f}" if sh is not None else "-")
            + f" | {int(ktf[k])} | {kp01[k]:.2f} | {kploc[k]:.2f} | {cls:.2f}")
    del cso, op, cal, pb01, ploc, lq01, lql

say(); say("done")
LOG.close()
print("ok")
