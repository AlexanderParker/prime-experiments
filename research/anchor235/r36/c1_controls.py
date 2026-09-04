# Branch 5g, controls and mechanism.
#
# (A) The baseline for "gear g is at its coverage maximum": share_max(g, L) = the share of the g
#     phases that attain m_g(L).  Without it, "at maximum in every record" can be free.
#     Reported at every machine m13..m31 and at the record and runner-up lengths.
# (B) The same on the window side: at every prime rung 23..1999, is gear 5 (7, 11, ...) at its
#     coverage maximum on the window's LONGEST blocked stretch, against share_max at that length?
# (C) Mechanism for the hinge's position: pooled over the rungs, the hinge density, the mean
#     striker count and the mean hinge-gear size by decile of the stretch.
# (D) The null for the length rule: over EVERY blocked stretch of the window (not only the
#     longest), the joint behaviour of L and g_h - is "L <= 2 g_h/3" a rule or a scale coincidence?
#
# Self-contained; numpy only.  Run: uv run python research/anchor235/r36/c1_controls.py
import os
import time
from math import prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
OUT = os.path.join(RES, "c1_controls.txt")
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def primes_upto(n):
    s = np.ones(n + 1, bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return [int(p) for p in np.flatnonzero(s)]


PR = primes_upto(2100)
NXT = {PR[i]: PR[i + 1] for i in range(len(PR) - 1)}


def teeth(g):
    u = pow(6, -1, g)
    return (u, g - u)


def cov(s, L, g):
    n = 0
    for t in teeth(g):
        off = (t - s) % g
        if off < L:
            n += 1 + (L - 1 - off) // g
    return n


def maxcov(L, g):
    a, b = teeth(g)
    return max(cov(a, L, g), cov(b, L, g))


def sharemax(L, g):
    m = maxcov(L, g)
    return sum(1 for r in range(g) if cov(r, L, g) == m) / g


say("Branch 5g, controls and mechanism.")
say("")
say("=== (A) The baseline.  share_max(g,L) = share of the g phases attaining m_g(L).")
say("  'obs' = share of the machine's extremal stretches with g at its maximum (from a1).")
LAD = [5, 7, 11, 13, 17, 19, 23, 29, 31]
FK = {13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58}
LENS = {13: [10, 9, 7], 17: [17, 15, 14], 19: [24, 22, 21], 23: [33, 32, 31],
        29: [42, 39], 31: [57, 54]}
for q in (13, 17, 19, 23, 29, 31):
    gears = [g for g in LAD if g <= q]
    say(f"  m{q}")
    say("    L    kind    " + "  ".join(f"{g:>10}" for g in gears))
    for i, L in enumerate(LENS[q]):
        kind = "RECORD" if i == 0 else "runner"
        say(f"    {L:<4} {kind:<7} " + "  ".join(
            f"m={maxcov(L,g)} s={sharemax(L,g):.2f}" for g in gears))

say("")
say("=== (B) The window side: coverage maximality on the window's longest blocked stretch.")
say("  For every prime rung 23..1999.  'at max' counted against the phase baseline share_max.")
t0 = time.time()
rows = []
for q in PR:
    if q < 23 or q > 1999:
        continue
    qq = NXT[q]
    lo, hi = q // 6 + 1, (qq * qq - 1) // 6
    n = hi - lo + 1
    gears = [g for g in PR if 5 <= g <= q]
    cnt = np.zeros(n, np.int16)
    gsum = np.zeros(n, np.int64)
    for g in gears:
        for t in teeth(g):
            s = (t - lo) % g
            cnt[s::g] += 1
            gsum[s::g] += g
    op = np.flatnonzero(cnt == 0) + lo
    if len(op) < 4:
        continue
    d = np.diff(op)
    j = int(d.argmax())
    L = int(d[j]) - 1
    s = int(op[j]) + 1
    i0 = s - lo
    c = cnt[i0:i0 + L]
    gs = gsum[i0:i0 + L]
    hin = [(int(gs[t]), int(t)) for t in np.flatnonzero(c == 1)]
    atmax = {}
    for g in (5, 7, 11, 13, 17, 19, 23):
        if g <= q:
            atmax[g] = (cov(s, L, g) == maxcov(L, g), sharemax(L, g))
    # (C) per-decile hinge and striker statistics for this stretch
    dec = np.minimum((np.arange(L) * 10) // max(1, L), 9)
    hd = np.zeros(10)
    hg = [[] for _ in range(10)]
    for g, t in hin:
        hd[dec[t]] += 1
        hg[dec[t]].append(g)
    sc = np.array([c[dec == k].mean() for k in range(10)])
    # (D) every blocked stretch of this window: L and its largest hinge gear
    allLg = []
    for t in range(len(d)):
        LL = int(d[t]) - 1
        if LL < 10:
            continue
        a0 = int(op[t]) + 1 - lo
        cc = cnt[a0:a0 + LL]
        gg = gsum[a0:a0 + LL]
        h = gg[cc == 1]
        allLg.append((LL, int(h.max()) if h.size else 0))
    rows.append(dict(q=q, L=L, s=s, hin=hin, atmax=atmax, hd=hd, sc=sc, hg=hg,
                     allLg=allLg, ngears=len(gears)))
    del cnt, gsum, op
say(f"  [{time.time()-t0:.1f}s over {len(rows)} rungs]")
say("  gear   at max on the longest window stretch   mean baseline share_max   lift")
for g in (5, 7, 11, 13, 17, 19, 23):
    sub = [r for r in rows if g in r['atmax']]
    obs = np.mean([r['atmax'][g][0] for r in sub])
    base = np.mean([r['atmax'][g][1] for r in sub])
    say(f"  {g:>4}   {obs:>36.3f}   {base:>22.3f}   {obs/base:>5.2f}")
say("  rungs where gear 5 is NOT at its coverage maximum on the longest window stretch: "
    + str([r['q'] for r in rows if not r['atmax'][5][0]]))

say("")
say("=== (C) Where the hinges sit inside the stretch (deciles, pooled over all rungs).")
HD = np.sum([r['hd'] for r in rows], axis=0)
SC = np.mean([r['sc'] for r in rows], axis=0)
MG = [np.mean([g for r in rows for g in r['hg'][k]]) for k in range(10)]
MGmax = [np.mean([max(r['hg'][k]) for r in rows if r['hg'][k]]) for k in range(10)]
say("  decile      0     1     2     3     4     5     6     7     8     9")
say("  hinges  " + "  ".join(f"{int(x):>5}" for x in HD))
say("  strikers" + "  ".join(f"{x:>5.2f}" for x in SC))
say("  meangear" + "  ".join(f"{x:>5.0f}" for x in MG))
say("  maxgear " + "  ".join(f"{x:>5.0f}" for x in MGmax))
say("  (hinges = count of single-striker columns in that tenth of the stretch, all rungs pooled;")
say("   strikers = mean striker count of the columns there; meangear/maxgear = size of the")
say("   hinge gears there.)")
allh = [(g, t / max(1, r['L'] - 1)) for r in rows for g, t in r['hin']]
gs_ = np.array([g for g, _ in allh], float)
ps_ = np.array([p for _, p in allh])
cen = np.abs(ps_ - 0.5)
say(f"  Spearman-style rank correlation of hinge gear size with |position - 0.5|: "
    f"{np.corrcoef(np.argsort(np.argsort(gs_)), np.argsort(np.argsort(cen)))[0,1]:+.3f} "
    f"over {len(allh)} hinge columns")
for lo_, hi_ in ((0, .1), (.1, .3), (.3, .5)):
    m = (cen >= lo_) & (cen < hi_)
    say(f"    |pos-0.5| in [{lo_},{hi_}): {m.sum():>6} hinges, mean gear {gs_[m].mean():>7.1f}, "
        f"median gear {np.median(gs_[m]):>7.1f}")

say("")
say("=== (D) The null for the length rule, over EVERY blocked stretch of the window.")
say("  q      stretches (L>=10)   share with 3L > 2 g_h   share with L > g_h   "
    "rank corr(L, g_h)   longest stretch's 2g_h/3L")
for r in rows:
    if r['q'] not in (59, 173, 499, 997, 1999):
        continue
    A = np.array(r['allLg'], float)
    LL, GG = A[:, 0], A[:, 1]
    v1 = np.mean(3 * LL > 2 * GG)
    v2 = np.mean(LL > GG)
    rc = np.corrcoef(np.argsort(np.argsort(LL)), np.argsort(np.argsort(GG)))[0, 1]
    gh0 = max(g for g, _ in r['hin'])
    say(f"  {r['q']:<6} {len(A):<19} {v1:<23.3f} {v2:<19.3f} {rc:>+16.3f}   "
        f"{2*gh0/(3*r['L']):.3f}")
allv = np.concatenate([np.array(r['allLg'], float) for r in rows])
say(f"  pooled over all rungs, {len(allv)} blocked stretches with L >= 10: "
    f"3L > 2 g_h at {np.mean(3*allv[:,0] > 2*allv[:,1]):.3f}, L > g_h at "
    f"{np.mean(allv[:,0] > allv[:,1]):.3f}")
say(f"  rank correlation of L with g_h over those: "
    f"{np.corrcoef(np.argsort(np.argsort(allv[:,0])), np.argsort(np.argsort(allv[:,1])))[0,1]:+.3f}")
say("  g_h/L by length band (all stretches, all rungs):")
for a, b in ((10, 25), (25, 50), (50, 100), (100, 300)):
    m = (allv[:, 0] >= a) & (allv[:, 0] < b)
    if m.sum():
        say(f"    L in [{a},{b}): n = {int(m.sum()):>6}, median g_h/L = "
            f"{np.median(allv[m,1]/allv[m,0]):.2f}, share 3L > 2 g_h = "
            f"{np.mean(3*allv[m,0] > 2*allv[m,1]):.3f}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("written", OUT)
