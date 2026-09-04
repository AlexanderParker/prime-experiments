"""R2.a.i.a - part 3: the interaction between the islands of [1, d) and the large gears.

For every prime q = 5..QMAX, over the offsets 1..d-1 (d = the top gear's forward tooth arc):

  * N_isl(B)  = how many islands for bound B lie in [1, d)   -- a q-FREE count, fixed by d alone;
  * strikes(B)= how many (gear, island) strikes land on them -- every striker is a gear > B, by
                the definition of an island, so this is exactly the large gears' work;
  * free(B)   = islands struck by NO gear: each one is an opening, so free >= 1 forces L < d;
  * where the first free island sits against the landing L;
  * the ratio strikes/N_isl against 2 (ln ln q - ln ln B), the Mertens count.

Also, for q <= 3000, the SMALLEST gear that strikes each island (descending overwrite pass).

Writes results/rl_interact.txt.
Usage: uv run python research/anchor235/r39/rl_interact.py [--QMAX 20000]
"""
import argparse
import os
from collections import Counter
from math import isqrt, log

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
LOG = open(os.path.join(OUT, "rl_interact.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


ap = argparse.ArgumentParser()
ap.add_argument("--QMAX", type=int, default=20000)
ap.add_argument("--MSQ", type=int, default=3000)
args = ap.parse_args()
QMAX = args.QMAX


def sieve(n):
    fl = bytearray([1]) * (n + 1)
    fl[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if fl[i]:
            fl[i * i:: i] = bytearray(len(range(i * i, n + 1, i)))
    return fl


FL = sieve(QMAX + 10)
GEARS = [p for p in range(5, QMAX + 1) if FL[p]]
UU = [pow(6, -1, g) for g in GEARS]
NG = len(GEARS)
DMAX = (2 * QMAX + 1) // 3 + 2
say("gears 5..%d: %d;  offsets up to d_max = %d" % (QMAX, NG, DMAX))

BS = (7, 11, 13, 17)
MASK = {}
for B in BS:
    res = np.load(os.path.join(OUT, "rl_isl_%d.npy" % B))
    mod = int(open(os.path.join(OUT, "rl_isl_%d_mod.txt" % B)).read())
    m = np.zeros(DMAX, dtype=bool)
    rs = np.array(sorted(int(v) for v in res), dtype=np.int64)
    for base in range(0, DMAX, mod):
        idx = rs + base
        idx = idx[idx < DMAX]
        m[idx] = True
    MASK[B] = m
    say("B = %2d: %d islands below d_max (density %.6f)" % (B, int(m.sum()), m.mean()))

rows = np.load(os.path.join(OUT, "rl_rows.npy"))
Lof = {int(r[0]): int(r[1]) for r in rows}

res_rows = []
msdist = Counter()
msn = 0
for qi, q in enumerate(GEARS):
    nq = qi + 1
    qq = q * q
    c = pow(6, -1, q)
    d = (2 * c) % q
    if d < 2:
        continue
    D = int(d)
    cnt = np.zeros(D, dtype=np.int16)
    for j in range(nq):
        g = GEARS[j]
        u = UU[j]
        r = qq % g
        a = ((2 - r) * u) % g
        b = ((-r) * u) % g
        if a < D:
            cnt[a::g] += 1
        if b < D:
            cnt[b::g] += 1
    L = Lof[q]
    open_mask = cnt == 0
    open_mask[0] = False
    rec = {"q": q, "d": D, "L": L}
    for B in BS:
        isl = MASK[B][:D].copy()
        isl[0] = False
        n_isl = int(isl.sum())
        st = int(cnt[isl].sum()) if n_isl else 0
        fr = np.flatnonzero(isl & open_mask)
        rec[B] = (n_isl, st, len(fr), int(fr[0]) if len(fr) else -1,
                  int((isl[:min(L, D)]).sum()))
    res_rows.append(rec)
    if q <= args.MSQ and rec[13][0]:
        ms = np.zeros(D, dtype=np.int32)
        for j in range(nq - 1, -1, -1):
            g = GEARS[j]
            u = UU[j]
            r = qq % g
            a = ((2 - r) * u) % g
            b = ((-r) * u) % g
            if a < D:
                ms[a::g] = g
            if b < D:
                ms[b::g] = g
        isl = MASK[13][:D].copy()
        isl[0] = False
        for v in ms[isl]:
            if v:
                msdist[int(v)] += 1
                msn += 1

say("walks measured: %d" % len(res_rows))

# ------------------------------------------------------------------ item 5: the interaction
say("")
say("=== 5a. islands in [1, d): are they all struck?  (free island => an opening => L < d) ===")
say(" B    walks with     walks with      walks with      q with 0 free islands")
say("      >=1 island     0 islands       >=1 free")
for B in BS:
    have = [r for r in res_rows if r[B][0] > 0]
    freeq = [r for r in have if r[B][2] > 0]
    nofree = [r["q"] for r in have if r[B][2] == 0]
    say("%3d   %6d         %6d          %6d          %s"
        % (B, len(have), len(res_rows) - len(have), len(freeq),
           nofree[:12] if nofree else "none"))
    if nofree:
        say("      (largest q with every island in [1,d) struck: %d;  count %d)"
            % (max(nofree), len(nofree)))

say("")
say("=== 5b. where the first free island sits against the landing L ===")
for B in BS:
    have = [r for r in res_rows if r[B][2] > 0]
    eq = sum(1 for r in have if r[B][3] == r["L"])
    gt = sum(1 for r in have if r[B][3] > r["L"])
    lt = sum(1 for r in have if r[B][3] < r["L"])
    say("B = %2d: first free island = L at %d, > L at %d, < L at %d (must be 0) of %d walks"
        % (B, eq, gt, lt, len(have)))
    below = [r[B][4] for r in have]
    say("        islands strictly below the landing (all struck): median %d, max %d, mean %.2f"
        % (int(np.median(below)), max(below), float(np.mean(below))))

say("")
say("=== 5c. how many free islands a walk has (the slack of the frame) ===")
say(" B     median   mean    max    q band 10000-20000: median  mean")
for B in BS:
    have = [r for r in res_rows if r[B][0] > 0]
    v = np.array([r[B][2] for r in have])
    hi = np.array([r[B][2] for r in have if r["q"] >= 10000])
    say("%3d    %6d   %5.2f   %4d                        %6d  %5.2f"
        % (B, int(np.median(v)), v.mean(), v.max(),
           int(np.median(hi)) if len(hi) else -1, hi.mean() if len(hi) else -1))

# ------------------------------------------------------------------ item 6: the counting
say("")
say("=== 6. strikes on islands against islands, exactly, per q ===")
say(" B    q band        walks   islands(med)  strikes/island   2(lnln q - lnln B)   ratio > 1 at")
bands = [(5, 100), (100, 1000), (1000, 5000), (5000, 10000), (10000, 20000)]
for B in BS:
    for lo, hi in bands:
        have = [r for r in res_rows if lo <= r["q"] < hi and r[B][0] > 0]
        if not have:
            continue
        ni = np.array([r[B][0] for r in have], dtype=float)
        st = np.array([r[B][1] for r in have], dtype=float)
        ratio = st.sum() / ni.sum()
        qm = float(np.median([r["q"] for r in have]))
        pred = 2 * (log(log(qm)) - log(log(B)))
        per = np.where(ni > 0, st / ni, 0)
        say("%3d   %6d-%-6d %6d   %10d    %8.3f         %8.3f            %d of %d"
            % (B, lo, hi, len(have), int(np.median(ni)), ratio, pred,
               int((per > 1).sum()), len(have)))

say("")
say("=== 6b. the ratio is the same for every B (the frame divides both sides by rho_B) ===")
QLO = QMAX // 2
have = [r for r in res_rows if r["q"] >= QLO]
for B in BS:
    hb = [r for r in have if r[B][0] > 0]
    ni = sum(r[B][0] for r in hb)
    st = sum(r[B][1] for r in hb)
    say("B = %2d over q in [%d, %d): islands %d, strikes %d, strikes/island %s"
        % (B, QLO, QMAX, ni, st, ("%.4f" % (st / ni)) if ni else "n/a"))

say("")
say("=== 5e. walks with no free island, by q band (the frame's failures) ===")
say(" B    q band          walks with islands   no free island   fraction")
for B in BS:
    for lo, hi in bands:
        have2 = [r for r in res_rows if lo <= r["q"] < hi and r[B][0] > 0]
        if not have2:
            continue
        nf = sum(1 for r in have2 if r[B][2] == 0)
        say("%3d   %6d-%-6d %12d %18d   %.4f"
            % (B, lo, hi, len(have2), nf, nf / len(have2)))

# ------------------------------------------------------------------ the smallest striker
say("")
say("=== 5d. which gear strikes an island: the smallest striker (q <= %d, B = 13) ===" % args.MSQ)
say("total struck islands counted: %d" % msn)
say("smallest striker   count    share    2/g share of sum 2/g over gears > 13")
tot = sum(msdist.values())
for g, n in msdist.most_common(14):
    say("   %8d   %8d   %.4f" % (g, n, n / tot if tot else 0))
say("distinct smallest strikers: %d; largest %d" % (len(msdist), max(msdist) if msdist else -1))
LOG.close()
