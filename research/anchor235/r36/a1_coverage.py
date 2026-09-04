# Branch 5g, Theory A: the coverage PROFILE at extremal stretches of the full period.
#
# For a stretch of L columns starting at column s:
#   c_g(s,L) = # columns of the stretch that gear g strikes  (a gear strikes a column at most once)
#   m_g(L)   = max over the g phases of c_g            ("maximal coverage")
#   r_g      = c_g/m_g                                  ("coverage ratio", in [0,1])
#   sole_g   = # columns of the stretch whose ONLY striker is g
#   waste_g  = c_g - sole_g                             (strikes landing on already-covered columns)
#
# Ranges: full periods of {5..13} .. {5..31}.  For each machine the RECORD stretches (gap F,
# so L = F-1 blocked columns) and the two runner-up lengths (gaps F-1 and F-2).
# m13..m23 by one array; m29 and m31 by a chunked pass that keeps only the long gaps.
#
# Self-contained; numpy only.  Run: uv run python research/anchor235/r36/a1_coverage.py
import os
import time
from math import prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
OUT = os.path.join(RES, "a1_coverage.txt")
lines = []


def say(s=""):
    print(s)
    lines.append(s)


LAD = [5, 7, 11, 13, 17, 19, 23, 29, 31]
FK = {13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58}


def teeth(g):
    u = pow(6, -1, g)
    return (u, g - u)


def gears_of(q):
    return [g for g in LAD if g <= q]


def cov(s, L, g):
    a, b = teeth(g)
    # count j in [0,L) with (s+j) mod g in {a,b}
    n = 0
    for t in (a, b):
        off = (t - s) % g
        if off < L:
            n += 1 + (L - 1 - off) // g
    return n


def maxcov(L, g):
    return max(cov(r, L, g) for r in range(g))


def gaps_from_openings(op, P):
    """gaps between consecutive openings, with the wrap; returns (gap array, start-of-gap array).
    A gap of length d after opening x means the blocked stretch is x+1 .. x+d-1 (d-1 columns)."""
    d = np.diff(np.concatenate([op, [op[0] + P]]))
    return d, op


def small_period_gaps(q):
    gears = gears_of(q)
    P = prod(gears)
    w = np.ones(P, bool)
    for g in gears:
        for t in teeth(g):
            w[t::g] = False
    op = np.flatnonzero(w)
    return gaps_from_openings(op, P), P


def big_period_gaps(q, thresh, chunk=200_000_000):
    """chunked full-period pass; returns (list of (gap, start_opening)) for gaps >= thresh, and P."""
    gears = gears_of(q)
    P = prod(gears)
    found = []
    prev = None          # last opening seen (absolute)
    first = None
    t0 = time.time()
    start = 0
    while start < P:
        n = min(chunk, P - start)
        b = np.zeros(n, bool)
        for g in gears:
            for t in teeth(g):
                b[(t - start) % g::g] = True
        op = np.flatnonzero(~b)
        if op.size:
            op = op + start
            if first is None:
                first = int(op[0])
            if prev is not None:
                d = int(op[0]) - prev
                if d >= thresh:
                    found.append((d, prev))
            if op.size > 1:
                dd = np.diff(op)
                sel = np.flatnonzero(dd >= thresh)
                for j in sel:
                    found.append((int(dd[j]), int(op[j])))
            prev = int(op[-1])
        start += n
        del b, op
    # wrap
    d = (first + P) - prev
    if d >= thresh:
        found.append((d, prev))
    say(f"  [chunked scan of m{q}: P = {P:,}, {time.time()-t0:.1f}s, "
        f"{len(found)} gaps >= {thresh}]")
    return found, P


say("Branch 5g / Theory A.  The coverage profile at extremal stretches of the full period.")
say("c_g = columns of the stretch struck by g; m_g = the max over g's phases; r_g = c_g/m_g.")
say("")

# ---------------------------------------------------------------- collect the stretches
STR = {}          # q -> {L: [starts]}
for q in (13, 17, 19, 23):
    (d, op), P = small_period_gaps(q)
    F = int(d.max())
    assert F == FK[q], (q, F, FK[q])
    tops = sorted(set(d.tolist()), reverse=True)[:3]
    STR[q] = {}
    for L in tops:
        idx = np.flatnonzero(d == L)
        STR[q][L - 1] = [int(op[j]) + 1 for j in idx]     # start of the blocked stretch
    say(f"  m{q}: P = {P:,}, F = {F}; top three gap lengths {tops} with counts "
        f"{[int((d == L).sum()) for L in tops]}")

for q in (29, 31):
    F = FK[q]
    found, P = big_period_gaps(q, F - 3)
    lens = sorted({g for g, _ in found}, reverse=True)
    assert lens[0] == F, (q, lens[:5], F)
    tops = lens[:3]
    STR[q] = {}
    for L in tops:
        STR[q][L - 1] = sorted(s + 1 for g, s in found if g == L)
    say(f"  m{q}: P = {P:,}, F = {F}; top three gap lengths {tops} with counts "
        f"{[sum(1 for g, _ in found if g == L) for L in tops]}")

say("")
say("=== Gate: every collected stretch is exactly blocked with open flanks.")
for q, byL in STR.items():
    gears = gears_of(q)
    for L, starts in byL.items():
        for s in starts[:50]:
            assert not any((s - 1) % g in teeth(g) for g in gears)
            assert not any((s + L) % g in teeth(g) for g in gears)
            assert all(any(k % g in teeth(g) for g in gears) for k in range(s, s + L))
say("  passed (first 50 stretches of every length checked at every machine).")

# ---------------------------------------------------------------- A1, A2: the record profile
say("")
say("=== A1/A2.  The coverage ratio r_g = c_g/m_g at every RECORD stretch, gear by gear.")
say("  cell = c_g/m_g (r_g); 'sole' = columns only that gear strikes; L = F-1.")
recprof = {}
for q in (13, 17, 19, 23, 29, 31):
    L = FK[q] - 1
    gears = gears_of(q)
    mg = {g: maxcov(L, g) for g in gears}
    say("")
    say(f"  m{q}   L = {L}   records = {len(STR[q][L])}   "
        f"sum m_g = {sum(mg.values())}   sum m_g / L = {sum(mg.values())/L:.3f}")
    say("    gear      " + "  ".join(f"{g:>9}" for g in gears))
    say("    m_g       " + "  ".join(f"{mg[g]:>9}" for g in gears))
    rows = []
    for s in STR[q][L]:
        cg = {g: cov(s, L, g) for g in gears}
        sole = {g: 0 for g in gears}
        for k in range(s, s + L):
            hit = [g for g in gears if k % g in teeth(g)]
            if len(hit) == 1:
                sole[hit[0]] += 1
        rows.append((s, cg, sole))
        say(f"    s={s:<12} " + "  ".join(
            f"{cg[g]}/{mg[g]}={cg[g]/mg[g]:.2f}" for g in gears))
        say(f"      sole    " + "  ".join(f"{sole[g]:>9}" for g in gears)
            + f"   | sum c = {sum(cg.values())}, overlap = {sum(cg.values())-L}")
    recprof[q] = (L, mg, rows)
    # per-gear: share of records at maximum
    say("    at max    " + "  ".join(
        f"{sum(1 for _, cg, _ in rows if cg[g] == mg[g])/len(rows):>9.2f}" for g in gears))

say("")
say("=== A2 scored: r_g against g/q, pooled over all records of m19..m31.")
say("  band g/q        n cells   mean r_g   min r_g   share r_g = 1")
bands = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.01)]
pool = []
for q in (19, 23, 29, 31):
    L, mg, rows = recprof[q]
    for s, cg, sole in rows:
        for g in gears_of(q):
            pool.append((g / q, cg[g] / mg[g]))
for lo, hi in bands:
    v = [r for x, r in pool if lo <= x < hi]
    if v:
        say(f"  [{lo:.2f},{hi:.2f})   {len(v):>7}   {np.mean(v):>8.3f}   {min(v):>7.3f}   "
            f"{sum(1 for r in v if r == 1)/len(v):>13.3f}")

say("")
say("=== A1 strong form: is r_g non-increasing in rank?  (count of rank-inversions per record)")
for q in (13, 17, 19, 23, 29, 31):
    L, mg, rows = recprof[q]
    gears = gears_of(q)
    inv = []
    for s, cg, sole in rows:
        r = [cg[g] / mg[g] for g in gears]
        n = sum(1 for i in range(len(r)) for j in range(i + 1, len(r)) if r[i] < r[j])
        inv.append(n)
    say(f"  m{q}: inversions per record {inv}  (0 = perfectly decreasing; "
        f"max possible {len(gears)*(len(gears)-1)//2})")

# ---------------------------------------------------------------- A3: runner-ups
say("")
say("=== A3.  The same at the RUNNER-UP stretches (gap F-1 and F-2).")
say("  machine  L    stretches   gear5 at max   gear7 at max   top gear at max   mean r_g (bottom3/top1)")
for q in (13, 17, 19, 23, 29, 31):
    gears = gears_of(q)
    for L in sorted(STR[q], reverse=True):
        starts = STR[q][L]
        mg = {g: maxcov(L, g) for g in gears}
        n5 = n7 = ntop = 0
        rb, rt = [], []
        cap = starts[:3000]
        for s in cap:
            cg = {g: cov(s, L, g) for g in gears}
            if cg[5] == mg[5]:
                n5 += 1
            if cg[7] == mg[7]:
                n7 += 1
            if cg[q] == mg[q]:
                ntop += 1
            rb.append(np.mean([cg[g] / mg[g] for g in gears[:3]]))
            rt.append(cg[q] / mg[q])
        tag = "RECORD" if L == FK[q] - 1 else "runner"
        say(f"  m{q:<5} {L:<4} {len(starts):<6}{'*' if len(starts) > 3000 else ' '}{tag:<6} "
            f"{n5/len(cap):>10.3f}   {n7/len(cap):>12.3f}   {ntop/len(cap):>14.3f}   "
            f"{np.mean(rb):>6.3f} / {np.mean(rt):.3f}")
say("  (* = more stretches than the 3000 analysed; the shares are over the first 3000 in order.)")

# ---------------------------------------------------------------- A5: capacity
say("")
say("=== A5.  sum m_g against L: how much room the capacity count already has.")
say("  machine   L     sum m_g   sum m_g / L   sum c_g   overlap   sum c_g / sum m_g")
for q in (13, 17, 19, 23, 29, 31):
    L, mg, rows = recprof[q]
    s, cg, sole = rows[0]
    sc = sum(cg.values())
    say(f"  m{q:<7} {L:<5} {sum(mg.values()):<9} {sum(mg.values())/L:<13.3f} {sc:<9} "
        f"{sc-L:<9} {sc/sum(mg.values()):.3f}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("written", OUT)
