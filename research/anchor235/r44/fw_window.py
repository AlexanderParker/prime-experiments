"""R3.h.i part 3 - the two-sided walk at the WINDOWS of q = 59..997.

Window of M = {5..q}: columns lo = q//6 + 1 .. hi = (q'^2-1)//6, where an opening of M is a
twin pair.  A junction is a window opening struck by q'.

Structural point checked here: inside the window both members of an opening are PRIME, so a
member divisible by q' must BE q' or q'^2 (a larger multiple of q' is composite and would need a
second factor >= q', putting it above q'^2).  So a window has at most two junctions.

Also measured, for scale: the two-sided walk at EVERY window opening (the generic flank), the
budget, the correlation, the gear bands and the umbrella bound.

Writes results/fw_window.txt.
"""
import os
from math import isqrt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "fw_window.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


QMAX = 1100
fl = bytearray([1]) * (QMAX + 1)
fl[0:2] = b"\x00\x00"
for i in range(2, isqrt(QMAX) + 1):
    if fl[i]:
        fl[i * i:: i] = bytearray(len(range(i * i, QMAX + 1, i)))
PR = [p for p in range(2, QMAX + 1) if fl[p]]
RUNGS = [p for p in PR if 59 <= p <= 997]
DEEP = [59, 71, 89, 101, 127, 149, 181, 211, 251, 293, 347, 409, 479, 571, 683, 809, 907, 997]
MARGIN = 800


def pearson(pairs):
    n = len(pairs)
    if n < 3:
        return None
    sx = sy = sxx = syy = sxy = 0
    for a, b in pairs:
        sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b
    num = n * sxy - sx * sy
    den = ((n * sxx - sx * sx) * (n * syy - sy * sy)) ** 0.5
    return num / den if den else None


A5 = {0, 2, 3}
COUP = {}
for lp in range(5):
    rs = [r for r in A5 if (r + lp) % 5 in A5]
    COUP[lp] = set((r - s) % 5 for r in rs for s in A5)

say("BAND: windows of q = 59..997 (%d rungs)" % len(RUNGS))
say("")
say("q     q'    winopen  F_W  maxS_all  F_W+q'  slack  jA(q')  jB(q'^2)  S_A      S_B")

nA = nB = 0
budget_exc = 0
allpairs = []
juncpairs = []
rows = []
c1exc = 0
deep_stats = []
for q in RUNGS:
    qp = PR[PR.index(q) + 1]
    gs = [p for p in PR if 5 <= p <= q]
    U = {g: pow(6, -1, g) for g in gs}
    lo = q // 6 + 1
    hi = (qp * qp - 1) // 6
    A = lo - MARGIN
    B = hi + MARGIN
    n = B - A + 1
    ba = bytearray(n)
    one = b"\x01"
    for g in gs:
        u = U[g]
        for t in (u % g, (-u) % g):
            s = (t - A) % g
            ba[s::g] = one * len(range(s, n, g))
    op = []
    pos = ba.find(0)
    while pos != -1:
        op.append(pos + A)
        pos = ba.find(0, pos + 1)
    idx = {v: i for i, v in enumerate(op)}
    winop = [v for v in op if lo <= v <= hi]
    F_W = max((winop[i + 1] - winop[i] for i in range(len(winop) - 1)), default=0)
    pairs = []
    for v in winop:
        i = idx[v]
        pairs.append((v, v - op[i - 1], op[i + 1] - v))
    maxS_all = max(a + b for (_, a, b) in pairs)
    allpairs.extend([(a, b) for (_, a, b) in pairs])
    for (_, a, b) in pairs:
        if a % 5 not in COUP[b % 5]:
            c1exc += 1
    if maxS_all > F_W + qp:
        budget_exc += 1
    up = pow(6, -1, qp)
    teeth = (up % qp, (-up) % qp)
    jl = [(v, a, b) for (v, a, b) in pairs if v % qp in teeth]
    xA = (qp - (qp % 6 == 1) * 1 - (qp % 6 == 5) * (-1)) // 6 if False else None
    colA = [v for (v, a, b) in jl if 6 * v - 1 == qp or 6 * v + 1 == qp]
    colB = [v for (v, a, b) in jl if 6 * v - 1 == qp * qp or 6 * v + 1 == qp * qp]
    other = [v for (v, a, b) in jl if v not in colA and v not in colB]
    SA = [a + b for (v, a, b) in jl if v in colA]
    SB = [a + b for (v, a, b) in jl if v in colB]
    nA += len(colA)
    nB += len(colB)
    juncpairs.extend([(a, b) for (v, a, b) in jl])
    rows.append((q, qp, len(winop), F_W, maxS_all, F_W + qp, F_W + qp - maxS_all,
                 len(colA), len(colB), len(other), SA[0] if SA else None,
                 SB[0] if SB else None))
    if len(other):
        say("!! rung %d has %d junction(s) that are neither the q' column nor the q'^2 column: %s"
            % (q, len(other), other))

    if q in DEEP:
        ARC = {g: min((2 * U[g]) % g, g - (2 * U[g]) % g) for g in gs}
        LONG = {g: g - ARC[g] for g in gs}
        f1 = f2 = f4 = f13 = 0
        bI = bII = bIII = hI = hII = hIII = 0
        us = []
        sh = []
        for (x, lm, lp) in pairs:
            S = lm + lp
            miss = []
            s0 = 0
            for g in gs:
                u = U[g]
                bp = min((u - x) % g, (-u - x) % g)
                bm = min((x - u) % g, (x + u) % g)
                if bp + bm not in (ARC[g], LONG[g]):
                    f1 += 1
                hf = bp <= lp - 1
                hb = bm <= lm - 1
                if LONG[g] < S + 2:
                    bI += 1
                    hI += 1 if (hf or hb) else 0
                elif ARC[g] <= S:
                    bII += 1
                    hII += 1 if (hf or hb) else 0
                else:
                    bIII += 1
                    hIII += 1 if (hf or hb) else 0
                if hf and hb:
                    s0 += 1
                    if ARC[g] > S:
                        f2 += 1
                elif not hf and not hb:
                    miss.append(g)
                    if LONG[g] < S + 2:
                        f4 += 1
            sh.append(s0)
            if miss:
                gm = min(miss)
                us.append((gm, LONG[gm] - S))
                if S > LONG[gm]:
                    f13 += 1
        deep_stats.append((q, len(pairs), f1, f2, f4, f13, bI, hI, bII, hII, bIII, hIII,
                           sum(sh) / len(sh), max(sh),
                           min(u[1] for u in us), sorted(u[1] for u in us)[len(us) // 2],
                           min(u[0] for u in us), sorted(u[0] for u in us)[len(us) // 2]))

for x in rows:
    if x[0] in DEEP or x[7] or x[8]:
        say("%-5d %-5d %-8d %-4d %-9d %-7d %-6d %-7d %-9d %-8s %s"
            % (x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8],
               str(x[10]), str(x[11])))

say("")
say("STRUCTURE OF THE WINDOW'S JUNCTIONS over %d rungs" % len(RUNGS))
say("  junctions that are the column of q'   : %d of %d rungs" % (nA, len(RUNGS)))
say("  junctions that are the column of q'^2 : %d of %d rungs" % (nB, len(RUNGS)))
say("  junctions that are neither            : 0 (structural: a window opening's members are "
    "prime, so a q'-multiple among them is q' or q'^2)")
say("  total window junctions: %d" % (nA + nB))
say("")
say("THE BUDGET IN THE WINDOW (all window openings, not only junctions)")
say("  max S > F_W + q' at %d of %d rungs" % (budget_exc, len(RUNGS)))
sl = [x[6] for x in rows]
say("  slack F_W + q' - max S: min %d median %d max %d"
    % (min(sl), sorted(sl)[len(sl) // 2], max(sl)))
rr = [x[4] / x[3] for x in rows if x[3]]
say("  max S / F_W: min %.3f median %.3f max %.3f"
    % (min(rr), sorted(rr)[len(rr) // 2], max(rr)))
say("  C1 anchor coupling exceptions over all window openings: %d of %d"
    % (c1exc, len(allpairs)))
say("  correlation of (L^-, L^+) over all window openings: %+.5f (n=%d)"
    % (pearson(allpairs), len(allpairs)))
if len(juncpairs) >= 3:
    say("  correlation over the %d window junctions: %+.5f"
        % (len(juncpairs), pearson(juncpairs)))
sa = [x[10] for x in rows if x[10]]
sb = [x[11] for x in rows if x[11]]
say("  span at the q' column   : n=%d min %d median %d max %d" % (len(sa), min(sa),
    sorted(sa)[len(sa) // 2], max(sa)))
say("  span at the q'^2 column : n=%d min %d median %d max %d" % (len(sb), min(sb),
    sorted(sb)[len(sb) // 2], max(sb)))

say("")
say("DEEP RUNGS - every window opening, every gear")
say("q     openings  F1  F2  F4  F13 | bandI cells/strike | bandII | bandIII | shared mean/max | "
    "umbrella slack min/med | smallest missing gear min/med")
for d in deep_stats:
    say("%-5d %-9d %-3d %-3d %-3d %-3d | %d/%d (%.4f) | %d/%d (%.4f) | %d/%d (%.4f) | %.2f/%d | "
        "%d/%d | %d/%d"
        % (d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[7] / d[6] if d[6] else 0,
           d[8], d[9], d[9] / d[8] if d[8] else 0, d[10], d[11], d[11] / d[10] if d[10] else 0,
           d[12], d[13], d[14], d[15], d[16], d[17]))
LOG.close()
