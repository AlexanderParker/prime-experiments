"""Branch W.a part 2: the bucket vector along the path (item 2), determinism and sensitivity
(item 3), the landing in detail (item 5), and the interaction census (P13).

Re-phasing one gear at a time: gear g's two strike classes on the path are
{a, a - d_g} with a = (2 - q^2) u_g mod g and d_g = 2 u_g mod g.  Two counterfactuals:
  FREE   - put a at each of the g classes (the counterfactual machine of the family work);
  REAL   - move q's residue mod g to each of the g-1 nonzero classes r, giving a = (2-r^2)u_g,
           which reaches only the (g+1)/2 classes a with 2 - a/u_g a square: the character
           constraint.
Writes results/pa_sens.txt.
"""
import os
from math import isqrt
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "pa_sens.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


QBUCK = 5000       # per-step bucket vector, landing detail, interaction census
QSENS = 1000       # one-gear re-phasing (cost sum_g g per q)


def sieve(n):
    fl = bytearray([1]) * (n + 1)
    fl[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if fl[i]:
            fl[i * i:: i] = bytearray(len(range(i * i, n + 1, i)))
    return fl


FL = sieve(QBUCK + 10)
GEARS = [p for p in range(5, QBUCK + 1) if FL[p]]
UU = [pow(6, -1, g) for g in GEARS]
say("gears 5..%d: %d" % (QBUCK, len(GEARS)))

CAP = 2048          # offsets 0..CAP-1 built for the re-phasing search

rows = []
for qi, q in enumerate(GEARS):
    nq = qi + 1
    qq = q * q
    POS = 512
    while True:
        marks = [[] for _ in range(POS)]
        A = [0] * nq
        D = [0] * nq
        for idx in range(nq):
            g = GEARS[idx]
            u = UU[idx]
            r = qq % g
            a = ((2 - r) * u) % g
            b = ((-r) * u) % g
            A[idx] = a
            D[idx] = (2 * u) % g
            for off in (a, b):
                j = off % g
                while j < POS:
                    marks[j].append(g)
                    j += g
        L = None
        for i in range(1, POS):
            if not marks[i]:
                L = i
                break
        if L is not None:
            break
        POS *= 2
    # ---------------- bucket vector along the path
    # b_g(i) = distance from offset i to g's next strike at or after i
    near = []          # (nearest gear with positive distance, that distance)
    qrank = []
    for i in range(L + 1):
        best = 10 ** 9
        bg = 0
        dq = 0
        pos = []
        for idx in range(nq):
            g = GEARS[idx]
            a = A[idx]
            b = (a - D[idx]) % g
            dist = min((a - i) % g, (b - i) % g)
            if dist > 0:
                if dist < best:
                    best = dist
                    bg = g
                pos.append(dist)
            else:
                pos.append(0)
            if g == q:
                dq = dist
        near.append((bg, best))
        qrank.append(sum(1 for d in pos if d > dq))
    # ---------------- landing detail
    land = L
    flank_prev = sorted(marks[L - 1])
    flank_next = sorted(marks[L + 1]) if L + 1 < POS else []
    # ---------------- interaction census
    multi = Counter()          # gears striking >= 2 columns of the path
    for idx in range(nq):
        g = GEARS[idx]
        a = A[idx]
        b = (a - D[idx]) % g
        c = 0
        for off in (a, b):
            j = off % g
            while j < L:
                c += 1
                j += g
        if c >= 2:
            multi[g] = c
    sole = sorted({marks[i][0] for i in range(L) if len(marks[i]) == 1})
    # exact minimum blocking set for the path (offsets 0..L-1), branch and bound
    cols = list(range(L))
    cover = {}
    for i in cols:
        for g in marks[i]:
            cover.setdefault(g, set()).add(i)
    # greedy first for an upper bound
    need = set(cols)
    greedy = []
    while need:
        g = max(cover, key=lambda g: len(cover[g] & need))
        greedy.append(g)
        need -= cover[g]
    best_sz = [len(greedy)]
    exact = False
    if L <= 40:
        budget = [200000]

        def bb(need, nch):
            if budget[0] <= 0:
                return
            budget[0] -= 1
            if not need:
                if nch < best_sz[0]:
                    best_sz[0] = nch
                return
            if nch + 1 >= best_sz[0]:
                return
            i = min(need)
            for g in marks[i]:
                bb(need - cover[g], nch + 1)
        bb(frozenset(cols), 0)
        exact = budget[0] > 0
    rows.append(dict(q=q, nq=nq, L=L, A=A, D=D, marks=marks, near=near, qrank=qrank,
                     flank_prev=flank_prev, flank_next=flank_next, multi=dict(multi),
                     sole=sole, mincov=best_sz[0], exact=exact, greedy=len(greedy), POS=POS))

say("paths computed for the bucket / census part:", len(rows))

# ===================================================== item 2: bucket vector along the path
say("")
say("=== the bucket vector along the path (item 2) ===")
nc = Counter()
for r in rows:
    for bg, d in r["near"]:
        nc[bg] += 1
tot = sum(nc.values())
say("nearest-tooth gear over all path steps (top 10):",
    [(g, c, round(c / tot, 4)) for g, c in nc.most_common(10)])
dd = Counter()
for r in rows:
    for bg, d in r["near"]:
        dd[d] += 1
say("distance to the nearest tooth at a path step:", sorted(dd.items()))
say("the nearest-tooth gear is 5 or 7 at %.4f of all steps" % ((nc[5] + nc[7]) / tot))
rk = []
for r in rows:
    rk.append(sum(r["qrank"]) / (len(r["qrank"]) * max(1, r["nq"] - 1)))
rk.sort()
say("top gear's normalised rank by bucket distance, averaged along its path"
    " (0 = always farthest): min %.4f, median %.4f, max %.4f" % (rk[0], rk[len(rk) // 2], rk[-1]))
say("paths where the top gear is the FARTHEST gear at every step: %d of %d" % (
    sum(1 for r in rows if all(x == 0 for x in r["qrank"])), len(rows)))

# ===================================================== item 5: the landing
say("")
say("=== the landing (item 5) ===")
fp = Counter()
fn = Counter()
for r in rows:
    for g in r["flank_prev"]:
        fp[g] += 1
    for g in r["flank_next"]:
        fn[g] += 1
say("strikers of the last blocked column (landing - 1), top 8:", fp.most_common(8))
say("strikers of landing + 1, top 8:", fn.most_common(8))
say("smallest striker of landing - 1 = 5 at %d of %d; of landing + 1 = 5 at %d" % (
    sum(1 for r in rows if r["flank_prev"] and r["flank_prev"][0] == 5), len(rows),
    sum(1 for r in rows if r["flank_next"] and r["flank_next"][0] == 5)))
say("depth of landing - 1: median %d; of landing + 1: median %d" % (
    sorted(len(r["flank_prev"]) for r in rows)[len(rows) // 2],
    sorted(len(r["flank_next"]) for r in rows)[len(rows) // 2]))

# ===================================================== P13: the interaction census
say("")
say("=== the interaction census (P13) ===")
m2 = [len(r["multi"]) for r in rows]
say("gears striking two or more columns of the path: min %d, median %d, max %d"
    % (min(m2), sorted(m2)[len(m2) // 2], max(m2)))
say("their share of the machine: median %.4f" % (
    sorted(len(r["multi"]) / r["nq"] for r in rows)[len(rows) // 2]))
say("minimum blocking set of the path (exact for L <= 45, greedy above):")
mc = sorted(r["mincov"] for r in rows)
say("  size: min %d, median %d, max %d" % (mc[0], mc[len(mc) // 2], mc[-1]))
say("  against L: median mincov/L %.4f" % (
    sorted(r["mincov"] / r["L"] for r in rows)[len(rows) // 2]))
say("  against the number of sole strikers: median %d sole, mincov - sole median %d" % (
    sorted(len(r["sole"]) for r in rows)[len(rows) // 2],
    sorted(r["mincov"] - len(r["sole"]) for r in rows)[len(rows) // 2]))
say("  paths whose minimum blocking set is exactly the sole strikers: %d of %d" % (
    sum(1 for r in rows if r["mincov"] == len(r["sole"])), len(rows)))
ex = sorted(rows, key=lambda r: -r["L"])[:5]
for r in ex:
    say("   q = %d: L = %d, mincov %d, sole %d, gears striking >=2: %d, machine %d"
        % (r["q"], r["L"], r["mincov"], len(r["sole"]), len(r["multi"]), r["nq"]))

# ===================================================== item 3: sensitivity
say("")
say("=== determinism and sensitivity: re-phasing one gear at a time (item 3) ===")
say("for each gear g of the machine, L as a function of g's phase, in two counterfactuals:")
say("  FREE - a runs over all g classes;  REAL - q mod g runs over the g-1 nonzero classes.")


def legendre(a, p):
    a %= p
    if a == 0:
        return 0
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1


sens_rows = []
for r in rows:
    q = r["q"]
    if q > QSENS:
        continue
    L = r["L"]
    marks = r["marks"]
    POS = r["POS"]
    nq = r["nq"]
    qq = q * q
    per = []
    for idx in range(nq):
        g = GEARS[idx]
        u = UU[idx]
        d = r["D"][idx]
        # offsets open when gear g is removed
        opens = [i for i in range(1, POS) if not marks[i] or (len(marks[i]) == 1 and marks[i][0] == g)]
        if not opens:
            continue
        Lfree = set()
        for a in range(g):
            b = (a - d) % g
            for i in opens:
                if i % g != a % g and i % g != b % g:
                    Lfree.add(i)
                    break
        Lreal = set()
        for rr in range(1, g):
            a = ((2 - rr * rr % g) * u) % g
            b = (a - d) % g
            for i in opens:
                if i % g != a % g and i % g != b % g:
                    Lreal.add(i)
                    break
        per.append((g, min(Lfree), max(Lfree), len(Lfree), min(Lreal), max(Lreal), len(Lreal)))
    sens_rows.append((q, L, per))

short_not_sole = 0
never_short = 0
lengthen_all = 0
tot_cells = 0
free_vs_real = []
for q, L, per in sens_rows:
    rr = [x for x in rows if x["q"] == q][0]
    sole = set(rr["sole"])
    for (g, mnF, mxF, nF, mnR, mxR, nR) in per:
        tot_cells += 1
        if mnF < L and g not in sole:
            short_not_sole += 1
        if mxF <= L:
            lengthen_all += 1
        free_vs_real.append((mnF, mnR, mxF, mxR))
say("machines swept for sensitivity: q <= %d (%d paths), gear-cells %d"
    % (QSENS, len(sens_rows), tot_cells))
say("gear-cells where re-phasing SHORTENS L but the gear is not a sole striker:", short_not_sole)
say("gear-cells where no phase LENGTHENS L:", lengthen_all)
say("min L over FREE phases equals min over REAL phases at %d of %d cells" % (
    sum(1 for a, b, c, d in free_vs_real if a == b), len(free_vs_real)))
say("max L over FREE phases equals max over REAL phases at %d of %d cells" % (
    sum(1 for a, b, c, d in free_vs_real if c == d), len(free_vs_real)))
say("mean over cells: min L free %.2f, min L real %.2f, max L free %.2f, max L real %.2f" % (
    sum(a for a, b, c, d in free_vs_real) / len(free_vs_real),
    sum(b for a, b, c, d in free_vs_real) / len(free_vs_real),
    sum(c for a, b, c, d in free_vs_real) / len(free_vs_real),
    sum(d for a, b, c, d in free_vs_real) / len(free_vs_real)))
# the sensitive set
say("")
say("the sensitive set: gears whose re-phasing changes L at all (FREE)")
sizes = []
for q, L, per in sens_rows:
    s = [g for (g, mnF, mxF, nF, mnR, mxR, nR) in per if nF > 1 or mnF != L]
    sh = [g for (g, mnF, mxF, nF, mnR, mxR, nR) in per if mnF < L]
    sizes.append((q, L, len(per), len(s), len(sh)))
say("  q, L, gears, sensitive gears, gears that can SHORTEN L - a sample:")
for t in sizes[::max(1, len(sizes) // 12)]:
    say("   ", t)
say("  sensitive fraction of the machine: median %.4f"
    % sorted(s / g for q, L, g, s, sh in sizes)[len(sizes) // 2])
say("  shortening fraction of the machine: median %.4f"
    % sorted(sh / g for q, L, g, s, sh in sizes)[len(sizes) // 2])
say("  shortening gears vs sole strikers: equal at %d of %d paths" % (
    sum(1 for (q, L, per), (q2, L2, g2, s2, sh2) in zip(sens_rows, sizes)
        if sh2 == len([x for x in rows if x["q"] == q][0]["sole"])), len(sizes)))
LOG.close()
