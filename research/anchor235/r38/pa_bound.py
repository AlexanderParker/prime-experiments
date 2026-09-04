"""Branch W.a part 6: which SIZE of gear the path needs - the columns no small gear blocks."""
import os
from math import isqrt
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "pa_bound.txt"), "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n")
QMAX = 20000
def sieve(n):
    fl = bytearray([1]) * (n + 1); fl[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if fl[i]: fl[i*i::i] = bytearray(len(range(i*i, n+1, i)))
    return fl
FL = sieve(QMAX + 10)
GEARS = [p for p in range(5, QMAX + 1) if FL[p]]
UU = [pow(6, -1, g) for g in GEARS]
tot_cols = 0; bigonly = 0; bigmin = 0
rows = []
for qi, q in enumerate(GEARS):
    nq = qi + 1; qq = q * q; POS = 768
    while True:
        marks = [[] for _ in range(POS)]
        for idx in range(nq):
            g = GEARS[idx]; u = UU[idx]; r = qq % g
            for off in (((2-r)*u) % g, ((-r)*u) % g):
                j = off
                while j < POS:
                    marks[j].append(g); j += g
        L = None
        for i in range(1, POS):
            if not marks[i]: L = i; break
        if L is not None: break
        POS *= 2
    s = isqrt(q)
    bo = sum(1 for i in range(L) if marks[i] and marks[i][0] > s)
    tot_cols += L; bigonly += bo
    # walk under the small machine {5..sqrt(q)} only
    Ls = next((i for i in range(1, POS) if not [g for g in marks[i] if g <= s]), POS)
    # walk under gears <= q/2, <= q/10
    Lh = next((i for i in range(1, POS) if not [g for g in marks[i] if g <= q // 2]), POS)
    rows.append((q, L, bo, Ls, Lh))
say("primes: %d" % len(rows))
say("")
say("=== columns of the path that only a gear above sqrt(q) blocks ===")
say("total path columns %d; columns whose SMALLEST striker exceeds sqrt(q): %d (%.4f)"
    % (tot_cols, bigonly, bigonly / tot_cols))
say("paths with at least one such column: %d of %d (%.4f)"
    % (sum(1 for r in rows if r[2] > 0), len(rows), sum(1 for r in rows if r[2] > 0)/len(rows)))
say("such columns per path: median %d, max %d" % (
    sorted(r[2] for r in rows)[len(rows)//2], max(r[2] for r in rows)))
say("")
say("=== how far the SMALL machine alone takes the walk ===")
rs = [r for r in rows if r[0] >= 11]
say("L(gears <= sqrt(q)) against L(full): equal at %d of %d; median small-walk %d vs full %d"
    % (sum(1 for r in rs if r[3] == r[1]), len(rs),
       sorted(r[3] for r in rs)[len(rs)//2], sorted(r[1] for r in rs)[len(rs)//2]))
say("L(gears <= q/2) equals L(full) at %d of %d" % (
    sum(1 for r in rs if r[4] == r[1]), len(rs)))
say("sample (q, L, big-only columns, L under gears<=sqrt(q), L under gears<=q/2):")
for r in rs[::max(1, len(rs)//14)]:
    say("   ", r)
LOG.close()
