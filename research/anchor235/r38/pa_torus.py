"""Branch W.a part 3: the path on the torus (item 6).

The path depends only on the residue vector (q mod g)_{g <= q}, and only through q^2 mod g,
which is a quadratic residue - so the walk from a SQUARE lives on a sub-torus of the phase
space.  Two comparisons:

 (a) the tooth-start record: walks under the SAME machine {5..q} started from the other teeth
     of the top gear inside the window, where the phases are unrestricted;
 (b) the residue tabulation: L against q^2 mod 5, mod 7, mod 35 and mod 11.

Writes results/pa_torus.txt.
"""
import os
from math import isqrt
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "pa_torus.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


QMAX = 3000
NSTART = 150


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
say("gears 5..%d: %d" % (QMAX, len(GEARS)))


def walk_from(t, nq, POS=384):
    """first i >= 1 with column t + i unstruck by any gear GEARS[:nq]"""
    while True:
        blocked = bytearray(POS)
        for idx in range(nq):
            g = GEARS[idx]
            u = UU[idx]
            for off in ((u - t) % g, ((-u) - t) % g):
                j = off
                while j < POS:
                    blocked[j] = 1
                    j += g
        for i in range(1, POS):
            if not blocked[i]:
                return i
        POS *= 2


rows = []
for qi, q in enumerate(GEARS):
    nq = qi + 1
    k0 = (q * q - 1) // 6
    L0 = walk_from(k0, nq)
    u = UU[qi]
    # tooth columns of the top gear inside the window (q/6, k0]
    lo = q // 6 + 1
    teeth = []
    for cls in (u % q, (-u) % q):
        s = lo + ((cls - lo) % q)
        teeth.extend(range(s, k0, q))
    teeth.sort()
    if len(teeth) > NSTART:
        step = len(teeth) / NSTART
        teeth = [teeth[int(i * step)] for i in range(NSTART)]
    Ls = [walk_from(t, nq) for t in teeth]
    Ls.sort()
    if Ls:
        below = sum(1 for x in Ls if x < L0)
        eq = sum(1 for x in Ls if x == L0)
        pct = (below + 0.5 * eq) / len(Ls)
    else:
        pct = None
    rows.append(dict(q=q, L0=L0, pct=pct, n=len(Ls),
                     med=Ls[len(Ls) // 2] if Ls else None,
                     mean=sum(Ls) / len(Ls) if Ls else None,
                     mx=Ls[-1] if Ls else None,
                     l1=sum(1 for x in Ls if x == 1) if Ls else 0))

say("machines: %d; tooth starts per machine: up to %d" % (len(rows), NSTART))

say("")
say("=== (a) the tooth-start record: the square start against the other teeth ===")
ok = [r for r in rows if r["pct"] is not None and r["n"] >= 20]
p = sorted(r["pct"] for r in ok)
say("machines compared: %d (>= 20 tooth starts each)" % len(ok))
say("percentile of L(q^2) among the tooth-start walks: min %.3f, median %.3f, mean %.3f, max %.3f"
    % (p[0], p[len(p) // 2], sum(p) / len(p), p[-1]))
say("L(q^2) strictly above the tooth-start median at %d of %d machines (%.4f)"
    % (sum(1 for r in ok if r["L0"] > r["med"]), len(ok),
       sum(1 for r in ok if r["L0"] > r["med"]) / len(ok)))
say("mean L(q^2) %.3f against mean tooth-start L %.3f (ratio %.4f)" % (
    sum(r["L0"] for r in ok) / len(ok), sum(r["mean"] for r in ok) / len(ok),
    (sum(r["L0"] for r in ok) / len(ok)) / (sum(r["mean"] for r in ok) / len(ok))))
say("tooth starts with L = 1 (landing at the very next column): %d of %d starts (%.4f);"
    " square starts with L = 1: %d of %d" % (
        sum(r["l1"] for r in ok), sum(r["n"] for r in ok),
        sum(r["l1"] for r in ok) / sum(r["n"] for r in ok),
        sum(1 for r in ok if r["L0"] == 1), len(ok)))
say("sample (q, L(q^2), tooth-start median, tooth-start mean, percentile):")
for r in ok[::max(1, len(ok) // 14)]:
    say("   %5d  %4d  %4d  %7.2f  %.3f" % (r["q"], r["L0"], r["med"], r["mean"], r["pct"]))

say("")
say("=== (b) the residue tabulation (item 6) ===")
say("the path's phase at gear g is q^2 mod g, a quadratic residue: (g-1)/2 of the g classes.")
pr = 1.0
for g in GEARS[:6]:
    pr *= (g - 1) / (2.0 * g)
say("density of the reachable sub-torus over the first six gears (5..17): %.5f" % pr)
for mod, name in ((5, "q^2 mod 5"), (7, "q^2 mod 7"), (11, "q^2 mod 11"), (35, "q^2 mod 35")):
    d = {}
    for r in rows:
        if r["q"] <= mod:
            continue
        c = (r["q"] * r["q"]) % mod
        d.setdefault(c, []).append(r["L0"])
    say("  %s:" % name)
    for c in sorted(d):
        v = sorted(d[c])
        say("    class %2d : n = %4d, median L = %3d, mean L = %6.2f, max %4d"
            % (c, len(v), v[len(v) // 2], sum(v) / len(v), v[-1]))
LOG.close()
