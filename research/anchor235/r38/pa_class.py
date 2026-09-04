"""Branch W.a part 5: L against the residue class of q, over the whole sweep; the
class-restriction rule for L mod 35; and the character of the no-lengthening gear cells."""
import os
from math import isqrt, gcd
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "pa_class.txt"), "w")
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

def walkL(qi, q, POS=768):
    nq = qi + 1; qq = q * q
    while True:
        bl = bytearray(POS)
        for idx in range(nq):
            g = GEARS[idx]; u = UU[idx]; r = qq % g
            for off in (((2 - r) * u) % g, ((-r) * u) % g):
                j = off
                while j < POS:
                    bl[j] = 1; j += g
        for i in range(1, POS):
            if not bl[i]:
                return i
        POS *= 2

L = {}
for qi, q in enumerate(GEARS):
    L[q] = walkL(qi, q)
say("primes swept: %d (5..%d)" % (len(L), QMAX))

# open-offset sets of {5,7} per class of q^2 mod 35
OPEN = {}
for c in range(1, 35):
    if gcd(c, 35) != 1 or not any((r*r) % 35 == c for r in range(1, 35)): continue
    o5 = {i % 5 for i in range(5) if (-6*i) % 5 == c % 5 or (2-6*i) % 5 == c % 5}
    o7 = {i % 7 for i in range(7) if (-6*i) % 7 == c % 7 or (2-6*i) % 7 == c % 7}
    OPEN[c] = sorted(i for i in range(35) if i % 5 not in o5 and i % 7 not in o7)

say("")
say("=== L mod 35 lies in the 15-class set fixed by q^2 mod 35 (rule N-W3) ===")
bad = 0
for q, l in L.items():
    if q <= 7: continue
    c = (q * q) % 35
    if l % 35 not in OPEN[c]:
        bad += 1
say("exceptions over %d primes q >= 11:" % sum(1 for q in L if q > 7), bad)

say("")
say("=== L by residue class, full sweep ===")
for mod in (5, 7, 35):
    d = {}
    for q, l in L.items():
        if q <= mod: continue
        d.setdefault((q*q) % mod, []).append(l)
    say("  q^2 mod %d:" % mod)
    for c in sorted(d):
        v = sorted(d[c])
        say("    class %2d : n = %4d, median %3d, mean %6.2f, 90th pct %4d, max %4d, "
            "open offsets 1..35 = %s" % (c, len(v), v[len(v)//2], sum(v)/len(v),
            v[int(.9*len(v))], v[-1], OPEN[c][:6] if mod == 35 else ""))

# ---- the no-lengthening cells
say("")
say("=== gear cells where no re-phasing of that gear lengthens L ===")
tot = 0; notsole = 0; ex = []
for qi, q in enumerate(GEARS):
    if q > 300: continue
    nq = qi + 1; qq = q * q; POS = 512
    marks = [[] for _ in range(POS)]
    A = [0]*nq; D = [0]*nq
    for idx in range(nq):
        g = GEARS[idx]; u = UU[idx]; r = qq % g
        a = ((2-r)*u) % g; b = ((-r)*u) % g
        A[idx] = a; D[idx] = (2*u) % g
        for off in (a, b):
            j = off
            while j < POS:
                marks[j].append(g); j += g
    l = L[q]
    sole = {marks[i][0] for i in range(l) if len(marks[i]) == 1}
    solecols = {}
    for i in range(l):
        if len(marks[i]) == 1:
            solecols.setdefault(marks[i][0], []).append(i)
    for idx in range(nq):
        g = GEARS[idx]; d = D[idx]
        opens = [i for i in range(1, POS) if not marks[i] or (len(marks[i]) == 1 and marks[i][0] == g)]
        mx = 0
        for a in range(g):
            b = (a - d) % g
            for i in opens:
                if i % g != a % g and i % g != b % g:
                    mx = max(mx, i); break
        if mx <= l:
            tot += 1
            if g not in sole:
                notsole += 1
                if len(ex) < 5: ex.append((q, g))
            else:
                if len(ex) < 5 and False: pass
say("cells with no lengthening phase (q <= 300): %d; of those, gears that are NOT sole"
    " strikers of the path: %d %s" % (tot, notsole, ex))
say("mechanism: a sole striker of >= 3 offsets in distinct classes cannot both keep them")
say("blocked and block the landing with only two teeth.")
LOG.close()
