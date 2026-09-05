"""R3.h.i part 5 - the inverse-shape test: is any bucket statistic at x a bound on the walk?

At an opening x the forward bucket b_g^+ is the distance to gear g's next tooth.  For ONE gear
the bucket IS the distance to its tooth; for the walk it is the first offset every bucket class
misses.  Tested here, at every junction of m11..m23 and at every opening of m17:

  A. L^+ <= sum of the k smallest forward buckets, for k = 1..|M| - the smallest k that works
  B. L^+ >= b_(1)                                        (the nearest tooth)
  C. S   <= sum of the k smallest (b^+ + b^-)
  D. the missing-gear (umbrella) bound S <= long arc of any gear missing the stretch, minus 2

Writes results/fw_shape.txt.
"""
import os
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "fw_shape.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31]
MACHINES = [11, 13, 17, 19, 23]


def build(q):
    gs = [p for p in PRIMES if p <= q]
    P = 1
    for g in gs:
        P *= g
    ba = bytearray(P)
    one = b"\x01"
    for g in gs:
        u = pow(6, -1, g)
        for t in (u, g - u):
            ba[t::g] = one * len(range(t, P, g))
    return gs, P, ba


for q in MACHINES:
    gs, P, ba = build(q)
    qp = next(p for p in PRIMES if p > q)
    up = pow(6, -1, qp)
    teeth = (up % qp, (-up) % qp)
    U = {g: pow(6, -1, g) for g in gs}
    op = array("l")
    pos = ba.find(0)
    while pos != -1:
        op.append(pos)
        pos = ba.find(0, pos + 1)
    n = len(op)
    gaps = array("l", bytes(4 * n))
    for i in range(n - 1):
        gaps[i] = op[i + 1] - op[i]
    gaps[n - 1] = P - op[n - 1] + op[0]

    targets = [i for i in range(n) if op[i] % qp in teeth]
    label = "junctions"
    if q == 17:
        targets = list(range(n))
        label = "all openings"

    kneedA = [0] * (len(gs) + 2)
    kneedC = [0] * (len(gs) + 2)
    excB = 0
    worstA = None
    for i in targets:
        x = op[i]
        lp = gaps[i]
        lm = gaps[(i - 1) % n]
        S = lm + lp
        bp = []
        bs = []
        for g in gs:
            u = U[g]
            a = min((u - x) % g, (-u - x) % g)
            b = min((x - u) % g, (x + u) % g)
            bp.append(a)
            bs.append(a + b)
        bp.sort()
        bs.sort()
        if lp < bp[0]:
            excB += 1
        t = 0
        k = 0
        while k < len(bp) and t < lp:
            t += bp[k]
            k += 1
        if t < lp:
            k = len(gs) + 1
        kneedA[k] += 1
        if worstA is None or k > worstA[0]:
            worstA = (k, x, lm, lp)
        t = 0
        k = 0
        while k < len(bs) and t < S:
            t += bs[k]
            k += 1
        if t < S:
            k = len(gs) + 1
        kneedC[k] += 1

    say("")
    say("MACHINE m%d (%d gears), %d %s" % (q, len(gs), len(targets), label))
    say("  A. smallest k with L^+ <= sum of k smallest forward buckets: histogram %s "
        "(k = %d means no k works)" % ({i: c for i, c in enumerate(kneedA) if c}, len(gs) + 1))
    say("     k = 2 suffices at %d of %d (%.4f); k = 3 at %d; all gears needed or failing at %d"
        % (sum(kneedA[:3]), len(targets), sum(kneedA[:3]) / len(targets), sum(kneedA[:4]),
           kneedA[len(gs)] + kneedA[len(gs) + 1]))
    say("     worst cell: k = %d at x=%d (%d, %d)" % worstA)
    say("  B. L^+ >= nearest forward tooth b_(1): %d exceptions" % excB)
    say("  C. smallest k with S <= sum of k smallest (b^+ + b^-): histogram %s"
        % {i: c for i, c in enumerate(kneedC) if c})
    del ba, op, gaps
LOG.close()
