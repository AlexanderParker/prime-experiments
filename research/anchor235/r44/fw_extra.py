"""R3.h.i part 4 - what the window's two junctions are, and the window correlation detrended.

E1: the junctions of the window of M = {5..q} are exactly
      - the column of q'   (an opening iff q' is a member of a twin pair), and
      - the column of q'^2 (an opening iff q'^2 - 2 is prime, i.e. W.a's square gate),
    and nothing else.
E2: at the q'^2 junction the forward flank in {5..q} is the walk length L(q') of branch W.a
    (the walk from q'^2 under {5..q'}), and the backward flank is W.t's L^-(q').
E3: the window's flank correlation, detrended against the local twin density.

Writes results/fw_extra.txt.
"""
import os
from math import isqrt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "fw_extra.txt"), "w")


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
MARGIN = 800


def isprime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


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


e1 = e2 = 0
nA = nB = 0
detr = []
raw = []
tabA = []
tabB = []
for q in RUNGS:
    qp = PR[PR.index(q) + 1]
    gsq = [p for p in PR if 5 <= p <= q]
    U = {g: pow(6, -1, g) for g in gsq}
    lo = q // 6 + 1
    hi = (qp * qp - 1) // 6
    A, B = lo - MARGIN, hi + MARGIN
    n = B - A + 1
    ba = bytearray(n)
    one = b"\x01"
    for g in gsq:
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
    up = pow(6, -1, qp)
    teeth = (up % qp, (-up) % qp)
    jl = [v for v in winop if v % qp in teeth]

    colq = (qp + 1) // 6 if (qp % 6 == 5) else (qp - 1) // 6
    colq2 = (qp * qp + 1) // 6 if (qp * qp % 6 == 5) else (qp * qp - 1) // 6
    predicted = []
    if isprime(qp - 2) or isprime(qp + 2):
        if colq in winop:
            predicted.append(colq)
    if isprime(qp * qp - 2):
        if colq2 in winop:
            predicted.append(colq2)
    if sorted(jl) != sorted(predicted):
        e1 += 1
        say("E1 mismatch at q=%d: junctions %s predicted %s" % (q, sorted(jl), sorted(predicted)))

    # E2 the q'^2 junction against the W.a walk under {5..q'}
    if colq2 in jl:
        nB += 1
        i = idx[colq2]
        lp = op[i + 1] - colq2
        lm = colq2 - op[i - 1]
        uq = pow(6, -1, qp)
        # walk under {5..q'} from colq2 (blocked there by q')

        def blocked_big(c):
            if (c - uq) % qp == 0 or (c + uq) % qp == 0:
                return True
            for g in gsq:
                if (c - U[g]) % g == 0 or (c + U[g]) % g == 0:
                    return True
            return False
        j = 1
        while blocked_big(colq2 + j):
            j += 1
        k = 1
        while blocked_big(colq2 - k):
            k += 1
        if j != lp or k != lm:
            e2 += 1
        tabB.append((q, qp, lm, lp, lm + lp, j, k))
    if colq in jl:
        nA += 1
        i = idx[colq]
        tabA.append((q, qp, colq - op[i - 1], op[i + 1] - colq))

    # E3 detrended correlation: divide each gap by the mean gap of a local block
    pairs = []
    for v in winop:
        i = idx[v]
        pairs.append((v, v - op[i - 1], op[i + 1] - v))
    raw.extend([(a, b) for (_, a, b) in pairs])
    BLK = 200
    for s in range(0, len(pairs) - BLK, BLK):
        blk = pairs[s:s + BLK]
        m = sum(a + b for (_, a, b) in blk) / (2 * len(blk))
        detr.extend([(a / m, b / m) for (_, a, b) in blk])

say("")
say("E1 the window's junctions are exactly {column of q' if q' is a twin member} u "
    "{column of q'^2 if q'^2 - 2 is prime}: %d mismatches over %d rungs" % (e1, len(RUNGS)))
say("   q'-column junctions %d, q'^2-column junctions %d, of %d rungs" % (nA, nB, len(RUNGS)))
say("E2 at the q'^2 junction the flanks in {5..q} equal the two-sided walk from q'^2 under "
    "{5..q'}: %d mismatches of %d" % (e2, nB))
say("E3 window flank correlation: raw %+.5f (n=%d) | detrended by local density %+.5f (n=%d)"
    % (pearson(raw), len(raw), pearson(detr), len(detr)))
say("")
say("THE q' COLUMN (bottom junction of the window): q, q', L^-, L^+, S")
for t in tabA:
    say("   %-5d %-5d %-4d %-4d %d" % (t[0], t[1], t[2], t[3], t[2] + t[3]))
say("")
say("THE q'^2 COLUMN (top junction of the window): q, q', L^-, L^+, S  [W.a walk L(q') = L^+]")
for t in tabB:
    say("   %-5d %-5d %-4d %-4d %-4d   (W.a forward %d, backward %d)"
        % (t[0], t[1], t[2], t[3], t[4], t[5], t[6]))
LOG.close()
