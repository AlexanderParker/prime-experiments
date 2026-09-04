"""R2.a.i.a - part 2b: the ORDER-ONE null for the landing's island status.

The naive conditional null "P(barred | not struck) = |Bar(g)|/(g-2)" is the wrong null: the
landing is not a random missed offset, it is the FIRST missed offset, and an offset that more
gears are barred from is more likely to be missed by all of them.  The correct order-one null
keeps every gear independent and asks for the first-passage distribution:

    p_g(i) = 2 chi_g(i) / (g - 1)          (g strikes offset i for 2 chi of the g-1 classes of q)
    pi_q(i) = prod_{5<=g<=q} (1 - p_g(i))  (offset i open)
    P(L = i)  =  pi_q(i) * prod_{1<=j<i} (1 - pi_q(j))          [offsets independent]

This script computes that null for every prime q <= 20000 and compares its island rate, its
landing histogram and its lambda profile with the measured ones.  Any excess of the MEASURED
over this null is structure of order two or more; agreement means the landscape's preference
for islands is exactly the per-gear bar and nothing else.

Writes results/rl_null.txt.
Usage: uv run python research/anchor235/r39/rl_null.py
"""
import os
from collections import Counter
from math import isqrt

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
LOG = open(os.path.join(OUT, "rl_null.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


QMAX = 20000
IW = 1600            # offsets 0..IW-1 carried in the null (max measured L is 402)


def sieve(n):
    fl = bytearray([1]) * (n + 1)
    fl[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if fl[i]:
            fl[i * i:: i] = bytearray(len(range(i * i, n + 1, i)))
    return fl


FL = sieve(QMAX + 10)
GEARS = [p for p in range(5, QMAX + 1) if FL[p]]
say("gears 5..%d: %d;  null carried over offsets 0..%d" % (QMAX, len(GEARS), IW - 1))

ISL = {}
for B in (5, 7, 11, 13, 17, 19):
    res = np.load(os.path.join(OUT, "rl_isl_%d.npy" % B))
    mod = int(open(os.path.join(OUT, "rl_isl_%d_mod.txt" % B)).read())
    ISL[B] = (set(int(v) for v in res), mod)
ISLMASK = {}
I = np.arange(IW)
for B, (res, mod) in ISL.items():
    ISLMASK[B] = np.array([(int(i) % mod) in res for i in I])

lam_glob = np.load(os.path.join(OUT, "rl_lambda.npy"))[:IW]
rows = np.load(os.path.join(OUT, "rl_rows.npy"))          # (q, L, d)
measL = {int(r[0]): int(r[1]) for r in rows}

logpi = np.zeros(IW)          # running sum of log(1 - p_g(i)) as gears are added
acc = {B: 0.0 for B in ISLMASK}
nullhist = np.zeros(IW)
nlam = 0.0
nq = 0
for g in GEARS:
    qr = np.zeros(g, dtype=bool)
    for t in range(1, (g + 1) // 2):
        qr[(t * t) % g] = True
    chi = qr.astype(np.int64) + np.roll(qr, -2).astype(np.int64)
    p = 2.0 * chi[(-6 * I) % g] / (g - 1)
    with np.errstate(divide="ignore"):
        logpi += np.log(np.maximum(1.0 - p, 1e-300))
    # first-passage distribution over offsets 1..IW-1 for the machine {5..g}
    pi = np.exp(logpi)
    surv = np.cumprod(1.0 - pi[1:])
    pl = np.empty(IW)
    pl[0] = 0.0
    pl[1] = pi[1]
    pl[2:] = pi[2:] * surv[:-1]
    s = pl.sum()
    if s <= 0:
        continue
    pl /= s
    nullhist += pl
    for B in ISLMASK:
        acc[B] += float((pl * ISLMASK[B]).sum())
    nlam += float((pl * lam_glob).sum())
    nq += 1

say("machines covered by the null: %d" % nq)
say("")
say("=== the landing's island rate: measured against the order-one first-passage null ===")
say(" B    measured   order-1 null   naive cond. null   island density   meas/null")
for B in (5, 7, 11, 13, 17, 19):
    meas = sum(1 for r in rows if ISLMASK[B][int(r[1])]) / len(rows)
    nul = acc[B] / nq
    naive = 1.0
    rho = 1.0
    for g in GEARS:
        if g > B:
            break
        qr = np.zeros(g, dtype=bool)
        for t in range(1, (g + 1) // 2):
            qr[(t * t) % g] = True
        chi = qr.astype(np.int64) + np.roll(qr, -2).astype(np.int64)
        bs = int((chi[(-6 * np.arange(g)) % g] == 0).sum())
        naive *= bs / (g - 2)
        rho *= bs / g
    say("%3d   %.4f     %.4f         %.4f             %.6f         %.3f"
        % (B, meas, nul, naive, rho, meas / nul if nul else float("nan")))

say("")
say("=== the landing histogram: measured against the order-one null (top offsets) ===")
mh = Counter(int(r[1]) for r in rows)
nullhist = nullhist / nq * len(rows)
say("offset   measured   null      meas/null   island B=7   lambda")
for i in [x for x, _ in mh.most_common(16)]:
    say("%6d   %8d   %7.1f   %7.2f     %-10s   %.4f"
        % (i, mh[i], nullhist[i], mh[i] / nullhist[i] if nullhist[i] else float("nan"),
           bool(ISLMASK[7][i]), lam_glob[i]))
say("")
say("mean lambda(L): measured %.4f, order-1 null %.4f"
    % (np.mean([lam_glob[int(r[1])] for r in rows]), nlam / nq))
say("mean L: measured %.3f, order-1 null %.3f"
    % (np.mean([int(r[1]) for r in rows]),
       float((nullhist * np.arange(IW)).sum() / nullhist.sum())))
LOG.close()
