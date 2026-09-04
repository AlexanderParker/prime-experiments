"""R2.a.i.a.1.a item 4 - the first-moment count, stated as a heuristic.

Model: gear g strikes offset i at its EXACT rate 2 chi_g(i)/(g - 1) (the doubling law, N-R6, which
is exact over the residues of q), and the gears' conditions are treated as INDEPENDENT.  Then

    p(i)   = prod_{7 < g <= q} (1 - 2 chi_g(i)/(g - 1))          [island i free]
    E(q)   = sum_{islands i in [1, d)} p(i)                      [expected free islands]
    P_fail = prod_i (1 - p(i))     and     exp(-E)               [two forms, both heuristic]

Also computed: the DEPTH lambda(i) = sum_g 2 chi_g(i)/(g-1), whose product over the islands is the
naive union bound over covers (one striking gear chosen per island) - the cover-side first moment
the brief names.

Nothing here is a proof; where it is not one is stated in the branch document.

Usage: uv run python research/anchor235/r41/cn_moment.py [--QMAX 20000] [--stride1 5000]
"""
import argparse
import os
from math import isqrt, log, exp

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def sieve(n):
    fl = bytearray([1]) * (n + 1)
    fl[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if fl[i]:
            fl[i * i:: i] = bytearray(len(range(i * i, n + 1, i)))
    return fl


def chi_table(g):
    """chi_g[j] = how many of 2-6j, -6j are NONZERO quadratic residues mod g (j = offset mod g)."""
    qr = np.zeros(g, dtype=np.int8)
    x = np.arange(1, g, dtype=np.int64)
    qr[(x * x) % g] = 1
    j = np.arange(g, dtype=np.int64)
    return (qr[(2 - 6 * j) % g] + qr[(-6 * j) % g]).astype(np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--QMAX", type=int, default=20000)
    ap.add_argument("--dense", type=int, default=5000)
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--tag", type=str, default="moment")
    args = ap.parse_args()
    LOG = open(os.path.join(OUT, "cn_%s.txt" % args.tag), "w")

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        LOG.write(s + "\n")
        LOG.flush()

    FL = sieve(args.QMAX + 10)
    PR = [p for p in range(11, args.QMAX + 1) if FL[p]]
    CT = {}

    def get(g):
        if g not in CT:
            CT[g] = chi_table(g)
        return CT[g]

    qs = []
    for q in range(11, args.QMAX + 1, 2):
        if q % 3 == 0 or q % 5 == 0:
            continue
        if q <= args.dense or (q // 2) % args.stride == 0:
            qs.append(q)
    say("# first moment, %d values of q coprime to 30 (all to %d, then 1 in %d)"
        % (len(qs), args.dense, args.stride))
    say("#  q      d      m      E[free]   P_fail=prod(1-p)  exp(-E)     depth mean  union bound"
        " log10")
    rows = []
    for q in qs:
        d = (q + 1) // 3 if q % 6 == 5 else (2 * q + 1) // 3
        isl = np.array([i for i in range(1, d) if i % 35 in (5, 10, 12, 17)], dtype=np.int64)
        m = len(isl)
        if m == 0:
            continue
        logp = np.zeros(m)
        lam = np.zeros(m)
        for g in PR:
            if g > q:
                break
            ct = get(g)
            chi = ct[isl % g].astype(np.float64)
            r = 2.0 * chi / (g - 1)
            logp += np.log1p(-r)
            lam += r
        p = np.exp(logp)
        E = float(p.sum())
        Pf = float(np.exp(np.log1p(-p).sum()))
        rows.append((q, d, m, E, Pf, exp(-E), float(lam.mean()),
                     float(np.log10(np.maximum(lam, 1e-9)).sum())))
    for q, d, m, E, Pf, Pe, lm, ub in rows:
        if q <= 400 or q % 500 < 20 or Pf > 1e-4:
            say("  %-6d %-6d %-6d %-9.4f %-17.4g %-11.4g %-11.4f %.1f"
                % (q, d, m, E, Pf, Pe, lm, ub))
    say("")
    # weights: with a stride above args.dense each sampled q stands for `stride` values
    say("# expected number of failing integers q coprime to 30, by band (weighted for the stride)")
    say("#  band                 sampled q   sum P_fail   observed failures (coprime to 30)")
    bands = [(11, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 10000), (10000, 20000)]
    for lo, hi in bands:
        s = 0.0
        n = 0
        for q, d, m, E, Pf, Pe, lm, ub in rows:
            if lo <= q < hi:
                w = 1.0 if q <= args.dense else args.stride
                s += w * Pf
                n += 1
        say("   [%-6d %-6d )       %-11d %-12.4g" % (lo, hi, n, s))
    say("")
    tail = {}
    for cut in (1000, 1500, 2000, 2849, 3000, 5000, 10000):
        s = 0.0
        for q, d, m, E, Pf, Pe, lm, ub in rows:
            if q > cut:
                w = 1.0 if q <= args.dense else args.stride
                s += w * Pf
        tail[cut] = s
        say("# expected failures above q = %-6d : %.4g" % (cut, s))
    say("")
    big = [r for r in rows if r[4] > 1e-3]
    say("# largest q with P_fail > 1e-3: %s ; > 1e-4: %s ; > 1e-6: %s"
        % (max((r[0] for r in rows if r[4] > 1e-3), default=None),
           max((r[0] for r in rows if r[4] > 1e-4), default=None),
           max((r[0] for r in rows if r[4] > 1e-6), default=None)))
    say("# (observed: the last failing integer coprime to 30 is q = 2849; none to 200,000)")
    say("")
    say("# the cover-side first moment: union bound over covers = prod_i lambda(i), lambda the")
    say("# depth.  log10 of it at a few q (it must be < 0 to say anything):")
    for q, d, m, E, Pf, Pe, lm, ub in rows:
        if q in (101, 251, 503, 1009, 2003, 5003, 10007, 19997) or q in (137, 1487, 2849):
            say("   q = %-6d m = %-5d mean depth %.4f   log10(prod lambda) = %+.1f" % (q, m, lm, ub))
    LOG.close()


main()
