"""R2.a.i.a.1.a item 1 - heuristic UPPER bounds on K(d) where the exact ILP is out of reach.

Randomised greedy with restarts over the same complete candidate list the exact solver uses
(gear at one reachable phase, sets of size >= 2, plus generic singletons).  Every value printed is
an achieved cover, so it is a genuine upper bound on K(d); it is never claimed to be K(d).

Usage: uv run python research/anchor235/r41/cn_heur.py --DS 1120,2240,4480 --restarts 400
"""
import argparse
import os
import time
from math import isqrt, log

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


def build(d, PR):
    isl = [i for i in range(1, d) if i % 35 in (5, 10, 12, 17)]
    m = len(isl)
    idx = {i: k for k, i in enumerate(isl)}
    cand = []                       # (gear, bitmask)
    for g in PR:
        if g > 3 * d + 2:
            break
        buck = {}
        for i in isl:
            buck.setdefault(i % g, []).append(idx[i])
        u = pow(6, -1, g)
        h = (g - 1) // 2
        rs = set()
        for i in isl:
            for r in (((-6 * i) % g), ((2 - 6 * i) % g)):
                if r and pow(r, h, g) == 1:
                    rs.add(r)
        seen = set()
        for r in rs:
            s = buck.get(((2 - r) * u) % g, []) + buck.get(((-r) * u) % g, [])
            if len(s) >= 2:
                fs = frozenset(s)
                if fs in seen:
                    continue
                seen.add(fs)
                mk = 0
                for e in fs:
                    mk |= 1 << e
                cand.append((g, mk, len(fs)))
    return isl, m, cand


def greedy(m, cand, rng, noise):
    full = (1 << m) - 1
    cov = 0
    used = set()
    K = 0
    while cov != full:
        best, bj = -1, -1
        for j, (g, mk, sz) in enumerate(cand):
            if g in used:
                continue
            gain = bin(mk & ~cov).count("1")
            if gain < 2:
                continue
            score = gain + noise * rng.random()
            if score > best:
                best, bj = score, j
        if bj < 0:
            break
        g, mk, sz = cand[bj]
        used.add(g)
        cov |= mk
        K += 1
    K += bin(full & ~cov).count("1")     # generic singletons for whatever is left
    return K, sorted(used)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--DS", type=str, default="1120,2240,4480")
    ap.add_argument("--restarts", type=int, default=200)
    ap.add_argument("--tag", type=str, default="heur")
    args = ap.parse_args()
    DS = [int(v) for v in args.DS.split(",")]
    LOG = open(os.path.join(OUT, "cn_%s.txt" % args.tag), "w")

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        LOG.write(s + "\n")
        LOG.flush()

    FL = sieve(3 * max(DS) + 10)
    PR = [p for p in range(11, 3 * max(DS) + 3) if FL[p]]
    rng = np.random.default_rng(2026)
    say("#  d      m     cand    best UB   K*(lnd)^3/d   restarts   secs")
    for d in DS:
        t0 = time.time()
        isl, m, cand = build(d, PR)
        best, bestset = 10 ** 9, None
        for t in range(args.restarts):
            noise = 0.0 if t == 0 else (0.5 + 2.0 * rng.random())
            K, us = greedy(m, cand, rng, noise)
            if K < best:
                best, bestset = K, us
        say("  %-6d %-5d %-7d %-9d %-13.3f %-10d %.1f"
            % (d, m, len(cand), best, best * log(d) ** 3 / d, args.restarts, time.time() - t0))
        say("      cover: %s" % bestset)
    LOG.close()


main()
