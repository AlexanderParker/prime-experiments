"""Round 20 (constructor): enumerate ALL maximal-depth qualifying runs.

At machine 29 the residue-qualifying runs of length 3 (the k=4 fuel) number
exactly 8 per period (tm_resid_runs.py).  This tool lists every one, with its
word, its two flanking gaps, and the full 5-gap window sum - the complete
inventory of the deepest fuel, exact.

Usage: uv run python research/tm_deepruns.py y m [--seg N]
Lists all runs of exactly-m consecutive qualifying gaps (maximal or not is
reported), with flanks.
"""
import sys
import numpy as np
from math import prod
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flank_envelope import primes_upto
from tm_resid_runs import next_prime

CTX = 24


def main():
    args = sys.argv[1:]
    seg = 256_000_000
    if "--seg" in args:
        i = args.index("--seg")
        seg = int(float(args[i + 1]))
        del args[i:i + 2]
    y, m = int(args[0]), int(args[1])
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    q1 = next_prime(y)
    c = pow(6, -1, q1)
    Qres = np.array(sorted({0, (2 * c) % q1, (-2 * c) % q1}))
    uvals = [pow(6, -1, g) for g in gears]
    print(f"machine {y}, q'={q1}, Qres={Qres.tolist()}, runs of m={m}")
    found = []
    tail = None
    head = None
    for lo in range(0, P, seg):
        hi = min(P, lo + seg)
        ex = np.zeros(hi - lo, bool)
        for g, u in zip(gears, uvals):
            ex[(u - lo) % g::g] = True
            ex[(-u - lo) % g::g] = True
        op = np.flatnonzero(~ex).astype(np.int64) + lo
        if head is None:
            head = op[:CTX].copy()
        ops = op if tail is None else np.concatenate([tail, op])
        scan(ops, lo, m, q1, Qres, found)
        tail = ops[-CTX:].copy()
    scan(np.concatenate([tail, head + P]), P, m, q1, Qres, found)
    print(f"total runs of length {m}: {len(found)}")
    for k0, word, fl, fr, s in sorted(found):
        print(f"  at opening {k0:>14,}: flanks ({fl},{fr})  word {word}  "
              f"window sum {s}  (span {sum(word)})")


def scan(ops, lo_new, m, q1, Qres, found):
    d = np.diff(ops)
    n = len(d)
    if n < m + 2:
        return
    new = ops[1:] >= lo_new
    qual = np.isin(d % q1, Qres)
    ok = qual[: n - m + 1].copy()
    for t in range(1, m):
        ok &= qual[t: n - m + 1 + t]
    ok &= new[m - 1: n]
    for i in np.flatnonzero(ok):
        if i == 0 or i + m >= n:
            continue  # flanks unavailable (only at stream edges; seam pass covers)
        word = tuple(int(x) for x in d[i: i + m])
        fl, fr = int(d[i - 1]), int(d[i + m])
        s = fl + fr + sum(word)
        found.append((int(ops[i]), word, fl, fr, s))


if __name__ == "__main__":
    main()
