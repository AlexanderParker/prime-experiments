"""u45_sieve.py -- the neighbour-sum profile N(v) and the J-run outer law, by DIRECT SIEVE.

Reference instrument for branch U4/U5.  It sieves the full period of the machine {5..y} in column
chunks and streams the cyclic gap sequence through four accumulators:

  * the gap spectrum  m(v)                       (the census c(d) = sum_{v>=d} m(v) of U5)
  * F_2 = max over adjacent gap pairs of (g_i + g_{i+1})
  * N(v) = max over gaps of size v of (left neighbour gap + right neighbour gap)
  * the J-run outer law: for J consecutive gaps g_1..g_J with every one of the J-2 middles
    >= 6, the maximum of g_1 + g_J.

Conventions: research/proof/neighbour_profile.md 0 (max-gap convention, cyclic over the period)
and research/proof/glue_covering.md 2.8(b).

Usage: uv run python research/anchor235/r72/u45_sieve.py <y> [jmax] [chunk]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]


def gears_upto(y):
    return [p for p in PRIMES if p <= y]


class Acc:
    """Streaming accumulators over the cyclic gap sequence."""

    def __init__(self, jmax):
        self.jmax = max(jmax, 3)
        self.spec = np.zeros(1024, dtype=np.int64)
        self.nv = {}                      # v -> max(L + R)
        self.f2 = -1
        self.outer = {J: -1 for J in range(3, self.jmax + 1)}
        self.owit = {J: None for J in range(3, self.jmax + 1)}
        self.carry = np.zeros(0, dtype=np.int64)
        self.ngaps = 0
        self.span = 0

    def feed(self, g, count=True):
        """g: int64 array of gaps, contiguous after the carry held from the last call."""
        g = np.asarray(g, dtype=np.int64)
        if g.size == 0:
            return
        if count:
            self.ngaps += g.size
            self.span += int(g.sum())
            self.spec += np.bincount(g, minlength=1024)[:1024]
        a = np.concatenate([self.carry, g]) if self.carry.size else g
        if a.size >= 2:
            self.f2 = max(self.f2, int((a[:-1] + a[1:]).max()))
        if a.size >= 3:
            L, v, R = a[:-2], a[1:-1], a[2:]
            u = np.unique(v * 4096 + (L + R))
            for k in u:
                vv, ss = int(k) // 4096, int(k) % 4096
                if ss > self.nv.get(vv, -1):
                    self.nv[vv] = ss
        for J in range(3, self.jmax + 1):
            if a.size < J:
                continue
            n = a.size - J + 1
            ok = np.ones(n, dtype=bool)
            for t in range(1, J - 1):
                ok &= a[t:t + n] >= 6
            if not ok.any():
                continue
            s = np.where(ok, a[0:n] + a[J - 1:J - 1 + n], -1)
            i = int(np.argmax(s))
            if int(s[i]) > self.outer[J]:
                self.outer[J] = int(s[i])
                self.owit[J] = [int(x) for x in a[i:i + J]]
        keep = self.jmax - 1
        self.carry = a[-keep:].copy() if a.size >= keep else a.copy()


def run(y, jmax=5, chunk=100_000_000):
    gears = gears_upto(y)
    P = 1
    for g in gears:
        P *= g
    us = [(g, pow(6, -1, g)) for g in gears]
    acc = Acc(jmax)
    t0 = time.time()
    prev_open = None
    lo = 0
    while lo < P:
        hi = min(lo + chunk, P)
        n = hi - lo
        blocked = np.zeros(n, dtype=bool)
        for g, u in us:
            blocked[(u - lo) % g::g] = True
            blocked[((-u) - lo) % g::g] = True
        op = np.flatnonzero(~blocked).astype(np.int64) + lo
        del blocked
        if op.size:
            if prev_open is not None:
                op = np.concatenate([[prev_open], op])
            prev_open = int(op[-1])
            if op.size >= 2:
                acc.feed(np.diff(op))
        lo = hi
    # close the cycle: the wrap gap (counted once) then the first jmax-1 gaps of the period again
    head_n = min(P, 1_000_000)
    blocked = np.zeros(head_n, dtype=bool)
    for g, u in us:
        blocked[u % g::g] = True
        blocked[(-u) % g::g] = True
    head = np.flatnonzero(~blocked).astype(np.int64)
    wrap = int(head[0]) + P - prev_open
    acc.feed(np.array([wrap], dtype=np.int64), count=True)
    acc.feed(np.diff(head[:acc.jmax]), count=False)
    stats = {
        "y": y, "gears": gears, "P": P,
        "ngaps": acc.ngaps, "span": acc.span,
        "F": int(np.flatnonzero(acc.spec)[-1]),
        "F2": acc.f2,
        "secs": round(time.time() - t0, 1),
    }
    return acc, stats


def main():
    y = int(sys.argv[1])
    jmax = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000_000
    acc, stats = run(y, jmax, chunk)
    nv = {int(k): int(v) for k, v in sorted(acc.nv.items())}
    rep = {
        **stats,
        "spectrum": {int(v): int(c) for v, c in enumerate(acc.spec) if c},
        "Nv": nv,
        "maxN_v_ge_6": max((s for v, s in nv.items() if v >= 6), default=None),
        "argmax_v": max(((s, v) for v, s in nv.items() if v >= 6), default=(None, None))[1],
        "outer": {str(J): acc.outer[J] for J in acc.outer},
        "outer_witness": {str(J): acc.owit[J] for J in acc.owit},
    }
    print(json.dumps({k: v for k, v in rep.items() if k not in ("spectrum", "Nv")}, indent=1))
    with open(os.path.join(OUT, f"sieve_m{y}.json"), "w") as f:
        json.dump(rep, f)


if __name__ == "__main__":
    main()
