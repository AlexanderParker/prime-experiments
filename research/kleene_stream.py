"""Round 22 (constructor): the Kleene identity and the finite-state
certificate at machines too big to hold in memory (segmented, ~300 MB).

Same objects as research/kleene_generator.py - see that file for the algebra:

    F(M + q')  =  L^T (x) K* (x) R          (max-plus Kleene star identity)
    (D)        <=>  exists h with (C1) h >= R, (C2) h >= K (x) h, (C3) L+h <= F+q'

but computed by streaming the period in segments with an opening overlap, so
machine 29 (period 1.08e9, 2.15e8 openings) runs in a few hundred MB instead
of the ~4 GB the dense build needs.  Every quantity is exact and full period:
the cyclic seam is stitched by appending the first OVL openings after the end
and every item is counted from its own left flank.

Also accumulated in the same pass: the SOUND class-level max-plus system for
bounded local states (value; (corridor phase mod m, value)), whose Kleene
closure is a genuine upper bound on F(M+q') - the machine-free certificate
test.

Usage: uv run python research/kleene_stream.py y [--seg N] [--mods 35,385]
"""
import os
import sys
import time
from math import prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DDIR = os.path.join(HERE, "data")

KNOWN_F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88,
           41: 91}
OVL = 16          # openings carried across a segment boundary
MAXK = 8          # max chain links tracked (assert if ever saturated)


def primes(a, b):
    return [n for n in range(a, b + 1)
            if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))]


def next_prime(y):
    p = y + 1
    while not all(p % d for d in range(2, int(p ** 0.5) + 1)):
        p += 1
    return p


class ClassSystem:
    """Sound class-level max-plus system for one bounded state map."""

    def __init__(self, name, mod, q1, maxval=128):
        self.name = name
        self.mod = mod                     # None = value only
        self.q1 = q1
        self.width = (maxval * 4 if mod is None else mod * maxval * 4)
        self.R = np.zeros(self.width, np.int32)
        self.L = np.zeros(self.width, np.int32)
        self.seen = np.zeros(self.width, bool)
        self.ew = {}                       # (src_key, s, dst_key, s') -> w

    def key(self, ph, d, ccode):
        if self.mod is None:
            return d.astype(np.int64) * 4 + ccode
        return (ph.astype(np.int64) * 128 + d) * 4 + ccode

    def absorb(self, k, d, lflank, k_next, ccode, cls):
        np.maximum.at(self.R, k, d.astype(np.int32))
        np.maximum.at(self.L, k, lflank.astype(np.int32))
        self.seen[k] = True
        for s, leg, land in (
                (0, (cls == 0) | (cls == 1),
                 np.where(cls == 0, 0, 1).astype(np.int8)),
                (1, (cls == 0) | (cls == -1),
                 np.where(cls == 0, 1, 0).astype(np.int8))):
            sel = np.flatnonzero(leg)
            if not len(sel):
                continue
            eid = (k[sel] * 4 + s) * (1 << 32) + k_next[sel] * 4 + land[sel]
            w = d[sel].astype(np.int64)
            u, first = np.unique(eid, return_index=True)
            mx = np.zeros(len(u), np.int64)
            np.maximum.at(mx, np.searchsorted(u, eid), w)
            for e, m in zip(u.tolist(), mx.tolist()):
                if self.ew.get(e, -1) < m:
                    self.ew[e] = m

    def close(self, budget, exact):
        ks = sorted({e >> 32 for e in self.ew} |
                    {e & 0xFFFFFFFF for e in self.ew} |
                    {int(k) * 4 + s for k in np.flatnonzero(self.seen)
                     for s in (0, 1)})
        idx = {k: i for i, k in enumerate(ks)}
        S = len(ks)
        hh = np.array([int(self.R[k >> 2]) for k in ks], np.int64)
        Ls = np.array([int(self.L[k >> 2]) for k in ks], np.int64)
        esrc = np.array([idx[e >> 32] for e in self.ew], np.int64)
        edst = np.array([idx[e & 0xFFFFFFFF] for e in self.ew], np.int64)
        ewv = np.array(list(self.ew.values()), np.int64)
        cyclic = False
        for _ in range(S + 2):
            new = hh.copy()
            np.maximum.at(new, esrc, ewv + hh[edst])
            if np.array_equal(new, hh):
                break
            hh = new
        else:
            cyclic = True
        bound = None if cyclic else int((Ls + hh).max())
        print("     %-26s states %7d  %s"
              % (self.name, S,
                 "CYCLIC -> class closure = +inf (vacuous)" if cyclic else
                 "bound %d  vs  F+q' = %d   %s (exact %d)"
                 % (bound, budget,
                    "CERTIFIES (D)" if bound <= budget else "FAILS by %+d"
                    % (bound - budget), exact)))
        return bound


def run(y, seg=64_000_000, mods=(35, 385, 5005)):
    gears = primes(5, y)
    P = prod(gears)
    q1 = next_prime(y)
    u1 = round(q1 / 6)
    a, b = 2 * u1, q1 - 2 * u1
    uvals = [pow(6, -1, g) for g in gears]
    sysx = [ClassSystem("value only", None, q1)] + \
           [ClassSystem("(phase mod %d, value)" % M, M, q1) for M in mods]
    F_old = 0
    F_new = 0
    layers = np.zeros(MAXK + 1, np.int64)
    ngaps = 0
    tail = None
    head = None
    t0 = time.time()

    def eat(ops, first_new):
        """ops: consecutive openings.  Items owned by this call are those
        whose LEFT FLANK starts at index >= first_new."""
        nonlocal F_old, F_new, ngaps
        d = np.diff(ops)
        n = len(d)
        if n <= MAXK + 2:
            return
        cls = np.full(n, 9, np.int8)
        r = d % q1
        cls[r == 0] = 0
        cls[r == a] = 1
        cls[r == b] = -1
        ccode = np.where(cls == 9, 3, cls.astype(np.int64) + 1)
        # h by explicit unrolling (chains are short); entries within MAXK of
        # the right edge are truncated, so only i <= n-1-MAXK is used
        nxt = np.arange(1, n)
        cm = np.where(cls[:-1] == 0, 0, 1)
        cp = np.where(cls[:-1] == 0, 1, 0)
        lm = (cls[:-1] == 0) | (cls[:-1] == 1)
        lp = (cls[:-1] == 0) | (cls[:-1] == -1)
        h = np.stack([d.astype(np.int64), d.astype(np.int64)])
        used = 0
        for _ in range(MAXK):
            hn = h.copy()
            hn[0][:-1] = np.where(lm, np.maximum(h[0][:-1],
                                                 d[:-1] + h[cm, nxt]),
                                  h[0][:-1])
            hn[1][:-1] = np.where(lp, np.maximum(h[1][:-1],
                                                 d[:-1] + h[cp, nxt]),
                                  h[1][:-1])
            if np.array_equal(hn, h):
                break
            h = hn
            used += 1
        assert used < MAXK, "chain longer than MAXK"
        i0 = max(1, first_new)
        i1 = n - MAXK - 1
        if i1 <= i0:
            return
        sl = slice(i0, i1)
        ngaps += i1 - i0
        F_old = max(F_old, int(d[sl].max()))
        L = d[i0 - 1:i1 - 1]
        F_new = max(F_new, int(max((L + h[0][sl]).max(),
                                   (L + h[1][sl]).max())))
        # layer maxima
        NEGB = -(1 << 40)
        cur = np.stack([d.astype(np.int64), d.astype(np.int64)])
        for lay in range(MAXK + 1):
            if cur.max() < NEGB / 2:
                break
            layers[lay] = max(layers[lay],
                              int(max((L + cur[0][sl]).max(),
                                      (L + cur[1][sl]).max())))
            nx = np.full_like(cur, NEGB)
            nx[0][:-1] = np.where(lm, d[:-1] + cur[cm, nxt], NEGB)
            nx[1][:-1] = np.where(lp, d[:-1] + cur[cp, nxt], NEGB)
            cur = nx
        # class systems
        for S in sysx:
            ph = None if S.mod is None else (ops[:-1] % S.mod)
            k = S.key(ph if ph is not None else 0, d, ccode)
            S.absorb(k[sl], d[sl], L, k[i0 + 1:i1 + 1], ccode[sl], cls[sl])

    for lo in range(0, P, seg):
        hi = min(P, lo + seg)
        ex = np.zeros(hi - lo, bool)
        for g, u in zip(gears, uvals):
            ex[(u - lo) % g::g] = True
            ex[(-u - lo) % g::g] = True
        op = (np.flatnonzero(~ex) + lo).astype(np.int64)
        del ex
        if head is None:
            head = op[:2 * OVL].copy()
        ops = op if tail is None else np.concatenate([tail, op])
        eat(ops, 1 if tail is None else OVL)
        tail = ops[-(MAXK + OVL + 2):].copy()
        del op, ops
        if (lo // seg) % 4 == 0:
            print("  seg to %.4g (%.1f%%) %.0fs" % (hi, 100 * hi / P,
                                                    time.time() - t0),
                  flush=True)
    eat(np.concatenate([tail, head + P]), len(tail) - 1)

    print("\n=== machine %d -> %d  (period %d, %d gaps counted)"
          % (y, q1, P, ngaps))
    print("  letters a=%d b=%d ; F(M) = %d" % (a, b, F_old))
    print("  L (x) K* (x) R = %d   vs   F(M+q') known = %s"
          % (F_new, KNOWN_F.get(q1)))
    if q1 in KNOWN_F:
        assert F_new == KNOWN_F[q1], (y, F_new, KNOWN_F[q1])
        print("     IDENTITY VERIFIED (exact, full period)")
    nz = [int(v) for v in layers if v > 0]
    print("  layer maxima k = 0.. : %s" % nz)
    budget = F_old + q1
    print("  CERTIFICATE (C3): %d <= F + q' = %d   margin %+d (%.3f q')"
          % (F_new, budget, budget - F_new, (budget - F_new) / q1))
    assert F_new <= budget
    print("  abstraction test (sound class-level closure):")
    for S in sysx:
        S.close(budget, F_new)
    print("  (%.0f s)" % (time.time() - t0))


def main():
    args = sys.argv[1:]
    seg = 64_000_000
    mods = (35, 385, 5005)
    if "--seg" in args:
        i = args.index("--seg")
        seg = int(float(args[i + 1]))
        del args[i:i + 2]
    if "--mods" in args:
        i = args.index("--mods")
        mods = tuple(int(x) for x in args[i + 1].split(","))
        del args[i:i + 2]
    for aY in args:
        run(int(aY), seg=seg, mods=mods)


if __name__ == "__main__":
    main()
