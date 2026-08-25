"""Round 23 (mechanic): machine-verify a multi-gear marked-spectrum witness.

research/j5_multi.py reports a witness as (x0, m, phase tuple, marks) on the OLD
machine.  This script turns it into an ADDRESS OF THE NEW MACHINE and asserts
the whole configuration there, so nothing rests on the scan's own bookkeeping.

The translation.  Phase c for new gear q means: the window sits in the lap j
with c = -j*P mod q, because slot k = x + j*P is blocked by gear q iff
k = +-u (mod q) iff x = c +- u (mod q) with c = -j*P.  With r new gears, CRT
over the r distinct primes gives the unique j mod (q_1...q_r) satisfying
j = -c_i * P^{-1} (mod q_i) for every i.  Then k* = x0 + j*P is the new
machine's address and the whole window is checked there directly.

usage: uv run python research/multi_witness_verify.py OLD q1,q2,.. a x0 span \
           c1,c2,.. mark_offsets(comma, relative to x0)
"""
import sys
from math import prod
import numpy as np


def primes_upto(n):
    s = np.ones(n + 1, bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i * i::i] = False
    return [int(p) for p in np.flatnonzero(s)]


def crt(res, mod):
    x, M = 0, 1
    for r, m in zip(res, mod):
        g = pow(M % m, -1, m)
        x += M * ((r - x) * g % m)
        M *= m
    return x % M, M


def main():
    OLD = int(sys.argv[1])
    NEW = [int(x) for x in sys.argv[2].split(',')]
    A = int(sys.argv[3])
    x0 = int(sys.argv[4])
    span = int(sys.argv[5])
    cs = [int(x) for x in sys.argv[6].split(',')]
    marg = sys.argv[7] if len(sys.argv) > 7 else ''
    IDX = marg.startswith('i')          # j5_multi prints INDICES into the
    if IDX:                             # interior list, not offsets
        idxs = [int(x) for x in marg[1:].split(',')]
        m_win = 0
        marks = None                    # resolved below, after P is known
    else:
        marks = [int(x) for x in marg.split(',')] if marg else []

    old_gears = [p for p in primes_upto(OLD) if p >= 5]
    P = prod(old_gears)
    all_gears = old_gears + NEW
    if IDX:                             # walk the old machine forward from x0
        def blk_old(x):
            for q in old_gears:
                u = pow(6, -1, q)
                if x % q in (u % q, (-u) % q):
                    return True
            return False
        assert not blk_old(x0), "x0 is not an opening of the old machine"
        interior = [x for x in range(x0 + 1, x0 + span) if not blk_old(x)]
        assert not blk_old(x0 + span), "window end is not an old-machine opening"
        marks = [interior[t] - x0 for t in idxs]
        print(f"  resolved marks (indices {idxs} of {len(interior)} interiors)"
              f" -> offsets {marks}")
    print(f"old machine {OLD}: gears {old_gears}, P = {P:,}")
    print(f"new machine {NEW[-1]}: gears {all_gears}")

    # lap number j from the phases
    js, ms = [], []
    for q, c in zip(NEW, cs):
        js.append((-c * pow(P % q, -1, q)) % q)
        ms.append(q)
    j, M = crt(js, ms)
    k = x0 + j * P
    Pnew = P * prod(NEW)
    print(f"  lap j = {j:,} (mod {M:,});  new-machine address k = {k:,} "
          f"(period {Pnew:,})")

    def blocked(x, gears):
        for q in gears:
            u = pow(6, -1, q)
            if x % q in (u % q, (-u) % q):
                return True
        return False

    pts = [0] + marks + [span]
    print(f"  claimed new-machine openings at k + {pts}")
    for p in pts:
        assert not blocked(k + p, all_gears), ("claimed opening is blocked", p)
    nblk = 0
    for d in range(1, span):
        if d in pts:
            continue
        assert blocked(k + d, all_gears), ("interior slot open at +%d" % d, d)
        nblk += 1
    gaps = [pts[t + 1] - pts[t] for t in range(len(pts) - 1)]
    mids = gaps[1:-1]
    print(f"  gaps {gaps}  (sum {sum(gaps)} = span {span})")
    print(f"  middles {mids}; floor a = {A}")
    assert all(g >= A for g in mids), ("middle gap below floor", mids)
    assert sum(gaps) == span
    print(f"  VERIFIED at machine {NEW[-1]}: {len(pts)} openings, all "
          f"{nblk} interior slots blocked, every middle gap >= {A}.")
    print(f"  => Q_{len(gaps)}({NEW[-1]}; {A}) >= {span}, witness k = {k:,}")


if __name__ == '__main__':
    main()
