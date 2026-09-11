"""ol2_certgate.py -- gate for the certifier of ol2_verify.py.

The certifier claims more than the instrument does: it names an explicit COLUMN of the machine at
which a window occurs.  Before it is used on m43 and m47 it is checked against a direct sieve at
the engines whose whole period fits in memory:

  * every 1-, 2- and 3-window that the sieve finds must be certified REALISED, and the certified
    column must itself carry the window in the sieve's own array (modulo the period);
  * every window the sieve does NOT find, up to the span the sieve covers, must be certified
    NOT realised.

A disagreement in either direction condemns the certifier.

Usage: uv run python research/anchor235/r71/ol2_certgate.py [y ...]      (default 11 13 17 19)
"""
import os
import sys
from math import prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ol2_core import gears_upto, u_of                      # noqa: E402
from ol2_verify import certify_word                        # noqa: E402


def sieve(gears):
    """The whole period of the machine {5..y} as a boolean array: True = open."""
    P = prod(gears)
    a = np.ones(P, dtype=bool)
    for g in gears:
        u = u_of(g)
        for r in (u % g, (-u) % g):
            a[r::g] = False
    return a


def windows(open_cols, P, kmax, span_cap):
    """Every k-window (k <= kmax) of consecutive gaps that occurs, with span <= span_cap."""
    gaps = np.diff(np.concatenate([open_cols, [open_cols[0] + P]]))
    out = {k: set() for k in range(1, kmax + 1)}
    n = gaps.size
    for i in range(n):
        s = 0
        for k in range(1, kmax + 1):
            s += int(gaps[(i + k - 1) % n])
            if s > span_cap:
                break
            out[k].add(tuple(int(gaps[(i + j) % n]) for j in range(k)))
    return out


def main():
    ys = [int(x) for x in sys.argv[1:]] or [11, 13, 17, 19]
    for y in ys:
        gears = gears_upto(y)
        P = int(prod(gears))
        a = sieve(gears)
        cols = np.flatnonzero(a)
        cap = int(np.diff(cols).max()) + 2
        occ = windows(cols, P, 3, cap)
        print(f"=== m{y}: period {P:,}, {cols.size:,} open columns, widest gap "
              f"{int(np.diff(cols).max())}, span cap {cap}", flush=True)
        bad = 0
        for k in (1, 2, 3):
            # POSITIVE direction: everything that occurs must certify, at a column that works
            for w in sorted(occ[k]):
                c = certify_word(list(w), gears)
                if c is None or not c["verified"]:
                    print(f"  FALSE NEGATIVE m{y} {w}: {c}", flush=True)
                    bad += 1
                    continue
                x = c["column"] % P
                off, o = [0], 0
                for g in w:
                    o += g
                    off.append(o)
                if not (all(a[(x + t) % P] for t in off)
                        and all(not a[(x + t) % P] for t in range(off[-1] + 1)
                                if t not in set(off))):
                    print(f"  COLUMN WRONG m{y} {w} at {x}", flush=True)
                    bad += 1
            # NEGATIVE direction: at k = 1 and k = 2 the complement is small enough to sweep
            if k == 1:
                absent = [(v,) for v in range(1, cap + 1) if (v,) not in occ[1]]
            elif k == 2:
                absent = [(u, v) for u in range(1, cap + 1) for v in range(1, cap + 1)
                          if u + v <= cap and (u, v) not in occ[2]]
            else:
                absent = []
            for w in absent:
                c = certify_word(list(w), gears)
                if c is not None:
                    print(f"  FALSE POSITIVE m{y} {w}: column {c['column']}", flush=True)
                    bad += 1
            print(f"  k = {k}: {len(occ[k]):,} occurring windows certified, "
                  f"{len(absent):,} absent windows refuted", flush=True)
        print(f"  m{y}: {bad} disagreement(s)", flush=True)


if __name__ == "__main__":
    main()
