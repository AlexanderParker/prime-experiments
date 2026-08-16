"""Exact maximum gap of the twin-admissible pattern (twin Jacobsthal function).

The twin blocking system of docs/twin-prime-program.md is periodic modulo
P(y) = product of primes q <= y. A position m survives the blocking exactly
when

    m is odd, and for every odd prime q <= y:  m != 0 (mod q) and m != -2 (mod q)

Call that set T(y). By CRT it is a union of residue classes mod P(y), and its
size per period is  S(y) = product over odd q <= y of (q - 2).

The quantity that matters for the twin prime conjecture in this framework is

    J2(y) = the largest gap between consecutive members of T(y)

because the first survivor the algorithm returns for any n is at most J2(y)
away, whatever n's residues happen to be. This script computes J2(y) exactly,
over the whole state space rather than over realisable integers only.

For y up to 23 the period fits in memory. For larger y the period is scanned
in chunks, carrying the last survivor across chunk boundaries, so no array
larger than the chunk is ever allocated.
"""

import math
import sys

import numpy as np


def primes_upto(limit):
    flags = bytearray([1]) * (limit + 1)
    flags[0:2] = b"\x00\x00"
    for i in range(2, math.isqrt(limit) + 1):
        if flags[i]:
            flags[i * i :: i] = bytearray(len(flags[i * i :: i]))
    return [i for i in range(2, limit + 1) if flags[i]]


def period(y):
    p = 1
    for q in primes_upto(y):
        p *= q
    return p


def survivor_count(y):
    total = 1
    for q in primes_upto(y):
        if q > 2:
            total *= q - 2
    return total


def twin_jacobsthal(y, chunk=1 << 26, verbose=False):
    """Return (J2, P, S): max gap in T(y), the period, the survivors per period.

    Gaps are measured cyclically, so the wrap-around gap from the last
    survivor of one period to the first survivor of the next is included.
    """
    qs = [q for q in primes_upto(y) if q > 2]
    P = period(y)
    max_gap = 0
    first = None
    last = None

    start = 0
    while start < P:
        size = min(chunk, P - start)
        alive = np.ones(size, dtype=bool)
        # m must be odd
        alive[(1 - start) % 2 :: 2] = False
        for q in qs:
            for r in (0, -2):
                offset = (r - start) % q
                alive[offset::q] = False
        idx = np.flatnonzero(alive)
        if idx.size:
            positions = idx + start
            if first is None:
                first = int(positions[0])
            if last is not None:
                max_gap = max(max_gap, int(positions[0]) - last)
            if idx.size > 1:
                max_gap = max(max_gap, int(np.diff(positions).max()))
            last = int(positions[-1])
        start += size
        if verbose:
            print(f"  y={y} scanned {start}/{P}", file=sys.stderr)

    # wrap-around gap
    max_gap = max(max_gap, (P - last) + first)
    return max_gap, P, survivor_count(y)


if __name__ == "__main__":
    ys = [int(a) for a in sys.argv[1:]] or [3, 5, 7, 11, 13, 17, 19, 23]
    print(f"{'y':>4} {'J2(y)':>10} {'y^2':>12} {'J2/y^2':>8} {'J2/(y*ln^2 y)':>14} "
          f"{'period':>16} {'survivors':>14}")
    for y in ys:
        j2, P, S = twin_jacobsthal(y)
        scale = y * math.log(y) ** 2
        print(f"{y:>4} {j2:>10} {y * y:>12} {j2 / (y * y):>8.3f} "
              f"{j2 / scale:>14.3f} {P:>16} {S:>14}")
