"""u45_window.py -- W32's first-hit law tested on the ENGINE's own window and its section.

W32 (research/proof/law_register.md; research/proof/top_machine_2.md 3.7, law L32): for a fixed
gear set with period W and full-period census c(d) = the number of gaps of size >= d per period,
the record on a range of N columns is

    F_range(N) = max { d : W / c(d) <= N } - 1 .

It was measured on three manifold wheels at 21 checkpoints, never on the engine, whose window is
one specific translate -- the phase-zero one, which starts at the origin of the period.

Here the engine is M = {5..y}, W = P(y) = prod_{5<=g<=y} g, and the census is the exact one of
u45_census.py (a covering count, no period).  Two ranges, both phase-zero translates:

  * the WINDOW of {5..y}: the columns k with 6k-1 > y and 6k+1 < y'^2, y' = nextprime(y)
    (research/proof/valve_existence.md's W(y) -- the range {5..y} certifies).  Also reported for
    the conservative form (y, y^2] of the tree's root question.
  * the SECTION of {5..y}: the window's new part, the columns k with 6k-1 > y^2, 6k+1 < y'^2.

Inside either range the openings of {5..y} are exactly the twin pairs, so F_W(y) is the window's
longest twin gap in columns.

Usage: uv run python research/anchor235/r72/u45_window.py [ymax]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)

from u45_census import c_of, gears_upto  # noqa: E402

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]


def nextprime(n):
    m = n + 1
    while True:
        if m > 1 and all(m % p for p in range(2, int(m ** 0.5) + 1)):
            return m
        m += 1


def openings(y, klo, khi):
    """Openings of {5..y} among columns klo..khi (inclusive)."""
    gears = gears_upto(y)
    n = khi - klo + 1
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    blocked = np.zeros(n, dtype=bool)
    for g in gears:
        u = pow(6, -1, g)
        blocked[(u - klo) % g::g] = True
        blocked[((-u) - klo) % g::g] = True
    return np.flatnonzero(~blocked).astype(np.int64) + klo


def record_of(op):
    if op.size < 2:
        return None, None
    d = np.diff(op)
    i = int(np.argmax(d))
    return int(d[i]), int(op[i])


def first_hit(y, N, cache):
    """max{d : P/c(d) <= N} - 1, with the exact census."""
    P = 1
    for g in gears_upto(y):
        P *= g
    last = 0
    d = 1
    while True:
        if (y, d) not in cache:
            cache[(y, d)] = c_of(y, d)[0]
        c = cache[(y, d)]
        if c == 0 or c * N < P:
            break
        last = d
        d += 1
        if d > 400:
            break
    return last - 1, last


def klo_of(y):
    k = 1
    while 6 * k - 1 <= y:
        k += 1
    return k


def khi_below(top):
    """largest k with 6k+1 < top"""
    k = (top - 2) // 6
    while 6 * k + 1 >= top:
        k -= 1
    return k


def main():
    ymax = int(sys.argv[1]) if len(sys.argv) > 1 else 53
    cache = {}
    rows = []
    t0 = time.time()
    for y in [p for p in PRIMES if 7 <= p <= ymax]:
        yp = nextprime(y)
        P = 1
        for g in gears_upto(y):
            P *= g
        klo = klo_of(y)
        row = {"y": y, "yprime": yp, "P": P}
        for tag, lo, hi in (
            ("window", klo, khi_below(yp * yp)),
            ("square", klo, (y * y - 1) // 6),
            ("section", klo_of(y * y) if y * y > 6 else klo, khi_below(yp * yp)),
        ):
            if tag == "section":
                lo = 1
                while 6 * lo - 1 <= y * y:
                    lo += 1
            N = hi - lo + 1
            op = openings(y, lo, hi)
            F, at = record_of(op)
            pred, dstar = first_hit(y, N, cache)
            row[tag] = {"klo": lo, "khi": hi, "N": N, "openings": int(op.size),
                        "F": F, "at": at, "pred": pred, "d_star": dstar,
                        "diff": None if F is None else F - pred}
        rows.append(row)
        w, s = row["window"], row["section"]
        print(f"y={y:3d} y'={yp:3d}  window N={w['N']:5d} twins={w['openings']:4d} "
              f"F_W={w['F']} pred={w['pred']} diff={w['diff']}   | "
              f"section N={s['N']:5d} twins={s['openings']:4d} F_sec={s['F']} "
              f"pred={s['pred']} diff={s['diff']}", flush=True)
    with open(os.path.join(OUT, "window.json"), "w") as f:
        json.dump({"rows": rows, "secs": round(time.time() - t0, 1)}, f, indent=1)
    print(f"{time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
