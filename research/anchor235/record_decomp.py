"""Decompose the new record at every rung: lower sieve {5..q} + new gear q'.

Old word: openings X in [0, P), P = prod gears. Opening X_i is killed by q' in lift j
(slot X_i + jP) iff X_i + jP = +-u' (mod q'). A new blocked run is an old stretch of
consecutive openings x_0 < ... < x_J with x_1..x_{J-1} killed in the same lift, x_0, x_J not.
Report F(M), F2(M) (best two consecutive gaps), F(M+q'), J, gaps, residues, margins.
"""
import sys
from math import prod

import numpy as np

PR = [5, 7, 11, 13, 17, 19, 23, 29, 31]


def openings(gears):
    P = prod(gears)
    k = np.arange(P, dtype=np.int64)
    w = np.ones(P, dtype=bool)
    for g in gears:
        u = pow(6, -1, g)
        w &= (k % g != u) & (k % g != g - u)
    return np.flatnonzero(w), P


def smin(q):
    u = pow(6, -1, q)
    return min((2 * u) % q, (-2 * u) % q)


def main():
    qmax = int(sys.argv[1]) if len(sys.argv) > 1 else 29
    for idx in range(1, len(PR)):
        q2 = PR[idx]
        if q2 > qmax:
            break
        gears = PR[:idx]
        X, P = openings(gears)
        N = len(X)
        gaps = np.diff(np.concatenate([X, [X[0] + P]]))  # opening-to-opening, cyclic
        F = int(gaps.max()) - 1
        two = gaps + np.roll(gaps, -1)
        F2 = int(two.max()) - 1
        u2 = pow(6, -1, q2)
        Pinv = pow(P % q2, -1, q2)
        r = X % q2
        j1 = ((u2 - r) * Pinv) % q2
        j2 = ((-u2 - r) * Pinv) % q2
        best = (0, None)
        perJ = {}
        for j in range(q2):
            m = (j1 == j) | (j2 == j)
            # runs of consecutive killed openings (cyclic handled by doubling)
            mm = np.concatenate([m, m])
            d = np.diff(np.concatenate([[0], mm.astype(np.int8), [0]]))
            starts = np.flatnonzero(d == 1)
            ends = np.flatnonzero(d == -1)  # exclusive
            keep = starts < N
            starts, ends = starts[keep], ends[keep]
            ends = np.minimum(ends, starts + N - 1)
            if len(starts) == 0:
                continue
            # span from opening before start to opening after end-1
            xa = X[(starts - 1) % N] + P * ((starts - 1) // N) - (P if starts.min() < 0 else 0)
            lo = (starts - 1) % N
            hi = ends % N
            span = (X[hi] + P * (ends // N)) - (X[lo] + P * ((starts - 1) // N))
            Jn = ends - starts + 1
            for Jv in np.unique(Jn):
                s = int(span[Jn == Jv].max()) - 1
                if s > perJ.get(int(Jv), 0):
                    perJ[int(Jv)] = s
            i = int(span.argmax())
            if int(span[i]) - 1 > best[0]:
                best = (int(span[i]) - 1, (j, int(starts[i]), int(ends[i])))
        Fn, (j, s, e) = best
        idxs = [(s - 1 + t) % N for t in range(e - s + 2)]
        xs = [int(X[(s - 1 + t) % N]) + P * ((s - 1 + t) // N) for t in range(e - s + 2)]
        gl = [b - a for a, b in zip(xs, xs[1:])]
        res = [(x + j * P) % q2 for x in xs]
        sm = smin(q2)
        J = len(gl)
        print(f"{'+'.join(map(str, gears))} + {q2}: F={F} F2={F2} -> F'={Fn}  J={J} kills={J - 1}  gaps={gl}  "
              f"gaps mod q'={[g % q2 for g in gl]}  ends res={res[0]},{res[-1]} (teeth +-{u2}={u2},{q2 - u2})  "
              f"s_min={sm}  contains old record gap? {F + 1 in gl}")
        print(f"    F' - F = {Fn - F} (budget q'={q2}, (D) {'HOLDS' if Fn <= F + q2 else 'FAILS'});  "
              f"F' - F2 = {Fn - F2} (budget s_min={sm}, {'HOLDS' if Fn <= F2 + sm else 'FAILS'});  "
              f"F2 <= F'? {F2 <= Fn};  F2 - F = {F2 - F} vs q' - s_min = {q2 - sm}")
        print(f"    max blocked run by number of kills J-1: " + ", ".join(f"{k - 1}:{v}" for k, v in sorted(perJ.items())))


if __name__ == "__main__":
    main()
