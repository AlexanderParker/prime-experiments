"""Round 13, corrected: F(M+q') with the RIGHT link condition.

Gear q' kills openings at the two teeth {+u, -u} mod q'. Walking a run of
consecutive killed openings, each spacing must be
    = 0      (mod q')  : same tooth again  -> PADDED link (costs >= q')
    = +2u    (mod q')  : -u -> +u          -> only legal from tooth -u
    = -2u    (mod q')  : +u -> -u          -> only legal from tooth +u
so the +-2u letters must ALTERNATE (0's may be inserted freely). Two equal
consecutive non-zero letters are impossible - that was the bug in
merge_general.py (it allowed (10,10) at 23->29 and overshot 43 to 45);
merge_decompose.py had the opposite bug, matching only literal spacing VALUES
and so missing padded links (undershooting 31->37).

    F(M+q') = max over maximal legal killed runs of ( o[i+k] - o[i-1] )
"""
import sys
from math import prod

import numpy as np

from split_gap_law import primes

KNOWN_F = {17: 18, 19: 25, 23: 34, 29: 43, 31: 58}

def step(y, qp, chunk=100_000_000):
    gears = primes(5, y)
    P = prod(gears)
    u = pow(6, -1, qp)
    A, B = (2 * u) % qp, (-2 * u) % qp
    best, bestinfo, F2, Fold = 0, None, 0, 0
    tail = None
    a = 0
    while a < P:
        S = min(chunk, P - a)
        killed = np.zeros(S, bool)
        for q in gears:
            uq = pow(6, -1, q)
            for t in (uq, q - uq):
                killed[(t - a) % q::q] = True
        o = np.flatnonzero(~killed).astype(np.int64) + a
        if tail is not None:
            o = np.concatenate((tail, o))
        d = np.diff(o)
        if len(d):
            Fold = max(Fold, int(d.max()))
        if len(o) > 2:
            F2 = max(F2, int((o[2:] - o[:-2]).max()))
        r = d % qp
        cls = np.where(r == 0, 0, np.where(r == A, 1, np.where(r == B, 2, 3)))
        pos = np.arange(len(cls))
        letter = cls != 0                       # A, B or X carry a letter
        lastl = np.maximum.accumulate(np.where(letter, pos, -1))
        prev = np.full(len(cls), -1)
        prev[1:] = lastl[:-1]
        bad = letter & (prev >= 0) & (cls[np.maximum(prev, 0)] == cls)
        brk = (cls == 3) | bad                  # illegal link positions
        good = ~brk
        idx = np.flatnonzero(good)
        if len(idx):
            cut = np.flatnonzero(np.diff(idx) != 1)
            starts = np.concatenate(([idx[0]], idx[cut + 1]))
            ends = np.concatenate((idx[cut], [idx[-1]]))
            for st, en in zip(starts, ends):
                if st - 1 < 0 or en + 2 >= len(o):
                    continue
                merged = int(o[en + 2] - o[st - 1])
                if merged > best:
                    best = merged
                    bestinfo = (int(o[st]), en - st + 2,
                                tuple(int(x) for x in d[st:en + 1]),
                                int(o[st] - o[st - 1]), int(o[en + 2] - o[en + 1]))
        tail = o[-400:]
        a += S
    Fnew = max(F2, best)
    chk = KNOWN_F.get(qp)
    v = "" if chk is None else f"  [known {chk}: {'OK' if chk == Fnew else 'MISMATCH'}]"
    print(f"STEP {y}->{qp} (u={u}, letters A={A} B={B} mod {qp}): "
          f"F_old {Fold}, F2 {F2}, F_new {Fnew}{v}")
    if bestinfo and best >= F2:
        p, k, w, fl, fr = bestinfo
        pad = [x for x in w if x % qp == 0] + [x for x in w if x % qp and x > qp]
        print(f"  winner: {k} kills at {p}, spacings {w} (span {sum(w)}), "
              f"flanks {fl}+{fr}; padded links: {pad if pad else 'none (literal)'}")
    print(f"  excess = {Fnew - F2} ({(Fnew - F2)/qp:+.3f} q')")
    return Fnew

if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 99
    for y, qp in [(13, 17), (17, 19), (19, 23), (23, 29), (29, 31)]:
        if y <= lim:
            step(y, qp)
