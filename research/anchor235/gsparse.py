"""G_t(M, q') = longest stretch of the old word {5..q} in which every window of q' consecutive
slots holds <= t openings. G_0 = F (record gap). (D) at rung q' follows from G_2 <= F + q'.
Print G_0, G_1, G_2 with margins, and the worst G_2 stretch.
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
    return w, P


def longest_sparse(cq, t, q2, P):
    """longest run of consecutive k (cyclic) with cq[k] <= t, in windows; the stretch length
    is run length + q' - 1 (a run of r window-starts covers r + q' - 1 slots)."""
    ok = cq[:P] <= t
    okk = np.concatenate([ok, ok]).astype(np.int8)
    d = np.diff(np.concatenate([[0], okk, [0]]))
    s = np.flatnonzero(d == 1)
    e = np.flatnonzero(d == -1)
    keep = s < P
    s, e = s[keep], e[keep]
    if len(s) == 0:
        return 0, -1
    r = np.minimum(e - s, P)
    i = int(r.argmax())
    return int(r[i]) + q2 - 1, int(s[i])


def main():
    qmax = int(sys.argv[1]) if len(sys.argv) > 1 else 29
    for idx in range(1, len(PR)):
        q2 = PR[idx]
        if q2 > qmax:
            break
        gears = PR[:idx]
        w, P = openings(gears)
        reps = 2 + (8 * q2 + 400) // P
        ww = np.concatenate([w] * reps).astype(np.int32)
        cs = np.concatenate([[0], np.cumsum(ww)])
        cq = cs[q2:q2 + P] - cs[:P]
        X = np.flatnonzero(w)
        gaps = np.diff(np.concatenate([X, [X[0] + P]]))
        F = int(gaps.max()) - 1
        F2 = int((gaps + np.roll(gaps, -1)).max()) - 1
        u2 = pow(6, -1, q2)
        sm = min((2 * u2) % q2, (-2 * u2) % q2)
        G = {}
        pos = {}
        for t in (0, 1, 2, 3):
            G[t], pos[t] = longest_sparse(cq, t, q2, P)
        # G_t counts slots in a stretch where every q'-window has <= t openings; for t = 0
        # this is the record gap F only if F >= q'; report raw
        print(f"{'+'.join(map(str, gears))} + {q2}: F={F} F2={F2} q'={q2} s_min={sm}:  "
              f"G0={G[0]} G1={G[1]} G2={G[2]} G3={G[3]}   (D) needs G2 <= F+q'={F + q2}: margin {F + q2 - G[2]}   "
              f"G2 - F2 = {G[2] - F2}   G1 - F = {G[1] - F}")
        i = pos[2]
        Lw = G[2]
        offs = list(map(int, np.flatnonzero(ww[i:i + Lw])))
        print(f"    worst 3-sparse stretch k={i}, {Lw} slots, {len(offs)} openings, gaps {[b - a for a, b in zip(offs, offs[1:])]}, first offset {offs[0] if offs else None}")


if __name__ == "__main__":
    main()
