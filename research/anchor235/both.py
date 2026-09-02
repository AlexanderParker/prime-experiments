"""(a) old word only: record gaps and their neighbours; F_m for m consecutive gaps.
(b) new gear: every kill chain, per number of kills, against F2 + s_min; and the same with
counterfactual tooth spacing delta (teeth {a, a+delta}; only delta matters since lifts jP
run over every residue mod q').
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


def chains(X, P, q2, delta):
    """max blocked run per number of kills for teeth {0, delta} shifted over all lifts.
    Returns dict kills -> (max run, count of chains with that many kills)."""
    N = len(X)
    Pinv = pow(P % q2, -1, q2)
    r = X % q2
    # opening i killed in lift j iff r + jP = 0 or delta  =>  j = -r Pinv  or (delta - r) Pinv
    j1 = ((-r) * Pinv) % q2
    j2 = ((delta - r) * Pinv) % q2
    out = {}
    Xd = np.concatenate([X, X + P])
    for j in range(q2):
        m = (j1 == j) | (j2 == j)
        mm = np.concatenate([m, m]).astype(np.int8)
        d = np.diff(np.concatenate([[0], mm, [0]]))
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        keep = (starts < N) & (starts > 0)
        starts, ends = starts[keep], ends[keep]
        if len(starts) == 0:
            continue
        span = Xd[ends] - Xd[starts - 1] - 1
        kills = ends - starts
        for kv in np.unique(kills):
            sel = kills == kv
            mx = int(span[sel].max())
            cnt = int(sel.sum())
            a = out.get(int(kv), (0, 0))
            out[int(kv)] = (max(a[0], mx), a[1] + cnt)
    return out


def main():
    qmax = int(sys.argv[1]) if len(sys.argv) > 1 else 23
    for idx in range(1, len(PR)):
        q2 = PR[idx]
        if q2 > qmax:
            break
        gears = PR[:idx]
        X, P = openings(gears)
        N = len(X)
        gaps = np.diff(np.concatenate([X, [X[0] + P]]))
        F = int(gaps.max()) - 1
        Fm = {}
        s = gaps.copy()
        for m in range(2, 6):
            s = s + np.roll(gaps, -(m - 1))
            Fm[m] = int(s.max()) - 1
        rec = np.flatnonzero(gaps == gaps.max())
        nb = sorted({(int(gaps[(i - 1) % N]), int(gaps[(i + 1) % N])) for i in rec})
        # second largest gap and the largest gap adjacent to any gap >= 0.8 F
        big = np.flatnonzero(gaps >= 0.8 * gaps.max())
        adj = max(max(int(gaps[(i - 1) % N]), int(gaps[(i + 1) % N])) for i in big)
        u2 = pow(6, -1, q2)
        sm = min((2 * u2) % q2, (-2 * u2) % q2)
        print(f"(a) {'+'.join(map(str, gears))}: N={N} F={F} record gaps={len(rec)} neighbours (left,right)={nb[:6]}"
              f"{' ...' if len(nb) > 6 else ''}")
        print(f"    F_m (m consecutive gaps) = " + ", ".join(f"m={m}:{v}" for m, v in Fm.items())
              + f";  F2-F={Fm[2] - F} vs q'-s_min={q2 - sm};  gaps>=0.8F: {len(big)}, largest neighbour {adj - 1}")
        # (b) real teeth
        real = (2 * u2) % q2
        res = chains(X, P, q2, real)
        print(f"(b) + {q2}: real delta=2u'={real}, s_min={sm}, F2+s_min={Fm[2] + sm}, F+q'={F + q2}: "
              + "; ".join(f"{k} kill{'s' if k > 1 else ''}: max {v[0]} ({v[1]} chains)" for k, v in sorted(res.items())))
        # counterfactual delta
        bad_inc, bad_D = [], []
        rows = []
        for delta in range(1, (q2 - 1) // 2 + 1):
            r2 = chains(X, P, q2, delta)
            Fn = max(v[0] for v in r2.values())
            smd = min(delta, q2 - delta)
            rows.append((delta, Fn, Fn - Fm[2], smd, max(r2)))
            if Fn > Fm[2] + smd:
                bad_inc.append(delta)
            if Fn > F + q2:
                bad_D.append(delta)
        print("    counterfactual delta: " + ", ".join(f"d={d}:F'={Fn}(+{e}/{smd},k{mk})" for d, Fn, e, smd, mk in rows))
        print(f"    increment law fails at delta={bad_inc or 'none'};  (D) fails at delta={bad_D or 'none'};  real delta={real} or {q2 - real}")


if __name__ == "__main__":
    main()
