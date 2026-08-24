"""Harvester round 22: fast exact family scanner (three-level version of
research/family_scan.py - same prefilter, same output, no full-period sieve).

The prefilter is unchanged and exact: a gap G >= Gmin needs L = Gmin-1 consecutive
killed positions; every survivor of the gears below the held-out top gear qt inside
such a window must be killed by qt, so the survivors' OFFSETS occupy at most two
residue classes mod qt, and two distinct classes r1 != r2 force delta = +-(r1-r2)
mod qt (the window's absolute position mod qt is free by CRT).

Speed comes from never sieving the full pre-period Qm = Qb*qmid.  The base gears are
sieved over Qb once per base class; the middle gear's copies are produced by tiling the
base survivor INDEX list; and the windows worth examining are found by one subtraction
on that list:

    a window of length L with at most CAP survivors starts just after idx[i]
    iff  idx[i + CAP + 1] - idx[i] >= L + 1        (cyclically),

after which the exact window starts are w in [idx[i]+1, min(idx[i+1], idx[i+CAP+1]-L)].

Validated in __main__ against family_scan.scan at y = 17 and against the round-22
exhaustive 19-winner set (family_scan.scan is itself validated against brute force at
y = 13 and y = 17).
"""
import os
import sys
import numpy as np
from math import prod
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from family_scan import survivors, max_gap


def fast_scan(qs_base, qmid, qt, Gmin, lo=None, hi=None, prog=None):
    Qb = prod(qs_base)
    Qm = Qb * qmid
    Q = Qm * qt
    L = Gmin - 1
    CAP = 2 * ((L + qt - 1) // qt)
    lo = 0 if lo is None else lo
    hi = Qb if hi is None else hi
    invb = pow(Qb, -1, qmid)
    invm = pow(Qm, -1, qt)
    tile = (np.arange(qmid, dtype=np.int64) * Qb)[:, None]
    out = set()
    for db in range(lo, hi):
        idxb = np.flatnonzero(survivors(qs_base, db, Qb)).astype(np.int64)
        full = (tile + idxb[None, :]).ravel()
        remM = full % qmid
        for rm in range(qmid):
            keep = (remM != 0) & (remM != (-rm) % qmid)
            idx = full[keep]
            n = idx.size
            if n <= CAP + 2:
                continue
            X = np.concatenate([idx, idx[:CAP + 2] + Qm])
            good = np.flatnonzero(X[CAP + 1:CAP + 1 + n] - X[:n] >= L + 1)
            if good.size == 0:
                continue
            wlo = X[good] + 1
            whi = np.minimum(X[good + 1], X[good + CAP + 1] - L)
            cntw = whi - wlo + 1
            ok = cntw > 0
            if not ok.any():
                continue
            wlo, cntw = wlo[ok], cntw[ok]
            starts = np.repeat(wlo, cntw)
            offs = np.arange(cntw.sum()) - np.repeat(
                np.concatenate([[0], np.cumsum(cntw)[:-1]]), cntw)
            cw = starts + offs
            a = np.searchsorted(X, cw)
            b = np.searchsorted(X, cw + L)
            m = int((b - a).max())
            assert m <= CAP, (m, CAP)
            res = np.full((cw.size, max(CAP, 1)), qt, np.int64)
            for k in range(m):
                sel = a + k < b
                res[sel, k] = X[a[sel] + k] % qt
            res.sort(axis=1)
            real = res < qt
            ndist = 1 + (np.diff(res, axis=1) != 0).sum(axis=1)
            nreal = ndist - (~real).any(axis=1).astype(int)
            hits = np.flatnonzero(nreal <= 2)
            if hits.size == 0:
                continue
            dm = (db + Qb * (((rm - db) * invb) % qmid)) % Qm
            for i in hits:
                r = np.unique(res[i][real[i]])
                dqs = range(qt) if r.size <= 1 else \
                    (int((r[1] - r[0]) % qt), int((r[0] - r[1]) % qt))
                for dq in dqs:
                    out.add((dm + Qm * (((dq - dm) * invm) % qt)) % Q)
        if prog and db % prog == 0:
            print(f"    db={db}/{Qb} cands={len(out)}", flush=True)
    return sorted(out)


def verify(qs, cands, Q, Gmin):
    res = [(int(d), max_gap(qs, int(d), Q)) for d in cands]
    return [r for r in res if r[1] >= Gmin]


if __name__ == "__main__":
    import time
    from family_scan import scan
    print("VALIDATION y=17: fast three-level vs family_scan.scan", flush=True)
    a = fast_scan([5, 7, 11], 13, 17, 32)
    b = [d for d, _ in scan([5, 7, 11, 13], 17, 32)]
    va = [d for d, g in verify([5, 7, 11, 13, 17], a, 85085, 32) if g == 32]
    assert set(va) == set(b) and len(b) == 64, (len(va), len(b))
    print(f"    {len(a)} prefilter candidates -> {len(va)} winners == scan's {len(b)}",
          flush=True)
    print("VALIDATION y=19: fast three-level vs the round-22 exhaustive 19-winner set",
          flush=True)
    t = time.time()
    a = fast_scan([5, 7, 11, 13], 17, 19, 43)
    dt = time.time() - t
    Q19 = 5 * 7 * 11 * 13 * 17 * 19
    va = [d for d, g in verify([5, 7, 11, 13, 17, 19], a, Q19, 43) if g == 43]
    ref = set(np.load("research/data/family_w19_delta.npy").tolist())
    assert set(va) == ref, (len(va), len(ref))
    print(f"    {len(a)} prefilter candidates -> {len(va)} winners == stored {len(ref)}"
          f"   [{dt:.1f} s for 85085 classes]", flush=True)
    print("family_scan_fast: ALL ASSERTIONS GREEN")
