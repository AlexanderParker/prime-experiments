"""Harvester round 22: EXHAUSTIVE family-maximiser scan via the delta reduction
plus a held-out-top-gear prefilter.

Problem.  For the paired-Jacobsthal family (halved coordinates), with 3 not dividing
the difference e, F_e(y) = 3 * G(delta), delta = e*3^{-1} mod Q, Q = prod_{5<=q<=y} q
(proved and verified in research/delta_frame.py), where G(delta) is the maximal cyclic
gap of S_delta = {k in Z_Q : k != 0, -delta mod q for every gear q}.

Goal: ALL delta with G(delta) >= Gmin, exhaustively, without a Q-length sieve per delta.

The prefilter (exact, not heuristic).  A gap G >= Gmin means a run of L = Gmin-1
consecutive KILLED positions.  Hold out the top gear qt and let Q' = Q/qt.  A window
of L consecutive k's reduces to a contiguous window of Z_{Q'}; every survivor of the
gears < qt inside it must be killed by qt, i.e. must lie in {0, -delta} mod qt.  Since
the window's absolute position mod qt is free (CRT), the condition is exactly

    |{ j mod qt : j an offset of a surviving position in the window }| <= 2,

and when there are exactly two such residues r1 != r2 they force
delta = +-(r1 - r2) mod qt; with <= 1 residue delta mod qt is unconstrained.

So: sweep delta mod Q' (Q' classes), sweep the Q' cyclic windows of each, keep the
windows whose survivor-offsets occupy <= 2 residues mod qt, read off delta mod qt,
CRT, and verify each surviving candidate by a direct sieve.  Nothing is discarded that
could have had a run of length L, so the output is complete.
"""
import numpy as np
from math import prod


def survivors(qs, delta, Q):
    a = np.ones(Q, bool)
    for q in qs:
        a[0::q] = False
        a[(-delta) % q::q] = False
    return a


def max_gap(qs, delta, Q, want_runs=False):
    idx = np.flatnonzero(survivors(qs, delta, Q))
    if idx.size < 2:
        return (Q, []) if want_runs else Q
    d = np.diff(np.append(idx, idx[0] + Q))
    g = int(d.max())
    if want_runs:
        return g, [int(idx[i]) for i in np.flatnonzero(d == g)]
    return g


def scan(qs_pre, qt, Gmin, verbose=False, prog=None):
    """Return sorted list of all delta in Z_Q (Q = prod(qs_pre)*qt) with
    max_gap >= Gmin, together with the realised max gap."""
    Qp = prod(qs_pre)
    Q = Qp * qt
    L = Gmin - 1                      # consecutive killed positions needed
    capacity = 2 * ((L + qt - 1) // qt)
    MAXOFF = capacity
    cands = set()
    inv_qp = pow(Qp, -1, qt)
    for dp in range(Qp):
        A = survivors(qs_pre, dp, Qp)
        ext = np.concatenate([A, A[:L]])
        c = np.concatenate([[0], np.cumsum(ext)])
        cnt = c[L:L + Qp] - c[:Qp]                    # survivors in window [w, w+L)
        cw = np.flatnonzero(cnt <= capacity)
        if cw.size == 0:
            continue
        idx = np.flatnonzero(ext)
        lo = np.searchsorted(idx, cw)
        hi = np.searchsorted(idx, cw + L)
        m = int((hi - lo).max())
        res = np.full((cw.size, MAXOFF), qt, np.int64)   # qt = sentinel
        for j in range(m):
            ok = lo + j < hi
            res[ok, j] = (idx[lo[ok] + j] - cw[ok]) % qt
        res.sort(axis=1)
        real = res < qt
        ndist_all = 1 + (np.diff(res, axis=1) != 0).sum(axis=1)   # sentinel included
        nreal = ndist_all - (~real).any(axis=1).astype(int)
        for i in np.flatnonzero(nreal <= 2):
            r = np.unique(res[i][real[i]])
            if r.size <= 1:
                d19s = range(qt)
            else:
                diff = int((r[1] - r[0]) % qt)
                d19s = (diff, (-diff) % qt)
            for dq in d19s:
                # CRT: delta = dp mod Qp, dq mod qt
                cands.add((dp + Qp * (((dq - dp) * inv_qp) % qt)) % Q)
        if prog and dp % prog == 0:
            print(f"    dp={dp}/{Qp}  cands={len(cands)}", flush=True)
    qs = qs_pre + [qt]
    out = []
    for d in sorted(cands):
        g = max_gap(qs, d, Q)
        if g >= Gmin:
            out.append((d, g))
    if verbose:
        print(f"    prefilter kept {len(cands)} of {Q} deltas "
              f"({100.0*len(cands)/Q:.4f}%); {len(out)} reach G >= {Gmin}")
    return out


def brute(qs, Q):
    g = np.zeros(Q, np.int32)
    for d in range(Q):
        g[d] = max_gap(qs, d, Q)
    return g


if __name__ == "__main__":
    import sys
    # validation: y = 13 and y = 17 against brute force in delta space
    print("VALIDATION y=13 (gears 5,7,11 prefilter, top gear 13):")
    g13 = brute([5, 7, 11, 13], 5005)
    best13 = int(g13.max())
    w13 = set(int(x) for x in np.flatnonzero(g13 == best13))
    s13 = scan([5, 7, 11], 13, best13, verbose=True)
    assert set(d for d, _ in s13) == w13, (len(s13), len(w13))
    print(f"    brute max G = {best13} (F = {3*best13}), {len(w13)} winners; "
          f"scan reproduces the set EXACTLY")

    print("VALIDATION y=17 (gears 5,7,11,13 prefilter, top gear 17):")
    g17 = brute([5, 7, 11, 13, 17], 85085)
    best17 = int(g17.max())
    w17 = set(int(x) for x in np.flatnonzero(g17 == best17))
    s17 = scan([5, 7, 11, 13], 17, best17, verbose=True)
    assert set(d for d, _ in s17) == w17, (len(s17), len(w17))
    print(f"    brute max G = {best17} (F = {3*best17}), {len(w17)} winners; "
          f"scan reproduces the set EXACTLY")
    np.save("research/data/family_G17_delta.npy", g17)
    print("family_scan validation: ALL ASSERTIONS GREEN")
