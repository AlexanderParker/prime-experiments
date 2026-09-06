"""rr_profile.py -- the neighbour profile n1(v) (largest SINGLE neighbour of a gap of size v)
and the sum profile Sig(v) = v + n1(v), on FULL periods, for the machines {5..11} .. {5..31}.

Branch: research/proof/record_2run.md (node 4.i.a.i.a.1.a.i, the record gap as a 2-run).

n1(v) is the same object as the adjacent-pair row top r(v) of short_letter_row.md; only the
neighbour SUM profile N(v) had been computed before (branch 2g.i, r45/deep_profile.py).  The
chunked-sieve skeleton and the "gaps owned by their left endpoint" convention are taken from
that script so the two agree cell for cell.

Computed per machine, exactly, over one full period:
  * m(v)          -- gap multiplicity
  * n1(v)         -- max over gaps of size v of max(left, right)
  * N(v)          -- max over gaps of size v of (left + right)      [instrument, vs r45]
  * a witness (column, L, R) for n1(v)
  * every occurrence of the record size F, with its two neighbours (capped at RECCAP)

Usage:  uv run python rr_profile.py <top gear> <outfile> [lo gear] [nproc]
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31]
MARGIN = 4096
VMAX = 128          # cap on a gap size
KMAX = 128          # cap on max(L, R)
SMAX = 256          # cap on L + R
CHUNK = 3 * 10 ** 7
RECCAP = 200

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def worker(args):
    gears, lo, hi, F_hint = args
    u = [pow(6, -1, g) for g in gears]
    spec = np.zeros(VMAX, dtype=np.int64)
    jmax = np.zeros(VMAX * KMAX, dtype=np.int64)      # (v, max(L,R))
    jsum = np.zeros(VMAX * SMAX, dtype=np.int64)      # (v, L+R)
    wit = {}                                          # v -> (n1, x, L, R)
    recs = []                                         # occurrences of size F_hint
    c0 = lo
    while c0 < hi:
        c1 = min(c0 + CHUNK, hi)
        s, e = c0 - MARGIN, c1 + MARGIN
        n = e - s
        blocked = np.zeros(n, dtype=bool)
        for g, ug in zip(gears, u):
            for t in (ug, g - ug):
                blocked[(t - s) % g::g] = True
        idx = np.flatnonzero(~blocked)
        del blocked
        opens = idx.astype(np.int64) + s
        del idx
        gaps = np.diff(opens).astype(np.int64)
        own = np.flatnonzero((opens[:-1] >= c0) & (opens[:-1] < c1))
        if own.size:
            spec += np.bincount(gaps[own], minlength=VMAX)[:VMAX]
            oi = own[(own >= 1) & (own + 1 < gaps.size)]
            v = gaps[oi]
            L = gaps[oi - 1]
            R = gaps[oi + 1]
            mx = np.maximum(L, R)
            jmax += np.bincount(v * KMAX + mx, minlength=VMAX * KMAX)[:VMAX * KMAX]
            jsum += np.bincount(v * SMAX + (L + R), minlength=VMAX * SMAX)[:VMAX * SMAX]
            # witness per size: the occurrence attaining the largest single neighbour
            order = np.lexsort((-mx, v))
            vs = v[order]
            first = np.flatnonzero(np.r_[True, vs[1:] != vs[:-1]])
            for f in first:
                k = order[f]
                vv = int(v[k])
                if wit.get(vv, (-1,))[0] < int(mx[k]):
                    wit[vv] = (int(mx[k]), int(opens[oi[k]]), int(L[k]), int(R[k]))
            if F_hint:
                hit = np.flatnonzero(v == F_hint)
                for k in hit[:RECCAP]:
                    recs.append((int(opens[oi[k]]), int(L[k]), int(R[k])))
        c0 = c1
    return spec, jmax, jsum, wit, recs


def run(gears, nproc, F_hint):
    t0 = time.time()
    P = 1
    for g in gears:
        P *= g
    bounds = [P * i // nproc for i in range(nproc + 1)]
    jobs = [(gears, bounds[i], bounds[i + 1], F_hint) for i in range(nproc)]
    if nproc == 1:
        parts = [worker(jobs[0])]
    else:
        with Pool(nproc) as pool:
            parts = pool.map(worker, jobs)
    spec = sum(p[0] for p in parts)
    jmax = sum(p[1] for p in parts).reshape(VMAX, KMAX)
    jsum = sum(p[2] for p in parts).reshape(VMAX, SMAX)
    wit = {}
    for p in parts:
        for vv, val in p[3].items():
            if wit.get(vv, (-1,))[0] < val[0]:
                wit[vv] = val
    recs = [r for p in parts for r in p[4]]
    return P, spec, jmax, jsum, wit, recs, time.time() - t0


def report(gears, out, nproc):
    F_hint = None
    # a cheap first pass on a short window is not exact; instead take F from the recorded ladder
    ladder = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 7: 5, 5: 2}
    F_hint = ladder.get(gears[-1])
    P, spec, jmax, jsum, wit, recs, dt = run(gears, nproc, F_hint)
    F = int(np.max(np.flatnonzero(spec)))
    assert F == F_hint, (F, F_hint)
    tot = int(spec.sum())
    n1 = {}
    N = {}
    for v in range(1, VMAX):
        nz = np.flatnonzero(jmax[v])
        if nz.size:
            n1[v] = int(nz.max())
        nz = np.flatnonzero(jsum[v])
        if nz.size:
            N[v] = int(nz.max())
    sizes = sorted(n1)
    F2 = max(v + n1[v] for v in sizes)
    argF2 = [v for v in sizes if v + n1[v] == F2]
    top = [v for v in sizes if v >= 0.8 * F]
    d = {
        "gears": gears, "P": P, "openings": tot, "F": F, "F2": F2,
        "spec": {v: int(spec[v]) for v in sizes},
        "n1": n1, "N": N, "Sig": {v: v + n1[v] for v in sizes},
        "argF2": argF2, "top_band": top,
        "D_top": F2 - max(v + n1[v] for v in top),
        "wit": {v: wit[v] for v in sizes},
        "record_occurrences": recs,
        "seconds": dt,
    }
    w = out.write
    w(f"\n=== machine {{5..{gears[-1]}}} P={P} openings={tot} F={F} F_2={F2}  [{dt:.1f}s]\n")
    w(f"  realised sizes: {len(sizes)} of 1..{F}; missing below F: "
      f"{[v for v in range(1, F) if v not in n1]}\n")
    w(f"  F_2 maximisers v* (pairs (v*, n1(v*))): {[(v, n1[v]) for v in argF2]}  "
      f"v*/F = {[round(v / F, 3) for v in argF2]}\n")
    w(f"  top band (v >= 0.8F = {0.8 * F:.1f}): {top}\n")
    w(f"  top band n1: {[(v, n1[v]) for v in top]}\n")
    w(f"  top band Sig = v+n1: {[(v, v + n1[v]) for v in top]}   max={max(v + n1[v] for v in top)}"
      f"  D_top = {d['D_top']}\n")
    w(f"  record 2-run: F={F}, n1(F)={n1[F]}, F+n1(F)={F + n1[F]}, "
      f"D_rec = F_2 - F - n1(F) = {F2 - F - n1[F]}\n")
    w(f"  record occurrences (x, L, R), {len(recs)} found (cap {RECCAP} per worker): {recs}\n")
    tops = sorted(sizes, reverse=True)[:8]
    w(f"  top of spectrum (size, count, n1, N): "
      f"{[(v, int(spec[v]), n1[v], N[v]) for v in tops]}\n")
    w(f"  isolation iso_k = s_k - s_(k+1): "
      f"{[tops[i] - tops[i + 1] for i in range(len(tops) - 1)]}\n")
    w("  profile v: n1(v) N(v) m(v) :\n    "
      + " ".join(f"{v}:{n1[v]}/{N[v]}({int(spec[v])})" for v in sizes) + "\n")
    w("  Sig(v) = v + n1(v): " + " ".join(f"{v}:{v + n1[v]}" for v in sizes) + "\n")
    w("  witnesses of n1(v)  v:(n1,x,L,R): "
      + " ".join(f"{v}:{wit[v]}" for v in sizes) + "\n")
    out.flush()
    return d


if __name__ == "__main__":
    top = int(sys.argv[1]) if len(sys.argv) > 1 else 23
    dest = sys.argv[2] if len(sys.argv) > 2 else None
    lo = int(sys.argv[3]) if len(sys.argv) > 3 else 11
    nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    out = open(dest, "w") if dest else sys.stdout
    alld = {}
    for i in range(2, len(PRIMES)):
        if PRIMES[i] > top:
            break
        if PRIMES[i] < lo:
            continue
        alld[PRIMES[i]] = report(PRIMES[:i + 1], out, nproc)
    if dest:
        out.close()
        with open(dest.replace(".txt", ".json"), "w") as f:
            json.dump(alld, f, default=int)
