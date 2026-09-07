"""r69 / new gears lengthen, never precede -- part 1: full periods (and long prefixes) m5 .. m43.

For each step M = {5..q} -> M + q' with q = 5..41: the Pareto staircase of both machines over the
full period of M + q' when it is at most XMAX columns (m7..m31; m31's period 33,426,748,355 is
streamed in chunks), else over the prefix [0, XMAX) (m37, m41, m43).  From the staircases:
  * R_min^M(L), R_min^{M+q'}(L) for every L in [d_0(M), F(M) - 1];
  * Form B exceptions (R_min^{M+q'}(L) < R_min^M(L)), each classified: absorbed (new start 1),
    straddle (new run contains W(q)), deep (new start > W(q)), other (inside [2, W(q)] without
    containing W(q); E6 predicts 0); Form A likewise with threshold d_0(M + q');
  * for every deep exception, the merge order J of the new run (1 + the number of openings of M
    inside it), by a direct small sieve;
  * c(M) = min_{L >= d_0(M)} R_min^M(L)/L on the scanned range and its minimiser;
  * E6 by direct sieve on [1, W(q)].

Self-contained, numpy only.  Peak memory about 300 MB (chunks of 2^25 columns).
Run: uv run python research/anchor235/r69/lnp_period.py [XMAX_log2]
"""
import json
import os
import sys
import time

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
XMAX = 1 << (int(sys.argv[1]) if len(sys.argv) > 1 else 31)
CHUNK = 1 << 25
PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]


def period(gears):
    P = 1
    for g in gears:
        P *= g
    return P


def scan(gears, N):
    """Pareto staircase [(x, L)] of the blocked runs of the machine over columns [0, N),
    plus d_0, d_1, number of runs, longest run.  Streamed in chunks."""
    us = [(g, pow(6, -1, g)) for g in gears]
    pareto = []
    best = 0
    cur_start = -1
    nruns = 0
    openings_seen = []
    for a in range(0, N, CHUNK):
        b = min(a + CHUNK, N)
        n = b - a
        blk = np.zeros(n, dtype=bool)
        for g, u in us:
            for r in (u % g, (g - u) % g):
                blk[(r - a) % g::g] = True
        pad = np.empty(n + 2, dtype=np.int8)
        pad[0] = 1 if cur_start >= 0 else 0
        pad[-1] = 0
        pad[1:-1] = blk
        d = np.diff(pad)
        st = np.flatnonzero(d == 1) + a
        en = np.flatnonzero(d == -1) + a
        if cur_start >= 0:
            st = np.concatenate(([cur_start], st))
        if en.size == st.size - 1:
            cur_start = int(st[-1])
            st = st[:-1]
        else:
            cur_start = -1
        assert en.size == st.size
        lens = en - st
        nruns += st.size
        if a == 0:
            ops = (np.flatnonzero(~blk[1:])[:2] + 1).tolist()
            openings_seen = [int(o) for o in ops]
        if st.size:
            rmv = np.maximum.accumulate(lens)
            keep = np.empty(lens.size, dtype=bool)
            keep[0] = lens[0] > best
            keep[1:] = rmv[1:] > rmv[:-1]
            keep &= lens > best
            for x, L in zip(st[keep].tolist(), lens[keep].tolist()):
                if L > best:
                    pareto.append((int(x), int(L)))
                    best = L
    return dict(pareto=pareto, d0=openings_seen[0], d1=openings_seen[1] if len(openings_seen) > 1 else None,
                nruns=nruns, longest=best, N=N)


def rmin_from_pareto(pareto):
    xs = [p[0] for p in pareto]
    ls = [p[1] for p in pareto]

    def f(L):
        for x, l in zip(xs, ls):
            if l >= L:
                return x
        return None
    return f


def direct_blocked(gears, top):
    b = np.zeros(top + 1, dtype=bool)
    for g in gears:
        u = pow(6, -1, g)
        b[u::g] = True
        b[(g - u) % g::g] = True
    return b


def isprime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def main():
    cache_path = os.path.join(OUT, "lnp_period_scans.json")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    lines = []
    Wr = lines.append
    scans = {}
    for i, q in enumerate(PRIMES[:-1]):
        gears = PRIMES[:i + 1]
        P = period(gears)
        N = min(P, XMAX)
        key = "%d_%d" % (q, N)
        if key in cache:
            scans[q] = cache[key]
            continue
        t = time.time()
        scans[q] = scan(gears, N)
        scans[q]["P"] = P
        print("scanned m%d over %d columns (period %d) in %.0f s: longest %d, d0 %d, pareto %d points" % (
            q, N, P, time.time() - t, scans[q]["longest"], scans[q]["d0"], len(scans[q]["pareto"])), flush=True)
        cache[key] = scans[q]
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    summary = []
    for i, q in enumerate(PRIMES[:-2]):
        qp = PRIMES[i + 1]
        qpp = PRIMES[i + 2]
        gM = PRIMES[:i + 1]
        SM, SN = scans[q], scans[qp]
        Wq = (qp * qp - 1) // 6
        full = SN["N"] == period(gM + [qp])
        RM, RN = rmin_from_pareto(SM["pareto"]), rmin_from_pareto(SN["pareto"])
        d0M, d0N = SM["d0"], SN["d0"]
        FM1 = SM["longest"]
        # E6 direct
        bM = direct_blocked(gM, Wq)
        bN = direct_blocked(gM + [qp], Wq)
        new = sorted(int(x) for x in np.flatnonzero(bN & ~bM) if x >= 1)
        twin_next = (qp % 6 == 5) and isprime(qp + 2) and (qpp == qp + 2)
        rider = isprime(qp * qp - 2)
        pred = sorted(set(([d0M] if twin_next else []) + ([Wq] if rider else [])))
        # c(M)
        cM, cmin = None, None
        for L in range(d0M, FM1 + 1):
            x = RM(L)
            if x is None:
                continue
            if cM is None or x / L < cM:
                cM, cmin = x / L, (x, L)
        excB, excA = [], []
        for L in range(d0M, FM1 + 1):
            a, b = RM(L), RN(L)
            if a is None or b is None:
                continue
            if b < a:
                # the run of N at b
                Lb = next(l for (x, l) in SN["pareto"] if x == b)
                if b == 1:
                    kind = "absorbed"
                elif b <= Wq < b + Lb - 1 + 1:
                    kind = "straddle"
                elif b > Wq:
                    kind = "deep"
                else:
                    kind = "other"
                J = None
                if kind in ("deep", "straddle"):
                    lo, hi = b, b + Lb - 1
                    # openings of M inside [lo, hi] by a direct sieve of M on that interval
                    blkM = np.zeros(hi - lo + 1, dtype=bool)
                    for g in gM:
                        u = pow(6, -1, g)
                        for r in (u % g, (g - u) % g):
                            blkM[(r - lo) % g::g] = True
                    J = 1 + int((~blkM).sum())
                excB.append((L, a, b, Lb, kind, J))
                if L >= d0N:
                    excA.append((L, a, b, Lb, kind, J))
        kindsB = {}
        for e in excB:
            kindsB[e[4]] = kindsB.get(e[4], 0) + 1
        kindsA = {}
        for e in excA:
            kindsA[e[4]] = kindsA.get(e[4], 0) + 1
        Wr("")
        Wr("=" * 100)
        Wr("step m%d -> m%d   P(M)=%d  P(M+q')=%d  scanned N(M)=%d N(M+q')=%d %s  W(q)=%d" % (
            q, qp, SM["P"], SN["P"], SM["N"], SN["N"], "FULL PERIOD" if full else "PREFIX", Wq))
        Wr("  longest run: %d -> %d ; d_0: %d -> %d ; d_1(M) = %s ; twin(q',q'')=%s rider(q'^2-2 prime)=%s" % (
            FM1, SN["longest"], d0M, d0N, SM["d1"], twin_next, rider))
        Wr("  E6 direct on [1,W(q)]: new columns %s predicted %s  %s" % (new, pred, "OK" if new == pred else "MISMATCH"))
        Wr("  c(M) over L>=d_0(M): %s at %s" % (("%.4f" % cM) if cM else None, cmin))
        Wr("  Pareto M   : %s" % SM["pareto"][:14])
        Wr("  Pareto M+q': %s" % SN["pareto"][:14])
        Wr("  Form B cells L in [%d, %d]: %d exceptions %s" % (d0M, FM1, len(excB), kindsB))
        Wr("  Form A cells L in [%d, %d]: %d exceptions %s" % (d0N, FM1, len(excA), kindsA))
        for e in excB:
            Wr("     L=%3d  R_min^M=%10d  R_min^{M+q'}=%10d (run length %d)  %s  J=%s" % e)
        summary.append(dict(q=q, qp=qp, full=full, Wq=Wq, FM1=FM1, FN1=SN["longest"], d0M=d0M, d0N=d0N, cM=cM, cmin=cmin,
                            e6_ok=(new == pred), excB=excB, excA=excA, paretoM=SM["pareto"], paretoN=SN["pareto"]))
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "lnp_period.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    with open(os.path.join(OUT, "lnp_period_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f)
    print(txt)


if __name__ == "__main__":
    main()
